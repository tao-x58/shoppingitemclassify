param(
    [switch]$UnitOnly,
    [string]$OutputDir = 'artifacts/acceptance'
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    # Keep uv's writable cache inside this project. Sync uses the checked-in lock.
    & uv --cache-dir .uv-cache sync --locked
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
    & uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest-tmp
    if ($LASTEXITCODE -ne 0) { throw 'Offline functional tests failed' }
    if ($UnitOnly) { return }

    $taskTrainHash = (Get-FileHash -LiteralPath data/train.jsonl -Algorithm SHA256).Hash
    $taskTestHash = (Get-FileHash -LiteralPath data/test.jsonl -Algorithm SHA256).Hash

    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    $taskModelDir = Join-Path $OutputDir 'model'
    & uv --cache-dir .uv-cache run --no-sync start audit --output (Join-Path $OutputDir 'data_audit.json')
    if ($LASTEXITCODE -ne 0) { throw 'Data audit failed' }
    & uv --cache-dir .uv-cache run --no-sync start train --output-dir $taskModelDir
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    & uv --cache-dir .uv-cache run --no-sync start evaluate --model-dir $taskModelDir --output-dir (Join-Path $OutputDir 'evaluation')
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
    & uv --cache-dir .uv-cache run --no-sync start predict --model-dir $taskModelDir --input data/test.jsonl --output (Join-Path $OutputDir 'predictions-257.jsonl') --batch-size 257
    if ($LASTEXITCODE -ne 0) { throw 'Batch prediction failed' }
    & uv --cache-dir .uv-cache run --no-sync start predict --model-dir $taskModelDir --input data/test.jsonl --output (Join-Path $OutputDir 'predictions-4096.jsonl') --batch-size 4096
    if ($LASTEXITCODE -ne 0) { throw 'Batch comparison failed' }
    $taskHashA = (Get-FileHash -LiteralPath (Join-Path $OutputDir 'predictions-257.jsonl') -Algorithm SHA256).Hash
    $taskHashB = (Get-FileHash -LiteralPath (Join-Path $OutputDir 'predictions-4096.jsonl') -Algorithm SHA256).Hash
    if ($taskHashA -ne $taskHashB) { throw 'Predictions differ across batch sizes' }
    $taskExpected = @(Get-Content -LiteralPath data/test.jsonl -Encoding UTF8 | Where-Object { $_.Trim() }).Count
    $taskActual = @(Get-Content -LiteralPath (Join-Path $OutputDir 'predictions-257.jsonl') -Encoding UTF8 | Where-Object { $_.Trim() }).Count
    if ($taskExpected -ne $taskActual) { throw 'Prediction row count differs from input' }
    foreach ($taskBatchSize in @(256, 1024, 4096)) {
        & uv --cache-dir .uv-cache run --no-sync start benchmark --model-dir $taskModelDir --input data/train.jsonl --batch-size $taskBatchSize --output (Join-Path $OutputDir "benchmark-$taskBatchSize.json")
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: batch size $taskBatchSize" }
    }
    if ((Get-FileHash -LiteralPath data/train.jsonl -Algorithm SHA256).Hash -ne $taskTrainHash) { throw 'Original training data changed' }
    if ((Get-FileHash -LiteralPath data/test.jsonl -Algorithm SHA256).Hash -ne $taskTestHash) { throw 'Original test data changed' }
    Write-Host "Results generated at $OutputDir. Review validation/test Macro-F1, per-class recall and benchmarks. No quality or performance acceptance threshold has been assumed."
}
finally {
    Pop-Location
}
