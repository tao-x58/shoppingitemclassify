# 商品分类-测试题

请根据下述数据集，实现商品标题的分类模型

## 说明：

* 大数据场景，需要有千万量级的商品进行预测。

## 数据集：

商品分类标注数据集: https://modelscope.cn/datasets/winwin_inc/product-classification-hiring-demo

* 商品标题
* 商品类目：`即饮奶茶, 即饮茶, 硬糖, 牙膏, 非冷藏即饮果汁, 冷藏即饮果汁, 白酒, 方便面, 软糖, 口香糖`

## 代码描述：

1. 请自行安装环境管理工具`uv`: https://github.com/astral-sh/uv
2. 执行代码测试：`uv run pytest` 将下载数据集预览。
3. 如需Jupyter Notebook：`uv run notebook`
4. 请继续实现你的代码。

## 结果要求：

* 编程语言：Python优先、Typescript/Go均可。
* 实现框架：不限（机器学习、深度学习、Agent方案）。
* 思路：给出你的分析因素、框架选择原因、流程优化分析与深度优化建议。
* 输出
    * 指标：根据测试集自行计算指标。
    * 如果使用Python语言请通过Py、Notebook+Py完成，并保留过程记录。
    * 其他语言请提供一个HTTP API的在线Service，或者可以被方便调用的Lib。
    * 【加分项】:
        * 对实现方案的思路描述、完整迭代记录。
        * 大模型微调、混合方案。
        * 实现缺陷、可优化方向。

## 提交方式

* 克隆仓库，打包完整项目Zip，发给HR。
* 或者Fork项目（不要提PR），提交你的仓库链接，发给HR。