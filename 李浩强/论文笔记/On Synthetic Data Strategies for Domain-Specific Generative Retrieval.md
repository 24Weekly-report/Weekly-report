# On Synthetic Data Strategies for Domain-Specific Generative Retrieval

这篇论文提出了一种两阶段训练框架，用于构建领域特定的生成式检索模型。核心思想是通过合成数据策略解决领域数据标注成本高的问题，并优化模型的记忆、泛化和相关性排序能力。

在该论文中，作者指示大语言模型生成一个描述文档内容的关键词列表，并将该关键词列表用作文档的语义标识符。

#### 第一阶段：监督微调

这一阶段目的是训练模型将输入（查询或文本片段）映射到对应的文档标识符（ID）。

1. Context2ID

输入：原始文档（如新闻文章、百科页面）。

输出：用LLM提取关键词或摘要作为ID。

2. Query2ID

输入：文档内容（与Context2ID相同，但用途不同）+查询。

输出：用LLM提取关键词或摘要作为ID

***\*多粒度查询生成\****

对于每个文本块，让大语言模型生成$m_c$个块级查询。随后，将该文本块拆分为单个句子，并让大语言模型为每个句子生成$m_s$个句子级查询。

***\*基于约束的查询生成\****

让大语言模型为每个文档生成$m_i$个包含这些约束的查询，从而使生成式检索模型能够处理更专业或特定领域的查询。

如政治数据集**AllSides**使用属性political polarity，即为表明文本段落的政治倾向标签（如"左翼/右翼/中立"）。

$$ \large \mathcal{L}_{\text{sft}}(q, d) = -\log P\left(d', q; \theta\right)  = -\sum_{i} \log P\left(q_i \mid q_{<i}; \theta\right)  - \sum_{i} \log P\left(d'_i \mid d'_{<i}, q; \theta\right) $$



#### 第二阶段：监督微调

首先生成困难查询。通过提示要求大语言模型生成尽可能困难的查询。同时，要求大语言模型不仅提供合成查询，还要提供相应的答案。

生成合成查询后，下一步是为正则化偏好优化（RPO）选择文档候选对。对于每个训练样本，需要一个正向候选和一个负向候选。

正向候选为生成查询的文档。会使用生成式检索模型对用于偏好学习的合成查询执行检索。策略主要是从检索结果中选择排名高于正向候选的前k个负向候选。

如果正向候选排名第一，将不会把该查询用于偏好学习。如果正向候选的排名高于k，那么负向候选的数量会因排名不同而有所差异；如果排名低于k，则会有k个不同的负向候选。当存在多个负向候选时，将每个负向候选与正向候选配对，形成偏好学习的候选对样本。

最后使用RPO微调模型。该方法接收输入查询q、正向候选d_p和负向候选d_n作为输入，其损失函数倾向于正向候选，同时抑制负向候选。

$$ \large \mathcal{L}_{\text{rpo}}(q, d_p, d_n) = -\log \delta \left(  \beta \log \frac{P\left(d'_p \mid q; \theta\right)}{P\left(d'_p \mid q; \theta_{\text{ref}}\right)} \right.  \left. -\beta \log \frac{P\left(d'_n \mid q; \theta\right)}{P\left(d'_n \mid q; \theta_{\text{ref}}\right)} \right)  -\alpha \frac{\log P\left(d'_p \mid q; \theta\right)}{\vert d'_p \vert} $$
