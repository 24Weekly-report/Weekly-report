# DREQ: Document Re-Ranking Using Entity-based Query Understanding（SIGIR 25）

论文提出的方法名为 DREQ（Document Re-ranking using Entity-based Query Understanding），是一种基于实体的神经文档重排序模型。其核心思想是通过结合查询相关的实体信息和文档的文本信息，生成一种混合的文档表示，从而更精准地评估文档与查询的相关性。



### 实体排序

从所有候选文档d∈D中汇集所有实体，从而为查询Q获取实体候选集E。基于“相关文档包含相关实体”的假设，将文档的相关性标签迁移至文档中的实体。利用该实体真值，训练独立的实体排序模型以对实体e∈E进行排序。



借助知识库（DBpedia）中实体e的描述$T_e$。查询标记$T_q$∈Q与实体描述标记 $T_e$∈$t_e$组成的序列，两者通过特殊标记[SEP]分隔，并在序列开头添加特殊标记[CLS]。将BERT最后隐藏层中[CLS]标记的k维嵌入作为实体e∈E的嵌入向量e

***\*评分函数S(e, Q)的公式为：S(e, Q) = W₁·e + b，其中b为标量偏置项。\****



### 文档表示

候选文档 d∈D 的***\*查询特定的以实体为中心\****的表示 $V_{e_d}^Q $∈$R^m$

首先通过 Wikipedia2Vec工具预训练得到的嵌入来表示文档中的每个实体，因为它能够捕捉实体之间的关系和深层语境。文档表示$V_{e_d}^Q$是文档中各实体嵌入的加权和：

$\Large V_{e_d}^Q = \sum_{e \in \mathcal{E}_d} w_e \cdot e$

其中，每个实体的权重$w_e$是特定于查询的，由该实体的排序分数决定。

$$\Large w_e = \frac{S(e, Q)}{\sum_{e' \in \mathcal{E}_d} S(e', Q)}$$



还使用BERT获取候选文档d∈D的以文本为中心的表示$V_{t_d}^Q$

首先将文档按滑动窗口分割为段落：采用M个句子的窗口大小，以S个句子的步长遍历文档。每个段落通过BERT的[CLS]标记嵌入表示。为获得文档表示$V_{t_d}^Q$，对段落嵌入取均值。随后，将以文本和实体为中心的文档表示相结合，通过线性投影$R^{m+n}$ →$R^p$学习单一的混合表示$d^Q$ ∈$R^p$，具体如下：

$\mathbf{d}^Q = W_2 \cdot [\mathbf{V}_{t_d}^Q, \mathbf{V}_{e_d}^Q] + \mathbf{b}$



### 文档排序

为学习文档评分函数S(d, Q)，首先挖掘查询嵌入Q∈ $R^p$（通过 BERT的[CLS]标记嵌入获取）与混合文档嵌入$d^Q$之间的多种细粒度交互关系，包括：加法交互：$$\mathbf{V}_{d}^{\text{add}} = Q + \mathbf{d}^Q$$，减法交互：$$\mathbf{V}_{d}^{\text{sub}} = Q - \mathbf{d}^Q$$，乘法交互（哈达玛积）：$$\mathbf{V}_{d}^{\text{mul}} = Q \circ \mathbf{d}^Q$$。随后通过线性投影R^{5p}→R学习评分函数S(d, Q)，具体形式为：

$S(d, Q) = W_3·V + b$

V是拼接嵌入向量$[Q, \mathbf{d}^Q, \mathbf{V}_{d}^{\text{add}}, \mathbf{V}_{d}^{\text{sub}}, \mathbf{V}_{d}^{\text{mul}}]$



### 端到端训练

使用以下二元交叉熵损失函数同时训练实体排序模型和文档排序模型：

$$\Large \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(p(\hat{y}_i)) + (1 - y_i) \cdot \left( 1 - \log(p(\hat{y}_i)) \right) \right)$$

其中，$y_i$为标签，$p(\hat{y}_i)$是实体/文档相关的预测概率。
