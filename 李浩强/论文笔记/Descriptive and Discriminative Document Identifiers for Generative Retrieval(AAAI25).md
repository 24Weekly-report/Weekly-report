# Descriptive and Discriminative Document Identifiers for Generative Retrieval(AAAI25)

这篇论文提出了一种名为​​D2-DocID​​的文档标识符设计方法，以及生成式检索模型​​D2Gen​。

### D2-DocID​​标识符

对于每个文档D，使用NLTK工具包来获取文档长度最多为k的单词或词组集合 $G_D$ ，文档中全部单词表示为D=( $d_1, d_2, ..., d_n$ )。 $G_D$ 每个单词或词组g=( $d_i, ..., d_j$ )。

将D输入到检索模型D2Gen编码器中，计算g和D之间的相关性，计算公式如下： $REL(D,g)=MP(Att(Encoder(D))[di:dj]$

其中，Att表示模型最后一层的注意力分数，MP表示均值池化。

遍历整个语料库C中的所有文档，并将相关性得分记录在矩阵M中，即： $M[D,g]=REL(D,g),for$  $every D∈C，g∈G_D$

作者设计了类似与tf-idf的方法，NR-IDR来选择DocID。

将语料库C中所有文档构成的单词或词组集合记为 $G = (g_1, g_2, ... , g_{|G|})$ 。对于任意单词或词组 $g_j ∈ G$ ，其逆文档相关度IDR[j]计算为：

$\text{IDR}[j] = \log \left( \frac{1}{\text{Mean}_{k \in I_j}(\mathcal{M}[k, j])} \right)$     

$= \log \left( \frac{|I_j|}{\sum_{k \in I_j} \mathcal{M}[k, j]} \right)$

IDR[j]由 $g_j$ 与所有相关文档的平均相关度决定。 $g_j$ 的IDR值越低，其与所有相关文档的平均相关度越高，这也表明 $g_j$ 独特性越弱，

其中， $I_j$ 表示矩阵M中第j列元素g和D之间相关性分数不为 None 的行的索引，其公式可表示为：

$I_j =$ { $\{k\in [0, |C|) \mid \mathcal{M}[k, j] \neq \text{None} \}$ }

任意文档Di及其任意单词或词组gi计算NR-IDR得分如下：

$\text{NR-IDR}[i, j] = \mathcal{M}[i, j] * \sqrt{\text{TF}[i, j]} * \text{IDR}[i]$

其中TF[i,j]表示gi在文档di中出现的次数。

基于NR-IDR分数选择DocID，去重规则为，如果某个单词或词组中的全部单词与已选DocID中的单词重复，则去除该单词或词组：

1. 对任意文档 D_i，根据NR-IDR对其单词或词组集合 $G_{D_i}$ 进行排序；
2. 按顺序对集合进行去重，并选择前k个高分单词或词组作为 D_i的DocID。

### D2Gen​模型

首次训练：

使用T5-base作为基础模型，首先通过两个损失增强模型的文档理解能力。给定训练数据集D ={(Q, D)}，输入给模型查询Q的编码向量，让模型生成DocID，生成损失 $L_r$ :

$L_r(\Theta_{e, d}) = \sum_{i}^{l} \log P(id_{i} \mid q, id_{<i}; \Theta_{e, d})$

其中，l表示 DocID 的长度， $Θ_{e,d}$ 表示模型参数。

对比损失：

![](https://github.com/jiey-nlp/informationRetreival/blob/16fb4124d4654c0abf84f332ef25edd6507c5051/%E6%9D%8E%E6%B5%A9%E5%BC%BA/paper/image/25-4-25-1.png)

其中 $h_q,h_d$ 分别表示查询与文档的编码向量，s(·,·)表示余弦相似度，d-表示负样本文档，τ为参数

最终损失为：  $L_p=L_r+λ*L_c$ 。

二次训练：

#### 构建数据集

给语料库中所有文档D生成​​D2-DocID​​。

将语料库中的每个文档D，将其分割为连续段落集合 P = {p_1, p_2, ...}，其中每个段落包含s = 3个句子，且段落间重叠o = 1 个句子。

随后，直接使用预训练的查询生成模型，输入文档原文D和段落P生成伪查询：从原文D生成n_d = 10 个伪查询，记为Q_d = {q_1, q_2, ...}；从每个段落p_i生成n_p = 3个伪查询，记为Q_p = {q_{ij}}，1<=i<=|P|，1<=j<= $n_p$

利用密集检索模型对上面每个伪查询进行检索，计算其 MRR@10，计算其与过滤后的伪查询集合 $Q_s$ 中所有已选伪查询的余弦相似度得分sim_q。

如果伪查询q的余弦相似度小于相似度阈值 $λ_1$ 且其MRR@10值大于阈值 $λ_2$ ，则将q加入 $Q_s$ 。

使用上面初步训练过的模型，输入语料库的文档，真实查询与过滤后的伪查询集合,​​D2-DocID，训练模型。

损失 $L_{gr}$ :

$L_{gr}(\Theta_{e, d}) = \sum_{i}^{l} \log P(id_{i} \mid q, id_{<i}; \Theta_{e, d})$
