# Translational Generative Retrieval via Potential Query Generation ICASSP 2025
作者提出翻译生成式检索（TGR）通过动态建模文档的潜在语义，生成多样化查询分布，替代传统的固定DocID。

具体分为两阶段：
## 状态生成阶段
有向无环解码器（DAD）：作者提出的一种非自回归解码器（NAR）的变体，其核心功能是生成一个有向无环图（DAG）​，而非传统的线性序列。

给编码器Enc输入文档d，得到文档d的语义表示Enc(d)。将图位置嵌入向量G={g1，...，gL}和Enc(d)输入给有向无环解码器DAD，L是图的规模，设置为源文本长度N的λ倍，λ是一个超参数。

DAD将会生成顶点状态 $V = [v_1, \dots, v_L]^T$ ,vi经过处理后表示该顶点位置可能的查询单词的概率分布。

每条边e表示顶点之间的转移概率。任何连接一系列不同顶点的边的序列都能构成一条路径 $γ = [ e_1, \dots, e_{L_q} ]$ ,存储着一个潜在查询Si。并且所有与查询长度相同 的路径构成路径集合 $\{\Gamma = \{ \gamma_1, \gamma_2, \dots, \gamma_M \}\}$

路径γ的概率定义为：  
$$ P(\gamma \mid d) = \prod_{i=1}^{L_q - 1} P_\theta (e_{i+1} \mid e_i, d) = \prod_{i=1}^{L_q - 1} \mathbf{E}_{e_i, e_{i+1}} $$

其中 $\mathbf{E} \in \mathbf{R}^{L \times L}$ 是按行归一化的转移矩阵。通过以下方式得到的：

$$ \mathbf{E} = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{h}} \right), \quad \mathbf{Q} = \mathbf{V} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{V} \mathbf{W}_K $$

h是隐藏层维度，W_Q和W_K是可学习的参数。对E应用下三角掩码，以确保有向无环图中不存在循环。

## 解码阶段
 ​在给定路径 γ 和文档 d 的条件下，生成查询 q 的概率：

$$ P(q \mid \gamma, d) = \prod_{i=1}^{L_q} P(q_i \mid e_i, d) = \prod_{i=1}^{L_q} \text{softmax}(W_P v_{e_i}) $$

其中 $W_P$ 是可学习的权重， $v_{e_i}$ 是路径γ上第i个顶点的表示。

从文档的潜在查询分布S中生成查询q的总概率:

$$ P(q \mid \mathcal{S}) = \left\{ P(q \mid s_i) \right\}_{i=1}^M = \left\{ P(\gamma_i \mid d) P(q \mid \gamma_i, d) \right\}_{i=1}^M $$

损失函数使用最大似然函数：

$$ \mathcal{L} = -\log P_\theta(q \mid d) = -\log \sum_{\gamma \in \Gamma} P_\theta(q, \gamma \mid d) $$





