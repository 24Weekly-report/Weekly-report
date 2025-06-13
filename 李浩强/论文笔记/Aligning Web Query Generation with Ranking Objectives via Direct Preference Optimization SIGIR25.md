# Aligning Web Query Generation with Ranking Objectives via Direct Preference Optimization SIGIR25

作者提出了一个利用直接偏好优化（DPO）将排序信号整合到查询生成过程中的框架，旨在直接优化模型以生成高质量查询，从而最大限度地提高下游检索效果。



### 查询生成的方法

给定文档d，从语料库中采样一个负例文档d^-，然后提示查询生成器M_g模型（例如，选定的大型语言模型或针对查询生成任务微调的现有模型）生成一个针对d的查询q，同时要求该查询与d^- 的相关性较低



从语料库C中随机采样文档池，记为D。对于D中的每个文档，我们使用M_g生成一组n个合成查询 $Q_d=\{q_{d,1}，...q_{d,n}\}$



### 获取奖励

使用排序模型R为每个查询-文档对打分

1. 其中对于每个文档提取偏好数据集 $D_{rr}$，随机采样一对查询，要求排序模型R进行打分，其中一个查询的打分高于另一个 $\large D_{rr}=\{(d,q^+,q^-)|d \in D,(q^+,q^-) \in D,R(q^+,d)>R(q^-,d)\}$
2. 通过GPT3.5提示构建。给定文档d和其对应的查询集合 $Q_d$，模型G会被提示从 $Q_d$中显式选择最佳和最差查询，从而生成偏好数据集$D_{\text{gpt}}$：$\large D_{gpt}=\{(d,q^+,q^-)|d \in D,(q^+,q^-) \sim G(d,Q_d)\}$

### 生成器微调

在偏好数据集$D_{rr}$和$D_{\text{gpt}}$上使用 **直接偏好优化（DPO）损失函数**对基线生成器$M_g$进行训练

$$\Large \mathcal{L} = -\mathbb{E}_{(d, q^+, q^-) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma \left( \beta \log \frac{M_q^* \left( q^+ \mid d \right)}{M_q \left( q^+ \mid d \right)} - \beta \log \frac{M_q^* \left( q^- \mid d \right)}{M_q \left( q^- \mid d \right)} \right) \right] $$



其中$D_{\text{pref}}$为偏好数据集（如$D_{rr} $或$D_{\text{gpt}}$），$\beta $为用于调优的超参数。



### 密集检索器的训练

给定随机采样的k个文档集合$D_k$，我们使用$M_g^*$为每个文档生成一个查询，从而获得一组合成正样本对：$\large D_{train}=\{(q,d)|d \in D_k,q \sim M_g^*(d)\}$

给定训练数据集$D_{\text{train}}$，使用外部模型为每个查询采样硬负样本。对于一个查询的前100个文档，从排名低于原始正样本的文档中采样5个负样本。如果原始正样本文档不在前100名中，则将排名第一的文档视为新的正样本文档，而非丢弃该查询。

利用批次内负样本和跨GPU负样本对嵌入模型 \( M_e \) 进行微调，目标是最小化InfoNCE损失函数

$$\huge \mathcal{L} = -\log \frac{e^{\cos(M_e(q), M_e(d))}}{e^{\cos(M_e(q), M_e(d))} + \sum_{n \in \mathcal{N}} e^{\cos(M_e(q), M_e(n))}} $$

其中N为所有负样本的集合。嵌入模型$M_e(\cdot)$的表示通过平均池化获得，并使用余弦相似度进行比较。