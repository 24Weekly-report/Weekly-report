# DRAMA： Diverse Augmentation from Large Language Models to Smaller Dense Retrievers  ACL2025

这篇论文提出了一种名为 **DRAMA** 的训练框架，旨在利用大语言模型（LLMs）生成多样化的增强数据，并通过剪裁LLM的模型结构来训练更小但性能强大的稠密检索器（dense retriever）。



#### **1. 数据增强（Data Augmentation）**

输入：文档库 C（例如：25M个文档，每个文档分为256个token的片段）。预训练的LLM检索器（如Llama3.1 8B）。

对每个文档随机裁剪一个句子作为伪查询 q。用LLM检索器对 q检索文档库，返回Top-50候选文档。将Top-1~10作为正例 $D^+$，Top-30~50作为硬负例$D_{HN}$。

输出：增强的三元组 (q,$D^+$,$D_{HN}$)。

##### 2.**基于指令LLM的查询生成**

输入：文档库 C。指令LLM（如Llama3.3 70B-Instruct）。

用指令LLM为每个文档生成更自然的查询 q(例如：*“请生成一个与该文档相关的问题”*）。用LLM检索器对生成的查询检索文档库，筛选正例和硬负例。

输出：更真实的查询及其三元组。

##### **3.基于指令LLM的排序偏好**

输入：方法2生成的查询及其Top-20候选文档。

用指令LLM对候选文档重新排序（如：*“请根据相关性排序以下文档”*）。

将重排后的Top-1作为正例，Top-10~20作为硬负例。

输出：质量更高的三元组。



#### **4. 模型剪裁（Pruning）**

将大LLM剪裁为更小的检索器骨干，保留多语言和长上下文能力。

输入：原始LLM（如Llama3.18B8*B*）。目标参数规模（如0.1B或0.3B）。

结构化剪裁：通过约束优化（如Lagrange乘子）剪裁注意力头、隐藏层等组件。连续预训练：用多语言数据（如10B token）微调剪裁后的模型。

修剪掩码z应用于模型的硬 concrete 分布。通过拉格朗日乘数法施加约束，以确保修剪后的模型符合目标架构。例如，针对目标注意力头数量H_T，某一层应用的损失函数定义为

$\Large \tilde{\mathcal{L}}^{\text{head}}(\lambda, \phi, z) = \lambda^{\text{head}} \cdot \left( \sum_{i} z_{i}^{\text{head}} - H_{\mathcal{T}} \right) + \phi^{\text{head}} \cdot \left( \sum_{i} z_{i}^{\text{head}} - H_{\mathcal{T}} \right)^2$

同样地，完整的修剪损失整合了对其他结构组件的约束，包括层掩码$z^{layer}$、隐藏维度掩码$z^{hidden}$、注意力头掩码$z^{head}$和中间维度掩码$z^{int} $。这些约束与标准的语言建模目标相结合，具体如下：

$\Large \mathcal{L}_{\text{prune}}(\theta, z, \lambda, \phi) = \mathcal{L}(\theta, z) + \sum_{j=1}^{L_S} \tilde{\mathcal{L}}_j^{\text{head}} + \sum_{j=1}^{L_S} \tilde{\mathcal{L}}_j^{\text{int}} + \tilde{\mathcal{L}}^{\text{layer}} + \tilde{\mathcal{L}}^{\text{hidden}}$



#### **5. 单阶段对比学习训练**

输入：增强的三元组数据（方法1-3）。剪裁后的小模型（如0.1B参数）。

使用InfoNCE损失函数，优化查询和文档的余弦相似度。每个查询配1个正例和7个硬负例（0.1B模型）。

$$\Large \mathcal{L}(q, D^+, \{D_{\mathcal{N}}\}) = -\log p(D = D^+ \mid q) = -\log \frac{\exp(\text{Sim}(q, D^+)/\tau)}{\sum_{D_i \in \{D^+\} \cup \{D_{\mathcal{N}}\}} \exp(\text{Sim}(q, D_i)/\tau)} $$

输出：训练好的小检索器。
