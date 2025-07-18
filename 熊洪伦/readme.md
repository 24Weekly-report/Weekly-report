# 25.7.18

## 读了一篇论文 ThinkQE: Query Expansion via an Evolving Thinking Process

ThinkQE——一种将基于LLM的思考过程与动态语料库交互紧密结合的查询扩展框架。

1.初始证据检索

设q₀表示原始用户查询。为使扩展过程基于语料库证据，我们首先使用第一阶段的词汇检索器（本实现采用BM25）从语料库C中检索初始文档集：​​D₀ = TopK(BM25(q₀, C))​​，其中D₀为按BM25相关度排序的Top-K文档列表，作为扩展的初始反馈信号。

2.基于思考的扩展

采用经过R1蒸馏训练的LLM（该模型被预训练为在回答前生成思考链），给定q₀和D₀，模型执行两阶段过程：

​​思考阶段​​：模型分析q₀与D₀，识别潜在概念、消除歧义，并挖掘信息需求的其他解释或缺失维度

​​扩展阶段​​：基于思考输出生成扩展段e₁，通过引入相关术语和概念扩充原始查询

3.基于语料反馈的迭代优化

通过多轮迭代（t=1,...,T）动态优化扩展，每轮包含：

​​检索​​：用当前查询qₗ检索文档 ​​Rₗ = BM25(qₗ, C)​​

​​冗余过滤​​：排除(a)黑名单Bₗ或(b)前轮结果Dₗ₋₁中的文档，得到 ​​Dₗⁿᵉʷ = TopK(Rₗ(Bₗ∪Dₗ₋₁))​​，并更新黑名单

​​思考扩展​​：基于q₀和Dₗⁿᵉʷ生成新扩展eₗ₊₁（同2.2节流程）

​​查询更新​​：连接扩展 ​​qₗ₊₁ = qₗ⊕eₗ₊₁​​

为缓解迭代过程中原始意图稀释，在最终查询中重复原始查询n次（n=扩展总词数/(λ×q₀词数)，λ=3）。

## 准备机器人比赛

# 25.7.11

## 1.给上海的横向解决了一个bug（输入错误数字的情况下，前端会提示输入错误，但是系统依旧会保存）

## 2.共同完成了一道建模题

## 3.看了一篇论文 ExpandR: Teaching Dense Retrievers Beyond Queries with LLM Guidance

方法概述：联合优化LLM（生成查询扩展）和稠密检索器，提升语义匹配效果。

1. 整体架构

输入：原始查询q,文档集合D

LLM生成扩展：

<img width="330" height="72" alt="image" src="https://github.com/user-attachments/assets/3f109eae-99fb-428c-9137-201beca4d5a5" />


指令模板："请根据问题生成信息丰富的扩展段落"

联合概率建模：

<img width="636" height="103" alt="image" src="https://github.com/user-attachments/assets/8b603d78-4873-4a5b-a1de-a1c66eee5fd6" />

其中  $$\Phi$$  （检索器参数） $$\mathrm{9}$$ （LLM参数）联合优化。

2. 检索器优化
   
   查询表示融合：  $$\vec{q}^{\exp}=\frac{\vec{q}+\vec{d}^{\exp}}{2}$$

   将原始查询嵌入 $$\vec{q}$$ 与扩展内容嵌入  $$\vec{d}^{\exp}$$ 平均，作为新查询表示。

   对比学习损失：

<img width="557" height="101" alt="image" src="https://github.com/user-attachments/assets/e8229069-ebcc-4dcf-86d8-6d1b07803344" />

负样本 $$\mathcal{D}^{-}$$ 来自同批次内采样

3. LLM优化

奖励函数设计

<img width="474" height="105" alt="image" src="https://github.com/user-attachments/assets/1a1c828e-dac0-42b6-80bb-474a033f1e1f" />

自奖励:用LLM基于标准答案 $$d_{*}$$ 生成回复y（指令："根据查询和相关文档生成直接答案"）。计算y与扩展 dexp的相似度排名  $$R_{\mathrm{self}}(d^{\exp})=\frac{1}{\mathrm{Rank}(y,d^{\exp})}$$    确保扩展与答案语义一致。

检索奖励:将标准答案  $$d_{*}$$   视为伪查询，计算其与扩展 dexp的排名：

$$R_{\mathrm{rank}}(d^{\exp})=\frac{1}{\mathrm{Rank}(d_*,d^{\exp})}$$   对齐检索器偏好，提升扩展的检索效用。

DPO优化LLM:对每个查询 q，用不同温度采样生成扩展候选集 $$\mathcal{D}^q=\{d_1^{\exp},\ldots,d_k^{\exp}\}$$  ,根据奖励 $$R(\cdot)$$ 构建偏好对 $$(d_+^{\exp},d_-^{\exp})$$  满足 $$R(d_+)>R(d_-)$$

DPO损失函数：

$$\mathcal{L}(\mathcal{M};\mathcal{M}^{\mathrm{Ref}})=-\mathbb{E}{(q,d_+,d_-)}\left[\log\sigma\left(\beta\log\frac{\mathcal{M}(d_+\mid q)}{\mathcal{M}^{\mathrm{Ref}}(d_+\mid q)}-\beta\log\frac{\mathcal{M}(d_-\mid q)}{\mathcal{M}^{\mathrm{Ref}}(d_-\mid q)}\right)\right]$$

其中β为缩放超参,  $$\mathcal{M}^{\mathrm{Ref}}$$  为冻结的参考模型。

# 25.7.4

## 1.看了一篇论文 Corpus-Steered Query Expansion with Large Language Models

方法概述：

1.初筛文档：用 BM25 从语料库检索原始查询的前 10 篇文档（长文档截断至 128 tokens）。

2. 提取关键句：通过 LLM 的单样本提示（如 “识别相关文档并提取贡献相关性的句子”），从初筛文档中抽取事实性句子（83% 与原文完全一致）。
   
3. 生成假设扩展：LLM 针对原始查询生成假设性回答（如 “解释鲨鱼温血机制” 的段落），补充语义知识。
   
4. 融合扩展查询：将原始查询（重复 2 次）、语料库关键句、假设回答拼接，强化主题与事实性。

5. 二次检索：用融合后的查询再跑 BM25，输出最终结果。

## 2. 跑新idea实验，跑出来结果没基线好，打算重新换个思路去做，下个周打算将看的两篇论文缝合起来看看效果

## 3.写数学建模一题。

# 25.6.27

## 1.完成数学建模一道题

## 2.看了一篇论文 QA-Expand: Multi-Question Answer Generation for Enhanced Query Expansion in Information Retrieval

## 3.正在mmlf这篇论文上加代码

# 25.6.20

## 1.复现了一篇论文，打算将这篇论文作为我的baseline，下一步思考在这篇论文的基础上创新，论文名（MMLF:Multi-query Multi-passage Late Fusion Retrieval），发表于naacl 2025

MMLF方法概述：

首先用大模型从原始查询分解出多个子查询（如针对“COVID-19治疗药物”生成“瑞德西韦疗效”“中药抗病毒作用”等子意图），接着将每个子查询扩展成独立段落（段落需同时回应原子查询和子查询），随后对原始查询和每个段落分别进行稠密检索，得到多个结果列表，最终通过排序融合算法（RRF）合并这些列表——即根据文档在不同列表中的排名计算加权得分，生成统一的排序结果。



## 2.读了一篇相同领域的论文

MILL: Mutual Verification with Large Language Models for Zero-Shot Query Expansion

MILL方法流程总结​

​​1.输入​​：原始查询q

​​2.生成阶段​​：

生成N个子查询 $\{q_1,q_2,\ldots,q_N\}$ ​​每个子查询生成一个文档​​ $d_n^{LLM}$ （包含子查询+对应段落）  得到N个生成文档， $\mathcal{D}^{LLM}=\{d_1^{LLM},\ldots,d_N^{LLM}\}$

3.​​检索阶段​​：

用 BM25 检索原始查询 q，得到K个文档 $\mathcal{D}^{PRF}=\{d_1^{PRF},\ldots,d_K^{PRF}\}$

​​4.互验证：

构建N×K相似度矩阵（计算每个 $d_n^{LLM}$ 与 $d_k^{PRF}$ 的余弦相似度）

计算生成文档得分： $s_n^{LLM}=\sum_{k=1}^K\sin(d_n^{LLM},d_k^{PRF})$

计算检索文档得分： $s_k^{PRF}=\sum_{n=1}^N\sin(d_k^{PRF},d_n^{LLM})$

筛选： 取 $\mathrm{Top}_{N^{\prime}}$ 生成文档 $\mathcal{D}_s^{LLM}$ （高 $s_n^{LLM}$  值）

取 $\mathrm{Top}_{K^{\prime}}$ 检索文档 $\mathcal{D}_s^{PRF}$ （高 $s_k^{PRF}$ 值）

5.查询扩展​​：

构造新查询 

$\mid q^{\prime}=\underbrace{qqqqq}_{\text{原始查询重复5次}}\mathcal{D}_s^{PRF}\mathcal{D}_s^{LLM}$

6.最终检索​​：用 $q^{\prime}$ 执行稠密检索（如 ANCE、DPR）

# 25.6.13  

1.解决了上海项目的两个新需求

2.完成了数学建模的一道题

3.读了一篇文章：Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation

笔记如下

# 方法

提出​​混合粒度方法：通过路由模块基于输入查询动态确定知识源的最优信息粒度。该路由模块采用软标签损失函数进行高效训练。我们进一步将MoG扩展为​​图混合粒度，将参考文档预处理为图结构，实现对分散文本片段的检索。

MOG

将知识库文档按n种粒度分块（如 n=5），其中粒度1为最细（如单句），粒度 j(j>=2)由粒度 j-1的 2 个相邻块合并而成，形成层级化分块结构。

对每个粒度下的片段，使用BM25算法计算与查询 q 的相似度分数，衡量语义相关性。

每个粒度提取前k个高相关片段（如 (k=3)），形成总大小为 n x k的候选片段池。

通过 RoBERTa 将查询 q 编码为向量，经路由器映射为权重向量w，维度与n一致，表征各粒度的重要性。

将各粒度下片段的 BM25 分数与 w 加权整合，突出最优粒度的片段相关性。

软标签构造（用于训练mlp）

对于每个查询 q，使用 BM25从每个粒度级别的参考文档中检索最相关的片段（S_best），然后通过静态模型（包括 TF-IDF、RoBERTa或命中率得分计算 S_best 中每个片段与标签 l 的语义相似度，并存储于 sim_best。我们为 S_best 中相似度最高（次高）的片段分配 0.8（0.2）的软标签，其余片段补 0。路由器通过最小化二元交叉熵损失函数进行训练

MoG 的两步选择策略：

第一步：从 最细粒度 中选 top-k 相关块（chunk_r，最细粒度的单元）；
第二步：找到 最优粒度g_r 中 包含chunk_r的块 ，作为最终检索结果。


MOGG

MoG 的局限：仅通过调整粒度处理相邻片段，无法有效应对复杂问题中分散在不同段落 / 文档的信息（如跨源推理需求）。

将文档拆分为 1-2 个句子的节点，基于 BM25 计算节点相似度，超过阈值 Tgraph 则连边，构建语义关联图。

延续 MoG 的多粒度路由框架，仅将分块逻辑从 “线性窗口” 改为 “图跳跃范围”，其余组件（如路由器、软标签训练）不变。




