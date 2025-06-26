# reinforcement learning for adversarial query generation to enhance relevance in cold start product search ACL25



作者提出了一种新颖的**对抗性强化学习框架（Adversarial Reinforcement Learning Framework）**，通过生成对抗性查询来增强分类器的鲁棒性。整个方法是一个**迭代循环**的过程，包含初始化、数据增强、强化训练三个主要阶段，循环执行多次（论文中是4次）。



### 初始化

初始步骤与后续强化训练的微调方法一致。

开始时使用预训练的LLM（如FLAN-T5）作为生成器基座，预训练的分类模型（如基于FLAN-Small Encoder的分类头）作为分类器基座。

输入给模型真实数据三元组(产品元数据, 用户真实查询, 相关性标签)。

**生成器** 进行监督微调得到 $G_0$：训练它根据给定的(产品元数据, 相关性标签)生成类似用户真实查询的文本。

**分类器 ** 进行监督训练得到$C_0$：训练它根据查询预测其所属类别（这里是二分类：药品相关 / 药品不相关）。



### 数据增强

对于每个冷启动商品（新出现的没有相关查询的商品），使用生成器 $G_N$(一开始是$G_0$)，输入(商品元数据, 相关性标签)，生成一批合成查询。

将这些新生成的合成查询（及其对应的商品元数据和标签）收集起来，形成$D_N^{syn}$

将$D_N^{syn}$与之前的训练数据$D_{N-1}$合并，得到增强数据集 $D_N = D_{N-1} ∪ D_N^{syn}$。

使用增强数据集 $D_N$重新训练/微调分类器，得到新版本的分类器 $C_{N+1}$。



### 强化训练

使用当前生成器$G_N$，刚更新的分类器 $C_{N+1}$，对生成器进行PPO优化。

PPO函数：$$\mathcal{L}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]$$

E表示经验期望， $\large r_t(\theta) = \frac{\pi_\theta(a_t \vert s_t)}{\pi_{\theta_{\text{old}}}(a_t \vert s_t)}$ 为新策略$\pi_\theta$与旧策略$\pi_{\theta_{\text{old}}}$的概率比值。$a_t$表示序列中第t位置选择的token，$s_t$为第t位置的上下文（所有先前token），$A_t$是优势估计值（由奖励函数推导得出），ϵ是约束策略更新的超参数。$A_t$通过序列末尾分类器的奖励函数计算得到，由超参数ϵ控制的裁剪操作可防止策略更新幅度过大导致训练不稳定。

对每个冷启动商品，$G_N$基于一对 (商品元数据, 相关标签)生成一个合成查询 $q_{syn}$。

将生成的查询 $q_{syn}$输入到当前最强的分类器$C_{N+1}$中，得到其预测的概率分布 $C_{N+1}(y|q_{syn})$和计算出的分类损失 $L_{cls}$（通常是交叉熵损失，基于预测概率和真实标签）。 $L_{cls}$ 越大，说明$C_{N+1}$对这个查询$q_{syn}$越不确定或越容易分错！这正是作者想要的“有挑战性”的查询。

##### ***\*参数化奖励函数\****

当生成器按序生成文本时，结构化奖励函数在每个token位置定义如下：

对于每个token位置t < T（序列结束符EOS之前）：$ \mathcal{R}(t) = -\beta \cdot D_{\text{KL}}(\pi_\theta \vert\vert \pi_{\text{FT}}) $

对于最终token位置t = T（EOS处）：$\mathcal{R}(T) = \alpha \cdot L_{\text{cls}} - (1 - \alpha) \cdot \log P_{\text{gen}} - \beta \cdot D_{\text{KL}}(\pi_\theta \vert\vert \pi_{\text{FT}})$，$D_{KL}$表示每个token位置上当前策略与微调策略之间的KL散度，以确保生成器不会过度偏离预训练的分布。项$P_{gen}$表示生成概率



 使用计算出的奖励 $R(T)$ 作为强化学习的奖励信号，通过近端策略优化（PPO）算法更新生成器的参数 θ。PPO 的核心是最大化期望奖励，同时通过裁剪（Clip）机制限制策略更新的幅度，保证训练稳定性。
