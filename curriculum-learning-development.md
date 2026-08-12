# 从课程学习到自动课程学习：论文时间树、方法谱系与非线性训练反思

> 更新日期：2026 年 7 月 29 日  
> 范围说明：本文讨论广义课程学习（Curriculum Learning, CL）及自动课程学习（Automatic Curriculum Learning, ACL），覆盖监督学习、强化学习、自动环境设计、开放式学习与 LLM Agent。预印本与正式发表版本会分别标明；文中实验结论均按原论文报告，不代表独立复现。

## 摘要

课程学习通常被概括为“先易后难”，但其发展史实际上经历了更重要的范式转移：从人工规定样本顺序，到模型依据学习进展选择样本或任务，再到自动生成目标、初始状态和环境，最后发展为环境、任务与智能体共同演化的开放式学习系统。2023 年以后，LLM 又开始直接承担出题、生成环境代码和组织技能发展的角色。

与此同时，固定的“由简到繁”并不天然匹配神经网络训练。样本难度依赖模型、表示、目标与训练阶段；梯度价值不等于低损失；模型会遗忘、干扰、出现关键期与阶段跃迁。已有大规模实验也表明，在一些标准图像分类设置中，显式难度排序的收益很小，其增益可能主要来自训练集规模的动态变化。课程学习真正持久的思想因此不是一条单调上升的难度阶梯，而是：**闭环控制训练分布，使学习者在当前状态下持续获得有用、可迁移且不过度遗忘的训练信号。**

## 1. 定义与边界

### 1.1 课程学习

设训练单元为样本、任务、目标或环境实例，课程学习通过改变它们在训练期间的贡献分布来塑造学习轨迹。经典形式通常包括：

1. **难度度量器**：判断什么对当前学习者更容易或更有价值；
2. **训练调度器**：决定何时、以何种概率或权重呈现这些训练单元。

Bengio 等人在 2009 年将“由易到难”形式化为课程学习，并把它解释为 continuation method：先优化较平滑或较简单的目标，再逐渐逼近原始目标。这是一种重要机制，但不是所有有效课程的必要形式。

### 1.2 自动课程学习

[Portelas 等人的 ACL 综述](https://www.ijcai.org/proceedings/2020/671)给出的窄义定义是：ACL 自动调整训练数据分布，根据 DRL 智能体的能力选择学习情境。这里的“数据”可以是经验、目标、任务、对手或环境。

因此需要区分：

- **显式 ACL**：存在专门机制，以学习效率、学习进展、遗憾或覆盖率为信号优化训练分布；
- **涌现课程**：训练分布虽随策略变化，但只是其他机制的副产物，例如普通 on-policy RL；
- **相关范式**：主动学习选择“值得标注的数据”，难例挖掘强调高损失样本，持续学习强调跨任务保持，自博弈、UED 与开放式学习可以产生课程，但它们不与 ACL 完全同义。

## 2. 论文时间树

```text
训练顺序与“从简单开始”
│
├─ 1985  Selfridge, Sutton & Barto
│        Training and Tracking in Robotics
│        长杆/宽轨道 → 短杆/窄轨道；RL 中早期的显式训练序列。
│
├─ 1993  Elman
│        The Importance of Starting Small
│        控制输入复杂度和记忆跨度，帮助 RNN 学习语言结构。
│
└─ 2009  Bengio et al., Curriculum Learning, ICML
         正式命名 CL；把课程解释为 continuation method。
         │
         ├──────────────── 样本级课程 ────────────────┐
         │                                             │
         ├─ 2010  Self-Paced Learning                  ├─ 2015  Self-Paced
         │          模型按当前损失选择样本。            │          Curriculum Learning
         │                                             │          融合先验与自步选择。
         │
         └──────────────── 任务级自动课程 ─────────────┐
                                                       │
            2017 Graves et al.                         2017/2019/2020 TSCL
            非平稳多臂老虎机根据学习进展选任务。         Teacher 按学习曲线斜率选任务，
                                                       用负斜率发现遗忘。
         │
         ├──────────────── 目标与起点生成 ─────────────┐
         │                                             │
         ├─ 2017 Reverse Curriculum                    ├─ 2017 Asymmetric Self-Play
         │  从目标附近逐渐扩展起始状态。                │  Alice 出题、Bob 解题。
         │                                             │
         └─ 2018 Goal GAN                              └─ 2019 Curriculum Policy
            生成成功率中等的目标。                         把课程建模为高层 MDP。
         │
         ├──────────────── 连续任务空间 ───────────────┐
         │
         └─ 2019 ALP-GMM
            用绝对学习进展和 GMM 定位值得训练的参数区域。
         │
         ├──────────────── 环境生成与开放式学习 ───────┐
         │
         ├─ 2019 POET           多环境、多求解器共同演化与跨分支迁移。
         ├─ 2020 PAIRED / UED   用 regret 生成困难但可解的环境。
         ├─ 2020 PLR            重放学习潜力高的程序化关卡。
         ├─ 2020 Self-Paced DRL 用价值与目标分布的 KL 权衡课程。
         └─ 2021 XLand          动态任务生成与群体训练推进一般能力边界。
         │
         └──────────────── LLM 生成课程 ───────────────┐

            2023 Voyager        GPT-4 根据状态与技能库提出下一目标。
            2023 Curriculum Discovery
                                搜索非单调课程，而非预设单一方向。
            2024 Eurekaverse    LLM 生成并演化机器人训练环境代码。
```

## 3. 发展阶段与关键工作

### 3.1 思想源头：训练序列能够改变可学习性

[Training and Tracking in Robotics](https://www.ijcai.org/Proceedings/85-1/Papers/129a.pdf)先训练较容易的杆平衡设置，再切换到短杆和窄轨道，显著减少达到目标所需的失败次数。[Elman 1993](https://doi.org/10.1016/0010-0277(93)90058-4)进一步表明，“starting small”可以通过限制早期输入或模型可见的依赖范围改变语言结构学习。

[Bengio et al. 2009](https://icml.cc/2009/papers/119.pdf)把这些现象统一为课程学习。其关键贡献不是证明所有任务都应先易后难，而是提出：改变训练分布的路径可以改变收敛速度、正则化效应和最终落入的参数区域。

### 3.2 从人工排序到模型自定节奏

[Self-Paced Learning](https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html)不再要求人直接标注难度，而是优先选择当前损失较低的样本。[Self-Paced Curriculum Learning](https://ojs.aaai.org/index.php/AAAI/article/view/9608)随后把人工先验课程与模型自身的选择结合。

[Graves et al. 2017](https://proceedings.mlr.press/v70/graves17a.html)将课程中的任务视为多臂老虎机，用预测增益或模型复杂度增益作为奖励，自动发现随机 syllabus。[TSCL](https://doi.org/10.1109/TNNLS.2019.2934906)把 Teacher–Student 交互形式化：Teacher 选择学习曲线斜率最大的任务，也回访表现下降的任务。该工作 2017 年发布预印本、2019 年在线发表、2020 年见刊，年份差异来自版本阶段。

[Learning Curriculum Policies for RL](https://dl.acm.org/doi/10.5555/3306127.3331801)则把 Teacher 自身建模为高层课程 MDP：其状态是 Student 的知识状态，动作是下一训练任务。

### 3.3 从选择已有任务到生成目标

[Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300)从目标附近的可解起点开始，再向外扩张初始状态分布。[Asymmetric Self-Play](https://arxiv.org/abs/1703.05407)让 Alice 生成轨迹或目标、Bob 尝试复现，由双方能力共同推动难度变化。

[Goal GAN](https://proceedings.mlr.press/v80/florensa18a.html)学习生成成功率处于中间区间的目标：过易目标缺乏信息，过难目标没有有效奖励。这使课程从“排序一个固定集合”转向“维护当前能力边界附近的目标分布”。

### 3.4 学习进展与连续任务空间

[ALP-GMM](https://proceedings.mlr.press/v100/portelas20a.html)在连续参数化环境中，用相邻任务回报差估计绝对学习进展，再用 GMM 找到高进展区域。绝对值同时关注正向学习和负向遗忘，因此它本身已经不是单调先易后难。

[Self-Paced Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)从推断视角在高价值任务与目标任务分布之间进行 KL 权衡，使课程最终回到真正关心的目标分布。

### 3.5 环境—智能体共同演化

[POET](https://arxiv.org/abs/1901.01753)同时演化环境和求解器，并尝试在不同环境分支间迁移策略。课程不再是通往预设终点的单一路径，而是多条环境谱系之间的探索与转移。

[PAIRED](https://proceedings.neurips.cc/paper/2020/hash/985e9a46e10005356bbaf194249f6856-Abstract.html)提出 Unsupervised Environment Design：环境生成器最大化 antagonist 与 protagonist 的回报差。不可解环境使双方都失败，遗憾反而较小，因此生成器倾向于构造困难但可解的环境。

[Prioritized Level Replay](https://arxiv.org/abs/2010.03934)不生成新环境，而是按未来学习潜力重放已有程序化关卡。它说明课程控制可以发生在环境生成之后的选择层。

[XLand](https://arxiv.org/abs/2107.12808)将动态世界、游戏、对手与群体训练组合起来，使目标从解决单一任务转向覆盖巨大任务空间和获得零样本一般能力。

### 3.6 LLM 成为出题器和环境设计器

[Voyager](https://arxiv.org/abs/2305.16291)让 GPT-4 根据 Minecraft 当前世界状态、探索历史和技能库提出下一目标。它不是通过模型参数更新学习 curriculum policy，而是利用 LLM 的知识在上下文中生成开放式任务。

[Eurekaverse](https://arxiv.org/abs/2411.01775)进一步让 LLM 生成和演化环境程序，并用策略表现反馈下一轮环境设计。该工作 2024 年发布预印本，后收入 CoRL 2024 论文集；PMLR 正式页面于 2025 年发布。

## 4. 模型训练并不沿单一难度轴线性前进

“训练并不线性”的判断触及了经典课程学习最强的隐含假设：存在一个稳定的标量难度轴，而且模型能力沿这条轴单调增加。对现代神经网络，这两个条件通常都不成立。

### 4.1 难度是关系，不是样本的固定属性

同一样本对不同架构、初始化、预训练表示、损失函数和训练阶段可能具有不同难度。句子长度、图像清晰度和任务步数只是代理量；它们未必等于模型获得正确梯度所需的表示距离。

更准确的对象不是固定的难度函数

\[
d(z),
\]

而是依赖学习者状态的关系

\[
d(z;\theta_t,\mathcal H_t,\mathcal L),
\]

其中 \(z\) 是训练单元，\(\theta_t\) 是当前参数，\(\mathcal H_t\) 是学习历史，\(\mathcal L\) 是目标与损失。难度排序可以随训练反转，也可能只形成偏序而不是全序。

### 4.2 “容易”不等于“有学习价值”

低损失样本可能只是已被掌握，继续训练几乎不给新信息。高损失样本则至少有四种可能：

1. 位于当前决策边界附近，梯度很有价值；
2. 暴露尚未形成的可迁移结构；
3. 本身含糊或标注错误，不应被强化；
4. 对当前模型不可达，训练成本高但没有有效进展。

因此，“低损失优先”和“高损失优先”都不是普遍策略。课程控制真正需要估计的是边际学习价值，而不是样本表面难度。

### 4.3 优化器已经产生隐式课程

[When Do Curricula Work?](https://openreview.net/forum?id=tW4QEInpni)发现，不同初始化和相近架构往往以高度一致的顺序学会样本，说明架构与 SGD 自身存在 simple-first 偏置。额外的显式 easy-to-hard 排序可能只是重复这一偏置。

该研究在 CIFAR-10、CIFAR-100、FOOD101 和 FOOD101N 的实验中发现：标准设置下，课程、反课程和随机课程差异很小，随机顺序配合同样的动态训练集规模可以匹配或超过难度排序；当训练预算很小或标签噪声较高时，easy-to-hard 才表现出更稳定的优势。这个结论不能外推到所有任务，但它要求实验必须把“排序收益”与“pacing/子集规模收益”分开。

正向证据同样存在。[Hacohen 与 Weinshall 2019](https://proceedings.mlr.press/v97/hacohen19a.html)在 CIFAR 和 ImageNet 子集上使用 teacher transfer 与 bootstrapping 估计难度，报告了更快学习和更好测试表现，并分析了课程对优化景观的改变。两组结果并不直接矛盾：它们使用的难度度量、pacing、预算和对照实验不同，说明“课程是否有效”不能脱离具体机制与训练条件回答。

### 4.4 学习、遗忘与干扰是非单调的

[Toneva et al.](https://openreview.net/forum?id=ByxQB1BKwH)记录了 forgetting events：一个已被正确分类的样本可能在后续训练中重新出错。顺序训练还会产生任务间干扰和灾难性遗忘。因此，学过“简单任务”不意味着它会永久成为稳定前置能力。

TSCL 使用学习曲线斜率绝对值、ALP-GMM 使用绝对学习进展，正是为了重新采样正在退化的任务。有效课程往往需要回访、交错和 replay，而不是学完一层后永久离开。

### 4.5 关键期和 primacy bias 使早期课程可能锁死表示

[Critical Learning Periods in Deep Neural Networks](https://arxiv.org/abs/1711.08856)表明，训练早期的数据缺失可能造成后续训练难以完全修复的表示损伤；预训练甚至可能在相近任务间产生负迁移。早期只呈现“简单但狭窄”的数据不一定是在搭脚手架，也可能形成错误的强先验并降低后续可塑性。

### 4.6 能力可能经历平台期与跃迁

[Grokking](https://arxiv.org/abs/2201.02177)展示了一类受控任务中的延迟泛化：模型先记忆训练集，经过很长时间后测试泛化突然提升。Grokking 不是所有训练的普遍规律，但它证明训练损失、训练时间和泛化能力不必保持线性对应。依据短期成功率判断“当前能力级别”，可能错过正在形成但尚未显现的内部结构。

### 4.7 理论上，普通 batch/replay 也可能消除课程优势

[2022 年 teacher–student 分析](https://proceedings.neurips.cc/paper_files/paper/2022/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html)在可解析模型中发现：online 学习可从课程中获得速度优势，但当样本能够保存并 batch/replay 时，标准网络中的课程优势可能消失。只有用显式正则耦合不同阶段、保存阶段知识时，课程才重新获得明显收益。

这提示一个重要边界：数据顺序本身未必足够，学习算法还必须具备利用顺序结构的记忆与迁移机制。

### 4.8 高性能课程可以是非单调的

[Curriculum Discovery](https://arxiv.org/abs/2307.07412)在其 NLP 实验中搜索课程权重轨迹，发现多个高性能课程彼此不同，且常呈非单调形式；固定 easy-to-hard 与 hard-to-easy 都有欠佳风险。容易样本可能在后期重新变重要，样本也可随学习进展在难度组间移动。

这不是对所有领域的普遍定理，但直接支持了“课程应被发现或自适应，而非预设为一条直线”的方向。

## 5. 这是否否定课程学习

并不否定。课程学习在下列结构性条件下仍尤其有价值：

- **稀疏奖励与不可达探索**：直接训练几乎得不到成功轨迹，必须从可解目标或起点扩张；
- **噪声标签**：先用高置信样本形成稳定表示，可推迟对噪声的记忆；
- **有限训练预算**：不能充分遍历全数据时，需要优先分配计算；
- **continuation/smoothing**：目标函数确实存在可逐步恢复的平滑参数；
- **前置技能迁移**：简单任务产生的表示或策略能可靠迁移到后续任务；
- **安全约束**：真实机器人不能一开始就尝试危险动作；
- **巨大任务空间**：随机采样几乎总落在过易、不可解或无信息区域。

问题不在“要不要组织训练”，而在于不应把组织训练等同于静态、单调的难度排序。

## 6. 修正模型：课程是训练分布的闭环控制器

设 \(z\) 表示样本、目标、任务、对手或环境，Student 状态为 \(s_t\)，历史为 \(\mathcal H_t\)。Teacher 在每一步选择训练分布：

\[
q_t(z)=\pi_{\phi}(z\mid s_t,\mathcal H_t,\mathcal D_{\text{target}}).
\]

与固定的 \(d(z_1)<d(z_2)<\cdots\) 不同，\(q_t\) 可以回退、分叉、混合和重访。一个更贴近实际的效用可以写成：

\[
U_t(z)=
\alpha\,\mathrm{LP}_t(z)
+\beta\,\mathrm{IG}_t(z)
+\gamma\,\mathrm{Transfer}_t(z)
+\delta\,\mathrm{ForgetRisk}_t(z)
+\eta\,\mathrm{Coverage}(z)
-\lambda\,\mathrm{Cost}(z).
\]

这里 \(\mathrm{LP}\) 是学习进展，\(\mathrm{IG}\) 是信息增益，Transfer 是对目标任务的迁移价值，ForgetRisk 是遗忘风险，Coverage 保持目标分布覆盖，Cost 表示交互、算力或安全成本。这个式子是本文用于综合现有路线的分析框架，不是文献中统一采用的标准目标。

对应的课程形态应从“楼梯”改为：

```text
评估当前能力与遗忘
        ↓
估计各训练单元的边际学习价值
        ↓
选择 frontier + replay + target-anchor 的混合分布
        ↓
训练并观察能力、表示与覆盖变化
        └──────────────────────↺
```

## 7. 方法对比

| 方法 | 控制对象 | 主要信号 | 必然单调 | 优势 | 主要失效方式 |
|---|---|---|---|---|---|
| 经典 CL | 样本/任务顺序 | 人工难度 | 通常是 | 简单、可解释 | 难度错配，重复隐式课程 |
| Self-Paced | 样本权重 | 当前损失 | 常近似单调 | 自动、可去噪 | 低损失不等于高价值 |
| TSCL | 离散任务 | 学习曲线斜率 | 否 | 能发现进展与遗忘 | 斜率噪声、任务集合需预定义 |
| Goal GAN | 连续目标 | 中等成功率 | 否 | 自动扩展可解边界 | 中等成功不保证真实进展 |
| ALP-GMM | 连续任务参数 | 绝对学习进展 | 否 | 适合未知连续空间 | 距离度量与局部估计敏感 |
| POET | 环境与求解器 | 可解性、创新与迁移 | 否 | 多分支、开放式 | 计算昂贵，缺少明确终点指标 |
| PAIRED | 环境 | 近似 minimax regret | 否 | 生成困难但可解环境 | 对抗训练不稳，regret 近似有偏 |
| PLR | 已见关卡 | TD-error/学习潜力 | 否 | 轻量、改善泛化 | 不能超出已有或可生成关卡空间 |
| Voyager | LLM 生成目标 | 状态、技能与新颖性 | 否 | 利用先验知识开放探索 | 依赖 LLM 判断与外部反馈可靠性 |
| Eurekaverse | 环境程序 | 策略表现与 LLM 演化 | 否 | 可生成语义丰富环境 | 成本高，环境代码与真实物理有偏差 |

## 8. 设计与评测原则

1. **能力应是向量而非标量**：分别跟踪感知、记忆、规划、控制和不同任务族能力。
2. **难度应相对当前学习者定义**：同时报告静态难度与模型条件难度，避免把人类直觉当作事实。
3. **采样能力边界而非只采“简单”**：优先选择有进展、可迁移且可解的 frontier。
4. **保留 replay 与回访**：把负学习进展和遗忘当作课程信号，而不是异常噪声。
5. **用 DAG 或任务图替代全序**：多个技能可以并行发展，复杂任务也可能依赖不同前置组合。
6. **锚定目标分布**：持续混入目标任务，避免课程分布与测试分布长期脱节。
7. **把课程与学习算法共同设计**：检查模型是否有机制保存、组合和迁移阶段知识。
8. **做必要消融**：至少比较 uniform shuffle、相同 pacing 的随机课程、反课程、难例优先和完整目标分布训练。
9. **分离三种收益**：排序、训练集规模变化和总计算量不能同时改变后只归因于课程。
10. **报告路径而不只报告终点**：包括样本效率、遗忘、覆盖率、最坏组性能、泛化与安全成本。

## 9. 推荐阅读顺序

1. Bengio et al. 2009：理解经典定义与 continuation 解释。
2. Kumar et al. 2010：从人工难度转向学习者自身损失。
3. Graves et al. 2017 与 TSCL：理解学习进展驱动的自动调度。
4. Goal GAN 与 ALP-GMM：理解连续目标/任务空间中的能力边界。
5. POET、PAIRED 与 PLR：比较环境演化、regret 和 replay 三种路线。
6. Portelas et al. 与 Narvekar et al. 两篇 2020 综述：建立 ACL/RL 分类框架。
7. Wu et al. 2021：理解“排序为何可能没有贡献”的负面证据。
8. 关键期、forgetting events 与 grokking：补足非线性训练动力学。
9. Curriculum Discovery：理解非单调课程搜索。
10. Voyager 与 Eurekaverse：观察 LLM 如何把课程扩展到开放任务和环境程序。

## 10. 结论

课程学习从人工排列样本，发展到自动选择任务、生成目标、设计环境和开放式共同演化。其历史看似不断扩大“课程”的对象，实际更深的变化是：课程从静态先验变成了学习系统内部的反馈控制问题。

固定的由简到繁只在难度稳定、迁移方向明确、能力近似单调时成立。现代模型训练则常包含隐式 simple-first 偏置、样本干扰、遗忘、关键期、目标分布偏移和阶段跃迁。未来更合理的 ACL 因此不是找到一条完美难度排序，而是持续回答：**对这个学习者、在这个时刻、考虑长期迁移与遗忘之后，什么训练经验最值得发生？**

## 参考文献

1. Selfridge, Sutton, Barto. [Training and Tracking in Robotics](https://www.ijcai.org/Proceedings/85-1/Papers/129a.pdf), IJCAI 1985.
2. Elman. [Learning and Development in Neural Networks: The Importance of Starting Small](https://doi.org/10.1016/0010-0277(93)90058-4), Cognition 1993.
3. Bengio, Louradour, Collobert, Weston. [Curriculum Learning](https://icml.cc/2009/papers/119.pdf), ICML 2009.
4. Kumar, Packer, Koller. [Self-Paced Learning for Latent Variable Models](https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html), NeurIPS 2010.
5. Jiang et al. [Self-Paced Curriculum Learning](https://ojs.aaai.org/index.php/AAAI/article/view/9608), AAAI 2015.
6. Graves et al. [Automated Curriculum Learning for Neural Networks](https://proceedings.mlr.press/v70/graves17a.html), ICML 2017.
7. Florensa et al. [Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/abs/1707.05300), CoRL 2017.
8. Sukhbaatar et al. [Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play](https://arxiv.org/abs/1703.05407), 2017.
9. Matiisen et al. [Teacher–Student Curriculum Learning](https://doi.org/10.1109/TNNLS.2019.2934906), 2017 preprint; 2019 online; IEEE TNNLS 2020 issue.
10. Florensa et al. [Automatic Goal Generation for Reinforcement Learning Agents](https://proceedings.mlr.press/v80/florensa18a.html), ICML 2018.
11. Narvekar, Stone. [Learning Curriculum Policies for Reinforcement Learning](https://dl.acm.org/doi/10.5555/3306127.3331801), AAMAS 2019.
12. Portelas et al. [Teacher Algorithms for Curriculum Learning of Deep RL in Continuously Parameterized Environments](https://proceedings.mlr.press/v100/portelas20a.html), CoRL 2019 proceedings published 2020.
13. Wang et al. [Paired Open-Ended Trailblazer](https://arxiv.org/abs/1901.01753), 2019.
14. Dennis et al. [Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design](https://proceedings.neurips.cc/paper/2020/hash/985e9a46e10005356bbaf194249f6856-Abstract.html), NeurIPS 2020.
15. Jiang et al. [Prioritized Level Replay](https://arxiv.org/abs/2010.03934), 2020.
16. Klink et al. [Self-Paced Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html), NeurIPS 2020.
17. Team et al. [Open-Ended Learning Leads to Generally Capable Agents](https://arxiv.org/abs/2107.12808), 2021.
18. Wang et al. [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291), 2023.
19. Elgaar, Amiri. [Curriculum Discovery](https://arxiv.org/abs/2307.07412), 2023.
20. Liang et al. [Environment Curriculum Generation via Large Language Models](https://proceedings.mlr.press/v270/liang25a.html), CoRL 2024 proceedings published 2025; arXiv 2024.
21. Portelas et al. [Automatic Curriculum Learning for Deep RL: A Short Survey](https://www.ijcai.org/proceedings/2020/671), IJCAI 2020.
22. Narvekar et al. [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://jmlr.org/papers/v21/20-212.html), JMLR 2020.
23. Wang, Chen, Zhu. [A Comprehensive Survey on Curriculum Learning](https://arxiv.org/abs/2010.13166), 2020/2021.
24. Hacohen, Weinshall. [On the Power of Curriculum Learning in Training Deep Networks](https://proceedings.mlr.press/v97/hacohen19a.html), ICML 2019.
25. Toneva et al. [An Empirical Study of Example Forgetting During Deep Neural Network Learning](https://openreview.net/forum?id=ByxQB1BKwH), ICLR 2019.
26. Achille, Rovere, Soatto. [Critical Learning Periods in Deep Neural Networks](https://arxiv.org/abs/1711.08856), ICLR 2019 version.
27. Power et al. [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177), 2022.
28. Wu, Dyer, Neyshabur. [When Do Curricula Work?](https://openreview.net/forum?id=tW4QEInpni), ICLR 2021.
29. Mannelli et al. [An Analytical Theory of Curriculum Learning in Teacher–Student Networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html), NeurIPS 2022.
