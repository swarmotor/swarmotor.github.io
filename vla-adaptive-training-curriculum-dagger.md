# VLA 自适应训练：课程学习、DAgger 与闭环数据改进

> 更新日期：2026-07-29  
> 主题：Vision-Language-Action（VLA）模型中的课程学习、数据聚合、失败恢复、在线强化学习与自我改进。

## 1. 核心判断

传统课程学习通常采用固定的 `easy -> hard` 顺序，但它并不完全匹配 VLA 的训练特性。更适合 VLA 的范式是：

> **Policy-dependent adaptive data curriculum**（策略依赖的自适应数据课程），或者 **closed-loop competence shaping**（闭环能力塑造）。

课程不应是预先排列好的静态教学大纲，而应由当前策略的失败分布、不确定性、学习进展、技能依赖关系和部署反馈动态生成。

一个更合理的总体训练闭环是：

```text
多任务行为克隆预训练
  -> 当前策略自主 rollout
  -> 定位能力边界和高风险状态
  -> 收集专家纠正、恢复及失败数据
  -> 根据价值、优势和信息增益重加权
  -> 与基础数据 replay 后重新训练
  -> 在仿真或世界模型中扩增 OOD 状态
  -> 少量真机校准
  -> 进入下一轮
```

## 2. 为什么固定课程学习不完全适合 VLA

### 2.1 传统课程学习的隐含假设

固定的简单到复杂课程通常假设：

1. 样本或任务难度能够用一个标量排序。
2. 简单任务和复杂任务共享近似一致的最优表征。
3. 简单任务的梯度大体指向复杂任务的解。
4. 模型能力随训练过程近似单调增长。
5. 从简单任务到复杂任务存在平滑、正向的迁移路径。

这些假设在 VLA 中经常不成立。

### 2.2 VLA 难度是多维的

VLA 样本的难度可以表示为：

\[
d(x) = \left(
d_{\text{semantic}},
d_{\text{spatial}},
d_{\text{control}},
d_{\text{contact}},
d_{\text{horizon}},
d_{\text{embodiment}}
\right)
\]

例如：

- 插孔任务语义简单，但接触和控制精度很高。
- 长时序整理任务可能由多个简单原子动作组成。
- 对单臂容易的动作，对双臂或灵巧手不一定容易。
- 视觉上困难的场景，控制轨迹可能很简单，反之亦然。

因此，很难对所有任务建立统一的全局 `easy -> hard` 排序。

### 2.3 梯度冲突与负迁移

课程学习真正需要的不是“模型是线性的”，而是前后阶段之间存在足够强的正迁移。但在异构 VLA 数据中可能出现：

\[
\cos\left(\nabla L_{\text{easy}}, \nabla L_{\text{target}}\right) < 0
\]

可能导致：

- 简单抓取形成短视策略或单峰动作分布。
- 模型依赖视觉、背景或位置捷径。
- 后续复杂任务难以覆盖早期形成的表示偏置。
- 新任务微调引起灾难性遗忘。
- 跨机器人、跨动作空间数据产生负迁移。

因此，问题不只是网络非线性，而是**难度不可标量化、目标非平稳、梯度冲突和策略诱导分布不断变化**。

## 3. 课程学习在 VLA 中仍然适用的场景

课程学习并非完全无效。它在存在明确前置关系或连续难度变量时仍然有效，例如：

- 短动作 chunk -> 长动作 chunk。
- 单技能 -> 技能组合。
- 短时序 -> 长时序。
- 无扰动 -> 轻扰动 -> 强扰动。
- 单机器人 -> 多 embodiment -> 目标机器人。
- 空间 grounding -> 动作学习 -> RL 对齐。
- 显式推理监督 -> latent reasoning -> 直接动作生成。
- 低自由度控制 -> 高自由度、双臂或灵巧操作。

更合理的做法是使用**技能依赖图**，而非强制线性顺序：

```text
定位 -> 接近 -> 接触 -> 抓取 -> 搬运 -> 放置
```

具有真实依赖关系的技能按阶段训练；无依赖或存在互补关系的技能应混合或并行训练。

## 4. 直接涉及 VLA 课程学习的论文

### 4.1 CRAFT：模态权重课程

- 论文：[CRAFT: Adapting VLA Models to Contact-rich Manipulation via Force-aware Curriculum Fine-tuning](https://arxiv.org/abs/2602.12532)
- 面向插接、擦拭、揉压等接触丰富任务。
- 训练初期压制高熵视觉和语言特征，使模型优先学习低熵但关键的力矩信号。
- 随训练推进，逐渐恢复完整视觉和语言信息。
- 本质是从力觉主导到完整多模态融合的**模态课程**。

### 4.2 LaRA-VLA：显式推理到隐式推理

- 论文：[Latent Reasoning VLA](https://arxiv.org/abs/2602.01166)
- 阶段一：文本和视觉 Chain-of-Thought 显式监督。
- 阶段二：逐步使用连续 latent 替换离散 CoT token。
- 阶段三：latent reasoning 直接条件化连续动作生成。
- 本质是**监督形式课程**，而非单纯样本难度课程。

### 4.3 Green-VLA：宏观分阶段训练

- 论文：[Green-VLA](https://arxiv.org/abs/2602.00919)
- 五阶段流程：基础 VLM、多模态物理 grounding、多 embodiment 机器人预训练、目标 embodiment 适配、RL 对齐。
- 体现从通用语义先验到物理执行和部署优化的宏观课程。

### 4.4 Simple-to-Complex Demonstrations：数据采集课程

- 论文：[Simple-to-Complex Structured Demonstrations for VLA Learning](https://arxiv.org/abs/2607.04591)
- 将复杂任务分解为有前置关系的子技能。
- 在数据采集阶段控制环境变化，并逐渐增加任务复杂度。
- 适用于双臂分拣、毛巾折叠等长时序任务。
- 强调示范组织和采集顺序，而不只是训练时重新排序数据。

### 4.5 AffordanceVLA：可供性数据课程

- 论文：[AffordanceVLA](https://arxiv.org/abs/2606.06155)
- 阶段一：指代、空间和 interaction grounding。
- 阶段二：带可供性标注的合成机器人数据联合训练。
- 阶段三：目标任务和目标机器人后训练。
- 逐步学习 `Which2Act`、`Where2Act` 和 `How2Act`。

## 5. 类似课程学习和 DAgger 的前沿训练范式

### 5.1 能力自适应课程

固定课程根据人工定义的难度推进；能力自适应课程根据模型当前状态决定训练内容：

- 选择成功率约为 20%--70% 的任务。
- 已掌握子任务自动降权。
- 尚未产生学习信号的任务先简化或分解。
- 按学习进展、信息增益或不确定性选择下一批数据。
- 当主任务成功率接近零时，临时退回相关子任务。

代表工作：

- [Agentic-VLA](https://arxiv.org/abs/2605.22896)：根据子目标掌握程度动态调整奖励和训练重点。
- [CurricuLLM](https://arxiv.org/abs/2409.18382)：使用 LLM 自动生成机器人 RL 子任务、奖励和目标分布。
- [Curriculum Learning for Vision-and-Language Navigation](https://proceedings.neurips.cc/paper/2021/hash/6f0442558302a6ededff195daf67f79b-Abstract.html)：具身视觉语言任务中的经典课程学习参照。

### 5.2 DAgger：处理策略诱导的状态分布

行为克隆在专家数据分布上训练，但部署时访问的是策略自身诱导的状态分布。DAgger 的目标是：

\[
s \sim d_{\pi_\theta}(s), \qquad a^* = \pi_{\text{expert}}(s)
\]

即让策略运行到自己容易犯错的状态，再请求专家提供正确动作并聚合数据。

DAgger 比固定课程更符合 VLA，因为它：

- 不要求预先定义全局难度。
- 数据分布随策略能力自动变化。
- 直接处理行为克隆的 covariate shift 和误差累积。
- 将训练重点放在实际部署会访问的状态上。

### 5.3 Active DAgger：只在关键状态请求专家

经典 DAgger 需要持续专家标注，成本较高。Active DAgger 的典型流程是：

1. 策略自主 rollout。
2. 使用不确定性、失败预测器或价值函数定位危险状态。
3. 只在关键状态请求人类接管。
4. 从接管点收集 recovery-to-success 轨迹。
5. 将纠正数据与旧数据 replay 后重新训练。

代表工作：

- [RECALL](https://arxiv.org/abs/2606.23617)：不确定性引导的 recovery 数据采集，并研究持续学习与遗忘。
- [CR-DAgger](https://arxiv.org/abs/2506.16685)：通过柔顺干预和残差策略学习接触任务纠正。
- [WM-DAgger](https://arxiv.org/abs/2604.11351)：通过世界模型合成 OOD 恢复状态，减少真人标注。

### 5.4 成功、失败与恢复数据联合学习

传统 VLA 训练通常只保留成功轨迹，但失败数据包含重要信息。更合理的轨迹分解包括：

- 正常成功段。
- 最终失败但早期有效的前缀。
- 发生偏离或失误的低价值动作。
- 从错误状态返回成功流形的恢复段。
- 灾难状态后的专用 reset 段。

不能把完整 recovery 轨迹统一标为正样本，否则模型可能同时模仿导致失败的动作和后续纠正动作。

代表工作：

- [RePO-VLA](https://arxiv.org/abs/2605.09410)：分别处理成功、失败和恢复轨迹，并进行价值条件策略优化。
- [FLARE](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html)：区分可直接重试的 ID 错误和需要 reset 技能的 OOD 错误。
- [VINE](https://arxiv.org/abs/2512.03913)：利用成功和失败数据学习高层计划的可行性。
- [FPC-VLA](https://arxiv.org/abs/2509.04018)：预测潜在失败并生成方向和幅度明确的纠正信号。

### 5.5 RECAP：Experience + Corrections + Offline RL

- 论文：[π*0.6: A VLA That Learns From Experience](https://arxiv.org/abs/2511.14759)
- 方法：RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）。

核心流程：

1. 聚合专家示范、策略 rollout 和在线人工纠正。
2. 使用全部数据训练任务进度或价值函数。
3. 为每个 action chunk 估计 advantage。
4. 将动作是否优于当前策略作为条件训练 VLA。
5. 部署改进后的策略并进入下一轮数据采集。

它比标准 DAgger 更一般，因为它能够使用：

- 失败和次优 rollout。
- 稀疏成功奖励。
- 人工纠正片段。
- 不同历史版本策略产生的数据。
- 有进展但不完全最优的动作。

目前它是最接近真实机器人规模化闭环学习的代表路线之一。

### 5.6 RL--SFT 交替训练

直接对大规模 VLA 全参数执行在线 RL 容易产生训练崩溃、昂贵计算和灾难性遗忘。一个稳定替代方案是：

```text
RL 探索
  -> 收集成功或高价值轨迹
  -> 与原始专家数据混合
  -> 执行稳定的监督学习
  -> 再次进入 RL 探索
```

代表工作：[iRe-VLA](https://arxiv.org/abs/2501.16664)。

- RL 阶段冻结 VLM，主要更新轻量 action head。
- SFT 阶段使用原始专家数据和新成功数据联合更新整个模型。
- RL 提供探索，SFT 提供稳定性并降低遗忘。

### 5.7 交互式 RL 后训练

行为克隆只能复制数据覆盖范围内的行为，交互式后训练可以优化最终成功率、恢复能力和动作效率。

代表工作：

- [RIPT-VLA](https://openreview.net/forum?id=oXYZHg7HiZ)：使用稀疏二值成功奖励进行交互式后训练。
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)：扩展 VLA 在线 RL，并强化策略探索。
- [πRL](https://arxiv.org/abs/2510.25889)：针对 flow-based VLA 的在线强化学习。
- [ReinFlow](https://arxiv.org/abs/2505.22094)：对 flow-matching 策略进行在线强化学习微调。
- [Z-1](https://arxiv.org/abs/2606.31846)：使用任务级 GRPO 和树状 rollout 改进 flow-based VLA。

这些方法的主要难点包括：

- 真机 rollout 和环境 reset 昂贵。
- 稀疏 reward 带来高方差。
- flow/diffusion 策略的动作概率难以准确计算。
- 对 VLM 的噪声 RL 梯度可能破坏预训练知识。

### 5.8 世界模型中的 Imagination DAgger

最新方向使用世界模型替代部分真实交互：

1. 使用真实 rollout 校准动作条件世界模型。
2. 在世界模型中生成偏离、失败和恢复状态。
3. 使用 reward/value model 对想象轨迹评分。
4. 通过 BC、优势条件训练或 RL 更新 VLA。
5. 定期使用真机数据重新校准世界模型。

代表工作：

- [VLAW](https://arxiv.org/abs/2602.12063)：真实 rollout 改进世界模型，世界模型再生成数据改进策略。
- [RISE](https://arxiv.org/abs/2602.11075)：在想象空间进行 advantage-conditioned 策略优化。
- [RehearseVLA](https://arxiv.org/abs/2509.24948)：在物理一致世界模型中进行模拟后训练。
- [WM-DAgger](https://arxiv.org/abs/2604.11351)：生成并过滤 OOD recovery 数据。

主要风险是世界模型在接触、摩擦、遮挡、物体形变和长时序预测中产生系统性偏差。未经筛选的合成恢复数据可能使策略性能下降，因此必须进行物理一致性过滤和真机校准。

## 6. 数据调度应从静态课程转向闭环选择

### 6.1 能力前沿采样

不按人工难度排序，而选择模型当前：

- 成功率中等、最可能产生学习进展的任务。
- 不确定性高但仍可恢复的状态。
- 价值估计误差大的轨迹。
- 与目标任务梯度一致或具有高信息增益的数据。

### 6.2 状态分布课程

课程不一定调整任务本身，可以调整策略访问的状态分布：

```text
专家状态
  -> 轻微动作偏差
  -> 物体位姿扰动
  -> 抓取失败和滑落
  -> 接触异常
  -> 可恢复 OOD 状态
  -> 需要专用 reset 的灾难状态
```

这类课程直接针对误差累积和部署鲁棒性，通常比任务名称层面的 easy-to-hard 更有效。

### 6.3 混合分布持续训练

新收集的纠正数据很窄，不能直接替代基础数据。训练分布应维持为：

\[
D_t =
\alpha D_{\text{base}}
+ \beta D_{\text{on-policy}}
+ \gamma D_{\text{recovery}}
+ \delta D_{\text{failure}}
+ \epsilon D_{\text{synthetic}}
\]

其中权重应根据能力、数据质量、梯度冲突和遗忘程度动态更新。RECALL 的实验表明，只使用新 recovery 数据微调容易造成明显的灾难性遗忘，而 replay 通常比单独正则化更可靠。

### 6.4 训练信号课程

相比严格改变样本顺序，更稳定的做法是保持数据混合，同时逐步调整损失权重：

\[
L = L_{\text{action}}
+ \lambda_1 L_{\text{future-latent}}
+ \lambda_2 L_{\text{affordance}}
+ \lambda_3 L_{\text{subgoal}}
+ \lambda_4 L_{\text{progress}}
+ \lambda_5 L_{\text{value}}
\]

训练初期强调 grounding、未来状态、可供性和进度；后期逐渐提高动作、价值和 RL 目标的权重。这属于监督信号或优化目标课程，不依赖脆弱的全局样本难度排序。

## 7. 与训练范式相关的模型趋势

以下内容不是课程学习本身，但会影响训练闭环的设计。

### 7.1 预测式训练而非纯动作模仿

VLA 正从单纯学习 `observation -> action` 转向联合学习：

```text
当前状态 -> 意图/子目标 -> 未来状态或可供性 -> 动作
```

代表工作：

- [World-Language-Action Model](https://arxiv.org/abs/2606.05979)
- [DIAL](https://arxiv.org/abs/2603.29844)
- [VLAFlow](https://arxiv.org/abs/2607.01586)

VLAFlow 的受控实验表明，单纯动作监督容易受异构数据影响；语言意图监督和未来 latent 对齐分别约束“要做什么”和“动作会改变什么”，二者具有互补性。

### 7.2 System-2 + System-1

- 慢速 VLM/System-2 负责语义、空间和任务推理。
- 高速 DiT/Action Expert/System-1 负责连续控制。
- 常先独立 warm-up，再进行端到端联合训练。
- 连续动作生成逐渐转向 flow matching 和 action chunking。

代表：[GR00T N1](https://arxiv.org/abs/2503.14734)、DIAL、Green-VLA。

### 7.3 结构化中间表征

常用的中间监督包括：

- Future latent：动作结果和状态变化。
- Affordance：对什么对象、在哪里、如何交互。
- Subgoal：当前高层目标。
- Progress：当前子任务完成程度。
- Latent action：跨 embodiment 的运动变化表示。

这些表征通常比显式生成完整未来 RGB 更聚焦控制，也比 VLM 直接回归低层动作更容易实现跨任务迁移。

### 7.4 异构数据金字塔

前沿 VLA 常混合：

```text
互联网视频
  -> 人类操作或第一视角视频
  -> 仿真数据
  -> 多机器人轨迹
  -> 目标机器人数据
  -> 自主部署经验
```

无动作视频可通过 latent action、光流、逆动力学或未来状态预测提供训练信号。关键不是简单扩大数据量，而是逐步实现从被动物理观察到动作-结果因果 grounding。

## 8. 推荐的综合训练方案

如果目标是构建可部署、可持续改进的 VLA，推荐组合如下。

### 阶段 A：基础能力

1. 使用多任务、多场景、多机器人数据进行行为克隆或 flow matching 预训练。
2. 加入语言意图、未来 latent、可供性和任务进度等辅助监督。
3. 使用技能依赖图组织真正具有前置关系的训练阶段。

### 阶段 B：目标平台适配

1. 使用少量目标机器人专家轨迹进行 SFT。
2. 保持基础数据 replay，防止语义和已有技能遗忘。
3. 对动作空间、控制频率、相机视角和 proprioception 进行目标平台适配。

### 阶段 C：Active DAgger

1. 部署当前策略进行 rollout。
2. 使用不确定性、failure predictor 或 progress/value model 触发专家介入。
3. 收集从策略实际访问状态开始的纠正和恢复轨迹。
4. 区分失败前缀、错误动作、恢复动作和最终成功段。

### 阶段 D：价值条件改进

1. 使用成功、失败、纠正和自主 rollout 训练进度或价值模型。
2. 为动作 chunk 计算 advantage 或质量等级。
3. 使用 advantage-conditioned imitation、offline RL 或 preference optimization 提取更优策略。
4. 将更新限制在 action expert 或 LoRA，必要时再联合更新 VLM。

### 阶段 E：想象与真机闭环

1. 使用真实失败和恢复数据校准世界模型。
2. 在世界模型中扩增 OOD、失败和恢复状态。
3. 过滤物理不一致的合成轨迹。
4. 使用少量真机 rollout 检验并重新校准。
5. 重复 Active DAgger 和价值条件训练。

## 9. 最值得研究的组合

如果只选择一条研究主线，建议：

> **Active DAgger + Recovery Data + Advantage-Conditioned Offline RL + Replay**

各部分分别解决：

| 组件 | 主要问题 |
| --- | --- |
| 能力自适应课程 | 冷启动和训练资源调度 |
| Active DAgger | 发现当前策略的数据缺口 |
| Recovery 数据 | 偏离、失败和部署鲁棒性 |
| Advantage conditioning | 利用成功、失败和次优混合数据 |
| Replay | 防止已有能力和语义知识遗忘 |
| 世界模型扩增 | 降低真实 OOD 数据采集成本 |

该组合比单独的 easy-to-hard curriculum 更符合 VLA 的非平稳、策略依赖和多维能力结构。

## 10. 实验设计建议

为了验证方法是否真正有效，应至少比较：

1. 随机混合训练。
2. 固定 easy-to-hard curriculum。
3. 反课程或 hard-first。
4. 基于成功率的能力前沿采样。
5. 标准 DAgger。
6. 不确定性触发的 Active DAgger。
7. Active DAgger + recovery segmentation。
8. Active DAgger + advantage conditioning + replay。

建议报告：

- 名义状态成功率。
- 扰动和 OOD 状态成功率。
- 失败后恢复率。
- 每单位专家干预时间的性能增益。
- 每单位真机 rollout 的性能增益。
- 旧任务遗忘程度。
- 长时序任务完成长度和平均完成时间。
- 不同数据类型的梯度相似度或干扰程度。

必须控制总示范数、专家时间、环境交互次数和训练计算量，否则不同数据闭环方法之间无法公平比较。

## 11. 基础与开源参考

- [RT-2](https://arxiv.org/abs/2307.15818)：互联网视觉语言数据与机器人动作数据联合微调。
- [OpenVLA](https://arxiv.org/abs/2406.09246)：开放的多机器人 VLA 预训练和目标任务微调基线。
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)：动作分块、连续动作、并行解码和高频双臂适配。
- [GR00T N1](https://arxiv.org/abs/2503.14734)：双系统、flow-matching action expert 和异构数据金字塔。
- [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380)：Bengio 等人在 ICML 2009 提出的经典课程学习工作。

## 12. 结论

VLA 不应简单照搬固定线性课程。真正重要的是让数据、任务和监督信号随策略能力共同演化：

```text
不是：预定义简单任务 -> 预定义困难任务

而是：
混合预训练
  -> 策略诱导状态
  -> 主动发现失败和不确定区域
  -> 收集纠正与恢复经验
  -> 价值化和重加权
  -> replay 稳定更新
  -> 再次部署
```

从这一视角看，DAgger、Active Learning、Offline RL、Recovery Learning、Continual Replay 和 World-Model Imagination 并不是相互独立的方法，而是同一个 VLA 闭环训练系统中的不同组件。

> 注：大量 2025--2026 年工作仍属于较新的预印本。不同论文的机器人平台、数据预算、任务划分和评测协议差异很大，不应直接比较其 SOTA 数值，应优先关注受控消融和真实交互成本。
