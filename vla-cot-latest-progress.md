# 从“先想后做”到训练期推理：VLA-CoT 技术进展综述

> 更新日期：2026 年 7 月 24 日  
> 范围说明：本文讨论广义的 VLA-CoT，即在 Vision-Language-Action（VLA）模型中引入文字、视觉、轨迹、潜变量或记忆等中间推理过程。它不只指 CVPR 2025 的具体模型 **CoT-VLA**。文中 2026 年工作多数仍为 arXiv 预印本，实验结果均按原论文报告，不代表独立复现结论。

## 摘要

VLA 模型试图把视觉观察、语言指令和机器人状态直接映射为动作。它继承了视觉语言模型的语义知识，却仍容易在长时程任务、空间歧义、分布外场景和执行扰动中失败。VLA-CoT 的基本思想是在动作前加入某种可学习的中间推理表示，让模型不仅回答“要做什么”，还表达“目标在哪里”“当前做到哪一步”“世界接下来会怎样变化”以及“应该沿什么路径到达目标”。

这一方向经历了四次明显转变：第一，从纯语言计划转向包含边界框、末端位置和运动方向的具身文字推理；第二，从文字推理扩展到未来图像、关键点和轨迹等视觉推理；第三，从单一模态的线性 CoT 转向文字、视觉和动作交错生成，以及带历史记忆的闭环推理；第四，从“推理时必须输出 CoT”转向“用 CoT 塑造训练表示、部署时直接预测动作”。最新证据表明，高层语言计划本身通常不是主要增益来源，真正有效的是与空间、运动和任务进度紧密相关的中间监督。

因此，VLA-CoT 的核心问题已经不再是“机器人是否应输出一段思维链”，而是：**怎样构造动作相关、可扩展且不会拖慢控制频率的内部推理接口。**

## 1. 问题定义

### 1.1 从直接策略到带推理的策略

标准 VLA 策略可以写为

\[
a_{t:t+H}=\pi_\theta(o_t,l,s_t),
\]

其中，\(o_t\) 是图像或多视角观察，\(l\) 是语言指令，\(s_t\) 是本体状态，\(a_{t:t+H}\) 是未来 \(H\) 步动作块。该形式结构简洁，但模型必须在一次映射中同时完成语义理解、空间定位、任务分解和低层控制。

带推理的 VLA 引入中间变量：

\[
r_t\sim p_\theta(r\mid o_t,l,s_t),
\qquad
a_{t:t+H}\sim p_\theta(a\mid o_t,l,s_t,r_t).
\]

这里的 \(r_t\) 不一定是自然语言。它可以是：

- 文字计划、当前子任务和任务进度；
- 目标物体边界框、末端执行器位置和运动描述；
- 未来子目标图像或视频特征；
- 二维关键点、点轨迹、三维几何表示；
- 隐空间中的未来状态或世界模型轨迹；
- 历史推理、图像或力觉压缩形成的记忆 token。

广义 VLA-CoT 研究的就是这些中间变量应如何表示、监督、生成和连接到动作策略。

### 1.2 为什么普通语言 CoT 不够

语言模型中的 CoT 主要解决符号和语义推理问题。机器人还必须处理连续几何和接触动力学。例如“把杯子放在盘子左边”至少包含四层问题：

1. 识别杯子和盘子；
2. 判断图像中的“左边”对应哪个空间区域；
3. 规划无碰撞的末端轨迹和抓取姿态；
4. 在滑动、遮挡或抓取失败后重新观察并修正。

“先抓杯子，再放到盘子左边”只解决第一层任务分解，没有直接约束后三层。因此，VLA-CoT 是否有效，取决于中间表示能否把语义意图转化为可执行的空间和运动信息。

## 2. 技术发展时间线

### 2.1 RT-2：Plan + Action 的早期证据

[RT-2](https://proceedings.mlr.press/v229/zitkovich23a.html) 将机器人动作编码成语言 token，并通过视觉语言与机器人数据联合训练，使模型继承网络规模预训练中的语义知识。其 CoT 实验在输出动作前增加一个自然语言 `Plan` 字段，例如先判断“疲劳的人需要能量饮料”，再输出抓取动作。

这项实验主要提供定性证据：VLA 可以联合生成计划和动作。但它没有系统解决具身落地、推理效率和长时程闭环控制问题，因此更适合作为 VLA-CoT 的概念起点，而不是成熟范式。

### 2.2 Embodied CoT：把语言推理落到场景和机器人状态

[Robotic Control via Embodied Chain-of-Thought Reasoning](https://arxiv.org/abs/2407.08693) 首次系统提出 Embodied CoT（ECoT）。其推理链不仅包含任务和子任务，还包括目标物体边界框、抓手位置、末端运动方向等具身元素，然后再生成离散动作 token。

论文使用基础模型自动标注机器人轨迹，避免人工逐帧书写推理。作者报告，在挑战性泛化任务上，ECoT 相对 OpenVLA 提升约 28 个百分点。更重要的消融结论是：只加入普通高层 CoT 的收益有限，视觉定位和具体运动指导才是关键。

ECoT 还提供了可解释和人机纠错接口：操作者可以查看模型判断的目标物体或子任务，并通过语言修改推理。然而，每个控制周期都自回归生成长文本，会引入明显延迟和误差累积。

### 2.3 CoT-VLA：用未来图像作为视觉思维链

CVPR 2025 的 [CoT-VLA](https://arxiv.org/abs/2503.22020) 不再只用文字描述目标，而是先生成未来子目标图像，再根据当前图像和子目标图像生成动作块：

\[
o_t,l\rightarrow \hat{o}_{t+n}\rightarrow a_{t:t+H}.
\]

未来图像能够同时表达物体位置、姿态、空间关系和部分场景变化，而且可以从无动作标签的视频中学习。CoT-VLA 在 VILA-U 基础上统一生成文字、图像和动作，使用因果注意力生成图像 token，并用全注意力和 action chunking 预测动作。论文报告，相对所选基线在真实机器人任务中提升 17%，在仿真中提升 6%。

视觉 CoT 的代价也很明确：论文每次需要生成 256 个图像 token，平均推理约慢 7 倍；低质量子目标图像还会成为新的错误源。因此，视觉“想象”能提高规划性，但不能无成本地放进高频控制环。

### 2.4 多模态和交错式 CoT

2026 年的研究开始组合文字、视觉和动作推理。[HALO](https://arxiv.org/abs/2602.21157) 依次执行文字任务推理、视觉子目标预测和动作生成，并通过 Mixture-of-Transformers（MoT）为三种过程设置专门专家。论文在其 RoboTwin 设置中报告 80.5% 平均成功率，比其 \(\pi_0\) 基线高 34.1%。

[VLA-Thinker](https://arxiv.org/abs/2603.14523) 进一步把视觉感知视作推理过程中可以动态调用的操作：模型不再只看一次初始图像，而是在推理和执行之间主动重新获取任务相关视觉证据。训练先通过 SFT 学习推理格式，再用 GRPO 根据整条推理—动作轨迹的任务结果进行对齐。论文报告 LIBERO 成功率 97.5%，但该数字只适用于其输入、训练和评测设置。

[ThinkingVLA](https://arxiv.org/abs/2606.17937) 则显式分解正向和逆向推理：

\[
\text{forward CoT}
\rightarrow \text{未来图像}
\rightarrow \text{inverse CoT}
\rightarrow \text{动作}.
\]

forward CoT 说明下一步应出现什么状态，未来图像提供目标状态，inverse CoT 再推断如何从当前状态到达目标。论文消融显示，移除 inverse CoT 带来的下降最大，提示“预测未来”本身不够，还需要从目标反推动作。

### 2.5 时序记忆和非马尔可夫推理

逐帧独立生成 CoT 会忽视历史。例如执行“依次按红、绿、蓝按钮”时，当前图像可能无法表明哪些按钮已经按过。[TRM-VLA](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_TRM-VLA_Temporal-Aware_Chain-of-Thought_Reasoning_and_Memorization_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf) 只在关键决策帧触发分层推理，并用动态记忆保存不同粒度的历史 CoT。论文报告 SIMPLER 为 72.9%，CoT token 生成量减少约 4 倍。

2026 年 7 月的 [FM-VLA](https://arxiv.org/abs/2607.18231) 把记忆扩展到力觉：它将接触力历史编码为紧凑 token，用于区分“按钮按了几次”“盘子擦了几遍”等视觉上近似相同的状态。这说明 VLA-CoT 正从视觉语言推理转向多传感器状态记忆。

## 3. 当前主要技术路线

### 3.1 显式文字 ECoT

优点是可读、可调试，能够接受人类语言纠正；缺点是 token 多、推理慢，并且自然语言未必保留精确几何。它更适合低频任务规划、语义消歧和关键节点决策，而不适合每个高频控制周期都完整生成。

### 3.2 显式视觉和轨迹 CoT

未来图像表达“应该到达哪里”，二维点轨迹表达“怎样过去”。[FoMoVLA](https://arxiv.org/abs/2607.14739) 同时学习未来特征和稀疏二维点跟踪，试图把目标状态与连续运动路径结合起来。这类紧凑轨迹监督通常比生成完整图像便宜，但可能缺少接触状态和三维深度信息。

### 3.3 潜变量和世界模型推理

另一条路线不生成可见文字或图像，而是在 latent space 中预测未来状态。优点是紧凑、速度快，可以围绕物理变化优化；缺点是难解释、难监督，也更难判断模型是否真的学到因果动力学，而不是数据相关性。

潜变量推理不能自动等同于更强推理。2026 年 7 月的[鲁棒性研究](https://arxiv.org/abs/2607.17786)比较了无推理、文字 CoT 和 latent iterative reasoning，在噪声与白盒扰动下发现所测潜迭代模型最脆弱；其输出级计划—动作一致性监控器在自适应攻击下也接近失效。这提醒我们：增加推理深度或内部循环，不代表自动获得安全性。

### 3.4 训练期 CoT、推理期直出动作

[ERVLA](https://arxiv.org/abs/2606.03784) 是这一转向的代表。作者在 978,743 条轨迹、226.3M 样本和 2592.5 小时数据上研究 ECoT，发现高层语义 CoT 单独带来的收益较小，末端移动和图像空间轨迹等动作指导更重要；显式 CoT 作为自回归动作前缀还会随推理长度产生不稳定误差。

ERVLA 使用 reasoning dropout：训练样本随机选择生成或不生成 CoT，使模型学习推理监督但不依赖测试时文本。部署时可以直接生成动作。论文报告 LIBERO-Plus 86.9%、VLABench 53.2%。

[ZR-0](https://arxiv.org/abs/2606.30552) 采用类似思想但聚焦跨机器人迁移。其 2.6B 模型在 ProcCorpus-60M 上训练，数据约含 6000 万帧、1000 小时、超过 40 万条轨迹，ECoT 标注覆盖率为 96.8%。VLM 分支接受密集 ECoT 监督，DiT 动作专家通过 flow matching 生成连续动作；推理时完全跳过 CoT。论文报告 LIBERO 平均 97.8%，单张 H100 每个动作块约 100 ms，并提供了[代码与模型入口](https://github.com/RUCKBReasoning/ZR-0)。

这类方法把 CoT 从“必须展示的答案”重新定义为“表示学习的辅助目标”。其代价是部署时可解释性下降，但通常能减少延迟和自回归错误。

### 3.5 快慢系统：低频思考，高频行动

\(\pi_0\) 本身不是典型显式 CoT 模型，但其 [MoT 与 flow-matching 动作专家](https://arxiv.org/abs/2410.24164)成为许多后续工作的架构基础：慢速 VLM 负责语义理解，快速动作专家生成连续动作块，控制频率可达 50 Hz。当前更合理的 VLA-CoT 形态往往是：

\[
\underbrace{\text{低频规划、记忆更新和语义消歧}}_{\text{System 2}}
\quad+\quad
\underbrace{\text{高频连续动作生成}}_{\text{System 1}}.
\]

两者通过 VLM hidden states、KV cache、cross-attention 或专用查询 token 连接。关键设计问题是动作专家应读取哪些推理表示，以及怎样避免读取合成标注中的捷径信息。

## 4. 代表工作对比

| 工作 | 时间/状态 | 中间推理 | 动作接口 | 推理时显式 CoT | 主要贡献与限制 |
|---|---|---|---|---|---|
| RT-2 CoT | 2023，CoRL | 文字计划 | 离散动作 token | 是 | 展示 Plan + Action 的定性可能性，系统性评估有限 |
| Embodied CoT | 2024，CoRL 2024 | 计划、框、抓手位置、运动描述 | OpenVLA 离散动作 | 是 | 强调具身落地和可纠错性，文本生成延迟较高 |
| CoT-VLA | 2025，CVPR | 未来子目标图像 | 动作 token/chunk | 是 | 可利用无动作视频，但完整图像生成约慢 7 倍 |
| HALO | 2026，arXiv | 文字 + 视觉子目标 | 专用动作专家 | 是 | MoT 统一多模态推理，训练和推理成本较高 |
| VLA-Thinker | 2026，arXiv | 动态图像调用 + 文字 | OpenVLA-OFT 路线 | 是 | SFT + 轨迹级 GRPO，在线训练成本和稳定性待验证 |
| TRM-VLA | 2026，CVPR | 关键帧 CoT + 历史记忆 | DiT 动作策略 | 关键帧生成 | 降低冗余并保持时序一致性，依赖关键帧识别质量 |
| ERVLA | 2026，arXiv | 动作相关 ECoT 监督 | flow matching | 可跳过 | 大规模研究指出显式 CoT 扩展不稳定，转向 reasoning dropout |
| ThinkingVLA | 2026，arXiv | forward CoT + 图像 + inverse CoT | 专用动作专家 | 是 | 连接未来预测与逆动力学，生成链较长 |
| ZR-0 | 2026，arXiv | 密集训练期 ECoT | DiT + flow matching | 否 | 面向跨 embodiment 表示对齐，推理快但可解释性下降 |
| FoMoVLA | 2026，arXiv | 未来特征 + 2D 点轨迹 | 连续动作策略 | 否/内部监督 | 紧凑结合目标与路径，物理接触表达仍有限 |
| FM-VLA | 2026，arXiv | 力觉历史记忆 | 动作专家 | 否 | 补足视觉不可观察的接触历史，任务范围仍较专门 |

表中的成功率不能直接排序：不同工作使用的训练数据、机器人平台、相机视角、动作空间、任务初始状态和评测次数并不一致。

## 5. 数据和训练方法

### 5.1 合成 CoT 标注

机器人轨迹通常没有逐帧推理标签，因此多数工作使用 VLM、检测器、跟踪器和机器人状态自动生成：

- 场景描述和目标物体；
- 当前任务阶段和后续子任务；
- 边界框、关键点和末端位置；
- 未来运动方向或离散动作描述。

这种方法可扩展到百万轨迹，但合成标注可能包含幻觉、视觉检测误差和语言模板偏差。若动作专家能直接从标注格式中推断动作，模型还可能学习“捷径”而非真实物理推理。reasoning dropout、attention mask 和只把部分 hidden states 传给动作专家，都是为缓解这一问题。

### 5.2 视觉监督和无动作视频

未来图像或特征可以从视频时间轴自动取得，不需要动作标签。这扩展了数据规模，但人类视频与机器人视频在视角、速度、执行器形态和可行动空间上存在差异。视频预测学到“看起来合理的未来”不等于机器人能够执行的未来。

### 5.3 SFT 与强化学习

SFT 适合教模型稳定的输出结构和基本动作能力，但不能保证生成的 CoT 与最终成功因果相关。GRPO 等强化学习方法可以根据任务成功对整条推理—动作轨迹进行优化，却面临奖励稀疏、仿真吞吐、策略崩溃以及 sim-to-real 偏差。当前较常见的流程是：大规模行为克隆预训练、带 CoT 的 SFT 冷启动，再进行任务级或轨迹级后训练。

## 6. 评测：数字为什么难以横向比较

常见基准衡量的能力不同：

- **LIBERO**：四类桌面操作套件，常用于标准化成功率评估；接近饱和后，小差异可能受实现细节影响。
- **LIBERO-Plus**：增加背景、光照和空间变化，更关注分布外泛化。
- **RoboTwin 2.0**：侧重双臂和多样化操作，任务设置与 LIBERO 不同。
- **VLABench**：强调语义理解、指令遵循和更复杂的分布外任务。
- **SIMPLER**：试图通过仿真重现真实机器人评测，但仍受仿真资产和控制接口影响。
- **真实机器人实验**：最接近部署，却常只有少量任务和几十次试验，置信区间较宽。

评估 VLA-CoT 至少应同时报告：任务成功率、任务进度、推理延迟、控制频率、CoT token 数、重规划次数、扰动恢复率和真实机器人试验数。只报告成功率会掩盖“推理提升了准确率，但速度慢到不能闭环控制”的情况。

新发布的 [IMBench](https://arxiv.org/abs/2607.15641) 包含 35 项任务和 14K 轨迹，尝试联合评估感知、物理推理、动作生成与迭代执行。其初步结果显示，VLM 往往能回答物理问题却不能生成可执行计划，VLA 则能执行熟悉动作但难以满足新约束。这正是 VLA-CoT 尚未解决的核心断层。

## 7. 当前瓶颈

### 7.1 推理是否真实有用

CoT 可能只是与正确动作相关的描述，而不是产生动作的必要原因。验证方法应包括打乱 CoT、替换错误子目标、屏蔽某类推理、干预历史记忆，并观察动作是否按预期变化，而不应只看生成文本是否“像在思考”。

### 7.2 延迟和误差累积

文字和图像自回归链越长，首 token 延迟越大，错误传播越严重。可能的解决方向包括关键帧触发、推理缓存、异步快慢系统、并行解码、紧凑轨迹 token，以及只在训练期使用 CoT。

### 7.3 三维、接触和不可观察状态

二维图像难以精确表达深度、接触力、摩擦和遮挡后的物体状态。未来推理表示需要融合多视角、三维场景、触觉、力觉和历史状态，而不是仅扩大语言模型。

### 7.4 鲁棒性和安全

显式推理便于审计，但“可读”不等于“可靠”。攻击者可以同时操纵观察和推理输出，使表面一致的计划仍产生危险动作。安全评测应覆盖分布偏移、传感器噪声、对抗扰动、计划—动作不一致、停止策略和故障恢复。

### 7.5 缺少统一、可复现的比较

目前论文常更换 backbone、训练数据、动作表示和评测协议，难以隔离 CoT 本身的贡献。理想对比应固定数据和动作专家，只改变中间推理形式，并报告相同算力与延迟预算下的结果。

## 8. 研究趋势与实践建议

未来两三年的主线可能不是让机器人生成更长的自然语言，而是以下组合：

1. **训练期密集推理监督，部署期按需触发**：平时直接动作，歧义或关键节点才生成可见 CoT。
2. **多尺度时序记忆**：同时保存任务级计划、子任务状态和短时接触历史。
3. **视觉目标与运动路径联合表示**：未来状态说明“去哪里”，轨迹或逆动力学说明“怎么去”。
4. **快慢异步架构**：慢速 VLM 更新语义上下文，快速动作专家持续读取最新观察。
5. **从模仿转向可验证后训练**：奖励不仅衡量最终成功，还衡量空间约束、碰撞、动作可执行性和恢复能力。
6. **干预式推理评测**：通过修改中间表示验证其是否真正控制行为，而不是把语言流畅度当作推理能力。

如果要在实际项目中选择路线，可按任务需求取舍：高层语义歧义和人机协作优先显式文字 ECoT；长时程视觉操作优先子目标与记忆；高频精细控制优先训练期 CoT 加 flow-matching 动作专家；接触丰富任务则应加入力觉或触觉记忆。

## 9. 结论

VLA-CoT 已从早期的“动作前写一句计划”，发展为包含具身定位、未来视觉预测、逆动力学、时序记忆、多传感器状态和训练期辅助监督的完整研究谱系。现有证据支持三个相对稳定的判断：

第一，纯高层语言 CoT 对机器人控制的帮助有限，空间和运动落地才是关键。第二，每步显式生成长 CoT 会带来延迟、误差累积和扩展不稳定，CoT 更可能成为按需启用或训练期使用的能力。第三，未来系统会趋向低频推理与高频控制分离，通过紧凑内部接口连接，而不是让同一自回归模型串行完成全部过程。

VLA-CoT 因此不是简单把 LLM 的思维链移植到机器人，而是在寻找语义理解、物理预测和连续控制之间可训练、可验证的中间层。这个中间层是否真正代表可执行的物理推理，将决定该方向能否从基准提升走向可靠的真实机器人部署。

## 参考文献

1. Brohan et al. [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://proceedings.mlr.press/v229/zitkovich23a.html), CoRL 2023.
2. Zawalski et al. [Robotic Control via Embodied Chain-of-Thought Reasoning](https://arxiv.org/abs/2407.08693), CoRL 2024.
3. Zhao et al. [CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models.html), CVPR 2025.
4. Black et al. [\(\pi_0\): A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164), 2024.
5. Shou et al. [HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning](https://arxiv.org/abs/2602.21157), arXiv 2026.
6. Wang et al. [VLA-Thinker: Boosting Vision-Language-Action Models through Thinking-with-Image Reasoning](https://arxiv.org/abs/2603.14523), arXiv 2026.
7. Li et al. [TRM-VLA: Temporal-Aware Chain-of-Thought Reasoning and Memorization for Vision-Language-Action Models](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_TRM-VLA_Temporal-Aware_Chain-of-Thought_Reasoning_and_Memorization_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf), CVPR 2026.
8. Sun et al. [Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation](https://arxiv.org/abs/2606.03784), arXiv 2026.
9. [ThinkingVLA: Interleaved Vision and Language Reasoning for Robotic Manipulation](https://arxiv.org/abs/2606.17937), arXiv 2026.
10. Li et al. [Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision](https://arxiv.org/abs/2606.30552), arXiv 2026.
11. Li et al. [FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation](https://arxiv.org/abs/2607.18231), arXiv 2026.
12. Li et al. [FoMoVLA: Bridging Visual Foresight and Motion Guidance for Vision-Language-Action Models](https://arxiv.org/abs/2607.14739), arXiv 2026.
13. Trinh et al. [Reasoning as a Double-Edged Sword: Architecture and Cross-Stage Robustness in Vision-Language-Action Models](https://arxiv.org/abs/2607.17786), arXiv 2026.
14. Maurya et al. [IMBench: A Benchmark for Intuitive Robotic Manipulation](https://arxiv.org/abs/2607.15641), RSS 2026 SemRob Workshop.
