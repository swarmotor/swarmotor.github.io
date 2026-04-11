# 多智能体强化学习 (MARL) 入门指南

> 系统介绍多智能体强化学习的基本概念、主流算法与实际应用场景。

---

## 目录

1. [什么是多智能体强化学习？](#1-什么是多智能体强化学习)
2. [核心概念](#2-核心概念)
3. [挑战与难点](#3-挑战与难点)
4. [常见算法详解](#4-常见算法详解)
   - [QMIX](#41-qmix)
   - [MAPPO](#42-mappo)
   - [MADDPG](#43-maddpg)
5. [算法对比](#5-算法对比)
6. [应用场景](#6-应用场景)
7. [学习路线与资源](#7-学习路线与资源)

---

## 1. 什么是多智能体强化学习？

**单智能体强化学习 (RL)** 研究单个 Agent 如何通过与环境交互来学习最优策略，目标是最大化累积奖励。

**多智能体强化学习 (MARL)** 则将问题扩展到多个 Agent 同时存在的环境中。每个 Agent 独立做决策，但所有 Agent 的行为共同影响环境的演化与奖励分配。

### 系统分类

根据 Agent 之间的关系，MARL 系统可以分为三类：

| 类型 | 描述 | 典型场景 |
|------|------|---------|
| **完全合作** | 所有 Agent 共享同一目标，协同完成任务 | 多机器人协作搬运、团队竞技游戏 |
| **完全竞争** | Agent 之间利益完全对立（零和博弈） | 围棋、扑克、1v1 对战 |
| **混合** | 既有合作又有竞争（如团队间对抗） | 5v5 MOBA 游戏、市场竞争 |

---

## 2. 核心概念

### 2.1 去中心化部分可观测马尔可夫决策过程 (Dec-POMDP)

MARL 的标准数学框架为 **Dec-POMDP**，由以下元素构成：

- **S** — 全局状态空间
- **A = A₁ × A₂ × … × Aₙ** — 联合动作空间
- **O = O₁ × O₂ × … × Oₙ** — 联合观测空间（每个 Agent 只能看到局部信息）
- **T(s' | s, a)** — 状态转移概率
- **R(s, a)** — 奖励函数（合作时共享，竞争时各异）
- **γ** — 折扣因子

### 2.2 集中式训练、去中心化执行 (CTDE)

这是 MARL 中最重要的范式之一：

```
训练阶段：可以访问全局状态 s、所有 Agent 的观测和动作
   ↓
执行阶段：每个 Agent 只依赖自身局部观测 oᵢ 做决策
```

CTDE 允许在训练时利用全局信息（提升策略质量），同时保持执行时的去中心化（适应现实部署约束）。

### 2.3 值函数分解

合作 MARL 的核心问题之一是：如何将**联合 Q 值 Q_tot** 分解为每个 Agent 的**独立 Q 值 Qᵢ**？

常见假设为 **IGM (Individual-Global-Max)**：
```
argmax_a Q_tot(s, a) = (argmax_{a₁} Q₁(o₁, a₁), ..., argmax_{aₙ} Qₙ(oₙ, aₙ))
```

---

## 3. 挑战与难点

### 3.1 非平稳性 (Non-stationarity)
从单个 Agent 的视角来看，环境会因其他 Agent 策略的变化而动态改变，导致学习目标不断移动。

### 3.2 信度分配 (Credit Assignment)
在合作场景中，如何将团队获得的整体奖励合理分配给每个 Agent，是 MARL 的核心难题。

### 3.3 维度爆炸 (Scalability)
联合动作空间随 Agent 数量呈指数级增长。10 个 Agent，每个有 10 种动作 → 10¹⁰ 种联合动作。

### 3.4 部分可观测性 (Partial Observability)
现实中每个 Agent 只能感知局部信息，无法访问完整的全局状态。

### 3.5 通信限制
Agent 之间的通信带宽有限甚至不允许通信，需要设计高效的隐式或显式通信协议。

---

## 4. 常见算法详解

### 4.1 QMIX

**论文**：*QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning* (Rashid et al., 2018, ICML)

#### 核心思想

QMIX 属于**值分解方法**，遵循 CTDE 范式。它引入一个**混合网络 (Mixing Network)** 将各 Agent 的局部 Q 值组合为全局 Q 值：

```
Q_tot(s, a) = f_mix(Q₁(o₁, a₁), Q₂(o₂, a₂), ..., Qₙ(oₙ, aₙ) | s)
```

**关键约束**：混合网络的权重始终非负，确保满足 IGM 条件：
```
∂Q_tot / ∂Qᵢ ≥ 0  对所有 i 成立
```

#### 网络架构

```
局部观测 oᵢ  ──→  Agent 网络 (DRQN)  ──→  Qᵢ(oᵢ, aᵢ)
                                              ↓
全局状态 s  ──→  超网络 (Hypernetwork)  ──→  混合网络权重
                                              ↓
                                         Q_tot(s, a)
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 单调性约束使训练稳定 | 表达能力受限（仅支持单调函数） |
| 执行时无需通信 | 无法处理非单调的合作关系 |
| 适合大规模合作任务 | 需要全局状态用于训练 |

---

### 4.2 MAPPO

**论文**：*The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (Yu et al., 2021)

#### 核心思想

MAPPO 是 **PPO (Proximal Policy Optimization)** 在多智能体场景下的直接扩展，属于**基于策略梯度**的方法。

**基本思路**：
- 每个 Agent 维护自己的 Actor（策略网络）
- 共享一个集中式 Critic（价值网络），输入全局状态 s
- 使用 PPO 的 Clipped Objective 更新策略

```
L_CLIP(θ) = E[ min(rₜ(θ)·Aₜ, clip(rₜ(θ), 1-ε, 1+ε)·Aₜ) ]

其中 rₜ(θ) = π_θ(aₜ|oₜ) / π_θ_old(aₜ|oₜ)
```

#### 与 IPPO 的区别

IPPO (Independent PPO) 让每个 Agent 独立运行 PPO，忽略其他 Agent 存在。

MAPPO 的 Critic 能看到全局信息，因此能产生更准确的优势估计，减少方差。

#### 工程技巧（论文关键发现）

1. **全局状态 vs. 拼接观测**：集中式 Critic 使用全局状态效果更好
2. **Agent 特定全局状态**：将 Agent ID 嵌入状态表示
3. **值函数归一化**：对 Critic 输出进行 PopArt 归一化
4. **Clip 参数**：ε = 0.2 在大多数任务上表现稳定

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单，可直接使用单智能体框架 | 需要访问全局状态（训练时） |
| 对连续动作空间支持好 | 在大规模离散动作任务上不如 QMIX |
| 样本效率较高 | 超参数较敏感 |

---

### 4.3 MADDPG

**论文**：*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (Lowe et al., 2017, NeurIPS)

#### 核心思想

MADDPG 将 **DDPG (Deep Deterministic Policy Gradient)** 扩展到多智能体场景，特别适合**混合合作-竞争**环境和**连续动作空间**。

每个 Agent i 拥有：
- **Actor** μᵢ(oᵢ | θᵢ)：只依赖自身观测，输出确定性动作
- **Critic** Qᵢ(x, a₁, ..., aₙ | φᵢ)：输入所有 Agent 的观测和动作，输出 Q 值

```
Critic 更新（最小化 TD 误差）：
L(φᵢ) = E[(Qᵢ(x, a₁,...,aₙ) - yᵢ)²]
yᵢ = rᵢ + γ·Qᵢ'(x', a₁',...,aₙ')

Actor 更新（策略梯度）：
∇_θᵢ J ≈ ∇_θᵢ μᵢ(oᵢ) · ∇_{aᵢ} Qᵢ(x, a₁,...,aₙ)|_{aᵢ=μᵢ(oᵢ)}
```

#### 关键设计

- **经验回放 (Experience Replay)**：存储 (x, x', a₁,...,aₙ, r₁,...,rₙ) 元组
- **目标网络**：每个 Actor 和 Critic 都有对应的慢更新目标网络
- **推断其他 Agent 策略**：在竞争场景中，可以通过交互数据估计对手策略

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 天然支持连续动作空间 | 扩展性差（Critic 输入随 Agent 数增长） |
| 适用于混合竞争-合作场景 | 训练不稳定（确定性策略）|
| 执行时无需其他 Agent 信息 | 需要其他 Agent 的动作用于 Critic 训练 |

---

## 5. 算法对比

| 特性 | QMIX | MAPPO | MADDPG |
|------|------|-------|--------|
| **方法类别** | 值分解 (Q-learning) | 策略梯度 (Actor-Critic) | 确定性策略梯度 |
| **动作空间** | 离散 | 离散 / 连续 | 连续 |
| **任务类型** | 纯合作 | 合作为主 | 合作 + 竞争 |
| **通信需求（执行）** | 无 | 无 | 无 |
| **可扩展性** | 优 | 良 | 差 |
| **样本效率** | 良 | 良 | 高（Off-policy）|
| **典型基准** | StarCraft II (SMAC) | StarCraft II, MPE | MPE, 粒子环境 |

---

## 6. 应用场景

### 6.1 游戏与竞技 AI
- **StarCraft II**：DeepMind AlphaStar 使用多智能体自博弈训练
- **Dota 2**：OpenAI Five，5 个 Agent 协同作战
- **德州扑克**：多方博弈下的策略学习

### 6.2 机器人协作
- **多无人机编队**：协同搜索、侦察、物流配送
- **仓储机器人**：亚马逊 Kiva 系统的路径规划与协作
- **多机械臂操控**：协同完成复杂装配任务

### 6.3 智能交通系统
- **自动驾驶**：多车辆协同避障、变道、交叉路口通行
- **信号灯控制**：城市交通信号的全局协调优化
- **车队管理**：物流车队的动态路线规划

### 6.4 能源与网络
- **智能电网**：分布式能源节点的负载均衡
- **通信网络**：多基站资源调度与干扰协调
- **数据中心**：冷却系统与计算资源联合优化

### 6.5 经济与金融
- **市场仿真**：多智能体建模市场微观结构
- **拍卖机制设计**：研究策略性竞标行为
- **供应链优化**：多方利益博弈下的库存与物流决策

---

## 7. 学习路线与资源

### 7.1 推荐学习路线

```
阶段一：单智能体 RL 基础（2-4 周）
  └─ DQN → Policy Gradient → PPO / SAC

阶段二：MARL 理论基础（2-3 周）
  └─ 博弈论基础 → Dec-POMDP → CTDE 范式

阶段三：算法实践（3-4 周）
  └─ QMIX（合作） → MADDPG（竞争） → MAPPO（通用）

阶段四：前沿方向（持续学习）
  └─ 通信学习 → 平均场 MARL → 离线 MARL → 大规模 MARL
```

### 7.2 实验平台

| 平台 | 适用算法 | 特点 |
|------|---------|------|
| **SMAC** (StarCraft Multi-Agent Challenge) | QMIX, MAPPO | 合作任务标准基准 |
| **MPE** (Multi-Particle Environments) | MADDPG, MAPPO | 轻量级、混合场景 |
| **Google Research Football** | 各类 MARL | 足球仿真 |
| **PettingZoo** | 通用 | OpenAI Gym 风格，多场景 |
| **MARLlib** | 通用 | 统一实现多种 MARL 算法 |

### 7.3 关键论文

1. **VDN** (Sunehag et al., 2017) — 值分解开山之作
2. **QMIX** (Rashid et al., 2018, ICML) — 单调值分解
3. **QPLEX** (Wang et al., 2020, ICLR) — 突破 QMIX 单调限制
4. **MADDPG** (Lowe et al., 2017, NeurIPS) — CTDE Actor-Critic
5. **MAPPO** (Yu et al., 2021) — PPO 的 MARL 有效性研究
6. **HAPPO** (Kuba et al., 2021) — 异构 Agent 策略梯度
7. **QMIX → SMAC** (Hu et al., 2021) — SMAC 上的深入分析

### 7.4 代码库推荐

- **PyMARL** — QMIX 官方实现：`github.com/oxwhirl/pymarl`
- **MAPPO** — 官方实现：`github.com/marlbenchmark/on-policy`
- **MARLlib** — 统一框架：`github.com/Replicable-MARL/MARLlib`
- **EPyMARL** — 扩展版 PyMARL，包含更多算法

---

*本文档版本：v1.0 | 日期：2026-04-11*
