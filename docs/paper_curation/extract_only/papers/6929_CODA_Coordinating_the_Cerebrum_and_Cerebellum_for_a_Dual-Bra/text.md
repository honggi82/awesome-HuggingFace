# arXiv:2508.20096v1[cs.CV]27Aug2025

CODA: COORDINATING THE CEREBRUM AND CEREBELLUM FOR A DUAL-BRAIN COMPUTER USE AGENT WITH DECOUPLED REINFORCEMENT LEARNING.

Zeyi Sun∗1,2, Yuhang Cao∗2, Jianze Liang∗2, Qiushi Sun∗4, Ziyu Liu∗1,2 Zhixiong Zhang1,2 Yuhang Zang†2, Xiaoyi Dong2,3, Kai Chen2, Dahua Lin2,3, Jiaqi Wang†2 1Shanghai Jiao Tong University 2Shanghai AI Laboratory 3The Chinese University of Hong Kong 4The University of Hong Kong

ABSTRACT

Autonomous agents for Graphical User Interfaces (GUIs) face significant challenges in specialized domains like scientific computing, require both long-horizon planning and precise, fine-grained execution. Existing approaches suffer from a trade-off: generalist agents excel at planning but falter in execution, while specialized agents show the opposite weakness. While recent compositional frameworks attempt to bridge this gap by combining a ”planner” and an ”actor,” they are typically static and non-trainable, preventing adaptation from experience—a critical limitation given the scarcity of high-quality data in scientific domains. To address these limitations, we introduce CODA, a novel and trainable compositional framework that synergizes a generalist planner (Cerebrum) with a specialist executor (Cerebellum), trained with a dedicated two-stage training pipeline. The first stage, Specialization, employs a decoupled GRPO approach to train an expert planner for each scientific application individually, bootstrapping from a small set of initial task trajectories. The second stage, Generalization, aggregates all successful trajectories from all specialized experts. This consolidated, highquality dataset is then used to perform supervised fine-tuning (SFT) on the final planner, equipping it with the robust, cross-domain capabilities of a generalist. Evaluated on four challenging applications from the ScienceBoard benchmark, our framework significantly outperforms the baseline and establishes a new stateof-the-art (SOTA) among open-source models. Our models and code are available at https://github.com/OpenIXCLab/CODA.

1 INTRODUCTION

Autonomous agents for Graphical User Interfaces (GUIs) (Anthropic, 2024; OpenAI, 2025; Qin et al., 2025; Lin et al., 2024; Wu et al., 2024b; Hong et al., 2023) promise to automate a wide range of digital tasks (Zhou et al., 2023; Xie et al., 2024). However, their application in specialized domains such as scientific computing and engineering analysis remains highly challenging (Sun et al., 2025a). These environments pose two primary difficulties: first, their interfaces are highly complex, requiring precise and fine-grained actions; second, the problems they address are intrinsically complicated, demanding long-horizon planning to achieve effective solutions.

Effective agency for computer task automation in these domains requires both high-level planning and low-level execution as well as domain knowledge. However, current models exhibit a clear trade-off. Generalist models like Qwen2.5-VL (Bai et al., 2025) provide robust planning capabilities but often struggle with the precise grounding needed for reliable execution. Conversely, specialized agents (Wu et al., 2024b; 2025; Xie et al., 2025) like UI-Tars (Qin et al., 2025) are highly proficient in execution, yet their capacity for complex, high-level planning is more constrained.

To bridge this gap, a natural approach has been to develop compositional frameworks that explicitly decouple planning from execution, effectively pairing a generalist “cerebrum” with a specialist

† Corresponding Authors. ∗ Equal contribution

[Figure 1]

[Figure 2]

[Figure 3]

Thought: The "Clashes" option under "Structure Analysis" is the relevant tool for detecting atomic clashes...

[Figure 4]

Detect all clashes with VDW overlap ≥ 0.3Å in ChimeraX.

[Figure 5]

Planing

Action

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

VM

Cerebrum Reward Signal

[Figure 10]

Decoupled RL

[Figure 11]

[Figure 12]

Thought: It seems that I got sidetracked a bit. However, I've now noticed a small icon in the upper left corner...

[Figure 13]

Before

[Figure 14]

[Figure 15]

RL

Cerebellum

[Figure 16]

Thought: The current VDW overlap threshold is set to 0.60 Å. The next step is to

Action: click(start_box='(758,427)')

[Figure 17]

Inference

...

After

- Figure 1: Overall architecture of the proposed learnable Planner–Executor framework. Analogous to the relationship between the cerebrum and the cerebellum in the human brain, the Planner (cerebrum) generates high-level thoughts based on the history and screenshots, while the Executor (cerebellum) executes concrete GUI actions accordingly.

“cerebellum” (Agashe et al., 2024; 2025). While promising, these pioneering approaches are fundamentally limited. They are typically static and non-trainable, relying on powerful, often closedsource models as their core planner. This design introduces significant drawbacks: it compromises transparency and replicability, and most critically, prevents the agent from learning and adapting through experience.

This architectural decoupling is not merely an engineering convenience but is inspired by the functional architecture of the human brain (illustrated in Fig.1). The specialization of high-level planning (the Cerebrum) and low-level motor control (the Cerebellum) is a key aspect of human intelligence. Crucially, these structures exhibit different learning patterns: the Cerebellum, once mature, provides stable and broadly applicable motor skills that require infrequent updates Ito (2000). In contrast, the Cerebrum continuously adapts its strategies based on the nuances of new tasks and environments Demarin & Morovi´c (2014); Hallett (2005). This biological parallel motivates our core hypothesis: an effective agent should pair a stable, proficient grounding model with a dynamic planner that is specialized for different software domains through targeted, experience-driven learning.

To realize this vision, we propose a trainable compositional framework that integrates Qwen2.5VL Bai et al. (2025) as the planner (cerebrum) and UI-Tars-1.5 Qin et al. (2025) as the executor (cerebellum). Unlike prompting-based systems that rely on proprietary closed-source planners, our framework makes the planner itself learnable through interaction with software environments mediated by a static executor. Concretely, the executor provides stable, software-agnostic grounding for low-level GUI actions, while the planner, by leveraging this reliable interface, can gradually acquire domain-specific knowledge and improve its high-level planning strategies. In contrast to end-to-end training of a full agent, which requires massive amounts of specialized data and costly retraining of both perception and execution modules, our decoupled approach is substantially more data efficient: only the planner is optimized for domain adaptation, while the executor remains fixed as a generalpurpose grounder that already possesses strong generalization ability after massive pretraining for grounding purposes. This design reduces reliance on curated trajectories, lowers training cost, and ensures controllable adaptation.

To train the planner effectively under this cerebrum–cerebellum separation, we avoid the need for costly human-labeled trajectories. Instead, we leverage a judging system built from open-source models to automatically provide dense reward signals, combined with autonomous interaction with scientific software environments through the static executor. This setup enables the planner to gradually acquire domain-specific planning ability with zero human effort. Furthermore, by distributing the interaction process across multiple software environments in parallel—coordinated by a central master—we can significantly accelerate reinforcement learning. This strategy not only makes the training process more efficient but also echoes our brain-inspired design: the cerebellum-like executor delivers stable grounding, while the cerebrum-like planner continually adapts through experience.

We validate our framework on four typical scientific software applications from the ScienceBoard benchmark (Sun et al., 2025a). Experiments show that our method not only significantly improves the baseline performance (Cerebrum: Qwen2.5-32B-VL, Cerebellum: UI-Tars-1.5) but also establishes a new state-of-the-art (SOTA) among open-source models, confirming its effectiveness.

- 2 RELATED WORKS

Reinforcement Learning for LVLMs. Training for LLMs and LVLMs (Touvron et al., 2023; Grattafiori et al., 2024; Liu et al., 2023a; Bai et al., 2025; Wang et al., 2024; Xing et al., 2025; Sun et al., 2024d;c; Ding et al., 2025) has progressed from data-intensive Supervised Fine-Tuning (SFT) (Liu et al., 2023a; Wei et al., 2022) towards Reinforcement Learning (RL). Algorithms like Group Relative Policy Optimization (GRPO) (Guo et al., 2025; Shao et al., 2024) have proven effective for reasoning tasks, moving beyond earlier single-turn RLHF applications (Ouyang et al., 2022; Ziegler et al., 2019; Rafailov et al., 2023). However, applying RL to complex agentic tasks (Bai et al., 2024; Qi et al., 2024; Zhou et al., 2024; Zhai et al., 2024; Carta et al., 2023) is challenging. Prevailing methods train monolithic agents end-to-end, often requiring co-trained critic models (Schulman et al., 2015) or preference-based optimization like DPO (Rafailov et al., 2023; Putta et al., 2024; Qin et al., 2025), which problematically entangles the distinct skills of planning and execution. In contrast, our work employs a decoupled reinforcement learning strategy: the high-level planner is optimized via environmental interaction while the execution model remains fixed. We adapt GRPO by computing rewards from the final action and backpropagating the advantage exclusively through planning tokens. This targeted optimization stably enhances strategic planning, distinguishing our method from prior works that train dedicated critic models (Bai et al., 2024; Qi et al., 2024) or use filtered behavior cloning (Pan et al., 2024; Chen et al., 2020).

Computer Use Agent. Fueled by advancements in Large Vision-Language Models (LVLMs) (Touvron et al., 2023; Grattafiori et al., 2024; Liu et al., 2023a; Bai et al., 2025; Wang et al., 2024), a new generation of agents capable of operating computers via multi-modal inputs is emerging (Hu et al., 2024b; Hong et al., 2024; Cheng et al., 2024; Nguyen et al., 2024; Lin et al., 2024; Sun et al., 2024b). Whether processing structured text and code (Qi et al., 2024; Putta et al., 2024; Lai et al., 2024; Sun et al., 2024a; Nakano et al., 2021) or screenshots (Hong et al., 2023; Lin et al., 2024; Wu et al., 2024b; OpenAI, 2025), these agents face an inherent dichotomy analogous to human cognition: the tension between high-level strategic planning and precise, low-level action execution. This has motivated the development of compositional frameworks that decouple these responsibilities (Agashe et al., 2024; 2025; Liu et al., 2023b; Zhang et al., 2025; Song et al., 2025). However, a significant portion of this research relies on static, non-trainable systems that orchestrate powerful, often proprietary models (Anthropic, 2024; OpenAI, 2025; Google DeepMind, 2025; Yan et al., 2023; He et al., 2024; Zhang et al., 2024; Wang et al., 2023; Wu et al., 2024a) as their core planner. This design fundamentally prevents the agent from adapting through experience—a critical flaw for mastering novel software where interaction data is scarce. Our work charts a different course by exploring reinforcement fine-tuning of the planner. By enabling the planner to learn specialized domain knowledge through direct software interaction via a fixed execution model, our strategy achieves robust performance on unfamiliar applications.

- 3 METHOD

- 3.1 PROBLEM FORMULATION

We formally define the task of autonomous GUI operation for software workflows as a Partially Observable Markov Decision Process (POMDP). Each task is initiated with a natural language instruction g from the task space G. At each timestep t, the agent perceives the latent environment state st ∈ S through a visual observation ot ∈ Ω, consisting of a screenshot of the user interface. The agent’s behavior is governed by a policy π, instantiated by a large vision-language model, which synthesizes an action program at ∈ A. The action space A consists of precisely parameterized pyautogui scripts, where precision in arguments (e.g., coordinates) is critical for execution. The policy generates this action based on the initial instruction and the history of interactions:

#### at = π(g,(o1,a1,...,at−1,ot))

( )

- (1) (1)

[Figure 18]

[Figure 19]

- (2)

(2)

[Figure 20]

Planning Tokens

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Planner

###### Executor

[Figure 27]

User Instruction

- Figure 2: Overall training process of the proposed Planner–Executor framework. The Planner generates high-level thoughts based on the history and screenshots, while the Executor executes concrete GUI actions accordingly. During training, the rewards are calculated from a(i) and applied to p(i) to calculate loss.

This sequential process induces a state trajectory τ = (s0,s1,...,sT) with the maximum time step T. A task is considered successful if the final state sT satisfies the predefined goal condition specified in G.

- 3.2 MODEL ARCHITECTURE

To address the inherent trade-off in monolithic models, which struggle to balance long-horizon planning with precise action grounding, we propose a composite agent architecture that structures the decision-making process into a Planner-Executor framework. This design decouples the task into two distinct yet collaborative modules: a high-level Planner responsible for strategic thinking and a low-level Executor for concrete action execution.

Planner The Planner is instantiated from the Qwen2.5-VL (Bai et al., 2025) model. Its primary responsibility is to analyze the task’s progress and formulate a high-level, explicit plan pt for each step. Specifically, at each timestep t, the Planner receives the interaction history up to the previous step mt−1 = (p1,a1,...,pt−1,at−1), the current visual observation ot, and the preceding observation ot−1. The output is a structured thought, denoted as pt, which outlines the immediate objective and explicitly identifies the target UI elements for interaction. The process can be summarized as:

pt = Planner(mt−1,ot−1,ot)

Executor The Executor employs a UI-TARS-1.5 (Qin et al., 2025) model. Its role is to translate the Planner’s abstract thought pt into a precise, executable action. The Executor is provided with the same historical and visual context as the Planner (mt−1, ot−1, and ot), but is critically augmented with the Planner’s newly generated thought pt. Its output is a low-level GUI action at in the form of a ‘pyautogui’ command, such as ‘click(x, y)’. The Executor’s operation is defined as:

at = Executor(mt−1,ot−1,ot,pt)

- 3.3 TRAINING PIPELINE

Our training methodology employs a two-stage curriculum designed for initial specialization followed by broad generalization.

- 3.3.1 STAGE 1: SPECIALIZATION VIA DECOUPLED REINFORCEMENT LEARNING

The primary objective of this initial training stage is to enhance the agent’s specialized performance on individual software applications.

Through empirical analysis, we observed that the Executor exhibits strong generalization capabilities, accurately translating well-structured plans into executable actions. However, the Planner module emerged as the primary bottleneck, often struggling to formulate effective high-level strategies. To address this, we adopt a decoupled training strategy that focuses reinforcement learning exclusively on the Planner (πθ = Planner). This targeted approach allows us to refine the agent’s strategic reasoning without altering the already competent Executor.

Since the initial Planner is relatively weak and generates a limited number of successful trajectories, standard reinforcement learning methods can be inefficient. Therefore, we adapt the Group Relative Policy Optimization (GRPO) framework (Guo et al., 2025; Shao et al., 2024), which is particularly effective in such scenarios. GRPO can derive a meaningful learning signal by comparing the relative quality of different outputs, even when most of them are suboptimal.

The training process for a given task unfolds as follows. Given the current state and interaction history, the Planner first generates a group of G candidate plans. Subsequently, the fixed Executor takes each plan as input and produces a corresponding low-level action. To generate a fine-grained learning signal, we compute a reward for each plan by comparing its resulting action a(i) to the labeled positive action aT (details of labeling process are in Sec.3.4 ). Our composite reward function assesses both the correctness of the action type and the precision of its parameters:

#### r(i) = r(a(i),aT) = I type(a(i)) = type(aT) + rdist(a(i),aT), (1)

Here, the indicator function I(·) provides a binary reward for selecting the correct type of action (e.g., click vs. type). The term rdist(a(i),aT) offers a continuous reward based on the parametric similarity between the predicted and ground-truth actions, such as L1 distance for coordinates or IoU for bounding boxes. These distance-based rewards are normalized to [0,1] to ensure consistent scaling.

Once the rewards are calculated, they are used to derive a relative advantage A(i) for each plan, which is then fed into the GRPO loss function to update the Planner policy:

r(i) − mean({r(j)}Gj=1) std({r(j)}Gj=1)

A(i) =

, i = 1,··· ,G. (2)

The GRPO loss is formulated as follows:

LGRPO(πθ) = −E(s,I)∼D,{a(i)}Gi=1∼πref(·|s,I) (3)

|p(i)|

G

1 |p(i)|

1 G

min rt(i)(θ)A(i),clip(rt(i)(θ),1 − ϵ,1 + ϵ)A(i) − β DKL(i,t)(πθ∥πref) ,

t=1

i=1

πθ(p(i)|s,I) πθ

where ri,t(θ) =

(p(i)|s,I)

ref

πref(p(i)|s,I) πθ(p(i)|s,I) − 1 − log

πref(p(i)|s,I) πθ(p(i)|s,I)

and DKLi,t(πθ,πref) =

.

Consistent with the approach in (Shao et al., 2024; Guo et al., 2025), this advantage is applied across all reasoning tokens in the plan p(i), encouraging the model to develop more robust and free-form planning capabilities.

- 3.3.2 STAGE 2: GENERALIZATION VIA AGGREGATED SUPERVISED FINE-TUNING

We adopt the specialist-to-generalist paradigm proposed in Sun et al. (2025b), where a generalist model is trained by leveraging multiple specialist models as teachers. We observe that directly applying reinforcement learning across all software leads to suboptimal performance. To address this, we first train four specialist models using the methods described in Sec. 3.3.1. These specialists are then employed to generate new trajectories for each software, which serve as supervision for training a generalist model. After learning from the four software-specific teachers, the resulting generalist not only surpasses its teachers in performance, but also demonstrates stronger reasoning and reflection abilities during planning, as well as broader domain knowledge across different software.

- 3.4 AUTO EXPLORATION PIPELINE.

Auto Task Generation. We employ Qwen2.5-72B (Wang et al., 2024) as the task generator to produce high-level tasks. Specifically, a small set of real human-instructed tasks on each software is provided as input, together with the prompt shown in Fig. 5. The agent then repeatedly executes these tasks to collect a diverse set of interaction trajectories, which are subsequently filtered by a judge system to retain only trajectories with positive actions for training.

Judge System for Providing Reward Signals. Our judge system labels the positive actions aT within an agent’s trajectory when performing a task. Given a full trajectory H = {o0,a0,...,ofinal}, the judge takes the complete sequence of screenshot observations (o1,o2,...,on) as input and outputs three signals: Correctness, Redundant, and First Error Step, using the detailed prompt shown in Fig. 6. A trajectory is considered clean and successful when Correctness is True and both Redundant and First Error Step are empty. In this case, all actions a in the trajectory are labeled as aT. We present a detailed evaluation of the judge’s precision and discuss approaches for improving it in Sec. 4.2.

Distributed Virtual Machine System. Task execution is the most time-consuming step in our pipeline, so we developed a lightweight distributed system to accelerate large-scale trajectory curation. As illustrated in Fig. 3b, the system follows an HTTP-based master–client architecture: the master node manages a dynamic task queue, monitors execution progress, and aggregates results, while multiple client nodes execute tasks in parallel within isolated virtual machine environments. This design enables efficient scaling to hundreds of concurrent environments, substantially reducing the time required to collect successful trajectories and making the framework well-suited for large-scale training and evaluation.

Task Instructions Software Specific RL Specialist to Generalist

Controller

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

......

🔥

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

Tasks Trajectories

[Figure 32]

......

🔥

|VM Cluster|
|---|
| |

🔥

[Figure 33]

......

🔥

Initial Agent

[Figure 34]

[Figure 35]

[Figure 36]

......

🔥

(a) Specialist-to-Generalist strategy.

(b) Distributed VM System.

### Figure 3: Exploration pipeline for training support.

- 4 EXPERIMENTS

- 4.1 AGENT PERFORMANCE EVALUATION.

Our planner-executor approach is based on Qwen2.5VL-32B (Bai et al., 2025) serve as planner and UI-TARS-1.5-7B (Qin et al., 2025) serve as executor. We use method proposed in Sec.3.4 to generate high level tasks for each software from ScienceBoard (Sun et al., 2025a). through decoupled reinforcement learning proposed in Sec.3.3.1. During Training, the reward signal is provided by our judge system evaluated in Sec.4.2. Our training is based on OpenRLHF Hu et al. (2024a). As reported in Tab.1, our evaluation is done on four GUI centric software from ScienceBoard Sun et al. (2025a). We also report other planner-executor decoupled approaches. This first-stage reinforcement learning approach lead to significant performance gain compared to baseline.

In second stage, we use these specialist planner serve as teachers to teach a generalist planner. This new model is also initialized from Qwen2.5VL-32B and perform supervised fine-tuning on 0.77K trajectories from teacher models labeled by our judge system. As shown in tab.1, this new model surpass the performance of the ensemble of individual specialist, showing improved reasoning and planning abilities. This result demonstrates the effectiveness of our specialist-to-generalist strategy.

[Figure 37]

Qwen2.5-VL-72B

[Figure 38]

[Figure 39]

[Figure 40]

It seems I clicked the wrong menu. The task requires setting the Julian date, so we need to access the \"Time\" menu instead of the \"Navigation\" menu. Let's correct this by clicking on the \"Time\" menu, which is adjacent to \"Navigation\".\n\nHere is the corrected code to click on the \"Time\" menu:\n\n```python\npyautogui.clic k(x=213, y=73)

A pop-up notification from the Software Updater has appeared, which may obstruct our view of the \"Set Time\" dialog box. Before proceeding to enter the Julian date, it's important to dismiss or move the pop-up out of the way to fully interact with the dialog\n\nHere is the code to close the Software Updater pop-up: python\npyautogui.click(x=922, y=55)

To set the Julian date in Celestia to 2400000, we need to access the Time menu. From the screenshot, I see that the \"Time\" menu is located in the top menu bar. The next step is to click on the \"Time\" menu to find the option for setting the Julian date.\n\nHere’s the code to click on the \"Time\" menu： python\npyautogui.click(x=188, y=73)

[Figure 41]

UI-TARS-1.5-7B

[Figure 42]

[Figure 43]

[Figure 44]

Thought: Let me take a look at the interface layout. I notice there's a \"Favorites\" option in the top toolbar, which seems to be the entry point I need. I need to click on it to expand the menu and see if there are any useful options available to help me with the annotation work.\nAction: click(start_box='(554,148)')

Thought: I took a close look at the current interface and realized that I had been heading in the wrong direction before. I now need to get back on track by clicking on the \"Tools\" option in the top menu bar. This will allow me to find the functionality for adding labels and notes.\nAction: click(start_box='(505,149)')

Thought: I’ve been looking at the interface, and it’s clear that just clicking on the tools menu doesn’t provide the feature I need for adding labels and notes. It seems I need to approach this differently. There’s a Home option in the top menu bar, and I should be able to find the tools I’m looking for there..\nAction: click(start_box='(314,149)')

Figure 4: Case studies. Qwen2.5-VL-72B Bai et al. (2025) struggles with precise grounding, whereas UI-TARS-1.5 Qin et al. (2025), though specialized, fails to generalize to out-of-distribution software.

- 4.2 TOWARDS PRECISE JUDGING SYSTEM

Our reinforcement learning framework heavily relies on accurate judgments of agent trajectories to provide reliable reward signals. In this section, we present a detailed evaluation of our judge model, which demonstrates improved precision in decision making.

Settings. We conduct experiments on two sources of trajectories. (1) AgentRewardBench (L`u et al., 2025), a benchmark designed specifically for judge evaluation. (2) A trajectory dataset we collected from ScienceBoard (Sun et al., 2025a). We run Qwen2.5-VL-72B (Bai et al., 2025) on ScienceBoard tasks and extract 377 labeled trajectories, which are then used as inputs to our judge model. This setup allows us to quantitatively assess the judge’s ability to discriminate between successful and failed executions. We report Precision and Recall as our primary metrics. For voting-

Success Rate (↑) Algebra Biochem GIS Astron Overall

Metrics Model

GPT-4o (OpenAI, 2023) 3.23% 0.00% 0.00% 0.00% 0.81% Claude-3.7-Sonnet (Anthropic, 2025) 9.67% 37.93% 2.94% 6.06% 14.15% Gemini-2.0-Flash (Team et al., 2023) 6.45% 3.45% 2.94% 6.06% 4.73% GPT4o−→UGround-V1-7B (Gou et al., 2024) 0.00% 3.45% 0.00% 3.03% 1.62% GPT4o−→OS-Atlas-Pro-7B (Wu et al., 2024b) 6.25% 10.34% 0.0% 3.03% 4.92% GPT4o−→UI-TARS-72B (Qin et al., 2025) 3.23% 10.34% 5.88% 6.06% 6.38% Qwen2.5-VL-72B (Bai et al., 2025) 22.58% 27.59% 5.88% 9.09% 12.94% InternVL3-78B (Zhu et al., 2025) 6.45% 3.45% 0.00% 0.00% 2.69% UI-TARS-1.5-7B (Qin et al., 2025) 12.90% 13.79% 0.00% 6.06% 8.19%

Average@1

Qwen2.5-VL-32B (Bai et al., 2025) 10.48% 13.79% 1.47% 4.55% 7.57% UI-TARS-1.5-7B (Qin et al., 2025) 6.49% 10.24% 0.80% 3.03% 5.14%

Average@8

- CODA (Stage-1)* 13.71% 26.29% 7.72% 9.85% 14.39%

- CODA (Stage-2) 20.16% 32.23% 14.71% 17.05% 21.04%

Qwen2.5-VL-32B (Bai et al., 2025) 29.03% 31.03% 8.82% 9.09% 19.49% UI-TARS-1.5-7B (Qin et al., 2025) 19.35% 24.14% 5.88% 12.12% 15.36%

Pass@8

- CODA (Stage-1)* 41.94% 44.83% 23.53% 18.18% 32.12%

- CODA (Stage-2) 48.39% 51.72% 29.41% 30.30% 39.96%

- Table 1: Success rates of various models on ScienceBoard (Sun et al., 2025a). Proprietary models and open-sourced models based methods are highlighted with purple and green backgrounds, respectively. *Indicates specialist agents trained separately for each software with ensembled results.

- Table 2: Evaluation of different judge methods on AgentRewardBench (L`u et al., 2025) and ScienceBoard (Sun et al., 2025a).

AgentRewardBench (L`u et al., 2025) ScienceBoard (Sun et al., 2025a) Precision Recall Precision Recall

Method

Qwen2.5-VL-72B-single 64.5 83.4 41.5 80.1 72B-GUI-Judge 73.5 79.0 43.7 80.1 72B-voting@4 76.1 79.5 58.6 75.3 72B-voting@4 w/ multi-res 78.9 77.4 65.7 77.9 72B-voting@4 Ensemble 81.2 76.8 69.5 74.2

based strategies, we adopt a sampling temperature of T = 1.0 and a nucleus sampling probability of top p = 0.6 over 4 independent inference runs.

Results. As summarized in Table 2, our evaluations reveal three effective strategies for improving precision, building upon difference description fine-tuning (Sun et al., 2025b): 1. Voting. Instead of a single query, we prompt the model multiple times with high randomness (T = 1.0, top p = 0.6). A trajectory is only deemed successful if all votes agree, which significantly reduces false positives. 2. Multi-resolution inputs. Trajectories often include long sequences of high-resolution screenshots. We observe that using a mixture of resolutions across voting rounds is beneficial: low-resolution images help capture global execution dynamics, while high-resolution images aid in detecting finegrained correctness. In practice, we first apply low-resolution inputs to quickly filter out failures, thereby improving both precision and efficiency. 3. Model ensembling. In addition to the fine-tuned judge model (see Sup. A), we find that ensembling two models within the voting strategy further enhances precision.

Across both ScienceBoard Sun et al. (2025a) and AgentRewardBench L`u et al. (2025), we observe a consistent progression: the fine-tuned model (72B-GUI-Judge) primarily improves recall, while voting substantially increases precision; multi-resolution inputs add further gains, and ensembling achieves the best balance with the highest precision while maintaining competitive recall. This consistent trend across benchmarks highlights the robustness and generality of our proposed strategies. With methods proposed in . This judge system provide high quality reward signal for the planner to perform RL to improve reasoning ability and learning software domain knowledge.

- 5 CONCLUSION

We presented a trainable Planner–Executor disentangled framework for GUI agents, inspired by the division of labor between the cerebrum and cerebellum. By coupling a fixed executor (UI-Tars-1.5) with a fine-tunable planner (Qwen2.5-VL), and supporting it with a robust judging system, GRPObased exploration, and a distributed data generation pipeline, our approach effectively addresses the challenges of complex interfaces and long-horizon planning. Experiments on ScienceBoard applications demonstrate substantial improvements over strong baselines, establishing a new opensource state-of-the-art. These results highlight the importance of combining stable execution with adaptive planning, and open promising directions for extending our framework to richer multi-modal feedback, broader professional domains, and continual learning for long-term adaptability.

REFERENCES

Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s: An open agentic framework that uses computers like a human. arXiv preprint arXiv:2410.08164, 2024.

Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents. arXiv preprint arXiv:2504.00906, 2025.

Anthropic. Claude computer use. 2024. URL https://www.anthropic.com/news/ 3-5-models-and-computer-use.

Anthropic. Claude’s extended thinking. 2025. URL https://www.anthropic.com/ research/visible-extended-thinking.

Hao Bai, Yifei Zhou, Jiayi Pan, Mert Cemri, Alane Suhr, Sergey Levine, and Aviral Kumar. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning. Advances in Neural Information Processing Systems, 37:12461–12495, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Thomas Carta, Cl´ement Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and Pierre-Yves Oudeyer. Grounding large language models in interactive environments with online reinforcement learning. In International Conference on Machine Learning, pp. 3676–3713. PMLR, 2023.

Xinyue Chen, Zijian Zhou, Zheng Wang, Che Wang, Yanqiu Wu, and Keith Ross. Bail: Bestaction imitation learning for batch deep reinforcement learning. Advances in Neural Information Processing Systems, 33:18353–18363, 2020.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024.

Vida Demarin and Sandra Morovi´c. Neuroplasticity. Periodicum biologorum, 116(2):209–211, 2014.

Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Mm-ifengine: Towards multimodal instruction following. arXiv preprint arXiv:2504.07957, 2025.

Google DeepMind. Gemini 2.5 Pro Preview (03-25). https://deepmind.google/ technologies/gemini, 2025.

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Mark Hallett. Neuroplasticity and rehabilitation. Journal of rehabilitation research and development, 42(4):R17, 2005.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for GUI agents. CoRR, abs/2312.08914, 2023. doi: 10.48550/ARXIV. 2312.08914. URL https://doi.org/10.48550/arXiv.2312.08914.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14281–14290, 2024.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024a.

Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, et al. Os agents: A survey on mllm-based agents for general computing devices use, 2024b.

Masao Ito. Mechanisms of motor learning in the cerebellum. Brain research, 886(1-2):237–245, 2000.

Hanyu Lai, Xiao Liu, Iat Long Iong, Shuntian Yao, Yuxuan Chen, Pengbo Shen, Hao Yu, Hanchen Zhang, Xiaohan Zhang, Yuxiao Dong, et al. Autowebglm: A large language model-based web navigating agent. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 5295–5306, 2024.

Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for gui visual agent. arXiv preprint arXiv:2411.17465, 2024.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023a.

Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, et al. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. arXiv preprint arXiv:2308.05960, 2023b.

Xing Han L`u, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Sta´nczak, Peter Shaw, Christopher J Pal, and Siva Reddy. Agentrewardbench: Evaluating automatic evaluations of web agent trajectories. arXiv preprint arXiv:2504.08942, 2025.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, et al. Gui agents: A survey. arXiv preprint arXiv:2412.13501, 2024.

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https://doi.org/10.48550/arXiv.2303.08774.

OpenAI. Operator. 2025. URL https://openai.com/research/operator.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Jiayi Pan, Yichi Zhang, Nicholas Tomlin, Yifei Zhou, Sergey Levine, and Alane Suhr. Autonomous evaluation and refinement of digital agents. arXiv preprint arXiv:2404.06474, 2024.

Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents. arXiv preprint arXiv:2408.07199, 2024.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. arXiv preprint arXiv:2411.02337, 2024.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: pioneering automated GUI interaction with native agents. CoRR, abs/2501.12326, 2025. doi: 10.48550/ARXIV.2501.12326. URL https://doi.org/10.48550/arXiv.2501.12326.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Linxin Song, Yutong Dai, Viraj Prabhu, Jieyu Zhang, Taiwei Shi, Li Li, Junnan Li, Silvio Savarese, Zeyuan Chen, Jieyu Zhao, et al. Coact-1: Computer-using agents with coding as actions. arXiv preprint arXiv:2508.03923, 2025.

Qiushi Sun, Zhirui Chen, Fangzhi Xu, Kanzhi Cheng, Chang Ma, Zhangyue Yin, Jianing Wang, Chengcheng Han, Renyu Zhu, Shuai Yuan, et al. A survey of neural code intelligence: Paradigms, advances and beyond. arXiv preprint arXiv:2403.14734, 2024a.

Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, et al. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. arXiv preprint arXiv:2412.19723, 2024b.

Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, et al. Scienceboard: Evaluating multimodal autonomous agents in realistic scientific workflows. arXiv preprint arXiv:2505.19897, 2025a.

Zeyi Sun, Ziyang Chu, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. X-prompt: Towards universal in-context image generation in autoregressive vision language foundation models, 2024c. URL https://arxiv.org/abs/ 2412.01824.

Zeyi Sun, Tong Wu, Pan Zhang, Yuhang Zang, Xiaoyi Dong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Bootstrap3d: Improving 3d content creation with synthetic data. arXiv e-prints, pp. arXiv– 2406, 2024d.

Zeyi Sun, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Tong Wu, Dahua Lin, and Jiaqi Wang. Seagent: Self-evolving computer use agent with autonomous learning from experience. arXiv preprint arXiv:2508.04700, 2025b.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, et al. Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143, 2025.

Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement. arXiv preprint arXiv:2402.07456, 2024a.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024b.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, et al. Scaling computer-use grounding via user interface decomposition and synthesis. arXiv preprint arXiv:2505.13227, 2025.

Long Xing, Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jinsong Li, Shuangrui Ding, Weiming Zhang, Nenghai Yu, et al. Scalecap: Inference-time scalable image captioning via dual-modality debiasing. arXiv preprint arXiv:2506.19848, 2025.

An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv preprint arXiv:2311.07562, 2023.

Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems, 37:110935–110971, 2024.

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pp. 1–20, 2025.

Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. arXiv preprint arXiv:2403.02713, 2024.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.

Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training language model agents via hierarchical multi-turn rl. arXiv preprint arXiv:2402.19446, 2024.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

- A JUDGE MODEL FINE-TUNING DETAILS

Inspired by Sun et al. (2025b), we adopt a fine-tuning approach to obtain a strong judge model. We scale up the model to Qwen2.5-VL-72B Bai et al. (2025), and use a dataset comprising 4.7K labeled judgment samples. These trajectories are generated by Qwen2.5-VL and Gemini-2.0-Pro on WebArena Zhou et al. (2023), UI-TARS-1.5 Qin et al. (2025), and GPT-4o OpenAI (2023) on OSWorld Qin et al. (2025). Judgments are provided by GPT-4o and Gemini2.5-Pro Google DeepMind (2025), with detailed captions for each screenshot frame during agent execution. The judgments are further filtered, retaining only those that align with verified ground-truth results. Additionally, change description data is incorporated inspired by SEAgent Sun et al. (2025b).

Training is conducted on 32 A100 GPUs for 370 steps, using LoRA Hu et al. (2022) with a rank of 8. The resulting model, trained on OSWorld trajectories, generalizes well to AgentRewardBench L`u et al. (2025) and ScienceBoard Sun et al. (2025a). This fine-tuned model is referred to as 72B-GUIJudge in Table 2, and demonstrates improved precision on two out-of-domain benchmarks. When further ensembled with the original 72B base model, it achieves even higher precision, providing more accurate reward signals—crucial for effective reinforcement learning of the planner agent.

- B PROMPT DETAILS.

We provide detailed prompt for task generator in Fig.5 and judge system in Fig.6. Detailed prompt for planner agent is in Fig.7. Prompt we used for executor agent aligns with UI-TARS Qin et al. (2025) official code.

- C VIRTUAL MACHINE SYSTEM DETAILS

We utilized a local cluster consisting of 15 servers to collect interaction trajectories. Among these, 13 servers were equipped with AMD EPYC 7742 processors, and 2 servers were equipped with Intel i9-13900K CPUs paired with NVIDIA GeForce RTX 4090 GPUs to support software with high graphical computing demands, such as ChimeraX. Using VMware Workstation Pro, we ran 4 to 8 independent virtual machines concurrently on each server to execute tasks in parallel.

## Task Generator Prompt Template

You are now a teacher training a Computer Use Agent (CUA). This CUA is operating in a new software environment and undergoes multiple rounds of iterative training. Your task is to issue new tasks for the agent to explore and learn from, based on the feedback from its actions. You are also responsible for updating and summarizing a software usage manual to help the agent retain and apply knowledge about the software.

The agent has provided the following feedback on its recent operations within the software: {json.dumps(action_decription_list)}

Here is the software usage document you summarized in the previous round: {document} And here is the agent's performance on the previous task you provided: {json.dumps(exam)} You also have access to previously given tasks along with the screenshot captions after the agent's execution. You may use these captions and results to evaluate the agent’s current capabilities and generate new tasks, while also updating the document accordingly, given the latest screen caption and instruction with its evaluated result: {json.dumps(prev_states)} Please:

- - Analyze the agent’s performance.
- - Incorporate new knowledge from the feedback.
- - Update the software usage manual to reflect any new findings.
- - Generate a new set of **20** tasks, each targeting a feasible objective. These should:
- - Describe only **a single, high-level objective** task, DO NOT involve low level instruction like

`through`, `Use`.

- - Each task involves multiple steps to complete (typically 5–20).
- - Ensure each step is independent, without dependency between steps.
- - Reinforce areas where the agent previously struggled.
- - Be relevant to the software’s current state and functionalities. For example, if there is an empty file currently open, generate tasks involving meaningful operations on that file that align with typical software usage.
- - Be executable from the current software state. For instance, DO NOT generate a task like "save xxx.txt" if the file `xxx.txt` does not exist or hasn't been created.
- - Break down and target prior mistakes by creating focused corrective exercises.

*An examplar task is `{instruction}`. Generate 20 new different tasks that are different and as high level as this task. First, output your reasoning and analysis. Then output the updated usage document and task list in the following JSON format (use a **single JSON dictionary** to facilitate parsing): ```json {{

"software_document_new": "...", "exam_new": ["task1", "task2", ...]

}} ```

Figure 5: Detailed prompt for task generation.

## Judge Prompt Template

I am evaluating the performance of a UI agent. The images provided are **sequential keyframes** that represent the full execution trajectory of the agent when attempting to follow a command. These keyframes correspond to the instruction: **'{instruction}'**.

Please thoroughly analyze the sequence to assess the following aspects:

- 1. **Correctness** — Did the agent successfully complete the task as instructed?
- 2. **Redundant Steps** — Identify any unnecessary or repeated actions that do not contribute to the goal.
- 3. **Optimization** — Did the agent follow an efficient plan with a minimal number of steps?
- 4. **First Error Step** — If the execution is incorrect or sub-optimal, determine the index of the **first keyframe where a mistake occurred**.
- 5. **Error Analysis** — Provide a brief explanation of the mistake at that step.
- 6. **Correct Action Suggestion** — Explain what the agent **should have done instead** at the point of error.

-----------------------------------------------------------------------------------------------------------------

**Important Instructions:**

- - The agent may have made progress toward the goal, but unless the task is **fully and correctly completed**, you must set 'Correctness' to **False**.
- - Be cautious in determining success. Missing confirmation screens, skipped inputs, or wrong UI elements clicked all count as errors.
- - Carefully examine all UI changes, button interactions, text entries, and any visual feedback in the screenshots.
- - Clearly indicate **which exact steps are redundant** (starting from 1).
- ---------------------------------------------------------------------------------------------------------------You output must be extremly concise and focused, with clear emphasis on key points. If the agent fails, only provide the core reason for the first step failure, ignoring other minor issues. Keep the language clear and direct.

This is a Question-Answering Task. In addition to the action sequence, the agent's final goal is to provide a correct answer. The agent submitted the final answer: '{agent_answer}' Your evaluation must also determine if this answer is correct based on the information visible in the

**final screenshot**

----------------------------------------------------------------------------------------------------------------Once you finish the analysis, return your evaluation in the following dictionary JSON format: <captions>

- Frame1: caption of the first frame
- Frame2: caption of the second frame (max 15 frame captions) </captions> <res_dict>{

"Correctness": True/False, "Redundant": [step_num, ...], "Optimized": True/False, "First_Error_Step": step_num or None, "Error_Type": "brief description of the mistake", "Correct_Action": "what should have been done instead"

}</res_dict>

Figure 6: Detailed prompt for the judge model. Text in gray all task type based on whether finish given task or answer the question from user.

## Planner Prompt Template on ScienceBoard

You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format ``` Thought: ... Action: ... ```

## Example: Thought: To adjust the time scale, next action is to click the "Time" menu. Action: click(start_box='(116,58,197,90)')

## Note

- - Use {language} in `Thought` part.
- - Clearly specify the target element and the action in thought.
- - To enter new text, first delete the existing content or select all of it in the input box.
- - When you are asked to submit an answer, if the current screenshot provides enough information to determine it, the action should be like ```ANS your_answer```. For example: ```ANS 5```.

## User Instruction {instruction}

Figure 7: Detailed prompt for the planner agent.

