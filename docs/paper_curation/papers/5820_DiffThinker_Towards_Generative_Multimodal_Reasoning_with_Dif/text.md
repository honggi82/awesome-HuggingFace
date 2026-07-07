## DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models

# arXiv:2512.24165v1[cs.CV]30Dec2025

#### Zefeng He12 Xiaoye Qu1 Yafu Li13 Tong Zhu1 Siyuan Huang14 Yu Cheng3

###### Project Page: https://diffthinker-project.github.io

[Figure 1]

(a) Overall performance. (b) Visualizations on VSP-Super, Sudoku and Jigsaw.

Figure 1. (a) Quantitative results across seven tasks. (b) DiffThinker produces solution images directly, whereas baseline results are post-processed visualizations of textual outputs with errors highlighted. By reformulating reasoning as a native image-to-image generative task, DiffThinker achieves superior logical consistency and spatial precision in complex long-horizon, vision-centric reasoning tasks.

### Abstract

While recent Multimodal Large Language Models (MLLMs) have attained significant strides in multimodal reasoning, their reasoning processes remain predominantly text-centric, leading to suboptimal performance in complex long-horizon, vision-centric tasks. In this paper, we establish a novel Generative Multimodal Reasoning paradigm and introduce DiffThinker, a diffusionbased reasoning framework. Conceptually, DiffThinker reformulates multimodal reasoning as a native generative image-to-image task, achieving superior logical consistency and spatial precision in vision-centric tasks. We perform a systematic comparison between DiffThinker and MLLMs, providing the first in-depth investigation into the intrinsic characteristics of this paradigm, reveal-

1Shanghai AI Laboratory 2Nanjing University 3The Chinese University of Hong Kong 4 Shanghai Jiao Tong University. Correspondence to: Xiaoye Qu <quxiaoye@pjlab.org.cn>, Yu Cheng <chengyu@cse.cuhk.edu.hk>.

Preprint. January 1, 2026.

ing four core properties: efficiency, controllability, native parallelism, and collaboration. Extensive experiments across four domains (sequential planning, combinatorial optimization, constraint satisfaction, and spatial configuration) demonstrate that DiffThinker significantly outperforms leading closed source models including GPT-5 (+314.2%) and Gemini-3-Flash (+111.6%), as well as the fine-tuned Qwen3-VL-32B baseline (+39.0%), highlighting generative multimodal reasoning as a promising approach for vision-centric reasoning.

### 1. Introduction

In recent years, Multimodal Large Language Models (MLLMs) (Google, 2025a; OpenAI, 2025a; Bai et al., 2025; Comanici et al., 2025) have achieved remarkable progress in multimodal understanding. The introduction of Chain-ofThought (CoT) empowers these models with complex reasoning capabilities. Furthermore, Reinforcement Learning with Verifiable Reward (Shao et al., 2024; Guo et al., 2025a; Zhang et al., 2025c) has substantially enhanced the reason-

ing capabilities of MLLMs (Wang et al., 2025a). Building upon these foundations, the emerging paradigm of “Thinking with Image” (OpenAI, 2025b; Zheng et al., 2025; Wang et al., 2025b; Su et al., 2025b) enables MLLMs to interact with multimodal inputs iteratively, further pushing the boundaries of multimodal reasoning.

Despite these advances, current MLLMs primarily rely on lengthy CoT for reasoning, resulting in uncontrollable generation and prohibitive latency (Sui et al., 2025; Qu et al., 2025). This inefficiency is further intensified by the multiturn interactions inherent in the “Thinking with Image” paradigm. More importantly, these reasoning processes stay predominantly text-centric and struggle to track the changing state of visual information over long sequences, posing significant challenges for complex long-horizon, visioncentric tasks (Wu et al., 2024; Ivanitskiy et al., 2023).

To address these limitations, in this paper, we introduce DiffThinker, and establish Generative Multimodal Reasoning as a novel paradigm that shifts the reasoning from symbolic space to visual space. Unlike MLLMs that typically conceptualize reasoning as a multimodal-to-text mapping, we propose to model multimodal reasoning directly as a generative image-to-image task with diffusion models. We conduct a systematic comparison between DiffThinker and MLLMs across a diverse set of challenging tasks, and provide the first in-depth investigation into the intrinsic characteristics of generative multimodal reasoning, revealing four core properties of DiffThinker: ① Efficient Reasoning: It demonstrates superior efficiency in both training and inference, as well as higher accuracy, significantly outperforming RL-based MLLMs. ② Controllable Reasoning: It provides stable and controllable inference costs, contrasting with the variable length CoT in MLLMs. ③ Native Parallel Reasoning: It inherently explores multiple candidate solutions in parallel, progressively pruning invalid paths. ④ Collaborative Reasoning: It acts as a partner with MLLMs, achieve performance surpassing either model alone.

To comprehensively evaluate the performance of DiffThinker, we conduct experiments across seven tasks in four domains including sequential planning, combinatorial optimization, constraint satisfaction, and spatial configuration. The results demonstrate that DiffThinker significantly outperforms state-of-the-art MLLMs, including GPT5 (+314.2%), Gemini-3-Flash (+111.6%), and the Qwen3VL-32B baseline fine-tuned on identical datasets (+39.0%). Furthermore, we extend DiffThinker to the image-to-video generation paradigm for multimodal reasoning, and propose the DiffThinker-Video variant, demonstrating that video generation also exhibits inherent multimodal reasoning capabilities, and further highlight the effectiveness and efficiency of DiffThinker through comparative evaluations.

In summary, our contributions are threefold:

- • We propose DiffThinker and establish Generative Multimodal Reasoning as a novel paradigm, reformulating multimodal reasoning from text-centric symbolic mapping to a native image-to-image generative process.
- • We perform a systematic comparison between DiffThinker and MLLMs across multiple domains and conduct the first investigation into the intrinsic characteristics of this generative multimodal reasoning paradigm, revealing four core properties: efficiency, controllability, native parallelism, and collaboration.
- • Extensive experiments on seven tasks demonstrate DiffThinker significantly outperforms SOTA MLLMs including GPT-5 (+314.2%) and Gemini-3-Flash (+111.6%), revealing generative multimodal reasoning as a promising approach for vision-centric reasoning.

### 2. Related Works

2.1. Multimodal Reasoning Reinforcement Learning with Verifiable Reward (Guo et al.,

- 2025a; Shao et al., 2024) has significantly enhanced LLM reasoning, and is rapidly extending to MLLMs (Huang et al.,
- 2025b; Shen et al., 2025b; Liu et al., 2025b; Huang et al., 2025a; He et al., 2025b; Wang et al., 2025a; Shen et al., 2025a). However, existing paradigms remain predominantly text-centric, which hinders performance in vision-centric tasks. Advancing this frontier, the paradigm of “Thinking with Image” (OpenAI, 2025b) introduces a mechanism for models to engage in multi-turn visual interactions during the reasoning process. While earlier approaches (Su et al., 2025a; Zheng et al., 2025; Wang et al., 2025b; Hong et al., 2025; Zhang et al., 2025d; Lai et al., 2025) relied on tool calls or code execution for image manipulation, recent works (Yang et al., 2025b; Xu et al., 2025; Du et al., 2025; Zhang et al., 2025b; Wang et al., 2025c; Chen et al., 2025; Qin et al., 2025; Gu et al., 2025) have shifted toward generating native images or latent visual tokens. Nevertheless, the underlying architectures of these methods remain rooted in autoregressive MLLMs, leading to limited performance in complex long-horizon, vision-centric tasks.

Building upon the success of “Thinking with Image,” the “Thinking with Video” paradigm enhances reasoning by enabling models to interact with video content through multiturn tool invocation (Zhang et al., 2025a; He et al., 2025a; Yan et al., 2025; Xie et al., 2025). This concept has recently advanced to performing multimodal reasoning directly through video generation (Wiedemer et al., 2025; Tong et al., 2025; Yang et al., 2025a; Luo et al., 2025; Liu et al., 2025a; Guo et al., 2025b; Wu et al., 2025b). However, these studies predominantly focus on benchmarking closed source models (Google, 2025b; OpenAI, 2025c) with undis-

[Figure 2]

- Figure 2. Overview of different multimodal reasoning paradigms. (a) Standard MLLMs map inputs directly to symbolic solutions. (e.g., ‘R’ and ‘D’ representing ‘Right’ and ‘Down’ actions) (b) “Thinking with Images” MLLMs interact with multimodal inputs through iterative tool calls. (c) DiffThinker reformulates multimodal reasoning as a direct generative image-to-image task, where solutions are produced in visual space and then parsed to symbolic solutions to ensure a fair comparison.

closed internal reasoning mechanisms. Furthermore, video generation itself entails prohibitive computational costs. Diverging from this, DiffThinker establishes image generation as a more efficient paradigm.

#### 2.2. Diffusion Models

Diffusion models have emerged as the dominant framework for generative modeling. Early research (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020; Song et al., 2020; Ho & Salimans, 2022) laid the theoretical foundations of this paradigm. Subsequently, flow-based methodologies (Lipman et al., 2022; Liu et al., 2022; Albergo & Vanden-Eijnden, 2022) have further advanced the field. The integration of latent diffusion models (Rombach et al., 2022), diffusion transformers (Peebles & Xie, 2023), and multimodal diffusion transformers (Esser et al., 2024) has established the current mainstream for generative modeling, paving the way for diverse downstream applications.

While one prominent direction of research concentrates on high-fidelity image (Rombach et al., 2022; Ramesh et al., 2022; Saharia et al., 2022) and video (Ho et al., 2022; Wan et al., 2025; Brooks et al., 2024) generation, other applications extend to specialized tasks (Avdeyev et al., 2023; Ubukata et al., 2024; Pogodzinski et al., 2025; Graikos et al., 2022; Li et al., 2024; 2023), such as Sudoku (Wewer et al., 2025), geometry (Goren et al., 2025), and the Traveling Salesperson Problem (Sun & Yang, 2023). Diverging from these specialized approaches, we focus on multimodal reasoning and introduce DiffThinker, establishing Generative Multimodal Reasoning as a novel paradigm. Also, unlike prior methods that typically require task-specific architectures and training from scratch, DiffThinker enables rapid adaptation to diverse multimodal reasoning tasks by formu-

lating them as a unified generative process in visual space.

### 3. Generative Multimodal Reasoning

#### 3.1. Problem Reformulation

In this work, we introduce DiffThinker, a generative multimodal reasoner that innovatively reformulates multimodal reasoning as an image-to-image task, as illustrated in Figure 2. To clarify the paradigm shift, we define and formalize three distinct reasoning paradigms.

Standard MLLMs: Multimodal-to-Text. Standard MLLMs model the reasoning process as a sequential mapping in the symbolic space. Given a visual input x ∈ X and a textual instruction c ∈ T , the process is defined as:

fStd(x,c) → z → y, (1)

where z represents the textual reasoning trace (e.g., Chainof-Thought) and y ∈ Y is the final solution. Despite their success, the reasoning process remains text-centric, often leading to suboptimal performance in vision-centric tasks.

“Thinking with Image” MLLMs: Iterative Interaction. This paradigm enhances reasoning by enabling models to interact with multimodal inputs to generate intermediate results via tool calls. The process is formulated as an interleaved sequence of reasoning, tool call, and observation:

fTwI(x,c) → {(z1,t1,o1),...,(zn,tn,on)} → y, (2)

where zi denotes the i-th reasoning step, ti represents the tool call, oi is the corresponding intermediate visual observation, and y ∈ Y signifies the final solution. Although this paradigm incorporates essential visual feedback, its inherent

[Figure 3]

- Figure 3. Main tasks. Each column represents a specific task. The first row displays the image input. The second row shows the results generated by DiffThinker. The third row presents the outputs from the MLLM baseline.

reliance on iterative multi-turn loops and the associated computational overhead pose significant challenges for scaling to complex long-horizon, vision-centric tasks.

DiffThinker: Multimodal-to-Image. Unlike MLLMs, which primarily reason within symbolic space, DiffThinker shifts the reasoning process into visual space through a direct multimodal-to-image transformation. In this generative multimodal reasoning paradigm, the model functions as a generator G that directly produces a solution image xsol from the visual input x and the textual instruction c:

G(x,c) → xsol ∈ X, (3)

where xsol visually encapsulates the reasoning trajectory and solution. To facilitate comparison with the symbolic ground-truth, we introduce a parsing function Ψ : X → Y to map the solution image to symbolic space:

##### yparsed = Ψ(xsol). (4)

Rather than relying on MLLMs to judge whether a solution image conforms to the textual ground-truth, our parsing mechanism ensures a fair comparison across different paradigms and precludes potential answer leakage.

#### 3.2. Flow Matching

DiffThinker is implemented based on Qwen-ImageEdit (Wu et al., 2025a). Mathematically, it employs Flow Matching (Lipman et al., 2022; Liu et al., 2022; Albergo & Vanden-Eijnden, 2022) as the theoretical framework to approximate the velocity field that transforms noise into the data distribution, ensuring stable learning dynamics via Ordinary Differential Equations (ODEs). Architecturally, the model leverages a Multimodal Diffusion Transformer

(MMDiT) (Esser et al., 2024) to capture intricate crossmodal dependencies. For efficiency, these generative processes are performed within the latent space of a Variational Autoencoder (VAE) (Kingma & Welling, 2013).

Training. Formally, let y denote the ground-truth image. The data latent x0 is obtained by encoding y through the VAE encoder E, i.e., x0 = E(y). A random noise vector x1 is sampled from the standard multivariate normal distribution, x1 ∼ N(0,I). To incorporate multimodal task constraints, the conditioning latent h is derived from the MLLM ϕ given the user instruction S (comprising text and visual inputs), such that h = ϕ(S). During training, a timestep t is sampled from a logit-normal distribution with t ∈ [0,1]. The intermediate latent variable xt is constructed via linear interpolation between the data x0 and noise x1:

xt = tx0 + (1 − t)x1. (5)

Consequently, the target velocity field vt driving the flow from noise to data is defined as:

dxt dt

vt =

= x0 − x1. (6)

The MMDiT-based vector field vθ is trained to predict this target velocity vt. The training objective is formulated as the mean squared error (MSE):

0,x1 ∥vθ(xt,t,h) − (x0 − x1)∥2 . (7)

LFM = Et,x

Inference. During inference, DiffThinker performs reasoning by solving the ODE defined by the learned velocity field dxt = vθ(xt,t,h)dt. From initial noise xt=0 = x1, the model numerically integrates the flow to recover the

Table 1. Comprehensive Results across All Tasks. We evaluate models across a total of four domains including Sequential Planning (VSP, VSP-Super, and Maze), Combinatorial Optimization (TSP), Constraint Satisfaction (Sudoku), and Spatial Configuration (Jigsaw and VisPuzzle). Evaluation is conducted on varying difficulty levels, defined by grid size for Sequential Planning and Jigsaw, number of cities for TSP, and number of given clues for Sudoku. “N/A” denotes vanilla models without training. The Avg column represents the grand mean calculated from individual task averages.

|VSP<br><br>|VSP-Super|Maze|TSP<br><br>|Sudoku<br><br>|Jigsaw|
|---|---|---|---|---|---|
|3 4 5 6 7 8|16 32<br><br>|8 16 32<br><br>|12 15 18<br><br>|45 40 35|2 3 4|

Model Setting

VisP. Avg Closed Source MLLMs

Gemini-3-Flash N/A 100 100 100 99 83 98 52 3 0 0 0 25 9 4 69 29 3 71 16 0 89.5 41.3 GPT-5 N/A 99 70 67 43 36 29 3 0 2 0 0 0 0 0 2 0 0 30 0 0 78.0 21.1

###### Open Source MLLMs

N/A 64 46 33 21 12 21 1 0 0 0 0 0 0 0 0 0 0 7 0 0 28.0 9.1

Qwen3-VL-8B

SFT 99 96 98 96 92 86 61 8 53 37 0 59 60 43 30 17 2 95 56 9 78.8 51.6 GRPO 91 70 70 31 34 24 0 0 0 0 0 0 0 0 0 0 0 6 0 0 28.0 11.9

N/A 75 51 47 25 23 26 0 0 0 0 0 0 0 0 0 0 0 9 0 0 29.5 10.5 SFT 96 99 98 100 99 90 85 21 91 57 3 69 59 52 32 22 2 97 72 28 95.8 62.9

Qwen3-VL-32B

GRPO 99 90 95 69 73 58 1 0 0 0 0 0 0 0 3 1 0 64 4 0 83.0 26.9

Generative Multimodal Reasoners

Qwen-Image-Edit-2509 N/A 33 36 22 12 11 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7.5 4.0 DiffThinker (Ours) Flow Matching 99 100 98 99 100 100 96 83 100 97 56 74 62 58 98 95 57 99 97 80 98.3 87.4

Qwen-Image-Edit-2511 N/A 50 55 44 16 23 23 0 0 0 0 0 0 0 0 0 0 0 0 0 0 11.5 6.7 DiffThinker++ (Ours) Flow Matching 100 100 100 98 100 100 99 80 100 100 65 76 72 59 97 94 55 99 98 80 98.8 88.5

solution latent xt=1 ≈ x0. Implementing a first-order Euler solver with a step size ∆t = 1/T, the update rule is:

xt+∆t = xt + ∆t · vθ(xt,t,h). (8)

After T steps, the final latent xt=1 (which approximates the data distribution) is decoded back to pixel space via the VAE decoder to yield the visual solution: ysol = D(xt=1).

#### 3.3. Task Formulation

To systematically verify the efficacy of DiffThinker within the proposed generative reasoning paradigm, we select tasks based on three perspectives. First, we target complex longhorizon, vision-centric tasks that fundamentally rely on visual perception. Second, we prioritize tasks offering controllable and scalable difficulty levels, which facilitates a precise exploration of the model’s capability boundaries. Third, we specifically select tasks featuring high structural parseability, such as grid-based configurations. Since the evaluation of DiffThinker involves parsing generated visual solutions into symbolic formats, this criterion ensures an objective assessment against ground-truth labels. Accordingly, our tasks contain five distinct classes as detailed below.

Visual Spatial Planning (VSP) (Wu et al., 2024) and VSPSuper. VSP evaluates perception and reasoning capabilities in spatial planning scenarios. We focus on its FrozenLake subset due to its parseability. Moreover, we introduce VSPSuper, which expands the environment scale. As illustrated in the first column of Figure 3, the model must navigate a grid-based frozen lake while avoiding holes. DiffThinker generates a continuous visual trajectory rendered as a red line. Conversely, MLLMs produce text-based action plans. We formalize these challenges as sequential planning tasks.

Maze (Ivanitskiy et al., 2023). This task involves longer routes than VSP, increasing navigation complexity. As illustrated in the second column of Figure 3, the model must identify a path avoiding walls between cells. DiffThinker renders a trajectory from the yellow start to the blue target. Conversely, MLLMs output an action plan via a series of text tokens. We categorize this as a sequential planning task.

Traveling Salesperson Problem (TSP) (J¨unger et al., 1995). This task requires solving the Traveling Salesperson Problem on a 2D plane, aiming to identify the shortest path visiting every city. As depicted in the third column of Figure 3, the problem is visualized by a yellow start dot and blue city dots. DiffThinker generates a geometric path connecting all nodes into a closed loop. In contrast, MLLMs provide numerical coordinates to represent the order. This is classified as a combinatorial optimization problem.

Sudoku. In this task, the model must fill in missing digits while adhering to Sudoku constraints. As shown in the fourth column of Figure 3, DiffThinker generates a completed grid with all empty cells populated. Conversely, MLLMs provide a text-based numerical sequence. This challenge is classified as a constraint satisfaction task.

Jigsaw and VisPuzzle (Gu et al., 2025). The Jigsaw task centers on spatial configuration and visual perception. As illustrated in the final column of Figure 3, the input consists of shuffled patches, each numerically labeled to facilitate automated parsing. DiffThinker reconstructs these patches into a globally consistent image. In contrast, MLLMs produce a sequence of indices representing the restoration order. We also introduce VisPuzzle (Gu et al., 2025), which serves as a simplified benchmark for puzzle reconstruction. These challenges are categorized as spatial configuration tasks.

[Figure 4]

- Figure 4. DiffThinker as a native parallel reasoner. Visualization of the native parallel reasoning process in DiffThinker. The model explores multiple candidate paths simultaneously in the early stages and iteratively refines them into a single valid trajectory.

| |
|---|

(a) Training Time (h)

| |
|---|

| |
|---|

(b) Inference Time (s)

- Figure 5. Computational efficiency analysis. (a) compares training duration in hours, and (b) shows inference latency in seconds.

performance of MLLMs decays rapidly as task complexity scales, whereas DiffThinker maintains high accuracy through generative reasoning. In spatial configuration tasks including Jigsaw and VisPuzzle, the model achieves nearperfect performance, while similarly delivering exceptional results in combinatorial optimization (TSP) and constraint satisfaction (Sudoku). These results underscore that our generative multimodal reasoning paradigm provides a more robust foundation for multimodal reasoning than that of traditional MLLMs in long-horizon, vision-centric tasks.

#### 4.2. Discussion and Observation

### 4. Experiments

Experimental Setup. DiffThinker is built upon QwenImage-Edit-2509 (Wu et al., 2025a), utilizing a 20B MMDiT (Esser et al., 2024). Additionally, we implement DiffThinker++ based on the updated Qwen-Image-Edit2511 for main results (Table 1), whereas all subsequent analysis and ablation studies are conducted using DiffThinker. Following previous works (Xu et al., 2025; Wu et al., 2025b; Yang et al., 2025a), we train independent models for VSP/VSP-Super, Maze, TSP, Sudoku, and Jigsaw, respectively. We also fine-tune Qwen3-VL baselines on identical datasets for a direct comparison. VisPuzzle serves as an out-of-distribution task for puzzle reconstruction. Evaluation is conducted on varying difficulty levels, as shown in Table 1. Details are provided in Appendix A.1.

4.1. Main Results

DiffThinker as an Extraordinary Multimodal Reasoner. As illustrated in Table 1, DiffThinker achieves state-of-theart performance across seven challenging tasks in four domains. Specifically, our approach drastically surpasses GPT-

###### 5 (+314.2%), Gemini-3-Flash (+111.6%), and the fine-tuned Qwen3-VL-32B (+39.0%) with fewer parameters.

Across all evaluated domains, DiffThinker demonstrates a clear advantage over traditional MLLMs. In sequential planning tasks such as VSP, VSP-Super, and Maze, the

DiffThinker as an Efficient Reasoner. To quantitatively assess the computational overhead of DiffThinker relative to standard MLLMs, we conduct experiments to measure both training and inference durations on a cluster of eight NVIDIA H200 GPUs. We report training durations of VSP/VSP-Super, and the average inference latency of VSPSuper level-16 per reasoning instance.

As illustrated in Figure 5(a), DiffThinker maintains a highly competitive training efficiency. Its training duration is nearly identical to Qwen3-VL-32B (SFT) baseline and is substantially lower than the overhead of GRPO (Shao et al., 2024), a reinforcement learning paradigm currently widely adopted for multimodal reasoning. Regarding inference speed, as illustrated in Figure 5(b), DiffThinker exhibits a highly competitive latency of 1.1s, which is comparable to Qwen3-VL8B (SFT) baseline (1.0s) and faster than Qwen3-VL-32B (SFT) model (1.4s). This result underscores the inherent inference efficiency of our generative reasoning paradigm.

DiffThinker as a Controllable Reasoner. DiffThinker establishes a controllable reasoning paradigm by reformulating tasks into a fixed-step generative process. By employing an Euler solver with a predefined number of steps, the model ensures a deterministic computational budget which is invariant to the task’s logical complexity. In contrast, MLLMs are plagued by unpredictable inference durations. Their autoregressive nature often leads to fluctuating latency caused by verbose Chain-of-Thought or repetitive output collapse, resulting in significantly longer average inference

[Figure 5]

[Figure 6]

(a) Collaborative Pipeline (b) Accuracy on Jigsaw level-4

- Figure 6. DiffThinker as a collaborative partner. (a) The partnership framework where DiffThinker generates N candidates for MLLM verification. (b) Performance on Jigsaw level-4, demonstrating that collaboration surpasses individual models and accuracy further scales with the number of candidates N.

times, as shown in Figure 5(b). Moreover, unlike MLLMs where imposing token limits risks premature truncation, the controllable generation of DiffThinker guarantees both execution stability and the derivation of reliable solutions.

DiffThinker as a Native Parallel Reasoner. Unlike MLLMs, which execute reasoning sequentially and often require explicit reflection or backtracking to rectify early errors, DiffThinker possesses an inherent capacity for native parallel reasoning. To visualize the progressive reasoning process, we estimate the clean data latent at each intermediate timestep by projecting the current state back to the data manifold and decoding it into pixel space. As illustrated in Figure 4, during the initial reasoning stages (e.g., Step 1), DiffThinker avoids premature commitment to a single path, instead exploring multiple candidate trajectories across the grid in parallel. Through successive iterations, the model simultaneously evaluates global constraints and environmental obstacles to prune invalid routes, progressively consolidating its focus onto the most plausible path and eventually converging to an optimal solution.

DiffThinker as a Collaborative Partner. Beyond a direct comparison, we explore the synergy between DiffThinker and MLLMs in solving complex tasks. As illustrated in Figure 6, DiffThinker first produces multiple candidate solution images, which the MLLM then evaluates against the original problem constraints to make a final decision.

We benchmark this collaborative approach on Jigsaw level-4, which demands both spatial reasoning and rigorous verification. Results demonstrate that this partnership achieves superior accuracy, outperforming either model in isolation. Specifically, DiffThinker compensates for the MLLM’s limited visual imagination in spatial reasoning, while the MLLM leverages its reflective capabilities to filter potential errors in the generated candidates. This synergy reveals that DiffThinker can serve as a powerful visual reasoning backend to augment the cognitive breadth of MLLMs.

|| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
| | |
<br><br>|
|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

|| |
|---|
<br><br>|
|---|

|| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

|| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

| |
|---|

Figure 7. Trade-off between accuracy and inference time across varying inference steps. The horizontal axis denotes the number of inference steps, while the vertical axis denotes accuracy or inference time. An optimal balance between reasoning performance and computational cost is achieved at approximately 20 steps.

#### 4.3. Ablation Studies

Ablation on Inference Steps. We first investigate the tradeoff between accuracy and inference time, as illustrated in Figure 7. DiffThinker demonstrates remarkable robustness, maintaining high performance even with as few as 10 inference steps. Increasing the step count to 20 yields a noticeable performance boost, identifying an optimal balance between solution quality and computational efficiency. Beyond 20 steps, the accuracy plateaus with only marginal fluctuations, suggesting that the underlying reasoning manifold is effectively captured early in the generative process. Based on these observations, we adopt 20 inference steps as our default configuration for evaluations to ensure superior performance with minimal inference overhead.

Ablation on Training Data Scale. We evaluate the influence of training data size on DiffThinker’s performance using our most complex tasks, Maze level-32 and Sudoku level-35. We first qualitatively analyze the model’s behavior under low-data regimes. As illustrated in Figure 8(a), due to the limited zero-shot reasoning capacity of the base model, DiffThinker initially focuses on mastering task-specific rendering syntax, such as grid alignment and trajectory continuity. As the training volume increases, DiffThinker transitions from superficial visual imitation to deep structural reasoning. Quantitative results in Figure 8(b) show that DiffThinker continues to benefit from data expansion, maintaining a consistent upward trajectory. With 105 samples, the model effectively internalizes underlying causal structures, achieving over 90% accuracy on Maze level-32, while the performance of MLLMs remain significantly limited despite the increased data. Based on these observations, we utilize a total of 30,000 samples across all difficulty levels for each task in our main experiments to achieve an optimal balance between performance and efficiency.

[Figure 7]

(a) Qualitative analysis with 100 training samples.

| |
|---|

| |
|---|

102 103 104 105

102 103 104 105

(b) Quantitative analysis with increasing training samples.

Figure 8. Ablation on Training Data Scale. (a) Qualitative analysis shows that with limited data, DiffThinker focuses on mastering rendering syntax. (b) Quantitative results on Maze level-32 and Sudoku level-35 demonstrate that DiffThinker scales consistently with data expansion.

Ablation on Classifier-Free Guidance Scale. We investigate the impact of Classifier-Free Guidance (CFG) (Ho & Salimans, 2022) on the reasoning capabilities of DiffThinker. As a core mechanism in diffusion models, CFG regulates the trade-off between conditional adherence and sample fidelity. The guided velocity field vˆθ is defined as:

vˆθ(xt,t,h) = vθ(xt,t,∅) + w(vθ(xt,t,h) − vθ(xt,t,∅))

(9) where vθ(xt,t,h) and vθ(xt,t,∅) represent the conditional and unconditional velocity predictions, respectively, and w denotes the CFG scale. We begin with a qualitative assessment of different CFG scales. Figure 9(a) visualizes the predicted original sample xˆ0 at the first step across varying scales. At w = 1, the insufficient conditioning produces faint and tentative trajectories, lacking the deterministic confidence for logical precision. Conversely, w = 7 triggers numerical over-saturation and visual artifacts, leading to distorted textures that severely degrade generative fidelity. Between these extremes, w = 4 effectively acts as a logic amplifier, generating bold and precise paths that perfectly align with task constraints.

Quantitatively, Figure 9(b) demonstrates that reasoning performance is robust across various guidance scales, with accuracy peaking at w = 4 across the majority of levels. Consequently, we adopt w = 4 as the default configuration

[Figure 8]

- (a) Qualitative results of the predicted sample xˆ0 at step 1.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- (b) Quantitative results of accuracy relative to CFG scales.

Figure 9. Ablation analysis of Classifier Free Guidance scales. (a) Impact of CFG scales on path clarity. (b) Accuracy trends across tasks confirming w = 4 as the peak performance point for balancing logic and fidelity.

for all experiments to ensure an optimal balance between logical precision and generative fidelity.

#### 4.4. Image Generation vs. Video Generation.

Video generation offers unique advantages for multimodal reasoning by explicitly modeling temporal coherence and the continuous evolution of state transitions. Its capacity to represent reasoning trajectories as a fluid sequence could naturally facilitate the resolution of complex planning tasks. Motivated by these potential benefits, we explore the feasibility of video-based reasoning and conduct a direct comparison with our image-based approach. Our video-based baseline, denoted as DiffThinker-Video, is implemented upon Wan2.2-TI2V-5B (Wan et al., 2025), a leading open source video foundation model. Due to the relatively weaker reasoning proficiency observed in current video generation models, we perform training and evaluation on Maze level-8, a relatively simple task that is also well-suited for videobased reasoning. To ensure a fair comparison, we train both models on identical datasets for varying numbers of epochs and report training duration and corresponding accuracy.

Qualitatively, Figure 10 demonstrates that DiffThinkerVideo possesses inherent reasoning capabilities; it resolves the maze problem by generating a video where a yellow ball progressively navigates the paths toward the target. Quantitatively, however, Figure 11 reveals that it yields lower accuracy with higher training overhead than DiffThinker. Furthermore, despite its smaller parameter count, DiffThinker-Video requires 2.0s per inference, nearly doubling the 1.1s latency of DiffThinker. These results highlight the prohibitive computational costs of video generation,

[Figure 9]

| |
|---|

| |
|---|

| |
|---|

Figure 10. Visual Trajectory of DiffThinker-Video. Visualized through accumulation of uniformly sampled frames.

Figure 11. Performance comparison between two paradigms.

underscoring the need for more efficient video models to advance generative multimodal reasoning.

### 5. Conclusion

In this paper, we introduce DiffThinker and establish Generative Multimodal Reasoning as a novel paradigm for complex vision-centric tasks. By leveraging diffusion models, we reformulate multimodal reasoning from a traditional textcentric symbolic mapping into a native generative image-toimage task, enabling models to perform reasoning within the visual space with superior logical consistency and spatial precision. Extensive experiments across four domains (sequential planning, combinatorial optimization, constraint satisfaction, and spatial configuration) demonstrate that DiffThinker significantly outperforms state-of-the-art MLLMs. Our systematic analysis further reveals the intrinsic advantages of this paradigm, including its efficiency, controllability, and native parallelism, while showcasing its potential as a collaborative backend to augment the cognitive breadth of MLLMs. We hope DiffThinker will inspire further exploration into Generative Multimodal Reasoning to unlock the full potential of multimodal intelligent agents.

### References

Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.

Avdeyev, P., Shi, C., Tan, Y., Dudnyk, K., and Zhou, J. Dirichlet diffusion score model for biological sequence generation. In International Conference on Machine Learning, pp. 1276–1301. PMLR, 2023.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv,

C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.

Chen, C., Ma, Z., Li, Y., Hu, Y., Wei, Y., Li, W., and Nie, L. Reasoning in the dark: Interleaved vision-text reasoning in latent space. arXiv preprint arXiv:2510.12603, 2025.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Du, Y., Zhou, K., Min, Y., Ling, Y., Zhao, W. X., and Wu, Y. Revisiting the necessity of lengthy chain-of-thought in vision-centric reasoning generalization. arXiv preprint arXiv:2511.22586, 2025.

Esser, P., Kulal, S., Andreas, A., Levi, L., Chertok, M., Gallo, H., Ganguli, D., Chou, K., Kim, S., Crowson, K., et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:https://arxiv.org/pdf/2403.03206, 2024.

Google. Gemini 3, 2025a. URL https://deepmind. google/models/gemini.

Google. Veo 3, 2025b. URL https://aistudio. google.com/models/veo-3.

Goren, N., Yehezkel, S., Dahary, O., Voynov, A., Patashnik, O., and Cohen-Or, D. Visual diffusion models are geometric solvers. arXiv preprint arXiv:2510.21697, 2025.

Graikos, A., Malkin, N., Jojic, N., and Samaras, D. Diffusion models as plug-and-play priors. Advances in Neural Information Processing Systems, 35:14715–14728, 2022.

Gu, J., Hao, Y., Wang, H. W., Li, L., Shieh, M. Q., Choi, Y., Krishna, R., and Cheng, Y. Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Guo, Z., Chen, X., Zhang, R., An, R., Qi, Y., Jiang, D., Li, X., Zhang, M., Li, H., and Heng, P.-A. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025b.

He, Z., Qu, X., Li, Y., Huang, S., Liu, D., and Cheng, Y. Framethinker: Learning to think with long videos via multi-turn frame spotlighting. arXiv preprint arXiv:2509.24304, 2025a.

He, Z., Qu, X., Li, Y., Huang, S., Liu, D., and Cheng, Y. Videossr: Video self-supervised reinforcement learning. arXiv preprint arXiv:2511.06281, 2025b.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D. J. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

Hong, J., Zhao, C., Zhu, C., Lu, W., Xu, G., and Yu, X. Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Huang, S., Qu, X., Li, Y., Luo, Y., He, Z., Liu, D., and Cheng, Y. Spotlight on token perception for multimodal reinforcement learning. arXiv preprint arXiv:2510.09285, 2025a.

Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., and Lin, S. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025b.

Ivanitskiy, M. I., Shah, R., Spies, A. F., R¨auker, T., Valentine, D., Rager, C., Quirke, L., Mathwin, C., Corlouer, G., Behn, C. D., et al. A configurable library for generating and manipulating maze datasets. arXiv preprint arXiv:2309.10498, 2023.

J¨unger, M., Reinelt, G., and Rinaldi, G. The traveling salesman problem. Handbooks in operations research and management science, 7:225–330, 1995.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Lai, X., Li, J., Li, W., Liu, T., Li, T., and Zhao, H. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025.

Li, Y., Guo, J., Wang, R., and Yan, J. T2t: From distribution learning in training to gradient search in testing for combinatorial optimization. Advances in Neural Information Processing Systems, 36:50020–50040, 2023.

Li, Y., Guo, J., Wang, R., Zha, H., and Yan, J. Fast t2t: Optimization consistency speeds up diffusion-based trainingto-testing solving for combinatorial optimization. Advances in Neural Information Processing Systems, 37: 30179–30206, 2024.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Liu, X., Xu, Z., Wang, K., Lee, Y. J., and Shang, Y. Can world simulators reason? gen-vire: A generative visual reasoning benchmark. arXiv preprint arXiv:2511.13853, 2025a.

Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., and Wang, J. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025b.

Luo, Y., Zhao, X., Lin, B., Zhu, L., Tang, L., Liu, Y., Chen, Y.-C., Qian, S., Wang, X., and You, Y. V-reasonbench: Toward unified reasoning benchmark suite for video generation models. arXiv preprint arXiv:2511.16668, 2025.

ModelScope. Diffsynth-studio. https://github. com/modelscope/DiffSynth-Studio, 2025.

OpenAI. Chatgpt, 2025a. URL https://chat. openai.com.

OpenAI. Thinking with images, 2025b. URL https://openai.com/index/ thinking-with-images/.

OpenAI. Sora, 2025c. URL https://sora.chatgpt. com/explore.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Pogodzinski, B., Wewer, C., Schiele, B., and Lenssen, J. E. Spatial reasoners for continuous variables in any domain. arXiv preprint arXiv:2507.10768, 2025.

Qin, Y., Wei, B., Ge, J., Kallidromitis, K., Fu, S., Darrell, T., and Wang, X. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025.

Qu, X., Li, Y., Su, Z., Sun, W., Yan, J., Liu, D., Cui, G., Liu, D., Liang, S., He, J., et al. A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond. arXiv preprint arXiv:2503.21614, 2025.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479–36494, 2022.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shen, C., Wei, W., Qu, X., and Cheng, Y. Satori-r1: Incentivizing multimodal reasoning with spatial grounding and verifiable rewards. arXiv preprint arXiv:2505.19094, 2025a.

Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025b.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. pmlr, 2015.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025a.

Su, Z., Xia, P., Guo, H., Liu, Z., Ma, Y., Qu, X., Liu, J., Li, Y., Zeng, K., Yang, Z., et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025b.

Sui, Y., Chuang, Y.-N., Wang, G., Zhang, J., Zhang, T., Yuan, J., Liu, H., Wen, A., Zhong, S., Chen, H., and Hu, X. Stop overthinking: A survey on efficient reasoning for large language models, 2025. URL https://arxiv.

org/abs/2503.16419.

Sun, Z. and Yang, Y. Difusco: Graph-based diffusion solvers for combinatorial optimization. Advances in neural information processing systems, 36:3706–3731, 2023.

Tong, J., Mou, Y., Li, H., Li, M., Yang, Y., Zhang, M., Chen, Q., Liang, T., Hu, X., Zheng, Y., et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025.

Ubukata, T., Li, J., and Tei, K. Diffusion model for planning: A systematic literature review. arXiv preprint arXiv:2408.10266, 2024.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu,

- Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang,
- Z., Han, Z., Wu, Z.-F., and Liu, Z. Wan: Open and advanced large-scale video generative models. arXiv

- preprint arXiv:2503.20314, 2025.

Wang, H., Qu, C., Huang, Z., Chu, W., Lin, F., and Chen, W. Vl-rethinker: Incentivizing self-reflection of visionlanguage models with reinforcement learning. arXiv

- preprint arXiv:2504.08837, 2025a.

Wang, H., Su, A., Ren, W., Lin, F., and Chen, W. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025b.

Wang, Q., Shi, Y., Wang, Y., Zhang, Y., Wan, P., Gai, K., Ying, X., and Wang, Y. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025c.

Wang, Z., Zhu, J., Tang, B., Li, Z., Xiong, F., Yu, J., and Blaschko, M. B. Jigsaw-r1: A study of rule-based visual reinforcement learning with jigsaw puzzles. arXiv preprint arXiv:2505.23590, 2025d.

Wewer, C., Pogodzinski, B., Schiele, B., and Lenssen, J. E. Spatial reasoning with denoising models. arXiv preprint arXiv:2502.21075, 2025.

Wiedemer, T., Li, Y., Vicol, P., Gu, S. S., Matarese, N., Swersky, K., Kim, B., Jaini, P., and Geirhos, R. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Wu, J., Huang, T., He, C., and Long, M. Miniveo3-reasoner: Thinking with videos from open-source priors. https: //github.com/thuml/MiniVeo3-Reasoner, 2025b.

- Wu, P., Zhang, Y., Diao, H., Li, B., Lu, L., and Liu, Z. Visual jigsaw post-training improves mllms. arXiv preprint arXiv:2509.25190, 2025c.
- Wu, Q., Zhao, H., Saxon, M., Bui, T., Wang, W. Y., Zhang, Y., and Chang, S. Vsp: Assessing the dual challenges of perception and reasoning in spatial planning tasks for vlms. arXiv preprint arXiv:2407.01863, 2024.

Xie, Y., Chen, T., Ge, Z., and Ni, L. Video-mtr: Reinforced multi-turn reasoning for long video understanding. arXiv preprint arXiv:2508.20478, 2025.

Xu, Y., Li, C., Zhou, H., Wan, X., Zhang, C., Korhonen, A., and Vuli´c, I. Visual planning: Let’s think only with images. arXiv preprint arXiv:2505.11409, 2025.

Yan, Z., Li, X., He, Y., Yue, Z., Zeng, X., Wang, Y., Qiao, Y., Wang, L., and Wang, Y. Videochat-r1. 5: Visual test-time scaling to reinforce multimodal reasoning by iterative perception. arXiv preprint arXiv:2509.21100, 2025.

Yang, C., Wan, H., Peng, Y., Cheng, X., Yu, Z., Zhang, J., Yu, J., Yu, X., Zheng, X., Zhou, D., et al. Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks. arXiv preprint arXiv:2511.15065, 2025a.

Yang, Z., Yu, X., Chen, D., Shen, M., and Gan, C. Machine mental imagery: Empower multimodal reasoning with

latent visual tokens. arXiv preprint arXiv:2506.17218, 2025b.

Zhang, H., Gu, X., Li, J., Ma, C., Bai, S., Zhang, C., Zhang, B., Zhou, Z., He, D., and Tang, Y. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416,

- 2025a.

Zhang, H., Wu, W., Li, C., Shang, N., Xia, Y., Huang, Y., Zhang, Y., Dong, L., Zhang, Z., Wang, L., et al. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514,

- 2025b.

Zhang, K., Zuo, Y., He, B., Sun, Y., Liu, R., Jiang, C., Fan, Y., Tian, K., Jia, G., Li, P., et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025c.

Zhang, Y.-F., Lu, X., Yin, S., Fu, C., Chen, W., Hu, X., Wen, B., Jiang, K., Liu, C., Zhang, T., et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025d.

Zhao, Y., Huang, J., Hu, J., Wang, X., Mao, Y., Zhang, D., Jiang, Z., Wu, Z., Ai, B., Wang, A., et al. Swift: a scalable lightweight infrastructure for fine-tuning, 2025.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

Table 2. Detailed statistics for training and testing datasets across five task categories. Task Category Difficulty Levels Training Samples Test Samples

3, 4, 5, 6 (Grid size) 500, 1,000, 2,500, 6,000 100 per level * VSP & VSP-Super 7, 8 (Grid size) – 100 per level †

16, 32 (Grid size) 10,000, 10,000 100 per level Maze 8, 16, 32 (Grid size) 10,000 100 per level

12, 15 (City count) 5000 100 per level TSP 13, 14, 16, 17 (City count) 5000 –

18 (City count) – 100 per level†

Sudoku 30 (Number of given clues) 7,500 – 35, 40, 45 (Number of given clues) 7,500 100 per level 1×2, 1×3, 2×1, 3×1 (Patch layout) 4000 per level –

Jigsaw & VisPuzzle 2×2, 3×3, 4×4 (Patch layout) 4,000, 5,000, 5,000 100 per level

VisPuzzle – 400*†

* denotes tasks utilizing official benchmarks from prior works. † denotes out-of-distribution testing scenarios.

Table 3. Hyperparameter settings for different training paradigms.

Flow Matching SFT GRPO

Framework DiffSynth-Studio (ModelScope, 2025) SWIFT (Zhao et al., 2025) verl (Sheng et al., 2024) Epochs 5 5 1 Learning Rate 1 × 10−4 1 × 10−4 1 × 10−6 LoRA Rank 32 32 – Batch Size 8 32 128 (8B) / 64 (32B) Rollout Size (n) – – 4 KL Coefficient – – 1 × 10−2

### A. Implementation Details

#### A.1. Training Details

- A.1.1. DATA PREPARATION.

The datasets utilized for training and evaluation are detailed in Table 2. Following previous research (Wang et al., 2025d; Wu et al., 2025c), we utilize the COCO (Lin et al., 2014) dataset to synthesize samples for both the training and testing of jigsaw puzzles. Specifically, we instantiate five independent models, each specialized for one of the five task categories, and subsequently evaluate them on their respective test benchmarks. All training datasets undergo thorough deduplication. Both DiffThinker and the baseline MLLMs are trained on identical data distributions to ensure an equitable comparison.

- A.1.2. HYPERPARAMETER CONFIGURATION.

We summarize the key training configurations and hyperparameters for Flow Matching, SFT, and GRPO in Table 3. In accordance with common practices, we employ Low-Rank Adaptation (LoRA) (Hu et al., 2022) for both the fine-tuning of Qwen-Image-Edit and the SFT of Qwen3-VL. For GRPO, considering the substantial computational overhead associated with reinforcement learning, we limit the training to a single epoch and utilize a reduced rollout number to maintain a manageable training budget while ensuring comparability across different experimental settings.

- A.1.3. REWARD FUNCTIONS FOR GRPO

Due to the limited zero-shot accuracy of the Qwen3-VL baselines on complex reasoning tasks, employing a strict binary reward based on exact matching results in extremely sparse signals, which significantly hinders the policy optimization process. Therefore, we design task-specific partial reward functions for each domain as follows:

Sequential Planning (VSP, VSP-Super, and Maze). For navigation tasks, we utilize a prefix matching reward. The reward evaluates the longest continuous sequence of correct actions from the starting point to ensure the model learns the correct

trajectory incrementally. Given a predicted action sequence P = (p1,p2,...,pm) and the ground truth G = (g1,g2,...,gn), the reward is defined as:

max{k | ∀i ≤ k,pi = gi and k ≤ min(m,n)} n

. (10)

Rplan =

Combinatorial Optimization (TSP). For the Traveling Salesperson Problem, the reward is designed to account for both coordinate set consistency and path length precision. Let Sp and Sg denote the sets of coordinates in the predicted and ground truth paths, and L(·) represent the total Euclidean distance of a trajectory. The reward is formulated as follows:

0.5(1 + I(|L(P) − L(G)| < ϵ)) if Sp = Sg 0 otherwise

, (11)

RTSP =

where ϵ = 1 × 10−4 serves as the tolerance for floating point comparisons. This tiered structure ensures that the model is first rewarded for identifying all required cities before optimizing the visitation order to match the ground truth distance.

Constraint Satisfaction (Sudoku). The Sudoku reward is based on the element wise accuracy of the completed grid. We first normalize the model output by extracting all numeric digits to form the predicted sequence P. If the length of the predicted sequence |P| matches the standard 81 digits required for a 9 × 9 grid, the reward is calculated as the proportion of correctly filled cells. The reward function is defined as:

RSudoku =

81 i=1 I(pi = gi) if |P| = 81

1 81

, (12)

0 otherwise

where I(·) denotes the indicator function and gi represents the ground truth value for the i-th cell. This objective encourages the model to respect both the structural integrity of the grid and the specific numerical constraints of the puzzle.

Spatial Configuration (Jigsaw and VisPuzzle). For Jigsaw tasks, the reward measures the positional accuracy of the restored image patches. After normalizing the predicted sequence P and the ground truth sequence G by removing extraneous whitespace, we evaluate the element wise matching rate. If the length of the predicted sequence |P| equals the total number of patches n, the reward is defined as the proportion of patches assigned to their correct absolute positions:

RJigsaw =

n i=1 I(pi = gi) if |P| = n

1 n

. (13)

0 otherwise

#### A.2. Prompt

Figures 12 through 17 provide a comprehensive overview of the prompt templates utilized in our study. For VSP and VSP-Super, we adopt the original prompt (Wu et al., 2024) designs as specified in the primary literature for the evaluation of Zero-Shot MLLMs. However, for SFT, the prompt structures are specifically adapted as illustrated in Figure 13. This modification is necessitated by the fact that our SFT paradigm does not employ Chain-of-Thought (CoT), requiring a more direct and concise instructional format to ensure consistency with the supervised training objectives.

### B. Limitations and Future Work

DiffThinker demonstrates state-of-the-art performance in vision-centric reasoning within targeted domains. However, its out-of-distribution (OOD) generalization remains constrained by the limited zero-shot reasoning proficiency of current generative foundation models. Since the reasoning process is directly modeled as a generative task, the model’s ability to handle unseen, complex scenarios is heavily tied to the representational depth of its underlying base. Future research should prioritize the development of more robust multimodal generative foundation models specifically optimized for reasoning. Building upon such foundations, we aim to further explore the boundaries of generative multimodal reasoning and enhance its capability to generalize across broader, out-of-distribution tasks.

Furthermore, this work primarily focuses on vision-centric challenges, where DiffThinker significantly surpasses traditional MLLMs. It is important to acknowledge, however, that MLLMs maintain a clear advantage in text-centric domains, such as complex mathematical problems. We do not view these paradigms as mutually exclusive; rather, a promising future direction lies in investigating deeper collaboration and synergy between generative reasoners and MLLMs. By integrating the superior visual precision of DiffThinker with the advanced linguistic and symbolic capabilities of MLLMs, we can extend the scope of multimodal reasoning to a wider spectrum of diverse and demanding tasks.

[Figure 10]

Figure 12. Prompt templates for DiffThinker.

### C. Qualitative Analysis

To facilitate a better understanding of the performance disparities between DiffThinker and MLLMs, we provide success and failure cases of DiffThinker for each task, along with the Thinking processes of Gemini-3-Pro (Google, 2025a), as shown in Figures 18 through 38. We utilize Google AI Studio to evaluate Gemini-3-Pro and obtain its reasoning duration. For each task, we evaluate Gemini-3-Pro on the same problem instances where DiffThinker achieved successful solutions.

[Figure 11]

###### Figure 13. Prompt templates of VSP and VSP-Super for MLLMs.

[Figure 12]

- Figure 14. Prompt templates of Maze for MLLMs.

[Figure 13]

- Figure 15. Prompt templates of TSP for MLLMs.

[Figure 14]

- Figure 16. Prompt templates of Sudoku for MLLMs.

[Figure 15]

Figure 17. Prompt templates of Jigsaw and VisPuzzle for MLLMs.

[Figure 16]

Figure 18. Failure case of DiffThinker on VSP. In a simple task, DiffThinker performs excessive parallel reasoning but fails to preserve a unique trajectory, ultimately leading to a failure.

[Figure 17]

###### Figure 19. Success case of DiffThinker on VSP.

[Figure 18]

###### Figure 20. Thinking process of Gemini-3-Pro on VSP. Gemini-3-Pro successfully provides the correct solution.

[Figure 19]

Figure 21. Failure case of DiffThinker on VSP-Super. In a complex task, DiffThinker identifies a nearly correct trajectory; however, the path is obstructed by a hole, preventing further progress and leading to an ultimate failure.

[Figure 20]

###### Figure 22. Success case of DiffThinker on VSP-Super.

[Figure 21]

Figure 23. Thinking process of Gemini-3-Pro on VSP-Super. Gemini-3-Pro fails to provide the correct solution.

[Figure 22]

Figure 24. Failure case of DiffThinker on Maze. In an instance characterized by a significant distance between the starting point and the goal, DiffThinker fails to sustain deep reasoning and provides only a preliminary trajectory.

[Figure 23]

###### Figure 25. Success case of DiffThinker on Maze.

[Figure 24]

###### Figure 26. Thinking process of Gemini-3-Pro on Maze. Gemini-3-Pro fails to provide the correct solution.

[Figure 25]

Figure 27. Failure case of DiffThinker on TSP. DiffThinker successfully identifies a feasible closed loop, yet it is not the shortest path.

[Figure 26]

###### Figure 28. Success case of DiffThinker on TSP.

[Figure 27]

###### Figure 29. Thinking process of Gemini-3-Pro on TSP. Gemini-3-Pro successfully provides the correct solution.

[Figure 28]

Figure 30. Failure case of DiffThinker on Sudoku. DiffThinker successfully populates most of entries, yet commits several errors.

[Figure 29]

###### Figure 31. Success case of DiffThinker on Sudoku.

[Figure 30]

###### Figure 32. Thinking process of Gemini-3-Pro on Sudoku. Gemini-3-Pro successfully provides the correct solution.

[Figure 31]

Figure 33. Failure case of DiffThinker on Jigsaw. Due to the fact that our test set is generated at random, certain instances contain regions that are extremely difficult to distinguish. DiffThinker produces a globally reasonable image, yet the fine details remain incorrect.

[Figure 32]

###### Figure 34. Success case of DiffThinker on Jigsaw.

[Figure 33]

Figure 35. Thinking process of Gemini-3-Pro on Jigsaw. Gemini-3-Pro successfully provides the correct solution.

[Figure 34]

###### Figure 36. Failure case of DiffThinker on VisPuzzle.

[Figure 35]

###### Figure 37. Success case of DiffThinker on VisPuzzle.

[Figure 36]

Figure 38. Thinking process of Gemini-3-Pro on VisPuzzle. Gemini-3-Pro successfully provides the correct solution.

