# VLMs are Good Teachers for Video Reasoning via Adaptive Test-Time Optimization

Junhao Cheng, Liang Hou, Tianxiong Zhong, Xin Tao, Pengfei Wan, Kun Gai, and Jing Liao

## arXiv:2606.02564v3[cs.CV]27Jun2026

Abstract—The emerging “Reasoning with Video” paradigm utilizes Video Generation Models (VGMs) to generate temporally coherent visual trajectories to complete reasoning tasks. Although state-of-the-art VGMs excel at visual quality, they often struggle to understand and adhere to task-specific rules, leading to logical mistakes across diverse reasoning scenarios. Existing efforts employ Vision-Language Models (VLMs) as problem pre-solvers to produce or refine textual guidance for VGMs. However, textual descriptions fail to capture intricate spatiotemporal details, and VGMs often struggle to faithfully execute fine-grained or long-tail instructions even with a valid plan. While VLMs struggle as solvers, they possess strong perception capabilities to evaluate whether process constraints are satisfied and the final goal is achieved. Leveraging this strength, we introduce a paradigm shift that transitions the role of VLMs to “teachers”. Specifically, a VLM teacher extracts task-specific rules to formulate differentiable rewards, guiding a VGM Reasoner via test-time optimization of a lightweight LoRA module. This strategy enables instancespecific adaptation at inference time and extends the reasoning capabilities beyond the VGM’s intrinsic boundaries. Extensive experiments on VBVR-Bench (symbolic reasoning) and RULERBench (general reasoning) show that our method gains 16.7 points on average. It surpasses the VLM-as-Solver scheme (+0.4 points) and Best-of-N scaling (+2.2 points) by a large margin under comparable test-time computational cost. These findings reveal that integrating VLMs as test-time teachers offers a promising paradigm for achieving generalizable video reasoning. Project Page: https://VLM-as-Teacher.github.io/

Index Terms—Video Reasoning, Video Generation, VisionLanguage Models (VLMs), Test-Time Optimization.

I. INTRODUCTION

Recent advancements in Video Generation Models (VGMs) demonstrate strong performance in synthesizing realistic and temporally coherent videos [1–3]. Beyond content creation, several pioneering studies [4, 5] try to employ VGMs to solve logical reasoning tasks, forming an emerging research direction called “Reasoning with Video”. By generating coherent visual trajectories, VGMs can address vision-centric reasoning challenges that are difficult to specify using language alone, such as the precise rotation of irregular objects. In certain tasks such as maze solving and puzzles, VGMs have been shown to match or even exceed the performance of state-of-the-art (SOTA) Vision-Language Models (VLMs) that rely primarily on textual reasoning chains [6]. However, the optimization goal of VGMs is primarily visual fidelity [7, 8], leading to the models’ intrinsic limitations in performing logical reasoning and following task-

Junhao Cheng and Jing Liao are with City University of Hong Kong. Liang Hou, Tianxiong Zhong, Xin Tao, Pengfei Wan, and Kun Gai are with

Kling Team, Kuaishou Technology. Corresponding author: Jing Liao.

specific rules. As a result, they often generate trajectories that are visually plausible but logically inconsistent with the goals.

To address the intrinsic limitations of VGMs, some efforts have explored test-time scaling (TTS) strategies, such as Bestof-N sampling or rejection-based schemes [9] in video reasoning. As illustrated in Fig. 1, these methods keep the VGM fixed and search among sampled videos. While effective at reducing stochastic errors, these approaches provide limited gains in video reasoning tasks. Systematic failures such as logically inconsistent trajectories and missed causal dependencies, cannot be easily corrected through repeated sampling because the model’s inherent generative capacity constrains the solution space. Another line of work explores integrating VLMs as pre-solvers or planners to guide video reasoning [10, 11]. As shown in Fig. 1, this “VLM-as-Solver” paradigm provides textual guidance for the VGM. However, reasoning via text alone remains challenging: linguistic prompts often fail to capture intricate spatiotemporal constraints, and even when a plan is detailed and logically sound, VGMs frequently struggle to faithfully execute fine-grained or long-tail instructions despite receiving a valid plan [12].

Nevertheless, VLMs that struggle to construct executable visual solution trajectories are well suited to verifying whether a generated trajectory satisfies observable process constraints and reaches the intended final goal. For instance, even when a VLM cannot plan the exact steps for navigating a ball through a maze, it can evaluate whether the ball reaches the exit and whether its trajectory preserves the ball’s identity and avoids crossing walls. Together, these conditions characterize successful task completion. Leveraging this strength, we uncover a new role for VLMs as “teachers”, as shown in Fig. 1. In this paradigm, a VLM extracts task-specific rules and formulates them as differentiable rewards by proposing queries that assess whether intermediate steps adhere to constraints and whether the final state satisfies the intended goal. Unlike TTS, which keeps the VGM fixed and searches among sampled videos, these rewards guide a VGM Reasoner through test-time optimization (TTO), where instance-specific parameters are optimized under a taskspecific objective during inference. Specifically, we optimize a lightweight LoRA module [13] to adapt the VGM Reasoner to each reasoning instance, while using early video prediction, lightweight decoding, and loss-based early stopping to keep per-instance optimization efficient. By directly backpropagating differentiable feedback from the VLM, the VGM can refine its reasoning trajectories during inference, effectively aligning rule logic with visual execution and extending capabilities beyond its intrinsic limits.

Evaluations on symbolic (VBVR-Bench [14]) and general-

The Challenge: Reasoning with Video Generation Model Performance Comparison on VBVR-Bench

| |
|---|

Condition

The scene contains multiple colored objects and star markers. Keep all star markers unchanged in position. Move each colored object to the star marker with the same color using straight paths.

Prompt

[Figure 1]

Ours

Step 16

OverallMetrics

NaïveBest-of-NScaling VLMBest-of-Nas Solver VLM asBest-of-NTeacher (Ours)

Step 10

Achieve the final Goal?

[Figure 2]

[Figure 3]

[Figure 4]

SelectBest

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### …

[Figure 13]

Step 5

LoRA

Naïve Scaling

[Figure 14]

[Figure 15]

VGM Video VGM Video VLM Videos

VGM VLM Text Solution

pass@5 pass@1

pass@3 pass@4

Correct reasoning path?

pass@2

Result Result

###### Problems

###### Problems

Result

Advantages

VLM as Solver

| |
|---|

| |
|---|

| |
|---|

Hard to describe rea-

[Figure 16]

Inherent capacity constraints

Generalized TaskCustomization

[Figure 17]

[Figure 18]

soning path via text

[Figure 19]

Marginal room for

Extending VGM’s

[Figure 20]

[Figure 21]

Text-to-execution Gap

improvement

Inherent Ability

Generation Cost (s)

- Fig. 1: Vision-Language Models (VLMs) as Teachers Rather Than Solvers for Video Reasoning. Unlike Test-time scaling with fixed Video Generation Models (VGMs) or VLM-as-Solver methods that rely on textual guidance, our VLM-as-Teacher paradigm supervises process constraints and final-goal achievement to guide a VGM Reasoner via online test-time optimization.

purpose (RULER-Bench [15]) video reasoning benchmarks show that the proposed method yields a 16.7-point average performance gain, comparing favorably against the VLM-asSolver paradigm (+0.4 points) and Best-of-N scaling (+2.2 points) at comparable test-time cost, offering a promising paradigm to empower reasoning in video generation models.

We make the following contributions in this work:

- • We uncover a new VLM-as-Teacher paradigm for video reasoning, which fundamentally shifts the role of VLMs from text-based solvers to test-time supervisors that provide optimization signals for reasoning.
- • We introduce a test-time online optimization approach for VGMs that adapts a VGM through differentiable VLM rewards, enabling reasoning capability beyond the model’s intrinsic generative limits.
- • We propose a task-adaptive reward synthesis strategy that automatically derives process and goal rewards from task descriptions, which together serve as sufficient conditions for successful reasoning task completion.

II. RELATED WORK

Reasoning with Video. Since the emergence of diffusion models and transformer-based scaling [16–18], video generation models have witnessed rapid proliferation. This includes closedsource pioneers such as Sora, Veo, and Seedance, as well as open-source counterparts like CogVideoX, HunyuanVideo, and Wan [1–3, 19–24]. While these models excel at synthesizing videos with high visual fidelity [17, 23, 25, 26], recent research has further sought to optimize their alignment with physical laws and real-world dynamics [27–38]. Despite these advancements in visual and physical realism, they are not specifically optimized for rule-based relational, causal, or counterfactual reasoning.

To bridge this gap, the emerging “Thinking with Frames” paradigm re-conceptualizes video generation as a computa-

tional substrate for visual reasoning rather than mere synthesis [4, 5, 39, 40]. Preliminary studies on models like Veo-3 provide early evidence that large-scale pre-training can evoke non-trivial zero-shot perceptual and manipulation behaviors, enabling the solution of simple tasks without taskspecific fine-tuning [40]. Drawing an analogy to the Chainof-Thought (CoT) prompting in LLMs [41], recent works suggest that reasoning emerges through multi-step “Chainof-Frame” (CoF) diagnosis [5, 6, 39, 42], where extended temporal sequences serve as explicit reasoning trajectories. Conversely, Wang et al. [43] argue that reasoning processes are latent within the early stages of the denoising process, formulated as “Chain-of-steps” (CoS) reasoning. To quantify these capabilities, various benchmarks have been established to evaluate reasoning through synthetic puzzles such as maze solving and Sudoku [44, 45], as well as complex Text-Imageto-Video (TI2V) tasks [11, 46, 47]. Large-scale synthetic datasets now span five core dimensions, including perception, transformation, spatiality, abstraction, and knowledge, and encompass thousands of diverse tasks [14]. Beyond symbolic visual reasoning, benchmarks such as RULER-Bench [15] and FAR [48] further evaluate general-purpose video reasoning in open-ended scenarios. Despite the rapid development of benchmarks and diagnostic analyses, generalizable algorithmic solutions that bridge visual synthesis and logical rule adherence remain scarce.

Test-Time Scaling for Video Reasoning. Test-time scaling has emerged as a powerful mechanism to enhance the performance of Large Language Models (LLMs) [49, 50] and diffusion models [51] by allocating additional compute during inference without modifying model parameters. Recent video-specific extensions [52–56] extend this concept to the temporal axis through frame-level tree searches, evolutionary sampling, and iterative self-refinement. Specifically for video reasoning, several approaches adapt Best-of-N scaling strategies; for

Reward Queries

Final Result

###### Condition

###### Instructions

Process Supervision:

Does the purple ball maintain its integrity without disappearing or

Propose supervision questions on the processes and final objective of this task.

[Figure 22]

VLM

exhibiting wall-clipping?

Goal Achievement: Do the purple circle and green square overlap at the end?

Prompt: Move the purple circle

along the maze to reach the green square without crossing walls.

[Figure 23]

Optimize until Convergence (Max N steps)

IntermediateResult

[Figure 24]

[Figure 25]

Light

VAE

### … VLM

VAE

Final Result

Online

VGM

Inference

Optimization

(Few Steps Prediction) (One Step Prediction)

LoRA

Step 1 Step 2 Step k

∇ℒMulti-VQA

- Fig. 2: Adaptive test-time optimization with a VLM Teacher. Given a rule-based video reasoning task, the VLM Teacher extracts task-specific process constraints and the final goal, and converts them into reward queries. During online optimization, an intermediate video prediction from the VGM is evaluated by the VLM Teacher. The resulting differentiable feedback updates a LoRA module. The optimized VGM then produces the final visual reasoning trajectory after the optimization loop ends.

instance, Wang et al. [43] aggregate early denoising layers across different sampling seeds to produce optimal results, while EPBS [9] leverages the “early commitment” characteristic of video reasoning to accelerate the scaling process.

However, these methods are fundamentally constrained by the inherent generative capacity of the base models. In complex reasoning tasks, failures are often systematic, such as logically flawed solution paths, skipped sub-goals, or physically inconsistent outcomes, rather than stochastic errors that can be mitigated through repeated sampling. Consequently, simply increasing test-time scaling through rejection sampling or ensemble methods yields limited gains. This motivates a different form of test-time computation: test-time optimization, which optimizes instance-specific variables or parameters under a test-time objective. In this work, we adopt TTO for video reasoning, allowing the VGM Reasoner to actively adapt toward rule-compliant visual trajectories.

#### Integrating VLMs for Video Reasoning.

Vision-Language Models (VLMs) possess formidable perceptual and reasoning capabilities, making them ideal candidates for enhancing reasoning tasks [57–62]. Current LLM/VLMguided generation paradigms typically cast the large model as a symbolic planner or a problem solver. These approaches, originating in the image domain [63–65] and extending to video [12, 31, 66–69], primarily optimize visual or physical attributes through text-based orchestration. Recent efforts have attempted to adapt this paradigm to video reasoning; for instance, VideoTPO [11] uses LLM critiques to iteratively refine prompts, while CollabVR [10] employs the VLM as a progressive planner and solver. However, these systems rely heavily on textual prompts, which often struggle to capture intricate spatiotemporal nuances. Furthermore, even with a logically sound plan, VGMs frequently fail to execute fine-grained or long-tail concepts due to the inherent gap between linguistic instructions and visual synthesis. While VLMs struggle as solvers, they excel at evaluating generative

processes. We therefore transition the role of a VLM from a “solver” to a “teacher”. Specifically, a VLM Teacher formulates differentiable rewards from task-specific rules and guides a VGM through test-time optimization, bridging the gap between high-level logic and visual execution.

III. METHOD A. Task Formulation

In this paper, we study rule-based video reasoning, where a VGM produces a temporally coherent visual trajectory (a video) that follows task-specific rules and achieves an intended goal. This setting covers symbolic visual reasoning tasks, such as spatial navigation, geometric manipulation, object arrangement, and sequential state transformation [6, 9], as well as generalpurpose scenarios, such as anomaly removal, object rotation, and hypothesis generation [14, 15].

Formally, a reasoning instance is specified by a condition c = (p,x), where p denotes a textual instruction and x denotes an optional condition image. Given c, a VGM Gθ generates a video as a visual reasoning trajectory:

v = Gθ(c;ϵ) = {v1,v2,...,vT}, (1) where θ denotes the parameters of the VGM and ϵ denotes the sampling randomness. Following prior formulations [5, 14, 15], successful task completion requires achieving the final goal while satisfying the process constraints. We denote the finalgoal predicate by g(v,c) and the set of process-constraint predicates by R(v,c) = {rm(v,c)}Mm=1. Accordingly, task success is formulated as

M

rm(v,c) = 1 . (2)

Succ(v,c) = I g(v,c) = 1 ∧

m=1

The central challenge is that the required rules vary across individual tasks and conditions. It is difficult for a general set of reward functions to characterize diverse task-specific

constraints [70]. To address this, we use a VLM Teacher to synthesize supervision queries for each case and directly guide the VGM via test-time optimization.

B. VLM-as-Teacher Framework

- Fig. 2 illustrates the proposed VLM-as-Teacher framework,

which consists of a VLM Teacher and a VGM Reasoner equipped with a lightweight LoRA module for test-time optimization. Rather than generating a textual solution trajectory, the VLM Teacher first identifies the requirements for successful task completion and then provides differentiable supervision to optimize the VGM Reasoner. This raises the challenge of how to convert the teacher’s evaluative feedback into an effective optimization signal for video reasoning.

Recent pioneering studies have shown that VLM feedback can be formulated as a differentiable objective for generative models [71–73]. Luo et al. [71] apply differentiable VLM rewards to image generation with manually specified queries, while others [72, 73] use VLM feedback to post-train generative models toward general visual quality rather than performing test-time optimization. Different from these works, we adapt differentiable VLM feedback to task-adaptive video reasoning. The VLM Teacher automatically derives goal-achievement and process-supervision queries from each reasoning condition, instead of relying on manually specified or task-agnostic queries. The resulting rewards are used to optimize the VGM Reasoner for the current test instance, rather than serving as a shared post-training objective. To make such video-level testtime optimization practical, we update only a lightweight LoRA module and evaluate an efficient first-step video prediction with a surrogate VAE decoder. In this way, VLM feedback directly supervises rule satisfaction, final-goal achievement, and reasoning-trajectory validity with manageable optimization cost. The overall procedure is summarized in Algorithm 1.

Task-Adaptive Supervision Synthesis. Given a task condition c, the VLM Teacher first analyzes the textual instruction and the optional visual context to identify the success requirements of the task. It then formulates these requirements as binary reward queries. Specifically, the teacher synthesizes one goal achievement query qgoal(c) and M process supervision queries {qprocm (c)}Mm=1, where typically 1 ≤ M ≤ 3. The resulting query set is defined as

Q(c) = {qgoal(c)} ∪ qprocm (c) Mm=1 . (3) The process supervision queries evaluate whether the generated trajectory follows the task-specific rules, such as object integrity, valid motion, temporal continuity, collision constraints, or state consistency. The goal achievement query evaluates whether the final state satisfies the intended objective. For example, in the maze navigation task shown in Fig. 2, the teacher generates process queries that examine whether the purple ball remains intact and avoids crossing walls, together with a goal query that examines whether the ball reaches the green target region.

All reward queries are phrased positively, i.e., a “Yes” response indicates satisfaction of the corresponding requirement. This formulation provides a unified reward interface for heterogeneous rule-based reasoning tasks without manually defining

Algorithm 1 Test-time Optimization with a VLM Teacher

Require: Task condition c = (p,x); VGM Reasoner Gθ,ϕ; VLM Teacher Pψ; lightweight surrogate decoder Dlite; maximum optimization steps N; loss threshold τL; sampled frame number K

Ensure: Final visual reasoning trajectory v∗

- 1: Q(c) ← SYNTHESIZEQUERIES (Pψ,c)
- 2: Initialize LoRA parameters ϕ0; set ϕ∗ ← ϕ0
- 3: Sample initial pure-noise latent z1 = ϵ
- 4: for n = 0,...,N − 1 do
- 5: zˆ0(n) ← z1 − uθ,ϕ

n

(z1,1,c)

- 6: v˜(n) ← SAMPLEK Dlite z ˆ0(n)
- 7: L(Multin) -VQA ← VLM-LOSS Pψ;v˜(n),Q(c)
- 8: if L(Multin) -VQA ≤ τL then
- 9: ϕ∗ ← ϕn
- 10: break
- 11: end if
- 12: ϕn+1 ← ϕn − η∇ϕnL(Multin) -VQA
- 13: ϕ∗ ← ϕn+1
- 14: end for
- 15: v∗ ← Gθ,ϕ∗(c;ϵ) {Decode with the standard VAE}
- 16: return v∗

reward functions for individual task categories. Moreover, the two types of supervision are complementary: the goal achievement query alone does not prevent invalid intermediate trajectories, while the process supervision queries alone do not ensure successful task completion.

Online Optimization Process. With the reward queries, we next utilize the VLM Teacher to guide the reasoning trajectory of the VGM Reasoner. We apply differentiable VLM supervision to test-time optimization of a VGM Reasoner, enabling task-specific optimization for each rule-based video reasoning instance.

For each reasoning instance, the pretrained VGM backbone and the VLM Teacher remain frozen, and only a lightweight LoRA module is optimized. Let ϕn denote the LoRA parameters at the n-th optimization step, and let v˜(n) denote the intermediate video result evaluated by the VLM Teacher. Following the differentiable VQA formulation, the VLM Teacher evaluates each video-query pair by predicting a target answer sequence. Since all synthesized reward queries are positively phrased, the target answer for every query is the response “Yes”. We denote the tokenized target answer by Sa+ = Tok(Yes) = {a+ℓ }Lℓ=1, where L is the number of tokens. For each reward query q ∈ Q(c), we define the corresponding VQA loss as

L

log Pψ a+ℓ | v˜(n),q,a+<ℓ , (4)

LVQA v ˜(n),q = −

ℓ=1

where Pψ denotes the frozen VLM Teacher. Unlike visual instruction tuning, which optimizes the parameters of the VLM, the proposed objective propagates gradients through the visual prediction to optimize the LoRA parameters of the VGM Reasoner. Based on the synthesized query set, the complete

objective consists of one goal achievement term and M process supervision terms:

L(Multin) -VQA =λLVQA v ˜(n),qgoal(c)

(5)

M

1 − λ M

LVQA v ˜(n),qprocm (c) ,

+

m=1

where λ is a balance factor. The LoRA parameters are then updated by

##### ϕn+1 = ϕn − η∇ϕnL(Multin) -VQA, (6) with learning rate η.

Efficient Adaptation. Applying differentiable VLM supervision to video generation is computationally demanding, since a straightforward implementation requires repeated multi-step denoising, decoding with a heavy video VAE [3], and VLM evaluation during optimization [73]. We introduce three designs to make the online optimization practical.

First, we replace the standard VAE with a lightweight surrogate decoder [74] during online optimization. This substantially reduces the memory and computation overhead of differentiable video decoding at the cost of moderate visual quality degradation. Experiments at Section 4.3 show that such degradation has a negligible effect on the VLM Teacher’s evaluation accuracy. After optimization, the final visual reasoning trajectory is generated by the adapted VGM Reasoner and decoded using the standard VAE.

Second, we distill the VGM Reasoner into a four-step generator using [75], and update only its first-step clean-latent prediction during online optimization. Let z1 = ϵ denote the initial pure-noise latent, and let uθ,ϕ

denote the velocity predicted by the adapted VGM Reasoner. We obtain the onestep clean-latent prediction by applying the full sampling interval to this velocity prediction:

n

##### zˆ0(n) = z1 − uθ,ϕ

##### (z1,1,c), z1 = ϵ. (7)

n

Recent analysis indicates that the high-level reasoning behavior of video generation models emerges in early denoising steps [43]. In addition, we observe that the first-step prediction of a few-step Reasoner already provides a visually perceptible approximation of the reasoning trajectory. Therefore, the VLM Teacher can evaluate the reasoning behavior without repeatedly completing the full denoising process. We then decode and uniformly sample K frames from the decoded first-step prediction as the input for VLM evaluation. Since the lightweight decoding and frame sampling operations preserve the computation graph, gradients from the VLM Teacher can be propagated through v˜(n) to the LoRA parameters ϕn.

Third, we employ loss-based early stopping to avoid unnec-

essary optimization steps. Since L(Multin) -VQA is defined by the negative log-likelihood of the positive answer “Yes” over the

goal achievement query and all process supervision queries, a lower loss indicates that the VLM Teacher assigns higher confidence to the satisfaction of the task requirements. Online

optimization terminates when L(Multin) -VQA ≤ τL or when the maximum number of optimization steps N is reached, where

τL denotes the predefined loss threshold. The resulting LoRA

module is then used by the VGM Reasoner to generate the final visual reasoning trajectory.

IV. EXPERIMENTS A. Experimental Setup

Benchmarks and Metrics. We evaluate the proposed method on two complementary video reasoning benchmarks. VBVRBench [14] focuses on symbolic visual reasoning tasks across five capability categories: abstraction, knowledge, perception, spatiality, and transformation. RULER-Bench [15] contains general-purpose reasoning scenarios spanning six rule categories: humanity, science, hypothesis, semantics, vision, and game. Since the game tasks in RULER-Bench substantially overlap with the symbolic reasoning scenarios evaluated in VBVR-Bench, we exclude this category and evaluate the remaining 30 tasks from the other five categories. For VBVRBench, we report the overall score together with the in-domain (ID) and out-of-domain (OOD) averages. Since its tasks have verifiable outcomes, VBVR-Bench evaluates generated videos using task-specific rule-based detection scorers that measure spatial accuracy, trajectory correctness, temporal consistency, and logical validity. For RULER-Bench, we report the average score over the 30 evaluated task categories. Following its official protocol, each generated video is evaluated using checklist questions under four dimensions: instruction following, visual consistency, visual fidelity, and rule coherence. The checklist responses are scored by GPT-o3 [76], following the evaluator adopted in the benchmark. For both benchmarks, we follow the officially released metrics and evaluation protocols to ensure fair comparison. We additionally report the average total generation time per sample for efficiency comparison.

Compared Methods. We compare the proposed method with SOTA closed-source and open-source VGMs, including Sora 2 [1], Kling 2.6 [77], Veo 3.1 [20], and Wan2.2 [3]. Based on these generators, we compare three types of test-time reasoning strategies. Pass@N performs sampling-based testtime scaling by generating N candidates with different initial noises and selecting the best result according to the evaluation criterion. PE (Prompt Engineering) and VideoTPO represent the “VLM-as-Solver” paradigm, where a VLM improves video generation through textual task specification. Specifically, PE uses a VLM to interpret the reasoning task and rewrite the initial prompt before video generation, while VideoTPO further observes generated results and iteratively refines the prompt through VLM feedback.

Implementation Details. Unless otherwise specified, we use a step-distilled Wan2.2-5B as our VGM Reasoner and Qwen3VL-4B [78] as the VLM Teacher. The VGM Reasoner is distilled into a four-step generator following DMD2 [75]. For VBVR-Bench, following the official setting [14], we first perform domain-adaptive supervised fine-tuning on its 30K training instances for all open-source baselines.

During online optimization, only the LoRA parameters are updated. The first-step clean-latent prediction is decoded using the lightweight surrogate decoder from LightX2V [74]. We uniformly sample K = 24 frames for VLM evaluation and set

- TABLE I: Benchmarking results on VBVR-Bench. Higher is better. Cost stands for average total inference generation seconds per sample. Bold stands for best in group; Underlined stands for second best in group. Tasks include Abstraction (Abs.), Knowledge (Know.), Perception (Perc.), Spatiality (Spat.), and Transformation (Trans.).

Models Cost (s) Overall

In-Domain by Category Out-of-Domain by Category Avg. Abst. Know. Perc. Spat. Trans. Avg. Abst. Know. Perc. Spat. Trans.

Closed-source Models Sora 2 - 0.546 0.569 0.602 0.477 0.581 0.572 0.597 0.523 0.546 0.472 0.525 0.462 0.546 Kling 2.6 - 0.369 0.408 0.465 0.323 0.375 0.347 0.519 0.330 0.528 0.135 0.272 0.356 0.359 Veo 3.1 - 0.480 0.531 0.611 0.503 0.520 0.444 0.510 0.429 0.577 0.277 0.420 0.441 0.404

Open-source Models VBVR-Wan2.2-14B 160 0.682 0.763 0.733 0.713 0.795 0.776 0.827 0.601 0.732 0.596 0.542 0.628 0.600 VBVR-Wan2.2-5B 87 0.676 0.713 0.675 0.722 0.715 0.733 0.715 0.639 0.711 0.618 0.642 0.678 0.548

- + Pass@2 174 0.690 0.729 0.686 0.749 0.727 0.751 0.727 0.650 0.718 0.659 0.647 0.680 0.559

- + Pass@3 261 0.693 0.733 0.693 0.751 0.728 0.753 0.736 0.652 0.720 0.660 0.650 0.682 0.560

- + Pass@4 348 0.700 0.740 0.713 0.757 0.729 0.756 0.740 0.660 0.723 0.661 0.665 0.695 0.563

- + Pass@5 435 0.701 0.741 0.714 0.762 0.730 0.757 0.741 0.661 0.725 0.664 0.665 0.695 0.564

+ VideoTPO 276 0.663 0.697 0.654 0.701 0.698 0.724 0.708 0.629 0.703 0.604 0.631 0.669 0.538 VBVR-Wan2.2-5B-Distilled 14 0.666 0.692 0.638 0.709 0.661 0.732 0.712 0.640 0.688 0.603 0.651 0.693 0.565

- + Pass@2 28 0.675 0.702 0.653 0.717 0.674 0.743 0.718 0.647 0.701 0.603 0.651 0.707 0.575

- + Pass@3 42 0.678 0.707 0.659 0.719 0.679 0.748 0.722 0.650 0.702 0.603 0.651 0.720 0.576

- + Pass@4 56 0.681 0.711 0.673 0.722 0.680 0.748 0.726 0.652 0.703 0.603 0.651 0.727 0.581

- + Pass@5 70 0.683 0.712 0.676 0.722 0.680 0.751 0.726 0.653 0.709 0.603 0.651 0.728 0.582

+ VideoTPO 57 0.634 0.671 0.624 0.687 0.643 0.712 0.689 0.597 0.652 0.584 0.613 0.641 0.495 + Ours 69 0.781 0.803 0.806 0.920 0.837 0.820 0.787 0.759 0.873 0.765 0.759 0.818 0.639

- TABLE II: Benchmarking results on RULER-Bench. Higher is better. Cost stands for average total generation seconds per sample. Bold denotes the best result in each group, and underlined denotes the second best. Tasks include Transportation (Tra.), Sports (Spo.), Social (Soc.), Safety (Saf.), Festival (Fes.), Dress (Dre.), Food (Foo.), Emotion (Emo.), Chemistry (Che.), Physics (Phy.), Biology (Bio.), Earth Science (Ear.), Mathematics (Mat.), Medicine (Med.), Life (Lif.), Subjective (Sub.), Objective (Obj.), Idiom (Idi.), Metaphor (Met.), Definition (Def.), Anomaly (Ano.), Color (Col.), Count (Cou.), Direction (Dir.), Position (Pos.), Shape (Sha.), Size (Siz.), Style (Sty.), Viewpoint (Vie.), and Motion (Mot.).

Humanity Science Hypothesis Semantics Vision

Models Cost (s) Avg.

Tra. Spo. Soc. Saf. Fes. Dre. Foo. Emo. Che. Phy. Bio. Ear. Mat. Med. Lif. Sub. Obj. Idi. Met. Def. Ano. Col. Cou. Dir. Pos. Sha. Siz. Sty. Vie. Mot.

Closed-source Models Veo 3.1 - 65.0 80.0 78.1 74.9 82.8 80.8 90.9 90.9 69.2 81.1 79.1 83.3 57.3 61.5 63.4 74.5 74.1 81.3 76.5 81.4 87.3 42.2 63.9 20.1 32.5 37.5 46.9 47.9 47.5 51.3 55.3

+ PE - 66.2 78.5 82.4 75.1 85.5 90.0 94.2 93.4 59.5 86.6 74.5 81.2 64.4 61.3 68.1 83.7 75.4 81.9 83.8 79.7 86.6 38.4 62.5 26.4 36.9 28.1 50.8 46.4 47.5 51.3 57.9 Sora 2 - 59.8 71.6 73.5 77.8 80.1 84.4 89.8 88.9 63.6 85.8 76.4 84.1 52.9 70.1 64.6 65.9 62.7 73.3 72.1 70.4 80.1 32.8 22.9 34.0 30.0 41.9 39.6 38.0 43.8 31.3 41.8 + PE - 62.9 69.4 79.7 75.2 87.9 89.7 93.7 85.7 55.6 81.6 82.3 76.4 51.1 72.9 80.6 73.5 73.0 84.3 75.1 82.1 79.0 35.6 38.2 38.2 28.8 42.5 40.3 31.3 46.9 32.5 53.7

Open-source Models Wan2.2-14B 212 49.4 50.7 59.7 51.7 59.4 68.0 85.7 61.0 40.1 51.6 54.2 52.4 48.9 43.3 47.8 55.6 44.9 61.7 57.8 70.5 64.0 28.4 33.3 38.9 44.4 36.9 37.5 41.7 32.5 38.1 50.2 Wan2.2-5B 98 46.7 48.6 57.5 47.9 56.5 65.2 81.7 59.1 37.7 49.2 50.7 50.2 46.0 41.5 44.9 52.5 42.2 58.2 53.7 66.7 60.7 26.7 31.2 35.7 40.7 34.7 35.2 38.7 30.2 35.7 48.2

+ Pass@5 490 49.6 51.1 59.5 50.4 58.7 66.0 83.3 60.6 41.7 51.6 53.7 54.2 48.5 44.3 45.7 53.3 42.6 58.7 54.4 68.9 63.2 32.5 36.2 37.7 45.9 40.5 40.2 44.5 35.2 41.2 50.2 + PE 101 48.6 48.1 60.0 47.4 59.5 70.2 84.5 62.6 34.2 51.2 51.7 48.7 48.5 42.7 50.4 57.0 47.2 63.7 57.7 69.2 60.2 27.7 32.7 39.2 42.2 33.2 37.2 36.9 31.7 37.7 52.2 + VideoTPO 311 50.6 49.4 62.0 48.6 61.7 73.0 86.6 64.9 36.2 53.3 53.5 49.9 50.5 44.2 53.5 59.8 50.6 67.1 60.5 71.4 61.3 29.1 34.4 41.4 43.8 34.3 38.9 38.1 33.4 39.6 54.7

Wan2.2-5B-Distilled 18 46.4 47.7 57.0 47.2 56.0 64.6 81.0 58.5 37.3 48.3 50.0 49.6 45.3 41.2 44.0 52.1 41.7 57.7 53.3 66.3 60.2 26.7 31.5 36.0 40.8 35.0 35.6 39.0 30.6 35.9 48.8 + Pass@5 90 49.1 49.9 58.8 49.4 58.0 65.3 82.4 59.9 41.1 50.5 52.8 53.4 47.6 43.8 44.7 52.8 42.0 58.1 53.9 68.3 62.5 32.1 36.2 37.8 45.6 40.4 40.3 44.4 35.3 41.0 50.6 + PE 20 48.3 47.2 59.5 46.7 59.0 69.6 83.8 62.0 33.8 50.3 51.0 48.1 47.8 42.4 49.5 56.6 46.7 63.2 57.3 68.8 59.7 27.7 33.0 39.5 42.3 33.5 37.6 37.2 32.1 37.9 52.8 + VideoTPO 71 50.3 48.5 61.5 47.9 61.2 72.4 85.9 64.3 35.8 52.4 52.8 49.3 49.8 43.9 52.6 59.4 50.1 66.6 60.1 71.0 60.8 29.1 34.7 41.7 43.9 34.6 39.3 38.4 33.8 39.8 55.3 + Ours 88 68.2 78.6 79.8 75.3 85.6 72.8 93.8 91.0 63.7 85.9 79.2 83.4 57.4 70.2 53.0 59.6 49.5 66.9 76.6 81.5 86.7 65.0 66.1 61.3 64.8 46.9 54.0 51.3 51.6 56.0 69.1

the maximum number of online optimization steps to N = 40. The LoRA rank is set to 16, the learning rate is 5×10−5, and the loss balance factor is set to λ = 0.5. We use a loss threshold of τL = 0.22 for early stopping, which approximately corresponds to an overall VLM confidence of 0.8 for answering “Yes” to the reward queries. After online optimization, the final video is generated by the optimized VGM Reasoner and decoded using the standard VAE. All compared open-source methods generate 89-frame videos under the same evaluation setting.

B. Comparison with SOTA Methods

Quantitative Comparisons. Tables I and II report the quantitative comparisons on VBVR-Bench and RULER-Bench, respectively. Notably, step distillation largely preserves the

reasoning performance of the backbone: it introduces only a 0.010 decrease on VBVR-Bench and a 0.3-point decrease on RULER-Bench, while reducing the generation cost from 87 s to 14 s and from 98 s to 18 s, respectively. This result suggests that effective video reasoning can be retained in a few-step Reasoner, which provides an efficient backbone for the proposed online optimization.

On VBVR-Bench, the proposed method improves the baseline by 0.115 overall, from 0.666 to 0.781, with consistent gains on both ID (+0.111) and OOD (+0.119) tasks. In comparison, at comparable test-time cost, Pass@5 provides only a 0.017 improvement, while “VLM-as-Solver” method VideoTPO decreases the overall score by 0.032. This gap is particularly pronounced on VBVR-Bench, where the structured

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

+VideoTPOBaselineKling2.6+Ours

+VideoTPOBaselineKling2.6+Ours

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

qgoal : By the end of the video, does the yellow ball reach the red square？ qproc1 : Does the yellow ball avoid all X crosses？ qproc2 : Does the yellow ball remain intact without splitting or vanishing？

qgoal : Do the three objects align with their dashed outlines at the end？ qproc1 : Do the shapes and colors of the three objects remain unchanged？ qproc2 : Are there always three objects throughout the video？

Prompt: The scene shows 3 objects on the left side and dashed target on

Prompt: The scene shows a grid with a yellow circular agent, a red end square and multiple black X marks indicating walls. The goal is to move the agent to the red square without entering any cells marked with black X.

the right side. Move each object horizontally to the right so that it aligns exactly cover and fits within its corresponding dashed target outline.

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

+VideoTPOBaselineKling2.6+Ours

+VideoTPOBaselineKling2.6+Ours

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

qgoal : By the end, has the green chair rotated 90°counterclockwise？ qproc1 : Does the green chair keep its features during rotation？ qproc2 : Do the plant and wall remain fixed in place？

qgoal : By the end of the video, does the hand have exactly five fingers？ qproc1 : Is the transformation natural and free of other anomalies？

Prompt: Rotate the green armchair 90 degrees counterclockwise around its base center while keeping its position and scale unchanged.

Prompt: Please fix the anomaly in the image so that the human anatomy conforms to normal proportions.

- Fig. 3: Qualitative comparisons on symbolic and general-purpose video reasoning examples. Baseline stands for the step-distilled

Wan2.2-5B model [3]. qgoal and qproc are representative supervision queries synthesized by the VLM Teacher. The proposed method satisfies both the final goal and the process constraints, leading to accurate reasoning results.

prompts already specify detailed task rules and target outcomes. Consequently, prompt refinement provides limited additional supervision, whereas the proposed method directly optimizes visual execution under the given rules. We do not evaluate PE on VBVR-Bench because the benchmark already provides carefully designed prompts that explicitly specify the task rules and target outcomes.

On RULER-Bench, the proposed method raises the average score of the baseline Reasoner from 46.4 to 68.2, yielding

a 21.8-point improvement. In contrast, PE, VideoTPO, and Pass@5 yield improvements of only 1.9, 3.9, and 2.7 points, respectively. More importantly, the proposed method consistently improves performance across all 30 evaluated task categories, whereas PE and VideoTPO decrease performance on 7 and 4 categories, respectively. Prompt-space methods remain effective on several tasks whose intended outcomes can be clarified through language or commonsense reasoning, such as Festival, Medicine, Life, and hypothetical state changes. However, their

benefits are less reliable on tasks that depend on precise visual execution. The proposed method achieves particularly substantial gains on such tasks, including Anomaly, Color, Count, and Direction, indicating that directly optimizing visual reasoning trajectories is more reliable than refining textual specifications alone.

The additional inference-time cost of our method remains manageable due to the efficient adaptation design. On VBVRBench, our method costs 69s per sample, which is comparable to Pass@5 at 70s and still lower than the original Wan2.25B baseline at 87s. On RULER-Bench, our method costs 88 s, comparable to Pass@5 at 90s and lower than the original Wan2.2-5B baseline at 98s. Under similar or even lower inference cost, the proposed method yields substantially gains than test-time scaling and VLM-as-Teacher paradigm, demonstrating a favorable cost–performance trade-off for perinstance optimization.

Qualitative Comparisons. Figure 3 presents qualitative comparisons on symbolic and general-purpose video reasoning tasks. Strong closed-source models such as Kling 2.6 can generate visually plausible videos, but it struggles to follow task-specific rules precisely. For example, in the object-moving task (upper-left of Figure 3), Kling 2.6 fails to place the objects into the correct dashed targets, and the blue square also changes its shape during the trajectory, violating the process constraint of preserving object identity.

The step-distilled baseline VGM exhibits more severe reasoning failures due to its smaller model capacity and weaker reasoning priors compared with large-scale closedsource models. In the maze-navigation task (upper-right), the yellow ball splits into multiple instances during the trajectory and therefore violates the entity-consistency constraint. In the anomaly-correction task (lower-right), the generated hand still contains six fingers, showing that the baseline fails to complete the intended correction.

VideoTPO, which refines the prompt using VLM feedback, does not effectively resolve these issues. In the maze example (upper-right), it still produces invalid intermediate trajectories with duplicated balls, and in the hand-correction example (lower-right), the anomaly remains uncorrected. These examples suggest that prompt refinement alone provides limited help when the main difficulty lies in precise visual execution rather than ambiguous task description.

In contrast, our proposed method consistently satisfies both the final-goal and process-constraint queries synthesized by the VLM Teacher. In the object-moving task (upper-left), it accurately aligns all objects with their corresponding dashed targets while preserving their shapes, colors, and cardinality. In the maze task (upper-right), it guides the yellow ball to the red square without crossing the blocked cells or introducing duplicated instances. In the chair-rotation task (lower-left), it correctly rotates the chair by 90◦ counterclockwise while preserving the chair appearance and keeping the surrounding plant and wall fixed. In the hand-correction task (lower-right), it gradually removes the anomaly and produces a realistic fivefinger hand. These examples show that directly optimizing the generated trajectory with process-aware and goal-aware

TABLE III: Ablation results on VBVR-Bench for fixed online optimization steps, reward design, and efficient adaptation.

Variants Overall ID Avg. OOD Avg.

Reward Design w/o Task-specific Online Optimization

+ Differentiable Reward 0.688 0.716 0.660 + Non-differentiable Reward [79] 0.681 0.707 0.655

w/o Task-specific Reward 0.712 0.739 0.685 w/o Process Reward 0.758 0.782 0.734 w/o Final Reward 0.692 0.718 0.666

###### Efficient Adaptation

w/o Step Distillation 0.714 0.739 0.689 w/ Last-step Optimization 0.705 0.713 0.698 w/ Full-step Optimization 0.769 0.792 0.746 Sample frames = 12 0.773 0.797 0.749 Sample frames = 48 0.782 0.805 0.759

Step = 0 0.666 0.692 0.640 Step = 5 0.710 0.735 0.685 Step = 10 0.750 0.774 0.726 Step = 16 0.781 0.803 0.759 Step = 20 0.783 0.804 0.762 Step = 40 0.778 0.800 0.756

Ours 0.781 0.803 0.759

supervision is more effective than relying on fixed generation or prompt-space refinement alone. Additional qualitative examples and video results are provided in our project page.

C. Ablation and Analysis

We conduct a series of ablation studies to analyze the proposed method from four perspectives. First, we examine the reward design, including the necessity of task-specific online optimization, task-adaptive reward synthesis, and the complementary roles of final-goal and process supervision. Second, we evaluate the efficient adaptation designs that make video-level test-time optimization practical, including step distillation, the choice of denoising step for supervision, the number of sampled frames, the lightweight decoder and the optimization budget by varying the number of online optimization steps. Finally, we analyze the generalization of the proposed method across different VLM Teachers and VGM backbones, and further discuss its remaining limitations. Please refer to our project page for video results.

Reward Design. The first block of Table III analyzes the design of the proposed supervision mechanism from three aspects. First, we examine whether the VLM reward should be used for instance-specific online optimization or shared post-training before inference. Replacing the proposed online optimization with shared post-training using differentiable VLM rewards decreases the overall score from 0.781 to 0.688. Using a nondifferentiable reward with Flow-GRPO [79] further decreases the score to 0.681. These results show that simply incorporating VLM feedback during post-training is insufficient; adapting the VGM Reasoner to the rules of each test instance is critical for video reasoning.

Second, we examine the importance of task-specific reward synthesis. Replacing the queries synthesized from each task condition with fixed generic queries, which only ask whether the goal is achieved and whether the process is valid, decreases

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]<br><br>| |
|---|
|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

Baselinew/o w/oqq Oursprocgoal

Baselinew/o w/oqq Oursprocgoal

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]<br><br>| |
|---|
|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]<br><br>| |
|---|
|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]<br><br>| |
|---|
|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]<br><br>| |
|---|
|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

qgoal : By the end, has the snail reach the wet area on the left？ qproc1 : Is the snail throughout the video the same snail as in the initial frame？

qgoal : Have all shapes reached their target positions by the end？ qproc1 : Do all the shapes maintain the same color and shape？

Prompt: Observe which area the snail moves toward over time.

Prompt: Keep all star markers unchanged. Move each colored object to the star marker with the same color using straight paths.

- Fig. 4: Qualitative analysis of reward-query ablations. In the left example, removing process supervision causes the red ball to split despite reaching the target region, while removing final-goal supervision prevents the shapes from reaching their target positions. In the right example, the snail is expected to move toward the moist region on the left; removing process supervision allows a shortcut trajectory by introducing another snail, while removing final-goal supervision fails to guide the snail to the correct region. Key failure areas are marked with red boxes.

the overall score from 0.781 to 0.712. This substantial drop demonstrates that video reasoning requires supervision tailored to each task’s specific goals and process constraints, rather than a shared set of generic reward queries.

We set the balance weight between final-goal and process supervision to λ = 0.5 by default, assigning equal importance to task completion and trajectory validity. We ablate the two components of the synthesized supervision by setting λ to 1 and 0, respectively. Removing process supervision decreases the score from 0.781 to 0.758, while removing final-goal supervision leads to a larger drop to 0.692. These results confirm that the two types of supervision serve complementary roles: final-goal supervision encourages successful task completion, whereas process supervision prevents invalid intermediate trajectories or shortcut solutions. Fig. 4 provides qualitative evidence for this distinction. In the symbolic example, removing final-goal supervision preserves the shapes more consistently during the intermediate process, but fails to guide them toward the required target positions. In the snail-moving example, removing process supervision reaches the target region through an invalid shortcut: a hand introduces another snail rather than moving the original one. In contrast, the full reward design satisfies both the intended final goal and reasoning process.

Efficient Optimization Designs. The second and third blocks of Table III evaluates the key designs that make online optimization efficient and effective. Removing step distillation decreases the overall score from 0.781 to 0.714. As shown in Fig. 5, without step distillation, the one-step prediction contains minimal task-relevant motion or state change, making it difficult for the VLM Teacher to judge whether the reasoning

task is being completed. In the symbolic example, the decoded square becomes blurry and ambiguous. In the rabbit example, the rabbit remains almost static, providing insufficient visual evidence for evaluating the intended action. In contrast, the step-distilled Reasoner produces a more perceptible one-step approximation of the reasoning trajectory, enabling effective VLM evaluation during online optimization.

We further ablate the denoising step used for optimization. Replacing the proposed first-step optimization with last-step optimization substantially decreases the overall score from 0.781 to 0.705. This indicates that optimizing only the final denoising step provides weak supervision for video reasoning, since the high-level motion pattern and task-relevant trajectory are largely determined in early denoising stages, while later steps mainly refine visual details. We also evaluate full-step optimization, where gradients are backpropagated through all four denoising steps. This variant achieves 0.769, which is still lower than the proposed first-step design. These results suggest that completing and optimizing the full denoising process is not necessary: the early prediction of the step-distilled Reasoner already exposes sufficient reasoning behavior for the VLM Teacher to provide effective supervision, while avoiding the additional cost and potential instability of backpropagating through the full sampling trajectory.

We then study the number of sampled frames used for VLM evaluation. Reducing the number of frames from 24 to 12 decreases the score to 0.773, suggesting that overly sparse sampling may miss important intermediate changes. Increasing the number to 48 obtains 0.782, only 0.001 higher than the default setting. We therefore use 24 frames as an effective

|[Figure 146]|
|---|

|[Figure 147]<br><br>| |
|---|
|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]<br><br>| |
|---|
|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

Vanilla

Vanilla

+VAE

+VAE Vanilla

|[Figure 156]|
|---|

|[Figure 157]<br><br>| |
|---|
|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]<br><br>| |
|---|
|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

+LightVAE

+LightVAE Stepdistilled

Vanilla

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]<br><br>| |
|---|
|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]<br><br>| |
|---|
|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

Stepdistilled

+VAE

+VAE Stepdistilled

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]<br><br>| |
|---|
|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]<br><br>| |
|---|
|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

Stepdistilled

+LightVAE

+LightVAE

Prompt: The scene shows two objects with a green attention box. Move the green attention box from the left object to the right object.

Prompt: Please fix the anatomical anomaly in the image so that the animal’s appearance aligns with natural characteristics.

- Fig. 5: Effects of step distillation and lightweight surrogate decoding. Without step distillation, the one-step prediction is blurry and ambiguous in the left symbolic example, and shows almost no visible motion in the right abnormal example, making VLM evaluation unreliable. Step distillation produces more task-informative one-step predictions, while the lightweight surrogate decoder largely preserves the key visual structures needed for supervision. Key differences are enlarged in red boxes.

[Figure 186]

RULER-BenchOverallScore

Video-MME Score of VLM Teacher

Qwen3-VL-8B

Qwen3-VL-4B

InternVL3-8B

𝑅2 = 0.733

- Fig. 6: Correlation between the video understanding capability of the VLM Teacher, measured on Video-MME, and the resulting performance on RULER-Bench.

TABLE IV: Generalization results on RULER-Bench with different VLM teachers and VGM backbones. ∗ denotes stepdistilled model.

Model Variant Overall Humanity Science Hypothesis Semantics Vision VLM Teacher

InternVL3-8B [81] 68.1 79.7 70.4 58.7 80.5 58.2 Qwen3-VL-8B [78] 69.2 80.7 71.5 59.5 81.6 59.4 Qwen3-VL-4B [78] 68.2 79.9 70.6 58.9 80.8 58.1

###### VGM Backbone

HunyuanVideo-1.5B∗ [82] 35.8 43.0 36.5 39.5 46.0 27.5

+ Ours 44.5 51.0 44.0 43.5 54.0 38.3 Wan2.2-5B∗[3] 46.4 55.5 47.5 50.3 58.6 36.1

###### + Ours 68.2 79.9 70.6 58.9 80.8 58.1

20 steps provides only a marginal gain of 0.002, while further increasing it to 40 steps slightly decreases the score to 0.778. These results indicate that the benefits of online optimization largely saturate after approximately 16 steps, while excessive optimization may over-optimize the VLM-based objective and introduce visual degradation.

Generalization across Teachers and Backbones. Table IV evaluates whether the proposed framework generalizes across different VLM Teachers and VGM backbones. Using Qwen3VL-4B as the default VLM Teacher, our method achieves an overall score of 68.2 on RULER-Bench. Replacing it with InternVL3-8B yields a comparable score of 68.1, while using Qwen3-VL-8B further improves the score to 69.2. Fig. 6 shows a strong positive correlation between the video understanding capability of the VLM Teacher, measured by Video-MME performance [80], and the resulting RULERBench performance, with R2 = 0.733. This result indicates that the proposed method is compatible with different VLM Teachers, while stronger video understanding generally leads to more effective supervision during online optimization.

trade-off between reasoning performance and VLM evaluation cost. Fig. 5 further shows that the lightweight surrogate decoder preserves the task-relevant visual structures required for VLM evaluation, despite moderate degradation in visual quality.

Finally, With loss-based early stopping, the proposed method performs only 16 online optimization steps on average on VBVR-Bench, achieving an overall score of 0.781 while avoiding unnecessary test-time overhead. To analyze the effective optimization budget, we disable early stopping and evaluate the model with different fixed numbers of online optimization steps. As shown in the first block of Table III, increasing the number of optimization steps from 0 to 16 steadily improves the overall score from 0.666 to 0.781. Extending optimization from 16 to

We further evaluate the proposed method with different VGM backbones. Applying our method improves the step-distilled

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

BaselineOurs

BaselineOurs

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]<br><br>| |
|---|
|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]<br><br>| |
|---|
|
|---|

qgoal : Does the pencil's painted body turn red？ qproc1 : Does the pencil change color without any external assistance？

qgoal : Is the shape in the bottom-right corner a diamond？ qproc1 : Does everything else remain unchanged throughout the process？

Prompt: Change the color of the pencil's painted body from teal to matte

Prompt: This is Raven's Progressive Matrices like task. Complete the missing pattern in this 3x3 matrix.

red while preserving other parts unchanged.

Fig. 7: Visualization of representative failure and limitation cases. Key failure regions are marked with red boxes.

HunyuanVideo-1.5B from 35.8 to 44.5 on RULER-Bench. The consistent improvements across both backbones demonstrate that the proposed method is not restricted to a specific VGM backbone.

Failure Cases and Limitations. Fig. 7 visualizes representative failure cases of the proposed method. In the RAVEN example, the VLM Teacher synthesizes an incorrect final-goal query by misidentifying the desired final configuration. The correct answer should contain two diamonds, but the synthesized query only checks whether the bottom-right shape is a diamond. As a result, online optimization is guided toward an incomplete objective, even though the generated trajectory satisfies the synthesized query. In the pencil example, the overall task is largely completed, but the VLM Teacher overlooks a subtle residual error: a small part of the pencil body is not fully transformed into red before the VLM loss falls below the stopping threshold. This illustrates that the proposed supervision can miss fine-grained local errors when they are not sufficiently perceived by the teacher.

To quantify the failure sources, we conduct a human evaluation on 200 generated cases, including 100 cases from VBVR-Bench and 100 cases from RULER-Bench. We count a case as a failure only when it violates the final goal or process constraints of the reasoning task. As shown in Table V, the baseline Reasoner fails on 39% of VBVR-Bench cases and 67% of RULER-Bench cases. In comparison, the proposed method reduces the failure rates to 18% and 29%, respectively, showing that VLM-guided test-time optimization substantially improves task completion and reasoning correctness. We further categorize the failure sources of the proposed method. Most remaining failures are caused by VLM perception errors, accounting for 16% and 22% of all evaluated cases on VBVRBench and RULER-Bench, respectively. These errors occur when the synthesized queries are correct, but the VLM Teacher overlooks fine-grained visual violations during evaluation. In contrast, incorrect reward-query synthesis accounts for only 2% and 7% of the cases, suggesting that the teacher usually derives reasonable task-specific supervision queries. These results indicate that the main limitation of the proposed method lies in the perception granularity of the VLM Teacher, rather than in the online optimization process itself.

We also observe a mild visual-quality trade-off. As shown in

TABLE V: Human evaluation of failure sources on 200 cases, including 100 cases from VBVR-Bench and 100 cases from RULER-Bench. A case is counted as a failure only when it violates the final goal or process constraints. Ratios are computed over the 100 evaluated cases in each benchmark.

VBVR-Bench RULER-Bench Count Ratio Count Ratio

Method Failure Source

Baseline Overall 39 39% 67 67%

VLM perception error 16 16% 22 22% Incorrect reward-query synthesis 2 2% 7 7% Overall 18 18% 29 29%

Ours

the qualitative results on VBVR-Bench in Fig. 3, the proposed method generally produces visually plausible trajectories, but the optimization process may occasionally introduce slight artifacts or reduce low-level visual fidelity. This is expected because our objective optimizes VLM-level rule satisfaction rather than pixel-level reconstruction quality. Importantly, such visual-quality degradation does not necessarily indicate task failure when the final goal and process constraints are satisfied. As reported in Table I, the proposed method substantially improves the VBVR-Bench overall score from 0.666 to 0.781, demonstrating stronger reasoning correctness in terms of final-goal achievement and process-constraint satisfaction. To quantify the visual-quality side effect, we report Fr´echet Video Distance (FVD) [83]. On VBVR-Bench, FVD slightly increases from 21.90 to 23.37; since lower FVD is better, this indicates a mild degradation in visual fidelity compared with the baseline Reasoner. Future work may further reduce this trade-off by incorporating lightweight visual-quality regularization during test-time optimization.

V. CONCLUSION

In this work, we introduce a VLM-as-Teacher paradigm for rule-based video reasoning, shifting the role of VLMs from producing textual solutions to supervising visual execution. Specifically, a VLM Teacher synthesizes task-specific reward queries that assess process-constraint satisfaction and final-goal achievement, and provides differentiable feedback to guide a VGM Reasoner through test-time online optimization. Together with efficient adaptation designs, the proposed method enables instance-specific refinement of visual reasoning trajectories

at practical test-time cost. Extensive experiments on the symbolic VBVR-Bench and the general-purpose RULERBench demonstrate consistent improvements across diverse reasoning tasks, yielding a 16.7-point average performance gain over the baseline Reasoner and substantially outperforming VLM-as-Solver and Best-of-N scaling strategies at comparable test-time cost. These results highlight the potential of using VLMs as test-time teachers to bridge high-level logic and visual execution in generative video reasoning.

ACKNOWLEDGEMENT

This work was supported by Kuaishou Technology and a grant from the NSFC/RGC Collaborative Research Scheme sponsored by the Research Grants Council of the Hong Kong Special Administrative Region, China and National Natural Science Foundation of China (Project No. CRSHKUST605/25).

REFERENCES

- [1] OpenAI, “Sora: Openai’s text-to-video model,” https://openai. com/index/sora-is-here, 2025, publicly released September 2025.
- [2] T. Seedance, D. Chen, L. Chen, X. Chen, Y. Chen, Z. Chen, Z. Chen, F. Cheng, T. Cheng, Y. Cheng et al., “Seedance 2.0: Advancing video generation for world complexity,” arXiv preprint arXiv:2604.14148, 2026.
- [3] WanTeam, “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025. [Online]. Available: https://arxiv.org/abs/2503.20314
- [4] J. Tong, Y. Mou, H. Li, M. Li, Y. Yang, M. Zhang, Q. Chen et al., “Thinking with video: Video generation as a promising multimodal reasoning paradigm,” arXiv preprint arXiv:2511.04570, 2025.
- [5] Z. Guo, X. Chen, R. Zhang, R. An, Y. Qi, D. Jiang, X. Li, M. Zhang, H. Li, and P.-A. Heng, “Are video models ready as zero-shot reasoners? an empirical study with the MME-CoF benchmark,” arXiv preprint arXiv:2510.26802, 2025. [Online]. Available: https://arxiv.org/abs/2510.26802
- [6] C. Li, Z. Wang, J. Li, Y. Xu, H. Zhou, H. Zhang, R. An, D. Jiang, Z. An, I. Vuli´c et al., “Thinking in frames: How visual context and test-time scaling empower video reasoning,” arXiv preprint arXiv:2601.21037, 2026.
- [7] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.
- [8] C. M. Bishop and N. M. Nasrabadi, Pattern recognition and machine learning. Springer, 2006, vol. 4, no. 4.
- [9] K. Newman, T. Zhu, and O. Russakovsky, “Video models reason early: Exploiting plan commitment for maze solving,” arXiv preprint arXiv:2603.30043, 2026.
- [10] J. Kim, S. Shin, J. Park, and E. Yang, “Collabvr: Collaborative video reasoning with vision-language and video generation models,” arXiv preprint arXiv:2605.08735, 2026.
- [11] H. H. Chen, D. Lan, W.-J. Shu, Q. Liu, Z. Wang, S. Chen, W. Cheng, K. Chen, H. Zhang, Z. Zhang, R. Guo, Y. Cheng, and Y.-C. Chen, “Tivibench: Benchmarking think-in-video reasoning for video generative models,” arXiv preprint arXiv:2511.13704,

2025. [Online]. Available: https://arxiv.org/abs/2511.13704

- [12] J. Cheng, L. Hou, X. Tao, and J. Liao, “Video-as-answer: Predict and generate next video event with joint-grpo,” arXiv preprint arXiv:2511.16669, 2025.
- [13] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models.” in International Conference on Learning Representations, 2022.

- [14] M. Wang, R. Wang, J. Lin, R. Ji, T. Wiedemer, Q. Gao, D. Luo, Y. Qian, L. Huang, Z. Hong et al., “A very big video reasoning suite,” arXiv preprint arXiv:2602.20159, 2026.
- [15] X. He, Z. Fan, H. Li, F. Zhuo, H. Xu, S. Cheng, D. Weng, H. Liu, C. Ye, and B. Wu, “RULER-bench: Probing rule-based reasoning abilities of next-level video generation models for vision foundation intelligence,” arXiv preprint arXiv:2512.02622,

2025. [Online]. Available: https://arxiv.org/abs/2512.02622

- [16] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6840–6851.
- [17] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205.
- [18] M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu, “Motiondiffuse: Text-driven human motion generation with diffusion model,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 6, pp. 4115–4128, 2024.
- [19] A. Polyak, A. Zohar, A. Brown, A. Tjandra, A. Sinha, A. Lee, A. Vyas, B. Shi, C. Ma, C. Chuang et al., “MovieGen: A cast of media foundation models,” arXiv preprint arXiv:2410.13720, 2024.
- [20] Google DeepMind, “Veo 3.1,” Google DeepMind, Tech. Rep., 2026, released January 13, 2026. [Online]. Available: https://blog.google/innovation-and-ai/technology/ai/ veo-3-1-ingredients-to-video/
- [21] Y. Gao, H. Guo, T. Hoang, W. Huang, L. Jiang, F. Kong, H. Li, J. Li, L. Li, X. Li et al., “Seedance 1.0: Exploring the boundaries of video generation models,” arXiv preprint arXiv:2506.09113, 2025.
- [22] T. Seedance, H. Chen, S. Chen, X. Chen, Y. Chen, Y. Chen, Z. Chen, F. Cheng, T. Cheng, X. Cheng et al., “Seedance 1.5 pro: A native audio-visual joint generation foundation model,” arXiv preprint arXiv:2512.13507, 2025.
- [23] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “CogVideoX: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.
- [24] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang et al., “HunyuanVideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.
- [25] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You, “Open-SORA: Democratizing efficient video production for all,” arXiv preprint arXiv:2412.20404, 2024.
- [26] Z. Huang, F. Zhang, X. Xu, Y. He, J. Yu, Z. Dong, Q. Ma, N. Chanpaisit, C. Si, Y. Jiang, Y. Wang, X. Chen, Y. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu, “VBench++: Comprehensive and versatile benchmark suite for video generative models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 3, pp. 3268–3285, 2026.
- [27] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding et al., “Cosmos world foundation model platform for physical ai,” arXiv preprint arXiv:2501.03575, 2025.
- [28] X. Zhang, J. Liao, S. Zhang, F. Meng, X. Wan, J. Yan, and Y. Cheng, “Videorepa: Learning physics for video generation through relational alignment with foundation models,” arXiv preprint arXiv:2505.23656, 2025.
- [29] J. Wang, A. Ma, K. Cao, J. Zheng, Z. Zhang, J. Feng, S. Liu, Y. Ma, B. Cheng, D. Leng et al., “Wisa: World simulator assistant for physics-aware text-to-video generation,” arXiv preprint arXiv:2503.08153, 2025.
- [30] Y. Chen, J. Cao, A. Kag, V. Goel, S. Korolev, C. Jiang, S. Tulyakov, and J. Ren, “Towards physical understanding in video generation: A 3d point regularization approach,” arXiv preprint arXiv:2502.03639, 2025.
- [31] Q. Xue, X. Yin, B. Yang, and W. Gao, “Phyt2v: Llm-guided iterative self-refinement for physics-grounded text-to-video gen-

- eration,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [32] T. Zhang, H.-X. Yu, R. Wu, B. Y. Feng, C. Zheng, N. Snavely, J. Wu, and W. T. Freeman, “Physdreamer: Physics-based interaction with 3d objects via video generation,” in European Conference on Computer Vision, 2024.
- [33] S. Liu, Z. Ren, S. Gupta, and S. Wang, “Physgen: Rigidbody physics-grounded image-to-video generation,” in European Conference on Computer Vision, 2024.
- [34] C. Gao, H. Zhang, Z. Xu, Z. Cai, and L. Shao, “Flip: Flowcentric generative planning as general-purpose manipulation world model,” arXiv preprint arXiv:2412.08261, 2024.
- [35] T. Xie, Y. Zhao, Y. Jiang, and C. Jiang, “Physanimator: Physicsguided generative cartoon animation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [36] A. Montanaro, L. Savant Aira, E. Aiello, D. Valsesia, and E. Magli, “Motioncraft: Physics-based zero-shot video generation,” in Advances in Neural Information Processing Systems, 2024.
- [37] Z. Wang, P. Hu, J. Wang, T. J. Zhang, Y. Cheng, L. Chen, Y. Yan, Z. Jiang, H. Li, and X. Liang, “Prophy: Progressive physical alignment for dynamic world simulation,” arXiv preprint arXiv:2512.05564, 2025.
- [38] S. Yuan, J. Huang, Y. Shi, Y. Xu, R. Zhu, B. Lin, X. Cheng, L. Yuan, and J. Luo, “MagicTime: Time-lapse video generation models as metamorphic simulators,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 9, pp. 7340–7351, 2025.
- [39] X. Liu, Z. Xu, M. Li, K. Wang, Y. J. Lee, and Y. Shang, “Can world simulators reason? Gen-ViRe: A generative visual reasoning benchmark,” arXiv preprint arXiv:2511.13853, 2025. [Online]. Available: https://arxiv.org/abs/2511.13853
- [40] T. Wiedemer, Y. Li, P. Vicol, S. S. Gu, N. Matarese, K. Swersky, B. Kim, P. Jaini, and R. Geirhos, “Video models are zero-shot learners and reasoners,” arXiv preprint arXiv:2509.20328, 2025. [Online]. Available: https://arxiv.org/abs/2509.20328
- [41] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 24824–24837.
- [42] Y. Qi, X. Xu, Z. Guo, S. Ma, R. Zhang, X. Chen, R. An, R. Xing, J. Zhang, H. Huang et al., “Mme-cof-pro: Evaluating reasoning coherence in video generative models with text and visual hints,” arXiv preprint arXiv:2603.20194, 2026.
- [43] R. Wang, Z. Cai, F. Pu, J. Xu, W. Yin et al., “Demystifying video reasoning,” arXiv preprint arXiv:2603.16870, 2026.
- [44] C. Yang, H. Wan, Y. Peng, X. Cheng, Z. Yu, J. Zhang, J. Yu, X. Yu, X. Zheng, D. Zhou, and C. Wu, “Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks,” arXiv preprint arXiv:2511.15065, 2025. [Online]. Available: https://arxiv.org/abs/2511.15065
- [45] Z. Cai, H. Qiu, T. Ma, H. Zhao, G. Zhou, K.-H. Huang, P. Kordjamshidi, M. Zhang, W. Xiao, J. Gu, N. Peng, and J. Hu, “MMGR: Multi-modal generative reasoning,” arXiv preprint arXiv:2512.14691, 2025. [Online]. Available: https://arxiv.org/abs/2512.14691
- [46] Y. Luo, X. Zhao, B. Lin, L. Zhu, L. Tang, Y. Liu, Y.-C. Chen, S. Qian, X. Wang, and Y. You, “V-reasonbench: Toward unified reasoning benchmark suite for video generation models,” arXiv preprint arXiv:2511.16668, 2025. [Online]. Available: https://arxiv.org/abs/2511.16668
- [47] A. Zhang, L. Lei, D. Kong, Z. Wang, J. Xu, F. Song, C.-L. Guo, C. Liu, F. Li, and J. Chen, “Ui2v-bench: An understandingbased image-to-video generation benchmark,” arXiv preprint arXiv:2509.24427, 2025.
- [48] X. Zhang, J. Wei, Y. Wang, J. Tan, Y. Li, Y. Zhang, Z. Chen, D. Zhang, D. Yu, W. Xu et al., “How far are video models from true multimodal reasoning?” arXiv preprint arXiv:2604.19193, 2026.

- [49] C. Snell, J. Lee, K. Xu, and A. Kumar, “Scaling LLM test-time compute optimally can be more effective than scaling model parameters,” arXiv preprint arXiv:2408.03314, 2024.
- [50] B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. R´e, and A. Mirhoseini, “Large language monkeys: Scaling inference compute with repeated sampling,” arXiv preprint arXiv:2407.21787, 2024.
- [51] N. Ma, S. Tong, H. Jia, H. Hu, Y.-C. Su, M. Zhang, X. Yang, Y. Li, T. Jaakkola, X. Jia, and S. Xie, “Inference-time scaling for diffusion models beyond scaling denoising steps,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [52] F. Liu, H. Wang, Y. Cai, K. Zhang, X. Zhan, and Y. Duan, “Video-t1: Test-time scaling for video generation,” in IEEE/CVF International Conference on Computer Vision, 2025.
- [53] H. He, J. Liang, X. Wang, P. Wan, D. Zhang, K. Gai, and L. Pan, “Scaling image and video generation via test-time evolutionary search,” arXiv preprint arXiv:2505.17618, 2025.
- [54] W. Cong, H. Zhu, P. Wang, B. Liu, D. Xu, K. Wang, D. Z. Pan, Y. Wang, Z. Fan, and Z. Wang, “Can test-time scaling improve world foundation model?” in Conference on Language Modeling, 2025.
- [55] C. Li, Z. Wang, J. Li, Y. Xu, H. Zhou et al., “Thinking in frames: How visual context and test-time scaling empower video reasoning,” arXiv preprint arXiv:2601.21037, 2026.
- [56] S. Jang, T. Ki, J. Jo, S. Xie, J. Yoon, and S. J. Hwang, “Selfrefining video sampling,” arXiv preprint arXiv:2601.18577, 2026.
- [57] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5625– 5644, 2024.
- [58] B. Li, Y. Zhang, L. Chen, J. Wang, F. Pu, J. A. Cahyono, J. Yang, C. Li, and Z. Liu, “Otter: A multi-modal model with in-context instruction tuning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 9, pp. 7543–7557, 2025.
- [59] Y. Liu, G. Li, and L. Lin, “Cross-modal causal relational reasoning for event-level visual question answering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 10, pp. 11624–11641, 2023.
- [60] T. Hui, S. Liu, Z. Ding, S. Huang, G. Li, W. Wang, L. Liu, and J. Han, “Language-aware spatial-temporal collaboration for referring video segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 8646–8659, 2023.
- [61] J. Cheng, Y. Ge, T. Wang, Y. Ge, J. Liao, and Y. Shan, “Video-holmes: Can mllm think like holmes for complex video reasoning?” arXiv preprint arXiv:2505.21374, 2025.
- [62] Y. Chen, Y. Ge, R. Wang, Y. Ge, J. Cheng, Y. Shan, and X. Liu, “Grpo-care: Consistency-aware reinforcement learning for multimodal reasoning,” arXiv preprint arXiv:2506.16141, 2025.
- [63] T.-H. Wu, L. Lian, J. E. Gonzalez, B. Li, and T. Darrell, “Self-correcting llm-controlled diffusion models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [64] L. Yang, Z. Yu, C. Meng, M. Xu, S. Ermon, and B. Cui, “Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms,” in International Conference on Machine Learning, 2024.
- [65] Y. Xiao, L. Song, Y. Chen, Y. Luo, Y. Chen, Y. Gan, W. Huang, X. Li, X. Qi, and Y. Shan, “Mindomni: Unleashing reasoning generation in vision language models with rgpo,” in Advances in Neural Information Processing Systems, 2025.
- [66] H. Lin, A. Zala, J. Cho, and M. Bansal, “Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning,” in Conference on Language Modeling, 2024.
- [67] X. Yang, B. Li, Y. Zhang, Z. Yin, L. Bai, L. Ma, Z. Wang, J. Cai, T.-T. Wong, H. Lu, and X. Jia, “Vlipp: Towards physically plausible video generation with vision and language informed physical prior,” in IEEE/CVF International Conference on Computer Vision, 2025, pp. 12360–12370.

- [68] X. Wang, Y. Zhang, O. Zohar, and S. Yeung-Levy, “Videoagent: Long-form video understanding with large language model as agent,” in European Conference on Computer Vision, 2024.
- [69] Z. Huang, N. Yu, G. Chen et al., “VChain: Chain-of-visualthought for reasoning in video generation,” arXiv preprint arXiv:2510.05094, 2025.
- [70] T. Zhu, S. Zhang, J. Y. Huang, S. Song, X. Wen, Y. Li, H. Poon, and M. Chen, “Video models can reason with verifiable rewards,” arXiv preprint arXiv:2605.15458, 2026.
- [71] G. Luo, J. Granskog, A. Holynski, and T. Darrell, “Dual-process image generation,” in IEEE/CVF International Conference on Computer Vision, 2025, pp. 17972–17983.
- [72] N. Kumari, S.-Y. Wang, N. Zhao, Y. Nitzan, Y. Li, K. K. Singh, R. Zhang, E. Shechtman, J.-Y. Zhu, and X. Huang, “Learning an image editing model without image editing pairs,” arXiv preprint arXiv:2510.14978, 2025.
- [73] Y. Wang, Y. Li, S. Tulyakov, Y. Fu, and A. Kag, “Diffusion-drf: Differentiable reward flow for video diffusion fine-tuning,” arXiv preprint arXiv:2601.04153, 2026.
- [74] L. Contributors, “Lightx2v: Light video generation inference framework,” https://github.com/ModelTC/lightx2v, 2025.
- [75] T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and W. T. Freeman, “Improved distribution matching distillation for fast image synthesis,” in Advances in Neural Information Processing Systems, 2024.
- [76] OpenAI, “Openai o3 and o4-mini system card,” OpenAI, Tech. Rep., Apr. 2025.
- [77] Kuaishou Technology, “Kling AI launches video 2.6 model with “simultaneous audio-visual generation” capability, redefining AI video creation workflow,” Press Release, Kuaishou Technology, Dec. 2025, model released December 3, 2025. Press release published December 5, 2025.
- [78] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.
- [79] J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang, “Flow-grpo: Training flow matching models via online rl,” in Advances in Neural Information Processing Systems, 2025.
- [80] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 24108–24118.
- [81] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.
- [82] T. H. F. M. Team, “Hunyuanvideo 1.5 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.18870
- [83] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Fvd: A new metric for video generation,” 2019.

