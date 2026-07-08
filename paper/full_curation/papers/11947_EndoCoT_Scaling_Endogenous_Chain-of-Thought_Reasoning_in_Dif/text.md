## arXiv:2603.12252v4[cs.CV]18Jun2026

[Figure 1]

2026-6-19

# EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models

###### Xuanlang Dai1,2, Yujie Zhou1,3, Long Xing1,4, Jiazi Bu1,3, Xilin Wei1,5, Yuhong Liu1,3, Beichen Zhang1,6, †Kai Chen1 and †Yuhang Zang1

1Shanghai AI Laboratory, 2Xi’an Jiaotong University, 3Shanghai Jiaotong University, 4University of Science and Technology of China, 5Fudan University, 6The Chinese University of Hong Kong

Recently, Multimodal Large Language Models (MLLMs) have been widely integrated into diffusion frameworks primarily as text encoders to tackle complex tasks such as spatial reasoning. However, this paradigm suffers from two critical limitations: (i) MLLMs text encoder exhibit insufficient reasoning depth. Single-step encoding fails to activate the Chain-of-Thought process, which is essential for MLLMs to provide accurate guidance for complex tasks. (ii) The guidance remains invariant during the decoding process. Invariant guidance during decoding prevents DiT from progressively decomposing complex instructions into actionable denoising steps, even with correct MLLM encodings. To this end, we propose Endogenous Chain-of-Thought (EndoCoT), a novel framework that first activates MLLMs’ reasoning potential by iteratively refining latent thought states through an iterative thought guidance module, and then bridges these states to the DiT’s denoising process. Second, a terminal thought grounding module is applied to ensure the reasoning trajectory remains grounded in textual supervision by aligning the final state with ground-truth answers. With these two components, the MLLM text encoder delivers meticulously reasoned guidance, enabling the DiT to execute it progressively and ultimately solve complex tasks in a step-by-step manner. Extensive evaluations across diverse benchmarks (e.g., Maze, TSP, VSP, and Sudoku) achieve an average accuracy of 92.1%, outperforming the strongest baseline by 8.3 percentage points.

###### 1. Introduction

Diffusion models [12, 17, 25, 26] have revolutionized visual generation, achieving unprecedented levels of photorealism and diversity. Recent efforts integrate Multimodal Large Language Models (MLLMs) [3, 5, 18, 32] as text encoders to augment semantic understanding and logical alignment. Yet despite these advances, they struggle with tasks requiring step-by-step logical reasoning, such as solving mazes, planning Traveling Salesperson Problem (TSP) routes, or completing Sudoku puzzles, where solutions must adhere to strict sequential constraints and generalize to novel configurations.

Current paradigms relegate MLLMs to the role of static conditional encoders, computing text embeddings once at the beginning of generation. This passive integration fails to exploit the dynamic reasoning potential of MLLMs. Even recent explicit attempts like DiffThinker [11] to “inject” reasoning into diffusion models result in superficial alignment rather than genuine cognitive processing. While previous work [11] reports improved metrics on specific benchmarks, they produce fragile solutions that fail catastrophically when generalized to novel domains, see Fig. 1(b). Our analysis reveals why these failures occur: despite being built on powerful MLLMs, these models do not actually perform reasoning during generation. Instead, as shown in in Fig. 1(c), they commit to their final solution within the first few denoising steps and merely refine visual quality thereafter. This suggests that without an endogenous mechanism, forced alignment results in fragile pattern matching rather than robust, iterative problem-solving.

Motivated by these persistent failures in supervised fitting, we conduct a systematic empirical analysis

† Corresponding authors: Kai Chen ( chenkai@pjlab.org.cn@pjlab.org.cn), Yuhang Zang (zangyuhang@pjlab.org.cn)

* Code is at https://github.com/InternLM/EndoCoT

###### (a) Overall Performance

(b) Generalization Task

96.67 92.00

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Generalization SudokuFont

MazeSize

88.33

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

75.67

[Figure 16]

89.50

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Input Target

Standard Size Novel Size

Standard Size Novel Size

69.00

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Generalization

99.57

64.75

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

99.57 89.75

[Figure 40]

Input Target Standard Font

Novel Font Standard Font Novel Font

Training Examples DiffThinker Ours

(c) Latent Reasoning Chain

[Figure 41]

###### Vanilla Denoising Ours

With Reasoning

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

I can fill (2,8) with 4 ...

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

No Reasoning

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

I can fill (9,1) with 5 ...

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Step1 Step2 Step5 Target

(Only partial intermediate results are displayed.)

- Figure 1: EndoCoT enables endogenous chain-of-thought reasoning. (a) Radar plot showing EndoCoT outperforms baselines across all benchmarks. (b) On visual reasoning tasks requiring generalization (maze size, Sudoku font), previous work [11] fails on novel domains while EndoCoT consistently generalizes correctly. (c) Vanilla denoising (left) commits to solutions early without reasoning, while our approach (right) enables interpretable, step-by-step reasoning chains.

(Sec. 3) that identifies two key bottlenecks: (1) limited single-step reasoning: MLLMs cannot encode all necessary logical constraints in a single forward pass; (2) static-guidance failure: Diffusion Transformers (DiTs) cannot maintain alignment with complex logical constraints when conditioned on static embeddings. Chain-of-Thought (CoT) reasoning [31] has shown how to overcome similar limitations in MLLMs [2, 35] through iterative refinement, but diffusion models currently lack an analogous mechanism.

[Figure 75]

To this end, we propose Endogenous CoT (EndoCoT), a framework that enables diffusion models to perform self-guided reasoning by iteratively exploring the semantic latent space. Specifically, EndoCoT encompasses two primary components: (1) Iterative Thought Guidance: iteratively updates latent states in the MLLM to create a genuine CoT-like reasoning process and establishes correspondence with DiT’s denoising process. (2) Terminal Thought Grounding: aligns the MLLM’s final reasoning state with ground-truth answers, ensuring the reasoning trajectory remains grounded in textual supervision and preventing cumulative drift in the terminal output. Consequently, EndoCoT serves as a complete end-to-end pipeline, seamlessly integrating reasoning and generation within a unified diffusion framework. Experiments are conducted across four diverse reasoning tasks: Maze, TSP, Visual Spatial Planning (VSP), and Sudoku, demonstrating consistently superior performance over state-of-the-art baselines including DiffThinker [11] and Qwen3-VL-8B [3]. As summarized in Fig. 1(a), EndoCoT outperforms baselines across all benchmarks. Moreover, as task complexity scales, EndoCoT maintains exceptional accuracy: it achieves 90% on Maze-32 and 95% on Sudoku-35, outperforming the strongest baseline by 25% and 40% respectively. Unlike vanilla baselines that commit early and fail catastrophically, EndoCoT exhibits interpretable, step-by-step reasoning chains (see Fig. 1(c)). Our main contributions are summarized as follows:

- 1. To the best of our knowledge, we present the first diffusion framework that enables genuine chain-ofthought reasoning via iterative latent state refinement, rather than pre-computing solutions in a single pass.
- 2. Through comprehensive layer-wise sensitivity and attention entropy analysis, we localize where reasoning

- arises in diffusion models and identify the two key bottlenecks that limit prior methods (Sec. 3).
- 3. EndoCoT achieves 25-40% gains over prior work on complex visual reasoning benchmarks, enables controllable inference-time scaling, and yields clearer and more controllable transformation trajectories in image editing tasks.

###### 2. Related Work

Reasoning in Multimodal Large Language Models Chain-of-thought (CoT) and other test-time scaling strategies [4, 36, 39, 43] have proven effective in autoregressive large language models (LLMs). Recent works [1, 34] extend this paradigm to multimodal settings. OpenAI [19] introduces “Think with Images”. Subsequent works [6, 20, 23, 28, 30, 37] further extend this line of research to “Thinking with Video”, using visual content as external evidence to support multi-step reasoning. Latent Sketchpad [38] proposes an interleaved autoregressive generation of text and visual latents. However, enabling reasoning in diffusion models (DMs) remains challenging. Since DMs typically rely on fixed-length text encoders [21, 22], their capacity to incorporate multi-step reasoning into generation is limited. Recent works [32] introduce vision-language models (VLMs) as encoders, providing richer semantic representations that facilitate reasoning-conditioned generation in diffusion models.

Reasoning in Diffusion Model Several works [15, 16] explore incorporating reasoning signals into DMs by injecting textual reasoning traces into the conditioning inputs. While those approaches enable reasoning-aware generation, the MMDiT is often treated as a conditional decoder. This over-reliance on VLMs leads to a decoupled pipeline in which the MLLM mainly acts as a prompt enhancer. Recently, approaches [29, 33, 40] have leveraged video priors to perform complex editing, essentially treating logical state transitions as temporal sequences. However, these methods rely on the inherent smoothness of video models to “reason” through changes, which may not translate to discrete logical tasks. DiffThinker [11] takes a step toward internalizing reasoning by exploring the reasoning potential of MMDiT directly. Works like [7, 9] have attempted visual generation under the next-token autoregressive paradigm. Despite these advancements, existing frameworks still cannot perform genuine, iterative chain-of-thought reasoning within the diffusion process itself. In this work, we enable endogenous CoT reasoning through iterative latent state refinement, achieving robust logical planning and spatial grounding.

Latent Reasoning Latent reasoning has been validated in text-only domains, enabling multi-step reasoning in continuous latent spaces [10, 27, 41]. Latent-space reasoning compresses long reasoning chains and supports tree-structured exploration, improving inference efficiency and diversity [24, 42]. Motivated by this line of work, we train diffusion models end-to-end with latent tokens to enable test-time scaling within the diffusion process.

###### 3. Analysis of Reasoning in Diffusion Models

To understand why current diffusion models fail at complex visual reasoning despite integrating powerful MLLMs, we first conduct a systematic empirical analysis. These analyses directly motivate the design of our EndoCoT framework and reveal fundamental bottlenecks that limit performance. All experiments are conducted on Qwen-Image-Edit-2511 [32].

Layer-Wise Sensitivity Analysis. To investigate the internal dynamics of how reasoning capabilities emerge, we conduct a layer-wise sensitivity analysis on a representative image editing model. Specifically, we quantify the relative sensitivity by measuring the magnitude of activation responses across the distinct layer blocks of the model’s three core components: the Vision Encoder, the Language Model, and the Diffusion Transformer.

As shown in Fig. 2(a), peak sensitivity is highly concentrated within the Vision Encoder and precisely at the architectural junction between the LLM (terminal layers) and the DiT (initial layers). This localized activation yields a critical insight: the heavy lifting of logical reasoning is predominantly handled by the MLLM. This observation strongly suggests that activating the MLLM during training is essential to fully unleash the model’s visual reasoning potential. This finding directly informs our design choice to jointly fine-tune both the MLLM and DiT components in EndoCoT.

(a) Layer-Wise Sensitivity Analysis

[Figure 76]

[Figure 77]

(c) Static-Guidance Failure

(b) Limited Single-Step Reasoning Depth in MLLMs

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Fragile cross-modal coupling leads to disorganized cooperative patterns, resulting in the incorrect representation.

The model displays insufficient reasoning capacity, failing to correctly trace logical path

Fail Fail

- Figure 2: (a) Layer-wise sensitivity across Vision Encoder, LLM, and DiT components (red: high sensitivity, white: low sensitivity). (b) Limited single-step reasoning: DiT performs spatial grounding but trajectory violates constraints. (c) Static-guidance failure: Dense topologies cause attention entropy to become diffuse.

Limited Single-Step Reasoning. While the MLLM handles the bulk of logical reasoning, we observe that a single forward pass through the MLLM is insufficient for complex tasks. As depicted in Fig.2(b), in lower-complexity scenarios (e.g., 8×8 mazes), the DiT successfully focuses on the generated trajectory. However, the generated paths systematically violate physical constraints (e.g., passing through walls). This indicates that while the DiT successfully executes spatial grounding effectively in simple tasks, the MLLM lacks the capacity to fully encode and enforce all logical constraints in a single pass. The MLLM needs to iteratively refine its understanding of the problem, rather than attempting to compute the complete solution in one shot. This motivates our design of multi-round reasoning in the latent space.

Static-Guidance Failure in Complex Dynamic DiT Decoding. Even if the MLLM could produce perfect reasoning in a single step, we observe a second critical limitation: the information coupling between the DiT and MLLM is fragile and static. To diagnose this, we analyze the cross-attention entropy between the generated spatial patches and the ground-truth reasoning tokens during the denoising process in Fig. 2(c). In high-complexity scenarios (e.g., 32×32 mazes), the attention entropy map becomes globally high and diffuse. This reveals a severe breakdown in the coupling between the DiT and the MLLM. When faced with dense spatial topologies, the DiT loses its ability to anchor spatial features to specific logical text tokens, causing the attention distribution to average out and resulting in a complete collapse of spatial grounding. The static, one-time injection of text embeddings is insufficient, and the conditioning signal needs to be dynamically updated throughout the reasoning process.

Summary. Our analysis across Fig. 2(a-c) highlights three key insights: (1) The MLLM has strong reasoning potential, but cannot fully exploit it in a single forward pass; (2) The DiT excels at spatial grounding, but requires dynamic, evolving conditioning to maintain alignment with complex logical constraints; (3) The current paradigm of static, one-time text injection is limited for multi-step reasoning tasks. This insight motivates our iterative latent reasoning mechanism, which we present in the next section.

[Figure 88]

(a) Training Stage

(b) Inference Stage

[Figure 89]

[Figure 90]

Maze VSP TSP Sudoku

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

###### Legend

[Figure 99]

[Figure 100]

Prefix Embeds

[Figure 101]

Conditional Generation

[Figure 102]

Implicit Tokens

GT Embeds

MMDiT

###### MMDiT

[Figure 103]

Result Embeds

LORA

###### * N

Implicit Reasoning

Progressive Thinking

[Figure 104]

Semantic Loss

[Figure 105]

MLLM

MLLM

[Figure 106]

LORA

[Figure 107]

[Figure 108]

L R U D

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Draw a continuous red line starting from the yellow dot towards the blue dot ... DDLUULUL ...

[Figure 121]

Draw a continuous red line starting from the yellow dot towards the blue dot ...

[Figure 122]

[Figure 123]

Embedding Layer

Embedding Layer

[Figure 124]

[Figure 125]

- Figure 3: Overview of EndoCoT. (a) Training: We propose a progressive two-stage training strategy: the first stage trains the model to fit both intermediate and final states at each reasoning step, capturing the full multi-step trajectory; the second stage freezes gradients on intermediate states and optimizes only the terminal state, refining generation quality while preserving learned reasoning dynamics (b) Inference: the model iteratively updates latent representations.

###### 4. Methods

Building on our analysis in Sec. 3, we present Endogenous Chain-of-Thought (EndoCoT), a framework that enables diffusion models to perform self-guided reasoning during generation. As shown in Fig. 3, our approach iteratively refines latent thought states in the MLLM to create a CoT reasoning process, aligns these reasoning states with explicit textual supervision, and uses progressive training to first build reasoning capability, then refine output quality.

We begin with necessary preliminaries on flow matching, then present our core methodology in three parts:

(1) iterative thought guidance module, (2) terminal thought grounding, and (3) progressive training strategy. We conclude with the full inference algorithm.

###### 4.1. Preliminary

Flow Matching (FM). Flow Matching [17] provides a simplified framework for generative modeling by constructing a linear probability path between two distributions. Let 𝑋0 ∼ 𝜋0 denote a clean data sample and 𝑋1 ∼ 𝜋1 represent Gaussian noise from the prior. A linear trajectory 𝑋𝑡 interpolates between 𝑋0 and 𝑋1 over a continuous time step 𝑡 ∈ [0,1]:

𝑋𝑡 = 𝑡𝑋1 + (1 − 𝑡)𝑋0. (1) By differentiating 𝑋𝑡 with respect to time step 𝑡, we obtain the ground-truth vector field 𝑢𝑡(𝑋𝑡):

𝑑𝑋𝑡 𝑑𝑡

= 𝑋1 − 𝑋0. (2)

𝑢𝑡(𝑋𝑡) =

A neural network 𝑣𝜃(𝑋𝑡,𝑡,𝑐) approximates this vector field, where 𝑐 is an optional conditioning variable. The training objective ℒFM is :

0∼𝜋0,𝑋1∼𝜋1 ‖𝑣𝜃(𝑋𝑡,𝑡,𝑐) − 𝑢𝑡(𝑋𝑡)‖2 . (3)

ℒFM = E𝑡,𝑋

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Text Encoder

GT Solution

[Figure 130]

Final Thought

[Figure 131]

Reference State

[Figure 132]

[Figure 133]

Input Image

MLLM

Prefix Embeds

[Figure 134]

Condition Signal

Current Thoght

... Diffusion Transformer

[Figure 135]

在此处键入公式。

[Figure 136]

Previous Thought

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Noise Target Image Diffusion Process

[Figure 145]

[Figure 146]

Latent Reasoning (MLLM) Conditional Generation (DiT）

- Figure 4: Overview of notations and iterative thought guidance module. EndoCoT iteratively refines latent states h𝜏 through the MLLM 𝑓𝜑, then conditions the DiT 𝑓𝜓 at each reasoning step 𝜏 to generate intermediate visual outputs I𝜏.

###### 4.2. EndoCoT

Current diffusion models exhibit limited reasoning capabilities for complex, multi-step tasks. Recent analyses suggest these models possess reasoning potential matching their parameter scale. Architectural advances in text encoders [5, 32] provide the structural foundation to unlock this potential. We propose Endogenous Chain-of-Thought (EndoCoT), a framework enabling diffusion models to perform autonomous chain-of-thought reasoning.

###### 4.2.1. Iterative Thought Guidance.

We formulate the iterative thought guidance process as an iterative refinement of conditional latent states. Standard diffusion models generate an output by conditioning on static text embeddings. In contrast, our approach performs 𝒯 reasoning steps, where each step refines both the visual output and the conditioning signal.

Intuitively, this process mimics human problem-solving: rather than attempting to generate the complete solution in one pass, we iteratively refine our understanding of the problem and proposed solution. Each reasoning step builds upon the previous one, allowing the model to explore solution spaces incrementally.

Let 𝑓𝜑 denote the MLLM (excluding embedding and projection layers) and 𝑓𝜓 denote the Diffusion Transformer (DiT). A summary of the main notations is provided in Fig. 4. The reasoning process is formulated as a sequence of hidden state updates in the latent manifold R𝑑. P ∈ R𝐿×𝑑 represents the prefix embeddings obtained by encoding the textual prompt and input image through their respective embedding layers. Given fixed prefix embeddings P (where 𝐿 is the sequence length of the textual prompt), the 𝜏-th reasoning step (𝜏 ∈ {1,...,𝒯 }) updates the thought state h𝜏 ∈ R𝑑 recursively:

h𝜏 = e⊤𝐿+1𝑓𝜑 ([P;h𝜏−1]), 𝜏 = 1,...,𝒯 , (4)

where [·;·] denotes concatenation along the sequence dimension, and e𝐿+1 is the one-hot basis vector used to extract the hidden state at the (𝐿+1)-th sequence position. Crucially, h𝜏−1 directly serves as a high-dimensional input to the first layer of 𝑓𝜑, bypassing the discrete embedding lookup table.

Conditional Flow Generation. Note that our reasoning steps 𝒯 are distinct from the diffusion model’s internal denoising timesteps. Each reasoning step 𝜏 involves a complete denoising trajectory from noise to image, conditioned on the current thought state h𝜏. At each reasoning step 𝜏, we condition the diffusion model on the

current thought state h𝜏 to generate an intermediate visual output I𝜏 by solving the flow ODE:

𝑑z𝜏(𝑡) 𝑑𝑡

= 𝑣𝜓(z𝜏(𝑡),𝑡,h𝜏), z𝜏(1) ∼ 𝒩(0,I), I𝜏 = z𝜏(0), (5)

where z𝜏(𝑡) denotes the latent state at flow time 𝑡 ∈ [0,1]. We optimize the model by supervising the generated output against a ground-truth intermediate target I*𝜏, using the Conditional Flow Matching loss based on Eq. 3:

[︀‖(z𝜏(0) − z𝜏(1)) − 𝑣𝜓(z𝜏(𝑡),𝑡,h𝜏)‖2]︀

. (6)

ℒreasoning = E𝜏,𝑡,z

𝜏(0),z𝜏(1)

The intermediate targets I*𝜏 can be obtained through sequential ground-truth decomposition (e.g., partial path segments for mazes).

Motivated by our analysis in Sec. 3, we apply LoRA fine-tuning to both 𝑓𝜑 and 𝑓𝜓. This joint adaptation enables collaborative reasoning across the conditioning and generation stages.

###### 4.2.2. Terminal Thought Grounding.

While the reasoning-enhanced framework enables iterative visual refinement, its reliance on purely visual supervision introduces two primary challenges: (1) a modality gap between visual targets and latent reasoning states, and (2) underutilization of explicit textual ground truth. Inspired by the success of explicit supervision in text-based latent reasoning, we introduce an auxiliary alignment objective that grounds the final reasoning state with textual supervision:

gt+1𝑓𝜑 ([Pgt,Iinput]), (7) where Pgt ∈ R𝐿

href = e⊤𝐿

gt×𝑑 is the embedding of the ground-truth reasoning steps, and Iinput denotes the task input (e.g., initial maze configuration). We align the final reasoning state h𝒯 with this reference using an L2 (Semantic) loss:

ℒalign = ‖h𝒯 − href‖2. (8) The overall training loss combines the flow matching loss with this alignment term:

ℒtotal = ℒFM + I{𝜏=𝒯 } · 𝜆alignℒalign, (9)

where I{𝜏=𝒯 } is the indicator function activating the alignment loss only at the final reasoning step, and 𝜆align balances visual generation quality and textual grounding. Empirically, we set 𝜆align = 1. As shown in our ablation study (Sec. 5.2), this term is critical for preventing drift in the latent reasoning states and ensuring the reasoning trajectory remains grounded.

###### 4.2.3. Progressive Training Strategy

Reasoning steps and final outputs serve distinct objectives: intermediate steps focus on exploring solution paths, while the final step emphasizes producing a correct answer. Directly optimizing both objectives simultaneously can lead to conflicting gradients. We therefore adopt a two-stage progressive training strategy.

- Stage 1: Reasoning Development. As shown in Fig. 3 (a), we supervise all reasoning steps 𝜏 = 1,...,𝒯 , enabling the model to learn step-by-step visual reasoning. The training objective is:

ℒstage1 =

∑︁𝒯

𝜏=1

(︀ℒ𝜏FM + I{𝜏=𝒯 }𝜆alignℒalign)︀

, (10)

where ℒ𝜏FM denotes the flow matching loss at reasoning step 𝜏. By providing supervision at each intermediate step, we encourage the model to develop coherent, incremental reasoning trajectories.

- Stage 2: Terminal Consolidation. Once the model has developed robust reasoning capabilities, we shift the training focus to solidifying the visual accuracy of the final output. As shown in Fig. 3 (b), while the intermediate reasoning process is preserved during the forward pass, gradients are computed exclusively for the final output:

ℒstage2 = ℒ𝒯FM + 𝜆alignℒalign. (11)

Input Step 1 Step 2 Step 3 Result

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

SudokuMazeTSPVSP

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

- Figure 5: Step-by-step reasoning process across four distinct tasks. Our model incrementally resolves complex visual reasoning tasks through intermediate reasoning steps. For each task (Maze, TSP, Sudoku, VSP), we show the initial input (leftmost), intermediate refinement steps, and the final optimal solution (rightmost).

The intermediate steps 𝜏 < 𝒯 are executed in the forward pass only, serving as reasoning scaffolding without gradient propagation. To prevent the degradation of the previously learned reasoning chains, Stage 2 employs a short-cycle fine-tuning strategy with limited iterations.

###### 4.2.4. Inference Process

As shown in Fig. 3(b), EndoCoT does not require decoding intermediate visual states during inference. By specifying the number of reasoning steps 𝒯 , the model recursively updates its latent thought states to generate the final result.

###### 5. Experiments

Model Setup. We build upon Qwen-Image-Edit-2511 [32] and apply LoRA fine-tuning [13] to efficiently adapt the model while preserving pre-trained knowledge. We use a LoRA rank of 32 for all targeted modules, a learning rate of 1 × 10−4, and train for 5 epochs. Detailed hyperparameter and LoRA module configuration are provided in the Appendix.

###### 5.1. Results on Visual Reasoning Tasks

Datasets. Following the evaluation protocol of [11], we select Maze navigation (Maze), Traveling Salesman Problem (TSP), Sudoku solving (Sudoku), Visual Spatial Planning (VSP), and VSP-Pro with larger map sizes (up to 32 × 32) as evaluation benchmarks. Each task requires multi-step logical reasoning. We also follow [11]

- Table 1: Evaluation results across visual reasoning tasks. Our model establishes a new state-of-the-art across both task-specific and unified training settings, demonstrating exceptional robustness to increasing task complexity.

Models

Maze TSP Sudoku VSP VSP-Super

Avg 8 16 32 12 15 18 45 40 35 30 3 4 5 6 7 8 16 32

▼ Zero-Shot ThinkGen [15] 0 0 0 0 0 0 0 0 0 0 44 4 11 13 9 10 0 0 5.1 ChronoEdit [33] 1 1 26 0 0 0 0 0 0 0 60 20 12 7 16 13 1 1 8.8 Qwen3-VL-8B [35] 1 0 0 0 0 0 0 0 0 0 64 46 33 21 12 21 1 0 11.1 Qwen-Image-Edit-2511 [32] 0 0 0 0 0 0 0 0 0 0 50 55 44 16 23 23 0 0 11.7

▼ Task-Specific Training Qwen3-VL-8B(SFT) [11] 53 37 0 59 60 43 30 17 2 0 99 96 98 96 92 86 61 8 58.56 Qwen3-VL-8B(GRPO) [11] 0 0 0 0 0 0 0 0 0 0 91 70 70 31 34 24 0 0 17.8 DiffThinker [11] 100 100 65 76 72 59 97 94 55 13 100 100 100 98 100 100 99 80 83.8 Ours 100 100 90 77 77 73 100 100 95 64 100 100 100 99 100 99 99 85 92.1

▼ Unified Training

DiffThinker [11] 98 99 66 64 49 34 94 45 26 37 100 99 99 99 100 97 97 84 77.1 Ours 97 98 52 64 55 46 100 88 80 59 100 98 98 100 100 100 100 80 84.2

- Table 2: Ablation Study on the Maze Benchmark. Removing the auxiliary semantic loss or using explicit tokens both lead to significant performance degradation compared to our default choice.

Table 3: Performance comparison across different maze scales. Our method achieves near-perfect accuracy and correct path repetition rate, significantly surpassing the baselines.

Accuracy (ACC) Path Repetition (%) 8 16 32 8 16 32

Accuracy Path Repetition (%) 8 16 32 8 16 32

Models

Models

w/o semantic loss 39 42 14 93.44 92.23 67.24 Explicit token 34 8 0 81.47 33.35 0.08

MLLM-Only 0 0 0 0.00 0.62 0.20 DiT-Only 57 43 18 89.75 90.80 80.96 Ours 100 100 90 100.00 100.00 98.13

Ours 100 100 90 100.00 100.00 98.13

to construct the fine-tuning data of each task. All generated data are rendered at a resolution of 512 × 512. Detailed statistics and construction methods of fine-tuning data are provided in the Appendix.

Settings. We evaluate models under three distinct settings. 1) Zero-shot: Vanilla baselines without any taskspecific training. 2) Task-specific training: Training and evaluation are performed on individual tasks separately (e.g., train on the Maze dataset, evaluate on Maze only), which is the default setting of DiffThinker [11]. 3) Unified training: A single model is trained on the combined dataset of all tasks (Maze+TSP+Sudoku+VSP), and evaluated on all tasks simultaneously. The ‘unified training’ setting is more challenging as it requires the model to handle a wide range of problems.

Results of Task-Specific Training. As shown in Tab. 1, our model establishes a new state-of-the-art across all evaluated visual reasoning benchmarks under task-specific training. Our approach demonstrates exceptional robustness to increasing complexity. At higher difficulty levels like Sudoku (scale 35) and Maze (scale 32), it achieves 95% and 90% accuracy respectively, significantly outperforming DiffThinker [11] (55% and 65%). On spatial planning tasks, our model maintains near-perfect scores across standard VSP levels and reaches 85% on the most challenging VSP-Super (scale 32), whereas generative baselines like ThinkGen [15] and ChronoEdit [33] fail completely. Fig. 5 shows the unique intermediate reasoning process of our model, which progressively refines solutions through multiple steps that are absent in prior approaches.

Results of Unified Training. The unified training results (single model trained on all tasks) show that our framework can learn transferable reasoning skills across tasks, though performance is slightly lower than task-specific training. As shown in Tab. 1, under unified training, our approach still achieves competitive performance compared to DiffThinker [11].

###### 5.2. Ablation Study and Analysis

To validate our design choices, we conduct comprehensive ablation studies on the Maze benchmark, which serves as a representative testbed for long-horizon reasoning. We report both the accuracy and path repetition rate that measures the overlap ratio between the generated path and the ground truth path.

Effectiveness of the Semantic Loss. The auxiliary Semantic loss provides explicit textual supervision by aligning our continuous latent token representations with ground-truth reasoning steps. This constraint guides the implicit tokens toward semantically meaningful trajectories.

As shown in Tab. 2, removing this supervision signal (w/o semantic loss) leads to severe performance

Table 4: Inference-Time CoT Scaling on Maze Benchmarks. Comparison of Accuracy and Path Repetition Rate across different token budgets (𝜏). Performance consistently improves, particularly on challenging Maze-32.

Accuracy Path Repetition (%)

Models

Time 8 16 32 8 16 32

DiffThinker [11] 100 97 56 100.00 100.00 73.74 15.72 Ours (𝜏 = 2) 100 72 11 100.00 89.15 45.26 16.02 Ours (𝜏 = 5) 100 94 27 100.00 98.49 63.90 16.54 Ours (𝜏 = 10) 100 99 49 100.00 99.93 82.33 17.52 Ours (𝜏 = 20) 100 100 74 100.00 100.00 96.47 19.27 Ours (𝜏 = 50) 100 100 90 100.00 100.00 98.13 24.81

120

Ours (Size 8)

Ours ( = 5) Ours ( = 10) Ours ( = 50) Ours ( = 100)

DiffThinker ChronoEdit ThinkGen Ours ( = 2)

| |
|---|

100

Ours (Size 16) Ours (Size 32)

| |
|---|

| |
|---|

| |
|---|

100

| |
|---|

| |
|---|

80

ExecutionTime(s)

###### Accuracy(%)

80

| |
|---|

60

60

Base Encoding Overhead (~15.7s)

40

40

20

20

0

0

0 5 10 15 20 25

512×512 960×960 1024×1024

Inference Time (s)

Resolution

- Figure 6: The trade-off between model accuracy and overall inference time across different token budgets.

Figure 7: Comparison of execution time against existing baselines across different resolutions and token budgets.

degradation. While the model retains basic routing ability on simple mazes, its logical consistency breaks down on harder instances: on the highly complex Maze-32 benchmark, accuracy drops from 90% to just 14%, and path repetition rate falls from 98.13% to 67.24%. This confirms that the semantic loss acts as a crucial regularizer, preventing implicit token drift and enabling reliable long-horizon reasoning.

Implicit vs. Explicit Tokens. While continuous implicit tokens offer flexibility, one might reasonably consider explicit text generation as a more interpretable alternative for reasoning [10, 24]. To investigate this design choice, we construct a variant that replaces implicit tokens with explicit autoregressive text generation, where the MLLM must produce discrete intermediate reasoning steps before generating the final visual output.

As reported in Tab. 2, explicit tokens lead to severe performance degradation. While the model achieves 34% accuracy on Maze-8, it fails on Maze-32 (0% accuracy). The failure case above reveals the cause: when planning long-horizon paths with a discrete vocabulary, the model becomes vulnerable to autoregressive error accumulation and mode collapse, degenerating into repetitive token loops.

Failure Case of Explicit Tokens

"⟨...⟩ <|im_start|>system\nDescribe the key features of the input image ⟨...⟩ Generate a new image that meets the user’s requirements ⟨...⟩ Direction keys: D=Down, U=Up, R=Right, L=Left.<|im_end|>\n<|im_start|>assistant\nRencontre Rencontre Rencontre ⟨...⟩ Rencontre Rencontre"

Inference-Time CoT Scaling. As shown in Tab. 4, our approach demonstrates a robust scaling law: dynamically increasing the number of implicit reasoning budgets 𝜏 smoothly and significantly elevates both accuracy and path repetition rate, particularly on the most challenging Maze-32 benchmark. This indicates that our implicit tokens effectively refine complex reasoning steps through iterative exploration. Furthermore, true scalability requires predictable computational costs. As shown in Fig. 6, our model steadily trades inference time for higher accuracy.

Resolution Scaling. Beyond the token budget 𝜏, our model demonstrates a crucial advantage in high-resolution tasks: as spatial resolution increases, the relative computational cost of our method significantly decreases. This efficiency gain stems fundamentally from the fact that our approach does not require repeating the computationally expensive DiT denoising steps. As shown in Fig. 7, our model maintains a stable and predictable inference latency.

###### (a) Poor adherence

(b) Generalization Task

A smart gardening system works as follows: If the soil moisture is low, the main irrigation sprinklers will turn on. If the irrigation sprinklers are on, a 'Wet Area' warning flag is automatically raised. It rained heavily all day yesterday. Draw the early morning scene.

Generalization SudokuFont

MazeSize

Input Target

Standard Domain Novel Domain

Standard Domain Novel Domain

Generalization

[Figure 203]

Input Target Standard Domain

Novel Domain Standard Domain Novel Domain

Training Examples DiffThinker Ours

(c) Latent Reasoning Chain

###### Vanilla Denoising Ours

I can fill (2,8) with 4 ...

[Figure 230]

I can fill (9,1) with 5 ...

[Figure 237]

Step1 Step2 Step5 Target

(Only partial intermediate results are displayed.)

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

EndoCoT : Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models

###### Case 1 Case 2 Case 3

Ori MLLM-Only Ori MLLM-Only Ori MLLM-Only

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Figure 8: Qualitative comparison of path reasoning. While MLLM-Only enables the model to understand the basic topological structure of the maze compared to the untuned baseline, it still lacks sufficient visual reasoning representations. Consequently, it exhibits erratic and wandering paths.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

- Figure 9: Progressive image editing results. Left: step-by-step object addition, where a stone lantern and a deer are sequentially introduced into the scene. Right: object transformation, where the deer is progressively modified into a sheep. Joint Training vs. MLLM-Only Baselines.

Finally, we investigate the necessity of jointly optimizing both the MLLM and DiT components, rather than relying on either module in isolation. Our core premise is that spatial reasoning requires both high-level cognitive planning (from the MLLM) and low-level physical grounding (from the DiT).

Quantitatively, Tab. 3 shows that decoupling components leads to severe performance degradation. Using only DiT limits the model’s logical reasoning capacity, causing accuracy to drop to 18% on Maze-32. Interestingly, while we attribute the model’s reasoning capability primarily to the MLLM (Section 3), relying solely on this module results in complete failure. This paradox indicates that while language models excel at abstract logic, they cannot directly map conceptual steps into spatial coordinates without the visual grounding provided by the DiT. This limitation is shown qualitatively in Fig. 8. The MLLM-only baseline produces erratic, wandering trajectories that frequently become trapped in dead ends, lacking global spatial awareness.

###### Results on Image Editing Tasks

As shown in Fig. 9, our approach demonstrates a distinctive progressive editing capability driven by the internal reasoning trajectory. Given step-by-step editing instructions, the model iteratively plans and executes each modification during generation. By varying the reasoning step 𝜏, we can control how many editing operations are carried out.

###### 6. Conclusion

We presented EndoCoT, a novel framework that enables diffusion models to perform endogenous Chain-ofThought reasoning. By iteratively refining latent thought states and grounding the final output in textual supervision, EndoCoT successfully bridges the gap between high-level logical planning and precise visual generation. Extensive experiments across diverse reasoning tasks demonstrate that this capability arises from the synergistic coupling of the multimodal text encoder and the DiT backbone, rather than from standard denoising processes alone. While highly effective, our approach currently requires manual tuning for the optimal number of reasoning steps and relies on high-quality datasets with explicit intermediate supervision. Future work will focus on adaptive mechanisms for automatic reasoning depth control and extending the framework to broader general-purpose tasks.

###### References

- [1] Aradhye Agarwal, Ayan Sengupta, and Tanmoy Chakraborty. The art of scaling test-time compute for large language models. arXiv preprint arXiv:2512.02008, 2025. 2
- [2] Anthropic. Claude opus 4.6. Anthropic Official Website, February 2026. Release; Claude Opus 4.6 achieves new state-of-the-art performance in complex reasoning, coding, and long-context understanding. 1
- [3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. 1
- [4] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 17682–17690, 2024. 2
- [5] Black Forest Labs. Flux.2: Frontier visual intelligence. Black Forest Labs Official Website. Release; FLUX.2 provides significant improvements in visual quality, prompt adherence, and architectural efficiency. 1, 4.2
- [6] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan Zhang, Mong-Li Lee, and Wynne Hsu. Videoof-thought: Step-by-step video reasoning from perception to cognition. arXiv preprint arXiv:2501.03230,

2024. 2

- [7] Ziteng Gao and Mike Zheng Shou. D-ar: Diffusion via autoregressive models. arXiv preprint arXiv:2505.23660, 2025. 2
- [8] Google DeepMind. Gemini 3. Google DeepMind Official Website, February 2026. Release; Gemini 3 is Google’s most capable model, featuring breakthrough capabilities in long-context understanding, multimodal reasoning, and native agentic performance. F
- [9] Jiatao Gu, Yuyang Wang, Yizhe Zhang, Qihang Zhang, Dinghuai Zhang, Navdeep Jaitly, Josh Susskind, and Shuangfei Zhai. Dart: Denoising autoregressive transformer for scalable text-to-image generation. arXiv preprint arXiv:2410.08159, 2024. 2
- [10] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769,

2024. 2, 5.2

- [11] Zefeng He, Xiaoye Qu, Yafu Li, Tong Zhu, Siyuan Huang, and Yu Cheng. Diffthinker: Towards generative multimodal reasoning with diffusion models. arXiv preprint arXiv:2512.24165, 2025. 1, 2, 5.1, 1, 5.1, 4
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 5
- [14] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In European conference on computer vision, pages 624–642. Springer, 2022. F
- [15] Siyu Jiao, Yiheng Lin, Yujie Zhong, Qi She, Wei Zhou, Xiaohan Lan, Zilong Huang, Fei Yu, Yingchen Yu, Yunqing Zhao, et al. Thinkgen: Generalized thinking for visual generation. arXiv preprint arXiv:2512.23568, 2025. 2, 1, 5.1, C.2

- [16] Siqi Kou, Jiachun Jin, Zetong Zhou, Ye Ma, Yugang Wang, Quan Chen, Peng Jiang, Xiao Yang, Jun Zhu, Kai Yu, et al. Think-then-generate: Reasoning-aware text-to-image diffusion with llm encoders. arXiv preprint arXiv:2601.10332, 2026. 2
- [17] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1, 4.1
- [18] Alexander H Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, et al. Ministral 3. arXiv preprint arXiv:2601.08584, 2026. 1
- [19] OpenAI. Thinking with images. OpenAI Official Website, April 2025. Release; OpenAI o3 and o4-mini visual reasoning models can think with images in chain-of-thought. 2
- [20] Yiming Qin, Bomin Wei, Jiaxin Ge, Konstantinos Kallidromitis, Stephanie Fu, Trevor Darrell, and XuDong Wang. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025. 2
- [21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR,

2021. 2

- [22] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 2
- [23] Noam Rotstein, Gal Yona, Daniel Silver, Roy Velich, David Bensaïd, and Ron Kimmel. Pathways on the image manifold: Image editing via video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7857–7866, 2025. 2
- [24] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025. 2, 5.2
- [25] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint

- arXiv:2010.02502, 2020. 1

[26] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint

- arXiv:2011.13456, 2020. 1

- [27] Wenhui Tan, Jiaze Li, Jianzhong Ju, Zhenbo Luo, Jian Luan, and Ruihua Song. Think silently, think fast: Dynamic latent compression of llm reasoning chains. arXiv preprint arXiv:2505.16552, 2025. 2
- [28] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025. 2
- [29] Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, Jiahui Ge, Qianli Ma, Hang He, Yifan Zhou, Lingzi Guo, Lantao Mei, Jiachen Li, Hanwen Xing, Tianqi Zhao, Fengyuan Yu, Weihang Xiao, Yizheng Jiao, Jianheng Hou, Danyang Zhang, Pengcheng Xu, Boyang Zhong, Zehong Zhao, Gaoyun Fang, John Kitaoka, Yile Xu, Hua Xu, Kenton Blacutt, Tin Nguyen, Siyuan Song, Haoran Sun, Shaoyue Wen, Linyang He, Runming Wang, Yanzhi Wang, Mengyue Yang, Ziqiao Ma, Raphaël Millière, Freda Shi, Nuno Vasconcelos, Daniel Khashabi, Alan Yuille, Yilun Du, Ziming Liu, Bo Li, Dahua Lin, Ziwei Liu, Vikash Kumar, Yijiang Li, Lei Yang, Zhongang Cai, and Hokin Deng. A very big video reasoning suite, 2026. 2
- [30] Shijian Wang, Jiarui Jin, Xingjian Wang, Linxin Song, Runhao Fu, Hecheng Wang, Zongyuan Ge, Yuan Lu, and Xuelian Cheng. Video-thinker: Sparking" thinking with videos" via reinforcement learning. arXiv preprint arXiv:2510.23473, 2025. 2

- [31] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 1
- [32] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 2, 3, 4.2, 5, 1, F
- [33] Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M Alvarez, et al. Chronoedit: Towards temporal reasoning for image editing and world simulation. arXiv preprint arXiv:2510.04290, 2025. 2, 1, 5.1, C.2
- [34] Ziang Yan, Xinhao Li, Yinan He, Zhengrong Yue, Xiangyu Zeng, Yali Wang, Yu Qiao, Limin Wang, and Yi Wang. Videochat-r1. 5: Visual test-time scaling to reinforce multimodal reasoning by iterative perception. arXiv preprint arXiv:2509.21100, 2025. 2
- [35] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1, 1
- [36] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023. 2
- [37] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He, and Yansong Tang. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416, 2025. 2
- [38] Huanyu Zhang, Wenshan Wu, Chengzu Li, Ning Shang, Yan Xia, Yangyu Huang, Yifan Zhang, Li Dong, Zhang Zhang, Liang Wang, et al. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514, 2025. 2
- [39] Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo, Yufei Wang, Niklas Muennighoff, et al. A survey on test-time scaling in large language models: What, how, where, and how well? arXiv preprint arXiv:2503.24235, 2025. 2
- [40] Zechuan Zhang, Zhenyuan Chen, Zongxin Yang, and Yi Yang. Are image-to-video models good zero-shot image editors? arXiv preprint arXiv:2511.19435, 2025. 2
- [41] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv preprint arXiv:2505.15778, 2025. 2
- [42] Zhi Zheng and Wee Sun Lee. Beyond imitation: Reinforcement learning for active latent planning, 2026. 2
- [43] Zhi Zheng, Zhuoliang Xie, Zhenkun Wang, and Bryan Hooi. Monte carlo tree search for comprehensive exploration in llm-based automatic heuristic design. arXiv preprint arXiv:2501.08603, 2025. 2

- A. Overview

In the supplementary material, Sec. B details the dataset statistics; Sec. C provides task prompts and additional results; Sec. D presents additional ablation studies; Sec. E offers EndoCoT insights and observations; and Sec. F elaborates on the additional editing datasets.

- B. Data Statistics

Appendix

- Table 5: Dataset Composition. Overview of the sample size for each task in our constructed dataset, totaling 182.4K instances.

Task Maze Sudoku TSP VSP Total Size 75K 40K 30K 37.4K 182.4K

To systematically evaluate our model’s reasoning capabilities, we construct a comprehensive dataset totaling 182.4K instances, comprising intermediate reasoning steps. As detailed in Tab. 5, this dataset spans four distinct reasoning tasks (Maze, TSP, Sudoku, and VSP).

###### B.1. Algorithmic Formulation

Maze. We use Depth-First Search (DFS) on a grid to carve a perfectly connected, loop-free maze. After randomly assigning a start (𝑠) and goal (𝑔), we compute the shortest path 𝑃 via Breadth-First Search (BFS). This path is recorded iteratively to simulate step-by-step spatial exploration.

TSP. We randomly distribute 𝑁 cities on a 2D grid and solve the resulting combinatorial optimization problem using the Held-Karp algorithm. The dynamic programming state transition is defined as:

{𝐷𝑃[𝑚 ∖ {𝑖}][𝑗] + dist(𝑗,𝑖)}, (12)

𝐷𝑃[𝑚][𝑖] = min

𝑗∈𝑚,𝑗̸=𝑖

where 𝑚 is the bitmask of visited cities and dist(𝑗,𝑖) is the Euclidean distance. The optimal node-to-node trajectory is rendered sequentially.

Sudoku. We construct a valid 9 × 9 board 𝐿 via backtracking, satisfying standard row, column, and block constraints:

∑︁9

∑︁9

𝐿𝑖,𝑗 = 𝑁; ∀block𝑘,∑︁𝐿 ∈ block𝑘 = 𝑁. (13)

∀𝑖,

𝐿𝑖,𝑗 = 𝑁; ∀𝑗,

𝑗=1

𝑖=1

We iteratively remove clues while ensuring the puzzle retains a unique solution. The intermediate reasoning steps are recorded by reversing this process, progressively filling empty cells from the ground-truth solution.

VSP. We generate a FrozenLake grid containing safe tiles, hazard holes, a start, and a goal. Framing this as a graph search, we apply Dijkstra’s algorithm to compute a safe path. The distance relaxation for adjacent safe nodes 𝑢 and 𝑣 is 𝑑(𝑣) = min{𝑑(𝑣),𝑑(𝑢) + 𝑤(𝑢,𝑣)}. The agent’s state transitions are recorded step-by-step to capture environment perception and hazard avoidance.

###### B.2. Pseudocode for Dataset Generation

The procedural generation for all four tasks are shown in Fig. 14: Maze Generation: Utilizes DFS to construct the maze, followed by BFS to determine the optimal path. TSP Generation: Applies the Held-Karp dynamic programming algorithm to derive the optimal tour. VSP Generation: Employs Dijkstra’s algorithm on the remaining safe graph to compute shortest paths. Sudoku Generation: Starts with a valid Latin square and iteratively removes values.

###### C. Case Studies

- C.1. Evaluation Prompts

Sudoku

Solve this Sudoku puzzle step-by-step from topleft to bottom-right. Identify the empty cell and fill in the correct digit.

TSP

Draw a continuous red line step-by-step from the start point to form the shortest closed loop. Mark the current endpoint with a green dot while avoiding other circles.

VSP

Draw a continuous red line from the Start point to the Goal point step-by-step, avoiding holes. Mark the current end of the path with a green dot. Directions: D=Down, U=Up, R=Right, L=Left.

Maze

Draw a continuous red line from the yellow dot to the blue dot step-by-step, avoiding walls. Mark the current end of the path with a green dot. Directions: D=Down, U=Up, R=Right, L=Left.

- C.2. Qualitative Examples

Figs. 15, 16, 17, and 18 provides additional qualitative examples comparing EndoCoT against ThinkGen [15] and ChronoEdit [33] across various reasoning tasks.

###### D. More Ablations

###### D.1. Effect of Two-Stage Training

Table 6: Single-stage training and multi-stage training on the Maze task.

Accuracy (ACC) Path Repetition (%) 8 16 32 8 16 32

Models

Single-Stage 39 42 14 83.49 91.34 90.34 Two-Stage 100 100 90 100.00 100.00 98.13

Tab. 6 presents an illustrative analysis on the Maze task, showing that Terminal Consolidation effectively enhances the model’s awareness of the final stage.

###### D.2. Impact of Terminal Consolidation Training Duration

Excessive training in Terminal Consolidation (Stage 2) causes intermediate steps to become sparse, as illustrated in Fig. 10. When choosing a constant number of reasoning steps 𝜏, checkpoints trained with more stage 2 training steps exhibit an inference behavior that converges more toward the target state.

###### D.3. Generalization under OOD Settings

On the Sudoku task, EndoCoT demonstrates stronger generalization than baseline methods across different resolutions and font styles, as shown in Fig. 11. For high-resolution inputs, EndoCoT consistently produces correct predictions for the 9x9 grid located in the center. For low-resolution inputs, it can recover partially cropped correct answers, while DiffThinker fails to generate valid solutions.

[Figure 262]

[Figure 263]

[Figure 264]

steps=0, 𝜏 = 2 steps=5000, 𝜏 = 2 steps=8000, 𝜏 = 2

[Figure 265]

[Figure 266]

[Figure 267]

steps=0, 𝜏 = 2 steps=1000, 𝜏 = 2 steps=8000, 𝜏 = 2

- Figure 10: Terminal Consolidation’s Excessive Training Results.

###### E. Training Hints

###### E.1. Training Settings

Table 7: Training hyperparameters and LoRA configuration.

General Settings LoRA Target Modules Hyperparameter Value Component Modules

Base Model Qwen Image Edit 2511 DiT Attn. to_q/k/v, add_*_proj, to_*_out Learning Rate 1 × 10−4 DiT FFN img/txt_mlp.net.2 Training Epochs 5 DiT Mod. img/txt_mod.1 LoRA Rank 32 Text Enc. Attn. q/k/v/o_proj LoRA Target DiT + Text Enc. Text Enc. FFN gate/up/down_proj

The detailed training settings, including hyperparameters and LoRA configurations, are summarized in Tab. 7.

###### E.2. Further Analysis

To further investigate where reasoning-related signals are primarily encoded in the model, we analyze the gradient activation differences between datasets that require explicit reasoning and those that do not. Specifically, we compute the layer-wise gradient magnitude difference across the MLLM and DiT.

As shown in Fig. 12, the largest gradient differences are consistently concentrated within the MLLM layers. This result further supports the analysis in Sec. 2: Reasoning capabilities are primarily mediated by the MLLM.

###### Different Fonts High Resolution Low Resolution

Baseline Ours Baseline Ours Baseline Ours

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

- Figure 11: OOD Test Result. EndoCoT demonstrates stronger generalization than baseline methods when solving Sudoku puzzles with varying resolutions and font styles.

[Figure 274]

Vision Encoder (ViT) Language Model (LLM) Diffusion Transformer (DiT)

- Figure 12: Gradient difference between reasoning and non-reasoning datasets.

- E.3. Experimental Observations

We obtain the following empirical observations while conducting experiments: Fine-grained step supervision is crucial. The number of reasoning steps performed during inference is typically much smaller than the number of steps supervised during training. Therefore, providing finer-grained intermediate step supervision is necessary for the model to learn latent reasoning processes.

More semantic supervision is not necessarily better. Applying semantic loss to intermediate steps can disrupt the synergy between the DiT rendering capability and the MLLM’s understanding. Empirically, this leads to inferior performance compared to the sparse semantic supervision strategy in EndoCoT.

Number of latent tokens depend on algorithmic and visual complexity. The number of latent tokens required to solve a puzzle is determined not only by the algorithmic complexity of the task, but also by its visual complexity.

- F. Image Editing Details

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

(a) Conceptual Grounding (b) Dataset Consolidation

[Figure 283]

[Figure 284]

Stage3

Stage2

Make the person painting.

Make the ship distressed.

Place a lone figure.

Add a ladder near them.

Add a rescue boat. Make figure gaze upward.

Img Gen.

Stage1

Add a person.

Add a small ship. Add a winding path.

A vibrant street art mural ...

An oil painting of a stormy sea...

A colossal ancient tree...

[Figure 285]

(c) Temporal Interpolation

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Edit Image Interpolation Target Image

- Figure 13: Overview of the multi-step image editing dataset generation pipeline. We create a high-quality multi-step image editing dataset containing 10,000 unique scenes and 30,000

images. To ensure diversity and quality, Gemini 2.5 [8] first generates diverse initial scene descriptions paired with multi-step editing instructions. Then Qwen-Image-Edit-2511 [32] generate the corresponding intermediate and final edited images, providing strict per-step supervision. Ultimately, to ensure editing continuity, we optionally employ RIFE [14] for intermediate frame interpolation. The complete data generation pipeline is shown in Fig. 13.

Algorithm 1 Maze Generation Require: Grid dimension 𝑁

Algorithm 2 TSP Generation Require: 2D plane dimensions 𝑊 ×𝐻, city count 𝑁𝑐

- 1: Initialize grid graph 𝐺 = (𝒱,ℰ∅) where 𝒱 = {1..𝑁}2
- 2: 𝑀 ← DFS_SpanningTree(𝐺) ◁ Perfect maze
- 3: Sample 𝑠,𝑔 ∼ 𝒰(𝒱) without replacement
- 4: 𝑃 ← BFS(𝑀,𝑠,𝑔) ◁ Path 𝑃 = (𝑣1,...,𝑣𝐾)
- 5: 𝒮 ← ∅ ◁ Intermediate visual states
- 6: for 𝑘 = 1 to 𝐾 do
- 7: 𝐼𝑘 ← Render(𝑀,𝑃1:𝑘)
- 8: 𝒮 ← 𝒮 ∪ {𝐼𝑘}
- 9: end for
- 10: return Dataset instance 𝒟 = (𝑀,𝑃,𝒮)

- 1: Sample cities C = {𝑐1,...,𝑐𝑁

𝑐} ∼ 𝒰(𝑊 × 𝐻)

- 2: Compute distance matrix D ∈ R𝑁

𝑐×𝑁𝑐

- 3: 𝑃 ← HeldKarpDP(D) ◁ Optimal tour
- 4: 𝒮 ← ∅
- 5: for 𝑘 = 1 to 𝑁𝑐 do
- 6: 𝐼𝑘 ← RenderTrajectory(C,𝑃1:𝑘)
- 7: 𝒮 ← 𝒮 ∪ {𝐼𝑘}
- 8: end for
- 9: return Dataset instance 𝒟 = (C,𝑃,𝒮)

Algorithm 4 Sudoku Generation Require: Target number of holes 𝐻𝑡𝑎𝑟𝑔𝑒𝑡

Algorithm 3 VSP Generation Require: Map dimension 𝑁

- 1: L ← GenerateLatinSquare() ◁ 9 × 9 solution
- 2: M ← L, ℎ ← 0
- 3: while ℎ < 𝐻𝑡𝑎𝑟𝑔𝑒𝑡 do
- 4: Sample 𝑐 ∼ NonEmptyCells(M)
- 5: 𝑣 ← M[𝑐]
- 6: M[𝑐] ← ∅ ◁ Dig hole
- 7: if NumSolutions(M) > 1 then
- 8: M[𝑐] ← 𝑣 ◁ Revert to maintain uniqueness
- 9: else
- 10: ℎ ← ℎ + 1
- 11: end if
- 12: end while
- 13: 𝒮 ← SimulateSolving(M,L)
- 14: return Dataset instance 𝒟 = (M,L,𝒮)

- 1: Generate map ℳ with grid 𝒱 = {1..𝑁}2
- 2: Assign types 𝒯 (𝑣) ∈ {Safe,Hole} ∼ Bern(𝑝ℎ𝑜𝑙𝑒)
- 3: Construct graph 𝒢 = (𝒱𝑠𝑎𝑓𝑒,ℰ𝑠𝑎𝑓𝑒)
- 4: Sample 𝑠,𝑔 ∼ 𝒰(𝒱𝑠𝑎𝑓𝑒)
- 5: 𝑃 ← Dijkstra(𝒢,𝑠,𝑔) ◁ Compute shortest path
- 6: Assert 𝑃 ̸= ∅ ◁ Ensure a valid path exists
- 7: 𝒮 ← ∅
- 8: for 𝑘 = 1 to |𝑃| do
- 9: 𝐼𝑘 ← RenderEnv(ℳ,𝑃1:𝑘)
- 10: 𝒮 ← 𝒮 ∪ {𝐼𝑘}
- 11: end for
- 12: return Dataset instance 𝒟 = (ℳ,𝑃,𝒮)

- Figure 14: Pseudocode for Dataset Generation

#### (a) Simple Cases

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

thhinkgen

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

thhinkgenOursChronoEditOursChronoEdit

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

#### (b) Harder Cases

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

- Figure 15: Qualitative comparison of EndoCoT and other methods on TSP.

##### (a) Simple Cases

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

thhinkgenOursthhinkgenChronoEditOursChronoEdit

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

##### (b) Harder Cases

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

- Figure 16: Qualitative comparison of EndoCoT and other methods on VSP.

### (a) Simple Cases

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

thhinkgenOursthhinkgenChronoEditOursChronoEdit

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

### (b) Harder Cases

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

- Figure 17: Qualitative comparison of EndoCoT and other methods on Sudoku.

#### (a) Simple Cases

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

thhinkgenOursthhinkgenChronoEditOursChronoEdit

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

#### (b) Harder Cases

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

- Figure 18: Qualitative comparison of EndoCoT and other methods on Maze.

