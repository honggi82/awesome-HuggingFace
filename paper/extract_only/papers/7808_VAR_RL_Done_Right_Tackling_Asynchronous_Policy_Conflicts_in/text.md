# arXiv:2601.02256v1[cs.CV]5Jan2026

[Figure 1]

## VAR RL Done Right: Tackling Asynchronous Policy Conflicts in Visual Autoregressive Generation

### Shikun Sun1,2, Liao Qu2, Huichao Zhang2,†, Yiheng Liu2, Yangyang Song2, Xian Li2, Xu Wang2, Yi Jiang2, Daniel K. Du2, Xinglong Wu2, Jia Jia1,†

1Tsinghua University, 2ByteDance †Corresponding Authors

### Abstract

Visual generation is dominated by three paradigms: AutoRegressive (AR), diffusion, and Visual AutoRegressive (VAR) models. Unlike AR and diffusion, VARs operate on heterogeneous input structures across their generation steps, which creates severe asynchronous policy conflicts. This issue becomes particularly acute in reinforcement learning (RL) scenarios, leading to unstable training and suboptimal alignment. To resolve this, we propose a novel framework to enhance Group Relative Policy Optimization (GRPO) by explicitly managing these conflicts. Our method integrates three synergistic components: 1) a stabilizing intermediate reward to guide early-stage generation; 2) a dynamic time-step reweighting scheme for precise credit assignment; and 3) a novel mask propagation algorithm, derived from principles of Reward Feedback Learning (ReFL), designed to isolate optimization effects both spatially and temporally. Our approach demonstrates significant improvements in sample quality and objective alignment over the vanilla GRPO baseline, enabling robust and effective optimization for VAR models.

Date: January 6, 2026 Official Page: https://github.com/ByteVisionLab/NextFlow

### 1 Introduction

Recent advances in visual generation are dominated by three paradigms: autoregressive (AR) models [11, 40, 49, 52, 58], diffusion models [5, 7, 12, 25, 36, 41, 45, 61], and visual autoregressive (VAR) [17, 30, 46] models. Unlike AR and diffusion, VAR operates over a hierarchy of discrete token grids whose spatial shape changes across resolution levels; at each step, the model emits a parallel grid of tokens rather than a single symbol or a fixed-length vector. While this heterogeneous, cross-scale design aligns with modern high-resolution backbones and enables fast synthesis, it also creates severe challenges for RL alignment, where the RL stage operates with far fewer samples than pretraining, exacerbating instability. To our knowledge, we are the first to conduct a systematic RL study for text-to-image VAR models, establishing a practical training recipe and revealing their unique RL failure modes. The crux of the difficulty is asynchronous policy conflict across timesteps. As illustrated in Figure 1, the number of query tokens can fluctuate by orders of magnitude from coarse to fine scales, inducing large variation in task similarity across steps, leading to instability, slow convergence, and suboptimal alignment when directly applying GRPO to VAR, as shown in training curve in Figure 2, where supervised RL at a partial-prefix scale outperforms its full-scale counterpart.

To address this core problem, we introduce a simple and effective framework that enhances GRPO [29, 57] for

###### VAR

###### AR

Diffusion

NumberofQueryTokens

NumberofQueryTokens

100

| |C|×H×W| |
|---|---|---|---|
| | | | |
| | | | |

DimensionofInputs

1

50

0

1 5 10

1 5 10

1 5 10

Generation Steps

Generation Steps

Generation Steps

- Figure 1 The number of query tokens across different timesteps in VAR generation fluctuates significantly, leading to varying task similarities and potential policy conflicts during RL optimization.

VAR with three synergistic components: (i) a stabilizing intermediate reward that provides dense, low-variance feedback to early steps while preserving family-optimality; (ii) a dynamic time-step reweighting that normalizes per-step contributions by the number of query tokens, balancing gradients across scales; and (iii) a mask propagation mechanism, inspired by gradient-based reward-feedback practices in RL for generative models [56], that spatially and temporally isolates optimization effects to the tokens most responsible for the final return. Concretely, we propose Value as Middle Return (VMR). Motivated by KL-regularized RL, we insert at a middle step m an intermediate soft return and optimize the pre-m and post-m segments with GRPO in a stage-wise fashion. This construction yields more frequent, lower-variance feedback to early decisions and, crucially, does not alter the optimal policy within the family: it is structure-preserving reward shaping that leaves the family-optimal solution unchanged while making it easier to reach in practice. Second, we propose Per-Action Normalization Weighting (PANW). To counteract step-wise heterogeneity, we weight each step-t loss by kt = 1/(htwt), where ht × wt is the token-grid size at that step, followed by step-level normalization. This balances KL usage and gradient scales across timesteps, mitigating the dominance of high-resolution updates and improving stability. Third, we introduce Mask Propagation (MP). We maintain a spatiotemporal mask that tracks tokens likely to contribute to the terminal reward and propagates this mask backward along the sequence. The mask gates intermediate rewards and gradients, focusing credit assignment on causally relevant regions and reducing variance across both space and time. Empirically, our framework consistently stabilizes training and accelerates convergence over vanilla GRPO in this setting, delivering substantial gains in sample quality and objective alignment on text rendering benchmarks [4, 10, 48]. It also achieves strong improvements over a TokenFlow-T2I starting point [37] and attains state-of-the-art results among diffusion-centric baselines [5, 7, 12, 25, 36, 41, 45, 61]. These results underscore that properly structuring the RL objective and balancing updates across heterogeneous steps are crucial for reliable text-to-image alignment. Our contributions are threefold:

- • We diagnose asynchronous policy conflicts in RL for VAR and formalize the process as a deterministic MDP, clarifying why bandit-style GRPO becomes unstable under heterogeneous, parallel actions.
- • We propose VMR, a structure-preserving intermediate reward that provides dense feedback to early steps and provably does not alter the family-optimal policy; together with PANW and MP, it balances step-wise gradients and sharpens spatiotemporal credit assignment.
- • We present, to our knowledge, the first systematic RL framework for text-to-image VAR, demonstrating robust, state-of-the-art improvements over GRPO baseline and diffusion-centric competitors, supported by comprehensive ablations.

### 2 Related Work

#### 2.1 Autoregressive Image Generation

Autoregressive (AR) image generation has emerged as a compelling paradigm, inspired by the success of nexttoken prediction in large language models [1, 3, 38, 39]. Early works such as VQ-VAE [11, 49] and VQGAN [11] introduced learned codebooks to discretize images into tokens, enabling sequential generation in raster-scan

1.2588

Vanilla GRPO

GRPO with prefix 512 GRPO with prefix 256 GRPO with prefix 128 GRPO with prefix 64

1.1844

RewardValue

1.1100

1.0357

0.9613

0 50 100 150 200 250 300 350 400

Training Step

- Figure 2 Comparison of training curves between vanilla GRPO and GRPO with VMR across varying prefix scales.

order. This approach has since been widely adopted in text-to-image synthesis [58] and unified multimodal understanding and generation tasks [6, 52, 54]. However, raster-scan AR models typically require significantly more sampling steps compared to diffusion-based approaches, limiting their inference-time efficiency. To address this limitation, VAR [46] proposed a next-scale prediction strategy—a coarse-to-fine framework that predicts tokens across progressively finer spatial scales, aligning more closely with human perceptual hierarchy. This method substantially improves both sampling efficiency and image fidelity. Subsequent studies have further advanced this paradigm [22, 26], demonstrating strong performance in high-resolution text-to-image generation [17, 30, 31, 37, 44, 50, 64]. Building on these advances, this work investigates the use of reinforcement learning to further enhance VAR-style autoregressive image generation, aiming to improve training stability and overall generative performance.

#### 2.2 Reinforcement Learning for Image Generation

Reinforcement Learning (RL) has been widely adopted for enhancing the reasoning capabilities and human alignment of Large Language Models (LLMs) [16, 20]. A significant recent advancement is GRPO [43], which simplifies policy optimization by eliminating the value model required in PPO [42], achieving notable empirical gains through a relative, group-based objective. This paradigm has been successfully extended to visual generation to improve output fidelity and controllability. For instance, flow-based generative models [12, 24] have been aligned with human preferences [23, 55] and specific prompt following objectives [15, 19] using GRPO-based techniques [29, 57]. The application of RL differs for autoregressive (AR) raster-scan models [21, 32, 47, 51, 59, 60]. T2I-R1 [21] employs RL to bolster semantic reasoning for superior text-to-image alignment, AR-GRPO [59] establishes a direct baseline by integrating GRPO into the AR sampling process, and SimpleAR [51] unifies model training with RL to holistically improve output quality. Despite these advances, the use of RL in next-scale prediction AR models remains largely unexplored. While [13] applies GRPO to a class-conditioned model [46], it does not address the core structural challenges introduced by multi-scale parallel token generation. Compared to raster-scan AR or diffusion models, this family of architectures presents unique challenges compared to raster-scan AR or diffusion-based approaches, primarily due to structural differences in policy optimization that arise from multi-scale, parallel token generation. These differences necessitate tailored reward design and optimization strategies that have yet to be thoroughly investigated.

### 3 Methodology

#### 3.1 Preliminary

- 3.1.1 Visual AutoRegressive Models

Visual AutoRegressive (VAR) models generate images by producing a sequence of discrete tokens that represent the image at multiple resolutions. Specifically, an image is represented as a sequence of token grids r1,r2,...,rT, where each rt corresponds to a grid of tokens at resolution level t shaped ht×wt. The generation process starts from the lowest resolution and progressively refines the image by generating higher-resolution token grids conditioned on the previously generated lower-resolution grids.

- 3.1.2 VAR Sequences as MDPs

Because we aim to optimize VAR using a Hierarchical Reinforcement Learning [34] framework, unlike the bandit-style setup used in vanilla GRPO, we must explicitly define the MDP (S,A,P,r) for VAR sequence generation. We formulate the process as follows (for simplicity, we omit external control inputs such as text or images):

- • Action Space A: Each action aθt(st) = rt+1 ∈ A corresponds to generating the next resolution tokens grid hr+1 × wr+1.
- • State Space S: Each state is the partial VAR sequence st = (r1,r2,...,rt) ∈ S.
- • State Transition Probability P: The transition is deterministic, given by P(st+1 | st,at) = δ(r

1,r2,...,rt+1).

- • Reward Function r: As in typical image generation tasks, the environment only provides a final return R(sT) = i ri, reflecting evaluation in real-world settings.

- 3.1.3 The Optimal Solution for KL-Regularized RL

A well-established result in KL-regularized reinforcement learning is that the optimal policy π∗, which maximizes the expected return under a soft KL constraint, is given by [65]:

π∗(at | st) ∝ πold(at | st)exp

1 η

Q∗(st,at) , (1)

where η is the temperature parameter and Q∗(st,at) denotes the optimal action-value function: Q∗(st,at) = Eπ∗ [R(sT) | st,at]. (2)

This formulation shows that the optimal policy is shaped jointly by the prior policy πold and the expected future return captured by the action-value function Q∗. Moreover, when the environment transition is deterministic [63], the future return is fully determined by πold, and the optimal Q-function can be written as:

Q∗(st,at) = η lnEπ

old

exp

1 η

R(sT) | st,at . (3)

#### 3.2 Motivation

The key challenge in optimizing VAR sequences using RL methods lies in the asynchronous policy conflicts that arise across different timesteps. As shown in Figure 1, unlike diffusion models and autoregressive models, the number of query tokens fluctuates dramatically during the generation process, leading to substantial variation in task similarity across timesteps. This inconsistency is especially problematic in RL settings, where the available training data is significantly more limited compared to large-scale pretraining, thereby exacerbating policy instability and slowing convergence. In addition, because multiple tokens are generated in parallel within the same resolution scale, the optimal policy structure may deviate from that of typical sequential RL formulations, requiring further analysis.

To address the challenges mentioned above, we propose three techniques to stabilize and accelerate the RL optimization of VAR:

𝐴 :   

KL

Reference Model

Group Computation

𝐴 :   

…

Reward Model

𝑜 :   

𝐴 :   

𝑜 :   

𝜋 :   

KL

…

𝑜 :   

𝑜 : 

𝐴 : 

Reference Model

𝜋 : 

Reward Model

𝑜 : 

𝐴 : 

…

…

𝑜 : 

𝐴 : 

- Figure 3 The pipeline of our two-stage GRPO is as follows: we use Monte Carlo estimation to compute the reward for π1:θ m−1, and then apply GRPO to the two states separately.

- • We introduce Value-as-Middle-Return (VMR) to decompose the full-sequence RL objective into a two-stage optimization problem, effectively mitigating multi-step conflict during training.
- • We apply Per-Action Normalization Weighting (PANW) to balance the contributions of different timesteps, ensuring stable learning dynamics even under varying task similarities.
- • We implement Mask Propagation (MP) to prioritize updates on the most relevant tokens and to reduce imbalance across scales.

#### 3.3 Value as Middle Return

###### 3.3.1 Formulation

The core idea of VMR is to decompose the full-horizon, KL-regularized RL problem into two coupled subproblems split at a middle timestep m. This alleviates cross-horizon action conflicts and yields denser feedback. Reusing the notation of (1), we define the middle-step soft value as follows:

- Definition 1 (Middle-step soft value). For any m ∈ {1,...,T},

Vm∗(sm) = η log Eπ

exp η 1R(sT) sm . (4)

old

VMR introduces a middle return Vm∗(sm) to partition the generation into prefix and suffix subtasks. Each is trained independently with a local KL penalty:

Suffix : max

E R(sT) | sm − η KL πm:T−1 ∥πold,m:T−1 , Prefix : max

πm:T−1

(5)

E Vm∗(sm) − η KL π1:m−1 ∥πold,1:m−1 .

π1:m−1

This stage-wise decomposition stabilizes training by keeping per-stage token lengths comparable, while preserving the global objective value. The detailed algorithm is shown in Figure 3.

- 3.3.2 Analysis We now examine two central questions:

- • What is the optimal solution for the VAR family under KL-regularized RL?
- • Does introducing the middle return V ∗(sm) alter the optimal policy?

VAR Family Definition. We first specify the policy class of interest as the following definition:

- Definition 2 (VAR family Mπ). A policy πθ belongs to Mπ if, at each timestep t, the action grid aθt =

rt+1 ∈ Rh

t+1×wt+1 factorizes across spatial sites given the state st: πθ(at | st) =

(i,j)

πt,θ(i,j)(at,(i,j) | st). (6)

This mirrors the implementation of VAR, where tokens on the same resolution are generated in parallel. And this does not mean that different blocks of the generated image are independent, because the state st contains all the previously generated tokens, which can provide global context.

- Q1: Optimality within Mπ. The globally optimal KL-regularized policy π∗ from (1) need not belong to Mπ. Nevertheless, within this restricted family, there exists a unique constrained optimum π†:

Definition 3 (Constrained optimal policy). The optimal VAR policy is

π† = arg max

π∈Mπ

J(π),J(π) = Eπ[R(sT)] − η KL(π ∥πold). (7)

The π† satisfies the following property:

- Theorem 1 (Reverse-KL characterization of π†). At each state st, the constrained optimum satisfies

π†(· | st) = arg min

π∈Mπ(st)

KL π(· | st)∥π∗(· | st) , (8)

where π∗ is the global soft-optimal policy. Thus, the best policy within the VAR family is obtained by reverse-KL projecting the globally optimal, potentially non-factorized policy π∗ onto Mπ. Q2: Effect of introducing VMR. We now analyze whether VMR changes the optimal solution. We have the following theorem:

- Theorem 2 (Two-stage invariance). Let Vm∗(sm) be defined as in (4). Within Mπ, solving the suffix problem in (5) yields πm† :T−1; optimizing the prefix problem using Vm∗ as its sole reward gives π1:† m−1. Then, the

concatenation π† = π1:† m−1⊕πm† :T−1 uniquely maximizes the full-horizon objective J(π). Within Mπ, replacing each subpolicy by its per-state reverse-KL projection onto the factorized family yields the constrained optimum

π†.

The theorem implies that the intermediate value Vm∗ encapsulates all downstream contributions from step m onward. When used as the sole reward for the prefix, it ensures that prefix and suffix optimization remain consistent with the original full-horizon solution. This two-stage structure preserves policy invariance, avoids credit-assignment interference, and stabilizes training across varying token lengths.

VMR provides a principled prefix–suffix decomposition of KL-regularized RL that maintains optimality within both the unconstrained and VAR-constrained settings, while reducing multi-step conflicts and improving learning stability in coarse-to-fine generation.

𝑟 𝑟 𝑟 𝑟 𝑟

- Figure 4 Our token masks are propagated in reverse through the model’s multi-scale hierarchy, moving from finer to coarser feature scales.

#### 3.4 Per-Action Normalization Weighting

As shown in Figure 1, the task similarity across different timesteps in VAR generation can vary significantly due to the fluctuating number of query tokens. To address this issue, we propose Per-Action Normalization Weighting, which normalize the contributions of each action based on the number of query tokens at that timestep. Specifically, for each timestep t, we compute a normalization weight kt defined as:

1 (ht × wt)α

(9)

kt =

where ht and wt are the height and width of the token grid at timestep t and α is a decay exponent. During training, we weight the loss associated with each action at by its corresponding normalization weight kt. This approach ensures that timesteps with a larger number of query tokens do not disproportionately influence the learning process, thereby promoting a more balanced optimization across all timesteps.

##### 3.5 Mask Propagation To precisely identify which tokens influence the final return, we introduce Mask Propagation. As shown in

- Figure 4, we first construct an initial mask from the output components that directly determine the reward (e.g., predicted bounding boxes). We then propagate this mask backward through the model’s multi-scale hierarchy, moving from finer to coarser feature scales. This process directs updates toward the most relevant tokens while simultaneously improving cross-scale balance.

### 4 Experiments

- 4.1 Experimental Setup We use NextFlow [37] as our base model. Images are generated at a resolution of 1024×1024.

Training Setup. We adopt an on-policy training strategy with group size = 16 (candidates per prompt) and batch size = 16 (prompts per update). The policy is initialized with a learning rate of 10−6. Prompts are sampled from our in-house pool and are strictly disjoint from the training and testing splits of all evaluation tasks. We train for up to 1,200 updates, i.e., up to 19,200 unique prompts per task. Optimization uses AdamW with learning rate 10−5, default β1 = 0.9, β2 = 0.95, and weight decay = 0.05. We alternate optimization at different scales following Eq. (5): for every three prefix GRPO updates (optimizing π1:m−1), we perform one suffix GRPO update (optimizing πm:T−1). In contrast to the sampling configuration, classifier-free guidance (CFG) [18] is not used during training.

Sampling Setup. We adopt the original sampling strategy of VAR [46] for both rollout and evaluation phases. The sampling parameters are fixed at CFG=5, top-k=2 and top-p=0.9 for all token sampling operations.

VMR Reward Construction. Instead of fitting a step-wise critic as in PPO [42], we estimate the middlestage value directly from on-policy terminal rewards. For a given state sm, we sample K on-policy rollouts {τk}Kk=1 ∼ πθ(· | sm) up to terminal step T, obtain terminal rewards {R(k)(sT)}Kk=1, and define the VMR as a risk-sensitive estimator:

K

1 K

exp η 1 R(k)(sT) , (10)

Vm(sm) = η log

k=1

where η > 0 is a temperature parameter and K is the number of on-policy samples. In practice, we set η = 1 and K = 2, which we find sufficient and stable (Figure 2). We primarily validate our method on the text rendering task with an extensive ablation study, and further evaluate on HPSv3 to demonstrate robustness and generalization.

Selection of m. We observe that the primary gains stem from applying RL to the prefix segment, making the choice of m crucial. As shown in Figure 2, a sweet spot emerges at m ∈ {m128,m256}, where these values represent the steps corresponding to resolutions 128 × 128 and 256 × 256 (see Table 6). Further ablations (Subsection 4.4) justify our final choice of m = m256.

#### 4.2 Text Rendering

[Figure 2]

- Figure 5 Visual Samples of text rendering task before (left) and after (right) RL optimization. The text required for each pairs: (1) Six illuminated letters (’A’, ’B’, ’C’, ’N’, ’O’, ’Y’) (2) "Wouldn’t you rather... VOTE. By Mail."

(3) "M-BOX 2.0 by: Jimmy-Fan" (4) "DESCUBRE LO QUE ESCONDEN NUESTRO CAMPING" and "UPBAN KIDS" (5) "L’ARANCAEY FIMMNA," (6) "FISHING CHALLENGE" and "GOTCHA!" (7) "Apps Way Crisps.", "Apps Wheat Crisps.", "Apps Potato Crisps." and "Apps Cheese Balls." (8) "VAR RL Done Right". Full captions are provided in the Appendix. The RL-refined outputs demonstrate improvements in correcting character misordering, erroneous glyphs, missing or extraneous characters. Better zoom in for details.

###### 4.2.1 Reward Design

OCR Backbone. We use PaddleOCRv5 [8, 9] to recognize text from rendered images. Let the lowercased groundtruth word sequence be G = (gk)Nk=1, the predicted word sequence be P = (pi)Mi=1, and the corresponding OCR confidences be S = (si)Mi=1 with si ∈ [0,1]. We measure string similarity by a normalized Levenshtein score:

EditDist(x,y) max{|x|,|y|} + ε

, (11)

LD(x,y) = 1 −

where |x| is the character length of x, EditDist(·,·) is the Levenshtein distance, and ε > 0 avoids division by zero.

Completeness (confidence-aware). For each g ∈ G, if g appears in P, we count the minimum confidence among identical predictions to discourage duplicate inflation:

1 N g∈G

si , (12)

1{g ∈ P} · min

Comp =

i: pi=g

where 1{·} is the indicator function. Length mismatch penalty. To penalize over/under-generation, we compare concatenated strings via a multiset distance:

BagDist Mi=1 pi, Nk=1 gk

, λ = 0.6, (13)

Pen = λ ·

M i=1 pi + Nk=1 gk

where BagDist(·,·) measures character-level multiset discrepancy and the denominator normalizes by total character count.

Similarity (confidence-weighted). Let i⋆(g) = arg maxi LD(g,pi) be the best match index for g; we can weight by the matched confidence:

1 N g∈G

LD g,pi⋆(g) si⋆(g) (14)

Sim =

Total reward. The final OCR reward combines the above:

Reward = Comp+Sim−Pen. (15)

###### 4.2.2 Experimental Analysis

On CVTG-2K [10] (Table 1), our RL-based method (NextFlow-RL) markedly outperforms NextFlow across all metrics. Specifically, NextFlow-RL improves Word Accuracy from 0.5536 to 0.7841 (+0.2305 absolute, +41.6% relative) and NED from 0.7816 to 0.9081 (+0.1265 absolute, +16.2% relative), while also achieving a higher CLIPScore (0.8224 vs. 0.8068). These results demonstrate that NextFlow-RL, driven by confidenceand similarity-aware OCR rewards with a length penalty, significantly improves word-level accuracy and character-level fidelity while preserving semantic alignment.

Overall, the consistent improvements over NextFlow and the leading results among diffusion models validate the effectiveness of our two-stage RL scheme: optimizing early-token (prefix) decisions guided by confidence/similarity-aware OCR rewards with a length penalty yields robust gains in both text fidelity and visual quality across broad content categories.

###### 4.2.3 Visual Results

- Figure 5 shows visual results before (left in each pair) and after (right) RL optimization on text rendering. RL-refined outputs correct character order, fix erroneous glyphs, and reduce missing or extraneous characters across diverse layouts and styles. Full prompts are provided in the Appendix.

[Figure 3]

###### Figure 6 Visual Samples of HPS refine task before (left) and after (right) RL optimization. Full captions are provided in the Appendix. Better zoom in for details.

- Table 1 Quantitative results on the CVTG-2K dataset. Bold denotes the best performance, underline denotes the second-best for each metric. Seedream 3.0 and GPT Image 1 are highlighted as proprietary/closed-source systems.

Model #Params Word Accuracy↑ NED↑ CLIPScore↑

FLUX.1 dev [24] 12B 0.4965 0.6879 0.7401 SD3.5 Large [12] 8B 0.6548 0.8470 0.7797 AnyText [48] 1.4B 0.1804 0.4675 0.7432 TextDiffuser-2 [4] 8B 0.2326 0.4353 0.6765 RAG-Diffusion [27] 12B 0.2648 0.4498 0.6688 3DIS [62] 12B 0.3813 0.6505 0.7767 TextCrafter (FLUX) [10] 12B 0.7370 0.8679 0.7868 TextCrafter (SD3.5) [10] 8B 0.7600 0.9038 0.8023

Seedream 3.0 [14] - 0.5924 0.8537 0.7821 GPT Image 1 [High] [35] - 0.8569 0.9478 0.7982 Qwen-Image [53] 20B 0.8288 0.9116 0.8017

NextFlow 7B 0.5536 0.7816 0.8068 NextFlow-RL 7B 0.7841 0.9081 0.8224

4.3 Human Preference Score

- 4.3.1 Reward Design

We adopt the best hyperparameters from the Text Rendering task and use HPSv3 [33] as the direct reward function during RL optimization. To handle the large model efficiently, we deploy a self-hosted HPS service for low-latency reward evaluation and high-throughput training.

- 4.3.2 Experimental Analysis

Table 2 shows that our method (NextFlow-RL) delivers substantial gains over the starting point NextFlow across all categories. On the overall metric All, NextFlow-RL improves from 8.43 to 10.64 (+2.21 absolute), and consistently outperforms NextFlow in every sub-category, with notable margins on Animals (10.79 vs. 8.34), Natural Scenery (10.43 vs. 8.22), Plants (10.57 vs. 8.19), and Food (11.18 vs. 8.93). These results indicate that our RL-based prefix optimization and OCR-guided reward design not only enhance word rendering fidelity but also generalize across diverse visual domains (e.g., transportation, products, and science).

Among diffusion-centric models, NextFlow-RL attains state-of-the-art performance on multiple key columns: it ranks 1st on All (10.64), Architecture (11.16), Animals (10.79), Natural Scenery (10.43), Plants (10.57), Food (11.18), and Others (10.66), while remaining competitive on Characters (11.72, second only to Kolors at 11.79) and Design (9.77, second only to Kolors at 9.87). Compared to strong baselines such as Flux-dev, Kolors, and Playground-v2.5, NextFlow-RL matches or surpasses their best scores in most categories, establishing a new SOTA within the diffusion-based family.

- 4.3.3 Visual Results

We showcase visual results on the HPS refine task before and after RL optimization in Figure 6. Across diverse prompts and styles, including cartoons, photorealistic portraits, astrophotography, landscapes, and object-centric scenes, our RL-enhanced model produces images with sharper details, cleaner structure, and stronger prompt adherence. Full captions are provided in the Appendix B.

- 4.4 Ablation We conduct ablations over key design choices and hyperparameter, summarizing findings below. Selection of m in Equation (10). From Table 3a, m=m128 achieves the best scores (Word Acc. 0.6677, NED

###### 0.8501, CLIPScore 0.8142), while m=m256 is very close (0.6565/0.8429/0.8133). Given comparable quality but

Table 2 Comparison of different models across various categories. The best results are in bold, and the second-best are underlined.

Models All Characters Arts Design Architecture Animals Natural Scenery Transportation Products Plants Food Science Others

|Stable Diffusion v2.0 [41] Stable Diffusion 3 [12] Hunyuan [28] Stable Diffusion XL [36] Gemini 2.0 Flash [7] PixArt-Σ [5] CogView4 [61] Infinity [17] Playground-v2.5 [25] Flux-dev [24] Kolors [45]<br><br>|-0.24 -0.34 -0.56 -1.35 -0.24 -0.54 -0.32 1.00 1.11 -0.01 -0.38 -0.38 -0.84 5.31 6.70 5.98 5.15 5.25 4.09 5.24 4.25 5.71 5.84 6.01 5.71 4.58<br><br>8.19 7.96 8.11 8.28 8.71 7.24 7.86 8.33 8.55 8.28 8.31 8.48 8.20<br>8.20 8.67 7.63 7.53 8.57 8.18 7.76 8.65 8.85 8.32 8.43 8.78 7.29<br><br><br>9.21 9.98 8.44 7.64 10.11 9.42 9.01 9.74 9.64 9.55 10.16 7.61 9.23<br><br>9.37 10.08 9.07 8.41 9.83 8.86 8.87 9.44 9.57 9.52 9.73 10.35 8.58<br><br>9.61 10.72 9.86 9.33 9.88 9.16 9.45 9.69 9.86 9.45 9.49 10.16 8.97<br>10.26 11.17 9.95 9.43 10.36 9.27 10.11 10.36 10.59 10.08 10.30 10.59 9.62<br><br><br>10.27 11.07 9.84 9.64 10.45 10.38 9.94 10.51 10.62 10.15 10.62 10.84 9.39<br><br><br>10.43 11.70 10.32 9.39 10.93 10.38 10.01 10.84 11.24 10.21 10.38 11.24 9.16 10.55 11.79 10.47 9.87 10.82 10.60 9.89 10.68 10.93 10.50 10.63 11.06 9.51<br><br><br>|
|---|---|
|NextFlow NextFlow-RL|8.43 9.27 8.00 7.51 8.98 8.34 8.22 8.68 8.70 8.19 8.93 7.75 8.57 10.64 11.72 10.26 9.77 11.16 10.79 10.43 10.77 10.73 10.57 11.18 9.60 10.66<br><br>|

Table 3 Ablation studies on the hyperparameters m and α. (a) Ablation on the choice of m, with K = 2 at step 300.

(b) Ablation on decay exponent α, measured at step 400.

m Word Accuracy↑ NED↑ CLIPScore↑

NextFlow 0.5536 0.7816 0.8068 1024 0.5741 0.7871 0.8038 512 0.6351 0.8236 0.8057 256 0.6565 0.8429 0.8133 128 0.6677 0.8501 0.8142

α Word Accuracy↑ NED↑ CLIPScore↑

NextFlow 0.5536 0.7816 0.8068 1.2 0.6602 0.8459 0.8144 1.0 0.6855 0.8601 0.8188 0.8 0.7051 0.8673 0.8172 0.6 0.7136 0.8709 0.8165

lower compute to estimate the VMR and better compatibility with the mask mechanism, we select m=m256

- as the default. Larger m (e.g., 512, 1024) underperform, indicating the benefit of placing the VMR earlier to reduce variance and improve early credit assignment.

Decay exponent α in Equation (9). As shown in Table 3b and Figure 7a, α ∈ [0.6,0.8] yields the strongest results, with α=0.6 giving the best Word Accuracy/NED and α=0.8 the top CLIPScore. This range provides a robust default, balancing gradient normalization across heterogeneous steps without over-suppressing high-resolution updates.

Mask Propagation (MP). Table 4a and Figure 7b show that enabling MP improves text fidelity metrics (Word Acc. 0.7071 vs. 0.6855; NED 0.8699 vs. 0.8601), with essentially unchanged CLIPScore. This confirms MP sharpens spatiotemporal credit assignment and is beneficial at scale.

Number of samplesK in Equation(10). From Table 5, K=2 performs best overall (0.6821/0.8505/0.8287). While K=1 underutilizes exploration, larger K (e.g., 4) shows slight degradation, suggesting compatibility issues with MP and increased variance from heterogeneous trajectories. Thus, K=2 offers the best stability–performance trade-off.

Alternating training granularity. Table 4b compares fine-grained versus coarse-grained alternation between low-scale (prefix) and high-scale (suffix) segments. In the fine-grained scheme, we perform three prefix updates for every suffix update; in the coarse-grained scheme, we first run 300 prefix updates and then 100

Table 4 Ablation studies on Mask Propagation (MP) and alternating training schemes. (a) Effect of MP, measured at step 400.

(b) Ablation on alternating training schemes.

Model Word Accuracy↑ NED↑ CLIPScore↑

NextFlow 0.5536 0.7816 0.8068 NextFlow-RL w/o MP 0.6855 0.8601 0.8188 NextFlow-RL w/ MP 0.7071 0.8699 0.8184

Scheme Word Accuracy↑ NED↑ CLIPScore↑

NextFlow 0.5536 0.7816 0.8068 Fine-grained 0.6855 0.8601 0.8188 Coarse-grained 0.6778 0.8564 0.8168

Table 5 Ablation on K, evaluated at step 300.

K Word Accuracy↑ NED↑ CLIPScore↑ NextFlow 0.5536 0.7816 0.8068

- 1 0.6575 0.8388 0.8258
- 2 0.6821 0.8505 0.8287 4 0.6720 0.8449 0.8278

###### Comparison of GRPO Methods with Different Alpha Values

###### Comparison of GRPO Methods: With vs Without MP

1.1948

1.2530

without MP

with MP

1.1463

1.1804

RewardValue

RewardValue

1.0978

1.1078

alpha=1.2 alpha=1.0 alpha=0.8 alpha=0.6 alpha=0

1.0493

1.0351

1.0008

0.9625

0 50 100 150 200 250 300 350 400

0 50 100 150 200 250 300 350 400 450

Training Step

Training Step

(b) Effect of Mask Propagation (MP). Figure 7 Ablations on training configurations. Left: varying decay exponent α; Right: enabling/disabling MP.

(a) Decay exponent α comparison.

suffix updates. Both settings are evaluated at the same total training step (400). Fine-grained alternation consistently outperforms coarse-grained (0.6855/0.8601/0.8188 vs. 0.6778/0.8564/0.8168), indicating that more frequent, localized updates better resolve asynchronous policy conflicts.

### 5 Conclusion

We present the first RL framework for VAR in T2I task, tackling asynchronous policy conflicts across heterogeneous scales. Our VMR decomposes the full-horizon objective into prefix/suffix stages without altering the optimal solution, while Per-Action Normalization Weighting and Mask Propagation stabilize credit assignment and concentrate updates on reward-relevant tokens. Empirically, our method delivers consistent gains over vanilla GRPO with strong improvements over original NextFlow.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [4] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser-2: Unleashing the power of language models for text rendering. In European Conference on Computer Vision, pages 386–402. Springer, 2024.

- [5] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024.

- [6] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

- [8] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, Yue Zhang, Yubo Zhang, Handong Zheng, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr-vl: Boosting multilingual document parsing via a 0.9b ultra-compact vision-language model, 2025. URL https://arxiv.org/abs/2510.14528.
- [9] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025. URL https://arxiv.org/abs/2507.05595.
- [10] Nikai Du, Zhennan Chen, Shan Gao, Zhizhou Chen, Xi Chen, Zhengkai Jiang, Jian Yang, and Ying Tai. Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461, 2025.

- [11] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

- [13] Matteo Gallici and Haitz Sáez de Ocáriz Borde. Fine-tuning next-scale visual autoregressive models with group relative policy optimization. arXiv preprint arXiv:2505.23331, 2025.

- [14] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

- [16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [17] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15733–15744, 2025.

- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [19] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36: 78723–78747, 2023.

- [20] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- [21] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025.

- [22] Siyu Jiao, Gengwei Zhang, Yinlong Qian, Jiancheng Huang, Yao Zhao, Humphrey Shi, Lin Ma, Yunchao Wei, and Zequn Jie. Flexvar: Flexible visual autoregressive modeling without residual prediction. arXiv preprint arXiv:2502.20313, 2025.

- [23] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

- [24] Black Forest Labs. Flux, 2024. URL https://github.com/black-forest-labs/flux.
- [25] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.

- [26] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024.

- [27] Yuhan Li, Xianfeng Tan, Wenxiang Shang, Yubo Wu, Jian Wang, Xuanhong Chen, Yi Zhang, Hangcheng Zhu, and Bingbing Ni. Ragdiffusion: Faithful cloth generation via external knowledge assimilation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17485–17495, 2025.

- [28] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.

- [29] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

- [30] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infinitystar: Unified spacetime autoregressive modeling for visual generation. arXiv preprint arXiv:2511.04675, 2025.

- [31] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Biye Li, Huaian Chen, and Yi Jin. Star: Scale-wise text-conditioned autoregressive image generation. arXiv preprint arXiv:2406.10797, 2024.

- [32] Xiaoxiao Ma, Haibo Qiu, Guohui Zhang, Zhixiong Zeng, Siqi Yang, Lin Ma, and Feng Zhao. Stage: Stable and generalizable grpo for autoregressive image generation. arXiv preprint arXiv:2509.25027, 2025.

- [33] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025.

- [34] Ofir Nachum, Shixiang Shane Gu, Honglak Lee, and Sergey Levine. Data-efficient hierarchical reinforcement learning. Advances in neural information processing systems, 31, 2018.

- [35] OpenAI. Gpt-image-1, 2025. URL https://openai.com/index/introducing-4o-image-generation/.
- [36] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [37] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025.

- [38] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [39] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- [40] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.

- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

- [42] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- [43] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [44] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812, 2024.

- [45] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024.

- [46] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

- [47] Chengzhuo Tong, Ziyu Guo, Renrui Zhang, Wenyu Shan, Xinyu Wei, Zhenghao Xing, Hongsheng Li, and Pheng-Ann Heng. Delving into rl for image generation with cot: A study on dpo vs. grpo. arXiv preprint arXiv:2505.17017, 2025.

- [48] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. arXiv preprint arXiv:2311.03054, 2023.

- [49] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

- [50] Anton Voronov, Denis Kuznedelev, Mikhail Khoroshikh, Valentin Khrulkov, and Dmitry Baranchuk. Switti: Designing scale-wise transformers for text-to-image synthesis. arXiv preprint arXiv:2412.01819, 2024.

- [51] Junke Wang, Zhi Tian, Xun Wang, Xinyu Zhang, Weilin Huang, Zuxuan Wu, and Yu-Gang Jiang. Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl. arXiv preprint arXiv:2504.11455, 2025.

- [52] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

- [53] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- [54] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025.

- [55] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

- [56] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 15903–15935, 2023.

- [57] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- [58] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

- [59] Shihao Yuan, Yahui Liu, Yang Yue, Jingyuan Zhang, Wangmeng Zuo, Qi Wang, Fuzheng Zhang, and Guorui Zhou. Ar-grpo: Training autoregressive image generation models via reinforcement learning. arXiv preprint

- arXiv:2508.06924, 2025.

[60] Guohui Zhang, Hu Yu, Xiaoxiao Ma, JingHao Zhang, Yaning Pan, Mingde Yao, Jie Xiao, Linjiang Huang, and Feng Zhao. Group critical-token policy optimization for autoregressive image generation. arXiv preprint

- arXiv:2509.22485, 2025.

- [61] Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogview3: Finer and faster text-to-image generation via relay diffusion. In European Conference on Computer Vision, pages 1–22. Springer, 2024.

- [62] Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. arXiv preprint arXiv:2410.12669, 2024.

- [63] Jin Peng Zhou, Kaiwen Wang, Jonathan D Chang, Zhaolin Gao, Nathan Kallus, Kilian Q Weinberger, Kianté Brantley, and Wen Sun. q-sharp: Provably optimal distributional rl for llm post-training. CoRR, 2025.

- [64] Xianwei Zhuang, Yuxin Xie, Yufan Deng, Dongchao Yang, Liming Liang, Jinghan Ru, Yuguo Yin, and Yuexian Zou. Vargpt-v1. 1: Improve visual autoregressive large unified model via iterative instruction tuning and reinforcement learning. arXiv preprint arXiv:2504.02949, 2025.

- [65] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pages 1433–1438. Chicago, IL, USA, 2008.

## Appendix

- A Details About NextFlow

- A.1 Architecture

The NextFlow main Transformer is initialized from Qwen2.5-VL-7B [2] and augmented with a newly introduced visual codebook and a revised logits-prediction head.

- A.2 Vision Generation

For image synthesis, the model first emits the special token ⟨boi⟩ (begin-of-image) and then switches to full attention, operating in the style of VAR [46]. Scale-specific configurations are provided in Table 6.

Table 6 Related tokens corresponding to different prefix resolutions are provided. Note that the method also supports varying aspect ratios.

Resolution Related Schedule 64×64 (1,1) (2,2) (3,3) (4,4) 128×128 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) 256×256 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (10,10) (12,12) (14,14) (16,16) 512×512 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (10,10) (12,12) (14,14) (16,16) (20,20) (24,24) (28,28) (32,32) 1024×1024 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (10,10) (12,12) (14,14) (16,16) (20,20) (24,24) (28,28) (32,32) (48,48) (64,64)

- B Captions The detailed prompts for Fig. 5 and Fig. 6 is shown in Tab. 8 and Tab. 7.
- C Distribution Gap between the Training and Evaluation Datasets

Our in-house training corpus exhibits a pronounced imbalance in the number of text-rendering regions per sample: an overabundance of single-region cases and a long tail of images containing more than five regions. To mitigate this skew, we adopt a region-count–based filtering strategy for the training data. For analysis and visualization, we discretize the per-sample region count into six categories—1, 2, 3, 4, 5, and >5—using the >5 bin to summarize the long tail. The evaluation set follows a target profile restricted to bins 2–5 with probabilities 0.2, 0.3, 0.3, and 0.2, respectively. Fig. 8 compares the empirical distributions for the pre-filter training set, the post-filter training set, and the evaluation set. Importantly, this filtering does not constitute evaluation hacking or test-set leakage; rather, it calibrates the training distribution to the intended task difficulty.

- D More Visual Results The visual samples from the HPSv3 evaluation dataset for each class are shown in Fig. 9 and Fig. 10.
- E Proofs

#### E.1 Proof of Theorem 1

- Theorem 1 (Reverse-KL characterization of π†). At each state st, the constrained optimum satisfies

π†(· | st) = arg min

π∈Mπ(st)

where π∗ is the global soft-optimal policy.

KL π(· | st)∥π∗(· | st) , (8)

Pre-filter

Post-filter

Evaluation

| |
|---|

| |
|---|

| |
|---|

0.60

0.6

0.54

0.5

0.4

Probability

0.30 0.30

0.3

0.24

0.21

0.20

0.20

0.2

0.12

0.09

0.1

0.06 0.05

0.05

0.02 0.03

0.0

1 2 3 4 5 >5

Number of text rendering areas

- Figure 8 Distributions over the number of text-rendering regions (1, 2, 3, 4, 5, and >5) for the pre-filter training set, the post-filter training set, and the evaluation set. The evaluation distribution is constrained to bins 2–5 with probabilities 0.2, 0.3, 0.3, and 0.2.

Proof. We first derive the trajectory-level variational identity, then identify the global soft-optimal policy π∗ and its associated Q∗, and finally show that the constrained optimal policy π† is a reverse-KL projection of π∗ at each state.

- 1. Trajectory-level formulation of J(π). Let a trajectory be τ = (s1,a1,...,sT), (16)

and denote by pπ(τ) the trajectory distribution induced by a policy π together with the environment dynamics. For a fixed reference policy πold, let pold(τ) be the corresponding trajectory distribution.

The KL-regularized objective is

π(τ)[R(sT)] − η KL pπ(τ) pold(τ) (17)

J(π) = Ep

pπ(τ) pold(τ)

. (18)

= Ep

π(τ)[R(sT)] − η Ep

π(τ) log

Using the factorization

T−1

π(at | st)p(st+1 | st,at), (19)

pπ(τ) = p(s1)

t=1

and similarly for pold(τ) with πold, the transition dynamics and initial-state distribution cancel in the ratio, giving

T−1

π(at | st) πold(at | st)

pπ(τ) pold(τ)

. (20)

log

log

=

t=1

Thus

T−1

π(at | st) πold(at | st)

. (21)

J(π) = Ep

π(τ) R(sT) − η

log

t=1

###### 2. Variational identity and global soft-optimal trajectory distribution. Define an unnormalized density over trajectories

p˜(τ) = pold(τ)exp η 1R(sT) , (22) with partition function

old(τ) exp η 1R(sT) . (23) Define the normalized distribution

Z = p ˜(τ)dτ = Ep

1 Z

p∗(τ) =

p˜(τ) =

1 Z

pold(τ)exp η 1R(sT) . (24)

We now relate J(π) to the KL divergence KL(pπ(τ)∥p∗(τ)). Compute

pπ(τ) p∗(τ)

KL pπ(τ) p∗(τ) = pπ(τ)log

dτ (25)

pπ(τ) pold(τ) − η1R(sT) + log Z dτ (26)

= pπ(τ) log

pπ(τ) pold(τ) − η1Ep

[R(sT)] + log Z. (27)

= Ep

log

π

π

Multiplying both sides by η gives η KL pπ(τ) p∗(τ) = η KL pπ(τ) pold(τ) − Ep

[R(sT)] + η log Z. (28) Rearranging (28) yields the trajectory-level variational identity:

π

[R(sT)] − η KL pπ(τ) pold(τ) = η log Z − η KL pπ(τ) p∗(τ) . (29) Thus

Ep

π

J(π) = η log Z − η KL pπ(τ) p∗(τ) , (30) and J(π) is maximized exactly when pπ(τ) = p∗(τ).

###### 3. Factorization of p∗(τ) and the soft-optimal policy. Since pold(τ) is induced by the Markov policy πold and the environment dynamics, it factorizes as

T−1

pold(τ) = p(s1)

πold(at | st)p(st+1 | st,at). (31)

t=1

Hence

T−1

1 Z

p∗(τ) =

πold(at | st)p(st+1 | st,at) exp η 1R(sT) . (32) This defines a Markov trajectory distribution; thus there exists a policy π∗ such that

p(s1)

t=1

p∗(τ) = p(s1)

T−1

π∗(at | st)p(st+1 | st,at). (33)

t=1

We now compute π∗(at | st) explicitly. Fix a time t and a state st. Using (32), the conditional distribution over at given st under p∗ is proportional to the joint density over all trajectories sharing (st,at):

π∗(at | st) = p∗(at | st) ∝

p∗(τ) (34)

trajectories consistent with (st,at)

pold(τ) exp η 1R(sT) , (35)

∝

futures

where the sum is over the future part of the trajectory (from t onward), and the past part s1:t−1,a1:t−1 only contributes a constant factor w.r.t. at. More precisely, for fixed st we have

T−1

π∗(at | st) ∝ πold(at | st)

p(sk+1 | sk,ak)πold(ak+1 | sk+1) exp η 1R(sT) (36)

st+1:T ,at+1:T−1

k=t

exp η 1R(sT) st,at . (37) Thus we obtain the general optimal solution

= πold(at | st)Eπ

old

π∗(at | st) ∝ πold(at | st)exp

where we have defined the optimal soft Q-function as

1 η

Q∗(st,at) , (38)

Q∗(st,at) = η lnEπ

old

exp

1 η

R(sT) st,at . (39)

Equations (1) and (3) fully characterize the global soft-optimal policy π∗.

- 4. Constrained optimum as trajectory-level reverse-KL projection. By the variational identity (29), for any policy π,

J(π) = η log Z − η KL pπ(τ) p∗(τ) . (40) Therefore, when we restrict π to the constraint set Mπ (Definition 3), the optimal policy

π† = arg max

π∈Mπ

J(π) (41)

is equivalently given by

π† = arg min

π∈Mπ

KL pπ(τ) p∗(τ) . (42)

- 5. Decomposition of trajectory KL into per-state KLs. Both pπ(τ) and p∗(τ) factorize according to the same dynamics:

pπ(τ) = p(s1)

T−1

t=1

π(at | st)p(st+1 | st,at), (43)

p∗(τ) = p(s1)

T−1

t=1

π∗(at | st)p(st+1 | st,at). (44)

Hence the KL divergence simplifies to

KL pπ(τ) p∗(τ) = Ep

π(τ) log

pπ(τ) p∗(τ)

(45)

= Ep

π(τ)

T−1

t=1

log

π(at | st) π∗(at | st)

(46)

=

T−1

t=1

Es

t∼dπ KL π(· | st) π∗(· | st) , (47)

where dπ denotes the (discount-free) state visitation distribution under π at time t. The key point is that the dynamics and initial-state distribution cancel and only the action distributions appear inside the KL.

- 6. Factorized constraint and per-state reverse-KL projection. The constraint set Mπ is assumed to factorize across states:

Mπ(s), (48)

Mπ =

s

where Mπ(s) is the feasible set of action distributions at state s. Because of this product structure, choosing π ∈ Mπ amounts to choosing independently each π(· | s) ∈ Mπ(s).

Since

t∼dπ KL π(· | st) π∗(· | st) , (49)

KL pπ(τ) p∗(τ) =

Es

t

minimizing this KL over the product set Mπ = s Mπ(s) decouples into independent minimizations at each state:

π†(· | st) = arg min

KL π(· | st) π∗(· | st) . (50)

π(·|st)∈Mπ(st)

- 7. Conclusion. Combining the above steps, we conclude that the constrained optimal policy π† is obtained by,

- at each state st, projecting the global soft-optimal policy π∗ onto the feasible action-distribution set Mπ(st) in the sense of reverse KL:

This proves the theorem.

π†(· | st) = arg min

KL π(· | st)∥π∗(· | st) . (51)

π∈Mπ(st)

| |
|---|

#### E.2 Proof of Theorem 2

- Theorem 2 (Two-stage invariance). Let Vm∗(sm) be defined as in (4). Within Mπ, solving the suffix problem in (5) yields πm† :T−1; optimizing the prefix problem using Vm∗ as its sole reward gives π1:† m−1. Then, the

concatenation π† = π1:† m−1⊕πm† :T−1 uniquely maximizes the full-horizon objective J(π). Within Mπ, replacing each subpolicy by its per-state reverse-KL projection onto the factorized family yields the constrained optimum

π†. Proof. Write the full objective by splitting the action sequence into prefix and suffix:

J(π) = E R(sT) sm,πm:T−1 − η KL πm:T−1 πold, m:T−1

≜ Jsuffix(sm,πm:T−1)

, (52)

+−η KL π1:m−1 πold,1:m−1

prefix KL

and take the outer expectation over sm induced by the prefix policy π1:m−1. For any fixed sm, the inner maximization over suffix policies is exactly the standard soft-control problem whose value is Vm∗(sm). Therefore

max

J(π) = max

π

π1:m−1

E Vm∗(sm) π1:m−1 − η KL π1:m−1 πold,1:m−1 , (53)

and the maximizer is obtained by (i) choosing the suffix soft-optimal πm∗ :T−1 for each sm, and (ii) choosing the prefix policy π1:∗ m−1 that maximizes the soft value of sm. Concatenation yields the unique full-horizon maximizer π∗.

| |
|---|

[Figure 4]

###### Figure 9 Random visual samples from the HPSv3 evaluation set by class. Each class includes two pairs; within each pair, the left image is before RL and the right image is after RL.

[Figure 5]

###### Figure 10 Random visual samples from the HPSv3 evaluation set by class. Each class includes two pairs; within each pair, the left image is before RL and the right image is after RL.

Table 7 Detailed prompts used in Fig. 5.

ID Prompt Content

- 1 Six illuminated letters (’A’, ’B’, ’C’, ’N’, ’O’, ’Y’) in two rows on a dark blue background, outlined with white bulbs glowing blue.

- 2 The image is a shield-shaped graphic with a black border. Inside the shield, there are three horizontal stripes: red at the top, white in the middle, and blue at the bottom. The red stripe contains white text that reads, “Wouldn’t you rather...” The white stripe in the middle has bold black text that says, “VOTE.” And the blue stripe at the bottom features white text that reads, “By Mail.” The overall design suggests a campaign or initiative encouraging people to vote by mail.

- 3 The image features two small, round, metallic tin containers placed on a dark, textured fabric background. The lids of the tins are slightly open, revealing the contents inside. The tins have a vintage or antique appearance, with one lid displaying an engraved design. The text “M-BOX 2.0” is prominently overlaid on the image in a bold, yellow font, while “by: Jimmy-Fan” is written in a smaller, yellow font below it. The overall composition suggests a promotional or product image for the tins, textitasizing their design and branding.

- 4 The image depicts a serene nighttime camping scene in a forest. The atmosphere is illuminated by a soft, glowing light emanating from within a tent, casting a warm and inviting glow on the surrounding area. The tents are set up on a forest floor covered with pine needles and leaves. The trees are tall and dense, creating a canopy that stretches out into the night sky. The overall mood of the image is tranquil and adventurous, capturing the essence of an outdoor camping experience. In the upper right corner, there is text in Spanish that reads “DESCUBRE LO QUE ESCONDEN NUESTRO CAMPING,” and in the bottom right corner, the words “UPBAN KIDS” are displayed, with “UPBAN” in white and “KIDS” in green.

- 5 The image shows a cardboard sign with handwritten text in French. The text reads “L’ARANCAEY FIMMNA,” with “L’ARANCAEY” written in purple and “FIMMNA” in black. The sign appears to be placed against a wall, and there are some items and containers visible in the background.

- 6 The image features a cartoon-style illustration of a person standing and holding a fishing rod. The person appears to be a stylized representation of a man with light skin, wearing a white shirt, a red tie, and dark pants. The fishing rod is angled upwards, and at the end of the line, there is a white fish with an upward-pointing arrow, suggesting the fish has been caught. Above the person, there is a red banner with the text “FISHING CHALLENGE” in white letters. To the right of the person, the word “GOTCHA!” is written in white text, indicating the successful catch of the fish. The background is a solid light green color.

- 7 The image features four packets of snacks against an orange background. Each packet has a distinct design and is labeled with different types of snacks. The first packet on the left has a black and orange swirl pattern and is labeled “Apps Way Crisps.” The second packet has a yellow and black crisscross pattern and is labeled “Apps Wheat Crisps.” The third packet has black diagonal stripes and is labeled “Apps Potato Crisps.” The fourth packet has a black polka dot pattern and is labeled “Apps Cheese Balls.” The design elements and text on each packet are consistent, suggesting they are part of a branded product line.

- 8 The image shows a piece of paper with handwritten notes on it, placed on a wooden surface. The text writes: ‘VAR RL Done Right’. The handwriting is informal and includes some slang or abbreviations. The paper seems to be printed with a grid pattern, suggesting it might be a printed form or a piece of paper with a pre-defined layout.

Table 8 Detailed prompts used in Fig. 6.

ID Prompt Content

- 1 A cheerful, anthropomorphic squirrel stands upright against a solid light green background. The squirrel has a rich brown coat with a lighter cream-colored underbelly, muzzle, and inner ears, which are pink at the tips. Its large, expressive black eyes are wide and friendly, with a small, triangular nose and a wide smile showing two prominent front teeth. The squirrel’s bushy tail is curled in a spiral at the end, with a gradient from dark brown at the base to a lighter cream color at the tip. Around its neck, it wears a vibrant flower lei composed of small, colorful blossoms in red, yellow, green, purple, and blue. The squirrel holds a large, smooth, light brown egg in both front paws, which are positioned close to its chest. Its posture is upright, with its hind legs planted firmly on the ground, and it casts a soft shadow beneath it on the green background.

- 2 An anime-style illustration depicts a muscular, metallic tiger with sharp, angular features, standing on a rooftop. The tiger is in a dynamic pose, gripping a sleek, red electric guitar, and its mouth is open wide as if caught in the midst of a powerful roar or song. Above the tiger, a bright spotlight casts a dramatic beam of light, illuminating the scene and creating stark shadows on the surrounding rooftop features.

- 3 A surreal figure appears to be sculpted from intertwining tendrils of gray smoke and whirling flurries of snow, giving the impression of a man caught in a blizzard. In one hand, this ethereal being holds what looks to be a gateway to the cosmos, depicted in a photorealistic manner with vibrant nebulae and star clusters visible within its confines. The entire scene is a highly detailed octane render, showcasing sharp contrasts and the interplay of light and shadow that imbues the image with a sens.

- 4 An ornate royal carriage, painted in deep red with golden trim, stands prominently against a landscape blanketed in pristine snow. Behind it, the silhouettes of tall pine trees dusted with white can be discerned through the soft haze of a winter’s day. In front of the carriage, the snow-covered ground glistens under the subtle light of the afternoon sun.

- 5 a detailed 17th-century Dutch Baroque painting depicting a chestnut horse standing amidst a vibrant field of tulips and daisies. The horse’s mane and tail are elegantly captured by the artist, flowing with the gentle breeze that seems to animate the scene. In the background, a traditional windmill sits under a partly cloudy sky, completing this pastoral landscape.

- 6 A photograph with a standard lens style. The subject is a man standing against a background that appears to be covered in splattered paint. The man has short, dark hair and a light beard. He is wearing a long-sleeved, button-up shirt that is also splattered with paint, suggesting he might have been involved in an artistic or creative activity. The shirt is a light color, possibly beige or off-white, and has two chest pockets. The man’s expression is serious and contemplative. The background is primarily white with red and black paint splatters scattered across it. The lighting in the photograph is soft and even, highlighting the details of the man’s face and the texture of his shirt. The overall mood of the image is artistic and somewhat moody, with a focus on the subject’s expression and the abstract splatters in the background.

- 7 Two young girls stand on a lush, green grassy field, both dressed in white lace sleeveless dresses adorned with a purple satin ribbon tied in a bow at the waist. The girl in the foreground has shoulder-length, wavy blonde hair and wears a floral crown featuring white and purple flowers, including roses and possibly lisianthus, with small greenery accents. She holds a small bouquet composed of white flowers, purple blooms, and clusters of dark red berries, with some green foliage. Behind her, the second girl, also with blonde hair (slightly lighter and more tousled), wears an identical floral crown and dress. She holds a bouquet of vibrant pink flowers, likely carnations, with green stems. Both girls are smiling brightly, their eyes a striking blue, and the wind gently lifts strands of their hair. The background is a soft, out-of-focus expanse of green grass, dotted with tiny white flowers, creating a serene, natural setting.

- 8 a breathtaking photograph capturing the vibrant hues of a sunset with streaks of pink and orange painting the sky behind the majestic Grand Canyon. The canyon’s intricate rock formations are silhouetted against the illuminated backdrop, showcasing the deep crevices and towering spires. In the foreground, the Colorado River can be glimpsed winding its way through the ancient geological marvel.

