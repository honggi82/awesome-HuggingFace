## arXiv:2601.06803v2[cs.CL]20Apr2026

### Forest Before Trees: Latent Superposition for Efficient Visual Reasoning

Yubo Wang♠†*, Juntian Zhang♣*, Yichen Wu♢, Yankai Lin♣, Nils Lukas♡, Yuhan Liu♡‡ ♡MBZUAI ♠Fudan University ♣Gaoling School of Artificial Intelligence, Renmin University of China ♢Harvard University yubowang25@m.fudan.edu.cn zhangjuntian@ruc.edu.cn yuhan.liu@mbzuai.ac.ae

Code Laser

Dataset Laser-ScanPath

###### Abstract

While Chain-of-Thought empowers Large Vision-Language Models with multi-step reasoning, explicit textual rationales suffer from an information bandwidth bottleneck, where continuous visual details are discarded during discrete tokenization. Recent latent reasoning methods attempt to address this challenge, but often fall prey to premature semantic collapse due to rigid autoregressive objectives. In this paper, we propose Laser, a novel paradigm that reformulates visual deduction via Dynamic Windowed Alignment Learning(DWAL). Instead of forcing a point-wise prediction, Laser aligns the latent state with a dynamic validity window of future semantics. This mechanism enforces a "Forest-before-Trees" cognitive hierarchy, enabling the model to maintain a probabilistic superposition of global features before narrowing down to local details. Crucially, Laser maintains interpretability via decodable trajectories while stabilizing unconstrained learning via Self-Refined Superposition. Extensive experiments on 6 benchmarks demonstrate that Laser achieves state-of-the-art performance among latent reasoning methods, surpassing the strong baseline Monet by 5.03% on average. Notably, it achieves these gains with extreme efficiency, reducing inference tokens by more than 97%, while demonstrating robust generalization to out-of-distribution domains. We hope this work encourages a paradigm shift from explicit next-token prediction to latent visual reasoning.

###### 1 Introduction

Vision-Language Models (VLMs) have revolutionized visual understanding by integrating Large Language Models (LLMs) with robust visual encoders [1, 2]. While adapting Chain-of-Thought (CoT) [3] has enabled multi-step reasoning, as shown in Fig 1 (a), explicit textual rationales suffer from an information bandwidth bottleneck, where continuous visual details are lost in discrete tokenization [4]. Emerging Latent Space Reasoning approaches [5, 6] attempt to bypass this by reasoning within high-dimensional hidden states. However, these methods typically retain standard autoregressive objectives, forcing the latent state to strictly minimize prediction error for a specific token at every step. We argue that this strict point-wise mapping is fundamentally misaligned with visual perception. Unlike text generation, visual reasoning is hierarchical, evolving from global semantic apprehension to local feature extraction [7]. Forcing the latent state to prematurely "collapse"

*Equal contribution. The order was decided by a coin flip. †Work done during an internship at MBZUAI. ‡Corresponding author.

Preprint.

into a precise object token before grasping the holistic context induces a premature semantic collapse, creating a "tunnel vision" effect that hinders the capture of complex relationships.

To address this gap between sequential objectives and hierarchical perception, we propose Laser (Latent Superposition for Effective Visual Reasoning). Laser is grounded in the insight that an effective reasoning state should not be a pointer to a single future word, but a container for a validity domain of future possibilities. We redefine the optimization goal via Dynamic Windowed Alignment Learning (DWAL). As shown in Fig 1 (b), instead of predicting the immediate next token, Laser trains the latent state to align with a dynamic semantic window encompassing the entire remaining reasoning path. This formulation encourages the latent representation to maintain a probabilistic superposition, simultaneously encoding high-level global semantics while keeping specific details in a potential state. As the reasoning process unfolds, this window naturally shrinks, enforcing a progressive transition from global exploration to local precision, mimicking the general-to-specific nature of human visual processing. Achieving this "uncollapsed" state without external supervision presents a significant optimization challenge: an unconstrained latent space risks diverging into meaningless high-entropy distributions. First, we introduce Self-Refined Superposition, which leverages the model’s own estimation of the future window to construct a stable soft target.Second, to prevent loss of focus, we design an EntropyRegularized Intervention. This mechanism acts as an implicit curriculum: it dynamically injects rigid ground-truth guidance when the model’s uncertainty is high, and reverts to soft superposition when the model demonstrates a grasp of the global context. In summary, Laser transforms visual reasoning from a rigid sequential matching task into a flexible, windowed manifold alignment problem. Our extensive experiments on complex visual reasoning benchmarks demonstrate that this approach significantly outperforms both standard CoT and baseline latent reasoning methods. By enabling VLMs to "think" in superpositions before collapsing to answers, Laser bridges the gap between the continuous nature of vision and the discrete nature of language. Our contributions are summarized as follows:

[Figure 1]

What is the boy to the right of the helmet wearing?

[Figure 2]

[Figure 3]

<think> …The person standing to

the right of the helmeted person

(a)

in the picture is at…so he is wearing…</think> Answer: White T-shirt

[Figure 4]

Token Count:105

Cognitive Scanpaths:

[Figure 5]

[Figure 6]

[Skatepark, Mural,..,

| |
|---|

[Figure 7]

| |
|---|

[Figure 8]

| |
|---|

[Figure 9]

| |
|---|

Helmet, Purple, Jeans,…, Short-Sleeve, Skater Dropping In]

Laser：

[Figure 10]

Reference Superposition

Distribution

(b)

<laser_start> colorful short-sleeve white t-shirt jeans boys Skater Dropping In <laser_end> Answer: Short-Sleeve and Jeans

Weighted Original

|⋯|
|---|

[Figure 11]

Token Count:15

⋯ Helmet ⋯ Jeans Purple ⋯

Figure 1: Laser replaces verbose textual rationales (a) with efficient latent superpositions (b).

- • We propose Laser, a latent reasoning paradigm that reformulates visual deduction via Dynamic Windowed Alignment Learning (DWAL). This approach prevents premature semantic collapse by enforcing a “forest-before-trees” cognitive process within the latent space.
- • We design a supervision framework combining Self-Refined Superposition and EntropyRegularized Intervention. This establishes an implicit curriculum that stabilizes latent learning without external annotations, dynamically balancing exploration and grounding.
- • We achieve a superior efficiency-performance balance: Laser obtains state-of-the-art results across 6 benchmarks while reducing inference tokens by over 97%. Furthermore, it demonstrates robust generalization on out-of-distribution tasks, validating the efficacy of the learned visual logic.

###### 2 Related Work

###### 2.1 Vision-Language Models

The evolution of VLMs has rapidly advanced from static cross-modal alignment to dynamic perception.Foundational architectures like Flamingo and BLIP-2 pioneered efficient alignment strategies using Q-Former bottlenecks to bridge frozen vision encoders with Large Language Models (LLMs) [8, 9].Subsequently, Open-source models such as LLaVA and MiniGPT-4 demonstrated that simple

linear projection layers, coupled with high-quality visual instruction tuning, could achieve strong multimodal following [10, 11]. To overcome perceptual limitations in fine-grained and temporal tasks, recent architectures have focused on scaling visual resolution and context length. The latest InternVL3.5, scaled vision encoders to massive parameters using dynamic tiling strategies to handle detailed imagery [12, 13]. Concurrently, Qwen2.5-VL and the advanced Qwen3-VL refined the Naive Dynamic Resolution mechanism, enabling the processing of images at arbitrary aspect ratios and extending capabilities to long-context video understanding [14, 15]. Recent research has focused extensively on enhancing VLMs’ reasoning capabilities. Vision-R1 [16] and VL-Rethinker [17] utilize Group Relative Policy Optimization (GRPO) with forced “caption-reason-answer” formats and “rethinking” tokens , while VISC focuses on the advancement of multi-image reasoning capabilities [18]. In parallel, architectural innovations focus on perceptual self-improvement: ViPER [19] establishes a self-evolutionary framework via self-critiquing cycles, and DeepEyes [20] integrates active perception through dynamic tool invocation. Differently from these approaches, our Laser proposes efficient latent-space reasoning.

###### 2.2 Latent Space Reasoning

Explicit Chain-of-Thought enhances model capabilities but suffers from information loss due to discrete tokenization; consequently, LLMs like Quiet-STaR [21], Coconut [22], and SoftCoT [23] perform intermediate computations entirely within latent states. In VLMs, the focus shifts to anchoring latent thinking in visual evidence: CoCoVa [24] and MCOUT [25] refine representations via latent attention, whereas Mirage [26], IVT-LR [27], and ILVR [28] employ interleaved decoding to stabilize reasoning. While LVR [4] strengthens alignment through autoregressive reconstruction, it risks representation collapse; alternatively, “visual scratchpads” like Latent Sketchpad [29] and SkiLa [30] preserve interpretability via reconstructable latent sketches.

Beyond supervised anchoring, other methods directly optimize latent reasoning trajectories (Monet [31], LaCoT [32], DMLR [33]). Mull-Tokens [34] offers a modality-agnostic workspace, while Titans [35] targets long-range dependencies. Distinct from these, our Laser method introduces a Dynamic Windowed Alignment mechanism. By encoding global visual semantics into a compact superposition state, it achieves a superior balance between latent reasoning and efficiency.

###### 3 Methodology

In this section, we propose Laser, an efficient visual reasoning method operating in the latent space. Section 3.1 provides a formal definition of the problem. The key to realizing Laser lies in data acquisition and the design of the training methodology, which are elaborated in Section 3.2 and Section 3.3 respectively.

###### 3.1 Problem Formulation

We formulate the visual reasoning task as a two-stage conditional generation process: Latent Visual Reasoning followed by Explicit Answer Generation. Given an input image I and a textual query Q, the model Mθ aims to synthesize a chain of visual concepts C = {c1,c2,...,cT} that acts as an intermediate reasoning path, before producing the final response A = {a1,a2,...,aM}.

The core of our formulation focuses on the Latent Reasoning Trajectory. The model processes the multimodal context to generate a sequence of high-dimensional hidden states H = {h1,h2,...,hT}, where ht = Mθ(I,Q,c<t). This latent state is projected onto the vocabulary space via a linear head to obtain the logits zt = Wuht. These logits define the probability distribution over the next token via the Softmax function:

Pθ(ct+1 | I,Q,c<t) = Softmax(zt). (1)

In standard autoregressive frameworks, the optimization objective strictly forces each latent state ht to minimize the negative log-likelihood of the immediate next token ct+1. However, this rigid, local constraint compels the latent representation to collapse prematurely into a single semantic point. In this work, we redefine the optimization goal: rather than solely predicting the next token, the latent state ht is tasked with aligning with a dynamic validity domain of future visual semantics. Thus, the problem transforms from minimizing point-wise prediction error to maximizing the Windowed

Dynamic Semantic Windows

###### Dynamic Windowed Alignment Learning

###### Reference Superposition Distribution

|Step: 𝑡 − 1|
|---|

|⋯|
|---|

𝑧Ƹ𝑡

|𝑃𝑡𝑡𝑎𝑟𝑔𝑒𝑡|
|---|

𝑇

1 𝑇

|⋯|
|---|

|Step: 𝑡|
|---|

|⋯|
|---|

𝑷𝒕𝒕𝒂𝒓𝒈𝒆𝒕 𝒌 log𝑃𝜃 𝑘|𝐼,𝑄,𝑐<𝑡

𝑳𝑫𝑾𝑨𝑳 = −

෍

෍

|𝑊𝑡|
|---|

𝑡=1

𝑘∈𝑊𝑡

| |𝑃𝜃(𝑘|𝐼,𝑄,𝑐<𝑡)|
|---|---|
| | |

|Step: 𝑡 + 1|
|---|

|⋯|
|---|

𝑀

1 𝑀

|⋯|
|---|

𝑳𝑪𝑬 = −

෍

log𝑃𝜃 𝑎𝑗|𝐼,𝑄,𝐶,𝑎<𝑡

𝑗=1

[Figure 12]

<|laser_end|>

Latent Reasoning

Cognitive Scanpaths

[Figure 13]

|[Figure 14]<br><br>|[Figure 15]<br><br>[Figure 16]<br><br>|
|---|---|

[Answer] Short-Sleeve and Jeans

LM Head

|⋯|
|---|

LM Head

Last Hidden State

##### Large Vision-Language Model

Text Token

|⋯|
|---|

|⋯|
|---|

|⋯|
|---|

Special Embedding

[Image] [Text]

Laser Embedding

<|laser_start|>

<|laser_end|>

[Figure 17]

What is the boy to the right of the helmet wearing?

Text Embedding

Training Objective:

Image Embedding

𝐿𝑇𝑜𝑡𝑎𝑙 = 𝑳𝑫𝑾𝑨𝑳+ 𝑳𝑪𝑬

- Figure 2: Overview of the Laser. Laser employs DWAL. At each step t, a dynamic validity window Wt is defined over future semantic tokens to construct a Reference Superposition Distribution. The latent state is then optimized to align with this distribution via LDWAL. The final answer is generated explicitly after the reasoning using LCE.

Semantic Alignment between the latent trajectory H and the reasoning chain C, laying the theoretical foundation for the Dynamic Windowed Alignment Learning (DWAL) detailed in Sec. 3.3.

###### 3.2 Synthesizing Cognitive Scanpaths

For the Laser method, we require a dataset that captures the intermediate visual reasoning process without relying on expensive human annotations. Unlike previous approaches such as Visual CoT [36], which anchor reasoning chains with explicit bounding boxes, our aim is to bridge perception and language through implicit latent alignment. Consequently, we construct a scalable annotation pipeline that operates under a weakly supervised setting: relying solely on synthesized semantic sequences while deliberately excluding explicit Region-of-Interest (ROI) supervision.

We leverage GPT-4o as a “Visual Cognitive Engine” to sequentially synthesize reasoning paths composed of discrete semantic tokens. Crucially, our prompt design is grounded in the Global Precedence Hypothesis [7], which posits that human perception inherently prioritizes holistic structures (“forest”) before processing detailed components (“trees”). Guided by this principle, we enforce a strict Global-to-Local Scanning Logic: the synthesized sequences must initiate with a global anchor, progressively narrow focus to relevant objects, and culminate in the critical visual evidence required to answer the query. This structure ensures that the data represent valid deductive trajectories rather than static descriptions. After applying a rigorous filtering protocol, we obtain a final dataset named ScanPath, consisting of 270k high-quality samples. More details on prompt engineering and filtering statistics are provided in the Appendix H and G.

###### 3.3 Dynamic Windowed Alignment Learning

At the core of Laser is the Dynamic Windowed Alignment Learning (DWAL). Standard autoregressive objectives typically enforce a rigid, point-wise collapse to a single ground-truth token at each step, which we argue is suboptimal for early-stage visual reasoning where global context is paramount. In contrast, DWAL formulates reasoning as a Windowed Probabilistic Alignment problem. By defining a dynamic validity domain of future semantics, this approach allows the latent state to encode a superposition of potential visual concepts, mimicking the general-to-specific nature of human visual processing.

###### 3.3.1 Dynamic Semantic Windows

Let H = {h1,h2,...,hT} denote the sequence of latent hidden states generated by the model during the reasoning phase, and let C = {c1,c2,...,cT} denote the corresponding sequence of text tokens representing visual concepts, which are annotated during data construction.

Previous approaches [4] typically enforce a Strict Point-wise Mapping, minimizing the divergence D(ht,ct) between the latent state ht and the specific ground-truth visual concept ct at every timestep t. To overcome the limitations of this premature semantic collapse, we introduce a Dynamic Semantic Window Wt for each reasoning step t. This window defines the “valid semantic field” that the current state should encompass:

Wt = {ck | t ≤ k ≤ T}. (2)

The objective is to encourage the latent representation ht to cover the full spectrum of the valid window Wt, rather than peaking solely at the immediate next token ct+1. As t increases, the window Wt naturally shrinks (|Wt| → 1), enforcing a progressive transition from global semantic superposition to local precision.

Note that to ensure the reasoning process terminates effectively, the dedicated special token <laser_end> is explicitly excluded from the validity window Wt for all steps t ∈ [1,T]. Instead, <laser_end> serves as a deterministic target only after the completion of the final reasoning step T, signaling the phase transition from implicit reasoning to explicit answer generation.

###### 3.3.2 Learning via Latent Superposition

To supervise the model within the dynamic window Wt without relying on external soft labels, we employ a Self-Refined Superposition mechanism. This approach leverages the model’s own estimation of the valid semantic manifold to construct a stable soft target. Specifically, we extract the logits corresponding to the tokens in Wt and apply a stop-gradient operation to prevent unstable self-reinforcement loops. Let zˆt(k) = StopGrad(zt(k)) denote the detached logit for token k ∈ Wt. We define a reference superposition distribution Qt via a temperature-scaled Softmax:

exp(ˆzt(k)/τ) j∈Wt exp(ˆzt(j)/τ)

, (3)

Qt(k) =

where τ is a hyperparameter controlling the sharpness of the distribution. This formulation encourages the hidden state ht to maintain a probabilistic superposition of future visual concepts.

However, relying solely on soft targets can lead to optimization divergence, where the model converges to a high-entropy uniform distribution lacking semantic focus. To mitigate this, we introduce an Entropy-Regularized Intervention. We first compute the normalized entropy of the reference distribution to gauge the model’s uncertainty:

1 log |Wt| k∈W

Qt(k)log Qt(k). (4)

H(Qt) = −

t

We then construct a hybrid target Pttarget that dynamically switches between the soft superposition and a rigid next-token alignment based on this uncertainty:

Pttarget =

α · yhard + (1 − α) · Qt, if H(Qt) > η Qt, otherwise

(5)

where yhard is the one-hot vector for the immediate next token ct+1, η is a predefined entropy threshold, and α ∈ [0,1] controls the intensity of the hard intervention. This mechanism creates an implicit curriculum: it enforces precise grounding when the model exhibits high uncertainty (high entropy), while enabling superposition-based reasoning when the model has grasped the global context.

###### 3.3.3 Optimization Objective

The total optimization objective unifies the latent reasoning process and the explicit answer generation. For the reasoning chain, the DWAL Loss minimizes the cross-entropy between the hybrid target and

the model’s prediction, effectively aligning the latent trajectory with the dynamic semantic windows:

T

1 T

Pttarget(k)log Pθ(k | I,Q,c<t). (6)

LDWAL = −

t=1 k∈Wt

Subsequently, for the answer generation phase, the model produces the final response tokens A = {aj}Mj=1 based on the evolved visual understanding. We adopt the standard Cross-Entropy (CE) loss, conditioning on the image I, the original query Q, and the completed visual chain C:

M

1 M

log Pθ(aj | I,Q,C,a<j). (7)

LCE = −

j=1

The final training objective is a weighted summation of these two components:

LTotal = LDWAL + LCE. (8)

By minimizing LTotal, Laser effectively balances the exploration of global visual semantics during the reasoning phase with the exploitation of precise local semantics for answer generation.

###### 4 Experiments

We first demonstrate the superiority of Laser through extensive evaluations on 6 diverse benchmarks against state-of-the-art models. We then conduct further investigations into our method through multi-faceted studies and in-depth analysis.

###### 4.1 Experimental Setup

We instantiate Laser using Qwen2.5-VL-7B-Instruct as the backbone. To preserve the pre-trained visual representations and ensure training efficiency, we freeze the vision tower and the modality merger, exclusively optimizing the LLM parameters. Regarding our specific Laser configuration, we set the temperature τ = 1.0 to modulate the softness of the reference distribution, and the entropy threshold η = 0.6 to control the intervention mechanism. Comprehensive hyperparameters are detailed in Appendix A.

###### 4.2 Baselines

We evaluate Laser against a comprehensive set of state-of-the-art baselines across three paradigms: (1) Zero-shot VLMs (GPT-4o [1], LLaVA-OneVision [2], InternVL3.5-8B [13], Qwen2.5-VL-7B [14]), (2) Explicit Visual Interaction methods, including tool-augmented reasoning (DeepEyes [20]) and RL-enhanced VLM reasoning (Vision-R1 [16], PAPO [37], VL-Rethinker [17], and (3) Latent VLM Reasoning approaches (LVR [4], Monet [31]). Please refer to Appendix B for more details.

###### 4.3 Benchmarks

We evaluate Laser on six comprehensive benchmarks covering diverse scenarios.For visual perception, we use BLINK [38] to stress image-dependent perception such as depth and spatial cues, and MMVP [39] to probe CLIP-blind visual patterns. For visual reasoning, we adopt MMStar [40], which evaluates fine-grained reasoning axes while minimizing shortcut leakage. To probe highresolution understanding, we include HRBench [41] for ultra-high-resolution (4K only) visual perception. Moreover, we assess trustworthiness and text-rich understanding using HallusionBench [42] to diagnose visual illusion and language hallucinations (measured by Question Accuracy, Q-Acc), and SEED-Bench-2-Plus [43] to evaluate comprehensive understanding of text-intensive visuals such as charts, maps, and web pages. Each benchmark are detailed in Appendix C.

###### 4.4 Results

The comparative results across six benchmarks are presented in Table 1. Laser establishes a new state-of-the-art among latent reasoning methods and demonstrates superior capabilities even against computationally intensive explicit reasoning paradigms.

Model MMVP BLINK SEEDBENCH2PLUS MMStar Hallusion Bench HRBENCH Overall

Zero-Shot VLMs GPT-4o [1] 68.70 68.00 72.00 64.70 – – – Qwen2.5-VL-7B [44] 65.67 53.60 65.31 59.70 56.57 68.25 61.52 LLaVA-OneVision [2] 74.00 49.34 61.22 59.13 51.10 63.00 59.63 InternVL3.5-8B [45] 57.67 54.81 69.78 53.33 56.15 59.38 58.52

Tool-use & RL Enhanced Reasoning PAPO [37] 68.67 52.66 54.11 45.80 57.52 68.12 57.81 Vision-R1 [16] 72.67 52.71 68.95 62.67 63.83 75.12 65.99 VL-Rethinker [17] 72.67 55.55 70.27 63.20 71.08 63.50 66.05 DeepEyes [20] 70.00 51.08 69.08 58.73 62.57 69.12 63.43

Latent Reasoning Monet [31] 68.00 50.71 65.88 60.33 56.36 68.00 61.55 LVR [46] 64.00 53.60 47.39 57.93 65.19 53.62 56.96 Laser (Ours) 72.00 56.92 70.05 60.27 67.72 72.50 66.58

∆ ↑ 4.00 6.21 4.17 -0.33 11.36 4.50 5.03

- Table 1: Main results comparing Laser with baselines across three paradigms: Zero-Shot VLMs, Tooluse & RL, and Latent Reasoning. The best results among latent reasoning methods are highlighted in bold, and the second best are underlined. ∆ denotes the absolute performance gain over the strongest latent baseline, Monet [31].

As our primary focus, Laser significantly outperforms existing latent reasoning baselines. Compared to the previous best method, Monet, Laser achieves a remarkable +5.03% gain in the overall score. Notably, we observe the most substantial improvements on HallusionBench (+11.36%) and BLINK (+6.21%). We attribute these gains to the proposed Dynamic Windowed Alignment. By maintaining a semantic superposition rather than collapsing to a rigid token prematurely, Laser effectively mitigates the hallucination issues common in point-wise latent methods and captures fine-grained visual details. In contrast, LVR, which enforces strict next-token reconstruction, lags significantly behind (-9.62%), highlighting the necessity of our flexible windowed strategy.

Beyond the latent domain, Laser compares favorably against heavyweight paradigms. Despite operating purely in the compact latent space without external tools or iterative reinforcement learning search, Laser surpasses both the leading RL-based method, Vision-R1, and the tool-augmented VL-Rethinker. This suggests that optimizing the internal cognitive trajectory via superposition is a more efficient path to enhanced reasoning than externalizing the process into lengthy textual chains. Laser consistently outperforms its backbone model, Qwen2.5-VL-7B, across all evaluated benchmarks. Specifically, on the MMVP benchmark, which tests CLIP-blind patterns, Laser improves upon the baseline by +6.33%. This demonstrates that our method is effectively unlocks latent visual discrimination capabilities that are otherwise dormant in standard supervision settings.

###### 5 Discussions

In this section, we address five key research questions through systematic investigation, providing deeper insights into Laser’s performance and behavior.

###### RQ1: How efficient is Laser compared to baselines?

Beyond raw accuracy, the practical deployment of VLMs is often constrained by inference latency and computational cost. We analyze the inference efficiency of Laser compared to both standard baselines and recent latent reasoning methods on the BLINK and HRBench. The results are summarized in Table 2. Existing explicit reasoning approaches, such as VL-Rethinker, typically rely on generating lengthy textual rationales to bridge the gap between perception and reasoning. As shown in Table 2, this imposes a severe computational burden. For instance, on HRBench, VL-Rethinker increases token consumption by 157.2% compared to the base model, directly leading to high inference latency. Interestingly, while Monet is designed as a latent reasoning framework, it still incurs a substantial computational overhead. On BLINK, Monet uses 118.3 tokens on average. Although this is a reduction compared to the base model, it remains orders of magnitude heavier than other latent approaches, suggesting that its “visual thoughts” still occupy a dense latent sequence.

Blink (N=1901) HrBench (N=800) Avg Tokens ∆ Avg Tokens ∆

Model

Qwen2.5-VL-7B 223.5 – 55.9 – VL-Rethinker 207.0 ↓-7.4% 143.8 ↑ +157.2% Monet-7B 118.3 ↓-47.1% 86.8 ↑+55.3% LVR 8.0 ↓-96.4% 8.0 ↓-85.7% Laser 6.0 ↓-97.3% 5.7 ↓-89.7%

- Table 2: Efficiency comparison on Blink and HrBench. Our Laser achieves a significant reduction in token usage. The light yellow background indicates Explicit Reasoning methods, while the

light blue background represents Latent Reasoning methods. N denotes the total number of samples, ∆ represents the relative change compared to the Qwen2.5-VL-7B, and ↓ indicates efficiency improvement.

In contrast, Laser achieves exceptional efficiency by shifting the reasoning process from the discrete token space to the continuous latent space, significantly outperforming both explicit and latent baselines. On the BLINK benchmark, Laser reduces the average token count to merely 6.0 tokens, a reduction of 97.3%. This makes Laser substantially more efficient than Monet (118.3 tokens) and even surpasses the comparable latent method LVR (8.0 tokens).

Crucially, as noted in Table 1, this efficiency does not compromise performance. While LVR suffers from semantic degradation due to its strict reconstruction objective, Laser’s superposition mechanism allows it to encode richer, non-collapsed semantic information within the same compact latent budget. This confirms that Laser achieves a superior trade-off between efficiency and accuracy: it delivers the reasoning depth of explicit Chain-of-Thought models while maintaining the inference speed of directanswer models. By condensing reasoning into a compact superposition state, Laser eliminates the need for generating hundreds of intermediate tokens, offering near-instantaneous inference suitable for real-time applications.

###### RQ2: What is Laser’s impact across variant tasks?

Art Style Visual

[Figure 18]

Counting

Similarity

Forensic

Visual Correspondence

Detection

Functional

Spatial

Correspondence

Relation

Semantic

IQ Test

Correspondence

Relative

Jigsaw

Reflectance

Relative Depth Object Localization

Multi-View Reasoning

- Figure 3: Fine-grained comparison across 14 distinct categories. Laser outperforms Qwen2.5-VL-7B and Monet in 11 tasks, highlighting superior high-level semantic and spatial reasoning.

We further conducted an in-depth experimental analysis of Laser across a range of task domains covering perception, understanding, and reasoning. As shown in Figure 7, Laser exhibits a distinct performance profile: it achieves dominant superiority in 11 out of 14 tasks, with the only exceptions being Object Localization, Jigsaw, and Functional Correspondence. This polarization reveals a fundamental insight into the "Forest Before Trees" cognitive mechanism. Laser excels in tasks

demanding high-level discrimination, such as Visual Similarity and Spatial Relation, by avoiding the premature information loss inherent in rigid tokenization. By preserving a "probabilistic haze" of visual nuances, the model effectively soft-matches ambiguous patterns and grasps relative 3D geometry. This confirms that our Dynamic Semantic Window successfully captures the holistic "Forest" structure, prioritizing scene-level contextual coherence over isolated object identification. Conversely, Laser slightly underperforms in Object Localization and Jigsaw. This creates an intriguing contrast: while Laser is superior at understanding spatial relations, it is less precise at absolute pixel-level grounding. We attribute this to our weakly-supervised design choice. Unlike methods that rely on explicit bounding box regression (Region-of-Interest supervision), Laser learns grounding implicitly via latent alignment. It prioritizes the semantic flow of reasoning over rigid pixel reconstruction. Tasks like Jigsaw require exact low-level feature matching rather than abstract semantic reasoning. Laser’s “Forest-first” strategy naturally favors holistic scene understanding, leading the model to abstract away fine-grained pixel details. While this results in a minor trade-off in absolute localization precision, it yields the significant robustness observed in complex reasoning scenarios, mimicking human cognition, which is often semantically precise but metrically approximate.

###### RQ3: Does Laser influence general vision-language abilities?

Model Multi-View Reasoning Relative Depth Geometry Math Web Chart Qwen2.5-VL-7B 51.88 70.16 53.24 66.00 75.45 61.98 Laser (Ours) 55.64 70.97 53.24 67.20 83.48 67.16

Table 3: We compare Laser against Qwen2.5-VL-7B across three dimensions: Spatial Perception (Multi-View, Depth, Geometry), Visual Logic (Math), and Structural Understanding (Web, Charts). The results confirm that Laser successfully transfers reasoning patterns to unseen domains while preserving general spatial capabilities

Addressing the pervasive risk of catastrophic forgetting, we evaluate whether Laser maintains its general-purpose visual foundation. The results on out-of-distribution domains1confirm that our method avoids performance degradation and effectively preserves general skills. Specifically, Laser achieves substantial gains of 8.03% on Web and 5.18% on Chart tasks while maintaining a steady score of 53.24 on Geometry, suggesting that our "Forest-before-Trees" hierarchy reinforces the grasp of global structures. Furthermore, the learned deductive patterns successfully transfer to unseen logic domains, evidenced by improvements of 1.20% in Math and 0.81% in Relative Depth. As shown in Table 3, Laser boosts specialized reasoning without compromising the model’s robust general capabilities.

###### RQ4: Is Laser’s reasoning interpretable?

Q: Where is the seated person visible in the image?

[Figure 19]

- A: On the fence.
- B: In the field
- C: Outside the field.
- D: Behind the bench

|Step Token|
|---|
|0 Se Spect Seats Crowd Audience|
|1 Field Fenc Behind Out|
|2 Outside C <|laser_end|>|

- Figure 4: Visualization of the latent cognitive trajectory. The decoded tokens reveal a structured multi-hop reasoning path, evolving from entity localization (Step 0: Seats) to spatial analysis (Step 1: Fence) and final deduction.

1Task sources: Multi-View and Relative Depth (Blink), Geometry (Geo), Math (MMStar), and Web/Chart (SEED-Bench-2+).

A pivotal advantage of Laser is its inherent interpretability, derived from the rigorous alignment between the visual projector and the LLM’s semantic space. Unlike the opaque continuous vectors found in standard latent reasoning models, Laser’s hidden states can be directly projected onto the vocabulary via the frozen LM head. This allows us to inspect the top-k tokens at each intermediate step, effectively visualizing the model’s “cognitive trajectory.” We demonstrate this capability using a representative case from MMStar, where the model must locate “the seated person” in a baseball scene dominated by a salient pitcher in the foreground. Initially (Step 0), the decoded tokens, led by “Se-” (suggesting Seats), “Spect-” (Spectators), and “Crowd”, reveal that the model successfully overcomes visual saliency bias, shifting its attention to the background audience. Subsequently (Step 1), the latent state evolves to encode spatial constraints, with tokens such as “Fence,” “Behind,” and “Out” emerging to define the boundary between the spectators and the field. Finally (Step 2), this spatial reasoning converges into a semantic decision, as the probability mass shifts to “Outside” and the correct option label “C.” This trajectory confirms that Laser performs explicit-like multi-hop reasoning, progressing from entity localization to spatial analysis and final deduction, entirely within the compact latent space.

###### RQ5: How critical is each component?

80

Larse (Ours) Laser-w/o DWAL Larse-w/o windows

| |
|---|

75

72.5

| |
|---|

72.0

70.0

70

67.7

67.6

67.4

66.6

64.6

65

63.8

63.6

61.2

Score

60.3

60.0

60

58.7

58.5

56.9

55.7

55

53.4

52.9 53.3

50

48.8

45

40

MMStar MMVP BLINKSEEDBENCH2PLUSHallusionBench HRBENCH Overall

- Figure 5: Ablation study. We contrast the full Laser model with variants lacking the DWAL (w/o DWAL) and the dynamic windows (w/o Windows). The consistent performance gap across six benchmarks highlights the necessity of the proposed Dynamic Windowed Alignment Learning for effective visual reasoning.

To investigate the mechanisms behind the performance gains of Laser, we conduct an ablation study isolating the superposition objective and the windowing strategy. As illustrated in Figure 5, the results reveal that these components serve distinct roles across different task types. First, removing the DWAL objective (reverting to standard next-token prediction) leads to a significant drop in fine-grained perception benchmarks. This empirically validates that probabilistic superposition is crucial for preventing the “premature semantic collapse” often observed in standard autoregressive models. Second, using a fixed validity window (“w/o Windows”) primarily impairs performance on complex reasoning tasks, with less impact on pure perception. This confirms that the Dynamic Window strategy is essential for enforcing the “Forest-before-Trees” hierarchy. By progressively shrinking the semantic scope, it ensures the model captures global context before focusing on local details.

- 6 Conclusion

We introduce Laser, a method that transcends discrete Chain-of-Thought via continuous latent superposition. By reformulating visual deduction with Dynamic Windowed Alignment Learning, we enforce a "Forest-before-Trees" hierarchy that prevents premature semantic collapse. Laser achieves state-of-the-art performance among latent reasoning methods with superior robustness and a 97% reduction in inference overhead. Notably, it stands as the first interpretable latent reasoning approach, suggesting that emancipating reasoning from rigid tokenization fosters native multimodal intelligence.

###### References

- [1] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2024.
- [3] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [4] Bangzheng Li et al. Latent visual reasoning. arXiv preprint arXiv:2509.24251, 2025.
- [5] Georges Zelikman, Eric andovor et al. Quiet-star: Language models can teach themselves to think before speaking. arXiv preprint arXiv:2403.09629, 2024.
- [6] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, et al. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.
- [7] David Navon. Forest before trees: The precedence of global features in visual perception. Cognitive psychology, 9(3):353–383, 1977.
- [8] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.
- [9] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.
- [10] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [11] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024.
- [12] Zhe Chen, Jiannan Wu, Pipu Wang, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [13] Weiyun Wang, Zhangwei Gao, Zhe Chen, et al. Internvl 3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [14] Shuai Bai, Keqin Chen, Xuejing Liu, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [15] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025.
- [16] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

- [17] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vlrethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.
- [18] Juntian Zhang, Chuanqi Cheng, Yuhan Liu, Wei Liu, Jian Luan, and Rui Yan. Weaving context across images: Improving vision-language models through focus-centric visual chains. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27782–27798, 2025.
- [19] Juntian Zhang, Song Jin, Chuanqi Cheng, Yuhan Liu, Yankai Lin, Xun Zhang, Yufei Zhang, Fei Jiang, Guojun Yin, Wei Lin, et al. Viper: Empowering the self-evolution of visual perception abilities in vision-language model. arXiv preprint arXiv:2510.24285, 2025.
- [20] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.
- [21] Eric Zelikman, Georges Raif Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah Goodman. Quiet-star: Language models can teach themselves to think before speaking. In First Conference on Language Modeling.
- [22] Eric Zelikman, Georges Raif Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah Goodman. Quiet-star: Language models can teach themselves to think before speaking. In First Conference on Language Modeling.
- [23] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot: Soft chain-of-thought for efficient reasoning with llms. arXiv preprint arXiv:2502.12134, 2025.
- [24] Jizheng Ma, Xiaofei Zhou, Yanlong Song, and Han Yan. Cocova: Chain of continuous visionlanguage thought for latent space reasoning. arXiv preprint arXiv:2511.02360, 2025.
- [25] Tan-Hanh Pham and Chris Ngo. Multimodal chain of continuous thought for latent-space reasoning in vision-language models. arXiv preprint arXiv:2508.12587, 2025.
- [26] Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. Machine mental imagery: Empower multimodal reasoning with latent visual tokens. arXiv preprint arXiv:2506.17218, 2025.
- [27] Chao Chen, Zhixin Ma, Yongqi Li, Yupeng Hu, Yinwei Wei, Wenjie Li, and Liqiang Nie. Reasoning in the dark: Interleaved vision-text reasoning in latent space. arXiv preprint arXiv:2510.12603, 2025.
- [28] Shuai Dong, Siyuan Wang, Xingyu Liu, and Zhongyu Wei. Interleaved latent visual reasoning with selective perceptual modeling. arXiv preprint arXiv:2512.05665, 2025.
- [29] Huanyu Zhang, Wenshan Wu, Chengzu Li, Ning Shang, Yan Xia, Yangyu Huang, Yifan Zhang, Li Dong, Zhang Zhang, Liang Wang, et al. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514, 2025.
- [30] Jintao Tong, Jiaqi Gu, Yujing Lou, Lubin Fan, Yixiong Zou, Yue Wu, Jieping Ye, and Ruixuan Li. Sketch-in-latents: Eliciting unified reasoning in mllms. arXiv preprint arXiv:2512.16584, 2025.
- [31] Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025.
- [32] Guohao Sun, Hang Hua, Jian Wang, Jiebo Luo, Sohail Dianat, MAJID RABBANI, Raghuveer Rao, and Zhiqiang Tao. Latent chain-of-thought for visual reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [33] Chengzhi Liu, Yuzhe Yang, Yue Fan, Qingyue Wei, Sheng Liu, and Xin Eric Wang. Reasoning within the mind: Dynamic multimodal interleaving in latent space. arXiv preprint arXiv:2512.12623, 2025.

- [34] Arijit Ray, Ahmed Abdelkader, Chengzhi Mao, Bryan A Plummer, Kate Saenko, Ranjay Krishna, Leonidas Guibas, and Wen-Sheng Chu. Mull-tokens: Modality-agnostic latent thinking. arXiv preprint arXiv:2512.10941, 2025.
- [35] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2025.
- [36] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642, 2024.
- [37] Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025.
- [38] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.
- [39] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024.
- [40] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large visionlanguage models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.
- [41] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7907–7915, 2025.
- [42] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.
- [43] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. SEED-Bench-2-Plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024.
- [44] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [45] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [46] Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Hao Chen, Emad Barsoum, Muhao Chen, and Zicheng Liu. Latent visual reasoning. arXiv preprint arXiv:2509.24251, 2025.

# Appendix

#### Table of Contents

- A Implementation Details 14
- B Baseline Details 14
- C Benchmark Details 15
- D RL Analysis 15

- D.1 Optimization Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- D.2 Exploration Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- D.3 Composite Reward Engineering . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- D.4 Experimental Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- E Threshold Analysis for an Entropy-Adaptive Mechanism 18

- E.1 Analysis of Entropy Threshold (η) . . . . . . . . . . . . . . . . . . . . . . . . 18
- E.2 Analysis of Intervention Intensity (α) . . . . . . . . . . . . . . . . . . . . . . . 18

- F Time-Aware Semantic Decay 18
- G Prompt Engineering 19
- H Dataset Details 19
- I Details of Human Annotations 19
- J Case Study 20

###### A Implementation Details

The model is fine-tuned for 320 steps on 8 MI210 GPUs using 8 gradient accumulation steps. We utilize the AdamW optimizer with ϵ = 1e−6 and a weight decay of 0.1. The learning rate is initialized at 1e−5 with a cosine decay scheduler and a warmup ratio of 0.03. To optimize memory efficiency, we employ DeepSpeed ZeRO-3 with CPU offloading and enable Flash Attention 2. We adopt a dynamic batching strategy with a token cap of 8,192 and a maximum per-device batch size of 16. For the proposed Laser objective, when the entropy intervention is triggered, the mixing coefficient for the hard target is set to α = 0.8. The input image resolution is dynamic, ranging from 128 to 8,192 tokens (approx. 100K to 6.4M pixels).

###### B Baseline Details

Vision-R1 [16]. Vision-R1 explores R1-style post-training for MLLMs by constructing a large multimodal chain-of-thought cold start dataset and then applying RL with strategies such as progressive thinking suppression to improve multimodal reasoning.

PAPO [37]. PAPO (Perception-Aware Policy Optimization) targets the perception bottleneck in multimodal RLVR by introducing an implicit perception loss (KL term) that can be plugged into GRPO/DAPO, improving vision-dependent tasks without extra reward models or teacher models.

DeepEyes [20]. DeepEyes studies interleaved multimodal reasoning (“thinking with images”) and incentivizes tool-assisted visual reasoning behaviors via end-to-end RL, using tailored data selection and reward design to improve grounding and reduce hallucinations.

Monet [31]. Monet enables latent visual reasoning by generating continuous visual embeddings as intermediate “visual thoughts” and proposes a multi-stage distillation SFT pipeline plus visual-latent policy optimization to better train reasoning in latent visual space.

VL-Rethinker [17]. VL-Rethinker enhances slow-thinking for VLMs via RL by adapting GRPO with selective sample replay and enforcing explicit self-reflection steps, improving multi-step multimodal reasoning performance.

LLaVA-OneVision [2]. LLaVA-OneVision is a unified open multimodal model designed to perform well across single-image, multi-image, and video scenarios, with strong cross-scenario transfer from images to videos.

InternVL3.5-8B [45]. InternVL3.5 is a family of open-source multimodal models that improves reasoning and efficiency using cascade RL training and dynamic visual resolution routing; we adopt the 8B variant as a representative strong open baseline.

Qwen2.5-VL-7B [44]. Qwen2.5-VL is a flagship open VLM series with improved recognition, localization, and document/video understanding; it introduces dynamic-resolution processing and absolute time encoding for long-video comprehension. We use the 7B model as the base open VLM baseline.

LVR [46]. Latent Visual Reasoning (LVR) enables autoregressive reasoning in the visual embedding space by training the model to generate latent states that reconstruct key visual tokens, interleaved with text generation; it can be further combined with RL to balance latent reasoning and textual outputs.

###### C Benchmark Details

MMStar [40] is a vision-indispensable multimodal benchmark with carefully curated and purified samples to reduce language-only shortcuts and data contamination. It contains 1.5K questions and evaluates LVLMs over multiple core capabilities with fine-grained axes, providing a balanced diagnosis of both perception and reasoning.

MMVP [39] is constructed from 150 CLIP-blind image pairs and probes whether models can truly discriminate basic visual patterns that are obvious to humans but challenging for CLIP-like embeddings. It emphasizes failures where models produce high-confidence yet incorrect answers, often accompanied by plausible-sounding but ungrounded rationales.

BLINK [38] is a perception-centric benchmark with 3.8K multiple-choice questions, designed to test visual skills that humans can solve “within a blink,” such as correspondence, depth/geometry cues, forensics, and multi-view reasoning. Its tasks are intentionally difficult to solve from language priors alone, thereby stressing image-dependent perception.

SEEDBench2Plus [43] targets text-rich visual comprehension in real-world formats (Charts, Maps, Web pages). It uses human-annotated multiple-choice questions to assess whether MLLMs can robustly read and reason over dense visual-text content.

Hallusionbench [42] is a diagnostic benchmark for entangled language hallucination and visual illusion in LVLMs. It uses human-crafted question structures with a control-group design to quantify logical consistency and common hallucination/illusion failure modes.

HRBench [41] evaluates fine-grained perception on high-resolution (4K/8K) images, where downsampling typically removes crucial details. It systematically tests HR visual understanding (including fine-grained single-/cross-instance perception) to measure whether models can handle true HR content.

###### D RL Analysis

While our proposed method, Laser, demonstrates robust performance in the supervised fine-tuning stage, we further explore a reinforcement learning framework to align training behavior with infer-

ence dynamics and enable an autonomous “early exit” mechanism. To this end, we introduce the EPG-GRPO (Expected Policy Gradient - Group Relative Policy Optimization) algorithm. This framework integrates a variance-reduced gradient estimator with a length-invariant policy optimization scheme, successfully guiding the model to significantly reduce token consumption while maintaining performance parity.

###### D.1 Optimization Objective

Our objective combines Expected Policy Gradients (EPG) to stabilize learning within the highvariance latent space and a modified GRPO formulation to eliminate length bias.

Within LVR regions, standard single-token sampling often leads to gradient instability due to semantic ambiguity. To mitigate this, we calculate the gradient over the expectation of the Top-P subspace (STopP) rather than a single sampled token. We define the importance ratio ρw for each token w ∈ STopP and compute the expected surrogate loss L(epgt) as follows:

πθ(w|st) πold(w|st) + ϵ

, (9)

ρw =

L(epgt) = Ew∈S

[min(ρwAt,clip(ρw,1 − ϵ,1 + ϵ)At)], (10) where At is the advantage derived from group sampling.

TopP

For global optimization, standard GRPO normalizes by sequence length, which inadvertently incentivizes verbosity. We address this by standardizing the global loss using a fixed maximum completion length Lmax, independent of the actual trajectory length Ti:

Ti

B

1 B × Lmax

L(tokeni,t) + β · KL(πref∥πθ). (11)

Lpolicy =

t=1

i=1

Here, Ltoken utilizes the derived Lepg for LVR steps and the standard clipped surrogate loss for explicit tokens.

###### D.2 Exploration Strategy

We employ a composite exploration strategy to prevent convergence to local optima and robustly learn termination conditions. First, we implement Relative Norm Perturbation to enhance the diversity of deterministic LVR hidden states. We inject Gaussian noise scaled by the signal’s norm during the forward pass. For a hidden state h, the perturbed state h′ is given by:

h′ = h + λ∥h∥ ∥ϵ∥

ϵ, ϵ ∼ N(0,I), (12)

where we set the noise ratio λ = 0.05. This ensures the perturbation is significant yet non-destructive across varying signal magnitudes. Second, to encourage efficiency, we apply Stochastic Horizon Truncation. For a subset of the generated samples, the maximum allowed steps Tmax are randomly sampled from a range [Tmin,Tupper]. This forces the model to attempt convergence within limited horizons, thereby learning to optimize the efficiency of its reasoning path without relying on fixedlength priors.

###### D.3 Composite Reward Engineering

The reward function Rtotal acts as a multi-objective optimization signal, aggregating components to balance accuracy, structural validity, efficiency, and diversity:

Rtotal = Racc + Rfmt + Reff + Rdiv. (13)

Accuracy Reward (Racc). Since our evaluation relies principally on multiple-choice questions or exact answer matching, we assign a binary reward based on the correctness of the final output:

Racc =

1 if answer is correct, 0 otherwise.

(14)

Format Reward (Rfmt). This component enforces the structural integrity of the dynamic protocol. A reward rfmt is granted solely if the sequence includes the requisite start token, successfully triggers the autonomous termination token <|laser_end|>, and correctly encloses the final result in answer tags.

Efficiency Bonus (Reff). To incentivize the model to voluntarily "exit early" when confident, we introduce a dynamic efficiency bonus. This reward is conditional on three strict constraints: the

answer must be correct, the trajectory must not be forcibly truncated by the system limit (Tmax), and the output must be free of format anomalies.

Reff =

βbase − λstep · Tactual if Correct ∧ Tactual < Tmax, 0 otherwise.

(15)

Here, βbase represents the maximum potential bonus, and λstep is a penalty coefficient that reduces the reward linearly with each reasoning step used. Under this formulation, a correct answer obtained via system truncation yields zero bonus, explicitly encouraging the model to learn autonomous termination logic.

Diversity Penalty (Rdiv). To prevent "state stagnation"—where adjacent reasoning steps exhibit excessive semantic redundancy—we apply a penalty based on the squared cosine similarity between

consecutive hidden states ht and ht−1.

T−1

1 T − 1

(sim(ht,ht−1))2 . (16)

Rdiv = −λdiv ·

t=1

We utilize the squared similarity to impose a non-linear penalty that aggressively punishes highsimilarity states while remaining tolerant of the minor correlations necessary for coherent reasoning flow. The term is weighted by λdiv and normalized by trajectory length to ensure consistent scaling across varying reasoning depths.

###### D.4 Experimental Analysis

To validate the efficacy of the EPG-GRPO framework, we compare the supervised baseline (Laser) with its RL-enhanced variant (Laser + EPG). As shown in Table 4, the results demonstrate that our strategy successfully balances computational efficiency with reasoning capability.

The most significant impact is observed in inference efficiency. The RL-enhanced model reduces the average number of generated tokens by approximately 50% across dynamic benchmarks such as BLINK and HRBench. This substantial decrease confirms that the model effectively internalized the autonomous early exit mechanism incentivized by the efficiency bonus Reff and stochastic horizon truncation, learning that concise reasoning paths are often sufficient.

Crucially, this efficiency gain is achieved without compromising general performance. The overall accuracy remains stable, with the method showing particular robustness on hallucinations (HallusionBench) and multi-modal reasoning (MMStar). While minor regressions were observed in select sensitivity-heavy tasks, the Subspace-EPG objective successfully preserved the semantic richness of the latent space, preventing the catastrophic forgetting or mode collapse often associated with RL fine-tuning.

Avg. Tokens ↓ Accuracy (%)

Model

BLINK HRBench MMStar MMVP BLINK SEED Hallusion HRBench Overall

Laser (Main) 6.03 5.74 60.27 72.00 56.92 70.05 67.72 72.50 66.58 Laser + EPG 3.36 2.87 60.87 72.00 55.76 70.79 68.98 72.00 66.73

- Table 4: Comparison between the main model (Laser) and the RL-enhanced model (Laser + EPG). Avg. Tokens denotes the average number of generated tokens on BLINK and HRBench.

###### Threshold (η) Trigger MMStar MMVP BLINK SEED Hallusion HRBench Overall

η = 1.0 0.0% 59.93 69.00 56.86 69.52 67.93 71.88 65.85 η = 0.8 2.5% 60.33 69.67 56.08 69.96 68.24 71.75 66.01 η = 0.6 (Ours) 10.0% 60.27 72.00 56.92 70.05 67.72 72.50 66.58 η = 0.5 18.0% 57.40 71.67 54.97 68.95 64.56 72.75 65.05

- Table 5: Ablation study on the entropy threshold η. “Trigger” denotes the intervention activation ratio. η = 0.6 yields the best balance and is used as our default setting.

- E Threshold Analysis for an Entropy-Adaptive Mechanism

- E.1 Analysis of Entropy Threshold (η)

We first analyze the impact of the entropy threshold η by correlating it with the Trigger Ratio, defined as the percentage of tokens where the model’s high uncertainty necessitates a hard teacher intervention. As shown in Table 5, optimal performance is achieved at η = 0.6, corresponding to a trigger ratio of approximately 10%. This suggests that intervening on roughly one in ten tokens provides sufficient grounding signals to correct the reasoning trajectory without disrupting the semantic flow.

When the threshold is lowered to η = 0.5, the trigger ratio rises to 18%, imposing a stricter constraint akin to standard supervision. While this rigid guidance benefits tasks requiring precise alignment (yielding the highest score on HRBench), it stifles the latent exploration necessary for complex logic, resulting in performance degradation on reasoning-heavy benchmarks like MMStar (57.40) and HallusionBench (64.56). Conversely, higher thresholds (η = 0.8,1.0) lead to negligible intervention (< 2.5%). Although this preserves flexibility, it lacks the necessary corrective mechanism to fix visual grounding errors, leading to suboptimal results in fine-grained perception tasks such as MMVP. Thus, η = 0.6 strikes an effective balance, enforcing visual validity while maintaining cognitive flexibility.

Threshold (α) MMStar MMVP BLINK SEED Hallusion HRBench Overall

α = 0.2 60.00 70.00 55.50 69.65 68.66 72.62 66.07 α = 0.5 59.87 73.00 55.65 69.83 68.24 72.38 66.50 α = 0.8 (Ours) 60.27 72.00 56.92 70.05 67.72 72.50 66.58

Table 6: Ablation study on the parameter α. α ∈ [0,1] controls the intensity of the hard intervention. We observe that α = 0.8 achieves the optimal balance.

- E.2 Analysis of Intervention Intensity (α)

We further examine the impact of the parameter α, which modulates the intensity of the hard intervention once high uncertainty is detected. As indicated in Table 6, increasing α leads to a consistent, albeit marginal, improvement in overall performance. We observe that a lower α (e.g., 0.2) applies a softer correction, which appears insufficient to fully resolve ambiguity when the model is highly uncertain. Conversely, a higher setting (α = 0.8) provides a more decisive guidance signal, effectively acting as a “hard reset” to realign the latent trajectory with the ground truth. This suggests that during critical states of high uncertainty, prioritizing deterministic constraints over soft superposition yields slightly more robust reasoning.

- F Time-Aware Semantic Decay

We further explore a Time-Aware Semantic Decay component to regulate the semantic distribution within the validity window. This strategy applies a temporal bias to the target logits based on their relative distance from the current step:

###### z˜t(k) = zˆt(k) + ln(γk−t), (17)

where k ∈ Wt denotes the token index and γ ∈ (0,1] acts as a decay factor. This formulation allows the framework to flexibly modulate the attention density assigned to distant future tokens relative to immediate deductive steps.

Model MMStar MMVP BLINK SEED Hallusion HRBench Overall

Larse (Default) 60.27 72.00 56.92 70.05 67.72 72.50 66.58 w/ Time Decay 59.60 73.00 55.34 70.18 67.40 72.38 66.32

Table 7: Ablation study comparing the standard Laser model with a time-aware variant.

###### G Prompt Engineering

To ensure the synthesis of high-quality cognitive scanpaths, we employed the following structured system prompt. It enforces a strict “Global-to-Local” scanning logic and output format constraints.

### Role Definition You are a Visual Cognitive Engine designed to deconstruct the visual reasoning process into a strictly ordered "Visual Scanpath". ### Core Objective Generate a sequential stream of Atomic Visual Concepts. This stream must represent a logical flow of discovery: scanning from the global environment, zooming into specific objects, and accumulating visual cues, culminating in the most critical information needed to answer the user’s query. ### Precision & Format Principles (CRITICAL)

- 1. Atomic Specificity: Be as specific as the image clarity allows immediately (e.g., "Ferrari" not "Car"), but strictly use single words or 1-3 word phrases.
- 2. De-Grammatized Output: Output dense information only. REMOVE all stop words (is, the, a, of, in).
- 3. Visual Certainty: Only output concepts that are visually observable. Use broader terms if blurry.
- 4. Contextual Anchoring: Repeat a previous entity ONLY if necessary to attach the final resolution. ### The 4-Stage Scanning Logic (Strict Order)

- 1. Global Anchor (Step 1-2): Start with the broadest visible context (e.g., "Kitchen", "Blue Sky").
- 2. Subject Localization (Step 3-X): Locate the main subject relevant to the question.
- 3. Visual Evidence (Step X-Y): List visible attributes or actions supporting the answer.
- 4. Critical Resolution (Final Step): The specific concept answering the query must appear at the very end. ### Negative Constraints

- - NO Premature Reveals: Do not output the answer early.
- - NO Artificial Hierarchy: No "Fruit" → "Apple", just "Apple".
- - NO Sentences: Raw concepts only.

### Output Format Output strictly valid JSON: {

"reasoning_chain": [

- "String1",
- "String2",

... ]

}

###### H Dataset Details

Table 8 outlines the detailed statistics of the ScanPath dataset, comprising 269,773 samples across six visual domains. We measure chain complexity using “Reasoning Nodes”—defined as discrete semantic anchors (e.g., region identification or attribute verification) rather than linguistic token length. As shown in the table, the node distribution naturally aligns with task difficulty: fine-grained tasks like CUB-200 require deeper reasoning paths (Mean: 8.55 nodes) compared to basic detection tasks like OpenImages (Mean: 6.38 nodes), resulting in a balanced overall average of 7.14 nodes.

###### I Details of Human Annotations

To assess the quality of our dataset, we recruited three expert annotators (Ph.D. candidates in Computer Science) to manually evaluate 200 randomly sampled instances from ScanPath. The evaluation focused on two key dimensions: (1) the validity of the visual reasoning chain, and (2) adherence to the global-to-local logic. The results demonstrated a validity rate of 91.5% with

Data Scale Reasoning Nodes

Source Task Domain

Count Ratio Min Max Mean

Flickr30k Captioning 103,790 38.5% 2 20 7.50 GQA VQA/Reasoning 86,218 32.0% 2 18 7.13 OpenImages Detection 42,639 15.8% 3 17 6.38 Visual7W VQA 30,271 11.2% 1 17 6.85 CUB Fine-grained Cls. 3,521 1.3% 4 18 8.55 VSR Spatial Reasoning 3,334 1.2% 2 16 7.08

Total / Avg. - 269,773 100% 1 20 7.14

Table 8: Detailed statistics of the ScanPath dataset. The dataset integrates six diverse visual tasks. Nodes refers to the number of discrete semantic anchors in the reasoning chain (distinct from token length).

substantial inter-annotator agreement (Fleiss’ κ = 0.677), confirming the reliability of our automated pipeline and filtering protocol.

###### J Case Study

We present test cases of our method during inference, as illustrated in Figure xx. These cases clearly demonstrate the efficiency and accuracy of our method during the reasoning process.

[Figure 20]

Question: What is the approximate time of day in the image? Options:

- A: Morning,
- B: Evening,
- C: Noon,
- D: Night, Answer: "C", Model_Output: “<|laser_start|> Sun Forest Morning Shadows Noon <|laser_end|><answer> C </answer>"

Figure 6: A test case from MMStar showcases the efficacy and efficiency of our Laser.

[Figure 21]

[Figure 22]

[Figure 23]

Question: Which image shares the same style as the first image? Option：

- A. the second image
- B. the third image Answer: "(A)",

Model_output: "<|laser_start|> Street Art Reference Painting Abstract Third Cartoon

Second<|laser_end|><answer> A </answer>"

Figure 7: This multi-image reasoning test case from MMStar illustrates the effectiveness and efficiency of our Laser.

