## TARS: MinMax Token-Adaptive Preference Strategy for Hallucination Reduction in MLLMs

Kejia Zhang1, Keda Tao2, Zhiming Luo1*, Chang Liu4†, Jiasheng Tang3,5, and Huan Wang2*

1Xiamen University 2Westlake University 3DAMO Academy, Alibaba Group 4AWS AI Lab, Amazon 5Hupan Laboratory *Corresponding Author. †Work done prior to joining Amazon.

# arXiv:2507.21584v4[cs.CV]3Apr2026

62

LLaVA-v1.5

[Figure 1]

[Figure 2]

Preference Feedback Language Response

| |
|---|

LLaVA-v1.5-PO Industrial MLLM

60

GPT 4o

| |
|---|

Large Language Model

Ours

Ours

58

CHiP DPO

CoverageScore

[Figure 3]

56

DPO

DeepSeek VL2

Text Encoder

54

Trainable Frozen

Important Textual Token Selection

Qwen VL2.5 Intern VL2.5

[Figure 4]

Similarity Score 0.1 0.5

Token Perturbation

HALVA

52

|0.7|
|---|

|0.9|
|---|

[Figure 5]

RLAIF

Visual Input Token

LLaVA v1.5 13B

RLHF

50

Textual Input Token

Vision Encoder

48

Visual-agnostic Token

[Figure 6]

OPA DPO

Inference Procedure

10 15 20 25 30 35

Hallucination Rate (%)

Similarity Calculation

Fig. 1: Left: We present TARS, a token-adaptive preference strategy for mitigating hallucinations in MLLMs. TARS reformulates direct preference optimization (DPO) as a principled min-max optimization objective: (1) minimizes behavioral misalignment via structured preference feedback supervision and (2) maximizes distributional adaptability through controlled perturbations of visual-agnostic tokens. Right: Comprehensive evaluation on LLaVA-v1.5-13B with preference optimization (PO) [45] and various state-of-the-art MLLMs under the AMBER benchmark [71] shows that TARS consistently surpasses PO baselines, yielding results competitive with GPT-4o [33].

Abstract. Multimodal large language models (MLLMs) are prone to hallucinations, generating plausible but visually ungrounded outputs, partly because direct preference optimization (DPO) overfits to superficial linguistic cues under static preference supervision. We propose TARS, a token-adaptive preference strategy that reformulates DPO as a principled min-max optimization problem. The inner maximization selectively perturbs visual-agnostic tokens to induce worst-case distributional shifts, while the outer minimization enforces alignment with causal visual signals rather than surface-level patterns. A novel spectral alignment loss further regularizes hidden representations in the frequency domain via the Fast Fourier Transform (FFT), preserving global semantic structure without rigid token-level correspondence. We evaluate TARS across multiple hallucination benchmarks. Using only 4.8k preference samples without expert feedback, TARS reduces hallucination rates from 26.4% to 13.2% and cognition scores from 2.5 to 0.4, outperforming standard DPO by a large margin. Notably, TARS surpasses 5× LLM-based data augmentation trained on 28.8k samples (Hal-Rate: 16.0% vs. 13.2%), demonstrating that reshaping the optimization landscape via adversarial

token perturbation is fundamentally more effective than scaling training data. TARS further narrows the gap with GPT-4o on key metrics.

Keywords: Multimodal Large Language Models · Hallucination · Preference Optimization

### 1 Introduction

Multimodal large language models (MLLMs) extend the reasoning capabilities of LLMs [14,28,72] to visual inputs, enabling grounded vision-language understanding [25,32,69]. Despite strong performance across diverse tasks [37,63], MLLMs remain prone to hallucinations, producing outputs that appear plausible but are factually incorrect or lack visual grounding [29, 31, 36, 38, 60]. Mitigating such failures is crucial for deploying reliable MLLMs.

Modern MLLMs follow a two-stage pipeline of knowledge pretraining [6,21, 81] and instruction tuning [43, 44]. Hallucinations often stem not from knowledge deficits but from behavioral biases that produce plausible yet ungrounded outputs [12,50]. Preference optimization (PO) addresses this by fine-tuning with ranked response pairs from human [52,67] or AI feedback [64,80], aligning outputs with factual expectations [2,61]. Direct preference optimization (DPO) [56] is widely used for hallucination reduction [26,77], yet current methods can overfit to shallow textual cues such as high-frequency phrases [31, 46], generating plausible but visually ungrounded responses (Fig. 2(c)). Our analysis (Fig. 2(d)) further reveals that DPO-trained models assign high preference to outputs with spurious correlation tokens (e.g., prepositions or frequently mentioned objects) that lack visual grounding [70, 74]. This reliance on static preference signals hinders generalization under shifting visual-textual contexts, leading to brittle alignment and increased hallucination [26,62].

We formulate this challenge as a min-max token-adaptive alignment problem: maximizing distributional variation under semantic constraints, then minimizing preference loss under these perturbations. A natural alternative is data augmentation, which diversifies the preference distribution by generating additional training pairs. However, our experiments (Sec. 4.7) reveal that even 5× LLM-based augmentation (28.8k samples) underperforms TARS using only the original 4.8k samples (Hal-Rate: 16.0% vs. 13.2%), demonstrating that reshaping the optimization landscape via adversarial token perturbation is fundamentally more effective than merely scaling training data. The key distinction is that data augmentation merely enriches the data distribution, whereas TARS actively reshapes the optimization landscape: the inner maximization dynamically generates worst-case token configurations per sample, forcing the outer minimization to learn alignment features that are invariant to distributional shifts rather than memorizing surface-level patterns.

Specifically, we perturb visual-agnostic tokens, i.e., textual elements with minimal cross-modal grounding, to shift the input distribution without altering semantic content. This forces the model to rely on causally grounded visual

###### (c) Response of MLLM with Different Preference Strategy

- (a) Traditional DPO Strategy
- (b) Our Strategy (TARS)

[Figure 7]

[Figure 8]

Murph is standing in the professor’s office, looking at the blackboard. A cat is resting on the sofa behind her, and the room also contains a bookshelf and a piano. The presence of the cat adds a cozy feel to the otherwise serious setting.

[Figure 9]

P 𝑦𝑤 ≻ y𝑟 𝑥,𝑞)

DPO

[Figure 10]

Image 𝑥 Textual Token 𝑞

Murph is standing in the professor’s office, looking at a blackboard filled with equations. The room includes a bookshelf, a sofa, and a piano, but there are no animals present. There is no cat or any other creature in the scene.

[Figure 11]

What is happening in this image? Is there a cat or any other animal in the room?

[Figure 12]

[Figure 13]

TARS

≻

###### (d) Attention Score - DPO

(e) Attention Score - TARS

Non-hallucination Response yw

Hallucination Response yr

[Figure 14]

[Figure 15]

[Figure 16]

P 𝑦𝑤 ≻ y𝑟 𝑥,𝜑(𝑞))

Modification 𝜑(∙)

[Figure 17]

Image 𝑥 Perturbed Text Token 𝜑(𝑞)

[Figure 18]

[Figure 19]

≻

Non-hallucination

Hallucination

Response yw

Response yr

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Spurious Correlation

Causal Correlation

Spurious Correlation

Causal Correlation

Decoding Step

Decoding Step

- Fig. 2: Motivation illustration for TARS. (a) and (b) illustrate standard DPO and our token-adaptive perturbation strategy. (c) shows a VQA example where DPO hallucinates, while TARS effectively avoids ungrounded output. (d) and (e) visualize tokento-query attention maps during autoregressive decoding. DPO over-attends to spurious tokens, while TARS attends to causally grounded visual-semantic cues.

signals rather than superficial correlations (Fig. 2(b)). Supervising perturbed representations requires maintaining semantic consistency without rigid tokenlevel correspondence. Standard spatial constraints (e.g., ℓ2 or contrastive losses) indiscriminately penalize any positional deviation equally, reintroducing the spurious correlations our perturbation aims to eliminate [68,83]. We instead propose a novel spectral alignment loss via the Fast Fourier Transform (FFT). Under the shift property of the Discrete Fourier Transform, a local token perturbation at position l contributes a bounded additive term proportional to e−2πikl/L to each frequency bin k, regardless of l [20]. This means spectral magnitudes vary smoothly under local perturbation, whereas spatial metrics (ℓ2, cosine) exhibit sharp discontinuities at the perturbed positions. Meanwhile, the dominant low-frequency components capture global semantic structure (e.g., topic, intent, visual grounding), while high-frequency components encode position-dependent lexical details. This makes frequency-domain alignment a theoretically principled choice for preserving global semantic invariance while tolerating the local distributional shifts inherent in our min-max formulation (Sec. 3.3).1

We refer to this approach as TARS (token-adaptive preference strategy), a lightweight and generalizable approach that enhances preference learning by combining adaptive token perturbation with frequency-domain spectral regularization. We evaluate TARS on LLaVA-v1.5 [45] at 7B and 13B scales across generative and discriminative hallucination benchmarks. TARS achieves consistently strong performance across all benchmarks and matches GPT-4o [33] in several settings, underscoring the effectiveness of token-adaptive preference optimization. Our contributions are as follows:

- 1 Empirical comparisons between spectral and spatial alignment strategies are provided in the Appendix, Section 9.4.

- – We reformulate preference learning as a min-max optimization that dynamically generates worst-case token perturbations per sample, reshaping the optimization landscape rather than augmenting data. With only 4.8k samples, TARS outperforms 5× LLM-based augmentation (28.8k samples).
- – We propose a spectral alignment loss via FFT that preserves global semantic structure in dominant low-frequency components while tolerating local distributional shifts from token perturbation, avoiding spurious positional correlations reintroduced by spatial constraints (e.g., ℓ2, cosine).
- – We present TARS, combining adaptive token perturbation with spectral regularization. Using only 4.8k samples without expert feedback, TARS achieves state-of-the-art hallucination reduction, matching or surpassing GPT4o on key metrics while preserving general reasoning capabilities.

### 2 Preliminaries

Multimodal Large Language Models. MLLMs extend LLMs by incorporating visual inputs alongside textual prompts [81]. Formally, given an image x and a prompt q, the model generates a textual response y = (y1,...,yl) in an autoregressive manner [45]:

yt ∼ πθ(yt | y<t,x,q), (1)

where πθ denotes the conditional generation policy parameterized by θ. Given a textual input q and a visual input x, the model tokenizes them into discrete sequences: textual tokens q = {q1,...,qm} and visual tokens x = {x1,...,xn}. In practice, a pretrained vision encoder extracts patch-level features from the image, which are projected into the language model’s embedding space through a learnable alignment module. These tokens are mapped to embeddings and fused via cross-attention to integrate semantic signals from both modalities. The resulting multimodal context is then used by the decoder to autoregressively generate the output sequence [22,76].

Direct Preference Optimization. Direct preference optimization (DPO) [56] is an effective and widely adopted approach for aligning model behavior with human preferences. It bypasses explicit reward models by directly optimizing preferences from pairwise comparisons.

Traditional methods such as reinforcement learning with human feedback (RLHF) [52] and AI feedback (RLAIF) [80] rely on training a scalar reward model rψ(x,q,y) from preference pairs. This reward model is typically trained using the Bradley-Terry formulation [8]:

exp(rψ(x,q,yw)) exp(rψ(x,q,yw)) + exp(rψ(x,q,yr))

P (yw ≻ yr | x,q) =

(2)

= σ (rψ(x,q,yw) − rψ(x,q,yr)),

where (x,q,yw,yr) is sampled from the preference data distribution D, and σ(z) = 1+exp(1 −z) denotes the sigmoid function. yw and yr denote the preferred

and dispreferred responses, respectively. The reward model rψ(x,q,y) is then trained to maximize the log-likelihood of correctly ranking the preferred response over the dispreferred one during optimization:

w,yr)∼D [−log σ (rψ(x,q,yw) − rψ(x,q,yr))]. (3)

E(x,q,y

min

rψ

After training, the learned reward model rψ(x,q,y) is used to guide the finetuning of the policy πθ. Specifically, the policy is optimized to generate highreward responses while minimizing divergence from a fixed reference policy πref, typically using KL-regularized objectives:

E(x,q)∼D, y∗∼πθ − rψ(x,q,y∗)−α·DKL(πθ(y∗ | x,q)∥πref(y∗ | x,q)) . (4)

min

πθ

where α controls the strength of KL regularization, which ensures alignment with the learned preferences. Rather than relying on the explicitly trained reward model, DPO [56] simplifies the learning process by leveraging the insight that the optimal policy can be expressed in closed form using relative log-likelihoods under πθ and πref:

θ(yw|x,q)

θ(yr|x,q)

w,yr)∼D − log σ α log π

πref(yw|x,q) − α log π

πref(yr|x,q) . (5)

E(x,q,y

min

πθ

This formulation enables direct policy optimization from preference pairs, aligning the output probabilities with human preferences and improving alignment stability and sample efficiency compared to RL-based approaches.

### 3 Method

We propose a token-adaptive min-max strategy with perturbations on visualagnostic tokens and a frequency-based regularizer for improved alignment. An overview is shown in Fig. 3, and the detailed algorithm is provided in the Appendix.2 We first define notation used throughout: x denotes the visual input, q = {q1,...,qm} the tokenized textual prompt, yw and yr the preferred and dispreferred responses, πθ the trainable policy, πref the frozen reference policy, and D the preference dataset.

#### 3.1 Min-Max Reformulation of DPO

To address the limitations of traditional DPO, we reformulate preference optimization as a token-adaptive min-max game. The inner maximization introduces a controlled token-level perturbation function φ(·), which modifies selected tokens in the prompt q to induce input distribution shifts. The outer minimization

2 Full pseudocode is given in Appendix, Section 8.

[Figure 20]

Token-Adaptive Learning

Preference Learning

LLM Inference

[Figure 21]

[Figure 22]

Large Language Model

Perturbation

[Figure 23]

≻

𝑞1 𝑞ො2 𝑞3 𝑞4 ⋯ 𝑞𝑚 𝑥1 𝑥2 𝑥3 𝑥4 𝑥5 ⋯ 𝑥6

Non-hallucination

Hallucination Response yr

Maximizing Token

Response yw

Adaptive Modification

|Vision Input Tokens<br><br>| |
|---|---|
|[Figure 24]<br><br>Vision Input 𝑥| |

Token-level Modification

Spectral Preference Alignment

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

ℎref(𝑥,𝑞,𝑦) ℒfreq

[Figure 29]

𝑞1 𝑞2 𝑞3 𝑞4 ⋯ 𝑞𝑚

[Figure 30]

fft

Textual Input Tokens

Frequency Alignment

Visual-Agnostic Token Selection

ℎ𝜃(𝑥,𝜑(𝑞),𝑦)

Tokenizer 𝜌

|Text Query 𝑞<br><br>[Figure 31]|
|---|

- Fig. 3: Overview of TARS. TARS reformulates preference optimization as a Min– Max problem: (1) The maximization branch perturbs visual-agnostic tokens to simulate semantically shifted contexts (red dashed box); (2) The minimization branch fine-tunes the model to align with human preferences via the DPO objective (purple dashed box). TARS encourages the model to attend to causally grounded visual signals rather than spurious correlations, thereby reducing hallucinations.

then aligns the policy πθ with preference signals under these perturbations. Formally, we define the min–max preference objective as:

w,yr)∼D LTARS x,φ(q),yw,yr , (6)

E(x,q,y

max

min

πθ

φ∈Φ(A)

where φ is constrained to modify only visually agnostic tokens. Specifically, Φ(A) denotes the set of admissible perturbation functions: Φ(A) = {φ | {i | φ(qi) ̸= qi} ⊆ A(x,q)}, where A(x,q) is the set of token indices identified as visually agnostic (defined in Eq. (9)). This min–max objective promotes preference alignment under distributional shifts, helping to mitigate spurious correlations and reduce hallucinated outputs.

#### 3.2 Maximizing with Token Perturbations

As shown in Eq. (5), DPO aligns models with preferred responses via loglikelihood ratios against a reference model. However, we observe that this formulation can encourage overfitting to superficial patterns such as frequent phrases and stylistic tokens, which in turn reduce effective alignment with the visual context and hinder robust multimodal understanding [26,62].

To counter this, we apply token-wise maximization to introduce distribution shifts and reduce overfitting to preference signals. Formally, we define:

Sim(φ(q),q), (7)

φ(q) = arg max

φ∈Φ(A)

where Φ(A) denotes allowable perturbations constrained to A(x,q), and Sim(φ(q),q) measures token-level deviation. In practice, we approximate φ∗(q) by applying token-level transformations:

φ(q) = {I[i ∈ A(x,q)] · φ(qi) + I[i ∈/ A(x,q)] · qi}|iq=1| , (8)

where φ(qi) denotes the perturbed token, constructed using either masking (replacing with [MASK]) or synonym substitution (implementation details are in Appendix, Section 7). I[·] is the indicator function. This approximation simulates worst-case alignment uncertainty while preserving semantic integrity.

To preserve semantics, we restrict changes to visual-agnostic tokens with minimal impact on cross-modal alignment. We compute token-level visual relevance using a visual encoder Gv(·) and a text encoder Gt(·) (both instantiated by the CLIP encoder [55]) as the cosine similarity between visual features Gv(x) ∈ Rd and each token embedding Gt(qi) ∈ Rd. We then identify a set A of Nt visually agnostic tokens with the lowest cross-modal alignment scores:

A = TopN

− Gv(x)Gt(qi)⊤ , Nt = ⌊ω · ∆P−1⌋ + 1. (9)

t

where ⌊·⌋ denotes the floor operation and ω is a scaling coefficient controlling perturbation intensity. The matrix P ∈ Rm is the negated similarity score vector, with Pi = −Gv(x) · Gt(qi)T. The confidence margin ∆P = maxj Pj − maxk̸=j Pk quantifies the predictive uncertainty of the cross-modal alignment: confident predictions (large ∆P) lead to fewer perturbations, while greater uncertainty (small ∆P) induces broader variation. An empirical comparison of this adaptive selection strategy against random and uniform perturbation baselines is provided in the Appendix, Section 9.5.

#### 3.3 Spectral Regularization for Token Alignment

Token-level perturbation introduces distribution shifts, yet the supervision from preference pairs (yw,yr) is static. This discrepancy may cause the model to learn distribution-specific artifacts under strong alignment constraints [19,26].

We align in the frequency domain rather than in the spatial domain (e.g., ℓ2). The key motivation is that a spatial loss ∥z −z′∥2 penalizes every positional deviation equally; when specific tokens are perturbed, this rigid constraint reintroduces spurious positional correlations [68,83]. In contrast, by the shift property of the Discrete Fourier Transform, perturbing a single token zl affects every frequency bin k by an additive term proportional to e−2πikl/L, whose magnitude is bounded regardless of l. The spectral representation thus absorbs local perturbations into a smooth global envelope, preserving dominant low-frequency semantics while tolerating position-specific noise [20].3

Concretely, we extract hidden states for (x,φ(q),yw) and contrast them with (x,q,yw) and (x,q,yr). Let z ∈ RL×D denote a hidden-state sequence. The spectral representation is:

F(z) = Re

L−1

zl · e−2πikl/L

l=0

, for k = 0, . . . , L−1, (10)

2

- 3 A formal theoretical analysis of spectral alignment under token-level adversarial perturbation, including proofs of energy dispersion and bounded semantic shift via low-frequency masking, is provided in the Appendix, Section 9.3.

where the FFT is applied along the token axis, Re[·] extracts the real part, and | · |2 yields a scalar spectral summary. The spectral preference loss is:

Lfreq = −log σ β log F(h

F(href(x,q,yw)) − log F(h

θ(x,φ(q),yw))

θ(x,φ(q),yr))

F(href(x,q,yr)) . (11)

Here hθ(·) and href(·) are hidden states from the policy and reference models, and β is a scaling temperature. This objective extends DPO alignment to the spectral domain, improving frequency-aware consistency and reducing hallucinations from overfitting to fixed preferences.

#### 3.4 Minimization Objective in TARS

We integrate the standard DPO loss with spectral regularization to yield the final TARS training objective. Given a perturbed input φ(q) obtained from the inner maximization, and its original counterpart q, the overall loss is defined as:

LTARS(x, q, φ(q), yw, yr) = LDPO(x, φ(q), yw, yr) + λ · Lfreq(x, q, φ(q), yw, yr). (12)

where λ is a weighting coefficient that balances preference alignment and spectral consistency. This joint formulation explicitly encourages the model to preserve causal alignment with preference signals under adversarial perturbation, thereby mitigating spurious correlation. Extended ablation studies on the sensitivity of ω and λ are presented in Appendix, Section 9.

### 4 Experiments

#### 4.1 Experiment Details

Experiment Setups. We evaluate our approach on the multimodal LLM LLaVAv1.5 [45] at both 7B and 13B scales, and on Muffin-13B [26,78] (Appendix Section 11). All methods are performed with greedy decoding and a temperature of 0.

For fair comparison, we carefully align our training configuration with the most data-efficient preference optimization baselines. Specifically, we randomly sample 4.8k instances from the RLHF-V-Dataset [79], consistent with OPADPO [77], and adopt the same training strategy as CHiP-DPO [26]. All models are trained on eight NVIDIA A100 (80GB) GPUs with identical hyperparameters. We set α = 1 in Eq. (5) and β = 1 in Eq. (11) for preference optimization. We implement φ(·) using both token masking (Mask) and replacement (Replace) strategies in Eq. (8), and set the perturbation constraint strength to ω = 0.1 in the adversarial min-max formulation Eq. (9). We use a frequency-domain loss weight of λ = 0.1 in Eq. (12). Full implementation details and ablation studies are reported in the Appendix.4

Evaluation Benchmarks. We evaluate TARS across four established hallucination benchmarks spanning generative and discriminative settings: AMBER [71] for fine-grained generative hallucination (CHAIR [57], Cover, Hal-Rate,

4 See Appendix, Sections 7 and 9 for configurations and hyperparameter ablations.

- Table 1: Comparison across benchmarks. We evaluate SOTA MLLMs as references, denoted by §. For algorithms with available checkpoints, re-tested results are marked with †; for those without, we reproduce results using settings from [26,40], denoted by ‡. Bold denotes the best performance, and underlined denotes the second-best.

AMBER MMHal POPE OBJHal CHAIR↓ Cover↑ Hal-Rate↓ Cog↓ Score↑ Hal-Rate↓ Acc↑ Pre↑ CRs↓ CRi↓

Algorithm

Intern-VL2.5-7B [17]§ 7.9 54.7 37.1 3.2 3.54 0.26 - - 36.0 9.1 Qwen-VL2.5-8B [4]§ 4.6 54.6 21.1 1.3 3.29 0.27 - - 40.7 8.6 DeepSeek-VL2-27B [73]§ 2.4 56.6 16.3 0.9 2.84 0.27 - - 10.0 7.0 GPT-4o [33]§ 2.5 60.9 17.6 0.8 3.87 0.24 - - 29.3 6.7

LLaVA-v1.5-7B [45]§ 7.6 51.7 35.4 4.2 2.02 0.61 80.0 61.8 54.0 15.8 + RLHF [67]† 8.3 52.2 41.8 4.5 1.93 0.67 82.0 69.3 56.0 15.2

+ RLAIF [80]† 3.0 50.3 16.5 1.0 2.89 0.42 88.1 88.0 13.7 4.2

+ HALVA [59]† 6.9 52.8 33.2 3.5 2.12 0.59 87.5 79.6 47.3 14.6

+ DPO [40]‡ 4.9 56.6 26.4 2.5 2.19 0.61 87.8 82.0 14.0 5.0 + CHiP-DPO [26]‡ 2.9 57.3 19.9 1.0 2.32 0.57 81.1 91.8 7.3 4.3 + OPA-DPO [77]† 2.7 47.4 12.5 0.9 2.78 0.46 87.4 86.2 13.3 4.5 + TARS (Mask) 2.4 59.6 13.2 0.4 2.48 0.45 88.7 97.5 12.0 3.2

- + TARS (Replace) 2.1 59.3 14.9 0.7 2.54 0.46 87.9 97.0 13.4 3.3

LLaVA-v1.5-13B [45]§ 6.7 52.1 32.5 3.5 2.39 0.53 74.6 55.2 50.0 14.5 + RLHF [67]† 7.1 51.4 36.3 3.6 2.10 0.67 83.6 71.2 46.7 11.6 + HALVA [59]† 6.5 53.4 30.1 3.3 2.28 0.56 86.8 75.6 42.7 12.1

+ DPO [40]‡ 4.1 56.7 24.3 2.2 2.48 0.50 85.2 84.3 19.0 7.2 + CHiP-DPO [26]‡ 3.8 58.6 20.8 1.7 2.70 0.46 86.6 74.9 30.0 6.2 + OPA-DPO [77]† 2.8 48.4 13.5 1.0 3.02 0.40 87.2 80.7 18.3 5.1 + TARS (Mask) 2.1 59.8 12.5 0.6 2.89 0.45 87.6 93.0 14.6 2.8

- + TARS (Replace) 2.1 59.4 13.6 0.7 2.63 0.47 86.9 92.5 14.9 3.4

Cog), MMHal [66] for VQA hallucination scored by GPT-4V, OBJHal [79] for captioning hallucination (response-level CRs and object-level CRi), and POPE [41] for binary discriminative object hallucination. Detailed benchmark descriptions and evaluation protocols are provided in Appendix, Section 7. Baseline Methods. We compare against two categories:

- (1) Advanced multimodal foundation models: Intern-VL2.5-7B [17], QwenVL2.5-8B [4], DeepSeek-VL2-27B [73], and GPT-4o [33].
- (2) LLaVA-v1.5 with RL techniques: We evaluate multiple RL-based approaches applied to both the 7B and 13B variants of LLaVA-v1.5, including RLHF [67], RLAIF [80], HALVA [59], as well as three state-of-the-art methods based on direct preference optimization (DPO): DPO [54], CHiP-DPO [26], and OPA-DPO [77]. A comparison of algorithmic properties is provided in Tab. 4.

#### 4.2 Evaluation on Hallucination Benchmarks

Tab. 1 presents results across four hallucination benchmarks. We adopt token masking and synonym replacement as perturbation strategies (extended results on Muffin-13B are in Appendix, Section 11).

- (1) Consistent hallucination mitigation across benchmarks. On the 7B scale, TARS lowers the AMBER Hal-Rate from 35.4% to 13.2% (–22.2 pp) while raising Cover from 51.7% to 59.6% (+7.9 pp) and reducing Cog from 4.2 to 0.4. On OBJHal, CRs drops sharply from 54.0% to 12.0%.

- 0

- 1

- 2

- 3

- 4

- 5

LLaVA

DPO

TARS

AverageScore

| |
|---|

| |
|---|

Overall Attribute Object Comparinson Counting Relation Environnment Description Others

- Fig. 4: Comparison of average scores across question categories on MMHal. TARS achieves consistently higher scores, demonstrating stronger visual grounding.

- Table 2: Ablation of token-level perturbation (TP), cross-modal alignment score (CAS), and spectral preference alignment (SPA).

35

TARS (7B)

DPO (7B)

TARS (13B)

DPO (13B)

HallucinationRate(%)

30

25

AMBER OBJHal Cover↑ Hal-Rate↓ Cog↓ CRs↓ CRi↓

Algorithm

20

15

LLaVA-v1.5-7B [45] 51.7 35.4 4.2 54.0 15.8 TARS 59.6 13.2 0.4 12.0 3.2 w/o TP 56.6 26.4 2.5 14.0 5.0 w/o CAS 55.9 17.7 1.3 12.7 3.5 w/o SPA 58.3 15.1 0.7 12.5 3.7 w/o CAS&SPA 55.1 18.5 1.5 12.6 3.8

0.0 0.3 0.6 1.2 1.8 2.4 3.6 4.8

Preference Data Scale (k)

Fig. 5: Comparison of AMBER hallucination rate versus preference data scale.

- (2) Strong data efficiency under limited supervision. Using only 4.8k public preference samples without expert feedback (Tab. 4), TARS outperforms OPA-DPO on AMBER-13B in both Cover (+11.4 pp) and Hal-Rate (–1.0 pp).
- (3) Scalability across model sizes. From 7B to 13B, CHAIR improves from 2.4 to 2.1, Hal-Rate drops from 13.2% to 12.5%, and Cog stays below 0.7. TARS13B consistently surpasses all 13B baselines by 1.0–1.5 pp in Hal-Rate.
- (4) Competitiveness with larger proprietary models. At 13B, TARS approaches GPT-4o in Cover (59.8% vs. 60.9%) while achieving a notably lower Hal-Rate (12.5% vs. 17.6%), and remains competitive with DeepSeek-VL2-27B in both metrics despite its smaller scale. POPE accuracy reaches 88.7% (+8.7 pp over LLaVA-7B), confirming preserved factual grounding (breakdowns in Appendix, Section 10).5

4.3 Ablation Analyses on Component

We analyze the key TARS components through ablations in Tab. 2, focusing on three elements:

- (1) Token-level perturbation (TP) in Eq. (6), which introduces distributional shifts and proves essential for revealing token-level vulnerabilities and improving robustness. Removing it substantially increases Cog from 0.4 to 2.5, highlighting its critical role in capturing fine-grained alignment uncertainty.
- (2) Cross-modal alignment score (CAS) in Eq. (9), which targets visually agnostic tokens to preserve semantic fidelity. Its absence leads to a 4.5-point increase in hallucination and a 0.9 rise in Cog, indicating weaker suppression

- 5 TARS also improves fine-grained discriminative detection (Appendix, Section 10.1) and general multimodal understanding (Appendix, Section 10.2).

| |Pr<br><br>No Ha<br><br>|eference n-halluci llucinatio|Data nation<br><br>n| | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 1.0

0.6

- 2.2
- 3.8

5.4

- 7.0

| |Pr<br><br>No Ha<br><br>|eference n-halluci llucinatio|Data nation<br><br>n| | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1 1 3 5 7 9

1.0

0.6

2.2

3.8

5.4

7.0

| |Pr<br><br>No Ha<br><br>|eference n-halluci llucinatio|Data nation<br><br>n| | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1 1 3 5 7 9

1.0

0.6

2.2

3.8

5.4

7.0

(a) LLaVA (b) DPO (c) TARS

- Fig. 6: Distribution of hidden representations across preference-aligned, nonhallucinated, and hallucinated responses of different MLLMs. Top and right margins show marginal distributions along key feature dimensions. We extract representations from 100 preference training instances and 200 AMBER inputs across text and vision modalities. Responses to AMBER inputs are categorized as non-hallucinated or hallucinated based on factual coherence. TARS aligns with preference data while avoiding overfitting to spurious correlations, demonstrating superior factual fidelity.

of spurious correlations and reduced visual grounding. A direct comparison between adaptive and random/uniform token selection is provided in the Appendix, Section 9.5, confirming the necessity of the proposed mechanism.

(3) Spectral preference alignment (SPA) in Eq. (11), which regularizes frequency-aware consistency across token representations. Removing it increases the hallucination rate by 1.9 points and CRi from 3.2 to 3.7, suggesting degraded fine-grained factual grounding and less stable preference alignment. A comparison of spectral alignment against spatial alternatives (ℓ2, cosine) is reported in the Appendix, Section 9.4.

- 4.4 Ablation Analyses on Preference Scale Impact

1 1 3 5 7 9

We examine how preference data scale influences alignment efficiency (Fig. 5). TARS consistently outperforms DPO across all scales and exhibits sharper gains in the low-data regime: from 0 to 1.8k examples, the 7B and 13B variants reduce hallucination rates by over 15 pp, indicating that TARS captures the core alignment signal early. Beyond 3.6k examples, marginal gains saturate while performance remains stable, highlighting strong data efficiency for practical scenarios where preference annotations are costly or scarce. This behavior is consistent with the dual-space regularization discussed in Sec. 3.4: by enforcing alignment in both the output probability space and the hidden representation space, TARS extracts richer supervision from each preference pair, reducing the sample complexity needed to reach competitive performance. In contrast, standard DPO requires substantially more data to compensate for its tendency to overfit to surface-level patterns, as reflected by the persistent hallucination rate gap across all data scales in Fig. 5.

#### 4.5 Stability of Semantic Representations

We analyze how preference optimization reshapes hidden-state distributions in Fig. 6. TARS yields a more structured latent space in which hallucinated and preference-aligned representations are clearly separated, whereas DPO produces entangled clusters that interleave hallucinated and preference features, indicating overfitting to superficial signals. Crucially, TARS selectively aligns non-hallucinated responses with preference features while isolating hallucinated content, creating a semantically faithful representation space that reinforces only factually grounded outputs.

The marginal distributions (top and right panels in Fig. 6) further reveal the mechanism behind this separation. Under DPO, hallucinated and preferencealigned marginals overlap heavily, reflecting a representation space where surfacelevel patterns dominate over grounded semantics. In contrast, TARS produces near-disjoint marginals for hallucinated content while maintaining tight overlap between non-hallucinated and preference-aligned distributions. This selective clustering indicates that spectral regularization combined with adversarial perturbation shapes a representation geometry that naturally distinguishes factual from hallucinated outputs, without requiring explicit labels during training.

#### 4.6 Training Dynamics Analysis

To validate the min–max formulation (Eq. (6)), we compare training dynamics of standard DPO, random perturbation (RandPert), and TARS (Min-Max) on LLaVA-v1.5-7B over 20 epochs (Fig. 7; all losses normalized to [0,1]). DPO converges to the lowest training loss (∼0.2) but exhibits the worst robust-sample test loss (0.65–0.75), confirming severe overfitting to static preference patterns [54]. RandPert shows intermediate test performance but with heavy oscillation, as indiscriminate perturbation yields inconsistent gradients. TARS maintains a moderately higher training loss (0.35–0.4) yet achieves the lowest test loss on both clean (<0.3) and robust (<0.5) splits with stable convergence. This confirms that the min–max formulation acts as an implicit regularizer: adversarial tokenlevel shifts during training prevent premature convergence and yield alignment features that generalize to unseen and perturbed data.

#### 4.7 Comparison with Data Augmentation

We compare TARS against data augmentation baselines across all four benchmarks (Tab. 3).6 We consider paraphrasing via back-translation and LLM-based instruction augmentation at 1× and 5× expansion rates, all applied on top of standard DPO training. As shown in Tab. 3, augmentation alone improves coverage and moderately reduces hallucination, yet even 5× LLM augmentation still falls short of TARS in Hal-Rate (16.0% vs. 13.2%) and Cog (1.2 vs. 0.4). The trend is consistent across MMHal and OBJHal: augmentation narrows the gap

6 See Appendix, Section 10.3 for detailed augmentation protocols and analysis.

###### (a) Training Objective

###### (b) Test Performance (Clean vs Robust)

1.0

1.0

| |DPO<br><br>| |
|---|
<br><br>RandPert Min Max<br><br>|
|---|---|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| | |
| | |
| | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | |DPO|(clean)|| |
|---|
|RandPert|(clean)| |Min Max|(clean)| |
| | | |DPO|(robust)| |RandPert|(robust)| |Min Max<br><br>|(robust)| |
| | | | | | | | | | | | |
| || |
|---|
<br><br>| | | | | | | | | | |
| | | | || |
|---|
<br><br>|| |
|---|
<br><br>| | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

0.9

0.8

0.8

TrainLoss

0.7

TestLoss

0.6

0.6

0.5

0.4

0.4

0.2

0.3

0.2

0 2 4 6 8 10 12 14 16 18 20

0 2 4 6 8 10 12 14 16 18 20

Epoch

Epoch

- Fig. 7: Training dynamics of DPO, RandPert, and TARS (Min-Max); all losses are normalized to [0, 1]. (a) Training objective: DPO converges fastest but overfits; MinMax maintains a higher loss reflecting adversarial difficulty. (b) Test performance on clean and robust (perturbed) samples: Min-Max achieves the lowest test loss on both splits, while DPO exhibits a large clean–robust gap indicating poor robustness.

- Table 3: Comparison of TARS against data augmentation baselines on four hallucination benchmarks. Paraphrasing denotes back-translation-based response diversification; LLM Aug. denotes LLM-generated preferred responses, where 1× and 5× indicate the expansion factor relative to the original 4.8k preference set. All methods build on standard DPO. TARS is complementary to augmentation strategies.

AMBER MMHal POPE OBJHal

Method

CHAIR↓ Cover↑ Hal-Rate↓ Cog↓ Score↑ Hal-Rate↓ Acc↑ Pre↑ CRs↓ CRi↓

LLaVA-7B-DPO 4.9 56.6 26.4 2.5 2.19 0.61 87.8 82.0 14.0 5.0 + Paraphrasing 4.5 57.9 24.8 2.3 2.24 0.58 87.9 83.5 13.6 4.8 + LLM Aug. (1×) 4.1 58.8 22.9 2.0 2.30 0.55 88.1 85.8 13.2 4.5 + LLM Aug. (5×) 3.1 59.3 16.0 1.2 2.38 0.50 88.3 90.4 12.6 4.0 + TARS 2.4 59.6 13.2 0.4 2.48 0.45 88.7 97.5 12.0 3.2 + TARS & Para. 2.4 60.1 12.8 0.4 2.50 0.45 88.9 97.5 11.6 3.0 + TARS & Aug. (1×) 2.2 60.5 12.5 0.3 2.53 0.43 88.9 97.7 11.2 2.9 + TARS & Aug. (5×) 2.1 60.8 12.3 0.2 2.55 0.42 89.1 97.8 10.6 2.7

but cannot close it without TARS’s token-level robustness mechanism. Combining TARS with augmentation yields the best results on every metric (CHAIR 2.1, Hal-Rate 12.3%, CRs 10.6%), confirming that the two strategies target complementary error sources.

Discussion. Across all experiments, TARS’s improvements manifest systematically across generative (AMBER, OBJHal), discriminative (POPE), and openended (MMHal) evaluations. The ablation studies (Tab. 2) confirm that each component contributes independently, the scale analysis (Fig. 5) demonstrates strong data efficiency, and the training dynamics (Fig. 7) validate the regularization effect of the min-max formulation. This convergence of evidence across complementary evaluation axes supports the claim that token-adaptive perturbation with spectral regularization addresses a fundamental limitation of static preference optimization rather than exploiting benchmark-specific biases.

- Table 4: Comparison of preference optimization strategies. TARS achieves hallucination mitigation and causal alignment with minimal data and no expert feedback.

Algorithm Data Size Feedback Reward-Free Hal. Mitigation Causal Align.

LLaVA-v1.5 [45] - - ✗ ✗ ✗ + RLHF [67] 122k self-reward ✓ ✗ ✗ + RLAIF [80] 16k LLaVA-Next ✗ ✓ ✗ + HALVA [59] 22k GPT-4V ✗ ✗ ✗ + DPO [40] 5k self-reward ✓ ✓ ✗ + CHiP-DPO [26] 5k self-reward ✓ ✓ ✗ + OPA-DPO [77] 4.8k GPT-4V ✗ ✓ ✗ + TARS (Ours) 4.8k self-reward ✓ ✓ ✓

- 5 Related Work

Multimodal large language models (MLLMs) extend LLMs by integrating visual inputs to support multimodal reasoning [13,23,34]. Typically, visual features are extracted by a vision encoder, aligned through a connector, and processed by the LLM [45,53]. Despite strong performance, MLLMs often produce factually incorrect or visually ungrounded outputs, undermining reliability [5, 12]. This issue is more severe than in unimodal LLMs [16, 36], mainly due to modality imbalance [30, 48] and ineffective fusion [7, 35]. Recent studies attribute these failures to persistent misalignment between multimodal representations and human expectations, rather than model capacity [15,47,58].

A key bottleneck in addressing MLLM hallucinations lies in aligning model outputs with human preferences for factual consistency. Unlike knowledge-intensive pretraining [10,49] and instruction tuning [11,45], recent methods typically leverage small-scale human preference data refined via reinforcement learning [9,24, 79]. Direct preference optimization (DPO) [54, 56] has become a leading approach due to its simplicity and effectiveness, demonstrated in CHiP-DPO [26] and OPA-DPO [77]. However, DPO’s reliance on limited data can cause overfitting to superficial linguistic cues [26,62], leading to distributional rigidity and reduced adaptability to modality-specific semantics [51, 65]. These limitations call for more adaptive alignment strategies that capture token-level variability and cross-modal dependencies for stable multimodal reasoning. Adversarial training has proven effective for improving robustness in vision-language pretraining [27,82], yet its integration into preference optimization for hallucination mitigation remains largely unexplored.

To address these challenges, we propose a token-adaptive min-max alignment strategy with spectral regularization that enhances preference learning without relying on high-resource expert feedback (e.g., GPT-4V [1]). Our spectral alignment loss operates in the frequency domain via FFT, encouraging global semantic consistency while tolerating the local distributional shifts induced by token perturbation. Using only a small public preference dataset, our method effectively mitigates hallucinations and consistently outperforms RL-based baselines across benchmarks. Tab. 4 compares preference optimization methods in terms of data scale, supervision, and alignment.

### 6 Conclusion

This work reveals a fundamental limitation of static preference optimization: its tendency to latch onto superficial linguistic correlations rather than causally grounded visual evidence. We address this not by scaling data, but by reshaping the optimization landscape itself. TARS recasts preference learning as a min-max game in which the inner adversarial maximization over visual-agnostic tokens forces the model to disentangle causal visual signals from spurious textual cues, while a spectral alignment loss in the frequency domain provides a theoretically principled regularizer that preserves global semantic invariance under the resulting distributional shifts. Empirically, using only 4.8k preference samples without expert feedback, TARS halves the hallucination rate (26.4%→13.2%), reduces cognitive inconsistency by an order of magnitude (2.5→0.4), and surpasses 5× LLM-based augmentation on 28.8k samples while narrowing the gap with GPT-4o at a fraction of the model scale. These results establish that principled adversarial token perturbation can rival brute-force data scaling for robust multimodal preference alignment.

Appendix Overview

This appendix provides additional details to support the main paper. It is organized as follows:

- – Section 7 details model configurations, training settings for DPO and TARS, and token perturbation procedures.
- – Section 8 presents the min-max optimization algorithm of TARS in pseudocode form.
- – Section 9 includes extended ablation studies, covering perturbation magnitude, spectral regularization strength, a formal theoretical analysis of spectral alignment (energy dispersion and low-frequency masking bounds), spectral regularization alternatives, adaptive vs. random token selection, and min–max formulation analysis.
- – Section 10 reports additional benchmark results, including fine-grained AMBER hallucination metrics, extended MS-COCO CHAIR and multimodal understanding evaluations, comparison with data augmentation baselines, and comparison with retrieval-augmented DPO.
- – Section 11 presents results on Muffin-13B, demonstrating the generality of TARS across different MLLM architectures.
- – Section 12 discusses model behavior, including sensitivity and design insights.
- – Section 13 showcases qualitative comparisons on representative examples.

### 7 Implementation Details

#### 7.1 Base Model Setups

We evaluate our method on LLaVA-v1.5 [45] models with 7B and 13B parameters. LLaVA-v1.5 adopts Vicuna-7B/13B [18] as the language backbone and

CLIP-ViT-L/14 [55] as the vision encoder. The vision encoder also serves as the similarity function G(·) used in Eq. (9) to compute alignment between visual inputs and text tokens. All experiments are conducted using greedy decoding with a temperature of 0 to ensure deterministic outputs and reproducibility.

#### 7.2 DPO Training Setups

For fair comparison, DPO [70], CHiP [26], and TARS follow the same training protocol as described in CHiP [26]. Specifically, we set the number of epochs to 3, learning rate to 5e-7, warmup ratio to 0.03, maximum sequence length to 2048, and gradient clipping threshold to 20.0. Notably, TARS requires no taskspecific hyperparameter tuning and demonstrates generalization across different base models and datasets. All experiments are conducted on 8×A100 GPUs (80GB). Each training run takes approximately 3.0 hours on LLaVA-v1.5-7B and 3.4 hours on LLaVA-v1.5-13B.

To generate perturbed inputs, we apply two token-level adversarial strategies: replace and mask. Both are guided by token similarity scores that estimate the alignment between each text token and the visual context. The similarity matrix is normalized into perturbation scores, such that tokens with lower alignment are more likely to be modified. In replace mode, these tokens are substituted with random vocabulary tokens. In mask mode, they are replaced with a special token such as [MASK], [UNK], or [PAD], depending on tokenizer availability. Special tokens (e.g., [BOS], [EOS], [PAD]) are explicitly excluded from perturbation.

##### 7.3 Evaluation Benchmark Setups We evaluate TARS on four established hallucination benchmarks:

- 1) AMBER [71] (Generative): A fine-grained benchmark for hallucination evaluation. Following prior works [26, 77], we evaluate only the generative subset using the official codebase. Metrics include CHAIR [57] for object hallucination detection, object coverage (Cover) for completeness, response-level hallucination rate (Hal-Rate), and alignment with human cognition (Cog).
- 2) MMHal [66] (Generative): A VQA benchmark with real-world scenarios, evaluated using GPT-4V feedback to measure overall scores and hallucination rates (Hal-Rate).
- 3) OBJHal [79] (Generative): A benchmark evaluating hallucinations in image captioning. We report hallucination rates at the response level (CRs) and object mention level (CRi).
- 4) POPE [41] (Discriminative): A binary VQA benchmark designed to assess object hallucination through yes/no questions, using popular, random, and adversarial sampling strategies.

We follow the original evaluation settings and benchmark splits for AMBER, MMHal, and OBJHal as specified in their respective papers. For POPE, we construct a benchmark of 9,000 VQA pairs by sampling using the three strategies above.

For evaluation metrics, we adopt four response-level hallucination measures across different benchmarks: CHAIR [57] for object hallucination detection, object coverage (Cover) for completeness measurement, response-level hallucination rate (Hal-Rate) for overall hallucination assessment, sentence-level hallucination rate (CRs) for holistic response evaluation, and object mention-level hallucination rate (CRi) for fine-grained object-level analysis.

For evaluation feedback collection, we employ the en-core-web-lg English NLP pipeline for AMBER to extract structured semantic cues as lightweight and reproducible evaluators. For MMHal and OBJHal, we utilize the expert GPT-

- 4V model [33] (gpt-4-1106-vision-preview) for feedback evaluation, following the established protocols.

### 8 Algorithm Flowchart

We present the full training procedure of TARS in Algorithm 1, which explicitly decomposes the learning process into two stages: a maximization phase that generates token-level perturbations based on visual relevance (Max Part), and a minimization phase that optimizes the model with preference supervision (Min Part). This min-max formulation allows TARS to effectively regularize overconfident preference patterns by injecting controlled distributional shifts during training. The maximization step identifies visually agnostic tokens and perturbs them via masking or replacement, while the minimization step jointly optimizes a DPO loss and a frequency-domain alignment objective. Overall, TARS effectively suppresses spurious token-visual correlations and significantly reduces hallucinations in multimodal preference optimization.

### 9 Extended Ablation Studies

#### 9.1 Impact of Token-Level Perturbation Magnitude

We vary the token-level perturbation ratio ω and report results in Tab. 5 to investigate how perturbation strength affects model performance. In our method, Eq. (9) governs the selection of tokens for perturbation based on their visual irrelevance. Specifically, we compute the similarity between visual features Gv(x) and text token embeddings Gt(qi) to estimate token-level visual alignment. Tokens with the lowest scores are considered visual-agnostic and thus are most eligible for perturbation.

As shown in Tab. 5, moderate values of ω lead to optimal hallucination suppression across both AMBER and OBJHal. Excessively low or high perturbation strengths either under-regularize or destabilize training. When ω is too small (e.g., 1e−4), the induced distributional shift is limited, resulting in marginal improvement over the baseline. Conversely, overly large values (e.g., 5e−3 or 1e−2) introduce excessive perturbation into visual regions, disrupting the semantic coherence of inputs. The best results are obtained at ω = 1e−3, which achieves a balance between perturbation diversity and input integrity.

Algorithm 1 TARS Training Procedure

Inputs: Trainable policy πθ, reference policy πref, and preference dataset D =

{x, q, yw, yr}N. Encoders: Visual encoder Gv; text encoder Gt. Hyperparameters: DPO scaling α, perturbation ratio ω, frequency scaling β, loss

weight λ.

- 1: for each epoch do
- 2: Sample preference tuple (x, q, yw, yr) ∼ D.
- 3: Max Part:
- 4: Compute token-level visual relevance: Pi = Gv(x) · Gt(qi)T.
- 5: Estimate model confidence margin: ∆P = maxj Pj − maxk̸=j Pk.
- 6: Determine adaptive perturbation budget: Nt = ω · ∆P−1 + 1.
- 7: Select visually agnostic tokens: A = TopNt(−P).
- 8: Apply controlled perturbation to obtain φ(q).
- 9: Min Part:
- 10: Compute the preference alignment loss LDPO via DPO.
- 11: Apply frequency-domain regularization Lfreq.
- 12: Compute final objective: LTARS = LDPO + λ · Lfreq.
- 13: Update πθ via gradient descent.
- 14: end for Learned Policy: Optimized policy πθ∗.

#### 9.2 Impact of Frequency Regularization

To assess the contribution of spectral regularization, we conduct an ablation study by varying the frequency loss weight λ in the TARS objective (Eq. (12)). We evaluate λ ∈ {0.01,0.02,0.05,0.10,0.20,0.50,1.00} and present results in Tab. 6.

Performance improves steadily as λ increases from 0.01 to 0.20, with hallucination rates decreasing and coverage improving. The best trade-off is achieved at λ = 0.20. Beyond this point, performance begins to degrade, particularly on MMHal and OBJHal. This trend suggests that overly aggressive regularization

- Table 5: Ablation study on the effect of token-level perturbation magnitude ω in TARS. All experiments use LLaVA-v1.5-13B as the base model. Bold results indicate the best-performing configuration.

AMBER MMHal POPE OBJHal

Perturbation Magnitude

CHAIR↓ Cover↑ Hal↓ Cog↓ Scr↑ Hal↓ Acc↑ Pre↑ CRs↓ CRi↓ Referenced Results

LLaVA-7B 7.6 51.7 35.4 4.2 2.02 0.61 80.0 61.8 54.0 15.8

ω = 1e−4 3.3 58.0 15.4 1.3 2.35 0.50 86.0 95.0 16.2 4.6 ω = 3e−4 3.0 58.3 15.1 1.1 2.38 0.50 86.9 95.4 15.4 4.3 ω = 5e−4 2.9 58.7 14.5 0.8 2.41 0.47 87.7 96.1 13.6 3.8 ω = 1e−3 2.4 59.6 13.2 0.4 2.48 0.45 88.7 97.5 12.0 3.2 ω = 5e−3 3.3 57.8 15.9 1.4 2.29 0.48 86.9 91.2 16.1 3.9 ω = 1e−2 4.0 56.9 20.2 1.9 2.23 0.51 84.0 85.7 21.9 5.6

- Table 6: Ablation study on the effect of spectral alignment weight λ. All experiments are conducted using LLaVA-v1.5-13B. Bold numbers indicate the best across each metric.

Spectral Coeff.

AMBER MMHal POPE OBJHal CHAIR↓ Cover↑ Hal↓ Cog↓ Scr↑ Hal↓ Acc↑ Pre↑ CRs↓ CRi↓

Referenced Results LLaVA-13B 6.7 52.1 32.5 3.5 2.39 0.53 74.6 55.2 50.0 14.5

- λ = 0.01 2.9 58.7 14.8 1.0 2.80 0.48 87.2 92.5 15.4 3.7
- λ = 0.02 2.7 59.0 14.1 0.8 2.83 0.46 87.5 93.2 14.8 3.4 λ = 0.05 2.6 59.3 13.5 0.6 2.85 0.46 87.6 93.5 14.7 3.1 λ = 0.10 2.4 59.6 13.2 0.4 2.88 0.45 88.2 94.3 13.2 2.9 λ = 0.20 2.1 59.8 12.5 0.6 2.89 0.45 88.5 95.0 12.8 2.8

- λ = 0.50 2.6 59.0 13.9 0.9 2.86 0.46 87.8 92.4 14.4 3.5
- λ = 1.00 3.0 58.2 15.1 1.3 2.81 0.47 86.7 91.0 15.6 4.1

may constrain the model’s ability to accommodate subtle semantic variations introduced by token-level perturbations.

#### 9.3 Theoretical Analysis of Spectral Alignment

We provide a formal justification for adopting frequency-domain alignment instead of spatial losses under token-level adversarial perturbations in TARS.

Setup. Let a hidden-state sequence be Z = (z0,...,zL−1) ∈ RL×d. A singletoken adversarial perturbation at position m induces a corrupted sequence Z˜:

z˜i = zi + δ · I(i = m), (13)

where δ ∈ Rd is bounded (∥δ∥ ≤ ϵ) and I(·) is the indicator function. The Discrete Fourier Transform (DFT) along the token axis maps Z into the frequency domain:

L−1

2π

zi e−j

L ki, k = 0,...,L−1, (14)

zˆk =

i=0

where j denotes the imaginary unit. Proposition 1 (Spectral Energy Dispersion of Local Perturbation). Under a single-token perturbation at position m, the frequency-domain deviation is:

2π

z˜ˆk = zˆk + δ e−j

L km. (15) Taking the ℓ2 norm yields:

∥z˜ˆk − zˆk∥2 = ∥δ∥2 ∀k ∈ {0,...,L−1}. (16)

Proof. By linearity of the DFT, z˜ˆk − zˆk = δ e−j2Lπkm. Since the complex exponential has unit modulus (|e−jθ| = 1), the ℓ2 norm of a real vector scaled by it is unchanged. ■

Remark. Proposition 1 reveals that a concentrated spatial spike (single-token perturbation) acts as a Dirac-like impulse whose energy is uniformly dispersed across all frequency bins k.

Theorem 1 (Bounded Semantic Shift via Low-Frequency Masking). Let the spatial alignment loss be Lspatial = ∥Z˜ − Zref∥2F. By Parseval’s theorem,

L−1

1 L

∥z˜ˆk − zˆrefk ∥22. (17)

Lspatial =

k=0

To decouple global semantics from local token noise, TARS introduces a lowfrequency alignment loss with a binary mask M (Mk = 1 for k < Kcut, and 0 otherwise):

L−1

1 L

Mk ∥z˜ˆk − zˆrefk ∥22. (18)

Lfreq =

k=0

Under the perturbation δ, the frequency loss is upper-bounded by:

Kcut L ∥δ∥2 < Lspatial. (19)

Lfreq ≤

Proof. Applying Proposition 1, each masked bin contributes at most ∥δ∥2/L to Lfreq. Summing over Kcut < L bins gives the bound. ■ Interpretation for Min-Max Optimization. In the inner maximization of TARS, the model generates local adversarial shifts δ to explore hallucination boundaries.

- – Limitation of spatial loss. Enforcing spatial alignment penalizes ∥δ∥2 entirely at the perturbed position m. This rigid positional constraint restricts the adversarial exploration space that the inner maximization is designed to exploit.
- – Advantage of spectral loss. Because the dominant semantic structure of MLLM hidden states concentrates in the low-frequency modes (i.e., ∥zˆk∥ is substantially larger for small k), the relative distortion ∥δ∥/∥zˆk∥ in the masked low-frequency regime approaches zero. Thus, Lfreq acts as a robust invariant: it forces the global semantic envelope (low frequencies) to align with the non-hallucinated reference, while safely ignoring the high-frequency positional noise induced by the min-max perturbation.

#### 9.4 Analysis of Spectral Regularization Alternatives

FFT-based regularization decomposes hidden representations into low- and highfrequency components along the token dimension. (1) Low-frequency alignment preserves global semantic structure (e.g., topic, intent, and visual grounding) under token-level perturbations, since the dominant low-frequency modes capture the overall distributional statistics of the hidden-state sequence. (2) High-frequency constraint mitigates overfitting to local lexical patterns and

- Table 7: Ablation on spectral regularization alternatives. FFT-based regularization outperforms ℓ2 and cosine baselines.

Method CHAIR↓ Cover↑ Hal-Rate↓ Cog↓ MMH↑ MMHal↓

∅ 2.8 58.3 15.1 0.7 2.29 0.52 ℓ2 2.7 58.5 15.4 0.9 2.32 0.49 Cosine 2.5 58.8 14.5 0.7 2.34 0.49 FFT 2.4 59.6 13.2 0.4 2.48 0.45

positional artifacts, improving robustness and generalization. (3) Why FFT over spatial metrics? Given hidden states H = [h1,...,hT] ∈ RT×D, an ℓ2 loss ∥H − H′∥2 treats every positional deviation equally, amplifying positiondependent noise from our token perturbations. In contrast, the FFT representation F(H) aggregates information across all positions into frequency bins; a local perturbation at position l contributes a bounded additive term ∝ e−2πikl/L to each frequency bin k, resulting in smooth spectral variation. As shown in Tab. 7, FFT-based regularization consistently outperforms ℓ2 and cosine baselines across all metrics, validating this design choice. Additionally, Tab. 15 compares spectrum-based alignment (TARS) with token-level contrastive alignment under identical perturbation policies, showing that spectral alignment achieves substantially lower hallucination rates (13.2% vs. 16.3%) and higher coverage (59.6% vs. 57.2%).

#### 9.5 Adaptive vs. Random Token Selection

A critical design choice in TARS is the adaptive selection of perturbation targets based on cross-modal alignment scores, rather than random or uniform token selection. We validate this by comparing three strategies: (1) vision-only perturbation (perturbing tokens with high visual relevance); (2) random perturbation (uniformly sampling tokens to perturb); and (3) our proposed visual-agnostic perturbation (selectively perturbing tokens with the lowest cross-modal alignment).

As shown in Tab. 8: (1) Perturbing vision-related tokens degrades visual grounding, increasing CHAIR from 4.9 to 5.8 and Hal-Rate from 26.4% to 31.6%, confirming that visual tokens carry essential grounding information that must be preserved. (2) Random perturbation provides modest improvements over DPO (Hal-Rate 28.7% vs. 26.4%) but falls far short of TARS (13.2%), as it indiscriminately modifies both semantically critical and irrelevant tokens. (3) Our adaptive visual-agnostic perturbation achieves the best results across all metrics, reducing Hal-Rate by 15.5 pp over random perturbation and providing a 2× improvement in Cog (0.4 vs. 3.1). These results directly demonstrate that the advantage of TARS comes from the principled adaptive selection mechanism, not merely from the act of perturbation itself.

- Table 8: Comparison of perturbation target strategies. Visual-agnostic perturbation (TARS) consistently outperforms alternatives.

Method CHAIR↓ Cover↑ Hal-Rate↓ Cog↓ MMH↑ MMHal↓

LLaVA-v1.5-7B 7.6 51.7 35.4 4.2 2.02 0.61 + DPO 4.9 56.6 26.4 2.5 2.19 0.61 + Vision-only Perturb. 5.8 55.2 31.6 3.8 2.12 0.66 + Random Perturb. 4.4 57.8 28.7 3.1 2.30 0.64 + TARS (ours) 2.4 59.6 13.2 0.4 2.48 0.45

#### 9.6 Min–Max Formulation Analysis

A key design choice in TARS is the adversarial min–max formulation (Eq. (6)), which replaces the static preference optimization of standard DPO with a dynamic two-stage procedure. The full empirical analysis of training dynamics is presented in the main text (Sec. 4.6, Fig. 7). Here we provide additional protocol details.

We train all three variants (standard DPO, DPO with random perturbation (RandPert), and TARS (Min-Max)) on LLaVA-v1.5-7B with identical hyperparameters using 4.8k preference samples over 20 epochs. All reported loss values are normalized to [0,1] to enable fair cross-method comparison. The test set is evaluated under two conditions: clean (original held-out samples) and robust (token-perturbed held-out samples), isolating generalization from robustness. The results confirm that: (1) DPO achieves the lowest training loss (∼0.2) but the worst robust test loss (0.65–0.75), exhibiting severe overfitting; (2) RandPert yields intermediate but highly unstable convergence; and (3) the Min-Max formulation achieves the best test performance on both clean (<0.3) and robust (<0.5) splits, acting as an implicit regularizer that prevents premature convergence to spurious patterns.

### 10 Additional Experimental Results

#### 10.1 AMBER Generative and Discriminative Metrics

We present extended results on the AMBER benchmark in Tab. 9, evaluating hallucination performance from both generative and discriminative perspectives.

The left portion of the table reports generative metrics, including CHAIR, Coverage, Hallucination Rate, and Cognitive Score. TARS achieves substantial improvements across all, reducing hallucination by over 13 points compared to DPO, and significantly improving image grounding as reflected by higher coverage and cognitive consistency.

Beyond generative evaluation, we further introduce fine-grained discriminative metrics that assess hallucination across four categories: Existence, Relation, Attribute, and Action. As shown in the right half of the table, TARS consistently outperforms both DPO and the LLaVA baseline in all dimensions.

- Table 9: Comparison of generative and fine-grained discriminative hallucination metrics on the AMBER benchmark.

Generative Discriminative CHAIR↓ Cover↑ Hal↓ Cog↓ Exist.↑ Rel.↑ Attr.↑ Act.↑

Algorithm

LLaVA-7B [45] 7.9 54.7 37.1 3.2 82.9 58.6 65.6 70.1 + DPO 4.9 56.6 26.4 2.5 87.1 59.7 74.6 79.4 + TARS 2.4 59.6 13.2 0.4 95.3 62.8 78.6 86.5

LLaVA-13B [45] 6.7 52.1 32.5 3.5 94.1 45.5 70.1 76.2 + DPO 4.1 56.7 54.3 2.2 95.0 58.8 73.1 81.5 + TARS 2.1 59.8 12.5 0.6 98.9 67.0 82.0 86.6

Table 10: Extended MS-COCO CHAIR evaluation across multiple settings.

MS-COCO CHAIR CHAIR−I CHAIR−S CHAIR1I CHAIR1S CHAIR2I CHAIR2S

LLaVA-v1.5-7B 10.5 32.7 18.8 62.7 64.3 90.7 DPO 7.2 21.4 13.6 41.9 49.8 72.3 TARS (ours) 5.9 17.2 10.8 33.6 38.5 57.9

#### 10.2 Extended Hallucination Benchmarks

We provide extended evaluation on two additional benchmark suites, MS-COCO [42] CHAIR and multimodal understanding benchmarks, to demonstrate that TARS improves hallucination suppression without degrading broader multimodal reasoning.

MS-COCO CHAIR (Tab. 10). We evaluate on the standard MS-COCO [42] CHAIR benchmark using three evaluation protocols: CHAIR− (negative-sample filtering), CHAIR1 (single-caption), and CHAIR2 (multi-caption), each reported at both instance-level (I) and sentence-level (S). TARS reduces CHAIR−I from 10.5 (LLaVA) to 5.9 (–4.6 pp) and CHAIR−S from 32.7 to 17.2 (–15.5 pp), outperforming DPO across all six metrics. The improvements are most pronounced on the harder CHAIR2 protocol (multi-caption), where TARS achieves 38.5% instance-level vs. 49.8% for DPO, a 22.7% relative reduction. These results confirm that TARS effectively mitigates object hallucination across diverse captioning settings.

LLaVA-Bench and SeedBench (Tab. 11). To verify that hallucination suppression does not impair general multimodal capabilities, we evaluate on LLaVABench [45] (conversation, detail, and reasoning sub-tasks) and SeedBench [39] (multi-choice visual reasoning). TARS improves over both the base LLaVA and DPO across all sub-tasks: the overall LLaVA-Bench score increases from 62.9 (LLaVA) to 67.2 (+4.3), and SeedBench accuracy rises from 32.4% to 38.7% (+6.3 pp). Notably, the largest gain appears in visual reasoning (+3.9 over LLaVA, +2.3 over DPO), indicating that TARS strengthens the model’s ability to perform multi-step inference grounded in visual evidence.

- Table 11: Evaluation on LLaVA-Bench and SeedBench, showing TARS improves multimodal understanding.

Method Conv. Detail Reasoning Overall SeedBench Acc.

LLaVA-v1.5-7B 61.8 53.6 71.2 62.9 32.4 DPO 63.5 55.4 72.8 64.6 37.1 TARS (ours) 65.9 57.8 75.1 67.2 38.7

#### 10.3 Comparison with Data Augmentation Baselines

A natural question is whether the gains of TARS can be replicated by simply augmenting the preference training data. We conduct a systematic comparison against two families of data augmentation strategies applied on top of standard DPO:

- (1) Paraphrasing. We use GPT-3.5-Turbo to back-translate each preferred response through Chinese and French, then paraphrase back to English. This yields one additional variant per preference pair, preserving semantic content while diversifying surface forms. The augmented pairs are mixed with the original

4.8k samples during DPO training.

- (2) LLM-based augmentation. We prompt an instruction-tuned LLM (Vicuna13B [18]) to generate new preferred responses conditioned on the same imagequestion pairs, using the original preferred response as a reference. We experiment with two scales: 1× (one new response per pair, yielding 9.6k total) and 5× (five new responses per pair, yielding 28.8k total). Generated responses are filtered by CLIP score to remove low-quality samples before mixing with the original data.

Protocol. All models use LLaVA-v1.5-7B as the base and train with identical hyperparameters (learning rate, batch size, number of epochs) to isolate the effect of data composition. The only variable is the training data: original 4.8k samples vs. augmented sets.

- Analysis. As shown in Tab. 12, augmentation provides incremental improvements: paraphrasing reduces Hal-Rate from 26.4% to 24.8% (–1.6 pp), and 5× LLM augmentation further lowers it to 16.0% (–10.4 pp). However, even the most aggressive augmentation (5×, using 6× more data) still underperforms TARS on every metric (Hal-Rate 16.0% vs. 13.2%, Cog 1.2 vs. 0.4). This gap demonstrates that TARS provides a fundamentally different and stronger form of robustness that cannot be recovered through data scaling alone.

Crucially, TARS and augmentation are complementary: combining TARS with 5× LLM augmentation achieves the best overall performance (CHAIR 2.1, Cover 60.8%, Hal-Rate 12.3%). This suggests that augmentation diversifies the preference distribution while TARS sharpens the model’s sensitivity to tokenlevel hallucination cues, and the two mechanisms address orthogonal failure modes.

- Table 12: Comparison of TARS against data augmentation baselines across four hallucination benchmarks. Paraphrasing denotes back-translation-based response diversification; LLM Aug. denotes LLM-generated preferred responses, where 1× and 5× indicate the expansion factor relative to the original 4.8k preference set. All methods build on standard DPO. TARS is complementary to augmentation strategies.

AMBER MMHal POPE OBJHal

Method

CHAIR↓ Cover↑ Hal-Rate↓ Cog↓ Score↑ Hal-Rate↓ Acc↑ Pre↑ CRs↓ CRi↓

LLaVA-v1.5-7B-DPO 4.9 56.6 26.4 2.5 2.19 0.61 87.8 82.0 14.0 5.0 + Paraphrasing 4.5 57.9 24.8 2.3 2.24 0.58 87.9 83.5 13.6 4.8 + LLM Aug. (1×) 4.1 58.8 22.9 2.0 2.30 0.55 88.1 85.8 13.2 4.5 + LLM Aug. (5×) 3.1 59.3 16.0 1.2 2.38 0.50 88.3 90.4 12.6 4.0 + TARS 2.4 59.6 13.2 0.4 2.48 0.45 88.7 97.5 12.0 3.2 + TARS & Paraphrasing 2.3 60.1 12.8 0.4 2.50 0.44 88.8 97.6 11.5 3.1 + TARS & LLM Aug. (1×) 2.2 60.5 12.5 0.3 2.53 0.43 88.9 97.7 11.3 3.1 + TARS & LLM Aug. (5×) 2.1 60.8 12.3 0.2 2.55 0.42 89.1 97.8 10.6 2.9

#### 10.4 Comparison with Retrieval-Augmented DPO

RE-ALIGN [75] augments standard DPO by injecting retrieved visual evidence, specifically relevant image patches and captions retrieved via CLIP, into the preference optimization pipeline. This approach enhances the factual grounding of preference pairs by providing the model with additional visual context during training. In contrast, TARS improves robustness through an entirely orthogonal mechanism: adversarial token-level perturbation within a min–max framework, without modifying the input data or retrieval pipeline.

Experimental setup. We compare four configurations on LLaVA-v1.5-7B using the same 4.8k preference samples: base LLaVA, standard DPO, DPO + RE-ALIGN, and TARS + RE-ALIGN. For RE-ALIGN, we follow the original implementation using CLIP-ViT-L/14 for retrieval and inject the top-3 retrieved evidence snippets.

- Analysis. As shown in Tab. 13, RE-ALIGN alone provides strong improvements over standard DPO, reducing Hal-Rate from 26.4% to 16.8% (–9.6 pp) and improving object coverage from 56.6% to 58.9% (+2.3 pp). However, combining TARS with RE-ALIGN yields substantial further gains across all six metrics: Hal-Rate drops to 12.5% (–4.3 pp below RE-ALIGN alone), CHAIR decreases from 3.6 to 2.2, and MMH improves from 2.31 to 2.57.

The complementarity of the two methods can be understood through their distinct mechanisms: RE-ALIGN enriches the input space by providing additional visual evidence, reducing hallucinations caused by insufficient grounding information. TARS, on the other hand, strengthens the optimization process by exposing the model to adversarial token perturbations during training, reducing hallucinations caused by fragile token-level associations. Together, they address both data-side and algorithm-side sources of hallucination, achieving the best performance across all metrics.

- Table 13: Complementarity of TARS and RE-ALIGN. Combining both methods achieves the best performance.

Method CHAIR↓ Cover↑ Hal-Rate↓ Cog↓ MMH↑ MMHal↓

LLaVA-v1.5-7B 7.6 51.7 35.4 4.2 2.02 0.61 + DPO 4.9 56.6 26.4 2.5 2.19 0.61 + RE-ALIGN 3.6 58.9 16.8 1.3 2.31 0.52 + TARS & RE-ALIGN 2.2 60.4 12.5 0.4 2.57 0.43

- Table 14: Comparison of hallucination benchmarks on Muffin-13B. Bold denotes the best, and underlined denotes the second-best.

AMBER MMHal POPE OBJHal CHAIR↓ Cover↑ Hal↓ Cog↓ Scr↑ Hal↓ Acc↑ Pre↑ CRs↓ CRi↓

Algorithm

Muffin-13B [78] 7.5 45.7 34.6 3.4 2.27 0.58 83.0 80.7 47.3 15.2 + RLHF [67] 7.1 45.2 37.1 3.5 2.12 0.64 84.0 79.8 45.5 12.7 + DPO [40] 6.0 46.4 29.6 2.8 2.45 0.55 83.7 81.2 43.8 13.9 + CHiP-DPO [26] 4.8 48.2 18.9 1.7 2.70 0.47 84.5 82.1 35.2 11.5 + TARS (Mask) 3.6 49.5 16.2 1.4 2.75 0.41 87.4 84.9 28.7 8.2 + TARS (Replace) 4.0 48.9 16.5 1.5 2.76 0.43 86.5 83.8 29.3 8.8

### 11 Additional Results on Muffin

We further validate TARS on the Muffin-13B architecture (Tab. 14). Consistent with our findings on LLaVA, both perturbation strategies yield substantial improvements over DPO and CHiP-DPO. TARS with token masking achieves the strongest overall performance, while synonym replacement remains competitive. These results confirm the versatility of our approach across different MLLM backbones.

### 12 Discussions and Insights

#### 12.1 Why TARS Outperforms DPO: Beyond Numbers

While TARS consistently outperforms standard DPO across hallucination benchmarks, its effectiveness stems not only from empirical gains, but from the design principles that enable better preference alignment under uncertainty.

Token-level perturbation enhances alignment robustness. DPO relies on static textual inputs, making it susceptible to overfitting on superficial linguistic patterns. TARS addresses this issue by introducing controlled perturbations on visually agnostic tokens. These perturbations simulate semantically equivalent variations, exposing the model to distributional shifts during training. As a result, the learned policy becomes more robust to alignment uncertainty.

Visual-agnostic targeting preserves grounding fidelity. Unlike random or uniform perturbation strategies, TARS selectively perturbs tokens with low cross-modal relevance. This design ensures that semantic shifts are injected without disrupting the causal connection between image and text, resulting in faith-

- Table 15: Comparison of spectrum-based alignment (TARS) versus token-level contrastive alignment. Both models use identical perturbation policies.

AMBER MMHal POPE OBJHal

Strategy

CH↓ Cov↑ Hal↓ Cog↓ Scr↑ Hal↓ Acc↑ Pre↑ CRs↓ CRi↓ Token-level 3.4 57.2 16.3 1.4 2.36 0.49 87.1 93.3 15.8 4.9

Spectrum (TARS) 2.4 59.6 13.2 0.4 2.48 0.45 88.7 97.5 12.0 3.2

ful responses that remain sensitive to visual semantics while being resilient to linguistic noise.

Spectral alignment encourages semantic consistency. TARS introduces a spectral regularizer that aligns representations in the frequency domain. This global constraint allows for flexible modifications while maintaining semantic coherence at the sequence level, discouraging the model from latching onto spurious token-level correlations.

#### 12.2 Limitations

In this work, we adopt two simple perturbation strategies: token masking and synonym replacement. These methods are chosen for their clarity, efficiency, and ease of interpretation. However, their simplicity may limit the generality and flexibility of the approach. Future work could explore adaptive or data-driven perturbation mechanisms [3] that better balance semantic preservation with distributional shift. Additionally, the current token selection strategy, based on cross-modal similarity heuristics, could be enhanced by learning-based relevance estimation or causal attribution techniques.

### 13 Qualitative Examples

We provide qualitative comparisons between standard DPO and our proposed TARS in Fig. 8, across diverse image-question pairs. TARS consistently demonstrates improved grounding and hallucination suppression, outperforming traditional DPO in several key aspects:

Reduced hallucination via improved visual grounding. Compared to DPO, TARS produces responses that more accurately reflect the image content. In all cases, DPO introduces visual details not present in the input, while TARS remains faithful to the scene.

No degradation in response completeness. TARS maintains response richness without sacrificing length or informativeness. Better fine-grained grounding. TARS exhibits improved alignment at the fine-grained level, such as object color and count. Enhanced descriptive richness. TARS responses incorporate more relevant visual details, indicating stronger cross-modal alignment.

[Figure 32]

The image depicts a city street on a rainy day, with a wet sidewalk reflecting the surroundings. A bus is also visible on the street, adding to the urban atmosphere. \n\n There are multiple traffic lights in the scene, with one near the center of the image and others further down the street. A fire hydrant can be seen on the right side of the street, and a parking meter is located near the middle of the scene.\n\n A few people are walking along the sidewalk, likely trying to

DPO

stay dry under the rain. The overall scene captures the essence of a typical rainy day in a city.

The image depicts a wet city street on a rainy day, with a line of trees on the left side of the street. The street appears to be wet, and the reflection of a street light can be seen on the wet surface. The street is lined with trees, and the sidewalk is also wet.\n\n There is a street sign on the right side of the image, and a building can be seen in the background. The street is not busy, with only a few cars visible in the scene. The street is also lined with trees, and the street

TARS

[Figure 33]

Describe the image in detail.

appears to be wet.

###### (a)

[Figure 34]

[Figure 35]

What color is the umbrella?

The umbrella is colorful, containing blue, grey, red, and yellow.

DPO

The umbrella is colorful, containing purple, yellow, green, and orange.

TARS

(b)

[Figure 36]

The image features a man and a woman standing on a pier, overlooking the ocean. They are smiling at each other while enjoying the soft ocean breeze, with the sun setting behind a few scattered clouds. The couple is holding hands, possibly sharing a romantic moment or quietly reflecting together. The pier is made of weathered wood and lined with decorative railings, offering a scenic view of the calm water and sky. The couple is dressed in light summer

DPO

clothing and stands near a small bench, fully immersed in the peaceful.

The image features a man and a woman standing on a railing, overlooking the ocean. The sky

TARS

is purple, and the sun is setting in the background, creating a beautiful scene. The man is on

the right side of the image, and the woman is on the left side, both looking out at the ocean. The railing they are standing on is located on the right side of the image, and the ocean is visible in the foreground, with the sunset in the background.

[Figure 37]

Describe the image.

(c)

[Figure 38]

The image features a man walking near a red mailbox on the sidewalk. He appears relaxed,

wearing a dark shirt. The stop sign is positioned on the left side of the street, and the man is

DPO

walking towards it.

The image features a man walking on the side of a road, with a stop sign in the foreground. The stop sign is positioned on the left side of the image, and the man is walking towards the

TARS

right side of the image. The man is wearing a striped shirt and khaki pants, and his arm is

extended as he walks. The main background of the image is a blue wall The man's shadow is visible on the ground.

[Figure 39]

Describe the image.

(d)

[Figure 40]

The image features a blue and green train traveling down the tracks, passing under a bridge. The train is moving along a curved track, suggesting a rural or remote setting. A few people are visible near the tracks, including one individual who appears to be waving at the train. In the distance, a small group seems to be gathered near a platform, although no clear station structure is visible.

DPO

The image features a yellow and red train traveling on a set of train tracks. The train is passing under a stone bridge, which is located in the background. The train is moving along the tracks,

TARS

and the bridge is visible above the train. The train is the main focus of the image, and the

[Figure 41]

bridge is the only other object in the background. A person is sitting on top of the bridge.

Describe the image.

(e)

Fig. 8: Qualitative comparisons between DPO and TARS across five diverse imageprompt pairs, denoted as (a)–(e). Hallucinated content is highlighted in red, while accurate visual grounding is marked in green. TARS consistently produces more faithful and informative responses.

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Achiam, J., Held, D., Tamar, A., Abbeel, P.: Constrained policy optimization. In: ICML (2017)
- 3. Bai, H., Kong, L., Chen, X., Wang, J., Tao, Z., Wang, H.: Dice: Diffusion large language models excel at generating cuda kernels. arXiv preprint arXiv:2602.11715

(2026)

- 4. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923

(2025)

- 5. Bai, Z., Wang, P., Xiao, T., He, T., Han, Z., Zhang, Z., Shou, M.Z.: Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930

(2024)

- 6. Bao, H., Wang, W., Dong, L., Liu, Q., Mohammed, O.K., Aggarwal, K., Som, S., Piao, S., Wei, F.: Vlmo: Unified vision-language pre-training with mixture-ofmodality-experts. In: NeurIPS (2022)
- 7. Bellagente, M., Brack, M., Teufel, H., Friedrich, F., Deiseroth, B., Eichenberg, C., Dai, A.M., Baldock, R., Nanda, S., Oostermeijer, K., et al.: Multifusion: Fusing pre-trained models for multi-lingual, multi-modal image generation. In: NeurIPS

(2023)

- 8. Bradley, R.A., Terry, M.E.: Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika 39(3/4), 324–345 (1952)
- 9. Casper, S., Davies, X., Shi, C., Gilbert, T.K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al.: Open problems and fundamental limitations of reinforcement learning from human feedback. TMLR (2023)
- 10. Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.S., Seo, M.: How do large language models acquire factual knowledge during pretraining? In: NeurIPS (2024)
- 11. Chen, C., Zhu, J., Luo, X., Shen, H., Song, J., Gao, L.: Coin: A benchmark of continual instruction tuning for multimodel large language models. In: NeurIPS

(2024)

- 12. Chen, C., Liu, M., Jing, C., Zhou, Y., Rao, F., Chen, H., Zhang, B., Shen, C.: Perturbollava: Reducing multimodal hallucinations with perturbative visual training. In: ICLR (2025)
- 13. Chen, D., Chen, R., Zhang, S., Wang, Y., Liu, Y., Zhou, H., Zhang, Q., Wan, Y., Zhou, P., Sun, L.: Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In: ICML (2024)
- 14. Chen, L., Li, B., Shen, S., Yang, J., Li, C., Keutzer, K., Darrell, T., Liu, Z.: Large language models are visual reasoning coordinators. In: NeurIPS (2023)
- 15. Chen, X., Wang, C., Xue, Y., Zhang, N., Yang, X., Li, Q., Shen, Y., Liang, L., Gu, J., Chen, H.: Unified hallucination detection for multimodal large language models. In: ACL (2024)
- 16. Chen, X., Ma, Z., Zhang, X., Xu, S., Qian, S., Yang, J., Fouhey, D., Chai, J.: Multi-object hallucination in vision language models. In: NeurIPS (2024)
- 17. Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271

(2024)

- 18. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., et al.: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https://vicuna.lmsys.org (2023)
- 19. Chowdhury, S.R., Kini, A., Natarajan, N.: Provably robust dpo: Aligning language models with noisy feedback. In: ICML (2024)
- 20. Cooley, J.W., Tukey, J.W.: An algorithm for the machine calculation of complex fourier series. Mathematics of computation 19(90), 297–301 (1965)
- 21. Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500 (2023)
- 22. Dou, Z.Y., Xu, Y., Gan, Z., Wang, J., Wang, S., Wang, L., Zhu, C., Zhang, P., Yuan, L., Peng, N., et al.: An empirical study of training end-to-end vision-andlanguage transformers. In: CVPR (2022)
- 23. Feng, S., Fang, G., Ma, X., Wang, X.: Efficient reasoning models: A survey. arXiv preprint arXiv:2504.10903 (2025)
- 24. Feng, S., Tuo, K., Wang, S., Kong, L., Zhu, J., Wang, H.: Rewardmap: Tackling sparse rewards in fine-grained visual reasoning via multi-stage reinforcement learning. arXiv preprint arXiv:2510.02240 (2025)
- 25. Feng, S., Wang, S., Ouyang, S., Kong, L., Song, Z., Zhu, J., Wang, H., Wang, X.: Can mllms guide me home? a benchmark study on fine-grained visual reasoning from transit maps. arXiv preprint arXiv:2505.18675 (2025)
- 26. Fu, J., Huangfu, S., Fei, H., Shen, X., Hooi, B., Qiu, X., Ng, S.K.: Chip: Crossmodal hierarchical direct preference optimization for multimodal llms. In: ICLR

(2025)

- 27. Gan, Z., Chen, Y.C., Li, L., Zhu, C., Cheng, Y., Liu, J.: Large-scale adversarial training for vision-and-language representation learning. In: NeurIPS (2020)
- 28. Gandhi, K., Fränken, J.P., Gerstenberg, T., Goodman, N.: Understanding social reasoning in language models with language models. In: NeurIPS (2023)
- 29. Gunjal, A., Yin, J., Bas, E.: Detecting and preventing hallucinations in large vision language models. In: AAAI (2024)
- 30. He, B., Li, H., Jang, Y.K., Jia, M., Cao, X., Shah, A., Shrivastava, A., Lim, S.N.: Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In: CVPR (2024)
- 31. Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., Lin, D., Zhang, W., Yu, N.: Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In: CVPR (2024)
- 32. Huang, S., Dong, L., Wang, W., Hao, Y., Singhal, S., Ma, S., Lv, T., Cui, L., Mohammed, O.K., Patra, B., et al.: Language is not all you need: Aligning perception with language models. In: NeurIPS (2023)
- 33. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)
- 34. Jain, J., Yang, J., Shi, H.: Vcoder: Versatile vision encoders for multimodal large language models. In: CVPR (2024)
- 35. Ji, Y., Wang, J., Gong, Y., Zhang, L., Zhu, Y., Wang, H., Zhang, J., Sakai, T., Yang, Y.: Map: Multimodal uncertainty-aware vision-language pre-training model. In: CVPR (2023)
- 36. Jiang, C., Xu, H., Dong, M., Chen, J., Ye, W., Yan, M., Ye, Q., Zhang, J., Huang, F., Zhang, S.: Hallucination augmented contrastive learning for multimodal large language model. In: CVPR (2024)

- 37. Jiang, Y., Sun, K., Sourati, Z., Ahrabian, K., Ma, K., Ilievski, F., Pujara, J., et al.: Marvel: Multidimensional abstraction and reasoning through visual evaluation and learning. In: NeurIPS (2024)
- 38. Kim, J., Kim, H., Yeonju, K., Ro, Y.M.: Code: Contrasting self-generated description to combat hallucination in large multi-modal models. In: NeurIPS (2024)
- 39. Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., Shan, Y.: SEED-Bench: Benchmarking multimodal LLMs with generative comprehension. In: CVPR (2024)
- 40. Li, S., Lin, R., Pei, S.: Multi-modal preference alignment remedies degradation of visual instruction tuning on language models. In: ACL (2024)
- 41. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: EMNLP (2023)
- 42. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV (2014)
- 43. Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., Wang, L.: Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565 (2023)
- 44. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: CVPR (2024)
- 45. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: NeurIPS (2023)
- 46. Liu, S., Zheng, K., Chen, W.: Paying more attention to image: A training-free method for alleviating hallucination in lvlms. In: ECCV (2024)
- 47. Liu, W., Wang, X., Wu, M., Li, T., Lv, C., Ling, Z., JianHao, Z., Zhang, C., Zheng, X., Huang, X.J.: Aligning large language models with human preferences through representation engineering. In: ACL (2024)
- 48. Ma, F., Xue, H., Zhou, Y., Wang, G., Rao, F., Yan, S., Zhang, Y., Wu, S., Shou, M.Z., Sun, X.: Visual perception by large language model’s weights. In: NeurIPS

(2024)

- 49. McKinzie, B., Gan, Z., Fauconnier, J.P., Dodge, S., Zhang, B., Dufter, P., Shah, D., Du, X., Peng, F., Belyi, A., et al.: Mm1: methods, analysis and insights from multimodal llm pre-training. In: ECCV (2024)
- 50. Oh, J., Kim, S., Seo, J., Wang, J., Xu, R., Xie, X., Whang, S.: Erbench: An entity-relationship based automatically verifiable hallucination benchmark for large language models. In: NeurIPS (2024)
- 51. Ouali, Y., Bulat, A., Martinez, B., Tzimiropoulos, G.: Clip-dpo: Vision-language models as a source of preference for fixing hallucinations in lvlms. In: ECCV (2024)
- 52. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. In: NeurIPS (2022)
- 53. Parekh, J., Khayatan, P., Shukor, M., Newson, A., Cord, M.: A concept-based explainability framework for large multimodal models. In: NeurIPS (2024)
- 54. Pi, R., Han, T., Xiong, W., Zhang, J., Liu, R., Pan, R., Zhang, T.: Strengthening multimodal large language model with bootstrapped preference optimization. In: ECCV (2024)
- 55. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)
- 56. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. In: NeurIPS (2023)
- 57. Rohrbach, A., Hendricks, L.A., Burns, K., Darrell, T., Saenko, K.: Object hallucination in image captioning. In: EMNLP (2018)

- 58. Ruan, H., Lin, J., Lai, Y., Luo, Z., Li, S.: Hccm: Hierarchical cross-granularity contrastive and matching learning for natural language-guided drones. In: ACM MM (2025)
- 59. Sarkar, P., Ebrahimi, S., Etemad, A., Beirami, A., Arık, S.Ö., Pfister, T.: Dataaugmented phrase-level alignment for mitigating object hallucination. In: ICLR

(2025)

- 60. Sarkar, P., Ebrahimi, S., Etemad, A., Beirami, A., Arik, S.O., Pfister, T.: Mitigating object hallucination in mllms via data-augmented phrase-level alignment. In: ICLR (2025)
- 61. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)
- 62. Setlur, A., Garg, S., Geng, X., Garg, N., Smith, V., Kumar, A.: Rl on incorrect synthetic data scales the efficiency of llm math reasoning by eight-fold. In: NeurIPS

- (2024)

63. Shao, K., Tao, K., Qin, C., You, H., Sui, Y., Wang, H.: Holitom: Holistic token merging for fast video large language models. arXiv preprint arXiv:2505.21334

- (2025)

- 64. Sharma, A., Keh, S.S., Mitchell, E., Finn, C., Arora, K., Kollar, T.: A critical evaluation of ai feedback for aligning large language models. In: NeurIPS (2024)
- 65. Song, Y., Swamy, G., Singh, A., Bagnell, J., Sun, W.: The importance of online data: Understanding preference fine-tuning via coverage. In: NeurIPS (2024)
- 66. Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L.Y., Wang, Y.X., Yang, Y., et al.: Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525 (2023)
- 67. Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L., Wang, Y.X., Yang, Y., et al.: Aligning large multimodal models with factually augmented rlhf. In: Findings: ACL (2024)
- 68. Tian, X., Zou, S., Yang, Z., He, M., Zhang, J.: Black sheep in the herd: Playing with spuriously correlated attributes for vision-language recognition. In: ICLR (2025)
- 69. Tong, P., Brown, E., Wu, P., Woo, S., IYER, A.J.V., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In: NeurIPS (2024)
- 70. Wang, F., Zhou, W., Huang, J.Y., Xu, N., Zhang, S., Poon, H., Chen, M.: mdpo: Conditional preference optimization for multimodal large language models. In: EMNLP (2024)
- 71. Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., Wang, J., Xu, H., Yan, M., Zhang, J., et al.: Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397 (2023)
- 72. Wang, S., Lin, J., Guo, X., Shun, J., Li, J., Zhu, Y.: Reasoning of large language models over knowledge graphs with super-relations. In: ICLR (2025)
- 73. Wu, Z., Chen, X., Pan, Z., Liu, X., Liu, W., Dai, D., Gao, H., Ma, Y., Wu, C., Wang, B., et al.: Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302 (2024)
- 74. Xie, Y., Li, G., Xu, X., Kan, M.Y.: V-dpo: Mitigating hallucination in large vision language models via vision-guided direct preference optimization. In: Findings: EMNLP (2024)
- 75. Xing, S., Li, P., Wang, Y., Bai, R., Wang, Y., Hu, C.W., Qian, C., Yao, H., Tu, Z.: Re-Align: Aligning vision language models via retrieval-augmented direct preference optimization. In: EMNLP (2025)
- 76. Yang, X., Zhang, H., Qi, G., Cai, J.: Causal attention for vision-language tasks. In: CVPR (2021)

- 77. Yang, Z., Luo, X., Han, D., Xu, Y., Li, D.: Mitigating hallucinations in large vision-language models via dpo: On-policy data hold the key. In: CVPR (2025)
- 78. Yu, T., Hu, J., Yao, Y., Zhang, H., Zhao, Y., Wang, C., Wang, S., Pan, Y., Xue, J., Li, D., et al.: Reformulating vision-language foundation models and datasets towards universal multimodal assistants. arXiv preprint arXiv:2310.00653 (2023)
- 79. Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H.T., Sun, M., et al.: Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In: CVPR (2024)
- 80. Yu, T., Zhang, H., Yao, Y., Dang, Y., Chen, D., Lu, X., Cui, G., He, T., Liu, Z., Chua, T.S., et al.: Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. In: CVPR (2025)
- 81. Zhang, J., Huang, J., Jin, S., Lu, S.: Vision-language models for vision tasks: A survey. IEEE TPAMI 46(8), 5625–5644 (2024)
- 82. Zhang, K., Weng, J., Li, S., Luo, Z.: Towards adversarial robustness via debiased high-confidence logit alignment. In: ICCV (2025)
- 83. Zhou, Y., Xu, P., Liu, X., An, B., Ai, W., Huang, F.: Explore spurious correlations at the concept level in language models for text classification. In: ACL (2024)

