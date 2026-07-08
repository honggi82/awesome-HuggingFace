# arXiv:2511.19413v3[cs.LG]30Mar2026

## UniGame: Turning a Unified Multimodal Model Into Its Own Adversary

Zhaolong Su1 Wang Lu2 Hao Chen3 Sharon Li4 Jindong Wang1* 1William & Mary 2Independent 3Carnegie Mellon University 4University of Wisconsin–Madison

{zsu05,jdw}@wm.edu, newlw230630@gmail.com, haoc3@andrew.cmu.edu, sharonli@cs.wisc.edu

#### Abstract

Unified Multimodal Models (UMMs) have shown impressive performance in both understanding and generation with a single architecture. However, UMMs still exhibit a fundamental inconsistency: understanding favors compact embeddings, whereas generation favors reconstruction-rich representations. This structural trade-off produces misaligned decision boundaries, degraded cross-modal coherence, and heightened vulnerability under distributional and adversarial shifts. In this paper, we present UniGame, a self-adversarial post-training framework that directly targets the inconsistencies. By applying a lightweight perturber at the shared token interface, UniGame enables the generation branch to actively seek and challenge fragile understanding, turning the model itself into its own adversary. Experiments demonstrate that UniGame significantly improves the consistency (+4.6%). Moreover, it also achieves substantial improvements in understanding (+3.6%), generation (+0.02) on GenEval, out-of-distribution and adversarial robustness (+4.8% and +6.2% on NaturalBench and AdVQA). The framework is architecture-agnostic, introduces < 1% additional parameters, and is complementary to existing post-training methods. These results position adversarial self-play as a general and effective principle for enhancing the coherence, stability, and unified competence of future multimodal models. Code is available at https: //github.com/AIFrontierLab/TorchUMM.

#### 1. Introduction

Unified Multimodal Models (UMMs) have recently demonstrated impressive capability in both visual understanding and image generation with a unified architecture [4, 28, 38, 40–42]. By jointly leveraging a language model backbone and a visual tokenizer–decoder stack [2, 18], these models promise a unified interface for cross-modal reasoning, grounded perception, and controllable generation. Specifically, the large-scale pre-training establishes general multi-

*Corresponding author.

72

BAGEL (14B) UniWorld-V1 (12B)

70

Base+UniGame (7B)

68

###### Performance

66

BLIP-3o (8B)

64

OmniGen2 (7B)

Base+T2I-R1 (7B)

62

60

Base+SFT (7B)

Harmon+RecA (1.5B)

Janus-Pro(Base) (7B)

58

Model size

Harmon (1.5B)

56

54 56 58 60 62 64 66 68

Consistency Score

(a) Performance vs. consistency.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | || |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
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
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
| | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
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
<br><br>|| |
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
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
| | | | | | |
| | | | | | |

Dim2

Dim 1

SFT

Post Train UniGame

| |
|---|

(b) Manifold coverage.

Figure 1. Qualitative and quantitative analyses of UniGame.1 (a) The performance vs. consistency score of several models, indicating significant improvement of both metrics of our models. (b) The manifold produced by SFT, reconstruction-based Post Train, and UniGame. UniGame expands the training distribution toward hard yet realistic neighborhoods.

modal capabilities, and the post-training stage (supervised fine-tuning, SFT, Figure 2a) can further improve their performance on downstream tasks with enhanced reliability.

Despite their great performance, UMMs exhibit structural inconsistency between their understanding and generation pathways [37, 43]. This inconsistency stems from the inherently conflicting nature of the two objectives, leading to the mismatch in various aspects such as semantics (i.e., the model can answer a question correctly yet fail to generate a corresponding image, or vice versa [6, 31]), capability (e.g., generation is harder to improve than understanding, or vice versa [30]), and feature compactness (e.g., understanding requires more compact feature space while generation prefers oppositely). Inconsistency widely exists in real-world applications, where models frequently encounter unexpected inputs far from the training manifold, compositional combinations unseen during training, counterfactual queries, or modality conflicts such as distribution shift [6, 14, 24, 25] and adversarial attack [16]. If not suf-

1(a) Where consistency Score computed as the average of WISE and UnifiedBench, performance is averaged over understanding bench MMMU and generation bench like GenEval. (b) We randomly sample 100 images, extract their unified embeddings, project to 2D with UMAP [22]; colored regions visualize each method’s on-manifold coverage.

Reconstruction loss

Reward

Generated Image

Generated Image

Vision Decoder

Vision Decoder

Vision Encoder

Vision Encoder

Generated

Vision Decoder

Vision

Image

Image

Image

Image

Encoder

LLM

LLM

LLM

Generated

Generated

Text Text

Text Text

Text Encoder

Text Encoder

Generated Text

Text Text

Text Encoder

Text

Text

Detokenizer

Detokenizer

Detokenizer

[Figure 1]

[Figure 2]

[Figure 3]

Reconstruction loss

Reward

Reward

Reward

Minimax optimization

Generated Image

Generated Image

Generated Image

Generated Image

Generated Image

Generated Image

Vision Decoder

Vision Decoder

Vision Decoder

Vision Decoder

Vision Decoder

Vision Decoder

Adversarial Generation

Vision Encoder

Vision Encoder

Vision Encoder

Vision Encoder

Vision Encoder

Vision Encoder

Generated

Generated

Generated

Vision Decoder

Vision Encoder

Vision Decoder

Vision Decoder

Vision Decoder

Vision

Vision

Vision

Image

Image

Image

Image

Image

Image

Image

Image

Image

Image

Image

Image

Image

Encoder

Encoder

Encoder

Perturber

LLM

LLM

LLM

LLM

LLM

LLM

LLM

LLM

LLM

LLM

Generated

Generated

Generated

Generated

Generated

Generated

Text Text

Text Text

Text Text

Text Text

Text Text

Text Text

Text Encoder

Text Encoder

Text Encoder

Text Encoder

Text Encoder

Text Encoder

Generated Text

Text Text

Text Encoder

Generated Text

Generated Text

Generated Text

Text Text

Text Text

Text Text

Text Encoder

Text Encoder

Text Encoder

Text

Text

Text

Text

Text

Text

Detokenizer

Detokenizer

Detokenizer

Detokenizer

Detokenizer

Detokenizer

Detokenizer

Detokenizer

Detokenizer

Detokenizer

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

(d) Ours: Minmax optimization

(a) Supervised fine-tuning

(b) Reconstruction-base approaches

(c) Reward-based approaches

Figure 2. Illustration of four different post-training paradigms.

Minimax optimization

ficiently studied, it would greatly undermine multimodal information fusion, model robustness, and further performance improvement.

Compared to existing efforts, UniGame explicitly converts representational weaknesses into decoded, semantically coherent counterexamples that efficiently harden understanding (Figure 2). Empirically, UniGame uncovers richer, actionable failure modes (e.g., counting, fine attributes, and occlusion in Figure 5a) and improves the consistency, performance, and robustness.

Adversaria l

Adversaria l

Adversaria l

Vision Decoder

Vision Decoder

Vision Decoder

Vision Encoder

Vision Encoder

Vision Encoder

Image

Image

Image

Generation

Generation

Generation

Perturber

Perturber

Perturber

LLM

LLM

LLM

Generated Text

Generated Text

Generated Text

Text Text

Text Text

Text Text

Text Encoder

Text Encoder

Text Encoder

Detokenizer

Detokenizer

Detokenizer

It remains challenging and unexplored to improve the consistency of UMMs, primarily due to the unclear learning objectives: only the consequences, but not the causes, are known. Therefore, recent post-training approaches tend to fill this gap using surrogate objectives. Reconstructionbased approaches (Figure 2b) regenerate the original images through the semantic embedding space derived from visual perception [37, 43]. They optimize a unified reconstruction objective, which trains models within a closed autoencoding loop. Reward-based methods (Figure 2c) typically optimize an ensembled reward function [13], combining task-specific or rule-based metrics to refine the output relying on external expert models. However, both are optimized using handcrafted objectives (reconstruction or reward), which only polish model behavior on a fixed training distribution and place no explicit constraints on the two coupling branches. As a result, they reproduce behaviors within a fixed manifold rather than expanding the shared generative space, leaving the inconsistency largely unresolved.

[Figure 14]

[Figure 15]

[Figure 16]

This paper makes the following contributions:

- 1. Novel framework. We propose UniGame as the first framework to formalize UMMs post-training as a selfplay game to improve the consistency between the understanding and generative pathways.
- 2. Self-play training algorithm. We instantiate UniGame by devising a flexible co-training algorithm that combines the perturber, regularizer, and hardness-aware mining modules. The algorithm is agnostic to both UMM architectures and post-training approaches.
- 3. Empirical improvement. UniGame yields improvements in consistency (4.6%), understanding (+3.6%), generation (+0.02 on GenEval), OOD (+4.8%) and adversarial robustness (+6.2%).

#### 2. Related Work

Can a UMM expose and correct its own inconsistencies from within? In this paper, motivated by the observation that adversarial signals reliably surface brittle reasoning in vision–language models [6, 14, 24, 31], we propose UniGame (Figure 3), the first self-adversarial post-training framework for UMMs. UniGame treats the generation pathway as an active adversary that searches for visually plausible, decoder-constrained perturbations that maximally challenge the understanding branch. It pushes the model beyond fixed data manifolds and produces structured adversarial samples along the uncertainty regions (Figure 1a). Concretely, UniGame installs a lightweight perturber at the shared visual-token interface to create bounded, structured perturbations. These perturbations are decoded into realistic adversarial images, filtered through a semantic consistency check, and stored in a hard-example buffer. The understanding branch is then optimized to correctly reason over both clean inputs and these internally generated, semantically aligned counterexamples. This forms a minimax self-play process where generation seeks to expose weaknesses while understanding learns from them, effectively expanding the shared generative manifold toward fragile yet meaningful regions (Figure 1b; theoretical insights in Appendix G.3).

Unified Multimodal Models. UMMs aim to combine multimodal understanding and generation within a single backbone, enabling compact deployment and richer cross-modal reasoning [1, 3, 4, 28, 28, 36, 38]. Among these, BLIP3o [1] explores hybrid autoregressive and diffusion training recipes to balance understanding and generation fidelity. Emu3 [38] treats images and text as an interleaved token stream and scales next-token prediction to unified multimodal outputs. TokenFlow [28] focuses on the tokenizer layer and introduces dual-granularity codebooks to reconcile the conflicting demands of discriminative understanding vs reconstructive generation. Despite architectural innovation, UMMs continue to face the inconsistency issue: the representation granularity and objective tension that underlie understanding and generation still cause ambiguous shared embeddings and latent failure modes that remain insufficiently addressed.

Post-training of UMMs. Aiming to resolve the inconsistency issue, existing post-training methods can be categorized into these types: (i) reconstruction and semanticalignment losses to encourage fidelity [37, 43]; (ii)

### Our UniGame

Minimax optimization

Adversarial Embedding

Embedding

Embedding

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Vision Encoder

Vision

###### ℬ

𝐶

Decoder

Multimodal

Adversarial

Hard Buffer

Adversarial Image

Large Language

Perturber

Image

Candidates

Model

“Is the cat standing on the bed?”

|Generation Challenges Understanding Understanding Challenges Generation<br><br>|
|---|

Answer: “Yes”

Text

Text Encoder

Detokenizer

(LoRA)

Embedding

Token

Figure 3. Overview of UniGame. This adversarial self-play improves understanding robustness and understanding-generation consistency. The perturber C is a lightweight (3-layer MLP) module and the hard buffer B is a filtering mechanism.

RL/reward-based optimization to directly optimize downstream metrics [13, 32]. Each family improves either fidelity or robustness [2, 6, 15, 28, 31, 32, 37], For instance, RecA [37] leverages reconstruction alignment by conditioning generation on understanding embeddings and using reconstruction losses to bring representations closer, while T2I-R1 [13] enhances image generation through collaborative semantic- and token-level chain-of-thought combined with reinforcement learning. In the vision–language domain, AT has shown potential: VILLA introduces large-scale embedding-space perturbations across image and text modalities, improving robustness and generalization [6, 31]. Nevertheless, most methods either improve alignment or generation separately, and adversarial mechanisms are rarely incorporated into full UMMs. They commonly fail in one crucial respect: they do not exploit generation as an active adversarial process to strengthen the understanding branch. Specifically, adversarial or unguided embedding perturbations often produce offmanifold samples—unrealistic or semantically invalid artifacts, while reconstruction objectives do not intentionally surface decision-critical failure modes. Although RL-based schemes are effective, they are computationally costly and do not guarantee that these discovered examples remain decodable and semantically plausible [13, 32]. This leaves a key gap in truly improving the consistency of the understanding and generation pathways.

are then fed into a language model to learn high-level embeddings. The understanding branch U aims to minimize the discriminative loss of the textual output, and the generation branch G tends to reconstruct the input image:

Lund(θU) := −E log pU(a | z,q) , (1) Lgen(θG) := E ℓgen(G(z),x) , (2)

where pU(a | z,q) is the predictive distribution of the understanding task and ℓgen is the reconstruction loss (e.g. MSE). These two objectives are commonly optimized jointly in a single multi-task objective:

Ljoint := Lund(θU) + λLgen(θG), (3)

min

θU,θG

where λ is the trade-off hyperparameter.3

##### 3.2. Motivation

UMMs inherently exhibit inconsistencies between the understanding and generation pathways due to their conflicting optimization requirements: the understanding branch favors task-oriented embeddings, but generation demands reconstruction-rich representations. Both branches operate on a shared generative manifold induced by encoding interface and decoder; any mismatch in how they carve up this manifold directly translates into structural inconsistencies. Improving consistency is challenging primarily owing to the lack of clear, direct learning objectives: We can only observe the consequences (e.g., semantic mismatches, capability gaps) but struggle to identify their underlying causes.

#### 3. UniGame

##### 3.1. Preliminary

Existing efforts [13, 37] optimize for individual goals on a fixed set of data distributions, and place no explicit constraints on the two coupling training objectives. They encourage cooperative training, reproducing existing samples rather than expanding coverage of the shared manifold, where boundary behavior is most fragile, thus aggravating inconsistency. We argue that improving consistency requires expanding this shared manifold, especially around

Let x denote an image, q a vision-grounded query, and a the ground-truth answer. After a frozen visual encoder and a projection layer, we write the unified visual tokenizer, which quantizes image representations into tokens aligned with the language model’s vocabulary embedding space as z = Enc(x) ∈ RN×H, where N is the token length and H the hidden dimension.2 Both visual and textual embeddings

2In addition to encoder and projection layers, real encoder modules consists of other parts such as semantic encoder and tokenizer. We omit these details for simplicity.

3The term “jointly” here denotes simultaneous (multi-task) optimization of both branches—possibly sharing backbone parameters—rather than strictly sequential stage-wise training.

decision boundaries, instead of merely polishing the model within its comfort regions.

Considering the unified architecture of UMMs: can we improve the consistency within the model itself? We turn to adversarial training [21], which creates adversarial perturbations to explore understanding failures [6, 31]. This indicates that adversarial signals, if properly constrained, can serve as an effective mechanism for regularizing the decision boundaries of UMMs. Within the UMM architecture, we focus on converting generative priors into decodable adversarial cases that (i) remain semantically valid and (ii) reliably expose genuine reasoning failures in the understanding branch. This intuitively motivates the self-play training paradigm that makes best use of both of the understanding and generation branches as a minimax optimization framework. The generative pathway no longer passively follows alignment objectives: it is explicitly trained to produce realistic, on-manifold adversarial cases that challenge the understanding module, while the understanding branch is optimized to solve these internally generated challenges.

##### 3.3. Overview of UniGame

As shown in Figure 3, the proposed UniGame introduces two lightweight, plug-in modules to a general UMM:

- • Perturber C: A compact network with parameters θC (|θC| ≪ min(|θU|,|θG|)) that maps the post-LM fused visual states zˆ to a perturbed token: z˜ = C(zˆ;θC) = zˆ + δ, where ∥δ∥ ≤ εmax is the budget to cap the perturbation magnitude for stabilization, We control the budget through an ablation study and use the best-performing setting in all main experiments. Details of the perturber architecture are in Appendix B.3.
- • Hard-sample buffer B: A component that scores decoded candidates and stores hard, semantically plausible examples for replay via semantic-consistency check [30]:

B = G(z˜) H(z˜) ≥ τ , (4)

where H = CE pU(ˆa | Enc(G(z˜)),q;θU),a is the cross-entropy loss and τ is the threshold.

We refer the original generation branch as the clean path, then the perturber introduces a perturbed path. In UniGame, a frozen visual encoder maps an input image to visual tokens at the unified interface. In the clean path, it is forwarded to the understanding head for supervised training with the query q; In the perturbed path, the embeddings with perturbations generated by C are decoded by G into candidate images. The semantically consistent candidates are stored in the buffer B. During training, the understanding module learns from hard examples from B and clean samples, and the perturber C is optimized to generate challenging yet plausible cases. The vision encoder (SigLIP [46]) is frozen, and only LoRA [12] adapters on the LLM backbone and the Perturber C are trainable.

The overall training objective is a minimax game:

min

max

θU

θC

LU(θU) + λLC(θC;θU) , (5)

where λ > 0 controls the strength of the self-play signal between these two branches. We provide some theoretical insights to the convergence in Appendix G: Under bounded perturbation and decoder constraints, the perturber optimizes a lower bound of the worst-case understanding loss. This ensures the minimax dynamics remain stable and prevents off-manifold adversarial drift. We will elaborate on LU and LC in next section.

##### 3.4. The Self-Play Training Process

The training objective in Eq. (5) consists primarily of two adversarial and iterative steps: to enable the understanding and generation branches to challenge each other. The complete training procedure is presented in Appendix A and we introduce only these two challenging steps.

Understanding Challenges Generation (the solid arrows in Figure 3). This is the naive feedforward path that optimizes the understanding module to prevent the generation branch from confusing it, i.e., to “challenge” the generation branch. The clean path forwards the original visual tokens z directly to the understanding head U, producing the model’s nominal predictions and the standard supervised loss.

Here, the clean path preserves the original semantic information and simply computes the supervised loss; semantic plausibility and regularization are enforced on the generation side via the perturber’s norm and CLIP-based filtering from the hard-example mining buffer. Formally:

LU(θU) = Eclean CE(pU(ˆa | z,q;θU),a)

(6)

+ β EB CE(pU(ˆa | z,q;θU),a) ,

where the first term keeps the model accurate on clean data, the second term forces U to correctly answer on current adversarial examples and mined hard cases, and β > 0 is a trade-off hyperparameter.

Generation Challenges Understanding (the dashed arrow in Figure 3). In this process, the perturbed embedding z˜ will be rendered by the decoder G into image candidates: x˜ = G(z˜), which are then subject to semantic-consistency checks (e.g., CLIP [30] similarity) and re-encoding/scoring by the understanding module. This path intentionally produces on-manifold adversarial examples to challenge the understanding branch, and hard candidates are stored in the buffer B for replay. Formally, the perturber is updated to maximize the understanding loss:

LC(θC;θU) = EcleanCE pU(ˆa | Enc(G(C(zˆ;θC))),q;θU),a − λ∥δ∥2.

(7)

- 3.5. Discussion

UniGame vs. GANs. UniGame is different from conventional GANs [9] that train a generator to fool a discriminator [9, 29]. First, GAN needs an extra discriminator for the adversarial game, while ours can operate within a UMM to leverage its own understanding and generation branches. Second, GANs primarily focus on generation tasks while ours targets both generation and understanding, involving more complex training and optimization process.

UniGame vs. Adversarial Training (AT). UniGame is the first attempt of applying AT [10, 21] to UMMs, but has the following differences. First, AT is mainly used for enhancing adversarial robustness, while ours explores AT for consistency improvement. Second, UniGame differs fundamentally from prior AT by enforcing decoder-constrained, on-manifold image perturbations, enabling self-generated adversarial cases that remain semantically meaningful.

Extensibility. UniGame is agnostic to most UMM architectures and post-training approaches. First, since it is a general training framework that only introduces a lightweight trainable perturber, it is flexible to be integrated into most UMMs. Second, it does not conflict with existing methods, but can serve as their complement for further improvement on consistency and performance. We further explore the intergration of UniGame with emerging post-training method e.g., [13, 37], we train their post-trained model and demonstrate further improvements (see §4.6).

- 4. Experiments

- 4.1. Experimental Setup

Tasks and datasets. We evaluated UniGame on popular benchmarks. Specifically, VQAv2 [11], MMMU [45], POPE [17], and MMBench [20] are adopted for understanding evaluation and GenEval [8] is employed for generation evaluation. For the evaluation of consistency, we report WISE score [23] and UnifiedBench [43]. For OOD robustness, we adopt NaturalBench [14], a challenging benchmark comprising real-world images captured in natural, uncontrolled environments (e.g., low lighting, occlusion, unusual viewpoints) that test robust visual reasoning. For adversarial robustness, we adopt AdVQA [16], an adversarially constructed VQA dataset where questions are intentionally designed to mislead models through linguistic ambiguity and visual distractors.

Implementation details. We implemented UniGame on a popular Janus-Pro-7B [3] UMM for main experiments, and further validated with two toy models that simulate distinct UMMs designs, similar to UAE [43]. The perturber is implemented as a 3-layer MLP operating on the shared visualtoken space RN×H, adding only 2.1M parameters and outputting token-wise perturbations followed by normalization and clipping. We adopt the open-source VQAv2 training set

Table 1. Consistency4evaluation on UnifiedBench and WISE (↑). Models marked with† denote our base model.

Model Params UnifiedBench WISE Avg Consistency Score

BAGEL [4] 14B 83.48 0.41 41.95 66.49 UniWorld-V1 [19] 12B 78.99 0.35 39.67 61.39 BLIP-3o [1] 8B 76.56 0.39 38.48 61.54 OmniGen2 [41] 7B 83.31 0.30 41.81 61.99 Janus-Pro† [3] 7B 82.77 0.35 41.54 63.66 Harmon [40] 1.5B 65.41 0.41 32.90 55.65 Show-o [42] 1.3B 69.16 0.30 34.73 53.50

Janus-Pro+SFT [3] 7B 83.20 0.37 41.79 64.72 (+1.06)‡ Harmon+RecA [37] 1.5B 66.94 0.40 33.67 56.16 (+0.51)‡ Janus-Pro+UniGame 7B 85.20 0.43 42.82 68.32 (+4.66)‡

and CC3M [35]. More training details are in Appendix B.

Baselines. We compare UniGame to two categories of baselines: (1) Different UMMs: (i) Auto-regressive models [3, 7, 28, 36, 42], which unify understanding and generation through next-token prediction in a shared token space, with improvements in vision tokenizers; (ii) Diffusion-based models [1, 41], which leverage latent diffusion for generation while maintaining autoregressive understanding (e.g., BLIP-3o [1], OmniGen2 [41]); (iii) Hybrid architectures [4, 19], which employ specialized modules for different modalities (e.g., UniWorld-V1 [19], BAGEL [4]). (2) Posttraining methods: (i) Reconstruction-based alignment, which uses caption-then-reconstruct cycles to enhance understanding-generation consistency (e.g., RecA [37]); (ii) Reward-based approaches, which use reward function to refine their outputs(e.g, [13]) Unlike these methods, UniGame introduces decoder-constrained self-adversarial training, converting latent inconsistencies into visually coherent counterexamples to improve reasoning robustness while preserving generation fidelity.

##### 4.2. Consistency Evaluation

We evaluated consistency on two benchmarks: UnifiedBench [43] and WISE score [23]. UnifiedBench is a reconstruction-based benchmark tailored for UMMs, where the “caption–generate–compare” protocol measures how consistently information is preserved when images is converted into text and decoded back into images; the unified score is defined as the similarity between the ground truth and reconstructed images. WISE is a world-knowledgeinformed text-to-image benchmark with 1000 knowledgeintensive prompts that are scored by a multimodal judge along consistency with the prompt, realism, and aesthetic quality; its overall Score emphasizes how faithfully the generated image reflects the textual description, making it a natural testbed for text-to-image consistency.

We followed Protocol 1 from [43] to evaluate the consistency of UniGame. Specifically, we randomly sampled 100

4Consistency Score = 0.6 × UnifiedBench + 0.4 × (WISE × 100), jointly assessing self-consistency in understanding generated content (UnifiedBench) and prompt-image alignment (WISE), the 0.6/0.4 weighting reflects the evaluation data ratio. ‡ show improvement over base model.

Table 2. Results for understanding benchmarks (↑).

Model Params VQAv2test MMMU MMBench POPE Overall

TokenFlow-XL [28] 14B 77.6 43.2 76.8 87.8 71.3 BAGEL [4] 14B — 55.3 85.0 — UniWorld-V1 [19] 12B — 58.6 83.5 — BLIP3-o [1] 8B 83.1 50.6 83.5 — Emu3 [38] 8B 75.1 31.6 58.5 85.2 62.6 SEED-X [7] 7B 71.2 35.6 70.1 84.1 65.2 Chameleon [36] 7B 66.0 22.4 — — OminiGen2 [41] 7B — 53.1 79.1 — Liquid [39] 7B 63.5 — — 76.8 Janus-Pro† [3] 7B 78.2 41.0 79.2 87.4 71.4 Show-o [42] 1.3B 69.4 26.7 — 80.0 —

SFT 7B 79.5 41.2 79.5 87.6 71.9 RecA [37] 1.5B — 35.7 — 83.9 UAE [43] — — — — — Ours 7B 83.4 43.8 83.2 89.6 75.0

Table 3. Results of text-to-image generation on GenEval [8] (↑).

Model S. Obj. Two Obj. Counting Colors Position Color Attri. Overall

SEED-X [7] 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o [42] 0.95 0.52 0.49 0.82 0.11 0.28 0.53 D-DiT [18] 0.97 0.80 0.54 0.76 0.32 0.50 0.65 TokenFlow-XL [28] 0.95 0.60 0.41 0.81 0.16 0.24 0.55 Chameleon [36] — — — — — — 0.39 OminiGen2 [41] 0.99 0.86 0.64 0.85 0.31 0.55 0.70 Janus-Pro† [3] 0.99 0.89 0.59 0.90 0.79 0.66 0.80

SFT 0.99 0.90 0.60 0.91 0.80 0.65 0.81 UAE [43] — — — — — — 0.86 RecA [37] 1.00 0.98 0.71 0.93 0.76 0.77 0.86 Ours 0.99 0.91 0.62 0.93 0.80 0.68 0.82

images from LAION-5B [34] as a test set. For each image, the model first generated a caption (understanding), based on which the model then reconstructed the image (generation). Finally, we computed the similarity scores between the original and reconstructed images using four visionlanguage backbones (CLIP [30], SigLIP [46], DINO-v2 [27], and DreamSim [5]).

As shown in Table 1, UniGame substantially improves the unification score across all metrics compared to both the Janus-Pro baseline and SFT-only training. This improvement demonstrates that self-play training not only enhances understanding, but also strengthens the bidirectional consistency between the understanding and generation branches, forcing the model to maintain semantic coherence across modalities and reducing the understanding-generation gap inherent in dual unified architectures.

##### 4.3. Benchmark Results

UniGame improves both understanding and generation. Table 2 and 3 show the results on understanding and generation benchmarks, respectively. The results demonstrate significant improvements of UniGame over most competitors in both tasks. Specifically for understanding tasks, UniGame shows an average improvement of 3.1% over SFT, and 3.6% over the baseline model. UniGame also outperforms other larger models such as TokenFlow-XL, Emu3, and BLIP-3o, demonstrating that creating adversarial examples can serve as an effective augmentation approach. As for generation, UniGame outperforms most competitors such as TokenFlow-XL and OminiGen, achiev-

NaturalBench AdVQA

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | || |
|---|
| | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

30

Score

20

10

Base Model +SFT +Ours

(a) OOD and adv. robustness.

(b) Training Loss Comparison

Figure 4. (a) Robustness evaluation. (b) We observe that over 5K of training steps, the hard-sample loss persistently dominates that of Clean/Adversarial, suggesting UniGame continuously generates samples that are most challenging for the current model state.

Table 4. Ablation: UniGame vs embedding perturbation. All methods use the same perturbation budget (εmax=0.02), update schedule, and training steps (16k). For embedding-only baselines, we apply perturbations directly in the visual token space without decoding. We report VQAv2 accuracy (%).

Method Performance Baseline (SFT) 79.5 Embedding-only perturbation

Random noise in token space 78.5 Adversarial emb. 78.9 Adv. emb. + Cosine Similarity 79.6 Adv. emb. + Cosine + Buffer 80.2

###### Decoder-constrained perturbation (Ours)

Decoding only 81.5 Decoding + Cosine Similarity 82.2 Decoding + CLIP 82.7 Full (+ CLIP + Buffer) 83.4

ing stronger performance than the base model and SFT. Note that UniGame performs slightly worse than UAE and RecA (0.82 vs. 0.86), which is mainly due to UAE and RecA’s explicit post-training on generation tasks, while ours was primarily trained on understanding tasks.

UniGame improves model robustness. We evaluated the OOD and adversarial robustness on NaturalBench [14] and AdVQA [16], respectively. Following Li et al. [14], we report Group Accuracy (G-Acc), which awards one point only when a model correctly answers all four (image, question) pairs in a test sample. For AdVQA, we report standard accuracy as used in prior work. As shown in Figure 4a, UniGame exhibits significant improvement in OOD and adversarial benchmarks (4.8% and 6.2% gain, respectively), suggesting its strong performance in robustness, confirming UniGame effectively expand the decision boundaries, as in Figure 5a we explicitly probe four fine-grind visual reasoning cases, where base models fail but UniGame reasoned correctly. For space reasons, we present full experiments, additional ablations, and detailed OOD/adversarial robustness breakdowns in Appendix C and D.

##### 4.4. Ablation Study

Table 4 shows the comparison between the embeddingonly and decoder-constrained adversarial perturbations under matched settings. For embedding-only baselines, we

|[Figure 22]<br><br>[Figure 23]<br><br>(C1) Are two dogs present in the image?<br><br>Baseline: No (✘) No (✓) Ours: Yes (✓) No (✓)|[Figure 24]<br><br>[Figure 25]<br><br>(C2) Is the<br><br>monster truck jumping over vehicles?<br><br>Baseline: Yes (✘) Yes (✓) Ours: No (✓) Yes (✓)|[Figure 26]<br><br>[Figure 27]<br><br>(C3) What is the women doing in the<br><br>image?<br><br>Baseline: Standing (✘) Standing (✓) Ours: Sitting (✓) Sitting (✓)|[Figure 28]<br><br>[Figure 29]<br><br>(C4) Is<br><br>anyone holding an<br><br>instrument?<br><br>Baseline: No (✘) No (✓) Ours: Yes (✓) No (✓)|
|---|---|---|---|
|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>there is a street sign<br><br>that is on the corner of guadala jara<br><br>there is a pizza on a<br><br>plate with a slice<br><br>missing<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>there are two sheep wearing coats standing in a field<br><br>a dog looking out of a car<br><br>window in a rear view mirror<br><br>skateboarder in mid air doing a trick at a skate park<br><br>there is a cat sitting on a window sill looking out the window<br><br>there is a woman standing in a kitchen<br><br>preparing food<br><br>there are many<br><br>vegetables on display at a farmers market| | | |

(a) Case study for close-ended and open-ended understanding tasks.

###### Baseline UniGame Baseline UniGame

###### Baseline UniGame

Baseline UniGame

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

A blue-eyed Siamese cat sitting on a green velvet armchair.

Grand Canyon at sunrise, layered red rock formations, dramatic

Three broccoli in a glass bowl on the left side of a table, two carrots lying on the side. A red sticker with the number“5” attached to the bowl.

Four red cubes stacked in a 2×2

square on the left, and four blue spheres on the right. One small green cube is placed between the cubes and spheres.

lighting, epic landscape.

(b) Case study for generation tasks.

Figure 5. Qualitative case studies of UniGame understanding and generation.

tics; (ii) decoder constraints enforce on-manifold perturbations, yielding stronger adversarial training (+2.0%); (iii) using CLIP to maintain semantics further amplifies gains by ensuring adversarial samples remain semantically consistent with the query text. Together, these components establish a principled framework for self-play training in UMMs.

apply perturbations directly in the visual token space without decoding, using cosine similarity constraints to prevent excessive token drift. The strongest embedding baseline incorporating adversarial perturbations, token-space cosine constraints, and buffer replay achieves 80.2% accuracy on VQAv2, representing a modest +0.7% improvement over the SFT baseline. In contrast, our decoderconstrained approach forces perturbations to pass through the model’s native decoder, rendering adversarial tokens into realistic images before evaluation. Notably, even without CLIP filtering, decoding alone improves accuracy to 81.5% (+2.0% over SFT and +1.3% over embedding perturbation), demonstrating that on-manifold constraints are inherently superior to token-space constraints. When we apply cosine similarity constraints in the decoded image feature space, performance further increases to 82.2%. Replacing feature-level cosine with CLIP’s text-image semantic matching yields 82.7%, validating that semantic constraints outperform purely geometric ones.

##### 4.5. Case Study

We provide case studies on both understanding and generation tasks for qualitative analysis.

Understanding tasks. Figure 5a illustrates four challenging categories of visual reasoning tasks: object counting, object interaction, spatial relation and location, and crowd object detection. UniGame outperformed the baseline models in all scenarios. For instance, in C4 (crowd object detection), dense and overlapping objects in crowded scenes challenge both localization and recognition. The baseline produces vague or incorrect answers, whereas UniGame maintains accuracy by learning from decoded adversarial samples that emphasize occlusion and clutter. These improvements align with our quantitative gains, confirming that UniGame systematically addresses decision-critical reasoning failures rather than merely fitting to benchmark statistics. And we evaluate the open-ended captioning task in Figure 5a, the qualitative examples align with our quantitative gains on benchmarks, and suggest that UniGame helps the model move toward semantically richer and accurate descriptions. More analysis is in Appendix E.1.

We further ablate the perturber and hard-sample buffer capacity. For the perturber, a 3-layer MLP (83.4%) outperforms both the 2-layer variant (82.8%) and the deeper 4-layer variant (81.2%), indicating that moderate capacity best balances expressiveness. For the buffer, a size of 50 yields the best accuracy (83.4%), while smaller sizes of 30 (83.1%) and 10 (82.5%) provide insufficient diversity. Key insights: (i) the embedding-level perturbations can only leverage weak adversarial signals (+0.7%) because they operate in an abstract space disconnected from visual seman-

Generation tasks. Figure 5b compares generations from

- Table 5. UniGame can be plugged into an existing post-training pipeline with modest extra training to jointly improve understanding, generation, and unification. Starting from a RecA-trained model harmon 1.5B, we further train with 5K UniGame steps (∼ 10 GPU-h), yielding consistent improvements.

Method MMMU GenEval UnifiedBench

understanding generation ocnsistency

RecA 35.7 0.86 66.94 RecA + Ours 36.2 (+0.5) 0.86 (−−) 68.21 (+1.27)

the same prompts before and after post-training with UniGame. Overall speaking, UniGame helps UMMs to generate more faithful, accurate, and stylistic images. For instance, on the synthetic shapes example, the baseline model already produces plausible objects but often violates fine-grained layout constraints (e.g., incorrect left/right ordering or cube–sphere counts), whereas UniGame yields images that respect the specified 2 × 2 red cube stack, the correct number of blue spheres, and the spatial relations such as “on the left / on the right” and “between”. More explanations of other cases are in Appendix E.2. Together with the understanding cases, it suggests that UniGame enhances cross-modal consistency without sacrificing, and in some cases even improving the generation quality.

4.6. Extensibility and Efficiency

UniGame remains agnostic to UMM architectures and is computationally efficient compared to other post-training methods. In this section, we evaluate its generality and efficiency using the full set of VQAv2 on 2×H100 (80 GB) with mixed precision. Unless noted otherwise, we use image generation size 384 and a global batch size of 8.

Extensibility. We implement UniGame as a complement to RecA [37]. Table 5 shows consistent performance gains. method outperforms the RecA by 0.5 on MMMU for understanding and 1.27 on UnifiedBench for consistency, while remaining the same on GenEval. These results indicate that UniGame can serve as a lightweight, plug-and-play post-training module that can be integrated into existing pipelines, requiring only minimal additional computation.

In addition to RecA, we further constructed two architectures: (1) UMM-1 uses a Qwen2.5-VL [44] backbone with a SigLIP2 [46] understanding encoder and a Stable Diffusion-1.5 [33] image branch; (2) UMM-2 keeps the vision/generation stack unchanged and replaces the backbone with GPT-OSS [26]. Since GPT-OSS is designed for texts, we inserted a trainable 2-layer MLP that projects vision embeddings into the language space. We further applied UniGame to two backbones: BLIP-3o [1] and Chameleon [36]. All show consistent gains (Appendix C.1).

- Table 6 shows that UniGame is agnostic to model architectures and can improve the performance of different backbones. Moreover, the gains are achieved with fewer train-

Table 6. Extensibility analysis using two toy backbones.

Model Baseline +UniGame Trainable

- UMM-1 (Qwen2.5-VL) 60.4 66.4 (+6.0%) ∼ 1.43% (100.3M/7B)
- UMM-2 (GPT-OSS) 28.9 53.2 (+24.3%) ∼ 0.45% (133.9M/30B)

Table 7. Efficiency study. Trainable parameter ratios are estimated from the official repository.

Method Baseline +UniGame Trainable ReCA 34.7 35.7 (+1.0%) ∼ 91% (∼ 1.4B/1.5B) UAE — — ∼ 1% (0.1B/11B) UniGame 41.0 43.8 (+2.8%) ∼ 1% (100.3M/7B)

able parameters (e.g., ∼0.45% for UMM-2 and ∼1.43% for UMM-1), indicating its parameter-efficient generalization.

Efficiency. We further evaluate the efficiency of UniGame in comparison with RecA [37] and UAE [43] on MMMU. Table 7 shows that while achieving stronger performance, UniGame uses fewer trainable parameters, indicating its efficiency over existing post-training approaches.

##### 4.7. Convergence and Hyperparameter Analysis

Finally, we present a large-scale analysis on the convergence and hyperparameters (e.g., the hard buffer threshold τ, trade-off β, and perturbation budget δ; see Figure 12). To study the training dynamic, we also conduct extensive ablation study on the learning rates of two major minmax opponents in Figure 7. Further, we systematically study the minimax dynamics by visualizing the optimization trajectory of each run. The best configuration yields a wellbehaved minimax trajectory, where the two players alternate smoothly without divergence, see Appendix 10. We also probe the self-play dynamics between the two opponents and clearly observe the interaction: the two branches alternately dominate the training objective, exhibiting a stable tug-of-war behavior, and change of dominance, see Figure 11 and 13. More details in Appendix F demonstrate that UniGame offers a steady training process and stays relatively robust to different hyperparameter choices. Appendix G further presents some theoretical insights.

#### 5. Conclusion and Limitation

UniGame is the first self-adversarial post-training framework to improve the consistency of UMMs. It formulated a minimax optimization game of the understanding and generation branches, thus enabling the model to autonomously discover its own failures. UniGame consistently showed increased consistency, performance, and robustness, highlighting the great potential of optimizing UMMs within the models for further improvement.

Limitations. This work has following limitations. First, we primarily evaluate Janus-Pro-7B; broader model coverage may reveal additional insights. Second, we use a limited set of datasets, and future work should test UniGame on more diverse and challenging benchmarks.

#### Acknowledgments

This paper is partially supported by unrestricted gift from Google, William & Mary Faculty Research Award, and Modal Academic Compute Award. The authors acknowledge William & Mary Research Computing for providing computational resources and/or technical support that have contributed to the results reported within this paper. URL: https://www.wm.edu/it/rc.

#### References

- [1] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 2, 5, 6, 8
- [2] Xiaokang Chen. Janus: Decoupling visual encoding for unified multimodal understanding and generation, 2024. 1, 3
- [3] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 2, 5, 6

- [4] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 2, 5, 6
- [5] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344,

2023. 6

- [6] Zhe Gan. Large-scale adversarial training for visionand-language representation learning. arXiv preprint arXiv:2006.06195, 2020. 1, 2, 3, 4
- [7] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024. 5, 6
- [8] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 5, 6, 1
- [9] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. arXiv preprint arXiv:1406.2661, 2014. 5
- [10] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014. 5
- [11] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 5, 1

- [12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 4
- [13] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025. 2, 3, 5
- [14] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. Naturalbench: Evaluating vision-language models on natural adversarial samples. Advances in Neural Information Processing Systems, 37:17044–17068, 2024. 1, 2, 5, 6
- [15] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 3
- [16] Linjie Li, Jie Lei, Zhe Gan, and Jingjing Liu. Adversarial vqa: A new benchmark for evaluating the robustness of vqa models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2042–2051, 2021. 1, 5, 6
- [17] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 5, 1
- [18] Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2779–2790, 2025. 1, 6
- [19] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 5, 6
- [20] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 5, 1
- [21] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017. 4, 5
- [22] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018. 1
- [23] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 5, 1
- [24] Changdae Oh, Zhen Fang, Shawn Im, Xuefeng Du, and Yixuan Li. Understanding multimodal llms under distribution

- shifts: An information-theoretic approach. In International Conference on Machine Learning, 2025. 1, 2
- [25] Changdae Oh, Jiatong Li, Shawn Im, and Sharon Li. Visual instruction bottleneck tuning. In Advances in Neural Information Processing Systems, 2025. 1
- [26] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. 8
- [27] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6
- [28] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025. 1, 2, 3, 5, 6
- [29] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. 5
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1, 4, 6
- [31] Javad Rajabi. Token perturbation guidance for diffusion models, 2025. 1, 2, 3, 4
- [32] Shyam Sundhar Ramesh. Group robust preference optimization in reward-free rlhf. arXiv preprint arXiv:2405.20304,

2024. 3

- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 8
- [34] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 6
- [35] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL, 2018. 5, 1
- [36] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2, 5, 6, 8
- [37] XuDong Wang. Reconstruction alignment improves unified multimodal models, 2024. 1, 2, 3, 5, 6, 8
- [38] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 2, 6

- [39] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable and unified multi-modal generators. arXiv preprint arXiv:2412.04332, 2024. 6
- [40] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025. 1, 5
- [41] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025. 5, 6
- [42] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 1, 5, 6
- [43] Zhiyuan Yan, Kaiqing Lin, Zongjian Li, Junyan Ye, Hui Han, Zhendong Wang, Hao Liu, Bin Lin, Hao Li, Xue Xu, et al. Can understanding and generation truly benefit together–or just coexist? In NeurIPS, 2025. 1, 2, 5, 6, 8
- [44] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 8
- [45] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 5, 1
- [46] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 4, 6, 8

## UniGame: Turning a Unified Multimodal Model Into Its Own Adversary Supplementary Material

#### Appendix Contents

- A Algorithm Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- B Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- C Detailed Experimental Results . . . . . . . . . . . . . . . . . . . . 2
- D Robustness Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- E Details on Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- F Convergence and Hyperparameter Analysis . . . . . . . . .4
- G Theoretical Insights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .5

#### A. Algorithm Details

The complete training algorithm of UniGame is shown in Algorithm 1.

Algorithm 1 UniGame

- 1: Initialize θU (understanding) and θC (Perturber);
- 2: for each training step t = 1,2,... do
- 3: Sample minibatch {(xi,qi,ai)}Mi=1 ∼ D and encode zi = Proj(Enc(xi))
- 4: Challenge step (update C):
- 5: Compute perturbations δi = C(zi;θC) and perturbed tokens z˜i = zi + δi with ∥δi∥ ≤ εmax
- 6: Decode candidates x˜i = G(z˜i)
- 7: Compute LC(θC;θU) as in Eq. (7)
- 8: Update θC ← θC + ηC∇θCLC
- 9: if t mod m = 0 then
- 10: Compute scores Hj and keep candidates passing CLIP threshold τ and push hard examples into B via Eq. (4)
- 11: end if
- 12: Understand step (update U):
- 13: Construct mixed batch: clean samples (zi,qi,ai), and hard samples (zˆj,qˆj,aˆj) drawn from B
- 14: Compute LU(θU) on the mixed batch as in Eq. (6)
- 15: Update θU ← θU − ηU∇θULU
- 16: end for

#### B. Training Details

##### B.1. Training and Testing Data

Data volumes. Unless otherwise noted, we follow the official training/evaluation splits and report results on the standard benchmarks. Training uses VQAV2 train-split [11] is a large-scale visual question answering benchmark (hundreds of thousands of image–question pairs) collected from MS-COCO images with crowd-sourced free-form answers;

it emphasizes grounded visual reasoning under natural images. CC3M [35] (training only) is a large web-scale image–caption corpus (∼3M pairs in the full set); we use a filtered subset of 100k as text–image supervision for the generative branch.

Benchmarks. We briefly introduce the benchmarks:

- • VQAv2 test-dev [11]: the official VQAv2 test-dev split contains 104 000 questions; evaluation is via the online server.5
- • MMMU [45]: a college-level, multi-discipline benchmark with 11 500 questions in total (we report on the official test set).
- • POPE [17]: object-hallucination evaluation with a balanced, image-grounded design; the test split has 9000 QA pairs.
- • MMBench [20]: curated multiple-choice suite; dev 1164 and test 1784 questions (4:6 split of ∼3K).
- • GenEval [8]: object/layout/attribute–focused T2I evaluation with 553 prompts (reference-free automatic checks).
- • UnifiedBench [43]: unification score via caption→reconstruction; Protocol-1 uses 100 source images.
- • WISE [23]: knowledge-informed T2I evaluation with 1000 structured prompts across 25 subdomains.
- • NaturalBench [14]: vision-centric VQA with natural adversarial samples, ∼10000 human-verified image–question pairs (2500 groups under the 2-image×2question protocol), scored by G-Acc.
- • AdVQA [16]: human-in-the-loop adversarial VQA; total size reported as ∼46 807 examples (commonly used splits include ∼5123 val / ∼23 399 test).

##### B.2. Hyperparameter Details

Optimization details. UniGame is like the current UMMs post-training, is an end-to-end method and involves decoding images in each batch, to balance performance and cost. Our optimizations are as followed. We use AdamW optimizers with learning rates for Generation (gen_lr) and Understanding (und_lr). We conduct extensive ablation on the learning rate ratio between these two components (detailed in Appendix C and Table 8), ultimately finding that a ratio of approximately 250 achieves optimal performance (gen_lr= 5×10−3, und_lr= 2×10−5).

We implement mixed precision for training, given that Uni-Game only learned and uses small-norm perturbation, insufficient numerical precision can quantize away the perturbation’s gradients and wash out all the supervision.

5Counts from the official VQA site; see also recent reports confirming 104K for test-dev.

We vary the Generation and understanding update ratio in {1:1, 1:5, 1:10}. We performed a precision ablation comparing fp16-all, bf16-all, tf32-enabled, fp32-all, fp16(G)+fp32(loss), and bf16(G/D)+fp32(loss), and found that our final choice—computing the perturbation update, regularizer, and losses in float32 while running the remaining forward/backward in bfloat16—consistently achieved the best stability–efficiency trade-off and the highest robust accuracy. We force all computations that determine the perturbation and its supervision to float32. Gradient norms and per-role clipping are also applied in FP32, and optimizer states remain FP32 (AdamW default). All other forward/backward passes (vision tower, diffusion decoder, and LLM blocks) run under bfloat16 autocast for throughput. This preserves the perturbation signal while retaining the speed benefits of mixed precision.

[Figure 46]

[Figure 47]

Q: How many toilets are there? GT: One; Prediction: Zero.

Q: Is the dog big? GT: No; Prediction: Yes.

[Figure 48]

[Figure 49]

Q: What is the color of the train? GT: Green; Prediction: Black.

Q: The color of the motorcycle? GT: Black; Prediction: Grey.

- Figure 6. Cases are drawn from the hard-sample buffer and represent failure cases that successfully challenged the model.

0 2500 5000 7500 10000 12500 15000

Training Steps

78

80

82

84

Accuracy

SFT

R025

R040 R060

R160 R250

R400 R800

- Figure 7. Training dynamics analysis. VQA accuracy evolution across different adversarial ratios, with best achieving optimal performance at 83.4%.

##### B.3. Perturber

Network architecture of C. We implement the perturber C as a lightweight three-layer MLP that operates on each fused visual token after the language model. The first two layers have the same width as the UMM hidden size and apply non-linear transformations that refine the token representation and extract a direction in the shared visual-token space. The third layer acts as a direction head, mapping the hidden representation back to the token space and indicating along which semantic direction each token should be pushed to maximally challenge the understanding branch. In parallel, C maintains a single learnable scalar gate ε, shared across tokens and constrained within the perturbation budget [0,εmax], which controls the overall perturbation strength. In this way, one part of C is responsible for discovering semantically adversarial directions, while the scalar gate ε controls how strongly these directions are applied, keeping the module compact (with |θC| ≪ min(|θU|,|θG|)) yet able to generate small but semantically meaningful adversarial perturbations.

GenEval 0.54), and Chameleon (+2.2, VQAv2 40.2 with +1.7 gain, MMMU 24.0, GenEval 0.40). These results confirm that UniGame generalizes across different UMM architectures, including both auto-regressive and diffusion-based designs.

##### B.4. Hard Samples

UniGame added a hard sampler buffer to select only the challenging adversarial samples for training. Figure 6 shows some challenging examples in our experiments.

#### C. Detailed Experimental Results C.1. Additional Backbone Validation

To validate the generality of UniGame beyond Janus-Pro7B, we applied it to two additional UMM backbones: BLIP-3o [1] (diffusion-based) and Chameleon [36] (autoregressive), both trained on a 0.4 split of the VQAv2 training set under-matched settings. All backbones show positive consistency gains: BLIP-3o (+2.5, MMMU 51.2,

##### C.2. Learning Rate Ratio Ablation

To determine the optimal balance between the generation and understanding branches, we conduct an extensive sweep of learning rate ratios. Table 8 lists the complete set of configurations tested.

Table 8. Learning-rate configurations for the adversarial ratio sweep. Each row (ID Rxxx) specifies a pair of learning rates for the generation (gen lr) and understanding module (und lr); the last column reports their ratio Gen/Und. For example, R250 corresponds to gen lr= 5 × 10−3 and und lr= 2 × 10−5, i.e., a 250:1 ratio. These IDs (R025–R800) are used in Fig. 7(b) to plot validation performance as a function of the adversarial ratio.

ID gen lr und lr Gen/Und R025 1.6 × 10−3 6.3 × 10−5 ≈ 25.4 R040 2 × 10−3 5 × 10−5 40 R060 2.4 × 10−3 4.1 × 10−5 ≈ 58.5 R100 3.2 × 10−3 3.2 × 10−5 100 R160 4 × 10−3 2.5 × 10−5 160 R250 5 × 10−3 2 × 10−5 250 R400 6.3 × 10−3 1.6 × 10−5 ≈ 394 R600 7.7 × 10−3 1.3 × 10−5 ≈ 592 R800 8.9 × 10−3 1.1 × 10−5 ≈ 809

[Figure 50]

Figure 9. Perturbation Sweetspot

##### C.3. Motivation Experiments

###### Ablation on CLIP Hyperparameters (VQAv2)

- 79

- 80

- 81

- 82

- 83

- 84

[Figure 51]

- 79.8 80.5 81.2 81.8 81.5
- 80.3 81.7 82.3 82.8 82.4

0.00.10.20.5

VQAv2Accuracy(%)

(CLIPWeight)cp

Optimal

80.1 82.4 82.9 83.4 82.6

79.5 81.8 82.2 82.3 81.9

0.00 0.20 0.25 0.30 0.35

(CLIP Similarity Threshold)

Figure 8. heatmap Ablation study on CLIP constraint configurations. We report VQAv2 accuracy for different combinations of CLIP weight and CLIP similarity threshold

To find an Optimal noise level, we inject i.i.d. Gaussian noise into the projected visual tokens with σ ∈ {0,0.005,0.01,0.015,0.02,0.05,0.1}. We observe a sweet spot near σ ≈ 0.01 where VQAv2 soft accuracy slightly

increases (74.50→75.58) before degrading at larger noise, see in Figure 9. This indicates that small, structured embedding perturbations can beneficially modulate the shared representation.

#### D. Robustness Results

The details results on OOD and adversarial robustness are shown in Table 9, indicating that UniGame significantly improves the robustness of the models.

Table 9. Results for OOD and adversarial robustness.

Model NaturalBench AdVQA

Janus-Pro 73.8 34.2 +SFT 73.9 36.4 +Ours 78.6 40.4

#### E. Details on Case Study E.1. Case Study on Understanding Tasks We offer more interpretations to Figure 5.

- • Object counting (C1): The baseline model fails to accurately count objects in cluttered scenes, often confusing similar-looking items or missing partially visible objects. After UniGame training, the model correctly identifies the precise count, demonstrating improved fine-grained visual attention.
- • Object interaction (C2): Understanding relational semantics between objects (e.g., ”person holding umbrella” vs. ”umbrella next to person”) requires compositional reasoning. The baseline misinterprets spatial relationships, while UniGame correctly recognizes the interaction pattern.
- • Spatial relation and location (C3): Queries about relative positions (e.g., ”left of”, ”behind”) expose fragile spatial understanding in the baseline. UniGame’s adversarial training—which systematically perturbs spatial layouts during decoding—hardens the model against such failures.
- • Crowd object detection (C4): dense and overlapping objects in crowded scenes challenge both localization and recognition. The baseline produces vague or incorrect answers, whereas UniGame maintains accuracy by learning from decoded adversarial samples that emphasize occlusion and clutter.

These qualitative improvements align with our quantitative gains, confirming that UniGame systematically addresses decision-critical reasoning failures rather than merely fitting to benchmark statistics.

In addition, we also present detailed analysis to the openended understanding tasks:

[Figure 52]

• Open-ended understanding. As illustrated in Figure 5a, UniGame produces more fine-grained and visually grounded captions than the baseline. The model not only recognizes the overall scene (e.g., pizza, street sign, animals) but also reliably captures details such as a missing pizza slice, vegetables on display at a farmers market, a cat sitting on a windowsill looking out the window, or two sheep wearing coats standing in a field. These examples show that adversarial self-play improves open-ended descriptions by encouraging the model to focus on decisioncritical visual evidence rather than hallucinated or overly generic content.

- E.2. Case Study on Generation Tasks

We offer more detailed explanation of the text-to-image generations in Figure 5b.

- • On the synthetic shapes example, the baseline model already produces plausible objects but often violates finegrained layout constraints (e.g., incorrect left/right ordering or cube–sphere counts), whereas UniGame yields images that respect the specified 2 × 2 red cube stack, the correct number of blue spheres, and the spatial relations such as “on the left / on the right” and “between”.
- • In the “broccoli in a glass bowl” example, UniGame more faithfully binds multiple attributes—three pieces of broccoli, two carrots on the side, and a clearly visible red sticker with the number “5” attached to the bowl—demonstrating stronger compositional control.
- • For the Grand Canyon scene, the baseline sometimes collapses the layered rock formations into a flatter composition, while UniGame better preserves depth and lighting that match the prompt description.
- • Finally, for the “blue-eyed Siamese cat sitting on a green velvet armchair”, UniGame produces a sharper Siamese appearance and a more coherent green velvet texture, indicating that self-play training can improve both semantic alignment and visual fidelity.

- F. Convergence and Hyperparameter Analysis

- Figure 10. The best result of all of our runs, optimization path are projected to a two dimension axis.

[Figure 53]

- Figure 11. Self-play dynamics between the generation and the understanding . The two branches alternately dominate the training objective, exhibiting a stable tug-of-war behavior.

adapts and induce catch-up oscillations (see Figure 13 Figure 11). Conversely, when the generation overpowers U, decoded candidates drift off-manifold and hurt clean accuracy. Thus, balance progression speeds: (i) use a slightly larger learning rate for C than for U’s adapters, and (ii) prefer short alternations over long unilateral bursts. Full grids, curves, and ablations are shown in Section C. Perturbation budget. The budget constraint εmax controls the perturbation magnitude in the token space. The results in Appendix C show a sweetspot that inverted U-shaped performance curve Figure 9, setting εmax too small (e.g., 0.005) produces weak perturbations that fail to expose critical reasoning failures, yielding limited robustness gains (+1.7% on NaturalBench).

- F.1. Convergence

Convergence of the minimax training. The minmax setup raises the practical question: when does the game converge and what schedules keep it stable? In our setup, only the Perturber C and LoRA adapters on the understanding branch U are trainable; due to U’s larger capacity, it can dominate and degrade the generation module. We restore stability by giving C a higher learning rate and using short, interleaved updates. We conducted an extensive sweep of the Generation/Understanding update ratio in Table 8, shows gen lr = 5 × 10−3, und lr = 2 × 10−5, provides the best clean–robust trade-off; prolonged generation phases saturate the attack success rate (ASR) before U

[Figure 54]

Figure 12. Perturebation Budget

##### F.2. Hyperparameter Sensitivity Analysis

Unless otherwise noted, we fix the perturbation budget to δ = εmax = 0.02 in all main experiments, which we found to provide a good clean–robust trade-off after sweeping δ ∈ {0.005,0.01,0.015,0.02,0.05,0.10} Figure 12. For hard-example mining, we define the hardness score H as the cross-entropy loss of the understanding branch on decoded candidates plus a CLIP-based hinge term, and select hard samples using a quantile-based threshold: the buffer threshold τ is set to the 60-th percentile of H within each mining batch, while additionally enforcing a minimum text– image CLIP similarity of 0.6 to filter out semantically offmanifold generations. The trade-off coefficient β in Eq. (6), which weights the contribution of buffer samples relative to clean examples, is set to β = 0.5 by default so that roughly half of the understanding gradient comes from adversarial or hard instances; we observed that UniGame is numerically stable for a broad range of β ∈ [0.3,1.0]. The hardsample replay buffer stores up to 50 decoded images ranked by H. We deliberately keep the capacity moderate, as substantially larger buffers (e.g., ≫ 104 entries) would store many full-resolution decoded images and quickly lead to a steep increase in GPU and host memory usage, without providing noticeable additional benefits in practice.

#### G. Theoretical Insights

In this section, we provide preliminary theoretical justification for why the proposed minimax self-play procedure improves (i) the stability of the understanding branch, (ii) convergence of the alternating optimization, and (iii) coverage of the shared generative manifold. The analysis is intentionally model-agnostic and applies to a broad class of unified multimodal architectures.

[Figure 55]

Figure 13. Dominance timeline. The trajectory alternates between understanding and generation phases, illustrating a stable tug-ofwar rather than collapse to either side during training.

##### G.1.ConvergenceoftheMinimaxSelf-PlayDynamics

Recall the UniGame objective min

L(θU,θC) = E ℓU(θU) + λE ℓC(θC;θU) ,

max

θU

θC

(8) where the perturber maximizes the understanding loss while the understanding head minimizes both clean and adversarial losses, subject to a bounded perturbation ∥δ∥ ≤ εmax in the shared token space. In this subsection, we analyze an idealized version of this minimax problem to provide theoretical intuition, rather than a full convergence proof for the actual deep network implementation.

- Assumption 1 (Lipschitz continuity). The understanding loss ℓU(a | z,q) is L-Lipschitz continuous in the token embedding z and continuously differentiable in θU.
- Assumption 2 (Bounded perturbation set and parameter domain). The perturber operates within a compact, convex set

D = {δ : ∥δ∥ ≤ εmax}. (9)

Moreover, the parameter sets ΘU and ΘC for θU and θC are assumed to be compact and convex.

- Assumption 3 (Local nonconvex–concave structure). For any fixed θU ∈ ΘU, the function θC  → L(θU,θC) is (locally) concave on ΘC in a neighborhood of the stationary points of interest. Equivalently, the game is nonconvex in θU and (locally) concave in θC around those points.

###### Proposition 1 (First-order stationary point and stability). Under Assumptions 1–3, the minimax problem in

Eq. (8) admits at least one first-order stationary point (θU∗ ,θC∗ ), i.e.,

∇θUL(θU∗ ,θC∗ ) = 0, ∇θCL(θU∗ ,θC∗ ) = 0.

Moreover, for sufficiently small learning rates (ηU,ηC), gradient descent–ascent generates a bounded sequence and converges to a neighborhood of a first-order stationary point of L.

Sketch of proof. By Assumption 2, the feasible set in (θU,θC,δ) is compact and convex, so a minimax solution and hence a first-order stationary point exist. Assumption 1 guarantees that the loss is smooth in θU, and Assumption 3 provides a local nonconvex–concave structure: for each fixed θU, the objective is (locally) concave in θC. Under such smooth nonconvex–concave conditions, standard results for two-player minimax optimization show that gradient descent–ascent with sufficiently small step sizes (ηU,ηC) generates a bounded sequence and converges to an O(ηU + ηC) neighborhood of a first-order stationary point of L.

Implication. These assumptions suggest that the adversarial self-play dynamics are stable and tend not to diverge, even though the perturber and understanding branches pursue opposing objectives.

##### G.2. Robustness Improvement via Worst-Case Regularization

For a fixed sample z from the shared representation space, the creator seeks a worst-case perturbation

ℓU(z + δ). (10) Using a first-order Taylor expansion around z, we obtain

max

∥δ∥≤εmax

ℓU(z + δ) ≈ ℓU(z) + δ⊤∇zℓU(z). (11) The optimal perturbation under the norm constraint is

δ⋆ = εmax ∇zℓU(z) ∥∇zℓU(z)∥

. (12)

Substituting δ⋆ into Eq. (11) and taking expectation over the data distribution yields the expected adversarial loss

E ℓU(z) + εmax∥∇zℓU(z)∥ . (13)

Proposition 2 (Implicit gradient regularization). Adversarial self-play is equivalent, to first order, to adding a Jacobian-norm penalty:

LU,adv = LU + λεmax E ∥∇zℓU(z)∥ . (14)

Consequently, the understanding branch is encouraged to reduce its sensitivity to small perturbations in z, leading to locally flatter decision boundaries.

Implication. This explains the empirically observed improvements in robustness: the understanding head learns to be less sensitive to challenging input variations, improving both in-distribution and out-of-distribution performance as well as adversarial robustness.

##### G.3. Manifold-Expanding Effect of DecoderConstrained Perturbations

Unlike conventional pixel-space adversarial training, UniGame produces decoder-constrained adversarial examples

x˜ = G(z + δ), x˜ ∈ M, (15)

where G is the decoder and M is the decodable image manifold. This architecture ensures adversarial samples are:

- 1. On-manifold: x˜ remains realistic and visually plausible;
- 2. Semantically valid: filtered by CLIP-based or similar consistency criteria;
- 3. Near boundary regions: targeted towards regions where the understanding model is fragile.

Assumption 3 (Local bi-Lipschitz decoder). The decoder G is locally bi-Lipschitz on the relevant region of the token space, i.e., there exist constants 0 < m ≤ M < ∞ such that for all z1,z2 in a neighborhood Z,

m∥z1 − z2∥ ≤ ∥G(z1) − G(z2)∥ ≤ M∥z1 − z2∥. (16)

Lemma 1 (Adversarial manifold expansion). Under Assumption 3, for any z ∈ Z the support of the perturbed output distribution satisfies

supp(G(z + D)) ⊇ supp(G(z)), (17)

and expands the empirical training distribution toward regions where ∥∇zℓU(z)∥ is large.

Implication. The decoder-constrained perturbations induce a structured “inflation” of the data manifold towards decision boundary regions where the understanding head is uncertain. The hard-sample buffer B collects such samples, which are approximately located near the understanding decision boundary. Training on B reduces the empirical risk in these critical regions:

1 |B| x∈B

Rˆadv =

ℓU(x) (18)

acts as a surrogate for minimizing the out-of-distribution risk ROOD.

##### G.4. Summary of Theoretical Insights

The above analysis provides a theoretical lens on the benefits of the UniGame framework:

- 1. Convergence of self-play: Alternating gradient descent–ascent admits a stationary saddle point under mild smoothness and compactness assumptions.
- 2. Robust optimization view: The adversarial creator implicitly enforces a gradient-norm penalty (Eq. (14)), flattening the understanding decision boundary.
- 3. Manifold expansion: Decoder-constrained perturbations generate semantically valid hard samples that expand coverage of the decodable manifold towards challenging regions.
- 4. Alignment with empirical gains: These properties theoretically support the empirical improvements in understanding, consistency, out-of-distribution robustness, and adversarial robustness observed in our experiments.

