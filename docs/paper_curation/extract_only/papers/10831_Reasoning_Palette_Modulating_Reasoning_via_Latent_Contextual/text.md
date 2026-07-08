[Figure 1]

[Figure 2]

[Figure 3]

## Reasoning Palette: Modulating Reasoning via Latent Contextualization for Controllable Exploration for (V)LMs

# arXiv:2512.17206v1[cs.CV]19Dec2025

Rujiao Long1* Yang Li1,2* Xingyao Zhang1 Weixun Wang1 Tianqianjin Lin1,3 Xi Zhao1 Yuchi Xu1 Wenbo Su1 Junchi Yan2† Bo Zheng1†

1Alibaba Group 2Shanghai Jiao Tong University 3Zhejiang University

### Abstract

Exploration capacity shapes both inference-time performance and reinforcement learning (RL) training for large (vision-) language models, as stochastic sampling often yields redundant reasoning paths with little high-level diversity. This paper proposes Reasoning Palette, a novel latent-modulation framework that endows the model with a stochastic latent variable for strategic contextualization, guiding its internal planning prior to token generation. This latent context is inferred from the mean-pooled embedding of a question–answer pair via a variational autoencoder (VAE), where each sampled latent potentially encodes a distinct reasoning context. During inference, a sampled latent is decoded into learnable token prefixes and prepended to the input prompt, modulating the model’s internal reasoning trajectory. In this way, the model performs internal sampling over reasoning strategies prior to output generation, which shapes the style and structure of the entire response sequence. A brief supervised fine-tuning (SFT) warm-up phase allows the model to adapt to this latent conditioning. Within RL optimization, Reasoning Palette facilitates structured exploration by enabling on-demand injection for diverse reasoning modes, significantly enhancing exploration efficiency and sustained learning capability. Experiments across multiple reasoning benchmarks demonstrate that our method enables interpretable and controllable control over the (vision-) language model’s strategic behavior, thereby achieving consistent performance gains over standard RL methods.

### 1. Introduction

Large language models (LLMs) and vision-language models (VLMs) have rapidly advanced the frontier of artificial general intelligence, enabling multi-step problem solving, tool use, and grounded decision making across textual and

*Equal Contribution. †Correspondence Author.

[Figure 4]

Figure 1. Motivation: Injecting a Gaussian noise token embedding before the prompt embeddings of Qwen-4B-Base enables substantial gains in pass@k accuracy by merely sampling in the Gaussian, despite using greedy decoding for each candidate.

multimodal domains [13, 24, 75, 76]. Recently, reinforcement learning with verifiable rewards (RLVR) [29] has been widely verified as an effective post-training paradigm that elicits step-by-step thought for powerful reasoning, where a model’s outputs are optimized with RL objectives guided by automated correctness checks. This setup encourages models to produce intermediate reasoning tokens before issuing a final answer, giving rise to Large Reasoning Models (LRMs) that excel on challenging problems in mathematics [9, 21, 49, 65], coding [5, 25, 26], and agentic decision making [38, 70].

Exploration capacity shapes both inference-time performance and reinforcement learning (RL) training for large (vision-) language models. This long-standing bottleneck in RL training becomes even more critical when it comes to language models: standard sampling schemes (e.g., stochastic decoding with temperature or nucleus sampling) often revisit closely related trajectories, resulting in limited diversity at the levels of strategy and plan structure. This mismatch, between the high-level variability required to discover effective reasoning strategies and the low-level variability of token-level sampling, limits the efficiency and robustness of RL training for LLMs. Existing remedies primarily encourage local diversity, e.g., entropy regulariza-

Reconstruction

Mean-Pooling

PromptCoTSolution

[Figure 5]

[Figure 6]

Code

QA

Multi-Hop Reasoning

Large Language Model

Symbolic Computation

[Figure 7]

Factual Retrieval

Decoder

Encoder

[Figure 8]

Code Math QA …

Program Synthesis

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 9]

[Figure 10]

Arithmetic Computation

Aggregated Embedding

Recovered Embedding

Question Decoder

CoT Solution

Formal Proof

Math

Sampled Latent 𝑧

Latent Gaussian

(a) Latent space learning via VAE. The encoder maps the mean-pooled embedding of a question–answer pair into a Gaussian latent space, where distinct regions correspond to different reasoning strategies. The decoder reconstructs the embedding, enabling structured exploration through latent sampling.

(b) Latent-guided inference. During generation, a sampled latent variable z is decoded into prefixes and prepended to the input prompt. The prefixes modulate the LLM’s internal state, steering its reasoning trajectory and shaping the style and structure of the generated response.

- Figure 2. Overview of the Reasoning Palette framework: a latent-modulation system that enables strategic, diverse reasoning in LLMs/VLMs by sampling and decoding contextual latent variables to guide internal planning before token generation.

By identifying regions of the latent space associated with specific reasoning patterns (e.g., via encoding reasoning trajectories of targeted patterns), one can deliberately sample corresponding latents to modulate the model’s behavior on new prompts. This enables both targeted intervention and systematic analysis of how different reasoning palettes influence task success.

tion [16, 59], but they rarely provide mechanisms to shape the model’s internal planning. As a result, the model often explores similar reasoning paths with superficial differences, rather than probing distinct strategic modes.

Motivated by the phenomenon shown in Fig. 1, where prepending a single randomly sampled Gaussian noise embedding to the input of Qwen-4B-Base dramatically boosts Pass@k performance even under greedy decoding, we propose Reasoning Palette, a latent modulation framework that endows the model with a compact stochastic variable acting as a latent context for internal planning, as shown in Fig. 2. Concretely, we construct a latent space via a variational autoencoder (VAE), where each latent corresponds to a mean-pooled embedding of question-answer pairs. Each sampled latent is decoded into a short sequence of learnable token prefixes and prepended to the prompt during inference, thereby steering the model’s internal trajectory before the first output token is produced. This design transforms exploration from token-level randomness into structured, pre-generative sampling over reasoning strategies. The latent context can be viewed as a palette of reasoning modes that can be explored through sampling, where different latent contexts inspire distinct organizational and explanatory styles of reasoning (e.g., mathematical reasoning, code reasoning, or question-answering logic).

We summarize our contributions as follows: (1) We introduce a stochastic latent variable learned by VAE that modulates internal planning in LLMs and VLMs via token prefixes, enabling sampling over reasoning strategies before generation begins. (2) We propose a two-stage procedure to promote the reasoning capability, SFT warm-up to sensitize the model to latent conditioning, followed by RL with latent sampling to enhance exploration in optimization, which yields structured exploration and sustained learning. (3) Experiments on multiple reasoning benchmarks show that Reasoning Palette shows clear gains in diversity and controllability during inference, and consistently improves performance over standard baselines in RL, significantly enhancing exploration efficiency and controllability.

### 2. Related Work

Controllable Reasoning and Prompting. Chain-ofthought (CoT) prompting elicits step-by-step rationales that improve reasoning accuracy [61], and self-consistency enhances reliability through majority voting over diverse rationales [60]. Complementary paradigms structure the decomposition or search process [68, 69, 78]. Recently, soft chain-of-thought [64] replaces discrete reasoning traces with learnable continuous representations to boost efficiency. Beyond hand-crafted prompts, a large literature studies controllable generation via discrete and continuous prompt parameterizations, e.g., soft prompt tuning [32] and prefix-tuning [35], which learn continuous prefix embeddings that steer model behavior with minimal updates, or

A brief supervised fine-tuning (SFT) warm-up aligns the base model to respond to these latent-conditioned prefixes without sacrificing its general capabilities. Within RL, we treat the latent as an auxiliary control variable sampled per episode, enabling on-demand injection of diverse reasoning contexts. This facilitates efficient and sustained exploration: rather than repeatedly sampling near-duplicate chains of thought, the policy explores distinct strategic families, improving the chance of discovering high-reward behaviors and accelerating learning. Crucially, because the prefixes are generated by an external variational autoencoder (VAE), the reasoning mode is explicitly controllable.

discrete prompt search methods [12, 51]. Orthogonal control uses external guidance at inference, such as plug-andplay gradients and classifiers [10, 28], or activation steering to modulate internal representations [55].

Reinforcement Learning for LLMs. Reinforcement learning (RL) is central to LLM post-training, popularized by RLHF for instruction following and preference alignment [2, 7, 44, 48, 79]. Two strands dominate: online policy-gradient methods with on-policy rollouts [47, 49, 62] and offline preference optimization without on-policy sampling [11, 42, 45]. Reinforcement Learning from AI Feedback (RLAiF) extends this by using LLMs to generate feedback for alignment [3, 15, 31, 52]. Outcome-based RL with verifiable rewards (RLVR) [30, 49] has driven recent advances through deterministic rule-based rewards [57, 63], with large-scale systems demonstrating that correctnessguided RL elicits extended reasoning [14, 29, 43, 53, 66]. A growing ecosystem studies data aggregation, policy updates, scalable training infrastructure, and critical reasoning nodes that enhance optimization [36, 39, 58, 73, 74, 77].

Latent Representation Learning. Learning compact and disentangled latent representations has been a longstanding theme in generative modeling. Canonical approaches include variational autoencoders (VAEs) [27] that optimize a tractable ELBO to learn continuous, sampleable latents, generative adversarial networks (GANs) that leverage adversarial formulations to align latent codes with data distributions, and diffusion models [23, 67] that construct latent-like representations by progressively denoising inputs across timesteps. Discrete codebooks (VQ-VAE) [56] provide robust, compositional latents with high reconstruction quality. For text, latent-variable language models capture global factors while mitigating posterior collapse via architectural and training advances [4, 19]. More recently, iterative and multi-step generation strategies that preserve or refine intermediate latent embeddings have been explored to support multi-step reasoning and better align latent dynamics with downstream generation [17].

### 3. Methodology

#### 3.1. Preliminaries and Notations

Let q ∼ Q denote a question, a decoder-only policy πθ autoregressively generates an output token sequence o = (o1,··· ,oT). When visual input v is present, we encode it with a visual encoder fv(·) to obtain visual tokens

z = fv(v) = (z1,...,zM), zi ∈ Rd, (1)

and the question q include the visual features. The autoregressive policy is formulated as:

pθ(o | q) =

T

pθ ot | q,o<t . (2)

t=1

The RL objective maximizes expected verifier reward:

θ(·|q)[r(o;q)]. (3)

J (θ) := Eq∼Q,o∼π

max

θ

Proximal Policy Optimization (PPO) [47] is a standard approach for optimization, which updates πθ using data from a frozen old policy πold and a clipped surrogate objective:

|o|

πθ(ot|q, o<t) πθold(ot|q, o<t)

1 |o|

J (θ) =Eq∼Q,o∼πθ(·|q)

min

At,

t=1

πθ(ot|q, o<t) πθold(ot|q, o<t)

, 1−ϵ, 1+ϵ At

clip

(4)

where At is the advantage at step t, typically estimated via Generalized Advantage Estimation (GAE) [46], and ϵ is a clipping hyperparameter for stabilizing updates. Group Relative Policy Optimization (GRPO) [49] replaces the critic with group-based advantage estimation. For a prompt q with G responses and rewards {ri}Gi=1, it computes Aˆi,t = ri−mean({ri}Gi=1)

std({ri}Gi=1) , enhancing gradient signal reliability under sparse rewards. GRPO further augments Eq. (4) with a KL penalty to stabilize training.

#### 3.2. Latent Space Learning for Reasoning Patterns

We learn a compact, continuous latent space Z ⊂ Rk that encodes diverse reasoning strategies. To construct this space, we train a Variational Autoencoder (VAE) [27] on a dataset D = {(q(i),o(i))}Ni=1 of question–answer pairs with high-quality reasoning traces. To ensure diversity in reasoning patterns across different modes, we curate the training data from a broad spectrum of sources spanning mathematics, question answering, and code generation. Each of these domains exhibits distinct reasoning patterns, while also containing multiple sub-modes. For example, question answering encompasses strategies such as multi-hop reasoning, factual retrieval, and more. This intentional design ensures that the source data and consequently the learned latent space reflects a rich and varied repertoire of reasoning behaviors.

For each question–answer pair (q,o), we aim to learn a compact latent representation z that captures high-level reasoning strategies and can be flexibly decoded into a short prefix to guide generation. Critically, we desire two practical properties for downstream use:

- • The number of prefix tokens L is adjustable at inference time, enabling users to adjust the control intensity through the length: longer prefixes provide stronger behavioral guidance and greater contextual diversity, while shorter prefixes offer lighter intervention.
- • The prefix can be easily absorbed by the pretrained model without extensive retraining. Ideally, it resides in a space that is naturally aligned with the model’s native token

embedding distribution, ensuring that the injected prefix harmonizes with the model’s internal dynamics.

To satisfy these desiderata, we construct the latent space directly over the model’s frozen token embedding layer E(·) (e.g., from Qwen3-4B), which maps discrete tokens to vectors in Rd. Specifically, we first form the concatenated sequence [q;o] = (x1,...,xN), then compute its contextual summary via mean-pooling of the raw token embeddings:

N

1 N

E [q;o]i ∈ Rd. (5)

h =

i=1

Since h lives in the same space as individual token embeddings, any reconstruction hˆ produced by the VAE decoder can be readily interpreted as a pseudo-token or used to initialize a small set of prefix embeddings that are distributionally consistent with the model’s input expectations. Moreover, mean-pooling provides a fixed-dimensional, orderinsensitive summary that abstracts away surface-level token variations, making it well-suited for representing reasoning style rather than verbatim output.

We then model the distribution over reasoning strategies by learning a probabilistic latent space using a Variational Autoencoder (VAE) [27]. The encoder Eϕ, implemented as a multi-layer perceptron (MLP), maps the embedding h to the parameters of a diagonal Gaussian distribution:

µ,σ = Eϕ(h), z ∼ N(µ,diag(σ2)), (6)

where µ ∈ Rk and σ ∈ Rk>0 define the posterior qϕ(z | h). The decoder Dψ, also an MLP, reconstructs the original embedding from a sampled latent code:

##### hˆ = Dψ(z). (7)

The VAE is trained end-to-end by minimizing the evidence lower bound (ELBO), which balances reconstruction accuracy against latent space regularization:

LVAE = Ez∼q

ϕ(z|h) ∥h − hˆ∥2 + β · KL qϕ(z | h)∥p(z) ,

(8) where p(z) = N(0,I) is a standard isotropic Gaussian prior, and β > 0 is a hyperparameter that controls the tradeoff between faithful reconstruction and smoothness of the latent manifold. A well-tuned β encourages disentangled and semantically meaningful regions in Z, such that nearby latents correspond to similar reasoning patterns, while distant latents induce qualitatively different strategies.

#### 3.3. Latent-Guided Inference and SFT Adaptation

Latent-Guided Inference. Given a trained VAE (Eϕ,Dψ), we leverage its latent space to modulate the reasoning behavior of the base model during both inference and RL training. At inference time, we sample a latent vector

z ∼ N(0,I) from the prior and decode it into a sequence of L prefix embeddings:

pz = Dψ(z(1)),Dψ(z(2)),...,Dψ(z(L)) ∈ RL×d, (9)

where each z(ℓ) is an independent sample from the same prior (or, optionally, a single z tiled L times for stronger coherence). For inference, these continuous prefix embeddings pz are prepended to the input token embeddings of the prompt q for autoregressive generation:

q˜ = [pz;E(q)], then o ∼ πθ(· | q˜). (10)

Crucially, the prefix length L is a tunable hyperparameter: a larger L provides stronger and more structured guidance (e.g., for complex multi-step tasks), while a smaller L offers lightweight modulation with minimal computational overhead. This flexibility allows users to balance control intensity against inference cost. Moreover, since the VAE is fixed after SFT, the latent space remains a stable, interpretable coordinate system. This enables post-hoc analysis (e.g., clustering high-reward latents) and targeted intervention (e.g., biasing sampling toward known effective regions). Thus, Reasoning Palette not only improves exploration efficiency but also provides a transparent mechanism for steering and diagnosing model behavior.

SFT Adaptation. To ensure the base (V)LM can effectively interpret latent-conditioned prefixes without compromising its general instruction-following capabilities, we perform a lightweight supervised fine-tuning (SFT) warm-up phase. Since the learned posterior distribution qϕ(z | h) can deviate from the isotropic Gaussian prior p(z) = N(0,I), to avoid degraded generalization capability when using the latent Gaussian distribution during inference, we construct the SFT dataset DSFT not by encoding ground-truth examples into latents, but by sampling latents directly from the prior z ∼ N(0,I), decoding them into prefix embeddings p = Dψ(z), and pairing these synthetic prefixes with original prompt–response pairs (q,o) from the VAE training corpus.

We strictly limit the SFT duration (typically 10 iterations) to prevent the model from downweighting the impact of the prefix and overfitting to any single response pattern. Excessive SFT on deterministic targets can suppress the model’s responsiveness to latent variation, effectively washing out the diversity induced by different z samples. By keeping SFT minimal, we strike a balance: the model learns to condition its output on arbitrary prefix embeddings drawn from the VAE decoder’s range, while retaining sufficient stochasticity to respond flexibly to novel latent codes.

The model is then fine-tuned to minimize the standard language modeling loss:

LSFT = −E(p,q,o)∼D

SFT

##### [log pθ(o | [p;E(q)])], (11)

where [p;E(q)] denotes the concatenation of prefix embeddings and question token embeddings. Notably, during SFT we fix the prefix length to L = 1, using only a single learnable token to modulate behavior. This minimal intervention enables the model to adapt to a noisy prefix token while avoiding overwhelming the model with strong control signals early on. During inference and RL, we freely increase L (e.g., L = 4 or 8) to enable richer, compositional guidance. This SFT phase sensitizes the model to the latent signal while preserving its pre-trained knowledge, enabling seamless integration of reasoning-mode control during subsequent RL.

#### 3.4. Controllable Exploration in RL

During reinforcement learning, we treat the latent variable z as an auxiliary control signal that induces structured exploration over high-level reasoning strategies. For each training episode, we sample a latent code z ∼ N(0,I), decode it into a sequence of L prefix embeddings p = (Dψ(z(1)),...,Dψ(z(L))), and condition the policy on this prefix. The resulting policy becomes:

πθ(o | q,z) =

T

pθ ot | [p;E(q)],o<t , (12)

t=1

and the RL objective is extended to:

θ(·|q,z) r(o;q) . (13)

Ez∼p(z)Eq∼Q,o∼π

max

θ

In practice, we approximate this by sampling one z per prompt during rollouts, which efficiently diversifies reasoning trajectories across episodes.

This design enables multi-granular exploration control in RL, operating along two complementary dimensions: (1) temporal scheduling, where the strength of latent guidance is modulated over the course of training to follow a highlevel exploration-to-exploitation trajectory; and (2) intragroup diversity control, where, within each training batch (or GRPO group), we regulate the proportion of rollouts that receive latent-conditioned prefixes, thereby fine-tuning the balance between diverse exploration and stable policy updates at the micro level. Based on this dual-axis framework, we evaluate two principled scheduling strategies:

- • Two-Phase: For the first 50% of training steps, all rollouts use latent prefixes with L = 8 to maximize strategic diversity; for the remaining steps, latent guidance is completely disabled (L = 0) across all samples, allowing the policy to fully exploit the high-reward behaviors discovered during exploration.
- • Linear Decay: Over the full training duration, we linearly reduce the fraction of latent-guided rollouts per group from 100% to 0%, while keeping L = 8 for those that are guided. This implements a smooth transition: early groups contain mostly diverse trajectories, while

later groups increasingly rely on high-confidence generations for advantage estimation and policy refinement.

Both strategies embody the classical explorationexploitation trade-off, but operate at the level of reasoning architecture rather than token-level stochasticity. As shown in Fig. 6, latent-augmented variants exhibit slower initial accuracy gains due to the cost of exploring diverse, sometimes suboptimal, reasoning paths, but ultimately surpass the GRPO baseline in the latter half of training. This late-stage overtaking occurs because early exploration exposes the policy to higher-quality regions of behavior space, which are then consolidated during the exploitation phase.

We formalize this time-varying control via a scheduled policy that mixes latent-guided and standard rollouts. Let τ ∈ [0,1] denote normalized training progress, and let ρ(τ) ∈ [0,1] be the scheduled fraction of rollouts that receive latent conditioning at step τ. The optimization objective becomes a weighted combination:

Jsched(θ) = EτEq∼Q [ρ(τ) · LPPO(θ; q, z) + (1 − ρ(τ)) · LPPO(θ; q)] ,

(14)

where ρ(τ) defines the exploration schedule. For the twophase strategy, ρ(τ) = 1 if τ < 0.5 and 0 otherwise; for linear decay, ρ(τ) = 1 − τ. In both cases, latent-guided rollouts use a fixed prefix length (e.g., L = 8), while unguided rollouts use L = 0. In practice, τ is determined deterministically by the current training step, and during each GRPO group rollout, we sample a fraction ρ(τ) of responses with latent prefixes and the remainder without.

### 4. Experiments

We conduct a comprehensive empirical study to validate the efficacy of Reasoning Palette across both inference and reinforcement learning settings. Our experiments are structured around two core questions: (1) Does latent-guided inference enable diverse and controllable reasoning? (2) Does integrating latent-conditioned exploration into RL improve training efficiency, stability, and final performance on challenging reasoning tasks? We evaluate primarily on mathematical reasoning benchmarks due to their clear reward structure and sensitivity to reasoning strategy diversity.

#### 4.1. Evaluation of Latent-Guided Inference

Direct Gaussian Noise Injection. Fig. 1 illustrates a key finding motivating our methodology: injecting a single Gaussian noise token (i.e., sampling z ∼ N(0,I) and using Dψ(z) as a prefix) before the prompt embeddings, while using greedy decoding for each candidate, yields substantial gains in pass@k accuracy across multiple benchmarks. For instance, on GSM8K, pass@32 improves from 52.9% to 85.3% with latent sampling, despite no change in decoding strategy per sample. This demonstrates that performance gains stem not from token-level stochasticity but

20

50

Count

Count

Count

10

20

Count

10

25

0

0

0

0

Datasets

10

MetaMathQA

7.5

20

competition_math

20

prm800k

5

5.0

codeparrot

10

ShareGPT_Vicuna

###### TS EDimension2

###### TS EDimension2

###### PCA Dimension2

0

PCA Dimension2

0

2.5

0

5

0.0

20

10

2.5

10

###### Datasets

###### Datasets

###### Datasets

40

5.0

MetaMathQA

MetaMathQA

MetaMathQA

15

20

competition_math

competition_math

competition_math

prm800k

prm800k

prm800k

7.5

20

codeparrot

codeparrot

codeparrot

60

30

ShareGPT_Vicuna

ShareGPT_Vicuna

ShareGPT_Vicuna

10.0

40 30 20 10 0 10 20 30 40

0 20

40 30 20 10 0 10 20 30 40

0 20

20 10 0 10 20 30

0 10

10.0 7.5 5.0 2.5 0.0 2.5 5.0 7.5

0 10

PCA Dimension 1

TS E Dimension 1

TS E Dimension 1

PCA Dimension 1

Count

Count

Count

Count

(a) PCA of Generated Prefixes

(b) t-SNE of Generated Prefixes

(c) PCA of Latents

(d) t-SNE of Latents

- Figure 3. Visualization of the learned latent space and generated prefix embeddings via PCA and t-SNE. Left two panels: projections of decoded prefix embeddings Dψ(z) (colored by domains). Right two panels: projections of the corresponding latent vectors z = Eϕ(mean-pool(q; o)). Clear clustering by reasoning domain in both spaces confirms that the VAE disentangles high-level reasoning strategies into distinct regions of the latent space.

Table 1. Pass@8 results of latent-guided inference. Latent Source MATH500 Olympic GSM8K

codeparrot (Code) 70.8 42.95 94.47 MetaMathQA (Math) 72.4 46.11 95.0 ShareGPT Vicuna (QA) 71.0 45.47 93.63

from strategic variation induced by the latent context. However, naively injecting random noise often degrades performance due to misalignment with the model’s native embedding space. To address this, we design our framework to ensure that latent-guided prefixes remain both diversityinfluential and well-aligned with the embedding distribution, enabling controllable and effective inference.

Latent-Guided Inference on LLMs: Targeted Intervention via Biased Latent Sampling. We evaluate latentguided inference on a supervised fine-tuned (SFT) Qwen34B-Base model. The VAE is trained on a curated dataset of 5K high-quality question–answer pairs spanning mathematics, code generation, and multi-hop question answering. Specifically, the training data includes MetaMathQA [72], competition math [22], PRM800K [37], ShareGPT Vicuna unfiltered [6], and CodeParrot [54]. After SFT adaptation (10 iterations with prefix length L = 1), we assess pass@8 performance under latent sampling during inference to demonstrate controllable reasoning.

We perform domain-specific intervention by leveraging the structure of the latent space. Specifically, we collect representative reasoning trajectories from three distinct domains: mathematical problem solving (e.g., from MetaMathQA), code generation (e.g., from CodeParrot), and general question answering (e.g., from ShareGPT Vicuna), and encode them into the latent space using the trained encoder Eϕ. For each domain, we compute the empirical mean and covariance of its latent embeddings, thereby identifying well-separated regions that correspond to characteristic reasoning styles. During inference, given a new prompt, we restrict latent sampling to the region associated with the given domain and decode it via Dψ(z) to obtain a domain-aligned prefix. As shown in Table 1, using math-

domain latents yields the best performance across all mathematical benchmarks: it achieves 72.4 on MATH500, 46.11 on Olympic, and 95.0 on GSM8K, consistently outperforming code- or QA-biased latents.

Latent-Guided Inference on VLMs: Enhancing Grounding via Structured Diversity. We further extend our framework to vision-language models (VLMs) and evaluate its effectiveness on referring expression comprehension, a canonical grounding task where the model must localize an object in an image based on a natural language description. Following the evaluation protocol in VLM-R1 [50], we measure accuracy as the fraction of predictions whose bounding box achieves an IoU ≥ 0.5 with the ground-truth annotation.

Our VLM architecture builds upon a Qwen2.5VL-3B backbone: visual inputs are encoded via a ViT into a sequence of image tokens, which replace special image placeholders in the text prompt. To enable latent-guided control, we sample a noise vector z ∈ R1024 from N(0,I), reshape it into an 8 × 128 matrix, and pass it through a lightweight GPT-style decoder to produce 8 prefix embeddings A ∈ R8×d. These are prepended to the prompt embedding (which already includes the image tokens) before being fed into the LLM decoder. During inference, we compare four configurations: 1) Baseline (greedy): standard greedy decoding without latent prefixes; 2) Baseline + sampling: stochastic decoding (e.g., temperature=0.7) without latent control; 3) Latent-guided (greedy): greedy decoding with latent prefixes (L = 8); 4) Latent-guided + sampling: stochastic decoding combined with latent prefixes.

We evaluate all variants using pass@32 on three standard benchmarks: RefCOCO [71], RefCOCO+ [71], and RefCOCOg [41]. As shown in Table 2 and Fig. 4, latent-guided inference significantly boosts performance under greedy decoding. Notably, latent-guided + sampling achieves the best results across all datasets, confirming that latent diversity and decoding stochasticity are complementary. Crucially, the improvement from latent guidance alone (greedy vs. latent-greedy) exceeds that from sampling alone (greedy

|[Figure 11]|
|---|

|[Figure 12]<br><br>|𝐈𝐎𝐔 = 𝟎.𝟓𝟓|
|---|
|
|---|

[Figure 13]

[Figure 14]

𝐈𝐎𝐔 = 𝟎.𝟒𝟏

|[Figure 15]|
|---|

|[Figure 16]<br><br>|𝐈𝐎𝐔 = 𝟎.𝟗𝟒|
|---|
|
|---|

|[Figure 17]<br><br>|𝐈𝐎𝐔 = 𝟎.𝟑𝟕|
|---|
|
|---|

Figure 4. Pass@32 curves on the RefCOCO datasets. Table 2. Pass@32 on referring expression comprehension.

Method RefCOCO RefCOCO+ RefCOCOg

Baseline (greedy) 2.0 2.0 4.67 Baseline + sampling 65.07 62.57 72.0 Latent-guided (greedy) 72.07 73.07 73.1 Latent-guided + sampling 87.53 86.03 85.7

Figure 5. Qualitative results on RefCOCO dataset. From left to right: input image with the ground-truth bounding box, prediction from Qwen2.5VL-3B (greedy decoding), and prediction from our method, Qwen2.5VL-3B (greedy decoding) with a randomly sampled latent. The referring expressions for the top and bottom rows are train closest to the bottom and a zebra standing behind two other zebras, with only its mane and rear showing, respectively.

vs. baseline+sampling), underscoring the unique value of structured, pre-generative exploration in multimodal reasoning. These results validate that Reasoning Palette generalizes beyond pure language tasks: by injecting domainagnostic yet semantically rich latent context, it enhances the model’s ability to explore diverse grounding hypotheses, leading to more robust and accurate spatial reasoning.

from the mathematical region, underscoring the divergence in their underlying reasoning patterns. Interestingly, the QA cluster displays the largest spatial spread, with some outlier samples extending toward the code and math regions. This suggests that open-ended question answering encompasses a broader spectrum of reasoning strategies, including occasional algorithmic or quantitative elements, consistent with its heterogeneous nature. Crucially, this structural coherence is preserved not only in the latent vectors z but also in their decoded prefix tokens p = Dψ(z). This mutual consistency confirms that the VAE has successfully learned a disentangled and functional mapping, one that captures high-level reasoning semantics and translates them into effective behavioral guidance for the model.

Upon analysis, we find that under the Qwen2.5VL-3B with greedy decoding setting, the model often correctly identifies the target entity but fails to produce outputs in the required format, leading to low metric scores. The performance improves substantially when evaluated with multiple sampling passes (Pass@32). Fig. 5 shows qualitative results, where introducing the latent leads to significant performance improvements on challenging cases.

Latent Space Analysis. To verify the controllability of our latent space, we perform post-hoc analysis using the same dataset employed for VAE training. We collect 500 reasoning trajectories covering the five datasets, including MetaMathQA (Math), competition math (Math), PRM800K (Math), ShareGPT Vicuna unfiltered (QA), and CodeParrot (Code), 100 trajectories for each. Each (q,o) pair is encoded into its latent representation z = Eϕ(mean-pool(q;o)), and we generate prefix tokens using the latents p = Dϕ(z). We visualize the resulting latents and the corresponding prefixes using dimensionality reduction techniques including PCA [1] and t-SNE [40].

#### 4.2. Reinforcement Learning with Latent Guidance

We now present a comprehensive evaluation of Reasoning Palette within reinforcement learning (RL) training, focusing on its ability to enhance exploration and final performance on complex reasoning tasks. Our experiments are built upon two state-of-the-art RL algorithms for language models: Group Relative Policy Optimization (GRPO) [49] and Reward-Label Optimized Off-policy (RLOO) [15]. Both methods rely on verifiable, outcome-based rewards

The visualization in Fig. 3 reveals interpretable clustering aligned with reasoning domains. Notably, the two advanced Math datasets competition math and PRM800K exhibit highly overlapping latent distributions, reflecting their shared focus on rigorous, formal mathematical reasoning. In contrast, MetaMathQA, while still within the mathematical cluster, occupies a slightly offset region, likely due to its emphasis on step-by-step pedagogical explanations rather than competition-level deductions. This subtle separation demonstrates the latent space’s sensitivity to fine-grained reasoning styles even within the same broad domain.

All models are initialized from the SFT-adapted checkpoints as described in Sec. 3.3, ensuring sensitivity to latent-conditioned prefixes while preserving general capabilities. The baselines are finetuned with standard SFT for fair comparison. Training is conducted on the DeepMath dataset [20], a large-scale collection of mathematical problems with verified solutions and step-by-step rationales.

We compare two latent scheduling strategies that modulate the strength of reasoning-mode control during RL, i.e., Two-phase and Linear decay, as detailed in Sec. 3.4. All models generate 8 responses per prompt during rollouts for advantage estimation. Evaluation uses pass@1 for mathematical datasets including AMC23 [34], GSM8K [8], Min-

Meanwhile, code generation (CodeParrot) and general QA (ShareGPT Vicuna) form two more separated clusters

Table 3. Results of Math reasoning over different model sizes. Bold denotes the best results.

Method MATH500 OlympiadBench AMC23 GSM8K MinervaMath Avg.

Qwen3-1.7B-Base GRPO 23.35 8.05 10.31 51.86 12.36 21.18

+ Reasoning Palette (Two-Phase) 26.02 +2.67 9.63 +1.58 9.69 -0.62 57.23 +5.37 11.99 -0.37 22.91 +1.73 + Reasoning Palette (Linear Decay) 27.32 +3.97 9.89 +1.84 12.80 +2.49 57.00 +5.14 12.82 +0.46 24.05 +2.87 RLOO 24.75 8.61 7.50 54.96 13.05 21.77

+ Reasoning Palette (Two-Phase) 25.77 +1.02 9.50 +0.89 10.31 +2.81 55.83 +0.87 12.59 -0.46 22.80 +1.03 + Reasoning Palette (Linear Decay) 25.72 +0.97 8.87 +0.26 11.25 +3.75 57.62 +2.66 13.98 +0.93 23.49 +1.72

Qwen3-4B-Base GRPO 68.65 41.32 50.94 91.05 39.39 58.27

+ Reasoning Palette (Two-Phase) 70.53 +1.88 43.95 +2.63 55.00 +4.06 92.19 +1.14 42.20 +2.81 60.77 +2.50 + Reasoning Palette (Linear Decay) 72.67 +4.02 45.29 +3.97 47.50 -3.44 92.64 +1.59 42.53 +3.14 60.12 +1.85 RLOO 74.55 43.53 51.25 91.55 36.07 59.39

+ Reasoning Palette (Two-Phase) 75.90 +1.35 44.21 +0.68 52.81 +1.56 91.80 +0.25 38.75 +2.68 60.69 +1.30

- + Reasoning Palette (Linear Decay) 71.50 -3.05 44.76 +1.23 55.62 +4.37 91.85 +0.3 42.67 +6.6 61.28 +1.89 Qwen3-8B-Base

GRPO 70.05 44.34 54.69 92.13 39.07 60.05

+ Reasoning Palette (Two-Phase) 70.38 +0.33 43.24 -1.10 55.62 +0.93 92.24 +0.11 41.37 +2.30 60.57 +0.52 + Reasoning Palette (Linear Decay) 70.78 +0.73 44.74 +0.40 57.19 +2.50 92.22 +0.09 40.91 +1.84 61.17 +1.12 RLOO 69.53 43.76 55.00 91.82 39.48 59.91

+ Reasoning Palette (Two-Phase) 72.27 +2.74 46.00 +2.24 57.50 +2.50 93.04 +1.22 42.57 +3.09 62.28 +2.37

- + Reasoning Palette (Linear Decay) 72.20 +2.67 46.61 +2.85 59.38 +4.38 93.03 +1.21 43.77 +4.29 63.00 +3.09

[Figure 18]

Figure 6. Performance curves of the GRPO baseline and the proposed latent variants during training. Latent variants perform more thorough exploration in early stages of training and then shift toward exploitation in the latter stages, resulting in a performance curve that gradually overtakes baselines.

ervaMath [33], MATH500 [21], and OlympiadBench [18].

Main Results. Table 3 reports performance across five core mathematical reasoning benchmarks. We evaluate on both Qwen3-1.7B-Base, Qwen3-4B-Base, and Qwen3-8B-Base backbones to assess scalability. Across all settings, Reasoning Palette consistently outperforms the respective RL baselines, demonstrating the robustness of our approach. On Qwen3-8B-Base + RLOO, Reasoning Palette improves average performance by +3.09 points, with the largest gains on complex domains (AMC23: +4.38, MinervaMath: +4.29). The linear-decay schedule yields slightly better final performance than the two-phase schedule (+0.75 points on average), suggesting that a smooth transition enables the model to adapt more gracefully during training, facilitating a more effective shift from exploration to exploitation and ultimately leading to higher asymptotic performance.

Learning Dynamics and Analysis. Fig. 6 shows the training curves of the GRPO baseline and the proposed latent

variants, indicating that the gain from injecting a latent exploration signal is not a simple one-shot boost but a change in the whole exploration-exploitation trajectory. The latent variants explore more broadly in the early phase of training which produces slower initial gains in pointwise accuracy but exposes the learner to higher-quality regions of policy space. As training progresses, the latent diversity signal is reduced, and the policy concentrates on the better behaviors discovered earlier. The net effect is a smoother transition from exploration to exploitation and a modest but consistent improvement in final accuracy relative to the GRPO baseline. Here we compare two latent scheduling strategies. The two-phase control strategy exhibits a clear performance improvement in the latter half stage, whereas the linear-decay strategy transitions gradually from exploration to exploitation, resulting in a smoother overall transition.

### 5. Conclusion

We introduce Reasoning Palette, a lightweight framework that shifts exploration in LLM reinforcement learning from token-level randomness to strategy-level diversity by sampling from a VAE-learned latent space of reasoning patterns. These latents are decoded into prefix embeddings that steer the model’s internal planning before generation. With a brief SFT warm-up and a two-phase RL schedule (exploration with latent prefixes followed by exploitation without), our method consistently boosts performance across reasoning benchmarks and enables interpretable, domainaware control during inference, demonstrating that structured latent modulation is a powerful and practical approach to enhancing reasoning in large models.

### References

- [1] Herv´e Abdi and Lynne J Williams. Principal component analysis. Wiley interdisciplinary reviews: computational statistics, 2(4):433–459, 2010. 7
- [2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022. 3
- [3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli TranJohnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022. 3
- [4] Samuel Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space. In Proceedings of the 20th SIGNLL conference on computational natural language learning, pages 10–21, 2016. 3
- [5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 1
- [6] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 6
- [7] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 3
- [8] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 7
- [9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 1
- [10] Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to con-

trolled text generation. arXiv preprint arXiv:1912.02164,

2019. 3

- [11] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306,

2024. 3

- [12] Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rockt¨aschel. Promptbreeder: Self-referential self-improvement via prompt evolution. arXiv preprint arXiv:2309.16797, 2023. 3
- [13] Akash Ghosh, Arkadeep Acharya, Sriparna Saha, Vinija Jain, and Aman Chadha. Exploring the frontier of visionlanguage models: A survey of current methodologies and future directions. arXiv preprint arXiv:2404.07214, 2024. 1
- [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3
- [15] Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, and Mathieu Blondel. Direct language model alignment from online ai feedback, 2024. 3, 7
- [16] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. Pmlr, 2018. 2
- [17] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024. 3
- [18] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024. 8
- [19] Junxian He, Daniel Spokoyny, Graham Neubig, and Taylor Berg-Kirkpatrick. Lagging inference networks and posterior collapse in variational autoencoders. arXiv preprint arXiv:1901.05534, 2019. 3
- [20] Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A largescale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025. 7
- [21] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 1, 8
- [22] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 6

- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [24] Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. In Findings of the association for computational linguistics: ACL 2023, pages 1049–1065, 2023. 1
- [25] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024. 1
- [26] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swebench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023. 1
- [27] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3, 4
- [28] Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq Joty, Richard Socher, and Nazneen Fatema Rajani. Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367, 2020. 3
- [29] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model posttraining. arXiv preprint arXiv:2411.15124, 2024. 1, 3
- [30] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training, 2025. 3
- [31] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023. 3
- [32] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021. 2
- [33] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35: 3843–3857, 2022. 8
- [34] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13

(9):9, 2024. 7

- [35] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021. 2

- [36] Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, et al. Attention illuminates llm reasoning: The preplan-and-anchor rhythm enables fine-grained policy optimization. arXiv preprint arXiv:2510.13554, 2025. 3
- [37] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. 6
- [38] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2023. 1
- [39] Zihe Liu, Jiashun Liu, Yancheng He, Weixun Wang, Jiaheng Liu, Ling Pan, Xinyu Hu, Shaopan Xiong, Ju Huang, Jian Hu, et al. Part i: Tricks or traps? a deep dive into rl for llm reasoning. arXiv preprint arXiv:2508.08221, 2025. 3
- [40] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9 (Nov):2579–2605, 2008. 7
- [41] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 6
- [42] Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems, 37: 124198–124235, 2024. 3
- [43] OpenAI. Learning to reason with llms, 2024. 3
- [44] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 3
- [45] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 3
- [46] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015. 3
- [47] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [48] John Schulman, Barret Zoph, Christina Kim, Jacob Hilton, Jacob Menick, Jiayi Weng, Juan Felipe Ceron Uribe, Liam Fedus, Luke Metz, Michael Pokorny, et al. Chatgpt: Optimizing language models for dialogue. OpenAI blog, 2(4),

2022. 3

- [49] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 3, 7

- [50] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 6
- [51] Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980, 2020. 3
- [52] Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. Principle-driven self-alignment of language models from scratch with minimal human supervision. Advances in Neural Information Processing Systems, 36:2511–2565,

2023. 3

- [53] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025. 3
- [54] Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. Natural language processing with transformers. ” O’Reilly Media, Inc.”, 2022. 6
- [55] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023. 3
- [56] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3
- [57] Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024. 3
- [58] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025. 3
- [59] Weixun Wang, Shaopan Xiong, Gengru Chen, Wei Gao, Sheng Guo, Yancheng He, Ju Huang, Jiaheng Liu, Zhendong Li, Xiaoyang Li, et al. Reinforcement learning optimization for large-scale learning: An efficient and user-friendly scaling library. arXiv preprint arXiv:2506.06122, 2025. 2
- [60] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171,

2022. 2

- [61] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 2
- [62] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992. 3

- [63] Huajian Xin, Z. Z. Ren, Junxiao Song, Zhihong Shao, Wanjia Zhao, Haocheng Wang, Bo Liu, Liyue Zhang, Xuan Lu, Qiushi Du, Wenjun Gao, Qihao Zhu, Dejian Yang, Zhibin Gou, Z. F. Wu, Fuli Luo, and Chong Ruan. Deepseek-proverv1.5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search, 2024. 3
- [64] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot: Soft chain-of-thought for efficient reasoning with llms. arXiv preprint arXiv:2502.12134, 2025. 2
- [65] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024. 1
- [66] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 3
- [67] Xingyi Yang and Xinchao Wang. Diffusion model as representation learner. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18938–18949,

2023. 3

- [68] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022. 2
- [69] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023. 2
- [70] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agentuser interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024. 1
- [71] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pages 69–85. Springer, 2016. 6
- [72] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 6
- [73] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 3
- [74] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025. 3
- [75] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey. IEEE

- transactions on pattern analysis and machine intelligence, 46(8):5625–5644, 2024. 1
- [76] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2), 2023. 1
- [77] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. 3
- [78] Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022. 2
- [79] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. 3

