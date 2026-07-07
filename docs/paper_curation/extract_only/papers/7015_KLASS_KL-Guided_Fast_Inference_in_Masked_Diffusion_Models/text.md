# arXiv:2511.05664v2[cs.LG]5Mar2026

## KLASS: KL-Guided Fast Inference in Masked Diffusion Models

Seo Hyun Kim1∗ Sunwoo Hong1∗ Hojung Jung1 Youngrok Park1 Se-Young Yun1 1KAIST AI {shkimsally, sunwoo, yunseyoung}@kaist.ac.kr

### Abstract

Masked diffusion models have demonstrated competitive results on various tasks including language generation. However, due to its iterative refinement process, the inference is often bottlenecked by slow and static sampling speed. To overcome this problem, we introduce ‘KL-Adaptive Stability Sampling’ (KLASS), a fast yet effective sampling method that exploits token-level KL divergence to identify stable, high-confidence predictions. By unmasking multiple tokens in each iteration without any additional model training, our approach speeds up generation significantly while maintaining sample quality. On reasoning benchmarks, KLASS achieves up to 2.78× wall-clock speedups while improving performance over standard greedy decoding, attaining state-of-the-art results among diffusion-based samplers. We further validate KLASS across diverse domains, including text, image, and molecular generation, showing its effectiveness as a broadly applicable sampler across different models. Our code is available at https://github.com/shkim0116/KLASS.

### 1 Introduction

Masked diffusion models [1, 28, 34, 38] have attracted growing attention for their ability to model joint distribution of sequences by iteratively refining samples from partially masked sequences to clean data, achieving competitive performance on complex language tasks [27], image generation [7], biological sequences [25, 34], and planning algorithms [50, 51].

Despite recent successes, these models are often restricted by slow and static sampling strategies such as Top-k or stochastic sampling, where only a limited number of high-confidence tokens are unmasked at each step. As a result, the generation process can become inefficient and prone to local suboptimalities, thus constraining the practical applicability of masked diffusion approaches.

Several works investigate efficient samplers by caching the logits if no tokens are unmasked at the specific timestep [34] or design a specific scheduler to unmask one token at a time [56]. Another natural solution might be to incorporate an additional “planner” or auxiliary distribution to guide sampling [48, 55]. However, doing so typically incurs substantial computational overhead, increases inference latency, and can lead to difficulty aligning the planner’s distribution with the base model’s learned distribution. Instead, our goal is to develop a lightweight yet effective sampling method that remains within the model’s own capabilities, yielding speedups in generation while simultaneously improving or maintaining overall accuracy.

To address these challenges, we propose ‘KL-Adaptive Stability Sampling’ (KLASS), an adaptive sampling strategy that leverages the diffusion model’s own feedback to guide unmasking. Unlike previous approaches that rely on fixed schedules (i.e., a predetermined number of tokens unmasked at each timestep), our method adapts to the evolving confidence of the model during generation. We accelerate inference by identifying stable tokens as low-risk candidates for early unmasking. To

∗Equal contribution

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

✘Top-k Confidence

LLaDA

DREAM

Correct

Correct

0.10

0.10

... Therefore, the number of cars that drove through in the first 15 minutes is:\n \\[\n 25 - 20 = 10\n \\]

Incorrect

Incorrect

KLDivergence

KLDivergence

Therefore, the number of cars that drove through the traffic jam in the first 15 minutes is \\(\\boxed{ 10 }\\).

0.08

0.08

###### ✓

conf: 0.9241, kl: 0.4517

0.06

0.06

KLASS

|… Therefore, the number of cars that drove through in the first 15 minutes is:\n \\[\n 25 - 20 = 5\n \\] Therefore, the number of cars that drove through the traffic jam in the first 15 minutes is \\(\\boxed{ 5 }\\).|
|---|

0.04

0.04

GSM8K MATH HumanEval MBPP

GSM8K MATH HumanEval MBPP

conf: 0.7587, kl: 0.0193

(a) Case study comparing Top-k confidence and KLASS solutions.

(b) Average KL divergence of tokens at unmasking for correct and incorrect predictions on LLaDA and DREAM.

- Figure 1: KL divergence as a strong indicator of solution correctness. (a) The Top-k method selects an incorrect solution despite high confidence, whereas KLASS identifies the correct solution, which exhibits a significantly lower KL divergence. (b) KL divergence distributions for the LLaDA and DREAM models show that correct predictions consistently have lower KL divergence than incorrect ones across all datasets.

quantify stability, we track the token-level Kullback-Leibler (KL) divergence between conditional distributions at consecutive timesteps. Tokens are unmasked when their distributions remain similar (KL below a threshold) and are predicted with high confidence (probability exceeding a confidence threshold). This dynamic allocation of unmasking tokens results in significant acceleration of generation speed while maintaining sample quality by avoiding premature or suboptimal token unmasking without additional model training or extra memory burden.

We empirically validate our method on challenging reasoning benchmarks, including GSM8K, MATH, HumanEval, and MBPP. We show that applying KLASS with large-scale masked diffusion models not only halves the number of sampling steps compared to standard greedy or Top-k decoding [19], but also achieves higher accuracy, achieving state-of-the-art results compared to other diffusion samplers. Figure 1a presents a comparison between solutions generated by Top-k confidence and KLASS sampler. KLASS successfully identifies the correct token with lower KL, whereas Top-k confidence tends to unmask incorrect tokens even with higher confidence. Furthermore, our experiment on plain text generation also proves the effectiveness of our method which results in reduced perplexity while maintaining entropy, thereby mitigating the inefficiencies inherent in conventional sampling. We further show that our sampler works in other modalities, including images and molecules.

Overall, our proposed sampler for masked diffusion models is both simple and practical, harnessing the latent potential of the base diffusion model itself, rather than relying on complex external planners. By strategically identifying stable tokens at each iteration, the algorithm accelerates generation and fosters more robust coverage of viable token candidates. We believe this work provides a practical and scalable way for large-scale masked diffusion models, particularly where reliable and efficient generation is essential, such as complex reasoning tasks.

We summarize our main contributions below:

- • We propose KLASS, a training-free sampler that leverages the model’s internal dynamics in terms of token level KL divergence and confidence without requiring external planners.
- • We achieve up to 2.78× faster sampling by more than halving the number of diffusion steps through parallel unmasking of stable tokens.
- • We provide comprehensive empirical validation, showing improved quality on reasoning benchmarks across math and code generation, text generation, image synthesis, and molecular generation.

### 2 Related Works

Discrete diffusion models D3PM [1] investigate how forward and backward processes can be constructed in discrete state spaces which is analogous to the continuous diffusion models [18, 40]. [6] leverage continuous time Markov chain (CTMC) theory to formulate the forward-backward process of discrete diffusion models with providing negative ELBO in continuous time limit as an objective. Following the success of denoising score matching [41], Lou et al. [25], Meng et al. [26] suggest

discrete score matching loss by defining Stein score in discrete space. Ou et al. [28], Sahoo et al. [34], Shi et al. [38] further shows that simplified version of masked diffusion model can significantly boost the performance of diffusion models closing the performance gap with AR models in language domains. Recently, LLaDA [27] demonstrates scaling law of discrete diffusion models in language domain and further shows reasoning abilities.

Discrete diffusion samplers Generating a text from language diffusion models involves iteratively refining a sequence from a noisy or masked state. Ancestral Sampling [25, 34] starts from a fully masked sequence and iteratively applies the learned reverse denoising process over a series of discrete timesteps to produce a clean sequence. SUBS parametrization [34] of the reverse step dictates how model predictions are used to unmask tokens, often by ensuring that already revealed tokens remain unchanged. To improve sample quality, ReMDM [43] adopts remasking strategies, where some newly predicted tokens are reset to a mask based on confidence or timestep.

Accelerated Sampling of Discrete diffusion models The iterative nature of ancestral sampling can result in high latency due to the large number of sequential steps. Consequently, much research has focused on reducing the number of function evaluations (NFEs) in diffusion models. Deschenaux and Gulcehre [11], Hayakawa et al. [15] leverage distillation methods to train the model with reduced NFEs in analogous to fast sampling of continuous diffusion models [35, 41, 53]. Ren et al. [33] improve discrete diffusion solvers by considering second-order numerical solver in CTMC framework. Zheng et al. [56] propose a First-Hitting Sampler (FHS) to skip the unnecessary timesteps and unmask one token at a time. Most of the existing samplers of masked diffusion models, however, resort to additional training or rely on other models (i.e., planners) to choose unmasking tokens at each timestep [21, 24, 29]. This could help avoiding suboptimal token selection but with considerable computational overhead and may fail to be aligned with the model’s intrinsic capability.

Recent training-free strategies for accelerating masked diffusion language models have emerged concurrently, with several works exploring heuristics based on model certainty to guide this process. Fast-dLLM [47] and Dimple [54] use confidence-aware decoding, SlowFast Sampling [45] alternates decoding stages based on certainty, convergence, and position principles, EB-Sampler [4] unmasks multiple tokens based on entropy bounds, and Prophet [23] uses the Top-2 confidence gap. While these concurrent approaches validate the utility of heuristics largely based on certainty, we empirically demonstrate that this signal alone is insufficient. To ensure tokens are not unmasked prematurely, we propose a novel method that utilizes KL divergence to identify stable tokens for parallel decoding.

### 3 Preliminaries

#### 3.1 Masked diffusion models

In masked diffusion models, one requires an additional mask index m for each tokens and forward process is defined by following absorbing process [1]:

q(zt|x) = Cat(zt;αtx + (1 − αt)m), (1)

where αt is predefined schedule, monotonically decreasing in t. Then one can analytically obtain posterior distribution as:

Cat(zs;zt) if zt ̸= m, Cat zs; (1−α

(2)

q(zs | zt,x) =

s)m+(αs−αt)x

1−αt if zt = m.

The goal of the masked diffusion model is to learn this reverse process by parameterizing the posterior (Eq. 2) by a neural network with pθ(zs|zt) := q(zs|zt,µθ(zt,t)).

In simplified masked diffusion models [28, 34, 38], learning objective can be simplified by parameterizing the models to focus on estimating only masked tokens while maintaining unmasked tokens throughout the generation.

Then the learning objective is to minimize Negative ELBO (NELBO) whose continuous form is the following:

Confidence: = 0.412 < 0.9 ✗ KL: = 0.163 > 0.01 ✗

𝐷 ( ∥ )

| | | | | | | | | |2| | |
|---|---|---|---|---|---|---|---|---|---|---|---|

t

|In|total| | |takes| |+| | |1| | |
|---|---|---|---|---|---|---|---|---|---|---|---|

- t-1
- t-2
- t-3

𝐷 ( ∥ )

𝑚𝑎𝑥

t-3 t-2

t-3

t-2 t-1

t-2

t-1

Both Not Satisfied

KL Not Satisfied

Both Satisfied

236 237 238 239 240 241 242 243 244 245 246 247

: Token probability distribution, : Masked, : Unmasked, : Fixed

𝑚𝑎𝑥

𝐷 ( ∥ )

𝑚𝑎𝑥

Unmask

Mask

t-1 t Mask

Figure 2: Illustration of parallel decoding with KLASS. Tokens are unmasked when they meet the two criteria: high predictive confidence and a stable probability distribution. Stability is measured by a low KL divergence between consecutive steps (illustrated with history length of 1 for simplicity). On the right it shows the sampling process for position 245: it remains masked due to low confidence or high KL score, and is unmasked when both conditions are satisfied.

L∞ = Ex∼q

0,zt∼qt(zt|x)

1

- 0

αt′

- 1 − αt

[δx,m x · log µθ (zt,t)]. (3)

Here, q0 denotes data distribution and αt′ is the derivative of noise schedule αt in time. In this continuous time framework, [34] further proves that above objective is invariant of noise schedule αt.

The above can be generalized to sequence-level of token length L modeling as follows.

L(∞L) =

1

- 0

αt′

- 1 − αt

Ex∼q

0,zt∼qt(zt|x)

   l:z(tl)=m

x(l) · log µ(θl)(zt,t)

   dt. (4)

3.2 Inference via Ancestral Sampling At inference, we discretize t ∈ [0,1] into times {tT > ··· > t1 ≈ 0}, initializing xt

T

= [mask]L. We then sample backward:

xt

i−1 ∼ pθ xt

i−1 | xt

i

, i = T,...,1.

In simplified MDM, unmasked tokens remain fixed and masked tokens are drawn from the model’s prediction. After T steps, we obtain a complete sequence xt

0

. We provide additional analysis of other sampling strategies in Appendix C.

- 4 Method

Confidence: = 0.928 ≥ 0.9 ✓ KL: = 0.641 > 0.01 ✗

|In|total|,|it|takes|2|+|1| |3| |.|
|---|---|---|---|---|---|---|---|---|---|---|---|

|In|total|,|it|takes|2|+|1|=|3|bolts|.|
|---|---|---|---|---|---|---|---|---|---|---|---|

Confidence: = 0.957 ≥ 0.9 ✓ KL: = 0.004 ≤ 0.01 ✓

#### 4.1 Defining Confidence Score and KL Score

KLASS aims to identify which tokens are stable enough to be unmasked at each step of the inference process, which we index by timesteps t = T,...,1. To guide this selection, we introduce two key metrics: a confidence score to measure the model’s certainty on a given token and a KL score to measure the temporal consistency of its predictions.

- Definition 4.1. (Confidence score) Denoting pit as the categorical distribution predicted by the

diffusion model at timestep t for token position i, we define the confidence confit to be the largest value of the probability function among vocabulary space V (v ∈ V ):

confit = max

pit(v). (5)

v

Intuitively, a higher confidence score indicates the model is more certain about estimating the current token, which increases the chance that the model’s estimate for that token is correct.

- Definition 4.2. (KL score) We define KL score dit of the token position i at timestep t as the Kullback-Leibler divergence between previous estimates and current estimates of the given token:

dit = DKL pit ∥pit+1 , (6)

where we denote pit,pit+1 be the probability distribution of the model estimates of token index i at time t and at time t + 1, respectively.

KL score should be low only when the model’s estimate is consistent throughout the reverse diffusion process, which implies the estimated token is more reliable.

To empirically demonstrate how KL score behaves in practical scenario, we first generate samples for a variety of math and programming reasoning benchmarks. As shown in Figure 1b, correct samples consistently exhibit significantly lower KL scores than incorrect ones, for all models and datasets. This observation motivates our use of KL scores as a guiding signal in the sampling algorithm of masked diffusion models, which we formally introduce in the next section.

#### 4.2 KLASS: KL-Adaptive Stability Sampling

We introduce ‘KL-Adaptive Stability Sampling’ (KLASS), a novel sampling algorithm for masked diffusion models. As illustrated in Figure 2, KLASS leverages confidence score and KL score during the unmasking process of the masked diffusion models (Eq. 2), by selectively choosing unmasking tokens that have low KL score and high confidence score.

Stable-token selection. To effectively set the standard using both KL and confidence score, we propose stable-token selection in the following way: Given a history length n, a KL threshold ϵKL, and a confidence threshold τ, we select the set of stable tokens at step t as,

##### ∧ confit > τ

St = i ∀k ∈ {1,...,n} DKL pit+k−1 ∥pit+k < ϵKL

. (7)

high confidence

all recent KL’s below threshold

Unmasking rule. KLASS adaptively chooses which tokens to unmask at given timestep with above defined stable index (Eq. 7). At each diffusion step t, we apply

unmask token at position i, i ∈ St, otherwise, unmask the Top-u positions by confit, St = ∅,

xit =

(8)

where u is a fixed fallback unmasking count. We provide a pseudocode of our algorithm with further analysis in Appendix B.

### 5 Theoretical Rationale

We provide a theoretical perspective on why using KL divergence can improve sample quality. We show that, for a well-trained model, a token that is predicted as incorrect at the current step cannot remain uniformly stable as the context is progressively resolved.

- Definition 5.1. For each context c (instantiation of variables outside Xi), let C(c) be the nonempty set of task–correct conditionals. Let C := {µ : µ(· | c) ∈ C(c) ∀c}. We say pθ is a conditional δ–approximation to the task if

inf

π∈C

sup

c

TV pθ(· | c), π(· | c) ≤ δ.

- Definition 5.2. Fix i. Let x⋆i be optimal under π(· | c⋆) at near-optimal context c⋆. Let x†i ̸= x⋆i be suboptimal. Assume a true margin γ > 0 at c⋆: π(x⋆i | c⋆) ≥ π(x†i | c⋆) + γ. Assume the model currently prefers x†i at cM by margin β ≥ 0: pθ(x†i | cM) ≥ pθ(x⋆i | cM) + β. Proposition 5.3. Suppose pθ is a conditional δ-approximation of π. For any context path cM → cM−1→ ···→ c0 (changing only variables outside Xi) ending at c0 = c⋆, let Pt := pθ(· | ct) and ∆ := 12(β + γ − 2δ)+. Then

M−1

2∆2 M2

1 M

TV(PM,P0) ≥ ∆,

KL Pt ∥Pt+1 ≥

.

t=0

Proof. The proof is in Appendix A.

| |
|---|

- Table 1: Performance and sampling steps on reasoning benchmarks for different diffusion samplers.

MATH GSM8K HumanEval MBPP Acc↑ Steps↓ Acc↑ Steps↓ Acc↑ Steps↓ Acc↑ Steps↓

Method Parallel

#### LLaDA

- Top-1 ✗ 31.4 256 75.13 256 39.63 256 46.69 256 Random ✗ 26.2 256 67.10 256 20.21 256 29.18 256
- Top-2 ✓ 29.6 128 72.40 128 33.54 128 37.74 128 Confidence ✓ 31.6 96.46 75.21 74.35 37.80 54.41 47.08 85.20 KL divergence ✓ 32.6 172.21 74.52 155.88 40.24 111.93 45.53 150.47 KLASS (ours) ✓ 33.8 128.62 76.50 98.57 40.85 91.98 47.86 119.59

#### Dream

- Top-1 ✗ 37.97 256 79.55 256 58.53 256 63.81 256 Random ✗ 18.73 256 37.35 256 18.09 256 28.14 256
- Top-2 ✓ 33.60 128 71.69 128 42.88 128 47.08 128 Confidence ✓ 41.80 95.10 73.67 74.81 50.00 52.47 57.59 72.49 KL divergence ✓ 41.27 162.49 76.70 150.02 59.35 73.94 62.65 108.15 KLASS (ours) ✓ 43.20 149.72 79.43 155.67 59.35 74.88 64.59 111.24

Remarks. A token that is wrong at c0 but correct at c⋆ must be dynamically unstable somewhere along the path: its average per-step KL is bounded away from 0. In essence, incorrect tokens cannot remain dynamically stable. Accordingly, KLASS delays unmasking until tokens exhibit dynamic stability thereby improving generation quality.

### 6 Experiments

To show effectiveness of our proposed sampler, we conduct experiments on multiple benchmarks including reasoning benchmarks with large scale models in Section 6.1, text generation in Section 6.2, along with other modalities including images in Section 6.3 and molecules in Section 6.4. We also present ablation studies in Section 6.5 and analyze computational overhead in Section 6.6.

#### 6.1 Reasoning tasks

Experimental setup We evaluate on four reasoning benchmarks: GSM8K [10] and MATH500 [16] for math, and HumanEval [9] and MBPP-sanitized [2] for code synthesis. We use two instructiontuned models, LLaDA 8B Instruct [27] and Dream 7B Instruct [52]. For both models we set the generation length to 256 tokens, with LLaDA using a block size of 64. The generation temperature is set to 0 for LLaDA and 0.2 for Dream. We report both the number of sampling steps and the pass@1 accuracy. The maximum inference timestep is set to 256. In KLASS, we compute per-token KL divergence over a history length of n = 2, and apply KL thresholds ranging from 0.001 to 0.01 and confidence thresholds from 0.5 to 0.9. Full configuration details and a lightweight guideline for hyperparameter selection are provided in Appendix D.1.2.

Baselines We compare KLASS against baselines across two categories. The first is sequential unmasking (single-token), which includes: (i) Top-1 sampling, selecting the highest-confidence token at each step [7]; and (ii) random sampling [1]. The second category is parallel unmasking, which accelerates generation by revealing multiple tokens per step: (iii) Top-2 sampling, decoding the two highest-confidence tokens per step to halve the total number of steps; (iv) confidence-threshold sampling, unmasking all tokens with a predicted probability over 0.9; and (v) KL-threshold sampling, unmasking all tokens with a KL divergence under 0.001, using a history length n = 2 as in KLASS.

Results As shown in Table 1, KLASS consistently improves accuracy across most tasks compared to the standard greedy decoding (Top-1) baseline. It demonstrates robust generalization for both LLaDA and Dream models across math and code synthesis benchmarks. Beyond accuracy, KLASS

- Table 2: Generative perplexity, MAUVE, and entropy on unconditional text generation sampled with 512 steps.

Method MAUVE↑ LLaMA2↓ LLaMA3↓ GPT-2↓ Entropy↑

*Data 1.0 7.0 9.4 14.8 5.44 AR 0.855 10.97 15.12 12.07 5.21 SEDD 0.037 53.09 109.60 105.40 5.62 D3PM 0.022 41.82 72.85 76.70 5.40 MDLM 0.115 30.88 54.15 51.78 5.46 KLASS (Ours) 0.179 26.94 49.19 45.50 5.43

- Table 3: Generative FID and IS on MMaDA with different step sizes.

Table 4: Molecular generation results on the QM9 dataset conditioned on different molecular properties.

Method Steps FID ↓ IS ↑

Confidence 16 34.48 75.72 KLASS (ours) 16 30.48 93.07

Confidence 32 36.45 72.40 KLASS (ours) 32 32.00 89.17

Method Property Reward↑ NFEs↓

MDLM QED 0.526 32.0 KLASS (ours) QED 0.546 18.8

MDLM Ring count 4.123 32.0 KLASS (ours) Ring count 4.258 24.4

is also highly efficient. It reduces sampling steps by 40–70% relative to the full 256-step schedule, yielding wall-clock speedups of up to 2.78× (Appendix D.1.3). Unlike other acceleration strategies such as halving steps with a confidence-based Top-2 method, which degrades accuracy, KLASS improves accuracy with fewer steps overall. KLASS achieves a superior balance between speed and accuracy compared to methods that rely on a single threshold for either confidence or KL score alone. This proves that the effectiveness of KLASS comes from its novel approach of combining token confidence with KL-divergence trajectories.

#### 6.2 Text generation

Experimental setup We evaluate KLASS on Masked Diffusion Language Model (MDLM) [34] pre-trained on the OpenWebText corpus [13]. As baselines, we include (i) the original autoregressive sampler, (ii) SEDD [25], and (iii) two variants of MDLM: one parameterized with SUBS (the standard 512-step sampler) and one parameterized with D3PM [3] (the absorbing variant). For all diffusion-based methods, we generate 1,000 sequences of length 1,024 tokens under a fixed 512-step schedule, with nucleus Top-p filtering at p = 0.9, history length n = 2, KL threshold ϵKL = 1e − 4, and confidence threshold τ = 0.57.

Evaluation We report generative perplexity by exponentiating the average token-level negative loglikelihood under three oracle models: LLaMA2 (7B) [42], LLaMA3 (8B) [14], and GPT-2 [30]. We measure Shannon entropy of the predicted token distributions and compute MAUVE by comparing our 1,000 generated samples to 1,000 held-out segments from the OpenWebText. Baseline (*Data) results are given from the corresponding literatures [43, 48].

- Results Table 2 shows that KLASS substantially improves generative quality over existing discrete diffusion samplers. Our method higher MAUVE and lower perplexity across all oracle models while maintaining comparable entropy. These results highlight that stability-aware multi-token unmasking guided by KLASS leads to more coherent and fluent text generation, all without any additional model training. We provide experimental details in Appendix D.2.

#### 6.3 Image generation

Experimental setup We evaluate KLASS on the MMaDA (Multimodal Large Diffusion Language Models) [49], a multimodal diffusion foundation model. We compare two samplers: (i) the standard confidence-based sampler used by MMaDA, and (ii) our proposed KLASS. For each method, we

|[Figure 1]| |
|---|---|
| | |
| | |
| | |
| | |

[Figure 2]

none

25.4 29.2 30.6 30.8 31.6

32

0.02

27.8 31.0 31.4 31.6 30.8

Accuracy(%)

KLThreshold

30

0.015

- 30.4 33.2 33.0 31.0 30.8
- 31.6 33.8 32.4 30.8 30.6
- 32.6 31.4 31.6 30.8 30.2 26

28

0.01

0.005

0.5 0.6 0.7 0.8 0.9 Confidence Threshold

(a) LLaDA

|[Figure 3]| |
|---|---|
| | |
| | |
| | |
| | |

[Figure 4]

none 0.02 0.015

22.2 25.0 25.0 36.8 41.8

40

29.2 33.2 38.6 39.4 40.2

Accuracy(%)

KLThreshold

35

32.4 33.4 38.4 39.2 40.4

0.01 0.005 0.001

- 31.0 29.6 37.8 39.2 42.0
- 32.6 35.6 37.8 40.6 43.2

30

25

40.4 39.6 41.4 41.8 41.6

0.5 0.6 0.7 0.8 0.9 Confidence Threshold

(b) Dream

Figure 3: KL Effect Across Confidence Levels on MATH.

generate 10,000 images conditioned on labels drawn uniformly from the 1,000 ImageNet classes, using 16 and 32 step decoding schedules. KLASS is configured with history length n = 1, KL divergence threshold ϵKL = 0.3, and confidence threshold τ = 0.1.

Evaluation We assess sample fidelity using two widely adopted metrics. First, we compute Fréchet Inception Distance (FID) [17] between our 10,000 generated samples and the ImageNet validation set, using the official Inception v3 implementation. Second, we measure Inception Score (IS) [36] on the same samples with the standard protocol.

- Results Table 3 shows that KLASS improves image quality on MMaDA over the standard confidence-based sampler. Across both decoding schedules, KLASS yielding lower FID and higher IS. The trend holds under the same decoding schedules and fairness controls, indicating that KLASS improves fidelity and class-consistency without modifying the backbone or adding auxiliary guidance. We provide experimental details in Appendix D.3.

#### 6.4 Molecular generation

Experimental setup We use QM9 [31], which contains molecules with up to nine heavy atoms, represented in SMILES [46]. For models we follow the training recipe of [37] to train seperate models conditioned on drug-likeness (QED) [5] and number of rings using classifier-free training of masked diffusion models.

Evaluation We test KLASS on conditional generation of small molecules using CFG guidance. Specifically, we aim to generate molecules with higher score of QED or maximizing ring counts while fixing the CFG strength for fair comparison. We generate 1,024 samples for each task and provide average value of number of function evaluation (NFEs). Further details of the experimental setups are provided in Appendix D.4.

Results The result shows that KLASS effectively reduces the total sampling steps while maintaining target reward in the conditional generation scenario for both target reward (QED and Ring count). We provide further experimental results in this setup in Appendix D.4.

#### 6.5 Ablation Studies

Effect of confidence and KL score thresholds Our evaluation of different confidence and KL thresholds on the MATH dataset reveals that combining both is essential for optimal performance. As shown in Figure 3, applying the KL threshold consistently enhances accuracy across all confidence levels compared to relying on a confidence threshold alone (‘none’ row). This synergistic relationship is further substantiated by Table 1, which demonstrates that using a single criterion leads to a notable reduction in accuracy.

Table 5: Comparison of single-token and parallel unmasking strategies under KLASS criteria.

Table 6: Memory and computational overhead of KL divergence per decoding step.

MATH GSM8K Acc↑ Steps↓ Acc↑ Steps↓

Unmasking

Single (conf) 31.2 256 72.86 256 Single (KL) 29.0 256 73.46 256 Parallel 33.8 128.6 76.50 98.57

Memory (MB) Time (s) Overhead Total Overhead Total

Model

LLaDA 247 18,702 0.000255 0.1218 Dream 296 18,875 0.000177 0.1275

While the optimal hyperparameter settings vary significantly between models, each model’s performance remains stable and robust around its unique optimal point. For example, LLaDA performs best with a lower confidence threshold, whereas Dream requires a higher one to achieve maximum accuracy. In both cases, however, accuracy does not degrade sharply near these values, indicating low sensitivity to minor hyperparameter adjustments. A more detailed sensitivity analysis, featuring additional tasks and a finer-grained grid of thresholds, is provided in Appendix D.5.1.

Effect of unmasking multiple tokens We evaluate whether unmasking multiple tokens per step improves LLaDA’s performance. Using KLASS, which selects tokens based on fixed thresholds, we compare parallel multi-token unmasking to two sequential variants. These variants unmask only a single token from the same stable pool satisfying the KLASS criteria: ‘Single (conf)’ unmasks the one with the highest confidence and ‘Single (KL)’ unmasks the one with the lowest KL score.

As shown in Table 5, parallel sampling of KLASS boosts both accuracy and efficiency. On MATH, it improves accuracy by up to 4.8 points while cutting sampling steps by nearly 50%. Similar trends hold on GSM8K. These results suggest that LLaDA benefits from unmasking multiple stable tokens in parallel, leading to faster and even more accurate reasoning.

#### 6.6 Analysis on Computational Overhead

The overhead of KL computation is negligible, as it is a lightweight post-processing step on existing logits that requires no additional forward pass. For the set of masked tokens Im = {i | zti = m}, we compute the KL score dit = DKL(pit∥pit+1) and cache the prior distribution. This yields a combined computational and memory overhead of O(|Im| · |V |), a linear cost that is negligible compared to the expensive matrix multiplications and multi-gigabyte footprint of the main diffusion step. Table 6 empirically supports this conclusion. We measure the overhead for LLaDA and Dream, with vocab sizes of 126,464 and 152,064, respectively, using a generation length of 256. The results show memory overheads below 1.57% of total memory and latency overheads below 0.21% per decoding step, confirming that KL computation adds only minimal cost.

### 7 Conclusion

We proposed KL-Adaptive Stability Sampling (KLASS), an efficient and adaptive sampling method for masked diffusion models that leverages token-level KL divergence and model confidence to guide the unmasking process. KLASS substantially reduces the number of sampling steps while maintaining or improving accuracy, achieving state-of-the-art performance on math and code reasoning benchmarks. Our approach is simple, requires no additional training, and generalizes well across multiple modalities, making it a practical solution for faster and more reliable generation in masked diffusion models.

For future work, one could extend this approach to discrete diffusion models with alternative noise schedules, such as the uniform or marginal prior [1]. Another direction is to evaluate the proposed sampler with larger models as they become available. We also discuss the broader impact and limitations of our work in Appendix G.

### Acknowledgments

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2024-00457882, AI Research Hub Project, 10%) and (RS-2022-II220311, Development of Goal-Oriented Reinforcement Learning Techniques for Contact-Rich Robotic Manipulation of Everyday Objects, 90%).

### References

- [1] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.
- [2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.
- [3] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces, 2023. URL https://arxiv. org/abs/2107.03006.
- [4] Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, and Brian Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking. arXiv preprint arXiv:2505.24857, 2025.
- [5] G Richard Bickerton, Gaia V Paolini, Jérémy Besnard, Sorel Muresan, and Andrew L Hopkins. Quantifying the chemical beauty of drugs. Nature chemistry, 4(2):90–98, 2012.
- [6] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.
- [7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022.
- [8] Hongrui Chen and Lexing Ying. Convergence analysis of discrete diffusion model: Exact implementation through uniformization. arXiv preprint arXiv:2402.08095, 2024.
- [9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [11] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation through time. arXiv preprint arXiv:2410.21035, 2024.
- [12] Daniel T Gillespie. A general method for numerically simulating the stochastic time evolution of coupled chemical reactions. Journal of computational physics, 22(4):403–434, 1976.
- [13] Aaron Gokaslan and Vanya Cohen. OpenWebText corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019. Accessed: 2025-05-16.
- [14] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius,

Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh,

Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

- [15] Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. arXiv preprint arXiv:2410.08709, 2024.
- [16] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. URL https://arxiv.org/abs/1706.08500.
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [19] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.
- [20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35: 26565–26577, 2022.
- [21] Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked diffusions. arXiv preprint arXiv:2502.06768, 2025.
- [22] Bin Lei, Yi Zhang, Shan Zuo, Ali Payani, and Caiwen Ding. Macm: Utilizing a multi-agent system for condition mining in solving complex mathematical problems. arXiv preprint arXiv:2404.04735, 2024.
- [23] Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, and Shiwei Liu. Diffusion language models know the answer before decoding. arXiv preprint arXiv:2508.19982, 2025.

- [24] Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stärk, Yilun Xu, Tommi Jaakkola, and Rafael Gómez-Bombarelli. Think while you generate: Discrete diffusion with planned denoising. arXiv preprint arXiv:2410.06264, 2024.
- [25] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.
- [26] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022.
- [27] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.
- [28] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.
- [29] Fred Zhangzhi Peng, Zachary Bezemek, Sawan Patel, Jarrid Rector-Brooks, Sherwood Yao, Alexander Tong, and Pranam Chatterjee. Path planning for masked diffusion model sampling. arXiv preprint arXiv:2502.03540, 2025.
- [30] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.
- [31] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7, 2014.
- [32] Yinuo Ren, Haoxuan Chen, Grant M Rotskoff, and Lexing Ying. How discrete and continuous diffusion meet: Comprehensive analysis of discrete diffusion models via a stochastic integral framework. arXiv preprint arXiv:2410.03601, 2024.
- [33] Yinuo Ren, Haoxuan Chen, Yuchen Zhu, Wei Guo, Yongxin Chen, Grant M Rotskoff, Molei Tao, and Lexing Ying. Fast solvers for discrete diffusion models: Theory and applications of high-order algorithms. arXiv preprint arXiv:2502.00234, 2025.
- [34] Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.
- [35] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [36] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.
- [37] Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dallatorre, Bernardo P de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. arXiv preprint arXiv:2412.10193, 2024.
- [38] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024.
- [39] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.
- [40] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

- [41] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288.
- [43] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. arXiv preprint arXiv:2503.00307, 2025.
- [44] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents. In Forty-first International Conference on Machine Learning, 2024.
- [45] Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, and Linfeng Zhang. Accelerating diffusion large language models with slowfast: The three golden principles. arXiv preprint arXiv:2506.10848, 2025.
- [46] David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28

(1):31–36, 1988.

- [47] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.
- [48] Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, and Arash Vahdat. Energy-based diffusion language models for text generation. arXiv preprint arXiv:2410.21357, 2024.
- [49] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models, 2025. URL https://arxiv.org/abs/ 2505.15809.
- [50] Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Beyond autoregression: Discrete diffusion for complex reasoning and planning. arXiv preprint arXiv:2410.14157, 2024.
- [51] Jiacheng Ye, Zhenyu Wu, Jiahui Gao, Zhiyong Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Implicit search via discrete diffusion: A study on chess. arXiv preprint arXiv:2502.19805, 2025.
- [52] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b, 2025. URL https://hkunlp.github.io/blog/2025/dream.
- [53] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.
- [54] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.

- [55] Yixiu Zhao, Jiaxin Shi, Feng Chen, Shaul Druckmann, Lester Mackey, and Scott Linderman. Informed correctors for discrete diffusion models. arXiv preprint arXiv:2407.21243, 2024.
- [56] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024.

### Table of contents

- 1 Introduction 1
- 2 Related Works 2
- 3 Preliminaries 3

- 3.1 Masked diffusion models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3.2 Inference via Ancestral Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 4 Method 4

- 4.1 Defining Confidence Score and KL Score . . . . . . . . . . . . . . . . . . . . . . 4
- 4.2 KLASS: KL-Adaptive Stability Sampling . . . . . . . . . . . . . . . . . . . . . . 5

- 5 Theoretical Rationale 5
- 6 Experiments 6

- 6.1 Reasoning tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 6.2 Text generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 6.3 Image generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 6.4 Molecular generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 6.5 Ablation Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 6.6 Analysis on Computational Overhead . . . . . . . . . . . . . . . . . . . . . . . . 9

- 7 Conclusion 9

- A Theoretical Proofs 18
- B Pseudo Code 18
- C Further Prior Works 18
- D Experiment details and additional results 20

- D.1 Reasoning tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- D.1.1 Experiment details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.1.2 Hyperparameter Selection Guideline . . . . . . . . . . . . . . . . . . . . . 20
- D.1.3 Wall-clock time comparison . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.1.4 Statistical significance of Dream 7B Instruct results . . . . . . . . . . . . . 21

- D.2 Text generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- D.2.1 Experiment details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

D.3 Image generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D.3.1 Experiment details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

D.4 Molecules . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D.4.1 Experiment details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D.5 Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- D.5.1 Hyperparameter Sensitivity . . . . . . . . . . . . . . . . . . . . . . . . . 23
- D.5.2 Effect of history length . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- D.5.3 Effect of temperature . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- E Comparison to other diffusion samplers 23 E.1 Performance on reasoning tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- F Examples of generated samples 25
- G Limitation & Broader impact 26

- G.1 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- G.2 Broader impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- A Theoretical Proofs In this section, we provide the proof for Section 5.

- Definition A.1. For each context c (instantiation of variables outside Xi), let C(c) be the nonempty set of task–correct conditionals. Let C := {µ : µ(· | c) ∈ C(c) ∀c}. We say pθ is a conditional δ–approximation to the task if

inf

π∈C

sup

c

TV pθ(· | c), π(· | c) ≤ δ.

- Definition A.2. Fix i. Let x⋆i be optimal under π(· | c⋆) at near-optimal context c⋆. Let x†i ̸= x⋆i be suboptimal. Assume a true margin γ > 0 at c⋆: π(x⋆i | c⋆) ≥ π(x†i | c⋆) + γ. Assume the model currently prefers x†i at cM by margin β ≥ 0: pθ(x†i | cM) ≥ pθ(x⋆i | cM) + β.

Proposition A.3. Suppose pθ is a conditional δ-approximation of π. For any context path cM → cM−1→ ···→ c0 (changing only variables outside Xi) ending at c0 = c⋆, let Pt := pθ(· | ct) and ∆ := 12(β + γ − 2δ)+. Then

TV(PM,P0) ≥ ∆,

1 M

M−1

t=0

KL Pt ∥Pt+1 ≥

2∆2 M2

.

Proof. Let f = 1{xi = x†i} − 1{xi = x⋆i } so ∥f∥∞ ≤ 1. Then

2TV(PM,P0) ≥ EP

0

[f] − EP

M

[f] = (p†θ(cM) − p⋆θ(cM)) − (p†θ(c⋆) − p⋆θ(c⋆)) . By the margin assumptions (Definition A.2) and the conditional δ–approximation (Definition A.1), p†θ(cM) − p⋆θ(cM) ≥ β and p⋆θ(c⋆) − p†θ(c⋆) ≥ γ − 2δ. Hence 2TV(PM,P0) ≥ β + γ − 2δ, which implies TV(PM,P0) ≥ ∆. By the triangle inequality in total variation,

M−1

t=0

TV(Pt+1,Pt) ≥ TV(PM,P0) ≥ ∆.

Let Tt := TV(Pt+1,Pt). Pinsker’s inequality gives KL(Pt∥Pt+1) ≥ 2Tt2 for each t (since TV (Pt+1,Pt) = TV (Pt+1,Pt)). Averaging and applying Cauchy–Schwarz,

1 M

M−1

t=0

KL(Pt∥Pt+1) ≥

2 M

M−1

t=0

Tt2 ≥

2 M ·

M−1 t=0 Tt 2

M ≥

2∆2 M2

.

| |
|---|

- B Pseudo Code

We provide the pseudo code for our KLASS algorithm, implementing the unmasking rule described in Section 4.2.

- C Further Prior Works

In this section, we review several approaches for sampling from discrete diffusion models described in prior work.

Ancestral sampling Generation proceeds by discretizing the reverse diffusion time-interval [0,1] into

0 = t0 < t1 < ··· < tT = 1. To sample a sequence of length L, one initializes

##### z1:(TL) = [mask]L,

#### Algorithm 1: KL-Adaptive Stability Sampling (KLASS)

Input: model M, total steps T, sequence length L, confidence threshold τ, KL

threshold ϵ, history window H, fallback count u Output: Generated sequence x

- 1 Initialize x ← [ MASK1:L], Pprev ← 0, KLbuf ← 0
- 2 for t ← 1 to T do

- 3 ℓ ← M(x).logits
- 4 P ← softmax(ℓ)
- 5 c ← max(P) // Get per-token confidence scores
- 6 δ ← DKL(P ∥ Pprev) // Get per-token KL scores // Update KL history buffer
- 7 KLbuf ← roll(KLbuf, shift = −1)
- 8 KLbuf[: H] ← δ
- 9 Pprev ← P // Identify stable tokens
- 10 stable_kl ← ∀(KLbuf < ϵ) // Check all history
- 11 high_conf ← (c > τ)
- 12 is_masked ← isMask(x)
- 13 ready ← stable_kl ∧ high_conf ∧ is_masked
- 14 if any(ready) then

- 15 unmask_at_indices(x, P, ready)
- 16 else // Fallback: unmask top-u tokens

- 17 scores ← c · is_masked // Zero out unmasked tokens
- 18 U ← topk_indices(scores, u)
- 19 unmask_at_indices(x, P, U)

- 20 Return x

and for i = T,T − 1,...,1 draws each coordinate independently:

δ zℓ(i) , zℓ(i) ̸= mask, pθ zℓ | z1:(iL) , zℓ(i) = mask,

zℓ(i−1) ∼

ℓ = 1,...,L.

Because unmasked tokens remain unchanged, if at step i no new tokens are decoded then z1:(iL−1) = z1:(iL) , and—when the denoiser µθ is time-invariant—its output at ti can be reused at ti−1, skipping that network evaluation [28, 34]. Models whose µθ depends on t (e.g. SEDD [25]) must recompute at every ti and cannot exploit this caching [25, 34, 38].

Exact simulation Exact simulation of the reverse CTMC in absorbing masked diffusion is achieved via uniformization: one bounds the time-varying generator Q(t) by λ, samples

iid∼ Unif(0,T),

M ∼ Poisson(λT), {τi}Mi=1

and at each τi transitions from state x to y with probability Qx,y(τi)/λ, preserving the exact law of the reversed path [8]. While unbiased, as the chain nears absorption the number of proposals—and hence cost—can grow large.

Alternatively, the first-hitting sampler of Zheng et al. [56] draws each unmasking time without discretization error: when n tokens remain masked, one samples

τn−1 = α−1 1 − u1n/n [1 − α(τn)] , un ∼ Unif(0,1),

then un-masks exactly one token (chosen uniformly among the n) according to the model’s conditional distribution. This procedure is unbiased but incurs O(L) sequential events, making runtime scale with sequence length [56].

τ-leaping Tau-leaping discretizes the reverse CTMC into fixed intervals of length τ, holding all jump rates constant and applying transitions in parallel. Let Rtθ(x,x′) be the learned rate from x to x′ at time t. Over [t − τ,t] one draws

Kx→x′ ∼ Poisson τ Rtθ(x,x′) ,

and updates

t→x′ ex′ − ex

Kx

xt−τ = xt +

.

t

x′̸=xt

Under mild regularity, the global weak error scales as O(τ), recovering Gillespie’s exact algorithm when τ → 0 [12]. Ren et al. derive a KL divergence bound

DKL Lτ-leap ∥ pdata ≤ O(τ T) + O(M T) + O e−cT/logD ,

showing modest τ suffices even in high dimensions [32]. In practice, coordinates with multiple jumps are rejected to enforce categorical integrity (a negligible event under suitable rates), trading O(1) network evaluations per leap for an O(τ) discretization bias.

High-order samplers To improve on first-order τ-leaping, multi-stage integrators achieve higher local accuracy. Ren et al. [33] introduce two-stage schemes with second-order convergence in KL. The θ-RK-2 method computes an intermediate state

y∗ = yt +

ν Poisson µt(ν)θ ∆t ,

ν∈D

then updates

ν Poisson 1 − 21θ µt(ν) + 21θ µ∗t(ν) ∆t,

yt−∆t = yt +

ν∈D

where µ∗t is the intensity at y∗. The θ-trapezoidal variant replaces the second stage with

yt−∆t = y∗ +

ν Poisson (α1 µ∗t − α2 µt)(ν)(1 − θ)∆t ,

ν∈D

2+θ2

with α1 = 2θ(11−θ), α2 = (1−θ)

2θ(1−θ) . These reduce the discretization error to O(∆t2T), enabling 3–5× fewer evaluations for comparable fidelity [33], and draw on high-order continuous schemes [20].

### D Experiment details and additional results

- D.1 Reasoning tasks

- D.1.1 Experiment details

We conduct our experiments using LLaDA 8B Instruct [27] and Dream 7B Instruct [52], which are masked diffusion models capable of advanced reasoning. Across all settings, we fix the generation length to 256 tokens. For LLaDA 8B Instruct, which supports semi-autoregressive sampling, we set the block size to 64 for all experiments. Sampling temperature is set to 0 for LLaDA 8B Instruct, and to 0.2 for Dream 7B Instruct. Additional experiments using Dream 7B Instruct with a temperature of 0 are reported in Appendix D.5.3. All sampling experiments are conducted on a single NVIDIA RTX A5000 GPU.

- D.1.2 Hyperparameter Selection Guideline

For hyperparameter selection, we adopt a lightweight three-step search procedure using a small validation set of around 100 examples.

- 1. Initial KL Threshold Estimation: We obtain an initial estimate of the KL threshold by inspecting the distribution of KL values during decoding.
- 2. Confidence Threshold Search: With the KL threshold fixed, we sweep confidence values (0.9 down to 0.6) to identify the best trade-off between accuracy and decoding speed.
- 3. KL Threshold Refinement: With the confidence threshold fixed, we refine the KL threshold through a finer-grained search around the initial estimate.

This procedure is efficient, requiring only a small number of validation samples and negligible computation relative to training. The final confidence and KL threshold configurations for each dataset and model are summarized in Table 7.

Table 7: Experiment threshold configurations for KLASS Model MATH GSM8K HumanEval MBPP Conf KL Conf KL Conf KL Conf KL

LLaDA 0.6 0.010 0.6 0.015 0.9 0.010 0.7 0.010 Dream 0.9 0.005 0.9 0.001 0.8 0.001 0.9 0.001

Table 8: Wall-clock time per sample for Top-1 and KLASS decoding. Model Dataset Accuracy Time (s) Speedup

Top-1 KLASS Top-1 KLASS

GSM8K 75.13 76.50 37.04 15.86 2.34× MATH 31.40 33.80 38.40 21.41 1.79× HumanEval 39.63 40.85 39.54 16.04 2.47× MBPP 46.69 47.86 39.12 20.68 1.89×

LLaDA

GSM8K 79.75 80.44 29.66 22.26 1.33× MATH 38.00 43.20 30.76 23.31 1.32× HumanEval 58.53 59.76 32.01 11.52 2.78× MBPP 63.81 64.59 31.89 17.65 1.81×

Dream

#### D.1.3 Wall-clock time comparison

We compare the actual wall-clock time of the KLASS and Top-k samplers on the MATH dataset in Table 8. Compared to the Top-k sampler with 256 steps (i.e., unmasking one token per step), KLASS reduces generation time by 47.4% for LLaDA and by 16.1% for Dream).

When comparing KLASS with the Top-k sampler using a similar number of steps (128 for LLaDA and 149 for Dream), the Top-k method achieves slightly lower generation times. However, this speed gain comes at the cost of reduced accuracy. KLASS not only maintains a comparable runtime but also improves accuracy, demonstrating its efficiency and effectiveness.

#### D.1.4 Statistical significance of Dream 7B Instruct results

In Table 9, we report the mean and standard deviation over three runs for all methods using Dream 7B Instruct with a temperature of 0.2. Since the experiments with LLaDA 8B Instruct were conducted with a temperature of 0, the results are deterministic, so we only report statistics for Dream.

- D.2 Text generation

- D.2.1 Experiment details

We evaluate KL-Adaptive Stability Sampling (KLASS) on a Masked Diffusion Language Model (MDLM) [34] pre-trained on the OpenWebText corpus [13]. As baselines, we include (i) the original autoregressive sampler (one-token-at-a-time unmasking), (ii) SEDD [25], and (iii) two MDLM variants: the standard 512-step sampler and the “absorb” variant [3]. For all diffusion-based methods, we generate 1,000 sequences of length 1,024 tokens under a fixed 512-step schedule, applying nucleus (top-p) filtering at p = 0.9, a history length n = 2, a KL divergence threshold ϵKL = 1e − 4, and a confidence threshold τ = 0.57. To ensure a fair comparison under this fixed step count, we cap the maximum number of tokens accepted by the thresholds at each step. In the fallback case where tokens do not pass this criterion, we revert to the original MDLM. We report generative perplexity by exponentiating the average token-level negative log-likelihood under three oracle models (LLaMA2 7B, LLaMA3 8B, and GPT-2), measure Shannon entropy of the predicted token distributions, and compute MAUVE by comparing our 1,000 generated samples to 1,000 held-out segments from the OpenWebText. All runs were executed on a single NVIDIA RTX A6000 GPU. To quantify run-to-run variability, each method was repeated with three fixed random seeds, and we report mean ± one standard deviation over these replicates.

Table 9: Mean and standard deviation for each sampler across three runs (mean ± std). MATH GSM8K Sampler Acc (%) Step Acc (%) Step

- Top-1 37.97 ± 0.12 256.00 ± 0.00 79.55 ± 0.14 256.00 ± 0.00 Random 18.73 ± 1.61 256.00 ± 0.00 37.35 ± 0.53 256.00 ± 0.00
- Top-2 33.60 ± 0.16 128.00 ± 0.00 71.69 ± 0.35 128.00 ± 0.00 conf > 0.9 41.80 ± 0.00 95.10 ± 0.00 73.67 ± 0.15 74.81 ± 0.08 KL < 0.001 41.27 ± 0.09 162.49 ± 0.00 76.70 ± 1.14 150.02 ± 0.32 KLASS (ours) 43.20 ± 0.00 149.72 ± 0.00 79.43 ± 0.72 155.67 ± 0.41

(a) MATH & GSM8K

HumanEval MBPP Sampler Acc (%) Step Acc (%) Step

- Top-1 58.53 ± 0.00 256.00 ± 0.00 63.81 ± 0.00 256.00 ± 0.00 Random 18.09 ± 2.51 256.00 ± 0.00 28.14 ± 0.91 256.00 ± 0.00
- Top-2 42.88 ± 0.29 128.00 ± 0.00 47.08 ± 0.32 128.00 ± 0.00 conf > 0.9 50.00 ± 0.00 52.47 ± 0.00 57.59 ± 0.00 72.49 ± 0.00 KL < 0.001 59.35 ± 0.29 73.94 ± 0.57 62.65 ± 0.00 108.15 ± 0.00 KLASS (ours) 59.35 ± 0.29 74.88 ± 0.74 64.59 ± 0.00 111.24 ± 0.00

(b) HumanEval & MBPP

- Table 10: Generative perplexity, MAUVE and entropy on unconditional text generation. Here, ‘D3PM’ denotes an MDLM that is parameterized using D3PM.

Method MAUVE↑ LLaMA2↓ LLaMA3↓ GPT2↓ Entropy↑

*Data 1.000 7.00 9.40 14.80 5.44 AR 0.855 ± 0.033 10.97 ± 0.10 15.12 ± 0.18 12.07 ± 0.12 5.21 ± 0.02 SEDD 0.037 ± 0.012 53.09 ± 0.24 109.60 ± 0.79 105.40 ± 0.67 5.62 ± 0.00 D3PM 0.022 ± 0.006 41.82 ± 5.88 72.85 ± 12.69 76.70 ± 0.62 5.40 ± 0.00 MDLM 0.115 ± 0.033 30.88 ± 0.20 54.15 ± 0.27 51.78 ± 0.14 5.46 ± 0.00 KLASS 0.179 ± 0.041 26.94 ± 0.24 49.19 ± 0.40 45.50 ± 0.42 5.43 ± 0.01

#### D.3 Image generation

- D.3.1 Experiment details

We evaluate KLASS on the MMaDA [49]. We compare two samplers—confidence-based and KLASS. For each sampler, we generate 10,000 class-conditional images with uniformly sampled ImageNet labels under a 16-step, 32-step decoding budget. For KLASS, we fix the hyperparameters to history length n = 1, KL divergence threshold ϵKL = 0.3, and confidence threshold τ = 0.1. For a fair comparison with a fixed step count, we restrict the per-step reveal count allowed by the thresholds. FID is computed against the 50k ImageNet validation set using official Inception-v3 features, and IS follows the standard protocol. All image-generation runs use a single NVIDIA RTX A5000.

D.4 Molecules

- D.4.1 Experiment details

We mainly follow the experimental settings in [37] for the experiment. For training, we train independent models for two target conditions: QED and ring count. We use diffusion step size 32, taking 25,000 gradient steps. Small size diffusion transformer (DIT) which is composed of 12 DIT

blocks, hidden dimension of 768 is utilized for the architecture. We train the model with classifier-free guidance (CFG) training with dropout condition probability of 0.1. We generate samples with CFG strength γ = 1 for the experiment. Reported values in Table 4 for KLASS use ϵKL = 0.001 and τ = 0.98 for both QED and Ring Count experiments. We utilize a single RTX 3090 GPU for both training and the inference.

- D.5 Ablations

- D.5.1 Hyperparameter Sensitivity

To address sensitivity, we performed a grid search across diverse models and tasks (LLaDA/Dream for reasoning, MDLM for molecular), demonstrating the robustness of KLASS.

Our findings show that configurations near the selected optimum consistently yield high accuracy while significantly reducing sampling steps. For instance, on HumanEval with LLaDA (Table 11), settings near (confidence = 0.9, KL = 0.01) maintain or improve upon the baseline accuracy of 39.63% while using far fewer than 256 steps. Similar trends are observed for other settings (Tables 12, 13, 14, 15). Slightly different hyperparameter settings can sometimes outperform the main reported configuration, indicating both robustness and potential for further tuning.

#### D.5.2 Effect of history length

We evaluate the effect of varying the KL score history length in KLASS across different KL divergence thresholds and confidence thresholds on the MATH dataset. Results are reported in Table 16 for both LLaDA and Dream models.

For LLaDA, a history length of 2 offers the best balance of accuracy and efficiency, particularly at KL threshold of 0.015 and a confidence threshold of 0.6. At the stricter threshold of 0.9, history length has less impact, suggesting that a more relaxed confidence threshold allows more informative token candidates to be considered for unmasking. For Dream, the highest accuracy is achieved with history length 2, ϵKL = 0.005, and τ = 0.9. At a lower confidence of 0.6, overall accuracy decreases, and longer history helps stabilize token predictions. In summary, history length 2 is optimal across most settings, providing improved accuracy with moderate computational cost. Therefore, we use history length 2 for all reasoning tasks.

#### D.5.3 Effect of temperature

- Table 17 shows that KLASS consistently improves over a deterministic Top-1 sampler across tasks and temperature settings. Gains are especially strong at temperature 0, where KLASS boosts accuracy by 6.22 to 8.00 percentage points and reduces steps by up to 79%. At temperature 0.2, it still provides solid improvements, with accuracy gains of 0.69 to 5.10 points and step reductions of 39% to 71%. These results highlight KLASS’s ability to accelerate reasoning and improve accuracy, with even greater boosts in more deterministic settings.

E Comparison to other diffusion samplers

E.1 Performance on reasoning tasks To recap the baselines used in the main experiment (Table 1), we consider:

- • Top-k: Tokens are generated by selecting the one with the highest confidence [7].
- • Random: Tokens are generated in a purely random order [1].

- Table 18 reports results for two additional samplers:

- • Top-k Margin: Unmasks the token with the largest probability margin between the highest and second-highest confidence [21].
- • Entropy: Tokens are ranked by their negative Shannon entropy, prioritizing those with lower uncertainty (i.e., higher confidence) in the model’s prediction.

- Table 11: Hyperparameter sensitivity on HumanEval with LLaDA. We report values as Accuracy (Steps). Conf = 0.9 and KL = 0.01 are chosen for KLASS. The baseline accuracy is 39.63% with 256 steps.

KL = 0.015 KL = 0.01 KL = 0.005 Conf = 0.95 39.63 (65.19) 40.24 (99.29) 40.24 (103.14)

Conf = 0.9 40.24 (89.29) 40.85 (91.98) 40.24 (96.75) Conf = 0.85 39.63 (84.98) 39.63 (88.78) 40.24 (94.48) Conf = 0.8 39.02 (83.14) 39.02 (87.21) 40.85 (94.01)

- Table 12: Hyperparameter sensitivity on MBPP with LLaDA. We report values as Accuracy (Steps). Conf = 0.7 and KL = 0.01 are chosen for KLASS. The baseline accuracy is 48.64% with 256 steps.

KL = 0.015 KL = 0.01 KL = 0.005

Conf = 0.8 49.42 (122.37) 49.42 (134.52) 49.42 (134.52) Conf = 0.75 49.03 (118.61) 49.42 (123.45) 48.25 (132.11) Conf = 0.7 48.25 (116.24) 49.03 (127.81) 49.03 (130.67) Conf = 0.65 46.30 (113.22) 48.64 (118.54) 49.42 (128.99)

- Table 13: Hyperparameter sensitivity on HumanEval with Dream. We report values as Accuracy (Steps). Conf = 0.8 and KL = 0.001 are chosen for KLASS. The baseline accuracy is 58.53% with 256 steps.

KL = 0.005 KL = 0.003 KL = 0.001 KL = 0.0005

Conf = 0.9 54.87 (64.78) 56.10 (67.41) 57.93 (74.86) 61.59 (79.57) Conf = 0.85 57.32 (59.39) 55.49 (65.31) 59.15 (73.38) 60.98 (79.43) Conf = 0.8 51.22 (58.09) 54.27 (62.82) 59.76 (73.73) 60.96 (79.51) Conf = 0.75 48.78 (55.21) 53.05 (62.61) 59.15 (73.41) 60.37 (79.37)

- Table 14: Hyperparameter sensitivity on MBPP with Dream. We report values as Accuracy (Steps). Conf = 0.9 and KL = 0.001 are chosen for KLASS. The baseline accuracy is 63.81% with 256 steps.

KL = 0.005 KL = 0.003 KL = 0.001 KL = 0.0005

Conf = 0.95 65.37 (108.93) 65.37 (110.03) 64.59 (112.56) 64.59 (113.14) Conf = 0.9 62.65 (103.99) 64.20 (107.09) 64.59 (111.24) 64.59 (112.65) Conf = 0.85 64.20 (101.34) 63.81 (105.43) 65.37 (112.54) 64.59 (112.76) Conf = 0.8 63.81 (96.02) 63.04 (103.22) 65.37 (112.52) 64.59 (112.82)

- Table 15: Hyperparameter sensitivity on Molecule QED (MDLM). Conf = 0.98 and KL = 0.001 are chosen for KLASS. The baseline QED is 0.526 with 32 steps. We report values as QED (Steps).

KL = 0.01 KL = 0.005 KL = 0.001 KL = 0.0005

Conf = 0.999 0.527 (18.61) 0.526 (18.61) 0.524 (18.58) 0.538 (18.45) Conf = 0.99 0.515 (18.22) 0.521 (18.33) 0.543 (18.69) 0.537 (18.66) Conf = 0.98 0.531 (18.43) 0.517 (18.38) 0.546 (18.78) 0.534 (18.82) Conf = 0.96 0.529 (18.43) 0.535 (18.51) 0.534 (18.63) 0.543 (18.80)

KLASS consistently outperforms Top-k Margin and Entropy-based methods with fewer sampling steps. On LLaDA, it achieves top results on GSM8K and HumanEval in less than half the usual iterations. While Entropy yields the highest accuracy on MATH and MBPP with 256 steps, its performance drops sharply with fewer steps. In contrast, KLASS maintains high accuracy at lower computational cost. On Dream, it also improves MATH and GSM8K accuracy while reducing steps, demonstrating more efficient and effective sampling.

- Table 16: Ablation results on the history length, across KL thresholds and models, grouped by confidence thresholds.

LLaDA Dream Conf KL History Length Acc (%) Steps Acc (%) Steps

- 1 32.2 77.13 37.6 123.73

- 2 33.8 128.62 39.6 165.68

- 3 32.2 153.60 41.8 182.93

0.010

- 1 30.4 89.04 34.8 83.11

- 2 33.2 121.11 35.6 119.33

- 3 33.2 146.19 36.6 146.40

0.6

0.015

- 1 31.0 74.42 34.4 73.66

- 2 31.0 117.01 29.6 104.19

- 3 30.6 140.56 34.6 128.46

0.020

- 1 31.4 128.76 42.2 141.36

- 2 30.6 152.75 41.6 169.45

- 3 30.8 170.66 42.0 183.51

0.010

- 1 31.0 127.05 41.0 126.42

- 2 30.8 149.72 43.2 149.72

- 3 31.0 167.11 40.2 165.37

0.9

0.015

- 1 30.8 125.85 40.4 120.35

- 2 30.8 148.92 42.0 142.75

- 3 31.2 164.75 40.4 158.01

0.020

Table 17: Effect of temperature on KLASS gains over Top-1 sampler with Dream. Method MATH GSM8K HumanEval MBPP Acc Steps Acc Steps Acc Steps Acc Steps Temperature = 0.2

Top-1 38.10 256 79.75 256 58.53 256 63.81 256 KLASS 43.20+5.10 150-106 80.44+0.69 156-100 59.76+1.23 74-182 64.59+0.78 111-145

Temperature = 0

Top-1 25.80 256 41.70 256 29.27 256 33.46 256 KLASS 33.80+8.00 121-135 47.92+6.22 106-150 37.19+7.92 53-203 40.86+7.40 76-180

### F Examples of generated samples

We present a qualitative comparison of generated samples to demonstrate KLASS’s improvements.

- Table 19 illustrates a mathematical reasoning task. The Top-1 baseline fails due to a simple arithmetic

error in the first step (f(−2) = −28 = −4), while the random sampling disregards the function’s structure. In contrast, KLASS correctly computes each intermediate step and combines them to reach

the correct answer, 143 , highlighting its improved reasoning reliability. We also compare long-form text coherence. Figure 4 shows an MDLM baseline sample that begins on-topic (‘SolarCity’) but

quickly degenerates. It exhibits severe generation artifacts such as repetition (‘SolarCitySolarCity’), nonsensical phrases (‘informs of capacity’). The sample’s coherence completely breaks down, ending with an unrelated spam link. Conversely, Figure 5 shows a KLASS sample on ‘urban sprawl.’ This text is relatively coherent and topically consistent from start to finish, maintaining a plausible journalistic style.

Table 18: Performance and sampling steps on reasoning benchmarks for Top-k Margin and Entropy samplers.

MATH GSM8K HumanEval MBPP Model Method Acc↑ Step↓ Acc↑ Step↓ Acc↑ Step↓ Acc↑ Step↓

Top-k Margin 32.0 256 74.14 256 39.63 256 47.86 256 Top-k Margin 31.4 128 74.45 128 30.48 128 40.08 128 Entropy 34.6 256 75.43 256 35.97 256 51.75 256 Entropy 32.6 128 73.01 128 25.60 128 40.08 128 KLASS (ours) 33.8 128.62 76.50 98.57 40.85 91.98 47.86 119.59

LLaDA

Top-k Margin 39.4 256 79.45 256 58.53 256 63.81 256 Top-k Margin 32.4 128 71.49 128 43.29 128 46.69 128 Entropy 39.4 256 79.45 256 58.53 256 63.81 256 Entropy 32.6 128 71.49 128 43.29 128 46.69 128 KLASS (ours) 43.2 149.72 80.44 156.24 59.76 73.73 64.59 111.24

Dream

### G Limitation & Broader impact

#### G.1 Limitations

Further scalability to larger models In the absence of larger-size discrete diffusion models compared to AR models, our method cannot be evaluated on the more challenging benchmarks such as in agentic systems of LLMs [22, 39, 44].

Hyperparameter Search Cost Although we provide a lightweight guideline in Appendix D.1.2, KLASS introduces a minimal search cost for its additional hyperparameters compared to simpler baselines like Top-1 sampling. While our analysis in Appendix D.5.1 shows performance is robust around the optimal points, further tuning could still be performed to find the best possible settings.

#### G.2 Broader impact

This work focuses on how to improve discrete diffusion sampler. However, improving diffusion samplers, if misused, could facilitate the creation of harmful or misleading text, images, or even novel molecules for malicious purposes.

To mitigate these risks, improvements in sampling techniques should be accompanied by a strong emphasis on AI safety research and ethical guidelines. This includes developing robust mechanisms for detecting and preventing the generation of harmful content and fostering public discourse on the societal implications of advanced generative models.

###### Table 19: Generated solutions with Dream on MATH.

|Question<br><br>|If f(x) = 3xx−−22, what is the value of f(−2) + f(−1) + f(0)? Express your answer as a common fraction.<br><br>|
|---|---|
|Answer|14 3<br><br>|
|Top-1<br><br>|First, let’s find f(−2): f(−2) = 3(−−22)−−22 = −28 = −4 Next, let’s find f(−1): f(−1) = 3(−−11)−−22 = −−53 = 53 Now, let’s find f(0): f(0) = 3(0)0−−22 = −−22 = 1 Finally, add: f(−2) + f(−1) + f(0) = −4 + 53 + 1 = −4 + 83 = −312 + 83 = −34 The final answer is −4<br><br>3 .|
|Random<br><br>|f(−2) = 3(−2−−2)2 = 1, f(−1) = 3(−1−−2)2 = 1, f(0) = 3(0−−22) = 1. So f(−2) + f(−1) + f(0) = 3. The final answer is 3.<br><br>|
|KLASS|First, let’s find f(−2): f(−2) = 3(−−22)−−22 = −−84 = 2 Next, let’s find f(−1): f(−1) = 3(−−11)−−22 = −−53 = 53 Now, let’s find f(0): f(0) = 3(0)0−−22 = −−22 = 1 Finally, add: f(−2) + f(−1) + f(0) = 2 + 53 + 1 = 2 + 38 = 63 + 83 = 143 The final answer is 143 .<br><br>|

Figure 4: MDLM Generated Sample (512 tokens)

<endoftext>City\u2019s new Solar City facility in San Diego will be providing up to 4 times New York\u2019s current capacity as energy efficient storage is on the way , according to new reporting StarTheSunCity reports.\n\nSolarCity says a site in Los Angeles is designed to replace NYC\u2019s current capacity in the emergency room. As

SolarCity Krieger confirmed , a facility in Richmond is planned for SolarCitySolarCity.\n\nThe company plans to adopt the city\u2019s own capacity taking advantage that the city is constantly being built to low energy standards.

... The NYTChronicle reports that SolarCity is aiming to provide a capacity increase by 10 to 16 percent of capacity , greater than New York. Even the

non-Manhattan capacity increase is about 2 percent.\n\nStarTheSunCity

says it will soon run out of space for an increasing number of ’informs of capacity’\u2019 including some rooms not dictated by capacity requirements .\n\nThe indoor solar center will be built to pay solar customers a $3.40

price per day, and based according to the budget for

SolarCitySanVoltSolarCity\u2019s facility opens in 2017, and the City Center ,

located in just one block from Microsoft Place , seats 25 million New Yorkers. SolarCity spokesman Barry said , \u201cIt is clear that the city represents the best leaders in energy efficiency. In fact , providing so much more energy for people is a vital leadership position for consumers\ u2019 transportation , health , and finances over the next decade.\u201d\n\ nClick to Download The New York Charger and look for True New York Charger to feel free download or for support.<endoftext>

Figure 5: KLASS Generated Sample (512 tokens)

<|endoftext|> an order of magnitude smaller than a rise in the number of people living in other cities.\n\nAs a result , urban sprawl is now central to a realignment in how planners view the contribution of each individual population and location to different cities.\n\nThe University

researchers measured growth in densities by a factor of two and space and pollution by 31 per cent in the period of urban sprawl more broadly.\ n\nThere is, of course , a period of growth in the number of inhabitants relocated or public space by a quarter to two per cent.

... Again , at this volume , it has tended to grow unevenly in other cities.\n\ nThe Berlin Centre for Urban Research has a map of low urban sprawl areas

around the borders in Brussels , Belgium.\n\nLichtner and co-authors of the study , published in UrbanLabs West , say places with more intense urban sprawl need infrastructure to expand public space.\n\nAs a result , their boundaries are denser , that means more land for construction and more roads for the new developers want to build. Public libraries and classrooms also go up dramatically , supposedly making urban sprawl more severe and steeper.\n\nLichtner and co-authors say the foundations for urban sprawl are powerful in this respect. They say much of the impact of the study is ultimately centred on the urban sprawl itself , that offers the means to thrive. \"If sprawl is the city , it might be infrastructure that allows for housing in sprawl , creating a productive concentration between the wealthier and the poor that stand for a new development ,\" they said.<|endoftext|>

