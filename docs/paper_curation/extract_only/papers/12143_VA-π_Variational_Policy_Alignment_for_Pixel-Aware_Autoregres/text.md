#### VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation

Xinyao Liao1,2* Qiyuan He2*† Kai Xu2 Xiaoye Qu1 Yicong Li2 Wei Wei1 Angela Yao2 1Huazhong University of Science & Technology 2National University of Singapore

## arXiv:2512.19680v1[cs.CV]22Dec2025

##### Abstract

Autoregressive (AR) visual generation relies on tokenizers to map images to and from discrete sequences. However, tokenizers are trained to reconstruct clean images from ground-truth tokens, while AR generators are optimized only for token likelihood. This misalignment leads to generated token sequences that may decode into low-quality images, without direct supervision from the pixel space. We propose VA-π, a lightweight post-training framework that directly optimizes AR models with a principled pixel-space objective. VA-π formulates the generator–tokenizer alignment as a variational optimization, deriving an evidence lower bound (ELBO) that unifies pixel reconstruction and autoregressive modeling. To optimize under the discrete token space, VA-π introduces a reinforcement-based alignment strategy that treats the AR generator as a policy, uses pixel-space reconstruction quality as its intrinsic reward. The reward is measured by how well the predicted token sequences can reconstruct the original image under teacher forcing, giving the model direct pixel-level guidance without expensive free-running sampling. The regularization term of the ELBO serves as a natural regularizer, maintaining distributional consistency of tokens. VA-π enables rapid adaptation of existing AR generators, without neither tokenizer retraining nor external reward models. With only 1% ImageNet-1K data and 25 minutes of tuning, it reduces FID from 14.36 to 7.65 and improves IS from 86.55 to 116.70 on LlamaGen-XXL, while also yielding notable gains in the text-to-image task on GenEval for both visual generation model (LlamaGen: from 0.306 to 0.339) and unified multi-modal model (Janus-Pro: from 0.725 to 0.744). Code is available at https://github.com/LilShake/VA-Pi.

##### 1. Introduction

Autoregressive (AR) image generation models represent visual data as sequences of discrete tokens. This formulation naturally aligns with the architecture of large language

*Equal contribution. †Project lead.

[Figure 1]

- (a) Qualitative comparison between LlamaGen-XXL (left) and VA-π (right) on gold fish image generation

[Figure 2]

- (b) Kernel Density Estimation (KDE) of image embeddings showing VA-π shifts the AR

[Figure 3]

(c) t-SNE visualization of image embeddings illustrating how VA-π aligns the generator’s representation with the ground-truth manifold.

generator’s output closer to the ground-truth manifold.

Figure 1. Pixel-Aware Alignment via VA-π. VA-π enables efficient post-training via variational policy optimization, aligning the pixel-space distribution of AR generated images with that of ground-truth images.

models (LLMs) [2, 3] and paves a way for unified multimodal systems [4–9]. The ultimate goal of any image generation model is to capture the distribution of images in pixel space. From this perspective, a more principled formulation would be to optimize an end-to-end objective defined directly over the pixel distribution. But such a direct approach is known to be intractable in practice [10, 11], motivating the use of visual tokenizers to make AR modeling feasible. The standard pipeline features two stages. In the first stage, a visual tokenizer [12] is trained, with an encoder that converts images into discrete token sequences, and a decoder that reconstructs the images from those tokens. In the second stage, an AR model is trained to capture the distribution over these discrete sequences.

[Figure 4]

Figure 2. Overview of VA-π. VA-π aligns the visual AR model with tokenizer via variational optimization. Given a reference image and its ground-truth tokens, VA-π adds context noise and lets the AR model compute logits under teacher forcing and samples target tokens. These sampled tokens are decoded back into an image, and the reconstruction reward is defined against the reference image. This reward is then used for policy updates within an RL framework such as GRPO [1]. Additionally, a likelihood regularization using cross-entropy loss between the logits and ground-truth tokens is retained to preserve the model’s original next-token prediction ability.

However, optimized only at the token level without pixel-space supervision, the AR generator can produce high-likelihood token sequences that decode into visually suboptimal images with artifacts and degraded perceptual quality [13, 14]. We refer to these as off-manifold token sequences, which, when decoded, deviate from the image manifold and thus produce incoherent visual structures.

Previous studies address the mismatch between tokenlevel likelihood and image-level fidelity by applying noisycontext regularization to either the AR generator [13, 15, 16] or the tokenizer [14, 17]. These methods inject noise during training to improve robustness against corrupted sequences. However, they did not directly address the pixellevel misalignment. Moreover, the tokenizer-centric approach only enables the tokenizer to be more tolerant of off-manifold token sequences rather than preventing their generation from the AR generator. As shown in Sec. 5.2, training on excessive noise can even overly smooth the decoder’s token-pixel mapping, leading to reduced reconstruction sharpness and visual fidelity.

In this work, we take a standpoint of solving the root cause by directly aligning AR-generated token sequences to the distribution of images in pixel space, fundamentally mitigating the generation of off-manifold token sequences. We pose the question as: Can we design an objective that aligns token-level modeling with pixel-level distributions? We show that this is possible by framing the AR-generated discrete token sequence as a latent random variable of the pixel-level image. This perspective leads to a tractable evidence lower bound (ELBO) of the image likelihood. Specifically, we interpret the pixel reconstruction produced by the tokenizer’s decoder as corresponding to the reconstruction term of the ELBO, while the AR model’s likelihood objective serves as the prior term that preserves proper token-

level likelihood modeling. Such a framing unifies pixelspace reconstruction with token-level predictions.

However, as the variables are discrete, maximizing the ELBO is non-trivial. A common solution is the straightthrough estimator (STE) [18], which provides a surrogate gradient path from discrete tokens to continuous logits, enabling gradient flow into the AR generator. Nevertheless, STE only propagates gradients along the ground-truth path, limiting learning to observed token sequences.

To overcome this, we propose Variational Policy Alignment for Pixel-aware Autoregressive Generation, or VA-π for short. VA-π treats the tokenizer’s teacher-forcing reconstruction loss as an intrinsic reward, offering a stable and informative signal directly tied to pixel-space fidelity. Meanwhile, the variational regularization term plays the role of a constraint for keeping the updated policy close to the base AR model, thereby preserving its learned token distribution. Unlike STE, which only updates ground-truth tokens, the RL formulation distributes gradients across all sampled token sequences according to their pixel-space rewards. This combination enables broader token-space exploration and rapid adaptation toward pixel-level consistency with limited data and compute.

Experiments on class-to-image and text-to-image generation tasks show that VA-π is effective and efficient. On ImageNet-1K [19], post-training LlamaGen-XXL [20] with VA-π substantially enhances visual fidelity and diversity, reducing the FID from 14.36 to 7.65 and increasing IS from 86.55 to 116.70 without classifier-free guidance. When applied to text-to-image and unified multimodal generation tasks, VA-π improves conditioning accuracy and perceptual quality. All these improvements are achieved within just training on about 1% of the pretrained dataset, without any external reward model, while using only 13.4% of

the compute cost required by conventional free-running RL methods such as AR-GRPO [1].

The contributions of this work are as follows:

- • We formulate a variational objective that bridges discrete token modeling and pixel-level reconstruction, aligning AR generators directly with the image distribution.
- • Building on this formulation, we propose VA-π, a posttraining framework that leverages RL for optimization. By treating the reconstruction as a reward, VA-π provides direct pixel-level feedback to the generator policy.
- • VA-π is highly compute- and data-efficient, requiring only 25 minutes of post-training on 8×A100 GPUs with 1% of pretraining dataset, without relying on external reward models or expensive free-running sampling.
- • VA-π consistently improves visual quality across both class-to-image and text-to-image generation tasks, with minimal data and compute, demonstrating a practical path toward efficient alignment of visual AR models.

##### 2. Related Work

###### 2.1. Auto-Regressive Visual Generation

Autoregressive (AR) models [20–25] have emerged as a competitive paradigm for visual generation, rivaling diffusion-based [26–28] and masked generative models [11, 29–31]. Direct pixel-level autoregression is computationally prohibitive, so modern AR frameworks use patchbased discrete tokenizers [12, 18]. The tokenizers compress local image regions into latent tokens; recent works focus on mitigating quantization artifacts in the tokenizers [8, 27, 32, 33] and on extending the tokenization beyond regular 2D grids [34–37]. However, these advances only refine individual components and overlook a more fundamental bottleneck: the objective is separated between the AR generator’s predictive likelihood training and the tokenizer’s pixel-level reconstruction goal.

###### 2.2. Tokenizer-Generator Alignment

The two-stage training pipeline for standard AR visual generation introduces an inherent inconsistency between the generator and the tokenizer. Prior work has reduced this mismatch from two directions. Generator-centric approaches modify the AR training objective, for instance, by adding noisy context to the token sequences [13] or by randomizing the token ordering [15, 16]. Tokenizer-centric approaches adjust the tokenizer to better accommodate the generator. One approach is to enhance decoder robustness to the generator’s sampled token distributions [14]; another is to embed the AR generator’s causal dependencies into the token structure [38–40]. While these strategies reduce the mismatch, they typically require costly retraining and intuitive regularization. Our methods are more effective based on principled objectives with superior efficiency.

###### 2.3. Reinforcement Learning in Visual Generation

Reinforcement learning (RL) has emerged as an effective finetuning strategy for visual generation, enabling enhanced reasoning [41, 42], alignment with human preferences [43–

- 45], and improving controllability or prompt alignment [1,
- 46, 47]. To the best of our knowledge, our work is the first to apply RL to directly optimize AR models with respect to pixel-space reconstruction quality.

##### 3. Preliminaries 3.1. Visual Autoregressive Models

State-of-the-art visual AR generation models [16, 38, 48, 49] feature two components: (1) a visual tokenizer that converts an image into a sequence of discrete codes, and (2) an autoregressive model that models and samples these codes. The Visual Tokenizer compresses an image into discrete tokens while preserving reconstruction quality. Discrete tokenizers [12, 20, 26, 27] includes an encoder E, a quantizer Q, and a decoder D. Given an image I ∈ R3×H×W, the feature dimension C, and the token sequence length N:

z = E(I), x = Q(z), ˆI = D(x), (1)

where z ∈ RC×N is the latent feature and x ∈ {1,...,K}N is the discrete code indices. The distribution of discrete codes can then be learned via the AR model.

The tokenizer is optimized using a combination of reconstruction and quantization objectives. Specifically, the loss typically consists of: (1) a pixel-wise reconstruction loss LMSE, (2) a perceptual reconstruction loss Lp such as LPIPS [50], and (3) a vector quantization loss Lq [18]. The overall training objective with coefficients λp and λq is:

. (2)

###### + λqLq

Ltok = LMSE + λpLp Reconstruction loss

Quantization loss

We provide the detailed formulation of each loss term, including the quantization objective, in the Appendix B.

The Autoregressive Model πθ, parameterized by θ, factorizes the distribution of a sequence x1:N into conditional probabilities, where each token xi depends on all preceding tokens. The model is trained using teacher forcing, where the ground-truth preceding tokens from the observed sequence are provided as context. Formally, the training maximizes the log-likelihood of the observed sequence:

θ = arg max

θ

N

log πθ(xi | x1:i−1). (3)

i=1

During inference, tokens are generated sequentially in free-running mode by sampling xi ∼ πθ(· | x1:i−1). The complete sequence x is then mapped back to the image space via the decoder D to obtain the synthesized image ˆI.

In the cases of class- and text-to-image generation, the class or text input would serve as an additional conditioning variable, which we omit from the notation above for simplicity.

###### 3.2. Reinforcement Learning

An autoregressive generation process can be formulated for a reinforcement learning (RL) problem, where the AR model πθ serves as the policy that sequentially samples tokens to maximize an expected reward. This formulation enables optimization of πθ for given rewards via policygradient methods such as PPO [51] or GRPO [52]. We adopt GRPO [52] for its stability, as it normalizes rewards across sample groups to reduce variance and leverages KLregularization to preserve pretrained priors.

Group Relative Policy Optimization (GRPO). In GRPO, for each condition q, the policy πθ generates a group of G samples o1,...,oG. Each sample is scored by a reward function R(o,q), yielding scalar rewards ri = R(oi,q). To enhance training stability, these rewards are normalized within the group to obtain the group-relative advantages:

ri − mean({rj}Gj=1) std({rj}Gj=1)

Aˆi =

, (4)

The GRPO objective has two terms: a clipped policyratio objective, weighted by normalized advantages Aˆi from Eq. 4, and a KL-divergence penalty for stability:

G

1 G

min ρiAˆi,

JGRPO(θ) = Eq,o∼π

θold

i=1

clip(ρi,1−ϵ,1+ϵ)Aˆi − β DKL πθ ∥πref ,

(5) Here, ρi = π

θ(oi|q)

πθold(oi|q) is the policy ratio, Aˆi denotes the group-normalized advantage defined in Eq. 4, and β controls the KL regularization strength. The first term encourages policy improvement, while the second term acts as a stability constraint, penalizing large deviations from the reference policy πref, thereby preventing policy collapse.

In AR generation, the policy πθ acts as a token-level generator that sequentially predicts visual tokens. GRPO therefore enhances AR generation quality by optimizing the policy to maximize perceptual or semantic rewards.

##### 4. VA-π: Variational Policy Alignment

We propose VA-π, a post-training framework that optimizes AR generators for pixel-space distribution alignment as shown in Fig 2. From the intractable pixel-level likelihood, we derive an ELBO that yields two training signals: a pixel-space reconstruction objective and a tokenlevel regularization that preserves the AR prior (Sec. 4.1). The regularization term reduces to a simple next-token prediction loss (Sec. 4.2), while the reconstruction term is

Table 1. Comparison on text–image alignment metrics. VA-π without reward model attains higher scores than AR-GRPO, even on the alignment reward that AR-GRPO itself is optimized for.

Model Ext. Rwd CLIP↑ HPS v2↑

LlamaGen-XL – 0.245 0.153 + AR-GRPO [1] ✓ 0.274 0.208 + VA-π (Ours) × 0.291 0.211

non-differentiable and is therefore optimized as a reward through reinforcement learning (Sec. 4.3). We then adopt GRPO [52] to integrate these two components into a single, stable training procedure, yielding the full VA-π algorithm (Sec. 4.4). All notations follow in Sec. 3.

###### 4.1. Evidence Lower Bound for Alignment

Let pdata(I) denote the real image distribution. We define the discrete token sequence x as a latent variable. Our alignment objective maximizes the pixel-space likelihood by decoding token sequences through the tokenizer during post-training, rather than relying solely on the token-level likelihood of the AR model as in Eq. 3:

EI∼p

log p(I;θ,ϕ) , p(I;θ,ϕ) =

max

data

θ

p(I,x;θ,ϕ) =

x

x

pϕ(I | x)πθ(x),

(6)

where the AR model πθ defines the likelihood over discrete token sequence, the decoder D’s parameters ϕ defines a pixel-space likelihood pϕ(I | x).

However, directly evaluating Eq. 6 is intractable because of the integral term1, as noted in the VAE framework [10]. To obtain a tractable surrogate objective similarly, we introduce a variational posterior qψ,θ(x | I) learned by AR models that approximates the posterior p(x | I). Analogous to how a VAE learns the posterior by reconstructing the ground-truth image, we learn a discrete posterior over ground-truth token sequences by training the AR generator under teacher forcing. Given encoder Eψ, quantizer Q and the AR model πθ, the posterior qψ,θ(x | I) is defined as:

N

πθ(xi | x∗1:i−1), x∗ = Q Eψ(I)

qψ,θ(x | I) =

i=1

(7)

Here, each token is predicted using the true prefix x∗1:i−1 rather than the model’s own outputs. Thus qψ,θ(x | I) concentrates on sequences that decode faithfully back to I, whereas free-running sampling xi ∼ πθ(· | x1:i−1) quickly drifts off the data manifold due to error accumulation. Teacher forcing therefore offers a stable, low-variance

1It marginalizes over all latent token sequences x, i.e., x pϕ(I | x) πθ(x), which is generally intractable due to the large discrete space.

approximation to the posterior. Based on the defined posterior, the evidence lower bound (ELBO) of p(I;θ,ψ,ϕ) is:

log p(I;θ,ψ,ϕ) ≥Eq

ϕ,θ(x|I) log pψ(I | x)

reconstruction term

###### − KL qϕ,θ(x | I)∥πθ(x)

prior regularization term

.

(8)

which can be derived by the Jensen Inequality [10]. (The derivation is provided in Appendix A.1.) Since jointly tuning both the AR model and tokenizer can be unstable, we only update the AR generator πθ while keeping the tokenizer ϕ,ψ frozen in our settings.

Maximizing the ELBO offers a principled objective that aligns the AR generator’s token distribution πθ(x) with the pixel-space likelihood. Specifically, the reconstruction term enforces that, given an image and its encoded tokens, the AR model under teacher forcing should generate token sequences capable of reconstructing the original image, thereby providing pixel-level supervision for optimization. The prior regularization term preserves the AR model’s original token-level likelihood modeling, ensuring consistency with its pretrained distribution. Theoretical analysis of our learning framework and comparison with VAE [10] and standard AR based on VQVAE [18] are provided in the Appendix A.

###### 4.2. Regularization with Next Token Prediction

During free-running inference, an AR model samples each token from its own history (xi ∼ πθ(· | x1:i−1)) instead of conditioning on the ground-truth prefix used in teacher forcing (xi ∼ πθ(· | x∗1:i−1)), causing small deviations to accumulate, known as exposure bias [53].

The key insight of the prior regularization term is that minimizing it can be viewed as directly minimizing the exposure bias, since the KL term in Eq. 8 measures the discrepancy between the teacher-forced distribution qψ,θ(x | I) and the free-running distribution πθ(x). While various approaches have been proposed to mitigate exposure bias, we follow reAR [13] for efficiency by introducing contextual noise and applying the next-token prediction loss:

1 N

Lprior(πθ,x∗,x˜∗) = −

N

log πθ(x∗t | x˜∗<t), (9)

t=1

where N denotes the sequence length, and x˜∗ ∼ Kξ(· | x∗) represents a dependent variable of x∗ corrupted by a kernel Kξ with perturbation rate ξ. Additional theoretical analysis and details of corruption are provided in Appendix A.3.

###### 4.3. Learning with Reconstruction Reward

Although the ELBO provides a tractable optimization objective, the reconstruction term is difficult to optimize

end-to-end because of several non-differentiable operations. Both the quantizer Q and the discrete teacherforcing sampling block gradient flow, preventing direct back-propagation of pixel-space losses. The StraightThrough Estimator (STE) [18] addresses the quantization issue by treating the codebook lookup as an identity mapping in the backward pass, allowing gradients from the decoder to reach the generator logits (see Appendix C.1). However, token sampling poses an additional challenge: while STE enables gradients through quantization, it does not account for sampling probabilities over the categorical distribution, leaving the overall objective biased. This mismatch motivates us to apply reinforcement learning instead. Empirical evidence supporting this analysis is provided in Sec. 5.2.

To resolve this, we formulate the problem as a policy optimization, where the AR model is optimized to produce the token sequence that maximizes the reconstruction reward, i.e., the negative reconstruction loss. Given a reference image I, ground-truth tokens x∗ = Q(E(I)), tokens sampled by teacher-forcing x ∼ πθ(· | x∗), and the decoded image ˆI = D(xˆ), the reconstruction reward is:

R(x,x∗) = − LMSE(ˆI,I) + λpLp(ˆI,I) (10)

We can then use the reward as the goal for reinforcement learning. To avoid multiple forward of the AR model, we use the noisy token sequence x˜∗ ∼ pξ(· | x∗) as the same used in next-token prediction regularization. Intuitively, maximizing such reward guides πθ to produce token sequences whose decoded images align with the reference images, maximizing the reconstruction term in Eq. 8.

###### 4.4. VA-π Policy Optimization

Our key insight is that the reconstruction reward (Eq. 10) and regularization with next-token prediction (Eq. 24) are analogous to the goal of policy optimization and KL penalty in reinforcement learning in Eq. 5. Although the reconstruction reward and regularization term are independent of any specific reinforcement learning framework, we employ GRPO to follow practices [1, 52]. Specifically, the VA-π objective is to maximize:

JVA-π(θ) = E I∼p

data, x∗=Q(E(I)),

x˜∗∼pξ(·|x∗), {xi}Gi=1∼πθold(·|x˜∗)

G

1 G

min ρiAi, clip(ρi,1 − ϵ,1 + ϵ)Ai

i=1

− β Lprior(πθ,x∗,x˜∗) ,

(11) where pdata is the data distribution of reference images; G is the number of teacher-forced samples per instance; ρi = π

θ(xi|x˜∗)

πθold(xi|x˜∗) is the importance ratio; Ai is the advantage computed from R(xi,x˜∗) as defined in Eq. 10; ϵ is

- Table 2. Quantitative results on class-conditional ImageNet-1k [19]. We compare both LlamaGen-XL (775M) and LlamaGen-XXL (1.4B) models. All models are evaluated both with and without classifier-free guidance (CFG). Generated 384 × 384 images are resized to 256 × 256 for evaluation. Metrics include Fr´echet Inception Distance (FID), Inception Score (IS), Precision (Pre.) and Recall (Rec.). “Ext. Rwd” denotes the use of external reward during reinforcement learning fine-tuning. Our proposed VA-π achieves competitive diversity (FID) and perceptual quality (IS) with substantially lower training cost. Best FID and IS results are highlighted in blue .

Model Ext. Rwd Time (min)↓

w/o cfg w/ cfg FID↓ IS↑ Pre.↑ Rec.↑ FID↓ IS↑ Pre.↑ Rec.↑

LlamaGen-XL (775M) [20] – – 15.55 79.16 0.62 0.69 2.79 286.88 0.84 0.54 + AR-GRPO [1] ✓ 149 – – – – 3.63 293.07 0.86 0.48 + VA-π (Ours) × 20 9.23 111.59 0.71 0.59 2.94 299.63 0.84 0.53

LlamaGen-XXL (1.4B) [20] – – 14.36 86.55 0.63 0.69 2.37 252.16 0.81 0.59 + Post-train Tokenizer × 18 14.26 86.70 0.63 0.68 2.72 246.97 0.80 0.59 + Post-train Tokenizer (longer) × 207 22.99 72.49 0.56 0.68 4.31 221.57 0.75 0.58 + STE based Post-train AR [18] × 381 11.46 102.21 0.68 0.61 4.17 267.34 0.83 0.51 + VA-π (Ours) × 25 7.65 116.70 0.71 0.64 2.28 273.53 0.83 0.56

- Table 3. Quantitative results on the GenEval benchmark. The upper block reports performance of LlamaGen-XL (T2I visual generation model), and the lower block reports Janus Pro-1B (unified multi-modal model). The abbreviation ”Ext. Rwd” denotes ”External Reward”, ”Attr. Bind.” denotes ”Attribute Binding”, ”obj.” denotes ”Object”. VA-π improves over both LlamaGen-XL [20] and AR-GRPO [1], achieving the highest overall GenEval [54] score. When applied to the unified multimodal model Janus-Pro 1B [5], VA-π further enhances fine-grained attributes, demonstrating its generalization across model architectures. Best results are highlighted in blue .

Model Ext. Rwd Position↑ Color↑ Attr. Bind.↑ Counting↑ Single Obj.↑ Two Obj.↑ Overall↑

LlamaGen-XL [20] – 0.042 0.550 0.032 0.197 0.750 0.263 0.306 + AR-GRPO [1] ✓ 0.040 0.593 0.030 0.228 0.791 0.263 0.324 + VA-π (Ours) × 0.050 0.606 0.040 0.238 0.769 0.328 0.339

Janus-Pro 1B [48] - 0.605 0.902 0.540 0.531 0.972 0.801 0.725 + VA-π (Ours) × 0.600 0.912 0.585 0.540 0.988 0.835 0.744

the clipping parameter; β weights the regularization, and ξ controls the contextual noise level.

Discussion with AR-GRPO. While the motivation of VAπ differs from that of standard AR-GRPO [1], which aims to maximize an externally defined reward, our method offers several additional advantages: (1) Unlike the GRPO framework, which requires maintaining a reference model, the prior regularization in VA-π introduces no extra storage overhead; (2) VA-π avoids the cost of additional rollouts used in AR-GRPO, as all terms are derived from teacher-forcing trajectories, thereby improving training efficiency significantly. (3) Even without an external reward model, VA-π achieves better performance than GRPO models specifically trained to maximize these rewards as Tab. 1, highlighting the importance of pixel-level alignment.

##### 5. Experiments 5.1. Experimental Setup

To validate the effectiveness of VA-π, we evaluate it on two visual generation tasks: (i) class-conditioned image generation (C2I) and (ii) text-conditioned image generation (T2I).

For C2I, we adopt LlamaGen [20] as the base autoregressive (AR) generator. For T2I, we assess VA-π on both LlamaGen and Janus-Pro 1B [48], a unified multimodal model (UMM) capable of both understanding and generation.

In the C2I setting, we employ LlamaGen-XXL (1.4B) and LlamaGen-XL (775M), continuing training on the ImageNet-1k [19] dataset for 100 steps using 12.8K samples. Each image–text pair produces eight samples for group-wise policy optimization. All training is conducted without classifier-free guidance (CFG) to ensure diverse yet stable exploration. For the T2I setting, we further fine-tune LlamaGen-XL on the LAION-COCO [55] dataset for 200 steps, following the same setup as in C2I except for the noise perturbation ratio, detailed in Section 5.4. For the UMM architecture, we fine-tune Janus-Pro 1B on the FluxReason dataset [56] for 100 steps. Additional implementation details are provided in Appendix C.

Evaluation metrics. For C2I, we assess image fidelity and diversity using Fr´echet Inception Distance (FID) and Inception Score (IS). FID quantifies the distributional gap between real and generated images, directly reflecting the alignment objective of VA-π. For T2I, we adopt the

2025/11/18 16:21 Optimizing visual autoregressive generation with end-to-end reward.svg

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 3. Left: Qualitative comparison of C2I generation among LlamaGen-XL [20] (top), AR-GRPO [1] (middle) and VA-π (bottom) on the ImageNet-1k [19] classes. Both models use a CFG scale of 2.0. VA-π produces clearer object structures (like the car mirror) than LlamaGen-XL (top) and AR-GRPO (middle), demonstrating that pixel-space alignment encourages realistic generations. Right: Qualitative comparison of T2I generation between Janus-Pro 1B [48] and VA-π on the GenEval Benchmark [54]. Both models use a CFG scale of 5.0. VA-π produces better object combination and counting accuracy, demonstrating stronger capability.

GenEval benchmark [54], which evaluates models on six compositional dimensions: object relations, counting, spatial layout, color, shape, and attribute binding.

15× faster training time than STE. VA-π uniquely improves both fidelity (FID) and diversity (IS) simultaneously, unlike prior methods that enhanced one at the expense of the other. Except for the LlamaGen-XL with CFG variant (reasonable since VA-π is trained without CFG), it achieves the significant improvement with the lowest training cost.

###### 5.2. Main Results on Class-to-Image Generation

We evaluate VA-π on the C2I generation task using ImageNet-1k’s validation set (50,000 images). As shown in Tab. 2, VA-π is compared with strong baselines built upon the LlamaGen architecture: (i) AR-GRPO [1]: RL finetuned method that aligns the model using multiple external reward models; (ii) Tokenizer-centric approaches: We post-train the tokenizer for 100 steps as VA-π and 10,000 steps (longer) on ImageNet-1k [19]; (iii) Generator-centric approaches: We facilitate gradients flowing back to the generator na¨ıvely using STE [10].

###### 5.3. Main Results on Text-to-Image Generation

We further evaluate VA-π on the text-to-image (T2I) generation task using the GenEval [54] benchmarks. As shown in Table 3, we implement our post-train method on both the visual generation model LlamaGen-XL [20] and the unified multi-modal model Janus-Pro 1B [48]. VA-π achieves consistent improvements, while it does not train on any textalignment or human preference rewards.

file:///E:/CCIIP实验室/论⽂/CVPR 2026/Optimizing visual autoregressive generation with end-to-end reward.svg 1/1

Results on LlamaGen-XL. On GenEval, our method outperforms AR-GRPO across most sub-tasks, improving the overall score (0.324 → 0.339), improved by 0.015. Clear gains are observed on semantically complex prompts like color understanding (+0.013), counting (+0.010), and twoobject composition (+0.065). Furthermore, we evaluate VA-π using CLIP [57] on Table 1 and HPS v2 [58] with DrawBench [59] prompts. VA-π outperforms AR-GRPO even without explicit fine-tuning on these evaluation metrics, whereas AR-GRPO requires task-specific adaptation. This demonstrates the strong generalization ability achieved by our pixel-level alignment strategy.

Post-trained based on LlamaGen-XL (775M), VA-π delivers remarkable gains without classifier-free guidance, boosting both FID (15.55→9.23) and IS (79.16→111.59). With classifier-free guidance (scale = 2.0), it achieves the highest IS of 299.63, surpassing AR-GRPO while requiring no external reward model and 7.5×faster training (20 minutes). For the larger LlamaGen-XXL (1.4B), VA-π reduces ∼50%FID (14.35→7.65) and increases IS by 30.16 with only 25 minutes of post-training, outperforming naive posttraining strategies such as post-train tokenizer and post-train tokenizer using STE. With classifier-free guidance (scale = 1.75), it achieves the best FID of 2.28 and IS of 273.53, with

[Figure 20]

[Figure 21]

(a) FID over weight β (b) IS over weight β

- Figure 4. Ablation on regularization weight (w/o cfg). CE regularization consistently outperforms KL regularization on FID and IS. Moderate CE regularization (0.1) provides the best results.

Results on Janus-Pro 1B. When applied to the unified multi-modal model, VA-π further enhances visual compositionality and semantic grounding, raising the overall score from 0.725 to 0.744. Improvements are particularly prominent in attribute binding (+0.045) and two-object relations (+0.034), indicating that VA-π generalizes effectively to large multi-modal systems. This suggests that the proposed alignment objective provides a scalable mechanism for bridging token-level and perceptual-level consistency in text-conditioned generation.

- 5.4. Ablation Study

We conduct extensive ablation studies to examine the effectiveness of each component in VA-π, focusing on reward composition, prior regularization, and Contextual Noise.

Reward and Loss Composition. Table 4 investigates how different reward components (in Sec. 4.3) influence training dynamics and generation quality. Using only the reconstruction reward (Lp / LMSE) as in Eq. 10 fails to provide meaningful alignment due to drifting away from the pre-trained AR token distribution without prior regularization constraints as in Eq. 8. Incorporating the prior regularization term as an auxiliary objective (i.e., crossentropy loss) significantly stabilizes optimization by maintaining the token-level likelihood that was already learned in the original AR model. The full combination achieves the best balance. This demonstrates that both reconstruction rewards and the prior regularization term are essential for generator–tokenizer consistency.

Prior Regularization Term. Fig. 4 analyzes the effect of regularization strength (in Sec. 4.2) across KL- and CEbased variants. the regularization strength is defined by β as in Eq. 11 Here, KL refers to Kullback–Leibler divergence regularization, which penalizes deviations from the base policy distribution, while CE denotes cross-entropy regularization, which enforces consistency with target token probabilities. Without regularization, optimization diverges rapidly, yielding poor fidelity (FID 38.63). Moderate weights (β = 0.1) effectively constrain policy updates, improving both FID and IS. Excessively strong regularization

- Table 4. Ablation on reward composition (w/o CFG). We analyze the contribution of each reward component: LMSE (pixellevel reconstruction), Lp (perceptual similarity via LPIPS [50]), and Lprior (token-level cross-entropy regularization).

LMSE Lp Lprior FID↓ IS↑ Pre.↑ Rec.↑

14.36 86.55 0.63 0.69 ✓ 38.76 49.78 0.48 0.46 ✓ ✓ 38.63 48.14 0.49 0.46 ✓ 14.17 88.78 0.63 0.69 ✓ ✓ ✓ 7.65 116.70 0.68 0.64

- Table 5. Ablation on noise ratio (ξ) during training. Moderate noise ratio (0.5) achieves the best overall performance on GenEval. Abbreviations: PT (Position), CL (Color), AB (Attribute Binding), CT (Counting), SO (Single Object), TO (Two objects).

ξ PT CL AB CT SO TO Overall 0 0.048 0.566 0.023 0.159 0.688 0.326 0.302

0.25 0.043 0.598 0.025 0.215 0.700 0.306 0.315

0.5 0.050 0.606 0.040 0.238 0.769 0.328 0.339 0.75 0.075 0.641 0.028 0.163 0.750 0.333 0.332 0.95 0.043 0.652 0.040 0.181 0.728 0.328 0.329

(β = 1.0) oversmooths gradients and suppresses diversity. Notably, CE regularization outperforms KL regularization under the same setting. This confirms that lightweight CE regularization (β = 0.1) achieves an optimal trade-off between stability and expressiveness.

Contextual Noise. Table 5 examines the impact of stochastic contextual noise to mitigate exposure bias during policy updates (in Sec. 4.2). We ablate the corruption probability ξ (in Eq. 9) from 0 to 0.95 in the LlamaGen T2I post-train setting. Injecting a moderate amount of noise (ξ = 0.5) yields the best overall performance on GenEval (Overall 0.339). In contrast, either no noise (ξ = 0) or excessive perturbation (ξ > 0.75) leads to suboptimal results.

- 6. Conclusion

We propose VA-π, a principled RL–based post-training framework that aligns visual AR generators to pixel space through a variational objective grounded in probabilistic modeling. It substantially improves visual fidelity and compositional alignment on both class-conditional and textconditional generation tasks, while reducing training cost by 86.6% compared to conventional RL fine-tuning. The framework also generalizes effectively to large unified multimodal models such as Janus-Pro 1B. Overall, VA-π provides a lightweight and theoretically grounded path toward bridging token-level modeling and pixel-level generation in a scalable multimodal generation scenario.

#### VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation Supplementary Material

##### A. Theoretical Details of VA-π

In this section, we (1) restate the proof of the ELBO with respect to the random variable and posterior defined in Sec. A.1; (2) provide additional insights into its relationship with the VAE [10] and autoregressive (AR) models based on VQVAE [18] in Sec. A.2; and (3) further justify the formulation of the prior regularization term in Sec. A.3.

###### A.1. Proof of Alignment ELBO

While the proof of ELBO is already provided in existing literature such as VAE [10]. We restate it with posterior based on discrete token sequence as latent variable.

Setup. Let I denote the observed image and x ∈ X the discrete token sequence generated from autoregressive model. We assume a generative model with parameters θ as:

###### pθ(I,x) = pθ(I | x)p(x), (12)

where p(x) is a prior over token sequences (e.g., the AR prior) and pθ(I | x) is a pixel-space likelihood (e.g., the tokenizer decoder composed with a reconstruction distribution). Our training objective is the marginal log-likelihood:

pθ(I,x) (integral for continuous x),

log pθ(I) = log

x∈X

(13) which is generally intractable to evaluate directly because summing/integrating over x is prohibitive.

Introducing a variational posterior. Let qϕ(x | I) be any distribution supported on X (in practice, produced by teacher forcing so that it is easy to sample from and to evaluate). Multiply and divide the integrand in (13) by qϕ(x | I):

log pθ(I) = log

x

pθ(I,x) qϕ(x | I)

. (14)

qϕ(x | I)

Jensen’s inequality. We now apply Jensen’s inequality to the concave function log(·):

log Eq

ϕ

f(x) ≥ Eq

ϕ

log f(x) (for f(x) > 0).

θ(I,x)

Using f(x)= p

qϕ(x|I) in (14) gives:

pθ(I,x) qϕ(x | I)

(15)

log pθ(I) ≥

qϕ(x | I)log

x

= Eq

ϕ(x|I) log pθ(I,x) − Eq

ϕ(x|I) log qϕ(x | I) (16)

= Eq

ϕ(x|I) log pθ(I | x) + Eq

ϕ(x|I) log p(x)

(17) − Eq

ϕ(x|I) log qϕ(x | I) (18)

= Eq

ϕ(x|I) log pθ(I | x) − KL(qϕ(x | I)∥p(x)).

(19)

We define the right-hand side of (19) as the evidence lower bound (ELBO):

L(θ,ϕ;I) ≜ Eq

ϕ(x|I) log pθ(I | x) −KL(qϕ(x | I)∥p(x)),

(20) so that log pθ(I) ≥ L(θ,ϕ;I).

Optimization. When qϕ(x | I) perfectly matches the true posterior pθ(x | I), the KL term becomes zero, and maximizing the ELBO is equivalent to maximum likelihood estimation (MLE) as proved in existing literature [10, 18, 60].

###### A.2. Comparison with VAE and AR on VQVAE

Our theoretical framework can be justified from the perspective of VAE and AR based on VQVAE. Notice that variational optimization is independent of the choice of the latent variable, posterior and prior. Specifically, the ELBO of VA-π can be seem as a variant of VQVAE’s ELBO with redefined prior and posterior as shown in Tab 6.

VAE [10] treats the latent variable is continuous feature, with prior in standard Gaussian distribution and the posterior is parameterized by the encoder. During sampling, the model firstly sample noise from standard Gaussian distribution and forward it to decoder to get the generated image.

VQVAE used in standard visual AR generation [12, 18] instead treats the latent variable as a discrete token sequence, consistent with our formulation. The key distinction, however, lies in how the posterior and prior are defined. In VQVAE, the posterior is determined solely by the encoder and quantizer: given a reference image, the model produces a deterministic token sequence that can be viewed as sampling from a delta distribution, i.e., qϕ(x | I) is one-hot. The prior, in contrast, is assumed to follow a uniform categorical distribution. Under these assumptions, the KL divergence term becomes a constant independent of learnable parameters and is thus omitted from the objective. However, since the mismatch exists between the assumed uni-

Formulation q(x | I) (Posterior) p(x) (Prior) Trainable Modules Objective VAE Continuous Gaussian Distribution Gaussian N(0,I) Encoder, Decoder Reconstruction + Generation VQVAE Categorical Distribution (Dirac) Uniform categorical Encoder, Decoder, Codebook Reconstruction AR on VQVAE Categorical Distribution (Dirac) AR model AR model Generation VA-π (Ours) Teacher-forced Posterior via AR AR model AR model Reconstruction + Generation

- Table 6. Comparison of probabilistic formulations among VAE, VQVAE, VQVAE + AR, and the proposed VA-π. Our method redefines the posterior through teacher-forced AR modeling, maintaining ELBO validity while enabling post-training alignment between the AR generator and the tokenizer.

form prior and the trained empirical posterior, it’s challenging to generate realistic samples by decoding directly from the uniform prior. To bridge this gap, an AR model is subsequently trained to learn a more accurate prior over the token space, replacing the uniform assumption in VQVAE during sampling. This two-stage design inherently creates a gap between the AR model and the original VQVAE tokenizer.

While such gap is difficult to be resolved in the pretrained-stage due to the challenging optimization over non-differentiable discrete latent variable, we consider that it can be solved in the post-training settings. Given pretrained autoregressive model, we can reuse it as the prior and redefine the posterior to introduce sampling.

VA-π also represents the latent variable as a discrete token sequence. We redefine the posterior as follows: given a reference image, the encoder and quantizer are used to obtain a deterministic code, similar to the standard VQVAE formulation. However, unlike VQVAE, we further teacher-force this code into the AR model to compute a categorical distribution. This design keeps the ELBO theoretically valid, as the bound holds regardless of the specific choice of posterior, while offering two practical advantages. (1) Since the AR model is incorporated into the posterior, it receives direct supervision from the reconstruction term defined in pixel space. (2) It makes the KL regularization more interpretable: the KL divergence between the teacher-forcing distribution and the free-running distribution corresponds to the exposure bias, which can be reduced through next-token prediction under noisy ground-truth contexts, as shown in prior works [13, 14]. Redefining the posterior enables posttraining of the AR model to better align with the tokenizer.

###### A.3. Details of Prior Regularization

To mitigate the inconsistency between the teacher-forced distribution (conditioned on a dataset) and the free-running distribution, we consider a regularization objective that directly addresses the exposure bias problem. While nexttoken prediction under noisy context has been shown to address exposure bias effectively in existing works [13, 61], we provide theoretical formulation below.

Next Token Prediction Regularization. We define the regularization objective as maximizing:

−KL(qϕ,θ(x | I)∥πθ(x)) w.r.t. θ,

since the AR model used for teacher forcing is parameterized by the same model, given x∗ ∼ Q(Eϕ(I)), the objective is equivalent to minimizing:

Lprior(θ) = KL(πθ(x | x∗)∥πθ(x)). (21)

where the AR model πθ factorizes as:

N

πθ(xt | x<t),

πθ(x) =

t=1

N

πθ(xt | x∗<t).

πθ(x | x∗) =

t=1

(22)

For any AR laws P(x) = t P(xt | x<t) and Q(x) = t Q(xt | x<t), the chain rule for KL gives:

N

Ex

KL(P∥Q) =

<t∼P KL P(·|x<t)∥Q(·|x<t) .

t=1

(23) We also use the cross-entropy decomposition KL(p∥q) = H(p,q) − H(p), with H(p,q) := −Ey∼p[log q(y)].

Applying (23) to πθ(x | x∗) vs. πθ(x),

N

Lprior(θ) =

t=1

N

=

t=1

θ(·|x∗) KL πθ(· | x∗<t)∥πθ(· | x<t)

Ex∼π

θ(·|x∗) H πθ(· | x∗<t),πθ(· | x<t)

Ex∼π

− H πθ(· | x∗<t) .

(24) We make two assumptions:

- A1 (Teacher-forced calibration) πθ(· | x∗<t) = p∗(· | x∗<t) for all t,x∗<t, given πθ is exactly pretrained on this.
- A2 (Prefix-matching corruption) There exists a Markov kernel Kξ(˜x<t | x∗<t) such that, when x ∼ πθ(· | x∗),

p(x<t | x∗) = Kt(· | x∗<t) for all t. (25)

Using (25) to replace the expectation over x<t in (24) by

an expectation over x˜<t ∼ Kt(· | x∗<t), and applying A1,

N

<t∼Kt(·|x∗<t)Ey∼p∗(·|x∗<t) − log πθ(y | x˜<t)

Ex˜

Lprior(θ) =

t=1

N

H p∗(· | x∗<t) .

−

t=1

(26)

Let C := t H(p∗(· | x∗<t)), which is independent of θ. Then:

reconstruction ˆI:

LMSE = ∥I − ˆI∥22. (31)

Perceptual reconstruction loss. Following [50], a perceptual loss Lp compares high-level feature activations of I and ˆI extracted by a pretrained VGG network:

Lp =

l

∥ϕl(I) − ϕl(ˆI)∥22, (32)

N

Ex˜

<t∼KtEy∼p∗(·|x∗<t) − log πθ(y | x˜<t)

Lprior(θ) =

t=1

LNTP-noisy(θ) − C.

(27) Hence, minimizing the prior KL is equivalent up to a constant to minimizing the next-token prediction (NTP) loss under perturbed prefixes x˜<t. While the assumption is not guaranteed in the experimental settings, we found it works empirically well by choosing proper corruption kernel.

Corruption Kernel Details. Following previous work [13], we use uniform noise to implement the corruption kernel. Given a discrete sequence x∗ = (x∗1,...,x∗N), we define a corruption kernel Kξ(xˆ | x∗) that introduces random perturbations with rate ξ ∈ [0,1]. For each position i ∈ {1,...,N}, we independently draw:

x∗i , with probability 1 − ξ, ui, with probability ξ,

(28)

xˆi =

where ui is sampled uniformly from the token vocabulary V = {1,...,K} excluding the ground-truth token, i.e. ui∼ Unif(V \{x∗i }). Equivalently, the conditional probability mass function is:

1[ˆxi̸=x∗i ] K − 1

Kξ(ˆxi | x∗i ) = (1 − ξ)δ(ˆxi=x∗i ) + ξ

, (29)

and the full sequence corruption factorizes as Kξ(xˆ | x∗) = N i=1 Kξ(ˆxi | x∗i ).

##### B. Tokenizer Training Objectives

The visual tokenizer is trained to map continuous image features into a compact set of discrete embeddings while maintaining perceptual reconstruction quality. As discussed in Sec. 3, the total objective in Eq. 2 consists of reconstruction and quantization terms:

Ltok = LMSE + λpLp + λqLq. (30)

Pixel-wise reconstruction loss. We employ the mean squared error (MSE) between the original image I and its

where ϕl(·) denotes features from the l-th layer.

Vector quantization loss. The quantization loss [18] encourages the encoder output z to commit to the closest embedding vector in the codebook E = {ek}Kk=1:

Lq = ∥sg[z] − e∥22 + β∥z − sg[e]∥22, (33)

where sg[·] denotes the stop-gradient operator and β controls the commitment strength. The first term updates the codebook entries to match encoder outputs, while the second prevents codebook collapse by penalizing large deviations of z from its assigned embedding.

Together, these terms ensure high-fidelity reconstruction, perceptual realism, and stable codebook learning. The resulting tokenizer provides discrete tokens that effectively balance compression and visual quality for subsequent autoregressive modeling.

##### C. Implementation Details

We first present the detailed formulation of baselines for mitigating discrepancy between decoding the AR-generated token sequences and the ground-truth image distribution by fine-tuning the AR model by STE algorithm C.1 or finetuning the tokenizer decoder C.2. Then, we present the implementation details of our VA=π on three experimental setting ref: LlamaGen for C2I and T2I, Janus-Pro for T2I.

###### C.1. Straight-Throught Estimator Variant

Autoregressive (AR) image generators operate over discrete token sequences produced by a visual tokenizer. Because the token selection process (e.g., arg max over logits) is non-differentiable, directly optimizing pixel-level objectives is intractable. To enable gradient-based training, we employ the Straight-Through Estimator (STE) [18], which provides a differentiable surrogate for the discrete sampling process of the AR model.

Forward pass. Given an image I, the tokenizer encodes it into discrete token indices x∗ = Q(E(I)). During training, the AR generator πθ predicts the next-token logits lt ∈ RV conditioned on the teacher-forced prefix x∗<t:

###### lt = πθ(x∗<t).

A temperature-scaled softmax produces a differentiable relaxation of the categorical distribution:

###### ysoft = softmax(lt/τ),

and a discrete token is sampled by a hard one-hot projection by arg max as:

###### yhard = onehot(arg max(lt)).

The final surrogate variable used for decoding is as:

yst = (yhard − ysoft).detach() + ysoft,

so that the forward path uses yhard (discrete tokens), while the backward path reuses the gradient of ysoft.

Backward pass. Let ek denote the codebook embedding corresponding to token index k. In the forward pass, yhard selects the embedding ek∗ from the codebook to form the quantized representation zq. During back-propagation, the STE treats this quantization step as an identity mapping:

∂ysoft ∂lt

∂zq ∂lt ≈

,

which effectively copies the gradient from the decoder output D(zq) to the generator logits lt. This allows the AR generator to receive pixel-space reconstruction gradients through the frozen tokenizer without requiring a differentiable relaxation of the discrete sampling.

The STE thus provides a simple yet effective mechanism for propagating pixel-level feedback to the AR generator. In our implementation, it is applied consistently during the optimization of both reconstruction and regularization objectives. While the forward path remains discrete for accurate decoding, the backward path transmits continuous gradients to stabilize learning. This approximation has been widely adopted in discrete generative models [18] and is crucial for enabling end-to-end optimization. However, STE-based fine-tuning requires more data and training time cost compred to our proposed VA-π for the model only update under the ground-truth path.

###### C.2. Tokenizer Post-training

A tokenizer trained only on ground-truth image reconstructions may not fully account for the distribution shift induced by AR-generated token sequences. From the tokenizer’s perspective, enhancing robustness to such off-manifold or suboptimal token sequences provides an additional means to reduce the mismatch between AR-generated token distributions and the underlying image distribution.

Formulation. Recall that the tokenizer consists of an encoder E, a quantizer Q, and a decoder Dϕ. Given an image I ∈ R3×H×W, the tokenizer produces its latent representation and discrete token sequence:

z = E(I), x∗ = Q(z), (34)

where z ∈ RC×N and x∗ ∈ {1,...,K}N. The sequence x∗ serves as the teacher-forcing target for the AR model, whose parameters are trained by maximizing:

N

θ = arg max

θ

i=1

log πθ(x∗i | x∗1:i−1). (35)

During tokenizer post-training, we keep the AR model πθ, the quantizer Q, and its codebook fixed. Given a teacher-forcing sequence x∗, the AR model generates its own token predictions autoregressively:

xi ∼ πθ(· | x∗1:i−1), x = (x1,...,xN), (36)

which reflect the model’s free-running token distribution conditioned on x∗. The AR-generated sequences x are decoded by the tokenizer decoder:

ˆI = Dϕ(x), (37)

and the decoder parameters ϕ are updated to improve the reconstruction fidelity of AR-generated tokens. Because the quantizer Q and the AR model πθ remain frozen, the teacher-forcing token distribution x∗ is preserved, ensuring compatibility with the pretrained AR model while enhancing robustness to its sampled token sequences.

Objective. Unlike the full tokenizer objective in Eq. 2, the post-training objective excludes the quantization loss and focuses solely on reconstruction fidelity:

LPT = ∥ˆI − I∥22 + λp LLPIPS(ˆI,I). (38)

This updates only the reconstruction pathway without altering the discrete token space.

The improved tokenizer thus serves as a drop-in replacement that yields higher-fidelity decoding for both groundtruth and AR-generated token sequences without inducing any distributional mismatch.

###### C.3. Implementaton Details of VA-π

We summarize the hyperparameters used for RL fine-tuning across all models and tasks in Table 7. For both C2I and T2I experiments with LlamaGen, we adopt the same optimizer configuration—AdamW with a learning rate of 1 × 10−6, weight decay of 1 × 10−4, bf16 mixed precision, and a global batch size of 128. For Janus-Pro 1B, we use a smaller learning rate (1 × 10−7) due to its increased sensitivity during RL updates. All models are trained with group size G=8, advantage clipping at 5.0, a classifier-free guidance scale of 1.0, and a maximum gradient norm of 1.0.

We fine-tune LlamaGen-XXL for 100 steps on C2I and LlamaGen-XL for 200 steps on T2I, while Janus-Pro 1B is trained for 100 steps given its heavier multimodal architecture. Image resolution is set to 384 × 384 for all LlamaGen models and 256 × 256 for Janus-Pro 1B.

[Figure 22]

[Figure 23]

[Figure 24]

(a) (b) (c)

- Figure 5. Learning curves of our reinforcement-learning framework VA-π across three model scenarios. (a) C2I (LlamaGen-XXL, 100 steps), (b) T2I (LlamaGen-XL, 500 steps), and (c) T2I (Janus-Pro 1B, 500 steps).

- Table 7. Hyperparameters used for RL fine-tuning across three model settings. LlamaGen-XXL (C2I) and LlamaGen-XL (T2I) use consistent optimization settings, while Janus-Pro 1B adopts a smaller learning rate and lower resolution due to its multimodal architecture and training stability considerations.

Name LlamaGen for C2I LlamaGen for T2I Janus-Pro 1B for T2I

Learning Rate 1e-6 1e-6 1e-7 Weight Decay 1e-4 1e-4 1e-4 Mixed Precision bf16 bf16 bf16 Beta β 0.1 0.1 0.1 Group Size G 8 8 8 Max Advantage Clip 5.0 5.0 5.0 Classifier-Free Guidance Scale 1.0 1.0 1.0 Max Gradient Norm 1.0 1.0 1.0 Total Batchsize per Step 128 128 128 Training Steps 100 200 100 Image Resolution h × w 384 × 384 256 × 256 384 × 384 Contextual Noise ξ 0 0.5 0.95

##### D. Additional Qualitative Results

###### D.1. Visualization of Learning Curves

We visualize the learning curves during reinforcement learning fine-tuning, showing how the reward evolves as training progresses. As illustrated in Fig. 5, the reward steadily increases across all settings, indicating that the policy consistently improves under our VA-π framework. These curves confirm that our reinforcement learning framework enables robust policy improvement across generators of varying scales and modalities.

###### D.2. Class-to-Image Generation

We present additional qualitative comparisons on the classto-image (C2I) generation task, showcasing examples from various ImageNet [19] classes. Each comparison visualizes generations from the baseline LlamaGen-XXL [20] and our VA-π, which applies reinforcement learning post-training on LlamaGen-XXL. All samples are generated under identical decoding configurations with CFG scale = 1.0, temper-

ature = 1.0, top-k = 0, and top-p = 1.0.

We also present additional qualitative comparisons on fine-tuning methods like STE [12] and post-train tokenizer

- as shown in Fig. 6. We further analyze the impact of long-term tokenizer de-

coder post-training. Surprisingly, although reconstruction loss steadily decreases during training, both FID and IS consistently worsen. This degradation arises because the decoder, trained with teacher-forced ground-truth tokens, gradually learns an overly smooth reconstruction mapping

- as shown in Fig. 7. It becomes tolerant to token inaccuracies, suppresses high-frequency textures, and specializes in cleaning input tokens that do not match the noisy AR tokens produced during inference. As a result, images retain coarse structure but lose sharpness and perceptual richness.

These observations highlight the inherent limitations of decoder-only fine-tuning: it cannot correct off-manifold token sequences, amplifies the train–inference mismatch, and ultimately smooths away meaningful details. This reinforces the central design choice of our method—fine-tuning

Table 8. Representative prompts used for qualitative comparison on GenEval tasks. Task Prompt Attribute Binding

# Visualization of Post-Train Tokenizer

“a photo of a brown laptop and a white bicycle” “a photo of a black cow and an orange donut” “a photo of a blue baseball glove and a black pizza” “a photo of a purple chair and a brown surfboard”

“a photo of two cups” “a photo of three cell phones” “a photo of three wine glasses” “a photo of four birds”

### ● The post-trained decoder decoding images with the same structure and layout but more smooth and sharpless texture, which results in worse FID and IS.

Counting

“a photo of a sports ball right of a handbag” “a photo of a handbag above a sheep” “a photo of a skateboard right of a fire hydrant” “a photo of a computer keyboard right of skis”

Position

“a photo of a frisbee and a bottle” “a photo of a book and a tv” “a photo of a skateboard and a baseball glove” “a photo of a chair and a tv”

Two-Object Combination

Stin ray Junco Lesser panda Grand piano

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

LlameGen-XXLPTSTEVA-π

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### Be ore A ter Be ore A ter Be ore A ter

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Figure 7. Observation on Post-Train Tokenizer. Post-training the tokenizer decoder produces smoother and less detailed textures, despite preserving global structure. This over-smoothing effect explains why extended decoder fine-tuning leads to worse FID (14.36 → 22.99) and IS (86.55 → 72.49), highlighting the inherent limitation of decoder-only fine-tuning.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

qualitative results highlight that pixel-level reward alignment is essential for correcting sampling errors that STE alone cannot address.

- Figure 6. Qualitative comparison of C2I generation among LlamaGen-XXL [20], post-train tokenizer (PT), STE based post-train AR and VA-π on the ImageNet-1k [19] classes. Both models use a CFG scale of 1.0. VA-π shows better semantic alignment and image quality, demonstrating that pixel-space alignment encourages realistic generations.

Overall, VA-π produces images that are more consistent with class semantics and exhibit improved structural fidelity and perceptual realism compared to the pre-trained LlamaGen-XXL baseline.

the AR generator itself is necessary to align the generated token distribution with the ground-truth image manifold.

Post-training the tokenizer (PT) preserves coarse structure but produces overly smooth textures, while STE-based AR fine-tuning partially improves details yet still suffers from off-manifold token transitions. This is because STE only updates the likelihood of teacher-forced tokens and cannot adjust the AR model’s sampling behavior. In contrast, our VA-π optimizes pixel-space rewards for sampled token sequences, yielding sharper textures, more coherent semantics, and overall more realistic generations. These

###### D.3. Text-to-Image Generation

We present additional qualitative comparisons on the textto-image (T2I) generation task, showcasing examples from various prompts from the the GenEval benchmark [54]. Especially, we present images generated from complex tasks: attribute binding, counting, position, and two-object combination. Each comparison visualizes generations from the unified multi-modal baseline Jans-Pro 1B [48] and our VA-π, which applies reinforcement learning post-training on Jans-Pro 1B. All samples are generated under identical decoding configurations with CFG scale = 5.0, temperature = 1.0, top-k = 0, and top-p = 1.0.

[Figure 47]

Figure 8. Qualitative comparison on the kite class.

[Figure 48]

- Figure 9. Qualitative comparison on the English foxhound class.

[Figure 49]

- Figure 10. Qualitative comparison on the Egyptian cat class.

[Figure 50]

Figure 11. Qualitative comparison on the dome class.

[Figure 51]

Figure 12. Qualitative comparison on the thresher / thrasher / threshing machine class.

[Figure 52]

Figure 13. Qualitative comparison on the cheese burger class.

[Figure 53]

Figure 14. Qualitative comparison on the pineapple / ananas class.

[Figure 54]

Figure 15. Qualitative comparison on the bolete class.

[Figure 55]

Figure 16. Qualitative comparison on the attribute binding task.

[Figure 56]

- Figure 17. Qualitative comparison on the counting task.

[Figure 57]

- Figure 18. Qualitative comparison on the position task.

[Figure 58]

Figure 19. Qualitative comparison on the two-object combination task.

##### References

- [1] S. Yuan, Y. Liu, Y. Yue, J. Zhang, W. Zuo, Q. Wang, F. Zhang, and G. Zhou, “Ar-grpo: Training autoregressive image generation models via reinforcement learning,” 2025. 2, 3, 4, 5, 6, 7
- [2] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023. 1
- [3] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023. 1
- [4] Y. Bai, X. Geng, K. Mangalam, A. Bar, A. L. Yuille, T. Darrell, J. Malik, and A. A. Efros, “Sequential modeling enables scalable learning for large vision models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22861–22872, 2024. 1
- [5] C. Team, “Chameleon: Mixed-modal early-fusion foundation models,” arXiv preprint arXiv:2405.09818, 2024. 6
- [6] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, et al., “Scaling instruction-finetuned language models,” Journal of Machine Learning Research, vol. 25, no. 70, pp. 1–53, 2024.
- [7] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, G. Shi, and H. Fan, “Emerging properties in unified multimodal pretraining,” arXiv preprint arXiv:2505.14683, 2025.
- [8] C. Ma, Y. Jiang, J. Wu, J. Yang, X. Yu, Z. Yuan, B. Peng, and X. Qi, “Unitok: A unified tokenizer for visual generation and understanding,” arXiv preprint arXiv:2502.20321, 2025. 3
- [9] J. Xiu, F. Hong, Y. Li, M. Li, W. Wang, S. Han, L. Pan, and Z. Liu, “Egotwin: Dreaming body and view in first person,” arXiv preprint arXiv:2508.13013, 2025. 1
- [10] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013. 1, 4, 5, 7
- [11] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021. 1, 3
- [12] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021. 1, 3, 5
- [13] Q. He, Y. Li, H. Ye, J. Wang, X. Liao, P.-A. Heng, S. Ermon, J. Zou, and A. Yao, “Rear: Rethinking visual autoregressive models via generator-tokenizer consistency regularization,” arXiv preprint arXiv:2510.04450, 2025. 2, 3, 5
- [14] K. Qiu, X. Li, H. Chen, J. Kuen, X. Xu, J. Gu, Y. Luo, B. Raj, Z. Lin, and M. Savvides, “Image tokenizer needs post-training,” 2025. 2, 3
- [15] Z. Pang, T. Zhang, F. Luan, Y. Man, H. Tan, K. Zhang, W. T. Freeman, and Y.-X. Wang, “Randar: Decoder-only autoregressive visual generation in random orders,” in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 45–55, 2025. 2, 3

- [16] Q. Yu, J. He, X. Deng, X. Shen, and L.-C. Chen, “Randomized autoregressive visual generation,” arXiv preprint arXiv:2411.00776, 2024. 2, 3
- [17] B. Zheng, N. Ma, S. Tong, and S. Xie, “Diffusion transformers with representation autoencoders,” 2025. 2
- [18] A. Van Den Oord, O. Vinyals, et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017. 2, 3, 5, 6, 1, 4
- [19] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255, Ieee, 2009. 2, 6, 7, 5
- [20] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan, “Autoregressive model beats diffusion: Llama for scalable image generation,” arXiv preprint arXiv:2406.06525, 2024. 2, 3, 6, 7, 5
- [21] K. Gregor, I. Danihelka, A. Mnih, C. Blundell, and D. Wierstra, “Deep autoregressive networks,” in International Conference on Machine Learning, pp. 1242–1250, PMLR, 2014.
- [22] A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves, et al., “Conditional image generation with pixelcnn decoders,” Advances in neural information processing systems, vol. 29, 2016.
- [23] A. Van Den Oord, N. Kalchbrenner, and K. Kavukcuoglu, “Pixel recurrent neural networks,” in International conference on machine learning, pp. 1747–1756, PMLR, 2016.
- [24] N. Parmar, A. Vaswani, J. Uszkoreit, L. Kaiser, N. Shazeer, A. Ku, and D. Tran, “Image transformer,” in International conference on machine learning, pp. 4055–4064, PMLR, 2018.
- [25] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, “Generative pretraining from pixels,” in International conference on machine learning, pp. 1691–1703, PMLR, 2020. 3
- [26] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman, “Maskgit: Masked generative image transformer,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11315–11325, 2022. 3
- [27] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, V. Birodkar, A. Gupta, X. Gu, et al., “Language model beats diffusion–tokenizer is key to visual generation,” arXiv preprint arXiv:2310.05737, 2023. 3
- [28] M. Weber, L. Yu, Q. Yu, X. Deng, X. Shen, D. Cremers, and L.-C. Chen, “Maskbit: Embedding-free image generation via bit tokens,” arXiv preprint arXiv:2409.16211, 2024. 3
- [29] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023. 3
- [30] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. VandenEijnden, and S. Xie, “Sit: Exploring flow and diffusionbased generative models with scalable interpolant transformers,” in European Conference on Computer Vision, pp. 23– 40, Springer, 2024.
- [31] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, “Representation alignment for generation: Training diffusion transformers is easier than you think,” arXiv preprint arXiv:2410.06940, 2024. 3

- [32] F. Mentzer, D. Minnen, E. Agustsson, and M. Tschannen, “Finite scalar quantization: Vq-vae made simple,” arXiv preprint arXiv:2309.15505, 2023. 3
- [33] T. Li, Y. Tian, H. Li, M. Deng, and K. He, “Autoregressive image generation without vector quantization,” Advances in Neural Information Processing Systems, vol. 37, pp. 56424– 56445, 2024. 3
- [34] Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L.C. Chen, “An image is worth 32 tokens for reconstruction and generation,” Advances in Neural Information Processing Systems, vol. 37, pp. 128940–128966, 2024. 3
- [35] K. Miwa, K. Sasaki, H. Arai, T. Takahashi, and Y. Yamaguchi, “One-d-piece: Image tokenizer meets quality-controllable compression,” arXiv preprint arXiv:2501.10064, 2025.
- [36] K. Sargent, K. Hsu, J. Johnson, L. Fei-Fei, and J. Wu, “Flow to the mode: Mode-seeking diffusion autoencoders for state-of-the-art image tokenization,” arXiv preprint

- arXiv:2503.11056, 2025.

[37] T. Xiong, J. H. Liew, Z. Huang, J. Feng, and X. Liu, “Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation,” arXiv preprint

- arXiv:2504.08736, 2025. 3

- [38] P. Wu, K. Zhu, Y. Liu, L. Tang, J. Yang, Y. Peng, W. Zhai, Y. Cao, and Z.-J. Zha, “Alitok: Towards sequence modeling alignment between tokenizer and autoregressive model,” arXiv preprint arXiv:2506.05289, 2025. 3
- [39] R. Bachmann, J. Allardice, D. Mizrahi, E. Fini, O. F. Kar, E. Amirloo, A. El-Nouby, A. Zamir, and A. Dehghan, “Flextok: Resampling images into 1d token sequences of flexible length,” in Forty-second International Conference on Machine Learning, 2025.
- [40] K. Pan, W. Lin, Z. Yue, T. Ao, L. Jia, W. Zhao, J. Li, S. Tang, and H. Zhang, “Generative multimodal pretraining with discrete diffusion timestep tokens,” in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26136–26146, 2025. 3
- [41] D. Jiang, Z. Guo, R. Zhang, Z. Zong, H. Li, L. Zhuo, S. Yan, P.-A. Heng, and H. Li, “T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot,” arXiv preprint arXiv:2505.00703, 2025. 3
- [42] K. Zhang, Y. Zuo, B. He, Y. Sun, R. Liu, C. Jiang, Y. Fan, K. Tian, G. Jia, P. Li, et al., “A survey of reinforcement learning for large reasoning models,” arXiv preprint arXiv:2509.08827, 2025. 3
- [43] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine, “Training diffusion models with reinforcement learning,”

2024. 3

- [44] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee, “Dpok: reinforcement learning for fine-tuning text-to-image diffusion models,” in Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, (Red Hook, NY, USA), 2023.
- [45] X. Liao, W. Wei, X. Qu, and Y. Cheng, “Step-level reward for free in rl-based t2i diffusion model fine-tuning,” 2025. 3

- [46] J. Wang, Z. Tian, X. Wang, X. Zhang, W. Huang, Z. Wu, and Y.-G. Jiang, “Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl,” arXiv preprint arXiv:2504.11455, 2025. 3
- [47] G. Chen, S. Huang, K. Liu, J. Zhu, X. Qu, P. Chen, Y. Cheng, and Y. Sun, “Flash-dmd: Towards high-fidelity few-step image generation with efficient distillation and joint reinforcement learning,” arXiv preprint arXiv:2511.20549, 2025. 3
- [48] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan, “Janus-pro: Unified multimodal understanding and generation with data and model scaling,” 2025. 3, 6, 7
- [49] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang, “Visual autoregressive modeling: Scalable image generation via nextscale prediction,” Advances in neural information processing systems, vol. 37, pp. 84839–84865, 2024. 3
- [50] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018. 3, 8
- [51] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,”

2017. 4

- [52] D.-A. team, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025. 4, 5
- [53] S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, “Scheduled sampling for sequence prediction with recurrent neural networks,” Advances in neural information processing systems, vol. 28, 2015. 5
- [54] D. Ghosh, H. Hajishirzi, and L. Schmidt, “Geneval: An object-focused framework for evaluating text-to-image alignment,” 2023. 6, 7
- [55] LAION, “Laion-coco 600m,” 2022. Available at https: //laion.ai/blog/laion-coco. 6
- [56] R. Fang, A. Yu, C. Duan, L. Huang, S. Bai, Y. Cai, K. Wang, S. Liu, X. Liu, and H. Li, “Flux-reason-6m and prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark,” 2025. 6
- [57] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139 of Proceedings of Machine Learning Research, pp. 8748–8763, 18–24 Jul 2021. 7
- [58] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li, “Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,” arXiv preprint arXiv:2306.09341, 2023. 7
- [59] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes, T. Salimans, J. Ho, D. J. Fleet, and M. Norouzi, “Photorealistic text-to-image diffusion models with deep language understanding,” 2022. 7
- [60] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020. 1

[61] B. Chen, D. Mart´ı Mons´o, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann, “Diffusion forcing: Next-token prediction meets full-sequence diffusion,” Advances in Neural Information Processing Systems, vol. 37, pp. 24081– 24125, 2024. 2

