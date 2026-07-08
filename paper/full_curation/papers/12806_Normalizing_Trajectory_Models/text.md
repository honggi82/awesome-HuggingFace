arXiv:2605.08078v2[cs.CV]12May2026

# Normalizing Trajectory Models

###### Jiatao Gu1, Tianrong Chen1, Ying Shen2, David Berthelot1, Shuangfei Zhai1, Josh Susskind1 1Apple, 2UIUC

Diffusion-based models decompose sampling into many small Gaussian denoising steps, an assumption that breaks down when generation is compressed to a few coarse transitions. Existing few-step methods address this through distillation, consistency training, or adversarial objectives, but sacrifice the likelihood framework in the process. We introduce Normalizing Trajectory Models (NTM), which models each reverse step as an expressive conditional normalizing flow with exact likelihood training. Architecturally, NTM combines shallow invertible blocks within each step with a deep parallel predictor across the trajectory, forming an end-to-end network trainable from scratch or initializable from pretrained flow-matching models. Its exact trajectory likelihood further enables self-distillation: a lightweight denoiser trained on the score function induced by the model itself produces high-quality samples in four steps. On text-to-image benchmarks, NTMmatchesoroutperformsstrongimagegenerationbaselinesinjustfoursamplingstepswhileuniquelyretainingexact likelihood over the generative trajectory.

Code: https://github.com/apple/ml-starflow Correspondence: jgu32@apple.com Date: May 14, 2026

[Figure 1]

Figure 1 Text-to-image generation with NTM with 4 denoising steps. We show samples from models trained from scratch at 256×256, and from models obtained by finetuning pretrained flow-matching checkpoints at 512×512.

Work done while JG holding a joint affiliation at University of Pennsylvania, and YS working as a research intern at Apple.

- 1 Introduction Diffusion-based models (Ho et al., 2020; Song et al., 2021; Lipman et al., 2023; Liu et al., 2023; Albergo et al.,

- 2023) have become the dominant paradigm for high-fidelity image generation (Rombach et al., 2022; Esser et al., 2024; Podell et al., 2024). These methods decompose generation into many small denoising steps, each modeled as a Gaussian transition whose mean is predicted by a neural network. When the step size is small,

this Gaussian approximation is accurate: the reverse conditional p(xs | xt) is close to Gaussian because the transition covers only a small portion of the diffusion trajectory. However, reducing the number of sampling steps to improve efficiency forces each transition to span a larger interval, and the true reverse conditional becomes a mixture of Gaussians that can be multimodal and heavy-tailed. The single-Gaussian assumption then becomes a fundamental bottleneck for few-step generation quality.

A growing body of work addresses the efficiency problem, but existing approaches sacrifice the likelihood framework. Distillation methods (Salimans and Ho, 2022; Yin et al., 2024b) and consistency models (Song et al., 2023; Luo et al., 2023) learn to map noise to data in fewer steps, yet provide no tractable density over the generative trajectory. DDGAN (Xiao et al., 2022) replaces the Gaussian reverse with an implicit distribution learned via adversarial training, but introduces mode-seeking behavior and training instability that limit scalability. No existing method achieves few-step generation with an exact likelihood model of the reverse process.

We introduce Normalizing Trajectory Models (NTM), a framework that models p(xs | xt) as a conditional normalizing flow with exact log-likelihood. The core idea is to learn a latent space—via an invertible transporter—where the reverse conditional becomes simple enough to be modeled by a Gaussian predictor. Unlike a compressive encoder, the transporter preserves dimensionality and invertibility, which together with the Gaussian predictor yields exact log-likelihood training through the change-of-variables formula. This bridges self-supervised representation learning and probabilistic generative modeling: the framework resembles a predictor–encoder architecture (Grill et al., 2020; Assran et al., 2023), but the invertibility constraint turns it into a normalizing flow.

[Figure 2]

Figure2 Denoising trajectories. Left: Flow matching with 50 steps and 4 steps. Right: NTM achieves comparable quality in 4 steps by modeling the non-Gaussian reverse conditional.

NTM can be trained from scratch using stochastic forward trajectories, or initialized from any pretrained flowmatching model by setting the transporter to identity and the predictor to the pretrained Gaussian posterior. The exact trajectory likelihood further enables score-based denoising: since the generated trajectory is an inherently noisy sequence from the Markov forward process, the gradient of the NTM loss provides a joint score that denoises all timesteps simultaneously by exploiting their correlations. A lightweight learned denoiser can distill this signal into a single forward pass, producing high-quality samples in as few as four steps. Experiments on class-conditional and text-to-image generation demonstrate that NTM matches or outperforms strong few-step baselines in image quality and compositional accuracy, achieving 0.82 on GenEval (Ghosh et al., 2023) with only 4 denoising steps when trained from scratch—significantly outperforming the prior normalizing flow model STARFlow (0.56, requiring 256 AR steps)—while uniquely retaining exact likelihood over the generative trajectory.

Our contributions are:

- • A framework that models the non-Gaussian reverse conditional p(xs | xt) via an invertible transporter and a Gaussian predictor, yielding exact log-likelihood while bridging representation learning and probabilistic modeling.
- • A finetuning recipe that initializes from pretrained diffusion or flow-matching models via identity transporter and zero-initialized scale correction, preserving pretrained quality at initialization.
- • Score-based trajectory denoising that exploits the exact likelihood and Markov covariance to jointly correct generated trajectories, distillable into a learned denoiser for four-step generation without additional training data.

## 2 Preliminaries

### 2.1 Flow Matching and Diffusion Models

Flow matching (Lipman et al., 2023; Liu et al., 2023; Albergo et al., 2023) defines a forward interpolation between clean data x0 and Gaussian noise ϵ ∼ N(0,I):

xt = (1 − t)x0 + tϵ, q(xt | x0) = N (1 − t)x0, t2I , t ∈ [0,1]. (2.1) A neural network vθ(xt,t) is trained to predict the velocity field by minimizing

0,ϵ vθ(xt,t) − (ϵ − x0) 2, (2.2)

LFM = Et,x

and samples are generated by integrating the learned ODE dx = vθ(x,t)dt from t=1 (noise) to t=0 (data). Mathematically, diffusion models (Ho et al., 2020) can be designed to share the same marginals q(xt | x0) under equivalent noise schedules, but define a stochastic forward process whose discretized reverse takes the form of a Gaussian transition kernel pθ(xs | xt) = N(µθ(xt,t,s), σ2(t,s)I).

In both frameworks, generation quality depends on the number of discretization steps: flow matching assumes the velocity field is locally linear within each step, while diffusion models assume the reverse conditional is Gaussian. With many steps these approximations are accurate; with few steps each transition must cover a large interval, and the true mapping from xt to xs becomes too complex for either a linear or Gaussian model to capture. To formalize and address this limitation, we adopt a stochastic trajectory framework that makes the per-step distribution an explicit modeling target.

### 2.2 Stochastic Trajectories and the Gaussian Bottleneck

Given a timestep schedule 0 = t0 < t1 < ··· < tT = 1, we construct a Markovian forward trajectory that satisfies the marginal constraint in Eq. (2.1) at every step. For any two consecutive timesteps s < t in the schedule, the forward transition is:

1 − t 1 − s

, σs,t = t2 − αs,t2 s2, (2.3)

xt = αs,t xs + σs,t ϵ, αs,t =

where ϵ ∼ N(0,I). Applying this transition sequentially yields a correlated stochastic path (xt

) from near-clean to near-noise, with each point marginally distributed as q(xt | x0). The Markovian structure defines a tractable joint distribution over the trajectory whose reverse conditionals q(xs | xt,x0) are Gaussian with known mean and variance.

#### ,xt

,...,xt

0

1

T

The Gaussian approximation. Standard diffusion and flow-matching models approximate the reverse conditional p(xs | xt) with a single Gaussian N(µθ(xt),σ2I). This is exact for the posterior conditioned on the clean image, p(xs | xt,x0), which is Gaussian by construction of the Markovian forward process. However, the marginal reverse conditional integrates over all possible clean images:

p(xs | xt) = p(xs | xt,x0)p(x0 | xt)dx0. (2.4)

uˆs

y fP(ut,z,y) D(us, uˆs)

ut

###### us

z ∼N(0,I)

−log|J| fT (xt, t) fT (xs, s) −log|J|

shared

xt xs

Figure 3 NTM overview. Shared transporter fT maps xt, xs to representations ut, us with a tractable Jacobian. The predictor fP takes ut and latent z ∼ N(0, I) to produce uˆs. D measures the distance between the prediction and the target at distribution level.

Since p(x0 | xt) is complex and potentially multimodal over natural images, the marginal p(xs | xt) is a mixture of Gaussians that a single Gaussian cannot capture. When the number of steps is small, each transition spans a large interval and the approximation error becomes severe.

### 2.3 Normalizing Flows

Normalizing flows (Dinh et al., 2014; Rezende and Mohamed, 2015; Dinh et al., 2016; Kingma and Dhariwal, 2018) learn an invertible mapping fθ : RD → RD between data x and a latent z = fθ(x) drawn from a simple prior p0(z) = N(0,I). The exact log-likelihood is given by the change-of-variables formula:

(x) . (2.5)

log p(x) = log p0 fθ(x) + log detJf

θ

A common design is the autoregressive flow (Kingma et al., 2016; Papamakarios et al., 2017), which transforms each element conditioned on all preceding elements via affine (NVP) coupling (Dinh et al., 2016), yielding a tractable triangular Jacobian. TarFlow (Zhai et al., 2025) parameterizes the affine coupling with a causal Transformer: each spatial token xn is transformed conditioned on all preceding tokens x<n via a self-exclusive causal mask:

xn − µθ(x<n) σθ(x<n)

log σθ(n), (2.6)

, log detJ = −

zn =

n

where σθ > 0 (scale) and µθ (shift) are predicted from preceding tokens. This allows normalizing flows to scale competitively for high-resolution image generation. STARFlow (Gu et al., 2026) further introduces a deep-shallow architecture: a single deep autoregressive flow block with many Transformer layers captures most of the model capacity, followed by a few lightweight shallow blocks with alternating scan directions (e.g., left-to-right and right-to-left) that refine spatial details. This deep-shallow design, extended to video in STARFlow-V (Gu et al., 2025), forms the architectural foundation of NTM.

## 3 Normalizing Trajectory Models

We present Normalizing Trajectory Models (NTM), a generative framework that models the full conditional distribution p(xs | xt) at each denoising step as a normalizing flow with exact log-likelihood (§ 3.1). NTM can be trained from scratch (§ 3.2), finetuned from pretrained diffusion or flow-matching models (§ 3.3), and accelerated to real-time generation via a learned denoiser (§ 3.4).

### 3.1 Model Formulation

- As discussed in § 2.2, modeling p(xs | xt) with a Gaussian formulation is fundamentally limited: the true reverse conditional is generally non-Gaussian because it marginalizes over all clean images consistent with

xt. We seek a more expressive family that provides exact likelihood for stable training, while remaining structurally close to the diffusion framework to preserve its scalability.

NTM models p(xs | xt) by learning to predict in a latent space where the conditional distribution is simple enough to be modeled by Gaussian. As shown in figure 3, a shared transporter fT maps both xs and xt to a latent u-space, and a stochastic predictor fP generates uˆs from the noisier representation ut and a latent variable z ∼ N(0,I), optionally conditioned on y (e.g., text or class label).

uˆs = fP(ut,z,y), us = fT (xs,s), ut = fT (xt,t). (3.1) The general training objective minimizes a distributional distance D between the prediction and the target, regularized by R(fT ) to prevent representation collapse (Grill et al., 2020):

L = Ez D(us, uˆs) + R(fT ). (3.2) Such objectives are common in self-supervised representation learning (Grill et al., 2020; Caron et al., 2021; Bardes et al., 2022; Assran et al., 2023), but are generally difficult to cast within a probabilistic framework for generative modeling. The key insight of NTM is that making fT an invertible, same-dimensional transporterrather than a compressive encoder—turns this representation-learning objective into exact log-likelihood optimization via the change-of-variables formula.

Specifically, we implement fT as a stack of TarFlow blocks (Zhai et al., 2025; Gu et al., 2026) with spatial NVP coupling (Eq. (2.6)), and fP as an affine map uˆs = µP(ut,t,s,y) + σP(ut,t,s,y) · z, which defines pP(us | ut,y) = N(µP, diag(σP2 )). Under these choices, setting D = −log pP and R = −log |detJf

| recovers the exact negative log-likelihood of p(xs | xt):

T

. (3.3)

LNTM = −log p(xs | xt) = −log pP(us | ut) − log detJf

T

The composed mapping xs ←→fT us ←→fP z forms a normalizing flow from xs to z ∼ N(0,I). By expanding over a trajectory of T steps, the NTM loss can be simplified as:

T

LNTM =

k=1

- 1

- 2∥zk∥2 + n

log σP(k,n) +

ℓ

log σT(k,ℓ,n) , (3.4)

where σP(k,n) is the predictor scale at step k and position n, and σT(k,ℓ,n) is the scale from transporter block ℓ. This is the exact negative log-likelihood of the trajectory and training minimizes it end-to-end.

### 3.2 Training from Scratch

Architecture. NTM adopts the deep-shallow architecture of STARFlow (Gu et al., 2026, 2025), with a key modification to the deep block. The predictor (fP) is a deep Transformer that replaces STARFlow’s spatial autoregressive flow with a non-causal full-attention coupling layer operating over the trajectory dimension. It predicts µP(ut,t,s,y) and σP(ut,t,s,y) for each denoising step. Despite its depth, the predictor processes all spatial positions in parallel, making it efficient at inference. The transporter (fT ) consists of a few shallow TarFlow-style (Zhai et al., 2025) causal autoregressive flow blocks with alternating scan directions. Although autoregressive by nature, each transporter block is lightweight and operates locally within a single denoising step without information leakage across timestep.

Training. Given a T-step schedule tmin = t0 < t1 < ··· < tT = 1, we model the joint trajectory distribution as:

T

k−1 | xt

p(xt

,...,xt

) = p(xt

)

p(xt

).

0

T

T

k

k=1

) = N(0,I) and skip both fT and fP at this level, so the model only learns the conditional factors p(xs | xt). Given clean data x0, we construct a stochastic forward trajectory via Eq. (2.3) and train with either:

Since tT = 1 is pure noise, we fix p(xt

T

- • End-to-end: compute the NTM loss (Eq. (3.4)) over all T conditional factors in the trajectory.
- • Pair-wise: randomly sample a single consecutive pair (t,s) with s < t per batch element.

In both modes, each batch element independently samples T from a predefined set (e.g., {4,8,16}), enabling a single model to generate with different step counts without retraining. For such cases, fT takes T as an additional input to adapt to the local timestep spacing.

Sampling. Given a schedule tmin = s0 < s1 < ··· < sT ≈ 1, sampling proceeds from noise to data by inverting Eq. (3.1): the predictor runs sequentially over T steps, drawing z ∼ N(0,I) and computing uˆs = µP(uˆt,t,s) + σP(uˆt,t,s) · z at each step, where each output feeds into the next. After all T predictor steps, the transporter inverts the spatial mapping xˆ0 = fT−1(uˆ0) via sequential AR decoding to produce the final sample in x-space. Classifier-free guidance (Ho and Salimans, 2022) is applied by interpolating the predictor’s conditional and unconditional outputs (Gu et al., 2026).

Trajectory Score Denoising. Normalizing flows require data to be dense for likelihood training, while natural images often lie on low-dimensional manifolds; TarFlow addresses this by adding a small noise and applying score-based denoising at test time (Zhai et al., 2025; Gu et al., 2026). In NTM, this extends naturally: the generated trajectory xˆ = (xˆt

) is inherently a noisy sequence from the Markov forward process, requiring no additional noise injection. However, unlike independent per-sample denoising, the trajectory elements are correlated across timesteps. The NTM loss provides −log p(xˆ), whose gradient gives the joint score of the full trajectory distribution. We exploit this to perform trajectory-level denoising:

,...,xˆt

0

T

1 1 − t

xˆden =

(xˆ − S · ∇xˆLNTM), (3.5)

where S is the covariance matrix of the trajectory under the pre-defined forward process (Eq. (2.3)), with [S]ij = min(ti,tj)2(1 − max(ti,tj))/(1 − min(ti,tj)), and division by (1−t) maps from the noisy domain to the clean domain. The final output is taken at tmin.

### 3.3 Finetuning from Pretrained Models

NTM can also be initialized from a pretrained flow matching or diffusion models. Taking flow matching as an example, the pretrained backbone is trained to predict the velocity field in x-space given noisy input xt and timestep t. Here, we reinterpret the prediction v and hidden states h from the input (ut,t) in u-space. We can readily compute a predicted clean sample uˆ0 = ut − t · v and derive the denoising posterior N(µpost,σpost2 I) for the transition from t to s:

µpost = A(t,s)ut + B(t,s)uˆ0, σpost = C(t,s), (3.6)

where A, B, and C are closed-form coefficients derived from the true reverse posterior of the Markovian forward process (§ 2.2; full derivation in § A.5). We initialize the predictor to match this posterior: µP = µpost, and learn a multiplicative scale correction via a zero-initialized projection:

Laux LNTM

µP σP

fP

µFM

ut

❄

fFM

fT

| | |
|---|---|
| | |

xt

Figure 4 Finetuning: Laux aligns µP with frozen µFM; LNTM trains the full model.

µP = µpost, σP = σpost · exp(δσ), δσ = projout(h), (3.7) where projout is initialized to zero so that σP = σpost at initialization. By further initializing the transporter

- as identity (fT = id), the full model starts as the pretrained Gaussian posterior in x-space. As training progresses, the NLL objective drives fT to drift from identity and δσ to depart from zero, jointly learning the non-Gaussian structure of p(xs | xt).

Mean-alignment auxiliary loss. To prevent early divergence from the pretrained solution, we add an auxiliary loss that aligns NTM’s learned shift µP with the denoising mean µFM produced by a frozen copy of the pretrained backbone predicting directly from x-space:

Laux = µP − µFM 2. (3.8)

The total loss is L = LNTM + λ Laux, where λ can be annealed during training. This auxiliary loss serves three purposes: (1) it encourages the model to remain close to the pretrained diffusion solution, preventing

ztT−1 zt1 zt0

ztT

gϕ

fP · · · fP fP

=

utT−1 ut1 ut0

utT

fT fT fT

=

xtT xtT−1 · · · xt1 xt0

| | | | | |
|---|---|---|---|---|
| | |−S·∇xLNTM| | |

xdentT xdentT−1 xdent1 xdent0

Lden

- Figure 5 Denoiser training via trajectory score denoising. The frozen NTM (dashed box) computes the trajectory NLL and its gradient refines every position via xden = x − Σ∇xLNTM, producing denoised targets (orange). A denoiser gϕ learns to predict xdent0 directly from the ut0.

catastrophic drift; (2) µFM itself defines a meaningful u-space—since it is a neural prediction of the next-step mean directly from xt, it is smooth and predictable, and Laux ensures the transporter learns to connect these per-step predictions into a coherent trajectory; (3) because the transporter and predictor can move jointly, the model can optimize the NF loss without drifting from the pretrained quality.

### 3.4 Fast Generation via Learned Denoiser

Standard sampling from NTM requires T sequential predictor steps with AR decoding at each step, together with the trajectory score-based denoising (Eq. (3.5)) using backpropagation at test time. Both of them, while acceptable due to the light-weight design, still introduce more latency than the predictor. To eliminate this cost, we can optionally train a lightweight denoiser network gϕ that amortizes the self-refinement into a single forward pass, following a similar distillation paradigm of NFM (Berthelot et al., 2026) and STARFlow-V (Gu et al., 2025). The denoiser is a Transformer with non-causal attention that takes the predictor’s output ut

0

- at the cleanest level in u-space along with text embeddings y, and directly outputs a denoised image xˆden0 .

Since we model a Markov trajectory and the designed invertibility, ut

already contains all the information

0

needed to deterministically predict the clean output. The denoiser can be post-trained after the main model converges, using MSE against score-based denoising targets derived from the frozen NTM model on real data trajectories (Eq. (3.5)):

,y) − xˆden0 2. (3.9)

Lden = gϕ(ut

0

- At inference, the new pipeline becomes: (1) run the predictor over T steps to produce ut

, (2) run gϕ in a single forward pass to obtain xˆ0. This bypasses both the transporter AR decoding and the backprop-based denoising, producing high-quality images in as few as four steps.

0

## 4 Experiments

### 4.1 Setup

Implementation. All NTM models are trained with AdamW in bfloat16 with FSDP on an internal text-image dataset of ∼70M pairs (including CC12M). We consider two settings:

- • From scratch: class-conditional ImageNet and text-to-image generation at 256×256 resolution with the latent space of FAE (Gao et al., 2025) (16× spatial compression, 32-dim latents), using Qwen-2.5-VL as the text encoder.
- • Finetuning: initializing from a pretrained flow-matching backbone (FLUX.2-klein, 4B)1 at 512×512 resolution with its native VAE latent space.

1https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B

Table 1 T2I Evaluation. GenEval (Ghosh et al., 2023) overall score and DPG-Bench (Hu et al., 2024) percentage.

Type Method GenEval↑ DPG↑

SDXL (Podell et al., 2024) 0.55 74.65 PixArt-α (Chen et al., 2024) 0.48 71.11 SD3-Medium (Esser et al., 2024) 0.62 84.08 FLUX.1-dev (Black Forest Labs, 2024) 0.66 83.84 Janus-Pro-7B (Chen et al., 2025) 0.80 84.19 HiDream-I1-Full (HiDream.ai, 2025) 0.83 85.89 Seedream 3.0 (ByteDance Seed Team, 2025) 0.84 88.27 Qwen-Image (Wu et al., 2025) 0.87 88.32 Nucleus-Image (Akiti et al., 2026) 0.87 88.79

DM

STARFlow (Gu et al., 2026) 0.56 – NTM (from scratch, 256 × 256) 0.82 79.64 NTM (finetune, 512 × 512) 0.76 83.38

NF

The transporter consists of 2 TarFlow-style blocks with 4 layers each and causal masks along alternating directions; the predictor is a 24-layer full-attention Transformer. All models use T=4 denoising steps and 10% CFG dropout. For finetuning, we apply the residual parameterization (§ 3.3) with the auxiliary loss (λ=2.5, MSE variant). Both settings use a batch size of 1024 on 64 H100 GPUs. Further details are in the Appendix.

Evaluation. We report compositional accuracy on GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al.,

- 2024) for text-to-image generation. We additionally evaluate class-conditional generation on ImageNet 256×256 for fair comparison when training NTM from scratch (§ D.3).

### 4.2 From-Scratch Results

Text-to-imagegeneration. table 1 reports compositional accuracy on GenEval and DPG-Bench. NTM trained from scratch at 256×256 achieves 0.82 on GenEval and 79.64 on DPG-Bench with only 4 steps, significantly outperforming the prior normalizing flow model STARFlow (Gu et al., 2026) (0.56 GenEval, 256 autoregressive steps) and matching strong diffusion baselines that require substantially more sampling steps.

Class-conditional ImageNet. As a controlled comparison for the from-scratch setting, we evaluate on classconditional ImageNet 256×256. NTM achieves 2.80 FID with 16 steps and 3.83 FID with 4 steps—comparable to STARFlow (FAE) at 2.67 FID which requires 256 autoregressive steps (§ D.3). These results use only the exact NLL training objective without any distribution-level losses (e.g., adversarial or perceptual), demonstrating that exact likelihood training alone produces competitive few-step generation.

### 4.3 Finetuning Results

Text-to-image generation. The finetuned variant at 512×512 achieves 0.76 on GenEval and 83.38 on DPGBench (table 1), demonstrating that NTM can scale to higher resolutions via pretrained initialization. The position and attribute-binding sub-tasks remain challenging at this stage of finetuning, suggesting room for improvement with longer training or stronger pretrained backbones.

Score denoising vs. learned denoiser. table 2 compares two inference strategies for the finetuned model: (i) transporter inversion followed by trajectory score denoising via Eq. (3.5), and (ii) the learned denoiser gϕ that amortizes the refinement into a single forward pass. The denoiser achieves ∼ 9× speedup while maintaining high fidelity to the score-based refinement output

Table 2 Score denoising vs. learned denoiser (finetuned setting).

Method img/s ↑ LPIPS ↓ Full NF + Traj. denoise 0.20 Predictor + Denoiser 1.88 0.121

[Figure 3]

- Figure 6 Ablation: multi-trajectory training. Comparison of the same NTM evaluated with T=4, T=8, T=16 denoising steps and the baseline FLUX (50 steps).

[Figure 4]

(a) Without aux loss (b) With aux loss (c) Trajectory score denoising vs. learned denoiser

[Figure 5]

[Figure 6]

- Figure7 Ablations. (a) Finetuning directly with the NF loss diverges. (b) Adding the mean-alignment loss (Eq. (3.8)) stabilizes training. (c) Comparison of denoising approaches.

(LPIPS 0.121), confirming that a single forward pass can effectively replace iterative backpropagation-based denoising.

- 4.4 Ablation Studies We conduct ablation studies on text-to-image generation to analyze the key design choices of NTM.

Multi-trajectorytraining(finetuned). figure 6 compares finetuned models trained with different trajectory lengths

T ∈ {4,8,16} against the baseline FLUX (50 steps). Longer trajectories provide finer-grained denoising steps, which can improve detail preservation at the cost of slower inference. We find that T=4 provides the best quality–speed trade-off for the finetuning setting.

Effect of the transporter (from scratch). As shown in figure 2, reducing flow matching to 4 steps without a transporter produces severely blurry outputs. The invertible mapping provides a latent space where the affine predictor becomes expressive, recovering 50-step quality in only 4 steps.

Auxiliary loss for finetuning. figure 7 ablates the mean-alignment auxiliary loss (Eq. (3.8)). Without the auxiliary loss (λ = 0), finetuning diverges early in training—the NLL objective alone provides insufficient signal to keep the predictor near the pretrained solution, causing catastrophic forgetting. The auxiliary loss stabilizes training by anchoring the predictor mean to the pretrained velocity field.

- 4.5 Qualitative Results figure 1 presents text-to-image samples from NTM in 4 denoising steps across both settings.

[Figure 7]

Figure 8 Comparison between TARFlows, Normalizing Trajectory Models, and Diffusion Models. Subscripts denote spatial patch indices, and superscripts denote trajectory timesteps, with x1, u1 representing pure Gaussian noise and x0, u0 the clean data.

From scratch (256×256). The from-scratch model demonstrates strong compositional generalization across multi-object scenes, fine-grained attribute control, and varied artistic styles despite being trained at moderate resolution.

Finetuned (512×512). The finetuned model preserves the visual quality and prompt adherence of the pretrained FLUX backbone (which requires 50 steps) while operating in only 4 steps, confirming that modeling the non-Gaussian reverse conditional recovers the information lost by naive step reduction. Samples exhibit high-resolution detail, text rendering capability, and diverse artistic styles.

## 5 Discussion

NTM as an interpolation between normalizing flows and flow matching. STARFlow (Gu et al., 2026) directly models the marginal image distribution p(x) by decomposing it via a deep spatial autoregressive flow within a single generation step—the entire generation is performed in one pass through many sequential AR blocks (e.g., 256 steps). At the other extreme, flow matching models a velocity field whose ODE integration requires many small Gaussian steps for high quality. NTM occupies a middle ground: it explicitly models each intermediate conditional p(xs | xt) along a T-step denoising trajectory as a normalizing flow, as shown in figure 8.

[Figure 8]

The key architectural tradeoff is where to place depth. STARFlow concentrates all capacity within a single step via deep spatial AR blocks; NTM distributes capacity across multiple denoising steps, using a shallow transporter (2 blocks × 4 layers) at each step paired with a deep trajectory-level predictor. This trades per-step expressiveness for multistep structure: the predictor reasons across timestep levels while each transporter handles only the local non-Gaussian residual within one step. As a result, the per-step normalizing flow in NTM can be lightweight because each step only needs to capture the conditional p(xs | xt)which is simpler than the full marginal p(x)—while the deep predictor captures the bulk of the denoising signal in u-space.

Why single-step generation remains challenging. As shown in figure 9, NTM with T=1 produces severely degraded outputs. This failure is not a training issue but a fundamental capacity constraint. At T=1, the entire non-Gaussian structure of the data distribution must be captured by the shallow transporter alone—the predictor reduces to a single-step Gaussian coupling. This configuration is effectively a STARFlow-like architecture with a parallel (non-causal) prior, but with far fewer transporter layers than STARFlow’s deep blocks (8 layers vs. STARFlow’s 24+ layers per block × multiple blocks). Making the transporter as deep as STARFlow would restore single-step

Figure 9 Failure Case: NTM with T=1 produces degraded outputs due to insufficient transporter capacity. Prompt: a corgi dog.

quality but defeat the purpose of the few-step design, as inference would again be dominated by sequential AR decoding.

For finetuning, the T=1 setting introduces additional challenges: the mean-alignment auxiliary loss (Eq. (3.8)) was designed to anchor the predictor to a multi-step denoising trajectory, and collapsing to a single step fundamentally changes the training dynamics.

Implications. NTM’s sweet spot is T=4–8: enough steps for the shallow transporter to distribute the nonGaussian modeling across the trajectory, while the deep predictor handles cross-timestep reasoning efficiently in parallel. The architecture naturally admits a spectrum—deeper transporters with fewer steps, or shallower transporters with more steps—offering a principled way to trade off sequential computation for generation quality. Pushing toward single-step generation with exact likelihood remains an open challenge that may require fundamentally different architectural choices, such as adaptive-depth transporters or progressive capacity allocation across the trajectory.

## 6 Related Work

Normalizing flows for image generation. Normalizing flows (Dinh et al., 2014, 2016; Rezende and Mohamed, 2015; Kingma and Dhariwal, 2018) learn invertible mappings with exact log-likelihood via the change-ofvariables formula. Classical approaches struggled to scale to high-resolution images due to the full-dimensional invertibility constraint. TarFlow (Zhai et al., 2025) addressed this by parameterizing autoregressive coupling layers with causal Transformers, enabling flows to leverage modern sequence-modeling architectures. STARFlow (Gu et al., 2026, 2025) further introduced a deep-shallow design—a single deep autoregressive block followed by lightweight shallow blocks—scaling normalizing flows to competitive text-to-image generation. While these methods model the marginal p(x) directly, NTM applies normalizing flows to the conditional distribution p(xs | xt) at each denoising step. Since conditioning on xt already constrains the space of plausible images, the per-step flow is simpler than the full marginal and requires fewer blocks.

Non-Gaussian reverse processes. The Gaussian assumption in diffusion reverse steps has been challenged by several works. DDGAN (Xiao et al., 2022) trains a GAN discriminator at each denoising step, enabling larger step sizes by modeling an implicit non-Gaussian conditional. However, GAN-based approaches provide no tractable density, suffer from mode-seeking behavior, and are difficult to scale. Diffusion Normalizing Flow (Zhang and Chen, 2021) combines normalizing flows with diffusion via neural SDEs, but models the entire generation trajectory as a single continuous flow rather than learning an expressive per-step reverse conditional. Concurrent work (Chen et al., 2026) also explores normalizing flows with iterative denoising using a different architectural design. NTM models the non-Gaussian reverse via normalizing flows with exact log-likelihood, providing mode-covering training, stable optimization, and a tractable score.

Few-step generation and distillation. Reducing sampling steps is a major research direction. Progressive distillation (Salimans and Ho, 2022) trains a student to match multi-step teacher outputs in fewer steps. Consistency models (Song et al., 2023) learn to map any point on the trajectory directly to the clean image. Distribution matching distillation (DMD) (Yin et al., 2024a) and latent consistency models (Luo et al., 2023) further improve few-step quality via distributional matching objectives. NFM (Berthelot et al., 2026) distills pretrained normalizing flow couplings to train faster flow-matching students. NTM is complementary to distillation approaches with a learned non-Gaussian reverse and trains a denoiser for fast inference.

Score-based denoising and refinement. The connection between denoising and score functions (Song et al., 2021) has been exploited for test-time sample improvement. TarFlow (Zhai et al., 2025) introduced adding a small amount of noise and applying the gradient of the NF log-likelihood as a score-based denoiser; STARFlow (Gu et al., 2026) extended this to latent-space generation. These methods perform independent per-sample denoising. NTM generalizes this to trajectory-level denoising: since the generated trajectory is a correlated sequence from the Markov forward process, the NTM loss provides a joint score over all timesteps, and the covariance-weighted gradient correction exploits cross-timestep correlations for more effective refinement than per-sample approaches.

Trajectory-level modeling and flow maps. Several methods model generation across multiple trajectory points rather than per-step. Consistency models (Song et al., 2023) and latent consistency models (Luo et al., 2023) learn to project any noisy point directly to the clean endpoint. FlowMaps (Boffi et al., 2025) generalizes this by learning direct mappings between arbitrary pairs of time points on the probability flow ODE. Mean flows (Geng et al., 2025) learn one-step generators via flow matching with mean prediction. These methods learn deterministic mappings via regression objectives. NTM is distinct in two ways: it retains a distributional model of p(xs | xt) (not a point estimate), enabling sampling diversity and likelihood evaluation; and it models the conditional at each step as a normalizing flow rather than collapsing all steps into a single mapping.

## 7 Conclusion

We introduced Normalizing Trajectory Models (NTM), a framework that models each reverse conditional as a normalizing flow via an invertible transporter and a Gaussian predictor, yielding exact log-likelihood training. NTM supports training from scratch and finetuning from pretrained models, and its trajectory likelihood enables score-based denoising distillable into a four-step sampler. On text-to-image benchmarks, NTM significantly outperforms prior normalizing flow models and matches strong diffusion baselines with only 4 steps, while uniquely retaining exact likelihood over the generative trajectory. Future work includes distribution-level post-training (e.g., adversarial or perceptual losses) to further boost few-step quality, scaling to higher resolutions, and exploring architectural designs that push exact-likelihood generation toward even fewer steps.

## References

Chandan Akiti et al. Nucleus-image: Sparse moe for image generation. arXiv preprint arXiv:2604.12163, 2026. Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for

flows and diffusions. In International Conference on Learning Representations, 2023.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. arXiv preprint arXiv:2301.08243, 2023.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2022.

David Berthelot, Tianrong Chen, Jiatao Gu, Marco Cuturi, Laurent Dinh, Bhavik Chandna, Michal Klein, Josh Susskind, and Shuangfei Zhai. The coupling within: Flow matching via distilled normalizing flows. arXiv preprint arXiv:2603.09014, 2026.

Black Forest Labs. Flux.1. Technical report / model release, 2024. Nicholas M Boffi, Michael S Albergo, and Eric Vanden-Eijnden. How to build a consistency model: Learning flow

maps via self-distillation. arXiv preprint arXiv:2505.18825, 2025. ByteDance Seed Team. Seedream 3.0. Technical report / model release, 2025. Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin.

Emerging properties in self-supervised vision transformers. arXiv preprint arXiv:2104.14294, 2021.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In International Conference on Learning Representations, 2024.

Tianrong Chen, Jiatao Gu, David Berthelot, Joshua Susskind, and Shuangfei Zhai. Normalizing flows with iterative denoising. arXiv preprint arXiv:2604.20041, 2026.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

Yuan Gao, Chen Chen, Tianrong Chen, and Jiatao Gu. One layer is enough: Adapting pretrained visual encoders for image generation. arXiv preprint arXiv:2512.07829, 2025.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, 2023.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

Jiatao Gu, Yuyang Wang, Yizhe Zhang, Qihang Zhang, Dinghuai Zhang, Navdeep Jaitly, Josh Susskind, and Shuangfei Zhai. Dart: Denoising autoregressive transformer for scalable text-to-image generation. arXiv preprint arXiv:2410.08159, 2024.

Jiatao Gu, Ying Shen, Tianrong Chen, Laurent Dinh, Yuyang Wang, Miguel Angel Bautista, David Berthelot, Josh Susskind, and Shuangfei Zhai. Starflow-v: End-to-end video generative modeling with normalizing flow. arXiv preprint arXiv:2511.20462, 2025.

Jiatao Gu, Tianrong Chen, David Berthelot, Huangjie Zheng, Yuyang Wang, Ruixiang Zhang, Laurent Dinh, Miguel Angel Bautista, Joshua Susskind, and Shuangfei Zhai. Starflow: Scaling latent normalizing flows for highresolution image synthesis. Advances in Neural Information Processing Systems, 38:120986–121022, 2026.

HiDream.ai. Hidream-i1. Technical report / model release, 2025. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative

Models and Downstream Applications, 2022. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31, 2018. Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. Advances in neural information processing systems, 29, 2016.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum id=PqvMRDCJT9t.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, 2024.

George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. Advances in neural information processing systems, 30, 2017.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pages 1530–1538. PMLR, 2015.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, 2023.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image

generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. Chenfei Wu et al. Qwen-image technical report. arXiv preprint, 2025. Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion

GANs. In International Conference on Learning Representations, 2022. Jiawei Yang, Zhengyang Geng, Xuan Ju, Yonglong Tian, and Yue Wang. Representation fréchet loss for visual generation. arXiv preprint arXiv:2604.28190, 2026.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in Neural Information Processing Systems, 2024a.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Shuangfei Zhai, Ruixiang ZHANG, Preetum Nakkiran, David Berthelot, Jiatao Gu, Huangjie Zheng, Tianrong Chen, Miguel Ángel Bautista, Navdeep Jaitly, and Joshua M Susskind. Normalizing flows are capable generative models. In Forty-second International Conference on Machine Learning, 2025.

Qinsheng Zhang and Yongxin Chen. Diffusion normalizing flow. In Advances in Neural Information Processing Systems, 2021.

## A Theoretical Analysis

### A.1 NTM as a Conditional Normalizing Flow

We establish that each reverse transition in NTM defines a valid conditional normalizing flow with exact log-likelihood. Consider a single denoising step from timestep t to s (s < t). NTM maps the clean-side sample xs to a latent z via the composition of two invertible transformations:

- 1. Transporter (spatial autoregressive flow): us = fT (xs), an invertible mapping from x-space to u-space via a stack of L TarFlow-style (Zhai et al., 2025) causal AR blocks with alternating scan directions.

Each block ℓ applies an elementwise affine coupling u(nℓ) = (u(nℓ−1) − µ(ℓ)(u(<nℓ−1)))/σ(ℓ)(u(<nℓ−1))) with a triangular Jacobian.

- 2. Predictor (trajectory-level affine coupling): z = (us − µP(ut,t,s))/σP(ut,t,s), an affine coupling in u-space conditioned on the noisier representation ut.

The composition z = fP(fT (xs);ut,t,s) is invertible (both components are), and the exact log-likelihood follows from the change-of-variables formula:

log p(xs | xt) = log p0(z)

+log detJf

P

Gaussian prior

predictor

L

+

log detJf(ℓ)

T

ℓ=1

transporter

. (A.1)

| = − n log σP(n). Each transporter block has a triangular Jacobian from the autoregressive structure. Expanding the Gaussian prior term log p0(z) = −12∥z∥2 + const recovers Eq. (3.3) in the main text.

Since the predictor is a diagonal affine coupling, log |detJf

P

Relation to STARFlow. STARFlow (Gu et al., 2026) is a normalizing flow that models the marginal image distribution p(x) using the same deep-shallow architectural building blocks. NTM applies these blocks to a fundamentally different object: the conditional distribution p(xs | xt) at each denoising step. This has two consequences. First, the predictor in STARFlow operates spatially (causal attention over image patches within a single image), whereas in NTM it operates over the trajectory dimension (non-causal attention across timestep levels), enabling cross-timestep reasoning. Second, because the conditional p(xs | xt) is simpler than the marginal p(x)—conditioning on xt already constrains the space of plausible images—NTM requires fewer transporter blocks per step to achieve comparable expressiveness.

### A.2 Decomposition: Gaussian Denoising + Spatial Flow

NTM cleanly decomposes into two complementary components: a Gaussian denoising model in u-space (the predictor) and a non-Gaussian spatial transformation (the transporter). We formalize this decomposition and show that without the transporter, NTM reduces exactly to standard Gaussian diffusion.

Predictor alone = Gaussian denoising. Suppose the transporter are absent, i.e., fT = id so that us = xs. The latent variable becomes z = (xs − µP(xt,t,s))/σP(xt,t,s), and the conditional distribution implied by the predictor is:

p(xs | xt) = N µP(xt,t,s), diag(σP2 ) . (A.2) This is a diagonal Gaussian—precisely the same family used by standard diffusion models and flow matching. The NTM loss in this case reduces to:

k − µ(Pk)

xs

2

- 1

- 2

log σP(k,n) , (A.3)

L =

+

σP(k)

n

k

which is the negative log-likelihood of a heteroscedastic Gaussian regression. If σP is further held fixed (not learned), minimizing over µP yields a weighted MSE loss, recovering the standard diffusion/flow-matching training objective up to a constant.

Transporter adds non-Gaussian expressiveness. The shallow autoregressive blocks introduce a nonlinear, invertible change of coordinates us = fT (xs) before the Gaussian coupling is applied. Even though the predictor still applies a Gaussian (affine) coupling in u-space, the implied distribution in x-space is non-Gaussian because the inverse xs = fT−1(us) is a nonlinear transformation of a Gaussian. Formally:

p(xs | xt) = N fT (xs); µP, diag(σP2 ) · detJf

(xs) . (A.4) The Jacobian determinant |detJf

T

| reweights the density to account for the nonlinear warping, allowing the model to represent multimodal, heavy-tailed, or skewed distributions in x-space even though u-space remains Gaussian.

T

Divisionoflabor. In practice, the predictor captures the bulk of the denoising signal—predicting a good mean µP and an appropriate scale σP in u-space—while the transporter learn the residual non-Gaussian structure via their spatial autoregressive coupling. This division is efficient: the predictor uses a large Transformer backbone and concentrates most parameters on cross-timestep reasoning, while each transporter block is lightweight (2 layers) and handles local spatial dependencies.

- A.3 Effect of the FM Auxiliary Loss When finetuning from a pretrained flow-matching model, NTM adds a mean-alignment auxiliary loss (Eq. (3.8)):

Laux = λ

k

µ(Pk) − µ(FMk) 2, (A.5)

where µFM is the denoising mean from a frozen copy of the pretrained backbone. We analyze how this loss interacts with the NTM likelihood objective.

Without Laux (λ = 0). The NTM NLL objective jointly optimizes both the predictor (µP, σP) and the transporter (fT ). In principle, the model is free to redistribute “work” between components: the predictor could learn a mean far from the pretrained solution if the transporter compensate. In practice, this freedom can cause early-training instability, as the zero-initialized residual projection departs from the pretrained posterior before the transporter have learned a meaningful spatial mapping.

With Laux (λ > 0). The auxiliary loss anchors the predictor’s mean prediction µP to the pretrained FM solution µFM. This has two consequences:

- 1. Stabilized u-space. The u-space representation remains close to the pretrained model’s latent space, providing a stable coordinate system in which the transporter can learn meaningful non-Gaussian corrections.
- 2. Non-Gaussian structure through σP and fT . Since µP ≈ µFM, the non-Gaussian expressiveness must come from two sources: the learned scale σP (which departs from the fixed Gaussian posterior variance σpost), and the transporter’ spatial flow fT .

In our experiments, we anneal λ during training: starting at full strength to ensure stable initialization, then decaying so that the NLL objective can fine-tune the mean beyond the Gaussian approximation.

Limiting case: λ → ∞. If λ is very large, µP is forced to exactly match µFM, and the predictor becomes equivalent to the pretrained Gaussian reverse step. The only source of non-Gaussian modeling is then the transporter. The model reduces to: first, apply a spatial normalizing flow to transform xs into u-space; then, evaluate a fixed Gaussian posterior in u-space. This is still more expressive than a standard diffusion model (which has no spatial flow), but less expressive than the full NTM with learned µP and σP.

- A.4 Forward Transition Preserves Marginals

- Proposition 1. Let q(xt | x0) = N((1−t)x0,t2I) and define the forward transition from time s to t (s < t) as:

xt = αs,t xs + σs,t ϵ, ϵ ∼ N(0,I), (A.6) with αs,t = (1 − t)/(1 − s) and σs,t = t2 − αs,t2 s2. If xs ∼ q(xs | x0), then xt ∼ q(xt | x0).

Proof. Since xs ∼ N((1 − s)x0, s2I), the transformed variable xt = αs,txs + σs,tϵ is Gaussian with:

E[xt | x0] = αs,t (1 − s)x0 =

1 − t 1 − s

(1 − s)x0 = (1 − t)x0, (A.7) Var[xt | x0] = αs,t2 s2 + σs,t2 = αs,t2 s2 + t2 − αs,t2 s2 = t2. (A.8)

Hence xt ∼ N((1 − t)x0,t2I) = q(xt | x0).

| |
|---|

- A.5 Reverse Posterior Coefficients

We derive the closed-form expressions for the coefficients A(t,s), B(t,s), and C(t,s) in the reverse Gaussian posterior q(xs | xt,x0), used in the finetuning parameterization (Eq. (3.6)).

- Proposition 2. Under the forward process q(xt | x0) = N((1−t)x0,t2I) with Markov transition (Eq. (2.3)), the reverse posterior is:

q(xs | xt,x0) = N A(t,s)xt + B(t,s)x0, C(t,s)2 I , (A.9) where

- A(t,s) =

- s2(1 − t)

- t2(1 − s)

, (A.10)

- B(t,s) =

(t − s)(t + s − 2ts) t2(1 − s)

, (A.11)

- C(t,s)2 =

s2(t − s)(t + s − 2ts) t2(1 − s)2

. (A.12)

Proof. By Bayes’ rule, q(xs | xt,x0) ∝ q(xt | xs)q(xs | x0), where:

q(xs | x0) = N (1−s)x0, s2I , (A.13) q(xt | xs) = N αs,txs, σs,t2 I . (A.14)

The product of two Gaussians in xs is proportional to exp(−21x⊤s Λxs+η⊤xs) with precision and information vector:

αs,t2 σs,t2

1 s2

Λ =

I +

(1 − s)x0 s2

η =

I, (A.15)

αs,txt σs,t2

. (A.16)

+

We first compute σs,t2 = t2 − αs,t2 s2. Defining D := t2(1−s)2 − s2(1−t)2, we note that: D = t(1−s) − s(1−t) t(1−s) + s(1−t) = (t−s)(t+s−2ts), (A.17)

so σs,t2 = D/(1−s)2. Precision.

(1−t)2/(1−s)2 D/(1−s)2

(1−t)2 D

D + s2(1−t)2 s2D

t2(1−s)2 s2D

1 s2

1 s2

, (A.18)

Λ =

+

=

+

=

=

where the last step uses D + s2(1−t)2 = t2(1−s)2. Posterior variance.

s2(t−s)(t+s−2ts) t2(1−s)2

- s2D

- t2(1−s)2

C2 = Λ−1 =

. (A.19)

=

Posterior mean. The information vector is η = (1−s)x0/s2 + αs,txt/σs,t2 . Computing αs,t/σs,t2 = [(1−t)/(1−s)] · (1−s)2/D = (1−t)(1−s)/D, the posterior mean is:

µpost = C2 · η =

- s2D

- t2(1−s)2

(1−s)x0 s2

(1−t)(1−s)xt D

+

- s2(1−t)xt

- t2(1−s)

D x0 t2(1−s)

+

=

= A(t,s)xt + B(t,s)x0. (A.20)

In the finetuning recipe (§ 3.3), x0 is replaced by the predicted clean sample xˆ0 = xt − t · vθ, where vθ is the pretrained velocity prediction.

| |
|---|

Table 3 Posterior coefficients for a representative 4-step schedule.

Step t s A(t, s) B(t, s) C(t, s)

- 1 1.000 0.754 0.140 0.614 0.371
- 2 0.754 0.509 0.271 0.496 0.362
- 3 0.509 0.263 0.470 0.362 0.297
- 4 0.263 0.020 0.948 0.049 0.049

Numerical values for a 4-step schedule. table 3 provides representative values of A, B, and C for the default 4-step schedule used in our experiments.

### A.6 Trajectory Covariance Matrix

The self-refinement step (Eq. (3.5)) uses a trajectory covariance matrix S whose (i,j)-th entry is the covariance between xt

and xt

conditioned on x0.

i

j

- Proposition 3. Under the Markov forward process, for any two timesteps ti,tj in the trajectory:

min(ti,tj)2 1 − max(ti,tj)

1 − min(ti,tj) · I. (A.21) Proof. Without loss of generality, assume ti ≤ tj (so min = ti, max = tj). Write xt

j | x0) =

Cov(xt

,xt

i

= (1−ti)x0 + ξi where ξi ∼ N(0,t2iI) is the noise component. By the Markov property (Eq. (2.3)):

i

i,tj ϵ, ϵ ∼ N(0,I) independent of xt

. (A.22) Therefore:

#### xt

= αt

i,tj xt

+ σt

j

i

i

j | x0) = Cov(ξi, αt

Cov(xt

,xt

i,tjξi + σt

i,tjϵ)

i

= αt

i,tj Var(ξi)

t2i(1 − tj) 1 − ti · I. (A.23)

1 − tj 1 − ti · t2i · I =

=

Since ti = min(ti,tj) and tj = max(ti,tj), this matches Eq. (A.21). The diagonal case (i = j) gives Var(xt

i | x0) = t2i, consistent with the formula via limt

j→ti t2i(1 − tj)/(1 − ti) = t2i.

| |
|---|

Self-refinement update. Using this covariance, the self-refinement (Eq. (3.5)) applies a covariance-weighted gradient step:

xˆ − S ∇xˆLNTM 1 − t

, (A.24)

xˆ ←

where the matrix-vector product S ∇xˆL couples the gradient across timesteps according to their noise correlation. This is more effective than per-step independent correction (which would use only the diagonal Var(xt

) = t2i), because correcting an error at one timestep propagates correlated corrections to all other timesteps.

i

- Algorithm 1 NTM Training (single iteration) Require: Clean data x0, condition y (text or class label), number of steps T, noise range [tlomin,thimin]

- 1: Sample per-example minimum noise: tmin ∼ Uniform[tlomin,thimin]
- 2: Compute shifted timestep schedule: (t0,t1,...,tT) with t0 = tmin
- 3: Forward trajectory: for k = 0,...,T−1:
- 4: ϵk ∼ N(0,I)
- 5: xt

k+1

= αt

k,tk+1xt

k

+ σt

k,tk+1ϵk

- 6: Transporter (spatial AR flow): ut

k

= fT (xt

k

) for all k, accumulate log |detJf

T

|

- 7: Predictor (trajectory coupling): for each consecutive pair (tk+1,tk):
- 8: (µ(Pk),σP(k)) = DeepBlock(ut

k+1

,tk+1,tk,y)

- 9: zk = (ut

k − µ(Pk))/σP(k)

- 10: NTM loss: LNTM = Tk=1 12∥zk∥2 + n log σP(k,n) − ℓ log |detJf(ℓ)

T

|

- 11: (Optional) FM auxiliary loss: Laux = λ k ∥µ(Pk) − µ(FMk) ∥2
- 12: Update θ via ∇θ(LNTM + Laux)

- Algorithm 2 NTM Sampling Require: Condition y, number of steps T, guidance scale w, schedule (t0,t1,...,tT)

- 1: Sample initial noise: uˆt

T ∼ N(0,I)

- 2: Predictor reverse (parallel over spatial, sequential over k):
- 3: for k = T,T−1,...,1 do
- 4: zk ∼ N(0,I)
- 5: (µ(Pk),σP(k)) = DeepBlock(uˆt

k

,tk,tk−1,y)

- 6: (If CFG) Apply guidance: (µ(Pk),σP(k)) ← CFG(·,w)
- 7: uˆt

k−1

= zk · σP(k) + µ(Pk)

- 8: end for
- 9: Transporter inverse (sequential AR decoding with KV-cache):
- 10: xˆt

0

= fT−1(uˆt

0

)

- 11: (Optional) Self-refinement: apply algorithm 3
- 12: (Optional) Learned denoiser: xˆ0 = Dϕ(uˆ,y,t)
- 13: Decode: image = VAE.decode(xˆt

)

0

## B Algorithm Pseudocode

- B.1 Training
- B.2 Sampling
- B.3 Trajectory Self-Refinement

## C Implementation Details

- C.1 Model Architecture table 4 summarizes the architectural specifications of the NTM models used in our experiments.

Predictor. In the from-scratch setting, the predictor is a standard non-causal Transformer that processes all timestep levels in parallel. It takes as input the u-space representations (ut

), concatenated with text embeddings y, and predicts per-step coupling parameters (µP,σP) via a linear projection layer. Timestep conditioning is provided through additive sinusoidal embeddings.

,...,ut

0

T

In the finetuned setting, the predictor wraps a pretrained flow-matching backbone (FLUX.2). The backbone’s last hidden states are captured and fed to a zero-initialized projection layer projout : Rd → R2c that outputs

- Algorithm 3 Trajectory Self-Refinement

Require: Generated trajectory xˆ = (xˆt

), frozen NTM model, schedule (t0,...,tT)

,...,xˆt

0

T

- 1: Enable gradients w.r.t. xˆ
- 2: Forward pass through NTM: compute LNTM(xˆ)
- 3: Compute gradient: g = ∇xˆLNTM
- 4: (Optional) Percentile-based gradient clipping on g
- 5: Compute trajectory covariance: [S]ij = min(ti,tj)2(1 − max(ti,tj))/(1 − min(ti,tj))
- 6: Covariance-weighted correction: xˆ ← xˆ − S g ▷ couples gradients across timesteps
- 7: Normalize to clean domain: xˆ ← xˆ /(1 − t)
- 8: return xˆ

Table 4 Architectural specifications of NTM models.

###### From scratch Finetuned

Hidden dimension 3072 3072 Number of blocks 3 3 Layers per block [4, 4, 24] [4, 4, 24]

Transporter Blocks 1–2 (4 layers each) Blocks 1–2 (4 layers each) Predictor Block 3 (24 layers) FLUX.2-klein (4B)

Patch size 1 2 KV heads 8 8 Positional encoding 2D RoPE 2D RoPE Transporter scan order Alternating (L→R, R→L) Alternating (L→R, R→L)

Pretrained backbone — FLUX.2-klein (4B) Denoising mode — true_reverse no_delta_mean — 1 (µP = µpost)

the residual corrections (δµ,δσ). At initialization, projout = 0, so the predictor exactly reproduces the pretrained Gaussian posterior.

Transporter. Each transporter block is a TarFlow-style causal autoregressive flow with 2 Transformer layers. Blocks alternate between identity and flip permutations (left-to-right and right-to-left scan directions) for better spatial mixing. At the highest noise level (t ≈ 1), the transporter are skipped (identity transform) since the input is nearly isotropic Gaussian and the spatial AR coupling would be uninformative.

- C.2 Training Hyperparameters The hyperparameters are listed in Table 5.
- C.3 Denoiser Architecture

The learned denoiser gϕ (§ 3.4) is a lightweight Transformer that takes the predictor output ut

at the cleanest

0

level as input and produces a denoised image xˆden0 in a single forward pass. Since the trajectory is Markov, ut

contains all the information needed to deterministically predict the clean output.

0

- • Position encoding: 2D rotary embeddings over spatial (row, column) dimensions.
- • Attention: Full non-causal attention over all spatial positions.
- • Conditioning: Text embeddings y are concatenated to the input sequence.
- • Output: A single predicted clean image in patch space.
- • Training: After the main NTM model converges, the frozen model generates targets via trajectory score denoising, and gϕ is trained with MSE loss (Eq. (3.9)).

Table 5 Training hyperparameters.

###### From Scratch (ImageNet) Finetuned

Optimizer AdamW AdamW (β1, β2) (0.9, 0.95) (0.9, 0.95) Weight decay 10−4 10−4 Peak learning rate 10−4 5 × 10−5 Minimum learning rate 10−6 10−6 LR schedule Cosine with warmup Cosine with warmup Precision bfloat16 bfloat16 Distributed strategy FSDP2 FSDP2

Denoising steps (T) 4 4 tmin range Uniform[0.0, 0.05] Uniform[0.0, 0.05] CFG dropout 10% 10% FM aux loss weight (λ) — 2.5 FM aux loss type — MSE λ annealing — Cosine decay

### C.4 Timestep Schedule

We use a shifted timestep schedule (Esser et al., 2024) that adapts to the input sequence length. Given T denoising steps, the base schedule is:

k T

, k = 1,...,T, (C.1) which is then shifted via a sequence-length-dependent parameter µ:

σ˜k =

eµ eµ + 1/σ˜k − 1

Lseq − 256 4096 − 256

, (C.2)

, µ = 0.5 + 0.65 ·

σk =

where Lseq is the spatial sequence length (number of patches). The final schedule is (tmin,σ1,...,σT) in ascending order, with tmin drawn per-sample from Uniform[tlomin,thimin] for robustness across noise levels during training.

### C.5 Classifier-Free Guidance

At inference, we use a logits-guided formulation of classifier-free guidance (Ho and Salimans, 2022) that operates on the coupling parameters rather than the predicted sample. Given conditional predictions (µc,σc) and unconditional predictions (µu,σu) from the predictor, the guided parameters are:

2

σc σu

(clipped to [0,1]), (C.3)

s =

σc √1 + w − w · s

, (C.4)

σeff =

(1 + w)µc − w · s · µu 1 + w − w · s

, (C.5)

µeff =

where w is the guidance scale. This formulation is inspired by the logit-space interpretation: it corresponds to (1+w)log pc−w log pu applied to the Gaussian coupling in u-space, which naturally adjusts both the mean and scale (unlike standard linear guidance that only modifies the mean).

## D Evaluation Benchmarks

### D.1 GenEval

GenEval (Ghosh et al., 2023) is a compositional text-to-image evaluation benchmark that tests fine-grained generation capabilities across six task categories:

- • Single object: generating a single named object correctly.
- • Two objects: generating two distinct objects in the same scene.
- • Counting: producing the exact number of objects specified.
- • Colors: assigning the correct color to objects.
- • Position: placing objects in the specified spatial relationship (e.g., “left of”, “above”).
- • Color attribution: binding the correct color to the correct object when multiple colored objects are described.

Each task is scored by an object-detection model that verifies whether the specified objects, attributes, and relations are present. The overall score is the average accuracy across all six tasks. We generate 4 images per prompt and report the average detection rate.

### D.2 DPG-Bench

DPG-Bench (Hu et al., 2024) (Dense Prompt Graph Benchmark) evaluates text-to-image alignment using long, detailed prompts that describe complex scenes with multiple entities, attributes, and relations. Unlike GenEval which uses short compositional prompts, DPG-Bench tests whether models can faithfully follow dense, paragraph-length descriptions. Evaluation is performed using a VQA model (BLIP-2) that answers questions about the generated image corresponding to each semantic element in the prompt. The benchmark reports scores across five L1 categories:

- • Attribute (color, shape, size, texture, other)
- • Entity (part, state, whole)
- • Global (overall scene coherence)
- • Other (counting, text rendering)
- • Relation (spatial, non-spatial)

The overall DPG-Bench score is the average across all L1 categories, reported as a percentage.

### D.3 Class-Conditional ImageNet

As a proof-of-concept for training NTM from scratch, we evaluate on class-conditional ImageNet 256×256 using the FAE latent space (Gao et al., 2025) (16× spatial compression, 32-dim latents). table 6 reports FID-50K for NTM at different step counts alongside representative baselines.

Table 6 Class-conditional ImageNet 256×256 (FID-50K). Steps: total sequential generation steps (denoising or autoregressive). NTM achieves competitive FID with significantly fewer steps than prior normalizing flows, using only the NLL training objective without distribution-level losses.

Method Type #Params Steps FID↓

DiT-XL/2 (Peebles and Xie, 2023) DM 675M 250 2.27 SiT-XL (Ma et al., 2024) DM 675M 250 2.06

LlamaGen (Sun et al., 2024) AR 3.1B 256 2.18 VAR (Tian et al., 2024) AR 2.0B 10 1.73 DART (Gu et al., 2024) AR 820M 16 3.82

TarFlow (Zhai et al., 2025) NF 1.4B 1024 5.56 STARFlow (VAE) (Gu et al., 2026) NF 1.4B 1024 2.40 STARFlow (FAE) (Gao et al., 2025) NF 1.4B 256 2.67

NTM (Ours) NF 1.4B 4 3.83 NTM (Ours) NF 1.4B 8 3.24 NTM (Ours) NF 1.4B 16 2.80

NTM achieves 2.80 FID with 16 steps—comparable to STARFlow (FAE) at 2.67 which requires 256 autoregressive steps—demonstrating that the normalizing flow framework can produce competitive results with dramatically fewer sequential steps. Notably, these results use only the exact NLL training objective without any distribution-level losses (e.g., adversarial or perceptual). Recent work (Yin et al., 2024b,a; Yang et al., 2026) has shown that distribution-level finetuning can substantially boost few-step generators beyond their base performance; NTM’s stable exact-likelihood training makes it a natural candidate for such post-training enhancement, which we leave for future work.

- E Additional Qualitative Results We show additional samples from our trained NTM models in Figure 10.

[Figure 9]

###### Figure 10 Additional examples from NTM trained from scratch (left) and fine-tuned from flow matching (right) under the same text prompts.

## F Broader Impact

NTM advances the state of the art in efficient image generation by enabling high-quality few-step sampling with exact likelihood. While improved generative models have many beneficial applications—including creative tools, data augmentation, and scientific visualization—they also raise concerns around potential misuse for generating misleading or harmful content. We believe that developing models with exact likelihood (as opposed to implicit or adversarial formulations) is a step toward more controllable and auditable generation, since the tractable density can support downstream applications such as anomaly detection and content verification. We encourage the research community to develop complementary safeguards, including watermarking and provenance tracking, alongside advances in generative modeling.

Apple and the Apple logo are trademarks of Apple Inc., registered in the U.S. and other countries and regions.

