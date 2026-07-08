# arXiv:2606.19162v1[cs.LG]17Jun2026

## The Reward Was in Your Data All Along: Correcting Flow Matching with Discriminator-Guided RL

Nicolas Beltran-Velez1,2, Felix Friedrich1, Zhang Xiaofeng1,3,5, Reyhane Askari-Hemmat1, Xiaochuang Han1, Adriana Romero-Soriano1,3,4,6,∗, Michal Drozdzal1,∗

1FAIR at Meta, 2Columbia University, 3Mila – Québec AI Institute, 4McGill University, 5Université de Montréal, 6Canada CIFAR AI Chair

∗Joint senior authors.

Score- and flow-matching models often rely on preference-based reinforcement learning for two purposes: aligning with subjective preferences and, surprisingly, recovering properties—such as visual realism and coherent object structure—that matching-based training is intended to learn from the data itself. We argue that this reflects a structural mismatch. Matching losses measure ℓ2 regression error on the velocity or score field under training-time marginals, a proxy poorly aligned with the visual and semantic properties that determine sample quality at inference. Given a reward aligned with these properties, RL sidesteps the mismatch by evaluating the model on its own samples and following the reward landscape directly. The challenge is to obtain such a reward without relying on human preferences, which are expensive and conflate data realism with annotator inclinations. We propose Discriminator-Guided RL (DRL). DRL trains a discriminator to separate data from base-model samples in a pretrained representation space and uses its logit as the reward in KL-regularized RL. The pretrained space restricts the discriminator to perceptually meaningful directions, and the logit estimates the log-likelihood ratio between data and model, which is the optimal reward for targeting the data distribution. Across SiT, JiT, REPA, and RAE, DRL reduces guidance-free FID (e.g., 9.38 → 2.62 on SiT) and semantic-space FD (e.g., 88.2 → 19.3 on DINOv3 for SiT), with consistent gains across all backbones, and improves human-preference rewards without training on them. It also yields a better Pareto frontier between preference reward and image fidelity under subsequent preference-based post-training, increasing alignment while reducing low-level artifacts such as oversaturation and excessive brightness.

Date: June 18, 2026 Correspondence: Nicolas Beltran-Velez, nb2838@columbia.edu

[Figure 1]

[Figure 2]

(a) Base model (REPA SiT-XL/2) (b) After DRL (ours)

Figure 1 Seed-matched samples from REPA before and after DRL post-training with CFG = 1. Our DRL model produces sharper, more coherent images even without using any CFG.

- 1 Introduction

Flow-based models have become the dominant paradigm for generative modeling in continuous domains due to their simulation-free, regression-based training objectives (Lipman et al., 2022; Albergo and Vanden-Eijnden, 2023; Ho et al., 2020; Song et al., 2020). In practice, however, these models are rarely trained only with these losses. Instead, they follow a multi-stage pipeline: (1) a flow or score matching (FSM) objective is used to fit a base model pbase to a data distribution q; and (2) reinforcement learning (RL) is used to tilt pbase towards desirable regions under a reward r(x).

The standard motivation for RL post-training is that it enables fitting a model to the implicit distribution p∗(x)∝pbase(x)exp(r(x)) for which no data exist but which is easy to specify via a reward (Xu et al., 2023; Black et al., 2024; Fan et al., 2023; Schuhmann et al., 2022). Nevertheless, in practice, RL post-training is also used to enforce properties that are already present in the data, like visual realism, coherent object structure (Wallace et al., 2024; Domingo-Enrich et al., 2025), or physically correct motion (Ye et al., 2025; Liu et al., 2025b). This is both puzzling—why can the model not learn these real-world properties from the data directly but recover them with RL?—and undesirable, as preference data, which is the norm for training the rewards in these applications, is expensive to collect and conflates the real-world properties we want to preserve with subjective concerns. This raises two questions that we focus on in this paper: (1) Why do matching-based objectives fail to capture properties that are present in the data but that RL helps recover? (2) Can we use RL to correct these failures without relying on preference-based rewards or data?

First, we argue that the fact that RL recovers properties already present in q points to the matching objective, rather than data or capacity, as a plausible bottleneck. To build intuition, we study a simplified setting with probability-flow ODEs and ask when FSM losses control the property gap Ep

[r]−Eq[r] on a property r of interest. The guarantees we find are weak. In the worst case, none exist: the FSM loss is measured under the interpolation marginals qt, but sampling is governed by the model’s rollout distribution pt (Proposition 3.1). Under regularity assumptions a guarantee does exist. However, it degrades with the reward’s ℓ2 Lipschitz constant (Proposition 3.2), which we expect to be large for many properties of interest. Although these bounds are worst-case, they suggest how RL can help: by construction it avoids both obstructions, evaluating the model on its own trajectories rather than qt and following the reward landscape directly instead of the geometry of velocity space.

base

However, RL is only useful if we optimize a reward that is a good proxy for the properties of q we want to preserve, and a priori we do not have one. To this end, motivated by the analysis above, we introduce Discriminator-Guided RL (DRL). DRL trains a discriminator to estimate the density ratio between the data and the model’s distributions in a pretrained self-supervised learning (SSL) representation space. The logit of this discriminator then serves as the reward for KL-regularized RL. Working in an SSL space is the central design choice: it restricts the reward to use discrepancies visible through the representation, both making density-ratio estimation tractable and confining corrections to semantically meaningful axes—all without ever relying on preference data.

We apply DRL to state-of-the-art image generation architectures, SiT (Ma et al., 2024), JiT (Li and He, 2025), REPA (Yu et al., 2024), and RAE (Zheng et al., 2025), and show that even with a simple linear discriminator, DRL delivers large improvements over the base model across the board (Figure 1), measured by Fréchet distances in multiple feature spaces. These gains directly translate into better scores under held-out human-preference rewards, despite DRL never being trained on such rewards. Beyond improving the base model, DRL also strengthens the post-training pipeline that follows. Applying preference-based RL (PRL) on top of DRL rather than the base model yields a better reward–distortion Pareto frontier, since DRL absorbs the corrections recoverable from q and leaves PRL to handle genuine subjective preferences. Full experimental details are in Section I.

Overall, our results suggest viewing RL post-training of flow-based models not only as a way to optimize external preferences, but also as a complementary mechanism for recovering structure in the training data that is imperfectly captured by standard matching-based objectives.

- 2 Preliminaries

Flow and Score Matching. Flow and diffusion models generate data by transporting a simple base distribution, typically Gaussian noise, to the data distribution q (Ho et al., 2020; Song et al., 2020; Lipman et al., 2022; Albergo and Vanden-Eijnden, 2023; Liu et al., 2022). A standard training construction introduces independent random variables X1 ∼ q and X0 ∼ N(0,I), and defines the auxiliary interpolation (Albergo and Vanden-Eijnden, 2023; Lipman et al., 2022; Liu et al., 2022)

Xt = α(t)X1 + β(t)X0, where t ∈ [0,1], α(0)=0, β(0)=1, α(1)=1, β(1)=0. (1)

This induces a family of marginals qt interpolating between q0 =N(0,I) and q1 =q. Sampling from q then reduces to learning the velocity field vt(x):=E[α˙(t)X1 + β˙(t)X0 | Xt=x] or the score field st(x)=∇x log qt(x), since the SDE

dXt = vt(Xt) + 21σ(t)2st(Xt) dt + σ(t)dWt, X0 ∼ N(0,I), (2)

has marginals qt for any noise schedule σ(t) (Song et al., 2020; Lipman et al., 2022; Albergo and VandenEijnden, 2023); see Section C.2. Both fields are conditional expectations, and so admit simulation-free ℓ2 regression targets (Hyvärinen and Dayan, 2005; Vincent, 2011; Ho et al., 2020; Song et al., 2020; Lipman et al., 2022; Albergo and Vanden-Eijnden, 2023): conditional flow matching (CFM) and denoising score matching (DSM) use, respectively,

E vθ(Xt,t) − α ˙(t)X1 + β˙(t)X0

2

and E sθ(Xt,t) + Xt − α(t)X1 /β(t)2 2 , (3)

with expectation over (t,X0,X1). Under Eq. (1), vt and st are recoverable from each other (Karras et al., 2022; Domingo-Enrich et al., 2025) (see Section C.1), so learning one suffices; we write LFSM for either objective when the distinction is unimportant.

RL Post-training for Flow Models. Given a pretrained model pbase and a reward r:X →R, KL-regularized RL (Ziebart, 2010) aims to move probability mass toward high-reward regions without drifting too far from the base model by solving

maxp Ex∼p[r(x)] − (1/λ) KL(p∥pbase),

or equivalently minimizing the reverse KL, KL(p∥p∗), where p∗(x) ∝ exp(λr(x))pbase(x) and λ > 0 is a hyper-parameter controlling the reward-KL trade-off.

Unfortunately, directly optimizing this objective is infeasible for flow models as the endpoint densities required by the KL term are unreliable. Instead, the standard approach is to use a nonzero noise schedule σ(t) in the sampling SDE (Eq. (2)) and replace the endpoint regularization by a KL over path distributions (Fan

- et al., 2023; Domingo-Enrich et al., 2025), which can be analytically evaluated using Girsanov’s theorem

(Girsanov, 1960; Domingo-Enrich et al., 2025). If Pθ and Pbase denote the trajectory distributions induced by the fine-tuned and base models under the same training SDE, the new objective is then

LRL(Pθ) := −Ex∼p

θ

[r(x)] + (1/λ) KL(Pθ ∥Pbase). (4)

If training uses a specific, so-called memoryless σ(t), the minimizer of Eq. (4) has the same endpoint as the original KL-regularized objective, and admits sampling after training with any noise schedule (DomingoEnrich et al., 2025). This objective can then be optimized with standard policy gradient methods such as REINFORCE, PPO, or GRPO (Williams, 1992; Black et al., 2024; Fan et al., 2023; Schulman et al., 2017; Shao

- et al., 2024; Liu et al., 2025a), or with adjoint-based methods if the reward is differentiable (Domingo-Enrich
- et al., 2025); details are deferred to Section F.

- 3 Motivation: Understanding the Limitations of Flow and Score Matching

To motivate our method, we first ask why FSM losses might fail to capture properties of the data distribution that RL is able to recover. With infinite capacity and perfect optimization, the minimizer of either matching loss should reproduce the data distribution q exactly. Therefore, a natural starting point is to consider

how these losses behave in the setting where optimization is only approximate. We analyze the ODE-based formulation for simplicity.

Let r : X → [0,1] quantify a property of interest in the data—say whether an object in an image is well formed. A model that preserves this property adequately satisfies Ep

[r]≈Eq[r]. Writing LFSM for either matching objective and v∗ for its minimizer (with overloaded notation for the score), we ask: when does

v

[r] − Eq[r] → 0 as ε → 0, and at what rate?

LFSM(v) − LFSM(v∗) ≤ ε imply Ep

v

Our first result shows that, in the worst case, no such rate exists: low FSM loss can be arbitrarily uninformative about the reward gap.

- Proposition 3.1 (No reward certificate from FSM). Fix a (sufficiently regular) q and a bounded reward r ∈ [0,1] with r(x) ≤ η on some region B ⊂ X. For every ε > 0 and every δ > 0 there is a velocity field v and a score field s, whose training error under qt is at most ε uniformly in t, but whose probability-flow ODE endpoint law p satisfies Ep[r] ≤ η + δ, regardless of Eq[r]. A precise statement and proof is given in Proposition D.3.

0 1

t

−1.0

−0.5

0.0

0.5

1.0

x

|qt band<br><br>true paths<br><br>rollout|
|---|

Figure 2 Distribution shift. Rollout paths (orange) drift off the noising band qt (gray) and miss the high-reward region at t=1. The further from qt, the larger the error of true field (gray) vs. learned field (orange). Construction in Proposition D.3.

The construction exploits a train–test mismatch. An early error pushes the trajectory into regions of low qt-mass, so subsequent errors occur at states unseen during training and compound along the rollout (Figure 2). The same pathology motivates DAgger in behavioral cloning (Ross et al., 2011), with qt and pt playing the roles of expert and learner.

This counterexample is, however, adversarial. Under regularity assumptions on q and v∗ a quantitative reward guarantee does exist — though, as we will see, a loose one.

- Proposition 3.2 (Reward certificate under uniform velocity control). Let r : X → [0,1] be Lr-Lipschitz in that |r(x) − r(y)| ≤ Lr∥x − y∥ for all x,y, and let v∗(·,t) be Lv-Lipschitz

in x uniformly in t ∈ [0,1]. If supt,x ∥v(x,t) − v∗(x,t)∥ ≤ ε, then the endpoint laws p and q of the probability-flow ODE satisfy

Ep[r] − Eq[r] ≤ εLr ((eL

− 1)/Lv). (5)

v

This dependence on Lr, Lv, and ε is essentially tight, and an on-policy variant under the rollout marginals pt holds as well; see Section D.2.1 for a formal statement and proof.

The bound factorizes into three terms: the velocity error ε, which FSM directly minimizes; the factor (eL

−1)/Lv, which controls how errors compound as the ODE is integrated from t=0 to 1; and the reward Lipschitz constant Lr, which measures how sharply r responds to small ℓ2 changes in the sample.

v

reward loss

large reward change

small loss change

FSM loss

v∗ vbad

The last term exposes a second, more practically relevant obstruction: FSM controls errors in velocity space, while the reward is a function of samples, and when the two geometries are misaligned small velocity errors can map to large reward errors (Figure 3). The constant Lr quantifies this: rearranged, the bound says that to achieve a reward gap of size δ we need ε ≈ δ/Lr, so the sharper the reward, the smaller the velocity error must be. For rewards of practical interest, this is unfavorable. For example, in pixel space, whether a hand or animal face looks well-formed can flip under a few edge pixels, making Lr effectively very large and the required ε correspondingly small. Similar phenomena will occur in any representation space not directly aligned with the reward.

Figure 3 Geometry mismatch. A small step in velocity space can produce large reward changes.

For matching losses, this is doubly unfavorable as the gradient signal-to-noise ratio degrades precisely in the small-ε regime

the bound demands. With ξ := Y − v∗(Xt,t) and E[ξ | Xt,t] = 0, the per-sample gradient of the CFM loss splits as

gθ = 2Jθ⊤(vθ − v∗) − 2Jθ⊤ξ, Jθ := ∇θvθ(Xt,t).

The first term is the optimization signal and vanishes as vθ → v∗; the second is irreducible noise from the conditional variance of the regression target and does not. The achievable ε can therefore plateau well above what Eq. (5) requires. A similar decomposition can be derived for score matching.

RL’s edge. These obstructions suggest how RL helps. First, it evaluates the model on the samples it generates, so there is no train–test mismatch. Second, and more importantly, with a reward in hand, we can use the reward landscape directly, focusing optimization on the directions of reward improvement and bypassing the geometry of velocity or score space. Moreover, while generally loose in practice, when the reward is such that the KL-regularized optimum is the desired target q and the noise schedule is fixed, the RL objective controls the property gap directly: a Pinsker-style bound (Section D.3.1) gives

|Ep[r] − Eq[r]| ≤

λ 2

(LRL(p) − LRL(q)).

This verifies that we optimize the right quantity in the idealized setting. The implication is that even with perfect data, RL post-training can be beneficial when a good reward is available: the reward provides an optimization signal directly aligned with the property of interest, whereas FSM must recover that property indirectly from the regression target alone.

To show that these distinctions matter in practice, we compare RL training against supervised distillation from an RL teacher. Starting from Stable Diffusion 1.5 (Rombach et al., 2022), we first train an RL teacher on ImageReward using adjoint matching (Domingo-Enrich et al., 2025), and then use its samples to fine-tune another Stable Diffusion student using score matching. If score matching could recover the reward-relevant properties learned by RL, the student should match the teacher. In practice it does not: the student stalls well below the teacher’s reward and fails to close the gap (Figure 4). See Section I.4 for the full setup.

- 4 Method: Discriminator-Guided RL

| | | | | |
|---|---|---|---|---|
| | | |RL–|FSM<br><br>|
| | | | | |
| | | | | |
| | |FSM|from p|RL|
| | |pRL (t|eacher|)|
| | | | | |

−0.25

gap

ImageReward

−0.50

−0.75

−1.00

−1.25

0 2k 4k 6k 8k

Optimization steps

Figure 4 Distillation gap on Stable Diffusion 1.5: a score-matching student trained on samples from an RL teacher plateaus well below the teacher’s reward.

Unfortunately, RL is only beneficial if we have a reward that captures the aspects of q that FSM may have failed to learn. Finding such a reward is non-trivial. The standard approach, learning a reward from human preferences, is only incidentally aligned with this goal. While preference rewards can correlate with q-properties (e.g., annotators prefer well-formed faces), they can also drag optimization along orthogonal axes like aesthetic appeal, so optimizing them is not necessarily desirable. Moreover, preference data is expensive to collect, and hard to define for many domains beyond images and video. We would instead like a reward derived directly from q. To this end, we introduce Discriminator-Guided RL (DRL).

From samples to rewards. The starting point of DRL is the observation that, in KL-regularized RL, there is a reward whose optimum is exactly q. Recall from Section 2 that, at λ=1, the unique maximizer of the KL-regularized objective is p∗(x)∝exp(r(x))pbase(x). Forcing p∗=q identifies the ideal reward, up to additive constants, as the log density ratio between target and reference,

q(x) pbase(x)

r∗(x) = log

. (6)

This ratio is intractable in closed form, but it can be estimated from samples. Training a discriminator D:X → (0,1) to separate q from pbase under the logistic loss has a well-known optimum, D∗(x) = q(x)/(q(x)+pbase(x)).

Hence, its logit recovers r∗. This motivates the reward estimator

D(x) 1 − D(x)

rˆ(x) := log

(7)

In practice, we parametrize the logit rˆ directly and use a sigmoid to define the discriminator.

Using representations. Estimating r∗ directly in the flow output space is, however, both statistically hard and semantically unreliable. For example, a discriminator might separate q from pbase using artifacts that have no bearing on the relevant properties, achieving low classification error without producing a useful estimate of the density ratio (see design choices). We therefore constrain the reward to a pretrained representation space. Given a frozen encoder ϕ:X →Z, we set rˆ(x)=h(ϕ(x)) for a learned head h. This both reduces the dimensionality of the estimation problem and restricts the discriminator to ϕ-visible structure.

###### Algorithm 1 Discriminator-Guided RL (DRL)

Stage 1: Reward Estimation

###### Require: Samples x ∼ q, encoder ϕ, base model pbase

- 1: repeat
- 2: Sample xreal ∼ q, xfake ∼ pbase
- 3: ℓdisc(ψ) ← − log Dψ(ϕ(xreal)) − log 1−Dψ(ϕ(xfake))
- 4: Update ψ by descending ℓdisc(ψ)
- 5: until convergence
- 6: Define rˆ(x) ← logit Dψ(ϕ(x))

Stage 2: KL-Reg. RL via Adjoint Matching

Require: Reward rˆ, base velocity vbase, KL weight λ Require: Schedule σ(t), grid 0=t0< · · · <tK=1

- 1: Initialize vθ ← vbase
- 2: repeat
- 3: Sample {Xk} from dXt = [2vθ − 1t Xt] dt + σ(t) dBt

- 4: a˜1 ← −λ ∇xrˆ(XK);
- 5: Solve backward
- 6: a˜˙t = −a˜⊤t ∇x[2vbase(Xt, t) − 1t Xt]

- 7: Set LAM(θ) ←
- 8: 12 Kk=0 σ( 2t

k) vθ−vbase + σ(tk)˜ak 2

- 9: Update θ by descending LAM(θ)
- 10: until convergence
- 11: return vθ

Mathematically, this restriction means that DRL will generally never be able to target q exactly. Nevertheless, its new target has a simple and intuitive characterization. It is the solution to the following constrained optimization problem:

minp KL(p∥pbase) subject to pϕ = qϕ, (8) where pϕ,qϕ denote the pushforwards of p,q under ϕ (see Proposition E.1 for a formal statement and proof). In words, DRL makes the smallest KL change to pbase that aligns its representation-space distribution with that of the target, while leaving any variation invisible to ϕ unchanged; choosing ϕ thus chooses which aspects of q DRL is allowed to correct.

Finally, when the learned reward is imperfect, we recover a feature-space test-function bound that augments the standard KL-regularized RL suboptimality bound with an additional term capturing the expected discrepancy between the ideal target in Equation (8) and the target implied by the learned reward. We defer the precise statement and proof to Section E.1.2.

The DRL pipeline. These two ideas combine into the DRL algorithm specialized to flow models summarized in Algorithm 1. Stage 1 trains a discriminator in a frozen representation space to distinguish q from pbase and defines the reward as its logit. Stage 2 fine-tunes the base model with KL-regularized RL under that reward. To exploit reward gradients, we instantiate the RL stage with adjoint matching (Domingo-Enrich et al., 2025), a state-of-the-art RL algorithm for flow models. Further implementation details for both stages are deferred to Section H. A short description of adjoint matching is given in Section F.

- 5 Experiments

We validate DRL by using it to fine-tune four ImageNet-pretrained flow models: SiT (Ma et al., 2024), JiT (Li and He, 2025), REPA (Yu et al., 2024), and RAE (Zheng et al., 2025), covering latent- and pixel-space architectures. Notably, RAE and REPA already use SSL representations during pretraining as the latent space and regularization respectively. Unless noted otherwise, the discriminator is a class-conditional linear projection head (Miyato and Koyama, 2018) on frozen DINOv2-Large (Oquab et al., 2024) features, trained for 10k steps. The RL stage is 3k steps of adjoint matching (Domingo-Enrich et al., 2025). Importantly, this is only a small fraction of the 1M+ steps typically used to pre-train these models. Full description of every experimental setup, evaluation protocol, and hyperparameter is in Section I. Finally, we run adjoint matching with a new local-linear integrator we introduce for the memoryless SDE required for RL; we found it essential for stable training and view it as an important contribution of independent interest but leave its full description in App. Section G.

λ=1 λ *

Base

| |
|---|

Base λ=1 λ *

Base λ=1 λ *

No CFG

#### Best CFG

242 95.5 54.4 159 58.3 40.3 148 97.3 42.8 37.5 30.2 20.6 88.2 43.8 19.3 63.9 29.4 15.6 33.9 23.7 11.3 6.66 5.71 4.49 42.6 20.2 16.9 31.6 14.3 12.8 39.4 29.7 21.0 10.2 9.54 9.13 9.38 2.62 2.62

47.0 42.4 42.4 36.9 32.9 32.9 41.5 36.1 30.6 25.9 23.7 20.2 21.6 18.3 16.4 18.6 14.6 13.0

SiT REPA

DINOv2DINOv3Incep.SigLIP

JiT RAE

SiT REPA

- 10.9 10.2 8.10 4.83 4.57 4.26
- 11.8 12.2 11.6 9.52 10.0 9.32 18.2 17.1 16.5 9.50 9.29 9.06 1.50 2.43 1.78 1.21 1.84 1.50 1.91 1.87 1.87 1.28 1.25 1.19

JiT RAE

SiT REPA

JiT RAE

SiT REPA

- 6.48 2.14 2.14
- 7.16 3.72 2.73 1.31 1.38 1.29

JiT RAE

0.25 0.5 0.75 1 1.25 1.5

0.25 0.5 0.75 1 1.25 1.5

FD / Base

FD / Base

Figure 5 Distribution alignment. Each row shows DRL’s FD normalized by the corresponding Base FD (vertical line); lower is better. Squares show λ=1 and filled circles show the value optimized over λ. Raw FD values in tables:

indicates improvement, indicates degradation over Base. Most markers fall left of the Base line, showing broad alignment gains, with the most significant gains in the no-CFG setting.

DRLReduces the Distributional Gap Along Semantically Meaningful Directions. We first investigate whether DRL reduces the distributional gap to the data along semantically meaningful directions by computing Fréchet Distance (FD) (Heusel et al., 2017) in four feature spaces: DINOv2 (Oquab et al., 2024), DINOv3 (Siméoni et al., 2025), SigLIP (Zhai et al., 2023), and InceptionV3 (Szegedy et al., 2016). We consider two settings: the theoretically motivated unit λ and a tuned λ⋆ value selected over 1,5,10,20,40. We consider values larger than 1 because they often improve RL optimization in practice. To stabilize training at larger λ, we found it helpful to add an R1 penalty (Mescheder et al., 2018), which penalizes the discriminator’s input gradient on real samples and smooths the reward landscape (see design choices for more details).

Figure 5 plots FD ratios (DRL/Base) for each (model, feature) pair, alongside their raw FD values with and without tuned CFG. Without CFG, DRL substantially improves distribution alignment: the tuned λ⋆ improves all 16 model–feature pairs, and λ = 1 improves all but one. The gains are especially large in DINOv2 and DINOv3, where FD often drops by more than half. With the best CFG setting, the Base models are already much closer to the data, but DRL still further improves alignment in most cases. In particular, λ⋆ improves FD in 14/16 pairs, with consistent gains in DINOv2, DINOv3, and SigLIP. The only clear exceptions are InceptionV3 for SiT and REPA under CFG, where FD increases slightly relative to Base. Across both settings, tuning λ consistently improves over the unit-weight setting, indicating that pushing past the theoretical value pays off in practice.

To further investigate the role of λ, Figure 6 plots FD against λ in DINOv2 space alongside the corresponding precision–coverage trajectory. FD bottoms out around λ ∈ [5,10] and then degrades; coverage drops at λ=20, and by λ=40 (not pictured; see Figures 17 and 18) the trajectory has collapsed off the frontier—the model overfits to the discriminator’s reward signal. We adopt λ=10 as the default for all downstream experiments, since it sits at the knee of this trade-off.

Additional metrics and λ ablations for each model– feature pair are in Section J.1. Importantly, KD and FDval follow the same trends despite not being used to tune λ.

RAE SiT JiT REPA base λ=10

FD vs. λ

Precision vs. Coverage

| | |
|---|---|
| | |
| | |
| | |
| | |

↓FD/baseFD

0.8

1

Coverage

.75

0.6

.5

.25

1 5 10 20 40

0.6 0.8

λ

Precision

Figure 6 FD vs. λ and Precision–Coverage in DINOv2-L. FD decreases through λ=5−10 and degrades at high λ as coverage drops.

Improvement/σbase

ImageReward ↑

HPSv2 ↑

Aesthetic ↑

PickScore ↑

1.0

1.0

0.4

0.5

0.5

0.5

0.2

0.0

0.0

0.0

0.0

REPA SiT JiT RAE

REPA SiT JiT RAE

REPA SiT JiT RAE

REPA SiT JiT RAE

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

No CFG

Best CFG

REPA

SiT

JiT

RAE

Figure 7 Held-out preference reward gains. Normalized improvement (rDRL − rbase)/σbase at λ=10 on four rewards, with and without CFG, where σbase is the base-model reward standard deviation. DRL improves every reward without ever seeing preference data.

DRL Improves Image Quality Without Preference Data. We next examine whether these distributional gains translate into perceptually better images. As proxies for image quality, we use four held-out preference reward models trained on human comparisons: ImageReward (Xu et al., 2023), PickScore (Kirstain et al., 2023), Aesthetics v2.5 (discus0434, 2024), and HPSv2 (Wu et al., 2023). Figure 7 plots the per-model improvement on each reward, (rDRL − rbase)/σbase, both without CFG and with best CFG. Without preference data or access to these rewards during training, DRL improves every reward on every architecture, with the largest gains for SiT and REPA and smaller but consistent gains for JiT and RAE.

These improvements are also evident visually. The accompanying samples show fixed-seed outputs at λ values of 1, 10, and 20. As λ grows, shapes sharpen and global structure becomes more coherent, e.g., the spider’s shape, the koala’s face, and the car’s front end all become noticeably more defined. We provide more samples and λ values in App. Figures 30 to 33.

DRL Provides a Better Foundation for Preference Alignment. DRL improves distribution matching, but aligning to genuine subjective preferences still requires fine-tuning with preference-based RL (PRL). We argue that DRL also makes PRL itself more effective. PRL from the base model is asked to do two things with one imperfect scalar reward: repair distributional errors left by the generative model and optimize subjective preference. The result is a reward–drift trade-off: small λPRL leaves structural failures uncorrected; large λPRL exploits the proxy along nuisance directions like oversaturation or excessive brightness (Domingo-Enrich et al., 2025). By handling part of the data repair before preference optimization begins, DRL should ease this trade-off and let PRL focus on genuinely subjective improvements.

Base DRL

ref λ=1 λ=10 λ=20

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

RAEJiT-H/16REPASiT-XL/2

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

We test this by running KL-regularized PRL from either the base model or the DRL checkpoint. We train with ImageReward (Xu et al., 2023) at λPRL ∈ {1,10,40} and evaluate on the held-out HPSv2 reward (Wu et al., 2023), tracking five lowlevel statistics that commonly drift under aggressive PRL: brightness, saturation, contrast, colorfulness, and whiteness; see Section I.3 for details on how these statistics are computed. Figure 9 plots HPSv2 against each statistic. DRL+PRL shifts the reward–drift frontier upward across all five axes, achieving higher HPSv2 than Base+PRL at comparable drift. The pattern holds with other forms of guidance, and on other rewards, including ImageReward itself (Section J.4).

Figure 8 Effect of λ. Same noise and class label at λ ∈ {1, 10, 20}. Larger λ produces sharper, more coherent samples while preserving content and composition.

DRL+PRL Base+PRL Base ImageNet

###### SiT REPA JiT RAE

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

λPRL 1

0.25

RAEJiTREPASiT

0.23

DRL+PRL

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

0.20

10

0.25

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

0.23

40

0.20

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

λPRL 1

0.22

0.20

Base+PRL

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

10

0.24

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

0.22

40

0.5 0.6

0.2 0.3

0.18 0.21

0.15 0.17

0.4 0.5

Bright.

Sat.

Contr.

Color

Whiteness

- Figure 9 DRL provides a better starting point for PRL. Left: HPSv2 reward against five low-level statistics under PRL from the base model (dashed, hollow) or the DRL checkpoint (solid, filled); green dashed lines mark ImageNet

reference statistics. Right: matched fixed-seed samples at λPRL ∈ {1, 10, 40}. Across statistics, DRL+PRL gives better reward–drift Pareto fronts than Base+PRL and less brightness/whiteness drift as PRL strength increases. See Section I.3 for details on how the statistics are computed.

Samples from the model further support this conclusion. In Figure 9, Base+PRL at low λPRL leaves structural errors unresolved—the JiT dog has a malformed face, the REPA scorpion a shell-like body, the SiT bus a distorted chassis. Larger λPRL corrects some of these, especially for JiT and REPA, but at the cost of noticeably brighter and whiter images. DRL+PRL avoids the trade-off: samples remain structurally coherent and naturally colored across all λPRL. Per-model grids and guided variants appear in Figures 34 to 37.

DRL Performance Cannot Be Reached with Flow Matching Alone. In Section 3 we argued that RL was needed because flow matching cannot faithfully recover many properties of the data distribution. To demonstrate that DRL genuinely learns aspects of the distribution that are hard to learn with flow matching alone, we use a distillation experiment similar to the one in Section 3. Concretely, we take the REPA model post-trained with DRL (λ = 1, R1 = 0) and use it as a teacher by fine-tuning REPA on samples from the teacher. If the features learned by DRL were learnable by the flow matching objective, we should see no gap between the teacher model and the student model. As Figure 10 demonstrates, this is not the case. Despite training for over 900k gradient steps, and seeing more than 50 million samples (over 150× what RL sees and over 40× the size of ImageNet), the student model is not able to faithfully replicate the teacher. This supports the argument from Section 3 that the RL objective matters, rather than only the samples produced by the teacher. We provide further details of the setup of this experiment in Section I.5.

Design Choices. We close by ablating DRL’s two main design choices—the discriminator feature space and the discriminator architecture—on REPA. Full setup is in Section I.

Discriminator design. We compare four discriminators—a linear head and an MLP-2 head on frozen DINOv2 features, a fully fine-tuned DINOv2, and a DINOv2 architecture trained from scratch—at λ=1 with R1=0, and at λ=10 with R1∈{0,10−5,10−3,10−1}. Figure 11 plots DINOv2-L FD across this sweep, with representative samples along the top row. The full R1 sweep at λ=1 is in Section J.5.

pDRL (teacher) Base model SFT from pDRL

Inception

DINOv2

DINOv3

SigLIP

| | |
|---|---|
| | |
| | |
| | |
| | |
| |gap|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| |gap|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| |gap|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| |gap|
| | |
| | |

160

30

60

- 3
- 4
- 5
- 6

140

↓FD()

25

50

120

100

20

40

80

15

30

0 500k Optimization Steps

0 500k Optimization Steps

0 500k Optimization Steps

0 500k Optimization Steps

- Figure 10 DRL performance cannot be reached with flow matching alone. FD (↓) over optimization steps for a student trained with flow matching on samples from a DRL teacher (λ=1, no R1), evaluated in four feature spaces. The student improves rapidly over the base model (grey, dotted) but plateaus well above the teacher (amber, dashed) across all four feature spaces.

Pretrained features are essential: the from-scratch discriminator underperforms despite reaching 95% validation accuracy, and even fine-tuning DINOv2 trails frozen features in the theoretically motivated setting (λ=1, R1=0). At λ=10 without R1, the model collapses to high-reward regions; a small R1 not only stabilizes training but improves over (λ=1,R1=0) in both FD and visual quality. Curiously, while R1 helps at λ=10, it degrades performance at λ=1 (Figure 23), suggesting its role here differs from the vanishing-gradient role it plays in standard GANs (Mescheder et al., 2018).

Finally, since the MLP and linear heads perform similarly, we prefer the linear head for simplicity.

Feature space. We re-train the discriminator on six other frozen embedders—DINOv2-B, DINOv3 (B/L), SigLIP (B/L), and InceptionV3—at λ=1, R1=0, and evaluate FD in every embedder’s space. Table 1 reports a representative subset; the full 7×7 table is in Table 5 with KD as an additional metric. All six SSL embedders improve both FD and KD over the base on every evaluation space, with small spread between them; DINOv2-L is consistently strongest, confirming our default. The one exception is InceptionV3, which barely improves over the base on any SSL evaluation space and is outperformed on its own evaluation space by every SSL embedder—we suspect its classification-only training fails to encode perceptual structure as linear directions. Larger λ and nonzero R1 narrow the gap (FID 3.4, FD 107 on DINOv2-B) but do not close it.

- 6 Related Work

λ = 10 R1 = 10−5

λ = 1 R1 = 0

λ = 10 R1 = 0

Base

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### Linear MLP-2 Finetune Scratch

- 102

- 103

###### FD

base

λ = 1 R1 = 0

λ = 10 R1 = 0

λ = 10 R1 = 10−5

λ = 10 R1 = 10−3

λ = 10 R1 = 0.1

Figure 11 Discriminator ablation. Frozen feature heads work well at λ=1 without R1. At λ=10, removing R1 leads to collapse while a small R1 restores performance. The "from-scratch" discriminator is the worst.

We highlight three primary threads of related work below and delay a more detailed discussion to App. Section A.

RL vs. Imitation Learning. A long line of research has studied the limitations of supervised imitation learning and the benefits of on-policy methods (Ross and Bagnell, 2010; Ross et al., 2011). We extend this line to flow and score matching, showing that they may suffer from similar pathologies despite not being demonstrationbased, with the continuous-time setting yielding even weaker guarantees than standard DAgger-style bounds.

Inverse RL. Our remedy mirrors the structure of inverse RL (Abbeel and Ng, 2004; Ziebart et al., 2008), which recovers a reward from expert demonstrations and trains a policy against it. Generative adversarial imitation learning (GAIL) (Ho and Ermon, 2016) and adversarial inverse reinforcement learning (AIRL) (Fu et al., 2018) apply this idea to standard RL control; we adapt it to improve generative models. Moreover, while these prior works are largely motivated by closing the train–test gap, our observation that the method only helps when paired with an embedder suggests that its principal benefit may be addressing geometric obstructions, not the mismatch itself. We provide further discussion of this point in Section B.

Calibration. Our theory also asks a calibration question: for a property r, does small flow- or score-matching loss imply Ep[r] ≈ Eq[r]? Related work enforces distribution-level constraints directly in controlled generation and fairness-oriented fine-tuning (Khalifa et al., 2021; Shen et al., 2023). Closest to us is Smith et al. (2025), who calibrate generators by finding the KL-closest model satisfying user-specified moment constraints. DRL solves a related KL-constrained problem: it seeks a distribution close to the original model whose pushforward through a feature map matches the data pushforward. Unlike Smith et al. (2025), we focus on aligning the full distribution in high-dimensional SSL feature spaces, and solve the implicit KL-constrained problem by learning a discriminator between real and generated samples, whose logit is then used for RL.

- 7 Conclusions

In this paper, we have argued that for flow-based models, RL is valuable not only as a method for steering toward externally specified rewards, but also as a means of evaluating the model on its own samples and exploiting reward geometry rather than ℓ2 distance. DRL turns this view into a method, and with its discriminator-based reward, consistently improves distributional alignment, preference metrics, and performance–drift tradeoffs under preference-based RL.

Table 1 Feature space ablation (REPA).

λ=1, R1=0, FD (↓). Each row is an evaluation feature space; each column is the feature space used to train the discriminator. Best DRL per row in bold. SSL embedders are L variants. Full FD/KD table in Table 5.

Training Base DINOv2 DINOv3 SigLIP Incep.

Several directions follow: 1) The reliance on frozen SSL features invites the question of whether comparable representations can be learned jointly with the model. 2) Our bounds are worst-case, and quantifying a priori which properties FSM struggles with, as well as how these obstructions translate to practical SDE/ODE samplers, would clarify when an RL stage is helpful; we discuss these limitations further in Section B. 3) More broadly, RL is one way to exploit on-policy evaluation and non-ℓ2 geometry; other sample-based losses like MMD may offer different trade-offs.

Incep. 6.43 2.14 3.02 2.34 5.68 DINOv2 159 58.3 63.2 87.1 151 DINOv3 63.7 29.4 18.5 38.8 62.0 SigLIP 31.2 14.3 16.5 16.1 29.9

Eval

Together our results point to a view of generative training in which matching objectives and RL are not two methods with different goals, but rather two complementary tools for getting the most out of our data.

Acknowledgments

We thank Brian Trippe, Brian Karrer, Ricky T. Q. Chen, and Sebastian Salazar for helpful comments on the manuscript and for insightful discussions about the ideas presented here.

References

Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning, 2004.

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations (ICLR), 2024.

Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In Proceedings

of the International Conference on Learning Representations (ICLR), 2023. V. I. Arnold. Ordinary Differential Equations. MIT Press, Cambridge, MA, 1978. Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement

learning. In The Twelfth International Conference on Learning Representations, 2024. https://openreview.net/ forum?id=YCWjhGrJFD.

Nicholas M Boffi, Michael S Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. TMLR, 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. ImageNet: A large-scale hierarchical image database. In IEEE conference on computer vision and pattern recognition (CVPR), 2009.

discus0434. Aesthetic predictor V2.5: SigLIP-based aesthetic score predictor. https://github.com/discus0434/ aesthetic-predictor-v2-5, 2024. GitHub repository.

Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. In International Conference on Learning Representations (ICLR), 2025.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 2011. Jiajun Fan, Shuaike Shen, Chaoran Cheng, Yuxin Chen, Chumeng Liang, and Ge Liu. Online reward-weighted

fine-tuning of flow matching with Wasserstein regularization. In The Thirteenth International Conference on Learning Representations, 2025.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023.

Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning. In International Conference on Learning Representations (ICLR), 2018.

Igor Vladimirovich Girsanov. On transforming a certain class of stochastic processes by absolutely continuous substitution of measures. Theory of Probability & Its Applications, 1960.

Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in Neural Information Processing Systems, 2014.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. Advances in Neural Information Processing Systems, 2016.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 2020.

Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 2005.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement learning via self-distillation,

2026. https://arxiv.org/abs/2601.20802. Alexia Jolicoeur-Martineau, Rémi Piché-Taillefer, Rémi Tachet des Combes, and Ioannis Mitliagkas. Adversarial score matching and improved sampling for image generation. arXiv preprint arXiv:2009.05475, 2020. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 2022. Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 2024. Muhammad Khalifa, Hady Elsahar, and Marc Dymetman. A distributional approach to controlled text generation. In The International Conference on Learning Representations (ICLR), 2021.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-Pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37:122458–122483, 2024.

Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. DART: Noise injection for robust imitation learning. In Conference on robot learning. PMLR, 2017.

Mingxiao Li, Tingyu Qu, Ruicong Yao, Wei Sun, and Marie-Francine Moens. Alleviating exposure bias in diffusion models through sampling with shifted time steps. arXiv preprint arXiv:2305.15583, 2023.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025.

Yangming Li and Mihaela van der Schaar. On error propagation of diffusion models. arXiv preprint arXiv:2308.05021, 2023.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL, 2025a. https://arxiv.org/abs/2505.05470.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for GANs do actually converge? In International conference on machine learning, pages 3481–3490. PMLR, 2018.

Takeru Miyato and Masanori Koyama. cGANs with projection discriminator, 2018. https://arxiv.org/abs/1802.05637. Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. Reliable fidelity and

diversity metrics for generative models. In International Conference on Machine Learning (ICML), 2020. Andrew Y Ng and Stuart Russell. Algorithms for inverse reinforcement learning. In International Conference on Machine Learning (ICML), pages 663–670, 2000.

Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, and Itir Onal Ertugrul. Elucidating the exposure bias in diffusion models. arXiv preprint arXiv:2308.15321, 2023a.

Mang Ning, Enver Sangineto, Angelo Porrello, Simone Calderara, and Rita Cucchiara. Input perturbation reduces exposure bias in diffusion models. arXiv preprint arXiv:2301.11706, 2023b.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pages 745–750, 2007.

Peter Potaptchik, Cheuk-Kit Lee, and Michael S Albergo. Tilt matching for scalable sampling and fine-tuning. arXiv preprint arXiv:2512.21829, 2025.

Nived Rajaraman, Lin Yang, Jiantao Jiao, and Kannan Ramchandran. Toward the fundamental limits of imitation learning. Advances in Neural Information Processing Systems, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Stéphane Ross and Drew Bagnell. Efficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 661–668. JMLR Workshop and Conference Proceedings, 2010.

Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer, 2024.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Xudong Shen, Chao Du, Tianyu Pang, Min Lin, Yongkang Wong, and Mohan Kankanhalli. Finetuning text-to-image diffusion models for fairness. arXiv preprint arXiv:2311.07604, 2023.

Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. Self-distillation enables continual learning, 2026. https://arxiv.org/abs/2601.19897.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Henry D Smith, Nathaniel L Diamant, and Brian L Trippe. Calibrating generative models to distributional constraints. arXiv preprint arXiv:2510.10020, 2025.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Gokul Swamy, Sanjiban Choudhury, J Andrew Bagnell, and Steven Wu. Of moments and matching: A game-theoretic framework for closing the imitation gap. In International Conference on Machine Learning, pages 10022–10032. PMLR, 2021.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 2011. Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming

Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 1992.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion GANs. arXiv preprint arXiv:2112.07804, 2021.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. UFOGen: You forward once large scale text-to-image generation via diffusion GANs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Haotian Ye, Kaiwen Zheng, Jiashu Xu, Puheng Li, Huayu Chen, Jiaqi Han, Sheng Liu, Qinsheng Zhang, Hanzi Mao, Zekun Hao, Prithvijit Chattopadhyay, Dinghao Yang, Liang Feng, Maosheng Liao, Junjie Bai, Ming-Yu Liu, James Zou, and Stefano Ermon. Data-regularized reinforcement learning for diffusion models at scale, 2025. https://arxiv.org/abs/2512.04332.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 2024.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Brian D Ziebart. Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Machine Learning Department, Carnegie Mellon University, Dec 2010.

Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In Proc. AAAI, pages 1433–1438, 2008.

## Appendix

Appendix Contents

- A Detailed Related Work. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C Proofs for Section 2: Preliminaries. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- C.1 Score-velocity relationship . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- C.2 SDE and ODE marginal equivalence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Proofs for Section 3: Understanding the Limitations of Flow and Score Matching . . . . . . . . . . . . . . . . . . . . . 24

- D.1 Distribution Shift and Error Accumulation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.2 Geometry mismatch . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- D.3 How RL Helps. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

- E Proofs for Section 4: Method: Discriminator-Guided RL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42 E.1 Why Representation Spaces Help . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- F Adjoint Matching: Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- G Local Linear Integrator for the Memoryless SDE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47

- G.1 The Memoryless SDE and the Local Linear Integrator . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- G.2 Derivation of the Local Linear Integrator . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

- H Additional Implementation Details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
- I Experimental Setup. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55

- I.1 Distributional alignment setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
- I.2 Better image quality (reward-transfer evaluation) setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
- I.3 Preference-based RL setup. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
- I.4 Distillation from RL teachers setup. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
- I.5 Distillation from DRL Teachers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- I.6 Feature-space ablation setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- I.7 Discriminator architecture and training ablation setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58

- J Extended Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60

- J.1 Alignment: Quantitative λ Sweep . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
- J.2 Alignment: Full Distribution Metrics (PRDC) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
- J.3 Image Quality: Reward Improvement vs. λ. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
- J.4 Image Quality / Preference RL: Pareto Plots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
- J.5 Ablations: Discriminator Full R1 Sweep. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70
- J.6 Ablations: Full Feature-Space Sweep . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70
- J.7 Effect of CFG on Reward Scores and Fréchet Distance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71

- K Qualitative Samples. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72

- K.1 Base vs. Fine-tuned . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
- K.2 Effect of DRL Strength . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
- K.3 RL Fine-tuning Sample Images. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81

Appendix

- A Detailed Related Work

Covariate shift, imitation learning, and exposure bias. The distribution mismatch pathology we identify in flow and score matching (Proposition 3.1) is analogous to the classical covariate-shift problem in imitation learning with behavioral cloning. Behavioral cloning controls prediction error under the expert’s state distribution, but the learned policy is deployed under its own induced state distribution. As a result, errors that are large under the model’s rollout distribution can appear small under the expert’s distribution (Ross and Bagnell, 2010; Ross et al., 2011). This phenomenon has been extensively studied in reinforcement learning and imitation learning (Laskey et al., 2017; Rajaraman et al., 2020; Swamy et al., 2021), and has recently received renewed attention in language modeling through on-policy distillation and related methods (Agarwal et al., 2024; Hübotter et al., 2026; Lu and Lab, 2025; Shenfeld et al., 2026).

Independently, the diffusion literature has studied analogous train–test mismatch under the names exposure bias, sampling drift, and error propagation (Ning et al., 2023b; Li and van der Schaar, 2023; Ning et al., 2023a). Because this thread has evolved separately from the RL literature, existing approaches often address the mismatch through ad hoc, model-specific off-distribution corrections, such as perturbing training inputs to mimic sampling-time errors (Ning et al., 2023b), modifying the sampling trajectory via shifted time steps (Li et al., 2023), or modifying the norm of transitions in the sampling process (Ning et al., 2023a). The closest work to an on-policy correction is Li and van der Schaar (2023), who use an MMD regularizer to match forward noising marginals to short model-induced denoising marginals. However, their rollouts are warm-started from the data noising distribution and are very short in their main experiments: only L = 5 reverse steps out of T = 1000. Thus the method addresses local train–test mismatch rather than the full sampling-trajectory occupancy mismatch.

We connect these two lines of work in two ways. First, we show that flow and score matching admit worst-case pathologies analogous to those in DAgger-style imitation learning, with even weaker guarantees due to the continuous-time nature of the dynamics. Second, motivated by this connection, we focus on an explicitly on-policy solution, closer in spirit to reinforcement learning and imitation learning than to off-policy perturbation-based corrections. Given our positive results, we believe an interesting direction for future work is to revisit sampling-based approaches such as Li and van der Schaar (2023) with the stronger tooling used in our paper, such as adjoint-based training and the use of semantic feature spaces in the endpoint distributions for the MMD computations.

Inverse RL and adversarial imitation. A classical alternative to behavioral cloning is inverse reinforcement learning (IRL), which infers a reward under which expert behavior is optimal and then optimizes that reward with RL (Ng and Russell, 2000; Abbeel and Ng, 2004). Maximum-entropy and maximum-causal-entropy IRL are especially close in form to our setting: they model expert trajectories as an exponential tilt of a reference trajectory measure by cumulative reward, pr(τ) ∝ µenv(τ)exp(Rr(τ)), where µenv is induced by the initial-state distribution and environment dynamics (Ziebart et al., 2008; Ziebart, 2010). KL-regularized posttraining has the analogous endpoint form pr(x) ∝ pbase(x)exp(λr(x)), so r∗(x) = λ−1(log q(x) − log pbase(x)) makes q the optimal tilted distribution, up to additive constants.

Closely related IRL algorithms include adversarial imitation methods, which use discriminators in related but different ways than DRL. GAIL trains a discriminator between expert and current learner state-action samples, yielding an adversarial objective for matching occupancy measures rather than a fixed reusable reward (Ho and Ermon, 2016). AIRL keeps the adversarial imitation loop but constrains the discriminator logit to decompose into a reward term and a potential-based shaping term, with the goal of recovering a reward that can be re-optimized under changes in dynamics rather than just a policy that matches the expert in the training MDP (Fu et al., 2018). DRL adapts the discriminator-as-reward idea from GANs and adversarial imitation (Goodfellow et al., 2014) to generative post-training, but with a fixed endpoint reference: the discriminator is trained only once on (q,pbase) through a representation ϕ, and its logit defines a fixed reward used by KL-regularized RL. This avoids the notoriously unstable alternating min-max optimization (Mescheder et al., 2018), and exposes a single hyperparameter λ controlling the strength of the correction. We also differ from prior work in our use of self-supervised representations for ϕ, without which, as we show, the procedure

is not practical.

Adversarial training in diffusion models. Adversarial training has also been combined with diffusion models directly, in two main ways. Early work on CIFAR-10 and LSUN Churches (Jolicoeur-Martineau et al., 2020) augmented the matching loss with an adversarial term on the model’s posterior-mean estimate E[X1 | Xt] sharpening the one-step denoiser used at the final sampling step. More recent approaches (Xiao et al., 2021; Sauer et al., 2024; Xu et al., 2024; Yin et al., 2024) use GANs to essentially amortize the transition kernel p(xs | xt) for s and t, exploiting the fact that GANs are good implicit models in order to enable few-step sampling and distillation.

In addition to the differences described in the previous section, DRL differs from these lines of work in two ways. First, our goal is to correct an existing flow, not to learn an implicit model or a few-step sampler. Second, while Jolicoeur-Martineau et al. (2020) share our motivation, training the discriminator on the posterior mean E[X1 | Xt] is not principled: E[X1 | Xt] is the MSE denoiser, not a sample from the model, so matching its distribution to data does not in general yield sharp, high-quality samples. DRL instead trains the discriminator on actual model samples and corrects the velocity field directly via adjoint matching.

Reward-based post-training for diffusion and flow models. RL post-training of diffusion and flow models requires two ingredients: a reward and an algorithm to optimize it. As discussed throughout the paper, DRL is a contribution to the first; here we discuss its relation to the second. A growing literature studies optimization algorithms for this setting (Black et al., 2024; Fan et al., 2023; Domingo-Enrich et al., 2025; Liu et al., 2025a). While we expect DRL to benefit directly from continued progress in this literature, our analysis in Section 3 predicts that not every optimizer is suited to the role: an effective Stage 2 optimizer needs to be on-policy and to propagate reward information through the model in a way that exploits its landscape—either through ∇xr, as in adjoint matching, or through score-function estimators that estimate ∇θEp

[r] directly.

θ

Methods that recast reward optimization as a matching problem against a reward-tilted target, such as Tilt Matching (Potaptchik et al., 2025) and reward-weighted regression (Peters and Schaal, 2007; Fan et al., 2025; Black et al., 2024), take a different route: they use the reward to estimate a matching objective under the target distribution, rather than leveraging the reward landscape directly in the update. From the perspective of our analysis, this leaves them subject to the same matching geometry that we identify as a potential limitation. Some instantiations are additionally off-policy, which may further reintroduce a train–test mismatch. These considerations need not be decisive in every problem instance, but they suggest a plausible partial explanation for the underperformance of RWR reported in Black et al. (2024) and of Tilt Matching reported in Table 1 of Potaptchik et al. (2025), relative to methods that either propagate ∇xr or rely on policy-gradient-style estimators. Further empirical validation of this hypothesis is an interesting direction for future work.

Calibration. Our analysis of surrogate objectives is closely related to calibration: for a statistic r, we ask whether low flow- or score-matching loss is enough to control the calibration error |Ep

[r]−Eq[r]|. Distributionlevel constraints of this form have been studied in controlled language generation (Khalifa et al., 2021) and in fairness-oriented text-to-image fine-tuning (Shen et al., 2023). The closest connection is to Smith et al. (2025). Their applications differ significantly from ours — e.g. they rebalance animal-class proportions in a conditional image model so that lions, tigers, and other wildlife categories appear evenly, and balance male and female character frequencies in language-model stories about different professions — but one of their methods, CGM-reward, is closely related to DRL. CGM-reward finds the KL-closest model satisfying moment constraints Ep

v

###### [h(x)] = h⋆, with h typically a class indicator (e.g., the gender of the character in a generated story); the max-entropy dual yields the exponential tilt pα⋆(x) ∝ pθ

θ

(x)exp(α⋆⊤h(x)). DRL targets a more general object, the full pushforward pϕ = qϕ under KL regularization (Proposition E.1), but in the linear case the two coincide in form: our class-conditional discriminator ry(x) = wy⊤ϕ(x) + by gives a tilt linear in ϕ. Nevertheless, even in this special case the two generally estimate different targets, since CGM-reward solves an empirical max-entropy dual against user-specified moments while DRL recovers the tilt from a discriminator’s logit. Moreover, as shown in design choices, the linear head is not essential: MLP discriminators yield similar gains.

base

- B Limitations

While our results provide evidence that DRL is an effective post-training method, several caveats remain. These include practical limitations inherited from RL-based fine-tuning, limitations of the representation space used to define the reward, and limitations of the simplified theoretical setting used to motivate the method.

Classifier-free guidance. One limitation of DRL, and of flow-based RL methods more generally, is that they do not provide a clean way to incorporate classifier-free guidance (CFG) during training. CFG typically disrupts the structure of the learned flow used to pass between ODE and SDE samplers. We experimented with CFG during training, but it often led to instability; similar difficulties have been reported by Domingo-Enrich et al. (2025).1 We also observed that the improvements of DRL over the base model are smaller when both are sampled with their best CFG scale, as shown in figure 5. We attribute these issues to the fact that current RL algorithms do not yet handle CFG cleanly, and consider developing principled CFG-compatible RL training an important direction for future work.

Choice of λ. As shown in section 5, the KL regularization weight λ plays a significant role in performance, and a small number of values typically need to be explored. While we found λ=10 to be a reliable default across our experiments, this may not hold in all settings, and the need to tune λ adds to the overall computational cost—a burden shared by RL methods more broadly. An interesting direction for future work is to make the method more robust to the choice of λ, and more generally to amortize the cost of tuning this hyperparameter, for example with a ControlNet-like mechanism (Zhang et al., 2023).

Reverse KL behavior. As discussed in the main text, DRL optimizes a reverse KL objective and therefore inherits both its advantages and its drawbacks. In particular, while DRL is robust at low λ values, at large λ the model can become mode-seeking, as visible in the PRDC curves (section J.2). We believe this partly explains why R1 gradient regularization proved beneficial at higher λ (design choices). Although we did not find this to be a practical issue at the λ values we recommend, it is worth keeping in mind.

Additional compute. As a post-training method, DRL requires additional compute beyond base-model training (see section I for details). While this cost is only a small fraction of pretraining, it is nonetheless an extra expense that may not be justified in all scenarios.

Role of the representation. Our experiments suggest that DRL is most effective when the discriminator is trained in a pretrained representation space. This is a strength, because it lets the reward focus on semantic discrepancies, but it is also a limitation: the method can only correct distributional differences that are visible to the chosen embedder, so it requires a suitable representation in the first place. Moreover, although our analysis emphasizes both train–test mismatch and reward-geometry mismatch, the empirical importance of the embedder suggests that a large part of DRL’s benefit may come from changing the geometry in which the reward is estimated, rather than from on-policy optimization alone.

Scope of the theory. Our theoretical analysis is intentionally carried out in a simplified probability-flow ODE setting. We believe this captures the core geometric obstructions behind propositions 3.1 and 3.2, but it is not a complete model of every practical training setup. In particular, while the reward certificate in proposition 3.2 should have close SDE analogues under suitable drift control, the no-certificate construction in proposition 3.1 uses the ODE structure directly. It is therefore not clear how far that worst-case construction extends to stochastic samplers, especially because path-KL objectives can yield Pinsker-type control at the trajectory level. This is nevertheless consistent with our intent: the theory is meant to identify failure modes and motivate the method, not to claim that the worst case occurs generically. Furthermore, while a construction like the ODE one is possibly too extreme, as discussed in Section A many works have documented significant train–test gaps in practice for SDE samplers as well.

A related gap is the distinction between training and inference dynamics. In our experiments, RL training uses the memoryless SDE, while evaluation uses the ODE sampler. The memoryless-schedule equivalence guarantees agreement at the optimum, but during training the SDE and ODE rollout distributions need not coincide exactly. While this means that SDE-level Pinsker bounds do not by themselves certify the ODE samples used at inference, we did not observe large discrepancies between the two in practice. Moreover, as argued in the paper, we hypothesize that the main driver of DRL’s improvements is correcting geometric

1https://github.com/microsoft/soc-fine-tuning-sd

###### obstructions, which should affect both SDE and ODE samplers similarly through the reward gradient. This does not affect our explanation of the limitations of flow and score matching.

### C Proofs for Section 2: Preliminaries

- This appendix collects proofs for the statements in section 2.

- C.1 Score-velocity relationship

- Proposition C.1 (Score-velocity equivalence). Let X1 ∼ q and X0 ∼ N(0,I) be independent, and define the interpolation Xt = α(t)X1 + β(t)X0 with marginal density qt. The velocity field vt(x) := E[α˙(t)X1 + β˙(t)X0 | Xt = x] and the score st(x) := ∇x log qt(x) are related by:

vt(x) =

α˙(t) α(t)

x + β(t)2

α ˙(t) α(t) −

β˙(t) β(t)

st(x).

Proof. To compute the velocity, we need E[X1 | Xt = x] and E[X0 | Xt = x]. From the interpolation, we can express the noise as X0 = (Xt − α(t)X1)/β(t). Therefore:

E[X0 | Xt = x] =

x − α(t)E[X1 | Xt = x] β(t)

.

By Tweedie’s identity (Efron, 2011), the posterior mean of X1 given Xt = x is:

E[X1 | Xt = x] =

x + β(t)2st(x) α(t)

=

x α(t)

+

β(t)2 α(t)

st(x).

Substituting:

E[X0 | Xt = x] =

x − α(t) α x(t) + β(t)

2

α(t) st(x) β(t)

= −β(t)2st(x) β(t)

= −β(t)st(x).

Now we compute the velocity: vt(x) = α˙(t)E[X1 | Xt = x] + β˙(t)E[X0 | Xt = x]

= α˙(t)

x α(t)

+

β(t)2 α(t)

st(x) + β˙(t)(−β(t)st(x))

=

α˙(t) α(t)

x +

α ˙(t)β(t)2 α(t) − β˙(t)β(t) st(x)

=

α˙(t) α(t)

x + β(t)2

α ˙(t) α(t) −

β˙(t) β(t)

st(x).

This is exactly (C.1).

| |
|---|

C.2 SDE and ODE marginal equivalence

- Proposition C.2 (Marginal equivalence). Let vt and st = ∇log qt be the true velocity and score fields for marginals qt. Then the SDE

dXt = vt(Xt) + 21σ(t)2st(Xt) dt + σ(t)dWt

and the probability flow ODE dXt/dt = vt(Xt) produce identical marginal distributions qt at all times t, for any noise schedule σ(t) ≥ 0.

Proof. The Fokker–Planck equation describes how the density ρt of a diffusion process evolves. For an SDE of the form dXt = b(Xt,t)dt + σ(t)dWt, the density satisfies

σ(t)2 2

∂tρt = −∇ · (bρt) +

∆ρt.

For our SDE with drift b = vt + 12σ(t)2st, this becomes

σ(t)2 2

∂tρt = −∇ · vt + 21σ(t)2st ρt +

∆ρt.

We verify that ρt = qt satisfies this equation. Using st = ∇log qt, we have st qt = ∇qt. Therefore:

σ(t)2 2 ∇ · (∇qt) +

σ(t)2 2

σ(t)2 2

−∇ · 12σ(t)2st qt +

∆qt = −

∆qt

σ(t)2 2

σ(t)2 2

= −

∆qt +

∆qt = 0.

The diffusion and score terms cancel exactly, leaving ∂tqt = −∇ · (vt qt),

which is the continuity equation for the probability flow ODE. Since both the ODE and SDE satisfy the same continuity equation with the same initial condition q0, they have identical marginals qt at all times.

| |
|---|

### D Proofs for Section 3: Understanding the Limitations of Flow and Score Matching

- This appendix collects proofs for the statements in section 3. The material follows the order of the main discussion: we first prove the no-certificate result for the standard probability-flow ODE, then collect the reward-geometry results used in proposition 3.2 into a single section, and finally prove the RL reward-regret bound (proposition D.7).

D.1 Distribution Shift and Error Accumulation

D.1.1 Proof of proposition 3.1: no reward certificate for the standard probability-flow sampler

We now prove the no-certificate claim by giving two complementary counterexamples for the probability-flow ODE. The first is a one-dimensional Gaussian velocity example with a closed-form calculation. It is useful because every quantity can be written down explicitly, so the qt versus pt mismatch is visible in a single formula and provides a simple illustration of what goes wrong. The second proof gives a general construction in the coordinate induced by the true target flow and applies equally to the velocity and score parametrizations.

Throughout we use the setup in Section 2 and the following standard assumptions and notation. We let (α,β) be a C1 interpolation schedule (1) with

α(0) = 0, β(0) = 1, α(1) = 1, β(1) = 0,

and α(t),β(t) > 0 for t ∈ (0,1). For a target endpoint law q, write qt for the interpolation marginal, sq for its score, and vq for its probability-flow velocity. For score-parametrized fields we also write

γ(t) := β(t)2

β˙(t) β(t)

α ˙(t) α(t) −

, (9)

so that

α˙(t) α(t)

vq(x,t) =

x + γ(t)sq(x,t)

by (C.1). Proposition D.1 (Velocity case: Gaussian variance inflation). For every ε > 0 and every δ ∈ (0,1/4), there exist a target distribution q, a bounded reward r : R → [0,1], and a velocity field vbad such that

t |vbad(X,t) − vq(X,t)|2 ≤ ε,

EX∼q

sup

t∈[0,1]

but if pbad denotes the endpoint law of the standard probability-flow ODE sampler

X˙t = vbad(Xt,t), X0 ∼ N(0,1), then

Eq[r] − Ep

[r] ≥ 1 − 2δ. Intuition. The marginals qt have analytical form given by:

bad

qt = N(0,ρτ(t)2), ρτ(t)2 := α(t)2τ2 + β(t)2. The idea is to define a perturbation given by,

∆v(x,t) = √ε

x ρτ(t)

.

Under qt, the normalized coordinate X/ρτ(t) is standard Gaussian, so the flow-matching error is exactly ε at every time. However, as time progresses this small error accumulates and becomes larger as |x| grows. This ends up increasing the variance of the rollout. Therefore, by choosing the target endpoint distribution to be sufficiently concentrated, we can make a reward on the high-likelihood regions of q1 to be missed by most of the distribution.

1.00

reward at t = 1

0.75

variance inﬂation

0.50

0.25

0.00

x

−0.25

−0.50

−0.75

−1.00

0 1

t

|qt band<br><br>true paths<br><br>pbad,t envelope<br><br>rollout paths<br><br>| |
|---|
<br><br>reward at t = 1|
|---|

- 0
- 1
- 2
- 3
- 4

shell branches

thin overlap with qt

###### St

###### x

−1

low-reward interval at t = 1

left shell branch sweeps the rollout inward

−2

0 1

t

shell branches St

qt band

low-reward interval

true paths

rollout paths

Figure 12 Appendix sketches of the two constructions. Left: velocity case, corresponding to Proposition D.1 and its proof below. The true trajectories contract with qt, while the learned rollout trajectories and the one-sigma envelope of pbad,t show variance inflation under the same linear perturbation. Right: a one-dimensional Gaussian instance of Proposition D.3, drawn directly in x-space. In the general proof the perturbation is a shrinking radial shell in the target-flow coordinate; in one dimension that shell becomes two moving intervals, shown in orange here, and the left branch sweeps representative rollout trajectories into the low-reward interval at t = 1. In both panels, the shaded regions indicate typical mass under the relevant marginals, the dashed dark paths denote representative true trajectories, and the solid orange paths denote representative learned rollout trajectories. The obstruction is the same: training measures error under the data marginals qt, while sampling evaluates the learned field along the rollout marginals pt.

Remark D.2. This is not the same statement in the main text but it immediately disproves the existence of a guarantee of the form “with small flow-matching error, we obtain a small reward gap”.

Proof. Fix ε > 0 and δ ∈ (0,1/4). Let τ ∈ (0,1) be chosen later and set

q := N(0,τ2). Under the interpolation (1),

Xt = α(t)X1 + β(t)X0, X1 ∼ q, X0 ∼ N(0,1), the marginal is

qt = N(0,ρτ(t)2), ρτ(t)2 := α(t)2τ2 + β(t)2. The corresponding probability-flow velocity is

ρ˙τ(t) ρτ(t)

x.

vq(x,t) =

Indeed,

X˙t = α˙(t)X1 + β˙(t)X0. Since (Xt,X˙t) are jointly centered Gaussian, the conditional expectation is linear:

Cov(X˙t,Xt) Var(Xt)

vq(x,t) = E[X˙t | Xt = x] =

x.

Using the independence of X1 and X0,

- 1

- 2

Cov(X˙t,Xt) = α(t)α˙(t)τ2 + β(t)β˙(t) =

while Var(Xt) = ρτ(t)2. Therefore

ρ˙τ(t) ρτ(t)

vq(x,t) =

x.

d dt

ρτ(t)2 = ρτ(t) ˙ρτ(t),

Now define

vbad(x,t) := vq(x,t) + ∆v(x,t), ∆v(x,t) := √ε

x ρτ(t)

.

If X ∼ qt, then E[X2] = ρτ(t)2, so

t |vbad(X,t) − vq(X,t)|2 = εEq

Eq

t

X2 ρτ(t)2

= ε.

Thus the training error is exactly ε uniformly in time. The learned ODE is

√ε ρτ(t)

ρ ˙τ(t) ρτ(t)

X˙t =

+

Xt. Solving this scalar linear ODE gives

Hence the rollout marginal is

Xt = ρτ(t)exp √ε

t

0

du ρτ(u)

X0.

pbad,t = N(0,ρτ(t)2k(t)), k(t) := exp 2√ε

t

0

du ρτ(u)

.

In particular,

X2 ρτ(t)2

bad,t |∆v(X,t)|2 = εEp

Ep

= εk(t),

bad,t

so the same perturbation is measured as ε under qt but as εk(t) under the rollout law pbad,t. We now show that k(1) can be made arbitrarily large by taking τ small. Because β is C1 with β(1) = 0, there exist M > 0 and t0 < 1 such that

β(t) ≤ M(1 − t) for all t ∈ [t0,1]. Let A := supt∈[0,1] |α(t)| < ∞. Then

ρτ(t) = α(t)2τ2 + β(t)2 ≤ A2τ2 + M2(1 − t)2 for t ∈ [t0,1]. Therefore

1

1

M(1 − t0) Aτ

dt ρτ(t) ≥

dt A2τ2 + M2(1 − t)2

1 M

sinh−1

,

=

0

t0

which diverges as τ ↓ 0. Hence k(1) → ∞ as τ ↓ 0. Choose K > 0 such that, for Z ∼ N(0,1),

Pr(|Z| ≤ K) ≥ 1 − δ. Define

r(x) := 1{|x| ≤ Kτ}.

Under the target endpoint law q = N(0,τ2),

Eq[r] = Pr(|Z| ≤ K) ≥ 1 − δ. Under the bad endpoint law pbad = N(0,τ2k(1)),

K k(1) ≤

2 π

K k(1)

Ep

[r] = Pr |Z| ≤

.

bad

Here we used that, for Z ∼ N(0,1) and any a ≥ 0,

a

1 √2π

2 π

2/2 dz ≤

e−z

Pr(|Z| ≤ a) =

a.

−a

Since k(1) → ∞, choose τ small enough that 2/π K/ k(1) ≤ δ. Then Eq[r] − Ep

[r] ≥ 1 − 2δ.

bad

| |
|---|

For the general proof, we use the standard notion of a flow map; see, for example, (Arnold, 1978; Boffi et al., 2025). Intuitively, the flow map is the function that, given an initial time a, a final time t, and a point y, returns the point reached by evolving y from time a to time t. Formally, given 0 < a < 1 and a point y ∈ Rd, let Xa,y denote the solution of

X˙t = vq(Xt,t), Xa = y. The associated flow map is defined by

Φa,t(y) := Xta,y, that is, Φa,t(y) is the point reached at time t by the exact probability-flow ODE initialized at y at time a.

Proposition D.3 (Radial-shell no-certificate construction). Let q be the endpoint target law, let (qt)t∈[0,1] be the interpolation marginals, let vq and sq denote the corresponding population velocity and score, and let r : Rd → [0,1] be a bounded reward. Fix 0 < a < b < 1. Assume that the exact probability-flow ODE

X˙t = vq(Xt,t)

generates flow maps Φa,t : Rd → Rd that are C1 diffeomorphisms for every t ∈ [a,1]. Assume further that there exist x⋆ ∈ Rd, ρ⋆ > 0, and η ∈ [0,1] such that

r(x) ≤ η whenever ∥x − x⋆∥ ≤ ρ⋆. Then for every ε > 0 and every δ ∈ (0,1), the following two statements hold. Velocity parametrization. There exists a field vbad such that

vbad(x,t) = vq(x,t) for all t ∈/ [a,b], and

t ∥vbad(X,t) − vq(X,t)∥2 ≤ ε. Let pbad denote the endpoint law of the probability-flow ODE sampler (2) run with vbad in place of vq. Then Ep

EX∼q

sup

t∈[0,1]

[r] ≤ η + δ.

bad

Score parametrization. If inft∈[a,b] |γ(t)| > 0, then there exists a field sbad such that

sbad(x,t) = sq(x,t) for all t ∈/ [a,b], and

t ∥sbad(X,t) − sq(X,t)∥2 ≤ ε.

EX∼q

sup

t∈[0,1]

Define the induced velocity field

α˙(t) α(t)

vbad(x,t) =

x + γ(t)sbad(x,t), and let pbad denote the endpoint law of the probability-flow ODE sampler (2) driven by this vbad. Then Ep

[r] ≤ η + δ.

bad

Intuition. The goal is to hide a harmful perturbation inside a small score-matching or flow-matching error. We do this with a thin shell that moves inward over time. Away from the shell, the learned field agrees with the true score/flow, so if the shell is thin enough then the training error stays small. At the same time, the shell is arranged to catch trajectories and carry them into a low-reward region by time 1. Figure 12 (right) gives a sketch in one dimension, where the shell becomes two moving intervals. We apply the construction only on a subinterval [a,b] ⊂ (0,1) to avoid the endpoints.

However, designing the shell directly in x-space is awkward because the true dynamics already move trajectories around on their own, so the shell’s effect gets tangled with the motion of the original dynamics. We avoid this by working in a coordinate in which the true dynamics are trivial: for a path Xt, define the inverse-flow coordinate

Yt := Φ−a,t1(Xt),

which records the starting point at time a of the trajectory passing through Xt. By construction, Yt is constant along the true ODE, so any motion of a trajectory in Y -space comes entirely from the perturbation. We then translate the motion in this space back to x-space and check that the training error is small while still carrying trajectories into the low-reward region.

Proof. The proof is relatively straightforward but requires some bookkeeping. We organize it in steps for clarity. Under the assumptions of the proposition, the laws qt for t ∈ [a,1] admit densities. With a slight abuse of notation, we use the same symbol for the law and its density when no confusion arises.

- Step 1: Work in the target-flow coordinate and rewrite the reward there. For any path Xt on [a,1], define Yt := Φ−a,t1(Xt).

This coordinate records where Xt came from at time a under the true target flow. If Xt follows the true dynamics then Xt = Φa,t(Ya), so Yt = Ya is constant.

Because the shell construction will be carried out in Y -space, it is convenient to rewrite the endpoint reward in the same coordinate. Set

µ := qa, B(z,ρ) := {y ∈ Rd : ∥y − z∥ ≤ ρ}, r(y) := r(Φa,1(y)). Let

z := Φ−a,11(x⋆),

and consider the open ball {x ∈ Rd : ∥x − x⋆∥ < ρ⋆} in endpoint space. Because Φa,1 is a diffeomorphism, its preimage is an open neighborhood of z, so there exists ρ > 0 such that

B(z,ρ) ⊂ Φ−a,11 x ∈ Rd : ∥x − x⋆∥ < ρ⋆ . Therefore

r(y) ≤ η for all y ∈ B(z,ρ).

Thus it is enough to construct a bad rollout whose terminal target-flow coordinate Y1 lies in the ball B(z,ρ). In the remaining steps we will first define the bad dynamics directly in this coordinate, where the true dynamics are frozen, and then recover the corresponding path in x-space by applying the flow map.

- Step 2: Build the shrinking shell in the target-flow coordinate. We first choose a large ball that contains almost all of the mass at time a. Because

µ(B(z,R)) → 1 as R → ∞, we can choose R > ρ such that

µ(B(z,R)) ≥ 1 − δ.

Our goal is to build a thin shell that starts just outside B(z,R) and then moves inward until it lies inside the low-reward ball B(z,ρ).

Let

R + 1 − ρ/2 T

ℓ(t) := R + 1 − c(t − a) for t ∈ [a,b].

T := b − a, c :=

Here T is the length of the active interval, c is the inward speed, and ℓ(t) is the radius of the shell at time t. Then

ℓ(a) = R + 1, ℓ(b) = ρ/2. So the shell starts strictly outside B(z,R) and ends strictly inside B(z,ρ). Fix w > 0, to be chosen later, with

ρ 4

- 1

- 2

w < min

,

.

This will be the thickness scale of the shell. The bound w ≤ 1/2 is only used later, when we keep the shell inside a fixed radial region for the training-error estimate.

Define a piecewise-linear function ψw : R → [0,c] that specifies how strongly the perturbation acts as a function of signed distance from the shell:



0, s ≤ −w,

- c 1 +

s w

, −w ≤ s ≤ 0, c, 0 ≤ s ≤ w,

- c 2 −



ψw(s) :=

s w

, w ≤ s ≤ 2w, 0, s ≥ 2w.



This function is zero outside [−w,2w], ramps up linearly on [−w,0], is flat at height c on [0,w], and ramps down linearly on [w,2w].

For y ̸= z, define the outward radial direction

y − z ∥y − z∥

n(y) :=

,

and set n(z) := 0. Now define the perturbation in the target-flow coordinate by

uw(y,t) := −1{t ∈ [a,b]}ψw(∥y − z∥ − ℓ(t))n(y). The minus sign means that the perturbation pushes points inward. It is supported on the moving shell St := {y ∈ Rd : ∥y − z∥ − ℓ(t) ∈ [−w,2w]}.

So at time t, the perturbation is active only on radii in the interval [ℓ(t)−w,ℓ(t)+2w]. Because ℓ(t) ∈ [ρ/2,R+1] and w < min{1/2,ρ/4}, every point y ∈ St satisfies

∥y − z∥ ∈ [ℓ(t) − w,ℓ(t) + 2w] ⊂ [ρ/4,R + 2].

Thus the shell St always lies inside the fixed radial region {ρ/4 ≤ ∥y − z∥ ≤ R + 2}.

- Step 3: Define and analyze the bad dynamics in Y -space. Under the true dynamics, the target-flow coordinate does not move: Yt is constant. We now define the bad dynamics in this same coordinate by letting Yt solve

Y˙t = uw(Yt,t) on [a,1]

with initial law Ya ∼ µ. Because the shell support stays in the radial region {ρ/4 ≤ ∥y − z∥ ≤ R + 2}, it stays a positive distance from z. Therefore the field uw(·,t) is globally Lipschitz in y for each t and piecewise continuous in t. Hence this ODE has a unique absolutely continuous solution on [a,1].

The mechanism has two phases: first the shell catches a trajectory while the trajectory stays fixed in Y -space, and then the shell carries the trajectory inward until it reaches B(z,ρ).

Let

ϱt := ∥Yt − z∥, St := ϱt − ℓ(t).

Here ϱt is the distance to the shell center and St records the trajectory’s position relative to the moving shell. The sign of St tells us where the trajectory sits: St < −w means it is strictly inside the shell, while St ∈ [−w,0] means it has been caught by the shell’s inner side.

Consider t ∈ [a,b]. Because uw(y,t) always points in the radial direction n(y), it changes only the distance to z. For ϱt > 0,

⊤

Yt − z ∥Yt − z∥

Y˙t = n(Yt)⊤uw(Yt,t) = −ψw(St). Since ℓ˙(t) = −c, we also have

ϱ˙t =

S˙t = ϱ˙t − ℓ˙(t) = −ψw(St) + c. If Ya ∈ B(z,ρ), then whenever ϱt > 0,

ϱ˙t = −ψw(St) ≤ 0, so the trajectory stays in B(z,ρ). It therefore remains to consider the case

ρ ≤ ϱa ≤ R. In this case

Sa = ϱa − ℓ(a) = ϱa − (R + 1) ∈ [ρ − R − 1,−1]. Since w < 1 the trajectory starts strictly inside the shell, with Sa < −w. While St ≤ −w, the function ψw vanishes and

S˙t = ϱ˙t − ℓ˙(t) = 0 − (−c) = c.

So before capture, the shell simply moves inward until it catches the trajectory. The latest catch occurs for the smallest radius that still needs to be captured, namely ϱa = ρ. Then the catch time is at most

−Sa − w c

R + 1 − ρ − w c

=

R + 1 − ρ/2 c

= b − a,

<

where we used w < ρ/2. Hence every trajectory with Ya ∈ B(z,R) \ B(z,ρ) reaches the inner side of the shell before time b.

Once St ∈ [−w,0], the definition of ψw gives

St w

S˙t = −c 1 +

c w

+ c = −

St.

Therefore St remains in [−w,0] and moves toward 0. In particular, Sb ≤ 0.

Since ℓ(b) = ρ/2, we conclude that

ϱb = ℓ(b) + Sb ≤ ℓ(b) = ρ/2 < ρ. So every trajectory that starts in B(z,R) lies in B(z,ρ) by time b. For t > b, the perturbation is switched off, so uw(Yt,t) = 0 and the target-flow coordinate remains fixed. Hence

Y1 ∈ B(z,ρ) on the event {Ya ∈ B(z,R)}.

At this point the geometry is complete: every trajectory that starts in B(z,R) has been moved into B(z,ρ) by time 1. By Step 1,

r(Y1) ≤ η on {Ya ∈ B(z,R)}. Since Ya ∼ µ and µ(B(z,R)) ≥ 1 − δ, we obtain

E[ r(Y1)] ≤ η · µ(B(z,R)) + 1 · µ(B(z,R)c) ≤ η + δ.

This is the bad behavior we want in the target-flow coordinate. It remains to realize these same dynamics as a perturbation of the probability-flow ODE in x-space.

- Step 4: Push the construction to x-space. Let Xt denote the exact probability-flow ODE started from X˙t = vq(Xt,t), X0 ∼ N(0,I).

Then Xa ∼ qa = µ. Set

Ya := Xa, let Yt evolve by

Y˙t = uw(Yt,t) on [a,1], and define the bad rollout by

Xt, t ∈ [0,a], Φa,t(Yt), t ∈ [a,1].

X¯t :=

This is well defined because

Φa,a(Ya) = Ya = Xa = X¯a. Moreover, because X¯1 = Φa,1(Y1), the definition of r and the Step 3 bound give E[r(X¯1)] = E[r(Φa,1(Y1))] = E[ r(Y1)] ≤ η + δ.

This proves the endpoint reward bound for the constructed path. It remains to identify this path with the standard probability-flow ODE driven by a suitable field.

For t ∈ [a,1], differentiating the second branch gives

X¯˙t = ∂tΦa,t(Yt) + DΦa,t(Yt)Y˙t = vq(Φa,t(Yt),t) + DΦa,t(Yt)uw(Yt,t)

Here DΦa,t(y) denotes the Jacobian matrix of Φa,t with respect to the spatial variable y. Because Φa,t is a diffeomorphism, the second term can be written as a function of (x,t) alone. Define

and set

∆v(x,t) :=

DΦa,t(Φ−a,t1(x))uw(Φ−a,t1(x),t), t ∈ [a,b], 0, t ∈/ [a,b].

vbad(x,t) := vq(x,t) + ∆v(x,t).

By definition, ∆v(x,t) = 0 for t < a, so vbad = vq on [0,a). Since X¯t = Xt there, X¯ agrees with the exact ODE on [0,a). On the other hand, for t ∈ [a,1] the computation above gives

X¯˙t = vbad(X¯t,t). Hence X¯ is a solution of

X˙t = vbad(Xt,t) started from X¯0 ∼ N(0,I). Therefore the endpoint law of X¯1 is exactly pbad, and so Ep

[r] ≤ η + δ.

bad

This proves the endpoint reward bound for the velocity construction. It remains to show that the training error under qt is small.

- Step 5: The velocity error is small under qt. We now show that we can choose w small enough to ensure

t ∥vbad(X,t) − vq(X,t)∥2 = sup

EX∼q

sup

t∈[0,1]

t∈[0,1]

t ∥∆v(X,t)∥2 ≤ ε.

EX∼q

If t ∈/ [a,b], then ∆v(·,t) = 0, so there is nothing to prove. It therefore suffices to consider t ∈ [a,b]. In that case,

∆v(x,t) = DΦa,t(Φ−a,t1(x))uw(Φ−a,t1(x),t). Now fix t ∈ [a,b], let X ∼ qt, and set

Y := Φ−a,t1(X), Because qt = (Φa,t)#µ, we have Y ∼ µ = qa. Therefore

t ∥∆v(X,t)∥2 = ∥DΦa,t(y)uw(y,t)∥2 qa(y)dy.

EX∼q

The key observation is that uw(·,t) is supported on a thin radial shell. Indeed, by definition,

uw(y,t) ̸= 0 =⇒ ∥y − z∥ ∈ [ℓ(t) − w,ℓ(t) + 2w]. This interval has length 3w. Thus the whole expectation is concentrated on a thin shell in Y -space, and it remains to control the weighted qa-mass of such shells. Since the shell support always stays in the radial range [ρ/4,R + 2], it is enough to control thin radial shells inside the fixed radial region

AR := y ∈ Rd : ρ/4 ≤ ∥y − z∥ ≤ R + 2 .

Because a < 1, we have β(a) > 0, so qa is a Gaussian-smoothed version of q. In particular, qa has bounded density:

∥x − α(a)u∥2 2β(a)2

1 (2πβ(a)2)d/2

1 (2πβ(a)2)d/2

q(u)du ≤

exp −

qa(x) =

.

Because Φa,t is a C1 flow, its Jacobian is uniformly bounded on the compact set [a,b] × AR. Also, on the bounded radial region AR, the volume of a radial shell is proportional to its thickness. Combining these two facts with the density bound above, there exists a finite constant CR such that for every interval I ⊂ [ρ/4,R+2] and every t ∈ [a,b],

∥DΦa,t(y)∥2op qa(y)dy ≤ CR|I|. Indeed, if SI := {y : ∥y − z∥ ∈ I}, then bounded density and bounded Jacobian on [a,b] × AR give

{y: ∥y−z∥∈I}

∥DΦa,t(y)∥2op qa(y)dy ≤ CR′ Vol(SI),

SI

while the volume of a radial shell inside the bounded annulus AR satisfies Vol(SI) ≤ CR′′|I|. Absorbing the constants gives the claim.

It := [ℓ(t) − w,ℓ(t) + 2w]. This interval has length 3w and is contained in [ρ/4,R + 2]. Since uw(·,t) is supported where

∥y − z∥ ∈ It, and |uw(y,t)| ≤ c, we obtain

t ∥∆v(X,t)∥2 ≤ ∥DΦa,t(y)∥2op |uw(y,t)|2 qa(y)dy ≤ c2

EX∼q

∥DΦa,t(y)∥2op qa(y)dy ≤ c2CR|It| = 3c2CRw.

{y: ∥y−z∥∈It}

Hence choosing

ε 3c2CR ensures

ρ 4

1 2

w ≤ min

,

,

t ∥vbad(X,t) − vq(X,t)∥2 ≤ ε.

EX∼q

sup

t∈[0,1]

- Step 6: Score parametrization. For the score statement, keep the same reward, the same center z, and the same shell radii. If needed, choose a smaller width w; this changes the perturbation but not the reward bound. Set

|γ(t)| > 0.

γ− := inf

t∈[a,b]

Define

 

∆v(x,t) γ(t)

, t ∈ [a,b],

∆s(x,t) :=



0, t ∈/ [a,b]. Set

sbad(x,t) := sq(x,t) + ∆s(x,t), and define

α˙(t) α(t)

x + γ(t)sbad(x,t). Because

vbad(x,t) =

α˙(t) α(t)

x + γ(t)sq(x,t), we obtain

vq(x,t) =

vbad(x,t) = vq(x,t) + ∆v(x,t).

So the score-parametrized model induces exactly the same bad ODE rollout as above, and therefore the same endpoint reward bound.

It remains to check the DSM error. If t ∈/ [a,b], the error is zero. If t ∈ [a,b], then

1 γ−2

t ∥sbad(X,t) − sq(X,t)∥2 = EX∼q

t ∥∆s(X,t)∥2 ≤

t ∥∆v(X,t)∥2 .

EX∼q

EX∼q

Using the bound from Step 5 gives

3c2CR γ−2

t ∥sbad(X,t) − sq(X,t)∥2 ≤

EX∼q

w.

Thus choosing

εγ−2 3c2CR ensures

1 2

ρ 4

w ≤ min

,

,

t ∥sbad(X,t) − sq(X,t)∥2 ≤ ε. This proves the score statement.

EX∼q

sup

t∈[0,1]

| |
|---|

- D.2 Geometry mismatch

- D.2.1 Reward geometry under stronger rollout control

This subsection collects the geometry results used in the main text into one place. We first prove the global certificate stated in proposition 3.2, then derive from the same coupling argument a weaker on-policy variant under the rollout marginals pt. The proposition also records a matching tightness example, and we close with the indicator-reward counterexample that explains why this kind of geometric control applies only to Lipschitz rewards.

We use only standard ODE stability ideas and the following elementary differential form of Grönwall’s inequality; see, for example, Arnold (1978) for background.

Lemma D.4 (Grönwall inequality). Let m : [0,T] → [0,∞) be absolutely continuous, let b : [0,T] → [0,∞) be integrable, and let L ≥ 0. If

m′(t) ≤ b(t) + Lm(t) for almost every t ∈ [0,T], then

t

eL(t−s)b(s)ds for every t ∈ [0,T].

m(t) ≤ eLtm(0) +

0

Proof. The idea is to remove the linear growth term first. Set

z(t) := m(t)e−Lt. Then z is absolutely continuous, and for almost every t,

z′(t) = e−Lt m′(t) − Lm(t) ≤ e−Ltb(t). Integrating from 0 to t gives

t

e−Lsb(s)ds.

z(t) − z(0) ≤

0

Multiplying by eLt yields

t

eL(t−s)b(s)ds.

m(t) ≤ eLtm(0) +

0

| |
|---|

Proposition D.5 (Reward certificates under stronger rollout control). Consider the coupled ODEs

X˙t = v(Xt,t), X˙t∗ = v∗(Xt∗,t), X0 = X0∗ ∼ p0,

and write pt and p∗t for the laws of Xt and Xt∗ respectively. Assume that v∗(·,t) is Lv-Lipschitz in x, uniformly over t ∈ [0,1].

- 1. Global control. If

∥v − v∗∥∞ := sup

∥v(x,t) − v∗(x,t)∥ ≤ ε,

t,x

then

∥X1 − X1∗∥ ≤

1

v(1−s)εds =

eL

0

and for every Lr-Lipschitz reward r,

 

ε Lv

(eL

− 1), Lv > 0, ε, Lv = 0,

v



|Ep

###### [r] − Ep∗

[r]| ≤ Lr

1

1

1

v(1−s)εds.

eL

0

a.s.

- 2. On-policy control. Define the rollout error function

e(t) := EX

t∼pt ∥v(Xt,t) − v∗(Xt,t)∥ . If e ∈ L1([0,1]), then

E∥X1 − X1∗∥ ≤

1

0

eL

v(1−s)e(s)ds. and for every Lr-Lipschitz reward r,

|Ep

1

[r] − Ep∗

1

[r]| ≤ Lr

1

0

eL

v(1−s)e(s)ds.

In particular, if supt∈[0,1] e(t) ≤ ε, then the same endpoint bound as in part 1 holds at t = 1.

- 3. Tightness. The exponential weighting factor eL

v(t−s) and its dependence on Lv, Lr, and the error function cannot be improved in general. More precisely, there exists an explicit one-dimensional population CFM problem whose population minimizer is v∗(x,t) = Lvx and such that, for every nonnegative e ∈ L1([0,1]), the perturbation

∆v(x,t) := e(t), v(x,t) = v∗(x,t) + ∆v(x,t) satisfies

t∼pt ∥v(Xt,t) − v∗(Xt,t)∥ = e(t) and

EX

1

E∥X1 − X1∗∥ =

v(1−s)e(s)ds.

eL

0

Thus part 2 is attained with equality for every admissible error function. For the linear reward r(x) = Lrx, the reward bound in part 2 is also attained with equality. In particular, taking e(t) ≡ ε gives ∥v−v∗∥∞ = ε and attains the bound in part 1.

Proof. The two certificates come from the same coupling argument. The only difference is how the field mismatch term is controlled: in part 1 it is bounded pathwise, while in part 2 it is bounded only after taking expectation.

Set

∆t := Xt − Xt∗. Because both trajectories solve ODEs with the same initial condition, ∆ is absolutely continuous and

∆˙ t = v(Xt,t) − v∗(Xt∗,t) for almost every t. We now use this in two slightly different ways.

- Part 1: global control. Set y(t) := ∥∆t∥.

###### Since ∆ is absolutely continuous, so is y, and for almost every t, y′(t) ≤ ∥∆˙ t∥ ≤ ∥v(Xt,t) − v∗(Xt,t)∥ + ∥v∗(Xt,t) − v∗(Xt∗,t)∥ ≤ ε + Lvy(t).

The first term is controlled by the uniform bound ∥v − v∗∥∞ ≤ ε, evaluated at the point Xt. The second term is controlled by the Lv-Lipschitz continuity of v∗(·,t) in its spatial argument, applied to the pair (Xt,Xt∗). In other words, we split the error into direct field mismatch of size at most ε, and the amplification of the current gap by the dynamics of v∗. Because y(0) = 0, applying lemma D.4 with b(t) ≡ ε gives

 

ε Lv

(eL

vt − 1), Lv > 0,

t

v(t−s)εds =

eL

y(t) ≤



εt, Lv = 0. Evaluating this at t = 1 gives

0

1

v(1−s)εds a.s. Taking expectation preserves this bound. Now let r be Lr-Lipschitz. Since

∥X1 − X1∗∥ ≤

eL

0

|r(X1) − r(X1∗)| ≤ Lr∥X1 − X1∗∥ a.s., we obtain

1

[r]| ≤ Lr E∥X1 − X1∗∥ ≤ Lr

v(1−s)εds.

eL

|Ep

###### [r] − Ep∗

1

1

0

- Part 2: on-policy control. Part 1 relied on a pathwise bound for the field mismatch term at each time t. Under

on-policy control we no longer have that: the quantity ∥v(Xt,t) − v∗(Xt,t)∥ is controlled only after averaging over the rollout law. We therefore rewrite the trajectory gap in integral form and then take expectation, which puts the assumption on the error function

t∼pt ∥v(Xt,t) − v∗(Xt,t)∥ directly into the estimate. Starting from the integral form,

e(t) = EX

∆t =

t

0

v(Xs,s) − v∗(Xs∗,s) ds

t

t

v(Xs,s) − v∗(Xs,s) ds +

v∗(Xs,s) − v∗(Xs∗,s) ds. Taking norms and using the triangle inequality gives

=

0

0

∥∆t∥ ≤

t

∥v(Xs,s) − v∗(Xs,s)∥ds +

0

t

∥v∗(Xs,s) − v∗(Xs∗,s)∥ds.

0

Now use that v∗(·,s) is Lv-Lipschitz in x, so

∥v∗(Xs,s) − v∗(Xs∗,s)∥ ≤ Lv∥Xs − Xs∗∥ = Lv∥∆s∥. Substituting this back yields

t

t

∥v(Xs,s) − v∗(Xs,s)∥ds + Lv

∥∆s∥ds. (∗) Now define

∥∆t∥ ≤

0

0

m(t) := E∥Xt − Xt∗∥ = E∥∆t∥. Taking expectation in (∗) and using Fubini’s theorem yields

t

t

t

t

E∥v(Xs,s) − v∗(Xs,s)∥ds + Lv

m(t) ≤

m(s)ds =

e(s)ds + Lv

m(s)ds.

0

0

0

0

To bring this into differential form, set

M(t) :=

t

e(s)ds + Lv

0

t

m(s)ds.

0

Then m(t) ≤ M(t), M(0) = 0, and for almost every t,

M′(t) = e(t) + Lvm(t) ≤ e(t) + LvM(t). Applying lemma D.4 to M with b = e gives

M(t) ≤

t

v(t−s)e(s)ds.

eL

0

Evaluating at t = 1 and using m ≤ M, we conclude that

1

v(1−s)e(s)ds.

eL

m(1) ≤

0

Therefore

1

v(1−s)e(s)ds. If r is Lr-Lipschitz, then

E∥X1 − X1∗∥ ≤

eL

0

|r(X1) − r(X1∗)| ≤ Lr∥X1 − X1∗∥ a.s., so

1

[r]| ≤ Lr E∥X1 − X1∗∥ ≤ Lr

v(1−s)e(s)ds.

eL

|Ep

###### [r] − Ep∗

1

1

0

If supt e(t) ≤ ε, then

1

1

v(1−s)e(s)ds ≤

v(1−s)εds, which recovers the same endpoint constant as in part 1.

eL

eL

0

0

- Part 3: tightness. Work in one dimension. Fix any C1 function

π 2

θ : [0,1] → 0,

satisfying

π 2

π 2

for t ∈ (0,1). Let

, θ(t) ∈ 0,

θ(0) = 0, θ(1) =

U0 ∼ N(0,1), U1 ∼ N(0,e2L

), with U0 and U1 independent, and define

v

X¯t = α(t)U1 + β(t)U0, Yt = α˙(t)U1 + β˙(t)U0, where

v(t−1) sinθ(t), β(t) := eL

α(t) := eL

vt cosθ(t).

Then α(0) = 0, β(0) = 1, α(1) = 1, and β(1) = 0, with α(t),β(t) > 0 for t ∈ (0,1), so this is a valid interpolation from N(0,1) to N(0,e2L

). The population CFM minimizer is

v

v∗(x,t) = E[Yt | X¯t = x]. Since (X¯t,Yt) is centered and jointly Gaussian,

Cov(Yt,X¯t) Var(X¯t)

v∗(x,t) =

x.

A direct calculation from the definitions of α and β gives Var(X¯t) = α(t)2e2L

vt, Cov(Yt,X¯t) = Lve2L

+ β(t)2 = e2L

vt.

v

Hence v∗(x,t) = Lvx. Now fix any nonnegative function e ∈ L1([0,1]) and define

∆v(x,t) := e(t), v(x,t) := v∗(x,t) + ∆v(x,t) = Lvx + e(t). Let W0 ∼ N(0,1) and run the coupled ODEs

X˙t∗ = v∗(Xt∗,t), X˙t = v(Xt,t), X0∗ = X0 = W0. A direct integration gives

Xt∗ = eL

vtW0, Xt = eL

vtW0 +

t

v(t−s)e(s)ds.

eL

0

Therefore

1

v(1−s)e(s)ds. Since ∆v is independent of x, the on-policy error function coincides with the prescribed function: EX

X1 − X1∗ =

eL

0

t∼pt |v(Xt,t) − v∗(Xt,t)| = e(t). Therefore

1

v(1−s)e(s)ds, so part 2 is attained with equality. For the linear reward

E|X1 − X1∗| =

eL

0

r(x) = Lrx,

1

[r]| = Lr |E[X1 − X1∗]| = Lr

v(1−s)e(s)ds.

eL

|Ep

###### [r] − Ep∗

1

1

0

Thus the reward bound in part 2 is also attained with equality. If e(t) ≡ ε, then ∥v − v∗∥∞ = ε and

 

ε Lv

(eL

− 1), Lv > 0,

1

v

E|X1 − X1∗| =

v(1−s)εds =

eL



ε, Lv = 0. so part 1 is also attained with equality.

0

The exact reward-tightness statement above uses the linear test function r(x) = Lrx, which is Lipschitz but unbounded. If one insists on bounded rewards while keeping this same amplification-tight construction, the endpoint Gaussians can be rescaled to make range truncation negligible. Indeed, replace the endpoint Gaussians by U0 ∼ N(0,σ2) and U1 ∼ N(0,σ2e2L

). The calculation of the CFM minimizer is unchanged, so v∗(x,t) = Lvx, and the perturbation ∆v(x,t) = e(t) still shifts every endpoint by

v

A :=

1

v(1−s)e(s)ds.

eL

0

Taking σ small makes X1∗ arbitrarily concentrated, so a clipped affine reward of slope Lr can be placed so that, with probability arbitrarily close to one, the shift by A either remains inside the linear region, giving

gap LrA, or crosses from the zero plateau to the one plateau, giving gap 1. Hence for every ζ > 0 there is a bounded Lr-Lipschitz reward such that

Ep

###### [r] − Ep∗

###### [r] ≥ min{1,LrA} − ζ. Thus the bounded-reward version of this amplification-tight construction is essentially tight as well.

1

1

| |
|---|

Remark: A simpler bounded-reward tightness statement. The tightness construction in part 3 is engineered to make the full amplification kernel eL

v(1−s) sharp, and for that reason it uses a somewhat contrived setup. This is useful for completeness, but it is more than is needed to show the main geometric point in the paper, namely that the factor Lrε itself cannot be removed. The next result isolates that point already for the standard linear Gaussian bridge with p0 = q = N(0,1).

- Theorem D.6 (Bounded-reward tightness of the Lrε factor). There is a universal constant c > 0 such that the following holds. Fix ε > 0 and L > 0 with εL < 1, and consider the one-dimensional CFM instance with standard linear interpolation

Xt = tX1 + (1 − t)X0, X0,X1 i.∼i.d. N(0,1),

whose source and target laws are both N(0,1) and whose population velocity v∗ is 1-Lipschitz in space. Then there is a velocity field v satisfying

|v(x,t) − v∗(x,t)| ≤ ε

sup

t∈[0,1], x∈R

and a reward r : R → [0,1] with Lip(r) ≤ L such that the endpoint law p of the ODE driven by v satisfies

Ep[r] − Eq[r] ≥ cLε, q = N(0,1). The explicit nonsmooth construction below gives c = 1/2.

Intuition. The idea is to build a reward together with a matching velocity perturbation that pushes every generated sample uphill of the reward. The reward is a periodic triangular wave: peaks of height 1 spaced 2/L apart, falling linearly to 0 at the midpoints. This makes the reward bounded in [0,1], L-Lipschitz, and ensures that every point is within distance 1/L of a peak. The bad velocity field is v∗ plus an admissible nudge of size ε pointing each sample toward its nearest peak. Over unit time, this moves each sample up a slope of magnitude L until it has traveled distance ε or reached a peak, so the reward gain is of order Lε on average. Since every sample moves locally uphill, there is no cancellation between samples moving in different directions. The gain is therefore insensitive to how spread out the samples are, which is why the standard bridge with p0 = q = N(0,1) already realizes an Ω(Lε) gap.

Proof. Write

ρt := t2 + (1 − t)2. Then qt = N(0,ρ2t), with ρ0 = ρ1 = 1 and 1/√2 ≤ ρt ≤ 1. First, the reference flow is a pure rescaling. The population CFM velocity is

v∗(x,t) = E[X1 − X0 | Xt = x]. Since (X0,X1,Xt) is centered and jointly Gaussian, as earlier we can compute this explicitly as: v∗(x,t) =

2t − 1 2t2 − 2t + 1

Cov(X1 − X0,Xt) Var(Xt)

ρ˙t ρt

x =

x =

x.

Since

2t − 1 2t2 − 2t + 1

2t − 1 2t2 − 2t + 1 ≤ 1 for all t ∈ [0,1],

∂xv∗(x,t) =

,

the field v∗(·,t) is 1-Lipschitz uniformly in t. Integrating X˙t = (ρ˙t/ρt)Xt gives Xt = ρtX0. Thus, in the rescaled coordinate

Xt ρt

,

Yt :=

the reference dynamics are frozen, Y˙t = 0, and the reference endpoint is X1∗ = Y0 ∼ N(0,1) = q. We now construct a bounded reward and an uphill perturbation to get the desired bound. Set T := 2/L. For a phase θ ∈ [0,T], to be chosen later, place reward peaks at the equally spaced points

{nT − θ : n ∈ Z}.

Let dθ(y) be the distance from y to the nearest peak, and define the triangular-wave reward rθ(y) := 1 − Ldθ(y).

Every point is at most half a period, T/2 = 1/L, from a nearest peak, so 0 ≤ rθ ≤ 1. Since dθ is 1-Lipschitz, rθ is L-Lipschitz. Let bθ(y) ∈ {−1,0,+1} be the unit direction of steepest ascent of rθ: it equals +1 if a nearest peak lies to the right of y, −1 if a nearest peak lies to the left, and 0 at peaks and troughs. Define

x ρt

vθ(x,t) := v∗(x,t) + ρtεbθ

.

Because ρt ≤ 1 and |bθ| ≤ 1,

|vθ(x,t) − v∗(x,t)| = sup

ρtε bθ

sup

t,x

t,x

x ρt ≤ ε.

Substituting this field into the dynamics of Yt = Xt/ρt, the rescaling terms cancel and Y˙t = εbθ(Yt).

Thus, in the rescaled coordinate, every trajectory moves at speed ε toward a nearest reward peak and stops on arrival.

Fix Y0 = y away from the null set of peaks and troughs. Over the unit time interval the trajectory moves a distance min{ε,dθ(y)} toward the nearest peak, so dθ decreases by exactly that amount and

rθ(Y1) − rθ(Y0) = Lmin{ε,dθ(Y0)}.

Couple the reference and perturbed flows through the same start Y0 ∼ N(0,1). The reference endpoint is X1∗ = Y0, while the perturbed endpoint is X1 = Y1 because ρ1 = 1. Therefore, for the fixed reward rθ,

G(θ) := Ep[rθ] − Eq[rθ] = E rθ(Y1) − rθ(Y0) = LEZ∼N(0,1) min{ε,dθ(Z)} .

It remains only to choose the phase. Average G over θ ∼ Unif[0,T]. For each fixed z, as θ ranges over one period, the location of z relative to the nearest peak is uniform over one period. Hence dθ(z) is uniform on [0,T/2] = [0,1/L] with density L. Since ε < 1/L,

Eθ min{ε,dθ(z)} = L

1/L

min{ε,u}du = L

0

ε

0

udu +

1/L

Lε2 2

εdu = ε −

.

ε

This quantity is independent of z, so

Lε2 2

EθG(θ) = L ε −

Lε 2 ≥

= Lε 1 −

- 1

- 2

Lε,

where the last inequality uses Lε < 1. Therefore some phase θ∗ satisfies G(θ∗) ≥ Lε/2 by simple properties of the expectation. Taking r = rθ∗ and v = vθ∗ proves the theorem with c = 1/2.

| |
|---|

- D.3 How RL Helps

- D.3.1 RL reward regret bound

- Proposition D.7 (RL reward regret bound). Let r : X → [0,1], let Pbase be the base trajectory distribution, define

dP∗ dPbase

exp(λr(X1)) Z

, Z := EP

(X0:1) =

[exp(λr(X1))],

base

and let P be any trajectory distribution with final-sample marginal p. Then the final-sample marginal of P∗ is the tilted target p∗ defined in Section 2, and

Ep∗[r] − Ep[r] ≤ 12 KL(P∥P∗) = λ2 LRL(P) − LRL(P∗) , (10) where LRL(P) = λ1 KL(P∥Pbase) − Ep[r].

Proof. Here Pbase, P, and P∗ are distributions over full sampled trajectories X0:1; formally, they are path measures. Because the Radon–Nikodym derivative of P∗ depends only on the terminal state X1, the final-sample marginal of P∗ is exactly

exp(λr(x))pbase(x) Z

p∗(x) =

, which is exactly the tilted target. Since

TV(p, p∗) = sup

|Ep[f] − Ep∗[f]|,

f: 0≤f≤1

the assumption r ∈ [0,1] gives

Ep∗[r] − Ep[r] ≤ TV(p, p∗). With the convention TV(p, p∗) = 21∥p − p∗∥1, Pinsker’s inequality yields

TV(p, p∗) ≤ 12 KL(p∥p∗), and the data processing inequality for the endpoint map X0:1  → X1 gives

KL(p∥p∗) ≤ KL(P∥P∗). Therefore

Ep∗[r] − Ep[r] ≤ 12 KL(P∥P∗).

The first inequality in (10) therefore follows by data processing for the endpoint map X0:1  → X1, which gives KL(p∥p∗) ≤ KL(P∥P∗). It remains to identify this trajectory-level KL with the RL objective gap. By the definition of P∗,

dP dP∗

KL(P∥P∗) = EP log

dP dPbase − EP log

dP∗ dPbase

= EP log

= KL(P∥Pbase) − EP [λr(X1) − log Z]

= KL(P∥Pbase) − λEp[r] + log Z. Rearranging gives

log Z λ

1 λ

KL(P∥P∗) −

LRL(P) =

. Since log Z/λ is constant, evaluating the same identity at P∗ and subtracting gives

KL(P∥P∗) = λ LRL(P) − LRL(P∗) , which is the second equality in (10).

| |
|---|

- E Proofs for Section 4: Method: Discriminator-Guided RL E.1 Why Representation Spaces Help

- E.1.1 Feature-space correction

For readability, we work throughout this subsection in the standard density setting. Assume every distribution we mention admits a well-defined continuous density, and that the corresponding conditionals p(x | z) for z = ϕ(x) are well defined. We write pϕ(z) for the density on Z induced by passing x ∼ p through ϕ, and

assume that qϕ(z) > 0 only where pϕbase(z) > 0.

- Proposition E.1 (Feature-space correction). Let ϕ : X → Z be an encoder, let h : Z → R be any scalar function, and let λ > 0. Assume

Zh,λ := Ex∼p

exp λh(ϕ(x)) < ∞. Define the tilted density by

base

exp(λh(ϕ(x))) Zh,λ

pbase(x). (11) Then:

ph,λ(x) :=

- (i) the feature marginal is tilted according to

pϕh,λ(z) =

exp(λh(z)) Zh,λ

pϕbase(z), (12)

where equivalently

Zh,λ = Ez∼pϕ

base

[exp(λh(z))];

- (ii) the conditional distribution inside each feature cell is unchanged:

ph,λ(x | z) = pbase(x | z) for pϕh,λ-a.e. z. (13)

- (iii) if

h∗(z) = log

qϕ(z) pϕbase(z)

, (14)

then pϕh∗,1 = qϕ;

- (iv) ph∗,1 is the unique solution of

KL(p∥pbase) subject to pϕ = qϕ. (15)

min

p

Proof. Let z = ϕ(x). For part (i), let g : Z → R be any bounded measurable test function. Then

Ez∼pϕ

###### [g(z)] = Ex∼p

[g(ϕ(x))]

h,λ

h,λ

1 Zh,λ

Ex∼p

[g(ϕ(x)) exp(λh(ϕ(x)))]

=

base

1 Zh,λ Z

g(z) exp(λh(z))pϕbase(z)dz.

=

Thus the pushed-forward law under ϕ has density

exp(λh(z)) Zh,λ

pϕbase(z),

pϕh,λ(z) =

since the right-hand side gives the same integral against every bounded measurable test function g. Hence

(12). Taking g ≡ 1 in the display above gives

1 Zh,λ Z

exp(λh(z))pϕbase(z)dz, so

1 =

Zh,λ = Ez∼pϕ

[exp(λh(z))]. This completes part (i).

base

- For part (ii), fix a measurable set A ⊆ X, and let g : Z → R be any bounded measurable test function. On the one hand, conditioning under ph,λ gives

Ex∼p

h,λ

[1A(x)g(ϕ(x))] =

Z

g(z)ph,λ(A | z)pϕh,λ(z)dz.

On the other hand, using the definition of ph,λ and then conditioning under pbase,

Ex∼p

h,λ

[1A(x)g(ϕ(x))] =

1 Zh,λ

Ex∼p

base

[1A(x)g(ϕ(x)) exp(λh(ϕ(x)))]

=

1 Zh,λ Z

g(z) exp(λh(z))pbase(A | z)pϕbase(z)dz

=

Z

g(z)pbase(A | z)pϕh,λ(z)dz,

where the last step uses part (i). Since these two expressions agree for every bounded measurable test function g, their integrands must agree dz-a.e.:

ph,λ(A | z)pϕh,λ(z) = pbase(A | z)pϕh,λ(z). Therefore

ph,λ(A | z) = pbase(A | z) for pϕh,λ-a.e. z. Since this holds for every measurable A, the conditional distributions agree. In our density setting, we record this as

ph,λ(x | z) = pbase(x | z), which proves part (ii).

- For part (iii), let

h∗(z) = log

qϕ(z) pϕbase(z)

.

Then

Zh∗,1 = exp(h∗(z))pϕbase(z)dz = qϕ(z)dz = 1. Therefore, by part (i),

pϕh∗,1(z) = exp(h∗(z))pϕbase(z) = qϕ(z), proving part (iii).

- For part (iv), let p be any distribution satisfying pϕ = qϕ. The KL chain rule gives

KL(p∥pbase) = KL pϕ ∥pϕbase + Ez∼pϕ[KL(p(· | z)∥pbase(· | z))]. (16) Under the constraint pϕ = qϕ, this becomes

KL(p∥pbase) = KL qϕ ∥pϕbase + Ez∼qϕ[KL(p(· | z)∥pbase(· | z))] ≥ KL qϕ ∥pϕbase .

The first term is fixed by the constraint, and the second term is always nonnegative. Equality holds if and only if

p(· | z) = pbase(· | z) for qϕ-a.e. z.

Hence the unique minimizer is the distribution with feature marginal qϕ and within-cell conditional pbase(· | z). Parts (ii) and (iii) show that ph∗,1 has exactly this marginal and these conditionals, so it attains the lower bound. Uniqueness follows because the marginal qϕ together with the conditionals pbase(· | z) determines the distribution p uniquely.

| |
|---|

Remark. We stated this subsection in the density setting because it is the clearest way to express the main idea: RL reweights feature cells by the feature-space density ratio and leaves the within-cell conditional unchanged. A more general measure-theoretic version replaces ordinary ratios such as qϕ(z)/pϕbase(z) with the Radon–Nikodym derivative dqϕ/dpϕbase, replaces p(x | z) with regular conditional distributions, and writes feature-space integrals against pϕbase(dz) rather than dz. The proof is otherwise the same.

- E.1.2 Feature-Space Test-Function Bound

- Theorem E.2 (Feature-space test-function bound). Let rˆ(x) = h(ϕ(x)) be a feature-based reward with implied target qˆλ ∝ exp(λrˆ)pbase, and suppose pθ is ∆-suboptimal for the KL-regularized objective with trade-off parameter λ. Then for any g : Z → R with ∥g∥∞ ≤ C,

√

. (17)

Ep

[g(ϕ(x))] − Eq[g(ϕ(x))] ≤ C

###### + Eqˆ

[g(ϕ(x))] − Eq[g(ϕ(x))] feature-space ratio error

2λ∆

θ

λ

RL suboptimality

Proof. By the triangle inequality, Ep

[g ◦ ϕ] − Eq[g ◦ ϕ] . (18)

###### [g(ϕ(x))] − Eq[g(ϕ(x))] ≤ Ep

###### [g ◦ ϕ] − Eqˆ

###### [g ◦ ϕ] + Eqˆ

θ

θ

λ

λ

First term (RL suboptimality). Since ∥g∥∞ ≤ C, the function g ◦ ϕ takes values in [−C,C]. Also, because qˆλ(x) ∝ exp(λrˆ(x))pbase(x),

1 λ

1 λ

1 λ

log Zˆλ (19)

KL(p∥pbase) − Ep[ˆr] =

LRL(p) =

KL(p∥qˆλ) −

for a constant Zˆλ independent of p. Therefore the unique minimizer of the RL loss is qˆλ, and the ∆suboptimality assumption implies KL(pθ ∥qˆλ) ≤ λ∆. By Pinsker’s inequality,

√

λ∆ 2

2λ∆. (20)

Ep

###### [g ◦ ϕ] − Eqˆ

[g ◦ ϕ] ≤ 2C TV(pθ, qˆλ) ≤ 2C

= C

θ

λ

Second term (feature-space ratio error). Since g ◦ ϕ depends on x only through ϕ(x),

[g] and Eq[g(ϕ(x))] = Eqϕ[g], (21)

Eqˆ

###### [g(ϕ(x))] = Eqˆϕ

λ

λ

so Eqˆ

###### [g] − Eqϕ[g] . Combining gives (17).

###### [g ◦ ϕ] − Eq[g ◦ ϕ] = Eqˆϕ

λ

λ

| |
|---|

- F Adjoint Matching: Background

This appendix contains a self-contained review of adjoint matching (Domingo-Enrich et al., 2025), the method we use to optimize the RL objective LRL (4). We first provide a general description of the method in the SOC setting and then specialize to flow models.

Stochastic optimal control formulation. Adjoint matching solves LRL (4) under the lens of stochastic optimal control (SOC), a framework for fine-tuning the dynamics of an SDE through a learned drift correction. Concretely, given a base SDE with drift b and noise σ(t), SOC adds a learned control uθ(x,t) to the drift,

dXt = b(Xt,t) + σ(t)uθ(Xt,t) dt + σ(t)dBt, X0 ∼ N(0,I), (22) and seeks the uθ minimizing a quadratic control cost minus the terminal reward,

E 12

min

uθ

1

∥uθ(Xt,t)∥2 dt − λr(X1) . (23)

0

In our setting b(x,t) = vt(x) + 12σ(t)2 st(x) is the base drift of the sampling SDE (2)—the dynamics that generate pbase—and uθ is the deviation we are learning to fine-tune the model. By Girsanov’s theorem, the path-KL between (22) and the σ(t)-base SDE is exactly the expected control energy,

KL(Pθ ∥Pbase) = 12 E

1

∥uθ(Xt,t)∥2 dt ,

0

so (23) is precisely the path-KL-regularized RL objective LRL (4). Optimizing uθ thus tunes the trajectory drift to push X1 toward high reward while staying close to the base process in path-KL. For flow models we parametrize uθ in terms of a learned velocity field; we defer the specifics to the specialization paragraph below.

The memoryless schedule. Solving (23) does not in general produce the desired endpoint tilt p∗(x) ∝ pbase(x)exp(λr(x)). Let

E 12

V (x,t) := min

uθ

1

∥uθ(Xs,s)∥2 ds − λr(X1) Xt = x

t

denote the optimal cost-to-go from state x at time t. Domingo-Enrich et al. (2025) show that the SOC optimum picks up an X0-dependent factor through the initial-time value V (·,0),

P∗(X0,X1) ∝ Pbase(X0,X1) exp λr(X1) + V (X0,0) ,

so once we marginalize over X0, the resulting tilt on X1 is no longer just exp(λr). The bias is largest when X0 and X1 are strongly coupled—exactly the case in flow matching, where the ODE makes X1 a deterministic function of X0.

To remove the bias, Domingo-Enrich et al. (2025) propose choosing σ during training so that the base process makes X0 and X1 independent. Under this memoryless schedule, V (X0,0) becomes a global constant and the endpoint marginal recovers the desired tilt p∗(X1) ∝ pbase(X1)exp(λr(X1)). For the linear interpolation, they show that the unique memoryless schedule is

σml(t) =

2(1 − t) t

, (24)

large at t → 0 (which destroys dependence on X0) and small at t → 1 (which keeps the endpoint clean). Importantly, Domingo-Enrich et al. (2025) further show that the memoryless schedule is the unique training schedule for which the fine-tuned optimum can be sampled at inference under any other noise schedule—in particular, the standard deterministic ODE (σ(t) = 0). Training with any non-memoryless schedule locks the fine-tuned model to that specific schedule at inference.

Adjoint methods and adjoint matching. A standard way of optimizing (23) is to build an estimator of its gradient in θ by sampling a trajectory {Xt} under uθ, computing the integrand, and backpropagating through

###### the discretized SDE (22). Domingo-Enrich et al. (2025) show that the resulting gradient agrees in expectation with the gradient of the matching objective

LBasic−AM(θ) = 12 E

1

0

uθ(Xt,t) + σ(t)⊤a(t) 2 dt , (25)

where {Xt} and {a(t)} are stop-gradient and a(t) is the adjoint state

1

∥uθ(Xs,s)∥2 ds − λr(X1) ,

- 1

- 2

a(t) := ∇Xt

t

computed by a single backward solve of the ODE

a˙(t) = − ∇x(b + σ uθ)(Xt,t) ⊤a(t) − 21∇x uθ(Xt,t) 2, a(1) = −λ∇r(X1). (26)

In practice, Domingo-Enrich et al. (2025) find this empirically underperforms. They instead propose replacing a(t) with the lean adjoint a˜(t), obtained by dropping the control-dependent terms of (26),

a˜˙(t) = − ∇xbbase(Xt,t) ⊤a˜(t), a˜(1) = −λ∇r(X1). (27) The adjoint matching loss is the same regression as (25) but with a˜ in place of a,

LAM(θ) = 12 E

1

0

uθ(Xt,t) + σ(t)⊤a˜(t) 2 dt . (28)

Domingo-Enrich et al. (2025) show that LAM has the same minimizer u∗ as LBasic−AM. Crucially, the lean adjoint is still anchored to the reward through its terminal condition a˜(1) = −λ∇r(X1), so the regression target remains informed by the reward landscape; dropping the control-dependent terms only reduces variance and yields significantly better empirical performance.

Specialization to flow matching. For the linear interpolation, the base drift of (2) under σml from (24) is bbase(x,t) = 2vbase(x,t) − x/t. We parametrize the control uθ so that the controlled drift bbase + σml uθ has the same shape as bbase but with a learned velocity vθ in place of vbase. Solving for uθ yields

2 σml(t)

uθ(x,t) =

vθ(x,t) − vbase(x,t) . (29)

The reason for this parametrization is practical: fine-tuning then reduces to learning a new velocity field vθ, which can be plugged directly into the standard deterministic ODE for sampling at inference time, with no need for the memoryless SDE. Substituting (29) into (22) gives the corresponding memoryless training SDE,

Xt t

dXt = 2vθ(Xt,t) −

dt +

2(1 − t) t

dBt. (30)

Substituting (29) and bbase into (27) and (28) gives the lean adjoint and adjoint matching loss for flow models,

⊤

a˜˙(t) = − ∇x 2vbase(Xt,t) − Xt/t

a˜(t), a˜(1) = −λ∇r(X1), (31)

2

1

2 σml(t)

LFMAM(θ) = 12 E

dt . (32)

vθ − vbase + σml(t) ˜a(t)

0

The discretization of (30) and (31) introduces numerical stiffness at t → 0, which motivates the local linear integrator (section G). Implementation choices are collected in section H.

### G Local Linear Integrator for the Memoryless SDE

###### JiT

###### RAE

###### REPA

###### SiT

600

600

80

80

500

xt 2

500

400

60

60

400

300

40

40

200

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

t

t

t

t

ODE Euler (target) SDE Euler (with oﬀset) SDE Local linear (ours)

- Figure 13 Latent norm trajectories under Euler ODE, Euler SDE, and local linear SDE sampling. Estimate of expected latent norm E[∥Xt∥] against time for 1000 trajectories sampled under the Euler ODE integrator, Euler SDE integrator with offsets, and the local linear SDE integrator. A perfect SDE integrator should match the ODE norm distribution. The panels show that the Euler SDE integrator contains an early time blow-up in norm, while our local linear SDE integrator stays close to the ODE reference across models. Red arrows mark the early-time Euler blow-up in each panel, while blue arrows point to the corresponding local linear trajectory at t = 0, where no such blow-up occurs.

This appendix describes the training-time integrator used in our experiments, explains the numerical issues that motivate it, and presents empirical results that validate its use. Although our experiments focus on adjoint matching, the same problems arise whenever an RL algorithm samples the memoryless SDE, and therefore apply to any path-based RL method on flow models, including non-adjoint-matching approaches such as Flow-GRPO (Liu et al., 2025a).

Given its superior performance and simplicity, we believe this integrator should become the standard for all methods that require sampling from the memoryless SDE for flow models.

- G.1 The Memoryless SDE and the Local Linear Integrator

Recall from section F that solving the KL-regularized RL problem for flow models requires sampling from the memoryless SDE during training,

dXt = 2vθ(Xt,t) −

1 t

Xt dt +

2(1 − t) t

dBt. (33)

Using this memoryless SDE during training is what guarantees that the optimal terminal marginal is the tilted distribution p∗(x) ∝ pbase(x)exp(r(x)), while still allowing us to switch back to the standard ODE at test time.

The difficulty is that equation (33) is numerically stiff near t = 0: both the linear drift coefficient −1/t and the diffusion coefficient 2(1 − t)/t diverge as t → 0+. A naive explicit discretization therefore struggles on the first few steps.

Euler–Maruyama with offsets. Domingo-Enrich et al. (2025) handle this stiffness by replacing the singular terms 1/t and 2(1 − t)/t with the regularized versions 1/(t + δ) and 2(1 − t + ε)/(t + δ), and then applying Euler–Maruyama to the modified SDE. With the practical choice δ = ε = ∆t, the update is

Euler–Maruyama with δ = ε = ∆t (Domingo-Enrich et al., 2025):

2(1 − tk + ∆t)∆t tk + ∆t

xk tk + ∆t

xEulerk+1 = xk + 2vθ(xk,tk) −

ξk, ξk ∼ N(0,I). (34)

∆t +

[Figure 47]

LocallinearEuler

- Figure 14 Matched-noise JiT samples at K=50. Top: Euler–Maruyama. Bottom: local linear. Under the same Brownian motion, Euler with δ=ε=∆t already produces visibly degraded samples while local linear with the same offsets remains stable.

a pronounced early-time blow-up.

The shift δ moves the singularity away from t = 0, and ε provides slack near t = 1 in the adjoint matching loss, which DomingoEnrich et al. (2025) found to improve training speed.2

In practice, we observed that this discretization still produced lowquality samples for some models. The failure mode matches the intuition above: even with the offsets, an explicit first-order step cannot keep up with the rapid contraction of the stiff linear part at early times, so the noise increment dominates and trajectories blow up in norm, taking the model far from the training distribution. Figure 13 illustrates the effect across the four ImageNet models used in the paper. It plots the latent norm of 1000 trajectories under the Euler ODE, under Euler SDE with offsets, and under the local linear SDE integrator introduced next. A marginal-preserving SDE integrator should produce the same norm distribution as the ODE, but the Euler–Maruyama SDE discretization instead exhibits

The local linear integrator. We address the same stiffness with a different splitting. Our key observation is that if we freeze the velocity vθ(Xt,t) to its left-endpoint value on each interval [tk,tk+1], the remaining dynamics form a linear SDE with time-dependent coefficients, which can be integrated in closed form via an integrating factor. Writing uk := tk + δ for the shifted time and specializing to δ = ε = ∆t as in the Euler–Maruyama baseline, this yields

Local linear update with δ = ε = ∆t (ours):

xLLk+1 = Φkxk + Ωk vθ(xk,tk) + Vk ξk, ξk ∼ N(0,I), (35) Φk =

uk uk+1

- 2

- 3

uk+1 − Φ2kuk .

, Ωk = uk+1(1 − Φ2k), Vk = (1 + 2∆t)(1 − Φ2k) −

A full derivation, including the general case δ,ε ≥ 0, is given in section G.2.

Compared to Euler–Maruyama, the local linear integrator has the same computational cost, namely one network evaluation per step, but it handles the stiff linear part of the SDE exactly. The contraction factor Φk = uk/uk+1 stays in (0,1) throughout and is smallest near t = 0 (where the stiffness is largest), so it damps the early-time dynamics that would otherwise drive Euler– Maruyama trajectories to blow up. This is why local linear trajectories stay close to the ODE reference in figure 13. Moreover, no term has any problematic singularity.

−0.2

−0.4

ImageReward

−0.6

SDE Euler (with offset) K=50

SDE Euler (with offset) K=100

SDE Local linear (ours) K=50

SDE Local linear (ours) K=100

−0.8

−1.0

Empirical improvements. Replacing Euler–Maruyama with the local linear integrator improves both sampling and training. The clearest case is JiT, which was the most challenging model in our experiments: figure 14 shows that at K = 50 steps the local linear integrator already produces coherent images, while Euler–Maruyama with offsets produces visibly degraded samples under the same Brownian motion.

0 500 1000 1500 2000 2500 3000 Training step

Figure 15 JiT training reward during adjoint matching.

Local linear with δ=ε=∆t at K=50 closely tracks the K=100 run, while Euler with the same offsets and step count barely improves the reward. Both local linear runs outperform the Euler run at K=100, showing that the integrator matters for both optimization quality and speed.

This can also be verified quantitatively, and the pattern persists across step counts and models. Figure 16 reports

2The precise role of ε is orthogonal to the present discussion, but we keep it in the expressions for completeness.

Fréchet distances under the Inception-v3, DINOv2-B, and DINOv3-L feature spaces for all four ImageNet models. Across model families, local linear with δ = ε = ∆t Pareto-dominates Euler–Maruyama with the same offsets, with the clearest gains at low step counts. The exception is RAE under Inception-v3, where both integrators produce high-quality samples and Euler-Maruyama with offsets slightly outperforms local linear.

These improvements carry over to training as well. Figure 15 shows the training reward for JiT on ImageNet using the setup of the main reward-finetuning experiments. At K = 50, local linear matches the reward achieved by its K = 100 equivalent. On the other hand, Euler–Maruyama with offsets at K = 50 barely improves over the baseline, and even at K = 100 it still underperforms the local linear runs. This shows that the choice of integrator changes the effective optimization problem that RL sees, and that a better integrator can lead to faster training and better final performance. The 2× speedup is significant as sampling is the most expensive part of training.

###### Inception-v3

DINOv2-B

###### DINOv3-L

150

1000

SDE Euler (with offset) SDE Local linear (ours)

750

100

100

JiT

FD

500

50

250

50

40 60 80 100

40 60 80 100

40 60 80 100

400

110

100

SiT

20

FD

300

90

10

200

80

40 60 80 100

40 60 80 100

40 60 80 100

30

90

300

REPA

80

20

FD

70

200

10

60

40 60 80 100

40 60 80 100

40 60 80 100

- 3

- 4

- 5

35.0

- 7

- 8

RAE

32.5

FD

30.0

27.5

40 60 80 100

40 60 80 100

40 60 80 100

K

K

K

Figure 16 Cross-model endpoint FD for Euler vs. local linear integrators. Rows correspond to models and columns to the three evaluation feature spaces. Local linear with δ=ε=∆t consistently matches or improves upon Euler with the same offsets, especially at low K. All metrics were computed with 50, 000 samples.

Extension to the backward adjoint. The same idea applies to the backward adjoint ODE that appears in adjoint matching. Defining u(t) = t + δ (so uk = u(tk) matches the shifted time used above) and writing Φk = uk/uk+1 for brevity, recall that adjoint matching requires integrating

a˙t = − ∇x 2vbase(Xt,t) − Xt/u(t) ⊤at = −2 ∇xvbase(Xt,t) ⊤at +

at u(t)

, (36)

where the Jacobian of the stiff linear drift −x/u(t) contributes the scalar −1/u(t) times the identity. (We write equation (36) in terms of u(t) rather than t, as in equation (31), given the practical choice of shifting the time to avoid the singularity at t = 0.) Domingo-Enrich et al. (2025) use a standard backward Euler step to discretize equation (36):

Standard backward Euler adjoint step (Domingo-Enrich et al., 2025):

a˜k+1 uk+1

a˜Eulerk = a˜k+1 + ∆t 2 ∇xvbase(Xk+1,tk+1) ⊤a˜k+1 −

. (37)

Using a similar splitting idea, which we derive in section G.2 below, one can instead integrate the stiff scalar term exactly and replace this step by:

Local linear backward adjoint step (ours):

uk uk+1

a˜LLk = Φk a ˜k+1 + 2∆t ∇xvbase(Xk+1,tk+1) ⊤a˜k+1 , Φk =

. (38)

In practice we did not see gains as large as for the forward integrator, but this step performed at least as well as the Euler discretization, so we kept it.

- G.2 Derivation of the Local Linear Integrator

Both derivations rest on the following simple lemma (for further background in the context of ODEs see Arnold

(1978)). Lemma G.1. Let f,g,h : [s,T] → R be continuous and deterministic. The unique solution of the SDE

dYt = f(t)Yt dt + g(t)dt + h(t)dBt, Ys = ys, is

T

T

T

T

T

s f(ζ)dζ ys +

τ f(ζ)dζ g(τ)dτ +

τ f(ζ)dζ h(τ)dBτ.

YT = e

e

e

s

s

The stochastic integral is a centered Gaussian with variance s T e2

T

τ f(ζ)dζ h(τ)2 dτ. The ODE case h ≡ 0 is recovered as a special case.

Proof. Let µ(t) = exp − s t f(ζ)dζ , so that µ(s) = 1 and µ˙(t) = −f(t)µ(t). Since µ is deterministic and of bounded variation, Itô’s product rule applied to µ(t)Yt gives

d[µ(t)Yt] = µ˙(t)Yt dt + µ(t)dYt = −f(t)µ(t)Yt dt + µ(t) f(t)Yt dt + g(t)dt + h(t)dBt . The two f(t)Yt terms cancel, leaving

d[µ(t)Yt] = µ(t)g(t)dt + µ(t)h(t)dBt. Integrating from s to T and using µ(s)Ys = ys,

µ(T)YT = ys +

Dividing through by µ(T) gives

T

µ(τ)g(τ)dτ +

s

T

µ(τ)h(τ)dBτ.

s

YT =

1 µ(T)

ys +

T

µ(τ) µ(T)

g(τ)dτ +

s

T

µ(τ) µ(T)

h(τ)dBτ,

s

and the identities 1/µ(T) = exp s T f(ζ)dζ and µ(τ)/µ(T) = exp τ T f(ζ)dζ yield the claimed formula. Gaussianity and the variance formula for the stochastic integral follow from Itô’s isometry applied to the

deterministic integrand µ(τ)h(τ)/µ(T). Throughout this section we keep the offsets δ,ε ≥ 0 and continue to use

| |
|---|

uk uk+1

u(t) = t + δ, uk = u(tk), Φk =

###### .

Forward SDE

The forward SDE

Xt u(t)

dXt = 2vθ(Xt,t) −

dt +

2(1 − t + ε) u(t)

dBt (39)

matches lemma G.1 with f(t) = −1/u(t), smooth forcing g(t) = 2vθ(Xt,t), and noise coefficient h(t) =

2(1 − t + ε)/u(t). The exponential factor in the lemma simplifies because −1/u has a logarithmic antiderivative:

u(s) u(t)

t

s f(ζ)dζ = e−[logu(t)−logu(s)] =

e

.

Applying lemma G.1 over a single step [tk,tk+1] gives the exact representation

Xt

###### = Φk Xt

k+1

k

2 uk+1

+

tk+1

1 uk+1

u(τ)vθ(Xτ,τ)dτ +

tk

tk+1

u(τ)

tk

2(1 − τ + ε) u(τ)

dBτ. (40)

We make a single approximation—freezing the velocity at its left-endpoint value, vθ(Xτ,τ) ≈ vθ(xk,tk) =: vkand integrate the rest exactly. The stochastic-integral clause of lemma G.1 turns the noise integral into a centered Gaussian.

The velocity term collapses to Ωk vk with

tk+1

2 uk+1

Ωk =

u(τ)dτ =

tk

The noise term is √Vk ξk with ξk ∼ N(0,I) and

2 u2k+1

Vk =

tk

u2k+1 − u2k uk+1

= uk+1 1 − Φ2k . (41)

tk+1

u(τ)(1 − τ + ε)dτ.

Using the algebraic identity 1 − τ + ε = (1 + δ + ε) − u(τ), the integrand splits into elementary terms:

u3k+1 − u3k 3

u2k+1 − u2k 2

tk+1

tk+1

u(τ)2 dτ =

,

.

u(τ)dτ =

tk

tk

Substituting and simplifying with u3k/u2k+1 = Φ2k uk,

- 2

- 3

uk+1 − Φ2k uk . (42) Setting δ = ε = ∆t recovers the boxed forward update equation (35).

Vk = (1 + δ + ε) 1 − Φ2k −

Backward adjoint ODE

We now derive the boxed local linear backward adjoint step equation (38). Recall the split form of the adjoint ODE equation (36),

at u(t)

a˙t = −2 ∇xvbase(Xt,t) ⊤at +

.

To handle the stiff scalar term exactly, apply lemma G.1 to this ODE—now with f(t) = 1/u(t) and smooth forcing g(t) = −2[∇xvbase(Xt,t)]⊤at. The exponential factor is again a ratio of u’s, with the opposite sign from the forward case:

u(t) u(s)

t

s f(ζ)dζ =

e

.

Applying the lemma over [tk,tk+1] to express at

and rearranging for the backward direction gives the exact identity

###### in terms of at

k+1

k

a˜k = Φk a˜k+1 + 2

tk+1

###### uk u(τ) ∇xvbase(Xτ,τ) ⊤aτ dτ.

tk

The integrand depends on the unknown aτ, so we must approximate. Following the same spirit as in the forward case, we freeze the integrand at a single point and use the resulting constant times ∆t. We freeze at τ = tk+1, the only point where aτ is known: this replaces aτ by a˜k+1, the Jacobian by ∇xvbase(Xk+1,tk+1), and the kernel uk/u(τ) by uk/uk+1 = Φk. The integral collapses to 2Φk ∆t times the frozen value, yielding

a˜k = Φk a ˜k+1 + 2∆t ∇xvbase(Xk+1,tk+1) ⊤a˜k+1 , which is equation (38).

- H Additional Implementation Details

This appendix gives the method-level implementation recipe used in our DRL experiments. The end-to-end procedure is summarized in algorithm 2, which combines the buffered training pattern used in practice for both stages with the local linear integrator for the memoryless SDE and its companion adjoint (section G). We provide further details below.

Experiment-level configuration (base models, default hyperparameters, sampling, evaluation metrics, baseline and ablation setups) is collected in section I. The shifted-time offsets in Stage 2 are written as δ,ε ≥ 0 throughout the algorithm; in practice we use the choice δ = ε = ∆t from section G, but stating the algorithm in terms of δ,ε keeps the dependence on the offsets explicit.

Buffered discriminator training. To amortize the cost of generating model samples, we do not draw a fresh mini-batch of fakes for every gradient step. Instead, we periodically refresh a buffer BD of size ND holding paired real samples xreal and base-model samples xfake, and then take ED epochs of gradient steps over that buffer with the standard logistic loss plus an R1 gradient penalty. The R1 penalty is computed by backpropagating the discriminator output through the frozen encoder ϕ to the image space. In principle, the flow-space could be used as well but we found this to work well. Given that it is also more memory efficient

- as we avoid backpropagating through the flow space, we decided to use it instead of the flow-space penalty.

Buffered adjoint matching. We use the same buffering pattern for the RL stage. Periodically, we sample NT trajectories from the memoryless training SDE under the current vθ using the local linear forward integrator (35), compute their adjoints by solving the local linear backward adjoint (38) starting from a˜K = −λ∇xrˆ(XK), and store both quantities in a trajectory buffer BT as stop-gradient tensors. We then take ET epochs of gradient steps over BT, with gradients flowing only through the current velocity evaluations vθ(Xk,tk). This is technically off-policy (the buffer drifts as vθ updates), but in practice we observed no degradation versus a fully on-policy implementation, and Girsanov-based importance weighting gave no measurable improvement either.

Training versus inference. The memoryless SDE used during post-training is integrated with the local linear scheme described in section G. After fine-tuning, we sample from the updated model with the standard deterministic ODE; differences between the two samplers were small but non-negligible.

Otherimplementationdetails. A few smaller choices, all following or adapted from the original adjoint-matching implementation:

- 1. Final denoising step. The forward loop in algorithm 2 runs K − 1 local linear SDE steps and replaces the final step from tK−1 to tK with a single deterministic denoising call to the network.
- 2. Stratified timestep subsampling for the loss. Rather than summing the AM loss over all K timesteps, each gradient step uses a fresh subset S ⊂ {0,...,K − 1} of size ⌈fK⌉, with f = 0.4 by default. The subset is split 50/50 between early (t < 0.6) and late (t ≥ 0.6) timesteps so that both ends of the trajectory are always represented. The loss line in Stage 2 then becomes LAM = (1/(2|S|)) k∈S ∥ · ∥2.
- 3. Adaptive outlier clipping for the AM loss. The per-sample loss ℓ(ki) = σ 2

(vθ − vbase) + σk a˜k 2 has a heavy right tail, dominated by a few trajectories whose adjoints blow up and destabilize training. We maintain an EMA-smoothed threshold τn across optimizer steps n on the per-sample norms ℓ(ki),

k

τn = ρτn−1 + (1 − ρ) min qα ℓ(ki) , cτn−1 , where qα is the empirical α-quantile over the current batch, ρ is the EMA decay, and the spike cap c prevents a single outlier batch from rapidly inflating the threshold. Samples with ℓ(ki) ≥ τn (or non-finite) are masked out of the gradient. We use α = 0.9, ρ = 0.9 for SiT-XL/2, REPA SiT-XL/2, and JiT-H/16, and α = 0.75, ρ = 0.995 for RAE DiTDH-XL, with spike cap c = 5 throughout. The spike cap c was particularly important for RAE, whose loss exhibited noticeably more frequent spikes than the other models.

Algorithm 2 Discriminator-Guided RL with Buffered Training and the Local Linear Integrator

Stage 1: Buffered Feature-Space Discriminator Training Require: Target distribution q, base model pbase, frozen encoder ϕ, buffer size ND, epochs per refresh ED, R1 weight γ, learning

rate ηD

- 1: Initialize discriminator parameters ψ
- 2: repeat
- 3: Refresh buffer BD:
- 4: for i = 1, . . . , ND do
- 5: Sample x(reali) ∼ q and x(fakei) ∼ pbase
- 6: Append x(reali) , x(fakei) to BD
- 7: end for
- 8:
- 9: Train discriminator on BD:
- 10: for epoch n = 1, . . . , ED do
- 11: for minibatch (xreal, xfake) ⊂ BD do
- 12: ℓdisc(ψ) ← − log Dψ(ϕ(xreal)) − log 1 − Dψ(ϕ(xfake)) + γ2 ∇xDψ(ϕ(xreal)) 2

- 13: ψ ← ψ − ηD ∇ψℓdisc(ψ)
- 14: end for
- 15: end for
- 16: until convergence
- 17: Define reward rˆ(x) ← logit Dψ(ϕ(x))

Stage 2: Buffered Adjoint Matching with the Local Linear Integrator Require: Reward rˆ, base velocity vbase, KL weight λ, uniform time grid 0 = t0 < · · · < tK = 1 with ∆t = 1/K, shifted-time

offsets δ, ε ≥ 0 with uk := tk + δ, trajectory buffer size NT , epochs per refresh ET , learning rate ηT

- 1: Initialize vθ ← vbase
- 2: repeat
- 3: Refresh trajectory buffer BT :
- 4: for i = 1, . . . , NT do
- 5: Sample X0(i) ∼ N(0, I)
- 6: Local linear forward pass (with current vθ):
- 7: for k = 0, . . . , K − 1 do
- 8: Φk ← uk/uk+1, Ωk ← uk+1 1 − Φ2k , Vk ← (1 + δ + ε) 1 − Φ2k − 23 uk+1 − Φ2kuk

- 9: ξk(i) ∼ N(0, I)
- 10: Xk(i+1) ← Φk Xk(i) + Ωk vθ(Xk(i), tk) + √Vk ξk(i)

- 11: end for
- 12: Terminal adjoint a˜(Ki) ← −λ ∇xrˆ XK(i)
- 13: Local linear backward adjoint pass (with vbase):
- 14: for k = K − 1, . . . , 0 do
- 15: a˜(ki) ← Φk a ˜(ki+1) + 2∆t ∇xvbase Xk(i+1) , tk+1 ⊤a˜(ki+1)
- 16: end for
- 17: Store {Xk(i)}Kk=0, {a˜(ki)}Kk=0 in BT as stop-gradient quantities
- 18: end for
- 19:
- 20: Train velocity on BT :
- 21: for epoch n = 1, . . . , ET do
- 22: for minibatch {Xk}, {a˜k} ⊂ BT do
- 23: Set σk ← 2(1 − tk + ε)/uk

- 24: LAM(θ) ← 21K Kk=0−1 σ 2

k

vθ(Xk, tk) − vbase(Xk, tk) + σk a˜k

2

- 25: θ ← θ − ηT ∇θLAM(θ)
- 26: end for
- 27: end for
- 28: until convergence
- 29: return vθ

### I Experimental Setup

We collect all experiment-level configuration here: shared setup (base models, sampling, evaluation metrics) followed by per-experiment subsections covering our main distributional alignment experiments (section 5), the baselines we compare against, and the ablations discussed in design choices. The general method recipe is given separately in section H.

Base models. We evaluate four pretrained ImageNet (Deng et al., 2009) 256×256, class-conditional generators spanning latent and pixel-space flow/diffusion architectures:

- • SiT-XL/2 (Ma et al., 2024) (0.72B params total: 675M flow model, 50M decoder): A generative model built on the DiT backbone (Peebles and Xie, 2023) using the stochastic interpolant framework (Albergo and Vanden-Eijnden, 2023). We use the XL/2 variant, operating in the standard SD VAE latent space (4 × 32 × 32).
- • JiT-H/16 (Li and He, 2025) (0.95B params total: 953M flow model, 0 decoder): A plain Vision Transformer (Dosovitskiy et al., 2021) that performs flow matching directly in pixel space on raw image patches, without a VAE tokenizer. We use the H/16 variant operating directly on 3×256×256 pixel inputs.
- • REPA SiT-XL/2 (Yu et al., 2024) (0.72B params total: 675M flow model, 50M decoder): A flow-based diffusion transformer regularized by aligning intermediate noisy representations with clean semantic features from a frozen encoder. We use the variant built on the SiT-XL/2 backbone (same SD VAE latent space, 4 × 32 × 32), trained with REPA alignment against DINOv2-ViT-B features.
- • RAE DiTDH-XL (Zheng et al., 2025) (1.25B params total: 839M flow model, 415M ViT-XL decoder): A diffusion transformer that replaces the VAE latent space with a Representation Autoencoder—a frozen vision encoder paired with a lightweight decoder—and trains via flow matching in this semantic latent space. We use the DiTDH-XL stage-2 model with a frozen DINOv2-ViT-B encoder and ViT-XL decoder, operating in a 768 × 16 × 16 semantic latent space.

Sampling. For all models, unless otherwise indicated, we use a Heun 50-step sampler with a linear schedule, except for RAE where we follow the original paper and use their suggested scheduler (Zheng et al., 2025). This corresponds to 99 NFEs since the final step reduces to a regular Euler step.

Evaluation metrics and feature spaces. For our main results, we report metrics in the DINOv2-Large, DINOv3Large, SigLIP-Large, and Inception-v3 feature spaces. The feature-space ablation also reports the corresponding base-size variants (DINOv2-B, DINOv3-B, SigLIP-B); patterns are largely similar to their large-size counterparts. All distributional metrics are computed on 50000 generated samples with class labels balanced across the 1000 ImageNet classes, against an equally sized reference set so that the generated and reference sets are matched in size. As is commonly done in the literature, FD is reported against the (µ,Σ) feature statistics of the full ImageNet training set; FDval, KD, Precision, Recall, Density, and Coverage are reported against 50000 ImageNet validation images. Precision/Recall are computed with k=3 nearest neighbors and Density/Coverage with k=5, matching the defaults of Kynkäänniemi et al. (2019) and Naeem et al. (2020) respectively. All manifold metrics use Euclidean distance directly on the embedding vectors.

Best-CFG configuration. Whenever we report “best CFG” numbers, the configuration is selected per checkpoint by sweeping the CFG scale from 1 to 3 in steps of 0.25 together with the application interval {(0.0,1.0),(0.1,1.0),(0.3,1.0)} (the time range t ∈ [low,high] over which CFG is applied; outside it the guidance scale falls back to 1.0) (Kynkäänniemi et al., 2024). For the autoguidance experiments (section J) we also sweep scales from 0 to 1 in the same step.

- I.1 Distributional alignment setup

DRL is run as two stages—a discriminator on frozen-encoder features, followed by KL-regularized RL via adjoint matching—both using buffered training (algorithm 2). The same checkpoints back the distributional alignment results and the image-quality transfer results.

Discriminator architecture. Unless stated otherwise, the discriminator operates on frozen DINOv2-Large features (CLS token; the encoder is never updated during DRL). The head is a class-conditional projection

discriminator (Miyato and Koyama, 2018) of the form D(x,y) = φ(h(x)) + ⟨h(x),e(y)⟩ + b(y), where h is a linear map (no hidden layers) with hidden dim 512 and e(y) is a learned class embedding. The reward used by the RL stage is the discriminator logit, rˆ(x) = logitD(ϕ(x)), as constructed in section 4.

Discriminator training. We optimize the standard logistic GAN loss with the R1 gradient penalty above. Each refresh of the discriminator buffer holds 20 mini-batches of paired (real, fake) images (in the ambient space of the model), and we take 10 epochs of gradient steps over that buffer before refreshing. We optimize with Adam at learning rate 10−4 at an effective batch size of 512, for up to 10 000 optimizer steps, evaluating on a held-out validation set every 50 steps and early-stopping with patience 15.

RL training (adjoint matching). We optimize the RL objective with adjoint matching (Domingo-Enrich et al., 2025), sampling trajectories from the memoryless training SDE using the local linear integrator (section G) at K = 100 uniform steps with η = 1 and offsets δ = ε = ∆t. CFG is disabled during training. We use Adam

- at learning rate 10−5 (no weight decay) with a 200-step linear warmup, run for 3000 optimizer steps, and sum the loss across timesteps. In addition to the per-trajectory adaptive clipping and stratified timestep subsampling described in section H, trajectories and their adjoints are pre-computed and stored stop-gradient in a buffer; we then take 5 epochs of velocity gradient steps over that buffer per refresh. We use an effective batch size of 576 for all models, with a buffer of 2304 trajectories for SiT-XL/2 and REPA SiT-XL/2 and a buffer of 3072 trajectories for JiT-H/16 and RAE DiTDH-XL. We observed that RAE was noticeably more unstable to train than the other models, so for RAE we increased the quantile clipping (q = 0.75 with EMA decay 0.995, vs q = 0.9 and EMA decay 0.9 elsewhere) and reduced the loss to its mean rather than its sum across timesteps to compensate for the larger latent space (768 × 16 × 16 vs 4 × 32 × 32). We also use a slightly higher learning rate of 5 × 10−5 (vs 10−5 elsewhere), which we found to work better in practice for RAE, together with gradient clipping of 1.0 and a spike cap of 5 on the clipped loss-threshold EMA.

Compute. Each base model requires a one-off discriminator training run plus one adjoint-matching run per λ. All numbers below are measured on a cluster of NVIDIA H200 GPUs (8 per node). The discriminator stage takes roughly 3–6 hours of wall-clock time on a single 8×H200 node, i.e. approximately ∼24–48 GPU-hours per base model. The per-λ adjoint-matching cost is approximately ∼190 GPU-hours for SiT-XL/2 and REPA SiT-XL/2 (∼110 in optimizer steps and ∼80 in buffer-fill sampling), ∼500 GPU-hours for JiT-H/16 (∼290 in optimizer steps, ∼210 in buffer-fill sampling, with the higher cost reflecting pixel-space ViT-H/16 forward passes), and ∼420 GPU-hours for RAE DiTDH-XL (∼190 in optimizer steps, ∼230 in buffer-fill sampling, with the buffer-fill share being largest because sampling in the 768 × 16 × 16 semantic latent is more expensive than in the SD VAE latent).

For comparison, the base models we post-train are themselves the result of substantial pretraining budgets: SiT-XL/2 is reported to be trained for approximately 18,300 TPU v4 chip-hours (Ma et al., 2024), and REPA SiT-XL/2 for 1,646 H100 GPU-hours (Yu et al., 2024) (to our knowledge no GPU-hour figures have been reported for JiT or RAE). DRL post-training is therefore a small fraction of the cost of producing the base model in either case. We also note that we did not spend significant effort tuning DRL for training-time efficiency: we believe it is straightforward to roughly halve the per-λ cost by trading off the configuration knobs we held fixed—fewer discretization steps, larger trajectory buffers, more epochs per buffer refresh, and so on—without changing the qualitative results.

- I.2 Better image quality (reward-transfer evaluation) setup

The image-quality transfer results reuse the distributional alignment checkpoints (section I.1) without any additional training, and only the evaluation protocol is new. We evaluate four held-out preference reward models—ImageReward (Xu et al., 2023), PickScore (Kirstain et al., 2023), Aesthetics v2.5 (discus0434, 2024), and HPSv2 (Wu et al., 2023)—over 50000 generated images per checkpoint (balanced across the 1000 ImageNet classes). For the text-conditioned rewards, the ImageNet class label is plugged into the prompt template “a photo of a {class}”.

- I.3 Preference-based RL setup

For the preference alignment results, we post-train each base model with KL-regularized RL using ImageReward (Xu et al., 2023) as the reward (computed on the rendered image with the same “a photo of a {class}”

prompt template as in section I.2), and compare two starting points: (i) the base model (Base+PRL) and (ii) our DRL λ=10 checkpoint (DRL+PRL). For each starting point we sweep the PRL reward–KL trade-off λPRL ∈ {1,10,40}. For DRL+PRL the DRL checkpoint serves as both the initial weights and the KL reference, so the KL is anchored to the distributionally aligned model rather than the base.

RL training (adjoint matching). PRL uses the same adjoint matching setup as the distributional alignment stage (section I.1—same batch size, buffer sizes, and RAE-specific overrides), trained for 5 000 optimizer steps instead of 3000. We chose this larger budget after observing that the Base+PRL runs at 3000 steps were close to but not yet fully converged.

Compute. Same per-λ figures as section I.1, scaled proportionally to the longer 5000-step budget (a factor of 5/3).

Evaluation for the Pareto / over-optimization analysis. Beyond the same distributional and reward metrics from section I.1, the analysis behind figure 9 (and the appendix Pareto figures) tracks five low-level image statistics that reward-hacking can push to extreme values: mean brightness, saturation, contrast, colorfulness, and whiteness. Each statistic is computed per image on RGB pixels in [0,1] and then averaged across all images:

- • Brightness: mean grayscale luminance Y = 0.299R + 0.587G + 0.114B over all pixels (BT.601 weights).
- • Saturation: mean of (maxc −minc)/maxc across pixels (channel-wise max/min over RGB; clamped to avoid division by zero).
- • Contrast: pixel-wise standard deviation of the grayscale luminance Y .
- • Colorfulness: the Hasler–Süsstrunk metric σ + 0.3µ, with σ = σrg2 + σyb2 and µ = µ2rg + µ2yb on the opponent channels rg = R − G and yb = 12(R + G) − B.

- • Whiteness: mean of minc(R,G,B) across pixels (high when pixels are close to white).

Statistics are computed on 3000 generated images per checkpoint (three images per ImageNet class, sampled with fixed noise seeds shared across checkpoints so trajectories are directly comparable). The ImageNet reference values are computed on 3000 ImageNet validation images (three per class, resized and 256 × 256 center-cropped).

- I.4 Distillation from RL teachers setup

The distillation-gap experiment (section 3) on Stable Diffusion 1.5 has two stages: (i) train an RL teacher with adjoint matching, and (ii) distill its samples into a fresh student with standard flow / score matching.

RL teacher (adjoint matching). We use the official adjoint-matching codebase released by Domingo-Enrich et al. (2025) (https://github.com/microsoft/soc-fine-tuning-sd). Starting from Stable Diffusion 1.5, we run buffered adjoint matching with the codebase’s multi_prompt_buffer.yaml settings, and only deviate as noted below. Training is on an ∼8000-prompt subset of the 10000 prompts shipped in the codebase, themselves extracted from the ReFL training data of Xu et al. (2023), with ImageReward as the reward and reward multiplier 300 (vs 100 in the default config). Sampling uses 50-step memoryless DDIM (η = 1). Optimization uses Adam at learning rate 3×10−6, β = (0.9,0.95), fp32; per-rank batch 6 with 11 accumulation steps on 8 H200 GPUs (effective batch 528). Buffered training uses buffer_size=100 trajectories with passes_per_buffer=10. We train for 10 epochs over the prompt set and keep the checkpoint with the best validation ImageReward as the teacher. These are essentially the default settings provided in multi_prompt_buffer.yaml.

Distillation (SFT student). The student is a fresh Stable Diffusion 1.5 UNet trained with the standard DDPM ε-prediction loss on samples drawn from the RL teacher. We use the same prompt set used to train the RL teacher above. Teacher samples are generated without classifier-free guidance using 50-step DDIM with η = 0 on the standard SD 1.5 noise schedule, which is also the schedule the student trains on (βstart = 8.5 × 10−4, βend = 1.2 × 10−2, scaled-linear). Training is buffered: every cycle we generate a fresh batch of 4096 teacher images and take 10 passes over the buffer, with per-rank batch 32 and 2 accumulation steps on 8 H200 GPUs (effective batch 512). Optimization uses Adam at learning rate 10−5 with 100-step linear warmup, gradient clipping at 1.0, and bfloat16 mixed precision. We evaluate ImageReward on a held-out validation split every

1000 optimizer steps and early-stop when no improvement is observed for 5 consecutive eval rounds. The run plotted in figure 4 stopped after ∼9000 optimizer steps under this criterion—roughly 50× more (prompt, teacher-image) pairs seen by the gradient than the RL teacher saw during its own training. The picture is unchanged if we evaluate on training prompts rather than the held-out validation split.

We report and train at CFG=1 (no guidance) as the “clean” comparison: the RL teacher is never fine-tuned along the non-conditional branch (adjoint matching only works well without CFG). This also avoids issues given that the CFG distribution of the teacher would not be the CFG distribution of the student if we were to fine-tune the student on the CFG teacher samples. This is also the setting we discuss in the main text.

- I.5 Distillation from DRL Teachers

For the DRL distillation experiment, we use the same DRL model (λ=1, R1=0, linear-conditional discriminator on DINOv2-L) used throughout the main paper as the frozen teacher. At a high level, the setup mirrors the distillation experiment on Stable Diffusion 1.5 in section 3: we sample from the post-DRL model to fill a buffer, train the student on those samples with the standard flow matching velocity loss, and repeat. Unlike the Stable Diffusion 1.5 experiments, we do not apply early stopping to drive home the point that the gap is not solvable with more compute.

Hyperparameters. The student is initialized from the base REPA SiT-XL/2 pretrained checkpoint and trained with the standard flow matching velocity loss on teacher-generated samples (no REPA projection loss). The teacher generates samples with Heun-50. We use AdamW with lr=10−4 (constant schedule, 100-step warmup), no weight decay, gradient clipping at 1.0, and EMA decay of 0.9999. Training runs in full precision (float32), as we occasionally encountered instabilities with bf16 accumulation. The effective batch size is 576. Teacher-generated samples are stored in a buffer of 57,600 samples; the student trains on this buffer for 10 passes (1,000 optimizer steps) before the buffer is discarded and refilled with fresh teacher generations. Evaluation settings follow those described in the main paper. The results reported in the main text are at CFG=1 and with EMA. Non-EMA results are similar, albeit slightly worse. Additionally, these hyperparameters correspond almost exactly to the original REPA training configuration, with the only differences being a larger batch size and the absence of the REPA projection loss. We tried other learning rates and buffer-refresh schedules, but found similar or worse results.

Training samples. We train for 900k optimizer steps. Since the buffer is refreshed every 1,000 steps, this amounts to 900 buffer fills, each producing 57,600 fresh samples. The student therefore sees a total of ∼52M teacher samples over the course of training—over 150× the number of samples consumed during DRL post-training (∼346k) and ∼40× the size of ImageNet (∼1.28M images).

Compute. The total cost of the 900k-step run is approximately 4,050 GPU-hours on H200 GPUs. Buffer filling dominates: each fill requires a 50-step Heun solve for 57,600 samples and takes ∼400 seconds, so 900 fills account for ∼2,385 GPU-hours (∼59% of the budget). The remaining ∼1,710 GPU-hours cover the optimization steps.

- I.6 Feature-space ablation setup

We base our feature-space ablation on REPA SiT-XL/2. We train separate linear conditional discriminators on features from seven embedders—DINOv2 (base, large), DINOv3 (base, large), SigLIP (base, large), and Inception-v3—and use each as the reward for adjoint matching. Only the embedder varies across configs; all other discriminator and adjoint-matching settings follow the distributional alignment setup (section I.1). Every fine-tuned model is then evaluated across all seven feature spaces; full results are in table 5.

- I.7 Discriminator architecture and training ablation setup

The discriminator ablation on REPA SiT-XL/2 with λ = 10 follows the distributional alignment setup (section I.1) with the following differences and additions.

Discriminator. We explore three additional configurations on top of the default linear projection head on frozen DINOv2-Large: (i) MLP head—two GELU residual blocks of hidden dim 512 on top of the frozen DINOv2Large CLS token, followed by the same linear map used in the default; (ii) fine-tuning the embedder—unfreeze

DINOv2-Large and train it jointly with the linear head; (iii) training the embedder from scratch—initialize the same DINOv2-L architecture randomly and train it jointly with the linear head. In all three configurations we keep the class-conditional projection wrapper D(x,y) = φ(h(x)) + ⟨h(x),e(y)⟩ + b(y) from the alignment setup.

Training. Everything else matches the alignment setup, except for the embedder learning rate when the embedder is updated. We use 10−5 for fine-tuning—we initially tried 10−4 but training was too unstable and the resulting discriminators were noticeably worse. We nevertheless kept 10−4 for the from-scratch variant. At these rates we observed no training instabilities and the discriminator reached the same convergence behavior as the frozen-embedder default.

- J Extended Results

This section reports the extended quantitative results that complement the main-text experiments. Subsections are ordered to match the main-text experiment flow: alignment (distributional alignment), image quality (image-quality transfer), preference RL (preference alignment), and ablations (design choices). Setups for each experiment, including the ablation training configurations, are collected in section I.

A note on autoguidance. Standard classifier-free guidance (CFG) replaces the conditional velocity field v(x,y) with a combination of the conditional and unconditional fields, vcfg(x,y) = v(x,∅) + w (v(x,y) − v(x,∅)), where w is a positive scalar, usually greater than 1, and ∅ denotes the null class. Autoguidance (Karras et al., 2024) proposes replacing v(x,∅) with the velocity field of a weaker version of the same model, typically an earlier checkpoint from pretraining. In practice, it has been shown to improve sample quality and diversity.

To our knowledge, autoguidance has only been applied to pretrained models. As a curiosity, we explored applying it in the RL setting, guiding the RL fine-tuned conditional model with the corresponding non-RL conditional version,

vag(x,y) = vbase(x,y) + w (vRL(x,y) − vbase(x,y)). (43)

We tried this in two settings: using the non-DRL model to guide the DRL model, and, after preference alignment, using the non-preference-aligned model to guide the preference-aligned model. Generally, we found the results to be mixed. In the DRL setting, when λ was set to a high value (e.g., 40), a value of w lower than 1 usually improved performance over both the base and the DRL model. However, autoguidance generally underperformed standard CFG with an interval. In the preference-alignment setting, w > 1 generally led to the highest reward values across all models, but also to the worst image fidelity: with sufficiently large w, images became very bright and white, which is presumably why they scored so highly under the proxy reward despite being poorly aligned with true human preferences.

For these reasons, we do not feature autoguidance in the main text. Nevertheless, we provide Pareto fronts with autoguidance in figures 20 to 22 and the corresponding sample grids in sections K.2 and K.3, so that interested readers can see the effect in more detail and so that the observations may prove useful for future work on the topic.

Alignment. figures 17 and 18 show how all seven distributional metrics vary with λ at CFG = 1 (no guidance). For each (model, embedder, λ) combination, the plotted value corresponds to the interval achieving the lowest FD, with all other metrics taken from the same entry. The dashed gray line marks the base model value, and the dotted line (in the model’s color) marks the theoretically motivated λ=1 (R1=0) checkpoint. tables 2 and 3 extend the Fréchet Distance comparison from figure 5 to the full set of distributional metrics: FD, FDval, KD, Precision, Recall, Density, and Coverage. For each (model, embedder) pair, we select the DRL configuration (λ, interval) that achieves the lowest FD and report all seven metrics from that same generation run, along with the chosen configuration. This ensures internal consistency across all numbers in each row.

The takeaways are largely unchanged from the main text: DRL consistently improves over the base model across the board. Two patterns become clearer when seeing the full sweep of metrics. First, λ ∈ {5,10} generally performs best across most metrics, with the exception of FID, which tends to prefer smaller λ. Second, as discussed in the main paper, larger λ generally brings improvements over the theoretically motivated baseline. Additionally, λ=40 is never selected as the best configuration and λ=20 is selected only sparingly, which further validates our choice of λ=10 as the default for the reward-improvement and preference-RL experiments.

Finally, while we select the best configuration according to its FD value (computed against the ImageNet training distribution), the improvements persist on FDval and KD, neither of which we select for. Together, these suggest genuine distributional alignment rather than overfitting to the optimized signal.

Image quality. Figure 19 extends the reward-improvement comparison from figure 7, which fixes λ=10, to the full λ sweep. Each panel plots normalized reward improvement (DRL−Base)/σbase against λ for one held-out reward. The dotted horizontal line per model marks the theoretically motivated λ=1 (R1=0) checkpoint for reference. Improvements are largely monotone in λ up to λ=20, with the strongest gains in ImageReward, HPSv2, and PickScore as reported in the main text. For λ=40, however, improvements often dip—consistent

with the alignment results in figures 17 and 18, where λ=40 is rarely the best configuration. Table 4 reports the raw reward values underlying figure 7: λ=10.

Preference RL: Pareto plots. figure 9 in the main text shows HPSv2 vs. low-level image statistics at CFG = 1 (no guidance). figures 20 to 22 extend this to a full 3 × 3 grid: three held-out rewards (HPSv2, ImageReward, Aesthetic v2.5) × three guidance settings (no CFG, CFG = 2, autoguidance = 2; see section J for context on autoguidance). We do not sweep over the guidance value because the results are easier to read this way; the plots look essentially identical for other CFG values.

The results are essentially unchanged from the main text: the patterns we showed for HPSv2 also hold for ImageReward and Aesthetic v2.5, with DRL achieving the best Pareto fronts in nearly every plot. Moreover, this conclusion is robust across the three guidance settings. Furthermore, as discussed in section J, autoguidance pushes reward values higher at the cost of substantially worse image-statistic distortion—for example, on REPA the maximum brightness rises from ∼0.6 under base+CFG to ∼0.7 under autoguidance, an effect also visible in the sample grids of section K.3, where the images are noticeably brighter. This further supports our claim that reward proxies are usually exploitable and negatively entangled with image fidelity. Even so, DRL retains the best Pareto front in the autoguidance setting.

Ablations. Figure 23 reports DINOv2-L FD across the full R1 sweep at both λ=1 and λ=10, complementing the path-based view in Figure 11. The two regimes show qualitatively different behavior: at λ=1 all R1>0 values hurt — the cleanest baseline is at R1=0 — while at λ=10 a small R1 rescues frozen-feature heads from collapse. This supports the claim in design choices that R1 here functions as a stabilizer for aggressive λ rather than as a vanishing-gradient remedy as in the standard GAN literature (Mescheder et al., 2018). Table 5 reports the full 7×7 feature-space ablation summarized by table 1 in the main text: FD and KD when training a discriminator on each of seven embedders (columns) and evaluating on each of seven embedders (rows). The training setup is identical across embedders (section I.6); only the embedder varies. We see that even with the additional embedders the conclusions are unchanged.

CFG effect. Finally, to provide additional insight into how our method interacts with CFG, Figure 24 and Figure 25 break down the per-reward and per-embedder FD performance of DRL as a function of the CFG scale. The reward plot is constructed by running CFG and using the same setup for estimating rewards described earlier. The FD plot is constructed by taking the minimum over the three CFG interval values described in section I at each CFG scale. We didn’t notice meaningful differences across the three intervals for the rewards. Several observations are in order.

For the reward values, we see that DRL has a strong positive effect that is maintained at all CFG levels. For some (model, reward) pairs (e.g., SiT on ImageReward) DRL with no CFG already attains higher reward than the base model at any CFG value, and the trend is consistent across model classes. The only exception is RAE, where a small λ slightly decreases the reward; this disappears at larger λ.

For FD the picture is more mixed, with two evident patterns. First, on Inception and SigLIP smaller λ generally yields better performance, while on DINOv2 and DINOv3 larger λ is generally more beneficial. Second, CFG becomes much less effective the larger λ gets. We believe these are artifacts of the current training setup, in which we only finetune the conditional branch (a constraint imposed by adjoint matching). Finding ways to make the model less dependent on λ and trainable jointly with both the CFG and non-CFG branches is an interesting direction for future work.

- J.1 Alignment: Quantitative λ Sweep

###### RAE DiTDH-XL Base DRL (λ=1, R1=0) DRL (λ sweep, R1=10−5)

FD ↓

FDval ↓

###### KD ×1000 ↓

Precision ↑

Recall ↑

###### Density ↑

Coverage ↑

30

.76

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

.88

DINOv2-L

- .7
- .8
- .9

.78

75

80

.84

.75

20

.72

50

60

.8

.72

10

.68

40

25

.76

.88

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

DINOv3-L

- 3
- 4
- 5

- .8

.85

- .9

.78

12

.68

10

.84

.76

7.5

9

.64

.8

.74

5

.6

6

.76

27

.92

15

.848

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

SigLIP-L

1.2

16

.52

.84

24

12.5

.88

14

1.15

.832

.48

21

12

10

.84

1.1

.824

.44

6

.96

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

- 6
- 7.5

- 1
- 2
- 3

.64

.775

Inception

1.2

4

.92

.6

.75

1.14

- 3
- 4.5

.56

2

.88

.725

1.08

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

λ

λ

λ

λ

λ

λ

λ

SiT-XL/2 Base DRL (λ=1, R1=0) DRL (λ sweep, R1=10−5)

FD ↓

FDval ↓

###### KD ×1000 ↓

###### Precision ↑

###### Recall ↑

Density ↑

Coverage ↑

1

240

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

DINOv2-L

240

- .4
- .5
- .6

.75

240

- .5
- .6
- .7

.75

160

160

160

.6

80

.5

80

80

.45

90

.35

120

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

.75

DINOv3-L

1

75

- .5
- .6
- .7

.3

80

60

- .8
- .9

.6

50

.25

40

30

25

.45

.84

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

.84

1.2

- .2
- .3
- .4

40

40

SigLIP-L

90

.8

.78

1

30

30

.76

60

.72

20

.8

.72

20

.66

30

1.5

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

18

7.5

15

Inception

- .6
- .7
- .8

.6

.9

1.2

5

12

10

.45

.84

2.5

.9

5

6

.3

.78

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

λ

λ

λ

λ

λ

λ

λ

###### Figure 17 Quantitative lambda sweep (RAE + SiT). Each subplot shows one metric for one embedder. Dashed gray: base model at CFG = 1. Dotted (model color): theoretically motivated λ=1 (R1=0) checkpoint. Solid line with markers: DRL values at λ ∈ {1, 5, 10, 20, 40} from the R1=10−5 sweep. Rows: embedders (DINOv2-L, DINOv3-L, SigLIP-L, Inception). Columns: metrics (FD↓, FDval ↓, KD↓, Precision↑, Recall↑, Density↑, Coverage↑).

###### JiT-H/16 Base DRL (λ=1, R1=0) DRL (λ sweep, R1=10−5)

FD ↓

FDval ↓

###### KD ×1000 ↓

Precision ↑

Recall ↑

Density ↑

###### Coverage ↑

160

| |
|---|
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |

150

DINOv2-L

- .6
- .7
- .8

.64

.66

.75

120

120

100

.6

.6

.6

80

80

50

.54

40

.45

.56

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

.75

DINOv3-L

.8

.475

32

.85

32

30

.7

24

.45

.8

24

.72

20

.65

16

16

.75

.425

10

.64

40

100

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

.87

40

SigLIP-L

.84

1.2

.3

.84

80

32

1.1

32

.81

.28

.81

60

1

24

.78

.26

24

.78

.78

- 6
- 7.5

| |
|---|
| |
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

- 1
- 2
- 3
- 4

1.2

.64

Inception

.93

8

.72

1

.56

6

.9

.66

- 3
- 4.5

.8

.48

4

.87

.6

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

λ

λ

λ

λ

λ

λ

λ

###### REPA SiT-XL/2 Base DRL (λ=1, R1=0) DRL (λ sweep, R1=10−5)

FD ↓

FDval ↓

###### KD ×1000 ↓

###### Precision ↑

###### Recall ↑

Density ↑

###### Coverage ↑

- .6
- .7
- .8

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |

DINOv2-L

150

1

150

- .6
- .7
- .8

- .4
- .5
- .6

150

100

.75

100

100

50

50

.5

50

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

75

DINOv3-L

60

- .3

.35

- .4

- .6
- .7
- .8

60

1.1

- .6
- .7
- .8

50

1

40

40

25

.9

20

20

.48

1.2

.88

.84

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |
| |

75

32

SigLIP-L

32

.4

.8

1.05

.8

24

50

24

.32

.76

.9

.72

16

16

25

.24

18

- .4
- .5
- .6
- .7

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

15

- .6
- .7
- .8

Inception

1.25

6

.9

12

10

1

3

.84

5

6

.75

.78

0

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

1 5 102040

λ

λ

λ

λ

λ

λ

λ

- Figure 18 Quantitative lambda sweep (JiT + REPA). Same layout as figure 17.

- J.2 Alignment: Full Distribution Metrics (PRDC)

- Table 2 Distribution metrics — No CFG. For each (model, embedder) pair, all rows are evaluated at cfg = 1; the DRL (λ∗) row uses the (λ, time interval) with the lowest FD. All seven metrics come from a single sweep entry. For each

(model, embedder), three rows compare Base, DRL at the theoretically motivated λ=1 (R1=0), and DRL with λ tuned over {1, 5, 10, 20, 40}. Bold marks the best value per metric within each block. FD is computed against the ImageNet training set; FDval against the validation set. KD is reported as value ± standard error of the subset-mean estimator (100 random subsets). The right-most column shows the (λ, R1) of the run each row is drawn from.

Model Embedder Method FD↓ FDval ↓ KD↓ Prec↑ Rec↑ Dens↑ Cov↑ (λ, R1)

Base 37.47 49.27 29.75 ± 0.33 0.669 0.787 0.692 0.832 (−−, −−) DRL (λ=1, R1=0) 30.17 42.47 22.00 ± 0.29 0.689 0.781 0.737 0.849 (1, 0) DRL (λ∗) 20.60 33.88 9.31 ± 0.20 0.734 0.759 0.849 0.882 (10, 10−5)

DINOv2-L

Base 6.66 8.04 5.20 ± 0.03 0.731 0.697 0.802 0.849 (−−, −−) DRL (λ=1, R1=0) 5.71 7.14 4.21 ± 0.03 0.742 0.698 0.821 0.859 (1, 0) DRL (λ∗) 4.49 6.10 2.75 ± 0.02 0.769 0.684 0.886 0.877 (10, 10−5)

DINOv3-L

RAE

Base 10.19 11.88 20.64 ± 0.13 0.822 0.541 1.118 0.910 (−−, −−) DRL (λ=1, R1=0) 9.53 11.31 19.30 ± 0.15 0.830 0.535 1.143 0.915 (1, 0) DRL (λ∗) 9.13 11.15 19.68 ± 0.17 0.844 0.518 1.200 0.919 (10, 10−5)

SigLIP-L

Base 1.31 2.39 0.43 ± 0.02 0.719 0.643 1.066 0.956 (−−, −−) DRL (λ=1, R1=0) 1.38 2.72 0.82 ± 0.03 0.736 0.632 1.114 0.957 (1, 0) DRL (λ∗) 1.29 2.52 0.61 ± 0.03 0.728 0.640 1.092 0.957 (1, 10−5)

Incep.

Base 241.93 249.99 276.55 ± 1.28 0.469 0.621 0.399 0.451 (−−, −−) DRL (λ=1, R1=0) 95.51 108.56 81.43 ± 0.51 0.564 0.609 0.551 0.659 (1, 0) DRL (λ∗) 54.37 70.97 22.62 ± 0.18 0.733 0.510 0.920 0.782 (10, 10−5)

DINOv2-L

Base 88.22 88.53 117.44 ± 0.51 0.470 0.314 0.803 0.453 (−−, −−) DRL (λ=1, R1=0) 43.76 44.88 46.23 ± 0.23 0.581 0.347 0.754 0.647 (1, 0) DRL (λ∗) 19.25 21.19 10.90 ± 0.05 0.781 0.278 1.019 0.749 (20, 10−5)

DINOv3-L

SiT

Base 42.59 42.91 108.37 ± 0.52 0.712 0.395 0.761 0.700 (−−, −−) DRL (λ=1, R1=0) 20.17 22.11 40.37 ± 0.24 0.776 0.417 0.977 0.823 (1, 0) DRL (λ∗) 16.90 18.94 30.20 ± 0.21 0.817 0.382 1.128 0.843 (5, 10−5)

SigLIP-L

Base 9.38 8.13 5.27 ± 0.08 0.586 0.675 0.719 0.843 (−−, −−)

Incep.

DRL (λ=1, R1=0) 2.62 3.63 0.85 ± 0.02 0.713 0.597 1.068 0.937 (1, 0) DRL (λ∗) 2.62 3.63 0.85 ± 0.02 0.713 0.597 1.068 0.937 (1, 0)

Base 148.03 158.85 153.22 ± 0.71 0.515 0.648 0.463 0.596 (−−, −−) DRL (λ=1, R1=0) 97.33 109.00 89.20 ± 0.52 0.568 0.659 0.545 0.690 (1, 0) DRL (λ∗) 42.84 57.29 21.31 ± 0.20 0.709 0.622 0.847 0.825 (20, 10−5)

DINOv2-L

Base 33.89 35.11 33.84 ± 0.13 0.620 0.428 0.749 0.636 (−−, −−) DRL (λ=1, R1=0) 23.66 25.08 21.87 ± 0.09 0.651 0.464 0.735 0.711 (1, 0) DRL (λ∗) 11.32 13.24 8.13 ± 0.03 0.754 0.470 0.862 0.810 (20, 10−5)

DINOv3-L

JiT

Base 39.41 40.94 100.22 ± 0.38 0.772 0.297 0.921 0.779 (−−, −−) DRL (λ=1, R1=0) 29.72 31.36 71.41 ± 0.28 0.802 0.315 1.030 0.833 (1, 0) DRL (λ∗) 20.99 23.04 45.97 ± 0.20 0.856 0.302 1.270 0.875 (20, 10−5)

SigLIP-L

Base 7.16 6.47 2.81 ± 0.05 0.588 0.664 0.714 0.860 (−−, −−) DRL (λ=1, R1=0) 3.72 3.73 0.79 ± 0.02 0.652 0.645 0.871 0.905 (1, 0) DRL (λ∗) 2.73 3.28 0.61 ± 0.01 0.699 0.616 1.001 0.930 (5, 10−5)

Incep.

Base 159.46 168.91 163.05 ± 0.95 0.517 0.668 0.465 0.606 (−−, −−) DRL (λ=1, R1=0) 58.33 72.52 45.83 ± 0.33 0.644 0.653 0.693 0.773 (1, 0) DRL (λ∗) 40.33 56.12 23.77 ± 0.22 0.721 0.615 0.877 0.829 (5, 10−5)

DINOv2-L

Base 63.87 64.46 77.62 ± 0.38 0.540 0.385 0.848 0.597 (−−, −−) DRL (λ=1, R1=0) 29.43 30.81 28.16 ± 0.16 0.660 0.412 0.855 0.763 (1, 0) DRL (λ∗) 15.58 17.37 10.88 ± 0.06 0.791 0.386 1.066 0.827 (10, 10−5)

DINOv3-L

REPA

Base 31.56 32.13 79.18 ± 0.45 0.729 0.456 0.809 0.769 (−−, −−) DRL (λ=1, R1=0) 14.34 16.56 28.71 ± 0.21 0.795 0.470 1.033 0.864 (1, 0) DRL (λ∗) 12.77 14.88 23.97 ± 0.21 0.823 0.449 1.138 0.874 (5, 10−5)

SigLIP-L

Base 6.48 5.81 2.94 ± 0.06 0.600 0.687 0.750 0.872 (−−, −−)

Incep.

DRL (λ=1, R1=0) 2.14 3.66 1.16 ± 0.03 0.719 0.611 1.072 0.943 (1, 0) DRL (λ∗) 2.14 3.66 1.16 ± 0.03 0.719 0.611 1.072 0.943 (1, 0)

- Table 3 Distribution metrics — Best CFG. For each (model, embedder) pair and each method row, the (CFG scale, time interval) combination minimising FD is selected; all seven metrics are taken from that single run. For each

(model, embedder), three rows compare Base, DRL at the theoretically motivated λ=1 (R1=0), and DRL with λ tuned over {1, 5, 10, 20, 40}. Bold marks the best value per metric within each block. FD is computed against the ImageNet training set; FDval against the validation set. KD is reported as value ± standard error of the subset-mean estimator (100 random subsets). The right-most column shows the configuration (λ, R1, cfg, [interval]) of the run each row is drawn from.

Model Embedder Method FD↓ FDval ↓ KD↓ Prec↑ Rec↑ Dens↑ Cov↑ (λ, R1, cfg, [int.])

Base 25.94 40.78 16.59 ± 0.22 0.764 0.677 0.924 0.887 (−−, −−, 1.5, [0, 1]) DRL (λ=1, R1=0) 23.66 37.74 15.22 ± 0.22 0.743 0.723 0.870 0.884 (1, 0, 1.25, [0, 1]) DRL (λ∗) 20.24 34.68 10.51 ± 0.18 0.762 0.719 0.915 0.897 (5, 10−5, 1.25, [0, 1])

DINOv2-L

Base 4.83 6.66 3.37 ± 0.02 0.810 0.608 0.992 0.895 (−−, −−, 1.5, [0, 1]) DRL (λ=1, R1=0) 4.57 6.44 2.91 ± 0.02 0.820 0.605 1.013 0.895 (1, 0, 1.5, [0, 1]) DRL (λ∗) 4.26 5.97 2.71 ± 0.02 0.800 0.651 0.954 0.895 (5, 10−5, 1.25, [0, 1])

DINOv3-L

RAE

Base 9.50 11.81 22.86 ± 0.20 0.855 0.494 1.252 0.925 (−−, −−, 1.25, [0, 1]) DRL (λ=1, R1=0) 9.40 11.77 23.36 ± 0.23 0.860 0.488 1.280 0.927 (1, 0, 1.25, [0, 1]) DRL (λ∗) 9.06 11.03 19.18 ± 0.17 0.839 0.524 1.185 0.917 (10, 10−5, 1.75, [.1, 1])

SigLIP-L

Base 1.28 2.32 0.35 ± 0.02 0.714 0.651 1.043 0.953 (−−, −−, 2, [.1, 1]) DRL (λ=1, R1=0) 1.25 2.43 0.49 ± 0.02 0.716 0.645 1.052 0.953 (1, 0, 2.5, [.1, 1]) DRL (λ∗) 1.19 2.31 0.38 ± 0.02 0.714 0.652 1.042 0.952 (1, 10−5, 2.25, [.1, 1])

Incep.

Base 46.96 62.68 28.17 ± 0.25 0.801 0.436 1.171 0.862 (−−, −−, 2.75, [.1, 1])

DINOv2-L

DRL (λ=1, R1=0) 42.36 58.94 23.46 ± 0.25 0.796 0.465 1.148 0.860 (1, 0, 2.75, [.3, 1]) DRL (λ∗) 42.36 58.94 23.46 ± 0.25 0.796 0.465 1.148 0.860 (1, 0, 2.75, [.3, 1])

Base 21.63 24.10 16.70 ± 0.08 0.835 0.156 1.337 0.800 (−−, −−, 3.5, [0, 1]) DRL (λ=1, R1=0) 18.32 20.61 15.16 ± 0.08 0.837 0.220 1.303 0.840 (1, 0, 3, [.1, 1]) DRL (λ∗) 16.40 18.54 10.99 ± 0.05 0.851 0.234 1.354 0.831 (10, 10−5, 2.25, [.3, 1])

DINOv3-L

SiT

Base 11.81 13.48 25.30 ± 0.21 0.842 0.400 1.241 0.898 (−−, −−, 3.5, [.3, 1]) DRL (λ=1, R1=0) 12.17 14.95 27.46 ± 0.29 0.842 0.405 1.238 0.887 (1, 0, 2.25, [.3, 1]) DRL (λ∗) 11.58 13.64 24.76 ± 0.26 0.845 0.401 1.253 0.896 (1, 10−5, 2.75, [.3, 1])

SigLIP-L

Base 1.50 2.90 0.70 ± 0.02 0.713 0.626 1.065 0.941 (−−, −−, 2, [.3, 1]) DRL (λ=1, R1=0) 2.43 4.17 1.77 ± 0.04 0.742 0.582 1.169 0.947 (1, 0, 1.25, [.3, 1]) DRL (λ∗) 1.78 3.26 1.08 ± 0.03 0.734 0.610 1.136 0.946 (1, 10−5, 1.5, [.3, 1])

Incep.

Base 41.55 54.96 31.69 ± 0.25 0.737 0.576 0.896 0.851 (−−, −−, 4, [.1, 1]) DRL (λ=1, R1=0) 36.10 49.96 24.87 ± 0.23 0.753 0.585 0.943 0.867 (1, 0, 3.5, [.1, 1]) DRL (λ∗) 30.64 45.59 15.39 ± 0.17 0.788 0.574 1.054 0.884 (10, 10−5, 2.75, [.1, 1])

DINOv2-L

Base 10.89 13.16 8.56 ± 0.04 0.813 0.376 1.038 0.841 (−−, −−, 2.75, [0, 1]) DRL (λ=1, R1=0) 10.19 12.43 7.74 ± 0.03 0.816 0.397 1.039 0.847 (1, 0, 2.25, [0, 1]) DRL (λ∗) 8.10 10.14 5.94 ± 0.03 0.818 0.456 1.029 0.862 (20, 10−5, 3.25, [.1, 1])

DINOv3-L

JiT

Base 18.21 19.93 46.80 ± 0.19 0.837 0.348 1.168 0.888 (−−, −−, 3.5, [.1, 1]) DRL (λ=1, R1=0) 17.11 18.94 42.95 ± 0.20 0.846 0.348 1.206 0.896 (1, 0, 3, [.1, 1]) DRL (λ∗) 16.47 18.43 42.03 ± 0.21 0.854 0.339 1.253 0.899 (5, 10−5, 3, [.1, 1])

SigLIP-L

Base 1.91 3.16 0.74 ± 0.02 0.684 0.629 0.950 0.928 (−−, −−, 2.25, [.1, 1])

Incep.

DRL (λ=1, R1=0) 1.87 3.05 0.74 ± 0.02 0.698 0.627 0.976 0.932 (1, 0, 2.5, [.3, 1]) DRL (λ∗) 1.87 3.05 0.74 ± 0.02 0.698 0.627 0.976 0.932 (1, 0, 2.5, [.3, 1])

Base 36.87 52.75 22.95 ± 0.22 0.809 0.511 1.111 0.884 (−−, −−, 2.75, [.1, 1])

DINOv2-L

DRL (λ=1, R1=0) 32.87 49.11 20.44 ± 0.21 0.797 0.567 1.058 0.882 (1, 0, 2.75, [.3, 1]) DRL (λ∗) 32.87 49.11 20.44 ± 0.21 0.797 0.567 1.058 0.882 (1, 0, 2.75, [.3, 1])

Base 18.60 20.97 14.02 ± 0.08 0.831 0.238 1.233 0.818 (−−, −−, 3.25, [0, 1]) DRL (λ=1, R1=0) 14.58 16.80 11.71 ± 0.06 0.841 0.324 1.235 0.866 (1, 0, 2.75, [.1, 1]) DRL (λ∗) 13.01 15.00 9.02 ± 0.05 0.852 0.354 1.243 0.862 (10, 10−5, 2.25, [.3, 1])

DINOv3-L

REPA

Base 9.52 11.18 20.65 ± 0.20 0.830 0.488 1.141 0.910 (−−, −−, 3.75, [.3, 1]) DRL (λ=1, R1=0) 10.02 12.77 23.03 ± 0.27 0.832 0.475 1.172 0.899 (1, 0, 2.25, [.3, 1]) DRL (λ∗) 9.32 11.34 20.71 ± 0.23 0.833 0.482 1.165 0.907 (1, 10−5, 3, [.3, 1])

SigLIP-L

Base 1.21 2.77 0.71 ± 0.03 0.702 0.642 1.004 0.939 (−−, −−, 2.25, [.3, 1]) DRL (λ=1, R1=0) 2.14 3.66 1.16 ± 0.03 0.719 0.611 1.072 0.943 (1, 0, 1, [0, 1]) DRL (λ∗) 1.50 3.05 0.94 ± 0.03 0.718 0.621 1.067 0.945 (1, 10−5, 1.5, [.3, 1])

Incep.

- J.3 Image Quality: Reward Improvement vs. λ

- Figure 19 Reward improvement vs. λ. Per-model normalized improvement on each of four held-out preference rewards. Solid lines with markers: DRL at λ ∈ {1, 5, 10, 20, 40} from the R1=10−5 sweep. Dotted lines (in the model’s color): theoretically motivated λ=1 (R1=0) checkpoint. Dashed gray line at zero: Base model. Top row: CFG = 1 (no guidance). Bottom row: best CFG. Same units as figure 7.

Base DRL (λ=1, R1=0) DRL (λ sweep, R1=10−5) SiT REPA JiT RAE

###### ImageReward

HPSv2

Aesthetic

PickScore

1.25

1.25

1.25

1.25

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |
| |

−(DRLBase)/σbase

1.00

1.00

1.00

1.00

NoCFG

0.75

0.75

0.75

0.75

0.50

0.50

0.50

0.50

0.25

0.25

0.25

0.25

0.00

0.00

0.00

0.00

0.4

0.4

0.4

0.4

| |
|---|
| |
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

−(DRLBase)/σbase

0.3

0.3

0.3

0.3

BestCFG

0.2

0.2

0.2

0.2

0.1

0.1

0.1

0.1

0.0

0.0

0.0

0.0

−0.1

−0.1

−0.1

−0.1

1 5 10 20 40

1 5 10 20 40

1 5 10 20 40

1 5 10 20 40

λ

λ

λ

λ

- Table 4 Numerical values for Figure 7. DRL is pinned to the headline λ=10; the guidance scale is tuned per (model, metric) cell. Bold indicates the better value within each Base/DRL pair.

No guidance Guidance Model Metric Base DRL Base DRL

ImageReward↑ -1.252 -0.486 -0.584 -0.309 HPSv2↑ 0.197 0.224 0.233 0.242 Aesthetic↑ 3.149 3.362 3.473 3.533 PickScore↑ 19.140 20.086 20.153 20.383

SiT-XL/2

ImageReward↑ -1.187 -0.562 -0.622 -0.415 HPSv2↑ 0.200 0.223 0.230 0.237 Aesthetic↑ 3.185 3.386 3.456 3.508 PickScore↑ 19.240 20.044 20.094 20.292

REPA SiT-XL/2

ImageReward↑ -1.159 -0.804 -0.723 -0.551 HPSv2↑ 0.196 0.210 0.226 0.231 Aesthetic↑ 3.078 3.192 3.321 3.366 PickScore↑ 19.239 19.710 19.904 20.079

JiT-H/16

ImageReward↑ -0.933 -0.764 -0.583 -0.488 HPSv2↑ 0.217 0.222 0.227 0.230 Aesthetic↑ 3.292 3.338 3.398 3.425 PickScore↑ 19.784 19.990 20.187 20.297

RAE DiTDH-XL

- J.4 Image Quality / Preference RL: Pareto Plots

DRL+PRL (ours)

ImageNet

λPRL=10 λPRL=40

λPRL=1

Base+PRL

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

0.25

###### SiT

0.20

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

0.25

###### REPA

noCFGCFG=2AG=2

0.20

0.22

JiT

0.20

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.24

###### RAE

0.22

0.5 0.6

0.2 0.3

0.18 0.20 0.22 0.14 0.16

0.4 0.5

0.26

###### SiT

0.24

0.26

###### REPA

0.24

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.24

JiT

0.22

0.25

###### RAE

0.24

0.23

0.5 0.6

0.25 0.30 0.35 0.200 0.225 0.16 0.18 0.20 0.4 0.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.25

SiT

0.20

0.250

###### REPA

0.225

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.225

JiT

0.200

0.250

###### RAE

0.225

0.5 0.6 0.7

0.2 0.3

0.15 0.20

0.12 0.14 0.16

0.4 0.6

Brightness

Saturation

Contrast

Colorfulness

Whiteness

- Figure 20 HPSv2 vs. image statistics across guidance settings. Each panel-group is a 4 (model) × 5 (image statistic) grid; rows correspond to no CFG (CFG=1), CFG=2, and autoguidance=2 (see section J for context on autoguidance). Markers: base (⋆), Base+PRL (dashed, hollow), DRL+PRL (solid, filled) at λ ∈ {1, 10, 40}. Vertical dashed green line: ImageNet reference value per statistic. The autoguidance group uses each method’s pre-PRL checkpoint as the negative signal at scale 2 (Base for Base+PRL, DRL for DRL+PRL); the base reference still uses standard CFG at the same scale. DRL+PRL consistently attains better Pareto fronts across guidance settings.

0

SiT

−1

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

###### REPA

0

noCFGCFG=2AG=2

−1

0

###### JiT

−1

0.0

###### RAE

−0.5

0.5 0.6

0.2 0.3 0.18 0.20 0.22 0.14 0.16 0.4 0.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.5

SiT

0.0

−0.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.5

###### REPA

0.0

−0.5

0.0

JiT

−0.5

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

###### RAE

0.0

−0.5

0.5 0.6

0.25 0.30 0.35

0.200 0.225 0.16 0.18 0.20 0.4 0.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

SiT

0

−1

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

###### REPA

0

−1

0

JiT

−1

0.5

###### RAE

0.0

−0.5

0.5 0.6 0.7

0.2 0.3

0.15 0.20

0.12 0.14 0.16

0.4 0.6

Brightness

Saturation

Contrast

Colorfulness

Whiteness

###### Figure 21 ImageReward vs. image statistics across guidance settings. Mirrors figure 20 using ImageReward as the held-out reward. DRL+PRL consistently attains better Pareto fronts across guidance settings.

3.50

SiT

3.25

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

###### REPA

3.50

noCFGCFG=2AG=2

3.25

3.4

JiT

3.2

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

3.6

###### RAE

3.4

0.5 0.6

0.2 0.3

0.18 0.20 0.22 0.14 0.16 0.4 0.5

3.75

###### SiT

3.50

3.75

###### REPA

3.50

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

3.5

JiT

3.4

3.3

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

3.6

###### RAE

3.4

0.5 0.6

0.25 0.30 0.35

0.200 0.225

0.16 0.18 0.20

0.4 0.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

3.75

SiT

3.50

3.25

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

3.75

###### REPA

3.50

3.25

3.50

JiT

3.25

3.75

###### RAE

3.50

0.5 0.6 0.7

0.2 0.3

0.15 0.20

0.12 0.14 0.16

0.4 0.6

Brightness

Saturation

Contrast

Colorfulness

Whiteness

###### Figure 22 Aesthetic v2.5 vs. image statistics across guidance settings. Mirrors figure 20 using Aesthetic v2.5 as the held-out reward. DRL+PRL consistently attains better Pareto fronts across guidance settings.

- J.5 Ablations: Discriminator Full R1 Sweep

- Figure 23 Discriminator ablation: full R1 sweep. DINOv2-L FD vs. R1 for four head/training configurations, at λ=1 (left) and λ=10 (right). Top: fixed-seed samples at four representative configurations. Dashed gray line: base model FD.

|[Figure 48]|
|---|

Base

|[Figure 49]|
|---|

λ = 1 R1 = 0

|[Figure 50]|
|---|

λ = 10 R1 = 0

|[Figure 51]|
|---|

λ = 10 R1 = 10−5

Linear MLP-2 Finetune Scratch

|λ = 1| | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 10−5 10−3 0.1

R1

- 102
- 103

FD

|λ = 10| | | | |
|---|---|---|---|---|
|base| | | | |
| | | | | |
| | | | | |
| | | | | |

0 10−5 10−3 0.1

R1

- J.6 Ablations: Full Feature-Space Sweep

- Table 5 Feature space ablation results. Fréchet Distance (FD, ↓) and Kernel Distance (KD×103, ↓) when training the discriminator in different feature spaces (columns) and evaluating in different feature spaces (rows). Best fine-tuned result per row in bold. Training a discriminator on a particular embedding consistently improves downstream alignment metrics based on that embedder; DINOv2-L gives the best cross-feature transfer.

Training feature space Eval. space Base DINOv2-B DINOv2-L DINOv3-B DINOv3-L SigLIP-B SigLIP-L Incep.

Inception 6.43 2.18 2.14 2.19 3.02 2.57 2.34 5.68 DINOv2-B 130 50.8 44.3 54.3 47.3 72.9 64.5 123 DINOv2-L 159 71.0 58.3 74.7 63.2 96.1 87.1 151 DINOv3-B 22.4 10.8 9.81 10.7 9.53 14.0 12.4 21.5 DINOv3-L 63.7 31.1 29.4 31.0 18.5 44.0 38.8 62.0 SigLIP-B 7.54 3.85 3.41 3.80 4.28 4.11 3.88 7.19 SigLIP-L 31.2 16.1 14.3 16.0 16.5 18.0 16.1 29.9

FD

Inception 2.89 0.66 1.16 0.62 1.69 0.38 0.36 2.30 DINOv2-B 192 57.0 47.5 63.8 52.9 94.5 80.3 181 DINOv2-L 165 60.5 45.8 64.7 53.1 89.1 78.9 156 DINOv3-B 36.3 16.3 14.5 15.7 13.9 21.9 19.1 34.9 DINOv3-L 78.3 30.0 28.2 29.2 14.3 49.0 41.1 75.8 SigLIP-B 18.5 8.42 6.84 8.47 10.8 8.90 8.28 17.5 SigLIP-L 78.8 34.4 28.7 33.9 38.7 38.5 33.2 74.8

KD3×10

- J.7 Effect of CFG on Reward Scores and Fréchet Distance

Base DRL λ = 1 DRL λ = 5 DRL λ = 10 DRL λ = 20 DRL λ=1 (R1=0)

HPSv2 ↑

Aesthetic ↑

ImageReward ↑

PickScore ↑

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

3.50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

−0.5

| |
|---|

| |
|---|

20.0

| |
|---|

| |
|---|

0.225

SiT

| |
|---|

19.5

3.25

−1.0

0.200

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

−0.5

| |
|---|

| |
|---|

| |
|---|

###### REPA

| |
|---|

| |
|---|

| |
|---|

20.0

| |
|---|

3.4

0.225

19.5

−1.0

3.2

0.200

−0.5

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

20.0

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.225

| |
|---|

3.3

| |
|---|

| |
|---|

| |
|---|

| |
|---|

JiT

| |
|---|

| |
|---|

| |
|---|

| |
|---|

3.2

| |
|---|

19.5

−1.0

0.200

3.1

0.23

−0.50

| |
|---|

| |
|---|

| |
|---|

20.25

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

3.4

###### RAE

| |
|---|

| |
|---|

| |
|---|

| |
|---|

−0.75

20.00

| |
|---|

| |
|---|

0.22

3.3

2 4

2 4

2 4

2 4

CFG scale

CFG scale

CFG scale

CFG scale

- Figure 24 Effect of CFG on reward scores. Reward vs. CFG scale for four models (rows) and four reward metrics (columns). Black: base model (drawn on top of the DRL curves for visibility); colored lines: DRL at λ ∈ {1, 5, 10, 20} (darker = higher λ); purple dashed: DRL λ=1 with R1=0 discriminator. DRL beats base at every CFG, and sometimes (e.g., SiT on ImageReward) even without CFG.

3

10

SiT

FD Inception ↓

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

80

160

240

FD DINOv2 ↓

| |
|---|
| |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

30

60

90

FD DINOv3 ↓

| |
|---|
| |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

15

30

45

FD SigLIP ↓

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

4

8

REPA

| |
|---|

50

100

150

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

20

40

60

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10

20

30

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

2

4

6

JiT

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

100

150

| |
|---|
| |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10

20

30

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

20

30

40

| |
|---|

| |
|---|
| |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

2 4

CFG scale

- 2

- 3

RAE

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

2 4

CFG scale

20

30

40

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

2 4

CFG scale

4.5

- 6

- 7.5

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

2 4

CFG scale

- 9

- 10

- 11

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Base DRL λ = 1 DRL λ = 5 DRL λ = 10 DRL λ = 20 DRL λ=1 (R1=0)

- Figure 25 Effect of CFG on Fréchet Distance. FD vs. CFG scale for four models (rows) and four evaluation embedders (columns). All panels use a log y-axis. Each point shows the best result across the three CFG intervals defined in section I. Same legend as figure 24. Inception/SigLIP prefer smaller λ, DINOv2/v3 prefer larger; CFG also helps less as λ grows. Both are likely artifacts of fine-tuning only the conditional branch (required by adjoint matching).

K Qualitative Samples

This section collects the qualitative sample grids referenced from the main text. All grids use matched noise seeds and class labels across conditions, so visual differences are attributable to model weights and guidance only.

Base vs. Fine-tuned. Uncurated samples comparing the base model against the DRL fine-tuned model at λ=10 for each architecture. Each figure shows two sections side by side: No CFG and Best CFG, where the latter selects the best CFG value according to DINOv3-L FD. We use DINOv3 as a held-out metric since the discriminator is trained on DINOv2 features.

Effect of DRL Strength. Samples for each model under varying DRL strength (λ ∈ {1,5,10,20,40}) and three guidance strategies. Each figure uses matched noise seeds and class labels across all conditions, so differences are purely due to model weights. Reading left-to-right shows the effect of increasing λ; the three sections compare no guidance (CFG = 1), standard CFG = 2, and the per-variant optimal guidance (DINOv3-L FD-optimal CFG or autoguidance, annotated above each column; see section J for context on autoguidance). As remarked in section J, autoguidance is often the best guidance setting at large λ, and noticeably improves the images compared to the CFG variant.

RL Fine-tuning Sample Images. Generated samples for each model under varying RL strength (λPRL ∈ {1,10,40}) and guidance strategies (no CFG, CFG = 2, and autoguidance = 2; see section J for context on autoguidance). Each figure uses matched noise seeds across all conditions, so differences are purely due to model weights and guidance. The three column groups compare Base, DRL+PRL, and Base+PRL. As discussed in the main text, increasing λPRL makes the images progressively brighter and more distorted. This effect is particularly pronounced under autoguidance, which—as noted in sections J and J.4—yields the highest reward but the largest distortion.

- K.1 Base vs. Fine-tuned

###### JiT H-16

|No CFG<br><br>recreation|Best Guidance<br><br>base: CFG λ=10: CFG<br><br>al vehicle|
|---|---|
|[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]|[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]|
|[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>ball|[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>oon|
|[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>zeb<br><br>leop|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>ra<br><br>ard|
|[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>coral|[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>reef|
|[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]|[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]|

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

###### Figure 26 Base vs. DRL fine-tuned (λ=10) samples for JiT H-16.

###### SiT-XL/2

|No CFG<br><br>recreation|Best Guidance<br><br>base: CFG λ=10: CFG<br><br>al vehicle|
|---|---|
|[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]|[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]|
|[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>ball|[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>oon|
|[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>zeb<br><br>leop|[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>ra<br><br>ard|
|[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>coral|[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>reef|
|[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]|[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]|

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

###### Figure 27 Base vs. DRL fine-tuned (λ=10) samples for SiT-XL/2.

###### REPA SiT-XL/2

|No CFG<br><br>recreation|Best Guidance<br><br>base: CFG λ=10: CFG<br><br>al vehicle|
|---|---|
|[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]|[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]|
|[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>ball|[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>oon|
|[Figure 244]<br><br>[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>zeb<br><br>leop|[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>ra<br><br>ard|
|[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>coral|[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>reef|
|[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]|[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]|

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

###### Figure 28 Base vs. DRL fine-tuned (λ=10) samples for REPA SiT-XL/2.

###### RAE DiTDH-XL

|No CFG<br><br>recreation|Best Guidance<br><br>base: CFG λ=10: CFG<br><br>al vehicle|
|---|---|
|[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]|[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]|
|[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]<br><br>[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>ball|[Figure 316]<br><br>[Figure 317]<br><br>[Figure 318]<br><br>[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]<br><br>[Figure 323]<br><br>oon|
|[Figure 324]<br><br>[Figure 325]<br><br>[Figure 326]<br><br>[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>[Figure 331]<br><br>zeb<br><br>leop|[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>ra<br><br>ard|
|[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]<br><br>coral|[Figure 348]<br><br>[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>[Figure 353]<br><br>[Figure 354]<br><br>[Figure 355]<br><br>reef|
|[Figure 356]<br><br>[Figure 357]<br><br>[Figure 358]<br><br>[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]<br><br>[Figure 363]|[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]<br><br>[Figure 367]<br><br>[Figure 368]<br><br>[Figure 369]<br><br>[Figure 370]<br><br>[Figure 371]|

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

Base

λ=10

###### Figure 29 Base vs. DRL fine-tuned (λ=10) samples for RAE DiTDH-XL.

- K.2 Effect of DRL Strength

##### JiT H-16

Base λ=1 λ=5 λ=10 λ=20 λ=40 No CFG

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

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

CFG = 2

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Best Guidance

CFG CFG CFG CFG CFG CFG

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

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

- Figure 30 Effect of DRL strength (λ) on samples for JiT H-16.

##### SiT-XL/2

Base λ=1 λ=5 λ=10 λ=20 λ=40 No CFG

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

CFG = 2

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

Best Guidance

CFG CFG CFG CFG AG AG

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

- Figure 31 Effect of DRL strength (λ) on samples for SiT-XL/2.

##### REPA SiT-XL/2

Base λ=1 λ=5 λ=10 λ=20 λ=40 No CFG

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

CFG = 2

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

Best Guidance

CFG CFG AG CFG AG AG

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

- Figure 32 Effect of DRL strength (λ) on samples for REPA SiT-XL/2.

##### RAE DiTDH-XL

Base λ=1 λ=5 λ=10 λ=20 λ=40 No CFG

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

CFG = 2

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

Best Guidance

CFG CFG CFG CFG AG AG

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

- Figure 33 Effect of DRL strength (λ) on samples for RAE DiTDH-XL.

- K.3 RL Fine-tuning Sample Images

###### JiT H-16

###### Base DRL + Pref. RL Base + Pref. RL

ref λ=1 λ=10 λ=40 λ=1 λ=10 λ=40

No CFG

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

CFG = 2

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

Autoguidance

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

###### Figure 34 RL fine-tuning samples for JiT H-16.

###### SiT-XL/2

###### Base DRL + Pref. RL Base + Pref. RL

ref λ=1 λ=10 λ=40 λ=1 λ=10 λ=40

No CFG

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

CFG = 2

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

Autoguidance

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

###### Figure 35 RL fine-tuning samples for SiT-XL/2.

###### REPA SiT-XL/2

###### Base DRL + Pref. RL Base + Pref. RL

ref λ=1 λ=10 λ=40 λ=1 λ=10 λ=40

No CFG

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

CFG = 2

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

Autoguidance

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

###### Figure 36 RL fine-tuning samples for REPA SiT-XL/2.

###### RAE

###### Base DRL + Pref. RL Base + Pref. RL

ref λ=1 λ=10 λ=40 λ=1 λ=10 λ=40

No CFG

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

CFG = 2

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

Autoguidance

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

###### Figure 37 RL fine-tuning samples for RAE.

