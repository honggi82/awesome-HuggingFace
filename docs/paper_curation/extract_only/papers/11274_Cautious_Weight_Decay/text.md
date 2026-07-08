# arXiv:2510.12402v2[cs.LG]24Feb2026

[Figure 1]

## Cautious Weight Decay

###### Lizhang Chen∗†‡ Jonathan Li∗† Kaizhao Liang† Baiyu Su† Cong Xie Nuo Wang Pierse‡ Chen Liang‡ Ni Lao‡ Qiang Liu†

###### Abstract

We introduce Cautious Weight Decay (CWD), a one-line, optimizer-agnostic modification that applies weight decay only to parameter coordinates whose signs align with the optimizer update. Unlike standard decoupled decay, which implicitly optimizes a regularized or constrained objective, CWD preserves the original loss and admits a bilevel interpretation: it induces sliding-mode behavior upon reaching the stationary manifold, allowing it to search for locally Pareto-optimal stationary points of the unmodified objective. In practice, CWD is a drop-in change for optimizers such as AdamW, Lion, and Muon, requiring no new hyperparameters or additional tuning. For language model pre-training and ImageNet classification, CWD consistently improves final loss and accuracy at million- to billion-parameter scales.

### 1 Introduction

Algorithm 1 Cautious Weight Decay (CWD)

given parameters xt, optimizer update ut, learning rates ηt > 0, weight decay coefficient λ ≥ 0 xt+1 ← xt − ηt ut + λI(utxt ≥ 0)xt ▷ entrywise multiplication

Optimization algorithms lie at the core of modern deep learning, shaping not only convergence speed but also training stability and generalization ability across domains such as natural language processing and computer vision. As models and datasets scale, traditional methods such as stochastic gradient descent (SGD) and SGD with momentum [SMDH13] encounter limitations, including slow convergence in non-convex landscapes, sensitivity to learning rate schedules, and poor robustness to sparse or noisy gradients [SM20, ZMB+25]. In response, a wide range of alternatives have emerged, including adaptive gradient methods [DHS11, KB15], approximate second-order approaches [MG15, GKS18, YGS+21, LLH+24, NCLL24, WHML25], and specialized algorithms for extreme training regimes [LLCL24, LYL24, XZL+24, HZJ+25, ZCL+25].

Among these advances, decoupled weight decay [LH19] has proven especially influential. In its general form, decoupled weight decay augments any optimizer update ut with a decay term applied directly to the parameters, i.e.

xt+1 ← xt − ηt(ut + λxt), ut = OptimizerUpdate(xt).

∗Equal contribution by LC and JL. Correspondence: lzchen, jli@cs.utexas.edu †University of Texas at Austin ‡Google

AdamW

ValidationLoss

3.200

Ours

3.100

3.000

0.00 0.10 0.20 0.30 0.40

Weight Decay

Figure 1: Final validation loss vs. weight decay coefficient λ for 338M models trained on C4 under Chinchilla scaling. Our approach (red) achieves lower final loss than standard weight decay (blue) while preserving the optimizer-specific optimum in λ. For each optimizer (AdamW, Lion, Muon), both methods use the same hyperparameters.

This technique improves training stability and generalization by preventing the adaptive learning rates from interfering with regularization, as exemplified by the success of AdamW in large model training [BMR+20, DBK+21, TMS+23] and the subsequent development of state-of-the-art optimizers such as Lion [CLH+23], Lion-K [CLLL24], and Muon [JJB+24, LSY+25].

However, decoupled weight decay remains agnostic to the directional alignment between the optimizer update and the parameters, which may hurt performance when they conflict. Intuitively, when the update ut and parameters xt point in the same direction for a given dimension, weight decay acts as a regularizer that improves stability; however, when their directions differ, applying decay actively resists beneficial movement toward the optimum. Furthermore, decoupled weight decay has been shown to implicitly impose regularization terms on the objective function [CLLL24, XL24], which corresponds to parameter norm constraints for AdamW, Lion, and Muon.

In light of these limitations, we propose a simple refinement: cautious weight decay (CWD), in which decay is applied only in dimensions where the update and parameter signs align (Algorithm 1). Our main contributions are as follows.

- • We introduce cautious weight decay, a sign-selective extension of decoupled decay that applies weight decay only when the parameters and update align. Our technique can be implemented as a one-line modification without introducing additional hyperparameters compared to standard decoupled decay.
- • We use Lyapunov analysis to show that standard optimizers (SGD(M), Lion-K, Adam) with cautious weight decay are asymptotically stable and unbiased, in the sense that they optimize the original loss rather than a regularized surrogate. The regularization effect of cautious weight decay instead becomes a bilevel objective of finding locally Pareto-optimal points within the stationary manifold (Figure 2). Furthermore, we show a convergence rate for discrete-time Adam with cautious weight decay in the smooth nonconvex setting

Figure 2: Trajectories of Adam, AdamW, and Adam + CWD on a toy example. Adam halts at a minimizer, while AdamW minimizes the objective within a constrained region (green). In contrast, Adam + CWD exhibits sliding mode dynamics within the minimizer manifold.

- under additional assumptions.
- • In language modeling [OWS+25, KFP+25] and ImageNet classification [DDS+09], we observe that cautious weight decay generally accelerates convergence and lowers final validation loss for AdamW, Lion, and Muon (e.g., Figure 1). These improvements translate into higher zero-shot accuracy on standard benchmarks from 338M to 2B parameters and across architectures without retuning baseline settings (≈20,000 NVIDIA H100 HBM3-80GB GPU hours for all experiments). 2 Background and Motivation

- 2.1 Decoupled weight decay Gradient-based optimizers with decoupled weight decay can be characterized by the update rule

xt+1 = (1 − ηtλ)xt − ηtut, (1)

where ut := U(xt,g1,...,gt,t) is an adaptive, often sign-normalized update vector constructed from first and second-moment estimates (e.g., momentum buffers, diagonal preconditioners), ηt > 0 is the learning rate, and λ ≥ 0 is the decoupled weight decay coefficient. This framework encapsulates a wide range of standard optimizers for machine learning, including AdamW and Lion-K.

AdamW. The update vector is given by ut = D−t 1 mt, where Dt is a diagonal preconditioner and mt is bias-corrected first-moment estimate. Explicitly,

mt =

β1mt−1 + (1 − β1)gt 1 − β1t

, vt =

β2vt−1 + (1 − β2)gt2 1 − β2t

, Dt = diag vt + ϵ1 ,

where β1 and β2 are momentum coefficients and ϵ is a numerical stability constant.

Lion-K. Given a convex function K, the update vector ut is a momentum-filtered step that is preconditioned using a subgradient, i.e.

mt = β2mt−1 − (1 − β2)gt, mt = β1mt−1 − (1 − β1)gt, ut = −∇K( mt),

where β1 and β2 are momentum coefficients and ∇K is a subgradient of K. Examples include Lion when K = ∥·∥1 and Muon when K = ∥·∥tr, where ∥·∥tr denotes the nuclear norm when the parameters are treated as a matrix.

- 2.2 Implicit regularization effects of weight decay

In general, the application of decoupled weight decay imposes a certain regularization or constraint effect on the objective function, where the specific effect depends on the choice of ut. For example, SGD with decoupled weight decay is exactly SGD on an ℓ2-regularized objective. To see the equivalence, let f : Rd → R be differentiable and consider the regularized variant f(x) := f(x)+ λ2 ∥x∥22 . A single SGD step on f with learning rate ηt > 0 yields the update

xt+1 = xt − ηt(∇f(xt) + λxt) = (1 − ηtλ)xt − ηt∇f(xt), which is precisely the decoupled weight decay update given by (1).

Given a convex function K with subgradient ∇K and convex conjugate K∗, suppose the iterates of Lion-K converge to a fixed point (x⋆,m⋆, m⋆). Then the moment estimators stabilize so that m⋆ = m⋆ = −∇f(x⋆), and the fixed-point condition yields −∇K(−∇f(x⋆))+λx⋆ = 0. Rearranging and using the identity (∇K)−1 = ∇K∗, we obtain ∇f(x⋆) + ∇K∗(λx⋆) = 0, where the left-hand side is the gradient of the function

1 λK∗(λx).

f(x) := f(x) +

This suggests that Lion-K optimizes the regularized objective f, an observation made by [CLLL24]. In the special cases of Lion and Muon, K∗ is the 0-∞ indicator function of a dual norm ball, corresponding to the constrained optimization problems

1 λ

1 λ

f(x) s.t. ∥x∥∞ ≤

and min

f(X) s.t. ∥X∥op ≤

min

,

X∈Rn×m

x∈Rd

respectively, where ∥·∥op is the spectral norm when the parameters are treated as a matrix. A similar analysis for AdamW suggests that it solves the box-constrained problem of minimizing f(x) such that ∥x∥∞ ≤ λ1, but convergence cannot be established due to the lack of a Lyapunov function. For more discussion, see Appendix C and [XL24].

While AdamW and Lion-K are practically strong, they implicitly optimize a regularized surrogate that is dependent on the weight decay coefficient λ. This motivates the development of a mechanism that maintains the beneficial effects of decoupled weight decay (e.g. regularization, training acceleration) while optimizing the original objective.

- 3 Cautious Weight Decay Cautious weight decay (CWD) modifies the update rule (1) as

xt+1 = xt − ηt(ut + λI(ut ⊙ xt ≥ 0) ⊙ xt),

where ⊙ denotes entrywise multiplication.1 As a one-line modification, cautious weight decay is implementation-trivial and universally compatible with gradient-based optimization algorithms. Theoretically, cautious weight decay also exhibits the following behavior.

- • Unbiased optimization, in the sense that every accumulation point x⋆ of the trajectory satisfies ∇f(x⋆) = 0 under the same convergence conditions required of the base optimizer without weight decay. In over-parameterized deep models, the set of stationary points is typically a union of connected submanifolds rather than isolated points. Consequently, the ω-limit set of the trajectory is contained in some stationary manifold, and the iterates eventually remain arbitrarily close to it.
- • Sliding mode dynamics within the stationary manifold, where cautious weight decay allows the trajectory to traverse along the manifold until it cannot decrease the parameter magnitudes in every coordinate. In other words, cautious weight decay steers the trajectory towards a local Pareto front of the stationary manifold under the ordering that prioritizes smaller parameter magnitudes.

1Throughout the paper, when it is clear from context, we also drop ⊙ and write v ⊙ x = vx for simplicity.

##### 3.1 Convergence to the stationary manifold

We construct Lyapunov functions for the continuous-time limits of several standard optimizers equipped with cautious weight decay. A Lyapunov function is a lower bounded function with nonpositive derivative that is used to certify the stability of systems of differential equations.

Consider the continuous-time dynamics of SGD with cautious weight decay

x˙t = −∇f(xt) − λI(∇f(xt)xt ≥ 0)xt.

This ODE has the Lyapunov function H(x) = f(x), since H is lower bounded and

dH dt

= ⟨∇f(xt),−∇f(xt) − λI(∇f(xt)xt ≥ 0)xt⟩ = −∥∇f(xt)∥22 − λ (∇f(xt)xt)+ 1 ≤ 0,

where (·)+ := max(0,·). LaSalle’s invariance principle [LaS60] states that the accumulation points of any trajectory lie within the union of trajectories zt that satisfy ddtH(zt) = 0 for all t ≥ 0. Consequently, we conclude that SGD with cautious weight decay produces trajectories that approach the stationary set {x | ∇f(x) = 0} of the original loss. This holds because cautious weight decay is applied only in a secondary fashion and is automatically deactivated whenever it conflicts with the main objective, thereby ensuring that the loss landscape remains unbiased.

Beyond the simple case of SGD, the same Lyapunov-type argument can be extended to more sophisticated algorithms such as SGDM, Lion-K, and Adam. In each case, cautious weight decay still minimizes the original objective without introducing explicit bias, but a key difficulty lies in constructing appropriate Lyapunov functions. Table 1 summarizes the Lyapunov functions of several major optimizers with cautious weight decay, and detailed derivations are provided in Appendix D. By applying LaSalle’s invariance principle, we can show that the momentum-based algorithms in Table 1 converge to the stationary set of the original objective, together with vanishing momentum:

{(x,m) | ∇f(x) = 0, m = 0}.

##### 3.2 Sliding mode dynamics

Although both standard optimization (with no weight decay) and cautious weight decay are unbiased with respect to the original objective, their behaviors diverge within the stationary manifold. In the former, the dynamics halt as the momentum m decays to zero, while, in contrast, the cautious weight decay dynamics induce a sliding mode, continuing to move along the manifold while reducing the parameter magnitudes as much as possible. Consequently, the algorithm converges to a subset of the stationary manifold where further simultaneous reduction of all coordinates of x is no longer possible. Equivalently, it converges to a locally Pareto-optimal stationary point under a preference for smaller parameter magnitudes.

To provide mathematical background, consider a possibly time-varying discontinuous ODE

z˙t = ft(zt), zt ∈ Rd.

Due to the discontinuity of ft, the solution may not be well defined in the classical or Carathéodory sense, especially across switching surfaces. We therefore interpret solutions in the Filippov sense

- Table 1: Comparison of the continuous-time dynamics of different optimizers. SGDM represents

SGD with momentum. Lion-K includes Lion (K = ∥·∥1) and Muon (K = ∥·∥tr) as special cases. f : Rd → R is assumed to be differentiable and lower bounded by f⋆.

Optimizer Continuous-time dynamics Lyapunov function SGD + CWD x˙t = −∇f(xt) − λI(∇f(xt)xt ≥ 0)xt H(x) = f(x) SGDM + CWD x˙t = −mt − λI(mtxt ≥ 0)xt

H(x,m) = βf(x) + 21 ∥m∥22 + λ∥(mx)+∥1

m˙ t = β(∇f(xt) − mt)

Lion-K + CWD x˙t = ∇K(mt) − λI(mtxt ≤ 0)xt

H(x,m) = αf(x) + K(m) + λ∥(−mx)+∥1

m˙ t = −α∇f(xt) − γmt

αtm2 2h 1

αtmt

+ λ (mx)+ 1

Adam + CWD x˙t = −

ht − λI(mtxt ≥ 0)xt m˙ t = α(∇f(xt) − mt)

Ht(x,m,h) = αf(x) +

v˙t = γ(∇f(xt)2 − vt)

Notation. We drop ⊙ for simplicity. αt := (1 − exp(−αt))−1, γt := (1 − exp(−γt))−1, ht := √γtvt + ϵ1.

[Fil88], where a discontinuous ODE is formally a differential inclusion that specifies that z˙t belongs to the closed convex envelope of the discontinuous vector field, i.e.

co(ft(B(zt,δ) \ S)),

z˙t ∈ F[ft](zt) :=

δ>0 µ(S)=0

where µ denotes the Lebesgue measure, B(z,δ) is the δ-ball centered at z, and co denotes the closed convex envelope. This construction captures all possible limiting directions of the vector field near discontinuities, ensuring well-defined dynamics even when ft is not continuous. The key idea is that the values of z˙t must be determined by the behavior of ft in a neighborhood around zt, rather than at the point itself. The inclusion, therefore, defines a range of admissible velocities consistent with the nearby values of the vector field.

In particular, whenever ft contains coordinatewise indicators such as I(g(zt) ≥ 0), the Filippov set replaces them by selectors st ∈ [0,1]d on the switching set {[g(zt)]i = 0}:

 

{1} [g(zt)]i > 0, {0} [g(zt)]i < 0, [0,1] [g(zt)]i = 0.

[st]i ∈



Recalling the Lyapunov analysis in Section 3.1, the continuous-time dynamics of standard optimizers with cautious weight decay converge to the stationary manifold M := {x | ∇f(x) = 0}, with the momentum mt also decaying to 0 for momentum-based methods. Consequently, once the trajectory enters the stationary manifold, the residual dynamics reduce to

x˙t = −λst ⊙ xt, st ∈ [0,1]d. (2)

- Figure 3: Toy objectives and trajectories. Left: f(x,y) = ((y − 3)2 + (x − 3)2 − 1)2. Right: f(x,y) = (y − 3 − (x − 3)2)2. We compare Adam, AdamW, and Adam + CWD; AdamW and

CWD use the same weight decay λ, and all other hyperparameters (η,β1,β2,ϵ) are identical. For both objectives, Adam converges to a generic point on the minimizer manifold, whereas AdamW

converges to a solution of the box-constrained problem minx,y f(x,y) subject to max{x,y} ≤ λ1. In contrast, Adam + CWD converges to the Pareto front of the minimizer manifold.

Moreover, since the Lyapunov function confines the dynamics to the stationary set, the selectors st must be chosen such that the trajectory remains within the manifold. Differentiating the stationarity condition yields

d dt∇f(xt) = −λ∇2f(xt)(st ⊙ xt) = 0, st ∈ [0,1]d.

This relation allows us to solve for admissible choices of st that guarantee invariance of the manifold. In general, the solution for st need not be unique, and the actual value realized in practice may be implicitly determined by the discretization scheme employed.

Effectively, cautious weight decay decreases parameter magnitudes along each coordinate while staying within the stationary manifold, pushing x toward the local Pareto front of the manifold

P := {x ∈ M | ∃δ > 0 ∀y ∈ (B(x,δ) ∩ M) \ {x},|y| ̸≤ |x|},

where the tangent space no longer allows a nonzero st in (2). In other words, a stationary point is locally Pareto-optimal if it has a neighborhood in the stationary manifold that contains no other point with a smaller or equal magnitude in every coordinate.

This argument shows that cautious weight decay dynamics converge to P. Since P may not be a singleton, the exact limit point depends intricately on initialization and the discretization of the continuous-time dynamics. Figure 3 illustrates this behavior on two toy problems.

##### 3.3 Discrete-time analysis

Leveraging the Lyapunov functions in Table 1, we can extend our continuous-time analysis to obtain convergence guarantees for the discrete-time dynamics of various optimizers with cautious weight decay. As a concrete example, we provide in Appendix E an explicit convergence rate for discretetime Adam with cautious weight decay.

- Figure 4: Evaluation loss across scales. 3×3 grid for 338M, 986M, and 2B Transformer models trained with AdamW, Lion, and Muon on C4 dataset. All panels show a zoom into the final ∼40% of training steps to highlight late-stage behavior. Baseline curves (dashed blue) use standard weight decay with tuned hyperparameters (learning rate schedule, β’s, weight decay, etc.; see Appendix F). Our method (solid red) follows Algorithm 1 and reuses the baseline hyperparameters without additional tuning. Full (non-zoomed) curves are in Figures 8, 9 and 10 in Appendix G.

### 4 Experiments

Overview. We evaluate CWD against three standard optimizers—AdamW, Lion, and Muon—on autoregressive language modeling and ImageNet classification. For Transformer models with similar architecture to Gemma [KFP+25] with 338M, 986M, and 2B parameters in the Simply [LHY+25] codebase, we follow the Chinchilla compute-optimal scaling rule—20 tokens per parameter (TPP) [HBM+22] and train on C4 [RSR+20]. For each size, we grid-search batch size, learning rate, weight decay, warmup ratio, and optimizer-specific hyperparameters for the baselines (AdamW, Lion, Muon); we then reuse the selected baseline settings for CWD without retuning (details in Appendix F). Under matched settings, CWD lowers final validation loss and improves zero-shot accuracy. On the OLMo codebase [OWS+25], we further study an over-training regime—OLMo-1B trained on 100B tokens (100 TPP) from Dolma [SKB+24]. Under matched settings, CWD lowers final validation loss and improves zero-shot accuracy (Table 4). We also observe similar gains on ImageNet [DDS+09]

- Table 2: Ablation study of selective weight decay strategies on OLMo-1B (100B tokens). We compare our momentum-based selection against alternative masking approaches. Baseline: standard weight decay (λ tuned). Ours: update-based mask I(ux ≥ 0) using baseline’s λ without retuning. Random: time-varying Bernoulli mask matching our method’s sparsity ratio (see Figure 7 in Appendix G). Gradient: uses I(gx ≥ 0) instead. No WD: λ = 0. Lower validation loss is better.

Weight Decay Active Ablated Masks Disabled Optimizer Baseline Ours Random Gradient No WD

AdamW 2.65 2.56 2.82 2.75 2.70 Muon 2.51 2.42 2.73 2.74 2.62

- Table 3: ImageNet validation accuracy (%) across architectures and optimizers. All models train for 300 epochs with standard augmentation. Base: optimizer with tuned weight decay. Ours: cautious weight decay using the same coefficient as baseline (no retuning).

AdamW Lion Muon Model Params Base Ours Base Ours Base Ours

ViT-S/16 22.05M 78.84 79.45 79.29 79.82 79.35 79.91 ResNet-50 25.56M 76.30 76.68 76.41 76.75 76.47 76.83 ViT-B/16 86.57M 80.15 80.71 80.76 80.92 80.83 81.04

across ViT [DBK+21] and ResNet [HZRS16].

Ablations of weight decay. Figure 1 sweeps the weight–decay coefficient λ for a 338M model on C4: λ∈[0, 0.4] for Muon and AdamW, and λ∈[0, 3.0] for Lion. Two patterns are consistent across runs: (i) at a fixed λ, CWD attains a lower final loss than the corresponding baseline with decoupled weight decay; (ii) the minimizing value λ⋆ is essentially unchanged when replacing the baseline with CWD. In practice, one can swap in CWD at an already tuned λ and obtain improvements without additional sweeps.

Ablations on masking. Table 2 tests whether the benefits arise from the amount of decay applied or from CWD’s structure. Replacing our mask with a time-matched Bernoulli “random mask” substantially degrades performance (e.g., 2.56 → 2.82 for AdamW, 2.42 → 2.73 for Muon), showing that simply reducing the frequency of decay is insufficient. Substituting the indicator with the gradient-based I(gx ≥ 0) also underperforms. Finally, λ = 0 remains worse than tuned decay, illustrating that explicit regularization is helpful and CWD leverages it more effectively. We further verify the difference between CWD and SPD [THK24] with additional instruction-following fine-tuning experiments. We fine-tune on the Alpaca GPT-4 dataset [PLH+23], which contains 52,000 instruction–response pairs generated by GPT-4 [Ope23], and evaluate on three benchmarks: MMLU [HBB+21], comprising 57 tasks and 14,079 questions covering a broad range of world knowledge; AGIEval [ZCG+24], a human-centric benchmark with 9,316 instances targeting general reasoning and problem-solving skills; and WinoGrande [SBBC21], a large-scale commonsense reasoning dataset with 44,000 instances. We consider two base models, TinyLlama [ZZWL24] and Mistral7B [JSM+23], and compare LoRA [HSW+22], SPD [THK24], a layerwise “inner-product” variant of CWD using I(⟨u,x⟩ ≥ 0), and our proposed “elementwise” CWD. Table 5 reports accuracy on MMLU, AGIEval, and WinoGrande for TinyLlama and Mistral-7B.

Training dynamics. On 1B models trained for 100B tokens, we observe that CWD tends to improve

###### Optimizer Hellaswag ↑ ARC-Easy ↑ ARC-C ↑ PIQA ↑ MMLU ↑ ComQA ↑

acc_norm acc_norm acc_norm acc_norm acc acc

AdamW 0.38 0.50 0.25 0.67 0.23 0.29 AdamW+CWD 0.40 0.53 0.27 0.69 0.25 0.31

Muon 0.39 0.51 0.26 0.68 0.24 0.30 Muon+CWD 0.41 0.51 0.28 0.71 0.26 0.33

- Table 4: Downstream accuracy across diverse reasoning benchmarks. All runs use the OLMo codebase with 1B-parameter models trained for 100B tokens under an over-training regime. Here ARC-C=ARC-Challenge and ComQA=CommonsenseQA. Figure 5 shows the corresponding loss curves.

- AdamW (wd=0.0)

- AdamW (wd=0.1)

- Muon (wd=0.0)

- Muon (wd=0.1)

10

Perplexity(logscale)

Perplexity(logscale)

10

Ours (wd=0.1)

Ours (wd=0.1)

8

8

6

6

4

4

2

0K 50K 100K 150K 200K 250K 300K

0K 50K 100K 150K 200K 250K 300K

Training Steps

Training Steps

Figure 5: Training loss of OLMo 1B on 100B tokens. Left: AdamW. Right: Muon.

the loss trajectory relative to tuned AdamW and Muon, rather than only the final value (Figure 5). A similar pattern appears at 986M: Figure 11 in Appendix G shows evaluation/training loss and RMS parameter norm over time. CWD generally achieves lower loss while ending with an intermediate norm. In contrast, removing decay entirely (λ = 0) descends faster mid-training but plateaus earlier, finishing at higher loss and the largest norm; tuned AdamW with λ > 0 yields the smallest norm. Overall, these results suggest that the gains come from a more selective application of regularization rather than from disabling it.

CWD outperforms standard decay across optimizers and scales. Under the common setup across 338M, 986M, and 2B parameters, CWD consistently lowers eval loss for AdamW, Lion, and Muon (see Figure 4 and Figures 8–10 in Appendix G) and increases downstream accuracy (Table 4).

CWD yields lower gradient norms than standard decay. Across model sizes, CWD produces lower RMS-normalized gradient norms than the corresponding baselines (see Figure 12 in Appendix G). This coincides with the lower end-of-training loss in Figure 5 and the accuracy gains in Table 4.

###### Model Method MMLU (5-shot) ↑ AGIEval (3-shot) ↑ WinoGrande (5-shot) ↑

TinyLlama LoRA (baseline) 25.81 ± 0.07 19.82 ± 0.11 61.33 ± 0.09 TinyLlama SPD 26.14 ± 0.08 20.21 ± 0.10 61.92 ± 0.08 TinyLlama Inner-product CWD 26.02 ± 0.08 19.80 ± 0.10 61.70 ± 0.09 TinyLlama Elementwise CWD 26.42 ± 0.09 20.12 ± 0.09 62.18 ± 0.08

Mistral-7B LoRA (baseline) 61.78 ± 0.09 27.56 ± 0.07 78.85 ± 0.11 Mistral-7B SPD 62.05 ± 0.08 27.98 ± 0.06 78.81 ± 0.10 Mistral-7B Inner-product CWD 61.76 ± 0.09 27.90 ± 0.07 78.83 ± 0.10 Mistral-7B Elementwise CWD 62.13 ± 0.07 28.31 ± 0.06 78.92 ± 0.09

- Table 5: Accuracy on MMLU, AGIEval, and WinoGrande for TinyLlama and Mistral-7B fine-tuned on Alpaca GPT-4 using LoRA, SPD, and two variants of CWD. Both SPD and CWD improve over the LoRA baseline, and the proposed elementwise CWD matches or outperforms SPD and the innerproduct variant on most benchmarks.

Scaling with model size. We measured final validation loss for AdamW and CWD (“Ours”) at 111M, 338M, 986M, and 2B parameters on the same dataset (Figure 6). At every scale, CWD attains a lower final validation loss than AdamW, and the gap remains stable or even widens with model size, indicating that the advantages of cautious weight decay persist into the large-model regime.

Figure 6: Final validation loss versus model size for AdamW and Adam + CWD (“Ours”) on the same pretraining dataset, at 111M, 338M, 986M, and 2B parameters. Across all scales, Adam + CWD achieves consistently lower final validation loss than AdamW, and the gap remains stable or slightly increases with model size, suggesting that the benefits of cautious weight decay persist in the largemodel regime.

111M338M 986M 2B

Model size (M parameters)

2.6

2.8

3.0

3.2

Finalvalidationloss

| |
|---|

| |
|---|

Ours

Baseline

Model Size Optimizer Final Validation Loss

338M AdamW (baseline) 3.0136 338M Adam + CWD 3.0059 338M AdamC 3.0087 338M AdamC + CWD 2.9915

- Table 6: Final validation loss for Gemma-based models with different optimizers at 338M parameters. Adding cautious weight decay improves both AdamW and AdamC, with AdamC + CWD achieving the lowest loss.

###### Model Size Optimizer Scheduler Final Validation Loss

338M AdamW (baseline) Cosine 3.0136 338M AdamW (baseline) WSD 3.0101 338M Adam + CWD Cosine 3.0059 338M Adam + CWD WSD 3.0014

- Table 7: Final validation loss for a 338M model under different optimizer–scheduler combinations. Adam + CWD improves over the AdamW baseline for both cosine and WSD schedules, with the best performance obtained by Adam + CWD with WSD.

Compound improvement with other techniques. We also observe compounding gains when combining cautious weight decay with other optimizer techniques on Gemma-based model architectures. In particular, we compare AdamW, AdamC [Def25], and their variants with CWD at 338M parameters (Table 6).

Robustness to learning-rate schedules. We observe that CWD improves performance for both cosine scheduling and the Warmup–Stable–Decay (WSD) schedule, which uses a 10% warmup, a long stable phase, and a 20% final decay (Table 7).

### 5 Related Work

Weight decay. Weight decay originated as an ℓ2 penalty for ill-posed problems and ridge regression [Tik63, HK70] and was introduced to neural networks as a generalization tool to mitigate overfitting [HP88, WRH90, KH91]. [LH19] showed that, for adaptive methods, weight decay and ℓ2 regularization are not equivalent, motivating the decoupled formulation in AdamW; subsequent work established decoupled decay as a standard feature of modern optimizers [CLH+23, CLLL24, LSY+25]. Recent analyses suggest that in contemporary networks, weight decay functions more as a training accelerator and stabilizer than as explicit regularization [KSH17, HBM+22, PC23, DAVF24]. Interactions with early-stage training [BWG21], normalization layers, and learning rate schedules [Def25] have also been clarified, and architectural designs can obviate explicit decay [LHSG25].

Weight decay variants. Various efforts have been made to develop different adaptive variants of weight decay. For example, [XXZ+23] found that weight decay can lead to large gradient norms at the final phase of training and proposed Scheduled Weight Decay (SWD) to dynamically adjust weight decay strength based on gradient norms. [KMJ24] investigate how weight decay affects individual neuron updates, revealing rotational equilibrium states that balance learning across layers and neurons. [GSA23] introduce adaptive weight decay that automatically tunes the hyperparameter during training based on classification and regularization loss gradients, achieving significant improvements in adversarial robustness. [THK24] introduce Selective Projection Decay (SPD) for robust fine-tuning, featuring selective weight decay via a mask that is somewhat similar to CWD. However, SPD and CWD differ in significant ways, including the intended setting, mechanism, theoretical properties, and empirical performance.

Masked or conditional updates. Several works have explored the sign-based conditioning of optimizer updates. [RB93] introduced Rprop, which adjusted step sizes based on current gradient and past gradient sign agreement. [LCLL24] propose the cautious optimizer, which restricts updates

to dimensions where the proposed update and current gradient share the same sign. [WLX+24] apply a similar mask to Adam to improve robustness in online learning.

Constrained and bilevel optimization. Decoupled weight decay can be interpreted through the lens of Frank–Wolfe algorithms for constrained optimization [FW56, Jag13, SW25, PXA+25]. This connection suggests that optimizers with decoupled weight decay implicitly solve constrained optimization problems, which was shown to be the case for Lion [CLLL24, SW25, PXA+25], AdamW [XL24, BN24], and Muon [CLL25, SW25, LLS25]. In contrast, optimizers with cautious weight decay perform bilevel optimization, a framework from classical optimization [Sol07a, Sol07b, SS17] that has been recently explored in machine learning [GLL21, LYW+22, PMA24].

### 6 Conclusion

We introduce cautious weight decay and formalize it as a simple, optimizer-agnostic modification of decoupled weight decay that preserves the optimization objective while retaining the practical benefits of weight decay. For standard optimizers (SGD, Adam, and Lion-K), we show the bilevel optimization structure of cautious weight decay and establish convergence guarantees in both continuous- and discrete-time regimes. Across diverse tasks and benchmarks, cautious weight decay consistently improves training dynamics compared to no decay and traditional decoupled decay, yielding faster loss reduction and more stable trajectories without changes to hyperparameters or model architectures. Our results indicate that cautious weight decay is a theoretically principled and empirically effective technique that retains the benefits of weight decay while addressing its fundamental limitations.

### Acknowledgments

This work was supported in part by the Institute for Foundations of Machine Learning (IFML) and the Office of Naval Research (ONR) under Grant No. N00014-25-1-2354.

### References

[ACD+23] Yossi Arjevani, Yair Carmon, John C. Duchi, Dylan J. Foster, Nathan Srebro, and Blake E. Woodworth. Lower bounds for non-convex stochastic optimization. Math. Program., 199(1):165–214, 2023.

- [BB21] Anas Barakat and Pascal Bianchi. Convergence and dynamical behavior of the ADAM algorithm for nonconvex stochastic optimization. SIAM J. Optim., 31(1):244–274, 2021.
- [BC99] Andrea Bacciotti and Francesca Ceragioli. Stability and stabilization of discontinuous systems and nonsmooth lyapunov functions. ESAIM: Control, Optimisation and Calculus of Variations, 4:361–376, 1999.

[BMR+20] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark,

Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, 2020.

[BN24] Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. CoRR, abs/2409.20325, 2024.

[BWAA18] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. SIGNSGD: compressed optimisation for non-convex problems. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, volume 80 of Proceedings of Machine Learning Research, pages 559–568, 2018.

[BWG21] Johan Bjorck, Kilian Q. Weinberger, and Carla P. Gomes. Understanding decoupled and early weight decay. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, pages 6777–6785, 2021.

[CLH+23] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic discovery of optimization algorithms. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, 2023.

[CLL25] Lizhang Chen, Jonathan Li, and Qiang Liu. Muon optimizes under spectral norm constraints. CoRR, abs/2506.15054, 2025.

[CLLL24] Lizhang Chen, Bo Liu, Kaizhao Liang, and Qiang Liu. Lion secretly solves a constrained optimization: As lyapunov predicts. In The Twelfth International Conference on Learning Representations, ICLR 2024, 2024.

[CLSH19] Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of A class of adam-type algorithms for non-convex optimization. In 7th International Conference on Learning Representations, ICLR 2019, 2019.

[CSZL22] Congliang Chen, Li Shen, Fangyu Zou, and Wei Liu. Towards practical adam: Nonconvexity, convergence theory, and mini-batch acceleration. J. Mach. Learn. Res., 23(229):1–47, 2022.

[DAVF24] Francesco D’Angelo, Maksym Andriushchenko, Aditya Vardhan Varre, and Nicolas Flammarion. Why do we need weight decay in modern deep learning? In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024.

[DBBU22] Alexandre Défossez, Léon Bottou, Francis R. Bach, and Nicolas Usunier. A simple convergence proof of adam and adagrad. Trans. Mach. Learn. Res., 2022, 2022.

[DBK+21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Trans-

formers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, 2021.

[DDS+09] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), pages 248–255, 2009.

[Def25] Aaron Defazio. Why gradients rapidly increase near the end of training. CoRR, abs/2506.02285, 2025.

[DHS11] John C. Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. J. Mach. Learn. Res., 12:2121–2159, 2011.

[Fil88] Aleksej F. Filippov. Differential Equations with Discontinuous Righthand Sides, volume 18 of Mathematics and Its Applications. Springer, 1988.

[FW56] Marguerite Frank and Philip Wolfe. An algorithm for quadratic programming. Nav. Res. Logist. Q., 3(1–2):95–110, 1956.

[GBB+21] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling. CoRR, abs/2101.00027, 2021.

[GKS18] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, volume 80 of Proceedings of Machine Learning Research, pages 1837–1845, 2018.

[GL13] Saeed Ghadimi and Guanghui Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM J. Optim., 23(4):2341–2368, 2013.

[GLL21] Chengyue Gong, Xingchao Liu, and Qiang Liu. Automatic and harmless regularization with constrained and lexicographic optimization: A dynamic barrier approach. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, pages 29630–29642, 2021.

[GSA23] Amin Ghiasi, Ali Shafahi, and Reza Ardekani. Improving robustness with adaptive weight decay. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, 2023.

[HBB+21] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, 2021.

[HBM+22] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems

35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, 2022.

[HK70] Arthur E. Hoerl and Robert W. Kennard. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67, 1970.

[HP88] Stephen Jose Hanson and Lorien Y. Pratt. Comparing biases for minimal network construction with back-propagation. In Advances in Neural Information Processing Systems 1, NIPS Conference, pages 177–185, 1988.

[HSW+22] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, 2022.

[HZJ+25] Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, and Shiwei Liu. SPAM: spike-aware adam with momentum reset for stable LLM training. In The Thirteenth International Conference on Learning Representations, ICLR 2025, 2025.

[HZRS16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, pages 770–778, 2016.

[Jag13] Martin Jaggi. Revisiting frank-wolfe: Projection-free sparse convex optimization. In Proceedings of the 30th International Conference on Machine Learning, ICML 2013, volume 28 of JMLR Workshop and Conference Proceedings, pages 427–435, 2013.

[JJB+24] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024.

[JSM+23] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. CoRR, abs/2310.06825, 2023.

[KB15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015, 2015.

[KFP+25] Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-Bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Róbert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine

Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. ChoquetteChoo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucinska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju-yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle K. Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clément Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry (Dima) Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical report. CoRR, abs/2503.19786, 2025.

[KH91] Anders Krogh and John A. Hertz. A simple weight decay can improve generalization. In Advances in Neural Information Processing Systems 4, NIPS Conference, pages 950– 957, 1991.

[KMJ24] Atli Kosson, Bettina Messmer, and Martin Jaggi. Rotational equilibrium: How weight decay balances learning across neural networks. In Forty-first International Conference on Machine Learning, ICML 2024, 2024.

[KSH17] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. Commun. ACM, 60(6):84–90, 2017.

[LaS60] Joseph P. LaSalle. Some extensions of liapunov’s second method. IRE Transactions on Circuit Theory, 7(4):520–527, 1960.

[LCLL24] Kaizhao Liang, Lizhang Chen, Bo Liu, and Qiang Liu. Cautious optimizers: Improving training with one line of code. CoRR, abs/2411.16085, 2024.

[LH19] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, 2019.

[LHSG25] Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg. ngpt: Normalized transformer with representation learning on the hypersphere. In The Thirteenth International Conference on Learning Representations, ICLR 2025, 2025.

[LHY+25] Chen Liang, Da Huang, Chengrun Yang, Xiaomeng Yang, Andrew Li, Xinchen Yan, and Simply Contributors. Simply: an experiment to accelerate and automate AI research. GitHub repository, 2025.

[LLCL24] Kaizhao Liang, Bo Liu, Lizhang Chen, and Qiang Liu. Memory-efficient LLM training with online subspace descent. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024.

[LLH+24] Hong Liu, Zhiyuan Li, David Leo Wright Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic second-order optimizer for language model pre-training. In The Twelfth International Conference on Learning Representations, ICLR 2024, 2024.

[LLS25] Tim Tsz-Kit Lau, Qi Long, and Weijie Su. Polargrad: A class of matrix-gradient optimizers from a unifying preconditioning perspective. CoRR, abs/2505.21799, 2025.

[LSY+25] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for LLM training. CoRR, abs/2502.16982, 2025.

[LYL24] Qijun Luo, Hengxu Yu, and Xiao Li. Badam: A memory efficient full parameter optimization method for large language models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024.

[LYW+22] Bo Liu, Mao Ye, Stephen Wright, Peter Stone, and Qiang Liu. Bome! bilevel optimization made easy: A simple first-order approach. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, 2022.

[MG15] James Martens and Roger B. Grosse. Optimizing neural networks with kroneckerfactored approximate curvature. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 2408–2417, 2015.

[NCLL24] Son Nguyen, Lizhang Chen, Bo Liu, and Qiang Liu. H-fac: Memory-efficient optimiza-

tion with factorized hamiltonian descent. CoRR, abs/2406.09958, 2024. [Ope23] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.

[OWS+25] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 olmo 2 furious. CoRR, abs/2501.00656, 2025.

[PC23] Leyan Pan and Xinyuan Cao. Towards understanding neural collapse: The effects of batch normalization and weight decay. CoRR, abs/2309.04644, 2023.

[PLH+23] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with GPT-4. CoRR, abs/2304.03277, 2023.

[PMA24] Ieva Petrulionyte, Julien Mairal, and Michael Arbel. Functional bilevel optimization for machine learning. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024.

[PXA+25] Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio SilvetiFalls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. In Forty-second International Conference on Machine Learning, ICML 2025, 2025.

[RB93] Martin A. Riedmiller and Heinrich Braun. A direct adaptive method for faster backpropagation learning: the RPROP algorithm. In Proceedings of International Conference on Neural Networks (ICNN’88), San Francisco, CA, USA, March 28 - April 1, 1993, pages 586–591. IEEE, 1993.

[RKK18] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In 6th International Conference on Learning Representations, ICLR 2018, 2018.

[RSR+20] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1–67, 2020.

[SBBC21] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: an adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99– 106, 2021.

[SKB+24] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Raghavi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus of three trillion tokens for language model pretraining

research. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, pages 15725–15788, 2024.

[SM20] Kevin Scaman and Cédric Malherbe. Robustness analysis of non-convex stochastic gradient descent using biased expectations. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, 2020.

[SMDH13] Ilya Sutskever, James Martens, George E. Dahl, and Geoffrey E. Hinton. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on Machine Learning, ICML 2013, volume 28 of JMLR Workshop and Conference Proceedings, pages 1139–1147, 2013.

- [Sol07a] Mikhail V. Solodov. A bundle method for a class of bilevel nonsmooth convex minimization problems. SIAM J. Optim., 18(1):242–259, 2007.
- [Sol07b] Mikhail V. Solodov. An explicit descent method for bilevel convex optimization. J. Convex Anal., 14(2):227–238, 2007.

[SP94] Daniel W. Shevitz and Brad Paden. Lyapunov stability theory of nonsmooth systems. IEEE Trans. Autom. Control., 39(9):1910–1914, 1994.

[SS17] Shoham Sabach and Shimrit Shtern. A first order method for solving convex bilevel optimization problems. SIAM J. Optim., 27(2):640–660, 2017.

[SW25] Maria-Eleni Sfyraki and Jun-Kun Wang. Lions and muons: Optimization via stochastic frank-wolfe. CoRR, abs/2506.04192, 2025.

[THK24] Junjiao Tian, Chengyue Huang, and Zsolt Kira. Rethinking weight decay for robust finetuning of foundation models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024.

[Tik63] Andrey Tikhonov. On the solution of ill-posed problems and the method of regularization. Dokl. Akad. Nauk SSSR, 151(3):501–504, 1963.

[TMS+23] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023.

[WHML25] Kaiyue Wen, David Hall, Tengyu Ma, and Percy Liang. Fantastic pretraining optimizers and where to find them. CoRR, abs/2509.02046, 2025.

[WLX+24] Shaowen Wang, Anan Liu, Jian Xiao, Huan Liu, Yuekui Yang, Cong Xu, Qianqian Pu, Suncong Zheng, Wei Zhang, and Jian Li. Cadam: Confidence-based optimization for online learning. CoRR, abs/2411.19647, 2024.

[WRH90] Andreas S. Weigend, David E. Rumelhart, and Bernardo A. Huberman. Generalization by weight-elimination with application to forecasting. In Advances in Neural Information Processing Systems 3, NIPS Conference, pages 875–882, 1990.

[XL24] Shuo Xie and Zhiyuan Li. Implicit bias of adamw: ℓ∞-norm constrained optimization. In Forty-first International Conference on Machine Learning, ICML 2024, 2024.

[XXZ+23] Zeke Xie, Zhiqiang Xu, Jingzhao Zhang, Issei Sato, and Masashi Sugiyama. On the overlooked pitfalls of weight decay and how to mitigate them: A gradient-norm perspective. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, 2023.

[XZL+24] Xingyu Xie, Pan Zhou, Huan Li, Zhouchen Lin, and Shuicheng Yan. Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. IEEE Trans. Pattern Anal. Mach. Intell., 46(12):9508–9520, 2024.

[YGS+21] Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa, Kurt Keutzer, and Michael W. Mahoney. ADAHESSIAN: an adaptive second order optimizer for machine learning. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, pages 10665–10673, 2021.

[YLY+25] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, 2025.

[ZCG+24] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2299–2314, 2024.

[ZCL+25] Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Diederik P. Kingma, Yinyu Ye, Zhi-Quan Luo, and Ruoyu Sun. Adam-mini: Use fewer learning rates to gain more. In The Thirteenth International Conference on Learning Representations, ICLR 2025, 2025.

[ZMB+25] Rosie Zhao, Depen Morwani, David Brandfonbrener, Nikhil Vyas, and Sham M. Kakade. Deconstructing what makes a good optimizer for autoregressive language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, 2025.

[ZRS+18] Manzil Zaheer, Sashank J. Reddi, Devendra Singh Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive methods for nonconvex optimization. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, pages 9815–9825, 2018.

[ZZWL24] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An opensource small language model. CoRR, abs/2401.02385, 2024.

### A Notation and Definitions

N := {1,2,3,...} denotes the natural numbers. For n ∈ N, [n] denotes the set {1,2,...,n}. Vectors are denoted in lowercase boldface, and matrices are denoted in capital boldface. 0 and 1 denote the all-zeros and all-ones tensors of appropriate dimension, respectively. Scalar operations and functions, e.g. multiplication, division, and square roots, are understood to be performed entrywise when applied to vectors. We also use ⊙ to explicitly denote the entrywise product. x+ denotes the positive part of x, i.e.

x if x > 0 0 otherwise

x+ := max(0,x) =

.

∥·∥p denotes the ℓp norm for p ∈ [1,∞]. ⟨·,·⟩ denotes the standard inner product on Rd. [x]i denotes the ith entry of a vector x. diag (x) denotes the diagonal matrix with diagonal entries given by x. I(x ≥ 0) denotes the indicator tensor that is 1 in a coordinate if x is nonnegative in that coordinate and 0 otherwise. If K : Rd → R is convex, we let ∂K(x) denote the set of subgradients of K at x and overload ∇K(x) to denote an element of ∂K(x).

- Definition 1 (L-smoothness). A function f : Rd → R is L-smooth if it is differentiable and

∥∇f(y) − ∇f(x)∥2 ≤ L∥y − x∥2 for all x,y ∈ Rd. If f is L-smooth, then

f(y) ≤ f(x) + ⟨∇f(x),y − x⟩ +

L 2 ∥y − x∥22 for all x,y ∈ Rd.

- Definition 2 (Coerciveness). A function f : Rd → R is coercive if f(x) → ∞ as ∥x∥ → ∞.

### B Pseudocode of Optimizers with CWD

##### B.1 SGD with momentum

- Algorithm 2 SGD with momentum and cautious weight decay

- 1: given learning rates {ηt}t∈N ⊂ R>0, momentum coefficient β ∈ [0, 1), weight decay coefficient λ > 0

- 2: initialize time step t ← 1, parameters x1 ∈ Rd, first moment m0 ← 0
- 3: repeat
- 4: gt ← StochasticGradient(xt)
- 5: mt ← βmt−1 + (1 − β)gt
- 6: xt+1 ← xt − ηt mt +λI(mtxt ≥ 0)xt ▷ entrywise multiplication

- 7: t ← t + 1
- 8: until stopping criterion is met
- 9: return optimized parameters xt

B.2 Lion-K

- Algorithm 3 Lion-K with cautious weight decay

- 1: given learning rates {ηt}t∈N ⊂ R>0, momentum coefficients β1, β2 ∈ [0, 1), convex K : Rd → R with subgradient ∇K, weight decay coefficient λ > 0

- 2: initialize time step t ← 1, parameters x1 ∈ Rd, first moment m1 ← 0
- 3: repeat
- 4: gt ← StochasticGradient(xt)
- 5: mt+1 ← β2mt − (1 − β2)gt
- 6: mt+1 ← β1mt − (1 − β1)gt
- 7: xt+1 ← xt + ηt ∇K( mt+1) −λI(∇K( mt+1)xt ≤ 0)xt ▷ entrywise multiplication

- 8: t ← t + 1
- 9: until stopping criterion is met
- 10: return optimized parameters xt

B.3 Lion

- Algorithm 4 Lion with cautious weight decay

- 1: given learning rates {ηt}t∈N ⊂ R>0, momentum coefficients β1, β2 ∈ [0, 1), weight decay coefficient λ > 0

- 2: initialize time step t ← 1, parameters x1 ∈ Rd, first moment m0 ← 0
- 3: repeat
- 4: gt ← StochasticGradient(xt)
- 5: mt ← β1mt−1 + (1 − β1)gt
- 6: xt+1 ← xt − ηt sgn( mt) +λI( mtxt ≥ 0)xt ▷ entrywise sgn and multiplication

- 7: mt ← β2mt−1 + (1 − β2)gt
- 8: t ← t + 1
- 9: until stopping criterion is met
- 10: return optimized parameters xt

##### B.4 Muon

- Algorithm 5 Muon with cautious weight decay

- 1: given learning rates {ηt}t∈N ⊂ R>0, momentum coefficient β ∈ [0, 1), weight decay coefficient λ > 0

- 2: initialize time step t ← 1, parameters X1 ∈ Rn×m, first moment M0 ← 0
- 3: repeat
- 4: Gt ← StochasticGradient(Xt)
- 5: Mt ← βMt−1 + Gt
- 6: Ot ← NewtonSchulz(Mt) ▷ approximation of matrix sign
- 7: Xt+1 ← Xt − ηt Ot +λI(OtXt ≥ 0)Xt ▷ entrywise matrix multiplication

- 8: t ← t + 1
- 9: until stopping criterion is met
- 10: return optimized parameters Xt

B.5 Adam

- Algorithm 6 Adam with cautious weight decay

- 1: given learning rates {ηt}t∈N ⊂ R>0, momentum coefficients 0 ≤ β1 ≤ β2 < 1, numerical stability constant ϵ ≥ 0, weight decay coefficient λ > 0

- 2: initialize time step t ← 1, parameters x1 ∈ Rd, first moment m0 ← 0, second moment v0 ← 0
- 3: repeat
- 4: gt ← StochasticGradient(xt)
- 5: mt ← β1mt−1 + (1 − β1)gt
- 6: vt ← β2vt−1 + (1 − β2)gt2 ▷ entrywise multiplication
- 7: mt ← (1 − β1t)−1mt
- 8: vt ← (1 − β2t)−1vt
- 9: xt+1 ← xt − ηt √ mt vt+ϵ1

+λI(mtxt ≥ 0)xt ▷ entrywise operations

- 10: t ← t + 1
- 11: until stopping criterion is met
- 12: return optimized parameters xt

### C Fixed-Point Analysis

Revisiting the fixed-point analysis in Section 2.2 for AdamW, suppose the trajectory of AdamW converges to a fixed point (x⋆, m⋆, v⋆), so that m⋆ = ∇f(x⋆) and v⋆ = ∇f(x⋆)2. Passing to the limit ϵ ↘ 0, the fixed-point condition gives

∇f(x⋆) |∇f(x)⋆| + ϵ1

+ λx⋆ → sgn(∇f(x⋆)) + λx⋆ = 0.

Taking inner products with ∇f(x⋆) yields ∥∇f(x⋆)∥1 + ⟨λx⋆,∇f(x⋆)⟩ = 0, which shows that x⋆ is a Karush–Kuhn–Tucker (KKT) point of the constrained optimization problem

1 λ

f(x) s.t. ∥x∥∞ ≤

(3)

min

x∈Rd

by Lemma 3.8 of [XL24]. Intuitively, AdamW normalizes the gradient to its coordinatewise sign at stationarity and then balances it against the linear pull of the decoupled weight decay, which

enforces a box constraint on the parameters. [XL24] formalize this intuition and show that whenever the iterates of AdamW converge, the limit point is a KKT point of the box-constrained problem (3). However, this guarantee holds only under the assumption of convergence, and AdamW is not known to converge in general.

We remark that we can adapt this argument for another, more heuristic insight into why optimizers with cautious weight decay perform unbiased optimization. Suppose Adam with cautious weight decay reaches a fixed point, so that

∇f(x⋆) |∇f(x⋆)| + ϵ1

= −λI(∇f(x⋆)x⋆ ≥ 0)x⋆.

For a fixed point of Lion-K with cautious weight decay, we have −∇K(−∇f(x⋆)) = λI(∇K(−∇f(x⋆))x⋆ ≤ 0)x⋆.

In either situation, casework on the signs of the update and x⋆ shows that both sides must be 0. It follows that ∇f(x⋆) = 0 for Adam and ∇K(−∇f(x⋆)) = 0 for Lion-K, and if K is a convex function that achieves a unique minimum at 0 (e.g. a norm), then this condition becomes ∇f(x⋆) = 0 as well. Hence, the fixed-point analysis suggests that Adam and Lion-K with cautious weight decay find a stationary point of the original objective f.

### D Lyapunov Functions

Throughout this section, vector variables are implicitly dependent on t when clear from context, and we drop the subscript for notational simplicity.

- D.1 SGD SGD with cautious weight decay admits the continuous-time dynamics

x˙ = −∇f(x) − λI(∇f(x)x ≥ 0)x, which has a Lyapunov function H(x) = f(x), since

dH dt

= ⟨∇f(x),−∇f(x) − λI(∇f(x)x ≥ 0)x⟩ = −∥∇f(x)∥22 − λ (∇f(x)x)+ 1 ≤ 0.

- D.2 SGD with momentum

When SGD is equipped with momentum [SMDH13] and cautious weight decay, the continuous-time dynamics becomes

which has a Lyapunov function

x˙ = −m − λI(mx ≥ 0)x m˙ = β(∇f(x) − m),

- 1

- 2 ∥m∥22 + λ (mx)+ 1 ,

H(x,m) = βf(x) +

since

dH dt

= ⟨β∇f(x) + λI(mx ≥ 0)m,−m − λI(mx ≥ 0)x⟩ + ⟨m + λI(mx ≥ 0)x,β(∇f(x) − m)⟩

= − λI(mx ≥ 0) + β1,m2 − λ(β + λ) (mx)+ 1 ≤ 0.

##### D.3 Lion-K

We assume that K is convex and satisfies sgn(∇K(m)) = sgn(m) for all m ∈ Rd. This assumption is mild and that holds for every example of K given by [CLLL24].

The continuous-time dynamics of Lion-K without gradient enhancement is given by

x˙ = ∇K(m) − λx m˙ = −α∇f(x) − γm.

(4)

[CLLL24] showed that this system has a Lyapunov function H(x,m) = αf(x) +

γ λK∗(λx) + K∗(λx) + K(m) − ⟨m,λx⟩,

thereby elucidating the origin of the K∗(λx) regularization term. However, when equipped with cautious weight decay, (4) becomes

and admits a Lyapunov function

x˙ = ∇K(m) − λI(mx ≤ 0)x m˙ = −α∇f(x) − γm

(5)

H(x,m) = αf(x) + K(m) + λ (−mx)+ 1 , (6)

which corresponds to optimizing the original objective f. To see that (6) is a Lyapunov function for (5), note that

dH dt

= ⟨α∇f(x) − λI(mx ≤ 0)m,∇K(m) − λI(mx ≤ 0)x⟩

+ ⟨∇K(m) − λI(mx ≤ 0)x,−α∇f(x) − γm⟩

= −⟨∇K(m) − λI(mx ≤ 0)x,(λI(mx ≤ 0) + γ1)m⟩

= −⟨λI(mx ≤ 0) + γ1,∇K(m)m⟩ − λ(λ + γ) (−mx)+ 1 ≤ 0.

##### D.4 Adam

The continuous-time limit of Adam with cautious weight decay yields the system of ordinary differential equations (cf. [BB21])

(1 − exp(−αt))−1m

(1 − exp(−γt))−1v + ϵ1 − λI(mx ≥ 0)x m˙ = α(∇f(x) − m)

x˙ = −

v˙ = γ(∇f(x)2 − v).

(7)

We assume that 0 < γ ≤ 4α, which is satisfied by standard implementations of Adam in practice. This system admits the Lyapunov function

αtm2 2(√γtv + ϵ1) 1

+ λ (mx)+ 1 , (8) where

H(x,m,v,t) = αf(x) +

αt := (1 − exp(−αt))−1 and γt := (1 − exp(−γt))−1. To see that (8) is a Lyapunov function for (7), note that H is lower bounded by αf⋆ and

dH dt

∂H ∂t

= ⟨∇xH,x˙⟩ + ⟨∇mH,m˙ ⟩ + ⟨∇vH,v˙ ⟩ +

αtm √γtv + ϵ1 − λI(mx ≥ 0)x

= α∇f(x) + λI(mx ≥ 0)m,−

αt√γtm2 4√v √γtv + ϵ1 2

αtm √γtv + ϵ1

,γ(∇f(x)2 − v)

+ λI(mx ≥ 0)x,α(∇f(x) − m) −

+

2αexp(−αt)(√γtv + ϵ1) − αt−1γ exp(−γt)γt√γtv 2 αt−1(√γtv + ϵ1) 2

###### m2

−

2 ·

###### ,1

αtγ√γtm2∇f(x)2 4√v √γtv + ϵ1 2

αtm2 √γtv + ϵ1

+ λ(α + λ)(mx)+ +

= − (α1 + λI(mx ≥ 0))

###### ,1

2α exp(−αt)(√γtv + ϵ1) − αt−1γ exp(−γt)γt√γtv 2 αt−1(√γtv + ϵ1) 2

αtγm2√γtv 4 √γtv + ϵ1 2

###### m2

,1 −

2 ·

###### ,1

+

αtm2 √γtv + ϵ1 −

αt(2αtα exp(−αt) − γtγ exp(−γt))m2 4(√γtv + ϵ1)

γ 4 − α 1 − λI(mx ≥ 0),

≤

###### ,1

αtm2 √γtv + ϵ1 ≤ 0,

γ 4 − α −

α 2(exp(αt) − 1)

γ 4(exp(γt) − 1)

1 − λI(mx ≥ 0),

=

+

where the first inequality drops some nonpositive terms and uses √γtv ≤

√γtv+ϵ1 and the second inequality uses

γ 4 − α −

α 2(exp(αt) − 1)

γ

4(exp(γt) − 1) ≤ 0 for 0 < γ ≤ 4α and t > 0.

+

- Remark 1. Cautious weight decay can be seen as an attempt to fix the asymptotic instability of AdamW via a Lyapunov function. Consider the simplified continuous-time AdamW dynamics

###### m

√v − λx m˙ = ∇f(x) − m v˙ = ∇f(x)2 − v

x˙ = −

(9)

and the function

###### m2

2√v 1

H(x,m,v) = f(x) +

+ ⟨m,λx⟩.

By straightforward computation,

dH dt

m2 4v32

###### m √v

###### m

√v − λx +

+ λx,∇f(x) − m + −

= ∇f(x) + λm,−

m2 √v

m2∇f(x)2 4v23

- 3

- 4

= − λ +

+ λ(λ + 1)mx +

###### ,1

m2 √v 1 − λ(λ + 1)⟨m,x⟩ −

m2∇f(x)2 v32 1

1 4

- 3

- 4

= − λ +

.

,∇f(x)2 − v

Note that H is not guaranteed to be lower bounded and −ddHt is not guaranteed to be nonnegative, since ⟨m,x⟩ has unknown sign. This motivates the introduction of a mask I(mx ≥ 0) to the weight

decay term and a slight adjustment to H so that the result is a Lyapunov function for (9).

- Remark 2. For expositional clarity, we treat the ODEs and Lyapunov candidates in this section as smooth, even though the dynamics include the discontinuous indicator function I(ux ≥ 0). A fully rigorous analysis can be developed by interpreting the systems in the sense of differential inclusions, specifically, using Filippov’s framework [Fil88], and by applying specialized tools from nonsmooth Lyapunov stability theory to obtain convergence guarantees [SP94, BC99].

### E Convergence Rate of Adam with Cautious Weight Decay

In this section, we show that under the following assumptions, Adam with cautious weight decay (Algorithm 6) achieves a convergence rate on the squared gradient norm and an additional stationarity measure.

- Assumption 1 (Smoothness). f is L-smooth and lower bounded by a finite constant f⋆.
- Assumption 2 (Bounded variance). The stochastic gradient gt satisfies

E[gt | xt] = ∇f(xt) and Var(gt) = E ∥gt − ∇f(xt)∥22 | xt ≤

σ2 nbatch

,

where σ is a constant and nbatch denotes the batch size.

- Assumption 3 (Bounded iterates and bounded gradients). There exist constants R and G such that ∥xt∥∞ ≤ R and ∥gt∥∞ ≤ G a.s. for all t ∈ N.

Assumptions 1 and 2 are standard and often used in the analysis of stochastic gradient algorithms [GL13, BB21, DBBU22, ACD+23]. Assumption 3 can be justified using the Lyapunov function (8) if f is additionally assumed to be coercive, since a Robbins–Siegmund argument with sufficiently small η shows that the optimizer states remain in a compact sublevel set of H a.s. For the sake of clarity, here we take it as an explicit assumption. Similar assumptions have often been used for the analysis of Adam-style algorithms [KB15, RKK18, ZRS+18, CLSH19, DBBU22, CSZL22].

Theorem 1. Under Assumptions 1, 2, and 3, let 0 ≤ β1 ≤ β2 < 1, λ ≥ 0, ϵ > 0, and ηt = η > 0, and suppose xt is updated using Algorithm 6. Then for all T ∈ N,

K2 T

K4σ √nbatch

1 T

K1 ηT

E ∥∇f(xt)∥22 + λ (∇f(xt)xt)+ 1 ≤

+

+ K3η +

,

t∈[T]

where K1, K2, K3, and K4 are constants depending only on L, R, G, d, ϵ, λ, β1, β2, and f(x1)−f⋆.

- Remark 3. The first term on the left-hand side, ∥∇f(xt)∥22, reflects how much f is optimized, while the second term, ∥(∇f(xt)xt)+∥1, reflects the degree of conflict between the objective f and the parameter magnitudes. If ∇f(xt)xt ≫ 0, then there is room to jointly decrease both f and

the magnitudes. Thus, a small value of ∥(∇f(xt)xt)+∥1 indicates that the optimizer has reached a state where it is difficult to further decrease f and shrink the magnitudes simultaneously. This

corresponds to convergence toward a Pareto front, where trade-offs between the two objectives become unavoidable.

- Remark 4. In the setting of Theorem 1, let T ∈ N and η = Θ √ 1T . Then

1 √

σ √nbatch

1 T

E ∥∇f(xt)∥22 + λ (∇f(xt)xt)+ 1 = O

+

.

T

t∈[T]

An O(T−12) bound can be obtained by making the unrealistic assumption nbatch = Θ(T). However, even without this assumption, the stated bound is of theoretical interest. For additional discussion, see [BWAA18, ZRS+18, CLLL24].

- Lemma 1. For all t ∈ N, mt

√ vt + ϵ1 ∞ ≤

1 − β1 1 − β2

=: C.

Proof. It suffices to work in an arbitrary coordinate i. Let m := [ mt]i, v := [ vt]i, and gt := [gt]i. By expanding the update rules for m and v, we obtain

m =

1 − β1 1 − β1t

k∈[t]

β1t−kgk and v =

1 − β2 1 − β2t

k∈[t]

β2t−kgk2.

Now by Cauchy–Schwarz, m2 v ≤

(1 − β1)2 (1 − β1t)2 ·

1 − β2t 1 − β2 ·

k∈[t]

β12 β2

t−k

≤

(1 − β1)2 (1 − β1t)2 ·

1 − β2t 1 − β2 ·

k∈[t]

β1t−k

=

(1 − β1)2 (1 − β1t)2 ·

1 − β2t 1 − β2 ·

1 − β1t 1 − β1

=

- 1 − β1

- 1 − β2 ·

1 − β2t 1 − β1t ≤

- 1 − β1

- 1 − β2

.

The conclusion follows from

m √v + ϵ ≤

m √v ≤

- 1 − β1

- 1 − β2

.

| |
|---|

Fact 1 (Lemma F.1, [BWAA18]). For all t ∈ N, i ∈ [d], and α1,α2,...,αt ∈ R,

E

 

 

k∈[t]

αk([gk]i − [∇f(xk)]i)

 

2  ≤

σ2 nbatch

k∈[t]

αk2.

- Lemma 2. For all t ∈ N,

β1ηLd(C + λR) 1 − β1

σd nbatch(1 + β1)

E[∥∇f(xt) − mt∥1] ≤ β1tGd +

+

.

Proof. Note that

β1t−k(gk − ∇f(xk)).

β1t−k(∇f(xk) − ∇f(xk+1)) + (1 − β1)

mt − ∇f(xt) = −β1t∇f(x1) +

k∈[t]

k∈[t−1]

(10) By smoothness, Lemma 1, and Assumption 3, we have

√

√

d∥xk+1 − xk∥2 ≤ ηLd(C + λR). (11) By Jensen’s inequality and Fact 1,

∥∇f(xk) − ∇f(xk+1)∥1 ≤

d∥∇f(xk) − ∇f(xk+1)∥2 ≤ L

  ≤ E

 

2 

 

 

 

β1t−k([gk]i − [∇f(xk)]i)

β1t−k([gk]i − [∇f(xk)]i)

E

k∈[t]

k∈[t]

σ2 nbatch

σ nbatch(1 − β12)

(β12)t−k ≤

≤

.

k∈[t]

(12)

Taking E[∥·∥1] of (10) and applying (11) and (12),

 

β1ηLd(C + λR) 1 − β1

E[∥∇f(xt) − mt∥1] ≤ β1t ∥∇f(x1)∥1 +

+ (1 − β1)E

σd nbatch(1 + β1)

β1ηLd(C + λR) 1 − β1

≤ β1tGd +

+

,

as desired.

k∈[t]

- Lemma 3. For all t ∈ N,

- E − ∇f(xt),

mt

√ vt + ϵ1 ≤ −

E ∥∇f(xt)∥22 G + ϵ

+

β1tG2d ϵ

+

β1ηGLd(C + λR) (1 − β1)ϵ

+

σGd ϵ nbatch(1 + β1)

.

Proof. We have

− ∇f(xt),

mt √ vt + ϵ1

= ∇f(xt) √ vt + ϵ1

,∇f(xt) − mt − ∇f(xt)

≤ −

1 G + ϵ ∥∇f(xt)∥22 + ∇f(xt) √ vt + ϵ1

,∇f(xt) − mt

≤ −

1 G + ϵ ∥∇f(xt)∥22 + ∇f(xt) √ vt + ϵ1 ∞ ∥∇f(xt) − mt∥1

The result follows by √ ∇ vf(xt)

t+ϵ1 ∞

≤ Gϵ and Lemma 2 .

| |
|---|

- Lemma 4. For all m,g,x ∈ R, |(I(mx ≥ 0) − I(gx ≥ 0))x| ≤ I(mg ≤ 0)|x|.

β1t−k(gk − ∇f(xk))

 

1

| |
|---|

Proof. If x = 0, then the inequality is trivially valid, so suppose x ̸= 0. We proceed by casework on the sign of mg. If mg > 0, then m and g have the same sign, and the conditions mx ≥ 0 and gx ≥ 0 are equivalent. Thus I(mx ≥ 0) − I(gx ≥ 0) = 0, and the inequality holds. If mg ≤ 0, then I(mg ≤ 0) = 1. It remains to show |(I(mx ≥ 0) − I(gx ≥ 0))x| ≤ |x|, which follows upon realizing I(mx ≥ 0) − I(gx ≥ 0) ∈ {−1,0,1}.

| |
|---|

- Lemma 5. For all t ∈ N,

β1ηLRd(C + λR) 1 − β1

σRd nbatch(1 + β1)

E[−⟨∇f(xt),I(mtxt ≥ 0)xt⟩] ≤ −E (∇f(xt)xt)+ 1 +β1tGRd+

+

.

Proof. We have −⟨∇f(xt),I(mtxt ≥ 0)xt⟩ = −⟨∇f(xt),I(xt∇f(xt) ≥ 0)xt + (I(mtxt ≥ 0) − I(xt∇f(xt) ≥ 0))xt⟩

= ⟨∇f(xt),(I(xt∇f(xt) ≥ 0) − I(mtxt ≥ 0))xt⟩ − (∇f(xt)xt)+ 1 ≤ ⟨|∇f(xt)|,|(I(xt∇f(xt) ≥ 0) − I(mtxt ≥ 0))xt|⟩ − (∇f(xt)xt)+ 1 ≤ ⟨|∇f(xt)|,I(mt∇f(xt) ≤ 0)|xt|⟩ − (∇f(xt)xt)+ 1 ,

(13) where the fourth line uses Lemma 4. Taking the expectation of (13) conditioned on xt and expanding the inner product,

E[⟨|∇f(xt)|,I(mt∇f(xt) ≤ 0)|xt|⟩ | xt] = ⟨|∇f(xt)|,E[I(mt∇f(xt) ≤ 0) | xt]|xt|⟩ =

|[∇f(xt)]i[xt]i| · E[I([mt]i[∇f(xt)]i ≤ 0) | xt]

i∈[d]

|[∇f(xt)]i[xt]i| · Pr([mt]i[∇f(xt)]i ≤ 0 | xt)

=

i∈[d]

≤

|[∇f(xt)]i[xt]i| · Pr(|[∇f(xt)]i − [mt]i| ≥ |[∇f(xt)]i| | xt)

i∈[d]

|[xt]i| · E[|[∇f(xt)]i − [mt]i| | xt]

≤

i∈[d]

≤ R · E[∥∇f(xt) − mt∥1 | xt],

(14) where the fifth line uses Markov’s inequality. Taking the expectation of (14) and applying Lemma 2,

β1ηLRd(C + λR) 1 − β1

E[−⟨∇f(xt),I(mtxt ≥ 0)xt⟩] ≤ −E (∇f(xt)xt)+ 1 +β1tGRd+

as desired. We are now ready to prove Theorem 1. Proof of Theorem 1. Let

mt √ vt + ϵ1

∆t := f(xt+1) − f(xt) and δt :=

+ λI(mtxt ≥ 0)xt.

σRd nbatch(1 + β1)

+

,

| |
|---|

By smoothness,

L 2 ∥xt+1 − xt∥22

∆t ≤ ⟨∇f(xt),xt+1 − xt⟩ +

η2L 2 ∥δt∥22

= −η ⟨∇f(xt),δt⟩ +

η2L 2 ∥δt∥22

###### mt

√ vt + ϵ1 − ηλ⟨∇f(xt),I(mtxt ≥ 0)xt⟩ +

= −η ∇f(xt),

η2L 2 ∥δt∥22 .

###### mt

η 1 − β1t ∇f(xt),

√ vt + ϵ1 − ηλ⟨∇f(xt),I(mtxt ≥ 0)xt⟩ +

= −

Taking the expectation of (15) and applying Lemmas 1, 3, and 5,

 −

 

E ∥∇f(xt)∥22 G + ϵ

β1tG2d ϵ

η 1 − β1t

β1ηGLd(C + λR) (1 − β1)ϵ

σGd ϵ nbatch(1 + β1)

E[∆t] ≤

+

+

+

β1ηLRd(C + λR) 1 − β1

σRd nbatch(1 + β1)

+ ηλ −E (∇f(xt)xt)+ 1 + β1tGRd +

+

+ η2L(C2d + λ2R2d).

(15)

(16)

without loss of generality. Rearranging (16), using 1 − β1t ≤ 1 and (1 − β1t)(G + ϵ) ≥ 1, summing over T iterations, and dividing both sides by T gives

We can assume G ≥ 1−1β

1

β1tG2d ϵ

1 T

G + ϵ T

β1ηGLd(C + λR)(G + ϵ) (1 − β1)ϵ

G + ϵ ηT

(f(x1) − f⋆) +

E[S(xt)] ≤

+

t∈[T]

t∈[T]

σGd(G + ϵ) ϵ nbatch(1 + β1)

λ(G + ϵ) T

λσRd(G + ϵ) nbatch(1 + β1)

β1tGRd +

+

+

t∈[T]

β1ηλLRd(C + λR)(G + ϵ) 1 − β1

+ ηL(G + ϵ)(C2d + λ2R2d)

+

K1 ηT

K2 T

K4σ √nbatch

≤

+

+ K3η +

,

where the fourth line uses t∈[T] β1t ≤ 1−β1β

and

1

S(xt) := ∥∇f(xt)∥22 + λ (∇f(xt)xt)+ 1

- K1 := (G + ϵ)(f(x1) − f⋆)
- K2 :=

β1Gd(G + ϵ) 1 − β1

G ϵ

+ λR

- K3 :=

β1Ld(C + λR)(G + ϵ) 1 − β1

G ϵ

+ λR + Ld(C2 + λ2R2)(G + ϵ)

- K4 :=

d(G + ϵ) √1 + β1

G ϵ

+ λR .

| |
|---|

Table 8: Hyperparameter configurations for the different model sizes. All models use an expansion factor of 8 and a vocabulary size of 100,864.

Hyperparameter 2.3B Model 986M Model 338M Model 111M Model Model Architecture Total Parameters 2,321.38M 985.89M 338.44M 110.55M Model Dimension 2048 1536 1024 512 Number of Layers 18 12 8 8 Number of Heads 8 8 8 8 Per Head Dimension 256 256 128 64 Sequence Length 2048 2048 2048 2048 Validation Setup Evaluation Batch Size 1024 512 128 256 Number of Eval Steps 2 4 4 8 Evaluation Interval 1000 steps 1000 steps 500 steps 500 steps

### F Model & Experiment Configurations

We evaluate cautious weight decay (CWD) across two experimental setups: (1) transformer models ranging from 111M to 2.3B parameters, and (2) the OLMo-1B architecture. All models employ SwiGLU activations and rotary position embeddings (RoPE). To ensure fair comparison, we conduct extensive grid searches to optimize hyperparameters for each baseline optimizer (AdamW, Lion, and Muon) before applying CWD with identical settings. Table 8 details the scaled model configurations, Table 9 presents the OLMo-1B architecture, and the following subsection describes our hyperparameter search methodology.

We conducted an extensive grid search to determine optimal hyperparameters for AdamW, Lion, and Muon optimizers. Our learning rate search employed a quasi-logarithmic grid spanning four orders of magnitude from 1 × 10−5 to 1 × 10−1, with denser sampling in the critical 10−4 to 10−2 range where transformer models typically achieve optimal performance. The grid included standard decade values (e.g., 0.001, 0.01) as well as intermediate points within each logarithmic interval (e.g., 0.2, 0.3, 0.5, 0.8 scaled to each decade) to capture potential performance peaks between order-ofmagnitude boundaries, totaling 24 distinct learning rate values. For the learning rate schedule, we systematically evaluated warmup ratios of {0,0.05,0.1,0.2,0.3,0.4,0.5}, corresponding to 0% to 50% of total training steps dedicated to linear warmup, followed by cosine annealing decay. For AdamW, we additionally performed a grid search over the momentum parameters β1 and β2, evaluating combinations of β1 ∈ {0.85,0.9,0.95} and β2 ∈ {0.95,0.98,0.99,0.995,0.999}. Our experiments identified β1 = 0.9 and β2 = 0.95 as the optimal configuration. For Lion, we swept β1 ∈ {0.85,0.9,0.95} and β2 ∈ {0.95,0.98,0.99}, finding β1 = 0.9 and β2 = 0.95 to be optimal. For Muon, we similarly swept momentum coefficients and confirmed 0.95 as optimal.

Table 9: Model Architecture Configuration for OLMo-1B

Hyperparameter Value Architecture

Hidden dimension (dmodel) 2048 Number of attention heads 16 Number of layers 16 MLP ratio 8 Vocabulary size 50,280 Embedding size 50,304 Max sequence length 2048

Attention Mechanism Positional encoding RoPE Flash attention ✓ Multi-query attention ✗ ALiBi ✗ Attention dropout 0.0 Attention layer norm ✗

Model Components Activation function SwiGLU Block type Sequential Weight tying ✓ Include bias ✗ Layer norm type Default Layer norm with affine ✗ Residual dropout 0.0 Embedding dropout 0.0

Initialization Initialization method Mitchell Initialization device CUDA

### G Additional Experiment Results

This section provides supplementary experimental analyses that further characterize the behavior of cautious weight decay (CWD) across different optimizers and training dynamics. We present detailed visualizations of the mask activation patterns (Figure 7), showing how the fraction of parameters receiving weight decay evolves during training for both AdamW and Muon optimizers. Additionally, we include comprehensive loss and accuracy curves for all three optimizers (AdamW, Lion, and Muon) across model scales from 111M to 2.3B parameters (Figures 8–10), demonstrating consistent improvements with CWD. Finally, Figure 13 tracks the evolution of parameter norms throughout training, revealing that CWD maintains stable regularization comparable to standard weight decay while achieving superior performance.

###### (a) Full Training Trajectory

(b) Initial Phase (0 2500 Steps)

0.50

0.5

0.45

Ratio

Ratio

0.40

0.4

0.35

zoom(b)

0.30

0.3

###### Ratio (AdamW)

Ratio (AdamW)

0.25

0 10k 20k

0 500 1000 1500 2000 2500

Training Steps

Training Steps

(a) Full Training Trajectory

(b) Initial Phase (0 2500 Steps)

| | | |
|---|---|---|
| |zoom(b)| |
| | |Ratio (Muon)|

0.50

0.5

0.45

Ratio

Ratio

0.40

0.4

0.35

0.30

0.3

Ratio (Muon)

0.25

0 10k 20k

0 500 1000 1500 2000 2500

Training Steps

Training Steps

- Figure 7: Masked weight-decay activation ratio rt := ∥I(utxdt>0)∥1, i.e., the fraction of parameters for which the sign-selective mask is active at step t (d = number of parameters). Left: AdamW; right: Muon. Insets zoom into the first 2.5k steps to highlight early-training behavior. Model: Qwen-0.6B [YLY+25] trained on The Pile [GBB+21].

Accuracy (higher is better) GPT AdamW Lion Muon Model Size Ours Base Ours Base Ours Base

338M 0.4232 0.4221 0.4230 0.4211 0.4256 0.4252 986M 0.4566 0.4556 0.4552 0.4545 0.4589 0.4575 2B 0.4847 0.4831 0.4839 0.4830 0.4873 0.4858

Loss (lower is better) GPT AdamW Lion Muon Model Size Ours Base Ours Base Ours Base

338M 3.0059 3.0136 3.0012 3.0121 2.9851 2.9896 986M 2.7053 2.7142 2.7171 2.7231 2.6873 2.6968 2B 2.4881 2.4973 2.4961 2.5012 2.4703 2.4803

Table 10: Final evaluation accuracy (higher is better) and loss (lower is better) comparisons across different model sizes, expanded to the full text width. Our proposed method is benchmarked against three baseline optimizers: AdamW, Lion, and Muon. The best result in each pair is bolded.

3.0

3.5

Ours

Ours

Ours

3.5

3.4

Muon

Muon

Muon

2.9

3.4

3.3

2.8

3.2

3.3

Loss

Loss

Loss

3.1

2.7

3.2

3.0

2.6

2.9

3.1

2.8

2.5

3.0

2.7

0 5000 10000 15000 20000

0 5000 10000 15000

0 5000 10000 15000 20000

Step

Step

Step

(a) 338M parameters

(b) 986M parameters

(c) 2B parameters

###### Figure 8: Training dynamics across model scales with Muon optimizer. Baseline Muon (dashed) vs. Muon with CWD (solid).

Eval Loss Comparison

3.6

Ours

3.5

AdamW

3.4

Loss

3.3

3.2

3.1

3.0

0 5000 10000 15000 20000

Step

(a) 338M parameters

Eval Loss Comparison

3.3

Ours

3.2

AdamW

3.1

Loss

3.0

2.9

2.8

2.7

0 5000 10000 15000

Step

(b) 986M parameters

Eval Loss Comparison

3.1

Ours

3.0

AdamW

2.9

Loss

2.8

2.7

2.6

2.5

0 5000 10000 15000 20000

Step

(c) 2B parameters

###### Figure 9: Training dynamics across model scales with AdamW optimizer. We compare baseline AdamW (dashed) against AdamW with CWD (solid) on models ranging from 338M to 2B parameters.

3.8

3.2

3.0

Ours

Ours

Ours

3.7

Lion

Lion

Lion

2.9

3.1

3.6

3.5

2.8

3.0

Loss

Loss

Loss

3.4

2.7

2.9

3.3

3.2

2.6

2.8

3.1

2.5

3.0

0 5000 10000 15000 20000

0 5000 10000 15000

0 5000 10000 15000 20000

Step

Step

Step

(a) 338M parameters

(b) 986M parameters

(c) 2B parameters

###### Figure 10: Training dynamics across model scales with Lion optimizer. Baseline Lion (dashed) vs. Lion with CWD (solid).

Eval Loss

Training Loss

RMS Norm of Params

3.3

3.3

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Ours

RMSNormofParams

3.2

3.2

AdamW

0.06

Adam

3.1

3.1

Loss

Loss

3.0

3.0

0.04

2.9

2.9

2.8

2.8

2.7

0.02

2.7

0 5000 10000 15000

0 5000 10000 15000

0 5000 10000 15000

Step

Step

Step

###### Figure 11: Training dynamics for the 986M-parameter Gemma model.

#### 111M 338M 986M 2B

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

AdamW

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

Lion

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Muon

- Figure 12: Comparison of gradient norms using RMS normalization across four model sizes: 111M, 338M, 986M, and 2B. All models are trained under Chinchilla settings. CWD achieves lower gradient norms across all configurations.

block_0/attn/o_proj

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | |our|s| | | | | | |
| | | |ada ada|mw m| | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.050

0.045

0.040

value

0.035

0.030

0.025

0.020

0.015

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/ffn_0_gate/w

0.07

ours

0.06

adamw

adam

0.05

value

0.04

0.03

0.02

0.01

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/pre_ln_0/scale

1.0

0.8

ours

value

adamw

0.6

adam

0.4

0.2

0 2500 5000 7500 10000 12500 15000 17500

step

block_1/attn/qkv_proj

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | |our ada|s<br><br>mw| | | | | | |
| | | | | | | | | | | |
| | | |ada|m| | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.07

0.06

value

0.05

0.04

0.03

0.02

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/attn/per_dim_scale/scale

0.060

ours

0.4

0.055

adamw

0.050

adam

0.3

0.045

value

value

0.040

0.2

0.035

0.1

0.030

0.025

0.0

0.020

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/ffn_1/w

1.0

ours

0.06

adamw

0.8

adam

0.05

value

value

0.6

0.04

0.4

0.03

0.02

0.2

0.01

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/pre_ln_1/scale

1.0

0.06

0.8

ours

0.05

value

value

0.6

adamw

adam

0.04

0.4

0.03

0.2

0.02

0 2500 5000 7500 10000 12500 15000 17500

step

block_1/ffn_0/w

0.08

0.08

ours

0.07

0.07

adamw

0.06

0.06

adam

value

value

0.05

0.05

0.04

0.04

0.03

0.03

0.02

0.02

0.01

0.01

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/attn/qkv_proj

ours

adamw

adam

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/post_ln_0/scale

ours

adamw

adam

0 2500 5000 7500 10000 12500 15000 17500

step

block_1/attn/o_proj

ours

adamw

adam

0 2500 5000 7500 10000 12500 15000 17500

step

block_1/ffn_0_gate/w

ours

adamw

adam

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/ffn_0/w

ours

0.06

adamw

adam

0.05

value

0.04

0.03

0.02

0.01

0 2500 5000 7500 10000 12500 15000 17500

step

block_0/post_ln_1/scale

1.0

0.9

ours

0.8

value

adamw

0.7

adam

0.6

0.5

0.4

0 2500 5000 7500 10000 12500 15000 17500

step

block_1/attn/per_dim_scale/scale

ours

1.0

adamw

0.8

adam

value

0.6

0.4

0.2

0.0

0 2500 5000 7500 10000 12500 15000 17500

step

block_1/ffn_1/w

ours

0.07

adamw

0.06

adam

0.05

value

0.04

0.03

0.02

0.01

0 2500 5000 7500 10000 12500 15000 17500

step

###### Figure 13: Evolution of parameter norm (RMS) during training for a 986M parameter model. We compare three optimization strategies: AdamW with weight decay 0.1 (orange), our proposed method (blue), and Adam without weight decay (green). Our method maintains stable parameter norms comparable to AdamW while achieving improved performance.

