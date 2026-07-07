# arXiv:2512.12602v4[cs.LG]8May2026

Exact Flow Linear Attention: Exact Solution from Continuous-Time Dynamics

Jingdi Lei1, Di Zhang2, Soujanya Poria1 1Nanyang Technological University, 2Fudan University

In this paper, we introduce Exact Flow Linear Attention (EFLA), an exact-flow formulation of deltarule linear attention. We show that the delta-rule update can be interpreted as an explicit Euler discretization of an underlying continuous-time system. EFLA replaces this first-order update with the exact closed-form flow. By exploiting the rank-1 structure of the dynamics matrix, both the matrix exponential and the input integral collapse to a simple update that preserves delta-rule linear attention’s algebraic structure, parameter count, linear-time complexity, and chunkwise parallelism. This attention mechanism removes the Euler discretization error of the delta-rule dynamics without introducing additional parameters. Experiments on robustness tests, language modeling benchmarks, and the MAD synthetic benchmark show that EFLA improves stability under corrupted and highenergy inputs, reduces perplexity, and achieves stronger downstream performance compared to SSM and Euler-style baselines. These results establish exact-flow integration as a principled and scalable update mechanism for delta-rule linear attention.

Github: https://github.com/declare-lab/EFLA Correspondence: Di Zhang (di.zhang@ustc.edu) Date: May 11, 2026

[Figure 1]

[Figure 2]

1 Introduction

As large language models (LLMs) evolve into increasingly capable agents (Yao et al., 2022; Team et al., 2025b; Google, 2025; OpenAI, 2025), inference efficiency has become a critical bottleneck (Dao et al., 2022; Kwon et al., 2023; Kim et al., 2024). This challenge is especially pronounced in long-context processing and reinforcement learning (RL) environments (Guo et al., 2025; Lai et al., 2025), where models are required to process extended reasoning trajectories, maintain long-range dependencies, and engage in complex tool-use interactions (Lightman et al., 2023; Yao et al., 2023). In these scenarios, the quadratic time complexity of standard softmax attention (Vaswani et al., 2017) introduces substantial computational overhead, constraining model throughput, scalability to long contexts, and real-time interactivity. (Liu et al., 2023; Jiang et al., 2024; Katharopoulos et al., 2020).

This has motivated a broad line of work on linear-time sequence models, including linear attention and state space models (SSMs). Among them, delta-rule linear attention (Schlag et al., 2021; Yang et al., 2024b) provides an appealing recurrent formulation by maintaining a matrix-valued associative memory. Compared with purely additive linear attention (Katharopoulos et al., 2020), the delta rule introduces an input-dependent correction mechanism: the memory is updated by partially erasing the component associated with the current key and injecting a new key–value association. This update admits efficient recurrent and chunkwise-parallel implementations (Yang et al., 2024b). Despite its empirical effectiveness, this update is usually motivated as a discrete online learning rule, leaving its underlying continuous-time structure less explored.

In this work, we revisit delta-rule linear attention from the perspective of numerical integration. Under a zero-order-hold (ZOH) assumption, where the current key and value are treated as fixed within each token interval, the delta-rule update can be interpreted as an explicit Euler discretization of a continuous-time system. This interpretation exposes a principled source of approximation error: the Euler update is only firstorder accurate. Although simple and efficient, Euler discretization can introduce accumulated discretization error and may become inaccurate when the effective dynamics are stiff, for example under large key norms or

large update scales. Existing methods often mitigate such issues by introducing gating mechanisms including decay factors, gating functions (Dao and Gu, 2024; Ma et al., 2022; Sun et al., 2023), or adaptive forgetting coefficients (Yang et al., 2023, 2024a; Team et al., 2025a). This motivates a natural question: instead of heuristically mitigating the instability of Euler-style updates, can we derive an exact update directly from the underlying continuous-time dynamics of delta-rule linear attention?

We answer this question by proposing Exact Flow Linear Attention (EFLA), a drop-in replacement for the delta-rule update that solves the corresponding ZOH continuous-time dynamics in closed form. From the first principle, we eliminate the discretization errors by solving the underlying ODE exactly. EFLA can be mathematically interpreted as the general solution of the first-order ODE, yielding a continuous-time, exact flow update of linear attention. The key observation is that the transition matrix is rank-1 in delta-rule update. Therefore, although a generic matrix exponential would be computationally expensive, both the exponential transition and the input integral collapse into a simple closed form. EFLA preserves the algebraic structure and computational order of delta-rule linear attention while removing the Euler discretization error of the underlying ZOH dynamics.

Since EFLA retains the same rank-1 update form as vanilla delta-rule linear attention, it can be integrated with existing hardware-efficient WY/UT-based chunkwise parallelization schemes. This allows EFLA to preserve linear-time recurrent inference and efficient parallel training. Empirically, we evaluate EFLA on robustness tests, language modeling benchmarks, and the MAD synthetic benchmark. Across these evaluations, EFLA consistently improves over previous Euler discretization baselines in both robustness, convergence, and downstream performance while maintaining comparable training throughput and introducing no additional parameters.

Our contributions are summarized as follows:

- • We reinterpret delta-rule linear attention as an explicit Euler discretization of an underlying ZOH continuous-time ODE, and propose Exact Flow Linear Attention (EFLA), which solves this ODE in closed form.
- • We show that the rank-1 structure of delta-rule learning makes the exact matrix exponential and input integral analytically tractable, yielding an exact-flow update with the same computational order and algebraic form as delta-rule linear attention.
- • We validate EFLA across robustness tests, language modeling benchmarks, and the MAD synthetic benchmark, showing consistent improvements over previous Euler-style baselines while maintaining comparable throughput.

- 2 Preliminary

Scaled Dot-Product Attention. Given queries Q ∈ Rn×d, keys K ∈ Rn×d, and values V ∈ Rn×d, scaled dot-product attention (Vaswani et al., 2017) is defined as:

Attn(Q,K,V) = softmax

#### QK⊤

+ M V, (1)

√

d

where M ∈ Rn×n is the additive causal mask, with zeros on and below the diagonal and −∞ above. LinearAttentionasAssociativeMemory. Linear attention (Katharopoulos et al., 2020) maintains a matrix-valued recurrent state that accumulates key–value associations:

St = St−1 + ktvt⊤, ot = S⊤t qt. (2)

From the fast-weight perspective (Schlag et al., 2021), St serves as an associative memory that stores transient mappings from keys to values. This additive update can be viewed as gradient descent on the unbounded correlation objective:

Lt(S) = −⟨S⊤kt,vt⟩. (3)

Since this objective only reinforces new key–value pairs and does not provide an explicit mechanism for erasing old associations, the accumulated memory may grow without bound and suffer from interference over long contexts.

Delta-rule Learning. Delta-rule linear attention addresses this issue by formulating the memory state as the parameter of an online reconstruction problem:

- 1

- 2

S⊤kt − vt 2 . (4) Taking one gradient step with learning rate βt gives:

Lt(S) =

St = St−1 − βt∇SLt(St−1) = I − βtktk⊤t St−1 + βtktvt⊤. (5)

This update corrects the associative memory toward the mapping kt  → vt. Its rank-1 correction structure enables WY-style products of rank-1 transformations, facilitating hardware-efficient chunkwise parallelization (Bischof and Van Loan, 1987; Yang et al., 2024b).

Euler Methods in ODEs. Given a first-order ordinary differential equation (ODE):

dS(τ) dτ

= f(τ,S(τ)), (6)

numerical integration methods approximate the solution at discrete time points. The explicit Euler method (Euler, 1792) updates the state by using the derivative at the current position:

St = St−1 + βtf(t − 1,St−1), (7)

where βt is the step size. Euler discretization is computationally simple but only first-order accurate, with local truncation error O(βt2). As shown below, Euler discretization delta-rule updates can be interpreted as such an explicit Euler discretization of an underlying continuous-time memory dynamics.

### 3 Method

- 3.1 Exact Flow of Delta-rule Linear Attention

We begin by revisiting DeltaNet (Schlag et al., 2021), which formulates linear attention as online gradient descent on a reconstruction objective:

- 1

- 2∥S⊤kt − vt∥2. (8)

Lt(S) =

Applying a single gradient descent step with learning rate βt yields:

St = St−1 + βt −ktk⊤t St−1 + ktvt⊤ . (9)

To formalize the underlying dynamics, we define the dynamics matrix At = ktk⊤t and the input forcing term bt = ktvt⊤. Since the input sequence is observed only at discrete time steps, we adopt the Zero-Order Hold (ZOH) (Iserles, 2009) assumption to construct a continuous-time interpolation. Under this assumption, At and bt are treated as piecewise constant within each update interval. The system evolves according to a first-order ODE:

dS(t) dt

= −AtS(t) + bt. (10)

It reveals that the standard Delta rule update is only a first-order numerical approximation to the underlying continuous dynamics. This observation motivates a natural question: can we avoid the Euler discretization error by solving the ODE exactly? Eq. (10) admits a closed-form matrix-exponential solution, yielding1:

βt

e−(β

t−τ)Atbt dτ. (11)

St = e−β

tAtSt−1 +

0

1The detailed derivation of this closed-form expression is provided in Appendix C.

- 3.2 Efficient Exact Flow via the Rank-1 Structure

While the infinite-order solution eliminates discretization error, naively evaluating the matrix exponential e−β

tAt for a general matrix typically requires O(d3) complexity (Gu et al., 2020). We bypass this computational

bottleneck by leveraging the rank-1 structure of the dynamics matrix At = ktk⊤t , which allows the exponential to be computed in linear time.

We observe that At satisfies the idempotence-like property Ant = λnt −1At for n ≥ 1, where λt = k⊤t kt (see Appendix B for the proof). This property allows us to collapse the Taylor series of the matrix exponential into a computable closed form:

e−β

tAt = I +

∞

n=1

(−βt)n n!

Ant = I −

1 − e−β

tλt λt

At. (12)

Substituting this transition operator into the integral term 0 βt e−(β

t−τ)Atbt dτ yields the exact input injection:

It =

βt

0

I −

1 − e−λ

t(βt−τ) λt

At bt dτ

= βtbt −

Atbt λt

βt −

1 − e−β

tλt λt

.

(13)

Crucially, since bt = ktvt⊤ and At = ktk⊤t , we have Atbt = λtbt. This algebraic relationship allows for significant simplification of the integral term:

It = βtbt − βtbt +

1 − e−β

tλt λt

bt =

1 − e−β

tλt λt

bt. (14)

Combining these results, the final Exact Flow Linear Attention update rule is given by:

St = I −

1 − e−β

tλt λt

ktk⊤t St−1 +

1 − e−β

tλt λt

ktvt⊤. (15)

Thus, EFLA has the same rank-1 algebraic form as the delta-rule update, with αt replacing βt. It removes the Euler discretization error of the ZOH dynamics while preserving linear complexity in the sequence length, namely O(Ld2) for a length-L sequence with a d × d memory state.

- 3.3 Chunkwise Parallelism Form

We observe that the EFLA update rule shares an identical algebraic structure with DeltaNet. Given this structural equivalence, we can seamlessly adapt the hardware-efficient parallelization strategies originally developed for DeltaNet (Yang et al., 2024b). In this section, we derive the chunkwise parallel formulation specifically for EFLA.

To derive the chunkwise parallel form, we first unroll the recurrence relation. Denoting 1−e−βtλt

λt = αt, the state update becomes:

 

t

St = (I − αtktk⊤t )St−1 + αtktvt⊤ =

i=1

Then we can define the following variables:

 αi(kivi⊤). (16)

t

(I − αjkjk⊤j )

j=i+1

Pji =

j

(I − αtktk⊤t ), Hji =

t=i

j

Pjt+1αtktvt⊤ (17)

t=i

where Pji = I when i > j. Pji can be considered as decay factor applied to Si to obtain Sj and Hji is an accumulation term to Sj from token i. The Chunkwise can be written as follows:

Sr[t] = Pr[t]S0[t] + Hr[t] (18)

where we define the chunkwise variables Sr[t] = StC+r, Pr[t] = PtCtC++1r, Hr[t] = HtCtC++1r. Here we have CL chunks of size C. We can use induction to derive the WY representations of Pr[t] and Hr[t]:

Pr[t] = I −

r

ki[t]w[it⊤] , Hr[t] =

i=1

r

ki[t]ui[t⊤] (19)

i=1

w[rt⊤] = α[rt] kr[t⊤] −

r−1

(kr[t⊤] ki[t])w[it⊤] , ur[t⊤] = α[rt] v[rt⊤] −

i=1

r−1

(kr[t⊤] ki[t])ui[t⊤] . (20)

i=1

subsequently, we can obtain the chunk-level recurrence for states and outputs:

Sr[t] = S0[t] −

r

ki[t]w[it⊤] S0[t] +

i=1

r

ki[t]ui[t⊤] = S0[t] +

i=1

r

ki[t] ui[t⊤] − w[it⊤] S0[t] . (21)

i=1

or[t] = Sr[t⊤] qr[t] = S0[t⊤] qr[t] +

r

(ui[t] − S0[t⊤] w[it])(ki[t⊤] qi[t]) (22)

i=1

letting S[t] = S0[t], the above can be simplified to matrix notations:

S[t+1] = S[t] + K⊤[t] U[t] − W[t]S[t] (23) O[t] = Q[t]S[t] + Q[t]K⊤[t] ⊙ M U[t] − W[t]S[t] , (24)

where □[t] = □1:[t]C ∈ RC×d for □ ∈ {Q,K,V,O,U,W} defines the chunkwise matrices that are formed from stacking the qt,kt,vt,ot,ut,wt vectors and M is the lower triangular causal mask.

Finally, we can apply the UT transform (Joffrain et al., 2006) to simplify the recurrence calculations of ur[t] and w[rt].

T[i] = I + StrictTril(diag(αt)K[i]K⊤[i])

−1

diag(αt) (25)

W[t] = T[t]K[t], U[t] = T[t]V[t] (26)

- 3.4 Discussion

Directional decay of memory. The EFLA formulation reveals the role of the key norm in controlling memory retention. Since At = ktk⊤t is symmetric positive semi-definite and rank-1, it has one non-zero eigenvalue λt = ∥kt∥2 when kt ̸= 0, while all remaining eigenvalues are zero. Consequently,

1 − e−β

tλt λt

e−β

ktk⊤t . (27)

tAt = I −

This transition contracts the component of the memory aligned with kt by the factor e−β

tλt, while leaving the orthogonal subspace unchanged. For βt ≥ 0, the eigenvalues of the homogeneous transition are e−β

tλt

##### and 1, so the homogeneous flow is non-expansive.

Connection to the delta rule. The effective update coefficient satisfies:

1 − e−β

tλt λt

= βt −

αt =

- 1

- 2

βt2λt + O(βt3λ2t). (28)

Therefore, when the effective update scale βtλt is small, EFLA recovers the delta-rule update up to higherorder terms. For larger βtλt, EFLA uses the bounded exponential flow, whereas the Euler-style delta rule keeps only the first-order approximation.

Role of key norm and stable dynamics. EFLA naturally exposes the role of the key norm through the exact-flow coefficient:

1 − e−β

tλt λt

, λt = ∥kt∥2. (29)

αt =

The denominator λt is not an additional normalization trick, but arises directly from solving the rank-1 ODE in closed form. This shows that the strength of the exact-flow update is intrinsically tied to the magnitude of the current key.

This dependence also affects the write term αtktvt⊤. When keys are L2-normalized, the model mainly uses the direction of kt. In contrast, EFLA preserves the magnitude of kt in the key–value injection term, allowing the current token to modulate the memory write not only through the learned gate βt, but also through the signal strength encoded in the key norm.

Finally, the continuous-time dynamics provide a stability interpretation. The homogeneous part of the ODE is governed by:

dS(τ) dτ

= −ktk⊤t S(τ), (30)

where −ktk⊤t is negative semi-definite. Therefore, the exact transition contracts the memory component aligned with kt by the factor e−β

tλt, while leaving the orthogonal subspace unchanged. This contractive flow suppress perturbations along the current key direction and provides a natural explanation for the improved stability and noise robustness observed in our experiments in Section 4.3.

GatedVariantofEFLA. EFLA is also compatible with gated recurrences by applying the exact-flow construction to a gated update mechanism. The derivation of this variant, which we term Gated EFLA, is provided in Appendix D.

### 4 Experiments

- 4.1 Experimental Setup

Baselines. We compare EFLA and its gated variants with SSM methods including Mamba-2 (Dao and Gu, 2024), and Euler-style methods including DeltaNet (Yang et al., 2024b), and Gated DeltaNet (Yang et al., 2024a). EFLA is implemented as a drop-in replacement for the delta-rule state update in Euler-style methods, using the exact-flow coefficient derived in Section 3.

Training setup. The 340M models are trained for 8B tokens, and the 1.3B models are trained for 50B tokens on the same subset of the SlimPajama dataset (Soboleva et al., 2023) with the Mistral tokenizer (Jiang et al., 2023). Detailed hyperparameter settings are provided in Appendix A.

Evaluation. We evaluate language modeling perplexity on Wikitext (Merity et al., 2016) and LAMBADA (Paperno et al., 2016), together with a suite of zero-shot commonsense reasoning tasks, including PiQA (Bisk et al., 2020), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), ARC-easy (ARC-e), ARC-challenge (ARC-c) (Clark et al., 2018), BoolQ (Clark et al., 2019), OpenBookQA (OBQA) (Mihaylov et al., 2018), and SciQ (Johannes Welbl, 2017). We report perplexity for Wikitext and LAMBADA, and accuracy for the downstream reasoning tasks.

Table 1 Main language modeling results. The 340M models are trained for 8B tokens, whereas the 1.3B models are trained for 50B tokens on the same subset of the SlimPajama dataset (Soboleva et al., 2023) with the Mistral (Jiang

- et al., 2023) tokenizer. Perplexity: Lower (↓) is better. Accuracy: Higher (↑) is better. Best results are bolded.

Perplexity (↓) Accuracy (↑)

Model Wiki. LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c BoolQ OBQA SciQ Avg.

ppl ppl acc acc acc_n acc acc acc_n acc acc_n acc

340M Parameters Vanilla

Mamba-2 36.90 105.69 20.7 61.9 30.7 50.2 40.4 22.1 53.0 27.0 71.9 42.0 DeltaNet 38.09 96.26 22.5 60.7 30.1 51.9 41.7 21.6 60.1 29.0 70.4 42.2 EFLA 35.26 79.97 23.9 61.0 30.9 51.1 42.5 23.8 59.7 30.8 72.9 44.1 Gated Variants

Gated DeltaNet 34.93 72.46 25.5 62.7 30.8 49.3 41.8 21.7 61.7 27.4 74.2 43.9 Gated EFLA 34.28 69.37 26.4 61.8 31.1 52.1 42.0 22.8 62.1 29.4 75.3 44.8

1.3B Parameters

DeltaNet 18.38 17.29 41.8 69.2 44.5 49.3 52.5 26.4 58.1 29.8 82.6 50.5 EFLA 18.30 16.54 43.2 68.9 44.5 52.1 54.4 26.4 60.4 31.6 84.2 51.8

- 4.2 Main Language Modeling Results

The main language modeling results are shown in Table 1. For 340M-parameter models trained with the same 8B-token budget, EFLA consistently improves over the Euler-style baseline on most metrics. On the perplexity metrics, EFLA reduces Wikitext perplexity from 38.09 to 35.26 and LAMBADA perplexity from 96.26 to 79.97, indicating better language modeling quality under the same training budget. On downstream accuracy, EFLA improves the average score from 42.2% to 44.1%, suggesting that the exact-flow update improves the general effectiveness of the Euler-style recurrence. The comparison with Mamba-2 further shows that EFLA is competitive among efficient recurrent architectures at this scale.

The same trend holds for the gated variants. Replacing the Gated DeltaNet update with the Gated EFLA update reduces Wikitext perplexity from 34.93 to 34.28 and LAMBADA perplexity from 72.46 to 69.37. It also improves the average downstream accuracy from 43.9% to 44.8%. This result is important because it shows that the exact-flow update is not limited to the vanilla formulation, but can also be incorporated into the gated variant and still produce consistent improvements.

At the larger 1.3B scale, the improvements persist under a longer 50B-token training budget. Compared with DeltaNet, EFLA reduces Wikitext perplexity from 18.38 to 18.30 and LAMBADA perplexity from 17.29 to 16.54. It also improves the average downstream accuracy from 50.5% to 51.8%. Therefore, the improvement is not a small-scale artifact, the exact-flow update continues to provide benefits when both model size and training tokens are increased.

- 4.3 Robustness under Corrupted and High-energy Inputs Robustness evaluation. As discussed in Section 3, Euler discretization delta-rule updates can be interpreted

- as explicit Euler discretizations of the corresponding ZOH continuous-time dynamics. While computationally efficient, this first-order discretization may become inaccurate under large effective update scales or corrupted inputs. To evaluate this behavior, we conduct three robustness tests on pixel-level Sequential MNIST (LeCun et al., 2010).

Perturbation settings. We consider three types of input perturbations. First, image pixel dropout applies Bernoulli dropout to input tokens with probability p, simulating information loss in corrupted inputs. Second, OOD intensity scaling amplifies input signals by a fixed factor, testing stability under high-energy inputs. Third, additive Gaussian noise injects random noise with varying standard deviations, evaluating robustness to signal corruption.

DeltaNet lr=1×10 EFLA lr=1×10

DeltaNet lr=1×10 EFLA lr=3×10

- Figure 1 EFLA outperforms DeltaNet in both convergence speed and robustness on sMNIST. The plots illustrate training dynamics and robustness against dropout, intensity scaling, and additive Gaussian noise. EFLA maintains higher accuracy as perturbation intensity increases, especially under the larger learning-rate setting 3 × 10−3.

Experimental setup. We flatten each 28×28 image into a sequence of length L = 784 and use a linear attention classifier with hidden dimension d = 64. We compare EFLA with the DeltaNet baseline. Both models are trained with AdamW using a batch size of 128.

Results. Figure 1 shows that DeltaNet is more sensitive to input scaling, with accuracy dropping rapidly as the scale factor increases. In contrast, EFLA maintains higher accuracy under large input scales, consistent with its exact-flow transition, whose homogeneous component remains non-expansive for non-negative decay. EFLA also outperforms DeltaNet under additive noise and pixel dropout, with a slower degradation rate under stronger perturbations. These results support the view that removing the Euler discretization error of the ZOH delta-rule state transition improves robustness under corrupted or high-energy inputs.

4.4 Robustness to Learning Rate Selection

[Figure 3]

(a) Training loss. (b) Accuracy.

[Figure 4]

- Figure 2 sMNIST results under different learning rates. EFLA maintains lower training loss and higher accuracy across the tested learning-rate range, indicating stronger robustness to learning-rate selection.

To further examine the optimization behavior of EFLA, we evaluate its sensitivity to the learning rate on the Sequential MNIST. We compare EFLA with DeltaNet under the same training setup while varying the learning rate over {1 × 10−4,3 × 10−4,10−3,3 × 10−3,1 × 10−2}.

- As the results shown in Figure 2, EFLA remains competitive across the entire learning-rate range. At very small learning rates, such as 1 × 10−4, both methods have relatively high training loss, but EFLA already achieves substantially higher accuracy than DeltaNet. As the learning rate increases, EFLA consistently obtains lower training loss and higher accuracy. Notably, when the learning rate is increased to 1 × 10−2,

DeltaNet shows no further accuracy improvement, whereas EFLA continues to achieve the best accuracy among all tested settings. These results suggest that EFLA is less sensitive to learning-rate selection than the Euler-style DeltaNet update. This behavior is consistent with our theoretical interpretation: by reducing the discretization error introduced at each recurrent state transition, EFLA provides a more faithful state update, which can improve optimization behavior across different learning-rate regimes.

- 4.5 Results on the MAD Synthetic Benchmark

Compress Fuzzy Recall

In-Context Recall

Memorize Noisy Recall

Selective Copy

Average

MAD benchmark task

0

20

40

60

80

100

Score

42.7

22.2

99.9

29.9

99.9 99.6

65.7

43.8

22.6

100

32.5

100 99.8

66.4

|DeltaNet<br><br>| |
|---|
<br><br>EFLA|
|---|

- Figure 3 Results on the synthetic MAD benchmark.

2K × 8 4K × 4 8K × 2 16K × 1

Training length × Batch size

- 35

- 36

- 37

- 38

- 39

- 40

- 41

Tokenspersecond(Ktok/s)

39.7 39.6 39.6

38.5

40.0 39.9

40.1

38.0

|DeltaNet<br><br>| |
|---|
<br><br>EFLA|
|---|

Figure4 Training throughput of DeltaNet and EFLA.

We further evaluate EFLA on the Mechanistic Architecture Design (MAD) benchmark (Poli et al., 2024), a suite of synthetic token-manipulation tasks designed to isolate architecture-level capabilities. As shown in Figure 3, EFLA improves upon DeltaNet across all six tasks, increasing the average score from 65.7% to 66.4%. The gains are most visible on Memorize and Compress, suggesting that the exact-flow state transition improves the model’s ability to maintain and update token-level memory.

- 4.6 Training Efficiency

We compare the training efficiency of DeltaNet and EFLA under matched sequence-length and batch-size configurations. As shown in Figure 4, EFLA maintains nearly identical throughput to DeltaNet across all settings. These results confirm that the exact-flow update preserves the hardware-efficient structure of the delta-rule recurrence and does not introduce meaningful training overhead.

### 5 Related Work

Kernel Linear Attention and Delta-Rule Updates. Linear-time attention methods reduce the quadratic cost of softmax attention by replacing or reformulating the softmax operation. Kernel-based methods such as Linear Transformers (Katharopoulos et al., 2020) and Performer (Choromanski et al., 2020) express causal attention through recurrently accumulated key–value statistics. From the fast-weight perspective, Schlag et al. (2021) interpret this matrix-valued state as an associative memory.

Delta-rule linear attention (Schlag et al., 2021; Yang et al., 2024b) improves this memory update by solving an online reconstruction problem. A gradient step with step size βt gives

St = (I − βtktk⊤t )St−1 + βtktvt⊤. (31) This update introduces input-dependent correction and forgetting. Later variants such as Gated DeltaNet (Yang

- et al., 2024a) and Kimi Delta Attention (KDA) (Team et al., 2025a) further strengthen Euler discretization recurrences through gating, adaptive update coefficients, and richer value mixing. In contrast, our work revisits the same delta-rule dynamics from a numerical-integration perspective: under a zero-order-hold (ZOH) formulation, the delta-rule update is an explicit Euler discretization. EFLA replaces this Euler transition with the exact ZOH flow induced by the rank-1 matrix At = ktk⊤t , while preserving the computational structure.

State Space Models and Continuous-Time Discretization. State Space Models (SSMs) provide another route to efficient long-sequence modeling by deriving discrete-time sequence operators from continuous-time dynamical

systems. The S4 family (Gu et al., 2022a,b, 2020) uses structured state matrices and discretization schemes such as the bilinear transform and ZOH. Mamba (Gu and Dao, 2024) introduces input-dependent selectivity with a hardware-efficient scan, and Mamba-2 (Dao and Gu, 2024) connects selective SSMs with gated linear attention. EMA-based models (Fu et al., 2022; Ma et al., 2022; Sun et al., 2023) can also be viewed as scalar or diagonal state-space recurrences.

EFLA is related to SSMs in that it also starts from a continuous-time ODE and derives a discrete-time update. The key difference is the transition structure: SSMs typically rely on scalar, diagonal, or specially structured matrices, whereas delta-rule linear attention induces a token-dependent rank-1 transition matrix. This rank-1 structure makes both the matrix exponential and input integral analytically tractable, yielding an exact token-wise ZOH flow that retains the algebraic form needed for efficient WY/UT-style chunkwise parallelization.

- 6 Conclusion

We presented Exact Flow Linear Attention (EFLA), an exact-flow formulation of delta-rule linear attention. By interpreting the Euler-style Delta-rule state update as an explicit Euler discretization of a zero-order-hold (ZOH) continuous-time system, we identified a principled source of approximation error in the standard delta-rule recurrence. Instead of heuristically modifying this Euler-style update, EFLA derives the closed-form flow of the underlying ZOH dynamics. By exploiting the token-wise rank-1 transition matrix At = ktk⊤t , the resulting update removes the Euler discretization error while preserving the algebraic structure, parameter count, and linear-time complexity of DeltaNet. Empirically, we showed that this exact-flow replacement leads to consistent gains across robustness tests, language modeling benchmarks, and the MAD synthetic benchmark. EFLA improves robustness under corrupted and high-energy inputs, achieves lower perplexity, and obtains stronger downstream performance than DeltaNet without introducing additional parameters. Moreover, its compatibility with WY/UT-style chunkwise parallelization preserves practical training efficiency. These results suggest that exact-flow integration provides a principled and scalable way to improve delta-rule linear attention, and may inspire future work on exact solvers for more general continuous-time attention-like architectures.

References

Christian Bischof and Charles Van Loan. The wy representation for products of householder matrices. SIAM Journal on Scientific and Statistical Computing, 8(1):s2–s13, 1987.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact

attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022. Leonhard Euler. Institutiones calculi integralis, volume 1. impensis Academiae imperialis scientiarum, 1792. Daniel Y Fu, Tri Dao, Khaled K Saab, Armin W Thomas, Atri Rudra, and Christopher Ré. Hungry hungry hippos:

Towards language modeling with state space models. arXiv preprint arXiv:2212.14052, 2022.

Google. Gemini deep research, 2025. https://gemini.google/overview/deep-research/. Accessed: 2025-12-11. Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on

language modeling, 2024. Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Re. Hippo: Recurrent memory with optimal polynomial projections, 2020. https://arxiv.org/abs/2008.07669. Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces, 2022a. https://arxiv.org/abs/2111.00396. Albert Gu, Ankit Gupta, Karan Goel, and Christopher Ré. On the parameterization and initialization of diagonal state space models, 2022b. https://arxiv.org/abs/2206.11893.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081): 633–638, 2025.

Arieh Iserles. A first course in the numerical analysis of differential equations. Number 44. Cambridge university press, 2009.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. https://arxiv.org/abs/2310.06825.

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1658–1677, 2024.

Thierry Joffrain, Tze Meng Low, Enrique S Quintana-Ortí, Robert van de Geijn, and Field G Van Zee. Accumulating

householder transformations, revisited. ACM Transactions on Mathematical Software (TOMS), 32(2):169–179, 2006. Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions. 2017. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregres-

sive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.

Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. An llm compiler for parallel function calling. In Forty-first International Conference on Machine Learning, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

Hanyu Lai, Xiao Liu, Junjie Gao, Jiale Cheng, Zehan Qi, Yifan Xu, Shuntian Yao, Dan Zhang, Jinhua Du, Zhenyu Hou, et al. A survey of post-training scaling in large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2771–2791, 2025.

Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023.

Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke

Zettlemoyer. Mega: Moving average equipped gated attention. arXiv preprint arXiv:2209.10655, 2022. Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new

dataset for open book question answering. In EMNLP, 2018.

OpenAI. Introducing deep research, 2025. https://openai.com/index/introducing-deep-research/. Accessed: 202512-11.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: Long papers), pages 1525–1534, 2016.

Michael Poli, Armin W Thomas, Eric Nguyen, Pragaash Ponnusamy, Björn Deiseroth, Kristian Kersting, Taiji Suzuki, Brian Hie, Stefano Ermon, Christopher Ré, Ce Zhang, and Stefano Massaroli. Mechanistic design and scaling of hybrid architectures, 2024. https://arxiv.org/abs/2403.17844.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR, 2021.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, 2023. https://huggingface.co/datasets/cerebras/ SlimPajama-627B.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, et al. Kimi linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692,

- 2025a.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701,

- 2025b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024a.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. Advances in neural information processing systems, 37:115491–115522, 2024b.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

## Appendix

- A Experimental Setting

We used 8 A100×80G GPUs for 340M and 1.3B language modeling experiments. The random seed it set to 42. Each model uses AdamW for optimization, with a peak learning rate of 3 × 10−4. The 340M models are trained for 8 billion tokens with a global batch size of 1M tokens, while the 1.3B models are trained for 50 billion tokens with a global batch size of 2M tokens. We use a cosine learning rate schedule, starting with a warm-up phase of 1 billion tokens for the 340M models and 2 billion tokens for the 1.3B models (corresponding to 1024 warm-up steps). Both have configurations that initial and final learning rates set at 3 × 10−5. We apply a weight decay of 0.1 and use gradient clipping at a maximum of 1.0. The head dimension is set to 128, and the kernel size for convolution layers is set at 4. To ensure numerical stability, specifically to prevent division by zero when the key norm ∥kt∥2 vanishes, we clip it with a lower bound of ϵ = 1 × 10−12. Additionally, we employ the expm1 function to compute the numerator 1 − e−β

tλt, preserving precision for small exponents.

- B Construction and Properties of rank-1 Matrices

At is a rank-1 matrix, and it satisfies: A2t = ktk⊤t ktk⊤t = kt(k⊤t kt)k⊤t = λtAt, (32)

Where λt = k⊤t kt is scalar value. Then it gives us a key property: At is a scaled projection matrix (i.e., A2t = λtAt).

- C General Solutions of Ordinary Differential Equations

We start with a first-order linear matrix ODE:

dS dt

= −AS + b, (33) Which can be rewrite as:

dS dt

+ AS = b, (34)

For this type of differential equation, the integrating factor is:

e Adt, (35) Since A is constant, the integrating factor is simply:

eAt (36)

Multiply the entire equation by eAt:

eAt

dS dt

+ AS = eAtb, (37)

Expanding the left-hand side:

dS dt

eAt

+ eAtAS = eAtb, (38)

By the product rule for matrix-vector multiplication:

d dt

(eAtS) =

d dt

eAt S + eAt

dS dt

, (39)

and since dtd eAt = AeAt, we have:

d dt

(eAtS) = (AeAt)S + eAt

dS dt

, (40)

Notice this matches exactly the left-hand side of Eq. 38 (since A and eAt commute). The equation becomes:

d dt

(eAtS) = eAtb, (41)

Integrate both sides from t (initial time) to t+βt (final time). To avoid confusion, we use τ as the integration variable:

t+βt

t+βt

d dτ

(eAτS(τ))dτ =

eAτbdτ, (42)

t

t

t+βt

eAτS(τ) t t+βt =

eAτbdτ, (43) Thus:

t

t+βt

eA(t+β

t)S(t + βt) − eAtS(t) =

eAτbdτ, (44)

t

Multiply both sides by e−A(t+β

t):

##### S(t + βt) − e−A(t+β

t)eAtS(t) = e−A(t+β

t)

t+βt

eAτbdτ, (45)

t

Simplify using exponential properties:

S(t + βt) − e−Aβ

S(t) =

t

t+βt

e−A(t+β

t−τ)bdτ, (46)

t

Since e−A(t+β

t) is constant, it can be moved inside the integral. Let s = τ − t. Then:

s = 0, τ = t s = βt, τ = t + βt

and dτ = ds, (47)

Substitute:

βt

e−A[t+β

t−(s+t)]bds =

0

βt

e−A(β

t−s)bds, (48)

0

Rename s back to τ (dummy variable) and replace constants with At,bt:

βt

e−(β

t−τ)Atbt dτ, (49)

0

Combining everything, the full solution is:

S(t + βt) = e−β

tAtS(t) +

βt

e−(β

t−τ)Atbt dτ, (50)

0

- D Exact-flow formulation of Gated DeltaNet. Gated DeltaNet updates the fast-weight state by

St = αt I − βtktk⊤t St−1 + βtktvt⊤. (51)

Expanding the transition term gives

αt I − βtktk⊤t = I − (1 − αt)I + αtβtktk⊤t . (52) Therefore, the update can be written in Euler form as

St = St−1 − (1 − αt)I + αtβtktk⊤t St−1 + βtktvt⊤. (53) Thus, it can be interpreted as an explicit Euler discretization of dS(τ) dτ

= − (1 − αt)I + αtβtktk⊤t S(τ) + βtktvt⊤. (54)

Let

δt = 1 − αt, ρt = αtβt, λt = k⊤t kt, (55) and define

At = δtI + ρtktk⊤t . (56) The ODE becomes

dS(τ) dτ

= −AtS(τ) + βtktvt⊤. (57) Under the zero-order hold assumption, the exact solution over one update interval is

St = e−A

St−1 +

t

1

e−(1−τ)A

βtktvt⊤ dτ. (58)

t

0

For the transition term, since:

At = δtI + ρtktk⊤t , (59) we have:

tktk⊤t . (60) Using the rank-one matrix exponential identity:

e−A

##### = e−δ

e−ρ

t

t

1 − e−ρ

tλt λt

tktk⊤t = I −

ktk⊤t , (61) we obtain:

e−ρ

1 − e−ρ

tλt λt

e−A

St−1 = e−δ

ktk⊤t St−1. (62)

I −

t

t

For the input term, since:

Atkt = (δt + ρtλt)kt, (63) we have:

t+ρtλt)ktvt⊤. (64) Let

e−sA

ktvt⊤ = e−s(δ

t

ηt = δt + ρtλt = (1 − αt) + αtβtλt. (65)

##### Then:

1

1

dτ ktvt⊤ (66)

βtktvt⊤ dτ = βt

e−(1−τ)η

e−(1−τ)A

t

t

0

0

1 − e−η

t

ktvt⊤. (67)

= βt

ηt

Combining the transition and input terms gives the exact-flow counterpart:

St = e−(1−α

t) I −

where λt = k⊤t kt.

1 − e−α

1 − e−[(1−α

tβtλt λt

t)+αtβtλt] (1 − αt) + αtβtλt

ktk⊤t St−1 + βt

ktvt⊤, (68)

