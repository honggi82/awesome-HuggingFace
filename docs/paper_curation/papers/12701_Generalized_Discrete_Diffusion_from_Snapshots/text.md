### Generalized Discrete Diffusion from Snapshots

# arXiv:2603.21342v1[stat.ML]22Mar2026

Oussama Zekri1 Th´eo Uscidda1 Nicolas Boull´e2 Anna Korba1

#### Abstract

We introduce Generalized Discrete Diffusion from Snapshots (GDDS), a unified framework for discrete diffusion modeling that supports arbitrary noising processes over large discrete state spaces. Our formulation encompasses all existing discrete diffusion approaches, while allowing significantly greater flexibility in the choice of corruption dynamics. The forward noising process relies on uniformization and enables fast arbitrary corruption. For the reverse process, we derive a simple evidence lower bound (ELBO) based on snapshot latents, instead of the entire noising path, that allows efficient training of standard generative modeling architectures with clear probabilistic interpretation. Our experiments on large-vocabulary discrete generation tasks suggest that the proposed framework outperforms existing discrete diffusion methods in terms of training efficiency and generation quality, and beats autoregressive models for the first time at this scale. We provide the code along with a blog post on the project page : https://oussamazekri.fr/gdds.

Zero-shotperplexity()↓

UDLM MDM AR

GDDS Gauss

200

150

100

50

0

PTB LM1B Wikitext

Figure 1. Zero-shot transfer of OWT-trained models. Zeroshot perplexity (↓) on three representative downstream validation sets from Table 3: PTB, LM1B, and Wikitext. Across this highto-low perplexity range, GDDS Gauss consistently achieves the lowest transfer perplexity, highlighting the stronger generalization capability induced by semantically structured noising processes.

#### 1. Introduction

Diffusion models (Ho et al., 2020; Song et al., 2021) recently became a core component of generative modeling and achieved remarkable success in high-dimensional tasks defined on continuous domains, such as image (Rombach et al., 2022; Saharia et al., 2022), audio (Kong et al., 2021; Liu et al., 2023), and video generation (Brooks et al., 2024; Wiedemer et al., 2025). The extension of diffusion modeling to discrete data is of great interest since many data structures (including text, graphs, and molecules) are inherently discrete. This has led to the emergence of diffusion Large Language Models (dLLMs) (Lou et al., 2024; Li et al., 2025). dLLMs offer a competitive alternative to the auto-regressive (AR) paradigm dominating language modeling (Touvron

1CREST, ENSAE, Institut Polytechnique de Paris, Paris, France 2Department of Mathematics, Imperial College London, London, SW7 2AZ, UK. Correspondence to: Oussama Zekri <oussama.zekri@ensae.fr>.

Preprint. March 24, 2026.

- et al., 2023; Team et al., 2023; Liu et al., 2024) due to their ability to generate all tokens simultaneously.

Discrete diffusion models come in several variants, mainly differing in the choice of the noising process and how denoising is performed. Masked diffusion models (MDM) (Sahoo

- et al., 2024; Shi et al., 2024; Ou et al., 2025; Nie et al., 2025) rely on a noising process where tokens are progressively replaced by a special [MASK] token. For uniform-state diffusion models (USDMs) (Austin et al., 2021; Schiff et al.,
- 2025; Sahoo et al., 2025), they are replaced with samples from the uniform distribution over the set of all possible tokens. These forward dynamics directly shape the reverse generation process: USDMs allow tokens to be updated continuously, whereas MDMs fix them once they are unmasked. The design space for discrete diffusion models remains surprisingly narrow. Most existing dLLMs pair a simplistic token-wise corruption rule (masking or uniform replacement) with the mean parametrization (Austin et al., 2021). Here, a denoiser predicts a distribution over

NOISING

TRAINING

Generalized forward noising

Predict clean tokens from noisy snapshots

Sample a noisy snapshot xt at time t.

Use (xt, t) directly as the training input.

Noisy snapshot

x0 My name is David

My word was David t

t ∼ Unif[0, 1]

xt Absorb My [MASK] [MASK] David Uniform My 12 soup David Semantic My word was David

µθ(xt, t)[x

0] ≈ q(x0 | xt, t)

µθ(xt,t)

###### name

0.73 0.18 0.09

word time

L(θ) = Et,x

CE x0,µθ(xt,t)

t

Examples of noisy snapshots induced by different Qt.

Figure 2. Overview of GDDS. A clean sequence x0 is first noised exactly by the forward CTMC at a sampled time t ∈ [0, 1], yielding a snapshot sequence (xt, t). The mean parametrization is then used as a denoiser: given the snapshot, the model predicts the clean-token posterior directly from (xt, t), so training is performed on snapshots rather than through a full path-wise objective.

the clean token and reverse transition probabilities are derived from this prediction through an ELBO objective. This leads to two bottlenecks: (i) the forward process is blind to any notion of neighborhood in discrete spaces (e.g., semantic proximity in language), and (ii) mean parametrization tightly constrains how denoising uncertainty can be translated into reverse dynamics, becoming increasingly restrictive beyond uniform/masked noise and at LLM scale. Advancing dLLMs calls for structure-aware noising and more flexible parametrizations that remain computationally scalable for large vocabularies and long contexts.

In this work, we generalize discrete diffusion methods by considering arbitrary noising processes and propose a tractable associated training method. We introduce the Generalized Discrete Diffusion from Snapshots (GDDS) framework, which builds upon the most general formulation of interpolating discrete diffusion and extends it far beyond the restricted subclasses explored in prior work (Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025; von R¨utte et al., 2025a; Zhou et al., 2025; Amin et al., 2025). GDDS introduces three key advances for discrete diffusion:

- 1) Generalized interpolating discrete diffusion: a mathematical framework covering arbitrary Markovian noising processes, encompassing all existing approaches.
- 2) Efficient noising process: a fast forward arbitrary corruption method for large vocabularies, requiring only column access to the rate matrix characterizing the noising process.
- 3) Parametrization and ELBO: a principled parametrization for reverse transition probabilities, yielding a simple ELBO training objective based on snapshot samples.

These components form the first discrete diffusion framework that is fully general and computationally efficient. Our experiments on large-scale language modeling tasks demon-

strate state-of-the-art modeling and generation quality. Figure 2 summarizes the two ingredients behind GDDS: exact forward noising to a snapshot and snapshot-level denoising.

#### 2. Background and Preliminaries

This section provides background material on discrete diffusion, including the definition of rate matrices that characterize the evolution of continuous-time Markov chains, as well as common choices used in the literature.

###### 2.1. Discrete Diffusion

In discrete diffusion, the dynamics of a single token are described by a continuous-time Markov chain (CTMC) (Campbell et al., 2022; Lou et al., 2024), which is a stochastic process (xt)t∈[0,T] operating on a finite vocabulary V = {v1,...,vm}. We denote by δ[x] the one-hot encoding of x ∈ V. For a column-stochastic matrix1 F ∈ Rm×m, we use the shorthand F(·,x) ≡ Fδ[x] to denote the probability vector corresponding to the column indexed by x.

Forward and reverse evolution. Let T > 0 be a fixed time horizon. At any time t ∈ [0,T], the distribution of xt ∈ V is denoted by qt ∈ ∆m, where ∆m is the probability simplex over V. We set q0 := qdata and qT := qref a simple reference distribution. We represent the forward noising process through a family of Markov transition matrices (Kt)t∈[0,T] acting on marginals as qt = Ktq0, with xt ∼ qt. Here, Kt ∈ Rm×m describes how probability mass flows across tokens as corruption increases. Existing discrete diffusion schemes, such as uniform or masking corruption,

1A column stochastic matrix is a matrix whose columns are

probability distributions, hence mi=1 F(i, j) = 1 with F(i, j) ∈ [0, 1] for any column 1 ≤ j ≤ m.

correspond to particular choices of Kt. Given a clean token x0 ∼ qdata, the noised token xt ∈ V at time t is drawn from the categorical distribution:

qt(xt | x0) = Cat(xt;Kt(·,x0)) = Kt(xt,x0). (1)

While Kt can be specified directly, we focus on the principled setting where these noising operators are induced by a continuous-time Markov process with (possibly time-inhomogeneous) rate matrix (also called infinitesimal generator) Qt. This matrix is defined as Qt(i,j) = limh↓0 P(xt+h=i|hxt=j)−δij for i ̸= j, and diagonal entries enforcing conservation of mass Qt(j,j) = − i̸=j Qt(i,j). In this case, Kt is defined as the solution to the Kolmogorov forward equation:

dKt dt

= QtKt, K0 = Im, for t ∈ [0,T]. (2)

Solving Eq. (2) yields Kt = T exp( 0 t Qs ds), where T is the time-ordering operator (see Section A). Since each column Kt(·,x0) for x0 ∈ V lies in the probability simplex ∆m and corresponds to the forward marginal qt(· | x0) as given in Eq. (1), a noisy token xt can be obtained by directly sampling from this categorical distribution, without simulating the underlying continuous-time trajectory. Moreover, since qt = Ktq0, token distributions evolve according to the Kolmogorov forward equation for marginals:

dqt dt

= Qtqt, xt ∼ qt, for t ∈ [0,T]. (3)

The time reversal of this equation (Kelly, 2011) defined through pt := qT−t is

dpt dt

= Qtpt, xT−t ∼ pt, for t ∈ [0,T], (4)

where Qt(i,j) = (qt(i)/qt(j))Qt(j,i) if i ̸= j and Qt(i,i) = − j̸=i Qt(j,i). Without loss of generality, we select T = 1 as any bounded interval [0,T] can be rescaled to [0,1] through the change of variable t  → t/T. In discrete diffusion, a neural network learns to simulate the reverse dynamics given by Eq. (4) to reconstruct a clean data x0 from a fully noised quantity x1.

Rate matrices. At any time t ∈ [0,1], the forward and reverse CTMCs evolve according to rate matrices Qt and Qt with non-negative off-diagonal entries Qt(i,j) ≥ 0 for i ̸= j, and diagonal entries verifying Qt(j,j) = − i̸=j Qt(i,j), and analogously for Qt. We factorize

- them into exit rates and jump kernels as

Qt = (Ft − Im)diag(f1(t),...,fm(t)), Qt = (Rt − Im)diag(r1(t),...,rm(t)),

where fj(t) = i̸=j Qt(i,j) and rj(t) = i̸=j Qt(i,j) are the (forward and reverse) exit rates, controlling how

often the chain leaves state j. The matrices Ft and Rt specify where the chain jumps when it leaves a state and are defined as column-stochastic matrices:

 

Qt(i,j) fj(t)

, fj(t) > 0, 0, fj(t) = 0,

for i ̸= j,

Ft(i,j) =



and Ft(j,j) = 1− i̸=j Ft(i,j), analogously for Rt. Note that this factorization is not exploited in the literature.

###### 2.2. Designs of the rate matrix

A common choice to simplify the forward noising process is to select equal forward exit rates: f1(t) = ... = fm(t) = f(t), and a time-independent forward jump kernel Ft = F. In this case, Qt = f(t)(F − Im), and a single matrix F ∈ Rm×m must be stored. In this model, the time-ordered exponential T simplifies to a standard matrix exponential, and ensures that Kt admits the following closed form:

Kt = exp f ¯(t)(F − Im) , where f¯(t) =

t

f(s)ds.

0

Being able to sample from columns of the matrix exponential Kt is crucial to design a scalable noising process.

Usual forms of F. For typical vocabulary sizes used in language models (m = 50,257 for GPT-2; Radford et al. 2019), storing a dense F ∈ Rm×m requires more than

- 2.5 × 109 parameters (≈ 20GB in double precision) and each matrix-vector products involving F costs O(m2) time complexity, making it computationally impractical. Hence, forward kernels are usually highly structured (Austin et al., 2021; Campbell et al., 2022; Lou et al., 2024), such as the ones related to the uniform and mask noising processes:

Funiform :=

1 m

11⊤, Fabsorb := δ[MASK]1⊤, (5)

for which Kt admits closed-form expressions (Austin et al., 2021; Lou et al., 2024), enabling efficient noising. Since these F matrices are idempotent, the exponential matrix writes Kt = exp(−f(t))Im + (1 − exp(−f(t)))F. However, these restrictive structures impose rigid corruption patterns on the tokens, motivating our flexible approach.

- 3. Forward noising with diffusion

###### 3.1. Generalized interpolating discrete diffusion

We consider a time-differentiable, decreasing, mixing rate t  → αt : [0,1] → [0,1] such that α0 = 1, α1 = 0, and αt < 1 for t > 0. We introduce a time-differentiable

column-stochastic mixing matrix Πt ∈ Rm×m, which specifies how probability mass is redistributed across tokens as noise increases, along with its interpolating matrix as2

Kt := αtIm + (1 − αt)Πt, t ∈ [0,1]. (6)

Here, Πt encodes the structure of the noising mechanism and αt its intensity. This formulation recovers common discrete diffusion schemes as special cases, such as masked or uniform. Yet, more general choices of Πt allow for structured and token-dependent corruption mechanisms. The rate matrix Qt associated with Kt is given in Proposition 3.1.

Proposition 3.1. Let t ≥ 0 and denote by K˙ t the time derivative of Kt. The rate matrix induced by Eq. (6) is Qt = K˙ tKt−1.

Choosing a column-constant mixing matrix Πt = πt1⊤ (a rank-one form with πt ∈ ∆m) in Eq. (6) yields the GIDD formulation of (von R¨utte et al., 2025a, Lem. 3.6), namely Qt = α˙

αt Im +(1−αt)˙πt1⊤ − α˙

αtπt1⊤ (see Corollary B.1). However, this formulation encompasses all existing frameworks, including (Zhou et al., 2025) and GenMD4 (Shi et al., 2024), unlike GIDD (von R¨utte et al., 2025a).

t

t

Expressiveness. Reversely, given any rate matrix Qt, we aim to find a mixing matrix Πt such that the interpolating matrix Kt defined by Eq. (6) coincides with a solution to Eq. (2); which induces the marginal (qt)t≥0 as in Eq. (3).

Proposition 3.2. Let αt a mixing rate such that α˙0 < 0 and Qt a rate matrix. There exists a unique mixing matrix Πt ∈ Rm×m such that for all t ∈ [0,1],

t

Kt = αtIm + (1 − αt)Πt = T exp

Qs ds .

0

Following Proposition 3.2, if Πt is known in closed-form,

- then simulating the noising process becomes possible. Indeed, qt(· | x0) = Cat(·;Kt(·,x0)) requires only the evaluation of the column Πt(·,x0) instead of costly matrix exponentiations. While columns are known in closed form for uniform or masked schemes, this is generally not the case for an arbitrary Πt.

###### 3.2. Efficient forward noising through uniformization

Since computing the marginals qt(· | x0) exactly is generally intractable, we employ an exact noising procedure based on uniformization. Classical uniformization provides an exact Poisson-based representation of the matrix

2We assume that, for every t ∈ (0, 1), Kt is invertible and the solution v(t,x) to the linear system Kt⊤v(t,x) = K˙ t⊤δx satisfies v[(yt,x] ) ≥ 0 for y ̸= x to guarantee that Kt induces a valid CTMC.

exponential Kt (Jensen, 1953; Stewart, 2009). Here, we use the same procedure to generate exact forward samples xt ∼ qt(· | x0) without requiring exact knowledge of these marginals; hence avoiding computing the exponential. Following Proposition 3.2, the interpolating matrix in Eq. (6) is expressive enough to represent any rate matrix Qt. Recall that any such Qt can be written in factored form as Qt = (Ft − Im)diag(f1(t),...,fm(t)). To simplify the exposition, we focus on the shared exit rates case:

Qt = f(t)(Ft − Im), (7)

which preserves the transition structure encoded in Ft, while making the uniformization-based noising process significantly easier to implement. This result can be extended to general non-shared exit rates through Poisson thinning. We

denote by f¯(t) = 0 t f(s)ds the integrated exit rate and set the mixing rate to αt = exp(−f¯(t)) for t ∈ [0,1].

Proposition 3.3 (Uniformization). Consider a rate matrix Qt of the form (7) and the mixing rate αt = exp(−f¯(t)), where f¯(t) = 0 t f(s)ds. Let (Nt)t∈[0,1] be a non-homogeneous Poisson process with intensity f¯(t), and denote by 0 < T1 < ... < TN

t ≤ t its jump times on [0,t]. The unique matrix Πt provided by Proposition 3.2 is Πt = E[FT

1 | Nt ≥ 1].

...FT

Nt

Therefore, computing qt(· | x0) at any t ∈ [0,1] amounts to computing a column of E[FT

1 | Nt ≥ 1], which can be done approximately even when the vocabulary size m is large (Dingle et al., 2004). If we only need to draw samples xt ∼ qt(· | x0) rather than evaluate the full distribution, we can instead sample exactly by performing Nt transitions with the matrix Ft, using only Poisson sampling and column access to Qt (see Section B.2). This procedures implicitly builds a discrete-time Markov chain zk = xT

...FT

Nt

for all 1 ≤ k ≤ Nt initialized at z0 = x0. Algorithm 1 details the resulting token-level noising procedure and returns the noised token zN

k

, which coincides exactly with xt ∼ qt(· | x0) following Proposition 3.3. Algorithm 1 Exact general noising, token level

= xT

t

Nt

- 1: Input: clean token z0 = x0, time t ∈ [0,1], intensity f¯(t), rate matrix Qt as in Eq. (7)
- 2: Sample number of jumps Nt ∼ Poisson(f¯(t))
- 3: Sample and sort the jump times as T1 < ... < TN

t

- 4: for k = 1 to Nt do
- 5: Sample jump zk ∼ FT

k

(·,zk−1)

- 6: end for
- 7: return noised token xt = zN

and jumps (zk,Tk)N

t

k=1

t

Algorithm 1 enables efficient noising for any continuoustime noising process (beyond masked and uniform), requir-

ing only column access to the rate matrix Qt (instead of its generally intractable matrix exponential). This procedure generalizes easily to a parallel sequence-level algorithm for a sequence x0 = x10 ...xn0 of length n ≥ 1 (see Algorithm 3 in Section B.2). While the time input can be any value t ∈ [0,1], selecting t = 1 yields a full forward noising path of the form ω = {N1,(zk,Tk)N

k=1}.

1

#### 4. Reverse learning: aligning the generative model, and the objective

Readers mostly interested in the implementation and the loss function may refer to Section 4.3 and Algorithm 2.

###### 4.1. The core mismatch in reverse parametrization

A common choice in the discrete diffusion litterature to simulate the reverse dynamics is to use the mean parametrization (also known as x0-parametrization). Concretely, µθ : V × [0,1] → ∆m is a neural network outputting a probability vector on the token space V, which aims to approximate the posterior of the clean token from snapshots latents

- s = (xt,t) generated by the forward noising process, i.e.,

µθ(xt,t)[x

0] ≈ q(x0 | xt,t). It is often plugged into the reverse-time model via Bayes’ rule as

pθ,u|patht (xu | xt) =

x0∈V

q(xu | xt,x0)µθ(x0 | xt,t), (8)

where q(xu | xt,x0) = qt|u(xqt|xu)qu(xu|x0)

t(xt|x0) and qt|u(xt | xu) denotes the forward conditional from time u ∈ [0,1] to

- t ∈ [0,1] with u ≤ t. However, plugging µθ into the reversetime model through Eq. (8) does not generally enforce

0] ≈ q(x0 | xt,t). This construction glues the mean denoiser to the entire reverse CTMC: the same µθ controls when the chain jumps (reverse intensities) and where it jumps (reverse destinations), creating a training burden mismatch. Our insight is that the mean network naturally parametrizes a snapshot generative model that should be trained to actually achieve µθ(xt,t)[x

µθ(xt,t)[x

0] ≈ q(x0 | xt,t), whereas modeling the reverse CTMC calls for a jump network jθ designed directly for the path-wise generative model with path-wise latents ω. To align the objective with the generative object, we first parametrize the reverse CTMC directly by disentangling jump times and jump destinations. Then, we focus on how one should design a snapshot generative model from the mean parametrization.

###### 4.2. Path-wise model and loss function

Jump-states parametrization. Inspired by the factorization Qt = (Rt − Im)diag(r1(t),...,rm(t)) of the true reverse generator, we directly learn Rt while keeping the exitrate schedule {ri(t)}mi=1 fixed to the true reverse rates. Note that even when the forward process uses shared exit rates

as in Eq. (7), the reverse exit rates are not shared in general (see e.g. the masked diffusion example in Section B.3.4). We consider a neural network jθ : V × [0,1] → ∆m that yields the following jump-states parametrization:

Qθt = (Rtθ − Im)diag(r1(t),...,rm(t)), (9)

where Rtθ ∈ Rm×m is the column-stochastic infinitesimal reverse jump kernel (where the chain jumps) defined by

t] := jθ(xt,t), and the exit rates ri(t) (when the chain jumps) remain fixed. This parametrization of Eq. (4) is fundamentally different from the score parametrization of (Lou et al., 2024), the schedule-conditioned parametrization of (Amin et al., 2025) and the parametrization in Eq. (8) (cf. Section B.3.1).

Rtθδ[x

Path-wise ELBO. We derive the Evidence Lower Bound (ELBO) associated with our jump-states parametrization given by Eq. (9) in Proposition 4.1. This parametrization is key to obtain a simple, CTMC-aligned, ELBO with a clean learning objective: a weighted cross-entropy that matches the model reverse jump kernel to the ideal reverse jump kernel, with θ-independent weights given by the reverse exit rates r[x

t](t). The result holds for any forward rate matrix Qt, as the interpolating family Kt = αtIm + (1 − αt)Πt

can represent an arbitrary rate matrix (Eq. (6) and Proposition 3.2). Here, pθ,t path is induced by Eq. (4) where we replace Qt by Qθt.

Proposition 4.1 (Path-wise ELBO). Let x0 ∈ V, the ELBO is log pθ,0 path(x0) ≥ −Lpathx0 (θ)+Cxpath

, where

0

1

t∼qt(·|x0) rx

[xt](t)CE(Rx

Lpathx0 (θ) =

t ,Rtθ) x

Ex

dt,

0

0

t

0

is independent of θ. Here, CE(Rx

and Cxpath

t ,Rtθ) x

0

0

t

denotes the cross-entropy between the vectors Rx

t (·,xt) and Rtθ(·,xt) = jθ(xt,t). Rx

t and rx

[xt](t) denote respectively the true conditional reverse jump kernel and its associated exit rate (see Definition B.4).

0

0

0

In the masked diffusion case, where Πt = δ[MASK]1⊤, our jump parametrization and ELBO coincide with the parametrization (8) and ELBO used in prior work (Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025); see Section B.3.4. Beyond this setting, other objectives such as (von R¨utte et al., 2025a; Zhou et al., 2025; Lou et al., 2024; Amin et al., 2025) apply to broad classes of noising processes but do not isolate such a clean weighted cross-entropy signal and involve additional terms. Indeed, we show in Section B.3.2 that Lpathx0 (θ) can be written as

 

 dt.

1

qt(y | x0) qt(xt | x0)

Lpathx0 (θ)=

Ex

Qt(xt,i)(−log jθ(xt,t)y)

t∼qt(·|x0)

0

y̸=xt

This loss remains useful beyond masking: when the forward marginal is tractable (typically, when Πt is known so that qt(· | x0) can be evaluated; e.g., in the masked or uniform case), Lpathx0 (θ) is fully computable and directly trains the path-wise reverse CTMC. However, we seek to avoid knowledge of qt(· | x0) for a general CTMC Qt, for which the associated Πt is unknown.

Campbell estimator. To mitigate this issue, we introduce a path-wise Campbell estimator that is a clean rewriting of the path-wise loss. It consists of applying Campbell’s formula (Campbell, 1909; Last & Penrose, 2018) to Lpathx0 (θ), which transforms the integral into a sum over the whole noising path in [0,1] given by the Poisson process of Algorithm 1. Importantly, this expression does not involve any other quantity than the network output and the uniformization path produced by Algorithm 1, making it computable even when qt(· | x0) (i.e., Πt) is unknown.

Proposition 4.2 (Campbell estimator). Let x0 ∈ V

and ω ∼ q[0,1](· | x0) denote the full forward noising path produced by Algorithm 1. Writing ω =

{N1,(zk,Tk)N

k=0} for its jump counts and marks, we have

1

N1

Lpathx0 (θ) = Eω∼q

−log jθ(zk,Tk)[z

k−1] .

[0,1](·|x0)

k=1

This loss is reminiscent of any-order AR objectives (e.g., XLNet; Yang et al. 2019), except that each term predicts the pre-jump token zk−1 from the post-jump noised context (zk,Tk), and the factorization order is induced by Poisson jump times (closer in spirit to denoising permutation objectives such as MPNet; Song et al. 2020). As a result, training requires evaluating many snapshot-wise conditionals along a single path and, in practice, calls for a two-stream mechanism (separating a content stream encoding the clean tokens from a query stream used to predict the target token [zk−1],

- as in XLNet), which is not naturally aligned with standard transformer architectures and empirically underperforms them (see Section E).

###### 4.3. Snapshot model and loss function

This observation motivates our approach: if powerful models mostly train on snapshot noised contexts, why should the variational latent variable be the entire path ω? Therefore, we consider a snapshot-latent variational formulation, where s = (xt,t) replaces ω as the latent variable. Crucially, this choice also aligns with the perspective of Li & He (2025) that denoising models should predict the clean quantity through the mean parametrization, rather than a noised quantity as the jump states parameterizations do. This aligns the mean parametrization µθ to its coherent

generative model, and yields an objective that is directly compatible with standard architectures for any general noising process.

Snapshot ELBO. Consider the snapshot latent s = (xt,t) and the variational distribution qsnap(s | x0) := qt(xt | x0). We define the snapshot predictor pθ,0 snap(x0 | s) := µθ(xt,t)x

from the output of the mean network, and derive the associated snapshot ELBO in Proposition 4.3.

0

Proposition 4.3 (Snapshot ELBO). Let x0 ∈ V, the ELBO is log pθ,0 snap(x0) ≥ −Lsnapx0 (θ)+Cxsnap

, where

0

1

Lsnapx0 (θ) =

Ex

t∼qt(·|x0) −log µθ(xt,t)[x

0] dt,

0

is independent of θ.

and Cxsnap

0

This computable snapshot ELBO boils down to denoising training on (x0,xt,t), without requiring the explicit knowledge of qt(· | x0).It is also well-suited to use with standard time-conditioned bidirectional transformer architectures (e.g., DDiT, Peebles & Xie 2023). Note that (Shi & Titsias, 2025) derived the same ELBO expression as a reweighted form of the path-wise ELBO (Proposition 4.1), but this equivalence holds only in the masked diffusion setting and for a specific “simple” weight.

Algorithm 2 GDDS training algorithm

- 1: Input: Training dataset x(1)0 ,...,x0(Ndata) ∼ qdata
- 2: for k = 1 to Ndata (potentially several epochs) do
- 3: Sample t ∼ Unif[0,1]
- 4: Sample xt ∼ qt(· | x(0k)) with Algorithm 3
- 5: Minimize −log(µθ(xt,t)[x(k)

0 ])

- 6: end for
- 7: return µθ

Algorithm 2 is reminiscent of the general procedure in (Bengio et al., 2013), but applied to the corruption process induced by the forward noising diffusion on discrete spaces.

Information-calibration decomposition. To make precise the trade-off between using less information (a snapshot) and optimizing better (lower miscalibration), we compare the expected negative log-likelihood (NLL) of predicting a clean token x0 ∼ qdata either from the full forward path ω ∼ q[0,1](· | x0) or from a randomly sampled snapshot s = (xt,t) with t ∼ Unif[0,1] independently sampled, through the quantity ∆NLLθ := E[−log pθ,0 snap(x0 | s)] − E[−log pθ,0 path(x0 | ω)]. The resulting NLL gap admits a clean decomposition into an intrinsic information path gap (IPG) and a calibration gap (CG), where the calibration is Calsθ := E[KL(q(· | s)∥pθ(· | s))].

used OpenWebText (OWT) dataset (Gokaslan & Cohen, 2019) for 500k steps. We compare small-model families under matched compute. All retrained models share the same Transformer backbone. Diffusion models use a DDiT backbone with bidirectional attention and time conditioning (Peebles & Xie, 2023) (≈ 96M non-embedding parameters); the autoregressive (AR) baseline uses the same backbone with causal self-attention and no time conditioning (≈ 89M non-embedding parameters). We retrain AR, a decoder-only Transformer trained with next-token cross-entropy, prior discrete diffusion baselines (UDLM & MDM) using their respective objectives (Schiff et al., 2025; Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025), and our GDDS snapshot framework with different forward noising processes: GDDS Absorb (masked), GDDS Uniform (uniform), and GDDS Gauss (semantic-informed; Section C.2). All reported GDDS Gauss results use the KNN implementation with k = 64 neighbors per token.

X4

 log j✓(x(4)t

Lpath(✓) =

###### ,tk)x(4)

k

x(4)t

###### tk 1

k=1

| | | | | | |
|---|---|---|---|---|---|
|! = {(| | | | | |
| |x(4)t<br><br>k<br><br>,tk)}k| 1| | | |
| | | | | | |

David

Leo

Emma

t

0 t1 t? t2 t3 t4

##### Lsnap(✓) =  log µ✓(x(4)t? ,t?)x(4)

0

Figure 3. Snapshot vs. path-wise training. The forward process corrupts the clean sequence “My name is David”. The blue path shows the beginning of the noising trajectory ω = {(x(4)t

, tk)}k≥1 of one tracked position (ℓ = 4). Path-wise objectives condition on the entire trajectory ω, whereas our GDDS snapshot objective uses only one random-time observation s = (xt⋆, t⋆).

k

Table 1. Bits Per Character (↓) on Text8. Baseline results reported from (Shi et al., 2024). All models are trained for 1M steps. Best results per model family are in bold, while best results among retrained models are underlined.

Proposition 4.4 (Snapshot vs. path-wise NLL gap). For any conditional predictors pθ,0 snap(· | s) and pθ,0 path(· | ω),

Method BPC (↓) Continuous Diffusion

Plaid (Gulrajani & Hashimoto, 2023) ≤ 1.48 BFN (Graves et al., 2023) ≤ 1.41

+ Calsθ − Calωθ

∆NLLθ = H(x0 | s) − H(x0 | ω)

.

Any-order Autoregressive

ARDM (Hoogeboom et al., 2022) ≤ 1.43 MAC (Shih et al., 2022) ≤ 1.40

IPG≥0

CG

Moreover, arg minθ E[Lsnapx0 (θ)]=arg minθCalsθ, but arg minθ E[Lpathx0 (θ)] ̸= arg minθCalωθ in general.

Autoregressive

IAF/SCF (Ziegler & Rush, 2019) 1.88 AR Argmax Flow (Hoogeboom et al., 2021) 1.39 Discrete Flow (Tran et al., 2019) 1.23 AR (Austin et al., 2021) 1.23 AR (retrain) 1.35

This decomposition exposes the core trade-off in replacing path-wise latents ω by snapshots s = (xt,t). Discarding the full path induces an intrinsic information loss as IPG ≥ 0, which can be compensated by the additional information in ω when CG ≤ 0. Here, snapshot-latent ELBOs offer a principled trade: they sacrifice some information for an objective that is better aligned with the architecture and easier to optimize, often yielding stronger generative models. In particular, minimizing the snapshot objective enforces µθ(xt,t)[x

Uniform Discrete Diffusion

Mult. Diffusion (Hoogeboom et al., 2021) ≤ 1.72 D3PM Uniform (Austin et al., 2021) ≤ 1.61 SEDD Uniform (Lou et al., 2024) ≤ 1.47 UDLM (Schiff et al., 2025) ≤ 1.44 UDLM (retrain) ≤ 1.67 GDDS Uniform (Ours) ≤ 1.50

Masked Discrete Diffusion

0] ≈ q(x0 | xt,t), which empirically produces better samples as it approximates the correct quantity in Eq. (8).This distinction between path-wise and snapshotlatent training is illustrated in Fig. 3.

D3PM Absorb (Austin et al., 2021) ≤ 1.45 SEDD Absorb (Lou et al., 2024) ≤ 1.39 GenMD4 (Shi et al., 2024) ≤ 1.34 MD4 (Shi et al., 2024) ≤ 1.37 MDM (retrain) ≤ 1.58 GDDS Absorb (Ours) ≤ 1.16

#### 5. Experiments

We evaluate GDDS on two complementary settings: language modeling and language generation. Additional results and details are deferred to Section C.

We train character-level models on Text8 (Mahoney, 2024) for 1M steps, and BPE-tokenized models on the widely

Semantic-Informed Kernel (SIK). Following Section 3, one can implement a variety of semantic-informed kernels that depend on the distance between tokens in the embedding space to noise according to words semantic similarities. Let e[x] denote the embedding vector associated with token

x. Here, we use a Gaussian SIK and, for x ̸= y, set

exp −∥e[x] − e[y]∥22/τ(t) z̸=y exp −∥e[z] − e[y]∥22/τ(t)

FtGauss(x,y) :=

,

and FtGauss(y,y) = 0, where τ(t) is chosen to increase with t so that the kernel progressively flattens and the forward process approaches the uniform distribution as its limiting distribution. The associated noising algorithm can then be implemented efficiently through Algorithm 3 together with either a KNN or a KEOPS implementation. In our experiments, GDDS Gauss is trained with the KNN instantiation using k = 64 neighbors per token; further implementation details and benchmarks are given in Section C.2. More general noising processes can be defined analogously and implemented efficiently thanks to the GDDS framework.

###### 5.1. Language modeling

We compute three metrics: Text8 BPC (bits per character) for models trained on Text8 in Table 1, OWT in-domain validation perplexity on the OWT validation split for OWTtrained models in Table 2, and OWT-trained zero-shot perplexity on validation sets of downstream tasks for OWTtrained models in Table 3. For autoregressive models, these metrics are computed from exact likelihood evaluation. For diffusion models, exact likelihoods are generally not available in closed form; whenever an ELBO is available, we report the corresponding variational upper bound on BPC/perplexity (denoted by ≤), using the ELBO associated with the objective the model was trained with.

Table 2. OWT validation perplexity. Validation perplexity (↓) on OWT. Best results are in bold. ⋆Trained on the WebText dataset. †Result taken from (Zhou et al., 2025). ‡Result taken from (Sahoo et al., 2025).

Method Training token PPL (↓) Autoregressive

GPT-2†⋆ (Radford et al., 2019) unknown 23.40 AR† 262B 16.11 AR (retrain) 262B 20.49

Uniform Discrete Diffusion

SEDD Uniform‡ (Lou et al., 2024) 524B ≤ 29.70 Duo (Sahoo et al., 2025) 524B ≤ 25.20 UDLM (retrain) 262B ≤ 36.82 GDDS Uniform (Ours) 262B ≤ 10.97

Masked Discrete Diffusion

SEDD Absorb‡ (Lou et al., 2024) 524B ≤ 24.10 GenMD4 (Shi et al., 2024) 524B ≤ 21.80 MDLM (Sahoo et al., 2024) 327B ≤ 23.00 MDM (retrain) 262B ≤ 31.03 GDDS Absorb (Ours) 262B ≤ 8.98

General Discrete Diffusion

HDLM (Zhou et al., 2025) 131B ≤ 23.25 GDDS Gauss (Ours) 262B ≤ 7.65

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | |AR|(r|etrain)|GDDS|Unifo|rm|
| | | | | | | | | | | | |
| | | | | | |UD|LM|(retrain)<br><br>|GDDS|Absor|b|
| | | | | | |MD|M|(retrain)<br><br>|GDDS|Gaus|s|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

- 101
- 102
- 103

ValidationPerplexity

36.82 31.03

20.49

10.97

8.98 7.65

104 2 × 104 5 × 104 105 2 × 105 5 × 105

Training Steps

Figure 4. OWT training curves. Evolution of OWT validation perplexity during training for the retrained models reported in

- Table 2. This complements the final numbers in Table 2 by showing the full optimization trajectory; both axes are shown on logarithmic scales.

Across both Text8 and OWT, GDDS substantially improves over prior discrete diffusion baselines under matched compute. On Text8, GDDS Absorb outperforms the AR baseline for the first time. On OWT, GDDS yields dramatically tighter variational bounds than retrained UDLM/MDM, and can outperform the matched AR baseline (see Table 2 and Fig. 4). Empirically, we often observe a negative expected NLL gap ∆NLLθ < 0, meaning that snapshot objectives improve calibration enough (i.e., CG < −IPG in Proposition 4.4). Moreover, because PPL = exp(NLL) (see Section C.3), this reduction in NLL can induce a much more pronounced multiplicative drop in perplexity. We find that the choice of forward process matters: on OWT, semantic-informed noise (e.g., GDDS Gauss) tends to outperform Uniform/Mask by proposing semantically proximal corruptions that are easier to denoise and better aligned with language structure, consistent with prior evidence that semantic similarity improves discrete diffusion modeling (Zhou et al., 2025).

- Table 3. Zero-shot transfer perplexity. Zero-shot perplexity (↓) of OWT-trained models. Models are evaluated on validation splits of 7 downstream datasets without additional fine-tuning. Best result per dataset is in bold; second-best is underlined.

PTB Wikitext103 LM1B Lambada AG News Pubmed Arxiv

Autoregressive AR (retrain) 147.90 42.91 81.29 75.93 105.34 79.93 76.29

Masked Discrete Diffusion (upper bounds ≤)

MDM (retrain) 181.36 45.42 92.58 58.89 123.51 66.46 56.82 GDDS Absorb 103.04 46.49 92.51 60.78 101.41 62.62 53.27

Uniform Discrete Diffusion (upper bounds ≤)

UDLM (retrain) 177.26 64.65 112.49 70.38 153.89 70.78 60.35 GDDS Uniform 115.12 42.63 108.83 68.92 136.24 71.69 58.83

General Discrete Diffusion (upper bounds ≤) GDDS Gauss 53.65 34.56 46.06 38.74 45.94 31.78 28.49

Overall, GDDS improves transfer compared to prior diffu-

sion baselines under matched compute, as shown in Fig. 1 and Table 3. In the masked setting, GDDS Absorb lowers zero-shot perplexity relative to the retrained MDM on most datasets, indicating better out-of-distribution generalization of the learned denoiser. In the uniform setting, GDDS Uniform yields large gains over the retrained UDLM across all datasets but one. Most notably, GDDS Gauss consistently outperforms all baselines by an important margin across every OOD dataset, suggesting that diffusion processes built from semantically structured corruptions may provide a clear generalization advantage.

Table 4. Lexical diversity. Distinct-1/2/3 (↑) computed on Ngen = 256 unconditional samples from OWT-trained models, measuring the fraction of unique n-grams among generated texts.

Model Dist-1 (↑) Dist-2 (↑) Dist-3 (↑)

AR (retrain) 0.10 0.57 0.88 MDM (retrain) 0.10 0.61 0.90 GDDS Absorb 0.10 0.60 0.89 UDLM (retrain) 0.08 0.53 0.85 GDDS Uniform 0.10 0.58 0.88

###### 5.2. Language generation

We now turn to evaluation generative performance of our models. For numerical stability, we follow (Zheng et al., 2025) and cast logits to float64 during sampling. We evaluate Ngen = 256 unconditional samples from OWTtrained models. In Fig. 5, we consider the Gen-PPL/entropy tradeoff, and report the generative perplexity of unconditional samples under a fixed evaluator (GPT2-large; Radford et al. 2019) against their sequence entropy, for multiple decoding budgets K. As a reference point for this entropy scale, Zheng et al. (2025) report that natural OWT text typically falls in the range 5.60–5.70.

300

UDLM

GDDS Uniform

MDM

GDDS Absorb

275

250

225

Gen-PPL()

200

175

150

Better (Diversity + low Gen-PPL)

125

100

5.55 5.60 5.65 5.70 5.75

Entropy

Figure 5. Generation quality-diversity tradeoff. Gen-PPL (↓) vs Entropy tradeoff. For K ∈ {32, 64, 128, 256, 512, 1024} decoding steps, we plot the generative perplexity of Ngen = 256 unconditional samples under a fixed evaluator (GPT2-large) against their sequence entropy (higher is better). Bubble radius increases with K. For reference, the AR baseline achieves Gen-PPL 56.82

- at entropy 5.60.

Two consistent patterns emerge. First, UDLM attains low Gen-PPL but remains stuck at noticeably lower entropy, suggesting conservative generations with limited diversity. Second, MDM increases entropy as K grows, but this comes with a steep degradation in Gen-PPL at larger budgets, in-

dicating that pushing diversity by running more steps can quickly harm sample quality. In contrast, GDDS improves the Pareto tradeoff under matched compute. GDDS Uniform shifts the uniform-diffusion frontier toward higher entropy while keeping Gen-PPL competitive, mitigating the low-diversity behavior of UDLM. More strikingly, GDDS Absorb yields a strictly better quality/diversity compromise than MDM in the highlighted regime: for comparable entropy, it achieves lower Gen-PPL, and reaches favorable points with substantially fewer decoding steps (smaller bubbles). In particular, at K = 64 decoding steps, GDDS Absorb attains a lower Gen-PPL at essentially the same entropy as MDM even when the latter is run with up to K = 1024 decoding steps, highlighting a large gain in sampling efficiency.

Next in Table 4, we consider judge-free quality metrics that do not rely on GPT2-large scoring, and report a diversity statistics (Distinct-n). This metric measures the fraction of unique n-grams in generated samples (higher is more diverse). Consistent with the low-entropy cluster in Fig. 5, UDLM exhibits the lowest lexical diversity (Distinct-1/2/3). GDDS Uniform increasing Distinct-1/2/3 over UDLM and matching the strong diversity of masked methods. Overall, Distinct-n corroborates the Pareto analysis: GDDS increases diversity without incurring the large Gen-PPL penalties observed for MDM at high decoding budgets.

#### 6. Conclusion

We introduced Generalized Discrete Diffusion from Snapshots (GDDS), a framework for discrete diffusion models that enables efficient noising processes with arbitrary rate matrix. Our training algorithm relies on a simple loss function based on snapshot samples instead of the entire noising path, and is compatible with standard architectures. GDDS beats previous discrete diffusion models as well as autoregressive models for the first time at this scale on language modeling tasks. Future work may build upon this approach and propose different Semantic-Informed Kernels (SIK) to enforce meaningful noising processes based on similarities between words.

#### Impact Statement

This paper presents work that aims to advance discrete diffusion-based generative modeling in machine learning. There are many potential societal consequences of our work, none of which must be specifically highlighted here.

#### Acknowledgements

The authors would like to thank Jules Samaran for precious advice. This project was provided with computing (HPC) and storage resources by GENCI at IDRIS thanks to the grant 2025-AD011017039 on the supercomputer Jean Zay’s V100/A100/H100 partition.

#### References

Amin, A. N., Gruver, N., and Wilson, A. G. Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion. In Advances on Neural Information Processing Systems, volume 38, 2025.

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and Van Den Berg, R. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, volume 34, pp. 17981–17993, 2021.

Bartlett, M. S. An inverse matrix adjustment arising in discriminant analysis. Ann. Math. Stat., 22(1):107–111, 1951.

Bengio, Y., Yao, L., Alain, G., and Vincent, P. Generalized denoising auto-encoders as generative models. In Advances in Neural Information Processing Systems, volume 26, 2013.

Brockett, R. W. Finite dimensional linear systems. SIAM, 2015.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.

Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Deligiannidis, G., and Doucet, A. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, volume 35, pp. 28266–28279, 2022.

Campbell, N. The study of discontinuous phenomena. Proc Cambr. Phil. Soc, 15:117–136, 1909.

Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., and Robinson, T. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, 2013.

Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., and Goharian, N. A discourse-aware attention model for abstractive summarization of long documents. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 615–621, 2018.

Dingle, N. J., Harrison, P. G., and Knottenbelt, W. J. Uniformization and hypergraph partitioning for the distributed computation of response time densities in very large Markov models. J. Parallel Distrib. Comput., 64(8): 908–920, 2004.

Gokaslan, A. and Cohen, V. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.

Graves, A., Srivastava, R. K., Atkinson, T., and Gomez, F. Bayesian flow networks. arXiv preprint arXiv:2308.07037, 2023.

Gulrajani, I. and Hashimoto, T. B. Likelihood-based diffusion language models. In Advances in Neural Information Processing Systems, volume 36, pp. 16693–16715, 2023.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.

Hoogeboom, E., Nielsen, D., Jaini, P., Forr´e, P., and Welling, M. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, volume 34, pp. 12454–12465, 2021.

Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B., Berg, R. v. d., and Salimans, T. Autoregressive diffusion models. In International Conference on Learning Representations, 2022.

Jensen, A. Markoff chains as an aid in the study of Markoff processes. Scand. Actuar. J., 1953(1):87–91, 1953.

Kelly, F. P. Reversibility and stochastic networks. Cambridge University Press, 2011.

Kim, J., Shah, K., Kontonis, V., Kakade, S., and Chen, S. Train for the worst, plan for the best: Understanding token ordering in masked diffusions. arXiv preprint arXiv:2502.06768, 2025.

Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021.

Lai, C.-H., Song, Y., Kim, D., Mitsufuji, Y., and Ermon, S. The principles of diffusion models. arXiv preprint arXiv:2510.21890, 2025.

Last, G. and Penrose, M. Lectures on the Poisson process. Cambridge University Press, 2018.

Li, T. and He, K. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025.

Li, T., Chen, M., Guo, B., and Shen, Z. A survey on diffusion language models. arXiv preprint arXiv:2508.10875, 2025.

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Liu, H., Chen, Z., Yuan, Y., Mei, X., Liu, X., Mandic, D., Wang, W., and Plumbley, M. D. Audioldm: Text-to-audio generation with latent diffusion models. In International Conference on Machine Learning, 2023.

Lou, A., Meng, C., and Ermon, S. Discrete diffusion language modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024.

Mahoney, M. Text8. https://mattmahoney.net/ dc/textdata.html, 2024. Accessed: 2025-01-05.

Marcus, M. P., Santorini, B., and Marcinkiewicz, M. A. Building a large annotated corpus of English: The Penn Treebank. Comput. Linguist., 19(2):313–330, 1993.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017.

Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. Large language diffusion models. In Advances in Neural Information Processing Systems, 2025.

Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., and Li, C. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In International Conference on Learning Representations, 2025.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N. Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fern´andez, R. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Association for Computational Linguistics, pp. 1525–1534, 2016.

Pauline, V., H¨oppe, T., Neklyudov, K., Tong, A., Bauer, S., and Dittadi, A. Foundations of diffusion models in general state spaces: A self-contained introduction. arXiv preprint arXiv:2512.05092, 2025.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In International Conference on Computer Vision, pp. 4195–4205, 2023.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, volume 35, pp. 36479–36494, 2022.

Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J., Rush, A., and Kuleshov, V. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems, volume 37, pp. 130136–130184, 2024.

Sahoo, S. S., Deschenaux, J., Gokaslan, A., Wang, G., Chiu, J., and Kuleshov, V. The diffusion duality. In International Conference on Machine Learning, 2025.

Schiff, Y., Sahoo, S. S., Phung, H., Wang, G., Boshar, S., Dalla-torre, H., de Almeida, B. P., Rush, A., Pierrot, T., and Kuleshov, V. Simple guidance mechanisms for discrete diffusion models. In International Conference on Learning Representations, 2025.

Shi, J. and Titsias, M. K. Demystifying diffusion objectives: Reweighted losses are better variational bounds. arXiv preprint arXiv:2511.19664, 2025.

Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems, volume 37, pp. 103131–103167, 2024.

Shih, A., Sadigh, D., and Ermon, S. Training and inference on any-order autoregressive models the right way. In Advances in Neural Information Processing Systems, volume 35, pp. 2762–2775, 2022.

Song, K., Tan, X., Qin, T., Lu, J., and Liu, T.-Y. Mpnet: Masked and permuted pre-training for language understanding. In Advances in Neural Information Processing Systems, volume 33, pp. 16857–16867, 2020.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Stewart, W. J. Probability, Markov chains, queues, and simulation: the mathematical basis of performance modeling. Princeton University Press, 2009.

Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Tran, D., Vafa, K., Agrawal, K., Dinh, L., and Poole, B. Discrete flows: Invertible generative models of discrete data. In Advances in Neural Information Processing Systems, volume 32, 2019.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

von R¨utte, D., Fluri, J., Ding, Y., Orvieto, A., Sch¨olkopf, B., and Hofmann, T. Generalized Interpolating Discrete Diffusion. In International Conference on Machine Learning, 2025a.

von R¨utte, D., Fluri, J., Pooladzandi, O., Sch¨olkopf, B., Hofmann, T., and Orvieto, A. Scaling behavior of discrete diffusion language models. arXiv preprint arXiv:2512.10858, 2025b.

Wiedemer, T., Li, Y., Vicol, P., Gu, S. S., Matarese, N., Swersky, K., Kim, B., Jaini, P., and Geirhos, R. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., and Le, Q. V. XLNet: Generalized Autoregressive Pretraining for Language Understanding. In Advances in Neural Information Processing Systems, volume 32, 2019.

Zelnik-Manor, L. and Perona, P. Self-tuning spectral clustering. In Advances in Neural Information Processing Systems, volume 17, 2004.

Zhang, X., Zhao, J., and LeCun, Y. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28, 2015.

Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., and Zhang, Q. Masked diffusion models are secretly timeagnostic masked models and exploit inaccurate categorical sampling. In International Conference on Learning Representations, 2025.

Zhou, C., Wang, C., Zhang, D., Tong, S., Wang, Y., Bates, S., and Jaakkola, T. Next semantic scale prediction via

hierarchical diffusion language models. In Advances in Neural Information Processing Systems, volume 38, 2025.

Ziegler, Z. and Rush, A. Latent normalizing flows for discrete sequences. In International Conference on Machine Learning, pp. 7673–7682, 2019.

## Appendix

Throughout the appendix, non-bold symbols refer to token-level objects on V, while bold symbols refer to their sequencelevel counterparts on Vn for n ≥ 1.

#### A. Time-dependent matrix exponential

###### A.1. Transition operator for time-inhomogeneous CTMCs

Let t ≥ 0, and consider two real d × d matrices Q, Qt to be time-independent and time-dependent. The matrix exponential of Q is defined by the power series: exp(Q) = ∞k=0 Qk/k!. When Qt varies with time, one naturally extends this notion by considering the matrix Kt,s with 0 ≤ s ≤ t that solves the following matrix-valued linear ODE:

d dt

Kt,s = QtKt,s, Ks,s = Im.

Its solution can be written using the time-ordered exponential (or Peano–Baker series) (Brockett, 2015, Thm 1.3.1),

Kt,s = T exp

t

Qτ dτ , (10)

s

where T is a time-ordering operator that orders products by decreasing time (later times to the left) as

t

t

t

τ1

t

τ1

τ2

T exp

dτ3 dτ2 dτ1 + ··· ,

Qτ dτ = Id +

Qτ

dτ1 +

Qτ

Qτ

dτ2 dτ1 +

Qτ

Qτ

Qτ

1

1

2

1

2

3

s

s

s

s

s

s

s

. Note that if the family {Qτ}τ∈[s,t] commutes pairwise (QτQτ′ = Qτ′Qτ for all τ,τ′ ∈ [s,t]), time ordering is unnecessary and Kt,s = exp( s t Qτ dτ). The time-ordered exponential defined in Eq. (10) is useful to describe the fundamental solution to the Kolmogorov forward equation (see Eq. (3)), which describes the time evolution of a probability vector qt governed by a (possibly time-dependent) rate matrix Qt. More precisely, the matrix Kt,s acts as the linear operator that maps an initial distribution qs to its future value qt as

i.e., the k-fold term integrates over s ≤ τk ≤ ··· ≤ τ1 ≤ t the ordered product Qτ

1 ···Qτ

k

qt = Kt,sqs = T exp

t

Qτ dτ qs.

s

For any initial condition x ∈ X, the evolution of this point mass under the forward dynamics is obtained by considering q0 = δx ∈ Rm at s = 0, i.e., qt(· | x) = Kt,0δx. Hence, the x-th column of Kt,s corresponds exactly to the conditional distribution of the CTMC transition matrix at time t given that it was at state x at time s, that is,

Kt,s(y,x) = P(xt = y | xs = x).

In this sense, the time-ordered exponential provides the explicit operator representation of the transition kernel solving the Kolmogorov forward equation.

###### A.2. Coordinate-wise rate matrix and product-form marginals

Recall that states are sequences x = x1 ...xn ∈ Vn, and that jumps only modify one token at a time with token-level generator Qℓt ∈ Rm×m at position ℓ ∈ {1,...,n}. Then, the joint rate matrix decomposes as a sum of coordinate-wise generators:

n

n×mn,

Im ⊗ ··· ⊗ Qℓt ⊗ ··· ⊗ Im ∈ Rm

Qt =

ℓ=1

where Qℓt = Qt for all 1 ≤ ℓ ≤ n acts on coordinate ℓ and identities on the others (Pauline et al., 2025). Here, A ⊗ B denotes the Kronecker product of matrices A and B. This structure implies that each coordinate (xℓt)1≤ℓ≤n evolves as an independent time-inhomogeneous CTMC with rate matrix Qt. In fact, let qt(· | xℓ) solve the Kolmogorov forward equation:

d dt

qt(· | xℓ) = Qtqt(· | xℓ), q0(· | xℓ) = δxℓ.

Define q˜t(· | x) := nℓ=1 qt(· | xℓ). By the product rule and the generator decomposition above, q˜t solves the joint forward equation

d dt

q˜t(· | x) = Qtq˜t(· | x), q˜0(· | x) = δx.

Uniqueness of solutions to the forward equation Eq. (3) yields qt(· | x) = q˜t(· | x) = nℓ=1 qt(· | xℓ), i.e., the conditional marginal factorizes across coordinates.

#### B. Technical results

###### B.1. CTMC rate matrices

- Proof of Proposition 3.1. Let t ∈ (0,1) such that Kt is invertible. For each x ∈ V, we denote by v(t,x) the solution to linear system Kt⊤v(t,x) = K˙ t⊤δx, which exists and is unique. We define a matrix Qt ∈ Rm×m by prescribing its x-th column as

Q⊤t δx := v(t,x), for all x ∈ V.

By construction, for every x ∈ V, we have K˙ t⊤δx = Kt⊤(Q⊤t δx), hence K˙ t⊤ = Kt⊤Q⊤t , which is equivalent to QtKt = K˙ t hence Qt = K˙ tKt−1 since Kt is invertible. For any x ̸= y ∈ V, we have

Qt(y,x) = (Q⊤t δx)[y] = v[(yt,x] ) ≥ 0.

Therefore, the off–diagonal entries of Qt are nonnegative. Since Πt is column–stochastic and 0 ≤ αt ≤ 1, each Kt is column–stochastic: 1⊤Kt = 1⊤(αtIm + (1 − αt)Πt) = 1⊤. Differentiating with respect to t gives 1⊤K˙ t = 0. Using the identity K˙ t = KtQt, we have 0 = 1⊤K˙ t = 1⊤KtQt = 1⊤Qt, because 1⊤Kt = 1⊤. Hence every column of Qt sums to 0. Consequently, for each x,

Qt(x,x) = −

Qt(y,x) ≤ 0.

y̸=x

Let px(t) := Ktδx be the law at time t when starting from state x at time 0. Then,

d dt

px(t) = K˙ tδx = QtKtδx = Qtpx(t),

with px(0) = K0δx = δx because α0 = 1 implies K0 = Im. Thus, Qt drives the Kolmogorov forward equation, and since its off–diagonals are nonnegative and its columns sum to zero, Qt is a valid CTMC rate matrix. Combining all previous steps yields the claimed expression Qt = K˙ tKt−1.

Corollary B.1 (GIDD case Πt = πt1⊤, von R¨utte et al. 2025a). Let t ≥ 0,πt ∈ ∆m and select Πt = πt1⊤ ∈ Rd×d, then

α˙t αt

α˙t αt

πt1⊤ + (1 − αt)˙πt1⊤.

Im −

Qt =

Proof. We begin by verifying that our regularity assumption on Kt holds. Following the choice Πt = πt1⊤, we have for all t ∈ [0,1],

Kt = αtIm + (1 − αt)πt1⊤, and K˙ t = α˙tIm − α˙tπt1⊤ + (1 − αt)π˙t1⊤.

Since Kt is a rank-1 update to an invertible matrix, the Sherman–Morrison formula (Bartlett, 1951) ensures that it is invertible and yields the following closed form for its inverse:

1 αt

Kt−1 =

(1 − αt)πt1⊤ αt(1 + 1−α

Im −

αt )

t

1 αt

(Im − (1 − αt)πt1⊤), t ∈ (0,1). (11)

=

Moreover for all x ∈ V and for all t ∈ (0,1), one has to verify that v(t,x) = (Kt⊤)−1K˙ t⊤δx ≥ 0, i.e. that (1 − αt)˙πt(x) − α˙t αt πt(x) ≥ 0, which is true since α˙t ≤ 0 because αt is a decreasing function. Then, combining Proposition 3.1 and Eq. (11)

yields

1 αt

Qt = K˙ tKt−1 = α ˙tIm − α˙tπt1⊤ + (1 − αt)˙πt1⊤

Im − (1 − αt)πt1⊤

1 − αt αt

α˙t αt

α˙t αt

Im − (1 − αt)πt1⊤ −

πt1⊤ − (1 − αt)πt1⊤πt1⊤ +

=

π ˙t1⊤ − (1 − αt)˙πt1⊤πt1⊤ .

Using the identity 1⊤πt = 1, the previous equality simplifies to

Qt =

α˙t αt

Im −

α˙t αt

πt1⊤ + (1 − αt)˙πt1⊤.

| |
|---|

- Proof of Proposition 3.2. Following the definition of Kt in Eq. (6), for any t ∈ [0,1], Kt = αtIm + (1 − αt)Πt. Differentiating the equation with respect to t yields:

K˙ t = α˙tIm + (1 − αt)Π˙ t − α˙tΠt = (1 − αt)Π˙ t + α˙t(Im − Πt). (12) Let Πt be the matrix defined as the unique solution to the following linear matrix-valued ODE:

Π0 = Im −

Q0 α˙0

, and (1 − αt)Π˙ t = (αtQt − α˙tIm)(Im − Πt) + QtΠt, t ∈ [0,1],

which is valid since 0 < 1 − αt ≤ 1 for all t ∈ (0,1]. Inserting the ODE into Eq. (12) yields

K˙ t = (1 − αt)Π˙ t + α˙t(Im − Πt) = (αtQt − α˙tIm)(Im − Πt) + QtΠt + α˙t(Im − Πt)

= αtQt − αtQtΠt − α˙tIm + α˙tΠt + QtΠt + α˙tIm − α˙tΠt

= αtQt + (1 − αt)QtΠt = Qt(αtIm + (1 − αt)Πt) = QtKt.

For the initial condition, using the Taylor expansion αt = 1 + α˙0t + o(t) as t ↓ 0, we obtain

lim

t↓0

Kt = lim t↓0

(αtIm + (1 − αt)Πt) = Im.

Hence, for t ∈ [0,1], Kt satisfies the linear matrix-valued ODE:

K˙ t = QtKt, K0 = Im,

whose unique solution is Kt = T exp( 0 t Qs ds) for all t ∈ [0,1] (Brockett, 2015, Theorem 1.3.1). Taking the column associated to the token x on both sides yields Kt(·,x) = T exp( 0 t Qs ds)(·,x), so the induced marginals coincide with those of the CTMC.

| |
|---|

- Remark B.1. Note that we never explicitly use the expression Π0 = Im − Q

0

α˙0 in the proof. We only require Π0 to be finite; the above choice merely ensures that Πt is uniquely defined as a continuous mixing matrix on [0,1]. In fact, we set Π0 = Im − Q

0

α˙0 because 1 − αt → 0 makes the coefficient of Π˙ t in the ODE vanish at t = 0, and keeping Π˙ t finite requires (Q0 − α˙0Im) + α˙0Π0 = 0.

B.2. Uniformization and exact sampling

- Proof of Proposition 3.3. Let (Zt)t≥0 be a CTMC with rate matrix Qt = f(t)(Ft − Im), started at Z0 = x. Let Nt

be its jump count process. Here, Nt is a non-homogeneous Poisson point process with parameter f¯(t) = 0 t f(s)ds with marginal distribution P(Nt = k) = exp(−f¯(t))f¯(t)k/k! for all k ∈ N. Conditionally on the event (Nt = k), let

- 0 < T1 < ... < Tk < t denote the jump times. For each 1 ≤ r ≤ k, the transition probabilities at a jump are given by the

columns of FT

, namely P(ZT

###### (i,j). Therefore, for any i,j ∈ {1,...,m}, P(Zt = i,Nt = k | Z0 = x) = E[(FT

###### = i | ZT

###### = j,Tr) = FT

r−1

r

r

r

t=k | Z0 = j].

###### ...FT

)(i,j)1N

1

k

Summing over k ≥ 0 yields

Kt(i,j) = P(Zt = i | Z0 = x) =

###### E[(FT

###### ...FT

)(i,j)1N

t=k],

1

k

k≥0

with the convention that FT

is the identity matrix when k = 0. Splitting the terms at k = 0 yields Kt(i,j) = P(Nt = 0)1i=j +

###### ...FT

0

1

###### E[(FT

###### ...FT

)(i,j)1N

t=k],

1

k

k≥1

Since P(Nt = 0) = αt, we identify Kt(i,j) = αt1i=j + (1 − αt)Πt(i,j), where

1 1 − αt k≥1

E[(FT

t=k] = E (FT

)(i,j) | Nt ≥ 1 ,

###### ...FT

Πt(i,j) :=

###### ...FT

)(i,j)1N

1

1

Nt

k

as P(Nt ≥ 1) = 1 − exp(−f¯(t)) = 1 − αt. Note that Πt is a correctly defined mixing matrix. Indeed, for a fixed k and any time sequence 0 < t1 < ... < tn < t, each FT

as well. Hence, the random matrix FT

is column stochastic and the product FT

###### ...FT

r

1

k

conditioned on (Nt ≥ 1) is column stochastic almost surely. Taking the conditional expectation (as a convex combination) preserves this property. Comparing this expression with the interpolating form Kt = αtIm + (1 − αt)Πt yields the unique

###### ...FT

1

Nt

Πt = E (FT

) | Nt ≥ 1 , as claimed, where uniqueness follows from Proposition 3.2.

###### ...FT

1

Nt

| |
|---|

- Remark B.2 (Exact sampling of xt ∼ qt(· | x0)). For a given continuous time t ∈ [0,1], the sampling of xt ∼ qt(· | x0) can be performed exactly using only a Poisson random variable sampling, uniform random variable sampling, and columns of Ft as follows.

- 1. Sample Nt ∼ Poisson(f¯(t)).
- 2. If Nt = 0, set xt ← x0.
- 3. If Nt ≥ 1, set z0 ← x then:

- (a) sample ordered jump time 0 < T1 < ... < TN

t ≤ t according to the Poisson point process with parameter f¯(t). This can be done by sampling i.i.d. Ur ∼ U([0,1]), set Vr = f¯−1(Urf¯(t)) and take the ordered statistics of V1,...,VN as T1,...,TN

t

.

- (b) for each 1 ≤ r ≤ Nt, sample the next state zr using the zr−1-th column of Ft as

P(zr = i | zr−1) = FT

r

(i,zr−1) for i = 1,...,m.

- (c) Set xt ← zN

.

t

This algorithm produces the conditional law P(xt = x | x0 = y,Nt = k,T1 = t1,...,Tk = tk) = Ft

(x,y). Marginalizing over (Nt,T1,...,TN

...Ft

1

k

) yields P(xt = x | x0 = y) =

t

###### E[(FT

###### ...FT

)(x,y)1N

t=k] = Kt(x,y).

1

k

k≥0

Note that this procedure is exact.

###### B.3. Path-wise parametrization and Evidence Lower Bound (ELBO)

- B.3.1. REVERSE PROCESS

Forward kernels and true reverse conditionals. Fix 0 ≤ u < t ≤ 1 and consider a time-inhomogeneous forward Markov process:

x0 → xu → xt, where q(x0,xu,xt) = qdata(x0)qu(xu | x0)qt|u(xt | xu),

Algorithm 3 Exact general noising, sequence level (parallel token level)

- 1: Input: clean sequence x0 = x10 ...xn0 of length n ≥ 1, time t ∈ [0,1], intensity f¯(t), rate matrix Qt as in Eq. (7)
- 2: for ℓ = 1 to n in parallel do
- 3: Set z0ℓ ← xℓ0
- 4: Sample number of jumps Ntℓ ∼ Poisson(f¯(t))
- 5: Sample and sort the jump times as T1ℓ < ... < TNℓ ℓ

t

- 6: for k = 1 to Ntℓ do
- 7: Sample jump zkℓ ∼ FTℓ

k

(·,zkℓ−1)

- 8: end for
- 9: Set xℓt ← zNℓ ℓ

t

- 10: end for
- 11: return noised sequence xt = x1t ...xnt and per-token jumps (Tkℓ,zkℓ)N

ℓ t

n ℓ=1

k=1

and qt|u denotes the forward transition kernel from time u to t (we use the shorthand qu := qu|0). The forward marginal from x0 to time t is

qt|u(xt | xu)qu(xu | x0). (13)

qt(xt | x0) =

xu∈V

By Bayes’ rule, the true reverse conditional given x0 reads

qt|u(xt | xu,x0)qu(xu | x0) xu∈V qt|u(xt | xu)qu(xu | x0)

qt|u(xt | xu)qu(xu | x0) qt(xt | x0)

, (14)

q(xu | xt,x0) =

=

where the second equality uses that the forward process is Markov hence qt|u(xt | xu,x0) = qt|u(xt | xu) and the marginalization (13). Finally, conditioning on the observed endpoint xt, the true reverse kernel is the mixture

q(xu | xt,x0)q0|t(x0 | xt), (15)

q(xu | xt) =

x0∈V

where q0|t(x0 | xt) is the exact forward posterior of x0 given the snapshot s = (xt,t).

Plug-in Bayes and realizability. At inference time x0 is unknown while (xt,t) is observed. Let µθ(· | xt,t) be a neural predictor (e.g. a transformer head; Vaswani et al. 2017) that outputs a distribution over x0 ∈ V given (xt,t). We define the plug-in reverse transition by mixing the exact conditional (14) with µθ as

q(xu | xt,x0)µθ(x0 | xt,t). (16)

pθ(xu | xt,t) :=

x0∈V

This parametrization is widely used in the literature (Austin et al., 2021; Campbell et al., 2022; Sahoo et al., 2024; Shi et al., 2024; von R¨utte et al., 2025a; Zhou et al., 2025). It is interesting, as it verifies the following lemma.

Lemma B.2 (Realizable limit). If µθ(· | xt,t) = q0|t(· | xt), then pθ(· | xt,t) equals the true reverse kernel:

###### pθ(xu | xt,t) = q(xu | xt).

Proof. If µθ(· | xt,t) = q0|t(· | xt), then substituting into Eq. (16) gives

pθ(xu | xt,t) =

where the last equality is exactly (15).

q(xu | xt,x0)q0|t(x0 | xt) = q(xu | xt),

x0∈V

| |
|---|

A first crucial insight is that, µθ(· | xt,t) should be train to approach the posterior q0|t(· | xt), so that pθ(xu | xt,t) approaches q(xu | xt).

CTMC view and factorization of the reverse dynamics. For an infinitesimal step u = t − ϵ with ϵ ↓ 0, the true reverse transition admits the standard “no-jump + jump” expansion:

t](t)Rt(·,xt) + o(ϵ), where r[x

q(xt−ϵ | xt) = (1 − ϵr[x

t](t))δx

+ ϵr[x

t

t](t) is the true reverse exit rate from state xt and Rt(·,xt) ∈ ∆m is the true jump destination distribution. Equivalently, the reverse generator factorizes as

Qt = (Rt − Im)diag(r1(t),...,rm(t)), i.e.,

r[x](t)Rt(y,x), y ̸= x, −r[x](t), y = x.

Qt(y,x) =

This factorization separates when the chain jumps (through r[x](t)) from where it jumps (through Rt).

Why the plug-in Bayes kernel entangles “where” and “when”. Eq. (16) induces a time-inhomogeneous reverse kernel whose short-time behaviour is governed by a θ-dependent generator. Indeed, for an infinitesimal step u = t − ϵ with ϵ ↓ 0, the induced kernel pθ(xt−ϵ | xt,t) admits a first-order expansion of the form

pθ(xt−ϵ | xt,t) = (1 − ϵr[θx

###### + ϵr[θx

t](t)Aθt(·,xt) + o(ϵ), where both the effective exit rate r[θx

t](t))δx

t

t](t) and the jump destination distribution Aθt(·,xt) ∈ ∆m depend on θ through µθ(· | xt,t). Equivalently, the induced reverse generator Qθt factorizes as

###### Qθt = (Aθt − Im)diag(r1θ(t),...,rmθ (t)), (17)

so that learning µθ implicitly learns both a jump kernel and a time-dependent clock. In other words, the Bayes plug-in construction couples where the chain jumps (through Aθt(·,xt)) and when it jumps (through r[θx

t](t)), which complicates the path-wise ELBO and its optimization: the event-level objective contains gradients through both the destination cross-entropy term and the rate/normalization term, rather than isolating a single θ-dependent jump component.

Jump-states parametrization: learn only the jump kernel, fix the exit rates. To mirror the true CTMC factorization, we instead parameterize only the jump destinations while prescribing the exit rates by a diffusion schedule. Concretely, we choose a nonnegative schedule r[x](t) ≥ 0 independently of θ and define the neural reverse generator

Qθt = (Rtθ − Im)diag(r1(t),...,rm(t)), where Rtθ(y,x) = jθ(x,t)[y], (18) such that

r[x](t)jθ(x,t)[y], y ̸= x, −r[x](t), y = x.

Qθt(y,x) =

Equivalently, for u = t − ϵ the associated reverse kernel satisfies the first-order expansion:

pθ(xt−ϵ | xt) = (1 − ϵr[x

t](t)jθ(xt,t)x

t](t))δx

+ ϵr[x

###### + o(ϵ).

t−ϵ

t

Compared to the plug-in Bayes transition in Eq. (16), the network now controls only where the chain jumps (via jθ(xt,t)), while when it jumps is entirely prescribed by the schedule r[x

###### t](t).

- B.3.2. PROOF OF PROPOSITION 4.1

The following proposition is valid for any general forward CTMC and its parametrized time reversal. It has been established and rewritten in several previous works (e.g., Campbell et al. 2022; Shi et al. 2024; von R¨utte et al. 2025a; Zhou et al. 2025), and we restate it with our notations.

Proposition B.3 (General Path-wise ELBO, Campbell et al. 2022). Let x0 ∈ V, the ELBO is given by log pθ,0 path(x0) ≥ −L˜pathx0 (θ) + C˜xpath

###### , where C˜xpath

is independent of θ and

0

0

 −Qθt(xt,xt) +

  dt.

1

L˜pathx0 (θ) =

Qt(y,xt)(−log Rtθ(xt,y))

Ex

t∼qt(·|x0)

0

y̸=xt

Let us apply Proposition B.3 to our jump-states parametrization detailed in Section 4.2. In our case, Qθt(xt,xt) = −r[x

t](t) is θ-independent (see Eq. (9)), we can discard it from the θ-dependent part, and define

Lpathx0 (θ) :=

 

1

Ex

t∼qt(·|x0)

0

  dt, (19)

Qt(y,xt)(−log Rtθ(xt,y))

y̸=xt

− 0 1 Ex

t](t)]dt. We can then rewrite the ELBO as log pθ,0 path(x0) ≥ −Lpathx0 (θ)+Cxpath

###### := C˜xpath

. We now define the following quantities similarly to (Shi et al., 2024, Lem. 2).

and Cxpath

t∼qt(·|x0)[r[x

0

0

0

Definition B.4 (True conditional reverse jump kernel Rx

x0(t)). Fix x0 ∈ V and let qt(· | x0) be the forward marginal at time t. For x ∈ V with qt(x | x0) > 0, define the conditional reverse jump kernel Rx

t and conditional reverse rate rx

0

t (·,x) by, for all y ̸= x,

0

qt(y | x0) qt(x | x0)

qt(y | x0) qt(x | x0)

Qt(x,y) rx

Rx

, where rx

t (y,x) :=

Qt(x,y).

[x](t) :=

0

0

[x](t)

0

y̸=x

If qt(x | x0) = 0, the value of Rx

t (·,x) is irrelevant under xt ∼ qt(· | x0) and is set arbitrarily to 0.

0

To conclude the proof, we provide the following lemma which gives two clean rewriting of the θ-dependent part Lpathx0 (θ).

Lemma B.5. The quantity Lpathx0 (θ) defined in Eq. (19) satisfies the following identities:

 

  dt

1

qt(y | x0) qt(xt | x0)

Lpathx0 (θ) =

Qt(xt,y)(−log Rtθ(y,xt))

Ex

t∼qt(·|x0)

0

y̸=xt

1

t∼qt(·|x0) rx

###### [xt](t)CE(Rx

t (·,xt),Rtθ(·,xt)) dt.

Ex

=

0

0

0

Here, CE(Rx

denotes the cross-entropy between Rx

###### t (·,xt) and Rtθ(·,xt). Rx

(t) denote respectively the conditional reverse jump kernel and its associated exit rate introduced in Definition B.4.

t and rx

###### t ,Rtθ) x

0

0

0

0

t

Proof. Let t ∈ [0,1], we start from the definition of Lpathx0 (θ) in Eq. (19) and expands the the expectation term inside the time integral to obtain

Ex

t∼qt(·|x0)

Qt(y,xt) − log Rtθ(xt,y) =

y̸=xt

=

Qt(y,x) − log Rtθ(x,y)

qt(x | x0)

x∈V

y̸=x

qt(x | x0)Qt(y,x) − log Rtθ(x,y) .

x∈V y̸=x

Swapping the variables names (x,y)  → (y,x) inside the double sum yields

qt(y | x0)Qt(x,y) − log Rtθ(y,x) (20)

Qt(y,xt) − log Rtθ(xt,y) =

Ex

t∼qt(·|x0)

x∈V y̸=x

y̸=xt

qt(y | x0) qt(x | x0)

Qt(x,y) − log Rtθ(y,x)

qt(x | x0)

=

x∈V

y̸=x

qt(y | x0) qt(xt | x0)

Qt(xt,y) − log Rtθ(y,xt) .

= Ex

t∼qt(·|x0)

y̸=xt

Integrating over t gives the first identity. For the second identity, we note that for y ̸= x, Definition B.4 can be restated as

qt(y | x0) qt(x | x0)

###### [x](t)Rx

Qt(x,y) = rx

###### t (y,x).

0

0

Plugging this into the inner sum in Eq. (20) yields, for each x ∈ V,

qt(y | x0) qt(x | x0)

Qt(x,y) − log Rtθ(y,x) = rx

Rx

###### t (y,x) − log Rtθ(y,x) = rx

###### [x](t)CE Rx

###### t (·,x),Rtθ(·,x) .

[x](t)

0

0

0

0

y̸=x

y̸=x

Taking the expectation over xt ∼ qt(· | x0) and integrating over t ∈ [0,1] provides the desired quantity:

1

t∼qt(·|x0) rx

###### [xt](t)CE Rx

Lpathx0 (θ) =

###### t (·,xt),Rtθ(·,xt) dt.

Ex

0

0

0

| |
|---|

A notable simplification comes from the jump-state parametrization. As shown in Lemma B.5, the only θ-dependent contribution is a weighted cross-entropy over reverse jump matrices. By contrast, in the SEDD formulation of Lou et al. (2024), the ELBO contains two θ-dependent components: (i) a linear score term and (ii) a cross-entropy term. In the mean parametrization of Campbell (1909) (recalled in Proposition B.3), the linear score term is replaced by a θ-dependent reverse-rate term. In both settings, training must therefore fit both jump destinations and a separate θ-dependent factor that controls the time-change, which adds burden and variance. Our jump-state parametrization removes this extra requirement: it keeps the reverse rate fixed and concentrates learning on the core problem (predicting where the chain jumps) via a single cross-entropy objective. The jump-state parametrization is closely related to the approach of Amin et al. (2025). However, we do not condition the path-wise generative model on an explicit event schedule; instead, we keep the same joint distribution defining the generative model as in prior work.

- B.3.3. PROOF OF PROPOSITION 4.2 Let x0 ∼ q0 and define the forward jump flow measure on the space Ω := [0,1] × V × V as

###### Λx

###### (dt, dy, dx) := qt(x | x0)Qt(y,x)1{y̸=x} dt.

0

We then rewrite the expression of Lpathx0 (θ) in Eq. (19) as

 

  dt =

1

Lpathx0 (θ) =

Qt(y,x)(−log Rtθ(x,y))

−log Rtθ(x,y)Λx

###### (dt, dy, dx). (21)

Ex∼q

0

t(·|x0)

0

Ω

y̸=x

###### (dt, dy, dx) := Nk=11 δ(T

We now introduce J x

k,zk,zk−1)(dt, dy, dx), the jump random measure on Ω of the forward CTMC started at x0. Here, N1 and (Tk,zk)1≤k≤N

0

are the output of Algorithm 1 with inputs x0, t = 1 and Qt. Equivalently, this notation means that for any measurable set A ⊂ Ω,J x

1

(A) = Nk=11 1{(T

k,zk,zk−1)∈A}. Its intensity measure is exactly Λx

0

0 as for any measurable set A ⊂ Ω, we have E[J x

(A). To see this, it suffices to verify the identity on a

###### (A)] = Λx

0

0

rectangle A = (a,b] × {y} × {x} with 0 ≤ a < b ≤ 1 and x ̸= y ∈ V, and extend to all measurable sets by a monotone class argument. For such a rectangle, we have

###### E[J x

(A)] = E

0

N1

k∈(a,b],zk=y,zk−1=x} =: E C(xa,b→y] ,

1{T

k=1

where (xt)t∈[0,1] is the forward CTMC started at x0, and C(xa,b→y] is the number of jumps between states x → y in the time interval (a,b]. Let r ≥ 1, we consider a uniform partition of (a,b] as t0 = a < ... < b = tr with step-size ∆ = (b − a)/r. The Markov property and the generator give E[C(xt→y

(y,x)∆ + o(∆)). Summing over 0 ≤ i ≤ r − 1 yields a Riemann sum along with a remainder term of the form ri=0−1 o(∆), which vanishes as the mesh goes to 0. Therefore, E[C(xa,b→y] ] = a b P(xt = x)Qt(y,x)dt and, since P(xt = x) = qt(x | x0), we obtain E[J x

###### i,ti+1]] = P(Xt

###### = x)(Qt

i

i

(A). Hence, Campbell’s formula (Last & Penrose, 2018, Prop. 2.7) states that for any nonnegative measurable function g : Ω → R≥0,

###### (A)] = Λx

0

0

N1

g(t,y,x)Λx

###### (dt, dy, dx). (22)

###### E(T

g(Tk,zk,zk−1) =

0

k,zk,zk−1)k

Ω

k=1

To simplify the notations, we introduce ω = {N1,(zk,Tk)N

[0,1](·|x0). Choosing g(t,y,x) = −log Rtθ(x,y) ≥ 0 and combining Eqs. (21) and (22) yield

k=1} and q[0,1](· | x0) such that E(T

k,zk,zk−1)k = Eω∼q

1

N1

N1

−log RTθ

Lpathx0 (θ) = Eω∼q

###### (zk−1,zk) = Eω∼q

−log jθ(zk,Tk)z

###### ,

[0,1](·|x0)

[0,1](·|x0)

k−1

k

k=1

k=1

by definition of jθ. For an expression at the sequence level (of size n ≥ 1), let x0 = x10 ...xn0 ∼ qdata. Using qt(· | x0) = nℓ=1 qt(· | xℓ0) implies that that the sequence-level measure is the superposition of the n independent token jump measure J x

ℓ

0 (i.e., Jx

ℓ

= nℓ=1 J x

0). The linearity of the expectation implies that

- 0

 

 ,

N1ℓ

n

−log RTθ ℓ

(zkℓ−1,zkℓ)

###### L(θ) = Eω∼q

[0,1](·|x0)

k

ℓ=1

k=1

where ω = {ωℓ}nℓ=1 ∼ q[0,1](· | x0) with ωℓ = {N1ℓ,(Tkℓ,zkℓ)k}, and q[0,1](· | x0) = nℓ=1 q[0,1](· | xℓ0).

- B.3.4. RECOVERING THE MASKED DIFFUSION LOSS

For masked diffusion, (Ou et al., 2025; Sahoo et al., 2024; Shi et al., 2024) concurrently discovered a simplified expression of the (Campbell et al., 2022)-ELBO, that collapses with Proposition 4.1. This is due to the fact that masked diffusion makes the reverse exit rates independent of θ (Amin et al., 2025). Hence, the seminal parametrization of (Austin et al., 2021; Campbell et al., 2022) collapses with our jump-states parametrization in this special case. To our knowledge, this property is only true for masked diffusion, and other attempts of generalization from the parametrization of (Austin et al., 2021; Campbell et al., 2022) led to more complicated losses than the weighted cross-entropy given by our jump-states parametrization (see e.g., von R¨utte et al. 2025a; Zhou et al. 2025).

Consider the forward generator Qabsorbt = f(t)(Fabsorb − Im) with a single mask state [MASK] (i.e., Ft = Fabsorb = δ[MASK]1⊤, or equivalently Πt = Π = δ[MASK]1⊤), so that Qabsorbt ([MASK],j) = f(t) for j ̸= [MASK], and no other off-diagonal is nonzero. Remember that qt(xt | x0) = αtδx

t=[MASK]. This means xt ∼ qt(· | x0) is either xt = x0 or xt = [MASK]. In this rank-1 absorbing case, the mean parametrization and the jump-states parametrization given by Eq. (9) coincide. Hence, θ only governs the transitions, while the scheduling is fixed by αt. In fact, one has the following lemma.

t=x0 + 1−αt δx

Lemma B.6. In masked diffusion, the mean parametrization and the jump-states parametrization introduced in Eq. (9) coincide. In particular, the (conditional) reverse rate is independent of θ and given by

 

−α˙t 1 − αt

, x = [MASK], 0, x ̸= [MASK].

rx

[x](t) = r[x](t) =

0



Proof. Recall that

, y ̸= [MASK], 1 − αt, y = [MASK],

αtδy=x

(23)

0

qt(y | x0) =

and that Definition B.4 defined the conditional reverse rate at state x as

rx

[x](t) =

0

qt(y | x0) qt(x | x0)

Qabsorbt (x,y)

.

y̸=x

Computation for x = [MASK]. Only the forward edges y → [MASK] (y ̸= [MASK]) are nonzero, i.e., Qt([MASK],y) = f(t). Therefore,

qt(y | x0) qt([MASK] | x0)

= −α˙t 1 − αt

αtδy=x

αt 1 − αt

rx

Qabsorbt ([MASK],y)

0

[MASK](t) =

=

f(t)

= f(t)

,

0

1 − αt

y̸=[MASK]

y̸=[MASK]

as f(t) = −α˙t/αt by definition of the mixing rate (cf. Section 3.2).

Computation for x ̸= [MASK]. There is no forward transition onto non-mask token x ̸= [MASK] (Qt(x,y) = 0 for all y ∈ V), hence rx

[x](t) = 0 for all i ̸= [MASK]. From rx

0

[x](t) to r[x](t). Marginalizing over x0 ∼ q0 yields qt([MASK]) =

0

q0(x0)qt([MASK] | x0) =

q0(x0)(1 − αt) = 1 − αt,

x0∈V

x0∈V

where we used Eq. (23). Hence, qt([MASK] | x0) = qt([MASK]) = 1 − αt, which implies that

###### qt(y | x0) = αt =

###### qt(y).

y̸=[MASK]

y̸=[MASK]

###### [x](t) since r[x](t) = y̸=x Qabsorbt (x,y)q

t(y)

Considering r[x](t), we can repeat the proof of rx

qt(x), and obtain

0

 

−α˙t 1 − αt

, x = [MASK], 0, x ̸= [MASK],

###### = rx

r[x](t) =

###### [x](t).

0



| |
|---|

We can now prove that our ELBO in Proposition 4.1 derived from the jump states parametrization coincides with the ELBO of masked diffusion models derived from the mean parametrization in (Ou et al., 2025; Sahoo et al., 2024; Shi et al., 2024).

Proposition B.7 (Recovering the MDM loss). For the case of masked diffusion, our ELBO in Proposition 4.1 coincides with the MDM loss:

1

Lpathx0 (θ) =

Ex

t=[MASK]}(−log µθ(xt,t))x

t∼qt(·|x0) 1{x

###### dt.

0

- 0

−α˙t

- 1 − αt

Proof. Lemma B.6 directly implies that Aθt = Rtθ in Eqs. (17) and (18), so the mean and jump parameterization collapses in the masked diffusion case (µθ = jθ). Recovering the MDM loss is pretty straightforward from the expression Lpathx0 (θ) =

1

- 0 Ex

t∼qt(·|x0) y̸=xt Qabsorbt (y,xt)(−log µθ(y,t)x

t

) dt. In fact,

- 1. if xt = x0, then the token is still clean, and we have y̸=x

t

Qabsorbt (y,xt)(−log µθ(y,t)x

t

) = Qabsorbt ([MASK],x0)(−log µθ([MASK],t)x

0

) = f(t)(−log µθ([MASK],t)x

0

);

- 2. if xt = [MASK], the token is masked. In this case, Qabsorbt (y,[MASK]) = 0 for all y ̸= [MASK]. Hence,

) = 0. Combining these cases and using P(xt = x0 | x0) = αt and P(xt = [MASK] | x0) = 1 − αt, we obtain Lpathx0 (θ) = 0 1 αtf(t)(−log µθ([MASK],t)x

y̸=xt Qabsorbt (y,xt)(−log µθ(y,t)x

t

)dt = 0 1 −α˙t(−log µθ([MASK],t)x

)dt, where we used α˙t = −αtf(t). This is the simplest form of the loss, but is usually written differently by leveraging the identity Ex

t

t

). Finally, we recover the MDM loss as

)] = (1 − αt)(−log µθ([MASK],t)x

t=[MASK]}(−log µθ(xt,t)x

t∼qt(·|x0)[1{x

0

0

1

Lpathx0 (θ) =

Ex

t=[MASK]}(−log µθ(xt,t))x

t∼qt(·|x0) 1{x

dt.

0

- 0

−α˙t

- 1 − αt

| |
|---|

###### B.4. Snapshot parametrization and Evidence Lower Bound (ELBO)

- Proof of Proposition 4.3. Consider the snapshot latent s = (xt,t) and the variational distribution qsnap(s | x0) = ρ(t)qt(xt | x0), where ρ is the uniform distribution over [0,1] for simplicity (i.e., ρ(t) = 1 for all t ∈ [0,1]). The

snapshot marignal is then qsnap(s) = Ex

0∼qdata[qt(xt | x0)] = qt(xt). We define the snapshot predictor pθ,0 snap(x0 | s) := µθ(xt,t)x

0∼qdata[qsnap(s | x0)] = Ex

from the output of the mean network. This defines a latent-variable model with joint probability

0

pθ,0 snap(x0,s) = pθ,0 snap(x0 | s)qsnap(s).

A standard ELBO derivation (see e.g., Lai et al. 2025, Thm. 2.1.1) consists of applying Jensen’s inequality as follows,

pθ,0 snap(x0,s) qsnap(s | x0)

pθ,0 snap(x0,s) qsnap(s | x0) ≥ Es∼qsnap(·|x0) log

log pθ,0 snap(x0) = log Es∼qsnap(·|x0)

Expanding the right hand-side using pθ,0 snap(x0,s) = pθ,0 snap(x0 | s)qsnap(s) yields

log pθ,0 snap(x0) ≥ −Lsnapx0 (θ) + Cxsnap

,

0

snap(s|x0) qsnap(s) is independent of θ and

:= −Es∼qsnap(·|x0) log q

where Cxsnap

0

.

Lsnapx0 (θ) := Es∼qsnap(·|x0) −log pθ,0 snap(x0 | s) =

1

Ex

t∼qt(·|x0) [−log µθ(xt,t)x

] dt.

0

0

| |
|---|

We now move on to the proof of Proposition 4.4. Consider a clean token x0 ∼ qdata. The path-wise latent ω = {N1,(zk,Tk)N

k=1} ∼ q[0,1](· | x0) is the full forward CTMC path on [0,1] and s = (xt,t) is the snapshot latent from this path. This means that we have sampled t ∼ ρ(t) (with ρ being the uniform distribution over [0,1]) independently of x0, and extracted the associated token xt from ω. This token is defined as follows : there exists some kt ∈ {1,...,N1} such that t ∈ (Tk

1

t−1 (where the edge case kt = 1 is covered by setting T0 = 0 and z0 = x0). More formally, if Ω is the path space and S the snapshot space, there exists a measurable map Ψ : [0,1] × Ω → S such that Ψ(t,ω) = s. For v ∈ {s,ω}, the joint law of both generative models can be written as q(x0,v) = qdata(x0)q(v | x0) and the corresponding posterior as q(x0 | v) = q(x0,v)/q(v). Let psnapθ (· | s) and ppathθ (· | ω) be any conditional predictors. We define the expected NLL gap as ∆NLLθ = E[−log pθ,0 snap(x0 | s)] − E[−log pθ,0 path(x0 | ω)] and the calibration error as Calvθ = E[KL(q(x0 | v)∥pθ(· | v))]. Let us start by introducing a simple lemma.

], and xt = zk

t−1,Tk

t

Lemma B.8 (Expected NLL decomposition). Let (X,V ) ∼ q(·,·) be any pair, and let pθ(· | v) be any conditional predictor for v ∈ {s,ω}. Then,

###### E(X,V )∼q(·,·) [−log pθ(X | V )] = H(X | V ) + EV [KL(q(· | V )∥pθ(· | V ))].

Proof. After conditioning on V = v, we obtain

###### E[−log pθ(X | V )] = EV EX∼q(·|V )[−log pθ(X | V )] .

For each fixed v ∈ V, the inner expectation is the cross-entropy between q(· | v) and pθ(· | v):

EX∼q(·|v)[−log pθ(X | v)] = H (q(· | v),pθ(· | v)) = H (q(· | v)) + KL(q(· | v)∥pθ(· | v)). Taking expectation over V gives

E[−log pθ(X | V )] = EV [H(q(· | V ))] + EV [KL(q(· | V )∥pθ(· | V ))]. However, EV [H(q(· | V ))] = H(X | V ) by definition of conditional entropy.

| |
|---|

- Proof of Proposition 4.4. Applying Lemma B.8 twice with V = s and V = ω, and subtracting the two yields the following decomposition:

+ Calsθ − Calωθ

∆NLLθ = H(x0 | s) − H(x0 | ω)

,

CG

IPG

where Calsθ := E[KL(q(· | s)∥pθ(· | s))]. Note that, by measurability of Ψ : [0,1]×Ω → S, the data processing inequality states that the mutual information can only decrease, i.e., I(x0;ω) ≥ I(x0;s), or equivalently H(x0 | s) ≥ H(x0 | ω), which implies that IPG ≥ 0.

Snapshot minimizer. The snapshot loss is given by

Lsnapx0 (θ) =

1

###### ] dt = E −log psnapθ (X0 | s) X0 = x0 . (24)

Ex

t∼qt(·|x0) [−log µθ(xt,t)x

0

0

Note that it would be weighted by ρ(t) inside the time integral if ρ was chosen differently from the uniform density. Averaging Eq. (24) over x0 ∼ qdata yields exactly the unconditional expected snapshot NLL:

0∼qdataEs∼qsnap(·|x0)[−log psnapθ (x0 | s)]. Following Lemma B.8, Ex

0∼qdata[Lsnapx0 (θ)] = Ex

Ex

0∼qdata[Lsnapx0 (θ)] = H(x0 | s) + Calsθ, where the term H(x0 | s) depends only on the data distribution and the forward process, and not on θ. Therefore

Calsθ.

E[Lsnapx0 (θ)] = arg min

arg min

θ

θ

Path-wise minimizer. Contrary to a snapshot ELBO, a path-wise ELBO (i.e., a diffusion ELBO; see Lai et al. 2025, Thm. 2.2.3) contains an additional diffusion term that depends on θ. This makes the model calibrates the local conditionals zk−1 | (zk,Tk) instead of x0 | ω. This can be clearly seen here with the Campbell form of the path-wise ELBO. Indeed, recall that Calωθ := Eω [KL(q(x0 | ω)∥pθ(· | ω))] measures calibration of a predictor of the initial token x0 given the full path ω. In contrast, the Campbell path-wise objective

Lpathx0 (θ) = Eω∼q

[0,1](·|x0)

N1

−log jθ(zk,Tk)z

k−1

k=1

is the negative log-likelihood of predicting the previous state zk−1 at each jump time from the event (zk,Tk). More precisely, using the other form given by Proposition 4.1,

Lpathx0 (θ) =

1

###### xt (t) CE Rx

###### t (·,xt),Rtθ(·,xt) dt,

t∼qt(·|x0) rx

Ex

0

0

0

hence, up to a θ-independent term,

Lpathx0 (θ) = const +

1

xt (t) KL Rx

t∼qt(·|x0) rx

t (·,xt)∥Rtθ(·,xt) dt.

Ex

0

0

0

Therefore, minimizing E[LpathX0 (θ)] calibrates the local reverse jump kernel Rtθ(·,x), not the posterior predictor pθ(x0 | ω). Since Calωθ concerns a different conditional distribution (namely x0 | ω), the minimizers do not coincide in general.

| |
|---|

#### C. Experimental details

###### C.1. Experimental setting

We train GDDS models on Text8 and OWT with bf16 precision (including the loss), global batch size 512, AdamW with learning rate 3.5 × 10−4, weight decay 10−2, (β1,β2) = (0.9,0.95), gradient clipping at 1.0, and EMA with decay 0.9999, for 1M optimizer steps, validating every 10k steps. All runs are executed on a single node with 4× NVIDIA H100 GPUs using DDP. More details are given in Table 5.

Table 5. GDDS Training configuration

Category Setting Sequence length 256 for Text8, 1024 for OWT Hidden size / heads 768 / 12 MLP ratio / dropout 4 / 0.1 Time conditioning AdaLN Layers 12 Optimizer AdamW (β1 = 0.9, β2 = 0.95, ϵ = 10−8), weight decay= 0.01 Learning rate 3.5 × 10−4 LR schedule Constant warmup for the 2500 first steps Precision bf16 (training and loss precision) EMA 0.9999 Batch size Global batch size 512 (eval global batch size 512) Gradient clipping 1.0 Max steps 1,000,000 Eval validation every 10,000 steps Noise log-linear Hardware 1 node, 4× NVIDIA H100 GPUs; 4 tasks per node (DDP)

Text8 dataset. We train models on Text8 for 1 million optimizer steps. While results in Table 1 are reported from (Shi et al., 2024), we additionally retrained three baselines: a standard autoregressive Transformer (AR), MDM (Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025) and UDLM (Schiff et al., 2025) using the same training recipe as for GDDS (optimizer, learning rate, schedule, batch size, precision, number of steps, and evaluation protocol). The only architectural difference is that the AR baseline uses no time conditioning and causal attention. Under this unified setup, we obtain 1.35, 1.58 and

- 1.67 BPC for AR, MDM and UDLM respectively (see Table 1). Notably, these values are worse than the corresponding numbers reported in Table 1 in (Shi et al., 2024), suggesting that cross-paper comparisons are sensitive to training details (e.g., optimization and implementation choices) and experimental conditions (e.g., hardware and system-level settings). Note that our AR result exactly matches the one found by (Hoogeboom et al., 2022). Since GDDS Absorb still outperforms these baselines by a large margin under our unified setup, we do not emphasize this mismatch further but mention it for completeness.

OpenWebText dataset. We follow the same unified protocol on OWT, training all models for 500k optimizer steps at sequence length 1024 (see Table 5), resulting in a total of 262B training tokens. We tokenize OpenWebText using the GPT-2 tokenizer, then concatenate documents and chunk the resulting stream into sequences of length 1024, inserting an <|endoftext|> token between consecutive documents. Since OpenWebText has no official validation split, we reserve the last 100k tokens for validation. Similarly than Text8, we retrain the matched-compute baselines (AR, MDM, and UDLM) with the same optimizer, learning-rate schedule, batch size, precision, EMA, and evaluation pipeline as our GDDS variants. Under this controlled setup, we observe again that our retrained baselines do not reproduce the perplexities reported in prior papers (see Table 2), highlighting the sensitivity of OWT results to implementation details and system-level factors (e.g.,

data processing, optimization hyperparameters, and hardware/software stacks). For this reason, we primarily rely on our retrained baselines for fair comparisons, while still reporting prior-work numbers when available. Importantly, GDDS yields substantially tighter likelihood bounds and stronger generation quality under the same protocol, and the main conclusions remain unchanged despite the mismatch with previously reported figures.

In Table 3, we evaluate zeroshot perplexity by taking the models trained on OpenWebText and evaluating likelihoods on the validation splits of 7 datasets: Penn Tree Bank (PTB; Marcus et al. 1993), Wikitext103 (Merity et al., 2017), One Billion Word Language Model Benchmark (LM1B; Chelba et al. 2013), Lambada (Paperno et al., 2016), AG News (Zhang et al., 2015), and scientific papers (Pubmed and Arxiv subsets; Cohan et al. 2018). Since the zeroshot datasets have different conventions for sequence segmentation, we wrap sequences to 1024 but do not add <|endoftext|> tokens in between sequences.

###### C.2. Semantic-Informed Kernel (SIK)

Recall that m is the vocabluary size. Let E ∈ Rm×d be a fixed embedding table, where ei ∈ Rd denote the embedding vector of the token number i ∈ {1,...,m}. Let {ρi}mi=1 be positive local bandwidths (e.g., ρi is the squared distance from ei to its k-th nearest neighbor). We fix (i) a decaying profile η : [0,∞) → R+, (ii) a distance on the embedding space demb : Rd × Rd → [0,∞), and (iii) a temperature schedule τ : [0,1] → (0,∞) that controls the amount of mixing (small τ(t) yields a near-identity kernel; large τ(t) yields a flatter kernel). Following self-tuning Diffusion Maps (Zelnik-Manor & Perona, 2004), we define the (symmetric) affinity as

Wt(x,y) := η

demb(e[x],e[y])2 τ(t)√ρ[x]ρ[y]

, for x ̸= y ∈ V, Wt(x,x) = 0. (25)

Hence, it is possible to define the Semantic-Informed Kernel by normalizing the affinity:

Wt(x,y) z̸=y Wt(z,y)

FtSIK(x,y) :=

, x ̸= y, FtSIK(y,y) = 0.

For the Gaussian metric, demb(e[x],e[y]) = ∥e[x] − e[y]∥22, while for the cosine metric, demb(e[x],e[y]) = 1 − ⟨e¯[x],e¯[y]⟩ after normalizing embeddings to unit norm. The corresponding CTMC rate matrix is QSIKt := f(t) FtSIK − Im , with f such that αt = exp(−f¯(t)) as in Proposition 3.3. The embeddings are extracted from GPT-2 (Radford et al., 2019), where d = 768 and m = 50257.

Gaussian and cosine SIKs. In the experiments, we consider two concrete choices for the affinity defining FtSIK. The first is the Gaussian SIK, based on squared Euclidean distance in embedding space. The second is a cosine SIK, obtained

by first normalizing embeddings to unit norm and then using the cosine distance demb(e[x],e[y]) = 1 − ⟨e¯[x],e¯[y]⟩. The Gaussian version favors tokens that are close in Euclidean geometry, while the cosine version instead depends only on the

angle between embeddings. The main text notation FtGauss corresponds precisely to this Gaussian choice of the SIK affinity. Two implementations: KNN and KEOPS. There are then two ways to implement FtSIK in practice. The first is a KNN implementation, where for each source token y we precompute a sparse neighbor set Nk(y) containing only its top-k nearest candidate neighbors in embedding space (in our benchmark, k = 64). In that case, we use the sparse approximation

exp −dembε√(ρe[[xx]]ρ,e[y[y]]) 1x∈N

1x̸=y m − 1

k(y) z∈Nk(y) exp −dembε√(ρe[[zz]]ρ,e[y[y]])

FtKNN(x,y) = (1 − λ(t))

+ λ(t)

so sampling takes place only on this sparse candidate set, which makes the method very fast in practice. The second is a KEOPS implementation, denoted FtKeOps, where we do not build a sparse graph and do not materialize the dense m × m kernel either; instead, we evaluate the entries of FtSIK on-the-fly with blockwise GPU reductions. Thus, KNN should be viewed as a sparse approximation of FtSIK, whereas KEOPS is the dense lazy implementation of the same normalized kernel.

Why this is KEOPS-friendly. For the dense implementation FtKeOps, we only specify the symbolic pairwise map

(ℓ,j)  −→ η

demb(eℓ,ej)2 τ(t)√ρℓρj

.

Thus, we let KEOPS generate fused GPU kernels that compute the needed sums on-the-fly. This avoids storing QSIKt explicitly and keeps forward noising practical.

Benchmark interpretation. Table 6 reports both similarity choices (Gaussian and cosine) and both implementations (KNN and KEOPS), in addition to the absorbing and uniform baselines. In our setup, the KNN versions are substantially faster because they sample only from a sparse neighbor list, while the KEOPS versions operate over the full vocabulary through lazy blockwise reductions. This induces an overhead because the dense KEOPS path still evaluates normalized kernel scores over large vocabulary blocks and then performs exact categorical sampling over those blocks. Our implementation uses a custom CUDA kernel for the blockwise sampler on top of the KEOPS score construction, which makes the dense path practical. We acknowledge that this CUDA kernel can still be optimized further to reduce the remaining overhead. However, even in its current form, around 150–160ms to noise an entire batch of 512 × 1024 = 524,288 tokens is small compared with the cost of a full forward-and-backward training step on batches of that size. The point of the KEOPS implementation is therefore not to beat the KNN approximation in raw latency, but to make the dense normalized kernel FtSIK feasible without ever materializing the full transition matrix.

Table 6. Noising-time benchmark. Mean wall-clock latency in milliseconds for sampling xt ∼ qt(· | x0) on batches of size 512 and sequence length 1024 (512 × 1024 = 524,288 positions). Reported values are mean ± standard deviation over 5 random seeds; for each seed, latency is averaged over 10 timed runs after 3 warmup runs. Results use absorbing, uniform, and SIK-based forward processes with KNN and KeOps implementations.

Method Mean ± Std (ms)

Absorbing 0.09 ± 0.01 Uniform 0.07 ± 0.00 SIK Gauss (KNN) 8.97 ± 0.26 SIK Gauss (KeOps) 152.49 ± 1.86 SIK Cosine (KNN) 8.94 ± 0.27 SIK Cosine (KeOps) 162.45 ± 2.19

All reported GDDS Gauss training results in this paper use the KNN implementation with k = 64 neighbors per token.

###### C.3. Metrics

We report both likelihood-based metrics (BPC / perplexity) and generation metrics computed from unconditional samples (Generative Perplexity, Sequence Entropy, Distinct-n). Throughout, let N denote the number of evaluated sequences (validation examples or generated samples, depending on the metric) and let x(j) = (x(j),1,...,x(j),n) denote the sequence j ∈ {1,...,n} of length n.

- C.3.1. NEGATIVE LOG-LIKELIHOOD (NLL), BPC, AND PPL

For likelihood-based evaluation, consider a model that defines a probability pθ(x) over sequences x = (x1,...,xn). The (per-token) negative log-likelihood on N sequences {x(i)}Ni=1 is

1 N n

NLL := −

N

log pθ x(i) .

i=1

When the model admits a tractable factorization (e.g. autoregressive),

n

log pθ(xj | x<j),

log pθ(x) =

j=1

which recovers the standard “next-token prediction loss” expression. In general, log pθ(x) may be intractable; in that case we report a variational upper bound on NLL (equivalently a lower bound on log pθ(x)), such as the ELBO induced by the training objective.

Bits per character (BPC). On character-level datasets (Text8), we report BPC, i.e. the NLL expressed in base 2:

1 N n

BPC := −

N

log2 pθ x(i) =

i=1

NLL log 2

.

Perplexity (PPL). On tokenized datasets (OWT), we report perplexity, defined as the exponential of the NLL:

1 N n

PPL := exp(NLL) = exp −

N

log pθ x(i) .

i=1

Lower NLL (equivalently lower BPC/PPL) indicates better likelihood-based performance.

- C.3.2. GENERATIVE PERPLEXITY

Likelihood metrics do not directly assess sample quality when generation uses an approximate sampler. Following prior work (Lou et al., 2024; Sahoo et al., 2024), we therefore report Generative Perplexity (Gen-PPL) under a fixed external evaluator (GPT2-large in our experiments). Given unconditional samples {x(i)}Ni=1 produced by a model, we compute

Gen-PPL = exp

 −

1 N n

N

i=1

n

j=1

log peval x(i),j | x(i),<j

 ,

where peval denotes the evaluator next-token distribution (GPT2-large). Lower Gen-PPL indicates that generated samples are more predictable under the reference model, which empirically correlates with higher perceived fluency.

- C.3.3. SEQUENCE ENTROPY

A low Gen-PPL can be achieved by overly repetitive or mode-collapsed generations, as noticed by (Zheng et al., 2025). To quantify diversity, we compute the average unigram entropy of each generated sequence and average across samples. Let c(v,x(i)) be the count of token v ∈ V in sample x(i). Define the empirical unigram distribution within a sample as pˆ(i)(v) = c(v,x(i))/n. We report

Huni = −

1 N

N

i=1 v∈V

pˆ(i)(v) log pˆ(i)(v).

Higher values indicate that a sample uses a broader set of tokens more evenly (i.e., less repetition). In the paper, we plot Gen-PPL against this entropy statistic to visualize the quality–diversity tradeoff, following (Zheng et al., 2025).

- C.3.4. DISTINCT-n

We additionally report Distinct-n statistics, which measure lexical diversity via the proportion of unique n-grams. To avoid overloading notation, we denote by Distinct-k the metric computed with k-grams. Let kgramsk(x(i)) denote the multiset of length-k contiguous k-grams in sample x(i). We compute corpus-level Distinct-k over the full set of samples:

N i=1 uniq kgramsk(x(i))

Distinct-k =

, k ∈ {1,2,3},

N i=1 kgramsk(x(i))

where uniq(·) removes duplicates within a sequence. Higher Distinct-k indicates fewer repeated k-grams and typically correlates with more diverse generations.

###### C.4. Qualitative samples on Text8

We report in Tables 7 and 8 two unconditional samples for the GDDS Absorb and Uniform models trained on the Text8 dataset.

- Table 7. Two unconditional samples for the GDDS Absorb model trained on Text8.

|[BOS]sessions studies professional present topics international literature university of global history of international shock artist as well as its global no actibilative clocks to ruin the strip individual martials in the image of information both side east[EOS]|
|---|
|[BOS]g strewn nearly confrentative and the bible tag strews which were very officialy dilgress in the usa japanepic first considered only parathislatic influences arabic christianity have played in their language slightly best known is the japanene term in ma[EOS]|

- Table 8. Two unconditional samples for the GDDS Uniform model trained on Text8.

|[BOS]heir ownest possession is the case despite cyclic subgraphs resulting in its first higher community area or less expensive [EOS]major national companies again below the company is a supporter of city trains or a record historically system is cut the large bu[EOS]|
|---|
|[BOS]ree the oxford university vol two six seven two two pp two seven three two x are not human definitions of oxford since others are seaset fara la conventions university of chicago based on bissaudi and known as the karlowe large content of the world elbia[EOS]|

###### C.5. Qualitative samples on OpenWebText

For readability, we apply a lightweight post-processing to unconditional samples before rendering them in LATEX. Concretely, we replace paragraph markers \n\n by a line breaks, decode common Unicode escape sequences into standard typography (e.g., \u2019  → ’, \u201c  → “, \u201d  → ”,\u2264  → ≤), and render special tokens such as <|endoftext|> in \texttt{}. We report in Tables 9 to 11 unconditional samples for the GDDS Absorb, Uniform, and Gauss models trained on the OpenWebText dataset.

###### Table 9. Unconditional sample for the GDDS Absorb model trained on OpenWebText.

|<|endoftext|> to conform to routine training procedures for the handling of a crime or any mass deaths committed by police when it investigates crimes. Opponents of the entertainer’s departure say the department will offer guarantees in coverage for a crisis that in 2002 and 2009 frees law enforcement. The protesters, say police are taught to commit crime continuously with tear gas or bullets. The chaos comes amid sweeping changes in policing. Illinois law enforcement took last month’s a surprise given the growing appetite in some places for police from function as order in the state.<br><br>“But more generally, it’s a safe haven,” said Lake-based Howard Hexano. Theneighbor Police Association of the United States said it replaced the training road used by law enforcement for initially international training exercises, rather opting to resemble standard “economic training” methods. Sarley opened questions about the leadership of Kenney Village, the same police department, that has the traditional candidate for the license for two men accused of shooting someone at the Colorado New Year’s Eve pageant annual competition, but it pays paid for the monthly pay of the victims. The agency won’t face any disciplinary action against officers in the scandal, and it has addressed the community about the past. Earley on Feb. 28 will officially investigate the incident, concluding that the men accused were not teaching acting well, or causing harm to others. The Tribune’s Matt Steward declined to comment. The department’s chief officer, J. Durkin, who was a local pastor following his Feb. 24 firing, did not respond to requests for earring for comment on the incidents. All grievances in the investigation came and there’s need for further technical inquiry. Even local law enforcement companies, including Illinois Public Affairs and other state law enforcement officials continue to have witness cooperation, such as W-Ferguson and Louisiana C-M., in the works.<br><br>“We actively consult — municipal agencies and other social health and architectural agencies to deliver information services and information regarding current events,” Charlie Paley said in an announcement.<br><br>“C-M., Under-Chief Shawn Leslie and Nick Stout, Executive Director are working closely with the person conducting the investigation for evaluation and possible update on future events. For more about what happened in Kidney Village, read on here.” The real highlights the six basic changes and guidelines recommended throughout the town’s history. For additional units, if completed, the Westlake Union Marriott Hotel will be located in the 650 block. Comprehensive plans include a spa spa. It is 45-foot by 40- feet wide and have a barbed roof. Regional — urban — will be 10,000 square feet and include RGB lighting, natural gas development, natural fire and swimming pool and golf courses. As the area grows diverse enough, police will become concerned about the potential threat to its security. Some cops want hotel gives new applicants for a recreational site to meet location hotel is home to the training headquarters of the U.S. Marines, home of about 700 horse and elderly confrontations with law enforcement officers. It’s normal for the U.S. government to increase its activist role — asserting that American police do slow to inflict damage on Americans during First World War. The U.S. Congress loosened constitutional restraints when it comes to police violations trial except that makes it a fine, similar to what is guilty under the federal Criminal Code. Also floating around is potential limited impacts — and public safety and safety have increased as a deterrent since when a school student was shot while running a road trip. Why is that? But tourists won’t allow it in. That common sense will prove little change for the town, either. And the impact will no longer homes. But any feeling of diversity will be gaining almost all in the surrounding institution, many of its core buildings and colleges. The Cenna-The Performing Arts exhibition is “Hop Art Thursday” by Sen. David D Ducey (D-D.).<br><br>“To make everyday Americans realize that I think the museum stands out throughout much of American history,” he says when a 65-year-old drops an invitation from Harvard’s Dr. Richard B. Hoffman to enter the White House in 2006. And always will have been the center of the museum’s popularity. The visitors that the design be more limited in future exhibits, said Fucakis, are themes from past films, such as the arenas and cop offices, “brain longening” and “Hey, Captain. I have the power of a cat.” Fucakis, also a pollster, announced Thursday that the megamstitute is opening a doghouse and would be introducing video of historical footage<|endoftext|>|
|---|

###### Table 10. Unconditional sample for the GDDS Uniform model trained on OpenWebText.

|<|endoftext|> the United Kingdom), in hopes that the U. president and his global leader will lodge a demand or promise to use its stated goal to sustain support to combat ISIL. (The United Arab Emirates responded for the first time.) On Saudi Arabia and its United States side on the matter, less than ever will affect the U.S. to longer series,” co-Nubal Hender explained. This kind of bullsrina zone is in the shape that Syria’s non-American interests may be in the West 2007. U.S recent political and economic quellings of the Egyptian mass-democratic regime, destabilizing Shiite and anti-Muslim militias will only escalate tensions between the core allies of Syria, Saudi Arabia, and Syria, independent of the central U.S., while maintaining a fair-minded, external foe whose leaders seek to assist them, as head of White House argued. Even if American and Pres. Kerry efforts succeed slack push, they are unlikely to continue to broker lobbying strategies. The best way to look at that is the case across the world that rather than accept the negative effects of Nazi method are much to completely out of reach for those who have demons. In fact, patients who are in progressanic<br><br>– for their malign position limiting behavior may turn into a - or recommended - - for becausefama. Elimuting unsenate drugs as long as Western medical prohibition on marketing and other ministries have harm’s effect, with a perils annual mortality rate, despite often everywhere no-one makes patients plan for a continued suicide away. Indeed, suicide rates among the American population are abnormally low, and coronary myocardic degeneration remains at a preexisting rate the single highest.41. A third study conducted by the IBM School of Medicine also archives patient interpretation data for physicians and health researchers and encouraged the study. The study in July 2015 was published in Dec. 4 of the Proceedings of the National Academy of Sciences, which was produced this year. The authors responded with an order of published papers published in each major journal for the study of hypertension. The journal’s published papers were published online from Thursday, February 16 to Jan. 6, 2015 by Mawall and Sarah Leonard.<br><br>“The 1VHP trials were the most selective Tract-based randomized trial to be administered under the ausp against the fact where they reported consistent that they were moderately popular in the internal health industry. But the motive of taking support for high-risk cardiovascular care tends to lead at even higher risk,” write Begojin Krishnan, who is the senior author of the New England Journal of Medicine. 41. Biopathological¨ benefits of shoulder plans was again useful in triggering several potentially dangerous activity that could trigger strokes.He¨ found that these outcomes cannot be altered because coronary artery consumption is harmful to the researchers, and his team among its patients to criticize the customary practice of nasation from shorter milliseconds of knee propelled toparse pain and flow from a higher-value knee plateau.<br><br>In a study that took 36 weeks, Pro Medel’s 37 weeks of placebo caused s¨udden activities¨--no orgasm for years of experience–and survive, and decreases were detected at ≤191 levels of cancer-seriousctal core actal. Backstrokeinduced metabolic syndrome is a major shift in vascular function to dependence, which may cause time distances among Traitor intensive coronary artery consumption more associated with chronic stroke-staffing,¨notes Schuning, the row center in the study. This¨ positively affected the vascular system almost all the time, and it represents his lower, older, rested region.¨42 Despite by scientists its efforts to reduce coronary artery injury, concussion therapy impacted fatal deaths among documented patients with Naondigran, who suffered increased cardiovascular toxicity and significantly increased strokes among the patients investigated the November 1 to 12 June 2015. The efficacy of placebo The scope of the study for shoulder disease was completely triletified when the study showed moderate improvements in coronary artery activity. The main test of effectiveness was the use of alcohol-free root oils for impact and the intervention. free-toverorption properties also contained wide variety of complimentary placebo techniques. Only a run on the annual National Study of Clinical Study, significant clinical correlation was achieved by the study 1 trial that did not show the effects of treatment from their respective programmes. When fails all studies measuring menstrual side effects, another systematic review showed that one trial showed a linear scheme of the retrospective performance across knowledge units 1VHP. Modeling was predicted to be followed for each stroke due to a health insurance plan featuring RCT and Morse sequences over the day of visits. Thus the commensatory between risk of death and risk of strokes was reached, soon agreement with in the 321 1VHP trials involving practitioner using mathematical reasoning and the distribution provided by himself.44 Thus, septoristic treatments show a significant sum of the energy received in one dose of intervention; only a fraction of the energy is collected from the release of<|endoftext|>|
|---|

###### Table 11. Unconditional sample for the GDDS Gauss model trained on OpenWebText.

|<|endoftext|> stories went in to TV in 2014 on Abboun on Zahora and became a busy town, rapidly knit and destabilized to men of almost 800 soldiers. Most frequently, Abcindi. the Aba Biba, boss of his lawyers, in Nahida, became a woman who enjoys his wife Helen‘s penchant for writing and writing to the LA Times. Abbindi, the wrinkled young woman, is sent fishing for the beach for one part in the incredibly horrific story find on eyewitness accounts and videos involving an immigrant family and local high school friends in AinsZi, a coastal enclave of Felton, N.C., Liberia and Dominican Republic. The Cairo story of The Los Angeles Times magazine tells the Egyptian State government to Abi ’Sadei was a huge hoax. The Prison cells of the LahaAidan of Cairo National University, a local university for the United States, built military antennas for the Egyptian army to cover the Ghouta bombing in Idlib. The Egyptian University builds antiIslamic propaganda, challenged the group of soldier to get their flack on a private plane, leading to a photographers<br><br>– Abno Ueh – having portrayed in photographs of Muslims. One explanation of what had caused the mayhem is for what’s worse happened on the covers and does not get to interest. That, they says, plenty of telefilmed spy community, of fear to beding in to the many stories and photos in the Abnot Ueh story. A little apart, however, the idea of the investigation was so near to this sites it became obvious that some that those individuals were presented for speculative speculation on that search result - and it is the way that has remained hidden for almost a year. It was not just a disturbing problem in trying to relate to personal data. Artificial intelligence is no confident in ability to pull a plot for it was as difficult to it was on a list the opponent has been to it. Research is to prone as it say to when we have requests in Google searches, they inevitably do asked to American espionage who care. Now they were looking for evidence pointing to action - to try to avoid that, and simply to try to get the first, or simply to get to the point for conjecture. Weaknesses could turn to brilliance on the very most dramatic weeks power and intelligence are not mere real technologies, they are a incurable danger they have face maybe to a decade or we didn’t want to fully understand. The debate there seems interesting.Is it to several: Perhaps to be certain, the answer may sound much more than the midnight hour of 9/11. And in November 2015, Donald Trump’s first response on an outreach to try to infiltrate Twitter on Muslims was to open a hotline for U.S. for control of the World Trade Centre. And it is not a reference to what was happening in Saudi Arabia, as it is ISIS providing explicit terrorism was meant for attack that as detonate the target. It took much more room to laid the groundwork for it for many people looking to that - for anticipatory details that is not easier to find. there is an increased degree of concern when the group specifically targets terror-related activists and the National security adviser, Donald Trump tells audience gather in Saudi Arabia. to provide his account to Donald Trump as the story for put ubaida on Europe’s emerging refugee crises. – Kimberler Michael A- Doh6 Follow us on Twitter Google Whatsapp Tumblr Tumblr Pinterest Pinterest LinkedIn Google sharePARIS, New Jersey A building is deserted. It didn’t get cashing in for several years or it bombed to several.The biggest problems for that necessity of the huge building were dizors, patio cushions and wooden stitched.The journalist could hardly make up the case.A columnist in Ashraf Darrow, a reporter who is helped Donald Trump keep his watch as he is keen to become the non-American.This month, on Sunday a Faredan revealed the plan to embark in Montreal- doomed flights. “My plane had come up and I was actually only say to hell I’m have.” he told Time.“So I couldn’t do that It wasn’t anything for the metal gros and a communication tools I did,. . . It was sort of like an crazy . . . , ’cause, it got me on how I had to, [I Rediscovered that].I just wanted to prove I can have what to have on,” . . . French Transport Minister UEP S, Mariese Louse (C) and how Canada Led The War on fiscal Policy to a Future’S international Airport Centre. available Online at: 893/2313,349, online 915, (13,A), Part 2 The Economic Analysis Of Inflorating Railway Product Taxes (2013). French Transport Minister UEP S.Louse (C), Center to Transport Policy Group <|endoftext|>|
|---|

#### D. Sampling procedures

Setup and notation. We consider a discrete-time grid 1 = tK > tK−1 > ··· > t0 = 0 with decoding budget K ∈ {32,64,128,256,512,1024}. The forward process is conditionally independent across positions (given x0), so all formulas below apply token-wise and are executed in parallel for a length-n sequence. For any 0 ≤ u ≤ 1, we denote by qu(· | x0) the forward marginal at time u starting from token x0 ∈ V, and by qt|s(· | xs) the forward transition from time s to t. At sampling time, the model provides a distribution over clean tokens, µθ(· | xt,t) ∈ ∆m (our “mean network”).

- D.1. Ancestral sampling Model-implied forward marginal. Given µθ(· | xt,t), define the model-implied forward marginal at any t ∈ [0,1] by

qt(xt | µθ) =

x′∈V

µθ(x′ | xt,t) qt(xt | x′). (26)

Ancestral reverse kernel. The exact Bayes reverse conditional is

qt|s(xt | xs)qs(xs | x0) qt(xt | x0)

q(xs | xt,x0) =

.

All our samplers use the standard plug-in approximation x0 ≈ µθ(· | xt,t), yielding the ancestral reverse kernel

- qs(xs | µθ)

- qt(xt | µθ)

pancθ (xs | xt) = q(xs | xt,x0 = µθ) = qt|s(xt | xs)

,

where qs(· | µθ) and qt(xt | µθ) are defined in Eq. (26).

Time-discretization. Given a discretization 0 = t0 < t1 < ··· < tK = 1 (decoding budget K), we use the plug-in Bayes/ancestral kernel

pancθ (xt

k−1 | xt

k−1 | xt

) = q(xt

,x0 = µθ) = qt

k|tk−1(xt

k

k

where the “mixture forward marginal” induced by the predictor µθ is

k | xt

k−1

k−1 | µθ) qt

qt

(xt

, (27)

k−1

)

k | µθ)

(xt

k

qu(xu | µθ) :=

x′∈[m]

,tk)x′ qu(xu | x′), for u ∈ {tk,tk−1}.

µθ(xt

k

Algorithm 4 summarizes the generic ancestral sampler. Below we give closed forms (Uniform / Absorb, already known in the literature (Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025; Schiff et al., 2025)) and the operator form (SIK).

- Algorithm 4 Generic ancestral sampler (token-wise, parallel over positions)

- 1: Input: Time grid 1 = tK > ··· > t0 = 0; model µθ(· | x,t).
- 2: Sample xt

K ∼ qt

K

(·).

- 3: for k = K,K − 1,...,1 do
- 4: t ← tk, s ← tk−1
- 5: µ ← µθ(· | xt,t)
- 6: Compute qs(· | µ) and qt(xt | µ) via Eq. (26)
- 7: Compute pancθ (· | xt) via Eq. (27)
- 8: Sample xs ∼ pancθ (· | xt)
- 9: end for
- 10: return xt

0

- D.2. Instantiations of ancestral sampling

- Case 1: Uniform diffusion. The forward marginal is

qt(y | x0) = αt δy=x

0

+

1 − αt m

.

Moreover, for 0 ≤ s < t ≤ 1, letting αt|s := αt/αs, we have

qt|s(y | x) = αt|s δy=x +

1 − αt|s m

. Plugging these into Eq. (27) yields

pancθ (xs | xt) = αt|sδx

s=xt +

1 − αt|s m

- x′ µθ(xt,t)x′ qs(xs | x′)

- x′ µθ(xt,t)x′ qt(xt | x′)

. (28)

- Case 2: Absorbing / masked diffusion. The forward marginal is

qt(y | x0) = αt δy=x

0

+ (1 − αt)δy=[MASK]. For 0 ≤ s < t ≤ 1 with αt|s := αt/αs, the conditional kernel is

qt|s(y | x) = αt|s δy=x + (1 − αt|s)δy=[MASK]. Thus,

pancθ (xs | xt) = αt|sδx

s=xt + (1 − αt|s)δx

s=[MASK]

- x′ µθ(xt,t)x′ qs(xs | x′)

- x′ µθ(xt,t)x′ qt(xt | x′)

. (29)

- Case 3: Semantic-Informed Kernel (SIK). Let Ft(· | x) be a (column-stochastic) semantic jump kernel with Ft(x | x) = 0, and let the (time-inhomogeneous) generator be

Qt = f(t)(Ft − I). The exact transition operator is the time-ordered exponential

Kt,s := T exp

t

Qτ dτ , qt|s(· | xs) = (Kt,sδx

)(·), qt(· | x0) = (Kt,0δx

)(·).

s

0

s

Therefore the plug-in ancestral kernel Eq. (27) can be written as

µθ(xt,t)x′ (Ks,0δx′)(xs) x′ µθ(xt,t)x′ (Kt,0δx′)(xt)

)(xt) x′

pancθ (xs | xt) = (Kt,sδx

. (30)

s

Practical computation and difficulty. For SIK, each factor in Eq. (30) is significantly harder to access than in the uniform and absorbing cases. The bridge term (Kt,sδx

)(xt) requires a short-time forward transition between two arbitrary tokens, while the numerator and denominator require evaluating the mixture marginals qs(· | µθ) and qt(xt | µθ), i.e. applying the forward operators Ks,0 and Kt,0 to many latent candidates weighted by µθ. In our implementation, these quantities are approximated through uniformization-based matrix-vector products with caching across timesteps and blocks. This makes ancestral decoding feasible, but also substantially more delicate than in the closed-form uniform and absorbing settings. Empirically, this is reflected by the fact that GDDS-SIK models can achieve very strong validation losses, while the corresponding ancestral samplers remain difficult to calibrate and, in our current experiments, do not yet outperform the GDDS-uniform and GDDS-absorb samplers reported in the main text. We therefore interpret the present SIK results as evidence that the model class is strong, but that sampling for semantic continuous-time kernels still requires additional work; the appendix ablations document this point.

s

The ablation in Table 12 clarifies the current GDDS-SIK behavior. On the positive side, the sampler does reach the desired entropy range: from K = 32 onward, the generated entropy is already close to that of natural OWT text, and by

- Table 12. GDDS-SIK sampling ablation on OpenWebText. We report decoding budget K, average sequence entropy, and Gen-PPL for unconditional samples. Natural OWT text typically lies around entropy 5.60–5.70 (Zheng et al., 2025).

K Entropy (↑) Gen-PPL (↓)

8 5.34 402.25 16 5.45 230.64 32 5.59 207.48 64 5.59 176.68

128 5.66 189.06 256 5.67 254.15

K ∈ {128,256} it lands squarely in the target regime. Qualitatively, the resulting generations also look reasonable; see the generated samples in Table 11. The difficulty is instead on the quality side. Unlike uniform and absorbing diffusion, where the ancestral kernel admits a closed form and each reverse step can be sampled directly, SIK requires approximating the time-ordered exponential through uniformization-based cached matrix-vector products. This has two drawbacks. First, it is slower, because each reverse step requires truncating a Poisson series and performing several matvecs, whereas ancestral sampling for uniform and absorbing diffusion is direct. Second, the approximation error appears to accumulate along the trajectory: Gen-PPL improves up to K = 64, but then worsens as K increases further. This suggests that once the discretization error is sufficiently small, the remaining operator-approximation error dominates and compounds across steps. In other words, we can already sample from the trained GDDS-SIK denoiser, but faithfully turning that denoiser into a strong ancestral sampler remains challenging.

Future work: avoiding ancestral sampling for SIK and more. While ancestral sampling via Eq. (27) is conceptually simple, its SIK instantiation remains computationally expensive. Indeed, even with sparsity and caching, evaluating the plug-in ratios requires repeatedly approximating forward operators (e.g., Kt,0 and Ks,0) and/or bridge terms, for which uniformization-based matvecs dominate the runtime. A natural direction is therefore to develop adaptive sampling procedures that better exploit the strength of the trained denoiser without committing to full ancestral updates at every step. This is consistent with recent evidence that adaptive or confidence-based schedules can outperform standard ancestral decoding in diffusion language models (von R¨utte et al., 2025b; Nie et al., 2025; Kim et al., 2025). For GDDS-SIK in particular, such methods are especially attractive because they need not rely on repeated explicit approximations of Kt,s or of closed-form forward marginals. Ultimately, we aim at samplers (and corresponding training objectives, as snapshot-ELBOs already encourage) in which GDDS with semantic kernels is truly blind to the exact forward transition operators, thereby removing the need to approximate Kt in closed form in both training and decoding.

#### E. Architectural details for Campbell

###### E.1. Campbell objective at the sequence-level : an any-order autoregressive objective

Recall that x0 = x10 ...xn0 is clean and xt = x1t ...xnt is the noised sequence at time t. At a jump time τ = Tkℓ of coordinate ℓ, we denote by xτ− (resp. xτ) the sequence immediately before (resp. after) the jump. By construction, only coordinate ℓ changes at time τ, so xτ− and xτ coincide at all positions j ̸= ℓ, and the observed pair is zkℓ−1 = xℓτ− and zkℓ = xℓτ. Let Rθτ(x′,x) denote the reverse kernel on sequences of size n at time τ, interpreted as the conditional probability of the predecessor sequence x′ given the current sequence x. Since consecutive states along ω differ in exactly one coordinate, Rθτ(xτ−,xτ) only concerns the predecessor token at the updated position ℓ given the post-jump state xτ, and for τ = Tkℓ we have Rθτ(xτ−,xτ) = Rτθ(zkℓ−1,zkℓ). Consider the conditional3

n

pθ0(x0 | ω) := C(x0;ω)

ℓ=1

Nℓ

Rθτ xτ−,xτ ,

k=1

3We write τ for a generic realized jump time; in the products/sums below, τ always refers to τ = Tkℓ for some (ℓ, k).

where C(x0;ω) collects all terms independent of θ. Then −log pθ0(x0 | ω) equals the event-wise cross-entropy sum up to an additive constant. Indeed, Jensen’s inequality yields the sequence-level ELBO

log pθ0(x0) ≥ Eω∼q

[0,1](·|x0) log pθ0(x0 | ω)

###### ,

−L(θ)+C(x0)

where C(x0) = Eω[log C(x0;ω)] = nℓ=1 C(xℓ0) is independent of θ, expanding the token-level ELBO of Proposition 4.1 beyond the case n = 1. Here, the θ-dependent term is exactly the Campbell objective of Proposition 4.2,

 

 . (31)

Nℓ

n

−log Rθτ(xτ−,xτ)

L(θ)=Eω∼q

[0,1](·|x0)

ℓ=1

k=1

This mirrors an any-order autoregressive training objective, such as XLNet (Yang et al., 2019). There, the factorization is induced by a random permutation of clean tokens, whereas in Eq. (31) it is induced by the time-ordered Poisson jump events along the diffusion path. Hence, the conditioning contexts xτ are noised, making it closer to the MPNet objective (Song et al., 2020). However, our objective still remains fundamentally different; first, this path-wise formulation also applies beyond masked diffusion to general forward corruption processes. Second, it trains over all jumps encountered along the forward path, yielding nℓ=1 Nℓ token-level supervision terms per clean sequence (rather than one).

###### E.2. Two-stream architecture for the Campbell estimator

A naive neural implementation (either XLNet/MPNet or a bidirectional-attention decoder only transformer) would require n ℓ=1 Nℓ NFEs per clean sequence to evaluate Eq. (31). This may be enormous for any reasonable sequence length n and hinders scalable training. This motivates a two-stream attention architecture that treats the whole path in E[Nℓ] NFEs in average (now independent of the sequence length n), i.e. in roughly one pass. Note that for masked diffusion, Nℓ = 1 for any ℓ, so it is exactly one pass in that case.

Neural parametrization (from pθ0 to jθ). Recall from Eq. (31) that the Campbell objective maximizes the path-conditioned product likelihood pθ0(x0 | ω), whose θ-dependent part factorizes over the observed jumps of the forward path ω. Each factor is a sequence-level reverse probability Rθτ(xτ−,xτ) at some realized jump time τ. Since only one coordinate ℓ changes at time τ, Rθτ(xτ−,xτ) is the probability assigned to the pre-jump token at position ℓ given the post-jump state xτ. We parameterize these factors with a single neural network jθ : Vn × Rn≥0 → ∆nm that outputs, for each context sequences xτ = (xτ)1≤ℓ≤n ∈ Vn×n and jump times τ = (τ)1≤ℓ≤n ∈ Rn≥0, a categorical distribution over predecessor tokens at positions 1 ≤ ℓ ≤ n:

jℓθ(xτ,τ) = softmax(lθℓ(xτ,τ)) ∈ ∆m, lθℓ(xτ,τ) ∈ Rm.

For an observed jump of coordinate ℓ at time τ = Tkℓ along ω, the predecessor token is zkℓ−1 = xℓτ−, so we set jℓθ(xτ,τ)zℓ

= Rθτ(xτ−,xτ). Hence, each Campbell term is exactly the cross-entropy loss contribution −log jℓθ(xτ,τ)zℓ

.

k−1

k−1

Minimizing L(θ) effectively corresponds to the maximum likelihood on the path-wise model pθ0(x0 | ω), i.e., to maximize pθ0(x0) (up to θ-independent factors).

Two-stream architecture. We introduce a two-stream architecture based on the XL-Net/MPNet idea: an encoder-decoder transformer (Vaswani et al., 2017) that combines ideas from XLNet (Yang et al., 2019) and DiT (Peebles & Xie, 2023).

The main challenge is to predict the pre-jump token xℓτ−

at an event time τℓ using only the clean set {j : τj > τℓ}, which induces a sample-dependent (non-causal) factorization that cannot be enforced by a fixed left-to-right mask. We therefore introduce the rank rℓ := rank(τℓ), defined as the position of τℓ in the sorted (decreasing) list of masking times (with ties broken so that r is a permutation), so that τj > τℓ ⇐⇒ rj < rℓ. Inspired by XLNet, our two-stream architecture enforces this permutation-style factorization with two streams and rank-based attention masks: an encoder (content stream) builds contextual representations, while a decoder (query stream) predicts with masked queries and is restricted to attend only to keys j such that rj < rℓ, preventing leakage from xℓ0 or yet-unrevealed tokens. The decoder is conditioned on continuous time via Adaptive LayerNorm (ADALN), using a local embedding of τℓ to modulate its blocks and output, while the encoder remains strictly time-agnostic. This yields the one-pass training loop of Algorithm 5.

ℓ

- Algorithm 5 Training with Campbell estimator

- 1: Input: distribution qdata, network jθ, batch size B.
- 2: Sample a batch of sequences x(1)0 ,...,x(0B) ∼ qdata.
- 3: For each sequence x(0b) and token position ℓ, run Algorithm 3 with t = 1 to sample Nℓ,(b) jumps and pairs

{(Tkℓ,(b),zkℓ,(b))}N

ℓ,(b)

k=1 .

- 4: Compute the Campbell loss estimate Eq. (31)

B

1 B

L(θ) =

b=1

Nℓ,(b)

n

−log jℓθ(x(b)

, Tkℓ,(b))zℓ,(b)

,

Tkℓ,(b)

k−1

ℓ=1

k=1

where x(tb) denotes the noised sequence at time t.

###### E.3. Empirical results

We train the two-stream architecture on both Text8 and OWT, with the same experimental setup as previously. We used an absorbing forward noising.

- Table 13. Validation perplexity of the two-stream architecture. We train the two-stream architecture on Text8 and OWT under the same experimental setup as in Section C, and report the BPC and validation perplexity.

Text8 BPC OWT PPL

≤1.75 ≤ 76.07

Figure 6. Training loss stability on Text8. Train loss curves for AR, MDM, GDDS Absorb, and Campbell two-stream training, using the same setup as in Section C. While snapshot-based training exhibit noticeably higher short-term fluctuations, Campbell training yields a markedly smoother optimization trajectory, comparable to AR.

We plot the training loss of the two-stream model “Campbell” in Fig. 6, as well as the training losses of MDM, GDDS Absorb and AR. Quantitatively, the standard deviation (std) of the training loss over the last 300k steps is std ≈ 3.08×10−2 (AR), 1.63×10−1 (MDM), 1.65×10−1 (GDDS Absorb), and 2.92×10−2 (Campbell). We remark that Campbell training yields a more stable optimization curve than snapshot-based objectives (MDM, GDDS, etc.), even though it conditions on full path information ω (as do AR models) rather than a single snapshot. Indeed, the Campbell estimator sums per-jump cross-entropies along the uniformization path(yielding at least n supervision terms per clean sequence, versus a single term for snapshot-based training) which reduces the variance of the learning objective across iterations. However, this stability does not translate into better likelihood performance under our architectural constraints (see Table 13). Indeed, compared to the results of Tables 1 and 2, we found that the two-stream architecture clearly underperforms.

