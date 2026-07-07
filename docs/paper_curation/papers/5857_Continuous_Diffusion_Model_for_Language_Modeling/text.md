# arXiv:2502.11564v2[cs.LG]23Oct2025

## Continuous Diffusion Model for Language Modeling

Jaehyeong Jo1, Sung Ju Hwang1,2 KAIST1, DeepAuto.ai2 { harryjo97, sjhwang82 }@kaist.ac.kr

### Abstract

Diffusion models have emerged as a promising alternative to autoregressive models in modeling discrete categorical data. However, diffusion models that directly work on discrete data space fail to fully exploit the power of iterative refinement, as the signals are lost during transitions between discrete states. Existing continuous diffusion models for discrete data underperform compared to discrete methods, and the lack of a clear connection between the two approaches hinders the development of effective diffusion models for discrete data. In this work, we propose a continuous diffusion model for language modeling that incorporates the geometry of the underlying categorical distribution. We establish a connection between the discrete diffusion and continuous flow on the statistical manifold, and building on this analogy, introduce a simple diffusion process that generalizes existing discrete diffusion models. We further propose a simulation-free training framework based on radial symmetry, along with a simple technique to address the high dimensionality of the manifold. Comprehensive experiments on language modeling benchmarks and other modalities show that our method outperforms existing discrete diffusion models and approaches the performance of autoregressive models. The code is available at https://github.com/harryjo97/RDLM.

### 1 Introduction

Discrete diffusion models [2, 39] emerged as a promising competitor to autoregressive models for the generative modeling of discrete data. These models have demonstrated competitive performance on tasks such as language modeling [49, 52] and code generation [20]. Unlike autoregressive models that generate data sequentially, diffusion models generate the sequence in parallel, allowing for bidirectional controllable generation and faster sampling.

However, discrete diffusion models do not fully harness the power of iterative refinement, which is the key to generative modeling of continuous data such as image synthesis [19, 48] and video generation [5, 46]. In these models, the forward process progressively corrupts data through stochastic jumps between discrete states, modeled as a Markov chain. Denoising is achieved through transitions between these discrete states, which results in the loss of informative signals during refinement. Hence, discrete diffusion models often exhibit limited generative performance and reduced controllability.

Several efforts have been made to adapt continuous diffusion models for discrete data, motivated by their advantages in controllability [26], efficient sampling [40, 41], optimized design choices [10, 34], and the potential to unify different modalities [35, 56]. However, their performance often significantly lags behind that of discrete diffusion models. Early methods [24, 36] extended image diffusion models to discrete domains by applying unconstrained continuous relaxation. Other approaches [3, 54] project discrete data onto the probability simplex using the Dirichlet distribution as its prior over categorical distributions, but often fail to capture complex patterns. Recent works [12, 15] apply flow matching on the statistical manifold to learn categorical distributions, but these methods are limited to short sequences and small vocabularies. In particular, the connection between discrete and continuous diffusion remains poorly understood, hindering the development of a unified diffusion framework.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

In this work, we present Riemannian Diffusion Language Model (RDLM), a continuous diffusion framework for language modeling that incorporates the geometry of the statistical manifold into the diffusion processes. We establish a connection between continuous flow on the statistical manifold and the discrete diffusion process, showing that the transition distribution can be modeled as a conditional flow on the manifold. Based on the analogy, we introduce a simple design of the diffusion processes on the manifold that generalizes previous discrete diffusion models. We further present a simulationfree training scheme that leverages radial symmetry, consisting of a simple parameterization and maximum likelihood-based training objectives. Through experiments on language modeling, image modeling, and biological sequence design, we validate that our framework outperforms existing discrete and continuous diffusion models.

### 2 Background

#### 2.1 Discrete diffusion models

Discrete diffusion models [2, 39, 49, 52] define the diffusion process directly on discrete states using the Markov chains. The forward process describes the transition from the current state to other states, which is formalized by multiplying the transition matrix Qt:

q(xt|xt−1) = Cat(xt;Qtxt−1), (1)

where xt is the random variable for the discrete states and Cat(·) denotes the categorical distribution. The marginal distribution corresponds to repeatedly multiplying transition matrices over time steps:

q(xt|x) = Cat(xt;Q¯tx) = Cat(xt;Qt ···Q1x). (2)

Austin et al. [2] introduced several designs of the transition matrices, including masked (absorbing state) and uniform diffusion, and has been extended to continuous-time Markov chains (CTMC) [2, 6].

#### 2.2 Statistical manifold of categorical distribution

Let X = {1,··· ,d} denote the discrete data space, and let ∆d−1 = {(p1,··· ,pd) ∈ Rd| i pi = 1,pi ≥ 0} denote the (d − 1)-dimensional probability simplex. A categorical distribution over X

can be parameterized by the parameters p1,··· ,pd satisfying i pi = 1 and pi ≥ 0. The statistical manifold P(X) of the categorical distributions thus corresponds to the simplex ∆d−1 equipped with the Fisher-Rao metric [1, 47] (see Appendix A.1). There exists a diffeomorphism from the statistical

manifold P(X) to the positive orthant of the (d − 1)-dimensional sphere Sd+−1:

π : P(X) → Sd+−1; pi  → ui = √pi, (3)

which induces the geodesic distance dg(u,v)=cos-1⟨u,v⟩ for u,v ∈ Sd+−1, where ⟨·,·⟩ denotes the Euclidean inner product. We provide a more detailed explanation in Appendix A.1.

### 3 Riemannian Diffusion Language Model

We introduce a novel continuous diffusion model for language modeling. In this section, we present a single token generation framework, which we generalize to modeling sequences in Section 4.

#### 3.1 Generalization of discrete diffusion

Continuous reparameterization of discrete data To incorporate the geometry of the underlying categorical distribution, we leverage the statistical manifold to parameterize discrete data [12, 15]. Each point on the statistical manifold P(X) corresponds to the parameters of a categorical distribution over the discrete sample space X = {1,··· ,d}. In this way, discrete data can be represented as continuous parameters of categorical distributions on the manifold.

Yet the Fisher-Rao metric is ill-defined on the boundary of the manifold where the initial distribution of the parameterized data lies, leading to numerical instabilities near the boundary. To address this, we leverage the diffeomorphism π (Eq. (3)) which maps P(X) to the positive orthant of a hypersphere

Sd+−1 [12, 15], where each point u ∈ Sd+−1 corresponds to Cat(·;π-1(u)). This mapping enables

(a) (b)

e3

e3

X0=em

e3

|0.3|
|---|
|0.2|
|0.5|

π

###### Xt

Yt=π(Zt)

Xt

Zt

X0

e1 e2

e2

e2

e2

e1

e1

e1

q(xt|x0) ∆2 S2

Masked Diffusion

Uniform Diffusion on X ={1,2,3}

on X ={1, 2,m}

Figure 1: Illustration of the continuous reparameterization of discrete data and two types of our generative process on hypersphere. (a) Example of a transition distribution of a discrete diffusion process modeled by a continuous flow on a d-dimensional sphere. (b) Illustration of the diffusion processes on S2 generalizing masked diffusion and uniform diffusion, respectively.

discrete data to be reparameterized as continuous states on Sd−1 while preserving the geometry of the categorical distribution, which we illustrate in Figure 1 (a). The reparameterized data distribution

pdata on the hypersphere can be written as pdata(x) = dk=1 pkδ(x−ek) where pk denotes the probability of the k-th state, and ek are d-dimensional one-hot vectors. In the case of masked diffusion, the discrete sample space is augmented with an additional mask state m.

From discrete diffusion to continuous flow Our key observation is that the transition distribution qt(xt|x) of a discrete diffusion process (Eq. (2)) is a categorical distribution on X. Therefore, modeling qt is equivalent to modeling continuous flow on the statistical manifold P(X). We show in the following proposition that discrete diffusion models over X can be modeled by a continuous flow on P(X) and further on Sd+−1 (we provide the full proof in Appendix A.2).

Proposition 3.1. The transition distribution of discrete diffusion processes can be modeled by the continuous flow on the statistical manifold, and further on the hypersphere. proof sketch. A flow on Sd+−1 that interpolates y0 and y1 as geodesic is described by the ODE:

dYt dt

dlog κt dt

exp-1

Yt(y1), Y0 = y0, (4)

= −

where exp-1 denotes the logarithm map on the hypersphere. Then, for a well-designed schedule κt and endpoint y1, the process Zt := π(Yt) on P(X) corresponds to the transition distribution of the discrete diffusion process. In particular, we obtain the masked diffusion process for y1 = em, i.e.,

√

the mask token, and the uniform diffusion process for y1 = di=1 ei/

d.

| |
|---|

Although discrete diffusion processes can be represented as a flow on the statistical manifold, this flow cannot be learned by a neural network. The network fails to generalize to points outside the geodesic that interpolates the prior and the data distribution, producing an incorrect vector field. Therefore, we present a simple design for the continuous diffusion model that generalizes existing discrete diffusion models.

#### 3.2 Generative process on hypersphere

The task of modeling the distribution of discrete data can be reformulated as modeling a distribution pdata on the hypersphere. Building upon the Riemannian diffusion mixture framework [33], we construct a diffusion process on Sd−1 such that its terminal distribution matches pdata. The construction entails deriving a diffusion mixture representation based on bridge processes defined on Sd−1.

We first derive a bridge process {X¯t}Tt=0 on Sd−1 from an arbitrary point x0 ∈ Sd−1 to ek as follows (we provide detailed derivation in Appendix A.3):

cos-1⟨X¯t,ek⟩(ek − ⟨X¯t,ek⟩X¯t) 1 − ⟨X¯t,ek⟩2

dX¯t = γt

dt + σtdBdt , X¯0 = x0, (5)

where γt := σt2/ t T σs2ds and Bdt denotes the Brownian motion defined on Sd−1. Intuitively, the current state Xt moves in the direction that minimizes the geodesic distance to the endpoint, resulting in a process that bridges the starting and end points. While different forms of the bridge process

exist, for example, scaling the drift or the diffusion coefficients, Eq. (5) yields a specific transition distribution that enables simulation-free training, which we explain in Section 3.3.

From the bridge processes, we construct a generative process {Xt}Tt=0 on Sd−1 using the diffusion mixture representation (see Appendix A.4 for the formal definition of the diffusion mixture representation and the derivation of the generative process in Corollary A.8):

d

pT|t(ek|Xt)ηk(Xt,t) dt + σtdBdt, X0 = x0, (6)

dXt =

k=1

where ηk(·,t) denote the drift of the bridge process in Eq. (5). Here, pT|t(ek|Xt) represents the probability that ek will be the final outcome of the process at time T, given the current state Xt at time t. Note that the construction guarantees the terminal distribution of the process to be pdata.

An ideal generative process is one that gradually refines the uninformative states to recover the original tokens. We analyze the convergence of the bridge process through its radial process rtk := dg(X¯t,ek) described by the following SDE (see Appendix A.3 for the derivation using Itô’s formula):

(d − 1)σt2 2

cotrtk dt + σtdWt, r0k = cos-1⟨x0,ek⟩, (7)

drtk = −γtrtk +

where Wt is a 1-dimensional Wiener process. For σ0 > σT, the radial process converges rapidly in early time steps, making it difficult for a neural network to approximate accurately. We empirically

find that the geometric schedule σt = σ0T−tσTt with σ0 < σT leads to gradual convergence.

Masked diffusion Based on Proposition 3.1, initializing the generative process in Eq. (6) with the mask token, i.e., X0=em, yields a mixture process that generalizes the discrete masked diffusion framework. The diffusion process starts at the mask token and progressively evolves toward one of the target tokens ek, as visualized in Figure 1 (b). From the perspective of the discrete diffusion model, our mixture process smoothly interpolates the discrete jump from em to ek through intermediate continuous states Xt, where the final token is determined by the probability pT|t(ek|Xt).

The fundamental difference is that discrete masked diffusion operates through direct jumps between a token and the mask token, where any incorrect transition is irreversible. In contrast, our continuous approach allows for gradual transitions, providing numerous opportunities to correct wrong predictions during the process. This leads to more accurate modeling of the underlying data distribution.

Uniform diffusion Based on Proposition 3.1, the generalization of uniform diffusion can be achieved by initializing the generative process of Eq. (6) with the barycenter of the simplex ∆d−1

√

projected onto Sd−1, i.e., X0=π( di=1 ei/d)= di=1 ei/

d. We visualize the diffusion process in Figure 1 (b). Intuitively, the barycenter of ∆d−1 corresponds to the uniform categorical distribution over d categories, which serves as the stationary distribution of the discrete uniform diffusion process.

Mixture paths We derive a new family of generative processes by constructing a mixture over the time marginals of generative processes {Qit: 1 ≤ i ≤ n} (see Appendix A.5 for derivation):

n

n

Qmixt :=

λitQit ,

λit = 1, 0 ≤ λit ≤ 1, (8)

i=1

i=1

where λit is the time-dependent mixing schedule assigned to the i-the generative path. This construction allows the resulting process to transition between different generative behaviors over time.

In particular, we propose a simple yet effective mixture path built from mixing the time marginals of the masked diffusion and uniform diffusion, for a time-dependent schedule λt as follows:

λtQmaskt + (1 − λt)Qunift , (9) with initial distribution λ0δ(em)+(1−λ0)δ( di=1 ei/

√

d). This formulation generalizes the mixture paths used in discrete flow matching [51] and the state-dependent schedule [52].

Generalizing flow matching Notably, our framework generalizes previous flow matching methods on the statistical manifold [12, 15]. By designing the noise schedule in Eq. (5) to be σt ≡ σ0 → 0, we obtain the conditional vector field of the flow matching models.

#### 3.3 Simulation-Free Training with Radial Symmetry

Next, we introduce our training scheme. We present a simple parameterization of our generative model and derive the likelihood bound and training objectives. Further, we present a simulation-free training method based on the radial symmetry of the hypersphere.

Model parameterization To use the diffusion process in Eq. (6) as a generative model, its unknown drift should be learned through a neural network, similarly to flow matching [8, 37] or bridge matching [33]. Yet the drift of the mixture process diverges near the terminal time T, which makes it challenging to learn. Therefore, instead of approximating the drift function directly, we propose to model the probability pT|t(XT|Xt) with a neural network sθ as follows:

T

, (10)

pθ(Xt,t) := softmax(sθ(Xt,t)) = pT|t(e1|Xt),··· ,pT|t(ed|Xt)

which converges to ek for some k as t → T. In the case of masked diffusion, we set the probability pT|t(em|Xt) to be zero for all t, indicating that the final state cannot be a mask token. From Eq. (10), the drift of the mixture process in Eq. (6) is parameterized as follows:

d

pθ(Xt,t),ek ηk(Xt,t). (11)

ηθ(Xt,t) =

k=1

Our parameterization shares similar properties with the discrete masked diffusion [49]: (1) Zero Mask Probabilities. The final state cannot be a mask token. (2) Carry-Over Unmasking. If Xt converges to a token ek before the terminal time, ηθ converges to zero, and the state Xt is carried over without changing to different token.

Likelihood bound We derive a tractable upper bound on the negative likelihood of our generative model by applying the Girsanov theorem on compact manifolds (De Bortoli et al. [16], Corollary H.3). Specifically, we first establish a point-wise upper bound on the negative log-likelihood under the parameterized mixture process Qθ, using the KL divergence between Qθ and a bridge process Qk, which is conditioned on endpoints x0 and ek. Applying the Girsanov theorem, we obtain the following variational upper bound (we provide a detailed derivation in Appendix A.6):

−log pˆθ(ek) = DKL(QkT∥QθT) ≤ EX∼Qk

- 1

- 2

T

0

σt−1 ηθ(Xt,t) − ηk(Xt,t)

2

dt (12)

2

where ηk is the drift defined in Eq. (5). The point-wise likelihood bound yields an upper bound on the negative log-likelihood of our generative model parameterized by pθ:

2

T

- 1

- 2

σt−1 ηθ(Xt,t) − ηk(Xt,t)

dt . (13)

Ez∼p

data − log pˆθ(z) ≤ Eek∼pdata

X∼Qk

0

2

Objective Based on the likelihood bound in Eq. (13), we introduce a maximum likelihood training objective for the model parameterization pθ in Eq. (10):

 1

 . (14)

2

d

T

σt−2

pθ(Xt,t),el ηl(Xt,t) − ηk(Xt,t)

L(θ) = Eek∼pdata

dt

2

X∼Qk

0

l=1

2

This objective corresponds to minimizing the mean squared error in approximating the drift term.

In particular, L(θ) can be minimized by reducing the cross-entropy between the predicted probability pθ(Xt,t) and the target one-hot vector ek. Therefore we present a cross-entropy-based training objective, analogous to those used in discrete diffusion models [49, 52]:

T

LCE(θ) = Eek∼pdata

−log pθ(Xt,t),ek dt . (15)

X∼Qk

0

We show in Appendix A.7 that minimizing the cross-entropy-based objective in Eq. (15) leads to minimizing L(θ), thereby ensuring maximum likelihood training. We experimentally find that the cross-entropy loss LCE(θ) yields faster convergence in training and leads to better performance than the mean squared error loss L(θ).

Importance sampling The difficulty of approximating the probability pT|t(XT|Xt) varies significantly across different time points t. While predicting XT is fairly easy in the later stage of the process, it is challenging to do so during the middle of the process. The training can be improved by training more on the challenging time points. We derive an equivalent objective by applying importance sampling over t, which reweights the time distribution to focus on a specific interval:

Et∼q − q(t)-1 log pθ(Xt,t),ek (16)

LCEq (θ) = Eek∼pdata

X∼Qk

where q is a normalized proposal distribution over t. We find that a simple choice q(t) = ϵ + (1 − 2ϵ)1[a,b](t) with small ϵ effectively concentrates sampling within the desired time interval.

Approximation of transition distribution Our training objective involves sampling Xt from the bridge processes at each iteration. Yet this introduces a significant bottleneck during training, as it requires simulating the process due to its intractable transition distribution on the d-dimensional sphere. Therefore, we present an approximate sampling method that bypasses the need for simulation, thereby enabling scalable training across large vocabularies.

We propose to approximate the distribution p(Xt|X0,XT) as the push-forward of a Gaussian distribution on the tangent space via the exponential map, i.e., the Riemannian normal. This approximation is justified by the fact that Eq. (5) results from applying a time change [64] to a simple bridge process (Eq. (59)), which yields a transition distribution similar to Riemannian normal.

We parameterize the mean of the Riemannian normal distribution as µt := EXt/∥EXt∥ and its covariance Σt := Cov exp-1

µt(Xt) , using the parameters αt and ρt as follows: µt =

αt cosϕ0 sinϕ0

αt sinϕ0

X0 , Σt = ρ2tI, (17)

XT + 1 − αt2 −

where ϕ0 := cos-1⟨X0,XT⟩. Intuitively, µt represents the normalized centroid of the samples Xt, and Σt captures to the covariance of the lifted samples in the tangent space Tµt

.

Parameters of Riemannian normal While the parameters αt and ρt are generally intractable, we derive them from the 1-dimensional projections of the mixture process. Our main idea is to express

the parameters in terms of the projected processes ztT :=⟨Xt|0,T,XT⟩ and zt0:=⟨Xt|0,T,X0⟩, where Xt|0,T denotes the diffusion process {Xt}Tt=0 conditioned on fixed endpoints X0 and XT. These projected processes are modeled by the following 1-dimensional SDEs (see Appendix A.8 for the derivation using the Itô’s formula and the radial symmetry of Sd−1):

(d − 1)σt2 2

dztT = γt cos-1ztT 1 − (ztT)2 −

ztT dt + σt 1 − (ztT)2 dWtT, (18)

cos-1ztT 1 − (ztT)2

(d − 1)σt2 2

z0T − zt0ztT −

dzt0 = γt

zt0 dt + σt 1 − (zt0)2 dWt0, (19)

with z0T = ⟨X0,XT⟩ and z00 = 1, where WtT and Wt0 denote 1-dimensional Wiener processes. In the case of masked and uniform diffusion, X0 is fixed to a single point such that ⟨X0,ek⟩ is identical for all non-mask tokens ek. As a result, the mean projections EztT and Ezt0 remain invariant with respect to the choice of XT.

Based on the radial symmetry of Sd−1, we derive the parameters αt and ρt from the mean projections Ezt0 and EztT as follows (we provide detailed derviation in Appendix A.9):

(EztT/Ezt0 − cosϕ0)2 sin2 ϕ0 + (EztT/Ezt0 − cosϕ0)2

, ρt = F-1

d Ezt0/ 1 − αt2 , (20)

αt =

where ϕ0 := cos-1⟨X0,XT⟩ and F-1

d denotes the inverse of a damped Kummer function (Eq. (115)). For small values of d, we calibrate ρt by applying a constant scaling factor.

The mean projections Ezt0 and EztT can be easily obtained by simulating the 1-dimensional processes Eq. (18) and Eq. (19). Therefore, prior to training our model pθ, we precompute the parameters {αi/K,ρi/K}Ki=0 once, using a sufficiently large value of K. The procedure for this precomputation is outlined in Algorithm 3 in the Appendix.

- Algorithm 1 Training Input: Initial point u, model pθ, vocabulary size d, token sequence length L, time distribution q(t),

pre-computed {αi/K,ρi/K}Ki=0 For each epoch:

- 1: Sample token sequence s from the training set
- 2: X0 ← (u)L and X1 ← ONE-HOT(si,d) Li=1
- 3: ϕ0 ← cos-1 X0i,X1i Li=1
- 4: t ∼ q and αt,ρt ← INTERPOLATE {αi/K,ρi/K}Ki=0
- 5: µt ← α

t

sinϕi0 X1i + 1 − αt2 − αt cosϕ

i 0

sinϕi0 X0i

L i=1

▷ Eq. (17)

- 6: Xt ∼ NSd−1(µ1t,ρ2tId) × ··· × NSd−1(µLt ,ρ2tId) ▷ Sample from Riemannian normal
- 7: Lθ ← −q(t)-1 log pθ(Xt,t),X1 ▷ Cross-entropy-based loss in Eq. (16)
- 8: Update θ using Lθ

- Algorithm 2 Sampling

Input: Initial point u, trained model pθ, vocabulary size d, number of sampling steps M, token sequence length L, noise schedule σt

- 1: X ∼ (u)L, t ← 0 and δt ← 1/M ▷ Start from the initial point
- 2: for m = 1 to M do
- 3: w ∼ N(0,Id) L
- 4: p ← pθ(X,t)
- 5: ηθ ← dk=1 pi,ek γtcos

-1⟨X√i,ek⟩(ek−⟨Xi,ek⟩Xi) 1−⟨Xi,ek⟩2

L

i=1

▷ Parameterization in Eq. (11)

- 6: X ← expXi ηθiδt + σt

√

δtwi

L i=1

▷ Geodesic random walk

- 7: t ← t + δt
- 8: end for
- 9: s ← ARGMAX(Xi) Li=1
- 10: Return: Token sequence s

During training, we can sample Xt from the Riemannian normal distribution without expensive simulation of the bridge processes. Compared to the simulation-based training, our approach yields a ×50 speedup. In Section 6.4, we experimentally demonstrate that the Riemannian normal provides an accurate approximation of the distribution of Xt.

### 4 Generation of Token Sequences

Modeling sequence of tokens We now extend the single-token modeling framework to the generation of token sequences. Since each token in the sequence is reparameterized onto a d-dimensional sphere, a sequence of length n is modeled on the product manifold (Sd−1)n. This formulation allows the sequence-level diffusion to be treated as a joint process over the spherical components.

We model the generative process as a system of n SDEs {(Xt1,···,Xtn)}Tt=0, where each Xti evolves according to a diffusion process on Sd−1, analogous to the single-token formulation in Eq. (6):

dXti =

d

p(XTi =ek|Xt1:n) ηk(Xti,t) dt + σtdBdt, 1 ≤ i ≤ n. (21)

k=1

Here p(XTi = ek|Xt1:n) denotes the probability that the i-th token corresponds to the k-th state, conditioned on the current intermediate sequence Xt1:n. Using the parameterization defined in Eq. (10), we train a neural network to predict p(XT1:n|Xt1:n). The training and sampling procedures for modeling token sequences are outlined in Algorithms 1 and 2, respectively.

Dimension splitting of statistical manifold For a large vocabulary set, the corresponding statistical manifold becomes high-dimensional, which introduces two challenges: (1) Sharp transition. Bridge processes on high-dimensional spheres tend to exhibit sharp transitions near the terminal time. This high-dimensional convergence behavior makes the mixture process difficult for neural networks to learn. (2) High input dimensionality. The input to the network resides in a high-dimensional space, requiring sufficiently large hidden dimensions to encode the data adequately. Models with limited capacity fail to learn the conditional probability p(XT1:n|Xt1:n). To address these challenges, we introduce dimension splitting, a simple technique to reduce the dimensionality of the parameterized manifold. Instead of mapping the k-th token directly to Sd−1, we first represent the index k in base b, and then map the represntation to the product manifold (Sb)m for m:=⌈logb d⌉. Dimension splitting reparameterizes a sequence of length L to a product manifold (Sb)mL. The resulting bridge processes on Sb with small b exhibit gradual convergence over time, making them significantly easier for neural networks to learn. Dimension splitting significantly enhances the likelihood when used together with the mixture path (Eq. (9)).

### 5 Related Work

Discrete diffusion models Discrete diffusion directly models the Markov chain on the discrete data space. One-hot data distributions are gradually corrupted to a stationary distribution using specific transition matrices, and the noising process corresponds to the stochastic jumps between states in the Markov chain. D3PM [2] introduces discrete-time Markov forward processes with both uniform and absorbing state transition matrices, and has been generalized to the continuous-time Markov chain framework [6]. SEDD [39] proposes learning the score entropy of discrete states instead of predicting the mean. Recent works [49, 52] introduce continuous-time masked diffusion models, which offer simpler likelihood bounds compared to previous works. We provide further discussions on comparison with discrete diffusion models in Appendix A.10.

Continuous diffusion models for discrete data Early approaches to discrete data modeling either fully relaxed discrete data into continuous space [24] or embedded tokens into a latent space [18, 36], without imposing any constraint. However, continuous relaxation without constraint fails to capture the discreteness of the categorical distribution. Recent works operate directly in logit space [21, 29] or on the probability simplex [3, 54], but rely on imperfect assumptions that fail to accurately represent the underlying categorical distribution. Flow matching has been applied to the statistical manifold to model the categorical distribution [12, 15], but these methods are limited to short sequences and small vocabularies. We provide a detailed comparison in Appendix A.10.

### 6 Experiments

#### 6.1 Text generation

We evaluate our Riemannian Diffusion Language Model (RDLM) for text generation tasks on two language benchmarks: Text8 [42] and One Billion Words Dataset [7].

Baselines We compare against state-of-the-art diffusion and autoregressive models. Multinomial Diffusion [29], D3PM [2], SEDD [39], MDLM [49], MD4 [52] are discrete diffusion models. Plaid [23] and Bayesian Flow Network (BFN) [21] are continuous diffusion models. IAF/SCF [63], AR Argmax Flow [29], and Discrete Flow [58] are flow-based models, and ARDM [30] and MAC [53] are any-order autoregressive models. We also compare with the transformer AR model [61]. We provide further details on the baselines in Appendix B.2

Implementation details For all experiments, we use the same data split and context size following Lou et al. [39] and Sahoo et al. [49]. For Text8, we randomly sample contiguous chunks of length 256 as done in previous works [2, 39]. For One Billion Words, we use the same tokenizer as in He et al. [25] with context size 128. We use a diffusion transformer architecture [44] with rotary positional embeddings [55] for all the experiments and match the number of parameters as used in the previous works [39, 49]. For our model, we use the mixture path of masked and uniform diffusion (Eq. (9)) and apply dimension splitting for a large vocabulary. We provide more details in Appendix B.2.

Table 2: Test perplexity results on LM1B dataset. Baseline results taken from Sahoo et al. [49].

Method # Param. PPL (↓) Autoregressive

Transformer-X Base [14] 0.46B 23.5 OmniNetT [57] 100M 21.5 Transformer [61] 110M 22.32

Discrete Diffusion

BERT-Mouth [62] 110M ≤ 142.89 D3PM Absorb [2] 70M ≤ 76.90 DiffusionBert [25] 110M ≤ 63.78 SEDD [39] 110M ≤ 32.79 MDLM [49] 110M ≤ 27.04

Continuous Diffusion

Diffusion-LM [36] 80M ≤ 118.62 RDLM (Ours) 110M ≤ 28.44

Table 3: BPD results on CIFAR-10 test set. Baseline results taken from Shi et al. [52].

Method # Param. BPD (↓) Autoregressive

PixelRNN [60] 3.00 Gated PixelCNN [59] 3.03 PixelCNN++ [50] 53M 2.92 PixelSNAIL [11] 46M 2.85 Image Transformer [43] 2.90 Sparse Transformer [13] 59M 2.80

Discrete Diffusion

D3PM Absorb [2] 37M ≤ 4.40 D3PM Gauss [2] 36M ≤ 3.44 τLDR [6] 36M ≤ 3.59 τLDR Absorb [6] 36M ≤ 3.52 MD4 [52] 28M ≤ 2.78

Continuous Diffusion RDLM (Ours) 28M ≤ 2.73

Text8 We first evaluate on a small-scale character-level language modeling task. The Text8 [42] dataset is a character-level text modeling benchmark extracted from English Wikipedia. We train models on short text chunks of length 256 and evaluate the performance using Bits Per Character (BPC). As shown in Table 1, our framework outperforms all previous diffusion models, including both discrete and continuous methods. We also outperform anyorder autoregressive models that generate texts in flexible decoding order, similar to discrete diffusion models. We achieve similar generative perplexity and entropy compared to existing discrete diffusion models. We provide generated texts from RDLM in Appendix C.1.

Table 1: Bits Per Character (BPC) results on Text8 test set. Results are taken from the corresponding papers.

Method BPC (↓) Autoregressive

AR Argmax Flow [29] 1.39 Transformer AR [61] 1.23 Discrete Flow [58] 1.23

Any-order Autoregressive

ARDM [30] ≤ 1.43 MAC [53] ≤ 1.40

Discrete Diffusion

Multinomial Diffusion [29] ≤ 1.72 D3PM Uniform [2] ≤ 1.61 D3PM Absorb [2] ≤ 1.45 SEDD Absorb [39] ≤ 1.39 MDLM [49] ≤ 1.40 MD4 [52] ≤ 1.37

One Billion Words We further evaluate RDLM on One Billion Words Dataset (LM1B) [7], a medium-scale realworld language benchmark with a vocabulary size of 30522. We evaluate the performance using perplexity (PPL), and the results are summarized in Table 2. RDLM outperforms most existing diffusion models and is competitive with the state-of-the-art discrete diffusion model [49]. Notably, ours significantly outperforms the prior continuous diffusion model [36], demonstrating the effectiveness of incorporating the geometry of the underlying categorical distribution. We provide a discussion of the results with MDLM [49] in Appendix B.2. The generated texts are presented in Appendix C.2.

Continuous Diffusion

Plaid [23] ≤ 1.48 BFN [21] ≤ 1.41 RDLM (Ours) ≤ 1.32

#### 6.2 Pixel-level image modeling

We further explore applications of RDLM beyond the text domain by applying it to order-agnostic image data. Each image is represented as a set of discrete tokens with a vocabulary of size 256, removing information about pixel proximity. Note that this is different from the experimental settings with image diffusion models [27, 34] that use spatial information. We compare RDLM against autoregressive models and discrete diffusion models that operate directly on raw pixel space, which we describe in Appendix B.3. As shown in Table 3, our method achieves the lowest Bits Per Dimension (BPD), outperforming the discrete diffusion models [2, 52] and autoregressive baselines [11, 13]. We attribute this strong performance on inherently continuous data to the continuous nature of our framework, which fully exploits iterative refinement, suggesting its potential for unifying modeling across different modalities.

#### 6.3 DNA sequence design

We demonstrate that our framework can be applied to biological sequence generation. We evaluate our method on the promoter DNA sequence design task, which aims to generate valid promoter DNA sequences conditioned on transcription profiles. A detailed description of the task is provided in Appendix B.4. Model performance is measured by the mean squared error (MSE) between the predicted regulatory activity of the generated sequence and that of the original sequence corresponding to the transcription profile. Table 4 shows that our framework achieves the lowest MSE, outperforming the flow matching methods [15, 54] and the discrete diffusion model [2].

Table 4: MSE results on the generated promoter DNA sequences. Baseline results are taken from Davis et al. [15].

Method MSE (↓)

Bit-Diffusion (bit) [10] 0.041 Bit-Diffusion (one-hot) [10] 0.040 D3PM Uniform [2] 0.038 DDSM [3] 0.033 DirichletFM [54] 0.034 Language Model 0.034 Fisher-Flow [15] 0.029 RDLM (Ours) 0.027

#### 6.4 Analysis

Training objective We validate the effectiveness of our cross-entropy-based loss of Eq. (15) in Table 5. Compared to the mean squared error loss of Eq. (14), the cross-entropy loss provides faster convergence in training and better NLL. Furthermore, Table 5 shows that applying importance sampling to the training objective as defined in Eq. (16) yields improved likelihood.

Approximation of transition distribution We validate that our approximate sampling method closely matches the true transition distribution of the mixture process. In Figure 2, we report the maximum mean discrepancy (MMD) [22] distance between the simulated transition distribution and the approximated distribution obtained using the Riemannian normal. The approximated distributions exhibit nearly identical MMD as the simulated distributions, indicating that he approximation is accurate and reliable. Notably, the discrepancy approaches zero in high-dimensional manifolds, where simulation becomes increasingly expensive, making simulation-based training impractical.

Dimension splitting For datasets with a large vocabulary, such as the LM1B dataset, our dimension splitting technique (Section 4) results in a significant improvement. Table 6 shows that directly training a model on discrete data with a large vocabulary fails to capture the underlying distribution, due to the high input dimensionality. In particular, the sharp transition near the terminal time for a high-dimensional mixture process makes it challenging for neural networks to learn. In large vocabulary settings, we achieve the best result via dimension splitting, combined with modeling the generative process using a mixture path of masked and uniform diffusion.

### 7 Conclusion

In this work, we introduced the Riemannian Diffusion Language Model (RDLM), a continuous diffusion model for language and discrete data. We present a simple framework that generalizes discrete diffusion models, building on the connection between the transition distribution and continuous flow on the statistical manifold. We provide general designs for generative processes and introduce a simulation-free training scheme leveraging the radial symmetry. Through experiments on language modeling benchmarks, RDLM demonstrates strong performance over prior discrete and continuous diffusion models. We further extend our approach to other modalities, including image and biological sequence generation, where RLDM achieves consistently strong results.

### 8 Acknowledgements

This work was supported by National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2023-00256259), Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.RS-2019-II190075 Artificial Intelligence Graduate School Program(KAIST)), Information & Communications Technology Planning & Evaluation (IITP) with a grant funded by the Ministry of Science and ICT (MSIT) of the Republic of Korea in connection with the Global AI Frontier Lab International Collaborative Research. (No. RS-2024-00469482 & RS-2024-00509279), and artificial intelligence industrial convergence

cluster development project funded by the Ministry of Science and ICT(MSIT, Korea)&Gwangju Metropolitan City.

### References

- [1] Shun-ichi Amari. Information geometry and its applications, volume 194. Springer, 2016.
- [2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, 2021.
- [3] Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In International Conference on Machine Learning, 2023.
- [4] Nihat Ay, Jürgen Jost, Hông Vân Lê, and Lorenz Schwachhöfer. Information geometry, volume 64. Springer, 2017.
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators, 2024.
- [6] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, 2022.
- [7] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, 2013.
- [8] Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In International Conference on Learning Representations, 2024.
- [9] Ting Chen. On the importance of noise scheduling for diffusion models. arXiv:2301.10972, 2023.
- [10] Ting Chen, Ruixiang Zhang, and Geoffrey E. Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In International Conference on Learning Representation, 2023.
- [11] Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. Pixelsnail: An improved autoregressive generative model. In International Conference on Machine Learning, 2018.
- [12] Chaoran Cheng, Jiahan Li, Jian Peng, and Ge Liu. Categorical flow matching on statistical manifolds. In Advances in Neural Information Processing Systems, 2024.
- [13] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv:1904.10509, 2019.
- [14] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In Association for Computational Linguistics, 2019.
- [15] Oscar Davis, Samuel Kessler, Mircea Petrache, ˙Ismail ˙Ilkan Ceylan, Michael M. Bronstein, and Avishek Joey Bose. Fisher flow matching for generative modeling over discrete data. In Advances in Neural Information Processing Systems, 2024.
- [16] Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. In Advances in Neural Information Processing Systems, 2022.
- [17] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, 2021.

- [18] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. arXiv:2211.15089, 2022.
- [19] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.
- [20] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. In Advances in Neural Information Processing Systems, 2024.
- [21] Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson, and Faustino Gomez. Bayesian flow networks. arXiv:2308.07037, 2023.
- [22] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. The Journal of Machine Learning Research, 13(1):723–773, 2012.
- [23] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. In Advances in Neural Information Processing Systems, 2024.
- [24] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control. arXiv:2210.17432, 2022.
- [25] Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. In Annual Meeting of the Association for Computational Linguistics, 2023.
- [26] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv:2207.12598, 2022.
- [27] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020.
- [28] Chung-Chau Hon, Jordan A Ramilowski, Jayson Harshbarger, Nicolas Bertin, Owen JL Rackham, Julian Gough, Elena Denisenko, Sebastian Schmeier, Thomas M Poulsen, Jessica Severin, et al. An atlas of human long non-coding rnas with accurate 5 ends. Nature, 543(7644):199–204, 2017.
- [29] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, 2021.
- [30] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In International Conference on Learning Representation, 2022.
- [31] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, 2023.
- [32] Elton P Hsu. Stochastic analysis on manifolds. Number 38 in Graduate studies in mathematics. American Mathematical Society, 2002.
- [33] Jaehyeong Jo and Sung Ju Hwang. Generative modeling on manifolds through mixture of riemannian diffusion processes. In International Conference on Machine Learning, 2024.
- [34] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems, 2022.
- [35] Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Zichun Liao, Yusuke Kato, Kazuki Kozuka, and Aditya Grover. Omniflow: Any-to-any generation with multi-modal rectified flows. arXiv:2412.01169, 2024.

- [36] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. In Advances in Neural Information Processing Systems, 2022.
- [37] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.
- [38] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv:1711.05101, 2017.
- [39] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024.
- [40] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, 2022.
- [41] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv:2211.01095, 2022.
- [42] Matt Mahoney. Large text compression benchmark. https://www.mattmahoney.net/dc/ text.html, 2006. .
- [43] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International Conference on Machine Learning, 2018.
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.
- [45] Stefano Peluchetti. Non-denoising forward-time diffusions. Openreview, 2021.
- [46] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam S. Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali K. Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dmitry Vengertsev, Edgar Schönfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models. arXiv:2410.13720, 2024.
- [47] C Radhakrishna Rao. Information and the accuracy attainable in the estimation of statistical parameters. In Breakthroughs in Statistics: Foundations and basic theory, pages 235–247. Springer, 1992.
- [48] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, 2022.
- [49] Subham Sekhar Sahoo, Marianne Arriola, Aaron Gokaslan, Edgar Mariano Marroquin, Alexander M Rush, Yair Schiff, Justin T Chiu, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems, 2024.

- [50] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. In International Conference on Learning Representations, 2017.
- [51] Neta Shaul, Itai Gat, Marton Havasi, Daniel Severo, Anuroop Sriram, Peter Holderrieth, Brian Karrer, Yaron Lipman, and Ricky TQ Chen. Flow matching with general discrete paths: A kinetic-optimal perspective. arXiv:2412.03487, 2024.
- [52] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems, 2024.
- [53] Andy Shih, Dorsa Sadigh, and Stefano Ermon. Training and inference on any-order autoregressive models the right way. In Advances in Neural Information Processing Systems, 2022.
- [54] Hannes Stärk, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and Tommi S. Jaakkola. Dirichlet flow matching with applications to DNA sequence design. In International Conference on Machine Learning, 2024.
- [55] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [56] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. In Advances in Neural Information Processing Systems, 2023.
- [57] Yi Tay, Mostafa Dehghani, Vamsi Aribandi, Jai Prakash Gupta, Philip Pham, Zhen Qin, Dara Bahri, Da-Cheng Juan, and Donald Metzler. Omninet: Omnidirectional representations from transformers. In International Conference on Machine Learning, 2021.
- [58] Dustin Tran, Keyon Vafa, Kumar Krishna Agrawal, Laurent Dinh, and Ben Poole. Discrete flows: Invertible generative models of discrete data. In Advances in Neural Information Processing Systems, 2019.
- [59] Aäron van den Oord, Nal Kalchbrenner, Lasse Espeholt, Koray Kavukcuoglu, Oriol Vinyals, and Alex Graves. Conditional image generation with pixelcnn decoders. In Advances in Neural Information Processing Systems, 2016.
- [60] Aäron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International Conference on Machine Learning, 2016.
- [61] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.
- [62] Alex Wang and Kyunghyun Cho. BERT has a mouth, and it must speak: BERT as a markov random field language model. arXiv:1902.04094, 2019.
- [63] Zachary M. Ziegler and Alexander M. Rush. Latent normalizing flows for discrete sequences. In International Conference on Machine Learning, 2019.
- [64] Bernt Øksendal. Stochastic Differential Equations. Universitext. Springer Berlin Heidelberg, 2003.

## Appendix

### A Derivations

#### A.1 Preliminaries

Statistical Manifold of Categorical Distributions For a discrete sample space X = {1,2,··· ,d}, a d-class categorical distribution over X is parameterized by d number of parameters p1,··· ,pd ≥ 0 such tat di=1 pi = 1. The parameter space corresponds to the (d − 1)-dimensional probability simplex:

∆d−1 = (p1,··· ,pd) ∈ Rd :

d

pi = 1,pi ≥ 0 , (22)

i=1

A natural choice of a Riemannian metric on the simplex is the Fisher-Rao metric [1, 47]. For an interior point p ∈ ∆d−1, the Fisher-Rao metric is defined as follows:

d

x √p

xiyi pi

y √p

, x,y ∈ Tp∆d−1, (23)

gFR(p)[x,y] := ⟨x,y⟩p :=

,

=

i=1

where the normalization by √p in the inner product is performed component-wise. This induces a geodesic distance on the simplex defined as follows:

d(p,q) = 2cos−1

d

√piqi , p,q ∈ ∆d−1, (24)

i=1

where p and q corresponds to the parameters of categorical distributions. The probability simplex ∆d−1 equipped with the Fisher-Rao metric is a Riemannian manifold called the statistical manifold of categorical distribution, denoted as P(X) throughout the paper. The tangent space at an interior

point p is identified as Tp(P(X)) = x ∈ Rd : di=1 xi = 0 . For further details on the geometry of the statistical manifold, we refer the reader to Ay et al. [4].

Hypersphere Sd−1 denotes the (d−1)-dimensional sphere u=(u1,··· ,ud) : i u2i = 1 and Sd+−1 = u=(u1,··· ,ud) : i u2i = 1,ui ≥ 0 denotes a positive orthant of Sd−1. The hypersphere Sd−1 can be embedded into the ambient Euclidean space Rd, which induces a canonical inner product x,y := di=1 xiyi. For a discrete sample space X = {1,2,··· ,d}, there exists a diffeomorphism from P(X) to Sd+−1 defined as follows:

π : P(X) → Sd+−1 ; pi  → ui = √pi, π−1 : Sd+−1 → P(X) ; ui  → pi = u2i.

(25)

The diffeomorphism induces the the geodesic distance on Sd+−1:

dg(u,v) = cos−1⟨u,v⟩, u,v ∈ Sd+−1, (26)

for which the geodesic corresponds to the great circle connecting two points u and v. The corresponding exponential and logarithm maps on Sd−1 can be computed as follows:

x ∥x∥

, u ∈ Sd−1,x ∈ Tu(Sd−1), (27)

expu x = cos(∥x∥)u + sin(∥x∥)

cos-1⟨u,v⟩ 1 − ⟨u,v⟩2

exp-1

v − ⟨u,v⟩u , u,v ∈ Sd−1. (28)

u(v) =

Additionally, define the radial distance rv(x) := dg(x,v) ∈ R where dg denotes the geodesic distance on Sd−1. Then we have the following identities:

#### v − ⟨v,x⟩x

∇rv(x) = −

, (29) ∆rv(x) = (d − 1)cot(rv(x)), (30)

1 − ⟨v,x⟩2

= ⟨v,w⟩ − cosrv(x)cosrw(x) sinrv(x)sinrw(x)

∇rv(x),∇rw(x) = ⟨v,w⟩ − ⟨v,x⟩⟨w,x⟩ (1 − ⟨v,x⟩2)(1 − ⟨w,x⟩2)

. (31)

In particular, the logarithm map in Eq. (28) can be represented in radial distance:

exp-1

x (v) = −rv(x)∇rv(x), (32)

#### A.2 Connection Between Discrete Diffusion Models and Continuous Flow

In this section, we derive the connection between the discrete diffusion models and the continuous flow on a hypersphere.

Continuous Flow on Hypersphere We first derive a useful lemma for continuous flows on hyperspheres. The following lemma describes a continuous flow on the hypersphere as a spherical linear interpolation.

Lemma A.1. Define a flow {Yt}Tt=0 on Sd−1 from y0 ∈ Sd−1 to y1 ∈ Sd−1\{y0,−y0}:

dlog κt dt

dYt dt

exp−Y1

(y1), Y0 = y0, (33)

= −

t

where κt : [0,T] → [0,1] is a scalar function satisfying κ0 = 1 and κT = 0. Then the flow Yt has a closed form solution:

sin(θ0 − θt) sinθ0

sinθt sinθ0

y0, θt := κt cos-1⟨y0,y1⟩, (34) which corresponds to the spherical linear interpolation, i.e., slerp:

Yt =

y1 +

κt exp−y1

(y0) (35)

Yt = expy

1

1

Proof. Let θt := cos-1⟨Yt,y1⟩. Then Yt can be written as follows:

Yt = cosθty1 + sinθtwt, (36) where wt ∈ Rd is an unit vector. From the definition of θt, we have the following identity:

θt(y1 − Yt cosθt) sinθt

1 sinθt

dYt dt

1 sinθt −

dlog κt dt

dθt dt

,y1 (37)

= −

,y1 = −

1 − cos2 θt sinθt

1 sinθt

dlog κt dt

dlog κt dt

θt, (38) which yields representation of the flow Yt in Eq. (33) with respect to θ:

=

θt

=

y1 − Yt cosθt sinθt

dYt dt

dθt dt

. (39) Using the result of Eq. (39), we can see that wt is a constant vector independent of t:

=

1 sin2 θt

dYt dt −

dcosθt dt

dsinθt dt

dwt dt

(40)

y1 sinθt − (Yt − cosθty1)

=

1 sin2 θt

dθt dt − (y1 − Yt cosθt) + sin2 θty1 − cosθtYt + cos2 θty1 = 0. (41)

=

Therefore we get the closed form solution for Yt:

y0 − cosθ0y1 sinθ0

sin(θ0 − θt) sinθ0

sinθt sinθ0

y0, (42)

Yt = cosθty1 + sinθt

=

y1 +

where θt = κtθ0 from Eq. (38). Note that the solution Eq. (34) is well-defined in the sense that sinθ0 > 0 always holds. This is because ∥⟨Yt,y1⟩∥ ≤ 1 as Yt and y1 are on Sd−1. Finally, using the definition of θt, we can show the following:

Yt − YT cosθt sinθt

exp−Y1

= κtθ0wt = κtθ0w0 = κt exp−Y1

(Y0), (43) which gives the spherical linear interpolation defined in Eq. (35).

(Yt) = θt

T

T

| |
|---|

Our key observation is that the transition distribution qt(xt|x) of a discrete diffusion process (Eq. (2)) is a categorical. Therefore, modeling qt is equivalent to modeling the continuous flow on the statistical manifold P(X). Here, we show that discrete diffusion models over X can be modeled by a continuous

flow on Sd+−1. Specifically, we derive that the transition distribution of discrete diffusion processes can be modeled by the continuous flow on the hypersphere.

Masked Diffusion Model We first show that discrete masked diffusion models correspond to a continuous flow on the statistical manifold starting from an absorbing state.

Proposition A.2. Define a flow {Yt}Tt=0 on Sd−1 from ek to em:

sin−1(√αt) (44)

dYt dt

dlog κt dt

2 π

exp−Y1

= −

(em), Y0 = ek, κt =

t

where em denotes the absorbing state (i.e., mask state) and αt ∈ [0,1] is some differentiable noise schedule satisfying α0 ≈ 1 and α1 ≈ 0. Then the random variable Zt := π (Yt) ∈ Rd satisfies the following:

Zt = αtek + (1 − αt)em, (45) which is a flow that interpolates ek and em on the probability simplex ∆d−1.

Proof. Using Lemma A.1 with θ0 = cos-1⟨em,ek⟩ = π/2, we have the representation of Yt:

##### Yt = sin(θ0 − θt)em + sinθtek = √1 − αtem + √αtek, (46) since θt = sin−1(√αt). Therefore, Zt has the following closed form:

Zt = (1 − αt)em + αtek, (47) which defines a flow that interpolates ek and em on the probability simplex ∆d−1.

| |
|---|

Note that Zt is a random variable on ∆d−1 representing the categorical distribution Cat(αtex

##### +

0

(1 − αt)em). This corresponds to the transition distribution q(xt|x0) of a discrete masked diffusion model, where the transition matrix for the diffusion process is given as follows:





αt 0 ··· 0 0 0 αt ··· 0 0

... . .

Qabsorbt =

(48)

. .

 

 

- 0 0 ··· αt 0
- 1 − αt 1 − αt ··· 1 − αt 0

Corollary A.3. The discrete masked diffusion process can be modeled by a continuous flow on Sd−1 that starts from the absorbing state em.

Uniform Diffusion Model We also show that discrete uniform diffusion models correspond to a continuous flow on the statistical manifold that starts from the barycenter of the simplex.

√

Proposition A.4. Define a flow {Yt}Tt=0 on Sd−1 from ek to di=1 ei/

d: dYt dt

d

dlog κt dt

1 √

exp−Y1

ei , Y0 = ek, (49)

= −

t

d

i=1

sin-1 √1 − αt sinθ0 θ0

, θ0 := cos-1 1

(50)

√

κt = 1 −

d

where αt ∈ [0,1] is a differentiable noise schedule satisfying α0 ≈ 1 and α1 ≈ 0. Then the random variable Zt := π (Yt) ∈ Rd satisfies the following:

1 − αt d

1 + (d − 1)αt d

ek, (51)

Zt =

ei +

i̸=k

√

which is a flow that interpolates ek and di=1 ei/

d on the probability simplex ∆d−1.

√

Proof. Using Lemma A.1 with θ0 = cos-1(1/

d), we have the following representation of Yt:

d

sin(θ0 − θt) sinθ0

1 √

sinθt sinθ0

ek (52)

Yt =

ei +

d

i=1

√

sin(θ0 − θt) √d − 1

sin(θ0 − θt) √d − 1

dsinθt √d − 1

ek. (53)

ei +

+

=

i̸=k

Due to the definition of κt, Zt has the following closed form:

1 − αt d

1 + (d − 1)αt d

ek, (54)

Zt =

ei +

i̸=k

√

which defines a flow that interpolates ek and di=1 ei/

d, i.e., the barycenter of the probability simplex ∆d−1.

| |
|---|

Note that Zt is a random variable on ∆d−1 representing the categorical distribution:

 

 , (55)

1 − (d − 1)α d

1 − αt d

Cat

ei +

#### ex

0

i̸=x0

which corresponds to the transition distribution q(xt|x0) of a discrete uniform diffusion model. The transition matrix for the uniform diffusion process is given as follows:

  

   (56)

1 − N 1 ··· 1 1 1 − N ··· 1

Qunif =

... .

. .

1 1 ··· 1 − N

Corollary A.5. The discrete uniform diffusion process can be modeled by a continuous flow on Sd−1 that starts from the barycenter of the probability simplex.

#### A.3 Generative Process on Hypersphere

On a general manifold M that is complete, orientable, connected, and boundaryless, the logarithm bridge process [33] from x0 ∈ M to x1 ∈ M is defined as follows:

σt2

dX¯t = γt exp-1

X¯t(x1)dt + σtdBMt , X¯0 = x0 ; γt :=

(57)

T t σs2ds

where exp-1

x (·) denotes the logarithm map on M at point x and BMt is the Brownian motion defined on M. In the case of M = Sd−1, we can derive the logarithm bridge process from x0 to ek:

cos-1⟨X¯t,ek⟩(ek − ⟨X¯t,ek⟩X¯t) 1 − ⟨X¯t,ek⟩2

dt + σtdBdt, X¯0 = x0, (58)

dX¯t = γt

where we used the logarithm map of Eq. (28) and Bdt is a Brownian motion defined on Sd−1. It is worth noting that Eq. (58) is derived from applying the time change [64] to a simple bridge process:

cos-1⟨X¯t,ek⟩(ek − ⟨X¯t,ek⟩X¯t) 1 − ⟨X¯t,ek⟩2

1 T − t

dX¯t =

dt + dBdt , X¯0 = x0. (59)

Note that the drift of the logarithm bridge process can be rewritten using the geodesic distance dg(·,·) as follows:

dX¯t = γt cos-1⟨X¯t,ek⟩∇X¯tdg(X¯t,ek) dt + σtdBdt, X¯0 = x0. (60)

The direction of the drift corresponds to the direction that minimizes the distance between the current state X¯t and the endpoint ek. Since γt → ∞ as t → T, the bridge process converges to the endpoint ek. The convergence behavior can be analyzed by examining the radial process rtk := dg(ek,Xt), which we describe below.

Radial Process Let rtw := dg(w,Xt) for arbitrary point w ∈ Sd−1. Then the bridge process of Eq. (58) can be rewritten as follows:

rtk(ek − cosrtkX¯t) sinrtk

dX¯t = γt

dt + σtdBdt, X0 = x0, (61)

where rtk := re

t . Then the SDE of rtw can be derived using the Itô’s formula as follows:

k

rtk(ek − cosrtkX¯t) sinrtk

drtw = ∇rtw,γt

σt2 2

∆rtw dt + ∇rtw,σtdBdt , (62)

+

where ∇ and ∆ denote the Riemannian gradient and the Laplace-Beltrami operator on Sd−1, respectively. From the identities in Appendix A.1 and the fact that ⟨∇rtw,dBdt⟩ is a 1-dimensional Brownian motion ([32] Example 3.3.3), we get the following result:

drtw = −γt rtk ⟨ek,w⟩ − cosrtk cosrtw sinrtk sinrtw

(d − 1)σt2 2

cot(rtw) dt + σtdWt,

+

(63)

r0w := cos-1⟨x0,w⟩,

where Wt denotes a 1-dimensional Brownian motion. For w = el, we obtain a simplified formulation:

(d − 1)σt2 2

drtl = −γtC(rtk,rtl)rtk +

cot(rtl) dt + σtdWt, r0l =

π 2

δk,l (64)

C(rtk,rtl) =

1 if k = l −cot(rtk)cot(rtl) otherwise

. (65)

#### A.4 Diffusion Mixture Representation

We provide the statement of the diffusion mixture representation from Jo and Hwang [33], which extends Peluchetti [45] to Riemannian manifolds. We refer the readers to Jo and Hwang [33] for a detailed derivation of the diffusion mixture representation for general Riemannian manifolds. We consider Riemannian manifolds that are complete, orientable, connected, and boundaryless.

Proposition A.6. Consider a collection of SDEs on a manifold M indexed by λ ∈ Λ:

dXtλ = ηλ(Xtλ,t)dt + σλ(Xtλ,t) dBMt , X0λ ∼ p0 (66)

with marginal distribution of Xtλ denoted by pλt . Let L be a mixing distribution over Λ. Then a diffusion process on M described by the SDE:

dXt = η(Xt,t)dt + σ(Xt,t) dBMt , X0 ∼ p0 (67) η(x,t) = ηλ(x,t)

1/2

pλt (x) pt(x) L(dλ) , σ(x,t) = aλ(x,t)

pλt (x) pt(x) L(dλ)

(68)

where aλ := σλ(σλ)⊤, admits the marginal distribution pt:

##### pt(x) = pλt (x)L(dλ), p0(x) = pλ0(x)L(dλ). (69)

From the diffusion mixture representation, Jo and Hwang [33] construct the generative process as a mixture of the bridge processes on M as shown in the following proposition.

Proposition A.7. Let p0 and p1 be probability distributions on a Riemannian manifold M. Consider a collection of SDEs that describes bridge processes on M from x ∼ p0 to y ∼ p1:

dXtx,y = ηx,y(Xtx,y,t)dt + σtdBMt , X0 = x, (70)

with marginal distribution of Xx,y denoted by px,yt . Then the following SDE defines a diffusion process that transports an initial distribution p0 to a target distribution p1:

dXt = η(Xt,t)dt + σtBMt , X0 ∼ p0, (71) η(z,t) := ηx,y(z,t)

px,yt (z) pt(z)

p0(dvolx)p1(dvoly), (72)

pt(z) := px,yt (z)p0(dvolx)p1(dvoly). (73)

In the case of M = Sd−1, we derive the generative process for the reparameterized data distribution pdata(x) = dk=1 pkδ(x−ek), by mixing the logarithm bridge processes on Sd−1 (Eq. (5)).

Corollary A.8. Let pdata(x) = dk=1 pkδ(x−ek) be a data distribution on Sd−1. Then the following SDE defines a diffusion process that transports the initial point x0 ∈ Sd−1 to the distribution pdata:

d

pT|t(ek|Xt)ηk(Xt,t) dt + σtdBdt, X0 = x0, (74)

dXt =

k=1

cos-1⟨z,ek⟩(ek − ⟨z,ek⟩z) 1 − ⟨z,ek⟩2

ηk(z,t) := γt

, (75)

where pT|t(ek|Xt) represents the conditional probability that the process will reach the endpoint ek at time T, given the current state Xt at time t.

#### A.5 Mixture Paths

We derive a new family of generative processes by constructing a mixture over the time marginals of generative processes. We first present a proposition for mixing diffusion processes with a general time-dependent mixing schedule.

Proposition A.9. Consider a collection of n SDEs on a closed Riemannian manifold M:

dXti = ηi(Xti,t)dt + σi(Xti,t) dBMt , X0i ∼ p0 (76)

with marginal distribution of Xti denoted by pit. Let λi ∈ C1([0,T]) satisfy λit ≥ 0 and n i=1 λit = 1 for all t. Then there exists a diffusion process with the marginal distribution pt:

n

λitpit(x). (77)

pt(x) =

i=1

Proof. We show that there exists a scalar potential Φ : M × [0,T] → R such that the following SDE defines a diffusion process that yields the desired marginal distribution:

dXt = η(Xt,t)dt + σ(Xt,t)dBMt , (78) η(x,t) :=

n

n

pit(x) pt(x) −

pit(x) pt(x)

∇Φ(x,t) pt(x) −

- 1

- 2

λitηi(x,t)

λitai(x,t)∇

(79)

i=1

i=1

1/2

n

pit(x) pt(x)

λitai(x,t)

, (80)

σ(x,t) :=

i=1

where ai:=σi(σi)⊤. Here, we assume that ηi and σi are bounded and ai are uniformly elliptic. First, define a function f : M → R that satisfies the zero-mean condition:

n

n

n

dλit dt

dλit dt M

dλit dt

pit(x) ;

pit(x)dvolx =

f(x,t)dvolx =

= 0, (81)

f(x,t) :=

M

i=1

i=1

i=1

where we used the fact that ni=1 λit = 1 for all t. As M is closed, its Laplace–Beltrami operator is invertible on the subspace of zero-mean functions. Therefore, the Poisson equation ∆Φ(·,t) = f(·,t)

admits a weak solution Φ. From the definition of pt, we can derive the following equality:

n

n

n

∂(λitpit(x)) ∂t

∂pit(x) ∂t

dλit dt

∂pt(x) ∂t

λit

pit(x) (82)

=

=

+

i=1

i=1

i=1

n

- 1

- 2

div ai(x,t)∇pit(x) + ∆Φ(x,t) (83)

λit −div pit(x)ηi(x,t) +

=

i=1

n

n

- 1

- 2

λitpit(x)ηi(x,t) +

λitdiv ai(x,t)∇pit(x) + div(∇Φ(x,t)) (84)

= −div

i=1

i=1

n

λitpit(x)ηi(x,t) − ∇Φ(x,t)

= −div

i=1

(85)

n

λitpit(x) pt(x)

pit(x) pt(x)

- 1

- 2

div ai(x,t) ∇pt(x)

+ pt(x)λit∇

+

i=1

i tpit(x)

where we used the product rule for divergence in λitpit(x) = pt(x)λ

pt(x) .

Reordering the terms in Eq. (85), we obtain the following result: ∂pt(x) ∂t

n

n

pit(x) pt(x)

pit(Xt) pt(Xt) −

∇Φ(Xt,t) pt(Xt) −

- 1

- 2

λitηi(Xt,t)

λitai(Xt,t)∇

= −div pt(x)

i=1

i=1

n

pit(Xt) pt(Xt) ∇pt(x) , (86)

- 1

- 2

λitai(Xt,t)

div

+

i=1

which corresponds to the Fokker-Planck equation for the SDE of Eq. (78). Therefore, the diffusion process described by the SDE in Eq. (78) has a marginal distribution pt in Eq. (77).

| |
|---|

From Proposition A.9, we can derive a new family of generative processes by constructing a mixture over the time marginals of generative processes {Qi: 1 ≤ i ≤ n}:

n

n

Qmixt :=

λitQit ,

λit = 1, 0 ≤ λit ≤ 1, (87)

i=1

i=1

where λit is the time-dependent mixing schedule assigned to the i-the generative path. One example is creating a mixture path by mixing the masked diffusion and the uniform diffusion on Sd−1, as defined in Section 3.2.

Corollary A.10. Let pmaskt and punift denote the marginal distributions of the masked diffusion and the uniform diffusion on Sd−1, as defined in Section 3.2, respectively. Then there exists a diffusion process on Sd−1 whose marginal distribution at time t satisfies:

pt(x) = λtpmaskt (x) + (1 − λt)punift (x), (88) where λt ∈ [0,1] for all t ∈ [0,T].

#### A.6 Likelihood Bound

We derive the point-wise likelihood bound and the upper bound on the negative log-likelihood of our generative model, defined as the parameterized mixture process Qθ with the drift ηθ in Eq. (11).

Let Qk be a bridge process with starting point x0 and endpoint ek. From the KL divergence between Qθ and Qk, we can derive a point-wise upper bound on the negative log-likelihood using the Girsanov theorem on compact manifolds (De Bortoli et al. [16], Corollary H.3):

−log pˆθ(ek) = DKL(δ(ek)∥pˆθ(ek)) = DKL(QkT∥QθT) (89)

2

T

- 1

- 2

σt−1 ηθ(Xt,t) − ηk(Xt,t)

≤ DKL(Qk∥Qθ) = EX∼Qk

dt , (90)

0

2

where the inequality comes from the data-processing inequality. The point-wise likelihood bound leads to the upper bound on the negative likelihood of our model:

data − log pˆθ(z) ≤ Eek∼pdata

Ez∼p

X∼Qk

- 1

- 2

T

0

σt−1 ηθ(Xt,t) − ηk(Xt,t)

2

dt . (91)

2

#### A.7 Training Objective

We show that minimizing the cross-entropy-based loss defined in Eq. (15) guarantees maximizing the likelihood of our generative model defined as the parameterized mixture process in Eq. (11). We start with deriving a uniform bound for the drift of the bridge process defined in Eq. (5):

cos-1⟨z,el⟩(el − ⟨z,el⟩z) 1 − ⟨z,el⟩2 2

= γt cos-1⟨z,el⟩ ≤ πγt. (92)

ηl(z,t)

= γt

2

Then the triangle inequality gives the following:

d

l=1

pθ(x,t),el ηl(x,t) − ηk(x,t)

≤ π2γt2

d

l=1

pθ(x,t),el − δk,l

2

2

d

pθ(x,t),el − δk,l ηl(x,t) 2

(93)

≤

l=1

2

2

≤ −2π2γt2 log pθ(x,t),ek . (94)

From Eq. (94), we derive the upper bound for the maximum likelihood training objective L(θ) in Eq. (14) as follows:

  (95)

 1

2

d

T

σt−2

pθ(Xt,t),el ηl(Xt,t) − ηk(Xt,t)

L(θ) = Eek∼pdata

dt

2

X∼Qk

0

l=1

2

T

2π2γt2 σt2

log pθ(Xt,t),ek dt (96)

≤ Eek∼pdata

−

X∼Qk

0

T−ϵ

2π2γt2 σt2

≤ Eek∼pdata

−log pθ(Xt,t),ek dt

sup

X∼Qk

t∈[0,T−ϵ]

0

(97)

T

2π2γt2 σt2

+ Eek∼pdata

−

log pθ(Xt,t),ek dt

X∼Qk

T−ϵ

≤ MϵLCE(θ) + F(ϵ), (98)

where F(ϵ) denotes the last term of Eq. (97). Since X ∼Qk is the bridge process with endpoint ek, Xt converges to ek as t → T and ⟨pθ(XT−ϵ,T − ϵ),ek⟩ ≈ 1 for sufficiently small ϵ > 0. As a result, F(ϵ) ≈ 0 for sufficiently small ϵ, which lead to the following result:

L(θ) ≤ MLCE(θ), (99)

for some constant M > 0. Therefore, minimizing the cross-entropy-based loss LCE(θ) approximately guarantees maximizing the likelihood.

#### A.8 Projected Processes

Let Xt|0,T denote the mixture process {Xt}Tt=0 on Sd−1 conditioned to the endpoints X0 = x0 and XT = x1. Then Xt|0,T corresponds to a bridge process described by the following SDE:

cos-1⟨X¯t,x1⟩(x1 − ⟨X¯t,x1⟩X¯t) 1 − ⟨X¯t,x1⟩2

dX¯t = γt

dt + σtdBdt, X¯0 = x0. (100)

We can derive the projection ztT = ⟨Xt|0,T,x1⟩ using the Itô’s formula for fT(·) := ⟨·,x1⟩:

cos-1⟨X¯t,x1⟩(x1 − ⟨X¯t,x1⟩X¯t) 1 − ⟨X¯t,x1⟩2

- 1

- 2

dztT = ∇fT(X¯t),γt

σt2∆fT(X¯t) dt

+

(101)

+ σt ∇fT(X¯t),dBdt

cos-1ztT 1 − (ztT)2

(d − 1)σt2 2

= x1 − X ¯t,x1 X ¯t,γt

x1 − X ¯t,x1 X ¯t −

ztT dt

(102)

+ σt 1 − (ztT)2dWt

(d − 1)σt2 2

= γt cos-1ztT 1 − (ztT)2 −

ztT dt + σt 1 − (ztT)2dWt, (103)

where we have used the identities ∇fT(z) = x1 − ⟨z,x1⟩z,∆fT(z) = −(d − 1)⟨z,x1⟩. Note that the Laplace-Beltrami operator defined on Sd−1 has a simple and tractable form due to the radial symmetry of the hypersphere.

Similarly, zt0 = ⟨X¯t,x0⟩ can be derived using Itô’s formula for f0(z) := ⟨z,x0⟩:

cos-1ztT 1 − (ztT)2

(d − 1)σt2 2

⟨x0,x1⟩ − zt0ztT −

dzt0 = γt

zt0 dt + σt 1 − (zt0)2dWt. (104)

Masked Diffusion Since the masked bridge process has x0 = em and x1 = ek with ⟨em,ek⟩ = 0 for all non-mask token ek, the projected processes are described as the follows:

cos-1 ztk 1 − (ztk)2

(d − 1)σt2 2

dztl = γt

δl,k − ztlztk −

ztl dt + σt 1 − (ztl)2dWtl, (105)

with initial condition z0l = 0 for all l and Wtl are 1-dimensional standard Wiener processes.

√

Uniform Diffusion The uniform bridge process has x0 = di=1 ei/

d and x1 = ek, and the projected processes have a simple form:

cos-1 ztk 1 − (ztk)2

(d − 1)σt2 2

dztl = γt

Al,k − ztlztk −

ztl dt + σt 1 − (ztl)2dWtl,

√

d if l ̸= k 1 otherwise

1/

Al,k =

√

d for all l.

with initial condition z0l = 1/

(106)

- A.9 Simulation-Free Training with Radial Symmetry Here we derive the parameters of the Riemannian normal distribution from the projected processes:

(d − 1)σt2 2

dztT = γt cos-1ztT 1 − (ztT)2 −

ztT dt + σt 1 − (ztT)2 dWtT, (107)

cos-1ztT 1 − (ztT)2

(d − 1)σt2 2

zt0 dt + σt 1 − (zt0)2 dWt0, (108)

dzt0 = γt

z0T − zt0ztT −

with initial conditions z0T = ⟨X0,XT⟩ and z00 = 1. From the definition ztT := ⟨Xt|0,T,x1⟩, we establish the connection between the mean projection EztT and the parameters αt and ρt:

EztT ≈ Ez expµ

(ρtz),x1 , z ∼ NTµtSd(0,I) (109) Eq.=(28) Ez cos(ρt∥z∥)µt + sin(ρt∥z∥)

t

z ∥z∥

,x1 (110)

z ∥z∥

(111)

= Ez cos(ρt∥z∥)⟨µt,x1⟩ + Ez sin(ρt∥z∥)

,x1

=0

αt sinϕ0

αt cosϕ0 sinϕ0

Eq.=(17) Ez cos(ρt∥z∥)

x0,x1 (112)

x1 + 1 − αt2 −

= Ez cos(ρt∥z∥) sinϕ0αt + cosϕ0 1 − αt2 , (113)

for ϕ0:=cos-1⟨X0,XT⟩, where the last term in Eq. (111) is zero due to radial symmetry. Similarly,

Ezt0 ≈ Ez⟨expµ

(ρtz),x0⟩ = Ez cos(ρt∥z∥) 1 − αt2, (114)

t

Notably, we have the following identity for z ∼ NTµtSd(0,I):

ρ2t 2

d 2

- 1

- 2

2 t/2

Ez cos(ρt∥z∥) = e−ρ

,−

1f1

,

:= Fd(ρt), (115)

where 1f1 denotes the Kummer function, also known as the confluent hypergeometric function. Therefore, the parameters αt and ρt can be derived from the mean projections EztT and Ezt0:

αt =

(EztT/Ezt0 − cosϕ0)2 sin2 ϕ0 + (EztT/Ezt0 − cosϕ0)2

, ρt = F-1

d Ezt0/ 1 − αt2 . (116)

#### A.10 Comparison with Prior Work

Comparison with Discrete Diffusion Models Discrete diffusion models [2, 39, 49, 52] do not fully leverage the power of iterative refinement, which is the key to generative modeling of continuous data, for example, image synthesis [19, 48] and video generation [5, 46]. In discrete diffusion models, the progressive corruption during the forward process is modeled by stochastic jumps between states in Markov chains. Since denoising is achieved by jumping between states, discrete diffusion loses valuable signals during refinement, which limits the generative performance and controllability. In contrast, our RDLM takes a continuous approach using the geometry of the statistical manifold and the hypersphere, and therefore avoids the signal loss that occurs during state transitions in discrete diffusion models, fully leveraging iterative refinement.

Advantage of Continuous Approach Due to fully leveraging the iterative refinement, RDLM can generate higher-quality samples, outperforming discrete diffusion models across diverse domains. Furthermore, our continuous approach offers additional advantages: (1) Controllable generation: Using a continuous diffusion model enables direct application of guidance, e.g., classifier [17] and classifier-free guidance [26]. (2) Optimized design choices: Benefit from advancements in continuous diffusion, e.g., optimized noise schedule [9, 31, 34] and self-conditioning [10]. (3) Efficient sampling: Our framework supports efficient and scalable sampling strategies such as DPM-Solver [40, 41]. In contrast, discrete diffusion models are restricted to using a simple ancestral sampling strategy.

Comparison with Flow Matching Our method outperforms previous works using flow matching [12, 15] due to three key contributions: (1) generalization of discrete diffusion, (2) parameterization and training objectives, and (3) scalability to higher dimensions.

First, our method generalizes discrete diffusion models, the current state-of-the-art in language modeling, and introduces a novel mixture path process that enhances performance. In contrast, prior works using flow matching [12, 15] lack a direct connection to discrete diffusion models, resulting in a suboptimal design that leads to inferior performance. Notably, flow matching-based approaches are a special case of our method, as shown in Section 3.

Second, we introduce a novel parameterization (Eq. (10)) and cross-entropy-based training loss (Eq. (15)), similar to the loss used in discrete diffusion models. This loss optimizes the likelihood during training, and when combined with our importance sampling loss (Eq. (16), achieves a superior performance. In comparison, Cheng et al. [12] uses a simple flow matching loss that does not guarantee maximum likelihood optimization.

Lastly, prior works are restricted to small vocabularies due to the difficulty of learning a generative process on high-dimensional manifolds (i.e., large vocabulary). This issue arises from the rapid convergence problems and insufficient model capacity, as discussed in Section 4. We address these challenges with dimension splitting, which significantly improves performance and enables effective scaling to large vocabularies.

### B Experimental Details

#### B.1 Training and Sampling

We provide the pseudocode for our training and sampling schemes in Algorithm 1 and Algorithm 2, respectively. We additionally provide pseudocode for pre-computing the parameters for the Rieman-

Algorithm 3 Pre-computing parameters of Riemannian normal before training Input: Initial point u, vocabulary size d, number of simulations N, number of discretization steps K, noise schedule σt, time change coefficient γt

- 1: t ← 0 and δt ← 1/K
- 2: ψ0 ← ⟨u,e1⟩ ▷ Radial symmetry
- 3: α0 ← 0 and ρ0 ← 0
- 4: a ← (ψ0)N and b ← (1)N ▷ Initialize N independent trajectories
- 5: for k = 1 to K do
- 6: Wa,Wb ∼ N(0,I) N
- 7: σ ← σk/K and γ ← γk/K
- 8: a ← a + γ cos-1a√1 − a2 − (d−1)σ

2

2 a δt + σ√1 − a2

√

δtWa ▷ Eq. (18)

- 9: b ← b + γ cos √1−-1aa2 (ψ0 − ab) − (d−1)σ

2

2 b δt + σ√1 − b2

√

δtWb ▷ Eq. (19)

- 10: r ← MEAN(a)/MEAN(b) ▷ Ratio of mean projections
- 11: αk/K ← (r−ψ

0)2

1−ψ02+(r−ψ0)2 ▷ Eq. (20)

- 12: ρk/K ← F-1

d b/ 1 − αk/K2 ▷ Eq. (20)

- 13: end for
- 14: Return: {αi/K,ρi/K}Ki=0

nian normal αt and ρt in Algorithm 3. Note that pre-computing takes only once before training our model, and the computation time is negligible compared to the training time.

Likelihood Computation For computing the upper bound for NLL, we use the Monte Carlo estimation of the negative ELBO derived in Eq. (13). Note that we use simulated Xt, instead of approximation from the Riemannian normal, for accurate computation.

Computing resources For all experiments, we use NVIDIA RTX A5000 and H100.

#### B.2 Text Generation

Baselines We compare against state-of-the-art diffusion models. Multinomial Diffusion [29], D3PM [2], SEDD [39], MDLM [49], MD4 [52] are discrete diffusion models. Plaid [23] and Bayesian Flow Network (BFN) [21] are continuous diffusion models. We do not use existing works for flow matching on the statistical manifold [12, 15] as they do not provide likelihood computation applicable for language modeling.

We also use the transformer AR model [61] and the following autoregressive models as baselines: IAF/SCF [63], AR Argmax Flow [29], and Discrete Flow [58] are flow-based models, and ARDM [30] and MAC [53] are any-order autoregressive models.

Text8 Text8 [42] is a small character-level text modeling benchmark extracted from English Wikipedia. Following the previous works [2, 39, 49], we split the dataset into 90M/5M/5M with a fixed sequence length of 256. We use a vocabulary size of 28, comprising 26 lowercase letters, a white space token, and a mask token. We use a 12-layer diffusion transformer [44] following Lou et al. [39] with 92.4M trainable parameters. We train our model for 1M iterations with batch size 512

- as done in previous works, using the same learning rate, optimizer AdamW [38], and exponential moving average (EMA) with decay rate 0.9999.

One Billion Words One Billion Word Benchmark is a dataset extracted from the WMT 2011 News Crawl dataset comprised of single sentences from news articles. Following Sahoo et al. [49], we use the bert-base-uncased tokenizer and pad and truncate the sequences to length 128. We use a 12-layer diffusion transformer [44] with the hidden dimension of 768 and 12 attention heads, following Sahoo et al. [49] with 110M trainable parameters. We train our model for 1M iterations

Table 5: Comparison between the training objectives. We compare Bits Per Character (BPC) on the Text8 test set.

Method BPC (↓)

Drift MSE (Eq. (14)) ≤ 1.36 Cross Entropy (Eq. (15)) ≤ 1.34 Cross Entropy + Importance Sampling ≤ 1.32

Table 6: Analysis of the dimension splitting (Section 4). We compare NLL on LM1B test set. TopK Feat. denotes adding additional features of top-k indices of the input state.

Method NLL (↓) w/o dimension splitting ≤ 11996.9 w/o dimension splitting + Top-K Feat. ≤ 661.1 w/ dimension splitting ≤ 428.5

with batch size 512 as done in previous works, using the same constant learning rate, optimizer AdamW [38], and exponential moving average (EMA) with decay rate 0.9999.

Comparison with MDLM Here we provide a detailed comparison with MDLM [49] on the language modeling task using the One Billion Words dataset.

First, we did not search for optimal training hyperparameters (e.g., learning rate). Instead, we directly adopted the hyperparameters used by MDLM to ensure a fair comparison. However, because RDLM employs a continuous approach, it might benefit from different hyperparameter choices than discrete diffusion models. Due to resource limitations, we could not explore these optimized settings.

Furthermore, MDLM was trained using the low-discrepancy sampler, which is crucial for reducing the variance of the ELBO during training, leading to better perplexity results. We did not use the low-discrepancy sampler during training, yet RDLM still achieved competitive results on the LM1B dataset.

Additionally, the reported RDLM and MDLM results are based on training up to 1 million iterations,

- at which point RDLM had not yet fully converged. Extrapolating RDLM’s validation perplexity through curve fitting shows that RDLM surpasses MDLM after 10 million iterations. Due to resource limitations, we were unable to train beyond 1 million iterations.

#### B.3 Pixel-level Image Modeling

Baselines We compare against autoregressive models and diffusion models that directly model raw pixel space. PixelRNN [60], Gated PixelCNN [59], PixelCNN++ [50], PixelSNAIL [11], Image Transformer [43], and Sparse Transformer [13] are autoregressive models. D3PM [2], τLDR [6], and MD4 [52] are discrete diffusion models.

Implementation Details We represent each image as a set of discrete tokens with a vocabulary size of 256. We use the 10-layer diffusion transformer [44] for our model with 35M trainable parameters. We train 100k iterations with batch size 128 and AdamW [38] optimizer following Shi et al. [52].

#### B.4 DNA Sequence Design

The dataset contains 100k promoter DNA sequences, each paired with a transcription signal profile. Each sequence consists of 1024 base pairs centered at the annotated transcription start site position [28], and the base pair has 4 categories (ATGC) conditioned on the profile.

Baselines We compare our model against diffusion models and language models. Bit Diffusion [10] is a continuous diffusion model, D3PM [2] is a discrete diffusion model, DDSM [3] and Dirichlet Flow Matching [54] are diffusion model and flow matching model using the probability simplex, respectively. Fisher-Flow [15] is a flow matching model using statistical manifold.

Implementation Details Following the previous work [15, 54], we use the same data split of 88,470/3,933/7,497 and identical model architecture consisting of 20-layer 1-D CNN with 13.3M trainable parameters. We train our model for 100k iterations with batch size 256 and AdamW [38] optimizer. We evaluate the MSE on the generated samples conditioned on the prescription signals from the test set, using 300 generation steps following the previous work [15].

[Figure 1]

[Figure 2]

[Figure 3]

(a) Dimension 4 (a) Dimension 256 (a) Dimension 30522

Figure 2: Maximum mean discrepancy (MMD) distance between the simulated distribution p(Xt|X0,XT) and the approximated distribution. We report the results for dimensions 4, 256, and 30522.

### C Generated Samples

#### C.1 Text8

We provide uncurated text samples generated by our RDLM trained on the Text8 dataset.

o zero one british single payrock neurologically related condition is a member of the original playboys oriental pbkr cat ii a boob one card featured in the late f one zero dippie dons as it became pigus in the cir the monoseur engine shair which became th

h delivered from the new meeting the construction of modern shooting begins kinington resurrects the hark or corped a hopper nightlife subjecting to turn his attention at a joyable moment he is able to explain that he is in recovery with a new orleans baby

wilder unrefreshed bup of lightmarks was pertified only at the head of sinar joseph avaret in the cetleben key in one nine nine seven this report has been portrayed as a shrinking feathor of the civil directs against urban rumour as that he was ana eichy

s seven two chromosomes regainally regular and contain number of mignain gnaning pros zopods or cells whose podic configuration divided agong the faces of dna generally replaced by b as therus group are non mit and elanisten special cayits regularly are ca

nine four although portrayals of frel appearance the novel include leaked to bratally targeted audiences largely by steve roper dart mer upick and j pernan s durk born one nine four zero s but stillly not they are created the western master and mag both m

idment indicates two different types drop tales have different charges which train structures having rare and light weight variations have lower weight impedients such as chawings starges and groove gloves shorter holes can be jumpliten don badld a horse i

d deliberately rejected this a different post however saw al sh ibn misha rody was revealed to be the lord curses of jesus one nine one nine he handled his journey to its historical map of the egyptians and was still nodged as he committed to reproete he a

ovincial governors regelrant a cursami governor granted to a spanish cominic in one seven eight three mateo s teltacheutes lebmo alexius jeano and pan dosien dostre of a ruguen de cosst originating specifically the treaty of st louis the extinctions remain

#### C.2 One Billion Words

We provide uncurated text samples generated by our RDLM trained on the LM1B dataset.

[CLS] social recklessly the obvious support 2013. [CLS] they were elected off by the english authorities, whose party subsequently named as principal when lawrence tang had to hold the property until they were turned to down their heads in the back - sky of which sank from matthews’s doorstep. [CLS] it has been pouring gladly with work and along the motorway, where certified sales will follow a new bone in the next several days to avoid commercial production problems, according to recommendations from both workplace and tropical mod. [CLS] he said he plans watchsty will b greens the old draft plunging sara, but have medics announced she would make you the taxpayer? [CLS] duchess [CLS]

[CLS] of lieberman. [CLS] analysts say since 5, 000 people have held a established council in 120 forums and levels, some have returned to the villages of the british capital, mideast and sprint. [CLS] his friends ring between ironing his body they forbid forrest. [CLS] seven babies missing and 27 french subcontinent and two development employees suffered injuries in a securing of greece, a spokeswoman said immediately, while tneye wedang. [CLS] both questions has already been considered. [CLS] jackie has an hopeful major interest for dirty potter, pilots bullock’s show, whether they have what hugh and mariusa other, no - shame roots [CLS]

[CLS] is the problem that worth most of a marriage to have a single car he doesn’t need. [CLS] mr obama will carry out more casualties however than president obama’s followers, and it mild to form the first cumulative current division ofers holding the guantanamo men that arches to injustice. [CLS] phillips said : " designer kaia kangaroo, 27, and herself rubbed jim reyes, the general patron of france light, have organized a building aimed at gunning film houses. [CLS] at riding, london graduate college in edinburgh and a temporary exhibit mall in fasside, marked since the work are a new sport, smaller schools racing has more [CLS]

[CLS]aceous that in spain had submitted one time the main website on mass wireless, in carpcsllo. [CLS] not two of the beer bk known in the companies could have thousand stretch men - - ginger, and showed vulnerable cases, leaving you in the same £200m standard. [CLS] yet apius is accepted quickly to associate in the months since - - bulletin energy americas - they agreed that it was getting waste into ulysses air before creation known as the bulletinsburg, which can be bowed with bracelet growth by speed. [CLS] rely will get another less energetic first - turn victory. [CLS] more than 2, 000 people arrived, out [CLS]

[CLS] more steadily increasing transit facilities with murray’s tax breaks. [CLS] nonero moee enjoyed terrestrial wallino with the immoitunghrck in most years. [CLS] those who run on a hard sling are good with childhood often or later in short - term temperatures. [CLS] top - seeded henin is shark seventh and isatin out in stanford. [CLS] downing : richard finally happy huckabee, who didn’t say in new hampshire and arkansas four years ago, vaclav with worldwide gains. [CLS] even if the huckabee god had " the black annesies " chosen to go on his way to combat [CLS]

[CLS] high school, was potya’s poker high - george she - former congressional class - flicked was a prosecutor. [CLS] coln has won the services of the sub - area tustiw university, near fort dodge, pa. [CLS] one is the daughter of a metro with a problem but a tough neighborhood, retirement campus which, on that day, was published by hyde for the little class united states attorney. [CLS] let’s sell a floral parachute in civil court on a lutheran case. [CLS] the virginia government says the ad, which

will add its new poll kind wednesday, had 10, drastically supervisors and 25 people. [CLS] [CLS]

[CLS] a memorandum posted to the university : model google, which makes the copies to sell patients seem off a significant stake in every final

- ep you programmes similar. [CLS] almost no day cbees will homemadei. [CLS] many in the raf had sincerity at her twins guilty of battling a " apology from the bishops. " [CLS] the courts have replayled their option for’welcome when the fed tends its view of the aec investors’chance. [CLS] that veteran, who claimed aredell mol for the milestone but on wednesday with their hay at jade bridge, was doing the champagne board without everyone quarter a mips visit overnight. [CLS]

[CLS] the bbc’s george washington is the first of 15, 000 people to put the calraircer range. [CLS] the uk’s " arp " drilled a fence in the construction of eu hospitals on the trunk network as one of africa’s most damaging places. [CLS] all looked after world over just um occasionallytau, which takes place victorious for schizophrenia consumed near the doc centre. [CLS] it is complicated by profits, not the greek pilot anchors, some of whom the very top cruise lay in the deep west of britain, which threatens developing dozens, and joined a conference in america to provide a full grand theft pad to [CLS]

### D Limitations and Broader Impacts

Limitations While our approach has shown promising results on language modeling tasks and other modalities, a performance gap remains in some tasks compared to autoregressive models. We hypothesize that this is because autoregressive models utilize model capacity more efficiently, as they learn from a single, fixed ordering of tokens. One interesting direction for future work is to design a position-dependent noise scheduler that converges sequentially from left to right, mimicking the autoregressive generation process. In addition, although the current framework can generate sequences up to a predefined maximum length, it is not capable of producing sequences beyond this limit. This limitation could potentially be addressed by incorporating a semi-autoregressive approach that generates text in a block-wise fashion.

Broader Impacts Our work may provide future directions for multimodal generative models that are capable of generating data from multiple domains, for example, text, images, and videos, simultaneously. Furthermore, our continuous approach may allow better controllability and improved quality with advanced sampling strategies. However, there is a risk that someone could misuse our framework to produce harmful content.

