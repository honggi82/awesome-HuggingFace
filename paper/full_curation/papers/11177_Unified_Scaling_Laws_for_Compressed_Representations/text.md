# arXiv:2506.01863v1[cs.LG]2Jun2025

## Unified Scaling Laws for Compressed Representations

Andrei Panferov∗

ISTA

Alexandra Volkova∗

ISTA

Ionut-Vlad Modoranu ISTA

Vage Egiazarian ISTA

Mher Safaryan ISTA

Dan Alistarh† ISTA & Red Hat AI

### Abstract

Scaling laws have shaped recent advances in machine learning by enabling predictable scaling of model performance based on model size, computation, and data volume. Concurrently, the rise in computational cost for AI has motivated model compression techniques, notably quantization and sparsification, which have emerged to mitigate the steep computational demands associated with large-scale training and inference. This paper investigates the interplay between scaling laws and compression formats, exploring whether a unified scaling framework can accurately predict model performance when training occurs over various compressed representations, such as sparse, scalar-quantized, sparse-quantized or even vectorquantized formats. Our key contributions include validating a general scaling law formulation and showing that it is applicable both individually but also composably across compression types. Based on this, our main finding is demonstrating both theoretically and empirically that there exists a simple “capacity” metric—based on the representation’s ability to fit random Gaussian data—which can robustly predict parameter efficiency across multiple compressed representations. On the practical side, we extend our formulation to directly compare the accuracy potential of different compressed formats, and to derive better algorithms for training over sparse-quantized formats.

### 1 Introduction

A key recent advance in machine learning has been the idea of predictable scaling of learning performance with respect to model, computation and data sizes. This approach is encompassed by the scaling laws Kaplan et al. [17], which allow researchers to predict the values of these three parameters required to reach a certain model performance. This powerful idea has been expanded upon in several directions, e.g. [15; 5; 21], and is a key ingredient behind the massive expansion of computational power for AI [12].

A parallel research direction, motivated by this massive increase in computational cost, has been model compression, which proposes a series of techniques to reduce the computational and memory footprint of model inference and training, via techniques such as sparsification [14] and quantization [11]. In this paper, we focus on the interplay between scaling laws and the degree of compression of the representation over which learning occurs. While there is significant emerging work in this direction, e.g. [9; 19; 30; 29], current scaling laws are specialized to single representations (e.g., quantization or sparsity) and/or formats (e.g., integer quantization), and cannot yet address the question of predicting model scaling behavior when training over general compressed representations.

Contributions. This paper is structured two main questions, and their practical ramifications:

∗Equal contribution. †Correspondence to: dan.alistarh@ist.ac.at.

- Q1: Is there a unified compression scaling law? First, we wish to find a single general law that not only applies to sparse [9] or quantized [19] representations in isolation, but that also provides a good fit for hybrid formats, such as sparse-and-quantized weights, or compound compression, i.e. sparse weights and activations. Through extensive experimentation, we identify this law to be of the form

Loss(N,D) ∼ A · (N · ρ(R))−α + B · D−β + E, (1)

where N is the number of model parameters, D is the dataset size, E is the irreducible error, A, B, α and β are constants, and ρ is a parametric function of the representation R. Crucially, we find that, even for very complex representations—e.g. 3-bit quantization with group size 32 and 1% outliers in full-precision—the parametric function ρ can still predict the scaling of model performance w.r.t. the parameter count N. We call ρ(R) the representation capacity of R. Consequently, there is always a “dense equivalent” parameter count N′ = N · ρ(R) which would yield the same loss during training. The capacity ρ(R) lies naturally in the interval (0,1], and the key goal of compression is to maximize the trade-off between model accuracy and the size and computational cost of the representation.

- Q2: Is capacity an “intrinsic” property of the representation? While related forms of the above law have been proposed in prior work [10; 19], we are the first to show that capacity is an intrinsic property of the representation, independent of the model and task for which the scaling law is obtained, but relatable to standard information-theoretic measures. Moreover, we establish the applicability of the law across hybrid (e.g. sparse-quantized weights) or composite (e.g. quantized weights-and-activations) representations.

More precisely, our main finding is that capacity is tightly-correlated with the representation’s ability to fit random Gaussian data, measured in terms of minimal mean-squared error (MSE). Concretely, ρ(R) is a simple parametric function of the MSE of the representation R when fitting random Gaussian data, i.e. ρ(R) = ρ˜(MSE(R)), where instances of the same representation R, e.g. 3 and 4-bit integer quantization, share the same parametric form ρ˜. This finding, which we validate across quantized, sparse, quantized-sparse, and even vector-quantized representations, provides a simple metric to “rank” different formats implementing the same representation. In addition, this also allows us to determine the “optimal” capacity at a certain bit-width, which is given by theoretical bounds on Gaussian fitting for a given support, which can be easily estimated via Monte Carlo algorithms. In addition, we also provide a non-trivial theoretical justification for this relationship in Theorem 1, for Adam-optimized compressed models: we relate the convergence of Adam over compressed representations with the product between the number of parameters N and the average root mean-squared error of compression across optimization, which connects to our notion of capacity.

Our second finding is that, except for pathological cases, capacity factorizes across composite representations: concretely, the capacity of a 4-bit and 2:4 sparse model is the product between the capacity of the 4-bit dense model, and that of a 2:4-sparse but unquantized model. Factorization allows us to evaluate the capacity of complex representations based on simple ones, and also holds when compressing different model representations, e.g. both weights and activations.

Practical Implications. The analytical metrics suggested by representation capacity also have non-trivial practical applications. First, the fact that we are able to relate the predictive parameter ρ to intrinsic properties of the underlying representation gives us the ability to analytically predict the representational power of different compressed numerical formats. This way, we can accurately compare and predict the efficacy of various formats such as Floating-Point, Integer (INT with and without grouping), or sparse-quantized formats (2:4 + INT) at different compression budgets. Second, this framework inspires an improved approach for sparse training, which we show provides significant improvements (above 20% in some sparsity regimes) in capacity at the same number of parameters.

Overall, our results provide a new lens to view the scaling properties of compressed models, with respect to intrinsic properties of the representation over which training is performed. Thus, we believe that capacity-aware scaling has the potential to become a practical design principle for the next generation of efficient foundation models.

### 2 Preliminaries

Scaling Laws. We start from the “Chinchilla” scaling law formulation [15] that proposed to model loss scaling as a function of the number of parameters in the model N and the number of data points D the model was trained on, in the form the parametric function:

Loss(N,D) = AN−α + BD−β + E, (2)

Table 1: Representation scaling laws (rows) versus the quantities of interest (columns). For all laws, N represents the number of parameters, D is the data, and E is the irreducible error. For the sparsity scaling law of Frantar et al. [8], S is the sparsity and the lowercase parameters are learnable constants. For the precision scaling law of Kumar et al. [19], Pw is the weight precision, and γP is a learnable weight sensitivity parameter. For the law of Frantar et al. [10], effC is the “effective parameter multiplier,” that is explicitly fitted for every instance of compression C. By contrast, our formulation postulates that the parameter efficiency is a simple parametric function of the representation’s capacity to fit random Gaussian data (GMSE(R)).

Sparsity fit Quantization fit (Error) (Error)

Parametrization Formulation for Loss(N, D)

|Sparsity S<br><br>Frantar et al. [9]|aS(1 − S)bS + cS NbN<br><br>+<br><br>aD D<br><br>bD<br><br>+ E<br><br>|5.7 · 10−4<br><br>|N/A|
|---|---|---|---|
|Quantization to Pw<br><br>Kumar et al. [19]<br><br>|A N(1 − e−Pw/γw) −α + BD−β + E<br><br>|N/A<br><br>|4.5 · 10−3|
|Compression C<br><br>Frantar et al. [10]|A (N · effC)α<br><br>+<br><br>B Dβ<br><br>+ E<br><br>|4.2 · 10−4|1.9 · 10−3|
|Representation R<br><br>(OURS)<br><br>|A (N · ρ(GMSE(R)))α<br><br>+<br><br>B Dβ<br><br>+ E|4.7 · 10−4<br><br>|2.1 · 10−3|

where A, B, E, α, and β are the scaling law parameters that can be fit empirically. It is important to note that such scaling laws assume an ideal, well-tuned training setup, and that the parameter may vary slightly depending on architecture, optimizer, and hyper-parameters.

Compressed Representations. For sparsity, we assume that a specific fraction, within each parameter group of a certain size G, is set to zero. Sparsity is unstructured if the group is the whole tensor, whereas it is semi-structured (N:M) if N parameters out of every M are set to zero. For quantization, unless otherwise stated, we assume that parameters are mapped onto a scalar, symmetric grid corresponding to the number of bits available for quantization, as is standard [11]. (We will also consider vector quantization in Section 4.1.) For sparse-quantized representations, we follow [13] by first applying sparsification, and then quantization, to map continuous parameters onto this format.

Prior Scaling Laws. The relationship between the learning representation and the the scaling law formulation was considered by Frantar et al. [9] for sparsity, and by Kumar et al. [19] for quantization. The scaling laws they propose are described in Table 1, together with their parametrization, for the special case of weight-only compression. While both these laws can predict loss with respect to training over their target representations, their formulation is not designed to generalize to other representations, or to hybrid ones (e.g. sparse-quantized).

The unified law we consider extends preliminary work by Frantar et al. [10], who, assuming that training happens over weights compressed in representation C, proposed a simple parametric law similar to Equation 1, but which is fitted independently for each instance of compressed training, yielding a value of the corresponding parameter efficiency factor, called effC. Frantar et al. [10] focuses on quantization; they fit sparsity in limited experiments, and do not consider hybrid formats.

Our Approach. By contrast, our focus is on relating parameter efficiency to intrinsic properties of the representation R: in their parlance, we show that, across all instances of a given compressed representation R, e.g. uniform integer (INT) quantization, the parameter efficiency has the same parametric form ρ(R), and, in fact, this parametric form is simply a function of the MSE for the representation R w.r.t. random Gaussian data, i.e. ρ(R) = ρ˜(GMSE(R)). Importantly, GMSE(R) is an intrinsic property of R, and only depends on its own parametrization: for INT, this would be the number of bits we employ per parameter.

The fact that this parametric form is shared across instances of the same representation (Section 4.1), is powerful since it allows us to compare and transfer parameters between instances of the same representation R. Clearly, if GMSE ≃ 0, then ρ(P) ≃ 1, and we recover the original “dense” scaling law [15]. Interestingly, Table 1 shows that our unified law can provide a better fit than the representation-specific formulations of Frantar et al. [9] and Kumar et al. [19], and (almost) matches the formulation of [10], which is fitted for each compression instantiation C.

Setting for Experimental Validation. For our scaling law investigations, we pretrained decoder-only Transformers following the Llama architecture [31] for 30M, 50M, 100M and 200M non-embedding parameters. Models were trained on the C4 dataset [26], using the Llama-2 tokenizer [31]. To ensure we operate in a data-rich regime, we use 100 training tokens per model parameter, and train on fixedlength context windows of 512 tokens. We used AdamW [22] with a 0.1 ratio of warm-up epochs with cosine scheduler. Our experimental setup is very similar to that of Frantar et al. [9]; Kumar et al. [19]; Frantar et al. [10].

We follow standard quantization-aware training (QAT) methods, combined with various levels of unstructured weight sparsity. For quantization we employ the gradient estimator of [25], a per-layer uniform quantizer with static scaling factors and gradient masking. Quantization levels range from 1-bit to 8-bit precision. We consider configurations with quantized weights only, activations only, and both simultaneously. For sparsity, we apply unstructured magnitude pruning via top-k thresholding on a per-layer basis. The sparsity mask is recomputed dynamically at each optimization step. For Vector Quantization (VQ), we follow QuEST scalar quantization and apply it to 2- and 4-dimensional HIGGS grids [23]. To restrain outliers we use the trust estimation method [25] that zeros out gradients for any point lying outside a hypersphere of a certain radius.

### 3 Theoretical Analysis

One key focus of our work is whether, given a compressed representation R over which learning is performed, we can identify a predictive metric that correlates with the representation’s efficiency ρ(R). To identify this metric, we first model the standard weight-compressed LLM optimization process, which combines the Adam algorithm [18] with the straight-through estimator (STE) [2]. We have:

STE Gradient: gt = ∇f( θt), where θt = C(θt), Optimizer states: mt = β1mt−1 + (1 − β1)gt

vt = β2vt−1 + (1 − β2)gt2, vt = max(vt, vt−1) Model update: θt+1 = θt − η

mt √ vt + ϵ

,

where ∇ represents stochastic mini-batch gradient operator, C: RN → RN is the compression scheme, β1,β2 ∈ (0,1) are momentum parameters, ϵ > 0 is a small constant for numerical stability and η > 0 is the learning rate or the step-size. All vector operations are element-wise, including the max operation. Our analysis relies on the following assumptions, which are standard in adaptive optimization [32; 27; 4; 20; 6; 24; 28]:

- Assumption 1 (Lower bound and smoothness). The loss function f : RN → R is lower bounded by some f∗ ∈ R and is L-smooth, namely, ∥∇f(θ) − ∇f(θ′)∥2 ≤ L∥θ − θ′∥2, for any θ,θ′ ∈ RN.
- Assumption 2 (Unbiased and bounded stochastic gradient). For all iterates t ≥ 1, the stochastic

gradient gt at θt is unbiased, E[gt] = ∇f( θt), and uniformly bounded, ∥gt∥∞ ≤ G∞, by some constant G∞ ≥ 0.

- Assumption 3 (Bounded variance). For all iterates t ≥ 1, the variance of the stochastic gradient gt at θt is uniformly bounded by some constant σ2 ≥ 0, namely E[∥gt − ∇f( θt)∥22] ≤ σ2.

In this context, our main claim is the following: Theorem 1 (Non-convex convergence analysis). Let Assumptions 1, 2 and 3 hold. Then, choosing step-size η = min(η0, √1T ) with η0 = ϵ(1−β

1) 2LC

N and C = 2 G2∞ + ϵ/N, a randomly chosen compressed iterate θ from { θ1,..., θT} satisfies

√

√

T

N3/

Lσ2 ϵ

CLG∞ √ϵ ·E

1 T

C

N √

2

f(θ1) − f∗ +

E[∥∇f( θ)∥2] ≤

∥ θt − θt∥2 ·N+

+O

.

T

T

t=1

Discussion. Similar convergence analysis for Adam under compressed iterates θt was performed in the setup of convex online learning with bounded domain condition [16], and in nonconvex optimization with restricted hyperparameter choices and slower rate in terms of constants and extra log-terms [3]. We now interpret this bound, whose complete proof can be found in the Supplementary.

1.0

QuEST

LSQ

0.8

- 0.6

###### ()R

0.4

Decoupled fit

Kumar et al. fit

0.2

Our Fit

0.0

0 2 4 6 8

Bitwidth

(a)

()R

Noise Injection

QuEST

10 4 10 3 10 2 10 1 100

GMSE

(b)

()R

Noise Injection W+A

QuEST W+A

10 4 10 3 10 2 10 1 100

GMSE

(c)

Figure 1: Comparison of ρ fits for scaling law forms from Table 1: (a, left) shows quantizations scaling laws, (b, middle) and (c, right) demonstrate the match between noise injection and QuEST quantization for weight-only and weights+activations quantization.

Specifically, the bound shows the ergodic convergence of the gradients taken at compressed iterates, which is the strongest property shown even in the uncompressed case [32]. In turn, this term is bounded by 3 terms on the RHS. The second and third bounding terms are standard in the analysis of uncompressed Adam [32], showing that our analysis is fairly tight. We focus our attention on the first term, whose key part is highlighted in blue: this term consists of absolute constants, multiplying the

critical term N ·E T 1 Tt=1 ∥ θt − θt∥2 . Specifically, this term consists of the number of parameters N times the average ℓ2 compression error over the model parameters throughout the execution. Thus, this analysis suggests that the parameter efficiency of such models may depend multiplicatively on both the number of parameters, and the compression root-mean-square-error (RMSE) throughout the execution. Importantly, this bound is independent of the compression type.

### 4 Findings

#### 4.1 Finding 1: Gaussian RMSE Predicts Representation Capacity

Table 1 presents a number of scaling laws that model the same functions via different parametrizations. One can notice, that both the Sparsity form of Frantar et al. [9] and the Quantization form of Kumar et al. [19] can be reduced to the Decoupled form of Frantar et al. [10] in the third row, by imposing additional constraints (e.g. effP = 1−e−P

w/γw for quantization). Naturally, the Decoupled form can achieve lower fit error, but it does not provide any information about the interpretation of the capacity term, which we call ρ(R), across different representations R. The Sparsity form and the Quantization form, on the other hand, feature intertwining and interpretable parameters. For simplicity, we first focus on the Quantization form for now.

The Functional Form. Kumar et al. [19] choose the functional form ρ(Pw) = 1 − e−P

w/γw to model quantization efficiency. By contrast, we propose a different form to model ρ(R):

ρ(GMSE(R)) = L · tanh(F · log1/4(GMSE(R)))C, (3)

which depends only on the representation R’s Gaussian-MSE fit, denoted by GMSE(R), and on the scalars L, F, and C, detailed below. The GMSE(R) is easily computable for any representation, and allows us to bypass the dependency on representation-specific parametrization, such as bit-width or sparsity. Specifically, we fit the scalar parameters for each compression type, e.g. scalar quantization, and then re-use these parameters while varying GMSE(R) w.r.t. compression parameters, e.g. bit-width. The scalar parameters L, F, and C allow us to accurately model observed effects such as:

- • Imperfect convergence in high-precision: While modern QAT algorithms such as QuEST reach efficiency ρ = 1 for low quantization error, older algorithms such as LSQ [7] (Figure 1 (a)), have an efficiency limit strictly below 1, since for instance its gradient estimator introduces consistent bias. The factor L, defaulting to 1 for saturating representations, allows us to model this imperfection.
- • Various low-precision curvature: As seen in Figures 1 (b) and (c), different representations behave differently around GMSE = 1, with some have noticeably higher curvature (“breakdown”). From Figure 1 (a), one can see how that region disproportionally affects the law of Kumar et al.

[19], leading to a very poor fit at higher bitwidths. The factor C, closer to 1 for representations “more linear” around GMSE = 1, allows us to more accurately model ρ(R).

Quality of Fit. Table 1 shows that our approach leads similar or better quality-of-fit relative to prior laws, covering both scalar quantization and sparsity, while Figure 1 shows ρ(R) alignment between scaling law forms, compared to Kumar et al. [19], for the QuEST and LSQ quantizers. Again, our approach provides significantly better fit. In Figure 2(a), we show that our method can also provide a good fit for models trained with vector-quantized (VQ) weights, using the projection method of [23], for lattice dimensions 2 and 4. This shows both the versatility of our approach, and the necessity of the L term, since higher-dimensional VQ appears to have clear sub-unit saturation due to higher bias. We provide further examples in Section 4.3.

#### 4.2 Finding 2: Noise Injection as a Scaling Law Predictor

Next, we turn our scaling law on its head, and ask: what if we plug the optimal achievable GMSE for a certain bit-width into the scaling law? In that case, the scaling law should allow us to compute a lower bound on the achievable parameter efficiency given a certain type of representation. In turn, we can find out how close existing quantization- or sparsity-aware training techniques, or numerical formats, are to the information-theoretic lower bound for that specific representation.

- Figure 1 (b) illustrates the “optimality gap” for the QuEST algorithm for scalar weight-only quantization across bit-widths, suggesting that this approach is fairly close to optimal. In Figure 1 (c), we compare the fit between actual runs of this QAT algorithm across bit-widths, and the predicted values via noise injection [1] (plugging in the equivalent GMSE) into the scaling law, showing a near-perfect fit.

#### 4.3 Finding 3: Representation Capacity Is Multiplicative Across Compression Types

In prior work, [19] have claimed that, for their formulation of the law, the representation capacity factorizes independently for quantization of weights and activations. Our experimental findings extend this result, showing that representation capacity, ρ(R), also factorizes naturally across a wide range of compression approaches, whether for the same tensor (sparse-and-quantized weights) or for different state tensors (sparse weights and sparse activations). We follow the experimental setup from Section 2, training models with sparse weights and activations, or sparse-quantized weights, or both. We fit a scaling law in the 100 toks/param regime. We show that representation capacity factorizes for the following scenarios:

- 1. Sparse weights and activations: For sparsity, independently applied to weight and activations,

ρ(Rs

w,sa) = ρ(Rs

w

) · ρ(Rs

a

). (4)

We summarize the fitted values of ρ(R) levels in a matrix M (Figure 2(b)), where each entry corresponds to the fitted efficiency for a model trained with a specific sparsity configuration. Remarkably, the matrix can be accurately approximated by a rank-1 outer product of the first column M0,: (weight-only) and the top row M:,0 (activations-only) elements, i.e. M ≈ M0,: ⊗ M0,:. The resulting parameter efficiencies closely match the product of efficiencies obtained for runs with weight-only and activations-only configurations.

- 2. Sparse and quantized weights: Given a weight sparsity level s combined with q-bit quantization, we claim that the representation capacity can be represented as the product: ρ(Rq,s) = ρ(Rq) · ρ(Rs). We report the results for different sparsity levels and bit width in Figure 9. Similarly, the matrix ρ(R) factorizes into the outer product of marginal vectors for quantizationonly and sparsity-only representation. Apart from extreme quantization to 2-bit precision, the approximation maintains the error of order of 10−2.
- 3. Sparse and quantized weights, and quantized activations: Finally, we observe that factorization extends to quantization of activations as well. In supplementary experiments, we apply quantization to activation tensors alongside weight sparsity and quantization. Our results indicate that the representation capacity with weight sparsity sw and quantization bitwidth qw, and

activation sparsity qa follows ρ(Rq

).

) · ρ(Rs

) · ρ(Rq

w,sw,qa) = ρ(Rq

w

w

a

The multiplicative property of the law aligns with the concept of representation capacity. When combining compression types R1 and R2, the overall capacity naturally satisfies ρ(R1,R2) =

1.0

0.8

0.6

###### ()R

0.4

dim=2 dim=4 Scalar

0.2

0.0

10 3 10 2 10 1 100

MSE

(a)

True (P)

Approximated (P)

Absolute error

[Figure 1]

[Figure 2]

[Figure 3]

1.00 0.76 0.50 0.32 0.19

1.00 0.76 0.50 0.32 0.19

-- -- -- -- --

0.0

0.0

0.0

WeightSparsity

WeightSparsity

WeightSparsity

0.84 0.64 0.42 0.27 0.17

0.84 0.63 0.42 0.27 0.16

- -- 0.00 0.01 0.01 0.01

- -- 0.00 0.01 0.01 0.03
- -- 0.00 0.02 0.01 0.03

- -- 0.01 0.02 0.03 0.03

0.2

0.2

0.2

0.65 0.49 0.34 0.22 0.15

0.65 0.49 0.33 0.21 0.12

0.4

0.4

0.4

0.47 0.36 0.25 0.16 0.12

0.47 0.36 0.24 0.15 0.09

0.6

0.6

0.6

0.27 0.21 0.16 0.12 0.08

0.27 0.21 0.14 0.09 0.05

0.8

0.8

0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

Activation Sparsity

Activation Sparsity

Activation Sparsity

(b)

- Figure 2: (a) Scaling law for 2- and 4-dimensional vector quantization. (b) Representation capacity across weight and activation sparsity levels: baseline, factorized prediction, and relative errors. Note the low errors for the factorized predictions, with slight increases at the larger sparsity levels.

4 bits 6 bits 8 bits

10 5

10 3

GMSE

E2M1 INT4 E4M1

E3M2

E2M3 INT6

E6M1

E5M2

E4M3

E3M4

E2M5

INT8

GMSE, C4 Val Loss by Floating Point Configuration

Compression lower bound

| |E2M1|INT4|E4M1| |E6M1| |
|---|---|---|---|---|---|---|
| | | |E3M2| |E5M2| |
| | | | |E2M3<br><br>INT6|E4M3|E3M4 E2M5 INT8|
| | | | | | | |

4 bits 6 bits 8 bits

3.20

3.22

3.24

3.26

C4ValLoss

10 4 10 3 10 2

GMSE

0.75

0.80

0.85

0.90

0.95

1.00

()R

FP4 variants FP6 variants FP8 variants Noise Injection

- Figure 3: Comparison of floating point and integer data-types in terms of GMSE, and C4 Validation Loss when trained using the corresponding formats via QuEST, and the resulting capacity ρ(R). Observe the high correlation between ranking in terms of GMSE (top), and Val. Loss (bottom).

ρ(R1) · ρ(R2) ≤ ρ(Ri) ≤ 1 for i ∈ {1,2}, reflecting that applying compression methods jointly should further reduce the representation capacity compared to each method applied individually. This result allows for low-cost comparison across compression comparison. Moreover, it facilitates compression hyperparameter tuning and thus predictable model training in a compressed regime.

10 2 10 1

MSE

0.3

0.4

0.5

0.6

0.7

0.8

0.9

()P

Bits:

- 1

- 2

- 3

- 4

- 5 Groups:

4 8

| |
|---|

16 32 64 128

10 4 10 3 10 2 10 1

MSE

0.2

0.4

0.6

0.8

1.0

()R

Outliers:

0%

5.0%

- Figure 4: Representation capacity ρ(R) versus MSE for (a) group-wise quantization, with markers indicate group counts (color encodes quantization bitwidth), and (b) outlier-aware quantization.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

(a) (b) (c) (d)

- Figure 5: Backward heuristics: (a) forward mask MFW determined by the Tk threshold, (b) backward mask determined by threshold Tp, (c) banded backward mask determined by both Tk and Tp, (d) capacity plot compared to the Magnitude Pruning baseline with MBW = MFW.

### 5 Applications

#### 5.1 Application 1: Comparing Compressed Numerical Formats

Practical Formats. The scaling law for compressed representations enables systematic comparison of numerical formats used in quantization, such as INT8, INT4, FP4, or custom low-precision representations, based just on their GMSE, which can be determined via fast Monte Carlo algorithms. Thus, it provides a clear guidance on which low-precision format delivers the best performance for given resource constraints. Figure 3 illustrates this for a number of floating-point and integer data-types. Specifically, we observe a direct correlation between the ranking of GMSE values (top) and the actual C4 validation loss obtained in actual experiments (bottom). This suggests that our GMSE metric is an accurate predictor of compressed pre-training performance. For instance, it suggests that switching to FP4 (E2M1) will not bring gains relative to INT4 training, and that both formats are close to the theoretical lower bound at 4 bits.

The Impact of Parameter Grouping and Outlier Preservation. A related question regarding formats is whether more complex approaches, such as group-wise quantization, or outlier preservation in higher precision, can disrupt the scaling law. We examine this in Figure 4, which shows that preserving no outliers (0 %) lies on the Pareto-optimal boundary: higher outlier ratios achieve a worse trade-off between the MSE and the representation capacity ρ(R). This suggests that, for pre-training it is more effective to allocate bits to encoding the values distribution rather than outlier preservation or careful grouping. This further demonstrates that the proposed RMSE dependency is a general property and remains valid even under diverse structured compression techniques.

Compositionality. An immediate practical application of the multiplicative behavior of the law (Section 4.3) is the ability to estimate the model’s performance in advance for arbitrary compression configuration. Given the individual efficiencies of different compression methods, such as quantization or sparsity, applied to weights or activations, one can predict the combined effect without spending additional compute for training.

#### 5.2 Application 2: Increasing the Capacity of Sparse Training

The Sparse Training Problem. For our second application, we investigate implications of the RMSE law to maximize the capacity of a sparse representation during training. Specifically, standard sparse training methods such as gradual magnitude pruning (GMP) [33; 9] compute a forward sparsity mask, which we denote by MFW during the forward pass based on the absolute-magnitude Top-K operation applied to the model parameters θ with respect to a target sparsity. Then, a gradient ∇L(TopK(θ)) is taken w.r.t. the the sparsified weights. Standard baselines, such as the ones we use for sparse training, re-use the forward sparsity mask for the backward, preventing the pruned weights from receiving any gradient. We are interested in heuristics to improve the parameter efficiency of this standard approach, increasing capacity at the same sparsity level.

RMSE-Banded Gradient Masking. For this, we follow the RMSE law and align the parameters θ ∈ RN with the standard normal distribution by dividing θ by its root mean square RMS(θ) =

N i=1 θi2, which results in ||θ/RMS(θ)||22 = N. We allow the user to provide a median

1 N

deviation parameter p ∈ (0,0.5), which determines the threshold for the backward mask Tp = RMS(θ) · ppf(0.5 + p), where ppf is the inverse cumulative distribution function of the standard

Normal distribution. The multiplication with RMS(x) “converts” the threshold for the standard Normal distribution to the threshold for the vector θ. As a result, MBW = |θ| > Tp.

Effectively, our approach, which we call RMSE-Banded Backward Masking (RBBM), sets a backward mask MBW that may be different than the TopK mask for the forward, whose sparsity is controllable via the parameter p. To address the fact that it may not allow gradient flow for small parameter values, we allow gradients to flow for the smallest and largest parameters, and create a band between Tp and the TopK threshold Tk, where we do not allow gradient flow. Let m = min(Tp,Tk) and M = max(Tp,Tk) and define MBW = (|θ| < m) ∨ (|θ| > M). Since we do not control the relationship between Tp and Tk, we need to ensure that the band is defined correctly. Concretely, the values θi < m and values θi > M will receive gradients, while the values θi ∈ [m,M] will not.

To illustrate, in Figure 5a we show the structure of forward mask MFW, were the red region corresponds to values |θi| < Tk that will not receive any gradient, while the green region corresponds to the values |θi| ≥ Tk which will receive gradient. The top-k threshold Tk is fixed. In Figure 5b, we have a similar behavior for the backward mask MBW determined by the threshold Tp, which is now user-controlled via the median deviation parameter p. In Figure 5c we show an example for Tp < Tk, where we obtain a banded-mask: the values |θi| ∈ [Tp,Tk] will not receive any gradient (red region), while the other values will receive gradient (green region). The band width can be controlled via the parameter p ∈ (0,0.5). When p is close to zero, the Tp value will decrease, having the effect of increasing the width of the red band, where the corresponding weights do not get gradient. When p = 0, the value of Tp will be equal to the median and this will be equivalent to the baseline (e.g. MBW = MFW) illustrated in Figure 5a.

Results. We apply RBBM for sparse training in our pretraining scenario, for the 30M parameter Llama model, using our training setup from Sec. 2, and for unstructured sparsities between 10% and 90%. We compute the capacity of the sparse representation. The results for our RMSE-based heuristic and the standard sparse training baseline (Magnitude Pruning) are provided in Figure 5d. The results show that our RMSE-based approach enables consistently higher capacity than the baseline.

### 6 Related Work

We focus on studies that extended classical scaling laws [17; 15] to model compression. Frantar et al. [9] presented the first scaling law for weight-sparse Transformers, across vision and language and unstructured and semi-structured sparsity. Earlier work by Clark et al. [5] studied mixture-of-experts sparsity, deriving scaling laws in terms of total parameters and compute per token, reinforcing the idea that only effective parameters govern scaling.

A recent breakthrough by Kumar et al. [19] introduced scaling laws that incorporate numerical quantization during training and inference, showing that, as for sparsity, a model trained in low precision behaves like a smaller high-precision model. They also apply their approach to post-training quantization (PTQ), showing that PTQ quality worsens as training data increases. For training-time quantization, their laws suggest that using lower precision allows training larger models on the same compute budget. Relative to their pioneering work, we bring the following improvements. First, we investigate a different and arguably simpler scaling law, showing that it yields a considerably better fit for quantization itself (see Table 1). Second, our key focus is different, we provide a first interpretation of the notion of representation “capacity”, together with a theoretical justification, and ample experimental validation. Finally, we validate the factorization property posited by Kumar et al. [19], as well as extensions to hybrid formats. Follow-up work by Sun et al. [30] examines scaling laws for floating-point (FP) formats, finding that the law of Kumar et al. [19] does not provide a good fit in this case, and investigates an extension of the law via additional parametrization.

Preliminary work by Frantar et al. [10] proposed the single-parameter scaling law on which we build, and showed that it can be applied to instances of weight quantization and sparsity, by directly fitting the efficiency parameter. By contrast, we identify a general law, in the sense that the same parametric form can transfer between compression types, to hybrid sparse-quantized formats, as well as to instances where both weights and activations are compressed. More interestingly, we equate the representation capacity factor in the law with a natural notion of representation capacity, show that the law factorizes across representations. In concurrent work, ParetoQ [21] aimed to unify the fragmented landscape of LLM quantization by systematically evaluating the interplay between training strategy, quantization function design, and bit selection. Our results complement their findings: for instance, we obtain that, for the architectures we consider, 2-bit weight-only quantization is Pareto optimal.

### 7 Discussion and Limitations

Our study introduces representation capacity—roughly defined as a simple monotone transform of the Gaussian MSE—as a unified metric when training compressed models across various representations. Capacity enables format comparisons without retraining or exhaustive grid searches, so that future hardware designers can expose any format whose capacity ρ dominates the Pareto frontier, confident that software will exploit it optimally. Moreover, our law factorizes, further simplifying the search for the “optimal” training format.

A few caveats remain. First, in line with prior work in this area, our experiments are limited to decoder-only Llama-style architectures trained on C4 in the data-rich regime (100 toks/param); we plan to extend this at larger scale. Second, the law may need specific fits for ultra-low precision (e.g. 2-bit or ternary formats) and for vector-quantization codebooks below 8 entries, suggesting second-order effects may need to be taken into account. Third, while our theoretical evidence uses standard assumptions, it could be extended to more complex representation types.

### References

- [1] Baskin, C., Liss, N., Schwartz, E., Zheltonozhskii, E., Giryes, R., Bronstein, A. M., and Mendelson, A. (2021). Uniq: Uniform noise injection for non-uniform quantization of neural networks. ACM Transactions on Computer Systems (TOCS), 37(1-4), 1–15.
- [2] Bengio, Y., Léonard, N., and Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432.
- [3] Chen, C., Shen, L., Huang, H., and Liu, W. (2021). Quantized Adam with Error Feedback. arXiv preprint arXiv:2004.14180.
- [4] Chen, X., Liu, S., Sun, R., and Hong, M. (2019). On the Convergence of A Class of Adam-Type Algorithms for Non-Convex Optimization. International Conference on Learning Representations.
- [5] Clark, A., de Las Casas, D., Guy, A., Mensch, A., Paganini, M., Hoffmann, J., Damoc, B., Hechtman, B., Cai, T., Borgeaud, S., et al. (2022). Unified scaling laws for routed language models. International Conference on Machine Learning, pages 4057–4086.
- [6] Défossez, A., Bottou, L., Bach, F., and Usunier, N. (2022). A simple convergence proof of adam and adagrad. Transactions on Machine Learning Research.
- [7] Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., and Modha, D. S. (2019). Learned step size quantization. arXiv preprint arXiv:1902.08153.
- [8] Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2022). Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323.
- [9] Frantar, E., Ruiz, C. R., Houlsby, N., Alistarh, D., and Evci, U. (2024). Scaling laws for sparsely-connected foundation models. In International Conference on Learning Representations.
- [10] Frantar, E., Evci, U., Park, W., Houlsby, N., and Alistarh, D. (2025). Compression scaling laws:unifying sparsity and quantization.
- [11] Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., and Keutzer, K. (2022). A survey of quantization methods for efficient neural network inference. In Low-power computer vision, pages 291–326. Chapman and Hall/CRC.
- [12] Gibney, E. et al. (2022). How to shrink ai’s ballooning carbon footprint. Nature, 607(7920), 648–648.
- [13] Harma, S. B., Chakraborty, A., Kostenok, E., Mishin, D., Ha, D., Falsafi, B., Jaggi, M., Liu, M., Oh, Y., Subramanian, S., and Yazdanbakhsh, A. (2025). Effective interplay between sparsity and quantization: From theory to practice. In International Conference on Learning Representations. arXiv:2405.20935.
- [14] Hoefler, T., Alistarh, D., Ben-Nun, T., Dryden, N., and Peste, A. (2021). Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. Journal of Machine Learning Research, 22(241), 1–124.

- [15] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J. W., and Sifre, L. (2024). Training compute-optimal large language models. In Advances in Neural Information Processing Systems.
- [16] Hou, L., Zhang, R., and Kwok, J. T. (2019). Analysis of quantized models. In International Conference on Learning Representations.
- [17] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. (2020). Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.
- [18] Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR). arXiv:1412.6980.
- [19] Kumar, T., Ankner, Z., Spector, B. F., Bordelon, B., Muennighoff, N., Paul, M., Pehlevan, C., Ré, C., and Raghunathan, A. (2024). Scaling laws for precision. arXiv preprint arXiv:2411.04330.
- [20] Li, X. and Li, P. (2022). Analysis of Error Feedback in Federated Non-Convex Optimization with Biased Compression. arXiv preprint arXiv:2211.14292.
- [21] Liu, Z., Zhao, C., Huang, H., Chen, S., Zhang, J., Zhao, J., Roy, S., Jin, L., Xiong, Y., Shi, Y., Xiao, L., Tian, Y., Soran, B., Krishnamoorthi, R., Blankevoort, T., and Chandra, V. (2025). Paretoq: Scaling laws in extremely low-bit llm quantization. arXiv preprint arXiv:2502.02631.
- [22] Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations.
- [23] Malinovskii, V., Panferov, A., Ilin, I., Guo, H., Richtárik, P., and Alistarh, D. (2025). HIGGS: Pushing the limits of large language model quantization via the linearity theorem. In L. Chiruzzo, A. Ritter, and L. Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 10857–10886, Albuquerque, New Mexico. Association for Computational Linguistics.
- [24] Modoranu, I.-V., Safaryan, M., Malinovsky, G., Kurtic, E., Robert, T., Richtarik, P., and Alistarh, D. (2024). Microadam: Accurate adaptive optimization with low space overhead and provable convergence.
- [25] Panferov, A., Chen, J., Tabesh, S., Castro, R. L., Nikdan, M., and Alistarh, D. (2025). Quest: Stable training of llms with 1-bit weights and activations. arXiv preprint arXiv:2502.05003.
- [26] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. In Proceedings of the 37th International Conference on Machine Learning (ICML), pages 13934–

13944. PMLR. T5 and C4 Dataset.

- [27] Reddi, S. J., Kale, S., and Kumar, S. (2019). On the convergence of Adam and beyond. arXiv preprint arXiv:1904.09237.
- [28] Robert, T., Safaryan, M., Modoranu, I.-V., and Alistarh, D. (2024). LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics. arXiv preprint arXiv:2410.16103.
- [29] Sardana, N., Portes, J., Doubov, S., and Frankle, J. (2024). Beyond chinchilla-optimal: Accounting for inference in language model scaling laws. In International Conference on Machine Learning.
- [30] Sun, X., Li, S., Xie, R., Han, W., Wu, K., Yang, Z., Li, Y., Wang, A., Li, S., Xue, J., Cheng, Y., Tao, Y., Kang, Z., Xu, C., Wang, D., and Jiang, J. (2025). Scaling laws for floating-point quantization training. arXiv preprint arXiv:2501.02423.

- [31] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.
- [32] Zhou, D., Chen, J., Cao, Y., Yang, Z., and Gu, Q. (2018). On the Convergence of Adaptive Gradient Methods for Nonconvex Optimization. arXiv preprint arXiv:1808.05671.
- [33] Zhu, M. and Gupta, S. (2017). To prune, or not to prune: exploring the efficacy of pruning for model compression. arXiv preprint arXiv:1710.01878.

### Appendix Roadmap

This appendix provides supporting material organized as follows:

- • Experimental Setup (Appendix A): Model architectures, hyperparameters, and training configurations.
- • Factorization of Representation Capacity (Appendix B): Detailed analysis showing how representation capacity matrices can be factorized for various compression techniques including quantization, sparsity, and their combinations.
- • Ablation Studies on Law Formulation (Appendix C): Investigation of different noise distributions (Gaussian, Logistic, Student’s t, Laplace) and functional forms (tanh, logistic) for the scaling law formulation.
- • Scaling Laws for Vector Quantization (Appendix D): Implementation details and algorithms for vector quantization approaches, including forward and backward pass descriptions for HIGGS-based training.
- • Theoretical Support (Appendix E): Convergence analysis for Adam optimizer with StraightThrough Estimation (STE), including complete proofs and supporting lemmas.
- • Improved Sparse Training via RBBM (Appendix F): Comparison of our backward mask heuristics against RigL and Gradual Magnitude Pruning, with detailed descriptions of different masking strategies.

### A Experimental setup

Hyperparameters. Table 2 summarizes the architectural and training hyperparameters for each model size.

Model size # Layers # Heads # Embeddings Learning rate

30M 6 5 640 1.2 · 10−3 50M 7 6 768 1.2 · 10−3

100M 8 8 1024 6 · 10−4 200M 10 10 1280 3 · 10−4

Table 2: Key training hyperparameters for each model size.

We use 8x80GB H100 machines for efficient training, and training one model takes on average 1 hour. To produce the full set of results we ran in total approximately 250 such training runs for various compression configurations.

### B Factorization of Representation Capacity

Figures 6-9 show factorization of the representation capacity matrix for various in-training compression techniques:

- 1. Quantized weights and activations (Fig. 6).
- 2. Sparsity + QuEST quantizer (Fig. 7).
- 3. Joint sparse & quantized weights + activations (Fig. 8), for all combinations (sa,qa,qb) for sparsity sa ∈ [0.25,0.5,0.75] and bit widths qa,qb ∈ [2,4,6].
- 4. Sparsity + uniform quantizer with maximum absolute value as a scale (Fig. 9).

From the factorized representation-capacity matrices we observe the following:

- 1. The element-wise error of the fitted coefficients ρ (from our scaling law) is of order 10−3–10−2.
- 2. A rank-1 row-column outer product accurately approximates the matrix, confirming the multiplicative property of representation capacity ρ in various scenarios.

- 3. Approximation error remains of the order 10−2, except for the cases of extreme 2-bit quantization, where ρ ≲ 0.1. We explain this gap due to the poorer performance of the optimizer in these extreme compression regimes, which is not taken into account currently by our model (as it uses the same coefficients for both 16 and 2 bits).

True (P)

Approximated (P)

Absolute error

[Figure 8]

[Figure 9]

[Figure 10]

0.09 0.15 0.19 0.20 0.21

0.04 0.10 0.16 0.19 0.21

0.04 0.06 0.03 0.01 --

- 1

- 2

- 3

- 4

- 1

- 2

- 3

- 4

- 1

- 2

- 3

- 4

0.15 0.30 0.40 0.45 0.45

0.09 0.22 0.35 0.41 0.45

0.06 0.09 0.06 0.04 --

Weightbits

Weightbits

Weightbits

- 0.19 0.42 0.62 0.70 0.74
- 0.20 0.46 0.72 0.85 0.90
- 0.21 0.48 0.76 0.91 1.00

0.15 0.36 0.57 0.67 0.74

0.03 0.06 0.05 0.03 --

0.19 0.43 0.69 0.82 0.90

0.01 0.03 0.03 0.03 --

0.21 0.48 0.76 0.91 1.00

-- -- -- -- --

16

16

16

1 2 3 4 16

1 2 3 4 16

1 2 3 4 16

Activation bits

Activation bits

Activation bits

- Figure 6: Representation capacity coefficients for independent quantization of weights and activations. Element-wise ρ fitting error is not greater than 5 · 10−3.

[Figure 11]

0.0 0.25 0.5 0.75

Sparsity

- 2

- 3

- 4

8

16

Bits

0.57±0.000.54±0.000.42±0.040.33±0.00

0.75±0.010.66±0.040.57±0.030.35±0.05

0.87±0.000.81±0.000.68±0.000.46±0.00

- 0.99±0.010.90±0.010.75±0.000.51±0.01
- 1.00±0.010.84±0.010.70±0.010.50±0.01

True (P)

[Figure 12]

0.0 0.25 0.5 0.75

Sparsity

- 2

- 3

- 4

8

16

Bits

0.57 0.48 0.39 0.28

0.75 0.63 0.52 0.37

0.87 0.74 0.61 0.43

- 0.99 0.84 0.69 0.49
- 1.00 0.84 0.70 0.50

Approximated (P)

[Figure 13]

0.0 0.25 0.5 0.75

Sparsity

- 2

- 3

- 4

8

16

Bits

- -- 0.07 0.02 0.05

-- 0.02 0.05 0.02

- -- 0.08 0.07 0.03

-- 0.06 0.06 0.02

-- -- -- --

Absolute error

- Figure 7: Representation capacity coefficients with fit errors in case of sparsity combined with the QuEST quantization.

0.3 0.4 0.5 0.6 0.7 Sparsity

0.00

0.02

0.04

0.06

0.08

Abs.Error

- Figure 8: Representation capacity fit errors for sparse+quantized weights and quantized activations. Error bars denote ±1 standard deviation from the mean.

### C Ablation studies on Law Formulation

#### C.1 Evaluating RMSE across Different Distributions

We investigate how the choice of noise distribution used in our law formulation from Sec. 4.1 affects the predicted representation capacity. In Figure 10a we plot the mapping ρ(MSE) for different bit widths using Logistic, Student’s t, and Laplace noise distributions. Each distribution is rescaled to have zero mean and unit variance.

True (P)

Approximated (P)

Absolute error

[Figure 14]

[Figure 15]

[Figure 16]

- 0.05 0.58 0.85 1.00 1.00 0.09 0.52 0.81 0.95 0.95 0.14 0.48 0.77 0.89 0.89 0.13 0.46 0.72 0.84 0.84 0.13 0.44 0.67 0.78 0.78 0.08 0.42 0.62 0.71 0.72

- 0.08 0.38 0.56 0.64 0.64
- 0.09 0.35 0.49 0.56 0.56

- 0.06 0.30 0.41 0.47 0.47 0.02 0.24 0.31 0.35 0.35

0.05 0.58 0.85 1.00 1.00 0.04 0.55 0.81 0.94 0.95 0.04 0.52 0.76 0.89 0.89 0.04 0.49 0.72 0.84 0.84 0.04 0.46 0.67 0.78 0.78 0.03 0.42 0.61 0.72 0.72 0.03 0.37 0.55 0.64 0.64 0.03 0.33 0.48 0.56 0.56 0.02 0.27 0.40 0.47 0.47 0.02 0.20 0.30 0.35 0.35

-- -- -- -- -----------

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

0.05 0.03 0.00 0.00 0.10 0.04 0.00 0.00

- 0.09 0.03 0.00 0.00
- 0.10 0.01 0.00 0.00 0.05 0.00 0.01 0.00

Sparsity

Sparsity

Sparsity

- 0.05 0.00 0.01 0.00
- 0.06 0.02 0.01 0.00 0.04 0.03 0.01 0.01 0.00 0.03 0.01 0.00

2 3 4 8 16

2 3 4 8 16

2 3 4 8 16

Bits

Bits

Bits

- Figure 9: Representation capacity coefficients matrix for sparsity applied with uniform quantization. Element-wise ρ fitting error is not greater than 2 · 10−3.

We observe that, no matter which noise distribution we choose, the mapping ρ(MSE) always remains strictly monotonically decreasing. In principle, one could use heavy-tailed distributions (for example, Student-t or Laplace) to give more weight to extreme outlier errors. However, this leads to a smaller range of MSE values. By contrast, assuming Gaussian noise—which we propose—produces the widest spread of MSE, which in turn allows for a better fit for the scaling law. In short, although monotonicity is preserved under various distributions, the Gaussian MSE delivers the best overall representation capacity prediction, so we adopt it as our default formulation.

Throughout this work, unless specified otherwise, MSE is computed over standard Gaussian input.

1.0

| |
|---|
| |

0.9

| |
|---|

0.8

0.7

###### ()R

| |
|---|

0.6

Gaussian Logistic

0.5

| |
|---|

0.4

Student's t

Laplace

0.3

10 4 10 3 10 2 10 1 100

MSE

(a) Effect of input noise distribution on the mapping ρ(MSE).

30 M 50 M 100 M

3.6

3.5

3.4

3.3

L

3.2

3.1

3.0

2.9

10 4 10 3 10 2 10 1

MSE

(b) L(MSE) fit for the best functional fit, tanh.

#### C.2 Functional form of the Law

The behavior of ρ(GMSE) observed in our experiments can be captured by fitting multiple smooth, monotonically decreasing functions, with no more than 3 additional parameters. In principle, a wide range of such functions can be used to model this relationship, depending on the desired fit properties.

For lower overall fitting error, we found it beneficial to constrain the function to satisfy boundary conditions f(0) = 1 and f(1) = 0. This way the correct behavior in the high-error region MSE ≲ 1 is enforced, which is critical for stable predictions in the extreme compression cases. We fit hyperparameters for data points (MSE,L) using the Huber loss with δ = 1e−4. The corresponding fits are summarized in Table 3, the fitting error is calculated for the combined scaling law L(MSE) = L(ρ(MSE)).

Throughout this work, we adopt the functional form of hyperbolic tangent as it provides the smallest fitting error.

### D Scaling Laws for Vector Quantization

In this section, we provide detailed information about the Vector Quantization approach used to produce the results in Figure 2(a). Algorithms 1 and 2 describe the forward and backward passes

Functional form Fitting error Decoupled Independently fitted ρ 4.2 · 10−4

Tanh ρ = tanh(F · log1/4 MSE)C 4.7 · 10−4 Logistic ρ = (1 + B · MSEA)−1 4.9 · 10−4

1 − MSEA 1 + B · MSEA

Logistic (1, 0) ρ =

1.1 · 10−3

Table 3: Functional form choices and associated fitting error.

over a linear layer actively quantized with HIGGS for row-major weights. As was described earlier, our method is combines ideas from Panferov et al. [25] for the gradient estimator, and Malinovskii et al. [23] for the lattice representation. We use the trust estimation method that zeros out gradients for any point lying outside a hypersphere of radius R: ∥x∥22 > R2. Our experiments were conducted on 30M and 50M models using the same set of hyperparameters as in Sec. 2.

- Algorithm 1 VQ Training Forward

- 1: Input: Input activations x, row-major weight w
- 2: wh = HT(w)
- 3: wˆh = projgrid wh
- 4: y = xwˆhT
- 5: Return: y, x, wˆh, Mgrid(wh;wˆh)

- Algorithm 2 VQ Training Backward

- 1: Input: ∂L∂y, x, wˆh, Mgrid(wh;wˆh)

- 2: ∂L∂x = ∂L∂ywˆh

- 3: ∂∂Lwˆ

h

= xT ∂L∂y

- 4: ∂∂Lw = IHT Mgrid(wh;wˆh) ⊙ ∂∂Lwˆ

h

- 5: Return: ∂L∂x, ∂∂Lw

E Theoretical Support

Here we provide the full proof of Theorem 1 giving a convergence analysis of the Adam optimizer when used with STE. For completeness, the description of the algorithm is presented in the Algorithm 3.

- Algorithm 3 Adam with Straight Through Estimation (STE) and AMSGrad normalization

- 1: Input: parameters β1, β2 ∈ (0,1), ϵ > 0, step-size η > 0, θ1 ∈ Rd, m0 = v0 = v0 = 0N
- 2: for t = {1,2,...,T} do
- 3: θt = C(θt) ⋄ Compress the model via quantization and/or sparsification
- 4: gt = ∇θf( θt) ⋄ Compute STE for compressed model
- 5: mt = β1mt−1 + (1 − β1)gt ⋄ Update first-order gradient momentum
- 6: vt = β2vt−1 + (1 − β2)gt2 ⋄ Update second-order gradient momentum
- 7: v˜t = max(vt,v˜t−1) ⋄ Apply AMSGrad normalization
- 8: θt+1 = θt − η m

√v˜tt+ϵ ⋄ Update the uncompressed model parameters

- 9: end for

Proof. Let G be the gradient bound with respect to ℓ2 norm, that is, ∥gt∥2 ≤ G. Using the relationship between ℓ2 and ℓ∞ norms, we conclude G ≤

##### √

dG∞. Let Γt = Diag−1/2(˜vt + ϵ) be the preconditioning (diagonal) matrix and rewrite the main update rule as

##### θt+1 = θt − ηΓtmt.

Letting θ0 = θ1, define virtual iterates xt as follows:

1 1 − β1

β1 1 − β1

θt −

xt =

θt−1.

In particular, x1 = θ1. Then, the update rule for the virtual iterates becomes

1 1 − β1

β1 1 − β1

xt+1 − xt =

(θt+1 − θt) −

(θt − θt−1)

η 1 − β1

ηβ1 1 − β1

= −

Γtmt +

Γt−1mt−1

η 1 − β1

ηβ1 1 − β1

ηβ1 1 − β1

= −

Γt−1mt−1 ±

Γtmt +

Γtmt−1

η 1 − β1

ηβ1 1 − β1

= −

Γt(mt − βmt−1) +

(Γt−1 − Γt)

mt−1

def= ∆Γt

ηβ1 1 − β1

= −ηΓtgt +

∆Γtmt−1.

Next we apply smoothness (Assumption 1) of the loss function f over the iterates xt:

L 2 ∥xt+1 − xt∥2.

f(xt+1) ≤ f(xt) + ⟨∇f(xt),xt+1 − xt⟩ +

Taking expectation and splitting the inner product into two part, we obtain

E[f(xt+1)] − E[f(xt)] ≤ −ηE[⟨∇f(xt),Γtgt⟩] + ηE ∇f(xt),

β1 1 − β1

∆Γtmt−1

2

η2L 2

β1 1 − β1

E Γtgt −

∆Γtmt−1

+

β1 1 − β1

= −ηE[⟨∇f(θt),Γtgt⟩]

+ηE ∇f(xt),

∆Γtmt−1

I

II

2

η2L 2

β1 1 − β1

. (5)

E Γtgt −

+ηE[⟨∇f(θt) − ∇f(xt),Γtgt⟩]

+

∆Γtmt−1

IV

III

In the following, we bound all the four terms mentioned above.

- Bounding term I. Let ∥∆Γt∥ be the operator norm (with respect to ℓ2 norm) of the matrix ∆Γt. Since ∆Γt is diagonal, the spectral norm coincides with the largest diagonal value in magnitude. Using unbiasedness of the stochastic gradients, we have I = −ηE[⟨∇f(θt),Γt−1gt⟩] − ηE[⟨∇f(θt),∆Γtgt⟩]

≤ −ηE ∇f(θt),Γt−1∇f( θt) + ηG2E[∥∆Γt∥].

= −ηE ∇f( θt),Γt−1∇f( θt) + ηE ∇f( θt) − ∇f(θt),Γt−1∇f( θt) + ηG2E[∥∆Γt∥]. ≤ −ηλmin(Γt−1)E[∥∇f( θt)∥2] + ηLGE[∥Γt−1∥∥ θt − θt∥] + ηG2E[∥∆Γt∥] ≤ −

η C0

E[∥∇f( θt)∥2] + ηLGE[∥ θt − θt∥ · ∥Γt−1∥] + ηG2E[∥∆Γt∥], (6)

- where we used Assumption 2 and Lemma 3 to bound

1 C0

λmin(Γt−1) ≥ (∥v˜t−1∥max + ϵ)−1/2 ≥ G2 + ϵ −1/2 def=

.

- Bounding term II. Splitting the inner product again and bounded each term, we get

II = ηE ∇f(θt),

β1 1 − β1

∆Γtmt−1 + ηE ∇f(xt) − ∇f(θt),

β1 1 − β1

∆Γtmt−1

≤

ηβ1 1 − β1

E[∥∇f(θt)∥∥∆Γtmt−1∥] +

η2Lβ12 (1 − β1)2

E[∥Γt−1mt−1∥ · ∥∆Γtmt−1∥]

≤

ηβ1 1 − β1

G2E[∥∆Γt∥] +

η2β12LG2 (1 − β1)2√ϵ

E[∥∆Γt∥], (7)

where we used the fact that the largest eigenvalue λmax(Γt) = ∥Γt∥ = (∥v˜t∥min + ϵ)−1/2 ≤ ϵ−1/2. The second inequality is due to the smoothness of f, and the last inequality is due to Lemma 1, Assumption 2 and the property of norms.

- Bounding term III. This term can be bounded as follows:

III ≤ η2LE ∥Γtgt∥2 +

η2Lβ12 (1 − β1)2

E ∥∆Γtmt−1∥2

≤

η2L ϵ

E[∥gt − ∇f( θt) + ∇f( θt)∥2] +

η2Lβ12 (1 − β1)2

E ∥∆Γtmt−1∥2

≤

η2L ϵ

E[∥∇f( θt)∥2] + σ2 +

η2Lβ12G2 (1 − β1)2

E[∥∆Γt∥2]

≤

η2L ϵ

E[∥∇f( θt)∥2] +

η2Lσ2 ϵ

+

η2Lβ12G2 (1 − β1)2

E[∥∆Γt∥2], (8)

- where we used Assumption 3 that gt is unbiased with bounded variance σ2.

- Bounding term IV. Finally, for the fourth term, we have

IV = ηE[⟨∇f(θt) − ∇f(xt),Γt−1gt⟩] + ηE[⟨∇f(θt) − ∇f(xt),∆Γtgt⟩] ≤ ηE ∇f(θt) − ∇f(xt),Γt−1∇f( θt) +

η2Lβ1 1 − β1

E[∥Γtmt−1∥∥∆Γtgt∥]

- (a)

≤

ηρ 2ϵ

E[∥∇f( θt)∥2] +

η 2ρ

E[∥∇f(θt) − ∇f(xt)∥2] +

η2β1LG2 (1 − β1)√ϵ

E[∥∆Γt∥]

- (b)

η3β12L2 2(1 − β1)2ρ

η2β1LG2 (1 − β1)√ϵ

ηρ 2ϵ

E ∥Γtmt−1∥2 +

E[∥∇f( θt)∥2] +

E[∥∆Γt∥]

≤

η2Lβ1G2 (1 − β1)√ϵ

η3β12L2 2(1 − β1)2ρϵ

ηρ 2ϵ

E ∥mt−1∥2 +

E[∥∇f( θt)∥2] +

E[∥∆Γt∥], (9)

≤

where (a) is due to Young’s inequality and (b) is based on Assumption 1. Now integrating (6), (7),

(8), (9) into (5),

- I ≤ −

η C0

E[∥∇f( θt)∥2] + ηLGE[∥ θt − θt∥ · ∥Γt−1∥] + ηG2E[∥∆Γt∥]

- II ≤

ηβ1 1 − β1

G2E[∥∆Γt∥] +

η2β12LG2 (1 − β1)2√ϵ

E[∥∆Γt∥]

- III ≤

η2L ϵ

E[∥∇f( θt)∥2] +

η2Lσ2 ϵ

+

η2β12LG2 (1 − β1)2

E[∥∆Γt∥2]

- IV ≤

ηρ 2ϵ

η3β12L2 2(1 − β1)2ρϵ

η2Lβ1G2 (1 − β1)√ϵ

E ∥mt−1∥2 +

E[∥∇f( θt)∥2] +

E[∥∆Γt∥],

and taking the telescoping summation over t = 1,...,T, we obtain E[f(xT+1)] − E[f(x1)]

T

T

η2L ϵ

η3β12L2 2(1 − β1)2ρϵ

Tη2Lσ2 ϵ

η C0

ηρ 2ϵ

E ∥mt−1∥2

E[∥∇f( θt)∥2] +

≤ −

+

+

+

t=1

t=1

T

T

T

ηG2 1 − β1

η2β1LG2 (1 − β1)2√ϵ

η2β12LG2 (1 − β1)2

ηLG T

E[∥∆Γt∥2] +

E[∥ θt − θt∥ · ∥Γt−1∥]

E[∥∆Γt∥] +

+

+

t=1

t=1

t=1

T

η2L ϵ

η3β22L2 2(1 − β1)2ρϵ

Tη3L2β12σ2 2(1 − β1)2ρϵ

Tη2Lσ2 ϵ

η C0

ηρ 2ϵ

E[∥∇f( θt)∥2] +

≤ −

+

+

+

+

t=1

T

T

T

ηG2 1 − β1

η2β1LG2 (1 − β1)2√ϵ

η2β12LG2 (1 − β1)2

ηLG T

E[∥∆Γt∥2] +

E[∥ θt − θt∥ · ∥Γt−1∥],

E[∥∆Γt∥] +

+

+

t=1

t=1

t=1

##### and η ≤ η0 def= ϵ(1−β

1)

where we used Lemma 1. Choosing ρ = 2Cϵ

4LC0 and using Lemma 2, we get

0

T

Tη2Lσ2 ϵ

Tη3L2C0β12σ2 (1 − β1)2ϵ2

η 2C0

E[∥∇f( θt)∥2] +

E[f(xT+1) − f(x1)] ≤ −

+

t=1

T

2ηG2 (1 − β1)√ϵ

4η2β1LG2 (1 − β1)2ϵ

ηLG T

E[∥ θt − θt∥ · ∥Γt−1∥].

+

+

+

t=1

Re-arranging terms, we get

T

f(θ1) − f∗ Tη

ηLσ2 ϵ

η2L2C0β12σ2 (1 − β1)2ϵ2

1 T

E[∥∇f( θt)∥2] ≤ 2C0

+

+

t=1

T

ηβ1LG2 T(1 − β1)2ϵ

G2 T(1 − β1)√ϵ

E ∥ θt − θt∥2 ϵ + ∥v˜t−1∥min

2C0LG T

+4C0

+

+

t=1

where in the last inequality we used x1 = θ1 and the lower bound f∗ ≤ f(θ) for all θ ∈ Rd. Finally, choosing η = min(η0, √1T ) and considering the two cases, we continue

,

T

f(θ1) − f∗ √

Lσ2 ϵ

L2C0β12σ2 (1 − β1)2ϵ2T

1 T

1 η0

E[∥∇f( θt)∥2] ≤ 2C0 max 1,

√

√

+

+

T

T

T

t=1

T

G2 T(1 − β1)√ϵ

β1LG2 T3/

E ∥ θt − θt∥2

2C0LG T

+4C0

+

+

(1 − β1)2ϵ

2

ϵ + ∥v˜t−1∥min ≤ 2C0

t=1

f(θ1) − f∗ √

L2C0β12σ2 (1 − β1)2ϵ2T

Lσ2 ϵ

√

+

+

T

T

f(θ1) − f∗ 2η0T

G2 T(1 − β1)√ϵ

β1LG2 T3/

+

+

+4C0

(1 − β1)2ϵ

2

T

2C0LG √ϵ

1 T

E

∥ θt − θt∥2 ,

+

t=1

√

√

Using the bounds G ≤

NG∞, C0 ≤

N 2 C and surpessing higher order terms, we simplify the

bound to

√

T

T

N3/

Lσ2 ϵ

1 T

CLG∞ √ϵ

C

N √

1 T

2

f(θ1) − f∗ +

E[∥∇f( θt)∥2] ≤

E

∥ θt − θt∥2 ·N+

+O

T

T

t=1

t=1

which completes the proof of the theorem.

| |
|---|

,

- Lemma 1. For any t ≥ 1 the following bounds hold:

∥mt∥ ≤ G,

T

t=1

E ∥mt∥2 ≤ Tσ2 +

T

t=1

E ∥∇f( θt)∥2 (10)

Proof. Let us start with the proof of the first bound on mt.

∥mt+1∥2 = ∥β1mt + (1 − β1)gt+1∥2 ≤ β1∥mt∥2 + (1 − β1)∥gt+1∥2

≤ β1t∥m1∥ + (1 − β1)

t+1

τ=2

β1t+1−τ∥gτ∥2 = (1 − β1)

t+1

τ=1

β1t+1−τ∥gτ∥2.

Using the bounded gradient assumption, we get

∥mt∥2 ≤ (1 − β1)G2

t

τ=1

β1t−τ ≤ G2.

To derive the bound with expectation, we apply Cauchy-Schwartz inequality and the bounded variance assumption:

T

t=1

E ∥mt∥2 ≤ (1 − β1)

T

t=1

t

τ=1

β1t−τE ∥gτ∥2

≤

T

t=1

E ∥gt∥2 =

T

t=1

E ∥gt − ∇f( θt) + ∇f( θt)∥2

≤

T

t=1

σ2 + E ∥∇f( θt)∥2 = Tσ2 +

T

t=1

E ∥∇f( θt)∥2 .

| |
|---|

- Lemma 2. For ∆Γt = Γt−1 − Γt we have T

T

1 √ϵ

∥∆Γt∥ ≤

,

t=1

t=1

1 ϵ

∥∆Γt∥2 ≤

.

Proof. From the definitions of Γt = Diag−1/2(˜vt + ϵ) and v˜t = max(vt,v˜t−1) imply that ∆Γt = Γt−1 − Γt is positive semidefinite. Hence, ∥∆Γt∥ = λmax(∆Γt) ≥ 0. Using the convexity of λmax over symmetric matrices, we get

T

T

∥∆Γt∥ = max

∆Γt,i

i

t=1

t=1

T

1 v˜t−1,i + ϵ −

1 v˜t,i + ϵ

1 v˜0,i + ϵ −

1 v˜T,i + ϵ ≤

1 √ϵ

= max

= max

i

i

t=1

For the second sum of squared norms, notice that for scalars a ≥ b ≥ 0, it holds that

##### (a − b)2 ≤ (a − b)(a + b) = a2 − b2.

Therefore, the above derivation can be repeated without the square roots as follows:

T

T

∥∆Γt∥2 = max

∆Γ2t,i

i

t=1

t=1

2

T

1 v˜t−1,i + ϵ −

1 v˜t,i + ϵ

= max

i

t=1

T

1 v˜t−1,i + ϵ −

1 v˜t,i + ϵ

1 v˜0,i + ϵ −

1 v˜T,i + ϵ ≤

1 ϵ

##### = max

= max

,

i

i

t=1

which completes the proof.

- Lemma 3. For all iterates t ≥ 1 the following bound holds ∥v˜t∥∞ ≤ G2.

Proof. From the update rules we get the bound for vt using the initialization v0 = 0:

∥vt∥∞ ≤ ∥vt∥1 ≤ β2∥vt−1∥1 + (1 − β2)∥gt∥2 ≤ β2∥vt−1∥1 + (1 − β2)G2

t−1

≤ β2t∥v0∥1 + (1 − β2)G2

β2τ ≤ G2.

τ=0

Hence, using the update rule of v˜t and initialization v˜0 = 0, we conclude

##### ∥v˜t∥∞ ≤ max(∥vt∥∞,∥v˜t−1∥∞) ≤ G2.

| |
|---|

| |
|---|

Next, we simplify the optimization setup by considering SGD optimizer over (still generally nonconvex) quadratics. In this special case, we provide improved and generally optimal asymptotic convergence rate. Moreover, we do not use the bounded gradient condition (i.e., ∥gt∥∞ ≤ G∞) of Assumption 2 in this analysis.

More formally, consider iterates θt+1 = θt − η ∇θf( θt), where θt = C(θt) is the compressed model. Suppose that the loss function is quadratic with Hessian matrix H ∈ RN×N and our compression

scheme C: RN → RN is unbiased, namely EC[ θt] = θt. Since the loss is quadratic, we have

##### ∇f( θt) = ∇f(θt + ( θt − θt)) = ∇f(θt) + H( θt − θt).

Denote by Et = E[·|θt] the conditional expectation conditioned on iterate θt, and apply unbiasedness of the compression to get

Et∥∇f( θt)∥2 = ∥∇f(θt)∥2 + Et∥H( θt − θt)∥2 (11) Therefore,

Et[f(θt+1) − f∗] ≤ (f(θt) − f∗) − ηEt[⟨∇f(θt), ∇f( θt)⟩] +

Lη2 2

Et[∥ ∇f( θt)∥2]

Lη2 2

Lη2 2

≤ (f(θt) − f∗) − ηEt[⟨∇f(θt),∇f( θt)⟩] +

Et[∥∇f( θt)∥2] +

σ2

Lη2 2

Lη2 2

Lη2 2

= (f(θt) − f∗) − ηEt[∥∇f(θt)∥2] +

Et[∥∇f(θt)∥2] +

Et[∥H( θt − θt)∥2] +

σ2

Lη2 2

= (f(θt) − f∗) − η(1 − Lη/2)Et[∥∇f(θt)∥2] +

(Et[∥ θt − θt∥2H2] + σ2)

Lη2 2

η 2

≤ (f(θt) − f∗) −

Et[∥∇f(θt)∥2] +

(Et[∥ θt − θt∥2H2] + σ2),

where we used Et[∇f( θt)] = ∇f(θt) due to the unbiasedness of compression and enforced the bound η ≤ L1 in the last step. Hence,

T

T

2(f(x1) − f∗) ηT

1 T

1 T

E[∥∇f(θt)∥2] ≤

∥ θt − θt∥2H2 + σ2 .

+ ηL E

t=1

t=1

##### √

Choosing the step size η = min(L1 , √1T ) and applying L-smoothness, we get O(1/

T) convergence rate for the uncompressed iterates θt:

T

1 T

1 √

E[∥∇f(θt)∥2] ≤

T

t=1

2(f(x1) − f∗) + Lσ2 + L3E

T

1 T

L √

∥ θt − θt∥22 max 1,

T

t=1

.

For the convergence bound with respect to the compressed iterates θt, we apply (11) to quantify the exact difference in average gradient norms with the following identity:

T

1 T

1 T

E[∥∇f( θt)∥22] =

t=1

T

E[∥∇f(θt)∥22] + E

t=1

T

1 T

∥H( θt − θt)∥22 .

t=1

Thus, a randomly chosen compressed iterate θ from { θ1,..., θT} satisfies

T

1 √

1 T

∥ θt − θt∥22 + O

E[∥∇f( θ)∥2] ≤ L2 · E

T

t=1

.

### F Improved Sparse Training via RBBM

#### F.1 Comparison against RigL

In this subsection we compare our backward mask heuristic in Figure5d with the RigL method of (Evci et al., 2020). We run two instances of RigL: 1) the default one that updates the mask once at 100 steps (i.e. ∆ = 100) and updates the mask for the last time at 75% of training and 2) a version of RigL that is closer to our RBBM setup, which changes the mask at each step (i.e. ∆ = 1) during the entire training. In Figure 11a we observe that both versions of RigL induce lower capacity than our naive baseline for a fixed sparsity.

[Figure 17]

(a) Pre-training Llama-30M with different sparsities using our MP baseline, RBBM heuristic and RigL variations.

[Figure 18]

(b) Pre-training Llama-30M with different sparsities using our constant MP (CMP) baseline, GMP and RBBM heuristic with GMP schedule.

Figure 11: Comparison of sparse training methods for Llama-30M.

#### F.2 Comparison against Gradual Magnitude Pruning (GMP)

In this section we show our results for applying the GMP sparsity schedule [33] for our setup in Figure 11b. Our first baseline is the constant Magnitude Pruning (CMP), where the backward mask is identical to the forward mask (determined by Top-K) and the sparsity is kept fixed during training. The second baseline is the original GMP where sparsity increases gradually and we compare against the gradual sparsity schedule applied to our b-rms heuristic.

We observe our RBBM heuristic with GMP schedule has lower capacity than both CMP and GMP baselines when sparsity is < 40%. However, for sparsities ≥ 40% there is no significant difference in capacity between CMP and GMP schedules.

#### F.3 Backward Mask Heuristics

In this section we provide more details about our backward heuristics. Our purpose is to perform sparse training for both forward and backward passes. All models are trained with the same learning rates as in the Quest project.

Notation. Let θ be the model parameters, MFW be the be the mask for the forward pass, and MBW be the mask for the backward pass.

Forward Pass. The mask for the forward pass is computed using Top-K operator, where K is chosen based on the target sparsity. Supposing Top-K returns the indices of largest entries by magnitude, the ith entry in the forward mask for a tensor x is computed using the indicator function I as follows:

##### MFWi (x) = I[i ∈ TopK(|x|)] (12)

Backward Pass. The mask for the backward pass is computed using a few heuristics described below.

- 1. fw: the backward mask is simply set to the forward mask: MBW = MFW. This heuristic allows gradients to flow only through the largest parameters by magnitude selected by Top-K, while the low-magnitude parameters will have zero gradient
- 2. rms: we align the tensor x with the standard normal distribution by dividing x by its root mean square RMS(x) = n1 ni=1 x2i, which results in ||x/RMS(x)||22 = n. For this heuristic, the user sets a median deviation parameter p ∈ (0,0.5), which is used to determine the threshold for

the backward mask TRMS(x,p) = RMS(x) · ppf(0.5 + p), where ppf is the inverse cumulative distribution function of the standard normal distribution (see the scipy.stats.norm.ppf function). The multiplication with RMS(x) has the purpose of converting the threshold for the standard normal distribution to the threshold for the vector x. As a result, MBW = |x| > TRMS(x,p).

- 3. banded-rms (b-rms): the rms heuristic has the property that the absolute values of x that are larger than the threshold TRMS(x,p) will have value 1 in the mask, while the smaller ones will have value 0. This banded heuristic determines the backward mask using the threshold TRMS(x,p) computed for the rms heuristic in conjunction with the Top-K threshold (which we denote by Tk). We want to allow gradients to flow for the small parameters and create a band between TRMS(x,p) and Tk where we do not allow gradients. Concretely, the backward mask is set as follows: MBW = (|x| < min(TRMS(x,p),Tk)) ∨ (max(TRMS(x,p),Tk) < |x|). Since the median deviation p is a hyper-parameter, we do not have any control over the relationship between TRMS(x,p) and Tk and we are using the min and max functions to make sure the band is valid, e.g. the parameters do not receive gradient if they lie in the interval [min(Tk,TRMS(x,p)),max(Tk,TRMS(x,p))].
- 4. area-banded-rms (a-b-rms): in the b-rms heuristic we do not have any control over the re-

lationship between the Top-K threshold Tk and TRMS(x,p). Let us discuss the two possible cases:

- (a) Tk < TRMS(x,p) : MBW = (|x| < Tk) ∨ (TRMS(x,p) < |x|), which means that all values from x with a lower magnitude than Tk and larger magnitude than TRMS(x,p) will get gradient, while the values in the range [Tk,TRMS(x,p)] will not receive gradient, even though they were selected among the Top-K during the forward pass.
- (b) TRMS(x,p) < Tk : MBW = (|x| < TRMS(x,p)) ∨ (Tk < |x|), which is the desired case we developed the b-rms heuristic for: the largest entries from x according to the Top-K rule will receive gradient, as well as the entries smaller than TRMS(x,p). The entries lying in the interval [TRMS(x,p),Tk] will not receive gradient.

We want to make sure that case (a) above does not happen in practice and force the heuristic to behave as in the case (b). For this, we need to change the way we compute the threshold TRMS(x,p).

The area-b-rms heuristic uses the area hyper-parameter a ∈ [0,1] (instead of median deviation p) and expresses the width of the band starting from the Top-K parameter Tk towards zero to compute the threshold Ta to make sure the condition Ta < Tk always holds. As a result, MBW = (|x| < Ta) ∨ (Tk < |x|). For example, a = 0 yields Tk = Ta and this heuristic turns into fw, while a = 1 yields Ta = 0 and is equivalent to MBW = 1d (all entries set to 1, meaning all parameters get gradients). When a ∈ (0,1), the parameters smaller than Ta or larger than Tk get gradients, while the parameters lying in the interval [Ta,Tk] do not get gradients.

How to compute the threshold Ta? Compared to the threshold computation for the previous heuristic, the definition for Ta is slightly more complicated and it was computed graphically. Let f be the cdf function and f−1 be the ppf function (inverse cdf) for the standard normal distribution.

Tk RMS(x) − 0.5 (13)

Ta = RMS(x) · f−1 0.5 + (1 − a) · f

Explanations for the formula above. Suppose the Top-K threshold Tk has a corresponding cdf of 0.8 and we set a = 0.5 (which means 50%). We need to set the threshold Ta such that (f(Tk) − f(Ta))/(f(Tk) − 0.5) = a, where 0.5 is the cdf of the mean (which is identical to the median for a standard normal distribution). This ratio expresses the length of the band [0.5,f(Tk)] in the cdf space starting from f(Tk) towards the median. As a consequence, the threshold Ta = ppf(0.65) because the quantile 0.65 is the center of the interval [0.5,cdf(Tk)] = [0.5,0.8]. The explanations of each term follow:









Tk RMS(x)

Ta = RMS(x) · f−1

0.5 + (1 − a) ·

−0.5

f

 

 

=A

=B

 

 

=C

=D

=E

=F

- • A: f = cdf computes the corresponding quantile of the Top-K threshold Tk normalized by RMS(x)
- • B: subtract 0.5 from term A to compute the length of the interval [0.5,f(TkRMS)]
- • C: multiply by 1 − a because we take into consideration the band length that starts at f(TkRMS) towards 0
- • D: compute the cdf of Ta by offsetting again by 0.5 (the quantile of the median)
- • E: use ppf = f−1 to obtain the value that corresponds to cdf(Ta) for the standard normal distribution
- • F: multiply by RMS(x) to obtain Ta in the same space as x

Technical note. One could determine the threshold Ta naively by employing the formula Tanaive = (1 − a)Tk. Despite simpler, this naive approach leads to a narrower band because the cdf space is non-linear.

Conclusion. The mask computed using the a-b-rms heuristic is more straightforward to understand because the parameter a describes the area of the red band (where parameters do not receive gradients) as a percentage of the area between 0 and the Top-K threshold Tk. This heuristic can be used as a replacement for b-rms and the parameter a should be tuned, similarly to parameter p for b-rms, with the distinction that a ∈ [0,1] (for a-b-rms) and p ∈ (0,0.5) (for b-rms).

