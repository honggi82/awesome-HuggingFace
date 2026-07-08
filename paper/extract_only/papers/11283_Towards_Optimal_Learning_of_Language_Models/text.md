## Towards Optimal Learning of Language Models

Yuxian Gu1,2∗, Li Dong2, Yaru Hao2, Qingxiu Dong2, Minlie Huang1, Furu Wei2 1The CoAI Group, Tsinghua University 2Microsoft Research https://aka.ms/GeneralAI

# arXiv:2402.17759v2[cs.CL]3Mar2024

### Abstract

This work studies the general principles of improving the learning of language models (LMs), which aims at reducing the necessary training steps for achieving superior performance. Specifically, we present a theory for the optimal learning of LMs. We first propose an objective that optimizes LM learning by maximizing the data compression ratio in an “LM-training-as-lossless-compression” view. Then, we derive a theorem, named Learning Law, to reveal the properties of the dynamics in the optimal learning process under our objective. The theorem is then validated by experiments on a linear classification and a real-world language modeling task. Finally, we empirically verify that the optimal learning of LMs essentially stems from the improvement of the coefficients in the scaling law of LMs, indicating great promise and significance for designing practical learning acceleration methods. Our code can be found at https://aka.ms/LearningLaw.

>

Conventional LM Learning Optimal LM Learning

Learning Speed( | )

Learning Speed( | )

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

>

Conventional LM Learning Optimal LM Learning

[Figure 6]

[Figure 7]

Area Under Loss Curve( | )

Area Under Loss Curve( | )

[Figure 8]

Loss

[Figure 9]

[Figure 10]

[Figure 11]

>

[Figure 12]

[Figure 13]

Data Compression Ratio( | ) Data Compression Ratio( | )

[Figure 14]

[Figure 15]

0 𝑇

Training Steps

- Figure 1: Our objective is to minimize the area under loss curve, which is equivalent to maximizing the compression ratio of training corpus in the “LM-training-as-lossless-compression” view. A learning law is proposed to reveal the training dynamics of the above optimal learning.

|Conventional LM Learning (Near-)Optimal LM Learning<br><br>2.41× Speedup<br><br>| | | | | |
|---|---|---|---|---|---|
| | | | | | |

0 1000 2000 3000 4000

4.5

5.0

4.0 3.5

Training Steps

Loss

- Figure 2: Optimal learning gets the theoretical speedup upper bound of Transformer LM training on TinyStories corpus [EL23].

Scaling Laws B β

Conventional LM Learning 3.16 × 108 0.12 (Near-)Optimal LM Learning 1.99 × 107 0.14

Table 1: The (near-)optimal LM learning improves the scaling laws [KMH+20] over conventional LM training. The coefficients B,β are used to fit the

loss curves in Figure 2, i.e., Loss = L0 + (B/t)β when t > t0. See Section 4.6 for details.

∗Contribution during an internship at Microsoft Research. ⟨guyx21@mails.tsinghua.edu.cn⟩

### 1 Introduction

With the thriving of language models (LMs; HZD+21, BHA+21), there is an increasing focus on improving the learning [PR12, WSSC19] of LMs, which aims at accelerating the learning speed and achieving a certain model performance with as few training steps as possible [WWL+23]. This focus helps humans explore the limits of LMs given the rapid growth of their computational demand [HBM+22], and promotes democratization [CMS+23] of large language models (LLMs; BMR+20, Ope22, Ope23, CND+23, ADF+23), which is valuable for both research communities and industry sectors [TLI+23, TMS+23, JSM+23].

In this paper, we present a theory for optimal learning of LMs. Unlike prior works exploring practical acceleration methods at the model-level [XYH+20, ZH20], optimizer-level [YLR+20, LLH+24], or data-level [TSAM23, ATS+23, XPD+23], our work demonstrates the principles of optimizing the LM learning speed, including the optimization objective, the property of optimal learning dynamics, and the essential improvement of the learning acceleration.

Specifically, for the optimization objective, we propose to minimize the area under the loss curve (AUC; CM03), which has a clear physical significance: the description length when we view the nexttoken-prediction LM training process as lossless compression of the training data [Bel19, MCKX22, Rae23]. As shown in Figure 1, a learning process with the smallest loss AUC corresponds to the highest compression ratio. Simultaneously, the loss in this process also converges to a small value at the highest rate, given sufficiently large total training steps. Therefore, we consider optimizing LM learning equivalent to maximizing the corresponding compression ratio of the learning process, and adopt the latter as the optimization objective in our theory. Similar objectives are also employed to interpret the remarkable generalization performance of recent LLMs [VNK+23, DRD+24].

We then derive a theorem, named Learning Law, that characterizes the property of dynamics in the LM learning process that achieves the optimum of our objective. Here, a learning process is induced by a learning policy that determines which data points the LM learns as the training progresses. In this way, we solve the optimal learning policy in the sense that the corresponding compression ratio is maximized, and obtain our Learning Law (see Theorem 3.1 for a formal expression):

Learning Law All examples have the same contribution to the LM in the optimal learning process.

#### A B

Optimal LM Learning Conventional LM Learning

Similarity of Example Contributions

Compression Ratio of Optimal Learning

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Conventional Learning

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Compression Ratio

Contribution of the 𝑛 example Contribution of the 𝑛 example

- Figure 3: A: 3-D illustration of Learning Law (Theorem 3.1). In the optimal learning process, all training examples should have the same contribution to LM learning, where the contribution is defined

as the dot-product of the gradient on individual samples (∇lm, ∇ln, and ∇lk) and the gradient of a desired loss (∇L). See Section 3.2 for rigorous notation definitions. B: Experimental evidence of Learning Law. When LM learning approaches the optimum, the similarity of example contributions tends to +∞, which means all examples have the same contribution to the LM.

As shown in Figure 3, the contribution of an example is defined as the dot-product of its gradient and the gradient of a desired loss2 , which measures its influence on the LM in the desired learning

2Note that the desired loss is not necessarily the same as the training loss as discussed in Section 2.

direction. Learning Law also suggests a matching of local and global learning speed in the optimal learning process, which interprets the optimal learning policy as a dynamic data re-weighting strategy that encourages the LM to learn highly contributive examples and simultaneously avoid over-fitting them. Similar mechanisms are also found critical to the best teaching methods for humans in psychological research [Met09, KPA12].

We examine our theory by experiments on linear classification tasks based on Perceptron3 [MP43] and real-world language modeling tasks based on Transformer [VSP+17]. We first design a gradientbased method to search for the optimal learning policy under our objective. Then, we verify that the dynamics of the learning process induced by the found near-optimal policy aligns well with our Learning Law. Finally, as shown in Table 1, we provide empirical evidence showing that the near-optimal learning policy essentially improves the coefficients in the training step scaling law of LMs [KMH+20], which leads to 5.50× and 2.41× speedup to Perceptron and Transformer learning, respectively. This emphasizes the promise and significance of exploring more scalable methods to optimize the learning policy in practice and accelerate the training of LLMs.

### 2 Problem Formulation

We consider LM training on a large-scale dataset with N examples {xtrnn }Nn=1 for a sufficiently large total training time steps T. Let γn,t denote the weight of the nth training example at the time step t, a learning policy is represented by a time-variant distribution over N training examples

γt = [γ1,t,γ2,t,··· ,γn,t]⊤, satisfying Nn=1 γn,t = 1 and γn,t ≥ 0 for 1 ≤ n ≤ N,0 ≤ t ≤ T − 1. The conventionally trained LM learns with a policy γn,tc = N1 (conventional learning). Recent works [PLKS20, ABL+22] have shown that theories derived based on Gradient Decent (GD) offer insights into other gradient-based algorithms [KB15]. Therefore, for simplicity, we assume the LM is trained with GD for t = 0,1,··· ,T − 1:

N

Ltrnt (θt) =

γn,tl(xtrnn ,θt),

(1)

n=1

θt+1 = θt − η∇Ltrnt (θt),

where θt ∈ RD is the model parameters flattened into a D-dimensional vector at the time step t, η is the learning rate, and l(·,·) is the loss function of the learning problem. For LMs, l(·,·) is

typically the Maximum Likelihood Estimation (MLE) loss: l(x,θt) = −log pθ

(x), where x is a text sequence. Following [XSML23] and [MBR+22], we focus on the learning speed reflected by the reduction rate of a desired loss Ldsr computed on K examples {xdsrk }Kk=1 that do not necessarily follow the same distribution as the training examples:

t

K

1 K

l(xdsrk ,θt). (2)

Ldsr(θt) =

k=1

This formulation applies to a broad of practical scenarios including classical machine learning using a validation set to prevent over-fitting [Vap99], large-scale pre-training relying on a carefully curated held-out corpus to evaluate generalization performance [KMH+20], and domain adaptation where a natural difference exists between training and target distribution [XSML23]. As such, we search for the learning policy γt that maximizes the reduction rate of Ldsr(θt) to optimize LM learning.

However, direct analysis of this optimization problem is difficult due to the discreteness of GD. Therefore, we focus on the continuous limit of GD by considering the corresponding gradient flow of Equation 1 for t ∈ [0,T], which is more amenable to theoretical analysis [SDBD20]:

N

d dt

θ(t) = −∇Ltrn(θ(t),t) = −∇

γn(t)l(xtrnn ,θ(t)), (3)

n=1

where γn(t) is a smooth interpolation function of γn,t. According to the results in numerical analysis, GD defined in Equation 1 is the Euler method to approximately solve the initial value problem of

the gradient flow in Equation 3, and θ(t) ≈ θt when η is sufficiently small [EC21]. In Section 4, we show that the results derived from this limit align well with the experiments in discrete settings.

3In Appendix A.3, we provide a lossless data compression view of the Perceptron training, indicating that our theory also applies.

### 3 Theory for Optimal Learning of LMs

In this section, we present our theory in the continuous limit of GD. We first propose an objective for “maximizing the reduction rate of Ldsr by optimizing the learning policy”. Then, we derive our main theorem, named Learning Law, which introduces a necessary condition for the dynamics of the learning process induced by the policy that achieves the optimum of the objective.

##### 3.1 Objective: Maximizing Compression Ratio

We characterize the reduction rate of Ldsr with the area under the curve of Ldsr(θ(t)) (AUC of Ldsr) and minimize this area to achieve high learning speed:

T

Ldsr(θγ(t))dt,

min

γ(t)

0

N

(4)

s.t.

γn(t) = 1, γn(t) ≥ 0,n = 1,2,··· ,N,

n=1

where γ(t) = [γ1(t),γ2(t),··· ,γn(t)]⊤ and θγ(t) is an alias of θ(t) satisfying Equation 3 to emphasize its dependency on γ(t). As shown in Figure 1, for sufficiently large T, a learning process with minimal loss AUC owns the highest loss reduction rate. Interestingly, the AUC of Ldsr has a physical significance from the “LM-training-as-lossless-compression” view [Rae23]: the resulting description length of compressing data drawn from the desired data distribution. Therefore, Equation 4 is equivalent to maximizing the corresponding compression ratio. Note that unlike [DRD+24] that studies encoding data using a well-trained LM, we view the entire LM training as a compression process. We provide more discussion of these two perspectives in Section 5. Besides, there are still slight differences between our statement and that in prior works viewing the training process as lossless compression [Bel19, MCKX22, Rae23]: we consider the desired loss AUC of GD training for multiple epochs, while the previous statement is about the training loss AUC with single-epoch SGD training. More discussion about this difference can be found in Appendix A.2.

##### 3.2 Learning Law

Equation 4 defines an Optimal Control problem that can be solved by Maximum Principle [Pon18]. However, we find the solution hard to interpret and verify in practical LM learning. Therefore, in this work, we derive a looser necessary condition for the optimum of Equation 4.

Theorem 3.1 (Learning Law). When an LM is trained with an optimal learning policy, which yields a learning process corresponding to a maximum compression ratio on the desired data distribution, the following condition holds for 0 < t ≤ T and any m, n such that γm(t) > 0, γn(t) > 0:

∇L · ∇lm = ∇L · ∇ln = Const, (5)

where ∇L = ∇Ldsr(θ(t)) = ∇K1 Kk=1 l(xdsrk ,θ(t)), ∇lm = ∇l(xtrnm ,θ(t)), ∇ln = ∇l(xtrnn ,θ(t)), and · is dot-product. Const = −ddtLdsr(θ(t)) is the desired loss change rate over time and is independent of n and m.

To prove Theorem 3.1, we apply the Euler-Lagrange (EL) equation [GS+00] and Karush–Kuhn–Tucker (KKT) conditions [Ber16] to Equation 4, which results in the condition:

∇Ldsr(θ(t)) · ∇l(xtrnn ,θ(t)) = −ddtLdsr(θ(t)). A full proof is shown in Appendix B.

∇L · ∇ln in Equation 5 represents the contribution of the training example xtrnn to Ldsr(θ(t)), which is maximized when the gradient on xtrnn shares the same direction with the gradient of Ldsr(θ(t)). We denote CTn(t) = ∇L · ∇ln = ∇Ldsr(θ(t)) · ∇l(xtrnn ,θ(t)) for convenience in the rest of the paper. Note that when the model is converged (∇Ltrn(θ(t),t) ≈ 0), CTn(t) can be viewed as an approximation of the Influence Function [KL17] by setting the Hessian matrix of Ltrn(θ,t) at θ = θ(t) to an identity matrix [PLKS20]. In essence, Equation 5 means CTn(t) equals a value

independent of n. Since the zero-weight examples (γn(t) = 0) are typically noisy (verified in Section 4.5), Theorem 3.1 suggests that all non-noisy examples should be identically contributive to the LM in the optimal learning process. In the following, we provide more discussion of this theorem.

##### 3.3 Discussion

Theorem 3.1 suggests a matching of the local and global learning. Another interpretation of CTn(t) is the “local learning speed”: how fast the LM learns the knowledge in xtrnn that is helpful to reduce Ldsr. This is because the dot-product operation in CTn(t) can be viewed as the projection of the individual loss descending velocity ∇l(xtrnn ,θ(t)) on the desired direction. Correspondingly,

d dtLdsr(θ(t)) represents the LM’s “global learning speed”: how fast the LM gets better by learning all individual xtrnn . As a result, CTn(t) = Const = −ddtLdsr(θ(t)) in Theorem 3.1 indicates that the local learning speed should match the global learning speed in the optimal learning process.

The optimal learning policy establishes a dynamic data re-weighting strategy. Generally,

- as the learning of LM progresses, CTn(t) drops because the gradient norm on each example ||∇l(xtrnn ,θ(t))|| decreases as the LM fits xtrnn . In addition, the direction of ∇l(xtrnn ,θ(t)) diverges from ∇Ldsr(θ(t)) due to the possible discrepancy between the distribution of xtrnn and xdsrk , which also contributes to the decrease of CTn(t). Therefore, Theorem 3.1 guarantees that highly contributive

example xtrnn with high CTn(t) obtains large weights for training, in order to reduce CTn(t) to meet the value of other examples. On the other hand, Theorem 3.1 also ensures that the weights

of xtrnn are lowered before the LM over-fits it because CTn(t) should not be too small to match the global learning speed. Altogether, this forms a dynamic training data re-weighting strategy, which is intuitively essential for the optimal learning policy that maximizes the learning speed of an LM.

Theorem 3.1 is a necessary condition for the optimal learning dynamics. This is because the E-L equation and KKT conditions are necessary conditions for the global optimum when the optimization problem is non-convex. Therefore, a learning process satisfying Theorem 3.1 is not guaranteed optimal. For example, by setting γ1(t) = 1 and γ2(t) = γ3(t) = ··· = γN(t) = 0, Equation 5 is satisfied, regardless of the values of CTn(t). This learning policy corresponds to using SGD with mini-batch size = 1, which is unlikely to be the optimal [MKAT18]. Therefore, searching for the optimal policy according to Theorem 3.1 may need regularization terms in practice, which we leave for future work to explore.

### 4 Experiments

We conduct experiments in the discrete setting of Equation 1, where the conclusions derived from the continuous limits in Section 3 are still applicable when η is sufficiently small [EC21]. We first design a method to find the optimal learning policy γt ∈ RN for 0 ≤ t ≤ T − 1, by explicitly minimizing the AUC of Ldsr(θt) in the discrete setting, which maximizes the corresponding compression ratio of data drawn from the desired distribution. Then we examine our Learning Law (Theorem 3.1) on the learning process induced by the found policies. Finally, we empirically verify that maximizing the compression ratio essentially improves the scaling law coefficients [KMH+20], indicating the practical significance and promise of our theory.

##### 4.1 Finding the Optimal Learning Policy

To find the optimal γt, we directly solve the discrete version of the optimization problem defined in Equation 4 with a Proximal Gradient Method [BC11]:

T

Ldsr(θt),

J(γ) =

(6)

t=1

γt ← Proj[γt − ϵ∇γt

J(γ)], 0 ≤ t ≤ T − 1,

where J(γ) is a discrete approximation of the integral in Equation 4, ϵ is the learning rate and Proj[·] projects a point in RN to the N-simplex, ensuring that γt is a probability distribution over N training examples. The optimization process can be implemented efficiently using dynamic programming and Jacobian-Vector-Product in PyTorch [PGM+19], which is described in detail in Appendix C.

- 4

- 5

- 6

- 7

AUCoftheDesiredLoss

CompressionRate(CR)

0.25

Learning Oplicy Optimization Loss

Compression Rate

0.20

Near-Optimal Learning Conventional Learning

| |
|---|

0.15

0 100 200 300 400 500 Optimization Epochs

(a) Perceptron Linear Classification

3.3

AUCoftheDesiredLoss

CompressionRate(CR)

4.1

3.2

4.0

Learning Oplicy Optimization Loss

Compression Rate

Near-Optimal Learning Conventional Learning

3.9

3.1

| |
|---|

3.8

3.0

0 5 10 15 Optimization Epochs

(b) Transformer Language Modeling

- Figure 4: Learning policy optimization results in Perceptron linear classification (a) and Transformer language modeling tasks (b). We plot the learning policy optimization loss J(γ) (solid lines), defined in Equation 6, which represents the area under the curve (AUC) of the desired Perceptron or Transformer loss. We also show the corresponding compression ratio of the training process (dashed lines) in an "LM-as-Lossless-Compression" view. The optimization starts from conventional learning and smoothly converges to near-optimal learning with low loss AUC and high comprehension rate.

|5.50 × Speedup<br><br>Conventional Learning Near-Optimal Learning<br><br>| | | | | |
|---|---|---|---|---|---|
| | | | | | |

0 500 1000 1500 2000 Training Time Steps t

0.100 0.09

0.20

dsrDesiredLoss()Lt

(a) Perceptron Linear Classification

|2.41 × Speedup<br><br>Conventional Learning Near-Optimal Learning<br><br>| | | | | |
|---|---|---|---|---|---|
| | | | | | |

0 1000 2000 3000 4000 Training Time Steps t

4.0

5.0

dsrDesiredLoss()Lt

(b) Transformer Language Modeling

- Figure 5: Curves of the desired loss Ldsr(θt) when the model is trained using the conventional and the near-optimal learning policy. The near-optimal learning process achieves 5.50× speedup in Perceptron linear classification (a) and 2.41× speedup in Transformer language modeling (b).

##### 4.2 Experimental Setup

We conduct experiments on a linear classification task based on Perceptron [MP43] and a language modeling task based on Transformer [VSP+17]. See Appendix D for hyper-parameter configurations.

Perceptron Linear Classification. We adopt a teacher-student setting [Eng01] where each example xn = (zn,yn) is a pair of D-dimensional vector zn ∈ RD drawn i.i.d. from Gaussian distribution, and a scalar yn = sign(T · zn) given the ground truth weight T ∈ RD. We introduce a shift between the training and the desired data distribution to reflect their differences. The data are learned by an one-layer Perception parameterized by θ ∈ RD: on = σ(θ · zn) = 1+exp(1−θ·z

n), which is trained with Maximum Likelihood Estimation (MLE) loss l(xn,θ) = −log oy

n. In Appendix A.3, we show that Perceptron can be viewed as a one-step LM, which means our theory still applies.

nn(1 − on)1−y

Transformer Language Modeling. Considering the computation cost of the optimal policy searching, we adopt a two-layer Transformer with about 1.7M parameters and train it on TinyStories [EL23], a high-quality pre-training corpus. We add perturbations to the training examples (see Appendix D for details), which mimics the relatively low quality of the pre-training corpus in practice. Since our theoretical derivation is generally applicable, we believe that our theory also applies to larger LMs. To migrate the risk of over-fitting the K examples used to compute Ldsr(θt) in Section 4.1, we additionally construct a held-out test set with K examples from the desired data distribution in both Perceptron linear classification and Transformer language modeling experiments. In the following,

| | |
|---|---|
| | |
| | |
| | |
| | |

[Figure 26]

0.75

- 4

- 5

- 6

0.50

SNRt

0.25

0.00

CR

100 10 1 Desired Loss Ldsr( t)

(a) Perceptron Linear Classification

| | |
|---|---|
| | |
| | |
| | |
| | |

[Figure 27]

6

3.2

4

SNRt

3.1

2

3.0

0

6 × 100 4 × 100 Desired Loss Ldsr( t)

CR

(b) Transformer Language Modeling

- Figure 6: Empirical evidence of our Learning Law (Theorem 3.1) in Perceptron linear classification (a) and Transformer language modeling (b) tasks. We measure the degree of similarity in contribution among different samples by SIMt, the Signal-Noise-Ratio of the contribution CTn,t of training examples, calculated as the mean divided by the standard deviation of CTn,t across examples (Equation 7). Higher SIMt means better contribution similarity. We plot SIMt with respect to the desired loss Ldsr(θt) under different learning processes. Each line is a certain learning process, whose color means the corresponding compression ratio (CR). Runs with higher CR generally get higher

SIMt throughout learning, indicating that the example contributions are more similar to each other in a learning process closer to the optimum, which is in line with our Learning Law (Theorem 3.1).

we compute and report the evaluation metrics by treating the test examples, unseen during the policy optimization, as xdsrk in Equation 2.

- 4.3 Learning Policy Optimization Results

A near-optimal learning policy can be found with the method in Section 4.1. In Figure 4, we show the optimization process of finding the optimal learning policy. We plot the learning policy optimization loss J(γ), which is also the AUC of Ldsr(θt) in the learning process induced by γt, and the corresponding compression ratio CR = T log|V |

T

t=1 Ldsr(θt), where V is the size of the label / vocabulary space for Perceptron / transformer (see Appendix A.1 for more explanation). The curve of J(γ) is smooth and almost converges at the end, indicating that a near-optimal learning policy is found.

The near-optimal learning policy yields a high acceleration ratio of the learning speed. In Figure 5, we plot the curve of Ldsr(θt) when the Perceptron and Transformer are trained under the conventional and near-optimal learning policies. The near-optimal policies significantly improve the loss AUC, bringing about acceleration 5.50× and 2.41× at the end of the Perceptron and Transformer training, respectively. Note that all reported metrics are computed on the test set unseen during the policy optimization, suggesting that the near-optimal policy does not over-fit the specific examples used to compute Ldsr(θt) but helps the model learn faster on the desired distribution.

- 4.4 Direct Verification of Learning Law (Theorem 3.1)

We examine the similarity between CTn,t which is the discrete version of the individual sample contribution CTn(t) in a certain learning policy and satisfies CTn,t = CTn(t) for t = 1,2,··· ,T. The similarity (SIM) is measured by the Signal-Noise-Ratio of CTn,t:

SIMt =

CTt sCT,t

, (7)

n=1 [γn,t̸=0](CTn,t−CTt)2 N n=1 [γn,t̸=0]−1

N

where CTt = Nn=1 γn,tCTn,t is the weighted mean and sCT,t =

is the standard deviation of CTn,t for training examples with non-zero weight. The higher SIMt means that the training examples have more similar CTn,t. Note that SIMt is dimensionless, which avoids the impact of the absolute value scale change of CTn,t during learning. We also consider

SIM = T1 Tt=1 SIMt, which summarizes the similarities of CTn,t throughout the learning process.

Compression Ratio of Optimal Learning

Compression Ratio of Optimal Learning

Conventional Learning

Conventional Learning

(a) Perceptron Linear Classification

(b) Transformer Language Modeling

- Figure 7: Empirical evidence of the Learning Law (Theorem 3.1) in Perceptron linear classification (a) and Transformer language modeling (b) tasks. Following Figure 6, we consider

SIM = T1 Tt=1 SIMt, which summarizes the similarity of the training example contributions in a learning process. We plot the relationship between SIM and CR, and observe an evident tendency

c

. When the learning process approaches the optimum (CR → b), the standard deviations of training example contributions should be zero to allow SIM → +∞. This verifies Learning Law (Theorem

that SIM → +∞ when CR approaches a certain value, which can be fit by SIM = log b− aCR

- 3.1) that all training examples have the same contribution to the model in optimal learning.

Higher compression ratio correlates with higher sample contribution similarities. In Figure 6, we examine the value of SIMt in the learning process induced by each policy found along the optimization process of γt. Since the found policies bring about faster convergence, we plot SIMt with respect to Ldsr(θt), rather than t. In this way, SIMt are compared at the same “stage” of the model learning, migrating the impact of different convergence speeds. Figure 6 demonstrates that the learning process with a higher compression ratio (CR) generally keeps higher SIMt in model learning, indicating that the contributions CTn,t of individual samples are more similar to each other throughout the learning process, which aligns with our Learning Law (Theorem 3.1).

Sample contributions tend to be equal when the learning process approaches the optimum. In Figure 7, we plot SIM with respect to CR for each learning process. We observe an evident tendency that SIM → +∞ when CR approaches a certain value. Accordingly, we use the function

SIM = log b− aCR

c

to fit the tendency of the experimental observations. Figure 7 indicates that when the learning process continuously improves until the optimum (CR → b), the standard deviation of CTn,t should be zero to allow SIM → +∞. This verifies Learning Law (Theorem 3.1) that the contributions of non-zero-weight training samples (CTn,t) are identical in optimal learning.

- 4.5 Properties of Zero-Weight Examples

The experiments in Section 4.4 mostly focus on the non-zero-weight examples. In this section, we provide more empirical evidence for Learning Law (Theorem 3.1) by examining the properties of the examples with γn,t = 0. We derive three properties of the optimal learning dynamics from Theorem 3.1 and then verify them through experiments. The first property guarantees that examples with non-positive contributions receive γn,t = 0, indicating that the “noisy” examples at each time step are excluded by the optimal learning policy:

- Property 4.1. The training example xtrnn whose CTn,t ≤ 0 gets γn,t = 0 before the model converges.

dsr(θ(t))

Proof. Before convergence, dL

dt < 0 holds, indicating CTn,t > 0 for xtrnn that satisfies γn,t > 0, according to Theorem 3.1. Therefore, CTn,t ≤ 0 ⇒ γn,t = 0.

| |
|---|

Fractionof=0(%),nt

80

Near-Optimal Learning Conventional Learning

| |
|---|

60

40

20

0

4 5 6 7 Compresson Rate (CR)

(a) Perceptron Linear Classification

Fractionof=0(%),nt

Near-Optimal Learning Conventional Learning

80

| |
|---|

60

40

20

0

| |
|---|

3.0 3.1 3.2 3.3 Compresson Rate (CR)

(b) Transformer Langnauge Modeling

- Figure 8: Empirical evidence of Property 4.1: non-contributive and noisy examples are excluded in

optimal learning. The y-axis is the fraction of zero-weight examples among those with CTn,t ≤ 0 at the same time step. Each point represents a learning policy, which tends to assign the example weight

γn,t = 0 to 100% of noisy and non-contributive data when it approaches the optimum.

0.0 0.5 1.0 n,t / max

n

{ n,t }

0.0

0.5

1.0

CumulativeProbility

[Figure 28]

- 4

- 5

- 6

CR

- Figure 9: Empirical evidence of Property 4.2: perfectly learned examples are ignored in optimal learning. We plot the cumulative distribution function (CDF) of the example weights γn,t that satisfies

l(xtrnn ,θt) < 1 × 10−6. Each line corresponds to a learning process. A large fraction of low-loss examples (perfectly learned) in the near-optimal learning obtain small γn,t values (ignored), and this tendency becomes more evident when the learning approaches its optimum (CR increases).

Empirical Evidence. We calculate the fraction of zero-weight examples (γn,t = 0) among all examples with non-positive contributions at t (CTn,t ≤ 0): n,t [γn,t=0] [CTn,t≤0]

n,t [CTn,t≤0] and plot this fraction with respect to the CR value of the corresponding learning process in Figure 8. We can see that when the learning process approaches the optimum, the fraction tends to 100%, indicating that the non-contributive examples are discarded.

The second property is derived only for Perceptron linear classification, which indicates that the optimal learning policy will ignore those perfectly learned training examples:

- Property 4.2. For Perceptrons, the perfectly learned xtrnn , whose margin (2yntrn −1)θt ·ztrnn → +∞

- at the time step t, gets γn,t = 0 in the optimal learning policy when the model is yet converged.

Proof. When (2yntrn − 1)θt · ztrnn → +∞, we have otrnn − yntrn → 0, which means ∇l(xtrnn ,θt) = (otrnn − yntrn)ztrnn → 0 and CTn,t → 0. Assuming γn,t ̸= 0, according to Theorem 3.1, we have CTn,t = −ddtLdsr(θ(t)) in the optimal learning process, which means that d dtLdsr(θ(t)) should be arbitrarily small. This does not hold when the model is not converged. Therefore, we have γn,t = 0.

| |
|---|

- Empirical Evidence. In Figure 9, we plot the cumulative probability distribution function of γn,t

maxn{γn,t} for the well-learned Perceptron training examples xtrnn with near-zero per-instance training loss: l(xtrnn ,θ) < 1 × 10−6. Figure 9 shows that for the near-optimal policy, more than 90% welllearned examples have relatively low γn,t (< 0.2 maxn {γn,t}). This trend becomes more evident as the learning policy approaches the optimum (CR increases), which verifies Property 4.2.

[Figure 29]

[Figure 30]

(a) Perceptron Linear Classification (b) Transformer Language Modeling

- Figure 10: Empirical evidence of Property 4.3: redundant training examples are discarded in optimal

learning. We randomly sample 2048 training examples satisfying CTn,t > 0 (contributive and unlearned examples) throughout the near-optimal learning process and show the dynamics of the

example weight γn,t (represented by the color in (a) and (b)). Since Perceptron converges quickly, we only plot its γn,t dynamics for t ≤ 50. The near-optimal policies assign γn,t = 0 to redundant examples in addition to the perfectly learned and non-contributive data points.

The third property suggests that the optimal learning policy will discard the “redundant” training examples. Although this property is derived from Perceptron linear classification, we empirically find that it also applies to Transformer language modeling. We call a set {xn}Nn=1 has “redundant” examples when the example inputs in the set are linearly correlated, i.e., there exist K scalars {αn}Nn=1, not all zero, such that Nn=1 αnzn = 0.

- Property 4.3. For Perceptrons, if the training set {xtrnn }Nn=1 has redundant examples, with probability

1, at least one example xtrni gets γi,t = 0 at the time step t when the model is yet converged in the optimal learning process.

Proof. Given that {xtrnn }Nn=1 has redundant examples, there exist scalars {αn}Nn=1, not all zero, such that Nn=1 αnztrnn = 0, which means Nn=1 αn

otrnn−yntrn CTn,t = 0. Assuming ∀1 ≤

n ≤ N, γn,t ̸= 0, according to Theorem 3.1, we have CTn,t = −ddtLdsr(θ(t)), suggesting N n=1

αn otrnn −yntrn

d dtLdsr(θ(t)) = 0. For i.i.d. inputs {ztrnn }Nn=1, with probability 1, Nn=1 αn

otrnn−yntrn ̸=

0, which means ddtLdsr(θ(t)) = 0. This does not hold when the model is yet converged. Therefore, we have the property that ∃1 ≤ n0 ≤ N,such that γn

0,t = 0.

| |
|---|

- Empirical Evidence. In Figure 10, we visualize the dynamics of the γn,t value satisfying CTn,t > 0 throughout the learning process of Perceptron and Transformer. For Perceptron, the model dimension

(128) is lower than the number of training examples (4096), which means the training dataset is redundant. Figure 10(a) shows that, given the absence of the non-contributive examples, a large fraction of γt still receives relatively small values before the model converges, which is caused by the redundancy of the training set. In Figure 10(b), we observe a similar phenomenon for Transformer, although the dimension of θt is larger than the number of training instances. We suspect the reason is that the intrinsic dimension of Transformer is usually much smaller than the dimension of θt [AGZ21], which leads to the redundancy of the training set.

- 4.6 Essence of Learning Acceleration

We investigate the essential improvement brought by the near-optimal learning policy in the perspective of the scaling laws of LMs [KMH+20], which reveals a power law between the increase of training steps and the reduction of the test loss (Ldsr(θt)) after a warming-up stage t04:

β

B t

Ldsr(θt) = L0 +

, t > t0, (8)

where (B,β) are scaling law coefficients. L0 contains the information of the model-size scaling and irreducible loss, and is assumed to be unaffected by the learning policy. In the following, we study the scaling properties of conventional and near-optimal learning processes.

4This requires the batch size to be sufficiently large [KMH+20], which is satisfied in our experiments.

[Figure 31]

𝐿 (𝜽 )Desired Loss:

Conventional LM Learning

𝐿 = 0.051 + 3.16×10 /𝑡  .  , 𝑟 = 0.995

Near-Optimal LM Learning

𝐿 = 0.051 + 1.99×10 /𝑡  .  , 𝑟 = 0.999

Training Steps 𝑡

- Figure 11: Illustration of the scaling law [KMH+20]: Ldsr(θt) = L0 + (B/t)β for conventional and near-optimal LM learning in Transformer language modeling. We fit the loss curves by the scaling law to obtain the correlation coefficient r2 and show the loss curve (solid lines) together with the fit curve (dashed lines) in a log-log plot. The scaling law fits well for both conventional and near-optimal LM learning. The near-optimal LM learning essentially improves the coefficients (B,β) in the scaling law by 96.6% and 21.2%, which shows great potential for speedup in training LLMs.

T N |∆BB| (%) |∆ββ| (%) AR

- 1K 212 88.5 10.0 2.16

- 2K 213 94.9 18.0 2.31 4K 214 93.7 18.7 2.41 8K 215 94.8 19.0 2.48

Table 2: The improvements of the scaling law coefficients brought by the near-optimal learning policy for different total training steps (T) and data sizes (N) in Transformer language modeling. The vocabulary size increases with the growth of N (see Appendix D for details). AR stands for the acceleration ratio as defined in Equation 9. The improvements hold for larger T and N.

The near-optimal learning policy improves the scaling law coefficients of LMs. In Figure 11, we fit the Transformer’s loss curves induced by the conventional and near-optimal learning policies with

- Equation 8 by setting t0 = 400 and L0 = 0.0515. We observe that the near-optimal learning process still follows the scaling law, with B and β improved by 96.6% and 21.2% respectively. Additionally, Table 2 shows that the improvement holds for the near-optimal policies found in the setting of larger T and N. We let N grow with T to ensure the sufficiency of training data [HBM+22]. The improvement of scaling law coefficients, especially β, provides significant potential in boosting the speed of LLM learning by taking advantage of power law growth. For two learning policies γ(1) and

γ(2) which induce two loss curves Ldsrγ

(θt) and Ldsrγ

(θt) with two sets of scaling law coefficients (B1,β1) and (B2,β2), the acceleration ratio of γ(2) over γ(1) is:

(1)

(2)

- β1

- β2

B

T arg min

- β1

- β2 . (9)

T1−

1 B2

=

AR =

###### (θt) ≤ Ldsrγ

Ldsrγ

(θT)

(1)

(2)

t

For an LM pre-trained for 10M steps, we will obtain more than 9× acceleration at the end of the training if the scaling property of the LM is improved as in Figure 11 and Table 2. Based on the recent experience in training LLMs [TLI+23, TMS+23], models are far from fully converged under the current training budget, which means small models (like 7B) have the potential to reach the performance of large models (like 65B), given enough training steps. However, according to Chinchilla’s law [HBM+22], extending the training steps requires more computation than enlarging

5In practice, we convert Equation 8 to ln(Ldsr(θt) − L0) = −β ln t + β ln B, (t > t0) and perform linear regression. We search for t0 and L0 to get the highest correlation coefficients.

the model to achieve a certain performance. Therefore, by optimizing the learning policy to improve learning speed, the cost of training well-performed small models can be largely reduced, which is beneficial both for open-source endeavors in the LM research community and for the efficiency of industrial products. This indicates the promise and significance of designing practical learning policy optimization approaches, and our theory can be a valuable guide.

### 5 Related Work

Improving the Learning Speed of Language Model. There is a broad range of works that propose approaches to accelerate LM learning speed such as modifying model architectures [XYH+20, ZH20] or optimizers [YLR+20, LLH+24, ZHSJ20]. There are also works studying the pre-training data programming to speed up LM convergence, such as data de-duplication [TSAM23, ATS+23], domain mixture [XPD+23], intrinsic task discovery [GHLH22], and online data selection or reordering [CRB+23, GAH23, APRW23], which can be viewed as special cases of optimizing learning policy. Unlike these works, we investigate the principles of optimizing LM learning in this paper.

Language Modeling and Lossless Compression. The recent success of LLMs calls for new interpretations beyond classical statistic learning theory for the fact that larger model sizes constantly cause better downstream generalization [NKB+19, WTB+22]. One of the interpretations is to view the next-token-prediction training process of an LM as lossless data compression [Bel19, MCKX22, Rae23]. In this perspective, larger LMs have higher compression ratios, corresponding to better modeling of data generation regularities. It is worth noting that some recent works [VNK+23, DRD+24] explore using well-trained LMs as compressors and thus the model sizes should be counted into the compressed data. Unlike these works, viewing LM training as compression does not require including the model parameters in the compressed data (see Appendix A.1 for a constructive proof) and thus is more compatible with the model size scaling law of LMs [KMH+20].

### 6 Discussion and Conclusion

Summary. In this work, we establish a theory for the optimal learning of LMs. We propose an objective that maximizes the compression ratio in an LM-training-as-losses-compression view. Then we derive a theorem, named Learning Law, suggesting that all examples should be equally contributive to the LM in the optimal learning process, which is then validated by experiments in linear classification and real-world language modeling tasks. Finally, we empirically show that the optimal learning process essentially improves the scaling law coefficients of LMs, which sheds light on future works that design practical learning acceleration approaches.

Limitations. One limitation of our work is that the experiments are conducted on relatively small scales. This is because our method to find the near-optimal learning policy corresponds to training a neural network with L × T layers, where L is the layers of the LM and T is the LM’s total training steps (see Appendix C for details). This leads to a high computational overhead when L and T scale up. However, since the theoretical derivation is generally applicable, we believe that our theory can be applied to LLMs. Another limitation is that our derivation assumes the LM is trained with full-batch GD, rather than some more commonly used techniques like mini-batch Adam [KB15]. Since these methods are essentially gradient-based, our theory can still offer insights to future LM learning acceleration studies based on these techniques [PLKS20, ABL+22].

Future Work. We believe that an important direction of future work is designing practical methods to find the optimal learning policies based on our theory for the large-scale training of LMs. Indeed, there are non-negligible challenges in this direction. Since the learning law provides a necessary condition for the learning policy’s optimality, more regularization conditions may be required to prevent sub-optimal solutions. In addition, the approach to finding the optimal learning policy should be efficient enough without contributing much to the overall computation cost. Nevertheless, our work demonstrates the promise and potential of this direction. According to recent works on LLMs training [TLI+23, TMS+23, JSM+23], the losses are still far from convergence, which means that small models have the potential to reach the similar performance as large models, but are hindered by the computation overhead brought by the large total training steps. The optimal learning policy potentially brings about a large acceleration of training with the help of the power-law growth in

- Equation 9, which makes it possible to explore the limits of LMs given (inevitably) constrained computation and train a well-performed small LM that replaces current LLMs in practice.

### References

[ABL+22] Ekin Akyurek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney, Jacob Andreas, and Kelvin Guu. Towards tracing knowledge in language models back to the training data. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Findings of EMNLP, 2022.

[ADF+23] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

[AGZ21] Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of ACL, 2021.

[APRW23] Alon Albalak, Liangming Pan, Colin Raffel, and William Yang Wang. Efficient online data mixing for language model pre-training. In NeurIPS 2023 Workshop on R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023.

[ATS+23] Amro Kamal Mohamed Abbas, Kushal Tirumala, Daniel Simig, Surya Ganguli, and Ari S Morcos. SemDeDup: Data-efficient learning at web-scale through semantic deduplication. In ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2023.

[BC11] HH Bauschke and PL Combettes. Convex Analysis and Monotone Operator Theory in

Hilbert Spaces. Springer, 2011. [Bel19] Fabrice Bellard. NNCP: Lossless data compression with neural networks, 2019. [Ber16] Dimitri Bertsekas. Nonlinear programming, volume 4. Athena scientific, 2016.

[BHA+21] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

[BMR+20] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et al. Language models are few-shot learners. In Proceedings of NeurIPS, 2020.

[CM03] Corinna Cortes and Mehryar Mohri. AUC optimization vs. error rate minimization. In Proceedings of NeurIPS, 2003.

[CMS+23] Arno Candel, Jon McKinney, Philipp Singer, Pascal Pfeiffer, Maximilian Jeblick, Prithvi Prabhu, Jeff Gambera, Mark Landry, Shivam Bansal, Ryan Chesler, et al. h2oGPT: Democratizing large language models. arXiv preprint arXiv:2306.08161, 2023.

[CND+23] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, et al. PaLM: Scaling language modeling with pathways. JMLR, 2023.

[CRB+23] Mayee F Chen, Nicholas Roberts, Kush Bhatia, Jue WANG, Ce Zhang, Frederic Sala, and Christopher Re. Skill-it! a data-driven skills framework for understanding and training language models. In Proceedings of NeurIPS, 2023.

[DRD+24] Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, et al. Language modeling is compression. In Proceddings of ICLR, 2024.

[EC21] Omer Elkabetz and Nadav Cohen. Continuous vs. discrete optimization of deep neural networks. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Proceedings of NeurIPS, 2021.

[EL23] Ronen Eldan and Yuanzhi Li. TinyStories: How small can language models be and still speak coherent english? arXiv preprint arXiv:2305.07759, 2023.

[Eng01] Andreas Engel. Statistical mechanics of learning. Cambridge University Press, 2001.

[GAH23] David Grangier, Pierre Ablin, and Awni Hannun. Bilevel optimization to learn training distributions for language modeling under domain shift. In NeurIPS 2023 Workshop on Distribution Shifts: New Frontiers with Foundation Models, 2023.

[GHLH22] Yuxian Gu, Xu Han, Zhiyuan Liu, and Minlie Huang. PPT: Pre-trained prompt tuning for few-shor learning. In Proceedings of ACL, 2022.

[GS+00] Izrail Moiseevitch Gelfand, Richard A Silverman, et al. Calculus of variations. Courier Corporation, 2000.

[HBD+20] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In Proceedings of ICLR, 2020.

[HBM+22] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

[HZD+21] Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, et al. Pre-trained models: Past, present and future. AI Open, 2021.

[HZRS16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of CVPR, 2016.

[JSM+23] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

[KB15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of ICLR, 2015.

[KEC22] Ilia Kulikov, Maksim Eremeev, and Kyunghyun Cho. Characterizing and addressing the issue of over-smoothing in neural autoregressive sequence modeling. In Proceedings of AACL, 2022.

[KL17] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of ICML, 2017.

[KMH+20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[KPA12] Celeste Kidd, Steven T Piantadosi, and Richard N Aslin. The goldilocks effect: Human infants allocate attention to visual sequences that are neither too simple nor too complex. PloS one, 7(5):e36399, 2012.

[LLH+24] Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic second-order optimizer for language model pre-training. In Proceedings of ICLR, 2024.

[LXLM23] Hong Liu, Sang Michael Xie, Zhiyuan Li, and Tengyu Ma. Same pre-training loss, better downstream: Implicit bias matters for language models. In Proceedings of ICML, 2023.

[MBR+22] Sören Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, et al. Prioritized training on points that are learnable, worth learning, and not yet learnt. In Proceedings of ICML, 2022.

[MCKX22] Yu Mao, Yufei Cui, Tei-Wei Kuo, and Chun Jason Xue. Trace: A fast transformer-based general-purpose lossless compressor. In Proceedings of WWW, New York, NY, USA, 2022.

[Met09] Janet Metcalfe. Metacognitive judgments and control of study. Current directions in psychological science, 18(3):159–163, 2009.

[MKAT18] Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.

[MP43] Warren S McCulloch and Walter Pitts. A logical calculus of the ideas immanent in nervous activity. The bulletin of mathematical biophysics, 1943.

[NKB+19] Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. Deep Double Descent: Where bigger models and more data hurt. In Proceedings of ICLR, 2019.

- [Ope22] OpenAI. OpenAI: Introducing chatgpt, 2022.
- [Ope23] OpenAI. GPT-4 technical report, 2023.

[PGM+19] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Proceedings of NeurIPS, 2019.

[PLKS20] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In NeurIPS, 2020.

[Pon18] Lev Semenovich Pontryagin. Mathematical theory of optimal processes. Routledge, 2018.

[PR12] Warren B Powell and Ilya O Ryzhov. Optimal learning, volume 841. John Wiley &

Sons, 2012. [Rae23] Jack Rae. Compression for agi, 2023. [RM87] David E. Rumelhart and James L. McClelland. Learning internal representations by error

propagation. In Parallel Distributed Processing: Explorations in the Microstructure of Cognition: Foundations, 1987.

[RRRH20] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory

optimizations toward training trillion parameter models. In Proceedings of SC20, 2020. [SDBD20] Samuel L Smith, Benoit Dherin, David Barrett, and Soham De. On the origin of implicit

regularization in stochastic gradient descent. In Proceedings of ICLR, 2020.

[TLI+23] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[TMS+23] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[TSAM23] Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari S Morcos. D4: Improving llm pretraining via document de-duplication and diversification. In Proceedings of NeurIPS, 2023.

[Vap99] Vladimir Vapnik. The nature of statistical learning theory. Springer science & business media, 1999.

[VNK+23] Chandra Shekhara Kaushik Valmeekam, Krishna Narayanan, Dileep Kalathil, JeanFrancois Chamberland, and Srinivas Shakkottai. LLMZip: Lossless text compression using large language models. arXiv preprint arXiv:2306.04050, 2023.

[VSP+17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of NeurIPS, 2017.

[WKR+19] Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural text generation with unlikelihood training. In Proceedings of ICLR, 2019.

[WSSC19] Robert C Wilson, Amitai Shenhav, Mark Straccia, and Jonathan D Cohen. The eighty five percent rule for optimal learning. Nature communications, 10(1):4646, 2019.

[WTB+22] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, et al. Emergent abilities of large language models. TMLR, 2022.

[WWL+23] Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, et al. Efficient large language models: A survey. arXiv preprint arXiv:2312.03863, 2023.

[XPD+23] Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing data mixtures speeds up language model pretraining. In Proceedings of NeurIPS, 2023.

[XSML23] Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy Liang. Data selection for language models via importance resampling. In Proceedings of NeurIPS, 2023.

[XYH+20] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In Proceedings of ICML, 2020.

[YLR+20] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. In Proceedings of ICLR, 2020.

[ZH20] Minjia Zhang and Yuxiong He. Accelerating training of transformer-based language models with progressive layer dropping. Proceedings of NeurIPS, 2020.

[ZHSJ20] Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. In Proceedings of ICLR, 2020.

### A Discussion of LM Training as Lossless Compression

##### A.1 Original View: Compressing the Training Data.

The idea of using an LM to compress data originates from the literature in the lossless text compression field [Bel19, MCKX22], and is recently adopted to interpret the essence of the next-token-predictionbased pre-training of LMs [Rae23]. We restate its core spirit by the following Theorem and the constructive proof from [Rae23]:

Theorem A.1. Consider an LM trained on a text corpus D with M tokens using mini-batch next-tokenprediction for one epoch. Let B be the number of tokens in a batch and Lt be the batch-averaged training loss at the time step t. Assume that M is divisible by B. The training process can be viewed as lossless compression of the training data. The description length of the compressed data C is

M/B

B · Lt + d(LM), (10)

d(C) =

t=1

where d(LM) is the length of the necessary code represented by a 0-1 string to run the LM training.

Proof. The basic idea of the proof is to construct a lossless encoding and decoding process for D with the LM. Let pθ

(·|w<m) be the output distribution of the LM parameterized by θt at the time step t, conditioning on the token prefix w<m = [wm−1,wm−2,··· ,w1]. For simplicity, we assume that the LM is trained using mini-batch Stochastic Gradient Decent (SGD) with a learning rate η, where each batch is linearized to a continuous list of tokens. The batch-averaged

t

training loss is Lt = −B1 Bm=1 log pθ

(wm|w<m)6. The encoding the decoding process are described in Algorithm 1 and 2. Basically, the main body of the algorithms other than the bluecolored parts implements the LM training. For encoding, the description length of a token wm is −log pθ

t

(wm|w<m) according to Arithmetic Coding 7 , and thus the compressed length of a batch W = {wm}Bm=1 is Bm=1 [−log pθ

t

(wm|w<m)] = B · Lt. d(C) equals the sum of per-batch description lengths throughout the training plus the length of the code for LM training. Therefore, we get d(C) = B · M/Bt=1 ·Lt + d(LM). For decoding, since the code for LM training is the same as that in encoding, we have θ1′ = θ1, and thus wm′ = wm for any steps in Algorithm 2, which can be easily proved by mathematical induction. As a result, D can be completely reconstructed from C, indicating the encoding (compression) is lossless.

t

| |
|---|

- Remark 1. The description length of the compressed data d(C) is approximately proportional to the area under the training loss curve (AUC) when M ≫ 1 because the size of LM training codes is much smaller than that of the compressed corpus and thus d(C) ≈ M/Bt=1 B · Lt = B · AUC.
- Remark 2. Let V be the vocabulary size of the LM and assume M ≫ 1. The corresponding compression ratio of the learning process in Theorem A.1 is CR = Mdlog(C)V ≈ M M/BlogV

t=1 B·Lt

∝ AUC1 .

As the LM fits the data, we generally have Lt < log V because log V is the loss for a randomly initialized LM. This means the compression is valid, resulting in a compression ratio CR > 1.

Altogether, Theorem A.1 bridges a connection between data compression and LM training. Generally, a higher compression ratio indicates that the compression algorithm models the underlying data knowledge better and corresponds to a better performed LM, as stated in the following remark:

- Remark 3. The LM’s ability to model the knowledge in data is characterized by the corresponding lossless compression ratio of its learning process, which is inversely proportional to the loss AUC.

Note that the model parameters are not included in the calculation of d(C), and enlarging the model sizes typically reduces the loss AUC, which explains the remarkable performance of LLMs. In addition, d(C) relates to the whole LM training process, not just the final loss. This is in line with the fact that larger LMs tend to perform better than smaller models, even if their final losses are the same [LXLM23]. This observation supports the perspective that LM training can be conceptualized

- as a process of lossless data compression.

6log(·) stands for log2(·) in the following sections. 7https://en.wikipedia.org/wiki/Arithmetic_coding

Algorithm 1 Encoding Input: Training corpus D Input: The code for LM training as a 0-1 string Output: Compressed data C: list of 0-1 strings

Algorithm 2 Decoding Input: Compressed data C: list of 0-1 string Output: Training corpus D

Get the LM training code from the first string in C Pop the first string from C Initialize D to an empty list Initialize the LM’s parameters to θ1′ for t ← 1 to M/B do

Initialize C to an empty list Append the LM training code to C Initialize the LM’s parameters to θ1 for t ← 1 to M/B do

Get a batch of tokens W = {wm}Bm=1 from the training corpus D

Get a batch of 0-1 strings S = {sm}Bm=1 from the compressed data C

for m ← 1 to B, wm ∈ W do Encode wm to a 0-1 string s with Arithmetic Coding based on pθ

for k ← 1 to B, sm ∈ S do Decode wm′ from sm with Arithmetic Coding based on pθ′

(·|w<m′ )

(·|w<m)

t

t

Append the token wm′ to D end for Lt ← −B1 Bm=1 log pθ′

Append the 0-1 string s to C end for Lt ← −B1 Bm=1 log pθ

(wm′ |w<m′ ) θt′+1 ← θt′ − η∇Lt

(wm|w<m) θt+1 ← θt − η∇Lt

t

t

end for

end for

- A.2 Our View: Compressing Data from the Desired Distribution.

Although we also focus on the loss AUC throughout the paper, our setting differs from that in Appendix A.1: (1) we assume the LM is trained with full-batch Gradient Descent (GD) for multiple epochs while Theorem A.1 lies in the scenario where the LM is trained with SGD for only one epoch; (2) we consider Ldsr computed on data other than the training examples, while pθ

t

in Equation 10 is computed on the training data. However, although not entirely rigorous, we argue that Remark 3 still holds despite the differences in (1) and (2). The reason is that: regarding (1), mini-batch SGD is an approximation of GD, which means they share the similar training dynamics when the batch size of SGD is large enough; regarding (2), just like the training loss AUC, the AUC of Ldsr can be viewed as the description length of compressing examples from the desired data distribution during the learning process. In this way, Remark 3 indicates that minimizing the AUC of Ldsr corresponds to optimizing the data compression on the desired distribution, which improves the LM’s ability to model the desired data knowledge. This is more of practical concern because in most scenarios, the model performance is measured on a dataset other than the training set, such as the validation set in classical machine learning [Vap99], the high-quality held-out corpus in large-scaling pre-training [KMH+20], or the target set in domain adaption [XSML23].

- A.3 Perceptron Training as Lossless Compression

Viewing model training as lossless compression stems from the next-token-prediction learning paradigm of LMs. We show that this perspective also fits in the one-epoch Maximum Likelihood Estimation (MLE) training of Perceptrons on the linear classification task, where the label of each example is compressed given the input vectors. Specifically, the proof in Appendix A.1 still applies if we treat linear classification as a one-step language modeling with vocabulary size V = 2. Following the notation in Section 4.2, for a Perceptron parameterized by θt at the time step t, its probability of outputting y conditioning on z is pθ

(y|z) = oy(1 − o)1−y, where o = σ(θt · z). For a batch Bt = {(zn,yn)}Bn=1, the batch-averaged loss is Lt = −B1 Bn=1 log pθ

t

(yn|zn). With Algorithm 1 and 2 applied for encoding and decoding, the description length of the compressed Bt is Bn=1 [−log pθ

t

(yn|zn)] = B · Lt, which means Theorem A.1 still holds and the discussion in Appendix A.2 also applies. For a dataset with N examples in total, the compression ratio CR ≈ N N/BlogV

t

. For a randomly initialized θ1, L1 ≈ 1, and as the model trains, Lt → 0, indicating a valid data compressing process of compression ratio CR > 1.

= N N/B t=1 B·Lt

t=1 B·Lt

### B Proof of Theorem 3.1

Theorem 3.1 essentially reflects the property of the dynamics in the learning process induced by the optimal learning policy γ(t) for the problem defined in Equation 4. We choose the accumulation

of each γn(t) over time: Γn(t) = 0 t γn(t′)dt′, as a set of free variables that θ(t) depends on to solve the optimization problem. In this way, the problem is simplified by considering a scalar that

summarizes “how much” an example is used for training until t, rather than the whole trajectory of γn(t). As such, Γ˙ n(t) = ddtΓn(t) = γn(t) and Equation 4 becomes:

T

Ldsr(θΓ,γ(t))dt,

min

Γ(t),γ(t)

0

N

(11)

Γ˙ n(t) = 1,

s.t.

n=1

Γ˙ n(t) ≥ 0,n = 1,2,··· ,N,

where Γ(t) = [Γ1(t),Γ2(t),··· ,ΓN(t)]⊤, and θΓ,γ(t) is an alias of θ(t) to show its dependency on Γ and γ. Let L be the Lagrangian depending on {Γn(t)}Nn=1 and {Γ˙ n(t)}Nn=1:

N

N

L = Ldsr(t) + λ(t)(

µn(t)Γ˙ n(t), (12)

Γ˙ n(t) − 1) +

n=1

n=1

where λ(t) and µn(t) are Lagrange multipliers and Ldsr(t) = Ldsr(θΓ,γ(t)) = Ldsr(θ(t)). To achieve the optimum of Equation 11, L should satisfy the Euler-Lagrange (EL) Equation [GS+00]:

∂L ∂Γ˙ n −

∂L ∂Γn

d dt

= 0. (13)

Together with other constraints in the Karush–Kuhn–Tucker (KKT) conditions [Ber16], we get the following formulas that characterize the optimum of the Equation 11:



∂Ldsr(t) ∂Γn

= λ˙(t) + µ˙n(t), N



Γ˙ n(t) = 1,

(14)

n=1

Γ˙ n(t) ≥ 0, µn(t) ≥ 0,



µn(t)Γ˙ n(t) = 0.

Note that we only consider the E-L Equations on {Γn(t)}Nn=1 and {Γ˙ n(t)}Nn=1, part of the free variables in the original problem, which also depends on the γ(t) trajectory. Since KKT conditions are necessary conditions to the optimization problem, Equation 14 is also necessary. In the following, we simplify Equation 14 to reach Theorem 3.1.

Simplifying Equation 14. We study the examples with non-zero weights during training. For γn(t) = Γ˙ n(t) > 0 we have µn(t) = 0 according to µn(t)Γ˙ n(t) = 0 in Equation 14 and the following Lemma bridges a connection between µn(t) and µ˙n(t):

Lemma B.1. For µn(t) ∈ C1[0,T], µn(t) = 0 ⇒ µ˙n(t) = 0 at the specific time step t.

Proof. Assuming ∃t0 ∈ [0,T], s.t. µn(t0) = 0 but µ˙n(t0) ̸= 0, we let µ˙n(t0) > 0 without loss of generality. Then ∃δt > 0, s.t. µn(t0 − δt) < 0 according to µn(t) ∈ C1[0,T], which contradicts µn(t) ≥ 0 in Equation 14. Therefore, we have µ˙n(t0) = 0.

| |
|---|

As such, γn(t) > 0 ⇒ µn(t) = 0 ⇒ µ˙n(t) = 0 (Lemma B.1), which means:

∂Ldsr(t) ∂Γm

∂Ldsr(t) ∂Γn

= λ˙(t), for γm(t) > 0 and γn(t) > 0. (15)

=

Note that λ˙(t) is independent of m and n. Equation 15 already resembles Equation 5 in their formats, if we have ∂L

dsr(t)

∂Γn ∝ ∇L · ∇ln, where ∇L = ∇Ldsr(θ(t)) and ∇ln = ∇l(xtrnn ,θ(t)).

dsr(t)

dsr(t)

Interpreting ∂L

∂Γn . ∂L

∂Γn measures how the change of Γn(t) influence the change of Ldsr(t)

- at the time step t when other free variables are fixed. Specifically, if Γn(t) changes by a small value Γn(t) → Γn(t) + ∆Γn(t), then Ldsr(t) correspondingly changes by a small value Ldsr(t) →

dsr(t)

dsr(t)

dsr(t) dt with Equation 3:

Ldsr(t) + ∆Ldsr(t), and ∂L

∂Γn = ∆L

∆Γn(t) . Then, we consider dL

dLdsr(t) dt

dθ(t) dt

= ∇Ldsr(θ(t)) ·

N

γn(t)∇Ldsr(θ(t)) · ∇l(xtrnn ,θ(t))

= −

(16)

n=1

N

dΓn(t) dt ∇L · ∇ln.

= −

n=1

As a result, for a small ∆t, we have:

N

Ldsr(t + ∆t) − Ldsr(t) = −

[Γn(t + ∆t) − Γn(t)]∇L · ∇ln. (17)

n=1

Now we consider the change of Ldsr(t) and Γn at t + ∆t. Since ∇L · ∇ln is computed at the time step t, it is not affected by the variants. Therefore, we get:

∆Ldsr(t + ∆t) = −∆Γn(t + ∆t)∇L · ∇ln, (18) When ∆t → 0, ∆L

dsr(t)

dsr(t+∆t)

dsr

∆Γn(t+∆t) → ∆L

∂Γn , which means: ∂Ldsr(t) ∂Γn

∆Γn(t) = ∂L

= −∇L · ∇ln. (19)

By substituting Equation 19 into Equation 15, we obtain that for the mth and nth training examples satisfying γm(t) > 0 and γn(t) > 0 the following equation holds:

∇L · ∇lm = ∇L · ∇ln = −λ˙(t) = Const, (20) where Const stands for “a constant independent of m and n”. Equation 20 is essentially equivalent to Equation 5.

dsr(t) dt . By substituting ∇L · ∇ln with Const in Equation 16, we get:

Proving Const = −dL

N

dLdsr(t) dt

dΓn(t) dt · Const,

= −

n=1

(21)

N

= −Const

γn(t),

n=1

= −Const.

As such, by combining Equation 20 with Equation 21, we complete the proof of Theorem 3.1.

### C Details of Learning Policy Optimization

In Section 4.1, we search for the optimal learning policy by Proximal Gradient Decent [BC11]. Specifically, we view the whole learning process in 0 ≤ t ≤ T as a neural network with T layers parameterized by γ = [γ0,··· ,γt−1] ∈ RN×T. As illustrated in Figure 12, each layer of the network consists of the gradient update function and a residual connection [HZRS16], where the “hidden states” are θt. Then, we adopt Backward Propagation (BP; RM87) to compute ∇γt

J(γ) in Equation

6. The backward operation at each layer is:

T

∂J ∂γt

∂θt′ ∂θt+1

∇Ldsr(θt′)

##### Gtrn(θt)

= −η

(22)

t′=t+1

∂θt′ ∂θt+1

∂θt′ ∂θt+2

I − ηHtrn(θt+1) ,

=

| | |
|---|---|
| | |

······

| |
|---|

+

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Figure 12: The architecture of the equivalent neural network to find the optimal learning policy, Each layer consists of the gradient update and a residual connection.

where Gtrn(θt) = [∇l(xtrn1 ,θt),··· ,∇l(xtrnN ,θt)] , I is the identity matrix, and Htrn(θt+1) is the Hessain matrix of Ltrn(θ) at θ = θt+1. We implement the BP operations with dynamic programming and Jacobian-Vector-Product8 in PyTorch [PGM+19] for efficiency. To reduce the single-device GPU memory use, we also implement an activation partition algorithm inspired by ZeRO-2 [RRRH20], where the “hidden states” θt in one model are stored in different GPU devices.

### D Hyper-Parameter Configurations

Perceptron Linear Classification. Following the teacher-setting described in Section 4.2, we use D = 128 and T is randomly drawn from a Gaussian distribution T ∼ N(0,

√

DI). We generate N = 4096 training inputs ztrn from N(0,3I), and M = 512 target inputs zdsr from N(0.51,I), where 1 = [1,1,··· ,1]⊤ ∈ RD. For each learning policy, we initialize γn,0 = N1 and train the Perceptron with η = 0.1 for T = 2000 time steps, which is sufficient for the model to converge. For learning policy optimization, we initialize the learning policy to the constant policy γn,tc = N1 , setting ϵ = 5 × 10−6 and train the network for 500 epochs.

Transformer Language Modeling. We conduct experiments based on a two-layer Transformer with 128 hidden dimensions and 8 attention heads. For all experiments except that in Table 2, we randomly sample N = 16,384 examples as xtrnn and K = 512 examples as xdsrk with the max sequence length 64 from the TinyStories [EL23] corpus9. We use the BPE tokenizer of Mistral [JSM+23] and construct a vocabulary with 5K tokens by mapping the infrequent tokens to [UNK]. The model contains about 1.7M parameters. To reflect the difference between the training and desired distribution, we add perturbations to 50% training sentences by iteratively applying one of the following operations 20 times: (1) replacing one token with a random token in the vocabulary [HBD+20]; (2) deleting the last token [KEC22]; (3) repeating one token in a sentence [WKR+19]. This corresponds to the fact that the large-scale pre-training corpus tends to be more noisy than the desired set (the carefully curated held-out corpus or high-quality downstream data) to evaluate the model generalization performance in practice. We set η = 0.1, T = 4,000, γn,0 = N1 for each learning policy. We start from the constant policy and optimize the learning policy for 15 epochs using η = 0.1,0.2,0.4. η = 0.4 yields the lowest loss at the end of the training. Therefore, we only plot the optimization process for η = 0.4 in Figure 4(b) and 5(b). For experiments in Table 2, we vary the total training steps and the corresponding training data sizes and simultaneously, change the vocabulary sizes to adapt to different data sizes. We use vocabulary sizes 4K, 4.5K, 5K, and 6K for N = 212,213,214, and 215, respectively.

- 8https://pytorch.org/docs/stable/func.api.html
- 9https://huggingface.co/datasets/roneneldan/TinyStories/tree/main

