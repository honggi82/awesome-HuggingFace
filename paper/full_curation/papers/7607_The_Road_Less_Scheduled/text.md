# arXiv:2405.15682v4[cs.LG]29Oct2024

## The Road Less Scheduled

Aaron Defazio1 Fundamental AI Research Team, Meta

Xingyu (Alice) Yang2 Fundamental AI Research Team, Meta

Harsh Mehta Google Research

Konstantin Mishchenko Samsung AI Center

Ahmed Khaled Princeton University

Ashok Cutkosky3 Boston University 1Research Co-lead 2 Engineering Co-lead 3 Senior Author

### Abstract

Existing learning rate schedules that do not require specification of the optimization stopping step T are greatly out-performed by learning rate schedules that depend on T. We propose an approach that avoids the need for this stopping time by eschewing the use of schedules entirely, while exhibiting state-of-the-art performance compared to schedules across a wide family of problems ranging from convex problems to large-scale deep learning problems. Our Schedule-Free approach introduces no additional hyper-parameters over standard optimizers with momentum. Our method is a direct consequence of a new theory we develop that unifies scheduling and iterate averaging. An open source implementation of our method is available1. Schedule-Free AdamW is the core algorithm behind our winning entry to the MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge Self-Tuning track.

### 1 Introduction

The theory of optimization, as applied in machine learning, has been successful at providing precise, prescriptive results for many problems. However, even in the simplest setting of stochastic gradient descent (SGD) applied to convex Lipschitz functions, there are glaring gaps between what our current theory prescribes and the methods used in practice.

Consider the stochastic gradient descent (SGD) step with step size γ > 0, zt+1 = zt − γgt where gt is the stochastic (sub-)gradient at time t, computed at the point zt (formally defined in Section 1.2) of a convex Lipschitz function f. Although standard practice for many classes of problems, classical convergence theory suggests that the expected loss of this z sequence is suboptimal, and that the Polyak-Ruppert (PR) average x of the sequence should be returned instead (Polyak, 1990; Ruppert, 1988):

zt+1 = zt − γgt (1) xt+1 = (1 − ct+1)xt + ct+1zt+1, (2)

where using ct+1 = 1/(t + 1) results in xt = T1 Tt=1 zt. Despite their theoretical optimality, PR averages give much worse results in practice than using the last-iterate of SGD (Figures 2,11) — a

folk-law result in the field of optimization, and a large theory-practice gap that is often attributed to

1https://github.com/facebookresearch/schedule_free

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

ILSVRC 2012 ImageNet (ResNet-50)

OpenWebText (GPT-2 124M)

3.2

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

TestAccuracy(%)

3.1

60

TestLoss

3.0

40

2.9

20

2.8

0 20 40 60 80 100 Epoch

0 200000 400000 600000 Step

- Figure 1: Schedule-Free methods (black) closely track the Pareto frontier of loss v.s. training time in a single run. Both Schedule-Free SGD (left) and AdamW (right) match or exceed the performance of cosine learning rate schedules of varying lengths (red).

the mismatch between this simplified problem class and the complexity of problems addressed in practice.

Recently, Zamani and Glineur (2023) and Defazio et al. (2023) showed that the exact worst-case optimal rates can be achieved via carefully chosen learning rate sequences (also known as schedules) alone, without the use of averaging. This result suggests that schedules have, in some sense, the same role to play as PR averaging in optimization. However, schedules have a critical disadvantage: they require setting the optimization stopping time T in advance.

Motivated by the theory-practice gap for Polyak-Ruppert averaging, we ask the following question:

Do there exist iterate averaging approaches that match the empirical performance of learning rate schedules, without sacrificing theoretical guarantees?

By developing a new link between averaging and learning rate sequences, we introduce a new approach to averaging that maintains the worst-case convergence rate theory of PR averaging, while matching and often exceeding the performance of schedule-based approaches – firmly answering this question in the affirmative.

#### 1.1 Summary of Results

- • Our approach does not require the stopping time T to be known or set in advance. It closely tracks the Pareto frontier of loss versus training time during a single training run (Figure 1), while requiring no additional hyper-parameters over the base SGD (with momentum) or Adam optimizer.
- • Our approach uses an alternative form of momentum that replaces traditional momentum. This form has appealing theoretical properties: it is worst case optimal for any choice of the momentum parameter in the convex Lipschitz setting, a property that does not hold for traditional momentum.
- • Our key theoretical result is a new online-to-batch conversion theorem, which establishes the optimality of our method while also unifying several existing online-to-batch theorems.
- • We perform, to our knowledge, one of the largest machine learning optimization algorithm evaluations to date, consisting of 28 problems, ranging from logistic regression to large-scale deep learning problems. This evaluation contains more distinct and diverse largescale machine-learning problems than any other optimizer evaluation we are aware of in the literature. Schedule-Free methods show strong performance, matching or out-performing heavily-tuned cosine schedules.
- • Schedule-Free AdamW won the MLCommons 20204 AlgoPerf Algorithmic Efficiency Challenge Self-Tuning track, providing independent verification of it’s state-of-the-art performance against other optimization algorithms when hyperparameter-tuning is limited. We provide details of our entry and plots comparing it to the competition baseline.

#### 1.2 Notation

Consider the stochastic convex minimization minx∈Rd f(x) = Eζ[f(x,ζ)], where each f(x,ζ) is Lipschitz and convex in x, and the expectation is taken over the random variable ζ. With a slight abuse of notation, we assume we are given, at time step t and any point y that we choose, an arbitrary sub-gradient ∇f(y,ζt) from the sub-differential of f.

### 2 Method

We propose the following method, which we call Schedule-Free SGD:

yt = (1 − β)zt + βxt, (3) zt+1 = zt − γ∇f(yt,ζt), (4) xt+1 = (1 − ct+1)xt + ct+1zt+1, (5)

- where ct+1 = 1/(t + 1) and z1 = x1 is the initial point. Note that with this weighting, the x sequence is just an online equal-weighted average of the z sequence. The y sequence is the gradient location sequence (on which gradients are evaluated at each step) and the x sequence is the evaluation sequence, our current best estimate of the parameters. The z sequence is the base sequence, which is where the base optimizer’s update is performed (in this case SGD).

This method has a momentum parameter β that interpolates between Polyak-Ruppert averaging (β = 0) and Primal averaging (β = 1). Primal averaging (Nesterov and Shikhman, 2015; Tao et al., 2018; Cutkosky, 2019; Kavis et al., 2019; Sebbouh et al., 2021; Defazio and Gower, 2021; Defazio and Jelassi, 2022), is an approach where the gradient is evaluated at the averaged point x, instead of z:

zt+1 = zt − γ∇f(xt,ζt) (6) xt+1 = (1 − ct+1)xt + ct+1zt+1, (7)

this approach maintains the worst-case optimality of PR averaging but is generally considered to converge too slowly to be practical (Figures 2,11). The advantage of our interpolation is that we get the best of both worlds. We can achieve the fast convergence of Polyak-Ruppert averaging (since the z sequence moves much quicker than the x sequence), while still keeping some coupling between the returned sequence x and the gradient-evaluation locations y, which increases stability. Values of β similar to standard momentum values β ≈ 0.9 appear to work well in practice. We will use the notation α = 1 − β when convenient.

In this formulation, β = 0.9 gives the practical advantages of momentum, dampening the immediate impact of large gradients, resulting in more stable training. To see this, notice that the immediate effect of the gradient gt at step t is to introduce (1 − β)gt = 0.1gt into the iterate sequence y. This is similar to exponential-moving-average (EMA) momentum, where also (1 − β)gt is added into the iterate sequence on step t. However, here the remainder of gt is very slowly added into y over time, via its place in the average x, whereas with an EMA with β = 0.9, the majority of the gradient is incorporated within the next 10 steps. So from this viewpoint, the Schedule-Free updates can be seen as a version of momentum that has the same immediate effect, but with a greater delay for adding in the remainder of the gradient. This form of momentum (by interpolation) also has a striking advantage: it does not result in any theoretical slowdown; it gives the optimal worst case (Nesterov, 2013) convergence for the non-smooth convex setting (including constants), for any choice of momentum β between 0 and 1 inclusive:

- Theorem 1. Suppose F is a convex function, and ζ1,...,ζT is an i.i.d. sequence of random variables such that F = E[f(x,ζ)] for some function f that is G-Lipschitz in x. For any minimizer x⋆, define

√

D = ∥x1 − x⋆∥ and γ = D/(G

T). Then for any β ∈ [0,1], Schedule-Free SGD ensures:

DG √

(8)

E[F(xT) − F(x⋆)] ≤

T

In contrast, exponential-moving-average momentum in the non-smooth setting actually hurts the theoretical worst-case convergence rate. The Schedule-Free approach maintains the advantages of momentum (Sutskever et al., 2013) without the potential worst-case slow-down.

IWSLT14 (LSTM)

5.5

| | | | |Pol<br><br>Prim|yak Avera al Avera|ging (4.35 ging (4.44|SE 0.004 SE 0.001|) )| |
|---|---|---|---|---|---|---|---|---|---|
| | | | |Sch Cos<br><br>|edule-Fre ine Sched|e (4.18 SE ule (4.23|0.001) SE 0.003)| | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

5.0

TestLoss

4.5

4.0

0 10000 20000 30000 40000 50000 60000 Step

- Figure 2: Schedule-Free learning converges faster than classical averaging approaches, often outperforming tuned schedules. Existing averaging approaches such as Polyak and Primal averaging significantly under-perform schedules.

#### 2.1 General Theory

The method analyzed in Theorem 1 is actually a special-case of a more general result that incorporates arbitrary online optimization algorithms rather than only SGD, as well as arbitrary time-varying sequences of βt. The proof is provided in Appendix A.

- Theorem 2. Let F be a convex function. Let ζ1,...,ζT be an iid sequence such that F(x) = Eζ[f(x,ζ)]. Let z1,...,zT be arbitrary vectors and let w1,...,wT and β1,...,βT be arbitrary numbers in [0,1] such that zt, wt and βt are independent of ζt,...,ζT. Set:

t i=1 wizi

wt

wt

zt (9)

= xt−1 1 −

xt =

+

t i=1 wi

t i=1 wi

t i=1 wi

≜ct

≜1−ct

yt = βtxt + (1 − βt)zt (10) gt = ∇f(yt,ζt). (11)

Then we have for all x⋆:

E[ Tt=1 wt⟨gt,zt − x⋆⟩] T i=1 wi

. (12)

E[F(xT) − F(x⋆)] ≤

To recover Theorem 1 from the above result, notice that the algorithm analyzed by Theorem 1 is captured by Theorem 2 with wt = 1, βt a constant β and zt+1 = zt − γgt for all t. Next, observe that the sequence z1,...,zT is performing online gradient descent (Zinkevich, 2003), for which it is well-known that the regret Tt=1⟨gt,zt − x⋆⟩ (appearing in the numerator of our result) is bounded by DG

√

T and so the result of Theorem 1 immediately follows.

The regret is the principle object of study in online convex optimization (Hazan, 2022; Orabona, 2019). Viewed in this light, Theorem 2 provides a way to convert an online convex optimization algorithm into a stochastic optimization algorithm: it is a form of online-to-batch conversion (Cesa-Bianchi et al., 2004). Classical online-to-batch conversions are a standard technique for obtaining convergence bounds for many stochastic optimization algorithms, including stochastic gradient descent (Zinkevich, 2003), AdaGrad (Duchi et al., 2011), AMSGrad (Reddi et al., 2018), and Adam (Kingma and Ba, 2014). All of these algorithms can be analyzed as online convex optimization algorithms: they

provide bounds on the regret Tt=1⟨gt,zt − x⋆⟩ rather than direct convergence guarantees. It is then necessary (although sometimes left unstated) to convert these regret bounds into stochastic

convergence guarantees via an online-to-batch conversion. Our result provides a more versatile method for effecting this conversion.

- Theorem 2 actually provides a “grand unification” of a number of different online-to-batch conversions that have been proposed over the years. Most of these conversion methods were first developed

specifically to provide convergence analysis for SGD (or some variant such as dual averaging or mirror descent), and then generalized into techniques that apply to any online convex optimization algorithm. For example, the classical Polyak averaging method can be generalized to form the “standard” online-to-batch conversion of Cesa-Bianchi et al. (2004), and is immediately recovered from Theorem 2 by setting wt = 1 and βt = 0 for all t. More recently Nesterov and Shikhman (2015); Tao et al. (2018) derived an alternative to Polyak averaging that was later generalized to work with arbitrarily online convex optimization algorithms by Cutkosky (2019); Kavis et al. (2019), and then observed to actually be equivalent to the heavy-ball momentum by Defazio (2020); Defazio and Gower (2021); Defazio and Jelassi (2022). This method is recovered by our Theorem 2 by setting wt = 1 and βt = 1 for all t. Finally, very recently Zamani and Glineur (2023) discovered that gradient descent with a linear decay stepsize provides a last-iterate convergence guarantee, which was again generalized to an online-to-batch conversion by Defazio et al. (2023). This final result is also recovered by Theorem 2 by setting wt = 1 and βt = Tt (see Appendix B).

In Appendix C, we give a further tightening of Theorem 2 – it can be improved to an equality by precisely tracking additional terms that appear on the right-hand-side. This tightened version can be used to show convergence rate results for smooth losses, both with and without strong-convexity. As an example application, we show that schedule-free optimistic-gradient methods (Rakhlin and Sridharan, 2013) converge with accelerated rates:

E[F(xT) − F(x⋆)] = O

D2L T2

Dσ √

+

T

. (13)

#### 2.2 On Large Learning Rates

Under classical worst-case convergence theory, the optimal choice of γ for a fixed duration training time T is γ = D/(G

√

T). This is the rate used in our bounds for Theorem 1 above. For any-time convergence (i.e. when stopping is allowed at any timestep), our proposed method can, in theory, be used with the standard learning rate sequence:

D G√t

. (14)

γt =

However, learning rate sequences of this form have poor practical performance (Defazio et al., 2023). Instead, much larger steps of the form D/G give far better performance across virtually all problems in applications (Defazio and Mishchenko, 2023) — another theory-practice mismatch that is virtually undiscussed in the literature. Existing theory suggests that this step-size is too large to give O(1/

√

T) convergence, however, as we show below, there is an important special case where such large step sizes also give optimal rates up to constant factors.

- Theorem 3. Consider the online learning setting with bounded gradients gt. Let zt+1 = zt − γgt. Let D = ∥z1 − z∗∥ for arbitrary reference point z∗ and define G = maxt≤T ∥gt∥. Suppose that the chosen step-size is γ = D/G, then if it holds that:

T

⟨gt,zt − z1⟩ ≤ D

t=1

T

∥gt∥2, (15)

t=1

then:

 D

 . (16)

T

T

1 T

∥gt∥2

⟨gt,zt − z∗⟩ = O

T

t=1

t=1

This regret bound for SGD implies a convergence rate bound for Schedule-Free SGD by application of our online-to-batch conversion. Condition 31 can be checked during a training run (Using reference point z∗ = xT, and so D = ∥x1 − xT∥), and we find that it holds for every problem we consider in our experiments in Section 4.1. More generally, the full conditions under which large learning rates can be used are not yet fully understood for stochastic problems. In the quadratic case, Bach and Moulines (2013) established that large fixed step-sizes give optimal convergence rates, and we conjecture that the success of large learning rates may be attributed to asymptotic quadratic behavior of the learning process.

###### Minimal Loss for Different and values.

0.700

200

| |[Figure 1]| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

|[Figure 2]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

175

0.775

150

125

MinimalLoss

0.851

100

75

0.926

50

25

0.01 25.13 50.26 75.38

- Figure 3: Incorporating the momentum parameter β allows for convergence despite using larger learning rates γ on quadratic problems. Dark region indicates convergence.

Empirically, we find that Schedule-Free momentum enables the use of larger learning rates γ > 0 even in quadratic minimization problems f(x) = 12x⊤Ax − b⊤x. We generate 10 different such 20-dimensional problems with eigenvalues drawn log-uniformly in [10−6,1]. We plot the average minimal loss achieved as a function of the two parameters β and γ in Figure 3. We can see that when the learning rate we use is small, what value of β we choose has little to no effect on the convergence of the algorithm. However, when γ is large, choosing β < 1 becomes crucial to achieving convergence.

- 2.3 How Schedule-Free Replaces a Schedule

- Figure 4 illustrates the link between Polyak Averaging, Primal Averaging, Schedule-Free methods and the Linear decay schedule. Each of the averaging approaches have a crucial relationship to (worst-case optimal) Linear Decay schedules: their x sequences simulate the effect of a Linear Decay schedule that ends at the current time-step t, rather than the final time-step T. This is why they can replace a schedule – they are implicitly applying one.

The four approaches differ primarily in how much each gradient contributes to the gradient location sequence y where gradients are evaluated at each time-step, which greatly effects the stability and convergence of each method. Polyak averaging resembles the linear decay schedule’s behavior at the beginning, but near the end of training the y sequence (blue) has much larger contributions from recent gradients. Primal averaging behaves in the opposite fashion – near the end of training it behaves similar to the Linear Decay schedules as gradients enter with a small weight, but at the early stages it significantly down-weights recent gradients, resulting in slow convergence. Schedule-Free (shown here with β = 0.6) remedies this down-side of Primal Averaging by modestly boosting the contribution of more recent gradients throughout all stages of training.

### 3 Related Work

The proposed method has a striking resemblance to Nesterov’s accelerated method (Nesterov, 1983,

2013) for L-smooth functions, which can be written in the AC-SA form (Lan, 2012):

yt = (1 − ct+1)xt + ct+1zt (17) zt+1 = zt −

k + 1

2L ∇f(yt) (18) xt+1 = (1 − ct+1)xt + ct+1zt+1, (19)

1/3 through Training

2/3 through Training

End of Training

1.0

1.0

1.0

| |
|---|

| |
|---|

0.8

0.8

0.8

0.6

0.6

0.6

Linear Decay Schedule

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0% 100%

0% 100%

0% 100%

1.0

1.0

1.0

0.8

0.8

0.8

Polyak Averaging y has no decay (blue) x behaves like linear decay to t (orange)

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0% 100%

0% 100%

0% 100%

1.0

1.0

1.0

| |
|---|

0.8

0.8

0.8

0.6

0.6

0.6

Primal Averaging x&y behave like linear decay to t

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0% 100%

0% 100%

0% 100%

1.0

1.0

1.0

0.8

0.8

0.8

Schedule-Free

y sequence has a larger contribution from recent gradients than for primal averaging (blue)

0.6

0.6

0.6

0.4

0.4

0.4

x behaves like linear decay to t (orange)

0.2

0.2

0.2

0.0

0.0

0.0

0% 100% T

0% 100% T

0% 100% T

Figure 4: Illustration of the contribution of the gradient at each time step to the gradient location sequence y and the returned evaluation sequence x. The horizontal axis is the time-step, and the vertical axis is the fraction of the gradient from each time-step incorporated into the iterate sequence.

- where ct+1 = 2/(t + 2). The averaging constant, and more generally

r + 1 t + r + 1

, (20)

ct+1 =

for any real r > −1 is equivalent to the weighted average (Shamir and Zhang, 2013; Defazio and Gower, 2021) xt ∝ Tt=1 tr¯zt, where tr¯ represents the rth factorial power of t. Our framework is compatible with factorial power averages without sacrificing theoretical guarantees.

Our approach differs from conventional accelerated methods by using a different weight for the yt and xt interpolations. We use a constant weight for yt and a decreasing weight for xt. Accelerated methods for strongly-convex problems use a constant weight for both, and those for non-strongly convex use an decreasing weight for both, so our approach doesn’t directly correspond to either class of accelerated method. Accelerated methods also use a much larger step size for the zt sequence than our approach.

The use of equal-weighted averages is less common than the use of exponential weighting in the practical deep learning optimization literature. Exponential moving averages (EMA) of the iterate sequence are used in the popular Lookahead optimizer (Zhang et al., 2019). In the case of SGD, it performs i = 1...k inner steps:

zt,i = zt,i−1 − γ∇f(zt,i−1) (21) followed by an outer step:

##### xt = xt−1 + α (zt,k − xt−1). (22)

The inner optimizer then starts at zt+1,0 = xt−1. The Lookahead method can be seen as the EMA version of primal averaging, just as exponential weight averaging is the EMA version of Polyak-Ruppert averaging.

Tail averaging, either using an exponential moving average or an equal-weighted average, is a common ‘folk-law’ technique that often yields a practical improvement. For instance, this kind of averaging is used without citation by the influential work of Szegedy et al. (2016): “Model evaluations are performed using a running average of the parameters computed over time.”, and by Vaswani et al. (2017): “...averaged the last 20 checkpoints”. Tail averages are typically “Polyak-Ruppert” style averaging as the average is not used for gradient evaluations during training.

More sophisticated tail averaging approaches such as Stochastic Weight Averaging (Izmailov et al., 2018) and LAtest Weight Averaging (Kaddour, 2022; Sanyal et al., 2023) combine averaging with large or cyclic learning rates. They are not a replacement for scheduling, instead they aim to improve final test metrics. They generally introduce additional hyper-parameters to tune, and require additional memory. It is possible to use SWA and LAWA on top of our approach, potentially giving further gains.

Sandler et al. (2023) show via a stochastic quadratic analysis framework that averaging and learning rate decreases achieve the same effective learning rate. For instance, and average of two points along the training trajectory can give almost identical results to using a learning rate two times smaller. Stochastic quadratic problems are particularly special, Bach and Moulines (2013) have shown that Polyak averaging gives optimal O(1/T) rates without the use of decreasing time-dependent step size sequences in this setting.

Within optimization theory, tail averages can be used to improve the convergence rate for stochastic non-smooth SGD in the strongly convex setting from O(log(T)/T) to O(1/T) (Rakhlin et al., 2012), although at the expense of worse constants compared to using weighted averages of the whole sequence (Lacoste-Julien et al., 2012).

Portes et al. (2022) use cyclic learning rate schedules with increasing cycle periods to give a method that explores multiple points along the Pareto frontier of training time vs eval performance. Each point at the end of a cycle is an approximation to the model from a tuned schedule ending at that time. Our method gives the entire frontier, rather than just a few points along the path. In addition, our method matches or improves upon best known schedules, whereas the “... cyclic trade-off curve underestimated the standard trade-off curve by a margin of 0.5% validation accuracy” (Portes et al., 2022).

### 4 Experiments

To validate the effectiveness of our method, we performed a large-scale comparison across multiple domains (computer vision, language, and categorical data) and covering a range of small scale to large-scale experiments (logistic regression to large language model training). Details of the implementation of our method for SGD and Adam used in the experiments are in Section 4.4.

#### 4.1 Deep Learning Problems

For our deep learning experiments, we evaluated Schedule-Free learning on a set benchmark tasks that are commonly used in the optimization research literature:

CIFAR10 A Wide ResNet (16-8) architecture (Zagoruyko and Komodakis, 2016) on the CIFAR10

image classification dataset.

CIFAR100 A DenseNet (Huang et al., 2017) architecture on the CIFAR-100 (100-class) classification

dataset. SVHN A deep ResNet architecture (3-96) on the Street View House Numbers (SVHN) dataset. ImageNet A standard ResNet-50 architecture (He et al., 2016) on the ILSVRC 2012 ImageNet

(Russakovsky et al., 2015) classification dataset.

IWSLT14 A LSTM architecture (Wiseman and Rush, 2016) on the IWSLT14 German-English

translation dataset (Cettolo et al., 2014).

DLRM The DLRM (Naumov et al., 2019) architecture on the Criteo Kaggle Display Advertising

dataset (Jean-Baptiste Tien, 2014).

MRI A stacked U-Net architecture (Sriram et al., 2020) on the fastMRI dataset (Zbontar et al., 2018). MAE Fine-tuning a pretrained Masked Autoencoder (He et al., 2021) ViT (patch16-512d-8b) on the

ILSVRC 2012 ImageNet dataset.

CIFAR-10 (WRN-16-8)

CIFAR-100 (DenseNet)

80

95

TestAccuracy(%)

TestAccuracy(%)

70

90

60

85

Schedule-Free (96.03% SE 0.04)

Schedule-Free (78.71% SE 0.06)

Step-Wise Schedule (95.59% SE 0.03)

Step-Wise Schedule (76.41% SE 0.14)

50

80

Cosine Schedule (95.73% SE 0.04)

Cosine Schedule (77.41% SE 0.09)

75

40

0 50 100 150 200 250 300 Epoch

0 50 100 150 200 250 300 Epoch

SVHN (ResNet-3-96)

ILSVRC 2012 ImageNet (ResNet-50)

- 94

- 95

- 96

- 97

- 98

- 99

TestAccuracy(%)

TestAccuracy(%)

70

60

Schedule-Free (98.40% SE 0.01)

Schedule-Free (76.90% SE 0.03)

Cosine Schedule (98.27% SE 0.02)

50

Cosine Schedule (76.90% SE 0.06)

Step-Wise Schedule (98.20% SE 0.01)

Step-Wise Schedule (76.49% SE 0.07)

40

0 50 100 150 200 250 300 Epoch

0 20 40 60 80 100 Epoch

fastMRI Knee (VarNet 2.0)

Criteo Kaggle (DLRM)

- 0.912

0.790

TestAccuracy

0.910

TestSSIM

0.785

0.908

Cosine Schedule (0.9110 SE 0.00016)

LD Schedule (0.7906 SE 0.00006)

0.780

0.906

Schedule-Free (0.9112 SE 0.00012)

Schedule-Free (0.7915 SE 0.00003)

0 10 20 30 40 50 Epoch

0 20 40 60 80 100 Epoch

MAE ImageNet Finetune (ViT)

OpenWebText (GPT-2 124M)

3.2

| | | |Cosine Sched Schedule-Fre<br><br>|ule (2.853 SE 0.004 e (2.831 SE 0.008)|)| |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

TestAccuracy(%)

80

3.1

TestLoss

75

3.0

70

Schedule-Free (83.54 SE 0.03)

2.9

Cosine Schedule (83.52 SE 0.02)

65

2.8

0 20 40 60 80 100 Epoch

0 200000 400000 600000 Step

Figure 5: Deep Learning Experiments

NanoGPT A 124M parameter GPT-2 (Radford et al., 2019) style decoder-only transformer on the

OpenWebText dataset (Gokaslan and Cohen, 2019).

For each problem, both the baseline and the Schedule-Free method were tuned by sweeping both the weight decay and learning rate on a grid. We also swept β over two values, 0.9 and 0.98. Final hyper-parameters are listed in the Appendix. Schedule-Free SGD was used for CIFAR10, CIFAR100, SVHN and ImageNet, and Schedule-Free AdamW (Loshchilov and Hutter, 2019) was used for the remaining tasks. We further include a step-wise schedule as a comparison on problems where step-wise schedules are customary. Further results for Polyak and Primal averaging are in Appendix H.

Our approach shows very strong performance (Figure 5) out-performing existing state-of-the-art cosine schedules on CIFAR-10, CIFAR-100, SVHN, IWSLT-14 (Figure 2) and OpenWebText GPT-2 problems, as well as the state-of-the-art Linear Decay schedules on the fastMRI and Criteo DLRM tasks. On the remaining two problems, MAE fine-tuning and ImageNet ResNet-50 training, it ties with the existing best schedules.

In general, the optimal learning rates for the Schedule-Free variants were larger than the optimal values for the base optimizers. The ability to use larger learning rates without diverging may be a contributing factor to the faster convergence of Schedule-Free methods. The β parameter works well at the default value of 0.9 for all problems except NanoGPT, where the loss started to increase rapidly when 0.9 was used (similar to the Polyak Averaging results in Appendix H). The larger β = 0.98 value in our sweep was stable.

CIFAR-10 Training Loss

CIFAR-10 Gradient Norm

- 0

- 1

- 2

- 3

SGD

SGD

Schedule-Free SGD

Schedule-Free SGD

GradientNorm

1.0

Loss

0.5

0.0

0 20 40 60 80 100 Iteration %

0 20 40 60 80 100 Iteration %

- Figure 6: Schedule-Free learning doesn’t exhibit the same gradient collapse during training time as schedules-based approaches do (Cosine schedule shown).

Schedule-Free learning works particularly well on problems that are prone to gradient norm collapse during training. As an example, when training CIFAR-10 using SGD with traditional schedule-based approaches, at the latter stages of training the training-loss goes to near-zero and the gradient norms like-wise collapse to near-zero (Figure 6). From this point on learning slows significantly. This happens even when weight decay is optimally tuned to give the best final test accuracy. Schedule-Free SGD in contrast does not show a collapse of either the training loss or gradient norm sequence, and continues to reliably learn. This lack-of-collapse is likely a contributing factor to the particularly good performance of Schedule-Free learning on CIFAR-10, CIFAR-100 and SVHN in our experiments.

#### 4.2 MLCommons Algorithmic Efficiency benchmark

The AlgoPerf challenge (Dahl et al., 2023) is designed to be a large-scale and comprehensive benchmark for deep learning optimization algorithms, covering major data domains and architectures. It includes Transformers, ConvNets and U-Net models across image, language, graph and speech domains, and contains 8 problems total. We evaluated Schedule-Free AdamW following the competition guidelines, comparing against NAdamW, the competition reference Algorithm, running 10 seeds of each. As this is a time-to-target competition, traditional error bars are not appropriate so we instead plot all 10 seeds separately. Note that we excluded one benchmark problem, ResNet-50 training, as neither AdamW nor NAdamW can hit the target accuracy on that task. The remaining tasks are:

WMT A Encoder-Decoder Transformer Model on the WMT17 German-to-english translation task

(Bojar et al., 2017).

VIT A S/16 Vision Transformer (Dehghani et al., 2023) model on the ILSVRC 2012 ImageNet

classification task (Russakovsky et al., 2015).

FASTMRI The reference U-Net architecture from the fastMRI challenge Knee MRI dataset (Zbontar

et al., 2018).

CONFORMER A Conformer (Gulati et al., 2020) Speech Recognition model on the LibriSpeech

ASR dataset (Panayotov et al., 2015).

OGBG A Graph-Neural Network in the style of Battaglia et al. (2018) on a Molecular property prediction task from the Open Graph Benchmark (Hu et al., 2020) suite (PubChem BioAssay data).

CRITEO Clickthrough-rate prediction on the criteo 1B dataset (Criteo, 2022) using the Deep

Learning Recommendation Model (DLRM) architecture. DEEPSPEECH The Deep Speech model on the LibriSpeech ASR dataset.

The self-tuning track restricts participants to provide a single set of hyper-parameters to use for all 8 problems. Given the large number of problems, this gives performance representative of a good default configuration.

Schedule-Free AdamW performs well across all considered tasks, out-performing the baseline on the WMT, VIT, FASTMRI and OGBG training, while tying on the Conformer and Criteo workloads, and marginally under-performing on the DeepSpeech workload. We attribute the performance on the Conformer and DeepSpeech tasks to their use of batch-norm - the AlgoPerf setup doesn’t easily allow us to update the BN running statistics on the x sequence, which is necessary with our method to get the best performance (See Section 4.4).

WMT

ViT

NormalizedTestMetric

NormalizedTestMetric

1.0

1.00

0.75

0.8

0.50

0.6

NAdamW Baseline

NAdamW Baseline

0.25

Schedule-Free

Schedule-Free

0.4

0.00

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Time

0.0 0.2 0.4 0.6 0.8 Normalized Time

fastMRI

Librispeech Conformer

NormalizedTestMetric

NormalizedTestMetric

1.000

1.00

0.99

0.995

0.98

Schedule-Free

Schedule-Free

NAdamW Baseline

NAdamW Baseline

0.990

0.97

0.0 0.1 0.2 0.3 0.4 Normalized Time

0.0 0.2 0.4 0.6 0.8 1.0 1.2 Normalized Time

OGBG

Criteo1TB

NormalizedTestMetric

NormalizedTestMetric

1.0

1.000

0.9

0.8

0.998

NAdamW Baseline

Schedule-Free

0.7

Schedule-Free

NAdamW Baseline

0.6

0.996

0.0 0.1 0.2 0.3 0.4 Normalized Time

0.0 0.2 0.4 0.6 0.8 Normalized Time

Librispeech Deepspeech

NormalizedTestMetric

1.00

0.99

Schedule-Free

NAdamW Baseline

0.98

0.0 0.2 0.4 0.6 Normalized Time

- Figure 7: Schedule-Free Adam compared to target-setting baseline on the Algoperf competition self-tuning track.

Sensorless

Aloi

DNA

100

100

80

90

Accuracy(%)

Accuracy(%)

Accuracy(%)

95

80

60

Polyak Averaging (85.5 SE 0.00) Primal Averaging (83.0 SE 0.44) Schedule-Free (89.7 SE 0.00)

Polyak Averaging (95.7 SE 0.01) Primal Averaging (95.4 SE 0.02) Schedule-Free (97.2 SE 0.01)

Polyak Averaging (100.0 SE 0.01) Primal Averaging (100.0 SE 0.00) Schedule-Free (100.0 SE 0.00)

70

90

40

60

LD Schedule (89.2 SE 0.01)

LD Schedule (95.8 SE 0.00)

LD Schedule (100.0 SE 0.00)

85

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

Glass

Iris

Letter

80

70

97.5

70

Accuracy(%)

Accuracy(%)

Accuracy(%)

65

95.0

60

60

92.5

Polyak Averaging (68.8 SE 0.10) Primal Averaging (65.0 SE 0.61) Schedule-Free (72.1 SE 0.14)

Polyak Averaging (98.3 SE 0.08) Primal Averaging (98.5 SE 0.06) Schedule-Free (98.6 SE 0.00)

Polyak Averaging (77.8 SE 0.00) Primal Averaging (77.8 SE 0.02) Schedule-Free (77.8 SE 0.00)

50

55

90.0

40

50

LD Schedule (70.3 SE 0.14)

LD Schedule (98.6 SE 0.00)

LD Schedule (77.8 SE 0.01)

87.5

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

Pendigits

smallNORB

USPS

100

100

100

90

98

90

Accuracy(%)

Accuracy(%)

Accuracy(%)

80

96

80

70

94

Polyak Averaging (95.8 SE 0.01) Primal Averaging (94.0 SE 0.11) Schedule-Free (96.0 SE 0.01)

Polyak Averaging (90.8 SE 0.01) Primal Averaging (90.8 SE 0.64) Schedule-Free (98.1 SE 0.00)

Polyak Averaging (98.0 SE 0.01) Primal Averaging (98.1 SE 0.03) Schedule-Free (99.5 SE 0.00)

70

60

92

60

LD Schedule (96.0 SE 0.02)

LD Schedule (96.5 SE 0.01)

LD Schedule (98.9 SE 0.01)

90

50

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

Vehicle

Vowel

Wine

- 96

- 97

- 98

- 99

- 100

- 101

80

80

Accuracy(%)

Accuracy(%)

Accuracy(%)

70

70

Polyak Averaging (80.8 SE 0.04) Primal Averaging (81.2 SE 0.16) Schedule-Free (83.4 SE 0.04)

Polyak Averaging (76.8 SE 0.10) Primal Averaging (77.2 SE 0.18) Schedule-Free (77.5 SE 0.03)

Polyak Averaging (100.0 SE 0.00) Primal Averaging (100.0 SE 0.00) Schedule-Free (100.0 SE 0.00)

60

60

LD Schedule (83.2 SE 0.14)

LD Schedule (77.6 SE 0.10)

LD Schedule (100.0 SE 0.00)

50

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

0 20 40 60 80 100 Epoch

Figure 8: Stochastic logistic regression experiments.

#### 4.3 Convex Problems

We validated the Schedule-Free learning approach on a set of standard logistic regression problems from the LibSVM repository. For each problem, and each method separately, we performed a full learning rate sweep on a power-of-two grid, and plotted mean and standard-error of the final train accuracy from 10 seeds using the best learning rate found.

Schedule-Free learning out-performs both averaging approaches and the state-of-the-art linear decay (LD) schedule baseline (Figure 8). It converges faster on all but 1 of 12 problems, has higher accuracy on 6 of the problems, and ties the baseline on the remaining problems. This demonstrates that the performance advantages of Schedule-Free methods are not limited to non-convex problems.

#### 4.4 Implementation Concerns

The Schedule-Free variant of a method typically has the same memory requirements as the base method. For instance, Schedule-Free SGD requires no extra memory over standard SGD with momentum. Whereas SGDM tracks the current point x and the momentum buffer m, we can track x and z. The quantity y can be computed directly from the latest values of x and z, and so doesn’t need to be explicitly stored. It’s also possible to instead store z and y, and then compute x when needed. This low memory usage is the case for AdamW also, see Algorithm 1.

Our efficient PyTorch implementation actually uses one buffer to always store z and the primary parameter buffer to store either x or y, with the stored quantity flipping between the two for training and test/inference passes.

Our method requires extra code to handle models where batch norm is used. This is due to the fact that BatchNorm layers maintain a running_mean and running_var to track batch statistics which is

Algorithm 1 Schedule-Free AdamW

- 1: Input: x1,learning rate γ,decay λ,warmup steps Twarmup,β1,β2,ϵ
- 2: z1 = x1
- 3: v0 = 0
- 4: for t = 1 to T do
- 5: yt = (1 − β1)zt + β1xt ▷ Momentum via interpolation
- 6: gt ∈ ∂f(yt,ζt) ▷ Gradient is evaluated at y
- 7: vt = β2vt−1 + (1 − β2)gt2
- 8: γt = γ 1 − β2t min(1,t/Twarmup) ▷ LR includes warmup and Adam bias-correction

- 9: zt+1 = zt − γtgt/(√vt + ϵ) − γtλyt

- 10: ct+1 = γ

2 t

t i=1 γi2

- 11: xt+1 = (1 − ct+1)xt + ct+1zt+1 ▷ Update weighted iterate average
- 12: end for
- 13: Return xT

ILSVRC 2012 ImageNet (ResNet-50) Momentum Sweep

TestAccuracy(%)

75

70

Schedule-Free MOM 0.75 (76.60% SE 0.02)

Schedule-Free MOM 0.9 (77.78% SE 0.04)

65

Schedule-Free MOM 0.95 (76.99% SE 0.05) Schedule-Free MOM 0.98 (75.37% SE 0.03)

60

0 25 50 75 100 125 150 175 200 Epoch

Figure 9: Sensitivity to momentum values

calculated at y. For model evaluation, these buffers need to be updated to match the statistics on the x sequence. This can be done by evaluating a small number of training batches using x right before each eval. More sophisticated approaches such as PreciseBN (Wu and Johnson, 2021) can also be used. This calculation is not needed for other normalization layers that do not use batch-statistics.

Learning rate warmup is still necessary for our method. We use a linear warmup for a fixed duration, and fuse the Adam bias-correction term into the learning rate for simplicity (this potentially impacts the effect of weight-decay during early iterations), giving a learning rate LR

γt = γ 1 − β2t min(1,t/Twarmup) that approaches γ when the warmup and bias-correction period ends. We found that performance was greatly improved by using a weighted ct sequence when warmup is used, weighted by the square of the γt used during warmup:

γt2

. (23)

ct+1 =

t i=1 γi2

This sequence decreases at a 1/t rate after the learning rate warmup. It is shifted by one from the indexing used in Theorem 2, which is done to simplify the implementation. This sequence is motivated by Theorem 2’s weighting sequences, which suggest weights proportional to polynomials of the learning rate.

Weight decay for Schedule-Free methods can be computed at either the y or z sequences. We used decay at y for our experiments, as this matches the interpretation of weight-decay as the use of an additional L2-regularizer term in the loss. We found that computing the regularization at y gives significantly better performance on some problems including ImageNet and NanoGPT training.

ILSVRC 2012 ImageNet (ResNet-50) Learning Rate Sweep

ILSVRC 2012 ImageNet (ResNet-50) Learning Rate Sweep

TestAccuracy(%)

TestAccuracy(%)

75

75

Schedule-Free LR 5.0 (76.27% SE 0.03) Schedule-Free LR 3.0 (77.39% SE 0.05) Schedule-Free LR 1.5 (77.83% SE 0.03) Schedule-Free LR 1.0 (77.42% SE 0.02) Schedule-Free LR 0.5 (76.17% SE 0.07)

SGD LR 0.5 (76.42% SE 0.05) SGD LR 0.3 (77.05% SE 0.03) SGD LR 0.15 (77.57% SE 0.02)

70

70

65

65

SGD LR 0.1 (77.68% SE 0.04)

SGD LR 0.05 (77.69% SE 0.04)

60

60

0 25 50 75 100 125 150 175 200 Epoch

0 25 50 75 100 125 150 175 200 Epoch

Figure 10: Comparison of the LR sensitivity of Schedule-Free training and cosine schedule training

### 5 Parameter Sensitivity

For Schedule-Free learning to be truly schedule-free, it’s important that the momentum hyperparameter doesn’t implicitly have a dependence on the time-horizon. If tuning this parameter gave different values depending on the training duration, then the problem of setting the horizon has just been shifted to setting the momentum value. In Figure 9 we run ImageNet training with Schedule-Free SGD for a longer-then-standard 200 epochs with a variety of momentum values, with the LR fixed to 1.5. We find that the best choice of momentum (β = 0.9) is the same for all durations of training.

Schedule-Free learning has a similar mild time-horizon dependency for the baseline learning rate value as schedule-based approaches. Figure 10 shows that the optimal learning rate stays the same for broad range of values, for both Schedule-Free and Schedule based training. For short duration training (≤ 25 epochs), larger LR values begin to show the best performance. Appendix I shows the sensitivity of the final test accuracy to the baseline learning rate for a selection of our test problems, in comparison to the baseline optimizer with a cosine schedule. We see that the overall sensitivity is similar to the baseline optimizer in each problem.

### 6 Conclusion

Two roads diverged in a wood, and II took the one less traveled by, And that has made all the difference. - Robert Frost

We have presented Schedule-Free learning, an optimization approach that removes the need to specify a learning rate schedule while matching or outperforming schedule-based learning. The primary practical limitation is the need to sweep learning rate and weight decay, as the best values differ from the those used with a schedule. We provide a preliminary theoretical exploration of the method, but further theory is needed to fully understand the method.

### 7 Contributions

Aaron Defazio discovered the method, led research experimentation and proved initial versions of Theorems 1 and 7, with experimental/theoretical contributions by Alice Yang. Alice Yang led the development of the research codebase. Ashok Cutkosky proved key results including Theorem 2 and led the theoretical investigation of the method. Ahmed Khaled developed preliminary theory for obtaining accelerated rates which was later supplanted by Theorem 2, and investigated the utility of β with large learning rates for quadratics. Additional derivations by Konstantin Mishchenko and Harsh Mehta are included in appendix sections. Discussions between Aaron Defazio, Ashok Cutkosky, Konstantin Mishchenko, Harsh Mehta, and Ahmed Khaled over the last year contributed to this scientific discovery.

### References

Bach, F. and Moulines, E. (2013). Non-strongly-convex smooth stochastic approximation with convergence rate O(1/n). In Burges, C., Bottou, L., Welling, M., Ghahramani, Z., and Weinberger,

K., editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc.

Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., Gulcehre, C., Song, F., Ballard, A., Gilmer, J., Dahl, G., Vaswani, A., Allen, K., Nash, C., Langston, V., Dyer, C., Heess, N., Wierstra, D., Kohli, P., Botvinick, M., Vinyals, O., Li, Y., and Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks.

Bojar, O., Chatterjee, R., Federmann, C., Graham, Y., Haddow, B., Huang, S., Huck, M., Koehn, P., Liu, Q., Logacheva, V., Monz, C., Negri, M., Post, M., Rubino, R., Specia, L., and Turchi, M. (2017). Findings of the 2017 conference on machine translation (wmt17). In Proceedings of the 2017 Conference on Machine Translation (WMT17).

Cesa-Bianchi, N., Conconi, A., and Gentile, C. (2004). On the generalization ability of on-line learning algorithms. IEEE Transactions on Information Theory, 50(9):2050–2057.

Cettolo, M., Niehues, J., Stüker, S., Bentivogli, L., and Federico, M. (2014). Report on the 11th IWSLT evaluation campaign. In IWSLT.

Chiang, C.-K., Yang, T., Lee, C.-J., Mahdavi, M., Lu, C.-J., Jin, R., and Zhu, S. (2012). Online optimization with gradual variations. In Conference on Learning Theory, pages 6–1. JMLR Workshop and Conference Proceedings.

Criteo (2022). Criteo 1TB click logs dataset. https://ailab.criteo.com/

##### download-criteo-1tb-click-logs-dataset/.

Cutkosky, A. (2019). Anytime online-to-batch, optimism and acceleration. In International conference on machine learning, pages 1446–1454. PMLR.

Dahl, G. E., Schneider, F., Nado, Z., Agarwal, N., Sastry, C. S., Hennig, P., Medapati, S., Eschenhagen, R., Kasimbeg, P., Suo, D., Bae, J., Gilmer, J., Peirson, A. L., Khan, B., Anil, R., Rabbat, M., Krishnan, S., Snider, D., Amid, E., Chen, K., Maddison, C. J., Vasudev, R., Badura, M., Garg, A., and Mattson, P. (2023). Benchmarking Neural Network Training Algorithms.

Defazio, A. (2020). Momentum via primal averaging: Theoretical insights and learning rate schedules for non-convex optimization.

Defazio, A., Cutkosky, A., Mehta, H., and Mishchenko, K. (2023). When, why and how much? adaptive learning rate scheduling by refinement.

Defazio, A. and Gower, R. M. (2021). The power of factorial powers: New parameter settings for (stochastic) optimization. In Balasubramanian, V. N. and Tsang, I., editors, Proceedings of The 13th Asian Conference on Machine Learning, volume 157 of Proceedings of Machine Learning Research, pages 49–64. PMLR.

Defazio, A. and Jelassi, S. (2022). Adaptivity without compromise: A momentumized, adaptive, dual averaged gradient method for stochastic optimization. Journal of Machine Learning Research, 23:1–34.

Defazio, A. and Mishchenko, K. (2023). Learning-rate-free learning by D-adaptation. The 40th International Conference on Machine Learning (ICML 2023).

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschannen, M., Arnab, A., Wang, X., Riquelme Ruiz, C., Minderer, M., Puigcerver, J., Evci, U., Kumar, M., Steenkiste, S. V., Elsayed, G. F., Mahendran, A., Yu, F., Oliver, A., Huot, F., Bastings, J., Collier, M., Gritsenko, A. A., Birodkar, V., Vasconcelos, C. N., Tay, Y., Mensink, T., Kolesnikov, A., Pavetic, F., Tran, D., Kipf, T., Lucic, M., Zhai, X., Keysers, D., Harmsen, J. J., and Houlsby, N. (2023). Scaling vision transformers to 22 billion parameters. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J., editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 7480–7512. PMLR.

Duchi, J., Hazan, E., and Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(61).

Gokaslan, A. and Cohen, V. (2019). Openwebtext corpus. http://Skylion007.github.io/

##### OpenWebTextCorpus.

Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., and Pang, R. (2020). Conformer: Convolution-augmented transformer for speech recognition. Hazan, E. (2022). Introduction to online convex optimization. MIT Press. Hazan, E. and Kale, S. (2010). Extracting certainty from uncertainty: Regret bounded by variation in

costs. Machine learning, 80:165–188. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. (2021). Masked autoencoders are scalable vision learners. arXiv:2111.06377. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition.

Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., and Leskovec, J. (2020). Open graph benchmark: datasets for machine learning on graphs. In Proceedings of the 34th International Conference on Neural Information Processing Systems.

Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. (2017). Densely connected convolutional networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2261–2269.

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., and Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. In Conference on Uncertainty in Artificial Intelligence (UAI).

Jean-Baptiste Tien, joycenv, O. C. (2014). Display advertising challenge. Joulani, P., György, A., and Szepesvári, C. (2017). A modular analysis of adaptive (non-) convex

optimization: Optimism, composite objectives, and variational bounds. In International Conference on Algorithmic Learning Theory, pages 681–720. PMLR.

Joulani, P., Raj, A., Gyorgy, A., and Szepesvári, C. (2020). A simpler approach to accelerated optimization: iterative averaging meets optimism. In International conference on machine learning, pages 4984–4993. PMLR.

Kaddour, J. (2022). Stop wasting my time! saving days of ImageNet and BERT training with latest weight averaging.

Kavis, A., Levy, K. Y., Bach, F., and Cevher, V. (2019). UniXGrad: A universal, adaptive algorithm with optimal guarantees for constrained optimization. Advances in neural information processing systems, 32.

Kingma, D. P. and Ba, J. (2014). Adam: a method for stochastic optimization. In International Conference on Learning Representations.

Lacoste-Julien, S., Schmidt, M., and Bach, F. (2012). A simpler approach to obtaining an o(1/t) convergence rate for the projected stochastic subgradient method.

Lan, G. (2012). An optimal method for stochastic composite optimization. Mathematical Programming, 133(1):365–397.

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations.

Naumov, M., Mudigere, D., Shi, H. M., Huang, J., Sundaraman, N., Park, J., Wang, X., Gupta, U., Wu, C., Azzolini, A. G., Dzhulgakov, D., Mallevich, A., Cherniavskii, I., Lu, Y., Krishnamoorthi, R., Yu, A., Kondratenko, V., Pereira, S., Chen, X., Chen, W., Rao, V., Jia, B., Xiong, L., and Smelyanskiy, M. (2019). Deep learning recommendation model for personalization and recommendation systems. CoRR.

Nesterov, Y. (1983). A method for solving a convex programming problem with convergence rate

O(1/k2). Soviet Mathematics Doklady. Nesterov, Y. (2013). Lectures on Convex Optimization. Springer Nature. Nesterov, Y. and Shikhman, V. (2015). Quasi-monotone subgradient methods for nonsmooth convex

minimization. Journal of Optimization Theory and Applications, 165(3):917–940. Orabona, F. (2019). A modern introduction to online learning. arXiv preprint arXiv:1912.13213. Panayotov, V., Chen, G., Povey, D., and Khudanpur, S. (2015). Librispeech: An asr corpus based on

public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210.

Polyak, B. (1990). New stochastic approximation type procedures. Avtomatica i Telemekhanika, 7:98–107.

Portes, J., Blalock, D., Stephenson, C., and Frankle, J. (2022). Fast benchmarking of accuracy vs. training time with cyclic learning rates.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners. Technical report, OpenAI.

Rakhlin, A., Shamir, O., and Sridharan, K. (2012). Making gradient descent optimal for strongly convex stochastic optimization. In Proceedings of the 29th International Coference on International Conference on Machine Learning.

Rakhlin, A. and Sridharan, K. (2013). Online learning with predictable sequences. In Conference on Learning Theory, pages 993–1019. PMLR.

Reddi, S. J., Kale, S., and Kumar, S. (2018). On the convergence of Adam and beyond. In International Conference on Learning Representations.

Ruppert, D. (1988). Efficient estimations from a slowly convergent Robbins-Monro process. Technical Report, Cornell University.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla,

- A., Bernstein, M., Berg, A. C., and Fei-Fei, L. (2015). ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3).

Sandler, M., Zhmoginov, A., Vladymyrov, M., and Miller, N. (2023). Training trajectories, mini-batch losses and the curious role of the learning rate.

Sanyal, S., Neerkaje, A., Kaddour, J., Kumar, A., and Sanghavi, S. (2023). Early weight averaging meets high learning rates for LLM pre-training.

Sebbouh, O., Gower, R. M., and Defazio, A. (2021). On the (asymptotic) convergence of stochastic gradient descent and stochastic heavy ball. In Conference on Learning Theory, COLT 2021, Proceedings of Machine Learning Research. PMLR.

Shamir, O. and Zhang, T. (2013). Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In Proceedings of the 30th International Conference on Machine Learning.

Sriram, A., Zbontar, J., Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., Knoll, F., and Johnson, P. (2020). End-to-end variational networks for accelerated MRI reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer.

Sutskever, I., Martens, J., Dahl, G., and Hinton, G. E. (2013). On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28. JMLR.org.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2818–2826.

Tao, W., Pan, Z., Wu, G., and Tao, Q. (2018). Primal averaging: A new gradient evaluation step to attain the optimal individual convergence. IEEE Transactions on Cybernetics, PP:1–11.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. (2017). Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Wiseman, S. and Rush, A. M. (2016). Sequence-to-sequence learning as beam-search optimization. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

Wu, Y. and Johnson, J. (2021). Rethinking "batch" in batchnorm. Zagoruyko, S. and Komodakis, N. (2016). Wide residual networks. In Proceedings of the British

Machine Vision Conference (BMVC).

Zamani, M. and Glineur, F. (2023). Exact convergence rate of the last iterate in subgradient methods. Zbontar, J., Knoll, F., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., Parente, M., Geras, K. J.,

Katsnelson, J., Chandarana, H., et al. (2018). fastMRI: An open dataset and benchmarks for accelerated MRI. arXiv preprint arXiv:1811.08839.

Zhang, M., Lucas, J., Ba, J., and Hinton, G. E. (2019). Lookahead optimizer: k steps forward, 1 step back. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Zinkevich, M. (2003). Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the Twentieth International Conference on International Conference on Machine Learning, pages 928–935.

### A Proof of Theorem 2

Theorem 2. Let F be a convex function. Let ζ1,...,ζT be an iid sequence such that F(x) = Eζ[f(x,ζ)]. Let z1,...,zT be arbitrary vectors and let w1,...,wT and β1,...,βT be arbitrary numbers in [0,1] such that zt, wt and βt are independent of ζt,...,ζT. Set:

t i=1 wizi

wt

wt

zt (9)

= xt−1 1 −

+

xt =

t i=1 wi

t i=1 wi

t i=1 wi

≜ct

≜1−ct

yt = βtxt + (1 − βt)zt (10) gt = ∇f(yt,ζt). (11)

Then we have for all x⋆:

E[ Tt=1 wt⟨gt,zt − x⋆⟩] T i=1 wi

. (12)

E[F(xT) − F(x⋆)] ≤

Proof. Throughout this proof, we will use the notation w1:t = ti=1 wi. The result is established by showing the following identity:

w1:tF(xt) − w1:t−1F(xt−1) − wtF(x⋆) ≤ wt⟨∇F(yt),zt − x⋆⟩. (24) Where here ∇F(yt) indicates a subgradient of F at yt with E[gt|zt] = ∇F(yt). Given the identity

(24), we sum over all t from 1 to T. Then the LHS will telescope to obtain:

T

wt⟨∇F(yt),zt − x⋆⟩,

w1:T(F(xT) − F(x⋆)) ≤

t=1

from which the conclusion immediately follows since E[gt|zt] = ∇F(yt). So, let us establish (24). To do so, it will help to observe the following identities:

wtzt = w1:txt − w1:t−1xt−1 w1:t−1(xt − xt−1) = wt(zt − xt) (25)

βt 1 − βt

(yt − xt). (26) Now, setting ∇F(xt) to be an arbitrary subgradient of F at xt, we have:

zt − yt =

w1:tF(xt) − w1:t−1F(xt−1) − wtF(x⋆)

= w1:t−1(F(xt) − F(xt−1)) + wt(F(xt) − F(x⋆)) ≤ w1:t−1⟨∇F(xt),xt − xt−1⟩ + wt(F(xt) − F(x⋆))

- using (25):

= wt⟨∇F(xt),zt − xt⟩ + wt(F(xt) − F(x⋆))

= wt⟨∇F(xt),zt − xt⟩ + wt(F(xt) − F(yt)) + wt(F(yt) − F(x⋆)) ≤ wt⟨∇F(xt),zt − xt⟩ + wt⟨∇F(xt),xt − yt⟩ + wt⟨∇F(yt),yt − x⋆⟩

= wt⟨∇F(xt) − ∇F(yt),zt − yt⟩ + wt⟨∇F(yt),zt − x⋆⟩

- using (26):

βt

1 − βt ⟨∇F(xt) − ∇F(yt),yt − xt⟩ + wt⟨∇F(yt),zt − x⋆⟩ Finally, recall that any convex function satisfies ⟨∇F(b) − ∇F(a),a − b⟩ ≤ 0 for all a,b. This classical fact can be established by adding the following two subgradient identities:

= wt

- F(a) ≥ F(b) + ⟨∇F(b),a − b⟩,
- F(b) ≥ F(a) + ⟨∇F(a),b − a⟩.

Then, since βt ∈ [0,1], we have wt β

1−βt ⟨∇F(xt) − ∇F(yt),yt − xt⟩ ≤ 0, which establishes the desired identity (24).

t

| |
|---|

### B Recovering Prior Conversions, and Connections to Momentum

The following recursions provide an equivalent update to our main algorithm that casts the update in a more “momentum-like” form.

- Theorem 4. Under the same assumptions and notation as Theorem 2, set:

∆t = zt+1 − zt, mt = xt+1 − xt,

ut = yt+1 − yt. Then:

wt+1w1:t−1 wtw1:t+1

wt+1 w1:t+1

mt =

mt−1 +

∆t

w1:t wt+1

mt + (1 − βt)∆t.

ut = βt + (βt − βt+1)

Here ut is playing the role of the “update vector”, as the sequence of points yt are where we will be evaluating gradients. The ∆t value can be interpreted as a “base update” value: for the case that the zt sequence is specified by SGD (as in Theorem 1), ∆t = −ηgt. Thus, the update can be interpreted

- as a momentum term mt, plus an extra “push” in the direction of ∆t scaled by 1 − βt.

Proof. Let’s solve for mt in terms of previous values:

mt = xt+1 − xt

wt+1 w1:t+1

(zt+1 − xt)

=

wt+1 w1:t+1

(∆t + zt − xt)

=

wt+1 w1:t+1

w1:t−1 wt

(xt − xt−1))

=

(∆t +

wt+1w1:t−1 wtw1:t+1

wt+1 w1:t+1

=

mt−1 +

∆t.

Now let’s solve for ut:

ut = βt+1xt+1 + (1 − βt+1)zt+1 − βtxt − (1 − βt)zt = βtmt + (1 − βt)∆t + (βt − βt+1)(zt+1 − xt+1)

w1:t wt+1

= βtmt + (1 − βt)∆t + (βt − βt+1)

(xt+1 − xt)

w1:t wt+1

= βtmt + (1 − βt)∆t + (βt − βt+1)

mt

w1:t wt+1

mt + (1 − βt)∆t

= βt + (βt − βt+1)

| |
|---|

In the special case that wt = 1 for all t, the updates simplify to:

t − 1 t + 1

1 t + 1

mt =

mt−1 +

∆t

ut = (βt + t(βt − βt+1))mt + (1 − βt)∆t. In the special case that βt = β for all t, the update for ut simplifies to:

ut = βmt + (1 − β)∆t.

From this, it is clear that if β = 1 and wt = 1, then we recover the standard Polyak momentum with a time-varying momentum factor mt = tt−+11mt−1 + t+11 ∆t, while if β = 0, then we have ordinary SGD without momentum.

- B.1 Recovering Linear Decay Let’s take a look at the update for ut = yt+1 − yt in the special case that wt = 1 for all t:

ut = (βt + t(βt − βt+1))mt + (1 − βt)∆t. Let us define αt = 1 − βt. Then we can re-write this update as:

ut = (1 − αt + t(αt+1 − αt))mt + αt∆t.

It looks like we might be able to set αt such that the coefficient of mt vanishes. In this case, αt would play the role of a “schedule” as the update would just be ut = αt∆t. Solving the recursion we get:

αt − 1 = t(αt+1 − αt), αt+1 =

(t + 1)αt − 1 t

.

Amazingly, this recursion is satisfied by αt = TT−t, which is the linear decay schedule! Notably, this schedule has αT = 0, which in turn implies that yT = xT, so that the last iterate of our algorithm is xT, for which Theorem 2 provides a convergence guarantee.

The recursion is also satisfied by αt = 1 for all t (which recovers standard Polyak-Ruppert averaging). Notably, this recursion shows that α1 will determine all subsequent α values. The values will decease linearly to zero, and then they will try to go negative, which is not allowed. So the linear decay schedule is the value of α1 that is “just barely” allowed since it hits zero at αT.

In general with arbitrary wt, the recursion is:

w1:t wt+1

1 − αt + (αt+1 − αt)

= 0.

If we insist that αT = 0 (so that yT = xT and we get a “last iterate” guarantee), then solving the recursion yields:

wt+1:T w1:T

αt =

,

which exactly recovers the main result of Defazio et al. (2023).

### C Generalizing Theorem 2 via Bregman Divergences

Here, we provide a generalized version of Theorem 2 in the style of Joulani et al. (2020). This result employs Bregman divergences to tighten the inequality of Theorem 2 to an equality.

- Theorem 5. Let F be a convex function. Let ζ1,...,ζT be a sequence of i.i.d. random variables, and let g be a function such that E[g(x,ζt)] ∈ ∂F(x) for all x and t. Let z1,...,zT be arbitrary vectors and let w1,...,wT and α1,...,αT be arbitrary non-negative real numbers with αt ≤ 1 such that zt, wt and αt are independent of ζt,...,ζT. Define the Bregman divergence of F as BF(a,b) = F(a) − F(b) − ⟨∇F(b),a − b⟩2. Set:

t i=1 wizi

wt

wi

= xt−1 1 −

+

zt

xt =

t i=1 wi

t i=1 wi

t i=1 wi

yt = (1 − αt)xt + αtzt gt = g(yt,ζt).

Define the “compressed sum” notation: w1:t = ti=1 wi, with w1:0 = 0.

2if F is not differentiable, then by abuse of notation define ∇F(b) = E[g(b, ζ)], which is a particular choice of subgradient of F.

Then we have for all x⋆:

E[F(xT) − F(x⋆)] = E

T t=1 wt⟨gt,zt − x⋆⟩

w1:T

− E

t(1−αt)

T t=1

wt αt BF(yt,xt) + w

αt BF(xt,yt) w1:T

− E

T t=1 w1:t−1BF(xt−1,xt) + wtBF(x⋆,yt)

w1:T

##### .

Let’s take a minute to unpack this result since it is depressingly complicated. Recall that the Bregman divergence for a convex function must be positive, and so all the subtracted Bregman divergence terms can be dropped to make the bound only looser. This recovers Theorem 2. However, in Section D, we show how to exploit the negative Bregman terms to achieve accelerated rates when F is smooth, and in Section E we show how to exploit the negative Bregman terms to achieve faster rates when F is strongly-convex.

Proof. The proof is nearly the same as that of Theorem 2. The only difference is that we keep track of all the error terms in the inequalities via Bregman divergences.

Throughout this proof, we use ∇F(x) to indicate Eζ[g(x,ζ)]. When F is differentiable, this is simply the ordinary gradient at x. When F is non-differentiable, this reprents a specific choice of subgradient

- at x.

Recall that any convex function satisfies ⟨∇F(b) − ∇F(a),a − b⟩ = −BF(a,b) − BF(b,a) for all a,b. This classical fact can be established by adding the following two subgradient identities:

- F(a) = F(b) + ⟨∇F(b),a − b⟩ + BF(a,b)
- F(b) = F(a) + ⟨∇F(a),b − a⟩ + BF(b,a)

⟨∇F(b) − ∇F(a),a − b⟩ = −BF(a,b) − BF(b,a). (27) The Theorem is established by showing the following identity:

w1:tF(xt) − w1:t−1F(xt−1) − wtF(x⋆) = wt⟨∇F(yt),zt − x⋆⟩ −

wt(1 − αt) αt

wt αt

BF(yt,xt) −

BF(xt,yt)

− w1:t−1BF(xt−1,xt) − wtBF(x⋆,yt). (28) Given the identity (28), we sum over all t from 1 to T. Then the LHS will telescope to obtain:

w1:T(F(xT) − F(x⋆)) =

T

wt⟨∇F(yt),zt − x⋆⟩

t=1

T

wt(1 − αt) αt

wt αt

−

BF(yt,xt) −

BF(xt,yt)

t=1

T

−

w1:t−1BF(xt−1,xt) − wtBF(x⋆,yt),

t=1

from which the conclusion immediately follows since E[gt|g1,...,gt−1] = E[∇F(yt)|g1,...,gt−1]. So, let us establish (28). To do so, it will help to observe the following identities:

wtzt = w1:txt − w1:t−1xt−1 w1:t−1(xt − xt−1) = wt(zt − xt) (29)

1 − αt αt

(yt − xt). (30)

zt − yt =

So, we have:

- using (29):

= wt⟨∇F(xt),zt − xt⟩ + wt(F(xt) − F(x⋆)) − w1:t−1BF(xt−1,xt)

= wt⟨∇F(xt),zt − xt⟩ + wt(F(xt) − F(yt)) + wt(F(yt) − F(x⋆)) − w1:t−1BF(xt−1,xt)

= wt⟨∇F(xt),zt − xt⟩ + wt⟨∇F(xt),xt − yt⟩ + wt⟨∇F(yt),yt − x⋆⟩ − w1:t−1BF(xt−1,xt) − wtBF(yt,xt) − wtBF(x⋆,yt)

= wt⟨∇F(xt) − ∇F(yt),zt − yt⟩ + wt⟨∇F(yt),zt − x⋆⟩ − w1:t−1BF(xt−1,xt) − wtBF(yt,xt) − wtBF(x⋆,yt)

- using (30):

w1:tF(xt) − w1:t−1F(xt−1) − wtF(x⋆)

= w1:t−1(F(xt) − F(xt−1) + wt(F(xt) − F(x⋆))

= w1:t−1⟨∇F(xt),xt − xt−1⟩ + wt(F(xt) − F(x⋆)) − w1:t−1BF(xt−1,xt)

- using (27):

1 − αt

αt ⟨∇F(xt) − ∇F(yt),yt − xt⟩ + wt⟨∇F(yt),zt − x⋆⟩ − w1:t−1BF(xt−1,xt) − wtBF(yt,xt) − wtBF(x⋆,yt)

= wt

= wt⟨∇F(yt),zt − x⋆⟩ − wt

1 − αt αt

(BF(xt,yt) + BF(yt,xt)) − w1:t−1BF(xt−1,xt) − wtBF(yt,xt) − wtBF(x⋆,yt)

= wt⟨∇F(yt),zt − x⋆⟩ −

wt(1 − αt) αt

wt αt

BF(yt,xt) −

BF(xt,yt) − w1:t−1BF(xt−1,xt) − wtBF(x⋆,yt).

| |
|---|

### D Acceleration

In this section, we show that by instantiating our framework with an optimistic online learning algorithm (Rakhlin and Sridharan, 2013), we achieve accelerated convergence guarantees. Our results match those available in the prior literature (Kavis et al., 2019; Joulani et al., 2020). Our approach is inspired by Joulani et al. (2020),: their method is based upon a version of Theorem 5 for the special case that αt = 0. Our result simply extends their analysis to αt = O(1/t).

First, we establish an important technical Corollary that simplifies Theorem 5 in the case that F is smooth and αt is sufficiently small.

- Corollary 1. Under the same conditions as Theorem 5, suppose additionally that F is L-smooth and

suppose αt ≤ w

10w1:t for all t. Then we have for all x⋆:

t

T t=1 wt⟨gt,zt − x⋆⟩

E[F(xT) − F(x⋆)] ≤ E

w1:T

− E

T t=1 w1:t−1∥∇F(yt) − ∇F(yt−1)∥2

6Lw1:T

,

where above the value of y0 is arbitrary (since the coefficient is w1:0 = 0).

Proof. The key thing is to observe that smoothness implies BF(a,b) ≥ 2L∥∇F(a)−∇F(b)∥2. The rest of the argument is straightforward manipulation of the terms in Theorem 5:

wt(1 − αt) αt

wt(2 − αt)

wt αt

2Lαt ∥∇F(xt) − ∇F(yt)∥2 −w1:t−1BF(xt−1,xt) − wtBF(x⋆,yt) ≤ −

−

BF(yt,xt) −

BF(xt,yt) ≤ −

w1:t−1

2L ∥∇F(xt) − ∇F(xt−1)∥2. Next, observe that for any vectors a,b,c, for any λ > 0:

−∥a + b + c∥2 = −∥a∥2 − ∥b∥2 − ∥c∥2 − 2⟨a,b⟩ − 2⟨b,c⟩ − 2⟨a,c⟩

≤ −(1 − 2/λ)∥a∥2 + (2λ − 1)(∥b∥2 + ∥c∥2), where we have used Young’s inequality: |⟨v,w⟩| ≤ ∥v∥

2

2

2λ + λ∥w∥

2 . Therefore, setting λt = 3 we obtain:

− w1:t−1BF(xt−1,xt) − wtBF(x⋆,yt) ≤ −

w1:t−1 6L ∥∇F(yt) − ∇F(yt−1)∥2

5w1:t−1 2L

(∥∇F(xt) − ∇F(yt)∥2 + ∥∇F(xt−1) − ∇F(yt−1)∥2). Now, since αt ≤ w

+

10w1:t ≤ 1, we obtain: −

t

wt(1 − αt) αt

wt αt

BF(xt,yt) − w1:t−1BF(xt−1,xt) − wtBF(x⋆,yt) ≤ −

BF(yt,xt) −

w1:t−1

6L ∥∇F(yt) − ∇F(yt−1)∥2 −

5w1:t 2L ∥∇F(xt) − ∇F(yt)∥2 + −

5w1:t−1 2L ∥∇F(xt−1) − ∇F(yt−1)∥2.

Now summing over t from 1 to T (and dropping one negative term), the sum telescopes to:

T

w1:t−1

6L ∥∇F(yt) − ∇F(yt−1)∥2. The result now follows from Theorem 5.

−

t=1

| |
|---|

Now, we consider the case that zt is given by an optimistic mirror descent algorithm:

- Corollary 2. Suppose F is L-smooth. Define g0 = 0 and suppose also that for some D satisfying

- D ≥ ∥y1 − x⋆∥:

T

T

wt2∥gt − gt−1∥2.

wt⟨gt,zt − x⋆⟩ ≤ D

t=1

t=1

Finally, suppose E[∥gt − gt−1∥2] ≤ ∥∇F(yt) − ∇F(yt−1)∥2 + σt2 for some constants σ1,...,σT (these are just variance bounds on the stochastic gradient oracle). Then with wt = t and αt ≤ 5(t1−1), we have:

2D Tt=1 t2σt2 T(T + 1)

14D2L T(T + 1)

E[F(xT) − F(x⋆)] ≤

+

D2L T2

Dσ √

= O

+

,

T

where σ is uniform upper-bound on σt. Note that the algorithm does not need to know L or σ.

Algorithms producing z sequences obtaining the guarantee stated here are called “optimistic online learning algorithms”.

Proof. Applying Corollary 1, we obtain immediately:

T(T + 1) 2

E[F(xT) − F(x⋆)]

 

 D

T

T

(t − 1)t 12L ∥∇F(yt) − ∇F(yt−1)∥2

≤ E

t2∥gt − gt−1∥2 −

t=1

t=1

T

t2E[∥∇F(yt) − ∇F(yt−1)∥2] + t2σt2 + ∥∇F(y1)∥2

≤ D

24L

t=1

T

1 24L

t2E ∥∇F(yt) − ∇F(yt−1)∥2

−

t=1

T

t2E[∥∇F(yt) − ∇F(yt−1)∥2] + D

≤ D

t=1

T

t2σt2 + ∥∇F(y1)∥2

24L

t=1

−

√

Using the identity A

1 24L

T

t2E ∥∇F(yt) − ∇F(yt−1)∥2

t=1

2

4B:

C − BC ≤ A

- ≤ 6D2L +

L∥y1 − x⋆∥2 24

+ D

T

t=1

t2σt2

- ≤ 7D2L + D

T

t2σt2.

t=1

Divide by T(T2+1) to conclude the result. D.1 An Optimistic Regret Bound

| |
|---|

In this section we provide an algorithm that achieves the optimistic regret bound required for our acceleration result Corollary 2. This algorithm is a mild variation on the established literature (Rakhlin and Sridharan, 2013; Chiang et al., 2012; Hazan and Kale, 2010; Joulani et al., 2017) to slightly improve a technical dependence on the maximum gradient value.

Lemma 1. For a sequence of vectors g1,...,gT, set ηt = √ D

with g0 = 0, define

2 ti=1 ∥gi−gi−1∥2

mt = maxi≤t ∥gi − gi−1∥ and define the sequence of vectors zt,zt′ and g˜t by the recursions: z1 = z1′ = 0 g˜t = gt−1 + min(mt−1,∥gt − gt−1∥)

gt − gt−1 ∥gt − gt−1∥ ηt =

D

m2t + ti=1 ∥g˜i − gi−1∥2 zt′+1 = Π∥z′

t+1∥≤Dzt′ − ηtg˜t zt+1 = Π∥z

t+1∥≤Dzt′+1 − ηtgt. Then:

T

T

∥gt − gt−1∥2.

⟨gt,zt − x⋆⟩ ≤ 7D 2

t=1

t=1

Proof. For purposes of notation, define g0 = 0 and z0′ = 0. Further, observe that:

∥g˜t − gt−1∥ ≤ mt−1 ∥g˜t − gt−1∥ ≤ ∥gt − gt−1∥

∥g˜t − gt∥ = mt − mt−1 ηt ≤

D

t+1 i=1 ∥g˜i − gi−1∥2

2 Tt=1 ∥gt − gt−1∥2 D

1 ηT ≤

.

Next, notice that zt′+1 = argmin∥z∥≤D⟨g˜t,z⟩ + 21η

∥z − zt′∥2. Therefore since ∥x⋆∥ ≤ D, by first order optimality conditions:

t

zt′+1 − zt′ ηt

,zt′+1 − x⋆ ≤ 0

g ˜t +

1 ηt⟨zt′ − zt′+1,zt′+1 − x⋆⟩

⟨g˜t,zt′+1 − x⋆⟩ ≤

∥zt′+1 − x⋆∥2 2ηt −

∥zt′+1 − zt′∥2 2ηt

= ∥zt′ − x⋆∥2 2ηt −

.

Similarly, we have zt = argmin∥z∥≤D⟨gt−1,z⟩ + 2η1

t−1

∥z − zt′∥2. From this we have:

zt − zt′ ηt−1

,zt − zt′+1 ≤ 0

gt−1 +

∥zt′ − zt′+1∥2 2ηt−1 −

∥zt − zt′+1∥2 2ηt−1 −

∥zt − zt′∥2 2ηt−1 ⟨g˜t,zt − zt′+1⟩ ≤

⟨gt−1,zt − zt′+1⟩ ≤

∥zt′ − zt′+1∥2 2ηt−1 −

∥zt − zt′+1∥2 2ηt−1 −

∥zt − zt′∥2 2ηt−1

+ ⟨g˜t − gt−1,zt − zt′+1⟩ by Young’s inequality:

∥zt′ − zt′+1∥2 2ηt−1 −

∥zt − zt′+1∥2 2ηt−1 −

∥zt − zt′∥2 2ηt−1

≤

+ ∥zt − zt′+1∥2

ηt−1∥g˜t − gt−1∥2 2

+

2ηt−1 ≤

∥zt′ − zt′+1∥2 2ηt−1

ηt−1∥g˜t − gt−1∥2 2

+

.

So, combining these facts (and noticing that ηt−1 ≥ ηt:)

∥zt′+1 − x⋆∥2 2ηt

∥zt′ − x⋆∥2 2ηt −

ηt−1∥g˜t − gt−1∥2 2 ⟨gt,zt − x⋆⟩ ≤

⟨g˜t,zt − x⋆⟩ ≤

+

∥zt′+1 − x⋆∥2 2ηt

∥zt′ − x⋆∥2 2ηt −

ηt−1∥g˜t − gt−1∥2 2

+ ⟨gt − g˜t,zt − x⋆⟩

+

∥zt′+1 − x⋆∥2 2ηt

∥zt′ − x⋆∥2 2ηt −

ηt−1∥g˜t − gt−1∥2 2

≤

+ 2D(mt − mt−1).

+

So, we have:

T

t=1

T

⟨gt,zt − x⋆⟩ ≤ 2DmT + ∥z1′ − x⋆∥2 2η1

∥zt′ − x⋆∥2 2

1 ηt −

1 ηt−1

+

t=2

T

ηt−1∥g˜t − gt−1∥2 2

+

t=1

T

ηt−1∥g˜t − gt−1∥2 2

≤ 2DmT + 4D2/ηT +

t=1

T

ηt−1∥g˜t − gt−1∥2 2

≤ 6D2/ηT +

t=1

T

D∥g˜t − gt−1∥2 2 ti=1 ∥g˜i − gi−1∥2

≤ 6D2/ηT +

t=1

- ≤ 6D2/ηT + D

T

t=1

∥g˜t − gt−1∥2

- ≤ 7D 2

T

∥gt − gt−1∥2.

t=1

| |
|---|

### E Strongly Convex Losses

Suppose that the expected loss F is actually known to be µ-strongly convex. Then we’d like to have a convergence guarantee of O(1/µT). This is achieved in Theorem 6 below.

- Theorem 6. Under the same assumptions as Theorem 5, define ℓt(z) = ⟨gt,z⟩+ µ2∥yt −z∥2. Define the “regret” of the sequence zt as:

Regretℓ(x⋆) =

Then we have for x⋆ = argmin F:

T

wt(ℓt(zt) − ℓt(x⋆)).

t=1

E[F(xT) − F(x⋆)] ≤ E

Regretℓ(x⋆) − Tt=1

wtµ

2 ∥zt − yt∥2 w1:T

.

In particular, suppose ∥x⋆∥ ≤ D for some known bound D and ∥gt∥ ≤ G for all t for some G so long as ∥yt∥ ≤ D. Then if we define wt = t for all t and set zt by:

2(gt + µ(zt − yt)) µ(t + 1)

zt+1 = Π∥z∥≤D zt −

. then we have:

2(G + 2µD)2 µ(T + 1)

E[F(xT) − F(x⋆)] ≤

.

Proof. From Theorem 5, we have:

E[F(xT) − F(x⋆)] ≤ E

T t=1 wt⟨gt,zt − x⋆⟩

T t=1 wtBF(x⋆,yt)

w1:T −

w1:T

.

Now, since F is µ-strongly convex, we have BF(x⋆,yt) ≥ µ2∥yt − x⋆∥2. Further, we have:

T

T

wtµ

wtµ 2 ∥zt − yt∥2 +

2 ∥x⋆ − yt∥2. From this we obtain the desired result:

wt⟨gt,zt − x⋆⟩ =

wt(ℓt(zt) − ℓt(x⋆)) −

t=1

t=1

Regretℓ(x⋆) − Tt=1

wtµ

2 ∥zt − yt∥2 w1:T

E[F(xT) − F(x⋆)] ≤ E

.

For the final statement, observe that with wt = t, wtℓt(z) = t⟨gt,z⟩ + tµ2 ∥z − yt∥2 is tµ-strongly convex. Therefore if we use learning rate ηt = µw1

= µt(t2+1), then standard analysis of projected OGD yields:

1:t

T

T

tµ 2 ∥zt − x⋆∥2

t(ℓt(zt) − ℓt(x⋆)) ≤

t⟨∇ℓt(zt),zt − x⋆⟩ −

t=1

t=1

∥zT+1 − x⋆∥2 2ηT

- 1

- 2η1 −

µ 2 ∥zt − x⋆∥2 −

≤ ∥z1 − x⋆∥2

T

T

ηtt2∥∇ℓt(zt)∥2 2

- 1

- 2ηt −

1 2ηt−1 −

tµ 2

∥zt − x⋆∥2

+

+

t=2

t=1

T

ηtt2∥∇ℓt(zt)∥2 2

≤

t=1

T

1 µ

∥∇ℓt(zt)∥2

≤

t=1

T

1 µ

∥gt + µ(zt − yt)∥2

=

t=1

T(G + 2µD)2 µ

≤

.

where in the last inequality we have observed that since ∥zt∥ ≤ D and yt is a linear combination of past z values, ∥yt∥ ≤ D as well. Finally, observing that w1:T = T(T2+1), the result follows.

| |
|---|

### F Large Step size convergence

- Theorem 7. Consider the online learning setting with bounded gradients gt. Let zt+1 = zt − γgt. Let D = ∥z1 − z∗∥ for arbitrary reference point z∗ and define G = maxt≤T ∥gt∥. Suppose that the chosen step-size is γ = D/G, then if it holds that:

T

T

⟨gt,zt − z1⟩ ≤ D

t=1

t=1

then:

 D

T

1 T

⟨gt,zt − z∗⟩ = O

T

t=1

Proof. Consider SGD with fixed step size γ:

zt+1 = zt − γgt. Let

T

sT+1 =

γgt.

t=1

∥gt∥2, (31)

 . (32)

T

∥gt∥2

t=1

Recall from D-Adaptation (Defazio and Mishchenko, 2023) theory that:

T

- 1

- 2

γ ⟨gt,zt − z1⟩ =

t=1

T

- 1

- 2 ∥st+1∥2 (33)

γ2 ∥gt∥2 −

t=1

and:

T

T

γ ⟨gt,zt − z1⟩. (34) Now suppose that the regret at time T is negative. Then trivially the theorem holds:

γ ⟨gt,zt − z∗⟩ ≤ ∥sT+1∥D +

t=1

t=1

 ,

 D

T

T

1 T

∥gt∥2

⟨gt,zt − z∗⟩ ≤ 0 = O

T

t=1

t=1

therefore, without loss of generality we may assume that Tt=1 γ ⟨gt,zt − z∗⟩ ≥ 0. Then from combining Equation 34 with Equation 33 we have:

- 1

- 2 ∥sT+1∥2 + ∥sT+1∥D +

- 1

- 2

0 ≤ −

T

γ2 ∥gt∥2 .

t=1

This is a quadratic equation in ∥sT+1∥ which we can solve explicitly via the quadratic formula, taking the largest root:

√b2 − 4ac 2a

−b ±

∥sT+1∥ ≤

.

Plugging in the values a = −12, b = D, c = 12 Tt=1 γ2 ∥gt∥2:

T

T

γ2 ∥gt∥2.

γ2 ∥gt∥2 ≤ 2D +

D ± D2 +

t=1

t=1

Therefore:

∥sT+1∥ ≤ 2D + γ

Substituting this into Equation 34:

T

∥gt∥2.

t=1

T

γ ⟨gt,zt − z∗⟩ ≤ 2D2 + γD

t=1

T

T

∥gt∥2 +

γ ⟨gt,zt − z1⟩.

t=1

t=1

Therefore, if Tt=1 ⟨gt,zt − z1⟩ ≤ D Tt=1 ∥gt∥2 then:

Plugging in γ = D/G:

T

γ ⟨gt,zt − z∗⟩ ≤ 2D2 + 2γD

t=1

T

∥gt∥2.

t=1

and the theorem follows.

T

⟨gt,zt − z∗⟩ ≤ 2DG + 2D

t=1

T

∥gt∥2

t=1

≤ 4D

T

∥gt∥2,

t=1

| |
|---|

### G Experimental Setup

- G.1 Convex experiments Each dataset is obtained from the LIBSVM repository and used without modifications.

|Hyper-parameter<br><br>|Value|
|---|---|
|GPUs<br><br>|1×V100|
|Batch size|16|
|Epochs<br><br>|100|
|Seeds<br><br>|10|
|Schedule-Free β1<br><br>|0.9|

|Hyper-parameter|Value|
|---|---|
|Decay<br><br>|0.0|
|Optimizer<br><br>|Adam|
|Baseline β1|0.9|
|β2<br><br>|0.95|

- G.2 CIFAR-10

We used custom training code based on the PyTorch tutorial code for this problem. Following standard data-augmentation practises, we appliyed random horizontal flips and random offset cropping down to 32x32, using reflection padding of 4 pixels. Input pixel data was normalized by centering around 0.5.

|Hyper-parameter|Value|
|---|---|
|Architecture|Wide ResNet 16-8|
|Epochs|300|
|GPUs<br><br>|1×V100|
|Batch size per GPU<br><br>|128|
|Cosine/Schedule-Free Warmup<br><br>|5%|
|Baseline Stepwise LR|0.1|

|Hyper-parameter<br><br>|Value|
|---|---|
|Seeds|10|
|decay|0.0001|
|Baseline Momentum|0.9|
|Schedule-Free LR<br><br>|10|
|Schedule-Free β|0.9|
|Baseline Cosine LR|0.2|

- G.3 CIFAR-100

We used the same codebase as for our CIFAR-10 experiments, with the same data augmentation. We normalized each input image using fixed mean and standard error values derived from preprocessing the data.

|Hyper-parameter<br><br>|Value|
|---|---|
|Architecture<br><br>|DenseNet [6,12,24,16], growth rate 12|
|Epochs<br><br>|300|
|GPUs<br><br>|1×V100|
|Schedule-Free β<br><br>|0.9|
|Cosine/Schedule-Free Warmup|5%|
|Baseline Stepwise LR|0.05|

|Hyper-parameter|Value|
|---|---|
|Batch size per GPU|64|
|Seeds<br><br>|10|
|Decay|0.0002|
|Baseline Momentum|0.9|
|Schedule-Free LR<br><br>|5|
|Baseline Cosine LR<br><br>|0.05|

- G.4 SVHN We used the same codebase as for our CIFAR experiments, and following the same data preprocessing.

|Hyper-parameter<br><br>|Value|
|---|---|
|Batch size|32|
|Weight decay Cosine<br><br>|0.0001|
|Weight decay Step Sched<br><br>|5e-5|
|Seeds|10|
|Baseline Stepwise LR|0.1|

|Hyper-parameter<br><br>|Value|
|---|---|
|Cosine/Schedule-Free Warmup<br><br>|5%|
|Schedule-Free decay|0.0002|
|Schedule-Free LR|1.0|
|Schedule-Free β|0.9|
|Baseline Cosine LR|0.1|

#### G.5 ImageNet

We used the same code-base as for our CIFAR-10 experiments, and applied the same preprocessing procedure. The data-augmentations consisted of PyTorch’s RandomResizedCrop, cropping to 224x224 followed by random horizontal flips. Test images used a fixed resize to 256x256 followed by a center crop to 224x224.

|Hyper-parameter<br><br>|Value|
|---|---|
|Architecture|ResNet50|
|Epochs<br><br>|100|
|GPUs<br><br>|8×V100|
|Batch size per GPU|32|
|Schedule-Free Decay<br><br>|0.00005|
|Baseline Stepwise LR<br><br>|0.1|
|Baseline Cosine LR|0.05|

|Hyper-parameter|Value|
|---|---|
|Seeds|5|
|Decay<br><br>|0.0001|
|Baseline Momentum<br><br>|0.9|
|Schedule-Free β<br><br>|0.9|
|Cosine/Schedule-Free Warmup<br><br>|5%|
|Schedule-Free LR<br><br>|1.5|

#### G.6 IWSLT14

We used the FairSeq framework 3 for our experiments. Rather than a vanilla LSTM we use the variant from Wiseman and Rush (2016) provided in the FairSeq codebase.

|Hyper-parameter<br><br>|Value|
|---|---|
|Architecture|lstm_wiseman_iwslt_de_en|
|Max Epoch<br><br>|55|
|GPUs<br><br>|1×V100|
|Tokens per batch<br><br>|4096|
|Warmup steps<br><br>|4000|
|Dropout<br><br>|0.3|
|Label smoothing|0.1|
|Schedule-Free LR<br><br>|0.02|
|Schedule-Free warmup<br><br>|5%|
|Baseline schedule|Linear Decay|

|Hyper-parameter|Value|
|---|---|
|Share decoder, input, output embed|True|
|Float16<br><br>|True|
|Update Frequency<br><br>|1|
|Seeds|10|
|Decay|0.05|
|Baseline β1<br><br>|0.9|
|Schedule-Free β1<br><br>|0.9|
|β2|0.98|
|Baseline LR<br><br>|0.01|

#### G.7 NanoGPT

We followed the NanoGPT codebase 4 as closely as possible, matching the default batch-size, training length and schedule. Our runs replicate the stated 2.85 loss in the documentation. Disabling gradient norm clipping is crucial for the Schedule-Free runs.

|Hyper-parameter<br><br>|Value|
|---|---|
|Architecture<br><br>|transformer_lm_gpt|
|Batch size per gpu<br><br>|12|
|Max Iters<br><br>|600,000|
|GPUs<br><br>|40×V100|
|Tokens per sample<br><br>|512|
|Dropout<br><br>|0.0|
|Baseline LR|0.0005|
|Warmup|2,000|
|Schedule-Free LR|0.005|
|Schedule-Free β|0.98|
|Schedule-Free decay<br><br>|0.05|

|Hyper-parameter<br><br>|Value|
|---|---|
|Block Size<br><br>|1024|
|Num layer<br><br>|12|
|Num head<br><br>|12|
|Num embd<br><br>|768|
|Float16<br><br>|True|
|Update Frequency<br><br>|16|
|Seeds<br><br>|5|
|Decay|0.1|
|Baseline β1,β2|0.9, 0.95|
|Gradient Clipping<br><br>|0.0|

3https://github.com/facebookresearch/fairseq 4https://github.com/karpathy/nanoGPT

#### G.8 MAE

Our implementation uses the offical code5, with hyper-parameters following examples given in the repository.

|Hyper-parameter<br><br>|Value|
|---|---|
|Model<br><br>|vit_base_patch16|
|Epochs<br><br>|100|
|GPUs<br><br>|32×V100|
|Batch Size|32|
|Baseline LR|5e-4|
|Layer Decay|0.65|
|Weight Decay<br><br>|0.05|
|Baseline β1|0.9|

|Hyper-parameter<br><br>|Value|
|---|---|
|β2<br><br>|0.999|
|Schedule-Free LR<br><br>|0.0.002|
|Schedule-Free decay|0.05|
|Schedule-Free β1<br><br>|0.9|
|Drop Path|0.1|
|Reprob|0.25|
|Mixup<br><br>|0.8|
|Cutmix|1.0|

#### G.9 DLRM

We used a custom implementation of the DLRM model based on the publicly available code. Our optimizer uses dense gradients for implementation simplicity, although sparse-gradients using AdaGrad is a more common baseline on this problem, we consider AdaGrad variants of our scheduling approach as future work.

|Hyper-parameter|Value|
|---|---|
|Iterations|300 000|
|Batch Size|128|
|Emb Dimension<br><br>|16|
|GPUs<br><br>|8×V100|
|Schedule-Free LR<br><br>|0.0005|
|Schedule-Free β1<br><br>|0.9|
|β2|0.999|

|Hyper-parameter|Value|
|---|---|
|Seeds<br><br>|5|
|Decay<br><br>|0.0|
|Baseline β1<br><br>|0.9|
|Warmup|0|
|Baseline LR<br><br>|0.0002|
|Baseline schedule|Linear Decay|

#### G.10 MRI

We used the version of the the fastMRI code base at https://github.com/facebookresearch/ fastMRI/tree/main/banding_removal. Note that we found that training failed using PyTorch 2 or newer, and so we ran these experiments using PyTorch 1.9.

|Hyper-parameter<br><br>|Value|
|---|---|
|Architecture<br><br>|12 layer VarNet 2.0|
|Epochs<br><br>|50|
|GPUs<br><br>|8×V100|
|Batch size per GPU|1|
|Acceleration factor<br><br>|4|
|Baseline Schedule<br><br>|Linear Decay|
|Baseline LR<br><br>|0.005|
|β2<br><br>|0.999|

|Hyper-parameter|Value|
|---|---|
|Low frequency lines<br><br>|16|
|Mask type|Offset-1|
|Seeds<br><br>|5|
|Decay<br><br>|0.0|
|Baseline β1<br><br>|0.9|
|Schedule-Free LR|0.005|
|Schedule-Free β<br><br>|0.9|

#### G.11 Algoperf

Our full algoperf entry is availiable at https://github.com/facebookresearch/schedule_ free/tree/main/schedulefree/algoperf. The hyper-parameters used for the self-tuning track submission are listed below.

5https://github.com/fairinternal/mae

|Hyper-parameter<br><br>|Value|
|---|---|
|Learning Rate<br><br>|0.0025|
|one-minus Beta1<br><br>|0.1|
|Beta2 (default)|0.9955159689799007|
|Weight Decay (default)<br><br>|0.08121616522670176|

|Hyper-parameter|Value|
|---|---|
|Dropout Rate|0.1|
|Warmup Percentage|2%|
|Label Smoothing<br><br>|0.2|
|Polynomial in ct average|0.75|

### H Polyak and Primal Averaging Runs

These experiments follow the same tuning setup as Figure 5, where the learning rate and momentum is tuned separately for each method. In each case the c weighting sequence used for Schedule-Free training is also used to ensure a fair comparison. The Polyak averaging runs include momentum in the base optimizer as we found this gave the best results. We ran the NanoGPT experiment for a shorter 200,000 steps due to computational budget considerations. The NanoGPT Polyak averaging runs show a divergence in test loss for Polyak averaging.

CIFAR-10 (WRN-16-8)

CIFAR-100 (DenseNet)

80

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | |Poly<br><br>Prim|ak Averag<br>al Averag<br>|ing (95.56% ing (92.09|SE 0.05 % SE 0.07|) )| |
| | | | |Sche|dule-Free|Reference|96.03%| | |
| | | | | | | | | | |
| | | | | | | | | | |

95

TestAccuracy(%)

TestAccuracy(%)

70

90

60

85

Polyak Averaging (76.99% SE 0.12) Primal Averaging (70.20% SE 0.19) Schedule-Free Reference 78.71%

50

80

75

40

0 50 100 150 200 250 300 Epoch

0 50 100 150 200 250 300 Epoch

SVHN (ResNet-3-96)

ILSVRC 2012 ImageNet (ResNet-50)

- 94
- 95
- 96
- 97
- 98
- 99

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | |Poly<br><br>Prim|ak Averag<br>al Averag<br>|ing (98.32 ing (97.36|% SE 0.01 % SE 0.03|) )| |
| | | | |Sche|dule-Free|Referenc|e 98.40%| | |
| | | | | | | | | | |
| | | | | | | | | | |

TestAccuracy(%)

TestAccuracy(%)

70

60

Polyak Averaging (72.84% SE 0.02) Primal Averaging (69.78% SE 0.09) Schedule-Free Reference 76.90%

50

40

0 50 100 150 200 250 300 Epoch

0 20 40 60 80 100 Epoch

fastMRI Knee (VarNet 2.0)

Criteo Kaggle (DLRM)

0.912

0.790

TestAccuracy

0.910

TestSSIM

0.785

0.908

Primal Averaging (0.9100 SE 0.00037) Polyak Averaging (0.9088 SE 0.00141) Schedule-Free Reference 0.9112

Primal Averaging (0.7899 SE 0.00006) Polyak Averaging (0.7901 SE 0.00004) Schedule-Free Reference 0.7915

0.780

0.906

0 10 20 30 40 50 Epoch

0 20 40 60 80 100 Epoch

MAE ImageNet Finetune (ViT)

OpenWebText (GPT-2 124M)

- 3
- 4
- 5
- 6
- 7

| | |Polyak Ave|raging (5.588 S|E 0.134)| |
|---|---|---|---|---|---|
| | |Primal Ave<br><br>Schedule-|raging (3.063 S Free Reference|E 0.004) @ 200k 2.878| |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

TestAccuracy(%)

80

TestLoss

75

Polyak Averaging (82.72 SE 0.03) Primal Averaging (81.98 SE 0.03) Schedule-Free Reference 83.54%

70

65

0 20 40 60 80 100 Epoch

0 50000 100000 150000 200000 Step

Figure 11: Polyak and Primal Averaging Experiments

### I Additional LR Sensitivity Plots

CIFAR-10 (WRN-16-8) LR Sensitivity

CIFAR-100 (DenseNet) LR Sensitivity

- 94

- 95

- 96

TestAccuracy(%)

TestAccuracy(%)

78

76

74

Schedule-Free

Schedule-Free

72

Cosine Schedule

Cosine Schedule

70

10 1 100 101 LR

10 2 10 1 100 101 LR

SVHN (ResNet-3-96) LR Sensitivity

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | |Sche<br><br>Cosi|dule-Free ne Schedule| |
| | | | | | | |

98.4

TestAccuracy(%)

98.3

98.2

98.1

10 2 10 1 100

LR

