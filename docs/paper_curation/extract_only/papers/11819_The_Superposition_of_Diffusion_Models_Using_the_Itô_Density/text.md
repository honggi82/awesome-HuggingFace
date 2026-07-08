Published as a conference paper at ICLR 2025
THE SUPERPOSITION OF DIFFUSION MODELS USING
THE ITÔ DENSITY ESTIMATOR
Marta Skreta∗,1,2
Lazar Atanackovic∗,1,2
Avishek Joey Bose3,4
Alexander Tong4,5
Kirill Neklyudov4,5,†
1University of Toronto
2Vector Institute
3University of Oxford
4Mila - Quebec AI Institute
5Université de Montréal
ABSTRACT
The Cambrian explosion of easily accessible pre-trained diffusion models suggests
a demand for methods that combine multiple different pre-trained diffusion
models without incurring the significant computational burden of re-training a
larger combined model. In this paper, we cast the problem of combining multiple
pre-trained diffusion models at the generation stage under a novel proposed
framework termed superposition. Theoretically, we derive superposition from
rigorous first principles stemming from the celebrated continuity equation and
design two novel algorithms tailor-made for combining diffusion models in
SUPERDIFF. SUPERDIFF leverages a new scalable Itô density estimator for the log
likelihood of the diffusion SDE which incurs no additional overhead compared
to the well-known Hutchinson’s estimator needed for divergence calculations. We
demonstrate that SUPERDIFF is scalable to large pre-trained diffusion models as
superposition is performed solely through composition during inference, and also
enjoys painless implementation as it combines different pre-trained vector fields
through an automated re-weighting scheme. Notably, we show that SUPERDIFF is
efficient during inference time, and mimics traditional composition operators such
as the logical OR and the logical AND. We empirically demonstrate the utility of
using SUPERDIFF for generating more diverse images on CIFAR-10, more faithful
prompt conditioned image editing using Stable Diffusion, as well as improved
conditional molecule generation and unconditional de novo structure design of
proteins. https://github.com/necludov/super-diffusion
1
INTRODUCTION
The design and application of generative models at scale are arguably one of the fastest-growing
use cases of machine learning, with generational leaps in performance that often exceed expert
expectations (Steinhardt, 2022). A few of the central facilitators of this rapid progress are the
availability of high-quality training data and large computing hardware (Kaplan et al., 2020); which in
tandem provide a tried and trusted recipe to scale generative models in a variety of data modalities such
as video generation (Brooks et al., 2024), natural language understanding (OpenAI, 2023; Achiam
et al., 2023; Dong et al., 2022), and other challenging domains like mathematical reasoning (Trinh
et al., 2024), or code assistance (Bubeck et al., 2023). As a result, it is not surprising that a driving force
behind current generative modeling research is centered around developing open-source tooling (Dao
et al., 2022; Kwon et al., 2023) to enable further scaling and understanding emergent behavior of
such models (Schaeffer et al., 2023), including probing current limitations (Dziri et al., 2024).
Indeed, the rapid escalation of generative model development has also induced a democratizing effect,
given the easy access to large pre-trained in the current AI climate (Stability AI, 2023; Midjourney,
2023; Ramesh et al., 2021). Furthermore, with the rise of open-source models, it is now easier than
ever to host and deploy fine-tuned models. However, the current pace of progress also makes it
infeasible to easily scale further models without confronting practical challenges. For instance, for
continuous domains such as natural images current pre-trained diffusion models already exhaust all
public data, with a growing proportion of the web already populated with synthetic data (Schuhmann
et al., 2022). Compounding these challenges is the tremendous cost of pre-training these large
∗Authors contributed equally.
†Correspondence to k.necludov@gmail.com
1
arXiv:2412.17762v2  [cs.LG]  28 Feb 2025

Published as a conference paper at ICLR 2025
A DOG and
A CAT
Avg. of outputs
Joint prompting
A FLAMINGO and
A CANDY CANE
A WAFFLE CONE
and A VOLCANO
A DONUT
and A MAP
FIREWORKS
and DANDELION
A BICYCLE WHEEL 
and A SPIDER WEB
Figure 1: Concept interpolations via different methods: SUPERDIFF (top row), the averaging of outputs with
different prompts (middle row), and joint prompting with standard Stable Diffusion (SD) (bottom row) for six
different prompt combinations. Here we use SUPERDIFF with the AND operation (sampling equal densities).
diffusion models, which at present makes it computationally unattractive for individuals to build
large pre-trained models on different datasets without re-training a larger combined model.
A compelling alternative to training ever larger models is to consider the efficacy of maximizing the
utility of existing pre-trained models. In particular, it is interesting to consider the compositional
benefits of combining pre-trained models at the generation stage in place of training a single mono-
lithic model (Du & Kaelbling, 2024). For diffusion models, which are the current de facto modeling
paradigm over continuous domains, compositional generation can be framed as modifying the in-
ference mechanism through either guidance (Dhariwal & Nichol, 2021; Ho & Salimans, 2022) or
applying complex MCMC correction schemes (Du et al., 2023). Despite advancements, methods that
utilize guidance lack a firm theoretical underpinning (Bradley & Nakkiran, 2024) and MCMC tech-
niques are prone to scaling issues, and more importantly, remain unproven at combining existing large
pre-trained diffusion models. This motivates the following timely research question: Can we combine
pre-trained diffusion models solely at inference in a theoretically sound and efficient manner?
Present work. In this paper, we cast the problem of combining and composing multiple pre-trained
diffusion models under a novel joint inference framework we term superposition. Intuitively we
develop our framework by starting from the well-known principle of superposition in physical systems
which summarizes the net response to multiple inputs in a linear system as the sum of individual
inputs. Applying the superposition principle, even in the simplest case of a mixture model, requires
constructing the resultant superimposed vector field; which can be built analytically by reweighting
using the marginal density of generated samples along each diffusion model’s inference trajectory.
We introduce SUPERDIFF for scalable superposition by proposing a novel estimator of the density
of generated samples along an SDE trajectory, which we expect to be of independent interest beyond
superposition. In particular, our Itô density estimator in Thm. 1 is general as it is not restricted to
the vector field under which the trajectories are generated, it requires no additional computation
during inference, and, when the ground true scores of the marginals are available, it is exact. This
is in stark contrast to all prior density estimation approaches that involve computing expensive and
high variance divergence estimates of a drift vector field associated with the probability flow ODE.
Armed with our new Itô density estimator we propose to combine pre-trained diffusion models
by guiding the joint inference process through fine-grained control of the relative superposition
weights of the model outputs. More precisely, in Sec. 3.2, we propose algorithms for two specific
instantiations, illustrated in (Fig. 1) of the superposition principle: (i) generating samples from a
mixture of densities (an OR operation over models) or (ii) generating samples that are equally likely
under all densities (an AND operation over models).
We test the applicability of our approach SUPERDIFF using the two proposed superimposition
strategies for image and protein generation tasks. For images, we first demonstrate the ability to
combine the outputs of two models trained on disjoint datasets such that they yield better performance
2

Published as a conference paper at ICLR 2025
than the model trained on both of the datasets (see Sec. 4.1). In addition, we demonstrate the ability
to interpolate between densities which correspond to concept interpolation in the image space (see
Sec. 4.2). For proteins, we demonstrate that combining two different generative models leads to
improvements in designability and novelty generation (see Sec. 4.3). Across all our experimental
settings, we find that combining pre-trained diffusion models using SUPERDIFF leads to higher
fidelity generated samples that better match task specification in text-conditioned image generation
and also produce more diverse generated protein structures than comparable composition strategies.
2
PRELIMARIES
Generative models learn to approximate a target data distribution pdata ∈P(Rd) defined over Rd
using a parametric model qθ with learnable parameters θ. In the conventional problem definition, the
data distribution is realized as an empirical distribution that is provided as a training set of samples
D = {µi}m
i=1. Whilst there are multiple generative model families to choose from, we restrict our
attention to diffusion models (Song et al., 2020; Ho et al., 2020) which are arguably the most popular
modeling family driving current application domains. We next review the basics of the continuous
time formulation of diffusion models before casting them within our superposition framework.
2.1
CONTINUOUS-TIME DIFFUSION MODELS
A diffusion model can be cast as the solution to the Stochastic Differential Equation (Øksendal, 2003),
dxt = ft(x)dt + gtdWt,
x0 ∼q0(x0).
(1)
In the Itô SDE literature, the function ft : Rd →Rd is known as the drift coefficient while gt : R →R
is a real-valued function called the diffusion coefficient and Wt is the standard Wiener process. The
subscript index t ∈[0, 1] indicates the time-valued nature of the stochastic process. Specifically, we fix
the starting time t = 0 to correspond to the data distribution pdata := q0(x0) and set t = 1 as the ter-
minal time t = 1 to an easy to sample prior such as a standard Normal distribution pnoise := q1(x1) =
N(x1|0, I). As such the diffusion SDE, also called the forward process, can be seen as progressively
corrupting the data distribution and ultimately hitting a terminal distribution devoid of any structure.
To generate samples from the marginal density qt(xt) induced by the diffusion SDE in equation 1
we leverage the reverse-time SDE with the same marginal density as demonstrated below.
Proposition 1. [Reverse-time SDEs/ODE] Marginal densities qt(x) induced by Eq. (1) corre-
spond to the densities induced by the following SDE that goes back in time (τ = 1 −t) with the
corresponding initial condition
dxτ =

−ft(xτ) +
g2
t
2 + ξτ

∇log qt(xτ)

dτ +
p
2ξτdW τ , xτ=0 ∼q1(x0) ,
(2)
where W τ is the standard Wiener process in time τ, and ξτ is any positive schedule.
See proof in App. A.1. The reverse SDE flows backward in time τ = 1 −t and is linked to the
diffusion SDE through the score ∇x log qτ(x), and dW τ is another Weiner process. As a result,
a parametric model ∇log qτ(x; θ) may directly learn to approximate this score function for every
point in time and then draw samples by simulating the reverse SDE in equation 2 by plugging back
in the learned score. Notably, for ξt ≡0, the SDE becomes an ODE which defines a smooth change
of measure corresponding to pnoise into the measure corresponding to pdata.
In practice, for generative modelling, the forward SDE from Eq. (1) is chosen to be so simple that it
can be integrated in time analytically without simulating the SDE itself. This is equivalent to choosing
the noising schedule first, and then deriving the SDE that corresponds to this schedule. Namely, for
every training sample µi, we can define the density of the corrupted µi as a normal density with the
mean scaled according to αt and the variance σ2
t , then the density of the entire corrupted dataset is
simply a mixture over all training samples µi, i.e.
qi
t(x) = N(x | αtµi, σ2
t I) , qt(x) = 1
N
N
X
i=1
qi
t(x) .
(3)
Clearly, choosing αt, σt such that α0 = 1, σ0 = 0 and α1 = 0, σ1 = 1, we guarantee pdata := q0(x0)
and pnoise := q1(x1) = N(x1|0, I). This perturbation of the data distribution using a Gaussian
kernel offers specific forms for the drift ft, and diffusion coefficient gt as described next.
3

Published as a conference paper at ICLR 2025
Proposition 2. [Ornstein–Uhlenbeck SDE] The time-dependent densities in Eq. (3) correspond
to the marginal densities of the following SDE, with the corresponding initial condition
dxt = ∂log αt
∂t
xt
|
{z
}
ft(xt)
dt +
r
2σ2
t
∂
∂t log σt
αt
|
{z
}
gt
dWt , x0 ∼q0(x0) .
(4)
See proof in App. A.2. We highlight the simplicity of the drift term, a linear scaling, that allows us
to simulate efficiently the reverse SDE and is crucial for the proposed density estimators in Sec. 3.1.
Altogether, the derivations of this section allow us to go from the noise schedules of samples
in Eq. (3) used during the training of a given diffusion model to the corresponding forward SDE
in Prop. 2, and finally to the reverse SDEs or ODE used in Prop. 1.
2.2
SUPERPOSITION OF ODES AND SDES
In this section, we introduce the superposition of multiple time-dependent densities that correspond
to different stochastic processes. A suggestive view of these densities is as processes corresponding
to different training data (e.g. different datasets), different conditions (e.g. different text prompts),
or simply differently trained diffusion models. Namely, we consider N forward noising process
{qi
t(x)}N
i=1 that possibly start from different initial distributions {qi
t=0(x)}N
i=1 (e.g. different datasets).
Assume that we know the individual vector fields vi
t(x) that define the change of corresponding
densities qi
t(x) via the state-space ODEs and continuity equations,
dxt
dt = vi
t(xt)
=⇒
∂
∂tqi
t(x) = −

∇x, qi
t(x)vi
t(x)

, ∀i ∈[N].
(5)
The superposition of the noising processes {qi
t(x)}N
i=1 is the mixture of corresponding densities:
qmix
t
(x) :=
N
X
j=1
ωjqj
t (x),
N
X
i=1
ωi = 1 , ωi ≥0 ,
(6)
where ωj is a mixing coefficient. Note that superimposed qmix
t
(x) also satisfies the continuity equation
for the superposition of the vector fields vi
t(x) as demonstrated in the following proposition.
Proposition 3. [Superposition of ODEs (Liu, 2022)] The mixture density in Eq. (6) follows the
continuity equation with the superposed vector fields from Eq. (5), i.e.
∂
∂tqmix
t
(x) = −

∇x, qmix
t
(x)vt(x)

, vt(x) =
1
PN
j=1 ωjqj
t (x)
N
X
i=1
ωiqi
t(x)vi
t(x) .
(7)
We reproduce the proof for Prop. 3 in the context of our superposition in App. A.3. The superposition
principle is the core principle that allows for efficient simulation-free learning of the flow-based
models (Liu et al., 2022b; Lipman et al., 2022; Albergo et al., 2023) and diffusion models (Song et al.,
2020). We discuss how these frameworks are derived from the superposition principle in App. B.
The superposition principle straightforwardly extends to the marginal densities of SDEs. That is,
consider marginals densities qi
τ(x) generated by the following SDEs
dxτ = ui
τ(xτ)dτ + gτdW τ , xτ=0 = qi
τ=0(x0) ,
(8)
where one has to note the same diffusion coefficient for all the SDEs. Then the mixture of densities
from Eq. (6) can be simulated by the SDE from the following proposition.
Proposition 4. [Superposition of SDEs] The mixture qmix
t
(x) := PN
i=1 ωiqi
t(x) of density
marginals {qi
t(x)}N
i=1 induced by SDEs from Eq. (8) corresponds to the following SDE
dxτ = uτ(xτ)dτ + gτdW τ , ut(x) =
1
PN
j=1 ωjqj
t (x)
N
X
i=1
ωiqi
t(x)ui
t(x) .
(9)
See App. A.4 for the proof. Both Prop. 3 and Prop. 4 can be easily extended to the families of
densities parameterized with a continuous variable (see Theorem 1 in Peluchetti (2023)).
4

Published as a conference paper at ICLR 2025
Train Data A
Train Data B
Generated samples
(a) Training datasets
(b) Averaging the vector
fields
(c) Sampling the mixture
of densities (ours)
(d) Sampling equal
densities (ours)
Figure 2: An intuitive illustration of using model superposition for improving inference performance. We show
an example of two disjoint datasets and train a model for each set. Each individual model learns to generate
samples only from their respective datasets. Using model superposition enables sampling from both densities.
3
SUPERPOSITION OF DIFFERENT MODELS
We now introduce our method for combining pre-trained diffusion models using the principle of
superposition. The result of this is a novel inference time algorithm SUPERDIFF which can be easily
applied without further fine-tuning or post-processing of any of the pre-trained diffusion weights. Our
proposed approach SUPERDIFF can be instantiated in two distinct ways that allow for the composition
diffusion models that can be informally interpreted as logical composition operators in the logical
AND and the logical OR. More precisely, given two pre-trained diffusion models that are trained on
datasets A and B inference using SUPERDIFF can be done by either sampling from the mixture of the
two learned densities, i.e. logical AND Fig. 2c, or sampling from the equal density locus (logical OR
Fig. 2d). In such a manner, superposition using AND leads to generated samples that are equally likely
under both pre-trained diffusion models while superposition using OR creates samples that are pref-
erentially generated by either pre-trained model—and thus mimics the empirical distributions A or B.
Method overview. SUPERDIFF is applicable in settings where the modeler has access to M
pre-trained diffusion models, along with its learned score function ∇x log qi
t(x). Each of the
pre-trained models follows a marginal density qi
t(x) and as a result must admit a corresponding
vector field vi
t(x) that satisfies the continuity equation in Equation 5. The key idea of our approach is
an adaptive re-weighting scheme of the pre-trained model’s vector fields that relies on the likelihood
of a sample under different models. A naive approach to estimating each marginal density during
generation immediately presents several technical challenges as it requires the estimation of the
divergence of superimposed vector fields. In particular, these challenges can be stated as follows:
(C1) The marginal superpositioned vector field differs from the vector fields of either of the models.
(C2) The divergence operation requires backpropagation through the network and is computationally
expensive even with Hutchinson’s trace estimator (Hutchinson, 1989).
Our proposed approach SUPERDIFF overcomes these computational challenges by introducing
a novel density estimator in Sec. 3.1. Crucially, this new estimator does not require divergence
estimation and enjoys having the same variance as the computationally expensive Hutchinson’s trace
estimator, making it a favorable choice when generating using large pre-trained diffusion models. In
section Sec. 3.2 we exploit this new density estimator to formally present our algorithm SUPERDIFF
and derive connections to the composition operators that intuitively resemble logical AND and OR.
3.1
EVALUATING THE DENSITIES ON THE FLY
In this section, we introduce a novel method for evaluating the marginal density of a diffusion model
during the inference process. The conventional way to evaluate the density uses the continuity
equation and solves the same ODE that is used for generating samples. This, however, is not easily
possible in the case of our superposition of vector fields framework as outlined in Prop. 3. To solve this,
we present the following proposition that disentangles the vector field generating the sample (ut(x) in
the proposition) and the vector fields corresponding to different generative models vi
t(x).
5

Published as a conference paper at ICLR 2025
Algorithm 1: SUPERDIFF pseudocode (for OR and AND operations)
Input :M pre-trained score models ∇x log qi
t(x), the parameters of the schedule αt, σt, stepsize
dτ > 0, temperature parameter T, bias parameter ℓ, and initial noise z ∼N(0, I).
for τ = 0, . . . , 1 do
t = 1 −τ , ε ∼N(0, I)
κi
τ ←



softmax(T log qi
t(xτ) + ℓ) //
for OR according to Prop. 3
solve Linear Equations //
for AND according to Prop. 6
ut(xτ) ←PM
i=1 κi
τ∇log qi
t(xτ)
dxτ ←
 −f1−τ(xτ) + g2
1−τut(xτ)

dτ + g1−τdW τ //
using Prop. 1
xτ+dτ ←xτ + dxτ
d log q1−τ(xτ) =

dxτ, ∇log q1−τ(xτ)

+

∇, f1−τ(xτ)

+
+

f1−τ(xτ) −
g2
1−τ
2
∇log q1−τ(xτ), ∇log q1−τ(xτ)

dτ //
using Thm. 1
return xτ
Proposition 5. [Smooth density estimator] For the integral curve x(t) solving dx/dt = ut(xt),
and the density qi
t(x(t)) satisfying the continuity equation ∂
∂tqi
t(x) = −

∇x, qi
t(x)vi
t(x)

, the
log-density along the curve changes according to the following ODE
d
dt log qi
t(x(t)) = −

∇x, vi
t(x)

−

∇x log qi
t(x), vi
t(x) −ut(x)

.
(10)
See proof in App. C. We use this proposition for our experiments conducted in Sec. 4.1. However,
as outlined previously, evaluating the marginal density via the continuity equation is restricted to
small-scale models due to the computational challenges associated with efficiently estimating the
divergence of the associated vector field. As a result, a common line of attack assumes constructing
a stochastic unbiased estimator that trades for increased speed by introducing a bit of variance into
the divergence estimate. This approach is known as Hutchinson’s estimator and requires computing
a Jacobian-vector product at every step of the inference.
Instead, we propose a new way to estimate density that allows for efficient computation while inte-
grating the backward SDE from Prop. 1 with a specific choice of the diffusion coefficient.
Theorem 1. [Itô density estimator] Consider time-dependent density qt(x) induced by the
marginals of the following SDE
dxt = ft(xt)dt + gtdWt , xt=0 ∼q0(x) , t ∈[0, 1] ,
(11)
where dWt is the Wiener process. For the reverse-time (τ = 1 −t) SDEs with any vector field
uτ and the same diffusion coefficient gt, i.e.
dxτ = uτ(xτ)dτ + g1−τdW τ , τ ∈[0, 1] ,
(12)
the change of the log-density log qτ(xτ) follows the following SDE
d log q1−τ(xτ) =

dxτ, ∇log q1−τ(xτ)

+

∇, f1−τ(xτ)

+
+

f1−τ(xτ) −g2
1−τ
2
∇log q1−τ(xτ), ∇log q1−τ(xτ)

dτ .
(13)
We provide the proof in App. C. Notably, the SDE Eq. (13) that keeps track of the change of log-density
includes only the divergence of the forward SDE drift

∇, f1−τ(xτ)

. However, in practice, when
using the Ornstein-Uhlenbeck SDE, this divergence is simply a constant due to a linear drift scaling.
In App. D, we derive the same estimator but in discrete time using the detailed balance condition.
3.2
SUPERDIFF: SUPERPOSING PRE-TRAINED DIFFUSION MODELS
Mixture of densities (logical OR). For a mixture of the densities, superposition of the models follows
directly from the propositions in Sec. 2.2. That is, for every qi
t(x), we assume that it can be generated
from SDEs or an ODE from Prop. 1 and we assume the knowledge of scores ∇log qi
t(x).
6

Published as a conference paper at ICLR 2025
Then, for the ODE simulation, according to Prop. 3, we can sample from the mixture of densities
qmix
t
(x) := 1/M PM
i=1 qi
t(x) using the following vector field:
vτ(x) = −f1−τ(x) + g2
1−τ
2
M
X
i=1
qi
t(x)
P
j qj
t (x)
∇log qi
1−τ(x) ,
(14)
starting from samples xτ=0 ∼q1(x0). The densities are estimated along the trajectory using Prop. 5.
We highlight this analogously applies for simulation using the SDE dxτ = uτ(xτ)dτ + g1−τdW τ.
Namely, according to Prop. 4, we use the following vector field:
uτ(x) = −f1−τ(x) + g2
1−τ
M
X
i=1
qi
t(x)
P
j qj
t (x)
∇log qi
1−τ(x) ,
(15)
starting from the samples xτ=0 ∼q1(x0). The densities are estimated along the trajectory using
Thm. 1. We provide the pseudocode in Algorithm 1.
Sampling equal densities (logical AND). To produce the sample that have equal densities under
different models we rely on our formula for the density update (see Thm. 1) to find the optimal
weights for the vector fields. Indeed, for M diffusion models, we have a system of M equations: the
equal change of density for every model and the normalization constraint for model weights, which
is a linear system w.r.t. vector field weights as we show in the following proposition.
Proposition 6. [Density control] For the SDE
dxτ =
M
X
j=1
κjuj
τ(xτ)dτ + g1−τdW τ ,
(16)
where κ are the weights of different models and P
j κj = 1, one can find κ that satisfies
d log qi
1−τ(xτ) = d log qj
1−τ(xτ) , ∀i, j ∈[M] ,
(17)
by solving a system of M + 1 linear equations w.r.t. κ.
We provide the proof and the formulas for the system of linear equations in App. C.1. This
proposition allows us to find κ that controls the densities to stay the same for all the models as
described in Algorithm 1. This approach can also be straightforwardly extended to the case of
diffusion models for satisfying different prescribed density ratios, i.e.
d log q1
1−τ(xτ) = d log qi
1−τ(xτ) + ℓi .
(18)
4
EXPERIMENTS
4.1
JOINING MODELS WITH DISJOINT TRAINING DATA
We validate the proposed Algorithm 1 for the generation of the mixture of the distributions (OR
setting). We split CIFAR-10 into two disjoint training sets of equal size (first 5 labels and last 5 labels),
train two diffusion models on each part, and generate the samples jointly using both models. Namely,
the stochastic inference is the OR implementation of Algorithm 1, whereas the deterministic setting
is the integration of the ODEs (see Prop. 1) and the estimation of the log-density according to Prop. 5.
For the choice of hyperparameters, architecture, and data preprocessing, we follow (Song et al., 2020).
In Table 1, we demonstrate that the performance of SUPERDIFF drastically outperforms the
performance of the individual models and performs even better than the model trained on the
union of both parts of the dataset. For comparison, we evaluate conventional image quality metrics:
Fréchet Inception Distance (FID), Inception Score (IS), and Feature Likelihood Divergence (FLD)
(Jiralerspong et al., 2023), which takes into account the generalization abilities of the model.
4.2
CONCEPT INTERPOLATION WITH SUPERDIFF (AND) AND STABLE DIFFUSION
Next, we evaluate the ability of SUPERDIFF to interpolate different concepts using Stable Diffusion
(SD) v1-4 (logical AND). In this setup, we generate an image from SD by conditioning it on a prompt
using classifier-free guidance. We define two models using two separate prompts that represent
concepts — e.g. "a sunflower" and "a lemon". We can thus consider each prompt-conditioned
model as a separate diffusion model and apply Algorithm 1 to generate images sampled with
7

Published as a conference paper at ICLR 2025
Table 1: Unconditional image generation performance for CIFAR-10 with models trained on two disjoint
partitions of the training data (labeled A and B). We compare SUPERDIFF (OR) with the respective models with
the model that is trained on the full dataset (modelA∪B) and random choice between two models (modelA OR B).
ODE inference
SDE inference
FID (↓)
IS (↑)
FLD (↓)
FID (↓)
IS (↑)
FLD (↓)
modelA
14.00
8.73
15.50 ± 0.17
15.33
7.98
15.47 ± 0.18
modelB
13.09
7.89
18.90 ± 0.18
13.50
7.98
18.54 ± 0.23
modelA∪B
6.00
8.95
8.06 ± 0.12
3.50
9.14
7.51 ± 0.11
modelA OR B
4.28
9.14
5.96 ± 0.11
3.99
9.36
5.29 ± 0.14
SUPERDIFF (OR)
4.41
9.12
6.10 ± 0.11
4.00
9.36
5.33 ± 0.05
SUPERDIFF T =100 (OR)
4.11
9.21
5.89 ± 0.11
4.00
9.48
5.20 ± 0.11
Table 2: Quantativate evaluation of SD-generated images. We compare SUPERDIFF (AND), joint prompting, and
averaging of outputs for two concept prompts. We report the average minimum CLIP, ImageReward, and TIFA
scores of each prompt pair, which gives a measure of how well both concepts are represented.
Min. CLIP(↑)
Min. ImageReward (↑)
Min. TIFA (↑)
Joint prompting
23.87
−1.62
27.58
Average of scores (Liu et al., 2022a)
24.23
−1.57
32.48
SUPERDIFF (AND)
24.79
−1.39
39.92
proportionally equal likelihood with respect to both prompt-conditioned models.1 We generate 20
images for 20 different concept pairs (tasks) for our model and baselines (c.f. App. J).2
Baselines for concept interpolation. We consider two approaches for composing images as
baselines. The first is simple averaging of SD outputs based on the approach in Liu et al. (2022a);
we set κ = 0.5. The second guides SD generation with a single joint prompt. The prompts are
constructed by joining two input concepts with the linking term "that looks like" — e.g. "a
sunflower that looks like a lemon". For fairness, we also flip the order of the prompt and
keep the image with the higher score for all metrics listed below (Luo et al., 2024).
SUPERDIFF qualitatively generates images with better concept interpolations. We plot sample
generated images for SUPERDIFF (AND) in Fig. 1. For the complete set of generated images, see
App. J (Figs. A5–A24). We observe that SUPERDIFF (AND) can interpolate concepts while also
maintaining high perceptual quality. In contrast, the averaging baseline either fails to interpolate
concepts from both prompts fully or yields images with lower perceptual quality. We observe that
SD using a single prompt struggles to interpolate both concepts.
SUPERDIFF outperforms baselines on three image evaluation metrics. To quantitatively evaluate
the generated images, we consider three metrics: CLIP Score (Radford et al., 2021), ImageReward (Xu
et al., 2024), and TIFA (Hu et al., 2023). CLIP Score measures the cosine similarity between an image
embedding and text prompt embedding. ImageReward evaluates generated images by assigning a
score that reflects how closely they align with human preferences. TIFA generates several question-
answer pairs using a Large Language Model for a given prompt and assigns a score by answering the
questions based on the image with a visual question-answering model. We report these metrics for SU-
PERDIFF and all baselines in Table 2. For the logical AND setting, we evaluate the image against each
concept prompt separately (i.e., "a sunflower" as one prompt and "a lemon" as the other) and
take the minimum score for each metric, which measures how well both concepts are represented. We
find that SUPERDIFF (AND) obtains the highest scores across all metrics, indicating that our method
can better represent both concepts in the images, while the baseline methods typically only represent
one concept or (especially in the case of averaging outputs), generate compositions with lower fidelity.
4.3
PROTEIN GENERATION
Next, we apply our method in the setting of unconditional de novo protein generation. Protein
generation has critical implications in drug discovery (Abramson et al., 2024). A good understanding
of the protein landscape is important to rationally find novel proteins. We evaluate proteins generated
by the superposition of two existing protein diffusion models, Proteus (Wang et al., 2024a) and
FrameDiff (Yim et al., 2023b). We report the results of our best model in Table 3, as well as results
for each model individually and simply averaging them.
1In Table 2, we report results for inference only using SDEs. For the ODE setting, evaluating densities takes
about 1082 ± 7 seconds per image with SD, whereas it only takes 209 ± 2 with our density estimator in the
SDE setting (over a 5-fold decrease!), while also requiring almost 30% less memory.
2SUPERDIFF (AND) SD v1-4: https://huggingface.co/superdiff/superdiff-sd-v1-4
8

Published as a conference paper at ICLR 2025
Table 3: Evaluation of SUPERDIFF and baseline models, Proteus & FrameDiff, for unconditional protein gen-
eration. We show results for three categories of metrics: designability, novelty, and diversity. We also include a
baseline of simple averaging of scores (κ = 0.5). We use the parameter ℓ(see Eq. (18)) to control and bias model
superposition towards Proteus, i.e. (ℓ> 0) or FrameDiff (ℓ< 0). We use temperature values T = 1 for all vari-
ants of SUPERDIFF. For each type of model composition (averaging or SUPERDIFF), we mark each metric with a
(†) if it is better than both Proteus and FrameDiff on their own, and with a (⋆) if it better than either one of them.
Designability
Novelty
Diversity
< 2Å scRMSD
(↑)
scRMSD
(↓)
< 0.5 scTM
(↑)
< 0.3 scTM
(↑)
Max. scTM
(↓)
Frac. β
(↑)
Pairwise scTM
(↓)
Max. cluster
(↑)
FrameDiff
0.392 ± 0.03
4.315 ± 0.25
0.152 ± 0.02
0.016 ± 0.01
0.570 ± 0.02
0.175 ± 0.01
0.337 ± 0.02
0.326 ± 0.05
Proteus
0.928 ± 0.02
1.014 ± 0.07
0.360 ± 0.03
0.020 ± 0.01
0.536 ± 0.01
0.119 ± 0.01
0.312 ± 0.01
0.217 ± 0.02
Average of scores (Liu et al., 2022a)
0.740 ± 0.03⋆
1.960 ± 0.14⋆
0.360 ± 0.03⋆
0.024 ± 0.01†
0.511 ± 0.01†
0.139 ± 0.01⋆
0.310 ± 0.01†
0.253 ± 0.01⋆
SUPERDIFFℓ=0(OR)
0.752 ± 0.03⋆
1.940 ± 0.14⋆
0.276 ± 0.03⋆
0.008 ± 0.01
0.547 ± 0.01⋆
0.147 ± 0.01⋆
0.309 ± 0.02†
0.268 ± 0.02⋆
SUPERDIFFℓ=1(OR)
0.976 ± 0.01†
0.929 ± 0.05†
0.396 ± 0.03†
0.024 ± 0.01†
0.528 ± 0.01†
0.127 ± 0.01⋆
0.307 ± 0.02†
0.246 ± 0.03⋆
SUPERDIFFℓ=0(AND)
0.752 ± 0.03⋆
2.079 ± 0.16⋆
0.296 ± 0.03⋆
0.040 ± 0.01†
0.521 ± 0.01†
0.141 ± 0.01⋆
0.306 ± 0.01†
0.256 ± 0.01⋆
Figure 3: UMAP visualization of protein structures
showing cluster archetypes where structure diversity is
maintained with SUPERDIFF ℓ=0 (OR).
Metrics for evaluations.
We consider des-
ignability,
novelty,
and
diversity
metrics
for evaluation of unconditional generation
of protein structures.
Protein designability
refers to the in-silico agreement between
generated structures and refolded structures
as computed using a purpose-built folding
model e.g. ESMFold (Lin et al., 2022), which
is positively correlated with the synthesizability
of the protein in a wet-lab setting. Generally,
if the root-mean-square distance between the
generated and refolded proteins (scRMSD) is
less than 2Å, it is considered to be designable.
We compute several novelty metrics based on
the similarity of generated proteins to those
from the set of known proteins (the training set),
which are called scTM scores; the lower the score, the less similar the generated protein is to the
training data. Lastly, we use diversity to assess the degree of heterogeneity present in the set of
generated proteins. This is done by clustering the generated proteins in terms of sequence overlap,
and measuring the fraction of proteins with challenging-to-generate secondary structures such as
β sheets (Bose et al., 2024). We provide details of all metrics in this section in App. H.2. All results
are averages over 50 generated proteins at lengths {100, 150, 200, 250, 300} for 500 timesteps.
SUPERDIFF improves structure generation. By combining two protein diffusion models, SU-
PERDIFF is able to outperform both of them. This is somewhat surprising as FrameDiff is substantially
less designable than Proteus (0.392 vs. 0.928 scRMSD) (see Table 3). Nevertheless, by using SU-
PERDIFF (OR), we can increase designability, novelty, and maintain diversity, outperforming simple
averaging. We further investigate the composition made by SUPERDIFF (OR) in Fig. 3. Here, we see a
few modes (particularly on the plot’s left-hand side) that Proteus does not generate; SUPERDIFF (OR)
can maintain knowledge of these clusters (although to a lesser extent) while maintaining designability.
We also find that SUPERDIFF (AND) outperforms averaging in designability and diversity. Perhaps
more impressively, SUPERDIFF (AND) can generate the most proteins that are furthest away from
the set of known proteins (scTM score < 0.3) by almost two times more than the next best method.
We visualize these proteins in Fig. A2 and explore composition in Fig. A1. This motivates the utility
of applying our method in novel discovery settings.
4.4
SMALL MOLECULE GENERATION
We evaluate SUPERDIFF (AND) for multi-property molecule generation, where candidates must
satisfy multiple constraints (e.g., binding to a target protein, non-toxicity, and bioavailability).
Conventional approaches filter molecules post-hoc or optimize properties separately (Schneuing et al.,
2024), but SUPERDIFF (AND) enables direct control. We investigate molecule generation with the
following pairs of property pairs: (1) high binding score to the protein GSK3β and high drug-likeness
(QED) (Bickerton et al., 2012) and (2) high binding scores to the proteins GSK3β and JNK3 (an
example of dual-target drug design (Jin et al., 2020; Chen et al., 2024)). We generate molecules using
LDMol (Chang & Ye, 2024), a latent diffusion model that generates a SMILES string (Weininger,
1988) conditioned on a text prompt. Again, we take each prompt-conditioned model as a separate
diffusion model.
Following Wang et al. (2024b), we use the prompts "This molecule
inhibits {GSK3B/JNK3}" and "This molecule looks like a drug".
9

Published as a conference paper at ICLR 2025
Table 4: Multi-property molecule generation results. For a set of two target properties (P1 and P2), we report the
average top-10 and top-1 best performing molecules (for the joint methods, “best" is taken as the largest P1*P2
scores). We also report the average minimum scores of each property pair, as well as the diversity, validity &
uniqueness (V&U), and quality of all molecules. Metrics are taken from 5 runs of batch-size 512.
Top-10 (↑)
Top-1 (↑)
All
P1 / P2
(P1 ∗P2)
min(P1, P2)
P1
P2
P1
P2
Div. (↑)
V&U (↑)
Qual. (↑)
P1 only
GSK3β
QED
0.107±0.024
0.240±0.049
0.411±0.034
0.266±0.058
0.550±0.037
0.258±0.085
0.907
0.742
0.07
P2 only
0.030±0.010
0.033±0.011
0.033±0.011
0.884±0.008
0.032±0.029
0.919±0.008
0.921
0.830
0.191
Joint prompting
0.171±0.029
0.273±0.034
0.287±0.040
0.631±0.050
0.338±0.106
0.743±0.102
0.901
0.741
0.114
Avg. of scores (Liu et al., 2022a)
0.154±0.012
0.269±0.023
0.287±0.014
0.580±0.020
0.392±0.043
0.566±0.114
0.916
0.773
0.137
SUPERDIFF (AND)
0.177±0.024
0.273±0.022
0.277±0.024
0.668±0.027
0.440±0.064
0.680±0.087
0.914
0.783
0.141
P1 only
GSK3β
JNK3
0.057±0.007
0.135±0.011
0.410±0.034
0.136±0.011
0.550±0.037
0.208±0.087
0.907
0.741
0.070
P2 only
0.056±0.011
0.183±0.014
0.275±0.048
0.195±0.009
0.340±0.187
0.326±0.053
0.910
0.690
0.082
Joint prompting
0.071±0.022
0.186±0.045
0.367±0.028
0.187±0.046
0.534±0.108
0.296±0.126
0.910
0.714
0.098
Avg. of scores (Liu et al., 2022a)
0.073±0.013
0.199±0.012
0.351±0.037
0.210±0.012
0.446±0.124
0.348±0.099
0.909
0.702
0.077
SUPERDIFF (AND)
0.080±0.025
0.209±0.035
0.361±0.060
0.214±0.038
0.466±0.139
0.360±0.104
0.908
0.726
0.088
Evaluation metrics. We evaluate properties using oracles from Therpeutic Data Commons (Huang
et al., 2021); we also compute diversity (Tanimoto distance on Morgan fingerprints (Rogers & Hahn,
2010)) and quality (Lee et al., 2025), which is the fraction of molecules that are valid, unique, have
QED ≥0.6 and synthetic accessibility (SA) ≤4.0 (Ertl & Schuffenhauer, 2009).
SUPERDIFF improves multi-property molecule generation. In Table 4, we evaluate molecules
generated by prompting for each for property separately (P1/P2 only), taking the average of scores,
and with SUPERDIFF (AND). We also compare to joint prompting, where we generate using a prompt
of both properties: "This molecule {prop_1} and {prop_2}". We create a prompt for
both property orderings and randomly sample 50% of the final molecules from each ordering to
evaluate the same number of molecules. We find that SUPERDIFF (AND) is able to generate molecules
that have higher products of predicted properties compared with averaging scores and joint prompting.
SUPERDIFF (AND) also has a higher average minimum score for each property pair, indicating that
the molecules better satisfy both properties simultaneously. We visualize the best molecules in App. I.
5
RELATED WORK
Compositional diffusion models. Combining multiple density models into one model with better
properties is a classical subject in machine learning (Hinton, 2002). For diffusion models, the most
straightforward combination is via averaging their respective learned score functions (Liu et al., 2022a;
Kong et al., 2024), which can also be viewed as a form of guidance (Ho et al., 2020; Dhariwal &
Nichol, 2021; Bansal et al., 2023). Another way to combine diffusion models comes from their connec-
tions to energy-based models (Du et al., 2023; 2020; Nie et al., 2021; Ajay et al., 2024). This, however,
comes under a strong assumption that the marginal densities of the noising process are given, which
is not the case in most of the modern applications (e.g. (Rombach et al., 2022)); our method resolves
this. Density estimation, also, is the main bottleneck for the continual learning of diffusion models,
e.g. Golatkar et al. (2023) proposes to learn a separate model for the densities in order to simulate the
vector fields from Prop. 3. Finally, Zhong et al. (2024); Biggs et al. (2024) propose to combine models
by averaging their weights which, however, does not allow for merging of models with different archi-
tectures or different conditions—both resolved by SUPERDIFF in Sec. 4.3 and Sec. 4.2 respectively.
The concurrent work (Karczewski et al., 2024) independently proposed the smooth density estimator
(Prop. 5) and an estimator similar to the Itô density estimator (Thm. 1). However, for the latter, the
authors do not consider other types of dynamics besides the reverse-time SDEs from Prop. 1. For
additional related works regarding protein generation, see App. E.
6
CONCLUSION
Despite the ubiquity of diffusion models, many possible ways of performing generation remain
unexplored, with classifier-free guidance being the only practical option (Ho & Salimans, 2022).
In this paper, we address this shortcoming by proposing two novel methods for combining different
models (or the same model with different condition variables) using SUPERDIFF for joint generation.
Limitations. While computationally efficient SUPERDIFF is still limited by the computational budget
required to produce the outputs of each model. In particular, the combination at the level of model
outputs cannot be simply done via cheap combinations of pre-trained model weights. This, however,
invites us to develop a more principled—architecture and training-agnostic—method that does not
require any assumptions prior assumptions which makes SUPERDIFF a general purpose method.
Future Work.
Our method unlocks novel research directions by allowing for the principled
generation of novel samples that were not previously possible to generate easily Sec. 4.2. Furthermore,
we argue that the proposed way to estimate the density during the generation enables numerous new
potential ways to control the generation process by providing information about the likelihood.
10

Published as a conference paper at ICLR 2025
REPRODUCIBILITY STATEMENT
To facilitate reproducibility of our empirical results and findings, we intend to make our code
publicly available in the final version. We describe all mathematical and algorithmic details necessary
to reproduce our results throughout this paper. In Sec. 3 we outline the theoretical basis and
mathematical framework for our method. Furthermore, we provide pseudocode for our method
in Algorithm 1. For our theoretical contributions, we provide detailed proofs for all theorems and
propositions in App. A, App. B and App. C. We provide experimental details for the CIFAR-10
image experiment results in Sec. 4.1. Details regarding experiments for concept interpolation via
Stable Diffusion are discussed in Sec. 4.2 and App. J. Experimental details for unconditional protein
generation are described in Sec. 4.3 and App. H.1.
ACKNOWLEDGMENTS
The authors thank Rob Brekelmans, Francisco Vargas, and the anonymous reviewers for providing
the feedback significantly improving the paper. The authors thank Viktor Ohanesian for imple-
menting the algorithm for Stable Diffusion XL https://huggingface.co/superdiff/
superdiff-sdxl-v1-0. The research was enabled in part by by the Province of Ontario and
companies sponsoring the Vector Institute (http://vectorinstitute.ai/partners/),
the computational resources provided by the Digital Research Alliance of Canada (https:
//alliancecan.ca), and Mila (https://mila.quebec). KN is supported by IVADO.
AJB is partially supported by an NSERC Post-doc fellowship. This research is partially supported by
EPSRC Turing AI World-Leading Research Fellowship No. EP/X040062/1.
REFERENCES
Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf
Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure
prediction of biomolecular interactions with alphafold 3. Nature, pp. 1–3, 2024.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774, 2023.
Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum,
Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for
hierarchical planning. Advances in Neural Information Processing Systems, 36, 2024.
Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying
framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.
Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas
Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 843–852, 2023.
Helen M Berman, John Westbrook, Zukang Feng, Gary Gilliland, Talapady N Bhat, Helge Weissig,
Ilya N Shindyalov, and Philip E Bourne. The protein data bank. Nucleic acids research, 28(1):
235–242, 2000.
G Richard Bickerton, Gaia V Paolini, Jérémy Besnard, Sorel Muresan, and Andrew L Hopkins.
Quantifying the chemical beauty of drugs. Nature chemistry, 4(2):90–98, 2012.
Benjamin Biggs, Arjun Seshadri, Yang Zou, Achin Jain, Aditya Golatkar, Yusheng Xie, Alessandro
Achille, Ashwin Swaminathan, and Stefano Soatto. Diffusion soup: Model merging for text-to-
image diffusion models. arXiv preprint arXiv:2406.08431, 2024.
Avishek Joey Bose, Tara Akhound-Sadegh, Guillaume Huguet, Killian Fatras, Jarrid Rector-Brooks,
Cheng-Hao Liu, Andrei Cristian Nica, Maksym Korablyov, Michael Bronstein, and Alexander
Tong. Se(3)-stochastic flow matching for protein backbone generation. In The International
Conference on Learning Representations (ICLR), 2024.
Arwen Bradley and Preetum Nakkiran. Classifier-free guidance is a predictor-corrector. arXiv
preprint arXiv:2408.09000, 2024.
11

Published as a conference paper at ICLR 2025
Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe
Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video
generation models as world simulators. 2024. URL https://openai.com/research/
video-generation-models-as-world-simulators.
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar,
Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence:
Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.
Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative
flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design.
arXiv preprint arXiv:2402.04997, 2024.
Jinho Chang and Jong Chul Ye. Ldmol: Text-conditioned molecule diffusion model leveraging
chemically informative latent space. arXiv preprint arXiv:2405.17829, 2024.
Sheng Chen, Junjie Xie, Renlong Ye, David Daqiang Xu, and Yuedong Yang. Structure-aware dual-
target drug design through collaborative learning of pharmacophore combination and molecular
simulation. Chemical Science, 15(27):10366–10380, 2024.
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-
efficient exact attention with io-awareness. Advances in Neural Information Processing Systems,
35:16344–16359, 2022.
Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J Ragotte, Lukas F Milles,
Basile IM Wicky, Alexis Courbet, Rob J de Haas, Neville Bethel, et al. Robust deep learning–based
protein sequence design using proteinmpnn. Science, 378(6615):49–56, 2022.
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances
in neural information processing systems, 34:8780–8794, 2021.
Q. Dong, Lei L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, and Z. Sui. A survey for
in-context learning. arXiv preprint arXiv:2301.00234, 2022.
Yilun Du and Leslie Kaelbling. Compositional generative modeling: A single model is not all you
need. arXiv preprint arXiv:2402.01103, 2024.
Yilun Du, Shuang Li, and Igor Mordatch. Compositional visual generation with energy based models.
Advances in Neural Information Processing Systems, 33:6637–6647, 2020.
Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha
Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Composi-
tional generation with energy-based diffusion models and mcmc. In International conference on
machine learning, pp. 8489–8510. PMLR, 2023.
Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean
Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, et al. Faith and fate: Limits of
transformers on compositionality. Advances in Neural Information Processing Systems, 36, 2024.
Peter Ertl and Ansgar Schuffenhauer.
Estimation of synthetic accessibility score of drug-like
molecules based on molecular complexity and fragment contributions. Journal of cheminfor-
matics, 1:1–11, 2009.
Aditya Golatkar, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto. Training data
protection with compositional diffusion models. arXiv preprint arXiv:2308.01937, 2023.
A Herbert and M Sternberg. Maxcluster: A tool for protein structure comparison and clustering,
2008.
Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural
computation, 14(8):1771–1800, 2002.
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598,
2022.
12

Published as a conference paper at ICLR 2025
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in
Neural Information Processing Systems, 33:6840–6851, 2020.
Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A.
Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question
answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
pp. 20406–20417, October 2023.
Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W Coley,
Cao Xiao, Jimeng Sun, and Marinka Zitnik. Therapeutics data commons: Machine learning
datasets and tasks for drug discovery and development. Proceedings of Neural Information
Processing Systems, NeurIPS Datasets and Benchmarks, 2021.
Guillaume Huguet, James Vuckovic, Kilian Fatras, Eric Thibodeau-Laufer, Pablo Lemos, Riashat
Islam, Cheng-Hao Liu, Jarrid Rector-Brooks, Tara Akhound-Sadegh, Michael Bronstein, et al.
Sequence-augmented se (3)-flow matching for conditional protein backbone generation. arXiv
preprint arXiv:2405.20313, 2024.
Michael F Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian
smoothing splines. Communications in Statistics-Simulation and Computation, 18(3):1059–1076,
1989.
John B Ingraham, Max Baranov, Zak Costello, Karl W Barber, Wujie Wang, Ahmed Ismail, Vincent
Frappier, Dana M Lord, Christopher Ng-Thow-Hing, Erik R Van Vlack, et al. Illuminating protein
space with a programmable generative model. Nature, 623(7989):1070–1078, 2023.
Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Multi-objective molecule generation using
interpretable substructures. In International conference on machine learning, pp. 4849–4859.
PMLR, 2020.
Marco Jiralerspong, Avishek Joey Bose, Ian Gemp, Chongli Qin, Yoram Bachrach, and Gauthier
Gidel. Feature likelihood score: Evaluating generalization of generative models using samples,
2023.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott
Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models.
arXiv preprint arXiv:2001.08361, 2020.
Rafał Karczewski, Markus Heinonen, and Vikas Garg. Diffusion models as cartoonists! the curious
case of high density regions. arXiv preprint arXiv:2411.01293, 2024.
Lingkai Kong, Yuanqi Du, Wenhao Mu, Kirill Neklyudov, Valentin De Bortol, Haorui Wang, Dongxia
Wu, Aaron Ferber, Yi-An Ma, Carla P Gomes, et al. Diffusion models as constrained samplers for
optimization with unknown constraints. arXiv preprint arXiv:2402.18012, 2024.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention.
In Proceedings of the 29th Symposium on Operating Systems
Principles, pp. 611–626, 2023.
Seul Lee, Karsten Kreis, Srimukh Prasad Veccham, Meng Liu, Danny Reidenbach, Yuxing Peng,
Saee Paliwal, Weili Nie, and Arash Vahdat. Genmol: A drug discovery generalist with discrete
diffusion. arXiv preprint arXiv:2501.06158, 2025.
Yeqing Lin, Minji Lee, Zhao Zhang, and Mohammed AlQuraishi. Out of many, one: Designing
and scaffolding proteins at the scale of the structural universe with genie 2. arXiv preprint
arXiv:2405.15489, 2024.
Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Allan
dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido, et al. Language models of
protein sequences at the scale of evolution enable accurate structure prediction. bioRxiv, 2022.
Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching
for generative modeling. International Conference on Learning Representations, 2022.
13

Published as a conference paper at ICLR 2025
Sidney Lyayuga Lisanza, Jake Merle Gershon, Sam Tipps, Lucas Arnoldt, Samuel Hendel,
Jeremiah Nelson Sims, Xinting Li, and David Baker. Joint generation of protein sequence and
structure with rosettafold sequence space diffusion. bioRxiv, pp. 2023–05, 2023.
Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual
generation with composable diffusion models. In European Conference on Computer Vision, pp.
423–439. Springer, 2022a.
Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint
arXiv:2209.14577, 2022.
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and
transfer data with rectified flow. International Conference on Learning Representations, 2022b.
Michael Luo, Justin Wong, Brandon Trabucco, Yanping Huang, Joseph E. Gonzalez, Zhifeng Chen,
Ruslan Salakhutdinov, and Ion Stoica. Stylus: Automatic adapter selection for diffusion models.
Advances in Neural Information Processing Systems, 2024.
Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and
projection for dimension reduction, 2018.
Midjourney. https://www.midjourney.com/home/, 2023. Accessed: 2023-09-09.
Weili Nie, Arash Vahdat, and Anima Anandkumar. Controllable and compositional generation
with latent-space energy-based models. Advances in Neural Information Processing Systems, 34:
13497–13510, 2021.
Bernt Øksendal. Stochastic differential equations. In Stochastic differential equations, pp. 65–84.
Springer, 2003.
OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.
Stefano Peluchetti. Diffusion bridge mixture transports, Schrödinger bridge problems and generative
modeling. arXiv preprint arXiv:2304.00917, 2023.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning, pp.
8748–8763. PMLR, 2021.
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen,
and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine
Learning. PMLR, 2021.
Milong Ren, Tian Zhu, and Haicang Zhang. Carbonnovo: Joint design of protein structure and
sequence using a unified energy-based model. In Forty-first International Conference on Machine
Learning, 2024.
David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of chemical information
and modeling, 50(5):742–754, 2010.
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.
Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language
models a mirage? arxiv. arXiv preprint arXiv:2304.15004, 2023.
Arne Schneuing, Charles Harris, Yuanqi Du, Kieran Didi, Arian Jamasb, Ilia Igashov, Weitao Du,
Carla Gomes, Tom L Blundell, Pietro Lio, et al. Structure-based drug design with equivariant
diffusion models. Nature Computational Science, 4(12):899–909, 2024.
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi
Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B:
An open large-scale dataset for training next generation image-text models. Advances in Neural
Information Processing Systems, 2022.
14

Published as a conference paper at ICLR 2025
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. In International
Conference on Learning Representations, 2020.
Stability AI. https://stability.ai/stablediffusion, 2023. Accessed: 2023-09-09.
Jacob Steinhardt. AI Forecasting: One Year In . https://bounded-regret.ghost.io/
ai-forecasting-one-year-in/, 2022. Accessed: 2023-09-09.
Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry
without human demonstrations. Nature, 625(7995):476–482, 2024.
Brian L. Trippe, Jason Yim, Doug Tischer, David Baker, Tamara Broderick, Regina Barzilay, and
Tommi S. Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motif-
scaffolding problem. In The Eleventh International Conference on Learning Representations, 2023.
URL https://openreview.net/forum?id=6TxBxqNME1Y.
Michel Van Kempen, Stephanie S Kim, Charlotte Tumescheit, Milot Mirdita, Jeongjae Lee,
Cameron LM Gilchrist, Johannes Söding, and Martin Steinegger. Fast and accurate protein
structure search with foldseek. Nature biotechnology, 42(2):243–246, 2024.
Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computa-
tion, 23(7):1661–1674, 2011.
Chentong Wang, Yannan Qu, Zhangzhi Peng, Yukai Wang, Hongli Zhu, Dachuan Chen, and Longxing
Cao. Proteus: exploring protein structure generation for enhanced designability and efficiency.
bioRxiv, pp. 2024–02, 2024a.
Haorui Wang, Marta Skreta, Cher-Tian Ser, Wenhao Gao, Lingkai Kong, Felix Strieth-Kalthoff,
Chenru Duan, Yuchen Zhuang, Yue Yu, Yanqiao Zhu, et al. Efficient evolutionary search over
chemical space with large language models. arXiv preprint arXiv:2406.16976, 2024b.
Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E.
Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, Basile I. M.
Wicky, Nikita Hanikel, Samuel J. Pellock, Alexis Courbet, William Sheffler, Jue Wang, Preetham
Venkatesh, Isaac Sappington, Susana Vázquez Torres, Anna Lauko, Valentin De Bortoli, Emile
Mathieu, Sergey Ovchinnikov, Regina Barzilay, Tommi S. Jaakkola, Frank DiMaio, Minkyung
Baek, and David Baker. De novo design of protein structure and function with rfdiffusion. Nature,
620(7976):1089–1100, 2023. ISSN 1476-4687.
David Weininger. Smiles, a chemical language and information system. 1. introduction to methodol-
ogy and encoding rules. Journal of chemical information and computer sciences, 28(1):31–36,
1988.
Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong.
Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances
in Neural Information Processing Systems, 36, 2024.
Jason Yim, Andrew Campbell, Andrew Y. K. Foong, Michael Gastegger, José Jiménez-Luna, Sarah
Lewis, Victor Garcia Satorras, Bastiaan S. Veeling, Regina Barzilay, Tommi Jaakkola, and Frank
Noé. Fast protein backbone generation with se(3) flow matching, 2023a.
Jason Yim, Brian L Trippe, Valentin De Bortoli, Emile Mathieu, Arnaud Doucet, Regina Barzilay,
and Tommi Jaakkola. Se (3) diffusion model with application to protein backbone generation.
arXiv preprint arXiv:2302.02277, 2023b.
Jason Yim, Andrew Campbell, Emile Mathieu, Andrew YK Foong, Michael Gastegger, José Jiménez-
Luna, Sarah Lewis, Victor Garcia Satorras, Bastiaan S Veeling, Frank Noé, et al. Improved
motif-scaffolding with se (3) flow matching. arXiv preprint arXiv:2401.04082, 2024.
Ming Zhong, Yelong Shen, Shuohang Wang, Yadong Lu, Yizhu Jiao, Siru Ouyang, Donghan Yu,
Jiawei Han, and Weizhu Chen. Multi-lora composition for image generation. arXiv preprint
arXiv:2402.16843, 2024.
15

Published as a conference paper at ICLR 2025
APPENDIX
A
PROOFS FOR SEC. 2
A.1
PROOF OF PROP. 1
Proposition 1. [Reverse-time SDEs/ODE] Marginal densities qt(x) induced by Eq. (1) corre-
spond to the densities induced by the following SDE that goes back in time (τ = 1 −t) with the
corresponding initial condition
dxτ =

−ft(xτ) +
g2
t
2 + ξτ

∇log qt(xτ)

dτ +
p
2ξτdW τ , xτ=0 ∼q1(x0) ,
(2)
where W τ is the standard Wiener process in time τ, and ξτ is any positive schedule.
For the forward SDE we can write the corresponding Fokker-Planck equation
∂
∂tqt(x) = −

∇, qt(x)ft(x)

+ g2
t
2 ∆qt(x) ,
(19)
then for the inverse time τ = 1 −t, we have
∂
∂τ q1−τ(x) =

∇, q1−τ(x)f1−τ(x)

−g2
1−τ
2
∆q1−τ(x)
(20)
=

∇, q1−τ(x)

f1−τ(x) −g2
1−τ
2
∇log q1−τ(x)

(21)
= −

∇, q1−τ(x)

−f1−τ(x) +
g2
1−τ
2
+ ξτ

∇log q1−τ(x)

+ ξτ∆q1−τ(x) ,
(22)
which corresponds to the SDE
dxτ = dτ

−f1−τ(xτ) +
g2
1−τ
2
+ ξτ

∇log q1−τ(xτ)

+
p
2ξτdW τ .
(23)
A.2
PROOF OF PROP. 2
Proposition 2. [Ornstein–Uhlenbeck SDE] The time-dependent densities in Eq. (3) correspond
to the marginal densities of the following SDE, with the corresponding initial condition
dxt = ∂log αt
∂t
xt
|
{z
}
ft(xt)
dt +
r
2σ2
t
∂
∂t log σt
αt
|
{z
}
gt
dWt , x0 ∼q0(x0) .
(4)
Proof. For individual components qi
t from Eq. (3), we have to find the vector field satisfying the
continuity equation, which we can rewrite as
∂
∂t log qi
t(x) = −

∇x, vi
t(x)

−

∇x log qi
t(x), vi
t(x)

.
(24)
Using the formula for the density of the normal distribution qi
t(x) = N(x | αtµi, σ2
t ), we have
∇x log qi
t(x) = −1
σ2
t
(x −αtµi) ,
(25)
∂
∂t log qi
t(x) = d ∂
∂t log σt + 1
σ3
t
∂σt
∂t
x −αtµi2 + 1
σ2
t

x −αtµi, µi∂αt
∂t
(26)
=

∇x, ∂log σt
∂t
x

−
*
∇x log qi
t(x), ∂log σt
∂t
(x −αtµi) + ∂αt
∂t µi
|
{z
}
vi
t(x)
+
(27)
=

∇x, vi
t(x)

−

∇x log qi
t(x), vi
t(x)

.
(28)
16

Published as a conference paper at ICLR 2025
For the mixture of densities qt from Eq. (3), we can use Prop. 3 and write
vt(x) =
1
qt(x)N
N
X
i=1
qi
t(x)
∂log σt
∂t
(x −αtµi) + ∂αt
∂t µi

(29)
= ∂log σt
∂t
x +
∂αt
∂t −∂log σt
∂t
αt

1
qt(x)N
N
X
i=1
qi
t(x)µi .
(30)
At the same time
∇x log qt(x) =
1
qt(x)N
N
X
i=1
qi
t(x)

−1
σ2
t
(x −αtµi)

,
(31)
1
qt(x)N
N
X
i=1
qi
t(x)µi =
1
αt

σ2
t ∇x log qt(x) + x

.
(32)
Using this formula in Eq. (30), we have
vt(x) =
∂log σt
∂t
+ 1
αt
∂αt
∂t −∂log σt
∂t

x +
∂αt
∂t −∂log σt
∂t
αt
σ2
t
αt
∇x log qt(x)
(33)
= ∂log αt
∂t
x −σ2
t
∂
∂t log σt
αt
∇x log qt(x) .
(34)
Hence, we have
∂
∂tqt(x) = −

∇, qt(x)vt(x)

= −

∇, qt(x)
∂log αt
∂t
x

+ σ2
t
∂
∂t log σt
αt
∆qt(x) ,
(35)
which corresponds to the SDE in the statement.
A.3
PROOF OF PROP. 3
Proposition 3. [Superposition of ODEs (Liu, 2022)] The mixture density in Eq. (6) follows the
continuity equation with the superposed vector fields from Eq. (5), i.e.
∂
∂tqmix
t
(x) = −

∇x, qmix
t
(x)vt(x)

, vt(x) =
1
PN
j=1 ωjqj
t (x)
N
X
i=1
ωiqi
t(x)vi
t(x) .
(7)
Proof. By the straightforward substitution, we have
∂
∂tqmix
t
(x) =
m
X
i=1
ωi ∂
∂tqi
t(x) = −
m
X
i=1
ωi
∇x, qi
t(x)vi
t(x)

(36)
= −
*
∇x,
m
X
i=1
ωiqi
t(x)vi
t(x)
+
= −
*
∇x, qmix
t
(x)
qmix
t
(x)
m
X
i=1
ωiqi
t(x)vi
t(x)
+
(37)
= −
*
∇x, qmix
t
(x)
1
Pm
j=1 ωjqj
t (x)
m
X
i=1
ωiqi
t(x)vi
t(x)
|
{z
}
vt(x)
+
.
(38)
A.4
PROOF OF PROP. 4
Proposition 4. [Superposition of SDEs] The mixture qmix
t
(x) := PN
i=1 ωiqi
t(x) of density
marginals {qi
t(x)}N
i=1 induced by SDEs from Eq. (8) corresponds to the following SDE
dxτ = uτ(xτ)dτ + gτdW τ , ut(x) =
1
PN
j=1 ωjqj
t (x)
N
X
i=1
ωiqi
t(x)ui
t(x) .
(9)
17

Published as a conference paper at ICLR 2025
Proof. By the straightforward substitution, we have
∂
∂tqmix
t
(x) =
m
X
i=1
ωi ∂
∂tqi
t(x) =
m
X
i=1
ωi

−

∇x, qi
t(x)ui
t(x)

+ g2
t
2 ∆qi
t(x)

.
(39)
The first term is analogous to Prop. 3, i.e.
m
X
i=1
ωi −

∇x, qi
t(x)ui
t(x)

= −
*
∇x, qmix
t
(x)
1
Pm
j=1 ωjqj
t (x)
m
X
i=1
ωiqi
t(x)ui
t(x)
|
{z
}
ut(x)
+
.
(40)
The second term is
m
X
i=1
ωi g2
t
2 ∆qi
t(x) = g2
t
2 ∆
m
X
i=1
ωi∆qi
t(x) = g2
t
2 ∆qmix
t
(x) .
(41)
This results in the following PDE
∂
∂tqmix
t
(x) = −
*
∇x, qmix
t
(x)
1
Pm
j=1 ωjqj
t (x)
m
X
i=1
ωiqi
t(x)ui
t(x)
|
{z
}
ut(x)
+
+ g2
t
2 ∆qmix
t
(x) .
(42)
B
FLOWS AND DIFFUSION AS A SUPERPOSITION OF ELEMENTARY VECTOR
FIELDS
From Prop. 3, one can immediately get the main principle for the simulation-free training of generative
flow models, as demonstrated in the following proposition.
Proposition 7. [Superposition of L2-losses] For the parametric vector field model vt(x; θ)
with parameters θ, the L2-loss for the vector field from Prop. 3 can be decomposed into the
losses for the vector fields from Eq. (5), i.e.
Z 1
0
dt Eqmix
t
(x)∥vt(x) −vt(x; θ)∥2 =
N
X
i=1
ωi
Z 1
0
dt Eqi
t(x)
vi
t(x) −vt(x; θ)
2 + constant ,
where the constant does not depend on the parameters θ.
Proof. By the straightforward calculation, we have
Z 1
0
dt Eqmix
t
(x)∥vt(x) −vt(x; θ)∥2 =
Z 1
0
dt Eqmix
t
(x)
h
∥vt(x)∥2 −2

vt(x), vt(x; θ)

+ ∥vt(x; θ)∥2i
,
where the first term is constant w.r.t. θ and the last term is amenable for a straightforward estimation.
The middle term, according to Prop. 3, is
Z
dx qmix
t
(x)

vt(x), vt(x; θ)

=
Z
dx
m
X
i=1
ωiqi
t(x)

vi
t(x), vt(x; θ)

.
(43)
Thus, we have
Z 1
0
dt Eqmix
t
(x)

∥vt(x)∥2 −2

vt(x), vt(x; θ)

+ ∥vt(x; θ)∥2

(44)
=
Z 1
0
dt

Eqmix
t
(x)∥vt(x)∥2 −
m
X
i=1
ωiEqi
t(x)
vi
t(x)
2
|
{z
}
constant
+
m
X
i=1
ωiEqi
t(x)
vi
t(x)
2−
(45)
−2
m
X
i=1
ωiEqi
t(x)

vi
t(x), vt(x; θ)

+
m
X
i=1
ωiEqi
t(x)∥vt(x; θ)∥2

(46)
18

Published as a conference paper at ICLR 2025
=
Z 1
0
dt
m
X
i=1
ωiEqi
t(x)
vi
t(x) −vt(x; θ)
2 + constant .
(47)
The proof for proposition follows a simple extension of Liu (2022). Consequently, the generative
modeling problem explicitly boils down to learning a time-dependent vector field vt(x, θ). Moreover,
for the generative modeling task, the noising process is required to satisfy the following boundary
conditions qi
0(x) = δ(x −µi) and qi
1(x) = pnoise(x) , ∀i ∈D. Generating samples that resemble
pdata(µ) during inference then simply amounts to solving the reverse-time ODE with the learned
model vt(x; θ) starting from a noisy sample x1 ∼pnoise(x).
Diffusion models can be incorporated into the flow-based framework by selecting a Gaussian forward
process which gives the marginal density at a timestep t as the mixture of corresponding Gaussians,
qi
t(x) = N(x | αtµi, σ2
t ) , qt(x) = 1
N
N
X
i=1
qi
t(x) .
(48)
Under this Gaussian diffusion framework, we can express the vector fields analytically using the
score, which is formalized by the following proposition.
Proposition 8. [Diffusion processes] The time-dependent densities in Eq. (48) satisfy the
following continuity equations
∂
∂tqi
t(x) = −

∇x, qi
t(x)vi
t(x)

, vi
t(x) = ∂log σt
∂t
(x −αtµi) + ∂αt
∂t µi ,
(49)
∂
∂tqt(x) = −

∇x, qt(x)vt(x)

, vt(x) = ∂log αt
∂t
x −

σ2
t
∂
∂t log σt
αt

∇x log qt(x) .
(50)
The proof repeats the proof for Prop. 2 in App. A.2 since Eq. (50) corresponds to the Ornstein-
Uhlenbeck process:
dxt = ∂log αt
∂t
xtdt +
s
2

σ2
t
∂
∂t log σt
αt

dWt.
(51)
Diffusion models using a Gaussian perturbation kernel enjoy the benefit of giving an exact expression
for the Stein score of the perturbed data which leads to the celebrated denoising score matching objec-
tive (Vincent, 2011; Ho et al., 2020). Analogously to Prop. 7 we can write the denoising score match-
ing objective as the superposition of the scores as demonstrated in the following proposition.
Proposition 9. [Denoising Score Matching (Vincent, 2011)] For the parametric score model
∇log qt(x; θ) with parameters θ, the score matching objective can be decomposed into the
corresponding objectives for the individual scores, i.e.
Z 1
0
dt Eqt(x)∥∇log qt(x) −∇log qt(x; θ)∥2 =
= 1
N
N
X
i=1
Z 1
0
dt Eqi
t(x)
∇log qi
t(x) −∇log qt(x; θ)
2 + constant ,
where the constant does not depend on the parameters θ and ∇log qi
t(x) = −1
σ2
t (x −αtµi).
Proof. From Prop. 8, we have
vi
t(x) = ∂log σt
∂t
(x −αtµi) + ∂αt
∂t µi
(52)
= ∂log σt
∂t
(x −(x + σ2
t ∇log qi
t(x))) + ∂αt
∂t
1
αt
(x + σ2
t ∇log qi
t(x))
(53)
19

Published as a conference paper at ICLR 2025
= ∂log αt
∂t
x + σ2
t
∂
∂t log αt
σt
∇x log qi
t(x)
(54)
vt(x) = ∂log αt
∂t
x + σ2
t
∂
∂t log αt
σt
∇x log qt(x) .
(55)
Applying the same change of variables to the parametric model
vt(x; θ) = ∂log αt
∂t
x + σ2
t
∂
∂t log αt
σt
∇x log qt(x; θ) ,
(56)
and using Prop. 7, we have
Z 1
0
dt Eqt(x)∥vt(x) −vt(x; θ)∥2 =
m
X
i=1
1
N
Z 1
0
dt Eqi
t(x)
vi
t(x) −vt(x; θ)
2 + constant , (57)
Z 1
0
dt Eqt(x)∥∇x log qt(x) −∇x log qt(x; θ)∥2
(58)
=
m
X
i=1
1
N
Z 1
0
dt Eqi
t(x)
∇x log qi
t(x) −∇x log qt(x; θ)
2 +
constant

σ2
t
∂
∂t log αt
σt
 ,
(59)
which concludes the proof.
C
DENSITY ESTIMATORS
Proposition 5. [Smooth density estimator] For the integral curve x(t) solving dx/dt = ut(xt),
and the density qi
t(x(t)) satisfying the continuity equation ∂
∂tqi
t(x) = −

∇x, qi
t(x)vi
t(x)

, the
log-density along the curve changes according to the following ODE
d
dt log qi
t(x(t)) = −

∇x, vi
t(x)

−

∇x log qi
t(x), vi
t(x) −ut(x)

.
(10)
Proof. By straightforward computation, we have
d
dt log qi
t(x(t)) = ∂
∂t log qi
t(x) +

∇x log qi
t(x), dx
dt

,
(60)
and using the continuity equation ∂
∂tqi
t(x) = −

∇x, qi
t(x)vi
t(x)

, we have
∂
∂t log qi
t(x) = −

∇x, vi
t(x)

−

∇x log qi
t(x), vi
t(x)

,
(61)
d
dt log qi
t(x(t)) = −

∇x, vi
t(x)

−

∇x log qi
t(x), vi
t(x) −ut(x)

.
(62)
Theorem 1. [Itô density estimator] Consider time-dependent density qt(x) induced by the
marginals of the following SDE
dxt = ft(xt)dt + gtdWt , xt=0 ∼q0(x) , t ∈[0, 1] ,
(11)
where dWt is the Wiener process. For the reverse-time (τ = 1 −t) SDEs with any vector field
uτ and the same diffusion coefficient gt, i.e.
dxτ = uτ(xτ)dτ + g1−τdW τ , τ ∈[0, 1] ,
(12)
the change of the log-density log qτ(xτ) follows the following SDE
d log q1−τ(xτ) =

dxτ, ∇log q1−τ(xτ)

+

∇, f1−τ(xτ)

+
+

f1−τ(xτ) −g2
1−τ
2
∇log q1−τ(xτ), ∇log q1−τ(xτ)

dτ .
(13)
20

Published as a conference paper at ICLR 2025
Proof. The Fokker-Planck equation describing the evolution of the marginals for the forward process
is
∂qt(x)
∂t
= −

∇, ft(x)qt(x)

+ g2
t
2 ∆qt(x) ,
(63)
hence, for the inverse time τ = 1 −t, we have
∂
∂τ q1−τ(x) =

∇, f1−τ(x)q1−τ(x)

−g2
1−τ
2
∆q1−τ(x)
(64)
∂
∂τ log q1−τ(x) =

∇, f1−τ(x)

+

∇log q1−τ(x), f1−τ(x)

−
−g2
1−τ
2
∆log q1−τ(x) −g2
1−τ
2
∥∇log q1−τ(x)∥2 .
(65)
For the following reverse-time SDE
dxτ = uτ(xτ)dτ + g1−τdW τ ,
(66)
using Itô’s lemma for log q1−τ(xτ), we have
d log q1−τ(xτ) =
 ∂
∂τ log q1−τ(xτ) +

∇log q1−τ(xτ), uτ(xτ)

+
(67)
+ g2
1−τ
2
∆log q1−τ(xτ)

dτ + g1−τ

∇log q1−τ(xτ), dW τ

.
(68)
Using Eq. (65), we have
d log q1−τ(xτ) =

∇log q1−τ(xτ), f1−τ(xτ) −g2
1−τ
2
∇log q1−τ(xτ)

+
(69)
+

∇, f1−τ(xτ)

dτ +
*
∇log q1−τ(xτ), uτ(xτ)dτ + g1−τdW τ
|
{z
}
dxτ
+
.
(70)
C.1
PROOF OF PROP. 6
Proposition 6. [Density control] For the SDE
dxτ =
M
X
j=1
κjuj
τ(xτ)dτ + g1−τdW τ ,
(16)
where κ are the weights of different models and P
j κj = 1, one can find κ that satisfies
d log qi
1−τ(xτ) = d log qj
1−τ(xτ) , ∀i, j ∈[M] ,
(17)
by solving a system of M + 1 linear equations w.r.t. κ.
Proof. Using the result of Thm. 1, for the density change of every model, we have
d log qi
1−τ(xτ) =

dxτ, ∇log qi
1−τ(xτ)

+

∇, f1−τ(xτ)

+
(71)
+

f1−τ(xτ) −g2
1−τ
2
∇log qi
1−τ(xτ), ∇log qi
1−τ(xτ)

dτ
(72)
=
M
X
j=1
κjdτ

uj
τ(xτ), ∇log qi
1−τ(xτ)

+

∇, f1−τ(xτ)

+
(73)
+

g1−τdW τ +

f1−τ(xτ) −g2
1−τ
2
∇log qi
1−τ(xτ)

dτ, ∇log qi
1−τ(xτ)

.
(74)
21

Published as a conference paper at ICLR 2025
Let us denote
aij = dτ

uj
τ(xτ), ∇log qi
1−τ(xτ)

,
(75)
bi =

∇, f1−τ(xτ)

+

g1−τdW τ +

f1−τ(xτ) −g2
1−τ
2
∇log qi
1−τ(xτ)

dτ, ∇log qi
1−τ(xτ)

.
(76)
Then the change of densities can be written as the following linear transformation


d log q1
1−τ(xτ)
. . .
d log qi
1−τ(xτ)
. . .
d log qM
1−τ(xτ)

= Aκ + b .
(77)
Note that we want to find κ such that d log qi
1−τ = d log qj
1−τ , ∀i, j ∈[M] and P
j κj = 1. This is
equivalent to the following system of M + 1 linear equations




a11
. . .
a1M
−1
...
...
...
−1
aM1
. . .
aMM
−1
1
. . .
1
0








κ1
...
κM
d log



=




−b1
...
−bM
1



,
(78)
where we have introduced new variable d log = d log qi
1−τ , ∀i ∈[M].
D
DISCRETE-TIME PERSPECTIVE ON THM. 1
In this section we derive Thm. 1 from another perspective by operating with discrete time transition
kernels and the detailed balance. Namely, we aim to compute the marginal density by starting from
the detailed balance equation, which states the equivalence of the following joint densities:
qt(y)k∆t(x | y) = qt+∆t(x)r∆t(y | x).
(79)
The detailed balance condition simply states the pair (x, y) can be sampled in two equivalent ways:
1.) sampling y from the marginal qt(y) and then sampling x via the noising kernel k∆t(x | y), or 2.)
sampling x from qt+∆t(y) and then denoising it via r∆t(y | x). Indeed, there are infinitely many valid
kernels that may satisfy the detailed balance; we make a specific choice informed by the following
principle. We aim to construct a universal noising kernel that is independent of the data distribution
or other densities, i.e. pdata(·), qt(·), qt+∆t(·). Remarkably, this principle results in a unique kernel
choice, which we formalize in the following proposition.
Proposition 10. [Universal noising kernel] For any data density pdata(µ), for the continuous
noising process qt(x) =
R
dµ N(x | αtµ, σ2
t )pdata(µ), there exists unique noising kernel
k∆t(x | y) = N

x

αt+∆t
αt
y, S2
t+∆t

, qt+∆t(x) =
Z
dy k∆t(y | x)qt(y),
(80)
where S2
t+∆t = σ2
t+∆t −σ2
t
α2
t+∆t
α2
t
and it is independent of the densities pdata(·), qt(·), qt+∆t(·).
The proof for this proposition is presented in App. D.1. Note that the proposition holds exactly for
any ∆t—i.e. there are no any assumptions on the scale of ∆t or any approximations.
Once the noising kernel is fixed, the detailed balance Eq. (79) uniquely defines the denoising kernel
r∆t(y | x) that propagates samples back in time. However, the analytic form of this kernel depends on
the densities qt+∆t(x) and qt(y), which are unavailable for the modern large-scale diffusion models
that are served using limited API endpoints. Therefore, we consider the Gaussian approximation of
the reverse kernel and justify its applicability in the following theorem.
Theorem 2. [Denoising kernel] For the universal noising kernel k∆t(x | y), the Gaussian
approximation of the corresponding reverse kernel r∆t(y | x) = k∆t(x | y)qt(y)/qt+∆t(x) is
˜r∆t(y | x) = N
 
y

αt
αt+∆t
x +
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x),

αt
αt+∆t
St+∆t
2!
,
(81)
22

Published as a conference paper at ICLR 2025
which corresponds to a single step of the Euler discretization of the following SDE
dxτ = −
 ∂
∂t log αtx −2σ2
t
∂
∂t log σt
αt
∇x log qt(x)

· dτ +
r
2σ2
t
∂
∂t log σt
αt
dWτ ,
(82)
and
DKL(˜r∆t(y | x)∥r∆t(y | x)) = o(∆t) .
(83)
See proof in App. D.3. Given the noising kernel from Prop. 10 and the approximation of the reverse
kernel from Thm. 2, we propose to estimate the log marginal density of the current sample using the
detailed balance as follows,
log qt(y) −log qt+∆t(x) = −log k∆t(x | y) + log ˜r∆t(y | x) + log r∆t(y | x)
˜r∆t(y | x) ,
(84)
where the last term involves the unknown reversed kernel. We argue that this term can be disregarded
due to Lemma 1. Namely, in the following theorem, we study the distribution of the last term for
different samples y generated from x and argue that this term can be ignored.
Theorem 3. [Detailed balance density estimator error] For y = x + ∆t · v + gt+∆t
√
∆tε,
where gt+∆t =
q
2σ2
t
∂
∂t log σt
αt and ε is a standard normal random variable, the following
estimator is unbiased w.r.t. the generated samples y, i.e.
Eε

log r∆t(y | x)
˜r∆t(y | x)

= o(∆t) ,
(85)
and the variance is
IDε

log r∆t(y | x)
˜r∆t(y | x)

= ∆t2 g4
t+∆t
4
IDε

εT ∂2 log qt(x)
∂x2
ε

+ o(∆t2) ,
(86)
where ∂2 log qt+∆t(x)
∂x2
is the Hessian matrix of the log-density.
We postpone the proof of this theorem until App. D.4. Note that the next variable y is completely
defined by the random variable ε. Thus, the mean and the variance are evaluated taking into
account different outcomes of y.
However, the variance of this distribution is related to the
variance of Hutchinson’s estimator. Indeed, for the Euler integration scheme of the log-density
log qt+∆t(xt+∆t) = log q(xt) −∆t ·

∇x, vt(xt)

, Hutchinson’s estimator is

∇x, vt(xt)

= Eε

εT ∂vt(xt)
∂xt
ε

≃εT ∂vt(xt)
∂xt
ε , ε ∼N(0, 1) ,
(87)
where ∂vt(xt)
∂xt
is the Jacobian matrix. The variance of this estimator is
IDε[log qt+∆t(xt+∆t)] = ∆t2IDε

εT ∂vt(xt)
∂xt
ε

.
(88)
Note that here the variable ε is newly introduced and is not related anyhow to xt+∆t.
Thus, we arrive at the following estimator of the change of density
log qt(y) −log qt+∆t(x) ≈−log k∆t(x | y) + log ˜r∆t(y | x) .
(89)
We further highlight that since the generation via the inference process happens in many steps with
small ∆t, we approximate the relation Eq. (89) up to the second order terms in ∆t.
Proposition 11. [Recurrence relation for the log-density] For y = x + ∆t · v + gt+∆t
√
∆tε,
where gt+∆t =
q
2σ2
t
∂
∂t log σt
αt and ε is a standard normal random variable, the estimator
from Eq. (89) can be expanded as follows
−log k∆t(x | y) + log ˜r∆t(y | x) = d∆t ∂
∂t log αt+∆t −∆tg2
t+∆t
2
∥∇log qt+∆t(x)∥2+
+

∆t · v + gt+∆t
√
∆tε + ∆t ∂
∂t log αt+∆tx, ∇log qt+∆t(x)

+ o(∆t)
(90)
23

Published as a conference paper at ICLR 2025
The proof for this proposition is included in App. D.5. For practical use cases we note that Eq. (90)
is the final form of our density estimator.
D.1
PROOF OF PROP. 10
Proposition 10. [Universal noising kernel] For any data density pdata(µ), for the continuous
noising process qt(x) =
R
dµ N(x | αtµ, σ2
t )pdata(µ), there exists unique noising kernel
k∆t(x | y) = N

x

αt+∆t
αt
y, S2
t+∆t

, qt+∆t(x) =
Z
dy k∆t(y | x)qt(y),
(80)
where S2
t+∆t = σ2
t+∆t −σ2
t
α2
t+∆t
α2
t
and it is independent of the densities pdata(·), qt(·), qt+∆t(·).
Proof.
qt(x) =
Z
dµ N(x | αtµ, σ2
t )pdata(µ)
(91)
Let’s derive the incremental kernel k∆t(x | y) from this formula, i.e.
qt+∆t(x) =
Z
dy k∆t(x | y)qt(y)
(92)
Z
dµ N(x | αt+∆tµ, σ2
t+∆t)pdata(µ) =
Z
dy k∆t(x | y)
Z
dµ N(y | αtµ, σ2
t )pdata(µ)
(93)
Z
dµ

N(x | αt+∆tµ, σ2
t+∆t) −
Z
dy k∆t(x | y)N(y | αtµ, σ2
t )

pdata(µ) = 0
(94)
N(x | αt+∆tµ, σ2
t+∆t) =
Z
dy k∆t(x | y)N(y | αtµ, σ2
t )
(95)
Obviously, one can perform the following change of variables
x = αt+∆tµ + y −αtµ
σt
S + Rε , where y ∼N(y | αtµ, σ2
t ) , and ε ∼N(ε | 0, 1) ,
(96)
where we have to take S2 + R2 = σ2
t+∆t .
(97)
To make the kernel independent of µ we have to choose R = σt
αt+∆t
αt
, then the kernel is
k∆t(x | y) = N

x | αt+∆t
αt
y, σ2
t+∆t −σ2
t
α2
t+∆t
α2
t
|
{z
}
S2
t+∆t

.
(98)
D.2
LEMMA 1
Lemma 1. [Reverse kernel lemma] For the universal noising kernel k∆t(x | y), the correspond-
ing reverse kernel r∆t(y | x) = k∆t(x | y)qt(y)/qt+∆t(x), and its Gaussian approximation
˜r∆t(y | x) = N
 
y

αt
αt+∆t
x +
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x),

αt
αt+∆t
St+∆t
2!
,
(99)
we have
log r∆t(x + ∆t · v + gt+∆t
√
∆tε | x)
˜r∆t(x + ∆t · v + gt+∆t
√
∆tε | x)
=
(100)
= ∆t · g2
t+∆t
2

∆log qt+∆t(x) −εT ∂2 log qt(x)
∂x2
ε

+ o(∆t) ,
(101)
where gt+∆t =
q
2σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t .
24

Published as a conference paper at ICLR 2025
Proof. From the detailed balance equation, we have
log r∆t(y | x) = log k∆t(x | y) + log qt(y) −log qt+∆t(x)
(102)
=
−1
2S2
t+∆t

x −αt+∆t
αt
y
2
−d
2 log 2πS2
t+∆t + log qt(y) −log qt+∆t(x) , (103)
where we use the definition of the forward kernel. Thus, the difference between kernels is
log r∆t(y | x)
˜r∆t(y | x) =
−1
2S2
t+∆t

x −αt+∆t
αt
y
2
−d
2 log 2πS2
t+∆t + log qt(y) −log qt+∆t(x)+
(104)
+
1
2(St+∆t
αt
αt+∆t )2

y −
αt
αt+∆t
x −
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x)
2
+
(105)
+ d
2 log 2πS2
t+∆t + d
2 log

αt
αt+∆t
2
.
(106)
Opening the brackets, we have
log r∆t(y | x)
˜r∆t(y | x) =

−1
2S2
t+∆t

x −αt+∆t
αt
y
2
+ log qt(y) −log qt+∆t(x)+
(107)
+

1
2S2
t+∆t
αt+∆t
αt
y −x
2
−
αt+∆t
αt
y −x, ∇log qt+∆t(x)

+
(108)
+ S2
t+∆t
2
∥∇log qt+∆t(x)∥2 + d
2 log

αt
αt+∆t
2
.
(109)
The constants can be estimated as follows
S2
t+∆t = σ2
t+∆t −σ2
t
α2
t+∆t
α2
t
= σ2
t+∆t −(σ2
t+∆t −2dtσt+∆t
∂σt+∆t
∂t
+ o(∆t))α2
t+∆t
α2
t
(110)
= σ2
t+∆t −(σ2
t+∆t −2dtσt+∆t
∂σt+∆t
∂t
+ o(∆t))

1 + 2dtα2
t+∆t
α3
t+∆t
∂αt+∆t
∂t
+ o(∆t)

(111)
= 2dtσ2
t+∆t
∂
∂t log σt+∆t
αt+∆t
+ o(∆t) ,
(112)
and
αt+∆t
αt
= 1 −αt+∆t
α2
t+∆t
∂αt+∆t
∂t
(−∆t) + o(∆t) = 1 + ∆t∂log αt+∆t
∂t
+ o(∆t)
(113)
d
2 log

αt
αt+∆t
2
= −dtd ∂
∂t log αt+∆t + o(∆t) .
(114)
Thus, we have
log r∆t(y | x)
˜r∆t(y | x) = log qt(y) −log qt+∆t(x) + ∆tσ2
t+∆t
∂
∂t log σt+∆t
αt+∆t
∥∇log qt+∆t(x)∥2−
(115)
−dtd ∂
∂t log αt+∆t −

(1 + ∆t∂log αt+∆t
∂t
)y −x, ∇log qt+∆t(x)

+ o(∆t) .
(116)
From Prop. 8, the time-derivative of the density is
∂
∂t log qt(x) = −

∇, vt(x)

−

∇log qt(x), vt(x)

(117)
= −d∂log αt
∂t
+

σ2
t
∂
∂t log σt
αt

∆log qt(x)−
(118)
−∂log αt
∂t

∇log qt(x), x

+

σ2
t
∂
∂t log σt
αt

∥∇log qt(x)∥2 .
(119)
25

Published as a conference paper at ICLR 2025
Hence, we have
log qt(x) −log qt+∆t(x) = −∆t ∂
∂t log qt+∆t(x) + o(∆t)
(120)
= ∆td∂log αt+∆t
∂t
−∆t

σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t

∆log qt+∆t(x)+
(121)
+∆t∂log αt+∆t
∂t

∇log qt+∆t(x), x

−∆t

σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t

∥∇log qt+∆t(x)∥2 .
(122)
Using it in Eq. (116), we have
log r∆t(y | x)
˜r∆t(y | x) = log qt(y) −log qt(x) −∆t

σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t

∆log qt+∆t(x)−
(123)
−(1 + ∆t∂log αt+∆t
∂t
)

y −x, ∇log qt+∆t(x)

+ o(∆t) .
(124)
Finally, we consider y = x + ∆t · v + gt+∆t
√
∆tε. Then we can estimate
log qt(y) = log qt(x) +

∇log qt(x), y −x

+ 1
2(y −x)T ∂2 log qt(x)
∂x2
(y −x)
(125)
+ o(∥y −x∥2)
(126)
= log qt(x) +
D
∇log qt(x), ∆t · v + gt+∆t
√
∆tε
E
(127)
+ ∆tg2
t+∆t
2
εT ∂2 log qt(x)
∂x2
ε + o(∆t) ,
(128)
where ∂2 log qt(x)
∂x2
denotes the Hessian of the log-density. Thus, for y = x + ∆t · v + gt+∆t
√
∆tε,
we have
log r∆t(y | x)
˜r∆t(y | x) = ∆tg2
t+∆t
2
εT ∂2 log qt(x)
∂x2
ε −∆t

σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t

∆log qt+∆t(x)+ (129)
+
D
∇log qt(x) −∇log qt+∆t(x), ∆t · v + gt+∆t
√
∆tε
E
+ o(∆t) . (130)
The last term is o(∆t) by expanding the scores in time, hence, for
g2
t+∆t
2
= σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t , we
have
log r∆t(x + ∆t · v + gt+∆t
√
∆tε | x)
˜r∆t(x + ∆t · v + gt+∆t
√
∆tε | x)
=
(131)
= ∆t

σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t

∆log qt+∆t(x) −εT ∂2 log qt(x)
∂x2
ε

+ o(∆t) .
(132)
D.3
PROOF OF THM. 2
Theorem 2. [Denoising kernel] For the universal noising kernel k∆t(x | y), the Gaussian
approximation of the corresponding reverse kernel r∆t(y | x) = k∆t(x | y)qt(y)/qt+∆t(x) is
˜r∆t(y | x) = N
 
y

αt
αt+∆t
x +
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x),

αt
αt+∆t
St+∆t
2!
,
(81)
which corresponds to a single step of the Euler discretization of the following SDE
dxτ = −
 ∂
∂t log αtx −2σ2
t
∂
∂t log σt
αt
∇x log qt(x)

· dτ +
r
2σ2
t
∂
∂t log σt
αt
dWτ ,
(82)
and
DKL(˜r∆t(y | x)∥r∆t(y | x)) = o(∆t) .
(83)
26

Published as a conference paper at ICLR 2025
Proof. To find the reverse the kernel, we first use Tweedie’s formula to find the expectation, i.e.
∇log qt+∆t(x) = −
1
qt+∆t(x)
Z
dy
1
S2
t+∆t

x −αt+∆t
αt
y

k∆t(x | y)qt(y)
(133)
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x) = −
1
qt+∆t(x)
Z
dy

αt
αt+∆t
x −y

k∆t(x | y)qt(y)
(134)
αt
αt+∆t
x +
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x) =
Z
dy y k∆t(x | y)qt(y)
qt+∆t(x)
=
Z
dy yr∆t(y | x) .
(135)
Thus, we are going to approximate
˜r∆t(y | x) = N
 
y

αt
αt+∆t
x +
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x),

αt
αt+∆t
St+∆t
2!
(136)
where

αt
αt+∆t St+∆t
2
is chosen to match the leading term of k∆t(x | y). The constants can be
estimated as follows
S2
t+∆t = σ2
t+∆t −σ2
t
α2
t+∆t
α2
t
= σ2
t+∆t −(σ2
t+∆t −2∆tσt+∆t
∂σt+∆t
∂t
+ o(∆t))α2
t+∆t
α2
t
(137)
= σ2
t+∆t −(σ2
t+∆t −2∆tσt+∆t
∂σt+∆t
∂t
+ o(∆t))

1 + 2∆tα2
t+∆t
α3
t+∆t
∂αt+∆t
∂t
+ o(∆t)

(138)
= 2∆tσ2
t+∆t
∂
∂t log σt+∆t
αt+∆t
+ o(∆t) ,
(139)
and
αt
αt+∆t
= 1 +
1
αt+∆t
∂αt+∆t
∂t
(−∆t) + o(∆t) = 1 −∆t∂log αt+∆t
∂t
+ o(∆t) .
(140)
Thus, y can be generated as
y = x −∆t∂log αt+∆t
∂t
x + 2∆tσ2
t+∆t
∂
∂t log σt+∆t
αt+∆t
∇log qt+∆t(x)+
(141)
+
s
2∆tσ2
t+∆t
∂
∂t log σt+∆t
αt+∆t
ε ,
(142)
where ε is a standard normal random variable. This corresponds to the single step of the Euler
discretization of the following SDE
dxt = −
 ∂
∂t log αtx −2σ2
t
∂
∂t log σt
αt
∇x log qt(x)

· ∆t +
r
2σ2
t
∂
∂t log σt
αt
dWt .
(143)
For the KL-divergence between the reverse kernel and its Gaussian approximation, we have
DKL(˜r∆t(y | x)∥r∆t(y | x)) = Eε log ˜r∆t(x + ∆t · v + gt+∆t
√
∆tε | x)
r∆t(x + ∆t · v + gt+∆t
√
∆tε | x)
+ o(∆t) ,
(144)
where
v = −
 ∂
∂t log αtx −2σ2
t
∂
∂t log σt
αt
∇x log qt(x)

,
and
gt+∆t =
r
2σ2
t
∂
∂t log σt
αt
.
(145)
From Lemma 1, we have
DKL (˜r∆t(y | x)∥r∆t(y | x)) =
(146)
= ∆t

σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t

Eε

εT ∂2 log qt(x)
∂x2
ε −∆log qt+∆t(x)

+ o(∆t)
(147)
= o(∆t) .
(148)
D.4
PROOF OF THM. 3
27

Published as a conference paper at ICLR 2025
Theorem 3. [Detailed balance density estimator error] For y = x + ∆t · v + gt+∆t
√
∆tε,
where gt+∆t =
q
2σ2
t
∂
∂t log σt
αt and ε is a standard normal random variable, the following
estimator is unbiased w.r.t. the generated samples y, i.e.
Eε

log r∆t(y | x)
˜r∆t(y | x)

= o(∆t) ,
(85)
and the variance is
IDε

log r∆t(y | x)
˜r∆t(y | x)

= ∆t2 g4
t+∆t
4
IDε

εT ∂2 log qt(x)
∂x2
ε

+ o(∆t2) ,
(86)
where ∂2 log qt+∆t(x)
∂x2
is the Hessian matrix of the log-density.
Proof. From Lemma 1, we have
Eε log r∆t(y | x)
˜r∆t(y | x) = Eε∆tg2
t+∆t
2

∆log qt+∆t(x) −εT ∂2 log qt(x)
∂x2
ε

+ o(∆t)
(149)
= ∆tg2
t+∆t
2

∆log qt+∆t(x) −Eε Tr

εT ∂2 log qt(x)
∂x2
ε

+ o(∆t)
(150)
= ∆tg2
t+∆t
2

∆log qt+∆t(x) −Eε Tr
∂2 log qt(x)
∂x2

+ o(∆t)
(151)
= o(∆t) .
(152)
For the variance, we have
IDε log r∆t(y | x)
˜r∆t(y | x) = ∆t2 g4
t+∆t
4
Eε

∆log qt+∆t(x) −εT ∂2 log qt(x)
∂x2
ε
2
+ o(∆t2)
(153)
= ∆t2 g4
t+∆t
4
Eε

Eη

ηT ∂2 log qt(x)
∂x2
η

−εT ∂2 log qt(x)
∂x2
ε
2
+ o(∆t2)
(154)
= ∆t2 g4
t+∆t
4
IDε

εT ∂2 log qt(x)
∂x2
ε

+ o(∆t2) ,
(155)
where we used a standard normal random variable η.
D.5
PROOF OF PROP. 11
Proposition 11. [Recurrence relation for the log-density] For y = x + ∆t · v + gt+∆t
√
∆tε,
where gt+∆t =
q
2σ2
t
∂
∂t log σt
αt and ε is a standard normal random variable, the estimator
from Eq. (89) can be expanded as follows
−log k∆t(x | y) + log ˜r∆t(y | x) = d∆t ∂
∂t log αt+∆t −∆tg2
t+∆t
2
∥∇log qt+∆t(x)∥2+
+

∆t · v + gt+∆t
√
∆tε + ∆t ∂
∂t log αt+∆tx, ∇log qt+∆t(x)

+ o(∆t)
(90)
28

Published as a conference paper at ICLR 2025
Proof. Indeed,
log ˜r∆t(y | x)
k∆t(x | y) =
1
2B2
t+∆t

x −αt+∆t
αt
y
2
+ d
2 log 2πS2
t+∆t−
(156)
−
1
2(St+∆t
αt
αt+∆t )2

y −
αt
αt+∆t
x −
αt
αt+∆t
S2
t+∆t∇log qt+∆t(x)
2
−(157)
−d
2 log 2πS2
t+∆t −d log
αt
αt+∆t
(158)
= −d log
αt
αt+∆t
+
αt+∆t
αt
y −x, ∇log qt+∆t(x)

−
(159)
−S2
t+∆t
2
∥∇log qt+∆t(x)∥2
(160)
Expanding the time around t + ∆t, we have
log ˜r∆t(y | x)
k∆t(x | y) = d∆t ∂
∂t log αt+∆t +

(1 + ∆t ∂
∂t log αt+∆t)y −x, ∇log qt+∆t(x)

−
(161)
−∆t

σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t

∥∇log qt+∆t(x)∥2 + o(∆t) .
(162)
For y = x + ∆t · v +
√
∆tgt+∆tε, gt+∆t =
q
2σ2
t+∆t
∂
∂t log σt+∆t
αt+∆t , we have
log ˜r∆t(y | x)
k∆t(x | y) = d∆t ∂
∂t log αt+∆t −∆tg2
t+∆t
2
∥∇log qt+∆t(x)∥2+
(163)
+

∆t · v + gt+∆t
√
∆tε + ∆t ∂
∂t log αt+∆tx, ∇log qt+∆t(x)

+ o(∆t) .
(164)
E
ADDITIONAL RELATED WORK
Protein generation. Structure-based de novo protein design using deep generative models has
recently seen a surge in interest, with a particular emphasis on diffusion-based approaches (Watson
et al., 2023; Yim et al., 2023b), and also flow matching methods (Bose et al., 2024; Yim et al.,
2023a; 2024; Huguet et al., 2024). Moreover, building on the initial SE(3) equivariant diffusion
paradigm multiple recent approaches have sought to increase the performance of the methods
through architectural innovations (Wang et al., 2024a), conditioning on auxiliary modalities such as
sequence or sidechains (Ingraham et al., 2023; Lin et al., 2024). Finally, recent approaches tackle the
problem of co-generation which seeks to define a joint inference procedure over both structure and
sequences (Campbell et al., 2024; Ren et al., 2024; Lisanza et al., 2023), but remains distinct from
the setting of this work which attempts to combine different pre-trained models using superposition.
F
BROADER IMPACTS
In this paper, we present theoretical results and demonstrate use cases in generation tasks such as
image generation and unconditional protein generation. Because of the theoretical nature of our
contributions, this work carries little societal impact. SUPERDIFF can be used to generated protein
structure from a composition of existing protein diffusion models. Better protein generation methods
can potentially lead to negative use in generating bio-hazardous molecules and proteins. We do not
perceive this as a great risk at the current stage of these models.
G
ADDITIONAL RESULTS FOR CIFAR-10
H
PROTEIN GENERATION
H.1
EXPERIMENTAL DETAILS
In our setting, we consider two state-of-the-art diffusion models for protein generation: Proteus (Wang
et al., 2024a) and FrameDiff (Yim et al., 2023b), which were trained on protein structures from
29

Published as a conference paper at ICLR 2025
Table A1: Image generation performance for CIFAR-10 with conditional models trained on two random
partitions of the training data (labeled A and B). We compare SUPERDIFF (OR) with the respective models
(modelA and modelB) with the model that is trained on the full dataset (modelA∪B) and random choice between
two models (modelA OR B).
ODE inference
SDE inference
FID (↓)
IS (↑)
FLD (↓)
FID (↓)
IS (↑)
FLD (↓)
modelA
4.75
8.98
6.95 ± 0.12
4.66
9.35
6.39 ± 0.13
modelB
4.78
8.97
6.86 ± 0.15
4.36
9.45
6.20 ± 0.14
modelA∪B
5.30
9.04
6.82 ± 0.09
2.83
9.44
6.26 ± 0.11
modelA OR B
4.75
8.94
6.86 ± 0.15
4.41
9.40
6.3 ± 0.18
SUPERDIFF (OR)
4.74
9.00
6.98 ± 0.10
4.46
9.40
6.22 ± 0.15
SUPERDIFF T =100 (OR)
4.63
8.98
6.81 ± 0.19
4.23
9.42
6.27 ± 0.12
Protein Data Bank (PDB, (Berman et al., 2000)) to estimate the special Euclidean group (SE(3))
equivariant score over multiple diffusion steps. The models’ outputs predict the coordinates of a
monomeric protein backbone.
We use pre-trained checkpoints from Proteus3 and FrameDiff4. During protein generation at inference,
we separately combine scores for translations and rotations from both models using Algorithm 1.
We investigate the use of different temperature (T) settings to scale κ for controlling the densities.
We also found that adding a small bias (ω) towards Proteus log densities improved designability.
H.2
METRICS FOR EVALUATING GENERATED PROTEINS
Designability. We assess designability using the self-consistency evaluation from Trippe et al. (2023),
where given a generated backbone, we predict its scaffold using a sequence prediction model (we use
ProteinMPNN (Dauparas et al., 2022)) and re-fold the sequence using a structure prediction model
(we use ESMFold (Lin et al., 2022)). We then compare the re-folded protein to the original generated
backbone by computing their template modeling score (scTM) and root-mean-square-distance
(scRMSD). A protein is considered to be designable if its scRMSD is < 2Å to the refolded structure.
For each protein, we repeat this process 8 times and keep the sequence with the lowest scRMSD.
We report the fraction of designable proteins for each method in Table 3 as well as the scRMSD
mean of the resulting designable proteins.
Novelty. A significant impetus for generative modelling in biology and chemistry is to propose
compounds that have not been previously identified (i.e., different from the training data), but are
also possible to make. We compute three novelty metrics: the fraction of designable proteins with
a scTM score < 0.5 (higher is better), the fraction of designable proteins with a scTM score < 0.3,
and the mean scTM score between generated proteins and the proteins from PDB that the original
diffusion models were trained on, which represents the collective human knowledge of protein
structures; a lower score is indicates higher distance from the training set and is desirable since it
shows generalization ability.
Diversity. To measure how diverse the generated proteins are, we compute their mean pairwise
scTM score (lower is better), and also report the fraction of unique clusters formed after clustering
them with MaxCluster (Herbert & Sternberg, 2008) (higher is better). Finally, we report the fraction
of proteins that contain β-sheet secondary structures, as it has been found that these structures are
typically more rare to generate (Bose et al., 2024).
H.3
PROTEIN DIVERSITY EXPLORATION
To embed and cluster the generated protein backbones we use the Foldseek (Van Kempen et al., 2024)
package to compute pairwise aligned TM-scores. We then use the UMAP (McInnes et al., 2018)
package to compute a 2D embedding. Where proteins are represented as points that are close to each
other if they are structurally similar (by aligned TM-Score).
We then clustered the proteins again with the Foldseek tool to find representative structures. Finally
we used KMeans to explore the space and narrow down which protein structures belong where on the
protein structure manifold.
3https://github.com/Wangchentong/Proteus
4https://github.com/jasonkyuyim/se3_diffusion
30

Published as a conference paper at ICLR 2025
(a)
(b)
Figure A1: UMAP visualizations of protein structures with (a) SUPERDIFF (AND) and (b) averaging of outputs.
Figure A2: Proteins generated by SUPERDIFF (AND) with scTM score < 0.3.
In Fig. A1, we show UMAP visualizations of proteins generated with SUPERDIFF (AND) and
averaging of outputs (Liu et al., 2022a).
H.4
PROTEIN NOVELTY EXPLORATION
In Fig. A2, we visualize the proteins generated by SUPERDIFF (AND) that are furthest away from the
set of known proteins (scTM score < 0.3).
I
SMALL MOLECULE GENERATION
I.1
VISUALIZING TOP-PERFORMING MOLECULES
In Fig. A3, we display the best molecules generated with the objective of GSK3β inhibition and QED.
We evaluate the performance of molecules according to both objectives by taking the product of the
individual objectives. In Fig. A4, we display the best molecules generated with the objective of high
GSK3β inhibition and JNK3 inhibition.
J
GENERATING IMAGE COMPOSITIONS WITH STABLE DIFFUSION
For our concept interpolation experiments, we use publicly-available pre-trained weights, mod-
els, and schedulers from Stable Diffusion v1-4 https://huggingface.co/CompVis/
stable-diffusion-v1-4.
In Figs. A5–A24, we show examples of image compositions generated by SUPERDIFF (AND),
averaging of outputs, and joint prompting. Prompts are shown in each image caption. We show
31

Published as a conference paper at ICLR 2025
GSK3β: 0.50 
QED:   0.54 
GSK3β: 0.40 
QED:   0.78 
GSK3β: 0.53 
GSK3β: 0.51 
GSK3β: 0.41 
QED:   0.71 
QED:   0.71 
QED:   0.62 
N
N
N
N
N
N
N
NH2
H2N
H
N
N
NH2
N
OH
N
N
H3Si
N
N
N
N
N
N
N
H
N
N
O
(a) Molecules generated by SUPERDIFF (AND).
GSK3β: 0.45 
QED:   0.53 
GSK3β: 0.32 
QED:   0.78 
GSK3β: 0.40 
QED:   0.58 
GSK3β: 0.37 
GSK3β: 0.27 
QED:   0.58 
QED:   0.80 
N
N
N
N
N
Cl
HN
N
HN
H
N
S
H2N
N
H2N
N
NH
NH2
N
N
N
N
N
NH2
N
H
HO
N NH
HN
N
(b) Molecules generated by averaging scores.
Figure A3: Best molecules generated by (a) SUPERDIFF (AND) and (b) averaging scores (Liu et al., 2022a)
where target properties are high GSK3β inhibition and high QED.
GSK3β: 0.69 
GSK3β: 0.49 
JNK3:  0.52 
JNK3:  0.44 
H2N
N
N
H
NH2
O
N
H
N
N
S
N
N
OH
O
O
HO
O
F
HO
H
N
F
F
F
N
N
OH
HO
N
NH
HN
F
F
F
OH
(a) Molecules generated by SUPERDIFF (AND).
H2N S
O
O
H
N
N
H
N
Cl
HN
N
N
GSK3β: 0.62 
GSK3β: 0.36 
JNK3:  0.34 
JNK3:  0.54 
O
H
N
N
N
H
N
N
HO
N
N
(b) Molecules generated by averaging scores.
Figure A4: Best molecules generated by (a) SUPERDIFF (AND) and (b) averaging scores (Liu et al., 2022a)
where where target properties are high GSK3β inhibition and high JNK3 inhibition.
images generated by the first 6 seeds (uniform sampling), as well as our favourite images generated
from 20 seeds. For joint prompting, we generate prompts from two concepts using the linking
term "that looks like". For example, given the concepts "a lemon" and "a sunflower",
the resulting prompt would be "a sunflower that looks like a lemon". We also generate
images with the reversed prompt ("a lemon that looks like a sunflower"), and keep the
images generated by the prompt resulting in the higher mean score for each metric. The order of the
concepts in each image caption reflects the ordering that obtained the higher TIFA score.
J.1
STABLE DIFFUSION WITH SUPERDIFF (OR)
Baselines for concept selection (OR). For the OR setting, our baseline is prompting SD that prompts
the model to select between two concepts: "a sunflower or a lemon". As with AND, we also
flip the prompt and keep the better image for each metric. As an upper bound, we generate images
from SD by prompting it with a random choice between the two concepts (PromptA OR B).
32

Published as a conference paper at ICLR 2025
Table A2: Quantativate evaluation of SD-generated images for logical OR. We compare SUPERDIFF (OR), joint
prompting, and an upper bound of randomly selecting a single concept prompt (PromptA OR B). We report the
maximum scores from each prompt pair, as well as the absolute difference between the maximum and minimum
scores (|∆|). These metrics reflect how well a method can select one concept to generate.
|∆|/Max. CLIP (↑)
|∆|/Max. ImageReward (↑)
|∆|/Max. TIFA (↑)
PromptA OR B (uncorrelated random choice)
9.13/29.72
2.87/0.70
88.21/97.58
Joint prompting
7.20/29.80
2.47/0.59
79.46/97.92
SUPERDIFF (OR)
8.58/29.87
2.76/0.64
84.10/95.83
SUPERDIFF qualitatively generates images with better concept interpolations and selections. In
Fig. A25, we show examples of image compositions generated by SUPERDIFF (OR), joint prompting,
and an upper bound of randomly selecting a prompt of one concept (coin flip). As with SUPERDIFF
(AND), we keep the prompt order that resulted in higher scores for the baseline. We find that
SUPERDIFF (OR) can faithfully generate images with a single concept. The joint prompting baseline
can sometimes generate images that combine fragments of both concepts, but other times it also
generates images of a single concept, typically the first concept in the joint prompt (this also
underscores why this method struggles with concept interpolation).
We evaluate SUPERDIFF (OR) using the same three metrics as SUPERDIFF (AND) (CLIP Score,
ImageReward, and TIFA) and display the results in Table A2. We again evaluate the image against
each concept prompt separately and take both the maximum score and the absolute difference between
both scores for each metric. This is so that we can measure how well one concept is represented. The
upper bound for this setting is randomly prompting SD with either of the prompts; we find that we
are almost able to match this setting across all scores, indicating that our method is able to faithfully
select a single concept. SD with joint prompting does not perform as well, as nothing prevents it from
combining components from both concepts.
33

Published as a conference paper at ICLR 2025
Figure A5: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting for
the concepts "an airplane" and "an eagle".
34

Published as a conference paper at ICLR 2025
Figure A6: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting for
the concepts "bottle cap" and "chess pawn".
35

Published as a conference paper at ICLR 2025
Figure A7: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting for
the concepts "a bicycle wheel" and "a spider web".
36

Published as a conference paper at ICLR 2025
Figure A8: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting for
the concepts "a chair" and "an avocado".
37

Published as a conference paper at ICLR 2025
Figure A9: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting for
the concepts "a cinnamon roll" and "a snail".
38

Published as a conference paper at ICLR 2025
Figure A10: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "cookie" and "moon".
39

Published as a conference paper at ICLR 2025
Figure A11: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a dog" and "a cat".
40

Published as a conference paper at ICLR 2025
Figure A12: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a donut" and "a map".
41

Published as a conference paper at ICLR 2025
Figure A13: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "duck" and "otter".
42

Published as a conference paper at ICLR 2025
Figure A14: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "fireworks" and "dandelion".
43

Published as a conference paper at ICLR 2025
Figure A15: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a flamingo" and "a candy cane".
44

Published as a conference paper at ICLR 2025
Figure A16: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a helicopter" and "a dragonfly".
45

Published as a conference paper at ICLR 2025
Figure A17: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "pebbles on a beach" and "a turtle".
46

Published as a conference paper at ICLR 2025
Figure A18: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a pineapple" and "a beehive".
47

Published as a conference paper at ICLR 2025
Figure A19: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a rocket" and "a cactus".
48

Published as a conference paper at ICLR 2025
Figure A20: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a silhouette of a dog" and "a mountain landscape".
49

Published as a conference paper at ICLR 2025
Figure A21: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a sunflower" and "a lemon".
50

Published as a conference paper at ICLR 2025
Figure A22: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "teddy bear" and "panda".
51

Published as a conference paper at ICLR 2025
Figure A23: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "a waffle cone" and "a volcano".
52

Published as a conference paper at ICLR 2025
Figure A24: Image compositions generated using SUPERDIFF (AND), averaging of outputs, or joint prompting
for the concepts "zebra" and "barcode".
53

Published as a conference paper at ICLR 2025
Figure A25: Concept selections using SUPERDIFF (OR), joint prompting, and randomly selecting a prompt of one
concept. The top subfigure shows generated images for the concepts "a candy cane" or "a flamingo".
The bottom subfigure shows generated images for the concepts "a pineapple" or "a beehive".
54
