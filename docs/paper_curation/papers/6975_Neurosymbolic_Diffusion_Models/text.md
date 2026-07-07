# arXiv:2505.13138v2[cs.LG]30Oct2025

## Neurosymbolic Diffusion Models

Emile van Krieken1 Pasquale Minervini1,2,∗ Edoardo Ponti1,∗ Antonio Vergari1,∗ 1School of Informatics, University of Edinburgh 2Miniml.AI e.van.krieken@vu.nl, {p.minervini, eponti, avergari}@ed.ac.uk

##### Abstract

Neurosymbolic (NeSy) predictors combine neural perception with symbolic reasoning to solve tasks like visual reasoning. However, standard NeSy predictors assume conditional independence between the symbols they extract, thus limiting their ability to model interactions and uncertainty — often leading to overconfident predictions and poor out-of-distribution generalisation. To overcome the limitations of the independence assumption, we introduce neurosymbolic diffusion models (NESYDMS), a new class of NeSy predictors that use discrete diffusion to model dependencies between symbols. Our approach reuses the independence assumption from NeSy predictors at each step of the diffusion process, enabling scalable learning while capturing symbol dependencies and uncertainty quantification. Across both synthetic and real-world benchmarks — including high-dimensional visual path planning and rule-based autonomous driving — NESYDMS achieve state-ofthe-art accuracy among NeSy predictors and demonstrate strong calibration.

##### 1 Introduction

Neurosymbolic (NeSy) methods aim to develop reliable and interpretable AI systems by augmenting neural networks with symbolic reasoning [25, 26, 77]. In particular, probabilistic neurosymbolic predictors [52, 54, 57, 80] learn neural networks that extract high-level symbols, also called concepts, from raw inputs. These concepts are latent variables used in interpretable symbolic programs to reason and predict output labels. However, recent work highlights that the reliability of NeSy predictors is not guaranteed, especially under certain common architectural choices.

More specifically, in many real-world settings, NeSy predictors fail silently: they can learn the wrong concepts while achieving high accuracy on output labels [22, 27]. This issue arises when the data and program together admit multiple concept assignments that are indistinguishable [54, 56]. How do we design NeSy predictors that handle this ambiguity? Marconato et al. [55] argued that NeSy predictors should express uncertainty over the concepts that are consistent with the data. Then, uncertainty can guide user intervention, inform trust, or trigger data acquisition when the model is uncertain [55].

However, most existing NeSy predictors cannot properly model this uncertainty, as they rely on neural networks that assume (conditional) independence between concepts [10, 80, 86]. While this assumption enables efficient probabilistic reasoning [6, 73, 80, 86], it also prevents these NeSy predictors from being aware of concept ambiguity and thus reliably generalising out-of-distribution [39, 79]. Therefore, designing expressive, scalable and reliable NeSy predictors is an open problem.

To fill this gap, we design neurosymbolic diffusion models (NESYDMS). NESYDMS are the first class of diffusion models that operate over the concepts of a NeSy predictor in conjunction with symbolic programs. In theory, discrete diffusion models [9, 68] are particularly suited for NeSy predictors, as each step of their denoising process involves predicting a discrete distribution that fully factorises. We use this local independence assumption to profit from the insights and machinery of classical

*: Shared supervision. Code is available at https://github.com/HEmile/neurosymbolic-diffusion.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

[Figure 1]

Input x

Predicted concepts c˜0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

p✓(c˜0 | x,ct)

Unmasking model

Program (Dijkstra)

'(c˜0) q✓(c0 | x,y0)

###### q(ct | c0)

Variational posterior

Mask

Masked concepts ct

Concepts c0

| |
|---|

| |
|---|

Gradient estimation r✓Ly(✓)

True path y0

Costs: grass sand rock masked

Predicted path y˜0

Figure 1: NESYDMS integrate masked diffusion models (orange boxes) with symbolic programs (blue box) to learn to predict the minimum cost path in a visual path-planning task. A variational posterior (Section 3.3) first obtains a candidate concept c0, that represents the costs of traversing each cell of the grid. Then, we partially mask c0 using the masking process q(cs | c0) to obtain masked

concepts c12 . We feed this to the discrete diffusion model’s unmasking model pθ(c˜ | x,c12 ) to predict the unmasked concepts c˜0. We use the symbolic program φ, which we choose as Dijkstra’s algorithm, to map the predicted concepts c˜0 to the predicted path y˜0. Finally, we use gradient estimation to update the parameters of the unmasking model. Dotted arrows denote samples from a distribution.

NeSy predictors, while modelling concepts as dependent entities globally. In practice, designing a diffusion process for NeSy predictors is highly non-trivial, as it requires dealing with a symbolic program and marginalising over all possible concepts, a task that is intractable in general. We show how to solve both aspects effectively by devising a novel continuous-time loss function for diffusion that incorporates symbolic programs, for which training scales gracefully.

Contributions. After discussing the background on NeSy predictors and (masked) diffusion models in Section 2, we (c1) introduce NESYDMS in Section 3, a class of scalable NeSy predictors that model concept dependencies by formalising a masked diffusion process [68]. Then in Section 3.2, we (c2) derive a principled loss function for NESYDMS and present an efficient gradient estimator for training it. To derive this loss, we prove that the continuous-time losses of masked diffusion models extend to non-factorised distributions. Finally, in Section 4, we (c3) empirically show that NESYDMS are (i) both calibrated and performant on tasks from the RSBench suite of visual reasoning problems [11] while (ii) scaling beyond the state-of-the-art on the complex visual path-planning task [64].

##### 2 Background

###### 2.1 Neurosymbolic predictors

We aim to learn a parametrised predictive model pθ(y | x) that maps high-dimensional inputs x to Y -dimensional discrete labels y ∈ [Vy]Y , where each label can take a value in [Vy] = {1,2,...Vy}. A typical (probabilistic) NeSy predictor implements pθ(y | x) by first (i) using a concept extractor, i.e., a neural network pθ(c | x) that maps the input x to a C-dimensional vector of symbolic concepts c ∈ [Vc]C, i.e., discrete variables encoding high-level information that can take V values.1 Then, (ii) the NeSy predictor maps concepts c through a program φ : [Vc]C → [Vy]Y to obtain output predictions yˆ. As usual in NeSy [10, 43, 52, 80], we only assume access to training data for inputoutput pairs (x,y) but no labelled data for concepts c, i.e., concepts c are latent variables. Formally, we define the predictor pθ(y | x) by marginalising over all concepts c ∈ VcC that are consistent with

1For simplicity of presentation, we assume that the number of possible values V is the same for both concepts and labels, but this is not necessary for the paper.

the output y, summing their probability masses:

pθ(c | x)[φ(c) = y]. (1)

pθ(y | x) :=

c

The equation above is also known as computing a conditional weighted model count (WMC), and it is central to several probabilistic neurosymbolic methods [6, 43, 52, 80, 86].

- Example 2.1 ([65]). Consider the visual path-planning task in Fig. 1 where the task is to predict a minimum cost path y from the top-left corner to the bottom-right corner of the visual map x. y is encoded as a binary matrix, where cells traversed form a path. A neural network extracts concepts that represent discrete costs c for each cell on the grid, then a search algorithm φ(c), like Dijkstra, is used to find the shortest path y according to costs c.

Reasoning shortcuts. Recent work proved NeSy predictors are susceptible to reasoning shortcuts [RSs; 56], which is when a model pθ(y | x) learns to predict the output labels y correctly given the input x, but incorrectly maps inputs to concepts c. Since we cannot catch RSs on the training data, it can dramatically harm model performance on unseen data [53]. Mitigating RSs is challenging and potentially costly [54, 56]. However, models can be made aware of their RS by properly expressing uncertainty over all concepts that are consistent with the input-output mapping, improving reliability and generalisation [39, 54, 55]. Then we can, for example, deploy NeSy predictors in an active learning setting where uncertain concepts are queried for extra labelling.

- Example 2.2. Consider an input x containing two MNIST digits that are either 0 or 1. The unseen concepts c are the digits, and φ(c) returns 1 if the two digits are different, otherwise 0. A neural

concept extractor pθ(c | x) that maps MNIST digits of 0 to 1s and MNIST digits of 1s to 0s will perfectly fit the input-output mapping.

The configuration in Example 2.2 maximises Eq. 1 without learning the ground-truth concepts. Given only input-output pairs, it is not possible to distinguish this RS from the correct input-concept mapping. Instead, given ground-truth concepts c∗ = (0,1), an RS-aware model would assign some belief to both options (0,1) and (1,0).

Independence assumption and its limitations. Unfortunately, in practice, the vast majority of NeSy predictors make an architectural assumption that prevents RS awareness: the conditional independence of concepts c given inputs x [43, 80, 86]. Formally, this assumption implies that

- pθ(c | x) in Eq. 1 factorises as Ci=1 pθ(ci | x). NeSy predictors use this assumption to perform efficient probabilistic reasoning via WMC solvers and knowledge compilation techniques [15, 18, 63], or by developing efficient approximation algorithms [73, 80].

Recent work proved that such models cannot simultaneously represent the relevant uncertainty over different concepts while maximising Eq. 1 [39]. To see why, consider Example 2.2, with true concepts c∗ = (0,1). The only maximisers of Eq. 1 for the independent model are to either deterministically return (0,1) or (1,0) [39, 79]. However, there is no maximiser that can simultaneously assign probability mass to both cases, meaning independent models cannot be RS-aware. To overcome this limitation, we should design a NeSy predictor that can express dependencies between concepts, which we address next.

###### 2.2 Which expressive model class for NeSy?

Previous work on NeSy predictors without the independence assumption explored mixture models and their generalisation as probabilistic circuits [6, 16]. An example is BEARS [55], which is specifically designed for RS-awareness. A related approach is to add extra variables and constraints to the WMC. This can, for instance, be done using a probabilistic programming language [43, 51]. However, these methods require (i) compiling the program into a logic circuit via knowledge compilation and (ii) ensuring the probabilistic circuit is compatible with this logic circuit [83]. The first step can require exponential time in the worst case, and as such scaling to high-dimensional spaces can be challenging [4, 80]. Furthermore, these methods require the neural concept extractor to predict many more additional parameters for the different mixture components.

Alternatively, autoregressive models are a common type of expressive model, but using these in NeSy predictors based on Eq. 1 is computationally hard, as the marginalisation over concepts does not commute with autoregressive conditioning [3, 5]. While this limitation also holds for diffusion

models, they do use a conditional independence assumption locally at every denoising step. This local assumption is sufficient to encode global dependencies. Furthermore, the locality allows us to design neural models that predict only C parameters, just like NeSy predictors with the independence assumption. Thus, we use masked diffusion models [68] that achieve expressiveness by iteratively unmasking a discrete sample. We discuss in Section 3 how to extend their local independence assumption to realise NeSy predictors.

Masked diffusion models. Diffusion models encode an expressive joint distribution over concepts c by defining a forward process that a neural network modelling a reverse process will learn to invert. As our concepts are symbolic, we need a diffusion process for discrete data [9, 90]. We choose masked diffusion models (MDMs) [68, 72], a type of discrete diffusion model with promising results on language modelling [61, 89] and reasoning [88]. MDMs allow us to derive a principled loss using the program φ (Section 3.2) and to develop scalable approximations (Section 3.4). We first review MDMs in their vanilla form, i.e., to model an unconditional distribution over concepts, pθ(c).

MDMs consider a continuous time diffusion process [9, 14], where the forward process gradually masks dimensions of a data point c0 into a partially masked data point ct ∈ [Vc + 1]C at time steps t ∈ [0,1]. We extend the vocabulary size to include a placeholder m = Vc +1 for masked dimensions. The data point becomes fully masked as c1 = m = [m,...,m]⊤ at time step 1. More formally, for 0 ≤ s < t ≤ 1, the forward process q masks a partially masked concept cs into ct with

C

αt αs

αt αs

q(ct | cs) =

[c ti = m], (2)

[c ti = csi] + 1 −

i=1

where α : [0,1] → [0,1] is a strictly decreasing noising schedule with α0 = 1 and α1 = 0. q(ct | cs) masks each dimension with probability 1 − α

αs , leaving it unchanged otherwise. Importantly, once masked, a dimension remains masked. MDMs learn to invert the forward process q(ct | cs) using a trained reverse process pθ(cs | ct). The reverse process starts at a fully masked input c1 = m at time step 1, and gradually unmasks dimensions by assigning values in {1,...,Vc}.

t

The reverse process pθ(cs | ct) is usually parameterised with conditionally independent unmasking models pθ(c˜0 | ct) = Ci=1 pθ(˜c0i | ct) that predict completely unmasked data c˜0 given (partially) masked versions ct. Then, MDMs remask some dimensions using the so-called reverse posterior

- q(cs | ct,c0 = c˜0) (see more details in Eq. 10 in Section A):

pθ(cs | ct) :=

pθ(c˜0 | ct) q(cs | ct,c0 = c˜0), (3)

c ˜0

The standard loss function masks c0 partially to obtain ct, and then uses the conditionally independent unmasking model pθ(c˜0 | ct) to attempt to reconstruct c0. This loss function requires that pθ(c˜0 | ct) implements the carry-over unmasking assumption, meaning it should assign a probability of 1 to the values of previously unmasked dimensions. We provide additional background on MDMs in Section A. Next, we discuss how to design novel MDMs tailored for NeSy prediction.

##### 3 Neurosymbolic Diffusion Models

To overcome the limitations of the independence assumption haunting NeSy predictors, our neurosymbolic diffusion models (NESYDMS) use MDMs to learn an expressive distribution over concepts and labels while retaining this assumption locally, enabling scaling. To develop NESYDMS, we extend MDMs by (i) conditioning on the input x, (ii) acting on both concepts c and outputs y, treating concepts as latent variables and (iii) providing differentiable feedback through the program φ. We first define this model in Section 3.1 and then derive a principled loss in Section 3.2. We discuss how to optimise this loss in Sections 3.3 and 3.4, and finish by discussing inference in Section 3.5. Finally, Fig. 1 provides an overview of the loss computation of NESYDMS.

###### 3.1 Model setup

We define NESYDMS using a conditionally independent unmasking model pθ(c˜0 | ct,x) and a program φ that maps concepts to outputs. We use forward processes for both the concepts q(ct | cs)

and the outputs q(yt | ys), each defined as in Eq. 2. The concept reverse process pθ(cs | ct,x) is

parameterised as in Eq. 3 with a conditional concept unmasking model pθ(c˜0 | cs,x), and the output reverse process pθ(ys | cs,yt,x) is parameterised by reusing the concept unmasking model:

pθ(ys | cs,yt,x) :=

pθ(c˜0 | cs,x)q(ys | yt,y˜0 = φyt(c˜0)). (4)

c ˜0

pθ(ys | cs,yt,x) takes the concept unmasking model and marginalises over all concepts c˜0 that are consistent with the partially masked output ys. To implement the carry-over unmasking assumption,

we use φyt to refer to a variation of the program φ that always returns yit if dimension i is unmasked in yt. We refer to Section D.1 for details. The neural network for the concept unmasking model

- pθ(c˜0 | ct,x) can be readily adapted from NeSy predictors as defined in Eq. 1 by additionally conditioning the neural network pθ(c | x) on the currently unmasked concepts ct.

Since we do not have direct access to ground-truth concepts c0, we will use a variational setup and derive a lower-bound for the intractable data log-likelihood pθ(y0 | x) (fully defined in Eq. 45). In particular, we use a variational distribution qθ(c0 | y0,x) that shares parameters θ with the MDM to approximate the posterior pθ(c0 | y0,x). To implement this, we repurpose our concept unmasking model pθ(cs | ct,x) with the controlled generation method from [29], which we describe in Section 3.3. We provide more details and a full derivation of the log-likelihood in Section D.1.

- 3.2 Loss function

We next derive a NELBO for NESYDMS. Intuitively, we define the NESYDM reverse process over T discrete steps, and then consider the data log-likelihood as T goes to infinity, giving a NELBO for a continuous-time process. This NELBO will be the base for the loss function used to train NESYDMS.

Theorem 3.1. Let pθ(c˜0 | ct,x) be a concept unmasking model, φ : [Vc]C → [Vy]Y a given program, qθ(c0 | y0,x) a variational distribution, and αt a noising schedule. Then, we have that the data log-likelihood as T → ∞ is bounded as limT→∞ −log pθNESYDM(y0 | x) ≤ LNESYDM, where

LNESYDM =Et∼[0,1],q

θ(c0|x,y0),q(ct|c0)

αt′ 1 − αt

C

i=1

log pθ(˜c0i = c0i | ct,x)

Lc: concept unmasking loss

+ αt′

Y

i=1

log

c ˜0

pθ(c˜0 | ct,x)[φ( c˜0)i = yi0]

Ly: output unmasking loss

− H[qθ(c0 | y0,x)]

LH[q]: variational entropy

(5)

We provide a derivation of this NELBO in Section D.2. This NELBO has three components:

• The concept unmasking loss Lc is like the unmasking loss used in MDMs (Eq. 14). Since we do not have access to the ground-truth concept c0, we sample c0 from the variational distribution

- qθ(c0 | y0,x) and ask the model to reconstruct c0 from a partially masked version ct ∼ q(ct | c0).

- • The output unmasking loss Ly is a sum of Y weighted model counts (WMC) like in Eq. 1, one for each dimension i of the output y0. Unlike Eq. 1, Ly weights concepts using the concept unmasking

model pθ(c˜0 | ct,x) that is conditioned on partially masked concepts ct. Importantly, we use conditionally independent concept unmasking models, meaning we can use standard techniques in the NeSy literature to compute this loss efficiently. Section B provides additional analysis.

- • The variational entropy LH[q] is maximised to encourage the variational distribution to cover all concepts c0 that are consistent with the input x and output y0.

To derive the NELBO, we had to prove a new theorem that extends the standard MDM NELBO to non-factorised unmasking models pθ(c˜0 | ct) (Section C), which can be an interesting result for future MDM architectures even outside NeSy predictors. We need this result because, unlike the concept reverse process, the output reverse process pθ(ys | cs,yt,x) in Eq. 47 does not factorise, and we cannot naively apply the standard MDM NELBO given in Eq. 14.

###### 3.3 Variational posterior

To compute the NESYDM NELBO, we require a variational distribution qθ(c0 | y0,x) to sample likely concepts c0 that are consistent with the ground-truth output y0. We achieve this by adapting

the sampling algorithm described in Section 3.5 using a concept unmasking model pθ(c˜0 | ct,x) that depends on the output y0 and the program φ:

pθ(c˜0 | ct,x)[φ( c˜0) = y0] Z(ct,x,y0)

qθ(c˜0 | ct,y0,x) :=

, (6)

where Z(ct,x,y0) is a normalising constant. This redefines the standard unmasking process from Eq. 3 by only considering valid c˜0. Unfortunately, sampling from pθ(c˜0 | ct,x,y0) is NP-hard [33, 49]. However, if we have a tractable representation of the program φ, e.g., a polysize circuit as the output of a knowledge compilation step [63], then we can represent qθ(c˜0 | ct,y0,x) compactly and exactly sample from it [6]. Without access to such a circuit, we can instead use a relaxation of the constraint similar to [29]. Let rβ(c˜0 | y0) = exp(−β Yi=1 [φ( c˜0)i ̸= yi0]), where β > 0 and β → ∞ approaches the hard constraint. At each step in the reverse process, we resample to approximately obtain samples from qθβ(c˜0 | ct,x,y0) ∝ pθ(c˜0 | ct,x)rβ(c˜0 | y0) [29]. This procedure may sample concepts c˜0 that are inconsistent with y0, but prefers samples that reconstruct more dimensions of y0. We find that reasonably large β > 10 works in our experiments. In practice, this effectively samples K times from pθ(c˜0 | ct,x) and chooses the sample that violates the fewest constraints. See Section F.1 for details.

###### 3.4 Loss optimisation and scalability

Next, we describe how we optimise the NESYDM NELBO LNESYDM using gradient descent. We design a gradient estimation algorithm that scales to large reasoning problems by approximating

intractable computation. Note that, given samples c0,ct ∼ qθ(c0 | x,y0) q(ct | c0), the empirical concept unmasking loss Lc is tractable, so we only discuss how to backpropagate through the output unmasking loss Ly and the variational entropy LH[q].

Computing the output unmasking loss Ly involves computing multiple WMCs, which are #P-hard. One option is to compute each WMC exactly using circuits obtained via knowledge compilation [37, 52, 86]. However, to ensure scalability, we develop a sampling-based approach that approximates the WMC gradients [73]. In particular, we use a REINFORCE-based gradient estimator [59], the REINFORCE Leave-One-Out (RLOO) estimator [1, 38]. RLOO is similar to the popular GRPO algorithm [71] while being unbiased. Furthermore, RLOO allows for flexible tradeoffs between variance and computation constraints by choosing the number of samples.

However, methods like RLOO can fail for problems where the probability of getting a sample c˜0 consistent with y0 is very low: when we only sample inconsistent concepts c˜0, RLOO does not provide any gradient signal. However, the output unmasking loss is subtly different, as Ly gives a signal for each of the dimensions of y0 independently. This helps structure the search for consistent concepts c˜0 by decomposing the problem into Y independent subproblems [8, 80]. More precisely, given a time step t ∈ [0,1], samples c0,ct ∼ qθ(c0,ct | y0,x) and samples c˜01,...,c˜0S ∼ pθ(c˜0 | ct,x), we use:

Y

S

1 µi(S − 1)

∇θLy ≈ αt′

[φ( c˜0j)i = yi0] − µi ∇θ log pθ(c˜0j | ct,x) (7)

i=1

j=1

where µi = S1 Sj=1 [φ( c˜0j)i = yi0]. We provide further details in Section E.

Maximising the variational entropy LH[q] is challenging: the variational distribution in Section 3.3 samples from a conditioned version of the unmasking model where computing likelihoods, and by

extension, maximising the entropy of qθ, is highly untractable. We therefore experimented with two biased approximations of this loss which sufficed for our experiments, and leave more sophisticated approximations for future work:

- • conditional 1-step entropy: If we have access to a tractable constraint circuit of φ, we can use it to compute the entropy of an independent distribution over c0 conditioned on y0 and x [7, 83]. Then, we maximise the entropy over the variational distribution when performing time discretisation with a single step (T = 1): H[qθ(c˜0 | c1 = m,y0,x)] using the distribution defined in Eq. 6.

- • unconditional 1-step entropy: Without access to a tractable constraint circuit, we instead maximise the unconditional 1-step entropy H[qθ(c˜0 | c1 = m,x)].

Furthermore, as is common in variational setups [30], we add hyperparameters that weight the contribution of each loss component Lc, Ly, and LH[q]. We found these hyperparameters critical to the performance of the model (see Section H.2 for an ablation study). Finally, unbiased optimisation of Lc and Ly also requires calculating the gradient through sampling a c0 from the variational distribution [59, 70]. Like with the variational entropy, we found that sidestepping this part of the gradient, which would be intractable and have high variance otherwise, simplifies optimisation and yields good performance in practice. See pseudocode for the learning algorithm in Algorithm 1 and additional discussion and definitions of the gradient estimation algorithm in Section E.

- Algorithm 1 Algorithm for estimating the gradients of the NELBO for training NESYDM

- 1: Given datapoints (x,y0) and unmasking model pθ(c˜0 | x,ct) with current parameters θ
- 2: c0 ∼ qθ(c0 | x,y0) ▷ Sample from variational distribution (Section 3.3).
- 3: t ∼ U(0,1) ▷ Sample a random time step.
- 4: ct ∼ q(ct | c0) ▷ Mask the concept c0 to ct (Eq. 9).
- 5: c˜01,...,c˜0S ∼ qθ(c˜0 | x,ct) ▷ Sample S samples from unmasking model.
- 6: gy ← gy0(c˜01,...c˜0S) ▷ Estimate gradient of Ly using Eq. 60.
- 7: gc ← α

′ t

1−αt i∈Mct ∇θ log pθ( ˜wi0 = c0i | x,ct) ▷ Compute gradient of Lc

- 8: gH ← ∇θLH ▷ Compute gradient of LH.
- 9: return γ

c

C gc + γYy gy + γ

H

C gH ▷ Return the weighted sum of the gradients.

3.5 Sampling and Inference

Next, we describe how we sample from trained NESYDMS to make predictions of y given x. Exactly computing the mode argmaxy0 pθNESYDM(y0 | x) is intractable even for representations supporting tractable marginals [2, 84], therefore we need to approximate it. We use a majority voting strategy, where we sample L concepts c0l from the trained MDM, compute the output with the program φ, and take the most frequent output:

yˆ = argmaxy

L

l=1

[φ(c 0l ) = y], c01,...,c0L ∼ pθ(c0 | x,c1 = m). (8)

If the concept dimension C is not too large, we use the first-hitting sampler from [94] to sample from pθ(c0 | x,c1 = m) exactly in C steps. Otherwise, we use a T-step time-discretisation of the reverse process [68], for pseudocode see Algorithm 2. For implementation details, we refer to Section F. Additionally, we experimented with different majority voting strategies, which we discuss in Section H.1. These mainly study whether to do majority voting before or after running the program.

- Algorithm 2 Standard time-discretised output prediction for NESYDM

- 1: Given datapoint x and unmasking model pθ(c˜0 | x,ct) with parameters θ
- 2: for l ← 1 to L do
- 3: c1 = m
- 4: for k ← T to 1 do
- 5: c˜0 ∼ pθ(c˜0 | x,ct) ▷ Sample from unmasking model (Section 3.3).
- 6: cs ∼ q(cs | ct,c0 = c˜0) ▷ Sample from remasking process (Eq. 10).
- 7: c0l ← c0 ▷ Store the sampled concept
- 8: yl ← φ(c0l ) ▷ Compute program output for this sample
- 9: yˆ ← argmaxy Ll=1 [y l = y] ▷ Majority vote
- 10: Return yˆ ▷ Return the most frequent output

##### 4 Experiments

We aim to answer the following research questions: (RQ1:) “Can NESYDMS scale to highdimensional reasoning problems?” and (RQ2:) “Does the expressiveness of NESYDMS improve

Table 1: Accuracy of predicting the correct sum on MNIST Addition with N = 4 and N = 15 digits. Methods above the horizontal line are exact, and below are approximate. We bold the best-scoring methods in the exact and approximate categories separately.

METHOD N = 4 N = 15 DEEPSOFTLOG [48] 93.5 ± 0.6 77.1 ± 1.6 PLIA [21] 91.84± 0.73 79.00± 0.73 SCALLOP [21, 43] 90.88± 0.48 T/O EXAL [85] 91.65± 0.57 73.27± 2.05 A-NESI [80] 92.56± 0.79 76.84± 2.82 NESYDM (ours) 92.49± 0.98 77.29± 1.40

Table 2: NESYDM significantly scales beyond current NeSy predictors. Accuracy of predicting a shortest path on visual path planning with different grid sizes. Above the horizontal line are methods predicting continuous costs, while below are approximate NeSy methods that predict discrete, binned costs.

METHOD 12 × 12 30 × 30 I-MLE [62] 97.2 ± 0.5 93.7 ± 0.6

EXAL [85] 94.19± 1.74 80.85± 3.83 A-NESI [80] 94.57± 2.27 17.13± 16.32 A-NESI+RL [80] 98.96± 1.33 67.57± 36.76 NESYDM (ours) 99.41± 0.06 97.40± 1.23

reasoning shortcut awareness compared to independent models?” Since there are currently no scalable RS-aware NeSy methods, the baselines we use are separated for the two research questions. We match experimental setups of the baselines, using the same datasets and neural network architectures for a fair comparison. To approximate the variational entropy (Section 3.4), we use the unconditional entropy for the experiments, as the conditional entropy is intractable. For the RSBench experiments, we tried both. We use the linear noising schedule αt = 1 − t for all experiments.

For all experiments, we repeat runs with 10 different random seeds. In all tables, we find the best-performing methods with bold font. In particular, we bold all methods that are not statistically different from the highest-scoring method according to an unpaired one-sided Mann-Whitney U test at a significance level of 0.05. We provide additional experimental details in Section G. Code is available at https://github.com/HEmile/neurosymbolic-diffusion.

###### 4.1 RQ1: Scalability of NESYDM

To evaluate the scalability of NESYDM, we consider two NeSy benchmark tasks with high combinatorial complexity: multidigit MNIST Addition and visual path planning. We compare to current approximate NeSy methods that use the independence assumption and are not RS-aware, namely A-NeSI [81], Scallop [43], and EXAL [85].

Multidigit MNIST Addition. The input x is a sequence of 2 numbers of N digits, and the output y is the sum of the two numbers, split up into N + 1 digits. The goal is to train a neural network that recognises the individual digits c ∈ {0,1,...,9}2N in the input from input-output examples. There are no dependencies between the digits and the problem is not affected by reasoning shortcuts, so we do not expect NESYDM to improve significantly over NeSy methods that use the independence assumption. Still, we find in Table 1 that NESYDM, which uses a much more expressive model than the baselines, performs similar to the state-of-the-art approximate method A-NeSI, and is competitive with exact methods [19, 48]. Therefore, the expressivity does not come at a cost of performance and scalability in traditional NeSy benchmarks.

Visual path planning. We study the problem described in Example 2.1. Specifically, we train a neural network to predict the correct cost ci,j at each of the N ×N grid cells. Then, we use Dijkstra’s algorithm to find the shortest path y ∈ {0,1}N×N, where yi,j = 1 if the shortest path passes through cell i,j and 0 otherwise. Like other NeSy methods, we predict costs with a 5-dimensional categorical variable c ∈ {1,...,5}N×N. We also compare to I-MLE, the state-of-the-art method that predicts costs as a single continuous variable [62]. We find in Table 2 that NESYDM significantly outperforms all baselines on the challenging 30×30 problem, including I-MLE. This problem has a combinatorial space of 5900 and is considered very challenging for NeSy and neural models [65]. On the 12 × 12 problem, we cannot reject the null hypothesis that NESYDM outperforms A-NeSI + RLOO, but it does have much lower variance, highlighting the reliability of our method.

###### 4.2 RQ2: RS-awareness of NESYDM

To evaluate the RS awareness of NESYDM, we use the RSBench dataset [56] of reasoning problems that cannot be disambiguated from data alone. We consider two synthetic problems and a real-

Table 3: NESYDM is a performant and RS-aware NeSy predictor as shown on several tasks from the RSBench dataset. We report relevant performance metrics for each task, and concept calibration using ECE to evaluate RS-awareness (see Section G.4.2 for a motivation for this metric). We underline the second-best-scoring method if there is only a single statistically significant best-scoring method. The first two methods use the independence assumption. Note that SL does not support BDD-OIA.

METHOD PNP⊥⊥ SL⊥⊥ BEARS [55] NESYDM (ours)

PNP SL UNCOND H COND H

ACCy ↑ 98.24± 0.12 99.62± 0.12 99.19± 0.12 99.76± 0.00 99.12± 0.10 99.12± 0.10 ACCc ↑ 42.76± 0.14 42.88± 0.09 43.26± 0.75 42.86± 0.00 79.41± 6.58 71.16± 1.77 ACCy,OOD ↑ 5.81± 0.07 0.48± 0.21 6.31± 1.10 0.11± 0.09 10.9± 0.05 28.44± 0.90 ACCc,OOD ↑ 38.97± 0.08 38.92± 0.11 39.49± 1.07 38.88± 0.03 57.22± 0.49 62.76± 0.89 ECEc,ID ↓ 69.40± 0.35 70.61± 0.18 36.81± 0.17 37.61± 1.22 39.52± 5.01 4.18± 2.56 ECEc,OOD ↓ 86.67± 0.18 87.95± 0.14 37.89± 2.18 35.99± 2.88 35.07± 2.67 11.74± 1.18

MNISTHALF

ACCy ↑ 70.77± 0.45 97.38± 0.31 92.02± 3.14 98.67± 0.27 97.52± 0.37 98.27± 0.44 ACCc ↑ 0.40± 0.04 0.33± 0.05 0.48± 0.10 0.19± 0.08 0.36± 0.27 20.33± 1.33 ACCy,OOD ↑ 7.29± 0.49 0.05± 0.06 1.60± 2.04 0.00± 0.00 0.00± 0.00 0.02± 0.04 ACCc,OOD ↑ 7.50± 0.32 7.07± 0.09 9.36± 2.13 6.25± 1.46 4.65± 0.49 14.25± 0.76 ECEc,ID ↓ 81.04± 1.15 82.18± 1.57 28.82± 2.19 34.51± 1.65 20.93± 0.49 2.70± 1.21 ECEc,OOD ↓ 85.44± 0.72 86.96± 1.15 26.83± 1.56 32.61± 3.32 19.13± 0.50 5.77± 0.98

MNISTE-O

MF1y ↑ 63.71± 1.50 – 60.80± 0.11 – 61.67± 0.32 62.63± 0.53 MF1c ↑ 10.41± 1.90 – 19.25± 0.16 – 18.50± 0.21 13.77± 0.51 ECEc ↓ 38.89± 1.34 – 16.00± 0.20 – 18.86± 1.75 21.72± 1.83

BDD

world task. MNIST Half and MNIST Even-Odd (MNIST E-O) are variations of MNIST Addition constructed to ensure disambiguation of concepts is impossible. They have OOD test-sets to diagnose overconfident classifiers. BDD-OIA (BDD) is a self-driving task [87] where a model predicts what actions a car can take given a dashcam image. NeSy predictors extract high-level concepts from the image and use rules to predict the allowed actions. We compare to NeSy predictors using the independence assumption, namely Semantic Loss (SL⊥⊥) [86] and a standard probabilistic NeSy predictor (PNP⊥⊥). We also compare to BEARS, an RS-aware ensemble of NeSy predictors with the independence assumption [55].

In Table 3, we find that NESYDM strikes a good balance between accuracy and RS-awareness throughout the datasets. On the MNIST tasks, it attains significantly better concept accuracy than competitors, both in- and out-of-distribution. Furthermore, NESYDM, especially using the conditional entropy, has much better concept calibration than both baselines using the independence assumption and RS-aware baselines. We report additional results on these datasets in Section H.1 and find that different majority voting strategies may improve OOD performance. On BDD-OIA, we find that NESYDM has better predictive performance on outputs than BEARS while significantly improving calibration and concept performance compared to PNP⊥⊥ using the independence assumption. Furthermore, we note that, unlike the baselines, NESYDM is much more scalable as highlighted in Section 4.1.

##### 5 Further related work

NeSy predictors. The field of NeSy predictors is primarily divided into methods using fuzzy logics [10, 17, 28, 78] and those using probabilistic logics [6, 43, 52, 80, 86]. Fuzzy methods implicitly assume a form of independence between concepts, while probabilistic methods can model dependencies. Previous methods that went beyond the independence assumption mixed multiple independent distributions, like in SPL [6] and BEARS [55] which is specifically designed for RS-awareness. Neurosymbolic probabilistic logic programming frameworks like DeepProbLog and Scallop [43, 52] allow modifying the program to increase expressivity compared to the naive independence over concepts. However, these methods are built on exact or top-k inference, which is difficult to scale to high-dimensional reasoning problems like visual path planning when the number of dependencies grows. Relatedly, DeepGraphLog [34] extends DeepProbLog by using graph neural networks to model dependencies between concepts, also relying on exact inference. Conversely, all current methods focussed on approximate inference to scale neurosymbolic predictors assume independence between concepts [73, 80, 85], hence lacking RS-awareness.

NeSy generative models. A closely related topic is generating from expressive models like large language models (LLMs) and diffusion models while involving programs and constraints. For LLMs, this was studied with NeSy loss functions encoding the constraints [2, 3, 13] and with constrained decoding, for example using sequential Monte Carlo methods [42, 46, 93] and by combining the LLM with approximations using probabilistic circuits [5, 91, 92]. However, these methods adopt heuristics to steer the LLM towards a constraint, for instance, by using a pseudo-likelihood formulation [2, 3] or training an HMM surrogate that approximates the LLM [91, 92]. Instead, for NESYDM we formulate a principled NELBO, and we do so by exploiting the local structure that diffusion models offer. Furthermore, some methods tackle constrained generation from GANs [24, 75, 76], VAEs [58], deep HMMs [74], and continuous diffusion models [31, 69]. We leave extensions of NESYDM to this generative setting to future work.

##### 6 Conclusion

In this paper, we introduced NESYDMS, the first method to integrate masked diffusion models as the neural network extractor in neurosymbolic predictors. We show how to scale NESYDMS by using efficient probabilistic reasoning techniques on local unmasking distributions while minimising a global NELBO that lower-bounds the data log-likelihood. Empirically, we show that NESYDMS position themselves as one of the best NeSy predictors available that can scale to high-dimensional reasoning problems while being RS-aware. This is a crucial property for NeSy predictors deployed in real-world safety-critical applications, as they need to be well calibrated and generalise robustly.

Limitations and future work. The NESYDM NELBO can be extended to incorporate additional exact inference routines if we can obtain an efficient circuit, e.g., as the tractable representation for a symbolic program [63]. Otherwise, as argued in Section 3.4, our sampling-based approach relies on the ability to decompose the output y into separate dimensions to ensure the search in RLOO is decomposed into independent subproblems. Together, this limits the scalability of NESYDM to tasks with either efficient circuit representations or decomposable output spaces. Understanding how to combine these two aspects, or how to automatically (and approximately) reduce a different setting into one of them, is an interesting and challenging future venue. Two other areas of improvement are our approach to maximising the variational entropy and the influence of the indirect gradient coming from sampling from the variational distribution. Finally, we believe studying how NESYDMS extend to other discrete diffusion models than masked diffusion [9] models is an interesting direction. NESYDM could even be extended to hybrid diffusion models that involve both symbolic, discrete concepts and continuous latent variables by using recent work on generating under continuous constraints [20, 40, 76].

##### Acknowledgements

Emile van Krieken was funded by ELIAI (The Edinburgh Laboratory for Integrated Artificial Intelligence), EPSRC (grant no. EP/W002876/1). Pasquale Minervini was partially funded by ELIAI, EPSRC (grant no. EP/W002876/1), an industry grant from Cisco, and a donation from Accenture LLP. Edoardo M. Ponti is supported by the ERC Starting Grant AToM-FM (101222956). Antonio Vergari was supported by the “UNREAL: Unified Reasoning Layer for Trustworthy ML” project (EP/Y023838/1) selected by the ERC and funded by UKRI EPSRC. We would like to express our gratitude to Samuele Bortolotti, Emanuele Marconato, Lennert de Smet, Adri´an Javaloy, and Jaron Maene for fruitful discussions during the writing of this paper.

##### References

- [1] Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨¨ un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. CoRR, abs/2402.14740, 2024.
- [2] Kareem Ahmed, Catarina G Belem, Padhraic Smyth, and Sameer Singh. Semantic probabilistic control of language models. arXiv preprint arXiv:2505.01954, 2025.
- [3] Kareem Ahmed, Kai-Wei Chang, and Guy Van den Broeck. A pseudo-semantic loss for autoregressive models with logical constraints. In Thirty-Seventh Conference on Neural Information Processing Systems, 2023.

- [4] Kareem Ahmed, Kai-Wei Chang, and Guy Van den Broeck. Semantic strengthening of neurosymbolic learning. In International Conference on Artificial Intelligence and Statistics, pages 10252–10261. PMLR, 2023.
- [5] Kareem Ahmed, Kai-Wei Chang, and Guy Van den Broeck. Controllable generation via locally constrained resampling. In Neurips Safe Generative AI Workshop 2024, 2024.
- [6] Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck, and Antonio Vergari. Semantic probabilistic layers for neuro-symbolic learning. 35:29944–29959, 2022.
- [7] Kareem Ahmed, Eric Wang, Kai-Wei Chang, and Guy van den Broeck. Neuro-Symbolic Entropy Regularization. 2022.
- [8] Yaniv Aspis, Krysia Broda, Jorge Lobo, and Alessandra Russo. Embed2sym-scalable neurosymbolic reasoning via clustered embeddings. In Proceedings of the International Conference on Principles of Knowledge Representation and Reasoning, volume 19, pages 421–431, 2022.
- [9] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured Denoising Diffusion Models in Discrete State-Spaces, 2023.
- [10] Samy Badreddine, Artur d’Avila Garcez, Luciano Serafini, and Michael Spranger. Logic Tensor Networks. Artificial Intelligence, 303:103649, February 2022.
- [11] Samuele Bortolotti, Emanuele Marconato, Tommaso Carraro, Paolo Morettin, Emile van Krieken, Antonio Vergari, Stefano Teso, and Andrea Passerini. A neuro-symbolic benchmark suite for concept quality and reasoning shortcuts. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.
- [12] Nicola Branchini and V´ıctor Elvira. Generalizing self-normalized importance sampling with couplings, 2024.
- [13] Diego Calanzone, Stefano Teso, and Antonio Vergari. Logically consistent language models via neuro-symbolic integration. In The Thirteenth International Conference on Learning Representations, 2025.
- [14] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.
- [15] Weixin Chen, Simon Yu, Huajie Shao, Lui Sha, and Han Zhao. Neural probabilistic circuits: Enabling compositional and interpretable predictions through logical reasoning. arXiv preprint arXiv:2501.07021, 2025.
- [16] Y Choi, Antonio Vergari, and Guy Van den Broeck. Probabilistic circuits: A unifying framework for tractable probabilistic models. UCLA. URL: http://starai. cs. ucla. edu/papers/ProbCirc20. pdf, page 6, 2020.
- [17] Alessandro Daniele, Emile van Krieken, Luciano Serafini, and Frank van Harmelen. Refining neural network predictions using background knowledge. Machine Learning, 112(9):3293–3331, 2023.
- [18] Adnan Darwiche and Pierre Marquis. Knowledge compilation: Preface. Annals of Mathematics and Artificial Intelligence, 92(5):1007–1011, 2024.
- [19] Lennert De Smet and Pedro Zuidberg Dos Martires. A fast convoluted story: Scaling probabilistic inference for integer arithmetics. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [20] Lennert De Smet, Pedro Zuidberg Dos Martires, Robin Manhaeve, Giuseppe Marra, Angelika Kimmig, and Luc De Readt. Neural probabilistic logic programming in discrete-continuous domains. In Uncertainty in Artificial Intelligence, pages 529–538. PMLR, 2023.
- [21] Lennert De Smet and Pedro Zuidberg Dos Martires. A fast convoluted story: Scaling probabilistic inference for integer arithmetics. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 102456–102478. Curran Associates, Inc., 2024.
- [22] Lauren Nicole DeLong, Yojana Gadiya, Paola Galdi, Jacques D Fleuriot, and Daniel DomingoFern´andez. Mars: A neurosymbolic approach for interpretable drug discovery. arXiv preprint arXiv:2410.05289, 2024.

- [23] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
- [24] Luca Di Liello, Pierfrancesco Ardino, Jacopo Gobbi, Paolo Morettin, Stefano Teso, and Andrea Passerini. Efficient Generation of Structured Objects with Constrained Adversarial Networks, 2020.
- [25] Jonathan Feldstein, Paulius Dilkas, Vaishak Belle, and Efthymia Tsamoura. Mapping the neurosymbolic ai landscape by architectures: A handbook on augmenting deep learning through symbolic reasoning. arXiv preprint arXiv:2410.22077, 2024.
- [26] Artur d’Avila Garcez and Luis C Lamb. Neurosymbolic ai: The 3 rd wave. Artificial Intelligence Review, 56(11):12387–12406, 2023.
- [27] Eleonora Giunchiglia, Mihaela C˘at˘alina Stoian, Salman Khan, Fabio Cuzzolin, and Thomas Lukasiewicz. Road-r: the autonomous driving dataset with logical requirements. Machine Learning, 112(9):3261–3291, 2023.
- [28] Eleonora Giunchiglia, Alex Tatomir, Mihaela C˘at˘alina Stoian, and Thomas Lukasiewicz. Ccn+: A neuro-symbolic framework for deep learning with requirements. International Journal of Approximate Reasoning, 171:109124, 2024.
- [29] Wei Guo, Yuchen Zhu, Molei Tao, and Yongxin Chen. Plug-and-play controllable generation for discrete masked models, 2024.
- [30] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In International conference on learning representations, 2017.
- [31] Yujia Huang, Adishree Ghatare, Yuanzhe Liu, Ziniu Hu, Qinsheng Zhang, Chandramouli S Sastry, Siddharth Gururani, Sageev Oore, and Yisong Yue. Symbolic music generation with non-differentiable rule guided diffusion. In Proceedings of the 41st International Conference on Machine Learning, pages 19772–19797, 2024.
- [32] Adrian Javaloy, Maryam Meghdadi, and Isabel Valera. Mitigating modality collapse in multimodal VAEs via impartial optimization. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9938–9964. PMLR, 17–23 Jul 2022.
- [33] Richard M Karp, Michael Luby, and Neal Madras. Monte-carlo approximation algorithms for enumeration problems. Journal of algorithms, 10(3):429–448, 1989.
- [34] Adem Kikaj, Giuseppe Marra, Floris Geerts, Robin Manhaeve, and Luc De Raedt. Deepgraphlog for layered neurosymbolic ai. arXiv preprint arXiv:2509.07665, 2025.
- [35] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 [cs], January 2017.
- [36] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational Diffusion Models. 2021.
- [37] Doga Kisa, Guy Van den Broeck, Arthur Choi, and Adnan Darwiche. Probabilistic sentential decision diagrams. In KR, 2014.
- [38] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free!, 2019.
- [39] Emile van Krieken, Pasquale Minervini, Edoardo Ponti, and Antonio Vergari. Neurosymbolic reasoning shortcuts under the independence assumption. In Leilani H. Gilpin, Eleonora Giunchiglia, Pascal Hitzler, and Emile van Krieken, editors, Proceedings of The 19th International Conference on Neurosymbolic Learning and Reasoning, volume 284 of Proceedings of Machine Learning Research, pages 285–302. PMLR, 08–10 Sep 2025.
- [40] Leander Kurscheidt, Paolo Morettin, Roberto Sebastiani, Andrea Passerini, and Antonio Vergari. A probabilistic neuro-symbolic layer for algebraic constraint satisfaction. arXiv preprint arXiv:2503.19466, 2025.

- [41] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
- [42] Alexander K Lew, Tan Zhi-Xuan, Gabriel Grand, and Vikash Mansinghka. Sequential monte carlo steering of large language models using probabilistic programs. In ICML 2023 Workshop: Sampling and Optimization in Discrete Space, 2023.
- [43] Ziyang Li, Jiani Huang, and Mayur Naik. Scallop: A language for neurosymbolic programming. Proceedings of the ACM on Programming Languages, 7(PLDI):1463–1487, 2023.
- [44] Anji Liu, Oliver Broadrick, Mathias Niepert, and Guy Van den Broeck. Discrete copula diffusion. arXiv preprint arXiv:2410.01949, 2024.
- [45] Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. In International Conference on Learning Representations, 2020.
- [46] Jo˜ao Loula, Benjamin LeBrun, Li Du, Ben Lipkin, Clemente Pasti, Gabriel Grand, Tianyu Liu, Yahya Emara, Marjorie Freedman, Jason Eisner, Ryan Cotterell, Vikash Mansinghka, Alexander K. Lew, Tim Vieira, and Timothy J. O’Donnell. Syntactic and semantic control of large language models via sequential monte carlo. In The Thirteenth International Conference on Learning Representations, 2025.
- [47] Calvin Luo. Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970, 2022.
- [48] Jaron Maene and Luc De Raedt. Soft-unification in deep probabilistic logic. Advances in Neural Information Processing Systems, 36:60804–60820, 2023.
- [49] Jaron Maene, Vincent Derkinderen, and Luc De Raedt. On the hardness of probabilistic neurosymbolic learning. In Forty-first International Conference on Machine Learning, 2024.
- [50] Jaron Maene, Vincent Derkinderen, and Pedro Zuidberg Dos Martires. Klay: Accelerating arithmetic circuits for neurosymbolic ai. In The Thirteenth International Conference on Learning Representations, 2025.
- [51] Robin Manhaeve, Sebastijan Dumanˇci´c, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. DeepProbLog: Neural probabilistic logic programming. In Samy Bengio, Hanna M Wallach, Hugo Larochelle, Kristen Grauman, Nicol`o Cesa-Bianchi, and Roman Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December 2018, Montr´eal, Canada, 2018.
- [52] Robin Manhaeve, Sebastijan Dumanˇci´c, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Neural probabilistic logic programming in DeepProbLog. Artificial Intelligence, 298:103504, 2021.
- [53] Emanuele Marconato, Gianpaolo Bontempo, Elisa Ficarra, Simone Calderara, Andrea Passerini, and Stefano Teso. Neuro-symbolic continual learning: Knowledge, reasoning shortcuts and concept rehearsal. arXiv preprint arXiv:2302.01242, 2023.
- [54] Emanuele Marconato, Samuele Bortolotti, Emile van Krieken, Paolo Morettin, Elena Umili, Antonio Vergari, Efthymia Tsamoura, Andrea Passerini, and Stefano Teso. Symbol grounding in neuro-symbolic ai: A gentle introduction to reasoning shortcuts, 2025.
- [55] Emanuele Marconato, Samuele Bortolotti, Emile van Krieken, Antonio Vergari, Andrea Passerini, and Stefano Teso. BEARS Make Neuro-Symbolic Models Aware of their Reasoning Shortcuts. In Uncertainty in Artificial Intelligenc, February 2024.
- [56] Emanuele Marconato, Stefano Teso, Antonio Vergari, and Andrea Passerini. Not All NeuroSymbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts. In Thirty-Seventh Conference on Neural Information Processing Systems, May 2023.
- [57] Giuseppe Marra, Sebastijan Dumanˇci´c, Robin Manhaeve, and Luc De Raedt. From statistical relational to neurosymbolic artificial intelligence: A survey. Artificial Intelligence, 328:104062, 2024.
- [58] Eleonora Misino, Giuseppe Marra, and Emanuele Sansone. Vael: Bridging variational autoencoders and probabilistic logic programming. Advances in Neural Information Processing Systems, 35:4667–4679, 2022.

- [59] Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte carlo gradient estimation in machine learning. Journal of Machine Learning Research, 21:132:1–132:62, 2020.
- [60] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI conference on artificial intelligence, volume 29, 2015.
- [61] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2025.
- [62] Mathias Niepert, Pasquale Minervini, and Luca Franceschi. Implicit mle: backpropagating through discrete exponential family distributions. Advances in Neural Information Processing Systems, 34:14567–14579, 2021.
- [63] Umut Oztok and Adnan Darwiche. A top-down compiler for sentential decision diagrams. In IJCAI, volume 15, pages 3141–3148, 2015.
- [64] Marin Vlastelica Poganˇci´c, Anselm Paulus, Vit Musil, Georg Martius, and Michal Rolinek. Differentiation of blackbox combinatorial solvers. In International Conference on Learning

- Representations, 2019.

[65] Marin Vlastelica Poganˇci´c, Anselm Paulus, Vit Musil, Georg Martius, and Michal Rolinek. Differentiation of blackbox combinatorial solvers. In International Conference on Learning

- Representations, 2020.

- [66] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.
- [67] Matthew Richardson and Pedro Domingos. Markov logic networks. Machine learning, 62:107– 136, 2006.
- [68] Subham Sekhar Sahoo, NYC Cornell Tech, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. 2024.
- [69] Davide Scassola, Sebastiano Saccani, Ginevra Carbone, and Luca Bortolussi. Conditioning score-based generative models by neuro-symbolic constraints. arXiv e-prints, pages arXiv–2308, 2023.
- [70] John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel. Gradient estimation using stochastic computation graphs. Advances in neural information processing systems, 28, 2015.
- [71] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [72] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. arXiv preprint arXiv:2406.04329, 2024.
- [73] Lennert De Smet, Emanuele Sansone, and Pedro Zuidberg Dos Martires. Differentiable sampling of categorical distributions using the catlog-derivative trick. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [74] Lennert De Smet, Gabriele Venturato, Luc De Raedt, and Giuseppe Marra. Relational neurosymbolic markov models. In Toby Walsh, Julie Shah, and Zico Kolter, editors, AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 16181–16189. AAAI Press, 2025.
- [75] Mihaela C Stoian, Salijona Dyrmishi, Maxime Cordy, Thomas Lukasiewicz, and Eleonora Giunchiglia. How realistic is your synthetic data? constraining deep generative models for tabular data. In The Twelfth International Conference on Learning Representations, 2024.
- [76] Mihaela C Stoian and Eleonora Giunchiglia. Beyond the convexity assumption: Realistic tabular data generation under quantifier-free real linear constraints. In The Thirteenth International Conference on Learning Representations, 2025.

- [77] Frank Van Harmelen and Annette Ten Teije. A boxology of design patterns for hybrid learning and reasoning systems. Journal of Web Engineering, 18(1-3):97–123, 2019.
- [78] Emile van Krieken, Erman Acar, and Frank van Harmelen. Analyzing differentiable fuzzy logic operators. Artificial Intelligence, 302:103602, 2022.
- [79] Emile van Krieken, Pasquale Minervini, Edoardo M Ponti, and Antonio Vergari. On the independence assumption in neurosymbolic learning. In Proceedings of the 41st International Conference on Machine Learning, pages 49078–49097, 2024.
- [80] Emile van Krieken, Thiviyan Thanapalasingam, Jakub Tomczak, Frank Van Harmelen, and Annette Ten Teije. A-nesi: A scalable approximate method for probabilistic neurosymbolic inference. Advances in Neural Information Processing Systems, 36:24586–24609, 2023.
- [81] Emile van Krieken, Thiviyan Thanapalasingam, Jakub Tomczak, Frank van Harmelen, and Annette Ten Teije. A-NeSI: A scalable approximate method for probabilistic neurosymbolic inference. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 24586–24609. Curran Associates, Inc., 2023.
- [82] Emile van Krieken, Jakub Tomczak, and Annette Ten Teije. Storchastic: A framework for general stochastic automatic differentiation. Advances in Neural Information Processing Systems, 34:7574–7587, 2021.
- [83] Antonio Vergari, YooJung Choi, Anji Liu, Stefano Teso, and Guy Van den Broeck. A compositional atlas of tractable circuit operations for probabilistic inference. Advances in Neural Information Processing Systems, 34:13189–13201, 2021.
- [84] Antonio Vergari, Nicola Di Mauro, and Guy Van den Broeck. Tractable probabilistic models: Representations, algorithms, learning, and applications, 2019. In Tutorial at the 35th Conference on Uncertainty in Artificial Intelligence (UAI 2019).
- [85] Victor Verreet, Lennert De Smet, Luc De Raedt, and Emanuele Sansone. Explain, agree, learn: Scaling learning for neural probabilistic logic. arXiv e-prints, pages arXiv–2408, 2024.
- [86] Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, and Guy den Broeck. A semantic loss function for deep learning with symbolic knowledge. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80, pages 5502–5511, Stockholmsm¨assan, Stockholm Sweden, 2018. PMLR.
- [87] Yiran Xu, Xiaoyin Yang, Lihang Gong, Hsuan-Chu Lin, Tz-Ying Wu, Yunsheng Li, and Nuno Vasconcelos. Explainable object-induced action decision for autonomous vehicles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9523–9532, 2020.
- [88] Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Beyond autoregression: Discrete diffusion for complex reasoning and planning. In The Thirteenth International Conference on Learning Representations, 2025.
- [89] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b, 2025.
- [90] Runpeng Yu, Qi Li, and Xinchao Wang. Discrete diffusion in large language and multimodal models: A survey, 2025.
- [91] Honghua Zhang, Meihua Dang, Nanyun Peng, and Guy Van den Broeck. Tractable control for autoregressive language generation. In International Conference on Machine Learning, pages 40932–40945. PMLR, 2023.
- [92] Honghua Zhang, Po-Nien Kung, Masahiro Yoshida, Guy Van den Broeck, and Nanyun Peng. Adaptable logical control for large language models. Advances in Neural Information Processing Systems, 37:115563–115587, 2024.
- [93] Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Baker Grosse. Probabilistic inference in language models via twisted sequential monte carlo. In International Conference on Machine Learning, pages 60704–60748. PMLR, 2024.
- [94] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024.

##### A Additional background on masked diffusion models

Here, we will discuss additional background and formalisation of masked diffusion models (MDMs). This background is used to derive the NELBO of the masked diffusion model in Section D.2 and the loss with arbitrary joints in Section C.

Forward process details. We first define the continuous-time forward process q(ct | c0), which masks the data up to timestep t ∈ [0,1] using the forward process defined in Eq. 2.

C

q(ct | c0) =

αt[c ti = c0i] + (1 − αt)[c ti = m] (9)

i=1

Secondly, we need the reverse posterior q(cs | ct,c0), which is the distribution of the initial state c0 given the state at timestep t and the final state. Here we assume cti is either equal to the mask value m or to the value of c0i, as otherwise the probability is not well-defined. The form for each case is (see [68], A.2.1)

C

q(cs | ct,c0) =

q(csi | cti,c0i) (10) q(csi | cti = c0i,c0i) = [c si = c0i] (11) q(csi | cti = m,c0i) =

i=1

αs − αt 1 − αt

- 1 − αs

- 1 − αt

[c si = m] +

[c si = c0i] (12)

We note that q(csi | cti = c0i,c0i) refers to the probability of csi conditioned on some value for the variable c0i and where the value of variable cti equals this value. If cti indeed is equal to the value of c0i, the distribution deterministically returns that value. If it is masked instead, it either stays masked or turns into the value of c0i with a probability depending on αt.

Additional notation. We let Mct = {i : cti = m} refer to the dimensions that are masked in ct. Similarly, Uct = {i : cti ̸= m} is the set of unmasked dimensions of ct. Furthermore, we will use cs ⪰ ct to denote that cs is a (partial) extension of ct. This means cs agrees on all unmasked dimensions of ct with ct, that is, wis = wit for all i ∈ Uct. We will also use c0 ⪰C ct to denote that c0 is a complete extension that does not have any masked dimensions. Finally, we use notation such

- as csU

to index cs using the set of indices Uct, the unmasked dimensions of ct.

ct

Reverse process definition. Using pθ(cs | ct) (Eq. 3), we can express the intractable generative model pMDMθ (c0), for time discretisation T, as

pMDMθ (c0) :=

C\{0}

T

pθ(cs(k) | ct(k)), (13)

k=1

T −1

where the sum over C\{0} iterates over all trajectories c1,...,c

T from fully masked c1 = m to unmasked c0, and s(k) = k−T1 and t(k) = Tk index the timesteps.

Several recent papers [68, 72] proved that this model has a simple negative variational lower bound (NELBO) under a continuous-time process, that is, when T → ∞. Given a dataset of samples c0, this NELBO resembles a weighted cross-entropy loss:

 . (14)

  αt′

−log pMDMθ (c0) ≤ LMDM = Et∼[0,1],ct∼q(ct|c0)

log pθ(˜c0i = c0i | ct)

1 − αt i∈M

ct

Here αt′ = ∂α

∂t , q(ct | c0) is computed with Eq. 9, and the cross-entropy term computes the loss on the factors of the unmasking model pθ(˜c0i | ct). When using the common linear noising schedule, then αt = 1 − t, α

t

′ t

1−αt = −1t. This bound holds when the unmasking model pθ(c˜0 | ct) assigns 0 probability to the mask value (zero masking probabilities), and assigns a probability of 1 to unmasked dimensions (carry-over unmasking), i.e., for all i ̸∈ Mct, pθ(˜c0i = c0i | ct) = 1 [68].

##### B Analysis of the output unmasking loss

Here, we will discuss the output unmasking loss Ly in more detail, and relate it to other common loss functions in the NeSy literature. In our problem setup, we assume a program φ : [V ]C → [V ]Y that maps concepts c0 to outputs y0. Then, we defined the WMC in Eq. 1 as the probability that some c0 maps to y0. This constraint can be understood as

Y

[φ(c 0) = y0] =

φ(c0)i = yi0 . (15)

i=1

That is, we can see this setup as actually having Y different programs, and we want each program to return the right output. Now, disregarding the weighting and sampling, Ly is

Y

pθ(c˜0 | ct,x)[φ( c˜0)i = yi0] (16)

Ly =

log

i=1

c ˜0

Y

log pθ(˜yi0 = yi0 | ct,x) (17)

=

i=1

This loss is a sum of Y different WMC terms, one for each of the Y different programs. Ly assumes, in a vacuum, that these programs are independent, meaning we can sum the losses for each program independently. How could that be possible?

This is actually a common property of continuous-time losses of discrete diffusion models. For instance, one can observe the same in the NELBO of MDMs in Eq. 14. There, the goal is to reconstruct the (masked) dimensions of c0 independently. In fact, to perfectly fit an MDM, the goal is merely to perfectly fit each of the C different conditional data marginals p(˜c0i | ct) perfectly, without regard for any dependencies between dimensions [44]. The dependencies for the full MDM are handled by the iterative unmasking process, which changes the condition at each step. The same property holds for Ly: the dependencies between the different programs are (ideally) handled by different conditions c0 at each step.

We highlight that this loss is related to existing loss functions in the NeSy literature. In particular, for programs that implement conjunctive normal forms (CNFs), this loss is equivalent to the logarithm of the product t-norm, which is a common loss function in the NeSy literature [10, 78]. More precisely, if c ∈ {0,1}C models the C variables of the CNF and y ∈ {0,1}Y the Y clauses consisting of

, then φ(c)i = kj=1i lij computes the truth value of the ith clause of the CNF. Under the independence assumption, the probability that disjunction i holds (that is, whether φ(c)i = 1) is

disjunctions of literals li1 ∨ ... ∨ li,k

i

ki

(1 − pθ(lij | x)) (18)

pθ(yi = 1 | x) = 1 −

j=1

which is equal to the product t-conorm of the probabilities of the literals. Finally, the logarithm product t-norm takes the logarithm over the product of these probabilities, implicitly assuming these clauses are independent:

Y

LLog-product = −

log pθ(yi = 1 | x). (19)

i=1

Note that, outside the reweighting with αt′, this is precisely what Ly would compute for this problem (Eq. 17).

This equality between Ly and LLog-product holds only for CNFs: for general programs, the product t-norm is not equal to the probability on the output of a program, unlike the disjunction case. For example, the different subprograms used in our experiments are not expressed as CNFs. Furthermore, our setup gives more flexibility even in the CNF case by allowing us to redefine what the dimensions of y represent. For instance, we can remove the independence assumption between a set of clauses by defining yi as the conjunction of these clauses. In that sense, it is highly related to Semantic Strengthening [4], which starts from LLog-product, and then dynamically joins clauses by building a probabilistic circuit to relax the independence assumption. This idea can be directly applied to our setup, which we leave as future work.

##### C Masked Diffusion with Arbitrary Joint Distributions

In this section, we will prove Theorem C.1 which states that the NELBO in Eq. 14 also holds for non-factorised unmasking models pθ(c˜0 | ct). We use the notation introduced in Section A and Section 2.2. During this proof, we will derive both discrete- and continuous-time versions of the NELBO. In this appendix, we will use C\0 to refer to c1/T,...,c1, t = Tk and s = k−T1. This result is related to the tractability result of [14], namely that in a continuous-time process, the probability that two dimensions are unmasked at exactly the same time step in [0,1] is 0.

Theorem C.1. Let pθ(c˜0 | ct) be any conditional joint distribution over c˜0 with conditional marginals pθ(˜ci | ct) that satisfy the following assumptions for all i ∈ {1,...,C}:

- 1. Zero masking probabilities: pθ(˜ci = m | ct) = 0.
- 2. Carry-over unmasking: Given some ct ∈ (Vc + 1)C, pθ(˜ci = cti | ct) = 1.
- 3. Proper prior: pθ(c1) = [c 1 = m].

Let pθ(cs | ct) be the reverse process defined in Eq. 3 using pθ(c˜0 | ct) instead of a fully factorised model. Then as T → ∞,

  αt′

 . (20)

−log pMDMθ (c0) ≤ LMDM = Et∼[0,1],ct∼q

log pθ(˜c0i = c0i | ct)

1 − αt i∈M

ct

Proof. We start with a standard variational diffusion models derivation that closely follows those presented in [36, 47].

−log pMDMθ (c0) = −log

C\0

- pθ(C)

- q(C\0 | c0)

pθ(C) ≤ −Eq(C

\0|c0) log

Now we reduce the nominator with Bayes theorem and by conditioning on c0, which is conditionally independent given cs:

T

T

q(ct | cs,c0)

q(ct | cs) = q(c1/T | c0)

q(C\0 | c0) = q(c1/T | c0)

k=2

k=2

(21)

T

T

q(cs | ct,c0)q(ct | c0) q(cs | c0)

= q(c1/T | c0)

= q(c1 | c0)

q(cs | ct,c0),

k=2

k=2

where in the last step we use that the q(ct | c0) and q(cs | c0) cancel out in the product over t, leaving only q(ct | cs,c0). Filling in Eq. 3,

T

- k=2 pθ(cs | ct)

T

- k=2 q(cs | ct,c0)

pθ(c0 | c1/T)p(c1) q(c1 | c0)

(22)

= −Eq(C

\0|c0) log

T

- k=2 pθ(cs | ct)

T

- k=2 q(cs | ct,c0)

- p(c1)

- q(c1 | c0)

\0|c0) log pθ(c0 | c1/T) + log

(23)

= −Eq(C

+ log

T

###### = Eq(c1/T|c0) − log pθ(c0 | c1/T)

Eq(ct|c0)KL[q(cs | ct,c0)∥pθ(cs | ct)] Lunm,T,k: unmasking loss at timestep k

+G (24)

+

k=2

Lrec,T : reconstruction loss

1)

where G = Eq(c1|c0) log p(c

q(c1|c0) is a constant and equal to 0 if p(c1) = [c 1 = m].

- Lemma C.2. Using the assumptions of Theorem C.1, for any integer T > 1, Lrec,T = Eq(c1/T|c0)[−log pθ(c˜0 = c0 | c1/T)] (25)

Proof. First note that, using Eq. 12,

α0 − α1/T 1 − α1/T

- 1 − α0

- 1 − α1/T

q(c0i | c1i/T = m,c˜0i) =

[c 0i = m] +

[˜c 0i = c0i]

1 − α1/T 1 − α1/T

1 − 1 1 − α1/T · 0 +

[c 0i = c0i] = [˜c 0i = c0i]

=

since no elements of c˜0 are masked and α0 = 1 by definition, and so combined with Eq. 11, we get q(c0i | c1i/T,c˜0i) = [˜c 0i = c0i]. Therefore,

C

q(c0i | c1i/T,c˜0i)

pθ(c˜0 | c1/T)

Eq(c1/T|c0)[−log pθ(c0 | c1/T)] = Eq(c1/T|c0) −log

i=1

c ˜0

C

pθ(c˜0 | c1/T)

[˜c 0i = c0i]

= Eq(c1/T|c0) −log

i=1

c ˜0

= Eq(c1/T|c0)[−log pθ(c˜0 = c0 | c1/T)] Where we use that the only nonzero term in the sum is when c˜0 = c0.

| |
|---|

Next, we focus on Lunm,T,k in Eq. 24. The standard derivation of the MDM NELBO in [68] computes the dimension-wise KL-divergence between the forward and reverse process, and then sums. This is not possible in our setting because we assume arbitrary joints for the unmasking model, and so the KL-divergence does not decompose trivially.

- Lemma C.3. Using the assumptions of Theorem C.1, for any integer T > 1 and k ∈ {2,...,T},

Lunm,T,k = Eq(ct|c0)KL[q(cs | ct,c0)∥pθ(cs | ct)]

(26)

= Eq(cs,ct|c0) − log pθ(c˜0U

cs\Uct = c0U

cs\Uct | ct)

Proof. We first consider what terms in the KL-divergence in Lunm,T,k are nonzero. First, note that ct needs to extend cs (i.e., cs ⪰ ct) as otherwise q(cs | ct,c0) = 0 by Eq. 11. Next, the unmasked dimensions in cs need to be consistent with c0 by Eq. 12, in other words, c0 ⪰ cs. Then, the |Mcs| dimensions that stay unmasked get a factor of 1−αs

1−αt , while the |Mct|−|Mcs| dimensions that become unmasked get a factor of αs−αt

1−αt . Assuming c0 ⪰C ct, we have

 

|Mcs| αs−αt 1−αt

|Mct|−|Mcs|

- 1−αs

- 1−αt

if c0 ⪰ cs ⪰ ct, 0 otherwise.

q(cs | ct,c0) =

(27)



Filling this into the KL of Lunm,T,k,

Eq(ct|c0)KL[q(cs | ct,c0)∥pθ(cs | ct,c0)]

q(cs | ct,c0)

(28)

q(cs | ct,c0)log

=Eq(ct|c0)

c ˜0 pθ(c˜0 | ct)q(cs | ct,c˜0)

c0⪰cs⪰ct

Now because of the carry-over unmasking assumption, we know that the only c˜0’s getting positive probabilities are those that extend ct. Focusing just on the log-ratio above and using Eq. 27 and Eq. 12 we have

|Mct|−|Mcs|

|Mcs| αs−αt 1−αt

- 1−αs

- 1−αt

log

|Mcs| αs−αt 1−αt

|Mct|−|Mcs|

c ˜0⪰ct pθ(c˜0 | ct) 1−α

i∈Ucs\Uct [˜c 0i = c0i]

s 1−αt

pθ(c˜0 | ct)

[˜c 0i = c0i] = −log

pθ(c˜0 | ct),

= − log

c ˜0⪰ct

c ˜0⪰cs

i∈Ucs\Uct

since the ratio’s involving αt and αs are independent of c˜0 and can be moved out of the sum, dividing away. Then note that i∈U

cs\Uct [˜c 0i = c0i] also requires that c˜0 extends cs. Giving the denoising loss:

−q(cs | ct,c0)log

pθ(c˜0 | ct) (29)

Lunm,T,k = Eq(ct|c0)

c0⪰cs⪰ct

c ˜0⪰cs

cs | ct) (30)

= ctU

pθ(c˜0 | ct) = Eq(cs,ct|c0) − log pθ(c˜0U

= Eq(cs,ct|c0) − log

cs

c ˜0⪰cs

= Eq(cs,ct|c0) − log pθ(c˜0U

= ctU

,c˜0U

cs\Uct = csU

cs\Uct | ct) (31)

ct

ct

cs\Uct | ct) (32)

cs\Uct = csU

= Eq(cs,ct|c0) − log pθ(c˜0U

= ctU

ct | ct)pθ(c˜0U

ct

cs\Uct | ct), (33) where we use the carry-over unmasking assumption twice. In Eq. 33, we use that pθ(c˜0U

cs\Uct = csU

= Eq(cs,ct|c0) − log pθ(c˜0U

= ctU

| ct) = 1 because pθ(˜c0i = cti | ct) = 1 for all i ∈ Uct, and so the joint over the variables c˜0U

ct

ct

ct

must also be deterministic and return ctU

is conditionally independent of c˜0U

. Similarly, in Eq. 32, we use that c˜0U

ct

ct

cs\Uct given ct since the support of c˜0U

has only one element.

| |
|---|

ct

Combining Eq. 24 and Lemmas C.2 and C.3, we get the discrete-time loss:

LMDMT =Eq(c 1

1

T |c0)[−log p(c˜0 = c0 | c

###### T )]+

(34)

T

Eq(cs,ct|c0) −log pθ(c˜0U

cs\Uct = csU

cs\Uct | ct) .

k=2

cs\Uct | ct) is the marginal probability of the newly unmasked dimensions in cs: Ucs \ Uct. Therefore, computing the discrete-time loss requires being able to be compute conditional marginal distributions over multiple variables. Of course, this is tractable for fully factorised distributions, in which case it’s just a product of individual marginals [68]. This loss can be estimated by sampling pairs cs and ct, and can be further simplified depending on the form of pθ.

pθ(c˜0U

cs\Uct = c0U

Next, we consider LT as T → ∞. We will show that this allows us to marginalise out cs, reducing the variance. We will do this by considering the two loss terms individually, and letting T → ∞.

- Lemma C.4. Using the assumptions of Theorem C.1,

lim

T→∞

Lrec,T = 0 (35)

Proof. Recall that in discrete time this is equal to (see Lemma C.2)

Lrec,T = Eq(c 1

T |c0) − log p(c˜0 = c0 | c

1

T ) (36)

Note that q(c1i/T = m | c0i) = 1 − α1/T. Then, limT→∞ α1/T = limt→0 αt = 1 by continuity and monotonicity of αt, giving limT→∞ q(c1i/T = m | c0i) = 0. Therefore, asymptotically, for all c1/T ̸= c0, we are left with a term that tends to 0 and a constant term independent of T, meaning the only relevant element of the sum is c1/T = c0:

lim

T→∞

c1/T

−q(c1/T | c0)log

c ˜0

pθ(c˜0 | c1/T) = lim

T→∞

−log

c ˜0

pθ(c˜0 | c0) (37)

= − log

c ˜0

pθ(c˜0 | c0) = log 1 = 0 (38)

where we use the carry-over unmasking assumption to get the last equality.

| |
|---|

- Lemma C.5. Using the assumptions of Theorem C.1,

∞

αt′ 1 − αt i∈M

log pθ(˜c0i = c0i | ct)] (39)

Lunm,T,k = Et∼(0,1]q(ct|c0)[

lim

T→∞

k=2

ct

Proof. Instead of having a sum over T − 1 timesteps, each computing a KL, we will now sample some t ∼ {T2 ,...,1}, redefining s := t − T1 . Then, we will weight the result by T − 1. Using Lemma C.3,

Lunm,T

lim

T→∞

T ,...,1}Eq(ct|c0)[(T − 1)KL[q(cs | ct,c0)∥pθ(cs | ct)]] (40)

Et∼{ 2

= lim

T→∞

(T − 1)q(cs | ct,c0)log pθ(c˜0U

cs\Uct = csU

cs\Uct | ct)].

=Et∼(0,1]Eq(ct|c0)[−

lim

T→∞

c0⪰cs⪰ct

(41)

|Mcs| αs−αt 1−αt

|Mct|−|Mcs|

Assuming that c0 ⪰ cs ⪰ ct, recall q(cs | ct,c0) = 1−α

, and assume

s 1−αt

- at least one dimension becomes unmasked: |Mct|−|Mcs| > 0. Then, using that limT→∞ 1−α

1−αt = 1, we get

s

|Mct|−|Mcs|

|Mcs| αs − αt 1 − αt

- 1 − αs

- 1 − αt

(T − 1)q(cs | ct,c0) = lim

(42)

lim

T

T→∞

T→∞

|Mct|−|Mcs|−1

T(αs − αt) 1 − αt

αs − αt 1 − αt

(43)

= lim

T→∞

Next, using that αt is differentiable, consider the first-order Taylor expansion of α around t to evaluate αs: αs = αt − T1 αt′ + O(T12 ). Then T(αs − αt) = T(αt − T1 αt′ + O(T12 ) − αt) = −αt′ + O(T1 ). And so limT→∞ T(αs − αt) = −αt′.

Now if |Mct|−|Mcs| ≥ 2, then the following term appears: T(αs −αt)2 = T(−T1 αt′ +O(T12 ))2 = T(α

′2 t

′2 t

′2 t

T2 + O(T13)) = α

T + O(T12 ). And so limT→∞ T(αs − αt)2 = limT→∞ α

###### T + O(T12 ) = 0.

Therefore, the only cs in the sum with a non-zero contribution are where |Mct| − |Mcs| ≤ 1, that is, when cs unmasks at most one dimension of ct. When no dimensions are unmasked, q(cs | ct,c0) = 1,

and if cs unmasks one dimension, we have q(cs | ct,c0) = −αt′. If cs does not unmask any dimensions, then there are no variables in c˜0U

cs\Uct to compute the probability over in −log pθ(c˜0U

cs\Uct | ct), giving probability 1 and a term equal to cs\Uct = csU

cs\Uct = csU

- 0. Next, if cs unmasks only dimension i ∈ Mct such that csi = c0i, then pθ(c˜0U

cs\Uct |

ct) = pθ(˜c0i = c0i | ct). Therefore,

∞

αt′ 1 − αt i∈M

Lunm,T,k = LMDM =Et∼(0,1]q(ct|c0)[

lim

T→∞

k=2

ct

log pθ(˜c0i = c0i | ct)] (44)

| |
|---|

Summing Lemmas C.4 and C.5 completes the proof of Theorem C.1.

| |
|---|

##### D Neurosymbolic diffusion models: formal definition and NELBO derivation

In this section, we will formally define derive the NELBO for neurosymbolic diffusion model. Throughout this section, we assume the same notation as in Section A. We refer to the graphical model in Fig. 2 to set up the model and the notation.

###### D.1 Formal model definition

First, we will define the discrete-time data log-likelihood. We let C be the trajectory of partially masked concepts over timesteps c0,cT1 ,cT2 ,...,c

T −1

T ,c1, and similarly Y\{0} is the trajectory

### x

#### ... ...

### c0 cs ct c1

... ys yt y1

...

### y0

x conditioning forward q reverse p✓

Figure 2: Probabilistic graphical model for neurosymbolic diffusion model. The forward process q, indicated by striped arrows, masks both concepts c and outputs y. Since only y0 is observed, a variational distribution qθ has to predict c0 from y0 and x. The reverse process, with regular arrows, unmasks both concepts c and outputs y, transforming concepts into outputs at every time step.

T −1

yT1 ,yT2 ,...,y

T ,y1 Marginalising out all latent variables according to the graphical model in the bottom of Fig. 2, we define the data log-likelihood pθNESYDM(y0 | x) of outputs y given inputs x as:

T

- pθNESYDM(y0 | x) := Y\{0} C

q(c1,y1)

pθ(cs(k) | ct(k),x)pθ(ys(k) | cs(k),yt(k),x). (45)

k=1

Here, pθ(cs | ct,x) and pθ(ys | cs,yt,x) are defined as in Section 3.1. First, we define the conditional program φyt as the program φ that maps concepts to outputs, but always returns yit if dimension i is unmasked in yt. To be precise,

φ(c˜0)i if yit = m yit if yit ̸= m

φyt(c˜0)i =

. (46)

We need this definition in Eq. 4 to ensure the output unmasking model satisfies the carry-over unmasking assumption from Theorem C.1.

Next, we define

pθ(c˜0 | cs,x)[φ yt(c˜0) = y˜0] (47)

pθ(y˜0 | cs,yt,x) :=

c ˜0

as the output unmasking model such that the MDM defined as y ˜0 pθ(y˜0 | cs,x)q(ys | yt,y˜0) is equal to pθ(ys | cs,yt,x) as defined in Eq. 4:

pθ(y˜0 | cs,x)q(ys | yt,y˜0) =

pθ(c˜0 | cs,x)[φ yt(c˜0) = y˜0]q(ys | yt,y˜0) (48)

y ˜0

y ˜0 c ˜0

=

c ˜0

pθ(c˜0 | cs,x)q(ys | yt,φyt(c˜0)) = pθ(ys | cs,yt,x).

(49)

Note that pθ(ys | cs,yt,x) does not decompose into a product of marginals, requiring the new results in Section C rather than the standard MDM NELBO derivation.

To be able to use Theorem C.1 and the other lemmas in Section C, we need to ensure that the output unmasking model satisfies the assumptions of Theorem C.1. Since φyt maps completely unmasked concepts to completely unmasked outputs, it satisfies zero masking probabilities. Further, the carry-over unmasking assumption is satisfied, since for any unmasked dimension i ∈ Uyt and any concept c˜0, φyt(c˜0)i = yit by Eq. 46 and hence pθ(˜yi0 = yit | cs,yt,x) = 1. Importantly, the carry-over unmasking assumption would not hold if we used φ(c˜0) instead of φyt(c˜0) in Eq. 47, and we would not have been able to use the results in Section C.

###### D.2 NELBO derivation

Theorem D.1. Let pθ(c˜0 | ct,x) be a concept unmasking model with zero masking probabilities and carry-over unmasking as defined in Theorem C.1, φ : [Vc]C → [Vy]Y be a given program,

- qθ(c0 | y0,x) be a variational distribution, and αt be a noising schedule. Then we have that the data log-likelihood as T → ∞ is bounded as limT→∞ −log pθNESYDM(y0 | x) ≤ LNESYDM, where

αt′ 1 − αt i∈M

log pθ(˜c0i = c0i | ct,x)

LNESYDM =Et∼[0,1],q

θ(c0,ct|x,y0)

ct

###### Lc: Concept unmasking loss

(50)

Y

+ αt′

− H[qθ(c0 | y0,x)]

pθ(c˜0 | ct,x)[φ( c˜0)i = yi0]

log

i=1

c ˜0

LH[q]: Variational entropy

Ly: Output unmasking loss

Proof.

− log pθNESYDM(y0 | x)

pθ(C,Y | x) qψ(C,Y\0 | y0,x)

(51)

pθ(C,Y | x) ≤ −Eq

= −log

θ(C,Y\0|y0,x) log

Y\0 C

Next, we again use the trick from Eq. 21, both for C and Y, and we expand the model pθ following the graphical model in Fig. 2:

T k=2 pθ(cs,ys | ct,yt,x)

- pθ(c0,y0 | cT1 ,yT1 ,x)p(c1,y1)

- qθ(c1,c0 | x,y0)q(y1 | y0)

= Eq

θ(C,Y\0|y0,x) log

T t=2 q(cs | ct,c0)q(ys | yt,y0)

- pθ(c0 | cT1 ,x)

- qθ(c0 | x,y0)

1

T |y0) −log pθ(y0 | c0,y

−Eq(c 1

= Eq

θ(c0|x,y0) Eq(y 1

T |c0) log

T ,x)

+

Lrec,y,T : y0 reconstruction

Lrec,c,T : c0 reconstruction

T

Eq(ct|c0) KL[q(cs | ct,c0)∥pθ(cs | ct,x)]

+

k=2

Lunm,c,T,k: c unmasking

Eq(cs,yt|c0,y0) KL[q(ys | yt,y0)∥pθ(ys | yt,cs,x)]

+ C

Lunm,y,T,k: y unmasking

1,y1)

where C = Eq(c1,y1|y0,x) log p(c

qθ(c1,y1|x,y0) is a constant and equal to 0 if p(c1,y1) = [c 1 = m ∧ y1 = m], which is true by assumption.

Discrete-time NELBO. Next, we use Lemma C.2 to rewrite Lrec,y,T and Lrec,c,T. Then, the first two terms are the reconstruction losses for y0 and c0 respectively, and the third term is the entropy of the variational distribution.

1

T |y0) −log pθ(y˜0 = y0 | c0,y

Lrec,y,T + Lrec,c,T =Eq

θ(c0|x,y0) Eq(y 1

T ,x) +

1

T |c0) −log pθ(c˜0 = c0 | c

T ,x) + log qθ(c0 | y0,x) ,

Eq(c 1

(52)

Similarly, using Lemma C.3, we have the concept unmasking loss as

T

T

Lunm,c,T,k =

k=2

k=2

θ(c0,cs,ct|x,y0) − log pθ(c˜0U

cs\Uct = csU

cs\Uct | ct,x) (53)

Eq

For the output unmasking loss Lunm,y,T, again using Lemma C.3, we have

T

T

cs\Uct | cs,yt,x) . (54)

ys\Uyt = yUs

θ(cs,ys,yt|x,y0) − log pθ(y˜U0

Eq

Lunm,y,T,k =

k=2

k=2

Summing Eqs. 52 to 54, we get the discrete-time NELBO.

Continuous-time NELBO. Using Lemma C.4 twice and adding the entropy of the variational distribution in Eq. 52, and then using Lemma C.5 twice, we get the continuous-time NELBO:

αt′ 1 − αt i∈M

LNESYDM′

log pθ(˜c0i = c0i | ct,x)

=Et∼[0,1],q

###### +

θ(c0,ct|x,y0)

ct

Lc: Concept denoising loss

(55)

log pθ(˜yi0 = yi0 | ct,yt,x)

###### − H[qθ(c0 | y0,x)]

Eq(yt|y0)

.

i∈Myt

LH[q]: Variational entropy

Ly: Output denoising loss

Next, we will further simplify the output unmasking loss Ly with a Rao-Blackwellisation to get the form given in Theorem 3.1. Using Eq. 47,

αt′ 1 − αt i∈M

log pθ(˜yi0 = yi0 | ct,yt,x)

Ly =Et∼[0,1],q

θ(yt,ct|x,y0)

yt

αt′ 1 − αt i∈M

pθ(c˜0 | ct,x)[φ yt(c˜0) = y˜0]

=Et∼[0,1],q

log

θ(yt,ct|x,y0)

y ˜0,y˜i0=yi0 c ˜0

yt

αt′ 1 − αt i∈M

pθ(c˜0 | ct,x)

[φ yt(c˜0) = y˜0]

=Et∼[0,1],q

log

θ(yt,ct|x,y0)

c ˜0

y ˜0,y˜i0=yi0

yt

αt′ 1 − αt i∈M

pθ(c˜0 | ct,x)[φ yt(c˜0)i = yi0] ,

=Et∼[0,1],q

log

θ(ct|x,y0)q(yt|y0)

c ˜0

yt

where in the last step we use that only a single y˜0 satisfies φyt(c˜0) = y˜0, and it appears in the sum only if exactly that y˜0 has y˜i0 = yi0, and the conditional independence in Fig. 2 of yt and ct given y0. Next, define the following inductive hypothesis based on the value of Y :

Y

θ(ct|x,y0) αt′

Ly =Et∼(0,1]Eq

log

i=1

c ˜0

pθ(c˜0 | ct,x)[φ( c˜0)i = yi0] . (56)

Base case Y = 1: The only elements in the support of q(yt | y0) are y1t = y10 and y1t = m. If it is y10 (probability αt), the set of unmasked values is empty, and so the loss is zero. If it is m (probability

- 1 − αt), the only masked dimension is i = 1. Furthermore, there are no unmasked dimensions in yt, hence φyt = φ and so the loss is

θ(ct|x) αt′ log

Ly = Et∼(0,1]Eq

c ˜0

pθ(c˜0 | ct,x)[φ( c˜0)1 = y10] .

Inductive step Y > 1: Assume the result holds for Y − 1. Like in the base case, q(yYt = yY0 | y0) = αt and q(yYt = m | y0) = 1 − αt. Then, let yˆt denote all variables in yt except yYt , and we assume the inductive hypothesis holds for yˆt. We again consider the two cases: Either yYt = yY0 with

probability αt or yYt = m with probability 1 − αt.

  αt′

 

pθ(c˜0 | ct,x)[φ yt(c˜0)i = yi0]

Ly =Et∼(0,1]Eq

log

θ(ct,yt|x,y0)

1 − αt i∈M

c ˜0

yt

αtαt′ 1 − αt i∈M

pθ(c˜0 | ct,x)[φ yˆt(c˜0)i = yi0]

q(yˆt | y0)

=Et∼(0,1]Eq

log

###### θ(ct|x)

y ˆt

c ˜0

yˆt

(1 − αt)αt′ 1 − αt

pθ(c˜0 | ct,x)[φ yˆt(c˜0)i = yi0]

+

log

c ˜0

i∈Myˆt∪{Y }

Note now that the second term contains the same sum over the i ∈ Myˆt as in the first term, but in addition it contains the dimension Y . We next move the other terms into the first term, leaving with a

′ t

weight of α

1−αt for the first term:

Ly =Et∼(0,1]Eq

###### θ(ct|x)

y ˆt

q(yˆt | y0)

αt′ 1 − αt i∈M

log

yˆt

c ˜0

pθ(c˜0 | ct,x)[φ yˆt(c˜0)i = yi0]

+ αt′ log

c ˜0

pθ(c˜0 | ct,x)[φ yˆt(c˜0)Y = yY0 ]

Next, we apply the inductive hypothesis to the first term. After, note that the second term is independent of the value of yˆt as the result of φyˆt(c˜0)Y does not depend on yˆt.

θ(ct|x) αt′

Ly =Et∼(0,1]Eq

+

y ˆt

q(yˆt | y0)αt′ log

θ(ct|x) αt′

=Et∼(0,1]Eq

c ˜0

Y

Y −1

log

i=1

c ˜0

pθ(c˜0 | ct,x)[φ yˆt(c˜0)i = yi0]

pθ(c˜0 | ct,x)[φ yˆt(c˜0)Y = yY0 ]

log

i=1

c ˜0

pθ(c˜0 | ct,x)[φ( c˜0)i = yi0] ,

completing the inductive proof. Finally, replacing Eq. 56 for Ly in Eq. 55 completes the proof.

| |
|---|

##### E Gradient estimation details

In this section, we provide additional details and formalisation on our gradient estimation procedure, extending the discussion in Section 3.4.

Given some input-output pair (x,y0) ∼ D, the gradient of the loss is given by [70, 82]

∇θLNESYDM = ∇θH[qθ(c0 | x,y0)]

+

Gradient of variational entropy ∇θLH[q]

αt′ 1 − αt i∈M

∇θ log pθ(˜c0i = c0i | ct,x)

Et∼[0,1],q

+

θ(c0,ct|x,y0)

ct

Gradient of concept unmasking loss ∇θLc

Y

αt′

∇θpθ(c˜0 | ct,x)[φ( c˜0)i = yi0]

+

log

i=1

c ˜0

Gradient of output unmasking loss ∇θLy

S j=1 [φ( c˜0j)i = yi0]

Y

αt′ 1 − αt i∈M

log pθ(˜c0i = c0i | ct,x) + αt′

S ∇θ log qθ(c0 | x,y0)

log

i=1

ct

Indirect gradient from sampling from variational distribution (ignored)

Monte carlo approximation. We will use a monte carlo approximation to estimate this gradient. We first sample a single c0 ∼ qθ(c0 | x,y0), t ∼ [0,1], and then a single ct ∼ q(ct | c0) using Eq. 9. Finally, we sample S samples c˜01,...,c˜0S ∼ pθ(c˜0 | ct,x) to approximate the gradient of the output unmasking loss Ly with the RLOO estimator [38]. Alternatively, one could use probabilistic circuits to compute this gradient exactly [50, 86].

Indirect gradient. The indirect gradient arises from the expectation over the variational distribution which depends on the parameter θ. This term has high variance in a monte-carlo estimator. Firstly, the vanilla score function estimator is known to have high variance, especially without additional variance reduction techniques [59]. However, the reward, which is given between the large braces, is doubly-stochastic: it depends on sampling t, ct, and c˜0,...,c˜S, making it an inherently noisy process. Furthermore, when using the variational distribution as defined in Section 3.3, the score term ∇θ log qθ(c0 | x,y0) is itself a NESYDM for which computing log-likelihoods is intractable, and thus we would require additional approximations to estimate it. Because of the variance, intractability, and to keep the algorithm simple, we ignore the term altogether.

Therefore, our gradient estimate g is given by

αt′ 1 − αt i∈M

γc C

γH C ∇θH[qθ(c0 | x,y0)]

∇θ log pθ(˜c0i = c0i | ct,x)

+

g =

+

ct

Choose estimate of ∇θLH[q]

Unbiased estimate of ∇θLc

Y

S

γy Y

1 (S − 1)µi

[φ( c˜0j)i = yi0] − µi ∇θ log pθ(c˜0j | ct,x)

,

i=1

j=1

Consistent estimate of ∇θLy

(57)

where µi = S1 Sj=1 [φ( c˜0j)i = yi0] is the empirical mean of the constraints, and γc,γH and γy are weighting coefficients. We keep γy = 1, and tune the other two. Additionally, inspired by the local step approach of [32], we average over dimensions rather than summing to stabilise hyperparameter tuning among different problems. This is especially useful in experiments with variable dimension size such as MNISTAdd and Warcraft Path Planning. We discuss how we estimate the gradient of the entropy of the variational distribution in Section 3.4.

Estimate of ∇θLy Next, we derive the consistent gradient estimator for ∇θLy using the RLOO estimator [38]. Assuming we have some x, y0, t and ct, and using the score-function estimator, the

gradient of the loss is given by

Y

∇θLy =αt′

i=1

Y

=αt′

i=1

Y

=αt′

i=1

Y

=αt′

i=1

∇θ log

c ˜0

pθ(c˜0 | ct,x)[φ( c˜0)i = yi0]

c ˜0 ∇θpθ(c˜0 | ct,x)[φ( c˜0)i = yi0] c ˜0 pθ(c˜0 | ct,x)[φ( c˜0)i = yi0]

c ˜0 pθ(c˜0 | ct,x)[φ( c˜0)i = yi0]∇θ log pθ(c˜0 | ct,x)

Ep

θ(c˜0|ct,x)[[φ( c˜0)i = yi0]]

Ep

θ(c˜0|ct,x)[[φ( c˜0)i = yi0]∇θ log pθ(c˜0 | ct,x)]

Ep

θ(c˜0|ct,x)[[φ( c˜0)i = yi0]]

(58)

Both the numerator and denominator are expectations under pθ(c˜0 | ct,x) of the constraints. A consistent (but not unbiased) estimator is given by sampling S samples c˜01,...,c˜0S ∼ pθ(c˜0 | ct,x) and taking averages at each of these 2Y expectations separately. Then, we will use RLOO as a baseline to reduce the variance of the numerators. A baseline is a constant b, where we use that for any distribution pθ(x), Ep

θ(x)[b∇θ log pθ(x)] = 0, and so by linearity of expectation, Ep

θ(x)[f(x)∇θ log pθ(x)]. Since we are using S samples, we choose for each sample c˜0j and dimension i the baseline bij to be the empirical mean over the other samples, leaving one sample out: bij = S−1 1 l̸=j [φ( c˜0l )i = yi0]. Then, ([φ( c˜0j)i = yi0] − bij)∇θ log pθ(c˜0j | ct,x) is an unbiased estimator of the numerator. Finally, we average over the S different estimators obtained this way to derive the RLOO gradient estimator as:

θ(x)[(f(x) − b)∇θ log pθ(x)] = Ep

θ(c˜0|ct,x)[[φ( c˜0)i = yi0]∇θ log pθ(c˜0 | ct,x)] ≈

Ep

S

1 S

([φ( c˜0j)i = yi0] − bij)∇θ log pθ(c˜0j | ct,x)

j=1

S

(S − 1)[φ( c˜0j)i = yi0] − l̸=j [φ( c˜0l )i = yi0] S(S − 1) ∇θ log pθ(c˜0j | ct,x)

=

j=1

S[φ( c˜0j)i = yi0] − Sl=1 [φ( c˜0l )i = yi0] S(S − 1) ∇θ log pθ(c˜0j | ct,x)

S

=

j=1

S

1 (S − 1)

[φ( c˜0j)i = yi0] − µi ∇θ log pθ(c˜0j | ct,x) (59)

=

j=1

Combining Eqs. 58 and 59 gives the gradient estimator:

Y

S

1 µi(S − 1)

∇θLy ≈ gy0(c˜01,...c˜0S) := αt′

[φ( c˜0j)i = yi0] − µi ∇θ log pθ(c˜0j | ct,x)

i=1

j=1

(60)

Full gradient estimation algorithm. In Algorithm 1, we provide the full algorithm for estimating gradients to train NESYDM. The algorithm proceeds by sampling c0 from the variational distribution, and then sampling a partially masked value ct. We then compute the gradients of the three individual losses using Eq. 57. This requires sampling S samples from the unmasking model, which is done in line 5. Finally, we weight the gradients appropriately and sum them up.

##### F Sampling details

We use the first-hitting sampler [94] if the configured number of discretisation steps T is larger or equal to the dimension of the concept space C. Otherwise, we use a T-step time-discretisation of the reverse process [68].

The first-hitting sampler in Algorithm 3 randomly samples the next timestep to unmask at. There, it randomly selects an index to unmask using the concept unmasking model. Note that α−1 is the inverse of the noising schedule. Since we do not provide the temperature to our neural networks, this sampler is, in practice, a concept-by-concept decoding process similar to masked models like BERT [23, 94].

- Algorithm 3 First-hitting sampler for pθ(c0 | x)

- 1: Input: x, unmasking model pθ(c˜0 | x,ct)
- 2: t ← 1
- 3: c1 = m
- 4: for k ← C to 1 do
- 5: s ← α−1(1 −

k

√u(1 − αt)), where u ∼ U(0,1) ▷ Select next timestep to unmask at

- 6: i ∼ Uniform(Mck) ▷ Select a random dimension to unmask
- 7: cs ← ct
- 8: csi ∼ pθ(˜c0i | x,ct) ▷ Sample the unmasked dimension
- 9: t ← s
- 10: Return c0

Instead, the time-discretised sampler in Algorithm 4 samples a completely unmasked sample c˜0 from the unmasking model at each timestep, then samples cs from the reverse process in Eq. 10 to obtain the next timestep. When sampling from the reverse process, the algorithm remasks some of the newly unmasked dimensions in c˜0, while keeping the unmasked dimensions in ct fixed.

- Algorithm 4 Time discretised sampler for p(c0 | x)

- 1: Input: x, unmasking model pθ(c˜0 | x,ct), number of discretisation steps T
- 2: c1 = m
- 3: for k ← T to 1 do
- 4: c˜0 ∼ pθ(c˜0 | x,ct) ▷ Sample from unmasking model
- 5: cs ∼ q(cs | ct,c0 = c˜0) ▷ Sample from reverse process in Eq. 10
- 6: Return c0

###### F.1 Sampling from the variational distribution

We adapted the two samplers above to sample from our model conditioned on the output y0. We use a simple resampling approach as described in Section 3.3, which we elaborate on here. First, we recall the relaxed constraint for β > 0 as

rβ(c˜0 | y0) = exp(−β

Y

Then, we define the distribution to sample from as

[φ( c˜0)i ̸= yi0]). (61)

i=1

qθ(c˜0 | x,ct,y0) ∝ pθ(c˜0 | x,ct)rβ(c˜0 | y0). (62)

Since we cannot tractably sample from this distribution, we use self-normalised importance sampling [12]. In other words, we sample K samples from the unmasking model, compute the relaxed constraint for each sample, and then normalise these values. Finally, we sample from the renormalised distribution. We provide the full algorithm in Algorithm 5.

We note that the distribution pθ(c˜0 | x,ct) does not appear in the importance weights. This holds because we are sampling from it, thus it divides away in the computation of the importance weights.

We implement this algorithm in the two samplers as follows. For the time-discretised sampler, we replace Algorithm 4 of Algorithm 4 with Algorithm 5. Together, this is the algorithm used in [29]. For the first-hitting sampler, we replace Algorithm 3 of Algorithm 3 by first calling Algorithm 5 to obtain some c˜0, and then returning the i-th dimension c˜0i.

Relation to Markov Logic Networks. The distribution in Eq. 62 is similar to a Markov Logic Network (MLN) [67]. Particularly, the formulas of the MLN are (1): the different constraints

- Algorithm 5 Self-normalised importance sampling for qθ(c˜0 | x,ct,y0)

- 1: Input: x, yt, unmasking model pθ(c˜0 | x,ct), number of samples K
- 2: c˜01,...,c˜0K ∼ pθ(c˜0 | x,ct) ▷ Sample K samples from unmasking model
- 3: wi = rβ(c˜0i | y0), for all i ∈ [K] ▷ Compute importance weights
- 4: Z = Ki=1 wi ▷ Normalisation constant
- 5: i ∼ Categorical(w

i

Z ) ▷ Sample from renormalised distribution

- 6: Return c˜0i

[φ( c˜0)i ̸= yi0], each weighted by −β2, and (2): the unmasking model pθ(c˜0 | x,ct), defined with formulas [˜c 0i = c0i] and weight log pθ(˜c0i = c0i | x,ct). In particular, this means that c˜0’s that violate constraints still have positive energy. However, the energy exponentially shrinks by a factor of

1 exp(β) for each violated constraint. Since we use rather high values of β > 10, the resampling step in Algorithm 5 is extremely likely to pick the sample that violates the least number of constraints.

Numerically stable implementation. In practice, computing Eq. 61 in Algorithm 5 is numerically highly unstable if multiple constraints are violated. Then, the reward is equal to exp(1lβ), where l is the number of violated constraints. PyTorch floats are roughly between exp(−104) and exp(88), meaning that for β > 10, the reward underflows at l = 8. First, we note that when sampling from the reweighted distribution in Algorithm 5, probabilities are computed as the relative proportion of the rewards. Therefore, we can simply rescale all rewards by a constant factor to ensure they do not underflow or overflow. Particularly, we redefine the reward as

rβ,L,U(c˜0 | y0) = exp −max β

Y

[φ( c˜0)i ̸= yi0] − L,U .

i=1

Here, L > 0 acts as a scaling on the reward as rβ,L,U(c˜0 | y0) = rβ(c˜0 | y0)exp(L) if the max is not active. U > 0 acts as a floor on the reward such that samples that violate many constraints still have non-zero probability, even if it is extraordinarily unlikely. We fix U = 100 to remain in the floating point range. Note that this is ever so slightly biased as it will over-estimate the probability of the samples that violate the least number of constraints. However, this bias is very small as the probability of choosing these samples is extraordinarily low.

We want to choose L to maximise the range of the reward among the samples without overflowing. Within this range, the differences between the samples that violate the least number of constraints is most important: these are the samples we are most likely to choose. Our intuition is as follows: we set L to the average number of violated constraints among the samples in Algorithm 5. However, if this would overflow the best sample, we instead set L such that the best sample has a reward of exp(M), where M = 70 to prevent overflow. Therefore, we choose

L = min

S

Y

βt S

S

[φ( c˜0k)i ̸= yi0],M +

min

k=1

i=1

k=1

Y

[φ( c˜0k)i ̸= yi0] . (63)

βt

i=1

##### G Experimental details

NESYDM is implemented in PyTorch. We used RAdam [45] for all experiments except for MNIST Addition, where we used Adam [35]. We did not compare these optimisers in detail, but we do not expect this choice to significantly affect the results. Furthermore, we used default momentum parameters for both optimisers. For all neural networks used to implement the unmasking model pθ(c˜0 | x,ct), we did not pass the current time step t to the network, as previous work found minimal impact on performance for doing so (Appendix E.5 of [68]).

For all experiments, we used GPU computing nodes, each with a single lower-end GPU. In particular, we used NVIDIA GeForce GTX 1080 Ti and GTX 2080 Ti GPUs. All our experiments were run with 12 CPU cores, although this was not the bottleneck in most experiments. On the GTX 1080 Ti, our experiments took between 1 and 17 hours, depending on the complexity of the task and number

2Unlike MLNs, we sum unsatisfied constraints rather than satisfied ones to ensure rβ(c˜0 | y0) ∈ [0, 1].

of epochs. The project required extra compute when testing different variations of the model, and by performing hyperparameter tuning. For repeating our runs and hyperparameter tuning, we further expect around 600 total GPU hours are needed.

###### G.1 Hyperparameter tuning

We list all hyperparameters in Table 4. We perform random search over the hyperparameters on the validation set of the benchmark tasks. For the random search, we used fixed ranges for each parameter, from which we sample log-uniformly. For the parameter β we sampled uniformly instead. We used a budget of 30 random samples for each problem, although for some problems we needed more when we found the ranges chosen were poor.

Several hyperparameters, namely the minibatch size, S, K, T and L, are compute dependent, and we keep these fixed when tuning depending on the compute budget and the problem size. The hyperparameters we do tune are the learning rate, γc, γH, and β. We found that low values of γc were usually fine, and that for large enough β above 10, its value did not matter much. Therefore, the most important hyperparameters to tune are γH and the learning rate, for which the optimal values varied between problems significantly. For an ablation study on the influence of the value of the loss weighting hyperparameters, see Section H.2.

Table 4: All hyperparameters used in the experiments, and rough recommendations for some of their values. We recommend at least tuning learning rate, and γH and γc to some extent (leaving γy at 1). Some hyperparameters are compute dependent, and higher is always better for reducing gradient estimation variance (S,K,T) and majority voting quality (L,T).

###### Variable Recommendation Description Range Definition

— (0.0001,0.0005) Overall learning rate R>0 — — Minibatch size N — — Epochs N —

γy 1 Weight of concept unmasking loss R≥0 Eq. 57 γc 10−5 Weight of output unmasking loss R≥0 Eq. 57 γH (0.002,2) Weight of variational entropy R≥0 Eq. 57

β 10 Penalty in soft constraint R>0 Section 3.3

- S ≥ 4 Number of RLOO samples N Eq. 7

- K ≥ 2 Number of SNIS samples for qθ N Section F.1

T ≥

√

C MDM discretisation steps N Section 3.5

- L ≥ 8 Number of majority voting samples N Section 3.5

###### G.2 MNIST Addition

We use the LeNet architecture [41] for the neural network architecture as is standard in the NeSy literature [52]. As there are no dependencies between the digits in the data generation process, making the neural network conditional on partially unmasked outputs is not useful: merely predicting marginals is sufficient. Therefore, we ignore the conditioning on ct when computing pθ(c˜0 | ct,x) in Eq. 47.

Since there is no standard dataset for multidigit MNIST addition, we use a generator defined as follows: for some dataset of MNIST images, we permute it randomly, then split it into 2N parts and stack them to obtain the different datapoints. This ensures we use each datapoint in the dataset

exactly once, ending up in ⌊600002N ⌋ training datapoints.

We tuned hyperparameters in accordance with Section G.1. Since MNIST has no separate validation dataset, we split the training dataset in a training dataset of 50.000 samples and a validation dataset 10.000 samples before creating the addition dataset. We tune with this split, then again train 10 times with the optimised parameters on the full training dataset of 60.000 samples for the reported test accuracy. We tune on N = 15, and reuse the same hyperparameters for N = 2 and N = 4. For the number of epochs, we use 100 for N = 2 and N = 4 as an epoch is more expensive for smaller N and because N = 15 requires moving beyond a cold-start phase. We found all 10 runs moved past this phase within 100 epochs, but needed more time to converge after.

Table 5: Hyperparameters for MNIST Addition and Warcraft Path Planning.

###### Variable MNIST Addition Path Planning

learning rate 0.0003 0.0005 minibatch size 16 50

epochs N = 4: 100, N = 15: 1000 40 γc 2 · 10−5 10−5 γH 0.01 0.002 γy 1 1

β 20 12

- S 1024 12 × 12: 16, 30 × 30: 4

- K 1024 12 × 12: 4, 30 × 30: 2

T 8 20

- L 8 8

Baselines. For all methods, we take the numbers reported in the papers where possible. We obtained numbers for Scallop from the PLIA paper. For A-NeSI, we pick the best-scoring variant as reported, which is Predict for N = 4 and Explain for N = 15. For DeepSoftLog, A-NeSI and PLIA, we obtained performance on 10 individual runs from the authors to compute the Mann-Whitney U test.

###### G.3 Visual Path Planning

Following [80], we use categorical costs for the Visual Path Planning task. We use Vc = 5, which corresponds to the possible cost values in the data, costs = [0.8,1.2,5.3,7.7,9.2]. Then, c0

corresponds to an index of the cost of each grid cell. That is, c0Ni+j ∈ {1,...,5} corresponds to the cost value costs[c0Ni+j] at grid cell i,j.

We adapted the ResNet18-based architecture from [62] for the unmasking model p(c˜0 | x,ct) over grid costs. This architecture consists of a single convolutional layer to start encoding the image, with batch normalisation and adaptive max-pooling to a grid of size N × N. After this, we have 64-dimensional embeddings for each grid cell. To condition on the currently unmasked values, we add embeddings of ct ∈ {1,...,5,m}N

2

for each cell: we use six 64-dimensional embeddings eC1 ,...,eC5 ,eCm for the different costs plus the mask value. Then we add these embeddings to the image embeddings cell-wise. That is, if eIi,j is the image embedding at cell i,j, then the new embedding is eIi,j + eCct

. After this, a ResNet layer containing two more convolutional layers

Ni+j

follows. Finally, we use an output layer that takes the grid cell embeddings and predicts a distribution over the 5 possible costs.

We performed hyperparameter tuning on the validation set of the 12 × 12 grid size problem, then reused the same hyperparameters for the 30 × 30 grid size problem. We only reduced the number of RLOO samples S and the number of samples for the SNIS algorithm in Section F.1 for the 30 × 30 grid size problem to reduce the overhead of many calls to Dijkstra’s algorithm. This algorithm quickly becomes the main compute bottleneck on large grids.

For 12 × 12, we evaluated test accuracy at 40 epochs, and for 30 × 30 we evaluated validation accuracy every 5 epochs within the 40 epoch timeframe, choosing the best performing model for the test accuracy. We found that on 30 × 30 the model was sometimes unstable, suddenly dropping in accuracy and recovering after a while. As is common in this task and our baselines, we consider a path prediction correct if the predicted path has the same cost as the gold-truth shortest path given. This is because shortest paths may not be unique.

Baselines. We take the numbers for EXAL as reported in the paper [85]. For A-NeSI, I-MLE and A-NeSI + RLOO, we obtained performance on 10 individual runs from the authors to compute the Mann-Whitney U test.

###### G.4 RSBench

For all experiments, we adapt the implementation of the benchmark in the RSBench repository [11]. We use the conditional 1-step entropy discussed in Section 3.4. For the MNIST experiments, we brute-force the conditional entropy computation, while for BDD-OIA, we adapt the inference procedure in [11] to obtain the conditional entropy.

###### G.4.1 Metrics

For all tasks, we compute the Expected Calibration Error (ECE) over marginal concept probabilities [60] as a metric for calibration. Since NESYDM is not tractable, we have to estimate these marginals. Therefore, we use simple maximum-likelihood estimation to obtain approximate marginal probabilities for pθ(wi | x) by sampling L samples from the model and taking the empirical mean. We used L = 1000 throughout to improve the accuracy of the ECE estimate.

For the MNIST tasks, we report both the output accuracy Accy and concept accuracy Accc. In particular, for output accuracy, we compute exact match accuracy over the output predictions. For concept accuracy, we use micro-averaged accuracy over the concept predictions (that is, the two digits). This requires NESYDM to output predictions for the digits separately. We tried two different majority voting strategies using the L samples. 1) Take the dimension-wise mode among the samples, or 2) take the most common complete concept vector c and use the individual dimensions of c as the predictions. We used the second strategy in Table 3 to ensure the predictions can capture dependencies between digits, and compare the two methods in Section H.1 and Table 8.

For BDD-OIA, we report macro-averaged F1 scores for both the output and concept prediction. For example, for the concept F1 score, we compute the F1 score for each concept separately, then take the unweighted mean of these F1 scores. Similarly, we computed macro-averaged ECE scores for concept prediction. For NESYDM, we computed marginal probabilities for concept and output predictions, that is, per dimension. Furthermore, we recomputed all metrics for the baselines reported in Table 3, as we found bugs in the code for both metrics in the RSBench and BEARS codebases. Note that PNP⊥⊥ was called DPL in the BEARS paper. We changed the name as 1) the baseline code did not actually use the DeepProbLog language, and 2) there are many different NeSy predictors that could be implemented in DeepProbLog, so it is not clear which one to compare to.

###### G.4.2 Why Expected Calibration Error for reasoning shortcut awareness?

In this section, we motivate the use of concept calibration, in particular using the Expected Calibration Error (ECE), to empirically measure reasoning shortcut (RS) awareness. BEARS [55] introduced RS-awareness as attaining high accuracy on concepts unaffected by RSs, while being calibrated on concepts affected by RSs. In the latter case, perfect concept accuracy is unattainable, and we should aim for high calibration. A model that predicts concepts in such a way by mixing over RSs that cannot be disambiguated from data alone maximises data likelihood [39].

For example, consider the XOR problem from Example 2.2, which has 1 RS that maps MNIST digits of 1s to 0s and MNIST digits of 0s to 1s. This RS cannot be distinguished from the ground-truth mapping. The ideal model under this ambiguity would assign 50% confidence to the ground-truth mapping and 50% to the RS. We can achieve this with a neural network that given two distinct MNIST digits outputs a uniform distribution over (0, 1) and (1, 0), and given two equivalent MNIST digits, outputs a uniform distribution over (0, 0) and (1, 1). In the first case, the XOR function will return 1 with probability 1, and in the second case, it will return 0 with probability 1.

This attains maximum data likelihood as the model returns the correct output label. Furthermore, it always assigns 0.5 probability to 1 and 0. Its concept accuracy will also be around 0.5 in our synthetic setup. Therefore, the ECE will also be around 0, highlighting its calibration, and in practical experiments low ECE values are attainable [39]. Instead, a non RS-aware method finding the RS will attain an ECE of around 1: it always predicts exactly the opposite of the correct concept. Since the non RS-aware method randomly finds the RS or the correct solution, the ECE will be around 0.5 on average [39]. For concepts that are not affected by RSs, the accuracy will be 1 and maximum likelihood will also attain high confidence, resulting also in a low ECE value.

That said, ECE is a proxy for measuring concept calibration, but is not necessarily the only or perfect metric for uncertainty quantification. A further theoretical study evaluating how to best measure RS-awareness could be of significant value.

###### G.4.3 Hyperparameters

Table 6: Hyperparameters for RSBench.

Variable MNIST Half & Even-Odd BDD-OIA learning rate 0.00009 0.0001

minibatch size 16 256

epochs 500 30 γc 1.5 · 10−6 5 · 10−6 γH 1.6 2.0 γy 1 1

β 10 10

- S 1024 1024

- K 1024 1024

T 8 22

- L 1000 1000

For the datasets in RSBench, we tuned on the validation set for all parameters using the conditional entropy. Then, we ran an additional hyperparameter search on just the entropy weight to find the right trade-off between calibration and accuracy. We found the entropy weight can be sensitive, where high values significantly slow down training, while low values may result in uncalibrated models. See Table 10 and Section H.2 for an ablation study on the effect of the entropy weight. For L, we use a much higher number of 1000 samples. This is to ensure the Expected Calibration Error is properly estimated (see Section G.4.1 for details). For the runs with the unconditional entropy, we used the same hyperparameters and no additional hyperparameter search.

###### G.4.4 Experimental details and architectures

MNIST Half and MNIST Even-Odd. We adapted the original architecture from our baselines in [55]. For both experiments, we encode the two individual digits in x with a convolutional neural network (ReLU activations) of 3 layers, with 32, 64 and 128 channels respectively. Then, we flatten the output, obtaining two embeddings e1 and e2. For predicting the unmasking distribution p(˜c01 | x,ct) for the first digit, we concatenate one-hot encodings of ct1, ct2 and e1, while for predicting the distribution of the second digit p(˜c02 | x,ct), we concatenate one-hot encodings of ct2, ct1 and e2. Note the order here: This is to ensure permutation equivariance, as the sum is a commutative operation. Therefore, like our baselines, we have a disentangled architecture that uses the same neural network to classify the two digits, while still incorporating the currently unmasked values. Finally, using the concatenated vector, we use a linear output layer and a softmax to obtain the distribution over the possible digits.

BDD-OIA. We used early stopping by running for 30 epochs, testing on the validation set every epoch and picking the model with the highest validation accuracy. As in [55], we used preprocessed embeddings of the dashcam images from a Faster-RCNN [66]. This Faster-RCNN was pre-trained on MS-COCO and fine-tuned on BDD-100k. These are provided in the RSBench dataset, and were also used for BEARS. For the unmasking model p(c˜0 | x,ct), we adapted the MLP from [55], using a single hidden layer with a dimensionality of 512, by simply concatenating a one-hot encoding of ct ∈ {0,1,m}21 to the input embedding of x. Note that, since the concepts are binary, this one-hot encoding is a 3-dimensional vector, as it can be 0, 1, or the mask value m.

Baselines. We obtained results of the 5 individual runs used for each method in [55] and re-evaluated them to obtain 4 digits of precision for all reported results, as [55] only reported 2 digits of precision. Furthermore, we used these results to compute statistical significance tests. We have different results than reported in [55] for BDD-OIA as we found bugs in the code for the metrics.

##### H Further ablation experiments

###### H.1 Other majority voting strategies

As stated in the main text, computing the exact mode argmaxy0 pθNESYDM(y0 | x) is intractable in general, also for representations supporting tractable marginals [2, 84]. Throughout this paper, we

used the majority voting strategy described in Section 3.5. However, when observing the results on MNIST-Even-Odd, one might be puzzled by the relatively high performance on concept accuracy while the output accuracy is low. We hypothesised that this was due to our chosen majority voting strategy, and repeated the evaluation of the models using different strategies, which we describe here. All assume access to a set of samples c01,...,c0L ∼ pθ(c0 | x,c1 = m).

- • Program-then-true-mode (PTM): The strategy described in Eq. 8, and the main one used in this paper. We emphasise that this is the “correct” strategy according to the generative process of NeSy predictors.
- • Program-then-marginal-mode (PMM): Similar to above, we feed all sampled concepts into the program, but rather than taking the most likely output, we choose the most likely output dimension-wise:

yˆi = argmaxy

i

L

l=1

[φ(c 0l )i = yi] (64)

- • True-mode-then-program (TMP): Find the mode of the sampled concepts, then feed that into the program:

yˆ = φ argmaxc

L

l=1

[c 0l = c] (65)

- • Marginal-mode-then-program (MMP): Compute the dimension-wise mode of the concepts cˆi, combine them into a single concept cˆ, and feed that into the program:

cˆi = argmaxc

i

L

[c 0l,i = ci], yˆ = φ(cˆ) (66)

l=1

- Table 7: Output accuracy, both in- and out-of-distribution, for different majority voting strategies on the MNIST-Half and MNIST-Even-Odd datasets.

Strategy Half, ID Half, OOD Even-Odd, ID Even-Odd, OOD NESYDM, Conditional entropy

PTM 99.12± 0.18 28.45± 0.90 98.65± 0.31 0.02± 0.04 PMM 99.12± 0.18 28.44± 0.91 98.65± 0.31 0.02± 0.04

- TMP 98.87± 0.23 28.46± 0.91 97.94± 0.49 0.18± 0.14 MMP 60.16± 4.77 33.15± 1.40 25.14± 2.81 5.39± 0.45

NESYDM, Unconditional entropy

PTM 99.12± 0.10 10.95± 0.05 97.52± 0.44 0.00± 0.00 PMM 99.12± 0.10 10.95± 0.05 97.52± 0.44 0.00± 0.00

- TMP 99.26± 0.26 15.71± 0.49 98.10± 0.37 0.02± 0.02 MMP 79.42± 3.14 44.11± 4.87 87.64± 0.37 5.27± 0.52

We evaluated these strategies on the validation set of all benchmarks, and found that they all performed similar, or at most marginally worse than the PTM strategy used in this paper. However, we found exceptions in MNIST-Half and MNIST-Even-Odd, where MMP significantly outperforms the other strategies in the OOD setting, while significantly underperforming in the ID setting, as highlighted in Table 7. This result holds for both NESYDM with the conditional entropy and the unconditional entropy.

ID performance of MMP takes a rather significant hit because there are strong statistical dependencies between the concepts in the construction of the ID datasets. Especially the Even-Odd OOD dataset

is rather adversarially constructed, as highlighted by the extremely low OOD performance of all methods. However, because NESYDM has relatively high concept accuracy OOD, using MMP still results in some correct outputs.

We performed a similar analysis for the two strategies for predicting concepts in Table 8. Here we find that, overall, the true mode strategy usually performs better, except that we find a significant difference between TM and MM on the OOD dataset of MNIST-Half.

- Table 8: Concept accuracy, both in- and out-of-distribution, for different majority voting strategies on the MNIST-Half and MNIST-Even-Odd datasets.

Strategy Half, ID Half, OOD Even-Odd, ID Even-Odd, OOD NESYDM, Conditional entropy

TM 71.16± 1.77 61.84± 0.89 20.33± 1.33 15.60± 0.99 MM 66.78± 2.84 62.76± 0.77 19.65± 2.37 14.56± 0.86

NESYDM, Unconditional entropy

TM 79.41± 6.58 57.22± 0.49 0.36± 0.39 4.65± 0.49 MM 80.56± 5.12 70.40± 2.71 0.39± 0.44 1.16± 0.44

H.2 Effect of loss weighting hyperparameters

In this appendix, we investigate the effect of the loss weighting hyperparameters γc, γH and γy on the performance of NESYDM. We experimented with the conditional entropy version of NESYDM on the MNIST-Half dataset with three repeated runs. Here, we kept the output unmasking weight γy = 1 and tuned the other two hyperparameters.

For the concept unmasking weight γc in Table 9, we find that all tested values achieve high label accuracy. However, its value significantly influences the concept accuracy both in and out of distribution, and the OOD label accuracy. We observe values below 1 are effective. We suspect the concept unmasking loss can provide the model information that it is useful, but it should not dominate the loss, as this results in significantly lower concept accuracy and OOD performance. Furthermore, it results in poor calibration, suggesting that the model converged onto a single reasoning shortcut.

We observe a significant influence of the entropy weight γH in Table 10. All values below 1.3 seem to converge on a single reasoning shortcut, exhibiting poor calibration and worse concept accuracy. These runs do not balance the maximisation of entropy as the weight is too low, resulting in models with low entropy. By increasing the entropy weight beyond the value we used (1.6), we find that the entropy loss can also impact the performance when above 2.0, resulting in reduced label and concept accuracy and calibration. As the optimal range of values is quite tight, we recommend tuning the entropy weight when using NESYDM.

- Table 9: Effect of concept unmasking weight on label and concept accuracies (in- and out-ofdistribution) and calibration (ECE) performance on MNIST-Half. Bold values indicate best results per column. We used 1e-06 in the experiments.

γc Accy ↑ Accc ↑ Accy,OOD ↑ Accc,OOD ↑ ECEc,ID ↓ ECEc,OOD ↓

1e-08 99.38± 0.27 70.45± 2.32 33.92± 5.10 65.13± 2.72 8.54± 4.46 12.23± 1.18 1e-07 99.61± 0.13 71.41± 0.72 37.16± 1.22 67.74± 0.56 7.77± 1.09 10.67± 0.98 1e-06 99.54± 0.80 72.88± 0.85 33.21± 7.06 64.86± 3.92 5.31± 1.19 11.00± 0.87

1.5e-06 99.12± 0.10 71.16± 1.77 28.44± 0.90 62.76± 0.89 4.18± 2.56 11.74± 1.18

1e-05 99.00± 0.53 69.79± 3.09 29.72± 3.00 63.10± 2.11 6.33± 3.87 11.39± 1.76 0.0001 99.23± 0.13 72.07± 0.77 36.46± 6.60 66.84± 4.04 4.90± 0.69 10.22± 1.96

0.001 99.61± 0.13 72.03± 0.84 32.75± 6.51 64.60± 4.44 4.98± 3.15 10.46± 2.86 0.01 99.15± 0.48 71.60± 0.35 31.99± 7.80 63.57± 5.41 3.80± 1.52 10.81± 1.07

- 0.1 99.46± 0.35 71.88± 0.81 36.76± 8.27 66.85± 5.05 6.96± 2.35 12.35± 4.01

- 1.0 98.84± 0.61 41.55± 0.12 5.70± 0.24 38.65± 0.14 56.87± 0.25 61.09± 0.11 10.0 99.38± 0.13 41.59± 0.13 5.64± 0.19 38.59± 0.19 57.00± 0.23 60.97± 0.08

- Table 10: Effect of entropy weight on label and concept accuracies (in- and out-of-distribution) and calibration (ECE) performance on MNIST-Half. Bold values indicate best results per column. We used 1.6 in the experiments.

γH Accy ↑ Accc ↑ Accy,OOD ↑ Accc,OOD ↑ ECEc,ID ↓ ECEc,OOD ↓ 0.001 99.15± 0.13 41.63± 0.07 5.61± 0.16 38.63± 0.03 57.05± 0.12 61.08± 0.16

- 0.01 99.00± 0.35 41.74± 0.18 5.79± 0.18 38.66± 0.03 56.80± 0.21 60.91± 0.11

- 0.1 98.77± 0.48 41.67± 0.12 5.61± 0.32 38.60± 0.23 56.98± 0.26 60.97± 0.09

- 0.5 99.31± 0.23 41.63± 0.07 5.58± 0.14 38.59± 0.15 56.94± 0.09 61.09± 0.01

- 1.0 99.23± 0.13 41.63± 0.13 5.85± 0.14 38.73± 0.12 56.90± 0.25 61.06± 0.09

- 1.3 99.77± 0.23 67.05± 1.99 35.51± 2.78 65.79± 1.79 9.33± 1.74 12.26± 1.31

- 1.6 99.12± 0.10 71.16± 1.77 28.44± 0.90 62.76± 0.89 4.18± 2.56 11.74± 1.18

- 2.0 99.61± 0.13 72.26± 0.41 29.35± 1.73 62.19± 1.03 3.79± 0.48 11.42± 0.61

- 3.0 89.81± 5.84 46.53± 2.50 17.03± 8.97 51.23± 7.03 12.70± 5.87 14.56± 2.71 5.0 85.73± 1.94 39.74± 1.33 11.24± 0.14 44.04± 1.39 12.56± 1.33 24.22± 2.12

10.0 85.73± 0.35 34.22± 1.27 10.81± 0.19 41.25± 1.03 15.40± 2.25 24.01± 1.56

