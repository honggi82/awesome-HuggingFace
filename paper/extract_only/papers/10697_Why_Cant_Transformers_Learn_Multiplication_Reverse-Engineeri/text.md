# arXiv:2510.00184v1[cs.LG]30Sep2025

WHY CAN’T TRANSFORMERS LEARN MULTIPLICATION? REVERSE-ENGINEERING REVEALS LONGRANGE DEPENDENCY PITFALLS

Xiaoyan Bai∗1, Itamar Pres∗2, Yuntian Deng3, Chenhao Tan1, Stuart Shieber4, Fernanda Vi´egas4,5†, Martin Wattenberg4,5†, Andrew Lee4

1University of Chicago 2MIT 3University of Waterloo 4Harvard University 5Google DeepMind

ABSTRACT

Language models are increasingly capable, yet still fail at a seemingly simple task of multi-digit multiplication. In this work, we study why, by reverse-engineering a model that successfully learns multiplication via implicit chain-of-thought, and report three findings: (1) Evidence of long-range structure: Logit attributions and linear probes indicate that the model encodes the necessary long-range dependencies for multi-digit multiplication. (2) Mechanism: the model encodes long-range dependencies using attention to construct a directed acyclic graph to “cache” and “retrieve” pairwise partial products. (3) Geometry: the model implements partial products in attention heads by forming Minkowski sums between pairs of digits, and digits are represented using a Fourier basis, both of which are intuitive and efficient representations that the standard fine-tuning model lacks. With these insights, we revisit the learning dynamics of standard fine-tuning and find that the model converges to a local optimum that lacks the required long-range dependencies. We further validate this understanding by introducing an auxiliary loss that predicts the “running sum” via a linear regression probe, which provides an inductive bias that enables the model to successfully learn multi-digit multiplication. In summary, by reverse-engineering the mechanisms of an implicit chain-of-thought model we uncover a pitfall for learning long-range dependencies in Transformers and provide an example of how the correct inductive bias can address this issue.

1 INTRODUCTION

Large language models demonstrate striking capabilities across reasoning, planning, and tool use. Yet, they also fail on surprisingly simple algorithmic tasks (Nye et al., 2021; Lee et al., 2023). Why do Transformers excel at some tasks, but fail to learn others? One such example is multi-digit multiplication. Despite having billions of parameters, models like Llama-3.2 90B or GPT4 still fail at 4x4-digit multiplication (Gambardella et al., 2024),1 even when explicitly fine-tuned on the task (Yang et al., 2023). Why do Transformers fail to learn multiplication?

We study these questions by contrasting a standard fine-tuned model (SFT), which fails at multiplication, with a model trained with implicit chain-of-thought (ICoT) (Deng et al., 2024; 2023), which succeeds. ICoT works by providing explicit chain-of-thought tokens during training, but gradually removes them and thus forces the model to internalize intermediate steps in its latent states.

We partially reverse-engineer the ICoT model and uncover several insights. First, unlike the SFT model, the ICoT model learns the correct long-range structure needed for multi-digit multiplication. We provide evidence of this using logit attributions and linear regression probes. Mechanistically, the ICoT model encodes long-range dependencies by organizing its attention into a sparse, binarytree-like graph, which (i) selects the correct digit pairs to compute partial products and (ii) “caches”

∗Equal contribution. †Work done entirely at Harvard. Contact: andrewlee@g.harvard.edu, smallyan@uchicago.edu Code: https://github.com/ajyl/icot

1Note that some recent proprietary models that do solve multi-digit multiplication may rely on tool-use.

these intermediate computations into earlier tokens for later retrieval. Lastly, geometrically, attention heads realize partial products as Minkowski sums of digit embeddings, and represent digits with Fourier bases, yielding a pentagonal prism structure – both of which are intuitive and efficient representations that the SFT model lacks.

With these insights, we revisit the dynamics of standard fine-tuning: under gradient descent and an auto-regressive loss, the model never learns these long-range dependencies, and thus loss plateaus on the middle digits. To confirm our understanding, we introduce a simple fix by introducing an auxiliary loss that supervises the model to predict a “running partial sum” through a lightweight linear regression probe. This provides an inductive bias to learn the proper long-range dependencies, allowing it to achieve perfect accuracy, without any supervision from chain-of-thought.

In summary, by partially reverse-engineering a network that successfully implements multi-digit multiplication, we uncover how it implements long-range dependencies, a mechanism that the unsuccessful model lacks. Our work highlights a challenge for Transformers to learn long-range dependency using gradient descent and an auto-regressive loss. While we demonstrate a task-specific inductive bias to address this issue, we anticipate generic improvements to address this limitation.

- 2 EXPERIMENT SETUP, TRAINING ICOT, NOTATIONS

Task, Models. We are interested in understanding the difference in a model trained with standard fine-tuning and ICoT. From experiments, we find that the simplest multi-digit multiplication in which standard fine-tuning fails but ICoT works is 4×4 digit multiplications. Similarly, the smallest architecture in which ICoT works is a 2-layer model with 4 attention heads. Thus we carefully study a 2-layer 4-head ICoT model and a standard fine-tuned model trained on 4×4 multiplication.

Training Procedures. Our ICoT setup is the same as that Deng et al. (2024). Here we provide an informal overview of ICoT, with details in Appendix A.1. Namely, assume two operands a = (a3,a2,a1,a0),b = (b3,b2,b1,b0) and their product c = (c7...c0). Operands are written leastsignificant digit first, similar to other algorithmic setups (Deng et al., 2024; 2023; Lee et al., 2023).

For ICoT, the training data includes intermediate chain-of-thought (CoT) tokens qi that explicitly record the step-by-step calculations. As a simple illustration, consider 12×34. The tokens appearing between the two equal signs follow the same CoT format used in our 4×4-digit multiplication tasks:

12 ∗ 34 = 48

### + 360

### (408)

### = 408

12∗4

12∗30

running sum

At each training epoch, a fixed number of CoT tokens are removed from the left of the chain. Concretely, the training examples at each epoch may have the following form:

- (Epoch 1) a0a1a2a3 ∗ b0b1b2b3%%% q0 ...qi ...qj ...qk ...qτ #### c0 ...c7
- (Epoch 2) a0a1a2a3 ∗ b0b1b2b3%%% qi ...qj ...qk ...qτ #### c0 ...c7
- (Epoch 3) a0a1a2a3 ∗ b0b1b2b3%%% qj ...qk ...qτ #### c0 ...c7

... (Epoch N) a0a1a2a3 ∗ b0b1b2b3%%% #### c0 ...c7

where qi are CoT tokens and %,# are special delimiters.2 Note that after each epoch, the model sees a shorter chain by truncating some tokens, and that by the end, only the operands and final

answer remain. For comparison, standard fine-tuning only trains on the operands: a0a1a2a3 ∗ b0b1b2b3%%%#### c0 ...c7.

Interestingly, the ICoT model is able to achieve 100% accuracy on 4×4 digit multiplication, while standard fine-tuning only achieves less than 1% accuracy. Note that scaling does not help – scaling to a 12 layer 8 head model achieves the same < 1% accuracy, and Yang et al. (2023) show that fine-tuning a 2B model still plateaus at 95% accuracy.

For more details regarding training (data format, sample size, hyperparameters), see Appendix A.

2These delimiters have no special meaning beyond matching the setup of Deng et al. (2024).

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

+ + +

+ +

+

=

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

+ +

[Figure 19]

=

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

+

[Figure 24]

=

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

=

=

=

=

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

+

+

+

+

+

+

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

=

=

=

=

=

=

=

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

- Figure 1: Multiplication has long-range dependencies, which can be captured by an intermediate value cˆi, from which both the solution (ci) and carries (ri) can be derived from.

Notations. hℓt indicates the hidden states at layer ℓ timestep t. Timesteps for solution tokens ck,k = [0,...,7] are notated tc

. ATTℓh(·), MLPℓ(·) indicate the output of the attention heads or MLP blocks at layer ℓ, head index h. E,U ∈ RV ×d indicate (un)embedding weights.

k

- 3 COMPARING THE MECHANISMS OF ICOT VERSUS SFT

- 3.1 LONG-RANGE DEPENDENCIES IN MULTI-DIGIT MULTIPLICATION

Here we discuss how one might solve multi-digit multiplication, and the required long-range dependencies needed to solve multiplication.

One approach to compute each digit, ck, is as follows:

sk ≜

i+j=k

aibj,

sum of partial products

ck = (sk + rk−1) mod 10, rk =

sk + rk−1 10 carry

, r−1 = 0 (1)

Note that both ck and rk can be expressed with an intermediary term cˆk, which encapsulates both the relevant information from the partial products and the carry:

cˆk ≜ sk + rk−1, ck = cˆk (mod 10), rk =

c ˆk 10

(2)

Importantly, note the long-range dependencies needed for multi-digit multiplication. Specifically, we highlight two observations: (i) To determine ck, one must use all the partial products {aibj|i + j ≤ k}, since all of these terms contribute to ck. (ii) Knowing the intermediary term cˆk suffices to compute ck and to propagate necessary information for later digits. Thus we use cˆk as a probing signature (Section 3.2) at each timestep tc

k

to check if the model is utilizing all the necessary longrange information to predict the correct tokens ck. In the following sections, we demonstrate how the ICoT model satisfies such long-range dependency while the standard fine-tuning model does not.

- 3.2 EVIDENCE OF LONG-RANGE DEPENDENCIES IN ICOT

We first demonstrate two lines of evidence that the ICoT model satisfies long-range dependencies in multi-digit multiplication, while the standard fine-tuning model does not.

Logit Attributions. Note from Figure 1 that digits ai,bi can only affect ck terms where k ≥ i. Also note that at timestep tc

, the pairwise products {aibj|i + j = k} affect the final prediction ck the most. “Earlier” pairwise products {aibj|i + j ≤ k} can still affect ck, but with diminishing effects as i + j gets smaller.

k

SFT

ICoT

- a3

- a2

- a1

- a0

b3 b2 b1

- b0

[Figure 58]

[Figure 59]

[Figure 60]

20

15

logit

10

5

0

c0 c1 c2 c3 c4 c5 c6 c7

c0 c1 c2 c3 c4 c5 c6 c7

- Figure 2: Logit Attribution. We test for whether each model has correctly learned long-range dependencies by measuring how sensitive the logits of output digits ci are to each operand digit (i.e., ai,bj). This is done by measuring the change in ci’s logits when a single operand digit is perturbed.

0 200

−100

0

100

200

ˆPredictedC

Cˆ2 (MAE 93.69)

Perfect predictions

Predictions

0 200

−100

0

100

200

Cˆ3 (MAE 113.27)

0 200

0

100

200

Cˆ4 (MAE 74.47)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

−100 0 100

−100

0

100

Cˆ5 (MAE 79.40)

0 50 100

0

50

100

Cˆ6 (MAE 28.22)

0 100 200

True Cˆ2

0

50

100

150

200

ˆPredictedC

Cˆ2 (MAE 2.00)

0 200

True Cˆ3

0

100

200

Cˆ3 (MAE 1.89)

0 100 200

True Cˆ4

0

100

200

Cˆ4 (MAE 1.74)

0 100

True Cˆ5

0

50

100

150

Cˆ5 (MAE 0.97)

0 50

True Cˆ6

0

25

50

75

Cˆ6 (MAE 0.56)

SFT

ICoT

- Figure 3: Linear regression probing results for cˆ. We probe from the middle of the last Transformer block, after attention heads but before MLPs.

We directly test for these relationships in our ICoT and SFT models using logit attributions. Namely, given an input sample ORIG := a0a1a2a3 ∗ b0b1b2b3, we measure the logits of the model’s predictions for c0−7 : logitc

(ORIG). We then randomly swap out one of the operand digits at timestep t (e.g., a˜2) to construct a counterfactual input COUNTERt = a0a1a˜2a3 ∗ b0b1b2b3 and measure the change in logits: ∆t,k = logitc

k

(COUNTERt) Thus ∆t,k measures the effect that digit at timestep t has on the prediction of token ck.

(ORIG) − logitc

k

k

We use 1,000 samples for each (t,k) pair and show the results in Figure 2. Note that for SFT, the model does not see the correct dependencies between earlier tokens to middle tokens, while the ICoT model does, suggesting that the model has indeed learned the correct long-range dependencies.

Probing for ˆck. Note from Figure 1 and Equation 2 that the long-range dependencies can be captured by an intermediate term, cˆk. We test for whether cˆk information can be decoded from the hidden states of the models using linear regression probes. Namely, at each timestep tc

we predict for cˆk by training a single vector wk ∈ Rd such that wkh2t.mid

k

= cˆk using a MSE loss, where h2.mid is the hidden state at layer 2 after attention heads, before MLPs.

ck

Figure 3 reports the mean absolute error from probing for cˆk for middle and late digits, k = 2,...,6. Note that the accuracy from the ICoT model is much higher than that of SFT, further suggesting that the ICoT model has learned the correct long-range dependencies while SFT has not.

- 3.3 ENCODING LONG-RANGE DEPENDENCIES VIA ATTENTION TREES

How does the ICoT model compute long-range dependencies? Here we describe how the model’s attention patterns induce a shallow directed acyclic graph, akin to a binary expression tree, in order to encode long-range dependencies.

- a0b0:
- a0b1:
- a0b2:

Layer 1 Head 2

Layer 1 Head 1

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 61]

[Figure 62]

- a1b0:
- a1b1: a2b0:

a2b0 a0b2

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

a1b0

% % #

% % #

a0b1 a1b0

### c0 c1

### c0 c1

a0b2

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |

a0a1a2a3 * b0b1b2b3 % % ####c0c1

a0a1a2a3 * b0b1b2b3 % % ####c0c1

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 63]

[Figure 64]

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

a0 a1 a2 a3 * b0 b1 b2 b3 % % % # ###c0 c1

% % #

% % #

### c0 c1

### c0 c1

a0a1a2a3 * b0b1b2b3 % % ####c0c1

a0a1a2a3 * b0b1b2b3 % % ####c0c1

Layer 2 Head 3

Layer 2 Head 4

- Figure 4: Visualization of attention tree to compute c2. Left: Attention maps for selected heads show the first layer “cache” pairwise products (aibj) across earlier timesteps, from which the second layer reads from (Not all tree paths are shown). Right: A visualization of the attention tree. Each arc indicates tokens being attended to at specific timesteps. Colored arcs above and below the digits indicate attention patterns from the first and second layers respectively. Example: orange arc indicates that at timestep b3, the model attends to a0 and b1, from which the second layer reads from.

Namely, in the first layer, across all timesteps t > 5,3 each attention head only attends to a pair of digit tokens, {ai,bj} (Figure 4, left). This allows the model to produce the pairwise product aibj (see Section 4.1 for how attention heads represent pairwise products), but also allows the model to cache the product aibj in the hidden state of layer 1 at timestep t (i.e., h1t). Put differently, product pairs {aibj}i,j∈{0,...4} are “cached” in the first layer across different timesteps (h1t,t < tc

). At later timesteps t ≥ tc

k

, when the model predicts solution tokens ck, this allows the second layer attention heads to attend to a small set of previous cache sites, i.e., where the appropriate pairs of products aibj,i + j = k are stored from earlier timesteps.

k

Example: Figure 4 depicts the attention patterns when the model predicts c2, given input “a0...3 ∗ b0...3 = c0c1”. These attention maps are averaged from 1,000 samples from a held out test set. The necessary terms to compute c2 are a2b0,a1b1,a0b2, and cˆ1 (which in turn requires a1b0,a0b1,a0b0).

Attention heads ATT23, ATT24 each attend to positions (b0,b2,c1) and (b3,“#”,c0). Inspecting what was “cached” in the first layer at those timesteps reveals the necessary partial products to compute

c2. For example, at timestep b0, ATT11, ATT12 attend to a2,b0; at timestep b2 ATT11 attends to a1,b1 while ATT12 attends to a0,b2; at timestep c0 ATT11 attends to a1,b0, ATT12 attends to a0b1. Thus the model can derive partial products, a2b0,a1b1,a0b2,a1b0,a0b1 with its attention tree.4

While Figure 4 shows an example of the “attention tree” for predicting c2, one can similarly reconstruct the correct trees for all digits c0,...,c7 using the attention patterns for all digits in Figure 10.

In summary, for each output step ck, the ICoT model constructs a binary-tree-like graph, spread out across timesteps, to attend to the correct pairs of tokens, allowing it to compute partial products.

- 4 FEATURE GEOMETRY OF ICOT

In addition to the mechanisms seen in Section 3, we also study the geometry of features in ICoT.

- 3Note that only after timestep 5, both a and b tokens appear in the context.
- 4Note that there may be a couple of different ways that a0b0 is derived. One possibility is to re-use a0, b0

information that was fetched at various timesteps. Another possibility is when a0 is slightly attended to at ATT23 (difficult to see in our visuals). Note that a0b0 plays a relatively minor role in computing c2 compared to all other partial products.

(a) Attention Layer 1 Colored by b0

(b) Attention Layer 1 Head 3 Colored by b1

(c) Attention Layer 1 Head 3 Colored by a1

(d) Attention Layer 1 Head 3 Zoomed in: Points inside box Colored by a1

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

- Figure 5: 3D PCA of attention head outputs can form Minkowski sums, which in turn can form nested representations. Each color represents a different digit.

- 4.1 DIGIT-WISE MULTIPLICATIONS AS MINKOWSKI SUMS

Note from Section 3.3 that the attention patterns are sparse, often only attending to the two digits ai,bj being multiplied. In such a case, the outputs of the attention head form a Minkowski sum.

Namely, consider a single head ATT1(i,j) at the first layer, attending to two digits ai, bj. Let WO ∈ Rd×d

head

,WV ∈ Rd

head×d be the output and value weights of the attention head, E[ai] ∈ Rd the token embedding for token ai, and Ai := WOWV E[ai],Bj := WOWV E[bj],Ai,Bj ∈ Rd.

In such a case, when the model spends α% of its attention on digit ai, and thus attends to digit bj by (1 − α)%, the set of all possible values for the output of the attention head forms a Minkowski sum:

ATT1(i,j) = αAi + (1 − α)Bj + ϵ (3)

{ATT1(i,j)}i,j ⊆ (αA) ⊕ ((1 − α)B) ⊕ ϵ (4) (ignoring position embeddings). See Figure 5 (a) for a visualization.

Visually, 3D PCAs can reveal nested representations. Namely, we can observe clusters, each cluster corresponding to a feature (i.e., ai). These clusters form a “global” geometry. When zoomed in to each cluster, we observe additional clusters for a second feature (i.e., bj) that form a “local” geometry of the same shape as its global counterpart. See Figure 5 (b-d) for examples.

This observation can be explained by deconstructing the covariance of the attention output:

ΣATT = α2ΣA + (1 − α)2ΣB, (5)

where ΣA = Cov(Ai),ΣB = Cov(Bj). First, note that if we ignore positional encodings, ΣA and ΣB share the same eigenvectors, as they each depend on the same terms (E[·],WO,WV ), which are picked by PCA. Further note that fixing a value for ai leaves a local covariance, Σlocal|a

i

=

(1 − α)2ΣB, which again share the same eigenvectors with the global ΣATT term, leading to the same local geometry when projected onto.

- 4.2 EMBEDDING DIGITS ON A PENTAGONAL PRISM VIA FOURIER BASES

Similar to Kantamneni & Tegmark (2025), we find that our model encodes digits in Fourier space. Specifically, the model’s embeddings E, the final hidden layer hL, and even the weights of the last MLP can be well reconstructed from a small set of Fourier basis functions.

- Figure 6 shows a 3D PCA visualization of the final hidden layer at timestep tc

, for both the SFT and ICoT models. While the SFT hidden states do not reveal any obvious patterns, the ICoT hidden states reveal a striking pattern: the ten digits form vertices of a pentagonal prism.

2

This structure is naturally explained by Fourier modes. Consider the Fourier expansion

kn 10 , n = 0,...,9.

Cn ∗ e−2πi

PCA of Last Hidden State (SFT) PCA of Last Hidden State (ICoT)

[Figure 69]

[Figure 70]

|cos, sin(2π n / 5) (k=2)|
|---|

|Parity (±5) (k=5)|
|---|

|4|
|---|

|0|
|---|

|9|
|---|

|5|
|---|

|8|
|---|

|6|
|---|

|1|
|---|

|2|
|---|

|3|
|---|

|7|
|---|

- Figure 6: Digits embedded in a pentagonal prism, using Fourier bases. No obvious patterns in the SFT model, but the ICoT model encodes digits in a pentagonal prism using Fourier bases.

where Cn(̸= ck) is some constant per digit n. Following Kantamneni & Tegmark (2025), we take frequencies k ∈ {0,1,2,5}, yielding the real Fourier basis

Φ(n) = 1(n) cos 2π10n sin 2π10n cos 2πn5 sin 2πn5 p(n) (k=0) (k=1) (k=1) (k=2) (k=2) (k=5) ,

where 1(n) ≡ 1 (the DC component) and p(n) ≡ (−1)n (the Nyquist/parity vector). The sine terms for k = 0 and k = 5 vanish over n = 0,...,9 and are omitted.

The final hidden layer hL can be reconstructed via these six terms (see Appendix B), indicating that the final hidden state is encoded using Fourier bases.

- Revisiting Figure 6, the first principal component (PC1) aligns with the parity vector p(n), separating even from odd digits. Second and third principal components span the k = 2 Fourier pair

(cos,sin(2πn5 )), so the digits lie on two regular pentagons: one each for even and odd digits. The digits within each pentagon advance by n+4 (mod 10) (e.g., n = 0→4→8..., same for odd dig-

its), allowing a walk around the pentagon while staying within the even/odd set. Interestingly, taking (mod 5) on such a sequence yields decreasing steps of 1 (n (mod 5) = 0 → 4 → 3...). Lastly, the two pentagons are parallel and stacked along PC1, with corresponding vertices differing by ±5 (same phase, opposite parity). Together, these yield the pentagonal-prism geometry in Figure 6.

- 5 PITFALLS OF LEARNING: LACK OF LONG-RANGE DEPENDENCY

Given our insights, we revisit why Transformers fail at multiplication under standard fine-tuning. In particular, in Figure 7 (a), we inspect the gradient norms (top row) and losses (bottom row) per token ck over the course of training. There are a few observations to make.

First, note from the loss curves that the first two digits, c0,c1, followed by the last digit, c7, are learned first, as indicated by their immediate drop in loss to near zero. This aligns with the gradient norms observed for these tokens: within the first few steps, these tokens receive gradients, but their norms quickly drop to near zero once the loss for these tokens reach zero. Also note that the order in which tokens are learned according to gradient norms and losses is consistent.

The model then eventually learns to predict c2. However, middle digits, c3 to c6 are never learned. Despite only the middle digits receiving gradients (as they are the only sources of loss remaining), their losses plateau, suggesting that the model is stuck in a local optimum that lacks the long-range dependencies to properly learn the middle digits.

Note that scaling to a larger model does not address this issue, as the same pattern can be found in a 12 layer 8 head model: see Appendix C.

Grad Norm (Log Scale)

###### (a) Standard Fine Tuning

###### (b) Training with Auxiliary Loss

[Figure 71]

[Figure 72]

C0 C1 C2 C3 C4 C5 C6 C7

C0 C1 C2 C3 C4 C5 C6 C7

[Figure 73]

- 100

- 101

- 102

PositionCk

PositionCk

10−1

0e+00 1e+05 2e+05 3e+05 4e+05 5e+05 6e+05 Gradient Steps

0e+00 2e+05 4e+05 6e+05 8e+05 1e+06 Gradient Steps

2.5

2.5

Position c0 c1 c2 c3 c4 c5 c6 c7

2.0

2.0

Loss(PerPosition)ck

Loss(PerPosition)ck

1.5

1.5

1.0

1.0

0.5

0.5

0.0

0.0

0e+00 1e+05 2e+05 3e+05 4e+05 5e+05 6e+05 Gradient Steps

0e+00 2e+05 4e+05 6e+05 8e+05 1e+06 Gradient Steps

- Figure 7: Gradient norms and losses per token ck. While both methods learn digits c0,c1,c7 first, standard fine-tuning gets stuck in a local optimum without having learned the right long-range dependencies, while training with the auxiliary loss allows the model to learn the middle digits.

- 6 LEARNING MULTIPLICATION WITHOUT ICOT

To further validate our understanding of why Transformers fail to learn multiplication, we demonstrate an example of a simple fix to teach Transformers multiplications without needing ICoT.

In particular, we leverage the observation from Section 3.1 in that (i) multi-digit multiplication requires long-range dependencies between digit ck and pairwise products {aibj|i+j ≤ k}, and (ii) such dependency can be summarized by an intermediate value cˆk to produce ck.

Thus in order to guide the Transformer to learn long-range dependencies, we simply add an auxiliary loss term to predict cˆk at each timestep tc

. We attach an additional linear regression head wh ∈ Rd to the output of H(= 2) attention heads in the second layer. These regression heads are trained to predict the correct accumulated sum cˆk at each timestep tc

k

,ck ∈ [0,...7] with a MSE loss:

k

zih = wh⊤ATT2h(·) (6) Laux =

7

1 8

1 H h∈H

(zih − cˆi)2 (7) L = LLM + λLaux (8)

i=0

where LLM is the standard language modeling loss.

This introduces an inductive bias for the task, and allows our 2-layer model to correctly learn 4x4 multiplication with 99% accuracy. Again, note that a larger 12-layer model still fails at multiplication under standard fine-tuning.

- Revisiting Figure 7 (b) demonstrates a very different learning dynamic. We observe the model learn early and last digits (c0,c1,c7) and work inwards (c2,c3,c4,c6, and finally c5).

Limitation. Obviously the suggested inductive bias pertains specifically to our task. However, our experiments demonstrate the pitfall of Transformers that require long-range dependencies, and that it is possible to overcome such a pitfall with the correct inductive biases. We speculate that there are other generalizing inductive biases that can improve performance on tasks with long-range dependencies (Tay et al., 2020), and leave this for future work.

Layer 1 Head 0

Layer 1 Head 1

Layer 1 Head 2

Layer 2 Head 2

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

- a0a1a2a3*b0b1b2b3 %%####c0c1c2c3c4c5c6

- a0

- a1

- a2

- a3

*

- b0

- a0a1a2a3*b0b1b2b3 %%####c0c1c2c3c4c5c6

- a0

- a1

- a2

- a3

*

- b0

- a0a1a2a3*b0b1b2b3 %%####c0c1c2c3c4c5c6

- a0

- a1

- a2

- a3

*

- b0

- a0a1a2a3*b0b1b2b3 %%####c0c1c2c3c4c5c6

- a0

- a1

- a2

- a3

*

- b0

- b1

- b2

- b3

- b1

- b2

- b3

- b1

- b2

- b3

- b1

- b2

- b3

% % #

% % #

% % #

% % #

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |

- Figure 8: Attention pattern of model trained with auxiliary loss. This model similarly produces a “binary attention tree” (red boxes), but interestingly, we also see an attention head that attends to all necessary pairwise digits simultaneously (black box), producing a pattern akin to Figure 2.

6.1 DOES THE MODEL WITH AUXILIARY LOSS LEARN THE SAME MECHANISM AS ICOT?

A natural question that arises is whether ICoT and our inductive bias leads to the same mechanisms. Inspecting the attention patterns suggests that a similar (but not necessarily exact) mechanism is learned: see Figure 8. Namely, the model similarly forms an “attention tree” to sparsely attend to the correct pairs of digits for each ci in the first layer (red boxes). Interestingly, in the auxiliary-loss model we also observe an attention head (Layer 2 Head 2) that simultaneously attends to all the necessary digits, {ai≤k,bi≤k}, at each timestep tc

, forming a parallelogram-like attention pattern (black box) akin to the shape seen in Figure 2.

k

- 7 RELATED WORK

Studying Transformers with Arithmetic Tasks. A growing line of work study Transformers under controlled settings to better characterize their behavior (Allen-Zhu & Li, 2023a;b; Li et al., 2023; Nanda et al., 2023b; Park et al., 2024b;a). Often, arithmetics is a natural and popular domain (Lee et al., 2023; Ye et al., 2024; Nikankin et al., 2024), which has led to numerous insights. For instance, Nanda et al. (2023a) study how Transformers perform modular addition to explain grokking. Kantamneni & Tegmark (2025) find that large language models use trigonometry to do addition, encoding digits using Fourier bases, while Nikankin et al. (2024) suggest that they also rely on heuristics. Cai et al. (2025) study length generalization in Transformers using arithmetic tasks. Similarly, we study the limitations of Transformers by studying why it fails to learn multi-digit multiplication.

Process Supervision. Recent work trains models with process supervision, in which feedback is given not just on final correctness but on each intermediate reasoning step. For example, Uesato et al. (2022) demonstrate that process-supervision can yield less reasoning errors on GSM8K compared to outcome-only supervision. Similarly, Lightman et al. (2023) show that step-level human feedback on MATH leads to stronger reward models. More recently, Zhong et al. (2023)’s Math-Shepherd automates step-wise rewards via continuation-based verification, improving performance on both GSM8K and MATH. ICoT similarly plays the role of process supervision in latent space, by slowly removing chain-of-thought tokens during training such that the model internalizes the reasoning procedure. We thus use ICoT’s success on multiplication to study why Transformers fail.

- 8 CONCLUSION

In this work, we study why Transformers fail on a seemingly simple task of multi-digit multiplication. We answer this question by reverse-engineering a model trained with implicit chain-of-thought, and uncover that it has learned to compute the correct long-range dependencies needed for multidigit multiplication. Our findings point to a pitfall of the standard recipe for training language models: using gradient descent with an auto-regressive loss on Transformers does not encourage the model to learn the right long-range dependencies. While we provide a simple example of how the right inductive bias can address such a limitation, we anticipate future work to provide a generic solution to improve on tasks with long-range dependencies.

ACKNOWLEDGMENTS

XB, CT acknowledge the support of NSF grants IIS-2126602. YD is supported by an NSERC Discovery Grant (RGPIN-2024-05178) and a Starter Grant from the University of Waterloo. AL, MW, and FV acknowledge support from the Superalignment Fast Grant from OpenAI, Effective Ventures Foundation, Effektiv Spenden Schweiz, and the Open Philanthropy Project. IP thanks Keya Hu for constructive feedback and AL thanks Thomas Fel for fruitful discussions regarding feature geometry.

REPRODUCIBILITY STATEMENT

Our code to reproduce all of our experiments can be found in https://github.com/ajyl/ icot . Appendix A provides details of our training setup, including data formats, sample size, and hyperparameters.

REFERENCES

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 1, context-free grammar. arXiv e-prints, pp. arXiv–2305, 2023a.

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. arXiv preprint arXiv:2309.14316, 2023b.

Ziyang Cai, Nayoung Lee, Avi Schwarzschild, Samet Oymak, and Dimitris Papailiopoulos. Extrapolation by association: Length generalization transfer in transformers. arXiv preprint arXiv:2506.09251, 2025.

Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. arXiv preprint arXiv:2311.01460, 2023.

Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit cot to implicit cot: Learning to internalize cot step by step, 2024. URL https://arxiv.org/abs/2405.14838.

Andrew Gambardella, Yusuke Iwasawa, and Yutaka Matsuo. Language models do hard arithmetic tasks easily and hardly do easy arithmetic tasks. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 85–91, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-short.8. URL https: //aclanthology.org/2024.acl-short.8/.

Subhash Kantamneni and Max Tegmark. Language models use trigonometry to do addition. arXiv preprint arXiv:2502.00873, 2025.

Nayoung Lee, Kartik Sreenivasan, Jason D Lee, Kangwook Lee, and Dimitris Papailiopoulos. Teaching arithmetic to small transformers. arXiv preprint arXiv:2307.03381, 2023.

Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. ICLR, 2023.

Alex Lightman, Yuntao Bai, Saurav Kadavath, Tamera Lanham, Nicholas Schiefer, et al. Let’s verify step by step, 2023. URL https://arxiv.org/abs/2305.20050.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217, 2023a.

Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. arXiv preprint arXiv:2309.00941, 2023b.

Yaniv Nikankin, Anja Reusch, Aaron Mueller, and Yonatan Belinkov. Arithmetic without algorithms: Language models solve math with a bag of heuristics. arXiv preprint arXiv:2410.21272, 2024.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. 2021.

Core Francisco Park, Andrew Lee, Ekdeep Singh Lubana, Yongyi Yang, Maya Okawa, Kento Nishi, Martin Wattenberg, and Hidenori Tanaka. Iclr: In-context learning of representations. arXiv preprint arXiv:2501.00070, 2024a.

Core Francisco Park, Ekdeep Singh Lubana, Itamar Pres, and Hidenori Tanaka. Competition dynamics shape algorithmic phases of in-context learning. arXiv preprint arXiv:2412.01003, 2024b.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. arXiv preprint arXiv:2011.04006, 2020.

Jonathan Uesato, Po-Sen Huang, Tim Rockt¨aschel, Pushmeet Kohli, et al. Solving math word problems with process- and outcome-based feedback, 2022. URL https://arxiv.org/ abs/2211.14275.

Zhen Yang, Ming Ding, Qingsong Lv, Zhihuan Jiang, Zehai He, Yuyi Guo, Jinfeng Bai, and Jie Tang. Gpt can solve mathematical problems without a calculator. arXiv preprint arXiv:2309.03241, 2023.

Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of language models: Part 2.1, grade-school math and the hidden reasoning process. arXiv preprint arXiv:2407.20311, 2024.

Zexue Zhong, Zihan Zhao, Yutao Sun, et al. Math-shepherd: Process supervision for large language models, 2023. URL https://arxiv.org/abs/2312.08935.

- A TRAINING DETAILS Here we provide additional details regarding the training procedures of each of our models.

- A.1 ICOT TRAINING

Our ICoT training setup follows the practice outlined in the original ICoT paper (Deng et al., 2024). ICoT works by initially presenting explicit chain-of-thought tokens during training, but gradually removing them across numerous “stages” (e.g., epochs). Concretely, the training examples at each epoch may have the following form:

- (Epoch 1) a0a1a2a3 ∗ b0b1b2b3%%% q0 ...qi ...qj ...qk ...qτ #### c0 ...c7
- (Epoch 2) a0a1a2a3 ∗ b0b1b2b3%%% qi ...qj ...qk ...qτ #### c0 ...c7
- (Epoch 3) a0a1a2a3 ∗ b0b1b2b3%%% qj ...qk ...qτ #### c0 ...c7

... (Epoch N) a0a1a2a3 ∗ b0b1b2b3%%% #### c0 ...c7

where qi are CoT tokens and %,# are special delimiters. These delimiters have no special meaning beyond matching the setup of Deng et al. (2024). Note that after each epoch, the model sees a shorter chain by truncating some tokens, and that by the end, only the operands and final answer remain.

The actual format of our ICoT data is as follows. Using an example input of 8331×5015, digits are presented in least-significant digits first, resulting in the following format:

### 1338 ∗ 5105||5614 + 013380(569421) + 0000000(5694210) + 0005561%%####56997714

Unlike Deng et al. (2024), instead of using a pre-trained 12-layer GPT model, we train a smaller 2-layer, 4-head GPT-based model from scratch, not only to remove any confounding factors from pre-trained knowledge, but also because the 2-layer 4-head architecture is the simplest form in which ICoT succeeds but standard fine-tuning fails. The training data consists of 80,800 samples, while the validation and test sets each contain 1,000 held out samples. We train with a learning rate of 5e-5, and remove 8 chain-of-thought tokens at every “stage” (which in our case is an epoch). Both training and validation loss converge after 13 epochs, and achieves 100% accuracy on the test set.

- A.2 STANDARD FINE-TUNING

Similar to ICoT, for our standard fine-tuning model, we train a 2-layer, 4-head GPT-based model from scratch, on the same data as ICoT. We use a learning rate of 5e-5, and the input format is a0a1a2a3 ∗ b0b1b2b3%%####c0 ...c7. All other hyperparameters match those in our ICoT setup. The model’s loss and accuracy plateaus after 13 epochs, it achieves only about 1% train and validation accuracy, while digit-level accuracy converges at approximately 81%, and remains the same even after 60 epochs.

Note that scaling the model larger to a 12-layer, 8-head model achieves the same low accuracy at 1% and digit-level accuracy of 80%.

- B FOURIER STRUCTURE IN MODEL’S WEIGHTS, ACTIVATIONS

Here we provide a deeper dive into the Fourier structure found in the ICoT model’s weights and hidden states. Namely, we analyze the model’s embedding weights, final MLP’s weights, and last hidden layer:

- 1. Embeddings E ∈ R10×d
- 2. Final layer MLP output weights Wout ∈ Rd

mlp×d, given MLP(x) = σ(Winx)Wout

- 3. Final hidden layer hLt ∈ RN×d

where N(= 1,000) is the size of our validation set. For the latter two, we first project them onto the model’s embedding space:

Wout = (EWout)⊤ ∈ Rd

mlp×10 hL = (EhL)⊤ ∈ RN×10

Each item X ∈ {E, Wout, hL} is a collection of row vectors x ∈ R10 whose ten entries correspond to digits n = 0,...,9.

We find that vectors x are encoded in a low-dimensional trigonometric subspace. Namely, consider the Fourier expansion

kn 10 , n = 0,...,9.

Cn ∗ e−2πi

where Cn(̸= ck) is some constant per n. Following Kantamneni & Tegmark (2025), we take frequencies k ∈ {0,1,2,5}, yielding the real Fourier basis

Φ(n) = 1(n) cos 2π10n sin 2π10n cos 2πn5 sin 2πn5 p(n) (k=0) (k=1) (k=1) (k=2) (k=2) (k=5) ,

where 1(n) ≡ 1 (the DC component) and p(n) ≡ (−1)n (the Nyquist/parity vector). The sine terms for k = 0 and k = 5 vanish over n = 0,...,9 and are omitted.

Let F ∈ 10 × 6 be a Fourier matrix with rows indexed by n ∈ {0,...,9} and columns as defined above.

For each row x ∈ R10 we fit least squares coefficients

∥x − FC∥22 and quantify goodness-of-fit using coefficient of determination

C = arg min

C∈R6

∥x − FC∥22 ∥x − x¯∥22

R2(x) = 1 −

,

We report the median R2 over the set of rows in each X (i.e., over dmlp rows for Wout, d rows for E, and over batch examples for hL.

In Table 1 we observe strong fits: the per-row medians lie between 0.85 and 0.99, indicating that a six-dimensional trigonometric basis over digits captures the vast majority of variance:

We can extend the Fourier bases to include additional terms, for k = 3,4, which forms a 8 dimensional basis (excluding sine terms for k = 0,5), which leads to perfect R2 fits.

Table 1: Median R2 of Fourier fits over digits (n = 0...9).

Object Fourier Basis Rows aggregated Median R2 E k = 0,1,2,5 dmodel 0.84

MLP Wout weights k = 0,1,2,5 dmlp 0.95

hL k = 0,1,2,5 batch examples 0.99 E k = 0,1,2,3,4,5 dmodel 1 MLP Wout weights k = 0,1,2,3,4,5 dmlp 1

hL k = 0,1,2,3,4,5 batch examples 1

Grad Norm (Log Scale)

##### (a) Standard Fine Tuning

[Figure 78]

C0 C1 C2 C3 C4 C5 C6 C7

[Figure 79]

- 100

- 101

- 102

PositionCk

10−1

0e+00 2e+04 5e+04 8e+04 1e+05 1e+05 2e+05 2e+05 2e+05 Gradient Steps

2.5

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

Position c0 c1 c2 c3 c4 c5 c6 c7

2.0

Loss(PerPosition)ck

1.5

1.0

0.5

0.0

0e+00 2e+04 5e+04 8e+04 1e+05 1e+05 2e+05 2e+05 2e+05 Gradient Steps

## Figure 9: Gradients and loss per token for a 12-layer model.

- C PER TOKEN GRADIENTS AND LOSSES: 12-LAYER MODEL

Even with a larger 12 layer model, the model fails to learn the right long-range dependencies. Figure 9 displays the results – we see the similar patterns as the 2-layer model, in which middle digits never receive the right gradients and loss does not drop.

- D ATTENTION PATTERNS OF ALL MODELS

In Section 3.3, we illustrate how a binary tree is constructed for c2 in the ICoT model. In Figure 10, we present the attention patterns for all digits across the three models, from with attention trees can

be derived for the ICoT model for each solution token ci.

- E LLM USAGE We used LLMs to proof read our draft and polish our notations.

## ICoT

Layer 1 Head 0

Layer 1 Head 1

Layer 1 Head 2

Layer 1 Head 3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

*

*

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

% % #

% % #

% % #

% % #

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

Layer 2 Head 0

Layer 2 Head 1

Layer 2 Head 2

Layer 2 Head 3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

*

*

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

% % #

% % #

% % #

% % #

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

## SFT

Layer 1 Head 0

Layer 1 Head 1

Layer 1 Head 2

Layer 1 Head 3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

*

*

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

% % #

% % #

% % #

% % #

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

Layer 2 Head 0

Layer 2 Head 1

Layer 2 Head 2

Layer 2 Head 3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

*

*

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

% % #

% % #

% % #

% % #

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

## Auxiliary Loss Model

Layer 1 Head 0

Layer 1 Head 1

Layer 1 Head 2

Layer 1 Head 3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

*

*

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

% % #

% % #

% % #

% % #

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

Layer 2 Head 0

Layer 2 Head 1

Layer 2 Head 2

Layer 2 Head 3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

- a0

- a1

- a2

- a3

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

*

*

*

*

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

- b0

- b1

- b2

- b3

% % #

% % #

% % #

% % #

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

### c0 c1 c2 c3 c4 c5 c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

a0a1a2a3 * b0b1b2b3 %% ####c0c1c2c3c4c5c6

Figure 10: Attention patterns of ICoT, standard fine-tuned model, and the model trained with auxiliary loss

