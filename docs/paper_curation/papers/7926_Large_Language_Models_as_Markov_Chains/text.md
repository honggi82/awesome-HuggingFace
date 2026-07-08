#### Large Language Models as Markov Chains

Oussama Zekri*1 Ambroise Odonnat*23 Abdelhakim Benechehab24 Linus Bleistein3 Nicolas Boull´e5 Ievgen Redko2

# arXiv:2410.02724v2[stat.ML]2Feb2025

##### Abstract

Large language models (LLMs) are remarkably efficient across a wide range of natural language processing tasks and well beyond them. However, a comprehensive theoretical analysis of the LLMs’ generalization capabilities remains elusive. In our paper, we approach this task by drawing an equivalence between autoregressive transformer-based language models and Markov chains defined on a finite state space. This allows us to study the multi-step inference mechanism of LLMs from first principles. We relate the obtained results to the pathological behavior observed with LLMs such as repetitions and incoherent replies with high temperature. Finally, we leverage the proposed formalization to derive pre-training and incontext learning generalization bounds for LLMs under realistic data and model assumptions. Experiments with the most recent Llama and Gemma herds of models show that our theory correctly captures their behavior in practice.

##### 1. Introduction

Large language models (LLMs) (Brown et al., 2020; Touvron et al., 2023a) have reshaped the landscape of machine learning and artificial intelligence. Trained on vast amounts of data, they have reached unprecedented performance in tasks such as machine translation (Brown et al., 2020), text generation, question answering (Roberts et al., 2020), and sentiment analysis (Zhang et al., 2023a), to name a few. Pre-trained LLMs also exhibit an intriguing capability of performing learning from the context, also called in-context learning (ICL), without explicitly updating their parameters.

Despite all these remarkable achievements, the theoretical justification for LLMs’ impressive performance remains elu-

*Equal contribution 1ENS Paris-Saclay 2Huawei Noah’s Ark Lab 3Inria 4Eurecom 5Imperial College London. Correspondence to: OZ <oussama.zekri@ens-paris-saclay.fr>, AO <ambroiseodonnattechnologie@gmail.com>.

Preprint. Copyright 2025 by the author(s).

Llama 7B

Llama3.2 3B

Gemma 2B Gemma 7B

Gemma2 9B

Llama2 7B

Llama3 8B

Gemma2 27B

78

MMLUPerformance(5-shot)

66

54

42

30

0.7 1.2 1.7 2.2

Approximation Error (x 1e-2)

Figure 1: LLMs’ Sample Complexity. We plot the Massive Multitask Language Understanding (MMLU) (Hendrycks et al., 2021) performance with respect to the approximation error ϵ predicted by Proposition 4.1. We set N∗ equal to the real number of pre-training tokens. Each point represents a model from the Llama or Gemma families (Dubey et al., 2024; Riviere et al., 2024). The approximation error ϵ predicted by our theory correlates with the real performance, with different trends between the models’ families.

sive. The two main reasons for this are 1) the complexity of analyzing modern deep transfer-based architectures used to train LLMs and 2) the non-iid nature of their pre-training data. Prior works usually bypass these challenges by imposing simplifying assumptions. Such assumptions include considering a shallow single-head transformer (Makkuva et al., 2024), only self-attention mechanism (Ildiz et al., 2024) or attention-only transformers (Edelman et al., 2024; Jeon et al., 2024) when studying LLMs. The others are to study the auto-regressive mechanism of the LLMs without taking into account their architecture explicitly (Lotfi et al., 2023; 2024; Xie et al., 2022). When the architecture is studied in full generality (Li et al., 2023; Zhang et al., 2023b), it is also common to relax the non-iid assumption by a certain predefined Markovian process.

This work provides a unified theoretical analysis of LLM’s inference, pre-training, and ICL under realistic assumptions on model architecture and pre-training data. Our core idea

##### 2. Background Knowledge

###### Method Pre-Train. ICL Input Model-Dep. Exp. val.

(Xie et al., 2022) × ✓ HMM × ✓ (Zhang et al., 2023b) ✓ ✓ MC ✓ × (Li et al., 2023) ✓ ✓ MC ✓ ✓ (Lotfi et al., 2024) ✓ × non-iid × ✓

We recall some facts about Markov chains (Paulin, 2015; Roberts & Rosenthal, 2004) and LLMs. More notations and background materials are available in Appendices A to C.

Ours ✓ ✓ non-iid ✓ ✓

Markov chains. Let Ω be a discrete finite set of size |Ω|. A discrete-time, time-homogeneous Markov chain

Table 1: LLMs generalization bounds proposed in the literature. Pre-train. stands for pre-training; Input refers to the assumptions on the data generating process (Markov Chains (MC); Hidden Markov Model (HMM); Model-dep. means explicit dependence on model’s architecture; Exp.val. stands for the experimental validation of the theory.

MC(Ω,Q) defined on a state space Ω = {xi}|iΩ=1| with transition matrix Q ∈ R|Ω|×|Ω| with entries Qij = Q(xi,xj) ∈ [0,1] is a sequence of random variables (X1,X2,...) taking values in Ω such that for any n ∈ N and (x1,...,xn+1) ∈ Ωn+1, we have

is to note that despite the seemingly infinite generation capacity of LLMs, their vocabulary and context window are finite, making all possible input and output sequences countable. This leads to an intuitive, yet overlooked, approach that interprets LLMs as Markov chains whose transition matrix operates on a finite state space of such sequences (see Fig. 2). The generalization of LLMs is then their capacity to learn the transition probabilities of language itself.

Main contributions. Our contributions can be summarized

- as follows.

- 1) We characterize the inference mechanism of any LLM as a finite-state Markov chain. This provides a deeper understanding of LLMs’ generation behavior and sheds new light on some of their pathological behaviors.
- 2) We prove sample complexity bounds for deep transformer-based LLMs pre-trained on non-iid data. They are more general than existing results (see Table 1 for a full comparison with the literature) and are verified in practice on the Llama and Gemma herd of models from 2B to 27B (see Fig. 1 and Section 4).
- 3) We derive ICL generalization bounds when LLMs are prompted with Markov chains. We validate our results by showing that the most recent open-source LLMs obey the in-context scaling laws predicted by our theory.

Organization of the paper. Section 2 provides background material on transformer-based autoregressive LLMs and Markov chains. In Section 3.1, we formalize the multistep inference mechanism of a pre-trained LLM as a Markov chain. In Section 4, we study how a deep transformer-based LLM learns this inference mechanism through pre-training on non-iid data using the formalism of Marton couplings and also extend the obtained results to the ICL setting. We verify them through a set of experiments in Section 5.

P(Xn+1 = xn+1 | Xn = xn,...,X1 = x1) = P(Xn+1 = xn+1 | Xn = xn) := Q(xn,xn+1).

A distribution π on Ω is called stationary if πQ = π. For any x ∈ Ω, MC(Ω,Q) converges to π if limn→∞ dTV(Qn(x,·),π) = 0, where Qn(x,·) denotes the probability of Xn conditioned on X1 = x and the total variation between two distributions P and Q, defined on (Ω,F), is dTV(P,Q) := supA∈F|P(A) − Q(A)|. We recall that the mixing time tmix(ε) of a Markov chain is the minimal time needed to be ε-close to its stationary distribution (see Definition C.8). Intuitively, a Markov chain mixes slowly when it remains close to the initial state after a given number of steps and doesn’t explore its state space. A Markov chain with rapid mixing quickly forgets its initial state and transitions more easily to a wider set of states.

Large language models. Let V denote a dictionary of size T, referred to as the vocabulary size, used to encode an arbitrary sequence into a sequence of predefined tokens belonging to V. We assume that our model admits a maximum of K tokens as input, referred to as the context window of the model. The domain of the transformer-based autoregressive LLM is the set of all sequences consisting of at most K elements of V. We denote this space by VK∗ , i.e., VK∗ := {v ∈ V∗, |v| ≤ K} with |v| the length of v. We define an LLM with trainable parameters Θ as a function fΘT,K : VK∗ → ∆(V), where ∆(V) is the probability simplex over V. Given a sequence of tokens v, fΘT,K outputs a probability distribution over the whole state space indicating the likelihood for each of its elements to appear after v (see Appendix B for more details). Noting that an LLM is defined for a given vocabulary size T and context window K, we will drop the exponents and simply write fΘ to ease the notations. We consider a setting where the learner’s objective is to approximate the probability distribution of sequences over an input vocabulary PL : P(VK∗ ) → [0,1] where P(VK∗ ) denotes the powerset of VK∗ . We also assume the existence of a constant c0 > 0 such that for any n ∈ [N]

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

| |
|---|

Transient class Recurrent class Zero

Figure 2: LLM as a Markov chain. A large language model with vocabulary size T and context window K is equivalent to a Markov chain with a sparse and block-structured transition matrix of size i≤K Ti ∼ O(TK). The latter captures all possible outputs of a given LLM for all possible input sequences allowed by its vocabulary and context window.

and (x1,...,xn+1) ∈ Ωn+1,

PL(Xn+1 = xn+1 | Xn = xn,...,X1 = x1) ≥ c0. (1)

This is a common assumption used in (Wies et al., 2024; Xie et al., 2022; Zhang et al., 2023b).

Transformer model. Without loss of generality, fΘ is assumed to be a transformer model with L layers and H heads, consisting of alternating multi-head attention (MHA) and feed-forward blocks (more details in Appendix B). The first layer receives an input S(0) = S embedded in an rdimensional space. To obtain a probability distribution on the vocabulary V, the output S(L) ∈ Rr×T of the final layer is projected back to the vocabulary size by an “unembedding layer” WU ∈ RT×r and averaged over the columns to obtain a vector in RT. A softmax layer is finally applied to obtain the probability distribution of the next token PΘ(· | S) := softmax nτ 1 WUS(L) n ∈ ∆T, where Θ denotes the parameters of the entire network and τ is the softmax temperature (Hinton, 2015). Unless otherwise specified, we assume that the unembedding layer is bounded. The classes of parameters and neural networks it generates respectively write W = {Θ s.t. ∥WU⊤∥2,1 ≤ BU} and F = {fΘ s.t. Θ ∈ W}.

##### 3. Large Language Models as Markov Chains

To proceed, we formally define a Markov chain that explicitly captures the inference of a given LLM fΘ. We build upon a high-level idea that associates a tokenized input sequence with a state vi, from which we transition to a new state vj = [vi,v] by concatenating the token v predicted by an LLM to it. We then provide a theoretical characterization of this Markov chain. Our analysis holds for any model with finite fixed vocabulary, and context window. This setup includes deep transformer-based LLMs while excluding models such as RNNs and LSTMs.

###### 3.1. Markov Chain Formalization

We first introduce the notion of “incompatible sequences” in the sense of next-token prediction.

Definition 3.1 (Incompatible Sequences). Let u,v ∈ VK∗ be two sequences of at most K tokens. We say that v is incompatible with u if v cannot be a plausible completion of u, that is

∃l ∈ {1,...,|u| − 1},s.t. (u)l+1 ̸= (v)l. (2)

We denote by I the set of incompatible sequences, i.e., I := {(u,v) s.t. v is incompatible with u}.

The order matters since a necessary condition for v to be compatible with u is that v has as many or one more token than u. We now proceed with the definition of the transition matrix associated with an LLM fΘ.

Proposition 3.2. Any large language model fΘ can be equivalently represented by a Markov chain

MC(VK∗ ,Qf), with a sparse transition matrix Qf ∈ R|V

K|×|VK∗ | that writes for any vi,vj ∈ VK∗ :

∗

0, if (vi,vj) ∈ I {fΘ(vi)}j0

Qf(vi,vj) =

, otherwise,

where j0 denotes the index in V of the last token of vj. We have |VK∗ | = T(TK − 1)/(T − 1) and the proportion of non-zero elements in Qf is (T − 1)/(TK − 1).

Proposition 3.2 states that if vj is incompatible with vi, then the probability to go from the state vi to vj is zero as LLM only transitions between compatible sequences. For such compatible sequences, the probability of going from vi to vj is naturally the next-token probability associated with the last token of vj, i.e., {fΘ(vi)}j0

.

Example: binary sequences. To help unpack Proposition 3.2, we illustrate it with T = 2 and K = 3 in Fig. 3 (see the general case in Fig. 2). Here, we start with an input prompt consisting of one token. We then transition to sequences of increased size, while attributing a 0 probability to incompatible sequences. Each generation step for different values of k < K defines green rectangular blocks of size Tk × Tk+1 in the transition matrix Qf.

limn→∞ Qnf = eπ, where π is the stationary distribution of the recurrent class R of states, expanded by 0’s

for each transient state of the unichain. Moreover, for all n ≥ K,

n

|(Qnf)i,j − (eπ)i,j| ≤ (1 − 2ε)⌊

K⌋−1, where ε = min

###### {(QKf )i,j} > 0.

i,j∈R2

Next state distribution

Initial distribution

The stationary distribution is the long-term equilibrium of the multi-step inference of an LLM. While its true value is intractable because of the size of VK∗ , we discuss below the implications of Proposition 3.4 for LLMs.

[Figure 4]

[Figure 5]

- 1. Looping. The stationary distribution is independent of the initial state (i.e., input prompt), and reaching it should send the LLM into a deterministic loop of repetitions. This behavior is well known and occurs frequently in practice with many existing LLMs (Ivgi et al., 2024), requiring adding a repetition penalty.
- 2. Temperature and coherence. Increasing temperature of the model leads to a decreasing coherence of its outputs as measured by perplexity (Peeperkorn et al., 2024). Temperature, in turn, impacts ε (the smallest element of the Kth power of the transition matrix), as it modifies the probabilities of the predicted next tokens. Consequently, increasing the temperature increases the speed of convergence to the stationary distribution making the output less coherent. We illustrate this on a toy example below.
- 3.2. Illustration on a Toy Model

### * =

Input prompt

[Figure 6]

Figure 3: Proposition 3.2 with T = 2 and K = 3.

Reaching the blue square block in the transition matrix means that the input sequence is of the maximum size K. The model can no longer append tokens to it and has to delete the first token to proceed. This blue block is of size TK × TK: it captures transitions between sequences of the maximum admissible length. We define similarly the reference transition matrix Q∗ of the language where the probability of transitions {fΘ(vi)}j are replaced by ground-truth probabilities PL(vj | vi). To use Qf as fΘ, it is sufficient to define an input distribution δ0 of the Markov chain based on input prompt v. It is a one-hot encoding vector of size |VK∗ | with 1 at the position of the state corresponding to v. Then, the transition to the next state writes δ1 = δ0Qf. The output of fΘ(v) for individual tokens in V would then correspond to probabilities in δ1 for states that are concatenations of v with T tokens from V. This process is illustrated in Fig. 3.

We illustrate the results of Section 3 on a toy model trained on a sequence of 0s and 1s. Here, each subsequent token is 0 if the sum of three previous tokens is even and 1 otherwise. Therefore, T = 2 and K = 3. We generate a sequence of 40 digits, resulting in 37 distinct supervised examples, and train a small “GPT-like” model (Karpathy, 2023) on it. We extract the logits from the model by prompting it with all possible combinations of 0s and 1s of length less than three to obtain the transition matrix Qf ∈ R14×14 depicted in Fig. 4(a). The transition matrix’s structure (e.g., presence of transient and recurrent classes) matches the one presented in Fig. 2. Fig. 4(b) displays the stationary distribution of the trained model obtained by raising Qf to power 105. We note that it has a strong bias toward seen training samples in accordance with our intuition behind the stationary distribution presented earlier. Finally, Fig. 4(c) illustrates the convergence rate of the toy model, predicted by Proposition 3.4, and compares it to models with larger dictionary size T and context window K. In Fig. 4(c), we set ε = 10−6 and note that this parameter reflects the ability of the LLM to explore the state space.

We study the properties of MC(VK∗ ,Qf) below.

Proposition 3.3. Let MC(VK∗ ,Qf) be a Markov chain defined in Proposition 3.2. Then MC(VK∗ ,Qf) is ergodic and admits a unique stationary distribution.

The proof builds on showing that such a Markov chain has

- at most one recurrent class (blue block in Fig. 2) plus some additional transition states (green blocks in Fig. 2). We now

characterize how many times one should apply Qf to the input to reach the stationary distribution.

- Proposition 3.4. Let MC(VK∗ ,Qf) be as in Proposition 3.3 and e = (1,1,...,1)⊤. Then we have that

(a) (b) (c)

Convergence speed to the sta onary distribu on

|[Figure 7]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 4: Markov chain with a small GPT-like model. (a) Transition matrix Qf of the model where □ denotes the examples from the training set. (b) The stationary distribution of the trained model assigns almost uniform probabilities to the states seen during training. (c) Convergence rate to the stationary distribution for the considered toy model along with three LLMs, highlighting the dependence on K. The y-axis is the upper bound in Proposition 3.4.

Role of the temperature. To better illustrate the role of ε, we now plot the transition matrix of the studied Markov chain obtained when applying different temperature scaling to the logits returned by the trained model. As the temperature is commonly linked to the ability of LLMs to transition more freely to a large set of states (Chen & Ding, 2023), we expect that lower temperatures should negatively impact the convergence speed to the stationary distribution. In Fig. 5(a), we show that for a low temperature (0.2), the Markov chain mixes slowly and is unable to reach its stationary distribution (same line in the transition matrix as in Fig. 4(c)) even after 106 steps. In the case of a more commonly used temperature equal to 1 (Fig. 5(b)), the model requires only 300 steps to converge. Finally, setting the model’s temperature to 2 (Fig. 5(c)) makes the convergence extremely fast, reaching the stationary distribution after only 30 steps. The interplay between ε and the model’s temperature is displayed in Fig. 5(d), increasing the temperature leads to a drastic improvement in the convergence speed.

We have demonstrated that any autoregressive transformerbased LLM admits an equivalent Markov chain formulation. This equivalence connects nicely to some of the pathological behaviors of LLMs observed in practice. We will now use this formalization to provide a fine-grained analysis of LLMs’ generalization capabilities that may be of interest to both theorists and practical users.

##### 4. Sample Complexity and Generalization

We now study the generalization of an LLM fΘ as its capacity to infer correctly all the elements of Qf, while approximating the true reference matrix of transition probabilities Q∗. The hardness of this task is highlighted by the fact that an LLM observes a negligible amount of Q∗’s elements during its pre-training. Indeed, for GPT-3 (Brown et al.,

2020), this represents 5 × 1011 training tokens, which pales in comparison with the number of non-zero elements in Qf, given by TK+1 ≈ 109632. In this section, we determine the number of pre-training tokens required to be ϵ-close to perfect generalization. This result stems from a novel pre-training generalization bound on non-iid random variables in Section 4.2. We extend this bound to the in-context learning in Section 4.3.

4.1. Main Result: Pre-training Sample Complexity Non-iid data. We denote by X = (X1,...,XN

) the Ntrain tokens in V that fΘ observes during pre-training. The training sequences of tokens can be written as Sn = (X1,...,Xn) if n ≤ K and Sn = (Xn−K+1,...,Xn) otherwise due to the deletion process (see Definition B.2). In particular, the Sn are elements of VK∗ . We assume that the pre-training data S = (S1,...,SN

train

) is a sequence of dependent random variables with a mild coupling structure, namely that a Marton coupling with mixing matrix Γ exists for S = (S1,...,SN

train

) (more details on Marton coupling can be found in Appendix C.3). This ensures that our setting remains very broad as it subsumes the case of independent variables, m-dependent variables, language bigrams (Bietti et al., 2023), and the Markov chain setting considered in state-of-the-art ICL analysis of LLMs (Hu et al., 2024; Zhang et al., 2023b).

train

We state our main result on the pre-training sample complexity below. The proof is deferred to Appendix F.4.

Proposition 4.1 (Sample complexity). Let δ ∈ [0,1] and ϵ > 0. Assuming a perfect pre-training of fΘ and Ntrain pre-training tokens with Ntrain ≥ N∗ :=

Temperature =0.2 Temperature =1

Temperature =2

(a) (b) (c) (d)

|[Figure 11]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|[Figure 12]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|[Figure 13]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

[Figure 14]

[Figure 15]

[Figure 16]

Temperature

- Figure 5: Dependence of ε on the temperature of the model. (a) For low temperatures, ε becomes too small to achieve convergence to the stationary distribution. (b)-(c) Increasing the temperature from 1 to 2 leads to a ×10 faster convergence. (d) ε (log-scale) increase for temperature values in [0.1,2].

###### ⌈4B¯

2

ϵ2 log 2δ ⌉, we have with probability at least 1 − δ, ES∼P

∥Q∗(S,·) − Qf(S,·)∥1 ≤ ϵ, where

L

B¯ = 2∥Γ∥max{log (T) + 2BU/τ,log (1/c0)}1/2 is a model- and data-dependent constant.

- Proposition 4.1 shows that the number of pre-training tokens needed to achieve a good approximation mostly depends on the problem’s parameters captured by B¯. This constant includes two main ingredients. On the one hand, it accounts for the model’s architecture: the context window size K, the temperature scaling of the output logits τ, and the norm of

the unembedding layer BU. On the other hand, it captures – via the mixing matrix Γ – how interdependent the tokens are, including the special cases of iid data (∥Γ∥ = 1) and strongly dependent sequences of tokens. Thus, the bound is both model and data-dependent, contrary to the previously proposed non-vacuous bounds of Lotfi et al. (2024) where the model’s architecture was not fully taken into account.

Practical insights. Proposition 4.1 can be used to predict how well an LLM understands language given a fixed number of pre-training tokens. Letting Ntrain = N∗ in Proposition 4.1 leads to an approximation error

ϵ = √N2B¯

log 2δ . Since B¯ can be approximated as B¯ ∼ 2(log (T) + 2T√r/τ)1/2, we can verify the correlation between ϵ and the reported performance of the Llama (Dubey et al., 2024; Touvron et al., 2023a;b) and Gemma models (Mesnard et al., 2024; Riviere et al., 2024) (see Appendix E.1.1 for the experimental details). We use the values of T, r and Ntrain given in the technical reports of the open-source LLMs and plot the MMLU performance with respect to the approximation error ϵ in Fig. 1. This shows that our theory is model-specific as it results in distinctly different trends for the Llama and Gemma models.

train

The difference is due to the larger T of the Gemmas that compensates for the generally smaller embedding dimension r and leads to higher values of B¯. The predicted approximation error correlates with the performance for each family of models: the larger the predicted error, the worse the models perform. Finally, we note that in this work we didn’t estimate the norm of the Marton coupling as it requires access to the pre-training data. Studying this quantity and its approximation is of independent interest and we leave it for future work.

###### 4.2. Pre-Training Generalization Bound

Our main result on sample complexity is derived from the generalization bound that we present below.

Risk definition. We recall that for any n ≥ 1, the true probability of next token Xn+1 given a past sequence Sn is PL(· | Sn) ∈ ∆T and the probability estimated by the model is denoted by PΘ(· | Sn). Following the Markov chain formalization introduced in Section 3.1, we define the theoretical and empirical risks for any Θ ∈ W as

[dTV(Q∗(S,·),Qf(S,·))] R(Θ) :=

R(Θ) := ES∼P

L

N

(3)

1 N

dTV(PL(· | Sn),PΘ(· | Sn)).

n=1

Formally, the expected risk can be written as

[dTV(Q∗(S,·),Qf(S,·))]

R(Θ) = ES∼P

L

= ES∼P

[dTV(PL(· | S),PΘ(· | S))]

L

= E[ R(Θ)].

The generalization problem consists of bounding the difference R(Θ) − R(Θ).

Remark 4.1 (Choice of risk). Our risk definition departs from empirical risk minimization commonly used in statistical learning theory (Bach, 2024; Marion, 2023; Redko et al., 2019; Vapnik, 1999). To assess how well the model

0.8

100

O(

tmin = 43.7 tmin = 32.6 tmin = 15.2 tmin = 4.6

tmin = 43.7 tmin = 32.6 tmin = 15.2 tmin = 4.6

(a)

(b)

(c)

###### O(N−1

t min/N icl)

icl /2

0.6

)

Erroricl

Erroricl

Erroricl

0.4

Llama2 7B

Llama2 13B

10 1

0.2

10 1

Mistral 7B v0.1

Gemma 2B

Small Nicl Scaling law

0.0

100 101 102 103

100 101 102 103

100 101 102

Context Length Nicl

Context Length Nicl

Ratio Nicl/tmin

- Figure 6: In-context scaling laws. The risk Ricl as functions of Nicl, with 95% confidence intervals. (a) Risks for different LLMs along with the scaling law of Theorem 4.3. (b)-(c) Risks with Mistral 7B v0.1 for random 3-state transition matrices and different tmin as functions of Nicl and Nicl/tmin.

estimates the probability distribution of the next token, we follow (Hu et al., 2024; Zhang et al., 2023b) and study the TV distance used in learning and identity testing of Markov chains literature (Wolfer & Kontorovich, 2019; 2023). Appendix E.3 provides extended results with the KL divergence.

Generalization bound. We denote the risks by Rpre(Θ) and Rpre(Θ) to indicate that we take N = Ntrain in Eq. (3). Below, we state a generalization bound on the estimation risk of pre-training (see Appendix F.5 for the proof).

Theorem 4.2 (Pre-training generalization bound). Consider an LLM fΘ ∈ F. We denote by Γ the mixing matrix of the pre-training sequences of tokens (S1,...,SN

). Let 0 < δ < 1, then with probability at least 1 − δ,

train

B¯ √Ntrain

2 δ

Rpre(Θ) ≤ Rpre(Θ) +

,

log

where B¯ is the constant of Proposition 4.1 and writes B¯ = 2∥Γ∥max{log (T) + 2BU/τ,log (1/c0)}1/2.

The obtained result has a desirable dependency on the amount of the pre-training data O(Ntrain−1/2). Additionally, it showcases an interplay between the model architecture and the unknown reference constant c0. If BU ≈ O(T√r) (due to the normalization of the unembedding layer), then the model’s hidden dimension r and vocabulary size T should be large enough to ensure log(T) + 2BU/τ ≥ log(1/c0). Below this threshold, the architecture of fΘ is not expressive enough to have any tangible impact on its generalization, although it may affect the training error Rpre(Θ).

###### 4.3. In-Context Learning of Markov Chains

Although insightful, the analysis presented above is related to the pre-training of LLMs – a process that is hard and extremely costly to reproduce in practice. Similarly, we do not

have access to the ground-truth matrix Q∗ to reason about LLM’s ability to infer it in practice. To provide theoretical results that can be confirmed experimentally, we now turn our attention to ICL on Markov chains: a setup where one feeds a pre-trained LLM with an input sequence formed by a Markov chain of size Nicl defined over a state space Ω of size d1. The risks Ricl(Θ) and Ricl(Θ) are then defined similarly to Eq. (3) where transition kernel P of the Markov chain replaces the language reference distribution PL and we have N = Nicl (see Appendix F.7 for more details). To relate the generalization error to the pre-training error, we quantify the discrepancy between an LLM pre-trained mostly on textual data, and a hypothetical LLM with parameters in Wmc that is pre-trained on a dataset of Markov chains with the same data distribution as the Markov chain used as an input during in-context inference. We define the divergence between two models with weights Θ1,Θ2 and estimated probability distributions PΘ

as

###### ,PΘ

1

2

N

1 N

K(Θ1,Θ2):=

n=1

###### ES

[dTV(PΘ

(· | Sn),PΘ

(· | Sn))].

n

1

2

The operator K is akin to a distance (the separation property is only verified almost surely, see Appendix C.4 for more details). The next result, whose proof is deferred to Appendix F.7, provides a generalization bound on the in-context learning phase.

Theorem 4.3 (In-Context Learning generalization bound). Consider an LLM fΘ ∈ F. We provide as input of fΘ a d−state Markov chain X = (X1,...,XN

). The sequence of subsequences of the first n terms is denoted by S = (S1,...,Sn). S is also a Markov chain, and we denote by tmix(ε) its mixing time. Let tmin := inf0≤ε<1 tmix 2 ε (21−−εε)2. Let δ > 0.

icl

1This is different from another variation of ICL where supervised (x,y) pairs are provided in-context. Rather, the supervision is provided by observing transitions between states (xi, xi+1 = f(xi)) as discussed in (Li et al., 2023, Fig.1).

Then, with probability at least 1 − δ,

Fig. 6 (left), we note that Llama2 models deviate from our O(Nicl−1/2) theoretical scaling law, while most recent models (Mistral and Gemma) stay much closer to Theorem 4.3, similarly to what was observed by Cabannes et al. (2024). Being randomly generated, the Markov chains provided to the models have not been seen during training, and older (weaker) models naturally struggle to generalize.

Ricl(Θ) ≤ inf

{ Ricl(ϑ) + K(ϑ,Θ)}

ϑ∈Wmc

(4)

tmin Nicl

2 δ

+ B¯

log

,

with a constant depending on the problem’s parameters B¯ = 2max{log (d) + 2BU/τ,log (1/pmin)}1/2.

Dependence on tmin. Theorem 4.3 states that Markov chains with slow mixing (higher tmin) are slower to learn. We now plot the true risk for a single model with different values of tmin highlighting in Fig. 6 (right) a two-stage regime of ICL. In a first stage, the bound in Eq. (4) is dominated by tmin/Nicl for small Nicl, and depends strongly on tmin, while the scaling law O(Nicl−1/2) dominates as Nicl increases beyond Nicl ≈ 20.

We note that instead of ||Γ|| seen before, we now have an explicit dependency on tmin, which is related to the mixing time of the input Markov chain. This, together with the availability of the ground-truth transition matrix, allows us to use Theorem 4.3 to verify experimentally the in-context learning scaling laws for popular LLMs. Theorem 4.3 also suggests that an LLM pre-trained on diverse data sequences different from Markov chains should exhibit a certain degree of invariance to correctly infer the transition probabilities of the latter. This is reminiscent of the domain adaptation bounds (Redko et al., 2019) that commonly involve a distribution shift (i.e., a distance or a divergence) term that vanishes if the model is invariant to classes of transformations linking the distribution of the input data with that of test data. A recent success of LLMs in time series analysis (Gruver et al., 2023) suggests that this term may be small for certain types of data not used during pre-training.

Dependence on d. We now verify Theorem 4.3 for Markov chains with a different state space size (previously d = 3). We also consider a baseline given by the frequentist method mentioned before. For the latter, its dependence on d behaves like O( d/Nicl) (Wolfer & Kontorovich, 2019), while Theorem 4.3 gives O( log(d)/Nicl). For Markov chains with a small number of states d, there is no clear difference between the frequentist estimator and a LLM. However, as d grows the frequentist estimator struggles to estimate the transition matrix due to the O(

√

d) scaling factor. This is verified experimentally in Fig. 7, where we vary the parameter d from 3 (left) to 700 (right). We observe that the LLM follows the theoretical neural scaling law O(Nicl−1/2) and outperforms the frequentist method for d = 700, while being close to it for d = 3. We conclude that our analysis gives theoretical insights on the ICL neural scaling law observed empirically in (Liu et al., 2024). The additional experiments conducted in Appendix D.5 show that our bounds remain valid for large values of d.

Finally, we note that Theorem 4.3 implies that the LLM’s ability to learn Markov chains exceeds the frequentist method2 (Wolfer & Kontorovich, 2019) that consists of counting the occurrences of different states to fill in the matrix Qf. This is experimentally confirmed in Fig. 7 of Section 5 (more details in Appendix E.1.2).

##### 5. Numerical Experiments

We now evaluate the ability of recent LLMs, namely Mistral 7Bv0.1 (Jiang et al., 2023), Llama2 7B & 13B (Touvron et al., 2023b), and Gemma 2B (Mesnard et al., 2024) to infer transition probabilities of Markov chains in-context based on Theorem 4.3. We associate each state in the d-state Markov chain with a token from the set {0,...,d − 1}, concatenated to obtain a prompt of length Nicl. We add comas whenever necessary to ensure that each state is tokenized separately. Details on the experimental setup and experiments with more Markov chains and Llama3.2 (Dubey et al., 2024) are in Appendix D.1.

Dependence on Nicl. We first analyze the effect of Nicl on the risk calculated for a randomly generated 3-state Markov transition matrix. From the results presented in

2To the best of our knowledge, this is the only existing approach of Markov chains learning with theoretical guarantees.

100

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |O|(N| |− c| |l| | | |
| | | |F|re|qu|e|n|t|i|s|t| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | |G|em| |m|a| |2|B| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

Errorc

###### 1/2)

10 1

100 101 102 103

Context Length Nicl

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

100

Errorc

O(Nicl−1/2)

10 1

100 101 102

Context Length Nicl

Figure 7: Impact of the number of states. We plot the risks Ricl as functions of Nicl for Gemma 2B and the frequentist approach (Wolfer & Kontorovich, 2019) with 95% confidence intervals. Left. The input sequence is a random 3-state Markov chain. Right. The input sequence is a Brownian motion discretized as a 700-state Markov chain, similarly to Liu et al. (2024).

##### 6. Conclusion

This paper proposed an explicit characterization of the inference mechanism in large language models through an equivalent finite-state Markov chain. We provided an insightful theoretical analysis based on the established characterization and the ability of the LLM to infer the transition kernel approximating the true transition probabilities of language. Experiments on commonly used LLMs validate our findings. We adapted our results to in-context learning where experiments confirm our theoretical insights. In the future, we hope that the proposed equivalence will have far-reaching implications on our understanding of LLMs and allow for a more fine-grained understanding of their expressiveness.

##### Impact Statement

This paper presents work that aims to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which must be specifically highlighted here.

##### Acknowledgements

The authors would like to thank Alexander H¨agele for insightful feedback on the experiments with the large language models as well as Alain Durmus, Vasilii Feofanov, Giuseppe Paolo, Albert Thomas, Malik Tiomoko and Aladin Virmaux for fruitful discussions on early versions of this work. Nicolas Boull´e was supported by the Office of Naval Research (ONR), under grant N00014-23-1-2729.

##### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Agarwal, A., Kakade, S., Krishnamurthy, A., and Sun, W. Flambe: Structural complexity and representation learning of low rank mdps. In Advances in Neural Information Processing Systems, volume 33, pp. 20095–20107, 2020.

Ali, M., Fromm, M., Thellmann, K., et al. Tokenizer Choice For LLM Training: Negligible or Crucial? In Findings of the Association for Computational Linguistics: NAACL 2024, pp. 3907–3924. Association for Computational Linguistics, 2024.

Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., and Hauth, A. e. a. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Bach, F. Learning Theory from First Principles. MIT Press, 2024.

Bietti, A., Cabannes, V., Bouchacourt, D., Jegou, H., and Bottou, L. Birth of a Transformer: A Memory Viewpoint. In Advances on Neural Information Processing Systems, 2023.

Blondel, M., Martins, A., and Niculae, V. Learning classifiers with fenchel-young losses: Generalized entropies, margins, and algorithms. In Chaudhuri, K. and Sugiyama, M. (eds.), Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pp. 606–615. PMLR, 16–18 Apr 2019. URL https://proceedings.mlr.press/ v89/blondel19a.html.

Brown, T., Mann, B., Ryder, N., et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901, 2020.

Cabannes, V., Dohmatob, E., and Bietti, A. Scaling Laws for Associative Memories. In International Conference on Learning Representations, 2024.

Chen, H. and Ding, N. Probing the “creativity” of large language models: Can models produce divergent semantic association? In EMNLP, pp. 12881–12888. ACL, 2023.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. In Burstein, J., Doran, C., and Solorio, T. (eds.), Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4171–4186, 2019.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Edelman, B. L., Goel, S., Kakade, S., and Zhang, C. Inductive biases and variable creation in self-attention mechanisms. In International Conference on Machine Learning, pp. 5793–5831. PMLR, 2022.

Edelman, B. L., Edelman, E., Goel, S., Malach, E., and Tsilivis, N. The evolution of statistical induction heads: In-context learning markov chains. arXiv preprint arXiv:2402.11004, 2024.

Folland, G. B. Real analysis: modern techniques and their applications, volume 40. John Wiley & Sons, 1999.

Furuya, T., de Hoop, M. V., and Peyr´e, G. Transformers are universal in-context learners. arXiv preprint arXiv:2408.01367, 2024.

Gallager, R. G. Finite state Markov chains. In Discrete Stochastic Processes, pp. 103–147. Springer, 1996.

Golkar, S., Pettee, M., Eickenberg, M., Bietti, A., Cranmer, M., Krawezik, G., Lanusse, F., McCabe, M., Ohana, R., Parker, L., et al. xval: A continuous number encoding for large language models. arXiv preprint arXiv:2310.02989, 2023.

Gruver, N., Finzi, M., Qiu, S., and Wilson, A. G. Large language models are zero-shot time series forecasters. Advances in Neural Information Processing Systems, 36, 2023.

Hao, Y., Orlitsky, A., and Pichapati, V. On learning markov chains. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/ paper files/paper/2018/file/ d34ab169b70c9dcd35e62896010cd9ffPaper.pdf.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https:// openreview.net/forum?id=d7KBjmI3GmQ.

Hinton, G. Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531, 2015.

Hu, X., Zhang, F., Chen, S., and Yang, Z. Unveiling the Statistical Foundations of Chain-of-Thought Prompting Methods. arXiv preprint arXiv:2408.14511, 2024.

Ildiz, M. E., Huang, Y., Li, Y., Rawat, A. S., and Oymak, S. From Self-Attention to Markov Models: Unveiling the Dynamics of Generative Transformers. arXiv preprint arXiv:2402.13512, 2024.

Ivgi, M., Yoran, O., Berant, J., and Geva, M. From loops to oops: Fallback behaviors of language models under uncertainty, 2024. URL https://arxiv.org/abs/ 2407.06071.

Jeon, H. J., Lee, J. D., Lei, Q., and Van Roy, B. An Information-Theoretic Analysis of In-Context Learning. In International Conference on Machine Learning, volume 235, pp. 21522–21554, 2024.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.

Jurafsky, D. and Martin, J. H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. Pearson, 3rd edition, 2024.

Karpathy, A. minGPT: A minimal PyTorch reimplementation of the GPT (Generative Pretrained Transformer). https://github.com/karpathy/ minGPT, 2023. GitHub repository.

Kim, J., Nakamaki, T., and Suzuki, T. Transformers are Minimax Optimal Nonparametric In-Context Learners. arXiv preprint arXiv:2408.12186, 2024.

Li, Y., Ildiz, M. E., Papailiopoulos, D., and Oymak, S. Transformers as algorithms: Generalization and stability in in-context learning. In International Conference on Machine Learning, pp. 19565–19594. PMLR, 2023.

Liu, T. J., Boull´e, N., Sarfati, R., and Earls, C. J. LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law. arXiv preprint arXiv:2402.00795, 2024.

Lotfi, S., Finzi, M., Kuang, Y., Rudner, T. G., Goldblum, M., and Wilson, A. G. Non-vacuous generalization bounds for large language models. arXiv preprint arXiv:2312.17173, 2023.

Lotfi, S., Kuang, Y., Amos, B., Goldblum, M., Finzi, M., and Wilson, A. G. Unlocking tokens as data points for generalization bounds on larger language models. arXiv preprint arXiv:2407.18158, 2024.

Makkuva, A. V., Bondaschi, M., Girish, A., Nagle, A., Jaggi, M., Kim, H., and Gastpar, M. Attention with Markov: A framework for principled analysis of transformers via markov chains. arXiv preprint arXiv:2402.04161, 2024.

Marion, P. Generalization bounds for neural ordinary differential equations and deep residual networks. In Advances on Neural Information Processing Systems, volume 36, 2023.

Marton, K. Measure concentration for Euclidean distance in the case of dependent random variables. Ann. Probab., 32(3):2526–2544, 2004.

Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivi`ere, M., Kale, M. S., and Love, J. e. a. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., K¨opf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: an imperative style,

high-performance deep learning library. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2019. Curran Associates Inc.

Paulin, D. Concentration inequalities for Markov chains by Marton couplings and spectral methods. Electron. J. Probab., 20:79, 2015.

Peeperkorn, M., Kouwenhoven, T., Brown, D., and Jordanous, A. Is temperature the creativity parameter of large language models?, 2024. URL https://arxiv.org/ abs/2405.00492.

Redko, I., Habrard, A., Morvant, E., Sebban, M., and Bennani, Y. State of the Art of Statistical Learning Theory. In Advances in Domain Adaptation Theory, pp. 1–19. Elsevier, 2019.

Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Ram´e, A., and et al., J. F. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv.org/abs/2408.00118.

Roberts, A., Raffel, C., and Shazeer, N. How much knowledge can you pack into the parameters of a language model? In Empirical Methods in Natural Language Processing, pp. 5418–5426. Association for Computational Linguistics, 2020.

Roberts, G. O. and Rosenthal, J. S. General state space Markov chains and MCMC algorithms. Probab. Surveys, 1:20 – 71, 2004.

Sennrich, R., Haddow, B., and Birch, A. Neural Machine Translation of Rare Words with Subword Units. In Erk, K. and Smith, N. A. (eds.), Association for Computational Linguistics, pp. 1715–1725, 2016.

Singh, A. K. and Strouse, D. Tokenization counts: the impact of tokenization on arithmetic in frontier llms. arXiv preprint arXiv:2402.14903, 2024.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,

- A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Tsybakov, A. B. Introduction to Nonparametric Estimation. Springer, 1st edition, 2008.

Vapnik, V. N. An overview of statistical learning theory. IEEE Trans. Neural Netw., 10(5):988–999, 1999.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Veliˇckovi´c, P., Perivolaropoulos, C., Barbero, F., and Pascanu, R. softmax is not enough (for sharp outof-distribution), 2024. URL https://arxiv.org/ abs/2410.01104.

Wies, N., Levine, Y., and Shashua, A. The learnability of in-context learning. In Advances in Neural Information Processing Systems, volume 36, 2024.

Wolfer, G. and Kontorovich, A. Minimax learning of ergodic Markov chains. In Algorithmic Learning Theory, pp. 904–930. PMLR, 2019.

Wolfer, G. and Kontorovich, A. Learning and identity testing of Markov chains. In Handbook of Statistics, volume 49, pp. 85–102. Elsevier, 2023.

Wu, Z., Qi, Q., Zhuang, Z., Sun, H., and Wang, J. Pretokenization of numbers for large language models. In The Second Tiny Papers Track at ICLR 2024, 2024.

- Xie, R., Odonnat, A., Feofanov, V., Deng, W., Zhang, J., and An, B. Mano: Exploiting matrix norm for unsupervised accuracy estimation under distribution shifts, 2024. URL https://arxiv.org/abs/2405.18979.
- Xie, S. M., Raghunathan, A., Liang, P., and Ma, T. An Explanation of In-context Learning as Implicit Bayesian Inference. In International Conference on Learning Representations, 2022.

Zhang, B. and Sennrich, R. Root Mean Square Layer Normalization. In Advances in Neural Information Processing Systems, volume 32, 2019a.

Zhang, B. and Sennrich, R. Root mean square layer normalization. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alch´e-Buc, F., Fox, E., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019b. URL https://proceedings.neurips.cc/ paper files/paper/2019/file/ 1e8a19426224ca89e83cef47f1e7f53bPaper.pdf.

Zhang, W., Deng, Y., Liu, B., Pan, S. J., and Bing, L. Sentiment analysis in the era of large language models: A reality check. arXiv preprint arXiv:2305.15005, 2023a.

Zhang, Y., Zhang, F., Yang, Z., and Wang, Z. What and how does in-context learning learn? Bayesian model averaging, parameterization, and generalization. arXiv preprint arXiv:2305.19420, 2023b.

## Appendix

Roadmap. In Appendix A, we first recall our notations. We provide additional details on large language models and transformers in Appendix B. Important notions and definitions related to Markov chains and Marton couplings are given in Appendix C. The detailed proofs of our theoretical results are given in Appendix F. Finally, we provide additional experiments in Appendix D.

#### Table of Contents

- A Notations 14
- B Background on Large Language Models 14

- B.1 Token Generation and Deletion Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- B.2 Autoregressive Transformer-Based LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- C Background on Markov Chains 16

- C.1 Basic Notions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C.2 Ergodic Unichains . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.3 Marton Couplings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.4 An (Almost) Distance between Markov Chains . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D Additional Experiments 20

- D.1 Experimental Setup and Tokenization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.2 Impact of the Number of States d . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.3 More Structured Markov Chains . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.4 Recent Models: Impact of the Tokenization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- D.5 Dynamical Systems . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- E Additional Theoretical Results 26

- E.1 Sample Complexity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.2 Depth-Dependent Generalization Bounds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- E.3 Generalization Bounds with the KL Divergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- F Proofs 31

- F.1 Proof of Proposition 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- F.2 Proof of Proposition 3.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- F.3 Proof of Proposition 3.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- F.4 Proof of Proposition 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- F.5 Proof of Theorem 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- F.6 Proof of Corollary E.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- F.7 Proof of Theorem 4.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

##### A. Notations

We denote {1,··· ,N} as [N]. We represent scalar values with regular letters (e.g., parameter λ), vectors with bold lowercase letters (e.g., vector x), and matrices with bold capital letters (e.g., matrix A). The i-th row of the matrix A is denoted by Ai, its j-th column is denoted by A,j and its transpose is denoted by by A⊤. The identity matrix of size n is denoted by In ∈ Rn×n. The vector of size n with each entry equal to 1 is denoted by n. We denote by ∥A∥p,q the Lp,q matrix norm where the p-norm is over columns and the q-norm is over rows. We denote by ∥A∥ the operator norm of

- A induced by the ℓ2 norm and by ∥A∥∞ = maxij|Aij| the operator norm induced by the ℓ∞-norm. Similarly, x⊤ is the transpose of the vector x and ∥x∥p is its ℓp-norm. The total variation between two probability distributions P,Q is denoted by dTV(P,Q). The term “almost surely” is denoted by the notation “a.s.” while the term random variable is denoted by the notation “r.v.”. ∆n := {p ∈ [0,1]n| ni=1 pi = 1} is the probability simplex of Rn.
- B. Background on Large Language Models

We first recall important notions regarding large language models before focusing on the most widely used ones, namely the transformer-based LLMs. We describe the components of the vanilla transformer architecture before describing the whole network at the heart of such a model and formally defining the class of parameters and neural networks considered in our work.

###### B.1. Token Generation and Deletion Process

In this section, we recall how the sequences of tokens are processed by the large language model notably regarding the next token generation and the deletion process.

Definition B.1 (Generation process). Given an input s ∈ VK∗ of size p, an large language model outputs a probability mass function fΘT,K(s) over the discrete vocabulary space. A next token x is then sampled from fΘT,K(s), to construct a new sequence (s,x) of size p + 1.

Generation can be repeated by considering (s,x) as new input sequence and iterating this process. Since these models are designed to handle only sequences of size at most K, a deletion process is required.

Definition B.2 (Deletion process). Given an input s of size p > K, an large language model outputs a probability mass function fΘT,K(sK) where sK is a truncation of K tokens of the sequence s. Large language models implement front truncation, which is done by setting sK as the last K tokens of s.

As shown in Fig. 8, only the last K tokens of a long input sequence are used. This is why we speak of deletion, since we ignore the first tokens.

x1 x2 x3 x4

x1 x2 x3 x4 x5 x6 x7 x8 x9 x10

- Figure 8: Deletion process, front truncation. A large language model with context window K = 7 in navy blue, processing sequences of different lengths. Top. A sequence of length 4. Bottom. Front truncation of a sequence of length 10.

Note that it is possible to implement other kinds of truncation, but large language models usually do not (Brown et al., 2020; Touvron et al., 2023a), however, in models like BERT (Devlin et al., 2019), which are not autoregressive, back truncation as described in Fig. 9 is also an option.

###### x1 x2 x3 x4 x5 x6 x7 x8 x9 x10

- Figure 9: Back truncation. A large language model with context window K = 7 in navy blue, processing back truncation of a sequence of a sequence of length 10.

###### B.2. Autoregressive Transformer-Based LLMs

The most popular autoregressive LLMs rely on the transformer architecture (Vaswani et al., 2017) which we describe below following (Brown et al., 2020; Edelman et al., 2022; Zhang et al., 2023b). An autoregressive transformer-based LLM takes as input a sequence of length n, with n ≤ K and K is the context window, tokens with values in a vocabulary V of size T. The tokens are embedded into a r-dimensional space and the input can be written as S ∈ Rr×n. We consider a transformer model with L layers and h heads. The output of the ℓ-th layer writes S(ℓ) and is fed as input of the (ℓ + 1)-th layer. The input of the whole model is S(0) = S. Below, we describe the operations performed by the model, including the embeddings of the tokens.

- • Token embeddings. The tokens are embedded in a r-dimensional space via an embedding layer WE which results in an input of the form Sr×n;
- • Positional embeddings. (Learnable) positional embeddings are added to each token depending on its position in the input sequence. This breaks the permutation-invariance of the transformer architecture and leads, by abuse of notation, to an output S ∈ Rr×n;
- • Multi-head attention (MHA). Given an input sequence S ∈ Rr×n, query, key, and value matrices WQ,WK,WV ∈ Rr×r (here the value and output matrices are merged for ease of notations), the self-attention module computes

A(S;WQ,WK,WV ) := softmax WQS(WKS)⊤/√r (WV S) ∈ Rr×n,

with softmax: x ∈ Rn → exp(x)/ i exp(x)i ∈ ∆n. The operation described below corresponds to single-head self-attention. In practice, multi-head attention (MHA) is used with H heads and the query and key matrices are in RHr ×Hr

and the value matrix is in RHr ×r (r,H are taken such that Hr is an integer). The MHA module concatenates on the row dimension the outputs of A for each head and then projects it back to the embedding dimension r with an output matrix

WO ∈ Rr×r. By abuse of notation, we also denote by A this operation which results in an output of dimension r × n, and we include the output matrix in the argument of the operator. The ℓ-th layer of the transformer applies attention with layer-specific weight matrices and a residual connection that leads to an output

Z(ℓ) = S(ℓ−1) + A S(ℓ−1);WQ(ℓ),WK(ℓ),WV(ℓ),WO(ℓ) .

This is followed by a layer normalization (Zhang & Sennrich, 2019a) that projects each token into the ℓ2-unit ball, i.e., each column S·(,nℓ) has an ℓ2-norm lower than 1;

- • Feed-forward block (FF). Finally, a feed-forward block is applied, consisting of a two-layer MLP with hidden dimension

m, layer-specific weight matrices W1(ℓ) ∈ Rm×r,W2(ℓ) ∈ Rr×m and ReLU activation denoted by ReLU(x) = max{0,x} and applied entry-wise. The output of this layer reads

Y(ℓ) = W2(ℓ) ReLU W1(ℓ)Z(ℓ) . It is followed by a residual connection to produce the output

S(ℓ) = Z(ℓ) + W2(ℓ) ReLU W1(ℓ)Z(ℓ) ∈ Rr×n,

on which layer normalization (Zhang & Sennrich, 2019a) is applied ensuring that each column S(·,nℓ) has an ℓ2-norm lower than 1.

- • softmax output layer. In the autoregressive setting, the model outputs a probability distribution on the vocabulary V. To that end, the output S(L) ∈ Rr×n of the final layer is projected back to the vocabulary size by an “unembedding layer” WU ∈ RT×r and averaged over the columns to obtain a vector in RT. A softmax layer is finally applied on top of it to obtain the probability distribution of the next token PΘ(· | S). Formally, we have

1 nτ

WUS(L) n ∈ ∆T,

PΘ(· | S) = softmax

n is the length (i.e., number of columns) of the input sequence S (and thus of the last layer output S(L)), Θ denotes the parameters of the whole network that subsume the parameters of each layer and each block and τ is the softmax temperature (Hinton, 2015).

Theory faithful to the practice. The architecture described above is used in most of the transformer-based autoregressive LLM (Anil et al., 2023; Brown et al., 2020; Dubey et al., 2024; Jiang et al., 2023). In the theoretical analysis of Section 4, and unless specified otherwise, we remain faithful to their practical implementation and only make the following mild assumption: we assume that the unembedding layer is bounded. The class of parameters and the class of neural networks it generates respectively writes

W = {Θ | ∥WU⊤∥2,1 ≤ BU} and F = {fΘ | Θ ∈ W}.

It should be noted that this assumption is significantly weaker than what is usually done in the literature (Edelman et al., 2022; Zhang et al., 2023b).

##### C. Background on Markov Chains

We recall below some important notions related to Markov chains based on (Paulin, 2015; Roberts & Rosenthal, 2004) and that will be used in our proofs.

- C.1. Basic Notions Consider two distribution probabilities P and Q defined on a measurable space (Ω,F).

Definition C.1. The total variation between P and Q is defined as

dTV(P,Q) := sup A∈F

|P(A) − Q(A)|.

In the setting considered in the main paper, we consider Markov chains with finite discrete state space Ω. In this section, we refer to Ω as a general Polish space, whose elements are referred to as states.

Informally, a discrete-time, time-homogeneous Markov chain with state space Ω is a sequence of random variables (X1,X2,...) taking values in Ω such that the next observation is independent of the past given the present. This property is referred to as the Markov property and is defined below.

Definition C.2. A sequence of random variables (X1,X2,...) is said to satisfy the Markov property if for all n ≥ 1 and any (x1,...,xn+1) ∈ Ωn+1

P(Xn+1 = xn+1 | Xn = xn,··· ,X1 = x1) = P(Xn+1 = xn+1 | Xn = xn).

To a given Markov chain, we associate its transition kernel Q : Ω2 → [0,1] which collects the transition probabilities from one state to another

∀n ∈ N,(x,y) ∈ Ω2, Q(x,y) = P(Sn+1 = y | Sn = x).

In the main text, we refer to Q as a transition matrix as the Markov chains we consider are of finite state space.

Definition C.3. A distribution π on Ω is said to be a stationary distribution if the action of the transition kernel leaves π unchanged, that is

(πQ)(A) :=

Q(x,y)dπ(x) = π(A) for all A ∈ F.

y∈A

A natural question is whether such a distribution exists for a generic Markov chain. Before stating an existence theorem, we introduce a classification of states below.

Class of states. All definitions below are borrowed from (Gallager, 1996)

Definition C.4 (Accessibility and communication). A state x is accessible from y (abbreviated as x → y) if there exists n > 0 such that Qn(x,y) > 0. Two distinct states x and y communicate (abbreviated x ↔ y) if x is accessible from y and y is accessible from x.

Accessibility and communication concepts define how states can reach each other within a Markov chain. This leads to an important classification of states into transient and recurrent categories.

Definition C.5 (Recurrent and transient states). For finite-state Markov chains, a recurrent state is a state i that is accessible from all states that are accessible from i (i is recurrent if i → j implies that j → i). A transient state is a state that is not recurrent.

With the distinction between recurrent and transient states established, we can now group states into classes based on their communication properties.

Definition C.6 (Class of states). A class C of states is a non-empty set of states such that each i ∈ C communicates with every other state j ∈ C and communicates with no j ∈/ C

Aperiodicity and Ergodicity. Aperiodicity ensures that the system does not exhibit cyclic behavior, which is a key condition for understanding the asymptotic behavior of states.

Definition C.7 (Aperiodicity). The period of a state i, denoted d(i), is the greatest common divisor (gcd) of those values of n for which Qn(i,i) > 0. If the period is 1, the state is aperiodic.

Under some conditions on the Markov chain (aperiodicity and irreducibility (Roberts & Rosenthal, 2004)), it can be proven that the chain converges to its stationary distribution i.e. for any x ∈ Ω, limn→∞ dTV(Qn(x,·),π) = 0, where Qn(x,·) denotes the probability of Sn conditioned on S1 = x.

We recall below the notion of mixing time that assesses the time taken by the Markov chain to be ε-close to its stationary distribution (see Definition C.8).

Definition C.8 (Mixing time for time-homogeneous Markov chains (Paulin, 2015)). Let X := (X1,X2,...) be a time-homogeneous Markov chain with a state space Ω, a transition kernel Q, and a stationary distribution π. Its mixing time is defined for any ε ∈ [0,1] as

dTV Qt(x,·),π .

tmix(ε) := min{t | d(t) ≤ ε} where d(t) := sup x∈Ω

We also introduce the quantity

which will be useful later on.

tmin := inf

tmix

0≤ε<1

ε 2 ·

2 − ε 1 − ε

2

Remark C.1 (Well-posedness of tmin). As we only consider finite state-space Markov chains in our work, we know that a stationary distribution always exists. However, its uniqueness and the convergence to it require additional assumptions (see

Appendix C.2). In particular, not all Markov chains admit a finite tmix(ε), tmin for some ε < 12. In such case, tmin can be infinite. In our practical experimentation, this is never the case despite considering various Markov chains.

###### C.2. Ergodic Unichains

We are now ready to state the following theorem, which formalizes the classification of states into recurrent, transient, and aperiodic classes.

Theorem C.9 (Recurrent and transient classes). For finite state Markov chains, either all states in a class are transient or all are recurrent. We refer to these classes as transient and recurrent, respectively. For any Markov chain, all states in a class have the same period. If the period is 1, the class is said to be aperiodic

Having categorized states into recurrent, transient, and aperiodic classes, we can now define ergodicity.

Definition C.10 (Ergodicity). For a finite-state Markov chain, an ergodic class of states is a class that is both recurrent and aperiodic. A Markov chain consisting entirely of one ergodic class is called an ergodic chain.

Unichains. We now introduce the concept of unichains.

Definition C.11 (Unichains and ergodic unichains). A unichain is a finite-state Markov chain containing a single recurrent class and transient states. An ergodic unichain is a unichain for which the recurrent class is ergodic.

###### C.3. Marton Couplings

While we consider Markov chain inputs in Section 4.3, we consider less structured inputs during the pre-training phase Section 4.2.

More specifically, we model the sequences of tokens used during pre-training as generic dependent random variables. To derive meaningful results, we rely on the notion of Marton couplings introduced by Marton (2004). A Marton coupling can be seen as a weak dependency structure between random variables. The associated notion of the mixing matrix, analogous to the mixing time of a Markov chain, is used to assess the strength of the dependence between those variables.

This minimal modeling choice is made to remain as faithful as possible to the pre-training considered in practical applications of LLMs, for which the pre-training data is not public and may contain arbitrary data points such as video, code snippets, text and images (Achiam et al., 2023; Anil et al., 2023; Brown et al., 2020; Dubey et al., 2024; Jiang et al., 2023; Touvron

- et al., 2023a).

As shown in Paulin (2015, Remark 2.2.), considering sequences of random variables linked through a Marton coupling is a weaker assumption than what is usually done in the literature on generalization bounds, which typically relies on independent random variables and Markov chains (Hu et al., 2024; Marion, 2023; Wolfer & Kontorovich, 2019; Zhang et al., 2023b).

In particular, the results stated in Section 4.2 encompass the case where the pre-training input sequences of tokens are independent random variables (Kim et al., 2024) or Markov chains (Zhang et al., 2023b). We also note that Markov chains can model bigrams used in natural language (Bietti et al., 2023; Jurafsky & Martin, 2024). We do not provide an exhaustive review of Marton couplings. We will simply recall its definition and introduce the associated mixing matrix. We refer the interested reader to Marton (2004) and Paulin (2015). Consider a sequence of dependent random variables S = (S1,...,SN) taking values in a polish space Ω = Ω1 × ... × ΩN. We will denote by P(S1,...,SN) the distribution of S.

- Definition C.12 (Marton coupling). We define a Marton coupling for S as a set of couplings

S(s

1,...,si,s′i),S′(s1,...,si,s

′

i) ∈ Ω × Ω,

for every i ∈ [N], every s1 ∈ Ω1,...,si ∈ Ωi,s′i ∈ Ωi, satisfying the following conditions.

- (i) S(s1,...,si,s

′ i)

1 = s1, ..., S(s1,...,si,s

′ i)

i = si, S′(s1,...,si,s

′ i)

1 = s1, ..., S′(s1,...,si,s

′ i)

i−1 = si−1, S′(s1,...,si,s

′ i)

i = s′i.

- (ii) S(s1,...,si,s

′ i)

i+1 ,...,S(s1,...,si,s

′ i)

N

∼ P(Si+1,...,SN | S1 = s1,...,Si = si),

S′(x1,...,xi,x

′ i)

i+1 ,...,S′(x1,...,xi,x

′ i)

N

∼ P(Si+1,...,SN | S1 = x1,...,Si−1 = xi−1,Si = x′i).

- (iii) If xi = x′i, then S(x

1,...,xi,x′i) = S′(x

1,...,xi,x′i).

- Definition C.13 (Mixing matrix (Paulin, 2015)). For a Marton coupling, we define the mixing matrix Γ ∈ RN×N as an upper diagonal matrix with

 

- Γi,i := 1,
- Γj,i := 0

∀1 ≤ i < j ≤ N,

###### .



′ i)

′ i)

1,...,si,s′i P S(s1,...,si,s

j ̸= S′(s1,...,si,s

Γi,j := sups

j

For independent random variables, one can define a Marton coupling with a mixing matrix equal to the identity (see Paulin, 2015, Remark 2.2). In particular, it means that for independent variables, we have the operator norm of the mixing matrix equal to 1, i.e., ∥Γ∥ = 1.

- C.4. An (Almost) Distance between Markov Chains In Theorem F.23, We state elementary properties of K in the proposition below.

Proposition C.14 (Properties of K). K is an almost-distance between transition matrices in the sense that it satisfies the properties below:

- 1. Non-negativity. For any Θ1,Θ2, K(Θ1,Θ2) ≥ 0.
- 2. Almost sure positivity. K(Θ1,Θ2) = 0 ⇐⇒ ∀n ∈ [N],PΘ

1

(· | Sn) = PΘ

2

(· | Sn) a.s..

- 3. Symmetry. For any Θ1,Θ2, K(Θ1,Θ2) = K(Θ1,Θ2).
- 4. Triangular inequality.. For any Θ1,Θ2,Θ3, K(Θ1,Θ3) ≤ K(Θ1,Θ2) + K(Θ2,Θ3).

Proof of Proposition C.14. We first recall the following technical lemma.

Lemma C.15 (Proposition 2.16 in (Folland, 1999)). Let Y be a non-negative random variable defined on a probability space Ω with probability function P. If E[Y ] = 0, then Y = 0 almost surely, i.e.,

P({ω ∈ Ω | Y (ω) = 0}) = 1

The non-negativity and symmetry of K directly come from the symmetry and non-negativity of the total variation distance. The triangular inequality follows from the fact that the total variation is a distance and that the expectation respects inequalities. For the almost positivity, consider Θ1,Θ2 such that K(Θ1,Θ2) = 0. By non-negativity of all the terms in the sum, it means that for all n ∈ [N], we have

###### ES

[dTV(PΘ

(· | Sn),PΘ

(· | Sn))] = 0.

n

1

2

As the total variation is a distance, we know that the random variable under the expectation is non-negative. Applying Lemma C.15 leads to

(· | Sn)) = 0 almost surely.

dTV(PΘ

(· | Sn),PΘ

1

2

On the probability space, deprived of the set where the distance is non-zero (which is of null measure), the total variation is equal to zero and as a distance between probability distributions, it means that on this subset of the probability space, the probabilities are equal. Again, as the set on which they are not equal is of null measure, we have

(· | Sn) almost surely. Putting everything together, we have

###### PΘ

(· | Sn) = PΘ

1

2

(· | Sn) a.s., (5)

∀n ∈ [N],PΘ

(· | Sn) = PΘ

1

2

which concludes the direct sense. The converse sense is proved by assuming that Eq. (5) holds and using the distance properties of the total variation. This concludes the proof.

| |
|---|

##### D. Additional Experiments

In this section, we present additional numerical experiments that confirm that our theory correctly predicts the in-context learning behavior of LLMs.

###### D.1. Experimental Setup and Tokenization

Experimental setup. To ensure a fair validation of our theoretical results, we conduct our experiments on some of the most recent and widely used LLMs: Gemma 2B (Mesnard et al., 2024), Llama2 7B & 13B (Touvron et al., 2023b), Llama3 8B, Llama3.2 1B & 3B (Dubey et al., 2024), and Mistral 7Bv0.1 (Jiang et al., 2023).

Tokenization. As the models we consider have different tokenizations, we need to do this step with extra care as it is a crucial part of the experimental procedure. Indeed, LLMs’ ability to handle numerical values has been proved to be dependent on the tokenization algorithm (Ali et al., 2024; Gruver et al., 2023; Singh & Strouse, 2024). The most widely used tokenization algorithm to-date, BPE (Sennrich et al., 2016), tends to assign tokens to arbitrary 3-digits numbers based on their occurrences in large-scale corpora, and the tokenizer’s vocabulary size. As highlighted by (Gruver et al., 2023), this artifact severely hinders LLMs’ ability to predict numerical values in-context. This is the case for popular LLMs such as GPT-3 (Brown et al., 2020). Newer models (LLama3, GPT-3.5, GPT-4) however, tend to have hard-coded rules on top of BPE, making them able to encode all 3-digits numbers with their own token. Although this feature would accelerate the ICL procedure by eliminating the need for the Hierarchy-PDF algorithm in (Liu et al., 2024),the under-representability of larger numbers in the training data could be an issue. Other tokenization techniques that are numerical values-focused has been presented in the literature (Golkar et al., 2023; Wu et al., 2024), paving the way for another research direction that may benefit our method.

Rodmap. In the rest of this section, we extend our experiments to study the following setups:

- • In Appendix D.2: impact of the number of states d;
- • In Appendix D.3: extension to Markov chains with pmin = 0;
- • In Appendix D.4: impact of the tokenization;
- • In Appendix D.5: extension to dynamical systems.

###### D.2. Impact of the Number of States d

We further analyze the effect of the number of states d on the risk and consider randomly generated d-state transition matrices in Fig. 10. After a first stage of stagnation, the risk tends to take the correct scaling law coefficient. As in (Liu

- et al., 2024), we notice that considering randomly generated transition matrices seems to be difficult for an LLM to learn when there are more than 9 states. We interpret this behavior as the distribution shift term in Theorem 4.3. Indeed, the lack of structure in these transition matrices can hinder the correct decay of this term. Note also that the increase in d tends to

implicitly increase tmin, which could have an impact on the upper bound on Ricl (both in the generalization term and in the distribution shift term). We will now consider more structured Markov chains, and look at their impact on decay.

100

100

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

100

100

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

Llama2 7B Llama2 13B Mistral 7B v0.1 Gemma 2B 2Nicl1/2

- Figure 10: Impact of the number of states d. We plot the risk Ricl as functions of Nicl, with 95% confidence intervals. Upper Left. 2−states Markov transition matrices. Upper Right. 4−states Markov transition matrices. Lower Left. 6−states Markov transition matrices. Lower Right. 8−states Markov transition matrices.

- D.3. More Structured Markov Chains In this section, we empirically verify our theoretical results on more general Markov chains that do not verify pmin > 0.

- D.3.1. RANDOM WALKS

Random walks are a simple example of more structured Markov chains. Although we still have the possibility of discretizing the kernel of Markov chains with infinite state spaces as it is done in (Liu et al., 2024), we consider two types of random walks on finite state spaces.

0.5

- 0.5

- 1

1

0 1 2 3

0.5

0.5

Figure 11: Constrained random walk with d = 3.

Constrained random walk. We define the transition matrix P of a constrained random walk of d states as in Eq. (6). We draw the probabilistic graph in Fig. 11 for the case d = 3.



1, if i = 0 and j = 1, 1, if i = d − 1 and j = d − 2, 0.5, if 1 ≤ i ≤ d − 2 and j = i − 1, 0.5, if 1 ≤ i ≤ d − 2 and j = i + 1, 0, otherwise.



(6)

Pij =



Fig. 12 highlights the scaling laws of Theorem 4.3, as well as the log(d) dependency. As before, the best-performing models generalize almost perfectly.

| | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

100

100

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |

100

100

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

###### d = 6 d = 5 d = 4 d = 3 Nicl1/2

Figure 12: Constrained random walks. We plot the risk Ricl as functions of Nicl, with 99% confidence intervals. We consider different size d. Upper Left. Llama2 7B Upper Right. Llama2 13B Lower Left. Mistral 7Bv0.1 Lower Right. Gemma 2B

Polygonal random walk. We define the transition matrix P of a polygonal random walk of d states as in Eq. (7). We draw the probabilistic graph in Fig. 13 for the case d = 4.

 

0.5, if j = (i + 1) mod d (clockwise transition), 0.5, if j = (i − 1) mod d (counterclockwise transition), 0, otherwise.

(7)

Pij =



We draw the same conclusions as above for this second type of random walk, in Fig. 14.

1

0.5

0.5

0.5

0.5

0

2

0.5

0.5

0.5

0.5

3

Figure 13: Polygonal random walk with d = 4.

- D.3.2. INNER CLIQUES AND OUTER RIMS

Inner Cliques and Outer Rims. We also want to test our method on the class of Markov chain put forward in (Wolfer & Kontorovich, 2019) to derive their lower bound. Let η > 0 and d = 3k for some k ∈ N, and define the collection of Markov matrices Hη = {Mη,τ : τ ∈ {0,1}d/3}. Every element of this set consists of an inner clique and an outer rim. Mη,τ is the block matrix defined as follows,

Cη Rτ Rτ⊺ Lτ

,

Mη,τ =

where Cη ∈ Rd/3×d/3, Lτ ∈ R2d/3×2d/3, and Rτ ∈ Rd/3×2d/3 are given by

1 8

diag 7 − 4τ1ε,7 + 4τ1ε,...,7 − 4τd/3ε,7 + 4τd/3ε ,

Lτ =





- 3

- 4 − η d/3η−1 ... d/3η−1

... .

η d/3−1

- 3

- 4 − η

Cη =

,

... d/3η−1 η

...

 

 

.

d/3−1 ... d/3η−1 34 − η





1 + 4τ1ε 1 − 4τ1ε 0 ... ... ... 0 0 0 1 + 4τ2ε 1 − 4τ2ε 0 ... 0 . . . . . . . 0 ... ... ... 0 1 + 4τd/3ε 1 − 4τd/3ε

1 8

Rτ =

.

 

 

We provide in Fig. 15 a probabilistic graph of the case Mη,0 and d = 9. Fig. 16 compares different LLMs with the frequentist method, on the case depicted in Fig. 16 with η = 0.02. Although the frequentist method achieves a lower loss, the power laws seem to be the same with LLMs.

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

100

100

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |

100

100

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

d = 6 d = 5 d = 4 d = 3 Nicl1/2

Figure 14: Polygonal random walks. We plot the risk Ricl as functions of Nicl, with 99% confidence intervals. We consider different size d. Upper Left. Llama2 7B Upper Right. Llama2 13B Lower Left. Mistral 7Bv0.1 Lower Right. Gemma 2B

- 7

- 8

- 7

- 8

###### 3 4

- 3

- 4 − η

1 8

1 8

0

η 2

η 2

1 8

1 8

5

7

η 2

- 7

- 8

- 7

- 8

1 2

- 3

- 4 − η

- 3

- 4 − η

1 8

1 8

6

8

- 7

- 8

- 7

- 8

Figure 15: Probabilistic graph of Mη,0 when d = 9.

| | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

100

100

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

100

100

Erroricl

Erroricl

10 1

10 1

100 101 102 103

100 101 102 103

Context Length Nicl

Context Length Nicl

Llama2 7B Llama2 13B Mistral 7B v0.1 Gemma 2B Frequentist 2Nicl1/2

- Figure 16: We plot the risk Ricl as functions of Nicl, with 95% confidence intervals. Upper Left. Llama2 7B Upper Right. Llama2 13B Lower Left. Mistral 7Bv0.1 Lower Right. Gemma 2B

D.4. Recent Models: Impact of the Tokenization

As explained in Appendix D.1, models like Llama 3 tokenize 3-digit numbers with a single token. This saves a lot of inference compute time, but not necessarily in terms of performance when considering Markov chains with a few number of states d, since we have to separate the states by a comma to force tokenization into a single digit (e.g. the transitions 1 → 0 → 1 will be prompted as 1,0,1 (5 tokens) instead of 101 (1 token). In Fig. 17, we reproduce the same experiment as in Fig. 6(left), but with Llama 3 models. The scaling laws are quite good, but much less so than those obtained with Gemma 2B and Mistral 7Bv0.1 on the same inputs. On the other hand, with these models, it can be extremely interesting to consider Markov chains with many states, as we did in Fig. 7(right). In the next section, we will use LLama3 to learn other dynamic systems presented in (Liu et al., 2024).

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |Lla|ma|3|8|B| | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |(|N| | | | | | | | | | |Lla|ma|3.|2|1|B| | | |
| | | | | | |i|c|l|1|/2| | | | | | | | | | | | | | |
| | | | | | | | | | |)| | | | | |Lla|ma|3.|2|3|B| | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

100 101 102 103

Context Length Nicl

10 1

100

Erroricl

- Figure 17: In-context scaling laws for LLama3 herd of models. We plot the risk Ricl as functions of Nicl, with 95% confidence intervals.

###### D.5. Dynamical Systems

We consider four of the dynamic systems highlighted in (Liu et al., 2024) : a geometric Brownian motion, a correlated Gaussian, an uncorrelated Gaussian, and an uncorrelated uniform processe. We display in Fig. 18 the risks of LLama3 8B and the frequentist method, which once again highlights the emerging capacity of in-context learning.

100

Erroricl

Erroricl

###### O(Nicl−1/2)

Frequentist

Llama3 8B

10 1

100 101 102

Context Length Nicl

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

100

Erroricl

Erroricl

###### O(Nicl−1/2)

10 1

100 101 102

Context Length Nicl

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

100

###### O(Nicl−1/2)

10 1

100 101 102

Context Length Nicl

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

100

O(Nicl−1/2)

10 1

100 101 102

Context Length Nicl

- Figure 18: LLama3 8B on dynamical systems. We plot the risks Ricl as functions of Nicl for LLama3 8B and the frequentist approach (Wolfer & Kontorovich, 2019) with 95% confidence intervals. Upper Left. Geometric Brownian motion. Upper Right. Correlated Gaussian. Lower Left. Uncorrelated Gaussian. Lower Right. Uncorrelated Uniform.

- E. Additional Theoretical Results In this section, we present additional theoretical results on the sample complexity and generalization capabilities of LLMs.

###### E.1. Sample Complexity

In this section, we provide more details to Section 4.1, including the experimental setting, the connections to the Markov chain literature, and the extension of Proposition 4.1 to the in-context learning setting.

- E.1.1. PRE-TRAINING We recall that Proposition 4.1 states

Proposition E.1 (Restatement of Proposition 4.1). Let δ ∈ [0,1] and ϵ > 0. Assuming a perfect pre-training of fΘ and Ntrain pre-training tokens with Ntrain ≥ N∗ := ⌈4B¯

2

ϵ2 log 2δ ⌉, we have with probability at least 1 − δ, ES∼P

∥Q∗(S,·) − Qf(S,·)∥1 ≤ ϵ. with a constant depending on the problem’s parameters

L

B¯ = 2∥Γ∥max{log (T) + 2BU/τ,log (1/c0)}1/2.

To determine the approximation error of open-source LLM, it suffices to plug N∗ = Ntrain in Proposition 4.1. Then, the approximation error writes

2B¯ √Ntrain

2 δ

ϵ =

log

.

We recall that B¯ = 2∥Γ∥ max{log (T) + 2BU/τ,log (1/c0)}. To numerically compute ϵ, we proceed to the following simplifications. Since we do not have access to the training data of the Gemmas and Llamas, we cannot compute Γ that captures the data dependency. Even if we had access to the data, the amount of it prevents us from numerically computing Γ for all the pre-training sequences. The same goes for co. We circumvent these issues using the fact that T is very high in practice and compute B¯ = O(2 log (T) + 2BU/τ). Finally, we notice that the softmax temperature is of order 1 in those models (we note that it evolves in [0.2 − 1] depending on the downstream tasks (Dubey et al., 2024)) and that BU ∼ T√r. This comes from the fact that BU controls the norm of the unembedding matrix WU ∈ RT×r (see Appendix B), i.e.

###### ∥WU⊤∥2,1 ≤ BU.

When looking at this norm for the unembedding layer of Llama3 8B, we observe that it behaves like T√r. This is not surprising since the layer normalizations project the outputs’ columns on the unit-ball (notably for the RMSNorm (Zhang & Sennrich, 2019b) used in the Llamas models (Touvron et al., 2023b)) and WU is initialized following N(0,1) (in PyTorch (Paszke et al., 2019)). This should lead the entries of WU to be in [0,1] at the end of training which would imply, using the fact that WU⊤ ∈ Rr×T,

T

r

r

T

≤ T√r.

∥WU⊤∥2,1 =

(WU)2ji ≤1

###### (W⊤

U)2ij =

j=1

i=1

i=1

j=1

In summary, we can indeed consider BU = T√r and we obtain

2B¯ √Ntrain

ϵ =

2 δ

log

.

where

B¯ = 2 log (T) + 2T√r/τ.

In the technical reports of the Gemmas (Mesnard et al., 2024; Riviere et al., 2024) and Llamas (Dubey et al., 2024; Touvron et al., 2023a;b) models, the vocabulary sizes T, the embedding dimension r, the number of pre-training tokens are given and the MMLU results are given. We summarize it in Table 2. This enables us to plot the evolution of the MMLU with respect to the predicted approximation error ϵ in Fig. 1.

Model Nb. Pre-Training Tokens Ntrain Vocabulary Size T Embedding Dimension r Llama 7B (Touvron et al., 2023a) 1012 32000 4096

- Llama2 7B (Touvron et al., 2023b) 2 × 1012 32000 4096
- Llama3 8B (Dubey et al., 2024) 1.5 × 1013 128000 4096 Llama3.2 3B (Dubey et al., 2024) 1.5 × 1013 128000 3072 Gemma 2B (Mesnard et al., 2024) 3 × 1012 256128 2048 Gemma 7B (Mesnard et al., 2024) 6 × 1012 256128 3072 Gemma2 9B (Riviere et al., 2024) 8 × 1012 256128 3584 Gemma2 27B (Riviere et al., 2024) 1.3 × 1013 256128 4608

Table 2: LLMs’ parameters reported in the technical reports of the Llamas and Gemmas models.

- E.1.2. CONNECTION OF PROPOSITION 4.1 TO MARKOV CHAIN LITERATURE

Proposition 4.1 allows us to contextualize LLMs’ ability to learn Markov chains with respect to the existing literature. To the best of our knowledge, the only existing approach with theoretical guarantees for learning Markov chains is the

frequentist method (Wolfer & Kontorovich, 2019): counting the number of occurrences of different states to fill in the matrix Qf. Wolfer & Kontorovich (2019) show that the sample complexity of approximating Q∗ up to ϵ with such approach is N∗ = O(max{|VK∗ |/ϵ2γs,1/γsπ∗}log 1δ ) samples, where |VK∗ | = O(TK) in our setting, γs is a (pseudo) spectral gap of the Markov chain and π∗ is the smallest element of its stationary distribution. The authors state that the frequentist approach is minimax optimal (up to logarithmic factors). Our sample complexity in Proposition E.1 has a dependence that behaves as B¯2 = O(max{log (T) + 2T

√r τ ,log (1/c0)}), using BU ∼ T√r. Given that in practice T > r, it then simplifies

to N∗ = O(max{T/ϵ2τ,1/ϵ2}log 1δ ). In particular, our LLMs’ sample complexity is linear in the vocabulary size T, which is remarkable compared to the sample complexity of the frequentist approach, which scales as O(TK) with K the

context window.

Extension to In-Context Learning. Using Theorem 4.3, we can extend Proposition 4.1 to the in-context learning framework. The following proposition gives the sample complexity of in-context learning.

Proposition E.2 (Sample complexity of in-context learning). Let δ ∈ [0,1] and ϵ > 0. The model fΘ receives as inputs a d−state Markov chain X = (X1,...,XN

) with transition matrix Q. Assuming that there is no distribution shift and Nicl ≥ N∗ := ⌈4B¯

icl

2

ϵ2 log 2δ ⌉, we have with probability at least 1 − δ,

###### ES∼P

∥Q(S,·) − Qf(S,·)∥1 ≤ ϵ. with a constant depending on the problem’s parameters

L

###### B¯ = 2max{log (d) + 2BU/τ,log (1/pmin)}1/2.

Proof. We recall that the probability distribution associated with the input Markov chain is P and its transition matrix writes Q (see Appendix C for the connection between P and Q). We first note that by definition of the total variation distance (Wolfer & Kontorovich, 2019), we have

ES∼P∥Q(S,·) − Qf(S,·)∥1 = ES∼P[2 · dTV(Q(S,·),Qf(S,·))] = 2 · ES∼P[dTV(Q(S,·),Qf(S,·))]

= 2 · Ricl(Θ). (by definition of the risk Eq. (3)) Applying Theorem 4.3, we know that

B¯ √Nicl

{ Ricl(ϑ) + K(ϑ,Θ)} +

Ricl(Θ) ≤ inf

ϑ∈Wmc

2 δ

log

,

where B¯ is formally defined in Theorem 4.3. Assuming no distribution shift amounts to considering the infimum attained and equal to 0. We denote by N∗ the integer such that the error is equal to 2ϵ, i.e.,

B¯ √

2 δ

N∗ log

=

ϵ 2 ⇐⇒

B¯2 N∗ log

2 δ

ϵ2 4 ⇐⇒ N∗ =

=

2B¯ ϵ

2

log

2 δ

.

Taking the ceiling function ensures that N∗ is an integer. Hence, taking Ntrain ≥ N∗ = ⌈ 2ϵB¯

2

log 2δ ⌉ ensures that

B¯ √Nicl

B¯ √

ϵ 2

2 δ ≤

2 δ

=

.

log

N∗ log

Putting everything together, taking Nicl ≥ N∗ leads to

ϵ 2

ES∼P∥Q(S,·) − Qf(S,·)∥1 ≤ 2 · Ricl(Θ) ≤ 2 ·

= ϵ,

which concludes the proof.

Similarly to the analysis above, we observe that our sample complexity of in-context learning scales logarithmically with the number of states d while the frequentist’s one (Wolfer & Kontorovich, 2019) scales in O(d). In Fig. 7 of Section 5, we show that this is confirmed experimentally: LLM’s ability to learn Markov chains exceeds the frequentist approach for Markov chains with a large state space.

###### E.2. Depth-Dependent Generalization Bounds

We extend Theorem 4.2 to make its dependency on fΘ more fine-grained. Rather than assuming that only the norm of the embedding layer’s matrix is bounded, we follow the setting of prior work (Edelman et al., 2022; Furuya et al., 2024; Marion, 2023; Zhang et al., 2023b) and consider the parameter space defined as follows:

W ={Θ ∈ W | ∀ℓ ∈ [L],∥WV(ℓ)∥∞ ≤ BV , ∥WO(ℓ)∥∞ ≤ BO,∥W1(ℓ)∥∞ ≤ B1,∥W2(ℓ)∥∞ ≤ B2}.

The definition of W concerns the query, key, and value matrices of all layers and heads. Similarly to Zhang et al. (2023b, Assumption 5.1), we assume that each token has an ℓ1-norm bounded by Btok. We have the following generalization bound, whose proof is deferred to Appendix F.6.

Corollary E.3 (Depth-dependent bound). Consider an LLM fΘ ∈ F˜ := {fΘ | Θ ∈ W}˜ . With the same assumptions as in Theorem 4.2, we have

B¯ √Ntrain

2 δ

Rpre(Θ) ≤ Rpre(Θ) +

log

,

with constants depending on the problem’s parameters

B¯ = 2∥Γ∥max{log (T) + 2(BΘ)L/τ,log (1/c0)}1/2, BΘ = [(1 + rmB1B2)(1 +

r3 H

BOBV )](BtokBU)1/L.

We note that B¯ exhibits an exponential dependence on the depth of the transformer, which also amplifies the hidden dimensionality (width) of the embedding layer r. This contrasts with the dependency in m, the hidden dimensionality of the MLP block, which is linear. All these factors are commonly associated with higher expressive power of transformers suggesting that they should contribute to a better minimization of Rpre(Θ) at the expense of requiring more training data. The number of heads H can be used as a counterbalance to increasing the width in the cubic term r3, suggesting that a good balance between these parameters may lead to more data-efficient models.

###### E.3. Generalization Bounds with the KL Divergence

As explained in Remark 4.1, the total variation is the natural choice to define the risks in Eq. (3). Another possibility in the Markov chain literature is to use the KL divergence to compare probability distributions (Hao et al., 2018). This is an interesting candidate as the KL divergence is naturally connected to the cross-entropy loss commonly used to train neural networks (the cross-entropy corresponds to the KL divergence between the true distribution and the predicted softmax distribution (Blondel et al., 2019). In this section, we discuss the extension of the theoretical results of Section 4 by replacing the TV distance with the KL divergence in the risks’ definition, i.e.,

R(Θ) := ES∼P

L

1 N

[dKL(Q∗(S,·)||Qf(S,·))], R(Θ) :=

N

dKL(PL(· | Sn)||PΘ(· | Sn)). (8)

n=1

Theorem 4.2 and Corollary E.3 related to the pre-training phase in Section 4.2 can be obtained similarly if the risks are defined with the KL divergence following Eq. (8). Indeed, the key step to derive the proofs is to obtain a similar result to Lemma F.6 but with the KL divergence. The next lemma provides this result.

Lemma E.4. Consider two probability distributions P,Q defined on a measure space (Ω,F) and a σ-finite measure ν on (Ω,F). Let p,q be the corresponding probabilities densities, i.e., we have P(dω) = q(ω)ν(dω) and Q(dω) =

p(ω)ν(dω). If there exists a non-negative constant B such that for any z ∈ Ω, log QP((zz)) ≤ B, then we have dKL(P||Q) ≤ B.

Proof. We have

0 ≤ dKL(P||Q) = |dKL(P||Q)|

- P(z)

- Q(z)

= P(z)log(

)dz

P(z) Q(z)

≤ |P(z)||log(

)|dz

≤ B |P(z)|dz

= B P(z)dz

= B, which concludes the proof.

| |
|---|

We can now state the results similar to Theorem 4.2, Corollary E.3 and Proposition 4.1 from the pre-training phase when the risk is defined according to Eq. (8).

Theorem E.5 (Pre-training generalization bound). Consider an LLM fΘ ∈ F. We denote by Γ the mixing matrix of the pre-training sequences of tokens (S1,...,SN

). Let 0 < δ < 1, then with probability at least 1 − δ,

train

B¯ √Ntrain

2 δ

Rpre(Θ) ≤ Rpre(Θ) +

log

,

where B¯ = √2∥Γ∥max{log (T) + 2BU/τ,log (1/c0)} is a constant depending on the parameters of the problem.

Proof. The proof simply follows from the proof of Theorem 4.2 by replacing the upper bound √

2B by B (with the appropriate upper-bound B) when Lemma F.6 is used in the proof.

Corollary E.6 (Depth-dependent bound). Consider an LLM fΘ ∈ F˜ := {fΘ | Θ ∈ W}˜ . With the same assumptions as in Theorem 4.2, we have

B¯ √Ntrain

2 δ

Rpre(Θ) ≤ Rpre(Θ) +

log

,

where B¯ = √2∥Γ∥max{log (T) + 2(BΘ)L/τ,log (1/c0)} is a constant depending on the parameters of the problem, and BΘ = [(1 + rmB1B2)(1 + r

3

H BOBV )](BtokBU)1/L.

Proof. The proof simply follows from the proof of Theorem 4.2 by replacing the upper bound √

2B by B (with the appropriate upper-bound B) when Lemma F.6 is used in the proof.

Limitations. We recall from Remark 4.1 that the TV distance is a natural choice to compare transition matrices in the Markov chain literature. In addition, while the KL divergence can be used to compare probability distributions, it does not define a metric space. Hence, we cannot straightforwardly extend Theorem 4.3 with the KL divergence because the proof relies on the use of the triangular inequality. As Theorem 4.3 is one of our main results and enables us to show that the theory and the practice align (Section 5), this also contributed to our preference for the TV distance instead of the KL divergence. We also note that the proof of Proposition 4.1 relies on properties of the total variation and hence we cannot extend it straightforwardly with the KL divergence.

##### F. Proofs

- F.1. Proof of Proposition 3.2 We detail below the proof of Proposition 3.2.

- Proof of Proposition 3.2. Step 1: Large language models as Markov chains. Given an input vi ∈ VK∗ of p tokens, a large

language model outputs a probability mass function fΘT,K(vi) over the discrete vocabulary space. As the temperature is positive, i.e., τ > 0, and as the exponential is positive, we know that all the tokens in the vocabulary will be given a positive

mass.

A next sequence vj ∈ VK∗ is then sampled according to fΘT,K(vi). But the vj sequences that fit necessarily contain the vi sequence (except possibly the first element of vi, thanks to Definition B.2), i.e. ∀l,(vj)l = (vi)l+1. Note also the size of vj is p + 1 when p < k and k when p = k. All other sequences vj that do not satisfy this condition are not suitable.

∗

K|×|VK∗ |, as defined

In that sense, fΘT,K can be represented by a Markov chain MC(VK∗ ,Qf) with transition kernel Qf ∈ R|V

in Proposition 3.2. Step 2: Proportion of non-zero elements. We denote by R the set of states of length K. The set of states of length strictly less than equal K is denoted by T . We can construct a transition matrix PR ∈ RT

K×TK with the states of this class, containing the probabilities of moving from one state of R to another. PR corresponds to the blue block in Fig. 2 while green rectangle blocks correspond to part of PT and PT R in the following description of large language models as Markov chains,

 . (9)

 

PT PT R 0 PR

Qf =

Now, let us count the number of non-zero elements in each of these 4 large blocks. PT block : The size of this block is

T T − 1

T T − 1

(TK−1 − 1) ×

(TK−1 − 1) . There are K − 2 green blocks contained

in PT . The block number i ∈ [K − 2] is of size Ti × Ti+1. Since each sentence of size i can be completed with non-zero probability, by any other token, there are a total of T

i

p=1 T = Ti+1 non-zero elements. There are therefore Ki=1−2 Ti+1

non-zero elements in the entire PT block. PT R block : The size of this block is

###### T T − 1

(TK−1−1) ×TK. The green block contained in PT R that contains non-zero elements is of size TK−1 × TK. Since each sentence of size K − 1 can be completed with non-zero probability, by any other token, there are a total of T

K−1

p=1 T = TK non-zero elements.

PR block : The size of this block is TK × TK. Each sentence v = (v1,...,vK) of size K is mapped to another sentence v′ = (v1′ ,...,vK′ ) of size K with non-zero probability, if and only if v1′ = v2,v2′ = v3,...,vk′ −1 = vK. The final token vK′ can by any other token in the vocabulary. It means that there are a total of T

K

p=1 T = TK+1 non-zero elements.

0′s block : There are no non-zero elements in this block. Finally, there are

K−2

K

TK − 1 T − 1

Ti+1 + TK + TK+1 =

Ti+1 = T2

i=1

i=1

non-zero elements. This means that the proportion of non-zero elements in the matrix is exactly

Note that for large T and K we have that

K−1 T−1

T2 T

T − 1 TK − 1

2 =

.

T TTK−−11

T − 1 TK − 1 ∼

1 TK−1.

| |
|---|

- F.2. Proof of Proposition 3.3 We begin with a preliminary lemma.

Lemma F.1 (Powers of Qf greater than K). For any initial state i, the following hold:

- • ∀k ≥ K,∀j ∈ T ,(Qkf)i,j = 0,
- • ∀k ≥ K,∀j ∈ R,(Qkf)i,j > 0.

Proof. By considering Qf as defined in (9), we can compute its powers. For any k ≥ 1,

 

 ,

PTk Bk 0 PRk

Qkf =

where Bk = km−=01 PTmPT RPRk−1−m.

To prove the first item, we focus on the blocks on the left of Qf. Since the lower left block is zero, we have that ∀k ≥ 1,∀i ∈ R,∀j ∈ T ,(Qkf)i,j = 0. In the upper left block, the element (PTk )i,j designates the probability of moving from one transient state i ∈ T to another transient state j ∈ T after k iterations. According to Definition B.1, if state i is a sequence of p ≥ 1 tokens, state j is necessarily a sequence of min{K,p + K} = K elements. Thus, PT is a nilpotent matrix and ∀k ≥ K,∀i,j,(PTk )i,j = 0. This proves that ∀k ≥ K,∀j ∈ T ,(Qkf)i,j = 0.

We now move on to the second item. From the above, ∀k ≥ K,Bk = Km=0−1 PTmPT RPRk−1−m. Note that this sum is finite, but there is still a dependence on k, in the powers of the matrix PR. In the lower right block, the element (PRk )i,j designates the probability of moving from one recurrent state i ∈ R to another recurrent state j ∈ R after k iterations. According to the definition of Qf in Proposition 3.2 and Definition B.2, ∀k ≥ K,∀i,j ∈ R2,(PRk )i,j is nonzero. Exploiting this also in Bk, we obtain the result, i.e. ∀k ≥ K,∀j ∈ R,(Qkf)i,j > 0.

| |
|---|

- We are now ready to prove Proposition 3.3, which is inspired by Gallager (1996).

- Proof of Proposition 3.3. The states of length strictly less than equal K (elements of T ) are transient, because of Definition B.1. To discuss the nature of states of length K (elements of R), let us introduce a result regarding the powers of the

Qf matrix as defined in (9). Thanks to Lemma F.1, the set of states R (i.e. the states of length K) form a class. Lemma F.1 gives us also that R is ergodic. In fact, every state in R only communicates with all the other states in R, which proves

the recurrence. Since ∀i,j ∈ R2,(QKf )i,j > 0, we can move between any two states in exactly K steps, regardless of the initial position. This ensures that R is aperiodic because the transition probabilities do not depend on a specific cycle, and

states can be revisited at various time steps, not just multiples of a particular number. More simply, by considering a token x, the state defined as i = xx...x

has period 1, i.e. (Qf)i,i > 0. This is a consequence of Definition B.2 and Proposition 3.2.

K times

Thanks to Theorem C.9, it means that the whole class R is aperiodic. Finally, this means that MC(VK∗ ,Qf) are ergodic unichains, in the sense of Definition C.11.

| |
|---|

###### F.3. Proof of Proposition 3.4

We start by introducing three technical lemmas that will be useful in the proof of Proposition 3.4. We start with the Chapman–Kolmogorov equation.

Lemma F.2 (Chapman-Kolmogorov equation). Let P be a matrix of size d. Then, P satisfies

d

′

′

∀i,j ∈ [d]2,∀n,n′ ∈ N2,(Pn+n

(Pn)i,k(Pn

)i,j =

)k,j.

k=1

′

Proof. The result follows from the fact that ∀n,n′ ∈ N2,Pn+n

Then, we introduce a simple but useful result of monotonicity.

′

.

= PnPn

| |
|---|

Lemma F.3 (Lemma 3.3.1. in Gallager (1996)). Let the transition matrix P of a finite state Markov chain. Then, for all states j and n ≥ 1, we have

(Pn+1)i,j ≤ max

(Pn)i,j, and min

(Pn+1)i,j ≥ min

(Pn)i,j.

max

i

i

i

i

We now refer to a result on Markov chains with positive transition matrices.

Lemma F.4 (Lemma 3.3.2. in Gallager (1996)). Let the transition matrix P of a finite state Markov chain satisfy ∀i,j,Pi,j > 0, and let α = min

Pi,j > 0. Then, for all states j and n ≥ 1,

i,j

(Pn)i,j , max i

(Pn)i,j − min

(Pn)i,j ≤ (1 − 2α) max

(Pn)i,j − min

max

i

i

i

i

(Pn)i,j − min

(Pn)i,j ≤ (1 − 2α)n, lim n→∞

i

(Pn)i,j = lim

(Pn)i,j > 0.

max

min

n→∞

i

i

- We are now ready to prove Proposition 3.4 using a similar argument as in (Gallager, 1996).

- Proof of Proposition 3.4. Let T and R denote respectively the sets of transient and recurrent states. For any state j, we define πj := limn→∞ maxi (Qnf)i,j = limn→∞ mini (Qnf)i,j. Then π = (πj)j∈Ω is the stationary distribution for Qf.

- Step 1: Stationary distribution for transient states. Lemma F.1 gives us that ∀i,∀k ≥ K,∀j ∈ T ,(Qkf)i,j = 0. This means that ∀j ∈ T ,πj = 0 and hence the limit is reached at most after K iteration.
- Step 2: Stationary distribution for recurrent states. Lemma F.1 gives us ∀i,j ∈ R2,(QKf )i,j > 0. By defining

(QKf )i,j, Lemma F.4 shows that for any integer ℓ ≥ 1,

ε := min

i,j∈R2

(QℓKf )i,j , (10) max

(QℓKf )i,j − min i∈R

(QℓKf )i,j ≤ (1 − 2ε) max i∈R

(QℓKf )i,j − min i∈R

max

i∈R

(QℓKf )i,j − min i∈R

(QℓKf )i,j ≤ (1 − 2ε)ℓ, (11) lim ℓ→∞

i∈R

(QℓKf )i,j > 0. (12) Thanks to Lemma F.3, max

(QℓKf )i,j = lim

max

min

ℓ→∞

i∈R

i∈R

(Qnf+1)i,j is non-decreasing in n, so the limit on the left in Eq. (12) can be replaced with a limit in n. The same argument for the limit on the right gives that, ∀j ∈ R,

i

(Qnf)i,j ≤ (1 − 2ε)⌊n/K⌋,

(Qnf)i,j − min i∈R

max

i∈R

(Qnf)i,j = lim

(Qnf)i,j > 0,

lim

max

min

n→∞

n→∞

i∈R

i∈R

where we have taken the floor function to also convert the result of (11). Since πj lies between the minimum and maximum (Qnf)i,j for each n, we have that ∀i,j ∈ R2,

n

|(Qnf)i,j − πj| ≤ (1 − 2ε)⌊

K⌋.

It means that ∀i,j ∈ R2,πj = limn→∞(Qnf)i,j. This also gives us the convergence rate when the initial state i is recurrent. In the next step, we consider the general convergence rate, regardless of the nature of the initial state i.

- Step 3: Convergence bound. We proceed to the remaining case, i.e. the case where the initial state i ∈ T and the final state j ∈ R. Lemma F.2 says that ∀n ≥ K,

(QKf )i,k(Qnf−K)k,j +

(QKf )i,k(Qnf−K)k,j.

(Qnf)i,j =

k∈T

k∈R

We then have that ∀i ∈ T ,∀n ∈ N,

(QKf )i,k (Qnf−K)k,j − πj +

(QKf )i,k (Qnf−K)k,j − πj

|(Qnf)i,j − πj| ≤

k∈T

k∈R

(QKf )i,k (Qnf−K)k,j − πj +

(QKf )i,k (Qnf−K)k,j − πj

≤

k∈T

k∈R

(QKf )i,k (Qnf−K)k,j − πj

(QKf )i,k +

≤

k∈T

k∈R

n−K

K ⌋,

≤ (1 − 2ε)⌊

where the first sum vanishes and k∈R(QKf )i,k ≤ 1. Finally, ∀i ∈ T ,∀n ≥ K,

n

K⌋−1. Combining this with the result of Step 2 concludes the proof.

|(Qnf)i,j − πj| ≤ (1 − 2ε)⌊

| |
|---|

- F.4. Proof of Proposition 4.1 We detail the proof of Proposition 4.1 below.

Proof. We first note that by definition of the total variation distance (Wolfer & Kontorovich, 2019), we have

∥Q∗(S,·) − Qf(S,·)∥1 = ES∼P

[2 · dTV(Q∗(S,·),Qf(S,·))]

ES∼P

L

L

[dTV(Q∗(S,·),Qf(S,·))]

= 2 · ES∼P

L

= 2 · Rpre(Θ). (by definition of the risk Eq. (3)) Applying Theorem 4.2 (a similar result can be derived using Corollary E.3), we know that

B¯ √Ntrain

Rpre(Θ) ≤ Rpre(Θ) +

2 δ

log

,

where B¯ is formally defined in Theorem 4.2 (respectively Corollary E.3). Assuming a perfect pre-training error amounts to consider Rpre(Θ) = 0. We denote by N∗ the integer such that the error is equal to 2ϵ, i.e.,

B¯ √

2 δ

N∗ log

=

ϵ 2 ⇐⇒

B¯2 N∗ log

2 δ

ϵ2 4 ⇐⇒ N∗ =

=

2B¯ ϵ

2

log

2 δ

.

2

Taking the ceiling function ensures that N∗ is an integer. Hence, taking Ntrain ≥ N∗ = ⌈ 2ϵB¯

log 2δ ⌉ ensures that B¯ √Ntrain

B¯ √

2 δ ≤

2 δ

ϵ 2

log

N∗ log

=

.

Putting everything together, taking Ntrain ≥ N∗ leads to

ϵ 2

∥Q∗(S,·) − Qf(S,·)∥1 ≤ 2 · Rpre(Θ) ≤ 2 ·

ES∼P

= ϵ, which concludes the proof.

L

| |
|---|

- F.5. Proof of Theorem 4.2 In this section, we detail the proof of Theorem 4.2. We provide below an overview of the proof before detailing it.

Overview of the proof. We are going to use McDiarmid’s inequality for dependent random variables of Paulin (2015, Theorem 2.9). To adapt the arguments of Paulin (2015, Theorem 2.9) to our setting, we bound the total variation between the true probability of the next token and the one estimated by the LLM. The rest of this section is organized as follows. First in Appendix F.5.1, we adapt the concentration inequality of Paulin (2015, Theorem 2.9). Then in Appendix F.5.2, we show how to bound the total variation between the true and the estimated probability of the next token. , in Appendix F.5.3, we restate Theorem 4.2 and conclude the proof.

- F.5.1. CONCENTRATION INEQUALITIES FOR DEPENDENT RANDOM VARIABLES We first state a concentration inequality for dependent random variables that will be used to obtain our final bound.

Proposition F.5 (McDiarmid’s inequality for dependent random variables). Let S := (S1,...,SN) be a sequence of random variables that take values in Ω = Ω1 × ... × ΩN. Assume there exists a Marton coupling for S with mixing matrix Γ. Let ∥Γ∥ be the operator norm of Γ. If f : Ω → R is such that there exists c ∈ RN satisfying

N

∀x,y ∈ Ω, f(x) − f(y) ≤

###### ci {x

i̸=yi},

i=1

then we have for any u ≥ 0,

P(|f(S) − ES[f(S)]| ≥ u) ≤ 2exp −2u2 ∥Γ∥2∥c∥22

.

Proof. Consider a function f verifying the properties of Proposition F.5. Paulin (2015, Theorem 2.9) ensures that for a partition Sˆ of S (see Paulin, 2015, Definition 2.3) the following inequality holds

∀u ≥ 0, P |f(Sˆ) − E f(Sˆ) | ≥ u ≤ 2exp −2u2 ∥Γ · C(c)∥22

, (13)

where C(c) is a vector of RN whose i-th element is the sum of the cj such that j is an index of the elements of Sˆi. Taking the trivial partition Sˆ = S implies that the index of the elements in Sˆi are reduced to {i}. Hence the i-th entry of C(c) is equal to ci and C(c) = c. By definition of the operator norm (naturally induced by the ℓ2-norm), we have

∥Γ · c∥2 = ∥Γc∥2 ∥c∥2

∥Γx∥2 ∥x∥2

·∥c∥2 ≤ ∥Γ∥ · ∥c∥2,

· ∥c∥2 ≤ sup x̸=0

=∥Γ∥

where the first inequality comes from the fact that c is non-zero (otherwise the only possible f is the zero function which is not of great interest). Using the fact that the function x → exp(−2u

2

x ) is increasing, we obtain exp −2u2

≤ exp −2u2 ∥Γ∥2 · ∥c∥22

,

∥Γ · c∥22

which concludes the proof.

| |
|---|

By looking at the definition of the risk Rpre(Θ), we can see that applying Proposition F.5 to the function

###### f : (S1,...,SN

) =

train

Ntrain

1 Ntrain

dTV(PL(· | Sn),PΘ(· | Sn)),

n=1

would lead to the desired bound as we already know S admits a Marton coupling with mixing matrix Γ. We investigate in the next section how to find the bounding vector c to apply Proposition F.5.

- F.5.2. FINDING THE BOUNDING VECTOR

Technical lemmas. We first recall the following important notions from (Tsybakov, 2008). Let (Ω,F) be a measure space and consider two probability distributions P,Q defined on (Ω,F). For any σ-finite measure ν on (Ω,F) such that P,Q are absolutely continuous with respect to ν, we can define p = dνdP,q = ddνQ which can also be written as P(dω) = q(ω)ν(dω) and Q(dω) = p(ω)ν(dω). We will adopt both notations interchangeably. It should be noted that there always exists at least one such measure ν as one can take ν = P + Q. With these notations, the squared Hellinger distance between P and Q is defined as

2

2

###### H(P,Q)2 :=

P(dω) − Q(dω)

p(ω) − q(ω)

ν(dω) =

###### .

ω∈Ω

ω∈Ω

The lemma below shows that the total variation between two probability distributions is controlled from above by the absolute value of the logarithm of their ratio.

Lemma F.6. Consider two probability distributions P,Q defined on a measure space (Ω,F) and a σ-finite measure ν on (Ω,F). Let p,q be the corresponding probabilities densities, i.e., we have P(dω) = q(ω)ν(dω) and Q(dω) = p(ω)ν(dω), the total variation between P and Q satisfies

1/2

- P(dω)

- Q(dω)

dTV(P,Q) ≤ 2

log

q(ω)dν(dω)

###### .

ω∈Ω

If there exists a non-negative constant B such that for any z ∈ Ω, log QP((zz)) ≤ B, then we have

√

dTV(P,Q) ≤

2B.

Proof. We have the following relation between the total variation and the Hellinger distance (cf. Tsybakov, 2008, Lem. 2.3, Chapt. 2, p. 86):

  1 − H(P,Q)2/4

######    ≤ H(P,Q)2, (14)

dTV(P,Q)2 ≤ H(P,Q)2 ·

≥0

where the last inequality uses the positivity of the Hellinger distance. Inspired by the decomposition of the Hellinger distance in (Agarwal et al., 2020, Lem. 25), we have

2

###### H(P,Q)2 =

P(dω) − Q(dω)

P(dω) + Q(dω) − 2 P(dω) Q(dω)

###### =

ω∈Ω

ω∈Ω

- P(dω)

- Q(dω)

P(dω) Q(dω) = 2 · 1 −

Q(dω)

= 2 · 1 −

ω∈Ω

ω∈Ω

- P(dω)

- Q(dω)

q(ω)dν(dω) (by definition of Q(dω))

= 2 · 1 −

ω∈Ω

≤ −2log

It follows using Eq. (14)

ω∈Ω

- P(dω)

- Q(dω)

q(ω)dν(dω) (using 1 − x ≤ −log (x))

dTV(P,Q)2 ≤ H(P,Q)2

- P(dω)

- Q(dω)

q(ω)dν(dω) (by Jensen as −log is convex)

≤ 2

−log

ω∈Ω

- P(dω)

- Q(dω)

−log

≤ 2

q(ω)dν(dω)

ω∈Ω

- P(dω)

- Q(dω)

q(ω)dν(dω) (by Jensen as |·| is convex)

−log

≤ 2

ω∈Ω

- P(dω)

- Q(dω)

q(ω)dν(dω) (first part of Lemma F.6)

≤ 2

log

ω∈Ω

≤B

≤ 2B. (second part of Lemma F.6)

###### ≤ 2B

q(ω)dν(dω)

ω∈Ω

=1

This concludes both parts of the proof.

| |
|---|

The next lemma provides a lower bound on the softmax output if its input is upper-bounded (in ℓ1-norm).

Lemma F.7. Let x ∈ Rm be such that ∥x∥1 ≤ c1 for some c1 > 0. Then, we have

1 mexp(2c1)

softmax(u) ≥

,

where the inequality holds for each component of softmax(u).

Proof. Using the fact that

m

∥x∥1 =

|xi| ≤ c1,

i=1

we know that for any i ∈ [m], we have

###### −c1 ≤ xi ≤ c1.

Hence, using the fact that the exponential is increasing, we have for any i ∈ [m]

exp(−c1) ≤ exp(xi) ≤ exp(c1). (15) Summing and taking the inverse leads to

m

m

m

exp(−c1) ≤

exp(xj) ≤

exp(c1)

i=1

i=1

i=1

1

1

1

⇐⇒

m j=1 exp(c1) ≤

m j=1 exp(xj) ≤

.

m j=1 exp(−c1)

Combining Eq. (15) and Eq. (16) yields

(16)

exp(−c1) m j=1 exp(c1) ≤

exp(xi) m j=1 exp(xj) ≤

exp(c1)

.

m j=1 exp(−c1)

As we desire a lower bound, we only focus on the left-hand side of the previous inequality. Multiplying the numerator and denominator by exp(c1) leads to

1 mexp(2c1)

exp(xi) m j=1 exp(xj) ≥

∀i ∈ [m], softmax(x)i =

,

which concludes the proof. While we only need the lower bound of Eq. (16) to obtain Lemma F.7, both bounds can be used, for instance in Xie et al. (2024, Lemma E.7) and Veliˇckovi´c et al. (2024, Lemma 2.1) to show that, under a mild assumption on x ∈ Rm, softmax(x) behaves as O m 1 when m grows to infinity.

| |
|---|

Upper-bounding the total variation. We now proceed with finding an upper bound on the total variation between the true probability of the next token and the one estimated by the LLM fΘ. It will enable us to find the bounding vector c. The next lemma shows that the input of the softmax layer of the model is bounded.

Lemma F.8. Consider an LLM fΘ ∈ F. For any input sequence S ∈ Rr×n, the following inequality holds

1 nτ

1 τ ∥WU⊤∥2,1,

WUS(L) n∥1 ≤

∥

where τ is the temperature, WU is the unembedding matrix (which is bounded as stated in the definition of the parameters space W), and S(L) is the output of the last transformer layer.

Proof. We recall that the layer normalization ensures that at each layer, the tokens are in the unit ℓ2-ball. This is, in particular, the case for the output of the last layer S(L). It means that the columns of S(L) verifies

∀k ∈ [n], ∥S(·,kL)∥2≤ 1, (17) which implies

∥S(·,iL)∥2 ≤ 1. (18) Recalling that the Lp,q-norm of a matrix A ∈ Rn×m can be rewritten as

max

1≤k≤n

 

 

1 q

q p

m

n

= ∥(∥A·,j∥p)mj=1∥q,

|Aij|p

∥A∥p,q :=

j=1

i=1

which corresponds to

m

∥A·,j∥p (19)

∥A∥2,1 =

j=1

with (p,q) = (2,1). Hence, the ℓ1-norm of the last layer before the softmax layer satisfies

1 nτ

1 nτ

WUS(L) n∥1 =

∥

1 nτ

≤

1 nτ

≤

1 nτ

≤

T

i=1

T

r

n

1 nτ

Wij

Sjk =

i=1

j=1

k=1

r

n

###### WijSjk

j=1

k=1

T

r

n

|WijSjk| (triangular inequality)

i=1

j=1

k=1

T

n

###### Wi⊤S·,k

i=1

k=1

T

n

∥Wi∥2∥S·,k∥2 (Cauchy-Schwartz inequality)

i=1

k=1

T

n

1 nτ

≤

∥Wi∥2 max

∥S·,k∥2

1≤k≤n

i=1

k=1

T

1 nτ

∥S·,k∥2

≤

∥Wi∥2

n max

1≤k≤n

i=1

T

1 τ

∥Wi∥2

≤

i=1

T

1 τ

∥(WU⊤)·,i∥2

≤

i=1

1 τ ∥WU⊤∥2,1 (by Eq. (18) and the def. of L2,1 in Eq. (19))

≤

where we dropped the subscript and superscript on WU and S(L) to ease the notations. This concludes the proof.

| |
|---|

The previous lemma can be used to show that the logarithm of the ratio between the true probability of the next token and the one estimated by the LLM fΘ is upper bounded as a function of the vocabulary size T, the temperature, the upper-bound on WU and some constant related to the ambiguity of language (see Eq. (1)).

Proposition F.9 (Upper-bound on the logarithm). Consider an LLM fΘ ∈ F with vocabulary size T. We recall that BU is the upper bound on the norm of WU in the definition of parameter space W, τ is the softmax temperature and c0 is the constant related to the ambiguity of language (see Eq. (1)). We have

PL(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ B¯ = max{log (T) +

1 c0 }.

2BU τ

∀n ∈ [N], log

,log

Proof. The main idea of the proof is to bound the probability ratio and use the fact that log is non-decreasing. Let n ∈ [N]. The model fΘ receives as input sequences of tokens Sn of size n ≤ K. We first lower-bound each term of the probability ratio. From Eq. (1), we have

PL(Xn+1 | Sn) ≥ c0. (20) We want to obtain a similar inequality for PΘ(Xn+1 | Sn). As the parameters Θ of the LLM are in W, we know that ∥WU⊤∥2,1 ≤ BU. Lemma F.8 ensures that

1 nτ

1 τ ∥WU⊤∥2,1 ≤

BU τ

WUS(L) n∥1 ≤

∥

.

We can then apply Lemma F.7 with c1 = B

τ and given that nτ1 WUS(L) n ∈ RT, it leads to

U

1 nτ

1 T exp(2BU/τ)

WUS(L) n ≥

PΘ(· | Sn) = softmax

,

where the inequality holds for each component of PΘ(· | Sn). This is in particular the case for PΘ(Xn+1 | Sn) which is the entry we are interested in, i.e., we have

1 T exp(2BU/τ)

. (21)

PΘ(Xn+1 | Sn) ≥

Going back to the ratio of probability, consider the situation where we have

Then, using Eq. (21), we have

PL(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≥ 1.

PL(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤

1 PΘ(Xn+1 | Sn) ≤ T exp(2BU/τ),

1 ≤

which implies, as the log is non-decreasing monotonically,

PL(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ log (T exp(2BU/τ)) = log (T) +

2BU τ

. (22) Similarly, consider the case where we have

0 ≤ log

PL(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ 1.

Then, we have

PΘ(Xn+1 | Sn) PL(Xn+1 | Sn) ≥ 1,

and similarly to above, we can use Eq. (20) to obtain

PΘ(Xn+1 | Sn) PL(Xn+1 | Sn) ≤

1 PL(Xn+1 | Sn) ≤

1 c0

1 ≤

. This implies

PΘ(Xn+1 | Sn) PL(Xn+1 | Sn) ≤ log

1 c0

0 ≤ log

, which also rewrites

PL(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ log

1 c0

. (23) By definition of the absolute value, combining Eq. (22) and Eq. (23) leads to

0 ≤ −log

This concludes the proof.

log

PL(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ max{log (T) +

2BU τ

,log

1 c0 }.

| |
|---|

We are now ready to upper-bound the total variation.

Corollary F.10 (Upper-bound on the total variation). Consider an LLM fΘ ∈ F with vocabulary size T. We recall that BU is the upper bound on the norm of WU in the definition of parameter space W, τ is the softmax temperature and c0 is the constant related to the ambiguity of language (see Eq. (1)). For n ∈ [N], we have

2BU τ

1 c0 } := c2. (24)

dTV(PL(· | Sn),PΘ(· | Sn)) ≤ 2max{log (T) +

,log

Proof. Using Proposition F.9, we can directly apply Lemma F.6 with B = max{log (T)+ 2B

} for any n ∈ [N]. This leads to

τ ,log c 1

U

0

2BU τ

1 c0 }.

∀n ∈ [N], dTV(PL(· | Sn),PΘ(· | Sn)) ≤ 2max{log (T) +

,log

This concludes the proof.

| |
|---|

- F.5.3. CONCLUDING THE PROOF We are now ready to state our main result.

Theorem F.11 (Restatement of Theorem 4.2). Consider an LLM fΘ ∈ F with vocabulary size T. We denote by Γ the mixing matrix of the pretraining sequences of tokens (S1,...,SN

). Let δ > 0. Then, with probability at least 1 − δ,

train

B¯ √Ntrain

2 δ

Rpre(Θ) ≤ Rpre(Θ) +

log

,

where B¯ is a constant depending on the parameters of the problem. More precisely,

2BU τ

1 c0 }.

B¯ = 2∥Γ∥ max{log (T) +

,log

Proof of Theorem 4.2. By definition of the risk, we have

1 Ntrain

Rpre(Θ) =

Ntrain

dTV(PL(· | Sn),PΘ(· | Sn)) =gn(Sn)

=

n=1

1 Ntrain

Ntrain

gn(Sn)

n=1

) = f(S). Using Corollary F.10, we know that

= f(S1,...,SN

train

2BU τ

1 c0 } := c2.

|gn(Sn)| ≤ 2max{log (T) +

,log

By definition, each sequence of tokens Sn takes its values in Vn (again by abuse of notation, n = min{n,K}) and S takes its values in V1 × ... × VN

train. For any two sequences ζ,Σ with values in V1 × ... × VN

train, we have

  

  dTV(PL(· | ζn),PΘ(· | ζn))

Ntrain

1 Ntrain

−dTV(PL(· | Σn),PΘ(· | Σn))

f(ζ) − f(Σ) =

n=1

=gn(Σn)

=gn(ζn)

Ntrain

1 Ntrain

(gn(ζn) − gn(Σn))

=

n=1

Ntrain

1 Ntrain

n̸=Σn} (removing the zero terms)

(gn(ζn) − gn(Σn)) {ζ

=

n=1

Ntrain

1 Ntrain

(gn(ζn) − gn(Σn)) {ζ

≤

n̸=Σn}

n=1

Ntrain

1 Ntrain

≤

(gn(ζn) − gn(Σn)) {ζ

n̸=Σn}

n=1

Ntrain

1 Ntrain

|gn(ζn) − gn(Σn)| {ζn̸=Σn}

≤

n=1

  |gn(ζn)|

   {ζ

Ntrain

1 Ntrain

n̸=Σn} (Corollary F.10)

≤

+|gn(Σn)|

n=1

≤c2

≤c2

Ntrain

Ntrain

1 Ntrain

2c2 Ntrain {ζn̸=Σn}

≤

,

2c2 {ζ

n̸=Σn} =

n=1

n=1

where c2 = 2max{log (T) + 2B

train with all entries equal to 2c2

}. As ζ and Σ were taken arbitrary, choosing c ∈ RN

τ ,log c 1

U

0

Ntrain ensures that f verifies the condition in Proposition F.5, i.e.,

Ntrain

∀ζ,Σ, f(ζ) − f(Σ) ≤

###### cn {ζ

n̸=Σn}.

n=1

We assumed in Section 4.2 that the sequences Sn were related via a Marton coupling with mixing matrix Γ. Putting everything together, we can apply Proposition F.5 which leads to

∀u ≥ 0, P(|f(S) − ES[f(S)]| ≥ u) ≤ 2exp −2u2 ∥Γ∥2∥c∥22

. (25)

Let u ≥ 0. We have the following events ordering

(ES[f(S)] − f(S) ≥ u) ⊆ (ES[f(S)] − f(S) ≥ u) ∪ (f(S) − ES[f(S)] ≥ u) = (|f(S) − ES[f(S)]| ≥ u). Hence, as u was taken arbitrary and using Eq. (25), we have

∀u ≥ 0, P(ES[f(S)] − f(S) ≥ u) ≤ 2exp −2u2 ∥Γ∥2∥c∥22

.

We recall that by definition

f(S) = Rpre(Θ) and Rpre(Θ) = ES Rpre(Θ) . Since the previous inequality holds for any u ≥ 0, we can hence choose u such that

δ = 2exp −2u2 ∥Γ∥2∥c∥22

−2u2 ∥Γ∥2∥c∥22

δ 2 ⇐⇒ u2 =

- 1

- 2∥Γ∥2∥c∥22 log

⇐⇒

= log

2 δ

1 √2∥Γ∥∥c∥2 log

⇐⇒ u =

.

2 δ

Using the fact that

∥c∥2 =

Ntrain

c2n =

n=1

Ntrain

2

2c2 Ntrain

=

n=1

Ntrain

4c22 Ntrain2

=

n=1

4c22 Ntrain

2c2 √Ntrain

=

and using the fact that c2 = 2max{log (T) + 2B

τ ,log c 1

U

0

} from Corollary F.10, we obtain

u =

1 √2

2c2 √Ntrain ∥Γ∥ log

2 δ

=

√2c2 √Ntrain∥Γ∥ log

2 δ

2∥Γ∥ max{log (T) + 2B

τ ,log c 1

} √Ntrain

U

0

=

2 δ

log

=

B¯ √Ntrain

2 δ

log

,

where we define

2BU τ

1 c0 }.

B¯ = 2∥Γ∥ max{log (T) +

,log

Putting everything together, we have

B¯ √Ntrain

P Rpre(Θ) − Rpre(Θ) ≥

2 δ ≤ δ.

log

Taking the opposite event leads to the following inequality with probability at least 1 − δ

B¯ √Ntrain

2 δ

Rpre(Θ) ≤ Rpre(Θ) +

log

,

which concludes the proof.

| |
|---|

###### F.6. Proof of Corollary E.3

As the layer norm is not applied anymore, each token is no longer in the ℓ2-unit ball, and Lemma F.8 does not hold anymore. We want to provide an analogous lemma for our setting. We first prove the following technical lemmas.

Lemma F.12. The ReLU is a norm-decreasing operator, i.e., we have

∀A ∈ Rn×m, ∥ReLU(A)∥1,1 ≤ ∥A∥1,1, where the ReLU is applied entry-wise.

Proof. Recalling that ReLU(x) = max{0,x} is applied entry-wise, using the fact that |max{0,x}| ≤ |x| and considering A and A˜ = ReLU(A), we have

###### ∥A˜ ∥1,1 =

|A˜ i,j| =

i,j

|max{0,A˜ i,j}| ≤

i,j

|Ai,j| ≤ ∥A∥1,1,

i,j

which concludes the proof.

Lemma F.13. The L1,1-norm verifies the following property:

∀A ∈ Rn×m,B ∈ Rm×p, ∥AB∥1,1 ≤ n∥A∥∞∥B∥1,1.

Proof. We have

p

p

p

n

n

m

|(AB)ij| =

|

AikBkj| ≤

∥AB∥1,1 =

i=1

j=1

i=1

j=1

j=1

k=1

p

p

n

m

n

m

≤

|Aik||Bkj| ≤ max

|Aik|

ik

j=1

i=1

j=1

i=1

k=1

k=1

p

m

≤ n∥A∥∞

|Bkj| ≤ n∥A∥∞∥B∥1,1,

j=1

k=1

n

m

|AikBkj|

i=1

k=1

|Bkj|

which concludes the proof.

Lemma F.14. The L2,1 and L∞,1-norms verify the following relation

∀A ∈ Rn×m, ∥A∥∞,1 ≤ ∥A∥2,1.

Proof. By definition of the Lp,q-norm, we have

M

∥A∥∞,1 =

max

1≤i≤n

j=1

M

n

≤

j=1

i=1

|Aij| =

|A2ij| ≤

M

j=1

|A2ij| (as x → x2 is increasing)

max

1≤i≤n

M

∥A·,j∥2 ≤ ∥A∥2,1,

j=1

where the first inequality comes from adding non-negative terms.

| |
|---|

We are now ready to state the lemma analogous to Lemma F.8.

Lemma F.15. Consider an LLM fΘ ∈ F˜ with L layers. For any input sequence S ∈ Rr×n, the following inequality holds

1 nτ

c3 τ ∥WU⊤∥2,1,

WUS(L) n∥1 ≤

∥

where τ is the temperature and c3 is a constant depending on the parameters upper-bound. More precisely,

L

r3 H

· Btok.

c3 = (1 + rmB1B2) · 1 +

BOBV

WU is the unembedding matrix (which is bounded as stated in the definition of the parameters space W), and S(L) is the output of the last transformer layer.

Proof of Lemma F.15. Our model fΘ ∈ F˜ is given as input a sequence S ∈ Rr×n. With similar computations than in Lemma F.8, we have

1 nτ ∥WUS(L) n∥1 =

1 nτ

T

i=1

r

n

T

1 nτ

Wij

Sjk =

j=1

i=1

k=1

r

n

###### WijSjk

j=1

k=1

T

r

n

1 nτ

|WijSjk| (triangular inequality)

≤

i=1

j=1

k=1

T

n

T

n

1 nτ

1 nτ

Wi⊤S·,k ≤

∥Wi∥∞∥S·,k∥1 (H¨older inequality)

≤

i=1

i=1

k=1

k=1

T

n

1 nτ

1

nτ ∥WU⊤∥∞,1∥S(L)∥1,1 ≤

≤

∥Wi∥∞ ·

∥S·,k∥1 ≤

i=1

k=1

1

nτ ∥WU⊤∥2,1∥S(L)∥1,1, (Lemma F.14) where, again, we dropped the subscript and superscript on WU and S(L) to ease the notations. We obtain

1 nτ

1 nτ ∥WU⊤∥2,1∥S(L)∥1,1. (26)

WUS(L) n∥1 ≤

∥

As we do not use layer normalization, we want to find another way to bound S(L). To that end, we will first express S(ℓ), the output of the (ℓ)-th layer of the transformer, as a function of S(ℓ−1), the output of the (ℓ − 1)-th layer. Using the definition of the transformer model (see Appendix B), we have

 

Z(ℓ) = S(ℓ−1) + A S(ℓ−1);WQ(ℓ),WK(ℓ),WV(ℓ),WO(ℓ) , Y(ℓ) = W2(ℓ) ReLU W1(ℓ)Z(ℓ) , S(ℓ) = Z(ℓ) + Y(ℓ).



We will compute each layer’s L1,1-norm.

- Step 1: MHA. By definition, denoting the number of heads by H, we know that A S(ℓ−1);WQ(ℓ),WK(ℓ),WV(ℓ),WO(ℓ) ∈ Rr×n multiplies W(ℓ) ∈ Rr×r with the concatenation on the rows of the H softmax layers that each writes

/√r WV(ℓ)S(ℓ−1) ∈ R

⊤

r

softmax WQ(ℓ)S(ℓ) WK(ℓ)S(ℓ−1)

H×n,

We keep the notations ℓ without explicating the index of the head to ease notations. Denoting the concatenation on the rows by C(ℓ) ∈ Rr×n, we have

∥A S(ℓ−1);WQ(ℓ),WK(ℓ),WV(ℓ),WO(ℓ) ∥1,1 = ∥WO(ℓ)C(ℓ)∥1,1 ≤ r · ∥WO(ℓ)∥∞∥C(ℓ)∥1,1

≤ rBO∥C(ℓ)∥1,1. (definition of W˜ ) Moreover, by definition of C(ℓ), we have

r

r

|Cij(ℓ)| =

∥C(ℓ)∥1,1 =

j=1

i=1

r/H

r

H

|C(ijℓ,h)| =

j=1

i=1

h=1

H

∥C(ℓ,h)∥1,1, (27)

h=1

where C(ℓ,h) ∈ RHr ×n is the softmax matrix of the h-th layer. We recall that the softmax matrix is a row-stochastic matrix of RHr ×r so it has all values lower than 1. In the next computations, we drop the h index on the query, key, and value

matrices to ease the notations. Using Lemma F.13 on the softmax matrix and on the value matrix WV(ℓ) ∈ RHr ×r, we have

/√r WV(ℓ)S(ℓ−1) ∥1,1

⊤

∥C(ℓ,h)∥1,1 = ∥softmax WQ(ℓ)S(ℓ) WK(ℓ)S(ℓ−1)

/√r ∥∞ · ∥ WV(ℓ)S(ℓ−1) ∥1,1 ≤

⊤

r H · ∥softmax WQ(ℓ)S(ℓ) WK(ℓ)S(ℓ−1)

≤

r

H · ∥ WV(ℓ)S(ℓ−1) ∥1,1 (the softmax matrix is row-stochastic) ≤

2

r H ·

r H ∥(WV(ℓ)∥∞∥S(ℓ−1)∥1,1 ≤

r H

BV ∥S(ℓ−1)∥1,1. (definition of W˜ ) Combining the previous inequality with Eq. (27) leads to

r2 H

BV ∥S(ℓ−1)∥1,1. In the end, the multi-head attention norm verifies

∥C(ℓ)∥1,1 ≤

r3 H

∥A S(ℓ−1);WQ(ℓ),WK(ℓ),WV(ℓ),WO(ℓ) ∥1,1 ≤

BOBV ∥S(ℓ−1)∥1,1. Using the triangular inequality, we obtain

∥Z(ℓ)∥1,1 ≤ 1 +

r3 H

BOBV · ∥S(ℓ−1)∥1,1. (28)

- Step 2: FF. We recall that W1 ∈ Rm×r and W2 ∈ Rr×m. Using similar arguments to the above, we have

∥Y(ℓ)∥1,1 = ∥W2(ℓ) ReLU W1(ℓ)Z(ℓ) ∥1,1

≤ r · ∥W2(ℓ)∥∞∥ReLU W1(ℓ)Z(ℓ) ∥1,1 (Lemma F.13) ≤ r · ∥W2(ℓ)∥∞∥W1(ℓ)Z(ℓ)∥1,1 (Lemma F.12) ≤ r · m · ∥W2(ℓ)∥∞∥W1(ℓ)∥∞∥Z(ℓ)∥1,1 (Lemma F.13) ≤ rmB1B2∥Z(ℓ)∥1,1. (definition of W˜ )

- Step 3: output layer. Again, applying the triangular inequality and using the previous inequality and Eq. (28), we have ∥S(ℓ)∥1,1 ≤ ∥Z(ℓ)∥1,1 + ∥Y(ℓ)∥1,1 ≤ (1 + rmB1B2)∥Z(ℓ)∥1,1

r3 H

BOBV ∥S(ℓ−1)∥1,1.

≤ (1 + rmB1B2) 1 +

Iterating through the layers, recalling that S(0) = S, we finally obtain

∥S(L)∥1,1 ≤ (1 + rmB1B2) 1 +

r3 H

BOBV

L

∥S∥1,1,

where S is the input sequence. Combining this inequality with Eq. (26) leads to

1 nτ

WUS(L) n∥1 ≤ (1 + rmB1B2) 1 +

∥

r3 H

BOBV

L ∥S∥1,1 n

1 τ ∥WU⊤∥2,1 .

Using the fact that each token has a ℓ1-norm bounded by Btok. Hence, each column of S is too and we have

n

n

r

1 n∥S∥1,1 =

1 n

1 n

∥S·,j∥1 ≤Btok

≤ Btok.

|Sij|=

j=1

j=1

i=1

Combining the last two inequalities concludes the proof.

| |
|---|

We can now restate Corollary E.3.

Corollary F.16 (Restatement of Corollary E.3). Consider an LLM fΘ ∈ F˜ with vocabulary size T composed of L transformer blocks and H attention heads. We denote by Γ the mixing matrix of the pretraining sequences of tokens (S1,...,SN

). Let δ > 0. Then, with probability at least 1 − δ,

train

B¯ √Ntrain

2 δ

Rpre(Θ) ≤ Rpre(Θ) +

log

,

where B¯ is a constant depending on the parameters of the problem. More precisely,

2(BΘ)L τ

1 c0 },

B¯ = 2∥Γ∥ max{log (T) +

,log

3

H BOBV (BtokBU)1/L.

with BΘ = (1 + rmB1B2) 1 + r

Proof of Corollary E.3. We first note that the only change from Lemma F.8 to Lemma F.15 is the multiplicative constant c3 = (1 + rmB1B2) 1 + r

L

3

Btok in front of τ1∥WU⊤∥2,1. In particular, as we know that W˜ ⊂ W, we also have ∥WU⊤∥2,1 ≤ BU. Hence, we can apply the proof of Theorem 4.2 in a straightforward manner by changing BU

H BOBV

τ by c3 · B

τ . This concludes the proof.

U

| |
|---|

- F.7. Proof of Theorem 4.3 In this section, we detail the proof of Theorem 4.3. We first recall the problem setup.

Markov chains inputs. In this section, we give as input of the model a single Markov chain X = (X1,...,XN

) with finite, discrete state space Ω of size d with transition probability P. We assume the Xn are already tokenized and thus we have Ω ⊂ V. We denote the sequence of tokens the LLM receives by Sn = (X1,...,Xn) if n ≤ K and Sn = (Xn−K+1,...,Xn) otherwise due to the deletion process (see Definition B.2). In particular, the Sn are elements of VK∗ . We note that S = (S1,...,SN

icl

) is also a Markov chain (see Appendix F.7.1). By definition of P, we know that for any n ∈ [Nicl], the next token Xn+1 follows the distribution P(· | Sn). We assume that there exists a positive constant pmin that lower bounds all the transition probability between states, i.e., ∀n ∈ [Nicl],∀x,y ∈ Ω, P(Xn+1 = y | Xn = x) ≥ pmin > 0. This is akin to the ambiguity of language constant c0 considered in the previous section and in Hu et al. (2024); Wies et al. (2024); Xie et al. (2022); Zhang et al. (2023b).

icl

Next token probability distribution. An important difference with the setting considered in Theorem 4.2 is that here, we predict a probability distribution on the state space Ω of the Markov chain and not on the vocabulary of the LLM V. To that end, we restrict the predicted probability given the past tokens Sn to the state space Ω. Formally, denoting the output of the last layer of fΘ by S(L), the last layer before the softmax outputs a vector u = nτ1 WUS(L) n ∈ RT. We first extract

the entries of u whose index i are such that the i-th element of the vocabulary space V is in Ω. This can be formalized as follows. We denote by Id = (i1 ≤ i2 ≤ ... ≤ id) ∈ [T]d the subset of d distinct elements of [T] and consider the matrix Mj = e⊤i

, where ei

j ∈ RT has value 1 at entry ij ∈ I and 0 elsewhere. Extracting only the d entries of u that corresponds

j

to the state space yields a vector in Rd that writes v = nτ1 MWUS(L) n ∈ RT. Similarly to Appendix B, the probability distribution of next token Xn+1 provided by the LLM fΘ now writes

PΘ(· | Sn) = softmax

1 nτ

MWUS(L) n ∈ ∆d.

We aim to obtain a similar generalization bound than in Theorem 4.2 where the reference probability distribution is the Markov chain transition probability P instead of the probability distribution of language PL. In particular, P will replace PL in the definition of the risks in Eq. (3). We provide below an overview of the proof before detailing it.

Overview of the proof. We are going to use McDiarmid’s inequality for Markov chains of Paulin (2015, Corollary 2.11). To adapt their arguments to our setting, we bound the total variation between the true probability of the next token and the one estimated by the LLM. The rest of this section is organized as follows. First, in Appendix F.7.1, we show that S = (S1,...,SN

) is a Markov chain. Then in Appendix F.7.2, we adapt the concentration inequality of Paulin (2015, Corollary 2.11). Afterwards in Appendix F.7.3, we show how to bound the total variation between the true and the estimated probability of the next token. Finally Appendix F.7.4 concludes the proof.

icl

- F.7.1. CONNECTION BETWEEN TOKENS AND SEQUENCES OF TOKENS MARKOV CHAINS

We first show that S = (S1,...,SN

icl

) is also a Markov chain.

Lemma F.17. Consider a sequence (not necessarily a Markov chain) X = (X1,...,XN) with values in Ω and let Sn = (X1,...,Xn) if n < K and Sn = (Xn−K+1,...,Xn) otherwise. Then, the sequence S = (S1,...,SN) is a Markov chain with state space Ω∗K that contains the sequence of elements in Ω of length smaller than K.

Proof. By definition of the Sn, we know that they take values in Ω∗K. Let x1,...,xn+1 ∈ Ω. We first assume that n > K and denote si = (xn−K+1,...,xi). We have

P(Sn+1 = sn+1 | Sn = sn,...,Sn−K+1 = sn−K+1)

= P(Sn+1 = sn+1 | Xn = xn,...,Xn−K+1 = xn−K+1)

= P(Sn+1 = sn+1 | Sn = sn). (by definition of Sn) Similarly, we assume n < K and denote si = (x1,...,xi). We have

P(Sn+1 = sn+1 | Sn = sn,...,S1 = s1)

= P(Sn+1 = sn+1 | Xn = xn,...,X1 = x1)

= P(Sn+1 = sn+1 | Sn = sn). (by definition of Sn) Finally, for n = K, we denote si = (x1,...,xi) for i ≤ K and sK+1 = (x2,...,xK+1). We have

P(SK+1 = sK+1 | Sn = sn,...,S2 = s2)

= P(SK+1 = sK+1 | XK = xK,...,X1 = x1)

= P(SK+1 = sK+1 | SK = sK). (by definition of SK) This establishes the Markov property for S.

| |
|---|

- F.7.2. CONCENTRATION INEQUALITIES FOR MARKOV CHAINS We first state a concentration inequality for time-homogeneous Markov chains that will be used to obtain our final bound.

Proposition F.18 (McDiarmid’s inequality for time-homogeneous Markov chains). Let S := (S1,...,SN) be a Markov chain with value in a discrete, finite state space Ω and mixing time tmix(ε). Let tmin := inf0≤ε<1 tmix 2 ε · 21−−εε

2

. If f : Ω → R is such that there exists c ∈ RN satisfying

N

∀x,y ∈ Ω, f(x) − f(y) ≤

###### ci {x

i̸=yi},

i=1

then we have for any u ≥ 0,

P(|f(S) − ES[f(S)]| ≥ u) ≤ 2exp −2u2 ∥c∥22 · tmin

.

Proof. We recall that Corollary 2.11 of Paulin (2015) ensures that for such a function f, we have

P(|f(S) − E[f(S)]| ≥ u) ≤ 2exp −2u2 ∥c∥22 · τmin

, (29)

where τmin is defined as

2

2 − ε 1 − ε

τmin := inf

τ(ε)

,

0≤ε<1

with τ(ε) being the mixing time of a Markov chain without assuming time homogeneity (see Paulin (2015, Definition 1.4)). As in our case, we assume the time homogeneity, this inequality in Eq. (29) has to be adapted. Following Remark 1.5 of Paulin (2015), we notice that

∀ε ∈ [0,1], τ(2ε) ≤ tmix(ε) ≤ τ(ε). Let 0 ≤ ε < 1. Using the fact that 2−ε

2

> 0, the previous inequality ensures

1−ε

2

2

2 − ε 1 − ε

2 − ε 1 − ε

ε 2 ⇐⇒ τ(ε)

ε 2

τ(ε) ≤ tmix

≤ tmix

.

Taking the infimum on the left-hand side leads to

2

2

2 − ε 1 − ε

2 − ε 1 − ε

ε 2

≤ tmix

τmin = inf

τ(ε)

.

0≤ε<1

As we took ε arbitrary in [0,1), we can take the infimum on the right-hand side, which leads to

τmin ≤ tmin. As the function x → exp −2u

2

∥c∥22x is decreasing, we finally obtain

exp −2u2 ∥c∥22τmin ≤ exp −2u2 ∥c∥22tmin

Combining Eqs. (29) and (30) concludes the proof.

. (30)

| |
|---|

Similarly to Theorem 4.2, we want to apply Proposition F.18 to a function f that consists of sums of total variation. We investigate in the next section how to find the bounding vector c to apply Proposition F.18.

- F.7.3. FINDING THE BOUNDING VECTOR

We want to apply the same arguments as in the proof of Theorem 4.2 to find the bounding vector c. The only difference in terms of setting is the definition of the probability of the next token. Indeed, in our case, we apply an extraction matrix M ∈ Rd×T to recover the d states of the input Markov chain. We first prove the following technical lemma.

Lemma F.19. Let d ≤ T and consider a subset of d distinct elements of [T] that writes Id = (i1 ≤ i2 ≤ ... ≤ id) ∈ [T]d. We denote by M ∈ Rd×T the matrix with rows Mj = e⊤i

j ∈ RT has value 1 at entry ij ∈ I and 0 elsewhere. For any vector u ∈ RT, we have

, where ei

j

###### ∥Mu∥1 ≤ ∥u∥1.

Proof. By definition of the ℓ1-norm, we have

d

∥Mu∥1 =

k=1

T

d

|

Mklul| ≤

l=1

k=1

T

T

|Mklul| ≤

l=1

l=1

d

|ul|

|Mkl|.

k=1

Moreover, each column of M contains at most one non-zero entry (with value 1). Otherwise, it means that two ei

are identical (as they only have one non-zero entry with value 1, having it at the same position ensures their equality) which contradicts the fact that the ij where taken distinct. Hence, for all l, we have dk=1|Mkl| ≤ 1, which concludes the proof.

j

| |
|---|

We now prove a lemma analogous to Lemma F.8.

Lemma F.20. Let S ∈ Rr×n denote the entry of the LLM fΘ and S(L) denote the output of the last layer before the softmax. Let d ≤ T and consider a subset of d distinct elements of [T] that writes Id = (i1 ≤ i2 ≤ ... ≤ id) ∈ [T]d. We denote by M ∈ Rd×T the matrix with rows Mj = e⊤i

j ∈ RT has value 1 at entry ij ∈ I and 0 elsewhere. Then, the following inequality holds

, where ei

j

1 τ ∥WU⊤∥2,1.

1 nτ ∥MWUS(L) n∥1 ≤

Proof. Applying Lemma F.19 with the matrix M ∈ Rd and the vector nτ1 WUX(L) n ∈ RT leads to

1 nτ ∥MWUS(L) n∥1 ≤

1

nτ ∥WUX(L) n∥1. Applying Lemma F.8 concludes the proof.

| |
|---|

The previous lemma can be used to show that the logarithm of the ratio between the true probability of the next token and the one estimated by the LLM fΘ is upper bounded as a function of the number of states of the Markov chain d, the temperature τ, the upper-bound on WU and some constant related to the ambiguity of language (see Eq. (1)).

Proposition F.21 (Upper-bound on the logarithm). Consider an LLM fΘ ∈ F and an input Markov chain X = (X1,...,XN

) with d states. We recall that BU is the upper bound on the norm of WU in the definition of parameter space W, τ is the softmax temperature, and pmin is the constant related to the minimal transition probability between states. We have

icl

P(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ B¯ = max{log (d) +

1 pmin }.

2BU τ

∀n ∈ [N], log

,log

Proof. The main idea of the proof is to bound the probability ratio and use the non-decreasing monotonicity of the log. Let n ∈ [N]. The model fΘ receives as input sequences of tokens Sn of size n ≤ K. We first lower-bound each term of the probability ratio. By definition of pmin, we have

###### P(Xn+1 | Sn) = P(Xn+1 | Xn) ≥ pmin > 0, (31)

where we used the Markov property for the first equality. We want to obtain a similar inequality for PΘ(Xn+1 | Sn). As the parameters Θ of the LLM are in W, we know that ∥WU⊤∥2,1 ≤ BU. Lemma F.20 ensures that

1 nτ

1 τ ∥WU⊤∥2,1 ≤

BU τ

MWUS(L) T∥1 ≤

∥

.

We can then apply Lemma F.7 with c1 = B

τ and given that Tτ1 MWUS(L) T ∈ Rd, it leads to

U

1 nτ

1 dexp(2BU/τ)

MWUS(L) n ≥

PΘ(· | Sn) = softmax

,

where the inequality holds for each component of PΘ(· | Sn). This is in particular the case PΘ(Xn+1 | Sn) which is the entry we are interested in, i.e., we have

1 dexp(2BU/τ)

. (32)

PΘ(Xn+1 | Sn) ≥

Going back to the ratio of probability, consider the situation where we have

###### P(Xn+1 | Sn)

PΘ(Xn+1 | Sn) ≥ 1. Then, using Eq. (32), we have

P(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤

1

PΘ(Xn+1 | Sn) ≤ dexp(2BU/τ), which implies, as the log is non-decreasing monotonically,

1 ≤

P(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ log (dexp(2BU/τ)) = log (d) +

0 ≤ log

Similarly, consider the case where we have

###### P(Xn+1 | Sn)

PΘ(Xn+1 | Sn) ≤ 1. Then, we have

###### PΘ(Xn+1 | Sn)

P(Xn+1 | Sn) ≥ 1, and similarly to above, we can use Eq. (31) to obtain

2BU τ

. (33)

PΘ(Xn+1 | Sn) P(Xn+1 | Sn) ≤

1 P(Xn+1 | Sn) ≤

1 pmin

1 ≤

.

This implies

PΘ(Xn+1 | Sn) P(Xn+1 | Sn) ≤ log

1 pmin

0 ≤ log

, which also rewrites

P(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ log

1 pmin

. (34) By definition of the absolute value, combining Eq. (33) and Eq. (34) leads to

0 ≤ −log

P(Xn+1 | Sn) PΘ(Xn+1 | Sn) ≤ max{log (d) +

1

2BU τ

pmin }. This concludes the proof.

log

,log

| |
|---|

We are now ready to upper-bound the total variation.

Corollary F.22 (Upper-bound on the total variation). Consider an LLM fΘ ∈ F and an input Markov chain X = (X1,...,XN

) with d states. We recall that BU is the upper bound on the norm of WU in the definition of parameter space W, τ is the softmax temperature, and pmin is the constant related to the minimal transition probability

icl

between states. We have

2BU τ

1 pmin } := c4. (35)

∀n ∈ [N], dTV(P(· | Sn),PΘ(· | Sn)) ≤ 2max{log (d) +

,log

Proof. Using Proposition F.21, we can directly apply Lemma F.6 with B = max{log (d) + 2B

} for any n ∈ [N]. It leads to

τ ,log p 1

U

min

2BU τ

1

∀n ∈ [N], dTV(P(· | Sn),PΘ(· | Sn)) ≤ 2max{log (d) +

pmin }. This concludes the proof.

,log

| |
|---|

- F.7.4. CONCLUDING THE PROOF We are now ready to state our main result.

Theorem F.23 (Restatement of Theorem 4.3). Consider an LLM fΘ ∈ F. We provide as input of fΘ a d−state Markov chain X = (X1,...,XN

). S is also a Markov chain, and we denote by tmix(ε) its mixing time. Let tmin := inf0≤ε<1 tmix 2 ε · 21−−εε

). The sequence of subsequences of the first n terms is denoted by S = (S1,...,SN

icl

icl

2

. Let δ > 0. Then, with probability at least 1 − δ,

2 δ

tmin Nicl

{ Ricl(ϑ) + K(ϑ,Θ)} + B¯

Ricl(Θ) ≤ inf

log

,

ϑ∈Wmc

where B¯ is a constant depending on the parameters of the problem. More precisely,

2BU τ

1 pmin }.

B¯ = 2 max{log (d) +

,log

Proof. Let ϑ ∈ Wmc. We first benefit from the metric properties of the total variation to decompose the risk.

Nicl

1 Nicl

###### ES

[dTV(P(· | Sn),PΘ(· | Sn))]

Ricl(Θ) =

n

n=1

Nicl

1 Nicl

###### ES

[dTV(P(· | Sn),Pϑ(· | Sn)) + dTV(Pϑ(· | Sn),PΘ(· | Sn))]

≤

n

n=1

Nicl

1 Nicl

###### ES

[dTV(P(· | Sn),Pϑ(· | Sn))]

≤

n

n=1

Nicl

1 Nicl

###### ES

[dTV(Pϑ(· | Sn),PΘ(· | Sn))]

+

n

n=1

≤ Ricl(ϑ) + K(ϑ,Θ). (36)

By definition of the risk, we have

1 Nicl

Ricl(ϑ) =

Nicl

dTV(P(· | Sn),Pϑ(· | Sn)) =gn(Sn)

=

n=1

1 Nicl

Ntrain

gn(Sn) = f(S1,...,SN

icl

n=1

###### ) = f(S).

Using Corollary F.22, we know that

2BU τ

1

pmin } := c4. Similarly to Theorem 4.2, and using the fact that S = (S1,...,SN

|gn(Sn)| ≤ 2max{log (d) +

,log

) is a Markov chain, we can show that choosing c ∈ RN

icl

icl with all entries equal to 2c4 Nicl ensures that f verifies the condition in Proposition F.5, i.e.,

Nicl

∀S,Σ, f(S) − f(Σ) ≤

###### cn {S

n̸=Σn}.

n=1

Putting everything together, we can apply Proposition F.18 which leads to

∀u ≥ 0, P(|f(S) − ES[f(S)]| ≥ u) ≤ 2exp −2u2 tmin∥c∥22

. (37)

Let u ≥ 0. We have the following events ordering

(ES[f(S)] − f(S) ≥ u) ⊆ (ES[f(S)] − f(S) ≥ u) ∪ (f(S) − ES[f(S)] ≥ u) = (|f(S) − ES[f(S)]| ≥ u). Hence, as u was taken arbitrary and using Eq. (37), we have

∀u ≥ 0, P(ES[f(S)] − f(S) ≥ u) ≤ 2exp −2u2 tmin∥c∥22

.

We recall that by definition

f(S) = Ricl(ϑ) and Ricl(ϑ) = ES Ricl(ϑ) .

Moreover, the inequality on the probability holds for any u ≥ 0, we can choose u such that

δ = 2exp −2u2 tminc∥22

−2u2 tmin∥c∥22

δ 2 ⇐⇒ u2 =

- 1

- 2

tmin∥c∥22 log

⇐⇒

= log

√tmin∥c∥2 log

1 √2

2 δ

⇐⇒ u =

.

2 δ

Using the fact that

∥c∥2 =

Nicl

c2n =

n=1

Nicl

2

2c4 Nicl

=

n=1

Nicl

4c24 Nicl2

=

n=1

4c24 Nicl

2c4 √Nicl

=

.

Using the fact that c4 = 2max{log (d) + 2B

τ ,log p 1

U

min

u =

√tmin log

1 √2

2c4 √Nicl

} (Corollary F.22), we obtain

√2c4 √Nicl

√tmin log

2 δ

2 δ

=

2√tmin max{log (d) + 2B

τ ,log p 1

} √Ntrain

U

min

=

2 δ

log

= B¯

tmin Nicl

2 δ

log

,

where we define

2BU τ

1 pmin }.

B¯ = 2 max{log (d) +

,log

Putting everything together, we have

P Ricl(ϑ) − Ricl(ϑ) ≥ B¯

tmin Nicl

2 δ ≤ δ.

log

Taking the opposite event leads to the following inequality with probability at least 1 − δ

√tmin √Nicl

2 δ

Ricl(ϑ) ≤ Ricl(ϑ) + B¯

log

.

Going back to the decomposition of the risk in Eq. (36) and rearranging the terms, we obtain

√tmin √Nicl

Ricl(Θ) ≤ Ricl(ϑ) + K(Θ,ϑ) + B¯

2 δ

log

.

As the left-hand side and the bound function of B¯ do not depend on ϑ, we can put them both on the left side of the inequality and then take the infimum on ϑ. Rearranging the terms to keep only Ricl(Θ) on the left side of the inequality leads to

2 δ

tmin Nicl

{ Ricl(ϑ) + K(ϑ,Θ)} + B¯

Ricl(Θ) ≤ inf

log

,

ϑ∈Wmc

which concludes the proof.

| |
|---|

