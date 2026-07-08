# arXiv:2308.06912v3[cs.LG]20Feb2024

## CAUSALLM IS NOT OPTIMAL FOR IN-CONTEXT LEARNING

Nan Ding Tomer Levinboim Jialin Wu Sebastian Goodman Radu Soricut Google Research {dingnan,tomerl,jialinwu,seabass,rsoricut}@google.com

ABSTRACT

Recent empirical evidence indicates that transformer based in-context learning performs better when using a prefix language model (prefixLM), in which incontext samples can all attend to each other, compared to causal language models (causalLM), which use auto-regressive attention that prohibits in-context samples to attend to future samples. While this result is intuitive, it is not understood from a theoretical perspective. In this paper we take a theoretical approach and analyze the convergence behavior of prefixLM and causalLM under a certain parameter construction. Our analysis shows that both LM types converge to their stationary points at a linear rate, but that while prefixLM converges to the optimal solution of linear regression, causalLM convergence dynamics follows that of an online gradient descent algorithm, which is not guaranteed to be optimal even as the number of samples grows infinitely. We supplement our theoretical claims with empirical experiments over synthetic and real tasks and using various types of transformers. Our experiments verify that causalLM consistently underperforms prefixLM in all settings.

1 INTRODUCTION

Transformer-based models (Vaswani et al., 2017) have become the default foundational model for various machine learning applications such as natural language processing (Devlin et al., 2018; Brown et al., 2020; Chowdhery et al., 2022) and computer vision (Dosovitskiy et al., 2020). Beyond their traditional usage in machine learning applications, it has recently been discovered that pretraining large transformers on a vast amounts of data leads them to develop a striking ability referred to as in-context learning (ICL) (Brown et al., 2020). Specifically, once such pretraining is complete, these models are able to solve new tasks at inference time (without changing their parameters) by simply ingesting a short sequence (prefix) of labeled examples from a task and then computing a prediction for a query example.

The ICL capability was first demonstrated by GPT-3 (Brown et al., 2020), where a causalLM (a Transformer decoder with auto-regressive attention masks) was used as the main model architecture. However, follow up work empirically found that restricting the auto-regressive masks on the entire sequence is too prohibitive and therefore proposed the so-called prefixLM (Raffel et al., 2020b; Tay et al., 2022) which allows full attention within the prefix tokens. Moreover, the latest models (such as PaLM2 (Google et al., 2023)) adopt a mixture of different LM objectives during pretraining to achieved state-of-art performance across a diverse set of tasks and capabilities.

However, beyond the few empirical results in those and related papers, there is yet no theoretical explanation that accounts for the different ICL behavior of prefixLM and causalLM. Indeed, theoretical studies of ICL are difficult due to the complicated non-linearity of the (ordinary) transformer architecture. However, recent work (Von Oswald et al., 2023) focusing on ICL of linear regression was able to show that a specifically designed parameter construction of a one-layer Linear Self-Attention (LSA) transformer can simulate a single step of gradient descent by using the in-context examples as training data. Moreover, a different recent study (Zhang et al., 2023) used gradient flow to prove that a randomly initialized LSA-transformer indeed converges to such a construction during training.

In this paper, we continue the theoretical line of work above by investigating the convergence properties of ICL for both prefixLM and causalLM multi-layer LSA-transformers in a linear regression setting. We summarizes our contributions as follows:

- • We first present a clear, formal proof that establishes the relationship between a multi-layer LSA and multi-step gradient descents in linear regression.
- • We then show that both causalLM and prefixLM based multi-layer LSA-transformers converge to their respective stationary points with linear rates of convergence. We prove that the stationary point of prefixLM corresponds to the optimal least square solution of the linear regression problem, while the stationary points of causalLM correspond to the weights obtained along the iterations of online gradient descent with non-decaying step sizes. Importantly, the stationary points obtained by causalLM may not become optimal even as the number of in-context examples increases, which indicates that causalLM is not optimal for in-context learning.
- • Finally, we verify the above theoretical insights by conducting experiments with LSAtransformers as well as ordinary softmax attention based transformers on various synthetic tasks including linear and non-linear regression, and multiclass classifications. We also compare causalLM and prefixLM ICL based on LLMs including T5 (Roberts et al., 2022) and PaLM2 (Google et al., 2023), as well as the multimodal model PaLI-X (Chen et al., 2023). Our experimental results support our theoretical findings and consistently show the superiority of prefixLM over causalLM on ICL for such settings.

- 2 BACKGROUND

We begin by reviewing a few types of transformer attention and in-context learning (ICL), as well as a specific transformer construction for linear regression ICL by (Von Oswald et al., 2023) which our theories will be based on. The discussions of other related work are deferred to Appendix A.

- 2.1 TRANSFORMERS: SSA, LSA, CAUSALLM, AND PREFIXLM

Given a sequence of input vectors Z = (z1,...,zn), the output of standard Softmax Self-Attention (SSA) layer is

##### zj ← zj +PV Zsoftmax(Z⊤ K⊤ Qzj),

where P,V,K,Q respectively corresponds to the output projection, value transformation, key transformation and query transformation.

Since the softmax attention of standard transformers is non-linear, its theoretical analysis becomes complicated even for a single layer. For this reason, theoretical approaches to analyze transformers have often resorted to the Linear Self-Attention (LSA) layer (Von Oswald et al., 2023; Zhang et al., 2023), which simply drops the softmax function from the attention,

n

zj ←zj +PV Z(Z⊤ K⊤ Qzj) = zj +PV

##### zi z⊤i K⊤ Qzj . (1)

i=1

Furthermore, since each input zj can attend to all positions j ∈ {1...n}, this form of attention is categorized as full (or bidirectional) attention, and is typically used in the transformer encoder.

On the other hand, a (linear) transformer decoder uses the auto-regressive attention

j

##### zi z⊤i K⊤ Qzj . (2)

zj ← zj +PV

i=1

which restricts each token zj to attend only to previous positions (and itself) from {1...j}. This restriction is due to the role of the decoder as a causal language model (causalLM) which predicts the next token in the context of the previously generated ones.

The original transformer involves both a full attention based encoder and an auto-regressive attention based decoder. However, prominent NLP research has often chosen either encoder-only (e.g.

BERT (Devlin et al., 2018)) or decoder-only (e.g. GPT (Brown et al., 2020), PaLM (Chowdhery et al., 2022)) models according to the task at hand. This is partially for the purpose of halving the parameter sizes.

Another version of attention, between full and auto-regressive, followed from the observation that some tasks can benefit from a prefix sequence such as context or prompt. That is, the input sequence Z is composed of n′ prefix tokens (z1,...,zn′) configured for the task, while the tokens (zn′+1,...,zn) represent the sample. Specifically, prefixLM (Raffel et al., 2020b) suggests the following attention (in its LSA version):

max(j,n′)

##### zi z⊤i K⊤ Qzj ,

zj ← zj +PV

i=1

where max(j,n′) ensures each prefix token zj with j < n′ can attend to all prefix tokens.

- 2.2 IN-CONTEXT LEARNING

A formal framework of in-context learning has been described in various existing literature such as (Garg et al., 2022; Zhang et al., 2023). Here, we briefly review the problem setting and introduce notation that will be used across the paper.

In-context learning refers to the ability of models to produce context-driven predictions at inference time. That is, at inference time, a model is fed with a sequence consisting of input-label pairs and a query input (x1,y1,...,xn,yn,xquery) and its goal is to predict the label yquery of xquery using the context examples (x1,y1,...,xn,yn) (specifically, without changing the model parameters).

- 2.3 LINEAR REGRESSION IN-CONTEXT LEARNERS

Linear regression is a classical machine learning problem. Given a set of input-label pairs (xi,yi), the goal is to find an optimal weight vector w that minimizes the l2-loss:

n

- 1

- 2n

∥w xi −yi∥22.

L(w) =

i=1

The gradient of the loss is ∇wL = n1 ni=1(w xi −yi)x⊤i , and a gradient descent algorithm with step size η follows the update rule:

n

η n

w(l) =w(l−1) +

(yi − w(l−1) xi)x⊤i . (3)

i=1

Using linear regression as a lens to study in-context learning was first proposed in (Garg et al., 2022), where the authors laid out an approach for training transformers to in-context learn a class of simple predictors, including linear regression. However, no theoretical study was provided. More recently, and most relevant to our work, (Von Oswald et al., 2023) proposed a succinct construction that demonstrates how a single LSA layer can effectively implement a single gradient descent step. According to their setup the input is formulated as

xj yj

Z = (z(0)1 ,...,z(0)n ), where z(0)j =

(4) and the parameter matrices of (1) are set as:

η n

0d×d 0 w(0) −1

Id×d 0 0 0

I, (5)

K = Q =

,V =

,P =

where w(0) is an initial weight vector. (Von Oswald et al., 2023) then showed that this configuration results in an update of their so-called transformed target yj ← yj + η (∇w(0)L)xj, and that this target update is equivalent to the one performed by a single-step gradient descent of linear regression. Although the construction of (Von Oswald et al., 2023) connected LSA-based ICL to the gradient descent of linear regression, the "transformed target" view seems unnatural* to work with. Moreover, their extension from single-layer to multi-layer LSA is unfortunately unclear.

*The traditional ML formulation updates the weight vector or the model prediction, while the groundtruth target remains fixed.

- 3 MULTI-LAYER IN-CONTEXT LEARNER

In this section, we provide a formal proof that a multi-layer LSA under the construction of (Von Oswald et al., 2023) progresses identically to multi-step gradient descent.

Instead of the "transformed target" view, the following proposition explicitly connects the GD weights of (3) to the outputs of the multi-layer LSA under the constructions of K, Q, P and V in (5). Note that we keep w(0) = 0 in the proposition because it simplifies the equations and makes the outputs more meaningful. However, such specification is not mandatory, and we provide general propositions, for arbitrary w(0), in Appendix C.

- Proposition 1 For a multi-layer LSA satisfying the construction (5) and with w(0) = 0, if its input Z

is formatted as (4), then its l-th layer output is z(jl) = (x⊤j ,δj(l))⊤, where δj(l) = yj − w(l) xj and w(l) is the l-th updated weight from the gradient descents update rule in (3).

Proof Sketch: Plugging in K, Q, P and V of (5) with w(0) = 0 and z(jl) = (x⊤j ,δj(l))⊤ into (1), we obtain that for all l > 0,

n

##### xj

xj δj(l)

0 δi(l−1)

η n

x⊤i xj .

δj(l−1) −

=

i=1

Since zj never changes its first d-dimension corresponding to xj, we can simplify it and focus only on δj(l), which is the last output coordinate of the j-th LSA-layer,

n

η n

δj(l) = δj(l−1) −

δi(l−1) x⊤i xj, (6)

i=1

with δj(0) = yj. Defining y˜j(l) = yj − δj(l) and rearranging (6), we obtain y˜j(0) = 0 and ∀l > 0:

n

η n

y˜j(l) = y˜j(l−1) +

(yi − y˜i(l−1))x⊤i xj . (7)

i=1

Finally, using (7) and the fact that y˜j(0) = 0 = w(0) xj, it can be proved by induction that ∀l : y˜j(l) = w(l) xj. A complete proof is provided in Appendix B.

To summarize, the newly introduced variable y˜j(l) is exactly the prediction of the l-th gradient descent weights w(l) for xj , and δj(l) is the difference between the true label yj and the predicted y˜j(l). Therefore, y˜j(l) serves as a bridge to connect the LSA output δj(l) and the GD weight w(l).

So far, we have dealt with the behavior of LSA layers with full attention. In what follows, we move on to the practical setting of in-context learning, where the input contains not only n in-context

(training) examples in the format of (4), but also an additional (test) query z(0)query = (x⊤query,0)⊤. In particular, we will focus on the two common ICL variants: prefixLM and causalLM, each with a different type of attention.

- 3.1 PREFIXLM ICL

A prefixLM ICL treats the in-context examples Z as the prefix and uses full attention on the first n positions, so that they can each freely attend to each other. The last query vector zquery can also attend to any example in Z, but cannot attend to itself†. As a result, the updates of the prefixLM-ICL under the same construction follow (6), with the outputs of the l-th layer being,

δj(l) = yj − y˜j(l) = yj − w(l) xj,

and δquery(l) = −y˜query(l) = −w(l) xquery,

†This is because the query does not contain a meaningful label. Attending to itself would cause it to include its last-dim input as a label, which would contaminate the resulting multi-layer prediction. This observation was not considered in (Von Oswald et al., 2023).

###### → 0 →

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

- Figure 1: The inputs/outputs of a multi-layer in-context learner. We omitted xj and xquery since they are unchanged.

where the initial y˜j(0) = y˜query(0) = 0.

Intuitively, the dynamics of the prefixLM ICL is as follows: all y˜j(l) starts as 0 at l = 0, and gradually approach to the true label yj as l increases, so that the difference (also as the output) δj(l) gradually approaches to 0. At the same time, δquery(l) starts at 0, and gradually approaches to −yquery, the negation of the query label. Figure 1 provides an illustration of these dynamics.

- 3.2 CAUSALLM ICL

A causalLM applies auto-regressive attention throughout the entire sequence. Therefore, plugging the same K, Q, P, V into (2), the update rules of (6) and (7) become:

j

η n

δj(l) = δj(l−1) −

δi(l−1) x⊤i xj, (8)

i=1

j

η n

y˜j(l) = y˜j(l−1) +

(yi − y˜i(l−1))x⊤i xj (9)

i=1

‡with δj(l) = yj −y˜j(l). Moreover, since different δj,y˜j are exposed to different ranges of inputs, there is no uniform w as in (3) that is associated with all y˜j. Instead, if we define wj for each different

position j with w(0)j = 0 and

j

η n

(yi − w(il−1) xi)x⊤i (10) then we have the following proposition:

w(jl) = w(jl−1) +

i=1

- Proposition 2 For a multi-layer causalLM-LSA satisfying (5) with w(0) = 0, if its input Z is

formatted as (4), then its l-th layer output is z(jl) = (x⊤j ,δj(l))⊤, where δj(l) = yj − w(jl) xj and w(jl) follow (10).

The proof of Proposition 2 is provided in Appendix B. Similar to prefixLM-ICL, causalLM-ICL also has y˜j(l) = w(jl) xj, and

δquery(l) = −y˜query(l) = −w(nl) xquery .

In summary, causalLM-ICL and prefixLM-ICL are associated with different update rules: w(jl) follows (10) while w(l) follows (3). Specifically, in causalLM, it can be seen that the w(il−1)

‡There is another way of update which changes η/n to η/j for the j-th example. We provide more details in Appendix D and show it performs worse than the main version in (8).

corresponding to the first positions are biased due to restricted access to only a few data points and furthermore, that these biases are propagated to later positions by (10). In prefixLM on the other hand, each position has access to all the data and a single w(l) can be used across the entire sequence as in (3). Although Eq. (3) and Eq. (10) only hold for the structured LSA case, the profound difference between causalLM and prefixLM stems from their architectural difference and therefore we believe extends to general transformers, as indicated by our experimental results in Section 5.

- 4 CONVERGENCE OF THE MULTI-LAYER IN-CONTEXT LEARNERS

In this section, we prove that both multi-layer prefixLM and causalLM converge to their respective stationary points with increasing layers (and with linear rates). In addition, we show that the stationary point of prefixLM corresponds to the optimal least-square solution of the linear regression problem, while the ones corresponding to causalLM are equivalent to the iterative weights of online gradient descent of linear regression, which are known to be sub-optimal for a limited number of examples.

- 4.1 CONVERGENCE OF THE PREFIXLM ICL

The fact that a multi-layer prefixLM computation exactly follows the update rule of w(l) as in (3), implies that the layer outputs of prefixLM have the same dynamics of multi-step gradient descent on a linear regression problem. The convergence properties of such dynamics are well-known, and are stated in the following proposition:

Proposition 3 If w(l) follows the iterative updates of (3), then there exists a stationary point w∗ with coefficients satisfying:

y X⊤ = w∗ XX⊤,

where y = (y1,...,yn) and X = (x1,...,xn). Furthermore, the iterative weights w(l) converge to w∗ with a linear rate of convergence:

w(l) −w∗ = (w(l−1) −w∗)(I−

η n

XX⊤).

That is, Proposition 3 holds for the multi-layer prefixLM, so that the same exact w∗ is also the stationary point of prefixLM, to which it converges in a linear rate. Furthermore this stationary point is exactly the (optimal) least square solution of the linear regression problem.

- 4.2 CONVERGENCE OF THE CAUSALLM ICL

Following the update rule of (10), we can view a multi-layer causalLM as implicitly maintaining different weight vectors wj for each position j. In what follows, we show that: (a) Each such position j has its own stationary point w∗j, which appears to be different from the global optimal point w∗ of linear regression; (b) even when the number of in-context samples n grows to infinity, convergence to w∗ is not guaranteed.

Specifically, in Appendix B we provide a proof for the following proposition:

- Proposition 4 If w(jl) = ji=1 a(i,jl) x⊤i follows the iterative updates of (10), then ai,j(l) = a(i,il) ≡ a(il) ∀j ≥ i,

and there exist stationary points w∗j = ji=1 a∗i x⊤i (for j ∈ 1,...,n) with coefficients from a∗ = (a∗1,...,a∗n) that satisfy y = a∗ T, where





x⊤1 x1 x⊤1 x2 ··· x⊤1 xn 0 x⊤2 x2 ··· x⊤2 xn

.

T =

 

 

... .

. .

0 0 ··· x⊤n xn

Furthermore, the coefficients a(l) converges to the stationary point a∗ with linear rate of convergence:

η n

a(l) −a∗ = (a(l−1) −a∗)(I−

##### T).

This proposition implies that the stationary points w∗j of causalLM-ICL are different from w∗, the least square solution of linear regression. However, a natural question is: if j increases, would w∗j ultimately converge to the optimal solution?

To answer this question, the next proposition shows that the stationary points w∗j follow an online gradient descent algorithm, whose loss and gradient at the j-th step is,

- 1

- 2

(wj xj+1 −yj+1)2, ∇wj

Lj(wj) =

Lj(wj) = (wj xj+1 −yj+1)x⊤j+1 .

- Proposition 5 Assuming that w∗j is the stationary points obtained in Proposition 4, then

1 ∥xj+1 ∥22

###### Lj(w∗j). (11)

w∗j+1 = w∗j −

∇w∗

j

The proof of Proposition 5 is provided in Appendix B. Note that online gradient descent is known to converge to an optimal solution only with a decaying step size j−ν for ν > 0 (Jentzen & Von Wurstemberger, 2020). Since the step size of (11) does not decay, we conclude that causalLM may not converge to w∗ even with increasing layers and increasing number of in-context examples. More concretely, as for the case of in-context learning, where the number of in-context examples n is limited, convergence to the optimal solution w∗ cannot be achieved by causalLM-ICL.

- 5 NUMERICAL EXPERIMENTS Our experiments contain three parts.

- • We first use LSA-transformers on linear regression problems to directly verify our theorems. In Section 5.1, we show that despite that the in-context example (training) error of causalLM and prefixLM both decays in linear rates, the query (test) error of causalLM is significantly larger, which indicates its stationary solution is not optimal.
- • Secondly, we use ordinary softmax transformers on a few synthetic tasks including linear regression, nonlinear regression and multiclass classification. In Section 5.2, we show that our theoretical insights generalize to other tasks types (i.e., that ICL prefixLM still outperforms causalLM in all these cases).
- • Lastly, in Section 5.3, we conduct LLM based ICL experiments using T5 (Roberts et al., 2022). We also provide additional experimental results on PaLM2 (Google et al., 2023) as well as large multimodal models (PaLI-X, Chen et al. (2023)) in Appendix E.6 and E.7.

- 5.1 LSA-TRANSFORMERS ON LINEAR REGRESSION

In order to directly verify our theorems from Section 4, we first study in-context learning on linear regression problem with the LSA transformer of (5). Each of the input sequence contains 40 incontext examples and 200 queries, and each query attends to all the in-context examples but does not attend to each other. See Appendix E for an illustration. The data input xi of the sequence is sampled from U(−1,1)16. Each sequence is associated with a single weight vector w that is sampled from

N(0,I), and the labels are computed as yi = w xi. Assuming the prediction of each layer is y˜i(l), we evaluate the MSE ∥yi − y˜i(l)∥22 on both in-context and query examples across different layers l.

The results are plotted in Figure 2 left (for prefixLM) and middle (for causalLM). Our results are averaged over 64 randomly generated sequences. As we can see, although both prefixLM and causalLM has a linear rate of convergence (with respect to the number of layers) on the in-context examples, the query errors of causalLM are stuck above the 10−1 level, while the query error of prefixLM decays in the same linear rate as its training error.

Furthermore, in Figure 2 right, we plot the query errors of the stationary points (following Proposition 4, corresponding to the outputs of infinite layers) of causalLM-ICL with increasing number of in-context examples up to 300. Although causalLM-ICL is able to eventually converge to optimal solution when µx = 0, it takes more than 100 examples to reach below 10−2. The convergence is

Convergence of multi-layer prefixLM-ICL

|In-context examples<br><br>Query examples| | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

100

10 1

10 2

MSEError

10 3

10 4

10 5

0 10 20 30 40 50 Number of layers

Convergence of multi-layer causalLM-ICL

|In-context examples<br><br>Query examples| | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

100

10 1

MSEError

10 2

10 3

10 4

10 5

0 10 20 30 40 50 Number of layers

Stationary points of causalLM-ICL

101

10 1

QueryMSEerror

10 3

10 5

- x = 0

- x = 1

- x = 2

- x = 3

10 7

50 100 150 200 250 300 Number of training exemplars

- Figure 2: Left/Middle: the MSE on in-context examples and query examples of multi-layer LSAbased prefixLM/causalLM-ICLs with 40 in-context training examples. Right: the query MSE of causalLM-ICL’s stationary points (per Proposition 4) using up to 300 in-context examples.

| |LR N-LR MC|
|---|---|
|PrefixLM-SL CausalLM-SL<br><br>|8.6e-3 1.5e-4 24.1 1.9e-1 2.7e-3 27.0|
|PrefixLM-UL CausalLM-UL<br><br>|2.5e-3 9.0e-5 27.6 1.6e-2 2.9e-3 32.1|

- Table 1: The test query errors of the unshared-layer (UL) and sharing-layer (SL) transformer-ICLs on linear regression (LR), non-linear regression (NLR), and multiclass classification (MC) tasks. Both regression tasks report mean squared errors; and the MC task reports the classification error.

even worse as we vary the input distribution x ∼ U(−1,1)d + µx with increasing µx ∈ {0,1,2,3}, which demonstrates that causalLM-ICL is not optimal for few-shot ICL.

- 5.2 STANDARD TRANSFORMERS ON SYNTHETIC TASKS

Previous experiments provided a proof of concept verification of the propositions from Section 4. Next we verify if a standard softmax transformer-based prefixLM and causalLM ICL exhibit similar differences on various types of synthetic tasks including linear regression, non-linear regression and multiclass classification.

All three tasks used 16-dim inputs with x ∼ U(−1,1)16 and w ∼ N(0,I). For non-linear regression, we apply a sigmoid activation on the logit such that y = sigmoid(w x); and for multiclass classification, we randomly generate three wc ∼ N(0,I), and assign labels based on y = argmaxc {wc x}. We trained a few 24-layer transformers containing 128 hidden units and 2 heads. Besides of the comparisons of prefixLM and causalLM, we also compare the transformers with or without sharing layers (SL vs UL). In particular, the sharing-layer transformer can be considered a recurrent system (Dehghani et al., 2018) where the dynamic is continual along the layers and a stationary point may exist given infinite number of layers, which makes it closer to our constructed LSA.

The ICL training dataset contains 64,000 training sequences. Each sequence contains 40 in-context examples and 20 queries, where queries are independent of each other similar to Section 5.1. The transformers are trained with batch size 64 for 100 epochs. More details of the hyper-parameters of the experiments are provided in Appendix E.

We evaluate the ICL performance using 64 holdout test sequences and report the test errors on the query examples. The results are summarized in Table 1. We find that both prefixLM-SL and prefixLM-UL significantly outperform their counterparts of causalLM in all cases. As a side note, transformer-SL appears to outperform transformer-UL in the classification tasks, which indicates the overfitting problem of the latter due to over-parameterization. In addition, we also add probes at the output of each SL-transformer layer to visualize the test errors of intermediate layers in Figure 3. Comparing Figure 3 and Figure 2 (left/middle) reveals some similarities. Although the test query errors of causalLM decay in roughly the same rate as the ones of prefixLM in earlier layers, the decays become much slower in later layers possibly due to the nature of its non-optimal stationary points. These results suggest that the title argument of the paper also holds beyond LSA-based transformers and linear regression.

Linear Regression

|PrefixLM<br><br>CausalLM| | | | |
|---|---|---|---|---|
| | | | | |

100

Testerror

10 1

10 2

0 5 10 15 20 25 Layer

Non-linear Regression

10 1

|PrefixLM<br><br>CausalLM| | | | |
|---|---|---|---|---|
| | | | | |

10 2

Testerror

10 3

0 5 10 15 20 25 Layer

Multiclass Classification

PrefixLM

0.34

CausalLM

0.32

0.30

Testerror

0.28

0.26

0.24

0 5 10 15 20 25 Layer

- Figure 3: The test query errors of the 24-layer SL-transformers based prefixLM/causalLM-ICLs on linear regression (left), non-linear regression (middle), and multiclass classification (right).

| |MMLU<br><br>Base Large XL<br><br>|BBH Base Large XL|
|---|---|---|
|PrefixLM CausalLM<br><br>|28.8 32.0 39.5 28.0 26.9 30.5|27.4 32.2 35.8 24.8 29.8 32.0<br><br>|

- Table 2: The averaged test query accuracies on 5-shot MMLU (57 tasks) and 3-shot BBH (23 tasks) with FLAN-finetuned T5 DecoderOnly prefixLM/causalLM checkpoints.

- 5.3 ICL ON LARGE LANGUAGE MODELS

In order to compare the ICL performance of causalLM and prefixLM in a large language model setting, we conduct experiments using the publicly available T5 family of models (Roberts et al.,

- 2022). Note that the existing public T5X § checkpoints are all based on EncDec models, which are similar to prefixLM. Thus, it would be unfair and unnatural to compare with causalLM by simply replacing the bidirectional attention of the encoder to the causal attention during the finetuning stage. To make a more reasonable comparison, we reran the pretraining stages of T5 on the C4 corpus (Raffel et al., 2020a) from a random initialization point using a span corruption objective, but in the DecoderOnly setting. Moreover, for each size (from Base, Large and XL) of the models, we pretrained two checkpoints, one with prefixLM and the other with causalLM, each for 1M steps using the same T5 pretraining recipe. After pretraining, we use the FLAN recipe (Chung

- et al., 2022) to finetune each checkpoint (40k steps for Base, 20k steps for Large and XL) with its pretrained attention mask and evaluate the ICL capability of the finetuned models on two benchmarks: MMLU (Hendrycks et al., 2020) and BBH (Suzgun et al., 2022).

Table 2 shows that for all three sizes of T5X DecoderOnly models, the MMLU and BBH accuracies of prefixLM surpasses that of causalLM consistently and such gap widens as the size of the model becomes larger. This result empirically verifies that our conjecture generalizes to the practical case. We supply additional empirical evidence on state-of-the-art models in Appendix E.6 and E.7.

- 6 CONCLUSION

In this paper, we analyzed the convergence properties of two types of widely-used transformer-based language models (causalLM and prefixLM), during in-context learning. Using a simplified LSA attention in a linear regression setting, we proved that both LM types converge to their stationary points in linear rates, but that their stationary points have significantly different properties. In particular, the stationary points of prefixLM coincides with the optimal least square solution; while the ones of causalLM is equivalent to the weights of an online learning system, that is not guaranteed to converge to the optimal solution. Our experiments verified the above theoretical results, and also empirically extend the findings to general transformer on non-linear regression as well as classification tasks. Finally, we compare causalLM and prefixLM on a few large language models and find that prefixLM also consistently wins over causalLM in practical few-shot tasks.

§https://github.com/google-research/t5x

ACKNOWLEDGEMENTS

We want to specially thank Xinhua Zhang for helpful discussions and references about online learning. We also thank Yi Tay for comments regarding the PaLM2 checkpoints.

REFERENCES

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, Siamak Shakeri, Mostafa Dehghani, Daniel M. Salz, Mario Lucic, Michael Tschannen, Arsha Nagrani, Hexiang Hu, Mandar Joshi, Bo Pang, Ceslee Montgomery, Paulina Pietrzyk, Marvin Ritter, A. J. Piergiovanni, Matthias Minderer, Filip Pavetic, Austin Waters, Gang Li, Ibrahim M. Alabdulmohsin, Lucas Beyer, Julien Amelot, Kenton Lee, Andreas Steiner, Yang Li, Daniel Keysers, Anurag Arnab, Yuanzhong Xu, Keran Rong, Alexander Kolesnikov, Mojtaba Seyedhosseini, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali-x: On scaling up a multilingual vision and language model. ArXiv, abs/2305.18565, 2023. URL https://api.semanticscholar.org/CorpusID: 258967670.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022.

Google, Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang

Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Arnulf Jentzen and Philippe Von Wurstemberger. Lower error bounds for the stochastic gradient descent optimization algorithm: Sharp convergence rates for slowly and fast decaying learning rates. Journal of Complexity, 57:101438, 2020.

Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3128–3137, 2015.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837, 2022.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020a. URL http://jmlr.org/papers/v21/20-074.html.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020b.

Adam Roberts, Hyung Won Chung, Anselm Levskaya, Gaurav Mishra, James Bradbury, Daniel Andor, Sharan Narang, Brian Lester, Colin Gaffney, Afroz Mohiuddin, Curtis Hawthorne, Aitor Lewkowycz, Alex Salcianu, Marc van Zee, Jacob Austin, Sebastian Goodman, Livio Baldini Soares, Haitang Hu, Sasha Tsvyashchenko, Aakanksha Chowdhery, Jasmijn Bastings, Jannis Bulian, Xavier Garcia, Jianmo Ni, Andrew Chen, Kathleen Kenealy, Jonathan H. Clark, Stephan Lee, Dan Garrette, James Lee-Thorp, Colin Raffel, Noam Shazeer, Marvin Ritter, Maarten Bosma, Alexandre Passos, Jeremy Maitin-Shepard, Noah Fiedel, Mark Omernick, Brennan Saeta, Ryan Sepassi, Alexander Spiridonov, Joshua Newlan, and Andrea Gesmundo. Scaling up models and data with t5x and seqio. arXiv preprint arXiv:2203.17189, 2022. URL https://arxiv.

org/abs/2203.17189.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Melbourne, Australia, July 2018.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, and Donald Metzler. Unifying language learning paradigms. arXiv preprint arXiv:2205.05131, 2022.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pp. 35151–35174. PMLR, 2023.

Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080, 2021.

Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. arXiv preprint arXiv:2306.09927, 2023.

### CausalLM is not optimal for in-context learning

#### Appendices

- A RELATED WORK

Ever since GPT-3 (Brown et al., 2020) exhibited its in-context learning abilities in various language inference and translation tasks, there has been tremendous interest in understanding the mechanics behind In-Context Learning (ICL). Currently, there are two main camps of thought that try to explain ICL: (1) the representation camp, which views ICL behavior as a topic model that extracts relevant memories based on the topic of the context (Xie et al., 2021; Min et al., 2022) - these works support this view with the findings that in-context learner sometimes behaved similarly even when the label of the training examples were permuted (Min et al., 2022). (2) the algorithmic camp, which holds that LLMs learns to implement a learning algorithm (Garg et al., 2022; Akyürek et al., 2022; Von Oswald

- et al., 2023) and then run it during ICL - these works usually propose a construction of the transformer parameters and show that it can solve certain simple tasks (e.g. linear regression), then empirically verify that transformers track the behavior of the algorithm of interest.

Moreover, recent studies of large-scale data and model (Wei et al., 2023) discovered that large language models seem to exhibit certain emergent behavior, where, ICL is memory-based on smallto-medium sized models or data, but becomes more algorithm-based on larger model and data. For example, (Wei et al., 2023) showed that a large language model is able to respond accordingly to the flipped label in in-context examples, opposing the findings of (Min et al., 2022).

Since most ICL applications only involve few shots of context examples, it seems reasonable to conjecture that the memory of a deep representation and a shallow predictor algorithm may co-exist in contributing the in-context learning capabilities. Since the representation learning of large language models have been universally acknowledged, it is more interesting to investigate how transformer learns to in-context learn shallow predictors with few-shot examples.

Focusing on work from the algorithmic camp, we note that (Garg et al., 2022) were the first to suggest using linear regression to study in-context learning. The authors empirically found that a 12-layer transformer is able to achieve similar results as a least-square solver on a 20-dim linear regression problem with around 20 in-context examples. Beyond linear regression, they also found that transformers can in-context learn a few other classes of shallow predictors, including two-layer Relu networks.

Probably the first formal theoretical investigation of the linear regression in-context learners is (Akyürek et al., 2022). They first showed that a transformer layer can approximately conduct four basic operations: mov, mul, div, aff. They then cleverly combined these four operations and showed that a gradient descent step of linear regression can be implemented with a 4-head 8-layer transformer with O(d) hidden units, where d is the dimension of the inputs x. Despite their novel construction, the result itself provides only a loose upper bound on the model size (or depth) that is required for simulating linear regression within a transformer - for example, (Von Oswald et al., 2023) reported that a 2 or 5-layer transformer already achieves significantly better results than a single-step gradient descent for linear regression.

Because of the significant discrepancy between the construction of (Akyürek et al., 2022) and the empirical results, the one-layer LSA construction of (Von Oswald et al., 2023) appears to be more appealing and matches the experimental results better. Moreover, a most recent work by (Zhang et al.,

- 2023) used gradient flow to prove that by initializing w(0) = 0, such matrix constructions can indeed be learned by an LSA transformer. This is why our paper follows this construction and studies its multi-layer convergence properties with different types of attention (prefixLM vs causalLM).

In terms of the comparison between prefixLM and causalLM, such research work can be traced back as early as (Raffel et al., 2020b), where they showed prefixLM outperforms causalLM in varieties of NL tasks. Later, UL-2 (Tay et al., 2022) proposed to mix prefixLM and span corruption objectives, and found it to be more efficient than the causalLM objective alone. It was also shown

in (Chung et al., 2022), that U-PaLM (a UL2-finetuning PaLM) outperforms PaLM (causalLM only) in various ICL tasks. Indeed, for the reasons above, some of the latest models have included prefixLM objectives in the pretraining mix (for example PaLM-2 by Google et al. (2023)). On the other hand, prominent models such as Flamingo as well as the ones in the GPT-family are still based on the causalLM structure, so the comparison between prefixLM and causalLM remains important and relevant. Furthermore, all previous studies were done in an empirical manner, whereas we set out to explain their differences from a theoretical perspective and back the theory with empirical evidence. While we are not the first to follow this path, our work is the first to provide a theoretical justification for the advantage of prefixLM over causalLM in a multi-layer transformer ICL setting by analyzing their theoretical convergence properties.

- B PROOFS In this section, we provide proofs of the propositions introduced in Section 3 and Section 4.

- Proposition 1 For a multi-layer LSA satisfying (5) with w(0) = 0, if its input Z is formatted as (4),

then its l-th layer output is z(jl) = (x⊤j ,δj(l))⊤, where δj(l) = yj −w(l) xj and w(l) is the weight from the l-th step gradient descents as in (3).

- Proof: Plugging in K, Q, P and V of (5) with w(0) = 0 into (1), we have

xj δj(l−1)

xj δj(l)

η n

0d×d 0

0 −1 ·

+

=

n

xj δj(l−1)

xi δi(l−1)

Id×d 0 0 0

(x⊤i , δi(l−1))

i=1

n

###### xj

0 δi(l−1)

η n

x⊤i xj .

δj(l−1) −

=

i=1

It is easy to see that zj never changes its first d-dimension corresponding to xj. Therefore, we can simplify the above equation and focus only on the last coordinate δj(l), where we have

η n

δj(l) = δj(l−1) −

n

δi(l−1) x⊤i xj, (12)

i=1

with δj(0) = yj. Defining y˜j(l) = yj − δj(l) and rearranging (12), we obtain y˜j(0) = 0 and

n

η n

y˜j(l) = y˜j(l−1) +

(yi − y˜i(l−1))x⊤i xj . (13)

i=1

Next we prove y˜j(l) = w(l) xj by induction. Since w(0) = 0, it is clear that y˜j(0) = w(0) xj = 0 for all j.

If y˜j(l−1) = w(l−1) xj for all j, then

η n

y˜j(l) = y˜j(l−1) +

n

i=1

η n

=w(l−1) xj +

n

= w(l−1) +

i=1

=w(l) xj .

(yi − y˜i(l−1))x⊤i xj

n

(yi − w(l−1) xi)x⊤i xj

i=1

(yi − w(l−1) xi)x⊤i xj

- Proposition 2 For a multi-layer causalLM-LSA satisfying (5) with w(0) = 0, if its input Z is

formatted as (4), then its l-th layer output is z(jl) = (x⊤j ,δj(l))⊤, where δj(l) = yj − w(jl) xj and w(jl) follow (10).

- Proof: Plugging in K, Q, P and V of (5) with w(0) = 0 into (2), we have

δj(l) = δj(l−1) −

η n

j

i=1

δi(l−1) x⊤i xj

y˜j(l) = y˜j(l−1) +

η n

j

i=1

(yi − y˜i(l−1))x⊤i xj

with y˜j(l) = yj − δj(l). Next we prove y˜j(l) = w(jl) xj by induction. Since w(0)j = 0, it is clear that y˜j(0) = w(0)j xj = 0 for all j.

If y˜j(l−1) = w(jl−1) xj for all j, then

y˜j(l) = y˜j(l−1) +

η j

n

i=1

(yi − y˜i(l−1))x⊤i xj

=w(jl−1) xj +

η n

j

i=1

(yi − w(il−1) xi)x⊤i xj

= wj(l−1) +

n

i=1

(yi − w(il−1) xi)x⊤i xj

=w(jl) xj .

□

- Proposition 3 If w(l) follows the iterative updates of (3), then there exists a stationary point w∗ with coefficients satisfying:

##### y x⊤ = w∗ XX⊤,

where y = (y1,...,yn) and X = (x1,...,xn). Furthermore, the iterative weights w(l) converges to the stationary point w∗ with linear rate of convergence:

η n

##### XX⊤).

w(l) −w∗ = (w(l−1) −w∗)(I−

Proof: From (3), we have

n

η n

w(l) = w(l−1) +

(yi − w(l−1) xi)x⊤i

###### .

i=1

(∗)

The stationary point must satisfy (∗) = 0. Written in vectorized form, we have

y X⊤ = w∗ XX⊤ . (14) Now plugging (14) back to (3), we have

η n

w(l) = w(l−1) +

w∗ X X⊤ − a w(l−1) X X⊤ ,

which can be reorganized to

η n

##### XX⊤).

w(l) −w∗ = (w(l−1) −w∗)(I−

- Proposition 4 If w(jl) = ji=1 a(i,jl) x⊤i follows the iterative updates of (10), then ai,j(l) = a(i,il) ≡ a(il) ∀j ≥ i,

and there exists stationary points w∗j = ji=1 a∗i x⊤i (for j ∈ 1,...,n) with coefficients from a∗ = (a∗1,...,a∗n) that satisfy y = a∗ T, where





x⊤1 x1 x⊤1 x2 ··· x⊤1 xn 0 x⊤2 x2 ··· x⊤2 xn

T =

.

 

 

... .

. .

0 0 ··· x⊤n xn

Furthermore, the coefficients a(l) converges to the stationary point a∗ with linear rate of convergence:

η n

a(l) −a∗ = (a(l−1) −a∗)(I−

##### T).

Proof: First notice that according to (10), we have

j

η n

w(jl) = w(jl−1) +

(yi − w(il−1) xi)x⊤i

i=1

j

η n

(yi − w(il−1) xi) x⊤i or

a(i,jl−1) +

=

i=1

η n

a(i,jl) = a(i,jl−1) +

(yi − w(il−1) xi) ∀j ≥ i.

Since a(0)i,j = 0, and the above update is the same for any j given any i, then it is obvious by induction that

a(i,jl) = a(i,il) ≡ a(il) ∀j ≥ i.

Therefore, we can simplify w(jl) = ji=1 a(il) x⊤i . Now plugging into (10), we have

which is equivalent to

j

a(il) x⊤i

i=1

j

j

η n

a(il−1) x⊤i +

=

i=1

i=1

j

a(il−1) + yi −

=

i=1

i

ak(l−1) x⊤k xi) x⊤i

(yi −

k=1

i

η n

a(kl−1) x⊤k xi x⊤i ,

k=1

i

η n

a(kl−1) x⊤k xi

a(il) = ai(l−1) +

yi −

k=1

(∗)

The stationary points satisfy (∗) = 0, which gives

- y1 =a∗1 x⊤1 x1
- y2 =a∗1 x⊤1 x2 +a∗2 x⊤2 x2

... yn =a∗1 x⊤1 xn +... + a∗n x⊤n xn,

. (15)

or in the vectorized form y = a∗ T, where





x⊤1 x1 x⊤1 x2 ··· x⊤1 xn 0 x⊤2 x2 ··· x⊤2 xn

T =

.

 

 

... .

. .

0 0 ··· x⊤n xn

Now plugging in y = a∗ T back to (15) and vectorize it, yields

η n

a(l) = a(l−1) +

(a∗ T−aT), which can be reorganized to

η n

a(l) −a∗ = (a(l−1) −a∗)(I−

##### T).

###### □

- Proposition 5 Assuming that w∗j is the stationary points obtained in Proposition 4, then

w∗j+1 = w∗j −

1 ∥xj+1 ∥22

∇w∗

j

Lj(w∗j).

Proof: Recall the online learning system with a sequence of data-label pairs (xj,yj) has the following online loss and its gradient at the j-th step,

Lj(wj) =

- 1

- 2

(wj xj+1 −yj+1)2, ∇wj

Lj(wj) = (wj xj+1 −yj+1)x⊤j+1 . According to Proposition 4, we have y = a∗ T, which gives

yj+1 =a∗1 x⊤1 xj+1 +... + a∗j x⊤j xj+1

+ a∗j+1 x⊤j+1 xj+1

=w∗j xj+1 +a∗j+1 x⊤j+1 xj+1 (16) where the last equation is due to w∗j = ji=1 a∗i x⊤i . Since w∗j = ji=1 a∗i x⊤i , we have

w∗j+1 =w∗j +a∗j+1 x⊤j+1

=w∗j +

1 ∥xj+1 ∥22

a∗j+1 x⊤j+1 xj+1 x⊤j+1

=w∗j −

1 ∥xj+1 ∥22

(w∗j xj+1 −yj+1)x⊤j+1

=w∗j −

1 ∥xj+1 ∥22

∇w∗

j

Lj(w∗j)

where the third equation is because of (16). □

C MULTI-LAYER LSA CONSTRUCTION WITH NON-ZERO W(0)

In this section, we introduce the proposition that connects a multi-layer LSA following the construction of (5) but with non-zero w(0) and the multi-step gradient descents of linear regression.

- Proposition 6 For a multi-layer LSA satisfying the construction (5), if its input Z is formatted as (4),

then its l-th layer output is z(jl) = (x⊤j ,δj(l))⊤, where δj(l) = yj − (w(l) −w(0))xj and w(l) is the l-th updated weight from the gradient descents update rule in (3).

Proof: Plugging in K, Q, P and V of (5) into (1), we have

n

η n

δj(l) = δj(l−1) −

δi(l−1) − w(0) xi x⊤i xj, (17)

i=1

with δj(0) = yj. Defining y˜j(l) = yj −δj(l) +w(0) xj and rearranging the (17), we obtain y˜j(0) = 0 and

n

η n

y˜j(l) = y˜j(l−1) +

(yi − y˜i(l−1))x⊤i xj .

i=1

Then it is easy to prove y˜j(l) = w(l) xj by induction, similar to the proof of Proposition 1. □

- D CAUSALLM WITH ATTENTION-LENGTH-BASED COEFFICIENTS Since there are j terms in the summation of (10), another reasonable update for causalLM would be

j

η j

(yi − w(il−1) xi)x⊤i , (18) which we call causalLM2. For causalLM2, we have the following proposition.

w(jl) = w(jl−1) +

i=1

- Proposition 7 If w(jl) = ji=1 a(i,jl) x⊤i follows the iterative updates of (18), then

a(i,jl) ≡

1 j

a(il) ∀j ≥ i,

and there exists stationary points w∗j = 1j ji=1 a∗i x⊤i (for j ∈ 1,...,n) with coefficients from a∗ = (a∗1,...,a∗n) that satisfy y = a∗ S, where

S =



 

x⊤1 x1 21 x⊤1 x2 ··· n1 x⊤1 xn 0 12 x⊤2 x2 ··· n1 x⊤2 xn

. .

... .

0 0 ··· n1 x⊤n xn



 

.

Furthermore, the coefficients a(l) converges to the stationary point a∗ with the following rate of convergence:

a(l) −a∗ = (a(l−1) −a∗)(I−η S).

The condition number κ(S) is about n/2 greater than the one of κ(T), which makes causalLM2 converge much slower than causalLM.

One can also prove that the stationary point of causalLM2 corresponds to the following online system with online loss and gradient at the j-th step,

Lj(w˜ j) =

- 1

- 2

(w˜ j xj+1 −yj+1)2, ∇w˜j

Lj(w˜ j) = (w˜ j xj+1 −yj+1)x⊤j+1, where w˜ = j+1j w.

- Proposition 8 Assuming that w∗j is the stationary points obtained in Proposition 4, then

1 ∥xj+1 ∥22

w∗j+1 = w˜ ∗j −

###### Lj(w˜ ∗j).

∇w˜∗

j

Since the step does not have j−ν (ν > 0) decay, such online system is not guaranteed to converge, therefore suffers the same problem as the original causalLM in Section 3.2.

In Figure 4, we plot the query MSE error of the stationary points of causalLM2-ICL with increasing number of in-context examples. We can see that the online system corresponding to causalLM2-ICL converges even slower than the ones of causalLM-ICL in Figure 2 right.

Stationary points of causalLM2-ICL

- 100
- 101

|x = 0<br><br>x = 1<br><br>x = 2<br><br>x = 3<br><br><br>| | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

QueryMSEerror

10 1

50 100 150 200 250 300 Number of training exemplars

- Figure 4: The test error on the stationary point of the causalLM2-ICL with up to 300 in-context examples.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

- Figure 5: The illustration of the attention mask. Green arrows represent the attentions between incontext examples. The dashed arrows only applies for prefixLM. Red arrows represent the attentions from queries to in-context examples. The query examples should not attend to themselves because the inputs do not contain labels.

- E ADDITIONAL EXPERIMENTAL DETAILS AND RESULTS

- E.1 EXPERIMENT SETTINGS FOR SECTION 5.1 In order to directly verify the theorem, we used the constructed LSA-based transformer, with

K = Q =

Id×d 0 0 0

, V =

0d×d 0

0 −1

and P = nη I. Although not a trained transformer, it was recently proved in (Zhang et al., 2023) that a randomly initialized LSA-transformer does converge to such a construction. In addition, we did an ablation test of η = {0.1,0.2,0.4,0.8,1.6,3.2} and chose η = 1.6 as it converges the fastest without any divergence problems.

We randomly generated 64 sequences for ICL evaluation. For each sequence, we put the first 40 examples as the in-context examples and the last 200 examples as the query examples. The queries are independent of each other without attention. See Figure 5 for an illustration of the transformer attention mask. Such multi-query design is for training efficiency purpose only and is equivalent to 200 sequences with the same w and input examples xi, but different one query per sequence.

- E.2 EXPERIMENT SETTINGS FOR SECTION 5.2

In order to verify that our theorems can be qualitatively applied beyond LSA and linear regression, we conducted several experiments over various synthetic tasks using regular transformers. We based our

| |LR N-LR MC<br><br>|
|---|---|
|PrefixLM-SL-L2 CausalLM-SL-L2<br><br>|8.6e-3 1.5e-4 24.1 1.9e-1 2.7e-3 27.0|
|PrefixLM-SL-no-L2 CausalLM-SL-no-L2|6.7e-3 1.5e-4 24.5 5.0e-2 1.9e-3 30.5<br><br>|
|PrefixLM-UL-L2 CausalLM-UL-L2<br><br>|7.6e-3 1.7e-4 26.7 4.4e-2 2.5e-3 30.4|
|PrefixLM-UL-no-L2 CausalLM-UL-no-L2|2.5e-3 9.0e-5 27.6 1.6e-2 2.9e-3 32.1<br><br>|

- Table 3: The test query errors of the unshared-layer (UL) and sharing-layer (SL) transformer-ICLs with or without L2 regularizer on linear regression (LR), non-linear regression (NLR), and multiclass classification (MC) tasks.

Linear Regression

- 100
- 101

|PrefixLM<br><br>CausalLM|
|---|

QueryMSE

10 1

10 2

100 101 102 103 # batches of training examplars

Non-linear Regression

|PrefixLM<br><br>CausalLM|
|---|

10 1

10 2

QueryMSE

10 3

100 101 102 103 # batches of training examplars

Multiclass Classification

| |PrefixLM<br><br>CausalLM|
|---|---|
| | |
| | |
| | |
| | |

0.6

0.5

TestError

0.4

0.3

100 101 102 103 # batches of training examplars

- Figure 6: The test query errors of the SL-transformers based prefixLM/causalLM-ICLs with various numbers of training sequences on linear regression (left), non-linear regression (middle), and multiclass classification (right).

code from the repository of (Akyürek et al., 2022)¶ and applied their default training hyperparameters of the code. We used a transformer of 24 layers with 128 hidden units and 2 heads. The FFN intermediate size is 4 × 128 = 512. The learning schedule is based on cosine decay with base learning rate 1e-4, for 100 epochs. In addition, since the target of the outputs of the in-context examples are 0 (see Fig. 1), we optionally add an additional L2 regularizer on the outputs of the in-context examples. See the comparison between the transformers with or without the L2-regularizer in Table 3. In Table 1 of the main paper, the reported numbers correspond to the SL-transformer with the L2 regularizer and the UL-transformer without the L2 regularizer. Across all these settings prefixLM consistently beats causalLM as our theorem predicts.

- E.3 THE IMPACT OF THE SIZE OF THE TRAINING DATA

Here we investigate the performance of prefixLM and causalLM as a function of the number of training samples. In Fig. 6, we plot their after having trained on 10 batches all the way up to 1000 batches (as in Section 5.2). We observe that when the amount of training data is low, ICL falls into the memorization regime, in which models perform perfectly on the training data, but do not generalize well to unseen test sequences. However, prefixLM transitions to the generalization regime earlier than causalLM, which is reflected by the positions of the largest gap between the two. (30 batches in LR, 100 batches in N-LR, and 10 batches in MC).

- E.4 TESTING WITH FEWER IN-CONTEXT EXAMPLES

In causalLM, different positions in the sequence are trained with different numbers of in-context examples (ICEs). This may bring advantage to pretrained causalLM models when tested with fewer number of in-context examples than it was trained on. To compare causalLM and prefixLM in such setting, we use the same models as before that were trained with 40 in-context examples, but test them on fewer (16, 24, 32) in-context examples. Note that 16 is the minimum number of examples to

¶https://github.com/google-research/google-research/tree/master/ incontext

|16 Test ICEs|LR N-LR MC<br><br>|
|---|---|
|PrefixLM-SL CausalLM-SL<br><br>|1.01 2.1e-2 42.8 1.76 2.7e-2 43.3|
|PrefixLM-UL CausalLM-UL<br><br>|0.97 1.9e-2 42.9<br>1.12 3.2e-2 46.6<br>|

###### Table 4: The test query errors with 16 ICEs on linear regression (LR), non-linear regression (NLR), and multiclass classification (MC) tasks.

|24 Test ICEs<br><br>|LR N-LR MC|
|---|---|
|PrefixLM-SL CausalLM-SL<br><br>|1.4e-1 2.0e-3 33.4 7.0e-1 1.0e-2 35.9|
|PrefixLM-UL CausalLM-UL<br><br>|1.0e-1 1.7e-3 37.1 1.3e-1 1.0e-2 41.2|

###### Table 5: The test query errors with 24 ICEs on linear regression (LR), non-linear regression (NLR), and multiclass classification (MC) tasks.

solve our 16-dim synthetic regression problems. The errors of prefixLM and causalLM are provided in the following Tables 4, 5, 6, where regression tasks (LR, N-LR) report mean squared errors and the MC task reports the classification error. From the tables we see that prefixLM still consistently outperforms causalLM, even when testing with fewer in-context examples than used during training time.

E.5 PERMUTATION ON IN-CONTEXT EXAMPLES

We further consider a simple approach for mitigating the problems of causalLM by randomly permuting the in-context examples during training time. This is motivated by the observation that for causalLM, every permutation representations a different view of the context in the example. The results of this experiment (Table 7) show that this style of causalLM training indeed improves over the fixed order training setting compared to the unpermuted ICEs (Table 1). However, prefixLM still outperforms causalLM in general.

E.6 IN-CONTEXT LEARNING USING PALM2

Going beyond the publicly available T5 models, we further verify our findings by conducting FLANbased finetuning experiments using the state-of-the-art PaLM2 family of models (Google et al., 2023). PaLM2 models were pretrained with a mixture of objectives that includes different LM types, which make them a relatively fair starting point to compare causalLM and prefixLM after finetuning. In practice we finetune three sizes of PaLM2 language models: Gecko, Otter and Unicorn||. We use the same default recipe for FLAN-PaLM2 finetuning (Google et al., 2023; Chung et al., 2022) and finetune the PaLM2 checkpoints for either causalLM or prefixLM. We then evaluate the ICL capability of the finetuned models on the Massive Multi-task Language Understanding (5-shot MMLU) tasks (Hendrycks et al., 2020).

||https://blog.google/technology/ai/google-palm-2-ai-large-language-model/

|32 Test ICEs|LR N-LR MC<br><br>|
|---|---|
|PrefixLM-SL CausalLM-SL<br><br>|2.4e-2 4.7e-4 32.4<br>3.1e-1 5.0e-3 34.6<br>|
|PrefixLM-UL CausalLM-UL<br><br>|9.5e-3 3.4e-4 36.2 4.0e-2 5.7e-3 37.3|

###### Table 6: The test query errors with 32 ICEs on linear regression (LR), non-linear regression (NLR), and multiclass classification (MC) tasks.

|Permuted ICEs<br><br>|LR N-LR MC|
|---|---|
|PrefixLM-SL CausalLM-SL<br><br>|9.0e-3 1.5e-4 24.1 1.9e-1 2.5e-3 26.9|
|PrefixLM-UL CausalLM-UL<br><br>|2.6e-3 9.5e-5 26.1 1.1e-2 1.8e-3 26.2|

- Table 7: The test query errors with randomly permuted ICEs on linear regression (LR), non-linear regression (NLR), and multiclass classification (MC) tasks.

| |Gecko Otter Unicorn<br><br>|
|---|---|
|PrefixLM CausalLM<br><br>|46.6 64.8 81.4 43.3 61.0 78.0|

- Table 8: The average test query accuracies on 5-shot MMLU tasks with FLAN-finetuned PaLM2Gecko/Otter/Unicorn prefixLM/causalLM checkpoints. (Google et al., 2023) reported a similar averaged accuracy of 81.2 on Unicorn-PrefixLM.

- Table 8 shows that for all three sizes of PaLM2, the MMLU accuracy (average over the 57 tasks) of prefixLM surpasses that of causalLM by more than 3%. This result again empirically verifies that our conjecture generalizes to the practical case, using a state of the art LLM**.

E.7 IN-CONTEXT LEARNING WITH MULTIMODAL MODELS

Lastly, we also demonstrate that prefix attention masks benefit ICL in multimodal models across various settings. We conducted experiments using both 4-shot and 8-shot COCO image captioning tasks on the Karpathy split (Karpathy & Fei-Fei, 2015) using the PaLI-X model (Chen et al., 2023), a 55B multimodal pretrained model.

The PaLI-X model employs an encoder-decoder architecture where ViT encoded image tokens and text tokens are fed to the multimodal encoder and decoder to generate outputs. During pretraining, the text prompts were split into two parts. The first part is the input to the multimodal prefix-encoder that self-attends to all the image and text tokens on the encoder side, following the style of prefixLM. The second part is the input to the causal-decoder that self-attends to only the previous text tokens on the decoder side, following the style of causalLM, and cross-attends to encoder tokens.

The prefix-encoder and causal-decoder nature allows us to consider different variants of the attention masks and placements of the in-context texts to showcase the benefits of prefix attention masks. We design two main categories of few-shot experiments with five self-attention mask settings, detailed below. We finetune the PaLI-X pretrained model using each setting’s attention mask with 4-shot Episodic WebLI dataset (Chen et al., 2023) for 20k steps.

In the first category, we place the few-shot text tokens on the encoder side and study the effect of manipulating the encoder self-attention masks, leaving the causal-decoder unchanged. Specifically,

**Besides of PaLM2, we also find that any checkpoint that is pretrained with a mixture of prefixLM and causalLM tends to do better with prefixLM for in-context learning. However, we do not claim that prefixLM would necessarily outperform causalLM when using solely causalLM pretrained checkpoints.

| |4-shot 8-shot|
|---|---|
|Prefix encoder Block-causal encoder Causal encoder|106.7 107.5 104.8 106.0 102.3 104.9<br><br>|
|Prefix decoder Causal decoder|103.9 104.2<br><br>102.4 92.9|

- Table 9: Cider scores of COCO captioning using various attention masks. The Prefix variant outperforms the Causal ones. Note that the official PaLI-X (Chen et al., 2023) reported a 4-shot Cider of 107.6, which was also based on the prefix encoder mask, but was finetuned with additional image captioning data from the Conceptual Captions 3M dataset (Sharma et al., 2018).

considering a 2-shot ICL case for simplicity, we adapt the prefix encoder attention mask Aencprefix in (19) into two causal variants, block-causal and causal encoder attention masks as Aencb−causal in (20) and Aenccausal in (21). In this case, the block-causal version is more inline with exposing the encoder to the examples one at a time, while the causal one strictly follows auto-regressive attention.

I1 T1 I2 T2 It

  

  

I1

- T1 I2
- T2 It

Aencprefix =

(19)

Aencb−causal =

I1 T1 I2 T2 It

  

  

I1

- T1 I2
- T2 It

(20)

I1 T1 I2 T2 It

  

  

I1 ⧹ T1 I2 ⧹ T2 It

Aenccausal =

(21)

I1, I2, It denotes the image tokens for the two shots and the target and T1, T2 denotes the text tokens for the two shots. denotes a matrix of all 1s and “⧹” denote an upper triangular matrix with 1s. A 1 at row i and column j indicates that token j is allowed to attend to token i. We report the results on few-shot COCO captioning in the top half of Table 9. We observe consistent improvement over both 4- and 8-shot when changing the encoder attention mask from causal mask, to block causal mask, and then to prefix mask.

Adeccausal =

T1 T2 Tt ⧹ T1

⧹ T2 ⧹ Tt

(22)

Adecprefix =

T1 T2 Tt

- T1
- T2

⧹ Tt

(23)

Similarly, in the second category, we place the few-shot text on the decoder side and study the effect of manipulating the decoder attention masks, leaving the prefix encoder unchanged. We adapt the causal decoder attention mask Adeccausal in (22) to prefix attention mask Adecprefix in (23). Note that all the image tokens from the prefix-encoder side are visible to all text tokens (on the decoder) via cross attention. However, the image tokens cannot attend to the text because of the encoder-decoder architecture. The second half of Table 9 reports the results of using prefix and causal decoder attention. Even though the decoder is pretrained in the causal manner, with additional finetuning using prefix masks, the new prefix decoder achieves a Cider score of 103.9 in 4-shot ICL, outperforming the finetuned causal decoder by 1.5. Furthermore, the prefix decoder also appears to be more robust when extrapolating to 8-shot evaluation (Cider 104.2), compared to the causal decoder (Cider 92.9).

In summary, the LLM experiments in Section 5.3 as well as the multimodal experiments in this section show that our conjectures hold up in practice with various types of large-scale models and a wide range of settings.

