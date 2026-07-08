# arXiv:2404.14994v3[cs.CL]20Jun2024

## Transformers Can Represent n-gram Language Models

Anej Svete Ryan Cotterell {asvete, ryan.cotterell}@inf.ethz.ch

|[Figure 1]|
|---|

### Abstract

Existing work has analyzed the representational capacity of the transformer architecture by means of formal models of computation. However, the focus so far has been on analyzing the architecture in terms of language acceptance. We contend that this is an ill-suited problem in the study of language models (LMs), which are definitionally probability distributions over strings. In this paper, we focus on the relationship between transformer LMs and n-gram LMs, a simple and historically relevant class of language models. We show that transformer LMs using the hard or sparse attention mechanisms can exactly represent any n-gram LM, giving us a concrete lower bound on their probabilistic representational capacity. This provides a first step towards understanding the mechanisms that transformer LMs can use to represent probability distributions over strings.

[Figure 2]

https://github.com/rycolab/ transformer-ngrams

### 1 Introduction

Neural language models (LMs) have become the backbone of many NLP systems. Their widespread adoption has prompted a plethora of theoretical work investigating what they can and cannot do by studying their representational capacity. Most state-of-the-art LMs are based on the transformer architecture (Vaswani et al., 2017), whose theoretical abilities and limitations have been studied extensively; see, e.g., the survey by Strobl et al. (2023). But, many questions remain unanswered. Most existing work studies the architecture in terms of binary language recognition. This introduces a category error between the object of study—an LM, which is definitionally a distribution over strings—and the theoretical abstraction—a set of strings. To amend this discrepancy, we ask: What classes of probability distributions over strings can transformer LMs represent?

Formal models of probabilistic computation provide a natural, well-understood, and precise framework for studying the classes of probability distributions language models can represent. Traditionally, the representational capacity of neural

ppyt | yătq softmaxpE ¨q

H

Head 3 Head 2 Head 1

y1 y2 ¨ ¨ ¨ yt´4 yt´3 yt´2 yt´1 yt ¨ ¨ ¨

Figure 1: A transformer LM can simulate a 4-gram LM using 3 heads. The stronger arrows from the heads to the symbols show where the heads focus their attention.

networks, both in terms of lower bounds (what they can provably do) as well as upper bounds (what they can provably not do), has been studied in terms of Boolean sequential models of computation, such as finite-state automata and Turing machines (e.g., Kleene, 1956; Minsky, 1954; Siegelmann and Sontag, 1992; Hao et al., 2018; Merrill, 2019; Merrill et al., 2020, 2022; Merrill and Tsilivis, 2022). Recent work has extended this paradigm to work with probabilistic models of computation (Svete and Cotterell, 2023; Nowak et al., 2023), but so far only for LMs based on recurrent neural networks.

However, the sequential nature of classical models makes the connection to the inherently parallelizable transformer architecture less straightforward and has resulted in a number of results upper-bounding their representational capacity (Hahn, 2020; Bhattamishra et al., 2020; Chiang and Cholak, 2022; Hao et al., 2022a; Merrill and Sabharwal, 2023c). We connect transformer LMs to a classical class of LMs that lend themselves particularly well to parallelized computations: n-gram LMs. We show that both hard as well as sparse attention transformer LMs can represent any n-gram LM (Theorems 3.1 and 4.1).1 This gives

1An analysis completely analogous to practical implementations would also consider soft attention transformer LMs, whose full support when attending over the preceding symbols makes the analysis trickier. We, therefore, omit its analysis

us a concrete lower bound on their probabilistic representational capacity. We also study the role of the number of heads (Theorem 3.1) and the number of layers (Theorem 3.2), illustrating a trade-off between the number of heads, layers, and the complexity of the non-linear transformations required for the simulation of n-gram LMs. Altogether, these results offer a step towards understanding the probabilistic representational capacity of transformer LMs and the mechanisms they might employ to implement formal models of computation.

### 2 Preliminaries

Let Σ be an alphabet, i.e., a finite, non-empty set of symbols, and Σ˚ the (infinite) set of all strings formed from symbols of Σ. Most modern LMs define ppyq for y P Σ˚ autoregressively—as a product of conditional probability distributions:

ź|y|

ppyq “def ppEOS | yq

ppyt | yătq. (1)

t“1

Here, EOS R Σ is a distinguished end-of-string symbol. The EOS symbol enables us to define the probability of a string purely based on the conditional distributions. Such a factorization can be done without loss of generality (Du et al., 2023). We define Σ “def Σ Y tEOSu. Further, the conditional probability distributions ppyt | yătq are usually defined based on vectorial representations of yăt computed by some function enc: Σ˚ Ñ RD. This leads us to the definition of representationbased LMs below.

Definition 2.1. Let Σ be an alphabet and enc: Σ˚ Ñ RD a representation function encoding strings as D-dimensional representations. Let E P R|Σ|ˆD be an output matrix. A representation-based LM p defines the conditional probability distributions ppyt | yătq as2

ppyt | yătq “def softmaxpE encpyătqqy

. (2)

t

At a high level, we are interested in encoding an arbitrary n-gram LM using a transformer LM. To do so, we need a notion of equivalence between language models. In this paper, we will work with the following simple definition.

here and reserve it for a separate treatment.

2One could, more generally, swap the softmax for any other normalization function, such as the sparsemax (Martins and Astudillo, 2016). Here, however, we focus on the softmax for conciseness.

- Definition 2.2. Two LMs p and q over Σ˚ are weakly equivalent if ppyq “ q pyq for all y P Σ˚.

This paper precisely explains and proves the following theorem, stated informally below.

Theorem 2.1 (Informal). For every n-gram LM, there exists a weakly equivalent ({hard, sparse} attention) transformer LM.

2.1 An Aside about Boolean Recognition

Fundamentally, Theorem 2.1 is about weak equivalence (Definition 2.2) between two LMs. In this subsection, we make our case against treating LMs as recognizers. The most common manner of analyzing a language model as a recognizer is based on using its representations as an input to a classifier (Merrill, 2019; Merrill et al., 2020). We recapitulate a common definition below.

- Definition 2.3. Let p be a representation-based LM with the representation function enc: Σ˚ Ñ RD and let g: RD Ñ t0,1u be a classifier. The binary language of p with g is defined as

Lg ppq “def ty P Σ˚ | g pencpyqq “ 1u. (3) Related is the notion of truncated recognition.

- Definition 2.4 (Hewitt et al. (2020), Definition 4). Let p be a language model over Σ˚ and α ą 0. The α-truncated language of p is defined as

Lα ppq “def ty P Σ˚ | ppEOS | yq ě α (4)

and ppyt | yătq ě α @t P r|y|su.

There are many results in the literature treating transformers’ ability to recognize languages in the sense of the two definitions above. For instance, transformers are unable to recognize the Dyck language with more than one bracket type and the PARITY language in the sense of Definition 2.3 (Hahn, 2020), but can recognize bounded Dyck languages in the sense of Definition 2.4 (Yao et al., 2021). Our indictment of analyzing Lg ppq and Lα ppq is that proceeding in such a manner disregards the probabilities assigned to strings by p, which we view as essential to language modeling. Moreover, Definition 2.3 depends on the form of the classifier g while Definition 2.4 depends on the hyperparameter α. For example, positively classified strings from a language could have their conditional probabilities only slightly above the classification threshold and the negatively classified ones only slightly below the threshold (Hahn, 2020), which hides the true distribution defined by the

The quick brown fox jumps over ... ppfox | The quick brownq

¨

¨

ppjumps | quick brown foxq ppover | brown fox jumpsq

¨

Figure 2: An illustration of how a 4-gram LM computes the probability of a string. All conditional probabilities can be computed in parallel and then multiplied into the probability of the entire string.

LM. In this context, our contention is that Lg ppq and Lα ppq are not useful definitions for studying the representational capacity of LMs. Instead, we advocate for analyzing LMs as probabilistic formal languages.

#### 2.2 Language Modeling with n-grams

The next-symbol probabilities in n-gram LMs are computed under the n-gram assumption.

Assumption 2.1. The n-gram assumption states that the conditional probability of the symbol yt given yăt only depends on n ´ 1 previous symbols ytt´´n1`1 “def yt´1,...,yt´n`1:

`

˘

yt | ytt´´n1`1

. (5)

ppyt | yătq “ p

We will refer to ytt´´n1`1 as the history of yt.

Padding. Eq. (5) assumes the existence of n ´ 1 preceding symbols that define the conditional distribution ppyt | yătq. To ensure this is the case even at the beginning of the string, it is standard to pad the input string with n ´ 1 beginning-of-string symbols BOS. For ease of notation, we index the n ´ 1 BOS tokens with indices ´n ` 2,...,0 so that n-gram LMs conveniently fit the autoregressive factorization from Eq. (1). We also define Σ “def Σ Y tBOSu.

Despite their simplicity, n-gram LMs have a storied place in language modeling (Shannon, 1948; Baker, 1975a,b; Jelinek, 1976; Bahl et al., 1983; Jelinek, 1990; Bengio et al., 2000, 2003, 2006; Schwenk, 2007; Heafield, 2011; Heafield et al., 2013). Because the conditional probabilities of n-gram LMs only depend on the previous n´1 symbols, different parts of the string can be processed independently, i.e., in parallel. This facilitates a natural connection to transformer LMs since parallelizability is a prevalent feature of the architecture and one of its main advantages over other neural LMs such as RNN LMs (Vaswani et al., 2017).

#### 2.3 Transformer Language Models

Transformer LMs are LMs whose conditional distributions ppyt | yătq are computed by a transformer. A transformer is a composition of multiple transformer layers, each of which implements the attention mechanism. We give definitions of these building blocks in what follows.

Notation. We use bold, unitalicized letters such as x P RD to denote real-valued vectors and italicized letters xj P R for their entries. Capital bold letters such as X P RNˆD denote matrices. All vectors are column vectors unless transposed. We define the vertically stacking operator p¨ ;¨¨¨ ; ¨q, which denotes the vertical concatenation of the D-dimensional column vectors x1,...,xN into a ND-dimensional vector px1;¨¨¨ ;xNq P RND and the concatenation of the D-dimensional row vectors xJ1 ,...,xJN into a matrix X P RNˆD with N` rows and D columns. Given the matrix X “

, we write Xn “ `

˘

˘

xJ1 ;¨¨¨ ;xJn

xJ1 ;¨¨¨ ;xJN

for the submatrix composed of the first n rows. We call a function f : RD ˆ RD Ñ R whose purpose is to evaluate the compatibility of two vectors a scoring function. A normalization function π: RN Ñ ∆N´1 maps vectors in RN to N probabilities. Here, ∆N´1 “def !x P r0,1sN | řN

n“1 xn “ 1) is the pN ´1q-dimensional probability simplex. This notation is summarized in Tab. 1.

The Attention Mechanism. The attention mechanism works as follows. It takes a query vector q P RD and two matrices: The matrix K P RNˆD of keys and the matrix V P RNˆD of values and computes a weighted average of the value vectors based on the compatibilities of the key vectors and the query vector, as determined by a scoring function f. A formal definition is given below.

Definition 2.5 (Attention Mechanism). Let f be a scoring function and π a normalization function. Let q P RD be a query vector and let K “ `

` ˘ P RNˆD and V “

kJ1 ;¨¨¨ ;kJN

˘ P RNˆD be matrices of keys and values, respectively. An attention mechanism Att: RD ˆ RNˆD ˆ RNˆD Ñ RD is defined as

v1J;¨¨¨ ;vNJ

ÿN

Attpq,K,Vq “def

snvn, (6)

n“1

where

s “def π pf pq,k1q,...,f pq,kNqq (7)

Symbol Type Meaning rNs Ă N The set t1, . . . , Nu for N P N.

y P Σ A symbol, element of Σ.

Σ, Σ, Σ alphabet Σ is a set of symbols, Σ def“ Σ Y tBOSu, Σ def“ Σ Y tEOSu y P Σ˚ A string over Σ. yji P Σ˚ A substring of y, a string.

y P t0, 1u|Σ| One-hot encoding of the symbol y P Σ. D P N Size of the contextual representations in the transformer.

∆N´1 Ď RN The N ´ 1-dimensional probability simplex. f RD ˆ RD Ñ R A scoring function. π RN Ñ ∆N´1 A normalization function.

Q, K, V , O RD Ñ RD The query, key, value, and output functions.

F RD Ñ RD The final transformer LM transformation function. enc Σ˚ Ñ RD The string representation function.

r Σ ˆ N Ñ RD The position-augmented representation function.

L P N Number of layers. H P N Number of heads. H RHD Ñ RD The head combining function.

p¨ ; ¨ ¨ ¨ ; ¨q Vertical concatenation operator of vectors or matrices.

Table 1: A summary of the notation used in the paper.

is the vector of normalized scores between the query q and the keys in K.

The most standard implementation of the scoring function f is the (scaled) inner product f pq,kq “def xq,ky. Some of our results rely on this standard formulation. However, some also rely on the more general, but still simple and just as efficiently computable scoring functions.

The Transformer Architecture. A transformer layer uses the attention mechanism to compute augmented representations zt “ Attpqt,Kt,Vtq of the input representations Xt “ px1;¨¨¨ ;xtq. The query qt, the keys Kt, and values Vt are all transformations of the input representations Xt.

Definition 2.6. Let Q,K,V,O: RD ÑRD be the query, key, value, and output functions. A transformer layer is a function L: RTˆD Ñ RTˆD

that computes

˘ P RTˆD (8) for t P rTs where

L`

˘ “ `

xJ1 ;...;xJT

zJ1 ;...;zJT

at “def Attpqt,Kt,Vtq ` xt P RD (9a) zt “def O patq ` at P RD. (9b)

Here, we define

qt “def Qpxtq P RD (10a) Kt “def ´K px1qJ ;¨¨¨ ;K pxtqJ¯ P RtˆD (10b) Vt “def ´V px1qJ ;¨¨¨ ;K pxtqJ¯ P RtˆD. (10c)

Note: For simplicity, we do not include layer normalization.

Without further modification, the transformations applied by the transformer layer are positioninvariant, which necessitates the addition of explicit positional information.

- Definition 2.7. A position-augmented symbol representation function r: Σ ˆ N Ñ RD is a function representing symbols and their positions as D-dimensional vectors.

Position-augmented symbol representation functions are often implemented as an addition or concatenation of separate symbol-only and positiononly representation functions (Vaswani et al., 2017). Here, we define it more generally as any function of the symbol and its position.

- Definition 2.8. A static encoding R is a function R: ΣT Ñ RTˆD defined for any T P N as

Rpyq “def ´rpy1,1qJ;¨¨¨ ;rpyT,TqJ¯. (11)

Multiple transformer layers are stacked into a transformer, which computes the (deep) contextual representations of all symbols in the string.

- Definition 2.9. For L P N, let Lℓ for ℓ P rLs be transformer layers. Let R be a static encoding. An L-layer transformer T is defined as

T pRq “def LL ˝ ¨¨¨ ˝ L1 ˝ R. (12)

A transformer computes the contextual representations of the symbols y “ y1 ¨¨¨yT as

˘ “def T pRqpyq. (13)

`

xL1J;¨¨¨ ;xLTJ

If R is clear from the context or arbitrary, we will omit it as an argument to T and just write T pyq.

- Definition 2.10. Let T be a transformer, F : RD Ñ RD the final representation transformation function, and y P Σ˚ with |y| “ T. We define

encpyq “def F

`

xLT

˘

, (14)

where xLT is the representation of the Tth symbol in y computed by T , i.e.,

`

x1LJ;¨¨¨ ;xTLJ

˘ “ T pyq.

Transformer Language Models. So far, we have only defined how the transformer architecture can be used to compute the contextual representations of the symbols. To complete the definition, we define a transformer language model as follows.

- Definition 2.11. A transformer LM pT is the representation-based autoregressive LM with the representation function enc from Eq. (14). That is, pT defines the conditional probability distributions

pT pyt | yătq “def softmaxpEencpyătqqy

t

. (15) 2.3.1 Variants of the Attention Mechanism

In this subsection, we discuss many common variants of the attention mechanism. First, multiheaded attention uses H attention heads to compute H representations of the symbols in the string. The representations constructed by the different attention heads are concatenated into a long vector and projected down to the output size of a single head with a head-combiner function H.

- Definition 2.12. For L,H P N, let Lhℓ : RTˆD Ñ RTˆD,ℓ P rLs,h P rHs be transformer layers. Define Lℓ: RTˆD Ñ RTˆHD as

Lℓ pXq “def ´L1ℓ pXqJ ;¨¨¨ ;LHℓ pXqJ¯J . (16)

Furthermore, let H: RHD Ñ RD. An L-layer transformer with H heads computes:

T pRq “def LL ˝ H ˝ ¨¨¨ ˝ H ˝ L1 ˝ R, (17)

where H is applied row-wise to project the representations of H heads to RD.

Attention types. Attention weights are computed by normalizing the scores f pq,k1q,...,f pq,ktq. The choice of the projection function π determines the type of attention and has concrete implications on representational capacity (Hao et al., 2022a).

- Definition 2.13. Hard attention is computed with the hardmax projection function:

hardmaxpxqd “def

#

1 m if d P argmaxpxq

0 otherwise

(18)

for d P rDs, where x P RD and m “def |argmaxpxq| is the cardinality of the argmax set.

We also introduce sparse attention, which uses the sparsemax normalization function to compute the attention weights.

- Definition 2.14. Sparse attention is computed with the sparsemax projection function:

sparsemaxpxq “def argmin pP∆D´1

∥p ´ x∥22. (19)

### 3 Hard Attention Transformer LMs

This section presents a set of results describing the representational capacity of hard attention transformer LMs. Concretely, we show that transformer LMs with hard attention can represent n-gram LMs, either using n ´ 1 heads (Theorem 3.1) or n ´ 1 layers (Theorem 3.2). Simulation is possible even with a single head and a single layer (Theorem 3.3) but might require a more elaborate set of non-linear transformations and positional encodings whose precision scales linearly with the string length.

Theorem 3.1. For any n-gram LM, there exists a weakly equivalent single-layer hard attention transformer LM with n ´ 1 heads.

Proof intuition. Given an n-gram LM p, we can construct a weakly equivalent LM pT defined by a transformer T that looks back at the preceding n ´ 1 positions using n ´ 1 heads, each of them uniquely attending to exactly one position. The symbols attended to can be used to identify the full history, which can be used to access the conditional distribution over the next symbol. This is illustrated in Fig. 1. See Appendix B.2 for the full proof. ■

Theorem 3.1 shows that transformer LMs with hard attention can represent n-gram LMs, establishing, to the best of our knowledge, the first concrete relationship between transformer LMs and probabilistic languages. A natural follow-up question then is whether n ´ 1 heads are necessary to correctly simulate an n-gram LM. Besides aiming to illuminate different mechanisms enabling the implementation of classical LMs, this

question also follows the line of inquiry about the uniqueness and interpretability of the representations of formal models by neural LMs (Liu et al., 2023). The following two theorems show that the intuitive construction using n ´ 1 heads is far from unique: Theorem 3.2 shows that a similarly simple simulation is possible with n ´ 1 layers and a single head, while Theorem 3.3 shows that even a transformer LM with a single head and a single layer can simulate an n-gram LM, albeit with more complex position invariant transformation F. This suggests that there is no canonical way of determining whether a transformer LM has learned an n-gram LM by looking at individual components (e.g., positions attended to by the different heads).

- Theorem 3.2. For any n-gram LM, there exists a weakly equivalent pn ´ 1q-layer hard attention transformer LM with a single head.

Proof intuition. Whereas the transformer LM constructed in Theorem 3.1 used n ´ 1 heads to look at all the n ´ 1 positions of interest, an n ´ 1layer transformer LM can use the n ´ 1 layers to look back at the immediately preceding position and copy it forward n ´ 1 times (keeping the current symbol there as well). After n ´ 1 layers of such transformations, the entire history can be read from the current contextual representation. See Appendix B.2 for the full proof. ■

Apart from using hard attention, both transformer LMs used in Theorems 3.1 and 3.2 rely on modeling assumptions often found in practical implementations of the transformer: The transformations Q,K, and V are linear functions, the scoring function is implemented as a dot-product and positional encodings are bounded. This makes the results comparable to practical implementations. The following theorem, in contrast, shows that if we permit the use of less standard components, transformer LMs can identify the history of interest using only a single head and a single layer.

- Theorem 3.3. For any n-gram LM, there exists a weakly equivalent single-layer hard attention transformer LM with a single head.

Proof intuition. The bulk of this construction lies in the encoding ytt´´n1`1 in a vector that can be constructed by a single attention head in one layer. This is done by an attention head that (1) puts non-zero attention on only the previous n ´ 1 symbols and (2) encodes the identities and the positions of symbols in a |Σ|-dimensional value vector. The

value vector can then be decoded into a one-hot encoding of ytt´´n1`1 by an n´1-layer MLP that defines F, which allows us to match the conditional probabilities of the n-gram LM as in Theorems 3.1 and 3.2. See Appendix B.2 for the full proof. ■

### 4 Sparse Attention Transformer LMs

While the results in §3 concretely characterize the abilities of hard attention transformer LMs, the assumption of hard attention is somewhat removed from practical implementations of the model. Those most often rely on differentiable normalization functions, such as the softmax.3 However, the full support of the softmax function makes the connection to formal models of computation difficult (Hahn, 2020). To bring the theoretical models closer to practical implementations yet still be able to make clear analogies to formal models of computation, we now consider sparse attention transformers, which use the sparsemax normalization function. The sparsity allows sparse attention transformers to simulate n-gram LMs just like hard attention transformers while relying on differentiable operations.

Theorem 4.1. For any n-gram LM, there exists a weakly equivalent single-layer sparse attention transformer LM with n ´ 1 heads.

Proof intuition. The intuition behind the simulation with sparse attention is similar to the hard attention one; each head attends to a single position, as illustrated in Fig. 1. Effectively, the construction results in a sparse attention transformer that simulates hard attention. In contrast to Theorems 3.1 and 3.2, we here require a model with unbounded positional encodings and a non-linearly transformed dot-product scoring function. Intuitively, the unbounded positional encodings are required to scale the unnormalized attention scores to differ enough for the sparsemax to focus on a single position. The rest of the proof follows that of

- Theorem 3.1; see Appendix C for the details. ■
- Theorem 4.1 (representing an n-gram LM with

n´1 heads) could naturally be extended to analogs

3As noted in §1, the analysis of soft attention transformers requires a different type of analysis in terms of approximation of the probabilities. A complete study would have to consider the approximation over arbitrarily long strings (since Σ˚ is an infinite set), which is difficult by simply scaling model parameters to a large constant. We focus on exact simulation here, but conjecture that soft attention transformers can approximate LMs whose average string length is finite.

of Theorem 3.2 (representing an n-gram LM with n ´ 1 layers) and Theorem 3.3 (representing an n-gram LM with a single head and a single layer) using a similar adaptation of the construction from the hard attention case to the sparse attention one as in Theorem 4.1.

### 5 Space Complexity

In §3 and §4, we describe lower bounds that tell us what types of probability distributions transformer LMs can represent, but do not say how efficiently they can do so. The space complexity of simulating n-gram LMs is discussed in this section. We focus on hard-attention transformer LMs with multiple heads or multiple layers (Theorems 3.1 and 3.2) since their modeling assumptions (bar hard attention) are closest to practical implementations. The constructive proofs of Theorems 3.1 and 3.2 allow us to directly analyze the space requirements for the simulation of n-gram LMs, both in terms of (1) the size of the contextual representations Xhℓ as well as (2) the number of bits required to represent the individual entries of the vectors xhℓ,t.

5.1 Scaling with Respect to the Number of Computational Steps

We first address the second point. Specifically, we are interested in how the number of bits scales with respect to t, the number of computational steps performed during the generation of a string y P Σ˚. As summarized by Tab. 2, the models constructed in the proofs of Theorems 3.1 and 3.2 use positional encodings with entries of the form b

1

t for t P N, i.e., they contain square roots of rational numbers. This makes the scaling of the space complexity difficult, as square roots of rational numbers are not in general representable with a finite number of bits. While this might seem discouraging, we emphasize that these specific positional encodings were only used to keep the contextual representations bounded and the scoring function in line with the original formulation (Vaswani et al., 2017) and concurrent work (Merrill and Sabharwal, 2023a). A closer look at the constructions in the proof of Theorems 3.1, 3.2 and 4.1 reveals that simpler (but unbounded) positional encodings with a less standard scoring function can be used to the same effect. In particular, we can use positional encodings that contain entries of the form t, which only require a logarithmic number of bits with respect to t. Since such scaling is required to uniquely encode

the positional information in general (Merrill and Sabharwal, 2023c), this represents an asymptotically optimal scaling of the space complexity of the contextual representations.4

Importantly, n-gram LMs are real-time: they, by definition, generate a symbol at each step of the computation. This means that the scaling of the space complexity with respect to the number of computation steps coincides with its scaling with respect to the length of the generated string y—the scaling is logarithmic in |y|. This is in contrast to non-real-time models of computation which might not generate a symbol at each step of the computation; while those might still require an asymptotically optimal scaling with respect to t, the additional computational steps that do not emit any symbol might mean that the space complexity is unbounded with respect to the length of the generated string. An example of such a model is a transformer LM simulating a (probabilistic) Turing machine, which would require the model to not emit symbols at some points of the computation (Nowak et al., 2023, 2024).

5.2 The Dimensionality of the Contextual Representations

We now discuss the size of the contextual representations required for the simulation of n-gram LMs. From a high level, we have to consider two stages: (1) the contextual representations xhℓ,t of the different layers and heads and (2) the size of the final representation encpyq. The contextual representations xhℓ,t in stage (1) are composed of the symbol and positional encodings. The symbol representations include (two copies of) the one-hot encodings while the positional encodings include between two and 2n dimensions encoding positional information. This means that the per-head and per-layer space complexity scales with |Σ| and n. For stage (2) we use the one-hot encodings of the entire history ytt´´n1`1, with which we index the matrix of |Σ|n´1 conditional probabilities defined by the n-gram LM. While the size of encpyq is theoretically only lower-bounded by |Σ| (Yang et al., 2018; Svete and Cotterell, 2023),5 reducing its size requires a lower-

- 4The construction in Theorem 3.3, in contrast, relies on

encoding the entire preceding string in a single dimension with one digit per position in the string, which requires a number of bits that scales linearly with respect to the string length. A simplification to a logarithmic number of bits does not seem as straightforward.

- 5This is because the (logits of the) conditional probabilities

can span a |Σ|-dimensional space.

rank decomposition of the matrix of conditional probabilities and a corresponding reparametrization of the contextual symbol representation. This might require a blow-up in the number of bits required to represent individual dimensions, or, in general, result in a real-valued vector that could not be represented on a finite-precision system. Altogether, most of the space complexity of the contextual representations comes from the one-hot encodings in encpyq, which require |Σ|n´1 dimensions.

The discussion in this section can be summarized by the following theorem.

Theorem 5.1. Let p be an n-gram LM over the alphabet Σ. There exists a weakly equivalent hardattention transformer LM with contextual representations x of size O pn|Σ|q and representation encpyq of size O `|Σ|n´1˘

. Each of x’s entries can be represented with O plog2 p|y|qq bits for y P Σ˚.

### 6 Discussion and Related Work

To the best of our knowledge, §3 and §4 provide the first results on the probabilistic representational capacity of transformer LMs.

The relevance of n-gram LMs to modern LMs. One might rightfully question the utility of connecting the state-of-the-art language modeling architecture to n-gram LMs. LMs based on the n-gram assumption indeed constitute some of the simplest and least expressive classes of probability distributions. Nevertheless, n-gram LMs provide a useful playground and theoretical foundation for contextualizing and interpreting the inner workings of modern LMs. For example, n-gram LMs have been found to comprise a crucial component of in-context learning, where attention heads in different layers of the model together identify individual n-grams and base their predictions on the presence of such n-grams (Olsson et al., 2022; Akyürek et al., 2024). The presence of specific n-grams might therefore present the foundations of in-context-based knowledge of transformer LMs. As such, understanding the requirements for correct simulation of n-gram LMs is important for a thorough grasp of the abilities of LMs to learn in context. Encouraging n-gram-LM-based learning has also been observed to aid in-context learning abilities (Akyürek et al., 2024). Existing work has also linked n-gram LMs to other neural network architectures such as one-dimensional convolutional neural networks (Merrill, 2019). Moreover, the theoretical grounding in classical formal language

theory makes precise statements about n-gram LMs possible while their simplicity makes them inherently interpretable and easy to analyze. n-gram LMs also have a cognitive interpretation (Jäger and Rogers, 2012). Most importantly, however, ngram LMs lend themselves to paralellized processing, which affords a succinct and natural connection to transformer LMs and has been suggested to be a crucial part of any theoretical treatment of transformer LMs (Strobl et al., 2023; Merrill and Sabharwal, 2023c).

Understanding the limitations and abilities of hard attention transformer LMs. Our results are the strongest in the hard attention setting, which follows the trend of using hard attention in theoretical treatments. Hard attention makes singling out the important aspects of the string (in our case, the history) possible. Concretely, we showcase three different mechanisms that make it possible for transformer LMs to implement n-gram LMs with hard attention and investigate the role of the number of heads and layers. Particularly, our constructions suggest a possible mechanism in which the different transformer heads or layers can specialize in focusing on different positions in the string, a feature that has previously been suggested as an explanation of how transformer LMs process strings (Elhage et al., 2021) and has been observed in trained transformer LMs (Olsson et al., 2022; Akyürek et al., 2024).6 Our presentation of multiple orthogonal mechanisms that can simulate n-gram LMs equivalently well is another confirmation of the observation that algorithmic principles learned or implemented by neural LMs do not always correspond to a single intuitive implementation of a formal model of computation, which has concrete implications on interpretability methods for the architecture. This has been observed in practical scenarios and warns us that focusing on individual components of the model (that is, using myopic interpretability methods) might result in misleading interpretability results (Wen et al., 2023).

Probabilistic representational capacity. We augment existing literature by providing an explicit connection between transformer LMs and n-gram

6Note that the constructions presented in this paper are purely meant to showcase the existence of a mechanism that can be used to simulate n-gram LMs; we do not suggest that the same mechanisms will be employed by models used in practice, which is also why do not present any empirical results. For example, the use of the sparse one-hot encodings differs from the standard dense representations of symbols.

language models. In line with work comparing transformers to circuits (Merrill and Sabharwal, 2023c,a; Weiss et al., 2021; Hao et al., 2022b; Merrill and Sabharwal, 2023b; Merrill et al., 2022; Chiang et al., 2023; Angluin et al., 2023), we also show the utility of analyzing transformers with parallelizable models of computation, which go hand in hand with the parallelizable nature of the transformer architecture. Moreover, the formalization of an LM with the output matrix indexed by the contextual representation (cf. Definition 2.11) makes it easy to connect existing results on the expressivity of the transformer architecture with the probabilistic setting by defining a mapping from the contextual representation to the conditional probability distribution through the output matrix. We suppose other simple and parallelizable classes of distributions might lend themselves well to similar probabilistic treatments; probabilistic analysis may be particularly interesting in the context of circuit complexity with sigmoid-activated circuits (Maass et al., 1991), (Smolensky et al., 1996, Chapter 4).

Connection to (sub-)regular languages This work focuses on connecting transformer LMs to n-gram LMs, which are a special instance of the more general class of sub-regular LMs. In words, sub-regular LMs are LMs that can be described without using the full power of the probabilistic finite-state automata (Jäger and Rogers, 2012). In this sense, sub-regular LMs fall below regular languages in the Chomsky hierarchy and themselves define their own hierarchy (Jäger and Rogers, 2012; Heinz and Rogers, 2013; Avcu et al., 2017). For example, some classes of sub-regular languages do not require sequential processing of the input string that is usually required by finite-state automata for correct recognition. The intuitive connection between transformer LMs and n-gram LMs encourages further work on the connections between other classes of (sub-)regular LMs and transformer LMs. Transformers have been linked to (sub)-regular languages by Yao et al. (2021) and Liu et al. (2023). Yao et al. (2021) study the ability of transformers to generate bounded hierarchical languages using a transformer but do not extend their analysis to the fully probabilistic case.7 Liu et al. (2023) connect transformers with both hard and soft attention

7The natural connection of the transformer architecture to both bounded hierarchical languages as well as to n-gram LMs is interesting since those two classes of LMs can both be represented particularly efficiently by recurrent neural LMs (Svete et al., 2024).

to general finite-state automata (FSAs), which define a strictly larger set of languages than n-gram languages. This additional generality, however, comes with some caveats. Liu et al. (2023, Theorem 1)—their most general result—relies on a model whose depth scales (logarithmically) with the string length. As such, no particular finitesize transformer construction can simulate FSAs on strings of arbitrary length (which is required for equivalence). This is in contrast to our results, which feature transformers of fixed size with respect to the string length. While Liu et al. (2023, Theorem 2) also provides a result using a finitedepth transformer for a subset of FSAs, that construction results in a network much bigger and deeper than ours. For example, their construction

would result in ´|Σ|n´1¯2 pn ´ 1qlog |Σ| layers in contrast to n ´ 1 layers with a single head in our case. Our focus on n-gram LMs thus allows for a more compact and simpler representation. Worth noting is that despite being close to our analysis, none of the related work treats probability distributions over strings but rather focuses on the binary decision of whether a string belongs to a language or not based on the conditional probabilities output by the model, or, in the case of Liu et al. (2023), only the computation of state sequences. Transformer LMs are studied probabilistically by Xie et al. (2022), who provide a Bayesian interpretation of their ability to learn in context by studying the learning abilities of hidden Markov models (HMMs). While HMMs are equivalent to probabilistic finite-state automata, Xie et al. (2022) do not connect the in-context learning ability to concrete representational capacity results. Rather, they seek to explain the behavior of in-context learning as implicit Bayesian inference.

### 7 Conclusion

We study the representational capacity of transformer LMs with n-gram LMs. We show how the parallelizable nature of n-gram LMs is easy to capture with the transformer architecture and provide multiple lower bounds on the probabilistic representational capacity of transformer LMs. Concretely, we show that transformer LMs can represent ngram LMs both with hard and sparse attention, exhibiting multiple mechanisms transformer LMs can employ to simulate n-gram LMs. Altogether, our results reinforce the utility of non-sequential models of computation for the study of transformers,

particularly in the language modeling setting.

### Limitations

We connect transformer LMs to n-gram LMs because of their parallelizable nature and their traditional popularity in NLP. However, n-gram LMs describe a very simple class of LMs, meaning that the lower bounds are somewhat less relevant than the characterization in terms of more expressive formal models of computation would be. Accordingly, we expect that the lower bounds are somewhat loose and that transformer LMs can represent more than n-gram LMs, which is also in line with the empirical success of transformer LMs. We leave it to future work to tighten the established lower bounds.

As with most theoretical investigations of transformers, our results are strongest and the most precise in the hard attention setting. However, hard attention is not used in practice, which limits the applicability of the results. The constructions presented in this paper are also purely meant to showcase the existence of a mechanism that can be used to simulate n-gram LMs. They do not suggest that the same mechanisms will be learned by models used in practice. Indeed, the very sparse representations are not in line with the common dense contextual representations usually learned by trained models.

We also only focus on lower bounds of the representational capacity. We do not consider any upper bounds and existing results for similar models to ours suggest that the lower bound is indeed somewhat loose (Yao et al., 2021).8 That is, we expect that transformer LMs can represent much more than n-gram LMs, and expect that many of the existing results on the computational power of such models can be extended to the probabilistic setting.

While we present a comprehensive analysis of transformer LMs in the context of n-gram LMs, we do not consider various aspects of the relationship that could be interesting. This is done to keep the presentation focused and concise. For example, we do not consider whether such simulations can be learned from data, an interesting avenue for future research. Lastly, note that while we focus here specifically on the commonly deployed transformer-based language models, there are many other interesting applications of

8For example, we cannot say that the lower bounds imply any limitations of hard attention transformer LMs.

transformers, such as encoder-only acceptors of unweighted languages. These applications are better covered by existing work.

### Ethics Statement

The paper provides a way to theoretically analyze language models. To the best knowledge of the authors, there are no ethical implications of this paper.

### Acknowledgements

Ryan Cotterell acknowledges support from the Swiss National Science Foundation (SNSF) as part of the “The Forgotten Role of Inductive Bias in Interpretability” project. Anej Svete is supported by the ETH AI Center Doctoral Fellowship. We thank Tim Vieira and Shannon Veitch for helpful discussions and comments on the manuscript.

### References

Ekin Akyürek, Bailin Wang, Yoon Kim, and Jacob Andreas. 2024. In-context language learning: Architectures and algorithms. arXiv preprint arXiv:2401.12973.

Dana Angluin, David Chiang, and Andy Yang. 2023. Masked hard-attention transformers and boolean rasp recognize exactly the star-free languages. arXiv preprint arXiv:2310.13897.

Enes Avcu, Chihiro Shibata, and Jeffrey Heinz. 2017. Subregular complexity and deep learning. arXiv preprint arXiv:1705.05940.

Lalit R. Bahl, Frederick Jelinek, and Robert L. Mercer. 1983. A maximum likelihood approach to continuous speech recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-5(2):179– 190.

J. Baker. 1975a. The DRAGON system–An overview. IEEE Transactions on Acoustics, Speech, and Signal Processing, 23(1):24–29.

James K. Baker. 1975b. Stochastic Modeling as a Means of Automatic Speech Recognition. Ph.D. thesis, Carnegie Mellon University, USA. AAI7519843.

Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. 2000. A neural probabilistic language model. In Advances in Neural Information Processing Systems, volume 13. MIT Press.

Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. J. Mach. Learn. Res., 3:1137–1155.

Yoshua Bengio, Holger Schwenk, Jean-Sébastien Senécal, Fréderic Morin, and Jean-Luc Gauvain. 2006. Neural Probabilistic Language Models, pages 137– 186. Springer Berlin Heidelberg, Berlin, Heidelberg.

Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. 2020. On the ability and limitations of transformers to recognize formal languages. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7096–7116, Online. Association for Computational Linguistics.

David Chiang and Peter Cholak. 2022. Overcoming a theoretical limitation of self-attention. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7654–7664, Dublin, Ireland. Association for Computational Linguistics.

David Chiang, Peter Cholak, and Anand Pillay. 2023. Tighter bounds on the expressivity of transformer encoders. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org.

Li Du, Lucas Torroba Hennigen, Tiago Pimentel, Clara Meister, Jason Eisner, and Ryan Cotterell. 2023. A measure-theoretic characterization of tight language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9744–9770, Toronto, Canada. Association for Computational Linguistics.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2021. A mathematical framework for transformer circuits. Transformer Circuits Thread.

Michael Hahn. 2020. Theoretical limitations of selfattention in neural sequence models. Transactions of the Association for Computational Linguistics, 8:156– 171.

- Yiding Hao, Dana Angluin, and Robert Frank. 2022a. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics, 10:800–810.
- Yiding Hao, Dana Angluin, and Robert Frank. 2022b. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics, 10:800–810.

Yiding Hao, William Merrill, Dana Angluin, Robert Frank, Noah Amsel, Andrew Benz, and Simon Mendelsohn. 2018. Context-free transductions with neural stacks. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 306–315, Brussels, Belgium. Association for Computational Linguistics.

Kenneth Heafield. 2011. KenLM: Faster and smaller language model queries. In Proceedings of the Sixth Workshop on Statistical Machine Translation, pages

187–197, Edinburgh, Scotland. Association for Computational Linguistics.

Kenneth Heafield, Ivan Pouzyrevsky, Jonathan H. Clark, and Philipp Koehn. 2013. Scalable modified KneserNey language model estimation. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 690–696, Sofia, Bulgaria. Association for Computational Linguistics.

Jeffrey Heinz and James Rogers. 2013. Learning subregular classes of languages with factored deterministic automata. In Proceedings of the 13th Meeting on the Mathematics of Language (MoL 13), pages 64– 71, Sofia, Bulgaria. Association for Computational Linguistics.

John Hewitt, Michael Hahn, Surya Ganguli, Percy Liang, and Christopher D. Manning. 2020. RNNs can generate bounded hierarchical languages with optimal memory. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1978–2010, Online. Association for Computational Linguistics.

Gerhard Jäger and James Rogers. 2012. Formal language theory: Refining the Chomsky hierarchy. Philos Trans R Soc Lond B Biol Sci, 367(1598):1956– 1970.

F. Jelinek. 1976. Continuous speech recognition by statistical methods. Proceedings of the IEEE, 64(4):532– 556.

F. Jelinek. 1990. Self-Organized Language Modeling for Speech Recognition, page 450–506. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.

S. C. Kleene. 1956. Representation of events in nerve nets and finite automata. In C. E. Shannon and J. McCarthy, editors, Automata Studies. (AM-34), Volume 34, pages 3–42. Princeton University Press, Princeton.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. 2023. Transformers learn shortcuts to automata. In International Conference on Learning Representations.

W. Maass, G. Schnitger, and E. D. Sontag. 1991. On the computational power of sigmoid versus boolean threshold circuits. In Proceedings 32nd Annual Symposium of Foundations of Computer Science, pages 767–776.

André F. T. Martins and Ramón F. Astudillo. 2016. From softmax to sparsemax: A sparse model of attention and multi-label classification. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ICML’16, page 1614–1623.

William Merrill. 2019. Sequential neural networks as automata. In Proceedings of the Workshop on Deep Learning and Formal Languages: Building Bridges,

pages 1–13, Florence. Association for Computational Linguistics.

- William Merrill and Ashish Sabharwal. 2023a. The expressive power of transformers with chain of thought. arXiv preprint arXiv:2310.07923.
- William Merrill and Ashish Sabharwal. 2023b. A logic for expressing log-precision transformers. arXiv preprint arXiv:2210.02671.
- William Merrill and Ashish Sabharwal. 2023c. The parallelism tradeoff: Limitations of log-precision transformers. Transactions of the Association for Computational Linguistics, 11:531–545.

William Merrill, Ashish Sabharwal, and Noah A. Smith. 2022. Saturated transformers are constant-depth threshold circuits. Transactions of the Association for Computational Linguistics, 10:843–856.

William Merrill and Nikolaos Tsilivis. 2022. Extracting finite automata from RNNs using state merging. arXiv preprint arXiv:2201.12451.

William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav. 2020. A formal hierarchy of RNN architectures. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 443–459, Online. Association for Computational Linguistics.

Marvin Lee Minsky. 1954. Neural Nets and the Brain Model Problem. Ph.D. thesis, Princeton University.

Franz Nowak, Anej Svete, Alexandra Butoi, and Ryan Cotterell. 2024. On the representational capacity of neural language models with chain-of-thought reasoning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Bangkok, Thailand. Association for Computational Linguistics.

Franz Nowak, Anej Svete, Li Du, and Ryan Cotterell. 2023. On the representational capacity of recurrent neural language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7011–7034, Singapore. Association for Computational Linguistics.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2022. In-context learning and induction heads. Transformer Circuits Thread.

Jorge Pérez, Pablo Barceló, and Javier Marinkovic.

2021. Attention is Turing-complete. Journal of Machine Learning Research, 22(75):1–35.

Holger Schwenk. 2007. Continuous space language models. Computer Speech & Language, 21(3):492– 518.

C. E. Shannon. 1948. A mathematical theory of communication. The Bell System Technical Journal, 27(3):379–423.

Hava T. Siegelmann and E. D. Sontag. 1992. On the computational power of neural nets. In Proceedings of the Fifth Annual Workshop on Computational Learning Theory, COLT ’92, page 440–449, New York, NY, USA. Association for Computing Machinery.

Paul Smolensky, Michael C. Mozer, and David E. Rumelhart. 1996. Mathematical Perspectives on Neural Networks. Psychology Press.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. 2023. Transformers as recognizers of formal languages: A survey on expressivity. arXiv preprint arXiv:2311.00208.

Anej Svete, Robin Shing Moon Chan, and Ryan Cotterell. 2024. A theoretical result on the inductive bias of RNN language models. arXiv preprint arXiv:2402.15814.

Anej Svete and Ryan Cotterell. 2023. Recurrent neural language models as probabilistic finite-state automata. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 8069–8086, Singapore. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2021. Thinking like transformers. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11080–11090. PMLR.

Kaiyue Wen, Yuchen Li, Bingbin Liu, and Andrej Risteski. 2023. Transformers are uninterpretable with myopic methods: A case study with bounded Dyck grammars. In Thirty-seventh Conference on Neural Information Processing Systems.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2022. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations.

Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. 2018. Breaking the softmax bottleneck: A high-rank RNN language model. In International Conference on Learning Representations.

Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. 2021. Self-attention networks can process bounded hierarchical languages. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3770–3785, Online. Association for Computational Linguistics.

### A Modelling Assumptions

As encouraged by Strobl et al. (2023), we provide in Tab. 2 a short summary of the assumptions behind the specific transformer architecture we consider in this work. This aims to facilitate the placement of the results in the context of existing work and to make the results more accessible to the reader.

Lower bound PE Precision Attention Attention Architecture Notes

¨ ˝

b

˛ ‚

1

b1 ´t`tk`1k

Q n-gram LMs

Q, R hard decoder-only n ´ 1 heads, 1 layer Theorem 3.1

k“0,...,n´1

b

˛

¨

1

b1 ´t 1t

b

Q n-gram LMs

Q, R hard decoder-only 1 head, n ´ 1 layers Theorem 3.2

1

‹ ‚

˚ ˝

b1 ´t`t1`11

Q n-gram LMs 1, t, 10´t Q, R hard decoder-only 1 head, 1 layer Theorem 3.3 Q n-gram LMs 1, t Q, R sparse decoder-only n ´ 1 heads, 1 layer Theorem 4.1

Table 2: A summary of the main assumptions about the models in the style of Strobl et al. (2023, Table 1). Hard attention here refers to average-hard in the vocabulary of Strobl et al. (2023).

### B Proofs: Hard Attention

This section provides detailed proofs of all theorems about the representational capacity of hard-attention transformer LMs stated in the main part of the paper.

- B.1 Computing Logical AND with an MLP Definition B.1. A ReLU-activated multi-layer-perceptron (MLP) MLP: RN Ñ RM is a function defined as the composition of functions f1,¨¨¨ ,fL

MLPpxq “def pfL ˝ fL´1 ˝ ¨¨¨ ˝ f1qpxq (20) where each fℓ for ℓ P rLs is defined as

fℓ pxq “def ReLUpWℓx ` bℓq ℓ P rL ´ 1s (21a) fL pxq “def WLx ` bL (21b)

where Wℓ P RNℓˆMℓ is a weight matrix with dimensions Nℓ and Mℓ specific to layer ℓ, bℓ P RMℓ is a bias vector. We refer to MLPs by the number of hidden layers, e.g., a one-layer-MLP is an MLP with one hidden layer.

In our construction, simulating a n-gram LM with a transformer LM requires a component of the transformer to perform the logical AND operation between specific entries of binary vectors x P BD. The following lemma shows how this can be performed by an MLP with appropriately set parameters.

- Lemma B.1. Consider m indices i1,...,im P rDs and vectors x,v P BD such that vi “ ti P ti1,...,imuu, (22)

vJx ´ pm ´ 1q˘ that

`

i.e., with entries 1 at indices i1,...,im. Then, it holds for the MLP MLPpxq “def ReLU

MLPpxq “ 1 if and only if xik “ 1 for all k “ 1,...,m. (23) In other words,

MLPpxq “ xi1 ^ ¨¨¨ ^ xim. (24)

Proof. By the definition of v, vJx ď m for all x P BD. Furthermore, vJx “ m if and only if xik “ 1 for all k “ 1,...,m. The ReLU function maps all other values to 0 and does not change the output value 1 in this case. ■

#### B.2 Proofs of the Hard Attention Case

This subsection contains all proofs of the representational capacity of hard attention transformer LMs. We tackle three cases: the simulation with n ´ 1 heads and a single layer, with n ´ 1 layers and a single head, and lastly, the simulation with a single head and a single layer. All sections first outline the intuition behind the proofs and then provide the details in the form of finer-grained lemmata.

- B.2.1 Simulation with n ´ 1 Heads: The Intuition We now outline the intuition behind the construction of a hard attention transformer LM simulating an n-gram LM, as first presented in Fig. 1.9 To ease the exposition, we start with the final step of the

construction: Assuming we have identified the appropriate history ytt´´n1`1 after combining the head values using the head combining function H, we show how pT can encode the conditional probability distribution ppyt | yt´n`1:t´1q. The intuition of this step is simple: Knowing what the individual p

`

˘

yt | ytt´´n1`1

for yt P Σ are, we can simply put their logits into a vector and combine the constructed vectors for all possible histories into the output matrix E:10,11

`

˘

yt | ytt´´n1`1

“def log p

(25)

Ey,yt´1

t´n`1

`

˘

In the following, we write encpyătq as shorthand notation for encpyătq “def F

(i.e., the representation which is linearly transformed by E to compute ppyt | yătq after normalization) where XL “ T pRqpyătq. If we one-hot encode the identified history with T as

xLt´1

encpyătq “def ytt´´n1`1 (26)

we can then, using the formulation of the transformer LM from Definition 2.11, use the encpyătq to look up the appropriate column in E containing the logits of the conditional probabilities given the identified

`

˘

y | ytt´´n1`1

history for all possible yt P Σ, i.e., pE encpyătqqy “ log p

.

We now consider the preceding step of the simulation: Identifying the history given that the n´1 heads identified the symbols y1,...,yn´1 in the positions they attended to. If we concatenate the values of the n ´ 1 heads into a vector v, this vector of size pn ´ 1q|Σ| will contain the multi-hot representation of the history of interest:

¨ ˚ ˝

˛ ‹ ‚ (27)

y1 . yn´1

v “

and vi|Σ|`j “ 1 if and only if mpyiq “ j for a bijection m: Σ Ñ “|Σ|‰

that determines the indices of the one-hot representations of the symbols. We would then like to transform this vector into a vector u P R|Σ|n´1 such that

ui “ 1 ðñ i “ spy1,...,yn´1q (28)

Ñ “|Σ|n´1‰

for a bijection s: Σlooooomooooonˆn´1¨¨¨timesˆ Σ

. This can be equivalently written as

jq “ 1 for all j “ 1,...,n ´ 1 (29)

ui “ 1 ðñ vj|Σ|`mpy

where i “ spy1,...,yn´1q. This is an instance of performing the logical AND operation, which can be implemented by an MLP as described in Lemma B.1. This MLP will form the transformation H combining the information obtained from all the heads of the transformer.

This brings us to the final part of the proof: Identifying the symbols at the previous n ´ 1 positions by the n ´ 1 transformer heads. To show how this can be done, let us consider the parameters we can still set to define a transformer:

9For simplicity, we disregard the role of residual connections in the following outline. Residual connections are, however, considered in the full proof later.

- 10To be able to take the log of 0 probabilities, we work over the set of extended reals R “ R Y t´8, 8u.

- 11Throughout the paper, we implicitly index the matrices directly with symbols and histories. We assume that the symbols and

histories are ordered in some way and that the matrices are ordered accordingly.

- • The position-augmented symbol representations r. Inspired by concurrent work from Merrill and Sabharwal (2023a), we use the representations of the form

rpyt,tq “

¨

˚ ˝

b yt

1

b1 ´t 1t

b

1

b1 ´t`t1`11 b .

1

b1 ´t`tn`´n11´1

˛

‹ ‚

P R2n (30)

This results in vectors rpyt,tq of size |Σ| ` 2 ` 2pn ´ 1q. Note that such a symbol representation function can be implemented by concatenating or adding symbol- ( yt ) and position(b

1 t,...,b1 ´ t`n1´1) specific components, which is in line with most practical implementations

of the transformer architecture.

- • The attention scoring function f. We will use the standard dot-product scoring function f pq,kq “def xq,ky. (31)

f will, together with the positional encodings, allow us to easily single out the relevant positions in the string.

- • The parameters of each of the attention heads, that is, the transformations Q, K, and V . Each of those will take the form of a linear transformation of the symbol (and positional) representations. We describe them and their roles in more detail below.

The parameters of all the heads will be identical, with the only difference being a single parameter that depends on the “index” of the head, h. In the following, we describe the construction of a single head. At any time step t (i.e., when modeling the conditional distribution ppyt | yătq), the head h will attend to the symbol at position t´h, yt´h. In Fig. 1, for example, Head 3 attends to the position t´3, which is denoted by the stronger arrow to that position. We now describe the individual transformations Qh, Kh, Vh, and Oh of the head h. All of them will be affine transformations. Since we are considering only the first layer of the transformer, we can think of the inputs to the layer as the original symbol representations together with their position encodings (rather than some contextual representations at higher levels). As mentioned, the head h will be responsible for identifying the symbol at position t ´ h. Therefore, we want it to put all its attention to this position. In other words, given the query qt´1, we want the attention function in Eq. (31) to be uniquely maximized by the key of the symbol at position t ´ h. Notice that, therefore, the key does not have to depend on the identity of the symbol at position t ´ h—only the positional information matters. Let us then consider the following query and key transformations for head h:

¨ ˝

b

˛ ‚ (32)

1

b1 ´t 1t

Qh: rpyt,tq ÞÑ

˛ ‚. (33)

¨ ˝

b

1

b1 ´t`th`1h

Kh: rpyt,tq ÞÑ

Given such a query and such keys, the scoring function computes

C¨ ˝

b

˛ ‚,

¨ ˝

b

˛ ‚

G. (34)

1

1

b1 ´j`jh`1h

b1 ´t 1t

f pqt,kjq “

Eq. (34) is an inner product between two unit vectors, and is therefore maximized if and only if they are the same, that is, if j “ t ´ h. This is exactly the position that we want the head h to attend to.12 Intuitively, both transformations keep only the positional information. The query transformation “injects” the knowledge of which position should maximize the attention score, while the key transformation simply “exposes” the positional information about the symbol. The constants 1 and ´1 and the index of the position ensure that the inner product simply computes the difference between the position of the symbol and the position of interest.

This leaves us with the question of how to use the position of the symbol of interest (t ´ h) to extract the one-hot encoding of the symbol at that position. Due to the information contained in the symbol representations rpyjq, this is trivial:

V : rpyt,tq ÞÑ yj . (35) With this, the identity of the symbol is carried forward through the attention mechanism. Notice that the only head-depend transformation is the query transformation—it depends on the index of the head, determining the position of interest, meaning that every head defines a different query transformation, while the keys and values transformations are the same among all heads. This concludes the outline of the proof.

- B.2.2 Simulation with n ´ 1 Heads: Proofs This subsection formally proves the construction intuited in Appendix B.2.1 by proving a sequence of lemmata that formalize each of the steps described in the intuition. Specifically,

- 1. Lemma B.2 shows how the one-hot encodings of individual symbols in the history can be combined into the one-hot encoding of the history.
- 2. Lemma B.3 shows that the scoring function is maximized at the position of interest.
- 3. Lemma B.4 shows how the one-hot encodings of the symbols in the history can be identified by the hard attention mechanism.
- 4. The proof of Theorem 3.1 shows how the construction of the one-hot encoding of the current history allows us to define the appropriate next-symbol conditional distribution of the n-gram LMs.

- Lemma B.2. Let T be a transformer with H “ n ´ 1 heads. Let zh “ yt´h be the output of the hth head at time t. Then, there exists a function H implemented by a single-layer MLP such that

H pz1,...,zn´1q “ ytt´´n1`1 . (36) Proof. Let y “ y1 ...yn´1 P Σn´1 and let i be the index in

“|Σ|n´1‰

that corresponds to y. Furthermore, let i1,...,in´1 be the indices corresponding to the symbols y1,...,yn´1 in z1,...,zn´1. Then, we have

ytt´´n1`1 i “ 1 ðñ z1,i1 “ 1 ^ ¨¨¨ ^ zn´1,in´1 “ 1 (37)

Eq. (36) is an instance of the logical AND operation on the indices encoding the individual histories, which can be implemented by an MLP as shown in Lemma B.1. ■

The next lemma presents a useful equality about the standard dot-product attention scoring function: For unit vectors, the attention score is maximized if and only if the vectors are identical. We will use this fact in our construction to attend to particular positions in the string.

- Lemma B.3. Given a fixed t P N, f, define

˛ ‚

C¨ ˝

b

˛ ‚,

¨ ˝

b

G (38)

1

1

b1 ´j`jh`1h

b1 ´t´t1´11

g pjq “def

12Note that while the choice of the positional encodings in this construction is uncommon in practice, the popular sinusoidal positional encodings (Vaswani et al., 2017) have also been linked to the ability of the transformer-based models to attend to specific positions of interest based on linear transformations of the positional encodings (Vaswani et al., 2017).

for j P rt ´ 1s. Then, g is maximized at j “ t ´ 1 ´ h:

C¨ ˝

¨ ˝

b

˛ ‚,

¨ ˝

˛ ‚

G˛ ‚“ t ´ 1 ´ h. (39)

b

1

1

b1 ´j`jh`1h

b1 ´t´t1´11

##### argmax

jPrt´1s

Proof. The two arguments to the inner product in Eq. (39) are unit vectors. Inner products of unit vectors are at most 1, with the maximum achieved only if the two vectors are identical. This means that the function in Eq. (39) is maximized if and only if

¨ ˝

b

˛ ‚“

¨ ˝

b

˛ ‚ ðñ (40a)

1

1

b1 ´j`jh`1h

b1 ´t´t1´11

c 1

t ´ 1 “ c

##### 1

j ` h ðñ (40b) 1

##### 1

j ` h ðñ (40c) j “ t ´ 1 ´ h. (40d)

t ´ 1 “

##### ■

The following lemma presents the core of the proof of Theorem 3.1, exhibiting the construction of a transformer head that can single out and one-hot encode a symbol at a specific position in the input string. The lemma relies on a simple pre-processing of the input string, where the string is prepended (padded) with n ´ 1 beginning of string symbols y1 “ ... “ yn´1 “def BOS, which is common practice in language modeling literature, especially when talking about n-gram LMs. We will denote Σ “def Σ Y tBOSu. This enables a cleaner presentation of the concrete construction of the attention mechanism.

The idea of the proof. Lemma B.4 contains a number of technical definitions of the parameters of a transformer layer (cf. Definition 2.6. Together, they describe a single head of a transformer layer (which will contain H “ n ´ 1 such heads) that is able to extract the one-hot encoding of a particular symbol in the history. The layer of n ´ 1 heads will then be able to extract the n ´ 1 symbols, as required by Lemma B.2. We now describe a single head more formally. Define the following position-augmented symbol representation function of the transformer head h:

˛

¨

y 0b|Σ|

1

b1 ´t 1t

b

1

P R2|Σ|`2n. (41)

rpy,tq “

b1 ´t`t1`11 b .

1

‹ ‚

˚ ˝

b1 ´t`tn`´n11´1

Here, ¨ P t0,1u|Σ| one-hot encodes symbols over Σ. This means that the entire static representations contain multiple components:

- • two |Σ|-dimensional slots for symbol representations and

- • n 2-dimensional slots for head-specific positional representations.

We further define

f pq,kq “def xq,ky, (42)

Qpxq “def Qx, Q P R2ˆp2|Σ|`2nq, (43a) K pxq “def Kx, K P R2ˆp2|Σ|`2nq, (43b) V pxq “def Vx, V P Rp2|Σ|`2nqˆp2|Σ|`2nq, (43c) O pxq “def Ox, O P Rp2|Σ|`2nqˆp2|Σ|`2nq, (43d)

Q:,2|Σ|`1:2|Σ|`2 “def I2 (44a) K:,2|Σ|`2h`1:2|Σ|`2h`2 “def I2, (44b)

V|Σ|`1:2|Σ|,1:|Σ| “def I|Σ| (44c)

O1:|Σ|,1:|Σ| “def ´I|Σ| (44d) We can visualize these matrices as

|Σ| |Σ| 2 pn´1q2 1hkkkkikkkkjst symbol slot 2hkkkkikkkkjnd symbol slot 0hkkkkikkkkjth position slot n ´hkkkkikkkkj1 position slots

Q “

p I2 q 2

(45a)

|Σ| |Σ| 2h 2 pn´1´hq2

1hkkkkikkkkjst symbol slot 2hkkkkikkkkjnd symbol slot h `hkkkkikkkkj1 position slots hhkkkkikkkkjth position slot n ´hkkkkikkkkjh position slots

K “

p I2 q 2

1hkkkkikkkkjst symbol slot 2hkkkkikkkkjnd symbol slot nhkkkkikkkkjposition slots

˜ I|Σ| ¸ 21ndst symbolsymbolslotslot((||ΣΣ||)) n position slots (2n)

V “

(45b)

(45c)

1hkkkkikkkkjst symbol slot 2hkkkkikkkkjnd symbol slot nhkkkkikkkkjposition slots

˜ ´I|Σ| ¸ 1st symbol slot (|Σ|) 2nd symbol slot (|Σ|) n position slots (2n)

O “

(45d)

where 0N is a N-dimensional vector of zeros, IN is the N-dimensional identity matrix and the unspecified elements of Q,K, V, O are 0.

- Lemma B.4. Let Σ be an alphabet and y P Σ˚. For any t “ 1,...,|y|, the hth transformer head (h P rn ´ 1s) defined with the parameters specified in Eq. (41) to Eq. (44d) outputs

¨ ˝

˛ ‚. (46)

0|Σ| yt´1´h 02n

zt´1 “

In particular, this means that the output zt´1 at time step t ´ 1 contains the one-hot encoding of the symbol at position t ´ h ´ 1.13

13For technical reasons—the residual connections—the output is not yt´1´h but larger (with additional zeros), as shown in Eq. (46). This, however, is equivalent for the purposes of Lemma B.2 and later Theorem 3.1.

Proof. Fix t P r|y|s. We compute the representation of yăt computed by the head. First, observe that the matrix defining the query function Q projects onto the first two components of the positional encoding from the representation of yt´1:

¨ ˝

b

˛ ‚. (47)

1

b1 ´t´t1´11

Qprpyt´1qq “ Q rpyt´1q “

Similarly, the matrix defining the key transformation projects onto the hth positional encoding slot, i.e., the dimensions 2|Σ| ` 2h ` 1 and 2|Σ| ` 2h ` 2:

˛ ‚ (48)

¨ ˝

b

1

b1 ´j`jh`1h

K prpyjqq “ K rpyjq “

for j “ ´n ` 2,...,t ´ 1. This results in the scoring function

¨ ˝

˛ ‚,

b

˛ ‚

C¨ ˝

b

G. (49)

1

1

b1 ´j`jh`1h

b1 ´t´t1´11

f pqt´1,kjq “def xqt´1,kjy “

In particular, as shown in Lemma B.3, f is maximized for

j “ t ´ 1 ´ h. (50) This means that

hardmaxpf pqt´1,k1q,f pqt´1,k2q,...,f pqt´1,kt´1qqj “ tj “ t ´ 1 ´ hu. (51) The definition of V further means that

¨ ˝

˛ ‚, (52)

0|Σ| yj 02n

V prpyjqq “ V rpyjq “

giving us, by Eq. (9a),

¨ ˝

˛ ‚. (53)

yt´1 yt´1´h 02n

at´1 “ vt´1´h ` xt´1 “

The definition of O then gives us

zt´1 “ O pat´1q ` at´1 (54a) “ Oat´1 ` at´1 (54b)

˛ ‚

¨ ˝

˛ ‚`

¨ ˝

˛ ‚ (54c)

¨ ˝

´I|Σ|

yt´1 yt´1´h 02n

yt´1 yt´1´h 02n

“

¨ ˝

˛ ‚`

¨ ˝

˛ ‚ (54d)

´ yt´1

yt´1 yt´1´h 02n

“

0|Σ| 02n

¨ ˝

˛ ‚, (54e)

0|Σ| yt´1´h 02n

“

which is what we wanted to prove. ■

Lemmas B.2 and B.4 show that we can define a transformer that one-hot encodes the history of interest ytt´´n1`1. We now show how to define an output matrix E to define a weakly equivalent transformer LM. Concretely, we define E P R|Σ|ˆ|Σ|n´1 with

Ey,yt´1

t´n`1

“def log p

`

yt | ytt´´n1`1

˘

. (55)

Theorem 3.1. For any n-gram LM, there exists a weakly equivalent single-layer hard attention transformer LM with n ´ 1 heads.

Proof. Let T be a transformer LM with n ´ 1 heads defined with the parameters specified in Eq. (41) to Eq. (44d) (h P rn ´ 1s). Let y P tBOSun´1Σ˚ with |y| “ T be a string and XL “ `

˘ “ T pyq. We derive:

xL´Jn`2;...;xLTJ

źT

pT pyt | yătq (56a, Autoregressive LM.)

pT pyq “ pT pEOS | yq

t“1

źT

`

`

˘˘EOS

`

`

˘˘yt (56b, Definition 2.11.)

xLT

xLt´1

“ softmax

E F

softmax

E F

t“1

źT

`

˘yt (56c, Lemma B.2.)

E ytt´´n1`1

“ softmaxpE yT´n`2:T qEOS

softmax

t“1

ř ˘yt

`

E ytt´´n1`1

źT

exp

řexppE yT´n`2:T qEOS

(56d, Definition of softmax.)

“

yPΣ exppE yT´n`2:T qy

yPΣ exppE yT´n`1:t´1 qy

t“1

˘˘ ř

`

`

`

`

˘˘ ř

yt | ytt´´n1`1

źT

EOS | ytT´n`2

exp

log p

exp

log p

˘˘ (56e, Eq. (55).)

“

`

`

˘˘

`

`

y | ytt´´n1`1

y | ytT´n`2

yPΣ exp

log p

yPΣ exp

log p

t“1

źT

`

˘ “ ppyq (56f, The n-gram LM is autoregressive.)

yt | ytt´´n1`1

“ ppEOS | yT´n`2:Tq

p

t“1

■

#### B.2.3 Simulation with n ´ 1 Layers: The Intuition

This section presents the construction of a transformer LM with a single head but n ´ 1 layers that is weakly equivalent to a given n-gram LM. We again outline the intuition first before giving the technical details below as part of Lemma B.5. The construction we present resembles the one from the proof of

- Theorem 3.1. The main difference is that, instead of using n ´ 1 attention heads to identify the history, we instead use n ´ 1 transformer layers. Intuitively, each of the n ´ 1 layers iteratively adds one of the n ´ 1 symbols needed to identify the history, resulting in the identification of the full history after the pn ´ 1qst layer. Once the history is identified, the pn ´ 1qst layer can compute the conditional distribution over the next symbol as in the proof of Theorem 3.1. This can be illustrated as the following sequence of

transformations:14

¨

˛

y1 y2 y3 ¨¨¨ yt ¨¨¨ yT 0|Σ| 0|Σ| 0|Σ| ¨¨¨ 0|Σ| ¨¨¨ 0|Σ| 0|Σ| 0|Σ| 0|Σ| ¨¨¨ 0|Σ| ¨¨¨ 0|Σ|

X1:

(57a)

‹ ‚

˚ ˝

... .

... .

. . .

0|Σ| 0|Σ| 0|Σ| ¨¨¨ 0|Σ| ¨¨¨ 0|Σ|

¨

˛

y1 y2 y3 ¨¨¨ yt ¨¨¨ yT 0|Σ| y1 y2 ¨¨¨ yt´1 ¨¨¨ yT´1 0|Σ| 0|Σ| 0|Σ| ¨¨¨ 0|Σ| ¨¨¨ 0|Σ|

X2:

(57b)

˚ ˝

‹ ‚

... .

... .

. . .

0|Σ| 0|Σ| 0|Σ| ¨¨¨ ¨¨¨ ¨¨¨ 0|Σ|

.

˛

¨

y1 y2 y3 ¨¨¨ yt ¨¨¨ yT 0|Σ| y1 y2 ¨¨¨ yt´1 ¨¨¨ yT´1 0|Σ| 0|Σ| y1 ¨¨¨ yt´2 ¨¨¨ yT´1

Xn´1:

(57c)

‹ ‚

˚ ˝

... .

... .

. . .

0|Σ| 0|Σ| 0|Σ| ¨¨¨ yt´n`1 ¨¨¨ yT´n`2

Such representations can be constructed by starting with the initial symbol encoding (X1) and then passing over the information from the tth symbol to the contextual representations of the pt ` 1qst symbol in X2 by adding a shifted the contextual representation of the tth symbol. This is performed n ´ 1 times, resulting in contextual representations of the form Xn´1 where the tth contextual representation contains the (ordered) information about the preceding n ´ 1 symbols, i.e., the required history. Intuitively, the ℓth transformation of the contextual representation xℓt´1 should be of the form

xℓt “ The previousloomoonxtℓ´1 ℓ ´ 1

` The symbolÓloomoonxℓℓtsymbols´´11 back shifted one “cell” downward.

(58)

symbols.

The first term is simply the residual connection of the transformer layer. The second term is the shifted representation of the symbol yt´ℓ, which can be performed by a simple linear transformation in the value transformation V .

- B.2.4 Simulation with n ´ 1 Layers: Proofs We now make the intuition presented in the previous section more formal. Concretely, we only investigate how to identify the correct n ´ 1 symbols in the history with the n ´ 1 layers. We then rely on Lemma B.2 and the derivation from the proof of Theorem 3.1 again to convert this information into a weakly equivalent transformer LM. Let ℓ P rn ´ 1s. Define the following parameters of the attention head of the ℓth transformer layer:

˛

¨

y 0|Σ| . 0b|Σ|

P Rpn´1q|Σ|`4, (59)

rpy,tq “def

1

b1 ´t 1t

b

1

‹ ‚

˚ ˝

b1 ´t`t1`11

14We leave out the positional encodings for clarity.

f pq,kq “def xq,ky, (60)

Qpxq “def Qx, Q P R2ˆppn´1q|Σ|`4q (61a) K pxq “def Kx, K P R2ˆppn´1q|Σ|`4q (61b) V pxq “def Vx, V P Rppn´1q|Σ|`4qˆppn´1q|Σ|`4q, (61c) O pxq “def Ox, O P Rppn´1q|Σ|`4qˆppn´1q|Σ|`4q, (61d)

Q:,pn´1q|Σ|`1:pn´1q|Σ|`2 “def I2, (62a) K:,pn´1q|Σ|`3:pn´1q|Σ|`4 “def I2, (62b)

V1`ℓ|Σ|:1`pℓ`1q|Σ|,1`pℓ´1q|Σ|:1`ℓ|Σ| “def I|Σ|, (62c) where the unspecified elements of Q,K, and V are 0. Schematically, V looks as follows:

pℓ´1q|Σ| |Σ| pn´1´ℓq|Σ| 4

pℓ ´hkkkkikkkkj1q symbol slots ℓhkkkkikkkkjth symbol slot pn ´ 1hkkkkikkkkj´ ℓq symbol slots 4hkkkkikkkkjposition slots

¨ ˚ ˝

˛ ‹ ‚

ℓ|Σ| I|Σ| |Σ|

V “

pn´1´pℓ`1qq|Σ| 4

(63)

That is, I|Σ| occupies the “cell” ℓ cells down and ℓ ´ 1 right of the top-left corner. Moreover, the matrix O is a matrix of all zeros, meaning that O pxq “ 0 for all x. Again, notice that the position-augmented symbol representation function r can be implemented by concatenating or summing a symbol- and a position-specific component.

- Lemma B.5. With the parameters defined above, it holds that

¨

˛

yt´1

- yt´1´1
- yt´1´2

. yt´1´ℓ 0|Σ| . 0b|Σ|

xℓt´1 “

(64)

1

b1 ´t 1t

b

1

˚ ˝

‹ ‚

b1 ´t`t1`11

Proof. We prove the lemma by induction. As in the proof of Lemma B.4, we pad the input string with n ´ 1 BOS symbols to resolve the case when t ´ 1 ă n ´ 1.

Base Case. For ℓ “ 1, the claim follows from the definition of the symbol representation function r.

Inductive Step. For ℓ ą 1, we assume that the claim holds for ℓ ´ 1. We then prove that it holds for ℓ as well. By the construction of the keys and values matrices, as in the proof of Lemma B.4, it holds that the attention head puts all its attention for query qℓt´1 on the key kℓt´2. This means that

aℓt´1 “ xℓt´´11 ` vtℓ´2. (65)

By the induction hypothesis, we have that

¨

xℓt´´11 “

˚ ˝

˛

- yt´1
- yt´2
- yt´3

. yt´1´pℓ´1q 0|Σ| . 0b|Σ|

1

b1 ´t 1t

b

1

‹ ‚

b1 ´t`t1`11

¨

(66) xℓt´´21 “

˚ ˝

- yt´2
- yt´3
- yt´4

. yt´2´pℓ´1q 0|Σ| . 0b|Σ|

1

b1 ´t 1t

b

1

b1 ´t`t1`11

˛

. (67)

‹ ‚

Furthermore, by definition of V, we have that

vtℓ´2 “ V ´xℓt´´21¯ (68a) “ Vxℓt´´21 (68b)

˛

¨

0|Σ| . 0|Σ| yt´2´pℓ´1q 0|Σ| . 0b|Σ|

(68c)

“

1

b1 ´t 1t

b

1

‹ ‚

˚ ˝

b1 ´t`t1`11

Inserting this into Eq. (65), we get that aℓt´1 satisfies the required equality:

aℓt´1 “ xℓt´´11 ` vtℓ´2 (69a)

˛

˛

˛

¨

¨

¨

¨

˛

0|Σ| . 0|Σ| yt´2´pℓ´1q 0|Σ| . 0b|Σ|

yt´1 .

yt´1 . yt´1´pℓ´1q yt´1´ℓ 0|Σ| . 0b|Σ|

yt´1 . yt´1´pℓ´1q 0|Σ| . 0b|Σ|

- yt´1´pℓ´1q
- yt´2´pℓ´1q 0|Σ|

. 0b|Σ|

(69b)

“

`

“

“

1

1

1

1

b1 ´t 1t

b1 ´t 1t

b1 ´t 1t

b1 ´t 1t

b

b

b

b

1

˚ ˝

‹ ‚

b1 ´t`t1`11

1

1

1

‹ ‚

‹ ‚

‹ ‚

˚ ˝

˚ ˝

˚ ˝

b1 ´t`t1`11

b1 ´t`t1`11

b1 ´t`t1`11

˘ ` aℓt´1 “ aℓt´1 with the definition of O as the zero function gives us zℓt´1 “ aℓt´1, we get the required equality, which finishes the proof. ■

`

Since the computation of xℓt´1 “def zℓt´1 “ O

aℓt´1

This allows us to prove the following theorem.

- Theorem 3.2. For any n-gram LM, there exists a weakly equivalent pn ´ 1q-layer hard attention transformer LM with a single head.

Proof. Lemma B.5 shows that the n ´ 1-layer transformer can identify the history of interest. Applying Lemma B.2 and the same derivation as in the proof of Theorem 3.1 shows that we can construct a weakly equivalent hard attention transformer. ■

#### B.2.5 Simulation with a Single Layer and a Single Head: Intuition

While the construction presented here is considerably less intuitive than that of Theorem 3.1, the steps of the proof remain the same—they include the identification of the individual symbols and their positions in the history, combining them into the one-hot encoding of the entire history, and using that to compute the correct next-symbol conditional distributions. This proof focuses on encoding the entire history of interest ytt´´n1`1 into a single vector in a way that can be decoded to index the conditional probability distribution as in Definition 2.11. This can then be used to index the appropriate conditional probability distributions as in the proof of Theorem 3.1.

Again, we outline the intuition first. Fix t ď |y|. We decompose the computation of the representation encpyătq constructed by a single-layer-single-head transformer network as follows.

1. Attending exactly to the history of interest with a single head and a single layer. This can be done by

assigning the same attention score with the scoring function to all positions within the history ytt´n`1 and a lower score to all other positions. Keeping the definitions of Q and V from Theorem 3.1, we can achieve that by defining

f pqt´1,kjq “ ´ReLUpxqt´1,kjyq (70)

which assigns positions in the history score 0 and others negative scores. This is illustrated in Fig. 3. Concretely, the scoring function f together with hard attention results in attention weights of the form

1

n ´ 1 tj ě t ´ n ` 1u. (71)

sj “

f pqt´1, kjq 0

‚ ‚ ‚

‚

´

- ´1
- ´2
- ´3

‚

´

‚

´

.

´ ‚

´t ` n ´ 1

y1 ¨ ¨ ¨ yt´6 yt´5 yt´4 yt´3 yt´2 yt´1 yt

j

Figure 3: An illustration of the attention scores in a single-layer-single-head transformer network for a history yt´3:t´1 “ yt´3yt´2yt´1 when determining the conditional distribution ppyt | yătq.

2. Storing the order of the symbols in the history. We will define the position-augmented representations as position-scaled one-hot encodings of the input symbols. In particular, define

¨ ˝

˛ ‚P R|Σ|`2 (72)

10´j ¨ yt´1

rpyt´1,jq “def

t 1

This effectively stores the information about both the position of the current symbol (with the magnitude) as well as the identity of the symbol yj (with the index of the non-zero entry). Unlike in the multi-head or multi-layer case, note that in this case, the function r is not a concatenation (or addition) of two separate components (one for symbols and one for positions).

Ignoring residual connections, Eq. (71) then implies that

tÿ´1

1 n ´ 1

##### 10´j yj . (73)

at´1 “

j“t´n`1

For simplicity, we now write a “def at´1. The individual entries of a will correspond to symbols in Σ. The digits of these symbols will encode the positions of the symbols in the history. Concretely, a will have a non-zero digit in the ith position if and only if the symbol y appears in the string yăt at position i for i P rt ´ 1s. For example, in a 5-gram LM over the alphabet Σ “ ta,bu, the contextual representation a for the history yt´4:t´1 “ abaa will be

ˆ

˙ “

10´t ˆ

˙. (74)

10t´1 ` 10t´3 ` 10t´4 10t´2

1 4

1 4

0.1011 0.0100

a “

Such representations therefore uniquely encode the history of interest.

Decoding the representations of the history. The representations a, therefore, contain the information about the history of interest compactly represented in a |Σ|-dimensional vector a. We now want to construct a function that transforms the constructed vector a into a one-hot encoding of the history. To make a invariant with respect to t, we first scale it by 10n´n´1t and define

n ´ 1 10n´t

a1 “

a. (75)

y

1

x

10´n

Figure 4: The step function txu and the approximated step function that matches txu outside of r0,10´ns.

Then

nÿ´1

ty “ yi`t´nu10´i. (76)

a1y “

i“1

In words, the |Σ| entries of a1 are of the form a1y “ 0.d1 ...dn´1, where di “ 1 if and only if y appears in the history yăt at position t ´ n ` i.

We now focus on a specific symbol y P Σ with the entry a1y in the vector a1 and decode it into a vector

that can be used to construct a one-hot encoding of the relevant history ytt´´n1`1 with F. The “decoding function” will take the form of a n ´ 1-layer ReLU-activated MLP. Intuitively, each of the n ´ 1 layers

will contain |Σ| neurons, where each of them will compute the values di for a particular y P Σ. Now, d1 can be computed as

d1 “ ␣

(

101 ¨ ay ´ 1 ` 10´n ą 0

. (77) Since t¨u is not a continuous function, and, unlike in Appendix B.1, the arguments can now take arbitrary real values, it cannot be implemented using a composition of ReLU functions. We can, however, simulate the discontinuous function with a linear combination of two ReLU functions that together define the same function as t¨u on a subset of R relevant for our purposes. Notice that, as long as d1 “ 1, we have that 101 ¨ ay ´ 1 ` 10´n ě 10´n while we have 101 ¨ ay ´ 1 ` 10´n`1 “ 101 ¨ 0 ´ 1 ` 10´n ă 0 if d1 “ 0. This means that our approximation of t¨u only has to map values greater or equal to 10´n to 1, rather than all positive values. This allows us to continuously transition from 0 to 1 as the input to the ReLU function increases from 0 to 10´n. Such a piecewise linear approximation of t¨u can be easily implemented by a linear combination of ReLU functions, i.e., with an MLP. See Fig. 4 for an illustration of the approximation.

- d2 can then be computed as

d2 “ ␣

(

102 ¨ ay ´ 101d1 ´ 1 ` 10´n ą 0

(78) and, in general,

#10i ¨ ay ´

10i´jdj ´ 1 ` 10´n ą 0+. (79)

- iÿ´1
- j“1

di “

The computation of di requires i layers (since dj for j ă i have to be computed first), meaning that n ´ 1 layers are required in total. Altogether, these layers compute the values di for a single y P Σ. Replicating this computation for every y P Σ and concatenating the results gives us the desired contextual representation z.

With this construction, it holds for every y P Σ that di “ 1 if and only if the symbol y appears in the history ytt´´n1`1 at position t ´ n ` i. This, therefore, gives us enough information to reconstruct the multi-hot encoding of the history of interest. As in Theorem 3.1, this can then be converted into a one-hot encoding using another ReLU layer to implement the logical AND operation. This intuition is formalized in the following section.

- B.2.6 Simulation with a Single Layer and a Single Head: Proofs We define the following parameters of the transformer head.15

15For simplicity, we ignore residual connections in this section since we do not require them and the omission makes the presentation cleaner. By duplicating the representations as in Theorem 3.1, residual connections could easily be added back to

- • Static encodings

rpyt´1,jq “def

¨ ˝

10´j ¨ yt´1

1 t

˛ ‚P R|Σ|`3 (80)

- • Query, key, value, and output transformations

Qpxq “def Qx ` bQ, Q P R2ˆp|Σ|`2q,bQ P R2, (81a) K pxq “def Kx, K P R2ˆp|Σ|`2q, (81b) V pxq “def Vx, V P R|Σ|ˆp|Σ|`2q, (81c) O pxq “def I|Σ|x, (81d)

˙, (82a)

Q:,|Σ|`1:|Σ|`2 “def ˆ

0 1 ´1 0

bQ “def ˆ´pn ´ 2q

˙ (82b)

0

K:,|Σ|`1:|Σ|`2 “def I2 (82c)

V:,1:|Σ| “def I|Σ| (82d) where the unspecified elements of Q,K, V, O are 0.

A large part of the proof of correctness will rely on identifying the digits of the contextual representations of strings. We will rely heavily on the following definition. Definition B.2. Let x P Q X `

‰

10´pN`1q,10´1

be a rational-valued number with at most N digits in its decimal representation. We define di pxq as the ith digit of x for i P rNs. We also group these N values into the vector dpxq:

¨ ˚ ˝

˛ ‹ ‚. (83)

d1 pxq . dN pxq

dpxq “def

- Lemma B.6. A transformer with the parameters and functions defined in Eq. (80)–Eq. (82d) computes for string y P Σ˚

tÿ´1

1 n ´ 1

##### 10´j yj . (84)

at´1 “

j“t´n`1

Proof. By construction, we have

qt´1 “ ˆ

˙ (85a)

t ´ 1 ´ pn ´ 2q ´1

kj “ ˆ

˙ (85b) vj “ 10´j ¨ yt´1 , (85c)

1 j

make this setting closer to the general transformer setting.

thus

f pqt´1,kjq “ ´ReLUpxqt´1,kjyq (86a)

˙F˙ (86b)

˙,ˆ

“ ´ReLUˆBˆ

t ´ 1 ´ pn ´ 2q ´1

1 j

“ ´ReLUpt ´ 1 ´ n ` 2 ´ jq (86c) “ ´ReLUpt ´ n ` 1 ´ jq (86d)

#0 if j ě t ´ n ` 1 ă 0 otherwise

(86e)

“

meaning that

1

n ´ 1 tj ě t ´ n ` 1u. (87) This means that

sj “

tÿ´1

tÿ´1

1 n ´ 1

1 n ´ 1

10´j yj (88)

at´1 “

vj “

j“t´n`1

j“t´n`1

which is what we needed to prove. ■

#### Lemma B.7. Define

n ´ 1 10n´t

a1 “def

at´1 (89) with at´1 from Lemma B.6. Indexing the |Σ| elements of a1 directly with y P Σ, it holds that

˘ “ 1 ðñ yt´n`i “ y (90) for all i P rNs and di pxq “ 0 for all i ą N.

`

a1y

di

- Proof. By Lemma B.6, a contains entries of the form

tÿ´1

1 n ´ 1

ty “ yju10´j (91a)

ay “

j“t´n`1

t´1´pÿt´nq

␣

(

1 n ´ 1

10´pj1`t´nq (91b, Change of variables.)

“

y “ yj1`t´n

j1“1

nÿ´1

1 n ´ 1

ty “ yj`t´nu10n´t´j (91c, Change of variables.)

“

j“1

nÿ´1

10n´t n ´ 1

ty “ yj`t´nu10´j (91d, Distributivity.)

“

j“1

for y P Σ. Then

n ´ 1 10n´t

a1y “def

ay (92a)

nÿ´1

10n´t n ´ 1

n ´ 1 10n´t

ty “ yj`t´nu10´j (92b)

“

j“1

nÿ´1

ty “ yj`t´nu10´j. (92c)

“

j“1

˘ “ 1 ðñ yt´n`i “ y for i P rNs and that di pxq “ 0 for i ą N, which is what we wanted to prove. ■

`

a1y

This implies that di

#### Lemma B.8. Let x P Q X `

‰

10´pN`1q,10´1

with di pxq P t0,1u for i P rNs. Then, di pxq satisfy the equality

10i´j dj pxq ´ 1 ` 10´pN`1q ą 0+. (93)

#10i ¨ x ´

- iÿ´1
- j“1

di pxq “

for all i P rNs.

Proof. Let x P Q X `

‰

10´pN`1q,10´1

with di pxq P t0,1u for i P rNs. Then, by definition of di pxq, we have that

ÿN

dj pxq10´j. (94)

x “

j“1

This means that

implying that

ÿN

dj pxq10´j (95a)

10ix “ 10i

j“1

ÿN

dj pxq10i´j (95b)

“

j“1

- iÿ´1
- j“1

ÿN

dj pxq10i´j ` di pxq `

dj pxq10i´j, (95c)

“

j“i`1

10i ¨ x ´

- iÿ´1
- j“1

10i´j dj pxq “ di pxq `

ÿN

##### dj pxq10i´j. (96)

j“i`1

Suppose now that di pxq “ 1. Then

10i ¨ x ´

meaning that

- iÿ´1
- j“1

ÿN

10i´j dj pxq ´ 1 ` 10´pN`1q “ di pxq `

dj pxq10i´j ´ 1 ` 10´pN`1q (97a)

j“i`1

ÿN

dj pxq10i´j ´ 1 ` 10´pN`1q (97b)

“ 1 `

j“i`1

ÿN

dj pxq10i´j ` 10´pN`1q ą 0, (97c)

“

j“i`1

#10i ¨ x ´

10i´j dj pxq ´ 1 ` 10´pN`1q+

- iÿ´1
- j“1

“ 1 “ di pxq. (98)

Suppose, on the contrary, that di pxq “ 0. Then

- iÿ´1
- j“1

ÿN

dj pxq10i´j ´ 1 ` 10´pN`1q (99a)

10i´j dj pxq ´ 1 ` 10´pN`1q “ di pxq `

10i ¨ x ´

j“i`1

ÿN

dj pxq10i´j ´ 1 ` 10´pN`1q (99b)

“ 0 `

j“i`1

ÿN

dj pxq10i´j ` 10´pN`1q ´ 1 (99c)

“

j“i`1

N´pÿi`1q

dj1`i`1 pxq10i´j1´i´1 ` 10´pN`1q ´ 1 (99d)

“

j1“1

N´pÿi`1q

dj1`i`1 pxq10´j1´1 ` 10´pN`1q loooooooooooooooooooooooomoooooooooooooooooooooooon

´1 ă 0 (99e)

“

j1“1

ă1

meaning that

#10i ¨ x ´

10i´j dj pxq ´ 1 ` 10´pN`1q+

- iÿ´1
- j“1

“ 0 “ di pxq. (100)

This finishes the proof. ■

#### Lemma B.9. Let x P Q X `

‰

10´pN`1q,10´1

with di pxq P t0,1u for i P rNs. Given dj pxq for j ă i,

- di pxq can be computed by a single layer MLP. Proof. By Lemma B.8, we can use the knowledge of dj pxq for j ă i, di pxq to implement the function

10i´j dj pxq ´ 1 ` 10´pN`1q ą 0+. (101)

#10i ¨ x ´

- iÿ´1
- j“1

di pxq “

with an MLP. For any i P rNs, the inner function ¨

˛

- d1 pxq

.

- di´1 pxq x

- iÿ´1
- j“1

10i´j dj pxq ´ 1 ` 10´pN`1q (102a)

ÞÑ 10i ¨ x ´

˚ ˝

‹ ‚

¨

˛

¨

˛

´10i´1 . ´101 10i

- d1 pxq

.

- di´1 pxq x

G

C

loooooooomoooooooon´1 ` 10b´pN`1q

(102b)

“

,

˚ ˝

‹ ‚

˚ ˝

‹ ‚

looooomooooonWJ

is an affine transformation. The indicator function in Eq. (101), however, is discontinuous and can thus not be implemented by a composition of continuous ReLU MLPs. Here, we take advantage of the fact that

10i ¨ x ´

10i´j dj pxq ´ 1 ` 10´pN`1q P p´8,0s Y ”10´pN`1q,8¯ looooooooooooooomooooooooooooooon

- iÿ´1
- j“1

. (103)

I

The MLP

##### MLPI pzq “ 10N`1 ´ReLUpzq ´ ReLU´z ´ 10´pN`1q¯¯ (104a)

˙z ` ˆ

ReLUˆˆ

˙˙ (104b)

“ `

˘

0 ´10´pN`1q

1 1

10N`1 ´10N`1

matches t¨ ą 0u on I, as we show next. See Fig. 4 for an illustration of the approximation. First, assume that z ď 0. Then

##### MLPI pzq “ 10N`1 ´ReLUpzq ´ ReLU´z ´ 10´pN`1q¯¯ “ 10N`1 p0 ´ 0q “ 0 “ t0 ą 0u.

(105) On the contrary, assuming z ě 10´pN`1q, we have that

MLPI pzq “ 10N`1 ´ReLUpzq ´ ReLU´z ´ 10´pN`1q¯¯ (106a) “ 10N`1 ´z ´ ´z ´ 10´pN`1q¯¯ (106b) “ 10N`1 ´z ´ z ` 10´pN`1q¯ (106c) “ 10N`1 ´10´pN`1q¯ (106d) “ 1 “ tz ą 0u. (106e)

We can therefore construct the MLP MLP computing Eq. (101) on I by a composition of Eq. (102b) (computing z in Eq. (104b)) and the MLP MLPI from Eq. (104b). ■

- Lemma B.10. Let N P N. There exists an MLP MLP such that MLPpxq “ dpxq (107)

‰

for all x P Q X `

10´pN`1q,10´1

with di pxq P t0,1u for i P rNs. Proof. At a very high level, we will construct an N-layer MLP performing the transformations

¨

˛

¨

˛

¨

˛

¨

˛

d1 pxq x . x

d1 pxq d2 pxq

d1 pxq d2 pxq . dN pxq

x x

“ dpxq. (108)

x ÞÑ

ÞÑ

ÞÑ

ÞÑ ¨¨¨ ÞÑ

˚ ˝

‹ ‚

˚ ˝

‹ ‚

˚ ˝

‹ ‚

˚ ˝

‹ ‚

. x

. x

- By Lemma B.9, all individual transformations can be performed exactly by a single-layer MLP. The composition of the N layers results in the vector dpxq.

¨

˛

x x

is a simple linear transformation. We now construct the ℓth layer fℓ of

The transformation x ÞÑ

˚ ˝

‹ ‚

. x

¨

˛

d1 pxq d2 pxq . dℓ´1 pxq x . x

the MLP with parameters Wℓ and bℓ (cf. Definition B.1), assuming that it has the input

. The

˚ ˝

‹ ‚

layer fℓ has to satisfy

¨

¨

˛

˛

¨

˛

d1 pxq d2 pxq . dℓ´1 pxq x x

d1 pxq d2 pxq . dℓ´1 pxq dℓ pxq x . x

, (109)

“

f

˚ ˝

˚ ˝

‹ ‚

‹ ‚

˚ ˝

‹ ‚

. x

i.e., it must copy all n ´ 1 entries apart from the ℓth one. Thus, Wk,k1 “ tk “ k1u for k ‰ ℓ and bk “ 0 for all k ‰ ℓ (here, we write W and b for Wℓ and bℓ to avoid clutter). To define the remaining ℓth row, we use Lemma B.9. It tells us that defining

Wℓ,: “def ˆ

˙ (110a)

10ℓ´1 ¨¨¨ 101 10ℓ 0 ¨¨¨ 0 10ℓ´1 ¨¨¨ 101 10ℓ 0 ¨¨¨ 0

bℓ “def ˆ ´1 ` 10´n

˙ “ ˆ´1 ` 10´n ´1

˙ (110b)

´1 ` 10´n ´ 10´n

will result in the ℓth row of fℓ computing exactly dℓ pxq after being multiplied by the matrix W1 “def `

˘

10n 10n

. (111)

This is what Eq. (109) requires. The parameters Wℓ,: and bℓ represent the parameters of the affine transformation in Lemma B.9. Note that the matrix W1 is not part of the original definition of the MLP.

However, it can easily be absorbed into the matrix Wℓ`1 in the actual implementation (at the cost of duplicating the size of the hidden state). We keep it here to make the presentation more intuitive. Since

any layer fℓ can be defined like this, and the final MLP is their composition, this finishes the proof. ■

- Lemma B.11. Given at´1 from Lemma B.6, it holds that

10n´1´t n ´ 1

1 ´ 10n1´1 1 ´ 101

. (112)

∥at´1∥1 “

- Proof. By Lemma B.7, we have that

We compute:

tÿ´1

a “def at´1 “

j“t´n`1

1 n ´ 1

10´j yj (113)

tÿ´1

1 n ´ 1

10´j (114a)

∥a∥1 “

j“t´n`1

t´1´pÿt´n`1q

1 n ´ 1

10´pj1`t´n`1q (114b)

“

j1“0

nÿ´2

1 n ´ 1

1 10j

10´pt´n`1q

(114c)

“

j“0

10´pt´n`1q n ´ 1

1 ´ 10n1´1 1 ´ 101

(114d)

“

10n´1´t n ´ 1

1 ´ 10n1´1 1 ´ 101

. (114e)

“

##### ■

Corollary B.1. Given a “def at´1 from Lemma B.6 and a1 “def a1t´1 from Lemma B.7, it holds that

1 ´ 10n1´1 1 ´ 101

1 ∥a∥1

¨ a “ a1. (115)

¨ 10 ¨

Proof. By Lemma B.11, we can write

10n´1´t n ´ 1

1 ´ 10n1´1 1 ´ 101

∥a∥1 “

“

and

n ´ 1 10n´t ¨

1 ∥a∥1

##### 1 Z

“

1 10n´1

where Z “def 101 ¨ 1´

1´101 is a constant independent of t. Then

10n´t n ´ 1 ¨ Z (116)

(117)

##### 1 ∥a∥1

1 ´ 10n1´1 1 ´ 101

1 10 ¨

¨

1 ´ 10n1´1

n ´ 1 10n´t ¨

1 Z ¨

1 10 ¨

¨a (118a)

¨ a “

looooooomooooooon1Z´ 101

n ´ 1 10n´t ¨

1 Z ¨ Z ¨ a (118b)

“

n ´ 1 10n´t ¨ a (118c)

“

“ a1 (118d) ■

- Lemma B.12. Let Σ be an alphabet. Given the transformer parameters and functions defined in

Eq. (80)–Eq. (82d), there exists an MLP F (whose inputs are ∥¨∥1-normalized) that maps the contextual representations a of yăt into a one-hot encoding of ytt´´n1`1 for any string y P Σ˚ and any t P r|y|s.

Proof. By Lemma B.7, we have that

tÿ´1

1 n ´ 1

10´j yj (119)

a “

j“t´n`1

˘ “ 1 ðñ yt´n`i “ y for a1 “def 10n´n´1t a (where a1i “ 0 for i ą n ´ 1, from the same lemma). We can express the entries of the one-hot encoding of the history ytt´´n1`1, ytt´´n1`1 , as

`

a1y

and that di

t´n`1...yt´1 “ 1 ðñ d1 ´a1yt´n`1¯ ^ ¨¨¨ ^ dn´1 ´a1yt´1¯. (120)

ytt´´n1`1 y

- By Lemma B.10, the vectors d´a1yt´n`1¯,...,d´a1yt´1¯ can be computed by an n ´ 1-layer MLP. Each of these vectors is of size n ´ 1 and, among others, contains the values d1 ´a1yt´n`1¯,...,dn´1 ´a1yt´1¯.

Since the entries ytt´´n1`1 y

t´n`1...yt´1 can be expressed as the results of the logical AND operation, their computation can be performed by a single-layer ReLU MLP as per Lemma B.1. The MLP F can therefore be constructed as a composition of three functions:

- 1. The scaling a ÞÑ ∥a1∥

1

¨ 101 ¨ 1´

1 10n´1

1´101 ¨ a, which results in a1 by Corollary B.1.

- 2. The concatenation of the |Σ| n ´ 1-layer MLPs computing d

`

˘

a1y

for all y P Σ. This results in pn ´ 1q|Σ| binary values altogether.

`

˘

a1y

. This finishes the proof. ■

- 3. The MLP performing the AND operation between the entries of d

- Theorem 3.3. For any n-gram LM, there exists a weakly equivalent single-layer hard attention transformer LM with a single head.

Proof. To show that there exists a weakly equivalent single-layer-single-head transformer LM to any n-gram LM, we combine the lemmata in this section. Let T be a transformer LM defined with the

parametersřt´1 and functions defined in Eq. (80)–Eq. (82d). By Lemma B.6, the representations at´1 “

n´110´j yj computed by T contain information about the symbols and their positions in the history ytt´´n1`1. By Lemma B.12 then, a can be mapped to the one-hot encoding of the history with a n ´ 1-layer MLP F. This one-hot encoding can then be used to index (the logits of) the probabilities stored in the output matrix E defining a weakly equivalent transformer. ■

1

j“t´n`1

### C Sparse Attention

We now prove a lemma analogous to Lemma B.4. It shows that a sparse attention transformer head can isolate a particular symbol in the string. First, define the following position-augmented symbol representation function of the transformer head h:

˛ ‹ ‚P t0,1u2|Σ|`2 (121)

¨ ˚ ˝

y

- 0|Σ|

- 1 t

rpy,tq “def

and the scoring function

f pq,kq “def ´ˇqJkˇ. (122)

Here, the position-augmented symbol representation function r can again be implemented by concatenating or summing a symbol- and a position-specific component. Lastly, we define the transformation matrices

Qpxq “def Qx ` bQ, Q P R2ˆp2|Σ|`2q,bQ P R2, (123a) K pxq “def Kx, K P R2ˆp2|Σ|`2q, (123b) V pxq “def Vx, V P Rp2|Σ|`2qˆp2|Σ|`2q, (123c) O pxq “def Ox, O P Rp2|Σ|`2qˆp2|Σ|`2q, (123d)

Q:,2|Σ|`1:2|Σ|`2 “def I2 (124a) bQ “def ˆ

˙ (124b)

0 ´h

K:,2|Σ|`1:2|Σ|`2 “def ˆ

˙, (124c)

0 1 ´1 0

V|Σ|`1:2|Σ|,1:|Σ| “def I|Σ| (124d)

O1:|Σ|,1:|Σ| “def ´I|Σ| (124e) where the unspecified elements of Q,K, V, O are 0.

- Lemma C.1. Let Σ be an alphabet and y P Σ˚. For any t P r|y|s, a transformer head defined with the parameters above outputs

¨ ˝

˛ ‚. (125)

0|Σ| yt´1´h 02

zt´1 “

In particular, this means that the output zt´1 at time step t ´ 1 contains the one-hot encoding of the symbol at position t ´ h ´ 1.

Proof. The construction is largely identical to the one in Theorem 3.1, with one crucial difference: It relies on simpler, but unbounded positional encodings and a less-standard, but still easily implementable attention scoring function in the form of an MLP.

It is easy to see that the query and value transformations result in:

Thus, we get that

˙ (126a)

qt´1 “ Qprpyt,tqq “ ˆ

1 t ´ 1 ´ h

˙. (126b)

kj “ K prpyj,jqq “ ˆ´j

1

f pqt´1,kjq “ ´ˇqJt´1kjˇ (127a)

˙

˙J ˆ´j

ˆ

1 t ´ 1 ´ h

(127b)

“ ´

1

ˇ

ˇ

“ ´|j ´ pt ´ h ´ 1q|. (127c)

˘ ě f pqt,kjq`1 for any j ‰ j˚. This is a crucial property of the scoring function and one that allows sparsemax to uniquely attend to j˚; by Lemma C.2, it holds that

`

f clearly has a unique maximum at j˚ “ t´1´h. Moreover, by construction, f

qt,kj˚

sparsemaxpf pqt´1,k1q,...,f pqt´1,kt´1qq “ hardmaxpf pqt´1,k1q,...,f pqt´1,kt´1qq, (128) meaning that

sparsemaxpf pqt´1,k1q,...,f pqt´1,kt´1qqj “ tj “ t ´ 1 ´ hu (129)

as in the proof of Lemma B.4. This is exactly the same as Eq. (51). Since O and V result in the same vectors as in Lemma B.4, the remainder of the proof is the same as in Lemma B.4. ■

#### Lemma C.2. Let x P RD. If maxD

xd ě maxD

xd ` 1, then sparsemaxpxq “ hardmaxpxq.

d“1

d“1 dRargmax pxq

Proof. Let x P RD and let xp1q ě xp2q ě ... ě xpDq be the non-increasing entries of x. Due to the additive invariance of the softmax (Martins and Astudillo (2016, Proposition 2)), we can assume that

xp1q “ 0 and xp2q ď ´1. By Martins and Astudillo (2016, Proposition 1),

sparsemaxpxqd “ maxp0,xd ´ τ pxqq, (130) where

řkpxq

j“1 xpjq ´ 1 k pxq

τ pxq “def

(131)

and

k pxq “def max˜k P rDs | 1 ` kxpkq ą

xpjq¸. (132)

ÿk

j“1

It suffices to show that k pxq “ 1. For k “ 1, we get

1 ` 1 ¨ xp1q “ 1 ` 1 ¨ 0 “ 1 “ 1 ě 0 “ xp1q. (133)

For k “ 2, we get

1 ` 2 ¨ xp2q “ 1 ` xp2q ` xp2q (134a) ď 1 ´ 1 ` xp2q (134b) “ xp2q (134c) “ xp1q ` xp2q (134d)

ÿ2

xpjq, (134e)

“

j“1

meaning that the condition from Eq. (132) is not fulfilled for k “ 2. This implies that k pxq “ 1 and τ pxq “ xp1q ´ 1. Thus, we get that

˘ “ 1 (135) and

`

sparsemaxpxqp1q “ max

0,xp1q ´ xp1q ` 1

¨ ˚ ˝0,loomoonxp1q

˛ ‹ ‚“ 0 (136)

`

˘ “ max

sparsemaxpxqpdq “ max

0,xpdq ´ xp1q ` 1

´0 ` 1

ď´1

for d ą 1. ■ This allows us to prove the main theorem for sparse-attention transformer LMs.

- Theorem 4.1. For any n-gram LM, there exists a weakly equivalent single-layer sparse attention transformer LM with n ´ 1 heads.

Proof. Lemma C.1 shows how individual heads of the transformer can identify the symbols in the position of interest. n ´ 1 of them can identify the entire history. The proof then follows the same reasoning as that of Theorem 3.1. ■

Adapting the same proof strategy to Theorem 3.2 would naturally result in an analogous result for n ´ 1 layers and a single head.

Notice that Lemma C.1 requires different and less standard positional encodings, which are, crucially, unbounded. Constructing a sparse attention transformer with bounded positional encodings seems more difficult; the contextual representations would in that case either converge or be non-unique with t Ñ 8 and since the sparsemax always contracts (Martins and Astudillo, 2016, Proposition 2), attending to individual positions would be difficult. While the positional encodings and the scoring function used in the proof of Lemma C.1 are somewhat less standard than those used in Lemma B.4, similar positional encodings and the same scoring function have been used in theoretical analyses before and even in practical implementations (Pérez et al., 2021).

