arXiv:2408.13233v2[cs.LG]15Oct2024

Multi-Layer Transformers Gradient Can be Approximated in Almost Linear Time

Yingyu Liang∗ Zhizhou Sha† Zhenmei Shi‡ Zhao Song§ Yufa Zhou¶

Abstract

The computational complexity of the self-attention mechanism in popular transformer architectures poses signiﬁcant challenges for training and inference, and becomes the bottleneck for long inputs. Is it possible to signiﬁcantly reduce the quadratic time complexity of computing the gradients in multi-layer transformer models? This paper proves that a novel fast approximation method can calculate the gradients in almost linear time n1+o(1) where n is the input sequence length, while it maintains a polynomially small approximation error 1/ poly(n) across the entire model. Our theory holds for general loss functions and when the multi-layer transformer model contains many practical sub-modules, such as residual connection, casual mask, and multi-head attention. By improving the eﬃciency of gradient computation, we hope that this work will facilitate more eﬀective training and deployment of long-context language models based on our theoretical results.

[Figure 1]

∗ yingyul@hku.hk. The University of Hong Kong. yliang@cs.wisc.edu. University of Wisconsin-Madison. † shazz20@mails.tsinghua.edu.cn. Tsinghua University. ‡ zhmeishi@cs.wisc.edu. University of Wisconsin-Madison. § magic.linuxkde@gmail.com. The Simons Institute for the Theory of Computing at the University of California,

Berkeley. ¶ yufazhou@seas.upenn.edu. University of Pennsylvania.

## Contents

- 1 Introduction 3

- 1.1 Key background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 1.2 Our contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 2 Related Work 5
- 3 Preliminary 6

- 3.1 Loss function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Closed forms of gradient components . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 4 Main Results 7

- 4.1 Fast computing for single layer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.2 Fast computing for multi-layer transformers . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.3 Beyond the previous work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 5 Technical Overview 10

- 5.1 Low-rank approximation for attention matrix . . . . . . . . . . . . . . . . . . . . . . 10
- 5.2 Accelerating gradient computation of Ti(X) . . . . . . . . . . . . . . . . . . . . . . . 10
- 5.3 Accelerating gradient computation of Wi and WVi . . . . . . . . . . . . . . . . . . . 11
- 5.4 Accelerating gradient computation for multi-Layer transformers . . . . . . . . . . . . 12

- 6 Extensions 13
- 7 Conclusion 13

- A More Related Work 24
- B Discussion and Extension Details 25

- B.1 Multi-head attention . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B.2 Residual connection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.3 Causal attention mask . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.4 System-level attention acceleration . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.5 Prompt tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- C Preliminary on Gradient Calculation 27

- C.1 Basic math facts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- C.2 Close form of three gradient components . . . . . . . . . . . . . . . . . . . . . . . . . 28
- C.3 Basic notations for computing gradients . . . . . . . . . . . . . . . . . . . . . . . . . 30
- C.4 Low rank representations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- C.5 Bounded entries of matrices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

- D Matrix View 34

- D.1 Gradient of s(X) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- D.2 Gradient on Ti(X) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- D.3 Matrix view of C(X) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.4 Matrix view of gradient on Ti(X) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- D.5 Matrix view of each term in gradient on Ti(X) . . . . . . . . . . . . . . . . . . . . . 41
- D.6 Components of gradient on Ti(X) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

#### E Fast Computation for Gradient on T(X) 48

- E.1 Fast computation for B6(X) term . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- E.2 Fast computation for B7(X) term . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- E.3 Fast computation for B8(X) term . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
- E.4 Fast computation for B2(X) term . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
- E.5 Fast computation for B4(X) term . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
- E.6 Putting everything together . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58

#### F Fast Computation for Gradient on W 59

- F.1 Key concepts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- F.2 Gradient of s(X) on W . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- F.3 Gradient of L(X) on W . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- F.4 Fast computation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62

#### G Fast Computation for Gradient on WV 63

- G.1 Gradient of s(X) on WV . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
- G.2 Gradient of L(X) on WV . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
- G.3 Fast computation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66

#### H Gradient Approximation for Entire Model 68

- H.1 Computation time for Gi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
- H.2 Fast computation for single-layer transformer . . . . . . . . . . . . . . . . . . . . . . 71
- H.3 Fast computation for multi-layer transformer . . . . . . . . . . . . . . . . . . . . . . 72

#### I Causal Attention Mask 75

- I.1 Tools from previous work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75
- I.2 Fast computation with causal mask . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76

#### J Residual Connection 80

- J.1 Key concepts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80
- J.2 Analysis of the residual connection . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
- J.3 Analysis for the entire model with the residual connection . . . . . . . . . . . . . . . 82

#### K Multi-head Attention 84

## 1 Introduction

Large Language Models (LLMs), such as ChatGPT [SZK+22], GPT-4 [AAA+23], Claude 3.5 [Ant24], Llama 3.1 [LT24], and others, have demonstrated immense potential to enhance various aspects of our daily lives, e.g., conversation AI [LCT+24], AI agent [XCG+23, CYL+24], search AI [Ope24], AI assistant [MWY+23, ZKAW23] and many so on. One of the most emergent abilities of LLMs is dealing with long-context information, a format that is crucial for recording material like academic papers, oﬃcial reports, legal documents, and so on. LLMs have proven adept at tackling long-context tasks, including Retrieval Augmented Generation (RAG) [LPP+20, GXG+23], zero-shot summarization [LSH+23, ZLD+24], and maintaining very long-term conversations [XSW21, XGW+22], and so on. This proﬁciency has necessitated the development of long-context modeling capabilities within LLMs.

The self-attention mechanism is crucial for the success of LLMs, since LLMs are mainly based on Transformer architecture whose key module is attention. In attention computation, we will compute the attention score between each pair of tokens, which is the complexity bottleneck during long context training and inference. In detail, we need to spend O(n2d) running time for each self-attention block, which is quadratic in n, where n is the length of the context input and d is the hidden feature dimension of the model. For example, LLaMA 3.1 405B [LT24], one of the cuttingedge LLMs, supports n =128k and d = 4096, while taking 30.84M GPU training hours, which underscores the need for more eﬃcient training processes for such extensive context models. Given the extensive context lengths of LLMs, this quadratic time complexity results in critical challenges: (i) a marked decrease in training eﬃciency [HLZ+23, LYL+23]; and (ii) signiﬁcant energy usage, which in turn contributes to higher carbon dioxide emissions [SZM+23, SCZ+24].

One seminal work [AS23] showed that the self-attention inference can be approximated in almost linear time. However, this result is for the inference time (forward pass), but does not address the main challenge, which is the expensive computation in the training time (backward pass). In this work, we address this main challenge, by proving that the gradient computation in the backpropagation of self-attention can be approximated in almost linear time. This suggests we may be able to save the substantial resources required for training LLMs.

### 1.1 Key background

We ﬁrst introduce some basic background, starting with deﬁning the softmax function and the self-attention module.

- Deﬁnition 1.1 (Softmax). Let z ∈ Rn. We deﬁne Softmax : Rn → Rn satisfying Softmax(z) := exp(z)/ exp(z),1n .

Here we apply exp to a vector entry-wise.

- Deﬁnition 1.2 (Self-attention module). Let X ∈ Rn×d denote the input sequence, where n is the

number of input tokens and d is the hidden dimension size. Let WQ,WK,WV ∈ Rd×d be the query, key and value weight matrix. The self-attention function Attn(X) with weights is:

Attn(X) = Softmax(XWQWK⊤X⊤/d) · XWV . where Softmax is applied to each row of its input matrix. The attention can be re-written as: Attn(X) = f(X) · XWV ,

where (1) A := exp(XWQWK⊤X⊤/d) ∈ Rn×n and exp is applied element-wise, (2) D := diag(A1n) ∈ Rn×n, and (3) f(X) := D−1A ∈ Rn×n is the attention matrix.

In contemporary LLMs, the architecture typically incorporates multiple layers of attention. Consequently, in order to design a fast training algorithm for the entire model, it is imperative to examine self-attention within the multi-layer transformer structure formally deﬁned as follows.

- Deﬁnition 1.3 (Multi-layer transformer). Let m denote the number of transformer layers in the

model. Let X be the input sequence. Let gi denote components other than self-attention in the i-th transformer layer, and assume its forward and backward computations can be run in time linear in its input sequence length. Let Attni denote the self-attention module in the i-th transformer layer with weights WQi,WKi,WVi (see also Deﬁnition 1.2). We deﬁne an m-layer transformer as

Fm(X) := gm ◦ Attnm ◦ gm−1 ◦ Attnm−1 ◦ ··· ◦ g1 ◦ Attn1 ◦ g0(X), where ◦ denotes function composition.

In Deﬁnition 1.3, the gi includes the layer norm, MLP, residual connection, dropout, positional encoding, multi-head concatenation, and other operations. All forward and backward computations of these practical modules can be run in linear time with respect to n. Thus, in this work, we mainly focus on the acceleration of self-attention module. Speciﬁcally, as shown in Deﬁnition 1.2, the n×n attention matrix f(X) dominates the computational complexity, introducing a quadratic bottleneck. In the exact computation case, if the attention matrix is full rank, no acceleration is possible. However, by compromising negligible accuracy, designing a fast sub-quadratic algorithm becomes feasible. Fortunately, by employing the polynomial kernel approximation method from [AA22], we can approximate the attention matrix and achieve an almost linear time n1+o(1) algorithm, eﬀectively breaking the quadratic bottleneck.

- 1.2 Our contributions We now state our main result as follows:

Theorem 1.4 (Main result, informal version of Theorem 4.2). Let n be the number of tokens, and d the hidden dimension size. We assume d = O(log n) and each number in matrices can be written using O(log n) bits. Assume the number of layers m = no(1). There exists an algorithm (Algorithm 1) that can compute the gradient of multi-layer self-attention (see also Deﬁnition 1.3) in almost linear time n1+o(1), where the approximation error of the entire model can be bounded by 1/poly(n).

Our assumption is mild when the context length n is large, as the feature dimension d is usually regarded as a constant, which is also used in [AA22]; similarly, the number of layers is usually much smaller than n and regarded as a constant. Our results indicate that large language models (LLMs) can be trained in almost linear time n1+o(1) and maintain a robust approximation guarantee, while the traditional way takes Ω(n2) time. This advancement is realized through the application of polynomial kernel approximation [AS23, AS24a]. To be more speciﬁc, by leveraging the inherent sparsity within the dense attention matrix, we perform eﬃcient low-rank approximation, thereby signiﬁcantly accelerating the computation of the dense matrices. Our framework is applicable to general loss functions, making it universally applicable. Furthermore, our analysis holds when the multi-layer transformer model contains many practical sub-modules, such as residual connection, casual mask, and multi-head attention (Section 6).

Numerous studies, including FlashAttention [DFE+22, Dao23, SBZ+24], quantization techniques [HCL+24, LTT+24], and sparsity approaches [HJK+24, MCW+24], have empirically focused on accelerating attention mechanisms. However, theoretically, these methods are still constrained

by quadratic time complexity. In this study, we introduce an innovative acceleration technique (Algorithm 1) that eﬀectively overcomes this quadratic bottleneck, backed by solid theoretical foundations (Theorem 4.2). Moreover, this new method is designed to be seamlessly integrated with existing approaches to further enhance their performance (see Section 6).

Our contributions are as follows:

- • We introduce a fast computation method that allows the gradient of each self-attention layer to be approximated in almost linear time n1+o(1) with 1/poly(n) error, where n is the input sequence length, breaking the quadratic time complexity bottleneck (Theorem 4.1).
- • We extend our single-layer results to module-wise gradient computation so that our Algorithm 1 approximates gradient computation in m · n1+o(1) time for m-layer transformer. Importantly, the approximation of the gradient diverges from the exact gradient by an error of 1/poly(n) across the entire model (Theorem 4.2).
- • Additionally, our analysis holds for general loss functions and when the multi-layer transformer model contains residual connection, casual mask, and multi-head attention. Our results can be applied to any gradient-based algorithm, e.g., training, full ﬁne-tuning, prompttuning, and so on (Section 6).

## 2 Related Work

Long-context modeling in LLMs. As LLMs grow in size and capability, in-context learning (ICL) [MLH+22, SWXL24, XSL24, CLL+24] has become a preferred method for directing these models to perform a variety of tasks, as opposed to the resource-intensive process of ﬁne-tuning. Nonetheless, research has indicated that longer prompts can impair LLMs performance due to the limitation on maximum sequence length during pre-training [LZD+24]. Consequently, extending the maximum sequence length during pre-training and ﬁne-tuning stages is imperative. Enhancing training eﬃciency is crucial given the prevalent use of the Transformer architecture in LLMs, which incurs a quadratic computational cost relative to sequence length. Addressing this challenge, some studies have explored continued ﬁne-tuning of LLMs with extended context lengths [TSP+24], while others have experimented with the interpolation and extrapolation capabilities of positional embedding [CWCT23]. [SMN+24] handles long context by compressing the input tokens. However, these approaches have not fundamentally addressed the core issue: the quadratic computational cost associated with sequence length in the attention mechanism [KWH23, FCA23]. In this study, we delve into accelerating the attention mechanism, thereby addressing the long-context modeling issue at its essence.

Attention acceleration. Attention mechanism has faced criticism due to its quadratic time complexity with respect to context length, a concern exacerbated by the increasing length in modern large language models (LLMs) such as GPT-4 [AAA+23], Claude 3.5 [Ant24], Llama 3.1 [TLI+23, LT24], etc. Nevertheless, this limitation can be circumvented by employing polynomial kernel approximation techniques [AA22], which enable the derivation of a low-rank representation of the attention matrix. This innovation signiﬁcantly accelerates both the training and inference processes of a single attention layer, achieving almost linear time complexity [AS23, AS24a], while our work supports both training and inference for any multi-layer transformer. Furthermore, this approach can be extended to higher-order attention mechanisms, i.e., tensor attention, maintaining almost linear time complexity during both training and inference [AS24b, LSSZ24b]. Moreover,

there are other theoretical approaches. For instance, [LLS+24d] introduces the conv-basis method to accelerate attention computation. [HJK+24] proposes a near-linear time algorithm under the assumptions of uniform softmax column norms and sparsity.

Roadmap. Our paper is organized as follows. Section 3 provides essential conceptions and key deﬁnitions across the whole paper. Section 4 presents our primary ﬁndings, where we articulate our novel algorithm that is capable of calculating gradients across the entire model in almost linear time. In Section 5, we explain the techniques we employ, including low-rank approximation, techniques for accelerating the computation of gradients, and an analysis of the approximation error. Section 6 provides various extensions of our algorithm. Lastly, we conclude this paper in Section 7.

## 3 Preliminary

Notations. For any positive integer n, we use [n] to denote set {1,2,··· ,n}. For two vectors x ∈ Rn and y ∈ Rn, we use x,y to denote the inner product between x,y. Namely, x,y = ni=1 xiyi. We use ei to denote a vector where only i-th coordinate is 1, and other entries are 0. For each a,b ∈ Rn, we use a ⊙ b ∈ Rn to denote the Hardamard product, i.e. the i-th entry of (a ⊙ b) is aibi for all i ∈ [n]. We use 1n to denote a length-n vector where all the entries are ones. We use A ∞ to denote the ℓ∞ norm of a matrix A ∈ Rn×d, i.e., A ∞ := maxi∈[n],j∈[d] |Ai,j|. We use poly(n) to denote some polynomial in n.

- 3.1 Loss function The loss function is the optimization objective in the training of LLMs, and we deﬁne it as follows.

Deﬁnition 3.1 (Loss function L(X)). For some input matrix X ∈ Rn×d, we deﬁne the one-unit loss function ℓ(X)j,k : Rn×d → R, for any j ∈ [n],k ∈ [d], and assume diﬀerentiability. Furthermore, we deﬁne the overall loss function L(X), such that

L(X) =

n

j=1

d

k=1

ℓ(X)j,k

Remark 3.2. Typically, the most widely used loss function in the LLM training procedure is the cross-entropy loss function, which can also be viewed as a summation of one unit loss function as in Deﬁnition 3.1. The output matrix of the multi-layer transformer needs to pass an additional linear layer to map the hidden dimension d to the vocabulary size dvoc. Assuming dvoc is a constant, the weight matrix dimensions for this additional MLP layer are d × dvoc. The probability tensor Ypred ∈ Rn×dvoc is the ﬁnal output. We denote the ground truth as Ygt ∈ Rn×dvoc corresponding to Ypred. According to the cross-entropy loss deﬁnition, the formula is expressed as

Lcross−entropy(X) = −

n

j=1

dvoc

k=1

(Ygt)j,k log((Ypred)j,k)

where the summation iterates over all elements, and the ground truth (Ygt)j,k = 1 for the correct class and 0 otherwise.

- 3.2 Closed forms of gradient components

In training large language models (LLMs), updating the model necessitates computing the gradient of weights for every layer. Consequently, it becomes essential to derive the closed-form expressions

for all corresponding gradient components with respect to the weights of the query, key, and value matrices in the transformer model. We ﬁrst deﬁne some intermediate variables before detailing these gradient components in each self-attention transformer layer.

Deﬁnition 3.3 (Intermediate variables Ti). Let m denote the number of transformer layers in the model. Let m-layer self-attention transformer be deﬁned as Deﬁnition 1.3. Let d denote the hidden dimension. Let n denote the sequence length. Let X ∈ Rn×d be the input sentence. Let gi denote components other than self-attention in the i-th transformer layer. Let Attni denote the self-attention module in the i-th transformer layer (see also Deﬁnition 1.2).

For i ∈ {0,1,2,··· ,m}, we deﬁne Ti(X) ∈ Rn×d be the intermediate variable (hidden states) output by i-th layer self-attention transformer. Namely, we have

g0(X), i = 0; (gi ◦ Attni)(Ti−1(X)), i ∈ [m].

Ti(X) =

Here, we use ◦ to denote function composition.

Then, we are ready to introduce the closed forms of the three gradient components in a single self-attention transformer layer. Notably, according to the chain rule, the gradient of the k-th transformer layer in LLMs depends on the gradient components from the (k + 1)-th transformer layer. The gradient can be calculated for every transformer layer by combining the upstream and local gradients. The closed forms of the gradients for each layer in multi-layer transformers are formalized in the following lemma (Lemma 3.4).

Lemma 3.4 (Closed form of gradient components, informal version of Lemma C.4). Let L(X) be deﬁned as in Deﬁnition 3.1, and the m-layer transformer deﬁned as in Deﬁnition 1.3. Let WQi,WKi,WVi ∈ Rd×d denote the attention weight in the i-th attention. Let Ti(X) denote the intermediate variable output by i-th self-attention transformer layer (see Deﬁnition 3.3). Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

i(Ti−1(X)). For j ∈ [n],k ∈ [d], let Gi(j,k) denote the (j,k)-th entry of Gi, let dAttni(Ti−1(X))j,k

[Figure 2]

dTi−1(X) ∈ Rn×d denote the gradient of (j,k)-th entry of Attni(Ti−1(X)). Then, we can show that

[Figure 3]

- • Part 1. dL(X)

[Figure 4]

dTi−1(X)

=

n

j=1

d

k=1

Gi(j,k) ·

dAttni(Ti−1(X))j,k dTi−1(X)

[Figure 5]

.

- • Part 2. Let W∗i be WQi,WKi or WVi, then dL(X)

n

d

dAttni(Ti−1(X))j,k dW∗i

=

.

Gi(j,k) ·

[Figure 6]

[Figure 7]

dW∗i

j=1

k=1

Our main results are based on the above closed forms of four gradient components.

- 4 Main Results Now, we present our main ﬁndings. We will work through this section in the following order:

- In Section 4.1, we delineate the computational eﬃciency of our gradient calculation methods in each single layer. In Section 4.2, we introduce our main theorem (Theorem 4.2) for multi-layer transformer by integrating the preceding results and provide our main algorithm (Algorithm 1). Section 4.3 discusses how we transcend the previous works.

### 4.1 Fast computing for single layer

In the case of single-layer attention, we provide our theorem that state the three gradient components can be calculated in almost linear time with negligible error.

- Theorem 4.1 (Single-layer gradient approximation). We assume d = O(log n) and each number in matrices can be written using O(log n) bits. Assume the number of layers m = no(1). Let L(X) be deﬁned as Deﬁnition 3.1. Suppose we have a single-layer self-attention transformer model (m = 1

in Deﬁnition 1.3). We can approximate one-layer self-attention for three gradient components, i.e.

dL(X)

[Figure 8]

dX , dWdL(X)

[Figure 9]

QWK⊤ and ddLW(X)

[Figure 10]

V

, in n1+o(1) time with 1/poly(n) error. Proof. We ﬁnish the proof by Lemma 5.1, 5.2 and 5.3.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

4.2 Fast computing for multi-layer transformers

Based on the results demonstrated in previous sections, we are ready to introduce our main result: the gradients of the whole transformer model can be approximated in almost linear time.

- Theorem 4.2 (Main result, formal version of Theorem 1.4). Let m denote the number of transformer layers. We assume d = O(log n) and each number in matrices can be written using O(log n) bits. Assume the number of layers m = no(1). We can show that, for any i ∈ [m], all the gradient components (see also Lemma 3.4) of the i-th layer can be computed by Algorithm 1 in almost linear time n1+o(1), and the approximation error of the entire m layer transformer model can be bounded by 1/poly(n). Proof. We prove the theorem by directly combining Theorem 4.1 and Lemma 5.5.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Theorem 4.2 demonstrates that, during the training of a multi-layer transformer model, at each training iteration, the gradient computation for the weight matrices of each layer can be performed in almost linear time n1+o(1). This result supports the feasibility of fast training for any transformer-based large language models (LLMs). In Algorithm 1, we illustrate the process of back-propagating gradients from the m-th transformer layer back to the ﬁrst layer. This algorithm highlights the signiﬁcance of the gradient with respect to the intermediate variables Ti(X). Due to the application of the chain rule in gradient computation, the gradient of Ti(X) is indispensable for determining the gradients of the weight matrices WQi,WKi and WVi at the i-th layer. Consequently, by iteratively computing the gradient for Ti(X), we systematically propagate the gradient through to the initial transformer layer. Additionally, our algorithm is capable of computing the gradient with respect to the input data X. Therefore, our algorithm also supports fast prompt tuning. For a more in-depth discussion on this topic, please refer to Section 6.

### 4.3 Beyond the previous work

Our algorithm exhibits signiﬁcant advancements over two seminal prior studies, namely [AS23] and [AS24a]. In [AS23], the authors proposed an almost linear time algorithm for computing the forward process of the attention mechanism. In contrast, [AS24a] introduced an almost linear time algorithm for the backward of attention mechanism. However, [AS24a] has the following limitations: (1) only computing gradients for a single layer of the attention mechanism, which cannot extend to multiple layers; (2) calculating gradients with respect to a speciﬁc loss, namely the ℓ2 loss; (3) computing gradients only for the weight matrix WQi,WKi (as deﬁned in Deﬁnition 1.2), but ignore other crucial components such as the MLP layer following attention computation and the activation

[Figure 19]

Algorithm 1 Almost Linear Time (ALT) Multi-layer Transformer Gradient Approximation

[Figure 20]

- 1: datastructure ALTGrad ⊲ Theorem 4.1 and 4.2
- 2: members
- 3: n ∈ R: the length of input sequence
- 4: d ∈ R: the hidden dimension
- 5: m ∈ R: the number of transformer layers
- 6: L(X) ∈ R: the loss function ⊲ Deﬁnition 3.1
- 7: Ti ∈ Rn×d: the output of i-th transformer layer
- 8: Attni ∈ Rn×d: the output that pass i-th attention layer
- 9: WQi,WKi,WVi ∈ Rd×d : the weight matrices in i-th transformer layer
- 10: end members
- 11:
- 12: procedure SingleGrad(dLdT(X)

[Figure 21]

i

) ⊲ Theorem 4.1

- 13: Compute Gi = ddLAttn(X)

[Figure 22]

i

via Lemma 5.4 ⊲ n1+o(1) time

- 14: Compute D6, D7, D8, D2, D4 via Lemma E.5, E.6, E.8, E.10 ⊲ n1+o(1) time
- 15: /* Approximate ddLT(X)

[Figure 23]

i−1

, Lemma 5.1 */

- 16: gt ← D6 + D7 + D8 + D2 + D4 ⊲ n1+o(1) time
- 17: /* Approximate dWdL(X)

[Figure 24]

QiWK⊤

i

, Lemma 5.2 */

- 18: Construct U3,V3 via Lemma 5.2 ⊲ n1+o(1) time
- 19: gw ← (Ti⊤−1U3) · (V3⊤Ti−1) ⊲ n1+o(1) time
- 20: /* Approximate ddLW(X)

[Figure 25]

Vi

, Lemma 5.3 */

- 21: Construct U1,V1 via Lemma C.13 ⊲ n1+o(1) time
- 22: gv ← (Ti⊤−1U1) · (V1⊤Gi) ⊲ n1+o(1) time
- 23: return gt, gw, gv ⊲ gt is the approximated ddLT(X)

[Figure 26]

i−1

for back-propagation

- 24: end procedure
- 25:
- 26: procedure MultiGrad(L(X)) ⊲ Theorem 4.2
- 27: Compute ddLT(X)

[Figure 27]

m

⊲ O(nd) time

- 28: gt ← ddLT(mX)

[Figure 28]

- 29: for i = m → 1 do
- 30: gt, gw, gv ← SingleGrad ( gt)
- 31: Optimize WQi,WKi via gw using optimizer
- 32: Optimize WVi via gv using optimizer
- 33: end for
- 34: end procedure
- 35: end datastructure

[Figure 29]

function. These limitations are inherent in their technique and prevents the applicability of the method in multiple layer transformers.

In our work, we have the following improvements beyond previous work: (1) we enable almost linear time gradient computation across an entire transformer layer, incorporating both the MLP layer and the activation function; (2) our algorithm supports gradient calculation for general loss function L(X) (see Deﬁnition 3.1); (3) we extend the gradient calculation to include not only WQi,WKi but also Ti(X) and WVi. These advancements collectively demonstrate a substantial

leap forward from the methodologies in [AS23] and [AS24a].

## 5 Technical Overview

### 5.1 Low-rank approximation for attention matrix

In this section, we delve into the crucial techniques behind our work: the low-rank approximation of the attention matrix, which is achieved through the polynomial method [ACSS20, AA22]. Drawing inspiration from [AS23], the intuition of this approximation lies in the fact that the attention matrix f(X) ∈ Rn×n (as deﬁned in Deﬁnition 1.2), also referred to as the similarity matrix in attention mechanism, can be eﬀectively approximated by low-rank matrices U1,V1 ∈ Rn×k1, where k1 = no(1). The naive method for calculating the attention matrix f(X) has a time complexity of O(n2), whereas the input data X ∈ Rn×d contains only d · n = n1+o(1) entries. This discrepancy suggests the potential of using low-rank representations of f(X) to design a fast algorithm.

An example of how to use the low-rank representations is the attention forward. First note

that approximating f(X) alone does not lead to a fast algorithm, since U1V1⊤ still requires n × n entries. But by using the structure of the attention Attn(X) := f(X)V where V = XWV , we can do it faster. By expressing f(X) as U1V1⊤, the attention forward becomes U1

V1⊤ k1×n

V

. It

n×d

n×k1

is well known that diﬀerent multiplication sequences can lead to dramatically diﬀerent numbers of operations required, so the order of matrix multiplications matters, which is indeed the case here. We ﬁrst perform V1⊤V ∈ Rk1×d and this cost O(k1nd) = n1+o(1) time. Then we can compute U1V1⊤V within O(nk1d) = n1+o(1) time.

This method signiﬁcantly reduces the computation time of the attention forward from O(n2) to almost linear time, n1+o(1). Driven by this technique and analyzing the close forms of the gradients, we extend the acceleration to the gradient of the entire model.

### 5.2 Accelerating gradient computation of Ti(X)

Based on the low-rank approximation method mentioned in Section 5.1, we compute the gradient of L(X) with respect to the intermediate variable Ti(X), which denotes the output of the i-th transformer layer. This computation is critical as it enables us to calculate gradients for other gradient components because of the chain rule.

Extending to general loss functions. According to the ﬁndings in [DSXY23], the gradient dL(X) dTi(X) can be decomposed into ﬁve components, namely C2(X),C4(X),C6(X),C7(X),C8(X), as detailed in Lemma D.1. However, the gradient result presented in [DSXY23] is tailored to a speciﬁc

[Figure 30]

loss function, the ℓ2 loss, limiting its applicability to a narrow range of scenarios. The primary challenge in extending the scope to encompass general loss functions is the absence of a uniﬁed analytical framework. Previous analyses [AS23, DSXY23] are limited to individual, speciﬁc loss functions. In this work, we introduce a comprehensive analysis framework (Deﬁnition 3.1) and we have demonstrated its applicability to the cross-entropy loss (see Remark 3.2). Consequently, by utilizing this generalized analysis framework, we extend the notation L(X) to include a wide range of general loss functions.

Accelerating the gradient computation. An important step in accelerating the gradient computation for the entire multi-layer transformer model is to accelerate the computation of the gradient on the intermediate variables Ti(X). The key challenge is that, to compute the gradient on Ti(X),

we need to compute the gradient on other components of one transformer layer, such as residual connection, multi-head attention, and causal attention mask (see more details in Section 6). We conduct comprehensive analysis on those components in the transformer layer, and prove that, by using low-rank approximation technique, the computation of gradient ddTL(X)

i(X) can be computed in almost linear time n1+o(1) (Lemma 5.1).

[Figure 31]

A crucial aspect of speeding up gradient computation for the entire multi-layer transformer model involves accelerating the calculation of gradients with respect to the intermediate variables Ti(X). The main challenge lies in the fact that computing the gradient of Ti(X) requires calculating the gradients for other components within a transformer layer, including the residual connection, multi-head attention, and causal attention mask (see Section 6). We have conducted an extensive analysis of these components within the transformer layer (see Section I, J, and K) and demonstrated that, through the application of low-rank approximation techniques, the gradient ddTL(X)

[Figure 32]

i(X) can be computed in almost linear time n1+o(1) (Lemma 5.1). In particular, we apply the lowrank approximation technique on the ﬁve terms C2(X),C4(X),C6(X),C7(X),C8(X) respectively, demonstrating that each term can be computed in almost linear time, n1+o(1), as shown in Section E. Then we aggregate those terms, as described in Section E.6. Since all ﬁve terms are n × d matrices, the summation of these terms remains almost linear in complexity. We then conclude that for any single-layer transformer, the gradient computation with respect to the input can be performed in almost linear time n1+o(1), as stated in Lemma 5.1.

The statement made for a single transformer layer can be readily generalized to any layer within an m-layer transformer model. For instance, consider the intermediate variables Ti(X) and Ti−1(X) (as deﬁned in Deﬁnition 3.3), where Ti(X) = (gi ◦ Attni)(Ti−1(X)). Given the gradient ddTL(X)

i(X), as established in the previous paragraph, we compute the gradient with respect to Ti−1(X), namely

[Figure 33]

dL(X)

dTi−1(X), in almost linear time n1+o(1). For a multi-layer transformer model, the above process can be conducted recursively. Thus, we can compute the gradient of the loss function L(X) on any

[Figure 34]

Ti(X) in almost linear time n1+o(1).

- Lemma 5.1 (Fast computation for ddTL(X)

i(X), informal version of Lemma E.11). Let L(X) be deﬁned as Deﬁnition 3.1. Let m denote the number of self-attention transformer layers (see Deﬁnition 1.3). Let Ti(X) denote the intermediate variable output by i-th self-attention transformer layer (see Deﬁnition 3.3). We show that ddTL(X)

[Figure 35]

i(X) can be approximated in n1+o(1) time, with 1/poly(n) approximation error.

[Figure 36]

### 5.3 Accelerating gradient computation of Wi and WV

i

- In Section 5.2, we detailed the fast computation of gradients for intermediate variables Ti(X). Let

Wi := WQiWK⊤

, with WQi and WKi representing the query and key weight matrices, respectively, the gradients of Wi and WVi represent all trainable weight matrices in a transformer layer. Consequently, by determining the gradients for Wi and WVi across each layer, we achieve almost linear time gradient back-propagation throughout multi-layer transformer models.

i

Fast gradient computation. The prior study in [AS24a] demonstrated that the gradient of Wi can be computed in almost linear time. We extend their ﬁndings by adapting their approach to accommodate general loss function L(X) (as deﬁned in Deﬁnition 3.1) and further generalize their results to include the gradient computation for both Wi and WVi in each transformer layer (Lemma 5.2 and 5.3).

- Lemma 5.2 (Fast computation for ddLW(X)

[Figure 37]

i

, informal version of Lemma F.5). Let L(X) be deﬁned as Deﬁnition 3.1, and m be the number of self-attention transformer layers (Deﬁnition 1.3). For any i ∈ [m], let Wi = WQiWK⊤

i

,WVi ∈ Rd×d denote the attention weight in the i-th transformer layer. We show that ddLW(X)

[Figure 38]

i

can be approximated in n1+o(1) time, with 1/poly(n) approximation error.

- Lemma 5.3 (Fast computation for ddLW(X)

[Figure 39]

Vi

, informal version of Lemma G.4). Let L(X) be deﬁned as Deﬁnition 3.1, and m be the number of self-attention transformer layers (Deﬁnition 1.3). For any i ∈ [m], let Wi = WQiWK⊤

i

,WVi ∈ Rd×d denote the attention weight in the i-th transformer layer. We show that ddLW(X)

[Figure 40]

Vi

can be approximated in n1+o(1) time, with 1/poly(n) approximation error.

5.4 Accelerating gradient computation for multi-Layer transformers

In this section, our focus turns to extending the single-layer transformer result from the previous section to a multi-layer transformer.

Running time analysis. We derive the closed-form gradient for the non-attention components within a transformer layer, namely the gi function deﬁned in Deﬁnition 1.3. With the closed-form gradient of gi established in Lemma H.1, we then demonstrate in Lemma 5.4 that the gradient computation for gi can also be achieved in almost linear time. Given that the number of layers m = no(1) is much smaller than n and the computation time for gradients on each layer is n1+o(1), we only need to iteratively repeat this procedure for m times. Therefore, the overall running time for computing gradients across the entire model is m · n1+o(1) = n1+o(1).

- Lemma 5.4 (Computation time for Gi, informal version of Lemma H.2). Let Ti(X) be deﬁned as Deﬁnition 3.3, i.e. Ti(X) = (gi ◦ Attni)(Ti−1(X)). Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 41]

i(Ti−1(X)). Assume we already have ddTL(X)

[Figure 42]

i(X). Assuming for any Z ∈ Rn×d, we have gi(Z) ∈ Rn×d, and gi(Z) = φ(Z · Wg), where Wg ∈ Rd×d and φ : R → R denotes any element-wise activation function. Let φ′ denote the derivative of φ. Then, we show that Gi can be computed in n1+o(1) time.

Error propagation analysis. Here, we consider the approximation error. In our setting, the approximation error originates from the low-rank approximation of the attention matrix, as detailed in Lemma C.13. As discussed in previous sections, the approximation error in each layer can be bounded by 1/poly(n). Then, we only need to focus on how error propagates in diﬀerent layers.

We ﬁrst prove that our 1/poly(n) approximation error statement holds for a single-layer transformer, as evidenced in Lemma H.3. Subsequently, through mathematical induction and leveraging the results of error propagation over the gradient of gi, we show that the approximation error can be bounded by 1/poly(n) for any m-layer transformer (Lemma 5.5), where the number of layers m is considered small.

- Lemma 5.5 (Multi-layer transformer gradient approximation, informal version of Theorem H.4). Let L(X) be deﬁned as Deﬁnition 3.1. Let X be deﬁned as Deﬁnition 1.2. Suppose we have a m-layer transformer (see Deﬁnition 1.3). Then, for any i ∈ [m], we can show that: (i) Running

, and ddLW(X)

i−1(X), ddLW(X)

time: Our algorithm can approximate dTdL(X)

in n1+o(1) time; (ii) Error bound: The approximation of the entire transformer model can be bounded by 1/poly(n). Namely, our algorithm output g satisﬁes g − dLd(XX) ∞ ≤ 1/poly(n).

[Figure 43]

[Figure 44]

[Figure 45]

i

Vi

[Figure 46]

## 6 Extensions

Multi-head attention and residual connections. Multi-head attention and residual connections are important components in attention mechanisms. While these components were not involved in our initial analysis for simplicity, incorporating them into our algorithm is straightforward, as detailed in Sections B.1 and B.2. Our algorithm maintains the capability to compute gradients for multi-layer transformers with multi-head attention and residual connection in almost linear time, suggesting that it can be readily adapted to more practical transformer models. The detailed analysis of incorporating residual connection with our framework can be found in Section J and Lemma J.3. For the synergy with multi-head attention, we provide comprehensive analysis in Section K and Lemma K.2.

Causal attention mask. The causal attention mask is critical to prevent transformers from “cheating” during training by ensuring future information is not used. The full-rank characteristic of the causal attention mask poses challenges for low-rank approximations. Nevertheless, we have identiﬁed a method to accelerate the computation of causal masked attention by exploiting its inherent properties, showing almost linear time complexity. A comprehensive explanation is provided in Section B.3. More detailed analysis can be found in Section I and Lemma I.7 and I.8.

Prompt tuning. Prompt tuning (or preﬁx learning) is a prevalent approach in parameter-eﬃcient ﬁne-tuning (PEFT), which requires the calculation of gradients on input data X. Given our algorithm’s ability to compute gradients for intermediate variables Ti in approximately linear time, we can similarly accelerate the gradient computation for input data X, thus enhancing the eﬃciency of the prompt tuning process. Additional details are provided in Section B.5.

Synergy with system-level attention acceleration. Many contemporary works focus on system-level acceleration of attention mechanisms, often by leveraging caching and mitigating I/O bottlenecks. Our algorithm has the potential to integrate with such advancements. By combining our theoretical improvements in computation time (from O(n2) to n1+o(1)) with system-level optimizations, the overall eﬃciency of attention mechanism computation may improve further. We leave the implementation of our method on GPU as future work since there are several coding challenges. More details can be found in Section B.4.

## 7 Conclusion

The attention mechanism in transformer models has quadratic time complexity with respect to the input token length. In this work, we proposed a novel Algorithm 1, which can approximately train a multi-layer transformer model in almost linear time, introducing only a small error. Importantly, our algorithm is designed to be compatible with general loss functions, practical sub-modules (residual connection, casual mask, multi-head attention), and general gradient-based algorithms. It may be seamlessly integrated with other system-level acceleration techniques. While we lack enterprise-scale computational resources for training large language models to provide empirical support, our theoretical ﬁndings suggest that we can accelerate the training of LLMs in practice.

## Acknowledgement

Research is partially supported by the National Science Foundation (NSF) Grants 2023239-DMS, CCF-2046710, and Air Force Grant FA9550-18-1-0166.

## References

[AA22] Amol Aggarwal and Josh Alman. Optimal-degree polynomial approximations for exponentials and gaussian kernel density estimation. In Proceedings of the 37th Computational Complexity Conference, pages 1–23, 2022.

[AAA+23] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[ACSS20] Josh Alman, Timothy Chu, Aaron Schild, and Zhao Song. Algorithms and hardness for linear algebra on geometric graphs. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS), pages 541–552. IEEE, 2020.

[Ant24] Anthropic. The claude 3 model family: Opus, sonnet, haiku, 2024.

- [AS23] Josh Alman and Zhao Song. Fast attention requires bounded entries. Advances in Neural Information Processing Systems, 36, 2023.
- [AS24a] Josh Alman and Zhao Song. The ﬁne-grained complexity of gradient computation for training large language models. arXiv preprint arXiv:2402.04497, 2024.

[AS24b] Josh Alman and Zhao Song. How to capture higher-order correlations? generalizing matrix softmax attention to kronecker computation. In The Twelfth International Conference on Learning Representations, 2024.

[BCB14] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

[BEPP22] Rouzbeh Behnia, Mohammadreza Reza Ebrahimi, Jason Pacheco, and Balaji Padmanabhan. Ew-tune: A framework for privately ﬁne-tuning large language models with diﬀerential privacy. In 2022 IEEE International Conference on Data Mining Workshops (ICDMW), pages 560–566. IEEE, 2022.

[BSZ23] Jan van den Brand, Zhao Song, and Tianyi Zhou. Algorithm and hardness for dynamic attention maintenance in large language models. arXiv preprint arXiv:2304.02207, 2023.

[CLG+24] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.

[CLL+24] Bo Chen, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, and Zhao Song. Bypassing the exponential dependency: Looped transformers eﬃciently learn in-context by multistep gradient descent, 2024.

[CLP+20] Beidi Chen, Zichang Liu, Binghui Peng, Zhaozhuo Xu, Jonathan Lingjie Li, Tri Dao, Zhao Song, Anshumali Shrivastava, and Christopher Re. Mongoose: A learnable lsh framework for eﬃcient neural network training. In International Conference on Learning Representations, 2020.

[CLS+24] Bo Chen, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, and Zhao Song. Hsr-enhanced sparse attention acceleration. arXiv preprint arXiv:2410.10165, 2024.

[CSY23] Timothy Chu, Zhao Song, and Chiwun Yang. How to protect copyright data in optimization of large language models? arXiv preprint arXiv:2308.12247, 2023.

[CWCT23] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.

[CYL+24] Weize Chen, Ziming You, Ran Li, Yitong Guan, Chen Qian, Chenyang Zhao, Cheng Yang, Ruobing Xie, Zhiyuan Liu, and Maosong Sun. Internet of agents: Weaving a web of heterogeneous agents for collaborative intelligence. arXiv preprint arXiv:2407.07061, 2024.

[CZY23] Shang Chai, Liansheng Zhuang, and Fengying Yan. Layoutdm: Transformer-based diﬀusion model for layout generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18349–18358, 2023.

[Dao23] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

[DBK+20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[DFE+22] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re´. Flashattention: Fast and memory-eﬃcient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

[DMS23] Yichuan Deng, Sridhar Mahadevan, and Zhao Song. Randomized and deterministic attention sparsiﬁcation algorithms for over-parameterized feature dimension. arXiv preprint arXiv:2304.04397, 2023.

[DSXY23] Yichuan Deng, Zhao Song, Shenghao Xie, and Chiwun Yang. Unmasking transformers: A theoretical approach to data recovery via attention weights. arXiv preprint arXiv:2310.12462, 2023.

[EKB+24] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Mu¨ller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectiﬁed ﬂow transformers for high-resolution image synthesis. In Forty-ﬁrst International Conference on Machine Learning, 2024.

- [FA22] Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. Advances in Neural Information Processing Systems, 35:4475–4488, 2022.

- [FA23] Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning, pages 10323–

10337. PMLR, 2023. [FCA23] Quentin Fournier, Gae´tan Marceau Caron, and Daniel Aloise. A practical survey on

faster and lighter transformers. ACM Computing Surveys, 55(14s):1–40, 2023. [GD23] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state

spaces. arXiv preprint arXiv:2312.00752, 2023.

[GLA+21] Kamal Gupta, Justin Lazarow, Alessandro Achille, Larry S Davis, Vijay Mahadevan, and Abhinav Shrivastava. Layouttransformer: Layout generation and completion with self-attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1004–1014, 2021.

[GMS23] Yeqi Gao, Sridhar Mahadevan, and Zhao Song. An over-parameterized exponential regression. arXiv preprint arXiv:2303.16504, 2023.

[GSY23] Yeqi Gao, Zhao Song, and Xin Yang. Diﬀerentially private attention computation. arXiv preprint arXiv:2305.04701, 2023.

[GSYZ23] Yeqi Gao, Zhao Song, Xin Yang, and Ruizhe Zhang. Fast quantum algorithm for attention computation. arXiv preprint arXiv:2307.08045, 2023.

[GXG+23] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

[HCI+21] Itay Hubara, Brian Chmiel, Moshe Island, Ron Banner, Joseph Naor, and Daniel Soudry. Accelerated sparse neural training: A provable and eﬃcient method to ﬁnd n: m transposable masks. Advances in neural information processing systems, 34:21099– 21111, 2021.

[HCL+24] Jerry Yao-Chieh Hu, Pei-Hsuan Chang, Haozheng Luo, Hong-Yu Chen, Weijian Li, Wei-Po Wang, and Han Liu. Outlier-eﬃcient hopﬁeld layers for large transformerbased models. In Forty-ﬁrst International Conference on Machine Learning (ICML), 2024.

[HCW+24] Jerry Yao-Chieh Hu, Bo-Yu Chen, Dennis Wu, Feng Ruan, and Han Liu. Nonparametric modern hopﬁeld models. arXiv preprint arXiv:2404.03900, 2024.

[HJK+24] Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David Woodruﬀ, and Amir Zandieh. Hyperattention: Long-context attention in near-linear time. In The Twelfth International Conference on Learning Representations, 2024.

[HLSL24] Jerry Yao-Chieh Hu, Thomas Lin, Zhao Song, and Han Liu. On computational limits of modern hopﬁeld models: A ﬁne-grained complexity analysis. In Forty-ﬁrst International Conference on Machine Learning (ICML), 2024.

[HLZ+23] Nan He, Hanyu Lai, Chenyang Zhao, Zirui Cheng, Junting Pan, Ruoyu Qin, Ruofan Lu, Rui Lu, Yunchen Zhang, Gangming Zhao, et al. Teacherlm: Teaching to ﬁsh rather than giving the ﬁsh, language modeling likewise. arXiv preprint arXiv:2310.19019, 2023.

[HSK+24] Jerry Yao-Chieh Hu, Maojiang Su, En-Jui Kuo, Zhao Song, and Han Liu. Computational limits of low-rank adaptation (lora) for transformer-based models. arXiv preprint arXiv:2406.03136, 2024.

[HWL24] Jerry Yao-Chieh Hu, Dennis Wu, and Han Liu. Provably optimal memory capacity for modern hopﬁeld models: Tight analysis for transformer-compatible dense associative memories. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, 2024.

[HWSL24] Jerry Yao-Chieh Hu, Weimin Wu, Zhao Song, and Han Liu. On statistical rates and provably eﬃcient criteria of latent diﬀusion transformers (dits). arXiv preprint arXiv:2407.01079, 2024.

[HYW+23] Jerry Yao-Chieh Hu, Donglin Yang, Dennis Wu, Chenwei Xu, Bo-Yu Chen, and Han Liu. On sparse modern hopﬁeld model. In Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS), 2023.

[JCR+22] Tian Jin, Michael Carbin, Dan Roy, Jonathan Frankle, and Gintare Karolina Dziugaite. Pruning’s eﬀect on generalization through the lens of training and regularization. Advances in Neural Information Processing Systems, 35:37947–37961, 2022.

[KKL20] Nikita Kitaev,  Lukasz Kaiser, and Anselm Levskaya. Reformer: The eﬃcient transformer. arXiv preprint arXiv:2001.04451, 2020.

[KMZ23] Praneeth Kacham, Vahab Mirrokni, and Peilin Zhong. Polysketchformer: Fast transformers via sketches for polynomial kernels. arXiv preprint arXiv:2310.01655, 2023.

[KWH23] Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. On the computational complexity of self-attention. In International Conference on Algorithmic Learning Theory, pages 597–619. PMLR, 2023.

[LARC21] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parametereﬃcient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, 2021.

[LCT+24] Na Liu, Liangyu Chen, Xiaoyu Tian, Wei Zou, Kaijiang Chen, and Ming Cui. From llm to conversational agent: A memory enhanced architecture with ﬁne-tuning of large language models. arXiv preprint arXiv:2401.02777, 2024.

[LJF+22] Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning: Prompt tuning can be comparable to ﬁne-tuning across scales and tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). Association for Computational Linguistics, 2022.

[LKM23] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.

[LL21] Xiang Lisa Li and Percy Liang. Preﬁx-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, 2021.

[LLR23] Yuchen Li, Yuanzhi Li, and Andrej Risteski. How do transformers learn topic structure: Towards a mechanistic understanding. In International Conference on Machine Learning, pages 19689–19729. PMLR, 2023.

- [LLS+24a] Chenyang Li, Yingyu Liang, Zhenmei Shi, Zhao Song, and Tianyi Zhou. Fourier circuits in neural networks: Unlocking the potential of large language models in mathematical reasoning and modular arithmetic. arXiv preprint arXiv:2402.09469, 2024.
- [LLS+24b] Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song, and Junwei Yu. Fast john ellipsoid computation with diﬀerential privacy optimization. arXiv preprint arXiv:2408.06395, 2024.
- [LLS+24c] Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song, and Yufa Zhou. Fine-grained attention i/o complexity: Comprehensive analysis for backward passes. arXiv preprint arXiv:2410.09397, 2024.
- [LLS+24d] Yingyu Liang, Heshan Liu, Zhenmei Shi, Zhao Song, and Junze Yin. Conv-basis: A new paradigm for eﬃcient attention inference and gradient computation in transformers. arXiv preprint arXiv:2405.05219, 2024.
- [LLS+24e] Yingyu Liang, Jiangxuan Long, Zhenmei Shi, Zhao Song, and Yufa Zhou. Beyond linear approximations: A novel pruning approach for attention matrix, 2024.

[LLSS24] Xiaoyu Li, Yingyu Liang, Zhenmei Shi, and Zhao Song. A tighter complexity analysis of sparsegpt. arXiv preprint arXiv:2408.12151, 2024.

[LMGH22] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In European conference on computer vision, pages 280–296. Springer, 2022.

[LPM15] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Eﬀective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.

[LPP+20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Ku¨ttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

[LSH+23] Yixin Liu, Kejian Shi, Katherine S He, Longtian Ye, Alexander R Fabbri, Pengfei Liu, Dragomir Radev, and Arman Cohan. On learning to summarize with large language models as references. arXiv preprint arXiv:2305.14239, 2023.

[LSS+24] Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, and Yufa Zhou. Looped relu mlps may be all you need as practical programmable computers. arXiv preprint arXiv:2410.09375, 2024.

[LSSS24] Yingyu Liang, Zhizhou Sha, Zhenmei Shi, and Zhao Song. Diﬀerential privacy mechanisms in neural tangent kernel regression. arXiv preprint arXiv:2407.13621, 2024.

- [LSSY24] Yingyu Liang, Zhenmei Shi, Zhao Song, and Chiwun Yang. Toward inﬁnite-long preﬁx in transformer. arXiv preprint arXiv:2406.14036, 2024.

- [LSSZ24a] Yingyu Liang, Zhenmei Shi, Zhao Song, and Yufa Zhou. Diﬀerential privacy of crossattention with provable guarantee. arXiv preprint arXiv:2407.14717, 2024.
- [LSSZ24b] Yingyu Liang, Zhenmei Shi, Zhao Song, and Yufa Zhou. Tensor attention training: Provably eﬃcient learning of higher-order transformers. arXiv preprint arXiv:2405.16411, 2024.
- [LSSZ24c] Yingyu Liang, Zhenmei Shi, Zhao Song, and Yufa Zhou. Unraveling the smoothness properties of diﬀusion models: A gaussian mixture perspective. arXiv preprint arXiv:2405.16418, 2024.

[LSW+24] Zhihang Li, Zhao Song, Weixin Wang, Junze Yin, and Zheng Yu. How to inverting the leverage score distribution? arXiv preprint arXiv:2404.13785, 2024.

- [LSY24] Xiaoyu Li, Zhao Song, and Junwei Yu. Quantum speedups for approximating the john ellipsoid. arXiv preprint arXiv:2408.14018, 2024.
- [LSZ23] Zhihang Li, Zhao Song, and Tianyi Zhou. Solving regularized exp, cosh and sinh regression problems. arXiv preprint arXiv:2303.15725, 2023.

[LT24] AI @ Meta Llama Team. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[LTT+24] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of Machine Learning and Systems, 6:87–100, 2024.

[LYL+23] Kai Lv, Yuqing Yang, Tengxiao Liu, Qinghui Gao, Qipeng Guo, and Xipeng Qiu. Full parameter ﬁne-tuning for large language models with limited resources. arXiv preprint arXiv:2306.09782, 2023.

[LZD+24] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. Long-context llms struggle with long in-context learning. arXiv preprint arXiv:2404.02060, 2024.

[MCW+24] Da Ma, Lu Chen, Pengyu Wang, Hongshen Xu, Hanqi Li, Liangtai Sun, Su Zhu, Shuai Fan, and Kai Yu. Sparsity-accelerated training for large language models. arXiv preprint arXiv:2406.01392, 2024.

[MGA+24] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boﬃ, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring ﬂow and diﬀusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.

[MLG24] Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36, 2024.

[MLH+22] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, 2022.

[MWY+23] Amama Mahmood, Junxiang Wang, Bingsheng Yao, Dakuo Wang, and Chien-Ming Huang. Llm-powered conversational voice assistants: Interaction patterns, opportunities, challenges, and design guidelines. arXiv preprint arXiv:2309.13879, 2023.

[Ope24] OpenAI. Searchgpt, 2024.

[PX23] William Peebles and Saining Xie. Scalable diﬀusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[QMS+23] Lianke Qin, Saayan Mitra, Zhao Song, Yuanyuan Yang, and Tianyi Zhou. Fast heavy inner product identiﬁcation between weights and inputs in neural network training. In 2023 IEEE International Conference on Big Data (BigData), pages 128–133. IEEE, 2023.

[QSS23] Lianke Qin, Zhao Song, and Baocheng Sun. Is solving graph neural tangent kernel equivalent to training graph neural network? arXiv preprint arXiv:2309.07452, 2023.

[QSZZ23] Lianke Qin, Zhao Song, Lichen Zhang, and Danyang Zhuo. An online and uniﬁed algorithm for projection matrix vector multiplication with application to empirical risk minimization. In International Conference on Artiﬁcial Intelligence and Statistics (AISTATS), pages 101–156. PMLR, 2023.

[RBL+22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjo¨rn Ommer. High-resolution image synthesis with latent diﬀusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[RWC+19] Alec Radford, Jeﬀrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 2019.

[SAMB24] Tanmay Singh, Harshvardhan Aditya, Vijay K Madisetti, and Arshdeep Bahga. Whispered tuning: Data privacy preservation in ﬁne-tuning llms through diﬀerential privacy. Journal of Software Engineering and Applications, 17(1):1–22, 2024.

[SBZ+24] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. arXiv preprint arXiv:2407.08608, 2024.

[SCZ+24] Jovan Stojkovic, Esha Choukse, Chaojie Zhang, Inigo Goiri, and Josep Torrellas. Towards greener llms: Bringing energy-eﬃciency to the forefront of llm inference. arXiv preprint arXiv:2403.20306, 2024.

[SLBK24] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and eﬀective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, 2024.

[SMN+24] Zhenmei Shi, Yifei Ming, Xuan-Phi Nguyen, Yingyu Liang, and Shaﬁq Joty. Discovering the gems in early layers: Accelerating long-context llms with 1000x input token reduction. arXiv preprint arXiv:2409.17422, 2024.

[SSC+22] Weiyan Shi, Ryan Shea, Si Chen, Chiyuan Zhang, Ruoxi Jia, and Zhou Yu. Just ﬁne-tune twice: Selective diﬀerential privacy for large language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6327–6340, 2022.

[SSU18] Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31, 2018.

[SWXL24] Zhenmei Shi, Junyi Wei, Zhuoyan Xu, and Yingyu Liang. Why larger language models do in-context learning diﬀerently? arXiv preprint arXiv:2405.19592, 2024.

[SY23] Zhao Song and Chiwun Yang. An automatic learning rate schedule algorithm for achieving faster convergence and steeper descent. arXiv preprint arXiv:2310.11291, 2023.

[SYYZ23] Zhao Song, Xin Yang, Yuanyuan Yang, and Lichen Zhang. Sketching meets diﬀerential privacy: fast algorithm for dynamic kronecker projection maintenance. In International Conference on Machine Learning (ICML), pages 32418–32462. PMLR, 2023.

[SYZ23] Zhao Song, Mingquan Ye, and Lichen Zhang. Streaming semideﬁnite programs: O(√n) passes, small space and fast runtime. arXiv preprint arXiv:2309.05135, 2023.

[Figure 47]

[SZK+22] John Schulman, Barret Zoph, Christina Kim, Jacob Hilton, Jacob Menick, Jiayi Weng, Juan Felipe Ceron Uribe, Liam Fedus, Luke Metz, Michael Pokorny, et al. Chatgpt: Optimizing language models for dialogue. OpenAI blog, 2(4), 2022.

[SZKS21] Charlie Snell, Ruiqi Zhong, Dan Klein, and Jacob Steinhardt. Approximating how single head attention learns. arXiv preprint arXiv:2103.07601, 2021.

[SZM+23] Siddharth Samsi, Dan Zhao, Joseph McDonald, Baolin Li, Adam Michaleas, Michael Jones, William Bergeron, Jeremy Kepner, Devesh Tiwari, and Vijay Gadepally. From words to watts: Benchmarking the energy costs of large language model inference. In 2023 IEEE High Performance Extreme Computing Conference (HPEC), pages 1–9. IEEE, 2023.

[TLI+23] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothe´e Lacroix, Baptiste Rozie`re, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and eﬃcient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[TSP+24] Szymon Tworkowski, Konrad Staniszewski, Miko aj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Mi os´. Focused transformer: Contrastive training for context scaling. Advances in Neural Information Processing Systems, 36, 2024.

[VSP+17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,  Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[VZB+23] Vijay Viswanathan, Chenyang Zhao, Amanda Bertsch, Tongshuang Wu, and Graham Neubig. Prompt2model: Generating deployable models from natural language instructions. arXiv preprint arXiv:2308.12261, 2023.

[WCY+23] Yuntao Wang, Zirui Cheng, Xin Yi, Yan Kong, Xueyang Wang, Xuhai Xu, Yukang Yan, Chun Yu, Shwetak Patel, and Yuanchun Shi. Modeling the trade-oﬀ of privacy preservation and activity recognition on low-resolution images. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pages 1–15, 2023.

[WCZ+23] Yilin Wang, Zeyuan Chen, Liangjun Zhong, Zheng Ding, Zhizhou Sha, and Zhuowen Tu. Dolﬁn: Diﬀusion layout transformers without autoencoder. arXiv preprint arXiv:2310.16305, 2023.

[WHHL24] Dennis Wu, Jerry Yao-Chieh Hu, Teng-Yun Hsiao, and Han Liu. Uniform memory retrieval with larger capacity for modern hopﬁeld models. In Forty-ﬁrst International Conference on Machine Learning (ICML), 2024.

[WHL+24] Dennis Wu, Jerry Yao-Chieh Hu, Weijian Li, Bo-Yu Chen, and Han Liu. STanhop: Sparse tandem hopﬁeld model for memory-enhanced time series prediction. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

[WMS+24] Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, and Neel Joshi. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. arXiv preprint arXiv:2406.14852, 2024.

[WSD+23] Zirui Wang, Zhizhou Sha, Zheng Ding, Yilin Wang, and Zhuowen Tu. Tokencompose: Grounding diﬀusion with token-level supervision. arXiv preprint arXiv:2312.03626, 2023.

[WXZ+24] Yilin Wang, Haiyang Xu, Xiang Zhang, Zeyuan Chen, Zhizhou Sha, Zirui Wang, and Zhuowen Tu. Omnicontrolnet: Dual-stage integration for conditional image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7436–7448, 2024.

[XCG+23] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023.

[XGH+21] Hu Xu, Gargi Ghosh, Po-Yao Huang, Prahal Arora, Masoumeh Aminzadeh, Christoph Feichtenhofer, Florian Metze, and Luke Zettlemoyer. Vlm: Task-agnostic videolanguage model pre-training for video understanding. arXiv preprint arXiv:2105.09996, 2021.

[XGW+22] Xinchao Xu, Zhibin Gou, Wenquan Wu, Zheng-Yu Niu, Hua Wu, Haifeng Wang, and Shihang Wang. Long time no see! open-domain conversation with long-term persona memory. arXiv preprint arXiv:2203.05797, 2022.

[XHH+24] Chenwei Xu, Yu-Chao Huang, Jerry Yao-Chieh Hu, Weijian Li, Ammar Gilani, HsiSheng Goan, and Han Liu. Bishop: Bi-directional cellular learning for tabular data with generalized sparse modern hopﬁeld model. In Forty-ﬁrst International Conference on Machine Learning (ICML), 2024.

[XSL24] Zhuoyan Xu, Zhenmei Shi, and Yingyu Liang. Do large language models have compositional ability? an investigation into limitations and scalability. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2024.

[XSW21] Jing Xu, Arthur Szlam, and Jason Weston. Beyond goldﬁsh memory: Long-term open-domain conversation. arXiv preprint arXiv:2107.07567, 2021.

[XZS+24] Chaojun Xiao, Zhengyan Zhang, Chenyang Song, Dazhi Jiang, Feng Yao, Xu Han, Xiaozhi Wang, Shuo Wang, Yufei Huang, Guanyu Lin, et al. Conﬁgurable foundation

models: Building llms from a modular perspective. arXiv preprint arXiv:2409.02877, 2024.

[ZBKR24] Michael Zhang, Kush Bhatia, Hermann Kumbong, and Christopher Re´. The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry. arXiv preprint arXiv:2402.04347, 2024.

[ZHDK23] Amir Zandieh, Insu Han, Majid Daliri, and Amin Karbasi. Kdeformer: Accelerating transformers via kernel density estimation. In International Conference on Machine Learning, pages 40605–40623. PMLR, 2023.

[ZHJL24] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[ZKAW23] Jieyu Zhang, Ranjay Krishna, Ahmed H Awadallah, and Chi Wang. Ecoassistant: Using llm assistant more aﬀordably and accurately. arXiv preprint arXiv:2310.03046, 2023.

[ZKV+20] Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, and Suvrit Sra. Why are adaptive methods good for attention models? Advances in Neural Information Processing Systems, 33:15383–15393, 2020.

[ZLD+24] Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B Hashimoto. Benchmarking large language models for news summarization. Transactions of the Association for Computational Linguistics, 12:39–57, 2024.

[ZTT+22] Bowen Zhang, Zhi Tian, Quan Tang, Xiangxiang Chu, Xiaolin Wei, Chunhua Shen, et al. Segvit: Semantic segmentation with plain vision transformers. Advances in Neural Information Processing Systems, 35:4971–4982, 2022.

# Appendix

Roadmap. In Section A, we provide further related works of this paper. In Section B, we provide a detailed discussion about several potential extensions of our framework.

In Section C, we introduce basic notations and concepts used in our paper, along with the low-rank approximation technique introduced in [AS23] and [AS24a]. In Section D, we provide details about how we integrate the gradient of Ti(X) into matrix form. In Section E, we explain how to apply the low-rank approximation technique to accelerate the computation for the gradient on Ti(X). In Section F, we extend the result of [AS24a] to arbitrary loss functions and accelerate the computation of gradient on W via the low-rank approximation technique. In Section G, we calculate the gradient on WV and accelerate the computation of the gradient on WV . In Section H, with the help of math induction, we analyze the time complexity and the approximation error across the entire model. In Section I, we discuss how our framework can expand to an attention mechanism with a causal attention mask. In Section J, we provide details about how to integrate our framework with attention mechanism with the residual connection. In Section K, we argue that, with the addition of multi-head attention, our algorithm can still achieve almost linear time gradient computation.

## A More Related Work

Attention mechanism. Attention mechanisms, including self-attention and cross-attention, are pivotal techniques employed in state-of-the-art neural networks. Since it was introduced in [VSP+17], it has gained widespread adoption across various domains. In particular, it is integral to decoderonly LLMs [RWC+19] and the Vision Transformer (ViT) architecture [DBK+20]. The former has been instrumental in the remarkable success of LLMs, while the latter has signiﬁcantly advanced the ﬁeld of computer vision, encompassing applications such as image generation [RBL+22, WSD+23, WXZ+24], detection [LMGH22], segmentation [ZTT+22], and layout generation [GLA+21, CZY23, WCZ+23]. Moreover, attention mechanism can be integrated into multi-modal models [XGH+21, ZHJL24, LSSZ24b, WMS+24], math reasoning [LLS+24a], diﬀusion models [PX23, LSSZ24c, HWSL24, EKB+24, MGA+24, LSW+24], diﬀerential privacy [BEPP22, SSC+22, WCY+23, LSSZ24a, SAMB24, CSY23, LSSS24, LLS+24b, SYYZ23] and many other techniques [LSS+24, LSY24, QMS+23, QSS23, QSZZ23, SYZ23, XZS+24, VZB+23].

Attention theory. [BCB14] introduced attention mechanisms in NLP, enhancing encoder-decoder

architecture with variable-length vectors to improve machine translation. Building on this, [LPM15] developed local and global attention variants, further reﬁning NLP tasks. Recent Large Language Model research has focused extensively on attention computation [DMS23, AS23, ZHDK23]. Studies by [ZHDK23, CLP+20, KKL20] use Locality Sensitive Hashing for attention approximation, with [ZHDK23] oﬀering eﬃcient dot-product attention. [BSZ23] and [AS23] explore static and dynamic attention calculations, while [LSZ23] investigates hyperbolic regression regularization. [DMS23] proposes algorithms for reducing attention matrix dimensionality in LLMs. Attention has also been examined from optimization and convergence perspectives [LLR23, GMS23, SZKS21, ZKV+20], investigating word co-occurrence learning [LLR23], regression problems with exponential activation functions [GMS23], attention mechanism evolution during training [SZKS21], and the impact of heavy-tailed noise on stochastic gradient descent [ZKV+20]. Theoretical explorations of attention variants include quantum attention [GSYZ23], tensor attention [AS24b, LSSZ24b], and diﬀerentially private attention [LSSZ24a, GSY23, LSSS24].

More methods for model acceleration. Various techniques have been developed for model acceleration. One approach involves modifying model architectures to enable faster inference, such as Mamba [GD23], Linearizing Transformers [ZBKR24], PolySketchFormer [KMZ23], and the Hopﬁeld Model [HCW+24, HCL+24, WHHL24, XHH+24, HLSL24, WHL+24, HYW+23, HWL24] and so on. Another line of work is to prune the weights in a neural network to reduce running time and memory consumption [HCI+21, JCR+22, FA22, FA23, SLBK24, LLSS24, LLS+24e]. In addition, speciﬁc techniques have been developed to accelerate LLM generation [CLS+24, CLL+24, SY23, LLS+24c].

## B Discussion and Extension Details

- In Section B.1, we argue that our framework can easily adapt to the multi-head attention mechanism. In Section B.2, we introduce how to integrate residual connection to our framework. In Section B.3, we detail the integration of the causal attention mask into our algorithm. In Section B.4, we discuss the possibility of the synergy between our theoretical side attention acceleration and the existing system-level attention acceleration mechanism. In Section B.5, we show how to expedite prompt tuning using our results.

### B.1 Multi-head attention

The multi-head attention mechanism was ﬁrst introduced by [VSP+17]. This innovation allows a token to simultaneously attend to multiple positions within the same layer, thereby enriching the model’s capacity for capturing various dependencies. However, this enhanced capability comes with an increase in the size of the attention matrix f(X) from 1 × n × n to h × n × n, where h is the number of attention heads. To mitigate the computational burden, each head’s vector is derived by splitting the original vector, reducing the dimensionality of each head to dh := d/h. To summarize, the key distinctions between multi-head and single-head attention are (1) an enlarged attention matrix f(X) and (2) a reduced dimensionality dh within each attention head.

Enlarged attention matrix. As previously discussed, the attention matrix’s dimensionality increases with the number of heads, h. Despite this expansion, the application of the low-rank approximation technique, as outlined in Section 5.1, ensures that the computation time for the attention matrix remains almost linear. Speciﬁcally, for a constant number of heads h in the multi-head mechanism, the time complexity for computing f(X) ∈ Rh×n×n is h · n1+o(1) = n1+o(1).

Reduced dimensionality. Another diﬀerentiating factor of multi-head attention is the lower dimensionality processed by each head, i.e. dh := d/h, compared the full d in single-head attention. This reduction ensures that the gradient computation time does not increase with the introduction of multiple attention heads.

We provide comprehensive analysis of the synergy of our algorithm with multi-head attention in Section K. We ﬁrst prove in Lemma K.2, with the addition of multi-head attention, the gradient over the attention mechanism can be computed in almost linear time. Then, we further prove that for any multi-layer transformer, with multi-head attention, the gradient can be computed in almost linear time as well.

### B.2 Residual connection

Residual connection is a pivotal technique in deep neural network architectures, eﬀectively addressing issues such as vanishing and exploding gradients during training process, and facilitating faster convergence of the model. Residual connection is also integrated into the standard attention mechanism. Formally, given the intermediate variable Ti(X) output by the i-th transformer layer as deﬁned in Deﬁnition 3.3, we provide the formal deﬁnition of residual connection in Deﬁnition J.1 and J.2. Since the residual connection only brings an additional add operation to each component and with Ti(X) belonging to the space Rn×d, the residual connection introduces only a marginal computational overhead of O(n · d) per layer. Consequently, the total computational cost for each layer is O(n · d) + n1+o(1) = n1+o(1). Hence, by intuition, the inclusion of residual connections does not compromise the overall complexity of our method.

The detailed analysis is provided in Section J, where we ﬁrst prove in Lemma J.3, that if the gradient over one structure can be computed in almost linear time, then with the addition of the residual connection, the gradient can also be computed in almost linear time. Then we use math induction to extend our result to the entire multi-layer transformer model.

### B.3 Causal attention mask

In transformer training, attention mask is a crucial component, designed to prevent a given token from attending to future tokens in the sequence. Causal attention mask is a widely used attention mask, which is conﬁgured as a lower triangular matrix, where elements on or below the main diagonal are ones, with all other entries being zeros.

Now we describe how to incorporate this into our algorithm. Let M ∈ {0,1}n×n represent the causal attention mask (see Deﬁnition I.2). Let f(X) := D−1(M ⊙ A) where A = exp(XWX⊤/d) and D := diag((M ⊙ A) · 1n). Lemma I.1 reveals that A has a low-rank representation given by U0V0⊤. Using Lemma I.3, we know (M ⊙ (U0V0⊤)) · v for any vector v ∈ Rn can be computed in almost linear time.

To integrate the causal mask into the gradient computation within each transformer layer, we ﬁrst ﬁnd all instances that have the structure of f(X)·H or (f(X)⊙(UV ⊤))·H, where H,U,V are low rank matrices. Then, we replace f(X) with f(X) in these instances. More detailed analysis of causal attention can be found in Section I. To be more speciﬁc, we group the gradient components for Ti,Wi,WVi into two categories, one for dot product (Lemma I.7), another for Hadamard product (Lemma I.8). After showing each component can be calculated in almost linear time, the overall gradient computation remains n1+o(1) time. Thus, our framework can seamlessly accommodate causal attention masks.

### B.4 System-level attention acceleration

The attention computing acceleration involves a two-pronged strategy that leverages both systemlevel improvements (e.g. Flash Attention [DFE+22, Dao23, SBZ+24]) and the theoretical time complexity improvements (e.g. our work and [HJK+24]).

Numerous eﬀorts have been made in the literature to accelerate attention calculations at the system level. For instance, Flash Attention [DFE+22, Dao23, SBZ+24] targets the I/O bottleneck inherent in attention mechanisms. Studies such as block-wise parallel decoding [SSU18] focus on implementing parallel decoding within transformer models to enhance inference speed. Additionally, recent advancements in the ﬁeld of speculative decoding, such as Medusa [CLG+24], leverage a smaller, more eﬃcient model to generate predictions, with the larger model only responsible for validating, the smaller model’s outputs [LKM23].

Despite these innovations, the aforementioned methods do not address the fundamental quadratic

time complexity O(n2) of the attention mechanisms. This presents an opportunity to complement our low-rank approximation technique, with these system-level optimizations, thereby achieving an even greater acceleration in attention computation. For instance, we could design an I/O-aware algorithm for Algorithm 1, similar to the approach taken by Flash Attention, to eﬀectively leverage GPU acceleration.

To implement our algorithm practically on GPU, we have some coding challenges to ﬁx: (1) we need to deﬁne some new tensor operations in PyTorch, e.g. Eq. (5), Eq. (8); (2) we need to systematically re-implement some back-propagation function of the current PyTorch function; (3) we need to implement some CUDA function to run our algorithm in parallel for the casual mask, see discussion in Section B.3. We may leave this as our future work.

### B.5 Prompt tuning

Prompt tuning, as introduced by various studies [LL21, LARC21, LJF+22, MLG24, HSK+24, LSSY24], has emerged as a parameter-eﬃcient ﬁne-tuning strategy for large language models (LLMs). Speciﬁcally, prompt tuning involves adjusting “soft prompts” conditioned on frozen LLMs. This method requires relatively small number of tuneable parameters compared with ﬁne-tuning the entire LLMs, making it a popular choice for conserving training resources, including data and computational power.

The analysis reveals that the essence of prompt tuning involves computing gradients with respect to the soft prompts Xp across the entire model. In both prompt tuning and full ﬁne-tuning, the quadratic O(n2) computational complexity of gradient calculation remains the same due to the self-attention mechanism inherent in LLMs.

In this work, leveraging the low-rank approximation technique discussed in Section 5.1, our algorithm (Algorithm 1) eﬃciently computes gradients on soft prompts Xp over the entire model in almost linear time. This suggests that our method is universal and can also be applied within traditional prompt tuning frameworks.

## C Preliminary on Gradient Calculation

- In Section C.1, we list several useful math facts used in the following sections of this paper. In Section C.2, we provide the close forms of the gradient components. In Section C.3, we introduce some mathematical deﬁnitions to facilitate understanding of gradient calculations. In Section C.4, we list some low rank approximation technique introduced in [AS23] and [AS24a]. In Section C.5, we demonstrate that the entries of matrices deﬁned in Section C.3 are bounded.

Notations. For two vectors x ∈ Rn and y ∈ Rn, we use x,y to denote the inner product between x,y. Namely, x,y = ni=1 xiyi. We use ei to denote a vector where only i-th coordinate is 1, and other entries are 0. For each a,b ∈ Rn, we use a ⊙ b ∈ Rn to denote the Hardamard product, i.e. the i-th entry of (a ⊙ b) is aibi for all i ∈ [n]. We use 1n to denote a length-n vector where all the entries are ones. We use A ∞ to denote the ℓ∞ norm of a matrix A ∈ Rn×d, i.e.

A ∞ := maxi∈[n],j∈[d] |Ai,j|. We use poly(n) to denote polynomial time complexity with respective to n.

- C.1 Basic math facts In this section, we provide some useful basic math facts,

- Fact C.1. Let x,y,z ∈ Rn. Then we have

- • x ⊙ y,z = x⊤ diag(y)z.
- • x,(y ⊙ z) = y,(x ⊙ z) = z,(y ⊙ x)
- • x,y = x ⊙ y,1n . Then, we introduce a classical folklore used for the Hadamard product of two matrices.

- Fact C.2 (Folklore, [AS24a]). Let U1,V1 ∈ Rn×k1. Let U2,V2 ∈ Rn×k2. Then we have

V2⊤ k2×n

V1⊤ k1×n

(V1 ⊘ V2)⊤ k1k2×n

) = (U1 ⊘ U2)

) ⊙ ( U2

( U1

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

n×k2

n×k1

n×k1k2

Here, given U1 ∈ Rn×k1 and U2 ∈ Rn×k2, the U1 ⊘ U2 ∈ Rn×k1k2 is the row-wise Kronecker product, i.e., (U1 ⊘ U2)i,l1+(l2−1)k1 := (U1)i,l1Ui,l2 for all i ∈ [n], l1 ∈ [k1] and l2 ∈ [k2].

### C.2 Close form of three gradient components

We ﬁrst restate the deﬁnition of self-attention, where we denote W := WQWK⊤ ∈ Rd×d for simplicity. Deﬁnition C.3 (Self-attention module). Let X ∈ Rn×d denote the input sequence, where n is the number of input tokens and d is the hidden dimension size. Let WV ∈ Rd×d be the value weight matrix, and let W := WQWK⊤ ∈ Rd×d be the key-query weight matrix. The self-attention function Attn(X) with weights W,WV is:

Attn(X) = Softmax(XWX⊤/d) · X · WV . where Softmax is applied to each row of its input matrix. The attention can be re-written as: Attn(X) = f(X) · X · WV ,

where (1) A := exp(XWX⊤/d) ∈ Rn×n and exp is applied element-wise, (2) D := diag(A1n) ∈ Rn×n, and (3) f(X) := D−1A ∈ Rn×n is the attention matrix.

Note that the gradient of WQ and WK can easily be calculated from the gradient of W, i.e.,

dL(X) dW ·

dW dWQ

dL(X) dWQ

=

[Figure 52]

[Figure 53]

[Figure 54]

dL(X) dW · WK where the ﬁrst step follows from the chain rule, and the second step follows from basic calculus.

=

[Figure 55]

Then, we show how to derive the close form for the gradient components within each layer of a multi-layer transformer. Lemma C.4 (Close form of gradient components, formal version of Lemma 3.4). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let Wi := WQiWK⊤

∈ Rd×d be the key-query weight matrix, WVi ∈ Rd×d be the value weight matrix for the i-th transformer layer.

i

- • Let Ti(X) denote the intermediate variable output by i-th self-attention transformer layer (see Deﬁnition 3.3).
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 56]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi, let dAttnid(TTi−1(X))i2,j2

i−1(X) ∈ Rn×d denote the gradient of (i2,j2)-th entry of Attni(Ti−1(X)).

[Figure 57]

Then, we can show that

- • Part 1. dL(X)

[Figure 58]

dTi−1(X)

=

n

i2=1

d

j2=1

Gi(i2,j2) ·

dAttni(Ti−1(X))i2,j2 dTi−1(X)

[Figure 59]

.

- • Part 2. dL(X)

[Figure 60]

dWi

=

n

i2=1

d

j2=1

Gi(i2,j2) ·

dAttni(Ti−1(X))i2,j2 dWi

[Figure 61]

.

- • Part 3.

dL(X) dWVi

[Figure 62]

n

=

i2=1

d

dAttni(Ti−1(X))i2,j2 dWVi

Gi(i2,j2) ·

.

[Figure 63]

j2=1

Proof. We have

- • L(X) ∈ R.
- • Attni(Ti−1(X)) ∈ Rn×d,Ti−1(X) ∈ Rn×d.
- • Wi ∈ Rd×d,WVi ∈ Rd×d. Therefore, we have
- • dTdL(X)

[Figure 64]

i−1(X) ∈ Rn×d, dAttndTi(Ti−1(X))

[Figure 65]

i−1(X) ∈ R(n×d)×(n×d).

- • ddLW(X)

[Figure 66]

i

∈ Rd×d, dAttnid(TWi−1(X))

[Figure 67]

i

∈ R(n×d)×(d×d).

- • ddLW(X)

∈ Rd×d, dAttnid(WTi−1(X))

∈ R(n×d)×(d×d). Then, simply applying chain rule, we can get the ﬁnal results.

[Figure 68]

[Figure 69]

Vi

Vi

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

### C.3 Basic notations for computing gradients

Before we move on to compute gradients, we need to deﬁne some useful notations. We begin with introducing the index for a matrix.

- Deﬁnition C.5 (Simpliﬁed notations). For any matrix Z ∈ Rn×d, for i ∈ [n],j ∈ [d], we have following deﬁnitions:

- • Let Zi,j scalar

and Z(i,j) denote the (i,j)-th entry of Z.

- • Let Zi,∗ d×1

and Z(i,∗) denote the i-th row of Z.

- • Let Z∗,j n×1

and Z(∗,j) denote the j-th column of Z.

Then, we deﬁne the exponential matrix in the attention mechanism.

- Deﬁnition C.6 (Exponential function u). If we have the below conditions,

- • Let X ∈ Rn×d
- • Let W := WQWK⊤ ∈ Rd×d

We deﬁne u(X) ∈ Rn×n as follows

u(X) := exp(XWX⊤) Then, we introduce the summation vector of the aforementioned exponential matrix.

- Deﬁnition C.7 (Sum function of softmax α). If we have the below conditions,

- • Let X ∈ Rn×d
- • Let u(X) be deﬁned as Deﬁnition C.6

We deﬁne α(X) ∈ Rn as follows

α(X) := u(X) · 1n

Then, with the help of the summation vector, we are ready to normalize the exponential matrix and get the softmax probability matrix.

- Deﬁnition C.8 (Softmax probability function f). If we have the below conditions,

- • Let X ∈ Rn×d
- • Let u(X) ∈ Rn×n be deﬁned as Deﬁnition C.6
- • Let α(X) ∈ Rn be deﬁned as Deﬁnition C.7

We deﬁne f(X) ∈ Rn×n as follows

f(X) := diag(α(X))−1u(X)

where we deﬁne f(X)⊤j

0 ∈ Rn is the j0-th row of f(X).

Besides the probability matrix introduced above, we introduce the value matrix in the following deﬁnition.

- Deﬁnition C.9 (Value function h). If we have the below conditions,

- • Let X ∈ Rn×d
- • Let WV ∈ Rd×d

We deﬁne h(X) ∈ Rn×d as follows

h(X) = XWV Then, we introduce s(X) to represent the output of the attention mechanism.

- Deﬁnition C.10 (Self-attention output s). If we have the below conditions,

- • Let f(X) be deﬁned as Deﬁnition C.8
- • Let h(X) be deﬁned as Deﬁnition C.9 We deﬁne s(X) ∈ Rn×d as follows

s(X) = f(X)h(X) Then, we introduce q(X) and p(X) to facilitate the calculation of the gradient on W.

- Deﬁnition C.11 (Deﬁnition of q(X)). If we have the below conditions,

- • Let h(X) ∈ Rn×d be deﬁned as in Deﬁnition C.9.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 74]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi. We deﬁne q(X) ∈ Rn×n as

q(X) = Gi

n×d

h(X)⊤ d×n

[Figure 75]

[Figure 76]

.

where we deﬁne q(X)⊤j

0 ∈ Rn is the j0-th row of q(X).

- Deﬁnition C.12 (Deﬁnition of p(X), Deﬁnition C.5 in [AS24a]). For every index j0 ∈ [n], we deﬁne p(X)j0 ∈ Rn as

p(X)j0 := (diag(f(X)j0) − f(X)j0f(X)⊤j0)q(X)j0 where we have p(X) ∈ Rn×n and we deﬁne p(X)⊤j

0 ∈ Rn is the j0-th row of p(X).

Furthermore, we deﬁne p1(X) = f(X)⊙q(X) and p2(X) = diag(p1(X)·1n)f(X). Additionally, we can calculate p(X) as

p(X) = p1(X) − p2(X)

### C.4 Low rank representations

Using [AS23]’s polynomial method techniques, we can obtain the following low-rank representation result.

- Lemma C.13 (Low rank representation to f, Section 3 of [AS23], Lemma D.1 of [AS24a]). For any A = o(√log n), there exists a k1 = no(1) such that: Let X ∈ Rn×d and W ∈ Rd×d be a square matrix. It holds that XW ∞ ≤ R, X ∞ ≤ R, then there are two matrices U1,V1 ∈ Rn×k1 such that U1V1⊤ −f(X) ∞ ≤ ǫ/poly(n). Here f(X) = D−1 exp(XWX⊤) (see also Deﬁnition C.8) and we deﬁne D = diag(exp(XWX⊤)1n) (see also Deﬁnition C.7). Moreover, these matrices U1,V1 can be explicitly constructed in n1+o(1) time.

[Figure 77]

A similar technique can be applied to s(X).

- Lemma C.14 (Low rank representation to s). Let d = O(log n). Assume that each number in the n × d matrices h(X) ∈ Rn×d can be written using O(log n) bits. Let n × d matrix s(X) ∈ Rn×d be

deﬁned as Deﬁnition C.10. Then, there are two matrices U1,V1 ∈ Rn×k1 we have U1V1⊤h(X) − s(X) ∞ ≤ ǫ/poly(n).

Proof. We can show that U1V1⊤h(X) − s(X) ∞ = U1V1⊤h(X) − f(X)h(X) ∞

= (U1V1⊤

[Figure 78]

[Figure 79]

n×n

−f(X)

[Figure 80]

[Figure 81]

n×n

)h(X)

[Figure 82]

[Figure 83]

n×d

∞

≤ n U1V1⊤

[Figure 84]

[Figure 85]

n×n

−f(X)

[Figure 86]

[Figure 87]

n×n

∞ h(X)

[Figure 88]

[Figure 89]

n×d

∞

≤ n U1V1⊤

[Figure 90]

[Figure 91]

n×n

−f(X)

[Figure 92]

[Figure 93]

n×n

∞ · poly(n)

≤ ǫ/poly(n)

where the 1st step is from the choice of s(X), the 2nd step comes from AC − BC = (A − B)C holds for any matrices A, B, and C, the 3rd step is because of basic linear algebra, the 4th step is due to each number in h(X) can be written using O(log(n)) bits, the ﬁfth step follows from

U1V1⊤ − f(X) ∞ ≤ ǫ/poly(n).

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

We can also get a low-rank representation of p1(x) and p2(x).

- Lemma C.15 (Low rank representation to p1(X), Lemma D.4 of [AS24a]). Let k1 = no(1). Let k2 = no(1). Assume that p1(X) := f(X) ⊙ q(X). Assume U1,V1 ∈ Rn×k1 approximates the f(X)

- such that U1V1⊤ − f(X) ∞ ≤ ǫ/poly(n). Assume U2,V2 ∈ Rn×k2 approximates the q(X) ∈ Rn×n
- such that U2V2⊤ − q(X) ∞ ≤ ǫ/poly(n). Then there are matrices U3,V3 ∈ Rn×k3 such that U3V3⊤ − p1(X) ∞ ≤ ǫ/poly(n). The matrices U3,V3 can be explicitly constructed in n1+o(1) time.

- Lemma C.16 (Low rank representation p2(X), Lemma D.5 of [AS24a]). Let k1 = no(1). Let k2 =

no(1). Let k4 = no(1). Assume that p2(X) is an n×n where j0-th row p2(X)j0 = f(X)j0f(X)⊤j

q(X)j0 for each j0 ∈ [n]. Assume U1,V1 ∈ Rn×k1 approximates the f(X) such that U1V1⊤ − f(X) ∞ ≤ ǫ/poly(n). Assume U2,V2 ∈ Rn×k2 approximates the q(X) ∈ Rn×n such that U2V2⊤ − q(X) ∞ ≤ ǫ/poly(n). Then there are matrices U4,V4 ∈ Rn×k4 such that U4V4⊤ −p2(X) ∞ ≤ ǫ/poly(n). The matrices U4,V4 can be explicitly constructed in n1+o(1) time.

0

### C.5 Bounded entries of matrices

In this section, we provide proof that entries of matrices are bounded. We begin with the exponential matrix f(X).

- Lemma C.17 (Bounded entries of f(X)). If we have the below conditions,

• Let f(X) ∈ Rn×n be deﬁned in Deﬁnition C.8. Then, we can show that

f(X) ∞ ≤ 1

- Proof. By Deﬁnition C.8, we have f(X) = diag(α(X))−1u(X)

By Deﬁnition C.7, we have

α(X) = u(X)1n Combining above two equations, we have

f(X) ∞ ≤ 1

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

A similar analysis can be applied to h(X) and s(X) as well. Lemma C.18 (Bounded entries of h(X)). If we have the below conditions,

- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV can be re represented using O(log(n)) bits.
- • Let h(X) ∈ Rn×d be deﬁned in Deﬁnition C.9. Then, we can show that

h(X) ∞ ≤ poly(n)

- Proof. By Deﬁnition C.9, we have h(X) := XWV

- Lemma C.19 (Bounded entries of s(X)). If we have the below conditions,

Then, we have

h(X) ∞ = XWV ∞ ≤ n X ∞ WV ∞ ≤ poly(n)

where the 1st step is from the deﬁnition of h(X), the 2nd step comes from basic linear algebra, the 3rd step is because of each entry in X and WV can be represented by O(log(n)) bits.

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV can be re represented using O(log(n)) bits.
- • Let s(X) ∈ Rn×d be deﬁned in Deﬁnition C.10. Then, we can show that

- Proof. By Deﬁnition C.10, we have

s(X) ∞ ≤ poly(n)

Then, we have

s(X)

= f(X)

h(X)

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

n×n

n×d

n×d

s(X) ∞ = f(X)h(X) ∞ ≤ n f(X) ∞ h(X) ∞ ≤ poly(n)

where the 1st step is from the deﬁnition of c(X), the 2nd step comes from basic linear algebra, the 3rd step is because of Lemma C.17, C.18.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

## D Matrix View

In this section, we dive into analyzing the gradient of dTdL(X)

i−1(X).

[Figure 116]

In Section D.1, we give the gradient of s(X) with respective to X. In Section D.2, we show the close form of the gradient on Ti(X) via the chain rule. In Section D.3, we integrate each Ci(X) to its corresponding matrix term Bi(X). In Section D.4, applying the similar technique used in the previous section, we integrate the gradient on Ti(X) into its corresponding matrix view. In Section D.5, we further apply matrix integration on each matrix term in the gradient on Ti(X) calculated in the previous section. In Section D.6, we give the matrix view of all gradient components.

### D.1 Gradient of s(X)

In this section, we give the gradient of s(X) with respective to X.

The results from [DSXY23] give the gradient of c(X). By chain rule, the gradient of s(X) is equivalent to the gradient of c(X) from [DSXY23], since c(X) = s(X) − B where B is a constant matrix.

- Lemma D.1 (Gradient of s(X)i0,j0, Lemma B.16 in [DSXY23]). If we have the below conditions,

• Let s(X) ∈ Rn×d be deﬁned as Deﬁnition C.10 Then, we have

- • Part 1. For all i0 = i1 ∈ [n], j0,j1 ∈ [d],

ds(X)i0,j0 dXi1,j1

= C1(X) + C2(X) + C3(X) + C4(X) + C5(X) where we have deﬁnitions:

[Figure 117]

- – C1(X) := −s(X)i0,j0 · f(X)i0,i0 · Wj1,∗,Xi0,∗
- – C2(X) := −s(X)i0,j0 · f(X)i0,∗,XW∗,j1
- – C3(X) := f(X)i0,i0 · h(X)i0,j0 · Wj1,∗,Xi0,∗
- – C4(X) := f(X)i0,∗ ⊙ (XW∗,j1),h(X)∗,j0
- – C5(X) := f(X)i0,i0 · (WV )j1,j0

- • Part 2. For all i0 = i1 ∈ [n], j0,j1 ∈ [d],

ds(X)i0,j0 dXi1,j1

= C6(X) + C7(X) + C8(X)

[Figure 118]

where we have deﬁnitions:

- – C6(X) := −s(X)i0,j0 · f(X)i1,i0 · Wj1,∗,Xi0,∗ ∗ This is corresponding to C1(X)
- – C7(X) := f(X)i1,i0 · h(X)i1,j0 · Wj1,∗,Xi0,∗ ∗ This is corresponding to C3(X)
- – C8(X) := f(X)i1,i0 · (WV )j1,j0 ∗ This is corresponding to C5(X)

- D.2 Gradient on Ti(X) In the Lemma D.2, we use the chain rule to calculate the close form of the gradient on Ti(X).

- Lemma D.2 (Gradient for Ti(X)). If we have the below conditions,

- • Let Attni be deﬁned as Deﬁnition C.3.
- • Let Ti(X) ∈ Rn×d be deﬁned as Deﬁnition 3.3.
- • Let s(X) be deﬁned as Deﬁnition C.10.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 119]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi. Then, we can show that, for i1 ∈ [n], j1 ∈ [d], we have

d

n

ds(X)i0,j0 dXi1,j1 Proof. By Lemma C.4, we have

dL(X) dTi−1(X)i1,j1

Gi(i0,j0) ·

=

[Figure 120]

[Figure 121]

j0=1

i0=1

n

dL(X) dTi−1(X)

=

[Figure 122]

i2=1

d

dAttni(Ti−1(X))i2,j2 dTi−1(X)

Gi(i2,j2) ·

.

[Figure 123]

j2=1

By Deﬁnition C.3 and Deﬁnition C.10, we have Attni(Ti−1(X)) = s(Ti−1(X))

Therefore, by combining above two equations and substituting variable Ti−1(X) = X, we have

dL(X) dTi−1(X)i1,j1

[Figure 124]

n

=

i0=1

d

ds(X)i0,j0 dXi1,j1

Gi(i0,j0) ·

[Figure 125]

j0=1

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

### D.3 Matrix view of C(X)

In this section, we will provide the matrix view of Ci(X) ∈ R, for i ∈ {6,7,8,2,4}. We will consider each Ci(X) one by one. We begin with C6(X).

- Lemma D.3 (Matrix view of C6(X)). If we have the below conditions,

- • Let C6(X,i1,j1) := −s(X)i0,j0 · f(X)i1,i0 · Wj1,∗,Xi0,∗ be deﬁned as in Lemma D.1.
- • We deﬁne a matrix B6(X) ∈ Rn×d. For all i1 ∈ [n],j1 ∈ [d], let B6(i1,j1) denote the (i1,j1)-th

entry of B6(X). We deﬁne B6(i1,j1) = C6(X,i1,j1). Then, we can show that

B6(X) n×d

[Figure 130]

[Figure 131]

= −s(X)i0,j0

[Figure 132]

[Figure 133]

1×1

f(X)∗,i0 n×1

[Figure 134]

[Figure 135]

(W · Xi0,∗)⊤ 1×d

[Figure 136]

[Figure 137]

Proof. We have

C6(X,i1,j1) = − s(X)i0,j0 · f(X)i1,i0 · Wj1,∗,Xi0,∗

= − s(X)i0,j0 · f(X)i1,i0 · Xi⊤0,∗Wj1,∗

where the 1st step is from the choice of C6(X), the 2nd step comes from a,b = a⊤b holds for any a,b ∈ Rd.

We have

B6(X)(i1,∗) d×1

[Figure 138]

[Figure 139]

= − s(X)i0,j0

[Figure 140]

[Figure 141]

1×1

f(X)i1,i0 1×1

[Figure 142]

[Figure 143]

W

d×d

Xi0,∗ d×1

[Figure 144]

[Figure 145]

Then, we have

B6(X) n×d

[Figure 146]

[Figure 147]

= −s(X)i0,j0

[Figure 148]

[Figure 149]

1×1

f(X)∗,i0 n×1

[Figure 150]

[Figure 151]

(W · Xi0,∗)⊤ 1×d

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

A similar analysis procedure can also be applied on C7(X).

- Lemma D.4 (Matrix view of C7(X)). If we have the below conditions,

- • Let C7(X,i1,j1) := f(X)i1,i0 · h(X)j0,i1 · Wj1,∗,Xi0,∗ be deﬁned as in Lemma D.1.
- • We deﬁne a matrix B7(X) ∈ Rn×d. For all i1 ∈ [n],j1 ∈ [d], let B7(i1,j1) denote the (i1,j1)-th entry of B7(X). We deﬁne B7(i1,j1) = C7(X,i1,j1).

Then, we can show that

Proof. We have

B7(X) n×d

[Figure 158]

[Figure 159]

·(W · Xi0,∗)⊤

= (f(X)∗,i0 ⊙ h(X)∗,j0)

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

n×1

1×d

C7(X,i1,j1) = f(X)i1,i0 · h(X)i1,j0 · Wj1,∗,Xi0,∗

= f(X)i1,i0 · h(X)i1,j0 · Wj⊤1,∗Xi0,∗

where the 1st step is from the choice of C7(X), the 2nd step comes from a,b = a⊤b holds for any a,b ∈ Rd.

We have

B7(X)(i1,∗) = f(X)i1,i0 · h(X)i1,j0 · W · Xi0,∗ Then, we have

·(W · Xi0,∗)⊤

= (f(X)∗,i0 ⊙ h(X)∗,j0)

B7(X) n×d

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

n×1

1×d

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Then, we provide an analysis of C8(X).

- Lemma D.5 (Matrix view of C8(X)). If we have the below conditions,

- • Let C8(X,i1,j1) := f(X)i1,i0 · (WV )j1,j0 be deﬁned as in Lemma D.1.
- • We deﬁne a matrix B8(X) ∈ Rn×d. For all i1 ∈ [n],j1 ∈ [d], let B8(i1,j1) denote the (i1,j1)-th entry of B8(X). We deﬁne B8(i1,j1) = C8(X,i1,j1).

Then, we can show that

Proof. We have

(WV )⊤∗,j0 1×d

= f(X)∗,i0

B8(X) n×d

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

n×1

C8(X,i1,j1) = f(X)i1,i0 · (WV )j1,j0 where the 1st step is from the choice of C7(X).

We have

B8(X)(i1,∗) = f(X)i1,i0 · (WV )∗,j0 Then, we have

(WV )⊤∗,j0 1×d

= f(X)∗,i0

B8(X) n×d

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

n×1

Now, we consider C2(X).

- Lemma D.6 (Matrix view of C2(X)). If we have the below conditions,

- • Let C2(X,j1) := −s(X)i0,j0 · f(X)i0,∗,XW∗,j1 be deﬁned as in Lemma D.1.
- • We deﬁne a matrix B2(X) ∈ Rd. For all j1 ∈ [d], the j1-th entry of B2(X) is deﬁned as C2(X,j1).

Then, we can show that

B2(X) d×1

[Figure 190]

[Figure 191]

= −s(X)i0,j0

[Figure 192]

[Figure 193]

1×1

W⊤ d×d

X⊤ d×n

f(X)i0,∗ n×1

[Figure 194]

[Figure 195]

Proof. We have

C2(X,j1) = − s(X)i0,j0 · f(X)i0,∗,XW∗,j1

= − s(X)i0,j0 · (XW∗,j1)⊤f(X)i0,∗

= −s(X)i0,j0

[Figure 196]

[Figure 197]

1×1

W∗⊤,j1 1×d

[Figure 198]

[Figure 199]

X⊤ d×n

f(X)i0,∗ n×1

[Figure 200]

[Figure 201]

where the 1st step is from the choice of C2(X), the second step follows from a,b = a⊤b, for any a,b ∈ Rn.

Then, we have

B2(X) d×1

[Figure 202]

[Figure 203]

= −s(X)i0,j0

[Figure 204]

[Figure 205]

1×1

W⊤ d×d

X⊤ d×n

f(X)i0,∗ n×1

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

Finally, we analyze C4(X), which is the last term we need to compute.

- Lemma D.7 (Matrix view of C4(X)). If we have the below conditions,

- • Let C4(X,j1) := f(X)i0,∗ ⊙ (XW∗,j1),h(X)∗,j0 be deﬁned as in Lemma D.1.
- • We deﬁne a matrix B4(X) ∈ Rd. For all j1 ∈ [d], the j1-th entry of B4(X) is deﬁned as C4(X,j1).

Then, we can show that

Proof. We have

= W⊤ d×d

B4(X) d×1

[Figure 212]

[Figure 213]

X⊤ d×n

(f(X)i0,∗ ⊙ h(X)∗,j0) n×1

[Figure 214]

[Figure 215]

C4(X,j1) = f(X)i0,∗ ⊙ (XW∗,j1),h(X)∗,j0 = f(X)i0,∗ ⊙ h(X)∗,j0,(XW∗,j1)

= (XW∗,j1)⊤(f(X)i0,∗ ⊙ h(X)∗,j0)

where the 1st step is from the choice of C4(X), the 2nd step comes from Fact C.1, and the last step follows from basic linear algebra.

### D.4 Matrix view of gradient on Ti(X)

Since we have got the matrix view of each Ci(X) term in the previous section, we can get the matrix view of the gradient on Ti(X) in Lemma D.8.

- Lemma D.8 (Matrix view of single entry of gradient). If we have the below conditions,

- • Let s(X) be deﬁned as Deﬁnition C.10.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 220]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.
- • Let B6(X),B7(X),B8(X) ∈ Rn×d be deﬁned in Lemma D.3, Lemma D.4, and Lemma D.5
- • Let B2(X),B4(X) ∈ Rd be deﬁned in Lemma D.6 and Lemma D.7. For any i0 ∈ [n],j0 ∈ [d], we have

ds(X)i0,j0 dX

(B2(X) + B4(X))⊤ 1×d

= Gi(i0,j0)

Gi(i0,j0) ·

+ ei0

)

·(B6(X) + B7(X) + B8(X)

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

1×1

n×1

n×d

Proof. By Lemma D.1, we have

- • Part 1. For all i0 = i1 ∈ [n], j0,j1 ∈ [d], ds(X)i0,j0

[Figure 228]

dXi1,j1

= C1(X) + C2(X) + C3(X) + C4(X) + C5(X) (1)

- • Part 2. For all i0 = i1 ∈ [n], j0,j1 ∈ [d],

ds(X)i0,j0 dXi1,j1

= C6(X) + C7(X) + C8(X) (2)

[Figure 229]

Since for any i1 ∈ [n],j1 ∈ [d], let Gi(i0,j0) · dsd(XXi)1i,j01,j0 denote the (i1,j1)-th entry of Gi(i0,j0) · ds(X)i0,j0

[Figure 230]

dX , we consider the following two cases:

[Figure 231]

- • Case 1. The i0-th row of Gi(i0,j0) · ds(XdX)i0,j0 .

[Figure 232]

- • Case 2. The other n − 1 rows of Gi(i0,j0) · ds(XdX)i0,j0 where i1 = i0. We ﬁrst consider Case 1. Recall that the matrix view of C2(X),C4(X) ∈ R are B2(X),B4(X) ∈ Rd, and the matrix view

[Figure 233]

of C6(X),C7(X),C8(X) ∈ R are B6(X),B7(X),B8(X) ∈ Rn×d, respectively. For k ∈ {6,7,8}, we use Bk(X)(s,∗) ∈ Rd to denote the s-th row of Bk(X). We use (Gi(i0,j0) · ds(XdX)i0,j0 )(i0,∗) ∈ Rd to denote the i0-th row of Gi(i0,j0) · ds(XdX)i0,j0 . Since C6(X),C7(X),C8(X) are the corresponding parts of C1(X),C3(X),C5(X), and by Eq. (1),

[Figure 234]

[Figure 235]

then we can have the following

ds(X)i0,j0 dX

)(i0,∗)

(Gi(i0,j0) ·

[Figure 236]

·(B6(X)(i0,∗) + B7(X)(i0,∗) + B8(X)(i0,∗) + B2(X) + B4(X))

= Gi(i0,j0)

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

1×1

d×1

We then consider Case 2. For k ∈ {6,7,8}, we use Bk(X)( = s,∗) ∈ R(n−1)×d to denote the matrix Bk(X) with the s-th

row removed. Similarly, we use (Gi(i0,j0) · ds(XdX)i0,j0 )( = i0,∗) ∈ R(n−1)×d to denote the matrix Gi(i0,j0) ·

[Figure 241]

ds(X)i0,j0 dX with the i0-th row removed. By Eq. (2), we have

[Figure 242]

ds(X)i0,j0 dX

)( = i0,∗) = Gi(i0,j0)

·(B6(X)( = i0,∗) + B7(X)( = i0,∗) + B8(X)( = i0,∗))

(Gi(i0,j0) ·

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

1×1

d×(n−1)

Combining Case 1 and Case 2 together, we have

ds(X)i0,j0 dX

= Gi(i0,j0)

Gi(i0,j0) ·

+ ei0

·(B6(X) + B7(X) + B8(X)

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

1×1

n×d

(B2(X) + B4(X))⊤ 1×d

)

[Figure 253]

[Figure 254]

n×1

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

Then, we have the matrix view of Ti(X) gradient.

- Lemma D.9 (Matrix view of Ti(X) gradient). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let T(X) be deﬁned as Deﬁnition 3.3.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 259]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.
- • Let B6(X),B7(X),B8(X) ∈ Rn×d be deﬁned in Lemma D.3, Lemma D.4, and Lemma D.5
- • Let B2(X),B4(X) ∈ Rd be deﬁned in Lemma D.6 and Lemma D.7. Then, we have

n

dL(X) dTi−1(X)

=

[Figure 260]

i0=1

d

Gi(i0,j0) 1×1

+ ei0

·(B6(X) + B7(X) + B8(X)

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

j0=1

n×d

Proof. By Lemma D.8, we have

ds(X)i0,j0 dX

Gi(i0,j0) ·

= Gi(i0,j0)

·(B6(X) + B7(X) + B8(X)

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

1×1

n×d

Then, by Lemma C.4 we have

(B2(X) + B4(X))⊤ 1×d

)

[Figure 270]

[Figure 271]

n×1

(B2(X) + B4(X))⊤ 1×d

+ ei0

)

[Figure 272]

[Figure 273]

n×1

n

dL(X) dTi−1(X)

=

[Figure 274]

i2=1

d

dAttni(Ti−1(X))i2,j2 dTi−1(X)

Gi(i2,j2) ·

.

[Figure 275]

j2=1

After combining the above two equations, we are done.

### D.5 Matrix view of each term in gradient on Ti(X)

In this subsection, we reduce the double summation to a matrix product for easy and clear analysis. We ﬁrst work on the B6 term.

- Lemma D.10 (Matrix view of B6(X) term). If we have the below conditions,

- • Let B6(X) n×d

[Figure 280]

[Figure 281]

= −s(X)i0,j0

[Figure 282]

[Figure 283]

1×1

f(X)∗,i0 n×1

[Figure 284]

[Figure 285]

(W · Xi0,∗)⊤ 1×d

[Figure 286]

[Figure 287]

be deﬁned in Lemma D.3.

- • We deﬁne z6(X) ∈ Rn×n, which satisﬁes z6(X)∗,i0

[Figure 288]

[Figure 289]

n×1

= (Gi(i0,∗)⊤

[Figure 290]

[Figure 291]

1×d

s(X)i0,∗ d×1

[Figure 292]

[Figure 293]

)f(X)∗,i0

[Figure 294]

[Figure 295]

n×1

- • Let f(X) ∈ Rn×n be deﬁned in Deﬁnition C.8.
- • Let W ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 296]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi. Then we have

d

n

W⊤ d×d

= −z6(X)

Gi(i0,j0) 1×1

#### X

B6(X) n×d

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

j0=1

i0=1

n×d

n×n

Proof.

n

d

Gi(i0,j0)B6(X) = −

i0=1

j0=1

= −

= −

= −

n

d

(W · Xi0,∗)⊤ 1×d

Gi(i0,j0) 1×1

s(X)i0,j0 1×1

f(X)∗,i0 n×1

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

i0=1

j0=1

d

n

(W · Xi0,∗)⊤ 1×d

)f(X)∗,i0

s(X)i0,j0 1×1

Gi(i0,j0) 1×1

(

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

j0=1

i0=1

n×1

n

(Gi(i0,∗)⊤

(W · Xi0,∗)⊤ 1×d

s(X)i0,∗ d×1

)f(X)∗,i0

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

i0=1

1×d

n×1

n

W⊤ d×d

(Gi(i0,∗)⊤

Xi⊤0,∗ 1×d

s(X)i0,∗ d×1

)f(X)∗,i0

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

i0=1

[Figure 333]

[Figure 334]

1×d

n×1

- where the 1st step is from the choice of B6(X), the 2nd step comes from basic algebra, the 3rd step

is because of a⊤b = di=1 ai · bi holds for any a,b ∈ Rd, the 4th step is due to (AB)⊤ = B⊤A⊤ for any matrices A and B.

= (Gi(i0,∗)⊤

s(X)i0,∗ d×1

)f(X)∗,i0

.

Recall that we have z6(X)∗,i0

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

1×d

n×1

n×1

Then, we have

n

W⊤ d×d

(Gi(i0,∗)⊤

Xi⊤0,∗ 1×d

)f(X)∗,i0

−

s(X)i0,∗ d×1

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

i0=1

[Figure 349]

[Figure 350]

1×d

n×1

n

W⊤ d×d

Xi⊤0,∗ 1×d

z6(X)∗,i0 n×1

= −

[Figure 351]

[Figure 352]

i0=1

[Figure 353]

[Figure 354]

W⊤ d×d

= − z6(X)

#### X

[Figure 355]

[Figure 356]

n×d

n×n

- where the 1st step is from the choice of z6(X), the 2nd step comes from basic linear algebra. Then, we can get the matrix view of B7(X) term.

- Lemma D.11 (Matrix view of B7(X) term). If we have the below conditions,

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

- • Let B7(X) n×d

[Figure 361]

[Figure 362]

= (f(X)∗,i0 ⊙ h(X)∗,j0)

[Figure 363]

[Figure 364]

n×1

·(W · Xi0,∗)⊤

[Figure 365]

[Figure 366]

1×d

be deﬁned in Lemma D.4.

- • We deﬁne z7(X) ∈ Rn×n, which satisﬁes z7(X)∗,i0

[Figure 367]

[Figure 368]

n×1

= f(X)∗,i0

[Figure 369]

[Figure 370]

n×1

⊙(h(X)

[Figure 371]

[Figure 372]

n×d

Gi(i0,∗) d×1

[Figure 373]

[Figure 374]

).

- • Let X ∈ Rn×d,W ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 375]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi. Then we have

n

d

W⊤ d×d

Gi(i0,j0) 1×1

= z7(X)

#### X

B7(X) n×d

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

i0=1

j0=1

n×d

n×n

Proof. We have

n

d

=

B7(X) n×d

Gi(i0,j0) 1×1

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

i0=1

j0=1

=

=

n

d

·(W · Xi0,∗)⊤

Gi(i0,j0) 1×1

(f(X)∗,i0 ⊙ h(X)∗,j0) n×1

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

i0=1

j0=1

1×d

n

d

)) · (W · Xi0,∗)⊤

(f(X)∗,i0

Gi(i0,j0) 1×1

⊙(

h(X)∗,j0 n×1

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

i0=1

j0=1

n×1

1×d

n

)) · (Xi⊤0,∗W⊤)

(f(X)∗,i0

⊙(h(X)

Gi(i0,∗) d×1

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

i0=1

[Figure 406]

[Figure 407]

n×1

n×d

1×d

- where the 1st step is from the choice of B7(X), the 2nd step comes from basic algebra, the 3rd step is because of basic linear algebra.

= f(X)∗,i0

⊙(h(X)

Gi(i0,∗) d×1

).

Recall that we have z7(X)∗,i0

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

n×d

n×1

n×1

Then we have

n

)) · (Xi⊤0,∗W⊤)

Gi(i0,∗) d×1

(f(X)∗,i0

⊙(h(X)

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

i0=1

[Figure 422]

[Figure 423]

n×d

n×1

1×d

n

W⊤ d×d

Xi⊤0,∗ 1×d

z7(X)∗,i0 n×1

=

[Figure 424]

[Figure 425]

i0=1

[Figure 426]

[Figure 427]

W⊤ d×d

= z7(X)

#### X

[Figure 428]

[Figure 429]

n×d

n×n

- where the 1st step is from the choice of z7(X), the 2nd step comes from basic linear algebra. Then, we consider B8(X).

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

- Lemma D.12 (Matrix view of B8(X) term). If we have the below conditions,

- • Let B8(X) n×d

[Figure 434]

[Figure 435]

= f(X)∗,i0

[Figure 436]

[Figure 437]

n×1

(WV )⊤∗,j0 1×d

[Figure 438]

[Figure 439]

be deﬁned in Lemma D.5.

- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 440]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi. Then we have

d

n

WV⊤ d×d

Gi(i0,j0) 1×1

= f(X)

Gi n×d

B8(X) n×d

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

j0=1

i0=1

n×n

Proof. We have

d

d

n

n

(WV )⊤∗,j0 1×d

Gi(i0,j0) 1×1

Gi(i0,j0) 1×1

=

B8(X) n×d

f(X)∗,i0 n×1

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

j0=1

j0=1

i0=1

i0=1

[Figure 455]

[Figure 456]

n

d

(WV )⊤∗,j0 1×d

f(X)∗,i0 n×1

=

)

Gi(i0,j0) 1×1

(

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

i0=1

j0=1

[Figure 461]

[Figure 462]

n

Gi(i0,∗)⊤ 1×d

WV⊤ d×d

f(X)∗,i0 n×1

=

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

i0=1

WV⊤ d×d

Gi n×d

= f(X)

[Figure 467]

[Figure 468]

n×n

- where the 1st step is from the choice of B8(X), the 2nd step comes from basic algebra, the 3rd step is because of basic linear algebra, the 4th step is due to basic linear algebra.

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

Now, we can do the matrix view of B2(X) term.

- Lemma D.13 (Matrix view of B2(X) term). If we have the below conditions,

- • Let B2(X) d×1

[Figure 473]

[Figure 474]

= −s(X)i0,j0

[Figure 475]

[Figure 476]

1×1

W⊤ d×d

X⊤ d×n

f(X)i0,∗ n×1

[Figure 477]

[Figure 478]

be deﬁned in Lemma D.6

- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 479]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.
- • We deﬁne z2(X) ∈ Rn×n, which satisﬁes z2(X)i0,∗

[Figure 480]

[Figure 481]

n×1

= (Gi(i0,∗)⊤

[Figure 482]

[Figure 483]

1×d

s(X)i0,∗ d×1

[Figure 484]

[Figure 485]

)f(X)i0,∗

[Figure 486]

[Figure 487]

n×1

- • Let X ∈ Rn×d,W ∈ Rd×d be deﬁned in Deﬁnition C.3

Then we have

n

d

Gi(i0,j0) 1×1

[Figure 488]

[Figure 489]

i0=1

j0=1

B2(X)⊤ 1×d

= −z2(X)

#### X

ei0 n×1

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

n×d

n×n

#### W

d×d

Proof. We have

d

n

B2(X)⊤ 1×d

Gi(i0,j0) 1×1

ei0 n×1

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

j0=1

i0=1

d

n

f(X)⊤i0,∗ 1×n

X

#### W

Gi(i0,j0) 1×1

ei0 n×1

= −

s(X)i0,j0 1×1

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

j0=1

i0=1

[Figure 502]

[Figure 503]

n×d

d×d

d

n

f(X)⊤i0,∗ 1×n

Gi(i0,j0) 1×1

(

= −

s(X)i0,j0 1×1

X

#### W

) ei0

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

j0=1

i0=1

[Figure 508]

[Figure 509]

n×d

d×d

n×1

n

(Gi(i0,∗)⊤

f(X)⊤i0,∗ 1×n

X

#### W

) ei0

= −

s(X)i0,∗ d×1

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

i0=1

[Figure 514]

[Figure 515]

n×d

d×d

n×1

1×d

n

(Gi(i0,∗)⊤

)f(X)⊤i0,∗

ei0 n×1

= −

#### X

#### W

s(X)i0,∗ d×1

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

i0=1

[Figure 520]

[Figure 521]

n×d

d×d

1×d

1×n

where the 1st step is from the choice of B2(X), the 2nd step comes from basic algebra, the 3rd step is because of a⊤b = di=1 ai · bi holds for any a,b ∈ Rd, the 4th step is due to (AB)⊤ = B⊤A⊤ holds for any matrix A,B.

= (Gi(i0,∗)⊤

Recall that we have z2(X)i0,∗

)f(X)i0,∗

.

s(X)i0,∗ d×1

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

n×1

n×1

1×d

Then, we have

n

)f(X)⊤i0,∗

(Gi(i0,∗)⊤

#### X

s(X)i0,∗ d×1

ei0 n×1

−

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

i0=1

[Figure 534]

[Figure 535]

n×d

1×d

1×n

#### W

d×d

n

z2(X)⊤i0,∗ 1×n

X

ei0 n×1

= −

i0=1

[Figure 536]

[Figure 537]

n×d

= − z2(X)

#### X

[Figure 538]

[Figure 539]

n×d

n×n

#### W

d×d

#### W

d×d

where the 1st step is from the choice of z2(X), the 2nd step comes from basic linear algebra.

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

Finally, we do a similar analysis for the term B4(X). Then, we get all the matrix views we need.

- Lemma D.14 (Matrix view of B4(X) term). If we have the below conditions,

- • Let B4(X) d×1

[Figure 544]

[Figure 545]

= W⊤ d×d

X⊤ d×n

(f(X)i0,∗ ⊙ h(X)∗,j0) n×1

[Figure 546]

[Figure 547]

be deﬁned in Lemma D.7.

- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 548]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.
- • We deﬁne z4(X) ∈ Rn×n, which satisﬁes

= f(X)i0,∗

⊙(h(X)Gi(i0,∗))

z4(X)i0,∗ n×1

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

n×1

n×1

Then we have

Proof. We have

d

n

Gi(i0,j0) 1×1

[Figure 555]

[Figure 556]

j0=1

i0=1

B4(X)⊤ 1×d

= z4(X)

#### X

ei0 n×1

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

n×d

n×n

#### W

d×d

n

d

j0=1

i0=1

=

n

d

i0=1

j0=1

=

=

=

n

ei0 n×1

i0=1

n

ei0 n×1

i0=1

n

ei0 n×1

i0=1

= z4(X)

[Figure 561]

[Figure 562]

n×n

B4(X)⊤ 1×d

Gi(i0,j0) 1×1

ei0 n×1

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

Gi(i0,j0) 1×1

[Figure 567]

[Figure 568]

(f(X)⊤i0,∗

⊙(

[Figure 569]

[Figure 570]

1×n

(f(X)⊤i0,∗ ⊙ h(X)⊤∗,j0) 1×n

ei0 n×1

X

#### W

[Figure 571]

[Figure 572]

n×d

d×d

d

h(X)⊤∗,j0 1×n

Gi(i0,j0) 1×1

)) X

#### W

[Figure 573]

[Figure 574]

j0=1

[Figure 575]

[Figure 576]

n×d

d×d

⊙(h(X)Gi(i0,∗))⊤

(f(X)⊤i0,∗

#### ) X

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

n×d

1×n

1×n

#### W

d×d

z4(X)⊤i0,∗ 1×n

X

[Figure 581]

[Figure 582]

n×d

#### X

#### W

#### W

d×d

n×d

d×d

where the 1st step is from the choice of B4(X), the 2nd step comes from basic algebra, the 3rd step is because of basic linear algebra, the 4th step is due to the choice of z4(X), the 5th step follows from basic linear algebra.

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

### D.6 Components of gradient on Ti(X)

- Deﬁnition D.15 (Deﬁnition of Dk). If we have the below conditions,

- • For k1 ∈ {6,7,8}, let Bk1(X) ∈ Rn×d be deﬁned as Lemma D.3, D.4, and D.5, respectively.
- • For k2 ∈ {2,4}, let Bk2(X) ∈ Rd×1 be deﬁned as Lemma D.6 and D.7, respectively.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X) i(Ti−1(X)). We deﬁne Dk ∈ Rn×d as follows:
- • For k1 ∈ {6,7,8}, we deﬁne

Dk1 :=

n

i0=1

d

j0=1

Gi(i0,j0) 1×1

[Figure 587]

[Figure 588]

Bk1(X) n×d

[Figure 589]

[Figure 590]

- • For k2 ∈ {2,4}, we deﬁne

[Figure 591]

Dk2 :=

n

i0=1

d

j0=1

Gi(i0,j0) 1×1

[Figure 592]

[Figure 593]

ei0 n×1

Bk2(X)⊤ 1×d

[Figure 594]

[Figure 595]

- Deﬁnition D.16 (Deﬁnition of K). If we have the below conditions,

- • Let s(X) ∈ Rn×d be deﬁned as Deﬁnition C.10.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up

to the function gi, i.e., Gi = dAttndL(X)

i(Ti−1(X)). We deﬁne K ∈ Rn, where for each i0 ∈ [n], we deﬁne

[Figure 596]

= Gi(i0,∗)⊤

Ki0 1×1

s(X)i0,∗ d×1

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

1×d

Furthermore, we have

= (Gi ⊙ s(X))

K

1d d×1

[Figure 601]

[Figure 602]

n×1

n×d

- Lemma D.17 (Close form of Dk). If we have the below conditions,

- • Let X ∈ Rn×d,W ∈ Rd×d be deﬁned as Deﬁnition C.3.
- • For k ∈ {6,7,8,2,4}, let Dk ∈ Rn×d be deﬁned as Deﬁnition D.15.
- • For k3 ∈ {6,7,2,4}, let zk3(X) ∈ Rn×n be deﬁned as Lemma D.10, D.11, D.13, and D.14, respectively.

- • Let K ∈ Rn be deﬁned as Deﬁnition D.16.
- • We deﬁne z6(X) ∈ Rn×n, which satisﬁes z6(X)

[Figure 603]

[Figure 604]

n×n

= f(X)

[Figure 605]

[Figure 606]

n×n

diag(K)

[Figure 607]

[Figure 608]

n×n

.

- • We deﬁne z7(X) ∈ Rn×n, which satisﬁes z7(X)

[Figure 609]

[Figure 610]

n×n

= f(X)

[Figure 611]

[Figure 612]

n×n

⊙(h(X)

[Figure 613]

[Figure 614]

n×d

G⊤i d×n

)

- • We deﬁne z2(X) ∈ Rn×n, which satisﬁes z2(X)

[Figure 615]

[Figure 616]

n×n

= diag(K)

[Figure 617]

[Figure 618]

n×n

f(X)

[Figure 619]

[Figure 620]

n×n

- • We deﬁne z4(X) ∈ Rn×n, which satisﬁes

z4(X) n×n

= f(X)

⊙( Gi

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

n×n

h(X)⊤ d×n

)

[Figure 625]

[Figure 626]

n×d

Then, we can show that the close forms of Dk can be written as follows:

- • D6 = −z6(X) n×n

[Figure 627]

[Figure 628]

X

n×d

W⊤ d×d

.

- • D7 = z7(X) n×n

[Figure 629]

[Figure 630]

X

n×d

W⊤ d×d

.

- • D8 = f(X) n×n

[Figure 631]

[Figure 632]

Gi n×d

WV⊤ d×d

.

- • D2 = −z2(X) n×n

[Figure 633]

[Figure 634]

X

n×d

W

d×d

.

- • D4 = z4(X) n×n

X

[Figure 635]

[Figure 636]

n×d

#### W

.

d×d

Proof. We ﬁnish the proof by parts.

- • By Lemma D.10, we have the close form of D6.
- • By Lemma D.11, we have the close form of D7.
- • By Lemma D.12, we have the close form of D8.
- • By Lemma D.13, we have the close form of D2.
- • By Lemma D.14, we have the close form of D4.

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

## E Fast Computation for Gradient on T(X)

In this section, we give an almost linear time n1+o(1) algorithm for each Bi(X) term. Namely, we consider B6(X),B7(X),B8(X),B2(X),B4(X) in Section E.1, E.2, E.3, E.4, and E.5, respectively.

### E.1 Fast computation for B6(X) term

Before we introduce the almost linear time algorithm for B6(X) term, we need to introduce the accelerated algorithm for the key component term, z6(X), in Lemma E.2.

We ﬁrst compute K, which is deﬁned in Deﬁnition D.16

- Lemma E.1 (Computation time for K). If we have the below conditions,

• Let K ∈ Rn be deﬁned as Deﬁnition D.16. Then, we can show that K can be computed in O(n · d) time.

Proof. Since for each i0 ∈ [n], we have

= Gi(i0,∗)⊤

s(X)i0,∗ d×1

Ki0 1×1

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

1×d

Then, we have that it takes O(d) time for calculating each entry. Since there are total n entries in K, the overall computation time for K is O(n · d).

We now compute z6(X).

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

- Lemma E.2 (Fast computation for z6(X)). If we have the below conditions,

- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 649]

i(Ti−1(X)).

- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let z6(X) ∈ Rn×n be deﬁned in Lemma D.10. Then, for some k6 = no(1), there are matrices U6,V6 ∈ Rn×k6 such that U6V6⊤ − z6(X) ∞ ≤

- ǫ/poly(n). The matrices U6,V6 can be constructed in n1+o(1) time. Proof. Recall in Lemma D.10, we have deﬁne z6(X) satisfying the following equation

= (Gi(i0,∗)⊤

)f(X)∗,i0

s(X)i0,∗ d×1

z6(X)∗,i0 n×1

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

1×d

n×1

(3)

Recall that K ∈ Rn has been deﬁned in Deﬁnition D.16. By Lemma E.1, we have K can be computed in O(n · d) time.

We also have

z6(X) n×n

= f(X)

diag(K)

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

n×n

n×n

By Lemma C.13, we have U1,V1 ∈ Rn×k1 such that U1V1⊤ − f(X) ∞ ≤ ǫ/poly(n)

Let U6 = U1, V6 = diag(K)V1. We have V6 = diag(K)

can be computed in nk1 time.

V1 n×k1

[Figure 664]

[Figure 665]

n×n

The overall running time for constructing U6 and V6 is n1+o(1). Then, we consider the error bound. We have

U6V6⊤ − z6(X) ∞ = U1V1⊤ diag(K) − f(X)diag(K) ∞ ≤ n U1V1⊤ − f(X) ∞ diag(K) ∞ ≤ n(ǫ/poly(n)) diag(K) ∞ ≤ ǫ/poly(n)

where the 1st step is from the choice of U6, V6, the 2nd step comes from basic linear algebra, the 3rd step is because of Lemma C.13, the 4th step is due to diag(K) ∞ ≤ poly(n).

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

Then, we are ready to introduce the almost linear time algorithm for B6(X) term.

- Lemma E.3 (Fast computation for B6(X) term). If we have the below conditions,

- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let B6(X) ∈ Rn×n be deﬁned in Lemma D.3.
- • We deﬁne D6 ∈ Rn×d, where D6 := ni

0=1

d j0=1 Gi(i0,j0)B6(X).

- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 670]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.

Then, we can show that, there is an algorithm to approximate D6 in n1+o(1) time, and it can achieve ǫ/poly(n) accuracy.

Namely, the algorithm output D6 satisfying D6 − D6 ∞ ≤ ǫ/poly(n)

- Proof. Recall that in Lemma D.10, we have deﬁned z6(X) ∈ Rn×n, which satisﬁes

= (Gi(i0,∗)⊤

z6(X)∗,i0 n×1

)f(X)∗,i0

s(X)i0,∗ d×1

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

1×d

n×1

And, in that Lemma, we also have

d

n

W⊤ d×d

Gi(i0,j0) 1×1

= −z6(X)

#### X

B6(X) n×d

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

j0=1

i0=1

n×d

n×n

Let U6,V6 ∈ Rn×k6 be deﬁned as Lemma E.2. Let z6(X) = U6V6⊤. By Lemma E.2, we have

z6(X) − z6(X) ∞ ≤ ǫ/poly(n) (4)

Proof of running time. We compute in the following way:

- • Compute V6⊤ k6×n

X

n×d

, which takes n1+o(1) time.

- • Compute V6⊤X k6×d

[Figure 685]

[Figure 686]

W⊤ d×d

, which takes n1+o(1) time.

- • Compute U6 n×k6

V6⊤XW⊤ k6×d

, which takes n1+o(1) time.

[Figure 687]

[Figure 688]

Therefore, the overall running time is n1+o(1). Proof of error bound. We have

z6(X)XW⊤ − z6(X)XW⊤ ∞ ≤ d · n z6(X) − z6(X) ∞ X ∞ W ∞ ≤ d · n(ǫ/poly(n)) X ∞ W ∞ ≤ ǫ/poly(n)

where the 1st step is from basic linear algebra, the 2nd step comes from Eq.(4), the 3rd step is because of W ∞ ≤ poly(n) and X ∞ ≤ poly(n).

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

### E.2 Fast computation for B7(X) term

Similar to the analysis process of B6(X) term, we ﬁrst provide the almost linear time algorithm for z7(X), then provide that algorithm for B7(X).

- Lemma E.4 (Fast computation for z7(X)). If we have the below conditions,

- • Let z7(X) ∈ Rn×n be deﬁned in Lemma D.11.
- • By Lemma C.13, let U1,V1 be the low rank approximation of f(X), such that U1V1⊤ − f(X) ∞ ≤ ǫ/poly(n).
- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up

to the function gi, i.e., Gi = dAttndL(X)

i(Ti−1(X)).

[Figure 693]

Then, for some k7 = no(1), there are matrices U7,V7 ∈ Rn×k7 such that U7V7⊤ − z7(X) ∞ ≤

- ǫ/poly(n). The matrices U7,V7 can be constructed in n1+o(1) time.

- Proof. Recall that in Lemma D.11, we have deﬁned z7(X) ∈ Rn×n, where the i0-th column of z7(X) satisﬁes

which is equivalent to

= f(X)∗,i0

⊙(h(X)

z7(X)∗,i0 n×1

Gi(i0,∗) d×1

)

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

n×1

n×d

G⊤i d×n

)

= f(X)

⊙(h(X)

z7(X) n×n

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

n×n

n×d

By Lemma C.13, we know f(X) := U1V1⊤ is a good approximation for f(X). We choose U7 = U1 ⊘ h(X) and V7 = V1 ⊘ Gi, where U7,V7 ∈ Rn×k1d. Proof of running time. For U7 = U1 ⊘ h(X), since U1 ∈ Rn×k1,h(X) ∈ Rn×d, constructing U7 takes O(ndk1) =

O(n1+o(1)) time. Similarly, constructing V7 takes O(n1+o(1)) time. Proof of error bound. Using Fact C.2, we have

U7V7⊤ − z7(X) ∞ = U7V7⊤ − f(X) ⊙ (h(X)G⊤i ) ∞

= (U1 ⊘ h(X))(V1 ⊘ Gi)⊤ − f(X) ⊙ (h(X)G⊤i ) ∞

= (U1V1⊤) ⊙ (h(X)G⊤i ) − f(X) ⊙ (h(X)G⊤i ) ∞

= f(X) ⊙ (h(X)G⊤i ) − f(X) ⊙ (h(X)G⊤i ) ∞ ≤ d h(X) ∞ Gi ∞ · ǫ/poly(n) ≤ ǫ/poly(n) (5)

where the 1st step is from the deﬁnition of z7(X), the 2nd step comes from the choice of U7 and V7, the 3rd step is because of Fact C.2, the 4th step is due to the deﬁnition of f(X), the 5th step follows from f(X)−f(X) ∞ ≤ ǫ/poly(n), the sixth step follows from Lemma C.18 and Gi ∞ ≤ poly(n).

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

Then, we can do similarly fast computation for B7 term.

- Lemma E.5 (Fast computation for B7(X) term). If we have the below conditions,

- • Let B7(X) ∈ Rn×d be deﬁned in Lemma D.4.
- • We deﬁne D7 ∈ Rn×d, where D7 := ni

0=1

d j0=1 Gi(i0,j0)B7(X).

- • Let X ∈ Rn×d,W,WV ∈ Rd×d,B ∈ Rn×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up

to the function gi, i.e., Gi = dAttndL(X)

i(Ti−1(X)).

[Figure 712]

Then, we can show that, there is an algorithm to approximate D7 in n1+o(1) time, and it can achieve ǫ/poly(n) accuracy.

Namely, the algorithm output D7 satisﬁes

D7 − D7 ∞ ≤ ǫ/poly(n) Proof. In Lemma D.11, we have

d

n

Gi(i0,j0) 1×1

B7(X) n×d

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

j0=1

i0=1

Let U7,V7 ∈ Rn×k7 be deﬁned in Lemma E.4. Let z7(X) := U7V7⊤. By Lemma E.4, we have

W⊤ d×d

= z7(X)

#### X

[Figure 717]

[Figure 718]

n×d

n×n

z7(X) − z7(X) ∞ ≤ ǫ/poly(n) (6)

Proof of running time. We compute in the following way:

- • Compute V7⊤ k7×n

X

n×d

, which takes n1+o(1) time.

- • Compute V7⊤X k7×d

[Figure 719]

[Figure 720]

W⊤ d×d

, which takes n1+o(1) time.

- • Compute U7 n×k7

V7⊤XW⊤ k7×d

, which takes n1+o(1) time.

[Figure 721]

[Figure 722]

Therefore, the overall running time is n1+o(1). Proof of error bound. We have

z7(X)XW⊤ − z7(X)XW⊤ ∞ ≤ d · n z7(X) − z7(X) ∞ X ∞ W ∞ ≤ d · n(ǫ/poly(n)) X ∞ W ∞ ≤ ǫ/poly(n)

where the 1st step is from basic linear algebra, the 2nd step comes from Eq. (6), the 3rd step is because of W ∞ ≤ poly(n) and X ∞ ≤ poly(n).

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

- E.3 Fast computation for B8(X) term Then, we can do fast computations on B8(X) term.

- Lemma E.6 (Fast computation for B8(X) term). If we have the below conditions,

- • Let B8(X) ∈ Rn×d be deﬁned in Lemma D.5.

- • We deﬁne D8 ∈ Rn×d, where D8 := ni

0=1

d j0=1 Gi(i0,j0)B8(X).

- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 727]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.

Then, we can show that, there is an algorithm to approximate D8 in n1+o(1) time, and it can achieve ǫ/poly(n) accuracy.

Namely, the algorithm output D8 satisﬁes D8 − D8 ∞ ≤ ǫ/poly(n)

- Proof. Recall that in Lemma D.12, we have n

d

WV⊤ d×d

= f(X)

Gi n×d

Gi(i0,j0) 1×1

B8(X) n×d

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

j0=1

i0=1

n×n

Let f(X) := U1V1⊤ denote the approximation of f(X). By Lemma C.13, we have

f(X) − f(X) ∞ ≤ ǫ/poly(n) (7)

Proof of running time. We compute in the following way:

- • Compute V1⊤ k1×n

Gi n×d

, which takes n1+o(1) time.

- • Compute V1⊤Gi k1×d

[Figure 734]

[Figure 735]

WV⊤ d×d

, which takes n1+o(1) time.

- • Compute U1 n×k1

V1⊤GiWV⊤ k1×d

, which takes n1+o(1) time.

[Figure 736]

[Figure 737]

Therefore, the overall running time is n1+o(1). Proof of error bound. We have

f(X)GiWV⊤ − f(X)GiWV⊤ ∞ ≤ d · n f(X) − f(X) ∞ Gi ∞ WV ∞ ≤ d · n(ǫ/poly(n)) Gi ∞ WV ∞ ≤ ǫ/poly(n)

where the 1st step is from basic linear algebra, the 2nd step comes from Eq.(7), the 3rd step is because of Gi ∞ ≤ poly(n) and WV ∞ ≤ poly(n).

- E.4 Fast computation for B2(X) term Then, we provide the proof of how to do fast computation on B2(X).

- Lemma E.7 (Fast computation for z2(X)). If we have the below conditions,

- • Let z2(X) ∈ Rn×n be deﬁned as in Lemma D.13.
- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 742]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.

Then, for some k9 = no(1), there are matrices U9,V9 ∈ Rn×k9 such that U9V9⊤ − z2(X) ∞ ≤ ǫ/poly(n). The matrices U9,V9 can be constructed in n1+o(1) time.

- Proof. Recall that in Lemma D.13, we have deﬁned z2(X) ∈ Rn×n, where the i0-th row of z2(X) satisﬁes

z2(X)i0,∗ n×1

[Figure 743]

[Figure 744]

= (Gi(i0,∗)⊤

[Figure 745]

[Figure 746]

1×d

s(X)i0,∗ d×1

[Figure 747]

[Figure 748]

)f(X)i0,∗

[Figure 749]

[Figure 750]

n×1

Recall that K ∈ Rn has been deﬁned in Deﬁnition D.16. By Lemma E.1, we have K can be computed in O(n · d) time. We also have

z2(X) n×n

[Figure 751]

[Figure 752]

= diag(K)

[Figure 753]

[Figure 754]

n×n

f(X)

[Figure 755]

[Figure 756]

n×n

By Lemma C.13, let U1,V1 be the low rank approximation of f(X), such that U1V1⊤ −

f(X) ∞ ≤ ǫ/poly(n). Let U9 = diag(K)U1, V6 = V1. We have U9 = diag(K)

[Figure 757]

[Figure 758]

n×n

U1 n×k1

can be computed in nk1 time.

The overall running time for constructing U9 and V9 is n1+o(1). Then, we consider the error bound. We have

U9V9⊤ − z2(X) ∞ = diag(K)U1V1⊤ − diag(K)f(X) ∞ ≤ n U1V1⊤ − f(X) ∞ diag(K) ∞ ≤ n(ǫ/poly(n)) diag(K) ∞ ≤ ǫ/poly(n) (8)

where the 1st step is from the choice of U6, V6, the 2nd step comes from basic linear algebra, the 3rd step is because of Lemma C.13, the 4th step is due to diag(K) ∞ ≤ poly(n).

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

- Lemma E.8 (Fast computation for B2(X) term). If we have the below conditions,

- • Let B2(X) ∈ Rn×d be deﬁned in Lemma D.6.
- • We deﬁne D2 ∈ Rn×d, where D2 := ni

0=1

d j0=1 Gi(i0,j0)

[Figure 763]

[Figure 764]

1×1

ei0 n×1

B2(X)⊤ 1×d

[Figure 765]

[Figure 766]

.

- • Let X ∈ Rd×n,W,WV ∈ Rd×d,B ∈ Rn×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,B,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 767]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.

Then, we can show that, there is an algorithm to approximate D2 in n1+o(1) time, and it can achieve ǫ/poly(n) accuracy.

Namely, the algorithm output D2 satisﬁes D2 − D2 ∞ ≤ ǫ/poly(n)

- Proof. In Lemma D.13, we have n

d

B2(X)⊤ 1×d

Gi(i0,j0) 1×1

= −z2(X)

#### X

#### W

ei0 n×1

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

j0=1

i0=1

n×d

d×d

n×n

Let U9,V9 ∈ Rn×k9 be deﬁned in Lemma E.7. Let z2(X) := U9V9⊤. By Lemma E.7, we have

z2(X) − z2(X) ∞ ≤ ǫ/poly(n) (9)

Proof of running time. We compute in the following way:

- • Compute V9⊤ k9×n

X

n×d

, which takes n1+o(1) time.

- • Compute V9⊤X k9×d

[Figure 774]

[Figure 775]

W

d×d

, which takes n1+o(1) time.

- • Compute U9 n×k9

V9⊤XW k9×d

, which takes n1+o(1) time.

[Figure 776]

[Figure 777]

Therefore, the overall running time is n1+o(1). Proof of error bound. We have

z2(X)XW − z2(X)XW ∞ ≤ d · n z2(X) − z2(X) ∞ X ∞ W ∞ ≤ d · n(ǫ/poly(n)) X ∞ W ∞ ≤ ǫ/poly(n)

- where the 1st step is from basic linear algebra, the 2nd step comes from Eq.(9), the 3rd step is because of W ∞ ≤ poly(n) and X ∞ ≤ poly(n).

### E.5 Fast computation for B4(X) term

Finally, our analysis shows that we can do fast computations for B4(X) term. After that, we showed that all terms can be computed quickly.

- Lemma E.9 (Fast computation for z4(X)). If we have the below conditions,

- • Let z4(X) ∈ Rn×n be deﬁned in Lemma D.14.
- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 782]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.

Then, for some k10 = no(1), there are matrices U10,V10 ∈ Rn×k10, let z4(X) := U10V10⊤, such that z4(X) − z4(X) ∞ ≤ ǫ/poly(n). The matrices U10,V10 can be constructed in n1+o(1) time.

- Proof. In Lemma D.14, we have deﬁned z4(X) ∈ Rn×n, where the i0-th column of z4(X) satisﬁes

= (f(X)i0,∗

⊙(h(X)Gi(i0,∗))

z4(X)i0,∗ n×1

)

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

n×1

n×1

which is equivalent to

h(X)⊤ d×n

= (f(X)

⊙ Gi

)

z4(X) n×n

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

n×d

n×n

By Lemma C.13, let U1,V1 be the low rank approximation of f(X), such that U1V1⊤ −

f(X) ∞ ≤ ǫ/poly(n). We choose U10 = U1 ⊘ Gi and V10 = V1 ⊘ h(X), where U10,V10 ∈ Rn×k1d. Proof of running time. For U10 = U1 ⊘ Gi, since U1 ∈ Rn×k1,Gi ∈ Rn×d, constructing U10 takes O(ndk1) = O(n1+o(1))

time. Similarly, constructing V10 takes O(n1+o(1)) time. Proof of error bound.

Let f(X) := U1V1⊤. Using Fact C.2, we have

z4(X) − z4(X) ∞

= U10V10⊤ − f(X) ⊙ (Gi · h(X)⊤) ∞

= (U1 ⊘ Gi)(V1 ⊘ h(X))⊤ − f(X) ⊙ (Gi · h(X)⊤) ∞ = (U1V1⊤) ⊙ (Gi · h(X)⊤) − f(X) ⊙ (Gi · h(X)⊤) ∞

where the 1st step is from the deﬁnition of z4(X),z4(X), the 2nd step comes from the choice of U10 and V10, the 3rd step is because of Fact C.2.

(U1V1⊤) ⊙ (Gi · h(X)⊤) − f(X) ⊙ (Gi · h(X)⊤) ∞

= U1V1⊤ − f(X) ∞ Gi · h(X)⊤ ∞ ≤ d · (ǫ/poly(n)) h(X) ∞ Gi ∞ ≤ ǫ/poly(n)

where the 1st step is from basic linear algebra, the 2nd step comes from U1V1 − f(X) ∞ ≤ ǫ/poly(n), the 3rd step is because of Lemma C.18 and Gi ∞ ≤ poly(n).

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

- Lemma E.10 (Fast computation for B4(X) term). If we have the below conditions,

- • Let B4(X) ∈ Rn×d be deﬁned in Lemma D.7.
- • We deﬁne D4 ∈ Rn×d, where D4 := ni

0=1

d j0=1 Gi(i0,j0)

[Figure 799]

[Figure 800]

1×1

ei0 n×1

B4(X)⊤ 1×d

[Figure 801]

[Figure 802]

.

- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 803]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.

Then, we can show that, there is an algorithm to approximate D4 in n1+o(1) time, and it can achieve ǫ/poly(n) accuracy.

Namely, the algorithm output D4 satisﬁes

D4 − D4 ∞ ≤ ǫ/poly(n) Proof. In Lemma D.14, we have

d

n

Gi(i0,j0) 1×1

[Figure 804]

[Figure 805]

j0=1

i0=1

B4(X)⊤ 1×d

= z4(X)

#### X

ei0 n×1

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

n×d

n×n

#### W

d×d

Let z4(X) := U10V10⊤. By Lemma E.9, we have

z4(X) − z4(X) ∞ ≤ ǫ/poly(n) (10)

Proof of running time. We compute in the following way:

- • Compute V10⊤ k10×n

X

n×d

, which takes n1+o(1) time.

- • Compute V10⊤X k10×d

[Figure 810]

[Figure 811]

W

d×d

, which takes n1+o(1) time.

- • Compute U10 n×k10

V10⊤XW k10×d

, which takes n1+o(1) time.

[Figure 812]

[Figure 813]

Therefore, the overall running time is n1+o(1). Proof of error bound. We have

z4(X)XW − z4(X)XW ∞ ≤ d · n z4(X) − z4(X) ∞ X ∞ W ∞ ≤ d · n(ǫ/poly(n)) X ∞ W ∞ ≤ ǫ/poly(n)

- where the 1st step is from basic linear algebra, the 2nd step comes from Eq.(10), the 3rd step is because of W ∞ ≤ poly(n) and X ∞ ≤ poly(n).

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

### E.6 Putting everything together

After we have analyzed each Bi(X) term in the previous section, we put them together in this section, to analyze the overall running time and error bound of the gradient of L(X) on Ti(X) in

- Lemma E.11.

- Lemma E.11 (Fast computation for dTdL(X)

i−1(X), formal version of Lemma 5.1). If we have the below conditions,

[Figure 818]

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let m denote the number of self-attention transformer model (see Deﬁnition 1.3).
- • For any i ∈ [m], let Ti(X) be deﬁned as Deﬁnition 3.3.
- • Let X ∈ Rn×d,W,WV ∈ Rd×d be deﬁned in Deﬁnition C.3.
- • Assuming each entry of X,W,WV ,Gi can be re represented using O(log(n)) bits.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 819]

i(Ti−1(X)).

- • Assume Gi can be computed in n1+o(1) time.

We can show that dTdL(X)

i−1(X) can be approximated in n1+o(1) time, with 1/poly(n) approximation error. Namely, our algorithm can output gt in n1+o(1) time, which satisﬁes

[Figure 820]

dL(X)

gt −

dTi−1(X) ∞ ≤ 1/poly(n) Proof. By Lemma D.9, we have

[Figure 821]

dL(X) dTi−1(X)

=

[Figure 822]

d

n

Gi(i0,j0) 1×1

+ ei0

·(B6(X) + B7(X) + B8(X)

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

j0=1

i0=1

n×d

(B2(X) + B4(X))⊤ 1×d

)

[Figure 827]

[Figure 828]

n×1

=

Di

i∈{2,4,6,7,8}

where the 1st step is from Lemma D.9, the 2nd step comes from the deﬁnition of D6,D7,D8,D2,D4.

Then, by Lemma E.3, E.5, E.6, E.8, E.10, we have D6,D7,D8,D2,D4 ∈ Rn×d can be approximated in n1+o(1) time, with up to ǫ/poly(n) error.

Namely, for i ∈ {2,4,6,7,8}, let Di ∈ Rn×d denote the approximated version of D, we have Di − D ∞ ≤ ǫ/poly(n)

Let gt = i∈{2,4,6,7,8} Di. Proof of running time.

The running time for gt = i∈{2,4,6,7,8} Di is 5nd. Therefore, the overall running time for computing gt is n1+o(1). Proof of error bound. We have

dL(X) dTi−1(X) ∞

=

gt −

[Figure 829]

( Di − Di) ∞

i∈{2,4,6,7,8}

≤

i∈{2,4,6,7,8}

( Di − Di) ∞

≤ ǫ/poly(n)

where the 1st step is from the deﬁnition of gt and dTdL(X)

i−1(X), the 2nd step comes from basic algebra, the 3rd step is because of Di − D ∞ ≤ ǫ/poly(n).

[Figure 830]

Then, choose ǫ = 1/poly(n), we have

dL(X) dTi−1(X) ∞ ≤ 1/poly(n)

gt −

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

## F Fast Computation for Gradient on W

- In Section F.1, we introduce some essential notations used in this section. In Section F.2, we oﬀer the gradient of s(X) on W, which is equivalent to the gradient of the output of the attention mechanism on W. In Section F.3, we illustrate the gradient of L(X) on W. In Section F.4, we introduce the almost linear time algorithm for calculating the gradient of L(X) on W, along with the error bound analysis.

### F.1 Key concepts

Deﬁnition F.1 (Deﬁnition of A, [AS24a]). Let A1,A2 ∈ Rn×d be two matrices. Suppose that A = A1 ⊗ A2 ∈ Rn2×d2. We deﬁne Aj0 ∈ Rn×d2 be a n × d2 size sub-block from A. Note that there are n such sub-blocks.

Remark F.2. Note that the A1,A2 matrices in Deﬁnition F.1 is X in our setting. Since in [AS24a], they consider a more general setting, where A1,A2 can be diﬀerence matrices, while in our problem, we consider self-attention. Therefore, in our paper, we have A1 = A2 = X.

### F.2 Gradient of s(X) on W

We begin with introducing the close form of the gradient of s(X).

[AS24a] proved the close form of the gradient of c(X) = s(X) − B with respect to W for a constant matrix B. By chain rule, this is equivalent to the gradient of s(X) with respect to W. Lemma F.3 (Gradient of s(X) on W, Lemma B.1 in [AS24a]). If we have the below conditions,

- • Let A be deﬁned as Deﬁnition F.1. For every i ∈ [d2], deﬁne Aj0,i ∈ Rn to be the i-th column for Aj0 ∈ Rn×d2.
- • Let f(X),h(X),s(X) be deﬁned as Deﬁnition C.8, C.9, C.10.
- • Let W ∈ Rd×d be deﬁned as Deﬁnition C.3. Let w ∈ Rd2 denote the vector representation of W.

Then, for each i ∈ [d2], we have For each j0 ∈ [n], for every i0 ∈ [d]

ds(X)j0,i0 dwi

= Aj0,i ⊙f(X)j0,h(X)i0 − f(X)j0,h(X)i0 · Aj0,i,f(X)j0

[Figure 836]

### F.3 Gradient of L(X) on W

Diﬀering from the ℓ2 loss function used in [AS24a], our framework supports arbitrary loss functions. Therefore, we use Lemma F.4 to illustrate the gradient of L(X) on W.

- Lemma F.4 (Gradient of L(X) on W). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let W ∈ Rd×d,X ∈ Rn×d be Deﬁned as Deﬁnition C.3.
- • Let p(X) be deﬁned as Deﬁnition C.12. Then, we can show that

dL(X) dWi

= X⊤ · p(X) · X

[Figure 837]

Proof. By Lemma F.3, we have, for each i ∈ [d2], we have For each j0 ∈ [n], for every i0 ∈ [d]

ds(X)j0,i0 dwi

= Aj0,i n×1

⊙f(X)j0

,h(X)i0

,h(X)i0

,f(X)j0

(11)

− f(X)j0

· Aj0,i n×1

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

n×1

n×1

n×1

n×1

n×1

By Fact C.1, we have

Aj0,i ⊙f(X)j0,h(X)i0 = A⊤j0,i diag(f(X)j0)h(X)i0 and

f(X)j0,h(X)i0 · f(X)j0,Aj0,i = A⊤j0,i f(X)j0f(X)⊤j0h(X)i0 By Eq. (11), for each i ∈ [d2], we have For each j0 ∈ [n], for every i0 ∈ [d], we have

ds(X)j0,i0 dwi

= A⊤j0,i(diag(f(X)j0) − f(X)j0f(X)⊤j0)h(X)i0

[Figure 853]

which implies,

ds(X)j0,i0 dW

= A⊤j0

[Figure 854]

d2×n

(diag(f(X)j0) − f(X)j0f(X)⊤j0) n×n

h(X)i0 n×1

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

(12)

By Lemma C.4, for i ∈ [m], we have

dL(X) dWi

[Figure 859]

d

n

=

j2=1

i2=1

dAttni(Ti−1(X))i2,j2 dWi

Gi(i2,j2) ·

. (13)

[Figure 860]

By the deﬁnition of s(X) (Deﬁnition C.10), we have s(X) = Attni(Ti−1(X)) Combining Eq. (12) and Eq. (13), for each i ∈ [m], we have

dL(X) dWi

[Figure 861]

d

n

· A⊤j0

Gi(j0,i0) 1×1

=

[Figure 862]

[Figure 863]

i0=1

j0=1

(diag(f(X)j0) − f(X)j0f(X)⊤j0) n×n

h(X)i0 n×1

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

d2×n

Recall that we have deﬁned q(X) in Deﬁnition C.11,

(14)

q(X)j0 :=

d

Gi(j0,i0) · h(X)i0 (15)

i0=1

Recall that p(x)j0 ∈ Rn is deﬁne as Deﬁnition C.12,

p(x)j0 := (diag(f(x)j0) − f(x)j0f(x)⊤j0)q(x)j0. (16) Then, we have

dL(X) dWi

[Figure 868]

d

n

· A⊤j0

(diag(f(X)j0) − f(X)j0f(X)⊤j0) n×n

Gi(j0,i0) 1×1

=

h(X)i0 n×1

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

i0=1

j0=1

[Figure 873]

[Figure 874]

d2×n

n

(diag(f(X)j0) − f(X)j0f(X)⊤j0) n×n

A⊤j0 d2×n

=

q(X)j0 n×1

[Figure 875]

[Figure 876]

j0=1

[Figure 877]

[Figure 878]

n

A⊤j0 pj0(X)

=

j0=1

= X⊤ d×n

p(X)

#### X

[Figure 879]

[Figure 880]

n×d

n×n

where the 1st step is from Eq. (14), the 2nd step comes from Eq. (15), the 3rd step is because of Eq. (16), the 4th step is due to the tensor tricks.

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

### F.4 Fast computation

Finally, we introduce the almost linear time algorithm and its error analysis of the gradient of L(X) on W in Lemma F.5.

- Lemma F.5 (Fast computation for ddLW(X)

). If we have the below conditions,

[Figure 885]

i

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let m denote the number of self-attention transformer layers (see Deﬁnition 1.3).
- • For any i ∈ [m], let Wi = WQiWK⊤

denote the attention weight in the i-th transformer layer.

i

We can show that ddLW(X)

can be approximated in n1+o(1) time, with 1/poly(n) approximation error. Namely, our algorithm can output gw in n1+o(1) time, which satisﬁes

[Figure 886]

i

dL(X) dWi ∞ ≤ 1/poly(n)

gw −

[Figure 887]

Proof. Recall by Lemma C.15, C.16, we have deﬁned p1(X),p2(X) ∈ Rn×n.

In those Lemmas, we have p1(X),p2(X) have low rank approximation U3V3⊤ and U4V4⊤, respectively.

By the deﬁnition of p(X) (Deﬁnition C.12), we have

p(X) = p1(X) − p2(X) (17) Then, by Lemma F.4, we have

dL(X) dWi = X⊤p(X)X

[Figure 888]

= X⊤(p1(X) − p2(X))X

where the 1st step is from Lemma F.4, the 2nd step comes from Eq. (17). Let p1(X), p2(X) denote the low rank approximations for p1(X),p2(X), respectively. Proof of running time. We ﬁrst compute X⊤ p1(X)X in following order

- • Compute X⊤ d×n

U3 n×k3

, which takes n1+o(1) time.

- • Compute X⊤U3 d×k3

[Figure 889]

[Figure 890]

V3⊤ k3×n

, which takes n1+o(1) time.

- • Compute X⊤U3V3⊤ d×n

[Figure 891]

[Figure 892]

X

, which takes n1+o(1) time.

n×d

The overall running time for X⊤ p1(X)X is n1+o(1). Similarly, the overall running time for X⊤ p2(X)X is n1+o(1). Since X⊤ p1(X)X,X⊤ p2(X)X ∈ Rd×d, the computation time for X⊤( p1(X)− p2(X))X is O(d2). Therefore, the overall running time for X⊤( p1(X) − p2(X))X is n1+o(1). Proof of error bound. We consider the error for X⊤ p1(X)X ﬁrst.

X⊤ p1(X)X − X⊤p1(X)X ∞

= X⊤( p1(X) − p1(X))X ∞ ≤ n2 X 2∞ p1(X) − p1(X) ∞ ≤ n2(ǫ/poly(n)) X 2∞ ≤ ǫ/poly(n) (18)

where the 1st step is from basic algebra, the 2nd step comes from basic linear algebra, the 3rd step is because of p1(X) − p1(X) ∞ ≤ ǫ/poly(n), the 4th step is due to X ∞ ≤ poly(n).

Similarly, we can have

X⊤ p2(X)X − X⊤p2(X)X ∞ ≤ ǫ/poly(n) (19) Therefore, we have

X⊤ p(X)X − X⊤p(X)X ∞

= X⊤ p1(X)X − X⊤p1(X)X + X⊤ p2(X)X − X⊤p2(X)X ∞ ≤ X⊤ p1(X)X − X⊤p1(X)X ∞ + X⊤ p2(X)X − X⊤p2(X)X ∞ ≤ (ǫ/poly(n)) + (ǫ/poly(n))

= ǫ/poly(n)

where the 1st step is from basic algebra, the 2nd step comes from triangle inequality, the 3rd step is because of Eq. (18) and Eq. (19), the 4th step is due to basic algebra.

Then, we choose ǫ = 1/poly(n), we have

dL(X) dWi ∞ ≤ 1/poly(n)

gw −

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

## G Fast Computation for Gradient on WV

- In Section G.1, we introduce the close form of the gradient of s(X) on WV . In Section G.2, we provide the close form of the gradient of L(X) on WV . In Section G.3, based on the close form calculated in the previous section, we introduce the almost linear time algorithm for computing the gradient of L(X) on WV .

- G.1 Gradient of s(X) on WV Since s(X) = f(X)h(X), we begin with considering the gradient of h(X) on WV in Lemma G.1.

- Lemma G.1 (Gradient of h(X) on WV ). If we have the below conditions,

- • Let h(X) be deﬁned as Deﬁnition C.9.
- • Let WV be deﬁned as Deﬁnition C.3.

Then, for any i0 ∈ [n],j0 ∈ [d] and any i1,j1 ∈ [d], we have

dh(X)i0,j0 d(WV )i1,j1

=

[Figure 898]

Xi0,i1 j0 = j1 0 j0 = j1

Proof. Since hi0,j0 satisﬁes

hi0,j0 = Xi⊤0,∗(WV )∗,j0, we have hi0,j0 only depends on (WV )∗,j0.

Hence, we have, for j0 = j1,

dh(X)i0,j0 d(WV )i1,j1

= 0

[Figure 899]

For j0 = j1 case, we have

dh(X)i0,j0 d(WV )i1,j0

= Xi0,i1

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

Combining the result in the previous Lemma and the chain rule, we can have the gradient of s(X) on WV in Lemma G.2.

- Lemma G.2 (Gradient of s(X) on WV ). If we have the below conditions,

- • Let s(X) be deﬁned as Deﬁnition C.10.
- • Let WV be deﬁned as Deﬁnition C.3. Then, for any i2 ∈ [n],j2 ∈ [d] and any i1,j1 ∈ [d], we have
- • Part 1. ds(X)i2,j2

[Figure 905]

d(WV )i1,j1

=

f(X)⊤i

2,∗X∗,i1 j2 = j1 0 j2 = j1

- • Part 2.

ds(X)i2,j2 dWV

= X⊤ d×n

e⊤j2 1×d

f(X)i2,∗ n×1

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

d×d

Proof. Proof of Part 1. By Deﬁnition C.10, we have

s(X)i2,j2 := f(X)⊤i2,∗h(X)∗,j2 (20)

Therefore, s(X)i2,j2 is only depends on h(X)∗,j2, which further means s(X)i2,j2 is only depends on (WV )∗,j2.

Hence, for j1 = j2, we have

We consider j1 = j2 case. By, Eq. (20), we can derive that

ds(X)i2,j2 d(WV )i1,j2

= 0

[Figure 911]

ds(X)i2,j2 dh(X)i3,j2

= f(X)i2,i3 (21)

[Figure 912]

By chain rule, we have

ds(X)i2,j2 d(WV )i1,j2

[Figure 913]

d

ds(X)i2,j2 dh(X)i3,j2

dh(X)i3,j2 d(WV )i1,j2

=

[Figure 914]

[Figure 915]

i3=1

=

d

dh(X)i3,j2 d(WV )i1,j2

f(X)i2,i3

[Figure 916]

i3=1

d

f(X)i2,i3Xi3,i1

=

i3=1

= f(X)⊤i2,∗X∗,i1 (22) where the 1st step is from chain rule, the 2nd step comes from Eq. (21), the 3rd step is because of

- Lemma G.1, the 4th step is due to basic linear algebra. Proof of Part 2. By Eq (22), we have

which implies

ds(X)i2,j2 d(WV )∗,j2 d×1

= X⊤ d×n

f(X)i2,∗ n×1

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

ds(X)i2,j2 dWV

= X⊤ d×n

e⊤j2 1×d

f(X)i2,∗ n×1

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

d×d

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

### G.2 Gradient of L(X) on WV

Since we have already got the close form of the gradient of s(X) on WV , we can easily extend it and get the close form of the gradient of L(X) on WV in Lemma G.3.

- Lemma G.3 (Gradient of L(X) on WV ). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.

- • Let WV be deﬁned as Deﬁnition C.3. Then, we can show that

dL(X) dWVi d×d

= X⊤ d×n

f(X)

Gi n×d

[Figure 931]

[Figure 932]

[Figure 933]

n×n

[Figure 934]

[Figure 935]

Proof. We slightly abuse the notation, using WV to represent Vi in Lemma G.1, G.2. By Lemma G.2, we have

By Lemma C.4, we have

ds(X)i2,j2 dWV

= X⊤ d×n

e⊤j2 1×d

f(X)i2,∗ n×1

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

d×d

(23)

d

n

dAttni(Ti−1(X))i2,j2 dWVi

dL(X) dWVi

Gi(i2,j2) ·

=

. (24)

[Figure 941]

[Figure 942]

j2=1

i2=1

By Deﬁnition C.10 and Deﬁnition C.3, we have

s(X) = Attni(Ti−1(X)) Therefore, combining Eq. (23) and Eq. (24), we have

dL(X) dWVi

[Figure 943]

d

n

e⊤j2 1×d

X⊤ d×n

Gi(i2,j2) 1×1

=

f(X)i2,∗ n×1

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

j2=1

i2=1

n

d

X⊤ d×n

e⊤j2 1×d

f(X)i2,∗ n×1

=

Gi(i2,j2) 1×1

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

i2=1

j2=1

n

X⊤ d×n

Gi(i2,∗)⊤ 1×d

f(X)i2,∗ n×1

=

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

i2=1

= X⊤ d×n

Gi n×d

f(X)

[Figure 956]

[Figure 957]

n×n

where the 1st step is from Eq. (23) and Eq. (24), the 2nd step comes from basic algebra, the 3rd step is because of basic linear algebra, the 4th step is due to basic linear algebra.

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

### G.3 Fast computation

Finally, we can introduce our almost linear time algorithm for computing the L(X) gradient on WV .

- Lemma G.4 (Fast computation for d(dLW(X)

V )i). If we have the below conditions,

[Figure 962]

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let m denote the number of self-attention transformer layers (see Deﬁnition 1.3).
- • For any i ∈ [m], let WVi ∈ Rd×d denote the attention weight in the i-th transformer layer. We can show that ddLW(X)

can be approximated in n1+o(1) time, with 1/poly(n) approximation error. Namely, our algorithm can output gv in n1+o(1) time, which satisﬁes

[Figure 963]

Vi

dL(X) dWVi ∞ ≤ 1/poly(n) Proof. Recall in Lemma C.13, U1V1⊤ is the low rank approximation of f(X).

gv −

[Figure 964]

Let f(X) := U1V1⊤ denote the low rank approximation of f(X). Recall in Lemma G.3, we have

dL(X) dWVi d×d

= X⊤ d×n

f(X)

Gi n×d

[Figure 965]

[Figure 966]

[Figure 967]

n×n

[Figure 968]

[Figure 969]

Proof of running time. We compute X⊤ f(X)Gi in following order

• Compute X⊤ d×n

· U1

, which takes n1+o(1) time.

n×k1

· V1⊤

• Compute X⊤ · U1

[Figure 970]

[Figure 971]

d×k1

, which takes n1+o(1) time.

k1×n

• Compute X⊤ · U1 · V1⊤

· Gi

[Figure 972]

[Figure 973]

d×n

, which takes d2 · n time.

n×d

The overall running time is n1+o(1). Proof of error bound. We have

X⊤ · f(X) · Gi − X⊤ · f(X) · Gi ∞

= X⊤ · (f(X) − f(X)) · Gi ∞ ≤ n2 X ∞ f(X) − f(X) ∞ Gi ∞ ≤ n2(ǫ/poly(n)) X ∞ Gi ∞ ≤ ǫ/poly(n)

where the 1st step is from basic algebra, the 2nd step comes from basic linear algebra, the 3rd step is because of f(X) − f(X) ∞ ≤ ǫ/poly(n), the 4th step is due to X ∞ ≤ poly(n) and

Gi ∞ ≤ poly(n). Let gv = X⊤ · f(X) · Gi. We choose ǫ = 1/poly(n). Then, we have

dL(X) dWVi ∞ ≤ 1/poly(n)

gv −

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

## H Gradient Approximation for Entire Model

- In Section H.1, we introduce the close form of Gi and argue that Gi can be computed in almost linear time n1+o(1). In Section H.2, we provide the almost linear time algorithm for gradient computing on a single-layer transformer. In Section H.3, with the help of math induction, we introduce the almost linear time algorithm for computing the gradient of the multi-layer transformer, along with its approximation error.

### H.1 Computation time for Gi

Here we consider gi in Deﬁnition 1.3 as a linear layer with an arbitrary non-linear activation φ. Since gi can be viewed as a composition of an MLP and an activation function, we begin with analyzing the Ti gradient on Attni.

- Lemma H.1 (Gradient of Ti on Attni ). If we have the below conditions,

- • Let Ti(X) be deﬁned as Deﬁnition 3.3.
- • Assuming for any Z ∈ Rn×d, we have gi(Z) ∈ Rn×d, and gi(Z) = φ(ZWg), where Wg ∈ Rd×d and φ : R → R denotes any element-wise activation function. Let φ′ denote the derivative of φ.
- • We simplify the notation, using Ti and Attni to represent Ti(X) and Attni(Ti−1(X)), respectively.
- • For any matrix Z ∈ Rn×d, we use Z(i,j) to denote the (i,j)-th entry of Z. Then, we can show that, for any i4,i5 ∈ [n],j4,j5 ∈ [d],
- • Part 1.

dTi(i4,j4) dAttni(i5,j5)

[Figure 979]

=

 



φ′(Attni(i4,∗)⊤Wg(∗,j4)) 1×1

[Figure 980]

[Figure 981]

Wg(j5,j4) 1×1

[Figure 982]

[Figure 983]

i4 = i5

0 i4 = i5

- • Part 2.

dTi(i4,j4) dAttni n×d

= φ′(Attni(i4,∗)⊤Wg(∗,j4))

ei4 n×1

[Figure 984]

[Figure 985]

[Figure 986]

1×1

[Figure 987]

[Figure 988]

Wg(∗,j4)⊤ 1×d

[Figure 989]

[Figure 990]

Proof. Proof of Part 1. By the deﬁnition of Ti (Deﬁnition 3.3), for i4 ∈ [d],j4 ∈ [n], we have

Ti(i4,j4) = φ(Attni(i4,∗)⊤Wg(∗,j4)) Therefore, for any i5 = i4, we have

Then, we consider i4 = i5 case.

dTi(i4,j4) dAttni(i5,j5)

= 0

[Figure 991]

By basic calculus, we have

dTi(i4,j4) dAttni(i4,j5)

= φ′(Attni(i4,∗)⊤Wg(∗,j4))

Wg(j5,j4) 1×1

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

1×1

Combining two equations mentioned above, we have the result for Part 1. Proof of Part 2. By result of Part 1, for i5 = i4, we have

dTi(i4,j4) dAttni(i4,j5)

= φ′(Attni(i4,∗)⊤Wg(∗,j4))

Wg(j5,j4) 1×1

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

1×1

which implies

dTi(i4,j4) dAttni(i4,∗)

= φ′(Attni(i4,∗)⊤Wg(∗,j4))

Wg(∗,j4) d×1 By result of Part 1, for i5 = i4, we have

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

1×1

dTi(i4,j4) dAttni(i5,∗)

= 0

[Figure 1007]

By basic linear algebra, combining the two equations mentioned above, we have

dTi(i4,j4) dAttni

= φ′(Attni(i4,∗)⊤Wg(∗,j4))

[Figure 1008]

[Figure 1009]

[Figure 1010]

1×1

Wg(∗,j4)⊤ 1×d

ei4 n×1

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

Then, we can argue that the computation for Gi can be done in almost linear time n1+o(1).

- Lemma H.2 (Computation time for Gi, formal version of Lemma 5.4). If we have the below conditions,

- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 1017]

i(Ti−1(X)).

- • Assuming we already have ddTL(X)

[Figure 1018]

i(X).

- • Assuming for any Z ∈ Rn×d, we have gi(Z) ∈ Rn×d, and gi(Z) = φ(ZWg), where Wg ∈ Rd×d and φ : R → R denotes any element-wise activation function. Let φ′ denote the derivative of φ.
- • We simplify the notation, using Ti and Attni to represent Ti(X) and Attni(Ti−1(X)), respectively.
- • For any matrix Z ∈ Rn×d, we use Z(i,j) to denote the (i,j)-th entry of Z. Then, we can show that Gi can be computed in n1+o(1) time.

Proof. Let gTi := dLd(TX)

, and for any i4 ∈ [n],j4 ∈ [d], let gTi(i4,j4) denote the (i4,j4)-th entry of gTi.

[Figure 1019]

i

Similarly, for any i5 ∈ [n],j5 ∈ [d], let Ti(i5,j5) denote the (i5,j5)-th entry of Ti. We can have

dL(X) dAttni

Gi =

[Figure 1020]

dTi dAttni

dL(X) dTi ·

=

[Figure 1021]

[Figure 1022]

dTi dAttni

= gTi ·

[Figure 1023]

n

d

dTi(i4,j4) dAttni

gTi(i4,j4) ·

=

[Figure 1024]

i4=1

j4=1

where the 1st step is from the deﬁnition of Gi, the 2nd step comes from chain rule, the 3rd step is because of the deﬁnition of gTi, the 4th step is due to chain rule.

d

n

dTi(i4,j4) dAttni

gTi(i4,j4) ·

[Figure 1025]

j4=1

i4=1

d

n

φ′(Attni(i4,∗)⊤Wg(∗,j4)) 1×1

Wg(∗,j4)⊤ 1×d

gTi(i4,j4) 1×1

=

ei4 n×1

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

j4=1

i4=1

d

n

Wg(∗,j4)⊤ 1×d

φ′(Attni(i4,∗)⊤Wg(∗,j4)) 1×1

gTi(i4,j4) 1×1

ei4 n×1

=

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

j4=1

i4=1

n

))⊤

⊙φ′(Attni(i4,∗)⊤Wg)

ei4 n×1

(gTi(i4,∗)

( Wg d×d

=

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

i4=1

d×1

d×1

= (gTi ⊙ φ′(AttniWg))

Wg⊤ d×d

[Figure 1042]

[Figure 1043]

n×d

(25)

where the 1st step is from Lemma H.1, the 2nd step comes from basic algebra, the 3rd step is because of basic linear algebra, the 4th step is due to basic linear algebra.

By Eq. (25), we have the close form of Gi. We can compute Gi in the following order

- • Compute (gTi ⊙ φ′(AttniWg)) n×d

[Figure 1044]

[Figure 1045]

, which takes n · d time.

- • Compute (gTi ⊙ φ′(AttniWg)) n×d

[Figure 1046]

[Figure 1047]

Wg⊤ d×d

, which takes d2 · n time.

Therefore, the overall running time for Gi is n1+o(1).

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

### H.2 Fast computation for single-layer transformer

In this section, we dive into the computation time and approximation error of the gradient of a single-layer transformer. We demonstrate in the following Lemma that the gradient of a singlelayer transformer can be computed in almost linear time n1+o(1), and its error can be bounded by 1/poly(n).

- Lemma H.3 (Single-layer transformer gradient approximation). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let X be deﬁned as Deﬁnition C.3.
- • Let the gradient matrix Gi ∈ Rn×d be deﬁned as Gi = dAttndL(X)

[Figure 1052]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.
- • Assuming for any Z ∈ Rn×d, we have gi(Z) ∈ Rn×d, and gi(Z) = φ(Z·Wg), where Wg ∈ Rd×d and φ : R → R denotes any element-wise activation function. Let φ′ denote the derivative of φ.
- • Suppose we have a single-layer transformer (see Deﬁnition 1.3). Then, we can show that,
- • Part 1: running time. Our algorithm can approximate dLd(XX) in n1+o(1) time.

[Figure 1053]

- • Part 2: error bound. The approximation error of the single-layer transformer can be bounded by 1/poly(n). Namely, our algorithm output g1 satisﬁes

dL(X) dX ∞ ≤ 1/poly(n)

g1 −

[Figure 1054]

Proof. By Deﬁnition 1.3, a single-layer transformer has following structure:

g1 ◦ Attn1 ◦ g0(X) By the deﬁnition of Gi, we have

dL(X) dAttn1(T0(X))

G1 =

[Figure 1055]

dT1(X) dAttn1(T0(X))

dL(X) dT1(X) ·

=

[Figure 1056]

[Figure 1057]

(26)

By Lemma H.2, we have G1 can be computed in n1+o(1) time. Proof of Part 1: running time.

For less confusion, in this part of the proof, we ignore the approximation error temporarily. Since we have got G1, we use methods mentioned in Lemma E.11, F.5, G.4 to compute

, ddLW(X)

dL(X) dT0(X), ddLW(X)

, respectively, which takes n1+o(1) time for each. Then, since we have ddTL(X)

[Figure 1058]

[Figure 1059]

[Figure 1060]

1

V1

0(X), again by Lemma H.2, we have dLd(XX) can be computed in n1+o(1)

[Figure 1061]

[Figure 1062]

time. Therefore, the overall running time is n1+o(1). Proof of Part 2: error bound.

Then, we move on to the error bound. By Lemma H.2 and Eq. (26), there is no approximation error when computing G1. By Lemma E.11, F.5, G.4, we have there is 1/poly(n) approximation error on ddTL(X)

, ddLW(X)

0(X), ddLW(X)

, respectively.

[Figure 1063]

[Figure 1064]

[Figure 1065]

1

V1

, ddLW(X)

0(X), ddLW(X)

Let gt0, gw1, gv1 denote the approximation results of ddTL(X)

, respectively. We have

[Figure 1066]

[Figure 1067]

[Figure 1068]

1

V1

dL(X) dT0(X) ∞ ≤ 1/poly(n) (27)

gt0 −

[Figure 1069]

and

dL(X)

gw1 −

dW1 ∞ ≤ 1/poly(n) and

[Figure 1070]

dL(X) dWV1 ∞ ≤ 1/poly(n)

gv1 −

[Figure 1071]

Let G0 = gt0 · dTd0X(X) denote the approximated version of G0. We have

[Figure 1072]

G0 − G0 ∞

dT0(X)

dL(X) dT0(X)

) ·

= ( gt0 −

dX ∞ ≤ n · d gt0 −

[Figure 1073]

[Figure 1074]

dL(X) dT0(X) ∞

dT0(X)

dX ∞ ≤ n · d(1/poly(n))

[Figure 1075]

[Figure 1076]

dT0(X)

dX ∞ ≤ 1/poly(n)

[Figure 1077]

where the 1st step is from the deﬁnition of G0, the 2nd step comes from basic linear algebra, the 3rd step is because of Eq. (27), the 4th step is due to each entry can be written by O(log n) bits.

Let g1 = G0. Therefore, we have

dL(X) dX ∞ ≤ 1/poly(n)

g1 −

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

### H.3 Fast computation for multi-layer transformer

Since we have already demonstrated that almost linear time gradient computation can be applied to a single-layer transformer, with the help of math induction, we can easily generalize that result to the multi-layer transformer. In the following Lemma, we display that the gradient of the multi-layer transformer can be computed in almost linear time, and its approximation error can be bounded by 1/poly(n).

- Lemma H.4 (Multi-layer transformer gradient approximation, formal version of Lemma 5.5). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let X be deﬁned as Deﬁnition C.3.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 1083]

i(Ti−1(X)).

- • For i2 ∈ [n],j2 ∈ [d], let Gi(i2,j2) denote the (i2,j2)-th entry of Gi.
- • Let gradient components for each layer be computed according to Lemma E.11, F.5, G.4.
- • Assuming for any Z ∈ Rn×d, we have gi(Z) ∈ Rn×d, and gi(Z) = φ(Z·Wg), where Wg ∈ Rd×d and φ : R → R denotes any element-wise activation function. Let φ′ denote the derivative of φ.
- • Suppose we have a m-layer transformer (see Deﬁnition 1.3). Then, we can show that,
- • Part 1: running time. Our algorithm can approximate dLd(XX) in n1+o(1) time.

[Figure 1084]

- • Part 2: error bound. The approximation error of the multi-layer transformer can be bounded by 1/poly(n). Namely, our algorithm output g satisﬁes

dL(X)

dX ∞ ≤ 1/poly(n) Proof. We use math induction to prove this Lemma.

g −

[Figure 1085]

- Step 1: Proof of a single-layer transformer. Firstly, by Lemma H.3, we have that for one-layer transformer, our conclusion is established.
- Step 2: Assumption for k-layer transformer. Secondly, we assume for any k, for k-layer transformer model, we have

- • Our algorithm can approximate dLd(XX) in O(n1+o(1)) time.

[Figure 1086]

- • The approximation error of the k-layer transformer can be bounded by 1/poly(n). Namely, our algorithm output g satisﬁes

g −

dL(X) dX ∞ ≤ 1/poly(n)

[Figure 1087]

- Step 3: Proof of (k + 1)-layer transformer. Thirdly, we consider the (k + 1)-layer transformer model. Without loss of generality, we assume that the additional transformer layer is added at the

beginning of the model. Namely, let Fk denote a k-layer transformer model. We have

Fk(X) = gk ◦ Attnk ◦ ··· ◦ g1 ◦ Attn1 ◦ g0(X) Let the (k + 1)-layer transformer model have the following structure: Fk+1(X) = Fk ◦ Attn ◦ g(X) (28)

Let T0 := g(X). By assumption, we have

- • dAttndL(X(T)

[Figure 1088]

0) can be approximated in n1+o(1) time.

- • Let gk denote the approximated version of dAttndL(X(T)

0). We have

[Figure 1089]

dL(X) dAttn(T0) ∞ ≤ 1/poly(n) (29)

gk −

[Figure 1090]

- Step 3.1: Proof of the running time for (k + 1)-layer transformer For less confusion, in this part of the proof, we ignore the approximation error temporarily. By the assumption, we have dAttndL(X(T)

[Figure 1091]

0) can be approximated in n1+o(1) time. We compute dLd(XX) in following order:

[Figure 1092]

- • Since we already have dAttndL(X(T)

[Figure 1093]

0), by Lemma E.11, the computation time for dLdT(X)

[Figure 1094]

0

is n1+o(1).

- • Since we have dLdT(X)

[Figure 1095]

0

, by Lemma H.2, the computation time for dLd(XX) is n1+o(1). Therefore, for (k + 1)-layer transformer, the overall running time for dLd(XX) is n1+o(1).

[Figure 1096]

[Figure 1097]

- Step 3.2: Proof of the error bound for (k + 1)-layer transformer

By Lemma E.11, during the process of solving the approximated version of ddLg((XX)), the approximation error will not be magniﬁed by more than poly(n).

[Figure 1098]

Let gt0 denote the approximated version of ddLg((XX)), we have

[Figure 1099]

dL(X) dg(X) ∞

gt0 −

[Figure 1100]

dL(X) dT(X) ∞

≤ poly(n) gk −

[Figure 1101]

≤ 1/poly(n) (30)

where the 1st step is from the above statement, the 2nd step comes from Eq. (29), the 3rd step is because of basic algebra. Then, we consider

dL(X) dX

=

[Figure 1102]

dL(X) dg(X) ·

dg(X) dX

[Figure 1103]

[Figure 1104]

(31)

Recall that we have g = dLd(XX). Then, we have g −

[Figure 1105]

dL(X) dX ∞

[Figure 1106]

dL(X) dg(X)

dg(X)

= ( gt0 −

) ·

dX ∞ ≤ n · d gt0 −

[Figure 1107]

[Figure 1108]

dL(X) dg(X) ∞

dg(X)

dX ∞ ≤ n · d(1/poly(n))

[Figure 1109]

[Figure 1110]

dg(X)

dX ∞ ≤ 1/poly(n)

[Figure 1111]

where the 1st step is from Eq. (31), the 2nd step comes from basic linear algebra, the 3rd step is because of Eq. (30), the 4th step is due to each entry can be written by O(log n) bits.

Step 4: Use math induction. So far, with the assumption that our statement holds under k-layer transformer, we have proved

that our statement still holds under (k + 1)-layer transformer. Therefore, by math induction, our statement holds for any m-layer transformer.

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

## I Causal Attention Mask

This section will discuss how to combine the causal attention mask with our framework. We argue that even with the causal attention mask, we can also achieve almost linear time gradient computing for the multi-layer transformer.

In Section I.1, we introduce essential tools from literature to deal with the causal mask added on the attention matrix. In Section I.2, we show that with the addition of causal mask, our framework can still achieve almost linear time gradient computation.

- I.1 Tools from previous work Firstly, we restate a classical low-rank approximation method in the literature.

Lemma I.1 (Low-rank approximation, [AS23]). Suppose Q,K ∈ Rn×d, with Q ∞ ≤ R, and

K ∞ ≤ R. Let A := exp(QK⊤/d) ∈ Rn×n. For accuracy parameter ǫ ∈ (0,1), there is a positive integer g bounded above by

g = O max

log(1/ǫ) log(log(1/ǫ)/R)

,R2 ,

[Figure 1116]

and a positive integer r bounded above by

r ≤

2(g + d) 2g

such that: There is a matrix A ∈ Rn×n that is an (ǫ,r)-approximation of A ∈ Rn×n. Furthermore, the matrices U0 and V0 deﬁning A can be computed in O(n · r) time.

Then, we provide the formal deﬁnition for the causal attention mask.

- Deﬁnition I.2 (Causal attention mask, [LLS+24d]). We deﬁne the causal attention mask as M ∈ {0,1}n×n, where Mi,j = 1 if i ≥ j and Mi,j = 0 otherwise.

In previous work [LLS+24d], they point out there exists an algorithm (Algorithm 2) that can calculate low-rank matrices (with the causal attention mask) multiplication with any vector v in almost linear time. We restate their results in Lemma I.3.

- Lemma I.3 (Fast computation for causal attention mask on tensor, [LLS+24d]). Let M ∈ {0,1}n×n

be a causal attention mask deﬁned in Deﬁnition I.2. Let U0,V0 ∈ Rn×k. Let v ∈ Rn. Then, there exists an algorithm (see Algorithm 2) whose output satisﬁes that

Y = (M ⊙ (U0V0⊤))v, which takes O(nk) time.

We extend their results to the multiplication of matrix with no(1) columns.

[Figure 1117]

Algorithm 2 Causal attention mask algorithm, Algorithm 4 in [LLS+24d]

[Figure 1118]

- 1: procedure CausalMask(U0 ∈ Rn×k,V0 ∈ Rn×k,v ∈ Rn) ⊲ Lemma I.3
- 2: c0 ← 0k
- 3: for j = 1 → n do
- 4: bj ← (V0⊤)j k×1

[Figure 1119]

[Figure 1120]

vj scalar

⊲ Let (V0⊤)j denote the j-th row of V0 ∈ Rn×k

- 5: cj ← cj−1 k×1

[Figure 1121]

[Figure 1122]

+ bj

k×1

- 6: end for
- 7: for j = 1 → n do
- 8: Yj ← (U0⊤)j k×1

[Figure 1123]

[Figure 1124]

, cj

k×1

- 9: end for
- 10: return Y ⊲ Y ∈ Rn
- 11: end procedure

[Figure 1125]

- Lemma I.4 (Fast computation for causal attention mask on matrix). If we have the below conditions,

- • Let M ∈ {0,1}n×n be a causal attention mask deﬁned in Deﬁnition I.2.
- • Let U0,V0 ∈ Rn×k where k = no(1).
- • Let H ∈ Rn×kH where kH = no(1). Then, there exists an algorithm, whose output satisﬁes that

Z = (M ⊙ (U0V0⊤))H,

which takes n1+o(1) time. Proof. For j ∈ [kH], let H∗,j ∈ Rn denote the j-th column of H.

By Lemma I.3, we can compute (M ⊙ (U0V0⊤))H∗,j in O(nk) time. There are kH columns in total. Therefore, the overall running time is O(nkkH) = O(n · no(1) ·

no(1)) = n1+o(1).

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

### I.2 Fast computation with causal mask

We can easily change all low-rank matrices multiplication to the algorithm mentioned in Lemma I.4. Then, our framework can support the causal attention mask and still achieves almost linear time gradient computing for the multi-layer transformer.

The causal mask directly aﬀects the attention matrix, so it’s necessary to deﬁne the attention matrix with the causal mask applied.

- Deﬁnition I.5. Let M ∈ {0,1}n×n be a causal attention mask deﬁned in Deﬁnition I.2. We deﬁne attention matrix with causal mask as:

f(X) := D−1(M ⊙ A) where A := exp(XWX⊤/d) and D := diag((M ⊙ A) · 1n).

After analyzing the components of gradients on Ti(X),Wi,WVi in Section E, F and G, we categorize them into two groups: one involving the dot product and the other involving the Hadamard product of the attention matrix. Then, we can show f(X)H and ( f(X) ⊙ (UV ⊤))H for low rank matrices U,V,H can be approximated in almost linear time.

- Lemma I.6. If we have the below conditions,

- • Let f(X) be deﬁned in Deﬁnition I.5.
- • Let U,V ∈ Rn×k where k = no(1).
- • Let H ∈ Rn×kH where kH = no(1). Then, approximating the following takes n1+o(1) time:
- • Part 1. f(X)H
- • Part 2. ( f(X) ⊙ (UV ⊤))H

Proof. From Deﬁnition I.5, we know

f(X) := D−1(M ⊙ A) where D := diag((M ⊙ A) · 1n).

By Lemma I.1, U0V0⊤ is a good approximation for A. Then, we can approximate f(X) by: D−1(M ⊙ (U0V0⊤)) where D := diag((M ⊙ (U0V0⊤)) · 1n).

Using Lemma I.3, we know (M ⊙ (U0V0⊤)) · v for any vector v ∈ Rn can be computed in almost linear time.

We begin by examining the normalization matrix D−1. Calling Lemma I.3, we compute (M ⊙

(U0V0⊤)) · 1n in almost linear time. Then, it takes O(n) time to make (M ⊙ (U0V0⊤)) · 1n diagonal. Given that D is diagonal, its inverse D−1 can be determined in O(n) time. Thus, we can compute D−1 in almost linear time.

Proof of Part 1. H can be viewed as a combination of kH vectors, each of size n. Calling

- Lemma I.4, we can compute (M ⊙ (U0V0⊤))H in n1+o(1) time. Finally, we compute D−1

(M ⊙ (U0V0⊤))H n×kH

, which takes n1+o(1) time since D−1 is diagonal. The

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

n×n

overall gradient computation remains n1+o(1) time. Proof of Part 2. The proof for this part involves Fact C.2. We can show

((D−1(M ⊙ (U0V0⊤))) ⊙ (UV ⊤))H

= ((M ⊙ (D−1U0V0⊤)) ⊙ (UV ⊤))H = (M ⊙ ((D−1U0V0⊤) ⊙ (UV ⊤)))H = (M ⊙ ((D−1U0) ⊘ U)(V0 ⊘ V )⊤)H

where the 1st step is from D(A⊙B) = (DA) ⊙B = A ⊙(DB) for diagonal matrix D ∈ Rm×m and A,B ∈ Rm×n, the 2nd step comes from (A ⊙ B) ⊙ C = A ⊙ (B ⊙ C) for A,B,C ∈ Rm×n, and the last step follows from Fact C.2.

Let UM := (D−1U0) ⊘ U and VM := V0 ⊘ V .

For UM, we compute D−1 n×n

which takes nk time. We then compute (D−1U0)

⊘ U

U0 n×k

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

n×k

n×k

takes O(nk2) time. For VM, we compute V0

which takes O(nk2) time.

⊘ V

n×k

n×k

We now have (M ⊙ (UMVM⊤)H. Calling Lemma I.4, we ﬁnish the proof. We now prove for gradient components that have dot product.

which

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

- Lemma I.7 (Components for dot product). If we have the below conditions,

- • Let f(X) be deﬁned in Deﬁnition I.5.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 1142]

i(Ti−1(X)).

- • Let D6 = −f(X)diag(K)XW⊤ be deﬁned in Lemma D.17.
- • Let D2 = −diag(K)f(X)XW be deﬁned in Lemma D.17.
- • Let D8 = f(X)GiWV⊤ be deﬁned in Lemma D.17.
- • Let gv := X⊤f(X)Gi be the gradient on WVi and deﬁned in Lemma G.3. Then, we can show the following can be approximated in almost linear time:
- • Part 1. D6 = − f(X)diag(K)XW⊤
- • Part 2. D2 = −diag(K) f(X)XW
- • Part 3. D8 = f(X)GiWV⊤
- • Part 4. gv := X⊤ f(X)Gi

Proof. Proof of Part 1. For D6, we compute diag(K)

#### X

ﬁrst, which takes nd time.

[Figure 1143]

[Figure 1144]

n×d

n×n

using Part 1. of Lemma I.6, which takes n1+o(1) time.

diag(K)X

Then, we compute f(X)

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

n×n

n×d

W⊤ d×d

, which takes n1+o(1) time.

Finally, we compute f(X)diag(K)X

[Figure 1149]

[Figure 1150]

n×d

using Part 1. of Lemma I.6, which takes

Proof of Part 2. For D2, we compute f(X)

#### X

[Figure 1151]

[Figure 1152]

n×d

n×n

n1+o(1) time. Then, we compute diag(K)

, which takes nd time.

f(X)X

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

n×n

n×d

, which takes n1+o(1) time.

#### W

After that, we compute diag(K) f(X)X

[Figure 1157]

[Figure 1158]

d×d

n×d

- Proof of Part 3. For D8, we compute in the following steps: We compute f(X)

using Part 1. of Lemma I.6, which takes n1+o(1) time.

Gi n×d

[Figure 1159]

[Figure 1160]

n×n

WV⊤ d×d

, which takes n · d2 time.

Then, we compute f(X)Gi

[Figure 1161]

[Figure 1162]

n×d

- Proof of Part 4. For gv, we compute in the following steps: We compute f(X)

using Part 1. of Lemma I.6, which takes n1+o(1) time.

Gi n×d

[Figure 1163]

[Figure 1164]

n×n

Then, we compute X⊤ d×n

, which takes n · d2 time.

f(X)Gi n×d

[Figure 1165]

[Figure 1166]

We then prove for gradient components that have Hadamard product.

- Lemma I.8 (Components for Hadamard product). If we have the below conditions,

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

- • Let f(X) be deﬁned in Deﬁnition I.5.
- • Let Gi ∈ Rn×d denote the gradient matrix resulting from the application of the chain rule up to the function gi, i.e., Gi = dAttndL(X)

[Figure 1171]

i(Ti−1(X)).

- • Let D7 = (f(X) ⊙ (h(X)G⊤i ))XW⊤ be deﬁned in Lemma D.17.
- • Let D4 = (f(X) ⊙ (Gih(X)⊤))XW be deﬁned in Lemma D.17.
- • Let gw := X⊤p(X)X = X⊤(p1(X) − p2(X))X be the gradient on Wi and deﬁned in Deﬁnition C.12 and Lemma F.5 where p1(X) = f(X) ⊙ q(X) and p2(X) = diag(p1(X) · 1n)f(X).

Then, we can show the following can be approximated in almost linear time:

- • Part 1. D7 = ( f(X) ⊙ (h(X)G⊤i ))XW⊤
- • Part 2. D4 = ( f(X) ⊙ (Gih(X)⊤))XW
- • Part 3. gw := X⊤( p1(X) − p2(X))X where p1(X) = f(X) ⊙ q(X) and p2(X) = diag( p1(X) · 1n) f(X).

Proof. Proof of Part 1. For D7, we can compute ( f(X) ⊙ (h(X)G⊤i ))

#### X

using Part 2. of

[Figure 1172]

[Figure 1173]

n×d

n×n

- Lemma I.6, which takes n1+o(1) time. We then compute ( f(X) ⊙ (h(X)G⊤i ))X

W⊤ d×d

, which takes nd2 time.

[Figure 1174]

[Figure 1175]

n×d

Proof of Part 2. For D7, we can compute ( f(X) ⊙ (Gih(X)⊤))

#### X

using Part 2. of

[Figure 1176]

[Figure 1177]

n×d

n×n

- Lemma I.6, which takes n1+o(1) time. We then compute ( f(X) ⊙ (Gih(X)⊤))X

, which takes nd2 time.

#### W

[Figure 1178]

[Figure 1179]

d×d

n×d

Proof of Part 3. For gw, we consider X⊤ p1(X)X ﬁrst. Based on Deﬁnition C.11, we have p1(X) = f(X) ⊙ q(X) = f(X) ⊙ (Gih(X)⊤). We then compute ( f(X) ⊙ (Gih(X)⊤))X using Part

2. of Lemma I.6, which takes n1+o(1) time. After that, we compute X⊤ d×n

( f(X) ⊙ (Gih(X)⊤))X n×d

,

[Figure 1180]

[Figure 1181]

which takes nd2 time.

Now we consider X⊤ p2(X)X. By deﬁnition, p2(X) = diag( p1(X) · 1n) f(X). We ﬁrst compute p1(X)·1n = ( f(X)⊙(Gih(X)⊤))·1n using Part 2. of Lemma I.6, which takes n1+o(1) time. Meanwhile, we compute f(X)X using Part 1. of Lemma I.6, which takes n1+o(1) time. We then have

, which takes nd time. Finally, we compute X⊤ d×n

f(X)X

,

diag( p1(X) · 1n) f(X)X n×d

diag( p1(X) · 1n) n×n

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

n×d

which takes nd2 time. Together, X⊤ p1(X)X

−X⊤ p2(X)X

takes d2 time.

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

d×d

d×d

Thus, we show that our framework can support causal attention masks.

## J Residual Connection

In this section, we discuss how to adapt our framework to the attention mechanism with the residual connection.

In Section J.1, we provide a formalized deﬁnition of the two residual connections used in the attention mechanism. In Section J.2, we argue that with the addition of the residual connection, the gradient over the attention mechanism can be computed in almost linear time n1+o(1) and the approximation error can be bound by 1/poly(n). In Section J.3, we use math induction to show that the gradient over the entire transformer with the residual connection can also be computed in almost linear time n1+o(1).

### J.1 Key concepts

Recall that in Deﬁnition 3.3, we have deﬁned Ti(X) ∈ Rn×d as the intermediate variable output by the i-th transformer layer. For simplicity, we use Ti to represent Ti(X) in the rest part of this section. Namely, we have

Ti = (gi ◦ Attni)(Ti−1)

Then, we consider adding the residual connection to our framework. Note that there are two residual connection operations in one transformer layer. We ﬁrst deﬁne the residual connection over the Attni in Deﬁnition J.1.

- Deﬁnition J.1 (Residual connection over Attni). If we have the below conditions,

- • Let Ti be deﬁned as Deﬁnition 3.3.
- • Let Attni be deﬁned as Deﬁnition C.3. We deﬁne Zi ∈ Rn×d as the output with the residual connection of Attni. Namely, we have

Zi = Ti−1 + Attni(Ti−1)

Then, we consider the second residual connection over the MLP layer gi, where we have the formal deﬁnition for this in Deﬁnition J.2.

- Deﬁnition J.2 (Residual connection over gi). If we have the below conditions,

- • Let the multi-layer transformer be deﬁned as Deﬁnition 1.3.
- • Let the intermediate variable Ti be deﬁned as Deﬁnition 3.3.
- • Let gi denote the components other than self-attention in the i-th transformer layer.

- • Let Zi ∈ Rn×d be deﬁned as Deﬁnition J.1.

Then Ti, the output of i-th layer transformer with the residual connection, should have the following form:

Ti = Zi + gi(Zi)

### J.2 Analysis of the residual connection

In the previous section, we have deﬁned the two residual connection operations.

In this section, we argue that if the gradient computation can be done in almost linear time without the residual connection, then with the addition of the residual connection, the gradient computation can also be completed in almost linear time.

- Lemma J.3 (Analysis of the residual connection). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let YR ∈ Rn×d and XR ∈ Rn×d denote the output and input of the residual connection, respectively.
- • Let H : Rn×d → Rn×d denote some layer in the transformer, such as MLP, Attn, etc.
- • Suppose the residual connection can be written as YR = XR + H(XR).
- • Assuming we have ddLY(X)

dH(XR)

∈ Rn×d, then we can calculate ddLY(X)

dXR in almost linear time n1+o(1).

[Figure 1196]

[Figure 1197]

[Figure 1198]

R

R

Then, we can show that,

- • ddLX(X)

[Figure 1199]

R

can be calculated in almost linear time n1+o(1).

- • If ddLY(X)

has 1/poly(n) approximation error, then the approximation error on ddLX(X)

is still 1/poly(n).

[Figure 1200]

[Figure 1201]

R

R

Proof. By the chain rule, we have

dL(X) dXR

=

[Figure 1202]

=

=

dYR dXR

dL(X) dYR

[Figure 1203]

[Figure 1204]

dL(X) dYR

dH(XR) dXR

(I +

)

[Figure 1205]

[Figure 1206]

dH(XR) dXR

dL(X) dYR

dL(X) dYR

+

[Figure 1207]

[Figure 1208]

[Figure 1209]

(32)

where the 1st step is from the chain rule, the 2nd step comes from basic calculus, the 3rd step is because of basic algebra.

, and ddLY(X)

dH(XR)

By the assumption, we already have ddLY(X)

dXR can be computed in almost linear time n1+o(1).

[Figure 1210]

[Figure 1211]

[Figure 1212]

R

R

and ddLY(X)

The addition operation between ddLY(X)

dH(XR)

dXR takes n · d time. Therefore, the overall running time for ddLX(X)

[Figure 1213]

[Figure 1214]

[Figure 1215]

R

R

is n1+o(1).

[Figure 1216]

R

Then, we consider the approximation error. By Eq. (32) and basic linear algebra, the approximation error will not be magniﬁed by more

than (n · dpoly(n) + 1). Since (n · dpoly(n) + 1)(1/poly(n)) = poly(n), the approximation error on ddLX(X)

can be bounded by 1/poly(n).

[Figure 1217]

R

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

### J.3 Analysis for the entire model with the residual connection

In the previous section, we have shown that, with the addition of the residual connection on a single component, the gradient computation time can still be done in almost linear time. We will apply this ﬁnding to the entire model. We begin by single layer proof.

- Lemma J.4 (Fast gradient computation for single-layer transformer with residual connection). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let X ∈ Rn×d be deﬁned as Deﬁnition C.3.
- • Suppose we have a single-layer transformer (see Deﬁnition 1.3).
- • Let the residual connection be deﬁned as Deﬁnition J.1 and J.2. Then, we can show that,
- • Part 1: running time. Our algorithm can approximate dLd(XX) in n1+o(1) time.

[Figure 1222]

- • Part 2: error bound. The approximation error of the single-layer transformer with the residual connection can be bounded by 1/poly(n). Namely, our algorithm output gr1 satisﬁes

dL(X) dX ∞ ≤ 1/poly(n)

gr1 −

[Figure 1223]

Proof. We use Ti to represent Ti(X) for simplicity. By the deﬁnition of Ti (see also Deﬁnition 3.3), we have the following equations

T0 = g0(X) Follow Deﬁnition J.1 and J.2, we have

Z1 = T0 + Attn1(T0) and

T1 = Z1 + g1(Z1) Then we calculate the gradient by the following steps:

- • Step 1: Calculate dLdT(X)

[Figure 1224]

1

. By the deﬁnition of L(X) (see also Deﬁnition 3.1), we have dLdT(X)

[Figure 1225]

1

can be computed in n · d time.

- • Step 2: Calculate dLdZ(X)

. By Lemma H.2, the assumption in Lemma J.3 is satisﬁed. Therefore, we have dLdZ(X)

[Figure 1226]

1

can be computed in almost linear time n1+o(1).

[Figure 1227]

1

- • Step 3: Calculate dLdT(X)

[Figure 1228]

0

. By Lemma E.11, the assumption in Lemma J.3 is satisﬁed. Hence, dLdT(X)

[Figure 1229]

0

can be computed in almost linear time. By Lemma E.11, the approximation error is 1/poly(n).

- • Step 4: Calculate dLd(XX). By Lemma H.2, dLd(XX) can be computed in n1+o(1). The approximation error is (n · d)(1/poly(n)) = (1/poly(n)).

[Figure 1230]

[Figure 1231]

To sum up, we can show that the overall running time for dLd(XX) is n1+o(1) and the approximation error is 1/poly(n).

[Figure 1232]

Let gr1 be the output of Step 4. Then we are done.

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

We now prove for multi-layer.

- Lemma J.5 (Fast gradient computation for multi-layer transformer with residual connection). If we have the below conditions,

- • Let L(X) be deﬁned as Deﬁnition 3.1.
- • Let X ∈ Rn×d be deﬁned as Deﬁnition C.3.
- • Let the residual connection be deﬁned as Deﬁnition J.1 and J.2.
- • Suppose we have a m-layer transformer (see Deﬁnition 1.3). Then, we can show that,
- • Part 1: running time. Our algorithm can approximate dLd(XX) in n1+o(1) time.

[Figure 1237]

- • Part 2: error bound. The approximation error of the m-layer transformer with the residual connection can be bounded by 1/poly(n). Namely, our algorithm output gr satisﬁes

dL(X)

gr −

dX ∞ ≤ 1/poly(n) Proof. We use math induction in this proof.

[Figure 1238]

- Step 1: Proof of a single-layer transformer. Firstly, by Lemma J.4, we have the statement holds for a single-layer transformer.
- Step 2: Assumption for k-layer transformer. Secondly, we assume for any k, for k-layer transformer model, we have

- • Part 1: running time. Our algorithm can approximate dLd(XX) in O(n1+o(1)) time.

[Figure 1239]

- • Part 2: error bound. The approximation error of the k-layer transformer can be bounded by 1/poly(n). Namely, our algorithm output g satisﬁes

g −

dL(X) dX ∞ ≤ 1/poly(n)

[Figure 1240]

- Step 3: Proof of (k + 1)-layer transformer. Thirdly, we consider the (k + 1)-layer transformer model. Let Fk denote a k-layer transformer with the residual connection.

Then, the entire model can be written as

(Fk ◦ g0)(X) By the deﬁnition of Ti, we have

T0 = g0(X) Then, by deﬁnition of Zi (see also Deﬁnition J.1), we have

Z1 = T0 + Attn1(T0) By Deﬁnition J.2, we have

T1 = Z1 + g1(Z1)

Without loss of generality, we assume that the additional transformer layer is added at the beginning of the model. Then, the (k + 1)-layer transformer model has the following structure:

Fk+1(X) = Fk(T1)

By the assumption for k-layer transformer, we have dLdT(X)

can be computed in almost linear time n1+o(1) and the approximation error can be bounded by 1/poly(n).

[Figure 1241]

1

We apply similar proof of Lemma J.4, then we can show that, we can compute dLd(XX) in almost linear time n1+o(1) and the approximation error can be bounded by 1/poly(n).

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

## K Multi-head Attention

Following the notation used in Section B.1, we use h to denote the number of heads, and dh = d/h to denote the dimension of each head.

- Deﬁnition K.1 (Multi-head attention). If we have the below conditions,

- • Let h denote the number of heads.
- • Let d denote the hidden dimension. Let dh = d/h denote the dimension of each attention head.
- • Let Q,K,V ∈ Rn×d be deﬁned as Deﬁnition C.3.
- • Let f(X) be deﬁned as Deﬁnition C.8.
- • Let s(X) be deﬁned as Deﬁnition C.10. The multi-head attention can be formalized as follows:
- • Step 1. Split the hidden dimension d of Q,K,V ∈ Rn×d into h parts. Then, for each l ∈ [h], we have Ql,Kl,Vl ∈ Rn×dh.
- • Step 2. For each l ∈ [h], calculate the attention matrix fl := Softmax(QlKl⊤/dh) ∈ Rn×n, and calculate the corresponding attention result sl := flVl ∈ Rn×dh.

- • Step 3. Concatenate sl ∈ Rn×dh together, then we have the ﬁnal multi-head attention output s ∈ Rn×d.

Then, we dive into the analysis of the gradient computation process over the attention mechanism with multi-head attention.

- Lemma K.2 (Analysis of the multi-head attention). If we have the below conditions,

- • Let Attn(X) be deﬁned as Deﬁnition C.3.
- • Let multi-head attention mechanism be deﬁned as Deﬁnition K.1.
- • Let Ym,Xm ∈ Rn×d denote the output and input of the multi-head attention, respectively. Then, we can show that,
- • ddLX(X)

[Figure 1247]

m

can be calculated in almost linear time n1+o(1).

- • If ddLY(X)

has 1/poly(n) approximation error, then the approximation error on ddLX(X)

is still 1/poly(n).

[Figure 1248]

[Figure 1249]

m

m

Proof. Following the notations used in Deﬁnition K.1, for l ∈ [h], we use sl ∈ Rn×dh to denote the output by each attention head. And we use s ∈ Rn×d to denote the concatenated version of the output of the multi-head attention.

By the chain rule and the deﬁnition of L(X) (see also Deﬁnition 3.1), we have

dL(X) dXm

=

[Figure 1250]

=

ds dXm

dL(X) dYm ·

dYm ds

[Figure 1251]

[Figure 1252]

[Figure 1253]

h

dL(X) dYm ·

dYm ds

dsl dXm

[Figure 1254]

[Figure 1255]

[Figure 1256]

l=1

where the 1st step is from the chain rule, the 2nd step comes from s ∈ Rn×d is the concatenated version of sl ∈ Rn×dh.

We calculate the gradient in the following steps:

- • Step 1: Calculate ddLY(X)

[Figure 1257]

m

. By the deﬁnition of L(X) (Deﬁnition 3.1), we have that ddLY(X)

[Figure 1258]

m

can be calculated in n · d time.

- • Step 2: Calculate ddLY(X)

[Figure 1259]

m

· ddYsm. Since we already have ddLY(X)

[Figure 1260]

[Figure 1261]

m

, by Lemma H.2, we have

dL(X)

[Figure 1262]

dYm · ddYsm can be computed in almost linear time n1+o(1).

[Figure 1263]

- • Step 3: Calculate ddLY(X)

dXm. For each l ∈ [h], by Lemma E.11, ddLY(X)

dsl

· ddYsm · ddXsml can be computed in n1+o(1). Since the number of heads h can be viewed as a constant here, it takes n1+o(1) time to compute the gradients on h heads.

h l=1

· ddYsm

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

m

m

Therefore, the overall running time for ddLX(X)

is n1+o(1).

[Figure 1270]

m

Then, we consider the error bound. By assumption, there is 1/poly(n) approximation error on ddLY(X)

. For each l ∈ [h], the approximation error will not be magniﬁed by more than n2 · d · dh · poly(n) on ddLY(X)

[Figure 1271]

m

· ddYsm · ddXsl

. Then, since there is total h heads, the approximation error on ddLX(X)

[Figure 1272]

[Figure 1273]

[Figure 1274]

m

m

can be bound by h · n2 · d · dh · poly(n) · (1/poly(n)) = 1/poly(n)

[Figure 1275]

m

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

Similar to the proof of Lemma H.3 and H.4, we apply Lemma K.2 to deal with the multi-head

attention in each transformer layer. Then, we can show that dLd(XX) can be computed in almost linear time n1+o(1) and the approximation error can be bounded by 1/poly(n).

[Figure 1280]

