## CoCA: Fusing Position Embedding with Collinear Constrained Attention in Transformers for Long Context Window Extending

Shiyi Zhu, Jing Ye, Wei Jiang, Siqiao Xue, Qi Zhang, Yifan Wu, Jianguo Li Ant Group {zhushiyi.zsy, qianye.yj, shouzhi.jw, lijg.zero}@antgroup.com

# arXiv:2309.08646v3[cs.LG]28Feb2024

### Abstract

Self-attention and position embedding are two key modules in transformer-based Large Language Models (LLMs). However, the potential relationship between them is far from well studied, especially for long context window extending. In fact, anomalous behaviors harming long context extrapolation exist between Rotary Position Embedding (RoPE) and vanilla self-attention unveiled by our work. To address this issue, we propose a novel attention mechanism, CoCA (Collinear Constrained Attention). Specifically, we enforce a collinear constraint between Q and K to seamlessly integrate RoPE and self-attention. While only adding minimal computational and spatial complexity, this integration significantly enhances long context window extrapolation ability. We provide an optimized implementation, making it a dropin replacement for any existing transformerbased models. Extensive experiments show that CoCA performs extraordinarily well in extending context windows. A CoCA-based GPT model, trained with a context length of 512, can seamlessly extend the context window up to 32K (60×), without any fine-tuning. Additionally, by dropping CoCA in LLaMA7B, we achieve extrapolation up to 32K within only 2K training length. Our code is publicly available at: https://github.com/codefuseai/Collinear-Constrained-Attention

### 1 Introduction

In the seminal work of Transformer (Vaswani et al., 2017), it claims the ability of "extrapolating to sequence length longer than the ones encountered during training". This is an ideal hypothesis, but actually not work in practice for vanilla Transformer. Several subsequent works, collectively known as long context extrapolation, have delved into exploring the capabilities of large language models (LLMs) trained within the range of [1,N − 1] to effectively extend the testing sequence ≥ N.

[Figure 1]

Figure 1: Perplexity evaluation on 100 PG-19 documents with a sliding window strategy (Stride = 512). The perplexity of RoFormer (Su et al., 2024) sharply exceeds 1000 beyond its training length, while CoCA maintains a low plateau even at 60 × its training length. ALibi (Press et al., 2022) encounters Out of Memory (OOM) issues for input Nmax > 8000 due to flash-attention (Dao et al., 2022) incompatibility, we suppose it maintains perplexity for Nmax > 8000.

Existing studies primarily focus on attention kernel (Beltagy et al., 2020; Ding et al., 2023; Han et al., 2023) or position embedding (Huang et al.,

- 2023), often neglecting the intrinsic relationship between the two key modules. Attention bias is an alternative to the explicit encoding of positional information. ALibi (Press et al., 2022) and KERPLE (Chi et al., 2022), incorporate heuristic and compositional triangle kernel-based negative causal attention bias, respectively. While these approaches effectively manage to maintain low perplexity, they fall short in capturing long-range dependencies due to introducing local hypotheses to context tokens. Another branch of methods involve simply scaling Rotary Position Embedding (RoPE) (Su et al.,
- 2024) to extrapolate the inference context length with minimal or no fine-tuning. For instance, Position Interpolation (PI) (Chen et al., 2023) employs linear scaling on each position number from n to n/k, where k is the extrapolation ratio. NTK-aware Scaled RoPE (bloc97, 2023) and Dynamic-NTK

(Emozilla, 2023) combine high-frequency extrapolation and low-frequency interpolation. They scale the basis in RoPE upon the sequence length to adapt to the unseen position indices. However, these methods primarily alleviate the problem of modeling the rotation angles in out-of-distribution positions, without recognizing the intrinsic correlation between attention matrices and rotation angles. Therefore, these methods still suffer from a limited context window extending ratio.

Here, we present a new perspective on the relationship between position embedding (with a focus on RoPE) and the self-attention mechanism. In a nutshell, RoPE utilizes a rotation matrix to encode absolute positions while simultaneously incorporating explicit relative position dependencies within the self-attention formulation (Su et al., 2024). It is designed based on the relative angular difference between the queries (Q) and keys (K). However, latent relationships exist between Q and K, as these two matrices are directly multiplied. We demonstrate that incorrect initialization of the angle between Q and K in RoPE leads to undesirable behavior around the context window boundary, harming its performance for context extrapolation.

To address this undesirable behavior , we propose an innovative architecture called Collinear Constrained Attention (CoCA). Specifically, we enforce a collinear constraint between Q and K by initializing the angle between every two hidden dimensions in the Q and K vectors to 0. This allows for a seamless integration of RoPE and selfattention. The model architecture and comparison with RoFomer (Su et al., 2024) is illustrated in Figure 2.

Extensive experiments show that a CoCA-based GPT model, trained within 512 context length, seamlessly extends the context window up to 32K (60x) without perplexity divergence. A comprehensive comparison between our method and existing methods is presented in Figure 1. Furthermore, it enhances long-context retrieval ability, achieving a passkey retrieval accuracy of 50%+ even when extrapolating to 16x longer than its training context length by applying Dynamic-NTK (Emozilla, 2023). Additionally, by dropping CoCA in LLaMA-7B, we achieve extrapolation up to 32K within only 2K training length.

Our main contributions can be summarized as follows:

• We unveil undesirable context boundary be-

havior resulting from the absence of modeling the relationship between position embeddings and self-attention.

- • To tackle the undesirable context boundary behavior, we propose Collinear Constrained Attention (CoCA) to seamlessly integrate the position embeddings and self-attention, achieving excellent long context window extrapolation performance.
- • CoCA extends its context window from 512 to 32K without fine-tuning, achieving over 50% accuracy even when 16 × longer than its training length. Using CoCA in LLaMA-7B, we achieve extrapolation up to 32K within just 2K training length.
- • CoCA introduces minimal computational and spatial complexity compared to vanilla selfattention. We provide an optimized implementation of CoCA, making it able to be a seamless drop-in replacement for existing transformer-based models.

### 2 Method

In this section, we describe our proposed Collinear Constrained Attention (CoCA). We begin with introducing the background theory of RoPE (Su et al., 2024) in Section 2.1, and then analyze the anomalous behaviors between the attention matrices and RoPE in Section 2.2. Finally, we introduce the proposed method CoCA in section 2.3 and derive a slack constraint version of CoCA in Section 2.4, respectively.

#### 2.1 Rotary Position Embedding

Position embedding is a crucial component in transformer-based models. Here we focus on Rotary Position Embedding (RoPE) (Su et al., 2024), which is widely used by LLMs including LLaMA (Touvron et al., 2023a), LLaMA-2 (Touvron et al., 2023b), GPT-NeoX (Black et al., 2022) and Qwen (Bai et al., 2023). Suppose the positional index is an integer n ∈ [1,N], and the corresponding input vector x = [x0,x1,...,xd−1]T, where N is the sequence length, d is the dimension of the attention head. RoPE defines a vector-valued complex function f(x,n):

f(x, n) = [(x0 + ix1)einθ0, (x2 + ix3)einθ1,

. . . , (xd−1 + ixd)einθd/2−1]T, where θj = B−2j/d,

(1)

[Figure 2]

- Figure 2: Architecture comparison between RoFormer and CoCA. (a) RoFormer; (b) CoCA; (c) The implementation detial of K in CoCA. Q, T, and V are produced using projection matrices identical to those employed in the vanilla self-attention. T undergoes a halving operation, with the other half being duplicated. K is then computed as the element-wise product of Q and T,

adhering to a collinear constraint with Q. Note that kn ∈ RN×d, where n ∈ [1, N] is the positional index of key, d is the head dimension, N is the sequence length.

in this paper, the base B = 10,000.

After the application of RoPE, the transformed vectors for query (q) and key (k) become f(q,m) and f(k,n), respectively. Here, m,n ∈ [0,N] represent the positional indices of q and k. The attention operation is computed as the dot product between f(q,m) and f(k,n), defined as follows:

a(m, n) = Re(⟨f(q, m), f(k, n)⟩)

 

 

d/2−1

(q2j + iq2j+1)(k2j − ik2j+1)ei(m−n)θj

= Re

j=0

d/2−1

[(q2jk2j + q2j+1k2j+1) cos((m − n)θj)

=

j=0

+ (q2jk2j+1 − q2j+1k2j) sin((m − n)θj)]

(2)

The attention score a(m − n) depends on the relative position (m − n).

- 2.2 Anomalous Behavior between RoPE and Attention Matrices

After applying RoPE, the attention score a(m − n) can be interpreted as the sum of d/2 inner products of complex numbers, as illustrated in Equation (2). For any pair of qj = (q2j,q2j+1) and kj = (k2j,k2j+1), which is the 2-dimensional slicing of q (or qm) and k (or kn), we introduce

the initial angle Θj between them, measured counterclockwise from kj to qj in the complex plane. Throughout our analysis, we keep the position of kj fixed, systematically rotating qj to comprehensively examine their relative positions. The final angle between qj and kj is represented as θ(qj,kj) = Θj + (m − n)θj, where m and n are positional indices of qj and kj.

In this concept, the attention score can be formulized as:

d/2−1

|qj||kj| cos(θ(qj, kj)) (3)

a(m, n) =

j=0

Refer to Figure 3 for a visual representation of this concept for any individual j ∈ [0,d/2] in the 2-D subspace. There are four distinct scenarios between qj and kj after rotation.

(1) Scenario (b) and (c): When m > n and Θj ≤ π, or m < n and Θj > π, the value of cos(θ(qj,kj)) between qj and kj decreases with the expanding distance between m and n. In these 2 scenarios, no anomalous behavior is observed, as the attention score naturally decreases with the positional distance. This trend persists until the relative angle θ(qj,kj) rotates beyond the boundary of π.

[Figure 3]

- Figure 3: Anomalous behavior of RoPE in 2-D plane. The inner product of vectors qj and kj is contingent upon the relative angle θ(qj, kj), defined as Θj + (m − n)θj. Here, Θj represents the initial angle, and (m − n)θj signifies the position-dependent rotation angle. (a) m < n and Θj ≤ π. (b) m > n and Θj ≤ π. (c) m < n and Θj > π. (d) m > n and Θj > π.

(2) Scenario (a) and (d): When m < n and Θj ≤ π, or m > n and Θj > π, intriguing phenomena emerge. As the distance between m and n grows, the value of cos(θ(qj,kj)) between qj and kj paradoxically increases. This anomaly has a notable impact on attention scores, particularly affecting the τ closest tokens. In this context, τ is defined as Θj/θj for scenario (a) and (2π−Θj)/θj for scenario (d). Consequently, attention scores for these tokens are abnormally diminished.

For bidirectional language models, all four cases may occur. For causal models, only scenario (b) and (d) manifest, as m consistently exceeds n.

The attention score a(m − n) is the sum of d/2 inner-products, one of them turns out anomalous may be insignificant, however, experiments confirmed this significance. Further analysis of this rotary borders anomalous behaviour is discussed in Appendix D.2.

#### 2.3 Collinear Constrained Attention

To tackle the anomalous behavior between RoPE and attention matrices, we propose a novel approach called Collinear Constrained Attention (CoCA). Specifically, by applying a collinear constraint to any pair of qj = (q2j,q2j+1) and kj = (k2j,k2j+1), we seamlessly integrate RoPE into self-attention mechanism, achieving long context extrapolation.

To formalize this, considering a sequence of N input tokens SN = {wn}Nn=1, with corresponding word embeddings EN = {xn}Nn=1, where xn ∈ Rd is the d-dimensional word embedding vector of token wn without position information. First, the queries qm are obtained:

qm = WQxm, ∀m ∈ [1, N] (4)

Next, we derive the keys kn with collinear constraints. This begins with the introducing of the constraint coefficient tn for each token position n, as depicted in Equation (5).

tn = WTxn, ∀n ∈ [1, N] (5)

Next, Equation (6) imposes the collinearity condition on the coefficients t2j and t2j+1, where

- tn = [t0,t1,...,td−1]T, ensuring that each pair is identical. This step effectively duplicates each 2dimensional segment of the tensor.

t2j = t2j+1, ∀j ∈ [0, d/2 − 1] tn = Relu(tn)

(6)

Subsequently, the keys are calculated as shown in Equation (7), where kn are represented by the element-wise multiplication of Q = (q1,...,qN) and tn. This results in an expansion of dimensionality, as kn ∈ RN×d now includes an additional sequence length dimension. We address potential memory pressure by optimizing tensor contractions, ensuring no net increase in memory consumption. For an in-depth analysis, please refer

- to Appendix C. kn = Q ⊙ tn = (q1 ◦ tn, ..., qN ◦ tn) (7)

After that, we apply RoPE on Q and K, with the function f detailed in Equation (1).

f(qm) = f(qm, m) f(kn) = f(Q ⊙ tn, n) = f(Q, n) ⊙ tn

(8)

Finally, the attention score of CoCA would be:

a(m, n) = Re(⟨f(qm, m), f(qm, n) ◦ tn⟩) (9)

Equation (9) illustrates the additional dimension of the keys in our CoCA mechanism. Specifically, it maps the index of each query to the additional dimension, establishing a collinear relationship between the n-th key and the m-th query. This is a critical aspect of our method.

#### 2.4 Slacking the Constraint on Query

In Section 2.3, we present a theoretically precise solution for CoCA. However, practical implementation faces challenges due to the complexity of

O(N2d) when storing f(Q,n). To address this issue, we provide a dual implementation with O(Nd) complexity in this section and prove their equivalence.

Theorem 1. (Dual implementation of CoCA) For any attention score defined in Equation (9), there exists an equivalent form as follows:

a(m, n) = Re(⟨f(qm, m), qm ◦ f(tn, n)⟩) (10)

with constraint:

q2j = q2j+1, ∀j ∈ [0, d/2 − 1] (11)

Proof: The proof consists of two steps.

- Step 1. We prove that, by imposing the constraint q2j = q2j+1, ∀j ∈ [0,d/2 − 1], Re(⟨f(qm,m),qm ◦ f(tn,n)⟩) is equivalent to Re(⟨f(qm,m),f(qm,n) ◦ tn⟩).

To see this, we calculate the difference between f(qm,n) ◦ tn and qm ◦ f(tn,n):

f(qm, n) ◦ tn − qm ◦ f(tn, n)

=



 

- t0(q0 cos nθ0 − q1 sin nθ0)
- t1(q0 sin nθ0 + q1 cos nθ0)

. . . td−2(qd−2 cos nθd/2−1 − qd−1 sin nθd/2−1) td−1(qd−2 sin nθd/2−1 + qd−1 cos nθd/2−1)



 

−



 

- q0(t0 cos nθ0 − t1 sin nθ0)
- q1(t0 sin nθ0 + t1 cos nθ0)

. . . qd−2(td−2 cos nθd/2−1 − td−1 sin nθd/2−1) qd−1(td−2 sin nθd/2−1 + td−1 cos nθd/2−1)



 

(12)

Recall that t2j = t2j+1,∀j ∈ [0,d/2 − 1] (see Equation (6)), Equation (12) is equivalent to:

f(qm, n) ◦ tn − qm ◦ f(tn, n)

=



 

- t0(q0 − q1) sin nθ0
- t1(q0 − q1) sin nθ0

. . . td−2(qd−2 − qd−1) sin nθd/2−1 td−1(qd−2 − qd−1) sin nθd/2−1



 

(13)

Clearly, if we impose the constraint q2j = q2j+1,∀j ∈ [0,d/2 − 1], the vector in Equation

(13) becomes null and we deduce that:

f(qm, n) ◦ tn − qm ◦ f(tn, n) = 0 (14)

Consequently, with the constraint q2j = q2j+1,∀j ∈ [0,d/2 − 1], we have:

Re(⟨f(qm, m), qm ◦ f(tn, n)⟩)

= Re(⟨f(qm, m), f(qm, n) ◦ tn⟩)

(15)

- Step 2. We further demonstrate that, q2j = q2j+1,∀j ∈ [0,d/2 − 1] is in fact

a redundant constraint when calculating Re(⟨f(qm,m),f(qm,n) ◦ tn⟩). To verify this, we expand the inner product:

Re(⟨f(qm, m), f(qm, n) ◦ tn⟩)

d/2−1

[(q22jt2j + q22j+1t2j+1) cos((m − n)θj)

=

j=0

+ (q2jq2j+1t2j − q2j+1q2jt2j+1) sin((m − n)θj)]

(16)

Recall again t2j = t2j+1,∀j ∈ [0,d/2 − 1], we have

Re(⟨f(qm, m), f(qm, n) ◦ tn⟩)

d/2−1

t2j[(q22j + q22j+1) cos((m − n)θj)]

=

j=0

d/2−1

t2j|qj|2 cos((m − n)θj)

=

j=0

(17)

This implies that Re(⟨f(qm,m),f(qm,n) ◦ tn⟩) depends solely on the magnitude of qj = (q2j,q2j+1) in 2-D subspace, demonstrating the independence of the relationship between q2j and q2j+1. Refer to Appendix D.3 for the rigorous proof.

Now we conclude that, with the constraint q2j = q2j+1,∀j ∈ [0,d/2 − 1], Re(⟨f(qm,m),qm ◦ f(tn,n)⟩) is equivalent to Re(⟨f(qm,m),f(qm,n) ◦ tn⟩) with no constraint on query.

| |
|---|

By removing q2j = q2j+1 constraint, we designate this modified version as CoCA-Slack. The mathematical definition is provided in Appendix D.4.

### 3 Experimental Setting

This section provides an overview of the experimental setup, including details regarding the training data utilized and the baseline models employed to evaluate the effectiveness of the proposed method.

#### 3.1 Training Data

Our model undergoes training on a combination of datasets, including the Pile training dataset (Gao et al., 2020), BookCorpus (Zhu et al., 2015), and the Wikipedia Corpus (Foundation, 2021). Additionally, we integrate manually collected opensource code from GitHub repositories with at least 1 star. From these datasets, we derive a sample of approximately 50B tokens, maintaining a composition of 75% text and 25% code.

#### 3.2 Model Variants

To evaluate the effectiveness of our proposed approach, we train 3 models from scratch under identical experimental settings, including ALibi (Press et al., 2022), RoFomer (Su et al., 2024), and RoFormer+CoCA. All models share common specifications, featuring a size of 350M, 24 layers, a hidden dimension of 1024, 16 attention heads, and a maximum sequence length of 512. The key distinctions among them lie in variations in self-attention mechanisms and position embeddings. The implementation is optimized based on EleutherAI GPTNeoX1. Training a model from scratch demands substantial computational resources. Therefore, we also conduct experiments involving fine-tuning existing LLMs with a drop-in CoCA module. For this purpose, we utilize the LLaMA-7B model (Touvron et al., 2023a), which was trained with a context length of 2,048. Additionally, we employ dynamic-NTK for all the above models.

In summary, our comparison models are categorized as follows: ALibi, RoFormer, RoFormer+CoCA, RoFormer+dynamic NTK, and RoFormer+dynamic NTK & CoCA, all falling under the training from scratch category. Meanwhile, LLaMA-7B, LLaMA-7B+CoCA, LLaMA7B+dynamic NTK, and LLaMA-7B+dynamic NTK & CoCA belong to the fine-tuning LLM with drop-in CoCA category.

#### 3.3 Implementation Detials

Pre-training Procedure We train all models using the next token prediction objective. We use AdamW (Loshchilov and Hutter, 2017) with β1 = 0.9 and β2 = 0.95. The learning rate follows a linear warm-up of 1% of total steps, starting from 1e-7. Subsequently, the learning rate is adjusted to 1e-4 with linear decay, eventually reaching 1e-5. The training utilizes 8 A100 GPUs, with a global batch size of 256 and 2 gradient steps accumulation, taking approximately 96 hours for 2 epochs.

Fine-tuning Procedure To integrate CoCA in LLaMA, we employ a three-stage fine-tuning strategy: (1) only updating the K projection (7% of parameters). This stage aims to reconstruct the K projection in CoCA. By freezing the other parameters, we maintain attention scores as closely as possible to those of vanilla self-attention. (2) updating the QKV projection (21% of parameters). This stage aims to address intrinsic over-fitting in vanilla

1https://github.com/EleutherAI/gpt-neox/tree/v2.0

self-attention caused by undesired behaviors between RoPE and attention matrices. (3) fine-tuning all parameters. Each stage involves 15K steps, totaling 7.5B tokens (22B tokens overall), using the next token prediction objective. The training length of LLaMA-7B + CoCA remains at 2,048 as in the original model. All experiments are conducted with 32 A100 GPUs, setting a per-device batch size to 8 without gradient accumulation.

### 4 Experiment Results

We conducted experiments to shed light on the following reasonable doubts:

- • Can our new attention mechanism CoCA improve the long context extrapolation performance of existing models?
- • Can combining CoCA with other extending methods for RoPE effectively solve the three types of rotational boundary problems discussed in Appendix D.2?

#### 4.1 Long Sequence Language Modeling

We evaluate the long sequence language modeling performance of both our model and baseline models on the test splits of the PG-19 dataset (Rae et al., 2020). For this evaluation, we randomly select a subsample comprising 100 documents, each containing at least 32,768 SentencePiece (Kudo and Richardson, 2018) tokens. We then truncate each test document to its initial 32,768 tokens. The evaluation involves calculating perplexity across different context window sizes using a sliding window approach, as described by (Press et al., 2022), with a stride of 512. The perplexity results for both our models and baselines are presented in Table 1 and Figure 1.

Based on our experiments, the evaluation results indicate that models combined with CoCA exhibit significantly improved perplexity with longer inference sequence length. For pre-trained models, by increasing the context window size from 512 (training context window size) to 32k, the perplexity of CoCA only increases from 20.11 to 171.63, whereas the perplexity of RoFormer becomes inf. Additionally, by increasing the context window size from 2K to 32K, the perplexity of finetuned LLaMA-7B+CoCA only increases 21.68, while LLaMA-7B with other extending methods increases more than 100. In general, we observe a consistent trend of CoCA achieving better perplexity with longer context windows. This suggests

Evaluation Context Window Size (Perplexity ↓) 512 1024 2048 4096 8192 16k 32k

Method

Training model from scratch ALibi 18.69 21.27 28.20 35.66 37.03 OOM OOM RoFomer 19.66 411.50 3276.00 3026.00 3028.00 inf inf

+ dynamic NTK 19.66 22.30 38.00 75.75 138.13 370.75 380.75 + CoCA 20.11 33.47 69.06 113.19 157.38 141.00 171.63 + dynamic NTK & CoCA 20.11 20.81 25.88 34.16 55.75 89.31 101.13

Fine-tuning LLM with drop-in CoCA LLaMA-7B 9.25 7.56 7.30 9673.14 inf inf inf

+ dynamic NTK 9.25 7.56 7.30 9.40 14.40 63.62 133.87

+ CoCA 9.91 8.49 8.27 24.23 42.00 23.83 29.95 + dynamic NTK & CoCA 9.91 8.49 8.27 8.61 9.56 11.10 13.98

Table 1: Evaluation perplexity on 100 PG-19 documents using sliding window (S = 512) strategy. Dynamic-NTK is employed without fine-tuning. The best result is highlighted in bold.

that CoCA has a more robust position embedding, enabling it to handle long context more effectively.

In contrast, we observe that models extended through the direct application of dynamic NTKaware Scaled RoPE exhibit a larger increase in perplexity at longer sequences. The perplexity of both RoFormer+dynamic NTK and LLaMA7B+dynamic NTK remains significantly higher than that combining CoCA. This difference becomes more pronounced as the sequence length increases. When the inference sequence length reaches 32k, the perplexity of RoFormer+dynamic NTK increases to 380.75, while the result for RoFormer+CoCA is only 171.63. Similarly, the perplexity of LLaMA-7B+dynamic NTK reaches 133.87, whereas LLaMA-7B+CoCA is only 29.95.

It is worth noting that the model achieves the best performance when both dynamic NTK and CoCA are combined. Particularly, LLaMA-7B+dynamic NTK & CoCA consistently maintains a very low perplexity. Even when the inference sequence length has reached 32k (16 × longer than the training length), the perplexity is only 13.89. This indicates that combining CoCA with other extending methods for RoPE can effectively address the three types of rotational boundary problems, achieving robust long-text extrapolation modeling capabilities.

#### 4.2 Long Context Retrieval

Perplexity evaluates the performance of language model in predicting the next token. However, it is insufficient for a comprehensive assessment of the effective context window size. To address this, we conducted experiments using a passkey retrieval

task (Mohtashami and Jaggi, 2023) to evaluate our method and baselines. The task involves identifying and retrieving a randomly hidden passkey within a lengthy document. More details of task definition and test sample generation settings can be found in Appendix B.1. Table 2 illustrates the accuracy of all tested models and their variants.

It is evident that ALibi exhibited failures when tested on sequences that were 1× longer than its training length, attributed to its local hypothesis. In contrast, our model consistently demonstrated superior accuracy. RoFormer+dynamic NTK & CoCA maintained a 50% accuracy, even with the test sequence length expanded to 16× its training length. Similarly, LLaMA-7B+dynamic NTK & CoCA still maintained a 30% accuracy when the test length was up to 32K.

4.3 Impact of Strict and Slack Constraint on Q

As mentioned in Section 2.4, we implement a slack version of CoCA, referred to as CoCA-Slack. In this section, under the same experimental settings, we implement two versions of CoCA based on RoFormer-350M, labeled as CoCA-Slack and CoCA-Strict. The comparison results between them are shown in Table 3.

We observe that the CoCA-Strict and CoCASlack models exhibit similar performance in long sequence language modeling, as evidenced by comparable perplexity results. However, in the passkey retrieval task, contrary to our initial expectations, the CoCA-Strict model produces significantly lower results. This unexpected outcome suggests that models with a slack constraint may

Evaluation Context Window Size (Accuracy↑) 512 1024 2048 4096 8192 16k 32k

Method

Traning model from scratch ALibi 0.82 0.65 0.28 0.18 0.12 OOM OOM RoFomer 0.99 0.53 0.30 0.18 0.04 0.02 0.04

- + dynamic NTK 0.99 1.00 0.95 0.70 0.41 0.16 0.06

- + CoCA 1.00 0.64 0.33 0.19 0.06 0.02 0.04

- + dynamic NTK & CoCA 1.00 1.00 0.96 0.89 0.50 0.23 0.08 Fine-tuning LLM with drop-in CoCA

LLaMA-7B 1.00 1.00 1.00 0.61 0.21 0.07 0.09 + dynamic NTK 1.00 1.00 1.00 0.81 0.26 0.06 0.03 + CoCA 1.00 1.00 1.00 0.71 0.28 0.11 0.10

- + dynamic NTK & CoCA 1.00 1.00 1.00 1.00 0.85 0.51 0.30

Table 2: Long context retrieval performance on passkey retrieval task. The best result is highlighted in bold.

###### Method 512 1024 2048 4096 8192 16384 32768

Performance on Long Sequence Modeling (Perplexity)

- CoCA-Slack 20.11 19.02 24.92 40.53 68.38 92.75 103.44

ntk-2

CoCA-Strict +0.07 +0.61 -1.58 -4.03 +15.37 +12.38 +1.94

- CoCA-Slack 20.11 20.81 25.88 34.16 55.75 89.31 101.13

ntk-4

CoCA-Strict +0.07 -0.49 -0.66 -0.88 +3.16 -18.25 -2.57 CoCA-Slack 20.11 23.66 29.05 37.47 55.5 88.88 111.38

ntk-8

CoCA-Strict +0.07 -1.74 -0.64 +1.16 +0.03 +0.5 +0.31 Performance on Long Context Retrieval (Passkey Accuracy)

- CoCA-Slack 1.0 0.99 0.94 0.77 0.47 0.27 0.15

ntk-2

CoCA-Strict +0.0 -0.12 -0.3 -0.42 -0.34 -0.22 -0.07

- CoCA-Slack 1.0 1.0 0.96 0.89 0.5 0.23 0.08

ntk-4

CoCA-Strict +0.0 -0.11 -0.38 -0.46 -0.38 -0.19 -0.02 CoCA-Slack 1.0 0.98 0.99 0.85 0.5 0.11 0.02

ntk-8

CoCA-Strict +0.0 -0.05 -0.34 -0.51 -0.4 -0.07 -0.01

- Table 3: Comparison results for the Strict and Slack Constraints of Q in our proposed CoCA module. Superior performance to CoCA-Slack is indicated by the green color, while inferior performance is signified by the red color. The perplexity of the strict and slack models is comparable, whereas the strict model achieved lower accuracy in the passkey retrieval task.

offer additional performance advantages, such as a larger effective context window size.

Understanding the reasons behind the superiority of slack constraints will be a key focus of our future work. In this regard, we provide some theoretical insights in Appendices D.3 and D.4. These insights aim to shed light on the underlying mechanisms that contribute to the observed differences and lay the groundwork for a more comprehensive analysis in subsequent research.

### 5 Conclusion

In this paper, we introduce Collinear Constrained Attention (CoCA), a novel approach that integrates position embedding with the self-attention mechanism. This innovation addresses undesired behaviors occurring around the context window boundary, which stem from discrepancies between RoPE

and attention matrices. To the best of our knowledge, we are the first to analyze the initial angles between queries and keys in the self-attention mechanism, which gives rise to anomalous phenomena in RoPE. Furthermore, we deduce a slack constraint for our implementation of CoCA. Extensive experiments demonstrate that incorporating CoCA into existing models significantly enhances performance in both long sequence language modeling and long context retrieval tasks. Additionally, the simultaneous integration of CoCA with other extended RoPE methods (e.g., dynamic-NTK) effectively mitigates three types of rotation boundary issues, resulting in remarkably improved capabilities for long context extrapolation.

### Limitations

Our current approach, CoCA, has thus far undergone exclusive validation on RoPE. Experimental results demonstrate that CoCA enhances the long-context extrapolation performance of LLMs and further augments other extension methods by addressing rotational boundary issues. However, questions arise regarding its applicability to more general methods. While the effectiveness of slack position embedding (SPE) is evident, a deeper understanding of the underlying reasons for its superior performance necessitates further investigation.

### References

Daniel G. a. Smith and Johnnie Gray. 2018. opt_einsum - a python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 3(26):753.

Jinze Bai, Shuai Bai, Yunfei Chu, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Sidney Black, Stella Biderman, Eric Hallahan, et al. 2022. GPT-NeoX-20B: An open-source autoregressive language model. In Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pages 95– 136, virtual+Dublin. Association for Computational Linguistics.

bloc97. 2023. Ntk-aware scaled rope allows llama models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. ArXiv, abs/2306.15595.

Ta-Chung Chi, Ting-Han Fan, Peter J. Ramadge, and Alexander Rudnicky. 2022. KERPLE: kernelized relative positional embedding for length extrapolation. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/ opencompass.

Tri Dao, Daniel Y. Fu, Stefano Ermon, et al. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems.

Jiayu Ding, Shuming Ma, Li Dong, et al. 2023. Longnet: Scaling transformers to 1,000,000,000 tokens. arXiv preprint arXiv:2307.02486.

Emozilla. 2023. Dynamically scaled rope further increases performance of long context llama with zero fine-tuning.

Wikimedia Foundation. 2021. Wikimedia downloads. Leo Gao, Stella Rose Biderman, Sid Black, et al. 2020.

The pile: An 800gb dataset of diverse text for language modeling. ArXiv, abs/2101.00027.

Chi Han, Qifan Wang, Wenhan Xiong, et al. 2023. Lm-infinite: Simple on-the-fly length generalization for large language models. arXiv preprint arXiv:2308.16137.

Yunpeng Huang, Jingwei Xu, Zixu Jiang, et al. 2023. Advancing transformer architecture in long-context large language models: A comprehensive survey. arXiv preprint arXiv:2311.12351.

Taku Kudo and John Richardson. 2018. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018: System Demonstrations, Brussels, Belgium, October 31 - November 4, 2018, pages 66–71. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2017. Fixing weight decay regularization in adam. ArXiv, abs/1711.05101.

Amirkeivan Mohtashami and Martin Jaggi. 2023. Landmark attention: Random-access infinite context length for transformers. CoRR, abs/2305.16300.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. Yarn: Efficient context window extension of large language models. CoRR, abs/2309.00071.

Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. 2020. Compressive transformers for long-range sequence modelling. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Sebastian Ruder, Matthew E. Peters, Swabha Swayamdipta, and Thomas Wolf. 2019. Transfer learning in natural language processing. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2,

2019, Tutorial Abstracts, pages 15–18. Association for Computational Linguistics.

Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. 2023. A length-extrapolatable transformer. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 14590–14604. Association for Computational Linguistics.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. 2023a. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971.

Hugo Touvron, Louis Martin, Kevin R. Stone, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, et al.

2023. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.

Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pages 19–27. IEEE Computer Society.

### A Related Work

Existing researches are mainly focused on the submodule of attention kernel or position embedding (Huang et al., 2023). In the following sections, we will separately introduce works on these two aspects: Section A.1 primarily addresses the former, while Section A.2 delves into the latter.

#### A.1 Efficient Attention Mechanisms

Several works aim to implement efficient attention mechanisms with reduced computational demands, even achieving linear complexity. This enables extending the effective context length boundary of LLMs during inference by directly increasing Lmax in the pre-training stage (Ding et al.,

2023; Mohtashami and Jaggi, 2023). Noteworthy approaches include Longformer (Beltagy et al., 2020), utilizing slide window attention, and models such as StreamingLLM (Xiao et al., 2023) and LM-Infinite (Han et al., 2023), which leverage a global-local attention mechanism. These variants have achieved success to a certain extent, but still face issues we unveiled in this work when using RoPE as their positional encoding method. A.2 Extrapolative Position Embedding

#### Methods

Extrapolative position embedding methods aim to enhance the length generalization capability of LLMs.

#### A.2.1 Attention Bias

In seeking alternatives to the explicit encoding of positional information, researchers have explored the integration of attention bias to capture the sequential and temporal nuances inherent in natural language. Early approaches, such as T5 (Ruder et al., 2019), incorporate learnable attention bias. However, these methods do not explicitly address the challenge of length extrapolation. ALibi (Press

- et al., 2022) introduces a negative causal attention bias in a heuristic manner. Extending the ALiBi-style attention bias, KERPLE (Chi et al.,

- 2022) treats it as a composition triangle kernel for self-attention and modifies style Xpos (Sun

et al., 2023) by integrating it with RoPE. While these approaches effectively manage to maintain low perplexity levels, they fall short in capturing long-range dependencies due to introducing local hypotheses to context tokens.

A.2.2 Extend RoPE

Besides, various strategies have been explored to extend RoPE (Su et al., 2024), a commonly employed positional encoding method in popular LLMs. Recent approaches involve simply scaling it to extrapolate the inference context length with minimal or no fine-tuning. For instance, Position Interpolation (PI) (Chen et al.,

- 2023) applies linear scaling on each position number from n to n/k, densifying the representation space to extend the farthest length boundary by k times. Other approaches, such as NTK-aware Scaled RoPE (bloc97, 2023) and Dynamic-NTK (Emozilla, 2023), combine high-frequency extrapolation and low-frequency interpolation. These training-free methods require limited code changes

during inference (Peng et al., 2023). However, these methods aim solely at alleviating the problem of modeling the rotation angles in out-ofdistribution (OOD) positions without recognizing the intrinsic correlation between attention matrices and rotation angles. Therefore, these methods still suffer from a limited context window extending ratio.

Previous methods independently investigate selfattention and position embedding without considering their intrinsic relationship, especially for the widely used RoPE method.

### B Additional Experiment

#### B.1 Passkey Retrieval Task Definition

There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.

The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.

- . // Repeat x times. // Passkey is 5 randomly generated numbers. The passkey is 12345. Remeber it. 12345 is the passkey. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.

- . // Repeat y times. What is the passkey?

Listing 1: Prompt format for passkey retrieval (Mohtashami and Jaggi, 2023). The passkey is randomly generated from 10,000 to 99,999.

The passkey retrieval task, as proposed by Mohtashami and Jaggi (2023), involves the model recovering a randomly generated passkey hidden in a long document (see Listing 1 for the task prompt format). Given a language model, we can determine the effective context window by assessing the upper and lower bounds. We assume a random passkey is k tokens away from the end of the input. If a model consistently fails to recover the passkey in multiple attempts, it suggests a context window size smaller than k. Conversely, successful retrievals indicate an effective context window size of at least k tokens (Chen et al., 2023).

In our experiments, we generate test samples based on the prompt template in Listing 1, with lengths ranging from 512 to 32k. There are 100

test cases for each length. Given a language model, we input the passkey task prompt, examine the model’s output for the new 64 tokens, and calculate the accuracy.

- B.2 Analysis I : Consistency of Optimization in Position Embedding

The passkey retrieval results are presented in Section 4.2. Our model demonstrates superior passkey retrieval accuracy compared to baseline models under various conditions. However, we remain intrigued about its optimization, specifically whether it occurs within or beyond the confines of the training context window. To probe this further, we categorize the experimental data into two segments: passkey distance shorter and farther than the training context window length.

Figure 4 (a) illustrates the comparison results when the passkey is inserted less than 512 tokens away from the end token, while Figure 4 (b) illustrates that outside this range. When the passkey is inserted outside the 512 window, RoFormer+NTK & CoCA consistently outperforms Roformer+NTK across various lengths of inference sequences. This superiority persists when the passkey is inserted inside the 512 window. Notably, with an increase in the length of the inference sequence, RoFormer + NTK & CoCA demonstrates increasingly superior performance compared to RoFormer + NTK. These experiments suggest that our model can consistently optimize the position embedding and extend the effective context window.

- B.3 Analysis II : Impact of Dynamic-NTK in CoCA

We utilize the dynamic NTK method (Emozilla, 2023) during the inference process, applying it separately to both our model and the baseline model. To comprehensively assess the robustness of these models, we conduct a thorough validation by varying scaling factors (2, 4, and 8).

The results in Figures 1 and 5 demonstrate that, with the integration of the dynamic NTK method, our model achieves higher passkey accuracy and lower perplexity. Additionally, when the scaling factor varies between 2, 4, and 8, the vanilla RoFormer model fails to maintain stable performance. In contrast, CoCA consistently outperforms RoFormer at different scaling rates. This consistent trend indicates that our model is more robust, showing minimal performance fluctuations with changes in the scaling factor.

[Figure 4]

[Figure 5]

(a) Inserting passkey inside 512 tokens away from end tokens (b) Inserting passkey outside 512 tokens away from end tokens Figure 4: Comparison of effective context window between RoFormer + NTK and RoFormer + NTK & CoCA.

[Figure 6]

Figure 5: Passkey accuracy distribution on 4 range of distances. CoCA outperforms RoFormer for all distances and scaling factors of NTK.

Furthermore, it suggests that by implementing collinear constraints, we can cleverly address anomalous behavior in RoPE, allowing RoPE to better leverage other extrapolation techniques.

B.4 Analysis III : Compatibility of CoCA with PI

B.4.1 Experiment Setup

We conduct experiments utilizing the pre-trained LLaMA-7B model (Touvron et al., 2023a) and LLaMA-7B + CoCA from Section 3.2. To apply PI , we follow the settings of Chen et al. (2023): We set the fine-tuning sequence length to 32,768. The learning rate is adjusted to 2e − 5 with no decay to match. All other settings are maintained as the LLaMA-7B configuration. All experiments are conducted with 32 A100 GPUs, setting a per-device batch size to 1 without gradient accumulation. The experiments take 6,000 steps to accomplish.

#### B.4.2 Long Context Validation

The results of fine-tuning with PI are presented in Table 4. In terms of long sequence modeling, both LLaMA-7B+PI and LLaMA-7B+CoCA & PI demonstrate competitive performance across sequence lengths ranging from 512 to 8192. However, at longer sequence lengths (16384 and 32768), LLaMA-7B+CoCA & PI exhibits a slight performance advantage over LLaMA-7B+PI. For long context retrieval, both methods achieve exceptionally high accuracy, with scores approaching the ideal value of 1.0 across all sequence lengths.

Overall, these findings suggest that the integration of PI and the CoCA module with the LLaMA7B model yields robust performance in both long sequence modeling and long context retrieval tasks. Additionally, the CoCA module demonstrates the ability to maintain performance levels comparable to PI, particularly evident at longer sequence lengths.

#### B.4.3 Short Context Validation

In addition to enhancing long-context extrapolation, it is imperative to consider the practicality and scalability of CoCA in short contexts. Hence, we evaluate our model on OpenCompass (Contributors, 2023), which comprises various dimensions, including reasoning, understanding, language, and examination. The results are presented in Table 5.

The table demonstrates that LLaMA-7B models integrated with CoCA achieve performance comparable to the baseline LLaMA-7B across all evaluated dimensions. Specifically, the integration of

###### Method 512 1024 2048 4096 8192 16384 32768

Performance on Long Sequence Modeling (Perplexity)

LLaMA-7B+PI 9.06 7.55 7.74 7.16 7.04 6.93 7.11 + CoCA & PI 9.65 8.19 8.37 7.87 7.84 7.83 7.96

Performance on Long Context Retrieval (Passkey Accuracy)

LLaMA-7B+PI 1.0 1.0 1.0 1.0 1.0 1.0 0.99 + CoCA & PI 1.0 1.0 1.0 1.0 1.0 0.99 0.99

- Table 4: Comparison results for LLaMA-7B+PI and LLaMA-7B+CoCA & PI after fine-tuning with sequence length of 32,768. CoCA succeeds in maintaining the performance of PI within fine-tuning window size.

Method Reasoning Understanding Language Examination Average

LLaMA-7B 48.25 47.57 46.41 29.63 42.97 + CoCA 45.55 51.14 55.27 25.14 44.28 + PI 44.98 51.54 54.79 27.03 44.59 + CoCA & PI 46.88 51.82 55.56 25.31 44.89

- Table 5: OpenCompass results of LLaMA-7B and its variants. Models integrated with CoCA achieved comparable performance to LLaMA-7B, leading no harm to the expression ability of the model.

CoCA yields no significant degradation in the expression ability of the model. This suggests that CoCA is effective not only in long-context scenarios but also in short-context tasks, demonstrating its versatility and suitability for practical applications.

- C Computational and Spatial Complexity Analysis

|Module|vanilla self-attention| |CoCA| |
|---|---|---|---|---|
| |Computational<br><br>|Spatial|Computational Spatial| |
|WQK(T)V T half T Relu QK(T) rotation Krot = Q ◦ Trot QrotKTrot Mask Softmax|3Nd2h 2Ndh N2dh N2 N2<br><br>|Nd Nd N2 N2 N2|3Nd2h Ndh Ndh 2Ndh N2dh N2dh N2 N2<br><br>|Nd Nd Nd Nd N2d N2 N2 N2<br><br>|

- Table 6: The comparison of computational and spatial complexity between vanilla self-attention block and CoCA. Here, N represents the sequence length, h denotes the number of heads, and d signifies the dimension of each head.

[Figure 7]

Figure 6: Inference speed comparison between CoCA and vanilla self-attention.

a significant increase in spatial complexity when expanding Krot = Q ◦ Trot, becoming d times that of the vanilla self-attention. This level of spatial complexity is not practical for applications.

To address this problem, we can draw inspiration

from the computation of QrotKTrot, which involves two steps: element-wise multiplication between

Qrot and Krot followed by summation along the hidden dimension. Optimization is attainable by condensing the hidden dimension before fully expanding the sequence length dimension. Consequently, the spatial complexity is effectively reduced from N2d to N2. This optimization strategy is equally applicable to Krot = Q ◦ Trot. These two components can be unified as articulated in Equation (18):

In this section, we analyze the computational and spatial complexities of CoCA. Table 6 provides a detailed comparison between the vanilla self-attention mechanism and CoCA.

When using the operation Krot = Q ◦ Trot, the computational complexity of CoCA does not exceed twice that of the vanilla self-attention. In practice, the training and inference speed of CoCA are comparable to the vanilla self-attention mechanism, with only a slight increase of about 5% to 10% , as depicted in Figure 6. However, there is

QrotKTrot = Qrot(Q ◦ Trot)T (18)

The commendable work accomplished by opt_einsum (a. Smith and Gray, 2018) facilitates the optimization of Equation (18). Experimental results indicate that Roformer+CoCA only demands approximately 60GB of GPU memory during inference with a sequence length of 32k, aligning closely with the memory consumption of the vanilla self-attention mechanism.

### D Theoretical Proof

D.1 Strong Form of Long-term Decay with CoCA

We have introduced the basic theory of Rotary Position Embedding in Section 2.1. In fact, (Su et al., 2024) shows that RoPE has the characteristic of long-term decay:

 

 

d/2−1

hjeisθj

|a(s)| = Re

j=0

(19)

d/2−1

|Sj+1|

≤ (max

|hi+1 − hi|)

i

j=0

where hj := (q2j + iq2j+1)(k2j − ik2j+1) and Sj := jk−=01 eisθk, s = (m − n), m for the index of query, n for the index of key. Since the value of

d/2−1 j=0 |Sj+1| decays with the relative distance s,

the attention score decays either.

This characteristic ensures the stability of RoPE during extrapolation to some extent by preventing outliers. For CoCA, a stronger deduction can be formulated as follows:

d/2−1

|Cj+1| (20)

|a(s)| ≤ (max

|li+1 − li|)

i

j=0

where lj := |q2j+iq2j+1||k2j+ik2j+1|, and Cj := j−1 k=0 cos(sθk). Furthermore, it holds that:

|li+1 − li| ≤ |hi+1 − hi| (21)

Proof: Notice that when the initial angle Θj between qj and kj is 0, from Equation (17), the attention score can be simplified as:

 

 

d/2−1

hjeisθj

a(s) = Re

j=0

(22)

d/2−1

=

lj cos(sθj)

j=0

By following the study of (Su et al., 2024), we can easily derive the estimation in Equation (20).

For Equation (21), applying the triangle inequality, we get:

|hi+1 − hi| ≥ ||hi+1| − |hi|| (23)

Reviewing the definition of hi = (q2j + iq2j+1)(k2j − ik2j+1), we will find:

|hi+1 − hi| ≥ ||hi+1| − |hi||

= ||qi+1k∗i+1| − |qik∗i || = ||qi+1ki+1| − |qiki|| = |li+1 − li|

(24)

[Figure 8]

Figure 7: Rotary Borders Analysis. Regarding qj as x-axis, 3 distinct boundaries correspond to kj, −qj, and qj

#### D.2 Rotary Borders Analysis

In Section 2.2, we analyzed the anomalous phenomena of RoPE. To illustrate the rotation anomalies, let’s focus on a specific instance (case (d) of Section 2.2). As shown in Figure 7, three distinct boundaries emerge during the rotation. By adopting a relative coordinate system with qj serving as the x-axis, these boundaries correspond to kj, −qj, and qj.

Everytime the relative angle of qj and kj crosses these boundaries, the monotonicity of their innerproduct < qj,kj > undergoes a reversal. Thus, for the vanilla self-attention, it learnt a piecewise monotonic function of < qj,kj >:



↑ (m − n), ∀ − (2π − Θj) ≤ θ(qj, kj) < 0 ↓ (m − n), ∀0 ≤ θ(qj, kj) < π ↑ (m − n), ∀π ≤ θ(qj, kj) < 2π



< qj, kj >=

...

↑ (m − n), ∀(2k − 1)π ≤ θ(qj, kj) < (2k)π ↓ (m − n), ∀(2k)π ≤ θ(qj, kj) < (2k + 1)π



(25)

where θ(qj,kj) = Θj + (m − n)θj defined in Section 2.2.

This introduces confusion into the model during direct context extrapolation. Therefore, methods like PI and NTK tried to introduce interpolation or extrapolation techniques to eliminate outof-distribution (OOD) positions.

Except the first equation in Equation (25), the two boundaries caused by −qj, and qj are regular with periodicity of 2π, it is easy to handle when applying methods like PI or NTK. However, the boundaries caused by kj are hard to handle. There are d/2∗h∗L (d for head dimension, h for number of heads, L for number of layers) different boundaries during context extrapolation, which break the periodicity of 2π.

Furthermore, after applying interpolation or extrapolation techniques, more positions will fall into

this abnormal area. It increased k times (k for interpolation factor) for PI and λ2j/d times (λ for scaling factor) for NTK.

From this perspective, positional concentration of PI resulted in more trouble than NTK, i.e. additionally more positions in abnormal area during context extrapolation. This may explain in some extent why NTK could be used without fine-tuning for vanilla self-attention, but PI requires fine-tuning.

By enforcing Θj to 0, our proposed CoCA, constraining kj to be collinear with qj, effectively resolves the border-related challenge associated with kj.

From experiments in Secton 4, with the integrating of CoCA, now NTK can be leveraged well through direct use, while PI achieved improvement for direct use but still limited, which requires further studies.

D.3 Homeomorphism of Representation Space

Theorem 2. (Homeomorphism of representation space) For any attention score defined as follows:

a(m, n) = Re(⟨f(qm, m), f(qm, n) ◦ tn⟩) (26)

where qm is the query, m is the index number of query, tn is the collinear coefficient of CoCA, n is the index number of key, f is the rotation operator.

Denote its representation space with respect to qm as:

F(Q) = {a(m, n)|∀qm ∈ Q ⊂ Rd} (27)

where qm = WQxm, xm ∈ EN, m ∈ [1,N] and EN is the word embedding space, WQ is the projection matrix.

Then we have the following homeomorphism:

F(Q) ∼= F(Qhalf) (28)

where Qhalf = Q|q2j=q2j+1,∀j∈[0,d/2−1]. Proof: We prove it by demonstrating the homeomorphism mapping G:

G : F(Q) → F(Qhalf)

qd2−2 + qd2−1 2

q02 + q12 2

F((q0, ..., qd−1)  → F((

, ...,

) (29)

It consists of three parts: Part I (G is a bijection): recall Equation (17), we have:

G(X) = X, ∀X ∈ F(Q) (30)

which implies that G is an identity mapping, naturally injective.

Next, we prove that G is also surjective: for any Y = F((q0,...,qd−1)|q2j=q2j+1) ∈ F(Qhalf), there exists Y ∈ F(Q) such that G( Y ) = Y . Let

Y = F((q0, ..., qd−1)|q2j=q2j+1) ∈ F(Q) (31)

obviously we have G( Y ) = Y .

- Part II (G is continuous): For any X0 ∈ F(Q), ϵ > 0, there exists δ, such that if |X − X0| < δ, then |G(X) − G(X0)| < ϵ.

From Part I, G is an identity mapping, let δ = ϵ, then the continuity of G holds.

- Part III (G−1 is continuous): G is an identity mapping, so is G−1. Following Part II, we immediately deduce that G−1 is continuous.

| |
|---|

#### D.4 Slack Position Embedding

Let H be a Hilbert space, and {T (n)|n ≥ 0} ⊂ L(H) is a family of bounded linear operator on H. A is the inner-product defined on H.

If it satisfies the following property, then we call {T (n)|n ≥ 0} is a relative (bounded linear) operator on H:

∃ {S(m)|m ∈ Z} : H × H → C (X, Y )  → S(m)(X, Y )

is a family of semi-bilinear operator on H

s.t. S(p − q)(X, Y ) = A(T (p)(X), T (q)(Y )) ∀ p, q ∈ [0, N], X, Y ∈ H,

(32)

Additionally, if it satisfies the following property, then we call {T (n)|n ≥ 0} is a slack relative (bounded linear) operator on H:

∃ {S(m)|m ∈ Z} : H × H → C (X, Y )  → S(m)(X, Y )

is a family of semi-bilinear operator on H

and H′ ⊂ H, H′ ̸= ∅ s.t. S(p − q)(X, Y ) = A(T (p)(X), T (q)(Y )) ∀ p, q ∈ [0, N], X, Y ∈ H′,

(33)

Specifically, when H represents our projection space in self-attention, and {T (n)|n ≥ 0} is a position embedding on it, such as the Rotary Position

##### Embedding (RoPE), we refer to it as a Slack Position Embedding (SPE) if it satisfies the property described in Equation (33).

