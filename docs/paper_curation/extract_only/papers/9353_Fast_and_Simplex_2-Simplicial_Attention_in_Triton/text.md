arXiv:2507.02754v1[cs.LG]3Jul2025

FAST AND SIMPLEX: 2-SIMPLICIAL ATTENTION IN TRITON

Aurko Roy Meta Menlo Park, CA roy.aurko@gmail.com

Sijia Chen Meta Menlo Park, CA sijiac@meta.com

Timothy Chou Meta Menlo Park, CA timchou@meta.com

Sai Surya Duvvuri∗ Department of Computer Science University of Texas at Austin saisurya@cs.utexas.edu

Jiecao Yu Meta Menlo Park, CA jiecaoyu@meta.com

Xiaodong Wang Meta Menlo Park, CA xdwang@meta.com

Manzil Zaheer Meta Menlo Park, CA manzilzaheer@meta.com

Rohan Anil† San Francisco, CA rohan.anil@gmail.com

ABSTRACT

Recent work has shown that training loss scales as a power law with both model size and the number of tokens, and that achieving compute-optimal models requires scaling model size and token count together. However, these scaling laws assume an infinite supply of data and apply primarily in compute-bound settings. As modern large language models increasingly rely on massive internet-scale datasets, the assumption that they are compute-bound is becoming less valid. This shift highlights the need for architectures that prioritize token efficiency.

In this work, we investigate the use of the 2-simplicial Transformer, an architecture that generalizes standard dot-product attention to trilinear functions through an efficient Triton kernel implementation. We demonstrate that the 2-simplicial Transformer achieves better token efficiency than standard Transformers: for a fixed token budget, similarly sized models outperform their dot-product counterparts on tasks involving mathematics, coding, reasoning, and logic. We quantify these gains by demonstrating that 2-simplicial attention changes the exponent in the scaling laws for knowledge and reasoning tasks compared to dot product attention.

- 1 INTRODUCTION

Large language models (LLMs) based on the Transformer architecture (Vaswani et al., 2017) have become foundational to many state-of-the-art artificial intelligence systems, including GPT-3 (Brown et al., 2020), GPT-4 (Achiam et al., 2023), Gemini (Team et al., 2023), and Llama (Touvron et al., 2023). The remarkable progress in scaling these models has been guided by neural scaling laws (Hestness et al., 2017; Kaplan et al., 2020; Hoffmann et al., 2022), which empirically establish a power-law relationship between training loss, the number of model parameters, and the volume of training data.

A key insight from this body of work is that optimal model performance is achieved not simply by increasing model size, but by scaling both the number of parameters and the amount of training data in tandem. Notably, Hoffmann et al. (2022) demonstrate that compute-optimal models require a balanced scaling approach. Their findings show that the Chinchilla model, with 70 billion parameters, outperforms the much larger Gopher model (280 billion parameters) by being trained on four times

∗Work done during an internship at Meta †Work done while at Meta

as much data. This result underscores the importance of data scaling alongside model scaling for achieving superior performance in large language models.

As artificial intelligence (AI) continues to advance, a significant emerging challenge is the availability of sufficiently high-quality tokens. As we approach this critical juncture, it becomes imperative to explore novel methods and architectures that can scale more efficiently than traditional Transformers under a limited token budget. However, most architectural and optimizer improvements merely shift the error but do not meaningfully change the exponent of the power law (Everett, 2025). The work of Kaplan et al. (2020); Shen et al. (2024) showed that most architectural modifications do not change the exponent, while Hestness et al. (2017) show a similar result for optimizers. The only positive result has been on data due to the works of Sorscher et al. (2022); Bahri et al. (2024); Brandfonbrener et al. (2024) who show that changing the data distribution can affect the exponent in the scaling laws.

In this context we revisit an old work Clift et al. (2019) which generalizes the dot product attention of Transformers to trilinear forms as the 2-simplicial Transformer. We explore generalizations of RoPE (Su et al., 2024) to trilinear functions and present a rotation invariant trilinear form that we prove is as expressive as 2-simplicial attention. We further show that the 2-simplicial Transformer scales better than the Transformer under a limited token budget: for a fixed number of tokens, a similar sized 2-simplicial Transformer out-performs the Transformer on math, coding and reasoning tasks. Furthermore, our experiments also reveal that the 2-simplicial Transformer has a more favorable scaling exponent corresponding to the number of parameters than the Transformer (Vaswani et al., 2017). This suggests that, unlike Chinchilla scaling (Hoffmann et al., 2022), it is possible to increase tokens at a slower rate than the parameters for the 2-simplicial Transformer. Our findings imply that, when operating under token constraints, the 2-simplicial Transformer can more effectively approach the irreducible entropy of natural language compared to dot product attention Transformers.

- 2 RELATED WORK

Several generalizations of attention have been proposed since the seminal work of Vaswani et al. (2017). A line of work that started immediately after was to reduce the quadratic complexity of attention with sequence length. In particular, the work of Parmar et al. (2018) proposed local attention in the context of image generation and several other works subsequently used it in conjunction with other methods for language modeling (Zaheer et al., 2020; Roy et al., 2021). Other work has proposed doing away with softmax attention altogether - e.g., Katharopoulos et al. (2020) show that replacing the softmax with an exponential without normalization leads to linear time Transformers using the associativity of matrix multiplication. Other linear time attention work are state space models such as Mamba (Gu & Dao, 2023); however these linear time attention methods have received less widespread adoption due to their worse quality compared to Transformers in practice. According to Allen (2025), the key factor contributing to Mamba’s success in practical applications is the utilization of the conv1d operator; see also So et al. (2021) and Roy et al. (2022) for similar proposals to the Transformer architecture.

The other end of the spectrum is going from quadratic to higher order attention. The first work in this direction to the best of our knowledge was 2-simplicial attention proposed by Clift et al. (2019) which showed that it is a good proxy for logical problems in the context of deep reinforcement learning. A similar generalization of Transformers was proposed in Bergen et al. (2021) which proposed the Edge Transformer where the authors proposed triangular attention. The AlphaFold (Jumper et al., 2021) paper also used an attention mechanism similar to the Edge Transformer which the authors called triangle self-attention induced by the 2D geometry of proteins. Higher order interactions were also explored in Wang et al. (2021) in the context of recommender systems. Recent work by Sanford et al. (2023) shows that the class of problems solved by an n-layer 2-simplicial Transformer is strictly larger than the class of problems solved by dot product attention Transformers. In particular, the authors define a class of problems referred to as Match3 and show that dot product attention requires exponentially many layers in the sequence length to solve this task. Follow up work by Kozachinskiy et al. (2025) propose a scalable approximation to 2-simplicial attention and prove lowerbounds between Strassen attention and dot product attention on tasks that require more complex reasoning using VC dimension (Vapnik, 1968) arguments.

Also related is work on looping Transformer layers (Dehghani et al., 2018) as in Universal Transformers; see also Yang et al. (2023); Saunshi et al. (2025) for a more recent treatment of the same idea.

Both higher order attention and looping serve a similar purpose: compute a more expressive function per parameter. It has been established in these works that looped Transformers are better at logical reasoning tasks. A key challenge in scaling looped Transformers to larger models is their trainability. Specifically, looping k times increases the model depth by a factor of k, which can significantly exacerbate the difficulties associated with training deeper models. As a result, it remains unclear how well large looped Transformers can be trained, and further research is needed to address this concern.

Notation. We use small and bold letters to denote vectors, capital letters to denote matrices and tensors and small letters to denote scalars. We denote ⟨a,b⟩ to denote dot product between two vectors

a and b. Similarly, the trilinear dot product is denoted as follows:⟨a,b,c⟩ = di=1⟨ai,bi,ci⟩. We use @ to highlight a matrix multiplication, for e.g., (AB)@C, for matrices A, B, C. To denote array

slicing, we use a[l : l + m] = (al,...,al+m−1) with zero-based indexing. Some tensor operations are described using Einstein summation notation as used in the Numpy library (Harris et al., 2020). We use FLOPs to denote floating point operations. Column stacking of arrays are denoted by [a,b,c]. We use det to denote determinant of a square matrix.

- 3 OVERVIEW OF NEURAL SCALING LAWS

In this section we provide a brief overview of neural scaling laws as introduced in Kaplan et al. (2020). We will adopt the approach outlined by Hoffmann et al. (2022), which proposes that the loss L(N,D) decays as a power law in the total number of model parameters N and the number of tokens D:

L(N,D) = E +

A Nα

+

B Dβ

. (1)

The first term E is often described as the irreducible loss which corresponds to the entropy of natural text. The second term captures the fact that a model with N parameters underperforms this ideal generative process. The third term corresponds to the fact that we train on only a finite sample of the data and do not train the model to convergence. Theoretically, as N → ∞ and D → ∞ a large language model should approach the irreducible loss E of the underlying text distribution.

For a given compute budget C where FLOPs(N,D) = C, one can express the optimal number of parameters as Nopt ∝ Ca and the optimal dataset size as Dopt ∝ Cb. The authors of Hoffmann et al. (2022) perform several experiments and fit parametric functions to the loss to estimate the exponents a and b: multiple different approaches confirm that roughly a ∼ 0.49 while b ∼ 0.5. This leads to the central thesis of Hoffmann et al. (2022): one must scale the number of tokens proportionally to the model size.

However, as discussed in Section 1, the quantity of sufficiently high-quality tokens is an emerging bottleneck in pre-training scaling, necessitating an exploration of alternative training algorithms and architectures. On the other hand recent studies have shown that most modeling and optimization techniques proposed in the literature merely shift the error (offset E) and do not fundamentally change the exponent in the power law. We refer the readers to this excellent discussion in Everett (2025).

- 4 THE 2-SIMPLICIAL TRANSFORMER

The 2-simplicial Transformer was introduced in Clift et al. (2019) where the authors extended the dot product attention from bilinear to trilinear forms, or equivalently from the 1-simplex to the 2-simplex. Let us recall the attention mechanism in a standard Transformer (Vaswani et al., 2017). Given a sequence X ∈ Rn×d we have three projection matrices WQ,WK,WV ∈ Rd×d which we refer to as the query, key and value projections respectively. These projection matrices are used to infer the query Q = XWQ, key K = XWK and value V = XWV respectively. This is then used to construct the attention logits:

√

A = QK⊤/

d ∈ Rn×n, (2) where each entry is a dot product Aij = ⟨qi,kj⟩/

## √

d which are both entries in Rd . The attention scores (logits) are then transformed into probability weights by using a row-wise softmax operation:

n

exp(Aij). (3)

Sij = exp(Aij)/

j=1

k

## i j

i j

(a) 1-simplex between two nodes i, j

(b) 2-simplex between three nodes i, j, k

Figure 1: Geometry of dot product attention and 2-simplical attention.

The final output of the attention layer is then a linear combination of the values according to these attention scores:

n

Aijvj (4)

v˜i =

j=1

The 2-simplicial Transformer paper Clift et al. (2019) generalizes this to trilinear products where we have two additional key and value projection matrices WK′ and WV ′, which give us K′ = XWK′ and V ′ = XWV ′. The attention logits for 2-simplicial Transformer are then given by the trilinear product between Q, K and K′, resulting in the following third-order tensor:

A(ijk2s) = ⟨qi,kj,k′k⟩

√

d

1 √

=

so that the attention tensor becomes:

Sijk(2s) = exp(A(ijk2s))/

j,k

d

QilKjlKkl′ , (5)

d

l=1

exp(A(ijk2s)), (6)

with the final output of the attention operation being defined as

v˜(2s)(i) =

n

Sijk(2s) (vj ◦ vk′ ), (7)

j,k=1

where vj ◦ vk′ represents the element wise Hadamard product between two vectors in Rd. The pseudo-code for 2-simplicial attention is depicted in Algorithm 1. Note that Equation 5 does not

incorporate any position encoding such as RoPE (Su et al., 2024); we discuss this in the next section.

- Algorithm 1 Pseudocode for the forward pass of 2-simplicial attention

- 1: procedure 2-SIMPLICIAL ATTENTION(Q, K, V , K′, V ′)
- 2: logits ← einsum(“btnh,bsnh,brnh → bntsr”,Q,K,K′)
- 3: attention ← softmax(logits+causal-mask,axis = [−1,−2])
- 4: output ← einsum(“bntsr,bsnh,brnh → btnh”,attention,V,V′)
- 5: return output
- 6: end procedure

- 5 DETERMINANT BASED TRILINEAR FORMS

RoPE (Su et al., 2024) was proposed as a way to capture the positional information in a sequence for Transformer language models. RoPE applies a position dependent rotation to the queries qi and the key kj so that the dot product ⟨qi,kj⟩ is a function of the relative distance i − j. In particular, note that the dot product is invariant to orthogonal transformations R ∈ Rd×d:

## ⟨qi,kj⟩ = ⟨Rqi,Rkj⟩.

This is important for RoPE to work as for a query qi and key ki at the same position i, we expect its dot product to be unchanged by the application of position based rotations: ⟨qi,ki⟩ = ⟨Rqi,Rki⟩.

Note that the trilinear form defined in Equation 5 is not invariant to rotation and the application of the same rotation to qi, ki and k′i no longer preserves the inner product: ⟨qi,ki,k′i⟩ = dl=1 qilkilk′il ̸= ⟨Rqi,Rki,Rk′i⟩. Therefore, to generalize RoPE to 2-simplicial attention, it is important to explore alternative bilinear and trilinear forms that are rotation invariant.

We note that the following functions are also invariant to rotations:

- fˆ2(a,b) = det

- a1 a2
- b1 b2

= a1b2 − a2b1,

- fˆ3(a,b,c) = det

- a1 a2 a3
- b1 b2 b3
- c1 c2 c3

,

= a1b2c3 + a2b3c1 + a3b1c2 − a1b3c2 − a2b1c3 − a3b2c1

= ⟨(a1,a2,a3),(b2,b3,b1),(c3,c1,c2)⟩ − ⟨(a1,a2,a3),(b3,b1,b2),(c2,c3,c1)⟩, (8)

the rearrangement in the last equality is popularly called Sarrus rule (Strang, 2022). Here, fˆ2 is a bilinear form in a = (a1,a2) and b = (b1,b2) and fˆ3 is a trilinear form in a = (a1,a2,a3), b = (b1,b2,b3), c = (c1,c2,c3). Geometrically, |fˆ2(a,b)| measures the area of the parallelogram spanned by a and b, and similarly, |fˆ2(a,b,c)| measures the volume of the parallelotope spanned by a, b and c. We use the signed determinant operation fˆ3 to compute A(det) ∈ Rn×n×n. For any vector q, let q(l) = q = q[3(l − 1) : 3l] be its lth chunk of size 3. The logits are defined as:

p

A(ijdet)

## (l)]). (9)

det([qi(l),k(jl)

## ,k′j

1j2 =

2

1

l=1

Since Equation 8 has 2 dot product terms due to Sarrus rule, it would modify Algorithm 1 to use 2 einsums instead of 1 in line 2. The final attention weights S are computed by applying a softmax function on the logits above, similar to Equation 6. The output for token i is then the weighted sum of value vectors as in Equation 7.

Theorem 5.1. For any input size n and input range m = nO(1), there exists a transformer architecture with a single head of attention with logits computed as in (9), with attention head dimension d = 7, such that for all X ∈ [M]N, the transformer’s output for element xi is 1 if ∃j1,j2 s.t. xi + xj

= 0 (mod M), and 0 otherwise.

## + xj

1

2

We provide the proof in Appendix A. Since the sum-of-determinants trilinear function of Equation 9 involves 6 terms compared to the simpler trilinear form of Equation 5, in the following sections where we compute the backwards function for 2-simplicial attention, we will use the simpler trilinear form of Equation 5 without loss of generality.

- 6 MODEL DESIGN

Since 2-simplicial attention scales as O(n3) in the sequence length n, it is impractical to apply it over the entire sequence. Instead, we parametrize it as O(n × w1 × w2), where w1 and w2 define the dimensions of a sliding window over the sequence. Each query vector Qi attends to a localized region of w1 K keys and w2 K′ keys, thereby reducing the computational burden. We systematically evaluate various configurations of w1 and w2 to identify optimal trade-offs between computational efficiency and model performance (see Table 1).

For causal dot product attention, the complexity for a sequence of length n is given by:

- 1

- 2 · 2 · 2n2 = 2n2,

O(A) =

where n is the sequence length. This involves two matrix multiplications: one for Q@K, one for P@V , each requiring two floating-point operations per element. The causal mask allows us to skip 12 of these computations.

In contrast, the complexity of 2-simplical attention, parameterized by w1 and w2, is expressed as:

## O(A(2s)) = 3 · 2nw1w2 = 6nw1w2

This increase in complexity arises from the trilinear einsum operation, which necessitates an additional multiplication compared to standard dot product attention.

w1 × w2 w1 w2 Latency (ms)

32k 1024 32 104.1 ms 32k 512 64 110.7 ms 16k 128 128 59.2 ms 16k 256 64 55.8 ms 16k 512 32 55.1 ms 16k 1024 16 55.1 ms 8k 256 32 28.3 ms

Table 1: Latency for different combinations of w1, w2

We choose a window size of (512, 32), balancing latency and quality. With this configuration, the computational complexity of 2-simplical attention is comparable to dot product attention at 48k context length.

A naive sliding window 2-simplicial attention implementation has each Qi vector attending to w1 + w2 − 1 different KK′ vectors, as illustrated in Figure 2. Thus, tiling queries Q like in flash attention leads to poor compute throughput. Inspired by Native Sparse Attention (Yuan et al., 2025), we adopt a model architecture leveraging a high Grouped Query Attention GQA (Ainslie et al., 2023) ratio of 64 . This approach enabled efficient tiling along query heads, ensuring dense computation and eliminating the need for costly element-wise masking.

- 7 KERNEL OPTIMIZATION

We introduce a series of kernel optimizatins tailored for 2-simplical attention, building off of Flash Attention (Dao et al., 2022) using online softmax. For the trilinear operations, we perform 2d tiling by merging one of the inputs via elementwise multiplication and executing matmul on the product as illustrated in Figure 2. This allows us to overlap both QK and V V ′ on CUDA Core with (QK)@K′ and P@(V V ′) on Tensor Core. Implementing this in Triton, we achieve 520 TFLOPS, rivaling the fastest FAv3 Triton implementations. Further optimization could be achieved with a lower-level language like CUTLASS for finer grained tuning and optimizations. Despite this, we achieve competitive performance compared to CUTLASS FAv3 for large sequence lengths, as shown in Figure 3.

For the backwards pass, we have

dVjd =

(Aijk · dOid · Vkd′ ) (10)

i,k

dVkd′ =

(Aijk · dOid · Vjd) (11)

i,j

(dOid · Vjd · Vkd′ ) (12)

dPijk =

d

dS = dsoftmaxjk(dP) (13) dKjd =

(Qid · dSijk · Kkd′ ) (14)

i,k

dKkd′ =

(Qid · dSijk · Kjd) (15)

i,k

[Figure 1]

Figure 2: Left: Visualization of sliding window 2-simplical attention. Each Qi attends to a [w1,w2] shaped rectangle of K, K′. Right: Tiling to reduce 2-simplicial einsum QKK′ to elementwise mul QK′ on CUDA core and tiled matmul (QK′)@K on tensor core.

[Figure 2]

[Figure 3]

Figure 3: FLOPs and Latencies of FAv3 vs 2-simplical attention

dQid =

(dSijk · Kjd · Kkd′ ) (16)

j,k

For the backwards pass, aggregations across three different dimension orderings introduces significant overhead from atomic operations. To mitigate this, we decompose the backward pass into two distinct kernels: one for computing dK and dV , and another for dK′, dV ′, and dQ. Although this approach incurs additional overhead from recomputing O and dS, we find it is better than the extra overhead from atomics needed for a single fused kernel. We note this may be a limitation of Triton’s coarser grained pipeline control making it difficult to hide the overhead from atomics.

For small w2, we employ a two-stage approach to compute dQ jointly with dK′, dV ′ without atomics as detailed in Algorithm 2. We divide Q along the sequence dimension into

## [w2,dim]

sized tiles. First we iterate over even tiles, storing dQ, dK, dK′, and dV , dV ′. Then we iterate over odd tiles, storing dQ, and adding to dK, dK′ and dV , dV ′.

- Algorithm 2 Backward pass for 2-simplicial attention

- 1: procedure 2-SIMPLICIAL FLASH ATTENTION BWD(Q, K, V , K′, V ′, w1, w2)
- 2: for stage in [0, 1] do
- 3: for q_start in range(stage * w2, seq_len, w2 * 2) do
- 4: q_end ← q_start + w2
- 5: for kv1_start in range(q_start - w1, q_end) do
- 6: q_tile ← Q[q_start : q_end]
- 7: ...
- 8: k2_tile ← K′[kv1_start : q_end]
- 9: dQ += dQ(q_tile,k2_tile,...)
- 10: dV ′ + = dV ′(q_tile,k2_tile,...)
- 11: dK′ + = dK′(q_tile,k2_tile,...)
- 12: end for
- 13: if stage == 1 then
- 14: dK′ += load dK′
- 15: dV ′ += load dV ′
- 16: end if
- 17: store dQ, ..., dK′
- 18: end for
- 19: end for
- 20: end procedure

- 8 EXPERIMENTS & RESULTS

We train a series of MoE models (Jordan & Jacobs, 1994; Shazeer et al., 2017) ranging from 1 billion active parameters and 57 billion total parameters to 3.5 billion active parameters and 176 billion total parameters. We use interleaved sliding-window 2-simplicial attention, where every fourth layer is a 2-simplicial attention layer. The choice of this particular ordering is to distribute the load in attention computation when using pipeline parallelism (Huang et al., 2019; Narayanan et al., 2019), since 2-simplicial attention and global attention are the most compute intensive operations in a single pipeline stage and have comparable FLOPs.

We use the AdamW optimizer (Loshchilov et al., 2017) with a peak learning rate of 4 × 10−3 and weight decay of 0.0125. We use a warmup of 4000 steps and use a cosine decay learning schedule decreasing the learning rate to 0.01× of the peak learning rate. We report the negative log-likelihood on GSM8k (Cobbe et al., 2021), MMLU (Hendrycks et al., 2020), MMLU-pro (Wang et al., 2024) and MBPP (Austin et al., 2021), since these benchmarks most strongly test math, reasoning and coding skills in pre-training.

#### Model Active Params Total Params GSM8k MMLU MMLU-pro MBPP

- Transformer 1B 57B 0.3277 0.6411 0.8718 0.2690

- 2-simplicial 1B 57B 0.3302 0.6423 0.8718 0.2714 ∆(%) +0.79% +0.19% -0.01% +0.88%

Transformer 2B 100B 0.2987 0.5932 0.8193 0.2435

- 2-simplicial 2B 100B 0.2942 0.5862 0.8135 0.2411

- ∆(%) -1.51% -1.19% -0.71% -1%

Transformer 3.5B 176B 0.2781 0.5543 0.7858 0.2203 2-simplicial 3.5B 176B 0.2718 0.5484 0.7689 0.2193

- ∆(%) -2.27% -1.06% -2.15% -0.45%

- Table 2: Negative log-likelihood of Transformer (Vaswani et al., 2017) versus 2-simplicial attention. For MMLU (Hendrycks et al., 2020) and MMLU-pro (Wang et al., 2024) we measure the negative log-likelihood of the choice together with the entire answer. For GSM8k (Cobbe et al., 2021) we use 5-shots for the results.

We see that the decrease (∆) in negative log-likelihood scaling from a 1.0 billion (active) parameter model increases going to a 3.5 billion (active) parameter model. Furthermore, on models smaller than 2.0 billion (active) parameters, we see no gains from using 2-simplicial attention. From Table 2 we can estimate how the power law coefficients for the 2-simplicial attention differ from dot product attention. Recall from Section 3 that the loss can be expressed as:

L(N,D) = E +

A Nα

+

B Dβ

. (17)

Since we train both the models on the same fixed number of tokens, we may ignore the third term and simply write the loss as:

L(N) = E′ +

A Nα

, (18) log L(N) ≈ log E′′ + log A − α log N (19) −log L(N) = α log N + β, (20)

where β = −log E′′ − logA and E′′ is an approximation to E′ since E′ is small. Note that here we used log(a + b) = log(1 + a/b) + log(b) to separate out the two terms, with the 1 + a/b term hidden in E′′. Therefore we can estimate α,β for both sets of models from the losses in Table 2 where we use for N the active parameters in each model. We estimate the slope α and the intercept β for both the Transformer as well as the 2-simplicial Transformer in Table 3. We see that 2-simplicial attention has a steeper slope α, i.e. a higher exponent in its scaling law compared to dot product attention Transformer (Vaswani et al., 2017).

Model GSM8k MMLU MMLU-pro MBPP

α β α β α β α β

Transformer 0.1420 -1.8280 0.1256 -2.1606 0.0901 -1.7289 0.1720 -2.2569 2-simplicial 0.1683 -2.3939 0.1364 -2.3960 0.1083 -2.1181 0.1837 -2.5201 ∆(%) 18.5% 8.5% 20.2% 6.8%

- Table 3: Estimates of the power law coefficients α and β for the Transformer (Vaswani et al., 2017) and 2-simplicial attention.

Model GSM8k MMLU MMLU-pro MBPP

R2 residual R2 residual R2 residual R2 residual

Transformer 0.9998 2.8 × 10−6 0.9995 4.7 × 10−6 0.9972 1.5 × 10−5 0.9962 7.5 × 10−5 2-simplicial 0.9974 4.9 × 10−5 0.9989 1.3 × 10−5 0.9999 4.6 × 10−8 0.9999 1.5 × 10−6

Table 4: R2 and residuals measuring goodness of fit for Table 3.

- 9 DISCUSSION

While 2-simplicial attention improves the exponent in the scaling laws, we should caveat that the technique maybe more useful when we are in the regime when token efficiency becomes more important. Our Triton kernel while efficient for prototyping is still far away from being used in production. More work in co-designing the implementation of 2-simplicial attention tailored to the specific hardware accelerator is needed in the future.

- 10 CONCLUSION

We show that a similar sized 2-simplicial attention (Clift et al., 2019) improves on dot product attention of Vaswani et al. (2017) by improving the negative log likelihood on reasoning, math and coding problems (see Table 2). We quantify this explicitly in Table 3 by demonstrating that 2-simplicial attention changes the exponent corresponding to parameters in the scaling law of Equation 18: in particular it has a higher α for reasoning and coding tasks compared to the Transformer (Vaswani et al., 2017) which leads to more favorable scaling under token constraints. Furthermore, the percentage increase in the scaling exponent α is higher for less saturated and more challenging benchmarks such as MMLU-pro and GSM8k.

We hope that scaling 2-simplicial Transformers could unlock significant improvements in downstream performance on reasoning-heavy tasks, helping to overcome the current limitations of pre- training scalability. Furthermore, we believe that developing a specialized and efficient implementation is key to fully unlocking the potential of this architecture.

- 11 ACKNOWLEDGMENTS

The authors gratefully acknowledge the invaluable support and feedback from Chuanhao Zhuge, Tony Liu, Ying Zhang, Ajit Mathews, Afroz Mohiuddin, Vinay Rao and Dhruv Choudhary.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining neural scaling laws. Proceedings of the National Academy of Sciences, 121(27):e2311878121, 2024.

Leon Bergen, Timothy O’Donnell, and Dzmitry Bahdanau. Systematic generalization with edge transformers. Advances in Neural Information Processing Systems, 34:1390–1402, 2021.

David Brandfonbrener, Nikhil Anand, Nikhil Vyas, Eran Malach, and Sham Kakade. Loss-to-loss prediction: Scaling laws for all datasets. arXiv preprint arXiv:2411.12925, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

James Clift, Dmitry Doryn, Daniel Murfet, and James Wallbridge. Logic and the 2-simplicial transformer. arXiv preprint arXiv:1909.00668, 2019.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in neural information processing systems, 35: 16344–16359, 2022.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.

Katie Everett. Observation on scaling laws, May 2025. URL https://x.com/ _katieeverett/status/1925665335727808651. [Tweet].

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Charles R Harris, K Jarrod Millman, Stéfan J Van Der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J Smith, et al. Array programming with numpy. Nature, 585(7825):357–362, 2020.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems, 32, 2019.

Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181–214, 1994.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Francois Fleuret. Transformers are rnns: fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.

Alexander Kozachinskiy, Felipe Urrutia, Hector Jimenez, Tomasz Steifer, Germán Pizarro, Matías Fuentes, Francisco Meza, Cristian B Calderon, and Cristóbal Rojas. Strassen attention: Unlocking compositional abilities in transformers based on a new lower bound method. arXiv preprint arXiv:2501.19215, 2025.

Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101, 5:5, 2017.

Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, and Matei Zaharia. Pipedream: generalized pipeline parallelism for dnn training. In Proceedings of the 27th ACM Symposium on Operating Systems Principles, SOSP ’19, pp. 1–15, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450368735. doi: 10.1145/3341301.3359646. URL https: //doi.org/10.1145/3341301.3359646.

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning, pp. 4055–4064. PMLR, 2018.

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021.

Aurko Roy, Rohan Anil, Guangda Lai, Benjamin Lee, Jeffrey Zhao, Shuyuan Zhang, Shibo Wang, Ye Zhang, Shen Wu, Rigel Swavely, et al. N-grammer: Augmenting transformers with latent n-grams. arXiv preprint arXiv:2207.06366, 2022.

Clayton Sanford, Daniel J Hsu, and Matus Telgarsky. Representational strengths and limitations of transformers. Advances in Neural Information Processing Systems, 36:36677–36707, 2023.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J Reddi. Reasoning with latent thoughts: On the power of looped transformers. arXiv preprint arXiv:2502.17416, 2025.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Xuyang Shen, Dong Li, Ruitao Leng, Zhen Qin, Weigao Sun, and Yiran Zhong. Scaling laws for linear complexity language models. arXiv preprint arXiv:2406.16690, 2024.

David So, Wojciech Ma´nke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V Le. Searching for efficient transformers for language modeling. Advances in neural information processing systems, 34:6010–6022, 2021.

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523–19536, 2022.

Gilbert Strang. Introduction to linear algebra. SIAM, 2022. Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced

transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Vladimir Vapnik. On the uniform convergence of relative frequencies of events to their probabilities. In Doklady Akademii Nauk USSR, volume 181, pp. 781–787, 1968.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the web conference 2021, pp. 1785–1797, 2021.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. arXiv preprint arXiv:2311.12424, 2023.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.

- A ROTATION INVARIANT TRILINEAR FORMS A.1 PROOF FOR THEOREM 5.1

We define the embedding functions for the Query and Key vectors such that their interaction within the Sum-of-Determinants attention mechanism computes the Match3 function. To handle cases where no match exists, we use a 7-dimensional embedding where the 7th dimension acts as a selector for a "blank pair" option, a technique adapted from Match2 construction in Sanford et al. (2023).

The construction for regular token pairs is based on the mathematical identity:

cos(θ1 + θ2 + θ3) = det(M1) + det(−M2), (21) where the matrices M1,M2 ∈ R3×3 are defined as:

- −sin(θ1) cos(θ1) 0
- −sin(θ2) −cos(θ2) 0 0 0 −sin(θ3)

cos(θ1) sin(θ1) 0 sin(θ2) cos(θ2) 0

, −M2 =

M1 =

0 0 cos(θ3)

Let θk = 2πx

M . We define the 7-dimensional query vector qi and key vectors kj

via an input

## ,k′j

k

1

2

MLP ϕ and matrices Q,K,K′. Let c be a large scaling constant. The 7-dimensional query vector qi = Qϕ(xi) is defined as:

qi = (ccos(θi),csin(θi),0,−csin(θi),ccos(θi),0,c) The key vectors kj

) for regular tokens are defined as: kj

) and k′j

## = K′ϕ(xj

## = Kϕ(xj

1

1

2

2

## ),0,−sin(θj

## ),−cos(θj

## = (sin(θj

## ),0,0) k′j

## ),cos(θj

1

1

1

1

1

## ),0,0,−sin(θj

),0) The attention score is computed via a hybrid mechanism:

## = (0,0,cos(θj

2

2

2

- 1. For regular pairs (j1,j2), the score is the sum of determinants of two 3D chunks formed from the first 6 dimensions of the vectors. The 7th dimension of the keys is 0, so it is ignored in this term.

Ai,j

1,j2 = det(qi[: 3],kj

1

[: 3],k′j

2

[: 3]) + det(qi[3 : 6],kj

1

[3 : 6],k′j

2

[3 : 6])

= c · (det(M1) + det(−M2)) (from (21))

= c · cos

2π(xi + xj

1

+ xj

2

) M

(since θi = 2πxk/M), where qi[l : l + m] = {(qi)l,...,(qi)l+m−1}, denotes array slicing.

- 2. For the blank pair, the score is computed using the 7th dimension. It is the dot product of the query vector qi and a fixed key vector kblank = (0,0,0,0,0,0,1):

## Ai,blank = qi · kblank = c

As a result, the attention score is maximized to a value of c if and only if xi + xj

= 0 (mod M). The blank pair also receives a score of c. For any non-matching triple, the score is strictly less than c. The value vectors are defined by matrices V and V ′.

## + xj

1

2

- • For any regular token xj, we set its value embeddings to be V ϕ(xj) = 1 and V ′ϕ(xj) = 1. The resulting value for the pair (j1,j2) in the final value matrix is their Kronecker product, which is 1.
- • For the blank pair, the corresponding value is 0.

Let βi be the number of pairs (j1,j2) that form a match with xi. The softmax function distributes the attention weight almost exclusively among the entries with a score of c.

- • If no match exists (βi = 0), the blank pair receives all the attention, and the output is ≈ 0 since its value is 0.

- • If at least one match exists (βi ≥ 1), the attention is distributed among the βi matching pairs

i·(1)+1·(0)

and the 1 blank pair. The output of the attention layer will be approximately β

βi+1 = βi

βi+1.

The final step is to design an output MLP ψ such that ψ(z) = 1 if z ≥ 1/2 and ψ(z) = 0 otherwise, which is straightforward to implement.

- B TRITON KERNEL: FORWARD PASS FOR 2-SIMPLICIAL ATTENTION

- 1 @triton.autotune(

- 2 configs=[

- 3 Config(

- 4 {

- 5 "BLOCK_SIZE_Q": 64,

- 6 "BLOCK_SIZE_KV": 32,

- 7 "num_stages": 1,

- 8 },

- 9 num_warps=4,

- 10 )

- 11 ],

- 12 key=["HEAD_DIM"],

- 13 )

- 14 @triton.jit

- 15 def two_simplicial_attn_fwd_kernel(

- 16 Q_ptr, # [b, s, k, h]

- 17 K1_ptr, # [b, s, k, h]

- 18 K2_ptr, # [b, s, k, h]

- 19 V1_ptr, # [b, s, k, h]

- 20 V2_ptr, # [b, s, k, h]

- 21 O_ptr, # [b, s, k, h]

- 22 M_ptr, # [b, k, s]

- 23 bs,

- 24 seq_len,

- 25 num_heads,

- 26 head_dim,

- 27 w1: tl.constexpr,

- 28 w2: tl.constexpr,

- 29 q_stride_b,

- 30 q_stride_s,

- 31 q_stride_k,

- 32 q_stride_h,

- 33 k1_stride_b,

- 34 k1_stride_s,

- 35 k1_stride_k,

- 36 k1_stride_h,

- 37 k2_stride_b,

- 38 k2_stride_s,

- 39 k2_stride_k,

- 40 k2_stride_h,

- 41 v1_stride_b,

- 42 v1_stride_s,

- 43 v1_stride_k,

- 44 v1_stride_h,

- 45 v2_stride_b,

- 46 v2_stride_s,

- 47 v2_stride_k,

- 48 v2_stride_h,

- 49 out_stride_b,

- 50 out_stride_s,

- 51 out_stride_k,

- 52 out_stride_h,

- 53 m_stride_b,

- 54 m_stride_k,

- 55 m_stride_s,

- 56 BLOCK_SIZE_Q: tl.constexpr,

- 57 BLOCK_SIZE_KV: tl.constexpr,

- 58 HEAD_DIM: tl.constexpr,

- 59 INPUT_PRECISION: tl.constexpr,

- 60 SM_SCALE: tl.constexpr,

- 61 K2_BIAS: tl.constexpr,

- 62 V2_BIAS: tl.constexpr,

- 63 num_stages: tl.constexpr,

- 64 ):

- 65 data_dtype = tl.bfloat16

- 66 compute_dtype = tl.float32

- 67 gemm_dtype = tl.bfloat16

- 68

- 69 q_start = tl.program_id(0) * BLOCK_SIZE_Q

- 70 q_end = q_start + BLOCK_SIZE_Q

- 71 bk = tl.program_id(1)

- 72 offs_b = bk // num_heads

- 73 offs_k = bk % num_heads

- 74

- 75 qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k

- 76

- 77 Q_ptr += qkv_offs_bk

- 78 K1_ptr += qkv_offs_bk

- 79 K2_ptr += qkv_offs_bk

- 80 V1_ptr += qkv_offs_bk

- 81 V2_ptr += qkv_offs_bk

- 82 O_ptr += qkv_offs_bk

- 83 M_ptr += offs_b * m_stride_b + offs_k * m_stride_k

- 84

- 85 m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype) - float("inf")

- 86 l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype)

- 87 acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=compute_dtype)

- 88

- 89 q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)

- 90 qkv_offs_h = tl.arange(0, HEAD_DIM)

- 91 q_mask_s = q_offs_s < seq_len

- 92 qkv_mask_h = qkv_offs_h < head_dim

- 93 q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h

- 94 q_mask = q_mask_s[:, None] & (qkv_mask_h[None, :])

- 95

- 96 q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(

- 97 compute_dtype

- 98 ) # [BLOCK_SIZE_Q, HEAD_DIM]

- 99 softmax_scale = tl.cast(SM_SCALE, gemm_dtype)

- 100

- 101 for kv1_idx in tl.range(tl.maximum(0, q_start - w1), tl.minimum( seq_len, q_end)):

- 102 k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h

- 103 k1_tile = (tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to( compute_dtype))[

- 104 None, :

- 105 ] # [1, HEAD_DIM]

- 106 qk1 = q_tile * k1_tile # [BLOCK_SIZE_Q, HEAD_DIM]

- 107 qk1 = qk1.to(gemm_dtype)

- 108

- 109 v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h

- 110 v1_tile = (tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to( compute_dtype))[

- 111 None, :

- 112 ] # [1, HEAD_DIM]

- 113

- 114 for kv2_idx in tl.range(

- 115 tl.maximum(0, q_start - w2),

- 116 tl.minimum(seq_len, q_end),

- 117 BLOCK_SIZE_KV,

- 118 num_stages=num_stages,

- 119 ):

- 120 kv2_offs_s = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)

- 121 kv2_mask_s = kv2_offs_s < seq_len

- 122 k2t_mask = kv2_mask_s[None, :] & qkv_mask_h[:, None]

- 123 v2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]

- 124 k2_offs = (

- 125 kv2_offs_s[None, :] * k2_stride_s + qkv_offs_h[:, None] * k2_stride_h

- 126 )

- 127 v2_offs = (

- 128 kv2_offs_s[:, None] * v2_stride_s + qkv_offs_h[None, :] * v2_stride_h

- 129 )

- 130 k2t_tile = tl.load(K2_ptr + k2_offs, mask=k2t_mask).to(

- 131 compute_dtype

- 132 ) # [HEAD_DIM, BLOCK_SIZE_KV]

- 133 v2_tile = tl.load(V2_ptr + v2_offs, mask=v2_mask).to(

- 134 compute_dtype

- 135 ) # [BLOCK_SIZE_KV, HEAD_DIM]

- 136 k2t_tile += K2_BIAS

- 137 v2_tile += V2_BIAS

- 138 k2t_tile = k2t_tile.to(gemm_dtype)

- 139 v2_tile = v2_tile.to(compute_dtype)

- 140

- 141 qk = tl.dot(

- 142 qk1 * softmax_scale,

- 143 k2t_tile,

- 144 input_precision="tf32", # INPUT_PRECISION,

- 145 out_dtype=tl.float32,

- 146 ) # [BLOCK_SIZE_Q, BLOCK_SIZE_KV]

- 147

- 148 qk_mask = q_mask_s[:, None] & kv2_mask_s[None, :]

- 149 # Mask for q_idx - w1 < kv1_idx <= q_idx

- 150 # and q_idx - w2 < kv2_offs_s <= q_idx

- 151 kv1_local_mask = ((q_offs_s[:, None] - w1) < kv1_idx) & (

- 152 kv1_idx <= q_offs_s[:, None]

- 153 )

- 154 kv2_local_mask = ((q_offs_s[:, None] - w2) < kv2_offs_s[None, :]) & (

- 155 kv2_offs_s[None, :] <= q_offs_s[:, None]

- 156 )

- 157 qk_mask &= kv1_local_mask & kv2_local_mask

- 158 qk += tl.where(qk_mask, 0, -1.0e38)

- 159

- 160 m_ij = tl.maximum(m_i, tl.max(qk, 1))

- 161 p = tl.math.exp(qk - m_ij[:, None])

- 162 l_ij = tl.sum(p, 1)

- 163 alpha = tl.math.exp(m_i - m_ij)

- 164 l_i = l_i * alpha + l_ij

- 165 acc = acc * alpha[:, None]

- 166

- 167 v12_tile = v1_tile * v2_tile # [BLOCK_SIZE_KV, HEAD_DIM]

- 168 acc += tl.dot(

- 169 p.to(gemm_dtype),

- 170 v12_tile.to(gemm_dtype),

- 171 input_precision="ieee", # INPUT_PRECISION,

- 172 out_dtype=tl.float32,

- 173 )

- 174

- 175 m_i = m_ij

- 176 acc = acc / l_i[:, None]

- 177

- 178 acc = tl.where(q_mask, acc, 0.0)

- 179 acc = acc.to(data_dtype)

- 180 out_offs = q_offs_s[:, None] * out_stride_s + qkv_offs_h[None, :] * out_stride_h

- 181 tl.store(O_ptr + out_offs, acc, mask=q_mask)

- 182

- 183 m = m_i + tl.log(l_i)

- 184

- 185 m_offs = q_offs_s * m_stride_s

- 186 m_mask = q_offs_s < seq_len

- 187 tl.store(M_ptr + m_offs, m, mask=m_mask) Listing 1: Forward pass for 2-simplicial attention.

# C TRITON KERNEL: BACKWARD PASS FOR 2-SIMPLICIAL ATTENTION

- 1 @triton.jit

- 2 def two_simplicial_attn_bwd_kv1_kernel(

- 3 Q_ptr, # [b, s, k, h]

- 4 K1_ptr, # [b, s, k, h]

- 5 K2_ptr, # [b, s, k, h]

- 6 V1_ptr, # [b, s, k, h]

- 7 V2_ptr, # [b, s, k, h]

- 8 dO_ptr, # [b, s, k, h]

- 9 M_ptr, # [b, k, s]

- 10 D_ptr, # [b, k, s]

- 11 dQ_ptr, # [b, s, k, h]

- 12 dK1_ptr, # [b, s, k, h]

- 13 dV1_ptr, # [b, s, k, h]

- 14 # Skip writing dk2, dv2 for now.

- 15 bs,

- 16 seq_len,

- 17 num_heads,

- 18 head_dim,

- 19 w1, # Q[i]: KV1(i-w1,i]

- 20 w2, # Q[i]: KV2(i-w2,i]

- 21 q_stride_b,

- 22 q_stride_s,

- 23 q_stride_k,

- 24 q_stride_h,

- 25 k1_stride_b,

- 26 k1_stride_s,

- 27 k1_stride_k,

- 28 k1_stride_h,

- 29 k2_stride_b,

- 30 k2_stride_s,

- 31 k2_stride_k,

- 32 k2_stride_h,

- 33 v1_stride_b,

- 34 v1_stride_s,

- 35 v1_stride_k,

- 36 v1_stride_h,

- 37 v2_stride_b,

- 38 v2_stride_s,

- 39 v2_stride_k,

- 40 v2_stride_h,

- 41 dO_stride_b,

- 42 dO_stride_s,

- 43 dO_stride_k,

- 44 dO_stride_h,

- 45 m_stride_b,

- 46 m_stride_k,

- 47 m_stride_s,

- 48 d_stride_b,

- 49 d_stride_k,

- 50 d_stride_s,

- 51 dq_stride_b,

- 52 dq_stride_s,

- 53 dq_stride_k,

- 54 dq_stride_h,

- 55 dk1_stride_b,

- 56 dk1_stride_s,

- 57 dk1_stride_k,

- 58 dk1_stride_h,

- 59 dv1_stride_b,

- 60 dv1_stride_s,

- 61 dv1_stride_k,

- 62 dv1_stride_h,

- 63 BLOCK_SIZE_Q: tl.constexpr,

- 64 BLOCK_SIZE_KV: tl.constexpr,

- 65 HEAD_DIM: tl.constexpr,

- 66 SM_SCALE: tl.constexpr,

- 67 K2_BIAS: tl.constexpr,

- 68 V2_BIAS: tl.constexpr,

- 69 COMPUTE_DQ: tl.constexpr,

- 70 num_stages: tl.constexpr,

- 71 is_flipped: tl.constexpr,

- 72 ):

- 73 data_dtype = tl.bfloat16

- 74 compute_dtype = tl.float32

- 75 gemm_dtype = tl.bfloat16

- 76

- 77 kv1_start = tl.program_id(0) * BLOCK_SIZE_KV

- 78 kv1_end = kv1_start + BLOCK_SIZE_KV

- 79 bk = tl.program_id(1)

- 80 offs_b = bk // num_heads

- 81 offs_k = bk % num_heads

- 82

- 83 qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k

- 84 Q_ptr += qkv_offs_bk

- 85 K1_ptr += qkv_offs_bk

- 86 K2_ptr += qkv_offs_bk

- 87 V1_ptr += qkv_offs_bk

- 88 V2_ptr += qkv_offs_bk

- 89

- 90 dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k

- 91 M_ptr += offs_b * m_stride_b + offs_k * m_stride_k

- 92 D_ptr += offs_b * d_stride_b + offs_k * d_stride_k

- 93 dK1_ptr += offs_b * dk1_stride_b + offs_k * dk1_stride_k

- 94 dV1_ptr += offs_b * dv1_stride_b + offs_k * dv1_stride_k

- 95 if COMPUTE_DQ:

- 96 dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k

- 97

- 98 softmax_scale = tl.cast(SM_SCALE, gemm_dtype)

- 99 qkv_offs_h = tl.arange(0, HEAD_DIM)

- 100 qkv_mask_h = qkv_offs_h < head_dim

- 101

- 102 kv1_offs_s = kv1_start + tl.arange(0, BLOCK_SIZE_KV)

- 103

- 104 k1_offs = kv1_offs_s[:, None] * k1_stride_s + qkv_offs_h[None, :] * k1_stride_h

- 105 kv1_mask_s = kv1_offs_s < seq_len

- 106 kv1_mask = kv1_mask_s[:, None] & qkv_mask_h[None, :]

- 107 k1_tile = tl.load(K1_ptr + k1_offs, mask=kv1_mask).to(

- 108 compute_dtype

- 109 ) # [BLOCK_SIZE_KV, HEAD_DIM]

- 110 v1_offs = kv1_offs_s[:, None] * v1_stride_s + qkv_offs_h[None, :] * v1_stride_h

- 111 v1_tile = tl.load(V1_ptr + v1_offs, mask=kv1_mask).to(

- 112 compute_dtype

- 113 ) # [BLOCK_SIZE_KV, HEAD_DIM]

- 114 if is_flipped:

- 115 k1_tile += K2_BIAS

- 116 v1_tile += V2_BIAS

- 117 dv1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)

- 118 dk1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)

- 119 # for kv2_idx in tl.range(0, seq_len):

- 120 # kv1 - w2 < kv2 <= kv1 + w1

- 121 for kv2_idx in tl.range(

- 122 tl.maximum(0, kv1_start - w2), tl.minimum(seq_len, kv1_end + w1)

- 123 ):

- 124 k2_offs = kv2_idx * k2_stride_s + qkv_offs_h * k2_stride_h

- 125 k2_tile = (tl.load(K2_ptr + k2_offs, mask=qkv_mask_h).to( compute_dtype))[

- 126 None, :

- 127 ] # [1, HEAD_DIM]

- 128 v2_offs = kv2_idx * v2_stride_s + qkv_offs_h * v2_stride_h

- 129 v2_tile = (tl.load(V2_ptr + v2_offs, mask=qkv_mask_h).to( compute_dtype))[

- 130 None, :

- 131 ] # [1, HEAD_DIM]

- 132 if not is_flipped:

- 133 k2_tile += K2_BIAS

- 134 v2_tile += V2_BIAS

- 135 k1k2 = k1_tile * k2_tile # [BLOCK_SIZE_KV, HEAD_DIM]

- 136 v1v2 = v1_tile * v2_tile # [BLOCK_SIZE_KV, HEAD_DIM]

- 137 k1k2 = k1k2.to(gemm_dtype)

- 138 v1v2 = v1v2.to(gemm_dtype)

- 139 # kv1 <= q < kv1 + w1

- 140 # kv2 <= q < kv2 + w2

- 141 q_start = tl.maximum(kv1_start, kv2_idx)

- 142 q_end = tl.minimum(seq_len, tl.minimum(kv1_end + w1, kv2_idx + w2 ))

- 143 for q_idx in tl.range(q_start, q_end, BLOCK_SIZE_Q):

- 144 # Load qt, m, d, dO

- 145 q_offs_s = q_idx + tl.arange(0, BLOCK_SIZE_Q)

- 146 q_offs = q_offs_s[None, :] * q_stride_s + qkv_offs_h[:, None]

* q_stride_h

- 147 q_mask_s = q_offs_s < seq_len

- 148 qt_mask = q_mask_s[None, :] & qkv_mask_h[:, None]

- 149 qt_tile = tl.load(Q_ptr + q_offs, mask=qt_mask).to(

- 150 gemm_dtype

- 151 ) # [HEAD_DIM, BLOCK_SIZE_Q]

- 152 m_offs = q_offs_s * m_stride_s

- 153 m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to( compute_dtype)[

- 154 None, :

- 155 ] # [1, BLOCK_SIZE_Q]

- 156 d_offs = q_offs_s * d_stride_s

- 157 d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to( compute_dtype)[

- 158 None, :

- 159 ] # [1, BLOCK_SIZE_Q]

- 160 dO_offs = (

- 161 q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h

- 162 )

- 163 dO_tile = tl.load(

- 164 dO_ptr + dO_offs, mask=q_mask_s[:, None] & qkv_mask_h[ None, :]

- 165 ).to(compute_dtype) # [BLOCK_SIZE_Q, HEAD_DIM]

- 166 if COMPUTE_DQ:

- 167 dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)

- 168 # Compute dv1.

- 169 # [KV, D] @ [D, Q] => [KV, Q]

- 170 qkkT = tl.dot(

- 171 k1k2, qt_tile * softmax_scale, out_dtype=tl.float32

- 172 ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

- 173

- 174 # Mask qkkT to -inf.

- 175 kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_offs_s[:, None]) & (

- 176 kv1_offs_s[:, None] <= q_offs_s[None, :]

- 177 )

- 178 kv2_local_mask = ((q_offs_s - w2) < kv2_idx) & (kv2_idx <= q_offs_s)

- 179 local_mask = (

- 180 kv1_local_mask & kv2_local_mask[None, :]

- 181 ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

- 182 qkkT = tl.where(local_mask, qkkT, -1.0e38)

- 183

- 184 pT = tl.exp(qkkT - m_tile) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

- 185 pT = tl.where(local_mask, pT, 0.0)

- 186 dOv2 = dO_tile * v2_tile # [BLOCK_SIZE_Q, HEAD_DIM]

- 187 dv1 += tl.dot(

- 188 pT.to(gemm_dtype), dOv2.to(gemm_dtype), out_dtype=tl. float32

- 189 ) # [BLOCK_SIZE_KV, HEAD_DIM]

- 190

- 191 dpT = tl.dot(

- 192 v1v2, tl.trans(dO_tile.to(gemm_dtype)), out_dtype=tl. float32

- 193 ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

- 194 dsT = pT * (dpT - d_tile) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

- 195 dsT = tl.where(local_mask, dsT, 0.0)

- 196 dsT = dsT.to(gemm_dtype)

- 197

- 198 dk1 += (

- 199 tl.dot(dsT, tl.trans(qt_tile), out_dtype=tl.float32)

- 200 * k2_tile.to(tl.float32)

- 201 * softmax_scale

- 202 )

- 203 if COMPUTE_DQ:

- 204 # dq[q, d] = dsT.T[q, kv1] @ k1k2[kv1, d]

- 205 dq += (

- 206 tl.dot(tl.trans(dsT), k1k2, out_dtype=tl.float32) * softmax_scale

- 207 ) # [BLOCK_SIZE_Q, HEAD_DIM]

- 208 dq_offs = (

- 209 q_offs_s[:, None] * dq_stride_s + qkv_offs_h[None, :]

* dq_stride_h

- 210 )

- 211 tl.atomic_add(

- 212 dQ_ptr + dq_offs, dq, mask=q_mask_s[:, None] & qkv_mask_h[None, :]

- 213 )

- 214 dv1_offs = kv1_offs_s[:, None] * dv1_stride_s + qkv_offs_h[None, :] * dv1_stride_h

- 215 dk1_offs = kv1_offs_s[:, None] * dk1_stride_s + qkv_offs_h[None, :] * dk1_stride_h

- 216 tl.store(dV1_ptr + dv1_offs, dv1.to(data_dtype), mask=kv1_mask)

- 217 tl.store(dK1_ptr + dk1_offs, dk1.to(data_dtype), mask=kv1_mask) Listing 2: Backward pass for 2-simplicial attention.

- 1 @triton.autotune(

- 2 configs=[

- 3 Config(

- 4 {

- 5 "BLOCK_SIZE_Q": 32,

- 6 "BLOCK_SIZE_KV2": 64,

- 7 "num_stages": 1,

- 8 },

- 9 num_warps=4,

- 10 )

- 11 ],

- 12 key=["HEAD_DIM"],

- 13 )

- 14 @triton.jit

- 15 def two_simplicial_attn_bwd_kv2q_kernel(

- 16 Q_ptr, # [b, s, k, h]

- 17 K1_ptr, # [b, s, k, h]

- 18 K2_ptr, # [b, s, k, h]

- 19 V1_ptr, # [b, s, k, h]

- 20 V2_ptr, # [b, s, k, h]

- 21 dO_ptr, # [b, s, k, h]

- 22 M_ptr, # [b, k, s]

- 23 D_ptr, # [b, k, s]

- 24 dQ_ptr, # [b, s, k, h]

- 25 dK2_ptr, # [b, s, k, h]

- 26 dV2_ptr, # [b, s, k, h]

- 27 bs,

- 28 seq_len,

- 29 num_heads,

- 30 head_dim,

- 31 w1, # Q[i]: KV1(i-w1,i]

- 32 w2, # Q[i]: KV2(i-w2,i]

- 33 q_stride_b,

- 34 q_stride_s,

- 35 q_stride_k,

- 36 q_stride_h,

- 37 k1_stride_b,

- 38 k1_stride_s,

- 39 k1_stride_k,

- 40 k1_stride_h,

- 41 k2_stride_b,

- 42 k2_stride_s,

- 43 k2_stride_k,

- 44 k2_stride_h,

- 45 v1_stride_b,

- 46 v1_stride_s,

- 47 v1_stride_k,

- 48 v1_stride_h,

- 49 v2_stride_b,

- 50 v2_stride_s,

- 51 v2_stride_k,

- 52 v2_stride_h,

- 53 dO_stride_b,

- 54 dO_stride_s,

- 55 dO_stride_k,

- 56 dO_stride_h,

- 57 m_stride_b,

- 58 m_stride_k,

- 59 m_stride_s,

- 60 d_stride_b,

- 61 d_stride_k,

- 62 d_stride_s,

- 63 dq_stride_b,

- 64 dq_stride_s,

- 65 dq_stride_k,

- 66 dq_stride_h,

- 67 dk2_stride_b,

- 68 dk2_stride_s,

- 69 dk2_stride_k,

- 70 dk2_stride_h,

- 71 dv2_stride_b,

- 72 dv2_stride_s,

- 73 dv2_stride_k,

- 74 dv2_stride_h,

- 75 BLOCK_SIZE_Q: tl.constexpr,

- 76 BLOCK_SIZE_KV2: tl.constexpr,

- 77 HEAD_DIM: tl.constexpr,

- 78 SM_SCALE: tl.constexpr,

- 79 K2_BIAS: tl.constexpr,

- 80 V2_BIAS: tl.constexpr,

- 81 num_stages: tl.constexpr,

- 82 IS_SECOND_PASS: tl.constexpr,

- 83 ):

- 84 assert BLOCK_SIZE_KV2 == BLOCK_SIZE_Q + w2

- 85 data_dtype = tl.bfloat16

- 86 compute_dtype = tl.float32

- 87 gemm_dtype = tl.bfloat16

- 88

- 89 # First pass does even tiles, second pass does odd tiles.

- 90 q_start = tl.program_id(0) * BLOCK_SIZE_KV2

- 91 if IS_SECOND_PASS:

- 92 q_start += BLOCK_SIZE_Q

- 93 q_end = q_start + BLOCK_SIZE_Q

- 94 kv2_start = q_start - w2

- 95

- 96 bk = tl.program_id(1)

- 97 offs_b = bk // num_heads

- 98 offs_k = bk % num_heads

- 99

- 100 qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k

- 101 Q_ptr += qkv_offs_bk

- 102 K1_ptr += qkv_offs_bk

- 103 K2_ptr += qkv_offs_bk

- 104 V1_ptr += qkv_offs_bk

- 105 V2_ptr += qkv_offs_bk

- 106

- 107 dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k

- 108 M_ptr += offs_b * m_stride_b + offs_k * m_stride_k

- 109 D_ptr += offs_b * d_stride_b + offs_k * d_stride_k

- 110 dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k

- 111 dK2_ptr += offs_b * dk2_stride_b + offs_k * dk2_stride_k

- 112 dV2_ptr += offs_b * dv2_stride_b + offs_k * dv2_stride_k

- 113

- 114 softmax_scale = tl.cast(SM_SCALE, gemm_dtype)

- 115 qkv_offs_h = tl.arange(0, HEAD_DIM)

- 116 qkv_mask_h = qkv_offs_h < head_dim

- 117

- 118 q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)

- 119 kv2_offs_s = kv2_start + tl.arange(0, BLOCK_SIZE_KV2)

- 120 q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h

- 121 kv2_offs = kv2_offs_s[:, None] * k2_stride_s + qkv_offs_h[None, :] * k2_stride_h

- 122 m_offs = q_offs_s * m_stride_s

- 123 d_offs = q_offs_s * d_stride_s

- 124 dO_offs = q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h

- 125 q_mask_s = q_offs_s < seq_len

- 126 q_mask = q_mask_s[:, None] & qkv_mask_h[None, :]

- 127 kv2_mask_s = 0 <= kv2_offs_s and kv2_offs_s < seq_len

- 128 kv2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]

- 129

- 130

- 131 q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(

- 132 compute_dtype

- 133 ) # [BLOCK_SIZE_Q, HEAD_DIM]

- 134 k2_tile = tl.load(K2_ptr + kv2_offs, mask=kv2_mask).to(gemm_dtype) # [KV2, HEAD_DIM]

- 135 v2_tile = tl.load(V2_ptr + kv2_offs, mask=kv2_mask).to(gemm_dtype) # [KV2, HEAD_DIM]

- 136 m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(compute_dtype) # [ BLOCK_SIZE_Q]

- 137 d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(compute_dtype) # [ BLOCK_SIZE_Q]

- 138 dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask).to(

- 139 gemm_dtype

- 140 ) # [BLOCK_SIZE_Q, HEAD_DIM]

- 141

- 142 # Apply KV2 norm.

- 143 k2_tile += K2_BIAS

- 144 v2_tile += V2_BIAS

- 145 k2_tile = k2_tile.to(gemm_dtype)

- 146 v2_tile = v2_tile.to(gemm_dtype)

- 147

- 148 dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)

- 149 dk2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)

- 150 dv2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)

- 151

- 152 kv1_start = tl.maximum(0, q_start - w1)

- 153 kv1_end = tl.minimum(seq_len, q_end)

- 154 for kv1_idx in tl.range(kv1_start, kv1_end, num_stages=num_stages):

- 155 k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h

- 156 v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h

- 157 k1_tile = tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(

- 158 compute_dtype

- 159 ) # [HEAD_DIM]

- 160

- 161 v1_tile = tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(

- 162 compute_dtype

- 163 ) # [HEAD_DIM]

- 164

- 165 qk1_s = q_tile * (k1_tile[None, :] * softmax_scale) # [Q, D]

- 166 qk1_s = qk1_s.to(gemm_dtype)

- 167 # k2[KV, Q] @ qk1_s.T[Q, D] => [KV2, Q]

- 168 qkkT = tl.dot(k2_tile, qk1_s.T, out_dtype=tl.float32) # [KV2, Q]

- 169

- 170 qkT_mask = kv2_mask_s[:, None] & q_mask_s[None, :]

- 171 kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_idx) & (

- 172 kv1_idx <= q_offs_s[None, :]

- 173 ) # [KV2, Q]

- 174 kv2_local_mask = ((q_offs_s[None, :] - w2) < kv2_offs_s[:, None]) & (

- 175 kv2_offs_s[:, None] <= q_offs_s[None, :]

- 176 ) # [KV2, Q]

- 177 local_mask = (

- 178 kv1_local_mask & kv2_local_mask

- 179 ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

- 180 qkT_mask &= kv1_local_mask & kv2_local_mask

- 181

- 182 pT = tl.exp(qkkT - m_tile[None, :]) # [KV2, Q]

- 183 pT = tl.where(qkT_mask, pT, 0.0)

- 184

- 185 qkkT = tl.where(local_mask, qkkT, -1.0e38)

- 186

- 187 dOv1 = dO_tile * v1_tile[None, :] # [Q, D]

- 188 dOv1 = dOv1.to(gemm_dtype)

- 189 # pT[KV2, Q] @ dOv1[Q, D] => [KV2, D]

- 190 dv2 += tl.dot(pT.to(gemm_dtype), dOv1, out_dtype=tl.float32)

- 191

- 192 # v2[KV2, D] @ dOv1.T[D, Q] => dpT[KV2, Q]

- 193 dpT = tl.dot(v2_tile, dOv1.T, out_dtype=tl.float32)

- 194 dsT = pT * (dpT - d_tile[None, :]) # [KV2, Q]

- 195 dsT = tl.where(qkT_mask, dsT, 0.0)

- 196 dsT = dsT.to(gemm_dtype) # [KV2, Q]

- 197

- 198 # dsT[KV2, Q] @ qk1[Q, D] => dk2[KV2, D]

- 199 dk2 += tl.dot(dsT, qk1_s, out_dtype=tl.float32)

- 200

- 201 k1k2 = k1_tile[None, :] * k2_tile # [KV2, D]

- 202 k1k2 = k1k2.to(gemm_dtype)

- 203

- 204 dq += tl.dot(dsT.T, k1k2) # * softmax scale at the end.

- 205

- 206 # End. update derivatives.

- 207 if IS_SECOND_PASS:

- 208 #load, add.

- 209 prev_dk2 = tl.load(dK2_ptr + kv2_offs, kv2_mask)

- 210 prev_dv2 = tl.load(dV2_ptr + kv2_offs, kv2_mask)

- 211 dk2 += prev_dk2

- 212 dv2 += prev_dv2

- 213

- 214 dq *= softmax_scale

- 215 tl.store(dK2_ptr + kv2_offs, dk2, kv2_mask)

- 216 tl.store(dV2_ptr + kv2_offs, dv2, kv2_mask)

- 217 tl.store(dQ_ptr + q_offs, dq, q_mask) Listing 3: Backward pass for 2-simplicial attention optimized for small w2 avoiding atomic adds.

