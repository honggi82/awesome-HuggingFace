## arXiv:2510.27258v3[cs.LG]14May2026

# Higher-order Linear Attention

Yifan Zhang1 Zhen Qin Mengdi Wang1 Quanquan Gu2 1Princeton University 2University of California, Los Angeles yifzhang@princeton.edu

Abstract

The quadratic cost of scaled dot-product attention is a central obstacle to scaling autoregressive language models to long contexts. Linear-time attention and State Space Models (SSMs) provide scalable alternatives but are typically restricted to first-order or kernel-based approximations, which can limit expressivity. We introduce Higher-order Linear Attention (HLA), a causal, streaming mechanism that realizes higher interactions via compact prefix sufficient statistics. In the second-order case, HLA maintains a constant-size state and computes per-token outputs in linear time without materializing any n×n matrices. We give closed-form streaming identities, a strictly causal masked variant using two additional summaries, and a chunk-parallel training scheme based on associative scans that reproduces the activations of a serial recurrence exactly. We also give the masked third-order streaming kernel and its exact chunk-parallel scan, which uses additional segment maps to compose third-order corrected states. Collectively, these results position HLA as a principled, scalable building block that combines attention-like, data-dependent mixing with the efficiency of modern recurrent architectures.

Project Page: https://github.com/yifanzhang-pro/HLA

### 1 Introduction

The Transformer architecture (Vaswani et al., 2017), powered by scaled dot-product attention, underpins modern large language models (LLMs). Yet the O(n2) computational and memory complexity in sequence length n constrains long-context use. A rich line of work therefore explores more efficient attention mechanisms (e.g., Linear Attention (Katharopoulos et al., 2020; Wang et al., 2020; Choromanski et al., 2020; Schlag et al., 2021; Sun et al., 2023; Qin et al., 2023; Yang et al., 2023; Qin et al., 2024; Yang et al., 2024b; von Oswald et al., 2025)), Modern Recurrent Neural Networks (RNNs) (Peng et al., 2023, 2024; Sun et al., 2024; Peng et al., 2025), Fast Weight Programmers (Delta Networks, Schlag et al. (2021)), State Space Models (SSMs) (Gu et al., 2021; Gu and Dao, 2023; Dao and Gu, 2024) and Memory Networks (Behrouz et al., 2024, 2025a,b), which admit O(1) per-token state updates at inference.

We propose Higher-order Linear Attention (HLA), generalizing linear attention by incorporating higher interactions through compact prefix summaries (sufficient statistics). The key observation is that higher attention-like operators admit factorized forms in terms of low-order moments (e.g., sums of key outer products), enabling exact causal streaming without constructing attention matrices. In the second-order case, HLA maintains an constant-size state per head and produces outputs in linear time per token (O(d2+ddv), here d is the query/key dimension and dv the value dimension, independent to the sequence length).

We address two central challenges: (i) enforcing strict autoregressive causality at second-order without sacrificing streaming updates or introducing any n×n intermediates; and (ii) enabling chunk-parallel training that exactly matches the activations of a serial recurrence. First, we derive an exact masked formulation that enforces strict autoregressive causality by augmenting the state with two additional summaries; the resulting algorithm remains streaming and efficient. Second, we present a chunk-parallel training scheme based on an associative (monoid/semidirect-product) operator that yields the same activations as a serial loop while exploiting intra- and inter-chunk parallelism.

Our contributions are summarized as follows:

- 1. Exact masked streaming at second order. We give a complete algebra of extended summaries that yields strictly causal second-order HLA with per-token constant cost, together with formal statements and proofs establishing masked streaming identities and online updates. The unnormalized HLA is the default operator; the ratio-normalized variant is an option built from the same summaries.
- 2. Associative scans that match serial activations. We define an associative (semidirect-product) operator for unmasked and masked settings (with and without exponential decay) and prove that a standard exclusive scan produces forward activations identical to those of a serial recurrence. We also state the reverse-mode algebra.
- 3. Third-order extension. We present the full masked third-order state and online updates, a strictly causal streaming kernel, and an exact chunk-parallel algorithm. The third-order scan augments the corrected state with segment-level linear maps; its associative composition reproduces the serial recurrence exactly.

HLA is intended as a drop-in, attention-like mixer for long-context models. It provides (i) attention-style, data-dependent weighting; (ii) strictly causal streaming with O(1) per-token update memory independent of sequence length; and (iii) parallel training via scans without resorting to approximate backpropagation through time. We deliberately focus on algorithmic structure and implementation.

### 2 Background

Notations. We use bold lowercase for vectors and bold uppercase for matrices/tensors. Token index t denotes the current time; d is the query/key dimension; dv is the value dimension. Unless otherwise stated, HLA outputs are in the default unnormalized form, which avoids length-dependent renormalization. We adopt row-vector outputs ot ∈ R1×dv; a ratio-normalized (row-normalized) variant divides by a masked scalar denominator built from the same summaries for scale control and comparability with linear attention. Throughout, prefix summaries are statistics computable in streaming fashion with O(1) memory per token and per head.

#### 2.1 Scaled dot-product attention

Given queries Q ∈ Rn×d, keys K ∈ Rn×d, and values V ∈ Rn×dv, scaled dot-product attention (Vaswani et al., 2017) is

Attn(Q,K,V) = softmax

##### QK⊤

√

##### + Λ V,

d

where Λ ∈ Rn×n is the additive causal mask (zeros on and below the diagonal; −∞ above). For algebraic manipulations outside the softmax (e.g., Section 3.1), we use the Hadamard product ⊙ with a binary causal mask, denoted by L (ones on and below the diagonal; zeros above), to mask bilinear forms consistently.

#### 2.2 Linear attention

Linear attentions (Wang et al., 2020; Katharopoulos et al., 2020; Choromanski et al., 2020) approximate the softmax kernel by a feature map ϕ : Rd→Rr (maybe unnormalized):

ϕ(qi)⊤ j ϕ(kj)vj⊤ ϕ(qi)⊤ j ϕ(kj)

Attn(Q,K,V)i ≈

.

Maintaining the running sums j ϕ(kj)vj⊤ and j ϕ(kj) yields O(nr(d+dv)) time and O(r dv) memory complexity.

### 3 Higher-order Linear Attention

In this section, we will introduce Higher-order Linear Attention (HLA). We begin with second-order linear attention as a warm-up, and present its extension to third-order linear attention in Section 7.

Second-order tensor attention mechanism. Second-order tensor attention can be written as

T2 := (QK⊤)(QK⊤)⊤ = Q(K⊤K)Q⊤ ∈ Rn×n, HLA2(Q,K,V) = T2V ∈ Rn×d,

so that [T2]ij = q⊤i (K⊤K)qj. The right-hand side shows a dependence on the second moment K⊤K ∈ Rd×d, suggesting streaming implementations via prefix moments.

We maintain prefix summaries at time t:

SKt :=

CQVt :=

mQt :=

kik⊤i ∈ Rd×d,

i≤t

qivi⊤ ∈ Rd×dv,

i≤t

qi ∈ Rd.

i≤t

All the above prefix summaries can be updated in a streaming fashion. In particular, the updates of SKt and CQVt cost O(d2) and O(ddv) time per token, respectively. Unnormalized HLA. The output of second-order HLA at time t is the numerator-style bilinear form built from prefix moments:

ot := q⊤t SKt CQVt . (3.1) This choice avoids length-dependent renormalization while preserving streaming updates and the same state as the normalized variant.

- A. Recurrent Form (causal HLA scan)
- B. Parallel Form (masked 2nd-order tensor attention)
- C. Chunk-wise Parallel Form (associative scan w/ semidirect product)

###### time →

q1 k1 v1 q2 k2 v2 qt kt vt

read q⊤t (SKt CQVt −Gt)

SK0 ,CQV0 , mQ0 ,G0,h0

SK1 ,CQV1 , mQ1 ,G1,h1

SK2 ,CQV2 , mQ2 ,G2,h2 ···

SKt ,CQVt , mQt ,Gt,ht

ot

write outer products + cross-term ks(k⊤s CQVs−1)

qt

SKt = SKt−1 + ktk⊤t , CQVt = CQVt−1 + qtv⊤t , mQt = mQt−1 + qt Gt = Gt−1 + kt k⊤t CQVt−1 , ht = ht−1 + kt k⊤t mQt−1

O(N(d2+d dv)) time O(d2+d dv) state at inference

ot = q⊤t SKt CQVt − Gt

kik⊤i , CQVt =

kik⊤i CQVi−1

qiv⊤i , Gt =

exact equality for zero-initialised state: SKt =

≡

i≤t

i≤t

i≤t

(q⊤t ki)(q⊤j ki) v⊤j = (L ⊙ QK⊤)(L ⊙ QK⊤)⊤⊙ L t,:V

=⇒ ot =

j≤t i≤j

N×d

N×N (causal)

N×dv

N×dv

N×N

N×N

d×N

= T2=WW⊤

= L⊙QK⊤

= O

Q

× K⊤

× W⊤

× V

⊙ L

=: W

###### O = L ⊙ QK⊤ L ⊙ QK⊤ ⊤ ⊙ L V ≡ ot =

###### (q⊤t ki)(q⊤j ki)v⊤j O(N2(d+dv)) time fully parallel over t

j≤t i≤j

regroup tokens into M chunks of size C, N = MC, associative scan with semidirect product over chunks: (·)A ⊕ (·)B = SKA+SKB, CQVA +CQVB , mQA+mQB,

≡

|SKBCQVA|
|---|

|SKBmQA|
|---|

GA+GB +

, hA+hB +

###### cross-term SKB CQVA (and SKB mQA)

###### Chunk 1

###### Chunk 2

Chunk M

S(2) ···

S(0)

S(1)

S(M)

K(1)⊤

K(2)⊤

K(M)⊤

Q(1)

V(1)

Q(2)

V(2)

Q(M)

V(M)

O(1) O(2) O(M)

S(k) = S(k−1) ⊕ T (k) (state tuple S = (SK,CQV ,mQ,G,h)) O(tk) = q⊤t SK,(k−1)CQV,(k−1) − G(k−1)

O NC(d+dv)+M(d2+d dv) time M = N/C sequential chunk steps

+ (L ⊙ Q(k)K(k)⊤)(L ⊙ Q(k)K(k)⊤)⊤⊙ L t,:V(k)

cross-chunk (carry from S(k−1))

intra-chunk (parallel masked 2nd-order)

###### Figure 1 Three equivalent views of Higher-order Linear Attention (HLA, second order). (A) The recurrent

form maintains a constant-size state tuple St = (SKt ,CQVt ,mQt ,Gt,ht), with two additional cross-summaries Gt,ht that enforce strict causality at second order. (C) The chunk-wise parallel form interpolates between (A) and (B): the sequence is split into M chunks of size C, intra-chunk computation evaluates the masked second-order form in parallel.

Normalized HLA. In order to define the normalized output of HLA, we define the numerator and denominator at t as follows:

numt = q⊤t SKt CQVt , dent = q⊤t SKt mQt , and the normalized output of HLA is given by

numt dent + ε

ot =

q⊤t SKt CQVt q⊤t SKt mQt + ε

=

, (3.2)

where ε > 0 is a small constant added for numerical stability. Notably, SKt acts as a learned, data-dependent metric on query space; CQVt is a value accumulator

modulated by past queries; and mQt provides a query mass for optional scale control. This mirrors a second-order polynomial kernel in (q,k) while remaining strictly streaming and causal once masked (Section 3.1).

##### Connection with linear attention. Setting SKt = I yields

numt = q⊤t CQVt =

(q⊤t qi)vi⊤, dent = q⊤t mQt =

q⊤t qi,

i≤t

i≤t

So the normalized output reduces to a linear-attention form with kernel K(qt,qi) = q⊤t qi. When queries and keys are tied (qi ≡ ki), this coincides with linear attention using the identity feature map ϕ(x) = x. In general, second-order HLA implements the data-adaptive degree-2 polynomial

kernel Kt(q,q′) = q⊤SKt q′ whose metric SKt = i≤t kik⊤i depends on the past keys, strictly enriching first-order linearizations while retaining streaming. Absent tying q ≡ k, this differs from

identity-feature linear attention.

#### 3.1 Causal masking via extended summaries

Let L denote the binary causal mask (lower-triangular, including the diagonal). For the masked second-order matrix,

(L ⊙ QK⊤)(L ⊙ QK⊤)⊤ t,j =

(q⊤t ki)(q⊤j ki) = q⊤t SKmin(t,j)qj.

i≤min(t,j)

Equivalently, the strictly causal second-order output at time t can be written in matrix form by masking on the right before applying values:

ot = (L ⊙ QK⊤)(L ⊙ QK⊤)⊤ ⊙ L

##### V.

t,:

This row-wise ⊙L enforces the restriction j ≤ t when multiplying by V. Define two additional prefix summaries

kik⊤i CQVi−1 ∈ Rd×dv,

Gt :=

i≤t

kik⊤i mQi−1 ∈ Rd.

ht :=

i≤t

We have the following theorem, which gives the unnormalized and normalized outputs of HLA with a causal mask.

- Theorem 3.1 (Masked streaming identity for second order). For each t, let

nummaskt = q⊤t SKt CQVt − Gt , denmaskt = q⊤t SKt mQt − ht . Consequently, the strictly causal, masked default unnormalized output is

ot = q⊤t SKt CQVt − Gt . (3.3) An optional linear normalization divides by the masked denominator,

q⊤t SKt CQVt − Gt q⊤t SKt mQt − ht + ε

ot =

, (3.4)

where ε > 0 is a small constant added for numerical stability.

Proof. Let W = L ⊙ (QK⊤) with L lower-triangular including the diagonal. For the second-order weight matrix, we have WW⊤ with entries (WW⊤)t,j = i≤min(t,j)(q⊤t ki)(q⊤j ki). Then the masked, unnormalized numerator at time t is

nummaskt =

(WW⊤)t,j vj⊤ =

j≤t

q⊤t ki k⊤i qj vj⊤ = q⊤t

kik⊤i qjvj⊤,

j≤t i≤j

j≤t i≤j

where the second equality uses the fact that min(t,j) = j when j ≤ t. Interchanging finite sums yields

kik⊤i qjvj⊤ =

j≤t i≤j

SKj qjvj⊤ =

j≤t

j≤t

I1

SKt qjvj⊤

−

##### kik⊤i qjvj⊤

, (3.5)

j≤t j<i≤t

I2

where the last equality holds due to SKj = SKt − j<i≤t kik⊤i .

In Eq. (3.5), the first term I1 equals SKt CQVt . For the second term I2, swap the order of summation: j≤t i>j(·) = i≤t j<i(·), we can obtain I2 = i≤t kik⊤i j<i qjvj⊤ = Gt. This proves the numerator identity. The proof for the denominator is analogous with vj replaced by 1 (i.e., qj replaced by 1-summaries), yielding SKt mQt − ht. Finally, the division by denmaskt + ε gives Eq. (3.4).

| |
|---|

Online updates. Using the fact that (kk⊤)X = k(k⊤X), we have

SKt = SKt−1 + ktk⊤t , CQVt = CQVt−1 + qtvt⊤, mQt = mQt−1 + qt, Gt = Gt−1 + kt(k⊤t CQVt−1), ht = ht−1 + kt(k⊤t mQt−1).

Therefore, the per-token cost remains O(d2+ddv) in total.

### 4 Chunk-parallel training via associative scans

In Section 3.1, we have presented the recurrent form for second-order HLA. As we know, training a purely recurrent model is inefficient on GPUs. We adopt within-chunk scans with width w and inter-chunk scans across B chunks (Blelloch, 1990). A similar technique has been widely used in the literature of linear attention (Yang et al., 2023; Qin et al., 2024). We write Bc for the number of chunks to avoid overloading B elsewhere; thus, inter-chunk scans are across Bc chunks.

#### 4.1 Unmasked monoid

Let S = (S,C,m) with token “deltas” ∆St = ktk⊤t , ∆Ct = qtvt⊤, ∆mt = qt. Define elementary segments Tt = (∆St,∆Ct,∆mt) and the additive monoid

(SA,CA,mA) ⊕ (SB,CB,mB) = (SA+SB, CA+CB, mA+mB).

An exclusive Blelloch scan on {T1,...,Tw} yields per-token prefixes Pt = i<t Ti, from which the inclusive state at t is obtained locally by adding Tt. Here and below, A then B denotes adjacent segments in time (all indices in A precede those in B).

#### 4.2 Masked semidirect product

For the masked case use S = (S,C,m,G,h). For a single-token segment, G = h = 0. Concatenation is

(SA,CA,mA,GA,hA) ⊕ (SB,CB,mB,GB,hB) =

(4.1)

SA+SB, CA+CB, mA+mB, GA+GB + SBCA, hA+hB + SBmA ,

which is associative (direct expansion). Perform the same exclusive scan; per-token inclusive states follow by adding the local deltas and the cross-terms ∆StCt−1 and ∆Stmt−1.

Decay-aware monoid. Let γ ∈ (0,1) be a fixed exponential decay and let a segment X carry its length ℓ(X) and attenuation ρ(X) := γℓ(X). For the unmasked triple S = (S,C,m) the decayed concatenation is

(SA,CA,mA,ρA) ⊕γ (SB,CB,mB,ρB) = ρBSA+SB, ρBCA+CB, ρBmA+mB, ρAρB , and analogously for the masked (S,C,m,G,h) state:

(SA,CA,mA,GA,hA,ρA) ⊕γ (SB,CB,mB,GB,hB,ρB) = ρBSA+SB, ρBCA+CB, ρBmA+mB,

ρBGA+GB + SB(ρBCA), ρBhA+hB + SB(ρBmA), ρAρB . Associativity follows from bilinearity and ρ-multiplicativity.

- Theorem 4.1 (Scan equivalence: serial vs. (decayed) associative scans). Consider a sequence of

token segments {T1,...,Tn} and either ⊕ (no decay) or ⊕γ (with decay). Let Pt be the exclusive prefix obtained by a Blelloch scan under the chosen operator. For each t, the inclusive state computed locally from Pt and Tt equals the state produced by a serial left-to-right recurrence on tokens 1:t. Consequently, the per-token masked outputs are identical to those of the serial algorithm.

Proof. We prove for the masked, decayed case; the other cases are specializations. Define the serial recurrence Xt = Φγ(Xt−1,Tt) given by the online updates in Section 3.1 with decay γ. By construction, Φγ coincides with the binary map fγ(X,Y) := X ⊕γ Y when Y is a single-token segment. Because ⊕γ is associative with identity the zero-length segment E (all-zero summaries, ρ = 1), the Blelloch scan yields Pt = E ⊕γ T1 ⊕γ ··· ⊕γ Tt−1. The local inclusive update computes Pt⊕γ Tt, which equals Xt by associativity and the definition of Φγ. The masked outputs are functions only of the inclusive state (Theorem 3.1), hence coincide with the serial outputs.

| |
|---|

Backward for gradients. Let ⊕∗ denote the vector-Jacobian adjoint of ⊕ evaluated at the forward states. A reverse (decayed) scan applying ⊕∗γ with checkpointing at tile boundaries yields gradients that match those of the serial recurrence, by Theorem 4.1 and the chain rule.

Remark 4.2 (Inclusive vs. exclusive scans). Given segments (T1,...,Tw) and an associative operator ⊕ with identity E, the exclusive scan returns prefixes Pt = E ⊕ T1 ⊕ ··· ⊕ Tt−1, while the inclusive scan returns It = Pt ⊕ Tt. Our forward algorithms compute Pt via an exclusive Blelloch scan and then form the inclusive state locally by combining Pt with the token’s deltas (and required cross-terms). This choice exposes maximal parallelism and ensures exact equality to the serial recurrence by Theorem 4.1. With decay, the identity is the zero-length segment (0,...,0,ρ=1); the exclusive/inclusive distinction is unchanged.

Intra-chunk parallelism. Within a chunk of width w, an exclusive Blelloch scan over {T1,...,Tw} under ⊕ (or ⊕γ) yields Pt for all t in O(log w) span and O(1) auxiliary memory per position. The per-token inclusive states are then computed independently as It = Pt ⊕ Tt.

Inter-chunk parallelism. For Bc chunks, each chunk c produces a single summary S(c) =

t∈chunk c Tt. An exclusive scan across the Bc summaries gives carry-in prefixes P(c) for every chunk. Each position t in chunk c then uses the merged prefix P(c) ⊕ Pt before adding its local Tt to obtain the inclusive state. This is the same parallel skeleton widely used in modern linear-attention and recurrent networks that maintain streaming sufficient statistics (Sun et al., 2023; Qin et al., 2023; Yang et al., 2023; Qin et al., 2024; Yang et al., 2024b).

Connection to linear attention. First-order linear attentions and related modern RNN kernels scan additive/decayed summaries (e.g., ϕ(k)v⊤ and denominators) using exactly this intra-/interchunk pattern. HLA plugs into the same infrastructure: only the state tuple and cross-terms change (e.g., (S,C,m,G,h) for second order), while the exclusive/inclusive logic and two-level scan strategy remain identical. Thus, HLA inherits the throughput characteristics of these systems with strictly higher expressivity.

- 4.3 Adding decay and regularization Decayed states. Introduce a time decay γ ∈ (0,1):

SKt = γSKt−1 + ktk⊤t , CQVt = γCQVt−1 + qtvt⊤, mQt = γmQt−1 + qt, and the cross-summaries obey

Gt = γGt−1 + kt k⊤t CQVt−1 , ht = γht−1 + kt k⊤t mQt−1 , which are the decayed analogues of the online updates in Section 3.1. Decay controls spectral growth and improves recency bias while maintaining associativity (with respect to segment-local normalization) (Peng et al., 2023; Sun et al., 2023; Qin et al., 2023; Yang et al., 2023, 2024a; Peng et al., 2024; Behrouz et al., 2024; Peng et al., 2025; Behrouz et al., 2025a,b).

### 5 Implementation details and complexity

In this section, we discuss the implementation details and provide a complexity analysis. Recall that for each token and each head (second order), we have

- • State: SKt ∈ Rd×d, CQVt ∈ Rd×dv, mQt ∈ Rd (and masked Gt ∈ Rd×dv, ht ∈ Rd).
- • Compute: evaluate ut = q⊤t SKt (mat–vec) and then utCQVt (row–matrix), with masked corrections −q⊤t Gt; the denominator uses utmQt − q⊤t ht. This avoids forming SKt CQVt explicitly; masked cross-terms still use k⊤t X to avoid cubic cost.
- • Parallelism: within-chunk Blelloch scans (span O(log w)) and inter-chunk exclusive scans across Bc chunks, both using the same ⊕.

- Algorithm 1 Masked (Second Order) HLA with Within-Chunk Scan Require: Chunk tokens (q[1:w],k[1:w],v[1:w]), ε, optional ridge λ, optional decay γ, optional flag normalize

- 1: Token segments: for t = 1..w, set ∆St ← ktk⊤t , ∆Ct ← qtvt⊤, ∆mt ← qt, and initialize Gt=0, ht=0.
- 2: Exclusive scan over {(∆St,∆Ct,∆mt,0,0)}wt=1 using ⊕ in Eq. (4.1) (with decay if used) to obtain prefixes Pt = (St−1,Ct−1,mt−1,Gt−1,ht−1).
- 3: for t = 1 to w in parallel do
- 4: Inclusive state:
- 5: St ← γSt−1 + ∆St; Ct ← γCt−1 + ∆Ct; mt ← γmt−1 + ∆mt
- 6: Gt ← γGt−1 + ∆St Ct−1; ht ← γht−1 + ∆St mt−1
- 7: Effective S: Sefft ← St + λI ▷ optional ridge for stability
- 8: Default masked unnormalized output:
- 9: u ← q⊤t Sefft ▷ O(d2) matvec
- 10: num ← uCt − q⊤t Gt ▷ O(ddv)
- 11: ohlat ← num
- 12: Optional normalization:
- 13: if normalize then
- 14: den ← umt − q⊤t ht + ε
- 15: ohlat ← ohlat /den
- 16: end if
- 17: end for
- 18: return {ohlat }wt=1

#### 5.1 Pseudocode

We present a PyTorch-like reference for masked second-order HLA with a within-chunk exclusive scan. Unmasked and/or diagonal-regularized variants follow by removing (G,h). Normalization is optional; by default, the implementation returns the unnormalized output and may divide by the masked denominator if requested.

Remark. Adding λI yields a stabilized causal variant of the masked operator; it does not correspond to the exact masked bilinear form of (L ⊙ QK⊤).

#### 5.2 Implementation considerations

HLA only replaces the standard attention sublayer in the transformer block, while the feed-forward sublayer and normalization sublayers remain unchanged. Drop-in replacement requires only swapping the kernel while keeping positional encodings and masking identical to the baseline. Multi-query

keys/values (sharing K,V across heads) reduce state from O(hd2) to O(d2+hddv) while preserving the algebra.

The summaries (S,C,m,G,h) are per head. With multi-query (K,V shared across heads),

the key moment SKt is shared and stored once per layer (O(d2)), while (CQVt ,mQt ,Gt,ht) remain per-head (O(hddv+hd)). This yields a total memory of O(d2+hddv) instead of O(hd2+hddv) when each head maintains its own SKt .

For throughput, maintain SKt in a packed symmetric layout (store only the upper triangle,

- 1

- 2d(d+1) entries) to reduce bandwidth without changing the algebra. Within a chunk of width w, use an exclusive Blelloch scan to obtain prefixes in O(log w) span and constant extra memory per position; inter-chunk scans use the same operator across Bc chunks.

### 6 Asymmetric Higher-order Linear Attention

Motivation. The second-order HLA in Section 3 realizes the symmetric triple product AA⊤V with A = QK⊤ (masked later). We introduce a complementary asymmetric variant that uses the left-cascaded product

##### AHLA(Q,K,V) := AAV = Q(K⊤Q)(K⊤V),

and show it admits strictly causal streaming with O(d2+ddv) per-token cost. We call this operator AHLA (Asymmetric Higher-order Linear Attention).

#### 6.1 Definition and masked streaming identity

Let A = L ⊙ (QK⊤) be the causally masked affinity, where L is the binary lower-triangular mask (including the diagonal). The AAV weights are

(AA)t,j =

t

(q⊤t ki)(q⊤i kj), j ≤ t.

i=j

To obtain strictly causal outputs when applying to values, interpret the final multiplication as (AA) ⊙ L V; the streaming identity in Theorem 6.1 implements exactly this row-wise masking.

Consequently, the (unnormalized) output is

oAHLAt =

j≤t

Introduce the streaming prefix summaries

t

(q⊤t ki)(q⊤i kj)vj⊤. (6.1)

i=j

kjvj⊤ ∈ Rd×dv, mKt :=

PKVt :=

kj ∈ Rd,

j≤t

j≤t

ki q⊤i PKVi ∈ Rd×dv,

Et :=

i≤t

ki q⊤i mKi ∈ Rd.

nt :=

i≤t

Note. For chunk-parallel scans used in training, we additionally introduce a segment-level cross moment RKQ; see Section 6.2 for its definition and role in the concatenation operator.

- A. Recurrent Form (causal AHLA streaming)
- B. Parallel Form (masked asymmetric AAV)
- C. Chunk-wise Parallel Form (associative scan w/ semidirect product)

###### time →

q1 k1 v1 q2 k2 v2 qt kt vt

read q⊤t Et

PKV0 ,mK0 , E0,n0

PKV1 ,mK1 , E1,n1

PKV2 ,mK2 , E2,n2 ···

PKVt ,mKt , Et,nt

ot

write outer products + routed term kt(q⊤t PKVt )

qt

PKVt = PKVt−1 + ktv⊤t , mKt = mKt−1 + kt

O(N d dv) time O(d dv) state at inference

Et = Et−1 + kt q⊤t PKVt , nt = nt−1 + kt q⊤t mKt ot = q⊤t Et (optional norm: divide by q⊤t nt + ε)

kjv⊤j , Et =

ki q⊤i PKVi

exact equality for zero-initialised state: PKVt =

≡

j≤t

i≤t

t

=⇒ ot = q⊤t Et =

(q⊤t ki)(q⊤i kj)v⊤j = (AA) ⊙ L t,:V

j≤t

i=j

N×d

N×N (causal)

N×dv

N×dv

N×N

N×N

d×N

= L⊙QK⊤

= AA⊙L

= O

Q

× K⊤

× A

× V

=: A

t

###### (q⊤t ki)(q⊤i kj)v⊤j Ofully(N2parallel(d+dv))overtimet

###### O = L ⊙ QK⊤ L ⊙ QK⊤ ⊙ L V ≡ ot =

j≤t

i=j

regroup tokens into M chunks of size C, N = MC, associative scan with semidirect product over chunks (adds segment-level key–query moment RKQ = i kiq⊤i ): (·)A ⊕AHLA (·)B = RKQA +RKQB , PKVA +PKVB , mKA+mKB,

≡

|RKQB PKVA|
|---|

|RKQB mKA|
|---|

EA+EB +

, nA+nB +

###### cross-term RKQB PKVA (and RKQB mKA )

###### Chunk 1

###### Chunk 2

Chunk M

S(2) ···

S(0)

S(1)

S(M)

K(1)⊤

K(2)⊤

K(M)⊤

Q(1)

V(1)

Q(2)

V(2)

Q(M)

V(M)

O(1) O(2) O(M)

S(k) = S(k−1) ⊕AHLA T (k) state tuple S = (RKQ,PKV ,mK,E,n) O(tk) = q⊤t E(k−1) + RKQ,<t (k) PKV,(k−1)

O NC(d+dv)+M(d2+d dv) time M = N/C sequential chunk steps

+ (A(k)A(k)) ⊙ L t,:V(k)

cross-chunk (carry from S(k−1))

intra-chunk (parallel masked AAV)

###### Figure 2 Three equivalent views of Asymmetric Higher-order Linear Attention (AHLA). (A) The recurrent

form maintains a constant-size state tuple St = (PKVt ,mKt ,Et,nt). (B) The parallel form materializes the asymmetric causal weight (AA)⊙L with A = L⊙QK⊤ and applies it to V. (C) The chunk-wise parallel form interpolates between (A) and (B) via an associative scan over the augmented tuple S = (RKQ,PKV ,mK,E,n).

- Theorem 6.1 (Masked streaming identity for AHLA). With the above definitions,

oAHLAt = q⊤t Et and oAHLAt =

q⊤t Et q⊤t nt + ε

,

where the second expression is an optional linear normalization using the masked denominator. The online (strictly causal) updates are

PKVt = PKVt−1 + ktvt⊤, mKt = mKt−1 + kt, Et = Et−1 + kt q⊤t PKVt , nt = nt−1 + kt q⊤t mKt .

Proof. From Eq. (6.1), fix i and sum over j ≤ i: j≤i(q⊤i kj)vj⊤ = q⊤i PKVi . Then oAHLAt =

i≤t(q⊤t ki) q⊤i PKVi = q⊤t i≤t ki(q⊤i PKVi ) = q⊤t Et. Replacing vj by 1 gives the denominator with mKi , hence nt. The stated updates follow by isolating index i=t and using that PKVt = PKVt−1 + ktvt⊤ and mKt = mKt−1 + kt.

| |
|---|

Cost. For streaming/serial inference, the dominant work is forming q⊤t PKVt ∈ R1×dv and the outer product kt(·); the total is O(ddv) time and O(ddv+d) state per head (for P,E,m,n). For chunk-parallel scans used in training, an additional block statistic RKQ appears only inside the concatenation operator (Section 6.2), contributing O(d2) memory per chunk summary but not to the streaming path.

Decay mechanism. With exponential decay γ ∈ (0,1), PKVt = γPKVt−1 + ktvt⊤, mKt = γmKt−1 + kt,

Et = γEt−1 + kt q⊤t PKVt , nt = γnt−1 + kt q⊤t mKt , which preserves associativity of the scan operator below.

- 6.2 Chunk-parallel scans for AHLA Unmasked/masked concatenation. For segment A followed by B, consider the augmented state

S = (RKQ,PKV ,mK,E,n). Here RKQ is the segment-level key–query cross moment, defined by

RKQ :=

kiq⊤i ∈ Rd×d.

i∈segment

It is used only during chunk concatenation to form the cross terms RBPA and RBmA in Eq. (6.2); It is not required by the serial/streaming forward path in Algorithm 2. With exponential decay, the segment’s RKQ attenuates as ρBRA+RB in the decayed concatenation.

The undecayed associative concatenation is (RA,PA,mA,EA,nA) ⊕AHLA (RB,PB,mB,EB,nB) =

RA+RB, PA+PB, mA+mB, EA+EB+RBPA, nA+nB+RBmA , (6.2)

which is associative by direct expansion of i∈A∪B ki(q⊤i P≤i) and the observation that for i ∈ B the missing cross-prefix equals RBPA (and analogously for n).

Decay-aware concatenation. Let each segment carry its attenuation ρ(·) = γℓ(·). Then

(R,P,m,E,n,ρ) = (ρBRA+RB, ρBPA+PB, ρBmA+mB,

ρBEA+EB+RB(ρBPA), ρBnA+nB+RB(ρBmA), ρAρB), which is associative by bilinearity and multiplicativity of ρ. Scan equivalence. An exclusive Blelloch scan under ⊕AHLA (or its decayed form) followed by local inclusion reproduces exactly the activations of the serial recurrence given above.

#### 6.3 Pseudocode

Algorithm 2 AHLA (Second-order) streaming with causal mask and optional decay

Require: {qt,kt,vt}nt=1, decay γ ∈ (0,1], stability ε > 0, flag normalize

- 1: Init: P=0d×d

v

, m=0d, E=0d×d

v

, n=0d

- 2: for t = 1 to n do
- 3: P ← γP + ktvt⊤; m ← γm + kt
- 4: r ← q⊤t P ▷ 1×dv
- 5: s ← q⊤t m ▷ scalar
- 6: E ← γE + kt r; n ← γn + skt
- 7: ot ← q⊤t E
- 8: if normalize then
- 9: den ← q⊤t n + ε; ot ← ot/den
- 10: end if
- 11: end for
- 12: return {ot}nt=1

Relation to AA⊤V. AHLA emphasizes a matrix power of A, weighting each value vj through a single pass q⊤i kj routed by an intermediate key index i. In contrast, the symmetric AA⊤V aggregates via the metric SK and query summaries. Both are second-order, strictly causal, and stream with identical asymptotic costs but induce different inductive biases.

7 Third-Order Linear Attention

In this section, we will introduce third-order HLA.

#### 7.1 Streaming form of Causal HLA

Third-order tensor attention mechanism. Let A = QK⊤ ∈ Rn×n and L be the binary causal mask. Unmasked third-order tensor attention uses the matrix AA⊤A. Its (t,j)-entry is

 (q⊤u kj) = q⊤t (K⊤K)

 

quq⊤u kj,

(q⊤t ki)(q⊤u ki)

[(AA⊤A)]t,j =

u

u≤n

i≤n

which immediately yields a streaming factorization through prefix moments. Then the third-order HLA is defined as

HLA3(Q,K,V) = [(AA⊤A)]V.

Unmasked factorization. Define prefix summaries SKt = i≤t kik⊤i ∈ Rd×d, SQt = i≤t qiq⊤i ∈ Rd×d, PKVt = i≤t kivi⊤ ∈ Rd×dv, mKt = i≤t ki ∈ Rd. The default (unnormalized) third-order operator is

o(3)t = q⊤t SKt SQt PKVt . An optional normalization divides by q⊤t SKt SQt mKt +ε if desired. Masked streaming summaries. To impose strict causality, we introduce cross-summaries:

- G(1)t := i≤t

(kik⊤i )SQi−1PKVi−1 ∈ Rd×dv, h(1)t :=

i≤t

(kik⊤i )SQi−1mKi−1 ∈ Rd,

- G(2)t := i≤t

SKi−1(qiq⊤i )PKVi−1 ∈ Rd×dv, h(2)t :=

i≤t

SKi−1(qiq⊤i )mKi−1 ∈ Rd,

- G(3)t := i≤t

SKi−1SQi−1(kivi⊤) ∈ Rd×dv, h(3)t :=

i≤t

###### Then the masked, unnormalized quantities are defined as follows:

SKi−1SQi−1ki ∈ Rd.

num(3)maskt = q⊤t SKt SQt PKVt − G(1)t − G(2)t − G(3)t ,

den(3)maskt = q⊤t SKt SQt mKt − h(1)t − h(2)t − h(3)t . The following theorem shows that the (normalized) output of third-order HLA can be computed based on num(3)maskt and den(3)maskt .

- Theorem 7.1 (Masked streaming identity for third order). For each t, the strictly causal third-order output in the default (unnormalized) form is

o(3)t = num(3)maskt .

An optional normalized variant divides by the masked denominator,

and the online updates are

num(3)maskt den(3)maskt + ε

o(3)t =

.

SKt = SKt−1 + ktk⊤t , SQt = SQt−1 + qtq⊤t , PKVt = PKVt−1 + ktvt⊤, mKt = mKt−1 + kt. (7.1)

- G(1)t = G(1)t−1 + (ktk⊤t )SQt−1PKVt−1,
- G(2)t = G(2)t−1 + SKt−1(qtq⊤t )PKVt−1,
- G(3)t = G(3)t−1 + SKt−1SQt−1(ktvt⊤). (7.2)

- h(1)t = h(1)t−1 + (ktk⊤t )SQt−1mKt−1,
- h(2)t = h(2)t−1 + SKt−1(qtq⊤t )mKt−1,
- h(3)t = h(3)t−1 + SKt−1SQt−1kt. (7.3)

Proof. Let W = L ⊙ (QK⊤) and consider (WW⊤W)V. The t-th row applied to V is

 kjvj⊤.

 

 vj⊤ = q⊤t

 

SKu quq⊤u

(WW⊤)t,u Wu,j

j≤t

u≤t

j≤t

u≤t

Using u≤t SKu = SKt + u≤t−1(SKu ) and repeatedly applying i≤u = i≤t − u<i≤t to peel off the dependence on future indices relative to each summation boundary yields

SKt SQt kjvj⊤ −

j≤t

(kik⊤i )SQi−1PKVi−1 −

i≤t

SKi−1(qiq⊤i )PKVi−1 −

i≤t

SKi−1SQi−1(kivi⊤),

i≤t

which is precisely SKt SQt PKVt − G(1)t − G(2)t − G(3)t . Left-multiplication by q⊤t gives the masked numerator, and replacing vj by 1 yields the denominator. Online updates follow by isolating the i = t contributions and using (kk⊤)X = k(k⊤X).

| |
|---|

#### 7.2 Pseudocode

We present explicit pseudocode for masked third-order HLA in two parts: (i) a strictly causal streaming kernel for inference, and (ii) the associative scan operator used for chunk-parallel training. All operations are per head; shapes follow Section 7.1.

- Algorithm 3 Masked (Third Order) HLA Streaming Kernel Require: Sequences {qt,kt,vt}nt=1, decay γ ∈ (0,1], stability ε > 0, flag normalize

- 1: Init: SK =0d×d, SQ=0d×d, PKV =0d×d

v

, mK =0d

- 2: G(1)=0d×d

v

, G(2)=0d×d

v

, G(3)=0d×d

v

, h(1)=0d, h(2)=0d, h(3)=0d

- 3: for t = 1 to n do
- 4: SKprev←SK; SQprev←SQ; Pprev←PKV ; mprev←mK
- 5: Inclusive first-order updates (with decay):
- 6: SK ← γSKprev + ktk⊤t ; SQ ← γSQprev + qtq⊤t
- 7: PKV ← γPprev + ktvt⊤; mK ← γmprev + kt
- 8: Cross-summaries (matvec/outer-product forms):
- 9: u1 ← SQprev kt; G(1) ← γG(1) + kt u⊤1 Pprev ; h(1) ← γh(1) + kt u⊤1 mprev
- 10: a2 ← SKprev qt; G(2) ← γG(2) + a2 q⊤t Pprev ; h(2) ← γh(2) + a2 q⊤t mprev
- 11: u3 ← SQprev kt; a3 ← SKprev u3; G(3) ← γG(3) + a3vt⊤; h(3) ← γh(3) + a3
- 12: Output (unnormalized by default):
- 13: y ← SK qt; z ← SQ y; termA ← z⊤PKV ; termB ← q⊤t G(1); termC ← q⊤t G(2); termD ← q⊤t G(3)
- 14: ot ← termA − termB − termC − termD
- 15: if normalize then
- 16: denvec ← SK (SQmK) − h(1) − h(2) − h(3)
- 17: den ← q⊤t denvec + ε; ot ← ot/den
- 18: end if
- 19: end for
- 20: return {ot}nt=1

#### 7.3 Chunk-parallel algorithm for third-order HLA

For chunk-parallel training it is convenient to scan a corrected third-order state rather than the three raw correction tensors separately. Define

Ft := SKt SQt PKVt − G(1)t − G(2)t − G(3)t ∈ Rd×dv, ηt := SKt SQt mKt − h(1)t − h(2)t − h(3)t ∈ Rd. (7.4)

Then the masked numerator and denominator are simply

num(3)maskt = q⊤t Ft, den(3)maskt = q⊤t ηt. Let the token-level increments be

DKt = ktk⊤t , DQt = qtq⊤t , DPt = ktvt⊤, dmt = kt.

From Eq. (7.4) and the online updates in Theorem 7.1, the corrected state obeys the recurrence Ft = Ft−1 + SKt−1DQt DPt + DKt SQt−1DPt + DKt DQt PKVt−1 + DKt DQt DPt , ηt = ηt−1 + SKt−1DQt dmt + DKt SQt−1dmt + DKt DQt mKt−1 + DKt DQt dmt . (7.5)

For a contiguous segment X, define its scan state

X = SKX,SQX,PKVX ,mKX,FX,ηX,RQPX ,rQmX ,UKQX ,MKQPX ,MKQmX , where the additional segment summaries are

DQt DPt ∈ Rd×dv, rQmX :=

DQt dmt ∈ Rd,

RQPX :=

t∈X

t∈X

DKt DQt ∈ Rd×d,

UKQX :=

t∈X

and MKQPX and MKQmX are linear maps acting on a matrix Z ∈ Rd×d: MKQPX [Z] :=

DKt ZDPt ∈ Rd×dv, MKQmX [Z] :=

DKt Zdmt ∈ Rd.

t∈X

t∈X

These maps are the only additional objects required at third order: they account for cross terms in which a whole previous segment contributes the middle query moment SQ.

Associative third-order concatenation. For adjacent segments A followed by B, define XAB = XA ⊗3 XB by

SKAB = SKA + SKB, SQAB = SQA + SQB, PKVAB = PKVA + PKVB , mKAB = mKA + mKB,

RQPAB = RQPA + RQPB , rQmAB = rQmA + rQmB , UKQAB = UKQA + UKQB , MKQPAB = MKQPA + MKQPB , MKQmAB = MKQmA + MKQmB , (7.6) and

FAB = FA + FB + SKA RQPB + MKQPB [SQA] + UKQB PKVA , ηAB = ηA + ηB + SKA rQmB + MKQmB [SQA] + UKQB mKA . (7.7)

The identity element is the all-zero segment, with both linear maps equal to the zero map.

Theorem 7.2 (Third-order chunk-scan equivalence). The operator ⊗3 in Eqs. (7.6)–(7.7) is associative. An exclusive scan under ⊗3, followed by local inclusion of the current token segment, produces the same corrected states (Ft,ηt) as the serial recurrence in Eq. (7.5). Consequently, the resulting third-order HLA outputs match Algorithm 3 for γ = 1.

Proof. The additive summaries in Eq. (7.6) are immediate. For the corrected numerator, run the recurrence in Eq. (7.5) on segment B with a carry-in state from segment A. The terms depending only on local prefixes in B give FB. The three carry-dependent terms are

DQt DPt ,

DKt SQADPt ,

DKt DQt PKVA ,

SKA

t∈B

t∈B

t∈B

which are exactly SKA RQPB , MKQPB [SQA], and UKQB PKVA . The denominator proof is identical with DPt replaced by dmt . Thus XA ⊗3 XB is precisely the summary of the concatenated segment AB. Since concatenation of contiguous segments is associative, ⊗3 is associative. The scan statement then follows from the standard exclusive-scan argument used in Theorem 4.1.

- Algorithm 4 Chunk-Parallel Masked (Third Order) HLA via Associative Scan Require: Sequence split into chunks; tokens (qt,kt,vt); stability ε; flag normalize

- 1: Token segments: for every token t, form DKt = ktk⊤t , DQt = qtq⊤t , DPt = ktvt⊤, dmt = kt.
- 2: Initialize the single-token segment Tt by

SK = DKt , SQ = DQt , PKV = DPt , mK = dmt , F = DKt DQt DPt , η = DKt DQt dmt , RQP = DQt DPt , rQm = DQt dmt , UKQ = DKt DQt , MKQP[Z] = DKt ZDPt , MKQm[Z] = DKt Zdmt .

- 3: Within each chunk: run an exclusive Blelloch scan over the token segments using ⊗3 to obtain local prefixes Ptloc and the chunk summary C(b).
- 4: Across chunks: run an exclusive Blelloch scan over {C(b)} using ⊗3 to obtain carry-in summaries P(b).
- 5: for each token t in chunk b in parallel do
- 6: It ← P(b) ⊗3 Ptloc ⊗3 Tt ▷ inclusive corrected state
- 7: ot ← q⊤t F(It)
- 8: if normalize then
- 9: den ← q⊤t η(It) + ε; ot ← ot/den
- 10: end if
- 11: end for
- 12: return {ot}nt=1

Complexity of the third-order scan state. The streaming kernel in Algorithm 3 uses only the compact corrected state in Eq. (7.4) together with the prefix moments. The exact chunk-parallel scan additionally stores the segment maps MKQP and MKQm. If materialized densely, these maps require O(d3dv) and O(d3) entries per segment summary, respectively; equivalently, they may be applied by tensor contractions. This cost is independent of sequence length and is the price of exact third-order chunk composition. The algorithm above is stated for γ = 1; exponential decay is incorporated by adjoining the usual segment attenuation ρ = γℓ and applying the same carry-scaling convention as in the second-order decayed scan.

### 8 Related Work

The literature on subquadratic sequence modeling spans (i) fast-weight style dynamic-parameter models, (ii) kernel/feature-map linearizations of attention, and (iii) recurrent/state-space approaches. HLA belongs to a complementary class: it preserves attention-style, data-dependent mixing but realizes higher interactions through compact prefix moments with exact causal masking and scanparallel training.

Fast weights and fast weight programmers (FWPs). Fast weights, dating to early connectionist memory models (Hinton and Plaut, 1987), implement short-term, input-dependent synaptic changes. Schmidhuber’s fast weight programmers (Schmidhuber, 1992) introduced differentiable controllers that program a separate fast-weight matrix; later, Ba et al. (2016) revived this idea to attend to the recent past. A series of works made the connection to modern attention explicit: Schlag et al. (2021) showed a formal equivalence between linearized self-attention and FWPs, where outer-product updates ∆Wt ∝ ktvt⊤ accumulate an associative memory queried by qt; Irie et al. (2021) extended FWPs with recurrence in the programmer and the fast net. Yang et al. (2024b) utilizes WY Transformations (Bischof and Van Loan, 1987) to implement chunkwise parallel training of Delta Network (Schlag et al., 2021). Parallel lines explore higher or preconditioned mixing by maintaining or inverting second-moment matrices. In our formulation, SKt plays the role of a learned kernel; working directly with SKt avoids explicit matrix inversion and preserves streaming updates, whereas inverse-based methods typically require heavier linear algebra (Behrouz et al., 2025a,b; von Oswald et al., 2025).

Linear Attention Mechanisms. A common route is to replace the softmax kernel by explicit features ϕ to enable streaming via running sums. Representative examples include Linear Transformers (Katharopoulos et al., 2020), Performer’s FAVOR+ random features (Choromanski et al., 2020), and Random Feature Attention (Peng et al., 2021). Earlier work also proposed multiplicative rearrangements that yield linear-complexity-efficient attention (Shen et al., 2021). These methods achieve O(ndr) time with r feature dimension but are typically first-order in the sense that they maintain only ϕ(k)v⊤ and (optionally) a scalar denominator. By contrast, second-order HLA maintains the full key moment SKt = i≤t kik⊤i together with query-value and query mass summaries and their masked cross-summaries, yielding strictly causal higher interactions while remaining streaming. Recent linear attention variants include Sun et al. (2023); Qin et al. (2023); Yang et al. (2023); Qin et al. (2024); Yang et al. (2024b); von Oswald et al. (2025).

State Space Models. SSMs (e.g., S4) (Gu et al., 2021) and selective SSMs (e.g., Mamba) (Gu and Dao, 2023; Dao and Gu, 2024) realize O(1) per-token state updates via linear recurrences and convolutions. These architectures excel at long-range dependencies but express data-dependent mixing differently from attention. HLA sits in between: it is attention-like (data-dependent queries/keys) yet streams via compact prefix statistics like recurrent models.

Modern RNNs. Recent modern RNN designs emphasize gating, decays, and associative scan–friendly recurrences that enable parallel training while preserving strictly constant per-token state at inference. Examples include gated linear mixers and decay-aware updates (Yang et al., 2023, 2024a), efficient gradient routing and training strategies for long sequences (Qin et al., 2024; Peng et al., 2024; Sun et al., 2024), and RWKV-style architectures that replace attention with learned decays and elementwise gating (Peng et al., 2025). These methods typically maintain first-order sufficient statistics and rely on fixed linear dynamics. In contrast, HLA retains attention-style, data-dependent

metrics via SKt and higher cross-summaries while keeping the same O(1) per-token state update paradigm, offering a complementary inductive bias to RNNs and SSMs.

Test Time Training and Memory Networks. Test-time adaptation and explicit long-term memory are emerging tools for extending context without quadratic cost. Test-time training variants adapt parameters from recent tokens to improve local coherence (Sun et al., 2024), whereas memory networks maintain external stores addressable by content keys (Behrouz et al., 2024, 2025a,b). The HLA view is orthogonal: it encodes higher interactions in compact prefix moments that are sufficient for exact masked streaming and scan-parallel training.

Associative memory and Hopfield views. Modern Hopfield networks show that transformer attention is a single-step retrieval in an energy-based associative memory (Ramsauer et al., 2020; Zhong et al., 2025). While this perspective clarifies why attention uses content-addressable memory, standard Hopfield-style layers remain first-order in their sufficient statistics. HLA complements this view by providing explicit higher sufficient statistics with strict causality.

### 9 Conclusion

We introduced Higher-order Linear Attention (HLA), a causal higher attention mechanism with exact streaming updates, a strictly causal masked formulation via extended summaries, and associative scans for parallel training that provably match serial recurrences. At second order, HLA maintains O(d2) state per head and computes each token in O(d2) time, with optional normalization and decay that preserve associativity. We further developed an asymmetric variant (AHLA) and a complete third-order masked algebra with streaming formulas and an exact chunk-parallel scan based on augmented segment maps.

### References

Jimmy Ba, Geoffrey E Hinton, Volodymyr Mnih, Joel Z Leibo, and Catalin Ionescu. Using fast

weights to attend to the recent past. Advances in neural information processing systems, 29, 2016. Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv

preprint arXiv:2501.00663, 2024.

Ali Behrouz, Zeman Li, Praneeth Kacham, Majid Daliri, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, and Vahab Mirrokni. Atlas: Learning to optimally memorize the context at test time. arXiv preprint arXiv:2505.23735, 2025a.

Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. It’s all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. arXiv preprint arXiv:2504.13173, 2025b.

Christian Bischof and Charles Van Loan. The wy representation for products of householder matrices.

SIAM Journal on Scientific and Statistical Computing, 8(1):s2–s13, 1987. Guy E Blelloch. Prefix sums and their applications, 1990.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

Geoffrey E Hinton and David C Plaut. Using fast weights to deblur old memories. In Proceedings of the ninth annual conference of the Cognitive Science Society, pages 177–186, 1987.

Kazuki Irie, Imanol Schlag, R´obert Csord´as, and J¨urgen Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. Advances in neural information processing systems, 34:7703–7717, 2021.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran¸cois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.

Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Xingjian Du, Teddy Ferdinan, Haowen Hou, et al. Eagle and finch: Rwkv with matrixvalued states and dynamic recurrence. arXiv preprint arXiv:2404.05892, 2024.

Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing Liu, Janna Lu, William Merrill, et al. Rwkv-7” goose” with expressive dynamic state evolution. arXiv preprint arXiv:2503.14456, 2025.

Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A Smith, and Lingpeng Kong. Random feature attention. arXiv preprint arXiv:2103.02143, 2021.

Zhen Qin, Dong Li, Weigao Sun, Weixuan Sun, Xuyang Shen, Xiaodong Han, Yunshen Wei, Baohong Lv, Xiao Luo, Yu Qiao, et al. Transnormerllm: A faster and better large language model with improved transnormer. arXiv preprint arXiv:2307.14995, 2023.

Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Lightning attention2: A free lunch for handling unlimited sequence lengths in large language models. arXiv preprint arXiv:2401.04658, 2024.

Hubert Ramsauer, Bernhard Sch¨afl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlovi´c, Geir Kjetil Sandve, et al. Hopfield networks is all you need. arXiv preprint arXiv:2008.02217, 2020.

Imanol Schlag, Kazuki Irie, and Ju¨rgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR, 2021.

Ju¨rgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131–139, 1992.

Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3531–3539, 2021.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,  Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximilian Schlegel, Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meulemans, et al. Mesanet: Sequence modeling by locally optimal test-time training. arXiv preprint arXiv:2506.05233, 2025.

Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024a.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. arXiv preprint arXiv:2406.06484, 2024b.

Shu Zhong, Mingyu Xu, Tenglong Ao, and Guang Shi. Understanding transformer from the perspective of associative memory. arXiv preprint arXiv:2505.19488, 2025.

