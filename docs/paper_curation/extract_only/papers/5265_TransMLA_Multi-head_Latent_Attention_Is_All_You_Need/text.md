arXiv:2502.07864v5[cs.LG]12Jun2025

# TransMLA: MLA Is All You Need

### Fanxu Meng1∗, Pingzhi Tang1∗, Xiaojuan Tang1, Zengwei Yao4, Xing Sun3, Muhan Zhang1,2†

1Institute for Artificial Intelligence, Peking University 2State Key Laboratory of General Artificial Intelligence, BIGAI 3Tencent Youtu Lab, Shanghai, China 4Xiaomi Corp., Beijing, China https://github.com/fxmeng/TransMLA

## Abstract

Modern large-language models often face communication bottlenecks on current hardware rather than computational limitations. Multi-head latent attention (MLA) addresses this by compressing the key-value cache using low-rank matrices, while the Absorb operation prevents the KV cache from reverting to its original size, significantly boosting both training and inference speed. Despite the success of DeepSeek V2/V3/R1, most model providers have heavily invested in optimizing GQA-based models and, therefore, lack strong incentives to retrain MLA-based models from scratch. This paper demonstrates that MLA provides superior expressive power compared to GQA with the same KV cache overhead, thereby offering a rationale for transitioning from GQA to MLA. In addition, we introduce TransMLA, a framework that seamlessly converts any GQA-based pre-trained model (e.g., LLaMA, Qwen, Gemma, Mistral/Mixtral) into an MLA-based model. For the first time, our method enables direct conversion of these models into a format compatible with DeepSeek’s codebase, allowing them to fully leverage DeepSeek-specific optimizations such as vLLM and SGlang. By compressing 93% of the KV cache in LLaMA-2-7B, we achieve a 10.6x speedup with an 8K context length while maintaining meaningful output. Moreover, the model requires only 6B tokens for fine-tuning to recover comparable performance across multiple benchmarks. TransMLA provides a practical path for migrating GQA-based models to the MLA structure, and when combined with DeepSeek’s advanced optimizations—such as FP8 quantization and Multi-Token Prediction—further inference acceleration can be achieved.

[Figure 1]

Cached During Inference

ℎ 𝑔𝑑

𝑑 𝑔𝑑

𝑑

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|𝑔<br><br>|
|---|

|K|
|---|

|𝑇|
|---|

···

< <

𝒑𝒓𝒐𝒋

|Q|
|---|

|MQA|
|---|

MLA

GQA

(a) Given the same KV cache size, the expressiveness increases in the order of GQA, MLA, and MQA.

RoPE 𝑲

𝑹𝒐𝑹𝒐𝑷𝑬 & 𝑭𝒓𝒆𝒒𝑭𝒐𝒍𝒅

Knope

###### V

###### BKV-PCA

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

[Figure 22]

Knope V

Krope Ckv

K

Krope

(b) TransMLA concentrates positional information into Krope and compresses Knope and V .

- Figure 1: GQA, MLA, and MQA can be equivalently transformed in one direction, illustrating a gradual increase in expressive power. RoRoPE aggregates positional information in the first head, eliminating the need for RoPE in others. FreqFold further enhances this effect. Finally, after balancing the magnitudes of Krope and V , a joint low-rank approximation is applied to compress the KV cache.

∗Equal contribution. †Corresponding author: muhan@pku.edu.cn

Preprint. Under review.

## 1 Introduction

Advanced Large language models (LLMs)—GPT-4o OpenAI [2024], Claude 3.7 Sonnet Anthropic

- [2024], Gemini-2.5 Team et al. [2024a], LLaMA-4 AI@Meta [2024], Mistral-3 Mistral [2024], Qwen-3 Qwen [2024], DeepSeek V3/R1 Liu et al. [2024a], Guo et al. [2025], Gemma-3 Team

- et al. [2024b], and Phi-4 Abdin et al. [2024]—rapidly evolving frontier for both research and applications. LLMs rely on next-token prediction Radford [2018], Brown et al. [2020]: tokens are generated sequentially, and self-attention is computed over all preceding tokens. To avoid redundant computation, implementations store the intermediate key–value (KV) pairs in a cache. Yet, as model and context sizes grow, the KV cache itself becomes a major bottleneck for inference.

To mitigate these challenges, Group-Query Attention (GQA) Ainslie et al. [2023] groups the query heads so that every head within a group shares a single key and value head. When the number of groups is one, GQA degenerates to Multi-Query Attention (MQA) Shazeer [2019]. When the number of groups equals the number of heads, it reduces to the standard Multi-Head Attention (MHA) Vaswani et al. [2017]. While both GQA and MQA cut the size of the KV cache relative to MHA, they do so at the cost of model quality. Post-training KV-cache compression techniques—such as Duo-Attention Xiao et al. [2024], KiVi Liu et al. [2024b], KV-Quant Hooper et al. [2024], and H2O Zhang et al. [2023]—further shrink memory usage, but their non-standard implementations demand specialized optimizations, hindering widespread adoption.

Multi-Head Latent Attention (MLA)—introduced with DeepSeek V2 DeepSeek-AI [2024] and further refined in DeepSeek V3 DeepSeek-AI [2024] and DeepSeek R1 Guo et al. [2025]—offers a pre-trained KV-cache compression strategy that strikes an excellent balance between computational efficiency and model quality. Models equipped with MLA deliver state-of-the-art results while driving training and inference costs to new lows. Moreover, the DeepSeek team’s ongoing commitment to open-source releases provides highly optimized implementations and deployment recipes, making these advances readily accessible to the community.

In this paper, we first prove that MLA consistently offers higher expressive power than GQA under the same KV cache overhead, which theoretically explains the advantage of MLA. However, a practical obstacle preventing model vendors from switching to MLA is the substantial prior investment on GQA-based models. This motivates us to ask, can we seamlessly convert a GQA-based pretrained model, such as LLaMA AI@Meta [2024] and Qwen Qwen [2024], to MLA so that we can inherit the model weights and pretraining effort, rather than training MLA from scratch?

A key obstacle to converting a GQA-based model to MLA is that every query–key head carries its own Rotary Positional Embedding (RoPE) [Su et al., 2024], blocking the Absorb operation [Chang et al., 2024] that DeepSeek uses to switch between compute- and memory-efficient modes. Borrowing from DeepSeek’s Decoupled RoPE scheme, we concentrate the positional signal in K into a small subset of dimensions, Krope. The remaining dimensions, Knope, contain little positional content; we drop their RoPE and merge them with V for low-rank decomposition. Once RoPE is isolated, the key up-projection—now RoPE-free—can be absorbed into the query projection exactly as in DeepSeek, enabling seamless MLA conversion.

To efficiently concentrate positional information into fewer dimensions, we introduce RoRoPE—a novel technique that performs principal component analysis (PCA) on the key output, applies rotation across the two ends of RoPE, and consolidates the principal components of all attention heads into the dimensions of the first attention head. We theoretically prove that the product remains invariant after rotating the query and key using a matrix U, as long as U satisfies two conditions: (1) rotation occurs only within the same dimension across all attention heads, and (2) the real and imaginary components of RoPE are rotated in the same manner. Additionally, by exploiting the frequency similarity between adjacent RoPE dimensions, we propose FreqFold, a technique that improves the concentration efficiency in the first attention head.

Finally, we found that the ℓ2-norm of Knope is far larger than that of V . If we run PCA on the concatenated matrix Knope;V without adjustment, the principal components are dominated by Knope, leading to severe information loss from the value subspace and a sharp drop in accuracy. We therefore introduce a Balanced Key–Value (BKV) procedure: we first rescale Knope and V so that their norms match, and only then perform joint PCA. This simple normalization restores balance between the two subspaces and delivers a marked improvement in compression quality.

The above innovations collectively form our TransMLA method. Using TransMLA, we compressed the KV cache of LLaMA-2 by 68.75%, with only a 1.65% performance drop across 6 benchmarks for training free. In contrast, a concurrent method, MHA2MLA Ji et al. [2025], experienced a 21.85% performance decline. At a compression rate of 93%, the model still maintained meaningful responses, and after training with just 6B tokens, its performance was mostly restored. We tested both the original and TransMLA models on three different hardware setups using vLLM, achieving up to a 10.6x speedup compared to the original GQA models, demonstrating the great potential of TransMLA. Moreover, the TransMLA models are fully compatible with DeepSeek’s code, enjoying DeepSeek’s ecosystem to accelerate inference and seamlessly integrate with various hardware and frameworks.

## 2 Related Work

In large language models, autoregressive decoding reuses past activations by storing their key–value (KV) pairs in a cache. Because the size of this cache grows linearly with sequence length, its memory footprint quickly becomes the limiting factor for very long contexts. Consequently, shrinking the KV cache without compromising accuracy has become a pivotal research focus, motivating a spectrum of architectural innovations and compression strategies.

Multi-Query Attention (MQA) Shazeer [2019] and Group-Query Attention (GQA) Ainslie et al. [2023] shrink the KV cache by letting every query in a group share a single key and value head. Although both schemes save memory relative to Multi-Head Attention (MHA) Vaswani et al. [2017], they usually give up some accuracy. Multi-Head Latent Attention (MLA)—introduced with DeepSeek V2 DeepSeek-AI [2024] and refined in later releases DeepSeek V3/R1 DeepSeek-AI [2024], Guo

- et al. [2025]—offers a more favorable trade-off, delivering near-state-of-the-art quality while cutting training and inference costs. Grouped Latent Attention (GLA) Zadouri et al. [2025] provides a parallel-friendly implementation of latent attention that further accelerates MLA inference. By contrast, Tensor Product Attention (TPA) Zhang et al. [2025] tackles the memory bottleneck by dynamically factorizing activations, slashing the runtime KV cache by an order of magnitude, but it necessitates training the model from scratch. TransMLA fills this gap: rather than proposing yet another attention variant, it converts an existing GQA model into an MLA model with only light fine-tuning, restoring accuracy while inheriting MLA’s memory and speed advantages.

Another approach is to optimize the KV cache of existing pre-trained models. For example, dynamic token pruning is employed by LazyLLM Fu et al. [2024], A2SF Jo and Shin [2024], and SnapKV Li et al. [2024]. These methods selectively prune less important tokens from the KV cache. Sharing KV representations across layers, as in YONO Sun et al. [2024], MiniCache Liu et al. [2024c], and MLKV Zuhri et al. [2024], reduces memory by reusing the same KV cache across multiple layers. This can drastically lower memory usage and speed up inference. Although effective, both families of methods usually require custom kernels or runtime tweaks, complicating deployment and limiting adoption. TransMLA, by contrast, plugs directly into the mature DeepSeek ecosystem—converted checkpoints load out-of-the-box, delivering MLA-level speed-ups across every DeepSeek-supported platform.

There are two works most related to TransMLA. One is Palu Chang et al. [2024], which reduces KV cache size by applying low-rank decomposition on both the keys and values, enabling speedup through tailored optimizations. However, Palu does not specifically handle RoPE, which prevents it from using the Absorb operation during inference. Therefore, Palu needs to project the compressed representations back to their original size. This projection incurs significant computational overhead during inference, limiting the overall acceleration. Another concurrent work, MHA2MLA Ji et al.

- [2025], also claims to convert MHA to MLA and decouple RoPE from the main computational path. It is important to clarify that TransMLA is not simply a GQA extension of MHA2MLA—both TransMLA and MHA2MLA support MHA and GQA architectures. However, MHA2MLA determines which RoPE dimensions to remove solely based on the norms of the query and key vectors, which tends to cause larger information loss when pruning the same proportion of positions. Also, the distribution of important dimensions in MHA2MLA is uneven, requiring sparse indexing that complicates optimization and acceleration. Their work reports compression ratios of the KV cache but does not demonstrate actual inference speedup. Furthermore, MHA2MLA directly applies joint singular value decomposition to KV, resulting in higher loss compared to our balanced key-value PCA method.

## 3 Preliminary

### 3.1 Rotary Position Embedding (RoPE)

RoPE Su et al. [2024] is a position encoding method that encodes the absolute positions with different rotations and incorporates the explicit relative position dependency in the self-attention formulation. It applies different rotations to tokens in different positions to encode the position information.

Consider xt ∈ Rd to be the embedding of the t-th token with the hidden size d. The RoPE operation upon xt produces a representation xRt that encodes both semantic and positional information:









−x(2)t x(1)t −x(4)t x(3)t .

- x(1)t
- x(2)t
- x(3)t
- x(4)t









- sintθ1

- sintθ1
- sintθ2

- sintθ2

- costθ1

- costθ1
- costθ2

- costθ2

xRt = RoPE(xt,t) =

, (1)

⊗

⊗

+

. sintθd/2 sintθd/2

. costθd/2 costθd/2

 

 

 

 

. x(td−1) x(td)

 

 

 

 

−x(td) xt(d−1)

where ⊗ denotes the element-wise multiplication of two vectors, x(ti) ∈ R denotes the i-th element of xt, and θi = 10000−2(i−1)/d is the i-th rotation angle. If we interpret every two elements in the embedding as a representation in the complex coordinate system, we can divide xt into paired dimensions, where the odd-indexed dimensions x(2t k−1) represent the real parts and the even-indexed dimensions x(2t k) represent the imaginary parts.

### 3.2 Group Query Attention

Let the t-th token of the input sequence be xt ∈ RD, where D denotes the hidden dimension. To reduce the memory overhead of the KV cache, GQA divides the h query heads uniformly into g groups, with all query heads within a group sharing the same key and value vectors. Specifically, let WQ ∈ Rhd×D, WK,WV ∈ Rgd×D and WO ∈ RD×hd be the projection matrices for the query, key, value and output, where d = D/h denotes the dimension per head. GQA first computes the concatenated queries qt, keys kt, and values vt, and then slices them into heads or groups for attention computation:

[qt,1;qt,2;...;qt,h] = qt = WQxt, (2) [kt,1;kt,2;...;kt,g] = kt = WKxt, (3) [vt,1;vt,2;...;vt,g] = vt = WV xt, (4)

where each qt,i ∈ Rd corresponds to the query vector of the i-th head, and kt,j,vt,j ∈ Rd correspond to the key and value vectors of the j-th group.

Using the notation in Section 3.1, after applying RoPE to qt,i,kt,i, we can obtain the attention output for the t-th token as follows:

qRt,i⊤kRj,⌈i/h

t

√ g⌉ d

g⌉, (5) yt = WO[ot,1;ot,2;...;ot,h]. (6)

softmaxj(

ot,i =

)vj,⌈i/h

j=1

As we can see, in GQA, each key and value head corresponds to hg query heads. When g = h, GQA becomes MHA, and when g = 1, GQA becomes Multi-Query Attention (MQA).

### 3.3 Multi-Head Latent Attention

MLA saves KV cache by multiplying the matrix WDKV ∈ Rr

kv×D with the input sequence to obtain low-rank latent features. Then, it uses the matrices WUK and WUV ∈ Rhd×r

kv to derive

the key k and value v representations for each attention head. Additionally, MLA also decomposes WQ to WDQ ∈ Rr

q, which reduces the activation memory during training. For positional embedding, MLA uses a decoupled RoPE strategy that uses additional multi-head queries qRt,i ∈ Rd

q×D and WUQ ∈ Rhd×r

R×rq and WKR ∈ Rd

R

R

and a shared key kRt ∈ Rd

, which are generated from WQR ∈ Rhd

R×d, to carry RoPE, where dR denotes the per-head dimension of the decoupled queries and key.

cQt = WDQxt, [qCt,1;qCt,2;...;qCt,h] = qCt = WUQcQt , [qRt,1;qRt,2;...;qRt,h] = qRt = RoPE(WQRcQt ,t),

cKVt = WDKV xt,

[kCt,1;kCt,2;...;kCt,h] = kCt = WUKcKVt ,

kRt = RoPE(WKRxt,t), kt,i = [kCt,i;kRt ], (7)

qt,i = [qCt,i;qRt,i]. (8)

MLA supports switching between two computational paradigms tailored for different stages. During the compute-intensive training phase, it operates in a paradigm similar to standard MHA, where the computational overhead is slightly lower than that of conventional MHA, as shown in Equation 9. For communication-intensive inference, it can seamlessly switch to a paradigm resembling MQA, as described in Equation 10. In this inference paradigm, the latent features function as a shared large KV head, which interacts with all query heads and output heads to produce the final output efficiently. This operation is called the Absorb operation, which is crucial for accelerating inference speed.

ˆqt,i = [WiUK⊤qCt,i; qRt,i], kˆt = [cKVt ; kRt ],

[vt,C1; vt,C2; ...; vt,hC ] = vtC = WUV cKVt ,

t

qTt,ikj,i √d + dR

t

ˆqTt,ikˆj √d + dR

)vj,iC ,

softmaxj(

)cKVj ,

softmaxj(

ot,i =

oˆt,i =

j=1

j=1

yt = WO[ot,1; ot,2; ...; ot,h], (9)

yt = WO[W1UV oˆt,1; ...; WhUV oˆt,h], (10)

where Wi{UK,UV } denotes slices of the projection matrices corresponding to the i-th attention head. One of the main contributions of this paper is the seamless support for the Absorb operation, significantly enhancing inference speed.

- 4 TransMLA In this section we formally present TransMLA, motivated by two observations:

- 1. For a fixed KV-cache budget, MLA is strictly more expressive than GQA. As proven in Appendix A (and illustrated in Figure 1a), any GQA layer can be rewritten as an MLA layer by introducing a single additional projection matrix. The reverse transformation is not always possible, implying that MLA subsumes GQA. When Rotary Positional Embeddings (RoPE) are present, the MLA equivalent must be expressed in the absorbed form.
- 2. Inference acceleration occurs only when MLA uses a smaller KV cache. Although one can build an MLA-equivalent representation of a GQA model, speedups arise only if the number of stored KV vectors is actually reduced. TransMLA therefore converts a GQA-based network into a DeepSeek-like MLA architecture, allowing the transformed model to run directly on DeepSeek’s optimized inference stack and realize the full memory–latency benefits.

### 4.1 Merging All Key Heads as One

Because MLA ties all KV heads to a single latent dimension, the first step in converting a GQA layer to MLA is to merge every GQA key–value group into one latent head before any KV-cache compression is applied. For each query head i, we introduce WiUK ∈ Rd×gd with the group index j = h/g i − 1, and initialize the matrix WiUK[:,jd : (j + 1)d] to be Id (identity matrix of shape d × d), with all other elements set to 0. (We adopt the matrix indexing notation used in Python and PyTorch, and will continue to do so throughout this work without further explanation.) This initialization allows the projection of the key’s latent representation to be simultaneously mapped

Group 1 Group 2 Group 3 Group 4

dim=1 dim=2 dim=3 dim=4 dim=5 dim=6 dim=7 dim=8

PCA

RoPE NoPE NoPE NoPE

- Figure 2: Pipeline of RoRoPE for decoupling RoPE. Blue lines denote real-part dimensions, orange lines denote imaginary-part dimensions. When the keys from multiple heads are concatenated, permuting dimensions does not change the computation, so we gather the same dimension (i.e., the same rotational frequency) across all heads and apply joint principal-component analysis—using the identical procedure for the real and imaginary parts. For each frequency we keep a single principal component, which captures the dominant positional variation and can be represented by a standard RoPE in one attention head.

onto multiple query heads, with only the corresponding key head being multiplied by the appropriate query head. Similarly, during the computation of the multiple heads of the values, the attention scores for each head are multiplied accordingly. To keep the output unchanged, we similarly initialize WiUV so that only the corresponding value head is an identity mapping, while all other elements are set to zero. Since we have now merged all the key heads into a single head, in order to ensure that this transformation is equivalent to the original, we also merge the RoPE operations from different heads into one large RoPE operation, denoted as RoPE, applied to the entire merged key head. Since the original RoPE operation in GQA is the same for each head, RoPE simply applies the same RoPE operation repeatedly for every d dimensions. In this way, the computation of GQA attention is transformed into the following form:

WK WV ∈ R2gd×D (11)

[cKt ;cVt ] = cKVt = WDKV xt, WDKV =

[qt,1;qt,2;...;qt,h] = qt = WQxt, WQ ∈ Rhd×D (12) qˆRt,i = RoPE( WiUK ⊤ qt,i,t), kˆRt = RoPE(cKt ,t) (13)

qˆRt,i ⊤ kˆRj

t

cVj , (14)

softmaxj

√

ˆt,i =

d

j=1

#### yt = WO[W1UV ˆt,1;...;WhUV ˆt,h]. (15)

It is evident that the total KV cache size remains unchanged since we still need to store cKVt ∈ R2gd for each token, which is the same as in the original GQA model. However, the dimension of each

attention head has increased by a factor of g, and the introduction of new parameters WiUK and WiUV leads to higher computational costs. To achieve actual acceleration in the transformed MLA, compressing the KV cache is therefore essential. By merging multiple KV heads, we can better identify shared principal components and represent the KV cache in a lower-dimensional latent space. Moreover, merging multiple key heads is crucial for efficiently decoupling RoPE in the subsequent steps.

### 4.2 Rotating Queries and Keys to Minimize Transformation Loss Towards Decoupled RoPE

- As illustrated in Figure 2, applying RoRoPE to the merged head removes the bulk of the positional

signal from K. In Equation 14, the term qˆRt,i ⊤ kˆRj can be expressed as the sum of inner products over paired dimensions across multiple attention heads, incorporating positional information:

d/2

qˆRt,i ⊤ kˆRj =

l=1

qˆt,i[2l−1::d];qˆ[2t,il::d]

R⊤ k ˆ[2j l−1::d];kˆ[2j l::d]

R

. (16)

Here, the notation [2l :: d] is inspired by Python slicing syntax and denotes selecting elements starting from the (2l)-th dimension, then taking every d-th element thereafter until the end of the vector. The

R

vector qˆ[2t,il−1::d];qˆ[2t,il::d]

thus has dimension 2g.

Since multiple key heads are concatenated into a single head and each head shares the same RoPE, the real and imaginary components corresponding to the l-th 2D subspace (i.e., the paired two dimensions) of RoPE within each original attention head can be expressed as follows:

R

= costθl q ˆt,i[2l−1::d];qˆ[2t,il::d] + sintθl −qˆ[2t,il::d];qˆ[2t,il−1::d] , (17) k ˆ[2j l−1::d];kˆj[2l::d]

q ˆ[2t,il−1::d];qˆ[2t,il::d]

R

= cosjθl k ˆj[2l−1::d];kˆ[2j l::d] + sinjθl −kˆ[2j l::d];kˆ[2j l−1::d] . (18)

When the concatenated real and imaginary components of qˆt,i and kˆj within the l-th 2-dimensional subspace are multiplied by an orthogonal matrix Ul ∈ Rg×g, the inner product with RoPE applied remains invariant. Specifically,

d/2

l=1

Ulqˆ[2t,il−1::d];Ulqˆt,i[2l::d]

R⊤

Ulkˆ[2j l−1::d];Ulkˆ[2j l::d]

R

⊤

t,i kˆRj . (19)

= qˆR

This demonstrates that the rotational transformation Ul preserves the RoPE-based inner product structure. Simply put, because the same rotation values (i.e., costθl and sintθl) are applied identically to each dimension of the l-th 2D subspace across all attention heads, any orthogonal transformation

⊤

t,i kˆRj unchanged. However, the preceding equation reveals a critical constraint: for the inner product’s value to remain unchanged after transformation, the same orthogonal matrix Ul must be applied to both the real (2l − 1) and imaginary (2l) components of the key vectors within each 2D subspace. For a detailed proof and our proposed solution, please refer to Appendix B.

U⊤l Ul = I applied to these dimensions within the same subspace leaves the inner product qˆR

We apply Principal Component Analysis (PCA) to the attention heads in the context of RoPE, introducing a method we call RoRoPE. Using a small dataset, we extract the key output, compute its principal component projection matrices {Ul}l∈{1,...,d/2}, and rotate both WK and WUK (as shown in Eq13, qˆt,i and kˆj are generated from WUK and WK respectively, so rotating qˆt,i and kˆj can be achieved by rotating WUK and WK) using a similar procedure as described above. This rotation effectively concentrates the essential information into the first few heads. As an equivalent transformation, instead of discarding all non-principal components of the key, we remove their RoPE encoding while preserving positional information within the principal components.

To enable the transformed model to utilize standard RoPE, we represent the principal component information for corresponding positions across other heads using the dimensions of the first attention head. However, using a one-dimensional space for all positional information proves limiting. To address this, we exploit the similar frequencies of adjacent dimensions in RoPE, treating them as equivalent positions. This allows us to use multiple dimensions within a single attention head to represent positional information, a technique we refer to as FreqFold. Additional information on FreqFold can be found in Appendix C.

### 4.3 A Balanced Approach to Joint Low-Rank Approximation of NoPE Keys and Values

In the previous section, we split the key heads into one carrying positional information and the others without positional information, achieving minimal loss. We then apply Principal Component Analysis

(PCA) jointly on the values and the non-positional components of the keys (i.e. NoPE-Key), using activations collected from a small calibration dataset, thereby compressing the projection matrices into a low-rank latent space. However, we observed that although the principal components of the keys were effectively separated with RoRoPE, the norm of the residual key features remained significantly larger than that of the Value. This imbalance caused the direct decomposition to favor principal component directions dominated by the keys.

To mitigate this, we scale WDK by dividing it by

Et[∥WNoPEDK xt∥2] Et[∥WDV xt∥2]

(20)

α =

and correspondingly scale WUK by multiplying it by α. Here, WRoPEDK ∈ Rd×D,WNoPEDK ∈ R(g−1)d×D represent the parts of WDK obtained after the operations described in the previous section, where

WRoPEDK corresponds to one head that uses RoPE, and WNoPEDK corresponds to the remaining heads that do not use RoPE.

This transformation is mathematically equivalent and does not affect the overall model outputs, while significantly enhancing the effectiveness of KV cache compression in subsequent steps. More details is provided in the Appendix D.

## 5 Experiment

### 5.1 Main Experiment

In this section, we present our main experimental results. Following the experimental setup of MHA2MLA, we converted two models—smolLM 1.7B and Llama 2 7B—into the MLA architecture. We evaluated the models’ performance on six benchmarks at three stages: before conversion, immediately after conversion without further training, and after conversion followed by training. For the training process, we used a subset of the pretraining corpus used for the smolLM model. The fine-tuned results of MHA2MLA are taken directly from the original paper. Our experiments were conducted on an 8-GPU machine, each GPU having 40GB of memory and delivering 312 TFLOPS of FP16 compute power. Detailed experimental hyperparameter settings are provided in the Appendix E.

From Table 1, we observe that TransMLA efficiently facilitates architecture migration across various models and KV cache compression ratios. Notably, the untrained performance of MLA models initialized with TransMLA shows minimal degradation in capability compared to the original models—significantly less than the degradation observed with MHA2MLA under equivalent KV cache compression. In fact, using TransMLA to compress the KV cache of Llama 2 7B to just 7.03% of its original size still results in better performance than MHA2MLA’s compression to 31.25% on the same model. This highlights the effectiveness of our proposed techniques includes RoRoPE, FreqFold and activation-based balanced KV low-rank factorization.

The low-loss transformation achieved by TransMLA enables us to recover the original model performance with minimal training overhead. As shown in the table, TransMLA achieves stronger performance than MHA2MLA-6B while using significantly fewer training tokens. For instance, when transforming smolLM 1.7B and compressing the KV cache to 31.25% of its original size, we only need 4.9% of the training data used by MHA2MLA and 2 hours training to surpass its performance.

### 5.2 Key Norm Analysis Reveals the Impact of RoRoPE and FreqFold

In this section, we conduct a detailed analysis of the distribution of key activations in the attention module to demonstrate the effectiveness of our proposed methods.

Figure 3a presents the average L2 norm of each key dimension in the first attention layer of the LLaMA 3 8B model, computed on a subset of the WikiText-2 Merity et al. [2016] dataset. We compare the original model (in blue), the model transformed using our RoRoPE equivalence method (in orange), and the further approximated model using 4D FreqFold (in green). The top and bottom halves of the plot correspond to pairs of dimensions that share the same rotation angle in RoPE, which we refer to as the real (Re-dim) and imaginary (Im-dim) dimensions.

Table 1: Commonsense reasoning ability of two LLMs with TransMLA compared to MHA2MLA. The six benchmarks include MMLU (Hendrycks et al. [2021]), ARC easy and challenge (ARC, Clark et al. [2018]), PIQA (Bisk et al. [2020]), HellaSwag (HS, Zellers et al. [2019]), OpenBookQA (OBQA, Mihaylov et al. [2018]), and Winogrande (WG, Sakaguchi et al. [2021]). Tokens refers to the number of tokens used for further training after the TransMLA conversion. A value of 0 indicates that the model was evaluated immediately after conversion, without any fine-tuning.

Model Tokens KV Mem. Avg. MMLU ARC PIQA HS OBQA WG

SmolLM-1.7B 1T – 55.90 39.27 59.87 75.73 62.93 42.80 54.85

- - MHA2MLA

0

- -68.75% 40.97 27.73 41.96 63.00 29.19 34.40 49.53

- -81.25% 37.14 26.57 32.73 55.77 26.90 31.40 49.49
- -87.50% 34.01 25.32 27.15 51.36 25.47 26.20 48.54

- -68.75% 54.76 38.11 57.13 76.12 61.35 42.00 53.83

6B -81.25% 54.65 37.87 56.81 75.84 60.41 42.60 54.38 -87.50% 53.61 37.17 55.50 74.86 58.55 41.20 54.38

- - TransMLA

-68.75% 51.95 35.70 55.68 73.94 53.04 39.80 53.51 0 -81.25% 47.73 32.87 47.89 69.75 48.16 36.20 51.46 -87.50% 44.12 29.97 41.72 66.87 41.15 34.80 50.28

300M -68.75% 55.24 38.60 58.95 74.97 61.52 43.00 54.38 700M -81.25% 54.78 37.79 57.53 75.52 59.88 42.80 55.17

1B -87.50% 54.01 37.24 56.32 74.81 60.08 42.40 53.20 LLaMA-2-7B 2T – 59.85 41.43 59.24 78.40 73.29 41.80 64.96

- - MHA2MLA

-68.75% 37.90 25.74 32.87 59.41 28.68 28.60 52.09 0 -81.25% 34.02 25.50 26.44 53.43 27.19 22.60 49.01

- -87.50% 32.70 25.41 25.79 50.60 26.52 19.40 48.46

- -68.75% 59.51 41.36 59.51 77.37 71.72 44.20 62.90

6B -81.25% 59.61 40.86 59.74 77.75 70.75 45.60 62.98 -87.50% 58.96 40.39 59.29 77.75 69.70 43.40 63.22

- - TransMLA

- -68.75% 58.20 39.90 57.66 77.48 70.22 41.00 62.90

- -87.50% 51.19 34.39 45.38 71.27 60.73 37.40 57.93
- -92.97% 43.26 28.93 36.32 63.38 45.87 31.60 53.43

0

500M -68.75% 59.82 40.87 59.18 77.91 71.82 45.20 63.93 3B -87.50% 59.36 40.77 58.84 78.18 71.28 43.60 63.46 6B -92.97% 58.68 40.82 59.72 76.55 69.97 43.60 61.40

We observe that the original model exhibits a highly uneven norm distribution across key dimensions, with numerous outliers. This suggests that naively removing RoPE from certain heads would likely result in significant performance degradation. After applying our RoRoPE transformation, as shown by the orange line, the key dimensions with large norms are nearly all concentrated to the first two heads (dimension 0-128). Further applying the 4D FreqFold approximation compresses the tail (i.e., higher-index dimensions) even more, leading to an even sharper concentration of high-norm key dimensions. This concentrated structure is highly beneficial for the subsequent RoPE removal step as shown in Figure 3b.

In Figure 3b, we present the log-perplexity of the LLaMA 3 8B model on WikiText-2 as RoPE components are progressively removed. We observe that our proposed RoRoPE method significantly outperforms MHA2MLA’s per-head dimension selection strategy, especially at high removal ratio. Furthermore, incorporating similar-dimension approximation leads to even better performance under extreme removal rates. At 90% removal ration, RoRoPE + 4D-FreqFold still maintains a logperplexity about 2, while MHA2MLA reaches nearly 6, which no longer generates meaningful outputs.

- At the same time, we observe that overly aggressive FreqFold (i.e., using too many dimensions) can degrade performance, as the loss introduced by approximation of nearby dimensions can outweigh the benefit in concentrating the principal components. This figure suggests that for LLaMA 3 8B, the sweet spot lies in applying RoRoPE combined with 4D FreqFold.

- -normofRe-dimL2

0 64 128 192 256 320 384 448 512 Dimension

0

25

50

75

- -normofIm-dimL2

100

Original RoRoPE

RoRoPE & 4D-FreqFold

50

0

0 64 128 192 256 320 384 448 512

(a) Key norms of the first layer.

8

Vanilla PCA

LogofPerplexity

MHA2MLA

6

RoRoPE

| |
|---|

RoRoPE & 2D-FreqFold RoRoPE & 4D-FreqFold RoRoPE & 8D-FreqFold

4

| |
|---|

2

0 20 40 60 80 RoPE Removal Ratio (%)

(b) RoPE removal results, evaluated on WikiText-2.

- Figure 3: Visualization of key norms and RoPE removal results on LLaMA 3 8B model. The top and bottom halves of the left figure correspond to pairs of dimensions that share the same rotation angle in RoPE, which we refer to as the real (Re-dim) and imaginary (Im-dim) dimensions.

0 128 256 384 512 640 768 896 1024

0

5

10

15

20

Originalkv-norm

0 128 256 384 512 640 768 896 1024 Dimension

0.0

0.5

1.0

1.5

2.0

Balancedkv-norm

Key

Value

(a) Norm magnitude of keys and values before and after KV balancing is applied.

| |W-based<br><br>W-based with BKV<br><br>WX-based<br><br>WX-based with BKV| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

75.0 78.125 87.5 Key-Value Cache Compression Ratio (%)

- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10

LogofPerplexity

(b) Comparison of different methods of KV cache compression, with and without KV balancing.

- Figure 4: Visualization of the norms of keys and values for the first layer of LLaMA 3 8B and the perplexity results after joint low-rank compression of keys and values under WikiText-2. Here, W-based and WX-based refer to PCA applied on the attention weights and the activation outputs, respectively. BKV denotes the application of KV balancing.

### 5.3 Key-Value Norm Disparity Motivates KV Balancing

- In Figure 4a, we visualize the norm magnitudes of the key and value activations in the first layer of LLaMA 3 8B before and after KV balancing. Note that both the key and value shown here are activations after applying the RoRoPE principal component concentration, and the first head of the key—reserved for RoPE—is excluded. As a result, the value and the remaining key components shown in the figure are precisely the elements we aim to compress jointly into a lower-dimensional space via PCA.

It is evident that even after removing the first head with the highest norm, the overall norm of the key remains significantly larger than that of the value. This norm disparity poses a substantial challenge for joint compression into a shared latent space. In particular, such imbalance can bias the PCA toward directions aligned with the key, rather than the value, leading to suboptimal representation of the value components.

The lower part of Figure 4a shows the norm distribution of keys and values after applying KV balancing. At this point, the norms of keys and values become more aligned, which is beneficial for performing joint PCA. This observation is further supported by the results in Figure 4b, where KV balancing consistently reduces the loss incurred by jointly applying low-rank approximation to keys and values—whether the PCA is based on weights or on activations. Figure 4b also demonstrates that activation-based PCA yields significantly better results than weight-based PCA.

### 5.4 Hardware-Agnostic Inference Speedup with TransMLA

By converting MHA/GQA models into MLA models that are fully compatible with the DeepSeek codebase and compressing the KV cache, TransMLA enables us to leverage all optimizations and tooling available in DeepSeek. Using the vLLM framework, we achieve substantial real-world inference speedups.

- In Figure 5, we benchmarked the inference performance of an MLA model—with a 92.97% reduction in KV cache size—on three consumer-grade AI accelerators with different compute capabilities and memory sizes: 165.2 TFLOPS with 24GB memory, 312 TFLOPS with 40GB memory, and 320 TFLOPS with 64GB memory. The figure shows the inference speedup of the MLA model relative to the original MHA model. Low-rank Q and Full-rank Q indicate whether the query projections were also compressed. Context length represents the total sequence length (i.e., context length plus generated tokens).

Our experiments show that the inference speedup of MLA models increases as the context length grows, which aligns with our expectations. Since the primary performance gain of MLA stems from KV cache compression, longer contexts lead to more substantial savings and thus higher speedups. Remarkably, for the 8K context window on the first hardware platform, the TransMLA-transformed model achieves an impressive 10.6x inference acceleration. To the best of our knowledge, the MHA2MLA method has not reported any inference speedup results.

10.6x Low Rank Q = 512

10

Full Rank Q

Speedup

8

6

4

1k 2k 4k 8k Context Length

(a) 165.2 TFLOPS | 24GB

Low Rank Q = 512 6.5x

Full Rank Q

6

Speedup

4

2

1k 2k 4k 8k 16k 32k Context Length

(b) 312 TFLOPS | 40GB

- 1

- 2

- 3

- 4

- 5

- 6

Low Rank Q = 512 5.1x

Full Rank Q

Speedup

1k 2k 4k 8k 16k 32k Context Length

(c) 320 TFLOPS | 64GB

- Figure 5: Inference speedups with TransMLA comparing to the original LLaMA2 7B model on three consumer-grade AI accelerators. Low-rank Q and Full-rank Q indicate whether the query projections were also compressed. Context length represents the total sequence length.
- 6 Conclusion, Limitation and Future Work

In this work, we demonstrate that the expressive power of TransMLA is stronger than GQA under the same KV cache. To help existing GQA transition to the MLA structure with minimal cost, we propose the TransMLA method. By using the RoRoPE method, the multi-head KV positional information is concentrated into the first head, and FreqFold further enhances this extraction effect. The positional information of the remaining query-key heads is removed, and the Balance KV norm method is used to jointly compress the values and the remaining heads of keys. TransMLA can convert models such as LLaMA and Qwen into MLA-based models, and the converted model incurs very little loss compared to the original model, with performance being recoverable through training with only a few tokens. Additionally, TransMLA can easily leverage the DeepSeek ecosystem for accelerated inference, achieving significant throughput improvements across various hardware platforms.

Although TransMLA significantly reduces the loss introduced by decoupling RoPE via RoRoPE and FreqFold, and alleviates KV cache compression loss through KV balancing, the KV balancing technique itself is relatively trivial. Are there alternative, more powerful mathematical tools that can better handle the disparity between the norm of the keys and values and deliver improved performance at the same compression rate, thereby enabling truly training-free conversion? Furthermore, TransMLA needs to be validated across a broader range of models, and should be integrated with pruning, quantization, token selection, and other optimization techniques to fully explore the upper bounds of inference acceleration.

## References

OpenAI. Hello GPT-4o, 2024. URL https://openai.com/index/hello-gpt-4o/. Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/

claude-3-5-sonnet.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024a.

AI@Meta. Llama 3 model card, 2024. URL https://github.com/meta-llama/llama3/blob/

main/MODEL_CARD.md.

Mistral. Cheaper, better, faster, stronger: Continuing to push the frontier of ai and making it accessible to all, 2024. URL https://mistral.ai/news/mixtral-8x22b.

Qwen. Qwen2.5: A party of foundation models, 2024. URL https://qwenlm.github.io/blog/

qwen2.5.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024a.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024b.

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Alec Radford. Improving language understanding by generative pre-training. 2018. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,

Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017.

Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. arXiv preprint arXiv:2410.10819, 2024.

Zhiyang Liu, Dong Zhang, Xinyi Li, and Ji Wu. Kivi: Quantized key-value representation for efficient long-context transformers. arXiv preprint arXiv:2402.06732, 2024b.

James Hooper, Li Dai, Zhen Zhang, and Seung-Hwan Lee. Kvquant: Quantization for efficient key-value caching in transformer models. arXiv preprint arXiv:2402.12345, 2024.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023.

DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. CoRR, abs/2405.04434, 2024. URL https://doi.org/10.48550/arXiv.2405.04434.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi Huang, Luis Ceze, Mohamed S Abdelfattah, and Kai-Chiang Wu. Palu: Compressing kv-cache with low-rank projection. arXiv preprint arXiv:2407.21118, 2024.

Tao Ji, Bin Guo, Yuanbin Wu, Qipeng Guo, Lixing Shen, Zhan Chen, Xipeng Qiu, Qi Zhang, and Tao Gui. Towards economical inference: Enabling deepseek’s multi-head latent attention in any transformer-based llms. arXiv preprint arXiv:2502.14837, 2025.

Ted Zadouri, Hubert Strauss, and Tri Dao. Hardware-efficient attention for fast decoding. arXiv preprint arXiv:2505.21487, 2025.

Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Zhen Qin, Yang Yuan, Quanquan Gu, and Andrew Chi-Chih Yao. Tensor product attention is all you need. arXiv preprint arXiv:2501.06425, 2025.

Qichen Fu, Minsik Cho, Thomas Merth, Sachin Mehta, Mohammad Rastegari, and Mahyar Najibi. Lazyllm: Dynamic token pruning for efficient long context llm inference. arXiv preprint arXiv:2407.14057, 2024.

Hyun-rae Jo and Dongkun Shin. A2sf: Accumulative attention scoring with forgetting factor for token pruning in transformer decoder. arXiv preprint arXiv:2407.20485, 2024.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024.

Tian Sun, Li Zhang, and Shuang Wu. You only need one: Efficient kv sharing across transformer layers. Proceedings of the 42nd International Conference on Machine Learning (ICML), 2024.

Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. Minicache: Kv cache compression in depth dimension for large language models. arXiv preprint arXiv:2405.14366, 2024c.

Zayd Muhammad Kawakibi Zuhri, Muhammad Farid Adilazuarda, Ayu Purwarianti, and Alham Fikri Aji. Mlkv: Multi-layer key-value heads for memory efficient transformer decoding. arXiv preprint arXiv:2406.09297, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the AI2 reasoning challenge. CoRR, abs/1803.05457, 2018. URL http://arxiv.org/abs/1803.05457.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7432–7439. AAAI Press, 2020. doi: 10.1609/AAAI.V34I05.6239. URL https://doi.org/10.1609/aaai.v34i05.6239.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Anna Korhonen, David R. Traum, and Lluís Màrquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4791–4800. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1472. URL https: //doi.org/10.18653/v1/p19-1472.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1260. URL https://doi.org/10.18653/v1/d18-1260.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: an adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106, 2021. doi: 10.1145/3474381. URL https://doi.org/10.1145/3474381.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.

Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. Smollm-corpus. 2024. URL https://huggingface.co/datasets/HuggingFaceTB/ smollm-corpus.

Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu: the finest collection of educational content, 2024a. URL https://huggingface.co/datasets/ HuggingFaceFW/fineweb-edu.

Xun Wu, Shaohan Huang, and Furu Wei. Mixture of lora experts. arXiv preprint arXiv:2404.13628, 2024.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder 2 and the stack v2: The next generation, 2024b.

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text, 2023.

Stack Overflow. Stack overflow, 2025. URL https://stackoverflow.com. Accessed: 2025-0521.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíˇcek, Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025a.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíˇcek, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big – data-centric training of a small language model, 2025b. URL https://arxiv.org/abs/2502.02737.

## A Enhanced Expressive Power of MLA with Decoupled RoPE

### A.1 Introduction

This section provides a theoretical analysis to demonstrate that Multi-Head Latent Attention (MLA) with decoupled Rotary Position Embedding (RoPE), as described in Section 3.3 of the main paper, possesses greater expressive power than Grouped-Query Attention (GQA) (Section 3.2). This analysis assumes comparable KV cache sizes and number of query heads.

Our primary argument focuses on the core projection mechanisms that generate queries, keys, and values, abstracting away from the specifics of RoPE application initially. We first present the following proposition concerning the relative expressiveness of these core mechanisms:

- Proposition 1. Given the same KV cache size and number of query heads, the expressiveness of the core attention projection mechanisms follows the order: GQA < MLAFactorized < MQA.

Here, MLAFactorized refers to an attention mechanism employing low-rank factorization for its key and value projections, representing the content-processing aspect of the full MLA. It is im-

portant to note that in the proposition, the query projection in MLAFactorized does not undergo low-rank factorization; this differs from the full MLA, where the query is also factorized. After proving this proposition, we will discuss how the full MLA architecture, which incorporates such an MLAFactorized core for its content components and an MQA core for its decoupled RoPE components, is thereby more expressive than GQA. For this analysis, we primarily consider the impact of the architectural structure on representational capacity, setting aside the direct effects of RoPE itself on the expressiveness comparison between the fundamental GQA, MLA-Factorized, and MQA structures.

Cached During Inference

ℎ 𝑔𝑑

𝑑 𝑔𝑑

𝑑

𝑔

V

···

𝒑𝒓𝒐𝒋

< <

𝑔

K

···

𝑇

𝒑𝒓𝒐𝒋

Q

MQA

MLA

GQA

- Figure 6: Comparison of Multi-Query Attention (MQA), Group Query Attention (GQA), and MultiHead Latent Attention (MLA). In this work, we illustrate that given the same KV cache size, the expressiveness increases in the order of GQA, MLA, and MQA. In the figure, h,d,g denote the number of heads, hidden dimension of each head, and the number of groups (K/V heads) in GQA, respectively. In MQA, the head dimension is set to gd to align the KV cache size with GQA and MLA. As a result, the KV cache size per token per layer for all three approaches is 2gd

### A.2 Proof of Proposition 1

Let D be the hidden dimension of the input token xt ∈ RD, h be the number of query heads, and d = D/h be the dimension per head. In GQA, query heads are divided into g groups. For fair KV

cache comparison, the latent dimension for keys and values in MLAFactorized (rkv) and the head dimension of MQA will be related to gd. Specifically, if the KV cache per token in GQA is 2gd for

both keys and values, then in MLAFactorized, rkv = 2gd, and in MQA, the head dimension is also 2gd; this ensures the KV cache sizes are aligned.

### A.2.1 GQA ≤ MLAFactorized

In GQA, query head qt,i attends to key kj,⌈i/(h/g)⌉ and value vj,⌈i/(h/g)⌉. The GQA key projection WK ∈ Rgd×D produces g distinct key vectors [kt,1;...;kt,g]. Similarly, WV ∈ Rgd×D produces value vectors. We define effective per-query-head projection matrices W′K ∈ Rhd×D and W′V ∈ Rhd×D for GQA:

  

W′K =

  

W′V =

  ,where Wi′K = W⌈Ki/(h/g)⌉, (21)

W1′K . Wh′K

  ,where Wi′V = W⌈Vi/(h/g)⌉. (22)

W1′V . Wh′V

Here, WkK is the k-th d × D block of WK. Thus, k′j,i = Wi′Kxj = kj,⌈i/(h/g)⌉, and similarly for values. The matrices W′K and W′V have ranks at most gd.

An MLAFactorized mechanism generates keys via kj,i = (WUK(WDKV xj))i, where WDKV ∈ Rr

kv×D and WUK ∈ Rhd×r

kv. A similar formulation applies for values with WUV ∈ Rhd×r

kv.

To demonstrate expressive capability, GQA ≤ MLAFactorized, we set rkv = 2gd. Let WDKV = WK WV ∈ R2gd×D. We seek WUK,WUV ∈ Rhd×2gd such that W′K = WUKWDKV ,W′V =

WUV WDKV . This is achieved by setting WiUK,WiUV ∈ Rd×2gd (the block for head i) as selector matrices:

WiUK = [0d×d,...,0d×d

], (23)

,Id×d,0d×d,...,0d×d

k−1 blocks

2g−k blocks

WiUV = [0d×d,...,0d×d

], (24)

,Id×d,0d×d,...,0d×d

g+k−1 blocks

g−k blocks

where k = ⌈i/(h/g)⌉. Thus, GQA’s key/value generation can be replicated by an MLAFactorized model with rkv = 2gd and specific sparse structures for WUK and WUV . The KV cache size 2gd × (sequence length) is preserved since we will be caching cKVj = WDKV xj ∈ R2gd. On that account, the theoretical expressive power of GQA is less than or equal to that of MLAFactorized given the same KV cache size.

### A.2.2 MLAFactorized ≤ MQA

Consider an MLA-Factorized model where queries are qt,i = WiQxt (assuming WiQ ∈ Rd×D is the i-th block of WQ) and keys are kj,i = (WiUK(WDKV xj)). The attention score for head i involves q⊤t,ikj,i:

#### q⊤t,ikj,i = (WiQxt)⊤(WiUK(WDKV xj)). (25)

This can be rewritten as:

q⊤t,ikj,i = ((WiUK)⊤WiQ

#### xt)⊤(WDKV xj). (26)

Wi′Q

Let qˆt,i = Wi′Qxt ∈ R2gd and cKVj = WDKV xj ∈ R2gd. The computation of attention output becomes:

qˆ⊤t,icKVj √

)WiUV cKVj , (27)

softmaxj(

ot,i =

d

j

yt = WO[ot,1;ot,2;...;ot,h]









⊤ t,1cKVj

W1UV

softmaxj(qˆ

d )cKVj

√

W2UV

= WO

. (28)

 

 

 

 

...

. softmaxj(qˆ

⊤ t,1cKVj

d )cKVj

WhUV

√

W′O

This is an MQA formulation where each modified query qˆt,i (now of dimension 2gd) attends to a shared key and value cKVj . This indicates that the computations within MLA-Factorized can be structured to use shared intermediate key and value representations akin to MQA’s core. Thus, any MLA-Factorized model can be represented as an MQA model with a shared key/value of dimension 2gd.

- A.2.3 Strict Inequalities: GQA < MLAFactorized < MQA The relationships are strict:

GQA < MLAFactorized When GQA is represented as an MLAFactorized model, the up-projection ma-

trices WUK and WUV must adopt specific sparse, block-selector structures. A general MLAFactorized model imposes no such constraints; WUK and WUV are typically dense and fully learnable. This

allows a general MLAFactorized to create h distinct key (and value) vectors by combining features from the rkv-dimensional latent space in complex ways. GQA is restricted to g unique key (and value) vectors that are merely replicated h/g times. If h > g, MLAFactorized can generate a richer set of interaction patterns. Thus, MLAFactorized has strictly greater expressive power.

MLAFactorized < MQA Consider the bilinear form x⊤t Mxj in the attention score. In MLAFactorized, for head i, MMLA,i = (WiQ)⊤WiUKWDKV . The maximum rank of the transformation is determined by the smallest one among the ranks of WiQ ∈ Rd×D, WiUK ∈ Rd×2gd, and WDKV ∈ R2gd×D, which is at most d.

However, in the MQA form derived from MLAFactorized, the rank of the interaction matrix here, (Wi′Q)⊤WDKV , is determined by the smallest one among the ranks of Wi′Q ∈ R2gd×D and WDKV ∈ R2gd×D, which is at most 2gd.

Since 2gd ≥ d, MQA allows for a potentially higher-rank interaction between the (modified) query and the shared key representations compared to the per-head effective rank in MLAFactorized’s original formulation. This indicates that MQA has a greater representational capacity for the scoring mechanism.

### A.3 Expressiveness of MLA with Decoupled RoPE

The full MLA architecture, as defined in Section 3.3 (main paper), employs a decoupled RoPE strategy. The query qt,i and key kt,i for head i (in the MHA-like training paradigm, Equation 9) are:

qt,i = [qCt,i;qRt,i] (29) kt,i = [kCt,i;kRt ] (30)

where kRt is a shared RoPE key component across all heads for token t. The bilinear attention score (numerator of the softmax argument) for head i between query at t and key at j is:

(qCt,i)⊤kCj,i + (qRt,i)⊤kRj (31) Let’s analyze the two components of this score:

### 1. Content Component Interaction: (qCt,i)⊤kCj,i. The content keys kCj,i are derived from

WUK(WDKV xj). This key generation mechanism for kCj,i is precisely that of the MLAFactorized model discussed in Section A.1. As established, MLA-Factorized is strictly more expressive than GQA for the non-positional part of the representation.

##### 2. Positional Component Interaction: (qRt,i)⊤kRj . This interaction, where h distinct query-

side RoPE components qRt,i attend to a single, shared key-side RoPE component kRj , is an MQA structure specifically for the positional information. As shown in Section A.2.3, MQA

is strictly more expressive than MLAFactorized, and by extension, GQA.

In summary, we have demonstrated that the expressive power of MLA with decoupled RoPE is stronger than that of the traditional GQA. However, it is worth noting that in the previously proven proposition, the MLAFactorized does not have a low-rank decomposition on the query; this differs from DeepSeek MLA. In the full MLA architecture, the query is also decomposed.

## B Proof of RoPE Inner Product Invariance under Orthogonal Transformation

In this subsection, we provide a rigorous proof of Equation 19, namely:

d/2

l=1

Ulqˆ[2t,il−1::d];Ulqˆt,i[2l::d]

R⊤

Ulkˆ[2j l−1::d];Ulkˆ[2j l::d]

R

⊤

t,i kˆRj .

= qˆR

Here, d is the dimension of each original attention head. The notation qt,i[2l−1::d] (and similarly for other terms) refers to an h-dimensional vector collecting the (2l − 1)-th components from each of the h original attention heads. The matrix Ul is an h × h orthogonal matrix. The superscript R denotes the application of RoPE.

Proof. For the sake of convenience, we omit all i,j,k and let qx,l = qt,i[2l−1::] and qy,l = q[2t,il::]. These are h-dimensional vectors. Similarly, let kx,l = kj[2l−1::] and ky,l = k[2j l::]. The RoPE transformation, as defined by Equations (17) and (18) in the main text, applies as follows for a query vector at position t and key vector at position j within the l-th subspace: For the query vector components:

- (qx,l)R = qx,l cos(tθl) − qy,l sin(tθl)
- (qy,l)R = qx,l sin(tθl) + qy,l cos(tθl)

For the key vector components:

- (kx,l)R = kx,l cos(jθl) − ky,l sin(jθl)
- (ky,l)R = kx,l sin(jθl) + ky,l cos(jθl)

We use the shorthand ct = cos(tθl), st = sin(tθl), cj = cos(jθl), and sj = sin(jθl). The right-hand side (RHS) of Equation (19) is given by the definition of the RoPE inner product:

d/2

qRt,i⊤kRj =

l=1

d/2

=

l=1

(qx,l)R;(qy,l)R ⊤ (kx,l)R;(ky,l)R

((qx,l)R)⊤(kx,l)R + ((qy,l)R)⊤(ky,l)R

Let Sl be the l-th term in this sum:

Sl = (ctqx,l − stqy,l)⊤(cjkx,l − sjky,l) + (stqx,l + ctqy,l)⊤(sjkx,l + cjky,l)

= ctcjq⊤x,lkx,l − ctsjq⊤x,lky,l − stcjq⊤y,lkx,l + stsjq⊤y,lky,l

+ stsjq⊤x,lkx,l + stcjq⊤x,lky,l + ctsjq⊤y,lkx,l + ctcjq⊤y,lky,l

= (ctcj + stsj)(q⊤x,lkx,l + q⊤y,lky,l) + (stcj − ctsj)(q⊤x,lky,l − q⊤y,lkx,l)

= cos((t − j)θl)(q⊤x,lkx,l + q⊤y,lky,l) + sin((t − j)θl)(q⊤x,lky,l − q⊤y,lkx,l).

Now, let’s analyze the left-hand side (LHS) of Equation (19). Let q′x,l = Ulqx,l and q′y,l = Ulqy,l. Similarly, let k′x,l = Ulkx,l and k′y,l = Ulky,l. The l-th term of the LHS sum, denoted Sl′, is:

Sl′ = ((q′x,l)R)⊤(k′x,l)R + ((q′y,l)R)⊤(k′y,l)R . This has the same structure as Sl, just with primed variables:

Sl′ = cos((t − j)θl)((q′x,l)⊤k′x,l + (q′y,l)⊤k′y,l) + sin((t − j)θl)((q′x,l)⊤k′y,l − (q′y,l)⊤k′x,l).

We need to show that the dot product terms involving primed variables are equal to their unprimed counterparts. Consider the first coefficient term:

(q′x,l)⊤k′x,l + (q′y,l)⊤k′y,l = (Ulqx,l)⊤(Ulkx,l) + (Ulqy,l)⊤(Ulky,l)

= q⊤x,lU⊤l Ulkx,l + q⊤y,lU⊤l Ulky,l

= q⊤x,lkx,l + q⊤y,lky,l.

The last equation holds because Ul is an orthogonal matrix. This matches the corresponding term in Sl.

The same applies to the second coefficient term. In this way, we have proven that Sl′ = Sl for each l ∈ {1,...,d/2}. This implies that the LHS of Equation (19) is equal to its RHS:

d/2

R⊤

R

= qRt,i⊤kRj .

Ulq[2t,il−1::];Ulq[2t,il::]

Ulk[2j l−1::];Ulk[2j l::]

l=1

This completes the proof, demonstrating that the orthogonal transformation Ul applied to the hdimensional vectors representing the l-th 2D subspace components across heads preserves the RoPE-based inner product structure.

| |
|---|

In practice, we leverage this rotational invariance property to find a set of optimal orthogonal matrices {Ul} that concentrate the principal components of the key vectors into the first few attention heads. The preceding proof reveals a critical constraint: for the inner product’s value to remain unchanged after transformation, the same orthogonal matrix Ul must be applied to both the real (2l − 1) and imaginary (2l) components of the key vectors within each 2D subspace. This requirement precludes performing separate PCA on the real and imaginary parts. We must therefore find a single rotation that is jointly optimal for both.

Specifically, we formulate this as a joint optimization problem. First, we process a calibration dataset (e.g., Wikitext-2) to collect the key activations at each layer. For each RoPE subspace l ∈ {1,...,d/2}, we obtain two collections of n × h-dimensional matrices (where n denotes the number of samples): the "real" parts {Kx,l}l and the "imaginary" parts {Ky,l}l. To find a single transformation Ul that simultaneously compresses the information from both sets into the first few heads, we proceed as follows.

Let σx,l = K⊤x,lKx,l and σy,l = K⊤y,lKy,l be the h×h covariance matrices of the real and imaginary key components, respectively. Our objective is to find an orthogonal matrix Ul that maximizes the variance—or energy—concentrated in the first m heads after rotation. This corresponds to maximizing the trace of the top-left m×m submatrix of the summed covariance of the rotated vectors. The problem is formally stated as:

max

Ul

Tr (UTl (σx,l + σy,l)Ul):m,:m s.t. UTl Ul = I. (32)

Group 1 Group 2 Group 3 Group 4

dim=[1,3] dim=[2,4] dim=[5,7] dim=[6,8]

PCA

RoPE NoPE NoPE NoPE

- Figure 7: Pipeline of RoRoPE with FreqFold. RoRoPE encodes the entire frequency spectrum of all attention heads in a single latent dimension, which limits its expressive power. FreqFold remedies this by clustering adjacent-frequency dimensions and extracting their principal components jointly,

allocating a higher-dimensional subspace to similar features. This richer representation enables Krope to retain far more positional information.

Here, Ul is the h × h orthogonal optimization variable, and (·):m,:m denotes the top-left m × m submatrix. The solution to this trace maximization problem is obtained by performing an eigende-

composition on the summed covariance matrix σx,l + σy,l. The resulting matrix Ul, whose columns are the eigenvectors sorted in descending order of their corresponding eigenvalues, is the optimal

orthogonal transformation Ul.

By applying this rotation, we ensure that the principal components from both the real and imaginary dimensions of the keys are aligned and concentrated within the first few heads. Consequently, we can discard the RoPE components from the remaining heads in both queries and keys while preserving the most significant positional information, thereby minimizing the performance degradation.

## C FreqFold: Detailed Mechanism, Example, and PCA Efficiency

This appendix provides a detailed explanation of the FreqFold technique, illustrates its operation with a concrete example, and formally connects its benefits to a general principle of Principal Component Analysis (PCA) concerning structured data. This justification clarifies FreqFold’s role in minimizing transformation loss towards decoupled RoPE within the RoRoPE framework (Section 4.2).

### C.1 Detailed Explanation of FreqFold and RoRoPE’s PCA

In the RoRoPE framework, Rotary Position Embedding (RoPE) is applied. RoPE encodes positional information by rotating pairs of feature dimensions. For each RoPE frequency index l ∈ {1,...,d/2}, the corresponding pair of dimensions ([2l − 1 :: d],[2l :: d]) from query and key vectors are rotated. When multiple original attention heads are used (say, g heads), and their key/query projection outputs are concatenated, the RoPE operation for a specific frequency index l applies to a 2g-dimensional vector segment (formed by concatenating the l-th 2D RoPE subspace from each of the g heads).

RoRoPE then applies PCA via matrices {Ul}d/l=12 to these 2g-dimensional segments, independently for each frequency index l.

The core idea of FreqFold is to approximate numerically similar RoPE base frequencies as being effectively identical. For instance, if RoPE uses original base frequencies θl

that are close in value, MD-FreqFold might treat them all as a single, representative frequency θ∗.

,θl

,...,θl

1

2

M

This approximation has a significant implication for how PCA is applied in RoRoPE:

- • Without FreqFold (Standard RoRoPE PCA): For each distinct RoPE frequency index l, a

separate PCA transformation Ul is learned and applied to the corresponding 2g-dimensional key/query segments.

- • With FreqFold: If M original RoPE frequency indices (say l1,...,lM) are grouped together by FreqFold due to their frequency similarity, the M corresponding 2g-dimensional segments are effectively concatenated. Instead of M separate PCAs on 2g-dimensional vectors, a single PCA is performed on the resulting M · 2g-dimensional vectors.

C.1.1 Illustrative Example of FreqFold

Let’s consider a scenario with g = 2 key heads, and each head has dhead = 8 dimensions. Thus, there are d/2 = 8/2 = 4 distinct RoPE frequency indices per head, which we denote as ϕ1,ϕ2,ϕ3,ϕ4. The total number of dimensions is 2 × 8 = 16. The RoPE angles for these 16 dimensions could be conceptualized as follows (repeating for each pair, and across heads):

- • Head 1 (dims 1-8): (ϕ1,ϕ1),(ϕ2,ϕ2),(ϕ3,ϕ3),(ϕ4,ϕ4)
- • Head 2 (dims 9-16): (ϕ1,ϕ1),(ϕ2,ϕ2),(ϕ3,ϕ3),(ϕ4,ϕ4)

- Case 1: RoRoPE without FreqFold For each frequency index ϕl, RoRoPE groups the corresponding dimensions from all g = 2 heads. Each such group forms 2g = 2 × 2 = 4-dimensional vectors (across N samples).

- • Group for ϕ1: Dimensions {1,2} from Head 1 and {9,10} from Head 2. PCA is applied to these N samples of 4D vectors.
- • Group for ϕ2: Dimensions {3,4} from Head 1 and {11,12} from Head 2. PCA is applied to these N samples of 4D vectors.
- • Group for ϕ3: Dimensions {5,6} from Head 1 and {13,14} from Head 2. PCA is applied to these N samples of 4D vectors.
- • Group for ϕ4: Dimensions {7,8} from Head 1 and {15,16} from Head 2. PCA is applied to these N samples of 4D vectors.

Here, RoRoPE performs 4 separate PCA operations.

- Case 2: RoRoPE with 2D-FreqFold 2D-FreqFold implies we are pairing up original frequencies. Suppose FreqFold approximates ϕ1 ≈ ϕ2 (calling this effective frequency ΦA = ϕ1) and ϕ3 ≈ ϕ4 (calling this ΦB = ϕ3).

- • Effective Group for ΦA: This group now includes all dimensions originally associated with ϕ1 OR ϕ2.

- – Original ϕ1-dimensions: {1,2} from Head 1; {9,10} from Head 2. (Forms a 4D

segment Sϕ

1

)

- – Original ϕ2-dimensions: {3,4} from Head 1; {11,12} from Head 2. (Forms a 4D

segment Sϕ

2

) With FreqFold, these segments Sϕ

1

and Sϕ

2

are concatenated. PCA is now applied to the N samples of (4 + 4) = 8-dimensional vectors formed by [Sϕ

1

,Sϕ

2

]. Effectively, dimensions {1,2,3,4} from Head 1 are combined with {9,10,11,12} from Head 2.

- • Effective Group for ΦB: Similarly, this group includes dimensions originally for ϕ3 OR ϕ4.

– Original ϕ3-dimensions: {5,6} from Head 1; {13,14} from Head 2. (Forms Sϕ

)

3

– Original ϕ4-dimensions: {7,8} from Head 1; {15,16} from Head 2. (Forms Sϕ

) PCA is applied to the N samples of 8-dimensional vectors formed by [Sϕ

4

].

,Sϕ

3

4

Here, RoRoPE with FreqFold performs 2 PCA operations, but each operates on larger, 8-dimensional vectors which are concatenations of what were previously separate PCA targets.

### C.2 Formalizing the Benefit of FreqFold in PCA

The example above illustrates that FreqFold causes a re-grouping and concatenation of data segments prior to PCA. The benefit of this concatenation is explained by the following proposition. It states that performing PCA jointly on these concatenated segments (as FreqFold enables) is more effective at preserving variance (and thus minimizing loss) than the alternative of performing separate PCAs on the original, smaller segments and then notionally combining their outcomes.

Consider one such FreqFold merge: suppose M original RoPE frequency indices l1,...,lM are deemed equivalent by FreqFold. Without FreqFold, each lp would correspond to a dataset Xp (e.g., N samples of 2g-dimensional key segments). With FreqFold, these M datasets are concatenated into a single larger dataset Xmerged = [X1,X2,...,XM], and PCA is applied to Xmerged.

Proposition 2. Let M distinct groups of key segments X1,X2,...,XM be identified. Each Xp ∈ RN×d

′

(where p ∈ {1,...,M}) consists of N samples of d′-dimensional vectors. Assume data in each Xp is mean-centered. Let Sp = N1−1XpTXp ∈ Rd

′×d′ be its covariance matrix. FreqFold causes these M groups to be merged for a single PCA operation.

Define V1 = Mp=1 λp,1, where λp,1 is the largest eigenvalue of Sp. This V1 represents the sum of variances if each of the M original groups Xp were individually reduced to its single most dominant dimension.

′) be the dataset formed by concatenating the features (columns) of these M groups. Let Sconcat = N1−1ZTZ ∈ R(M·d

Let Z = [X1,X2,...,XM] ∈ RN×(M·d

′)×(M·d′) be its covariance matrix.

Define V2 = Mj=1 µj, where µ1 ≥ µ2 ≥ ... ≥ µM are the M largest eigenvalues of Sconcat. This V2 represents the variance captured if the concatenated data Z is reduced to M dimensions using PCA.

Then, the variance captured by the joint PCA on the FreqFold-merged data (V2) is greater than or equal to the sum of variances from optimally reducing each original group to one dimension (V1):

V2 ≥ V1

This proposition explains that FreqFold’s strategy of enabling PCA over larger, concatenated segments (formed by merging data from RoPE frequencies deemed similar) is mathematically favored for variance preservation compared to separate, more fragmented PCAs.

### C.3 Proof of Proposition 2

The objective is to prove that V2 ≥ V1, using the notation from Proposition 2. The proof strategy is to construct a specific M-dimensional subspace for the concatenated data Z. We show that the variance

captured by projecting Z onto this particular subspace equals V1. Since the PCA procedure yielding V2 finds the optimal M-dimensional subspace maximizing captured variance, V2 must be at least V1.

′

Let λp,1 be the largest eigenvalue of Sp (covariance of Xp), and wp,1 ∈ Rd

be its corresponding eigenvector. So, Spwp,1 = λp,1wp,1 and wp,T1wp,1 = 1. The variance λp,1 = wp,T1Spwp,1. V1 = Mp=1 λp,1.

For the concatenated data Z, V2 = Mj=1 µj. By Ky Fan’s theorem for matrix eigenvalues:

Tr(UTSconcatU)

V2 = max

U∈R(M·d′)×M UT U=IM

′

. Construct U∗ = [u∗1,...,u∗M] ∈ R(M·d

where U’s columns form an orthonormal basis for an M-dimensional subspace of RM·d

′)×M. For p ∈ {1,...,M}, define u∗p ∈ RM·d

′

:





0d′×1 .

u∗p =

wp,1 (as the p-th block of size d′)

 

 

. 0d′×1

The set {u∗1,...,u∗M} is orthonormal. The variance retained by projecting Z onto the subspace of U∗ is:

M

Tr((U∗)TSconcatU∗) =

(u∗p)TSconcatu∗p

p=1

Let Sqr be the (q,r)-th block of Sconcat, where Sqr = N1−1XqTXr. Note Spp = Sp. Each term (u∗p)TSconcatu∗p = wp,T1Sppwp,1 = wp,T1Spwp,1 = λp,1. So, Tr((U∗)TSconcatU∗) =

M p=1 λp,1 = V1. Since V2 is the maximum possible variance:

V2 ≥ Tr((U∗)TSconcatU∗) = V1 Thus, V2 ≥ V1. This proves Proposition 2.

### C.4 Discussion on the Trade-off in FreqFold

While Proposition 2 demonstrates a clear benefit of FreqFold in terms of PCA efficiency—specifically, that merging M original frequency groups allows for greater variance preservation when reducing to M dimensions—it is crucial to acknowledge an inherent trade-off. The foundational assumption of FreqFold is the approximation of numerically similar RoPE base frequencies as effectively identical. This approximation, by its very nature, introduces a degree of deviation from the original, precise RoPE formulation.

The extent of this deviation, and thus the potential loss in the fidelity of positional encoding, typically correlates with how aggressively frequencies are grouped. A larger M or a looser criterion for similarity when grouping frequencies can amplify this approximation error. Consequently, while increasing the dimensionality of vectors undergoing PCA is beneficial from the perspective of PCA variance capture as shown by the proposition, it may simultaneously increase the lossiness of the RoPE approximation itself. Therefore, the practical application of FreqFold requires a careful balancing act. The parameter M (representing the number of original RoPE frequencies treated as one effective frequency for PCA purposes) or the specific grouping strategy for frequencies must be chosen to optimize this trade-off.

## D Balancing Key-Value Norms and Low-Rank Approximation

This appendix elaborates on the Key-Value (KV) balancing technique and the subsequent joint lowrank approximation applied to the NoPE (No Positional Encoding) components of the keys and the values, as mentioned in Section 4.3 of the main paper. After the RoRoPE procedure (Section 4.2), the key projection matrix WK is effectively split into two components: WRoPEDK ∈ Rd×D corresponding to the single head that retains RoPE, and WNoPEDK ∈ R(g−1)d×D corresponding to the remaining g − 1 head components that do not use RoPE. The value projection matrix is denoted as WDV ∈ Rgd×D.

### D.1 KV Balancing: Purpose and Formulation

Purpose The primary goal of KV balancing is to ensure that the principal component analysis (PCA), when applied jointly to the NoPE key and value activations, is not disproportionately in-

fluenced by components with larger norms. We observed that the activations derived from WNoPEDK (i.e., kNoPE,t = WNoPEDK xt) often have a significantly larger average norm than those from WDV (i.e., vt = WDV xt). Without balancing, PCA would predominantly capture the variance within the NoPE key components, potentially neglecting important variations in the value components.

Formulation To address this imbalance, we introduce a scaling factor α. This factor is computed as the ratio of the expected L2 norms of the NoPE key activations to the value activations, based on a calibration dataset:

Et[∥WNoPEDK xt∥2] Et[∥WDV xt∥2]

(33) where xt ∈ RD is the t-th input token.

α =

While the main paper states scaling WNoPEDK by 1/α and WUK by α for mathematical equivalence in the model’s output, for the purpose of deriving the PCA projection, we effectively use scaled NoPE

key activations. That is, the activations used to compute the PCA basis are k′NoPE,t = 1/α · WNoPEDK xt and vt = WDV xt. This ensures that the PCA process considers features from keys and values on a more equitable footing with respect to their magnitudes. The subsequent low-rank decomposition will then be applied to WNoPEDK and WDV , using the PCA basis derived from these balanced activations.

### D.2 Joint Low-Rank Approximation of NoPE Keys and Values using PCA

After determining the scaling factor α, we proceed to compress the projection matrices associated with the NoPE keys (WNoPEDK ) and all values (WDV ) jointly. The process is as follows:

- 1. Collect Calibrated Activations: A small calibration dataset (WikiText-2) is used. For each input

xt from this dataset, we compute the scaled NoPE key activations k′NoPE,t and the value activations vt. These are concatenated to form combined activation vectors:

cNoPE,t =

k′NoPE,t vt ∈ R(2g−1)d (34)

- 2. Perform PCA: PCA is performed on the set of collected combined activation vectors {cNoPE,t}. This involves computing the covariance matrix of these vectors and finding its principal components. The eigenvectors (corresponding to the largest eigenvalues) are selected to form the columns of a

projection matrix RKV ∈ R((2g−1)d)×r

kv, where rkv is the reduced rank. This matrix RKV captures the directions of highest variance in the (balanced) combined NoPE key and value activation space.

- 3. Low-Rank Decomposition of Projection Matrices: Let WDKV =

WNoPEDK WDV ∈ R((2g−1)d)×D

be the initial projection matrix that transforms the input xt into an intermediate NoPE Key and Value representation cNoPE,t = WDKV xt. Further, let WUKV =

WNoPEUK 0 0 WUV ∈ R2hd×((2g−1)d)

represent the subsequent collective projection matrix that takes cNoPE,t and processes it to produce the actual keys and values required by the attention mechanism for the NoPE components, where

WRoPEUK ∈ Rhd×gd and WNoPEUK ∈ Rhd×(g−1)d are two parts of WUK hat participate in and do not participate in the RoPE computation, respectively. The original sequence of operations for these

components can be expressed as WUKV WDKV xt ∈ R2hd, in which the first hd elements correspond to the keys and the following hd elements correspond to the values.

To introduce a low-rank bottleneck, we modify both WDKV and WUKV using the PCA projection matrix RKV .

- • The initial projection matrix WDKV is transformed into WDKV ′ ∈ Rr

kv×D: WDKV ′ = RKVT WDKV (35)

This new matrix WDKV ′ takes the original input xt and projects it into a compressed rkv-dimensional latent space, which is the actual content stored in the KV cache for the NoPE components.

- • The subsequent projection matrix WUKV is transformed into WUKV ′ ∈ R2hd×r

kv: WUKV ′ = WUKV RKV (36)

This new matrix WUKV ′ now takes the compressed latent representation as input and produces the final representations for the NoPE components that are used in the attention calculation. As we can see, WUKV ′ is actually the concatenated form of WUK and WUV in MLA:

WUK WUV

WUKV ′ =

(37)

This joint decomposition allows for a more holistic compression by identifying shared latent structures between NoPE keys and values, guided by the balanced PCA.

Table 2: Composition of the training dataset. Dataset Sampling Weight

fineweb-edu-dedup 0.70 cosmopedia-v2 0.15 python-edu 0.06 open-web-math 0.08 stackoverflow 0.01

## E Experimental Settings of Fine-tuning

Datasets Following the experimental setups of MHA2MLA, we fine-tune our models using the prtraining corpus from SmolLM Ben Allal et al. [2024]. The dataset comprises FineWeb-Edu-Dedup Lozhkov et al. [2024a], Cosmopedia-v2 — a synthetic dataset generated by Mixtral Wu et al. [2024], Python-Edu from StarCoder Lozhkov et al. [2024b], Open-Web-Math Paster et al. [2023], and data from StackOverflow Stack Overflow [2025]. To ensure a fair comparison with the MHA2MLA baseline, we constructed our training dataset using the same data composition strategy. Specifically, we replicate the dataset mixing ratios used in the MHA2MLA setup to maintain experimental consistency, which is shown in Table 2.

Hyperparameters The fine-tuning hyperparameters for models of all sizes are listed in Table 3. In the table, entries with a slash (/) indicate a two-step training process.

Table 3: Training details across different models. SmolLM 1B7 LLaMA2 7B

-68.75% -87.50% -68.75% -87.50% -92.97%

Batch size 64 64 64 64 / 64 256 / 64 Learning rate 1e-4 1e-4 2e-5 2e-5 / 2e-5 1e-4 / 2e-5 Tokens 300M 1B 500M 2B / 1B 5B / 1B Warmup ratio 0.03 0.08 0 0 / 0.03 0 / 0.03 lr scheduler constant constant constant constant / cosine constant / cosine Sequence length 2048 2048 4096 4096 4096

## F Detail Information for vLLM Benchmark

In Section 5.4, we demonstrated the speedup achieved by TransMLA—which compresses 92.97% of the KV cache—compared to the original LLaMA-2-7B model. This section provides a detailed analysis of throughput across various hardware configurations.

To account for the effects of both the prefilling and decoding stages, we adopt a setting where the input and output lengths are equal. For instance, with a total context length of 1k, we set the input length to 512 tokens and the output length to 512 tokens. Most experiments are conducted using 100 requests to compute the average throughput. However, for shorter context lengths such as 1k, inference is extremely fast, leading to some timing fluctuations. To mitigate this, we increase the number of requests to 1000 for more stable measurements.

While the original LLaMA-2-7B model supports a maximum context length of 4096 tokens, we extend this limit to 32k tokens in our evaluation. Detailed throughput results are presented in Table 4.

On a GPU with 165.2 TFLOPS of compute and 24GB of memory, the LLaMA-2-7B model runs out of memory when the context length reaches 16k tokens. In contrast, TransMLA sustains a throughput of 414.41 tokens per second under the same conditions. On a more powerful GPU with 320 TFLOPS and 64GB of memory, we employ a development version of the vLLM framework. We anticipate that the throughput of TransMLA will improve further with the release of future optimized versions of the framework tailored for this hardware.

- Table 4: Throughput comparison between LLaMA-2-7b and TransMLA at varying input lengths and number of requests.

|Context Length Requests Model|Throughput(output tokens/s) 165.2 TF|24GB 312 TF|40GB 320 TF|64GB<br><br>|
|---|---|
|1K 1000<br><br>LLaMA-2-7b TransMLA<br><br>|653.81 1579.26 1249.13 3043.65 4062.43 1798.17|
|2K 100<br><br>LLaMA-2-7b TransMLA|352.85 850.14 789.31 2241.87 2577.01 1080.73<br><br>|
|4K 100<br><br>LLaMA-2-7b TransMLA<br><br>|173.09 441.37 442.63 1318.78 1926.15 1021.03|
|8K 100<br><br>LLaMA-2-7b TransMLA|85.80 218.51 216.66 832.69 1118.18 870.15<br><br>|
|16K 100<br><br>LLaMA-2-7b TransMLA|OOM 110.58 112.13 414.41 601.36 483.22<br><br>|
|32K 100<br><br>LLaMA-2-7b TransMLA<br><br>|OOM 38.32 55.69 OOM 243.81 278.09|

## G Case Study

To provide an intuitive understanding of TransMLA’s impact on model performance, this section presents several examples from vLLM’s docs. We compare the outputs of three model variants: (1) a model with 92.97% of its KV cache compressed without any fine-tuning; (2) a model pretrained on 6B tokens, as detailed in Table 1; and (3) a model fine-tuned for one epoch on the SmolTalk dataset, following the setup described in Allal et al. [2025a]. The results are summarized in Table 5.

As shown in Table 5, even without any additional training, the compressed model is still able to produce coherent and meaningful responses. This demonstrates the effectiveness of techniques such as RoRoPE, FreqFold, and BKV-PCA in significantly mitigating performance degradation. Moreover, with a modest amount of pretraining or supervised fine-tuning (SFT), the model’s performance improves substantially. These findings highlight TransMLA’s potential as a general framework for converting various GQA models into MLA models, with promising prospects for aligning with the performance of advanced systems like DeepSeek R1.

- Table 5: Examples from different model configurations. Red indicates input; black indicates output. “w/o Training” denotes the TransMLA-compressed model (92.97% KV cache) without further training. “Pre-Training” and “Fine-Tuning” show outputs after pretraining on a 6B-token corpus and SFT on SmolTalk Allal et al. [2025b], respectively.

Model Prompt & Generated Text w/o Training Hello, my name is Katiu, my father’s dog, the pet of the 3600 year-old tribe,

Kint. The Kangs were part of a race of reptiles. A small handful

Pre-Training Hello, my name is Sasha and I am in third grade at Meadows. You may be wondering what this article is about. Well, I have been doing a lot of research on the water cycle and decided to write about it.

Fine-Tuning Hello, my name is Emily, and I’m a 20-year-old college student. My hobbies

include painting, writing, and photography. I also enjoy playing the guitar.

w/o Training The president of the United States is elected by the legislature. The legislature controls the national armed forces, but only provides the funds to establishing a national guard.

Pre-Training The president of the United States is elected to a four-year term by the people of each state in a general election held every four years on the Tuesday following the first Monday in November.

Fine-Tuning The president of the United States is not a position to be taken lightly. This person is the chief executive of the United States of America, and has immense power and influence.

w/o Training The capital of France is Paris. Its geographical position in the Iberian Plain of France, Spain, Spain, and Morocco are the four largest cities. This region is located in Asia, Spain and Morocco.

Pre-Training The capital of France is a major business city and it is a favorite destination for businesses from all over the world. It has a strategic location in the heart of the European Union, which makes it one of the most popular cities in Europe.

Fine-Tuning The capital of France is Paris, and it is one of the most popular tourist destinations in the world. It is a city that offers something for everyone, from art and history to food and fashion.

w/o Training The future of AI is in serious risk to create a major breakthrough in this

emerging phenomenon in the history of artificial intelligence. Pre-Training The future of AI is looking bright. With advancements in technology and the increasing availability of data, AI is expected to become more intelligent and capable of performing even more complex tasks.

Fine-Tuning The future of AI is The future of AI is more nuanced and complex than we might

think. Here are some potential developments that could shape the future of AI.

