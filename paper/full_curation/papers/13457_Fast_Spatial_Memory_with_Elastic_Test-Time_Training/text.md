## Fast Spatial Memory with Elastic Test-Time Training

Ziqiao Ma1,2∗ Xueyang Yu3∗ Haoyu Zhen3 Yuncong Yang3 Joyce Chai2 Chuang Gan1,3 1MIT-IBM Watson AI Lab 2University of Michigan 3University of Massachusetts Amherst https://fast-spatial-memory.github.io/

# arXiv:2604.07350v1[cs.CV]8Apr2026

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

4D-LRM Ground

Truth FSM

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

4D-LVSM

Large Chunk Elastic Test Time Training (LaCET)

###### or

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

FSM

LRM-style Decoder

LVSM-style Decoder

Figure 1. Fast Spatial Memory (FSM) is an efficient, scalable 4D reconstruction model that learns spatiotemporal representations from long sequences to render novel views at novel times. The model is powered by Large Chunk Elastic Test-Time Training (LaCET) blocks and is compatible with a range of rendering decoders, including LRM-style and LVSM-style decoders.

### Abstract

ers high-quality 3D/4D reconstruction with smaller chunks and mitigating the camera-interpolation shortcut. Overall, we hope to advance LaCT beyond the bounded single-chunk setting toward robust multi-chunk adaptation, a necessary step for generalization to genuinely longer sequences, while substantially alleviating the activation-memory bottleneck.

Large Chunk Test-Time Training (LaCT) has shown strong performance on long-context 3D reconstruction, but its fully plastic inference-time updates remain vulnerable to catastrophic forgetting and overfitting. As a result, LaCT is typically instantiated with a single large chunk spanning the full input sequence, falling short of the broader goal of handling arbitrarily long sequences in a single pass. We propose Elastic Test-Time Training inspired by elastic weight consolidation, that stabilizes LaCT fast-weight updates with a Fisher-weighted elastic prior around a maintained anchor state. The anchor evolves as an exponential moving average of past fast weights to balance stability and plasticity. Based on this updated architecture, we introduce Fast Spatial Memory (FSM), an efficient and scalable model for 4D reconstruction that learns spatiotemporal representations from long observation sequences and renders novel view-time combinations. We pre-trained FSM on large-scale curated 3D/4D data to capture the dynamics and semantics of complex spatial environments. Extensive experiments show that FSM supports fast adaptation over long sequences and deliv-

### 1. Introduction

Building a spatial memory would require learning to compress visual observations across viewpoints and time into a unified 4D representation that preserves both spatial structure and temporal dynamics. This capability would advance applications in 4D asset generation [39, 58] for video games, film production, and AR/VR, as well as world modeling [22, 74] for embodied AI and robotics. Especially, reconstructing dynamic scenes from temporally extended and dynamically sampled observations (e.g., long videos captured by moving cameras) remains a central challenge.

Recent advances in Large Reconstruction Models (LRMs) [14, 71] and Large View Synthesis Models (LVSM)[17,23]offerpromisingrendering-basedalternatives to efficient and high-quality 3D/4D reconstruction. Typically built on Transformer-based sequence modeling, these meth-

∗Authors contribute equally to this work.

ods achieve strong reconstruction performance by learning powerful priors over structure and appearance from largescale multi-view data. Despite these advances, these models remain constrained by the amount of activation memory available for a single forward pass, leaving long-context modeling largely unresolved. This is particularly the case in 4D domain, where videos that are temporally extended yet spatially sparsely observed, and their reconstruction quality degrades sharply beyond the training context length, indicating limited temporal scalability [34]. While several 3D reconstruction works have explored hybrid sequence models that combine linear-time state-based mixers with full attention [79, 80], the central question for practical 4D modeling remains open: How can we design a simple, scalable, and efficient spatial memory architecture that learns scene-level spatiotemporal representations from long sequences?

Test-Time Training (TTT) [41, 44] has shown promise in addressing the long-context issue in geometric reconstruction and view synthesis [8, 49, 69]. Especially, Large Chunk Test-Time Training (LaCT) [73] enables in-forward, chunk-wise fast-weight adaptation that lets a transformer recalibrate its internal representations during inference, efficiently updating small parameters from key-value statistics without backpropagation to achieve self-refining, test-time adaptation. Yet, these techniques do not directly generalize to the 4D regime, where scene dynamics evolve across space and time during inference, since the fully plastic nature of continuous LaCT updates leads to uncontrolled fast-weight drift, leading to overfitting in training and unstable updates at test time. This is analogous to catastrophic forgetting at inference time. To address this issue, we introduce Elastic Test-Time Training that executes an additional consolidate operation after LaCT update, inspired by the Elastic Weight Consolidation (EWC) [24] in continual learning. Each fastweight module keeps a reference set of anchor parameters (the values before adaptation) and continuously estimates their importance through an online Fisher-style statistic. During inference, important parameters are softly pulled back toward their anchors, while less critical ones remain free to adjust. This elastic behavior acts as an adaptive spring: it constrains unstable drift without sacrificing responsiveness to new lighting, pose, or scene conditions, transforming the base transformer into a fast, self-refining yet elastic 4D learner, one that keeps adapting to the stream while remembering where it came from. We refer to this new architecture as Large Chunk Elastic Test-Time Training (LaCET).

We scale LaCET up to pretrain a Fast Spatial Memory (FSM) on a curated set of 3D/4D datasets with posed images captured over time and from different cameras. We primarily evaluated FSM on the novel view synthesis (NVS) and demonstrated its competitive performance on a variety of benchmarks and the scalability of LaCET. The model scales effectively with more data and larger model size and general-

izes well to novel scenes. With careful ablation studies, we show that LaCET can effectively mitigate the overfitting and undesirable inference time behaviors of LaCT, e.g., camera interpolation. To our knowledge, FSM is the first large-scale 4D reconstruction model design that supports input from long sequences of views and arbitrary timestamps and renders arbitrary novel view-time combinations. Overall, we hope to advance LaCT beyond the bounded single-chunk setting toward robust multi-chunk adaptation, a necessary step for generalization to genuinely longer sequences, while substantially alleviating the activation-memory bottleneck.

### 2. Algorithmic Preliminaries

#### 2.1. Fast Weights and Test-Time Training

Test-Time Training (TTT) [44] introduces fast weights [41] with rapidly adaptable parameters, which get updated at both training and inference time. This is in sharp contrast to slow weights (conventional model parameters) that remain fixed at inference time. In the context of attention, we consider a sequence of N tokens x = [x1,x2,...,xN], where each tokenxi isprojectedintokeyki, queryqi, andvaluevi vectors. Formally, TTT defines a function fθ(·) parameterized by the fast weights θ, and it involves an update and an apply operation. The (per-token) update operation defines:

θ′ = θ − η ∇θL fθ(ki),vi , (1)

where η represents the learning rate and L(·,·) denotes a loss between the transformed key fθ(ki) and its corresponding value vi, encouraging the network to learn key-value associations. Intuitively, this objective trains the model to compress the ever-growing KV cache (whose memory cost scales linearly with context length) into a fixed-size neural memory, preserving critical key-value associations within a bounded memory budget. The apply operation defines:

zi = fθ′(qi), (2)

where the updated fast weights θ′ are used to compute the output vector zi given the query qi. The per-token TTT layer iteratively performs the update and apply operations on each token xi in sequence.

#### 2.2. Test-Time Training Done Right

Naïve TTT methods often struggle to scale to long contexts, largely due to the low hardware efficiency of their TTT layers, which operate on extremely small mini-batches. To address this, [73] proposed Large-Chunk Test-Time Training (LaCT), a chunk-wise formulation that improves scalability and throughput. The apply operation oi = fθ(qi) follows Eq. (2), where all query vectors qi within a chunk share the same fast weight. Unlike the per-token update in Eq. (1), LaCT aggregates the loss over all keys ki and values vi in a

|Input Plücker|
|---|

Posed Input Image Token

initialize

Window Attn.

[Figure 29]

Target View Query Token

t = 0

###### +

Anchor Weights

Fast Weights

chunk

|Input Plücker|
|---|

Large Chunk Elastic Test Time Training (LaCET)

[Figure 30]

θ* θ K V Q

t = 16

Patchify & Linear

Update Consolidate

|Input Plücker|
|---|

Δθ

[Figure 31]

Apply

θ* θ

t = 32

Predicted Novel View

###### +

|Target Plücker|
|---|
|t = 24|

[Figure 32]

Feed-Forward

+

xN

- Figure 2. (Left) Overview of FSM. The model takes a sequence of posed images captured at different times and learns to infer novel view-time combinations. Camera information is converted into Plücker ray maps as geometric augmentation for visual tokens. The model directly predict the target view with decoders. (Right) The LaCET Block. It maintains two sets of parameters, anchor weights and fast weights. During adaptation, the fast weights are updated using information from the current chunk (queries, keys, and values), while the anchor weights act as a stable reference. The model tracks parameter importance online and softly restores critical weights toward their anchors to prevent drift. This stabilizes rapid updates while preserving the adaptability of TTT, addressing the plasticity issue. chunk and computes a single surrogate update for chunk c:

propose Elastic Test-Time Training, which enhances the LaCT update operator with an Elastic Weight Consolidation (EWC) [24] regularizer, introducing a soft stability prior over fast-weight dynamics. We refer to our algorithm as LargeChunk Elastic Test-Time Training (LaCET, to distinguish from LaCT), combining its scalability, efficiency, and elastic stability for robust long sequence modeling.

b

. (3)

θc+1 = θc − ∇θ

ηi(xi)L fθ(ki),vi

i=1

θ=θc per-chunk surrogate pseudo-gradient

Here, b denotes the chunk size and ηi is the (learnable) per-token learning rate. Intuitively, this objective strengthens the association between each key and its corresponding value by updating the fast weights so that fθ(ki) becomes more consistent with vi under the training loss. In practice, LaCT regularizes the updated fast weights using L2 weight normalization [40] along the input dimension and optionally applies the Muon-style Newton-Schulz iteration [19, 32], without weight decay. Because each chunk aggregates thousands of tokens, updates occur infrequently, enabling richer update-rule designs while amortizing computational cost.

Elastic Weight Consolidation. Kirkpatrick et al. [24] introduces a quadratic penalty that discourages important parameters from drifting too far from a reference set of anchor weights, originally designed for a classic continual learning setting where a model learn a new task TB without forgetting a previously learned task TA. All knowledge about TA is captured in the posterior distribution p(θ |DA). Since this posterior is intractable for large neural networks, EWC approximates itusingaGaussiancentered at the previously optimized parameters θA⋆ with a diagonal precision given by the Fisher Information Matrix F, i.e., p(θ |DA) ≈ N θA⋆ ,F−1 . The Fisher Information has three desirable properties: (i) it corresponds to the local curvature of the loss near θA⋆ , (ii) it can be estimated from first-order gradients alone, and (iii) it is guaranteed to be positive semi-definite. The overall objective when learning TB becomes a combination of the new-task loss and a quadratic penalty at θA⋆ :

#### 2.3. Test-Time Training Done Better

While LaCT significantly improves the scalability of TTT by amortizing adaptation across large chunks, its updates remain fully plastic, as the fast weights in each chunk drift freely in parameter space at inference time. In the novel view synthesis task, LaCT works the best with one single chunk. In long and dynamic 4D scenes, where illumination, pose, or motion continuously evolve during inference, such unconstrained plasticity can cause cumulative instability, leading to temporal ghosting artifacts. To address this, we

λ 2

Fi θi − θA,i⋆ 2, (4)

L(θ) = LB(θ) +

i

where LB(θ) is the loss for the new task TB, λ controls

the relative importance of retaining old knowledge, and i indexes each model parameter. Intuitively, parameters with high Fisher values Fi are crucial for TA and are therefore strongly constrained to remain near θA⋆ , whereas parameters with small Fi can adapt freely to TB.

Elastic Test-Time Training. In our formulation, we reinterpret this idea at the time of the test: each incoming chunk of data acts as a new task TB, and the fast-weight state of the previous chunk plays the role of θA⋆ . The Fisher-weighted penalty in Eq. (4) thus serves as a continuously updated elastic prior, stabilizing the model’s adaptation over time (e.g., foreground dynamics) while preserving useful past information (e.g., static background). The EWC penalty defines an elastic prior after the LaCT update in Eq. (3), which we refer to as the consolidate operator. Formally, let θc′ denote the intermediate fast weights after the update but before elastic consolidation in chunk c, and θc⋆ their corresponding anchor parameters (the reference state before adaptation or at the last re-anchor).

(5)

θc+1 = θc′ − λFc ⊙ θc′ − θc⋆ ,

elastic consolidation

where Fc is a per-parameter Fisher-style importance estimate, ⊙ denotes the Hadamard (elementwise) product, and λ is a constant controlling the strength of the elastic prior.

Importance Estimates. We maintain the importance matrix Fc as an EMA with decay α ∈ [0,1) over chunk index c:

Fc+1 = α Fc + (1 − α)φ Sc , (6)

where α ∈ [0,1) is the decay factor. The statistic Sc depends on the chosen estimator. Besides EWC [24], we also consider tworelatedalternativesmotivatedbymemory-awaresynapses (MAS) [1] and synaptic intelligence (SI) [67]. Concretely,

θc′ − θc, (MAS / EWC) (θc′ − θc) ⊙ (θc′ − θc⋆), (SI)

Sc =

φ(Sc) = |Sc|, (MAS / SI)

Sc2, (EWC)

with all operations applied elementwise. When Sc has a leading batch dimension, we average over that dimension before applying Eq. (6). Intuitively, the MAS-like variant tracks the magnitude of the chunkwise update, the EWC-like variant emphasizes parameters that consistently receive large squared updates, and the SI-like variant additionally weights the update by its drift from the current anchor. In our setting, since the anchor-relative displacement is itself induced by the chunkwise update, the SI-like statistic tends to behave similarly to a rescaled squared-update estimator.

Anchor Update Policies. We consider different anchoring policies that control how θ⋆ is maintained:

- • Global: anchors remain fixed to initialization.
- • Streaming: anchors update at each chunk boundary, ensuring local temporal continuity.
- • Streaming-EMA: anchors update via an exponential moving average[47], θ⋆ ← βθ⋆+(1−β)θ, forming alow-pass filter over the fast-weight trajectory.

We will show later that Streaming-EMA is the best practice for genuinely elastic memory behaviors.

### 3. Fast Spatial Memory (FSM)

FSMadoptsanend-to-endfeedforwardnetworktolearnscene representations, trained using only photometric supervision. Input images are patchified and augmented with temporal and camera information to form visual tokens, which are then processed by the sequence model. We consider two decoding variants: (i) direct RGB patch prediction with a lightweight linear head, in the spirit of LVSMs [17, 23]; and (ii) prediction of pixel-aligned Gaussian Splatting primitives followed by rasterization into target views, in the spirit of GS-LRMs [34, 49, 71].

#### 3.1. Model Architecture

Image Tokenization. As shown in Figure 3, the input consists of V posed images from arbitrary view-time combinations, denoted as {Ij ∈ RH×W×3}Vj=1, together with their camera intrinsics and extrinsics. Here, H and W denote the image height and width, respectively. We convert the provided camera parameters into canonical Plücker ray maps [37], represented as [rd, ro × rd], where rd and ro denote the ray direction and origin, respectively. Following 4D-LRM [34], temporal conditioning is encoded using a timestamp map {Tj ∈ RH×W×1}Vj=1, which records the normalized time of each frame. For input j, We concatenate the timestamp Tj, RGB image Ij, and Plücker ray map Pj along the channel dimension to form a per-view feature map Ij = Concat(Ij, Pj, Tj) ∈ RH×W×10, which provides per-pixel spatial and temporal embeddings to distinguish both frame time and camera view. Each Ij is partitioned into non-overlapping patches of size p × p. Every patch is flattened into a vector of length 10p2 and linearly projected to a D-dimensional token embedding.

LaCET Backbone. We adopt SwiGLU-MLP [43] without bias terms as the fast weight network in Eq. (3), consisting of three parameter matrices θ = {θ1,θ2,θ3}. The network and its loss is:

L fθ(ki),vi = −fθ(ki)⊤vi

= −[θ2(SiLU(θ1ki) ◦ (θ3ki))]⊤vi, (7)

where ◦ denotes elementwise multiplication. We emphasize that only the input-view tokens are passed through the KV projections to generate gradients for the update operation.

Target Queries 4DGS (Vtar, 6+1)

[Figure 33]

[Figure 34]

[Figure 35]

sample & render

(Vtar, H, W, 3)

(Vin*H*W, 20)

(Vtar, H, W, 3)

4D GS Decoder (Linear)

Image Decoder (Linear+Sigmoid+Unpatchify)

(Vin*NP*NP, D) (Vin*NP*NP, D)

(Vin*NP*NP, D) (Vtar*NP*NP, D)

LaCET Blocks (xN)

LaCET Blocks (xN)

(Vin*NP*NP, D) (Vin*NP*NP, D) Image Tokenizer (Patchify+Linear)

(Vin*NP*NP, D) (Vtar*NP*NP, D) Image Tokenizer (Patchify+Linear)

| | |
|---|---|
|Time + Plücker|[Figure 36]|

| | |
|---|---|
|Time + Plücker|zeros|

|Time + Plücker|[Figure 37]|
|---|---|

|Time + Plücker|[Figure 38]|
|---|---|

Input Views Target Queries

Input Views Virtual Views

copy

(Vin, H, W, 3+6+1) (Vtar, H, W, 3+6+1)

(Vin, H, W, 3+6+1) (Vin, H, W, 3+6+1)

(a) FSM-LVSM Overview.

(b) FSM-LRM Overview.

- Figure 3. FSM-LVSM and FSM-LRM architectural designs. (a) LVSM-style rendering predicts target image patches directly from query tokens and does not build an explicit scene representation. (b) LRM-style rendering first predicts an explicit 4D scene representation with Gaussian primitives and then renders target views from that representation.

This design ensures that the target-view tokens do not interact with one another, allowing each novel view to be synthesized independently and efficiently. In contrast, allowing target tokens to interact across views would correspond to a form of dynamic evaluation [25] or few-shot in-context learning [48], which introduces additional information leakage and renders the comparison unfair.

LRM-style rendering (Figure 3b), we adopt an explicit 4D representation, e.g., 4DGS [64] similar to 4D-LRM [34]. To adapt the sequence model for explicit GS modeling, we follow tttLRM [49] to query the fast weights for a set of virtual view planes for 4DGS and use the input views as virtual views. We adopt pixel-aligned Gaussian rendering, leading to V × H × W Gaussian primitives, each parameterized by g ∈ R20. We split it into (gxyz ∈ R3,gt ∈ R,grgb ∈ R3,gscale,xyz ∈ R3,gscale,t ∈ R,grotation,left ∈ R4,grotation,right ∈ R4,gopacity ∈ R). We mostly followed the parameterization of 4D-LRM except we set the permissible depth interval δnear = 0.01 and δfar = 100 for scene-level reconstruction. We adopt tilebased rasterization with deferred backpropagation during rendering to reduce GPU memory consumption [70].

LVSM-Style Rendering. In the LVSM-style variant (Figure 3a), the model does not rely on an explicit scene representation. For each target view-time query, we construct an empty image-token map whose appearance channels are set to zero, while its camera and temporal channels are populated with the target metadata. These query tokens are concatenated with the input tokens and processed jointly by the model. We then use a lightweight image-token decoder to reconstruct RGB patches from the output token embeddings. Concretely, each token is first passed through layer normalization, then projected linearly from the token dimension to 3p2. The resulting vector is interpreted as the flattened RGB values of the reconstructed patch. The resulting vector is interpreted as the flattened RGB values of the reconstructed patch, followed by a sigmoid activation to bound predictions to [0,1] in normalized pixel space.

#### 3.2. Training Objectives

To train the model, we render U target views for supervision and minimize the image reconstruction loss. Let {I∗i′ | i′ = 1,2,...,U} denote the ground truth views and { I∗i′} the corresponding rendered images. The photometric training loss combines ℓ2 (MSE) loss and LPIPS (w/

(Alternatively) LRM-Style Rendering. Following an

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Ground Truth LaCET (4 chunks) Streaming-EMA

LaCET (4 chunks) LaCT (1 chunk) LaCT (4 chunks) Streaming

LaCET (4 chunks) Global

- Figure 4. Qualitative illustration of the ablation studies, obtained after the same training steps (16K) with the same training and inference random seed on the same Stereo4D test set example.

Train Test Test Anchor Fisher Train Test #Chunk #Chunk Batch Size Update Estimate ℓ2 Loss (×103)↓ PSNR↑ LPIPS↓ SSIM↑

EWC

- ✗ 1 1 1 - - 1.80 26.021 0.1179 0.792
- ✗ 4 4 1 - - 2.04 26.908 0.0988 0.814

✓ 4 4 1 streaming-ema SI 2.36 29.989 0.0517 0.903 ✓ 4 4 1 streaming-ema EWC 2.36 29.781 0.0537 0.897 ✓ 4 4 1 streaming-ema MAS 2.28 29.922 0.0519 0.899 ✓ 4 4 1 streaming MAS 1.71 26.960 0.0966 0.817 ✓ 4 4 1 global MAS 3.00 28.347 0.0653 0.863 ✓ 1 1 1 global∗ MAS 1.73 26.965 0.0960 0.817 ✓ 1 4 1 streaming-ema MAS 1.73 21.993 0.3429 0.650 ✓ 4 4 16 streaming-ema MAS 2.28 29.928 0.0519 0.898

* The choice of anchor update policy makes no difference when chunk size is set to full sequence.

- Table 1. Ablation Studies. The training ℓ2 loss is reported from the exponential moving average (EMA) model (α = 0.1) to ensure robustness against noise. When the number of chunks is 1, it corresponds to the original full-sequence setup in LaCT. With 4 chunks, each chunk contains 2048 input tokens. We find that EWC effectively mitigates the overfitting issue observed in LaCT due to full plasticity. The streaming-ema anchor update policy proves critical for achieving stable performance.

Dataset Source Dyn. #Frames #Scenes Ratio RealEstate10K [77] Real ✗ 10M 80K 1 DL3DV [30] Real ✗ 51M 10K 1 PointOdyssey [75] Syn. ✓ 6K 131 200 Spring [35] Syn. ✓ 200K 37 500 Multi-Cam Video [2] Syn. ✓ 11M 13.6K 1 DynamicReplica [20] Real ✓ 145K 484 100 Stereo4D [18] Real ✓ 15M 80K 1

- Table 2. Summary of datasets. Source indicates whether the dataset is captured from the real world or synthesized. Dynamic specifies whether the scenes are dynamic. #Frames and #Scenes denote the total number of image frames and unique scenes, respectively. Ratio represents the per-scene sampling multiplier used during training for data balancing. VGGNet) loss [72]:

L =

1 U

U

i′=1

ℓ2( I∗i′,I∗i′) + µ · LPIPS( I∗i′,I∗i′) , (8)

where µ controls the weight of the LPIPS loss and is set to 0.5 empirically.

- 3.3. Pretraining Dataset

PointOdyssey [75], Spring [35], DynamicReplica [20], MultiCam Video [2], and Stereo4D [18]. Due to the limited availability of 4D data, we retain several static datasets and assign timestamps according to the natural camera trajectory. For other synthetic datasets, frame timestamps are randomly assigned to each view. All datasets are rescaled to maintain a consistent metric scale across sources. Data pre-processing details are in Appendix A.1.

### 4. Ablation: When and Why Elasticity Helps

Before scaling up the full pretraining pipeline, we perform controlled ablation studies with FSM-LVSM at a moderate scale. These experiments investigate the key algorithmic components added on top of the vanilla LaCT block, including the effects of chunking, anchor update policies, and Fisher estimation. For this purpose, we start by training the model exclusively on internet stereo videos from Stereo4D [18], trimmed to a maximum temporal window of 136 frames. All ablation models use a 12-layer LaCET backbone, trained with a per-GPU batch size of 16 on 8 H100 GPUs, using 32 input and 32 target views, a maximum temporal span of 128 frames, and an image resolution of 128 × 128 for 32K steps (≈32B tokens). We deliberately use these smaller networks so that its long-context performance saturates with a reasonably small number of tokens. We evaluate on the Stereo4D

A summary of the datasets used for pretraining is provided in Table 2, including RealEstate10K [77], DL3DV [30],

LaCT (1 chunk) LaCT (4 chunks) LaCET (4 chunks) LaCT (1 chunk) LaCT (4 chunks) LaCET (4 chunks) Sparse View Sparse View Sparse View Continuous View Continuous View Continuous View

0.9

0.4

| |
|---|

| |
|---|

| |
|---|

| |
|---|

30

| |
|---|

0.8

0.3

PSNR

LPIPS

| |
|---|

SSIM

| |
|---|

25

0.7

| |
|---|

0.2

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.6

0.1

| |
|---|

20

0.5

| |
|---|

| |
|---|

8/204816/409632/819264/16384128/32768

8/204816/409632/819264/16384128/32768

8/204816/409632/819264/16384128/32768

#Images / #Tokens

#Images / #Tokens

#Images / #Tokens

(a) PSNR vs. #input imgs / #tokens.

(b) SSIM vs. #input imgs / #tokens.

(c) LPIPS vs. #input imgs / #tokens.

- Figure 5. Test-time scaling curves. Shown are PSNR/SSIM/LPIPS of LaCT (1/4 chunks) and LaCET (4 chunks; streaming-ema), trained with 32 images (vertical line) and evaluated with varying numbers of input images. Each point uses a 136-frame Stereo4D clip. For sparse views, input and target frames are randomly sampled across the long full span. For continuous views, we select a contiguous sub-sequence (e.g., 40 frames for 32-in/8-out) and randomly mask the target frames inside it for the model to predict, reducing to frame interpolation.

#### 4.2. Elasticity Improves Generalization

test set using PSNR [7], SSIM [56], and LPIPS [72], using 32 randomly sampled views along the trajectory as inputs and averaged over 8 randomly sampled target views per scene. The results over different settings are aggregated in Table 1. More details are available in Appendix A.3.

As shown in Table 1, we observe a clear gap between training and test PSNR, i.e., training vs. test ℓ2, which points to substantial overfitting. This generalization gap is reduced by consolidation, suggesting that consolidation improves information transfer across chunks while also suppressing fast-weight drift caused by repeated fully plastic inferencetime updates. We hypothesize that LaCT-LVSM tends to exploit local pattern shortcuts, effectively memorizing localized cues within its limited fast-weight memory instead of maintaining a more distributed spatiotemporal representation, consistent with similar findings in other efficient architectures [66]. We next provide a deeper analysis of what LaCT-LVSM overfits to in practice.

#### 4.1. Anchor Update Policies

We analyze how elastic consolidation behaves under different chunking and anchoring configurations.

Full-sequence setup (single chunk). When the chunk size equals the full sequence length, the model performs exactly one forward pass and one fast-weight update per scene. All anchor update policies become equivalent. The consolidation term scales with both the update magnitude and the anchorrelative drift, and in the single-chunk regime reduces to a second-order correction O(λ(∆θ)2) in the update size, which is negligible for small λ.

Setups. Figure 5 examines how LaCT and LaCET behave under different test-time input densities. Both models are trained with 32 input images, and we vary the number of input frames at inference on 136-frame Stereo4D clips. In the discrete-view setting, input and target frames are uniformly sampled across the full span. In the continuous-view setting, we crop a contiguous sub-sequence (e.g., 40 frames for the 32-in/8-out case) and mask the target frames within that window, reducing the problem to frame interpolation. Two settings converge when the full 136-frame span is used.

Global anchoring. If the anchor weights remain fixed globally, consolidation degenerates into an importance-weighted ℓ2 regularizer. This stabilizes inference-time adaptation, but does not encode temporal continuity beyond the fixed prior, similar to weight decay.

Streaming anchoring. Under streaming (w/o EMA) update, the anchor is reset to the current fast weights at the beginning of each chunk. The consolidation term then only regularizes within-chunk drift, applying adaptive shrinkage to the accumulated fast-weight change. This configuration lacks memory consolidation across chunks, making it more prone to overfitting.

LaCET consistently dominates LaCT under sparse inputs. When input views are sparse in time and space, the advantages of LaCET are large and systematic across all PSNR/SSIM/LPIPS metrics. Both LaCET (4 chunks) and LaCT (4 chunks) degrade sharply as sparsity increases, while LaCT (1 chunk) collapses gracefully, as more activation memory is used to process the full sequence (which is not sustainable for longer sequences). Nevertheless, smaller chunks remain appealing due to their reduced activation memory footprint, since backpropagation spans fewer samples, making them more suitable for scaling and for real streaming applications.

Streaming-EMA anchoring. The non-trivial, genuinely elastic behavior emerges when streaming anchors are combined with EMA updates. The consolidation term acts as a low-pass, importance-weighted constraint on the fast-weight trajectory, penalizing cumulative drift relative to an dynamically evolving consolidated anchor weight rather than the instantaneous update.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

GroundMoVieS

Truth FSM

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

4DGT4D-LVSM

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Figure 6. Qualitative comparison on Steoro4D test set. Note that for MoVieS we use a higher default resolution (504 × 504).

Stereo4D [18] NVIDIA [65] Resolution PSNR↑ LPIPS↓ SSIM↑ Resolution PSNR↑ LPIPS↓ SSIM↑

Model

Optimization-based SoM [53] —— OOT⋆ —— 379 × 672 15.30 0.509 0.317 MoSca [26] —— OOT⋆ —— 379 × 672 21.45 0.265 0.712

Rendering-based L4GM [39] —— OOT† —— 256 × 256 10.07 0.587 0.235 4DGT [60] 504 × 504 24.62 0.102 0.785 504 × 504 14.13 0.640 0.131 MoVieS [29] 504 × 504 27.19 0.114 0.888 379 × 672 19.16 0.315 0.514 FSM-LRM 256 × 256 27.29 0.147 0.876 256 × 256 20.17 0.337 0.567 FSM-LVSM 256 × 256 32.16 0.043 0.931 256 × 256 23.90 0.105 0.747

⋆ SoM takes around 10min per scene and MoSca takes around 45min per scene. † L4GM requires multi-view diffusion as prior.

- Table 3. 4D NVS Results. Metrics are resolution-dependent (e.g., higher resolutions typically produce higher PSNR). We adopt the lowest resolution for meaningful comparison with baselines. Steoro4D test set contains 7109 scenes, which is out of time (OOT) for some methods.

LaCET mitigates camera-pose interpolation shortcuts. In the continuous-view regime, LaCET (4 chunks) begins to outperform both LaCT (1 chunk) but still outperforms LaCT (4 chunks). This behavior reveals that LaCT learns to exploit short-range temporal redundancy rather than learning a true view-conditioned spatial representation. When input frames are continuous, the task effectively degenerates into a frame interpolation problem. The model can simply latch onto neighboring frames in the context window and does not need to perform genuine NVS for 4D representation, i.e., no camera-pose extrapolation or long-range temporal modeling. [36] made similar observations. LaCET still improves with more continuous inputs, but the gap between discrete-view and continuous-view performance is substantially smaller. This indicates that LaCET is less prone to collapsing into an

interpolation-only solution and instead preserves the ability to model long-range 4D dynamics.

### 5. Scaling LaCET for Fast Spatial Memory

#### 5.1. Pretraining Curriculum

Based on the controlled studies described above, we default LaCET blocks to (i) the streaming-EMA anchor update policy and (ii) the SI-style importance estimate for empirically better training stability. We train both the FSM-LVSM and FSM-LRM variants. Given compute limitations, we bootstrap the LVSM variant from a DL3DV-pretrained LaCT backbone with a resolution of 128, introduce additional temporal encodings, and continue pretraining it for poseconditioned 4D reconstruction. For data scheduling, we

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

GSLRM LVSM RayZer FSM LVSM Ground Truth

Figure 7. Qualitative comparison on DL3DV benchmark.

4D Novel View Synthesis. Unlike 3D NVS, there is currently no well-established benchmark for feedforward 4D evaluation. Existing datasets were originally designed for optimization-based pipelines, and the community has not yet converged on a standard evaluation protocol. We use the NVIDIA [65] benchmarks (with the same evaluation setup in [29]) and Steoro4D [18] benchmarks for fair comparison within this regime. In Table 3, we show our method outperforms existing approaches evaluated at similar resolutions. In particular, on Stereo4D our model achieves clear improvements over prior rendering-based methods across all metrics. On the NVIDIA benchmark, our method achieves the best performance among feed-forward approaches at 256 × 256 resolution, and approaches the performance of the strongest optimization-based methods, which require perscene test-time optimization. These results suggest that the proposed LaCET effectively benefits dynamic scene modeling, where maintaining consistent spatial information across time becomes critical.

DL3DV [30] Resolution PSNR↑ LPIPS↓ SSIM↑

Model

Static Models DepthSplat [59] 512 × 448 17.81 0.356 0.596 GS-LRM [71] 256 × 256 23.02 0.266 0.705 LVSM [17] 256 × 256 23.10 0.257 0.703 RayZer† [16] 256 × 256 23.72 0.222 0.733 LongLRM [80] 540 × 960 24.10 0.254 0.783 tttLRM [49] 540 × 960 25.07 0.215 0.822 tttLVSM [73] 540 × 960 26.90 0.185 0.837 FSM-LRM 256 × 256 23.59 0.206 0.766 FSM-LVSM 256 × 256 26.69 0.091 0.846

Dynamic Models FSM-LRM 256 × 256 21.89 0.314 0.692 FSM-LVSM 256 × 256 24.61 0.118 0.787

† RayZer ignores input poses and uses target reference images instead, placing it somewhere between pose-conditioned and fully pose-free approaches.

- Table 4. 3D NVS Results. Metrics are resolution-dependent (e.g., higher resolutions typically produce higher PSNR). We adopt the lowest resolution for meaningful comparison with baselines. employ a long-context curriculum that gradually increases the input resolution (128 → 256), the temporal span (128

→ 256), and dynamic number of input views as training progresses. Complete implementation details are available in Appendix A.4.

- 5.2. Novel View Synthesis Performance

3D Novel View Synthesis. We use the DL3DV-140 benchmark [30] for evaluation. Since evaluation metrics scale with resolution, we adopt the minimal 256 resolution to ensure fair comparison across both categories. In Table 4, we show our method delivers performance comparable to existing approaches evaluated at similar resolutions, demonstrating that the proposed LaCET blocks preserve strong capability on static scenes where spatial memory is less critical.

For fair comparisons, we report the highest score among (i) our reproduced results, (ii) reported by the authors, and (iii) those reported by the community. Note that metrics like PSNR are resolution-dependent (e.g., higher resolutions typically produce higher PSNR). We adopt the lowest resolution (256×256) for meaningful comparison with baselines.

### 6. Related Work

Fast Weights and Test-Time Training (TTT). Recently, many sequence models have been reformulated under the lens of inference-time learning or regression, which interprets the recurrent update of model states as a form of

online learning [31] from context [3, 11, 48]. This view commonly connects modern sequence models to the longstanding notion of fast weights [42], i.e., parameters that evolve in-context at each timestep to capture short-term associations. Fast-weight mechanisms thus act as associative memories [6, 38], balancing retention and adaptation through architectures such as DeltaNet [41, 63]. Recently, Test-Time Training (TTT) extends fast-weight adaptation to general neural components that update online using self-supervised signals [44, 51]. Recent works explore specialized test-time optimizers [5, 21] and online learning objectives [4], with applications in video generation, 3D reconstruction, and beyond [8, 10, 73]. However, naïve TTT remains bottlenecked by poor hardware utilization, limited state capacity, and unstable long-horizon dynamics [45]. Large-Chunk Test-Time Training (LaCT) improves this paradigm by enabling efficient in-forward fast-weight updates over larger contexts [33, 73]. Still, LaCT relies on fully plastic fast-weight dynamics, which can lead to overfitting and catastrophic forgetting over long sequences. This work addresses this issue with Elastic TTT, which stabilizes fast-weight adaptation by introducing additional elasticity across chunks.

Large Rendering-Based Reconstruction Models. Large Reconstruction Models (LRMs) have recently emerged as a unified framework for producing view-consistent 3D reconstructions. Trained on massive 3D and 4D datasets, these models leverage triplane-based NeRFs [14, 15, 27, 52] or Gaussian Splatting [49, 57, 71, 79, 80] to encode strong priors over shape and appearance, achieving high-quality reconstruction from only a few posed views. In the 4D setting, similarly, existing LRMs still rely heavily on geometric supervision to maintain rendering consistency, typically requiring posed inputs together with explicit Gaussian primitives [28, 29, 34, 39, 60, 61]. More recently, Large View Synthesis Models (LVSMs) have begun to relax these geometric constraints, achieving high-quality view synthesis without explicit geometric representations [17, 23, 73] and, in some cases, supporting self-supervised autoencoding reconstruction[9,16,36]. Ourworkfollowsthisdirectionbydeveloping a fast 4D reconstruction model that learns scene-level spatiotemporal representations, and by instantiating it both with and without minimal geometric priors. A parallel line of research explores feed-forward, geometry-centric reconstruction models [46, 50, 55, 62, 69] through large-scale training. These methods have inspired several 4D counterparts that estimate dynamic geometry or camera poses without supporting novel view-time synthesis [8, 12, 54, 68, 76, 78]. This work departs from explicit geometric reconstruction and instead treats novel view-time synthesis as the core objective of 4D representation learning, following prior work [23, 33, 73] that has used this task as the primary task for training, evaluation, and scaling-law studies of model architecture.

### 7. Conclusion and Limitations

ScalingtoLongerSequences. LaCETenablesfastinferencetime adaptation for high-quality rendering from, in principle, arbitrarily long sequences in a single forward pass, where activation memory is no longer the bottleneck. However, due to limitations in licensable training data and suitable benchmarks, as well as our compute budget, we focus in this work on architectural advances rather than training and scaling a model that fully realizes the method’s potential.

Pose Estimation in Dynamic Scenes. Recently, several works have explored 3D reconstruction from unposed images [16, 36, 52]. However, jointly estimating camera intrinsics and poses in dynamic scenes, where both camera motion and scene dynamics are present remains challenging. In this work, we assume posed input images and do not treat unposed reconstruction as a primary target.

Geometrically Faithful 4D Reconstruction. While NVS is a key task for spatial intelligence, solving it does not by itself ensuregeometricfaithfulnessortemporallyconsistentmotion. Accurate 4D geometry requires additional constraints and evaluation protocols beyond view synthesis quality. There is ongoing debate in the community over whether explicit geometric supervision is necessary, or whether rendering-based supervision alone is sufficient for learning geometrically faithful representations. In this work, we deliberately focus on the architectural aspects of this problem. While LaCET reduces the tendency of the model to interpolate nearby context frames instead of performing true NVS, this behavior does not fully disappear under rendering-only supervision. We expect that incorporating additional geometric supervision, e.g., depth, correspondence, multi-view consistency, or motion cues such as optical flow, could further mitigate this issue, and we leave this direction to future work.

Acknowledgment. The authors would like to thank Zefan Cai, Xuweiyi Chen, Yinpei Dai, Yilun Du, Chenguo Lin, Freda Shi, Hao Tan, Zeyuan Yang, and Tianyuan Zhang for their insightful discussions.

### References

- [1] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In European conference on computer vision (ECCV), pages 139–154, 2018. 4
- [2] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In International Conference on Computer Vision, 2025. 6
- [3] Ali Behrouz, Zeman Li, Praneeth Kacham, Majid Daliri, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, and Vahab

- Mirrokni. Atlas: Learning to optimally memorize the context at test time. arXiv preprint arXiv:2505.23735, 2025. 10
- [4] Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. It’s all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. arXiv preprint arXiv:2504.13173, 2025. 10
- [5] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. In Conference on Neural Information Processing Systems, 2025. 10
- [6] Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint. In Conference on Neural Information Processing Systems, pages 1560–1588, 2023. 10
- [7] Luen C Chan and Peter Whiteman. Hardware-constrained hybridcodingofvideoimagery. IEEETransactionsonAerospace and Electronic Systems, (1):71–84, 1983. 7
- [8] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. In International Conference on Learning Representations, 2026. 2, 10
- [9] Xuweiyi Chen, Wentao Zhou, and Zezhou Cheng. Wildrayzer: Self-supervised large view synthesis in dynamic environments. In Conference on Computer Vision and Pattern Recognition,

2026. 10

- [10] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Conference on Computer Vision and Pattern Recognition, pages 17702–17711, 2025. 10
- [11] Benoit Dherin, Michael Munn, Hanna Mazzawi, Michael Wunder, and Javier Gonzalvo. Learning without training: The implicit dynamics of in-context learning. arXiv preprint arXiv:2507.16003, 2025. 10
- [12] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J Black, Trevor Darrell, and Angjoo Kanazawa. St4rtrack: Simultaneous 4d reconstruction and tracking in the world. In International Conference on Computer Vision, pages 8503–8513, 2025. 10
- [13] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4246–4253, 2020. 14
- [14] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In International Conference on Learning Representations,

2024. 1, 10

- [15] Hanwen Jiang, Qixing Huang, and Georgios Pavlakos. Real3d: Scaling up large reconstruction models with real-world images. In International Conference on Computer Vision, pages 5821– 5833, 2025. 10
- [16] Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, et al. Rayzer: A self-supervised large view synthesis model. In International Conference on Computer Vision, 2025. 9, 10
- [17] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang

- Xu. LVSM: A large view synthesis model with minimal 3d inductive bias. In International Conference on Learning Representations, 2025. 1, 4, 9, 10
- [18] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how things move in 3d from internet stereo videos. In Conference on Computer Vision and Pattern Recognition, pages 10497– 10509, 2025. 6, 8, 9, 14, 15
- [19] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. https: //kellerjordan.github.io/posts/muon, 2024. 3
- [20] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Conference on Computer Vision and Pattern Recognition, pages 13229–13239, 2023. 6
- [21] Mahdi Karami and Vahab Mirrokni. Lattice: Learning to efficiently compress the memory. arXiv preprint arXiv:2504.05646, 2025. 10
- [22] Justin Kerr, Chung Min Kim, Mingxuan Wu, Brent Yi, Qianqian Wang, Ken Goldberg, and Angjoo Kanazawa. Robot see robot do: Imitating articulated object manipulation with monocular 4d reconstruction. In Conference on Robot Learning, 2024. 1
- [23] Evan Kim, Hyunwoo Ryu, Thomas W Mitchel, and Vincent Sitzmann. Scaling view synthesis transformers. arXiv preprint arXiv:2602.21341, 2026. 1, 4, 10
- [24] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13): 3521–3526, 2017. 2, 3, 4
- [25] Ben Krause, Emmanuel Kahembwe, Iain Murray, and Steve Renals. Dynamic evaluation of neural sequence models. In International Conference on Machine Learning, pages 2766– 2775, 2018. 5, 14
- [26] Jiahui Lei, Yijia Weng, Adam W Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In Conference on Computer Vision and Pattern Recognition, pages 6165–6177,

2025. 8

- [27] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In International Conference on Learning Representations, 2024. 10
- [28] Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschenski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, and Jiahui Huang. Feed-forward bullet-time reconstruction of dynamic scenes from monocular videos. In Conference on Neural Information Processing Systems, 2025. 10
- [29] Chenguo Lin, Yuchen Lin, Panwang Pan, Yifan Yu, Tao Hu, Honglei Yan, Katerina Fragkiadaki, and Yadong Mu. Movies:

- Motion-aware 4d dynamic view synthesis in one second. In Conference on Computer Vision and Pattern Recognition, 2026. 8, 9, 10
- [30] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learningbased 3d vision. In Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 6, 9, 15
- [31] Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, and qiang liu. Longhorn: State space models are amortized online learners. In International Conference on Learning Representations, 2025. 10
- [32] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025. 3
- [33] Junchen Liu, Sven Elflein, Or Litany, Zan Gojcic, and Ruilong Li. Test-time training with kv binding is secretly linear attention. arXiv preprint arXiv:2602.21204, 2026. 10
- [34] Ziqiao Ma, Xuweiyi Chen, Shoubin Yu, Sai Bi, Kai Zhang, Chen Ziwen, Sihan Xu, Jianing Yang, Zexiang Xu, Kalyan Sunkavalli, et al. 4d-lrm: Large space-time reconstruction model from and to any view at any time. In Conference on Neural Information Processing Systems, 2025. 2, 4, 5, 10, 15
- [35] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn. Spring: A high-resolution highdetail dataset and benchmark for scene flow, optical flow and stereo. In Conference on Computer Vision and Pattern Recognition, pages 4981–4991, 2023. 6
- [36] Thomas Mitchel, Hyunwoo Ryu, and Vincent Sitzmann. True self-supervised novel view synthesis is transferable. In International Conference on Learning Representations, 2026. 8, 10
- [37] Julius Plücker. Xvii. on a new geometry of space. Philosophical Transactions of the Royal Society of London, (155): 725–791, 1865. 4
- [38] Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber, Markus Holzleitner, Thomas Adler, David Kreil, Michael K Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. Hopfield networks is all you need. In International Conference on Learning Representations, 2021. 10
- [39] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. In Conference on Neural Information Processing Systems, pages 56828–56858, 2024. 1, 8, 10
- [40] Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Conference on Neural Information Processing Systems, 2016. 3
- [41] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366,

2021. 2, 10

- [42] Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131–139, 1992. 10

- [43] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 4
- [44] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. In International Conference on Machine Learning, pages 57503–57522, 2025. 2, 10
- [45] ArnuvTandon, KaranDalal, XinhaoLi, DanielKoceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, et al. End-to-end test-time training for long context. arXiv preprint arXiv:2512.23675,

2025. 10

- [46] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. Mvdust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. In Conference on Computer Vision and Pattern Recognition, 2024. 10
- [47] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Conference on Neural Information Processing Systems, 2017. 4
- [48] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pages 35151–35174, 2023. 5, 10
- [49] Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu, Kalyan Sunkavalli, Sai Bi, Lingjie Liu, and Yiwei Hu. tttlrm: Test-time training for long context and autoregressive 3d reconstruction. In Conference on Computer Vision and Pattern Recognition, 2026. 2, 4, 5, 9, 10, 15
- [50] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Conference on Computer Vision and Pattern Recognition, pages 5294–5306,

2025. 10

- [51] Ke Alexander Wang, Jiaxin Shi, and Emily B Fox. Test-time regression: a unifying framework for designing sequence modelswithassociativememory. arXivpreprintarXiv:2501.12352,

2025. 10

- [52] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan, Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai Zhang. Pflrm: Pose-free large reconstruction model for joint pose and shape prediction. In International Conference on Learning Representations, 2024. 10
- [53] Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. In International Conference on Computer Vision, pages 9660–9672, 2025. 8
- [54] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Conference on Computer Vision and Pattern Recognition, pages 10510–10522, 2025. 10
- [55] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024. 10

- [56] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 7
- [57] Desai Xie, Sai Bi, Zhixin Shu, Kai Zhang, Zexiang Xu, Yi Zhou, Soren Pirk, Arie Kaufman, Xin Sun, and Hao Tan. Lrmzero: Training large reconstruction models with synthesized data. InConferenceonNeuralInformationProcessingSystems,

2024. 10

- [58] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. SV4d: Dynamic 3d content generation with multi-frame and multi-view consistency. In International Conference on Learning Representations, 2025. 1
- [59] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Conference on Computer Vision and Pattern Recognition, pages 16453– 16463, 2025. 9
- [60] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard Newcombe, and Zhaoyang Lv. 4dgt: Learning a 4d gaussian transformer using real-world monocular videos. In Conference on Neural Information Processing Systems, 2025. 8, 10
- [61] Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Apoorva Sharma, Maximilian Igl, Peter Karkus, DanfeiXu, etal. Storm: Spatio-temporalreconstructionmodel for large-scale outdoor scenes. In International Conference on Learning Representations, 2025. 10
- [62] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Conference on Computer Vision and Pattern Recognition, 2025. 10
- [63] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In Conference on Neural Information Processing Systems, pages 115491–115522, 2024. 10
- [64] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In International Conference on Learning Representations, 2024. 5, 15
- [65] Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera. In Conference on Computer Vision and Pattern Recognition, pages 5336–5345, 2020. 8, 9
- [66] Wangjie You, Zecheng Tang, Juntao Li, Lili Yao, and Min Zhang. Revealing and mitigating the local pattern shortcuts of mamba. In Findings of the Association for Computational Linguistics: ACL 2025, pages 12156–12178, 2025. 7
- [67] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pages 3987–3995, 2017. 4
- [68] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. In International Conference on Learning Representations, 2025. 10

- [69] Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, and Deqing Sun. Loger: Long-context geometric reconstruction with hybrid memory. arXiv preprint arXiv:2603.03269, 2026. 2, 10
- [70] Kai Zhang, Nick Kolkin, Sai Bi, Fujun Luan, Zexiang Xu, Eli Shechtman, and Noah Snavely. Arf: Artistic radiance fields. In European Conference on Computer Vision, pages 717–733,

2022. 5

- [71] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pages 1–19, 2024. 1, 4, 9, 10, 15
- [72] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6, 7
- [73] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. In International Conference on Learning Representations, 2026. 2, 9, 10
- [74] Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, and Chuang Gan. Learning 4d embodied world models. In International Conference on Computer Vision, pages 5337–5347, 2025. 1
- [75] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In International Conference on Computer Vision, pages 19855–19865, 2023. 6
- [76] Kaichen Zhou, Yuhan Wang, Grace Chen, Gaspard Beaudouin, Fangneng Zhan, Paul Pu Liang, and Mengyu Wang. Page-4d: Disentangled pose and geometry estimation for 4d perception. In International Conference on Learning Representations,

2026. 10

- [77] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Transactions on Graphics, 37

(4):1–12, 2018. 6

- [78] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025. 10
- [79] Chen Ziwen, Hao Tan, Peng Wang, Zexiang Xu, and Li Fuxin. Long-lrm++: Preserving fine details in feed-forward widecoverage reconstruction. arXiv preprint arXiv:2512.10267,

2025. 2, 10

- [80] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. In International Conference on Computer Vision, pages 4349– 4359, 2025. 2, 9, 10

### A. Implementation and Training Details

#### A.1. Data Pre-processing

For each training sample, we load a video clip together with per-frame camera metadata, including intrinsics and world-to-camera poses. We first sample a temporal window from the full clip, then randomly select input and target frames within that window. For each selected frame, we extract the RGB image from the video, convert the stored world-to-camera matrix to camera-to-world form, and collect the corresponding intrinsic parameters. The image is resized and cropped to the target resolution, while the intrinsics are updated accordingly. All images are converted to RGB and normalized to tensors. The frame timestamp is taken from the frame index, then normalized within the sampled clip segment with linear rescale. This preserves relative temporal ordering while keeping timestamps in a fixed range across videos of different lengths. We further normalize camera poses at the scene level by centering them with respect to the mean pose.

#### A.2. Algorithm and Model Architecture

Fortheelastictest-timetrainingalgorithm, weuseαewc = 0.5, βewc = 0.5 and λewc = 0.5 after grid search. Each block uses a model dimension of 768 and the fast-weight module is implemented as a single-head SwiGLU MLP with a hidden dimension of 1536. The window attention module contains 12 heads with a head dimension of 64 and applies QKNorm [13]. The feed-forward network uses an intermediate hidden dimension of 3072. Both the tokenization and decoder layer are linear projections, with a sigmoid applied at the decoder. During both training and inference, the update operation is applied to all input tokens, and the fast weights are subsequently used to process the target tokens. All model variants inthispaperusethesame LaCET block configuration and update rule.

#### A.3. Ablation Study Settings

For the ablation study in Sec. 4, we adopt a controlled configuration with 12 LaCET blocks.

Data usage. We conducted all experiments on Stereo4D [18], a large dataset containing diverse camera trajectories and both static and dynamic object motion, which makes it well suited for modeling 4D scenes. We followed its official train-test splits.

Training details. For ablation study, we train with with 32 input views and 32 novel views at 128 × 128 resolution for 32K steps. During training, we first sample a window of 128 consecutive frames, then randomly select 64 frames, from which 32 are used as input and the remaining 32 as target views. The detailed training configuration is provided in

- Table 5. All experiments are trained on 8 H100 GPUs.

Config Ablation Base Resolution Multi-Length Parameters Training Training Scaling Fine-tuning

#layers 12 24 24 24 #input frames 32 32 32 12-64 #target frames 32 32 32 32

resolution 128 128 256 256 temporal window 128 128 256 256

optimizer Adam Adam Adam Adam

- beta 1 0.9 0.9 0.9 0.9
- beta 2 0.95 0.95 0.95 0.95

weight decay 0.05 0.05 0.05 0.05 learning rate 2e-4 1e-4 5e-5 1e-4 lambda L2 1.0 1.0 1.0 1.0

lambda LPIPS 0.5 0.5 0.5 0.5 batch size per gpu 16 16 4 4

#gpus 8 64 64 64 L2 warmup 1000 2500 500 0

warmup steps 1000 2500 1000 0 total steps 32000 80000 20000 20000

Table 5. Summary of configurations across ablation studies, base training, resolution scaling, and variable-length fine-tuneing.

- A.4. Full-Scale Pre-training Settings

Data usage. To scale up the model capacity, we train the complete FSM model on a large collection of both synthetic and real data in Table 2.

Training details. We first pre-train our model at 128 × 128 resolution for 80K steps, and then fine-tune it at 256 × 256 resolution for an additional 10k steps. All training configurations use 32 context frames and 32 target frames, sampled from a window of 128 consecutive frames. Detailed training settings are provided in Table 5. Both training stages are done with 64 H100 GPUs.

- B. Addendum to Results and Discussions

- B.1. Batch Inference

Unlike standard inference, LaCET modifies the model state during inference through fast-weight updates. When the inference batch size is greater than 1, updates from all examples in the batch are averaged (or accumulated) and applied once per chunk. Consequently, batch size directly affects the adaptation dynamics rather than merely the throughput, which is a distinctive property of test-time-training architectures that makes batched inference behave similarly to dynamic evaluation [25] or few-shot adaptation. Empirically, we found the effect to be minimal (Table 1); nevertheless, we fix the inference batch size to 1 in all subsequent experiments.

- B.2. LVSM-style Decoder vs. LRM-style Decoder

We provide additional side-by-side ablations comparing LVSM-style vs. LRM-style decoders.

LVSM-style decoder. In a typical LVSM-style, no explicit scene representation is used in modeling. We use a shallow image-token decoder to reconstruct pixel patches from token embeddings. Specifically, for each token, we first apply

DL3DV [30] Stereo4D [18] Res. PSNR↑ LPIPS↓ SSIM↑ Res. PSNR↑ LPIPS↓ SSIM↑

Model

FSM-LRM 128 × 128 20.99 0.243 0.683 128 × 128 28.19 0.097 0.897 FSM-LVSM 128 × 128 21.25 0.169 0.655 128 × 128 31.06 0.041 0.931 FSM-LVSM (w/ RoPE) 128 × 128 20.75 0.237 0.680 128 × 128 30.54 0.059 0.922

- Table 6. Additional ablation study results on (i) side-by-side comparison of LVSM-style decoder vs. LRM-style decoder and (ii) using explicit temporal channel vs. using RoPE.

layer normalization, followed by a linear projection from the token dimension to 3p2, where p denotes the patch size. The resulting vector is interpreted as the flattened RGB values of the reconstructed patch. A sigmoid activation is applied at the output to bound predictions to ([0,1]), matching normalized pixel space.

LRM-style decoder. With explicit 4D representation, e.g., 4DGS [64], we implement a model following 4D-LRM [34] and tttLRM [49]. To adapt large-chunk TTT for explicit GS modeling, we query the fast weights for a set of virtual view planes for 4DGS and used the input views as the virtual views. We adopt pixel-aligned Gaussian rendering, giving V × H × W Gaussians, each with dim4DGS = 20. From each decoded 4D Gaussian parameter g ∈ R20, we split the 4-channel space-time vector (gx,gy,gz,gt), retain the time µt = gt, and normalize the xyz features to a scalar distance δ. We strictly followed the tile-based rasterization pipeline introduced in 4D-LRM with deferred backpropagation during rendering to reduce GPU memory consumption. Following the setup in [71], we set δnear = 0 and δfar = 400.

Results. We find that monocular video training leads to substantially less overfitting to camera interpolation, although convergence becomes markedly slower. With the same number of training steps as in Table 6, LVSM-style decoding performs better than explicit 4DGS modeling. We hypothesize that, while explicit scene representations may offer stronger generalization and robustness, they are also considerably harder to optimize and more computationally expensive.

#### B.3. Explicit Temporal Encoding vs. RoPE

Timestamp maps as time conditioning. Following 4DLRM [34], we represent temporal conditioning with a timestamp map that stores the normalized time of each frame. For each view, we concatenate this timestamp map with the RGB image and the Plücker ray map along the channel dimension to form a 10-channel feature map. This per-pixel representation encodes both spatial and temporal cues, enabling the model to distinguish not only between camera views but also between different points in time.

RoPE-style time conditioning. As an alternative to explicit temporal conditioning, we encode frame time directly in the

latent tokens using rotary positional embeddings (RoPE). Each frame is assigned a normalized timestamp, which determines a sinusoidal rotation applied to the first few channels of every token from that frame. Since all tokens within a view share the same temporal rotation, the encoding captures frame identity at the view level without entangling time with local spatial layout. This provides a parameter-free and computationally efficient alternative to explicit temporal conditioning.

Results. We find that using RoPE leads to slower convergence. With the same number of training steps as in Table 6, explicit temporal encoding performs better than RoPE. We hypothesize that explicit time conditioning provides a stronger and more direct optimization signal, whereas RoPE injects temporal information more implicitly through feature-space rotations, making it harder for the model to learn to use temporal cues efficiently under a limited training budget.

- B.4. Addition Qualitative Results We provide additional results in Figures 11, 9 and 12.
- B.5. Failure Cases and Analysis

Figure 10 illustrates a typical failure case. Under large camera or view interpolation, the model may fail to update subject motion consistently, instead preserving stale gestures or partial motion patterns from neighboring frames. The results also exhibit ghosting artifacts, with residual duplicated structures around moving limbs and bodies. This suggests that the model still struggles to maintain accurate space-time correspondence and motion consistency when extrapolating across more challenging viewpoints.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

GroundMoVieS

Truth FSM

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

4DGT4D-LVSM

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Figure 8. Additional comparison on Steoro4D test set. Note that for MoVieS we use a higher default resolution (504 × 504).

###### PSNR

30.36

30.31 29.80 30.22 31.69 30.10

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

4D-LVSM

Truth FSM

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Ground

###### PSNR

28.46

27.41 26.03 30.36 30.83

29.16

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

4D-LVSM

FSM

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Ground

Truth

Figure 9. Qualitative examples on Steoro4D test set.

###### PSNR

29.47

25.88 23.79 21.35 19.64 21.83

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

4D-LVSM

Truth FSM

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Ground

Figure 10. Qualitative failure example.

###### PSNR

26.84

24.51 24.17 23.69 23.74 23.45

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

4D-LVSM

Truth FSM

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

4D-LVSM Ground

###### PSNR

28.10

22.56 22.27 22.59 23.46 24.30

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

FSM

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Ground

Truth

Figure 11. Qualitative results on NVIDIA benchmark.

###### PSNR

28.25

27.33 30.09 28.68 32.12 30.97

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

3D-LVSM

Truth FSM

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Ground

###### PSNR

29.57

33.31 31.30 28.69 34.54 31.85

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

3D-LVSM

FSM

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Ground

Truth

Figure 12. Qualitative results on DL3DV-140 benchmark.

