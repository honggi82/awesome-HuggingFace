# ViT-Up: Faithful Feature Upsampling for Vision Transformers

Krispin Wandel, Jingchuan Wang, and Hesheng Wang

## arXiv:2606.14024v1[cs.CV]12Jun2026

Abstract—Vision Transformers (ViTs) have become a dominant architecture for visual representation learning, providing exceptionally strong and broadly reusable backbone features. However, ViTs are commonly operated on relatively small patchtoken grids due to the quadratic cost of global self-attention, which creates a persistent bottleneck for dense prediction tasks such as semantic segmentation and depth estimation. This has motivated the development of task-agnostic feature upsamplers. While recent state-of-the-art methods produce visually sharp dense representations, their reliance on shallow image encoders for guided upsampling can introduce feature leakage, fragmentation, and blur. We introduce ViT-Up, an implicit feature upsampling framework that replaces external image guidance with layer-wise query construction from intermediate ViT hidden states. This enables feature prediction at arbitrary continuous image coordinates while preserving alignment with the backbone feature space. Experiments demonstrate that ViT-Up consistently outperforms state-of-the-art image-guided upsamplers across dense prediction and semantic correspondence. On DINOv3-S+, ViT-Up improves over prior methods by up to +2.07 mIoU on Cityscapes and +4.17 PCK@0.10 on SPair-71k. With the larger DINOv3-B backbone, these gains increase to +3.36 mIoU and +8.09 PCK@0.10, demonstrating that ViT-Up scales favorably with backbone capacity.

Index Terms—Feature Upsampling, Image Representation Learning, Vision Transformers

I. INTRODUCTION

Vision Transformers (ViTs) [1] have established themselves as the dominant architecture for visual representation learning, providing exceptionally strong and broadly reusable semantic features [2]–[4]. By representing images as sequences of patch tokens, ViTs model long-range visual interactions through global self-attention, whose cost scales quadratically with the number of tokens. As a result, foundation vision encoders are commonly evaluated at image resolutions that yield only coarse patch-token grids, for example 14×14 or 28×28 tokens. While these low-resolution feature maps provide powerful image representations, they introduce a critical bottleneck for dense prediction tasks, such as semantic segmentation [5], [6] and monocular depth estimation [7], which inherently require precise, pixel-level spatial reasoning.

A straightforward way to address this resolution mismatch is to evaluate the backbone at a higher input resolution, disregarding the associated computational cost. However, this can move the foundation backbone out-of-distribution and degrade feature quality [8]. Alternatively, dense prediction systems often use specialized decoders [9]–[11]. However, these decoders require task-specific training and additional

K. Wandel, J. Wang, and H. Wang are with Shanghai Jiao Tong University, Shanghai, China. Corresponding author: Hesheng Wang.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Fig. 1. (A) Overview of ViT-Up. ViT-Up treats feature upsampling as coordinate-conditioned feature prediction: given an input image and an arbitrary continuous query coordinate, it predicts the corresponding ViT feature from low-resolution backbone hidden states, e.g., the 28×28 patchtoken grid produced by DINOv3-S+ for a 448×448 input. This enables feature upsampling by independently evaluating the implicit decoder over a dense query grid, yielding high-resolution feature maps at arbitrary output resolutions H×W. (B) The resulting dense feature maps can be used for downstream dense prediction tasks such as semantic segmentation and depth estimation.

computation, undermining one of the main advantages of foundation models: fast and efficient adaptation to downstream tasks. Consequently, recent work has introduced task-agnostic feature upsamplers, such as JAFAR [8], AnyUp [12], UPLiFT [13], and NAF [14], to bridge the gap between coarse backbone features and dense prediction.

A highly prevalent strategy in these state-of-the-art architectures is to rely on high-resolution image guidance. By employing a separate, lightweight image encoder to condition the upsampling process, these methods can produce dense, visually sharp feature maps. However, we find that this visual sharpness often masks severe underlying feature leakage, where features from visually similar but semantically distinct regions are mixed. Disentangling a compressed patchtoken grid without mixing adjacent concepts requires the high-

resolution guidance signal to possess semantic understanding comparable to the foundation backbone itself. Because the dedicated image encoders used in prior work are extremely shallow, they lack this semantic capacity, making imageguided upsampling highly susceptible to feature leakage.

While prior works have pursued backbone-agnostic upsampling as a primary design goal, we purposefully center our approach on the hierarchical structure of Vision Transformers (ViTs) [1]. Since ViTs have become the dominant architecture for representation learning, this specialization entails only a modest trade-off in generality, yet unlocks largely untapped potential by leveraging the backbone’s intrinsic hierarchical structure. In a ViT, the feature hierarchy is distinct: shallow layers retain high-resolution spatial and structural evidence, while deeper layers consolidate this into increasingly abstract, global semantic structures. Because these rich, multi-scale representations are already computed by the backbone, relying on an auxiliary, shallow image encoder for guidance is not only computationally redundant but semantically suboptimal. Instead, we can extract this necessary signal directly from the backbone’s internal layers, enabling an upsampling process that is natively aligned with the model’s own learned representation.

Building on this principle, we introduce ViT-Up, a coordinate-conditioned implicit feature decoder for Vision Transformers, illustrated in Fig. 1. ViT-Up constructs dense query features at arbitrary continuous image coordinates by progressively integrating hidden states from the ViT hierarchy, preserving alignment with the backbone feature space while producing features beyond the native token grid.

We demonstrate that this approach significantly outperforms image-guided methods on standard dense linear probing tasks. On the DINOv3-S+ backbone, ViT-Up yields improvements of +2.07 mIoU on Cityscapes and a +0.55 increase in δ1 on COCO depth [5]–[7] over the prior state-of-the-art methods NAF [14] and UPLiFT [13], nearly doubling the performance gains relative to naive bilinear interpolation. Beyond dense prediction, we additionally evaluate ViT-Up on semantic correspondence, which is a critical measure of feature faithfulness because it directly exposes structural weaknesses such as feature blur, fragmentation, and drift [4], [15]. In contrast to prior methods, which yield only marginal gains over bilinearly upsampled low-resolution features, ViT-Up substantially improves semantic correspondence on SPair-71k [16], increasing PCK@0.10 by +4.17 points. Furthermore, ViT-Up scales effectively with backbone capacity: while backbone-agnostic methods are bottlenecked by the limited capacity of their fixed, shallow image encoders, ViT-Up exploits the increased semantic depth of larger backbones, further widening the performance gap to +3.36 mIoU on Cityscapes and +8.09 PCK@0.1 on SPair-71k when scaled to DINOv3-B.

Our contributions are three-fold:

• We introduce ViT-Up, a task-agnostic implicit feature upsampling framework that predicts backbone-aligned dense feature maps at arbitrary continuous image coordinates. By constructing dense query features layer by layer from intermediate ViT representations, ViTUp preserves the backbone’s feature-space structure and

mitigates common upsampling artifacts such as leakage, fragmentation, and blur.

- • We propose a multi-scale feature supervision strategy for training implicit feature upsamplers. Multi-scale teacher features extracted from the training image supervise a student that receives a downscaled version of the same image padded to a fixed input resolution and is queried densely over the visible image region. This forces the student to recover fine spatial detail while remaining consistent with the backbone feature space across scales.
- • We demonstrate that ViT-Up consistently outperforms prior state-of-the-art feature upsamplers across dense prediction and semantic correspondence. Moreover, ViT-Up scales effectively with backbone capacity by leveraging richer intermediate representations, unlike image-guided upsamplers whose external guidance signal is largely decoupled from the ViT backbone.

Code, pretrained models, and evaluation scripts are available at https://github.com/krispinwandel/vit-up.

II. RELATED WORK A. Task-Dependent Feature Upsamplers for Dense Prediction

Dense prediction tasks such as semantic segmentation and monocular depth estimation require spatially detailed feature maps for producing pixel-level predictions. This is true for classical CNN-based models [9], [17], [18] as well as transformer-based dense prediction systems [10], [11], [19], [20]. Since modern backbones often produce features at a lower spatial resolution than the desired output, dense prediction architectures require mechanisms for transforming coarse features into spatially dense representations. For CNNs and hierarchical vision transformers, Feature Pyramid Network (FPN)-style decoders [21]–[23] exploit the native feature hierarchy by upsampling coarse features, typically with bilinear interpolation or learned transposed convolutions, and fusing them with finer backbone features at matching resolutions. Plain ViT backbones, however, do not provide such a spatial hierarchy. Dense prediction systems using these backbones therefore require additional decoder mechanisms to recover spatial resolution: DPT [10] reassembles intermediate ViT hidden states into multi-scale decoder features, while ViTDet [24] builds an FPN-like pyramid from a single-scale ViT feature map. In both cases, the required resolution changes are still implemented using standard operations such as bilinear interpolation or learned transposed convolutions. Dynamic upsampling aims to improve on these standard upsampling operators by predicting input-dependent reassembly weights or sampling locations for local feature aggregation.

CARAFE [25], [26] replaces fixed bilinear interpolation with adaptive feature reassembly, using a shared kernelprediction module to generate spatially varying kernels conditioned on the local low-resolution feature content. DySample [27] replaces dynamic kernels with predicted sampling offsets for grid-based bilinear feature interpolation. Other methods additionally use high-resolution encoder information. FADE [28] upsamples a low-resolution decoder feature map using kernels generated from both the decoder feature and the

corresponding high-resolution encoder skip feature. SAPA [29] computes similarities between each high-resolution encoder feature point and a local neighborhood of low-resolution decoder features, and uses these similarities as upsampling weights.

While replacing standard bilinear interpolation or transposed convolutions with dynamic upsampling operators can improve dense prediction performance, these methods are still learned inside task-specific encoder–decoder or featurepyramid architectures, where upsampling is optimized as one component of a downstream dense prediction model.

This raises the question we target in foundation-model feature upsampling: can feature upsampling be learned once in a task-agnostic way? If a pretrained vision backbone already provides strong semantic features, then a reusable upsampler could densify these features before downstream training. This would make dense adaptation more efficient: instead of training a full high-resolution encoder–decoder model for every task, one could train or finetune lightweight downstream heads on dense foundation-model features with less computation and less task-specific data.

B. Image Super-Resolution and Local Implicit Functions

A natural source of inspiration is image super-resolution. If low-resolution RGB images can be upsampled to arbitrary resolutions, can low-resolution latent feature maps be upsampled in a similar task-agnostic manner? Local implicit functions [30] provide an elegant formulation for this question by replacing fixed-grid prediction [31] with continuous coordinate-based decoding, enabling the same representation to be evaluated at arbitrary resolutions and sampled at arbitrary locations or densities.

LIIF [30] encodes the low-resolution input image into a low-resolution grid of latent features, and then predicts the RGB value at arbitrary image coordinates from nearby latent features and the corresponding relative offsets to the queried coordinate. The success of this formulation relies on a locality assumption: nearby latent features, and in particular the closest latent cell, contain the information needed to reconstruct the queried high-resolution RGB value.

This assumption is reasonable for image super-resolution because the target signal is a three-channel RGB value and the encoder is trained specifically for local photometric reconstruction. Even though each latent cell supports many subpixel details, it only needs to provide enough information for the decoder to predict local color values. This assumption may break in feature upsampling because the target changes from the three-channel RGB domain to a high-dimensional semantic feature space. A low-resolution patch token creates an information bottleneck: it may not have enough capacity to preserve the high-dimensional semantics of all fine-grained details inside its corresponding image patch. As a result, semantic information for high-frequency image detail may be suppressed or mixed before the upsampler is even applied. Thus, the nearest patch token remains a useful local anchor, but it is not sufficient as the sole source of sub-token semantics.

C. Implicit Feature Upsampling

The optimization-based variant of FeatUp [32] adapts coordinate-based decoding to vision foundation model features by fitting an implicit neural feature field [33], [34] to each input image. It achieves this by iteratively optimizing an MLP through multi-view consistency, leveraging extensive augmentations of the input image. While this approach produces highquality dense features because the representation is adapted directly to the specific input, the optimization must be repeated for every image. This makes the process prohibitively slow for routine dense evaluation, large-scale probing, or deployment.

LoftUp [35] pursues an end-to-end learnable alternative to this per-image optimization. It improves upon local implicit functions [30] by discarding the restrictive locality assumption where queries rely solely on nearby latent codes. Instead, LoftUp enables high-resolution queries to access the full set of low-resolution feature tokens via cross-attention. Concretely, it adds sinusoidal positional encodings to both the RGB values and the low-resolution feature tokens. The positionenhanced RGB values are projected into the latent feature space using a single 3×3 convolution, and multiple crossattention blocks derive the dense features. By allowing each query to attend to the entire feature map rather than a fixed local window, this global access decouples the query’s spatial position from its information source, effectively eliminating the query-to-nearest-cell bottleneck. However, this replaces the locality problem with a semantic retrieval problem. The dense query is initialized from RGB values and coordinates using only a shallow convolution, while the keys and values are semantically rich low-resolution foundation-model features. This shallow projection may not be sufficient to map the RGB signal into the semantic feature space of the backbone. As a result, early cross-attention blocks may attend to incorrect tokens, causing feature leakage.

D. Guided Feature Upsampling

Guided upsampling methods use a high-resolution signal to guide the densification of a lower-resolution target. Classical joint bilateral upsampling [36], guided image filtering [37], and fast guided filtering [38] propagate low-resolution signals while respecting edges and structures in a high-resolution guidance image. FeatUp [32] was among the first methods to apply this idea directly to vision foundation model features. Its feed-forward variant uses joint bilateral upsampling with the input image as guidance. FeatSharp [39] improves the sharpness of FeatUp’s JBU features by combining them with features extracted from a mosaic of higher-resolution image tiles, but this increases feature-extraction cost. Subsequent feature upsamplers demonstrate that sharp dense features can be recovered without repeatedly evaluating the vision backbone on high-resolution tiles.

JAFAR [8], AnyUp [12], and NAF [14] use a singlestage cross-attention mechanism for image-guided feature upsampling. JAFAR constructs high-resolution image-derived queries and semantically enriched low-resolution keys. AnyUp follows a similar architecture, but introduces a feature-agnostic projection layer that maps low-resolution features of arbitrary

dimensionality into the query/key feature space, enabling zeroshot upsampling across different backbones and layers. NAF also targets zero-shot upsampling, but avoids feature-specific key construction by deriving the attention keys from the image encoder alone.

Recursive methods densify features progressively across scales. LiFT [40] repeatedly upsamples ViT features by fusing the current feature grid with CNN image features extracted at the corresponding image scale. Since each stage operates on the output of the previous one, recursive upsampling can degrade features across scales. UPLiFT [13] mitigates this with local attenders that provide image-guided aggregation at each step.

These methods improve spatial sharpness because the guidance signal provides high-resolution boundary and texture cues. However, the semantic quality of the upsampled features remains limited by the semantic understanding of the guidance encoder. If visually similar regions are not sufficiently distinguished by the image-derived guidance signal, the upsampler may aggregate tokens from regions that look alike but correspond to different semantic entities. Moreover, most guided upsamplers operate by reassembling low-resolution token features. Even when the resulting feature maps appear spatially sharper, the features at each high-resolution location are still formed from mixtures of the low-resolution token features, which can make the underlying semantic representation more diffuse.

ViT-Up avoids both limitations by formulating feature upsampling as coordinate-conditioned implicit feature decoding rather than guidance-based token reassembly. Given an arbitrary continuous image coordinate, ViT-Up constructs the corresponding feature from the hierarchy of intermediate ViT hidden states. This keeps the prediction process tied to the backbone feature space while producing dense features beyond the native token grid.

III. METHOD

- A. Problem Formulation

Given a Vision Transformer (ViT) [1] backbone with L layers, including the embedding layer, patch size p, and model dimension C, processing an input image of size (H,W) yields low-resolution spatial patch tokens

Hl ∈ Rh×w×C, l = 0,...,L, h = H/p,;w = W/p.

(1) The goal is to obtain substantially denser last hidden states HLup ∈ Rh

∗×w∗×C with h∗,w∗ ≫ h,w. Importantly, the upsampled features should preserve the structure of the ViT feature space while providing additional spatial detail.

- B. Architecture

- Fig. 1A shows the overall architecture of ViT-Up, while
- Fig. 2 provides a more detailed view of its individual components. As shown in Fig. 1A, ViT-Up is an implicit feature upsampling method for Vision Transformers [1] that predicts dense features at arbitrary and continuous query image coordinates xq ∈ R2 from the low-resolution ViT hidden states

Hl. Hence, HLup can be simply obtained by querying ViT-Up at high-resolution image-grid coordinates.

The central idea of ViT-Up is to follow the layer-wise organization of the ViT backbone. Given a query coordinate xq, ViT-Up first constructs an initial query embedding q0 ∈ RC (see Fig. 2B). It then progressively refines this embedding through T ViT-Up blocks {Ut | t = 1,...,T}, producing a sequence of intermediate query representations q1,...,qT ∈ RC (see Fig. 2A). Each block Ut takes as input the previous query representation qt−1, the query coordinate xq, and the low-resolution hidden state Hl[t] from backbone layer l[t] (see Fig. 2C). Therefore, ViT-Up only consumes a subset {Hl[t] | t = 1,...,T} of the available backbone hidden states and skips the remaining layers. In our main configuration, we use T = 6 with l[t] = 2t, corresponding to backbone layers {2,4,6,8,10,12}, i.e., every second backbone layer is skipped. For clarity, Fig. 2A visualizes a smaller configuration with T = 3 and l[t] = 4t. Finally, a decoder DT, implemented as a single-layer MLP with LayerNorm followed by a linear projection, maps the final query representation qT from the latent ViT-Up space back to the ViT feature space, yielding the output feature oqT at coordinate xq.

In the following, we explain the individual components in more detail.

- a) Query Embedding (Fig. 2B): The main idea of our

query embedding module is to reuse the patch embedding layer of the ViT backbone. This layer is typically implemented as a convolutional layer with kernel size p, stride p, input dimension 3, and output dimension C, where p denotes the patch size and C is the backbone feature dimension. This design has two advantages. First, it allows us to reuse the patch embedding weights of the backbone. Second, it keeps the initial query embedding aligned with the backbone patchembedding space. Since patch embedding is computationally lightweight, we can apply it at a higher input resolution. In our main configuration, we resize the input image such that the resulting patch-token grid has resolution 224 × 224. We then bilinearly interpolate this high-resolution patch-token grid at the query coordinate xq to obtain the initial query embedding q0 ∈ RC. Importantly, the high-resolution patch-token grid is cached and reused across subsequent queries.

- b) ViT-Up Block (Fig. 2C): After query embedding, the

initial query feature q0 is propagated through T ViT-Up blocks as

### qt = Ut qt−1,xq,Hl[t] , t = 1,...,T, (2)

where xq is the query coordinate and Hl[t] denotes the spatial low-resolution hidden state at backbone layer l[t].

Since ViT-Up may skip intermediate backbone layers, each ViT-Up block first aligns the incoming query representation with the feature space of the current backbone layer. To this end, a transition MLP transforms qt−1 with a residual update as

### x = qt−1 + MLPtransition (LN(qt−1)), (3)

where LN denotes LayerNorm. Next, the query aggregates token-level context from the spatial low-resolution hidden state

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

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- Fig. 2. Overview of ViT-Up. (A) Training strategy. A frozen teacher ViT is evaluated on the same image at resolutions 224×224, 448×448, and 896×896, providing feature targets at token-grid resolutions 14×14, 28×28, and 56×56. The student receives a downscaled version of the image pasted into a fixed-size image container, together with query coordinates sampled over the visible image region. A LoRA-adapted ViT backbone provides low-resolution intermediate hidden states to ViT-Up. For each sampled continuous coordinate, ViT-Up first constructs an initial query embedding and then refines it through T ViT-Up blocks that consume the corresponding backbone hidden states. For visualization, the figure shows three ViT-Up blocks consuming hidden states from layers 4, 8, and 12. In our DINOv3-S+/B main configurations, we use six ViT-Up blocks consuming hidden states from layers {2, 4, 6, 8, 10, 12}. The decoded query features are supervised at the finest teacher resolution and, after average pooling, at the corresponding coarser teacher token-grid resolutions using feature reconstruction losses and relational KL regularization. (B) Query embedding. The input image is processed with the ViT patch-embedding layer to obtain a 224×224 patch-token grid, which is bilinearly interpolated at the query coordinate xq to initialize the query representation q0. (C) ViT-Up block. Each block refines the previous query representation qt−1 by fusing token-level context from cross-attention with sub-token detail extracted from the nearest patch token using FeatX. (D) FeatX block. FeatX encodes the relative offset between the query coordinate and its nearest patch-token center, predicts FiLM parameters, and modulates the nearest-token feature to extract query-specific sub-token information.

Hl[t] via cross-attention. Inside the cross-attention block, we first normalize the query and backbone tokens as

### x˜ = LNQ (x), H˜l[t] = LNKV Hl[t] . (4)

We then apply cross-window multi-head attention, where each query attends to the spatial backbone tokens within its attention window. Queries and keys are modulated with continuous two-dimensional RoPE [41]: the query rotation Rq is evaluated at the continuous coordinate xq, while the key rotations RX are evaluated at the corresponding patch-token centers. The attention output is therefore computed as

### zattn = CW-MHA RqWQx,R˜ XWKH˜l[t],WV H˜l[t] . (5)

We accelerate this operation with NATTEN [42], [43]. The cross-window attention output is projected as

### xattn = WOzattn. (6)

While cross-window attention aggregates token-level context, it can blur high-frequency detail, especially in shallower layers. We therefore add a local feature extractor, called FeatX (see Fig. 2D). FeatX recovers sub-token detail from

the nearest-neighbor patch-token feature hnn ∈ Hl[t] and its patch-token grid coordinate xnn ∈ X relative to the query coordinate xq as

xsub-token = FeatX(hnn,xnn,xq). (7) FeatX is discussed in more detail in the next section.

The transition, attention, and FeatX outputs are then fused as

xfused = x + xattn + xsub-token. (8)

Finally, a residual fusion MLP produces the next query representation as

### qt = xfused + MLPfusion LN xfused . (9)

c) FeatX (see Fig. 2D): We introduce FeatX, a feature extractor designed to recover sub-token detail. Concretely, let X ∈ Rh×w×2 denote the grid of low-resolution patch-token center image coordinates, and let knn denote the patch index whose center is closest to the query coordinate xq. The nearestneighbor patch coordinate and feature vector at layer l[t] are then given by

### xnn = X[knn] ∈ R2, hnn = Hl[t][knn] ∈ RC. (10)

We compute the relative offset between the query coordinate and its nearest patch-token center as

### ∆x = (xq − xnn)/p, (11)

expressed in token-grid units with patch size p. Similar to coordinate-based neural fields [33], we embed this relative coordinate with a sinusoidal positional encoding, yielding p∆x = Epos(∆x) ∈ R64. Next, we use an MLP to predict position-conditioned FiLM [44] parameters from the relative offset encoding and modulate the nearest-neighbor token feature as

(γ,β) = MLPFiLM (p∆x), (12) h˜nn = (1 + γ) ⊙ LN(hnn) + β. (13)

Finally, an MLP extracts a query-specific feature from the modulated nearest-neighbor token representation as

### xsub-token = MLPsub-token h ˜nn . (14)

d) Backbone Adaptation (Fig. 2A-top): To provide additional capacity for feature upsampling without fully finetuning the backbone, we adapt the low-resolution backbone with LoRA [45]. Specifically, we apply LoRA to the patch embedding and to the query, key, value, and output projections of the ViT attention blocks. For a linear projection W, LoRA parameterizes the adapted projection as

α r

BA, (15)

Wadapted = W +

where A and B are low-rank matrices of rank r, and α controls the adapter scale. In our main setting, we use rank r = 16, scale α = 32, and adapter dropout 0.05.

- C. Training

a) Multi-scale Feature Supervision (Fig. 2A): A fundamental challenge in training feature upsamplers is the lack of an efficient high-resolution supervision signal. First, due to the quadratic cost of self-attention, evaluating a ViT backbone at high image resolutions is computationally expensive. Second, even disregarding computational cost, evaluating a ViT backbone on token grids far denser than those encountered during backbone training can lead to substantial feature degradation [8].

We address this problem by exploiting the implicit nature of ViT-Up through a student-teacher distillation strategy [46]. The teacher processes the same training image at multiple square resolutions,

S = 224,448,896, (16) which, for patch size p = 16, correspond to token-grid sizes

N = 14,28,56. (17) We denote the resulting teacher hidden states at layer l and token-grid size n by Hln.

The student receives a downscaled version of the same training image. Specifically, we sample a scale factor s ∼ U(0.1,1.0), resize the image to (s · 448,s · 448), and paste it at a random position inside a black 448×448 canvas such that

the full downscaled image remains visible. We then sample a regular grid of image coordinates {xij}56i,j=1 over the pasted image region, yielding a 56 × 56 query grid that matches the finest teacher token-grid resolution.

Let I denote the student input image, E the query embedding block, Ut the ViT-Up blocks, and Dt the linear output projections. Evaluating ViT-Up at the query coordinates xij produces dense query feature maps Hˆt56 as

q0ij = E(xij,I), (18) qtij = Ut qtij−1,xij,Hl[t] , t = 1,...,T, (19) oijt = Dt qtij , t = 0,...,T, (20)

56 i,j=1

Hˆt56 = oijt

∈ R56×56×C. (21)

For supervision at coarser teacher resolutions, we average-pool the predicted query feature map to the corresponding tokengrid size as

### Hˆtn = AvgPool56→n H ˆt56 , n ∈ N. (22)

Similar to hint-based distillation [47], we compare the student predictions Hˆtn with the corresponding teacher feature maps Hln[t] for all token-grid sizes n ∈ N and all refinement stages t = 0,...,T, with l[0] = 0 denoting the embedding layer. Thus, the same dense query prediction is supervised both at the finest teacher resolution and after aggregation to coarser ViT token grids.

Because the student must recover teacher features across multiple scales from a downscaled image pasted into a larger canvas, this supervision encourages ViT-Up to learn dense feature maps that remain scale-consistent, without requiring prohibitively expensive or degraded ultra-high-resolution teacher features.

b) Losses: We use three complementary losses. Let f denote the teacher feature vectors and ˆf the predicted feature vectors. First, we apply a target-normalized L2 loss. For each teacher feature vector f, we compute its channel-wise mean and standard deviation:

C

C

1 C

1 C

(fc − µ(f))2 + ϵ

fc, σ(f) =

µ(f) =

c=1

c=1

(23) The normalized L2 loss is:

2

ˆf − µ(f) σ(f) −

f − µ(f) σ(f)

(24)

LL2 =

2

Second, we encourage angular alignment in feature space:

ˆf⊤f ∥ˆf∥2∥f∥2 + ϵ

(25)

Lcos = 1 −

Finally, we preserve the pairwise relational structure of the teacher feature space. For a set of N spatial features from one image, let ¯fi and ¯ˆfi denote L2-normalized teacher and student features. We compute pairwise similarity matrices

- ¯ˆfi⊤
- ¯ˆfj

¯fi⊤¯fj τ

, Sˆij =

, (26)

Sij =

τ

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

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

- Fig. 3. Qualitative comparison of feature upsampling methods on DINOv3-S+. All methods use a 448×448 input image; the native backbone produces a 28×28 feature grid, and upsampled feature maps are shown at 448×448 output resolution. We show two examples: a vegetable-store scene in the top two rows and a traffic scene in the bottom two rows. For each example, the top row visualizes the feature structure using PCA, including the input image, the original 28×28 backbone feature map, and the upsampled feature maps produced by ViT-Up, NAF [14], UpLiFT [13], AnyUp [12], and JAFAR [8]. The bottom row shows the corresponding query-based similarity maps, including the input image with the query point encircled in blue, the similarity map obtained from the low-resolution backbone features, and the similarity maps obtained from the upsampled features of each method. ViT-Up produces coherent PCA structures and semantically selective similarity maps that remain aligned with the queried region. In contrast, NAF, AnyUp, and JAFAR can produce visually sharp but fragmented feature maps with leakage into nearby structures, while UpLiFT tends to produce smoother features and weaker similarity responses for small semantic regions.

where τ is a temperature. Diagonal entries are masked, and we minimize

Lrel = KL softmax(S) softmax(Sˆ) . (27) The full objective is averaged over all selected layers and

token-grid resolutions:

λL2Lℓ,nL2 + λcosLℓ,ncos + λrelLℓ,nrel (28)

### L =

ℓ∈L n∈N

In our main configuration, all loss weights (λL2,λcos,λrel) are set to one.

c) Dataset and optimization: Following prior feature upsampling work, we train on ImageNet-1K [48], for a fair comparison. As in UpLiFT [13], we train for one epoch. We use a batch size of 24, an initial learning rate of 2 × 10−4, and cosine annealing.

IV. EXPERIMENTS

- A. Evaluation Setup

a) Backbone: We use the DINOv3 [4] backbone family for our main experiments. DINOv3 is a widely used ViT backbone and is commonly used to evaluate feature upsampling

methods, including NAF [14] and UpLiFT [13]. This choice is also technically well suited to our objective: ViT-Up targets faithful upsampling of modern ViT feature maps, and DINOv3 provides cleaner intermediate representations than earlier DINOv2 [3] features. If the backbone feature map is dominated by systematic position-dependent artifacts, dense upsampling partly becomes an artifact-suppression problem [49] rather than a clean evaluation of feature upsampling. We therefore focus the main experiments on DINOv3 and provide additional DINOv2 results in Appendix B.

b) Baselines: We compare ViT-Up with standard bilinear interpolation and four recent state-of-the-art feature upsampling methods: JAFAR [8], AnyUp [12], NAF [14], and UpLiFT [13]. For JAFAR and UpLiFT, we use the publicly available DINOv3 checkpoints released by the authors. For AnyUp, we use the multi-backbone checkpoint trained on DINOv2 (ViT-S) [3], CLIP (ViT-B) [50], SigLIP (ViT-B) [51], DINOv2 with registers (ViT-S) [52], and an ImageNet-supervised ViT-B [12]. For NAF, we use the official released checkpoint, which corresponds to the DINOv3-B default training configuration described by the authors. Both AnyUp and NAF are designed for feature-agnostic inference, and we therefore evaluate them directly on DINOv3 features.

- B. Qualitative Analysis

Fig. 3 qualitatively analyzes dense feature maps on DINOv3-S+ for two examples: a vegetable-store scene and a traffic scene. All methods use an input image resolution of 448×448, and upsampling methods produce matching 448×448 output feature maps. For each example, we show a PCA projection of the feature map and a query-based similarity map. To compute the similarity map for each method, we select the feature nearest to the marked query point and compute its similarity to all other features in the corresponding feature map. The DINOv3-S+ backbone reference is shown at its original 28×28 feature resolution.

In the vegetable-store example, ViT-Up produces a more coherent feature representation than the competing methods. In the PCA visualization, neighboring vegetable regions are more clearly separated, and the shelf labels remain more consistent across instances. The similarity map confirms this behavior: when the query point is placed on a shelf label, ViT-Up assigns high similarity to other shelf labels while limiting leakage into the surrounding vegetables.

In contrast, NAF, AnyUp, and JAFAR exhibit substantial feature leakage around the selected shelf label despite producing visually sharp feature maps. Their similarity maps spread into nearby vegetables rather than remaining concentrated on label-like structures, showing that visual sharpness alone does not imply coherent dense features. This failure mode is particularly visible because the selected label is green and visually similar to the vegetables behind it. UpLiFT [13] reduces this leakage, but its features appear blurrier, and several shelf labels receive only weak similarity responses.

The traffic-scene example shows the same pattern. In the PCA visualization, feature noise is clearly visible for NAF, AnyUp, and JAFAR, especially around vehicles and background structures. These methods appear sharp, but their highfrequency variations are fragmented and do not correspond to stable semantic regions. UpLiFT, in contrast, produces a much blurrier feature map.

The similarity maps further support this observation. When the query point is placed on a traffic light, ViT-Up selectively highlights other traffic lights, including small traffic lights farther away in the scene. The competing methods produce less coherent responses: NAF, AnyUp, and JAFAR are either diffuse or leak into nearby background structures, while UpLiFT gives weaker responses on small distant traffic lights. Overall, these visualizations show that ViT-Up better preserves the semantic structure of the DINOv3 feature space, whereas visually sharp high-resolution maps do not necessarily correspond to semantically coherent dense features.

- C. Dense Linear Probing

We use dense linear probing as a controlled evaluation of upsampled feature quality on DINOv3-S+: a lightweight task-specific prediction head is trained on top of the features produced by each method, while the backbone and upsampler remain frozen. Following the linear probing setup of JAFAR [8], we evaluate semantic segmentation on VOC [53], COCO [54], ADE20K [6], and Cityscapes [5]. For monocular

depth estimation, we use COCO images with pseudo-depth targets generated by Depth Anything V2 [7], [55]. We train the probing head with a cosine learning-rate schedule initialized at 2×10−3. All probing heads are trained with a batch size of 4 for 20 epochs, except on COCO, where we train for 5 epochs. For all methods, the input resolution is fixed to 448×448, and the target feature-map resolution is matched to the input resolution.

As shown in Table I, ViT-Up consistently improves over all baselines across semantic segmentation and depth estimation. On semantic segmentation, ViT-Up reaches 64.09 mIoU on COCO, 87.47 mIoU on VOC, 44.72 mIoU on ADE20K, and 65.41 mIoU on Cityscapes. This improves over the best baseline in each dataset by +0.23, +1.63, +0.49, and +2.07 mIoU, respectively. The corresponding pixel-accuracy gains are +0.16 on COCO, +0.42 on VOC, +0.35 on ADE20K, and +0.60 on Cityscapes.

The largest gains appear on Cityscapes, VOC, and COCO depth. Since Cityscapes contains many small objects and thin structures, including pedestrians, poles, traffic signs, and traffic lights, the strong improvement on this dataset suggests that ViT-Up more effectively extracts fine spatial detail from the backbone representation. On VOC, where images are dominated by foreground objects with strong categorylevel structure, the gain suggests that ViT-Up better maintains coherent object representations across the upsampled feature field. For COCO depth estimation, ViT-Up improves δ1 from 62.17 to 62.72 and reduces RMSE from 61.15 to 59.82, corresponding to gains of +0.55 in δ1 and 1.33 in RMSE reduction. This indicates that ViT-Up produces features with strong geometric information for dense prediction.

Fig. 4 provides qualitative evidence for the same behavior. In the segmentation example, NAF, AnyUp, and JAFAR produce sharp but fragmented PCA maps, with visible feature leakage between the person and the bus. This is especially pronounced where the yellow bus region overlaps visually with the person’s sweater, leading to incorrect segmentation around the upper body and arm. UpLiFT avoids some of this leakage, but its features are blurrier, especially around the head region, which is also reflected in the segmentation mask. ViTUp produces a more coherent person representation and yields a cleaner segmentation prediction.

The depth example shows a similar pattern. For NAF, AnyUp, and JAFAR, the PCA maps contain fragmented highfrequency variations, most visibly around the small ducks behind the larger duck. UpLiFT produces smoother features, but the distant ducks blur into the background. These feature artifacts translate directly into the depth predictions: the competing methods produce degraded depth estimates in the same regions where the feature maps are fragmented or blurred, whereas ViT-Up better preserves the individual duck shapes and their spatial structure.

Overall, the probing results show that ViT-Up improves dense prediction across both semantic segmentation and depth estimation. The qualitative examples in Fig. 4 show that feature artifacts translate into task errors: leakage in NAF, AnyUp, and JAFAR leads to incorrect local predictions, while blurring in UpLiFT weakens small structures. In contrast, ViT-

TABLE I PROBING RESULTS ON DINOV3-S+. LINEAR PROBING HEADS ARE TRAINED WITH BATCH SIZE 4 FOR 20 EPOCHS, EXCEPT ON COCO WHERE HEADS ARE TRAINED FOR 5 EPOCHS, USING A COSINE LEARNING-RATE SCHEDULE INITIALIZED AT 2×10−3. HIGHER IS BETTER FOR MIOU, ACCURACY, AND δ1; LOWER IS BETTER FOR RMSE. GAINS ARE COMPUTED AGAINST THE BEST NON-VIT-UP BASELINE.

Semantic Segmentation Depth Estimation COCO VOC ADE20K Cityscapes COCO mIoU ↑ Acc ↑ mIoU ↑ Acc ↑ mIoU ↑ Acc ↑ mIoU ↑ Acc ↑ δ1 ↑ RMSE ↓

Method

Bilinear 63.10 81.85 84.88 96.45 43.27 76.17 61.36 93.44 61.52 62.80 JAFAR 62.50 81.50 83.88 96.16 42.48 75.81 57.78 92.47 60.64 64.92 AnyUp 63.03 81.83 84.54 96.34 42.77 76.02 58.96 92.93 61.66 62.62 UpLiFT 63.79 82.28 85.69 96.72 44.24 76.71 63.08 93.94 61.84 61.79 NAF 63.86 82.33 85.84 96.72 44.17 76.69 63.34 94.13 62.17 61.15

#### ViT-Up (Ours) 64.09 82.49 87.47 97.14 44.73 77.06 65.41 94.73 62.72 59.82

Gain vs. best baseline +0.23 +0.16 +1.63 +0.42 +0.49 +0.35 +2.07 +0.60 +0.55 +1.33

TABLE II CORRESPONDENCE RESULTS ON DINOV3-S+. WE REPORT PCK AT DIFFERENT TOLERANCE THRESHOLDS. SEMANTIC CORRESPONDENCE IS EVALUATED ON SPAIR-71K AND GEOMETRIC CORRESPONDENCE ON NAVI. HIGHER IS BETTER FOR ALL METRICS. GAINS ARE COMPUTED AGAINST THE BEST NON-VIT-UP BASELINE.

Semantic Correspondence Geometric Correspondence SPair-71k NAVI

Method

0.10 ↑ 0.05 ↑ 0.01 ↑ 0.10 ↑ 0.05 ↑ 0.01 ↑

Bilinear 51.27 33.74 3.83 80.16 51.18 33.58 JAFAR 36.82 18.59 1.89 79.04 47.02 26.60 AnyUp 37.63 19.31 1.97 80.31 48.78 28.37 UpLiFT 46.87 29.15 3.43 79.35 49.05 30.49 NAF 48.68 33.96 2.89 80.03 50.29 31.62

#### ViT-Up (Ours) 55.44 39.07 7.30 80.81 51.59 33.83

Gain vs. best baseline +4.17 +5.11 +3.47 +0.50 +0.41 +0.25

Up produces more coherent dense features and cleaner outputs.

- D. Correspondence Estimation

We further evaluate dense feature quality on semantic and geometric correspondence. Correspondence directly probes whether feature similarity preserves semantic and geometric structure, and is therefore complementary to dense linear probing. To the best of our knowledge, prior feature upsampling methods have not been systematically evaluated on correspondence benchmarks, despite correspondence being an important feature assessment protocol in DINOv3 [4]. We therefore include these experiments both to assess ViT-Up and to establish correspondence as a useful benchmark for dense feature upsampling.

For semantic correspondence, we use SPair-71k [16] and adapt the protocol used in the original DINOv3 evaluation. We report percentage of correct keypoints (PCK), where a predicted correspondence is counted as correct if it falls within a given normalized distance threshold from the annotated target point. While DINOv3 evaluates semantic correspondence at 1024×1024 input resolution, we use 448×448 inputs for consistency with our probing setup and evaluate all upsampling methods at 448×448 output resolution, matching the input

resolution. For geometric correspondence, we use NAVI [56] and adapt the DINOv3/Probe3D protocol [15]. On NAVI, we report 3D PCK, where a predicted correspondence is counted as correct if its reconstructed 3D point lies within the specified distance threshold in meters from the annotated target point. We use 448×448 inputs instead of 512×512 and keep the scale factor and number of correspondences fixed at 0.25 and 1000, respectively. The scale factor of 0.25 evaluates correspondences on a 112×112 target feature grid, so all methods are evaluated at 112×112 output resolution on NAVI.

Table II shows that ViT-Up substantially improves semantic correspondence on SPair-71k. ViT-Up obtains 55.44, 39.07, and 7.30 PCK at thresholds 0.10, 0.05, and 0.01, respectively. Compared with the strongest baseline, this corresponds to gains of +4.17, +5.11, and +3.47 points. The improvement remains large even at the strictest threshold of 0.01, where correspondence accuracy depends most strongly on fine spatial and semantic detail. Compared with bilinear interpolation of the backbone features, ViT-Up improves SPair-71k PCK from 51.27 to 55.44 at threshold 0.10, from 33.74 to 39.07 at threshold 0.05, and from 3.83 to 7.30 at threshold 0.01. At this finest evaluation scale, ViT-Up nearly doubles the performance of bilinear interpolation, indicating that it recovers dense features that preserve precise part-level correspondences rather

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- Fig. 4. Qualitative dense probing results on DINOv3-S+. All methods use a 448×448 input image; the native backbone produces a 28×28 feature grid, and dense predictions are shown at 448×448 output resolution. Top two rows: semantic segmentation example with a person in front of a bus, showing PCA projections of the dense features and the corresponding predicted segmentation masks together with the ground-truth mask (left). Bottom two rows: monocular depth example with a group of ducks, showing PCA projections of the dense features and the corresponding predicted depth maps together with the pseudo-depth target (left) generated by Depth Anything V2 [7]. For the bilinear baseline, we visualize PCA on the original 28×28 backbone feature grid. NAF [14], AnyUp [12], and JAFAR [8] produce visually sharp but fragmented feature maps and noisy dense predictions. UpLiFT [13] produces more localized but blurrier features, weakening fine structures such as the person’s head region and small distant ducks. ViT-Up produces more coherent dense features, resulting in cleaner segmentation masks and more consistent depth estimates.

than only coarse semantic alignment.

- Fig. 5 provides qualitative evidence for this behavior on

fully matches the tail, while NAF and UpLiFT mix the tail feature with the obstacle and produce substantially incorrect correspondences. Importantly, the remaining incorrect ViTUp matches are still semantically and spatially close to the annotated target point, but fall outside the strict SPair-71k PCK@0.1 threshold.

two SPair-71k image pairs. For each pair, we visualize PCA projections and semantic correspondences obtained from dense feature similarity. Specifically, a match is obtained by taking a source query feature from the upsampled source features and selecting the target location with maximum cosine similarity over all upsampled target features. The PCA basis is computed on object regions extracted with Segment Anything V3 [57]– [59] masks by collecting multi-scale backbone features on the mask and applying the resulting projection to the predicted dense features. Due to space constraints, we restrict the qualitative comparison to ViT-Up, NAF, and UpLiFT, the strongest competing learned upsampling baselines in our quantitative results.

This result is important because SPair-71k requires subclass and part-level semantic discrimination, not only classlevel consistency. Previous feature upsamplers can produce visually plausible or smooth dense maps, but feature leakage and over-smoothing can destroy the local feature geometry needed for correspondence. ViT-Up better preserves finegrained feature structure, which explains the large gains on SPair-71k, especially at stricter PCK thresholds.

On NAVI, ViT-Up also obtains the best results across all thresholds. It reaches 80.81 PCK at threshold 0.10, improving over the strongest baseline, AnyUp, at 80.31. At stricter thresholds, ViT-Up obtains 51.59 and 33.83 PCK at 0.05 and 0.01, respectively, improving over the strongest baseline results of 51.18 and 33.58 from bilinear interpolation. Although the gains on NAVI are smaller than on SPair-71k, they show that ViT-Up preserves geometric correspondence while substantially improving the semantic correspondence regime

In the bird example, ViT-Up produces more fine-grained object features and yields accurate matches across the two birds. In contrast, NAF and UpLiFT show feature leakage around the bird tail, where the target feature becomes mixed with background regions and leads to incorrect matches. The horse example shows an even more challenging case: the tail in the target image is partially occluded by an obstacle with a grid-like structure. ViT-Up is the only method that success-

[Figure 74]

[Figure 75]

- Fig. 5. Qualitative semantic correspondence results on SPair-71k. We show two image pairs, birds in the top two rows and horses in the bottom two rows. For each pair, the first row shows PCA projections of the dense features, and the second row shows correspondences obtained by matching each source query point to the target location with maximum cosine similarity. Green lines indicate correct matches under the SPair-71k threshold, while red lines indicate incorrect matches. ViT-Up produces fine-grained object features and accurate part-level correspondences. NAF [14] and UpLiFT [13] suffer from feature fragmentation or blurring in small structures such as the bird tail and the partially occluded horse tail, leading to incorrect matches.

TABLE III FEATURE PRESERVATION UNDER FROZEN-HEAD EVALUATION. A TASK HEAD IS TRAINED ON NATIVE DINOV3-S+@28 FEATURES AND EVALUATED WITHOUT RETRAINING ON UPSAMPLED HIGH-RESOLUTION FEATURES.

Segmentation mIoU↑ Depth δ1↑ COCO VOC ADE City. COCO

Method

Bilinear 63.10 84.88 43.27 61.36 61.52 JAFAR 61.97 82.35 41.79 55.15 58.85 AnyUp 62.88 84.47 41.89 55.18 61.47 UpLiFT 63.01 83.22 42.80 60.01 59.96 NAF 63.78 85.78 43.82 61.41 61.99 ViT-Up (Ours) 63.67 87.01 43.78 63.87 59.83

where fine-grained feature faithfulness is most critical.

- E. Feature Preservation

We further evaluate whether the upsampled features remain compatible with predictors trained on the original lowresolution feature space. To this end, we train task heads on native DINOv3-S+@28 features and evaluate the same frozen heads on high-resolution upsampled features. For semantic segmentation, this protocol measures class-level feature preservation: an upsampler should increase spatial resolution without changing the semantic organization of the feature space expected by the frozen classifier.

As shown in Table III, ViT-Up preserves class-level semantics particularly well on VOC and Cityscapes, where it substantially outperforms all prior upsamplers. Notably, on these datasets ViT-Up even exceeds the finetuned variants of competing methods reported in Table I, despite using a head trained only on low-resolution features. This indicates that ViT-Up does not merely sharpen features visually, but preserves the semantic feature organization required by a frozen dense predictor. NAF also exhibits strong feature preservation and remains close to its finetuned counterpart. This behavior is consistent with its formulation: similar to JAFAR and AnyUp, NAF obtains each upsampled feature through a single finallayer cross-attention operation over the low-resolution token features. When the attention weights are accurate, as observed for NAF, the resulting weighted combination of tokens remains close to the original low-resolution feature distribution by construction. On COCO and ADE20K, NAF is slightly better than ViT-Up under frozen probing, suggesting that feature reassembly from low-resolution tokens can be beneficial when the evaluation head is fixed. However, when the segmentation head is trained on the corresponding high-resolution features, ViT-Up outperforms NAF, showing that the proposed representation contains more usable high-resolution information once the predictor is allowed to adapt. In contrast, UpLiFT exhibits weaker feature preservation across the frozen probing results. We hypothesize that this is due to its recursive upsampling strategy, where small deviations from the original feature space

TABLE IV COMPARISON ON DINOV3-B. WE REPORT SEGMENTATION PROBING MIOU ON VOC AND CITYSCAPES, AND CORRESPONDENCE PCK ON SPAIR-71K.

Seg. Probing, mIoU ↑ SPair-71k, PCK ↑ VOC Cityscapes 0.1 0.05 0.01

Method

Bilinear 87.20 64.21 50.01 34.03 3.89 NAF 88.07 66.45 47.19 29.09 3.41

ViT-Up (Ours) 89.18 69.81 58.10 42.00 7.93 Gain vs. best baseline +1.11 +3.36 +8.09 +7.97 +4.04

may compound over successive refinement stages.

For depth estimation, frozen probing should be interpreted more cautiously. Unlike segmentation, depth prediction is a continuous regression problem and is highly sensitive to the resolution and local smoothness of the input feature map. The frozen depth head is trained on native DINOv3-S+@28 tokens, where each token aggregates information over a relatively large image region. Consequently, the head only observes coarse features during training. Upsamplers that mainly interpolate or reassemble the low-resolution representation, such as AnyUp or NAF, can remain closer to the distribution expected by this frozen predictor and may therefore perform favorably under this protocol. ViT-Up, in contrast, produces more localized high-resolution features, which changes the feature statistics and introduces an unavoidable distribution shift for the frozen low-resolution depth head. This may explain why ViT-Up is not strongest under frozen depth probing, even though it substantially outperforms NAF when the depth head is trained on ViT-Up features. We therefore include frozen depth probing for completeness, while treating finetuned depth estimation as the more informative measure of usable highresolution geometric information.

- F. Backbone Scaling

To evaluate whether ViT-Up generalizes to larger backbones, we additionally test our architecture on DINOv3-B. Since DINOv3-B doubles the feature dimension compared to DINOv3-S+, we scale ViT-Up accordingly by doubling its internal dimension, while keeping the overall architecture unchanged. Table IV compares ViT-Up to bilinear interpolation and NAF on DINOv3-B. We omit UpLiFT because no publicly available DINOv3-B checkpoint is provided. Since NAF reports superior performance over AnyUp and JAFAR on DINOv3-B, consistent with our findings on DINOv3-S+, we keep the learned-baseline comparison focused on NAF and include bilinear interpolation as a standard non-parametric baseline. For segmentation probing, we use the same hyperparameters as in the main DINOv3-S+ experiments.

Scaling to DINOv3-B substantially increases the advantage of ViT-Up on Cityscapes and SPair-71k. On Cityscapes, the margin over NAF grows from 2.07 to 3.36 mIoU. Since Cityscapes contains many thin and small structures, this suggests that ViT-Up can better exploit the increased backbone capacity to recover fine spatial detail. The effect is even clearer for correspondence: on SPair-71k, ViT-Up widens its

TABLE V ABLATION OF INDIVIDUAL VIT-UP COMPONENTS. WE REPORT FROZEN-HEAD SEMANTIC SEGMENTATION MIOU ON VOC AND CITYSCAPES, AND SEMANTIC CORRESPONDENCE PCK ON SPAIR-71K.

Frozen Seg. Head, mIoU ↑ SPair-71k, PCK ↑ VOC Cityscapes 0.10 0.05 0.01

Ablation

Cross-Attention off 86.08 60.44 51.29 33.77 5.45 FeatX off 86.54 62.99 55.24 38.21 6.24 LoRA off 86.47 63.49 54.58 37.98 6.31 Decoder off 86.77 62.98 54.80 37.97 6.25 Transition off 86.72 63.17 54.57 37.52 6.04 KL off 86.84 63.41 54.82 38.05 6.44

Base 86.85 63.38 54.90 38.10 6.42

margin over the best baseline across all PCK thresholds, with the largest increase at the standard PCK@0.1 threshold. VOC shows a more moderate trend: ViT-Up still substantially outperforms NAF, but the margin decreases from 1.69 to 1.11 points as both methods benefit from the larger backbone. This is consistent with VOC being dominated by foregroundbackground object segmentation, a favorable setting for imageguided aggregation because less sub-token detail needs to be recovered.

Interestingly, in our protocol, increasing the backbone size does not automatically improve semantic correspondence at the coarse native token resolution. With bilinear interpolation from the native 28×28 token grid to 448×448, DINOv3B performs slightly worse than DINOv3-S+ on SPair-71k PCK@0.1, decreasing from 51.27 to 50.01, while the stricter thresholds improve only marginally. We hypothesize that, at this coarse token resolution, the larger DINOv3-B backbone may use its increased capacity to encode more fine-grained intra-patch information. While this can enrich the representation, it may also produce less selective similarity maps for semantic matching. In contrast, the smaller DINOv3-S+ backbone may discard some local detail and emphasize the dominant object-level content within each patch, which can be favorable for coarse semantic correspondence.

This behavior may also explain why NAF performs worse on DINOv3-B than on DINOv3-S+ in our SPair-71k evaluation, decreasing by 1.49 points at PCK@0.1 from 48.68 to 47.19. Since NAF produces high-resolution features by aggregating low-resolution tokens under external image guidance, it cannot explicitly recover sub-token structure that may be encoded inside the higher-capacity ViT features. Moreover, scaling the ViT backbone does not increase the capacity of NAF’s separate image encoder, leaving the aggregation weights constrained by the same guidance representation.

Overall, these results show that ViT-Up scales favorably to larger backbones under our evaluation protocol. Unlike prior guidance-based methods that rely on a separate image encoder, ViT-Up constructs dense features directly from the ViT hidden states and can therefore make better use of the increased feature dimension of the larger backbone.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

- Fig. 6. Qualitative ablation of cross-attention and FeatX on DINOv3-S+. The input image has resolution 448×448, the native DINOv3-S+ feature grid has resolution 28×28, and all dense feature maps are shown at 448×448 resolution. From left to right, we show the input RGB image, DINOv3-S+@28, ViT-Up after one epoch of training, and the one-quarter-epoch ablated variants of ViT-Up in its base configuration, without FeatX, and without cross-attention. Without FeatX, the features remain spatially consistent but lose fine local texture, most visibly near the arm ends of the sea star highlighted by the green box. Without cross-attention, the features preserve more local detail but exhibit pronounced pixelation artifacts, especially in the region highlighted by the red box. Combining both components yields coherent features with visible local detail; the remaining pixelation artifacts after one-quarter epoch of training largely disappear after one full epoch. Best viewed digitally.

TABLE VI ABLATION OF THE NUMBER OF REFINEMENT LAYERS IN VIT-UP. WE REPORT FROZEN-HEAD SEMANTIC SEGMENTATION MIOU ON VOC AND CITYSCAPES, AND SEMANTIC CORRESPONDENCE PCK ON SPAIR-71K.

Frozen Seg. Head, mIoU ↑ SPair-71k, PCK ↑ VOC Cityscapes 0.10 0.05 0.01

Layers

- 1 86.01 61.47 52.13 33.64 4.10
- 2 86.51 62.78 53.69 36.48 6.02
- 3 86.71 62.99 54.11 36.88 6.21 6 (base) 86.85 63.38 54.90 38.10 6.42 12 86.83 63.42 55.03 38.20 6.57

TABLE VII ABLATION ON THE OUTPUT FEATURE RESOLUTION. WE REPORT FROZEN-HEAD SEMANTIC SEGMENTATION MIOU ON VOC AND CITYSCAPES, AND SEMANTIC CORRESPONDENCE PCK ON SPAIR-71K.

Frozen Seg. Head, mIoU ↑ SPair-71k, PCK ↑ VOC Cityscapes 0.10 0.05 0.01

Resolution

28×28 85.07 59.03 47.25 28.19 2.63 56×56 86.69 62.63 53.41 35.79 4.92 112×112 87.00 63.74 54.92 38.19 6.87 224×224 87.02 63.95 55.39 39.10 7.33 448×448 87.00 63.91 55.44 39.07 7.30

- G. Ablation Studies

For all ablations, we train ViT-Up on ImageNet for 20k iterations with a batch size of 16. We use frozen segmentation probing and SPair-71k correspondence as ablation metrics because they directly evaluate the two properties targeted by ViTUp: preservation of semantic feature structure and recovery of spatial detail. Frozen probing tests compatibility with a fixed segmentation head trained on low-resolution DinoV3-S+ features, while SPair-71k measures spatially precise semantic correspondence. In all ablation experiments, the input image resolution is 448×448, and the target output resolution is set to 112×112 unless stated otherwise.

a) Individual Components: Table V ablates the main architectural components of ViT-Up. Removing either crossattention or FeatX substantially degrades frozen probing performance. Interestingly, removing FeatX gives a slight edge over the base model on SPair-71k at the coarse PCK@0.10

TABLE VIII PROBING RESULTS FOR VIT-UP AT DIFFERENT OUTPUT RESOLUTIONS. WE REPORT SEGMENTATION MIOU AND DEPTH ESTIMATION δ1. HIGHER IS BETTER FOR ALL METRICS.

Segmentation mIoU ↑ Depth δ1 ↑ COCO VOC ADE20K Cityscapes COCO

Resolution

112 × 112 64.12 87.46 44.72 65.14 62.77 448 × 448 64.09 87.47 44.73 65.41 62.72

threshold. A possible explanation is that the full model has to fuse the cross-attention output with the extracted sub-token features from FeatX, which can slightly perturb the coarse semantic structure of the features. However, the full model remains stronger at the stricter PCK@0.01 threshold, indicating that this fusion improves fine-grained correspondence accuracy.

The qualitative comparison in Fig. 6 helps explain this behavior. When FeatX is disabled, the features remain spatially consistent, but lack fine detail and texture; this is especially visible in the reduced texture towards the arm ends of the sea star. In contrast, disabling cross-attention preserves more local texture and detail on the sea star, but introduces clear pixelation artifacts. Combining both components yields features that are both coherent and detailed. At one-quarter epoch training, minor pixelation artifacts can still remain, but these artifacts largely disappear when training for a full epoch.

Removing LoRA or the KL regularization has a smaller effect than removing cross-attention or FeatX, but both variants slightly reduce SPair-71k performance. Disabling LoRA marginally improves Cityscapes mIoU by 0.11, which may be caused by the frozen-head evaluation protocol: adapting the backbone features through LoRA can slightly change the feature distribution seen by the frozen probe. Nevertheless, the full model gives the best overall trade-off across probing and correspondence.

Finally, disabling either the decoder or the transition MLP consistently lowers performance. The decoder ablation is particularly worse on Cityscapes, suggesting that even a simple linear output projection is beneficial. This is notable because the latent dimension equals the output feature dimension;

the decoder therefore does not merely change dimensionality, but appears to help the model better organize and utilize the feature channels before producing the final upsampled representation.

b) Number of Refinement Layers: Table VI ablates the refinement depth of ViT-Up. Increasing the number of refinement layers yields consistent and substantial gains across both frozen probing and correspondence. From one to twelve layers, Cityscapes improves by 1.95 mIoU and VOC improves by 0.82 mIoU. The gains are even more pronounced for correspondence: SPair-71k improves by 2.90, 4.56, and 2.47 points at PCK levels 0.10, 0.05, and 0.01, respectively.

These improvements are largest on metrics that depend strongly on spatial precision, namely Cityscapes and the stricter SPair-71k thresholds. This shows that refinement depth is critical for producing spatially localized, discriminative high-resolution features. A shallow one-layer variant can still produce usable outputs, but it leaves substantial performance on the table and fails to match the spatial precision of deeper variants. The ablation therefore validates a central design choice of ViT-Up: faithful feature upsampling benefits from constructing the output representation through multiple intermediate refinement layers, rather than relying on a single shallow prediction.

We use six layers as the base configuration because it captures most of the improvement while providing a better accuracy–runtime trade-off than the twelve-layer variant.

c) Output Feature Resolution: Table VII analyzes the effect of output feature resolution under frozen probing and correspondence evaluation. Increasing the resolution beyond the native 28×28 token grid yields substantial gains, with the largest improvement already obtained at 56×56 and a further clear gain at 112×112. Performance continues to improve at 224×224, but saturates at full 448×448 resolution.

We observe a similar trend when the task head is trained directly on the upsampled features. As shown in Table VIII, performance is largely similar between 112×112 and 448×448 resolutions, except on Cityscapes, where mIoU increases slightly from 65.14 to 65.41 at full resolution. Since Cityscapes contains many thin structures and small objects, this suggests that full-resolution features can still provide useful spatial detail when fine image structure is particularly important. The slight fluctuations on the other datasets may reflect resolution-specific convergence behavior, since we keep the learning rate and number of training epochs fixed across resolutions.

Overall, these results indicate that ViT-Up benefits from predicting a denser feature grid, but that most of the recoverable semantic and spatial information is already captured before reaching full image resolution. This saturation likely reflects both the limited precision of ground-truth segmentation masks and the difficulty of producing sufficiently sharp features at full image resolution, which may require longer training, larger latent dimensions, or stronger high-resolution supervision.

- H. Runtime and Memory

Table IX reports runtime and peak CUDA memory on a single H100 SXM GPU using bfloat16 inference. We measure

TABLE IX RUNTIME AND PEAK MEMORY ON DINOV3-S+ USING A SINGLE H100 SXM GPU IN BFLOAT16. WE REPORT MULTIPLE OUTPUT RESOLUTIONS TO SHOW HOW EACH METHOD SCALES. FOR VIT-UP, THE QUERY CHUNK SIZE CONTROLS THE MEMORY–RUNTIME TRADEOFF WITHOUT CHANGING THE OUTPUT RESOLUTION.

Time [ms] ↓ VRAM [MiB] ↓

Method Params Chunk

1122 2242 4482 1122 2242 4482

JAFAR 0.6M – 41.3 43.5 52.9 541.0 1721.6 6448.5 AnyUp 0.8M – 6.8 15.3 59.8 485.9 774.1 2760.7 UpLiFT 0.8M – 6.4 7.2 10.1 606.6 765.7 1391.5 NAF 0.7M – 29.1 30.7 37.4 602.6 602.9 1780.5

1122 14.2 24.1 62.6 503.7 503.9 626.7 2242 - 22.1 55.4 - 1069.6 1143.6 4482 - - 52.0 - - 3709.3

ViT-Up (Ours) 24.9M

end-to-end forward runtime with batch size 1 and 448×448 input images, including the DINOv3-S+ backbone for all methods. This is important for fairness, since ViT-Up adapts the backbone with LoRA and therefore cannot be timed independently from it. For each method, we perform 10 warmup iterations and report the average runtime over 50 subsequent iterations.

Although ViT-Up has substantially more parameters than prior feature upsamplers, the table shows that parameter count is a poor predictor of both runtime and memory usage. In dense feature upsampling, peak memory is dominated by intermediate activations rather than parameter storage. A useful property of ViT-Up is that output queries are conditionally independent given the backbone features: each query can be evaluated independently of the other output queries. We can therefore process queries in chunks and concatenate the resulting features, which yields exactly the same upsampled feature map as processing all queries at once. This bounds the number of active output queries without changing the model or the final output resolution. As a result, ViT-Up can substantially reduce memory usage while trading only a small amount of runtime: at 448×448 output resolution, using 112×112 query chunks gives the lowest measured peak memory among the compared dense upsampling methods. This property is also important during training: in our experiments, query chunking enables training on a single RTX 5090 with batch size 24, which would not fit in memory without chunking.

In terms of runtime, ViT-Up is competitive with existing high-resolution upsampling baselines. At 448×448 output resolution, ViT-Up is on par with JAFAR and AnyUp, while being slower than NAF and UpLiFT. UpLiFT is the fastest method in our benchmark; however, this is a favorable setting: we use its highly optimized compiled implementation, which requires an additional compilation stage of roughly 30–60 seconds before inference, and evaluate it at a recursion-aligned output resolution, avoiding an additional recursion stage followed by downsampling.

Although our main experiments evaluate ViT-Up at the same full output resolution, the resolution ablation in Table VIII shows that ViT-Up already outperforms all full-resolution prior upsamplers when queried at only 112×112. At this efficient

operating point, ViT-Up runs only about 4 ms slower than UpLiFT at full resolution, while keeping memory usage low. This indicates that ViT-Up can match the practical runtime regime of the fastest baseline, while providing stronger feature quality even at substantially lower output resolution.

V. LIMITATIONS AND OUTLOOK

a) Information Bottleneck: All post-hoc feature upsamplers are ultimately bounded by the information encoded in the backbone hidden states. ViT-Up mitigates this limitation by exploiting intermediate ViT representations, allowing it to recover substantial sub-token detail and produce dense features that are more spatially precise than the native token grid. However, structures that are not represented in the hidden states cannot be fully recovered from the upsampled features alone.

A practical way to reduce this limitation is to use moderately higher-resolution backbone features. While our main setting uses 28×28 hidden states, we find that the backbone remains effective at 56×56 resolution, with feature quality degrading only at substantially higher resolutions such as 112×112. Moreover, at 56×56, the feed-forward and projection operations still dominate over the quadratic attention term. Using 56×56 hidden states as input to ViT-Up may therefore provide additional spatial evidence while remaining computationally practical.

b) Backbone Coupling: ViT-Up is currently trained separately for each backbone because several parts of the model are tightly coupled to the backbone’s internal representation. For example, the transition MLP must learn projections between skipped hidden layers, while FeatX must learn how to extract sub-token information from the intermediate ViT representations. Both depend on how a specific backbone organizes semantic and spatial information across layers, making it difficult to design ViT-Up as a fully backbone-agnostic upsampler. In practice, this remains a small one-time cost: once trained, the same ViT-Up module can be reused across downstream tasks, datasets, and output resolutions for a fixed backbone.

A complementary future direction is to train ViT-Up jointly with the backbone. Current ViT backbones are not explicitly optimized for continuous high-resolution feature reconstruction and may therefore discard local spatial detail that is not required by their native training objective. Joint training would allow high-resolution reconstruction losses to shape the hidden states directly, encouraging intermediate representations that better support coordinate-conditioned query modulation. This may also alleviate the information bottleneck discussed above, since the backbone would no longer be treated as a fixed source of low-resolution features. While our LoRA-based adaptation is a lightweight step in this direction, full backbone finetuning may be necessary to realize the full benefit of this coupling.

VI. CONCLUSION

We introduced ViT-Up, an implicit feature upsampling framework that predicts vision transformer features at arbitrary continuous image coordinates. To reconstruct dense feature

TABLE X COMPARISON TO HIGHER NATIVE DINOV3-S+ TOKEN RESOLUTIONS. WE REPORT SEGMENTATION PROBING MIOU AND SPAIR-71K SEMANTIC CORRESPONDENCE PCK. FOR SEGMENTATION PROBING WITH NATIVE BACKBONE FEATURES, WE USE BILINEAR INTERPOLATION TO MATCH THE TARGET OUTPUT RESOLUTION OF 448×448.

Seg. mIoU ↑ SPair PCK ↑ COCO VOC ADE City. 0.10 0.05 0.01

Method

DINOv3-S+@28 63.1 84.9 43.3 61.4 48.1 28.9 2.7 DINOv3-S+@56 63.7 87.0 44.5 67.4 54.8 38.0 5.9 DINOv3-S+@112 – – – – 49.0 32.2 5.8

ViT-Up (Ours) from DINOv3-S+@28 64.1 87.5 44.7 65.4 55.4 39.1 7.3

maps, ViT-Up follows the layer hierarchy of the backbone. Starting from a query embedding derived from the backbone patch embedding, ViT-Up progressively refines the queries with low-resolution hidden states from intermediate backbone layers. This avoids relying only on the final hidden state and makes the dense prediction process consistent with the backbone’s internal representation hierarchy.

Across linear probing for segmentation and depth as well

- as semantic correspondence, ViT-Up shows significant gains over existing feature upsampling methods, demonstrating its effectiveness for dense visual prediction and fine-grained correspondence.

Overall, ViT-Up provides an effective and faithful way to obtain dense feature maps from vision transformers. We hope this work encourages future vision backbones to support continuous, high-resolution feature querying as a native capability.

ACKNOWLEDGMENTS

We thank Nils Wandel for proofreading the manuscript. The authors used GitHub Copilot, OpenAI Codex, and ChatGPT for implementation support and language editing. All scientific content was developed and verified by the authors.

APPENDIX A COMPARISON TO HIGHER NATIVE TOKEN RESOLUTIONS

Table X compares ViT-Up to DINOv3-S+ evaluated

- at higher native token resolutions. These high-resolution DINOv3-S+ variants are not feature upsampling baselines, since they require running the full ViT backbone on a denser token grid. Instead, they serve as references for the alternative strategy of obtaining denser features by increasing the native backbone resolution.

Increasing the DINOv3-S+ token grid from 28×28 to 56×56 substantially improves both segmentation probing and semantic correspondence, confirming the importance of spatial resolution for dense prediction. However, higher native token resolution does not necessarily lead to better features. On SPair-71k, DINOv3-S+@112 performs worse than DINOv3S+@56 across all PCK thresholds, despite using a denser token grid. This indicates that simply evaluating the backbone at increasingly higher native resolutions is not a reliable substitute for feature upsampling.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- Fig. 7. Upsampling versus artifact suppression. For each example, the top row shows DINOv2-based features and the bottom row shows DINOv3-based features under the same visualization protocol. DINOv2 exhibits stronger spatial leakage and position-dependent artifacts, while DINOv3 provides a cleaner dense feature field. Image-guided upsamplers can suppress such artifacts by injecting high-resolution image cues, whereas ViT-Up reconstructs the target ViT representation more directly. This explains the trade-off observed across backbones: artifact-prone features favor suppression-based behavior, while clean modern ViT features favor faithful reconstruction.

ViT-Up provides a different tradeoff. Starting from the standard 28×28 DINOv3-S+ features, it produces dense features at the target output resolution without running the full backbone on a denser token grid. Compared to the stronger DINOv3-S+@56 reference, ViT-Up improves COCO, VOC, and ADE20K segmentation mIoU, as well as all SPair-71k PCK thresholds, with the largest gain at the strict PCK@0.01 threshold. The only exception is Cityscapes, where DINOv3S+@56 remains stronger. Compared to DINOv3-S+@112, ViT-Up is consistently better on semantic correspondence. These results show that ViT-Up does not merely approximate expensive high-resolution backbone inference, but can produce dense features that are more effective than naively increasing the native token resolution.

APPENDIX B UPSAMPLING VERSUS ARTIFACT SUPPRESSION

The distinction between faithful feature upsampling and artifact suppression is important because benchmarking on DINOv2 [3] does not only measure whether a method can

recover higher-resolution features; it also measures whether the method can remove artifacts already present in the backbone feature field. DINOv2 features are known to contain position-dependent artifacts, and recent work on denoising vision transformers traces such artifacts to the use of positional embeddings in ViTs [49]. In our visualizations, we observe two recurring artifact modes: grid-like position encoding artifacts and spatial feature leakage across object boundaries.

These artifacts are closely related to the design of the target representation and the upsampler. Several prior upsampling methods use a separate image encoder or image-guided pathway to construct high-resolution queries, keys, or guidance features. Such a pathway provides an additional image-aligned spatial prior and can therefore suppress artifacts in the target feature map. ViT-Up follows a different design: it directly reconstructs the target ViT feature representation without a separate image encoder. Consequently, if the target backbone features contain position artifacts or spatial leakage, these structures can enter both the query and key features. Moreover, our reconstruction objective treats them as part of the target

representation. Suppressing them is therefore not explicitly encouraged; in fact, removing them can be penalized when they are present in the supervision signal.

Fig. 7 illustrates this trade-off. On DINOv2, NAF and UpLiFT suppress visible position artifacts and spatial leakage more effectively than ViT-Up. ViT-Up instead preserves the target feature field more directly, including its undesired spatial structure. This explains why image-guided methods can be favorable on DINOv2 for tasks that benefit from artifact suppression. Evaluated on linear semantic matching probing, bilinear interpolation reaches 81.22 mIoU on VOC, while UpLiFT, NAF, and ViT-Up obtain 84.76, 84.05, and 83.02 mIoU, respectively. ViT-Up therefore still improves substantially over standard bilinear interpolation, but it does not match the strongest artifact-suppressing baseline in this setting.

However, artifact suppression and feature faithfulness are not identical objectives. For semantic correspondence on SPair-71k, ViT-Up remains the strongest method even with DINOv2 features, reaching 53.75 PCK@0.1 compared to 53.08 for UpLiFT and 50.46 for NAF. This suggests that artifact suppression and feature faithfulness are not identical objectives: suppressing artifacts can improve visual cleanliness and semantic probing, whereas preserving the target feature geometry remains favorable for correspondence.

This trade-off becomes much less restrictive for modern dense-feature backbones. In contrast to DINOv2, DINOv3 uses a more suitable positional design and produces substantially cleaner dense features in Fig. 7. Rotary position embeddings encode relative spatial relations in the attention mechanism rather than adding a fixed positional vector to the token representation [41]. This reduces the need for the feature vectors themselves to carry additive position-dependent offsets, which is consistent with the lower amount of visible spatial leakage observed for DINOv3. In this cleaner regime, artifact suppression becomes less central, and faithful reconstruction becomes the more important property. This is reflected in our DINOv3-S+ results, where ViT-Up substantially outperforms image-guided upsampling methods across the main benchmarks.

Overall, DINOv2 should be interpreted as an artifact-prone stress test rather than a clean benchmark for faithful feature upsampling alone. Image-guided methods can be advantageous in this setting because they partially solve an additional denoising problem. ViT-Up instead targets faithful reconstruction of high-quality ViT feature fields, which explains both its weaker behavior under DINOv2 artifacts and its strong performance on DINOv3.

REFERENCES

- [1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
- [2] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9650–9660.

- [3] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. J´egou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.
- [4] O. Sim´eoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. J´egou, P. Labatut, and P. Bojanowski, “DINOv3,” 2025.
- [5] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [6] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, “Scene parsing through ADE20K dataset,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5122–5130.
- [7] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything V2,” in Advances in Neural Information Processing Systems, 2024.
- [8] P. Couairon, L. Chambon, L. Serrano, J.-E. Haugeard, M. Cord, and N. Thome, “JAFAR: Jack up any feature at any resolution,” in Advances in Neural Information Processing Systems, 2025.
- [9] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 3431– 3440.
- [10] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 12179–12188.
- [11] F. Li, H. Zhang, H. Xu, S. Liu, L. Zhang, L. M. Ni, and H.-Y. Shum, “Mask DINO: Towards a unified transformer-based framework for object detection and segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 3041–3050.
- [12] T. Wimmer, P. Truong, M.-J. Rakotosaona, M. Oechsle, F. Tombari, B. Schiele, and J. E. Lenssen, “Anyup: Universal feature upsampling,” in Proceedings of the International Conference on Learning Representations (ICLR), 2026.
- [13] M. Walmer, S. Suri, A. Aggarwal, and A. Shrivastava, “UPLiFT: Efficient pixel-dense feature upsampling with local attenders,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 41288–41298.
- [14] L. Chambon, P. Couairon, E.´ Zablocki, A. Boulch, N. Thome, and M. Cord, “NAF: Zero-shot feature upsampling via neighborhood attention filtering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 26604–26613.
- [15] M. El Banani, A. Raj, K.-K. Maninis, A. Kar, Y. Li, M. Rubinstein, D. Sun, L. Guibas, J. Johnson, and V. Jampani, “Probing the 3d awareness of visual foundation models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [16] J. Min, J. Lee, J. Ponce, and M. Cho, “SPair-71k: A large-scale benchmark for semantic correspondence,” 2019.
- [17] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015, pp. 234–241.
- [18] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2881–2890.
- [19] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” in Advances in Neural Information Processing Systems, vol. 34, 2021.
- [20] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked-attention mask transformer for universal image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 1290–1299.
- [21] T.-Y. Lin, P. Doll´ar, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature pyramid networks for object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2117–2125.
- [22] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, “Unified perceptual parsing for scene understanding,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 418–434.

- [23] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10012–10022.
- [24] Y. Li, H. Mao, R. Girshick, and K. He, “Exploring plain vision transformer backbones for object detection,” in European Conference on Computer Vision (ECCV), 2022, pp. 280–296.
- [25] J. Wang, K. Chen, R. Xu, Z. Liu, C. C. Loy, and D. Lin, “CARAFE: Content-aware reassembly of features,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 3007– 3016.
- [26] ——, “CARAFE++: Unified content-aware reassembly of features,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 9, pp. 4674–4687, 2022.
- [27] W. Liu, H. Lu, H. Fu, and Z. Cao, “Learning to upsample by learning to sample,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 6027–6037.
- [28] H. Lu, W. Liu, H. Fu, and Z. Cao, “Fade: A task-agnostic upsampling operator for encoder–decoder architectures,” International Journal of Computer Vision, vol. 133, no. 1, pp. 151–172, 2025.
- [29] H. Lu, W. Liu, Z. Ye, H. Fu, Y. Liu, and Z. Cao, “Sapa: Similarity-aware point affiliation for feature upsampling,” Advances in Neural Information Processing Systems, vol. 35, pp. 20889–20901, 2022.
- [30] Y. Chen, S. Liu, and X. Wang, “Learning continuous image representation with local implicit image function,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 8628–8638.
- [31] B. Lim, S. Son, H. Kim, S. Nah, and K. Mu Lee, “Enhanced deep residual networks for single image super-resolution,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017.
- [32] S. Fu, M. Hamilton, L. E. Brandt, A. Feldmann, Z. Zhang, and W. T. Freeman, “Featup: A model-agnostic framework for features at any resolution,” in Proceedings of the International Conference on Learning Representations (ICLR), 2024. [Online]. Available: https://openreview.net/forum?id=GkJiNn2QDF
- [33] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in Proceedings of the European Conference on Computer Vision (ECCV), 2020.
- [34] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.
- [35] H. Huang, A. Chen, V. Havrylov, A. Geiger, and D. Zhang, “Loftup: Learning a coordinate-based feature upsampler for vision foundation models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.
- [36] J. Kopf, M. F. Cohen, D. Lischinski, and M. Uyttendaele, “Joint bilateral upsampling,” ACM Transactions on Graphics, vol. 26, no. 3, 2007.
- [37] K. He, J. Sun, and X. Tang, “Guided image filtering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 6, pp. 1397– 1409, 2013.
- [38] K. He and J. Sun, “Fast guided filter,” in Proceedings of the European Conference on Computer Vision (ECCV), 2016.
- [39] M. Ranzinger, G. Heinrich, P. Molchanov, B. Catanzaro, and A. Tao, “FeatSharp: Your vision model features, sharper,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 51156– 51182.
- [40] S. Suri, M. Walmer, K. Gupta, and A. Shrivastava, “LiFT: A surprisingly simple lightweight feature transform for dense vit descriptors,” in Proceedings of the European Conference on Computer Vision (ECCV), 2024, pp. 110–128.
- [41] B. Heo, S. Park, D. Han, and S. Yun, “Rotary position embedding for vision transformer,” in Proceedings of the European Conference on Computer Vision (ECCV), 2024.
- [42] A. Hassani, S. Walton, J. Li, S. Li, and H. Shi, “Neighborhood attention transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
- [43] A. Hassani, F. Zhou, A. Kane, J. Huang, C.-Y. Chen, M. Shi, S. Walton, M. Hoehnerbach, V. Thakkar, M. Isaev, Q. Zhang, B. Xu, H. Wu, W.m. Hwu, M.-Y. Liu, and H. Shi, “Generalized neighborhood attention: Multi-dimensional sparse attention at the speed of light,” arXiv preprint arXiv:2504.16922, 2025.

- [44] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2018.
- [45] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.
- [46] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.
- [47] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, “Fitnets: Hints for thin deep nets,” in International Conference on Learning Representations, 2015.
- [48] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, “Do imagenet classifiers generalize to imagenet?” in Proceedings of the International Conference on Machine Learning (ICML), 2019, pp. 5389–5400.
- [49] J. Yang, K. Z. Luo, J. Li, C. Deng, L. Guibas, D. Krishnan, K. Q. Weinberger, Y. Tian, and Y. Wang, “Denoising vision transformers,” in Proceedings of the European Conference on Computer Vision (ECCV), 2024.
- [50] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the International Conference on Machine Learning (ICML), 2021, pp. 8748–8763.
- [51] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
- [52] T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, “Vision transformers need registers,” in Proceedings of the International Conference on Learning Representations (ICLR), 2024.
- [53] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL visual object classes (VOC) challenge,” International Journal of Computer Vision, vol. 88, no. 2, pp. 303–338, 2010.
- [54] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in Proceedings of the European Conference on Computer Vision (ECCV), 2014, pp. 740–755.
- [55] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [56] V. Jampani, K.-K. Maninis, A. Engelhardt, A. Karpur, K. Truong, K. Sargent, S. Popov, A. Araujo, R. Martin-Brualla, K. Patel, D. Vlasic, V. Ferrari, A. Makadia, C. Liu, Y. Li, and H. Zhou, “NAVI: Categoryagnostic image collections with high-quality 3d shape and pose annotations,” in Advances in Neural Information Processing Systems, 2023.
- [57] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, J. Lei, T. Ma, B. Guo, A. Kalla, M. Marks, J. Greer, M. Wang, P. Sun, R. R¨adle, T. Afouras, E. Mavroudi, K. Xu, T.-H. Wu, Y. Zhou, L. Momeni, R. Hazra, S. Ding, S. Vaze, F. Porcher, F. Li, S. Li, A. Kamath, H. K. Cheng, P. Doll´ar, N. Ravi, K. Saenko, P. Zhang, and C. Feichtenhofer, “SAM 3: Segment anything with concepts,” 2025.
- [58] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. R¨adle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Doll´ar, and C. Feichtenhofer, “Sam 2: Segment anything in images and videos,” in International Conference on Learning Representations (ICLR), 2025.
- [59] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Doll´ar, and R. Girshick, “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4015–4026.

