# arXiv:2603.06351v2[cs.CV]7May2026

[Figure 1]

## DC-DiT: Adaptive Compute and Elastic Inference for Visual Generation via Dynamic Chunking

##### Akash Haridas *1♣ Utkarsh Saxena1 Parsa Ashrafi Fashi1 Mehdi Rezagholizadeh1♣ Vikram Appia1 Emad Barsoum1

1Advanced Micro Devices Inc. (AMD) ♣ Correspondence to: akash.haridas@amd.com, mehdi.rezagholizadeh@amd.com.

#### Abstract

Diffusion Transformers rely on static patchify tokenization, assigning the same token budget to smooth backgrounds, detailed object regions, noisy early timesteps, and late-stage refinements. We introduce the Dynamic Chunking Diffusion Transformer (DC-DiT), which replaces fixed patchification with a learned encoderrouter-decoder scaffold that adaptively compresses the 2D input into a shorter token sequence through a chunking mechanism learned end-to-end with diffusion training. DC-DiT allocates fewer tokens to predictable regions and noisy timesteps, and more tokens to detailed regions and later refinement stages, yielding meaningful spatial segmentations and timestep-adaptive compression schedules without supervision. Furthermore, the router provides an importance ordering over retained tokens, enabling elastic inference: a single checkpoint can be evaluated at flexible compute budgets with a smooth quality-compute tradeoff. Additionally, DC-DiT can be upcycled from pretrained DiT checkpoints and is also compatible with orthogonal dynamic computation approaches. On class-conditional ImageNet generation, DC-DiT reduces inference FLOPs by up to 36.8% and improves FID by up to 37.8% over DiT baselines, yielding a stronger quality–compute Pareto frontier across model scales, resolutions, and guidance settings. More broadly, these results suggest that adaptive tokenization is a general mechanism for making visual generation both more efficient and more flexible at inference time.

### 1 Introduction

Transformer-based diffusion models [1] achieve strong image generation quality, but they commonly rely on fixed tokenization schemes: a fixed patchify operation converts every image into the same grid of tokens at every denoising step. This design makes token count an architectural constant. A uniform background patch, a high-frequency object boundary, an early noisy timestep, and a late detail-refinement timestep all receive the same token budget. This ignores two sources of natural adaptivity in image generation: different spatial regions contain different amounts of detail, and diffusion trajectories typically progress from coarse to fine structure across timesteps.

Recent work has explored dynamic computation as a way to reduce redundancy in diffusion transformers. These methods typically adapt computation after a fixed token sequence has already been formed, for example by pruning or dropping less informative tokens [2], merging redundant tokens [3], reducing intermediate feature dimensions [2], or reusing hidden states across nearby denoising steps [4]. Such approaches have shown that substantial computation can be saved along both spatial and temporal axes with limited degradation in generation quality. However, they generally operate within a fixed-tokenization regime: the initial patchify operation still maps every image into the same regular grid of tokens, independent of image content or denoising timestep. Thus, while these methods reduce computation inside or around the transformer backbone, they do not address the more fundamental limitation that the token sequence itself is statically defined before the model begins processing.

To address this limitation, we introduce the Dynamic Chunking Diffusion Transformer (DC-DiT), which replaces fixed patchification with an encoder–router–decoder scaffold learned end-to-end through diffusion training. Rather

No classifier-free guidance

With classifier-free guidance

60

80

DC-DiT

DC-DiT

22%

DiT, P=2 DiT, P=4

DiT, P=2

p=0.6

50

32%

70

29%

14%

p=0.7 p=0.9

FID-50K↓

40

p=0.4

60

p=0.2

p=0.6

20%

50

30

23%

38%

33M

40

20%

20

p=0.0 p=0.5

130M 458M 675M

33M

p=0.4

130M 458M

34% 19%

p=0.2

30

10

2 3 5 7 10 15 20

5 7 10 20 30 50 70 100

TFLOPs/img

TFLOPs/img

DiT-XL/2 DC-DiT-XL

Elastic inference

0% tokens dropped 63 TFLOPs/img

30% tokens dropped 46 TFLOPs/img

60% tokens dropped 29 TFLOPs/img

59 TFLOPs/img

|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|
|---|---|---|---|
|[Figure 6]|[Figure 7]|[Figure 8]|[Figure 9]|

- Figure 1: Left: By learning to re-allocate compute across spatial regions and timesteps, DC-DiT improves the FID–FLOPs Pareto frontier of DiT. Right: DC-DiT enables elastic inference, where a single trained checkpoint can be evaluated at multiple inference budgets while preserving visual quality.

than processing every latent position uniformly, DC-DiT selects a compact set of informative tokens for transformer computation and reconstructs the dense diffusion prediction afterward. This makes tokenization adaptive: for each input and denoising timestep, the model learns where it should allocate its token budget.

This adaptive routing mechanism learns meaningful spatial and temporal allocation patterns without explicit supervision. Spatially, smooth background regions and low-variation areas are compressed into fewer tokens, while object regions, boundaries, and high-frequency details are represented with denser token allocations. Temporally, the model naturally uses fewer tokens at noisier denoising steps and progressively increases the token budget as the sample becomes cleaner, mirroring the coarse-to-fine structure of the diffusion trajectory. We further introduce a multi-budget training objective that enables elastic inference: a single trained checkpoint can be evaluated at multiple token budgets, allowing a smooth quality–compute curve at test time. This also enables Lite-CFG, which spends more compute on the conditional branch and more aggressively compresses the unconditional branch during classifier-free guidance (CFG).

Beyond training from scratch, DC-DiT provides a practical path for upgrading existing DiT models: a pretrained fixedpatch DiT can be upcycled into DC-DiT with lightweight finetuning, retaining the benefits of pretraining while enabling adaptive tokenization. Since DC-DiT addresses token allocation before backbone computation, it remains orthogonal to post-hoc efficiency techniques such as token merging, token pruning, hidden-dimension reduction, and timestep caching and can be combined with them to enhance benefits. As a result, DC-DiT meaningfully improves the FID–FLOPs Pareto frontier across model scales, guidance settings, and resolutions, and exposes a smooth quality–compute tradeoff from a single checkpoint. Our contributions can be summarized as follows:

- • We propose DC-DiT, a Diffusion Transformer that learns to adaptively compress the 2D input into a token sequence in a data-dependent manner with a mechanism learned end-to-end during diffusion training.
- • Through multi-budget training, we enable elastic inference, which allows a single trained DC-DiT checkpoint to be evaluated at multiple token budgets, further enabling Lite-CFG, which allocates asymmetric compute in classifier-free guidance.
- • We evaluate DC-DiT across ImageNet model scales, resolutions, and guidance settings, showing meaningful improvements in the quality–compute Pareto frontier, with up to 36.8% lower inference FLOPs and up to 37.8% better FID than fixed-patch DiT baselines.
- • We demonstrate that DC-DiT extends beyond class-conditional generation by upcycling Z-Image with lightweight adaptation, and show that DC-DiT composes well with orthogonal acceleration methods such as DyDiT and TeaCache.

### 2 Related Work

Compute-adaptive Diffusion Transformers. Several methods introduce adaptivity directly into the DiT backbone architecture or training procedure. DyDiT [2] adapts hidden width across timesteps and prunes spatial tokens that are predicted to be less informative. D2iT [5] moves adaptivity into the latent representation, using a Dynamic VAE to encode different regions at different downsampling rates. Other approaches operate at the token-routing level. DiffCR [6] learns layer- and timestep-dependent compression ratios, SparseDiT [7] varies token density across network depth, and DiffMoE [8] combines token-level routing with mixture-of-experts capacity allocation. Alternative to learned adaptive architectures, many inference-time methods accelerate pretrained DiTs without additional training. Token merging [3, 9, 10, 11] reduces sequence length by combining redundant tokens, while early-exit strategies [12] skip unnecessary computation in later layers. Caching methods exploit the temporal redundancy of the denoising trajectory: TeaCache [4], BlockCache [13], TaylorSeer [14], ToCA [15], ∆-DiT [16], and HarmoniCa [17] reuse outputs, hidden states, token features, or feature differences across adjacent timesteps. Sparse-attention methods [18] reduce the cost of self-attention by limiting each token to a subset of keys and values, and structured-sparsity approaches such as Chipmunk [19] and Just-in-Time spatial acceleration [20] further exploit activation- or token-level sparsity. Together, these works show that the computational budget of DiTs can be dynamically concentrated on the most useful regions, tokens, layers, and timesteps instead of being spent uniformly throughout the generation process.

Content-Adaptive Tokenization. Beyond diffusion model architectures, a broad line of work studies tokenizers that adapt the number, size, or granularity of tokens to the input content. In vision and vision-language models, DynamicViT [21] progressively prunes tokens using learned importance scores, while APT [22] allocates variable-size patches according to local information content. Variable-length image encoders such as ALIT [23], ElasticTok [24], and DOVE [25] similarly emit content-dependent token counts, producing compact representations for simple regions and denser representations for more informative ones. Content-adaptive tokenization has also gained attention in language modeling. BLT [26] groups raw bytes into variable-length patches using entropy-based heuristics, while SuperBPE [27] learns cross-whitespace merges to reduce sequence length. More recent learned approaches such

- as H-Net [28] and DLCM [29] predict dynamic boundaries and route computation through compressed chunk- or concept-level representations. Together, these methods reduce redundancy by shifting computation from individual tokens toward higher-level, content-dependent units.

### 3 Method

DC-DiT replaces the static patch grid with a shorter data-dependent sequence, making tokenization learnable through an encoder-router-decoder scaffold. In this section, we describe the architecture of DC-DiT and its training procedure.

#### 3.1 Overall architecture

The standard DiT patchifies the input latent image into non-overlapping P × P patches, with P > 1 fixed during both training and inference [1]. In contrast, DC-DiT operates on the flattened latent grid, equivalent to patching with P = 1, and learns to dynamically group nearby latent pixels into content-dependent vision tokens jointly with diffusion training.

DC-DiT wraps a DiT backbone with an encoder-router-decoder scaffold, illustrated in Figure 2. First, an isotropic encoder mixes local information across neighboring latent tokens, producing features suitable for routing. A router then predicts boundary probabilities: tokens that are hard to predict from their neighborhood are assigned high probabilities, while locally redundant tokens are assigned lower probabilities. The elastic chunking layer retains a shortened sequence of tokens based on boundary probabilities. The shortened tokens are further processed by a DiT backbone, which comprises a series of blocks with architecture similar to [1]. Before the backbone, sinusoidal positional embeddings are added using each retained token’s original 2D grid position. After the backbone, a de-chunking layer restores the sequence to the original latent-grid resolution, and an isotropic decoder maps it back to the diffusion prediction space.

- A residual connection from the encoder output is added after de-chunking and before decoding to preserve fine-grained spatial information. This residual is gated by the router’s boundary probabilities using a straight-through estimator (STE) [30], which keeps discrete routing decisions in the forward pass while allowing gradients to flow through the routing probabilities during training. This inference procedure is repeated for each diffusion timestep.

[Figure 10]

- Figure 2: Architecture of DC-DiT. The isotropic encoder aggregates local context across the input tokens. The chunking layer selects a subset of boundary tokens via a learned routing module, yielding a compressed sequence that is processed by the DiT blocks. The de-chunking layer restores the original resolution through spatial smoothing followed by plug-back.

Overall, DC-DiT consists of: (1) an encoder that prepares local features for routing, (2) a router that assigns boundary probabilities to tokens, (3) a chunking layer that keeps boundary tokens and drops predictable non-boundaries, (4) DiT blocks operating on the shortened sequence with original-position embeddings, (5) a de-chunking layer that reconstructs full resolution, and (6) a decoder that produces the diffusion-model prediction.

#### 3.2 Encoder and Decoder

The encoder and decoder are isotropic modules that preserve the token count while mixing information over the 2D spatial grid. The encoder aggregates local context to support the router’s boundary decisions, while the decoder maps the de-chunked features back to the diffusion model’s prediction space. We instantiate these modules with convolutional residual blocks following [31]. Each block reshapes the token sequence into a 2D feature map of shape (H,W,D), applies two 3×3 convolutions with GroupNorm and SiLU activations, injects the conditioning vector after the first convolution, and adds a residual connection after the second. The result is then flattened back into a token sequence. Because the encoder and decoder operate before sequence compression and after decompression, respectively, their cost is non-negligible. To reduce overhead, both modules use an intermediate hidden width equal to one quarter of the main transformer dimension, projecting to the full dimension only at the encoder output for routing and projecting back down at the decoder input.

#### 3.3 Router and Elastic Chunking Layer

The chunking layer maps a full token sequence X ∈ RB×L×d to a shorter sequence X′ ∈ RB×M×d by selecting a subset of tokens as boundary tokens. These tokens define the retained representatives that are processed by the DiT backbone.

Routing score and boundary probability. The router predicts a boundary probability pi ∈ [0,1] for each encoded token. Its design is based on local spatial predictability: after the encoder has mixed nearby information, tokens that are difficult to predict from their neighborhood should be retained, while locally predictable tokens can be dropped and later reconstructed from nearby representatives. Concretely, we reshape the encoded sequence into an H × W feature grid and project it to a bottleneck representation z. A lightweight 3×3 convolutional predictor estimates each feature from its local spatial context, yielding zˆ. The residual r = z − zˆ measures local unpredictability. A small convolutional

scorer f(·) maps this residual to a logit, and the boundary probability at position i is

pi = σ(fscore(r)i). (1)

Elastic Chunking Layer. The chunking layer generates a hard boundary mask by thresholding pi > 0.5. To maintain differentiability and enable end to end learning, we train with a straight through estimator (STE) [30]. During the forward pass, chunking layer uses the hard boundary mask while during the backward pass, gradients are propagated as if the mask were the continuous probability pi. Without any external supervision, we observe that high-probability boundary tokens concentrate around edges, textures, and salient regions, whereas low-probability tokens typically occur in locally smooth or predictable areas.

Elastic inference via tail dropping. The router’s boundary probabilities provide more than a binary keep/drop decision: they also induce a ranking of retained tokens from most to least important. We exploit this ranking to enable elastic inference through tail dropping. Let B = {i : pi > 0.5} be the natural boundary set selected by the router. For a user-specified tail-dropping fraction ρ ∈ [0,1), we sort the selected boundaries by pi and drop the lowest ρ|B| of them before packing the sequence for the inner DiT blocks. The retained set Bρ ⊆ B preserves the router’s most confident representatives while increasing the effective compression ratio from L/|B| to L/|Bρ|. As a result, a single trained checkpoint can be evaluated at multiple compute budgets by changing only ρ, without modifying model weights.

Lite CFG. During Classifier Free Guidance (CFG), each sampling step evaluates both conditional and unconditional branches. We introduce Lite-CFG, which leverages DC-DiT’s elastic budget control to assign a conservative tail-dropping fraction to the conditional branch and a larger drop fraction to the unconditional branch. This spends most of the token budget on the branch that carries class information, while reducing the cost of the unconditional prediction used for guidance.

Batched chunking and sequence packing. During batched training and inference, each sample may have a different number of boundary tokens M selected by the router. To enable efficient processing and avoid wasting compute from padding, we apply sequence packing: the valid boundary tokens from all samples are concatenated along the sequence dimension and the per-sample boundaries are handled by the variable-length attention kernel of FlashAttention [32]. The pointwise components of the inner DiT blocks (linear layers, MLP, normalization) operate on the packed tensor at no extra cost. After the main network, we unpack back to a (B,Mmax,D) tensor for de-chunking. This eliminates wasted FLOPs on padding tokens and keeps the realized inference cost proportional to the average compression ratio rather than the worst-case per-batch token count.

#### 3.4 De-chunking Layer

After the inner network processes the shortened sequence, we reconstruct the token sequence back to its original resolution via a de-chunking layer with two conceptual components: smoothing over boundary-token representations and a plug-back map that assigns each original token position a boundary-derived representation.

Motivation for smoothing. Hard keep/drop decisions can make chunk assignments unstable: small changes in router probabilities may shift a boundary and abruptly reassign many positions to a different retained token. This is especially common early in training, when probabilities often lie near the threshold. To improve stability, we smooth the reconstructed representation using the router’s confidence rather than relying solely on the hard mask. High-confidence retained tokens are treated as reliable representatives, while low-confidence retained tokens are blended with nearby retained tokens, reducing discontinuities when the router is uncertain.

Spatial smoothing. Let S = {s1,...,sM} be the retained boundary indices after chunking. For each retained token si, let hi ∈ RD denote its representation after the DiT backbone, ui = (ri,ci) its original 2D grid coordinate, and pi its router probability. Smoothing operates only over these M retained tokens. We compute pairwise squared distances d2ij = ∥ui − uj∥22 and use a Gaussian kernel weighted by the source token’s confidence:

d2ij 2σ2 · pj, W˜ij =

Wij k Wik

Wij = exp −

.

For each retained token, we compute a neighborhood-smoothed representation h˜i = j W˜ijhj and blend it with the original representation according to the target token’s confidence:

houti = pi hi + (1 − pi)h˜i. (2)

High-confidence boundaries retain their original features, while low-confidence boundaries are smoothed toward their spatial neighbors. This confidence-weighted blend makes uncertain representatives less sensitive to a single hard routing decision: they borrow context from nearby retained tokens, whereas confident representatives are passed through with little change.

The plug-back map then reconstructs the full L-token grid by assigning each original grid position the representation of its spatially nearest boundary (Euclidean distance on the 2D grid). Thus, each dropped token receives the smoothed representation of its closest retained representative, while retained tokens plug back their own smoothed features at their original grid locations.

#### 3.5 Training objective and multi-budget training

We train DC-DiT with the same diffusion training objective as in DiT [1]. In addition, we include a lightweight regularizer following the load balancing mechanism of Mixture-of-Experts models [33] that encourages a target average downsampling factor for the routing module. We denote this target compression ratio by N > 1; N is a training hyperparameter that specifies the desired average ratio between the original sequence length and the retained boundary-token length. Concretely, for a routing module output with boundary mask m ∈ {0,1}B×L and boundary probabilities p ∈ [0,1]B×L, we define rˆ = E[m] and p¯ = E[p]. We use the following regularizer:

N N − 1

Lratio =

((1 − rˆ)(1 − p¯) + (N − 1)ˆrp¯).

Multi-budget training. Applying tail dropping only at inference would create a train–test mismatch: the inner DiT blocks would be trained only on the router’s natural budget and then evaluated on more aggressively compressed sequences. We therefore train DC-DiT across several tail-dropping settings. Let R = {ρ1,ρ2,...,ρK} denote a fixed set of tail-dropping fractions, including ρ = 0. Conceptually, we optimize the average diffusion objective over these budgets,

1 K ρ∈RL(diffusionρ) ,

Lmb =

where L(diffusionρ) is the standard diffusion loss evaluated after applying tail dropping with fraction ρ. In practice, after an initial warmup phase we optimize an unbiased stochastic estimate of this objective by sampling one ρ ∼ R per iteration

and applying exactly the same tail-dropping path used at inference. The ratio loss Lratio, however, is always computed on the router’s natural boundary set before tail dropping, which keeps the router anchored to the target compression factor N while ρ provides an additional inference-time budget on top of the learned routing policy. The resulting training objective is

L = Lmb + λLratio.

### 4 Experiments and Results

We evaluate DC-DiT on class-conditional ImageNet generation at 256px and 512px resolutions and compare quality– compute tradeoff against DiT baselines that use fixed patchification. Furthermore, we evaluate DC-DiT’s composability with other dynamic computation and FLOPs-saving techniques such as DyDiT [2] and TeaCache [4]. Additionally, we demonstrate lightweight upcycling of the text-to-image foundation model Z-Image [34] into a DC-DiT to enable elastic inference while preserving high-quality image generation and prompt following.

|[Figure 11]|[Figure 12]|[Figure 13]|[Figure 14]|[Figure 15]|
|---|---|---|---|---|
|[Figure 16]|[Figure 17]|[Figure 18]|[Figure 19]|[Figure 20]|

xt Boundary

prob.

22.3% tokens 24.4% tokens 30.2% tokens 37.1% tokens 45.1% tokens

Adaptive compute

Noisy timestep Less compute

Clean timestep More compute

1.0

|[Figure 21]|
|---|

Boundaryprobability

0.8

0.6

0.4

0.2

0.0

- Figure 3: Adaptive compute allocation learned by DC-DiT. Across diffusion timesteps, the router retains boundary tokens in spatially informative regions such as object structure, edges, and texture, while compressing more predictable background regions. The retained-token pattern changes over the denoising trajectory, allocating compute differently as images evolve from noisy structure to fine detail.

#### 4.1 Experimental setup

For our primary experiments on class-conditional ImageNet generation, we report FID-50K as our generation quality metrics. We use the same diffusion formulation as standard DiT [1]: a linear noise schedule with 1000 diffusion steps during training, and DDPM sampling with 250 steps. All models operate in the latent space of a pretrained Stable Diffusion VAE encoder [31], with class conditioning via adaLN-Zero.

Model configurations. We train DC-DiT at multiple model scales corresponding to the S, B, L, and XL variants of DiT. For each scale, we keep the transformer backbone identical to the corresponding DiT baseline and augment it with the encoder-router-decoder scaffold. This scaffold adds a small FLOP overhead relative to the matched DiT backbone, but unlike fixed-patch DiT, DC-DiT can be evaluated across a range of inference budgets by varying the tail-dropping fraction ρ. We train DC-DiT with target compression ratio N=4, corresponding to patch size P=2 in standard DiT.

Training. All models are trained with a global batch size of 256 using AdamW with learning rate 1×10−4. The DiT baselines are trained for 400K steps. For DC-DiT, we use multi-budget training and sample the tail-dropping fraction ρ from {0.0,0.1,0.2,0.3,0.4,0.5,0.6}. Because tail dropping reduces the effective FLOPs of a DC-DiT training step, we extend DC-DiT training so that its total training compute matches the corresponding baseline budget (details in Appendix A.3). We set the ratio-loss weight to λ=0.03 based on a grid search. All models are trained on AMD Instinct MI325X and MI300X GPUs.

#### 4.2 Main Results

Table 1 presents the main ImageNet results across model scales, guidance settings, and resolutions. Across these settings, DC-DiT improves the FID–FLOPs Pareto frontier relative to fixed-patch DiT baselines: near the DiT compute budget, dynamic chunking yields better FID, while more aggressive tail dropping exposes substantially cheaper operating points from the same checkpoint. The benefits of DC-DiT are emphasized under Lite-CFG, where DC-DiT improves FID by up to 37.8% while reducing inference compute, and at 512×512, where DC-DiT-XL reduces compute by 36.8% with only a small FID tradeoff. Overall, we observe highly meaningful gains in compute reduction and FID scores for the S,

- B and L model scales while the XL model scale achieves competitive performance to baseline DiT while utilizing lower FLOPs.

- Table 1: Main ImageNet results across model scale, guidance setting, and resolution. In Lite-CFG rows, the tail-drop value applies to the unconditional branch.

Scale Model Params (M) Tail drop TFLOPs/img ↓ FID-50K ↓

Scale Model Params (M) Tail drop TFLOPs/img ↓ FID-50K ↓

256×256, no classifier-free guidance S DiT-S/2 33 – 3.02 68.40 S DC-DiT-S 34.7 0.2 3.06 58.43 (-14.6%) S DC-DiT-S 34.7 0.5 2.14 (-29.1%) 66.56 B DiT-B/2 131 – 11.46 43.47 B DC-DiT-B 137 0.2 11.94 33.91 (-22.0%) B DC-DiT-B 137 0.5 8.55 (-25.4%) 44.64 L DiT-L/2 459 – 40.21 23.33 L DC-DiT-L 469 0.1 39.61 22.32 (-4.3%) L DC-DiT-L 469 0.3 31.86 (-20.8%) 23.94 XL DiT-XL/2 675 – 59.00 19.47 XL DC-DiT-XL 689 0 62.99 19.73 XL DC-DiT-XL 689 0.2 51.60 20.45 XL DC-DiT-XL 689 0.4 40.26 (-31.8%) 22.52

256×256, Lite-CFG with CFG = 1.25 S DiT-S/2 33 – 6.04 54.98 S DC-DiT-S 34.7 0.9 4.73 (-21.7%) 38.99 (-29.1%)

B DiT-B/2 131 – 22.93 29.72 B DC-DiT-B 137 0.9 18.38 (-19.8%) 18.49 (-37.8%)

L DiT-L/2 459 – 80.41 12.59 L DC-DiT-L 469 0.9 52.88 (-34.2%) 10.17 (-19.2%)

XL DiT-XL/2 675 – 118.13 9.45 XL DC-DiT-XL 689 0.9 76.13 (-35.6%) 8.80 (-6.9%)

512×512, no classifier-free guidance B DiT-B/2† 131 – 53.09 49.58 B DC-DiT-B 139 0.2 52.60 42.64 (-14.0%) B DC-DiT-B 139 0.5 35.94 (-32.3%) 48.40 XL DiT-XL/2 675 – 261.25 20.13 XL DC-DiT-XL 693 0.2 228.75 (-12.4%) 23.59

###### 512×512, Lite-CFG with CFG = 1.25

XL DiT-XL/2 675 – 522.50 11.04 XL DC-DiT-XL 693 0.9 330.00 (-36.8%) 12.50

Learned spatio-temporal compression. Figure 3 visualizes the router’s boundary predictions on a representative ImageNet sample. Spatially, the router assigns high boundary probability to object edges, fine textures, and regions of high local variation, while dropping tokens in uniform backgrounds and other predictable areas. Temporally, it retains fewer tokens at early, noisier denoising steps and more tokens at later steps, when fine details are resolved. Thus, DC-DiT learns an implicit content- and timestep-adaptive tokenization: simple background regions and noisy early states are compressed more aggressively, whereas detailed object regions and later denoising stages receive more tokens. This behavior emerges solely from the diffusion training objective, without explicit supervision for segmentation, boundary detection, or timestep-dependent compute scheduling. It is consistent with the coarse-to-fine allocation studied in prior work on elastic visual generation, such as ELIT [35] and MaGNeTS [36], but arises naturally rather than being manually prescribed.

#### 4.3 Ablations

We ablate DC-DiT’s main components and evaluate robustness under increasingly aggressive tail dropping.

TeaCache

DyDiT

75

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | |ea|C|ac|he| | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | |C|Di|T|+|Te|aC|ac|he| | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

- Table 3 show that multi-budget training, the spatial predictability router, and de-chunk smoothing all improve generation quality. The spatial predictability router outperforms a direct 2D adaptation of the HNet cosine-similarity router [28]. Finally, removing de-chunk smoothing worsens FID, consistent with its role in stabilizing hard routing decisions and improving the quality of the learned segmentations. Multi-budget training is particularly important for elastic inference: by randomly dropping low-confidence retained tokens during training, it encourages reconstruction-critical tokens to appear earlier in the router ranking, leading to more graceful degradation as the inference drop ratio ρ increases and slightly better FID even at ρ = 0 as shown in Figure 6.

DyDiT

DC-DiT + DyDiT

70

FID-50K

65

60

55

1 2 3 4

1.8 2.0 2.2 2.4

TFLOPs/img

TFLOPs/img

Figure 4: DCDiT is complementary to TeaCache and DyDiT, improving FID at comparable compute budgets.

†At 512px, DC-DiT-S and DC-DiT-L trained stably, whereas the corresponding vanilla DiT-S/2 and DiT-L/2 baselines repeatedly encountered loss divergence despite multiple restarts.

- Table 2: DPG-Bench preservation after upcycling Z-Image. DC-Z-Image preserves prompt-following quality across elastic inference budgets while reducing inference compute at higher tail-dropping fractions.

Model Tail drop TFLOPs/img Relation Entity Other Attribute Global DPG-Bench

Z-Image – 5603 92.72 89.95 90.02 91.08 93.06 85.97 DC-Z-Image 0 5901 93.15 92.38 89.47 90.72 91.03 86.82 DC-Z-Image 30% 4211 92.89 91.70 88.15 92.02 89.34 87.64 DC-Z-Image 60% 2570 92.03 90.79 92.91 89.23 91.64 86.32

Z-Image DC-Z-Image Elastic inference Adaptive compute

0% tokens dropped 5901 TFLOPs/img

30% tokens dropped 4211 TFLOPs/img Boundary prob. map

5603 TFLOPs/img

|[Figure 22]|[Figure 23]|[Figure 24]|[Figure 25]|
|---|---|---|---|
|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|
|[Figure 30]|[Figure 31]|[Figure 32]|[Figure 33]|

Figure 5: Qualitative elastic inference examples after upcycling Z-Image. DC-Z-Image preserves high-quality, prompt-aligned generations across compute budgets.

Ablation TFLOPs/img FID-50K

DC-DiT-S 3.69 56.73 No multi-budget training 3.67 57.77 Cosine similarity router 3.67 57.96 No dechunk smoothing 3.47 59.79

Table 3: Component ablations on DC-DiT-S. Removing any critical component meaningfully worsens FID.

110

Multi-budget training

100

No multi-budget training

FID-50K↓

90

80

70

60

0 .1 .2 .3 .4 .5 .6

Tail-dropping fraction ρ

Figure 6: Tail-dropping robustness on DCDiT-S. Multi-budget training keeps FID stable as ρ increases.

#### 4.4 Upcycling Z-Image to DC-Z-Image

To test whether dynamic chunking scales beyond class-conditional ImageNet models, we apply DC to Z-Image [34], a state-of-the-art text-to-image diffusion-transformer trained with flow matching. Instead of training from scratch, we upcycle the Z-Image model to DC-Z-Image by replacing fixed patchification with the encoder-router-decoder scaffold and performing lightweight adaptation. Our upcycling uses 5K steps of distillation following the grafting-style adaptation outlined in [37] and 10K flow-matching steps on 1M synthetic Z-Image samples with prompts from Recap-DataComp-1B [38]. During upcycling, we freeze the timestep and text-embedding modules and add a trainable LayerNorm adaptor to the encoder/decoder conditioning vector. As shown in Table 2 and Figure 5, DC-Z-Image preserves prompt-following quality across elastic inference budgets: increasing tail dropping reduces cost from 5901 to 2570 TFLOPs/img, while maintaining DPG-Bench scores comparable to the original Z-Image baseline.

#### 4.5 Composability with other dynamic computation techniques

DC-DiT introduces content-adaptive patchification while leaving the DiT backbone unchanged, making it compatible with orthogonal dynamic execution methods [2, 3, 4]. To demonstrate composability, we combine DC-DiT with DyDiT [2] and TeaCache [4]. DyDiT adds lightweight learned gates to modulate backbone computation across timesteps, while TeaCache is a training-free method that reuses model outputs across denoising steps. We apply both methods to

the DiT backbone inside DC-DiT to further reduce FLOPs. As shown in Figure 4, DC-DiT remains compatible with both approaches, achieving additional compute reductions while preserving or improving FID.

### 5 Conclusion

We introduced DC-DiT, a diffusion transformer that replaces fixed patchification with adaptive tokenization learned end-to-end through diffusion training. Its router reallocates compute across spatial regions and denoising timesteps without explicit supervision, while multi-budget training turns the same routing signal into elastic inference and Lite-CFG. Across ImageNet settings, DC-DiT improves the quality–compute Pareto frontier, reducing inference FLOPs by up to 36.8% and improving FID by up to 37.8%. The approach also upcycles Z-Image with lightweight adaptation and composes well with DyDiT and TeaCache. These results position adaptive tokenization as a practical primitive for efficient diffusion models.

### References

- [1] William Peebles and Saining Xie. Scalable diffusion models with transformers. In International Conference on Computer Vision (ICCV), 2023.
- [2] Wangbo Zhao, Yizeng Han, Jiasheng Tang, Kai Wang, Yibing Song, Gao Huang, Fan Wang, and Yang You. Dynamic diffusion transformer, 2024. URL https://arxiv.org/abs/2410.03456.
- [3] Haoyu Wu, Jingyi Xu, Hieu Le, and Dimitris Samaras. Importance-based token merging for efficient image and video generation, 2025. URL https://arxiv.org/abs/2411.16720.
- [4] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. URL https: //arxiv.org/abs/2411.19108.
- [5] Weinan Jia, Mengqi Huang, Nan Chen, Lei Zhang, and Zhendong Mao. D2it: Dynamic diffusion transformer for accurate image generation, 2025. URL https://arxiv.org/abs/2504.09454.
- [6] Haoran You, Connelly Barnes, Yuqian Zhou, Yan Kang, Zhenbang Du, Wei Zhou, Lingzhi Zhang, Yotam Nitzan, Xiaoyang Liu, Zhe Lin, Eli Shechtman, Sohrab Amirghodsi, and Yingyan Celine Lin. Layer- and timestep-adaptive differentiable token compression ratios for efficient diffusion transformers, 2024. URL https://arxiv.org/abs/2412.16822.
- [7] Shuning Chang, Pichao Wang, Jiasheng Tang, and Yi Yang. Sparsedit: Token sparsification for efficient diffusion transformer, 2024. URL https://arxiv.org/abs/2412.06028.
- [8] Minglei Shi, Ziyang Yuan, Haotian Yang, Xintao Wang, Mingwu Zheng, Xin Tao, Wenliang Zhao, Wenzhao Zheng, Jie Zhou, Jiwen Lu, Pengfei Wan, Di Zhang, and Kun Gai. Diffmoe: Dynamic token selection for scalable diffusion transformers, 2025. URL https://arxiv.org/abs/2503.14487.
- [9] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.
- [10] Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4599–4603, 2023.
- [11] Haipeng Fang, Sheng Tang, Juan Cao, Enshuo Zhang, Fan Tang, and Tong-Yee Lee. Attend to not attended: Structure-then-detail token merging for post-training dit acceleration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18083–18092, 2025.

- [12] Taehong Moon, Moonseok Choi, EungGu Yun, Jongmin Yoon, Gayoung Lee, Jaewoong Cho, and Juho Lee. A simple early exiting framework for accelerated sampling in diffusion models. arXiv preprint arXiv:2408.05927, 2024.
- [13] Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6211–6220, 2024.
- [14] Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From reusing to forecasting: Accelerating diffusion models with taylorseers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15853–15863, 2025.
- [15] Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. arXiv preprint arXiv:2410.05317, 2024.
- [16] Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. Delta-dit: A training-free acceleration method tailored for diffusion transformers. arXiv preprint arXiv:2406.01125, 2024.
- [17] Yushi Huang, Zining Wang, Ruihao Gong, Jing Liu, Xinjie Zhang, Jinyang Guo, Xianglong Liu, and Jun Zhang. Harmonica: Harmonizing training and inference for better feature caching in diffusion transformer acceleration. arXiv preprint arXiv:2410.01723, 2024.
- [18] Sucheng Ren, Qihang Yu, Ju He, Alan Yuille, and Liang-Chieh Chen. Grouping first, attending smartly: Training-free acceleration for diffusion transformers, 2025. URL https://arxiv.org/abs/2505.14687.
- [19] Austin Silveria, Soham V Govande, and Daniel Y Fu. Chipmunk: Training-free acceleration of diffusion transformers with dynamic column-sparse deltas. In ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models, 2025.
- [20] Wenhao Sun, Ji Li, and Zhaoqiang Liu. Just-in-time: Training-free spatial acceleration for diffusion transformers. arXiv preprint arXiv:2603.10744, 2026.
- [21] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification, 2021. URL https://arxiv.org/abs/2106.02034.
- [22] Rohan Choudhury, JungEun Kim, Jinhyung Park, Eunho Yang, L´aszl´o A. Jeni, and Kris M. Kitani. Accelerating vision transformers with adaptive patch sizes, 2025. URL https://arxiv.org/abs/2510.18091.
- [23] Shivam Duggal, Phillip Isola, Antonio Torralba, and William T. Freeman. Adaptive length image tokenization via recurrent allocation, 2024. URL https://arxiv.org/abs/2411.02393.
- [24] Wilson Yan, Matei Zaharia, Volodymyr Mnih, Pieter Abbeel, Aleksandra Faust, and Hao Liu. Elastictok: Adaptive tokenization for image and video. ArXiv, abs/2410.08368, 2024. URL https://arxiv.org/abs/2410. 08368.
- [25] Lingjun Mao, Rodolfo Corona, Xin Liang, Wenhao Yan, and Zineng Tang. Images are worth variable length of representations, 2025. URL https://arxiv.org/abs/2506.03643.
- [26] Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, and Srinivasan Iyer. Byte latent transformer: Patches scale better than tokens, 2024. URL https://arxiv.org/abs/2412.09871.
- [27] Alisa Liu, Noah A. Smith, Jonathan Hayase, Yejin Choi, Sewoong Oh, and Valentin Hofmann. Superbpe: Space travel for language models, 2025. URL https://arxiv.org/abs/2503.13423.
- [28] Sukjun Hwang, Brandon Wang, and Albert Gu. Dynamic chunking for end-to-end hierarchical sequence modeling. arXiv preprint arXiv:2507.07955, 2025.

- [29] Xingwei Qu, Shaowen Wang, Zihao Huang, Kai Hua, Fan Yin, Rui-Jie Zhu, Jundong Zhou, Qiyang Min, Zihao Wang, Yizhi Li, Tianyu Zhang, He Xing, Zheng Zhang, Yuxuan Song, Tianyu Zheng, Zhiyuan Zeng, Chenghua Lin, Ge Zhang, and Wenhao Huang. Dynamic large concept models: Latent reasoning in an adaptive semantic space, 2025. URL https://arxiv.org/abs/2512.24617.
- [30] Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.
- [32] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-efficient exact attention with IO-awareness, 2022. URL https://arxiv.org/abs/2205.14135.
- [33] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.
- [34] Z-Image Team, Alibaba Group. Z-image: An efficient image generation foundation model with single-stream diffusion transformer, 2025. URL https://arxiv.org/abs/2511.22699.
- [35] Moayed Haji-Ali, Willi Menapace, Ivan Skorokhodov, Dogyun Park, Anil Kag, Michael Vasilkovsky, Sergey Tulyakov, Vicente Ordonez, and Aliaksandr Siarohin. One model, many budgets: Elastic latent interfaces for diffusion transformers. arXiv preprint arXiv:2603.12245, 2026.
- [36] Sahil Goyal, Debapriya Tula, Gagan Jain, Pradeep Shenoy, Prateek Jain, and Sujoy Paul. Masked generative nested transformers with decode time scaling. arXiv preprint arXiv:2502.00382, 2025.
- [37] Keshigeyan Chandrasegaran, Michael Poli, Daniel Y. Fu, Dongjun Kim, Lea M. Hadzic, Manling Li, Agrim Gupta, Stefano Massaroli, Azalia Mirhoseini, Juan Carlos Niebles, Stefano Ermon, and Fei-Fei Li. Exploring diffusion transformer designs via grafting, 2025. URL https://arxiv.org/abs/2506.05340.
- [38] Xianhang Li, Haoqin Tu, Mude Hui, Zeyu Wang, Bingchen Zhao, Junfei Xiao, Sucheng Ren, Jieru Mei, Qing Liu, Huangjie Zheng, Yuyin Zhou, and Cihang Xie. What if we recaption billions of web images with llama-3? arXiv preprint arXiv:2406.08478, 2024. URL https://arxiv.org/abs/2406.08478.

### A Additional Visual Results and Reproducibility Details

#### A.1 Additional visual results

Figure 7 provides an extended version of the qualitative elastic inference grid shown in Figure 1. Figure 8 provides an extended version of the adaptive compute visualization shown in Figure 3, with additional samples illustrating the learned spatial chunking and timestep-dependent token allocation.

#### A.2 FLOPs accounting

We report TFLOPs/img as the total floating-point operation count required to generate one image with the sampling protocol used for evaluation. For unguided ImageNet sampling, this is the sum of the per-forward cost across the 250 DDPM sampling steps. For standard classifier-free guidance, each denoising step includes both conditional and unconditional model evaluations. For Lite-CFG, we count these two branches separately, using the conditional branch cost at its conservative tail-dropping budget and the unconditional branch cost at the more aggressive tail-dropping budget reported in Table 1.

For each model forward pass, we count all matrix multiplications, convolutions, normalizations, elementwise operations, embeddings, and output projections. This includes the overhead introduced by the encoder-router-decoder scaffold. Consequently, the reported FLOPs reflect the realized end-to-end cost of DC-DiT at a given tail-dropping fraction.

DiT-XL/2 DC-DiT-XL

Elastic inference

0% tokens dropped 63 TFLOPs/img

30% tokens dropped 46 TFLOPs/img

60% tokens dropped 29 TFLOPs/img

59 TFLOPs/img

|[Figure 34]|[Figure 35]|[Figure 36]|[Figure 37]|
|---|---|---|---|
|[Figure 38]|[Figure 39]|[Figure 40]|[Figure 41]|
|[Figure 42]|[Figure 43]|[Figure 44]|[Figure 45]|
|[Figure 46]|[Figure 47]|[Figure 48]|[Figure 49]|
|[Figure 50]|[Figure 51]|[Figure 52]|[Figure 53]|

- Figure 7: Extended qualitative elastic inference results. This figure expands the tail-dropping grid from Figure 1, showing additional generations from the same checkpoint across user-selected compute budgets.

|[Figure 54]|[Figure 55]|[Figure 56]|[Figure 57]|[Figure 58]|
|---|---|---|---|---|
|[Figure 59]|[Figure 60]|[Figure 61]|[Figure 62]|[Figure 63]|

xt Boundary

prob.

21.5% tokens 21.9% tokens 27.5% tokens 34.7% tokens 37.6% tokens

Adaptive compute

Noisy timestep Less compute

Clean timestep More compute

1.0

|[Figure 64]|
|---|

Boundaryprobability

0.8

0.6

0.4

0.2

0.0

- Figure 8: Extended adaptive compute visualization. This figure expands Figure 3 with additional examples of the router’s learned boundary predictions across diffusion timesteps.

Variable-length sequence packing is accounted for using the actual retained sequence length of each sample. Let Lb denote the number of retained tokens for sample b, H the number of attention heads, and dh the head dimension. Pointwise components of the packed DiT blocks, such as linear layers, MLPs, and normalization, scale with b Lb. Self-attention is counted using the packed variable-length cost

###### H

b

L2b(4dh + 3),

rather than the padded cost BHMmax2 (4dh + 3), where Mmax = maxb Lb. This distinction is important because DC-DiT samples in a batch may retain different numbers of tokens; padding to the longest sequence would overestimate the cost and would not match the packed attention computation used during inference. For fixed-patch DiT baselines, Lb is constant across the batch and the expression reduces to the standard dense attention cost.

#### A.3 Multi-budget training and compute matching

DC-DiT uses multi-budget training so that a single checkpoint can be evaluated at several tail-dropping fractions. We first train for a 5K-step warmup with no tail dropping, which lets the router converge toward the target compression ratio before exposing the inner DiT blocks to more aggressively compressed sequences. After this warmup, each training step samples ρ uniformly from R = {0.0,0.1,0.2,0.3,0.4,0.5,0.6} and applies the same tail-dropping path used at inference.

Because tail dropping changes the cost of a DC-DiT training step, we choose the total number of DC-DiT steps to match the training compute of the corresponding 400K-step DiT baseline. Let SDiT = 400,000, Swarm = 5,000, FDiT be the measured DiT FLOPs per image, F0 be the measured DC-DiT FLOPs per image during warmup, and F¯mb be the average measured DC-DiT FLOPs per image over the sampled tail-dropping budgets in R. We set

SDiTFDiT − SwarmF0 F¯mb

SDC = Swarm +

,

rounded to the nearest integer. Table 4 reports the resulting training lengths for the ImageNet settings that have corresponding matched DiT baselines in Table 1.

- Table 4: Compute-matched training lengths for the ImageNet DC-DiT experiments. DC-DiT trains longer than 400K steps when multi-budget tail dropping reduces the average cost per post-warmup step.

Resolution Scale DiT TFLOPs/img DiT steps DC-DiT steps

256×256 S 3.02 400K 421,230 256×256 B 11.46 400K 408,071 256×256 L 40.21 400K 486,372 256×256 XL 59.00 400K 494,943 512×512 B 53.09 400K 432,097 512×512 XL 261.25 400K 495,365

- A.4 Architecture details by scale

Table 5 summarizes the basic shape parameters for the DC-DiT scale configurations used in the paper. The encoder/decoder bottleneck width is computed from the configured hidden width and dimension reduction factor.

Table 5: Basic architecture shapes for the DC-DiT scale configurations.

Configuration Enc. blocks Transformer blocks Dec. blocks Hidden width Heads Enc./Dec. bottleneck DC-DiT-S 2 12 2 384 6 96 DC-DiT-B 2 12 2 768 12 192 DC-DiT-L 2 24 2 1024 16 256 DC-DiT-XL 2 28 2 1152 16 288

- A.5 Hyperparameters

- Table 6 lists the main hyperparameters used for the ImageNet DC-DiT-B N = 4 experiments. Unless otherwise noted, the same optimizer, diffusion, routing, and multi-budget settings are used across ImageNet model scales.

Table 6: Main hyperparameters for ImageNet DC-DiT training.

Hyperparameter Value Dataset ImageNet, 1000 classes Latent resolution 256×256 Diffusion schedule Linear, 1000 training steps Sampler DDPM, 250 sampling steps Global batch size 256 Optimizer AdamW Learning rate 1×10−4 Weight decay 0 EMA decay 0.9999 Gradient clipping 1.0 Classifier-free dropout 0.1 Variance prediction Learned σ Target compression N = 4 Ratio loss weight 0.03 Ratio-loss batch size 16 Router Spatial predictability De-chunk smoothing Enabled, Gaussian σ = 1.0 Multi-budget warmup 5K steps, no tail dropping Tail-drop fractions {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}

