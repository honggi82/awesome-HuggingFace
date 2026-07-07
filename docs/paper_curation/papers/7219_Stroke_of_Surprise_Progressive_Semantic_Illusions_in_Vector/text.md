## Stroke of Surprise: Progressive Semantic Illusions in Vector Sketching

HUAI-HSUN CHENG, National Yang Ming Chiao Tung University, Taiwan SIANG-LING ZHANG, National Yang Ming Chiao Tung University, Taiwan YU-LUN LIU, National Yang Ming Chiao Tung University, Taiwan

#### (a) Input: Pair of Text Prompts

- A: “pig” &
- B: “angel”

- A: “chicken” &
- B: “monkey”

- A: “duck” &
- B: “sheep”

A: “swan” & B: “hamburger”

- A: “rabbit” &
- B: “Einstein”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

# arXiv:2602.12280v2[cs.CV]16May2026

- (b)Initial

GeneratedSketch

- (c)Further

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

GeneratedStrokes

(d) Output: Progressive Semantic Illusion A→B

Fig. 1. Progressive semantic illusions from text. Given a pair of text prompts (a), our method generates a vector sketch that evolves over time. The initial generated sketch (b) depicts the first concept (e.g., “pig”). By adding further generated strokes (c), the drawing is transformed into a totally different object (e.g., “angel”). This creates a Stroke of Surprise: the process subverts the viewer’s expectation of the initial concept, triggering a dramatic semantic reversal as the final strokes re-contextualize the entire composition.

Visual illusions traditionally rely on spatial manipulations such as multiview consistency. In this work, we introduce Progressive Semantic Illusions, a novel vector sketching task where a single sketch undergoes a dramatic semantic transformation through the sequential addition of strokes. We present Stroke of Surprise, a generative framework that optimizes vector strokes to satisfy distinct semantic interpretations at different drawing stages. The core challenge lies in the “dual-constraint”: initial prefix strokes must form a coherent object (e.g., a duck) while simultaneously serving as the structural foundation for a second concept (e.g., a sheep) upon adding delta strokes. To address this, we propose a sequence-aware joint optimization

framework driven by a dual-branch Score Distillation Sampling (SDS) mechanism. Unlike sequential approaches that freeze the initial state, our method dynamically adjusts prefix strokes to discover a “common structural subspace” valid for both targets. Furthermore, we introduce a novel Overlay Loss that enforces spatial complementarity, ensuring structural integration rather than occlusion. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art baselines in recognizability and illusion strength, successfully expanding visual anagrams from the spatial to the temporal dimension. Project page: https://stroke-of-surprise.github.io/

CCS Concepts: • Computing methodologies → Shape modeling; Computer vision; Neural networks.

Authors’ Contact Information: Huai-Hsun Cheng, National Yang Ming Chiao Tung University, Taiwan, huaish.cs13@nycu.edu.tw; Siang-Ling Zhang, National Yang Ming Chiao Tung University, Taiwan, siang1105.cs13@nycu.edu.tw; Yu-Lun Liu, National Yang Ming Chiao Tung University, Taiwan, yulunliu@cs.nycu.edu.tw.

##### ACM Reference Format:

Huai-Hsun Cheng, Siang-Ling Zhang, and Yu-Lun Liu. 2026. Stroke of Surprise: Progressive Semantic Illusions in Vector Sketching. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers ’26), July 19–23, 2026, Los Angeles, CA, USA. ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3799902. 3811150

This work is licensed under a Creative Commons Attribution 4.0 International License. SIGGRAPH Conference Papers ’26, Los Angeles, CA, USA

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2554-8/2026/07 https://doi.org/10.1145/3799902.3811150

baselines fail to find a ”Common Subspace”, which is a shared geometric configuration valid for both semantic interpretations.

|[Figure 11]|
|---|
|[Figure 12]|

|[Figure 13]|
|---|
|[Figure 14]|

|[Figure 15]|
|---|
|[Figure 16]|

|[Figure 17]|
|---|
|[Figure 18]|

|[Figure 19]<br><br>“rabbit”|
|---|
|[Figure 20]<br><br>“elephant”|

- Phase1:

“rabbit”

- Phase2:

To overcome these limitations, we propose Stroke of Surprise, a sequence-aware joint optimization framework designed to discover this common structural subspace (Fig. 3). Unlike sequential approaches, we optimize parameters for both the prefix (Object “A”) and full phase (Object “B”) simultaneously using a dual-branch Score Distillation Sampling (SDS) mechanism. This guidance ensures prefix strokes are recognizable as the initial concept yet “primed” for re-interpretation. Furthermore, we introduce a geometric Overlay Loss to enforce spatial complementarity and prevent occlusion. This enables delta strokes to structurally integrate with and recontextualize the prefix. For example, it can transform pig ears into angel wings, creating a seamless illusion.

“elephant”

[Figure 21]

[Figure 22]

[Figure 23]

❌ Destruc ve Edi ng ✅ Dual-Semantic Coherency

❌ Semantic Noise

Nano Banana Pro SketchDreamer SketchAgent

###### (a) Raster-Based Methods (b) Vector-Based Methods

(c) Ours

- Fig. 2. Challenges in progressive illusion sketching. (a) Raster-based methods (e.g., Nano Banana Pro) rely on destructive editing, modifying the initial structure to fit the final target and thus violating the progressive constraint. (b) Vector-based baselines (e.g., SketchDreamer [Qu et al. 2023] or SketchAgent [Vinker et al. 2025]) employ a greedy strategy, where specific Phase 1 details become semantic noise or clutter in Phase 2. (c) Ours achieves dual-semantic coherency by jointly optimizing for a common structural subspace, ensuring the initial strokes are valid building blocks for both interpretations (e.g., “rabbit” → “elephant”).

Our main contributions are summarized as follows:

- • Task: We introduce Progressive Semantic Illusion, extending visual illusions from the spatial to the temporal dimension. This task requires a single vector sketch to reveal distinct semantic interpretations through progressive stroke accumulation.
- • Method: We formalize this as constrained optimization over shared Bézier parameters, enabling joint discovery of a Common Structural Subspace. A novel Overlay Loss enforces spatial complementarity between prefix and delta strokes, preventing crowding and ensuring integration over occlusion. We further introduce a VLM-based filtering and ranking pipeline for candidate selection.
- • Evaluation & Scalability: Experiments and user studies show our method significantly outperforms baselines in recognizability and illusion strength. Our framework generalizes to 𝐾-phase illusions (“A” → “B” → “C”) and alternative representations including B-splines, colored strokes, and general vector graphics.

1 Introduction

Visual illusions traditionally exploit spatial ambiguities, requiring viewpoint changes to reveal hidden meanings [Geng et al. 2024b]. We introduce a new dimension: time. We propose Progressive Semantic Illusions, where the drawing process itself drives semantic transformation. Sparse line drawings are uniquely suited: their incompleteness invites Gestalt closure [Wagemans et al. 2012], letting the visual system re-interpret existing strokes as new ones arrive. As shown in Fig. 1, our method generates an initial sketch (e.g., “a pig”) that additional strokes re-contextualize into a distinct concept (e.g., “an angel”), achieving perceptual shift through sequential accumulation rather than spatial manipulation.

2 Related Work

Generative Vector Graphic Synthesis. Early sketch synthesis relied on category-specific corpora [Eitz et al. 2012; Jongejan et al. 2016; Sangkloy et al. 2016] with a fixed vocabulary. CLIP [Radford et al. 2021] lifted this constraint. CLIPDraw [Frans et al. 2022] and CLIPasso [Vinker et al. 2022] optimize Bézier curves [Bézier 1968; Casteljau 1959] through a differentiable rasterizer [Li et al. 2020] against CLIP similarity. This approach later extended to scenes [Vinker et al.

Sketch generation has evolved from category-specific RNNs [Ha and Eck 2017] to open-vocabulary models leveraging CLIP [Radford et al. 2021] and diffusion priors [Rombach et al. 2022]. Methods like CLIPasso [Vinker et al. 2022] and VectorFusion [Jain et al. 2023] utilize differentiable rasterization for high-fidelity sketching, while sequential approaches like SketchAgent [Vinker et al. 2025] and SketchDreamer [Qu et al. 2023] mimic step-by-step human drawing. Regarding illusions, Visual Anagrams [Geng et al. 2024b] and ShadowDraw [Luo et al. 2025] explore multi-view effects via diffusion. However, these prior works focus on static pixel representations or spatial rearrangements, leaving the challenge of temporal semantic transformation in vector graphics unexplored.

- 2023], though global image-text alignment lacks dense structural supervision. Score Distillation Sampling (SDS) [Poole et al. 2022] replaces this with per-pixel diffusion gradients. VectorFusion [Jain

- et al. 2023] first ported SDS to SVGs. DiffSketcher [Xing et al. 2023] initializes strokes from cross-attention. SVGDreamer [Xing et al.

2024] decomposes prompts into semantic components, and SketchDreamer [Qu et al. 2023] adds interactive prompting. The objective itself has been refined by ProlificDreamer’s variational particles [Wang et al. 2023b], LucidDreamer’s interval matching [Liang

- et al. 2024], and SDI’s reparametrized DDIM [Lukoianov et al. 2024]. Others bypass optimization entirely. SwiftSketch [Arar et al. 2025] predicts strokes feed-forward. DeepSVG [Carlier et al. 2020] learns a hierarchical SVG latent. IconShop [Wu et al. 2023b] and StarVector [Rodriguez et al. 2025] autoregressively decode SVG tokens. A parallellinereplacesdiscretecontrol points with implicit fields [Reddy

Generatingprogressiveillusionspresents a unique “Dual-Constraint”:

early strokes must depict object “A” while simultaneously functioning as the structural foundation for object “B” (Fig. 2). Existing methods fail to address this additive nature. Raster-based models (e.g., Nano Banana Pro) rely on destructive editing, overwriting initial pixels and violating the progressive constraint. Conversely, sequential vector models (e.g., SketchAgent) employ a greedy strategy, optimizing strokes solely for “A”. This renders the fixed prefix as semantic noise when extending to “B”, resulting in clutter. Crucially, these

ℒ

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]|
|---|

|[Figure 27]|
|---|

|𝑠|
|---|
|𝑠|
|⋮|
|𝑠|

𝜖~𝒩 0,𝐼 𝑡~𝒰(1,𝑇)

|[Figure 28]| |
|---|---|
| | |

|[Figure 29]<br><br>[Figure 30]| |
|---|---|
| | |

“a rabbit”

PrefixStrokes𝑆

Differentiable 𝑠 ,…,𝑠 Rasterizer

Frozen Diffusion

|𝑠|
|---|
|𝑠|
|⋮|
|𝑠|

|[Figure 31]| |
|---|---|
| | |

|[Figure 32]<br><br>[Figure 33]| |
|---|---|
| | |

|[Figure 34]|
|---|

“a horse”

DeltaStrokes𝑆

Full Strokes

Differentiable 𝑠 ,…,𝑠 Rasterizer

Frozen Diffusion

𝜖~𝒩 0,𝐼 𝑡~𝒰(1,𝑇)

ℒ

Update Learnable Strokes Parameters

Strokes Initialization

Gradient

SDS Guidance ℒ = ℒ + ℒ

- Fig. 3. Pipeline overview. Our method optimizes a set of learnable stroke parameters, which are divided into prefix strokes 𝑆prefix and delta strokes 𝑆delta. The optimization process involves two parallel branches. In the top branch, only the prefix strokes are rendered by a differentiable rasterizer to create a partial sketch (e.g., a rabbit). This sketch is then guided by a pre-trained, frozen text-to-image diffusion model using a prompt corresponding to the prefix (“a

rabbit”), resulting in the prefix SDS loss LSDSprefix. In the bottom branch, the full set of strokes is rendered to create the complete sketch (e.g., a horse). This is guided by the same diffusion model using a prompt for the full object (“a horse”), resulting in the full SDS loss LSDSfull . The total SDS guidance loss is the sum of these two terms LSDS = LSDSprefix + LSDSfull . Gradients from this total loss are backpropagated to update all learnable stroke parameters.

et al. 2021; Thamizharasan et al. 2024]. NeuralSVG [Polaczek et al. 2025] combines these with SDS. All of these treat drawing as a single static target. Our setting requires the same strokes to satisfy two interpretations at different completion stages, a constraint no static-target formulation addresses.

Hybrid images [Oliva et al. 2006] interleave frequency bands that change perception with viewing distance. Wire art [Hsiao et al. 2018] renders different 2D projections from different angles. Anamorphic sculptures [Pratt et al. 2023; Wu et al. 2022] or warped canvases [Chang et al. 2025; Debnath et al. 2025] reveal hidden images only under specific viewpoints. Diffusion priors extend this paradigm. Visual Anagrams [Geng et al. 2024b] averages scores across flips and rotations. Factorized Diffusion [Geng et al. 2024a] splits the score into frequency bands depicting different concepts. PTDiffusion [Gao et al. 2025] transfers spectral phase between prompts. Illusion3D [Feng et al. 2024] enforces 3D-viewpoint consistency. Diffusion Illusions [Burgert et al. 2024] adds fabrication constraints. Images that Sound [Chen et al. 2024] jointly satisfies visual and spectrogram interpretations. AmbiGen [Zhao et al. 2023] produces rotation-readable ambigrams. Every one of these relies on a symmetric spatial transform, like flip, rotate, reproject, that swaps one complete image for another. Ours is asymmetric and additive: the prefix is a strict geometric subset of the final drawing, not a transformed counterpart. Our dual-branch SDS and Overlay Loss target precisely this temporal constraint.

Sequential Sketch Generation. Sequential methods model drawing as a temporal process. DRAW [Gregor et al. 2015] introduced iterative glimpses, and SketchRNN [Ha and Eck 2017] adapted this to stroke sequences with an LSTM, later replaced by Transformers for longer-range modeling [Ribeiro et al. 2020]. For stroke geometry, BézierSketch [Das et al. 2020] autoregressively emits parametric curves rather than polylines. DoodleFormer [Bhunia et al.

- 2022] decouples coarse layout from fine detail. SketchKnitter [Wang et al. 2023a] replaces autoregressive decoding with parallel diffusion. Complementary work tackles partial-input completion [Liu et al. 2019; Su et al. 2020] and stroke-level hierarchical editing [Zang et al. 2025]. More recently, SketchAgent [Vinker et al. 2025] drives stroke generation through LLM dialogue. While Chat2SVG [Wu et al. 2025] and LLM4SVG [Xing et al. 2025] prompt LLMs to emit SVG code directly. Despite their temporal formulation, all commit each stroke greedily to a single target. Once emitted, a stroke is frozen. As our ablation shows (Fig. 10), this traps the prefix in a local minimum incompatible with a second concept. Our joint optimization instead lets prefix strokes shift under dual semantic pressure.

3 Method

Progressive illusions require prefix strokes to depict an initial object while forming the structural basis for a final one. We propose a joint optimization framework via multi-branch Score Distillation Sampling to discover a common structural subspace valid for both interpretations. Prefix strokes receive simultaneous gradients to satisfy dual roles, while an overlay loss enforces spatial separation, ensuring structural integration rather than occlusion.

Sketch Perception and Visual Illusions. Line drawings are cognitively robust. Gestalt closure and figure-ground segregation [Wagemans et al. 2012] let viewers complete fragmentary contours [Biederman 1987; Kanizsa et al. 1979]. Sparse sketches reliably trigger recognition in both humans [Cavanagh 2005; Eitz et al. 2012; Fan et al. 2023] and networks [Yu et al. 2017]. This makes them a natural substrate for progressive illusions. Computational illusions have almost exclusively exploited spatial manipulation. Shadow art [Mitra and Pauly 2009] casts different silhouettes from distinct lighting.

3.1 Progressive Semantic illusion in Vector Form

We partition a set of learnable Bézier strokes 𝑆 into disjoint subsets: prefix 𝑆prefix = {𝑠1, . . .,𝑠𝑘} and delta 𝑆delta = {𝑠𝑘+1, . . .,𝑠𝑁 }. The

progressive illusion requires 𝑆prefix to depict the initial concept 𝑝1, while the full sketch 𝑆full = 𝑆 depicts the target 𝑝2, achieved by delta strokes recontextualizing the prefix. We optimize stroke parameters 𝜃 such that the rasterized outputs R(𝑆prefix;𝜃) and R(𝑆full;𝜃) align with 𝑝1 and 𝑝2, respectively. The core challenge lies in discovering configurations where prefix strokes meaningfully serve both semantic interpretations.

|(a) 𝑆<br><br>[Figure 35]|(b) 𝑆<br><br>[Figure 36]<br><br>Redundant Strokes<br><br>[Figure 37]<br><br>[Figure 38]|[Figure 39]<br><br>(c) Hard Intersec on<br><br>[Figure 40]<br><br>[Figure 41]|
|---|---|---|
|[Figure 42]<br><br>(d) Blurred 𝑆|[Figure 43]<br><br>(e) Blurred 𝑆|[Figure 44]<br><br>(f) Penalty Map: (d) ⨀ (e)|

- 3.2 Joint Optimization Pipeline

We employ a dual-branch strategy to simultaneously refine both stroke subsets (Fig. 3). Unlike sequential methods, our pipeline coordinates semantic objectives via parallel guidance on shared learnable parameters 𝜃. We initialize 𝑁 strokes near the canvas center, partitioning them into 𝑆prefix (first 𝑘) and 𝑆delta (remaining). At each iteration, the prefix branch renders 𝐼prefix = R(𝑆prefix;𝜃). We apply the gradient of the Score Distillation Sampling (SDS) loss conditioned on 𝑝1:

∇𝜃 LSDSprefix = 𝑤(𝑡) 𝜖𝜙 (𝑧𝑡,𝑡,𝑝1) − 𝜖

𝜕𝑧𝑡 𝜕𝜃

, (1)

where 𝑧𝑡 is the noised latent, 𝜖𝜙 the noise predictor, and 𝑤(𝑡) a weighting function.

Simultaneously, the full branch renders 𝐼full = R(𝑆full;𝜃) conditioned on 𝑝2, yielding ∇𝜃LSDSfull . We combine these gradients as

∇𝜃LSDS = ∇𝜃LSDSprefix + ∇𝜃LSDSfull . (2) This ensures prefix strokes receive simultaneous gradients from both targets, satisfying dual roles, while delta strokes optimize to complement them. To prevent delta strokes from merely occluding the prefix, which is a common issue with pure semantic guidance, we introduce an overlay loss that penalizes spatial overlap to enforce structural integration.

- 3.3 Overlay Loss for Spatial Coordination

Semantic guidance alone fails to prevent spatial redundancy, often causing delta strokes to clutter prefix strokes (Fig. 4(b)). We introduce an overlay loss to enforce spatial complementarity. We render stroke subsets separately and apply Gaussian blur 𝐺𝜎 to create soft spatial buffers (𝐼˜prefix,𝐼˜delta), as shown in Fig. 4(d,e). We then compute the normalized overlap:

Loverlay =

2⟨𝐼˜prefix,𝐼˜delta⟩ ∥𝐼˜prefix∥1 + ∥𝐼˜delta∥1

, (3)

where ⟨·, ·⟩ denotes the inner product over pixel space.

This constraint promotes structural integration and smoother semantic transitions, ensuring prefix strokes serve as essential components rather than being obscured. The final objective is:

L = LSDS + 𝜆overlayLoverlay, (4)

where 𝜆overlay weights the penalty. Gradients are backpropagated via differentiable rasterization.

- 3.4 Filtering and Ranking

- Fig. 4. Motivation and formulation of the overlay loss. (Top) Moti-

vation: (a) The prefix sketch 𝑆prefix (blue, e.g., “chicken”) is the structural foundation. Without spatial constraints, delta strokes (orange) cause redundant occlusions (arrows in (b)); the hard intersection map (c) highlights severely crowded regions. (Bottom) Formulation: The soft overlay loss (f) is the normalized inner product of Gaussian-blurred maps of 𝑆prefix (d) and 𝑆delta (e). Blurring creates a spatial buffer beyond stroke boundaries, enforcing minimum separation and structural complementarity.

|[Figure 45]<br><br>Sketch 𝑆|
|---|

[Figure 46]

🤖

Question: How clearly does the sketch depict a rabbit?

|[Figure 47]<br><br>First Sketch 𝑆|
|---|

|[Figure 48]<br><br>Second Sketch 𝑆|
|---|

Question:

- (1) How clearly does the first sketch depict a horse?
- (2) Is the first sketch clearly more recognizable than the second sketch alone?

Answer: Clear rabbit (0.9/1.0)

Answer: Clear horse, ﬁrst sketch more

🤖 recognizablethanthesecond(0.75/1.0)

[Figure 49]

(a) Phase 1 Score (b) Phase 2 Score

- Fig. 5. VLM-based evaluation and ranking pipeline. We employ GPT4o to assess the quality of illusion sketches. (a) For Phase 1, the model evaluates the recognizability of the prefix sketch (𝑆prefix). (b) For Phase 2, the model evaluates the full sketch (𝑆full) while simultaneously comparing it against the delta strokes (𝑆delta). This comparison ensures that the prefix strokes provide essential structural scaffolding for the second concept, rather than being merely overwritten. High scores are awarded only when 𝑆full is significantly more recognizable than 𝑆delta alone.

VLM-based Quality Assessment. We employ GPT-4o to evaluate four dimensions (Fig. 5). Phase recognizability and Single-object integrity ensure semantic accuracy and coherence. Illusion quality validates the prefix’s structural contribution by confirming 𝑆full is significantly more recognizable than 𝑆delta alone. Sketch quality penalizes visual clutter. Each phase receives individual scores across these dimensions, and candidates failing minimum thresholds are filtered.

To ensure quality, our systematic pipeline selects the best candidates using VLM assessment and quantitative metrics.

overlaying; vector baselines natively support stroke addition. CLIPasso and ControlSketch require image input, so we supply SDXLgenerated [Podell et al. 2023] references from the same prompts, with matched stroke count and segments. SketchDreamer uses its default 5-segment cubic Bézier curves, as our 1-segment setting triggers reinitialization that degrades quality. (2) Ours-to-illusion: We supply our optimized prefix sketches to test whether baselines can complete the transformation given an ideal structural foundation.

ℒ

|𝑠|
|---|
|𝑠|
|⋮|
|𝑠|

𝜖~𝒩 0,𝐼 𝑡~𝒰(1,𝑇)

𝑝 =“apple”

|[Figure 50]|
|---|

Rasterizer

|[Figure 51]<br><br>[Figure 52]| |
|---|---|
| | |

|[Figure 53]<br><br>𝐼 : | |
|---|---|
| | |

StrokeSubset𝑆

𝑆 :  = 𝑆

Frozen Diffusion

[Figure 54]

𝑠 ,…,𝑠

|𝑠|
|---|
|𝑠|
|⋮|
|𝑠|

ℒ

StrokeSubset𝑆

𝜖~𝒩 0,𝐼 𝑡~𝒰(1,𝑇)

𝑝 =“sheep”

|[Figure 55]|
|---|

StrokesInitialization

Rasterizer

|[Figure 56]<br><br>[Figure 57]| |
|---|---|
| | |

|[Figure 58]<br><br>𝐼 : | |
|---|---|
| | |

[Figure 59]

𝑆 :  = 𝑆 ∪ 𝑆

Frozen Diffusion

𝑠 ,…,𝑠 , 𝑠 ,…,𝑠

[Figure 60]

…

…

…

ℒ

|⋮|
|---|
|⋮|
|⋮|
|𝑠|

𝜖~𝒩 0,𝐼 𝑡~𝒰(1,𝑇)

𝑝 =“Einstein”

|[Figure 61]|
|---|

StrokeSubset𝑆

Rasterizer

|[Figure 62]<br><br>[Figure 63]| |
|---|---|
| | |

|𝐼𝐼 :  : <br><br>[Figure 64]| |
|---|---|
| | |

Data. Our evaluation dataset comprises 64 common objects span-

Full Strokes

Frozen Diffusion

𝑠 ,…,𝑠

ning diverse categories. We randomly sample pairs to form (𝑝1,𝑝2) combinations, run multiple optimization iterations per pair, then apply filtering and ranking to select top-k results for evaluation.

Gradient

SDS Guidance ℒ = ∑ ℒ

- Fig. 6. Multi-phase pipeline. We scale to 𝐾 phases (e.g., Apple→Sheep→Einstein) using cumulative stroke subsets (𝑆1, . . .,𝑆𝐾). Parallel branches optimize each cumulative sketch 𝐼1:𝑖 against prompt 𝑝𝑖. Joint optimization ensures early strokes receive gradients from all

Implementation Details. We implement our framework using Stable Diffusion v1.5 for Score Distillation Sampling guidance on an NVIDIA RTX 4090 GPU. We optimize stroke parameters 𝜃 for 2,000 iterations using Adam optimizer with guidance scale 100 and overlay loss weight 𝜆overlay = 0.1. Generation requires approximately 13 minutes for two-phase and 15 minutes for three-phase illusions.

subsequent losses ( LSDS𝑖 ), creating a structure primed for the entire evolutionary sequence.

Metrics. For quantitative evaluation, we employ both standard and specialized metrics to assess illusion quality. We use CLIP score computed as the minimum across all phases to measure semantic alignment. Beyond standard metrics, we define two illusion-specific measures. Structural concealment evaluates whether prefix strokes contribute substantively to the full sketch rather than being occluded by delta strokes. For any metric 𝑀 ∈ {CLIP, ImageReward, HPS} [Hessel et al. 2022; Wu et al. 2023a; Xu et al. 2023], we compute: 𝐶struct𝑀 = 𝑀full − 𝑀delta. Higher scores indicate prefix strokes retain significant structural roles. Semantic concealment measures whether non-current phase semantics are effectively hidden. Following [Geng et al. 2024b], we compute:

Ranking Strategies. GPT-based ranking (Fig. 18) favors semantic accuracy: RGPT = ScorePhase 1 · ScorePhase 2. Metric-based ranking (Fig. 17) emphasizes perceptual contrast [Luo et al. 2025] by penalizing independent delta stroke quality:

𝑆CLIP = (CLIPp1 · CLIPp2)/CLIP2delta, (5) 𝑆IR = Φ(IRp1)2 + Φ(IRp2)2 − Φ(IRdelta)2, (6)

𝑆HPS = HPS2p1 + HPS2p2 − HPS2delta, (7) where Φ(·) is the standard Gaussian CDF. The final score R = 𝑆CLIP · 𝑆IR · 𝑆HPS ensures the prefix provides substantial structural contribution.

𝐶semantic = tr(softmax(𝑆/𝜏)), (9)

- 3.5 Extension to Multi-Phase Illusions

where𝑆 is the CLIP image-text similarity matrix and𝜏 is temperature. Higher scores indicate clear phase-specific semantics.

Our framework naturally scales to 𝐾-phase illusions (𝐴1, . . .,𝐴𝐾) by partitioning strokes into disjoint subsets 𝑆1, . . .,𝑆𝐾. Each cumulative prefix 𝑆1:𝑖 = 𝑖𝑗=1 𝑆𝑗 renders concept 𝐴𝑖. We employ parallel branches (Fig. 6) to jointly optimize all parameters, rendering 𝐼1:𝑖 conditioned on prompt𝑝𝑖. This ensures early strokes (e.g.,𝑆1) receive gradients from all subsequent branches, coordinating cumulative interpretations. We extend the overlay loss to penalize overlap between 𝑆1:𝑖 and the next subset 𝑆𝑖+1:

We further conduct two user studies with 143 participants for additional quantitative validation. The first compares our top-1 result against baselines across five prompt pairs. The second assesses our ranking pipeline by asking participants to select satisfactory results from our top-4 outputs across four prompt pairs, evaluating both technical performance and practical user satisfaction.

∑︁𝐾

𝐾∑︁−1

4.2 Results and Analysis

𝜆𝑖overlayLoverlay𝑖 . (8)

LSDS𝑖 +

L =

As shown in Tab. 1(a,c), our method substantially outperforms baselines in CLIP and concealment scores, achieving 100% coverage versus Nano Banana Pro’s 34.9%. Among image-based baselines, CLIPasso attains the highest Phase-1 CLIP score via direct image conditioning, yet concealment remains low; ControlSketch shows negative concealment scores. Both optimize strokes to reproduce the input image, so prefix strokes lack flexibility for a second semantic interpretation. Fig. 7 highlights characteristic failures: clutter (SketchDreamer), oversimplification (SketchAgent), destructive editing (Nano Banana Pro), and prefix strokes too tightly bound to references for recontextualization (CLIPasso, ControlSketch). Tab. 1(b,c)

𝑖=1

𝑖=1

- 4 Experiments 4.1 Experimental Setup

Baseline. We adapt state-of-the-art methods to the progressive illusion task: Nano Banana Pro (raster), SketchAgent [Vinker et al. 2025], SketchDreamer [Qu et al. 2023] (vector), and two image-based methods, CLIPasso [Vinker et al. 2022] and ControlSketch [Arar et al. 2025]. We design two protocols: (1) Text-to-illusion: Baselines generate sketches sequentially—prefix from𝑝1, full sketch from 𝑝2. Nano Banana Pro enforces the progressive constraint via prefix

###### Image-based

Text-based

|[Figure 65]|[Figure 66]|
|---|---|

|[Figure 67]|[Figure 68]|
|---|---|

|[Figure 69]|[Figure 70]|
|---|---|

|[Figure 71]|[Figure 72]|
|---|---|

|[Figure 73]|[Figure 74]|
|---|---|

|[Figure 75]|[Figure 76]|
|---|---|

→→→chickenmonkeyowlangelrabbitelephant

|[Figure 77]|[Figure 78]|
|---|---|

|[Figure 79]|[Figure 80]|
|---|---|

|[Figure 81]|[Figure 82]|
|---|---|

|[Figure 83]|[Figure 84]|
|---|---|

|[Figure 85]|[Figure 86]|
|---|---|

|[Figure 87]|[Figure 88]|
|---|---|

|[Figure 89]|[Figure 90]|
|---|---|

|[Figure 91]|[Figure 92]|
|---|---|

|[Figure 93]|[Figure 94]|
|---|---|

|[Figure 95]|[Figure 96]|
|---|---|

|[Figure 97]|[Figure 98]|
|---|---|

|[Figure 99]|[Figure 100]|
|---|---|

(c) SketchDreamer

(d) SketchAgent

(e) Nano Banana Pro

(f) Ours

(a) CLIPasso*

(b) ControlSketch*

- Fig. 7. Qualitative comparisons. We compare against CLIPasso [Vinker et al. 2022] and ControlSketch [Arar et al. 2025] (image-based, denoted ∗), SketchDreamer [Qu et al. 2023], SketchAgent [Vinker et al. 2025], and Nano Banana Pro (text-based). Image-based methods use SDXL-generated [Podell et al.

2023] references from the same prompts. (a, b) CLIPasso and ControlSketch follow reference contours too closely, failing to integrate prefix with delta strokes. (c) SketchDreamer produces noisy strokes with severe clutter. (d) SketchAgent yields overly abstract results with low recognizability. (e) Nano Banana Pro relies on destructive editing, violating the progressive constraint despite high image quality. (f) Ours generates clean, structurally consistent sketches where prefix strokes are creatively repurposed (e.g., rabbit ears becoming elephant ears). Additional results and optimization visualizations are in the supplement.

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

[Figure 115]

[Figure 116]

(a) Ours → SketchDreamer (b) Ours → SketchAgent (c) Ours → Nano Banana Pro (d) Ours → Ours

(Ours)owl

→angel (Ours)chicken

→monkey

- Fig. 8. Phase 2 extension with fixed prefix (ours). We evaluate how methods extend a fixed Phase 1 sketch generated by our method. Interestingly, baselines produce better Phase 2 results here than in Fig. 7 (where they generate Phase 1 themselves). This indicates that our Phase 1 strokes inherently embed structural cues for the second concept, validating that our joint optimization successfully finds a versatile common subspace. However, comparing (a-c)

with (d), our method still achieves the highest success rate and structural consistency, as 𝑆delta is jointly optimized with the prefix rather than sequentially appended.

and Fig. 8 show that with fixed prefixes, baselines improve, suggesting our prefixes embed implicit structural cues (“common subspace”), yet remain substantially inferior to ours, confirming that joint optimization is essential for seamless integration.

SketchDreamer SketchAgent Nano Banana Pro Ours No Preference

GPT-ranking Metric-ranking Overall

100.0%

100.0%

75.0%

75.0%

50.0%

50.0%

25.0%

25.0%

0.0%

0.0%

GPT-ranking Metric-ranking

Success Rate

Fig. 9. User study. (Left) Preference: Participants overwhelmingly favor our method (green) over baselines across both ranking strategies. (Right) Reliability: A high success rate (>97%) confirms that our pipeline consistently yields valid illusions, ensuring robustness against the inherent stochasticity of the generation process.

User Studies. Our user studies strongly reinforce these findings. In comparisons against baselines, participants selected our method in 67.7% of GPT-ranking and 87.1% of Metric-ranking cases (Fig. 22(a)). Our ranking pipeline demonstrates strong reliability with over 98% overall satisfaction rates (Fig. 22(b)), thoroughly validating our framework’s effectiveness.

Table 1. Quantitative comparison. (a) Vector baselines lack quality; Nano Banana fails coverage (∼35%). (b) Extending our Phase 1 helps, but still lags behind (c), validating joint optimization. (c) Ours achieves top metrics with 100% coverage. Best in bold, second underlined.

|[Figure 117]<br><br>[Figure 118]|
|---|

|[Figure 119]<br><br>[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

- (d)

Scattered

|[Figure 123]|[Figure 124]|
|---|---|

[Figure 125]

[Figure 126]

|[Figure 127]|[Figure 128]|
|---|---|

- (e)Gathered

(Ours)

- (f)Gathered

- (a)

Scattered

- (b)

Gathered

|[Figure 129]|
|---|

- (c)Gathered

|[Figure 130]<br><br>|[Figure 131]|
|---|
|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]<br><br>|[Figure 135]|
|---|
|
|---|

|[Figure 136]|[Figure 137]|
|---|---|

Phase 1 CLIP ↑ Concealment (structural) 𝐶semantic Coverage Method Source Avg min CLIP ↑ IR ↑ HPS ↑ CLIP ↑ (%) ↑

CLIPasso - 32.213 1.690 0.090 0.004 1.000 100.0% ControlSketch - 27.524 -2.378 -0.789 -0.018 0.875 100.0%

|[Figure 138]|
|---|

[Figure 139]

+Shifted

+Shifted

- (a) SketchDreamer - 24.803 -0.393 0.338 0.011 0.887 100.0% SketchAgent - 24.393 -2.544 0.095 0.000 0.752 100.0% Nano Banana Pro - 26.821 -2.774 -0.663 -0.019 0.875 34.9% SketchDreamer Ours 28.148 0.060 0.302 0.011 0.961 100.0%

- (b) SketchAgent Ours 24.019 -2.778 0.080 0.003 0.762 100.0% Nano Banana Pro Ours 28.903 -1.065 -0.426 -0.014 0.958 35.2%

- (c)

[Figure 140]

Initial Strokes duck sheep Individual Generation Progressive Illusion

Initial Strokes duck → sheep

Ours (GPT-ranking) – 29.873 1.668 0.839 0.023 0.983 100.0% Ours (Metric-ranking) – 30.044 3.282 1.237 0.029 0.980 100.0%

- Fig. 11. Ablation on stroke initialization. (a, d) Scattered fails to aggregate strokes, resulting in disconnected artifacts. (c, f) Shifted yields valid sketches, proving that spatial concentration is critical for convergence, though it risks boundary cropping. (b, e) Centered (Ours) offers the optimal balance, ensuring structural integrity without clipping.

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

(b)w/overlayloss(a)w/ooverlayloss

𝑆 (chicken) 𝑆 (monkey) 𝑆 & 𝑆 Intersection

539 px

174 px

[Figure 149]

Redundant Strokes

- Fig. 12. Ablation of overlay loss (Loverlay). (a) Without Loverlay, the model generates redundant strokes atop existing ones to satisfy the semantic target, resulting in visual clutter (red circle) and high intersection artifacts. (b) With Loverlay, the generated strokes (𝑆delta) become spatially complementary to the prefix (𝑆prefix), avoiding collisions to produce a clean, coherent line drawing.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

→duckcow

(a) Sequential Generation (b) Joint Optimization (Ours)

Fig. 10. Ablation on optimization strategy. (a) Sequential generation yields a rigid Phase 1, creating structural conflicts (e.g., the duck’s beak) that fail Phase 2 repurposing. (b) Joint optimization (Ours) identifies a common structural subspace, yielding a versatile Phase 1 where features serve both interpretations (e.g., the beak doubles as the cow’s ear).

- 4.3 Ablation Studies

Optimization Strategy. We evaluate our joint optimization approach against a sequential alternative that first optimizes prefix strokes independently for the initial concept, then fixes these parameters and optimizes only delta strokes. As shown in Fig. 10(a), this sequential approach produces rigid prefix structures where specific features conflict with the final object, resulting in failed illusion transitions. The prefix optimization focuses solely on the initial concept without considering final target requirements. In contrast, our joint optimization (Fig. 10(b)) updates both stroke sets simultaneously, enabling continuous coordination. This allows the framework to discover a common structural subspace where prefix strokes both represent the initial concept and integrate naturally into the final representation. The results demonstrate improved visual consistency and smooth transitions, confirming that joint optimization is essential for high-quality progressive illusion sketches.

We adopt centered gathered initialization to balance density with coverage and avoid boundary clipping.

Overlay Loss. We validate the necessity of Loverlay. As shown in Fig. 12(a), without it, semantic guidance alone fails to prevent spatial redundancy, producing delta strokes that clutter the prefix; Loverlay penalizes overlap, enforces spatial complementarity, and substantially reduces intersection artifacts (Fig. 12(b)). Crucially, it promotes structural coherence: prefix strokes integrate naturally into the subsequent concept rather than being obscured, confirming that geometric constraints are essential for clean progressive illusions.

Stroke Initialization. Since our objective is highly non-convex, initialization is critical for convergence. Centered initialization is standard in SDS-based sketch methods [Jain et al. 2023; Vinker et al. 2022; Xing et al. 2023]; in our dual-constraint setting it is especially necessary, as prefix strokes must form a spatially coherent structure valid for both semantic interpretations. Fig. 11 shows spatial concentration is paramount: scattered initialization fails to capture essential features, while both centered and shifted gathered configurations succeed, indicating local stroke density outweighs absolute position.

Stroke Count. Optimal stroke budget depends on concept complexity (Fig. 13). Simple transformations (e.g., rabbit-to-horse) succeed with minimal strokes (8–16), whereas complex subjects like Einstein require 32–64 strokes to capture essential details; insufficient budgets compromise recognizability. We therefore adopt a default of

|[Figure 154]|[Figure 155]|
|---|---|

|[Figure 156]|[Figure 157]|
|---|---|

|[Figure 158]|[Figure 159]|
|---|---|

Acknowledgments

rabbitrabbit→→horseEinstein

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-E-A49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

|[Figure 160]|[Figure 161]|
|---|---|

|[Figure 162]|[Figure 163]|
|---|---|

|[Figure 164]|[Figure 165]|
|---|---|

8 strokes → 16 strokes 16 strokes → 32 strokes 32 strokes → 64 strokes

Less strokes More strokes

References

Ellie Arar, Yarden Frenkel, Daniel Cohen-Or, Ariel Shamir, and Yael Vinker. 2025. Swiftsketch: A diffusion model for image-to-vector sketch generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–12.

Fig. 13. Analysis of stroke count. (Top) Simple concepts (horse) form recognizable silhouettes with minimal strokes (8→16). (Bottom) While complex concepts (Einstein) require a larger budget (32→64) to capture essential details. Fewer strokes result in abstraction. Our default (16→32) balances structural simplicity and semantic fidelity.

Pierre E. Bézier. 1968. How Renault Uses Numerical Control for Car Body Design and Tooling. Technical Report. SAE Technical Paper. https://www.sae.org/publications/ technical-papers/content/680010/

Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Fahad Shahbaz Khan, Jorma Laaksonen, and Michael Felsberg. 2022. DoodleFormer: Creative Sketch Drawing with Transformers. In European Conference on Computer Vision. 338–355. https://arxiv.org/abs/2112.03258

16 prefix strokes and 32 total strokes, robustly balancing structural simplicity with semantic fidelity.

Irving Biederman. 1987. Recognition-by-Components: A Theory of Human Image Understanding. Psychological Review 94, 2 (1987), 115. https://doi.org/10.1037/0033295X.94.2.115

- 4.4 Applications

Technical Versatility. We demonstrate versatility beyond standard two-phase scenarios. Fig. 14 confirms robustness across diverse concept pairs, ranging from structurally similar to semantically distant. Fig. 15 extends this to three-phase illusions (e.g., apple-to-rabbit-topig), showcasing effective multi-target coordination. Furthermore, our framework generalizes to alternative representations, including B-spline curves (Fig. 19), vector graphics (Fig. 20), and colored sketches (Fig. 21), validating the broad applicability of our joint optimization principle.

Practical Applications. Creative education: Progressive illusions serve as spatial reasoning exercises, fostering Gestalt perception. Brand and logo design: Animated illusions bridge two brand identities in a single vector asset for mergers or motion graphics. Physical media steganography: Outputs support thermochromic printing—a mug shows one concept at rest; heat reveals the transformation. Dynamic visual storytelling: Native vector output enables arbitrary-resolution rendering and frame-by-frame animation for interactive media. Cognitive science: The framework generates controlled stimuli for studying temporal semantic perception with calibrated structural overlap.

- 5 Conclusion

Ryan Burgert, Xiang Li, Abe Leite, Kanchana Ranasinghe, and Michael Ryoo. 2024. Diffusion Illusions: Hiding Images in Plain Sight. In ACM SIGGRAPH 2024 Conference Papers. 1–11. https://arxiv.org/abs/2312.03817

Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. 2020. DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation. Advances in Neural Information Processing Systems 33 (2020), 16351–16361.

Paul De Casteljau. 1959. Outillages méthodes calcul. Technical Report. André Citroën Automobiles SA. Patrick Cavanagh. 2005. The Artist as Neuroscientist. Nature 434, 7031 (2005), 301–307. https://doi.org/10.1038/434301a Pascal Chang, Sergio Sancho, Jingwei Tang, Markus Gross, and Vinicius Azevedo.

2025. LookingGlass: Generative Anamorphoses via Laplacian Pyramid Warping. In Proceedings of the Computer Vision and Pattern Recognition Conference. 24–33.

Ziyang Chen, Daniel Geng, and Andrew Owens. 2024. Images That Sound: Composing Images and Sounds on a Single Canvas. In Advances in Neural Information Processing Systems, Vol. 37. 85045–85073. https://arxiv.org/abs/2405.12221

Ayan Das, Yongxin Yang, Timothy Hospedales, Tao Xiang, and Yi-Zhe Song. 2020. Béziersketch: A generative model for scalable vector sketches. In European conference on computer vision. Springer, 632–647.

Soumyaratna Debnath, Ashish Tiwari, Kaustubh Sadekar, and Shanmuganathan Raman. 2025. RASP: Revisiting 3D Anamorphic Art for Shadow-Guided Packing of Irregular Objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5849–5858. https://arxiv.org/abs/2504.02289

Mathias Eitz, James Hays, and Marc Alexa. 2012. How Do Humans Sketch Objects?. In ACM Transactions on Graphics, Vol. 31. 1–10. https://dl.acm.org/doi/10.1145/ 2185520.2185540

Judith E Fan, Wilma A Bainbridge, Rebecca Chamberlain, and Jeffrey D Wammes.

2023. Drawing as a versatile cognitive tool. Nature Reviews Psychology 2, 9 (2023), 556–568.

Yue Feng, Vaibhav Sanjay, Spencer Lutz, Badour AlBahar, Songwei Ge, and Jia-Bin Huang. 2024. Illusion3D: 3D Multiview Illusion with 2D Diffusion Priors. arXiv preprint arXiv:2412.09625 (2024). https://arxiv.org/abs/2412.09625

We present Stroke of Surprise, the first framework for progressive semantic illusions in vector sketching. By shifting from spatial to temporal dimensions, we enable real-time semantic re-contextualization. Our joint optimization strategy demonstrates that prefix strokes must be "primed" for future semantics. Greedy baselines do not have this ability. Meanwhile, the Overlay Loss ensures structural integration without obfuscation. Evaluations confirm our results are both semantically accurate and perceptually surprising.

Kevin Frans, Lisa Soros, and Olaf Witkowski. 2022. Clipdraw: Exploring text-to-drawing synthesis through language-image encoders. Advances in Neural Information Processing Systems 35 (2022), 5207–5218. https://arxiv.org/abs/2106.14843

Xiang Gao, Shuai Yang, and Jiaying Liu. 2025. PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18240–18249.

- Daniel Geng, Inbum Park, and Andrew Owens. 2024a. Factorized Diffusion: Perceptual Illusions by Noise Decomposition. In European Conference on Computer Vision. 366–384. https://arxiv.org/abs/2404.11615
- Daniel Geng, Inbum Park, and Andrew Owens. 2024b. Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 24154–24163. https://arxiv. org/abs/2311.17919

Limitations. Our method inherits limitations from pre-trained diffusion priors; weak SDS guidance for complex structures (e.g., “scissors”) causes optimization failure. We provide visual examples in the supplementary material.

Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Rezende, and Daan Wierstra. 2015. Draw: A recurrent neural network for image generation. In International conference on machine learning. PMLR, 1462–1471.

David Ha and Douglas Eck. 2017. A Neural Representation of Sketch Drawings. arXiv preprint arXiv:1704.03477 (2017). https://arxiv.org/abs/1704.03477 Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi.

2022. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. arXiv:2104.08718 [cs.CV] https://arxiv.org/abs/2104.08718

Kai-Wen Hsiao, Jia-Bin Huang, and Hung-Kuo Chu. 2018. Multi-View Wire Art. In ACM Transactions on Graphics, Vol. 37. 242. https://dl.acm.org/doi/10.1145/3272127. 3275087

Ajay Jain, Amber Xie, and Pieter Abbeel. 2023. VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1911–1920.

Jonas Jongejan, Henry Rowley, Takashi Kawashima, Jongmin Kim, and Nick Fox-Gieg.

2016. Quick, Draw! The Data. https://quickdraw.withgoogle.com/data Gaetano Kanizsa, Paolo Legrenzi, and Paolo Bozzi. 1979. Organization in Vision: Essays on Gestalt Perception. Praeger.

Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, and Jonathan Ragan-Kelley. 2020. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG) 39, 6 (2020), 1–15. https://people.csail.mit.edu/tzumao/diffvg/

Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. 2024. LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6517–6526. https://arxiv.org/abs/2311.11284

Fang Liu, Xiaoming Deng, Yu-Kun Lai, Yong-Jin Liu, Cuixia Ma, and Hongan Wang. 2019. SketchGAN: Joint Sketch Completion and Recognition with Generative Adversarial Network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5830–5839.

Artem Lukoianov, Haitz Sáez de Ocáriz Borde, Kristjan Greenewald, Vitor Guizilini, Timur Bagautdinov, Vincent Sitzmann, and Justin M. Solomon. 2024. Score Distillation via Reparametrized DDIM. In Advances in Neural Information Processing Systems, Vol. 37. 26011–26044. https://arxiv.org/abs/2405.15891

Rundong Luo, Noah Snavely, and Wei-Chiu Ma. 2025. ShadowDraw: From Any Object to Shadow-Drawing Compositional Art. arXiv preprint arXiv:2512.05110 (2025). https://arxiv.org/abs/2512.05110

Niloy J. Mitra and Mark Pauly. 2009. Shadow Art. In ACM Transactions on Graphics, Vol. 28. 156. https://dl.acm.org/doi/10.1145/1618452.1618502

Aude Oliva, Antonio Torralba, and Philippe G. Schyns. 2006. Hybrid Images. In ACM Transactions on Graphics, Vol. 25. 527–532. https://dl.acm.org/doi/10.1145/1141911. 1141951

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952 [cs.CV] https: //arxiv.org/abs/2307.01952

Sagi Polaczek, Yuval Alaluf, Elad Richardson, Yael Vinker, and Daniel Cohen-Or. 2025. Neuralsvg: An implicit representation for text-to-vector generation. arXiv preprint arXiv:2501.03992 (2025).

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2022. DreamFusion: Text-to-3D Using 2D Diffusion. arXiv preprint arXiv:2209.14988 (2022). https: //arxiv.org/abs/2209.14988

Louis Pratt, Andrew Johnston, and Nico Pietroni. 2023. Bending the Light: Next Generation Anamorphic Sculptures. Computers & Graphics 114 (2023), 210–218. Zhiyu Qu, Tao Xiang, and Yi-Zhe Song. 2023. SketchDreamer: Interactive TextAugmented Creative Sketch Ideation. arXiv preprint arXiv:2308.14191. https: //arxiv.org/abs/2308.14191

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning Transferable Visual Models from Natural Language Supervision. In International Conference on Machine Learning (ICML). 8748–8763. https: //arxiv.org/abs/2103.00020

Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J. Mitra. 2021. Im2Vec: Synthesizing Vector Graphics Without Vector Supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7342–7351. https: //arxiv.org/abs/2102.02798

Leo Sampaio Ferraz Ribeiro, Tu Bui, John Collomosse, and Moacir Ponti. 2020. Sketchformer: Transformer-Based Representation for Sketched Structure. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 14153–14162.

Juan A Rodriguez, Abhay Puri, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. 2025. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the Computer Vision and Pattern Recognition Conference. 16175–16186.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695. https://arxiv.org/abs/2112.10752

Patsorn Sangkloy, Nathan Burnell, Cusuh Ham, and James Hays. 2016. The sketchy database: learning to retrieve badly drawn bunnies. ACM Trans. Graph. 35, 4 (2016). https://doi.org/10.1145/2897824.2925954

Guoyao Su, Yonggang Qi, Kaiyue Pang, Jie Yang, and Yi-Zhe Song. 2020. SketchHealer: A Graph-to-Sequence Network for Recreating Partial Human Sketches. In Proceedings of The 31st British Machine Vision Conference (BMVC).

Vikas Thamizharasan, Difan Liu, Matthew Fisher, Nanxuan Zhao, Evangelos Kalogerakis, and Michal Lukac. 2024. Nivel: Neural implicit vector layers for text-to-vector generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4589–4597.

Yael Vinker, Yuval Alaluf, Daniel Cohen-Or, and Ariel Shamir. 2023. CLIPascene: Scene Sketching with Different Types and Levels of Abstraction. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4146–4156. https: //arxiv.org/abs/2211.17256

Yael Vinker, Ehsan Pajouheshgar, Jessica Y Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir. 2022. Clipasso: Semantically-aware object sketching. ACM Transactions on Graphics 41, 4 (2022), 1–11.

Yael Vinker, Tamar Rott Shaham, Kristine Zheng, Alex Zhao, Judith E Fan, and Antonio Torralba. 2025. SketchAgent: Language-Driven Sequential Sketch Generation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 23355–23368.

Johan Wagemans, James H. Elder, Michael Kubovy, Stephen E. Palmer, Mary A. Peterson, Manish Singh, and Rüdiger Von der Heydt. 2012. A Century of Gestalt Psychology in Visual Perception: I. Perceptual Grouping and Figure–Ground Organization. Psychological Bulletin 138, 6 (2012), 1172.

Qiang Wang, Haoge Deng, Yonggang Qi, Da Li, and Yi-Zhe Song. 2023a. SketchKnitter: Vectorized Sketch Generation with Diffusion Models. In International Conference on Learning Representations. https://openreview.net/forum?id=4eJ43EN2g6l

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023b. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. In Advances in Neural Information Processing Systems, Vol. 36. 8406–8441. https://arxiv.org/abs/2305.16213

Kang Wu, Renjie Chen, Xiao-Ming Fu, and Ligang Liu. 2022. Computational Mirror Cup and Saucer Art. In ACM Transactions on Graphics, Vol. 41. 1–15. https://dl.acm. org/doi/10.1145/3516428

Ronghuan Wu, Wanchao Su, and Jing Liao. 2025. Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 23690–23700.

Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. 2023b. Iconshop: Text-guided vector icon synthesis with autoregressive transformers. ACM Transactions on Graphics (TOG) 42, 6 (2023), 1–14.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023a. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341 (2023).

Ximing Xing, Juncheng Hu, Jing Zhang, Dong Xu, and Qian Yu. 2025. Empowering LLMs to Understand and Generate Complex Vector Graphics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19487–19497. https://arxiv.org/abs/2412.11102

XiMing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu. 2023. DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models. In Thirty-seventh Conference on Neural Information Processing Systems. https: //openreview.net/forum?id=CY1xatvEQj

Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. 2024. Svgdreamer: Text guided svg generation with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4546–4555.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. ImageReward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems. 15903–15935.

Qian Yu, Yongxin Yang, Feng Liu, Yi-Zhe Song, Tao Xiang, and Timothy M. Hospedales.

2017. Sketch-a-Net: A Deep Neural Network That Beats Humans. International Journal of Computer Vision 122, 3, 411–425. https://arxiv.org/abs/1501.07873

Sicong Zang, Shuhui Gao, and Zhijun Fang. 2025. Generating Sketches in a Hierarchical Auto-Regressive Process for Flexible Sketch Drawing Manipulation at Stroke-Level. arXiv preprint arXiv:2511.07889 (2025).

Boheng Zhao, Rana Hanocka, and Raymond A. Yeh. 2023. AmbiGen: Generating Ambigrams from Pre-Trained Diffusion Model. arXiv preprint arXiv:2312.02967

(2023). https://arxiv.org/abs/2312.02967

A VLM Prompt Templates

This section provides the complete GPT-4o prompt templates used in our VLM-based evaluation pipeline (Sec. 3.4 of the main paper), corresponding to the two scoring phases illustrated in Fig. 5. All prompts request JSON-formatted output and are queried at temperature = 0 to ensure deterministic, reproducible scoring.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

bear

chicken

cow

dolphin

horse

lighthouse

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

cat

dog

angel

peacock

monkey

firefighter

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

fox

koala

rabbit

sheep

dog

flamingo

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

greek statue

cow

horse

pig

detective

giraffe

Fig. 14. Additional 2-phase progressive illusion results produced by our method.

- A.1 Phase 1 Prompt (Prefix Sketch Scoring)

Secondary consideration:

The Phase 1 prompt evaluates the prefix-only sketch (Sprefix), assessing how clearly it depicts the first target concept. Scoring priorities are, in order: (1) likeness to the target, (2) recognizability as a coherent object, and (3) single-object integrity.

3) Single-object integrity: The image should depict one main

object only (not multiple separate objects). Penalties (apply only as needed):

- - Not recognizable as any coherent object: final score must be <= 2.
- - Multiple distinct objects or clearly separate parts: cap score at <= 4.
- - Depicts a different object more strongly than {PHASE_LABEL}: cap score at <= 3.

### System Prompt:

You are evaluating a single-object sketch image for a pairwise illusion task.

Output format (STRICT JSON, no extra text): {"final_score": <number>,

User Prompt ({PHASE_LABEL} is replaced at runtime with the target concept string, e.g. rabbit):

"short_reason": "<max 18 words>"}

A.2 Phase 2 Prompt (Full Sketch vs. Delta Stroke Scoring)

You are evaluating a single-object sketch image for a pairwise illusion task.

The Phase 2 prompt receives two images simultaneously: the full sketch (Sfull, Image 1) and the delta strokes alone (Sdelta, Image 2). Beyond assessing how clearly the full sketch depicts the second target concept, the prompt enforces an integration check (anti-overlay criterion): the full sketch must be meaningfully more complete than the delta strokes alone, confirming that prefix strokes provide genuine structural scaffolding rather than being overwritten.

Target concept for this image: {PHASE_LABEL} You must output ONE final score from 0 to 10 (can be decimals) that reflects overall quality for this phase. Scoring priorities (most important first):

- 1) Likeness: How clearly the sketch depicts {PHASE_LABEL}.
- 2) Recognizability: Whether it looks like a coherent, recognizable object rather than random scribbles.

### System Prompt:

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

apple

horse

apple

pig

rabbit

apple

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

plush bear

rabbit

sheep

rabbit

cow

chicken

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

pig

chef

Einstein

angel

greek statue

angel

##### Fig. 15. Additional 3-phase progressive illusion results produced by our method.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

→chickenangel

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

→→→swanhamburgerbearcatrabbitgiraffe

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

(a) SketchDreamer (b) SketchAgent (c) Nano Banana Pro (d) Ours

##### Fig. 16. Additional qualitative comparisons.

You are evaluating a pairwise-illusion sketch with two images.

You are evaluating a pairwise-illusion sketch with two images:

User Prompt ({PHASE_LABEL} is replaced with the second target concept, e.g. horse; Image 1 = Sfull, Image 2 = Sdelta):

- Image 1: FULL sketch (phase_full) -- all strokes combined.
- Image 2: DELTA sketch (phase_delta) -- only strokes added in this phase.

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

Phase 1 Score: 0.9

Phase 2 Score: 0.85

Phase 1 Score: 0.9

Phase 2 Score: 0.85

- CLIP : 33.0; CLIP : 31.7; CLIP : 16.1 Φ IR : 0.920; Φ IR : 0.629; Φ IR : 0.048 HPS : 0.242; HPS : 0.224; HPS : 0.167

Ranking Score: 0.404; Rank: #1

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

CLIP : 30.9; CLIP : 33.0; CLIP : 17.6 Φ IR : 0.858; Φ IR : 0.532; Φ IR : 0.050 HPS : 0.222; HPS : 0.198; HPS : 0.171

Ranking Score: 0.212; Rank: #2

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

- CLIP : 34.5; CLIP : 31.4; CLIP : 22.9 Φ IR : 0.891; Φ IR : 0.584; Φ IR : 0.114 HPS : 0.253; HPS : 0.228; HPS : 0.169

Ranking Score: 0.765; Rank: #1

Ranking Score: 0.765; Rank: #1

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

Phase 1 Score: 0.85

Phase 2 Score: 0.85

Phase 1 Score: 0.85

Phase 2 Score: 0.85

CLIP : 33.0; CLIP : 33.0; CLIP : 20.1 Φ IR : 0.907; Φ IR : 0.438; Φ IR : 0.059 HPS : 0.230; HPS : 0.194; HPS : 0.165

Ranking Score: 0.7225; Rank: #3

Ranking Score: 0.7225; Rank: #3

Ranking Score: 0.200; Rank: #3

Ranking Score: 0.173; Rank: #4

Fig. 17. Metric-based ranking.

Fig. 18. GPT-based ranking.

|[Figure 264]<br><br>rabbit|
|---|
|[Figure 265]<br><br>elephant|

|[Figure 266]<br><br>chicken|
|---|
|[Figure 267]<br><br>monkey|

|[Figure 268]<br><br>carrot|
|---|
|[Figure 269]<br><br>rabbit|

|[Figure 270]<br><br>apple|
|---|
|[Figure 271]<br><br>pig|

|[Figure 272]<br><br>rabbit|
|---|
|[Figure 273]<br><br>horse|

|[Figure 274]<br><br>pikachu|
|---|
|[Figure 275]<br><br>sunflower|

Fig. 20. Extension on vector graph.

Fig. 21. Extension on colored strokes.

Fig. 19. Variable-width B-spline.

Target concept for the final result: {PHASE_LABEL} Your job: produce ONE final score from 0 to 10 (can be decimals) that reflects the overall quality of the FULL sketch as the final phase. Primary scoring priorities (most important):

- phase_full not clearly better than phase_delta: <= 3. Output format (STRICT JSON, no extra text): {"final_score": <number>,

"short_reason": "<max 20 words>", "integration_note": "<'full>>delta' / 'full>delta'

/ 'similar' / 'delta>full'>"}

- 1) Likeness on phase_full: How clearly phase_full depicts {PHASE_LABEL}.
- 2) Recognizability on phase_full: Whether phase_full depicts a coherent object. Dense or complex strokes are acceptable if they form a clear structure.
- 3) Integration check (anti-overlay): phase_full should be meaningfully better/more complete as {PHASE_LABEL} than phase_delta.

- If phase_delta alone already looks as complete as phase_full, or phase_full is not clearly better, apply a penalty.

Secondary considerations (supporting, not dominant):

- 4) Single-object integrity: should depict one main object only (not two separate objects).
- 5) Cleanliness: avoid excessive messy strokes. If phase_full is clearly recognizable as {PHASE_LABEL}, cleanliness is not a major concern.

B User Study Details

We conducted two user studies with 143 participants using Google Forms. Participants were aged approximately 20–50 years, and each participant completed both studies sequentially.

B.1 Study 1: Method Comparison

Participants were shown 10 questions, each presenting a different prompt pair. For each question, four illusion sketches labeled (A)– (D) were displayed side by side, generated by SketchDreamer (A)[Qu et al. 2023], our method (B), SketchAgent (C)[Vinker et al. 2025], and Nano Banana Pro (D), respectively. Each sketch was displayed as a triplet: Phase 1 (black), Phase 2 (black), and an animated GIF toggling between Phase 1 and Phase 2 (blue), allowing participants to directly perceive the structural transition. Participants were asked: “Which sketch best represents a good illusion sketch?” and judged based on

Hard constraints / caps:

- - phase_full not recognizable: final score <= 5.
- - phase_full wrong class: final score <= 3.
- - phase_full contains two distinct objects: <= 4.

|[Figure 276]<br><br>Initial strokes|
|---|

|[Figure 277]<br><br>Initial strokes|
|---|

|[Figure 278]<br><br>Initial strokes|
|---|

|[Figure 279]<br><br>Initial strokes|
|---|

|[Figure 280]<br><br>Initial strokes|
|---|

|Study1|[Figure 281]<br><br>Q. Which sketch best represents a good illusion sketch? (chicken → monkey)|
|---|---|
|Study2|Q. Which of the following sketches do you consider successful illusion sketches? (chicken → dog)<br><br>[Figure 282]|

Study1Study2

|[Figure 283]<br><br>rabbit|
|---|
|[Figure 284]<br><br>elephant|

|[Figure 285]<br><br>rabbit|
|---|
|[Figure 286]<br><br>elephant|

|[Figure 287]<br><br>rabbit|
|---|
|[Figure 288]<br><br>elephant|

|[Figure 289]<br><br>rabbit|
|---|
|[Figure 290]<br><br>elephant|

|[Figure 291]<br><br>rabbit|
|---|
|[Figure 292]<br><br>elephant|

Fig. 23. Five independent runs on the same prompt pair (rabbit → elephant) with a fixed random seed. Minor variations in stroke geometry arise from CUDA nondeterminism, but all runs yield recognizable and structurally coherent illusions at both phases.

giving a displacement radius of 0.05. Points are then scaled to pixel coordinates. The random seed is fixed to 0; minor variations may arise from CUDA nondeterminism.

Fig. 22. Representative survey questions from Study 1 (top) and Study 2 (bottom). In Study 1, participants selected the best illusion sketch among four methods: SketchDreamer (A), Ours (B), SketchAgent (C), and Nano Banana Pro (D). In Study 2, participants selected all sketches they considered successful from four candidates generated by our method.

Optimization. We use Adam (lr = 0.8, 2,000 iterations). Both SDS branches share a classifier-free guidance scale of 100 over a fully frozen Stable Diffusion v1.5 backbone. The overlay loss weight is 𝜆overlay = 0.1. Default stroke counts are 𝑘 = 16 (prefix) and 𝑁 = 32 (total), giving 16 delta strokes.

three criteria: (1) clear semantics at each phase, (2) smooth structural transition from Phase 1 to Phase 2, and (3) a perceptual reversal effect rather than mere stroke accumulation. Participants selected one of (A)–(D), or “Other” if none were satisfactory. A representative question is shown in Fig. 22 (top).

Overlay loss. The spatial buffer in Eq. (3) of the main paper is computed by applying Gaussian blur (𝜎 = 2.0, kernel 15 × 15) to the separately rasterized prefix and delta maps before the normalized inner product.

Runtime. All experiments run on a single NVIDIA RTX 4090, requiring ∼13 minutes for two-phase and ∼15 minutes for threephase illusions.

- B.2 Study 2: Ranking Pipeline Validation

Participants were shown 4 questions, each presenting our top-4 ranked outputs for the same prompt pair. Participants were asked: “Which of the following sketches do you consider successful illusion sketches?” Multiple selections were allowed, including selecting none. The same three criteria were provided as guidance. This study measures whether our ranking pipeline reliably surfaces highquality results. A representative question is shown in Fig. 22 (bottom).

- D Reproducibility and Variance Analysis

To assess the robustness of our method under CUDA nondeterminism, we ran the same prompt pair (rabbit → elephant) five times with a fixed random seed, varying only the CUDA execution order. Figure 23 shows all five results. Despite minor geometric variations across runs, all five outputs are recognizable at both phases and exhibit a clear structural transition, confirming that our method produces consistently high-quality illusions under fixed initialization.

- E Quantitative Ablation Studies

C Initialization and Implementation Details

Stroke representation. Each stroke is a single-segment cubic Bézier curve with 4 control points, yielding a learnable parameter tensor of shape 𝑁 × 4 × 2. Optimization is performed at a resolution of 512 × 512 with stroke width 2.5px. The final SVG is exported at 1024 × 1024 with stroke width 5px, preserving the stroke-to-canvas ratio.

We provide quantitative results for the three ablation studies discussed in Sec. 4.3 of the main paper, evaluating stroke initialization, optimization strategy, and overlay loss. Each setting is run on 5 prompt pairs with 30 illusions per setting, using identical prompts, stroke counts, and stroke widths. Results are summarized in Tab. 2.

Stroke initialization. Strokes are initialized near the canvas center (gathered strategy; see ablation Fig. 11 of the main paper). An anchor 𝑝0 is sampled from U([0.3, 0.7]2) in normalized coordinates, and eachsubsequentcontrolpointis perturbed by𝛿 ∼ U([−0.025, 0.025]2),

Optimization Strategy. As shown in Tab. 2 (Abl. 2 vs. 3), sequential optimization scores only 1.520 on concealment CLIP compared to 2.421 for joint optimization, with consistent trends observed across

Table 2. Ablation study. We evaluate three design choices: stroke initialization (scattered vs. gathered), optimization strategy (sequential vs. joint), and overlay loss. Abl. 1 vs. Ours confirms gathered initialization is critical for convergence; Abl. 2 vs. 3 shows joint optimization clearly beats sequential; Abl. 3 vs. Ours establishes overlay loss as the key enabler of structural concealment. Best in bold, second underlined.

|[Figure 293]<br><br>16s horse|
|---|
|[Figure 294]<br><br>plush 32s bear|
|[Figure 295]<br><br>56s chef|

|[Figure 296]<br><br>16s apple|
|---|
|[Figure 297]<br><br>32s sheep|
|[Figure 298]<br><br>56s Einstein|

|[Figure 299]<br><br>32s angel|
|---|
|[Figure 300]<br><br>20s rabbit|
|[Figure 301]<br><br>12s pig|

|[Figure 302]<br><br>32s statue|
|---|
|[Figure 303]<br><br>16s cow|
|[Figure 304]<br><br>10s rabbit|

|[Figure 305]<br><br>10s apple|
|---|
|[Figure 306]<br><br>32s angel|
|[Figure 307]<br><br>18s chicken|

|[Figure 308]<br><br>16s rabbit|
|---|
|[Figure 309]<br><br>48s Einstein|
|[Figure 310]<br><br>32s horse|

Init Optim. Overlay CLIP ↑ Concealment (structural) 𝐶semantic (g/s) (j/seq) Loss Avg min CLIP ↑ IR ↑ HPS ↑ CLIP ↑

- Abl. 1 scattered joint ✓ 28.121 3.759 1.019 0.033 0.921

- Abl. 2 gathered seq × 27.792 1.520 0.365 0.015 1.000

- Abl. 3 gathered joint × 30.690 2.421 0.765 0.027 1.000 Ours gathered joint ✓ 30.494 5.723 1.259 0.036 1.000

(a) Additive Mode (b) Subtractive Mode (c) Mixed Mode

Fig. 24. Three interaction paradigms enabled by our framework. Yellow arrows denote stroke addition; green arrows denote stroke removal. (a) Additive mode: concepts emerge through sequential stroke accumulation. (b) Subtractive mode: the full sketch is presented first; delta strokes are removed in reverse order to reveal earlier concepts, with no modification to the underlying optimization. (c) Mixed mode: addition and subtraction are interleaved within a single sequence, where the perceived concept at each step is determined by which stroke subset is rendered.

IR and HPS metrics. This confirms that freezing prefix strokes commits them to a rigid local minimum for Concept A, leaving delta strokes to build on an incompatible foundation. Joint optimization navigates two competing gradient fields simultaneously, discovering a common structural subspace where features serve dual roles, and is therefore essential for high-quality progressive illusion sketches.

Stroke Initialization. Tab. 2 (Abl. 1 vs. Ours) shows that scattered initialization scores only 3.759 on concealment CLIP compared to 5.723 for centered gathered initialization, with IR and HPS further corroborating this gap. Since our objective function is highly nonconvex, spatial concentration is critical for convergence; scattered strokes fail to form the coherent spatial structure required to simultaneously serve two semantic interpretations. We therefore adopt centered gathered initialization to balance density with spatial coverage, avoiding potential boundary clipping.

controlled by the rendered subset, order, or a minor reformulation of the optimization objective.

Overlay Loss. Removing Loverlay drops concealment CLIP from 5.723 to 2.421 (Tab. 2, Abl. 3 vs. Ours), the most significant drop among all ablated components, with IR and HPS showing similarly pronounced degradation. Without this constraint, semantic guidance alone fails to prevent spatial redundancy, causing delta strokes to clutter the prefix rather than structurally integrating with it. Loverlay enforces spatial complementarity by penalizing overlap between prefix and delta strokes, ensuring that prefix strokes serve as essential structural scaffolding for the final concept rather than being obscured. This confirms that geometric constraints are indispensable for generating clean progressive illusions.

F Applications: Additive and Subtractive Modes.

Our framework supports three interaction paradigms beyond standard generation (Fig. 24). (a) Additive mode progressively accumulates strokes across phases. (b) Subtractive mode begins from the full sketch and progressively removes delta strokes (𝑆𝛿2, then 𝑆𝛿1) to transition concepts, requiring no re-optimization—only a reversal of rendering order. (c) Mixed mode interleaves both directions within one sequence. For apple→angel→chicken, adding 𝑆𝛿1 ∪𝑆𝛿2 to the prefix yields angel, while subsequently removing 𝑆𝛿2 recovers chicken. For rabbit→Einstein→horse, a minor optimization adjustment ensures that adding𝑆𝛿 to𝑆prefix yields Einstein, while subtracting 𝑆prefix instead reveals horse as an emergent concept. These variants demonstrate that semantic interpretation can be flexibly

