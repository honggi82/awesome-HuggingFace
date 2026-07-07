## When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models

Zhengyang Sun1∗, Yu Chen1∗, Xin Zhou1,3, Xiaofan Li2, Xiwu Chen3†, Dingkang Liang1†, Xiang Bai1

1 Huazhong University of Science and Technology, 2 Zhejiang University, 3 Afari Intelligent Drive

{zysun,yuchen66,dkliang}@hust.edu.cn Project webpage: https://h-embodvis.github.io/NUMINA/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

# arXiv:2604.08546v1[cs.CV]9Apr2026

Wan2.1

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

NUMINA (ours)

“Three cats playfully chasing each other through the house.” “A wolf and a fox running through the snow-covered forest.”

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Wan2.1

[Figure 17]

[Figure 18]

[Figure 19]

NUMINA (ours)

“Four hikers are celebrating reaching the summit while planting two flags.”

“Four builders and two excavators are engaged in the construction of one brick wall.”

Figure 1. We present NUMINA, a training-free framework that alleviates the misalignment between precise numerals and visual instances in text-to-video diffusion models. We significantly improve counting accuracy while maintaining natural layouts and temporal coherence.

#### 1. Introduction

#### Abstract

Recent advances in text-to-video (T2V) models [3, 5, 10, 36] have greatly enhanced the ability to generate coherent and high-quality videos from textual descriptions. This progress is largely facilitated by the Diffusion Transformer (DiT) architecture [53], enabling scalable training and stronger semantic alignment. By making high-quality video creation more accessible, these models enable emerging applications across entertainment, education, and other domains.

Text-to-video diffusion models have enabled open-ended video synthesis, but often struggle with generating the correct number of objects specified in a prompt. We introduce NUMINA , a training-free identify-then-guide framework for improved numerical alignment. NUMINA identifies prompt-layout inconsistencies by selecting discriminative self- and cross-attention heads to derive a countable latent layout. It then refines this layout conservatively and modulates cross-attention to guide regeneration. On the introduced CountBench, NUMINA improves counting accuracy by up to 7.4% on Wan2.1-1.3B, and by 4.9% and 5.5% on 5B and 14B models, respectively. Furthermore, CLIP alignment is improved while maintaining temporal consistency. These results demonstrate that structural guidance complements seed search and prompt enhancement, offering a practical path toward count-accurate text-to-video diffusion. The code is available at https: //github.com/H-EmbodVis/NUMINA.

Despite the significant progress, most state-of-the-art T2V models [37, 59] primarily emphasize visual fidelity [11, 16], motion smoothness [62, 67], and overall semantic alignment [17, 28]. However, they often struggle with precise numerical alignment between prompts and objects, as shown in Fig. 1. This limitation, where models fail to represent object counts accurately, hinders their reliability in precision-sensitive applications like instructional visualization. This naturally raises a question: what prevents T2V models from achieving precise numerical alignment?

To probe this limitation, we analyze Wan2.1-1.3B [59], a

* Equal contribution. † Project lead. Corresponding author.

Prompt: “Three men in black jackets are running in the park.”

lizes targeted adjustments to refine the latent layout under count constraints, heuristically considering spatial relationships between instances. The subsequent regeneration process is guided by this adjusted layout, improving count accuracy without degrading non-numerical attributes.

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

To systematically evaluate NUMINA, we also introduce the CountBench benchmark, comprising 210 prompts covering counts from 1-8 for scenes involving 1-3 object categories. On CountBench, NUMINA improves by 7.4% counting accuracy on Wan2.1-1.3B [59] and by 5.5% on a larger 14B model. Moreover, we observe a consistent increase in CLIP score for various baselines, suggesting that enforcing correct instance counts strengthens overall textvideo alignment and yields cleaner scene layouts. The successful integration with inference acceleration techniques like EasyCache [73] also reduces inference overhead.

Video

n. “men”

n. “jackets”

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

v. “are running”

adj. “black” num. “Three”

- Figure 2. Visualization of the cross-attention maps corresponding to different texts in the prompt. The highlighted areas represent a stronger level of attention between the pixels and the text.

representative and community-recognized T2V model, and identify two contributing factors: 1) Semantic weakness. Numerical tokens exhibit diffuse cross-attention responses compared to other word types. As shown in Fig. 2, the cross-attention maps for nouns, verbs, and adjectives produce strong, localized patterns. This suggests insufficient semantic grounding of numerals in the latent space, weakening the model’s ability to encode count constraints during generation. 2) Instance ambiguity. The heavily downsampled spatiotemporal latent space in DiT-based architectures [20, 60] limits the separability of individual object representations, making stable count control difficult. While retraining could potentially address these issues, it is computationally prohibitive and requires large-scale datasets with precise numerical annotations, which are non-trivial to construct. Moreover, enhancing the model’s attention to numerical tokens demands careful rebalancing of the attention mechanism to maintain performance on other critical attributes such as visual quality and motion coherence. These constraints motivate us to pursue alternative solutions to enhance numerical alignment during generation.

Our major contributions can be summarized as follows:

- 1) We reveal that the attentions in T2V models expose critical visual information related to the number of instances.
- 2) We introduce a training-free framework that guides modifications during generation, enhancing the alignment between object counts and prompt instructions. 3) We demonstrate that NUMINA advances count-accurate text-to-video generation, highlighting its effectiveness and practicality.

#### 2. Related Work

##### 2.1. Diffusion Transformer for Video Generation

Text-to-video (T2V) generation has rapidly progressed from early 3D U-Net architectures [9, 22, 23, 32] to scalable Diffusion Transformer (DiT) frameworks [48, 53]. Built on DiT [72], leading video generation models [24, 30, 34, 59] have achieved remarkable capabilities in synthesizing coherent, high-fidelity videos. They effectively inject textual semantics via attention mechanisms and operate in compressed latent spaces for efficiency [6, 20]. Despite these advantages, current T2V models often exhibit weak grounding of textual numerals and insufficient instance separability, resulting in numerical misalignments during generation.

Therefore, we propose NUMINA, a training-free video generation framework that enhances numerical alignment in T2V generation while preserving visual fidelity and temporal coherence, as shown in Fig. 1. We explore the model’s latent ability to separate object instances, while allowing natural instance-level addition and removal. NUMINA introduces an identify-then-guide paradigm, which yields accurate cardinalities and retains appearance, motion, and semantics. As the intervention is lightweight and trainingfree, it is broadly applicable across various T2V models.

##### 2.2. Video Editing for T2V Models

Recent advances in T2V models have catalyzed a surge in video editing methods [2, 27, 31, 33, 63, 69]. Existing approaches predominantly focus on motion control [18, 49, 65], style transfer [61, 66, 71], appearance editing [45, 50, 54], etc. For example, VideoGrain [68] supports multi-region and multi-grained editing conditioned on prompts via an attention modulation. Meanwhile, some researchers focus on video inpainting [14, 19, 57]. OmnimatteZero [55] and DiffuEraser [35] remove objects and their associated visual effects via video inpainting. However, these methods overlook instance-level addition, failing to align textual numerals with visual content. Moreover, most

Specifically, in the first phase, NUMINA operates early during denoising to detect misalignment between numeral tokens and the evolving latent layout (i.e., the spatial distribution of object-related activations). It performs a dynamic selection of attention heads using an object discriminability criterion, then applies a cluster-based algorithm to obtain precise segmentation. In the second phase, NUMINA uti-

Phase 2: Numerically Aligned Video Generation

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

DiT Blocks

[Figure 46]

[Figure 47]

###### …

Guided Cross-Attention

… …

|Text prompt<br><br>“Three people are walking one dog.”|
|---|

“people” “dog”

###### Phase 1: Numerical Misalignments Identification

Self-Attention Cross-Attention

Layout-Guided Generation

[Figure 48]

+ + +

[Figure 49]

-

[Figure 50]

+ -

+ + +

+ Attention Boost

𝑙𝑙∗-th DiT Block

-

Layout Localization

Layout Refinement

…

…

…

+

-

- Attention Suppression

𝑡𝑡 = 𝑡𝑡∗

Attention Head Selection Cluster

###### “people”: 3 “dog”: 1

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

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

T2V CrossAttention

[Figure 91]

“people”

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

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Object Addition

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

“dog”

Object Removal

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Numerically Aligned Layout

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Object Layout 𝐌𝐌𝑇𝑇

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

SelfAttention

[Figure 157]

[Figure 158]

[Figure 159]

𝐌𝐌𝑇𝑇,𝑓𝑓

Clustered Masks

Frame-wise Selected Heads

Multi-Head Attention

- Figure 3. The pipeline of our NUMINA follows a two-phase paradigm. Given a text prompt containing numerals, we first perform the numerical misalignment identification to extract explicitly countable layouts from attention maps. Based on the layout, we further conduct a refinement and a layout-guided generation for the numerically aligned video generation.

methods operate in a video-to-video setting and rely on object masks obtained from segmentation models [12, 29].

prompt is injected mainly through the multi-head crossattention mechanism. Given spatiotemporal latent features X ∈ RN×d, and text embeddings c ∈ RL×d, the head-wise cross-attention for head h is computed as follows:

##### 2.3. Counting in Vision and Generation

While attention mechanisms and vision-language alignment have proven effective for object counting and localization [38–42], enforcing such precise numerical constraints in generative models remains challenging. Recently, CountGen [4] attempts to optimize count-correct text-to-image (T2I) generation by detecting miscounts and employing a learned layout-completion model. However, it is primarily designed for static images, relies on SDXL-specific observations, and requires training additional networks alongside explicit masks for inference-time optimization.

QK⊤ √dh

, (1)

Ch = Softmax

where Q is projected from the visual latent X, K comes from text embeddings c, and dh = d/n is the per-head dimension. The resulting attention map Ch ∈ RN×L encodes the relevance between each visual and text token.

The cross-attention mechanism is effective for localized attributes due to its per-query matching, but struggles with global constraints. As a result, numeral tokens often exhibit diffuse, low-contrast activations, as in Fig. 2. This suggests that standard cross-attention alone may be insufficient to faithfully encode global numerical constraints, implying that simply increasing training data or model size may not be sufficient to address the problem.

In comparison, our training-free approach provides global guidance for T2V models without requiring input videos, spatial masks, or auxiliary re-layout networks. Importantly, it readily adapts to text-to-video generation while preserving strict temporal consistency.

#### 3. Preliminary

In this paper, we alleviate this gap by introducing a training-free framework that explicitly exposes prompt inconsistencies early in denoising and then corrects them. By extracting an instance-aware layout from attentions and enforcing the desired count during regeneration, we provide global guidance for numerical alignment.

Recent text-to-video (T2V) models [7, 37, 59] mainly employ the Diffusion Transformer (DiT) [53] together with flow-matching [43, 46] or diffusion sampling [51, 53] pipelines that evolve Gaussian noise into a video latent conditioned on a text prompt. In each vanilla DiT block, the

ℓ⋆. This balances the emergence of instance contours with the injection of high-level semantics from the prompt, resulting in attention maps with usable structure and limited noise. We process self- and cross-attention separately, due to their distinct roles: self-attention organizes spatial structure, while cross-attention injects prompt semantics. For simplicity, we discuss head selection for a single frame, but the same procedure applies to every latent frame.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

|[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]|
|---|

0 1 2 3

|[Figure 179]|
|---|

[Figure 180]

[Figure 181]

[Figure 182]

4 5 6 7

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

푖-th Frame

8 9 10 11

(a) Diverse patterns in different attention heads

To score self-attention heads for instance separability, we first calculate the attention map for each head h. For a single frame, we extract the HW × HW normalized attention matrix and apply PCA to project its row vectors onto their top three principal components. The resulting components are reshaped into an H × W × 3 tensor and converted to grayscale, yielding the final map SAh ∈ RH×W for evaluation. We then design three complementary scores to measure the separability: 1) Foreground-background separation S1h is measured by the standard deviation of intensities. 2) Structural richness S2h is quantified by partitioning SAh into non-overlapping blocks, computing the summed feature for each block, and taking the variance across blocks. This metric favors maps with intermediatescale spatial quality, penalizing both over-smoothing and degeneracy. 3) Edge clarity S3h is captured by applying the Sobel operator to SAh and averaging the gradient magnitude over all pixels, which emphasizes clear object contours that support instance separation. The overall discriminability score for head h is a weighted sum formed as:

Low High

[Figure 187]

|[Figure 188]|
|---|

[Figure 189]

[Figure 190]

[Figure 191]

…

(b) Instance separability ranking from low to high

- Figure 4. The PCA visualization of self-attention maps for Wan2.1-1.3B. (a) Different attention heads naturally capture diverse spatial patterns. (b) We select the head with the highest instance separability for countable layout construction.

#### 4. Method

As shown in Fig. 3, we utilize a two-phase pipeline for the training-free framework, following an identify-thenguide paradigm. We first perform a pre-generation step using an input prompt and a sampled noise vector to establish the scene layout localization (Sec. 4.1). Then, we re-generate numerically aligned video through the modified layout guidance (Sec. 4.2). Overall, our NUMINA transforms implicit attention into an explicit layout signal, guiding the generation process to produce more accurate counts.

###### S SAh = S1h + S2h + γ S3h, (2)

##### 4.1. Numerical Misalignment Identification

where γ > 0 balances the contribution of edge clarity against the global contrast and intermediate-scale structure. Finally, as shown in Fig. 4(b), we select the self-attention

The first phase identifies count discrepancies by analyzing the DiT’s attention mechanisms. Since attention patterns are distributed across heads, we select the most instance-discriminative self-attention head and the most text-concentrated cross-attention head, and then fuse their maps to obtain an instance-level layout that is explicitly countable, allowing for direct comparison between the estimated cardinality and the prompted numeral.

∗

map As = SAh

s by h∗s = arg maxh S SAh , providing a layout with highest instance separability.

For each target noun token T in the prompt, we obtain its cross-attention map CAhT ∈ RH×W from head h. We empirically find that the peak activation effectively indicates the model’s alignment of the token with a specific visual region. Since these maps are softmax normalized, a higher maximum value CTh = maxx,y CAhT(x,y) signifies a more concentrated response. For token T, we select its best crossattention head h∗c(T) = arg maxh CTh and denote the corresponding map as Ac,T = CAh

Instance-separable Attention Patterns. To assess instance awareness, we analyze multi-head attention during early denoising. We observe substantial head-wise diversity in spatial focus, category selectivity, and instance separability. As shown in Fig. 4(a), within the same layer and timestep, many heads are spatially diffuse, some retain coarse class-level structure, and only a small subset clearly delineates boundaries between instances of the same category. This motivates a dynamic head-selection strategy, as naive head averaging or random selection produces blurred maps that fail to separate instances.

∗ c(T)

T . This computationally efficient criterion stably identifies relevant regions across scenes without extra processing.

After the selection, we assign one self-attention head h∗s for an instance-discriminative spatial scaffold and one cross-attention head h∗c(T) per noun token for focused semantic alignment to each frame. These maps jointly yield a countable foreground layout used to compare estimated object cardinality with the prompt numerals.

Attention Head Selection. Based on these observations, at a reference timestep t⋆ during the pre-generation trajectory, we select attention heads from an intermediate layer

Countable Layout Construction. To derive a countable layout for a target noun T, we fuse the selected instancediscriminative self-attention map As and the corresponding text-aligned cross-attention map Ac,T for each frame.

First, spatial proposals {ri} are generated by partitioning the self-attention map As into contiguous regions using clustering [13]. Meanwhile, Ac,T is processed by suppressing values below a 0.1 peak-ratio threshold to isolate peak responses, and density-based clustering [15] is then applied to group the resulting map, forming the focus mask F.

We then filter these proposals to construct the final layout. For each region ri, we compute its semantic overlap score So with the focus mask F as:

So(ri,F) = |ri ∩ F| |ri|

, (3)

where | · | denotes the area (number of pixels). A region is retained as a valid instance if So ≥ τ. The final layout MT is then constructed as a 2D semantic map, initialized with a background label lbg. Pixel (with coordinate p) belonging to any valid region is assigned the corresponding class lT:

MT(p) =

lT, if p ∈ i:So(ri,F)≥τ ri lbg, otherwise

. (4)

By construction, MT is a semantic map containing disjoint foreground regions, where each region ideally corresponds to a single object instance of category T. The number of foreground regions, |{i : So(ri,F) ≥ τ}|, provides an explicit object count. Thus, as in Fig. 3, the first stage enables direct identification of numerical misalignments.

##### 4.2. Numerically Aligned Video Generation

After identifying numerical misalignment using the layout MT, this phase alleviates count errors during generation. Since the initial layout MT reflects an intrinsic coupling between the sampled noise and the prompt semantics, aggressive manipulation of the latent space can degrade realism. We adopt a conservative two-step approach: Layout Refinement to add or remove object instances at the layout level, and Layout-Guided Generation to steer the re-synthesis process to adhere to this corrected layout.

Layout Refinement. This process refines the per-frame layout map MT,f (layout mask of the f-th frame for noun T) to match the target count kT parsed from the prompt. Let mT,f be the current number of instance regions in MT,f. The layout is corrected at the instance level until mT,f = kT, guided by a principle of minimal structural change.

For object removal, we erase the smallest region of category T in MT,f as it incurs minimal perturbation to the existing visual composition. All pixels within this region are reassigned to the background label. This simple strategy reduces visual impact because small instances carry less spatial support, and it preserves the dominant layout.

For object addition, we insert a new instance using a layout template. If at least one region of category T already exists, the smallest existing region is copied as the template Ci to preserve the category’s intrinsic scale and shape. If no such region exists (i.e., mT,f = 0), a circle with radius r is used as Ci. This template defines only the instance’s geometry, while the appearance is not constrained.

The template Ci is then placed at an optimal location in each frame f by minimizing a heuristic cost over a uniform grid of candidate centers. Let c = (cx,cy) be the candidate center of Ci, (c0x,c0y) be the geometric center of MT,f (which defaults to the spatial center of the frame if MT,f is empty), and (c′x,c′y) be the instance’s center in the previous frame. The heuristic cost promotes conservative insertion and is composed of three terms defined as:

Co = Ci ∩ MT,f , Cc = (cx − c0x)2 + (cy − c0y)2, Ct = ⊮[f > 1] (cx − c′x)2 + (cy − c′y)2 ,

(5)

where ⊮[f > 1] equals 1 when f > 1 and 0 otherwise. The overlap term Co penalizes collisions with the existing layout. The center term Cc encourages plausible placements close to the existing spatial distribution. The temporal term Ct ensures the inserted instance remains stable across frames. The total cost C is a weighted sum as:

C(c) = Co + Cc + λCt, (6)

where λ > 0 is a balancing coefficient. The optimal center c∗ = arg minc C(c) is selected, and MT,f is updated by assigning the class label to the pixels in Cc∗. This cycle is repeated until the count mT,f matches kT.

The resulting refined layout M˜ T,f preserves the original spatial organization while correcting count errors, serving as the control guidance for the subsequent re-generation.

Layout-Guided Generation. Finally, the refined layout M˜ T,f guides the regeneration process through a trainingfree modulation of the cross-attention: softmax(Spre + B)V, where Spre = QK

⊤

√dh represents the pre-softmax attention scores and B is an initially zero bias term. To enforce the corrected layout, we strategically modify either Spre or B for each attention head. These modifications are scaled by a monotonically decreasing intensity function δ(t) at the t-th denoising step, applying stronger guidance early in the denoising process when the object layout is established, and weaker guidance later to preserve fine-grained details.

For object removal, we perform an attention suppression by modifying the bias B for regions ∆Mrem corresponding to category token T to a large negative constant. This forces the post-softmax attention weights in these regions to near zero, effectively suppressing unwanted instance generation.

For object addition, we boost attention in the new area ∆Madd, and the boost strategy depends on the template’s

origin. If the new instance is obtained from the manual circle template, we modify the bias term B by setting it to k · δ(t) for all p ∈ ∆Madd, where k is a scalar coefficient. This provides a strong, externally-imposed attention signal. Conversely, if the instance is templated from an existing reference region Mref, we directly overwrite the presoftmax scores in Spre to promote consistency. Specifically, for each frame f, we compute the mean pre-softmax score a¯f from Mref and then overwrite the scores in Spre for all p ∈ ∆Madd at frame f with a¯f ·δ(t). This transfers the pretrained attention properties of the existing object onto the new location, with δ(t) modulating the intensity.

This process is applied independently to each category T, and the localized guidance ensures stable control superposition while preserving overall visual fidelity.

#### 5. Experiments

##### 5.1. Experiment Setup

Benchmark. Existing text-to-video (T2V) benchmarks [25, 47, 64] often overlook precise numerical generation, focusing instead on visual quality [26], temporal coherence [58], or general text alignment [21]. While T2VCompBench [56] includes a numeracy subset, its formulaic structure (“[X] and [Y]”) limits its ability to represent diverse user prompts.

To evaluate numerical alignment in T2V generation, we introduce CountBench, comprising 210 prompts that evaluate numerical fidelity in complex scenarios. These prompts encompass a range of conditions, including instance counts from 1 to 8 and compositions involving 1 to 3 object categories, systematically evaluating a T2V model’s ability to manage multiple numerical constraints. We initially generated prompt candidates using GPT-5 [52] to ensure simple and dynamic descriptions, followed by a manual review to eliminate repetitive or illogical prompts.

Evaluation metrics. We employ three complementary metrics to quantitatively assess numerical alignment and generation quality. 1) Counting Accuracy (CountAcc) measures adherence to numerical instructions by scoring a target object class as 1 if the detected count matches the prompt, and 0 otherwise. For each frame, scores are averaged across classes, and then averaged across frames to produce the video-level score. 2) Temporal Consistency (TC) measures the stability of generated counts. For each adjacent frame pair, a class scores 1 if counts are identical, and 0 otherwise, with the final score averaged over all pairs and classes. 3) The CLIP score [21] evaluates semantic alignment between generated videos and text prompts by averaging frame-wise CLIP scores. The CountAcc and TC are computed using GroundingDINO [44] to obtain per-frame object counts via category-specific text prompts.

Implementation Details. We implement NUMINA using the official Wan T2V series [59] with 50 denoising steps.

Table 1. Comparison of NUMINA with other practical strategies. We report Counting Accuracy (CountAcc), Temporal Consistency (TC), and CLIP Score on Wan [59] of varying scales.

Models CountAcc (%) TC (%) CLIP Score

- Wan2.1-1.3B [59] (81 frames, 832×480)

- Wan2.1-1.3B 42.3 81.2 33.9

+ Seed search 45.5(+3.2) 82.3(+1.1) 34.6(+0.7) + Prompt enhancement 47.2(+4.9) 82.1(+0.9) 33.7(-0.2) + NUMINA (ours) 49.7(+7.4) 83.4(+2.2) 35.6(+1.7)

Wan2.2-5B [59] (81 frames, 1280×704)

- Wan2.2-5B 47.8 85.0 34.3

+ Seed search 48.8(+1.0) 84.7(-0.3) 34.1(-0.2) + Prompt enhancement 49.0(+1.2) 84.3(-0.7) 34.2(-0.1) + NUMINA (ours) 52.7(+4.9) 85.0(+0.0) 34.7(+0.4)

Wan2.1-14B [59] (81 frames, 1280×720) Wan2.1-14B 53.6 83.3 34.2

+ Seed search 56.1(+2.5) 83.5(+0.2) 34.0(-0.2) + Prompt enhancement 56.9(+3.3) 84.3(+1.0) 34.0(-0.2) + NUMINA (ours) 59.1(+5.5) 84.0(+0.7) 34.4(+0.2)

For the numerical misalignment identification stage, we extract attention at timestep t⋆=20 and layer ℓ⋆=15. All baseline methods share identical inference settings to ensure fair comparison. Experiments on the 1.3B model are conducted on NVIDIA 4090 GPUs, while larger models (5B and 14B) are evaluated on A800 GPUs.

##### 5.2. Main Results

We conduct experiments on leading video generation models, with Wan models [59] across different scales, i.e., Wan2.1-1.3B, Wan2.2-5B, Wan2.1-14B, with a fixed seed 1. Since the field of count-aligned T2V generation remains unexplored, we compare our NUMINA against the original models and the two most practical and viable training-free strategies within existing generation workflows: 1) Seed search, a common “trial-and-error” practice involving generating 5 videos with random seeds (1-5) per prompt and selecting the best result based on counting accuracy; 2) Prompt enhancement, which enriches object descriptions with detailed attributes using a Large Language Model [1].

As shown in Tab. 1, NUMINA consistently and significantly improves counting accuracy (CountAcc) across all baselines. For instance, Wan2.1-1.3B achieves only 42.3% accuracy with a single trial, while Seed search and Prompt enhancement offer marginal improvements to 45.5% and 47.2%, respectively. In contrast, NUMINA substantially boosts the accuracy to 49.7% with only a single trial and a simple prompt. This superior performance extends to larger models, where our method outperforms the 5B and 14B baselines by 4.9% and 5.5%, respectively. Notably, our method enables the 1.3B model (49.7%) to surpass the counting accuracy of the much larger Wan2.2-5B (47.8%), highlighting its efficiency and effectiveness.

Furthermore, our method improves counting accuracy

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

...

[Figure 196]

Sora2

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

...

[Figure 201]

Veo3.1

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

...

[Figure 206]

Grok Imagine

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

...

[Figure 211]

NUMINA (ours)

Prompt: “Three cyclists ride through a trail with three mountain goats.”

Figure 5. Qualitative comparison of NUMINA with the most advanced commercial models.

85

53

Baseline (Wan2.1-1.3B)

51.3

71.0

51

Prompt enhancement

[Figure 212]

51.1

70

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

50.9

60.7

Seed search

50.4

50.2

68.7

49

Accuracy(%)

[Figure 218]

49.7

[Figure 219]

NUMINA (ours)(ours)

Accuracy(%)

55

47

48.2

41.6 40.7

[Figure 220]

[Figure 221]

46.5

###### (ours)

[Figure 222]

NUMINA Ours

[Figure 223]

40

[Figure 224]

45

[Figure 225]

44.5

[Figure 226]

[Figure 227]

30.3

Baseline

38.6

44.9

26.3

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

43

20.7

[Figure 232]

31.7

25

[Figure 233]

[Figure 234]

41.8

[Figure 235]

[Figure 236]

41

[Figure 237]

[Figure 238]

18.6 15.2 11.3

[Figure 239]

[Figure 240]

10 15 20 25 30 35 40 45 50

[Figure 241]

10

[Figure 242]

2 3 4 5 6 7 8

푡*

Reference Timesteps

Figure 7. Ablation on the reference timesteps t⋆ for head selection.

Ground Truth Count

Figure 6. The per-numeral accuracies for Wan2.1-1.3B.

ready performs well for simple prompts requesting a few objects (e.g., 68.7% for two objects), as this relies more on category recognition than precise counting. However, its performance rapidly degrades as the ground truth count increases. For prompts requiring three objects, the baseline accuracy plummets to 44.5%. In contrast, NUMINA achieves a 16.2% improvement, significantly outperforming both Seed search and Prompt enhancement. This advantage is even more pronounced in high-count scenarios. For eight objects, the baseline accuracy is a mere 11.3%, while NUMINA makes a significant improvement by nearly doubling this accuracy to 20.7%. These results demonstrate that our NUMINA provides a scalable solution for complex, high-count scenarios, proving far more effective than augmentation strategies.

while maintaining competitive overall generation quality. As shown in Tab. 1, we observe a consistent increase in the CLIP score, particularly for smaller models (e.g., an improvement from 33.9 to 35.6 for the 1.3B model). This indicates that enforcing correct spatial layouts and appending missing instances strengthens video-text semantic alignment. Moreover, we find that even state-of-the-art models can exhibit instability in Temporal Consistency (TC). Despite actively adding or removing objects, our method maintains this temporal coherence, and even notably improves it to 84.0% for the 14B model. This highlights that our instance-level guidance is stable and does not introduce flickering or temporal artifacts, resulting in numerically accurate and temporally smooth videos.

Qualitative Results. We further present qualitative comparisons with the most advanced commercial T2V generation models in Fig. 5. It is worth noting that even these cutting-edge models frequently fail to satisfy the precise numerical constraints specified in the prompt. In contrast, our method reliably produces the exact requested counts while preserving natural layouts and temporal coherence.

##### 5.3. Analysis and Ablation Study

We conduct ablation studies using CountBench prompts, each generating one video with Wan2.1-1.3B unless otherwise specified. The default settings are marked in green .

Analysis on reference timesteps. We analyze the impact of the reference timestep t⋆ for attention head selection. As shown in Fig. 7, the CountAcc rises quickly and reaches

Per-numeral Accuracy Breakdown. Fig. 6 details a pernumeral breakdown for Wan2.1-1.3B. The 1.3B model al-

Table 2. Ablation on the layout construction method.

Method CountAcc (%) TC (%) Baseline 42.3 81.2

GroundingDINO 47.5(+5.2) 82.8(+1.6) Attention (ours) 49.7(+7.4) 83.4(+2.2)

Table 3. Ablation on the components of the layout refinement cost.

Co Cc Ct CountAcc (%) TC (%)

Baseline 42.3 81.2

✓ 45.1(+2.8) 82.1(+0.9) ✓ ✓ 46.9(+4.6) 82.3(+1.1) ✓ ✓ 48.9(+6.6) 83.1(+1.9) ✓ ✓ ✓ 49.7(+7.4) 83.4(+2.2)

Table 4. Ablation on the self-attention head selection strategy.

Method CountAcc (%) TC (%) Baseline 42.3 81.2

Random 44.1(+1.8) 82.6(+1.4) All average 43.0(+0.7) 82.4(+1.2) Top-3 48.2(+5.9) 82.5(+1.3) Top-2 49.4(+7.1) 83.3(+2.1) Top-1 49.7(+7.4) 83.4(+2.2)

Table 5. Additional time and VRAM cost.

Method wall-clock (s) VRAM (GB) CountAcc (%)

Wan2.1-1.3B 292 14.3 42.3 NUMINA 431 16.3 49.7 NUMINA + EasyCache [73] 355 16.3 49.4

49.7% at timestep 20, indicating that early denoising steps are needed to form instance-separable attention. Increasing t⋆ to 40 yields only a 3.2% relative gain over t⋆=20 but doubles the pre-generation cost. For t⋆>40, we observe an accuracy decline, possibly due to fragmented or over-fused late-stage attention reducing separability. We set t⋆=20 by default as a favorable accuracy-efficiency trade-off.

Efficacy of countable layout construction. Tab. 2 presents the quality of the countable layout MT. For fairness, we perform a truncated pre-generation with t⋆=20. We then derive our layout from selected self-/cross-attention heads and, in parallel, use GroundingDINO [44] on the same frames to generate a per-category layout. Both layouts are used in the second phase. Our attention-derived layout outperforms the detector-derived layout by 2.2%, likely because it is native to the DiT’s latent and better captures nascent instance structures. More importantly, both layoutguided methods substantially outperform the baseline, validating the effectiveness of our Layout-Guided strategy.

Analysis on the layout refinement cost. We also assess the components of our layout refinement cost, which are designed to guide object addition. As shown in Tab. 3, using only the primary overlap cost (Co) brings a promising 2.8% accuracy improvement, demonstrating the layoutguided approach’s effectiveness. Building on this, adding the center cost (Cc) for plausible spatial placement further improves accuracy to 46.9%. Meanwhile, the temporal cost (Ct) yields a more substantial gain to 48.9%, highlighting the importance of temporal stability. Combining all three costs in NUMINA achieves the highest accuracy of 49.7%, confirming that these heuristic costs are complementary and enable stable and accurate layout correction M˜ T,f.

Analysis on the self-attention head selection. We then validate our strategy of selecting the single best selfattention head using the score S(SAh). As shown in Tab. 4, selecting a single random head (44.1%) or averaging all heads (43.0%) provides only a marginal benefit over the baseline. In contrast, our Top-1 selection based on S(SAh)

significantly boosts accuracy to 49.7%. This demonstrates that our scoring metric is highly effective at identifying the most useful head. Furthermore, performance consistently degrades as we average fewer discriminative heads, strongly confirming our hypothesis that instance-separable information is a sparse property held by only a few heads.

Analysis of computational overhead. Besides, NUMINA is compatible with inference acceleration techniques like EasyCache [73], as the pre-processing stage focuses on creating a coarse latent layout, avoiding the need for highprecision computation. As shown in Tab. 5, this integration effectively reduces pre-processing overhead with minimal VRAM usage and acceptable wall-clock time. This accelerated pipeline offers a highly efficient and deterministic alternative to the exhaustive seed search typically required for accurate counting.

#### 6. Conclusion

This paper presents NUMINA, a training-free framework for count alignment in text-to-video diffusion. By leveraging instance-separable attention heads in DiTs, NUMINA identifies and corrects prompt-layout inconsistencies through explicit layout construction, conservative refinement, and layout-guided generation. NUMINA significantly boosts counting accuracy, particularly at higher counts where baselines falter, without sacrificing video quality. These results highlight the value of structural guidance as a complement to existing methods, offering a practical approach to count-accurate text-to-video generation and improving numeral alignment for broader applicability.

Limitations. While NUMINA significantly improves numerical alignment, achieving perfect accuracy across all scenarios remains challenging. Besides, generating very dense instances (e.g., tens or hundreds) remains unexplored. Enabling fully numerically precise video generation for any number is an important direction for future research.

#### References

- [1] Anthropic. Introducing claude sonnet 4.5. https:// www.anthropic.com/news/claude-sonnet-45, 2025. 6
- [2] Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. Uniedit: A unified tuningfree framework for video motion and appearance editing. In Proc. of ACM Multimedia, pages 10171–10180, 2025. 2
- [3] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia Conf., pages 1–11, 2024. 1
- [4] Lital Binyamin, Yoad Tewel, Hilit Segev, Eran Hirsch, Royi Rassin, and Gal Chechik. Make it count: Text-to-image generation with an accurate number of objects. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 13242–13251, 2025. 3
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 22563–22575,

2023. 2

- [7] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 7763–7772, 2025. 3
- [8] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 9650– 9660, 2021. 12
- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [10] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 7310–7320, 2024. 1
- [11] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Diffusion transformers for image and video generation. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 6441– 6451, 2024. 1

- [12] Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, and Yi Yang. Segment and track anything. arXiv preprint arXiv:2305.06558, 2023. 3
- [13] Dorin Comaniciu and Peter Meer. Mean shift: A robust approach toward feature space analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(5):603–619,

2002. 5

- [14] Ding Ding, Yueming Pan, Ruoyu Feng, Qi Dai, Kai Qiu, Jianmin Bao, Chong Luo, and Zhenzhong Chen. Homogen: Enhanced video inpainting via homography propagation and diffusion. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 22953–22962, 2025. 2
- [15] Martin Ester, Hans-Peter Kriegel, J¨org Sander, Xiaowei Xu, et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In Proc. ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining, pages 226– 231, 1996. 5
- [16] Zixun Fang, Kai Zhu, Zhiheng Liu, Yu Liu, Wei Zhai, Yang Cao, and Zheng-Jun Zha. Viewpoint: Panoramic video generation with pretrained diffusion models. In Proc. of Advances in Neural Information Processing Systems, 2025. 1
- [17] Bingjie Gao, Xinyu Gao, Xiaoxue Wu, Yujie Zhou, Yu Qiao, Li Niu, Xinyuan Chen, and Yaohui Wang. The devil is in the prompts: Retrieval-augmented prompt optimization for text-to-video generation. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 3173– 3183, 2025. 1
- [18] Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao, Jay Zhangjie Wu, David Junhao Zhang, Mike Zheng Shou, and Kevin Tang. Videoswap: Customized video subject swapping with interactive semantic point correspondence. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 7621–7630, 2024. 2
- [19] Yuwei Guo, Ceyuan Yang, Anyi Rao, Chenlin Meng, Omer Bar-Tal, Shuangrui Ding, Maneesh Agrawala, Dahua Lin, and Bo Dai. Keyframe-guided creative video inpainting. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 13009–13020, 2025. 2
- [20] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103,

2024. 2

- [21] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proc. Conference on Empirical Methods in Natural Language Processing, pages 7514–7528,

2021. 6

- [22] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [23] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In Proc. of Advances in Neural Information Processing Systems, pages 8633–8646, 2022. 2

- [24] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In Proc. of Intl. Conf. on Learning Representations, 2023. 2
- [25] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 6, 12
- [26] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 9307–9315, 2024. 6
- [27] Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zeroshot grounded video editing using text-to-image diffusion models. In Proc. of Intl. Conf. on Learning Representations,

2024. 2

- [28] Jaemin Kim, Bryan Sangwoo Kim, and Jong Chul Ye. Free2guide: Training-free text-to-video alignment using image lvlm. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 17920–17929, 2025. 1
- [29] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 4015–4026, 2023. 3
- [30] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [31] Yao-Chih Lee, Erika Lu, Sarah Rumbley, Michal Geyer, JiaBin Huang, Tali Dekel, and Forrester Cole. Generative omnimatte: Learning to decompose video into layers. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 12522–12532, 2025. 2
- [32] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scenarios video generation with latent diffusion model. In Proc. of European Conference on Computer Vision, pages 469–485, 2024. 2
- [33] Xiaofan Li, Yanpeng Sun, Chenming Wu, Fan Duan, YuAn Wang, Weihao Bo, Yumeng Zhang, and Dingkang Liang. Video4edit: Viewing image editing as a degenerate temporal process. arXiv preprint arXiv:2511.18131, 2025. 2
- [34] Xiaofan Li, Chenming Wu, Zhao Yang, Zhihao Xu, Yumeng Zhang, Dingkang Liang, Ji Wan, and Jun Wang. Driverse: Navigation world model for driving simulation via multimodal trajectory prompting and motion alignment. In Proc. of ACM Multimedia, pages 9753–9762, 2025. 2
- [35] Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. Diffueraser: A diffusion model for video inpainting. arXiv preprint arXiv:2501.10018, 2025. 2
- [36] Xiaofan Li, Chenming Wu, Yanpeng Sun, Jiaming Zhou, Delin Qu, Yansong Qu, Weihao Bo, Haibao Yu, and Dingkang Liang. Fvar: Visual autoregressive modeling via

- next focus prediction. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2026. 1
- [37] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 1, 3
- [38] Dingkang Liang, Xiwu Chen, Wei Xu, Yu Zhou, and Xiang Bai. Transcrowd: weakly-supervised crowd counting with transformers. Science China Information Sciences, 65(6): 160104, 2022. 3
- [39] Dingkang Liang, Wei Xu, and Xiang Bai. An end-to-end transformer model for crowd localization. In Proc. of European Conference on Computer Vision, pages 38–54, 2022.
- [40] Dingkang Liang, Wei Xu, Yingying Zhu, and Yu Zhou. Focal inverse distance transform maps for crowd localization. IEEE Transactions on Multimedia, 25:6040–6052, 2022.
- [41] Dingkang Liang, Jiahao Xie, Zhikang Zou, Xiaoqing Ye, Wei Xu, and Xiang Bai. Crowdclip: Unsupervised crowd counting via vision-language model. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 2893–2903, 2023.
- [42] Dingkang Liang, Wei Hua, Chunsheng Shi, Zhikang Zou, Xiaoqing Ye, and Xiang Bai. Sood++: Leveraging unlabeled data to boost oriented object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 3
- [43] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In Proc. of Intl. Conf. on Learning Representations,

- 2023. 3

[44] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In Proc. of European Conference on Computer Vision, pages 38–55,

- 2024. 6, 8

- [45] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 8599–8608, 2024. 2
- [46] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In Proc. of Intl. Conf. on Learning Representations, 2023. 3
- [47] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 22139–22149, 2024. 6
- [48] Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. Transactions on Machine Learning Research, 2025. 2
- [49] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid

- Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 2
- [50] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. In Proc. of Advances in Neural Information Processing Systems, pages 18481– 18505, 2024. 2
- [51] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Proc. of Intl. Conf. on Machine Learning, pages 8162–8171, 2021. 3
- [52] OpenAI. Introducing gpt-5. https://openai.com/ blog/introducing-gpt-5, 2025. 6
- [53] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 4195–4205, 2023. 1, 2, 3
- [54] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 15932–15942,

2023. 2

- [55] Dvir Samuel, Matan Levy, Nir Darshan, Gal Chechik, and Rami Ben-Ari. Omnimattezero: Fast training-free omnimatte with pre-trained video diffusion models. In SIGGRAPH Asia Conf., 2025. 2
- [56] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 8406–8416, 2025. 6, 12
- [57] Guillaume Thiry, Hao Tang, Radu Timofte, and Luc Van Gool. Towards online real-time memory-based video inpainting transformers. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 6035–6044,

2024. 2

- [58] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. In Proc. of Intl. Conf. on Learning Representations Workshop, 2019. 6
- [59] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 6, 12, 13
- [60] Shulei Wang, Wang Lin, Hai Huang, Hanting Wang, Sihang Cai, WenKang Han, Tao Jin, Jingyuan Chen, Jiacheng Sun, Jieming Zhu, et al. Towards transformer-based aligned generation with self-coherence guidance. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 18455–18464, 2025. 2
- [61] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. In Proc. of Advances in Neural Information Processing Systems, pages 7594–7611, 2023. 2
- [62] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video genera-

- tion. In Proc. of Advances in Neural Information Processing Systems, pages 34322–34348, 2024. 1
- [63] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 7623– 7633, 2023. 2
- [64] Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang, Yixiao Ge, Xiaodong Cun, David Junhao Zhang, Jia-Wei Liu, Yuchao Gu, Rui Zhao, et al. Towards a better metric for text-to-video generation. arXiv preprint arXiv:2401.07781,

2024. 6

- [65] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. In Proc. of European Conference on Computer Vision, pages 331–348, 2024. 2
- [66] Shuai Yang, Liming Jiang, Ziwei Liu, and Chen Change Loy. Vtoonify: Controllable high-resolution portrait video style transfer. ACM Transactions ON Graphics, 41(6):1–15, 2022. 2
- [67] Xi Yang, Chenhang He, Jianqi Ma, and Lei Zhang. Motionguided latent diffusion for temporally consistent real-world video super-resolution. In Proc. of European Conference on Computer Vision, pages 224–242, 2024. 1
- [68] Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang. Videograin: Modulating space-time attention for multigrained video editing. In Proc. of Intl. Conf. on Learning Representations, 2025. 2
- [69] Zhao Yang, Zezhong Qian, Xiaofan Li, Weixiang Xu, Gongpeng Zhao, Ruohong Yu, Lingsi Zhu, and Longjun Liu. Dualdiff+: Dual-branch diffusion for high-fidelity video generation with reward guidance. arXiv preprint arXiv:2503.03689, 2025. 2
- [70] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In Proc. of Intl. Conf. on Learning Representations, 2025. 12, 13
- [71] Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, and Wenhan Luo. Stylemaster: Stylize your video with artistic generation and translation. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 2630–2640, 2025. 2
- [72] Yuanyang Yin, Yaqi Zhao, Mingwu Zheng, Ke Lin, Jiarong Ou, Rui Chen, Victor Shea-Jay Huang, Jiahao Wang, Xin Tao, Pengfei Wan, et al. Towards precise scaling laws for video diffusion transformers. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 18155– 18165, 2025. 2
- [73] Xin Zhou, Dingkang Liang, Kaijin Chen, Tianrui Feng, Xiwu Chen, Hongkai Lin, Yikang Ding, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Less is enough: Training-free video diffusion acceleration via runtime-adaptive caching. arXiv preprint arXiv:2507.02860, 2025. 2, 8

## When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models

### Supplementary Material

#### S1. Additional Results

Compatibility with CogVideoX [70]. To substantiate the generalizability and robustness of our method beyond a single model architecture, we evaluate our method on CogVideoX-5B, which employs a Multi-Modal Diffusion Transformer (MMDiT). Unlike vanilla DiTs in Wan models [59], MMDiT employs a unified global attention mechanism over concatenated visual-textual tokens without a dedicated cross-attention module. To bridge this gap, we adapt our strategy in Sec. 4.1 of the manuscript by decomposing the unified attention into distinct components. The videoto-video attention is treated as self-attention, while the textto-video attention sub-matrix is extracted as cross-attention.

As shown in Tab. 6, quantitative results demonstrate a consistent and significant improvement in numerical accuracy when our method is applied to CogVideoX-5B. Specifically, CogVideoX-5B achieves only 40.2% accuracy under minimal settings, while Seed search and Prompt enhancement provide limited gains of only 2.5% and 2.3%, respectively. In contrast, NUMINA substantially elevates the performance to 44.4% using simple prompts and a single generation pass. Furthermore, our method improves overall generation quality, improving the TC and CLIP scores to 80.2% and 35.4%, respectively. This successful extension to MMDiT further confirms the general applicability of our training-free approach across different implementations of the architecture.

Integration with enhancement strategies. As shown in Tab. 1 of the manuscript, our method alone achieves substantial improvements on CountBench. We further demonstrate that NUMINA is fully compatible with prompt enhancement and seed search, which represent the most accessible techniques for boosting counting accuracy. By integrating our method with these enhancement strategies, we achieve the best performance with 54.2% counting accuracy, reported in Tab. 7. This combined approach significantly surpasses all compared methods, including our standalone NUMINA (49.7%), prompt enhancement (47.2%), and seed search (45.5%). In particular, it also enables the 1.3B model to outperform larger baseline models, including Wan2.2-5B at 47.8% and Wan2.1-14B at 53.6%. These results establish our approach as a superior alternative to existing workflows, providing a more effective solution for the challenging counting alignment in video generation.

Evaluation on VBench [25] metric. To assess the temporal stability of the generated object instances, we adopt the Subject-Consistency metric from VBench. For each in-

Table 6. Evaluation results on CogVideoX [70].

Models CountAcc (%) TC (%) CLIP Score CogVideoX-5B [70] (81 frames, 1360×768)

CogVideoX-5B 40.2 78.1 34.8

+ Seed search 42.7(+2.5) 78.3(+0.2) 34.8(-0.0) + Prompt enhancement 42.5(+2.3) 79.0(+0.9) 34.5(-0.3) + NUMINA (ours) 44.4(+4.2) 80.2(+2.1) 35.4(+0.6)

Table 7. Ablation on combined methods.

Models CountAcc (%) TC (%) CLIP Score Wan2.1-1.3B [59] (81 frames, 832×480)

Wan2.1-1.3B 42.3 81.2 33.9

+ Seed search 45.5(+3.2) 82.3(+1.1) 34.6(+0.7) + Prompt enhancement 47.2(+4.9) 82.1(+0.9) 33.7(-0.2) + NUMINA (ours) 49.7(+7.4) 83.4(+2.2) 35.6(+1.7) + Combined method (ours) 54.2(+11.9) 83.6(+2.4) 35.5(+1.6)

Table 8. VBench [25] Subject-Consistency scores.

Models Baseline + NUMINA (ours) Wan2.1-1.3B 83.1 83.6(+0.5)

- Wan2.1-14B 84.3 84.7(+0.4)
- Wan2.2-5B 83.4 83.5(+0.1) CogVideoX-5B 84.6 84.6(+0.0)

stance, we extract DINO [8] features in all frames and compute the cosine similarity with both the first frame and the preceding frame. The two similarities are averaged, and the final video-level score is obtained by averaging over all non-initial frames. We report the mean score across instances. As shown in Tab. 8, our method achieves competitive performance on this metric, indicating that the edited instances remain temporally stable and visually coherent. This result further validates the reliability of our TC metric, as both measurements capture complementary aspects of temporal coherence. In addition, our counting accuracy follows the Generative Numeracy evaluation protocol in T2V-CompBench [56], ensuring that our overall evaluation framework is both consistent and reliable.

Analysis on no-reference addition. We analyze the effectiveness of adding missing instances when no reference instances are available. This presents a particularly challenging setting where baseline models typically fail to generate the required objects. As shown in Tab. 9, the nointervention baseline achieves only 48.8% accuracy without layout refinements in such cases. To address this limitation, we compare two geometric priors for layout refinement: a circular template and a rectangular alternative of equivalent

Table 10. Ablation results for different hyperparameter values.

Table 9. Ablation on strategy for no-reference addition.

Method CountAcc (%) TC (%) Baseline 42.3 81.2

λ / CountAcc (%) τ / CountAcc (%) k / CountAcc (%)

4 / 49.3 0.1 / 48.4 0.5 / 48.2 8 / 49.7 0.2 / 49.7 0.8 / 49.7

No intervention 48.8(+6.5) 83.0(+1.8) Rectangle 49.5(+7.2) 83.3(+2.1) Circle 49.7(+7.4) 83.4(+2.2)

16 / 49.5 0.3 / 49.2 1.0 / 49.2

- Table 11. Ablation on object addition or removal.

Addition Removal CountAcc (%) TC (%) Baseline 42.3 81.2

✓ 47.7(+5.4) 83.0(+1.8) ✓ 43.8(+1.5) 82.4(+1.2) ✓ ✓ 49.7(+7.4) 83.4(+2.2)

- Table 12. VBench Aesthetic & Imaging Quality scores. Method Imaging↑ Aesthetic↑

layer

1 1 15

10 15 20 30

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

25

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

35

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Wan2.1-1.3B 71.3% 61.5% +NUMINA 70.9% 63.5%

50

timestep

Prompt: “Three men are walking in the park.”

Figure 8. PCA visualization across timesteps and layers.

i.e., object addition and removal. Tab. 11 shows that addition alone significantly boosts accuracy by 5.4%, while removal yields a smaller 1.5% gain. This suggests that the baseline model primarily struggles with object omission, making addition the more impactful correction. Furthermore, combining both operations achieves the highest accuracy, slightly exceeding the sum of individual gains, proving a synergistic effect between the two complementary guidance methods.

area. Experimental results demonstrate the effectiveness of both strategies, with the rectangular prior reaching 49.5% accuracy and the circular prior achieving 49.7%. In practice, we employ the circular prior as described in Sec. 4.2 of the manuscript. This design minimizes structural assumptions, granting T2V models the flexibility to interpret and form the most contextually plausible objects.

Analysis on layout localization. We next analyze the feasibility of layout localization based on Wan2.11.3B [59]. As visualized in Fig. 8, our analysis reveals clear instance-separable attention patterns during denoising. These discriminative layouts emerge most distinctly at middle denoising steps, with intermediate layers providing the sharpest spatial separation of object instances. We accordingly set t⋆ = 20 and ℓ⋆ = 15 to balance efficiency and accuracy. By performing layout localization at this point and early stopping, we reduce the denoising steps for pregeneration by approximately 60% without significantly sacrificing accuracy, as quantified in Fig. 7 of the manuscript. This early termination delivers significant computational savings, particularly for larger models. The same relative proportions can be directly applied to other model architectures through straightforward scaling.

Evaluation of visual quality. We evaluate visual generation quality using VBench (Aesthetic & Imaging Quality). As shown in Tab. 12, our method maintains comparable or even superior metric scores, introducing no degradation in video generation quality while significantly enhancing numerical alignment, which is further confirmed by the user study, demonstrating the quality of our approach.

User study. We conduct a blind user study involving 10 participants (balanced gender ratio) using 100 pairs of randomly sampled videos. Participants are asked to evaluate both visual quality and instruction following. The results show a 61% preference for our method versus 39% for the baseline. This clear preference confirms that our method delivers not only better objective metric performance but also a superior user experience.

Analysis on hyperparameters. We emphasize that our hyperparameters are generic and are largely set without exhaustive tuning. Selections of layer and timestep vary solely due to intrinsic model differences (e.g., the total number of inference steps) rather than specific heuristic design. We uniformly set t⋆ = 20 and ℓ⋆ = 15 in this section for a fair ablation study. As detailed in Tab. 10, our method maintains stable performance across a wide range of hyperparameter values.

#### S2. More Visualization

Additional demos. We provide more comprehensive qualitative comparisons in Fig. 10, showcasing our method’s effectiveness across different model architectures. The consolidated visualization presents successful numerical alignment cases on Wan2.1 [59] and CogVideoX [70], demonstrating consistent improvement in generating accurate object counts. These cross-architecture validations collectively confirm our method’s strong generalizability and practical utility for enhancing numerical accuracy in text-to-

Analysis on the object addition/removal. We finally analyze the effect of layout-guided generation operations,

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Prompt: “ Three parrots mimicking three whistles.”

Figure 9. A failure case of NUMINA. The parrots’ heads become decoupled from their bodies in layout construction.

video generation systems. More video demos can be found on our project page.

Failure cases. A characteristic failure mode of our method occurs when instance-separable attention heads focus excessively on the most salient parts of an object (e.g., an animal’s head) rather than its entirety, as demonstrated by the representative failure case in Fig. 9. This leads to an over-segmented layout where parts of a single instance are mistaken for multiple objects, ultimately propagating an irrecoverable error into the final video output. This limitation underscores the challenge of defining instances solely via raw attention and suggests the need for future work to incorporate more holistic perceptual grouping cues.

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Wan2.1-1.3B

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

NUMINA(ours)

“Three riders are competing in the horse race.” “Two friends and two dogs hiking up a mountain.”

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Wan2.1-1.3B

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

NUMINA(ours)

“A wolf and a fox running through the snow-covered forest.” “Three cats are surrounding two mice.”

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

Wan2.1-14B

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

NUMINA(ours)

“Six bicycles cycling through a park, with riders pedaling past trees and a serene lake.”

“Three tourists, two suitcases, and one luggage trolley at the airport, the tourists loading their bags onto the trolley as they head to their gate.”

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Wan2.2-5B

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

NUMINA(ours)

“Two men and two dogs resting near a campfire.” “A penguin waddling on an icy landscape.”

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

CogVideoX-5B

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

NUMINA(ours)

“Six people skied down from the mountain. ” “One robot and two humans assembling a machine.”

Figure 10. More representative examples where our method faithfully generates the specified number of objects.

