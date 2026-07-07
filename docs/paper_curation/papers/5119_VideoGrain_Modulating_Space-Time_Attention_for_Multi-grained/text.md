# arXiv:2502.17258v1[cs.CV]24Feb2025

## VIDEOGRAIN: MODULATING SPACE-TIME ATTENTION FOR MULTI-GRAINED VIDEO EDITING

Xiangpeng Yang 1 Linchao Zhu 2 Hehe Fan 2 Yi Yang 2 1 ReLER Lab, AAII, University of Technology Sydney 2 ReLER Lab, CCAI, Zhejiang University

Project Page: https://knightyxp.github.io/VideoGrain_project_page

Class level Instance level

Editing to same class objects Separated editing to different instances

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

6

left cat → Samoyed, right cat → Tiger， background → sunrise

man → Iron Man

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

###### left cat → Panda, right cat → Toy poodle， ground → grassy meadow, background → starry night

man → Batman, clay court → snow court, stone wall → iced wall

Adding new objects or modifying existing attributes at the part-level

Part level

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

man → Superman Superman + cap

gray, half-sleeve gray → blue half-sleeve→full suit

Figure 1: VideoGrain enables multi-grained video editing across class, instance, and part levels.

ABSTRACT

Recent advancements in diffusion models have significantly improved video generation and editing capabilities. However, multi-grained video editing, which encompasses class-level, instance-level, and part-level modifications, remains a formidable challenge. The major difficulties in multi-grained editing include semantic misalignment of text-to-region control and feature coupling within the diffusion model. To address these difficulties, we present VideoGrain, a zeroshot approach that modulates space-time (cross- and self-) attention mechanisms to achieve fine-grained control over video content. We enhance text-to-region control by amplifying each local prompt’s attention to its corresponding spatialdisentangled region while minimizing interactions with irrelevant areas in crossattention. Additionally, we improve feature separation by increasing intra-region awareness and reducing inter-region interference in self-attention. Extensive experiments demonstrate our method achieves state-of-the-art performance in realworld scenarios. Our code, data, and demos are available on the project page.

### 1 INTRODUCTION

T2V based: Pika

T2V based: DMT

T2I class-level: TokenFlow

T2I instance-level: Ground-A-Video

Instance Level Part Level

Source Video Class Level

|[Figure 25]|
|---|
|[Figure 26]|

|[Figure 27]|
|---|
|[Figure 28]|

|[Figure 29]|
|---|
|[Figure 30]|

|[Figure 31]|
|---|
|[Figure 32]|

|[Figure 33]|
|---|
|[Figure 34]|

|[Figure 35]|
|---|
|[Figure 36]|

|[Figure 37]|
|---|
|[Figure 38]|

|[Figure 39]|
|---|
|[Figure 40]|

left man→Spiderman, right man→Polar bear

Polar bear

"A Spiderman and a polar bear are jogging on grassy meadow before cherry trees"

man→Spiderman

+sunglasses

Figure 2: Definition of multi-grained video editing and comparison on instance editing

Recent advances in Text-to-Image (T2I) and Text-to-Video (T2V) diffusion models (Rombach et al., 2022; Wang et al., 2023a; Brooks et al., 2024) have enabled video manipulation through natural language prompts. In practical applications, enabling users to edit regions at various levels of granularity based on textual prompts offers greater flexibility. To investigate this, we introduce a new task called multi-grained video editing, which encompasses class-level, instance-level, and partlevel editing, as shown in Fig. 2 left. Class-level editing refers to modifying objects within the same class. Instance-level editing means editing different instances into distinct objects. Part-level going further, requires adding new objects or modifying existing attributes at part-level.

While existing methods employ various visual consistency techniques, such as optical flow (Cong et al., 2023; Yang et al., 2023), control signals (Zhang et al., 2023b), or feature correspondence (Geyer et al., 2023). These methods remain instance-agnostic, often mixing features of different instances during editing (see Fig. 2 right). Ground-A-Video (Jeong & Ye, 2023), which inherits text-to-bounding box generation priors (Li et al., 2023), should be instance-level editing but still suffer from artifacts. Similarly, recent T2V-based methods like DMT (Yatim et al., 2024) and Pika (pik), although equipped with video generation priors, struggle with multi-grained edits. We find that the core issue is that diffusion models tend to treat different instances as the same class segments, leading to strong feature coupling across instances, as illustrated in Figure 3.

To address this problem, our primary insight is to 1) enable text-to-region control and 2) keep feature separation between regions. In the typical diffusion models, the cross-attention layer serves as a key component to update textual features control over each spatial region, while the self-attention layer generates globally coherent structures by connecting each frame token across time. Therefore, we propose Spatial-Temporal Layout-Guided Attention (ST-Layout Attn), which modulates both spacetime cross- and self-attention in a unified manner to achieve the above goals.

In the cross-attention layer, the uniform application of global text prompts across all frame tokens leads to severe semantic misalignment, which reduces the precision of multi-grained text-to-region control. To address this, we modulate cross-attention to amplify each local prompt’s focus on its corresponding spatial-disentangled region while suppressing attention to irrelevant areas. In the self-attention layer, pixels from one region may attend to outside or similar regions within the same class, leading to feature coupling and texture mixing, which is an inherent limitation of diffusion models that complicates multi-grained video editing. To mitigate this, we modulate self-attention to enhance feature separation by increasing intra-region focus and reducing inter-region interactions, ensuring each query attends only to its target region.

Our key contributions can be summarized as follows:

- • To the best of our knowledge, this is the first attempt at multi-grained video editing. Our method enables both class-level, instance-level and part-level editing.
- • We propose a novel framework, dubbed VideoGrain, which modulates spatial-temporal cross- and self-attention for text-to-region control and feature separation between regions.
- • Without tuning any parameters, we achieve state-of-the-art results on existing benchmarks and real-world videos both qualitatively and quantitatively.

- 2 RELATED WORK

- 2.1 TEXT-TO-IMAGE EDITING/GENERATION

In the realm of single attribute text-to-image editing, various approaches have been explored, from manipulating attention maps in Pix2Pix-Zero (Parmar et al., 2023) and Prompt2Prompt (Hertz et al.,

- 2022) to employing masks in DiffEdit (Couairon et al., 2023) and Latent Blend (Avrahami et al., 2022; 2023) for foreground modifications while preserving the background.

For multi-grained editing, efforts like Attention and Excite (Chefer et al., 2023) and DPL (Wang et al., 2023b) focus on maximizing attention scores for each subject token and reducing attention leakage. In image generation, (Kim et al., 2023) modulates attention based on layout masks and dense captions, while (Phung et al., 2023) proposed an attention refocus loss for regularization. However, using single-frame layout masks and dense captioning alone is insufficient for video editing, as it fails to maintain the original video’s integrity and temporal consistency.

2.2 TEXT-TO-VIDEO EDITING

Video Editing based on Image Diffusion Models. Tune-A-Video (TAV) (Wu et al., 2022) is the first work to extend latent diffusion models to the spatial-temporal domain and encode the source motion implicitly by one-shot tuning but still fails to preserve local details. Fatezero (Qi et al.,

- 2023) and Pix2Video (Ceylan et al., 2023) fuse self- or cross-attention maps in the inversion process for temporal consistency. However, (Qi et al., 2023) requires extensive RAM usage and suffers from layout preservation even when equipping TAV for local object editing. (Chai et al., 2023) and (Ouyang et al., 2023), following the Neural Atlas (Kasten et al., 2021) or dynamic Nerf’s deformation field (Pumarola et al., 2021), struggle with non-grid human motion. Subsequent methods like Rerender-A-Video (Yang et al., 2023), FLATTEN (Cong et al., 2023) ControlVideo (Zhang et al., 2023b) achieve strict temporal consistency via optical-flow, depth/edge maps, but failed in multi-grained editing while preserving original layouts. Tokenflow (Geyer et al., 2023) enforces a linear mix of nearest key-frame features to ensure consistency but results in detail loss. Ground-AVideo (Jeong & Ye, 2023) leverages groundings for multi-grained editing, but it suffers from feature mixing when bounding boxes overlap.

- 3 METHOD

Video Editing based on Video Diffusion Models. Previous video editing work primarily utilized text-to-image SD model (Rombach et al., 2022). Recent advancements in video foundation models (Yu et al., 2023; Guo et al., 2023; Wang et al., 2023a; Yang et al., 2024e) have led efforts like VideoSwap (Gu et al., 2023) to employ temporal priors for customized motion transfer or motion editing (Mou et al., 2025). Yet, current video foundation models are limited to fixed views and struggle with non-grid human motions. Additionally, these editing methods require tuning parameters, which poses a challenge for real-time video editing applications. In contrast, our VideoGrain method requires no parameter tuning, enabling zero-shot, multi-grained video editing.

- 3.1 MOTIVATION

To investigate why previous methods failed in instance-level video editing (see Fig. 2), we begin with a basic analysis of the self-attention and cross-attention features within the diffusion model.

- As shown in Fig. 3 (b), we apply K-Means clustering to the per-frame self-attention features during DDIM Inversion. Although the clustering captures a clear semantic layout, it fails to distinguish between distinct instances (e.g., “left man” and “right man”). Increasing the number of clusters leads to finer segmentation at the part level but does not resolve this issue, indicating that feature homogeneity across instances limits the diffusion model’s effectiveness in multi-grained video editing.

Next, we attempt to edit the same class of two men into different instances using SDEdit (Meng et al., 2022). However, Fig. 3 (d) shows that the weights for “Iron Man” and “Spiderman” overlap on the left man, and “blossoms” weight leaks onto the right man, resulting in the failed edit in (c).

Frame 1 Frame 2 Frame 3

segment = 3 segment = 4 segment =5

|[Figure 41]|[Figure 42]|[Figure 43]|
|---|---|---|

|[Figure 44]|[Figure 45]|[Figure 46]|
|---|---|---|

(a) Source video input (b) K-Means cluster Self-Attention feature

|[Figure 47]|[Figure 48]|
|---|---|

|[Figure 49]|[Figure 50]|[Figure 51]|
|---|---|---|

(c) Instance-level failed case (d) Cross-Attention Map: "An Iron Man and a Spiderman are jogging under cherry blossoms"

- Figure 3: Analysis of why the diffusion model failed in instance-level video editing. Our goal is to edit left man into “Iron Man,” right man into “Spiderman,” and trees into “cherry blossoms.” In (b), we apply K-Means on self-attention, and in (d), we visualize the 32x32 cross-attention map.

Thus, for effective multi-grained editing, we pose the following question: Can we modulate attention to ensure that each local edit’s attention weights are accurately distributed in the intended regions?

To answer this, we propose VideoGrain with two key designs: (1) Modulate cross-attention to induce textual features to congregate in corresponding spatial-disentangled regions, thereby enabling textto-region control. (2) Modulate self-attention across the spatial-temporal axis to enhance intraregion focus and reduce inter-region interference, avoiding feature coupling within diffusion model.

- 3.2 PROBLEM FORMULATION

The purpose of this work is to perform multi-grained video editing across multiple regions based on the given prompts. This involves three hierarchical levels:

- (1) Class-level editing: Editing objects within the same class. (e.g., changing two men to “Spiderman,” where both belong to the human class, as seen in Fig. 2 second column)
- (2) Instance-level editing: Editing each individual instance to distinct object. (e.g., editing left man to “Spiderman,” right man to “Polar Bear,” as shown in Fig. 2 third column).
- (3) Part-level editing: Applying part-level edit to specific elements of individual instances. (e.g., adding “sunglasses ”when editing the right man to “Polar Bear” in Fig. 2 fourth column).

Given a source video V ∈ RN×3×H×W, where N is the number of frames, our goal is to obtain an edited video V′ based on specified edits. We aim to improve multi-grained control in video editing by conditioning on each region’s location and its text prompt. More formally, we optimize a video editing model f(τg,(τ1,m1),...,(τk,mk)), where τg is a global prompt, and (τk,mk) are the kth region’s prompt and corresponding location.

- 3.3 OVERALL FRAMEWORK

The proposed zero-shot multi-grained video editing pipeline is illustrated in Fig. 4 top. Initially, to retain high fidelity, we perform DDIM Inversion (Song et al., 2021) over the clean latent x0 to get the noisy latent xt. After the inversion process, we cluster the self-attention features to get the semantic layout as in Fig. 3 (b). Since self-attention features alone cannot distinguish between individual instances, we further employ SAM-Track (Cheng et al., 2023) to segment each instance. Finally, in the denoising process, we introduce ST-Layout Attn to modulate cross- and self-attention for text-to-region control and keep feature separation between regions, as detailed in Sec. 3.4.

Different from one global text prompt control of all frames, VideoGrain allows paired instance- or part-level prompts and their locations to be specified in the denoising process. Our method is also versatile to ControlNet condition e, which can be depth or pose maps to provide structure conditions.

|Spatial-Temporal Layout Attention Intergration|
|---|

|Conv Block Self Attention Cross Attention|
|---|

semantic void condition

DDIM denoising

[Figure 52]

[Figure 53]

Predicted noise

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

###### ...

...

...

[Figure 60]

Inverted Latents

[Figure 61]

DDIM Inversion

" Spiderman and Polar Bear are jogging on grassy meadow before cherry trees "

Noise predictor

###### Before Cross-Attention Modulation

Cross-Attention Layer Modulating Cross Attention

Spiderman

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

jogging

Polar

Bear

###### ...

and

are

Tokens of interest

[Figure 65]

|NegativePair|
|---|

[Figure 66]

[Figure 67]

[Figure 68]

NegativePair

|PositivePair|
|---|

NegativePairPositivePair

[Figure 69]

[Figure 70]

||Spiderman|
|---|
<br><br>Spiderman|
|---|

|are|
|---|

|Bear|
|---|

|jogging|
|---|

|Polar|
|---|

|and|
|---|

Spiderman (w =1 ) Bear (w = 4)

Polar ( w = 3)

Spiderman

jogging

Polar

|↑positive ↓negative|
|---|

Bear

After Cross-Attention Modulation

and

are

...

[Figure 71]

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

[Figure 75]

PositivePair

Tokens of interest

[Figure 76]

[Figure 77]

Spiderman (w = 1) Polar (w = 3) Bear (w = 4)

: original cross attention. : positive score. : negative score

Modulating Self Attention

Before Self-Attention Modulation

Self-Attention Layer

Key( )

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]| |
|---|---|
| | |

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

[Figure 91]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | |[Figure 92]| |
| | | | | |
| | | | | |

[Figure 93]

Positive Pair Positive Pair

|[Figure 94]|
|---|

[Figure 95]

. . .

|↑positive ↓negative|
|---|

[Figure 96]

After Self-Attention Modulation

[Figure 97]

Query

Negative Pair

Negative Pair

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

[Figure 101]

[Figure 102]

[Figure 103]

: original self attention. : positive score. : negative score

- Figure 4: VideoGrain pipeline. (1) we integrate ST-Layout Attn into the frozen SD for multi-grained editing, where we modulate self- and cross-attention in a unified manner. (2) In cross-attention, we view each local prompt and its location as positive pairs, while the prompt and outside-location areas are negative pairs, enabling text-to-region control. (3) In self-attention, we enhance positive awareness within intra-regions and restrict negative interactions between inter-regions across frames, making each query only attend to the target region and keep feature separation. In the bottom two figures, p denotes original attention score and w,i denotes the word and frame index.

- 3.4 SPATIAL-TEMPORAL LAYOUT-GUIDED ATTENTION

Based on the observation in Sec.3.1, cross-attention weight distribution adheres to the edit result. Meanwhile, self-attention is also crucial to generate temporal consistent video. However, the pixels in one region may attend to outside or similar regions, which poses an obstacle for multi-grained video editing. Therefore, we need to modulate both self- and cross-attention to make each pixel or local prompt only focus on the correct region.

To achieve this goal, we modulate both cross- and self-attention mechanisms via a unified increase positive and decrease negative manner. Specifically, for the ith frame of the query feature, we modulate the query-key QK⊤ condition map as follows:

QK⊤ + λMself/cross √

Aselfi /cross = softmax(

),

d

Mself/cross = Ri ⊙ Mipos − (1 − Ri) ⊙ Mineg,

(1)

where Ri ∈ R|queries|×|keys| indicates the query-key pair condition map at frame i, manipulating whether to increase or decrease the attention score for a particular pair. And λ = ξ(t) · (1 − Si) is a regularization term. We follow the conclusion from (Kim et al., 2023), the ξ(t) controls the modulation intensity across time-steps, allowing for gradual refinement of shape and appearance details. The latter is a size regulation term, making smaller region mk subjected to larger modulation, enabling dynamic attention weight adjustments to layout size variations.

Modulate Cross-Attention for Text-to-Region Control. In the cross-attention layer, the textual feature serves as key and value, and interacts with the query feature from the video latent. Since each instance’s appearance and location are closely related to the cross-attention weight distribution, we aim to encourage each instance’s textual features to congregate in the corresponding location.

- As shown in Fig. 4 mid, given the layout condition (τk,mk). For example, for τ1 = Spiderman, within the query-key cross-attention map, we can manually specify that the portion of the query

feature corresponding to m1 is positive, while all the remaining parts are designated as negative. Therefore, for each frame i, we can set the modulation value in cross attention layer as:

Mipos = max(QK⊤) − QK⊤, Mineg = QK⊤ − min(QK⊤),

(2)

Ricross[x,y] =

mi,k, if y ∈ τk 0, otherwise , (3)

where x and y are the query and key indices, and Ricross is the query-key condition map in the cross attention layer. We regularize this condition map by initially broadcasting each region’s mask mi,k to its corresponding text key embedding Kτ

, resulting in a condition map Ricross ∈ R(H×W)×L.

k

Each sub-region intensity then adjusts gradually in the generation process. We set Mipos/neg based on the gap between max/min values and the original scores, to keep modulated values within the

original range. Our modulation is applied to all frames to achieve spatial-temporal region control.

As shown in Fig. 4 (mid right), after adding positive and subtracting negative values, the original cross-attn weight of “Spiderman” (e.g., p) is amplified and focused on the left man. While the distract weight of “polar” “bear” become concentrated on the right man. These indicate our modulation redistributes each prompt’s weight align with target areas, enabling precise text-to-region control.

Modulate Self-Attention to Keep Feature Separation. To adapt the T2I model for T2V editing, we treat the full video as ”a larger picture,” replacing spatial attention with spatial-temporal selfattention while retaining the pretrained weights. This enhances cross-frame interaction and provides a broader visual context. However, naive self-attention can cause regions to attend to irrelevant or similar areas (e.g., Fig. 4 bottom, before modulation query p attend to two-man), which leads to mixed texture. To address this, we need to strengthen positive focus within the same region and restrict negative interactions between different regions.

As shown in Fig. 4 (bottom left), the maximum cross-frame diffusion feature indicates the strongest response among tokens within the same region. Note that DIFT (Tang et al., 2023) uses this to match different images, while we focus on cross-frame correspondences and intra-region attention modulation in the generation process. Nevertheless, negative inter-region correspondence is equally crucial for decoupling feature mixing. Beyond DIFT, we find that the minimum cross-frame diffusion feature similarity effectively captures the relations between tokens across different regions. Therefore, we define the spatial-temporal positive/negative values as:

Mipos = max(Qi[K1,··· ,Kn]⊤) − Qi[K1,··· ,Kn]⊤), Mineg = Qi[K1,··· ,Kn]⊤ − min(Qi[K1,··· ,Kn]⊤).

(4)

To ensure each patch attends to intra-regions feature while avoiding interaction in inter-regions feature. We define the spatial-temporal query-key condition map:

Riself[x,y] =

- 0,∀j ∈ [1 : N],if mi,k[x] ̸= mj,k[y]
- 1,otherwise . (5)

For frame indices i and j, the value is zero when tokens belong to different instances across frames. As shown in the right part of Fig. 4 bottom, after applying our self-attention modulation, the query feature from the left man’s nose (e.g., p) attends only to the left instance, avoiding distraction to the right instance. This demonstrates that our self-attention modulation breaks the diffusion model’s class-level feature correspondence, ensuring feature separation at the instance level.

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

" A cute pigin the autumn forest " " A firetruckand a school bus are driving on the road "

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

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

"A Spider Man and a Wonder Womanare playing badminton before charcoal grey wall"

"AnIron Man pushes a Stormtrooper in the soapbox on mossy stone bridge over lake in the forest "

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

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

[Figure 150]

[Figure 151]

"An Iron Man and a Spidermanare jogging under cherry blossoms"

###### "Superman spins moon under cherry blossoms "

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Part-level editing: Superman + sunglasses

Instances swap identity + bg -> asphalt road with building under sky

- Figure 5: Qualitative results. VideoGrain achieves multi-grained video editing, including classlevel, instance-level, and part-level. We refer the reader to our project page for full-video results.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETTINGS

In the experiment, we adopt the pretrained Stable Diffusion v1.5 as the base model, using 50 steps of DDIM inversion and denoising. Our VideoGrain operates in a zero-shot manner, requiring no additional parameter tuning. To enhance memory efficiency, we re-engineer slice attention within our ST Layout Attn. ST Layout Attn is applied during the first 15 denoising steps. We set ξ(t) = 0.3·t5 for self-attention and ξ(t) = t5 for cross-attention, where the timestep t ∈ [0,1] is normalized. All The experiments are conducted on an NVIDIA A40 GPU. We evaluate our VideoGrain using a dataset of 76 video-text pairs, including videos from DAVIS (Perazzi et al., 2016), TGVE1, and the Internet2 , with 16-32 frames per video. Four automatic metrics are employed for evaluation: CLIPT, CLIP-F, Warp-Err, and Q-edit, following (Wu et al., 2022; Cong et al., 2023). All metrics are scaled by 100 for clarity. For baselines, we compare against T2I-based methods, including FateZero (Qi et al., 2023), ControlVideo (Zhang et al., 2023b), TokenFlow (Geyer et al., 2023), GroundVideo (Jeong & Ye, 2023) and T2V-based DMT (Yatim et al., 2024). To ensure temporal consistency, we employ FLATTEN (Cong et al., 2023) and PnP (Tumanyan et al., 2023). For fairness, all T2I baselines are equipped with the same ControlNet conditions.

- 1https://sites.google.com/view/loveucvpr23/track4
- 2https://www.istockphoto.com/ and https://www.pexels.com/

Animal Instances Part level

###### Human Instances

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Thor in sunglasses, punching red boxing gloves in starry night sky

A Panda and a toy poodle are playing toys in starry night on grassy meadow

An Iron Man and a monkey are riding bikes on the snowy ground under cherry blossoms

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Groud-A-VideoVideoGrainTokenFlowControlVideoFateZero

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

DMT

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

- Figure 6: Qualitative comparisons. We refer the reader to our project page for detailed assessment.

- 4.2 RESULTS

We evaluate VideoGrain on videos covering class-level, instance-level, and part-level edits. Our method demonstrates versatility in handling animals, such as transforming a “wolf” into a “pig” (Fig. 5, top left). For instance-level editing, we can modify vehicles separately (e.g., transforming an “SUV” into a “firetruck” and a “van” into a “school bus”) in Fig. 5, top right. VideoGrain excels at editing multiple instances in complex, occluded scenes, like “Spider-Man and Wonder Woman playing badminton” (Fig. 5, middle left). Previous methods often struggle with such nonrigid motion. In addition, our method is capable of multi-region editing, where both foreground and background are edited, as shown in the soap-box scene, where the background changes to “a mossy stone bridge over a lake in the forest” (Fig. 5, middle right). Thanks to precise attention weight distribution, we can swap identities seamlessly, such as in the jogging scene, where “Iron Man” and “Spider-Man” swap identities (Fig. 5, bottom left). For part-level edits, VideoGrain excels in adjusting a character to wear a Superman suit while keeping sunglasses intact (Fig. 5, bottom right). Overall, for multi-grained editing, our VideoGrain demonstrates outstanding performance.

- 4.3 QUALITATIVE AND QUANTITATIVE COMPARISONS

Qualitative Comparison. Figure 6 shows a comparison between VideoGrain and baseline methods, including T2I-based and T2V-based approaches, for instance-level and part-level editing. For fairness, all T2I-based methods use ControlNet conditioning. (1) Animal instances: In the left column, T2I-based methods like FateZero, ControlVideo, and TokenFlow edit both cats into pandas due to same-class feature coupling in diffusion models, failing to perform separate edits. DMT, even with video generation priors, still blends the panda and toy poodle features. In contrast, VideoGrain successfully edits one into a panda and the other into a toy poodle. (2) Human instances: In the middle column, baselines struggle with same-class feature coupling, partially editing both men into

##### Automatic Metric Human Evaluation

Method CLIP-F ↑ CLIP-T ↑ Warp-Err ↓ Q-edit ↑ Edit-Acc ↑ Temp-Con ↑ Overall ↑ FateZero 95.75 33.78 3.08 10.96 59.8 78.6 59.6

ControlVideo 97.71 34.41 4.73 7.27 53.2 50.0 43.6

TokenFlow 96.48 34.59 2.82 12.28 45.4 50.4 39.8 Ground-A-Video 95.17 35.09 4.43 7.92 69.0 72.0 63.2

DMT 96.34 34.09 2.05 16.63 58.7 79.4 64.5 VideoGrain(ours) 98.63 36.56 1.42 25.75 88.4 85.0 83.0

Table 1: Quantitative comparison of automatic metrics and human evaluation. The best results are bolded.

Iron Man. DMT and Ground-A-Video also fail to follow user intent, incorrectly editing the left and right instances. VideoGrain, however, correctly transforms the right man into a monkey, breaking the human-class limitation. (3) Part-level editing: In the third column, VideoGrain manages partlevel edits, such as sunglasses and boxing gloves. ControlVideo edits the gloves but struggles with sunglasses and motion consistency. TokenFlow and DMT edit the sunglasses but fail to modify the gloves or background. In comparison, VideoGrain achieves both instance-level and part-level edits, significantly outperforming previous methods.

Quantitative Comparison. We compare the performance of different methods using both automatic metrics and human evaluation. CLIP-T calculates the average cosine similarity between the input prompt and all video frames, while CLIP-F measures the average cosine similarity between consecutive frames. Additionally, Warp-Err captures pixel-level differences by warping the edited video frames according to the optical flow of the source video, extracted using RAFT-Large (Teed & Deng, 2020). To provide a more comprehensive measure of video editing quality, we follow (Cong et al., 2023) and use Q-edit, defined as CLIP-T/Warp-Err. For clarity, we scale all automatic metrics by 100. In terms of human evaluation, we assess three key aspects: Edit-Accuracy (whether each local edit is accurately applied), Temporal Consistency (evaluated by participants for coherence between video frames), and Overall Edit Quality. We invited 20 participants to rate 76 video-text pairs on a scale of 20 to 100 across these three criteria, following (Jeong & Ye, 2023). As demonstrated in Table 1, VideoGrain consistently outperforms both T2I- and T2V-based methods. This is primarily due to ST-Layout Attn’s precise text-to-region control and maintaining feature separation between regions. As a result, our method achieves significantly higher CLIP-T and Edit-Accuracy scores compared to other baselines. The improved Warp-Err and Temporal Consistency metrics further indicate that VideoGrain delivers temporally coherent video edits.

Efficiency Comparison. To evaluate efficiency, we compared baselines with VideoGrain on a single A6000 GPU for editing 16 video frames. The metrics include editing time (time taken to perform one edit) and both GPU and CPU memory usage. From Tab. 2, it is clear our method achieves the fastest editing time with the lowest memory usage, indicating its computational efficiency.

Time(min) ↓ Memory (GB) ↓ RAM (GB) ↓ FateZero 8.68 27.35 144.22 ControlVideo 4.41 16.15 7.03 TokenFlow 4.56 17.84 5.35 Ground-A-Video 5.81 17.31 9.96 DMT 5.79 27.88 8.12 VideoGrain 3.83 15.94 4.42

###### ❌ W/O ST-Layout Attn Attn Weight Before Attn Weight After ✅ ST-Layout Attn

|[Figure 223]|[Figure 224]|
|---|---|

|[Figure 225]|[Figure 226]|
|---|---|

Table 2: Efficiency comparison.

Figure 7: Attention weight distribution.

- 4.4 ABLATION STUDY

To assess the contributions of different components in our proposed ST-Layout Attn, we first evaluate whether our attention can achieve attention weight distribution, then decouple the self-attention modulation and cross-attention modulation to evaluate their individual effectiveness.

Attention Weight Distribution. We evaluate the impact of ST-Layout Attn on attention weight distribution. As shown in Fig. 7, the target prompt is “An Iron Man is playing tennis on a snow court.” We visualize the cross-attention map for “man” to assess weight distribution. Without STLayout Attn, feature mixing occurs, with “snow” weight spilling onto “Iron Man.” With ST-Layout Attn, the man’s weight is correctly distributed. This is because we enhance positive pair scores and suppress negative pairs in both cross- and self-attention. This enables precise, separate edits for “Iron Man” and “snow.” Additional visualizations are in the Appendix.

###### left man → Iron Man, right man → Spider man, ground, trees → frosty yellow leaves

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

Source video

(—) Cross-Attention Modulation (—) Self-Attention Modulation

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

(+) Cross-Attention Modulation (+) Self-Attention Modulation

(+) Cross-Attention Modulation (—) Self-Attention Modulation

Figure 8: Ablation of cross- and self-modulation in ST-Layout Attn.

Method CLIP-F ↑ CLIP-T ↑ Warp-Err ↓ Qedit ↑ Baseline 95.21 33.59 3.86 8.70

Baseline + Cross Modulation 96.28 36.09 2.53 14.26 Baseline + Cross Modulation + Self Modulation 98.63 36.56 1.42 25.75

Table 3: Quantitative ablation of cross- and self-modulation in ST-Layout Attn.

Cross-Attention Modulation. In Fig. 8 and Tab. 3, we illustrate video editing results under different set up: (1) Baseline (2) Baseline + Cross-Attn Modulation (3) Baseline + Cross-Attn Modulation + Self-Attn Modulation. As shown in Fig. 8 top right, direct editing fails to discriminate between the left and right instances, leading to incorrect (left) or no edits(right). However, when equipped with cross-attention modulation, we achieve accurate text-to-region control, thereby editing left man to “Iron Man” and right man to “Spiderman” separately. The quantitative results in Tab. 3 indicate that with cross-attention modulation (second row), CLIP-T increases by 7.4%, and Q-edit increases by 63.9%. This demonstrates the effectiveness of our cross-attention modulation.

Self-Attention Modulation. However, modulating only cross-attention still leads to structure distortions, such as the spider web appearing on the left man. This is caused by the coupling of same class-level features (e.g., human). When using our self-attention modulation, the feature mixing is significantly reduced, and the left man retains unique object features. This is achieved by decreasing the negative pair scores between different instances, while increasing positive scores within the same instance. As a result, more part-level details, such as the distinctive blue sides, are generated in the optimized areas. The quantitative decrease in Warp-Err by 43.9% and increase in Q-edit by 80.6% in Tab. 3 further prove the effectiveness of self-attention modulation.

- 5 CONCLUSION

In this paper, we aim to solve the problem of multi-grained video editing, which includes both class-level, instance-level and part-level video editing. To the best of our knowledge, this is the first attempt at this task. In this task, we find that the key problem is that the diffusion model views different instances as same-class features and direct global editing will mix different local regions. To wrestle with these problems, we propose VideoGrain to modulate spatial-temporal crossand self-attention for text-to-region control while keeping feature separation between regions. In cross-attention, we enhance each local prompt’s focus on its corresponding spatial-disentangled region while suppressing attention to irrelevant areas, thereby enabling text-to-region control. In self-attention, we increase intra-region awareness and reduce inter-region interactions to keep feature separation between regions. Extensive experiments demonstrate that our VideoGrain surpasses previous video editing methods on both class-level, instance-level, and part-level video editing.

- 6 ETHICS STATEMENT

This project aims to solve multi-grained video editing. However, the potential misuse of this technology, such as the creation of deceptive videos by altering identities, poses a risk. Strategies like incorporating invisible watermarking could be explored to ensure videos are not used maliciously.

REFERENCES

https://www.pika.art/. URL https://www.pika.art/. Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of

natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18208–18218, 2022.

Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM Transactions on Graphics (TOG), 42(4):1–11, 2023.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 23206–23217, 2023.

Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 23040–23050, 2023.

Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023.

Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, and Yi Yang. Segment and track anything. arXiv preprint arXiv:2305.06558, 2023.

Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flow-guided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023.

Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusionbased semantic image editing with mask guidance. In ICLR 2023 (Eleventh International Conference on Learning Representations), 2023.

Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023.

Yuchao Gu, Yipin Zhou, and Mike Zheng et al. Videoswap: Customized video subject swapping with interactive semantic point correspondence. arXiv preprint arXiv:2312.02087, 2023.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. 2022.

Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zero-shot grounded video editing using textto-image diffusion models. arXiv preprint arXiv:2310.01107, 2023.

Heng Jia, Yunqiu Xu, Linchao Zhu, Guang Chen, Yufei Wang, and Yi Yang. Mos2: Mixture of scale and shift experts for text-only video captioning. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 8498–8507, 2024.

Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG), 40(6):1–12, 2021.

Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In ICCV, 2023.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. CVPR, 2023.

Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8599–8608, 2024.

Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Yue Ma, Xiaodong Cun, Yingqing He, Chenyang Qi, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Magicstick: Controllable video editing via control handle transformations. arXiv preprint arXiv:2312.03047, 2023.

Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 4117–4125, 2024a.

Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. arXiv preprint arXiv:2406.01900, 2024b.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. Advances in Neural Information Processing Systems, 37:18481–18505, 2025.

Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926, 2023.

Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pp. 1–11, 2023.

Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, and Van Gool et al. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 724–732, 2016.

Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. arXiv preprint arXiv:2306.05427, 2023.

Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10318–10327, 2021.

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=ypOiXjdfnU.

Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pp. 402–419. Springer, 2020.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1921–1930, 2023.

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023a.

Kai Wang, Fei Yang, Shiqi Yang, Muhammad Atif Butt, and Joost van de Weijer. Dynamic prompt learning: Addressing cross-attention leakage for text-based image editing. arXiv preprint arXiv:2309.15664, 2023b.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022.

Shuai Yang, Yifan Zhou, Ziwei Liu, , and Chen Change Loy. Rerender a video: Zero-shot textguided video-to-video translation. In ACM SIGGRAPH Asia Conference Proceedings, 2023.

Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang. Eva: Zero-shot accurate attributes and multi-object video editing. arXiv preprint arXiv:2403.16111, 2024a.

Xiangpeng Yang, Linchao Zhu, Xiaohan Wang, and Yi Yang. Dgl: Dynamic global-local prompt tuning for text-video retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 6540–6548, 2024b.

Yiyuan Yang, Guodong Long, Michael Blumenstein, Xiubo Geng, Chongyang Tao, Tao Shen, and Daxin Jiang. Pre-training cross-modal retrieval by expansive lexicon-patch alignment. In LRECCOLING 2024, pp. 12977–12987, 2024c.

Yiyuan Yang, Guodong Long, Tao Shen, Jing Jiang, and Michael Blumenstein. Dual-personalizing adapter for federated foundation models. arXiv preprint arXiv:2403.19211, 2024d.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024e.

Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8466–8476, 2024.

Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10459–10469, 2023.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3836–3847, October 2023a.

Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023b.

Different from multi-modal learning (Yang et al., 2024b;c;d; Jia et al., 2024), controllable video generation (Ma et al., 2024b;a; Lu et al., 2024) or video editing (Yang et al., 2024a; Ma et al., 2023) requires explicit control signals. Multi-grained editing further relies on additional layout conditions to edit in the class, instance, or part level. Therefore, in the appendix, we first evaluate the SAMTrack masks’ impact in Section A, then validate whether our method can work without SAM-Track masks in Section B. Continually, we show that our method can solely edit specific subjects in Section C and part-level modification example in Section D. We also evaluate our ST-Layout Attn’s temporal focus in Section E and ControlNet’s effect in Section F.

- A EVALUATE SAM-TRACK MASKS’ IMPACT

"Spiderman and Polar Bearare jogging under cherry blossoms"

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

SAM-Track Instance masks

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

VideoP2P Joint Edit

"spiderman" weight "bear" weight "cherry" weight

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Ours

"spiderman" weight "bear" weight "cherry" weight

(1) VideoP2P Joint Edit with SAM-Track masks

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

VideoP2P 1st Edit

- 1st edit input
- 2nd edit input

- 1st: red man → Spider Man
- 2nd: grey man → Polar Bear
- 3rd: green trees → cherry blossoms

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

VideoP2P 2nd Edit

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

VideoP2P 3rd Edit

2nd edit input

(2) VideoP2P Sequential Edit based on Separate SAM-Track masks

- Figure 9: VideoP2P joint and sequential edit with SAM-Track masks

To evaluate the impact of using SAM-Track (Cheng et al., 2023) for instance segmentation, we compare our VideoGrain against VideoP2P (Liu et al., 2024), which is equipped with SAM-Track instance masks. The instance masks replace cross-attention masks during editing. A 16-frame oneshot tuning is performed, and ControlNet conditioning Zhang et al. (2023a) is added for fairness.

Two experiments are tested: (1) jointly editing multiple areas in a single denoising process and (2) sequentially editing three areas by inputting separate masks.

Results show that joint editing (Fig. 9(1)) modifies only left man into ”Spiderman,” leaving other areas unchanged due to inaccurate cross-attn weight distribution. Sequential editing (Fig. 9(2)) succeeds initially but fails later due to error accumulation in denoising, resulting in blurred details.

"Spiderman and Polar Bearare jogging under cherry blossoms"

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Ground-A-Video Input Ground-A-Video Result

- Figure 10: Ground-A-Video joint edit with instance information

Additionally, as shown in figure above 10, also in Figs 2 and 6, Ground-A-Video (Jeong & Ye, 2023) struggles with multi-grained video editing tasks, even with instance-level grounding information (e.g., text-to-bounding box), which is comparable to SAM-Track’s masks.

These comparisons indicate that while SAM-Track provides layout guidance, it does not guarantee successful edits. In contrast, our method enables zero-shot multi-grained editing, which was not achievable by any previous methods, even when providing existing SOTA with SAM-Track masks.

- B VIDEOGRAIN CAN WORK WITHOUT SAM-TRACK MASKS

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

(1) Input Video

(3) DDIM Inversion Cluster Masks (4) Our Results

(2) Ground-A-Video Results

"Batmanis playing tennis on snowy court before an iced wall"

[Figure 290]

[Figure 291]

[Figure 292]

Figure 11: Our method without additional SAM-Track masks

Our method is not strictly dependent on SAM-Track masks. As shown in Fig.11(3), we can cluster DDIM inversion self-attention features to get inaccurate coarse layouts. Our method still achieves high-quality multi-area editing results (4). In contrast, even with precise groundings (converted from SAM-Track masks in (1)), Ground-A-Video fails to edit all three regions. These comparisons indicate that our method does not rely on SAM-Track segmentation. Instead, it works effectively only using the self-attention feature inside the diffusion model, even without accurate layout guidance.

- C SOLELY EDIT ON SPECIFIC SUBJECTS, WITHOUT BACKGROUND CHANGED

Our method is designed for multi-area editing and can naturally perform background-preserved subject editing, as it treats multi-area editing as selecting regions restricted to the foreground. As

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

(2) Left man -> Iron Man

(1) Input Video

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

(3) Right man -> Spiderman (4) Joint Edit

Figure 12: Soely edit on specific subjects, without background changed

shown in Fig 12, our method can separately edit the “left man” and “right man” or jointly edit both subjects while keeping the background unchanged.

- D PART-LEVEL MODIFICATION EXAMPLES

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

Color change: Gray shirt -> Blue shirt

Color & shape change: Gray shirt -> Black suit

Head color change: Black -> Ginger

Body color change: Black -> Ginger

Figure 13: Part-level modifications on humans and animals

Our part-level editing supports not only adding objects but also part-level attribute modifications. In the human case (Fig. 13 left), our method changes the color of a gray shirt to blue (second row) and edits a half-sleeve shirt into a black suit (third row), showcasing part-level attribute and structure editing. Similarly, in the animal case, our method can change a cat’s head or body color from black to ginger while preserving the belt’s color, demonstrating precise part-level modifications.

- E TEMPORAL FOCUS OF ST-LAYOUT ATTN

Our ST-Layout Attn is designed as a full-frame approach to ensure inter-frame consistency. As shown in Fig. 14, per-frame ST-Layout Attn causes feature coupling on Iron Man, while the sparsecausal method results in flickering and misses Spider Man’s blue details due to their limited receptive fields for positive/negative value selection across different layouts. In contrast, our ST-Layout Attn

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

(1) Input Video

(2) Per-frame

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

(3) Sparse-Causal (first frame + previous frame) (4) Ours

Figure 14: Temporal Focus of ST-Layout Attn

effectively preserves texture details and prevents flickering, achieving temporal consistent and layout unified multi-grained video editing.

- F CONTROLNET ABLATION

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

(1) Input Video (2) Without ControlNet (3) With ControlNet

Figure 15: ControlNet ablation

Our method utilizes ControlNet depth/pose conditioning in certain complex motion cases to provide necessary structural guidance. As shown in Fig. 15, even without ControlNet, our method can still achieve simultaneous multi-region editing. However, in such cases, there may be some structural inconsistencies between the edit object and source object due to the lack of explicit structure guidance.

- G MORE GENERAL OBJECTS AND SHAPE EDITING
- H MORE VISUALIZATION
- I LATENT BLEND

To preserve areas not intended for editing (i.e., τ3 in ∆τ = {τ1→τ1′,τ2→τ2′,τ3→τ3,···}), we employ Latent Blend (Avrahami et al., 2022; 2023), which leverages masks to direct the model focus on areas requiring editing while keeping the background region identical to the source video.

For each frame i in the video, we first merge each attribute mask to form the global foreground mask Mi by applying the logical OR operation across all layouts masks mi,k = [mi,1,mi,2,··· ,mi,k] :

Mi = mi,1 ∨ mi,2 ∨ ··· ∨ mi,k. (6)

We aggregate the masks Mi from all frames to obtain a combined mask M, and then blend the latent states zt at each timestep t during the denoising process as follows:

zt = (1 − M) · z˜t + M · zt, (7)

where z˜t indicates the latent feature in the DDIM inversion process and zt is corresponding latent feature during the DDIM denoising process.

Edit prompt: A husky and a corgi and a zebra in autumn view lawn

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

Source Video

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

Edit prompt: A teddy bear and a Golden Retriever are on the grass

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

Source Video

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

Edit prompt: A red porsche car driving before the autumn view lawn

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Source Video

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Edit prompt: A red porsche car driving in the autumn view lawn forest

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Source Video

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

Figure 16: More general objects instance editing (animals) and shape editing (cars) results.

The key behind employing Latent Blend for preserving the background is that, given a desired area mask, the less noisy foreground latent can be guided by the target text prompt ∆τ. Meanwhile, the latent features outside the mask (the background) can be preserved. This blending ensures that, even if the latent feature within the edit area is modified, the background features stay consistent.

- J EXPERIMENTAL DETAILS

For FateZero3 (Qi et al., 2023), we employ prompt-to-prompt(Hertz et al., 2022) replace editing. To enhance the identity binding of the edited object, we set the self/cross replacement steps at 0.3 and

3https://github.com/ChenyangQiQi/FateZero

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

SourceVideoAttnWeightAfterST-LayoutAttnST-LayoutAttnAttnWeightBefore

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

W/O

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Figure 17: More frames ablation of ST-Layout Attn’s effects on attention weight distribution.

the blending threshold at 0.7. In TokenFlow4 (Geyer et al., 2023), we utilize SD editing and default to 4 keyframes for 16-frame videos. For other comparative methods like ControlVideo5 (Zhang et al., 2023b) and Ground-A-Video6 (Jeong & Ye, 2023) and DMT7 (Yatim et al., 2024), we adhere to their default hyperparameter settings. To ensure fairness across all T2I-based methods compared, we re-implement ControlNet (Zhang et al., 2023a) on their codebases.

- K LIMITATIONS.

First, although our method can achieve multi-grained editing of video, the generation quality is still limited by the base model since we are a training-free method. In scenarios where the generation prior to SD is not ideal, artifacts may occur in the editing results. Second, since our method is based on a T2I model, it struggles with large shape deformations and significant appearance changes. This limitation is inherent in zero-shot methods. A potential future direction is to incorporate motion priors from T2V generation models (Yang et al., 2024e) to handle such challenges.

- 4https://github.com/omerbt/TokenFlow
- 5https://github.com/YBYBZhang/ControlVideo
- 6https://github.com/Ground-A-Video/Ground-A-Video
- 7https://github.com/diffusion-motion-transfer/diffusion-motion-transfer

