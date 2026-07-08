### EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing

Yang Fu Yike Zheng Ziyun Dai Henghui Ding

Institute of Big Data, College of Computer Science and Artificial Intelligence, Fudan University, China

https://henghuiding.com/EffectErase/

Input video / mask

Object Removal

## arXiv:2603.19224v1[cs.CV]19Mar2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Occlusion

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

12 36

48 12 36 48

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

LightingShadow

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

0 48

72 0 48 72

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

[Figure 42]

0 12

36

0 12 36

Reflection

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

0 24

48

0 24 48

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Deformation

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

0 36 72

0 36

72

Figure 1. EffectErase effectively removes target objects together with various object-induced effects in videos, such as occlusion, shadow, lighting, reflection, and deformation.

#### Abstract

from captured and synthetic sources, covers five effects types, and spans a wide range of object categories as well as complex, dynamic multi-object scenes. Building on VOR, we propose EffectErase, an effect-aware video object removal method that treats video object insertion as the inverse auxiliary task within a reciprocal learning scheme. The model includes task-aware region guidance that focuses learning on affected areas and enables flexible task switching. Then, an insertion-removal consistency objective that encourages complementary behaviors and shared localization of effect regions and structural cues. Trained on VOR, EffectErase achieves superior performance in extensive experiments, delivering high-quality video object effect erasing across diverse scenarios.

Video object removal aims to eliminate dynamic target objects and their visual effects, such as deformation, shadows, and reflections, while restoring seamless backgrounds. Recent diffusion-based video inpainting and object removal methods can remove the objects but often struggle to erase these effects and to synthesize coherent backgrounds. Beyond method limitations, progress is further hampered by the lack of a comprehensive dataset that systematically captures common object effects across varied environments for training and evaluation. To address this, we introduce VOR (Video Object Removal), a large-scale dataset that provides diverse paired videos, each consisting of one video where the target object is present with its effects and a counterpart where the object and effects are absent, with corresponding object masks. VOR contains 60K high-quality video pairs

#### 1. Introduction

Video object removal has emerged as a key technique that enables users to erase unwanted dynamic content from

Corresponding author (henghui.ding@gmail.com).

[Figure 61]

[Figure 62]

[Figure 63]

Input

[Figure 64]

[Figure 65]

[Figure 66]

24 48 72

[Figure 67]

[Figure 68]

[Figure 69]

ProPainterROSEOurs

| |
|---|

| |
|---|

| |
|---|

[Figure 70]

[Figure 71]

[Figure 72]

24

72

48

[Figure 73]

[Figure 74]

[Figure 75]

| |
|---|

| |
|---|

| |
|---|

[Figure 76]

[Figure 77]

[Figure 78]

24

72

48

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

24 48 72

Figure 2. Limitations of existing video object removal methods. While existing methods [26, 47] can remove the main body within the input mask region, they often struggle to discover and remove the side effects (e.g., reflections) caused by the target object.

videos while preserving realistic visual quality. It is widely used in film post-production and video editing. Recent advances in generative models [4, 19, 35, 42] have demonstrated remarkable progress in video generation and editing quality. Leveraging the capabilities of large generative models, recent video object removal methods [3, 17, 21, 26, 48] have shown promising performance across diverse scenarios. However, as shown in Fig. 2, these methods still struggle to achieve high-fidelity results when removing objects with complex visual effects such as reflections.

This limitation can be attributed to the heavy reliance on the input mask for guidance in most video object removal methods [3, 21, 23, 44, 47], which often leads to overlooking the side effects that objects introduce into the scene. To mitigate this issue, some methods, such as Minmax-Remover [48], implicitly trains the model to discover these effects, while ROSE [26] explicitly predicts a difference mask for side effects and uses it as additional guidance. However, they still lack explicit modeling of spatiotemporal correlations between objects and their effects, limiting their robustness in complex real-world scenes and preventing stable, precise localization of effect regions.

Beyond these methodological limitations, progress in this field is also limited by the lack of a large-scale and publicly available dataset that captures common object effects across various scenes. Recently, several image-based object removal datasets [22, 31, 46] have been introduced to address the visual side effects caused by object, but they remain restricted to image-level, preventing video-based models from learning the temporal consistency required for handling moving objects. Constructing large-scale and diverse video datasets is more challenging, as the paired videos must maintain spatially consistent backgrounds and temporally coherent motion across frames. SVOR [6] syn-

[Figure 85]

[Figure 86]

Shared the same affected area

[Figure 87]

Remove

Insert

[Figure 88]

[Figure 89]

Inverse

Figure 3. Removal–Insertion. Video object removal and insertion are inverse tasks that operate on the same affected regions.

thesizes video pairs by overlaying object masks from foreground videos in YouTube-VOS [40] onto background videos, but does not account for the visual side effects. ROSE [26] employs a 3D rendering engine to generate wellaligned synthetic video pairs, but it neglects object motion and relies solely on camera movement.

New Dataset and Benchmark. To support research on effect-aware Video Object Removal in real-world scenarios, we construct VOR, a large-scale hybrid dataset that combines camera-captured and 3D-synthesized videos featuring diverse foreground objects, background scenes, and object effects. For the camera captured data, we use multiple tripod-mounted cameras to record paired videos across 293 scenes, broadly covering typical real-world use cases of video object removal. For the synthesized data, we construct over 150 diverse 3D scenes containing multiple dynamic objects, rendered by a 3D graphics engine. To approximate real-world scenarios, we manually design realistic camera and object trajectories. By combining the realism of camera-captured data with the diversity of synthesized content, VOR provides a high-quality, large-scale dataset comprising 60K paired videos. For a comprehensive evaluation of video object removal methods, we further introduce two benchmarks, VOR-Eval, a curated set with ground truth, and VOR-Wild, an in-the-wild set without ground truth covering a wide range of real-world videos.

EffectErase: Joint Removal–Insertion. Motivated by the complementary relationship of video object removal and insertion, which operate on the same affected regions as shown in Fig. 3, we propose EffectErase, an effectaware dual learning framework that jointly learns video object removal and insertion, treating insertion as an inverse auxiliary task to enhance removal quality. EffectErase incorporates a Task-Aware Region Guidance (TARG) module and an Effect Consistency (EC) loss. The TARG module builds spatiotemporal correlations between the target object and its side effects through a cross-attention mechanism, guiding the model to accurately identify the affected regions. In addition, a task token in this module enables flexible switching between the removal and insertion tasks. EC encourages the two inverse tasks to share consistent effect regions and structural feature representations, enforc-

Table 1. Comparison of video object removal datasets. Our VOR dataset exceeds prior datasets in scale and diversity, offering broader object coverage and richer camera, object, and background dynamics. Further comparisons with image-level datasets are in supplementary.

|Dataset<br><br>|Source<br><br>Dynamic Camera<br><br>Dynamic Object<br><br>Dynamic Background<br><br>Scene Types<br><br>Object Classes<br><br>Image Pairs<br><br>Video Pairs<br><br>Average Duration (s)<br><br>Total Hours|
|---|---|
|RORD [31] Video4Removal [38] ROSE [26] VOR (Ours)<br><br>|Real ✗ ✓ ✗ 24 76 516.7K 3,106 – 5.98 Real ✗ ✓ ✗ 6 – 134.3K – – 1.55<br><br>Synth. ✓ ✗ ✗ 25 102 1,501.0K 16,678 6.00 27.79 Real + Synth. ✓ ✓ ✓ 67 366 12,556.8K 60,000 8.72 145.33<br><br>|

ing cross-task consistency and strengthening effect-aware learning. Together, these components allow EffectErase to accurately localize and erase visual side effects across diverse and complex video scenes.

Our work advances video object removal in three key aspects: (i) We introduce VOR, a high-quality, large-scale hybrid dataset featuring diverse dynamic objects and complex multi-object scenarios across both camera-captured and synthesized environments. (ii) We propose EffectErase, a joint learning framework that integrates a Task-Aware Region Guidance module and an Effect Consistency loss to accurately identify and remove objects together with their visual effects. (iii) We establish two benchmarks, VOREval and VOR-Wild, providing a solid foundation for future research. The proposed method EffectErase achieves new state-of-the-art performance, surpassing existing methods in both quantitative metrics and visual quality.

#### 2. Related Work

Video Inpainting aims to reconstruct missing regions specified by a sequence of masks. Early methods [5, 36] use convolutional networks for spatiotemporal modeling but struggle with long-range propagation. Subsequent works [44, 47] exploit optical flow for additional motion cues. For example, ProPainter [47] uses recurrent flow completion to improve controllability and temporal consistency. To further enhance controllability, recent studies explore textguided video inpainting by leveraging the priors of video diffusion models. COCOCO [49], for example, introduces motion capture to stabilize results. Building on architectural advances, FloED [11] combines motion guidance with a multi-scale flow adapter to improve temporal consistency for removal and background restoration, while VideoPainter [3] employs a lightweight context encoder to enhance background integration, foreground synthesis, and user control. More recently, the unified video-synthesis baseline VACE [17] introduces a context adapter with formalized temporal and spatial representations to support multiple tasks. Despite these advances, existing inpainting models often overlook object effects, resulting in incomplete or visually inconsistent object removal.

Object Removal is a specialized form of inpainting that requires precise modeling of object-induced visual effects to achieve realistic results. Early works primarily

focus on image-level effects to ensure completeness and realism. ObjectDrop [39] captures real scenes before and after removing a single object, but with limited scale. SmartEraser [16] and Erase Diffusion [24] rely on synthetic datasets generated with segmentation [7, 8] or matting, fail to reproduce realistic side effects such as shadows and reflections. To improve realism, LayerDecomp [41] and OmniPaint [43] construct costly camera-captured datasets. OmniPaint auto-labels unlabeled images with a model trained on limited real data, whereas RORem [20] employs human annotators for refinement. RORD [31] and OmniEraser [38] mine static-camera videos to pair frames with and without the target, preserving natural effects, but remain limited to image-level removal and struggle in dynamic scenes.

Video Object Removal is more challenging, further requiring temporal consistency across frames beyond spatial fidelity. Minmax-Remover [48] simplifies a pretrained video generator by discarding text inputs and cross-attention layers while distilling stage-1 outputs using a tailored minimax optimization objective. However, this method only implicitly models video object effects and lacks access to a large and high-quality dataset. ROSE [26] introduces a synthesized dataset comprising multiple environments and approximately 27.8 hours of randomly captured video, along with a side-effect mask predictor. However, its limited scale, omission of key effects such as deformation and dynamic object motion, and synthetic composition restrict generalization to real-world scenarios.

#### 3. Methodology

##### 3.1. VOR Dataset

Overview. As shown in Fig. 4, VOR is a hybrid dataset with two components: (1) camera-captured videos emphasizing physical realism and real-world distributions, and (2) synthesized videos rendered with a 3D graphics engine to model dynamic cameras and multi-object interactions.

Representative Object-Induced Effects. To better characterize object-induced effects under diverse conditions, as shown in Fig. 5, we group them into five representative types: (1) Occlusion. This is the most common case where objects block parts of the scene. We further consider three subtypes based on transparency: opaque, semi-transparent (e.g., smoke), and transparent (e.g., glass), which pose

[Figure 90]

3D World Environment Synthesized Object

Natural Object & Camera Trajectory

FG_BG BG Mask

[Figure 91]

|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>|
|---|

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

# VOR

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

###### SAM2

[Figure 141]

###### Real World

[Figure 142]

[Figure 143]

Segmentation

Data Cleaning & Refinement

[Figure 144]

[Figure 145]

[Figure 146]

|[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>|[Figure 155]|
|---|
<br><br>|[Figure 156]|
|---|
<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]|
|---|

[Figure 180]

[Figure 181]

[Figure 182]

Triplet Samples

Real World Environment Real World Object Camera with Tripod & Ken Burns Effect

Figure 4. Dataset Construction Pipeline of VOR. VOR is a hybrid dataset combining synthetic data and real-world captures. Synthetic data are generated in Blender using 3D environments, objects, and animations collected from public sources, together with carefully designed natural object and camera trajectories. Real-world data are recorded across diverse scenes and object categories using cameras, followed by the Ken Burns effect to simulate camera motion. All videos are segmented by SAM2 [30] and manually cleaned and refined by human annotators. The final dataset comprises triplet pairs of videos with and without the target object, and the corresponding mask.

FG_BG BG

videos that with and without target objects while keeping all other factors unchanged. These videos are captured across diverse real-world scenes, such as streets, parks, classrooms, rivers, and gyms, covering a wide range of static and dynamic objects, e.g., humans, animals, balls, and umbrellas. The dataset spans different times of day and various weather conditions, e.g., sunny, cloudy, and rainy.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Occlusion

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

0 24 0 24

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Shadow

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

0 24 0 24

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Lighting

Synthesized Data. (1) Diverse Scenes. We construct over 150 diverse 3D scenes from public repositories, covering a wide range of environments, weather, seasons, and full day lighting variations from morning to night. (2) Objects and Motion. Unlike ROSE [26], where motion dynamics are solely induced by the camera, we curate common 3D objects and manually rig their motions, trajectories, and interactions. We also design multi-object scenarios where only a subset of objects is removed, a setting largely overlooked in previous works. (3) Multi-Camera Rendering. Rather than random trajectories, we design naturalistic multi-camera placements and motion paths to better approximate realworld cinematography and viewpoint diversity.

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

0 24 0 24

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Reflection

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

0 48 0 48

Deformation

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

6 36 6 36

Figure 5. Representative side effects in VOR dataset.

different challenges for recovering occluded content from surrounding context. (2) Shadow. Objects obstruct light, producing regions with varying intensity and shape. The main challenge lies in accurately localizing and inpainting these shadowed areas under diverse illumination. (3) Lighting. Removing a light source changes scene brightness and color balance, requiring the model to estimate illumination effects on nearby regions and restore consistent lighting across frames. (4) Reflection. Objects are reflected on surfaces such as mirrors, water, or tiles. The model needs to disentangle and remove reflection artifacts while preserving the surface appearance. (5) Deformation. Objects physically deform surrounding structures, e.g., curtains, grass, or nets. The model should recover the original geometry and texture with temporal coherence once the object is removed.

Triplet Data Pairs. (1) Camera Motion Simulation. For camera-captured pairs with and without the target object, we enrich motion diversity by applying the Ken Burns effect, combining smooth pans, zooms, and handheld head bob, following 14 predefined camera motion rules. We vary camera speed and trajectory within bounds so the moving window remains within the original frame. For each pair, five motion patterns are sampled from the 14 rules.

- (2) Synthetic Data Combination. Given n objects and m camera configurations, we can construct (3n-2n)×m pairs, substantially increasing both dataset scale and diversity.
- (3) Mask Generation. To generate high-quality masks, we

Real-World Data. We use fixed cameras to record paired

Inputs

Task-Aware Region Guidance

Effect Consistency Loss

[Figure 223]

[Figure 224]

[Figure 225]

𝐴

Remove

Text prompt

LayerPool

[Figure 226]

[Figure 227]

<TASK> the specified <object> and all related effects …

TextEnc

𝑓 𝑉

|[Figure 228]<br><br>[Figure 229]|[Figure 230]|
|---|---|

|Mapper|
|---|

[Figure 231]

𝑓

[Figure 232]

[Figure 233]

###### MASKFG_BG

[Figure 234]

𝒆

Mapper

K, V

ℒEC

[Figure 235]

𝒆

[Figure 236]

Att. Map

Q

𝑓

ImgEnc

Projector

[Figure 237]

Diff (𝑉 , 𝑉 )

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Cross Att.

|[Figure 246]<br><br>[Figure 247]|[Figure 248]|
|---|---|

Cropped FG

Prompt emb

𝑀

Condition Adaptor

Diffusion Transformer

Outputs

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

𝒙

|FG|_BG| |
|---|---|---|

|[Figure 256]|[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]|
|---|---|

VideoEnc

𝒙̇𝒕

AdaptorAdaptor

Insert

DiT Block DiT Block …

|M|ASK| |𝒙|
|---|---|---|---|
| | | | |

[Figure 261]

[Figure 262]

|[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]|[Figure 267]|
|---|---|

𝒙𝒕

|[Figure 268]<br><br>[Figure 269]<br><br>N|[Figure 270]<br><br>OISE|[Figure 271]<br><br>[Figure 272]|
|---|---|---|

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

VideoDec

FGBG

𝑉

Object Removal Result

[Figure 277]

[Figure 278]

N blocks

𝒙𝒃

| |BG| |
|---|---|---|

|[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]|[Figure 283]|
|---|---|

|[Figure 284]|[Figure 285]<br><br>[Figure 286]|
|---|---|

ℒremove

ℒinsert

𝒙̇𝒕

denoise denoise

𝒙𝒇

𝑉

[Figure 287]

###### FG

[Figure 288]

[Figure 289]

Remove Diffusion Loss

Insert Diffusion Loss

|[Figure 290]<br><br>[Figure 291]<br><br>N|[Figure 292]<br><br>OISE|[Figure 293]<br><br>[Figure 294]|
|---|---|---|

𝒙𝒕

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

Object Insertion Result

- Figure 6. The framework of EffectErase. During training, removal and insertion pairs are encoded into the latent space by VAE and fused with noise via the Adaptor. Each DiT block performs cross-attention using the fused features x˙ t as Query and eprompt from Task-Aware Region Guidance as Key/Value, producing attention maps that highlight affected regions. We aggregate attention maps from all blocks and

apply max pooling to obtain a maximal-activation map, which is supervised by the effect consistency loss LEC to encourage both tasks to focus on the same affected area. At inference, users can flexibly switch the model between removal and insertion by modifying the inputs.

Removal–Insertion Joint Learning. Most existing video object removal methods treat removal as an isolated task, often leading to insufficient awareness of affected regions and making it difficult to accurately localize and restore these areas. We propose a dual-learning paradigm in which removal and insertion share a common denoising backbone. Joint optimization of the two tasks provides complementary supervision, enabling the model to learn consistent affected regions and structural cues. Specifically, video inputs are first encoded into the latent space using a pretrained VAE. The video with objects V o, the background video without objects V b, and the corresponding mask M are encoded into latent representations xo, xb, and xm, respectively.

manually provide point prompts on key frames, verify the segmentation results, and propagate them across sequences using SAM2 [30] to obtain object masks sequences. We then inspect each video segmentation result for data cleaning and manually refine the masks. Finally, by combining the validated masks with the video pairs, we construct triplet training data for subsequent learning.

Data Statistics. As summarized in Table 1, our dataset provides over 145 hours of video and 60K paired videos, spanning 366 object classes and 443 different scenes. It substantially exceeds prior datasets in both scale and diversity, offering broader object coverage and richer variations in camera motion, object motion, and background dynamics.

To construct the noisy input xt for diffusion training, a clean latent x obtained from the VAE is used, where x = xb for removal and x = xo for insertion. Random noise z ∼ N(0,I) is added through the forward process [9]:

##### 3.2. EffectErase

Overview. As shown in Fig. 6, the network encodes paired removal and insertion inputs with a pretrained VAE [18] and denoises the latents using a DiT [34]. On this backbone, our EffectErase incorporates three components: 1) Removal–Insertion Joint Learning, which trains both tasks together on the same affected regions and structural cues. 2) Task-Aware Region Guidance, which encodes object visual tokens and task-specific tokens to model spatiotemporal correlations between the object and its effects via cross attention, enabling flexible task switching; 3) Effect Consistency Loss, which enforces consistent effect regions between removal and insertion.

xt = tx + (1 − t)z, (1)

where the timestep t ∈ [0,1] is sampled from a logit-normal distribution. The denoising model vθ is trained to predict the velocity v = x−z from the noisy latent xt, the timestep t, and the condition c, with the objective defined as:

Ldenoise = Ez,x,t,c vθ(xt,t,c) − v 2, (2)

where the condition c guides the model to user-specified regions and differs across tasks: for removal, c = [xo;xm];

for insertion, c = [xb;xf]. Here [; ] denotes concatenation along the channel dimension and xf = xo ⊙ xm with ⊙ denoting element-wise multiplication.

To better fuse condition with noisy latents, we introduce a lightweight adaptor Aϕ(·) that combines xt and c:

x˙ t = Aϕ([xt;c]). (3)

Task-Aware Region Guidance. To model spatiotemporal correlations between the affected areas and objects and to support flexible switching between removal and insertion, we design a Task-Aware Region Guidance (TARG) module. Task tokens etask are extracted from a language model [29], while foreground tokens ef are obtained by feeding a cropped foreground patch from a frame of V f = V o ⊙ M into the CLIP image encoder [28]. A lightweight projector Pψ(·) maps CLIP features into the token space. The projected foreground embedding Pψ(ef) then replaces the placeholder token “object” in etask, forming a task-aware region representation:

eprompt = etask[object] ← Pψ(ef), (4)

which is injected into the backbone via cross-attention [33] to guide the model in capturing spatiotemporal effect correlations between the object and its effects, enabling accurate localization of effect-related regions and flexible switching between removal and insertion.

Effect Consistency Loss. Since video object removal and insertion are inverse operations, they share the same effect regions, covering both the object and its induced environmental changes. Under the joint-learning described above, the removal and insertion branches use different inputs and task tokens and therefore produce two sets of crossattention maps. Because cross attention highlights effectaffected regions, we introduce an Effect Consistency (EC) loss to align the two branches, using insertion as auxiliary supervision for removal. We collect cross-attention maps of each DiT block from both branches and max-pool across blocks to obtain Arm and Ain for removal and insertion, respectively. A lightweight mapper Gω(·) then projects them into soft affected region estimations:

frm = Gω(Arm), fin = Gω(Ain). (5)

As the implicitly learned affected areas may be unstable, we build a difference map prior fdiff from the normalized distribution of the downsampled difference between V o and V b. Unlike previous work [26] that employs binary masks and loses change intensity information, such as variations in illumination and shadows, our soft distribution preserves detailed variations, better capturing the magnitude of the effects. EC is computed once on the pooled maps, and gradients backpropagate through the mapper into all crossattention layers, sharpening their focus on affected regions.

The EC loss is formulated as:

LEC = KL fdiff ∥frm + KL fdiff ∥fin , (6)

which aligns effect regions across tasks and lets insertion provide complementary guidance for removal.

During training, the model is jointly optimized:

Ltotal = Lremovedenoise + Linsertdenoise + λLEC, (7) where the EC term is weighted by λ.

#### 4. Experiments

Implementation. Our method is built on the Wan 2.1 [35] video generation model and fine-tuned with LoRA [15] on the VOR dataset. The input resolution is set to 832 × 480, and 81 consecutive frames are randomly sampled for training. The model is trained for 120K iterations with a total batch size of 8 on 8 H100 GPUs, using a learning rate of 1 × 10−5 and a LoRA rank of 256. All results are generated with 50 denoising steps.

Evaluation Data. We evaluate EffectErase against existing methods on three datasets: (1) ROSE-Benchmark, a synthetic dataset that provides paired videos for object removal evaluation; (2) VOR-Eval: the test split of our VOR dataset described in Sec. 3.1, which contains 43 paired videos. (3) VOR-Wild: a test set consisting of 195 diverse realworld videos collected from the internet, featuring dynamic objects and their associated effects.

Evaluation Metrics. For datasets with ground truth (ROSE and VOR-Eval), we adopt standard fidelity metrics, including PSNR [14], SSIM [37], LPIPS [45], and FVD [32]. For VOR-Wild, which lacks ground truth, we conduct a user study where 20 volunteers rate the results, and further introduce Qscore, a metric that leverages the Qwen-VL model [2] to assess the quality of generated videos based on removal completeness and visual artifacts.

##### 4.1. Comparison with State-of-the-Art Methods.

We compare EffectErase with several state-of-the-art image inpainting methods [43, 46] applied in a per-frame manner, video inpainting methods [17, 21, 47], and advanced video object removal methods [26, 48].

Quantitative Evaluation. As shown in Table 2, current image inpainting methods [43, 46] operate on individual frames using 2D models without temporal modeling, and therefore fail to maintain temporal consistency in videos. Recent video inpainting methods [17, 21, 47] do not explicitly model object side effects, resulting in unnatural removal outcomes. Existing video object removal methods [26, 48] lack spatiotemporal correlation modeling between the object and its side effects, and consequently often produce artifacts and residual traces of the removed objects. Overall,

Table 2. Quantitative results on ROSE and VOR. The best and second-best results are highlighted in bold and underlined, respectively.

ROSE-Benchmark (with GT) VOR-Eval (with GT) VOR-Wild (without GT)

Method

PSNR↑ SSIM↑ LPIPS↓ FVD↓ PSNR↑ SSIM↑ LPIPS↓ FVD↓ QScore↑ User↑

|ObjectClear [46] OmniPaint [43]<br><br>|29.535 0.920 0.076 742.829 27.569 0.910 0.085 809.645<br><br>|22.583 0.787 0.190 1391.858 21.511 0.781 0.201 1439.867<br><br>|8.979 4.75 8.942 4.38|
|---|---|---|---|
|Propainter [47] DiffuEraser [21] VACE [17]|27.200 0.915 0.095 171.020 26.502 0.898 0.128 167.483 20.805 0.694 0.174 254.117<br><br>|21.975 0.800 0.225 589.012 21.946 0.802 0.214 559.497 17.677 0.591 0.294 806.476<br><br>|8.860 4.88<br><br>9.113 5.50<br><br><br>8.229 1.50<br><br>|
|MinMax-Remover [48] ROSE [26] EffectErase (Ours)<br><br>|26.770 0.905 0.099 137.840<br><br>31.122 0.917 0.077 72.177<br><br>32.161 0.948 0.039 55.578<br><br><br>|21.963 0.802 0.217 539.427<br><br>22.966 0.792 0.203 383.084<br><br>23.750 0.806 0.170 342.871<br><br><br>|8.984 5.90<br>9.240 6.38 9.280 7.20<br><br><br>|

Input frame / mask EffectErase (Ours) ROSE MinMax-Remover VACE Propainter

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

OcclusionDeformation

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

ShadowReflectionLighting

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

[Figure 323]

[Figure 324]

|[Figure 325]|
|---|

|[Figure 326]|
|---|

|[Figure 327]|
|---|

[Figure 328]

| |
|---|

- Figure 7. Qualitative results on VOR-Eval. Inpainting models (VACE [17], Propainter [47]) fail to erase effects beyond the mask, while removal models (ROSE [26], MinMax-Remover [48]) leave artifacts. EffectErase effectively removes the target objects and their effects.

EffectErase achieves state-of-the-art performance across all datasets and evaluation metrics. It obtains the best scores on the video quality metric FVD, demonstrating superior temporal smoothness and consistency of the generated videos. Our method also achieves the highest QScore and user feedback ratings, further demonstrating its effectiveness in producing visually convincing removal results.

Qualitative Evaluation. Qualitative comparisons are presented in Fig. 7 and Fig. 8. Video inpainting methods [17, 47] often produce artifacts in masked regions and fail to completely remove the side effects caused by the removed objects. Previous object removal approaches, such as ROSE [26] and MinMax-Remover [48], perform well in removing target objects but still struggle with side effects, especially in occlusion, shadow, lighting, reflection and deformation scenarios. In contrast, EffectErase effectively removes both target objects and their associated effects, resulting in clean, coherent, and high-quality outcomes.

Table 3. Ablation study on VOR-Eval. Based on VOR real-world data (Real), the removal performance improves progressively by adding the consistency loss (LEC), Task-Aware Region Guidance (TARG), and synthesized training data (Syn.).

Exp. Real LEC TARG Syn. PSNR↑ SSIM↑ LPIPS↓ FVD↓

- (a) ✓ 20.409 0.720 0.243 368.664

- (b) ✓ ✓ 21.020 0.737 0.224 354.545

- (c) ✓ ✓ ✓ 23.101 0.780 0.193 349.094

- (d) ✓ ✓ ✓ ✓ 23.750 0.806 0.170 342.871

##### 4.2. Ablation Studies

Effectiveness of Consistency Loss. The proposed EC loss encourages removal and insertion to focus on the same side-effect regions, strengthening the model’s attention to affected areas. As shown in the Table 3, adding the EC loss consistently improves the baseline across all metrics, with FVD decreasing from 368.664 to 354.545.

Input frame / mask EffectErase (Ours) ROSE MinMax-Remover VACE Propainter

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

OcclusionShadowLightingReflectionDeformation

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

- Figure 8. Qualitative results on VOR-Wild. EffectErase remains robust across in-the-wild scenarios such as multi-person occlusions, fastmoving sports, nighttime headlights, mirror reflections, and open-water boat scenes. Best viewed zoomed in.

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

Input Object Insertion

[Figure 370]

[Figure 371]

12 24 48

[Figure 372]

[Figure 373]

[Figure 374]

12 24 48

[Figure 375]

[Figure 376]

[Figure 377]

12 24 48

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

- Figure 9. Video Object Insertion by EffectErase. EffectErase seamlessly adapts to insertion, preserving background content while naturally integrating dynamic objects with realistic objectinduced effects, e.g., shadows and reflections.

##### 4.3. More Applications.

EffectErase can be directly adapted to object insertion by simply modifying the task prompt without additional training. As shown in Fig. 9, the model synthesizes realistic object side effects even when only the target objects are specified. In the first two rows, EffectErase generates realistic shadows for inserted dynamic objects such as a leaf and a traffic cone, while the third row shows its ability to produce natural light reflections on glossy ceramic tiles.

#### 5. Conclusion

Effectiveness of Task-Aware Region Guidance. The TARG module captures spatiotemporal correlations between objects and their side effects, enabling the model to localize and perceive affected regions. As shown in Table 3, TARG enables the model to produce higher-quality erasure results, with SSIM improving significantly from 0.737 to 0.780, validating the effectiveness of this design.

Effectiveness of Synthesized Data. Incorporating highquality synthesized data increases data diversity and exposes the model to a broader range of appearance variations and motion patterns. As shown in Table 3, training with both real and synthetic data leads to noticeably better generalization on VOR-Eval, producing cleaner backgrounds and more stable temporal restoration. This mixed training setup yields consistent improvements across metrics, with LPIPS decreasing markedly from 0.193 to 0.170.

We address the challenging effect-aware video object removal by introducing the VOR dataset and EffectErase framework. VOR is a large hybrid dataset consisting of camera-captured and synthesized videos, covering common categories of object-induced effects, with two evaluation benchmarks VOR-Eval and VOR-Wild. Building on VOR, we propose EffectErase to jointly learn video object removal and insertion. EffectErase leverages Task-Aware Region Guidance to model spatiotemporal object–effect correlations, and enforces an Effect Consistency loss to align effect regions across tasks. Extensive experiments and ablations validate the contribution of each component. EffectErase achieves state-of-the-art performance, delivering high-quality removal of objects and their effects in complex scenes, and naturally extends to realistic object insertion.

Limitation. EffectErase requires an input mask to specify the removal region, and a future direction is to support more user-friendly interactions, e.g., text and speech.

#### References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 12

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6, 14
- [3] Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. Videopainter: Any-length video inpainting and editing with plug-and-play context control. arXiv preprint arXiv:2503.05639, 2025. 2, 3
- [4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8), 2024. 2
- [5] Ya-Liang Chang, Zhe Yu Liu, Kuan-Ying Lee, and Winston Hsu. Free-form video inpainting with 3d gated convolution and temporal patchgan. In CVPR, 2019. 3
- [6] Ya-Liang Chang, Zhe Yu Liu, and Winston Hsu. Vornet: Spatio-temporally consistent video inpainting for object removal. In CVPRW, 2019. 2
- [7] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. MOSE: A new dataset for video object segmentation in complex scenes. In ICCV,

2023. 3

- [8] Henghui Ding, Kaining Ying, Chang Liu, Shuting He, Xudong Jiang, Yu-Gang Jiang, Philip HS Torr, and Song Bai. MOSEv2: A more challenging dataset for video object segmentation in complex scenes. arXiv preprint arXiv:2508.05630, 2025. 3
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 5
- [10] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, pages 249–256, 2010. 12
- [11] Bohai Gu, Hao Luo, Song Guo, Peiran Dong, and Qihua Zhou. Coherent video inpainting using optical flow-guided efficient diffusion. arXiv preprint arXiv:2412.00857, 2024. 3
- [12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, pages 1026–1034, 2015. 13
- [13] D Hendrycks. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 13
- [14] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In ICPR, 2010. 6
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 6, 13

- [16] Longtao Jiang, Zhendong Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Lei Shi, Dong Chen, and Houqiang Li. Smarteraser: Remove anything from images using masked-region guidance. In CVPR, 2025. 3, 12
- [17] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 2, 3, 6, 7
- [18] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 5
- [19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [20] Ruibin Li, Tao Yang, Song Guo, and Lei Zhang. Rorem: Training a robust object remover with human-in-the-loop. In CVPR, 2025. 3, 12
- [21] Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. Diffueraser: A diffusion model for video inpainting. arXiv preprint arXiv:2501.10018, 2025. 2, 6, 7
- [22] Qingyang Liu, Junqi You, Jianting Wang, Xinhao Tao, Bo Zhang, and Li Niu. Shadow generation for composite image using diffusion model. In CVPR, 2024. 2
- [23] Rui Liu, Hanming Deng, Yangyi Huang, Xiaoyu Shi, Lewei Lu, Wenxiu Sun, Xiaogang Wang, Jifeng Dai, and Hongsheng Li. Fuseformer: Fusing fine-grained information in transformers for video inpainting. In CVPR, 2021. 2
- [24] Yi Liu, Hao Zhou, Benlei Cui, Wenxiang Shang, and Ran Lin. Erase diffusion: Empowering object removal through calibrating diffusion pathways. In CVPR, 2025. 3
- [25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 13
- [26] Chenxuan Miao, Yutong Feng, Jianshu Zeng, Zixiang Gao, Liu Hantang, Yunfeng Yan, Donglian Qi, Xi Chen, Bin Wang, and Hengshuang Zhao. ROSE: Remove objects with side effects in videos. In NeurIPS, 2025. 2, 3, 4, 6, 7, 12, 13
- [27] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 13
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [29] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 21(140), 2020. 6
- [30] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. In ICLR, 2025. 4, 5, 12
- [31] Min-Cheol Sagong, Yoon-Jae Yeo, Seung-Won Jung, and Sung-Jea Ko. Rord: A real-world object removal dataset. In BMVC, 2022. 2, 3, 12

- [32] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. In ICLR Workshop,

2019. 6

- [33] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 6
- [34] Team Wan, Ang Wang, Baole Ai, Bin Wen, and et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 5
- [35] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 6, 13
- [36] Chuan Wang, Haibin Huang, Xiaoguang Han, and Jue Wang. Video inpainting by jointly learning temporal structure and spatial details. In AAAI, 2019. 3
- [37] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4), 2004. 6
- [38] Runpu Wei, Zijin Yin, Shuo Zhang, Lanxiang Zhou, Xueyi Wang, Chao Ban, Tianwei Cao, Hao Sun, Zhongjiang He, Kongming Liang, et al. Omnieraser: Remove objects and their effects in images with paired video-frame data. arXiv preprint arXiv:2501.07397, 2025. 3, 12
- [39] Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion. In ECCV, 2024. 3, 12
- [40] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327, 2018. 2
- [41] Jinrui Yang, Qing Liu, Yijun Li, Soo Ye Kim, Daniil Pakhomov, Mengwei Ren, Jianming Zhang, Zhe Lin, Cihang Xie, and Yuyin Zhou. Generative image layer decomposition with visual effects. In CVPR, 2025. 3, 12
- [42] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [43] Yongsheng Yu, Ziyun Zeng, Haitian Zheng, and Jiebo Luo. Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting. arXiv preprint arXiv:2503.08677, 2025. 3, 6, 7, 12
- [44] Kaidong Zhang, Jingjing Fu, and Dong Liu. Flow-guided transformer for video inpainting. In ECCV, 2022. 2, 3
- [45] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [46] Jixin Zhao, Shangchen Zhou, Zhouxia Wang, Peiqing Yang, and Chen Change Loy. Objectclear: Complete object removal via object-effect attention. arXiv preprint arXiv:2505.22636, 2025. 2, 6, 7

- [47] Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In CVPR, 2023. 2, 3, 6, 7
- [48] Bojia Zi, Weixuan Peng, Xianbiao Qi, Jianan Wang, Shihao Zhao, Rong Xiao, and Kam-Fai Wong. Minimax-remover: Taming bad noise helps video object removal, 2025. 2, 3, 6, 7
- [49] Bojia Zi, Shihao Zhao, Xianbiao Qi, Jianan Wang, Yukai Shi, Qianyu Chen, Bin Liang, Rong Xiao, Kam-Fai Wong, and Lei Zhang. Cococo: Improving text-guided video inpainting for better consistency, controllability and compatibility. In AAAI, 2025. 3

### Supplementary Material for EffectErase

In the supplement, we provide additional dataset details in Sec. A, further method descriptions in Sec. B, and more qualitative results in Sec. C.

#### A. Details of Dataset Construction

In this section, we provide a detailed description of the captured and rendered components of our Video Object Removal (VOR) dataset used to train EffectErase.

##### A.1. Real-World Data

Consistent Data Pairs. Each pair consists of one video where the target object is present with its effects and a counterpart where both are absent. To keep the two recordings identical, as shown in Fig. I, we develop a custom capture app that locks exposure and focus across the entire pair, ensures matched file names and fixed recording durations, enables Bluetooth triggering to avoid screentouch motion, and uses a tripod to eliminate camera shake.

Diverse Scenes and Objects. We collect data across a wide range of real-world environments, including parks, campuses, and streets, spanning a total of 293 scenes and covering over 45 scene categories. The dataset also features a broad set of objects, ranging from static items such as sports balls and tools to dynamic subjects including children, teenagers, and various vehicles.

Ken Burns Effects. We propose an extended version of the Ken Burns effect that provides fourteen distinct cameramotion patterns. These include basic zoom-in and zoomout motions; directional motions such as panning left or right and tilting up or down; combined zoom–translation motions; a walk-bob motion that mimics the vertical sway of handheld footage; and a random-combo mode that randomly mixes zoom and translation directions. For each clip, we randomly select five motion types and assign each type a randomized zoom curve and translation intensity. The module then updates a virtual camera center over time and crops the corresponding view to a fixed resolution, producing natural and diverse camera-movement variants that enhance training for the video object removal model.

##### A.2. Synthesized Data

3D Enviroments. We collect 150 high-quality 3D environment assets from free online resources. These scenes cover a wide range of realistic daily-life settings across both

[Figure 383]

[Figure 384]

[Figure 385]

Figure I. Data capture software. Our app records aligned video pairs by locking exposure and focus, matching file names and durations, enabling reliable Bluetooth triggering for stable control, and using a tripod to remove camera shake.

indoor and outdoor domains, e.g. city streets, farms, coastal areas, mountains, parking lots, classrooms and forests.

Characters with Animations. We include a diverse set of animated characters and objects, such as dancing humans, walking bears, moving boats, and flying balloons, covering realistic, anime, and game-style visual domains.

Camera Trajectories. Due to the wide variety of camera motions and shooting angles in real scenarios, we aim to cover as many camera movement patterns as possible. To this end, we manually design both realistic camera paths and natural camera motion behaviors such as zoom and pan, thereby ensuring that the synthesized movements closely mimic human-operated filming practices.

Table I. Comparison of object removal datasets. Image-level datasets are listed above the line, and video-level datasets are listed below. “–” denotes unreported or not applicable. Synth. (3D) denotes data generated using a graphics rendering engine, while Synth. (paste) denotes data created by directly pasting cropped foreground objects onto backgrounds.

|Dataset<br><br>|Source<br><br>Dynamic Camera<br><br>Dynamic Object<br><br>Dynamic Background<br><br>Scene Types<br><br>Object Classes<br><br>Image Pairs<br><br>Video Pairs<br><br>Average Duration (s)<br><br>Total Hours|
|---|---|
|ObjectDrop [39] Syn4Removal [16] LayerDecomp [41]<br><br>OmniPaint [43] RORem [20]|Real ✗ ✗ ✗ – – 2.5K – – –<br><br>Synth. (paste) ✗ ✗ ✗ – – 1,000K – – – Synth. (paste) ✗ ✗ ✗ – – 6.0K – – –<br><br>Real ✗ ✗ ✗ – – 3.3K – – –<br><br><br>Synth. (paste) ✗ ✗ ✗ – – 201.1K – – –|
|RORD [31] Video4Removal [38] ROSE [26] VOR (Ours)<br><br>|Real ✗ ✓ ✗ 24 76 516.7K 3,106 – 5.98 Real ✗ ✓ ✗ 6 – 134.3K – – 1.55<br><br>Synth.(3D) ✓ ✗ ✗ 25 102 1,501.0K 16,678 6.00 27.79 Real + Synth.(3D) ✓ ✓ ✓ 67 366 12,556.8K 60,000 8.72 145.33<br><br>|

##### A.3. Mask Annotation

We first provide a point prompt to obtain the mask in the first frame and manually verify its quality. The same point prompt is then fed to SAM2 [30] to propagate the mask across the entire sequence. We review all propagated mask sequences and remove those that fail to maintain stable and complete object coverage across all frames.

##### A.4. Dataset Statics

As shown in Table I, we provide a detailed comparison between our VOR dataset and existing image- and videobased removal datasets. We summarize the image-based datasets and the video-based datasets. Compared with prior work, VOR offers substantially richer scene diversity, broader object coverage, longer video durations, and a significantly larger number of paired sequences.

Since no unified scene taxonomy exists across datasets, we introduce our own categorization scheme to standardize all scene types in Fig. II, covering both indoor and outdoor environments with a total of 67 comprehensive categories. Specifically, for RORD [31], its original scene labels are merged into our taxonomy; for Video4Removal [38], scene types are assigned based on the descriptions in the paper and aligned with our scheme; and for ROSE [26], we manually inspect every scene in the raw data and annotate them according to our proposed categorization.

For video pair counts, the numbers for RORD [31] are obtained by counting the lowest-level folders in the dataset structure. The total video hours of RORD [31] and Video4Removal [38] are estimated by converting the total frame count to duration using 24 fps.

#### B. Method Details

##### B.1. Details of the Proposed modules

Adaptor Details. The adaptor is implemented as a 3D convolutional layer with a kernel size of 1×2×2 and a stride of 1×2×2. To improve convergence, the first sixteen input

###### Outdoor (41)

###### Urban (13)

street/road; downtown; market; plaza; alley; bridge; parking lot; construction site; harbor / dock; garden; amusement park; outdoor cafe; night market.

Natural (19) park; forest; grassland; mountain; river; lake; waterfall; beach; sea; island; cave; snowfield; garden; bamboo grove; hot spring; canyon; hill; farm; greenhouse. Transportation (3) highway; subway/bus station; gas station. Sports / Athletic (6) tennis court; basketball court; soccer field; table tennis; volleyball; running tack.

###### Indoor(26)

Home spaces(9) living room; bedroom; dining room; kitchen; bathroom; balcony; restroom; garage; home office. Commercial / Public (17) store; corridor; cafe; restaurant; gym; hotel; office; meeting room; classroom; lab; dormitory; Library; exhibition hall; warehouse; shopping mall; zoo; factory/workshop.

Figure II. Scene category hierarchy. Our taxonomy organizes 67 scene types into structured outdoor and indoor groups.

channels of its weights are copied from the convolution used in the original patch-embedding module, while the remaining channels are initialized with Xavier uniform initialization [10]. All bias terms are zero-initialized.

Projector Details. The projector maps the object-image features extracted by the image encoder into the latent space required by our model. It is composed of two sequential MLP blocks: the first transforms the input embedding dimension to the output dimension, and the second further refines the representation with a residual MLP. Each block applies LayerNorm [1], a linear projection, a GELU activa-

You are an expert evaluator for video object removal quality. You will be shown 2 images:

- - The FIRST image is the reference frame before removal, with an ORANGE MASK marking the object/person to be removed
- - The SECOND image is the first frame of the video after removal Please evaluate the removal quality based on the following criteria (0-10 scale, BE STRICT):

- 1. **Completeness**: Is the target marked by the orange mask completely removed?
- 2. **Artifacts**: Are there any blur, distortion, or other artifacts in the removal area?
- 3. **Secondary Effects**: Are shadows, reflections, lighting effects, etc. eliminated?
- 4. **Background Quality**: Does the inpainted background blend naturally? Scoring Guidelines:

- - 10: Perfect - target completely invisible, zero artifacts, no secondary effects
- - 9: Nearly perfect - only extremely minor edge artifacts
- - 8: Target completely removed, slight edge artifacts, no secondary effects
- - 7: Target removed with visible artifacts but good overall quality
- - 6: Target removed, noticeable artifacts OR minor secondary effects
- - 5: Target removed, obvious artifacts AND secondary effects
- - 4: Mostly removed, significant artifacts OR secondary effects
- - 3: Removed but severe artifacts
- - 2: Not fully removed OR extremely severe artifacts
- - 1: Barely removed
- - 0: Complete failure CRITICAL RULES:
- - Score 10 is EXTREMELY RARE
- - ANY visible secondary effects (shadow/light/reflection) = score 6 or below
- - ANY noticeable artifacts = score 6 or below
- - Most results should fall in 4-7 range

Please respond in the following format: <think> [Brief analysis: what is marked by the orange mask in reference image, completeness of removal, artifacts, secondary effects, background quality] </think> <result>[score 0-10]</result>

Figure III. Prompt used for QScore evaluation. The prompt guides Qwen-VL to assess removal completeness and visual artifacts.

tion [13], and a second linear projection, while the second block includes a residual connection. A final LayerNorm is applied to stabilize the projected token.

Mapper Details. The mapper predicts an effect-area distribution map from the fused cross-attention features. We aggregate the cross-attention maps from all DiT layers [27] and apply max-pooling across layers to obtain a compact feature volume. This volume is then processed by the mapper, implemented as a lightweight per-pixel MLP operating on the channel dimension. The module applies a linear projection, a GELU activation [13], and a second linear projection to produce a logit map for each frame.

##### B.2. Training details

Similar to previous work [26], the backbone model is a controllable generation variant of Wan2.1 1.3B [35]. We optimize the network with AdamW [25] using a learning

rate of 1 × 10−4 and a batch size of 1 through gradient accumulation. Training is conducted for up to 120K iterations. To adapt the base model to the video object-removal task, we apply LoRA [15] to the attention projections q,k,v,o and the feed-forward layers ffn.0 and ffn.2, with all LoRA weights initialized using Kaiming initialization [12].

##### B.3. Inference details

During inference, the model supports both removal and insertion. For removal, we provide the input video together with a mask video, and the model outputs the objectremoved result. For insertion, we provide the background video and an object video, and the model generates the inserted output. All denoising steps are set to 50.

- B.4. Metric details

QScore. To further assess the removal quality, we use the Qwen-VL model [2] to evaluate each removed video with a designed prompt as shown in Fig. III. The evaluation considers both removal completeness and visual artifacts, and the final QScore is obtained by averaging the results.

User Study. We conduct a user study with 20 volunteers, where each participant scores 195 generated videos from VOR-Wild, and the final score is obtained by averaging all individual ratings across participants.

- C. More Results

- C.1. Effect-region Erasing Evaluation

As shown in the Tab. II, EffectErase effectively removes effect regions outside the object mask, with evaluation metrics computed only over the corresponding effect regions.

Table II. Effect-region erasing evaluation.

|Method<br><br>|PSNR ↑ SSIM ↑ LPIPS ↓ FVD ↓|
|---|---|
|ROSE EffectErase<br><br>|30.267 0.930 0.084 135.013 32.747 0.939 0.069 98.266<br><br>|

##### C.2. More Results of the Insertion Task

- Please refer to Fig. IV for additional results of EffectErase applied to the insertion task.

C.3. More Results of EffectErase

- Please refer to Fig. V for additional results of EffectErase on in-the-wild data.

C.4. More Comparison with SOTA Methods

- Please refer to Fig. VI for additional qualitative comparisons with state-of-the-art methods.

##### C.5. Failure Cases and Analysis

Failure cases mainly arise when it is ambiguous whether effects or accessories belong to the target object. As shown in the Fig. VII, 1) the residual lighting may originate from other light sources, yet remains visually natural after removal; 2) parts of the dog’s shadow are heavily entangled with the person’s shadow, and the leash cannot be clearly assigned to either the dog or the person.

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

###### 1) 2)

Input video / mask Object Removal Input video / mask Object Removal

Figure VII. Failure cases when effects or accessories cannot be clearly attributed to the target object.

Input Object Insertion

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

12 24 48

64

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

12 24 48

64

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

12 24 48

64

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

12 24 48

64

- Figure IV. More insertion results of EffectErase.

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

InputEffectEraseEffectEraseEffectEraseEffectEraseEffectEraseInputInputInputInput

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

- Figure V. More removal results of EffectErase.

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

InputVideo

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

0 36

72 0 36 72

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

ROSEEffectErase

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

0 36

72 0 36 72

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

0 36

72

0 36 72

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

MiniMax-

Remover

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

0 36

72

0 36 72

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

VACE

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

0 36 72

0 36

72

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

DiffuEraserPropainterOmniPaintObjectClear

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

0 36 72

0 36 72

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

0 36 72

0 36 72

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

0 36 72

0 36 72

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

0 36 72

0 36 72

Figure VI. More comparison with state-of-the-art methods.

