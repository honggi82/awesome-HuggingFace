## AniDoc: Animation Creation Made Easier

Yihao Meng1,2 Hao Ouyang2 Hanlin Wang3,2 Qiuyu Wang2 Wen Wang4,2 Ka Leong Cheng1,2 Zhiheng Liu5 Yujun Shen2 Huamin Qu†,1

1 HKUST 2 Ant Group 3 NJU 4 ZJU 5 HKU

# arXiv:2412.14173v2[cs.CV]30Jan2025

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

Figure 1. AniDoc colorizes a sequence of sketches based on a character design reference with high fidelity, even when the sketches significantly differ in pose and scale. Additionally, the model supports sparse sketch inputs, enabling effective interpolation and highquality colorization simultaneously, as shown in the last row.

### Abstract

character and each line art frame. In addition, our model could even automate the in-betweening process, such that users can easily create a temporally consistent animation by simply providing a character image as well as the start and end sketches. Our code is available at: https://yihaomeng.github.io/AniDoc demo.

The production of 2D animation follows an industrystandard workflow, encompassing four essential stages: character design, keyframe animation, in-betweening, and coloring. Our research focuses on reducing the labor costs in the above process by harnessing the potential of increasingly powerful generative AI. Using video diffusion models as the foundation, AniDoc1 emerges as a video line art colorization tool, which automatically converts sketch sequences into colored animations following the reference character specification. Our model exploits correspondence matching as an explicit guidance, yielding strong robustness to the variations (e.g., posture) between the reference

### 1. Introduction

The animation industry, particularly in the realm of 2D anime production, relies heavily on the meticulous process of coloring line art to bring characters and scenes to life. Colorization of line art [14, 15, 20, 34, 44, 53, 67] in videos is a critical task that not only adds aesthetic value but also enhances the storytelling experience by conveying emotions and actions vividly. Automating this process holds signifi-

1“Doc” is one of the seven dwarfs in Snow White and the Seven Dwarfs, the first animated feature film produced by Disney.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

|[Figure 48]<br><br>[Figure 49]|
|---|

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

Reference character design

Keyframe sketches Inbetweening Colorization

- Figure 2. Illustration of the workflow of 2D animation production.

cant potential to streamline production workflows, reduce labor costs, and accelerate content creation, meeting the growing demand for high-quality animated content.

In the current anime production pipeline, artists typically begin with character design sheets that define the visual attributes of characters. These designs are then translated into keyframe sketches — crucial frames that outline the primary poses and movements in a scene. Next, artists create in-betweening sketches, which are the frames drawn between the keyframes to define the detailed movement and transition [41, 65]. Traditionally, these frames are manually colored, a time-consuming task that involves careful attention to ensure consistency with the original character designs. Fig. 2 illustrates each step of this pipeline. Our work aligns seamlessly with this pipeline, aiming to automate the colorization process while maintaining fidelity to the original character designs and ensuring temporal consistency across frames.

However, automating line art colorization [28, 59] presents several challenges. One primary difficulty lies in the mismatch between the character design and the line art sketches, where the angles, proportions, and poses in the design may not align with those in the keyframe sketches. Additionally, achieving temporal consistency is crucial; colorizing each frame individually can lead to flickering or inconsistencies, detracting from the viewer’s experience [5, 32, 64]. Previous approaches [21, 46, 56] have attempted to address these challenges but with limitations. They often assume the availability of colorized versions of keyframes and rely on dense line art guidance. This assumption significantly increases the workload on artists, as it requires manual coloring of multiple keyframes and detailed line art inputs, making the process tedious and labor-intensive. Moreover, some methods suffer from color information leakage due to their training pipelines. Specifically, they use non-binarized sketches extracted from color images using neural networks for training, unintentionally incorporating color information from the original images into the sketches. This information leakage undermines the practicality of these methods, as real-world sketches do not contain such implicit color information—a concern we analyze further in our methodology.

To overcome these challenges, we propose a novel all-inone model that streamlines the colorization process within a single framework. Our model leverages the priors from pretrained diffusion-based video generation models [1, 37],

harnessing their ability to capture temporal dynamics and visual coherence. The key designs of our approach are as follows: Correspondence-guided Colorization: We address the misalignment between the reference character design and the input line art sketches by incorporating an explicit correspondence mechanism. The injection module is designed to integrate color and style information from the reference into the line art, effectively improving color accuracy and consistency. Binarization and Background Augmentation: To reflect real usage scenarios, we binarize the condition sketches, forcing the model to truly learn to extract color information from the reference character design, rather than relying on recovering color information leaked from the non-binarized sketches. This constraint poses additional challenges for the model to accurately colorize the line art. To mitigate the instability during training due to this reduced information, we incorporate background augmentation strategy which greatly improve the colorization result. Sparse Sketch Training: Our model employs a two-stage training strategy that first learns the colorization ability and then removes the intermediate sketches to learn the interpolation ability. By learning the interpolation between keyframes, our model maintains temporal consistency without extensive human intervention.

Our method demonstrates superior performance both quantitatively and qualitatively compared to existing approaches. It effectively colorizes line art sketches in videos, maintaining high fidelity to the reference character designs and ensuring temporal consistency across frames. Moreover, we demonstrate that a single reference character image can be used to colorize sketches from different segments featuring the same character, even when these sketches differ significantly in scale, pose, and action from the reference design. Our work represents a significant step toward automated, efficient, and artistically consistent animation production, with potential applications extending beyond anime to various forms of digital art and media.

### 2. Related Work

#### 2.1. Line Art Image Colorization

Line art colorization [2, 13, 25, 30, 38, 58–60, 63] differs from natural image colorization, as it lacks an illuminance channel and only contains structural information, offering more flexibility. Traditional methods [36, 43] rely on users manually adding color to specific regions. The advent of deep learning has advanced this field, with techniques like color hint points [57], color scribbles [11], text tags [26], and natural language prompts [66]. Reference-based colorization, where users provide a reference image to guide the coloring, has also gained popularity. Methods like [39] use segmented graphs, while Chen et al. [9] employ active learning, and [3] apply attention networks. AnimeDiffu-

sion [4] introduces the first diffusion-based reference-based framework for anime face colorization. However, these methods colorize sketch images separately, struggling with temporal coherence when colorize a sequence of sketches.

#### 2.2. Reference-based Line Art Video Colorization

Several approaches extend reference-based colorization to videos. Given an input of reference image, these methods colorize the corresponding sketch sequence based on the reference image’s color information. LCMFTN [62] trains a model using animation frame pairs but lacked temporal coherence. TCVC [46] uses previously colorized frames to maintain short-term consistency, but errors accumulate over time. TRE-Net [48] mitigates this by using both the first frame and previously generated frames. ACOF [56] propagates colors based on optical flow but requires refinement. The most recent work LVCD [21] proposes the first video diffusion model to colorize sketch video, but suffer from the issue of non-binarized sketch information leakage, limiting its real-world applicability. Most importantly, all of these methods require the color version of the first frame of each video clip, while in the actual anime production workflow, colorists typically only receive the character design image and need to use that image to colorize different clips featuring the character.

#### 2.3. Video Interpolation

Unlike reference-based colorization, video interpolation [12, 19, 22, 23, 27, 49] aims to generate in-between frames from both the first and last frames. SparseCtrl [16] and SEINE [10] extend video interpolation using pretrained text-to-video diffusion models and additional image conditions. Shi et al. [40] adapt video interpolation for cartoon animation with temporal constraint networks. AnimeInterp [41] interpolates middle frames by warping with predicted flows. To improve animation quality, EISAI [8] uses forward warping to prevent artifacts. ToonCrafter [52] introduces a diffusion-based video interpolation model. Zhu et al. [65] proposes a thin-plate spline-based interpolation module for animation sketch inbetweening. We aim at an all-in-one model that simultaneously performs automatic interpolation and colorization for anime. However, more precise interpolation methods [49], can be integrated to achieve enhanced control over sketch manipulation.

### 3. Method

We formally define the problem of line art video colorization with reference images. Given a reference image Iref that encapsulates the desired color and style of the character and a sequence of binarized line art sketches {St}Tt=1, where St is the sketch at time frame t, our objective is to generate a sequence of colorized frames {It}Tt=1 such that: 1. Each frame It is a colorized version of the sketch St.

- 2. The colorization is consistent with the character design.
- 3. The sequence {It} is temporally coherent, ensuring smooth transitions without flickering artifacts.

Formally, we aim to learn a function f that maps the sketches and the reference image to the colorized frames:

{It}Tt=1 = f({St}Tt=1,Iref). (1)

#### 3.1. Motivation and Pipeline Design

We summarize the shortcomings of state-of-the-art approaches when handling real-world animation production scenarios and design modules to overcome them.

Mismatch Between Character Design and Input Sketches. Existing methods rely on the assumption that the reference image is strongly pixel-aligned with the first frame of the sketch sequence. When the reference provided is not the first-frame image, but instead a character design from a different angle, the model struggles to correctly match the colors and details, as shown in Fig. 5. This limitation necessitates manual coloring for each clip’s keyframes in the animation, which is labor-intensive and counteracts the benefits of automation. To address this, we aligns and transfers color information from reference character design to input sketches with correspondenceguided control module, as described in Sec. 3.2.

Degradation with Binarized Sketches. Previous methods often rely on sketches extracted from color images using learned neural networks. These non-binarized sketches contain unintended color information leaks from the original color images, although invisible to the human eye. When training with these non-binarized sketches, the model tends to learn how to recover these hidden color information rather than correctly learning to find the corresponding parts in the reference to color the sketch, resulting in models that perform poorly when applied to binarized sketches. As depicted in Fig. 4, when previous methods are applied to binarized sketches, the outputs suffer from severe degradation. The colorization is inaccurate, and the visual quality is significantly reduced compared to when non-binarized sketches are used. To address this issue, we mirror the real production conditions by adopting the binary sketch in training and apply background augmentation to enhance the robustness as in Sec. 3.3.

Reliance on Dense Sketches as Conditions. Previous methods often require dense sketches to maintain temporal consistency. Manually drawing inbetweening sketches in an animation is costly. We aim to use only S1,ST to further improve the scalability of automatic creation and propose the sparse sketch training scheme in Sec. 3.4.

Pipeline Design. Following Stable Video Diffusion (SVD) [1], our main architecture consists of a denoising 3D U-Net designed for video generation. The reference image latent is duplicated across the number of frames and concatenated along the channel dimension with the

[Figure 68]

[Figure 69]

[Figure 70]

Point map

Dense-sketch

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

|[Figure 76]<br><br>[Figure 77]<br><br>|
|---|

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

3D-UNet

Reference character

Noisy latents

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

|[Figure 90]<br><br>[Figure 91]<br><br>|
|---|

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Training

…

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

SIFT

[Figure 105]

Fusion

Encoder

Sketch

[Figure 106]

|[Figure 107]<br><br>[Figure 108]<br><br>|
|---|

|[Figure 109]<br><br>[Figure 110]<br><br>|
|---|

|[Figure 111]<br><br>[Figure 112]<br><br>|
|---|

|[Figure 113]<br><br>[Figure 114]<br><br>|
|---|

|[Figure 115]<br><br>[Figure 116]<br><br>|
|---|

[Figure 117]

Inference

…

DIFT

Input sketch

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Sparse-sketch

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Corr. Encoder

|[Figure 129]<br><br>[Figure 130]<br><br>|
|---|

|empty|
|---|

|empty|
|---|

|[Figure 131]<br><br>[Figure 132]<br><br>|
|---|

|[Figure 133]<br><br>[Figure 134]<br><br>|
|---|

[Figure 135]

…

[Figure 136]

Correspondence guidance

Output video

- Figure 3. Overview of AniDoc pipeline. We adopt a two-stage training strategy. In the dense-sketch training stage, we explicitly extract matching keypoints pairs between the reference image and each frame of the training video, constructing point maps to represent the correspondences. In the sparse-sketch training stage, we remove the intermediate frame sketches and use the matching points from the start and end frames to interpolate point trajectories, which guide the generation of the intermediate frames.

|[Figure 137]<br><br>Empty Image|
|---|

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Reference Non-binarized sketch Binarized sketch

[Figure 146]

- Figure 4. Illustration of color leakage issue in non-binarized sketch. For previous video colorization method [21], when given non-binarized sketch, even if the reference is an empty image, it can still generate colorized results with similar color pattern to the ground truth. After binarizing the sketch, the colorization results degrade significantly.

where z0 denotes VAE-encoded latent feature of the reference image, csketch means the sketch control signal, ccorr means the correspondence control signal, and ϵcθ is the combination of the denoising U-Net and the control branch.

#### 3.2. Correspondence-guided Colorization

During training, we use an off-the-shelf keypoint matching method LightGlue [29] with SIFT descriptor [31] to extract matching keypoints between the reference image and the first frame of the training video. The matched keypoints are

denoted as (xiref,yrefi ),(xi1,y1i) ni=1, where n refers to the number of matching pairs. Based on these matched key-

points, we construct a point map pair P1 = (Pref,P1,sketch), representing the matching correspondence of the reference image and the first frame. Each point map is in the size of H × W (same as the image resolution), where the coordinates corresponding to the matched keypoints are marked with the same integer label:

noisy latent so that the reference image information can be integrated into the colorization process through multiple self-attention layers in the 3D U-Net encoder. To inject the correspondence between the reference character design and the sketch, we explicitly extract corresponded keypoints and construct point maps. The correspondence information and sketch information are then encoded by a 3D U-Net dual branch similar to [61] , and injected into the main branch as control signals. To construct training pairs of reference character designs and corresponding sketch sequences, we select long videos from the Sakuga-42M dataset [33], where the characters naturally undergo rich transformations, including changes in position, posture, angle, scale, etc., which align with the needs of our model.

Pref(xiref,yrefi ) = P1,sketch(xi1,y1i) = i. (3) For unmatched pixels, the value is set to 0.

We then employ Co-Tracker [24] to track the move-

ment of keypoints (xi1,y1i) ni=1 in each video frame and construct correspondence point maps in the same way.

As a result, we obtain a point map Pseq = {Pt}Tt=1 ∈ R2×T×H×W, which explicitly encodes the correspondence between the reference image and each sketch in the training video.

We concatenate the point map Pseq with the reference image Iref together as explicit correspondence guidance information and integrate it into the control branch. Specif-

The training objective for our model is as follows:

ically, for a given keypoint (xit,yti) in sketch frame St, the model can use the point map Pt to obtain the corresponding

t,z0,t,ϵ ϵ − ϵcθ zt;t,z0,csketch,ccorr 2 , (2)

L = Ez

location (xiref,yrefi ) in the reference image Iref and extract corresponding color information at that location. Thus, our video generation process can be represented as:

{It}Tt=1 = D(Iref,E({St}Tt=1,{Pt}Tt=1,Iref)), (4)

where D refers to the denoising process of the 3D diffusion U-net, E refers to the control branch encoder.

Semantic Keypoint Matching During Inference. During training, we apply LightGlue [29] with SIFT descriptor [31] for keypoint selection and matching between the reference image and the training video frames due to its fast speed. However, during inference, we do not have the ground truth color image to extract corresponding keypoints. Methods like the SIFT descriptor, which are focusing on low-level image feature, fail to correctly match keypoints between the color reference image and the sketch due to the large domain gap. One recent work DIFT [45] have found that features extracted by diffusion models can achieve semantic-level matching. Thus, during inference, we first extract keypoints in the reference character design using X-Pose [54], and find the matching keypoints in the given sketch using semantic feature DIFT.

#### 3.3. Binarization and Background Augmentation

In real animation production, sketches provided to colorists are typically binarized line drawings without grayscale shading or hidden color information. To simulate this realworld condition, we apply a binarization process to the extracted sketches during training by setting pixel values greater than 200 to 255 and others to 0. This converts the sketch Sraw into a binary image Sbin, with black lines (0) and white background (255).

Using binarized sketches as conditioning inputs poses significant challenges for the model. One primary issue is the ambiguity between the background and large white regions in the foreground. Both are represented by the same pixel value (255), making it difficult for the model to distinguish. This ambiguity can lead to confusion during colorization, resulting in erroneous outputs where the model may incorrectly color background regions or fail to accurately color foreground elements. Such failure cases are evident in our ablation studies (refer to Sec. 4.4).

To address this problem, we enhance the training process through background augmentation. Specifically, we randomly remove the background of the reference image with a 50% probability during training process, using an offthe-shelf background removal model [35]. This forces the model to learn to distinguish between the foreground and background regions. In the foreground region, the model learns to extract color information from the correct areas of the reference character design. In the background region, the model is required to rely more heavily on its internal

generative prior, enabling it to produce a background that is coherent with the foreground character.

#### 3.4. Sparse Sketch Training

To further reduce the necessity of drawing intermediate sketches for animations with simple motions, we introduce a sparse sketch training strategy in a two stage way. In the first stage, the training is conducted with all frame sketches available. We hope that the model learns how to correctly extract information from the point map in this stage, which will guide the training in the next stage.

After completing the first stage training, we remove the sketch condition for intermediate frames and use keypoints information to guide the interpolation. Inspired by [50], we additionally transform the keypoint coordinates into a Gaussian heatmap Gt which is more suitable for trajectory control, and concatenate Gt with the original point map Pt together. Specifically, the model is conditioned on the start and end sketches, S1 and ST, as well as the pointmap Pt ⊕ Gt at each frame:

{It}Tt=1 = D(Iref,E(S1,ST,{Pt ⊕ Gt}Tt=1,Iref)). (5) Noted that during training the point trajectories are ob-

tained through point tracking in the training video using CoTracker [24], while during inference we extract matching keypoints pairs between the start and end sketches and linearly interpolate the intermediate point trajectories.

In this sparse-sketch training stage, we randomly select up to 5 keypoints, corresponding to 5 trajectories. The selection probability is determined by the magnitude of the motion, with trajectories having larger motion being more likely to be selected. Guided by the keypoint trajectories, our model can produce smooth intermediate colorized frames with only sparse sketch inputs.

### 4. Experiments 4.1. Implementation Details

Our method is built upon SVD and is trained on the Sakuga42M [33] dataset, which comprises a large number of anime clips with diverse styles. To create reference images and sketch videos with large differences, we exclude clips with fewer than 50 frames, ultimately retaining around 150k video clips. We set the interval between the reference image and the first frame of the target video to 32 frames, with the target video length being 14 frames. During the first training stage, where the generation is conditioned on perframe sketches, we simultaneously fine-tune all parameters of both the U-Net and ControlNet, including both the spatial and temporal attention layers. This is done for 100k steps using the AdamW optimizer with a learning rate of 1 × 10−5, at a resolution of 256 × 256 due to GPU memory constraints. Subsequently, we freeze other layers

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Ours

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

LVCD

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

LVCD+IP

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

ID-animator

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

ToonCrafter+IP

[Figure 221]

[Figure 222]

[Figure 223]

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

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Ours

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

[Figure 258]

LVCD

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

LVCD+IP

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

ID-animator

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

ToonCrafter+IP

- Figure 5. Visual comparison of reference-based colorization with four methods LVCD [21], LVCD+IP-Adapter [55], ID-animator [17], ToonCrafter [52].

and fine-tune the spatial layers for an additional 10k steps at a resolution of 512 × 320. We randomly select up to 50 keypoints to enhance the model’s robustness to varying

numbers of keypoints. During the sparse sketch training stage, we remove the middle frames’ sketches and further fine-tuned all parameters for 100k steps. The training is

- Table 1. Quantitative comparison with existing baselines on reconstruction and generative metrics.

Method PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓ ID-Animator 15.61 0.3129 0.5151 158.16 677.61 LVCD 15.77 0.6446 0.2580 121.98 584.33 LVCD + IP 16.52 0.6404 0.2639 118.28 496.45 ToonCrafter + IP 14.97 0.3983 0.4532 110.48 492.10 AniDoc (Ours) 19.23 0.7720 0.1704 54.33 230.18

- Table 2. Ablations on correspondence matching module and binarization + background augmentation.

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

w/omatchingw/obinarizefullmodel

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Reference

| |
|---|

| |
|---|

| |
|---|

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

- Figure 6. Ablations on each component. “w/o matching” indicates without the corresponding matching module, “w/o binarize” indicates without binarization and background augmentation.

Settings PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓ w/o binary 15.41 0.6639 0.2471 94.89 503.23 w/o matching 17.80 0.7068 0.2252 75.91 273.86 AniDoc (Ours) 19.23 0.7720 0.1704 54.33 230.18

conducted on 16 NVIDIA A100 GPUs with a total batch size of 16, and the entire training process takes 5 days.

#### 4.2. Comparison

To comprehensively evaluate the coloring ability of our model, we randomly select 200 clips from 10 different eras and styles of anime to construct the test set. We select corresponding character design images (without background) as reference images.

racy: we measure the similarity of the colorized frames and the original animation frames using reconstruction metrics including PSNR, SSIM, and LPIPS. For all the metrics, we resize the frames to 256 × 256 and remove the background for every frame because the reference character design does not include background information.

We compare our proposed method with two state-of-theart reference-based line art video colorization frameworks: LVCD [21] and ToonCrafter [52], both of which are based on video diffusion models. Since LVCD’s original setting requires the colorized version of the first sketch as a reference, we also compare it with the IP-Adapter [55] + LVCD version. In this case, we first use a diffusion-based image colorization method IP-Adapter to colorize the first frame’s sketch, and then use this colorized image as the reference for LVCD. For ToonCrafter, which requires color versions of both the start and end frames, we first use IPAdapter to colorize the sketches of both frames. Additionally, we select a recent video personalization method, ID-animator [17], which excels at identity preservation in general domains and can achieve a similar function to colorization when combined with ControlNet [61].

As shown in Tab. 1, our method achieves the best scores across all metrics, indicating high-quality colorization results. For a more comprehensive assessment, we recommend reviewing the supplementary comparison videos.

#### 4.3. Flexible Usage

We assess the flexibility of our model in three distinct settings, as depicted in Fig. 7.

Same Reference with Varying Sketches. By using the same reference, our model is able to generate consistent colorizations across different video clips, even when the sketches differ significantly in terms of pose or scale.

Same Sketch with Different References. When applying different reference images to the same sketch sequence, our method preserves the identity of the character while adapting the finer details, such as lighting and background, according to the distinct styles of the references.

Qualitative Comparison. As shown in Fig. 5, our method produces significantly clearer textures and better preserves the character’s identity. It performs especially well in scenarios with substantial differences between the reference character design and input sketches, where LVCD and IDAnimator fail to accurately colorize the sketches. Even when providing the colorized first frame using IP-Adapter for these baselines, our method still outperforms them in both visual quality and identity preservation.

Sparse Input Sketches. Thanks to our two-stage training strategy, AniDoc supports animation with sparse sketches. By using only the start and end sketches, the model effectively produces smooth and coherent animations.

#### 4.4. Ablation Study

Quantitative Comparison. We evaluate the quality of the colorized animations in two aspects: 1). Frame and Video Quality: we adopt Frechet Inception Distance [18] (FID) to measure image quality and assess video quality using Frechet Video Distance [47] (FVD). 2). Colorization Accu-

We perform ablation studies Tab. 2 on two key components. Effect of Correspondence Matching. Without the correspondence matching module, the model struggles to localize and transfer detailed color information accurately. As shown in the Fig. 6, the character’s black sideburns

|[Figure 315]<br><br>[Figure 316]| |[Figure 317]<br><br>[Figure 318]|[Figure 319]<br><br>[Figure 320]|[Figure 321]<br><br>[Figure 322]|[Figure 323]<br><br>[Figure 324]|[Figure 325]<br><br>[Figure 326]|
|---|---|---|---|---|---|---|
|[Figure 327]|[Figure 328]|[Figure 329]<br><br>[Figure 330]|[Figure 331]<br><br>[Figure 332]|[Figure 333]<br><br>[Figure 334]|[Figure 335]<br><br>[Figure 336]|[Figure 337]<br><br>[Figure 338]|
| | | | | | | |
|[Figure 339]<br><br>[Figure 340]| |[Figure 341]<br><br>[Figure 342]|[Figure 343]<br><br>[Figure 344]|[Figure 345]<br><br>[Figure 346]|[Figure 347]<br><br>[Figure 348]|[Figure 349]<br><br>[Figure 350]|
| | | | | | | |

[Figure 351]

Same Reference

(a) Animation with different sketches

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

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

[Figure 370]

[Figure 371]

[Figure 372]

Different references (b) Animation with the same sketch

|[Figure 373]|
|---|
|[Figure 374]|

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

|[Figure 382]|
|---|
|[Figure 383]|

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

(c) Animation with sparse sketches

- Figure 7. Illustration of the flexible usage of our model. Figure (a) shows the ability of using same reference to colorize different sketches. Figure (b) demonstrates the robustness to different references. Figure (c) shows the sparse-sketch generation results.

have not been colored correctly, and the pink color of the hair has mistakenly been applied to the clothing. This miscoloring leads to poor preservation of the character’s identity. Incorporating correspondence matching ensures precise alignment between the reference and the input sketches, significantly improving color accuracy.

Effect of Background Augmentation. Without background augmentation during training, the model struggles to distinguish between the foreground (character) and the background. Consequently, it may generate frames where certain regions are incorrectly colorized as pure white or contain artifacts, due to the limited information provided by the binarized sketches. Incorporating background augmentation helps the model to better understand scene context, leading to more accurate and visually pleasing results.

### 5. Conclusion

In this paper, we introduced a novel all-in-one model for automatic lineart video colorization that integrates seamlessly with existing anime production pipelines. Our approach tackles key challenges such as misalignment between character designs, limited information in binarized sketches and situations with only sparse sketches. Comprehensive experiments demonstrate that our method outperforms stateof-the-art baselines in both quality and temporal consistency. For future work, we aim to incorporate interactive point control to handle subtle color variations and develop stronger, more efficient video models to support longer and higher-quality animation creation, further enhancing the efficiency and creativity of animation production.

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023. 2, 3, 1
- [2] Ruizhi Cao, Haoran Mo, and Chengying Gao. Line art colorization based on explicit region segmentation. In Computer Graphics Forum, pages 1–10, 2021. 2
- [3] Yu Cao, Hao Tian, and P. Y. Mok. Attention-aware anime line drawing colorization. In IEEE International Conference on Multimedia and Expo, ICME 2023, Brisbane, Australia, July 10-14, 2023, pages 1637–1642. IEEE, 2023. 2
- [4] Yu Cao, Xiangqiao Meng, PY Mok, Tong-Yee Lee, Xueting Liu, and Ping Li. Animediffusion: anime diffusion colorization. IEEE Transactions on Visualization and Computer Graphics, 2024. 3
- [5] Hernan Carrillo, Micha¨el Cl´ement, Aur´elie Bugeau, and Edgar Simo-Serra. Diffusart: Enhancing line art colorization with conditional diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3486–3490, 2023. 2
- [6] Caroline Chan, Fr´edo Durand, and Phillip Isola. Learning to generate line drawings that convey geometry and semantics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7915–7925, 2022. 1
- [7] Lei Chen and Contributors. Animegan: A fast and simple image animation method, 2024. 1
- [8] Shuhong Chen and Matthias Zwicker. Improving the perceptual quality of 2d animation interpolation. In European Conference on Computer Vision, pages 271–287. Springer,

2022. 3

- [9] Shu-Yu Chen, Jia-Qi Zhang, Lin Gao, Yue He, Shihong Xia, Min Shi, and Fang-Lue Zhang. Active colorization for cartoon line drawings. IEEE Transactions on Visualization and Computer Graphics, 28(2):1198–1208, 2020. 2
- [10] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023. 3
- [11] Yuanzheng Ci, Xinzhu Ma, Zhihui Wang, Haojie Li, and Zhongxuan Luo. User-guided deep anime line art colorization with conditional adversarial networks. In Proceedings of the 26th ACM international conference on Multimedia, 2018. 2
- [12] Duolikun Danier, Fan Zhang, and David Bull. LDMVFI: video frame interpolation with latent diffusion models. In Assoc. Adv. Artif. Intell., 2024. 3
- [13] Zhi Dou, Ning Wang, Baopu Li, Zhihui Wang, Haojie Li, and Bin Liu. Dual color space guided sketch colorization. IEEE Trans. Image Process., 30:7292–7304, 2021. 2
- [14] S´ebastien Fourey, David Tschumperl´e, and David Revoy. A fast and efficient semi-guided algorithm for flat coloring line-arts. Le Centre pour la Communication Scientifique

- Directe - HAL - Archive ouverte HAL,Le Centre pour la Communication Scientifique Directe - HAL - Archive ouverte HAL, 2018. 1
- [15] Chie Furusawa, Kazuyuki Hiroshiba, Keisuke Ogaki, and Yuri Odagiri. Comicolorization: Semi-automatic manga colorization. Cornell University - arXiv,Cornell University

- arXiv, 2017. 1

- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In Eur. Conf. Comput. Vis., pages 330–348, 2024. 3
- [17] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024. 6, 7
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [19] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. RIFE: real-time intermediate flow estimation for video frame interpolation. CoRR, abs/2011.06294,

2020. 3

- [20] Zhitong Huang, Nanxuan Zhao, and Jing Liao. Unicolor: A unified framework for multi-modal colorization with transformer. ACM Transactions on Graphics (TOG), 41(6):1–16,

2022. 1

- [21] Zhitong Huang, Mohan Zhang, and Jing Liao. Lvcd: reference-based lineart video colorization with diffusion models. arXiv preprint arXiv:2409.12960, 2024. 2, 3, 4, 6, 7
- [22] Siddhant Jain, Daniel Watson, Eric Tabellion, Aleksander Holynski, Ben Poole, and Janne Kontkanen. Video interpolation with diffusion models. CoRR, abs/2404.01203, 2024. 3
- [23] Huaizu Jiang, Deqing Sun, Varun Jampani, Ming-Hsuan Yang, Erik G. Learned-Miller, and Jan Kautz. Super slomo: High quality estimation of multiple intermediate frames for video interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2018. 3
- [24] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. arXiv preprint arXiv:2307.07635, 2023. 4, 5
- [25] Hyunsu Kim, Ho Young Jhoo, Eunhyeok Park, and Sungjoo Yoo. Tag2pix: Line art colorization using text tag with secat and changing loss. In Int. Conf. Comput. Vis., pages 9056– 9065, 2019. 2
- [26] Hyun-Su Kim, HoYoung Jhoo, Eunhyeok Park, and Sungjoo Yoo. Tag2pix: Line art colorization using text tag with secat and changing loss. Cornell University - arXiv,Cornell University - arXiv, 2019. 2
- [27] Changlin Li, Guangyang Wu, Yanan Sun, Xin Tao, ChiKeung Tang, and Yu-Wing Tai. H-VFI: hierarchical frame interpolation for videos with large motions. CoRR, abs/2211.11309, 2022. 3

- [28] Zekun Li, Zhengyang Geng, Zhao Kang, Wenyu Chen, and Yibo Yang. Eliminating gradient conflict in reference-based line-art colorization. In European Conference on Computer Vision, pages 579–596. Springer, 2022. 2
- [29] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17627–17638, 2023. 4, 5, 2
- [30] Yifan Liu, Zengchang Qin, Tao Wan, and Zhenbo Luo. Auto-painter: Cartoon image generation from sketch by using conditional wasserstein generative adversarial networks. Neurocomputing, 311:78–87, 2018. 2
- [31] David G Lowe. Distinctive image features from scaleinvariant keypoints. International journal of computer vision, 60:91–110, 2004. 4, 5, 2
- [32] Akinobu Maejima, Hiroyuki Kubo, Takuya Funatomi, Tatsuo Yotsukura, Satoshi Nakamura, and Yasuhiro Mukaigawa. Graph matching based anime colorization with multiple references. In Special Interest Group on Computer Graphics and Interactive Techniques Conference, SIGGRAPH 2019, Los Angeles, CA, USA, July 28 - August 1, 2019, Posters, pages 13:1–13:2, 2019. 2
- [33] Zhenglin Pan, Yu Zhu, and Yuxuan Mu. Sakuga-42m dataset: Scaling up cartoon research. arXiv preprint arXiv:2405.07425, 2024. 4, 5
- [34] Amal Dev Parakkat, Pooran Memari, and Marie-Paule Cani. Delaunay painting: Perceptual image colouring from raster contours with gaps. Computer Graphics Forum, page 166–181, 2022. 1
- [35] Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R Zaiane, and Martin Jagersand. U2net: Going deeper with nested u-structure for salient object detection. Pattern recognition, 106:107404, 2020. 5
- [36] Yingge Qu, Tien-Tsin Wong, and Pheng-Ann Heng. Manga colorization. ACM Transactions on Graphics, 25(3): 1214–1220, 2006. 2
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [38] Patsorn Sangkloy, Jingwan Lu, Chen Fang, Fisher Yu, and James Hays. Scribbler: Controlling deep image synthesis with sketch and color. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5400–5409, 2017. 2
- [39] Kazuhiro Sato, Yusuke Matsui, Toshihiko Yamasaki, and Kiyoharu Aizawa. Reference-based manga colorization by graph correspondence using quadratic programming. In SIGGRAPH Asia 2014 Technical Briefs, page 1–4, 2014. 2
- [40] Min Shi, Jia-Qi Zhang, Shu-Yu Chen, Lin Gao, Yu-Kun Lai, and Fang-Lue Zhang. Deep line art video colorization with a few references. arXiv preprint arXiv:2003.10685, 2020. 3
- [41] Li Siyao, Shiyu Zhao, Weijiang Yu, Wenxiu Sun, Dimitris Metaxas, Chen Change Loy, and Ziwei Liu. Deep animation video interpolation in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6587–6595, 2021. 2, 3

- [42] Zhuo Su, Wenzhe Liu, Zitong Yu, Dewen Hu, Qing Liao, Qi Tian, Matti Pietik¨ainen, and Li Liu. Pixel difference networks for efficient edge detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5117–5127, 2021. 1
- [43] Daniel S´ykora, John Dingliana, and Steven Collins. Lazybrush: Flexible painting tool for hand-drawn cartoons. Computer Graphics Forum, 28(2):599–608, 2009. 2
- [44] Daniel S´ykora, John Dingliana, and Steven Collins. Lazybrush: Flexible painting tool for hand-drawn cartoons. Computer Graphics Forum, page 599–608, 2009. 1
- [45] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems, 36:1363–1389, 2023. 5, 2
- [46] Harrish Thasarathan, Kamyar Nazeri, and Mehran Ebrahimi. Automatic temporally coherent video colorization. In 2019 16th conference on computer and robot vision (CRV), pages 189–194. IEEE, 2019. 2, 3
- [47] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [48] Ning Wang, Muyao Niu, Zhi Dou, Zhihui Wang, Zhiyong Wang, Zhaoyan Ming, Bin Liu, and Haojie Li. Coloring anime line art videos with transformation region enhancement network. Pattern Recognition, 141:109562, 2023. 3
- [49] Wen Wang, Qiuyu Wang, Kecheng Zheng, Hao Ouyang, Zhekai Chen, Biao Gong, Hao Chen, Yujun Shen, and Chunhua Shen. Framer: Interactive video interpolation. arXiv preprint https://arxiv.org/abs/2410.18978, 2024. 3
- [50] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. In European Conference on Computer Vision, pages 331–348. Springer, 2025. 5
- [51] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision, pages 1395–1403, 2015. 1
- [52] Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. arXiv preprint arXiv:2405.17933, 2024. 3, 6, 7
- [53] Dingkun Yan, Liang Yuan, Erwin Wu, Yuma Nishioka, Issei Fujishiro, and Suguru Saito. Colorizediffusion: Adjustable sketch colorization with reference image and text. arXiv preprint arXiv:2401.01456, 2024. 1
- [54] Jie Yang, Ailing Zeng, Ruimao Zhang, and Lei Zhang. Xpose: Detecting any keypoints. In European Conference on Computer Vision, pages 249–268. Springer, 2025. 5
- [55] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 6, 7

- [56] Yifeng Yu, Jiangbo Qian, Chong Wang, Yihong Dong, and Baisong Liu. Animation line art colorization based on the optical flow method. Computer Animation and Virtual Worlds, 35(1):e2229, 2024. 2, 3

- [57] Lvmin Zhang, Chengze Li, Tien-Tsin Wong, Yi Ji, and Chunping Liu. Two-stage sketch colorization. ACM Transactions on Graphics, page 1–14, 2018. 2
- [58] Lvmin Zhang, Chengze Li, Tien-Tsin Wong, Yi Ji, and Chunping Liu. Two-stage sketch colorization. ACM Trans. Graph., 37(6):1–14, 2018. 2
- [59] Lvmin Zhang, Chengze Li, Edgar Simo-Serra, Yi Ji, TienTsin Wong, and Chunping Liu. User-guided line art flat filling with split filling mechanism. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9889–9898, 2021. 2
- [60] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Int. Conf. Comput. Vis., pages 3836–3847, 2023. 2
- [61] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 4, 7
- [62] Qian Zhang, Wang Bo, Wen Wang, Hai Li, and Liu Hui. Line art correlation matching feature transfer network for automatic animation colorization. Cornell University arXiv,Cornell University - arXiv, 2020. 3
- [63] Qian Zhang, Bo Wang, Wei Wen, Hai Li, and Junhui Liu. Line art correlation matching feature transfer network for automatic animation colorization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3872–3881, 2021. 2
- [64] Xingran Zhou, Bo Zhang, Ting Zhang, Pan Zhang, Jianmin Bao, Dong Chen, Zhongfei Zhang, and Fang Wen. Cocosnet v2: Full-resolution correspondence learning for image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11465– 11475, 2021. 2
- [65] Tianyi Zhu, Wei Shang, Dongwei Ren, and Wangmeng Zuo. Thin-plate spline-based interpolation for animation line inbetweening. arXiv preprint arXiv:2408.09131, 2024. 2, 3
- [66] Changqing Zou, Haoran Mo, Chengying Gao, Ruofei Du, and Hongbo Fu. Language-based colorization of scene sketches. ACM Transactions on Graphics, page 1–16, 2019. 2
- [67] Chengyi Zou, Shuai Wan, Marc Gorriz Blanch, Luka Murn, Marta Mrak, Juil Sock, Fei Yang, and Luis Herranz. Lightweight deep exemplar colorization via semantic attention-guided laplacian pyramid. IEEE Transactions on Visualization and Computer Graphics, 2024. 1

## AniDoc: Animation Creation Made Easier Supplementary Material

Appendix

- A. Reference with Different Background

When using images of the same character with different backgrounds as references, our model can transfer the style from the reference images to generate new backgrounds with diverse styles. The original character remains consistent in their core features, such as expressions and clothing, while the integration of varied backgrounds enriches the overall visual effect, as shown in Fig. S1.

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

- Figure S1. Illustration of reference with different backgrounds.

B. Multiple Characters

Although our work primarily focuses on a single reference image and does not include specific training or processing for handling multiple references, we observe that our model can automatically distinguish between multiple characters in a reference image based on their unique features. It applies the correct coloring to each character, even when there are significant differences in poses, angles, or relative positions between the reference and the line art, as demonstrated in Fig. S2.

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

- Figure S2. Illustration of the multiple characters situation. When the reference image contains multiple characters, our method can correctly infer the correspondence and apply colorization to each character accordingly.

- C. Different Line Art Extraction Methods

To evaluate the generalization capability of our method under different line art conditions, we test its performance

using various line art extraction methods. Besides the default line art extraction method [6] used in our paper, we also apply three additional methods: Anime Lineart [7], HED [51] and PiDiNet [42]. Among these, Anime Lineart is a line art extraction method specifically trained on anime datasets. HED, as an edge detection method, produces relatively thick line art, whereas PiDiNet creates simplistic line art that is closer to hand-drawn style.

After extraction, we apply the same binarization process to the line art as described in the main text. Our method successfully colors the line art under different conditions while maintaining consistency with the reference. Due to the varying characteristics of the extracted line art, our method generates different results accordingly.

Ours Anime Lineart HED PiDiNet

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

Reference

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Figure S3. Impact of different line art extraction methods.

### D. Motivation for Correspondence Matching

[Figure 416]

Figure S4. In the early training stage (10k step), the video generation model produces static videos that closely resemble the given reference design.

As an image-to-video model, SVD (Stable Video Diffusion) [1] inherently possesses the ability to extract information from an input image to generate a video. However, during training, we observe that the strong prior in SVD restricts the first frame to be the same with the input reference image, as shown in Fig. S4.

In our formulation, the input image is not the first frame of the video but rather a reference character design from a different viewpoint. The model needs to query colors from this reference image, while the structure information should

align with the given sketch list. This conflicting prior makes training the model significantly more challenging.

colors based on the color patterns of the character’s clothing in the reference image. However, our method cannot guarantee accuracy in this situation, as shown in the 2nd row of Fig. S6.

To better establish the relationship between the reference character design and the sketch, reduce the learning difficulty for the model, and improve the fine-grained details, we propose a Correspondence Matching Module. This module explicitly injects the matching relationships between the reference image and the sketch, enabling the model to better query and color the correct areas.

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

| |
|---|

| |
|---|

| |
|---|

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

### E. Illustration of DIFT Matching

Figure S6. Limitations: in the 1st row, the cartoon bear highlighted within the red square is not present in the reference image. Consequently, the model can only infer the bear’s color as purple, based on the main color of the reference character, which deviates from its actual appearance. In the 2nd row, the character’s clothing in the line art clip is different from the reference. Therefore, our model can only infer the color of the dress and scarf based on the dominant color patterns identified in the reference image.

[Figure 425]

Figure S5. Semantic feature can effectively find matching keypoints between reference color image and binarized sketch.

During training, we apply low-level techniques LightGlue [29] with SIFT descriptor [31] for keypoint selection and matching between the reference image and the training video frames due to its fast speed. During inference, we lack access to the ground truth color image. Techniques that rely on low-level image features, such as SIFT descriptors, are ineffective at accurately matching keypoints between sketches and color reference images due to the significant domain gap between them. Therefore, we use the semantic level keypoint matching method DIFT [45] to establish the correspondence between the color reference image and the sketches, as shown in Fig. S5.

### F. Limitation

Although our method can colorize multiple clips containing the same character based on a single character design sheet while maintaining good character consistency, it still has certain limitations.

First, when a line art clip contains objects that are not present in the reference, the model struggles to determine the appropriate colors for these objects, as shown in the 1st row of Fig. S6. It can only infer colors based on the color information available in the reference, leading to inaccuracies in the colorization.

Second, when the clothing of a character in the line art clip differs from that in the reference image (even though it is the same character), our model can only infer reasonable

