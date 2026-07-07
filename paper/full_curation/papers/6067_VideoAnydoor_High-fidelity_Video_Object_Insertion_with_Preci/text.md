### VideoAnydoor: High-fidelity Video Object Insertion with Precise Motion Control

Yuanpeng Tu1,2∗ Hao Luo2,3 Xi Chen1 Sihui Ji1 Xiang Bai4 Hengshuang Zhao1,† 1The University of Hong Kong 2DAMO Academy, Alibaba Group 3Hupan Lab 4HUST https://videoanydoor.github.io

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

# arXiv:2501.01427v4[cs.CV]28May2025

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

| |
|---|

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

| |
|---|

| |
|---|

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

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

| | |
|---|---|
| | |

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

|[Figure 53]|
|---|

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

###### OursReVideo

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

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Figure 1. Demonstrations for video object insertion. VideoAnydoor preserves the fine-grained object details and enables users to control the motion with boxes or point trajectories. Based on the robust insertion, users could further add multiple objects iteratively or swap objects in the same video. Compared with the previous works, VideoAnydoor demonstrates significant superiority.

#### Abstract

tailed appearance and meanwhile support fine-grained motion control, we design a pixel warper. It takes the reference image with arbitrary key-points and the corresponding keypoint trajectories as inputs. It warps the pixel details according to the trajectories and fuses the warped features with the diffusion U-Net, thus improving detail preservation and supporting users in manipulating the motion trajectories. In addition, we propose a training strategy involving both videos and static images with a weighted loss to enhance insertion quality. VideoAnydoor demonstrates significant superiority over existing methods and naturally supports various downstream applications (e.g., video face swapping, video virtual try-on, multi-region editing) without task-specific fine-tuning.

Despite significant advancements in video generation, inserting a given object into videos remains a challenging task. The difficulty lies in preserving the appearance details of the reference object and accurately modeling coherent motion at the same time. In this paper, we propose VideoAnydoor, a zero-shot video object insertion framework with high-fidelity detail preservation and precise motion control. Starting from a text-to-video model, we utilize an ID extractor to inject the global identity and leverage a box sequence to control the overall motion. To preserve the de-

*Work during DAMO Academy internship. † Corresponding author.

#### 1. Introduction

The booming of diffusion models [19, 30, 35] has spurred significant advancements in text-to-video generation [1, 12, 47] and editing [13, 21, 36, 46]. Some works [14, 37, 38, 53] learn to edit the video based on posture [14, 53] or styles [20] while other works [10, 17] explore modifying specific objects based on text descriptions.

In this paper, we investigate video object insertion, which means seamlessly placing a specific object (with a reference image) into a given video with the desired motion and location. This ability has broad potential for real-world applications, like video composition, video virtual try-on, video face changing, etc.

Although strongly in need, this topic remains underexplored by existing works. We analyze that the challenge of video object insertion mainly lies in two folds: accurate ID preservation and precise motion control. Recently, some works have made initial attempts in this field. AnyV2V [17] and ReVideo [24] leverage image composition model [4] to insert the object in the first frame. Then, they propagate this modification to subsequent frames are under the guidance of text or trajectory control. However, this two-stage scheme may lead to suboptimal results if the first frame insertion is not satisfactory. Besides, as they do not inject ID information in the following frames, the object’s identity and motion tend to collapse in the later frames.

Faced with this challenge, we attempt to accurately preserve the object’s identity and precisely control the object’s motion throughout the whole video. Specifically, we propose an end-to-end framework termed VideoAnydoor. Starting from a text-to-video diffusion model, the concatenation of random noise, object masks, and the masked video is utilized as input. Meanwhile, the reference image with no background is fed into the ID extractor to extract compact and discriminative ID tokens. Then these ID tokens are injected into the diffusion model together with the box sequence as coarse guidance of identity and motion to generate the desired composition. Additionally, a pixel warper module is designed for joint modeling of the fine-grained appearance and precise motion. It takes the reference image with arbitrary key-points and the corresponding keypoint trajectories to warp the pixel details into the target regions according to the desired motion and pose. Moreover, to address the scarcity of high-quality videos, we manually augment extensive high-quality image data as videos to improve the fine-grained alignment of appearance details. As shown in Fig. 1, with these techniques, users can edit specific regions in the video by providing target images, drawing boxes and trajectory lines. It should be noted that our inserted object is not constrained by shape, appearance or the range of the given movement, demonstrating great robustness and generality in diverse scenarios.

Our contributions can be summarized as follows:

- • We construct the first end-to-end video object insertion framework that supports both motion and content editing. Our framework seamlessly supports diverse applications, e.g., multi-region editing, video virtual try-on, and video face changing, etc.
- • We propose pixel warper to warp the pixel details according to the desired motion. It takes the reference image with arbitrary key-points and the trajectories as inputs for fine-grained modeling of identity and motion, enabling accurate ID preservation and motion control.
- • We design multiple strategies to further enhance the capability of accurate insertion, including image-video mix training, training trajectory filtering. Extensive experiments demonstrate their effectiveness in precise ID preservation and motion control.

#### 2. Related Work

Image-level object insertion. Generative object compositing [4, 32, 33, 43, 43, 50] focus on implanting subjects in diverse contexts. Among these methods, Paintby-Example [43] proposes an information bottleneck to avoid the trivial solution. CustomNet [49] incorporates 3D novel view synthesis capabilities. IMPRINT [33] decouples learning of identity preservation from that of compositing. AnyDoor [4] utilizes a frequency-aware detail extractor to obtain detail maps. However, directly transferring similar insertion schemes as these to videos may result in imperfect performance as they fail to preserve fine-grained appearance details, while the quality of object insertion in videos is crucial for precise motion control. Nevertheless, these methods generally fail to insert objects with proper postures for motion control. Thus, to address these two issues, we conduct a detailed investigation on object insertion with accurate ID preservation and proper posture control.

Video editing. Early methods [2, 9, 26, 40] primarily adopt training-free or one-shot tuning schemes owing to the lack of proper training data. For example, Pix2Video [2] first edits the first frame and then produces followed frames with cross-frame attention. Recently, tuning-based methods [10, 17, 24, 41] have exhibited better results. Among them, text-prompt based schemes struggle to locate target regions. AnyV2V [17] uses an off-the-shelf image editing model to modify the first frame. Image-prompt based methods like ReVideo [24] design a three-stage training scheme that decouples content and motion control. VideoSwap [10] uses semantic points to achieve video subject replacement. However, these methods either require extra fine-tuning, fail to keep the unedited region unchanged, or achieve poor motion/identity consistency with a two-stage scheme. To address these issues, we aim to design an end-to-end zeroshot video insertion framework that precisely modifies both content and motion according to user-provided instructions while keeping the unedited content unchanged in zero-shot.

| |[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]| |
|---|---|---|
| | | |

ID Extractor

[Figure 130]

[Figure 131]

Key-point BackgroundTrajectoryReferenceMaskNoise

[Figure 132]

Cross-Atten.

[Figure 133]

|[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>| |
|---|---|
| | |

Weighted Loss

###### 3D U-Net

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

[Figure 152]

[Figure 153]

[Figure 154]

Temporal

Temporal

Temporal

Temporal

Spatial

## …

Spatial

Spatial

Spatial

Conv.

Conv.

Conv.

Conv.

|[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>Real Videos Sim. Videos<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>+<br><br>TrainingData|
|---|

| | | |
|---|---|---|
| | | |
||[Figure 161]<br><br>[Figure 162]|
|---|
| | |

Layer-wise Add

[Figure 163]

[Figure 164]

Motion Encoder

Fusion

ControlNet

Cross-Atten.

Reference+

[Figure 165]

Content Encoder

Pixel Warper

- Figure 2. The pipelines of our VideoAnydoor. First, we input the concatenation of the original video, object masks, and masked video into the 3D U-Net. Meanwhile, the background-removed reference image is fed into the ID extractor, and the obtained features are injected into the 3D U-Net. In our pixel warper, the reference image marked with key points and the trajectories are utilized as inputs for the content and motion encoders. Then, the extracted embeddings are input into cross-attentions for further fusion. The fused results serve as the input of a ControlNet, which extracts multi-scale features for fine-grained injection of motion and identity. The framework is trained with weighted losses. We use a blend of real videos and image-simulated videos for training to compensate for the data scarcity.

#### 3. Method

motion guidance. Before feeding the reference image into the extractor, we remove its background with a segmentor [16] and align the object to the image center to retain compact and ID-related representations. For fine-grained control, we adopt the interaction-friendly trajectory lines as the control signals and propose a pixel warper to warp the pixel details according to the desired motion for joint modeling of appearance details and precise motion. Finally, a weighted loss is used amplify the influence of key-points and design a novel image-video mix-training strategy to address the scarcity of high-quality video data.

##### 3.1. Overview of Framework

Task formulation. In this paper, we focus on high-fidelity video object insertion, with the goal of subject insertion with user-provided trajectories, where the unedited regions should remain the same as the source video. The primary challenge of this task lies in aligning the motion trajectory of the given one while preserving the identity of the target concept, particularly its appearance details.

Overall pipeline. The VideoAnydoor pipeline is illustrated in Fig. 2. To reconstruct the background within the masked region, we build our method on a 2D in-painting diffusion model. Following the latent diffusion model [31], we encode both the source video and the masked video with a VAE encoder to obtain the latent space representations zori and zmask. The corresponding masks are 8 times downsampled as the mask latent. Subsequently, DDIM inversion [5] is applied to transform the clean latent zori back to the noisy latent zT. Then we concatenate zT, zmask and zmask as a 9-channel tensor for 3D U-Net. To utilize the priors of video generation, we integrate the motion layers [11] into the in-painting model as the 3D U-Net to ensure essential temporal consistency. For coarse-grained control, we leverage the powerful visual encoder DINOv2 [25] as the ID extractor for ID preservation and use the bounding boxes as

For convenience, the trajectory map and correspondence reference image are denoted as cmot ∈ RN×3×H×W and cref−key ∈ R3×H×W respectively. V ∈ RN×3×H×W, A ∈ RN×1×H×W and cref ∈ R3×H×W represent the original video, masks of the edited region and the reference image respectively. N,H,W is the frame number, height, and width of the original video. The content and motion encoders are denoted as Ec and Em respectively.

Inference configuration. For users, they only need to provide a subject image, a source video, and a trajectory sequence. For the trajectory sequence, the users can directly use the trajectory of the object within the source video or just draw a start box and an end box to flexibly generate edited videos precisely aligned with the given conditions.

Table 1. Statistics of datasets used for training our VideoAnydoor. “quality” particularly refers to the image resolution.

[Figure 166]

[Figure 167]

[Figure 168]

CoTracker

Dataset Type # Samples Mask Quality Video Quality

YouTubeVOS [45] Video 4,453 High Low YouTubeVIS [45] Video 2,883 High Low UVO [7] Video 10,337 High Low MOSE [6] Video 2,149 High High VIPSeg [22] Video 3,110 High High VSPW [23] Video 3,536 High High SAM2 [28] Video 51,000 High High Pexel Video 6,000 Medium High MVImgNet [48] Video 219,188 High High ViViD [8] Video 9,700 High High CHDTF [54] Video 362 High High CelebV-HQ [55] Video 35,666 High High Pexel Image 95,000 Medium High

Select points with larger motion

X-Pose/Grid

[Figure 169]

[Figure 170]

NMS

- Figure 3. Pipeline of trajectory generation for training data. We first perform NMS to filter out densely-distributed points and then select points with larger motion. The retained ones can be sparsely distributed in each part of the target and contain more motion information, thus inducing more precise control.

Then these features are added to the corresponding layers for fine-grained modeling of appearance details and precise motion. This procedure can be formulated as:

##### 3.2. Pixel Warper

Trajectory sampling. During training, it is essential to extract trajectories from videos to provide motion conditions. Previous works [44] show that the movement of objects can be controlled by general key-points. Thus, as shown in Fig. 3, we first input the first frame to X-Pose [44] to initialize the points for subsequent trajectory generation. For cases in which X-Pose fails to detect any key-points, we use a grid to sparsify dense sampling points. We empirically find that points with larger motion are more helpful for trajectory control. However, these points are mostly densely distributed in certain regions, resulting in severe information redundancy. Therefore, to filter out the undesired points, we first perform non-maximum suppression (NMS) to filter out points that are densely distributed. Then we apply motion tracking on each point to obtain their path lengths, e.g., {l0,l1,...,lN

yc = F(zt,t,cref;Θ) + Z(F(zt + Z(fc),t,cref;Θc)),

(1) where yc denotes the new diffusion features. Z represents the function of zero-conv [52]. Θc and Θ are the parameters of the ControlNet and the diffusion model.

Weighted loss. Inspired by [39], we try to adopt a reweight scheme to improve the fine-grained modeling of identity and motion. Specifically, we solely perform reweighting on the region surrounding the trajectory. For trajectories with relatively large motion amplitudes, we apply greater weights to achieve more accurate motion control. Denote the ratio of the region that i-th trajectory covers as Ri, we perform 8 times down-sampling on the region, which is denoted as Aitrj. The weighted loss can be formulated as:

init−1}, where Ninit denotes the number of initial points. Then we retain N points with the largest motion and use the corresponding trajectory map cmot as control signals. Different colors are assigned for N points to represent different trajectories.

N

((λRiAitrj + (1 − Aitrj)/N) · ∥δ − δ∗∥22), (2)

L =

i=1

where λ denotes the balancing loss weight. δ∗,δ are the prediction of the 3D U-Net and the target.

Motion injection. A naive implementation of motion injection is only training a similar control module to inject the motion conditions as [10, 51]. However, such a scheme may fail to accurately insert the objects with desired motion and appearance details, since it has no explicit semantic correspondence with the reference object. Thus the object may be inserted into the video with an undesired pose, leading to severe distortion in foreground regions. To address this issue, we input a pair of trajectory maps cmot and correspondence reference image cref−key as fine-grained guidance. As show in Fig. 2, cmot/cref−key are first encoded by Ec/Em respectively. Then these two embeddings are input into two cross-attention modules respectively for semantic-aware fusion. Afterward, the fused two features are added and utilized as the input of a ControlNet [51] to extract multi-scale intermediate features {fc0,fc0,...,fcP}, where P denotes the layer number of the diffusion model.

##### 3.3. Training Strategy

Dataset preparation. The ideal training samples are video pairs for “different objects in the same scene”, which are hard to collect with existing datasets. As alternatives, we sample all the needed data from the same video. Specifically, for a video, we pick a video clip and a frame that has the largest distance from the clip, which is assumed to contain the most dissimilar object from the one in the video clip. We take their masks for the foreground objects and remove the background for the select frame. Then we crop it around the mask as the target object. For the video clip, we generate the box sequence and remove the box region to get the scene video, where the unmasked video could be used as the training ground truth. Specifically, we use the expanded bounding box rather than the tightly-surrounded one in implementation. For boxes with a small moving range, we use

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

OursReVideoAnyV2VOriginal

| |
|---|

[Figure 179]

| |
|---|

[Figure 180]

| |
|---|

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

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

OursReVideoAnyV2VOriginal

| |
|---|

| |
|---|

|[Figure 221]|
|---|

| |
|---|

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

[Figure 271]

- Figure 4. Comparison results between VideoAnydoor and existing state-of-the-art video editing works. Our VideoAnydoor can achieve superior performance on precise control of both motion and content.

#### 4. Experiments

the union of the boxes as the final box to reduce the impact of the bounding box on the motion. The full data for training is shown in Tab. 1, which covers both videos from diverse domains and high-quality images to compensate for the scarcity of high-quality videos.

##### 4.1. Experimental Setup

Implementation details. In this work, we choose Stable Diffusion XL with motion modules as the base generator. During training, we process the image resolution to 512×512 and adopt Adam optimizer with an initial learning rate of 1e−5. We use DDIM for 50-step sampling and classifier-free guidance with a cfg of 10.0 for inference. The model is optimized for 120K iterations on 16 NVIDIA A100 GPUs with a batch size of 32. We only use 8 points for trajectory generation of each sample. In actual use, these parameters can be adjusted by the user according to different subjects and the desired generation effect.

Image-video mixed training. Different from previous works [3] that utilize high-quality images for two-stage disentangled training, we resort to employing them for joint training with videos. However, directly repeating images will impair the discriminative learning of temporal modules. Instead, we augment the images as videos with manual camera operation. Specifically, we either randomly translate the image at equal intervals from different directions, or gradually crop the original image at equal intervals to obtain an image sequence. Then the image sequence is processed with bilinear interpolation to enhance the video smoothness. Although the augmented videos would benefit the learning of appearance variation, the essential difference between them and real videos will potentially impair motion learning. Thus similar to [4], we adopt adaptive timestep sampling to enable different modalities of data to contribute to different stages of denoising training.

Benchmarks. For comprehensive evaluation, we construct a benchmark consisting of around 200 videos collected from Pexel, which includes ten different categories (e.g., persons, dogs). We also make qualitative analysis on the ViViD [8] and CHDTF [54] test set to evaluate the performance for virtual video try-on and talking head generation.

Evaluation metrics. On our constructed benchmarks, for

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

[Figure 295]

[Figure 296]

[Figure 297]

- Figure 5. Demonstrations for precise motion control. VideoAnydoor can achieve precise alignment with the given trajectories and objects when using a pair of reference images marked with key-points and corresponding trajectory maps as input.

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

| |
|---|

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

| |
|---|

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

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

| |
|---|

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

| |
|---|

[Figure 346]

[Figure 347]

- Figure 6. More visual examples of VideoAnydoor. It preserves fine-grained details (e.g., logos on the car) and achieves smooth motion control (e.g., the tail of the cat) with our pixel warper.

Table 2. Quantitative comparison between our VideoAnydoor and other related work. Six automatic metrics are employed for the performance evaluation of both content and motion. VideoAnydoor outperforms these methods across all the metrics.

PSNR (↑) CLIP-Score (↑) DINO-Score (↑) AJ (↑) δavgvis (↑) OA (↑)

ConsistI2V [29] 25.1 64.7 40.6 49.3 51.1 57.2 I2VAdapter [42] 24.3 67.1 42.2 51.2 53.7 59.9 AnyV2V [17] 30.1 70.2 47.2 54.1 55.8 61.1 ReVideo [24] 33.5 74.2 51.7 79.2 81.4 83.2 VideoAnydoor (ours) 38.0 81.4 59.1 88.3 91.5 92.5

the edited and unedited areas. Moreover, it has poor motion consistency due to using texts as control signals. For ReVideo, there exists an obvious loss of edited content as well, especially for the cases with large motion. It exhibits inferior pose control over the inserted object owing to the lack of semantic information within motion signals. In comparison, our VideoAnydoor can effectively preserve the unedited content while allowing users to customize the motion in editing areas. We provide more examples in Fig. 5 and Fig. 6, where we insert the same object with different trajectories and different objects in diverse scenarios.

evaluation of ID preservation, we calculate both CLIPScore [27] and DINO-Score [25] to reflect the similarity between the edited region and the target subject, where PSNR [34] is employed to measure the reconstruction quality of unedited regions as well. Additionally, for customized concepts, we follow Custom Diffusion [18] to compute pairwise image alignment between each edited frame and each reference concept image. In addition, we feed the edited videos to Cotracker [15] to calculate the tracking metrics [15] with the points in original videos as ground truth labels. Finally, we organize user studies with a group of 15 annotators to rate the edited results from the perspective of quality, fidelity, fluidity of movement, and diversity.

##### 4.3. Quantitative Comparison

ID preservation. We first conduct quantitative evaluation with CLIP-Score [27], DINO-Score [25], PSNR [34]. Previous approaches impose heavy reliance on the existing image customization methods to acquire the first frame, making them retain the distortions within the first frame for subsequent generation. Thus it can be observed in Tab. 2 that they generally achieve inferior results to our method. Moreover, since there is no explicit condition for AnyV2V [17], ConsistI2V [29], I2VAdapter [42] to keep the unedited regions unchanged, these methods perform much worse than our method and ReVideo on PSNR. Overall, our method achieves a clear advantage over the compared methods across all the metrics.

##### 4.2. Qualitative Comparison

Among existing methods, ReVideo [24] heavily relies on the first frame modified by existing image customization methods and it adopts a semantic-unaware motion injection manner. AnyV2V [17] adopts a similar two-stage generation scheme with text prompts. In Fig. 4, we can observe that AnyV2V [17] suffers from content distortions for both

Motion consistency. We conduct further quantitative experiments with the metric in tracking tasks to evaluate the motion alignment. Specifically, we track the key-points in

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

w/oPixel-

[Figure 359]

Reweight

[Figure 360]

Warper FrozenOurs-full

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

w/o

[Figure 366]

[Figure 367]

| |
|---|

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

Static-image Only

[Figure 378]

[Figure 379]

Dinov2

Only

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

| |
|---|

[Figure 388]

[Figure 389]

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

Real-video

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

- Figure 7. Qualitative ablation studies on the core components of VideoAnydoor. When removing the pixel warper, it suffers from poor motion consistency due to the undesired posture. And it can be observed that all the components contribute to the best performance.

- Table 3. User study on the comparison between our VideoAnydoor and existing alternatives. “Quality”, “Fidelity”, “Smooth”, and “Diversity” measure synthesis quality, object identity preservation, motion consistency, and object local variation, respectively. Each metric is rated from 1 (worst) to 4 (best).

Quality (↑) Fidelity (↑) Smooth (↑) Diversity (↑)

ConsistI2V [10] 1.80 1.75 2.30 1.50 AnyV2V [17] 1.90 1.85 1.50 2.10 ReVideo [24] 2.65 2.55 2.50 2.25 VideoAnydoor (ours) 3.75 3.80 3.65 3.70

- Table 4. Quantitative evaluation of core components in VideoAnydoor on ID preservation. † denotes removing the semantic points in the key-point image.

PSNR (↑) CLIP-Score (↑) DINO-Score (↑)

Only Real-video Data 34.6 75.0 51.7 Only Static-image Data 33.9 73.8 51.2 FrozenDINOv2 33.5 74.3 51.6 w/o PixelWarper† 35.3 77.4 53.0 w/o Pixel Warper 33.8 72.4 48.5 w/o Weighted Loss 35.1 77.0 53.1 w/ Box Loss [39] 37.7 81.1 58.9 Ours-full 38.0 81.4 59.1

Table 5. Quantitative evaluation of core components in VideoAnydoor on motion consistency. † denotes removing the semantic points in the key-point image.

AJ (↑) δavgvis (↑) OA (↑)

Only Real-video Data 66.5 67.5 69.9 Only Static-image Data 71.4 72.0 74.3 FrozenDINOv2 80.0 82.4 85.3 w/o PixelWarper† 80.3 81.3 84.4 w/o PixelWarper 78.5 81.7 83.7 w/o Weighted Loss 75.4 84.2 85.1 w/ Box Loss [39] 88.0 91.1 92.3 Ours-full 88.3 91.5 92.5

four views: “Quality”, “Smooth”, “Fidelity”, “Diversity”. “Fidelity” measures ID preservation, and “Quality” counts for whether the result is harmonized without considering fidelity. “Smooth” assesses the motion consistency. We use “Diversity” to measure the differences among the synthesized results. The user-study results are shown in Tab. 3. It can be noted that our model demonstrates significant superiority, especially for “Fidelity”, and “Smooth”. Such results fully verify the effectiveness of our method.

the first frame of the original video in the edited one with the Cotracker model [15] and adopt the original trajectories as ground truths. The results are summarized in Tab. 2. Due to the lack of explicit motion control, AnyV2V, I2VAdapter, and ConsistI2V usually generate static or distorted motion for the edited content. Compared with them, VideoAnydoor demonstrates the best performance. Besides these results, we provide evaluation from aesthetic and technical views in the Appendix as well.

User study. We organize a user study to compare ConsistI2V [29], ReVideo [24], AnyV2V [17], and our VideoAnydoor. Specifically, we let 20 annotators rate 20 groups of videos, where each group contains the original video and four edited videos. For each group, we provide one image edited by AnyDoor [4] as the first frame for the compared methods. Besides, we provide detailed regulations to rate the generated videos for scores of 1 to 4 from

##### 4.4. Ablation Studies

ID preservation. We conduct an investigation of the core components on ID preservation. From Tab. 4, we can observe that training with fixed DINOv2 induces much inferior performance. Training only on videos suffers from severe accuracy degradation across all the metrics. Moreover, our pixel warper can effectively help inject the appearance details according to the motion. The results also show that the weighted loss is superior to the box loss [39] and can bring a performance boost to the baseline by making the model focus on the foreground regions. The best performance can be achieved by training with all the modules.

Motion consistency. We present evaluation outcomes of the motion control for core components in Tab. 5. Results show that training with static images causes a significant accuracy drop due to impaired temporal module learning and

[Figure 408]

[Figure 409]

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

- Figure 8. More applications of VideoAnydoor. Our framework seamlessly supports various tasks like video virtual try-on, talk head generation, multi-region editing, etc. The results show that VideoAnydoor could effectively preserve the structure and identity and impose precise control on movements of multiple objects in diverse scenarios.

Table 6. Detailed quantitative evaluation of the pixel warper in VideoAnydoor on motion consistency. “Tight box” denotes training with tightly-surrounded boxes.

of face identities. Specifically, we conduct evaluations on the CHDTF dataset [54] and use 16 points with the largest movement in the face landmarks as the initial key-points. As demonstrated in Fig. 8, VideoAnydoor could give satisfactory performance for this task as well.

AJ (↑) δavgvis (↑) OA (↑)

Random X-Pose points 80.6 82.4 83.0 Grid points 82.4 83.3 85.0 w/o NMS 82.2 83.0 84.3 Tight box 83.4 85.6 86.3 Ours-full 88.3 91.5 92.5

Multi-region editing. In addition, we extend our VideoAnydoor to multi-region editing. As shown in Fig. 8, we can achieve precise control of multi-object insertion for both motion and content. Besides, it can be used for object swapping and inserting logos or ornaments as well. As shown in Fig. 8, we can precisely place the hat on the head and achieve smooth posture control.

inferior motion consistency when only using key-point trajectories in the pixel warper. Training with image-simulated videos aids precise motion control, likely due to facilitating fine-grained appearance reconstruction and making interframe key-point correspondence easier. Beyond this, we conduct more comparisons with different variants of pixel warper in Tab. 6. It can be observed that selecting points with larger motion gives a significant performance boost. Moreover, using all grid-sampled points is inferior to extracted key-points. Selecting samples without considering distance leads to inferior performance as points are densely distributed in certain regions as well. Training with looselysurrounded boxes leads to precise motion control for keypoint trajectories. Qualitative comparisons in Fig. 7 show a similar phenomenon to quantitative results. All modules contribute to the best performance.

#### 5. Conclusion

In this paper, we present VideoAnydoor for end-to-end video object insertion with precise motion control. Specifically, it can effectively characterize the reference target with an ID extractor when trained with a combination of videos and high-quality images. Moreover, it can achieve smooth motion consistency and effective preservation of appearance details through the proposed pixel warper. Our VideoAnydoor has promising performance on both video object insertion task and diverse precise video editing applications, e.g., (1) object insertion, (2) virtual video tryon, (3) video face swapping, and (4) multi-region editing. Extensive qualitative and quantitative experimental results demonstrate its superiority over previous methods on the alignment of both motion and identity with the given control signals. It provides a universal solution for general regionto-region mapping tasks as well.

##### 4.5. More Applications

Virtual video try-on. As shown in Fig. 8, without extra task-specific tuning, our VideoAnydoor demonstrates satisfactory performance for virtual try-on on the ViViD dataset [8], where diverse patterns of the target clothes can be well preserved across different frames. Such results underscore the strong generalization abilities of our method.

Limitations. Despite impressive results, our method still struggles with complex logos. This issue might be solved by collecting related data or using stronger backbones.

Talking head generation. Besides video try-on, we further apply our method to talking head generation, which requires more precise control of tiny movements and preservation

Acknowledgements. This work was supported by DAMO Academy via DAMO Academy Research Intern Program.

#### References

- [1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. Report, 2024. 2
- [2] Duygu Ceylan, Chun-Hao Huang, and Niloy J. Mitra. Pix2video: Video editing using image diffusion. In ICCV,

2023. 2

- [3] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024. 5
- [4] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In CVPR, 2024. 2, 5, 7
- [5] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 3
- [6] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. MOSE: A new dataset for video object segmentation in complex scenes. In ICCV,

2023. 4

- [7] Yuming Du, Wen Guo, Yang Xiao, and Vincent Lepetit. 1st place solution for the uvo challenge on video-based openworld segmentation 2021. arXiv:2110.11661, 2021. 4
- [8] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and ZhengJun Zha. Vivid: Video virtual try-on using diffusion models. arXiv: 2405.11794, 2024. 4, 5, 8
- [9] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In ICLR, 2024. 2
- [10] Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao, Jay Zhangjie Wu, David Junhao Zhang, Mike Zheng Shou, and Kevin Tang. Videoswap: Customized video subject swapping with interactive semantic point correspondence. In CVPR, 2024. 2, 4, 7
- [11] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. In ICLR,

2024. 3

- [12] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv:2205.15868, 2022. 2
- [13] Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zeroshot grounded video editing using text-to-image diffusion models. In ICLR, 2024. 2
- [14] Johanna, Aleksander Karras, Ting-Chun Holynski, Ira Wang, and Kemelmacher-Shlizerman. Dreampose: Fashion imageto-video synthesis via stable diffusion. arXiv:2304.06025,

2023. 2

- [15] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. arXiv:2307.07635,

2023. 6, 7

- [16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and

- Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 3
- [17] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-tovideo editing tasks. arXiv:2403.14468, 2024. 2, 6, 7
- [18] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 6
- [19] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

2023. 2

- [20] Gongye Liu, Menghan Xia, Yong Zhang, Haoxin Chen, Jinbo Xing, Xintao Wang, Yujiu Yang, and Ying Shan. Stylecrafter: Enhancing stylized text-to-video generation with style adapter. arXiv:2312.00330, 2023. 2
- [21] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In CVPR, 2024. 2
- [22] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-scale video panoptic segmentation in the wild: A benchmark. In CVPR, 2022. 4
- [23] Jiaxu Miao, Yunchao Wei, Yu Wu, Chen Liang, Guangrui Li, and Yi Yang. Vspw: A large-scale dataset for video scene parsing in the wild. In CVPR, 2021. 4
- [24] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. arXiv:2405.13865,

2024. 2, 6, 7

- [25] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. TMLR, 2023. 3, 6
- [26] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. In CVPR, 2024. 2
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [28] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv:2408.00714, 2024. 4
- [29] Weiming Ren, Harry Yang, Ge Zhang, Cong Wei, Xinrun Du, Stephen Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-to-video generation. TMLR,

2024. 6, 7

- [30] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [31] Jaskirat Singh, Stephen Gould, and Liang Zheng. Highfidelity guided image synthesis with latent diffusion models. In CVPR, 2023. 3
- [32] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian L. Price, Jianming Zhang, Soo Ye Kim, and Daniel G. Aliaga. Objectstitch: Generative object compositing. arXiv.2212.00932, 2022. 2
- [33] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian L. Price, Jianming Zhang, Soo Ye Kim, He Zhang, Wei Xiong, and Daniel G. Aliaga. IMPRINT: generative object compositing by learning identity-preserving representation. In CVPR, 2024. 2
- [34] Yule Sun, Ang Lu, and Lu Yu. Weighted-to-sphericallyuniform quality evaluation for omnidirectional video. Signal Processing Letters, 2017. 6
- [35] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint,

2024. 2

- [36] Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. Motioneditor: Editing video motion via content-aware diffusion. In CVPR, 2024. 2
- [37] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. In NeurIPS, 2024. 2
- [38] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In CVPR, 2024. 2
- [39] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, Yingya Zhang, and Hongming Shan. Dreamvideo-2: Zero-shot subject-driven video customization with precise motion control. arXiv:2410.13830, 2024. 4, 7
- [40] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 2
- [41] Tao Wu, Yong Zhang, Xintao Wang, Xianpan Zhou, Guangcong Zheng, Zhongang Qi, Ying Shan, and Xi Li. Customcrafter: Customized video generation with preserving motion and concept composition abilities. arXiv:2408.13239,

2024. 2

- [42] Xun, Mingwu Guo, Liang Zheng, Yuan Hou, Yufan Gao, Pengfei Deng, Di Wan, Yufan Zhang, Weiming Liu, Zhengjun Hu, Haibin Zha, Chongyang Huang, and Ma. I2v-adapter: A general image-to-video adapter for diffusion models. In SIGGRAPH, 2024. 6
- [43] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion mod-

- els. arXiv:2211.13227, 2022. 2
- [44] Jie Yang, Ailing Zeng, Ruimao Zhang, and Lei Zhang. Xpose: Detection any keypoints. In ECCV, 2024. 4
- [45] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In ICCV, 2019. 4
- [46] Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang. EVA: zero-shot accurate attributes and multi-object video editing. arXiv: 2403.16111, 2024. 2
- [47] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv:2408.06072,

2024. 2

- [48] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. Mvimgnet: A large-scale dataset of multi-view images. In CVPR, 2023. 4
- [49] Ziyang Yuan, Mingdeng Cao, Xintao Wang, Zhongang Qi, Chun Yuan, and Ying Shan. Customnet: Object customization with variable-viewpoints in text-to-image diffusion models. In ACMMM, 2024. 2
- [50] Bo Zhang, Yuxuan Duan, Jun Lan, Yan Hong, Huijia Zhu, Weiqiang Wang, and Li Niu. Controlcom: Controllable image composition using diffusion model. arXiv.2308.10040,

2023. 2

- [51] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 4
- [52] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 4
- [53] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv:2406.19680, 2024. 2
- [54] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a highresolution audio-visual dataset. In CVPR, 2021. 4, 5, 8
- [55] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. CelebVHQ: A large-scale video facial attributes dataset. In ECCV,

2022. 4

