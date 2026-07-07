# arXiv:2410.18978v2[cs.CV]4Nov2024

[Figure 1]

## FRAMER: INTERACTIVE FRAME INTERPOLATION

Wen Wang1,2, Qiuyu Wang2, Kecheng Zheng2, Hao Ouyang2, Zhekai Chen1, Biao Gong2, Hao Chen1, Yujun Shen2, Chunhua Shen1

1Zhejiang University 2Ant Group

Start Frame Generated Frames End Frame

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

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

(a) Same input image pair with varying drags

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

| |
|---|

| |
|---|

|[Figure 60]<br><br>[Figure 61]|
|---|

(b) Image morphing (inputs highlighted by dashed boxes)

Figure 1: Showcases produced by our Framer. It facilitates fine-grained customization of local motions and generates varying interpolation results given the same input start and end frame pair (first 3 rows). Moreover, Framer handles challenging cases and can realize smooth image morphing (last 2 rows). The input trajectories are overlayed on the frames.

ABSTRACT

We propose Framer for interactive frame interpolation, which targets producing smoothly transitioning frames between two images as per user creativity. Concretely, besides taking the start and end frames as inputs, our approach supports customizing the transition process by tailoring the trajectory of some selected keypoints. Such a design enjoys two clear benefits. First, incorporating human interaction mitigates the issue arising from numerous possibilities of transforming one image to another, and in turn enables finer control of local motions. Second, as the most basic form of interaction, keypoints help establish the correspondence across frames, enhancing the model to handle challenging cases (e.g., objects on the start and end frames are of different shapes and styles). It is noteworthy that our system also offers an “autopilot” mode, where we introduce a module to estimate the keypoints and refine the trajectory automatically, to simplify the usage in practice. Extensive experimental results demonstrate the appealing performance of Framer on various applications, such as image morphing, timelapse video generation, cartoon interpolation, etc. The code, the model, and the interface will be released to facilitate further research.

Project page: aim-uofa.github.io/Framer

- 1 INTRODUCTION

The creation of seamless and visually appealing transitions between frames (Dong et al., 2023) is a fundamental requirement in various applications, including image morphing (Aloraibi, 2023), slowmotion video generation (Reda et al., 2022), and cartoon interpolation (Xing et al., 2024). Users often need to control the motion trajectories, deformation dynamics, and temporal coherence of interpolated frames to achieve specific outcomes. Therefore, incorporating interactive capabilities into frame interpolation frameworks is crucial for expanding the practical applicability.

Traditional video frame interpolation methods (Jiang et al., 2018; Xu et al., 2019; Liu et al., 2020; Niklaus & Liu, 2020; Sim et al., 2021; Lee et al., 2020; Ding et al., 2021) often rely on estimating optical flow or motion to predict intermediate frames deterministically. While significant progress has been made in this area, these approaches struggle in scenarios involving large motion or substantial changes in object appearance, due to an inaccurate flow estimation. What’s more, when transforming one image to another, there can be numerous plausible ways objects and scenes can transition. A deterministic result may not align with user expectations or creative intent.

Orthogonal to existing methods, we propose Framer, an interactive frame interpolation framework designed to produce smoothly transitioning frames between two images. Our approach allows users to customize the transition process by tailoring the trajectories of selected keypoints, thus directly influencing the motion and deformation of objects within the scene. Such design offers two significant benefits. First, the incorporation of keypoint-based interaction resolves the ambiguity inherent in transforming one image into another, allowing for precise control over how specific regions of the image move and change. As shown in Fig. 1a, users can control the movements of the dog’s paw and head through simple and intuitive interactions. Second, keypoint trajectories establish explicit correspondences across frames, which is especially beneficial in challenging cases where objects change in shape, style, or even semantic meaning. As shown in Fig. 1b, the keypoint trajectories establish the correspondences between keypoints from Pok´emon in varying forms and help produce a smooth “evolution” process of Pok´emon.

Concretely, we view video frame interpolation from a generative perspective and finetune a large-scale pre-trained image-to-video diffusion model (Blattmann et al., 2023a) on open-domain video datasets (Nan et al., 2024) to facilitate video frame interpolation. The additional lastframe conditioning is introduced during the fine-tuning process. Afterward, a point trajectory controlling branch is introduced to take the additional point trajectory inputs, thus guiding the video interpolation process. During inference, Framer supports the “interactive” mode for customized video frame interpolation, following user-input point trajectories.

Understanding that manual keypoint annotation may not always be desirable, we offer an “autopilot” mode for Framer. Technically, we propose a novel bi-directional point-tracking method that estimates the trajectories of matched points over the entire video sequence, by analyzing both forward and backward motions between frames. It automates the process of obtaining keypoint trajectories, enabling Framer to generate motion-natural and temporally coherent interpolation results without requiring extensive user input. The “autopilot” mode simplifies the workflow while still benefiting from the enhanced correspondence provided by the points trajectories.

We conduct extensive experiments to evaluate the performance of Framer across various applications, including image morphing, time-lapse video generation, and cartoon interpolation. The results demonstrate that Framer produces smooth and visually appealing transitions, outperforming existing methods, particularly in cases involving complex motions and significant appearance changes. By combining the strengths of generative models with user-guided interactions, Framer improves both the quality and controllability of the interpolated frames.

- 2 RELATED WORK

- 2.1 VIDEO FRAME INTERPOLATION

Video frame interpolation (VFI) aims to synthesize intermediate frames from two successive video frames. Most previous methods view VFI as a low-level task, assuming a moderate motion between frames. These methods can roughly be categorized as flow-based methods and kernel-

based methods. Specifically, the flow-based methods leverage estimated optical flow for frame synthesis (Jiang et al., 2018; Xu et al., 2019; Liu et al., 2020; Niklaus & Liu, 2020; 2018; Sim et al., 2021; Huang et al., 2020; Jin et al., 2023; Xue et al., 2019; Park et al., 2020; 2021; Kong

- et al., 2022). By contrast, the kernel-based methods rely on spatially adaptive kernels to synthesize the interpolated pixels (Lee et al., 2020; Cheng & Chen, 2022; Ding et al., 2021; Niklaus et al., 2017; Cheng & Chen, 2020; Gui et al., 2020; Lu et al., 2022). While the former potentially suffers from inaccurate flow estimation, the latter are often constrained by kernel size. To obtain the best of both worlds, some methods combine the flow- and kernel-based methods for end-to-end video frame interpolation (Bao et al., 2019; 2021; Danier et al., 2022; Li et al., 2022). We refer readers to (Dong et al., 2023) for a more comprehensive survey on these methods.

Recently, inspired by the generative capacity of large-scale pre-trained video diffusion models, some methods attempt to tackle VFI from a generation perspective (Danier et al., 2024; Feng et al., 2024; Jain et al., 2024; Xing et al., 2023; Wang et al., 2024a). For example, LDMVFI (Danier et al., 2024) formulates VFI as a conditional generation problem and utilizes a latent diffusion model for perceptually oriented video frame interpolation. Similarly, VIDIM (Jain et al., 2024) leverages cascaded diffusion models to generate high-fidelity interpolated videos with nonlinear motions. Though progress has been made, these methods still have difficulties in tackling large differences between the starting and ending frames. Moreover, they focus on generating a single deterministic solution for video frame interpolation, without controllability. Differently, we can generate multiple plausible solutions under large motion changes, and allow simple and intuitive drag interaction for user-intended interpolation results.

- 2.2 VIDEO DIFFUSION MODELS

Large-scale pre-trained video diffusion models (Brooks et al., 2024; Blattmann et al., 2023b; Ge et al., 2023; Chen et al., 2023; 2024; Wang et al., 2023a; Blattmann et al., 2023a) have shown unprecedented generation results in visual quality, diversity, and realism. These methods leverage text or starting image controls, which are often insufficient in precision and interactiveness. Inspired by the success in controllable image generation (Zhang et al., 2023; Mou et al., 2024b), several works attempt to add additional controls to video diffusion models. Early explorations (Wang et al., 2023b; Guo et al., 2023) utilize structural controls, like sketch and depth maps, for video generation. However, these control signals are often difficult to obtain during sampling, limiting their practical applications. Differently, recent works focus on motion control and introduce trajectory control for object motion (Wu et al., 2024; Mou et al., 2024a; Yin et al., 2023) and camera pose control for camera motion (Wang et al., 2024b; He et al., 2024; Bahmani et al., 2024). Both control signals can be obtained through easy and intuitive user interactions. In this paper, we enhance the creative potential and flexibility of the video framer interpolation process, allowing users to produce plausible frame interpolation results following their control.

- 3 METHOD

Given two frames, I0 and In, indicating the start and end frame in a video, our goal is to generate the plausible contiguous video I = {Ii}ni=0 by sampling from the conditional distribution p I | I0,In . Here, n is the number of frames in the video. Our method, termed Framer, supports a user-interactive mode for customized point trajectories and an “autopilot” mode for video frame interpolation without trajectory inputs, as shown in Fig. 2a and Fig. 2b. In the following, we will introduce how we add frame conditions to the video diffusion model to achieve video interpolation in Sec. 3.1. To support user-interactive drag control, we introduce a control branch in Sec. 3.2 for point trajectory guidance, which also enhances point correspondences across frames. In the “autopilot” mode, we estimate trajectories of matched points in the video with our novel bi-directional point tracking method, as illustrated in Sec. 3.3.

- 3.1 MODEL ARCHITECTURE

Large-scale pre-trained video diffusion models have a strong visual prior on the appearance, structure, and movement of open-world objects (Brooks et al., 2024). Our approach builds on the video diffusion model to exploit this prior. Considering that the Image-to-Video (I2V) diffusion

(a) User Interactive Mode

(c) Trajrectory Controlling Branch

|[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>|
|---|

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

StartFrameStartFrameEndFrameEndFrame

| |[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]| | | |
|---|---|---|---|---|
| | | | | |

Condition Encoder

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

| | |
|---|---|
| | |

zero-conv

Point Trajectory

UNet Encoder

User customizes the keypoints

Output Video

Trajectory

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

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

|[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>|
|---|

[Figure 157]

[Figure 158]

|[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]|
|---|

Start Frame

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Output

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

Video

[Figure 185]

Noise

3D-UNet

End Frame

Point Matching

Trajectory Initialization

Trajectory Updating

[Figure 186]

[Figure 187]

[Figure 188]

Output Video

[Figure 189]

(d) Video Frame Interpolation Fine-tuning

(b) Autopilot Mode

- Figure 2: Framer supports (a) a user-interactive mode for customized point trajectories and (b) an “autopilot” mode for video frame interpolation without trajectory inputs. During training, (d) we fine-tune the 3D-UNet of a pre-trained video diffusion model for video frame interpolation. Afterward, (c) we introduce point trajectory control by freezing the 3D-UNet and fine-tuning the controlling branch.

model naturally supports first-frame conditioning, we choose the representative I2V diffusion model, Stable Video Diffusion (SVD) (Blattmann et al., 2023a), as our base model, as shown in Fig. 2d.

Based on the I2V model, we need to introduce additional end-frame conditioning to realize video interpolation. To preserve the visual prior of the pre-trained SVD as much as possible, we follow the conditioning paradigm of SVD and inject end-frame conditions in the latent space and semantic space, respectively. Specifically, we concatenate the VAE-encoded latent feature of the first frame, denoted as z0, with the noisy latent of the first frame, as did in SVD. Additionally, we concatenate the latent feature of the last frame, zn, with the noisy latent of the end frame, considering that the conditions and the corresponding noisy latents are spatially aligned. In addition, we extract the CLIP image embedding of the first and last frames separately and concatenate them for cross-attention feature injection. The U-Net ϵθ is trained using the denoising score matching objective:

t,z0,zn,t,ϵ∼N(0,I) ϵ − ϵθ zt;t,z0,zn 2 . (1)

L = Ez

- 3.2 INTERACTIVE FRAME INTERPOLATION

Ambiguity remains given the start and end frames, especially when the distinction between the two frames is large. The reason is that multiple plausible interpolation results can be obtained by sampling video from the conditional distribution P I | I0,In for the same input pair. To better align with the user intention, we introduce a control branch for custmized point trajectory guidance.

Technically, we train a point trajectory-based control branch for correspondence modeling, as shown in Fig. 2c. During training, we use the following steps to obtain the point trajectory as control signals. Firstly, we randomly initialize some sampled points around a fixed sparse grid in the first frame, and use Co-Tracker (Karaev et al., 2023) to obtain the trajectories of these points in the whole video. Secondly, we remove trajectories that are not visible in more than half of the video frames. Lastly, we sample the point trajectories with larger motions with greater probability. Considering that the users usually only input a small number of point trajectories, we keep only 1 to 10 trajectories during training. Please refer to the App. A for more details.

After obtaining the sampled point trajectories, we follow DragNUWA (Yin et al., 2023) and DragAnything (Wu et al., 2024) to transform the point coordinates into a Gaussian heatmap, denoted as ctraj, which is used as input to the control module. We follow the conditioning mechanism in ControlNet (Zhang et al., 2023) to incorporate the trajectory control. Specifically, we copy the encoder of 3D-UNet to encode the trajectory map and add it into the decoder of U-Net after zero-

SIFT Feature Matching

- (a) Forward Point Tracking
- (b) Backward Point Tracking

Start Frame Feature End Frame Feature Middle Frame Feature

[Figure 190]

| |
|---|

| |
|---|

Nearest

Start Frame

Neighbor

| |
|---|

| |
|---|

[Figure 191]

Point Unpdating

Nearest Neighbor

End Frame

Bi-directional

Consistency

- Figure 3: Point trajectory estimation. The point trajectory is initialized by interpolating the coordinates of matched keypoints. In each de-noising step, we perform point tracking by finding the nearest neighbor of keypoints in the start and end frames, respectively. Lastly, We check the bi-directional tracking consistency before updating the point coordinate.

convolution (Zhang et al., 2023). This training process can be represented as:

t,z0,zn,t,ϵ∼N(0,I) ϵ − ϵcθ zt;t,z0,zn,ctraj 2 . (2) Here, ϵcθ is the combination of the denoising U-Net and the ControlNet branch.

L = Ez

Discussion. The introduction of point trajectory control not only facilitates user interaction, but also enhances the correspondence among points from different frames. As demonstrated in experiments, this approach enables the model to effectively tackle challenging cases, such as when the start and end frames differ significantly.

- 3.3 “AUTOPILOT” MODE FOR FRAME INTERPOLATION

In practical applications, users may not always prefer manual drag controls. For this reason, we propose an “autopilot” mode to enhance the ease of use of our Framer. It mainly contains a trajectory initialization and a trajectory updating process, as illustrated in Fig. 2b.

Trajectory Initialization. Given the start and end frames of the input video, we can obtain the matching points between the two frames by applying feature-matching algorithms. The matched points are denoted as {pi}mi=1, where m is the number of matching points. pi denotes the known anchor points on the trajectory. At initialization, it contains the matched points on the first and last frames, i.e., pi = [p0i,pni ]. Although varying feature matching algorithms are feasible, we use the classical SIFT feature matching (Lowe, 2004) here for its simplicity and effectiveness. Subsequently, we can obtain the i-th trajectory cˆi by interpolating the anchor points pi. The estimated trajectory for all m matched points, denoted as cˆtraj = {cˆi}mi=1, are used as the input condition in Eq. (2).

Trajectory Updating. Although the initial trajectory provides temporally consistent point correspondence, the trajectory obtained by connecting points in the first and last frames may not be accurate. Inspired by DragGAN (Pan et al., 2023) and DragDiffusion (Shi et al., 2023), we perform point tracking using the intermediate feature in U-Net to update the trajectories. Specifically, in each denoising step, we interpolate the U-Net features to the image resolution, denoted as F. Here we use the feature of the penultimate upsampled block in U-Net, since it enjoys a good trade-off between feature resolution and discriminativeness. We use F(p) to represent the feature of the point p, which is obtained via bilinear interpolation, since the coordinates may not be integers.

In each denoising step, we apply point tracking to update the coordinates of the middle frame points. We use nearest neighbor search in a feature patch around the point. The feature patch represents a set of points whose distance to point p is less than r, and is denoted as Ω(p,r) = {(x,y)||x − xp |< r,|y − yp |< r}. For a middle frame point pki in the k-th frame, we find the nearest point relative to the anchor point p0i via:

pk,i 0 := arg min qik∈Ω(pki ,r1)

F qik − F p0i 1 . (3)

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

Blended Inputs GT AMT RIFE FLAVR FILM LDMVFI DynamiCrafter SVDKFI Ours

- Figure 4: Qualitative comparison. ‘GT” strands for ground truth. For each method, we only present the middle frame of 7 interpolated frames. The full results can be seen in Fig. S4 and Fig. S5 in the Appendix.

Similarly, we can obtain the nearest point relative to the last anchor point pni :

pk,ni := arg min qik∈Ω(pki ,r1)

F qik − F(pni ) 1 . (4)

As shown in Fig. 3, to further ensure the accuracy of the coordinates of the updated points, we check the consistency of the two nearest points obtained by matching with p0i and pni . When the distance between the two is less than a threshold r2, i.e., pk,ni ∈ Ω pk,i 0,r2 , we update the point coordinates by setting pki = (pk,i 0 + pk,ni )/2. Then, we add the point to the anchor points list pi and interpolate pi to get the updated trajectory ci, which is used as the input condition to the next denoising step.

- 4 EXPERIMENTS

- 4.1 IMPLEMENTATION DETAILS

Our method is built on SVD and trained on the high-quality OpenVidHD-0.4M dataset (Nan et al., 2024). During the training of U-Net, we fixed the spatial attention and residual blocks, and only fine-tuned the input convolutional and temporal attention layers. The model is trained for 100K iterations using the AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate of 1e-5. We obtained the point trajectories by pre-processing the video using the Co-Tracker (Karaev et al.,

- 2023). When training the control module, we fixed the U-Net and optimized the control module for 10K steps using the AdamW optimizer, with a learning rate of 1e-5. All training is performed on 16 NVIDIA A100 GPUs, and the total batch size is 16. During “autopilot” mode sampling, we keep m = 5 best matching keypoints for trajectory guidance, and the distance thresholds for point tracking are set as r1 = 5 and r2 = 3. Please refer to App. A for more details.

4.2 COMPARISON

[Figure 212]

[Figure 213]

[Figure 214]

Framer 90.5%

SVDKFI 1.2%

LDMVFI 0.7%

AMT 1.7%

RIFE 0.7%

FLAVR 0.8%

FILM 4.4%

Figure 5: Reults on human preference.

Existing methods do not support drag-user interaction. Thus, we use the “autopilot” mode of Framer to make fair comparisons. We select baselines from two distinct categories. The first category includes the latest general diffusion-based video interpolation models, including LDMVFI (Danier et al., 2024), DynamicCrafter (Xing et al., 2023), and SVDKFI (Wang et al.,

- 2024a). The second category encompasses traditional video interpolation methods, such as AMT (Li et al., 2023), RIFE (Huang et al., 2020), FLAVR (Kalluri

- et al., 2023), and FILM (Reda et al., 2022). We conduct quantitative and qualitative analyses, as well as user studies, on two publicly available datasets: DAVIS (Pont-Tuset et al., 2017) and UCF101 (Soomro et al., 2012).

Qualitative Comparison. As shown in Fig. 4, our method produces significantly clearer textures and natural motion compared to existing interpolation techniques. It performs especially well in scenarios with substantial differences between the input frames, where traditional methods often

##### DAVIS-7 UCF101-7

|PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓|PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓|
|---|---|
|AMT (Li et al., 2023) 21.66 0.7229 0.2860 39.17 245.25 RIFE (Huang et al., 2020) 22.00 0.7216 0.2663 39.16 319.79 FLAVR (Kalluri et al., 2023) 20.94 0.6880 0.3305 52.23 296.37 FILM (Reda et al., 2022) 21.67 0.7121 0.2191 17.20 162.86 LDMVFI (Danier et al., 2024) 21.11 0.6900 0.2535 21.96 269.72 DynamicCrafter (Xing et al., 2023) 15.48 0.4668 0.4628 35.95 468.78 SVDKFI (Wang et al., 2024a) 16.71 0.5274 0.3440 26.59 382.19|26.64 0.9000 0.1878 37.80 270.98<br><br>27.04 0.9020 0.1575 27.96 300.40<br><br><br>26.50 0.8982 0.1836 37.79 279.58 26.74 0.8983 0.1378 16.22 239.48 26.68 0.8955 0.1446 17.55 270.33 17.62 0.7082 0.3361 61.71 646.91 21.04 0.7991 0.2146 44.81 301.33<br><br>|
|Framer (Ours) 21.23 0.7218 0.2525 27.13 115.65 Framer with Co-Tracker (Ours) 22.75 0.7931 0.2199 27.43 102.31|25.04 0.8806 0.1714 31.69 181.55 27.08 0.9024 0.1714 32.37 159.87<br><br>|

Table 1: Quantitative comparison with existing video interpolation methods on reconstruction and generative metrics, evaluated on all 7 generated frames.

Strat Frame Generated Frames End Frame

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

w/odragdrag1drag2

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

Figure 6: Results on user interaction. The first row is generated without drag input, while the other two are generated with different drag controls. Customized trajectories ares overlaid on frames.

fail to interpolate content accurately. Compared to other diffusion-based methods like LDMVFI and SVDKFI, Framer demonstrates superior adaptability to challenging cases and offers better control.

Quantitative Comparison. As discussed in VIDIM (Jain et al., 2024), reconstruction metrics like PSNR, SSIM, and LPIPS fail to capture the quality of interpolated frames accurately, since they penalize other plausible interpolation results that are not pixel-aligned with the original video. While generation metrics such as FID offer some improvement, they still fall short as they do not account for temporal consistency and evaluate frames in isolation. Despite this, we present the quantitative metrics for various settings on both datasets, where our method achieves the best FVD score among all baselines as in Tab. 1. We also evaluate Framer with 5 random point trajectories from ground-truth videos, estimated using Co-Tracker. As can be seen, “Framer with Co-Tracker” achieves superior performance even in reconstruction metric. For a more comprehensive assessment of quality, we recommend reviewing the supplementary comparison videos.

User Study. Since quantitative metrics fall short in reflecting video quality, we further assessed our method’s performance through a user study. In this study, participants reviewed video sets generated from the same input frame pair by both existing methods and our Framer. Participants assessed up to 100 randomly ordered video sets and selected the one they found most realistic. In total, 20 participants provided 1,000 ratings across these video sets. As illustrated in Fig. 5, the results demonstrate a strong preference among human raters for the outputs produced by our method.

Strat Frame Generated Frames End Frame

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

Figure 7: Novel view synthesis on both static (1st row) and dynamic scenes (2nd row).

Strat Frame Generated Frames End Frame

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

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

Figure 8: Applications on cartoon (1st row) and sketch (2nd row) interpolation.

Strat Frame Generated Frames End Frame

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Figure 9: Applications on time-lapsing video generation.

- 4.3 APPLICATIONS

Optional drag control. Given the same input start and end frames, multiple plausible results can satisfy the goal of video interpolation. With Framer, users can direct the motion of the entities in input frames with simple drags for their intention, or simply obtain a default interpolation result without drags. As shown in Fig. 6, the seal moves in varying directions given the same input frames.

Novel view synthesis (NVS) is a classical 3D vision task, with a wide range of applications. Using images of different viewpoints as the start and end frames of the video respectively, we can realize the NVS from sparse viewpoint input by performing video interpolation. As shown in Fig. 7, our method achieves pleasing NVS in both static scenes (first row) and dynamic scenes (second and third rows). Taking the second row as an example, the house gradually moves out of the scene as the camera keeps moving forward. In the meantime, the car moves in the opposite direction to the camera and gradually takes up a larger proportion in the frame.

Cartoon and sketch interpolation. We can dramatically simplify the process of cartoon video production, by interpolating manually created cartoon images. To this end, we tested our method on cartoon data. Although our method is not specifically trained on cartoon videos, it produces appealing cartoon video results and supports both color images and sktech drawing frame interpolation, as shown in Fig. 8. For example, our method successfully models the motion of two objects, i.e., the front vehicle pulls sideways while the rear vehicle follows, as shown in the first row. In the third row, Framer produces a smooth motion of the hand lifting in sketch drawings.

Time-lapsing video generation. Time-lapse photography can vividly demonstrate slow changes that are difficult to detect with the naked eye. Typically, it requires sufficient storage space to hold a large amount of image data and a complex post-processing procedure to organize and edit the images. Video interpolation provides a simple and effective way to obtain time-lapse videos by interpolating frames with only a few images of key moments. As shown in Fig. 9, Framer produces the smooth change of moon waxing and waning.

Strat Frame Generated Frames End Frame y-t

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

- Figure 10: Applications on slow-motion video generation. The y-t slice highlighted in red on video frames is visualized on the right.

Strat Frame Generated Frames End Frame

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

- Figure 11: Applications on image morphing. Customized trajectories ares overlaid on end frames.

w/o

traj. w/o

traj.update w/o

bi-directional Framer

(Ours)

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

[Figure 359]

Inputs with SIFT matching

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Frame 7

[Figure 360]

- Figure 12: Ablations on each component. “w/o trajectory” denotes inference without guidance from point trajectory, “w/o traj. update” indicates inference without trajectory updates, and “w/o bi” suggests trajectory updating without bi-directional consistency verification.

Slow-motion video generation enhances visual effects by highlighting fine details and allows closer examination of fast phenomena. Our Framer inherently supports fast frame interpolation, as demonstrated in Fig. 10, enabling smooth slow-motion effects suitable for films and animations.

Image morphing (Aloraibi, 2023) is a popular image transformation technique with many applications in computer vision and computer graphics. Given two topologically similar images, it aims to generate a series of reasonable intermediate images. Using tue two images as the start and end frames, Framer can produce natural and smooth image morphing results. For example, in Fig. 1, we show the “evolution” process of Pokemon. More cases can be found in Fig. S13.

- 4.4 ABLATIONS STUDIES

We conducted ablation studies on the individual components of Framer to validate their effectiveness. The results are illustrated in Fig. 12. Our observations are as follows. First, when the trajectory guidance is removed (denoted as “w/o traj.”), the foreground motorcycle exhibits significant distortion, as shown in the 1st row of Fig. 12. Conversely, with the inclusion of trajectory guidance, the temporal consistency of the video is notably enhanced, as depicted in the 2nd row. We believe this is due to the enhancement of point correspondence modeling across frames. Second, removing trajectory updates (denoted as “w/o traj. update”) or updating the trajectory without bidirectional consistency checks (denoted as “w/o bi-directional”) results in blurring in the wheel regions of the output video. We suspect the blurring is caused by the guidance of unnatural motion from inaccurate trajectories, which conflicts with the generation prior in the pre-trained diffusion model, leading to local blurring. In contrast, our method produces video frame interpolation results with natural motion and smooth temporal coherence. The quantitative results in Tabs. S1 and S2 in App. B further support these findings, showing a similar trend to the qualitative ablation experiments.

- 5 CONCLUSION AND FUTURE WORK

In this paper, we introduce Framer, an interactive frame interpolation pipeline designed to produce smoothly transitioning frames between two images, guided by user-defined point trajectories. By harnessing user input point controls from the start and end frames, we effectively guide the video interpolation process. Moreover, our method offers an “autopilot” mode that introduces a module to automatically estimate keypoints and refine trajectories without manual input. Through extensive experiments and user studies, we demonstrate the superiority of our method in achieving promising results in terms of both the quality and controllability of the interpolated frames. However, challenges remain, particularly in transitioning between different clips. A potential solution involves splitting the clips into several keyframes and then interpolating these keyframes sequentially. Future work will focus on addressing these challenges.

REFERENCES

Alyaa Aloraibi. Image morphing techniques: A review. Technium: Romanian Journal of Applied Sciences and Technology, 2023.

Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, David B. Lindell, and Sergey Tulyakov. VD3D: taming large video diffusion transformers for 3d camera control. arXiv: Computing Research Repo., abs/2407.12781, 2024.

Wenbo Bao, Wei-Sheng Lai, Chao Ma, Xiaoyun Zhang, Zhiyong Gao, and Ming-Hsuan Yang. Depth-aware video frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2019.

Wenbo Bao, Wei-Sheng Lai, Xiaoyun Zhang, Zhiyong Gao, and Ming-Hsuan Yang. Memc-net: Motion estimation and motion compensation driven neural network for video interpolation and enhancement. IEEE Trans. Pattern Anal. Mach. Intell., 2021.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv: Computing Research Repo., abs/2311.15127, 2023a.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2023b.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. OpenAI technical reports, 2024.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation. arXiv: Computing Research Repo., abs/2310.19512, 2023.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv: Computing Research Repo., abs/2401.09047, 2024.

Xianhang Cheng and Zhenzhong Chen. Video frame interpolation via deformable separable convolution. In Assoc. Adv. Artif. Intell., 2020.

Xianhang Cheng and Zhenzhong Chen. Multiple video frame interpolation via enhanced deformable separable convolution. IEEE Trans. Pattern Anal. Mach. Intell., 2022.

Duolikun Danier, Fan Zhang, and David Bull. St-mfnet: A spatio-temporal multi-flow network for frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2022.

Duolikun Danier, Fan Zhang, and David Bull. LDMVFI: video frame interpolation with latent diffusion models. In Assoc. Adv. Artif. Intell., 2024.

Tianyu Ding, Luming Liang, Zhihui Zhu, and Ilya Zharkov. CDFI: compression-driven network design for frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2021.

Jiong Dong, Kaoru Ota, and Mianxiong Dong. Video frame interpolation: A comprehensive survey. ACM Trans. Multim. Comput. Commun. Appl., 2023.

Haiwen Feng, Zheng Ding, Zhihao Xia, Simon Niklaus, Victoria Fern´andez Abrevaya, Michael J. Black, and Xuaner Zhang. Explorative inbetweening of time and space. arXiv: Computing Research Repo., abs/2403.14611, 2024.

Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Int. Conf. Comput. Vis., 2023.

Shurui Gui, Chaoyue Wang, Qihua Chen, and Dacheng Tao. Featureflow: Robust video interpolation via structure-to-texture generation. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. arXiv: Computing Research Repo., abs/2311.16933, 2023.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv: Computing Research Repo., abs/2404.02101, 2024.

Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. RIFE: real-time intermediate flow estimation for video frame interpolation. arXiv: Computing Research Repo., abs/2011.06294, 2020.

Siddhant Jain, Daniel Watson, Eric Tabellion, Aleksander Holynski, Ben Poole, and Janne Kontkanen. Video interpolation with diffusion models. arXiv: Computing Research Repo., abs/2404.01203, 2024.

Huaizu Jiang, Deqing Sun, Varun Jampani, Ming-Hsuan Yang, Erik G. Learned-Miller, and Jan Kautz. Super slomo: High quality estimation of multiple intermediate frames for video interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2018.

Xin Jin, Longhai Wu, Guotao Shen, Youxin Chen, Jie Chen, Jayoon Koo, and Cheul-Hee Hahm. Enhanced bi-directional motion estimation for video frame interpolation. In IEEE Winter Conf. Appl. Comput. Vis., 2023.

Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, and Du Tran. FLAVR: flow-agnostic video representations for fast frame interpolation. In IEEE Winter Conf. Appl. Comput. Vis., 2023.

Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. arXiv: Computing Research Repo., abs/2307.07635, 2023.

Lingtong Kong, Boyuan Jiang, Donghao Luo, Wenqing Chu, Xiaoming Huang, Ying Tai, Chengjie Wang, and Jie Yang. Ifrnet: Intermediate feature refine network for efficient frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2022.

Hyeongmin Lee, Taeoh Kim, Tae-Young Chung, Daehyun Pak, Yuseok Ban, and Sangyoun Lee. Adacof: Adaptive collaboration of flows for video frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.

Changlin Li, Guangyang Wu, Yanan Sun, Xin Tao, Chi-Keung Tang, and Yu-Wing Tai. H-VFI: hierarchical frame interpolation for videos with large motions. arXiv: Computing Research Repo., abs/2211.11309, 2022.

Zhen Li, Zuo-Liang Zhu, Linghao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. AMT: all-pairs multi-field transforms for efficient frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023.

Yihao Liu, Liangbin Xie, Siyao Li, Wenxiu Sun, Yu Qiao, and Chao Dong. Enhanced quadratic video interpolation. In Eur. Conf. Comput. Vis. Worksh., 2020.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Int. Conf. Learn. Represent., 2019.

David G. Lowe. Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vis., 2004.

Liying Lu, Ruizheng Wu, Huaijia Lin, Jiangbo Lu, and Jiaya Jia. Video frame interpolation with transformer. In IEEE Conf. Comput. Vis. Pattern Recog., 2022.

Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. arXiv: Computing Research Repo., abs/2405.13865, 2024a.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Assoc. Adv. Artif. Intell., 2024b.

Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv: Computing Research Repo., abs/2407.02371, 2024.

Simon Niklaus and Feng Liu. Context-aware synthesis for video frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2018.

Simon Niklaus and Feng Liu. Softmax splatting for video frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.

Simon Niklaus, Long Mai, and Feng Liu. Video frame interpolation via adaptive separable convolution. In Int. Conf. Comput. Vis., 2017.

Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your GAN: interactive point-based manipulation on the generative image manifold. In Erik Brunvand, Alla Sheffer, and Michael Wimmer (eds.), SIGGRAPH, 2023.

Junheum Park, Keunsoo Ko, Chul Lee, and Chang-Su Kim. BMBC: bilateral motion estimation with bilateral cost volume for video interpolation. In Eur. Conf. Comput. Vis., 2020.

Junheum Park, Chul Lee, and Chang-Su Kim. Asymmetric bilateral motion estimation for video frame interpolation. In Int. Conf. Comput. Vis., 2021.

Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alexander Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv: Computing Research Repo., abs/1704.00675, 2017.

Fitsum A. Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless. FILM: frame interpolation for large motion. In Eur. Conf. Comput. Vis., 2022.

Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent Y. F. Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv: Computing Research Repo., abs/2306.14435, 2023.

Hyeonjun Sim, Jihyong Oh, and Munchurl Kim. XVFI: extreme video frame interpolation. In Int. Conf. Comput. Vis., 2021.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv: Computing Research Repo., abs/1212.0402, 2012.

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv: Computing Research Repo., abs/2308.06571, 2023a.

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. In Adv. Neural Inform. Process. Syst., 2023b.

Xiaojuan Wang, Boyang Zhou, Brian Curless, Ira Kemelmacher-Shlizerman, Aleksander Holynski, and Steven M Seitz. Generative inbetweening: Adapting image-to-video models for keyframe interpolation. arXiv: Computing Research Repo., abs/2408.15239, 2024a.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH, 2024b.

Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. arXiv: Computing Research Repo., abs/2403.07420, 2024.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv: Computing Research Repo., abs/2310.12190, 2023.

Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. arXiv: Computing Research Repo., abs/2405.17933, 2024.

Xiangyu Xu, Li Si-Yao, Wenxiu Sun, Qian Yin, and Ming-Hsuan Yang. Quadratic video interpolation. In Adv. Neural Inform. Process. Syst., 2019.

Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T. Freeman. Video enhancement with task-oriented flow. Int. J. Comput. Vis., 2019.

Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. RAPHAEL: text-to-image generation via large mixture of diffusion paths. In Adv. Neural Inform. Process. Syst., 2023.

Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv: Computing Research Repo., abs/2308.08089, 2023.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Int. Conf. Comput. Vis., 2023.

APPENDIX

- A MORE IMPLEMENTATION DETAILS

During training, we sample 14 consecutive frames from videos, with a spatial resolution of 512×320. Specifically, we center-crop the video to an aspect ratio of 512/320, then resize the video frames to the resolution of 512 × 320. Random horizontal flip is utilized for data augmentation. We sample the video in temporal dimension, with a frame interval of 2. For the training of the point trajectory-based ControlNet, we sample 1 to 10 trajectories with larger motions for training. Specifically, we follow ReVideo (Mou et al., 2024a) and sample the trajectories by setting the normalized lengths of the trajectories as sampling probabilities. During “autopilot” mode sampling, we use the Euler sampler with 30 diffusion steps in total. For point tracking in Sec. 3.3, we use the output feature of the second decoder block in the 3D-UNet. We resize the shorter side of the video to the length of 512, then center crop the video to the resolution of 512 × 320.

- B MORE DETAILED ABLATION RESULTS

Qualitative results for ablation studies. In Fig. 12, we show the qualitative results for ablation studies. We supplement these results with the quantitative results in Tab. S1 and Tab. S2, which show a similar trend to the qualitative ablation experiments.

#### DAVIS-7 UCF101-7

|PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓<br><br>|PSNR↑ SSIM↑ LPIPS↓ FID↓ FVD↓|
|---|---|
|w/o trajectory 20.19 0.6831 0.2787 28.25 128.71 w/o traj. updating 20.82 0.7054 0.2621 27.33 120.73 w/o bi-directional 20.94 0.7102 0.2602 27.23 116.81|24.16 0.8677 0.1798 32.64 195.54 24.69 0.8748 0.1842 31.95 187.37 24.73 0.8746 0.1845 31.66 183.74<br><br>|
|Framer (Ours) 21.23 0.7218 0.2525 27.13 115.65<br><br>|25.04 0.8806 0.1714 31.69 181.55|

- Table S1: Ablations on each component, evaluating all 7 generated frames. “w/o trajectory” denotes inference without guidance from point trajectory, “w/o traj. updating” indicates inference without trajectory updating, and “w/o bi” suggests trajectory updating without bi-directional consistency verification.

DAVIS-7 (mid-frame) UCF101-7 (mid-frame)

|PSNR↑ SSIM↑ LPIPS↓ FID↓|PSNR↑ SSIM↑ LPIPS↓ FID↓<br><br>|
|---|---|
|w/o trajectory 19.30 0.6504 0.3093 57.10 w/o traj. updating 19.84 0.6700 0.2935 55.37 w/o bi-directional 19.95 0.6739 0.2919 54.75<br><br>|23.14 0.8523 0.1967 54.98 23.60 0.8590 0.2009 53.83 23.65 0.8586 0.2016 53.54|
|Framer (Ours) 20.18 0.6850 0.2845 55.13|23.92 0.8646 0.1889 53.33|

- Table S2: Ablations on each component, evaluating only the middle frame out of all 7 generated frames. “w/o trajectory” denotes inference without guidance from point trajectory, “w/o traj. updating” indicates inference without trajectory updating, and “w/o bi” suggests trajectory updating without bi-directional consistency verification.

Ablations on diffusion feature for point tracking. As detailed in Sec. 3.3, we perform point tracking using the diffusion feature for point trajectory updating. Here we perform ablated experiments on the selection of the diffusion feature. The results are shown in Fig. S1. It can be seen that in both DAVIS-7 and UCF-7, point tracking with the output feature from the second diffusion block gives rise to the best-performing results in FVD.

Ablations on diffusion steps for correspondence guidance. We ablate the diffusion steps for correspondence guidance by only applying the guidance at the early steps or late steps in diffusion sampling. The results are shown in Fig. S2. As can be seen, the early steps are often more important than the late steps for correspondence modeling. For example, on DAVIS-7, a pleasing FVD can be obtained when performing guidance only on 0-18 diffusion steps. By contrast, performing guidance only on 18-30 diffusion steps brings litter improvements. We speculate that this is because the

[Figure 361]

[Figure 362]

- Figure S1: Ablations on diffusion feature for point tracking at test time, experiments conducted on DAVIS-7 (left) and UCF101-7 (right).

[Figure 363]

[Figure 364]

- Figure S2: Ablations on the start and end diffusion steps for correspondence guidance, experiments conducted on DAVIS-7 (left) and UCF101-7 (right). We use a total sampling step of 30.

early diffusion steps focus on the structural information of the video, while the late diffusion steps focus on the texture and details (Xue et al., 2023). The correspondence guidance at early steps already helps the model obtain a reasonable video structure. In the implementation, we simply apply correspondence guidance in all diffusion steps, without detailed searches on the hyper-parameter.

Ablations on the number of trajectories for correspondence guidance. As described in Sec. 3.3, we use m trajectories for correspondence guidance during sampling. Here we perform ablated experiments on this hyper-parameter, and the result is shown in Fig. S3. It can be seen that sampling with the 5 trajectories leads to the best performance. Thus we set m = 5 by default.

- C MORE DETAILS ON COMPARISON WITH PREVIOUS METHODS

Benchmark. We follow the practice of VIDIM (Jain et al., 2024) and perform the quantitative evaluation on DAVIS-7 and UCF101-7 datasets using both reconstruction and generative metrics. Both DAVIS-7 and UCF101-7 are obtained by sampling 7 consecutive video frames from the corresponding datasets. We use all videos in the DAVIS dataset and a subset of 400 videos in the UCF101 dataset.

More results on comparisons. In Tab. S3 we provide the quantitative comparison based on the middle frame of the 7 interpolated video frames. Besides, in Fig. S4, Fig. S5, Fig. S6, and Fig. S7, we show more qualitatively comparisons with exiting methods.

[Figure 365]

[Figure 366]

- Figure S3: Ablations on the number of trajectories for guidance during sampling, experiments conducted on DAVIS-7 (left) and UCF101-7 (right).

DAVIS-7 (mid-frame) UCF101-7 (mid-frame)

|PSNR↑ SSIM↑ LPIPS↓ FID↓|PSNR↑ SSIM↑ LPIPS↓ FID↓<br><br>|
|---|---|
|AMT (Li et al., 2023) 20.59 0.6834 0.3564 100.36 RIFE (Huang et al., 2020) 20.74 0.6813 0.3102 80.78<br><br>FLAVR (Kalluri et al., 2023) 19.93 0.6514 0.4074 118.45<br><br>FILM (Reda et al., 2022) 20.28 0.6671 0.2620 48.70 LDMVFI (Danier et al., 2024) 19.87 0.6435 0.2985 56.46 DynamicCrafter (Xing et al., 2023) 14.61 0.4280 0.5082 77.65 SVDKFI (Wang et al., 2024a) 16.06 0.4974 0.3719 53.49<br><br>|25.24 0.8837 0.2237 75.97 25.68 0.8842 0.1835 59.33<br><br>24.93 0.8796 0.2164 79.86<br>25.31 0.8818 0.1623 41.23 25.16 0.8789 0.1695 43.01 17.05 0.6935 0.3502 97.01 20.03 0.7775 0.2326 69.26<br>|
|Framer (Ours) 20.18 0.6850 0.2845 55.13 Framer with Co-Tracker (Ours) 21.94 0.7693 0.2437 55.77<br><br>|23.92 0.8646 0.1889 53.33 25.86 0.8868 0.1873 54.64|

- Table S3: Quantitative comparison with existing video interpolation methods on reconstruction and generative metrics, evaluated only on the middle frame out of all 7 generated frames.

- D MORE QUALITATIVE RESULTS

We provide more qualitative results on drag control, novel view synthesis, cartoon and sketch interpolation, time-lapsing video generation, slow-motion video generation, and image morphing in Fig. S8, Fig. S9, Fig. S10, Fig. S11, Fig. S12, and Fig. S13, respectively.

- E DISCUSSIONS ON LIMITATIONS

Framer is built on top of the large-scale pre-trained video diffusion model, thus it inherits the limitations of the pre-trained model. Moreover, the point trajectories in Framer rely on the matching points between the input image pair for interpolating complex motions. While this is a step forward compared with current models that can only simply motions, our method still faces difficulties when the differences between the front and back frames are so large that no matched points can be found at all. Thus, we will explore more powerful pre-trained video diffusion models, as well as training video interpolation models on larger-scale video data in the future. Lastly, our approach currently only supports drag control and does not explore other interaction methods. In the future, we will continue to explore other user-friendly controls such as text control and camera pose control.

[Figure 367]

DynamiCrafterLDMVFISVDKFIFLAVRFILMRIFEAMTOursGT

Oursw/tracker

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Frame 7

- Figure S4: More qualitative comparison with existing methods. “GT” strands for ground truth.

[Figure 368]

DynamiCrafterLDMVFISVDKFIFLAVRFILMRIFEAMTOursGT

Oursw/tracker

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Frame 7

- Figure S5: More qualitative comparison with existing methods. “GT” strands for ground truth.

[Figure 369]

DynamiCrafterLDMVFISVDKFIFLAVRFILMRIFEAMTOursGT

Oursw/tracker

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Frame 7

- Figure S6: More qualitative comparison with existing methods. “GT” strands for ground truth.

[Figure 370]

DynamiCrafterLDMVFISVDKFIFLAVRFILMRIFEAMTOursGT

Oursw/tracker

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Frame 7

- Figure S7: More qualitative comparison with existing methods. “GT” strands for ground truth.

Strat Frame Generated Frames End Frame

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

drag1drag2

[Figure 385]

[Figure 386]

[Figure 387]

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

- Figure S8: More results on user interaction. We show the results of two trajectory controls with the same input image pair.

[Figure 399]

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

[Figure 410]

[Figure 411]

[Figure 412]

Strat Frame Generated Frames End Frame

- Figure S9: More results on novel view synthesis. The first and second rows show results on static and dynamic scenes, respectively.

Strat Frame Generated Frames End Frame

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

(a) Cartoon Interpolation

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

(b) Sketch Interpolation

Figure S10: More results on (a) cartoon and (b) sketch interpolation.

Strat Frame Generated Frames End Frame

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

Figure S11: More results on time-lapsing video generation.

Strat Frame Generated Frames End Frame x-t

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

Figure S12: More results on slow-motion video generation. The x-t slice highlighted in red on video frames is visualized on the right.

Strat Frame Generated Frames End Frame

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

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

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

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

Figure S13: More results on image morphing.

