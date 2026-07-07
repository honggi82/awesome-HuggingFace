CharacterShot: Controllable and Consistent 4D Character Animation

JUNYAO GAO1,*, JIAXING LI3,*, WENRAN LIU2, YANHONG ZENG2, FEI SHEN4, KAI CHEN2, YANAN SUN2,‡, CAIRONG ZHAO1,‡,

1Tongji University, 2Shanghai AI Lab, 3Nanyang Technological University, 4National University of Singapore.

Character Input

||[Figure 1]|
|---|
<br><br>| |
|---|
<br><br>|[Figure 2]<br><br>[Figure 3]|
|---|
<br><br>|[Figure 4]|
|---|
<br><br>[Figure 5]|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

# arXiv:2508.07409v1[cs.CV]10Aug2025

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

4D Character Animation

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

###### Multi-View Videos 2D Pose Sequence

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Optimizing

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Fig. 1. Given any character image and a 2D pose sequence, CharacterShot synthesizes dynamic 3D characters with precise motion control and arbitrary viewpoint rendering, achieving both spatial-temporal and spatial-view consistency in 4D space.

are created by computer-generated imagery (CGI), which includes sophisticated technical chains-from professional 3D modeling and advanced motion capture to complex rigging and retargeting. This CGI pipeline is widely used in film, gaming, and the metaverse, and it requires specialized equipment and significant manual effort to build dynamic 3D characters—a process also known as 4D character animation. In this paper, we introduce CharacterShot, a novel framework that democratizes a low-cost CGI pipeline accessible to individual creators. As shown in Figure 1, CharacterShot supports diverse character designs and custom motion control (2D pose sequence), enabling 4D character animation in minutes and without specialized hardware.

In this paper, we propose CharacterShot, a controllable and consistent 4D character animation framework that enables any individual designer to create dynamic 3D characters (i.e., 4D character animation) from a single reference character image and a 2D pose sequence. We begin by pretraining a powerful 2D character animation model based on a cutting-edge DiT-based image-to-video model, which allows for any 2D pose sequnce as controllable signal. We then lift the animation model from 2D to 3D through introducing dual-attention module together with camera prior to generate multiview videos with spatial-temporal and spatial-view consistency. Finally, we employ a novel neighbor-constrained 4D gaussian splatting optimization on these multi-view videos, resulting in continuous and stable 4D character representations. Moreover, to improve character-centric performance, we construct a large-scale dataset Character4D, containing 13,115 unique characters with diverse appearances and motions, rendered from multiple viewpoints. Extensive experiments on our newly constructed benchmark, CharacterBench, demonstrate that our approach outperforms current stateof-the-art methods. Code, models, and datasets will be publicly available at https://github.com/Jeoyal/CharacterShot.

With the remarkable progress in recent generative models [16, 41], 4D generation [21, 91, 94] has demonstrated the impressive effectiveness in synthesizing 4D content. These methods aim to generate 4D content from a single-view character video. However, they often fall short in practical scenarios—such as those involving hand-drawn or AI-generated characters—where a single-view video including custom motions may not be available. A natural solution is to firstly generate the single-view character video using a 2D character animation method [40, 96], which excels at animating a character based on the pose sequence extracted from a target motion video. Such a two-stage framework forms a 4D character animation baseline exhibiting many limitations: 1) Disjoint modeling of pose and view makes it difficult to maintain consistent appearance and motion across views; 2) These methods are trained on general 3D

Additional Key Words and Phrases: Diffusion Model, 4D Generation

1 Introduction

When people watch the scientific films such as The Iron Man1 series, they are often amazed by the films’ astonishing realism, which leads some to wonder whether such advanced flying suits actually exist in real life. Unfortunately, the answer is no, these characters

∗Equal contributions. Work done during the internships in Shanghai AI Lab. ‡Corresponding authors. 1https://en.wikipedia.org/wiki/Iron_Man_(2008_film)

objects from static 3D object datasets such as Objverse [8], suffering from limited diversity in character representations and pose variations—both of which are crucial for generating compelling 4D character animations [3, 30, 61].

To address the above limitations, we propose CharacterShot, which is able to generate dynamic 3D characters from a given reference character image and a 2D pose sequence. This flexible and robust 4D character animation requires the model to possess the ability to precisely express the given motion and preserve consistent character appearance across both time and views. To this end, we first enhance the DiT-based image-to-video (I2V) model CogVideoX [87] by integrating pose conditions, enabling user-defined motion control for a given character image. Next, we extend the I2V model to a multi-view setting by introducing a dual-attention module and a camera prior, ensuring both spatio-temporal and cross-view consistency. Finally, we adopt neighboring 3D points as groups with constrained inner-distances within a coarse-to-fine 4D Gaussian Splatting (4DGS) framework to generate a continuous and stable 4D representation from multi-view videos. With these components, CharacterShot produces high-quality and consistent 4D character animation results aligned with the custom motion from 2D pose sequence across different views. Furthermore, to address the scarcity of character-centric 4D animation datasets, we construct a large-scale 4D dataset Character4D. Character4D contains 13,115 unique characters with varied appearances, building upon [75]. Each character undergoes rigging and motion retargeting with diverse 3D motion sequences, followed by multi-view rendering (up to 21 viewpoints), establishing large-scale character-centric 4D dataset specifically designed for 4D character animation.

Moreover, to address the lack of a benchmark for 4D character animation, we establish CharacterBench, a benchmark featuring diverse dynamic characters. Extensive qualitative and quantitative comparisons on CharacterBench demonstrate that our method, CharacterShot, outperforms existing state-of-the-art (SOTA) approaches and excels at generating spatial-temporal and spatial-view consistent 4D character animations conditioned on pose inputs. Additionally, ablation studies validate the effectiveness of our framework and highlight its superiority, offering valuable insights to the community. The contributions are summarized as follows:

- • To the best of our knowledge, CharacterShot is the first DiT-based 4D character animation framework capable of generating dynamic 3D characters from a single reference character image and a 2D pose sequence.
- • We propose a novel dual-attention module, which effectively ensuring spatial-temporal and spatial-view consistency in generating multi-view videos.
- • A novel neighbor-constrained 4DGS is proposed to enhance the robustness against outliers or noisy 3D points during 4D optimization, resulting in more continuous and stable 4D representations.
- • A large-scale character-centric dataset containing 13k characters with high-fidelity appearances rendered with varied motions and viewpoints for 4D character animation.
- • Extensive experiments demonstrate that CharacterShot has achieved SOTA performance compared to other methods.

- 2 Related Work

- 2.1 Character Animation

Recently, with the significant progress in image and video generation made by diffusion models [11, 12, 16, 26, 41, 42, 67, 97], numerous character animation methods [4, 9, 10, 18, 39, 40, 57, 64, 73, 96, 100] have exhibited remarkable performance. These works typically generate consistent animation results by using pose skeletons—extracted from off-the-shelf human pose detectors—as motion indicators, and further fine-tuning U-Net [54] or diffusion transformers (DiT) based [47] video generation models. In this paper, we build our CharacterShot on the powerful DiT-based image-to-video model CogVideoX [87] to enable higher-quality character animation.

- 2.2 3D Generation

Generating 3D content is essential and in high demand across realworld applications. Traditional methods typically rely on 3D supervision to learn 3D representations such as point clouds [23, 55], meshes [33, 77, 82], and neural radiance fields (NeRFs) [17, 20, 50, 68]. Recent works [6, 13, 27, 28, 43, 49, 56, 60, 62, 66, 76, 78, 83, 90, 99] borrow the prior information from 2D image diffusion models, using SDS loss [49] to optimize the 3D content from text or image. Other approaches [22, 25, 32, 34, 36, 37, 58, 59, 70, 72, 89] first generate multi-view images from diffusion models and then perform

- 3D reconstruction based on these views. In our work, we use the view images generated by a fine-tuned SV3D [70], as reference view images in the 4D generation stage.

2.3 4D Generation

Similar to 3D generation, many methods [3, 21, 30, 45, 52, 61, 91, 94, 98] utilize SDS-based optimization to generate 4D content by distilling pre-trained diffusion models in a 4D representation. However, optimizing SDS loss is often computationally intensive and timeconsuming. Another line of work [19, 35, 44, 46, 63, 81, 84, 86, 86, 94] fine-tunes diffusion models to generate multi-view videos and further optimize 4D content. These methods are limited to single-view video-driven generation and often struggle to effectively control the motion specified by the user. More recently, Human4DiT [57] introduces SMPL model [38] for all views to enable controllable multi-view video generation. However, the SMPL pipeline, which involves mesh vertex optimization and SMPL body rendering, is complex and computationally expensive, making it impractical for real-world applications. In contrast, CharacterShot is capable of generating spatial-temporal and spatial-view consistent 4D character animation results from just a single reference character and custom motion from a simple 2D pose sequence.

- 3 Method

Previous studies [81, 94] optimize 4D representations using singleview character video. However, generating this from a custom character image and corresponding motion control is complex and costly in real-world applications. To address this limitation, we propose CharacterShot, a novel framework that enables pose-controlled 4D character animation from a single reference character image with a 2D driving pose sequence. The overall framework of CharacterShot

Multi-View Images Multi-View Videos

2D Pose Sequence

|[Figure 37]|[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]|
|---|---|
|[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]|[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]|

[Figure 53]

[Figure 54]

[Figure 55]

View

|[Figure 56]|
|---|

Concat Add 3D-VAE Enc. & Dec.

|[Figure 57]<br><br>Reference|
|---|

[Figure 58]

| |Camera Parameters<br><br>|
|---|---|
| |𝐸,𝐾|

Camera Encoder

[Figure 59]

[Figure 60]

ViewGenerator

[Figure 61]

|Spatial-View Attn.| |
|---|---|
| | |

|Patchify|
|---|

|Gate| |
|---|---|
| | |

|Feedforward|
|---|

|Gate|
|---|

[Figure 62]

Feedforward

[Figure 63]

[Figure 64]

Patchify

Gate

Gate

| | |
|---|---|
|Spatial-Temporal Attn.| |

[Figure 65]

Dual-Attn.

Noisy Latent

DiT Block ×T

Frame

Fig. 2. Overview of CharacterShot. Given a reference character image and a 2D pose sequence as custom motion input, our framework generates multi-view videos with spatio-temporal and cross-view consistency. Next CharacterShot apply a neighbor-constrained 4DGS to generate 4D content.

is illustrated in Figure 2, including pose-controlled 2D character animation (Section 3.2), multi-view videos generation (Section 3.3), and neighbor-constrained 4DGS optimization (Section 3.4). We also introduce the foundational concepts of the DiT model and the detailed illustration of our proposed dataset, Character4D, in Section 3.1 and Section 3.5, respectively.

spatial attention view attention temporal attention

[Figure 66]

[Figure 67]

[Figure 68]

no attention no attention

implicit

implicit transmission

transmission

- 3.1 Preliminaries

In CharacterShot, we utilize a DiT-based image-to-video (I2V) model, CogVideoX [87], as the base model. It consists of a 3D Variational Autoencoder (3D VAE) [92], a T5 text encoder [51], and a denoising diffusion transformer [47]. CogVideoX fine-tunes a 3D VAE E to compress both the spatial and temporal information of the input video with the shape 4𝑓 × 8ℎ × 8𝑤 × 3 into a latent representation zi = E(I), where zi ∈ R𝑓 ×ℎ×𝑤×16. To enable I2V generation, a reference latent zr ∈ R1×ℎ×𝑤×16 is concatenated with zi along the channel dimension to form the final input z0 ∈ R𝑓 ×ℎ×𝑤×32, where zr will be derived from the latent padding of the reference image. After that, a patchify module is applied to convert the latent z0 into video tokens x0 ∈ R𝑓 ×(𝑛ℎ · 𝑤𝑛 )×𝐶, where 𝑛 = 2 denotes the patch size and 𝐶 = 3072 represents the output channel dimension. And the denoising diffusion transformer 𝜖𝜃 is trained by minimizing the Mean Squared Error (MSE) loss L at each time step 𝑡, as follows:

L = Ex𝑡,𝜖∼N(0,I),c,𝑡 ∥𝜖𝜃 (x𝑡, c,𝑡) − 𝜖𝑡 ∥2,

where x𝑡 is the noisy latent at time step 𝑡, and the gaussian noise 𝜖𝑡 is added to the video latent zi before the patchify module. c is the text condition.

- 3.2 Pose-Controlled Character Animation

view j frame a view i frame a view i frame b

Fig. 3. The separated spatial, temporal and view attention mechanisms are difficult to learn the implicit transmission across views and time.

zr and the corresponding pose latent zp′ of the reference image are concatenated to provide reference information as follows:

[Figure 69]

z0 = Concat [zr, zi], [zp′, zp] ,

where z0 ∈ R(𝑓 +1)×ℎ×𝑤×32. During training, we exclude the loss from the reference frame and only update the parameters of diffusion transformer. Moreover, to improve the model’s robustness to misaligned pose inputs during animation generation, we select the reference image and its corresponding pose image—originally taken from the first frame of the input video—with those from a randomly selected frame.

3.3 Multi-View Video Generation

CharacterShot aims to generate multi-view videos with the shape 𝑉 × (4𝑓 + 1) × 8ℎ × 8𝑤 × 3 for 4D optimization, where𝑉 represents the number of the target views. We first expand the input latent z0 from 2D pretraining stage with an additional view dimension:

z0 ∈ R𝑉×(𝑓 +1)×ℎ×𝑤×32, where the reference images are taken from different views of the same character at the same time, and the pose latent zp from a single view is concatenated across all views to enable more adaptive and robust controllable generation. Following SV4D [81], the multi-view images are generated by a view generator SV3D [70]. We fine-tune

To enable controllable generation on CogVideoX, we treat the pose information as an additional reference and perform 2D character animation pretraining as the base model for the next stage. Specifically, we utilize 3D VAE to compress pose sequence 𝑃 ∈ R4𝑓 ×8ℎ×8𝑤×3

into pose latent zp ∈ R𝑓 ×ℎ×𝑤×16. The pose latent zp is then concatenated with the video latent zi as a condition, and the reference latent

this view generator using our Character4D dataset to improve its performance to characters. Additionally, we encode the camera

prior 𝜋 = (𝐸𝑣,𝐾𝑣)𝑉𝑣=1 into a camera tokens xv and add it to the input tokens x0 ∈ R𝑉×(𝑓 +1)×(𝑛ℎ · 𝑤𝑛 )×𝐶 for a each specific view 𝑣:

ℎ 𝑛 ·

𝑥𝑣 = rearrange E𝑐(𝜙plücker(𝐸𝑣,𝐾𝑣)), (

𝑤 𝑛 ) ×𝐶 ,

where 𝐸𝑣 and 𝐾𝑣 represent the intrinsic and extrinsic parameters, respectively; 𝜙plücker denotes the Plücker embedding [14] with the shape 6 × 8ℎ × 8𝑤; and the camera encoder E𝑐 encodes the Plücker embedding derived from 𝐸𝑣 and 𝐾𝑣 into a feature map 𝐶 × 𝑛ℎ × 𝑤𝑛 .

Previous methods [81, 86] employ separated spatial, temporal and view attention mechanisms, which are ineffective to learn the implicit transmission of visual information [87], as shown in Figure 3. To address this, we introduce a dual-attention module that includes parallel 3D full attention blocks to model the coherent and consistent visual transmission across spatial-temporal and spatial-view correlations. As shown in Figure 2, we rearrange the tokens 𝑥0 with shapes 𝑉 × (𝑓 + 1) · 𝑛ℎ · 𝑤𝑛 × 𝐶 and (𝑓 + 1) × 𝑉 · 𝑛ℎ · 𝑤𝑛 × 𝐶 as the input to our dual-attention module. We continue training from the 2D pretraining model on our Character4D dataset and initialize the dual-attention module using the weights of its 3D full attention blocks. The synergy of these components enables CharacterShot to generate smooth, spatial-temporal and spatial-view consistent multi-view videos that follow the custom motion defined by the given pose sequences.

- 3.4 Neighbor-Constrained 4DGS Optimization

Afterobtainingmulti-viewvideos,we apply the neighbor-constrained 4D Gaussian Splatting (4DGS) to optimize the 4D representations. Specifically, we adopt a coarse-to-fine optimization framework followed [84] to model the 4D representations as deformable 3D Gaussians along the temporal axis, with each Gaussian 𝐺 at time 𝑡 is represented as:

### 𝐺𝑡 (X) = 𝐺 (X) + 𝐹 (𝛾(X), 𝛾(𝑡)) ,

where 𝐺(X) is the static 3D Gaussians. 𝐹 is a deformation function and 𝛾(·) is a positional encoding function [65].

In the coarse stage, we optimize the static 3D Gaussians 𝐺𝑇/2(X) using L1 loss at 𝑇/2-th frame, where 𝑇 denotes the number of frames, to quickly build the initial 4D space first. In the fine stage, we utilize a 4D progressive fitting [84] to gradually refine the deformable Gaussians at time 𝑡 with the grid-based total variation loss LTV [84] and image-space reconstruction losses L1 and LLPIPS from the entire multi-view videos. However, the synthesized multiview videos might have slight misalignments across views, which often lead to outliers and noisy 3D points during optimization. As shown in Figure 8, previous 4D methods [31, 79, 84, 85] results in suddenly disappear hands or visible artifacts. To address this, we introduce a novel neighbor constraint in the fine stage to enforce geometric consistency, which preserves the relative configuration between each 3D point and its neighboring points over time, promoting local deformations. Specifically, we calculate the distances

of each 3D point 𝑖 from the group center at frames 𝑡 and 𝑡 − 1 as:

∑︁

1 |N(𝑖)|

L𝑖𝑡 = u𝑖𝑡 −

u𝑡𝑗,

𝑗∈N(𝑖)

∑︁

1 |N(𝑖)|

L𝑖𝑡−1 = u𝑖𝑡−1 −

u𝑡𝑗−1,

𝑗∈N(𝑖)

whereN(𝑖) representstheneighborpoints. Theneighbor loss Lneighbor is then defined as:

𝑚𝑖 = ∥u𝑖𝑡 − u𝑖𝑡−1∥ > 𝜏, 𝑚𝑖𝑗 = 𝑚𝑖 ·𝑚𝑗,

Lneighbor = ∑︁

L𝑖𝑡 − L𝑖𝑡−1 2 · 𝑤𝑖𝑗 ·𝑚𝑖𝑗,

(𝑖,𝑗)∈𝐸

where 𝜏 is a predefined displacement threshold, 𝑚𝑖𝑗 is a binary gate that activates only when neighboring points turn into outliers or

noisy 3D points, and 𝑤𝑖𝑗 = ∥u𝑖𝑡−1 − u𝑡𝑗−1∥ is a spatial edge weight. The full loss function in fine stage can be defined as:

Lfine = 𝜆1 · L1 + 𝜆2 · LLPIPS + 𝜆3 · Lneighbor + 𝜆4 · LTV,

where the coefficients 𝜆1, 𝜆2, 𝜆3, and 𝜆4 are the corresponding weighting factors.

3.5 Character4D

Current 4D character datasets [7, 93] only include a very small variety of character types and motion types. To enable a more generalized 4D character animation, we construct a large-scale 4D character dataset by filtering high-quality characters from VRoidHub2 [71]—a platform for sharing and showcasing 3D character models—and collect a total of 13,115 characters in OBJ file format. First, we load the characters into Blender3, a widely used 3D modeling software, with an initial configuration: A-pose4 and a centered camera positioned at a fixed height, with the radius and field of view (FoV) set to 2.5 and 40◦, respectively. After that, we bind 40 diverse motions (e.g., dancing, singing, and jumping) in skeletons collected from Mixamo5 [1] to these characters, following the data curation pipeline used in previous methods [5, 48, 75]. Specifically, we assign one randomly selected motion to each character [75] using the automatic retargeting software Rokoko [2]. Binding motion using skeletons helps the clothing swing naturally with the movements, allowing the model to learn the principles of physical reality. Next, we generate 21 camera viewpoints along a horizontal static trajectory, following the setup used in SV3D [70]. Finally, we render frames of all characters from 21 viewpoints in the A-pose for view generator finetuning, and with various motions for diffusion transformer finetuning to generate spatial-temporal and spatial-view consistent multi-view videos from any reference character image and custom motion in pose sequence. Visual examples are shown in Appendix A.

2All the 3D avatars we used in our dataset clearly show the permission of usage in their individual websites. 3https://www.blender.org/

- 4A standard initial posture in which the character stands upright with arms slightly angled downward and outward, forming an "A" shape.
- 5An online platform by Adobe that provides automatic rigging and a large library of motions.

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

Pose Ground Truth CharacterShot SV3D SV4D Diffusion2

- Fig. 4. Visual comparison of multi-view videos synthesis. CharacterShot generates high-quality character videos with both spatial-temporal and multi-view consistency, faithfully preserving the reference character image and driving pose.

Table 1. Quantitative comparison of multi-view videos synthesis on CharacterBench. The best result is marked in bold.

Methods SSIM ↑ LPIPS ↓ CLIP-S ↑ FVD-F ↓ FVD-V ↓ FVD-D ↓ FV4D ↓ SV3D 0.873 0.241 0.864 1639.020 1471.051 1378.806 2078.984 Diffusion2 0.889 0.135 0.878 1198.645 1044.424 994.202 1392.323 SV4D 0.891 0.138 0.856 1280.620 1537.853 1467.422 1477.972 CharacterShot 0.967 0.021 0.957 469.677 489.963 388.797 490.457

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

Ground Truth (a)Baseline (b)+Camera Prior (c)+Dual Attn (d)w/ View Attn

- Fig. 5. Visualization from the baseline to variants incorporating different model components. 4 Experiments

CharacterBench. As with the dataset challenges faced by existing 4D generation methods, there is currently no character benchmark for evaluating 4D character animation. To address this, we introduce a new benchmark CharacterBench built from the test sets of Character4D, together with characters that are curated from Mixamo. Characters in the A-pose are used to assess the view generator’s performance, while characters with motion are used to evaluate the effectiveness of 4D character animation. To evaluate the generalization of CharacterShot, we also select characters that are out-ofCharacter4D, gathered additional examples from the Internet, and generated a suite of virtual characters using Flux [24], spanning 2D anime characters, real-world humans, and other distinct 3D models with diverse motions,

4.2 Comparison with SOTA Methods

Multi-View Videos Synthesis. As mentioned in Section 1, previous 4D generation models require single-view videos and are unable to be conditioned on custom motion such as pose sequences. To enable a fair comparison, we adopt a two-stage generation for these methods by fine-tuning the SOTA 2D character animation model MimicMotion [96] on our collected high-quality 2D pose-driving dataset to generate single-view videos based on each specified character and corresponding pose input. We then compare the proposed CharacterShot with SOTA single-view video-driven 4D generation methods, including SV3D [70], SV4D [81] and Diffusion2 [86]. We first present the qualitative comparison in Figure 4. It is evident that Diffusion2 and SV4D generate results with inconsistent poses across different views (see rows 1 and 3). Notably, all these baselines

- 4.1 Implementation Details

Evaluation Metrics. To verify the effectiveness of our Character4D in improving character-specific view generation, we follow the protocols of [34, 70, 82, 83] and use PSNR [29], SSIM [74], and LPIPS [95] to evaluate the quality and similarity between the generated view images and the ground-truth images from low-level. Also, CLIP-score (CLIP-S) and FID [15] are employed to evaluate high-level semantic consistency. For multi-view video generation and 4D optimization, we follow SV4D [81] and apply FV4D, FVD-F, FVD-V, and FVD-D to evaluate consistency across frames and views. Visual quality is further evaluated using CLIP-S, LPIPS, and SSIM metrics. More details are shown in Appendix A.

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

Pose Ground Truth CharacterShot STAG4D SC4D L4GM DG4D

Fig. 6. Visual comparison of 4D generation. CharacterShot outperforms other methods in terms of texture and detail. Table 2. Quantitative comparison of 4D generation on CharacterBench. The best result is marked in bold.

Methods SSIM ↑ LPIPS ↓ CLIP-S ↑ FVD-F ↓ FVD-V ↓ FVD-D ↓ FV4D ↓ STAG4D 0.915 0.082 0.904 966.979 876.033 817.523 970.241 SC4D 0.907 0.089 0.907 961.941 849.578 813.812 995.497 L4GM 0.907 0.091 0.892 1056.498 889.114 846.307 1042.443 DG4D 0.888 0.116 0.897 1006.051 1200.049 1171.713 1059.921 CharacterShot 0.971 0.025 0.959 368.235 289.279 271.886 406.624

Table 3. Quantitative experiments on model components. "w/ ViewAttention" indicates that we use separate view attention as a replacement for our spatial-view attention in dual-attention module.

in specific 9 views after 4D optimization, while the optimization stage for SV4D and Diffusion2 is not open source. As the qualitative comparison shown in Figure 6, we notice that the results of STAG4D and SC4D exhibit inconsistent shapes and textures (e.g., the left hand and clothing in row 1), while DG4D suffers from flickering artifacts. L4GM generates clearer details compared to these three SDS loss-based methods, but it has some black artifacts. In contrast, our CharacterShot generates consistent and continuous high-quality 4D contents by applying dual-attention module and neighbor-constrained 4DGS. The quantitative experiments in Table 2 further demonstrate that our method consistently outperforms the baselines across all metrics.

Methods SSIM ↑ LPIPS ↓ FVD-F ↓ FV4D ↓ Baseline 0.956 0.032 614.010 639.733 + Camera Prior 0.961 0.029 545.662 570.046 + Dual-Attention 0.967 0.021 469.677 490.457 w/ View-Attention 0.964 0.025 491.865 520.737

generate blurred or incorrect details in both the facial and body regions. Thanks to our proposed dual-attention module—which explicitly models both spatial-temporal and spatial-view consistency with camera priors—CharacterShot generates more coherent results with consistent, high-quality details across poses, frames and views. Quantitative results in Table 1 further verify the effectiveness of the proposed CharacterShot. Specifically, CharacterShot achieves the highest SSIM, LPIPS, and CLIP-S scores, demonstrating strong identity preservation and indicating superior image quality. Additionally, the proposed dual-attention module contributes to the best performance on FVD-F, FVD-V, FVD-D, and FV4D, highlighting its effectiveness in providing high-quality videos and maintaining spatial-temporal and spatial-view consistency. More results of unseen and out-of-Character4D test samples from Flux and Internet are presented in Section B.3 and Figure 10, Appendix.

4.3 Ablation Studies

Contribution Decomposition of Model Components. We finetune our pretrained 2D character animation model on Character4D and generate videos for each view separately as a single-view baseline, then investigate the impact of our proposed components in the following analysis. As shown in Figure 5(a), the baseline struggles to transform the pose sequence accurately across different viewpoints, leading to noticeable distortions. By incorporating the camera prior, the single-view model achieves more accurate viewpoint-aware pose alignment, resulting in more reasonable position (see Figure 5(b)). The visual results in Figure 5(c) effectively follow the reference’s appearance and pose, demonstrating the necessity of simultaneously generating multi-view videos and the effectiveness of our dual-attention module. Moreover, to further verify the important in modeling implicit spatial-view information—rather than treating

4D Generation. We also present the comparison between SOTA 4D generation methods, including STAG4D [94], SC4D [80], L4GM [53], and DG4D [52]—with our CharacterShot by rendering images

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

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

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

Hi3D

Grond Truth ChracterShot SV3D Zero123xl InstantMesh

- Fig. 7. Visual comparison of 3D multi-view image synthesis. Fine-tuning SV3D on the Character4D dataset, our view generator generates novel character views that are vivid and more detail-oriented.

view information separately—we compare in the spatial-view attention with a separate view attention. As shown in Figure 5 (c)(d), our dual-attention module with spatial-view attention achieves better performance, demonstrating its superiority in enhancing spatialview consistency. The experiments in Table 3 further support the observations from the visual results and demonstrate the effectiveness of each component in our proposed framework.

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

Pseudo GT CharacterShot 4DGaussians DG-Mesh Deformable-GS WR4D

[Figure 198]

[Figure 199]

- Fig. 8. Visual comparison of 4D optimization. “Pseudo GT” refers to the multi-view videos produced in the preceding stage.

- Table 4. Quantitative comparison of 4D optimization on CharacterBench. Ground truths are generated multi-view videos.

Methods SSIM ↑ LPIPS ↓ FVD-F ↓ FV4D ↓

4DGaussians 0.984 0.017 89.726 66.962 WR4D 0.985 0.015 80.651 59.509 Deformable-GS 0.979 0.025 194.451 198.861 DG-Mesh 0.980 0.023 154.596 168.652 CharacterShot 0.987 0.015 73.284 55.472

Experiments in Table 5 also highlights the necessity of the charactercentric dataset for multi-view images generation.

- Table 5. Experiments of view images generation on CharacterBench between SOTA methods and our fine-tuned view generator.

#### Methods PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ CLIP-S ↑

Hi3D 18.279 0.922 0.073 77.351 94.184 Zero123xl 15.704 0.889 0.112 78.855 93.149 InstantMesh 17.011 0.878 0.087 76.623 92.824 SV3D 17.340 0.906 0.153 103.543 88.499 CharacterShot 21.098 0.945 0.054 71.656 94.513

5 Conclusion

4DGS Optimization. To verify neighbor-constrained 4DGS’ effectiveness, we compare it with SOTA 4DGS methods 4DGaussians[79], WR4D[84], Deformable-GS[85] and DG-Mesh[31]. For a fair comparison, we optimize the 4D representations of these methods using our generated multi-view videos (as pseudo ground truth). As shown in Figure 8, sudden hand disappearance can be observed in the first row for 4DGaussians, Deformable-GS, and G-Mesh. In addition, outlier and noisy 3D points also result in blurring and artifacts on the face and body for these methods. In contrast, CharacterShot produces continuous and stable 4D content by applying the neighbor constraint. The quantitative results in Table 4 further validate the effectiveness of our proposed neighbor-constrained 4DGS method. Character Datasets. We evaluate the effectiveness of our proposed Character4D by comparing our fine-tuned view generator with the base model SV3D [70] and other SOTA methods such as Zero123XL [34], InstantMesh [82], and Hi3D [83]. Visualizations in Figure 7 demonstrate that our view generator achieves superior performance in preserving character details for different views—such as facial features, hair, and body structure—compared to other baselines.

In this work, we propose CharacterShot, a controllable and consistent 4D character animation framework that generates dynamic 3D characters from just a single reference image and a 2D pose sequence. By leveraging the powerful DiT-based I2V model CogVideoX, CharacterShot first constructs a pose-controlled 2D character animation. Subsequently, CharacterShot introduces a dual-attention module to model implicit visual transmission across views and time, along with a camera prior to help transform pose positions. Finally, a neighborconstrained 4DGS is employed to generate continuous and stable 4D representations. To further enhance character performance, we construct a large-scale dataset, Character4D, containing 13,115 highquality characters with corresponding diverse motions. Extensive experiments on our newly introduced benchmark, CharacterBench, demonstrate the advantages of our method in capturing character details and achieving both spatial-temporal and spatial-view consistency. We hope that CharacterShot, along with its models and datasets, will contribute valuable and affordable resources to any individual creator and researcher to advance 4D character animation.

References

- [1] [n.d.]. Mixamo. https://www.mixamo.com.
- [2] [n.d.]. rokoko. https://www.rokoko.com/.
- [3] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 2024. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7996–8006.
- [4] Caroline Chan, Shiry Ginosar, Tinghui Zhou, and Alexei A Efros. 2019. Everybody dance now. In Proceedings of the IEEE/CVF international conference on computer vision. 5933–5942.
- [5] Shuhong Chen, Kevin Zhang, Yichun Shi, Heng Wang, Yiheng Zhu, Guoxian Song, Sizhe An, Janus Kristjansson, Xiao Yang, and Matthias Zwicker. 2023. Panic-3d: Stylized single-view 3d reconstruction from portraits of anime characters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21068–21077.
- [6] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. 2024. Text-to-3d using gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21401–21412.
- [7] Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. 2023. Dnarendering: A diverse neural actor repository for high-fidelity human-centric rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19982–19993.
- [8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi.

2023. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13142–13153.

- [9] Mengyang Feng, Jinlin Liu, Kai Yu, Yuan Yao, Zheng Hui, Xiefan Guo, Xianhui Lin, Haolan Xue, Chen Shi, Xiaowen Li, et al. 2023. Dreamoving: A human video generation framework based on diffusion models. arXiv e-prints (2023), arXiv–2312.
- [10] Qijun Gan, Yi Ren, Chen Zhang, Zhenhui Ye, Pan Xie, Xiang Yin, Zehuan Yuan, Bingyue Peng, and Jianke Zhu. 2025. HumanDiT: Pose-Guided Diffusion Transformer for Long-form Human Motion Video Generation. arXiv preprint arXiv:2502.04847 (2025).
- [11] Junyao Gao, Yanchen Liu, Yanan Sun, Yinhao Tang, Yanhong Zeng, Kai Chen, and Cairong Zhao. 2024. Styleshot: A snapshot on any style. arXiv preprint arXiv:2407.01414 (2024).
- [12] Junyao Gao, Yanan Sun, Fei Shen, Xin Jiang, Zhening Xing, Kai Chen, and Cairong Zhao. 2025. Faceshot: Bring any character into life. arXiv preprint arXiv:2503.00740 (2025).
- [13] Pengsheng Guo, Hans Hao, Adam Caccavale, Zhongzheng Ren, Edward Zhang, Qi Shan, Aditya Sankar, Alexander G Schwing, Alex Colburn, and Fangchang Ma.

2023. StableDreamer: Taming Noisy Score Distillation Sampling for Text-to-3D. arXiv preprint arXiv:2312.02189 (2023).

- [14] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. [n. d.]. CameraCtrl: Enabling Camera Control for Video Diffusion Models. In The Thirteenth International Conference on Learning Representations.
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017).
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.
- [17] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).
- [18] Li Hu. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8153–8163.
- [19] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. 2024. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 634–644.
- [20] Hanwen Jiang, Zhenyu Jiang, Yue Zhao, and Qixing Huang. 2023. Leap: Liberate sparse-view 3d modeling from camera poses. arXiv preprint arXiv:2310.01410

(2023).

- [21] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. 2023. Consistent4d: Consistent 360 {\deg} dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848 (2023).
- [22] Animesh Karnewar, Niloy J Mitra, Andrea Vedaldi, and David Novotny. 2023. Holofusion: Towards photo-realistic 3d generative modeling. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 22976–22985.
- [23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.

2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans.

Graph. 42, 4 (2023), 139–1.

- [24] Black Forest Labs. 2024. FLUX: Official inference repository for FLUX.1 models. https://github.com/black-forest-labs/flux
- [25] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. 2023. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214 (2023).
- [26] Jiaxing Li, Hongbo Zhao, Yijun Wang, and Jianxin Lin. 2024. Towards photorealistic video colorization via gated color-guided image diffusion models. In Proceedings of the 32nd ACM International Conference on Multimedia. 10891– 10900.
- [27] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. 2023. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arXiv preprint arXiv:2310.02596 (2023).
- [28] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. 2024. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6517–6526.
- [29] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee.

2017. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 136–144.

- [30] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis.

2024. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8576–8588.

- [31] Isabella Liu, Hao Su, and Xiaolong Wang. 2024. Dynamic gaussians mesh: Consistent mesh reconstruction from monocular videos. arXiv preprint arXiv:2404.12379

(2024).

- [32] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. 2024. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10072–10083.
- [33] Minghua Liu, Chong Zeng, Xinyue Wei, Ruoxi Shi, Linghao Chen, Chao Xu, Mengqi Zhang, Zhaoning Wang, Xiaoshuai Zhang, Isabella Liu, et al. 2024. Meshformer: High-quality mesh generation with 3d-guided reconstruction model. arXiv preprint arXiv:2408.10198 (2024).
- [34] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision. 9298– 9309.
- [35] Tianqi Liu, Zihao Huang, Zhaoxi Chen, Guangcong Wang, Shoukang Hu, Liao Shen, Huiqiang Sun, Zhiguo Cao, Wei Li, and Ziwei Liu. 2025. Free4D: Tuningfree 4D Scene Generation with Spatial-Temporal Consistency. arXiv preprint arXiv:2503.20785 (2025).
- [36] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2023. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453 (2023).
- [37] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2024. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9970–9980.
- [38] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. 2023. SMPL: A skinned multi-person linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 851–866.
- [39] Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, Tianshu Hu, and Yongming Zhu. 2025. DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance. arXiv preprint arXiv:2504.01724 (2025).
- [40] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. 2024. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 4117–4125.
- [41] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021).
- [42] Alexander Quinn Nichol and Prafulla Dhariwal. 2021. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning. PMLR, 8162–8171.
- [43] Zijie Pan, Jiachen Lu, Xiatian Zhu, and Li Zhang. 2023. Enhancing highresolution 3d generation through pixel-wise gradient clipping. arXiv preprint arXiv:2310.12474 (2023).
- [44] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang. 2024. Fast dynamic 3d object generation from a single-view video. arXiv preprint arXiv:2401.08742 (2024).

- [45] Hui En Pang, Shuai Liu, Zhongang Cai, Lei Yang, Tianwei Zhang, and Ziwei Liu. 2024. Disco4D: Disentangled 4D Human Generation and Animation from a Single Image. arXiv preprint arXiv:2409.17280 (2024).
- [46] Jangho Park, Taesung Kwon, and Jong Chul Ye. 2025. Zero4D: Training-Free 4D Video Generation From Single Video Using Off-the-Shelf Video Diffusion Model. arXiv preprint arXiv:2503.22622 (2025).
- [47] William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.
- [48] Hao-Yang Peng, Jia-Peng Zhang, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu.

2024. CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Canonicalization. ACM Transactions on Graphics (TOG) 43, 4 (2024). https://doi.org/10.1145/3658217

- [49] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022).
- [50] Zefan Qu, Ke Xu, Gerhard Petrus Hancke, and Rynson WH Lau. 2024. LuShNeRF: Lighting up and Sharpening NeRFs for Low-light Scenes. arXiv preprint arXiv:2411.06757 (2024).
- [51] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.
- [52] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. 2023. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142 (2023).
- [53] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. 2024. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems 37 (2024), 56828–56858.
- [54] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.
- [55] Darius Rückert, Linus Franke, and Marc Stamminger. 2022. Adop: Approximate differentiable one-pixel point rendering. ACM Transactions on Graphics (ToG) 41, 4 (2022), 1–14.
- [56] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. 2023. Zeronvs: Zero-shot 360-degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994 (2023).
- [57] Ruizhi Shao, Youxin Pang, Zerong Zheng, Jingxiang Sun, and Yebin Liu. 2024. Human4dit: 360-degree human video generation with 4d diffusion transformer. arXiv preprint arXiv:2405.17405 (2024).
- [58] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110

(2023).

- [59] Yukai Shi, Jianan Wang, He Cao, Boshi Tang, Xianbiao Qi, Tianyu Yang, Yukun Huang, Shilong Liu, Lei Zhang, and Heung-Yeung Shum. 2023. Toss: Highquality text-guided novel view synthesis from a single image. arXiv preprint arXiv:2310.10644 (2023).
- [60] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512

(2023).

- [61] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al.

2023. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280

(2023).

- [62] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. 2023. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818 (2023).
- [63] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. 2024. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928 (2024).
- [64] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. 2024. Animate-x: Universal character image animation with enhanced motion representation. arXiv preprint arXiv:2410.10306 (2024).
- [65] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng.

2020. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems 33 (2020), 7537–7547.

- [66] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2023. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023).

- [67] Kexian Tang, Junyao Gao, Yanhong Zeng, Haodong Duan, Yanan Sun, Zhening Xing, Wenran Liu, Kaifeng Lyu, and Kai Chen. 2025. LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning? arXiv preprint arXiv:2503.19990

(2025).

- [68] Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao.

2024. Triposr: Fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151 (2024).

- [69] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. 2019. FVD: A new metric for video generation. (2019).
- [70] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. 2025. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision. Springer, 439–457.
- [71] VRoid. 2022. VRoid Hub. https://vroid.com/.
- [72] Peng Wang and Yichun Shi. 2023. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201 (2023).
- [73] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. 2025. UniAnimate-DiT: Human Image Animation with Large-Scale Video Diffusion Transformer. arXiv preprint arXiv:2504.11289

(2025).

- [74] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.
- [75] Zhenzhi Wang, Yixuan Li, Yanhong Zeng, Youqing Fang, Yuwei Guo, Wenran Liu, Jing Tan, Kai Chen, Tianfan Xue, Bo Dai, et al. 2024. HumanVid: Demystifying Training Data for Camera-controllable Human Image Animation. arXiv preprint arXiv:2407.17438 (2024).
- [76] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2024. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems 36 (2024).
- [77] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. 2024. MeshLRM: Large Reconstruction Model for High-Quality Meshes. arXiv preprint arXiv:2404.12385 (2024).
- [78] Haohan Weng, Tianyu Yang, Jianan Wang, Yu Li, Tong Zhang, CL Chen, and Lei Zhang. 2023. Consistent123: Improve consistency for one image to 3d object synthesis. arXiv preprint arXiv:2310.08092 (2023).
- [79] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 2024. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 20310–20320.
- [80] Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, and Xiang Bai.

2025. Sc4d: Sparse-controlled video-to-4d generation and motion transfer. In European Conference on Computer Vision. Springer, 361–379.

- [81] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani.

2024. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470 (2024).

- [82] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. 2024. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191 (2024).
- [83] Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng Chen, Chong-Wah Ngo, and Tao Mei. 2024. Hi3D: Pursuing High-Resolution Image-to-3D Generation with Video Diffusion Models. In Proceedings of the 32nd ACM International Conference on Multimedia. 6870–6879.
- [84] Ling Yang, Kaixin Zhu, Juanxi Tian, Bohan Zeng, Mingbao Lin, Hongjuan Pei, Wentao Zhang, and Shuicheng Yan. 2025. WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes. arXiv preprint arXiv:2503.13435 (2025).
- [85] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2024. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20331–20341.
- [86] Zeyu Yang, Zijie Pan, Chun Gu, and Li Zhang. 2024. Diffusion 2: Dynamic 3D Content Generation via Score Composition of Orthogonal Diffusion Models. arXiv preprint arXiv:2404.02148 (2024).
- [87] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024).
- [88] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. 2023. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4210–4220.
- [89] Jianglong Ye, Peng Wang, Kejie Li, Yichun Shi, and Heng Wang. 2024. Consistent1-to-3: Consistent image to 3d view synthesis via geometry-aware diffusion

- models. In 2024 International Conference on 3D Vision (3DV). IEEE, 664–674.
- [90] Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. 2023. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. arXiv preprint arXiv:2310.08529

(2023).

- [91] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 2023. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225 (2023).
- [92] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. 2023. Language Model Beats Diffusion–Tokenizer is Key to Visual Generation. arXiv preprint arXiv:2310.05737 (2023).
- [93] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. 2021. Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5746–5756.
- [94] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. 2025. Stag4d: Spatial-temporal anchored generative 4d gaussians. In European Conference on Computer Vision. Springer, 163–179.
- [95] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang.

2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

- [96] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. 2024. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680

(2024).

- [97] Hongbo Zhao, Jiaxing Li, Peiyi Zhang, Peng Xiao, Jianxin Lin, and Yijun Wang.

2025. ColorSurge: Bringing Vibrancy and Efficiency to Automatic Video Colorization via Dual-Branch Fusion. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–11.

- [98] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. 2023. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603 (2023).
- [99] Linqi Zhou, Andy Shih, Chenlin Meng, and Stefano Ermon. 2024. Dreampropeller: Supercharge text-to-3d generation with parallel sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4610– 4619.
- [100] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. 2024. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision. Springer, 145–162.

Appendix A Implementation Details

In the pose-controlled 2D character animation pretraining stage, we initialize our DiT model weights using the pretrained imageto-video model CogVideoX-I2V-5B [87]. The pretraining dataset comprises 21,000 dancing videos collected from the Internet, which are processed into 336,000 video clips, each containing 25 frames at a resolution of 480×720. Next, we apply the widely used pose detector DWpose [88] to extract pose images. We follow the full training script from CogVideoX, using a learning rate of 2e-5, and train this stage for 11,000 steps on eight A800 GPUs. In the second stage, we continue fine-tuning the model on Character4D with dual-attention module and a camera encoder, starting from the checkpoint obtained in the first stage. During training, we set𝑉 = 5 and randomly sample views from the view pool. This stage is trained for 1,500 steps on 16 A800 GPUs with a learning rate of 5e-5. We also fine-tune the view generator from SV3D using the Character4D dataset with A-pose, training for 20,000 iterations on eight A800 GPUs at a resolution of 768×768, with each sample consisting of 21 frames. Please note that the view-generator is a plugin component that allows us to seamlessly replace SV3D with any more powerful view-generator at no additional cost.

We finetune MimicMotion on our 2D pretrained dataset to improve its performance on characters, and we only update the parameters of temporal layers and pose guider at (lr=1e-4, batch size=8, gpus=8, resolution=1024, num frames=15, training steps=30000). For neighbor-constrained 4DGS, both the coarse stage and each progressive step [84] in the fine stage are trained for 3000 iterations. In the coarse stage, we select the video frame at time step𝑇/2 to optimize a static Gaussian representation. In the fine stage, we utilize the full multi-view video sequence for progressive optimization. For the Lneighbor , we define the local neighborhood of a point as its 20 nearest neighbors. For loss weighting, we set 𝜆2 = 0.01, while all other coefficients 𝜆1,3,4 = 1. The learning rate is 1.6e-4.

Character4D. Following the introduction of Character4D on Section 3.5, we provide the visual examples of our Character4D dataset in Figure 9. The top row shows the character in the A-pose, while the bottom row depicts the character performing a specific motion.

View0 View5 View12 View16

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Frame0 Frame10 Frame20 Frame30

[Figure 207]

- Fig. 9. A character sample from our Character4D dataset shown across four views and frames. Metrics. For FV4D, we compute the Fréchet Video Distance (FVD) [69] over all images, which are traversed in a bidirectional raster

pattern. In addition, we employ three specialized FVD variants to evaluate video coherence at a more granular level: FVD-F, which computes FVD across frames within each view; FVD-V, which computes FVD across views for each frame; and FVD-D, which computes FVD across the diagonal elements of the view–frame matrix. Specifically, we generate 21 views for evaluating the view generator. FV4D, FVD-F, FVD-V, and FVD-D are computed from a 9 × 9 multi-view video matrix, which consists of nine viewpoints and nine frames.

Table 6. Ablation study for our neighbor-constrained 4DGS.

## Methods SSIM ↑ LPIPS ↓ FVD-F ↓ FV4D ↓

w/o Binary Gate 0.987 0.015 78.218 57.284 w/o Neighbor Loss 0.986 0.017 83.421 61.324 Full setting 0.987 0.015 73.284 55.472

B Experiments

- B.1 Different Settings on 4D Optimization

In this subsection, we conduct an ablation study on our neighbor loss and its corresponding binary gate. As shown in Table 6, without the full neighbor loss leads to a notable drop in performance metrics, with FV4D and FVD-F suffering the most, showing over 10% degradation. Moreover, only removing the binary gate in the neighbor loss also results in performance degradation, whereas using the full setting achieves the best results across all metrics.

Table 7. Experiments on different typesof single-view video inputs for L4GM. "Original" and "Finetuned" refer to single-view video inputs generated using the original or finetuned MimicMotion models, respectively, while "GroundTruth" refers to the input ground-truth single-view video.

Methods SSIM ↑ LPIPS ↓ FVD-F ↓ FV4D ↓ Original 0.904 0.099 1198.655 1258.118 Finetuned 0.907 0.091 1056.498 1042.443 Ground-Truth 0.916 0.081 901.819 922.767 CharacterShot 0.971 0.025 368.235 406.624

- B.2 CharacterShot vs. Two-Stage 4D Generation

Experiments in Section 4.2 have demonstrated that CharacterShot significantly outperforms other single-view video-driven 4D generation methods [81, 86, 94]. To comprehensively explore the advantages of CharacterShot over existing two-stage 4D generation methods, we extend the single-view videos from the original MimicMotion and the ground truth for comparison. We conduct this ablation study on L4GM. As shown in Table 7, L4GM achieves better evaluation scores when given ground-truth single-view video as input. However, producing such high-quality and coherent single-view videos through 3D modeling or manual creation is time-consuming and labor-intensive. In contrast, CharacterShot achieves significantly superior performance using only a single reference character and a pose sequence, demonstrating its flexible and effective 4D character animation capability. We also observe that the finetuned MimicMotion outperforms the original model, although it still falls

short of the ground-truth videos, demonstrating the fairness of our comparison using the finetuned MimicMotion.

B.3 User Study on Out-of-Character4D Test Samples

To evaluate the CharacterShot’s generalize ability to characters that are out-of-Character4D (OOC), we construct a test set, which includes characters sourced from the Internet and Flux, spanning 2D anime characters, real-world humans, and other distinct 3D models with diverse motions, to compare CharacterShot with the 4D baselines. Since ground-truth multi-view videos aren’t available for these OOC characters, we conduct a user study with 30 volunteers to assess consistency in appearance, pose, time, and view in Table 8. CharacterShot generalize well to these OOC characters and motions, outperforming all baselines on the OOC test set.

Table 8. User Study on characters that are out-of-Character4D.

Methods Appearance ↑ Pose ↑ Time ↑ View ↑

SC4D 21.79 19.99 21.04 20.77 STAG4D 18.80 16.50 17.10 19.59 L4GM 12.22 17.76 17.41 12.29 DG4D 7.91 16.77 14.15 10.30

CharacterShot 39.24 29.01 30.33 37.05

B.4 Inference Cost

CharacterShot requires 20 or 40 minutes and 37 GB or 8 GB of VRAM to generate multi-view videos on a single H800 GPU, depending on whether CPU-offload is used. The 4DGS stage takes 30 minutes for optimization. While a standard CGI pipeline—including 3D modeling, motion capture, rigging, and more—typically takes several weeks, CharacterShot offers a low-cost CGI solution for individual creators on consumer-grade GPUs.

C Limitation

Although CharacterShot improves robustness to varied pose sequences through confidence-aware pose guidance, which uses the brightness of keypoints and limbs to encode pose-estimation confidence, animating with significantly inaccurate poses remains challenging, highlighting direction for future exploration.

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

[Figure 298]

##### Fig. 10. Visual results of multi-view videos generation for characters from Flux and Internet, which are out-of-Character4D.

