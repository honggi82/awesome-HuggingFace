## PERSE: Personalized 3D Generative Avatars from A Single Portrait

Hyunsoo Cha Inhee Lee Hanbyul Joo Seoul National University

{243stephen, ininin0516, hbjoo}@snu.ac.kr https://hyunsoocha.github.io/perse/

# arXiv:2412.21206v2[cs.CV]28Sep2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Reference Portrait Image

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

Render

[Figure 21]

[Figure 22]

[Figure 23]

PERSE Avatar

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

Generated Monocular Video

with Modified Attributes Facial Reenactment Interpolated Sample Generation

Figure 1. PERSE. Given a reference portrait image as input, our method constructs an animatable personalized 3D generative avatar with disentangled and editable control over various facial attributes such as beard, hair, and hat.

### Abstract

### 1. Introduction

A personalized 3D face avatar can represent each individual in VR/AR environments, replicating the user’s appearance and facial expressions. However, the exact replication of the appearance does not fully reflect real-world humans. In reality, people often change the attributes of their appearance, like hairstyles, or start growing a beard or mustache. Users may also wish to adjust their facial features in the virtual world, like the shape of their nose, eyebrows, or mouth, enhancing their desired look while preserving their core identity. While most prior avatar creation methods focus on building an exact digital twin of the person from images or video data [7, 16, 53, 55, 64, 78, 79, 81], the personalized avatar model with the generative ability to control and edit facial attributes remains underexplored.

We present PERSE, a method for building a personalized 3D generative avatar from a reference portrait. Our avatar enables facial attribute editing in a continuous and disentangled latent space to control each facial attribute, while preserving the individual’s identity. To achieve this, our method begins by synthesizing large-scale synthetic 2D video datasets, where each video contains consistent changes in facial expression and viewpoint, along with variations in a specific facial attribute from the original input. We propose a novel pipeline to produce high-quality, photorealistic 2D videos with facial attribute editing. Leveraging this synthetic attribute dataset, we present a personalized avatar creation method based on 3D Gaussian Splatting, learning a continuous and disentangled latent space for intuitive facial attribute manipulation. To enforce smooth transitions in this latent space, we introduce a latent space regularization technique by using interpolated 2D faces as supervision. Compared to previous approaches, we demonstrate that PERSE generates high-quality avatars with interpolated attributes while preserving the identity of the reference individual.

In this work, we present PERSE, a method to build an animatable personalized 3D generative avatar from a single reference portrait image. Our method goes beyond merely creating an exact twin from video inputs, introducing a novel approach that emphasizes flexibility and control over facial attributes, such as changing hairstyles or beards shown in Fig. 1. To build PERSE from a single reference portrait image, we generate a large-scale 2D monocular synthetic video

dataset of the reference identity, where each video has a variation in a specific facial attribute from the original input (e.g., a different hairstyle) driven by the face motion guidance, as shown in Fig. 5. Each video is also paired with a text prompt description addressing the changed attribute. To build this high-quality, photorealistic synthetic video dataset, we introduce a new pipeline that begins with synthesizing 2D images with attribute editing in a fully automated procedure. This is followed by a portrait animation process that leverages a combination of an existing pre-trained 2D portrait animation method [18] and our newly trained image-to-video model extending a prior work [80]. Notably, our synthetic video generation process is efficient, scalable, and provides significantly more attribute diversity by effectively synthesizing a thousand attribute videos compared to tens in prior work [5]. Using this synthetic video dataset, we train an avatar model with the continuous and disentangled attribute latent space.

To enhance the generative ability of our avatar model for unseen or interpolated attribute appearances, we also present a novel technique to enforce continuous and smooth latent space. To achieve this, we present a latent space regularization technique by using interpolated 2D face images from an image morphing technique [70] (e.g., synthesizing medium-length hair from short hair and long hair attributes), providing pseudo supervisions for the interpolated latent spaces. We show the efficacy of our regularization technique by producing novel and unseen attributes from interpolated latent spaces, as shown in Fig. 7. Furthermore, we present an efficient fine-tuning technique via Low-Rank Adaptation (LoRA) [26], to integrate any new facial attributes from in-the-wild images into our avatar model.

Our contributions are summarized as follows: (1) the first method to generate an animatable 3D personalized generative avatar from a reference portrait image with controllable facial attributes; (2) a method to generate high-quality synthetic 2D video datasets with diverse attribute editing from a reference portrait image; (3) latent space regularization by using face morphing supervision for continuous and smooth latent space to enhance the generative ability for unseen or interpolated attribute appearances; (4) an efficient finetuning technique via Low-Rank Adaptation (LoRA) [26] to integrate any new facial attribute into the avatar model.

- 2. Related Work 3D Facial Avatar Reconstruction. Since the introduction of foundational 3D Morphable Model [1] (3DMM), parametric 3D face models [1, 3, 39] have evolved to capture the diverse and dynamic nature of human faces, representing variations in shape, head pose, and facial expression through a set of parameters. Building on these models, various methods reconstruct 3D face avatars from single portrait images by estimating 3DMM parameters [10, 12, 54]. Recently, monocular 3D avatar reconstruction methods [7, 16, 36, 37, 78, 79] generate morphable photorealistic avatars leveraging advance-

ments in 3D representation [32, 42].

To move beyond single-subject avatars, the PEGASUS [5] reconstructs a personalized 3D generative avatar enabling control over facial attributes while preserving the reference identity, using synthetic DB. Similarly, HeadGAP [77] trains a generalizable prior model for 3D head avatar leveraging a large-scale multiview dataset and an avatar model with part-specific and point-specific feature codes. Despite advancements, constructing a unified 3D representation that can precisely capture and control all facial attributes remains challenging. To address this, disentangled or hybrid representations have been proposed, enabling selective modification of facial features or garments [13, 14, 35]. However, these approaches are limited by discrete 3D structures, restricting continuous interpolation capabilities. Recently, latent-conditioned generative models [5, 22, 34, 38] have been introduced to mitigate these constraints, yet they often lack the capacity for fine-grained editing and are confined to specific categories.

Smooth Image Morphing and Interpolation. Generating a plausible intermediate image between two pivot images has been widely studied within the context of image generative field [23, 30, 60]. The recent breakthroughs of diffusion model [23, 52, 61] improved the image interpolation methods to generate more plausible and better quality interpolated images with less limitation on categories [17, 56, 61, 62, 70, 73, 76]. Many diffusion-based interpolation methods follow the procedure of DDIM inversion [43, 60], interpolation in diffusion latent space, and DDIM forward sampling with slight modification. DiffMorpher [70] additionally utilizes personalized diffusion models finetuned on each pivot image with LoRA [26] to produce smooth interpolated sequences. SmoothDiffusion [17] finetunes diffusion models with LoRA [26] to preserve the distance of interpolated sample and pivots during denoising.

Portrait Animation from Single Image. Generating animations from a single image is a challenging task that has seen significant advancements through generative models, particularly based on implicit keypoints and diffusion methods.

Several approaches [25, 41, 58, 59, 63, 75] have introduced intermediate motion representations based on implicit keypoints estimation, enabling the mapping of a source portrait image to a driving image using optical flow. Extending the previous work [63], LivePortrait [18] enhances animation quality by integrating a GAN-based decoder [47], resulting in effective and controllable portrait animations.

Recent advancements in diffusion models have significantly enhanced portrait animation, offering improved control and realism. Several methods [6, 27, 65] have explored full-body animations guided by motion sequence drived from body keypoints. Building upon previous approaches [27], Champ [80] generates full-body animations guided by multiple reference videos such as SMPL [40] renderings.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

|Latent|
|---|

Rasterizer

EditedPortraits

[Figure 41]

[Figure 42]

[Figure 43]

Latent

Avatar Network (GS)

Encoder

| | |
|---|---|
| | |

Sapiens

Text Prompt

RGB Rendered Sequences

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

FLAME Params Same as Gudiance

[Figure 48]

[Figure 49]

[Figure 50]

Avatar Model

RGB/ Normal Encoder

[Figure 51]

ReferenceNet

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Guidance

RGB/ Normal Decoder

Denoising UNet

Portrait Animator

RGB Sequence for the target attribute

Figure 2. Overview of Synthetic Dataset Generation and Avatar Model Training. Starting with a collection of edited portrait images, we generate RGB videos for each target attribute using Portrait Animator. The guidance for the Portrait Animator is derived from tracked FLAME parameters of a predefined training motion sequence, which also serve as inputs to the avatar network in our avatar model. Using the generated RGB videos, we train our avatar model with a reconstruction loss. Each edited portrait is paired with the text prompt used for its generation. The process of creating these edited portraits based on text prompts is detailed in Sec. 4.2.

### 3. Preliminary: PEGASUS [5]

Our avatar model is based on a previously proposed personalized generative 3D avatar, PEGASUS [5] , by modifying its original 3D point cloud representation [79] into 3D Gaussian Splatting [32]. The PEGASUS avatar model is an animatable 3D avatar model of a reference individual with disentangled controls to selectively alter facial attributes such as hair or nose, while preserving the reference identity. The PEGASUS model takes a latent code z ∈ R(N

c+1)×d along with FLAME parameters β (shape), θ (pose), and ψ (expression) as inputs, and infers a colorized point cloud to express the target individual’s appearance, pose and expressions changes:

{xdi ,nd,ai} = Mϕ(xgci ,z,β,θ,ψ), (1)

where xdi is the 3D point locations, ni ∈ R3 is the point normals, and ai ∈ R3 is the point albedo colors. The input latent code z ∈ R(N

c+1)×d is a concatenation of Nc subpart latent codes {zj}j=0...Nc

, where each subpart latent code zj ∈ Rd controls specific aspects of the human identity or a subpart. The identity latent code z0 controls overall identity variations, and the other latent codes zj̸=0 control each subpart, preserving the identity defined by z0.

Notably, the PEGASUS model relies on constructing a synthetic video collection of the reference identity with edited facial attributes. This is performed by replacing specific facial attributes in the reference person’s video with those from multiple other individuals’ videos. Consequently, building the synthetic dataset requires not only the video of the reference individual but also numerous videos from other individuals for attribute variations. Moreover, this approach involves a time-intensive process of creating 3D avatars for each individual to synthesize all attribute variations, which limits the scalability of the method.

### 4. Method

We first describe our personalized 3D generative avatar model creation (Sec. 4.1), extending the previous work [5]. Then, we introduce our pipeline for generating a large-scale synthetic 2D facial attribute dataset (Sec. 4.2). Additionally, we present our novel training scheme including latent space regularization with interpolated 2D faces (Sec. 4.3), and also present our efficient fine-tuning technique to integrate arbitrary new attributes into our optimized latent space while preserving the existing distribution (Sec. 4.4).

#### 4.1. Personalized Generative Avatar Model

3D Gaussian Splatting for Avatar. Our avatar model builds on the structure of PEGASUS [5] with several modifications. First, we change the 3D representation of the avatar from a colorized point cloud to 3D Gaussian Splatting (3D-GS) [32] which enhances rendering quality. This is achieved by estimating 3D Gaussian parameters for each point, replacing the original point normal and albedo. Specifically, our model takes a latent code z and FLAME parameters {β,θ,ψ} as inputs, and infers 3D Gaussian parameters of posed avatar, including the 3D position xdi ∈ R3, rotation rdi ∈ R4, scale sdi ∈ R3, opacity odi ∈ R, and color ci ∈ R3 as follows:

{xdi ,rid,sdi ,odi ,cdi } = MΘ(xgci ,z,β,θ,ψ). (2)

To capture fine-grained deformations conditioned on head pose, we introduce an additional MLP deforming 3D Gaussians based on the input FLAME parameters {β,θ,ψ}, similar to MonoGaussianAvatar [7]. We densify the 3D Gaussians to capture fine detail using the upsampling strategy of prior work [5, 79] and prune distracting Gaussians through opacity resetting and thresholding as in the original 3D-GS framework [32]. By rasterizing the 3D Gaussians, we get a rendering of the avatar as follows:

ˆI = GSR {xdi ,rdi ,ssci ,odi ,csci }i∈{1···N} , (3) where GSR represents a 3D-GS Rasterizer [32].

CLIP-guided Latent Space Configuration. Following PEGASUS [5] model, we represent our avatar model latent code z ∈ RN

c×d as a concatenation of Nc subpart latent

codes {zj ∈ Rd}j=[1···Nc]. This part-wise separated latent configuration allows to control each facial attribute while

preserving other facial attributes. We can also selectively transfer the target attribute of the k-th subpart, such as hair, to the reference avatar by substituting the k-th subpart latent as follows:

###### zrefj if j ̸= k ztargetj if j = k

znewj =

(4)

To achieve this disentangled latent space, we optimize a single reference latent code zref ∈ RN

c×d, representing the identity of the input portrait image, along with a set of subpart latent codes {ztargetk ∈ Rd}, where each corresponds to a specific subject in our synthetic dataset.

However, directly optimizing these latent codes {ztargetk ∈ Rd} are prone to be overfitted on each subject, resulting in poor generalization to unseen subjects. To address this and achieve more compact latent space, we constrain latent codes using a well-established text-image feature model CLIP [50], which is a key difference over previous work [5]. We define the target subpart latent as an output of shallow MLP network conditioned on CLIP image and text features fI,fT ∈ R512:

zsubjectk = MLPz(fI,fT). (5) The CLIP features are calculated from front-view reference synthetic image and text pairs from our synthetic datasets. Additionally, we define zzero as a unique shared subpart latent code representing an empty subpart, such as the absence of a hat or beard.

- 4.2. Synthetic Dataset Generation We create a synthetic portrait video dataset with varying facial attributes from the input image of the reference individual to enable the generative ability for our 3D avatar model. Our synthetic dataset generation pipeline is performed via a two-stage process: generating attribute-edited images and animating the edited portrait images. Attribute-Edited Portrait Image Generation. Given a por-

trait image Iinput of the reference individual, our goal is to photo-realistically edit each attribute to reflect a different style. We consider 9 attribute categories: beard, clothes, earrings, eyebrows, hair, hat, headphones, mouth, and nose. To achieve this goal, we present a text-conditioned image inpainting pipeline by leveraging multiple tools including pre-trained 2D diffusion models [9]. We first determine a list of text prompts for each attribute category with specific adjectives (e.g., curly, straight, and wavy for hair). We leverage ChatGPT [46] to explore various possible distinctive

Reference

Fixed prompt: “A person with very shortcut hair”

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Sapiens Mask

FLUX Inpaint

DWPose Combine Mask

[Figure 65]

[Figure 66]

[Figure 67]

Sapiens Mask

FLUX Inpaint

SDXL T2I

2D Pose

Edited Portrait

Target prompt: “A person with curly auburn thick hair”

Figure 3. Image Synthesis. Starting from a reference portrait image, we present a fully automatic pipeline that generates an edited portrait without any manual manipulation such as user scribbles. To automatically generate the optimal mask image for inpainting, our method leverages SDXL, Sapiens, and FLUX [9, 33, 49].

adjectives. Then, for each text description T, we synthesize a corresponding portrait image with attribute changes using a text-conditioned inpainting model [9]:

Igen = I2Iinpaint(Iinput,T,Medit), (6)

where Medit denotes the mask region where the inpainting module needs to modify. Importantly, we find providing a suitable mask region Medit is essential to synthesizing photo-realistic output that adheres to the text guidance. A segmentation mask directly derived from the original input typically results in minor color changes without substantial shape variations.

To generate mask images Medit that are optimally aligned with a given text prompt T, we introduce a fully automatic image synthesis pipeline. Specifically, we synthesize a new portrait image Itext from the text T using a text-to-image diffusion model [49], where we enforce the facial poses and expressions of the synthesized image align with the Iinput using ControlNet [71]:

Itext = T2I T,C(Iinput) , (7)

where C(Iinput) is the OpenPose [4] keypoint image obtained by applying off-the-shelf keypoint estimator [68] on Iinput. Although the identity of Itext is not necessarily the same as Iinput, its facial pose is aligned to the Iinput, allowing us to obtain the attribute mask Medit accordingly. We extract the attribute mask Mtext from Itext using an off-the-shelf segmentation network [33] and use it as the target area to edit Medit = Mtext for Eq. (6). Examples are shown in the first column of Fig. 5.

For attributes of hat and hair, an additional step is required to remove the original parts that may unexpectedly remain after the inpainting process (e.g., the case that the original hair shape is bigger than Mtext). We resolve this issue by editing the original input image Minput with a version containing shortcut hair, denoted Mshortcut, before applying inpainting:

Ishortcut = I2Iinpaint(Iinput,Tshortcut,Minput), (8)

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

(a)Beard

(c)Hair

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

(b)Hat

Edited Image LivePortrait portrait-Champ Edited Image LivePortrait portrait-Champ

Figure 4. Comparison of LivePortrait and portrait-Champ. Examples from LivePortrait [18] and portrait-Champ demonstrate several limitations: (a) artifacts are visible in the hair region, (b) LivePortrait lacks adaptability to head poses involving hats, and (c) beard artifacts are prone to aliasing and disappearance.

where Minput is the hair mask of the Iinput, and Tshortcut is a corresponding text prompt: “A person with very shortcut hair”. We also find that, in these categories, combining the mask of this shortcut hair image Ishortcut with the mask from the text-to-image output Mtext produces superior results with fewer artifacts:

Medit = (Mshortcut ∪ Mtext). (9) See Fig. 3 for the overview of the editing pipeline.

Animated Portrait Video Generation. We animate each edited portrait image Igen to synthesize a video with varying head poses and facial expressions, which are used as a pseudo monocular video dataset for training our animatable 3D personalized generative avatar model.

To achieve this goal, we utilize two different portrait animation techniques, LivePortrait [18] and our customized face-specialized Champ [80]: portrait-Champ. These methods are chosen for their complementary strengths. The goal of both animators is the same producing a video output following the motion guidance while preserving the identity given by the input image:

Vgen = I2V(Igen,G), (10)

where G denotes a set of motion guidance, including the FLAME depth map, FLAME normal map, and 2D body and facial keypoints, as shown in the guidance at Fig. 2. To obtain G, we capture a short video with varied head poses and facial expressions, and apply a monocular face capture method [10] to extract FLAME parameters, from which we extract the motion guidance cues G. The same G is used for all generated videos, Vgen, resulting in a collection of videos with the same motions and diverse attribute changes. Examples are shown in Fig. 5.

For attribute categories excluding beard, earrings, hair, hat, and headphones, we use LivePortrait [18] to animate the edited portrait images. Although LivePortrait successfully generates high-quality face-animation videos, it performs suboptimally with certain attributes and conditions. For example, with portrait images featuring voluminous hair, long beards, or large hats, particularly in cases with extensive

|[Figure 80]|
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

EyebrowsClothesBeardInput

Hair

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

NoseMouthHat

Headphones

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

Earrings

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Edited Image

Animated Frames

Edited Image

Animated Frames

(a)

(b)

(a)

(b)

Figure 5. Our Synthetic Dataset. The upper left black bounded image is the input portrait. (a) is an edited image from the input portrait and (b) is a generated frame by the portrait animator.

head movements, LivePortrait model often generates unnatural deformations, such as stretching and shrinking with noticeable artifacts as shown in Fig. 4.

To address these limitations, we build and train our own alternative image-to-video diffusion model, portrait-Champ to leverage high temporal consistency of 2D video diffusion model [19, 80]. Our model shows superior performance for synthesizing beard, earrings, hair, hat, and headphones over LivePortrait [18], as demonstrated in our experiments. We build our model based on the Champ [80] that is originally designed for full-body animations, with a few extensions. For concise control of head and facial expression, our portrait-Champ inputs normal and depth rendering of EMOCA [10] as conditioning input. We add a normal channel in VAE encoder and decoder of portrait-Champ to enhance 3D-awareness of the video diffusion model [20], and trained it with 6k real-world videos capturing diverse identities and motions [69].

#### 4.3. Training

In essence, training our avatar model on the synthetic dataset is identical to the process of reconstructing a 3D avatar from real 2D video inputs. At each iteration, we render an image ˆI of a posed subject from the synthetic dataset and calculate the reconstruction loss, Lrecon, by comparing it to the ground truth image, I.

Lrecon(ˆI,I) = λL1||ˆI − I||1

+ λSSIMSSIM(ˆI,I) + λVGGVGG(ˆI,I)

(11)

We then compute latent regularization loss Lz enforcing the norm of the latent code close to be zero and estimated FLAME parameters regularizing loss LFLAME following PE-

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

α = 0.0 α = 0.3 α = 0.5 α = 0.7 α = 1.0

[Figure 116]

LatentA

Encoder

|InterpolatedLatent|
|---|

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

InterpolatedLatent

Rasterizer

- Text Prompt A

FLAME Params

Finetuned UNet

Encoder

- Text Prompt B

Avatar Network (GS)

α

weighted sum

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Rendered Images

[Figure 128]

LatentB

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

Synthetic Datasets Finetuning Inputs

Generated Images as Pseudo GT

- Figure 6. Overview of Supervision for Interpolation. We propose an additional training strategy that leverages a finetuned 2D diffusion model [26, 70] to enhance the quality of interpolated samples in latent space. Starting from two samples with text prompts A and B, we generate interpolated latent codes through weighted summation based on α. We then compute the part-wise loss and backpropagate it through the avatar model. GASUS [5]. Our total loss is as follows:

- 4.4. Facial Attribute Transfer from Image To transfer facial attribute from an arbitrary image, such as transferring an unseen hairstyle to the reference individual, we need to find the corresponding latent code in our model. Although our model can retrieve the latent code by inputting the CLIP [24] features of an image into our MLP as described in Eq. (5), it struggles with perfectly handling unseen attributes. To incorporate these unseen attributes while preserving learned ones, we finetune our avatar model by optimizing the weights ∆Θ of additional LoRA [26] layers while keeping the other network weights Θ frozen. Specifically, our model with additional LoRA layers takes the same inputs and outputs as described in Eq. (2):

{xdi ,rid,odi ,sdi ,cdi } = MΘ+∆Θ(xgci ,z,β,θ,ψ). (16)

We animate the image for transfer with our animation generation pipeline and use the resulting frames to optimize the LoRA layers. The loss is calculated only on the region targeted for transfer, using a masked loss similar to Eq. (15). Refer to the Supp. Mat. for more details.

- 5. Experiments

Ltot = λreconLrecon ˆI,I + λFLAMELFLAME + λzLz. (12)

We train our model with this objective until convergence. Fine-tuning for Interpolated Samples. After convergence, our avatar model still suffers from sampling high quality avatar which is not included in the trained subject. The sampled avatars frequently contain artifacts, such as floating Gaussians or unnatural color blobs as illustrated in Fig. 8. To mitigate these artifacts, we propose an interpolation regularization loss leveraging prior knowledge from a pretrained image diffusion model [52], as demonstrated in Fig. 6. By regularizing the interpolated renderings to be closer to image generated by the diffusion interpolation generator [70], we improve both the rendering quality and realism of interpolated samples.

To calculate the interpolation loss, we sample two pivot subjects, (a,b) from the same category in our synthetic dataset and render an interpolated subject in every iteration:

ˆIinterp,α = GSR MΘ(za(1 − α) + zbα) , (13)

where α denotes an interpolation weight. We use DiffMorpher [70] to generate semantically plausible and visually realistic interpolations between their images, controlled by the same interpolation weight α:

#### 5.1. Synthetic Dataset Configuration

To assess the effectiveness of our method, we generate a synthetic dataset using a single portrait for model evaluation. We define 9 attribute categories (beard, clothes, earrings, eyebrows, hair, hat, headphones, mouth, and nose) and produce over 50 videos for each, resulting in a total of 957 attributeedited videos for quantitative comparison. The text prompts are constructed from non-contradictory combinations of predefined, category-specific adjectives, such as curly, straight, wavy, and coily for hair. To animate the images, we employ a single 513-frame video that captures a variety of head poses and expressions, applying it consistently across all

Iinterp,α = DiffMorpherα Ia,Ib . (14)

As DiffMorpher [70] generated image Iinterp,α often fails to preserve identity, we apply loss only on the subpart region Mpart which alters during interpolation:

Linterp = Lrecon Mpart ◦ Iinterp,α,Mpart ◦ ˆIinterp,α . (15)

We finetune the converged avatar model together with total loss in Eq. (12) until it converges.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

OursCond.TAPEGASUSCond.SA

Fullw/oDiffMorpherFullw/oDiffMorpher

| |
|---|

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

(a)SupervisedSamples(b)UnsupervisedSamples

[Figure 157]

[Figure 158]

Color Artifacts Geometry Artifacts

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

Geometry Artifacts Geometry & Color Artifacts

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

α = 0.1 α = 0.3 α = 0.5 α = 0.7 α = 0.9

- Figure 7. Interpolation Comparison on Baselines. Our method shows better interpolation smoothness and less artifact on interpolated samples, particularly on the texture and color of hair.

Pivot A Pivot B

Interpolated Pivots

Figure 8. Effect of Interpolation Loss. (a) and (b) represent supervised and unsupervised samples respectively supervised by a personalized diffusion model [26, 70]. Even for unsupervised samples, our supervision method for interpolated samples mitigates unnatural artifacts and textures. Additionally, our method preserves the quality of the pivot samples.

instances. We split all video frames in our synthetic dataset into training and test sets with a 400:113 frame ratio, using the first 400 frames for training and the remaining 113 for evaluation. Examples of our dataset can be found in Fig. 5.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

#### 5.2. Baselines and Metrics

performance. Following standard practices in monocular 3D avatar reconstruction [55, 79, 81], we use peak signalto-noise ratio (PSNR), structural similarity (SSIM), and perceptual similarity (LPIPS) [72] to evaluate reconstruction performance of learned subjects in synthetic dataset. Additionally, we evaluate identity preservation by computing the cosine similarity of ArcFace [11] identity features.

We compare our model with three different baselines, each using a distinct 3D representation for avatar modeling: colorized point clouds [79], NeRF [42], and 3D Gaussians [32]. PEGASUS [5] is the first method for constructing a personalized 3D generative avatar from 2D monocular video inputs. It creates a personalized avatar model using a set of MLP networks and a colorized point cloud, following the approach of PointAvatar [79]. For a fair comparison, we train PEGASUS with its public code, replacing its synthetic database with our synthetic datasets.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

We compute the Fr´echet Inception Distance (FID) [21] and Kernel Inception Distance (KID) against FFHQ dataset [29] and our synthetic evaluation dataset to assess the quality of generated subjects. In addition, we compute the sum (Perceptual Path Length, PPL) and deviation (Perceptual Distance Variance, PDV) of perceptual loss between adjacent interpolated images to evaluate the smoothness of interpolation following DiffMorpher [57, 70].

Conditional INSTA (Cond.TA) is a modified version of vanilla INSTA [81], which reconstructs head avatars using an implicit representation, specifically iNGP [45]. To enable the model to capture diverse facial attributes, we add latent code conditioning the MLP of vanilla INSTA. We follow the PEGASUS latent code configuration and train Cond.TA with our synthetic dataset until it converges.

#### 5.3. Quantitative and Qualitative Results

We present the quantitative results of unseen head pose and facial expression rendering in Tab. 1. As shown in the table, our avatar model achieves the best results across all metrics, demonstrating superior reconstruction quality for the subjects in the synthetic dataset while preserving the identity.

Conditional SplattingAvatar (Cond.SA) is a modified SplattingAvatar [55] which is a method for reconstructing 3D avatar models from monocular video using 3D Gaussian Splatting [32]. Vanilla SplattingAvatar explicitly represents an avatar as a set of 3D Gaussians embedded on a 3D head mesh. To incorporate conditional latent code as input, we add an implicit network estimating changes of the 3D Gaussian parameters conditioned by the latent code. Similar to other baselines, we train the model until convergence using our synthetic dataset. See the Supp. Mat. for more details.

In Tab. 2, we provide additional quantitative comparisons on interpolation, along with qualitative comparisons in Fig. 7. Our avatar model outperforms baselines on both FIDFFHQ and KIDFFHQ scores, indicating that our interpolated samples align more closely with real human distribution in the FFHQ dataset. Additionally, our model achieves better FIDSYN and KIDSYN scores, confirming that our interpolated samples preserve the identity of the reference individual more effectively

Metrics. We evaluate our personalized generative model in two aspects: reconstruction performance and generative

Method PSNR↑ SSIM↑ LPIPS↓ Identity↑

PEGASUS [5] 23.56 0.8661 0.1508 0.6471 Cond.TA [81] 19.01 0.7730 0.2875 0.3022 Cond.SA [55] 22.17 0.8690 0.2760 0.4759

Ours 23.84 0.8852 0.1458 0.7059

- Table 1. Quantitative Results of Unseen Pose Renderings. We compare our method with the baselines for training accuracy of pivots in our synthetic dataset. Our method achieves the best results across all metrics, demonstrating superior accuracy in reconstructing samples in our synthetic dataset while preserving identity.

Method FIDFFHQ↓ KIDFFHQ↓ FIDsyn↓ KIDsyn↓ PDV∗↓ PPL↓ User Study↑ PEGASUS [5] 223.86 0.2502 84.94 0.0959 0.2373 0.4047 36.3 Cond.TA [81] 258.48 0.3015 127.45 0.1454 0.9724 0.6739 Cond.SA [55] 230.21 0.2551 180.79 0.2316 0.9641 0.5789 Ours 214.46 0.2201 57.78 0.0420 0.2481 0.3308 63.7

- Table 2. Quantitative Results of Interpolated Renderings. PDV*=100×PDV. Ours shows the best score among the baselines including user study except for PDV.

Method Interpolation w/ Linterp w/ CLIP FIDFFHQ↓ KIDFFHQ↓ FIDsyn↓ KIDsyn↓ PDV∗↓ PPL↓ Identity↑ 224.00 0.2357 66.34 0.0546 0.3046 0.3387 0.7001

✓ 224.67 0.2335 69.03 0.0568 0.3037 0.3268 0.66724

✓ ✓ 214.46 0.2201 57.78 0.0420 0.2481 0.3308 0.7013

- Table 3. Ablation Studies. PDV*=100×PDV. “w/ Linterp” denotes fine-tuning model with interpolation loss and “w/ CLIP” means using latent conditioned on CLIP feature. Our full method achieves the best results on all metric except for PPL.

than the baselines.

While PEGASUS [5] achieves slightly better performance on the PDV metric with a small gap, its lower FID, KID, and PPL scores suggest limited naturalness and smoothness in interpolation. It can be checked in Fig. 7, where PEGASUS shows unnatural transitions in hair color and texture, while ours produces smoother results. Moreover, in user studies, our interpolation results are preferred over PEGASUS.

#### 5.4. Ablation Studies and More Results

Ablation Studies. We conduct ablation studies to assess the effectiveness of our CLIP-guided latent configuration and interpolation loss Linterp. As shown in Tab. 3 and Fig. 8, our interpolation loss is essential for improving interpolated sample quality and reducing artifacts. The CLIP-guided latent also reduces PPL, resulting in smoother transitions while preserving rendering quality.

Facial Attribute Transfer. We conduct facial attribute transfer experiments using a few in-the-wild images. As shown in Fig. 9, our LoRA fine-tuning method successfully transfers the hair and hat attributes while preserving other aspects of identity. The transferred attributes are well integrated into the latent space, as reflected in the smooth interpolation results between subject in our synthetic dataset in Fig. 10

- (a) Input Image

Initial Results by Zero-Shot

(b) (c) Optimized Results with Reenactment

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

- Figure 9. Transferred Facial Attribute Results from In-TheWild Images. (a) is an in-the-wild image of attribute to transfer,

(b) is an initial transferred result without optimization, and (c) is optimized results using LoRA layers.

[Figure 209]

(a) Input Image

[Figure 210]

Synthetic Sample

(b) Interpolation Results (c)

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

- Figure 10. Transferred Facial Attribute Interpolation. (a) represents an in-the-wild input image, (b) denotes the interpolation result between (c), a sample of our synthetic dataset.

### 6. Discussion

We present PERSE, an animatable 3D personalized generative avatar from a single portrait image, enabling continuous and disentangled facial attribute editing while preserving the individual’s identity. To achieve this goal, we present several key contributions, including: (1) a method to generate highquality synthetic attribute video datasets from a single image along with our newly trained portrait-Champ model; (2) latent space regularization for unseen or interpolated attribute appearances; and (3) an efficient fine-tuning technique via LoRA to integrate new facial attribute into the avatar model.

As limitations, our avatar-building process is computationally intensive, requiring approximately 1.5 days on eight RTX A6000 GPUs for each new identity. Additionally, while our 3D avatars are of high quality, they do not yet achieve photorealism, particularly in fine hair strand details.

Acknowledgments. We thank Byungjun Kim for his helpful discussions and advice. This work was supported by NRF grant funded by the Korean government (MSIT) (No. 2022R1A2C2092724), and IITP grant funded by the Korea government (MSIT) [No. RS-2024-00439854, No. RS-2021-II211343, and No.2022-0-00156]. H. Joo is the corresponding author.

### References

- [1] V. Blanz and T. Vetter. A morphable model for the synthesis of 3d faces. In SIGGRAPH, 1999. 2
- [2] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127,

2023. 17

- [3] C. Cao, Y. Weng, S. Zhou, Y. Tong, and K. Zhou. Facewarehouse: A 3d facial expression database for visual computing. IEEE Transactions on Visualization and Computer Graphics,

2013. 2

- [4] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh. Realtime multiperson 2d pose estimation using part affinity fields. In ICCV,

2017. 4

- [5] H. Cha, B. Kim, and H. Joo. Pegasus: Personalized generative 3d avatars with composable attributes. In CVPR, 2024. 2, 3, 4, 6, 7, 8, 12, 15, 16
- [6] D. Chang, Y. Shi, Q. Gao, H. Xu, J. Fu, G. Song, Q. Yan, Y. Zhu, X. Yang, and M. Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identityaware diffusion. In ICML, 2023. 2
- [7] Y. Chen, L. Wang, Q. Li, H. Xiao, S. Zhang, H. Yao, and Y. Liu. Monogaussianavatar: Monocular gaussian point-based head avatar. In SIGGRAPH, 2024. 1, 2, 3, 12, 13
- [8] CloudResearch. Connect cloud research. URL https:// connect.cloudresearch.com/researcher/. 16
- [9] A. Creative. Flux.1-dev-controlnet-inpainting-alpha. https: //huggingface.co/alimama-creative/FLUX. 1-dev-Controlnet-Inpainting-Alpha, 2024. 4, 14
- [10] R. Danˇeˇcek, M. J. Black, and T. Bolkart. Emoca: Emotion driven monocular face capture and animation. In CVPR, 2022. 2, 5, 14, 15
- [11] J. Deng, J. Guo, N. Xue, and S. Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 7, 16
- [12] Y. Feng, H. Feng, M. J. Black, and T. Bolkart. Learning an animatable detailed 3d face model from in-the-wild images. ACM TOG, 2021. 2
- [13] Y. Feng, J. Yang, M. Pollefeys, M. J. Black, and T. Bolkart. Capturing and animation of body and clothing from monocular video. In SIGGRAPH ASIA, 2022. 2
- [14] Y. Feng, W. Liu, T. Bolkart, J. Yang, M. Pollefeys, and M. J. Black. Learning disentangled avatars with hybrid 3d representations. arXiv preprint arXiv:2309.06441, 2023. 2
- [15] Freepik. Portrait search results. https://www.freepik. com/search?ai=excluded&query=Portrait. Ac-

- cessed: 2024-11-22. 17
- [16] G. Gafni, J. Thies, M. Zollhofer, and M. Nießner. Dynamic neural radiance fields for monocular 4d facial avatar reconstruction. In CVPR, 2021. 1, 2
- [17] J. Guo, X. Xu, Y. Pu, Z. Ni, C. Wang, M. Vasu, S. Song, G. Huang, and H. Shi. Smooth diffusion: Crafting smooth latent spaces in diffusion models. In CVPR, 2024. 2
- [18] J. Guo, D. Zhang, X. Liu, Z. Zhong, Y. Zhang, P. Wan, and D. Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168, 2024. 2, 5, 14, 15, 27
- [19] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In ICLR, 2024. 5
- [20] X. He, X. Li, D. Kang, J. Ye, C. Zhang, L. Chen, X. Gao, H. Zhang, Z. Wu, and H. Zhuang. Magicman: Generative novel view synthesis of humans with 3d-aware diffusion and iterative refinement. arXiv preprint arXiv:2408.14211, 2024. 5, 14
- [21] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 7
- [22] H.-I. Ho, L. Xue, J. Song, and O. Hilliges. Learning locally editable virtual humans. In CVPR, 2023. 2
- [23] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [24] F. Hong, M. Zhang, L. Pan, Z. Cai, L. Yang, and Z. Liu. Avatarclip: Zero-shot text-driven generation and animation of 3d avatars. arXiv preprint arXiv:2205.08535, 2022. 6
- [25] F.-T. Hong, L. Zhang, L. Shen, and D. Xu. Depth-aware generative adversarial network for talking head video generation. In CVPR, 2022. 2
- [26] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 6, 7, 13, 15
- [27] L. Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In CVPR, 2024. 2, 16, 17
- [28] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 16
- [29] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In CVPR,

2019. 7, 16, 17

- [30] T. Karras, M. Aittala, S. Laine, E. H¨ark¨onen, J. Hellsten, J. Lehtinen, and T. Aila. Alias-free generative adversarial networks. In NeurIPS, 2021. 2
- [31] Z. Ke, J. Sun, K. Li, Q. Yan, and R. W. Lau. Modnet: Realtime trimap-free portrait matting via objective decomposition. In AAAI, 2022. 16
- [32] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 2023. 2, 3, 4, 7, 13
- [33] R. Khirodkar, T. Bagautdinov, J. Martinez, S. Zhaoen, A. James, P. Selednik, S. Anderson, and S. Saito. Sapiens:

- Foundation for human vision models. In ECCV, 2025. 4, 15
- [34] T. Kim, S. Saito, and H. Joo. Ncho: Unsupervised learning for neural 3d composition of humans and objects. In ICCV,

2023. 2

- [35] T. Kim, B. Kim, S. Saito, and H. Joo. Gala: Generating animatable layered assets from a single scan. In CVPR, 2024. 2
- [36] T. Kirschstein, S. Giebenhain, and M. Nießner. Diffusionavatars: Deferred diffusion for high-fidelity 3d head avatars. In CVPR, 2024. 2
- [37] I. Lee, B. Kim, and H. Joo. Guess the unseen: Dynamic 3d scene reconstruction from partial 2d glimpses. In CVPR,

2024. 2

- [38] J. Li, S. Saito, T. Simon, S. Lombardi, H. Li, and J. Saragih. Megane: Morphable eyeglass and avatar network. In CVPR,

2023. 2

- [39] T. Li, T. Bolkart, M. J. Black, H. Li, and J. Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 2017. 2, 12, 13, 14
- [40] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black. SMPL: A skinned multi-person linear model. SIGGRAPH ASIA, 2015. 2, 14
- [41] A. Mallya, T.-C. Wang, and M.-Y. Liu. Implicit warping for animation with image sets. In NeurIPS, 2022. 2
- [42] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 2, 7
- [43] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. CohenOr. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 2
- [44] MooreThreads. Moore-animateanyone. https : / / github . com / MooreThreads / Moore AnimateAnyone, 2023. 16, 17
- [45] T. M¨uller, A. Evans, C. Schied, and A. Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 2022. 7
- [46] OpenAI. Chatgpt, 2024. URL https://chat.openai. com/. 4
- [47] T. Park, M.-Y. Liu, T.-C. Wang, and J.-Y. Zhu. Semantic image synthesis with spatially-adaptive normalization. In CVPR, 2019. 2
- [48] W. Peebles and S. Xie. Scalable diffusion models with transformers. In ICCV, 2023. 17
- [49] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 4, 14
- [50] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In Proc. ICML, 2021. 4
- [51] N. Ravi, J. Reizenstein, D. Novotny, T. Gordon, W.-Y. Lo, J. Johnson, and G. Gkioxari. Accelerating 3d deep learning with pytorch3d. arXiv:2007.08501, 2020. 15
- [52] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 6, 14

- [53] S. Saito, G. Schwartz, T. Simon, J. Li, and G. Nam. Relightable gaussian codec avatars. In CVPR, 2024. 1
- [54] S. Sanyal, T. Bolkart, H. Feng, and M. J. Black. Learning to regress 3d face shape and expression from an image without 3d supervision. In CVPR, 2019. 2
- [55] Z. Shao, Z. Wang, Z. Li, D. Wang, X. Lin, Y. Zhang, M. Fan, and Z. Wang. Splattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting. In CVPR,

2024. 1, 7, 8, 15, 16

- [56] L. Shen, T. Liu, H. Sun, X. Ye, B. Li, J. Zhang, and Z. Cao. Dreammover: Leveraging the prior of diffusion models for image interpolation with large motion. arXiv preprint arXiv:2409.09605, 2024. 2
- [57] K. Shoemake. Animating rotation with quaternion curves. In SIGGRAPH, 1985. 7, 13
- [58] A. Siarohin, S. Lathuili`ere, S. Tulyakov, E. Ricci, and N. Sebe. First order motion model for image animation. In NeurIPS,

2019. 2

- [59] A. Siarohin, O. J. Woodford, J. Ren, M. Chai, and S. Tulyakov. Motion representations for articulated animation. In CVPR,

2021. 2

- [60] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In Proc. ICLR, 2020. 2, 13
- [61] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. 2
- [62] C. Wang and P. Golland. Interpolating between images with diffusion models. 2023. 2
- [63] T.-C. Wang, A. Mallya, and M.-Y. Liu. One-shot free-view neural talking-head synthesis for video conferencing. In CVPR, 2021. 2
- [64] Y. Xu, H. Zhang, L. Wang, X. Zhao, H. Huang, G. Qi, and Y. Liu. Latentavatar: Learning latent expression code for expressive neural head avatar. In SIGGRAPH, 2023. 1
- [65] Z. Xu, J. Zhang, J. H. Liew, H. Yan, J.-W. Liu, C. Zhang, J. Feng, and M. Z. Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In CVPR,

2024. 2

- [66] S. Yang, H. Li, J. Wu, M. Jing, L. Li, R. Ji, J. Liang, and H. Fan. Megactor: Harness the power of raw video for vivid portrait animation. arXiv preprint arXiv:2405.20851, 2024. 16
- [67] S. Yang, H. Li, J. Wu, M. Jing, L. Li, R. Ji, J. Liang, H. Fan, and J. Wang. Megactor-σ: Unlocking flexible mixed-modal control in portrait animation with diffusion transformer. arXiv preprint arXiv:2408.14975, 2024. 16
- [68] Z. Yang, A. Zeng, C. Yuan, and Y. Li. Effective whole-body pose estimation with two-stages distillation. In ICCV, 2023.

- 4, 14, 15

[69] J. Yu, H. Zhu, L. Jiang, C. C. Loy, W. Cai, and W. Wu. Celebvtext: A large-scale facial text-video dataset. In CVPR, 2023.

- 5, 14, 17

- [70] K. Zhang, Y. Zhou, X. Xu, B. Dai, and X. Pan. Diffmorpher: Unleashing the capability of diffusion models for image morphing. In CVPR, 2024. 2, 6, 7, 13, 14
- [71] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 4, 14
- [72] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang.

- The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 7
- [73] R. Zhang, Y. Chen, Y. Liu, W. Wang, X. Wen, and H. Wang. Tvg: A training-free transition video generation method with diffusion models. arXiv preprint arXiv:2408.13413, 2024. 2
- [74] Y. Zhang, J. Gu, L.-W. Wang, H. Wang, J. Cheng, Y. Zhu, and F. Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680, 2024. 16, 17
- [75] J. Zhao and H. Zhang. Thin-plate spline motion model for image animation. In CVPR, 2022. 2
- [76] P. Zheng, Y. Zhang, Z. Fang, T. Liu, D. Lian, and B. Han. Noisediffusion: Correcting noise for image interpolation with diffusion models beyond spherical linear interpolation. In ICLR, 2024. 2
- [77] X. Zheng, C. Wen, Z. Li, W. Zhang, Z. Su, X. Chang, Y. Zhao, Z. Lv, X. Zhang, Y. Zhang, et al. Headgap: Few-shot 3d head avatar via generalizable gaussian priors. arXiv preprint arXiv:2408.06019, 2024. 2
- [78] Y. Zheng, V. F. Abrevaya, M. C. B¨uhler, X. Chen, M. J. Black, and O. Hilliges. Im avatar: Implicit morphable head avatars from videos. In CVPR, 2022. 1, 2
- [79] Y. Zheng, W. Yifan, G. Wetzstein, M. J. Black, and O. Hilliges. Pointavatar: Deformable point-based head avatars from videos. In CVPR, 2023. 1, 2, 3, 7, 12, 13
- [80] S. Zhu, J. L. Chen, Z. Dai, Y. Xu, X. Cao, Y. Yao, H. Zhu, and S. Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. arXiv preprint arXiv:2403.14781, 2024. 2, 5, 14
- [81] W. Zielonka, T. Bolkart, and J. Thies. Instant volumetric head avatars. In CVPR, 2023. 1, 7, 8

## PERSE: Personalized 3D Generative Avatars from A Single Portrait Supplementary Material

### A. Implementation Details

#### A.1. Avatar Model

[Figure 233]

512

512

512

512

512

512

512

512

512512512512

[Figure 234]

Sig 1

3

##### A.1.1. Avatar Model Architecture

[Figure 235]

To model diverse attributes with a single model, our avatar model follows three-stage deformations proposed in PEGASUS [5] with a few modifications. First, we initialize the learnable generic canonical points Pgc with the vertices of a FLAME [39] mesh with an open mouth:

Sig

[Figure 236]

[Figure 237]

128

128

128

Sig 1

[Figure 238]

3

[Figure 239]

[Figure 240]

- 3

Sig

- 4

Pgc = {xgci }i={1···N}, (17) where N is the number of points. The generical canonical points Pgc are shared start points for all subjects in the synthetic dataset. By deforming the points with subject-specific latent z as a condition, we obtain subject-specific canonical points Psc containing the shape of a specific attribute, such as having long hair or grey cap. The mapping between two states is defined as an offset of each point Oigc→sc, which is regressed using coordinate-based deformation MLP as follows:

[Figure 241]

Sig

[Figure 242]

- 3

Sig

- 4

[Figure 243]

[Figure 244]

(b) Canonical

(a) Pose-conditioned Deformation

[Figure 245]

[Figure 246]

Softmax

[Figure 247]

128

128

128

128

[Figure 248]

5

[Figure 249]

{Oigc→sc,Oisc→fc,Ei,Pi,Wi} = MLPd(z,xgci ). (18)

1501083128

[Figure 250]

It regresses FLAME LBS weight Wi and blendshapes {Ei,Pi} of each point jointly, which is crucial to reenact our avatars into any novel pose and expression. Subsequently, our avatar model defines a mapping of subject-specific canonical points Psc to the FLAME canonical points Pfc for better fidelity following the previous work [5, 79]. The mappings between two points are defined as another point offset Oisc→fc which is also regressed by the deforming MLP jointly. The transformation between each state are summarized as follows:

[Figure 251]

[Figure 252]

[Figure 253]

512

512

32

[Figure 254]

128

[Figure 255]

3

[Figure 256]

(c) Latent Mapping

(d) Deformation

[Figure 257]

[Figure 258]

Figure 11. Network Configuration. We show a detailed structure of the networks of our avatar model: pose-conditioned deformation MLPpose, canonical MLPc, latent mapping MLPz, and deformation MLPd.

xsci = xgci + Oigc→sc, (19) xfci = xsci + Oisc→fc. (20)

Finally, the points in the FLAME-canonical space Pfc are deformed into the final posed space Pd using Linear Blend Skinning (LBS) and FLAME parameters {β,θ,ψ} as follows:

using a coordinated-based MLP as follows:

{osci ,rsci ,ssci ,csci } = MLPc(z,xsci ). (23)

xd− = xfc + BS(β;S) + BP(θ;P) + BE(ψ;E) (21) xd = LBS(xd−,J(ψ),θ,W), (22)

This canonical MLPc is defined against subject-specific canonical points and conditioned by latent code z. We model additional 3D Gaussian change depending on the pose changes following MonoGaussianAvatar [7]. We calculate the deviation of each Gaussian center between before and after LBS deformation of (22) and query the change of each center to an MLP network together with latent z to

where xd− denotes the point after applying the blendshapes and before applying transformation via linear blend skinning.

Similar to PEGASUS [5], we infer the attributes of each Gaussian, oi (opacity), ri (rotation), si (scale), and ci (color)

estimate pose-conditioned deformation:

∆xi = xdi − xfci , (24) {∆ri,∆si,∆oi,∆ci} = MLPpose(∆xi,z). (25)

We change all Gaussian parameters except the center xi. The final deformed Gaussians which are queried in the Gaussian Rasterizer [32] are as follows:

odi = ∆oi + osci , (26) sdi = ∆si + ssci , (27) cdi = ∆ci + csci , (28)

∂xdi ∂xfci

rdi = ∆ri + Rot(rsci ,

), (29)

where Rot(·) denotes multiplying a corresponding rotation

∂xdi

∂xfci on each quaternion rsci occurred during LBS of (22). The overall optimizable parameters of our avatar model are

summarized below:

Θ = {MLPc,MLPd,MLPz,MLPpose,{xgci }i∈{1···N}}.

(30) The detailed network structure is shown in Fig. 11.

##### A.1.2. Training Strategy

We set the first epoch as a warm-up stage for stable optimization. During this stage, the pose-conditioned deformation MLP is disabled, and only the remaining MLPs and points are optimized. It encourages the deformation module of the avatar network to generate valid offsets from the generic canonical space to the final deformed space. We optimize our avatar model for 112 epochs using DDP with 8 A6000 GPUs, which takes around 2 days.

We follow prior work [7, 79] to iteratively densify the Gaussians via upsampling every 5 epochs until the number of points reaches 130,000. Once this target is reached, we reduce the length of the existing Gaussian attributes’ 3D covariance by a factor of 0.75, and prune Gaussian attributes with opacity lower than 0.5 every 5 epochs. To maintain the point count at 130,000, we additionally upsample new Gaussian attributes with a fixed radius of 0.004.

##### A.1.3. Loss Functions

The FLAME loss [7, 79] included in total loss Ltot is regularization enforcing the inferred FLAME blendshapes and LBS weights (Eˆ,Pˆ,Wˆ ) of each Gaussian to be close to the FLAME mesh’s one:

1 N

LFLAME =

N

(λe∥Ei − Eˆi∥2

i=1

+ λp∥Pi − Pˆi∥2 + λw∥Wi − Wˆi∥2),

(31)

where E,P, and W are the pseudo ground truth from the k-nearest neighbor vertices of the FLAME [39]. This regularization is important to obtain better reenactment with unseen pose.

#### A.2. Finetuning for Interpolated Samples

##### A.2.1. Preliminaries: DiffMorpher

By viewing a diffusion sampling process as a solution of ODE, we obtain a deterministic mapping between a latent variable in the Gaussian distribution ξT ∈ N and an image I through DDIM forward and inversion [60]:

ξ = DDIMinv(I;W), I = DDIM(ξ;W),

where W means a pre-trained image diffusion model. By interpolating latents (ξa,ξb) inverted from two images (Ia,Ib), we obtain semantically meaningful smooth interpolation as follows:

ξinterp,α = slerp(ξb,ξa,α), Iinterp,α = DDIM(ξinterp,α;W),

where α is an interpolation weight and slerp(·) is spherical linear interpolation [57].

DiffMorpher [70] uses personalized diffusion models for DDIM sampling and inversion, resulting in smoother and better natural image interpolation. For two images (Ia,Ib), it trains LoRA [26] on UNet (∆Wa,∆Wb) for each image and uses the LoRA-integrated UNet for DDIM inversion:

- ξa = DDIMinv(Ia;W + ∆Wa),
- ξb = DDIMinv(Ib;W + ∆Wb).

For the forward process on interpolated latent ξinterp,α, it uses interpolated LoRA with attention interpolation:

Iinterp,α = DDIM(ξinterp,α;Θinterp,α), (32)

where Winterp,α is an interpolated LoRA derived as Winterp,α = W+∆Wa(1−α)+∆Wbα. For brevity, we denote the overall interpolation process with DiffMorpher from two images (Ia,Ib) and a weight α as follows:

Iα

i

= DiffMorpherα

i

Ia,Ib . (33)

##### A.2.2. DiffMorpher LoRA Optimization

We use DiffMorpher [70] to generate interpolated images, which serve as pseudo ground truth to fine-tune our avatar model. Specifically, we select two subjects from the synthetic dataset and fine-tune the model for interpolated renderings between them. To obtain the corresponding pseudo ground truth images with DiffMorpher, we require a LoRA for each image.

Training a LoRA for each posed image is computationally prohibitive considering the number of images in our synthetic dataset. Therefore, unlike vanilla DiffMorpher [70], which uses a single image, we train LoRA subject-wise using all animated frames in each subject. The LoRA training objective is equal to the standard diffusion training objectives [52] as follows:

L(∆Θ) = Eϵ,τ,i[||ϵ − ϵΘ+∆Θ(ξτi,τ,ci)||2], (34) ξτi = √α¯τξ0i + √1 − α¯τϵ, (35)

where ξ0i = E(Ii) represents the latent encoded by the VAE encoder of diffusion model, Ii is the ith animated image of the subject randomly selected at each iteration, ϵ ∼ N(0,I) is Gaussian noise, and ξτi denotes the perturbed latent at diffusion step τ. To avoid confusion with our model’s latent variable z, we use ξ to refer to the VAE-encoded latents of the diffusion model here. We train the subject-specific LoRA with batch size 8 for 5 epochs per subject.

##### A.2.3. Interpolation Loss Details

To enhance the quality of the interpolated sample and ensure interpolation smoothness, we calculate reconstruction loss on the interpolated samples. In every iteration, we randomly sample two subjects (a,b) from the same category of our synthetic dataset, referred to here as pivots. Then, we generate 5 interpolated samples using linear interpolation as follows:

zα,i = za(1 − αi) + zbαi, (36)

where {αi}[i=1···5] are 5 equally distributed interpolation weights from 1/6 to 5/6. For all 5 interpolated samples, we compare the rendering with DiffMorpher [70] generated images as follows:

ˆIα

= GSR MΘ(zα,i) , (37) Iα

i

= DiffMorpherα

Ia,Ib , (38)

i

i

Linterp =

5

,Mpart ◦ ˆIα

Lpart Mpart ◦ Iα

i

i

i=1

. (39)

As the image Iα generated by DiffMorpher [70] fails to preserve the identity of the remaining regions, we apply the loss only to the subpart region Mpart is that changes during interpolation.

All DiffMorpher inferences and target part segmentations are performed online during optimization, as the number of possible pairs is too large to process in advance. We finetune our avatar model using an interpolation loss applied to 40 arbitrary pairs per subject, resulting in a total of 38,600 pairs. In each iteration, we also apply the total loss Ltot to the pivot subjects (a,b) to preserve their quality.

Category # of attributes w/ portrait-Champ

Hair 395 ✓ Beard 69 ✓ Cloth 57 Earrings 59 ✓ Eyebrows 58 Headphones 59 ✓

Hat 110 ✓

Mouth 75 Nose 75 Total 957 -

Table 4. Number of Attributes in Our Synthetic Dataset. We use portrait-Champ to animate the portrait images when ‘w/ portraitChamp ’ is indicated; otherwise, we use LivePortrait [18].

#### A.3. Synthetic Dataset

##### A.3.1. Attribute-Edited Portrait Image Generation

The number of attributes in each category is shown in Tab. 4. While we generate approximately 1k samples to demonstrate the effectiveness of PERSE, the pipeline can be extended to produce any desired amount, as our synthetic dataset generation process is fully automated. We use FLUX with inpainting controlnet [9] for Image-to-Image (I2I) inpainting and SDXL with pose controlnet [49, 71] for attribute mask generation.

###### A.3.2. Training portrait-Champ

Our portrait-Champ builds upon the architecture introduced in Champ [80], incorporating modifications to enhance 3Dawareness and improve reenactment performance. Specifically, we integrate an additional Variational Autoencoder (VAE) encoder-decoder pair dedicated to normal maps, drawing inspiration from MagicMan [20]. Adopting the dualbranch strategy proposed in MagicMan [20], we introduce an additional U-Net for the normal maps. This U-Net shares all weights with the original RGB U-Net except for the first layer. The shared layers between the two U-Nets enable cross-domain feature integration, allowing the model to fuse features from both normal map and RGB image. By combining geometric and visual information, our approach enhances the geometric awareness of model, resulting in improved structural coherence.

We replace the original SMPL [40] rendered motion guidance in vanilla Champ [80] with FLAME rendering. Specifically, we employ a monocular face capture method [10] to extract FLAME parameters [39]. Using these parameters, we render the FLAME depth map and FLAME normal map. To provide motion guidance for the body, including shoulders, which are not covered by FLAME rendering, we supplement the guidance with full-body keypoints and facial landmarks inferred from RGB videos using DWPose [68].

We use 5,196 videos from CelebV-Text [69] datasets to train our portrait-Champ. Following previous work [80], we train portrait-Champ using 8 A6000 GPUs in two stages:

58,732 iterations with a batch size of 32 in stage 1, and 26,450 iterations with a batch size of 8 in stage 2. In stage 1, we optimize the model using randomly sampled frames from videos as an image diffusion model. In stage 2, we train only the temporal motion module with videos while freezing other modules.

##### A.3.3. Animating Portrait Images

Enhancing the reenactment capability of our avatar model requires training videos that cover a wide range of facial expressions and head poses. We achieve this by animating portrait images with a motion sequence containing diverse expressions and poses. To obtain a motion sequence that satisfies both continuity and the minimal number of frames required by portrait-Champ, we record a video for this motion sequence ourselves.

Using a reference portrait image and a predefined motion sequence in an RGB video, we first generate an animated portrait video centered on the reference image using LivePortrait [18]. From this video, we extract normal maps, depth maps, and facial keypoint motion guidance using EMOCA [10] and DWPose [68]. With this guidance, we animate images edited in the hair, hat, and beard attributes using portrait-Champ. For other facial attributes, we directly generate RGB videos using LivePortrait [18].

#### A.4. Attribute Transfer

To transfer facial attributes from in-the-wild images, we incorporate LoRA layers [26] into the MLP network of the avatar model and optimize these layers. The LoRA layers are trained using animated videos generated from input inthe-wild images. We generate the animated videos following the procedure outlined in Sec. 4.2 of the main paper. To ensure only the desired attribute is transferred, we segment the relevant sub-part using an off-the-shelf segmentation network [33] and apply a part-wise loss as described in Eq. (15) of the main paper:

Lpartwise, lora = Lrecon Mpart ◦ Iitw,Mpart ◦ ˆIattr , (40)

where Iitw represents the image from video animated in-thewild portrait image, Iˆattr denotes the rendered image with latent zitw regressed by latent mapping MLPz from CLIP features of input in-the-wild image.

We observe that using only the partwise loss fails to preserve reference identity of our avatar model and collapse the pretrained latent space. To address this, we introduce a 3D loss. The 3D loss encourages the LoRA layers in the avatar model to produce the same output as when the LoRA layers are absent. Specifically, Gaussian random latent codes zrandom from the pretrained latent space are sampled and used as inputs along with the FLAME parameters of an animatable portrait video. The model is trained to minimize the difference between the outputs of the avatar model with

and without the LoRA layers, ensuring consistency in 3D Gaussian parameters and 3D positions. Specifically, for the Gaussian attributes inferred with and without LoRA layers:

{xdi ,rid,sdi ,odi ,cdi } = MΘ(xgci ,zrandom,β,θ,ψ), (41) {xdi,lora,ri,dlora,odi,lora,sdi,lora,cdi,lora}

= MΘ+∆Θ(xgci ,zrandom,β,θ,ψ), (42) we calculate the distance between them as follows:

L3d = ∥xdi,lora − xdi ∥1 + ∥ri,dlora − rid∥1

+∥odi,lora − odi ∥1 + ∥sdi,lora − sdi ∥1 + ∥cdi,lora − cdi ∥1. (43)

The total loss for LoRA layer optimization is defined as follows:

Ltotal, lora = L3d + Lpartwise, lora (44)

We perform LoRA layer optimization with a learning rate of 1e−4 for 5 epochs.

### B. Evaluation Details

#### B.1. Baseline Implementation Details

To demonstrate our pipeline’s effectiveness, we evaluated our methods compared to three different methods.

##### B.1.1. PEGASUS

We train PEGASUS [5] with our synthetic dataset using publicly available code, strictly following the settings described in the paper, including the latent space configuration and network configurations. The model is trained using DDP across 8 RTX 6000 GPUs until convergence. After point rendering with PyTorch3D [51], no additional denoising steps are applied.

##### B.1.2. Conditional INSTA (Cond.TA)

To train INSTA with multiple subjects, we introduce a latent condition to the density MLP network, referred to as Conditional INSTA (Cond.TA). We adopt the PEGASUS [5] latent configuration to achieve similar sub-part disentangled control. Since the original density MLP network of INSTA is too small to encode a thousand of attributes, we increase the MLP width from 64 to 512 and the depth from 2 to 4. As this adjustment sacrifices rendering speed and increases training time, we focus our comparisons solely on quality, excluding rendering speed. The final Conditional INSTA model is trained using DDP with 8 RTX 4090 GPUs until convergence.

##### B.1.3. Conditional SplattingAvatar (Cond.SA)

Since SplattingAvatar [55] does not include any network for receiving conditioning, we incorporate an MLP to deform a single set of shared canonical 3D Gaussians into subjectspecific canonical 3D Gaussians, similar to the approach in

PEGASUS [5]. To ensure a fair comparison, we configure the MLP with the same size as PEGASUS’s canonical MLP, providing sufficient capacity to represent all subjects in the synthetic dataset. The densification interval is increased from vanilla SplattingAvatar [55] to address the low stability of optimization in early stages. Densification is halted after

- 5 epochs, as the gathered gradients do not converge, possibly due to exposure to different subjects in each iteration. We adopt the same latent configuration as the PEGASUS model, and the final Conditional SplattingAvatar model is trained using DDP on 8 RTX 4090 GPUs until convergence.

#### B.2. Interpolation Evaluation Details

To evaluate the rendering quality of avatars with unseen attributes and interpolation smoothness, we sample avatars from our model using interpolated latent codes. For each of the 9 categories in our synthetic dataset, we randomly select 200 subject pairs and generate 9 interpolated latent codes per pair, following (36). The intervals between the sampled latent codes are evenly spaced. Each interpolated latent code is used to render the corresponding avatar in 5 different poses. This process produces 9,000 images per category and a total of 81,000 images across all categories for evaluation.

Metrics. We compute FID and KID scores by comparing our renderings with two different image sets: FFHQ [29] and our synthetic evaluation dataset, which is built with the same input reference individual. Specifically, we use (FIDFFHQ,KIDFFHQ) to asses the realism and quality of the renderings by comparing with real face images, and (FIDSYN,KIDSYN) to evaluate identity preservation by comparison with the synthetic evaluation dataset.

Since the rendered outputs do not include backgrounds, we remove the backgrounds of all portrait images in FFHQ using MODNet [31] before calculating metrics. The synthetic evaluation image sets are constructed with the same reference image, following our edited portraits generation pipeline. To prevent potential information leaks, we synthesize 2k novel images using text prompts not included in the training dataset. This approach provides a more reliable measurement of identity preservation during attribute editing, particularly for changes that partially alter identity features, such as the eyes, eyebrows, and nose, which are challenging to evaluate with existing identity metrics like ArcFace [11].

#### B.3. User Studies

We also conduct a user study to evaluate the rendering quality of interpolated samples, as shown in Fig. 21. Since only PEGASUS [5] and our method receive votes among the four methods in preliminary study, we exclude Cond.TA and Cond.SA from the options. Participants are asked to choose the better images based on interpolation smoothness

Accuracy Naturalness PSNR↑ SSIM↑ LPIPS↓ FID↓ KID↓

Method

Moore-AnimateAnyone [27, 44] 17.77 0.6841 0.2536 146.59 0.0530 MimicMotion [74] 17.27 0.6641 0.3012 178.87 0.0980 MegActor-Σ [66, 67] 17.89 0.6986 0.2599 155.04 0.0572

Ours (portrait-Champ) 20.58 0.7417 0.1878 150.59 0.0555

- Table 5. Quantitative Comparisons for Image-to-Video Models. We evaluate our portrait-Champ with recent diffusion based baselines in face reenactment scenarios. Ours portrait-Champ obtain the best scores in accuracy and comparable FID and KID.

Input Type

2D Video Rendering Quality

Subject Consistency ↑ PSNR↑ SSIM↑ LPIPS↓ Imaging Quality↑

Real Video 0.9761 22.26 0.9045 0.1352 0.5366 Synthetic Video self-driving 0.9719 21.23 0.9241 0.1582 0.5896

- Table 6. Quantitative Comparison of Impact of Inconsistency. Quantitative comparison of PERSE avatar models trained on real and synthetic videos. Note that 2D video evaluated for subject consistency is used for training, and rendering quality is evaluated on unseen head poses and facial expressions using a test sequence.

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

(a) GT (b) Real Video (c) Synthetic Video

Figure 12. Qualitative Comparison of Impact of Inconsistency. We show qualitative comparision of impact of inconsistency between real and 2D generated video.

and image quality for 20 pairs of interpolations. The pairs are randomly selected from the hair category. We collect responses from 229 participants via CloudResearch [8].

### C. More Experiments

#### C.1. Additional Results

We present additional sample results of attribute-edited portrait image generation, providing seven results for each attribute in Fig. 13 and Fig. 14. Furthermore, we demonstrate the rendering results of our personalized 3D generative avatar on unseen poses, trained with synthetic datasets created using additional portrait images in Fig. 15, Fig. 16, Fig. 17, Fig. 18, and Fig. 19. Finally, we provide the interpolation results between two latent codes for each attribute in Fig. 20.

#### C.2. Impact of Video Inconsistency

The different between real and generated 2D video is negligible, as monocular avatar-building pipeline handles temporal deformations and inconsistencies. To assess this, we present evaluations by building a 3D avatar from each single video, as demonstrated in Tab. 6 and Fig. 12. We measure subject consistency and imaging quality following VBench [28], comparing real video and generated video from portrait-

Champ by animating the first frame in a self-driven manner, where they show minor differences. After building 3D avatars from each 2D real and generated video separately, we also compare the rendering quality under novel head poses and facial expressions. As shown in Tab. 6 and Fig. 12, the avatar renderings also show negligible differences in quality, with comparable PSNR, LPIPS, and SSIM scores.

#### C.3. Synthetic Monocular Dataset Generation from Single Image

To demonstrate the effectivness of our portrait-Champ, we evaluate the reconstruction quality and rendering realism compared to diffusion based baselines. Moore AnimateAnyone [44] is open-source repository fine-tuned AnymateAnyone [27] to be specialized on facial reenactment. MimicMotion [74] is a full body animating model based on Stable Video Diffusion [2] also capable of reenactment using facial landmarks in DWPose. MegActor-Σ is Diffusion Transformer [48] based approach to solve reenactement problem. We disable the additional audio input option of MegActor-Σ during test here.

We test the methods using 20 sequence randomly selected from CelebV-Text dataset [69] not seen during the trainining. We animate the first frame to make other frames and compare with ground truth frames in the video to compute accuracy. We additionally calculate FID and KID against FFHQ dataset [29] to evaluate the naturaless of the animated images. As shown in Tab. 5, our approach achieves the highest reconstruction score across all metrics compared to previous SOTA methods.

### D. Rights

All portrait reference images used in this work are sourced from the FreePik [15] website under a free license. Note that all of our portraits to show our results are not AI-generated images. Our code and samples of synthetic datasets are publicly released for research purposes only. For more details, refer to https://github.com/snuvclab/perse about our implementations.

### E. Notations

Refer to Tab. 7 for an overview of the notations used in this paper.

[Figure 265]

EyebrowsEarringsClothBeardHatHairNoseMouthHeadphones

Figure 13. Example of Attribute-Edited Portrait Image Generation (1). We present samples of attribute-edited portrait image generation. For each attribute, we display results obtained through random sampling.

[Figure 266]

EyebrowsEarringsClothBeardHatHairNoseMouthHeadphones

###### Figure 14. Example of Attribute-Edited Portrait Image Generation (2). Our method can be applied to various portrait images

[Figure 267]

[Figure 268]

EyebrowsClothHatHairMouthHeadphonesEarringsBeard

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Figure 15. Unseen Pose Rendering Results (1). We present the rendering results using latent codes for novel head poses and facial expressions not included in the training dataset, categorized by each attribute.

[Figure 273]

EyebrowsClothHatHairNoseMouthHeadphonesEarringsBeard

###### Figure 16. Unseen Pose Rendering Results (2).

[Figure 274]

[Figure 275]

EyebrowsClothHatHairNoseMouth

###### Figure 17. Unseen Pose Rendering Results (3).

[Figure 276]

- 23

- Figure 18. Unseen Pose Rendering Results (4). We show hair-only rendering results for unseen poses.

[Figure 277]

- 24

- Figure 19. Unseen Pose Rendering Results (5). We show hair-only rendering results for unseen poses.

[Figure 278]

HairHatHairHatHatBeardBeard

Figure 20. Interpolation Between Two Latent Codes. We present the rendering results obtained by interpolating between two latent codes.

[Figure 279]

###### Figure 21. User Study. We26show user study screenshot.

Table 7. Table of notations.

Symbol Description

##### Index

- i Gaussian index i ∈ {1,...,N} in 3D Gaussian attributes
- j Category index j ∈ {1,...,Nc} of edited attributes in synthetic dataset. Learnable Parameters and Networks

MLPc Canonical MLP estimating attributes of 3D Gaussians MLPd Deformation MLP estimating deformation attributes MLPpose Pose-conditioned deformation MLP estimating change of Gaussian attributes MLPz Latent mapping MLP from CLIP feature fI,fT to subject-specific latent z Pgc = {xgci }i={1···N} Learnable positions of 3D Gaussians

Spaces of our Avatar Model Pgc Generic canonical space, single space shared on all subject Psc Subject-specific canonical space, conditioned by subject latent z Pfc FLAME-canonical space, deformed from subject-specific canonical space with blendshape Pd Deformed space, deforming Pfc with FLAME pose parameters

Diffusion Related T Text-prompt queried into the diffusion model C(·) 2D key points and face landmarks estimator and renderer (OpenPose) τ Diffusion denoising time-step ξ0 Encoded latent of the queried RGB images of diffusion model ξτ Perturbed latent with noise time-step τ ∈ [0,1] ϵ Noise added to the latent

Attributes of 3D Gaussians xi ∈ R3 Center of i-th Gaussian, or point position in PEGASUS qi ∈ R4 Covariance Matrix’s Quaternion of i-th Gaussian si ∈ R3 Covariance Matrix’s Scale Component of i-th Gaussian ci ∈ R3 Color of i-th Gaussian oi ∈ R Opacity of i-th Gaussian

Off-the-Shelf Network I2Iinpaint Text-conditioned Image-to-Image inpainting pipeline, based on image diffusion T2I Text-to-Image diffusion model I2V Portrait animating Image-to-Video model, portrait-Champ or LivePortrait [18].

FLAME Parameters of Avatar Deformation θ ∈ R15 FLAME pose parameter β ∈ R100 FLAME shape parameters ψ ∈ R50 FLAME expression parameters

E ∈ R50×5023 FLAME expression blendshape parameters, estimated by MLPd for each Gaussian P ∈ R100×5023 FLAME shape blendshape parameters, estimated by MLPd for each Gaussian W ∈ R15×5023 FLAME Linear Blend Skinning (LBS) weight, estimated by MLPd for each Gaussian

Rendered and Observed Images ˆI/I Rendered / Ground Truth Image M Mask of subpart region

