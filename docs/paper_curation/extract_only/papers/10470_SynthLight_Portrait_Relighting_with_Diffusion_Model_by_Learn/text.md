# arXiv:2501.09756v1[cs.CV]16Jan2025

## SynthLight: Portrait Relighting with Diffusion Model by Learning to Re-render Synthetic Faces

Sumit Chaturvedi1∗ Mengwei Ren2 Yannick Hold-Geoffroy2 Jingyuan Liu2 Julie Dorsey1 Zhixin Shu2†

1Yale University 2Adobe Research

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

Figure 1. SynthLight performs relighting on portraits using an environment map lighting. By learning to re-render synthetic human faces, our diffusion model produces realistic illumination effects on real portrait photographs, including distinct cast shadows on the neck and natural specular highlights on the skin. Despite being trained exclusively on synthetic headshot images for relighting, the model demonstrates remarkable generalization to diverse scenarios, successfully handling half-body portraits and even full-body figurines.

#### Abstract

We introduce SynthLight, a diffusion model for portrait relighting. Our approach frames image relighting as a re-rendering problem, where pixels are transformed in response to changes in environmental lighting conditions. Using a physically-based rendering engine, we synthesize a dataset to simulate this lighting-conditioned transformation with 3D head assets under varying lighting. We propose two training and inference strategies to bridge the gap between the synthetic and real image domains: (1) multi-task training that takes advantage of real human portraits without lighting labels; (2) an inference time diffusion sampling procedure based on classifier-free guidance that leverages the input portrait to better preserve details. Our method generalizes to diverse real photographs and produces real-

*Work done as an intern at Adobe Research. †Corresponding author.

istic illumination effects, including specular highlights and cast shadows, while preserving the subject’s identity. Our quantitative experiments on Light Stage data demonstrate results comparable to state-of-the-art relighting methods. Our qualitative results on in-the-wild images showcase rich and unprecedented illumination effects. Project Page: https://vrroom.github.io/synthlight/

#### 1. Introduction

Lighting is fundamental to portrait photography, yet manipulating it after capture remains challenging. Recent advances in generative imaging models have demonstrated promising capabilities for controlling lighting in existing images [15, 19, 33, 57, 59]. However, these approaches typically require labeled training data. For portrait relighting specifically, the most effective results have come from training on Light Stage data—portraits rendered with linear combinations of one-light-at-a-time (OLAT) captures.

While powerful, Light Stage setups are constrained by physical limitations in light source density and require specialized artificial lighting equipment. In contrast, 3D workflows in VFX and gaming have long treated lighting as a relatively straightforward endeavor through modern physically based rendering engines, where light source control is nearly arbitrary. To relight a rendering, artists simply adjust the lighting configurations and re-render the scene.

Given a scene S and lighting L1, we denote the rendering

- as I1 = fr(S,L1). The inverse graphics problem aims to find S from I1: S = finv(I1,L1) with known or unknown lighting. To relight rendering I1 to I2 under lighting L2, one aims to compute: I2 = fr(S,L2). Given only I1, a relighting procedure seeks: I2 = fr(finv(I1),L2) = fre(I1,L2), where fre is the relighting/re-rendering function. Previous approaches [22, 28, 56] have tackled this problem through inverse graphics, either explicitly or implicitly, by estimating lighting-invariant intrinsic scene representations such as depth, surface normals, and albedo. This imposes limitations on subsequent rendering functions and often fails to capture complex illumination effects like inter-reflections, occlusion shadows, and subsurface scattering. In this paper, we propose bypassing inverse rendering entirely by learning the relighting function using physically based 3D renderings of human heads. Specifically, we render pairs of por-

trait images using Blender (Cycles) (I1,L1) and (I2,L2) and train a diffusion model to directly learn to “re-render” I2 from I1 and L2.

However, this approach introduces an inevitable domain gap between simulated 3D renders and real photographs. To address this challenge, we leverage a latent diffusion model pretrained on vast internet images for text-to-image generation. We propose to finetune the network with our face renderings and introduce simple yet effective training and testing schemes to narrow the gap between training data and in-the-wild images. During training, we propose multitask training that incorporates in-the-wild images without ground truth relighting information. This allows the model to learn relighting from our synthetic dataset while maintaining knowledge of the real image domain, preventing distributional drift. We further observe that input portraits contain rich textural information. Leveraging the flexibility of diffusion model inference, we design an inference time adaptation scheme that effectively preserves input portrait details in the relit result.

We evaluate our methods on in-the-wild portrait images, demonstrating highly detailed illumination effects that accurately capture interactions between the portrait scene and lighting. Our results produce realistic cast shadows and specular highlights on the skin. For the first time, we demonstrate an end-to-end system capable of non-trivial lighting effects including catch lights in eyes, subsurface scattering in ears, and inter-reflections with clothing. No-

tably, despite training only on simple headshot renderings of 3D faces without accessories, facial hair, or hats, our network generalizes effectively to complex portrait images, including half-body shots and multi-person photographs.

We quantitatively evaluate our method on a test set of our synthetic faces dataset as well as on a Light Stage OLAT dataset. Despite using no Light Stage data for training, our method achieves comparable or superior results to state-ofthe-art portrait relighting methods trained on OLAT data. User studies show that our results are preferred across all evaluated aspects, including perceptual lighting accuracy, identity preservation, and overall image quality.

We summarize our contributions as follows:

- 1. We propose modeling portrait relighting as a task of learning to re-render a portrait scene in 3D. Using physically based renderings of human heads under varying lighting conditions, we train a diffusion model to learn pixel transformations conditioned on lighting.
- 2. We introduce two techniques enabling synthetic data learning while minimizing domain gap with real images, through the use of a training-time multi-task strategy that incorporates real images through a text-to-image task, and an inference-time approach based on classifier-free guidance that preserves portrait details in the relit result.
- 3. Through extensive qualitative and quantitative evaluations, we demonstrate state-of-the-art portrait relighting results, achieving high-quality lighting effects previously unattainable by existing methods.

#### 2. Related Work

##### 2.1. Portrait Relighting

Portrait relighting has been explored in both 2D [19, 22, 28, 29, 33, 43, 46, 52, 57, 59] and 3D domains [3, 6, 32, 48, 49, 51, 61], with 2D image-based approaches being more relevant to our work. Since 2D portrait relighting is under-constrained, various priors have been proposed, such as morphable models [4] as 3D face priors in [42], explicit inverse rendering in [2, 40], and a style transfer approach for relighting in [41].

Recently, deep learning methods [27, 46] trained on light stage data [9] have driven the state-of-the-art for relighting, with [22, 28] demonstrating a widely adopted physicsguided architecture for relighting based on image decomposition into intrinsics such as albedo, normals, diffuse, and specular reflectance maps, conditioned on an HDR environment map lighting representation [8]. However, this formulation presents two main shortcomings. First, the rendering model assumes a BRDF-based reflectance model [7, 31], where light is reflected directly from the surface point of incidence, thus neglecting other modes of light transport such as subsurface scattering, which are significant in certain types of human skin (e.g., fair skin) [12, 23, 26]. Addition-

ally, albedo estimation becomes challenging in the presence of face accessories, inter-reflections and face paint [22, 56]. Second, light stage setups inherently limit the types of lighting that can be captured due to restricted light intensity [46] and lighting resolution [47], hindering the ability to learn complex lighting effects such as specular reflections and subsurface scattering. Motivated by these constraints, we employ diffusion models to learn face relighting, without assuming any appearance model, from a synthetic dataset rendered with a physically based renderer that provides input and relit training pairs for supervision. This enables our method to synthesize interesting illumination effects for human portraits such as hard cast shadows, subsurface scattering and inter-reflections.

##### 2.2. Diffusion Models for Relighting

Diffusion models [1, 10, 20, 21, 34, 36, 37, 44, 45] have become the standard framework for tasks ranging from textto-image generation to image-to-image translation and appearance editing. Their ability to scale well with large datasets, coupled with pretrained weights [34] that can be readily adapted to new domains [18, 58], makes them especially suited for these applications. They also offer flexible inference mechanisms, where improved sampling procedures can significantly boost image quality [16, 24].

Several recent works employ diffusion models specifically for relighting. DiLightNet [57] demonstrates finegrained control of object lighting by incorporating radiance hints. However, their multi-step pipeline, depends on scene reconstruction [50], which is error-prone. Similarly, Neural Gaffer [19] focuses on object relighting, leveraging HDR environment maps. For human portrait relighting, Relightful Harmonization [33] and IC-Light [59] train on highquality datasets (including light stage captures, synthetic Objaverse renders, and composited shadow materials) to synthesize background-harmonized portraits. Both methods rely on the background as lighting condition. In contrast, our approach directly tackles portrait-based relighting, using a diffusion model that learns to re-render synthetic faces. By starting from a pretrained model, and through our multi-task training strategy, we retain rich facial priors, while classifier-free guidance [16] on the input portrait further improves the preservation of texture and detail in the final relit output.

##### 2.3. Domain Adaptation

Naively training on synthetic data often creates a domain gap for in-the-wild portraits, causing poor identity preservation and reduced photo-realism. Prior diffusion-based domain adaptation approaches [13, 18, 35, 55, 58] mainly target style transfer or focused editing, not relighting.

[15] propose training a personalized diffusion model per subject, preserving identity but require light-stage cap-

ture and dedicated training for each subject. Other methods leverage real data to mitigate the synthetic-to-real gap: SwitchLight [22] pre-trains with a masked-autoencoder [14] on real images before training on light-stage data, learning visual features (e.g. structure, color, texture) that are essential for relighting; Relightful Harmonization [33] bootstraps a relighting model learned from light-stage data to pseudo-label in-the-wild images, subsequently finetuning on these pseudo-labels for improved photorealism; ICLight [59] uses large-scale data augmentation; and Lumos [56] finetunes its albedo-prediction branch on real images, though its decomposition approach can fail with face paint, accessories, or strong shadows.

We propose a multi-task training scheme that unifies text-to-image and relighting tasks, enabling the training of our diffusion model with real images along with our synthetic dataset. In addition, our inference scheme based on classifier-free guidance helps preserve fine details from the input portrait. Our user study shows that the resulting relit portraits exhibit superior visual quality, identity, and lighting compared to existing methods.

#### 3. Method

Given a portrait image I captured under unknown illumination conditions, our goal is to synthesize a relit version IR under a target lighting environment specified by a panoramic environment map E. The relit portrait IR should simultaneously: (1) preserve the subject’s facial identity and characteristics from the original image I; (2) accurately reflect the illumination effects defined by the target environment map E and (3) maintain photorealism in the final rendering. We first simulate this re-rendering to build a synthetic dataset for human portraits using Blender.

##### 3.1. Synthetic Data for Relighting

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Figure 2. Synthetic Faces: Subjects are rendered under various lighting conditions (details in Sec. 3.1). We show two examples, where each pair consists of a subject rendered using two different environment maps. The network is trained to re-render synthetic faces by transforming a subject rendered with one environment map into its counterpart rendered with the other environment map.

We build a 3D human portrait generation pipeline similar to [53]. Our system begins with a collection of high-quality, artist-created 3D head meshes, which we enhance by incorporating detailed facial components, including eyes, teeth, gums, and hair. We then augment these base

models through rigging for pose variation and blendshape deformation for diverse facial expressions. To render realistic appearances, we incorporated a set of high quality PBR texture maps, including albedo, normal, roughness, specular, and subsurface scattering maps. We combine the head with random clothing meshes to build a portrait scene. The system is built with Blender and the images are rendered with the Cycles renderer.

the-wild images, resulting in degraded output quality. For instance, when applied to real-world images, the model fails to reproduce critical details, such as textures in clothing, jewelry, and accessories, which are absent in the synthetic data distribution (e.g., as seen in the ’Base’ result in Fig. 9). To address this, we propose a multitask training strategy to mitigate potential model distribution drifting to synthetic renderings. Similar techniques have been applied in the context of inpainting [54] to combat the lack of diversity in training data.

To train our networks, we render images (samples shown in Fig. 2) at 512×512 resolution from 350 subjects, each with roughly 10 varied appearance samples, including different hairstyles, skin tones, expressions, clothes, poses, etc. We render each sample with 10 random HDR environment maps, each rotated 36 times evenly with a random initial rotation. In total, the dataset contains roughly 1.26 million images. See Fig. 24 in the supplemental material for more examples from the dataset.

Specifically, we incorporate a text-to-portrait generation task, which constraints the diffusion model to produce a realistic portrait image given an input prompt. This task is trained alongside the original relighting task, and this helps to improve the photorealism and generalization of the trained model. Since both tasks share the same network architecture, we simply replace the image and LDR inputs with two black images, as illustrated in Fig. 3.

##### 3.2. Modeling Relighting with Diffusion Model

To obtain training samples for the text-to-portrait, we curate a subset of human portrait images from the LAION [39] dataset by sampling the images filtered by a face detector. Details on detection and filtering are provided in the supplementary material (see Appendix B). During training, we empirically set the sampling ratios of the synthetic dataset versus the real dataset as 0.7 and 0.3, respectively. We observe significant benefits from incorporating the real images during training in improving identity preservation and photorealism. This echoes the findings in [33], where a bootstrapped dataset helps generalize of image harmonization, emphasizing the benefits of data diversity.

We build on top of Stable Diffusion [34], a text-to-image foundation model pretrained with vast internet data. As shown in Fig. 3, we incorporate the input portrait I, along with the target environment map E to the input of the network backbone, by expanding the number of channels in the first convolutional layer of the Unet as per [36].

To generate training samples (I,E,T,IR), where T is a text prompt, we render portrait images from a subject S with n different HDR maps E1HDR ···EnHDR to obtain portraits I1S ···InS. We use an off-the-shelf image captioning model [25] to caption these images. Training samples are constructed by sampling two indices i,j ∈ {1···n} and then using them to select input portrait, environment map, text prompt and target portrait as (IiS,EjHDR,TjS,IjS). In the following, we drop the superscript S for the subject to simplify notation. We use the sample to supervise our diffusion model in the following manner. First, we convert the HDR environment map EjHDR into LDR EjLDR by tone-mapping similar to [19]. The LDR environment map along with the input and target portraits are encoded using the encoder Enc of Stable Diffusion’s VAE, i.e., Iˆi = Enc(Ii),EjLDRˆ = Enc(EjLDR),Iˆj = Enc(Ij).

###### Task 1: Relighting Input Envmap Relit

Prompt: "a woman with brown hair looking at the camera"

[Figure 21]

[Figure 22]

[Figure 23]

|N|e|
|---|---|

U t

###### Task 2: Text-to-Portrait Prompt: "A bride and groom hugging

each other ona beach" Generated

[Figure 24]

|Drop envmap|
|---|

|Drop image input|
|---|

|N|e|
|---|---|

Following the DDPM formulation [17], we randomly sample Gaussian noise ϵ and a diffusion timestep t to add noise to the relit image latent Iˆj to obtain the noised latent Iˆjt. We concatenate Iˆi,EjLDRˆ ,Iˆjt along the channel axis and feed it to the Unet, following [19]. The Unet ϵθ is trained with the DDPM objective:

U t

Figure 3. Training pipeline of SynthLight. We first enable the relighting modeling by training the diffusion backbone with synthetic relighting tuples (Task 1, top row), detailed in Sec. 3.2. To further alleviate the domain gap between synthetic and real image domain, we include a joint training of the text-to-image task (Task 2, bottom row), detailed in Sec. 3.3. Our model is based on LDM [34] and is composed of a VAE and a UNet. For simplicity, VAE is omitted in the diagram.

R),t,ϵ∈N(0,I)∥ϵθ(xt,I,E,T) − ϵ∥ (1)

Ex∈Enc(I

min

θ

##### 3.3. Multitask Training

Training or fine-tuning a diffusion model on a synthetic dataset creates a substantial domain gap when applied to in-

Envmap (E) Noise

Input (I)

- (a) Input Portrait
- (b) Environment map and Reference

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(c) Varying Guidance Scale

𝜆 = 1 𝜆 = 2

𝜆 = 4 𝜆 = 8

(d) Effect on detail preservation

[Figure 29]

[Figure 30]

[Figure 31]

Input Portrait 𝜆 = 2 𝜆 = 1

(e) Effect on shadow artifacts from input portrait

[Figure 32]

[Figure 33]

[Figure 34]

𝜆 = 2 𝜆 = 8

Input Portrait

Figure 5. Effect of input portrait guidance parameter λI: We show (a) the input portrait, (b) the lighting condition and a reference image rendered in Blender with the same lighting, and (c) outputs with varying λI. (d) highlights that λI = 1, equivalent to removing inference-time adaptation, alters the eye shape (in red rectangle). (e) shows that higher λI introduces undesired lighting artifacts, such as shadow artifacts from the input portrait (in yellow rectangle).

4. Experiments

- 4.1. Setup and Metrics

We create three test sets for evaluating our method: (a) 300 Light Stage rendered relighting pairs, (b) a held out subset of our synthetic faces dataset consisting of 500 images, (c) in-the-wild portraits for qualitative evaluation of visual quality. For test sets (a) and (b), we use standard quantitative metrics such as SSIM, PSNR, LPIPS [60] to evaluate image fidelity and face embedding distance such as FaceNet [38] for evaluating identity preservation. We train on the entire synthetic dataset but withhold 20% of the environment maps to create the Light Stage test set. We also hold out 10% of the subject identities and 10% of the environment maps for the synthetic test set, ensuring they remain unseen during training.

- 4.2. Implementation details

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

|N|et|
|---|---|

U

[Figure 39]

[Figure 40]

"A beautiful woman"

[Figure 41]

|Drop image input|
|---|

[Figure 42]

[Figure 43]

|N|et|
|---|---|

U

[Figure 44]

Figure 4. We employ the image-conditioning classifier-free guidance during inference to proportionally balance between identity preservation, and relighting effects. The final score estimate is computed as per Eq. (2).

##### 3.4. Inference Time Adaptation

We further employ a simple yet effective inference time adaptation scheme that proportionally balances between the identity preservation of the input portrait and the relighting strength. Inspired by the dual-conditioning classifier-free guidance [16] proposed in InstructPix2Pix [5], we define an analogous concept in our inference. As illustrated in Fig. 4,

- at each step of the diffusion inference, the diffusion score is a composition of scores from both image-conditional and unconditional output. Specifically, for unconditional inference, we drop the input image while keeping the LDR and text-prompt conditioning identical. Formally, we apply the following score estimate at a particular timestep t:

ϵt = ϵθ(xt+1,ϕ,E,ϕ)

+ λT(ϵθ(xt+1,I,E,T) − ϵθ(xt+1,I,E,ϕ))

+ λI(ϵθ(xt+1,I,E,ϕ) − ϵθ(xt+1,ϕ,E,ϕ)) . (2)

Here, λT and λI are the guidance parameters, where λT is inherited from the original definition of CFG, which specifies the how much the model respects to the text prompts, while λI specifies the strength of the input portrait guidance. With this score estimate, we use DDIM [45] to obtain the latent at current timestep xt = DDIM(xt+1,ϵt). We empirically find that using a guidance value of λI ∈ [2,3] for the input portrait helps achieve a balance between the details and identity preservation while performing reasonable relighting.

We implement our model in PyTorch [30] using 32 × 40GB A100 GPUs. We use a batch size of 192, a learning rate of 10−5, and the Adam [11] optimizer. We train our model (and ablations) for 40K steps, which takes around 1 day. We initialized from the IC-Light [59] checkpoint for background conditioned image relighting, which is fine-tuned based on Stable Diffusion 1.5 [34]. We chose this particular checkpoint because we found it to be beneficial for learning our environment map based relighting model compare to a text-to-image checkpoint. We show more analysis and comparisons of this choice in supplemental material (see Fig. 17 and Tab. 4).

In Fig. 5, we illustrate the effects of varying λI. Smaller values provide the strongest relighting effect while sacrificing some visual quality and losing the facial details of the input. Large values provide much better identity preservation but weaken the relighting effects where lighting information, such as shadows, leaks from the input into the output.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- (a) Our method demonstrates the ability to relight subjects effectively in both outdoor (left) and indoor (right) settings. In outdoor scenarios, strong cast shadows are produced due to self-occlusion from facial features and glasses (see inset). For indoor scenes, our method handles complex lighting conditions, such as casting neon lights on the input portrait.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

- (b) Our method captures interesting lighting effects for portraits, synthesizing fine details like catch light in the eye for realistic relighting (left, see inset) and subsurface scattering in the ear under strong backlight conditions, such as sunlight (right, see inset).

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

- (c) Our method enables studio-style lighting for portraits, creating dramatic effects in studio-like environments (left). Using hand-designed environment maps, we relight with two presets (right): Backlight, which uses a light behind the subject to define edges and produce a distinctive rim effect (see inset); and Rembrandt, where light comes from an angle, illuminating one portion of the face while casting the other in shadow to create depth and contrast. The Rembrandt image also highlights inter-reflections from clothing (rightmost, see inset).

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

- (d) While trained only on a synthetic dataset, our method generalizes to unseen image categories such as a clown (left), a photograph of two people (middle), and a teddy-bear (right).

- Figure 6. Real-world results showcasing our method’s ability to handle diverse lighting scenarios. Each example includes the input portrait (left), the environment map used for relighting (top right), and the relit output (bottom right). The subfigures highlight: (a) relighting under indoor and outdoor environments, (b) capturing interesting lighting effects such as catch lights in eyes and sub-surface scattering on ears, (c) studio-style lighting setups, and (d) generalization across various challenging scenarios.

##### 4.3. Evaluation Results

We compare our method against state-of-the-art methods for portrait harmonization [59], portrait relighting [22] and object relighting [19, 57] on both the synthetic and the light stage test set quantitatively (see Tab. 1) and qualitatively (see Fig. 8 and Fig. 7). Quantitative evaluation shows that

our method outperforms baselines on the synthetic test set and performs comparably to state-of-the-art portrait relighting methods such as SwitchLight, on the Test Light Stage dataset. Even though our results do not always attain the highest PSNR, they display better visual relighting quality than baselines.

We further conduct a user study (see Tab. 2) to quantify

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

DiLightNet IC-Light Neural Gaffer Total Relighting SwitchLight Ours

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

- Figure 7. In-the-wild portrait results: We display the input portrait, environment map, a reference image, rendered in Blender, and baseline comparisons. DiLightNet [57] shows artifacts from 3D reconstruction failures central to its pipeline. Neural Gaffer [19] generates inaccurate shadow contours on relit faces since it isn’t trained on human portraits. IC-Light [59] struggles with relighting due to its choice of background as the lighting condition. Total Relighting and SwitchLight [22, 28], trained on light stage data, produce soft shadows even under strong sunlight and alter skin tones. In contrast, our method achieves superior relighting while preserving subject identity.

Inputs DiLightNet IC-Light Neural Gaffer Total Relighting SwitchLight Ours GT

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

[Figure 127]

[Figure 128]

- Figure 8. Light Stage test results: We compare our method against baselines on the input portrait (bottom left) from the Light Stage test set relit with a target environment map (top left).

human perceptual preference for relighting. For each pair (our method vs. a baseline), participants are asked three

questions: (1) which method has better lighting (2) which has better image quality (3) which better preserves iden-

Ours 0.063 0.945 29.572 0.165 0.165 0.813 19.698 0.173 SwitchLight 0.088 0.911 21.432 0.198 0.141 0.853 20.299 0.152 IC-Light 0.108 0.874 20.283 0.284 0.172 0.789 17.440 0.195 DiLightNet 0.128 0.860 22.991 0.333 0.245 0.703 16.619 0.576 Neural Gaffer 0.102 0.900 25.327 0.357 0.196 0.788 19.311 0.247

- Table 1. Comparisons: We compare against baselines on a held-out set of our synthetic dataset and data rendered through a Light Stage. While trained only on synthetic data, our model performs comparably to SwitchLight, a commercial relighting method trained with Light Stage data.

Base Base + Multi-Task

Base + Inference Adaptation

Ours + Light Stage Ours

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

Figure 9. Ablations: We display the input portrait with its lighting condition and a reference image rendered in Blender (left). The Base configuration fails to reproduce the portrait’s textures and alters its identity. In contrast, Base + Multi-Task recovers some details, such as realistic skin tone (bottom row, yellow rectangle). The Base + Inference Adaptation configuration struggles with unseen textures and accessories (e.g., the cigarette, top row, red rectangle) and produces unnatural textures for sleeveless skin (bottom row, yellow rectangle). Meanwhile, Ours + Light Stage enhances details but inherits biases from Light Stage data and cannot remove strong shadows (neck region, bottom row, red rectangle). Finally, Ours achieves plausible lighting, harmonizes well with the background, and preserves key details from the input portrait.

IC-Light SwitchLight Neural Gaffer

Lighting 0.92 0.56 0.65 Quality 0.57 0.64 0.73 Identity 0.52 0.70 0.65

- Table 2. User Study: Preference rates indicate how often our method was preferred over baselines. For example, a rate of 0.92 under Lighting means our method was preferred 92% of the time over IC-Light. Based on 482 responses from 20 participants, our method consistently outperforms baselines in lighting, image quality, and subject identity, since all preference rates exceed 0.5. This highlights superior image quality over relighting methods [19, 22] and better lighting over harmonization methods [59].

ipants with diverse backgrounds, ranging from design to computer science. Results show that our methods outperforms baselines in perceived image lighting, quality, and identity preservation. Refer to the supplementary material for screenshots, Fig. 22 and Fig. 23, showcasing the precise format of our user study.

##### 4.4. Ablations

We conduct an ablation study to evaluate the contribution of our two key methods for domain adaptation: multi-task training (See Sec. 3.3) and inference-time adaptation (See Sec. 3.4).

We start with a Base configuration that excludes both multi-task training and inference time adaptation. Next, we examine the individual impact of each component by separately adding multi-task training, denoted as Base + MultiTask, and inference time adaptation, denoted as Base + In-

tity. All questions are presented as a 2-alternative forced choice (2AFC). We collect 482 responses from 20 partic-

Base 0.066 0.942 29.131 0.193 0.210 0.790 18.919 0.295 Base + Multi-Task 0.066 0.942 29.049 0.196 0.186 0.797 19.184 0.242 Base + Inference Adaptation 0.062 0.946 29.638 0.163 0.178 0.810 19.484 0.179

Ours 0.063 0.945 29.572 0.165 0.165 0.813 19.698 0.173 Ours + Light Stage 0.065 0.942 29.126 0.171 0.156 0.822 20.136 0.149

- Table 3. Ablations highlight the contributions of each component i.e. Multi-Task training and Inference-time Adaptation (Sec. 3.3 and Sec. 3.4 respectively). Adding Light Stage data during training improves performance on Light Stage Test set, and qualitatively improves details but brings lighting biases (See Fig. 9).

(1) (2) (3)

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

- (a) Lighting & Reference
- (b)
- (c)
- (d)

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Our Model

(Environment Map Conditioned)

[Figure 161]

[Figure 162]

[Figure 163]

Our Model

(Background Conditioned)

[Figure 164]

[Figure 165]

[Figure 166]

IC-Light

(Background Conditioned)

- Figure 10. Background vs. Environment Map as Lighting Conditions: The background provides limited lighting cues, leading the background-conditioned model to produce inaccurate lighting (note the wrong lighting direction in (1)-(c)). Even so, by utilizing our Synthetic Faces dataset, the background-conditioned model is able to generate plausible lighting, characterized by strong cast shadows, whereas harmonization methods such as IC-Light [59] fall short. See Fig. 7, Row 3 for the input portrait.

ference Adaptation. Our full configuration, combining both techniques is referred to as Ours. Finally, we explore the role of Light Stage data, by adding a fraction of it to each training batch, denoted as Ours + Light Stage. Please refer to the supplementary material, Appendix A, for more details on the Light Stage data.

Fig. 9 shows the effect of each configuration. Base loses important details from the input and fails to produce textures in clothing or accessories. Base + Multi-Task shows partial detail recovery, and Base + Inference Adaptation enhances finer details by leveraging information present in the

input portrait but still lacks photo-realism. Ours + Light Stage addresses identity and texture issues but inherits lighting biases from the Light Stage dataset. For example, under strong sunlight, it yields oversaturated images (see Fig. 20, in the supplementary material). Similar artifacts appear in other methods (e.g., SwitchLight) that are trained on Light Stage data. It also struggles to remove strong shadows, which are rarely present in Light Stage captures. Finally, Ours, generates images with plausible lighting, that are well harmonized with background and preserve important details from the input portrait. These findings are corroborated by our quantitative evaluation in Sec. 4.3.

##### 4.5. Environment map better than background

We train two variants of our model, one using a background and the other using an environment map as lighting condition. We observe that while in many cases, the backgroundconditioned model produces plausible lighting and appears well harmonized with the background, when we continuously rotate the environment map, lighting inconsistencies appear. See Fig. 10 for lighting inaccuracies in a background-conditioned method. Despite these, leveraging our synthetic dataset makes our background-conditioned model generate plausible self-occlusions, whereas harmonization methods such as [59] fail in this use case.

#### 5. Limitations & Discussion

Despite the advances proposed by our method both in terms of simplicity and image quality, it bears some limitations. In particular, our rendering pipeline could achieve a higher level of realism if we specialized it for rendering humans. Of note, it does not model unseen occluders casting shadows on the subject’s face, accessories such as hats, glasses, or even facial hair, which limits the diversity of lighting our method saw during training. Despite this, our method achieves great generalization capabilities. Furthermore, user editing of the light is cumbersome in the current representation; we could improve this aspect by proposing a parametric representation of the light, such as 3D point

lights or spherical Gaussians, that is easier to understand and edit for users. Additional qualitative examples illustrating the limitations of our method are provided in the supplementary material, see Fig. 25.

#### 6. Conclusion

We present SynthLight, a Portrait Relighting Diffusion model that relights in-the-wild images while garnering lighting supervision only from synthetic data. It underscores the potential of using synthetic data to achieve plausible portrait relighting, enabling interesting lighting effects such as strong cast shadows, catch light in the eyes, and inter-reflections.

#### Acknowledgement

We thank Weijie Lyu, Ziwen Chen, Haian Jin, Vikas Thamizharasan, Natalia Pacheco-Tallaj, Ryusuke Sugimoto and Christophe Bolduc for their insightful discussions and the many participants in our user study. We also thank Kalyan Sunkavalli and Nathan Carr for their support.

#### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 3
- [2] Jonathan T Barron and Jitendra Malik. Shape, illumination, and reflectance from shading. IEEE transactions on pattern analysis and machine intelligence, 37(8):1670–1687, 2014. 2
- [3] Sai Bi, Stephen Lombardi, Shunsuke Saito, Tomas Simon, Shih-En Wei, Kevyn Mcphail, Ravi Ramamoorthi, Yaser Sheikh, and Jason Saragih. Deep relightable appearance models for animatable faces. ACM Transactions on Graphics (ToG), 40(4):1–15, 2021. 2
- [4] Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3d faces. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 157–164. 2023. 2
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 5
- [6] Ziqi Cai, Kaiwen Jiang, Shu-Yu Chen, Yu-Kun Lai, Hongbo Fu, Boxin Shi, and Lin Gao. Real-time 3d-aware portrait video relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6221–6231, 2024. 2
- [7] Robert L Cook and Kenneth E. Torrance. A reflectance model for computer graphics. ACM Transactions on Graphics (ToG), 1(1):7–24, 1982. 2

- [8] Paul Debevec. Rendering synthetic objects into real scenes: Bridging traditional and image-based graphics with global illumination and high dynamic range photography. In Acm siggraph 2008 classes, pages 1–10. 2008. 2
- [9] Paul Debevec, Tim Hawkins, Chris Tchou, Haarm-Pieter Duiker, Westley Sarokin, and Mark Sagar. Acquiring the reflectance field of a human face. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 145–156, 2000. 2
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 3
- [11] P Kingma Diederik. Adam: A method for stochastic optimization. (No Title), 2014. 5
- [12] Craig Donner and Henrik Wann Jensen. A spectral bssrdf for shading human skin. Rendering techniques, 2006:409–418,

2006. 2

- [13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [14] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 3
- [15] Mingming He, Pascal Clausen, Ahmet Levent Ta¸sel, Li Ma, Oliver Pilarski, Wenqi Xian, Laszlo Rikker, Xueming Yu, Ryan Burgert, Ning Yu, et al. Diffrelight: Diffusionbased facial performance relighting. arXiv preprint arXiv:2410.08188, 2024. 1, 3
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3, 5
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4
- [18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [19] Haian Jin, Yuan Li, Fujun Luan, Yuanbo Xiangli, Sai Bi, Kai Zhang, Zexiang Xu, Jin Sun, and Noah Snavely. Neural gaffer: Relighting any object via diffusion, 2024. 1, 2, 3, 4, 6, 7, 8
- [20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 3
- [21] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024. 3
- [22] Hoon Kim, Minje Jang, Wonjun Yoon, Jisoo Lee, Donghyun Na, and Sanghyun Woo. Switchlight: Co-design of physicsdriven architecture and pre-training framework for human

- portrait relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25096–25106, 2024. 2, 3, 6, 7, 8
- [23] Theodore Kim, Holly Rushmeier, Julie Dorsey, Derek Nowrouzezahrai, Raqi Syed, Wojciech Jarosz, and AM Darke. Countering racial bias in computer graphics research. In ACM SIGGRAPH 2022 Talks, pages 1–2. 2022. 2
- [24] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. arXiv preprint arXiv:2404.07724, 2024. 3
- [25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 4
- [26] Tomohiro Mashita, Yasuhiro Mukaigawa, and Yasushi Yagi. Measuring and modeling of multi-layered subsurface scattering for human skin. In Virtual and Mixed Reality-New Trends: International Conference, Virtual and Mixed Reality 2011, Held as Part of HCI International 2011, Orlando, FL, USA, July 9-14, 2011, Proceedings, Part I 4, pages 335–344. Springer, 2011. 2
- [27] Thomas Nestmeyer, Jean-Fran¸cois Lalonde, Iain Matthews, and Andreas Lehrmann. Learning physics-guided face relighting under directional light. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5124–5133, 2020. 2
- [28] Rohit Pandey, Sergio Orts-Escolano, Chloe Legendre, Christian Haene, Sofien Bouaziz, Christoph Rhemann, Paul E Debevec, and Sean Ryan Fanello. Total relighting: learning to relight portraits for background replacement. ACM Trans. Graph., 40(4):43–1, 2021. 2, 7
- [29] Sylvain Paris, Franc¸ois X Sillion, and Long Quan. Lightweight face relighting. In 11th Pacific Conference onComputer Graphics and Applications, 2003. Proceedings., pages 41–50. IEEE, 2003. 2
- [30] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 5
- [31] Bui Tuong Phong. Illumination for computer generated pictures. In Seminal graphics: pioneering efforts that shaped the field, pages 95–101. 1998. 2
- [32] Pramod Rao, Gereon Fox, Abhimitra Meka, Mallikarjun BR, Fangneng Zhan, Tim Weyrich, Bernd Bickel, Hanspeter Pfister, Wojciech Matusik, Mohamed Elgharib, et al. Lite2relight: 3d-aware single image portrait relighting. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2
- [33] Mengwei Ren, Wei Xiong, Jae Shin Yoon, Zhixin Shu, Jianming Zhang, HyunJoon Jung, Guido Gerig, and He Zhang. Relightful harmonization: Lighting-aware portrait background replacement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6452–6462, 2024. 1, 2, 3, 4, 13

- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 3, 4, 5
- [35] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 3
- [36] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10,

2022. 3, 4

- [37] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 3
- [38] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015. 5
- [39] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 4
- [40] Soumyadip Sengupta, Angjoo Kanazawa, Carlos D Castillo, and David W Jacobs. Sfsnet: Learning shape, reflectance and illuminance of facesin the wild’. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6296–6305, 2018. 2
- [41] YiChang Shih, Sylvain Paris, Connelly Barnes, William T Freeman, and Fr´edo Durand. Style transfer for headshot portraits. 2014. 2
- [42] Zhixin Shu, Sunil Hadap, Eli Shechtman, Kalyan Sunkavalli, Sylvain Paris, and Dimitris Samaras. Portrait lighting transfer using a mass transport approach. ACM Transactions on Graphics (TOG), 36(4):1, 2017. 2
- [43] Zhixin Shu, Ersin Yumer, Sunil Hadap, Kalyan Sunkavalli, Eli Shechtman, and Dimitris Samaras. Neural face editing with intrinsic image disentangling. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5541–5550, 2017. 2
- [44] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 3
- [45] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3, 5
- [46] Tiancheng Sun, Jonathan T Barron, Yun-Ta Tsai, Zexiang Xu, Xueming Yu, Graham Fyffe, Christoph Rhemann, Jay

- Busch, Paul Debevec, and Ravi Ramamoorthi. Single image portrait relighting. ACM Transactions on Graphics (TOG), 38(4):1–12, 2019. 2, 3
- [47] Tiancheng Sun, Zexiang Xu, Xiuming Zhang, Sean Fanello, Christoph Rhemann, Paul Debevec, Yun-Ta Tsai, Jonathan T Barron, and Ravi Ramamoorthi. Light stage superresolution: continuous high-frequency relighting. ACM Transactions on Graphics (TOG), 39(6):1–12, 2020. 3
- [48] Tiancheng Sun, Kai-En Lin, Sai Bi, Zexiang Xu, and Ravi Ramamoorthi. Nelf: Neural light-transport field for portrait view synthesis and relighting. arXiv preprint arXiv:2107.12351, 2021. 2
- [49] Feitong Tan, Sean Fanello, Abhimitra Meka, Sergio OrtsEscolano, Danhang Tang, Rohit Pandey, Jonathan Taylor, Ping Tan, and Yinda Zhang. Volux-gan: A generative model for 3d face synthesis with hdri relighting. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1–9, 2022. 2
- [50] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 3
- [51] Yifan Wang, Aleksander Holynski, Xiuming Zhang, and Xuaner Zhang. Sunstage: Portrait reconstruction and relighting using the sun as a light stage. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20792–20802, 2023. 2
- [52] Zhibo Wang, Xin Yu, Ming Lu, Quan Wang, Chen Qian, and Feng Xu. Single image portrait relighting via explicit multiple reflectance channel modeling. ACM Transactions on Graphics (ToG), 39(6):1–13, 2020. 2
- [53] Erroll Wood, Tadas Baltruˇsaitis, Charlie Hewitt, Sebastian Dziadzio, Thomas J Cashman, and Jamie Shotton. Fake it till you make it: face analysis in the wild using synthetic data alone. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3681–3691, 2021. 3
- [54] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22428–22437, 2023. 4
- [55] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [56] Yu-Ying Yeh, Koki Nagano, Sameh Khamis, Jan Kautz, Ming-Yu Liu, and Ting-Chun Wang. Learning to relight portrait images via a virtual light stage and synthetic-to-real adaptation. ACM Transactions on Graphics (TOG), 41(6): 1–21, 2022. 2, 3
- [57] Chong Zeng, Yue Dong, Pieter Peers, Youkang Kong, Hongzhi Wu, and Xin Tong. Dilightnet: Fine-grained lighting control for diffusion-based image generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 1, 2, 3, 6, 7
- [58] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In

- Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3
- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Ic-light github page, 2024. 1, 2, 3, 5, 6, 7, 8, 9
- [60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [61] Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, and David W Jacobs. Deep single-image portrait relighting. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7194–7202, 2019. 2

## SynthLight: Portrait Relighting with Diffusion Model by Learning to Re-render Synthetic Faces

### Supplementary Material

#### A. Additional Results

We present additional results on input portraits from various stock websites such as Adobe Stock, Unsplash and Pexels as well as from our internal Light-Stage captures.

In-the-wild Test Portraits We demonstrate portrait relighting in the presence of strong sunlight to produce effects such as strong cast shadow from facial features, rim-effects in hair and specular highlights in Fig. 11. In Fig. 12, we demonstrate applying a studio environment map on in-thewild test portraits to accentuate prominent features such as facial contours and expressions in the portraits. In Fig. 13, we showcase that SynthLight generalises to several to several challenging cases such as a 2D cartoon, a boy with face paint and a full body portrait, beyond the diversity present in the synthetic training data.

Comparison with Baselines We evaluate SynthLight against several baseline methods on in-the-wild portraits. As shown in Fig. 14, SynthLight achieves lighting effects, such as the rim-light effect in hair and subsurface scattering in the ears. Additionally, Fig. 15 illustrates specular highlights on darker skin tones, a capability not replicated by baseline methods.

These limitations in baselines can be attributed to the nature of the underlying methods. For instance, IC-Light, being an image harmonization technique, is not trained on physically based rendered data and hence struggles with achieving these effects. Surprisingly, even relighting approaches, such as Neural Gaffer and SwitchLight fall short. While Neural Gaffer is trained on rendered images, it is not explicitly trained on human facial data, leading to limited effectiveness in such scenarios. Even SwitchLight, despite leveraging Light Stage data, does not capture these intricate lighting effects.

Ablations Fig. 19 showcases additional examples from our ablation study, illustrating the contribution of each component to the final qualitative results. The Base model struggles with identity preservation and fails to capture key details present in the input portrait. Adding either Base + Multi-Task or Base + Inference Adaptation improves detail recovery but remains insufficient for reproducing complex accessories, materials, and textures. For example, in Fig. 19, the cigarette in the input portrait (top) and the specularity of the choker necklace or the accurate dress color (bottom) are not faithfully replicated. In contrast, our method successfully addresses these challenges, achieving superior results.

We train an additional model, Ours + Light Stage, where Light Stage-rendered data is combined with the synthetic dataset for relighting. The Light Stage data is same as in [33], and consists of roughly 6000 light stage captures, rendered under 100 environment maps. Fig. 20 illustrates a spectrum of overexposure issues. Models trained purely on Light Stage data, such as SwitchLight, often suffer from severe overexposure, resulting in unnatural yellowish skin tones. Ours + Light Stage reduces this issue due to the inclusion of physically-based rendered synthetic data, though some overexposure persists. In contrast, our method trained exclusively on physically-based rendered synthetic data avoids this problem entirely, producing natural and balanced skin tones.

Comparison with Background-Conditioned Models In Fig. 21, we compare SynthLight, trained on our synthetic physically-based rendered data using environment maps with comprehensive 360° lighting information, to a background-conditioned variant of SynthLight, and ICLight. SynthLight excels at capturing nuanced lighting effects, such as cast shadows and self-occlusion, due to its precise environmental lighting inputs. The backgroundconditioned model, while able to generate these lighting effects, generates inaccurate lighting. IC-Light, an image harmonisation method, neither generates these effects nor generates accurate lighting.

#### B. Dataset

Synthetic Dataset In Fig. 24 we show more examples from our synthetic dataset of subjects rendered under different environment maps. Each group of 4 visualizes a subject rendered under 4 lighting conditions, highlighting variety across race and gender.

LAION Data Filtration We filter a subset of LAION by first running a face detector. Since this results in a large number of false positives, we additionally curate a set of query phrases whose matching images we seek to avoid. We filter the set of images further by evaluating the CLIP score of each image against the query words and retaining only those images whose CLIP score is below a threshold. Emperically, we set this threshold to 0.15.

#### C. Additional Implementation Details

Network Architecture The inputs to SynthLight are a portrait image and an environment map, both with a resolution of 512 × 512. The environment map is transformed

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

###### Figure 11. In order to demonstrate portrait lighting effects in the presence of strong sunlight such as strong cast shadows by facial features, rim-effects in hair and specular highlights, we show in-the-wild portraits relit using outdoor environment maps.

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

###### Figure 12. To demonstrate SynthLight’s ability to enhance portraits with studio-style lighting, we present in-the-wild portraits relit using a studio environment map, where the studio lights accentuate prominent features such as facial contours and expressions.

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

###### Figure 13. We show challenging in-the-wild portraits featuring 2D cartoon characters, child wearing face paint and a full body portrait, demonstrating that our method can generalize beyond the synthetic dataset seen during training.

[Figure 288]

[Figure 289]

SwitchLight Ours

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

IC-Light Neural Gaffer

[Figure 294]

[Figure 295]

SwitchLight Ours

- Figure 14. We show the input portrait, the environment map used to relight and a reference synthetic data rendering from Blender (left) and results from our method and baselines (right). SynthLight achieves lighting effects such as rim-light on hair (top) and subsurface scattering in ears (bottom). These cannot be generated by baselines.

[Figure 300]

[Figure 301]

SwitchLight Ours

- Figure 15. We highlight lighting effects that our method achieves in contrast to baselines such as specular highlights in response to lighting direction.

Test Synthetic Test Light Stage Method LPIPS↓ SSIM↑ PSNR↑ FN↓ LPIPS↓ SSIM↑ PSNR↑ FN↓

Ours (init SD 1.5) 0.061 0.945 30.002 0.143 0.177 0.808 19.317 0.188 Ours (init IC-Light) 0.057 0.948 30.268 0.125 0.165 0.813 19.698 0.173

- Table 4. Ablating initial checkpoint: We evaluate our method, initialized with IC-Light, against initialization with SD 1.5. All tables in both main paper and supplementary, including non-inference specific ablations, are generated with classifier-free guidance parameters, λT = 2, λI = 3. See main paper for detailed descriptions of them.

from high-dynamic range to low-dynamic range through the following sequence of operations: clipping, normalization, and exponentiation by 21.2. These inputs are encoded into latents of shape 64 × 64 × 4 using the VAE from Stable Diffusion.

SynthLight extends Stable Diffusion 1.5 by adding 8 additional channels to the first convolutional layer of the UNet, yielding a total of 12 channels (4 each for the denoising latent, input portrait, and environment map). The weights for these extra channels are initialized to 0.

Training and Inference We evaluate the performance of training with SD 1.5 initialization compared to ICLight initialization (see Tab. 4 and Fig. 17). While IC-Light initialization yields slightly better test set performance—prompting us to report it as our primary

method—our approach is not reliant on IC-Light. As shown in Fig. 17, even without IC-Light, our method generates advanced lighting effects, such as strong cast shadows and subsurface scattering in the ear. Conversely, without our training and inference procedures, IC-Light alone cannot produce the nuanced lighting effects (e.g. rim-effects, subsurface scattering and specular highlights) as illustrated in Fig. 14 and Fig. 15.

During training, a foreground mask is applied to the input portrait. Each condition—input portrait, environment map, and text prompt—is randomly dropped with a probability of 0.1. For inference, classifier-free guidance is applied with λI = 3 and λT = 2, using the prompt ”A nice person.”

Ablation Details Base serves as the baseline model, trained solely on the synthetic dataset. During inference,

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Without Finetune (IC-Light) With Finetune (Ours)

- Figure 16. We show the input portrait, the environment map used to relight and a reference synthetic data rendering from Blender (left) and results from our method and ablations (right). We demonstrate the impact of fine-tuning with our synthetic dataset. The base model, IC-Light, without this fine-tuning, is unable to relight images using an environment map.

it omits inference time adaptation, meaning no classifierfree guidance is applied to the input portrait. Base + MultiTask incorporates additional training with LAION data using a text-to-image task, where the input portrait and environment maps are randomly dropped. The relighting and text-to-image tasks are mixed in a 7:3 ratio. Base + Inference time Adaptation applies classifier-free guidance on input portrait, while keeping the same training configuration as Base. Finally, Ours combines both strategies. We train an additional model where Light Stage-rendered data complement the synthetic dataset for relighting – Ours + Light Stage.

#### D. User Study

We provide additional details about our user study. Screenshots illustrating the setup can be found in Fig. 22 and Fig. 23. The user study is conducted in three phases, with each phase focusing on a specific aspect of evaluation:

- Phase 1: Visual Quality In the first phase, participants are asked to specify their preference between our method and the baseline in terms of visual quality. Each comparison is presented as a two-option forced choice.
- Phase 2: Lighting In the second phase, participants evaluate the lighting of the renderings. To aid their judgment, we provide a synthetic reference rendered in Blender under the same environment map. This phase also uses a twooption forced choice format.

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Finetuning with IC-Light initialization

Finetuning with SD 1.5 initialization

- Figure 17. We show the input portrait, the environment map used for relighting, and a reference synthetic data rendering from Blender (left). On the right, we present results with IC-Light and SD 1.5 initialization for finetuning on our synthetic dataset. We note that while IC-Light initialization yields slightly better performance on our Light Stage Test set, both are comparable in terms of visual quality and achieve realistic lighting effects such as shadows and subsurface scattering.

Phase 3: Identity In the final phase, participants assess the identity of the renderings. A reference input portrait is provided, and users judge which option better preserves the subject’s identity. As with the previous phases, this is conducted as a two-option forced choice task.

General Instructions Participants are instructed to choose at random if making a selection is too difficult. At the beginning of each phase, a tutorial question is presented, where the answer is obvious. For example, in these cases:

- • One example has severe degradation in visual quality.
- • The lighting in one example is clearly incorrect.
- • One rendering fails to match the reference identity.

The correct answer and the reasoning are explained to participants to familiarize them with the task.

Study Statistics The study consists of 30 questions in total, including three tutorial questions (one per phase). Participants can opt to exit the study at any time. In total, we collected 482 responses from 20 participants over a oneweek period.

#### E. Limitations

Fig. 25 highlights some limitations observed with our method. We notice minor loss of detail, particularly in small or intricate facial features. This can be attributed to limited camera pose diversity in our synthetic dataset, i.e. headshot-only renderings, and the reliance on Stable Diffusion 1.5, which causes our method to inherit image reconstruction artifacts from Stable Diffusion’s VAE. These issues can be mitigated by leveraging larger models with with better VAEs, such as those in Flux or Stable Diffusion 3, and incorporating greater camera pose variation in our synthetic dataset.

Fig. 25 illustrate another failure mode where our method struggles with accurately capturing cloth textures. While this limitation is rare, it arises from the restricted range of materials and textures used for clothing in the synthetic dataset. Expanding the diversity and quality of the dataset’s cloth-related materials could effectively address this issue.

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

DiLightNet IC-Light Neural Gaffer Total Relighting SwitchLight Ours

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

[Figure 360]

[Figure 361]

- Figure 18. We show additional comparisons against baselines, illustrating, that unlike baselines, our method produces accurate lighting, that matches given reference, while preserving identity and maintaining high visual quality.

Base Base + Multi-Task

[Figure 366]

[Figure 367]

Base + Inference-time Adaptation

Ours

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

Base Base + Multi-Task

[Figure 372]

[Figure 373]

Base + Inference-time Adaptation

Ours

- Figure 19. We show the input portrait, the environment map used to relight and a reference synthetic data rendering from Blender (left) and results from our method and ablations (right). Examples show the contributions of each component in our proposed method. The Base model struggles with identity preservation and detail reproduction. Base + Multitask and Base + Inference-Time Adaptation improve detail recovery but fail to replicate complex features like accessories and textures. Our method successfully preserves identity and reproduces intricate details, such as the cigarette (top) and specularity of the necklace (bottom).

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

SwitchLight Ours + Light Stage Ours

- Figure 20. Overexposure issues due to Light Stage data. SwitchLight, trained purely on Light Stage data, suffers from severe overexposure and unnatural skin tones. Ours + Light Stage reduces this issue but retains some artifacts. Ours, trained on synthetic data alone, avoids these problems entirely.

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

Reference

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

SynthLight

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

Background Conditioned Model

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

IC-Light

- Figure 21. Background vs Environment Map as Lighting Condition: We compare SynthLight with a background conditioned model and IC-Light and show a reference model rendered in blender (top row). Background contains insufficient lighting cues, causing a background conditioned model to generate inaccurate lighting (columns 3-4). By leveraging our synthetic dataset, the background conditioned model can still generate lighting effects like strong cast shadows, whereas harmonization methods, for example, IC-Light can neither reproduce these effects or relight accurately.

[Figure 400]

[Figure 401]

###### Figure 22. User Study: We ask users to pick between our method and baseline on visual quality of image (top) and lighting, with a given reference (bottom).

[Figure 402]

###### Figure 23. User Study: We ask users to judge identity preservation by providing a reference identity and asking them to select between our method and baseline.

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

###### Figure 24. More examples from synthetic dataset: Each group of four represents a subject rendered under four different lighting conditions.

[Figure 435]

[Figure 436]

(a) We observe minor detail loss in facial features, such as the eyes, arising from limited camera pose diversity and Stable Diffusion 1.5’s VAE artifacts. Mitigations include using improved VAEs (e.g., Flux, Stable Diffusion 3) and enhancing pose variation in the dataset.

[Figure 437]

[Figure 438]

Figure 25. Limitations of our method include minor detail loss in full-body portraits and inaccuracies in cloth texture.

