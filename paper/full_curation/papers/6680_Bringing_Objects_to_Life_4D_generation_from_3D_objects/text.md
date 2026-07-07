# arXiv:2412.20422v2[cs.CV]27May2025

## Bringing Objects to Life: training-free 4D generation from 3D objects through view consistent noise

Ohad Rahamim1 Ori Malca1 Dvir Samuel1 Gal Chechik1,2

1Bar-Ilan University 2NVIDIA

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

The ice

cream is melting

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

An elephant

grows his ears

into long,

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

powerful wings

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

A powerful spell

is cast through

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

the purple flame

time

Figure 1: Our method, 3D24D, takes a static 3D object and a textual prompt describing a desired action. It then adds dynamics to the object based on the prompt to create a 4D animation, essentially a video viewable from any perspective. On the right, we display four 3D frames from the generated 4D animation. Each 3D frame contains an RGB image and a corresponding depth map on its bottom left.

### Abstract

Recent advancements in generative models have enabled the creation of dynamic 4D content — 3D objects in motion — based on text prompts, which holds potential for applications in virtual worlds, media, and gaming. Existing methods provide control over the appearance of generated content, including the ability to animate 3D objects. However, their ability to generate dynamics is limited to the mesh datasets they were trained on, lacking any growth or structural development capability. In this work, we introduce a training-free method for animating 3D objects by conditioning on textual prompts to guide 4D generation, enabling custom general scenes while maintaining the original object’s identity. We first convert a 3D mesh into a static 4D Neural Radiance Field (NeRF) that preserves the object’s visual attributes. Then, we animate the object using an Image-to-Video diffusion model driven by text. To improve motion realism, we introduce a view-consistent noising protocol that aligns object perspectives with the noising process to promote lifelike movement, and a masked Score Distillation Sampling (SDS) loss that leverages attention maps to focus optimization on relevant regions, better preserving the

Preprint. Under review.

original object. We evaluate our model on two different 3D object datasets for temporal coherence, prompt adherence, and visual fidelity, and find that our method outperforms the baseline based on multiview training, achieving better consistency with the textual prompt in hard scenarios. Project page

### 1 Introduction

Generative models are progressing rapidly, making it possible to generate images, videos, 3D objects, and scenes from text instructions only. It is now becoming possible to generate 4D content: dynamic

- 3D content conditioned on text prompts using text-to-4D methods Singer et al. [2023], Bahmani et al. [2024b], Ling et al. [2024], Miao et al. [2024], Yuan et al. [2024], Xu et al. [2024], Bahmani et al. [2024a], Deng et al. [2025], Zeng et al. [2024], 4D generation has the potential to change content creation, from movies and games to simulating virtual worlds.

Despite this promise, text-to-4D methods provide very limited control over the appearance of generated 4D content. Instead of generating a 4D dynamic object using text control only, latest work on 4D generation established better conditioning like image-to-4d Zhao et al. [2023], Ren et al.

- [2023], Gao et al. [2024], Yin et al. [2023] and video-to-4D Wu et al. [2024], Zhang et al. [2025], Xie et al. [2024], Ren et al. [2025], Park et al. [2025], Yang et al. [2025], where videos provide more information relevant to the dynamics. The latest advancement in conditioning is 3D-to-4D generation, specifically Animate3D Jiang et al. [2024] and Diffusion4D Liang et al. [2024], which train multi-view image-to-video diffusion models. These works capture a 3D object from multiple viewpoints and generate temporally consistent videos for each, ensuring coherence across different perspectives. To achieve this consistency, they rely on large-scale datasets of multi-view videos derived from existing

- 4D objects. However, these 4D objects are represented as meshes, which are inherently constrained by their fixed number of vertices and faces. As a result, approaches trained on this dataset tend to be more limited in handling evolution, volume change or growth deformation. Moreover, training-based approaches need to be retrained for new models, which may reduce usability. In this paper, we introduce a novel training-free method for generating 4D scenes from user-provided

- 3D representations, taking a simple approach that incorporates textual descriptions to govern the animation of the 3D objects. First, we train a “static“ 4D Neural Radiance Field (NeRF) based on the 3D mesh input, effectively capturing the object structure and appearance from multiple views, replicated across time. Then, our method modifies the 4D object using an image-to-video diffusion model [Xing et al., 2023, Zhang et al., 2023, Ho et al., 2022, HaCohen et al., 2024], conditioning the first frame on renderings of the input object. This maintains the identity of the original object and adds motion based on a provided text prompt.

Unfortunately, we find that applying this approach naively is insufficient because it dramatically reduces the level of dynamic motion. We propose two key improvements that both enhance the generation of dynamic movements and ensure better preservation of the input object. First, we design a new view-consistent noising strategy for 4D generation, which constructs a noise pattern associated with the rendered viewpoint during optimization. This association between the viewpoint and the noising approach enhances the generation process, resulting in more pronounced motion in the animated 4D output. Second, we introduce a masked variant of the SDS loss that uses attention maps obtained from the image-to-video model. This masked SDS focuses optimization on the object across temporally relevant regions of the latent space, enhancing the fidelity of object-related elements and better preserving its identity. We name our approach simply 3D24D.

We evaluate 3D24D on two different datasets, with a comprehensive set of metrics designed to assess various aspects of the generated 4D scenes across multiple viewpoints. We focus on four main criteria: temporal coherence of the generated video, adherence to the prompt description, and visual consistency with the initial 3D object. Given that only one 3D-to-4D generation method is publicly available Jiang et al. [2024], our comparison demonstrates that our approach achieves a better alignment with the text prompt while exhibiting more relevant dynamics, for prompts that elicit significant non-rigid deformations. We also find that our proposed view-consistency noising protocol and the attention-masked SDS enhance the dynamic content of the generated videos while still maintaining a high degree of consistency with the original object’s appearance. These improvements demonstrate that our method generates more realistic 4D scenes and also effectively balances visual quality and dynamic richness.

|[Figure 34]|
|---|

render

[Figure 35]

[Figure 36]

[Figure 37]

- 3D input object

Initialize a static 4D Adding Dynamics

𝑟𝑎𝑛𝑑𝑜𝑚 𝑛𝑜𝑖𝑠𝑒 − ℝ4𝑥16

[Figure 38]

[Figure 39]

[Figure 40]

- Figure 2: Workflow of our 3D24D approach, designed to optimize a 4D radiance field using a neural representation that captures both static and dynamic elements. First, a 4D NeRF is trained to represent the static object (plant, left), having the same 3D structure at each time step. Then, we introduce dynamics to the 4D NeRF by distilling the prior from a pre-trained image-to-video model. At each SDS step, we select a viewpoint and render both the input object, the noise sphere, and the 4D NeRF from the same selected viewpoint. These renders, along with the textual prompts, are then fed into the image-to-video model, and the SDS loss is calculated to guide the generation of motion while preserving the object’s identity. The noise is rendered from the sphere using the same viewpoint as the static object, providing better consistency at each step.

This paper makes the following contributions:

- 1. A novel training-free workflow for generating 4D scenes conditioned on a given 3D object model and on a text prompt.
- 2. We introduce two enhancements to improve motion generation and optimization: (a) A viewpoint-consistent noising strategy that aligns the noise injection process with the rendered viewpoint, creating more dynamic and coherent movement in the 4D scene. (b) A maskedSDS loss that uses the cross-attention mechanism of the diffusion model to enhance the optimization of 4D content.
- 3. Our method reaches improved 4D quality over current baselines on an extensive set of metrics, for objects that undergo significant non-rigid deformations.

2 Related Work

- 4D Generation. Recent advances in 4D generation span several domains, reflecting the diverse ways in which temporal and spatial information can be synthesized or manipulated. In text-to-4D approaches exemplified by earlier works Bahmani et al. [2024b], Ling et al. [2024], Miao et al.

view selector

״A blooming plant״

|∇𝜃ℒ𝐼2𝑉|
|---|

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

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Image-To-Video initialize model

[Figure 56]

render

4D NeRF

∇𝜃ℒ𝑚𝑎𝑠𝑘𝑒𝑑

Attention mask

𝑆𝐷𝑆

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Backpropagating gradients to the NeRF

- [2024], Yuan et al. [2024], Xu et al. [2024], Bahmani et al. [2024a], Yu et al. [2024], researchers focus on converting textual prompts into dynamic 3D scenes over time, leveraging techniques like diffusion-based models and Gaussian priors to ensure coherent spatiotemporal structure. Image-to-

- 4D methods Zhao et al. [2023], Ren et al. [2023], Gao et al. [2024], Sang et al. [2025], Yin et al.

- [2023], Nag et al. [2025], Li et al. [2024a], Sun et al. [2024], typically transform single or multiple

- 2D images into volumetric sequences, often employing flow estimation or learned shape priors to extrapolate consistent motion in 3D. In the video-to-4D Wu et al. [2024], Zhang et al. [2025], Xie et al. [2024], Yao et al. [2025], Ren et al. [2025], Chu et al. [2024], Li et al. [2024b] approaches expand existing 2D video into time-varying 3D representations, introducing techniques for multiview consistency and temporal alignment. Finally, 3D-to-4D works Jiang et al. [2024], Liang et al.

- [2024] tackle the challenge of adding a temporal dimension to static 3D models, enabling dynamic animations or evolutions of geometry through learned or procedural transformations. Collectively,

these methods highlight a rapidly evolving field aiming to bridge the gap between static 3D content and rich, time-aware volumetric experiences. Recent advancements in 3D-to-4D generation involve training multi-view video diffusion models on collected datasets of dynamic 3D assets. Diffusion4D Liang et al. [2024] focuses on efficient and spatially-temporally consistent 4D content generation by adapting video diffusion models trained on such datasets. Animate3D Jiang et al. [2024] trains a multi-view video diffusion model conditioned on multi-view renderings of a static 3D object, It introduces a spatio-temporal attention module to enhance spatial and temporal consistency.

- 3D consistent noising. 3D consistent noising in diffusion models addresses the challenge of generating coherent 3D content from 2D diffusion models by ensuring consistency across multiple views or time. A key approach involves generating noise directly in 3D space Liu and Vahdat

- [2025], such as attaching Gaussian noise as textures to 3D meshes and rendering them from different viewpoints to provide consistent noise input. This method leverages the equivariance properties of diffusion models trained with temporally consistent noise to produce video frames that align with the underlying 3D geometry, enhancing consistency in applications like video generation and scene editing. Consistent Flow Distillation Yan et al. [2025] also proposes applying multi-view consistent Gaussian noise directly to the underlying 3D object representation for text-to-3D generation.

Image to video generation. Image-to-video models [Blattmann et al., 2023] condition on an image alone, whereas text-to-video models condition on a textual prompt alone. The image-to-video approach allows users to create motion directly from the provided image, whereas text-to-video models generate motion from a textual prompt, limiting the user’s ability to explicitly control the motion dynamics. Notable works that incorporate both image and prompt conditioning include I2VGen [Zhang et al., 2023], which can generate high-resolution videos, and DynamiCrafter [Xing et al., 2023], a family of models designed to handle various input image resolutions. Both models project the input image into a text-aligned representation space using a pre-trained CLIP image encoder, similar to how text prompts are encoded. Newer approaches, such as HaCohen et al. [2024], employ a highly compressed Video-VAE to enable real-time, high-quality image-to-video generation.

### 3 Method

Our method receives an input 3D model (like a model of your favorite plant), and a textual prompt (like “A plant blooming"). Our goal is to animate the object, generating a 4D scene that reflects the described action in the prompt, yielding a 4D object of your favorite flower blooming. This approach transforms static assets into animated objects, adding life to 3D objects by introducing motion that aligns with the user’s descriptions. Our approach is illustrated in Figure 2.

#### 3.1 Initialize a static 4D from a 3D object

We first optimize a “static" 4D representation, where at every time t, the 4D NeRF captures the same static form of the input object. More specifically, beginning with the input 3D mesh, we randomly select a camera position. A ray is cast from the camera center through both the mesh and the neural representation. Along this ray, 3D points are sampled, and three properties are computed: color (RGB), depth, and surface normals from both representations. We then optimize the neural representation to align with the properties of the input object. This process is illustrated in the loss function:

Lstatic = LMAE(RGBmesh,RGBNeRF)

+ LMAE(Depthmesh,DepthNeRF)

+ LMAE(Normalmesh,NormalNeRF). (1)

Here, LMAE represents the mean absolute error between the properties of the mesh and those of the neural representation. Along this process, we also randomly sample the time dimension, resulting in a static 4D scene where the object remains unchanged over time.

#### 3.2 Adding Dynamics

Next, we aim to “bring our object to life" by introducing motion to the static 3D input object. To do this, we need to condition the SDS process on the input 3D object to achieve the desired 4D output. Here, we propose using image-to-video diffusion models to enhance this process. By conditioning the generation on the provided object, we align the 3D model’s render from the same viewpoint as the NeRF and use this render as input to the generation model, effectively anchoring the generated motion to the object’s identity. In our proposed SDS approach, renderings of the input object condition the distillation process, guiding the generation toward both the intended object appearance and desired dynamics. By rendering the object from all viewpoints, we can maintain the input object’s identity in the 3D space while introducing motion. To further ensure that the object remains consistent throughout the animation, we will also use multi-view loss to preserve its characteristics across different perspectives.

Optimizing the dynamic of the 4D scene using an image-to-video model can then be done using SDS [Poole et al., 2022] loss:

∂Xθ ∂θ

;td,y,Xobj − ϵ

(2)

∇θLI2V = Et

d,ϵ ω(td) ϵϕ zt

d

Here, ϵϕ and ϵ denote the predicted and actual noise for each video frame, respectively. We denote Xθ as a collection of V video frames, where Xθ = x0θ,...,xVθ −1 , which are rendered from the representation. Additionally, the rendered object is denoted as Xobj.

This SDS loss (Eq. 2) is then added to the static loss (Eq. 1), which is applied to the frame. This combined approach generates dynamic motion while ensuring that the 3D object remains consistent at t = 0. The overall loss is:

L = LI2V (x0θ,...,V ) + LMV (xiθ) + λLstatic(x0θ). (3) Here, λ is a weighting hyperparameter used to balance the magnitude of Lstatic with that of LI2V , and x0θ is the render at time t = 0.

#### 3.3 Viewpoint consistent Noising

When computing distillation scores from text-to-image or text-to-video models, it is common practice to randomly sample both a camera position and a noise pattern at each iteration. This means that noise patterns differ across camera viewpoints.

Our key observation is that this random sampling of noise patterns across views may reduce the consistency of appearances and motions guided by the cleaning process from different views. Trying to generate a 4D object consistent with several different motions may cause optimization to converge to a less-dynamic solution. Indeed, we observe this degradation of motion quality. See ablation Sec. 5, Table 2, and Figure 5 for more details.

We propose a viewpoint-consistent noise strategy that conditions the noise on both the rendered viewpoint s = (θs,ϕs) and a set of sampled time steps T = {ti | i = 0,...,V −1}, where ti ∈ [0,1] and ti+1 > ti. In standard video SDS, a viewpoint s and time steps T are randomly selected, and Gaussian noise ϵ ∼ N(0,I) is independently applied to any viewpoint. This approach neglects spatial and temporal structure, often resulting in incoherent motion dynamics.

To address this, we introduce a viewpoint consistent noising mechanism N(s,T) ∈ RC×V ×H×W that varies smoothly with both viewpoint and time, where H,W ∈ Z the space dimension C ∈ Z is the features dimention and V ∈ Z is the amount of frames. We construct this by associating a canonical 3D sphere mesh with gaussian noise attributes. Each face f of the sphere is assigned a latent noise vector nf ∈ RV ×C. Therefore, for a given viewpoint s, we render the sphere to obtain a pixel-to-face mapping, allowing us to construct a noise field S(s) ∈ RH×W×V ×C for each predefined time anchor tqi = Vi , where i = 0,...,V − 1. To support arbitrary smooth time sampling, we first randomly initialize a constant latent noise tensor Sˆ ∈ RH×W×V ×C at the start of optimization. This tensor remains fixed throughout optimization and is used to interpolate the noise fields at each sampled time step tˆ∈ T via:

N(s,tˆ) = √1 − τi · Sˆ(s) + √τi · S(s), (4)

where τi = tˆi − tqi denotes the fractional offset from the preceding temporal anchor. And for latent space H = 32,W = 32,C = 4 and a model video of V = 16 frames.

This strategy produces a structured noise tensor that is both viewpoint-consistent and temporally smooth, thereby enhancing the stability of optimization and improving the realism of generated

- 4D content. This approach maintains coherence in the dynamic effects while preserving spatial correspondence, allowing the generated motion to remain consistent regardless of the viewing angle.

#### 3.4 Attention-masked SDS

In our approach, since the 3D initial object already exists, we need to focus the loss specifically on the regions undergoing growth. In contrast, the standard implementation of SDS loss computes it over the entire object’s latent representation. By leveraging attention maps, we can guide the learning process toward the most relevant regions, ensuring that optimization is focused on the object. This approach ultimately enhances subject preservation while capturing more dynamic changes. For example, in Fig.2, the attention masks highlight higher values in the branches, where growth is occurring, while the pot has lower values since it remains unchanged. In our approach, we found that the first cross-attention mask between the input object rendering and the NeRF renders provides the most accurate masking, best highlighting the regions where growth occurs. The masked SDS loss is the pointwise product:

Lmasked−SDS = MLI2V , (5) where M is the attention mask.

#### 3.5 Modeling time

Video generative models typically operate on V = 16 frame sequences, but we aim for a 4D representation that can generate videos at any frame count, ensuring smooth and continuous dynamics without fixed frame limits. To achieve this, we currently sample a video from the NeRF by selecting a starting time t0 = U[0,1/V ] and uniformly sampling more V − 1 frames from the range, [t0,1], allowing continuous sampling Singer et al. [2023], Bahmani et al. [2024b]. However, this approach is suboptimal for image-to-video models, as it forces the static object to remain across all time steps, limiting dynamics. Additionally, the initial frame t = 0, this frame is rarely selected.

In 3D24D we propose a new time sampling strategy: we evenly select 16 frame times within the range [0,1], with the first frame fixed at t0 = 0 and later frame times adjusted with small noise tˆi = i/V +ϵi. Our time sampling strategy ensures uniform sampling across the entire time range while maintaining the input object condition requirements.

### 4 Experiments

#### 4.1 Dataset

We used two public datasets of 3D objects in our experiments. The first is the Google Scanned Objects (GSO) dataset [Downs et al., 2022], which consists of high-quality 3D scans of everyday items. The second is the Objaverse dataset [Deitke et al., 2023], which contains a large-scale collection of diverse 3D assets gathered from various sources. We selected objects from the GSO dataset and from objaverse, focusing on those that could support interesting growth dynamics and motion. For this purpose, we queried ChatGPT for objects and corresponding prompts that elicit significant non-rigid deformations. This resulted in a selection of 20 objects from GSO and 10 from Objaverse.

#### 4.2 Metrics

To evaluate our approach, we assess three main qualities: (1) preservation of the input object’s identity, (2) natural appearance of the generated 4D content, and (3) alignment with the text prompt. We use four evaluation metrics from Vbench [Huang et al., 2024].

(1) Motion Smoothness. We assess whether the motion in the generated video is smooth. To do so, we leverage motion priors from the video frame interpolation model [Li et al., 2023] ("smoothness"), as suggested in VBench [Huang et al., 2024]. (2) Dynamic Degree. Since static objects can also exhibit high motion smoothness, we introduce an additional metric to evaluate the presence of

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

an apple gets a

bite, revealing the

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

white flesh

beneath its skin

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

broccoli is growing and

blooming, its

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

green stalks

stretching

upward

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

a unicorn grows a

[Figure 97]

[Figure 98]

[Figure 99]

colorful

[Figure 100]

[Figure 101]

[Figure 102]

rainbow tail

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

a chocolate

is poured over

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

the ice cream

and drips from its side

time

- Figure 3: 3D24D brings various objects to life. On the left, we display the input object along with a textual prompt describing the desired action. On the right, we present four frames from the generated object, viewed from the front. Each 3D frame is split into an RGB image and its corresponding depth map, shown in the top right corner.

Google Scanned Objects Method Temporal style ↑ Motion smoothness ↑ Dynamic degree ↑ Identity Preservation ↓

Ours 17.8 ± 1.3 99.1 ± 0.07 50.5 ± 2.1 18.9 ± 2.0 Animate3D 13.7 ± 1.0 99.2 ± 0.008 21.5 ± 4.8 17.6 ± 2.1

Objaverse Method Temporal style ↑ Motion smoothness ↑ Dynamic degree ↑ Identity Preservation ↓

Ours 17.2 ± 1.4 99.1 ± 0.12 51.7 ± 7.5 19.4 ± 3.2 Animate3D 16.3 ± 1.5 99.3 ± 0.01 27.7 ± 1.2 15.8 ± 3.0

Table 1: Comparison between 3D24D and Animate3D. Metrics are explained in 4.2. Our 3D24D excels in dynamic degree and agreement with the prompt, while Animate3D better preserves object identity, as it does not attempt to evolve the objects—such as the melting of ice in the cream shown in Figure 1.

dynamic content in the video. Specifically, we quantify the amount of movement by computing optical flow between frames using RAFT [Teed and Deng, 2020], following the protocol in VBench [Huang et al., 2024]. (3) Agreement with prompt. To measure consistency with the prompt, we followed Vbench [Huang et al., 2024] and measured the similarity between the video frame features and the textual description features with ViCLIP [Radford et al., 2021] ("style"). (4) Agreement with input object. Ensuring visual consistency between the input 3D object and the generated 4D data. To evaluate this, we used LPIPS [Zhang et al., 2018] to measure the perceptual similarity between the input object renders and the generated frames, assessing the consistency of visual appearance over time.

We compute all metrics across four viewpoints with azimuth angles 0◦,90◦,180◦,270◦ and fixed elevation 0◦, averaging scores per object across views. Final results are reported as the mean and standard error across all objects.

Google Scanned Objects (20 objects) Method Temporal style ↑ Motion smoothness ↑ Dynamic degree ↑ Identity Preservation ↓ Ours (full) 17.8 ± 1.3 99.1 ± 0.07 50.5 ± 2.1 18.9 ± 2.0

- w.o. view consistency 16.5 ± 1.0 99.1 ± 0.06 41.4 ± 2.0 18.9 ± 1.8 w.o. masked SDS 17.0 ± 1.1 99.1 ± 0.06 46.6 ± 3.0 19.8 ± 3.1

Objaverse (10 objects) Method Temporal style ↑ Motion smoothness ↑ Dynamic degree ↑ Subject consistency ↓ Ours (full) 17.2 ± 1.4 99.2 ± 0.08 51.7 ± 7.5 19.4 ± 3.2

- w.o. view consistency 17.5 ± 1.3 99.1 ± 0.1 45.3 ± 3.9 18.5 ± 3.3 w.o. masked SDS 13.7 ± 1.3 99.1 ± 0.1 48.7 ± 4.4 17.7 ± 3.2

Table 2: Ablation study. Evaluating the contribution of various components of our method.

#### 4.3 Compared methods

Two previous studies have animated 3D objects using multi-view diffusion models, Animate3D Jiang et al. [2024] and Diffusion4D Liang et al. [2024]. Unfortunately, Diffusion4D did not released their code, and we can only present comparisons with Animate3D.

#### 4.4 Implementation details

We implement Image-to-Video SDS using the ThreeStudio framework [Guo et al., 2023]. Our implementation builds upon the text-to-4D capabilities of ThreeStudio [Bahmani et al., 2024b], replacing its viewpoint sampling protocol with the method proposed by Kasten et al. [2024].

Networks and rendering: We used a hash encoding-based neural representation, following the implementation in [Bahmani et al., 2024b]. For image-to-video model, we used DynamiCrafter [Xing et al., 2023], which generates videos at a resolution of 256x256. The input 3D object was rendered using PyTorch3D [Ravi et al., 2020], matching DynamiCrafter resolution with a rendering size of 256x256. The number of frames is V = 16.

Running Time: Our NeRF representation conversion was performed over 5000 iterations with uniform viewpoint sampling, taking approximately 10 minutes on an NVIDIA H100 GPU. The second phases was run for 20,000 steps and took ∼ 240 minutes.

### 5 Results

We first provide qualitative examples of 3D24D, then a quantitative and qualitative comparison of 3D24D with the baselines methods. Finally, quantitative and qualitative results of an ablation study, and the effect of different prompts.

Qualitative results: Figure 3 shows four examples of 4D generations (right) from a 3D object (left). Quantitative comparison with baselines: Table (1) compares 3D24D with the three baselines.

- 3D24D achieves far better agreement with the input object (identity preservation) in both LPIPS. It also generates a slightly more smooth and natural-looking 4D than other baselines. Agreement with the prompt is lower, presumably because the content adheres to the input object, which may deviate from the canonical representation of the corresponding text term. In other words, the text prompt may push an object to have other appearance than the given input object.

Qualitative comparisons: Figure 4 presents a qualitative comparison between 3D24D and Animate3D. While 3D24D aligns closely with the prompt and generates high dynamic motion, Animate3D fails to follow the prompt and produces a 4D output with limited dynamics.

Ablation analysis: We conducted an ablation study to evaluate the contributions of each component of 3D24D. Table 2 provides the quantitative results. Without view-consist-noise, the dynamic degree is strongly reduced, because the model tends to average across inconsistent videos. The dynamic degree is also hurt without attention-masked SDS Altogether, 3D24D achieves a balanced trade-off between preserving the input object’s identity, fulfilling the prompt, and maintaining a high dynamic degree. Figure 5 illustrates the ablation effect using two qualitative examples.

Input object 3D24D (ours) Animate3D

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

- Figure 4: Qualitative comparison. A render of the input object is shown on the left, alongside renders from 3D24D (middle) and Animate3D (right). In this example, our method generates a 4D object that is better aligned with the prompt "an elephant grows its ears as long as wings to fly,".

[Figure 124]

Input object wo consistent-view wo masked SDS Ours

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

broccoli is

growing and

blooming, its

green stalks

stretching upward

[Figure 133]

The ice cream is melting

- Figure 5: Qualitative ablation results demonstrate the contribution of each part of our method. Without our view-consistency noising the broccoli does not “bloom". Without our attention-masked SDS, the plant is less rich in details.

5.1 Noise consistency and video consistency

[Figure 134]

Figure 6: MSE across video pairs from near viewpoints, using view-consistent noise (y-axis) and random noise (x-axis). Yellow line represents the equality (y=x). Each dot denotes one object tested. View-consistent noise results in a lower mean MSE across all objects.

To gain more insight into the effect of viewconsistency noising in video generation across different viewing angles, we render images of the objects from angles 0◦,5◦,...,355◦, each with corresponding view-consistent noise. We then generate a video for each angle and compute the MSE between adjacent video pairs, i.e., (0◦,5◦),(5◦,10◦),...,(350◦,355◦). We then repeated this process, but this time using random noise.

- Figure 6 presents two video MSE values: from the view-consistent noise (y-axis) and random noise (x-axis). The yellow line is the identity function y = x. Each dot denotes the average MSE for one object. The view-consistent noise achieves lower MSE across all objects, demonstrating that it encourages the text-tovideo model to generate more consistent appearance and movement across camera viewpoints.

### 6 Conclusion

We present 3D24D, a novel method for animating 3D objects into dynamic 4D scenes from textual motion prompts. It uses Image-to-Video diffusion models, ensuring object consistency via rendered image conditioning and a tailored SDS loss. To boost motion realism, we introduce a view-consistent noise and an attention-guided masked SDS loss. 3D24D achieves better, prompt alignment, and visual fidelity, offering an effective solution for controlled 4D content creation.

Limitations: 3D24D builds on given video-generation models, and therefore inherits their underlying limitations such as limb confusion and missing object parts. Also, our current implementation has a large memory footprint, which may not work with new video generation models.

### References

Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. Tc4d: Trajectory-conditioned text-to-4d generation. In European Conference on Computer Vision, pages 53–72. Springer, 2024a.

Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006, 2024b.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Wen-Hsuan Chu, Lei Ke, and Katerina Fragkiadaki. Dreamscene4d: Dynamic multi-object scene generation from monocular videos. arXiv preprint arXiv:2405.02280, 2024.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023.

Yunze Deng, Haijun Xiong, Bin Feng, Xinggang Wang, and Wenyu Liu. Stp4d: Spatio-temporal-prompt consistent modeling for text-to-4d gaussian splatting. arXiv preprint arXiv:2504.18318, 2025.

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022.

Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wenchao Ma, Le Chen, Danhang Tang, and Ulrich Neumann. Gaussianflow: Splatting gaussian dynamics for 4d content creation. arXiv preprint arXiv:2403.12365, 2024.

Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, Zi-Xin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3d content generation. https://github.com/threestudio-project/threestudio, 2023.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Yanqin Jiang, Chaohui Yu, Chenjie Cao, Fan Wang, Weiming Hu, and Jin Gao. Animate3d: Animating any 3d model with multi-view video diffusion. arXiv preprint arXiv:2407.11398, 2024.

Yoni Kasten, Ohad Rahamim, and Gal Chechik. Point cloud completion with pretrained text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, Zhengzhong Tu, et al. 4k4dgen: Panoramic 4d generation at 4k resolution. arXiv preprint arXiv:2406.13527, 2024a.

Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9801–9810, 2023.

Zhiqi Li, Yiming Chen, and Peidong Liu. Dreammesh4d: Video-to-4d generation with sparse-controlled gaussian-mesh hybrid representation. Advances in Neural Information Processing Systems, 37:21377–21400, 2024b.

Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645, 2024.

Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8576–8588, 2024.

Chao Liu and Arash Vahdat. Equivdm: Equivariant video diffusion models with temporally consistent noise. arXiv preprint arXiv:2504.09789, 2025.

Qiaowei Miao, JinSheng Quan, Kehan Li, and Yawei Luo. Pla4d: Pixel-level alignments for text-to-4d gaussian splatting. arXiv preprint arXiv:2405.19957, 2024.

Sauradip Nag, Daniel Cohen-Or, Hao Zhang, and Ali Mahdavi-Amiri. In-2-4d: Inbetweening from two single-view images to 4d generation. arXiv preprint arXiv:2504.08366, 2025.

Jangho Park, Taesung Kwon, and Jong Chul Ye. Zero4d: Training-free 4d video generation from single video using off-the-shelf video diffusion model. arXiv preprint arXiv:2503.22622, 2025.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Nikhila Ravi, Jeremy Reizenstein, David Novotny, Taylor Gordon, Wan-Yen Lo, Justin Johnson, and Georgia Gkioxari. Accelerating 3d deep learning with pytorch3d. arXiv:2007.08501, 2020.

Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023.

Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems, 37:56828–56858, 2025.

Lu Sang, Zehranaz Canfes, Dongliang Cao, Riccardo Marin, Florian Bernard, and Daniel Cremers. Twosquared: 4d generation from 2d image pairs. arXiv preprint arXiv:2504.12825, 2025.

Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280, 2023.

Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024.

Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020.

Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, and Xiang Bai. Sc4d: Sparse-controlled video-to4d generation and motion transfer. In European Conference on Computer Vision, pages 361–379. Springer, 2024.

Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023.

Dejia Xu, Hanwen Liang, Neel P Bhatt, Hezhen Hu, Hanxue Liang, Konstantinos N Plataniotis, and Zhangyang Wang. Comp4d: Llm-guided compositional 4d scene generation. arXiv preprint arXiv:2403.16993, 2024.

Runjie Yan, Yinbo Chen, and Xiaolong Wang. Consistent flow distillation for text-to-3d generation. arXiv preprint arXiv:2501.05445, 2025.

Liying Yang, Chen Liu, Zhenwei Zhu, Ajian Liu, Hui Ma, Jian Nong, and Yanyan Liang. Not all frame features are equal: Video-to-4d generation via decoupling dynamic-static features. arXiv preprint arXiv:2502.08377, 2025.

Chun-Han Yao, Yiming Xie, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d 2.0: Enhancing spatio-temporal consistency in multi-view video diffusion for high-quality 4d generation. arXiv preprint arXiv:2503.16396, 2025.

Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023.

Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, László Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models. Advances in Neural Information Processing Systems, 37:45256–45280, 2024.

Yu-Jie Yuan, Leif Kobbelt, Jiwen Liu, Yuan Zhang, Pengfei Wan, Yu-Kun Lai, and Lin Gao. 4dynamic: Text-to-4d generation with hybrid priors. arXiv preprint arXiv:2407.12684, 2024.

Bohan Zeng, Ling Yang, Siyu Li, Jiaming Liu, Zixiang Zhang, Juanxi Tian, Kaixin Zhu, Yongzhen Guo, Fu-Yun Wang, Minkai Xu, et al. Trans4d: Realistic geometry-aware transition for compositional text-to-4d synthesis. arXiv preprint arXiv:2410.07155, 2024.

Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. 4diffusion: Multiview video diffusion model for 4d generation. Advances in Neural Information Processing Systems, 37: 15272–15295, 2025.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.

Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023.

### A Videos of generated 4D NeRFs

[Figure 135]

[Figure 136]

We provided a webpage with example objects and short videos of their 4D animations. To view the content, please unzip the Supplementary.zip file first. Then, open the webpage.html file to explore the 4D object videos.

### B view-consistency.

Figure 7 illustrates our viewpoint-consistent noising strategy. On the left, we visualize a fixed canonical noise field projected from one viewpoint, where the noise assigned to each pixel is derived from its corresponding face on a canonical 3D sphere. On the right, the same canonical noise field is reprojected from a different viewpoint. Crucially, the colored patch (highlighted in green, black, blue and red channels) remains consistent across views its position changes due to the viewpoint shift, but its local structure and values remain identical. The white arrow denotes the 3D reprojection path of the patch center between the two views. This example demonstrates how our method ensures consistent spatial alignment of noise across viewpoints, which helps preserve coherent appearance and motion cues throughout the distillation process.

[Figure 137]

[Figure 138]

- Figure 7: A specific noise patch on the sphere remains consistent across different camera viewpoints. The left and right panels show the same noise field rendered from two distinct viewpoints. The highlighted patch appears in different image locations due to the camera shift but retains identical structure and values. The white arrow indicates the 3D correspondence of the patch across views.

### C Sensitivity to prompt.

We explore the effect of different prompts, describing different dynamics, on the generated 4D scene. Figure. 8 shows the results when using the 3D object “Mario" and supplying it with three different dynamic prompts: “jumping" (top row), “running" (middle row), and “waving" (bottom row). The “Mario" figure, moves differently, according to the specified actions in the description

### D Fail cases

Some object classes, particularly in the Objaverse dataset, cause severe deformation and color changes in the input object. These severe deformations cause the evaluation metrics to deviate significantly from the norm. Example shown in Figure 9. For example, the object classes we identified include "Christmas tree", "legoman", "sunflower", "banana" and "balloon"

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Mario

[Figure 143]

[Figure 144]

jumping

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

[Figure 155]

[Figure 156]

Mario

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

running

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Mario

[Figure 177]

[Figure 178]

waving

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

Mario

[Figure 195]

[Figure 196]

walking

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

time

- Figure 8: Different prompts generate different 4D, matching the movement description. The object in question is a Mario figure (on the left), and we provide three distinct prompts that describe three different dynamics of the figure. On the right, the generated 4D illustrates the corresponding movements based on these prompts.

[Figure 205]

[Figure 206]

[Figure 207]

christmas tree

lights start to dance like flames,

flickering and glowing

[Figure 208]

[Figure 209]

[Figure 210]

a Lego man

grows a sword from his hand

[Figure 211]

[Figure 212]

[Figure 213]

a hand peeling a

banana

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

a balloon inflates,

until it reaches its

limit and bursts

with a loud pop

[Figure 219]

[Figure 220]

Input object Prompt Generated 4D Input object Prompt Generated 4D

- Figure 9: Despite plausible object-prompt pairings, the model occasionally fails to generate coherent or semantically aligned dynamics. These examples highlight limitations with 3D24D prompt and dynamic depiction

