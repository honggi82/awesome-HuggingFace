## Map2World: Segment Map Conditioned Text to 3D World Generation

Jaeyoung Chung*1 Suyoung Lee*†1 Jianfeng Xiang2 Jiaolong Yang2 Kyoung Mu Lee1

1 Seoul National University, {robot0321,esw0116,kyoungmu}@snu.ac.kr 2 Microsoft Research Asia, {t-jxiang,jiaoyan}@microsoft.com

# arXiv:2605.00781v1[cs.CV]1May2026

[Figure 1]

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

[Figure 5]

[Figure 6]

[Figure 7]

Input Text Prompt

“forest with tall green trees” “city consists of skyscrapers and roads”

Fig. 1: (Left) A generated sample of the center of the city, where the four edges consist of green forest. Our model takes as input a segment map with text prompts for each segment and produces a world that corresponds to the input segment map (violet box) with high quality (blue and orange boxes). (Right) Two examples of generated worlds in large-scale.

Abstract. 3D world generation is essential for applications such as immersive content creation or autonomous driving simulation. Recent advances in 3D world generation have shown promising results; however, these methods are constrained by grid layouts and suffer from inconsistencies in object scale throughout the entire world. In this work, we introduce a novel framework, Map2World, that first enables 3D world generation conditioned on user-defined segment maps of arbitrary shapes and scales, ensuring global-scale consistency and flexibility across expansive environments. To further enhance the quality, we propose a detail enhancer network that generates fine details of the world. The detail enhancer enables the addition of fine-grained details without compromising

* indicates equal contribution. † works done during internship at Microsoft Research.

overall scene coherence by incorporating global structure information. We design the entire pipeline to leverage strong priors from asset generators, achieving robust generalization across diverse domains, even under limited training data for scene generation. Extensive experiments demonstrate that our method significantly outperforms existing approaches in usercontrollability, scale consistency, and content coherence, enabling users to generate 3D worlds under more complex conditions.

Keywords: 3D World Generation · Structured Latent · 3DGS

### 1 Introduction

Three-dimensional (3D) world generation plays a pivotal role in a wide range of industrial applications, including immersive content creation, gaming, autonomous driving simulation, and virtual realities. While recent years have witnessed remarkable progress in 3D asset or object generation, extending these capabilities to the world scale remains challenging. The primary bottleneck is the lack of high-quality world-level datasets, which are far more difficult to construct than object-centric collections. Several works, such as BlockFusion [38], NuiScene [18], LT3SD [27], WorldGrow [22] or SCube [30], trained their own generator with existing datasets. However, these methods can only generate scenes within a limited domain, such as indoor or driving scenes, severely limiting their applicability.

To circumvent the absence of world-scale data, researchers often resort to leveraging the power of pre-trained diffusion models. One approach combines image diffusion models with a depth estimator to lift generated 2D images into 3D; however, these pipelines suffer from view-dependent inconsistencies, yielding incomplete reconstructions. Another direction employs video diffusion models to generate scene-like sequences. Still, these methods struggle with 3D-consistency and are fundamentally constrained by the limited memory span of diffusion models.

Recent 3D generation models, such as TRELLIS [39] and CLAY [45], demonstrate the feasibility of high-quality, domain-agnostic 3D generation at an asset scale. Building on these foundations, several attempts have been made to scale asset generators to the world level. For instance, SynCity [9] divides the ground into grids and generates proper 3D assets for each tile using TRELLIS. Then, the asset boundaries are blended using an image inpainting module. Although these approaches effectively leverage the expressive power of 3D asset generators, the failure to model relationships among generated assets poses challenges for large-world generation. For example, contextual disconnectedness between adjacent assets and inconsistencies in object scale can make the overall scene appear less coherent and harmonious. In addition, the assets should be arranged in a grid-like manner, which is impractical in real scenarios, as district boundaries are irregularly shaped.

In this paper, we present a novel text-conditioned 3D world generation framework called Map2World that builds on TRELLIS and explicitly addresses the limitations. To ensure global context and scale consistency, we introduce a

multi-diffusion strategy applied within the structured latent space. By coordinating overlapping diffusion windows, our method not only preserves the latent prior of TRELLIS but also enables seamless connections that extend beyond the boundaries of individual cubes. This design naturally supports arbitrary resolution, as the scene can be generated progressively while maintaining coherence across local neighborhoods. Furthermore, our approach can flexibly incorporate semantic maps as conditions without additional training, enabling controllable, semantically aligned scene synthesis. The ability to balance local detail, global context, and semantic structure allows our method to generate globally coherent, high-resolution 3D scenes that surpass the limitations of existing modular approaches. Through extensive experiments, we demonstrate that our framework significantly improves both structural fidelity and perceptual realism, paving the way toward scalable and versatile 3D scene generation. Our contributions can be summarized as follows:

- – Flexible segment map conditioning: generates 3D worlds from any type of user-defined segment maps, not limited to grid-based layouts.
- – Consistent detail enhancement: adds fine details to the assets while preserving overall structure.
- – Domain-generalized world generation: leverages powerful asset generator priors to achieve robust generation across domains with limited data.

- 2 Related work
- 3D world generation. Compared to 3D asset generation, creating a world is considered a more complex task, as a world consists of multiple objects that are harmonized with each other. We classify the approaches to generate a 3D world into two categories: 3D reconstruction from generated rendered views, and direct explicit 3D generation.

The former approach uses diffusion models to generate the images or videos, and reconstructs a 3D scene from the generated views. 2D image diffusion modelbased approaches [5,13,21,34,42,43] lift the pixels of the initial image into 3D with a monocular depth estimator. The image is then outpainted, and the generated region is lifted and stitched to the original scene to expand the world. The video diffusion model-based approaches [25,36,40,47] generate a video that navigates a virtual environment and reconstructs the scene from the video frames. Yet, these methods cannot guarantee the 3D consistency since the diffusion models are not trained to be aware of such consistency.

The other approach directly creates the scene with explicit 3D representations, achieving perfect 3D consistency. Some methods, such as BlockFusion [38], NuiScene [18], and LT3SD [27], train the generator from scratch to learn the distribution of cubes that correspond to a scene part. However, they share two major drawbacks resulting from the limited availability of scene datasets. First, these models can only generate geometry, and the generated scene does not have textures. Second, the domain of the generation is limited to the dataset they used

for training. SCube [30] and InfiniCube [26] add color estimation and use sparsevoxel-hierarchy [29] to improve global consistency, but they are also constrained to representing only driving scenes, which poses a significant limitation. Several works [9,48] leverage off-the-shelf 3D asset generators [4,23,29,39,45] and expand the scope of generation to 3D world. Although the quality of each asset can be prominent, the overall quality of the scene is highly dependent on the dataset. The strong prior knowledge of a high-quality 3D asset generator enhances both the quality and diversity of generated results; however, the spatial resolution of the generator’s output is too limited to represent an entire world. SynCity addresses this issue by dividing the space into multiple grid tiles, generating assets for each tile, and then merging them. Nevertheless, this approach suffers from weak connectivity between tiles and is unable to create large objects with arbitrary shapes. In contrast, our model supports seamless world generation from arbitrary segment maps using the latent fusion strategy.

Scaling diffusion models to large spatial extents. The idea of large-scale 3D generation can originate from the 2D diffusion literature, where training-free techniques have been developed to expand the resolution of pretrained models beyond their limits. Patch-based approaches synthesize large images by denoising overlapping patches and stitching them together, effectively bypassing the base model’s resolution and memory constraints [7, 14]. However, these methods typically rely on rigid grid tilings and heuristic blending, which can introduce boundary artifacts and offer limited flexibility for region-wise semantic control. Outpainting-based methods progressively extend an input canvas beyond its original field of view, enabling directional or large-aspect-ratio image expansion [3, 17,35,37]. Despite this flexibility, they remain confined to 2D planar domains and are usually driven by a single global context, providing only coarse control over the semantics of newly generated regions. Multi-window diffusion frameworks treat a large canvas as a collection of overlapping windows that are denoised jointly, allowing different prompts or conditions to be assigned to different spatial regions while maintaining global coherence through latent or score fusion [2,8,15,19,20]. Building on this idea, we extend the multi-window paradigm to volumetric 3D, operating on arbitrary-shaped, user-defined 3D regions and fusing their denoising trajectories in a shared 3D latent space to construct large, coherent 3D scenes.

### 3 Preliminary: Structured Latent and Generation Pipeline

We summarize TRELLIS [39], a state-of-the-art 3D asset generation model, using the new representation, structured latent, and the two-stage generation process. The structured latent (or SLAT) [39] encodes geometry and appearance with a set of local latents on a 3D grid,

s = {(zi,pi)}Li=1 , (1) where pi ∈ {0,1,...,N-1}3 is the positional index of an active voxel in the N3

- 3D grid, zi ∈ RC denotes a latent vector that encodes geometry and appearance at the corresponding position pi, and L is the number of active voxels.

Input Text Prompt and Map

[Figure 8]

“forest with tall green trees” “city consists of skyscrapers and roads”

[Figure 9]

Generated 3D World

[Figure 10]

[Figure 11]

[Figure 12]

Mask

Feature Fusion

[Figure 13]

Guide

|𝒢𝑆/𝐿|
|---|

|Spatial Regions in 3D Latent Space (Sec. 4.1)<br><br>[Figure 14]<br><br>| |
|---|---|
| | |

- (a) Expanding

- (b) Enriching Details in 3D Latent Space (Sec. 4.2)

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Interpolate

𝜃ℱMLP

𝒢𝑆/𝐿

[Figure 19]

- Fig. 2: Visualization of the overall generation pipeline. First, we estimate the structured latent for a large world conditioned by a segmentation map and a text prompt for each segment using latent fusion (top). Then, we further upscale the resolution of the generated scene using the detail enhancer. Detail enhancer includes an MLP layer to incorporate condition latent and noise, as well as a flow Transformer (bottom).

The structured latent is generated by a two-stage generation pipeline, from geometry to texture, based on the rectified flow model. Here, we assume the input text prompt y is given as a condition. For the first stage, they generate sparse structure {pi}Li=1 by iteratively applying the flow Transformer GS conditioned by y to the Gaussian noise. The fully denoised sparse structure latent is passed through the decoder, denoted as DS, and transformed into a signed 3D scalar field where the voxels with positive values are filled with contents and called active. We record a set of position indices of all active voxels as {pi}Li=1. In the second stage, the set of latent features {zi}Li=1 is also estimated through iterative denoising steps using the Transformer called GL, similar to the first stage. The structured latent can be transformed with various types of 3D representations, including 3D Gaussian splatting [16], radiance fields [11], and mesh [33], by putting the latent to the corresponding decoder (DL). In this experiment, we mainly focus on generating 3DGS as a final 3D representation.

### 4 Proposed Method

We describe how Map2World exploits the knowledge of a 3D asset generator to synthesize a 3D world. In Sec. 4.1, we present our latent fusion strategy to expand generation to a wider scene and to support conditioning on multiple text-driven spatial regions of arbitrary shapes. Sec. 4.2 proposes a detail enhancing network to enrich the details of the world by manipulating the structured latent. Here, we

explain the design choice of the detail enhancer to leverage the representational power of TRELLIS while preserving the content of the input large 3D scene. Finally, Sec. 4.3 introduces decoder fine-tuning to generate high-quality 3D scenes. The overall generation process is illustrated in Fig. 2.

#### 4.1 Expanding Spatial Regions in 3D Latent Space

The active voxel space of assets generated by the pre-trained TRELLIS model is limited to a cube of size 64. While this resolution may be sufficient for representing a single object, it is inadequate for modeling a large world composed of numerous objects. To achieve spatial expansion using the pre-trained model, we draw inspiration from spatial expansion techniques in 2D diffusion [2] and apply the idea to both 3D volumetric tensors and sparse structures in the rectified-flow Transformers, GS and GL. Our method supports precise and controllable world generation conditioned on fine-grained, pixel-level spatial regions annotated with textual conditions. We further describe a 3D noise initialization strategy that reinforces global scale consistency during progressive generation.

Latent fusion for rectified flow models. We conceptually split the space into a set of overlapping 3D cube windows {Ωj}, where each window spans a cubic region of size 64 and adjacent windows overlap by half of their spatial resolution. From a given position x, we update its local latent by aggregating the velocity field predictions vj(x) from all windows spatially covered as A(x) = {j | x ∈ Ωj }. Using a shared 3D Gaussian kernel W(·), the fused velocity for the position x is:

W(x − cj)vt,j(x|y) j∈A(x) W(x − cj)

vt(x|y) = j∈A(x)

, (2)

where cj is the center of Ωj. This strategy is applied to both GS and GL, and the latent at x is updated at every step according to the rectified-flow formulation:

st−1(x|y) = st(x|y) − ∆t · vt(x|y). (3)

Segment-map-guided latent fusion. Map2World supports 3D world generation conditioned by multiple text prompts with a segment map. Assuming that we have K labels, we define Mk as a binary map that indicates the region of the k-th label, and yk as the corresponding text prompt. Here, the velocity at the position x is calculated by the weighted sum of velocities estimated for each label:

K k=1 (Mk(x) ⊙ G(σt)) · vt(x|yk)

. (4)

v˜t(x) =

K k=1 (Mk(x) ⊙ G(σt))

G(σt) is a normalized 3D Gaussian kernel with zero mean and standard deviation σt to ensure smooth transition across regions, improving denoising

stability. Here, σt is controlled by the diffusion time step, starting from a large value to produce soft boundaries and gradually changing toward a sharp mask as t decreases. The smooth transitions of the segment mask across regions improve stability during the diffusion process. It is noteworthy that this pipeline can be applied to any shapes of Mk, whereas the existing framework, SynCity, can only generate objects of the same type in a square-shaped region.

Optimization for scale-aware initial latent. We observe two consistent behaviors in the noisy latent space. First, coarse geometric structures vary significantly with different initial noise samples. Second, similar initial noise samples tend to produce similar outputs. These observations indicate that the mapping from the initial latent xT to the final generation is locally smooth yet highly sensitive to the initial condition.

Although TRELLIS does not explicitly split the scale of scene, we empirically find that sparse structures exhibit different scene scales, and structures with similar scales tend to appear near each other in the latent space, as illustrated in Fig. 5. This suggests that valid sparse structures lie on a scale-dependent manifold. To steer generation toward a desired scale, we optimize the initial noisy latent inspired by the idea of initial noise optimization [1]. For the sparse structure ST and rectified flow models GS, we approximate the denoising trajectory by

S(t) ≈ ST + (1 − Tt )[GS(ST) − ST]sg, (5)

to calculate the gradient without backpropagating through the full denoising trajectory. Here, [·]sg denotes the stop-gradient operator. Under this approximation, the initialization is optimized via

Llinear = y − M([GL(ST) − ST]sg + ST) 22, (6)

where M denotes the target mask and y represents the target constraint for guiding the scale. Using a representative sparse structure empirically selected from samples, we adjust the scale by defining the excluded regions and the ground as the optimization target y. Further, to stabilize optimization under large learning rates required for early structural updates, we parameterize the sparse structure feature S in the spectral domain using a 3D FFT. This parameterization stabilizes the optimization trajectory and enables the use of a larger learning rate.

#### 4.2 Enriching Details in 3D Latent Space

The generated structured latent represents the geometry and the texture of an entire world. Since the entire world consists of numerous objects with a ground, it is almost impossible to encode all the details of the world into the latent space with limited capacity. Thus, we propose a detail-enhancing framework which increases the resolution of the world conditioned by the latent of the input world. It is challenging to train a network that directly adds details in 3D space since collecting such data pairs for detail enhancement is infeasible. Instead,

we design the detail enhancer to learn the relationship between coarse and fine scenes in the latent space, thereby maximizing the utilization of TRELLIS’ prior knowledge. To construct dataset pairs for training, we segment existing 3D scene data into cubes. Each cube is designed to contain a sufficient amount of content. Subsequently, each cube is divided into two parts along each axis, resulting in a total of eight smaller cubes. Both the large cube and the eight small cubes are then passed through the TRELLIS encoder. Since each latent corresponding to a small cube encodes a relatively smaller spatial region, it contains more detailed information than an input scene latent. The detail enhancer is trained to estimate the eight latent tensors, where each corresponds to a small cube, conditioned on the latent tensor of the input scene. While existing auto-regressive pipelines such as BlockFusion [38] and NuiScene [18] aim to predict latents for rendering individual cubes, they face challenges in maintaining global consistency, as these models rely solely on the information of neighboring cubes. In contrast, our method incorporates a latent variable that encapsulates the entire scene as a condition, and it can achieve high consistency across the whole scene.

We denote the input 3D representation inside a large cube as CO and the split small cubes as Cj, where j = 0,1,...,7. The structured latent of the large cube is denoted as sO = {(ziO,pOi )}i, and the structured latent of the small cube at index j is denoted as sj = {(zij,pji)}i.

Network architecture. The goal of the detail enhancer is to generate the structured latent of small cubes ({sj}) conditioned by the structured latent of the input scene (sO). Instead of estimating the structured latent for all small cubes at once, the detail enhancer receives the cube index (j) and predicts the corresponding structured latent(sj). When designing the network for detail enhancement, we propose an additional module that accommodates our new conditions and integrates it with the existing flow Transformer of TRELLIS. The rationale behind this design choice is twofold: first, due to the scarcity of world-level data, it is impractical to develop an entirely new architecture and train all parameters from scratch; second, since the input conditions also reside within TRELLIS’s latent manifold, leveraging the same structure facilitates more efficient processing and integration of condition information.

We employ structured latents from two different cubes as conditioning inputs for detail enhancing network. First, the structured latent of a large cube, sO, provides rough information about the target cube. To leverage the information, we extract only the part of the latent corresponding to the spatial location of our target cube, which we denote the truncated latent as sO|j. This structured latent plays a role analogous to a low-resolution image in image super-resolution. Thus, we concatenate the noise and the condition latent along the channel axis and define an MLP layer (Fθ) to mix the information, which is proven effective in a diffusion-based image enhancement field [31,32]. We denote the channel dimensions of the noise feature and latent feature tensor as c and C, respectively; then, the input dimension of the MLP layer is (c + C), and the output dimension is c, which is the same as the dimension of the noise feature.

Next, while sO|j can be sufficient to enhance the details of the corresponding small target cube, relying solely on this condition does not guarantee that the geometry and textures between adjacent cubes will be seamlessly connected. To achieve the seamless connection, we also incorporate the structured latent of adjacent cubes facing the target cube. The set of structured latent for adjacent cubes with target index j is denoted as sAdj(j). Here, Adj(j) denotes the position indices of cubes that are adjacent to the j-th cube. In order to concatenate the latent features with the noise, as we did with the large cube latent (sO|j), we temporarily expand the spatial size of the noise. The expanded noise is concatenated channel-wise with the adjacent latent and then compressed through the MLP layer. We note that the MLP layers for sO|j and sAdj(j) share the same parameters. The mixed feature after the MLP layer is then fed into the flow Transformer of the original model(GS and GL). From the predicted flow, we crop the expanded region and retain only the portion corresponding to the target cube. The overall process can be summarized as follows:

vθ sj,t = GS/L Fθ sjt,sO|j,sAdj(j) ,t , (7)

where sjt = (1 − t)sj + tε with ε ∈ N(0,I).

Initialization, training, and sampling. Similar to many diffusion fine-tuning strategies [28,46], our model is initialized to behave identically to the original TRELLIS before fine-tuning, and gradually incorporates the new conditioning as training progresses. Consequently, we use the pre-trained parameters of the original model for the flow Transformer. For the proposed MLP layer(Fθ), the weight matrix is initialized such that its diagonal elements are set to 1, while all other elements and the bias are set to 0. This ensures that, before training, the output of Fθ is identical to the noise regardless of the condition values. Then, the flow matching loss [24] is applied to fine-tune our model. The loss function to fine-tune the detail enhancer can be written as:

Lθ = Esj,t∥vθ sj,t − (ε − sj)∥22. (8) We note that only the parameters of the MLP layer (Fθ) are updated during

fine-tuning.

After fine-tuning, we consecutively apply the model to sample the latent from the noise. We auto-regressively estimate the structured latent of small cubes from index 0 to 7. When estimating the latent of cube 0 (s0), only the large cube structured latent is used since there is no information on adjacent cubes. When generating latent for subsequent cubes, the latent of adjacent cubes that have already been generated are also utilized as conditioning inputs. The eight estimated structured latents are merged and compose a latent that can represent a scene with enhanced details.

It is noteworthy to mention that we do not use classifier-free guidance (CFG) [12] when fine-tuning or sampling the model. CFG encourages the model to generate same output for both conditional and unconditional settings, and use

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

(a) Segment map (b) Top-view map (c) Generated world (d) Roaming view

- Fig. 3: Qualitative results conditioned by arbitrary shaped segment maps with userdefined text prompts. Our model creates the 3D world corresponds to the input map regardless of its size and shape. We note that Syncity cannot generate the scene from such a complicated segment map.

the difference to guide the denoising direction during sampling; however, in our case, the existence of conditions leads to substantial difference in reconstruction performance, thereby reducing the effectiveness of the CFG strategy.

#### 4.3 Fine-tuning SLAT Decoder

Since the original TRELLIS decoder was trained exclusively on data representing complete objects, its performance degrades when tasked with representing cubes extracted from partial scenes. To overcome the distribution discrepancy, we finetuned the latent decoder (DL) using the small cubes used for training of the detail enhancer. Specifically, we extracted structured latents from the 3D meshes of these small cubes using the pre-trained encoder. Then we fine-tuned the decoder by minimizing the difference between the reconstructed 3D representation and the original mesh. The loss function follows the original TRELLIS formulation. Through decoder fine-tuning, we can obtain higher-quality 3D representations from the structured latents estimated by the detail enhancer.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

(b) Ours - map (c) Ours - generated world (d) Ours - roam-

ing view

[Figure 36]

[Figure 37]

[Figure 38]

(a) Segment map

(e) SynCity map

(f) SynCity - generated world

(g) SynCity roaming view

- Fig. 4: Qualitative comparison with SynCity in a grid-type segmentation map. Syncity only supports generating assets on square-type tiles, and it lacks contextual continuity between adjacent tiles. Our model fills the entire world with appropriate objects in a seamless manner.
- 5 Experiments

Dataset curation. Using the labels of NuiScene43 [18], we choose a set of filtered 43 high-quality scene meshes from Objaverse [6] to train the detail enhancers. We exclude eight scenes without textural information and extract pairs from the remaining 35 scenes. We randomly cropped 500 cubes with various sizes (∈ {64,128,192,256}) for each scene. Here, we checked that each cube includes more than 10,000 vertices. After a total of 17,500 cubes have been extracted, we treat each cube as a large cube and split it into eight identical small cubes. For a large cube and the corresponding small cubes, we extract sparse structure latent and structured latent tensors using the pre-trained encoder models of TRELLIS. We use 16,000 pairs to train the enhancer and 1,500 pairs for validation.

#### 5.1 Comparison on World Generation

Map2World takes user-provided text prompts and a segmentation map comprising multiple segments labeled by each prompt as input, and generates a large-scale world that satisfies these conditions. We note that our model is not restricted to the size of the voxel grid of the original asset generator and can create the world with any user-defined dimensions. Also, the region of each segment can be formed in an arbitrary shape. Fig. 3 displays the rendered samples of the world created by Map2World, given the segment map with free-form segments. The three examples shown in the figure differ not only in their input prompts but also in the number of segments, the segment shapes, and the overall dimensions of the target world. SynCity cannot generate an appropriate 3D world under these conditions. In contrast, our model successfully produces a 3D world that

aligns well with both the segment map shapes and the text prompts in all cases. Fig. 3b displays a top-down view of the generated world, where each region closely matches the input segment map. Furthermore, as illustrated in Figs. 3c and 3d, the structures of adjacent regions are seamlessly connected, ensuring smooth transitions between segments.

Fig. 4 shows rendered views of the generated world where the map consists of grid-shaped segments. We compare the generated world by Map2World with SynCity [9] that also uses TRELLIS for scene generation. Our model surpasses SynCity in two key aspects. First, our method can generate a large connected structure, while SynCity cannot create large objects that occupy multiple tiles with the same labels. Moreover, for SynCity, the coherence among generated objects is weak, and gaps between tiles are often left unfilled, giving the impression of multiple disconnected assets placed together. Second, our model produces a relatively denser and more complex world. Even if SynCity shows better interobject coherence, the world looks like a grid-like arrangement of assets with numerous empty spaces, which is not considered a realistic example. In contrast, Map2World generates large and structurally complex assets within each segment and naturally connects the contents between adjacent segments. As shown in the rendering example on the right columns of Fig. 4, our results appear significantly closer to a real-world environment.

For a quantitative comparison, we generated text captions for 35 meshes from NuiScene43 and synthesized scenes from each caption using both SynCity and our proposed method. The rendered images were then evaluated across four criteria — sharpness, world completeness, coherence, and realism — using GPTscore [10]. When averaged across all images, our model achieved a score of 7.93/10, outperforming SynCity’s 7.48/10, indicating that our approach generates more complete and high-quality 3D worlds.

Moreover, we evaluate generated environments using a composite metric, World Quality (WQ), defined as

WQ = 0.15S + 0.45W + 0.25C + 0.15R. (9)

Here, S denotes sharpness, measuring the visual fidelity and clarity of geometric edges in the rendered scene. W represents world completeness and complexity, evaluating the scale of the environment, the richness of spatial structures, and the diversity of scene elements. C denotes coherence and consistency, assessing whether the generated layout forms a structurally consistent world, such as aligned roads, plausible spatial relationships, and the absence of geometric conflicts. R represents realism, measuring the overall plausibility of the scene, including lighting, materials, and geometric plausibility. All metrics are measured using the GPT 5.3 model.

The proposed WQ metric emphasizes the structural quality of the generated world rather than the fidelity of individual objects. By assigning the highest weight to world completeness, the metric prioritizes large-scale environmental structure and layout complexity, while coherence and realism ensure that increased scale does not come at the cost of structural inconsistency or visual artifacts.

- Table 1: Evaluation using the proposed World Quality (WQ) metric where WQ = 0.15S + 0.45W + 0.25C + 0.15R.

Images / Score GaussianCube SynCity Map2World

[Figure 39]

[Figure 40]

[Figure 41]

Sample Images

[Figure 42]

[Figure 43]

[Figure 44]

S (sharpness) 6.8 8.2 8.0 W (world completeness) 4.5 6.8 7.8 C (coherence) 5.0 7.6 7.9

- R (realism) 5.1 7.3 7.6 WQ 5.08 7.25 7.76

This design allows WQ to better capture the quality of generated environments as coherent worlds, making it particularly suitable for evaluating world-scale generative models.

Tab. 1 shows the samples of generated results with the same text prompt and calculated WQ metric using the same input prompt with different generation models. We selected GaussianCube [44] and SynCity [9] as the comparison frameworks. GaussianCube produces environments with noticeably lower visual quality and incomplete scene structures, resulting in low scores across most evaluation criteria. While SynCity benefits from the strong image-to-3D expressive capability of TRELLIS, thereby achieving high sharpness, the generated assets are often connected in an unnatural manner. These inconsistencies reduce the structural completeness of the generated environments and lead to a relatively lower world completeness score. In contrast, Map2World achieves consistently high scores across all evaluation criteria, producing environments that are not only visually clear but also structurally coherent and complete at the world level.

#### 5.2 Ablation Studies

Spectral parameterization for stable initial latent optimization. In the iterative optimization of initial noise for scale control illustrated in Fig. 5, the repeated generation increases both generation time and computational cost. Therefore, we introduce the spectral domain parameterization of the sparse structure S as described in Sec. 4.1, to stabilize the optimization trajectory, enable the use of a large learning rate, and enrich the target scale within a few optimization steps. We present the IoU and Dice plots measuring how well the target geometry constraint is satisfied in Fig. 6. In our setting, the blue

[Figure 45]

Prompt: “Trees on the grass”

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

64

64

64

64

| |
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

|𝑦|
|---|

|𝑦|
|---|

|𝑦|
|---|

48

40

36

32

32

32

32

28

24

|𝑦|
|---|

16

|𝑦|
|---|

|𝑦|
|---|

|𝑦|
|---|

0

0

0

0

(d) Scale Level 3

(b) Scale Level 1

(a) Scale Level 0

(c) Scale Level 2

- Fig. 5: Samples in various scales generated by the same prompt but different seeds.

[Figure 53]

- Fig. 6: Ablation on spectral-domain parameterization. This parameterization stabilizes the optimization process, enabling rapid convergence to the target within a few steps using a high learning rate.

curve shows that with a high learning rate of 9.0, the IoU and Dice scores reach approximately 0.9 on average within five optimization steps. In contrast, the orange curve, which directly optimizes the sparse structure without spectraldomain parameterization under the same setting, exhibits unstable behavior with large loss spikes and fails to converge. Reducing the learning rate to 1.0 (green curve) avoids divergence but requires significantly more optimization steps, increasing the computational burden.

Design choices for the detail enhancer. We conduct ablation studies on the design of the detail enhancer and decoder fine-tuning to justify our design choice.

- Fig. 7 shows rendered images from generated scenes by changing the options while keeping the condition, seed, and camera parameters the same. Fig. 7e is the rendered sample without using the detail enhancer; the generated structured latent with latent fusion is decoded and rendered.

First, we apply different network architectures to compare the performance. Specifically, we adapt the architecture from IP-Adapter [28,41] to fine-tune GS/L. The IP-adapter is also designed for fine-tuning the model to the new condition in a parameter-efficient manner. However, as shown in Fig. 7b, IP-Adapter fails to generate the appropriate structure and shows disconnectedness at the boundary. Also, we use classifier-free guidance during fine-tuning and sampling. The sampled result is shown in Fig. 7c. In our setting, the difference in generation

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

(a) Ours (b) IP-Adapter (c) Use CFG (d) No DL fine-

(e) No detail enhancing

tuning

Fig. 7: Qualitative comparison of rendered scenes on design choices for the detail enhancer. Best viewed when zoomed in.

- Table 2: Quantitative comparison for the detail enhancer design. The configuration of our choice and the best metric are written in bold.

FID↓

GS/L DL PSNR↑ LPIPS↓

Architecture CFG F.T. Incep.v3 DINOv2 CLIP

- (a) Concatenation No Yes 22.53 0.2137 16.98 32.67 11.79

- (b) IP-Adapter No Yes 20.28 0.2499 29.62 80.81 19.85

- (c) Concatenation Yes Yes 21.95 0.2174 19.06 38.15 21.32

- (d) Concatenation No No 22.08 0.2165 17.89 32.94 13.17

quality between the conditional and unconditional models would be substantial. When denoising with CFG, the difference between the denoised features in the conditional and unconditional settings becomes excessively large, causing severe geometric distortions and overly saturated colors. Finally, we analyze the effect of SLAT decoder fine-tuning in Fig. 7d. Although the improvement was less pronounced compared to the changes in the detail enhancer, we observed that the fine-tuned decoder produced sharper geometry and more refined textures.

To quantitatively compare the options, we use the test meshes and consider the rendered images from the mesh as ground truth. For the metrics, we compute PSNR, LPIPS, and FID scores on the rendered images from the test-set generated scenes and record the results in Tab. 2. For FID, we measured the scores across three baseline models: Inceptionv3, DINOv2, and CLIP. We note that our detail enhancer is a generator; thus, PSNR and LPIPS are not ideal metrics for evaluating quality. Still, our model achieves the best PSNR and LPIPS among the options, indicating the highest fidelity to the input condition. Our choice also yields the best FID metrics across all three baselines, suggesting that our model generates the most natural and high-quality detailed structures and textures.

### 6 Conclusion

Despite the growing demand for creating worlds, existing methods fail to achieve both flexible and scalable 3D world generation due to the shortage of appropriate

data. In this work, we present Map2World, a novel text-guided 3D world generation pipeline by exploiting the prior knowledge of TRELLIS [39], a popular

- 3D object generation model. We divide the generation process into two stages. First, we generate a structured latent that encodes the entire world. We impose constraints on noise initialization and employ a latent fusion strategy [2] during sampling to share features across the world to ensure global consistency. Also, our model supports segment-map-guided sampling and allows users to generate the world from a user-defined map composed of arbitrary-shaped regions, which is not available in any existing works. Next, we propose a detail enhancing module to further improve the quality of the generated world. The detail enhancer incorporates the global structure latent as a condition to preserve the consistency of generated details on a global scale. The detail enhancer is trained by fine-tuning an MLP layer with a small number of parameters that is added prior to the frozen TRELLIS generator, which enables maintaining the generalizable capacity of TRELLIS. Our model successfully generates worlds that are well aligned with the segment maps and exhibit smooth boundary transitions, even for segment maps of diverse sizes and shapes.

### References

- 1. Baek, S., Dong, E., Namazifard, S., Matthews, M.J., Yi, K.M.: Sonic: Spectral optimization of noise for inpainting with consistency. arXiv preprint arXiv:2511.19985

(2025)

- 2. Bar-Tal, O., Yariv, L., Lipman, Y., Dekel, T.: Multidiffusion: Fusing diffusion paths for controlled image generation. International Conference on Machine Learning

(2023)

- 3. Chen, Q., Ma, Y., Wang, H., Yuan, J., Zhao, W., Tian, Q., Wang, H., Min, S., Chen, Q., Liu, W.: Follow-your-canvas: Higher-resolution video outpainting with extensive content generation. arXiv preprint arXiv:2409.01055 (2024)
- 4. Chen, R., Chen, Y., Jiao, N., Jia, K.: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 22246–22256

(2023)

- 5. Chung, J., Lee, S., Nam, H., Lee, J., Lee, K.M.: Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384 (2023)
- 6. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 13142–13153 (2023)
- 7. Ding, Z., Zhang, M., Wu, J., Tu, Z.: Patched denoising diffusion models for highresolution image synthesis. In: The twelfth international conference on learning representations (2023)
- 8. Du, R., Chang, D., Hospedales, T., Song, Y.Z., Ma, Z.: Demofusion: Democratising high-resolution image generation with no $$$. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2024)
- 9. Engstler, P., Shtedritski, A., Laina, I., Rupprecht, C., Vedaldi, A.: Syncity: Trainingfree generation of 3d worlds. arXiv preprint arXiv:2503.16420 (2025)

- 10. Fu, J., Ng, S.K., Jiang, Z., Liu, P.: Gptscore: Evaluate as you desire. In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 6556–6576 (2024)
- 11. Gao, Q., Xu, Q., Su, H., Neumann, U., Xu, Z.: Strivec: Sparse tri-vector radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17569–17579 (2023)
- 12. Ho, J., Salimans, T.: Classifier-free diffusion guidance. In: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (2021)
- 13. Höllein, L., Cao, A., Owens, A., Johnson, J., Nießner, M.: Text2room: Extracting textured 3d meshes from 2d text-to-image models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7909–7920 (October 2023)
- 14. Jiang, W., Jangid, D.K., Lee, S.J., Sheikh, H.R.: Latent patched efficient diffusion model for high resolution image synthesis. In: Proceedings of the Computer Vision and Pattern Recognition Conference (2025)
- 15. Jiménez, Á.B.: Mixture of diffusers for scene composition and high resolution image generation. arXiv preprint arXiv:2302.02412 (2023)
- 16. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)
- 17. Kim, K., Yun, Y., Kang, K.W., Kong, K., Lee, S., Kang, S.J.: Painting outside as inside: Edge guided image outpainting via bidirectional rearrangement with progressive step learning. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision (2021)
- 18. Lee, H.H., Han, Q., Chang, A.X.: Nuiscene: Exploring efficient generation of unbounded outdoor scenes. arXiv preprint arXiv:2503.16375 (2025)
- 19. Lee, J., Jung, D.S., Lee, K., Lee, K.M.: Streammultidiffusion: real-time interactive generation with region-based semantic control. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2025)
- 20. Lee, Y., Kim, K., Kim, H., Sung, M.: Syncdiffusion: Coherent montage via synchronized joint diffusions. Advances in Neural Information Processing Systems

- (2023)

21. Li, H., Shi, H., Zhang, W., Wu, W., Liao, Y., Wang, L., Lee, L.h., Zhou, P.Y.: Dreamscene: 3d gaussian-based text-to-3d scene generation via formation pattern sampling. In: European Conference on Computer Vision. pp. 214–230. Springer

- (2024)

- 22. Li, S., Yang, C., Fang, J., Yi, T., Lu, J., Cen, J., Xie, L., Shen, W., Tian, Q.: Worldgrow: Generating infinite 3d world. arXiv preprint arXiv:2510.21682 (2025)
- 23. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 24. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: International Conference on Learning Representations

(2023)

- 25. Liu, X., Zhou, C., Huang, S.: 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with view-consistent 2d diffusion priors. Advances in Neural Information Processing Systems 37, 133305–133327 (2024)
- 26. Lu, Y., Ren, X., Yang, J., Shen, T., Wu, Z., Gao, J., Wang, Y., Chen, S., Chen, M., Fidler, S., Huang, J.: Infinicube: Unbounded and controllable dynamic 3d driving scene generation with world-guided video models (2024), https://arxiv.org/abs/ 2412.03934

- 27. Meng, Q., Li, L., Nießner, M., Dai, A.: Lt3sd: Latent trees for 3d scene diffusion. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 650–660 (2025)
- 28. Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y., Qie, X.: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453 (2023)
- 29. Ren, X., Huang, J., Zeng, X., Museth, K., Fidler, S., Williams, F.: Xcube: Largescale 3d generative modeling using sparse voxel hierarchies. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)
- 30. Ren, X., Lu, Y., Liang, H., Wu, J.Z., Ling, H., Chen, M., Fidler, S., Williams, F., Huang, J.: Scube: Instant large-scale scene reconstruction using voxsplats. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024)
- 31. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022)
- 32. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in neural information processing systems 35 (2022)
- 33. Shen, T., Munkberg, J., Hasselgren, J., Yin, K., Wang, Z., Chen, W., Gojcic, Z., Fidler, S., Sharp, N., Gao, J.: Flexible isosurface extraction for gradient-based mesh optimization. ACM Transactions on Graphics (TOG) 42(4), 1–16 (2023)
- 34. Shriram, J., Trevithick, A., Liu, L., Ramamoorthi, R.: Realmdreamer: Text-driven 3d scene generation with inpainting and depth diffusion. In: International Conference on 3D Vision (3DV) (2025)
- 35. Song, D.Y., Yu, J.J., Cho, D.: Progressive artwork outpainting via latent diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2025)
- 36. Wang, H., Liu, F., Chi, J., Duan, Y.: Videoscene: Distilling video diffusion model to generate 3d scenes in one step. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16475–16485. IEEE (2025)
- 37. Wu, T., Zheng, C., Cham, T.J.: Panodiffusion: 360-degree panorama outpainting via diffusion. arXiv preprint arXiv:2307.03177 (2023)
- 38. Wu, Z., Li, Y., Yan, H., Shang, T., Sun, W., Wang, S., Cui, R., Liu, W., Sato, H., Li, H., Ji, P.: Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. ACM TOG (2024)
- 39. Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J.: Structured 3d latents for scalable and versatile 3d generation. In: CVPR. pp. 21469–21480 (2025)
- 40. Yan, Y., Xu, Z., Lin, H., Jin, H., Guo, H., Wang, Y., Zhan, K., Lang, X., Bao, H., Zhou, X., et al.: Streetcrafter: Street view synthesis with controllable video diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 822–832 (2025)
- 41. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arxiv:2308.06721

(2023)

- 42. Yu, H.X., Duan, H., Herrmann, C., Freeman, W.T., Wu, J.: Wonderworld: Interactive 3d scene generation from a single image. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5916–5926 (2025)

- 43. Yu, H.X., Duan, H., Hur, J., Sargent, K., Rubinstein, M., Freeman, W.T., Cole, F., Sun, D., Snavely, N., Wu, J., et al.: Wonderjourney: Going from anywhere to everywhere. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6658–6667 (2024)
- 44. Zhang, B., Cheng, Y., Yang, J., Wang, C., Zhao, F., Tang, Y., Chen, D., Guo, B.: Gaussiancube: A structured and explicit radiance representation for 3d generative modeling. NeurIPS (2024)
- 45. Zhang, L., Wang, Z., Zhang, Q., Qiu, Q., Pang, A., Jiang, H., Yang, W., Xu, L., Yu, J.: Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG) 43(4), 1–20 (2024)
- 46. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF international conference on computer vision (2023)
- 47. Zhang, Q., Zhai, S., Martin, M.A.B., Miao, K., Toshev, A., Susskind, J., Gu, J.: World-consistent video diffusion with explicit 3d modeling. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21685–21695 (2025)
- 48. Zheng, K., Zhang, R., Gu, J., Yang, J., Wang, X.E.: Constructing a 3d town from a single image. arXiv preprint arXiv:2505.15765 (2025)

## Supplementary Materials for Map2World: Segment Map Conditioned Text to 3D World Generation

### S1 Implementation Details

#### S1.1 Details on Network Architectures

Feature interpolation for structured latent flow Transformer (GL). The latent tensor used to predict the positions of active voxels(pi) is a typical 5D tensor with a spatial dimension in the form of a standard 3D cube, and it is straightforward to concatenate with the conditional latent tensors along the channel dimension. However, when predicting the latent feature (zi) for each active voxel, the latent feature is not defined at every position but only at locations where active voxels exist. Consequently, the conditional latent feature corresponding to a target cube’s active voxel position may be absent. To address the issue, we estimate the conditional latent feature value at the target position using trilinear interpolation from neighboring positions. This trilinear interpolation is fully parallelized, enabling fast execution. To incorporate the latent information from adjacent cubes(sAdj(j)), we expand the spatial size of the target cube. In this process, the structured latent retains the original positions of the adjacent cube latents. As a result, the latent features of adjacent cubes are concatenated directly without interpolation, and self-attention is then performed on this expanded target cube latent.

The process for concatenating features is illustrated in Fig. S1. In the figure, the green and purple quadrangles indicate the spatial axes, with the vertical direction corresponding to the channel axis. To estimate the features at the positions of the target cube (blue bars), we apply trilinear interpolation from the original positions of the condition latent (orange bars). For the expanded part of adjacent cubes, we directly bring the features and the positions of adjacent condition latents (pink bars) and concatenate with noises that are expanded at the same positions of condition latents.

Proportion of number of fine-tuned parameters. We write the number of fine-tuned parameters and the total parameters in Tab. S1. The number of fine-tuned parameters is approximately 4% of the total, similar to the ratio observed in typical LoRA fine-tuning.

#### S1.2 Training Details

Hyperparameter settings. We follow the configurations of the original TRELLIS when fine-tuning the flow Transformers and the 3DGS decoder, with 100k

Target Cube Adjacent Cube

ConcatenatedfeaturesConditions

𝑝𝑗

Usesamecoordinate

𝑝𝒪|𝑗

𝑝𝐴𝑑𝑗(𝑗)

|Large cube condition (𝑠𝒪|j )|
|---|

Trilinear interpolation

|Adjacent cube condition (𝑠Adj(j) )|
|---|

Channel concatenation

Spatial concatenation Noised features

- Fig. S1: Visualization of concatenating feature process in structured latent flow Transformer (GL).

Table S1: Number of trained parameters and total parameters for each flow Transformer.

# of parameters Fine-tuned Fine-tuned Total Percent(%)

Base network

GS 25.39M 580.93M 4.37% GL 25.39M 625.82M 4.06%

iterations for all networks. The batch size is set to 4 for the geometry flow Transformer(GS), and 8 for the texture flow Transformer(GL) and SLAT decoder(DL). Each fine-tuning is performed on a single NVIDIA A100 80GB GPU and takes approximately 60 hours per network.

Fine-tuning detail enhancers. As mentioned in the last paragraph of Sec. 4.2, the cubes are sampled in an auto-regressive way during inference. Thus, the number of adjacent cubes used for the detail enhancer varies from 0 to 3, depending on the index of the cube being rendered. We design the model to handle a wide range of adjacent cube conditions within a single architecture. To ensure robust sampling results regardless of the number of adjacent cubes, we vary the number of adjacent cubes used for input conditions when training the detail enhancer. Specifically, during training, we select an adjacent cube along each of the x(left and right), y(back and forth), and z(up and down) axes relative to the target cube, totally extracting three adjacent cube candidates. Then, at each iteration, we randomly choose the number of adjacent cubes to use, ranging from 0 to 3, and select the corresponding cubes accordingly for that iteration. Our model is capable of adding sufficiently fine details using only the latent of the large cube, regardless of the existence of adjacent cube information. It is noteworthy that when computing the flow and subsequently calculating the loss(Eqs. (7)

Input

Large Cube Adjacent Cube

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Target Cube Enc Enc

[Figure 64]

|𝒔𝓞|𝑗|
|---|

|𝒔Adj 𝑗<br><br>|
|---|

[Figure 65]

Enc

Noise

Expanded Noise

[Figure 66]

𝒔𝑗

[Figure 67]

𝑭𝜽 𝓖𝓢/𝓛

[Figure 68]

𝓛𝜽

| | |
|---|---|
| | |

𝒗𝜽

Only use target cube pos.

- Fig. S2: Visualization of the estimation of the flow tensor and the calculation of the loss.

and (8)), we exclude the regions containing noise introduced for the adjacent cube condition and perform the computation only for the positions corresponding to the target cube.

### S2 Additional Experiment Results

#### S2.1 Measuring Consistency with Segment Map

To quantitatively assess the quality of region-specific generation under segmentbased conditioning, we compute a CLIP-Score–based alignment metric that measures how well each segmented region corresponds to its associated textual prompt. For a controlled comparison with SynCity, we adopt a grid-aligned generation protocol and report both qualitative and quantitative results in Fig. S3. As illustrated in Fig. S3a, our method successfully produces significantly larger and more structurally coherent worlds. We further compute region-wise CLIP-Scores based on top-view renderings, and the aggregated results are shown in Fig. S3b. The scores are averaged over 50 random seeds, capturing the relative alignment between the four segmented regions and their four textual descriptions. Using the ViT-H/14 CLIP model, our approach exhibits substantially clearer region–text separability compared to SynCity, indicating stronger adherence to the intended semantic layout.

Additional grid-wise samples and extended quantitative results, including CLIP-Scores obtained from alternative CLIP backbones (i.e., ViT-L/14, ViTbigG/14, SigLIP-So400m/14, PE-Core/14) as well as their softmax-normalized variants, are presented in Fig. S4. These results demonstrate that our method maintains consistent region-text alignment across different backbone configurations.

Furthermore, Figs. S5 to S7 show that our framework generalizes effectively to arbitrary user-defined segmentation maps. Given a free-form segmentation mask

[Figure 69]

[Figure 70]

Map2world(ours)SyncitySyncity

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Map2world(ours)

(a) Top-view Example

(b) CLIP-Score

- Fig. S3: Comparisons in CLIP-Score heatmap evaluated with ViT-H-14 baseline.

and textual descriptions, the proposed method reliably generates 3D scenes whose spatial structure and semantic content faithfully reflect the provided user inputs, highlighting the flexibility and robustness of our approach. For completeness, the quantitative results presented in these figures are aggregated over 50 randomly seeded generations, ensuring a consistent and statistically stable comparison across different CLIP backbones and scene configurations. Each figure shows representative samples and their averaged CLIP-Scores.

#### S2.2 Detail Enhancer

Evaluation protocol. Since the evaluation metrics require ground truth images, we used 1,500 cubes designated as test samples during Objaverse data curation. For each cube, we generated four images by rotating the yaw angle by 90° increments while maintaining a radius of 2 and looking toward the cube’s center, with a field of view (FoV) of 40°. This resulted in a total of 6,000 images for evaluation.

Recursive detail enhancement. Since the detail enhancer is trained by data with various metric scales, we can recursively apply the detail enhancer to further upscale the resolution of the scene. We visualize the rendered images of recursively enhanced worlds in Fig. S8. To evaluate the performance of the detail enhancer alone, we first generate the initial structured latent without applying any latent fusion strategy, producing a cube of size 64 as in the original TRELLIS output, which is displayed in Fig. S8a. This cube is then progressively upscaled by factors of 2 (Fig. S8b) and 4 (Fig. S8c) through the detail enhancer.

The top row shows the example from an urban scene from the Objaverse dataset. Here, the image in the left column is rendered by sequentially encoding and decoding the input 3D scene. The windows, before passing through the detail enhancer, are blurred, but their boundaries become sharper after passing through the enhancer. The second and third rows are rendered from TRELLIS-generated samples. Due to the output size limit of the original model, the images in the left column appear blurred and cannot represent fine-grained textures. Although the detail enhancer slightly transforms the contents of the original 3D world (×1), it generates sharper details while maintaining harmony with surrounding elements. Since our goal is not to perfectly reconstruct the original 3D world but to produce a high-quality world aligned with the input text prompt, we argue that minor content modifications are acceptable if they enhance overall quality.

#### S2.3 Additional Qualitative Results

In Fig. S9, we presented results obtained through latent fusion and the detail enhancer for a wider variety of segment maps. Our model demonstrates strong adherence to the given conditions and produces high-quality outputs even for irregularly shaped segment maps that SynCity cannot handle.

We also attached a video demo that displays the rendered views of the generated world from the given image segment maps. The video demonstrates that our model successfully generates a high-quality 3D world that faithfully satisfies the segment map condition.

### S3 Discussion

Limitations. Since our model is built on TRELLIS, Map2World share the same limitations of TRELLIS. First, TRELLIS uses absolute position encoding to give positional information to the model. However, when applying TRELLIS in our pipeline, merging small cubes can alter positional information before and after the merge, which may lead to changes in the decoded 3D structure. This issue can be mitigated by using a baseline model that uses relative positions instead of absolute positions, or by applying training strategies that enable the model to adapt effectively to changing positional encodings, thereby improving the overall quality of the results.

Future works. While the current detail enhancer is fine-tuned only with scenelevel cropped data, the generalizability of the enhancer can be improved by using more data. Specifically, quality can be improved by training on both object-level and world-level data. Additionally, since Objaverse primarily consists of simple meshes, incorporating datasets with more complex and realistic textures would encourage the detail enhancer to move toward a more photo-realistic direction.

Social impact. Our work focuses on controllable 3D scene generation and does not directly address personal data, identity modeling, or downstream decisionmaking tasks. Accordingly, we do not anticipate significant adverse social impacts. Potential applications—such as simulation, virtual environment creation, and content generation—are primarily creative or industrial, and the method does not inherently facilitate misuse beyond the general considerations associated with generative models. We encourage responsible use within appropriate ethical and safety guidelines.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

text1:“Spring village with

text2:“Summer village on green grass, stone houses with red roofs, green leafy trees”

wildflower meadows,

cherry blossoms, clear sky”

text4:“Winter village with wooden cabins, snow-

text3:“Autumn village with goldenred maple trees, fallen leaves”

covered roofs,

glowing windows, pine trees”

(a) Segment Condition (b) Examples of Generated Scenes with Different Random Seeds

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

(c) Scores across Different CLIP Models

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

(d) Softmax-Normalized Scores across Different CLIP Models

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

- text1:"Futuristic cyberpunk skyscrapers with neon lights, holographic billboards"
- text2:"Grand medieval stone castle with tall towers, arched gates, surrounding walls"

(a) Segment Condition

(b) Examples of Generated Scenes with Different Random Seeds

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

(c) Scores across Different CLIP Models

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

(d) Softmax-Normalized Scores across Different CLIP Models

##### Fig. S4: CLIP-Score heatmap visualization with grid-type segmentation map conditions. Best viewed when zoomed in.

text1:“Spring village with wildflower meadows, cherry blossoms, clear sky” text2:“Summer village on green grass, stone houses with red roofs, green leafy trees” text3:“Autumn village with golden-red maple trees, fallen leaves” text4:“Winter village with wooden cabins, snow-covered roofs, glowing windows, pine trees”

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

(a) Examples of User-defined Segmentation Inputs and Generated Scenes

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

(b) Scores across Different CLIP Models

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

(c) Softmax-Normalized Scores across Different CLIP Models

text1:“Futuristic city with glass skyscrapers, neon lights, holographic signs” text2:“Medieval stone castle with towers, battlements, wooden bridges” text3:“Traditional Asian village with tiled roofs, bamboo forests, paper lanterns” text4:“Desert nomad camp with beige tents, camels, sand dunes, golden sunlight”

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

(a) Examples of User-defined Segmentation Inputs and Generated Scenes

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

(b) Scores across Different CLIP Models

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

- text1:“Contemporary city with glass skyscrapers, busy streets, cars, flatland, square tile, realistic textures, meticulous detailing”
- text2:“Historic Ancient Rome with yellow stone architecture, marble columns, arches, yellow cobblestone streets, flatland, square tile, realistic textures, meticulous detailing”
- text3:“Futuristic solarpunk city with eco-friendly towers, green rooftop gardens, lush greenery integrated into buildings, flatland, square tile, realistic textures, meticulous detailing”

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

(a) Examples of User-defined Segmentation Inputs and Generated Scenes

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

(b) Scores across Different CLIP Models

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

|[Figure 190]|[Figure 191]|
|---|---|

|[Figure 192]|[Figure 193]|
|---|---|

|[Figure 194]|[Figure 195]|
|---|---|

[Figure 196]

[Figure 197]

[Figure 198]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 199]|[Figure 200]|
|---|---|

|[Figure 201]|[Figure 202]|
|---|---|

|[Figure 203]|[Figure 204]|
|---|---|

[Figure 205]

[Figure 206]

[Figure 207]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 208]|[Figure 209]|
|---|---|

|[Figure 210]|[Figure 211]|
|---|---|

|[Figure 212]|[Figure 213]|
|---|---|

[Figure 214]

[Figure 215]

[Figure 216]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

(a) ×1

(b) ×2

(c) ×4

- Fig. S8: Qualitative comparison of recursive detail enhancement. We do not use latent fusion tricks when generating the initial world (×1) in this experiment. ×4 indicates the latent tensors are passed through the detail enhancer twice to further upscale the resolution of the world.

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

(a) Segment map (b) Top-view map (c) Generated world (d) Roaming view

- Fig. S9: Additional qualitative results conditioned by arbitrarily shaped segment maps with user-defined text prompts. Our model creates a 3D world that corresponds to the input segment map regardless of the shape of each region. We stress that SynCity cannot generate the scene from such a complicated segment map.

