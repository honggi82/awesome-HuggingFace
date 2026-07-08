## FlashTex: Fast Relightable Mesh Texturing with LightControlNet

# arXiv:2402.13251v3[cs.GR]17Oct2024

Kangle Deng2⋆, Timothy Omernick1, Alexander Weiss1, Deva Ramanan2, Jun-Yan Zhu2, Tinghui Zhou1, and Maneesh Agrawala1,3

1 Roblox 2 Carnegie Mellon University 3 Stanford University

Abstract. Manually creating textures for 3D meshes is time-consuming, even for expert visual content creators. We propose a fast approach for automatically texturing an input 3D mesh based on a user-provided text prompt. Importantly, our approach disentangles lighting from surface material/reflectance in the resulting texture so that the mesh can be properly relit and rendered in any lighting environment. We introduce LightControlNet, a new text-to-image model based on the ControlNet architecture, which allows the specification of the desired lighting as a conditioning image to the model. Our text-to-texture pipeline then constructs the texture in two stages. The first stage produces a sparse set of visually consistent reference views of the mesh using LightControlNet. The second stage applies a texture optimization based on Score Distillation Sampling (SDS) that works with LightControlNet to increase the texture quality while disentangling surface material from lighting. Our algorithm is significantly faster than previous text-to-texture methods, while producing high-quality and relightable textures.

### 1 Introduction

Creating high-quality textures for 3D meshes is crucial across industries such as gaming, film, animation, AR/VR, and industrial design. Traditional mesh texturing tools are labor-intensive, and require extensive training in visual design. As the demand for immersive 3D content continues to surge, there is a pressing need to streamline and automate the mesh texturing process (Figure 1).

In the past year, significant progress in text-to-image diffusion models [42, 44, 45] has created a paradigm shift in how artists create images. These models allow anyone who can describe an image in a text prompt to generate a corresponding picture. More recently, researchers have proposed techniques for leveraging such 2D diffusion models for automatically generating textures for an input 3D mesh based on a user-specified text prompt [8,9,29,43]. But these methods suffer from three significant limitations that restrict their wide-spread adoption in commercial applications: (1) slow generation speed (taking tens of

⋆ Work done when interning at Roblox.

###### minutes per texture), (2) potential visual artifacts (e.g., seams, blurriness, lack of details), and (3) baked-in lighting causing visual inconsistency in new lighting environments (Figure 2). While some recent methods address one or two of these issues, none adequately address all three.

[Figure 1]

[Figure 2]

[Figure 3]

“Stone Goblet carved with runes and symbols”

“Marble goblet with white base color and red veins”

“Metal goblet intricately designed to reflect a Van Gogh painting”

“Wooden goblet with grain patterns”

Light Probe

- Fig. 1: We propose an efficient approach for texturing an input 3D mesh given a userprovided text prompt. Our generated texture can be relit properly in different lighting environments. The light probe shows the varied lighting environment. We suggest the readers check our video results of rotating lighting at our website.

In this work, we propose an efficient approach for texturing an input 3D mesh based on a user-provided text prompt that disentangles the lighting from surface material/reflectance to enable relighting (Figure 1). Our method introduces LightControlNet, an illumination-aware text-to-image diffusion model based on the ControlNet [63] architecture, which allows specification of the desired lighting as a conditioning image for the diffusion model. Our text-to-texture pipeline uses LightControlNet to generate relightable textures in two stages. In

- stage 1, we use multi-view visual prompting in combination with the LightControlNet to produce visually consistent reference views of the 3D mesh for a small set of viewpoints. In stage 2, we perform a new texture optimization procedure that uses the reference views from stage 1 as guidance, and extends Score Distillation Sampling (SDS) [39] to work with LightControlNet. This allows us to increase the texture quality while disentangling the lighting from surface material/reflectance. We show that the guidance from the reference views allows our optimization to generate textures with over 10x speed-up than previous SDS-based relightable texture generation methods such as Fantasia3D [9].

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(a) Mesh (b) Reference Lighting Composite RGB Diffuse Specular Composite RGB Diffuse Specular (c) Fantasia3D Texture (d) Our Texture

- Fig. 2: Given a 3D mesh of a helmet (a) and a lighting environment L, the reference rendering (b) depicts the “correct” highlights on the mesh due to L, by treating its surface reflectance as half-metal and half-smooth with a gray diffuse color. (c) The texture generated by the leading method Fantasia3D [9] is not properly relit as Fantasia3D bakes most of the lighting into the diffuse texture for the mesh and does not capture the bright highlights in the specular texture. (d) In contrast, our pipeline disentangles lighting from material, better capturing the diffuse and specular components of the metal helmet in this environment. Text prompt: “A medieval steel helmet.”

Furthermore, our experiments show that the quality of our textures is generally better than those of existing baselines in terms of FID, KID, and user study.

### 2 Related Work

Text-to-Image generation. Recent years have seen significant advancements in text-to-image generation empowered by diffusion models [42, 44, 45]. Stable Diffusion [44], for example, trains a latent diffusion model (LDM) on the latent space rather than pixel space, delivering highly impressive results with affordable computational costs. Further extending the scope of text-based diffusion models, works such as GLIGEN [23], PITI [56], T2IAdapter [31], and ControlNet [63] incorporate spatial conditioning inputs (e.g., depth maps, normal maps, edge maps, etc.) to enable localized control over the composition of the result. Beyond their power in image generation, these 2D diffusion models, trained on large-scale text-image paired datasets, also contribute valuable priors to various other tasks such as image editing [15,28], 3D generation [39,41], and 3D editing [13,19,54,66]. Text-to-3D synthesis. The success of text-to-image synthesis has sparked considerable interest in its 3D counterpart. Some approaches [21,34,48,65] train a text-conditioned 3D generative model akin to 2D models, while others employ 2D priors from pre-trained diffusion models for optimization [9,22,25,29,39,50,55,57] and multi-view synthesis [27,47]. For instance, DreamFusion [39] and Score Jacobian Chaining [55] were the first to propose Score Distillation Sampling to optimize a 3D representation using 2D diffusion model gradients. Zero-1-to-3 [27] synthesizes novel views using a pose-conditioned 2D diffusion model. Yet, these methods often produce blurry, low-frequency textures that bake lighting into surface reflectance. Fantasia3D [9] can generate more realistic textures by incorporating physics-based materials. However, the resulting materials remain entangled with lighting, making it difficult to relight the textured object in a new lighting environment. In contrast, our method effectively disentangles the lighting and surface reflectance texture. Concurrent to our work, MATLABER [60]

Text Prompt: “Doc Martens Boot”

Noise Text Prompt

###### Stage 1: Multi-view Visual Prompting

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

LightControlNet

###### UNet LSDS

Rendered Image

LightControlNet

[Figure 16]

Lrecon

Mesh

LightControlNet SDS Loss

Rendered Conditioning Image

Icond Iref

Reference Image

R( ( (·)),L⇤,C⇤)

###### Stage 2: Texture Optimization Canonical views and lighting

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

 ( (p)) = (kc,km,kr,kn)

LSDS

[Figure 23]

###### p

R( ( (·)),L,C)

[Figure 24]

[Figure 25]

[Figure 26]

#### LSDS

[Figure 27]

 ( (·))

Texture Hash Grid

Random views and lighting

- Fig. 3: Text-to-Texture pipeline. Our method efficiently synthesizes relightable textures given a 3D mesh and text prompt. In stage 1 (top left), we use multi-view visual prompting with our LightControlNet to generate four visually consistent canonical views of the mesh under fixed lighting, concatenated into a reference image Iref. In

- stage 2, we apply a new texture optimization procedure using Iref as guidance along with a multi-resolution hash-grid representation of the texture Γ(β(·)). For each iteration, we render two batches of images using Γ(β(·)) – one using the canonical views and lighting of Iref to compute a reconstruction loss Lrecon and the other using randomly sampled views and lighting to compute an SDS loss LSDS based on LightControlNet.

aims to recover material information in text-to-3D generation using a material autoencoder. Our method, however, differs in approach and improves efficiency.

- 3D texture generation. The area of 3D texture generation has evolved over time. Earlier models either directly took 3D representations as input to neural networks [4,49,61] or used them as templates [36,38]. While some methods also use differentiable rendering to learn from 2D images [4, 14, 38, 61], the models often fail to generalize beyond the limited training categories. Closest to our work are the recent works that use pre-trained 2D diffusion models and treat texture generation as a byproduct of text-to-3D generation. Examples include LatentPaint [29], which uses Score Distillation Sampling in latent space, Text2tex [8], which leverages depth-based 2D ControlNet, and TEXTure [43], which exploits both previous methods. Nonetheless, similar to recent text-to-3D models, such methods produce textures with entangled lighting effects and suffer from slow generation. On the other hand, TANGO [10], generates material textures using a Spherical-Gaussian-based differentiable renderer, but struggles with complex texture generation. A concurrent work, Paint3D [62], aims to generate lightingless textures, yet it cannot produce material-based textures like ours. Material generation. Bidirectional Reflection Distribution Function (BRDF) [35] is widely used for modeling surface materials in computer vision and graphics. Techniques for recovering material information from images often leverage

neural networks to resolve the inherent ambiguities when applied to a limited range of view angles or unknown illuminations. However, these methods often require controlled setups [24] or curated datasets [2, 12, 58], and struggle with in-the-wild images. Meanwhile, material generation models like ControlMat [51], Matfuse [52], and Matfusion [46] use diffusion models for generating SpatiallyVarying BRDF (SVBRDF) maps but limit themselves to 2D generation. In contrast, our method creates relightable materials for 3D meshes.

### 3 Preliminaries

Our text-to-texture pipeline builds on several techniques that have been recently introduced for text-to-image diffusion models. Here, we briefly describe these prior methods and then present our pipeline in Section 4.

ControlNet. ControlNet [63] is an architecture designed to add spatially localized compositional controls to a text-to-image diffusion model, such as Stable Diffusion [44], in the form of conditioning imagery (e.g., Canny edges [5], OpenPose keypoints [7], depth images, etc.). In our work, where we take a 3D mesh as input, the conditioning image Icond(C) is a rendering of the mesh from a given camera viewpoint C. Then, given text prompt y,

Iout = ControlNet(Icond(C),y),

where the output image Iout is conditioned on y and Icond. ControlNet introduces a parameter s that sets the strength of the conditioning image. When s = 0, the ControlNet simply produces an image using the underlying Stable Diffusion model, and when s = 1, the conditioning is strongly applied.

Score Distillation Sampling (SDS). DreamFusion [39] optimizes a 3D NeRF representation θ [1,30] conditioned on text prompts using a pre-trained 2D textto-image diffusion model. A differentiable renderer R applied to θ with a randomly sampled camera view C then generates a 2D image x = R(θ,C). A small amount of noise ϵ ∼ N(0,1) is then added to x to obtain a noisy image xt. DreamFusion leverages a diffusion model ϕ (Imagen [45]) to provide a score function ϵˆϕ(xt;y,t), which predicts the sampled noise ϵ given the noisy image xt, text prompt y, and noise level t. This score function can update the scene parameters θ, using the gradient calculated by SDS:

∂x ∂θ

∇θLSDS(ϕ,x) = Et,ϵ w(t)(ˆϵϕ(xt;y,t) − ϵ)

,

where w(t) is a weighting function. During each iteration, to calculate the SDS loss, we randomly choose a camera view C, render the NeRF θ to form an image x, add noise ϵ to it, and predict the noise using the diffusion model ϕ. DreamFusion optimzes for 5,000 to 10,000 iterations. In our work, we introduce an illumination-aware SDS loss to optimize surface texture on a mesh to suppress inconsistency artifacts and simultaneously separate lighting from the material.

[Figure 28]

Conditioning Image “Leather …” “Metal …” “Wooden …”

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Non-Metal, Not Smooth

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

stack

Mesh

Half-Metal, Half Smooth

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Conditioning Image

Pure Metal, Smooth

Full Color Image

(a) Rendering Conditioning Image

(b) Inference with LightControlNet

- Fig. 4: (a) LightControlNet requires a conditioning image that specifies desired lighting L for a view C of a 3D mesh. To form the conditioning image, we render the mesh with the desired L and C using three different materials: (1) non-metal, not smooth, (2) half-metal, half-smooth, and (3) pure metal, smooth, and then combine the renderings into a single three-channel image. (b) LightControlNet is a diffusion model conditioned on such light-conditioning images and text prompts.

### 4 Method

Our text-to-texture pipeline operates in two main stages to generate a relightable texture for an input 3D mesh with a text prompt (Figure 3). In Stage 1, we use multi-view visual prompting to obtain visually consistent views of the object from a small set of viewpoints, using a 2D ControlNet. Simply backprojecting these sparse views onto the 3D mesh could produce patches of high-quality texture but would also generate visible seams and other visual artifacts where the views do not fully match. The resulting texture would also have lighting bakedin, making it difficult to relight the textured mesh in a new lighting environment. Therefore, in Stage 2, we apply a texture optimization that uses a ControlNet in combination with Score Distillation Sampling (SDS) [39] to mitigate such artifacts and separate lighting from the surface material properties. In both stages, we introduce a new illumination-aware ControlNet that allows us to specify the desired lighting as a conditioning image for an underlying text-to-image diffusion model. We call this model LightControlNet and describe how it works in Section 4.1. We then detail each stage in Section 4.2 and Section 4.3, respectively.

##### 4.1 LightControlNet

LightControlNet adapts the ControlNet architecture to enable control over the lighting in the generated image. Specifically, we create a conditioning image for a 3D mesh by rendering it using three pre-defined materials and under known lighting conditions (Figure 4). These renderings encapsulate information about the desired shape and lighting for the object, and we stack them into a threechannel conditioning image. We have found that setting the pre-defined materials to (1) non-metal, not smooth; (2) half-metal, half-smooth; and (3) pure metal, extremely smooth, respectively, works well in practice.

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

Conditioning Images

LightControlNet Outputs Conditioning Image LightControlNet Output

(a) Independent Inputs to LightControlNet produce visual inconsistencies

(b) Concatenated Input to LightControlNet produces more consistent output

- Fig. 5: Multi-view visual prompting. (a) When we independently input four canonical conditioning images to LightControlNet, it generates four very different appearances and styles even with a fixed random seed. (b) When we concatenate the four images into a 2×2 grid and pass them as a single image into LightControlNet, it produces a far more consistent appearance and style. Text prompt: “A hiking boot”.

To train our LightControlNet, we use 40K objects from the Objaverse dataset [11]. Each object is rendered from 12 views using a randomly sampled camera C and lighting L sampled from 6 environment maps sourced from the Internet. L is also subject to random rotation and intensity scaling. For each resulting (L,C) pair, we render the conditioning image using the pre-defined materials, as well as the full-color rendering of the object using its original materials and textures. We use the resulting 480K pairs of (conditioning images, full-color rendering) to train LightControlNet using the approach of Zhang et al. [63].

Once LightControlNet is trained, we can specify the desired view and lighting for any 3D mesh. We first render the conditioning image with the desired view and lighting and then pass it along with a text prompt into LightControlNet, to obtain high-quality images. These images are spatially aligned to the desired view, lit with the desired lighting, and contain detailed textures (Figure 4).

Distilling the encoder. We improve the efficiency of SDS by distilling the image encoder in Stable Diffusion (SD) [44], the base diffusion model in the ControlNet architecture. The original SD encoder consumes almost 50% of the forward and backward time of SDS calculation, primarily in downsampling the input image. Metzer et al. [29] have found the image decoder can be closely approximated by per-pixel matrix multiplication. Inspired by this, we distill the encoder by removing its attention modules and training it on the COCO dataset [26] to match the original output. This distilled encoder runs 5x faster than the original one, resulting in an approximately 2x acceleration of our textto-texture pipeline without compromising output quality (Table 3).

##### 4.2 Stage 1: Multi-view Visual Prompting

- In Stage 1, we leverage LightControlNet to synthesize high-quality 2D images for a sparse set of views of the 3D mesh. Specifically, we create conditioning images for four canonical views C∗ around the equator of the 3D mesh using a

fixed lighting environment map L∗ sampled from a set of environment maps. One approach to generating the complete texture for the mesh would be to apply the LightControlNet independently with each such conditioning image, but using the same text prompt, and then backprojecting the four output images to the surface of the 3D mesh. In practice, however, applying the LightControlNet to each view independently produces inconsistent images of varying appearance and style, even when the text prompt and random seed remain fixed (Figure 5).

We use a multi-view visual prompting approach to mitigate this multi-view inconsistency issue. We concatenate the conditioning images for the four canonical views into a single 2 × 2 grid and treat it as a single conditioning image. We observe that applying LightControlNet to all four views simultaneously, using this combined multi-view conditioning image, results in a far more consistent appearance and style across the views, compared to independent prompting (Figure 5). We suspect this property arises from the presence of similar training data samples – grid-organized sets depicting the same object – in Stable Diffusion’s training set, which is also observed in concurrent works [59, 64]. Formally, we generate the conditioning image Icond(L∗,C∗) under a fixed canonical lighting condition L∗ using four canonical viewpoints C∗. We then apply our LightControlNet with text prompt y to generate the corresponding reference image Iref:

Iref = ControlNet(Icond(L∗,C∗),y).

##### 4.3 Stage 2: Texture Optimization

- In Stage 2, we could directly backproject the four reference views output in Stage 1 onto the 3D mesh using the camera matrix C associated with each view. While the resulting texture would contain some high-quality regions, it would also suffer from two problems: (1) It would contain seams and visual artifacts due to remaining inconsistencies between overlapping views, occlusions in the views that leave parts of the mesh untextured, and loss of detail when applying the backprojection transformation and resampling the views. (2) as lighting is baked into the LightControlNet’s RGB images, it would also be baked into the backprojected texture, making it difficult to relight the mesh.

To address both issues, we employ texture optimization using SDS loss. Specifically, we use a multi-resolution hash-grid [32] as our 3D texture representation. Given a 3D point p ∈ R3 on the mesh, our hash-grid produces a 32-dim multi-resolution feature, which is then fed to a 2-layer MLP Γ to obtain the texture material parameters for this point. Similar to Fantasia3D [9], these material parameters consist of metallicness km ∈ R, roughness kr ∈ R, a bump vector kn ∈ R3 and the base color kc ∈ R3. Formally,

(kc,km,kr,kn) = Γ(β(p)),

where β is the multi-resolution hash encoding function. Notably, this 3D hashgrid representation can be easily converted to 2D UV texture maps, which

are more friendly to downstream applications. Given the mesh M, the texture Γ(β(·)), a camera view C and lighting L we can use nvdiffrast [20], a differentiable renderer R to produce a 2D rendering of it, x, as

x = R(M,Γ(β(·)),L,C).

More details about the rendering equation are in the appendix. Since the mesh geometry is fixed, we omit M in the remainder of the paper.

Recall that the optimization approach of DreamFusion [39] randomly samples camera views C, generates an image for C using diffusion model ϕ, and supervises the optimization using the SDS loss. We extend this optimization in two ways. First, we use four fixed reference images Iref with their canonical views C∗ and lighting L∗ to guide the texture optimization through a reconstruction loss:

Lrecon = ||Iref − R(Γ(β(·)),L∗,C∗)||2 + Lperceptual(Iref,R(Γ(β(·)),L∗,C∗)),

where both L2 loss and perceptual loss [18] are used. For a non-canonical view C, we sample a random lighting L and use the SDS loss to supervise the optimization, but with our LightControlNet as the diffusion model ϕLCN, so

∂x ∂Γ(β(·))

∇Γ,βLSDS(ϕLCN,x) = Et,ϵ w(t)(ˆϵϕ

(xt;y,t,Icond(L,C)) − ϵ)

,

LCN

where x = R(Γ(β(·)),L,C) and w(t) is the weight.

Finally, we employ a material smoothness regularizer on every iteration to enforce smooth base colors, using the approach of nvdiffrec [33]. For a surface point p with base color kc(p), the smoothness regularizer is defined as

Lreg =

|kc(p) − kc(p + ϵ)|,

p∈S

where S denotes the object surface and ϵ is a small random 3D perturbation. We use λrecon = 1000 and λreg = 10 to reweight the loss Lrecon and Lreg.

Scheduling the optimization. We warm up the optimization by rendering the four canonical views and applying Lrecon for 50 iterations. We then add in iterations using LSDS and optimize over randomly chosen camera views and randomly selected lighting from a pre-defined set of environmental lighting maps. Specifically we alternate iterations between using LSDS and Lrecon. In addition, for a quarter of the SDS iterations, we use the canonical views rather than randomly selecting the views. This ensures that the resulting texture does not overfit to the reference images corresponding to the canonical views. The warm-up iterations capture the large-scale structure of our texture and allow us to use relatively small noise levels (t ≤ 0.1) in the SDS optimization. We sample the noise following a linearly decreasing schedule [17] with tmax = 0.1 and tmin = 0.02. We also adjust the conditioning strength s of our LightControlNet in LSDS linearly from

- 1 to 0 over these iterations so that LightControlNet is only lightly applied by the

end of the optimization. We also experimented with a recent variant Variational Score Distillation [57], but did not observe notable improvement. We have experimentally found that we obtain high-quality textures after 400 total iterations of this optimization and this is far fewer iterations than other SDS-based texture generation techniques such as Fantasia3D [9] which requires 5000 iterations.

Faster pipeline without relightability. Our two-stage pipeline is also compatible with off-the-shelf depth ControlNet and Stable Diffusion [44] as the backbone replacement of LightControlNet. Specifically, we can replace the LightControlNet in Stage 1 with a depth ControlNet that uses a depth rendering of the mesh as the conditioning image, and uses Stable Diffusion based SDS in Stage

- 2. In scenarios where texture relightability is not required, this variant offers an additional 2× speed-up (as shown in Table 1), since it eliminates the additional computation required by LightControlNet forward pass in the SDS optimization.

### 5 Experiments

In this section, we present comprehensive experiments to evaluate the efficacy of our proposed method for relightable, text-based mesh texturing. We perform both qualitative and quantitative comparisons with existing baselines, along with an ablation study on the significance of each of our major components.

Dataset. As illustrated in Figure 3, we employ Objaverse [11] to render paired data to train our LightControlNet. Objaverse consists of approximately 800k objects, of which we use the names and tags as their text descriptions. We filter out objects with low CLIP similarity [40] to their text descriptions and select around 40k as our training set. To evaluate baselines and our method, we hold out 70 random meshes from Objaverse [11] as the test set. We additionally gather 22 mesh assets from 3D online games with 5 prompts each to assess our method. Baselines. We compare our approach with existing mesh texturing methods. Specifically, Latent-Paint [29] employs SDS loss in latent space for texture generation. Text2tex [8] progressively produces 2D views from chosen viewpoints, followed by an inverse projection to lift them to 3D. TEXTure [43] utilizes a similar lifting approach but supplements it with a swift SDS optimization postlifting. Beyond these texture generation methods, text-to-3D approaches serve as additional baselines, given that texture is a component of 3D generation. Notably, we choose Fantasia3D [9] as a baseline, the first to use a material-based representation for textures in text-to-3D processing.

Quantative evaluation. In Table 1, we compare our method with the baselines on the Objaverse [11] test set. For each method, we generate 16 views and evaluate Frechet Inception Distance (FID) [16,37] and Kernel Inception Distance (KID) [3] compared with ground-truth renderings. Two variations of our method are assessed. Both variants use ourr two-stage pipeline, and the first employs a standard depth ControlNet, while the second uses our proposed LightControlNet. Our method outperforms the baselines in both quality and runtime.

Qualitative analysis. In Figure 6, our method can generate highly detailed textures that can be rendered properly with the environment lighting across various

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

“Stylish Boot”

“Pinecone”

[Figure 65]

[Figure 66]

[Figure 67]

“1978 Puch Moped, motorcycle”

“Wooden Boat”

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

“A vintage space explorer jacket with a matching helmet, …”

“Hylian goblin soldier “Futuristic Helmet, …” from legend of zelda …”

“Jacket made from the fabrics of a ghost ship, …”

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

“A stylish jacket, …” “Jacket that gives the “Mermaid warrior, …” impression of a swirling nebula, …”

“An astronaut wolf, …”

- Fig. 6: Sample results from our method applied to Objaverse test meshes (top half) and 3D game assets (bottom half). To illustrate the efficacy of our relightable textures, for each textured mesh, we fix the environment lighting and render the mesh under different rotations. As shown above, our method is able to generate textures that are not only highly detailed, but also relightable with realistic lighting effects.

meshes. We also visually compare our method and the baselines in Figure 7. Our method produces textures with higher visual fidelity for both the relightable and non-relightable variants. In particular, when compared with Fantasia3D [9], a recent work that also aims to generate material-based texture, our results not only have superior visual quality, but also disentangle the lighting more successfully.

User study. To further evaluate the texture quality quantitatively, we conduct a user study comparing our results with each of the baselines on the Objaverse test set in Table 2. We asked 30 participants to evaluate (1) the realism of the results, (2) the consistency of the generated texture with the input text, and (3) the plausibility of the results when placed under varying lighting conditions. Each result is presented in the form of 360-degree rotation to display full texture details. The reference lighting is provided alongside when participants evaluate (3). Across all three aspects, participants consistently prefer our method.

Lighting 1 Lighting 2

[Figure 84]

[Figure 85]

OursFantasia3D

[Figure 86]

[Figure 87]

(a) Close-up Comparison with Fantasia3D. Left Prompt: “A medieval steel helmet” ; Right Prompt: “A leather horse saddle”.

###### PBR Texture RGB Texture

Untextured Mesh (reference lighting)

Ours (Non-relightable)

Ours

Fantasia3D TEXTure Latent-Paint Text2tex

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

View1View2

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

View1View2

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

(b) Comparison with relightable and non-relightable baselines. Top Prompt: “A hiking boot”; Bottom Prompt: “A leather horse saddle”.

- Fig. 7: Qualitative analysis. (a) We compare with Fantasia3D [9] that also attempts to generate Physically Based Rendering (PBR) texture. However, their results often exhibit baked-in lighting, leading to artifacts when put under varied lighting. (b) We also compare with other baselines that only generate non-relightable (RGB) texture. For non-relightable texture generation, we can replace our LightControlNet with depth ControlNet and generate textures with a shorter runtime. More details are in Table 1.

Ablation study. We perform an ablation analysis on different aspects of our method in Table 3. When replacing our distilled encoder with the original SD encoder, performance is twice as slow without noticeably superior quality. On the other hand, without the multi-view visual prompting for the initial generation, the system requires 2000 iterations (a 5x slowdown compared to our 400 itera-

- Table 1: Quantitative Evaluation. We test our methods and baselines on 70 test objects from Objaverse [11] and 22 objects curated from 3D game assets. With depth ControlNet, our method yields superior results to all baselines while being three times as fast as the fastest baseline. Using LightControlNet (Ours) within our model improves the lighting disentanglement while maintaining comparable image quality.

Objaverse test set Game Asset

Runtime ↓ FID ↓ KID ↓ FID ↓ KID ↓

(×10−3) (×10−3) (mins)

Latent-Paint [29] 73.65 7.26 204.43 9.25 10 Fantasia3D [9] 120.32 8.34 164.32 9.34 30 TEXTure [43] 71.64 5.43 103.49 5.64 6 Text2tex [8] 95.59 4.71 119.98 5.21 15

Ours (w/ depth) 60.49 3.96 85.92 3.87 2 Ours 62.67 2.69 83.32 3.34 4

- Table 2: User study. We conduct a user preference study to evaluate (1) result realism, (2) texture consistency with input text, and (3) plausibility under varied lighting. Participants consistently prefer our results over all baselines in these respects.

Preferred Percentage

Objaverse test set Realistic Consistent with text Relightable

Ours v.s. Latent-Paint [29] 92.6% 74.5% 84.3% Ours v.s. Fantasia3D [9] 81.9% 67.6% 74.3% Ours v.s. TEXTure [43] 70.8% 57.3% 87.1% Ours v.s. Text2tex [8] 75.4% 61.6% 88.6%

- Table 3: Ablation study on algorithmic components. We analyze the role of our distilled encoder (1st row) and multi-view visual prompting (2nd row). Replacing the distilled encoder with the original encoder doubles the running time without a noticeable improvement. When removing the multi-view visual prompting for initial generation, the system requires 2,000 iterations (5x compared to our 400 iterations) to produce reasonable results, which produces slightly worse texture quality.

Objaverse test set FID ↓ KID (×10−3) ↓ Runtime ↓ (mins) Ours (w/o dist. enc.) 60.34 2.84 8 Ours (w/o m.v.v.p) 74.23 3.54 19 Ours 62.67 2.69 4

tions) to produce reasonable results while still leading to slightly worse texture quality. In Section 4.1, we render a conditioning image using three pre-defined materials to encompass a broad range of feasible effects. Table 4 shows omitting any one of these bases degrades quality. Table 5 evaluates our selection of four

- Table 4: Ablation study on material bases. We verify the impact of the material bases in rendering conditioning images. Omitting any one of these degrades quality.

Material Basis

non-metal, half-metal, pure metal, FID ↓ KID (×10−3) ↓ not smooth half-smooth smooth

✓ ✓ ✓ 62.67 2.69

✓ ✓ 66.34 3.11 ✓ ✓ 64.32 3.42 ✓ ✓ 67.43 4.12

✓ 72.13 4.53

- Table 5: Ablation study on canonical view selection in Section 4.2. Using only front and back views provides insufficient supervision while adding top and bottom views worsens quality. This likely stems from pre-trained 2D diffusion models struggling with top and bottom views. Additionally, stacking more views reduces each view’s resolution, leading to poorer initialization for Stage 2.

Num. of canonical views FID ↓ KID (×10−3) ↓

2 views (front, back) 67.43 3.47 4 views (Ours: front, back, left, right) 62.67 2.69 6 views (front, back, left, right, top, bottom) 70.14 3.72

canonical views in Section 4.2. Relying on only the front and back views provides insufficient supervision. Interestingly, incorporating top and bottom views degrades the performance. We hypothesize that this is likely due to the limitation of 2D diffusion model backbones in reliably generating top and bottom views. Furthermore, stacking more views within a single image results in a decreased resolution for each view, given the fixed resolution of the multi-view image.

- 6 Discussion

We proposed an automated texturing technique based on user-provided prompts. Our method employs an illumination-aware 2D diffusion model (LightControlNet) and an improved optimization process based on the SDS loss. Our approach is substantially faster than previous methods while yielding high-fidelity textures with illumination disentangled from surface reflectance/albedo. We demonstrated the efficacy of our method through quantitative and qualitative evaluation on the Objaverse dataset and meshes curated from game assets.

Limitations. Our approach still poses a few limitations: (1) Baked-in lighting can still be found in certain cases, especially for meshes that are outside of the training data distribution; (2) The generated material maps are sometimes not fully disentangled and interpretable as metallicness, roughness, etc.

### Acknowledgements

We thank Benjamin Akrish, Victor Zordan, Dmitry Trifonov, Derek Liu, ShengYu Wang, Gaurav Parmer, Ruihan Gao, Nupur Kumari, and Sean Liu for their discussion and help. This work was done when KD was an intern at Roblox. The project is partly supported by Roblox. JYZ is partly supported by the Packard Fellowship. The Microsoft Research PhD Fellowship supports KD.

### References

- 1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: IEEE International Conference on Computer Vision (ICCV) (2021)
- 2. Bi, S., Xu, Z., Srinivasan, P., Mildenhall, B., Sunkavalli, K., Hasan, M., HoldGeoffroy, Y., Kriegman, D., Ramamoorthi, R.: Neural reflectance fields for appearance acquisition. arXiv preprint arXiv:2008.03824 (2020)
- 3. Bińkowski, M., Sutherland, D.J., Arbel, M., Gretton, A.: Demystifying mmd gans. In: International Conference on Learning Representations (ICLR) (2018)
- 4. Bokhovkin, A., Tulsiani, S., Dai, A.: Mesh2tex: Generating mesh textures from image queries. In: IEEE International Conference on Computer Vision (ICCV)

(2023)

- 5. Canny, J.: A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence pp. 679–698 (1986)
- 6. Cao, T., Kreis, K., Fidler, S., Sharp, N., Yin, K.: Texfusion: Synthesizing 3d textures with text-guided image diffusion models. In: IEEE International Conference on Computer Vision (ICCV) (2023)
- 7. Cao, Z., Simon, T., Wei, S.E., Sheikh, Y.: Realtime multi-person 2d pose estimation using part affinity fields. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7291–7299 (2017)
- 8. Chen, D.Z., Siddiqui, Y., Lee, H.Y., Tulyakov, S., Nießner, M.: Text2tex: Textdriven texture synthesis via diffusion models. In: IEEE International Conference on Computer Vision (ICCV) (2023)
- 9. Chen, R., Chen, Y., Jiao, N., Jia, K.: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In: IEEE International Conference on Computer Vision (ICCV) (2023)
- 10. Chen, Y., Chen, R., Lei, J., Zhang, Y., Jia, K.: Tango: Text-driven photorealistic and robust 3d stylization via lighting decomposition. In: Advances in Neural Information Processing Systems (NeurIPS) (2022)
- 11. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 12. Gao, D., Li, X., Dong, Y., Peers, P., Xu, K., Tong, X.: Deep inverse rendering for high-resolution svbrdf estimation from an arbitrary number of images. In: ACM SIGGRAPH (2019)
- 13. Haque, A., Tancik, M., Efros, A., Holynski, A., Kanazawa, A.: Instruct-nerf2nerf: Editing 3d scenes with instructions. In: IEEE International Conference on Computer Vision (ICCV) (2023)

- 14. Henderson, P., Tsiminaki, V., Lampert, C.: Leveraging 2D data to learn textured 3D mesh generation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2020)
- 15. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022)
- 16. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. In: Advances in Neural Information Processing Systems (NeurIPS) (2017)
- 17. Huang, Y., Wang, J., Shi, Y., Qi, X., Zha, Z.J., Zhang, L.: Dreamtime: An improved optimization strategy for text-to-3d content creation. arXiv preprint arXiv:2306.12422 (2023)
- 18. Johnson, J., Alahi, A., Fei-Fei, L.: Perceptual losses for real-time style transfer and super-resolution. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. pp. 694–711. Springer (2016)
- 19. Kobayashi, S., Matsumoto, E., Sitzmann, V.: Decomposing nerf for editing via feature field distillation. Advances in Neural Information Processing Systems 35, 23311–23330 (2022)
- 20. Laine, S., Hellsten, J., Karras, T., Seol, Y., Lehtinen, J., Aila, T.: Modular primitives for high-performance differentiable rendering. In: ACM SIGGRAPH (2020)
- 21. Li, M., Duan, Y., Zhou, J., Lu, J.: Diffusion-sdf: Text-to-shape via voxelized diffusion. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

(2023)

- 22. Li, W., Chen, R., Chen, X., Tan, P.: Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arxiv:2310.02596 (2023)
- 23. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Openset grounded text-to-image generation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 24. Li, Z., Sunkavalli, K., Chandraker, M.: Materials for masses: Svbrdf acquisition with a single mobile phone image. In: European Conference on Computer Vision (ECCV) (2018)
- 25. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 26. Lin, T.Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C.L., Dollár, P.: Microsoft coco: Common objects in context

(2015)

- 27. Liu, R., Wu, R., Hoorick, B.V., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero1-to-3: Zero-shot one image to 3d object. In: IEEE International Conference on Computer Vision (ICCV) (2023)
- 28. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: SDEdit: Guided image synthesis and editing with stochastic differential equations. In: International Conference on Learning Representations (ICLR) (2022)
- 29. Metzer, G., Richardson, E., Patashnik, O., Giryes, R., Cohen-Or, D.: Latent-nerf for shape-guided generation of 3d shapes and textures. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 30. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European Conference on Computer Vision (ECCV) (2020)

- 31. Mou, C., Wang, X., Xie, L., Zhang, J., Qi, Z., Shan, Y., Qie, X.: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453 (2023)
- 32. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. In: ACM SIGGRAPH (2022)
- 33. Munkberg, J., Hasselgren, J., Shen, T., Gao, J., Chen, W., Evans, A., Müller, T., Fidler, S.: Extracting Triangular 3D Models, Materials, and Lighting From Images. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- 34. Nam, G., Khlifi, M., Rodriguez, A., Tono, A., Zhou, L., Guerrero, P.: 3d-ldm: Neural implicit 3d shape generation with latent diffusion models. arXiv preprint arXiv:2212.00842 (2022)
- 35. Nicodemus, F.E.: Directional reflectance and emissivity of an opaque surface. Applied optics 4(7), 767–775 (1965)
- 36. Park, K., Rematas, K., Farhadi, A., Seitz, S.M.: Photoshape: Photorealistic materials for large-scale shape collections. In: ACM SIGGRAPH Asia (2018)
- 37. Parmar, G., Zhang, R., Zhu, J.Y.: On aliased resizing and surprising subtleties in gan evaluation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- 38. Pavllo, D., Kohler, J., Hofmann, T., Lucchi, A.: Learning generative models of textured 3d meshes from real-world images. In: IEEE International Conference on Computer Vision (ICCV) (2021)
- 39. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. In: International Conference on Learning Representations (ICLR) (2023)
- 40. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML) (2021)
- 41. Raj, A., Kaza, S., Poole, B., Niemeyer, M., Ruiz, N., Mildenhall, B., Zada, S., Aberman, K., Rubinstein, M., Barron, J., et al.: Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508 (2023)
- 42. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125

(2022)

- 43. Richardson, E., Metzer, G., Alaluf, Y., Giryes, R., Cohen-Or, D.: Texture: Textguided texturing of 3d shapes. In: ACM SIGGRAPH (2023)
- 44. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- 45. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic text-toimage diffusion models with deep language understanding. In: Advances in Neural Information Processing Systems (NeurIPS) (2022)
- 46. Sartor, S., Peers, P.: Matfusion: a generative diffusion model for svbrdf capture. In: ACM SIGGRAPH Asia (2023)
- 47. Shi, Y., Wang, P., Ye, J., Mai, L., Li, K., Yang, X.: Mvdream: Multi-view diffusion for 3d generation. arXiv:2308.16512 (2023)
- 48. Shue, J.R., Chan, E.R., Po, R., Ankner, Z., Wu, J., Wetzstein, G.: 3d neural field generation using triplane diffusion. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)

- 49. Siddiqui, Y., Thies, J., Ma, F., Shan, Q., Nießner, M., Dai, A.: Texturify: Generating textures on 3d shape surfaces. In: European Conference on Computer Vision (ECCV) (2022)
- 50. Sun, J., Zhang, B., Shao, R., Wang, L., Liu, W., Xie, Z., Liu, Y.: Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior (2023)
- 51. Vecchio, G., Martin, R., Roullier, A., Kaiser, A., Rouffet, R., Deschaintre, V., Boubekeur, T.: Controlmat: Controlled generative approach to material capture. arXiv preprint arXiv:2309.01700 (2023)
- 52. Vecchio, G., Sortino, R., Palazzo, S., Spampinato, C.: Matfuse: Controllable material generation with diffusion models. arXiv preprint arXiv:2308.11408 (2023)
- 53. Walter, B., Marschner, S.R., Li, H., Torrance, K.E.: Microfacet models for refraction through rough surfaces. In: Proceedings of the 18th Eurographics conference on Rendering Techniques (2007)
- 54. Wang, C., Chai, M., He, M., Chen, D., Liao, J.: Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3835–3844 (2022)
- 55. Wang, H., Du, X., Li, J., Yeh, R.A., Shakhnarovich, G.: Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 56. Wang, T., Zhang, T., Zhang, B., Ouyang, H., Chen, D., Chen, Q., Wen, F.: Pretraining is all you need for image-to-image translation. arXiv preprint arXiv:2205.12952 (2022)
- 57. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In: Advances in Neural Information Processing Systems (NeurIPS) (2023)
- 58. Wang, Z., Philion, J., Fidler, S., Kautz, J.: Learning indoor inverse rendering with 3d spatially-varying lighting. In: IEEE International Conference on Computer Vision (ICCV) (2021)
- 59. Weber, E., Holynski, A., Jampani, V., Saxena, S., Snavely, N., Kar, A., Kanazawa, A.: Nerfiller: Completing scenes via generative 3d inpainting. In: arXiv (2023)
- 60. Xu, X., Lyu, Z., Pan, X., Dai, B.: Matlaber: Material-aware text-to-3d via latent brdf auto-encoder. arXiv preprint arXiv:2308.09278 (2023)
- 61. Yu, R., Dong, Y., Peers, P., Tong, X.: Learning texture generators for 3d shape collections from internet photo sets. In: British Machine Vision Conference (BMVC)

(2021)

- 62. Zeng, X., Chen, X., Qi, Z., Liu, W., Zhao, Z., Wang, Z., FU, B., Liu, Y., Yu, G.: Paint3d: Paint anything 3d with lighting-less texture diffusion models (2023)
- 63. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: IEEE International Conference on Computer Vision (ICCV)

(2023)

- 64. Zhao, M., Zhao, C., Liang, X., Li, L., Zhao, Z., Hu, Z., Fan, C., Yu, X.: Efficientdreamer: High-fidelity and robust 3d creation via orthogonal-view diffusion prior. In: arXiv (2023)
- 65. Zhou, L., Du, Y., Wu, J.: 3d shape generation and completion through point-voxel diffusion. In: IEEE International Conference on Computer Vision (ICCV) (2021)
- 66. Zhuang, J., Wang, C., Lin, L., Liu, L., Li, G.: Dreameditor: Text-driven 3d scene editing with neural fields. In: SIGGRAPH Asia 2023 Conference Papers. pp. 1–10

(2023)

### A Implementation Details and Additional Results

Distilled Encoder. In Section 4.1 of the main paper, we improve the efficiency of LightControlNet by distilling the image encoder in Stable Diffusion [44]. We profile the running time of our distilled and original encoder in the following table.

Time (ms) Forward Backward Total

Original 113 569 682 Distilled 42 81 123 (5.5×)

Table 6: We profile the forward and backward pass of our distilled and original encoder in Stable Diffusion [44] on an A100 GPU. Our distilled encoder runs more than 5× faster than the original one for a single forward and backward pass.

Hyper-parameters. We provide the hyper-parameters used by our pipeline in Section 4.3 of the main paper. We use a batch of 4, a learning rate of 0.01 for optimization, and a CFG scale of 50 in Score Distillation Sampling loss. In Section 4.1 and Section 4.2 (main paper), we set 3 pre-defined materials to generate a conditioning image. The specific material parameters are (1) nonmetal, non-smooth: km = 0,kr = 1; (2) half metal, half smooth: km = 0.5,kr = 0.5; (3) pure metal, extremely smooth: km = 1,kr = 0. The color kc is always set to (1,1,1). We train our LightControlNet for 20,000 iterations with a batch size of 16.

Base Models. We use stable diffusion v1.5 (SD1.5) as our base model for the experiments in Tables 1, 2, and 3 of the main paper. Our pipeline is also compatible with other base models fine-tuned from SD1.5. For example, we have also used Dreamshaper, a community fine-tuned checkpoint of SD1.5, to generate a variety of captivating textures. We include some of the results in Figure 6 (main paper) and Figure 10. Specifically, the results of the jackets, goblins, fishmen, and wolves are obtained using Dreamshaper.

Environmental Light Maps. In Section 4, we use randomly rotated environmental light maps to represent different lighting conditions. Specifically, we download 6 HDRI light maps from polyhaven. These HDRI maps are captured in a studio environment. We show these light maps in Figure 8. We also study the effect of different lighting by using various environment maps in training LightControlNet. We contrast the indoor set used in the main paper (FID: 62.67, KID: 2.69) with an outdoor set of 6 maps featuring brighter ambient light (FID: 63.23,

- KID: 2.66), and another set of 6 maps with less directional lighting (FID: 70.14,
- KID: 3.72). The results suggest directional lighting is important in accurately modeling lighting effects with LightControlNet, whereas ambient light intensity minimally impacts quality. BRDF Model and Rendering Equation. As described in Section 4.3 of the main paper, our material model [53] consists of metallicness km ∈ R, roughness

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

- Fig. 8: Environmental Light Maps. We download 6 HDRI maps from polyhaven to represent different different lighting conditions. For random lighting samples, we select one map from them and apply a random rotation.

kr ∈ R, a bump vector kn ∈ R3 which is a perturbation of surface normal in tangent space, and the base color kc ∈ R3. We show an example of our generated material maps in Figure 12. The general rendering equation is:

L(p,ω) =

Li (p,ωi)f (p,ωi,ω)(ωi · np)dωi,

Ω

where L is the rendered pixel color at the surface point p from the direction ω, Ω denotes a hemisphere with the surface normal np at p. Li is the incident light represented by an environment map, and f is the BRDF function determined by the material parameters (km,kr,kn,kc). L can be calculated as the summation of diffuse intensity Ld and specular intensity Ls as follows:

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Artifacts

Baked-in lighting

[Figure 127]

Missing Bottom

Grid Representation Backprojection Baseline Our method

- Fig. 9: Backprojection Baseline. Directly backprojecting the grid representation onto 3D mesh leads to baked-in lighting, stitching artifacts, and missing areas of texture.

L(p,ω) = Ld(p) + Ls(p,ω), Ld(p) = kc(1 − km)

Li (p,ωi)(ωi · np)dωi, Ls(p,ω) =

Ω

D(np)F(ωi,ω,np)G(ωi,ω,np) 4(ω · np)(ωi · np)

Li (p,ωi)(ωi · np)dωi,

Ω

where F, G, and D are functions representing the Fresnel term, the geometric attenuation, and the GGX normal distribution [53], respectively. Following Nvdiffrec [33] and Fantasia3D [9], the hemisphere integration can be calculated using the split-sum method.

Rendering Engine. We use nvdiffrast [20], a differentiable renderer, as our rendering engine in both of our training and testing phases. Notably, our exported material maps are also directly compatible with widely used rendering applications, such as Blender. This compatibility is demonstrated in the main paper, where Figure 1 and the top half of Figure 7 are rendered using Blender, showcasing the practical application of our generated textures.

Backprojection Baseline. We mention an alternative baseline by directly backprojecting the sparse views in Section 4 of the main paper. We compare this baseline with our method in Figure 9.

Additional Results. We show additional results in Figures 10 and 11. We also show an example of our generated material maps in Figure 12.

### B Additional Related Works

In Section 2 of the main paper, we discuss prior works on 3D texture generation. Another excellent recent work, TexFusion [6], aligns with the research

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

“A sculpture of horse without rider”

“Peony Flower”

[Figure 137]

[Figure 138]

[Figure 139]

“Ballet costume”

“LAV-25, tank”

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

“A dog head statue”

“Doc Martens Boot”

[Figure 149]

[Figure 150]

[Figure 151]

“Camra 16 mm, Paillard Bolex, H16 REX-5” “Suede Women’s Heeled Boot”

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

“Medieval Windmill”

“Vintage cash register”

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

“Hylian wolf soldier from Legend of Zelda”

“Moon necklace” “Casual gothic outfit”

“Pirate tribal wolf”

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

“Deep sea diver” “Cave Dweller”

“A sculpture of monkey”

“Thorsberg tunic”

###### Fig. 10: Additional Results. We rotate the objects to show different views while fixing the lighting condition.

###### Relightable Texture Non-relightable Texture

Untextured Mesh (reference lighting)

Fantasia3D TEXTure Latent-Paint Text2tex

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

- View1

Text Prompt Ours Ours (Non-relightable)

- View2

“vintage cash register”

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

View1View2

“wooden boat”

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

View1View2

“Medieval windmill”

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

- Fig. 11: Additional Qualitative Analysis. We present additional comparisons with the baselines. To evaluate the quality of relightable textures, our results and those from Fantasia3D are placed under identical lighting conditions. An untextured mesh is also rendered under this condition to denote the reference lighting direction. Our results adhere to the lighting conditions and generally outperform all existing baselines in quality.

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

“Stone Goblet carved with runes and symbols”

Base Color Metallic Map Roughness Map Bump Map

Fig. 12: We show an example of our generated material maps.

using depth-conditioned diffusion models for texture creation. They introduce a Sequential Interlaced Multiview Sampler, which performs denoising steps across multiple camera viewpoints and integrates the outcomes into a unified latent texture map. The execution time for TexFusion is comparable to that of TEXTure [43], with the comprehensive approach taking approximately 6 minutes, whereas a quicker variant takes about 2.2 minutes but yields lower-quality re-

sults. Similar to both TEXTure [43] and Text2tex [8], TexFusion’s textures are entangled with lighting effects. The non-relightable textures make it challenging to adapt to various lighting conditions. On the other hand, our generated texture can be relit properly in different lighting environments with a comparable runtime (4 mins). As their code is unavailable, we cannot preform a direct comparison. However, we will be happy to compare our method with this baseline once their code becomes available.

### C Societal Impact

Our method in text-based mesh texturing will enable many applications in 3D content creation. First, our method drastically reduces the time and expertise required for texture authoring, making 3D content generation accessible to a broader audience. Additionally, our faster inference time and improved result quality also reduce computational costs and energy consumption. However, our method might also be misused to generate fake content for misinformation. Nevertheless, we believe humans can currently distinguish our synthesized objects from photo captures of real objects.

