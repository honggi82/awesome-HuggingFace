## HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions

Haiyang Zhou, Xinhua Cheng, Wangbo Yu, Yonghong Tian, and Li Yuan

### arXiv:2407.15187v1[cs.CV]21Jul2024

Abstract—3D scene generation is in high demand across various domains, including virtual reality, gaming, and the film industry. Owing to the powerful generative capabilities of text-to-image diffusion models that provide reliable priors, the creation of 3D scenes using only text prompts has become viable, thereby significantly advancing researches in text-driven 3D scene generation. In order to obtain multiple-view supervision from

- 2D diffusion models, prevailing methods typically employ the diffusion model to generate an initial local image, followed by iteratively outpainting the local image using diffusion models to gradually generate scenes. Nevertheless, these outpaintingbased approaches prone to produce global inconsistent scene generation results without high degree of completeness, restricting their broader applications. To tackle these problems, we introduce HoloDreamer, a framework that first generates highdefinition panorama as a holistic initialization of the full 3D scene, then leverage 3D Gaussian Splatting (3D-GS) to quickly reconstruct the 3D scene, thereby facilitating the creation of viewconsistent and fully enclosed 3D scenes. Specifically, we propose Stylized Equirectangular Panorama Generation, a pipeline that combines multiple diffusion models to enable stylized and detailed equirectangular panorama generation from complex text prompts. Subsequently, Enhanced Two-Stage Panorama Reconstruction is introduced, conducting a two-stage optimization of
- 3D-GS to inpaint the missing region and enhance the integrity of the scene. Comprehensive experiments demonstrated that our method outperforms prior works in terms of overall visual consistency and harmony as well as reconstruction quality and rendering robustness when generating fully enclosed scenes.

Index Terms—text-to-3D, 3D Gaussian Splatting, scene generation, panorama generation, panorama reconstruction.

I. INTRODUCTION

# A

S the field of 2D generation [1] and 3D representation evolves [2], [3], 3D content generation has become a

significant task within the realm of computer vision. Text prompts can intuitively and comprehensively describe user’s needs, as a consequence, the zero-shot text-driven generation of 3D scenes will lower the barrier to entry for newcomers, and save considerable manual effort in 3D modeling. This makes it a promising application in industries such as metaverse, virtual

Corresponding author: Yonghong Tian; Li Yuan. Haiyang Zhou is interning at Peking University, School of Electronic and

Computer Engineering, Shenzhen Graduate School, Shenzhen, Guangdong Province 518055, China (e-mail: zhouhaiyang000@gmail.com).

Xinhua Cheng is with the Peking University, School of Electronic and Computer Engineering, Shenzhen Graduate School, Shenzhen, Guangdong Province 518055, China (e-mail: chengxinhua@stu.pku.edu.cn; yuwangbo98@gmail.com).

Wangbo Yu, Yonghong Tian, and Li Yuan are with the Peking University, School of Electronic and Computer Engineering, Shenzhen Graduate School, Shenzhen, Guangdong Province 518055, China, and also with Peng Cheng Laboratory, Shenzhen, Guangdong Province 518066, China (e-mail: yhtian@pku.edu.cn; yuanli@u.nus.edu).

reality and film production. However, unlike the abundance of paired text-to-image data in the field of 2D, paired textto-3D data is significantly scarce currently. The creation of 3D datasets often requires substantial human and material resources, which results in challenges to directly train for 3D content generation from user prompts. Despite numerous efforts [4]–[6] to leverage diffusion models to conduct endto-end and feedforward generation of 3D content, the results are still constrained by the size and quality of training data, leading to poor performance in details.

To overcome this limitation, some prior works [7]–[12] harness the high-level priors from pre-trained text-to-image models, i.e., CLIP [13] and image diffusion models [1], to guide the optimization of 3D representations, achieving zero-shot effects. However, these methods are limited to generating simple geometric shapes, with the cameras all converging on the object’s position. For the generation of more complex scenes with camera orientations diverge outward, several past efforts, including SceneScape [14], Text2Room [15], Text2NeRF [16] and LucidDreamer [17], progressively outpaint an initial image using diffusion models, followed by the integration of monocular depth estimation networks to obtain depth information for subsequent 3D reconstruction. However, the large field of view necessitates a substantially increased number of outpainting iterations. Each iteration of the outpainting process is solely based on the local existing scene, leading to difficulties in maintaining global consistency and harmony during the prolonged outpainting process. The generated scene is visually chaotic, particularly when the scene is fully enclosed. In addition to this, the scene exhibits low rendering robustness for outside preset views.

In this work, we introduce a framework named HoloDreamer, a novel method for text-driven generation of viewconsistent and fully enclosed 3D scenes with strong rendering robustness. Unlike previous approachs, which are prone to consistency issues arising from progressive outpainting, we propose Stylized Equirectangular Panorama Generation, utilizing text-to-image diffusion models to directly generate a highly consistent 360-degree equirectangular panorama from text prompts. The generated panorama boasts excellent visual quality, characterized by high-resolution details that contribute to a coherent and immersive viewing experience. Specifically, to preserve the accuracy of the equirectangular projection, we first generate a base panorama using a diffusion model finetuned on the panorama database, and subsequently perform style transfer and detail enhancement using conditional controlled diffusion models, ensuring that the panorama are not only detailed but also aesthetically pleasing and true to the

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Lego city with lego shops, lego road with street lamp, cars and lego mans on the street, lego trees and lake at a park.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Stylized Equirectangular Panorama Generation

[Figure 12]

[Figure 13]

Enhanced Two-Stage Panorama Reconstruction

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

HoloDreamer

- Fig. 1. We propose HoloDreamer, a text-driven 3D scene generation framework to generate immersive and fully enclosed 3D scenes with high view-consistency. It consists of two basic modules: Stylized Equirectangular Panorama Generation, which generates a stylized high-quality equirectangular panorama from the input user prompt, and Enhanced Two-Stage Panorama Reconstruction, which employs 3D Gaussian Splatting for rapid 3D reconstruction of the panorama with enhanced integrity.

visual style inferred from the text description.

consistent, immersive and fully enclosed 3D scenes based on text descriptions, as shown in Fig. 1. Furthermore, the pipeline exhibits a high degree of generality, encompassing a diverse spectrum of styles ranging from interior to exterior environments, as shown in Fig. 10. Comprehensive experiments strongly demonstrate that our approach surpasses other textdriven 3D scene generation methods in terms of overall visual consistency and harmony, reconstruction quality, and rendering robustness when it comes to generating full-enclosed scenes.

We choose 3D Gaussian Splatting (3D-GS) [3] as the 3D representation for the scene due to its robust capability to represent highly granular details across various scenes and its significant optimization speed, which enables rapid, highfidelity 3D reconstruction with a panoramic field of view. We propose Enhanced Two-Stage Panorama Reconstruction, a pipeline that reconstructs enhanced 3D scene from generated panoram using 3D-GS. Initially, leveraging the depth prior provided by a monocular depth estimation model, we perform depth estimation on the panorama. The resulting RGBD data is then transformed into point clouds, which serve as the initialization for the 3D Gaussians. Furthermore, to enhance the robustness of the scene rendering, a two-stage 3D-GS optimization process is designed for the reconstruction of 3D scenes. In the Pre Optimization stage, we project multiple additional perspective images from the point cloud to apply multi-view constraints on the 3D Gaussians during the optimization process. This strategy overcomes the limitation of having a single viewpoint in panorama and prevents the generation of artifacts. After Pre Optimization results, we employ a

In summary, our contributions can be outlined as follows:

- • We propose HoloDreamer, a text-driven 3D scene generation approach that combines diffusion models and 3D Gaussian Splatting to generate fully enclosed immersive 3D scenes with visual consistency.
- • We introduce Stylized Equirectangular Panorama Generation, a framework for panorama generation by leveraging the power of diffusion models, which can maintain the geometric features of equirectangular projection while expanding the range of generative capabilities and diversity.
- • Our proposed Enhanced Two-Stage Panorama Reconstruction module provides multi-view constraints for the single viewpoint of the panorama and introduces inpainting to 3D-GS optimization, reducing artifacts and improving the integrity of the scene, achieving fast and high fidelity 3D reconstruction from single panorama.

- 2D inpainting model to fill in missing areas within the images rendered from the scene. Ultimately, the inpainted images are incorporated as supervision for Transfer Optimization stage to achieve high-level integrity of the final reconstructed scene.

Our proposed HoloDreamer can generate highly view-

II. RELATED WORKS

- A. 3D Representation

The field of 3D representation has seen a multitude of approaches, each with its own set of trade-offs and applications. Traditional primitives such as point clouds, meshes and voxels have been the cornerstone of 3D modeling for years. However, these methods face limitations in terms of representational ability: they often require a large quantity of data to achieve high resolution, which can be cumbersome and computationally expensive. With the advent of deep learning, implicit neural representations have emerged as a powerful alternative, including Signed Distance Functions (SDF) [18], Occupancy Networks [19] and Neural Radiance Fields (NeRF) [2]. Especially, NeRF has been demonstrated the ability to represent complex 3D shapes and textures with rich details, and has been applied extensively. Nonetheless, these methods are not without their challenges. Implicit forms can be difficult to handle. What’s more, training process is time-consuming, and often relies on a considerable number of views to optimize the representation, which may not always be feasible.

More recent advancements have led to the development of 3D Gaussian Splatting (3D-GS) [3] , a novel approach that offers a more efficient and versatile method for the representation of 3D scenes. This method could represent complete and unbounded 3D scenes by effectively ‘splatting’ Gaussians. Spherical harmonics and opacity ensure strong representation capabilities, while differentiable rasterization greatly improves rendering speed and optimization efficiency. It can be initialized based on a point cloud which is widely applied in many scenarios and relatively easy to acquire, as a strong reference of initial positional and geometric information. In addition to this, the process involves a split-and-clone mechanism that could naturally propagate new Gaussians, allowing for gradual supplementation of intricate details.

Balancing the quality and efficiency of reconstruction, 3DGS is our most suitable choice, and the split-and-clone process provides the foundation for inpainting in 3D scenes.

- B. 3D Scene Generation

3D content generation has become a focal point in the field of AI-generated content (AIGC). Generative Adversarial Networks (GAN) [20] was once particularly influential in

- 2D creation. Inspired by this, a range of GAN models are designed to produce 3D content within specific domains, such as faces, cars, cats and chairs. 3D-GAN [21], l-GAN [22], and Tree-GAN [23] utilize simple explicit primitives to represent textureless geometric shapes. HoloGAN [24] and BlockGAN [25], on the other hand, learn geometric and textural representations to generate textured 3D content. GRAF [26], Pi-GAN [27], as well as Giraffe [28], leverage implicit neural networks to achieve superior consistency and fidelity in the generated
- 3D scenes. However, training GANs is notoriously difficult due to their complex and unstable training dynamics. Beyond that, GANs struggle to effectively handle text prompts, leading to limited controllability, and their outputs are constrained by the specific training datasets, which prevents widespread application. There are some efforts, such as Point-E [4] and

Shape-E [5], that train more stable diffusion models [1] to generate 3D object end to end. But due to the scarcity of high-quality paired text-to-3D datasets, the generated content remains confined to specific domains and exhibits relatively coarse geometries and textures.

More recently, the emergence of language-image pre-trained models has catalyzed a multitude of zero-shot tasks and also has emerged as a potent tool in text-driven 3D generation. A significant amount of effort is invested in utilizing semantic priors in pre-trained models to generate domain-free objects and scenes with a high degree of detail and coherence. CLIPNeRF [7] and DreamFields [8] use the priors of CLIP [13] for supervision of optimization. Diffusion Models have made strides for generating complex data distributions [1], [29]– [31]. DreamFusion [9] introduces a method called Score Distillation Sampling (SDS), which distills high-level semantic priors from diffusion models to optimize 3D representations within different viewpoints, ensuring the consistency across viewpoints and correspondence between the prompt and the generated 3D objects. The techniques inspired a quantity of subsequent works, such as Magic3D [11], ProlificDreamer [12], HiFi-123 [32], Progressive3D [33] and DreamGaussian [34]. However, these methods only work when generating objects with relatively simple geometry, but are unable to generate large, wrap-around 3D scenes with rich details.

Several studies [14]–[17] harness the low-level priors of diffusion models as direct and explicit supervision to generate large 3D scenes from text prompts. Initially, a 2D image is either provided by the user or generated from the text prompts using a diffusion model. Subsequently, a monocular depth estimation model is employed to infer the corresponding depth information, thereby transferring the 2D image information into a 3D context. To cover a vast field of view and generate large-scale 3D scenes, these methods progressively apply the diffusion model to outpaint existing scene, following a preset trajectory. Early endeavors utilized mesh as the 3D representation. SceneScape [14] generates scenes that recede into the distance through a zoom-out trajectory, while Text2Room [15] is primarily confined to indoor scenes. However, the capabilities of mesh as a 3D representation are quite limited. More recent approaches, such as LucidDreamer [17] and Text2NeRF [16], capitalize on the robust and flexible 3D representational capabilities of 3D-GS [3] and NeRF [2] to synthesize 3D scenes that are not restricted to specific domains. However, each outpainting iteration only perceives a fraction of the existing scene, not the entire scene. This can lead to chaotic objects and overall visual inconsistencies. Additionally, the camera pose of each image outpainting step cannot be effectively constrained by diffusion models, resulting in viewpoints in the preset trajectory that are often nearly horizontal, and instability often occurs when generating the top and bottom parts. To address these limitations, our approach generates a panorama from text prompts that directly covers the panoramic field of view, followed by a 3D-GS reconstruction from the single panorama. This methodology significantly enhances the visual consistency of the scene and enables the generation of diverse fully enclosed 3D scenes.

###### “Cyberpunk style streets, skyscrapers, high-tech, futuristic”

rotate 180° horizontally

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Control Control

Tile-Controlled Diffusion Model

Lineart-Controlled Diffusion Model

Panorama Diffusion Model

[Figure 27]

Super Resolution Network

Stylized Panorama w/o Circular Blending

Extract Lineart

rotate 180° horizontally

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Base Panorama Stylized Panorama Detailed Panorama

Stylized Panorama w/ Circular Blending

(a) Stylized Equirectangular Panorama Generation

(b) Effectiveness of Circular Blending

- Fig. 2. Overview of our Stylized Equirectangular Panorama Generation. Given a user prompt, multiple diffusion models are used to generate stylized highquality panoramas. Additionally the circular blending technique is applied to avoid cracks when rotating the panorama.

3D-GS with enhanced integrity, as shown in Fig. 3, which is introduced in the following Sec. III-B.

- C. Panorama Generation

Panorama has a wide and unobstructed view that catches a vast area of scene. Some works, such as PanoGen [35] and MultiDiffusion [36], utilize pre-trained diffusion models to generate long-image from text prompt. However, these socalled “panoramas” are essentially stitched from a series of perspective images. They do not align with the true projection relationships inherent in panorama, and there is an absence of view-consistency across the entirety. Additionally, this kind of panoramas don’t include a full 360-degree horizontal field of view. Furthermore, there is often discontinuity between the leftmost and rightmost parts of the image.

A. Stylized Equirectangular Panorama Generation

In order to generate a panorama that geometrically conforms to the equirectangular projection, we apply a diffusion model that is fine-tuned using a comprehensive panorama database including both indoor and outdoor scenes. This fine-tuning process enables the model to adeptly capture the geometric principles and distinctive features inherent to the equirectangular projection. The model is used to generate the initial base panorama with a relatively reduced resolution. Its geometric features of an equirectangular projection set the groundwork for subsequent refinements and enhancements.

An intuitive solution to generate 360-degree panoramas that conforms to the equirectangular projection involves finetuning models using 360-degree panorama database. MVDiffusion [37] introduces a Correspondence-aware Attention (CAA) mechanism to simultaneously denoise and generate eight images that are consistent across views. However, all eight images are in the vertically middle area, which means it falls short of generating images covering the top and bottom parts of the panorama. StitchDiffusion [38], on the other hand, performs LoRA [39] fine-tuning to generate the whole 360-degree panorama and employs a global crop to ensure continuity between the leftmost and rightmost parts of the image. While Diffusion360 [40] uses dreambooth [41] finetuning and utilizes circular blending techniques to prevent discontinuities. For our approach, we employ the pre-trained Diffusion360 model to generate the base panorama that highly conforms to the equirectangular projection.

During the denoising process, we integrate a circular blending technique in the Diffusion360 [40] framework, facilitating the creation of a seamless panorama. In the inference phase, following each sampling iteration, the left border and the right border of the image in the latent space are subjected to a blending procedure. This method is meticulously designed to maintain spatial continuity across left and right boundaries. We extend the application of this technique to apply it on all diffusion models discussed within this subsection, thereby effectively preventing the emergence of cracks when rotating the panorama, as shown in Fig. 2(b).

Given that the majority of existing panorama datasets are comprised of real-world images, there is a risk of overfitting when employing fine-tuning techniques. This leads to a resulting domain that is highly constrained for the generated data. To mitigate this issue, we first extracted the lineart from the generated base panorama. The form of lineart effectively retains the geometric features of the equirectangular projection. Subsequently, we use a lineart-controlled diffusion model with extracted lineart as a conditional control to generate stylized panorama endowing the panorama with the ability to express to express a diversity of novel styles and features that extend beyond the confines of the original database, without compromising the geometric characteristics.

III. METHOD

We propose a text-driven 3D scene generation framework that is capable of generating fully enclosed immersive scenes with a high level of overall visual effect and rendering robustness. Firstly, we use the diffusion model to progressively generate stylized, high-quality equirectangular panorama with high view harmony based on text prompts, as shown in Fig. 2, which is introduced in the following Sec. III-A. And then we perform two-stage panorama reconstruction represented by

[Figure 33]

#### Stylized Equirectangular Panorama Generation

5

[Figure 34]

projection w/

###### O

###### Stage Ⅰ: Pre Optimization

RGBD Panorama

Point Cloud

PCD Set

PANO Set

Ps

[Figure 35]

projection w/

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

• • •

DepthFilteringDownsampling

[Figure 40]

[Figure 41]

[Figure 42]

Ip

Initialized Gaussians

Densified Gaussians

Refinement Gaussians G0

[Figure 43]

[Figure 44]

Inverse Equirectangular Projection

Panorama Depth Estimation

Image Inpainting Model Stage Ⅱ: Transfer Optimization

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Point Cloud P0

render w/

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

INP Set

PANO Set

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

• • •

＋

[Figure 58]

initialize 3D Gaussians

[Figure 59]

Dp Pf

base cameras supplementary cameras

[Figure 60]

base cameras supplementary cameras

Initialized Gaussians

Final Gaussians G1

render/projection optimization supervision

- Fig. 3. Overview of our Enhanced Two-Stage Panorama Reconstruction. We perform depth estimation on the generated panorama and then project RGBD data to obtain the point cloud. Two types of cameras — base cameras and supplementary cameras — for projection and rendering in different scenarios, and prepare three image sets for supervision at different stages of 3D-GS optimization. The rendering images of the reconstructed scene from Pre Optimization stage are inpainted for optimization in Transfer Optimization stage, resulting in the final reconstructed scene.

To ensure the clarity and visual appeal of the reconstructed scene in the forthcoming panorama reconstruction, it is essential to procure a panorama of superior resolution and enhanced detail. To this end, we harness a super-resolution network to elevate the resolution of the stylized panorama. Subsequently, we apply a tile-controlled diffusion model, which serves to augment the details of the image. This meticulous process culminates in the derivation of our final detailed panorama that will be utilized in the subsequent subsection.

B. Enhanced Two-Stage Panorama Reconstruction

Depth Estimation. Considering the diverse distributions of generated panoramas, our panorama depth estimation methodology must have a high degree of generalization ability, applicable to both bounded indoor scenes and unbounded outdoor scenes. We apply 360MonoDepth [42], a zero-shot framework, which utilizes a pre-trained monocular depth estimation model to achieve high-resolution panorama depth estimation. The process involves projecting the panorama onto twenty perspective tangent images, each corresponding to a face of an icosahedron. Using state-of-the-art monocular depth estimation techniques, we ascertain the disparity of each individual image. Subsequently, these disparity maps are globally aligned and blended into the panprama’s disparity map. We opt for the frustum blending method to combine the images, thereby effectively enhancing global smoothness.

After blending, we obtain the overall disparity map of the generated panorama. However, the scale and offset of the map remain ambiguous and require calibration to ensure accuracy. To convert the disparity map to an absolute depth map, we randomly select a subset of perspective tangent faces from the

icosahedron. Utilizing a pre-trained metric depth estimation model, we estimate absolute depth on the images corresponding to the chosen faces, and then convert these obtained metric depth maps into disparity maps, which serve as the ground truth with reference scales and offsets. By minimizing the sum of squared differences between the overall disparity map of the generated panorama and the set of reference perspective disparity maps, we determine the parameters — global offset and scale — using the least squares method. finally obtaining the metric depth of the generated panorama.

Point Cloud Reconstruction. Given the RGB image Ip and corresponding depth map Dp of the panorama, data in the form of point cloud can be easily obtained. We conduct a reverse equirectangular projection from the RGBD panorama image onto a raw point cloud, denoted as P0. The projection converts the pixel coordinates of each pixel of the panorama into 3D world coordinates by determining the corresponding longitude and latitude. In this context, the camera position of the panorama is fixed as the center of the sphere, which also serves as the origin of the world coordinate system. The reverse projection is represented by the following formula:

P0 = ϕ−erp1 ([Ip,Dp]). (1)

However, a prevalent issue across almost all existing depth estimation models is the depth mixing problem [43], which manifests as a difficulty in accurately discerning the depth of pixels proximal to object boundaries. This challenge often leads to a blending of depth values at the edges, thereby introducing artifacts in both the raw point cloud and the subsequent reconstructed 3D-GS. Despite this, the depth gradient at the edges tends to be notably steep compared to other areas of the

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

O

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

(a) Raw (b) Depth Filtered (c) Downsampled

[Figure 78]

[Figure 79]

- Fig. 4. Visual comparison of point cloud data: raw point cloud P0, depth filtered point cloud Pf, and downsampled point cloud Ps.

base cameras supplementary cameras

[Figure 80]

Fig. 5. The relationship between the base camera and corresponding supplementary cameras as well as their projection results. Supplementary cameras surround the corresponding base camera, each with variations in position and orientation. The base camera is used for projection of the panorama, while the supplementary cameras are used for projecting point clouds.

image. To mitigate the problem, we introduce a preprocessing step that involves calculating the 2D gradient on the depth map of the panorama. Subsequently, we apply a thresholdbased filter to exclude points with excessively high gradients, resulting in a point cloud denoted as Pf that is more free from artifacts, which is utilized for the initialization of the 3D-GS, as shown in Fig. 4. The initialization provides a high degree of depth and geometric priors. What’s more, we avoid resetting the opacity during the 3D-GS optimization to avoid losing the spatial information obtained from the point cloud.

sponding base camera with extrinsics Ei, where j ranges from 1 to N. Their positions and orientations have both changed compared to the corresponding base camera as shown in Fig. 5, offering multi-view supplementation. For each supplementary view, we obtain the images that constitute PCD set by projecting point clouds. Considering the efficiency of projection, the original point cloud P0 is first downsampled to lower the density, as shown in Fig. 4. The downsampled point cloud Ps is projected from the world coordinate system to the pixel coordinate system using the following formula:

Two-stage 3D-GS Reconstruction. To effectively utilize information of multiple formats, we design two types of cameras, base cameras and supplementary cameras, for projection and rendering in different scenarios, as well as three distinct image sets derived from different types of data, named PCD set, PANO set, and INP set, for supervision across various stages of optimization. This strategy helps us achieve optimal performance in both reconstruction speed and quality, and significantly enhance rendering robustness.

##### Iij,Mij = ϕ3→2(Ps,K,Eij). (3)

Where Iij is the perspective image projected by the j-th supplementary camera of the i-th base camera and Mij is the corresponding mask that illustrates the missing areas in the supplementary view Eij, which will be filled in to improve the integrity during the subsequent 3D-GS optimization process.

Specifically, the panorama Ip is converted to a series of perspective images for supervision of the 3D-GS optimization. We configure a set of M base cameras, with each camera sharing the same intrinsic parameters denoted by K. The extrinsic parameters are denoted as Ei, describing the specific pose of the i-th base camera. These camera poses are strategically arranged to provide coverage across the entirety of a sphere projected from the panorama. All cameras are positioned at the center of the sphere. These images projected from panorama Ip using base cameras, constitute the PANO set and inherit the high-resolution characteristics of the panorama Ip. Supervision with PANO set ensures the fidelity of the reconstruction. The projection is based on the following formula:

Initiating with the filtered point cloud Pf, we execute a two-stage 3D-GS optimization utilizing the previously derived image sets. In the Pre Optimization stage, the PCD set is initially engaged for supervision. These images furnish multiview constraints within the initial densification process, ensuring the appropriate spatial arrangement of the emergent 3D Gaussians. Nonetheless, the downsampling process of the point cloud diminishes the resolution of images within the PCD set, leading to a loss of clarity in the preliminary Gaussian representations. To restore more details in the panorama, the PANO set, characterized by its high-resolution images derived from the panorama, is then used solely for supervision to refine the densified Gaussians. This subsequent refinement of the 3D Gaussians is instrumental in preserving the fidelity of the reconstructed scene G0.

Ii = ϕerp2pers(Ip,K,Ei). (2) Where Ii is the image projected by the i-th base camera.

However, the images in PANO set offer very limited camera poses. Because it is designed for application to panoramic images, the position of the base cameras is restricted to a single location. The scarcity of diverse viewpoints for supervision can easily lead to overfitting on constrained poses and poor rendering robustness when the camera moves, including the emergence of visual artifacts and excessively elongated 3D Gaussians. To overcome this, we add an additional sample of N supplementary cameras with the same shared intrinsic parameters K and extrinsics Eij which surround the corre-

Because the panorama has only a single viewpoint, the reconstructed scene contains numerous missing regions as a result of object occlusions. In the second stage, we render Gaussians G0 for each supplementary view Eij and then use an image inpainting model to obtain Vij filled in the missing pixels. The formula is as follows:

###### Vij,= Finpaint(RG(G0,K,Eij),Mij). (4)

Base Panorama Stylized Panorama Detailed Panorama

[Figure 81]

[Figure 82]

[Figure 83]

“A tranquil Chinese ink wash scene with ethereal mountains and a misty river.”

[Figure 84]

[Figure 85]

[Figure 86]

“A vibrant fantasy realm with magical forests and floating castles.”

[Figure 87]

[Figure 88]

[Figure 89]

“A bustling cartoon village with many anthropomorphic animals.”

- Fig. 6. Comparison of Base Panorama, Stylized Panorama, and Detailed Panorama. Stylized Panorama has a style that closely matches the description than Base Panorama. Furthermore, Detailed Panorama is added more details.

Where i ranges from 1 to M, and j ranges from 1 to N, these inpainted images constitute the third image set, named INP set. Ultimately, we perform Transfer Optimization on newly initialized Gaussians. The INP set and the PANO set are integrated as supervision concurrently to achieve final reconstruction Gaussians G1 with greater rendering robustness. The split-and-clone process of 3D-GS automatically inpaint missing regions during optimization.

TABLE I QUANTITATIVE COMPARISON OF IMAGE AESTHETIC AND QUALITY FOR PANORAMAS GENERATED BY OUR METHOD AND BASELINES IN THE FORM OF PANORAMA AND FOUR MIDDLE FACES OF CUBEMAP.(BEST)

Panorama Cube Faces TANet↑ CLIP-Aesthetic↑ TANet↑ CLIP-Aesthetic↑ Outpainting 5.397 5.755 5.438 5.739

Method

MultiDiffusion [36] 5.459 5.956 5.392 5.848 MVDiffusion [37] 5.394 5.642 5.463 5.775 Ours 5.583 6.198 5.521 6.017

IV. EXPERIMENTS

In this section, we employ rigorous and comprehensive experiments to demonstrate the superiority of our approach. The evaluation is segmented into two primary components: experiments of panorama generation in Sec. IV-A and experiments of panorama reconstruction in Sec. IV-B. We carefully compare our method with benchmark methodologies and conduct additional ablation study to evaluate the generation capability and the reconstruction performance of our framework, respectively.

Both methods require no training of the reference diffusion model. While MVDiffusion fine-tunes a diffusion model with correspondence-aware attention (CAA) mechanism on a panorama dataset enabling the generation of eight cross-view consistent images, but it encounters challenges in producing the top and bottom parts of the panorama.

Comparisons. We compare our panorama generation approach with baseline methods on different text prompts in the form of both panorama and cubemap, as shown in Fig. 7. Due to the lack of global consideration, Outpainting can lead to severely chaotic objects and global inconsistencies. The panoramas generated by MultiDiffusion are planar and do not conform to equirectangular projection, resulting in distortion after projection onto perspective images, and failing to ensure 360-degree continuity. MVDiffusion generates multiple images and stitches them together, which reduces the overall consistency and harmony of the stitched panorama. In contrast, our method directly generates high-quality panoramas that conform to the equirectangular projection, achieving a high level of overall consistency and harmony in the scene, and also ensuring excellent visual effects in perspective views.

- A. Experiments of Panorama Generation

Baseline Methods. We compare our Stylized Equirectangular Panorama Generation approach with three panorama generation methods: Outpainting, MultiDiffusion [36] and MVDiffusion [37]. Outpainting is a widely utilized progressive generation method prevalent in the field of 3D scene generation. It initiates from an initial image and progressively generates outward extensions, guided by a reference diffusion model and following a preset camera trajectory. MultiDiffusion leverages a reference diffusion model to incrementally generate a long image that can be considered as the middle part of the 360degree panorama. It achieves this by simultaneously constraining multiple image crops during the denoising process.

We employ two image aesthetic quality assessment met-

“A bustling ancient market with stalls, vendors' cries, and a variety of goods”

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Outpainting

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

MultiDiffusion (ICML 2023)

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

MVDiffusion (NeurPIS 2023)

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Ours

###### “An industrial-style studio with exposed brick walls, metal pipes, and vintage furniture”

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Outpainting

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

MultiDiffusion (ICML 2023)

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

MVDiffusion (NeurPIS 2023)

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Ours

“A luxurious villa on top of a mountain with panoramic floor-to-ceiling windows, mountain views, and a private pool”

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Outpainting

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

MultiDiffusion (ICML 2023)

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

MVDiffusion (NeurPIS 2023)

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Ours

- Fig. 7. Qualitative comparison of panoramas generated by our method and baselines based on diverse text prompts. We show the panoramas and middle faces of the corresponding cubemaps for an intuitive assessment of visual effects.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Text2Room (ICCV 2023)

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Text2NeRF (TVCG 2024)

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

LucidDreamer

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Ours

- Fig. 8. Qualitative comparison of panorama reconstruction with baselines from different panoramas. Our method effectively avoids artifacts and fill in the missing areas, achieving better rendering robustness.

rics, TANet [44] and CLIP-aesthetic [45], to quantitatively compare the aesthetic quality both on panoramic images and cubemap images, as shown in Tab. I. Our method achieved higher aesthetic quality scores on both forms of images, which demonstrates the superiority of our method over the baseline in terms of visual effects. Additionally, we compare the panoramas at different stages of the generation process in Fig. 6, which confirms the effectiveness of applying multiple diffusion models in enhancing style and detail.

- B. Experiments of Panorama Reconstruction

Baseline Methods. We compare our Enhanced Two-Stage Panorama Reconstruction approach with three 3D scene generation methods that combine the outpainting technique with different 3D representations: Text2Room [15] represented with mesh, Text2NeRF [16] represented with NeRF and LucidDreamer [17] represented with 3D-GS. All these baseline methods utilize a diffusion model to outpaint rendered RGB image and progressively generate the overall scene. Text2Room directly extracts mesh from inpainted RGBD image to represent watertight indoor scenes. Text2NeRF uses inpainted RGBD images as supervision to train a NeRF network with proposed depth loss. While LucidDreamer projects the outpainted RGBD images into the point cloud, and subsequently projects multiple images from the point cloud for supervision during the optimization of 3D-GS. Given the

RGB panorama images and corresponding depth map obtained by our method, we avoid outpainting and instead adjust the respective baseline methods to directly reconstruct the single panorama. For Text2Room, we extract the mesh directly from the panorama without filtering the mesh to avoid excessive holes. Text2NeRF is supervised using the PANO set projected from the panorama. And LucidDreamer is supervised using the images projected from raw point cloud P0.

Comparisons. We compare our panorama reconstruction approach with baseline methods on different generated panoramas, as shown in Tab. II. We evaluate the reconstruction quality, rendering robustness and average reconstruction time of the panorama reconstruction. For the evaluation of reconstruction quality, base cameras are used to render images from 3D scenes, and corresponding PANO set serves as the reference. We employ three reference image quality evaluation metrics: structure similarity index measure (SSIM) , peak signalto-noise ratio (PSNR), and learned perceptual image patch similarity (LPIPS) [46], to evaluate the quality of the rendered images. For the evaluation of rendering robustness, we render images using supplementary cameras, and employ traditional no-reference image quality assessment metrics: Natural Image Quality Evaluator (NIQE) [47] and Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) [48].

Notably, taking into account the final file size of 3D-GS, the point cloud derived from our process is of a reduced size

TABLE II QUANTITATIVE COMPARISONS OF PANORAMA RECONSTRUCTION FROM SINGLE PANORAMA ON 3D REPRESENTATION, RECONSTRUCTION QUALITY, RENDERING ROBUSTNESS AND RECONSTRUCTION TIME. (BEST)

Reconstruction Quality Rendering Robustness

Method 3D Representation

Reconstruction Time (hours) PSNR↑ SSIM↑ LPIPS↓ NIQE↓ BRISQUE↓

Text2Room [15] mesh 34.497 0.957 0.036 5.935 32.829 0.008 Text2NeRF [16] NeRF 34.336 0.927 0.162 7.316 32.891 9.738

LucidDreamer [17] 3D-GS 34.501 0.958 0.068 6.255 44.738 0.962 Ours 3D-GS 40.189 0.984 0.041 5.372 32.372 0.271

TABLE III QUANTITATIVE ABLATION STUDIES OF PANORAMA RECONSTRUCTION FROM SINGLE PANORAMA. WE EVALUATE THE EFFECTS OF IMPORTANT COMPONENTS ON RENDERING ROBUSTNESS. (BEST)

[Figure 174]

[Figure 175]

[Figure 176]

Ours w/o initial & filter

Method NIQE↓ BRISQUE↓ TReS↑ MANIQA↑

Full 5.372 32.372 77.009 0.402 w/o initial 5.369 32.598 76.491 0.399

[Figure 177]

[Figure 178]

[Figure 179]

w/o filter 5.441 32.693 76.990 0.401 w/o initial & filter 5.419 32.625 76.367 0.398

Ours w/o filter

To improve the discrimination, we incorporate two additional deep learning-based no-reference image quality assessment metrics: Manifold based Image Quality Assessment (ManIQA) [49] and Training-free Referenceless Image Quality Evaluator (TReS) [50]. The incorporation of these metrics allows for a more nuanced evaluation of image quality without the need for reference images. As Fig. 9 depicts, depth filtering of the point cloud and multi-view constraints of PCD set collectively contribute to reducing artifacts and inpainting missing regions, demonstrating the indispensable nature of each component.

[Figure 180]

[Figure 181]

[Figure 182]

Ours w/o initial

[Figure 183]

[Figure 184]

[Figure 185]

Ours full

V. LIMITATIONS AND FUTURE WORK

Although our research has yielded commendable outcomes, several challenges persist with our current model. Panoramic image data are significantly scarcer compared to perspective image data, which poses a substantial challenge for the development of panorama generation methods. The data scarcity limits the capacity for effectively processing more complex text descriptions during generation. Combining of multiple diffusion models can expand the domain for generated panoramas, but can also result in compounded errors and an increase in stochastic variability. It is conceivable that forthcoming video generation models could contribute to producing more extensive and diverse panorama datasets.

- Fig. 9. Effective validation of depth filtering of point cloud and initial optimization with PCD set in our panorama reconstruction approach. These two components effectively mitigated the occurrence of artifacts.

relative to the original RGB panoramic images. which leads to a loss of pixel information in the point clouds. Text2Room and LucidDreamer perform reconstruction based solely on the point clouds and consequently the reconstruction quality suffers loss. Although Text2NeRF could use RGB images from PANO set that is decoupled from depth information for supervision, the poor training efficiency of NeRF leads to excessively long reconstruction times. Our method, however, is capable of rapidly reconstructing from a single panorama and achieving excellent reconstruction quality. Additionally, All baseline methods do not adequately account for the robustness of rendering, which results in the presence of artifacts or missing regions within the reconstructed scenes, as shown in Fig. 8. Our method significantly enhances the quality of rendered images under supplementary view, effectively improving the rendering robustness of the reconstructed scene. Ablation Study. Furthermore, we conduct ablation studies to ascertain the critical role of depth filtering of the point cloud and initial optimization using PCD set in the Pre Optimization stage in rendering robustness, as shown in Tab. III.

Additionally, to ensure the efficiency of the 3D reconstruction, we have limited our approach to a two-stage reconstruction, with additional cameras introduced in the second stage to fill in missing areas. To further enhance the integrity of the scene and the robustness of the rendering, future work could increase the number of iterative inpainting stages. In addition, optimizing the camera setup strategy for each stage and refining the parameters of the optimization will be necessary to balance reconstruction quality and efficiency.

VI. CONCLUSION

In this paper, we introduce HoloDreamer for generating highly consistent, fully enclosed 3D scenes with enhanced

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

“A futuristic shopping center with sleek flooring, interactive screens, and trendy shops.”

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

“A classical library filled with ancient books and scrolls, oak bookcases, and vintage chandeliers.”

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

“A smokey grey kitchen with modern touches, small windows opening onto the garden, Bauhaus furniture and decoration, high ceiling, beige blue salmon pastel palette, interior design magazine, cozy atmosphere, Basket of fruits and vegetables, a bottle of drinking water, walls painted magazine style photo, wooden parquet floor.”

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

(a) Generation Results of Indoor Scenes.

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

“A mountain town in anime style with blooming cherry blossoms, quaint streets.”

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

“A dense tropical rainforest with towering trees, exotic birds, and waterfalls.”

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

“A vibrant, colorful floating community city, clouds above a beautiful, enchanted landscape filled with whimsical flora, enchanted forest landscape, unique fantastical elements like floating islands and floating orbs, Highly detailed vegetation and foliage, deep contrast and color vibrancy, texture and intricate details in a floating element.”

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

(b) Generation Results of Outdoor Scenes.

- Fig. 10. More results of our 3D scene generation. Our approach can generate fully enclosed 3D scenes with high consistency and style diversity, both for indoor and outdoor environments.

rendering robustness based on text descriptions. The pipeline consists of two proposed modules: Stylized Equirectangular Panorama Generation and Enhanced Two-Stage Panorama Reconstruction. Our method not only refines the visual consistency and visual harmony but also enhances the integrity of the scenes and robustness of the rendering. The results of extensive experiments indicate that HoloDreamer represents a significant advancement in the field of 3D scene creation, offering a comprehensive solution that transforms text descriptions into intricate, immersive, and visually coherent 3D scenes.

APPENDIX IMPLEMENT DETAILS

We implement HoloDreamer with the PyTorch framework. In the stage of panorama generation, we use the base model pre-trained in Diffusion360 [40], which was fine-tuned using the dreambooth [41] training method in the SUN360 dataset

[51] to produce the base panorama with a resolution of 1024x512. For style transfer on the panorama, we employ version 1.1 of ControlNet Lineart [52], which is based on version 1.5 of Stable Diffusion [1], to generate the stylized panorama with a resolution of 1536x728. Subsequently, we refine the panorama by using the ControlNet Tile and Real-ESRGAN [53] following the refinement process in Diffusion360, achieving a detailed panorama of 6144x3072 resolution.

For depth estimation of the panorama, we balance quality and speed by initially downsampling the panorama to 4096x2048 resolution. We adhere to the strategies and parameter settings of the image projection and alignment stage in 360monodepth [42]. Disparity estimation is conducted using Depth Anything [54], a zero shot monocular relative depth estimation model, and then blended with frustum weights. Subsequently, ZoeDepth-NK [55] is utilized to estimate metric depth to provide a reference for converting disparity map into a depth map. The size of the raw point cloud P0 is the same as the resolution of the depth map, that is, 4096x2048, and the downsampled point cloud Ps is 1024x512. To obtain depth filtered point cloud Pf, the point cloud is first downsampled to 2048x1024, and the threshold of depth gradient is set to 0.4. During the phase of the 3D Gaussian optimization, the camera’s intrinsic parameters K are identical to the settings in LucidDreamer [17]. All perspective images for supervision have the same resolution of 512x512. For the trajectory of the base cameras, we arrange a total of 38 base views that provide a comprehensive coverage of a sphere, with each base camera corresponding to 4 supplementary cameras positioned above, below, left and right. We use LaMa [56] to fill in the missing pixels of rendered images. The learning rate for the optimization of 3D Gaussian Splatting (3D-GS) is consistent with the original paper’s [3] settings. The split and clone technique triggers every 100 iterations. The Pre Optimization stage involves an initial 2000 iterations for the optimization using PCD set and subsequently 2000 iterations for the optimization using PANO set. Furthermore, the Transfer Optimization stage consists of a total of 5,000 iterations.

In addition, we compare the performance of 360monodepth using Depth Anything and MiDaS (in the original paper

TABLE IV QUANTITATIVE COMPARISON OF 360MONODEPTH USING MIDAS AND DEPTH ANYTHING ON REPLICA360-4K AT 4096×2048 WITH FRUSTUM BLENDING AND MULTI-SCALE DEFORMABLE ALIGNMENT.(BEST)

Method AbsRel↓ MAE↓ RMSE↓ RMSE-log↓ δ<1.25↑ δ<1.252↑ δ<1.253↑

MiDaS v2 [57] 0.153 0.346 0.579 0.082 0.810 0.949 0.982 MiDaS v3 [58] 0.148 0.341 0.570 0.079 0.814 0.969 0.991

Depth Anything [54] 0.116 0.281 0.481 0.063 0.894 0.984 0.995

[42]) in Tab.IV, which proves that Depth Anything has better accuracy in panorama disparity estimation.

REFERENCES

- [1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10674–10685.
- [2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in European Conference on Computer Vision, 2020.
- [3] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 1–14, 2023.
- [4] A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, “Point-e: A system for generating 3d point clouds from complex prompts,” arXiv preprint arXiv:2212.08751, 2022.
- [5] H. Jun and A. Nichol, “Shap-e: Generating conditional 3d implicit functions,” arXiv preprint arXiv:2305.02463, 2023.
- [6] Z. Liu, Y. Feng, M. J. Black, D. Nowrouzezahrai, L. Paull, and W. Liu, “Meshdiffusion: Score-based generative 3d mesh modeling,” in International Conference on Learning Representations, 2023.
- [7] C. Wang, M. Chai, M. He, D. Chen, and J. Liao, “Clip-nerf: Text-andimage driven manipulation of neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3835–3844.
- [8] A. Jain, B. Mildenhall, J. T. Barron, P. Abbeel, and B. Poole, “Zeroshot text-guided object generation with dream fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 867–876.
- [9] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Text-to-3d using 2d diffusion,” in International Conference on Learning Representations, 2023.
- [10] H. Wang, X. Du, J. Li, R. A. Yeh, and G. Shakhnarovich, “Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12619–12629.
- [11] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, “Magic3d: High-resolution text-to3d content creation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 300–309.
- [12] Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu, “Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [13] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.
- [14] R. Fridman, A. Abecasis, Y. Kasten, and T. Dekel, “Scenescape: Textdriven consistent scene generation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [15] L. H¨ollein, A. Cao, A. Owens, J. Johnson, and M. Nießner, “Text2room: Extracting textured 3d meshes from 2d text-to-image models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7909–7920.
- [16] J. Zhang, X. Li, Z. Wan, C. Wang, and J. Liao, “Text2nerf: Text-driven 3d scene generation with neural radiance fields,” IEEE Transactions on Visualization and Computer Graphics, 2024.
- [17] J. Chung, S. Lee, H. Nam, J. Lee, and K. M. Lee, “Luciddreamer: Domain-free generation of 3d gaussian splatting scenes,” arXiv preprint arXiv:2311.13384, 2023.

- [18] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, “Deepsdf: Learning continuous signed distance functions for shape representation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 165–174.
- [19] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger, “Occupancy networks: Learning 3d reconstruction in function space,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 4460–4470.
- [20] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Communications of the ACM, vol. 63, no. 11, pp. 139–144, 2020.
- [21] J. Wu, C. Zhang, T. Xue, B. Freeman, and J. Tenenbaum, “Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling,” Advances in Neural Information Processing Systems, vol. 29, 2016.
- [22] P. Achlioptas, O. Diamanti, I. Mitliagkas, and L. Guibas, “Learning representations and generative models for 3d point clouds,” in International Conference on Machine Learning. PMLR, 2018, pp. 40–49.
- [23] D. W. Shu, S. W. Park, and J. Kwon, “3d point cloud generative adversarial network based on tree structured graph convolutions,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 3859–3868.
- [24] T. Nguyen-Phuoc, C. Li, L. Theis, C. Richardt, and Y.-L. Yang, “Hologan: Unsupervised learning of 3d representations from natural images,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 7588–7597.
- [25] T. H. Nguyen-Phuoc, C. Richardt, L. Mai, Y. Yang, and N. Mitra, “Blockgan: Learning 3d object-aware scene representations from unlabelled images,” Advances in Neural Information Processing Systems, vol. 33, pp. 6767–6778, 2020.
- [26] K. Schwarz, Y. Liao, M. Niemeyer, and A. Geiger, “Graf: Generative radiance fields for 3d-aware image synthesis,” Advances in Neural Information Processing Systems, vol. 33, pp. 20154–20166, 2020.
- [27] E. R. Chan, M. Monteiro, P. Kellnhofer, J. Wu, and G. Wetzstein, “pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 5799–5809.
- [28] M. Niemeyer and A. Geiger, “Giraffe: Representing scenes as compositional generative neural feature fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11453–11464.
- [29] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in Neural Information Processing Systems, vol. 35, pp. 36479–36494, 2022.
- [30] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in International Conference on Machine Learning. PMLR, 2021, pp. 8821–8831.
- [31] A. Q. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mcgrew, I. Sutskever, and M. Chen, “GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 17–23 Jul 2022, pp. 16784–16804. [Online]. Available: https://proceedings.mlr.press/v162/nichol22a.html
- [32] W. Yu, L. Yuan, Y.-P. Cao, X. Gao, X. Li, L. Quan, Y. Shan, and Y. Tian, “Hifi-123: Towards high-fidelity one image to 3d content generation,” arXiv preprint arXiv:2310.06744, 2023.
- [33] X. Cheng, T. Yang, J. Wang, Y. Li, L. Zhang, J. Zhang, and L. Yuan, “Progressive3d: Progressively local editing for text-to-3d content creation with complex semantic prompts,” 2024.
- [34] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, “Dreamgaussian: Generative gaussian splatting for efficient 3d content creation,” in International Conference on Learning Representations, 2024.
- [35] J. Li and M. Bansal, “Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [36] O. Bar-Tal, L. Yariv, Y. Lipman, and T. Dekel, “Multidiffusion: Fusing diffusion paths for controlled image generation,” in Proceedings of the 40th International Conference on Machine Learning, vol. 202. PMLR, 23–29 Jul 2023, pp. 1737–1752.
- [37] S. Tang, F. Zhang, J. Chen, P. Wang, and Y. Furukawa, “Mvdiffusion: Enabling holistic multi-view image generation with correspondenceaware diffusion,” in Advances in Neural Information Processing Systems, vol. 36. Curran Associates, Inc., 2023, pp. 51202–51233.

- [38] H. Wang, X. Xiang, Y. Fan, and J.-H. Xue, “Customizing 360-degree panoramas through text-to-image diffusion models,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 4933–4943.
- [39] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.
- [40] M. Feng, J. Liu, M. Cui, and X. Xie, “Diffusion360: Seamless 360 degree panoramic image generation based on diffusion models,” arXiv preprint arXiv:2311.13141, 2023.
- [41] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22500–22510.
- [42] M. Rey-Area, M. Yuan, and C. Richardt, “360monodepth: Highresolution 360deg monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3762–3772.
- [43] J. Hu, C. Bao, M. Ozay, C. Fan, Q. Gao, H. Liu, and T. L. Lam, “Deep depth completion from extremely sparse data: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 8244–8264, 2022.
- [44] S. He, Y. Zhang, R. Xie, D. Jiang, and A. Ming, “Rethinking image aesthetics assessment: Models, datasets and benchmarks,” IJCAI, 2022.
- [45] C. Schuhmann, R. Beaumont, R. Vencu, C. W. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, P. Schramowski, S. R. Kundurthy, K. Crowson, L. Schmidt, R. Kaczmarczyk, and J. Jitsev, “LAION-5b: An open large-scale dataset for training next generation image-text models,” in Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022.
- [46] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.
- [47] A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a “completely blind” image quality analyzer,” IEEE Signal processing letters, vol. 20, no. 3, pp. 209–212, 2012.
- [48] A. Mittal, A. K. Moorthy, and A. C. Bovik, “No-reference image quality assessment in the spatial domain,” IEEE Transactions on image processing, vol. 21, no. 12, pp. 4695–4708, 2012.
- [49] S. Yang, T. Wu, S. Shi, S. Lao, Y. Gong, M. Cao, J. Wang, and Y. Yang, “Maniqa: Multi-dimension attention network for no-reference image quality assessment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 1191–1200.
- [50] S. A. Golestaneh, S. Dadsetan, and K. M. Kitani, “No-reference image quality assessment via transformers, relative ranking, and selfconsistency,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2022, pp. 1220–1230.
- [51] J. Xiao, K. A. Ehinger, A. Oliva, and A. Torralba, “Recognizing scene viewpoint using panoramic place representation,” in 2012 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2012, pp. 2695–2702.
- [52] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, October 2023, pp. 3836– 3847.
- [53] X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-esrgan: Training realworld blind super-resolution with pure synthetic data,” in International Conference on Computer Vision Workshops.
- [54] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [55] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. M¨uller, “Zoedepth: Zero-shot transfer by combining relative and metric depth,” arXiv preprint arXiv:2302.12288, 2023.
- [56] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, and V. Lempitsky, “Resolution-robust large mask inpainting with fourier convolutions,” in 2022 IEEE/CVF Winter Conference on Applications of Computer Vision, 2022, pp. 3172–3182.
- [57] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 3, 2022.

[58] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” ICCV, 2021.

[Figure 269]

Haiyang Zhou is an undergraduate student at the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China. He is currently interning at School of Electron and Computer Engineering, Peking University. His research interests include AIGC and 3D computer vision.

[Figure 270]

Li Yuan received the B.E. degree from University of Science and Technology of China, in 2017, and the PhD degree from National University of Singapore, in 2021. He is currently a tenure-track assistant professor with School of Electrical and Computer Engineering with Peking University. He has published more than 40 papers on top conferences/journals. His research interests include deep learning, image processing, and computer vision.

[Figure 271]

Xinhua Cheng recieved B.E. degree of computer science at College of Computer Science, Sichuan University. He is currently a PhD student of computer applications technology at School of Electron and Computer Engineering, Peking University, China. His research interests include 3D vision and AIGC, especially text-to-3D content creation and editing.

[Figure 272]

Wangbo Yu received the B.E. degree in telecommunications engineering from Xidian University, Xi’an, China, in 2021. He is currently a PhD student with Peking University, Beijing, China. His research interests include computer vision, machine learning, and computer graphics.

Yonghong Tian (Fellow, IEEE) is currently the Dean of the School of Electronics and Computer Engineering, a Boya Distinguished Professor with the School of Computer Science, Peking University, China, and the Deputy Director of the Artificial Intelligence Research, Peng Cheng Laboratory, Shenzhen, China. He is the author or coauthor of over 350 technical papers in refereed journals and conferences. His research interests include neuromorphic vision, distributed machine learning, and AI for science. He is a TPC Member of more than ten

[Figure 273]

conferences, such as CVPR, ICCV, ACM KDD, AAAI, ACM MM, and ECCV. He is a Senior Member of CIE and CCF and a member of ACM. He was a recipient of the Chinese National Science Foundation for Distinguished Young Scholars in 2018, two National Science and Technology Awards, and three ministerial-level awards in China. He received the 2015 Best Paper Award for EURASIP Journal on Image and Video Processing, the Best Paper Award from IEEE BigMM 2018, and the 2022 IEEE SA Standards Medallion and SA Emerging Technology Award. He served as the TPC Co-Chair for BigMM 2015, the Technical Program Co-Chair for IEEE ICME 2015, IEEE ISM 2015, and IEEE MIPR 2018/2019, and the General Co-Chair for IEEE MIPR 2020 and ICME 2021. He was/is an Associate Editor of IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY from January 2018 to December 2021, IEEE TRANSACTIONS ON MULTIMEDIA from August 2014 to August 2018, IEEE Multimedia Magazine from January 2018 to August 2022, and IEEE ACCESS from January 2017 to December 2021. He co-initiated the IEEE International Conference on Multimedia Big Data (BigMM).

