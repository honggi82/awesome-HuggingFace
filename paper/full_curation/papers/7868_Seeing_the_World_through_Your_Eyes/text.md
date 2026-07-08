## Seeing the World through Your Eyes

Hadi Alzayer* Kevin Zhang* Brandon Feng Christopher Metzler Jia-Bin Huang University of Maryland, College Park https://world-from-eyes.github.io/

[Figure 1]

[Figure 2]

# arXiv:2306.09348v2[cs.CV]2Mar2024

Input Output Input Output

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

... ...

Captured frames Captured frames

... ...

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Cropped eye images Novel view rendering Cropped eye images Novel view rendering

Figure 1. Radiance field reconstruction using eye reflections. The human eye is highly reflective. We show that from a sequence of frames that capture a moving head, we can reconstruct and render the 3D scene of what the person is observing using only the reflections off their eyes.

### Abstract

### 1. Introduction

The only true voyage of discovery ... would be not to visit strange lands but to possess other eyes, to behold the universe through the eyes of another ... – Marcel Proust, 1927

The reflective nature of the human eye is an underappreciated source of information about what the world around us looks like. By imaging the eyes of a moving person, we can collect multiple views of a scene outside the camera’s direct line of sight through the reflections in the eyes. In this paper, we reconstruct a 3D scene beyond the camera’s line of sight using portrait images containing eye reflections. This task is challenging due to 1) the difficulty of accurately estimating eye poses and 2) the entangled appearance of the eye iris and the scene reflections. Our method jointly refines the cornea poses, the radiance field depicting the scene, and the observer’s eye iris texture. We further propose a simple regularization prior on the iris texture pattern to improve reconstruction quality. Through various experiments on synthetic and real-world captures featuring people with varied eye colors, we demonstrate the feasibility of our approach to recover 3D scenes using eye reflections.

The human eye is a remarkable organ that enables vision and holds valuable information about the surrounding world. While we typically use our own eyes as two lenses to focus light onto the photosensitive cells composing our retina, we would also capture the light reflected from the cornea if we look at someone else’s eyes. When we use a camera to image the eyes of another, we effectively turn their eyes as a set of mirrors in the overall imaging system. Since the light that reflects off the observer’s eyes share the same source as the light that reaches their retina, our camera should form images containing information about the world the observer sees.

Prior studies have explored recovering a panoramic image of the world the observer sees from an image of two eyes [30, 31]. Follow-up works have further explored applications such as personal identification [12,28], detecting

*Equal contribution

[Figure 19]

[Figure 20]

grasp posture [53], focused object estimation [42], and relighting [29]. Given the recent advancements in 3D vision and graphics, we wonder: Can we do more than reconstruct a single panoramic environment map or recognize patterns? Is it possible to recover the world seen by the observer in full 3D?

[Figure 21]

[Figure 22]

Movingperson

Fixed camera

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

In this paper, we answer these questions by reconstructing a 3D scene from a sequence of eye images. We start from the insight that our eyes capture/reflect multi-view information as we naturally move our heads. We draw inspiration from the classical imaging formulation proposed by [30] and integrate it with the recent advances in 3D reconstruction spearheaded by Neural Radiance Fields (NeRF) [26]. Unlike the standard NeRF capture setup, which requires a moving camera to capture multi-view information (often followed by camera pose estimation), our approach employs a stationary camera and extracts the multi-view cues from eye images under head movement.

Moving cameras (Outside camera view)

(a) NeRF Setup (b) Our Setup

Figure 2. NeRF for non-line-of-sight scene. The typical NeRF capture setup requires multiple posed images (e.g., captured from a moving camera) for reconstruction. In our setup, we gather multi-view information of the scene through light reflected from the eyes of a moving person.

- • Radial prior for irises. We introduce a radial prior for iris texture decomposition in eye images, significantly improving the quality of the reconstructed radiance field.
- • Cornea pose refinement. We develop a cornea pose refinement procedure to alleviate the noisy pose estimates of eyes, which overcomes the unique challenge of extracting features from human eyes.

While conceptually straightforward, reconstructing a 3D NeRF from eye images is extremely challenging in practice. The first challenge is source separation. We need to separate the reflections from the intricate iris textures of human eyes. These complex patterns add a level of ambiguity to the 3D reconstruction process. Unlike the clear images of the scene typically assumed in standard captures, the eye images we obtain are inherently blended with iris textures. This composition disrupts the pixel correspondence and complicates the reconstruction process. The second challenge is cornea pose estimation. Eyes are small and hard to localize accurately from image observations. The multi-view reconstruction, however, depends on the accuracy of their locations and 3D orientations.

These advancements extend the current capabilities of 3D scene reconstruction through neural rendering to handle partially corrupted image observations obtained from eye reflections, opening up new possibilities for research and development in the broader area of accidental imaging [6, 20,38,45] to reveal and capture 3D scenes beyond the visible line-of-sight.

To address these challenges, in this work, we repurpose NeRF for training on eye images by introducing two crucial components: a) texture decomposition, which leverages a simple radial prior to facilitate separating the iris texture from the overall radiance field, and b) eye pose refinement, which enhances the accuracy of pose estimation despite the challenges presented by the small size of eyes.

### 2. Related Work

Catadioptric imaging. Catadioptric imaging uses a combination of lenses and mirrors for image capturing. The word catadioptric is derived from catoptrics (related to the Greek words for specular and mirrors) and dioptrics (related to an Ancient Greek lens-like instrument). In essence, catadioptric imaging seeks to leverage an additional (often curved) mirror to expand a lens-based imaging system’s effective field of view. Early studies in catadioptric imaging focused primarily on the design of the mirror profiles and their impact on the final image quality. [2] studied three design criteria of a catadioptric imaging system: the shape of the mirrors, the resolution of the cameras, and the focus settings of the cameras. [41] provided a metric to quantify distortions and a method to minimize distortions in images acquired with a single viewpoint catadioptric camera. Moreover, a creative way to realize an accidental catadioptric imaging system is by treating human eyes as external

To evaluate the performance and effectiveness of our approach, we generate a synthetic dataset of a complex indoor environment with images that capture the reflection from a synthetic cornea with realistic texture. We further implement a real-world setup with multiple objects to capture eye images. We conduct extensive experiments on synthetic and real-world captured eye images to validate several design choices in our approach.

Our primary contributions are as follows:

• New 3D reconstruction problem. We present a novel method for reconstructing 3D scenes of the observer’s world from eye images, integrating earlier foundational work with the latest advancements in neural rendering.

curved mirrors [31]. [30] uses a single image of the eyes as a stereo system to identify pixel correspondences with epipolar geometry, even successfully identifying what the person is looking at. Another application of using human eyes as part of the imaging system is estimating light direction from the eyes to perform relighting [29,46]. Our work draws inspiration from previous works on eye-based catadioptric imaging systems and further extends this concept to achieve 3D scene recovery through NeRF-based modeling. In particular, this paper introduces several new techniques to process catadiotrically captured eye images, such as learnable texture decomposition and refined iris estimations.

Neural radiance field. Neural radiance fields (NeRF) [26] represent a significant milestone in novel view synthesis. NeRF adopts differentiable volume rendering to represent a 3D scene and uses neural networks to learn the density and color of each scene point. Following the success of NeRF, a plethora of follow-up works have been introduced to improve its rendering quality [3,4], ability to handle scene dynamics [10, 17, 22, 34–36], inaccurate camera poses [5,18,22,25,50], and rendering speed [1,27,52]. Our work uses NeRF to parametrize the unknown scene we wish to recover from eye reflections. In particular, we modify the training framework from nerfstudio [43] to implement the NeRF-based scene reconstruction. We note that our input images are captured at a fixed viewpoint, which differs from the typical NeRF setup, which requires multi-view input with additional requirements of camera pose optimization.

Reflection removal. Removing reflections from captured images is a longstanding computational photography problem. The related literature on this topic can be summarized into two main categories: multi-frame and single-image. Multi-frame reflection removal methods [9, 23, 24, 40, 51] often exploit the differences of motion patterns between the background and reflection layers and impose various image priors as regularization. Single-image reflection removal methods tend to exploit visual cues available in a single image, such as depth-of-field [16, 49], defocusdisparity [37], or learned image features [55]. More recently, NeRF has emerged as a new tool for reflection removal, specifically under the multi-frame setting. Various NeRF-based methods have studied how to accurately model and extract specular reflections from shiny or metallic objects [7,44,48,54]. Nerfren [11] demonstrates that by fitting two NeRFs to model the reflection and diffuse components of the scene separately, reflections from planar surfaces like mirrors can be removed and re-rendered as a separate 3D scene. Due to the simplicity of planar reflections, Nerfren achieves the joint learning of reflection and diffuse components by simply aggregating predictions from two NeRF models together (reflection and diffuse) weighted by alpha-

[Figure 27]

Figure 3. Cornea geometry. The cornea can be modeled as an ellipsoid. The key fact that we exploit is that the cornea shape and size are largely consistent among adults, with similar eccentricity and curvature.

compositing. In this work, unlike prior works that focus on planar surface geometry, our object of interest (the human eye) has an inherently more complicated curved geometry, which necessitates us developing several modifications to the standard NeRF rendering workflow, which we will detail in the following sections.

Non-line-of-sight imaging. Non-line-of-sight (NLOS) imaging attempts to recover images of objects that are not directly visible from the camera’s position or are obstructed by an object in the line of sight. The principle behind NLOS imaging is that one can use light reflected off a visible relay surface to record information about an object outside of the line of sight. The NLOS literature largely falls under two categories: active and passive. Active NLOS imaging techniques involve using controlled light sources, such as lasers, and often rely on time-of-flight measurements to reconstruct the hidden scene. [47] introduced an ultra-fast imaging system that records light in flight, allowing the reconstruction of non-line-of-sight objects. [13, 19, 32] later presented various methods to improve the resolution of active NLOS imaging systems. NeRF has also been recently introduced to active NLOS imaging, enabling more accurate reconstructions and better handling of noise [8, 39]. Passive NLOS imaging, on the other hand, exploits natural or ambient light and does not require a controlled light source. [45] introduced the concept of accidental pinhole and pinspeck cameras, which involves using incidental or unintentional imaging elements in the environment to capture unique perspectives or resolve hidden scenes. [6, 38] analyzed shadow patterns and showed that these patterns contain sufficient information to reconstruct the shape of the hidden scene. [20] use reflections captured by a thermal camera to reconstruct the 3D body pose of non-line-of-sight humans. [44] recently presented Orca, which uses reflections from a glossy object observed in multi-view images to train a 3D NeRF for the surrounding environment. In this context, our paper can be regarded as a special case of passive NLOS scene reconstruction. We focus on a specific relay surface (the human eye) and introduce techniques tai-

𝐿𝑟𝑎𝑑𝑖𝑎𝑙

Input image (cropped)

Eq. 6

[Figure 28]

[Figure 29]

Cameraorigin:𝑂

[Figure 30]

Projeye(𝑂′) Φ

Camera ray direction: 𝑑⃗ Cornea hit: 𝑂′ Normal:𝑛 Reflection: 𝑑′

Hit point in eye coordinate system

Eye texture field

Estimated eye texture

[Figure 31]

+ 𝐿𝑟𝑒𝑐𝑜𝑛

⊙

[Figure 32]

[Figure 33]

RGB

Volumetric

Color

𝜃 Rendering

𝑥 , 𝑑′

Sampled point & direction

σ

Density

Cornea mask

Radiance field

Rendered image

- Figure 4. Joint optimization of radiance field and iris texture. Standard NeRF rendering uses rays starting from the camera origin O along a viewing direction d. In contrast, in our setup, we need to use rays that bounce off the cornea. The reflected ray origin O′ is where the initial camera ray intersects with the cornea, and the new ray direction d′ is the reflection of d across the cornea’s normal −→n . Consequently, the eye image we observe is a composition of the iris texture and the reflected scene. The composition hinders standard NeRF training due to the highly-detailed iris texture. To address this issue, alongside the radiance field θ, we train an eye texture field Φ whose input is the projection of O′ on the eye coordinate system in the given image (Eq. 5). The eye texture field is computed relative to the eye in the current image, while the radiance field takes 3D points in the world coordinates. The outputs from volumetric rendering with θ and texture estimation with Φ are composited together to reconstruct the cornea image. We apply a reconstruction loss Lrecon. We

further regularize the texture field Φ with a radial loss Lradial that encourages the estimated texture to be radially constant, reducing the absorption of scene regions into the eye texture.

lored for better information extraction from eye reflections. Unlike Orca, which relies on images captured with a moving camera while the “mirror” object is fixed, our method works for a stationary camera and uses the natural movement of the human eye “mirrors”, which is visualized in Figure 2.

little variation across different people. The bounds of the ellipsoidal section are determined by the distance from the apex to the base, labeled tb in Figure 3. From rL, the radius of the base of the ellipsoidal section, known to be approximately 5.5 mm in people, we can calculate tb as about 2.18 mm. To compute the normal at each point on the surface of the ellipsoid, we can take the gradient of Eq. 1 and get

### 3. Background: Eye Model

−→n (x,y,z) = ⟨2x,2y,2(1−e)z−2R⟩ (2)

The geometry of the human eye has been extensively studied [33]. The major components that are visible in the eye are: the sclera; which is the white region of the eye, and the cornea; which includes the iris and the pupil. The cornea is covered by a thin film of tear fluid, making it highly reflective. As noted by [30], since the cornea can act as a mirror, the combination of a camera and the cornea resembles a catadioptric system. In our work, we follow the eye model adopted by [30] for the geometry we assume for the eye.

To compute the depth of the cornea, we first assume a weak perspective projection model, which is valid because the diameter of the base is at most 11 mm and thus small compared to the depth. Next, notice that the projection of the cornea onto the image will be an ellipse. Let the major radius of the ellipse be rimg. Then under the projection model, the average depth of the cornea can be computed as

f rimg

depthavg = rL

. (3)

The eye is modeled as a section of an ellipsoid, as illustrated in Figure 3, which can be described using the equation

### 4. Method

(1−e)z2 −2Rz + r2 = 0 (1)

Radiance field from reflection. NeRF trains a parameterized radiance field through volumetric rendering. Each pixel color is computed by sampling the color and density along a ray using a parameterized MLP θ. In NeRF, the ray

where e is the eccentricity, R is the radius of the curvature at the apex, and r2 = x2 + y2. For an adult with healthy eyes, on average e is about 0.5 and R is about 7.8 mm, with very

associated with a pixel starts from the origin of that image’s camera, denoted by O, and the direction, denoted by −→d , is towards the projection of that pixel on the camera plane. By training the radiance field this way, we can recover a 3D reconstruction of the scene. However, in our setup, what we are interested in is to do a reconstruction of the scene reflected from the person’s eyes. In Figure 4, we illustrate how we use the rays reflected from the eye. The reflected ray starts with the origin where the camera ray intersects with the cornea at O′, and in the direction of the reflected ray −→d′ instead of using O and −→d . We compute the reflected ray explicitly using the standard reflection equation:

−→d′ = −→d −2 −→n ·

−→d −→n , (4)

where −→n is the normal at the hit point O′. Note we only need to compute the hit points and normals once before training for pixels associated with the cornea. Since we model the cornea geometry as an ellipsoid, we directly compute the hit points and normals using closed-form ellipsoid ray intersection formulas during the data processing step.

Texture decomposition Since the target images are the scene reflections off the cornea, training NeRF naively cause the output radiance field of mixing scene geometry and iris texture. To recover only the scene geometry in the radiance field, we jointly optimize a 2D field Φ to learn the eye texture. We assume that the iris texture remains the same across the different views while the person moves, while the scene reflections vary. For each pixel, the input to

- the 2D texture field is the pixel coordinate projected on the eye in the input image

projeye(y,x) =

y−cy rimg

,

x−cx rimg

(5)

where (cx,cy) is the coordinate of the center of the cornea, and rimg is the observed cornea radius. This parameterization enforces the texture field to naturally learn the invariant regions of the cornea, while the radiance field learns the 3D geometry of the scene.

However, when a part of the scene does not display considerable motion across the training views, it can be “absorbed” as part of the texture instead of the 3D scene. To resolve this issue, we propose a radial regularization that encourages radial symmetry of the recovered texture. We implement the loss by randomly sampling a rotation matrix R˜, and penalize the model on the color deviation between coordinate p and coordinate Rp˜ as follows:

Lradial (p) = λradial∥Φ(p)−Φ Rp ˜ ∥22 (6)

where λradial is the weight of the radial loss. While the iris is not perfectly radially constant, the simple radial loss

Table 1. Texture Decomposition Ablation. We show that using a neural field to decompose the iris texture from the reflection improves reconstruction performance.

Scene Method SSIM ↑ LPIPS ↓ Classroom w/o texture decomposition 0.40 0.72

w/ texture decomposition 0.42 0.62 Kitchen w/o texture decomposition 0.44 0.9

w/ texture decomposition 0.48 0.82

effectively removes the scene reflection while maintaining an accurate estimated texture.

Cornea pose optimization Due to the small cornea size in the captured images, the cornea pose and normals estimate inevitably have some errors. Training with the erroneous poses significantly affects the radiance field reconstruction’s quality. To alleviate the pose errors, we optimize the pose of each cornea independently. For each cornea, we optimize for a transformation matrix T = [R,t] ∈ SE(3), where R ∈ SO(3) and t ∈ R3 denote the rotation and translation, respectively. We optimize the cornea poses during training similar to [18,25,50].

### 5. Experiments

#### 5.1. Synthetic data evaluation

We generate synthetic data in Blender with eye models placed in the scene. In Figure 5 we show the scene we reconstruct using only the reflections from the eyes reflections. Since we cannot estimate the cornea eye perfectly in real life, we evaluate the robustness of our cornea pose optimization to the noise in the estimated cornea radius. To simulate the depth estimation errors we may encounter in real data, we corrupt the observed cornea rimg radius for each image by scaling the estimated radius with varying noise levels. In Figure 7 we show how our method’s performance varies for different noise levels. Note that as the amount of noise increases, our reconstruction with pose optimization is robust in terms of the reconstructed geometry and colors when compared to the reconstruction without pose optimization. This demonstrates that pose optimization is essential for our method to work in realistic scenarios where the initial ellipse fitting in the image to the projected cornea is imperfect. Furthermore, we show quantitative comparisons of our method with and without texture decomposition in Table 1. Our method performs better in terms of SSIM and LPIPS with texture decomposition than without. Notably, we do not compute PSNR because in our setting there is a drastic difference in lighting between the reflection and the scene itself.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

| |
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Cornea Cornea

RGB Accumulation RGB Accumulation

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

[Figure 48]

[Figure 49]

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

Cornea Cornea

RGB Accumulation RGB Accumulation

|[Figure 54]|
|---|

|[Figure 55]|
|---|

[Figure 56]

|[Figure 57]|
|---|

|[Figure 58]|
|---|

Cornea

RGB Accumulation

- Figure 5. Qualitative synthetic results. We show that our method can achieve reasonable reconstructions from challenging measurements in simulation. We demonstrate that our method can reconstruct the 3D geometry of the scene by visualizing the accumulation of the learned radiance fields with respect to the camera poses. The accumulation is defined as the integral of the density along the camera rays.

#### 5.2. Real-world experiments

We describe capturing and processing real-world images and demonstrate the effectiveness of our method on real captures.

Image capture. To maintain a realistic field of view, we capture images with a field of view that matches a standard portrait capture where the entire head is visible within the frame. We place area lights on the person’s sides to illumi-

nate the object of interest. Figure 9 illustrates the capture setup. We ask the person to move within the camera’s field of view and capture 5-15 frames per scene. We capture the images using a Sony RX IV camera and post-process the images using Adobe Lightroom to reduce the noise in the cornea’s reflection. Since the captured images have a deep dynamic range due to the scene illumination, we use 16-bit images in all our experiments to avoid losing information

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

|[Figure 64]|
|---|

| |
|---|

| |
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

| |
|---|

|[Figure 68]|
|---|

[Figure 69]

[Figure 70]

|[Figure 71]|
|---|

| |
|---|

|[Figure 72]|
|---|

| |
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

Sample captured frame Eye crop Novel view rendering

- Figure 6. Additional real results. We show that our method works in a variety of capture conditions, like smaller objects as in the small plant on the top row, and varying eye colors. We show that we can also reconstruct the observed object with a significantly smaller eye observations like in the bottom example.

from the observed reflections. We vary the illumination brightness and the reflected object size for a comprehensive evaluation. On average, the cornea only covers around 0.1% of each image, and the object of interest is reflected in a region of about 20x20 pixels and composited with the iris texture.

3D location using the average depth from Eq. 3 and the camera’s focal length, and also compute its surface normals using Eq. 2. To automate the process, we locate the eyes bounding boxes using Grounding Dino [21] and then use ELLSeg [15] to perform ellipse fitting for the iris. While the corneas are typically occluded, we only need the unoccluded regions, so we obtain a segmentation mask for the iris using Segment Anything [14].

##### 5.2.1 Data processing

We estimate the cornea’s center and radius on images to get an initial estimate of the cornea’s 3D location. Once we have the radius, we can directly approximate the cornea’s

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

With pose optimization

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

Ground truth

No pose optimization Low noise Medium noise High noise

- Figure 7. Synthetic pose optimization ablation. In simulation, the cornea pose optimization refines the noisy initial poses and results in clearer reconstruction.

[Figure 82]

[Figure 83]

[Figure 84]

(a) Eye localization with GroundingDINO (b) Ellipse fitting with ELLSeg (c) Iris segmentation with SAM

- Figure 8. Data processing pipeline. To compute the iris ellipse parameters, we first obtain eye bounding boxes using GroundingDINO [21] and then conduct ellipse fitting using ELLSeg [15]. Since we only want to use the visible regions of the cornea in our radiance field optimization, to handle occlusion, we generate a segmentation mask of the iris from the approximated cornea ellipse using Segment Anything [14].

##### 5.2.2 Results from real captures

Using our captured images, we show that our method enables the reconstruction of 3D scenes from real-world portrait captures, as shown in Figures 1 and 6, despite the cornea location and geometry estimate inaccuracies. In Figure 10, by ablating the cornea pose optimization and texture decomposition from our method, we demonstrate that cornea pose optimization and texture decomposition are necessary for successful 3D scene reconstructions. The initial pose estimate of the corneas is noisy because the blurriness of the boundary of the cornea makes it challenging to be localized precisely in the image, as shown in Figure 11. In Figure 10 we show the rendered radiance field with

and without the learned texture decomposition. We notice significantly more floaters when not explicitly modeling the texture. Furthermore, Figure 11 demonstrates that the radial regularization improves the quality of our reconstruction because, without it, the texture decomposition will absorb parts of the scene with low disparity among observed views. We note that for some eye colors, like green and blue, the 3D reconstruction is more difficult because the iris texture is brighter. One such example of a green iris texture is given in Figure 11, and to handle these cases, we can increase the amount of radial regularization.

[Figure 85]

[Figure 86]

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

Eye Crop

No Lradial With Lradial

- Figure 9. Capture setup. We illuminate the objects of interest with two area lights to ensure that sufficient amount of light is reflected off the eyes.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Full method

No pose optimization

No texture decomposition

- Figure 10. Ablating texture decomposition and cornea pose optimization. Top: not doing cornea pose estimation but still doing texture decomposition is not sufficient for 3D reconstruction at all. Bottom: not doing texture decomposition but still doing cornea pose estimation can recover some geometry and textures, but produces inferior visual quality.

Figure 11. Ablating radial regularization. Without radial regularization, the reconstructed iris texture contains parts of the scene with low disparity among observed views (e.g, here part of Kirby’s body). Our radial loss alleviates this issue by penalizing the radial color variation of the iris texture. Notice that our synthesized view for the field trained with the radial prior has more detail than the view for the field trained without.

### 6. Conclusions

By leveraging the subtle reflections of light off human eyes, we develop a method that can reconstruct the (nonline-of-sight) scene observed by a person using monocular image sequences captured at a fixed camera position. We demonstrate that naively training a radiance field on the observed reflections is insufficient due to several factors: 1) the inherent noise in cornea localization, 2) the complexity of iris textures, and 3) the low-resolution reflections captured in each image. To address these challenges, we introduce cornea pose optimization and iris texture decomposition during training, aided by a radial texture regularization loss based on the nature of the human eye iris. We showcase the effectiveness of our approach to real-world data. Unlike conventional methods of training a neural field that requires a moving camera, our method places the camera at a fixed viewpoint and relies solely on the user’s motion. With this work, we hope to inspire future explorations that leverage unexpected, accidental visual signals to reveal information about the world around us, broadening the horizons of 3D scene reconstruction.

#### 5.3. Limitations

Our work demonstrates the feasibility of reconstructing

- the 3D world only from eye reflections. Two major limitations remain. First, our current real-world results are from a “laboratory setup”, such as a zoom-in capture of a person’s face, area lights to illuminate the scene, and deliberate person’s movement. We believe more unconstrained settings remain challenging (e.g., video conferencing with natural head movement) due to lower sensor resolution, dynamic range, and motion blur. Second, our assumptions on the iris texture (e.g., constant texture, radially constant colors) may be too simplistic so our approach may break down with large eye rotations.

### References

- [1] Benjamin Attal, Jia-Bin Huang, Michael Zollh¨ofer, Johannes Kopf, and Changil Kim. Learning neural light fields with ray-space embedding. In CVPR, 2022. 3
- [2] Simon Baker and Shree K. Nayar. A theory of catadioptric image formation. ICCV, 1998. 2
- [3] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, 2021. 3

- [4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022. 3
- [5] Wenjing Bian, Zirui Wang, Kejie Li, Jiawang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. In CVPR, 2023. 3
- [6] Katherine L. Bouman, Vickie Ye, Adam B. Yedidia, Fr´edo Durand, Gregory W. Wornell, Antonio Torralba, and William T. Freeman. Turning corners into cameras: Principles and methods. In ICCV, 2017. 2, 3
- [7] Akshat Dave, Yongyi Zhao, and Ashok Veeraraghavan. Pandora: Polarization-aided neural decomposition of radiance. In ECCV, 2022. 3
- [8] Yuki Fujimura, Takahiro Kushida, Takuya Funatomi, and Yasuhiro Mukaigawa. Nlos-neus: Non-line-of-sight neural implicit surface. arXiv preprint arXiv:2303.12280, 2023. 3
- [9] Kun Gai, Zhenwei Shi, and Changshui Zhang. Blind separation of superimposed moving images using image statistics. IEEE TPAMI, 2012. 3
- [10] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In ICCV, 2021. 3
- [11] Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, and SongHai Zhang. Nerfren: Neural radiance fields with reflections. In CVPR, 2022. 3
- [12] Rob Jenkins and Christie Kerr. Identifiable images of bystanders extracted from corneal reflections. PloS one, 8(12):e83325, 2013. 1
- [13] Achuta Kadambi, Hang Zhao, Boxin Shi, and Ramesh Raskar. Occluded imaging with time-of-flight sensors. ACM Transactions on Graphics, 2016. 3
- [14] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 7, 8
- [15] Rakshit S Kothari, Aayush K Chaudhary, Reynold J Bailey, Jeff B Pelz, and Gabriel J Diaz. Ellseg: An ellipse segmentation framework for robust gaze tracking. IEEE Transactions on Visualization and Computer Graphics, 2020. 7, 8
- [16] Yu Li and Michael S. Brown. Single image layer separation using relative smoothness. In CVPR, 2014. 3
- [17] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In CVPR, 2021. 3
- [18] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In ICCV, 2021. 3, 5
- [19] David B Lindell, Gordon Wetzstein, and Matthew O’Toole. Wave-based non-line-of-sight imaging using fast fk migration. ACM Transactions on Graphics, 2019. 3
- [20] Ruoshi Liu and Carl Vondrick. Humans as light bulbs: 3d human reconstruction from thermal reflection. In CVPR, 2023. 2, 3
- [21] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded

- pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 7, 8
- [22] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In CVPR, 2023. 3
- [23] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Learning to see through obstructions. In CVPR, 2020. 3
- [24] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Learning to see through obstructions with layered decomposition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):8387– 8402, 2021. 3
- [25] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H. Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In CVPR, 2023. 3, 5
- [26] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2, 3
- [27] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics,

2022. 3

- [28] Ko Nishino, Peter N. Belhumeur, and Shree K. Nayar. Using eye reflections for face recognition under varying illumination. In ICCV, 2005. 1
- [29] Ko Nishino and Shree K. Nayar. Eyes for relighting. ACM SIGGRAPH, 2004. 2, 3
- [30] Ko Nishino and Shree K. Nayar. The World in an Eye. CVPR, 2004. 1, 2, 3, 4
- [31] Ko Nishino and Shree K. Nayar. Corneal Imaging System: Environment from Eyes. IJCV, 2006. 1, 3
- [32] Matthew O’Toole, David B. Lindell, and Gordon Wetzstein. Confocal non-line-of-sight imaging based on the light-cone transform. Nature, 555, 2018. 3
- [33] Anna Pandolfi and Federico Manganiello. A model for the human cornea: Constitutive formulation and numerical analysis. Biomechanics and Modeling in Mechanobiology, 5(4),

2006. 4

- [34] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Steven M. Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In ICCV, 2021. 3
- [35] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M. Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. ACM TOG (Proc. SIGGRAPH), 2021. 3
- [36] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In CVPR, 2021. 3
- [37] Abhijith Punnappurath and M. S. Brown. Reflection removal using a dual-pixel sensor. In CVPR, 2019. 3

- [38] Prafull Sharma, Miika Aittala, Yoav Y. Schechner, Antonio Torralba, Gregory W. Wornell, William T. Freeman, and Fr´edo Durand. What you can learn by staring at a blank wall. In ICCV, 2021. 2, 3
- [39] Siyuan Shen, Zi Wang, Ping Liu, Zhengqing Pan, Ruiqian Li, Tian Gao, Shiying Li, and Jingyi Yu. Non-line-of-sight imaging via neural transient fields. IEEE TPAMI, 2021. 3
- [40] Sudipta N. Sinha, Johannes Kopf, Michael Goesele, Daniel Scharstein, and Richard Szeliski. Image-based rendering for scenes with reflections. ACM Transactions on Graphics,

2012. 3

- [41] Rahul Swaminathan, Michael D. Grossberg, and Shree K. Nayar. A perspective on distortions. In CVPR, 2003. 2
- [42] Kentaro Takemura, Tomohisa Yamakawa, Jun Takamatsu, and Tsukasa Ogasawara. Estimation of a focused object using a corneal surface image for eye-based interaction. Journal of eye movement research, 7(3):1–9, 2014. 2
- [43] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and Angjoo Kanazawa. Nerfstudio: A modular framework for neural radiance field development. arXiv preprint arXiv:2302.04264, 2023. 3
- [44] Kushagra Tiwary, Akshat Dave, Nikhil Behari, Tzofi Klinghoffer, Ashok Veeraraghavan, and Ramesh Raskar. Orca: Glossy objects as radiance-field cameras. In CVPR,

2023. 3

- [45] Antonio Torralba and William T. Freeman. Accidental pinhole and pinspeck cameras: Revealing the scene outside the picture. In CVPR, 2012. 2, 3
- [46] Norimichi Tsumura, Minh Dang, and Yoichi Miyake. Estimating the directions to light sources using images of eye for reconstructing 3d human face. International Conference on Communications in Computing, 2003. 3
- [47] Andreas Velten, Thomas Willwacher, Otkrist Gupta, Ashok Veeraraghavan, Moungi G Bawendi, and Ramesh Raskar. Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging. Nature Communications,

2012. 3

- [48] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd E. Zickler, Jonathan T. Barron, and Pratul P. Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. CVPR, 2022. 3
- [49] Renjie Wan, Boxin Shi, Ah-Hwee Tan, and Alex Chichung Kot. Depth of field guided reflection removal. In IEEE International Conference on Image Processing, 2016. 3
- [50] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. NeRF−−: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021. 3, 5
- [51] Tianfan Xue, Michael Rubinstein, Ce Liu, and William T. Freeman. A computational approach for obstruction-free photography. ACM Transactions on Graphics, 2015. 3
- [52] Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2021. 3

- [53] Xiang Zhang, Kaori Ikematsu, Kunihiro Kato, and Yuta Sugiura. Reflectouch: Detecting grasp posture of smartphone using corneal reflection images. In CHI Conference on Human Factors in Computing Systems, 2022. 2
- [54] Xiuming Zhang, Pratul P. Srinivasan, Boyang Deng, Paul E. Debevec, William T. Freeman, and Jonathan T. Barron. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM Transactions on Graphics,

2021. 3

- [55] Xuaner Cecilia Zhang, Ren Ng, and Qifeng Chen. Single image reflection separation with perceptual losses. In CVPR,

2018. 3

