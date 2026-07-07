# arXiv:2506.21520v2[cs.CV]11Dec2025

## MADrive: Memory-Augmented Driving Scene Modeling

Polina Karpikova* † Yandex

Kirill Struminsky∗ Yandex Research, HSE University Ruslan Musaev Yandex

Daniil Selikhanovych∗ Yandex Research, HSE University

Maria Golitsyna Yandex

Dmitry Baranchuk Yandex Research

### Abstract

Recent advances in scene reconstruction have pushed toward highly realistic modeling of autonomous driving (AD) environments using 3D Gaussian splatting. However, the resulting reconstructions remain closely tied to the original observations and struggle to support photorealistic synthesis of significantly altered or novel driving scenarios. This work introduces MADRIVE, a memory-augmented reconstruction framework designed to extend the capabilities of existing scene reconstruction methods by replacing observed vehicles with visually similar 3D assets retrieved from a large-scale external memory bank. Specifically, we release MAD-CARS, a curated dataset of ∼70K 360° car videos captured in the wild and present a retrieval module that finds the most similar car instances in the memory bank, reconstructs the corresponding 3D assets from video, and integrates them into the target scene through orientation alignment and relighting. The resulting replacements provide complete multi-view representations of vehicles in the scene, enabling photorealistic synthesis of substantially altered configurations, as demonstrated in our experiments.

Project page:

https://yandex-research.github.io/madrive/

### 1. Introduction

Modern autonomous driving (AD) systems heavily rely on computer vision and machine learning models trained on large-scale and diverse datasets [8, 9, 20, 49, 62]. However, collecting and annotating such data in the real world is expensive, time-consuming, and often limited by safety and practicality. Driving simulators [57, 67, 73] offer an alternative by generating realistic novel views and rare, safetycritical scenarios that are difficult to record in the real world. Realistic simulation helps reduce the domain gap between synthetic and real data and improves model robustness. Moreover, controllable reenactments of driving scenes also

*Equal contribution. †Email: p.karpikova@yandex.ru

allow systematic testing of perception and planning models by reproducing failure cases under different conditions. In this work, we reconstruct driving sequences captured by autonomous vehicle to enable such controllable, photorealistic reenactments for safety testing and data augmentation.

Recent progress in multi-view reconstruction and novel view synthesis [31, 34, 70] has enabled photorealistic reconstruction of complex real-world scenes, forming a strong foundation for realistic driving simulators [73]. Unlike game engine–based environments, such methods preserve the visual properties of real data, reducing domain shift. Modern driving scene reconstruction methods [33, 64, 74] can accurately reproduce observed trajectories and support small viewpoint changes. However, their quality drops when extrapolating far from the observed data, as unseen regions lack geometric and photometric information. As a result, they cannot reliably simulate large trajectory changes such as U-turns or parking maneuvers, where missing observations cause visible artifacts and incomplete geometry.

To overcome these limitations, we replace dynamic objects with high-quality external assets. Traditional computer graphics assets are manually created by artists, making it costly to build a large and diverse library. Instead, we reconstruct assets directly from real-world data. We introduce MAD-CARS, Multi-view Auto Dataset — a largescale collection of 360◦ vehicle video videos. The dataset contains about 70,000 car instances covering various makes, models, colors, providing diverse and realistic inputs for asset reconstruction. This approach greatly expands the diversity of assets compared to existing public car datasets.

To reconstruct realistic and reusable assets, we develop a car reconstruction pipeline tailored for our in-the-wild captures. A key challenge is adapting models captured under unknown and scene-specific lighting to new environments. To solve this, we propose a relightable variant of Gaussian splatting that supports physically based rendering under novel illumination. Our optimization scheme uses a generative model [60] to simulate multi-lighting supervision during reconstruction, effectively separating material properties from lighting. This enables high-quality relight-

Observed frames Future frames Timeline

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Training frame Future frame, ground truth Future frame, MADrive Alternative scenario, MADrive

- Figure 1. MADRIVE reconstructs a 3D driving scene from training frames (Left) and replaces partially observed vehicles in the scene with realistically reconstructed counterparts retrieved from MAD-CARS, our novel multi-view auto dataset. MADRIVE enables high-fidelity modeling of future scene views (Middle-left vs. Middle-right) and supports simulation of alternative scenarios, advancing novel-view synthesis in dynamic environments (Right).

ing and seamless integration of assets into new scenes.

Building on these components, we present MADRIVE, a framework that replicates driving sequences by replacing observed vehicles with visually similar reconstructed 3D assets retrieved from a large external dataset. Using high-fidelity, relightable assets, MADRIVE preserves realistic appearance even under large trajectory changes. The framework enables controllable reenactments by rendering future or alternative frames beyond the original sequence. Quantitative results show that pre-trained perception models perform similarly on our rendered frames and real holdout data, confirming the realism and consistency of our reconstructions for downstream AD tasks.

In summary, our contributions are threefold: (1) a largescale dataset of in-the-wild 360◦ vehicle captures for diverse asset reconstruction, (2) a relightable Gaussian splatting pipeline that separates object appearance from illumination for realistic rendering, and (3) an integrated framework that reconstructs, reenacts, and validates driving scenes with controllable trajectories, achieving photorealism and consistency with real-world perception results.

### 2. Related Work

Dynamic Urban Scene Reconstruction. Recent dynamic 3D scene reconstruction works adopt 3D Gaussian splatting [32] as an efficient and expressive representation [11, 59, 68, 69]. Several approaches, including StreetGS [64], AutoSplat [33], and HUGS [74], apply this representation to driving scenes by decomposing them into static backgrounds and dynamic vehicles placed with tracked 3D bounding boxes. HUGS incorporates optical flow and semantic segmentation to guide optimization and adds realistic shadow modeling, while AutoSplat improves vehicle reconstruction by exploiting bilateral symmetry and better initialization from image-to-3D priors [44]. Other works [12, 75] introduce dynamic Gaussian graphs to han-

dle multiple moving objects of different nature. We refer the reader to Appendix A for a review of earlier NeRFbased scene representation methods. While these methods improve the fidelity of reconstructed urban scenes, they rely solely on observational data to model dynamic objects, making it difficult to capture complete vehicle geometry and appearance under sparse or occluded views.

3D Car Datasets. Several public datasets provide 3D car assets. Early collections such as SRN-Car [10] and Objaverse-Car [15] consist of CAD models that differ notably from real vehicles in texture realism and geometric detail. More recent efforts [16, 71] have focused on realcaptured 3D car datasets. MVMC [71] includes 576 cars, each with an average of 10 views. 3DRealCar [16] provides 2,500 car instances, each with ∼200 dense highresolution RGB-D views. In contrast, MAD-CARS includes ∼70,000 360° car videos at a comparable resolution and average number of views as 3DRealCar, offering substantially greater generalization and diversity.

NVS with External 3D Car Assets. HUGSim [73] builds a closed-loop AD simulator by inserting 3D car models from 3DRealCar [16] into reconstructed scenes. We instead replace only the vehicles observed in the recording with retrieved matches, producing a close replica of the captured scene for reenactment.

Several approaches leverage CAD models for scene representation [2, 17, 22, 54, 57, 58], though such models often lack photorealistic textures and accurate geometry. To improve realism, some methods perform geometry refinement [17, 54, 57], while UrbanCAD [38] retrieves visually similar CAD models and refines their textures and illumination to better match the scene. However, the obtained models still have a noticeable gap in realism and correspondence to actual cars. In contrast, MADRIVE retrieves real car instances from a large-scale dataset spanning diverse brands,

materials, and lighting conditions, helping to narrow this realism gap.

Relighting. Scene reconstruction methods based on radiance fields jointly recover geometry and outgoing radiance from multi-view inputs. Since radiance depends on scene illumination, object appearance changes under different lighting conditions.gt The outgoing radiance follows the rendering equation [28], and exact relighting would require full light transport modeling via ray tracing. Several NeRFbased methods extend radiance fields to simulate light transport and relighting [26, 50, 55, 72], but they demand substantial computational resources and are not suitable for real-time simulation. Although several recent works explore efficient ray tracing for Gaussian splats [7, 21, 41, 63], relighting remains outside their scope.

Several methods instead approximate light transport using simplified shading models compatible with rasterization. LumiGauss [29] performs splat-based relighting with spherical harmonics [46], but requires multi-illumination data and is limited to diffuse surfaces. GaussianShader [25] employs a split-sum approximation [30] to enhance specular reflections during reconstruction. R3DG [19] improves relighting quality by jointly decomposing materials and illumination from single-lighting captures using an enhanced illumination model. While our approach also employs a physically based shading model, it augments training with synthetic multi-illumination data to improve relighting.

To estimate the environmental map for object insertion, DiPIR [35] employs a generative model and a gradientbased procedure. In constrast, we estimate the environmental map from training frames using DiffusionLight [45] without additional costly optimization procedures. Our approach is modular and can be easily adapted to incorporate alternative environmental map estimation methods such as [36] if needed.

### 3. Method

In this section, we describe MADRIVE that replaces the vehicles in the scene with visually similar, fully-observed 3D car assets, thereby enabling the prediction of future vehicle appearances following sharp turns or other complex maneuvers. The overview of our method is presented in Figure 2.

#### 3.1. Scene Decomposition and Reconstruction Overview

Following [64], we decompose each driving scene into static and dynamic components. The static part represents the background—ground, surroundings, and sky—while the dynamic part contains all moving vehicles. The static component can be reliably reconstructed from the egovehicle’s motion and depth sensor data, which provide sufficient parallax to recover 3D structure.

We adapt the approach of Street Gaussians [33] to model the static scene. The surroundings are parameterized using 3D Gaussian splats [31], the ground is represented by horizontal 2D splats, and the sky is placed at an infinite distance and blended during rendering to avoid depth ambiguities.

Dynamic elements include all moving vehicles; however, we treat cars marked as stationary in the metadata as part of the static background for simplicity. Modeling dynamic objects poses two main challenges: handling compound motion and reconstructing objects from incomplete observations (e.g., when only one side of a vehicle is visible). Following prior work [33, 64], we approximate each observed vehicle as a set of static Gaussian splats within its moving 3D bounding box, effectively capturing its motion during training. Both static and dynamic components are initialized from LiDAR points and jointly optimized using a photometric loss.

During inference, we reuse the static part of the scene. At the same time, we replace moving vehicles with 3D car models extracted from a bank of cars using the retrievalbased approach, described in the next section. This substitution allows obtaining high-quality renders for configurations diverging from the ones observed during training. The computational bottlenecks of our method are static scene reconstruction (≈3 hours on a single GPU) and per-asset reconstruction (≈30 minutes). Retrieval, asset insertion, relighting, and rendering add only minor overhead and run efficiently in comparison.

#### 3.2. Database Retrieval

Our goal is to replace observed vehicles in driving sequences with visually similar 3D assets retrieved from an external database. This subsection outlines our retrieval pipeline and the corresponding database construction.

Retrieval Query Information. For each driving sequence, we project the 3D bounding box of every detected vehicle onto the image plane to obtain a segmentation mask. After discarding small and overlapping masks, we extract cropped images centered on individual cars. Each crop is embedded using SigLIP2 [52], while the vehicle color is estimated with Qwen2.5-VL [65]. The color cue complements the embedding features, which primarily capture brand and type information, improving retrieval accuracy.

Database Collection and Statistics We introduce MADCARS, a large-scale collection of in-the-wild multi-view car videos sourced from online car-sale advertisements. The dataset contains approximately ∼70,000 car instances, each with about ∼85 frames at 1920×1080 resolution. It spans roughly 150 brands and covers diverse colors, car types, and various lighting conditions. Figure 3 summa-

Background Reconstruction Environmental Map Estimate

[Figure 9]

[Figure 10]

New envlights (fixed) Synthetic Data

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

img2img render enhancement (Difix)

[Figure 17]

| | |
|---|---|
| | |

[Figure 18]

[Figure 19]

Splat parameters

Car relighting Background Model

(shape, opacity, albedo, roughness & metallicity)

[Figure 20]

L = Lrgb + Lopacity + Lnormal

[Figure 21]

[Figure 22]

Enc

[Figure 23]

[Figure 24]

[Figure 25]

closest

Scene envlights (learnable)

Mask Estimator

Normal Estimator

[Figure 26]

Photographs from the database

Enc

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

...

i. Retrieval ii. Car Reconstruction

iii. Scene Composition

- Figure 2. MADRIVE Overview. Given an input frame sequence, our retrieval scheme finds similar vehicles in an external database (Left). The 3D reconstruction pipeline then produces detailed vehicle models from the retrieved videos. The vehicles are represented using relightable 2D Gaussian splats. To enable relighting, we generate synthetic novel views under multiple illumination conditions. Opacity masks are applied to remove background splats, and the model geometry is regularized using external normal maps. (Middle). The reconstructed vehicles are adapted to the scene’s lighting and composed with the background to produce the overall scene representation (Right).

rizes the distributions of color, type, and illumination. Each instance includes metadata describing car attributes.

To ensure high-quality reconstruction, we curate the dataset by filtering out frames and instances that degrade multi-view consistency. Specifically, we remove lowquality or overly dark frames using CLIP-IQA [56], and employ Qwen2.5-VL [65] to detect persistent occlusions that obscure parts of the vehicle across most views—such as nearby grass, bushes, or fences—as well as to filter out interior and obstructed shots. Further data-collection details are provided in Appendix B.

To retrieve a matching asset, we first pre-select candidates with similar color and then identify the closest match in the embedding space. The selected instance is reconstructed into a 3D model using its associated multi-view image set. The reconstruction pipeline is detailed in the following section.

- 3.3. Car Reconstruction Deatails

transparency α ∈ R, and two scale parameters σx,σy ∈ R. Unlike 3D splats, 2D splats have well-defined surface normals n = n(R), which are essential for surface relighting.

To disentangle scene lighting from surface materials, we adopt the lighting model from [42] for each splat. The model assumes distant illumination with incident radiance Li(ωi) and defines the outgoing radiance in direction ωo according to the rendering equation [28]:

Li(ωi)f(ωi,ωo)(ωi · n)dωi, (1)

L(ωo) =

Ω

where f(ωi,ωo) is the surface BSDF and the integration is taken over the hemisphere Ω around the surface point. The environment lighting Li is parameterized as a highresolution cubemap. Following [42], we parameterize each splat’s BSDF using the Cook–Torrance shading model [14], with appearance defined by albedo c ∈ R3, roughness r ∈ R, and metallicity m ∈ R.

Finally, to avoid the cost of directly evaluating Eq. 1, we employ the differentiable split-sum approximation from [42], which allows us to jointly infer incident radiance and splat BSDF parameters during optimization.

Relightable Car Models. We begin by specifying the representation used to model vehicles. By default, Gaussian splatting models the entangled radiance observed in the training frames, implicitly coupling surface reflectance and illumination. In our setup, however, we need to explicitly separate lighting and material effects to enable model insertion into environments with different illumination. To this end, we adopt a relighting strategy based on physicallybased shading [6].

Reconstruction Algorithm. Next, we specify the details of the reconstruction algorithm used for the representation above.

For a rendered frame Ii and the ground truth frame Iˆi, our objective consists of image-based loss Lrgb = L1(Ii,Iˆi) + LSSIM(Ii,Iˆi) along with several regularizers. To exclude unnecessary background objects from the model, we generate masks Mˆi(x,y) =

We use a two-dimensional modification of Gaussian splats [24], which approximates the 3D model with a collection of flat Gaussian splats. Each splat is parameterized by its location µ ∈ R3, orientation matrix R ∈ SO(3),

Figure 3. MAD-CARS Analysis. Memory-bank statistics on colors (Left), car types (Middle) and lighting conditions (Right).

[Iˆi(x,y) is part of a car] with Mask2Former [13] to indicate pixels that belong to the model. Our opacity loss promotes high transparency outside of car pixels Lopacity =

computing instance segmentation [27] and discarding those that lie behind the training cameras or project outside the car mask in most views, using a soft threshold to handle segmentation errors. The cleaned point cloud is then oriented along its principal components, and the front direction is estimated using an orientation model [48].

x,y(1 − Mˆi) · Ti, where Ti is the transparency map of the rendered frame. In our model, proper relighting requires accurate surface normals, so we additionally estimate normal maps Nˆi = n(Iˆi) with a NormalCrafter model [4] and use the estimates to regularize Gaussian orientations. For the rendered normal maps Ni, the regularizer is Lnormal = x,y Mˆi · (1 − NiTNˆi). The resulting objective is

#### 3.4. Car Insertion and Relighting

After reconstructing the vehicle models with a standardized orientation, we integrate them into the captured scene. We first estimate each car’s orientation by aligning the asset’s bounding box with the detected vehicle bounding box in the scene. However, directly matching the bounding boxes in scale and position often results in noticeable misalignment. To obtain accurate placement, we refine the transformation using the Iterative Closest Point (ICP) algorithm [3], aligning the reconstructed asset to the corresponding LiDAR point cloud and deriving the final scale and location from this alignment.

Lgt(Ii,Iˆi) = Lrgb(Ii,Iˆi) + λopacityLopacity(Ii,Iˆi)+ λnormalLnormal(Ii,Iˆi).

(2)

To address the limited lighting variability in our real captures, we additionally introduce a generative augmentation strategy that extends the Difix framework [60] beyond its original purpose. While Difix was originally proposed to enhance the photorealism of rendered novel views, we repurpose it to simulate appearance under diverse illumination conditions, effectively approximating multiillumination supervision.

To ensure consistent lighting between the inserted assets and the reconstructed scene, we estimate the scene’s environmental map to compute the outgoing radiance of each asset in Eq. 1. Since the Waymo dataset [49] lacks full 360◦ camera coverage and is captured in low dynamic range, we approximate the full high dynamic range environmental map with DiffusionLight [45] using the last training frame from the frontal camera. We further scale the estimated environmental map to minimize tone discrepancies between the last training frame and the rendered image. Finally, to enhance visual realism, we add a shadow beneath the car, modeled as a black semi-transparent plane composed of 2D splats positioned right below the wheels.

Concretely, we render the reconstructed model from random novel viewpoints and relight each render using randomly sampled environmental maps. These low-quality renders are then refined with Difix to produce synthetic enhanced frames I˜i that emulate novel realistic lighting. By pairing each I˜i with a physically based render Ii under the same illumination, we compute the objective in Eq. 2, using the α-mask of I˜i in Lopacity and omitting Lnormal term.

This augmentation effectively disentangles illumination and material properties, allowing our model to generalize across lighting conditions that were never observed in the training data. We mix synthetic and real frames in equal proportion, introducing synthetic samples after the initial 10k iterations and refreshing them every 2.5k steps for 20k additional iterations.

### 4. Experiments

In the following, we present the evaluation of MADRIVE. Section 4.1 describes the experimental setup, followed by the main results in Section 4.2. Finally, Section 4.3 analyzes the retrieval, car reconstruction components, and relighting in the scene.

After the Gaussian splatting reconstruction, we apply several postprocessing steps to obtain the final car asset. We remove stray splats that do not belong to the depicted car by

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Seen

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Future

Ground Truth Street Gaussians AutoSplat HUGS MADrive

- Figure 4. Qualitative comparison of MADRIVE with non-retrieval-based driving scene reconstruction methods. Reconstruction of the training views (Top). Reconstruction of the hold-out (future) views (Bottom).

#### 4.2. Main Experiments

#### 4.1. Evaluation Setup

Scene Reconstruction Dataset. We reconstruct driving scenes from the Waymo Open Motion Dataset [18]. We select 12 challenging sequences featuring multiple vehicles, complex driving maneuvers, and diverse lighting conditions. Each scene is manually segmented into training and evaluation clips. In our experiments, we simultaneously use videos from frontal and two side cameras to capture a wide field of view and track cars moving across the scene. More evaluation setup details are provided in Appendix C.

Qualitative Evaluation. First, we provide visual scene reconstruction results for qualitative analysis. In Figure 4, we compare renderings on the training and hold-out frames. While SG, AutoSplat, and HUGS reproduce the training frames with high accuracy, their reconstructions tend to break down under novel viewpoints, causing vehicles to fall apart or distort. In contrast, our method, though slightly less precise on training frames, demonstrates substantially greater robustness to unseen configurations. Additional visual examples are shown in Figures 9, 10. We also provide the visualizations with modified trajectories in Figure 11.

Scene Extrapolation with Novel View Synthesis. For our evaluation, we selected driving scenes involving Uturns, intersection crossings, and parking departures common accident scenarios that expose vehicles from diverse viewpoints and pose significant challenges for reconstruction. Each sequence was manually divided into training and testing subsets at the midpoint of the maneuver. We use the whole sequence to reconstruct the background and then remove the cars using the annotated bounding boxes in the Waymo dataset. Car reconstruction relies solely on the training portion, while the remaining frames are reserved for evaluating scene reconstruction quality.

Table 1. Comparison in terms of tracking and segmentation metrics.

Model MOTA ↑ MOTP ↓ IDF1 ↑ Segmentation IoU ↑

Street-GS [64] 0.654 0.105 0.776 0.556 HUGS [74] 0.556 0.221 0.699 0.333 AutoSplat* [33] 0.589 0.154 0.716 0.489

MADRIVE (Ours) 0.841 0.138 0.913 0.818

*Denotes our reimplementation.

Quantitative Evaluation. In our main experiments, we evaluate tracking and segmentation performance on synthesized test frames. Specifically, we apply state-of-theart tracking and segmentation models to both synthesized and ground truth frames and compare their outputs using established metrics for each task. For tracking, we use BotSort [1] with a YOLOv8n backbone and report Multiple Object Tracking Accuracy (MOTA↑), Multiple Object Tracking Precision (MOTP↓), and the identity F1 score (IDF1) [39]. For segmentation, we compute the average intersection-over-union (IoU) using instance segmentation masks predicted by Mask2Former [13].

Our goal is to generate realistic novel views by extrapolating beyond the observed data. Specifically, we insert the reconstructed car models into the background according to location and orientation specified by the bounding boxes on the holdout sequence. This setup intentionally tests the model under configurations that differ from the training views, while ensuring no test data leaks into the reconstruction process.

Baselines. We compare MADRIVE with the scene reconstruction Gaussian splatting-based methods that were previously considered for novel view synthesis: StreetGaussians (SG) [64], AutoSplat [33] (our implementation), and HUGS [74]. Details on training and evaluation of baselines are given in Appendix E.

Table 1 compares MADRIVE against baseline approaches. MADRIVE achieves substantially higher scores in two tracking metrics (MOTA and IDF1) and in the segmentation metric (IoU), confirming improved scene consis-

[Figure 43]

Reference Rotated Reference Rotated Reference Rotated Reference Rotated

[Figure 44]

[Figure 45]

[Figure 46]

GTMADriveUrbanCADDiffSplatAmodal3R

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

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

- Figure 5. Qualitative comparison of reconstructed vehicle assets based on queries from KITTI-360 (reference and rotated viewpoints). MADRIVE, trained on the MAD-CARS dataset, produces 3D assets that better preserve shape consistency and visual realism across views compared to CAD-based and image-to-3D generative baselines.

tency and object reconstruction quality. This observation is also supported by the visual examples provided in Figure 4.

limitation has been observed with non-vision-language encoders such as DINOv2 [43]. Additional results in Appendix G show that applying a color-based pre-filtering improves color consistency between the retrieved and target vehicles.

The lower MOTP score of MADRIVE compared to Street-GS arises from Street-GS’s better initial alignment in early test frames; however, later frames—where Street-GS tracking often fails—are excluded from the MOTP calculation. Per-scene results for all 12 sequences are provided in Appendix D, and the choice of reference masks used in the evaluation protocol is discussed in Appendix F.

Table 2. Retrieval performance w/o color filtering in terms of accuracy on the car brand, model, color and type and the distance to the closest instance for the MAD-CARS and 3DRealCar [16] datasets. MAD-CARS enables more accurate retrieval of cars across all attributes.

#### 4.3. Ablation Study

Dataset Brand ↑ Model ↑ Color ↑ Car Type ↑ Distance ↓

Retrieval. Here, we evaluate the retrieval module in isolation to assess how accurately the retrieved cars correspond to the original vehicles in the scene.

3DRealCars 0.626 0.503 0.508 0.888 0.502 MAD-CARS 0.750 0.663 0.533 0.913 0.445

We compare retrieval performance on MAD-CARS against 3DRealCars [16], a high-quality publicly available dataset containing 2,500 car assets. To evaluate retrieval quality, we first compute the average L2 distance between each car image from the driving scene and its nearest neighbor in the memory bank, using SigLIP2 [52] as the image feature extractor. Then, we provide accuracy obtained with the Qwen2.5-VL-32B-Instruct model, which compares cars based on brand, model, color, and car type. For a fair comparison, we do not use the color filtering in this experiment.

Car reconstruction. We provide a qualitative comparison with other car reconstruction approaches in Figure 5, where we visualized reconstruction alternatives. Given a query frame from the KITTI dataset, we compared the proposed approach with three alternatives: matching with a car model from a CAD dataset (UrbanCAD, [38]), and running a cutting edge image-to-3D models (DiffSplat [37], Amodal3R [61]). Even though the latter closely matches the query frame, the second view indicates a subpar geometry recovery. Compared to other methods, we see that the diversity of our dataset allows MADRIVE to obtain models that closely match the query frame in terms of appearance (e.g., color, shape) and realism.

Table 2 reports retrieval accuracy across different attributes, along with the average L2 distance to the closest instance. Cars retrieved from MAD-CARS more closely match the driving scene vehicles, which we attribute to the dataset’s larger scale and diversity.

We also quantitatively compare the realism of reconstructed cars for the retrieval-based approach of MADRIVE and the recent image-to-3D generative model, Amodal3R.

Notably, Table 2 highlights that retrieval based solely on feature embeddings often ignores car color, despite its importance for accurate vehicle replacement. A similar

For this, we collected random crops using a hold-out

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

(a) Lrgb (b) +Lopacity (c) +Lnormal (d) + Difix

(a) Lrgb (b) +Lopacity (c) +Lnormal (d) + Difix

(e) + Difix (varying light)

(e) + Difix (varying light)

(fixed light)

(fixed light)

- Figure 6. Qualitative ablation of reconstruction components, progressively adding each regularizer to the previous configuration. Starting without regularization (a), the reconstruction shows shape and texture artifacts with uneven edges. Adding opacity regularization (b) improves edge quality, while normal regularization (c) enhances surface smoothness. Incorporating synthetic data (d) further refines the results, and training with synthetic frames under varying lighting (e) helps disentangle illumination from object color, producing cleaner albedo and overall more consistent reconstructions.

set of real cars from the Waymo dataset and reconstructed car models with MADRIVE and Amodal3R. Then we rendered reconstructed 3D models from 360 degrees. In Table 3, we provide FID [23] and KID [5] between the set of renderings for the reconstructed models with MADRIVE and Amodal3R and the set of real images of hold-out cars from MAD-CARS. The results support our claim that the retrieval-augmented approach yields cars with better realism. We provide the details in Appendix H.

ing illumination, reducing visual inconsistencies and making the inserted vehicles appear more naturally integrated into the scene.

[Figure 119]

[Figure 120]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 121]

[Figure 122]

| |
|---|

| |
|---|

Table 3. Quantitative evaluation of the realism for reconstructed car models.

[Figure 123]

[Figure 124]

| |
|---|

| |
|---|

| |FID ↓ KID×103 ↓<br><br>|
|---|---|
|Amodal3R MADRIVE<br><br>|81.65 51.91 62.64 39.40|

W/o relighting W/ relighting

Figure 7. Relighting ablation. Rendered hold-out frames without (Left) and with (Right) relighting.

We further ablate the components of the proposed reconstruction algorithm. Figure 6 shows the recovered geometry and albedo for several cars. Reconstruction without regularization (a) produces noticeable artifacts in both shape and texture, leading to uneven edges after background removal. Introducing opacity regularization to suppress the background (b) improves edge quality. Adding normal regularization (c) further enhances surface smoothness and consistency. Incorporating synthetic views enhanced with Difix (d) improves reconstruction quality in some cases. Finally, training with synthetic frames rendered under varying lighting conditions (e) helps disentangle scene illumination from object color, resulting in more uniform albedo and overall cleaner reconstructions.

Relighting. We performed a qualitative comparison to evaluate the impact of the proposed relighting scheme. For several scenes, we reconstructed frames both with and without the relighting module. As shown in Figure 7, the relighting module effectively adjusts model colors to the surround-

### 5. Conclusion

This work presents MADRIVE, a novel driving scene reconstruction approach specifically designed to model significantly altered vehicle positions. Powered by MAD-CARS, our large-scale multi-view car dataset, MADRIVE replaces dynamic vehicles in a scene with similar car instances from the database. We believe that MADRIVE could make a step towards modeling multiple potential outcomes for analyzing an autonomous driving system’s behavior in safetycritical situations.

Although our generated future frames look promising, they still differ from the ground truth, as discussed in Appendix K. At the moment, we only use the future frame to create the retrieval query, but it could also help adapt the inserted asset to better match the observed vehicle in the future work. Another direction is to include more advanced relighting methods that better capture reflections and light interactions, making the results more realistic under new lighting conditions.

### References

- [1] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Botsort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022. 6
- [2] Armen Avetisyan, Manuel Dahnert, Angela Dai, Manolis Savva, Angel X Chang, and Matthias Nießner. Scan2cad: Learning cad model alignment in rgb-d scans. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 2614–2623, 2019. 2
- [3] Paul Besl and H.D. McKay. A method for registration of 3-d shapes. ieee trans pattern anal mach intell. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 14:239– 256, 1992. 5
- [4] Yanrui Bin, Wenbo Hu, Haoyuan Wang, Xinya Chen, and Bing Wang. Normalcrafter: Learning temporally consistent normals from video diffusion priors. arXiv preprint arXiv:2504.11427, 2025. 5
- [5] Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In International Conference on Learning Representations, 2018. 8, 15
- [6] Brent Burley and Walt Disney Animation Studios. Physically-based shading at disney. In Acm siggraph, pages 1–7. vol. 2012, 2012. 4
- [7] Krzysztof Byrski, Marcin Mazur, Jacek Tabor, Tadeusz Dziarmaga, Marcin Ka˛dziołka, Dawid Baran, and Przemysław Spurek. Raysplats: Ray tracing based gaussian splatting. arXiv preprint arXiv:2501.19196, 2025. 3
- [8] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 1
- [9] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 1
- [10] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015. 2
- [11] Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, and Li Zhang. Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering. arXiv preprint arXiv:2311.18561, 2023. 2
- [12] Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco Pavone, Li Song, and Yue Wang. Omnire: Omni urban scene reconstruction. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [13] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1280–1289, 2022. 5, 6

- [14] Robert L Cook and Kenneth E. Torrance. A reflectance model for computer graphics. ACM Transactions on Graphics (ToG), 1(1):7–24, 1982. 4
- [15] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023. 2
- [16] Xiaobiao Du, Haiyang Sun, Shuyun Wang, Zhuojie Wu, Hongwei Sheng, Jiaying Ying, Ming Lu, Tianqing Zhu, Kun Zhan, and Xin Yu. 3drealcar: An in-the-wild rgbd car dataset with 360-degree views. arXiv preprint arXiv:2406.04875, 2024. 2, 7
- [17] Francis Engelmann, Jörg Stückler, and Bastian Leibe. Samp: shape and motion priors for 4d vehicle reconstruction. In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 400–408. IEEE, 2017. 2
- [18] Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, Ben Sapp, Charles R Qi, Yin Zhou, et al. Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9710–9719, 2021. 6, 15
- [19] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing. In European Conference on Computer Vision, pages 73–89. Springer, 2024. 3
- [20] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 3354–3361, 2012. 1
- [21] Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. Radiant foam: Real-time differentiable ray tracing. arXiv preprint arXiv:2502.01157, 2025. 3
- [22] Can Gümeli, Angela Dai, and Matthias Nießner. Roca: Robust cad model retrieval and alignment from a single image. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4012–4021, 2022. 2
- [23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2017. 8, 15
- [24] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024. 4
- [25] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5322–5332, 2024. 3

- [26] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, and Hao Su. Tensoir: Tensorial inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 165–174, 2023. 3
- [27] Glenn Jocher and Jing Qiu. Ultralytics yolo11, 2024. 5, 15
- [28] James T Kajiya. The rendering equation. In Proceedings of the 13th annual conference on Computer graphics and interactive techniques, pages 143–150, 1986. 3, 4
- [29] Joanna Kaleta, Kacper Kania, Tomasz Trzcinski, and Marek Kowalski. Lumigauss: High-fidelity outdoor relighting with 2d gaussian splatting. arXiv preprint arXiv:2408.04474,

2024. 3

- [30] Brian Karis and Epic Games. Real shading in unreal engine

4. Proc. Physically Based Shading Theory Practice, 4(3):1,

2013. 3

- [31] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 1, 3

- [32] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42

(4), 2023. 2

- [33] Mustafa Khan, Hamidreza Fazlali, Dhruv Sharma, Tongtong Cao, Dongfeng Bai, Yuan Ren, and Bingbing Liu. Autosplat: Constrained gaussian splatting for autonomous driving scene reconstruction. arXiv preprint arXiv:2407.02598, 2024. 1, 2, 3, 6
- [34] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as markov chain monte carlo. Advances in Neural Information Processing Systems, 37:80965–80986, 2024. 1
- [35] Ruofan Liang, Zan Gojcic, Merlin Nimier-David, David Acuna, Nandita Vijaykumar, Sanja Fidler, and Zian Wang. Photorealistic object insertion with diffusion-guided inverse rendering. In European Conference on Computer Vision, pages 446–465. Springer, 2024. 3
- [36] Ruofan Liang, Kai He, Zan Gojcic, Igor Gilitschenski, Sanja Fidler, Nandita Vijaykumar, and Zian Wang. Luxdit: Lighting estimation with video diffusion transformer. arXiv preprint arXiv:2509.03680, 2025. 3
- [37] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, and Yadong Mu. Diffsplat: Repurposing image diffusion models for scalable gaussian splat generation. arXiv preprint arXiv:2501.16764, 2025. 7
- [38] Yichong Lu, Yichi Cai, Shangzhan Zhang, Hongyu Zhou, Haoji Hu, Huimin Yu, Andreas Geiger, and Yiyi Liao. Urbancad: Towards highly controllable and photorealistic 3d vehicles for urban scene simulation. arXiv preprint arXiv:2411.19292, 2024. 2, 7
- [39] Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, and Konrad Schindler. Mot16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831, 2016. 6
- [40] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:

- Representing scenes as neural radiance fields for view synthesis. In Computer Vision – ECCV 2020, pages 405–421, Cham, 2020. Springer International Publishing. 13
- [41] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes. ACM Transactions on Graphics (TOG), 43(6):1–19, 2024. 3
- [42] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas Müller, and Sanja Fidler. Extracting triangular 3d models, materials, and lighting from images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8280– 8290, 2022. 4
- [43] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 7
- [44] Dario Pavllo, David Joseph Tan, Marie-Julie Rakotosaona, and Federico Tombari. Shape, pose, and appearance from a single image via bootstrapped radiance field inversion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [45] Pakkapon Phongthawee, Worameth Chinchuthakun, Nontaphat Sinsunthithet, Varun Jampani, Amit Raj, Pramook Khungurn, and Supasorn Suwajanakorn. Diffusionlight: Light probes for free by painting a chrome ball. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 98–108, 2024. 3, 5
- [46] Ravi Ramamoorthi and Pat Hanrahan. An efficient representation for irradiance environment maps. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 497–500, 2001. 3
- [47] Yossi Rubner, Carlo Tomasi, and Leonidas J Guibas. The earth mover’s distance as a metric for image retrieval. International journal of computer vision, 40(2):99–121, 2000. 15
- [48] Christopher Scarvelis, David Benhaim, and Paul Zhang. Orient anything. arXiv preprint arXiv:2410.02101, 2024. 5
- [49] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurélien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2443–2451,

2020. 1, 5, 13

- [50] Jiapeng Tang, Matthew Lavine, Dor Verbin, Stephan J Garbin, Matthias Nießner, Ricardo Martin Brualla, Pratul P Srinivasan, and Philipp Henzler. Rogr: Relightable

- 3d objects using generative relighting. arXiv preprint arXiv:2510.03163, 2025. 3
- [51] Adam Tonderski, Carl Lindström, Georg Hess, William Ljungbergh, Lennart Svensson, and Christoffer Petersson. Neurad: Neural rendering for autonomous driving. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14895–14904, 2024. 13
- [52] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 3, 7
- [53] Haithem Turki, Jason Y. Zhang, Francesco Ferroni, and Deva Ramanan. Suds: Scalable urban dynamic scenes. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12375–12385, 2023. 13
- [54] Mikaela Angelina Uy, Jingwei Huang, Minhyuk Sung, Tolga Birdal, and Leonidas Guibas. Deformation-aware 3d model embedding and retrieval. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16, pages 397–413. Springer,

2020. 2

- [55] Dor Verbin, Pratul P Srinivasan, Peter Hedman, Ben Mildenhall, Benjamin Attal, Richard Szeliski, and Jonathan T Barron. Nerf-casting: Improved view-dependent appearance with consistent reflections. In SIGGRAPH Asia 2024 Conference Papers, pages 1–10, 2024. 3
- [56] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 4, 13
- [57] Jingkang Wang, Sivabalan Manivasagam, Yun Chen, Ze Yang, Ioan Andrei Bârsan, Anqi Joyce Yang, Wei-Chiu Ma, and Raquel Urtasun. Cadsim: Robust and scalable in-thewild 3d reconstruction for controllable sensor simulation. arXiv preprint arXiv:2311.01447, 2023. 1, 2
- [58] Yuxi Wei, Zi Wang, Yifan Lu, Chenxin Xu, Changxing Liu, Hao Zhao, Siheng Chen, and Yanfeng Wang. Editable scene simulation for autonomous driving via collaborative llm-agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15077– 15087, 2024. 2
- [59] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310–20320, 2024. 2
- [60] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26024–26035, 2025. 1, 5
- [61] Tianhao Wu, Chuanxia Zheng, Frank Guan, Andrea Vedaldi, and Tat-Jen Cham. Amodal3r: Amodal 3d reconstruction

from occluded 2d images. arXiv preprint arXiv:2503.13439,

2025. 7, 15

- [62] Pengchuan Xiao, Zhenlei Shao, Steven Hao, Zishuo Zhang, Xiaolin Chai, Judy Jiao, Zesong Li, Jian Wu, Kai Sun, Kun Jiang, et al. Pandaset: Advanced sensor suite dataset for autonomous driving. In 2021 IEEE international intelligent transportation systems conference (ITSC), pages 3095–3101. IEEE, 2021. 1
- [63] Tao Xie, Xi Chen, Zhen Xu, Yiman Xie, Yudong Jin, Yujun Shen, Sida Peng, Hujun Bao, and Xiaowei Zhou. Envgs: Modeling view-dependent appearance with environment gaussian. arXiv preprint arXiv:2412.15215, 2024. 3
- [64] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In European Conference on Computer Vision, pages 156–173. Springer, 2024. 1, 2, 3, 6
- [65] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 3, 4, 13
- [66] Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, and Yue Wang. EmerneRF: Emergent spatialtemporal scene decomposition via self-supervision. In The Twelfth International Conference on Learning Representations, 2024. 13
- [67] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1389–1399, 2023. 1
- [68] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642, 2023. 2
- [69] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331–20341, 2024. 2
- [70] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447–19456,

2024. 1

- [71] Jason Zhang, Gengshan Yang, Shubham Tulsiani, and Deva Ramanan. Ners: Neural reflectance surfaces for sparse-view 3d reconstruction in the wild. Advances in Neural Information Processing Systems, 34:29835–29847, 2021. 2
- [72] Xiaoming Zhao, Pratul Srinivasan, Dor Verbin, Keunhong Park, Ricardo Martin Brualla, and Philipp Henzler. Illuminerf: 3d relighting without inverse rendering. Advances in Neural Information Processing Systems, 37:42593–42617,

2024. 3

- [73] Hongyu Zhou, Longzhong Lin, Jiabao Wang, Yichong Lu, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger,

- and Yiyi Liao. Hugsim: A real-time, photo-realistic and closed-loop simulator for autonomous driving. arXiv preprint arXiv:2412.01718, 2024. 1, 2
- [74] Hongyu Zhou, Jiahao Shao, Lu Xu, Dongfeng Bai, Weichao Qiu, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi Liao. Hugs: Holistic urban 3d scene understanding via gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21336–21345, 2024. 1, 2, 6
- [75] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21634–21643, 2024. 2

Supplementary material

- A. Urban Scene Reconstruction with NeRFs

NeRFs [40] can be used to model dynamic urban scenes. SUDS [53] uses a single network for dynamic actors, which limits the possibility of altering the behavior of the actors. EmerNeRF [66] follows a similar idea to SUDS by decomposing the scene purely into static and dynamic components. NeuRAD [51] takes advantage of monocular or LiDAR-based 3D bounding box predictions and proposes a joint optimization of object poses during the reconstruction process. Although these methods produce reasonable results, they are still 1) limited to the high training cost and low rendering speed; or 2) do not address extrapolation of future vehicle appearance far beyond the original camera views.

- B. Data Collection Details

The initial database contained ∼95,000 car videos of ∼100 views on average. The first filtering stage includes the filtering of low quality and overly dark images with the CLIPIQA model [56], discarding frames with a score < 0.2. Then, we use Qwen-2.5-VL-Instruct (7B) [65] to respond several questions for each frame:

- • “Does the image depict a car?”
- • “Is the car directly occluded?”
- • “Does the image depict the car interior?”
- • “Does a hand or finger block the view?”
- • “Is the car door open?”
- • “Does the image mainly depict the car window?” Based on the responses, we filter out the corresponding

frames or, in some cases, entire car instances. Also, if fewer than 45 valid frames remain for a given instance, the entire instance is discarded.

- C. Evaluation Setup Details

For scene reconstruction evaluation, we selected 12 scenes from the Waymo Open Dataset [49], with labels listed in Table 4. This table also provides the correspondence between the original scene labels from the Waymo Cloud Storage and the short names used in our work. We split each scene into training and testing subsets based on time (Table 5) and camera selection (Table 6). Specifically, frames with indices itrain, where itrain ∈ [itrainstart,itrainend ], were used for training. For evaluation, we used frames itest ∈ [iteststart,itestend], with all split indices provided in Table 5.

- D. Per-Scene Quantitative Evaluation.

In addition to the aggregated results in Table 1, we report per-scene metric values in Table 7, Table 8, Table 9, and Table 10, corresponding to MOTA, MOTP, IDF1, and IoU,

- Table 4. Waymo scenes used for evaluation of scene reconstruction.

Label Scene name

1231623110026745648_480_000_500_000 123 1432918953215186312_5101_320_5121_320 143 1906113358876584689_1359_560_1379_560 190

10500357041547037089_1474_800_1494_800 105 10940952441434390507_1888_710_1908_710 109

16504318334867223853_480_000_500_000 165 17407069523496279950_4354_900_4374_900 174

18025338595059503802_571_216_591_216 180 14183710428479823719_3140_000_3160_000 141 15834329472172048691_2956_760_2976_760 158 17647858901077503501_1500_000_1520_000 176

7799671367768576481_260_000_280_000 779

- Table 5. Train and test frame splits for Waymo scenes over time. All values, except those in the leftmost column, indicate frame indices starting from 0.

Scene name itrainstart itrainend iteststart itestend 123 106 116 117 175 143 43 53 54 62 190 115 125 126 137 105 164 174 175 196 109 1 16 17 55 165 7 40 41 111 174 34 51 52 72 180 49 55 56 68 141 60 80 81 117 158 44 62 63 100 176 31 42 43 67 779 50 65 66 84

- Table 6. Train and test frame splits for Waymo scenes based on camera selection.

Scene name Train cameras Test cameras

123 frontal, frontal left frontal, frontal left 143 frontal, frontal left frontal, frontal left 190 frontal, frontal left frontal, frontal left 105 frontal, frontal left frontal 109 frontal, frontal right frontal right 165 frontal, frontal left frontal, frontal left 174 frontal frontal 180 frontal, frontal right frontal, frontal right 141 frontal frontal 158 frontal frontal 176 frontal frontal 779 frontal, frontal left, frontal right frontal, frontal right

respectively. We observe that MADRIVE consistently outperforms the baselines across most scenes.

- Table 7. Mean MOTA ↑ results on test frames for all Waymo scenes.

Scene name SG HUGS AutoSplat MADRIVE

123 0.687 0.685 0.327 0.887 143 0.650 0.513 0.600 0.825 190 0.787 0.795 0.904 0.858 105 0.906 0.656 0.742 0.906 109 0.242 0.448 0.605 0.925 165 0.684 0.461 0.788 0.883 174 0.809 0.886 0.830 0.936 180 0.611 0.528 0.695 0.778 141 0.667 0.607 0.163 0.767 158 0.423 0.233 0.681 0.639 176 0.727 0.562 0.176 0.912 779 0.661 0.296 0.545 0.779

- Table 8. Mean MOTP ↓ results on test frames for all Waymo scenes.

Scene name SG HUGS AutoSplat MADRIVE

123 0.073 0.093 0.099 0.079 143 0.114 0.461 0.095 0.203 190 0.088 0.112 0.115 0.144 105 0.073 0.262 0.222 0.118 109 0.093 0.132 0.094 0.122 165 0.125 0.202 0.119 0.149 174 0.075 0.886 0.078 0.093 180 0.150 0.231 0.194 0.195 141 0.119 0.261 0.237 0.179 158 0.087 0.128 0.123 0.119 176 0.093 0.246 0.167 0.072 779 0.176 0.443 0.305 0.180

- Table 9. Mean IDF1 ↑ results on test frames for all Waymo scenes.

Scene name SG HUGS AutoSplat MADRIVE

123 0.804 0.806 0.475 0.940 143 0.787 0.709 0.750 0.904 190 0.880 0.887 0.950 0.924 105 0.952 0.780 0.877 0.951 109 0.390 0.619 0.754 0.961 165 0.806 0.612 0.894 0.936 174 0.894 0.940 0.907 0.967 180 0.753 0.709 0.820 0.871 141 0.805 0.698 0.278 0.866 158 0.605 0.377 0.829 0.797 176 0.847 0.720 0.316 0.955 779 0.793 0.532 0.739 0.881

### E. Baseline Details

Baselines training and evaluation. We trained all baselines (Street-Gaussians, HUGS, and AutoSplat) for 10K

Table 10. Mean IoU ↑ results on test frames for all Waymo scenes.

Scene name SG HUGS AutoSplat MADRIVE

123 0.753 0.608 0.500 0.866 143 0.485 0.243 0.510 0.779 190 0.707 0.519 0.740 0.846 105 0.671 0.425 0.439 0.731 109 0.499 0.246 0.419 0.832 165 0.633 0.459 0.647 0.730 174 0.695 0.581 0.655 0.829 180 0.475 0.238 0.582 0.814 141 0.607 0.196 0.226 0.765 158 0.404 0.135 0.498 0.862 176 0.499 0.187 0.263 0.886 779 0.247 0.153 0.273 0.874

iterations using the training frames with indices i ∈ [itrainstart,itrainend ] as specified in Table 5. Additionally, we trained the background models for both the baselines and MADRIVE for 30K iterations using all available frames. These pretrained background models were then used during the rendering of the test frames (i ∈ [iteststart,itestend]), on which we compute the metrics reported in Table 7, Table 8, Table 9, and Table 10.

Street-Gaussians. We used the official implementation available at https://github.com/zju3dv/ street_gaussians.

HUGS. We used the official implementation provided at https://github.com/hyzhou404/HUGSIM.

AutoSplat. As no official implementation is publicly available, we re-implemented the core contributions of AutoSplat on top of the Street-Gaussians codebase.

### F. Choice of Reference Masks in the Evaluation

In our validation setup, we used predictions from tracking and segmentation models on ground-truth images as targets, since the Waymo dataset lacks segmentation masks and 2D bounding boxes.

To evaluate whether cars in the synthesized frames are as identifiable as those in the original frames, we applied the same detection algorithm to both. Our method outperforms the baseline, primarily because our system inserts visually coherent cars on test frames by leveraging reconstructed models, whereas baseline approaches result in degraded or incomplete vehicle representations.

However, the inserted cars might be easier to detect. To test this, we conducted an additional experiment. Specifically, we generated a new set of detector targets by project-

ing the 3D bounding boxes provided in the Waymo dataset onto the image plane. We then evaluated the performance of the detector on both the ground-truth and synthesized (MADrive) frames using the new "ground-truth" annotation.

The results in Table 11 show that the predictions on ground-truth images align slightly better with the projected 3D bounding boxes than those on the synthesized MADrive frames. This indicates that our inserted cars do not artificially simplify detection, supporting the validity of our evaluation.

Table 11. Comparison between detector performance on both the ground-truth and synthesized (MADrive) frames using projected 3D bounding boxes.

Model MOTA ↑ MOTP ↓ IDF1 ↑ GT frames 0.879 0.270 0.928 MADRIVE frames 0.861 0.340 0.908

### G. Additional Retrieval Evaluation Results

In this section, we provide an additional illustration of our retrieval algorithm. As shown in our Figure 8, introducing a color-based pre-filter enhances the alignment of vehicle colors between retrieved candidates and the target.

### H. Evaluation of Realism for Reconstructed Models

To evaluate the realism of the reconstructed 3D car models using retrieval approach of MADRIVE and generative image-to-3D method of Amodal3R [61], we collected 38 random crops of hold-out cars from the Waymo dataset [18]. We reconstructed 3D car models for all these crops with MADRIVE and Amodal3R and rendered them from 360 degrees with 60 renderings per reconstructed model. For Amodal3R, we also segment cars in crops using the YOLOv11 instance segmentation model [27]. For reference images, we took 30 hold-out cars from MAD-CARS and collected near 2200 images for these cars. To reduce the domain shift, we excluded backgrounds from real images from MAD-CARS with YOLOv11. Finally, we computed FID [23] and KID [5] scores between two sets: 1) the set of all renderings from the reconstructed 3D models with MADRIVE and Amodal3R, which we ran on 38 crops from Waymo; 2) the set of real images for 30 test cars from MAD-CARS. Visually, we observed that the results of Amodal3R typically produce low-resolution cars with limited detail and a cartoon-like appearance, which explains the results in the Table 3.

### I. Relighting Evaluation

To assess the impact of our relighting module, we analyze the cars shown in Figure 7.

Standard pixel-wise metrics (e.g., LPIPS) failed to capture meaningful differences, largely because geometric mismatches between the inserted asset and the original vehicle dominate these scores. Instead, we evaluate relighting by comparing the color statistics of the integrated assets to those of the real cars.

We extract crops of the inserted vehicles and matching crops from the corresponding ground-truth frames. Each crop is converted to the perceptually uniform CIELAB color space, and we compare their pixel-intensity distributions using the sliced Wasserstein distance [47]. Average distances with standard deviations between different crops are provided in Table 12.

Across all evaluated cars, relighting consistently reduces the discrepancy between synthetic and real crops. A more detailed breakdown reveals that the improvement is especially pronounced in the lightness (L*) channel, while differences in the chromatic channels (a*, b*) are smaller—reflecting the fact that lighting primarily affects perceived brightness rather than intrinsic surface color.

W1 W1,L∗ W1,(a∗,b∗)

W/orelighting 6.74 ± 1, 83 10.89 ± 4.78 3.82 ± 3.34 W/relighting 3.15 ± 0.90 3.54 ± 2.75 2.89 ± 1.30

Table 12. Sliced Wasserstein distances between real and synthetic car crops in CIELAB space. Lower values indicate closer color distribution matches. Relighting consistently reduces the discrepancy, with the largest improvement observed in the lightness (L*) channel.

### J. Additional Qualitative Comparisons and New Trajectories

We provide additional visual results in Figures 9 and 10. We also demonstrate our method’s capability to render novel views with substantial scene variations. Figure 11 showcases results across four test scenes, where all modifications preserve high image quality.

### K. Limitations

Reconstruction limitations. To run reconstruction, we estimate camera parameters from the input images. In particular, we run bundle adjustment starting from the initialization obtained with VGGT. At present, errors in camera estimation remain a primary source of reconstruction failures. We expect that continued advances in foundational vision models will substantially reduce this limitation.

[Figure 125]

Query Top-4 candidates, w/o color filtering Query Top-4 candidates, w/ color filtering

Figure 8. Retrieval illustration. Top-4 candidates retrieved using SigLIP 2 without (Left) and with (Right) color filtering.

State-of-the-art multiview reconstruction methods continue to struggle with reflective and glossy surfaces like cars even up to this day. Accurate modeling of reflections on metallic surfaces on real datasets demands more precise representations of illumination - beyond what conventional environment maps can provide.

Dataset limitations. We rely on an external dataset sourced from online car sale advertisements, which primarily features passenger vehicles. As a result, other vehicle categories (e.g. buses, trucks, and service vehicles) are underrepresented and cannot yet be reliably replaced by our method.

Nonetheless, our pipeline is fully modular: adding support for additional vehicle types only requires capturing a 360° video of the target vehicle to add it to the retrieval database.

### L. Statement on LLM usage

The authors used the large language model (LLM) only to improve the writing and grammar of the text. All the results from the LLM were checked by the authors.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Seen

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

SeenFuture

Ground Truth Street Gaussians AutoSplat HUGS MADrive

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

SeenFuture

Ground Truth Street Gaussians AutoSplat HUGS MADrive

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

Future

Ground Truth Street Gaussians AutoSplat HUGS MADrive

- Figure 9. Additional qualitative comparison of MADRIVE with non-retrieval-based driving scene reconstruction methods. Reconstruction of the training views (Top). Reconstruction of the hold-out (future) views (Bottom).

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Seen

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

SeenFuture

Ground Truth Street Gaussians AutoSplat HUGS MADrive

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

SeenFuture

Ground Truth Street Gaussians AutoSplat HUGS MADrive

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

Future

Ground Truth Street Gaussians AutoSplat HUGS MADrive

- Figure 10. Additional qualitative comparison of MADRIVE with non-retrieval-based driving scene reconstruction methods. Reconstruction of the training views (Top). Reconstruction of the hold-out (future) views (Bottom).

##### OriginalOriginalOriginalOriginalModifiedModifiedModifiedModified

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

- Figure 11. Visualization of original and modified trajectories with MADRIVE. The cars retain high-fidelity appearance even at close distances to the ego camera.

