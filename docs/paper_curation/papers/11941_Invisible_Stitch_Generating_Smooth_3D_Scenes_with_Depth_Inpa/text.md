# arXiv:2404.19758v1[cs.CV]30Apr2024

## Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting

##### Paul Engstler, Andrea Vedaldi, Iro Laina, and Christian Rupprecht

Visual Geometry Group, University of Oxford {paule,vedaldi,iro,chrisr}@robots.ox.ac.uk https://research.paulengstler.com/invisible-stitch

Abstract. 3D scene generation has quickly become a challenging new research direction, fueled by consistent improvements of 2D generative diffusion models. Most prior work in this area generates scenes by iteratively stitching newly generated frames with existing geometry. These works often depend on pre-trained monocular depth estimators to lift the generated images into 3D, fusing them with the existing scene representation. These approaches are then often evaluated via a text metric, measuring the similarity between the generated images and a given text prompt. In this work, we make two fundamental contributions to the field of 3D scene generation. First, we note that lifting images to 3D with a monocular depth estimation model is suboptimal as it ignores the geometry of the existing scene. We thus introduce a novel depth completion model, trained via teacher distillation and self-training to learn the 3D fusion process, resulting in improved geometric coherence of the scene. Second, we introduce a new benchmarking scheme for scene generation methods that is based on ground truth geometry, and thus measures the quality of the structure of the scene.

Keywords: Scene Generation · Novel View Synthesis · 3D Geometry

### 1 Introduction

The rise of high-quality image generative models is opening up many new interesting computer vision applications. Especially the field of novel-view synthesis is being transformed through leveraging the visual priors learned by large-scale generative models. In this field, a new direction is emerging: scene generation. Here, the goal is to generate not just a new image or view, but a whole scene, often starting from a single input image or text description. To this end, current approaches apply an iterative process of alternating geometry estimation, moving the camera and inpainting holes until the whole scene is generated. To estimate the geometry of the scene, these methods rely on general-purpose depth estimation models, that can infer relative or absolute scene depth from a single input image. This approach, however, often results in inconsistencies, as the depth prediction is purely based on image information and does not take into account the geometry of the already existing scene.

Currently, scene generation is evaluated only visually, through image-based metrics such as the CLIP score between a scene description and individual views. This evaluation does not assess the geometry of the scene and often simply evaluates the image quality of the generator instead of evaluating the generated scene.

Thus, in this paper, we make two contributions to the field of scene generation. Firstly, we introduce a rigorous scene geometry evaluation benchmark, based on real and synthetic scene datasets. The idea is to use ground truth images and depth and evaluate the geometry of generated scenes on the basis of depth maps. Given a view of a scene, a method is tasked to generate a scene from a given view point, for which there exists a ground-truth depth map. The generated scene geometry can then be easily evaluated. Secondly, we introduce a new general-purpose depth estimation model that can be conditioned on an incomplete depth map. Incomplete depth maps are obtained by projecting the existing scene into a new view point, where occlusions and unseen parts of the scene leave holes. In this manner, the newly predicted depth is conditional on the previous generations and results in a greatly increased consistency across the scene.

Our conditional depth prediction model can be trained in a simple selfsupervised way. We fine-tune an existing depth prediction model and condition it on partially masked depth maps. This can be done on a simple image dataset using pseudo-ground-truth from the model that is being fine-tuned and does not rely on camera poses or other annotations. To obtain realistic depth masks, we automatically create a dataset of masks by warping predicted depth maps to random views. This allows masking depth maps with masks similar to the ones that will be encountered during inference when generating scenes.

In our experiments, we show that existing scene generation methods suffer from geometric inconsistencies that are uncovered by our new benchmark. Furthermore, we can show that our conditional depth prediction model drastically reduces these artefacts as it, by design, is trained to retain geometric consistency across subsequent frames. Our method obtains state-of-the-art performance and is general: it can be used in other scene-generation approaches for improved consistency.

We summarize the contributions of our work as follows:

- – We introduce a depth-inpainting model that extrapolates depth in 3D scene generation tasks.
- – We provide a new benchmark for evaluating the geometric quality of scene generation methods.
- – We show that our approach generates depth that is consistent with the existing scene and is superior to prior alignment approaches that utilize unconditioned monocular depth estimation.

### 2 Related Work

Depth Completion With the emergence of depth sensing technologies, inferring a dense depth map of a 3D scene from a sparse depth representation and a given RGB image has gained significant importance. Works in this field seek to integrate cues from both modalities either in a 2D [8, 9, 37, 62] or 3D feature space [3,7,19,21,54] to produce a complete depth map.

While these methods are able to recover the depth of an entire scene from a possibly very sparse depth input, they have not been designed to complete depth for regions without any depth information, which naturally occur in a scene generation task.

##### 3D Scene Generation Text-to-3D or image-to-3D scene generation has seen tremendous progress in recent years, where the majority of works in this field can either be categorized as object-centric [13,20,27,29,30,38–40,46,47,49,52,55,59], i.e., focusing on objects without background, or holistic, generating a single 3D scene or 3D trajectories with background.

Earlier methods in the former group are only concerned with novel view synthesis, not considering the scene’s geometry. These have been based on layerstructured representations [25, 31, 48, 53], e.g., layered depth images, or more implicit ones, such as in SynSin [56]. More moderns approaches typically distill a 3D representation, such as a NeRF [32] or 3D Gaussians [23], from the supervision of a 2D image generation model like Stable Diffusion [43]. Other works directly learn a 3D representation from 2D images [5,6,15,34,35,51].

More holistic methods consider generating entire scenes beyond a single object. These methods generally build a scene in a sequential manner using supervision from 2D image generation models. PixelSynth [42], Text2Room [17], Text2NeRF [61], LucidDreamer [10], WonderJourney [60], Infinite Nature [28] and Text2Immersion [36] rely on an off-the-shelf general-purpose depth estimation model, such as ZoeDepth [2], to project the hallucinated 2D scene extensions into a 3D representation. The Denoising Diffusion Vision Model [45] has been proposed as another backbone for this approach. Other approaches learn an implicit representation, such as GAUDI [1], ZeroNVS [44], DiffDreamer [4], and InfiniteNature-Zero [26]. LDM3D [50], 3D-aware Image Generation using

- 2D Diffusion Models [58], and RGBD2 [24] fuse image and depth prediction to generate a scene. More specialized methods introduce different representations, such as BlockFusion [57], Worldsheet [18], and Set-the-Scene [11].

While multiple works explicitly acknowledge the importance of consistent scene geometry, it has not been considered broadly. As we consider scene generation an inherently geometric task, we dedicate it our full attention in this work.

- 3 Method

Current 3D scene generation methods rely on 2D-based models like Stable Diffusion [43] to hallucinate scenes beyond known regions and lift generated images

into three dimensions by utilizing depth estimation networks. In this task, it is crucial to seamlessly integrate the newly hallucinated regions into the existing scene representation. Previous methods [10,17,28,36,60,61] align these by applying simple global scale-and-shift operations to the predicted depth map along with other minor modifications, if any. These operations, however, might be too coarse and insufficiently local, potentially causing the transition between the scene and its hallucinated extension to be not perfectly smooth, leading to visible discontinuities.

#### 3.1 3D Scene Generation

[Figure 1]

###### Fig. 1: Overview of our 3D scene generation method. Starting from an input

image I0, we project it to a point cloud based on a depth map predicted by a depth estimation network g. To extend the scene, we render it from a new view point and query a generative model f to hallucinate beyond the scene’s boundary. Now, we condition g on the depth of the existing scene and the image of the scene extended by f to produce a geometrically consistent depth map to project the hallucinated points. This process may be repeated until a 360-degree scene has been generated.

The task of scene generation from a single image I0 ∈ R3×Ω (where Ω = {1,...,H} × {1,...,W} is a lattice representing pixels) can be formulated as follows. Given an arbitrary view point V = [R|t] ∈ SE(3) and an intrinsic camera matrix K, the task is to provide a view of the scene Iˆ(V,K) ∈ R3×Ω that is consistent with the original image and any other views that have already been generated. To achieve this consistency, we parametrise the scene as a pointcloud P = {(Cj,Xj)}j of points at 3D locations Xj ∈ R3 with color Cj ∈ R3.

Generating a new view from a pointcloud can be done by projecting the

- 3D points to pixels into the image plane of the new view xj ≡ KV −1Xj (since xj ∈ R2, here ≡ represents the mapping from homogeneous coordinates to image coordinates). However, this forward projection can leave holes in the resulting

image I˜(xi) = Cj when the camera is looking at previously unseen regions. We thus also obtain a binary mask M = {0,1}Ω that indicates these holes in the image.

To complete the sparse projection I˜ one can leverage a large-scale generative

- model f (e.g., Stable Diffusion [43]) which has learned a visual prior for a massive collection of visual data. In particular, we use an inpainting variant of the Stable Diffusion model that has been trained to fill in missing regions in an image Iˆ = f(I,M˜ ).

In 3D scene generation from a single image or a text description, one does not have access to a 3D pointcloud of the scene. Instead, the goal is to generate said pointcloud. We obtain P through an iterative process that by design enforces the consistency between each view of the scene. A natural mapping from images to pointclouds can be established via depth maps D ∈ RH×W as it allows the projection of the image pixels into the scene

 

 

 

 

- u
- v

Pˆi =

ViKi−1

. (1)





Di(u,v)

(u,v)∈Ω

We construct P by iteratively expanding the scene representation. Let Pi be the pointcloud at the i-th iteration. Each iteration expands the representation with new geometry as Pi+1 = Pi ∪ Pˆi+1.

At each iteration, we choose a new view point Vi and camera matrix Ki to expand the scene. We achieve this by passing the current image Iˆi, which corresponds to a new generation, to a depth estimation network to obtain the corresponding depth map Di. Specifically, the depth estimation network is defined as Di = g(Iˆi,Mi,D˜i), i.e., it takes as additional inputs the mask Mi, which signifies which pixels have no depth estimate (since Iˆi is an inpainted image), and D˜i, which is the depth map obtained by projecting the pointcloud Pi−1 into the current view (and thus contains holes). Note that since the depth map D˜i is obtained by projecting the existing point cloud Pi−1 into the current view, only the holes indicated by Mi that get filled in by g contribute new points to Pi. The other points already exist in the scene.

The depth prediction model g plays a critical role, as it uses a complete image and a partial depth map to predict new scene geometry that matches the image and the existing geometry. Prior work uses off-the-shelf depth prediction models, that are not conditional on existing geometry. They thus need to rely on heuristics such as depth interpolation to make the new prediction consistent with the scene which introduces strong artifacts.

In the next section, we will describe our training scheme for learning the depth completion model g.

#### 3.2 Unsupervised Depth Completion

[Figure 2]

- Fig. 2: Overview of our training procedure. In this compact training scheme a depth completion network g is learned by jointly training depth inpainting as well as depth prediction without a sparse depth input (the ratio being determined by the task probability p). A teacher network gT is utilized to generate a pseudo ground-truth depth map D for a given image I. This depth map is then masked with a random mask M, to obtain a sparse depth input D˜.

Our goal is to learn a model g(I,M,D˜) for the purpose of 3D scene generation that provides a robust depth estimate given an image Iˆ, a depth-mask M, and a partial depth map D˜.

Naturally, this task can be trained with supervision, given a multi-view dataset with known camera poses and depth ground truth. However, these datasets are usually comparatively small and limited to specific scene types (e.g., indoors, driving, etc.). Yet scene generation is a broad task and models should generalise well to any kind of scene. We thus train g in a self-supervised fashion using an off-the-shelf general-purpose monocular depth prediction model gT(I) that predicts unconditional depth from a single RGB image. This model takes on the role of the teacher model in a student-teacher training scheme.

Given a dataset of only images Ik, we generate a pseudo-labelled training dataset for g as follows. For each image in the dataset, we obtain a target depth map from a teacher gT, Dk = gT(Ik). Then, for each depth map, and similar to the scene generation step, we sample one or more random view points Vl and camera matrix Kl and we warp the depth map Dk to the new view point obtaining a mask Mk,l and reprojected depth Dk,l. We collect all masks generated this way in a set M = {Mk,l}k,l that represents the typical occlusion patterns generated by view point changes.

Now, given the lack of multi-view data, during training, we sample a random mask from Mn ∈ M (1 ≤ n ≤ |M|) for each image Ik in the batch. And train

g to reconstruct the pseudo depth Dk guided by the scale-invariant loss [14], where d˜= g(Ik,Mn,Dk ⊙ Mn), d = Dk, and ψi = log d˜i − log di.

Ldepth =

λ T2

1 T i

ψi)2 (2)

ψi2 −

(

i

T is the number of pixels in Dk that have valid ground-truth values. This scheme allows learning g only from pseudo-supervision. While a naïve scheme would use random masks, instead of masks generated by warping depth maps, we found that a model trained this way does not generalize well to the particular patterns intrinsic to depth projection.

An additional benefit is that g can be initialized with a depth estimation model itself, effectively fine-tuning it for depth completion. Moreover, we can then retain its original depth prediction (instead of depth completion) capabilities by choosing Mn = 0 with probability p, effectively masking all input depth, and recovering the depth prediction task. Finally, as is typical (but not necessary) in student-teacher training, we can choose gT as a large model while g can be a more lightweight architecture, which improves g via distillation.

### 4 Scene Geometry Evaluation Benchmark

Within the fully generative task of scene generation, evaluating the geometric properties of generated scenes is difficult due to the lack of ground-truth data. As a result, most existing work resorts to image-text similarity scores, such as the CLIP-Score [16], which only measures the global semantic alignment of the generation with a text description. This leaves open questions about the geometric consistency and quality of the depth predictions used to build the scene. To answer these questions, we propose a new evaluation benchmark that quantifies the depth-reconstruction quality on a partial scene with known ground truth depth. More specifically, we seek to measure the deviation between the extrapolated depth and the ground-truth.

Approach Starting from a scene representation fully constructed from the ground-truth information of one view, we seek to extrapolate the depth for another ground-truth view that is highly overlapping with the first one. As the depth is known for the second view, we compute the error between the generated depth and the ground-truth depth. We only consider the error in regions that were extrapolated. A detailed description of this approach is provided in Figure 3.

We use a point cloud as our representation of choice and base the overlap of two views ϕ(vi,vj) on the number of pixels that show part of the vi scene from the view point of vj. Put differently, if a rendering pipeline renders an image with dimensions H ×W and assigns a default value x for a pixel p that does not

[Figure 3]

- Fig. 3: Overview of our scene consistency evaluation approach. Assume a scene is described by a set of views {v1, v2, . . . } with associated images, depth maps, and camera poses, where the overlap of two views is described by a function ϕ(vi, vj). For a given view pair (vi, vj) with ϕ(vi, vj) ≥ τ, we generate a representation, e.g., a point cloud, from the ground-truth (GT) data for vi. Then, we render the representation from the view point of vj. We feed the corresponding ground-truth image and the representation’s depth into the model under consideration to extrapolate the missing depth. Finally, we calculate the mean absolute error between the result and the groundtruth depth for vj, only considering those regions that were extrapolated.

represent any parts of a scene, we define ϕ as:

H×W i,j p(i,j)̸=x

ϕ(vi,vj) =

H × W

(3)

Datasets In our evaluation, we consider ScanNet [12] and Hypersim [41] as they provide images, dense depth, and camera poses to accurately reconstruct scenes. They boast a wide range of scenes with varying complexity, making them an ideal test bed for evaluating the quality of depth predictions. As the former is a real-world dataset featuring indoor scenes and the latter is a photorealistic one, they lie within the distributions of most depth estimation models.

Scenes in ScanNet are described by highly-overlapping sequential frames. Thus, we chunk them into blocks of 50 frames and consider the first and tenth frame in each block for our evaluation. This allows us to yield ample views from each scene and maintains diversity, while limiting the number of evaluations to run. As the sequential frames are naturally highly overlapping, we refrain from setting a specific threshold τ. To maintain reasonable evaluation times, we only consider the first 50 scenes, which yields a total of 7,832 view pairs.

With Hypersim, we compute ϕ across all views of a single camera trajectory within a scene and set τ := 0.8. We exclude scenes rendered with non-standard projection matrices1. The resulting number of view pairs that we evaluate on is 19,243.

1 See https://github.com/apple/ml-hypersim/issues/24

For both datasets, we report the average absolute error on the extrapolated region across all pairs of views across all scenes.

### 5 Experiments

#### 5.1 Implementation Details

We fine-tune a pre-trained ZoeDepth model to obtain our depth completion model g, re-initializing its patch embedding layer to receive two additional channels apart from the image input. These channels provide the sparse depth input D˜ as well as a mask M describing the presence of sparse depth, i.e., D˜ > 0. While we only replace this layer, we keep the entire model unfrozen to ensure the additional information can be integrated in later layers. We set λ = 0.85 in the scale-invariant loss.

We train on images from the NYU Depth v2 [33] dataset, using the monocular depth estimation network Marigold [22] as a teacher gT to distill its prediction capabilities into our depth estimation network g.

To construct the set of warped masks M, which contains typical masking patterns seen with view point changes, we consider the Places365 [63] dataset, generating one mask from each image. Masks are randomly chosen to be applied to a training sample depth D.

Ensuring we retain the original depth prediction task, we set the probability to zero out the sparse depth input to p := 0.5.

#### 5.2 Generating 360-Degree Scenes

Having shown that our model produces depth with properties beneficial to scene generation, we now utilize it for this very task. Specifically, we aim to generate immersive 360-degree scenes starting from a single real-world image. We will first outline the design of our pipeline to generate the scene, and then present quantitative and qualitative results.

Pipeline Design Our pipeline relies on multiple components to generate a scene from a single image: First, in the same vein as current 3D scene generation methods, we enlist the help of a Stable Diffusion inpainting model (f) to hallucinate how a scene looks like beyond its boundaries. Second, we use our depth inpainting model (g) to produce an initial depth estimation for the original image and inpaint the depth in subsequent steps to attach the extrapolations. Third, we utilize Gaussian splat optimization [23] to obtain a smooth representation, filling in the gaps between points.

Generating the Point Cloud Starting from a given single image, we obtain a depth estimation to project it to a point cloud. We use a stationary perspective camera with fixed intrinsics. With each step, we rotate the camera slightly further

along its azimuth to obtain a view that provides a canvas for the Stable Diffusion model to inpaint while still partially including the existing scene.

When inpainting images with Stable Diffusion, distortion artifacts have been known to appear in those regions that are not supposed to be edited, which has been attributed to its variational autoencoder [64]. These alterations then cause a mismatch between the input image and the sparse depth input, which is difficult to resolve. To minimize these effects, we utilize an asymmetric autoencoder [64] that emphasizes the decoder, making it heavier than the encoder and providing it with additional information about the inpainting task. We find that this autoencoder leads to a significant decrease in the prevalence of these artifacts.

Once the expanded scene has been visually hallucinated by Stable Diffusion, we pass the image onto our depth inpainting model with the depth of the existing scene. We project all hallucinated pixels based on the depth prediction, which seamlessly connect to the point cloud without the need for any alignment steps. We observe that depth predictions might have a gradient instead of a hard boundary at object edges, which leads to floaters radiating around objects in the point cloud. To minimize their occurrence, we identify regions in the predicted depth map with a high gradient, mask them, and assign them new values based on their nearest neighbors. This snaps pixels in these gradient regions either to the object or its surrounding, creating a hard boundary.

We repeat this process until the loop is closed, yielding a 360-degree scene. We make sure that the final hallucination step has a wide canvas to connect both ends of the loop, assuming there has been a slight domain shift between the original image and the cascade of hallucinated views.

Adding Support Views While the scene is complete from the view of the stationary camera, it might have lots of holes due to occlusions. To fill out the scene, we generate additional views where we employ a look-at camera that looks behind objects, enabling us to hallucinate disocclusion regions.

For a given view from the point cloud generation phase, we determine that view’s center point by averaging its depth values and define a camera looking at that particular point, with small random azimuth and elevation rotation values. As this camera might uncover unobserved regions of a scene, we query Stable Diffusion and our depth inpainting model again to extend the point cloud.

Additionally, these views also further constrain the Gaussian splat optimization process, yielding better results due to less ambiguity.

Gaussian Splatting As the final step of our pipeline, we turn the obtained point cloud into Gaussians that we optimize with the hallucinated and supporting views that we obtained from the previous two phases.

We add this step to our pipeline to move away from the limiting point cloud representation to achieve a smooth scene, filling additional holes that might be present in this scene.

Qualitative Results Using this pipeline, we can generate 360-degree scenes given a single input image as well as a text prompt. In Figure 4, we show three example scenes generated by our approach that feature complex geometry. Our depth inpainting model is able to seamlessly extend scenes with believable geometry, creating an immersive experience.

We generate these results by rotating the camera 25 degrees along its azimuth with each step, slightly tapering it towards the end to close the loop. To aid the Gaussian splat optimization, we use eight support views per original view, letting the supporting view’s look-at camera peek behind objects by allowing a relative change of the azimuth and elevation angles of up to ±5 degrees.

CLIP-Score [16] Prague Kyoto North Carolina

Method

LucidDreamer [10] 26.01 29.56 24.96 Ours 26.87 27.65 24.24

- Table 1: Quantitative results for the visual quality of generated scenes. Both methods receive the same input image and text prompt (see Figure 4) to generate 360degree scenes. We use all hallucinated views to compute the average CLIP-Score.

Visual Quality To compare the visual quality of generated scenes with LucidDreamer [10], we utilize the CLIP-Score. We use the same input images and prompts shown in Figure 4 to produce similar scenes with both methods. As LucidDreamer [10] and our method both use Stable Diffusion to hallucinate images, the results we present for this metric presented in Table 1 carry limited information.

#### 5.3 Evaluating the Scene Geometry

To add another dimension to our evaluation, we detail the results of our proposed scene geometry evaluation benchmark (see Section 4) for LucidDreamer [10], the state-of-the-art depth completion network CostDCNet [21], ZoeDepth [2], as well

- as our fine-tuned inpainting network in Table 2. With ZoeDepth, we predict depth and run a global scale-and-shift alignment

optimization with the existing scene representation. This approach is representative of the depth estimation methods used by other current scene generation methods like Text2Immersion [36], Text2Room [17], and WonderJourney [60]. We note that while LucidDreamer is similar to these methods, i.e., it is uses a monocular depth estimation network and a global alignment procedure, it adds another step to smoothly connect the predicted depth with the existing scene: It interpolates the depth values between the existing scene and the predicted depth to avoid seams. We observe that this operation appears to come at the cost of geometric consistency.

Prompt

Input

25◦

50◦

75◦

100◦

125◦

175◦

225◦

272.5◦

316.25◦

Prague during the golden hour

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

A street with traditional buildings in Kyoto, Japan

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

A suburban street in North Carolina on a bright, sunny day

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

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

3D Scene

Cut-Away

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

- Fig. 4: Qualitative results of our method on real-world images. We show hallucinated views and the corresponding depth maps of 360-degree scenes. We also provide a full view of the generated 360-degree scene as well as a more detailed cut-away view.

In both, a real-world and a photorealistic setting, our inpainting model produces predictions that are more faithful to the ground-truth than the other methods.

Method ScanNet [12] Hypersim [41]

LucidDreamer [10] 0.1604 0.8057 CostDCNet [21] 0.5854 4.0149

ZoeDepth [2] 0.1293 0.7872 Ours 0.0816 0.7295

- Table 2: Scene geometry evaluation results of scene-generation methods on a real-world and a photorealistic dataset. We report the average depth reconstruction error produced by our proposed depth-inpainting method as well as related approaches. The notable difference in the errors between ScanNet and Hypersim can be explained by the complexity and simulated nature of Hypersim: First, it features notably more fine details than ScanNet that cannot be recovered by models operating

- at a lower resolution than its image size, such as ZoeDepth. Second, unlike real-world depth sensors, the depth in Hypersim is exact with sharp boundaries, which makes it more difficult for models trained on real-world data.

#### 5.4 Ablations

To validate the effectiveness of the design choices in our training pipeline, we ablate them and provide their results on our scene geometry evaluation benchmark in Table 3.

Distillation of High-Resolution Models It generally seems beneficial for the fine-tuning process to obtain a stronger training signal for depth annotations that are fully dense. Due to its simulated nature, Hypersim has ground-truth depth maps with notably finer structures than ScanNet, which was captured with less precise real-world tools. With Marigold [22] being trained on Hypersim, it is able to reproduce these high-resolution details. As we observe improved performance for this dataset once we use Marigold predictions as ground-truth, we deem our knowledge distillation setup to be effective.

Inpainting Task Probability We observe that there is merit to not allocating too little or too much time in the training procedure to learning the inpainting task. Not dedicating any time to the original depth estimation task without any sparse input appears to negatively impact the performance. We find that spending between 50-75% (i.e., p ∈ [0.25,0.5]) of the time in the fine-tuning process training the inpainting task yields performant models.

Input Depth Annot. p Warped Masks Align. ScanNet [12] Hypersim [41]

RGB+sd Original 0.5 - - 0.7734 2.2913 RGB+sd Original 0.5 ✓ - 0.1015 0.7615 RGB+sd ZoeDepth 0.5 ✓ - 0.0793 0.7555 RGB+sd Marigold 0.0 ✓ - 0.0864 0.7547 RGB+sd Marigold 0.25 ✓ - 0.0791 0.7301 RGB+sd Marigold 0.75 ✓ - 0.0869 0.7578

RGB Marigold 0.5 ✓ - 0.2553 1.1536 RGB Marigold 0.5 ✓ ✓ 0.1335 0.8152

RGB+sd Marigold 0.5 ✓ - 0.0816 0.7295

- Table 3: Scene geometry evaluation results for ablations of our method. We consider the input for our model (image-only or supplemented with sparse depth), the source of depth annotations in our fine-tuning process to learn the inpainting task, the probability p that we mask out the sparse depth input, whether we use warped masks during the fine-tuning process that mimic characteristic inpainting patterns in scene generation, and if the final depth prediction is aligned with the existing point cloud through a global scale-and-shift operation.

Masking Strategy We observe that using warped masks that mimic the characteristic inpainting patterns that occur when changing view points is critical to yield a high-performing depth inpainting model for the task of depth inpainting in a scene generation setting. A naive patch-based masking approach, where regions of varying size are randomly masked, produces inferior results.

Zero Sparse Depth Input In our fine-tuning setup, we can set a probability p to zero out the sparse depth input to the model, effectively reverting to the monocular depth estimation task. We observe that in this setting, our model is highly competitive with the original ZoeDepth model (see Table 2), suggesting the inpainting ability has been bolted onto the network in our fine-tuning setup with only minor degradation of the original task. However, this setting is not able to reach the performance of a network that is given sparse depth input. This supports our hypothesis that adding sparse depth information of the existing scene leads to predictions that are geometrically more coherent and overall more faithful to observed depth.

### 6 Conclusion

We introduce a scene geometry evaluation benchmark that allows assessing the geometric quality of scene generation methods. While generated scenes are difficult to evaluate due to the lack of ground-truth data, we provide a rigorous method to uncover geometric inconsistencies. We also propose a general-purpose

depth inpainting model that completes depth maps, highly suitable for the inpainting challenges arising in a scene generation task. Compared to previous methods, this network produces depth predictions that are more consistent with existing scenes. Finally, we showcase the network as part of a 360-degree scene generation pipeline, where it seamlessly stitches newly hallucinated frames to existing ones, allowing it to generate compelling and immersive 3D scenes.

Ethics. For further details on ethics, data protection, and copyright please see https://www.robots.ox.ac.uk/~vedaldi/research/union/ethics.html.

Acknowledgements. P. E., A. V., I. L., and C.R. are supported by ERC-UNIONCoG-101001212. P.E. is also supported by Meta Research. I.L. and C.R. also receive support from VisualAI EP/T028572/1.

### References

- 1. Bautista, M.Á., Guo, P., Abnar, S., Talbott, W., Toshev, A., Chen, Z., Dinh, L., Zhai, S., Goh, H., Ulbricht, D., Dehghan, A., Susskind, J.M.: GAUDI: A neural architect for immersive 3d scene generation abs/2207.13751 (2022) 3
- 2. Bhat, S.F., Birkl, R., Wofk, D., Wonka, P., Müller, M.: Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288 (2023) 3, 11, 13
- 3. Boulch, A., Puy, G., Marlet, R.: Fkaconv: Feature-kernel alignment for point cloud convolution. In: Proceedings of the Asian Conference on Computer Vision (2020) 3
- 4. Cai, S., Chan, E.R., Peng, S., Shahbazi, M., Obukhov, A., Gool, L.V., Wetzstein, G.: DiffDreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models. In: ICCV (2023) 3
- 5. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., De Mello, S., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., et al.: Efficient geometry-aware 3d generative adversarial networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16123–16133 (2022) 3
- 6. Chan, E.R., Monteiro, M., Kellnhofer, P., Wu, J., Wetzstein, G.: pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5799–5809 (2021) 3
- 7. Chen, Y., Yang, B., Liang, M., Urtasun, R.: Learning joint 2d-3d representations for depth completion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10023–10032 (2019) 3
- 8. Cheng, X., Wang, P., Guan, C., Yang, R.: Cspn++: Learning context and resource aware convolutional spatial propagation networks for depth completion. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 34, pp. 10615–10622

(2020) 3

- 9. Cheng, X., Wang, P., Yang, R.: Depth estimation via affinity learned with convolutional spatial propagation network. In: Proceedings of the European conference on computer vision (ECCV). pp. 103–119 (2018) 3
- 10. Chung, J., Lee, S., Nam, H., Lee, J., Lee, K.M.: Luciddreamer: Domain-free generation of 3d gaussian splatting scenes abs/2311.13384 (2023) 3, 4, 11, 13
- 11. Cohen-Bar, D., Richardson, E., Metzer, G., Giryes, R., Cohen-Or, D.: Set-thescene: Global-local training for generating controllable nerf scenes (2023) 3
- 12. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5828–5839 (2017) 8, 13, 14
- 13. Deng, K., Liu, A., Zhu, J.Y., Ramanan, D.: Depth-supervised nerf: Fewer views and faster training for free. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12882–12891 (2022) 3
- 14. Eigen, D., Puhrsch, C., Fergus, R.: Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems 27

(2014) 7

- 15. Gu, J., Liu, L., Wang, P., Theobalt, C.: Stylenerf: A style-based 3d-aware generator for high-resolution image synthesis. arXiv preprint arXiv:2110.08985 (2021) 3
- 16. Hessel, J., Holtzman, A., Forbes, M., Bras, R.L., Choi, Y.: CLIPScore: a referencefree evaluation metric for image captioning. In: EMNLP (2021) 7, 11

- 17. Höllein, L., Cao, A., Owens, A., Johnson, J., Nießner, M.: Text2Room: Extracting textured 3D meshes from 2D text-to-image models. In: ICCV (2023) 3, 4, 11
- 18. Hu, R., Ravi, N., Berg, A.C., Pathak, D.: Worldsheet: Wrapping the world in a 3d sheet for view synthesis from a single image. In: ICCV (2021) 3
- 19. Huynh, L., Nguyen, P., Matas, J., Rahtu, E., Heikkilä, J.: Boosting monocular depth estimation with lightweight 3d point fusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12767–12776 (2021) 3
- 20. Jain, A., Tancik, M., Abbeel, P.: Putting nerf on a diet: Semantically consistent few-shot view synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5885–5894 (2021) 3
- 21. Kam, J., Kim, J., Kim, S., Park, J., Lee, S.: Costdcnet: Cost volume based depth completion for a single rgb-d image. In: European Conference on Computer Vision. pp. 257–274. Springer (2022) 3, 11, 13
- 22. Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R.C., Schindler, K.: Repurposing diffusion-based image generators for monocular depth estimation. arXiv preprint arXiv:2312.02145 (2023) 9, 13
- 23. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (2023) 3, 9
- 24. Lei, J., Tang, J., Jia, K.: RGBD2: generative scene synthesis via incremental view inpainting using RGBD diffusion models. In: CVPR (2023) 3
- 25. Li, J., Feng, Z., She, Q., Ding, H., Wang, C., Lee, G.H.: MINE: towards continuous depth MPI with nerf for novel view synthesis. In: ICCV (2021) 3
- 26. Li, Z., Wang, Q., Snavely, N., Kanazawa, A.: Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In: European Conference on Computer Vision. pp. 515–534. Springer (2022) 3
- 27. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 300–309 (2023) 3
- 28. Liu, A., Tucker, R., Jampani, V., Makadia, A., Snavely, N., Kanazawa, A.: Infinite nature: Perpetual view generation of natural scenes from a single image. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14458–14467 (2021) 3, 4
- 29. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero1-to-3: Zero-shot one image to 3d object. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9298–9309 (2023) 3
- 30. Melas-Kyriazi, L., Laina, I., Rupprecht, C., Vedaldi, A.: Realfusion: 360deg reconstruction of any object from a single image. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8446–8455 (2023) 3
- 31. Mildenhall, B., Srinivasan, P.P., Cayon, R.O., Kalantari, N.K., Ramamoorthi, R., Ng, R., Kar, A.: Local light field fusion: practical view synthesis with prescriptive sampling guidelines 38(4) (2019) 3
- 32. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021) 3
- 33. Nathan Silberman, Derek Hoiem, P.K., Fergus, R.: Indoor segmentation and support inference from rgbd images. In: ECCV (2012) 9

- 34. Nguyen-Phuoc, T., Li, C., Theis, L., Richardt, C., Yang, Y.L.: Hologan: Unsupervised learning of 3d representations from natural images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7588–7597 (2019) 3
- 35. Niemeyer, M., Geiger, A.: Giraffe: Representing scenes as compositional generative neural feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11453–11464 (2021) 3
- 36. Ouyang, H., Heal, K., Lombardi, S., Sun, T.: Text2immersion: Generative immersive scene with 3d gaussians abs/2312.09242 (2023) 3, 4, 11
- 37. Park, J., Joo, K., Hu, Z., Liu, C.K., So Kweon, I.: Non-local spatial propagation network for depth completion. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16. pp. 120–136. Springer (2020) 3
- 38. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022) 3
- 39. Qian, G., Mai, J., Hamdi, A., Ren, J., Siarohin, A., Li, B., Lee, H.Y., Skorokhodov,

I., Wonka, P., Tulyakov, S., et al.: Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843 (2023) 3

- 40. Raj, A., Kaza, S., Poole, B., Niemeyer, M., Ruiz, N., Mildenhall, B., Zada, S., Aberman, K., Rubinstein, M., Barron, J., et al.: Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508 (2023) 3
- 41. Roberts, M., Ramapuram, J., Ranjan, A., Kumar, A., Bautista, M.A., Paczan, N., Webb, R., Susskind, J.M.: Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10912–10922 (2021) 8, 13, 14
- 42. Rockwell, C., Fouhey, D.F., Johnson, J.: PixelSynth: Generating a 3D-consistent experience from a single image. In: ICCV (2021) 3
- 43. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 3, 5
- 44. Sargent, K., Li, Z., Shah, T., Herrmann, C., Yu, H.X., Zhang, Y., Chan, E.R., Lagun, D., Fei-Fei, L., Sun, D., et al.: Zeronvs: Zero-shot 360-degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994 (2023) 3
- 45. Saxena, S., Herrmann, C., Hur, J., Kar, A., Norouzi, M., Sun, D., Fleet, D.J.: The surprising effectiveness of diffusion models for optical flow and monocular depth estimation. arXiv preprint arXiv:2306.01923 (2023) 3
- 46. Seitz, S.M., Dyer, C.R.: Photorealistic scene reconstruction by voxel coloring. In: CVPR (1997) 3
- 47. Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X.: Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023) 3
- 48. Shih, M., Su, S., Kopf, J., Huang, J.: 3d photography using context-aware layered depth inpainting. In: CVPR (2020) 3
- 49. Sitzmann, V., Thies, J., Heide, F., Nießner, M., Wetzstein, G., Zollhöfer, M.: Deepvoxels: Learning persistent 3d feature embeddings. In: CVPR (2019) 3
- 50. Stan, G.B.M., Wofk, D., Fox, S., Redden, A., Saxton, W., Yu, J., Aflalo, E., Tseng, S.Y., Nonato, F., Muller, M., et al.: Ldm3d: Latent diffusion model for 3d. arXiv preprint arXiv:2305.10853 (2023) 3
- 51. Szymanowicz, S., Rupprecht, C., Vedaldi, A.: Splatter image: Ultra-fast single-view 3d reconstruction. arXiv preprint arXiv:2312.13150 (2023) 3

- 52. Trevithick, A., Yang, B.: GRF: learning a general radiance field for 3d representation and rendering. In: ICCV (2021) 3
- 53. Tulsiani, S., Tucker, R., Snavely, N.: Layer-structured 3d scene inference via view synthesis. In: ECCV (2018) 3
- 54. Wang, S., Suo, S., Ma, W.C., Pokrovsky, A., Urtasun, R.: Deep parametric continuous convolutional neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2589–2597 (2018) 3
- 55. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems 36 (2024) 3
- 56. Wiles, O., Gkioxari, G., Szeliski, R., Johnson, J.: Synsin: End-to-end view synthesis from a single image. In: CVPR (2020) 3
- 57. Wu, Z., Li, Y., Yan, H., Shang, T., Sun, W., Wang, S., Cui, R., Liu, W., Sato, H., Li, H., Ji, P.: BlockFusion: Expandable 3D scene generation using latent tri-plane extrapolation (2024) 3
- 58. Xiang, J., Yang, J., Huang, B., Tong, X.: 3D-aware image generation using 2D diffusion models abs/2303.17905 (2023) 3
- 59. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4578–4587 (2021) 3
- 60. Yu, H., Duan, H., Hur, J., Sargent, K., Rubinstein, M., Freeman, W.T., Cole, F., Sun, D., Snavely, N., Wu, J., Herrmann, C.: Wonderjourney: Going from anywhere to everywhere abs/2312.03884 (2023) 3, 4, 11
- 61. Zhang, J., Li, X., Wan, Z., Wang, C., Liao, J.: Text2nerf: Text-driven 3d scene generation with neural radiance fields abs/2305.11588 (2023) 3, 4
- 62. Zhang, Y., Guo, X., Poggi, M., Zhu, Z., Huang, G., Mattoccia, S.: Completionformer: Depth completion with convolutions and vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18527–18536 (2023) 3
- 63. Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., Torralba, A.: Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence 40(6), 1452–1464 (2017) 9
- 64. Zhu, Z., Feng, X., Chen, D., Bao, J., Wang, L., Chen, Y., Yuan, L., Hua, G.: Designing a better asymmetric vqgan for stablediffusion. arXiv preprint arXiv:2306.04632

(2023) 10

