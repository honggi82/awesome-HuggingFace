### InseRF: Text-Driven Generative Object Insertion in Neural 3D Scenes

Mohamad Shahbazi1,2∗† Liesbeth Claessens1∗ Michael Niemeyer2 Edo Collins2 Alessio Tonioni2 Luc Van Gool1 Federico Tombari2

1ETH Z¨urich 2Google Z¨urich

Method inputs 3D Generative Object Insertion

# arXiv:2401.05335v1[cs.CV]10Jan2024

[Figure 1]

[Figure 2]

[Figure 3]

Put a garden gnome on the grass

[Figure 4]

Text prompt

(a) Instruct-NeRF2NeRF

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

3D scene

A garden gnome

(b) InseRF (ours)

Text prompt + Reference bounding box

Figure 1. Generative insertion of objects in 3D scenes: while Instruct-NeRF2NeRF allows for altering the overall style of scenes or performing edits with a strong positional prior (such as placing a mustache on a face), it fails at inserting objects in arbitrary locations due to the multiview inconsistency of the edits. Our method, in contrast, is multiview consistent by design and can insert new objects in user-specified locations

this study. Specifically, we propose grounding the 3D object insertion to a 2D object insertion in a reference view of the scene. The 2D edit is then lifted to 3D using a single-view object reconstruction method. The reconstructed object is then inserted into the scene, guided by the priors of monocular depth estimation methods. We evaluate our method on various 3D scenes and provide an in-depth analysis of the proposed components. Our experiments with generative insertion of objects in several 3D scenes indicate the effectiveness of our method compared to the existing methods. InseRF is capable of controllable and 3D-consistent object insertion without requiring explicit 3D information as input. Please visit our project page at https://mohamadshahbazi.github.io/inserf.

#### Abstract

We introduce InseRF, a novel method for generative object insertion in the NeRF reconstructions of 3D scenes. Based on a user-provided textual description and a 2D bounding box in a reference viewpoint, InseRF generates new objects in 3D scenes. Recently, methods for 3D scene editing have been profoundly transformed, owing to the use of strong priors of text-to-image diffusion models in 3D generative modeling. Existing methods are mostly effective in editing 3D scenes via style and appearance changes or removing existing objects. Generating new objects, however, remains a challenge for such methods, which we address in

* Equal contribution. † Work was mainly done during an internship at Google.

Correspondence: Mohamad Shahbazi (mshahbazi@vision.ethz.ch) .

#### 1. Introduction

Recent advances in the areas of novel view synthesis and generative modeling have led to substantial progress in methods for the generation and manipulation of 3D assets and scenes. Diffusion models [12, 33] and their integration with neural reconstruction methods, such as neural radiance fields (NeRFs) [20], have enabled the development of powerful 3D generative models for various applications, including text-to-3D [14, 28, 40], single-image-to-3D [15– 18, 29], 3D shape texturing [32], and 3D editing [10].

In the particular case of 3D scene editing, recent methods have shown remarkable promise in modifying the style and appearance of real-world scene representations based on textual and spatial guidance. Currently, models that are capable of direct 3D generation and editing are mainly limited to simple and object-centric scenes [3, 5, 16, 25]. As a result, for more complex scenes, the majority of the recent editing methods rely on performing edits on different views of the scenes using 2D editing models. One of the most prominent works in 3D scene editing is the recently proposed Instruct-NeRF2NeRF [10], an iterative method that performs multi-view edits on the NeRF reconstruction of 3D scenes from textual instructions. Although achieving impressive results, Instruct-NeRF2NeRF is mainly limited to editing the style and appearance of scenes. When prompted to with localized edits or geometry manipulations (such as object removal or insertion) at specified locations, Instruct-NeRF2NeRF often fails to perform the desired edits. This is mainly due to the 3D inconsistency of 2D edits across viewpoints and the lack of proper spatial control.

Recent works have aimed at the 3D-consistent [9, 46] and localized editing [22, 38, 46] of 3D scenes. In addition, several studies have specifically tackled object removal and inpainting in 3D scene representations [21, 23, 39, 41, 42]. However, generating and inserting new objects in scenes in a 3D-consistent way remains an open problem and is mainly limited to cases where edits are strongly constrained by spatial priors (e.g. putting a hat on a head or a mustache on a face). Therefore, in this work, we specifically focus on generative object insertion in 3D scenes, in a way that is consistent across multiple views and placed in arbitrary positions.

Generative object insertion in 3D scenes using 2D generative models is a particularly challenging task, as it requires 3D-consistent generation and placement of objects in different viewpoints. A simplistic approach is to separately generate the desired objects using 3D shape generation models [28, 40] and insert them into the scene using 3D spatial information. However, such an approach requires the accurate location, orientation, and scale of the object in 3D, a non-trivial requirement, especially when in contact with other objects in the scene. Moreover, scene-independent generation of the objects can lead to a mismatch between the style and appearance of the scene and the inserted ob-

jects. In this work, we propose a method capable of sceneaware generation and insertion of objects in 3D scenes using the textual description of the objects and a single-view 2D bounding box as spatial guidance.

To circumvent multi-view inconsistencies in appearance and location, the scene-independent generation, and the need for explicit 3D spatial information, we propose grounding the 3D insertion by a 2D view of the object inserted in one reference view of the scene. Given a 3D reconstruction of the scene, we first render a reference view. Then, conditioned on a text prompt and a 2D bounding box, we use an image editing method to add the target object in the reference view. The generated object is then lifted to 3D using a single-view-to-3D object reconstruction method [15–18, 29]. To place the object in 3D, we propose using the estimated depth of the object in the reference view. After inserting the object in the scene, we perform an optional refinement of the fused scene and objects using the proposed method in Instruct-NeRF2NeRF [10].

To evaluate the proposed method, we apply our method to several 3D scenes. Our experiments indicate the ability of the proposed method to insert diverse objects in 3D scenes without the need for explicit 3D spatial guidance. To summarize our contributions:

- • We address the task of consistent generative object insertion in 3D scenes based on a textual description and a single-view 2D bounding box, which is beyond the capability of the existing 3D scene editing methods
- • We propose a novel method, based on grounding the insertion using a reference 2D edit, which is capable of 3Dconsistent object insertion without requiring explicit information for the 3D placement.
- • Through our experiments and visualizations, we show the advantage of the proposed method in generative object insertion in comparison to the existing baselines.

#### 2. Related Works

Language-based 3D scene editing: 3D scene editing has recently undergone a considerable transformation by incorporating the strong priors of 2D text-conditioned diffusion models into 3D generative modeling [9, 10, 22, 26, 38, 43, 46]. Instruct-NeRF2NeRF [10] proposes an iterative method for 3D scene editing, where different viewpoints of the scene are edited using a text-based 2D editing model and used to fine-tune the scene’s NeRF representation. Although highly effective with modifying the existing content, Instruct-NeRF2NeRF often struggles with 3D consistent and localized edits, especially when instructed to remove objects or create new ones in the scene [9, 39]. To address the view consistency of edits, ViCA-NeRF [9] proposes a method based on a viewpoint-correspondence regularization and a strategy to align the latent space of edited and unedited viewpoints. DreamEditor [46] tackles

the 3D consistency by adapting the 2D diffusion model to the multi-view images of the scene using DreamBooth [34]. DreamEditor additionally identifies a 3D region of interest for localized editing of an existing object based on textimage semantic similarity. The method in [22] addresses localized editing differently by obtaining a 3D relevance field for the edits based on the discrepancy between the predictions of the diffusion model with and without instruction conditioning. These methods, despite the improvements, remain limited in their ability to generate new objects, often struggling with cases where a strong spatial prior for the placement of the object does not exist.

Removing objects from 3D scenes: another direction recently explored in the area of 3D scene editing is 3Dconsistent removal and inpainting of objects in the scenes. Some studies assume having multi-view masks of the target object [21, 41]. These multi-view masks, along with other strategies are used to determine the regions to inpaint in different rendered viewpoints. Other studies assume userprovided single-view annotations of the objects and propose approaches to automatically obtain multi-view masks from the reference one [23, 39, 42]. However, such approaches for extracting multi-view masks do not transfer to the task of object insertion, as they rely on the assumption that the objects already exist in the scene.

Generative object insertion: In contrast to scene stylization and object removal, generating objects in 3D scenes is not well-explored in the existing works. The inpainting method proposed in [21], although mainly designed and evaluated for object removal, has been showcased for examples of object insertion as well. To do so, the authors assume multi-view masks of the object are provided, and they propose a method to propagate a single-view inpainting to other viewpoints. However, in addition to requiring multiview masks as input, the proposed method is mainly limited to forward-facing scenes [21]. FocalDreamer [13] is a concurrent work proposed for adding editable parts to a base 3D shape. Provided with a text prompt and the rough 3D placement of the target edits, FocalDreamer applies score distillation [28] to add the desired parts to the base shape. Although achieving compelling results, FocalDreamer requires user-provided 3D regions (rotation, translation, and scale), and its generalization beyond base shapes to complex 3D scenes is not investigated. Language-driven Object Fusion [37] is another concurrent work that aims at fusing an existing or generated foreground object with a background 3D scene. The authors first adopt a 2D diffusion model for view synthesis from the scene and the object using DreamBooth [34]. Then, conditioned on a userprovided 3D bounding box, the authors propose a poseconditioned dataset update strategy for the training of scene NeRF containing the object. The proposed fusion strategy requires users to provide an exact 3D bounding box. In con-

trast to the existing language-driven object insertion methods, our approach works well with both forward-facing and 360 scenes, and it only requires a rough 2D bounding box from one rendered view of the scene, making it more suitable for real-world applications.

#### 3. Method

Our method takes as input a NeRF reconstruction of a 3D scene, a textual description of the target object to be inserted, and a 2D bounding box in a reference rendered view of the scene. As output, our method returns a NeRF reconstruction of the same scene containing the generated target 3D object placed in a location guided by the 2D bounding box. It is noteworthy that our method only requires a rough bounding box, as we rely on the priors of the diffusion models for the exact 2D positioning.

The proposed method consists of five main steps: 1) a 2D view of the target object is created in a chosen reference view of the scene based on a text prompt and a 2D bounding box; 2) a 3D object NeRF is reconstructed from the generated 2D view in the reference image; 3) the 3D placement of the object in the scene is estimated with the help of monocular depth estimation; 4) the object and scene NeRFs are fused into a single scene containing the object in the estimated placement; 5) optionally, a refinement step is applied to the fused 3D representation to improve the insertion further. Fig. 2 shows an overview of the proposed pipeline. In the following, we discuss each step in more detail.

##### 3.1. Preliminaries

Diffusion Models Diffusion models are a type of generative model that maps Gaussian noise to highly realistic and diverse samples. They consist of (1) a forward process that maps data samples x0 to noise xT, and (2) a backward process that creates data samples from noise.

The forward process consists of T steps t ∈ [0,T − 1]:

q(xt+1|xt) = N(xt| 1 − βtxt−1,βtI), (1) with variances βt chosen such that the noise xT ∼ N(0,I).

The backward process, which is used to generate data samples from Gaussian noise and optionally an additional conditioning signal, has the following shape:

q(xt−1|xt) = N(xt−1 | µθ(xt,t,c),

(xt,t,c)), (2)

θ

where the parameters of the backward/denoising distributions are predicted by a U-Net, whose weights θ are optimized by increasing the likelihood of the data samples. Diffusion models can be conditioned on different types of signals, such as images or text, as well as masks, and can be extended for different tasks, such as 2D editing [4, 11, 45] and inpainting [1, 19].

Object NeRF

Updated Scene NeRF

[Figure 9]

SingleImageto

3Dmethod

[Figure 10]

Annotated Reference View

Edited Reference View

[Figure 11]

[Figure 12]

2Dediting

Refinement

3DObject

method

[Figure 13]

Insertion

[Figure 14]

A cup

Estimation

Depth

Figure 2. Overview of the proposed method. Given a single reference view annotated with a 2D bounding box and a text prompt describing the object to be inserted, a 2D edit is generated portraying a view of the object. This 2D edit is then warped to a 3D model of the object and placed into the scene using the procedure described in section 3.4. After the 3D placement, the object and scene representations are fused as described in section 3.5. Finally, an optional refinement can be performed to further improve the appearance.

Neural Radiance Fields NeRFs are a novel view synthesis method trained on a set of posed images by minimizing the photometric loss between ground truth and rendered pixels. A key aspect of NeRF is that pixel colors are not predicted directly. Instead, the density σ and color c at 3D points in space are predicted by a neural function fϕ. Using these predictions, the pixel color corresponding to a ray ⃗r = (⃗o,⃗d) with origin ⃗o and viewing direction ⃗d can be composed through volumetric rendering. To do so, a set of points along the ray ti = ⃗o + t⃗d is sampled, splitting the ray into a set of intervals δi = (ti,ti+1]. The pixel color of the ray can then be composed as:

ing text-guided editing methods often struggle with localized 2D object insertions when only receiving spatial guidance from text prompts [27, 44, 45]. To ensure localized

- 2D insertion within the input bounding box, we opt for a mask-conditioned inpainting method as our 2D generative model. Specifically, we choose Imagen [35], a powerful text-to-image diffusion model, and further adapt it to maskconditioning by using RePaint [19], a method for maskconditioned inpainting with diffusion models.

- 3.3. Single-View Object Reconstruction

After obtaining the reference edit, we extract the 2D view of the object generated within the bounding box and create a 3D reconstruction of it. To do so, we propose exploiting the recent paradigm of single-view object reconstruction using 3D-aware diffusion models [15–18, 29]. Such reconstruction methods are typically trained on large-scale 3D shape datasets, such as Objaverse [8] and therefore contain strong priors over the geometry and appearance of 3D objects. We use the recently proposed SyncDreamer [17] for our object reconstruction, as it offers a good trade-off between reconstruction quality and efficiency.

- 3.4. 3D Placement

N

wici, (3)

C(r) ≈

i=1

wi = Ti(1 − exp(−σiδi)), (4)

- i−1
- j=1

σjδj). (5)

Ti = exp(−

In the above equations:

###### (σi, ci) = f (γ (ti); ϕ), (6)

where the positional encoding function γ and the location of the samples ti depend on the NeRF variant being used.

Depth Estimation: The reference 2D bounding box constrains the 3D location of the target object to a frustum in the scene. To determine the location of the object in the 3D frustum, we propose using the prior from monocular depth estimation methods. We apply MiDaS [31] on the edited reference image to estimate the depth of the object with respect to the reference camera. As MiDaS provides non-metric depth measurements, we perform an extra depth alignment between the estimated depth of the edited reference view and the reference depth rendered from

##### 3.2. Editing the Reference View

Our editing pipeline starts by choosing one rendered view of the scene as the reference and inserting a 2D view of the target object based on a user-provided text prompt and a 2D bounding box. The reference view is used to ground the 3D insertion by providing a reference appearance and location. Through empirical experiments, we find the additional use of the bounding box important, as the exist-

the scene NeRF by estimating a global scale and shift between the reference and estimated depth maps. Specifically, to make the alignment more accurate around the object area, we estimate the alignment parameters using weighted least-square estimation, where measurements are inversely weighted based on their distance to the center of the object bounding box (details are provided in the supplementary). After the alignment, we use the depth of the center pixel d in the object bounding box as a rough estimate of the object’s center in the frustum, which will be further optimized in the next step.

Scale and Distance Optimization: Using the estimated depth d as the distance of the object’s center from the reference camera helps with resolving the scale-depth ambiguity of the target 3D object, but it is not accurate enough to closely match the original edit. Additionally, singleview reconstruction methods like SyncDreamer (discussed in Sec. 3.3) are trained to generate multi-view images from fixed camera distance r′ and focal length f′. In general, as these parameters are different from those of the reference camera, the reconstructed object NeRF appears with a different scale in the reference view once placed at the estimated distance. Therefore, we propose an additional optimization step for the scale and the distance of the object with two constraints: 1) the object must reside at the estimated depth; 2) the rendered view of the object in the reference camera should match the initial edit in scale and appearance. To ensure a proper initial state for the optimization we initialize our scale s and object’s distance as:

r′ f

d f′.

(7) r0 = s0.l + d (8)

s0 =

where s0 and r0 are the initial object scale and distance, and l is the distance of the 3D point corresponding to the center of the bounding box from the origin of the object NeRF’s coordinate system. Given a 3D point P⃗′ in the original object NeRF’s coordinate system, the corresponding 3D point P⃗ in the scaled coordinate system is obtained as:

P⃗ = sP⃗′ (9)

To obtain the optimized scale s∗ and distance r∗, we optimize the Mean Square Error (MSE) between the groundtruth 2D edit IG and the image IR rendered using the new parameters:

r∗,s∗ = arg min

###### ||IG − IR||2 (10)

r,s

Fig. 5 in our ablation study visualizes the effect of scale and distance optimization.

Rotation and Translation: After obtaining the scale and distance of the object from the reference camera, we proceed to estimate the placement of the object in the scene

by estimating its 3D rotation and translation with respect to the camera coordinate system. The origin of the object in the scene’s coordinate system is obtained as the point along the ray from the reference camera center passing through the center of the bounding box at the desired distance. To obtain the 3D rotation, we align the x-axis of the object’s coordinate system to the vector pointing to the reference camera center from the object’s origin.

##### 3.5. Scene and Object Fusion

Once the location and the orientation of the 3D object in the scene are known, we fuse the NeRF representations of the object and scene to be able to render multi-view images of the scene containing the target object. Given a viewpoint, we transform the rays to the coordinate systems of the scene and the object. Each NeRF representation is applied to the corresponding transformed rays to predict the color and density of the object and scene at each 3D position. To render a viewpoint using the predictions of the two NeRFs, we follow the proposed strategy in [36], where the density σi and color ci at each 3D point i across a ray in the fused representation are defined as:

σi = σis + σio, (11) ci =

σiscsi + σiocoi σis + σio

, (12)

where σis and csi are the density and the color of the corresponding sample the scene NeRF, and σio and coi are those of the one in the object NeRF. To be able to use such formulation in our method for merging the object and the scene, it is crucial to take the scaling of the object’s coordinate system into account. Going back to the approximation of the volumetric rendering integration, discussed in Sec. 3.1, in equation 4, σiδi can be seen as the Riemann approximation of the area under the density curve across the ray at interval δi. Simply replacing σi in equation 4 with the definition in equation 11 results in an inaccurate estimation of the area under the density curve for the merged representation, as the intervals between every two consecutive samples across the rays are not equal between scene and object coordinate systems due to the scaling of the object coordinate system (discussed in section 3.4):

δis = s∗ · δio, (13)

δis and δio are the intervals in the scene and object NeRFs, respectively, and s∗ is the optimized scale obtained in section 3.4. To compensate for the scaling of the intervals, we modify equations 11 and 12 as:

σio s∗ (14)

σi = σis +

σiscsi + σiocoi/s∗ σis + σio/s∗ , (15)

ci =

As we also show in Fig. 6 in our ablation study, the proposed modification is necessary for the correct rendering of the fused NeRFs.

##### 3.6. Refinement

As the final step in our pipeline, we optionally refine the fused scene and object to improve upon the imperfections introduced in the initial reference edit or the single-view reconstruction. To do so, we adapt the iterative refinement proposed in Instruct-NeRF2NeRF [10] to our setup. First, a set of images is rendered from different viewpoints of the fused NeRF. Then the sampled views are further refined using the 2D diffusion model and added to the optimization of the NeRF consecutively. An important difference between our refinement and Instruct-NeRF2NeRF is that we can obtain multi-view object masks for free from the inserted object to restrict the refinements to the object region. Additionally, in contrast to Instruct-NeRF2NeRF, as the location of the object is known in our refinement step, we adjust our camera trajectory to revolve around the object. We also arrange the sampled viewpoints such that more frontal views are edited and used for NeRF optimization earlier. We find such adjustments to increase the capability of our refinement step. The effect of the proposed refinement is visualized in Fig. 7 of our ablation study.

#### 4. Experiments

In this section, we explain our training and evaluation procedures in more detail. Moreover, we provide the results of our evaluation and comparison with baselines. Finally, we provide an ablation study and analysis of different components of the proposed method.

##### 4.1. Experimental Details

Implementation Details: For the NeRF representation of objects and scenes, we use MipNeRF-360 [2] adapted to the hash grids introduced in Instant-NGP [24]. For a more exhaustive description of the implementation of our method, we kindly refer the reader to the supplementary materials.

Datasets: We evaluate our method on a subset of real indoor and outdoor scenes from datasets proposed in MipNeRF-360 [2] and Instruct-NeRF2NeRF [10].

Baselines: In our evaluation, we compare the proposed method to the following baselines:

- • Instruct-NeRF2NeRF (I-N2N) [10]: We choose I-N2N as our main baseline, as it is a recent and well-established method for 3D scene editing.
- • Multi-View Inpainting (MV-Inpainting): We propose another baseline that follows the refinement strategy in Instruct-NeRF2NeRF, but is additionally provided with

Existing works more related to our method mostly require extra inputs (e.g. 3D boxes) or do not currently provide an implementation [13, 21, 37].

accurate multi-view masks for the target object. It is worth emphasizing that, in contrast, our methods only require a rough 2D bounding box in a single reference view.

More details on the implementation of our baselines are provided in the supplementary material.

##### 4.2. Visual Results and Comparisons

To assess the ability of the proposed method in generative object insertion, we provide visual examples of applying our method to different 3D scenes in Fig. 3. As shown, our method can insert 3D-consistent objects in the scenes. Especially noteworthy is the ability of our method to insert objects on different surfaces, a challenging task in the absence of exact 3D placement information.

In Fig. 4, we provide a visual comparison with the baselines discussed in Sec. 4.1. Attempting to insert new objects in the scene using I-N2N often results in global changes in the scene and modifying existing objects toward the target instead of creating new ones (note how I-N2N changes the Lego truck in 4a toward a mug and the items on the kitchen counter 4b toward a panettone). Using multi-view masks in the MV-Inpainting baseline helps with limiting the 2D edits to the object region and provides strong spatial guidance. However, 2D edits remain inconsistent from different viewpoints. Therefore, using the edits to optimize the NeRF representation results in 3D floaters and failure to generate the target object in a 3D consistent way. In contrast, our method is capable of localized modification of the scene and of inserting 3D-consistent objects in 3D using only one singleview bounding box as spatial guidance. More visual results are provided in the supplementary material.

##### 4.3. Ablation and Analysis

Scale and radius optimization: In Fig. 5, we provide a visual ablation demonstrating the importance of the scale and radius optimization proposed in 3.4, where we compare the placement of the object in the scene using the initial estimation according to Eq. 7 and 8 and placement with the extra optimization. As can be seen, the proposed initial estimation would only result in a rough and inaccurate placement of the object. With the proposed optimization, our method can insert objects with the scale and depth matching those of the reference view.

Object density scaling: In Sec. 3.5, we proposed an adapted strategy for fusing the NeRF representations of the scene and the object that takes the scaling of the object into account. In Fig. 6, we visualize the importance of our adapted formulation for accurate rendering of the objects inserted in the scene.

Refinement: in Sec. 3.6, we proposed an optional refinement step after inserting the objects in the scenes. Fig 7 shows examples of the effect of the refinement. As shown, the additional refinement can improve some of the details of

Reference View Reference Edit Edited Neural Scene Representation

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

A panettone

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

A cup

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

A pouffe

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

A mug

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

| |
|---|

A duck

Figure 3. Examples of using InseRF to insert an object into the neural representation of different indoor and outdoor scenes.

Original Scene I-N2N MV-Inpainting InseRF (ours)

Original Scene I-N2N MV-Inpainting InseRF (ours)

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

(a) A mug on the table

(b) A panettone on the tray

- Figure 4. Qualitative comparison of object insertion with different methods. I-N2N modifies existing objects instead of inserting a new object, and the inpainting baseline fails to create geometry at the desired location. Our method, in contrast, can insert new 3D-consistent objects at the desired location.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

NoReﬁnementReﬁnementNoReﬁnementReﬁnementNoReﬁnementReﬁnementNoReﬁnementReﬁnement

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

No Scale/Distance Optimization

Ground Truth 2D Edit

Scale/Distance Optimization

[Figure 86]

[Figure 87]

[Figure 88]

5

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

- Figure 5. Visualization of the effect of scale optimisation on object insertion. The placement of objects is more realistic and faithful to the original edit when performing scale/distance optimization to improve the alignment.

[Figure 100]

NoDensityScalingDensityScaling

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- Figure 6. Visualization of the effect of scaling the densities when fusing the object and scene representation. When the re-scaling of the object NeRF is not accounted for in the volumetric rendering, the object is not properly displayed in the synthesized views.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Figure 7. Visualization of the effect of refinement on object insertion. Our refinement step can add additional texture details and lighting effects.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

based spatial guidance can lead to improved performance of our whole pipeline. Lastly, integrating the concurrently proposed view-consistent editing methods [9, 37] and existing approaches for scene-consistent shadowing and harmonization (e.g. [6, 7]) with our refinement step may bring further improvements to the quality and realism of the insertions.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

#### 5. Conclusion

We introduced InseRF, a method specifically designed for generative object insertion in 3D scenes. InseRF takes as input a textual description of the desired object, as well as a 2D bounding box in a single reference viewpoint of the scene. Based on the provided inputs, InseRF generates an object in the 3D scene in a 3D consistent way. To do so, InseRF relies on the priors of 2D diffusion models and singleview object reconstruction methods. The proposed method includes different steps necessary to integrate such methods for the task of in-scene object generation. Through evaluations and visualizations on different 3D scenes, we showed the ability of InseRF in the 3D-consistent generation of objects in the scene without requiring explicit 3D placement information.

the inserted objects, such as the lighting and the texture.

##### 4.4. Limitations and Future Work

Our method is a general pipeline for generative object insertion that is built on top of the existing 2D and 3D generative models and whose parts can be easily swapped. Currently, the performance of our method is limited by the capabilities of the underlying generative models, such as the 2D diffusion model or the single-view object reconstruction method. On the other hand, given our general formulation, future improvements in such models readily transfer to our pipeline.

#### References

Our method provides spatial control using a single-view bounding box, as current 2D editing models struggle with the spatial guidance provided in the text prompts. Exploring methods both capable of localized 2D insertion and text-

[1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision

and Pattern Recognition (CVPR), pages 18208–18218, 2022. 3

- [2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5470–5479, 2022. 6
- [3] Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, Afshin Dehghan, and Josh Susskind. Gaudi: A neural architect for immersive 3d scene generation. arXiv, 2022. 2
- [4] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 3, 12
- [5] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. Efficient geometry-aware 3D generative adversarial networks. In arXiv, 2021. 2
- [6] Jianqi Chen, Yilan Zhang, Zhengxia Zou, Keyan Chen, and Zhenwei Shi. Dense pixel-to-pixel harmonization via continuous image representation. IEEE Transactions on Circuits and Systems for Video Technology, pages 1–1, 2023. 8
- [7] Jianqi Chen, Zhengxia Zou, Yilan Zhang, Keyan Chen, and Zhenwei Shi. Zero-shot image harmonization with generative model prior, 2023. 8
- [8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13142–13153, 2023. 4
- [9] Jiahua Dong and Yu-Xiong Wang. ViCA-neRF: Viewconsistency-aware 3d editing of neural radiance fields. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2, 8
- [10] Ayaan Haque, Matthew Tancik, Alexei Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision,

2023. 2, 6, 11

- [11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. In arXiv preprint arXiv:2208.01626, 2022. 3
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arxiv:2006.11239,

2020. 2

- [13] Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. Focaldreamer: Textdriven 3d editing via focal-fusion assembly, 2023. 3, 6
- [14] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2

- [15] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without pershape optimization, 2023. 2, 4
- [16] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object, 2023. 2
- [17] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Learning to generate multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 4
- [18] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3d: Single image to 3d using cross-domain diffusion,

2023. 2, 4

- [19] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11461–11471, 2022. 3, 4, 11, 12
- [20] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2
- [21] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, and Igor Gilitschenski. Reference-guided controllable inpainting of neural radiance fields. In ICCV, 2023. 2, 3, 6
- [22] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, and Igor Gilitschenski. Watch your steps: Local image and scene editing by text instructions. In arXiv, 2023. 2, 3
- [23] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G. Derpanis, Jonathan Kelly, Marcus A. Brubaker, Igor Gilitschenski, and Alex Levinshtein. SPIn-NeRF: Multiview segmentation and perceptual inpainting with neural radiance fields. In CVPR, 2023. 2, 3
- [24] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1–15, 2022. 6
- [25] Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc Van Gool, and Sergey Tulyakov. Autodecoding latent 3d diffusion models, 2023. 2
- [26] Jangho Park, Gihyun Kwon, and Jong Chul Ye. Ed-nerf: Efficient text-guided editing of 3d scene using latent space nerf, 2023. 2
- [27] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar AverbuchElor, and Daniel Cohen-Or. Localizing object-level shape variations with text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 4

- [28] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv,

2022. 2, 3

- [29] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 2, 4
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 11
- [31] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(3), 2022. 4, 12
- [32] Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. Texture: Text-guided texturing of 3d shapes. arXiv preprint arXiv:2305.16213, 2023. 2
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 2
- [34] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In arXiv preprint arxiv:2208.12242, 2022. 3
- [35] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, 2022. 4, 12
- [36] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. Graf: Generative radiance fields for 3d-aware image synthesis. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 5
- [37] Ka Chun Shum, Jaeyeon Kim, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Language-driven object fusion into neural radiance fields with pose-conditioned dataset updates, 2023. 3, 6, 8
- [38] Hyeonseop Song, Seokhun Choi, Hoseok Do, Chul Lee, and Taehyeong Kim. Blending-nerf: Text-driven localized editing in neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14383–14393, 2023. 2
- [39] Dongqing Wang, Tong Zhang, Alaa Abboud, and Sabine S¨usstrunk. Inpaintnerf360: Text-guided 3d inpainting on unbounded neural radiance fields. arXiv preprint arXiv:2305.15094, 2023. 2, 3
- [40] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and

- diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2
- [41] Silvan Weder, Guillermo Garcia-Hernando, Aron´ Monszpart, Marc Pollefeys, Gabriel Brostow, Michael Firman, and Sara Vicente. Removing objects from neural radiance fields. In CVPR, 2023. 2, 3
- [42] Youtan Yin, Zhoujie Fu, Fan Yang, and Guosheng Lin. Ornerf: Object removing from 3d scenes guided by multiview segmentation with neural radiance fields, 2023. 2, 3
- [43] Lu Yu, Wei Xiang, and Kang Han. Edit-diffnerf: Editing 3d neural radiance fields using 2d diffusion model, 2023. 2
- [44] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. In Advances in Neural Information Processing Systems, 2023. 4
- [45] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, Caiming Xiong, and Ran Xu. Hive: Harnessing human feedback for instructional visual editing. arXiv preprint arXiv:2303.09618, 2023. 3, 4
- [46] Jingyu Zhuang, Chen Wang, Lingjie Liu, Liang Lin, and Guanbin Li. Dreameditor: Text-driven 3d scene editing with neural fields. arXiv preprint arXiv:2306.13455, 2023. 2

#### Appendix

In the appendix, we provide additional visual results, a quantitative evaluation of our method, and an in-depth discussion of the implementation.

#### A. Additional Visual Results

Visual examples: in Fig. 1 and 3 of the main paper, we provided examples of generative object insertion in 3D scenes using our proposed method. Here in Fig. 8, we provide more visual examples showing the ability of our method to generate objects in 3D scenes.

Comparison to the baseline: in Fig. 4 of the main paper, we provided visual comparisons between the proposed method and our baselines (introduced in Sec. 4.1 of the main paper). Fig. 9 here shows more comparisons with the baselines for a better assessment. As depicted, the two compared baselines struggle with creating the target objects in the scene.

Refinement: in Fig. 7 of the main paper, we provided a visual ablation on the impact of the proposed refinement step in Sec. 3.6. Here in Fig. 10, we extend the ablation to more examples. As can be seen, the proposed refinement step can improve the texture and details of the inserted objects, resulting in higher-quality and more realistic insertions. For the details of our refinement step, please refer to Sec. 3.6 of the main paper and Sec. C.6 of this supplementary.

Video visualizations: To better visualize the inserted objects using our method, we additionally provide video visualizations in the supplementary files, showing several examples of our inserted objects, as well as examples of the refinement step.

#### B. Quantitative Evaluation

In addition to the provided visual evaluations, we provide a quantitative evaluation of the proposed method and its comparison with our baselines. Following InstructNeRF2NeRF [10], we evaluate the methods using three different metrics:

- • CLIP Text-Image Similarity (Text-Image): the cosine similarity between the CLIP [30] embeddings of the edit prompt (e.g. ”A blue cup”) and the images rendered from different viewpoints of the edited scene (We exclude the views where the inserted objects are occluded).
- • Directional Text-Image Similarity (Directional): Given a textual description of the original scene (e.g. ”A kitchen counter”) and an edit prompt describing the scene and the edit (e.g. ”A kitchen counter with a mug on top”), this metric measures the similarity of the direction of change from the original scene to the edited one between the image and text CLIP embeddings.
- • Temporal Direction Consistency (Temporal): Given two adjacent rendered viewpoints of original and edited

scenes, this metric measures how much the change of image embeddings between the two viewpoints in the edited scene is consistent with the one in the original scene.

We provide the results of our quantitative evaluation on 8 different edits (5 different scenes) in Tab. 1. All three metrics are based on Cosine similarity, which ranges from -1 to 1. We bring the values between 0 to 1 (the higher the better) for ease of comparison. As depicted, our method effectively outperforms the baselines in the three evaluated metrics. It is worth discussing that, although the provided metrics indicate the advantage of our method over the baselines, we refer the readers to the qualitative results for a better assessment of the evaluated methods. As also highlighted by Instruct-NeRF2NeRF [10]., the metrics above, although helpful, do not fully capture the effectiveness of methods in 3D scene editing. Exploring alternative metrics that better measure such edits would be an important direction for future studies.

#### C. Implementation Details

##### C.1. Inpainting with RePaint

As mentioned in Sec. 3.2 of the main paper, to generate a 2D view of the target object in the reference view, we condition our diffusion model on a bounding box using RePaint [19]. Repaint is a training-free inpainting method for pretrained diffusion models that is capable of adding new content to an image in the regions specified by an arbitrary binary mask. Repaint primarily consists of 2 components: 1.) mask conditioning and 2.) re-sampling.

To enable mask conditioning, in every step t of the diffusion process, RePaint applies a mask-based blending to the output xt−1 as follows:

xt−1 = (1 − M) ⊙ xknownt−1 + M ⊙ xunknownt−1 (16)

where xknownt−1 is sampled using known pixels in the given image, xunknownt−1 is sampled from the model given the previous iteration xt, and M is the binary mask. ⊙ denotes element-wise multiplication. In our setup, we set M to be

the area inside the condition bounding box and xknownt−1 to be noisy versions of the reference image x0 obtained using the forward diffusion process (Eq. 1 in the main paper).

Table 1. Quantitative evaluation of InseRF and its comparison with the baselines on three different metrics proposed in [10]. For ease of comparison, we report the values (cosine similarities ranging from -1 to 1) after bringing them between 0 and 1. Our proposed method effectively outperforms the baselines in all three metrics.

Method Text-Image ↑ Directional ↑ Temporal ↑

I-N2N [10] 0.610 0.515 0.637 MV-Inpainting 0.606 0.499 0.724 InseRF (ours) 0.618 0.545 0.805

When only applying the mask-based blending, the authors of RePaint observe that, although the inpainted region matches the texture of the neighboring region, it is not well-harmonized in the image. Therefore, an additional re-sampling step is proposed, where the blended noisy images go through a few forward diffusion steps and are denoised again, to increase the harmonization of the inpainted regions. The proposed re-sampling step is characterized by two hyperparameters: 1) jump length: the number of applied forward diffusion steps; 2) steps: the number of repetitions of adding noise and de-noising of the blended images. In our experiments, we set both parameters to the value 2.

##### C.2. Baselines

Instruct-NeRF2NeRF (I-N2N): For our I-N2N baseline, we created a reimplementation in JAX on top of the MipNeRF360 code. Our implementation uses the official pretrained checkpoints of Instruct-Pix2Pix [4] and is compatible with LLFF datasets used in our experiments.

Multi-View Inpainting (MV-Inpainting): In Sec. 4.1 of the main paper, we proposed a baseline called Multi-View Inpainting (MV-Inpainting). MV-Inpainint is designed to insert objects into a 3D scene given accurate multi-view binary masks at the input. To ensure a fair comparison, MVInpainting uses the same 2D editing method as ours (Imagen [35] with RePaint [19]) to generate the target object in each viewpoint within the corresponding mask. In contrast to I-N2N, MV-Inpainting is equipped with localized editing to specifically investigate the importance of 3D consistency between different edited viewpoints.

To obtain the multi-view masks required for MVInpainting, we first generate and insert an object in the scene using our proposed object insertion. Then, we extract the multi-view masks of the target object by rendering the 3D object into the training viewpoints. The extracted masks are then used as input to MV-inpainting along with the corresponding text prompt. We would like to emphasize that our method only requires a single-view rough bounding box, in contrast to the multi-view accurate masks in MV-Inpainting.

##### C.3. Depth Estimation

As discussed in Sec. 3.4 of the main paper, we use the monocular depth estimated by MiDaS [31] to determine the location of the target object in the 3D frustum formed by the input bounding box in the reference image. As the estimated depth using MiDaS is non-metric, we perform a global affine depth alignment with the reference depth from the scene’s NeRF reconstruction, which we explain in greater detail in the following.

Let DR be the depth of the reference viewpoint rendered from the scene NeRF (not containing the object), and DˆE be the estimated depth of the edited reference view (containing the 2D object) using MiDaS. We define the aligned depth

map DˆA of the edited reference view as:

DˆA = a · DˆE + b (17)

where a and b are the scalar parameters of a global affine transformation. a and b are estimated by solving the following weighted least-square estimation:

min

a,b

i j

(1 − M(i,j)) · W(i,j) · (DR(i,j) − DˆA(i,j))2,

(18)

where M is a binary mask corresponding to the reference bounding box. For a 2D matrix A, A(i,j) denotes the element at row i and column j. W is the matrix containing pixel-wise weights for the estimation, negatively correlated with the distance of the pixel from the center of the bounding box located at row ic and column jc:

Wij = 1 − (i − ic)2 + (j − jc)2/z, (19) z = max( (i − ic)2 + (j − jc)2), (20) i ∈ {0,...,h − 1} & j ∈ {0,...,w − 1}, (21)

where z is a normalization term, and h and w are the height and width of the reference image, respectively. The weighted estimation of the alignment parameters helps with a more accurate alignment in the region surrounding the inserted object. In practice, we perform our alignments on image crops containing the object and its surroundings instead of the full image.

After aligning the estimated depth map, in order to determine the location of the object in the 3D scene, we first roughly estimate the distance of the center of the object from the camera center to be equal to the depth value at the center of the bounding box d. Then, we perform the scale and distance optimization proposed in Sec. 3.4 of the main paper, with the constraint that the depth of the center of the object’s rendered view from the reference viewpoint must be equal to d (please refer to discussion on the scale and distance optimization in Sec. 3.4 of the main paper for more details).

##### C.4. Rotation and Translation

Here we provide more details on the process of calculating the rotation and translation of the target object in the scene, discussed in Sec. 3.4 of the main paper. Specifically, we obtain the 3D location p⃗c of the center of the object in the 3D scene as the point along the normalized direction ⃗v pointing from the camera center to the center of the reference bounding box:

p⃗c = ⃗o + r∗ · ⃗v (22)

where r∗ is the optimized distance obtained from the scale and radius optimization (explained in Sec. 3.4 of the main paper).

We use the right-handed coordinate system convention for our scene and object NeRFs and place the object in an upward position in the scene centered at p⃗c. Moreover, we align the reference view of the object in its coordinate system (corresponding to zero azimuth and elevation) with the reference camera viewpoint in the scene’s coordinate system. In other words, we define the axes of the object coordinate system in the scene’s coordinate system as follows:

⃗uobject = [0,0,1]T, (23) ⃗xobject = −⃗v, (24) ⃗yobject = normalize(⃗uobject × xobject⃗ ), (25) z⃗object = normalize(⃗xobject × yobject⃗ ), (26)

The rotation R and the translation ⃗t are then obtained as:

R = [⃗xobject,⃗yobject,z⃗object]T (27) ⃗t = −Rp⃗c (28)

Using the obtained rotation, translation, and optimized object scale s∗, a point p⃗ in the scene’s coordinate system can be mapped to a point p⃗′ in the object’s one as follows:

1 s∗ [R,⃗t]p⃗ (29)

p⃗′ =

##### C.5. Scene and Object Fusion

- In Sec. 3.5 of the main paper, we provided a detailed discussion on how the scene and object NeRFs are fused in our method. In practice, object NeRFs fused in the scene may be queried with points in the 3D space that have not been seen during the object NeRF optimization, resulting in unwanted artifacts. To prevent such artifacts, we consider a 3D bounding box around the inserted objects, setting the density of the points sampled outside to zero. The dimensions of the 3D bounding box are determined based on the camera radius used in the single-view object reconstruction step and are fixed across edits and scenes.

C.6. Refinement

- In Sec. 3.6 of the main paper, we proposed an optional refinement based on the iterative NeRF optimization proposed in Instruct-NeRF2NeRF with two modifications: 1) using the multi-view masks obtained from the inserted object to make the refinement localized and 2) sampling viewpoints on a sphere encapsulating the inserted object in the scene. In particular, we sample the viewpoints on a sphere with the radius r∗ (the optimized object distance) from the ob-

ject’s center p⃗c. Such a sampling strategy allows for better edits by the 2D diffusion model. Moreover, instead of randomly picking the next viewpoint to edit and include in the NeRF optimization, as done in Instruct-NeRF2NeRF, we

order the viewpoints in a way that more frontal views are selected first. For example, viewpoints (azimuth,elevation) sampled from n equally-distanced azimuths with step size ∆theta and m equally-distanced elevations with the step size ∆phi are arranged as an ordered set V :

V = {(i · ∆θ,j · ∆ϕ)) | i ∈ I & j ∈ J} (30)

- I = {0,1,−1...,n/2,−n/2}, (31)
- J = {0,1,−1,...,m/2,−m/2}, (32)

Such ordering improves the 3D consistency of the refinement step, as it decreases the conflict caused by randomly selected and independently edited viewpoints.

Reference View Reference Edit Edited Neural Scene Representation

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

A blue butterfly

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

| | | |
|---|---|---|
| | | |

A pepper grinder

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

A duck

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

| | |
|---|---|
|A moai sta|tue|

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

A wedding cake

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

A teddy bear

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

A succulent

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

A pink crystal

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

A beige purse

###### Figure 8. Examples of using InseRF to insert an object into the neural representation of different indoor and outdoor scenes. More examples can be found in Fig. 3 of the main paper. 14

Original Scene I-N2N MV-Inpainting InseRF (ours) Original Scene I-N2N MV-Inpainting InseRF (ours)

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

Original Scene I-N2N MV-Inpainting InseRF (ours)

Original Scene I-N2N MV-Inpainting InseRF (ours)

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

(a) A cup on the table

(b) A duck on the road

Original Scene I-N2N MV-Inpainting InseRF (ours) Original Scene I-N2N MV-Inpainting InseRF (ours)

Original Scene I-N2N MV-Inpainting InseRF (ours) Original Scene I-N2N MV-Inpainting InseRF (ours)

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

(c) A pepper grinder on the counter

(d) A pouffe on the carpet

Original Scene I-N2N MV-Inpainting InseRF (ours) Original Scene I-N2N MV-Inpainting

Original Scene I-N2N MV-Inpainting InseRF (ours) Original Scene I-N2N MV-Inpainting

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

- Figure 9. Qualitative comparison of object insertion with different methods. I-N2N and multiview inpainting both fail at inserting the geometry of the object at the desired location. Our method, in contrast, can insert new 3D-consistent objects at the desired location. More examples can be found in Fig. 4 of the main paper.

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

15

[Figure 325]

[Figure 326]

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

## NoReﬁnementReﬁnementNoReﬁnementReﬁnement

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

- Figure 10. The refinement step proposed in our pipeline can improve the texture and the details of the inserted objects, leading to the higher quality and realism of the insertions. More examples can be found in Fig. 7 of the main paper.

16

