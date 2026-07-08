## LOOSECONTROL: Lifting ControlNet for Generalized Depth Conditioning

Shariq Farooq Bhat KAUST

Niloy J. Mitra University College London, Adobe Research

Peter Wonka KAUST

# arXiv:2312.03079v1[cs.CV]5Dec2023

[Figure 1]

Figure 1. Our framework LooseControl enables multiple ways to control the generative image modeling process. Left: (C1) Scene boundary control lets the user specify the boundary of the scene. We show the control inputs on top, the ControlNet results in the middle, and our results at the bottom. Middle: (C2) 3D box control can additionally specify object locations with the help of approximate 3D bounding boxes. We again show the control inputs on top, the ControlNet results in the middle, and our results on the bottom. Right-Top: (E1) 3D box editing. Note how the overall style of the scene is preserved across edits. Right-Bottom: (E2) Attribute editing: an example of changing a couch to another one and changing the overall furniture density in a room.

### Abstract

We present LOOSECONTROL to allow generalized depth conditioning for diffusion-based image generation. ControlNet, the SOTA for depth-conditioned image generation, produces remarkable results but relies on having access to detailed depth maps for guidance. Creating such exact depth maps, in many scenarios, is challenging. This paper introduces a generalized version of depth conditioning that enables many new content-creation workflows. Specifically, we allow (C1) scene boundary control for loosely specifying scenes with only boundary conditions, and (C2) 3D box control for specifying layout locations of the target objects rather than the exact shape and appearance of the objects. Using LOOSECONTROL, along with text guidance, users can create complex environments (e.g., rooms, street views, etc.) by specifying only scene boundaries and locations of primary objects. Further, we provide two editing mechanisms to refine the results: (E1) 3D box editing enables the user to refine images by changing, adding, or remov-

ing boxes while freezing the style of the image. This yields minimal changes apart from changes induced by the edited boxes. (E2) Attribute editing proposes possible editing directions to change one particular aspect of the scene, such as the overall object density or a particular object. Extensive tests and comparisons with baselines demonstrate the generality of our method. We believe that LOOSECONTROL can become an important design tool for easily creating complex environments and be extended to other forms of guidance channels. Code and more information is available at https://shariqfarooq123.github.io/ loose-control/.1

### 1. Introduction

Diffusion-based generative models now produce images with a remarkable degree of photorealism. ControlNet [51], trained on top of StableDiffusion [37], is the most powerful way to control such a generation process. Specifically,

1The project was supported in part by “NTGC-AI” funding at KAUST.

ControlNet allows guidance in the form of one or more of depth, edges, normal, or semantic channels. This ability to accurately control the image generation process enables various creative applications without requiring different architectures for different applications. However, providing perfect guidance for ControlNet can itself be challenging.

For example, imagine a scenario where a user wants to create living room images using depth guidance. She is now expected to supply a depth map containing information about walls, room furniture, all the way to even smaller objects and decorations. This is non-trivial. Producing realistic depth maps, especially for cluttered scenes, is probably as challenging as solving her original task. On the one hand, providing rough guidance, e.g., providing depth information for walls, floor, and ceiling, results in unsatisfactory images. While one may expect a furnished room, the room will be empty. Also providing approximate depth using target bounding boxes for furniture does not yield the desired result, because only boxy objects are generated. Figure 1 shows some examples using ControlNet [51].

We present LOOSECONTROL that allows controlling image generation using generalized guidance. In this work, we focus on guidance through depth maps. We consider two types of specifications. (C1) Scene boundary control where a layout can be given by its boundaries, e.g., walls and floors, but the final scene can be filled by an arbitrary number of objects that are not part of the depth conditioning as long as these additional objects are closer to the camera than the scene boundary. (C2) 3D box control where, in addition to layout boundaries, users can provide finer-scale guidance in the form of approximate bounding boxes for target objects. In the final generation, however, there can be additional secondary objects that do not need to strictly adhere to the given boxes. We demonstrate that many layoutguided image generation scenarios (e.g., rooms, streets, underwater) can be addressed in this framework. Figure 1 (left, middle) shows some of our generations.

We also provide two interactive editing modes to refine the results. (E1) 3D box editing enables the user to change, add, and remove boxes while freezing the style of the resulting image. The goal is to obtain minimal changes apart from the changes induced by the edited boxes. We describe how a notion of style can be formalized and preserved in our diffusion setup. (E2) 3D attribute editing enables users to change one particular attribute of the result, such as the type of one particular piece of furniture. Since diffusion processes return a distribution of results, we perform local analysis around any generation to reveal dominant modes of variation. Figure 1 (right) shows a few editing examples.

The technical realization of these four components is built on ControlNet with a frozen StableDiffusion backbone. We first propose a Low Rank (LoRA) based network adaptation. The LoRA-based architecture allows the

network to be fine-tuned in a few steps with only a small amount of training data, preventing the network from forgetting the original generation weights. A second important component is automatically synthesizing the necessary training data without requiring manual annotations. Finally, the edits are realized by manipulating the “keys” and “values” in attention layers, and a singular value analysis of the ControlNet Jacobian.

We evaluate a range of use cases. Since we enable a strictly more general depth control, there are no direct competitors. Hence, we compare strict and weaker guidance by varying the control weighting in ControlNet, as well as retraining ControlNet on synthetic data. We evaluate our setup on a variety of different scenes and also perform a user study that reveals over 95% preference for our method compared to baselines. In summary, we are the first to allow image generation from loose depth specifications and provide multiple types of edits to semantically explore variations in generations.

### 2. Related Work

Diffusion models have rapidly evolved as a leading generative approach, demonstrating remarkable success both in 2D image generation [6, 14, 18, 32, 33, 35, 37, 39] as well as 3D shape generation [11, 12, 15, 20, 21, 40, 50, 52]. While there are many similarities and synergies between 2D and 3D diffusion, in this paper, we only focus on the problem of adding conditional control to text-to-image diffusion models. In this context, many existing methods proposed solutions on direct guidance through well-known condition inputs like inpainting masks [45, 48], sketches [44], scene graphs [49], color palletes [19, 30, 43], 2D bounding boxes [27], segmentation maps [13, 37, 46], composition of multiple text descriptions [28], depth maps [30, 51] or by fine-tuning these models on a few subject-specific images [16, 29, 38, 42]. We focus on introducing new types of control. ControlNet [51] stands out as a key method in spatial control, supporting a diverse array of conditions like edge maps, depth maps, segmentation masks, normal maps, and OpenPose [8, 9, 41, 47] under a single framework. It is notably based on the widely-used open-source model, Stable Diffusion [37], which contributes to its popularity. However, creating spatial control conditions manually by a user for ControlNet can be challenging, often resulting in indirect control where conditions are derived from another image source, which can stifle the creative generation process. Additionally, this method can be restrictive in terms of the diversity of generations per condition, posing limitations in various scenarios. Our work builds on top of ControlNet and introduces novel conditioning mechanisms that are not only easy to construct manually by a user but also offer enhanced control and flexibility. In particular, we contribute novel forms of loose control described in the next

section.

### 3. Problem Setup - LOOSECONTROL

We formally introduce the core problem statement proposed in this work - loose depth control. Ordinary depth control, as implemented by the original ControlNet can be formally described as follows: Given an input condition depth map Dc, and access to an off-the-shelf monocular depth estimator fD, generate an image Igen such that the estimated depth fD(Igen) respects the input depth condition i.e.:

Generate Igen such that fD(Igen) = Dc. (1)

Thus, by design, the conditioning imposed by ControlNet is strict: as per training, the equality in Eq. (1) must exactly hold. Our goal is to extend this notion of control to a more flexible generalized form. We therefore define generalized control via the following setup: Given an input condition depth map Dc, and access to an off-the-shelf monocular depth estimator fD, and an arbitrary Boolean condition function ϕ(·,·):

Generate Igen such that ϕ(fD(Igen),Dc) is TRUE. (2) It is easy to observe that Eq. (1) is a special case of

Eq. (2). In this work, we propose to consider two other cases: scene boundary control and 3D box control, as described next.

(C1) Scene boundary control. In this case, we impose the condition such that the input depth condition Dc only specifies the tight upper bound of depth at each pixel:

ϕ : fD(Igen) ≤ Dc. (3)

(C2) 3D box control. In this case, we let the condition Dc control only the approximate position, orientation, and size of objects by specifying their approximate 3D bounding boxes. This leads to finer control than (C1) yet still avoids specifying the exact shape and appearance of the objects. Essentially, we design a condition function ϕ that ensures that the objects Ogeni generated in an image conform to their respective specified 3D bounding boxes Bi:

ϕ : Bi ∼ 3DBox(Ogeni ) ∀i, (4)

where 3DBox(Ogeni ) represents the oriented 3D bounding box of the i-th object segment in the generated image. This means that the position, size, and orientation of each object will be approximately within the bounds set by its corresponding 3D bounding box. Although our 3D box control training is strict, we show that boxes can be treated as only approximate, which proves to be highly beneficial in practice.

Both Eq. (3) and Eq. (4) specify a form of depth condition that is strictly more general than the ordinary control such as realized by ControlNet. We term such conditioning as LOOSECONTROL.

### 4. Realizing LOOSECONTROL

To realize ordinary depth control of text-to-image diffusion models, one needs access to triplets of the form (T,Dc,I) for training, where T is the text description of the image I and Dc represents the depth condition. Generally, one has access to the pairs (T,I) and the ordinary depth condition Dc is obtained by applying an off-the-shelf depth estimator fD on the given image (i.e., Dc = fD(I)) to obtain the triplets (T,Dc = fD(I),I). However, in our case, the depth condition Dc must take a more generalized form. For our goal, the form of Dc is constrained by three main requirements: (i) Compatibility with ControlNet, where the depth condition should resemble a conventional depth map. (ii) Efficient to extract from a given image without manual annotation. (iii) Easy to construct manually by a user. We describe below how to obtain the appropriate form of Dc such that LOOSECONTROL is realized.

#### 4.1. How to represent scenes as boundaries?

We begin by outlining the estimation of the depth condition Dc from a given image for implementing scene boundary control as described in Eq. (3). In this context, we seek a depth condition that acts as an upper depth limit for individual pixels. We propose to extract the scene boundary surface for upper-bound training in the following manner. Specifically, we define the boundary as a set of planar surfaces encompassing the scene that accurately delineates the scene’s spatial extent. For a given bounded 3D scene and a specific camera viewpoint, the boundary depth is defined as the z-depth of these boundary surfaces. In practice, this means that all pixels in the image inherit the depth value associated with the boundary surface, even if the boundary surfaces are occluded by objects within the scene’s interior. To provide a concrete example, consider an indoor scene (refer to Fig. 3). The boundary typically encompasses the walls, ceiling, and floor planes. Consequently, a boundary depth map for an indoor scene exclusively reflects the depths of walls, ceiling, and floor, irrespective of the presence of room furniture.

A naive approach to extracting boundary depth involves leveraging annotated room layout datasets. However, such datasets often contain ambiguous annotations in multi-room images, and the room-level annotations directly violate our “upper” bound condition (see Supplementary Materials). Additionally, this approach necessitates manual annotation for boundary extraction and confines the scope to roomcentric scenarios. Our objective is to devise a strategy that is applicable across diverse scene types and readily applicable to extract boundary depth from any given image. To this end, we propose the following algorithm:

We use a multi-step approach to efficiently extract boundary depth from a given image. Refer to Fig. 2 for the outline of our approach. We begin by estimating the depth

[Figure 2]

[Figure 3]

Backprojection

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Orthographic Projection

Approx Poly

Extrusion

Depth Render

- Figure 2. Pipeline for extracting boundary depth from an image. From left to right: Input image and its estimated depth map, backprojected 3D mesh, orthographic projection of the mesh on a horizontal plane, polygon approximation of the 2D boundary, extrusion of poly sides, resulting boundary depth map. For ease of visualization, the ceiling is not shown.

map of the given image using an off-the-shelf monocular depth estimator [7]. We then back-project the image into a 3D triangular mesh within the world space. Our goal is to extract the planar surfaces that encompass this mesh. For efficiency, we only make use of vertical planes during training. This reduces the 3D boundary extraction problem to

- 2D. We project the 3D mesh of the scene onto a horizontal plane using orthographic projection. This projection facilitates the precise delineation of the 2D boundary that encapsulates the scene. We then approximate the 2D boundary of projection with a polygon and extrude the sides into planes matching the scene height. For an indoor scene, for example, these ‘poly’-planes represent a good approximation of wall planes. These poly planes and optionally the ground and ceiling plane (if available) together form the boundary surface. We then render the depth of this boundary surface from the original camera view to get the boundary depth

map Dc, serving as a proxy for the scene boundary condition.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Scene Boundary Control 3D Box Control

- Figure 3. Illustration of proxy depth for scene boundary control training (left) and 3D box control training (right).
- 4.2. How to represent scenes as 3D boxes?

which, currently, presents challenges. Existing monocular 3D object detection pipelines tend to be either sparse, targeting specific object categories, or domain-specific, focusing on scenarios like road scenes, limiting their versatility. To address these limitations, we introduce a custom monocular pipeline that approximates the 3D bounding boxes of objects and is simultaneously dense (capable of recovering any object type) and generalizable (no domain preference).

For a given image, we first obtain its depth map and segmentation map using an off-the-shelf monocular depth estimator ZoeDepth [7] and SAM [24], respectively. The depth map provides 3D spatial information, while the segmentation map delineates object boundaries, which we use together to approximate 3D bounding boxes of objects. For each segment, we perform back-projection, transforming the image segment using its depth map into a point cloud in 3D world space. We then estimate the corresponding oriented 3D bounding box with minimal volume for each back-projected segment point cloud and represent it as a cuboidal mesh. Finally, to obtain Dc, we render this scene of boxes using the original camera parameters used in backprojection and extract the depth map. This process is summarized in Fig. 5.

#### 4.3. How to train for LOOSECONTROL?

We build our framework on StableDiffusion [37] and ControlNet [51]. In both scene boundary control and 3D box control, the constructed depth condition Dc serves as a proxy for specifying a more generalized control - scene boundary and 3D bounding box, respectively - while retain-

We now describe the estimation of Dc from a given image to realize 3D box control (Eq. (4)). For a given image, our goal is to construct Dc such that it contains the information about the 3D bounding boxes of the objects present in the scene. At the same time, we need to ensure compatibility with depth-conditioned frameworks, such as ControlNet, where the depth condition should resemble a conventional depth map. To achieve this, our idea is to obtain 3D bounding boxes of the objects in the image and render the depth of the boxes. The resulting depth map can serve as a proxy for specifying 3D bounding boxes (See Fig. 3). However, this task necessitates a robust monocular 3D object detector,

[Figure 13]

[Figure 14]

[Figure 15]

ControlNet

Stable Diffusion

[Figure 16]

Depth Estimator

photo of a helicopter over a pine forest in winter

[Figure 17]

[Figure 18]

[Figure 19]

/

[Figure 20]

[Figure 21]

LoRA

Proxy Estimator

Stable Diffusion

ControlNet

Figure 4. Training pipelines for ControlNet (Top) and LOOSECONTROL (Bottom). Proxy Estimator represents our proxy depth extraction algorithms. During inference, we enable an option for a user to design the condition depth manually via a UI.

Backprojection

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Depth Render

3D Lifting

[Figure 28]

Compute BBox

- Figure 5. Pipeline for extracting proxy depth for 3D box control from an image. Left to right: Input image, its estimated depth and segmentation maps, back-projected 3D mesh, 3D lifting of segmentation, 3D bounding boxes for segments, resulting proxy depth map.

ing compatibility with ordinary depth-conditioned frameworks such as ControlNet. This ensures a smooth transition and facilitates the easy adoption of these approaches. Thanks to this backward compatibility, we can efficiently fine-tune a pre-trained depth ControlNet to achieve generalized depth control. We prepare the triplets (T, Dc, I) from the given (T, I) pairs where Dc is constructed using the algorithms presented in previous sections. We explore two primary options for fine-tuning: (i) Naive fine-tuning: where we finetune the entire ControlNet; and (ii) LoRA fine-tuning: where we employ LoRA-based fine-tuning of ControlNet (see Fig. 4). Specifically, for every attention block in ControlNet, we learn a low-rank update:

W′x = Wx + BAx (5)

where W ∈ RM×N is the original frozen projection, B ∈ RM×r and A ∈ Rr×N are trainable low rank matrices of rank r ≪ min(M,N). It is important to note that in both cases Stable Diffusion U-net remains frozen during finetuning. For both control types (C1 and C2), we use the NYU-Depth-v2 [31] dataset for fine-tuning and obtain textual image descriptions using BLIPv2 [26] captioning.

Inference time adjustments. We observe that naive fine-tuning results in saturated generations (Fig. 6). Interestingly, the color artifacts are eliminated if the residuals injected into Stable Diffusion U-net from ControlNet are reweighted at inference time. We observe that residual to the bottleneck block is responsible for most of the condition adherence and other higher resolution skip residuals largely add only local texture information. On the other hand, LoRA fine-tuning proves to be robust without any color issues. However, we introduce a controllable scale factor γ at inference time such that the update is given as:

[Figure 29]

- Figure 6. Given scene boundary (left), comparing naive finetuning, adjusted skip weights, and our LoRA-based training, resp.

W′x = Wx + γBAx. We observe that controlling γ can lead to higher quality results, especially for γ > 1. We use γ = 1.2 as the default.

#### 4.4. 3D Box Editing

Both the scene boundary condition and the box condition are easy to manipulate by a user. This opens up possibilities where a user can interactively manipulate the scene and objects, for example, by changing, adding, or removing the boxes. However, it is desirable to ‘lock in’ and maintain the style and composition of the scene while manipulating the condition. A naive way to maintain the overall style would to be fix the seed but we observe that changing the condition even when fixing the seed can lead to diverging generations. To this end, we propose Style Preserving Edits, inspired by video diffusion models [22], which allows users to maintain the desired style of the scene while conducting sequential edits. This feature is realized by replacing the “keys” and “values” for the given image in the attention layers of the Stable Diffusion U-net with those from the source image. However, in our case, sharing “keys” and “values” through all layers of the U-net leads to undesirable results, producing images identical to source images. We find that sharing “keys” and “values” from only the last two decoder layers ensures the preservation of the desired identity or style while adhering to the new target condition.

#### 4.5. Attribute Editing

LOOSECONTROL’s generalized nature compared to ordinary depth control allows for a wider range of possible scenes under a given condition. To explore this expansive space, we conduct various forms of latent exploration.

ControlNet produces two types of residuals for a given condition: residuals added to the SD U-net bottleneck and residuals added to decoder skip connections. Our focus centers on the bottleneck feature space (H-space), known for its semantic behavior [17, 25] and from our experiments, its pivotal role in condition adherence.

In our exploration, we examine how the latent input, x, affects ControlNet’s output for a given fixed condition. We extract the most influential directions in x-space, which cause substantial changes in ∆h ∈ H, and find their counterparts in H-space, using Singular Value Decomposition

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

inputguidanceControlNetLooseControl(ours)

###### (C1)SceneBoundaryControl

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

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

crowd celebrating Guy Fawkes’s Night on Trafalgar Square

a clothing store seating area of charming Irish pub with dark wood paneling

a heavily equipped kitchen with appliances

a living room with sofa, house plants and a table

inside of an old cathedral with pew

busy street in Toyko at night with a lot of pedestrians

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

inputguidanceControlNetLooseControl(ours)

(C2)3DBoxControl

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

swivel office sofa chair with a round backrest

a bedroom with a canopy bed with puffy bedding & nightstands

a set of conical icebergs in the ocean

a white English arm roll sofa a living room four office chairs a helicopter flying over a pine forest in winter

- Figure 7. Qualitative results of (C1) Scene boundary control (rows 1-3) and (C2) 3D Box control (rows 4-6). Top: Input depth condition Middle: ControlNet generations. Bottom: Ours. Our generations are more realistic and adhere to the prompt better than ControlNet.

(SVD) of the ControlNet Jacobian, ∂∆h/∂x. To keep the SVD computation feasible, we only extract the first N directions, labeled {ei}, which guide our modification of the bottleneck residual via: ∆h′ = ∆h+βei, where the scalar β controls the magnitude of the edit.

Per-condition latent exploration enhances user control by revealing various semantic edit directions, for example, sizes, types, and even the number of objects in the scene. This gives rise to a continuous control space, allowing for fine-grained adjustments that were challenging with conventional conditioning methods. This continuous control space operates orthogonal to the input depth condition, enabling the maintenance of specified conditions—whether scene boundary control or 3D box control—while simultaneously exploring a wide range of generated images that

adhere to the given condition.

### 5. Applications and Results

We developed an interface to enable users to conveniently design a scene boundary and 3D boxes, and generate images interactively (see supplementary material). Here, we present qualitative and quantitative results.

(C1) Scene Boundary Control. Scene boundary control enables diverse applications across various domains. We present some examples and comparisons between LOOSECONTROL and ordinary control in Fig. 7. One particularly noteworthy application of scene boundary control is in indoor scene generation. Users can provide somewhat

[Figure 71]

###### Figure 8. Qualitative results of (E1) 3D Box Editing, and (E2) Attribute Editing. (E1) 3D Box editing enables us to move and re-orient objects in the scene, change their size, split them, and more. (E2) Attribute Editing lets us change properties like the density of furniture (e.g., increase or decrease density) and the material of objects (e.g., fabric to leather or vica versa).

abstract specifications for room layouts, generally confined to walls and floor, and still generate images of fully furnished rooms. This is different from ordinary depth conditioning, which demands precise depth information.

Another scenario involves (partially) bounded outdoor scenes where the user can provide the scene boundary depth, for example, in terms of locations of buildings (See Fig. 1). Unlike the ordinary depth control implemented by the original ControlNet, which leads to empty scenes, our LOOSECONTROL model excels at generating realistic scenes, inclusive of objects like cars, traffic lights, and pedestrians, all while adhering to the user-provided scene boundary condition. This feat underscores the reliability and generalization capability of LOOSECONTROL via proxy fine-tuning.

In Fig. 9, we show the effect of varying the conditioning scale, λ, in the context of scene boundary control. We observe that ControlNet still produces ‘empty’ rooms with lesser prompt adherence for most of the conditioning scales.

(C2) 3D Box Control. We present generations and comparisons with ControlNet for 3D box control for a variety of scenes in Fig. 7. ControlNet with ordinary control produces ‘boxy’ generations which are not only unrealistic but also result in lesser prompt adherence. On the other hand, our method is able to produce more realistic images and shows better adherence to the given prompt demonstrating the utility of such control.

(E1) 3D Box Editing. We show qualitative results for 3D Box Editing in Fig. 8. 3D box Editing allows users to manipulate objects, for example, by changing their position in

- 3D space, their orientation, and size. Users can perform 3D-aware edits, such as relocating furniture within a room or converting a small sofa chair to a big sofa. The Style Preserving Edit mechanism facilitates sequential editing of object poses and locations, preserving the scene’s essence while introducing creative modifications. Since the 3D box control implicitly specifies the rough shape and size of the objects, we can edit the boxes themselves such as splitting

|[Figure 72]<br><br>condition|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

ControlNetLooseControl(ours)

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

- Figure 9. Effect of conditioning control scale λ. Top: ControlNet. Bottom: Ours. The results were produced for the prompt “A cozy coastal bedroom with a sea view”.

a box into two, resulting in semantic edits on objects.

(E2) Attribute Editing. Extracting the edit directions for a fixed scene boundary leads to continuous control directions for attributes such as the density of furniture and furniture type. For a given 3D box control condition (see Fig. 8), the edit directions represent attributes like the shape, thickness, and material properties such as “leather”. We observe that edit directions are smooth and continuous for the most part and varying the magnitude β directly reflects that magnitude of change in the attribute.

User Study. We conducted a user study to quantify the

improvement in SceneBoundaryControl 3DBoxControl control and quality of generations. 41 users were presented with the prompt and condition and asked to pick the image they preferred between ours and ControlNet. A majority of users, more than 95%, ranked our result better. See inset for results.

ours

ControlNet

### 6. Conclusion

We have presented LOOSECONTROL to support generalized depth control for image generation. We introduced two types of control: (C1) Scene boundary control and (C2) 3D Box control to generate initial images. To refine initial results, we proposed (E1) 3D box editing and (E2) Attribute editing. Our framework provides new modes of creative generation and editing allowing users to more effectively explore the design space with depth guidance. A user study revealed over 95% preference for LOOSECONTROL compared to previous work.

Limitations. Although our method works well for primary objects, control over secondary objects is harder to achieve – we attribute this to the training data where secondary objects are less common. We expect a scene-based generator may lead to improved results. Also, similar to the original ControlNet we find that providing too many constraints as input reduces the diversity of the results.

Future work. We would like to use LOOSECONTROL in conjunction with masks and also with MultiControlNet through the generalized specification of other maps. For example, in the context of sketch-based generation, specifying only a selection of (dominant) edges is easier for users than drawing all the Canny edges. Further, we would like to explore temporally consistent generation to produce output videos under smoothly changing guidance as in an interpolated 6DOF object specification.

### References

- [1] https://huggingface.co/. 1
- [2] https://www.babylonjs.com/. 1
- [3] https://huggingface.co/docs/diffusers/ index. 1
- [4] https://streamlit.io/. 1
- [5] Abubakar Abid, Ali Abdalla, Ali Abid, Dawood Khan, Abdulrahman Alfozan, and James Zou. Gradio: Hassle-free sharing and testing of ml models in the wild. arXiv preprint arXiv:1906.02569, 2019. 1
- [6] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [7] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023. 4, 1
- [8] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In CVPR, 2017. 2
- [9] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019. 2
- [10] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017. 1
- [11] Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. Single-stage diffusion nerf: A unified approach to 3d generation and reconstruction. In ICCV, 2023. 2
- [12] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-SDF: Conditional generative modeling of signed distance functions, 2023. 2
- [13] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022. 2
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2
- [15] Ziya Erkoc¸, Fangchang Ma, Qi Shan, Matthias Nießner, and Angela Dai. Hyperdiffusion: Generating implicit neural fields with weight-space diffusion. arXiv preprint arXiv:2303.17015, 2023. 2
- [16] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2
- [17] Ren´e Haas, Inbar Huberman-Spiegelglas, Rotem Mulayoff, and Tomer Michaeli. Discovering interpretable directions in

- the semantic latent space of diffusion models. arXiv preprint arXiv:2303.11073, 2023. 5
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [19] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 2
- [20] Ka-Hei Hui, Ruihui Li, Jingyu Hu, and Chi-Wing Fu. Neural wavelet-domain diffusion for 3d shape generation. In SIGGRAPH Asia 2022 Conference Papers, New York, NY, USA,

2022. Association for Computing Machinery. 2

- [21] Animesh Karnewar, Niloy J Mitra, Andrea Vedaldi, and David Novotny. Holofusion: Towards photo-realistic 3d generative modeling. ICCV, 2023. 2
- [22] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 5
- [23] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 1

- [24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 4, 1
- [25] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion models already have a semantic latent space. arXiv preprint arXiv:2210.10960, 2022. 5
- [26] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 5
- [27] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 2
- [28] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022. 2
- [29] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 2
- [30] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2
- [31] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 5

- [32] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [33] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR,

2021. 2

- [34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 1
- [35] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [36] Nikhila Ravi, Jeremy Reizenstein, David Novotny, Taylor Gordon, Wan-Yen Lo, Justin Johnson, and Georgia Gkioxari. Accelerating 3d deep learning with pytorch3d. arXiv preprint arXiv:2007.08501, 2020. 1
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 4
- [38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2
- [39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [40] J. Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20875–20886, 2023. 2
- [41] Tomas Simon, Hanbyul Joo, Iain Matthews, and Yaser Sheikh. Hand keypoint detection in single images using multiview bootstrapping. In CVPR, 2017. 2
- [42] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. ACM SIGGRAPH 2023 Conference Proceedings, 2023. 2
- [43] Vaibhav Vavilala and David Forsyth. Applying a color palette with local control using diffusion models, 2023. 2
- [44] Andrey Voynov, Kfir Aberman, and Daniel Cohen-Or. Sketch-guided text-to-image diffusion models. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 2

- [45] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi PontTuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18359– 18369, 2023. 2
- [46] Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong Chen, Qifeng Chen, and Fang Wen. Pretraining is all you need for image-to-image translation. arXiv preprint arXiv:2205.12952, 2022. 2
- [47] Shih-En Wei, Varun Ramakrishna, Takeo Kanade, and Yaser Sheikh. Convolutional pose machines. In CVPR, 2016. 2
- [48] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22428–22437, 2023. 2
- [49] Ling Yang, Zhilin Huang, Yang Song, Shenda Hong, Guohao Li, Wentao Zhang, Bin Cui, Bernard Ghanem, and MingHsuan Yang. Diffusion-based scene graph to image generation with masked contrastive pre-training. arXiv preprint arXiv:2211.11138, 2022. 2
- [50] Biao Zhang, Jiapeng Tang, Matthias Nießner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Trans. Graph., 42(4), 2023. 2
- [51] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 1, 2, 4
- [52] Xin-Yang Zheng, Hao Pan, Peng-Shuai Wang, Xin Tong, Yang Liu, and Heung-Yeung Shum. Locally attentional SDF diffusion for controllable 3d shape generation. ACM Trans. Graph., 42(4):91:1–91:13, 2023. 2
- [53] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3d: A modern library for 3d data processing. arXiv preprint arXiv:1801.09847, 2018. 1

## LOOSECONTROL: Lifting ControlNet for Generalized Depth Conditioning Supplementary Material

[Figure 83]

- Figure 10. Upper-bound violations in Matterport3D [10]. Top: RGB panorama images. Bottom: Depth rendered using the layout labels from the dataset.

### 7. Implementation details

For all our experiments, we use Stable Diffusion v1.5 [37] and the corresponding ControlNet [51] checkpoint “lllyasviel/control v11f1p sd15 depth” hosted on HugginFace [1]. We use the PyTorch [34] framework and the diffusers [3] library as the framework for diffusion models. We use ZoeDepth [7] and SAM [24] for extracting depth maps and segmentation maps respectively. For SAM, we use min_mask_area=1e4. We make use of Pytorch3D [36] and Open3D [53] in our 3D framework and PyTorch3D for rendering the depth maps for obtaining the proxy depths for fine-tuning. For backprojection, we randomly choose an FOV in the range of 43 and 57 degrees. For both Scene Boundary Control and 3D Box Control, we use LoRA rank r = 8 and fine-tune only LoRA layers for 200 steps with a learning rate of 0.0001, and batch size of 12 with Adam [23] optimizer. We use controlnet_conditioning_scale=1.0 and LoRA scale factor γ = 1.2 as default. We built our 3D editor user interface using Gradio [5] and BabylonJS [2].

### 9. User study additional details

We created an anonymous user study as a web form using streamlit [4]. Users were presented with ‘Two-alternative forced choice’ (2-AFC) with two options as images generated by our baseline (ControlNet) and our result and asked to respond with their preference for the given text prompt and condition image. The options were anonymized and did not indicate the name of the method. Each user was asked to respond to 10 randomized questions in total (5 for Scene Boundary Control, and 5 for 3D box control). The order of options was also randomized. As mentioned in the main paper, over 95% of responses were in favor of our method.

### 8. Issues with Room Layout Datasets

As discussed in the main paper, a possible alternative for extracting the scene boundary for the preparation of the dataset for Scene Boundary Control could be room layout datasets. However, we note that these datasets often contain ambiguous layout labels that directly violate our tight upper-bound condition required to implement scene boundary control. We provide some examples of these violations for the popular MatterPort3D dataset as the representative in Fig. 10

