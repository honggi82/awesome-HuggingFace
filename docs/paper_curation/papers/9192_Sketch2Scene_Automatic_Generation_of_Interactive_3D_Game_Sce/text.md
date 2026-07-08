## Sketch2Scene: Automatic Generation of Interactive 3D Game Scenes from User’s Casual Sketches

Yongzhi Xu1∗, Yonhon Ng1∗, Yifu Wang1∗, Inkyu Sa, Yunfei Duan1, Zhenhong Sun1, Yang Li1, Pan Ji1, Hongdong Li2

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

# arXiv:2408.04567v2[cs.CV]6Feb2026

Prompt: An isometric game map featuring a river at the center and bridges crossing the river.

|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>|
|---|

|[Figure 7]|
|---|

Fig. 1: We propose an efficient pipeline for automatically generating interactive 3D scenes from user’s natural input prompt (e.g., handdrawn sketch and text description of the scene). An interactive and playable 3D game scene can be created instantly with a hand-drawn sketch of the scene.

Abstract—3D Content Generation is at the heart of many computer graphics applications, including video gaming, filmmaking, virtual and augmented reality, etc. This paper proposes a novel deep-learning based approach for automatically generating interactive and playable 3D game scenes, all from the user’s casual prompts such as a hand-drawn sketch. Sketchbased input offers a natural, and convenient way to convey the user’s design intention in the content creation process. To circumvent the data-deficient challenge in learning (i.e. the lack of large training data of 3D scenes), our method leverages a pretrained 2D denoising diffusion model to generate a 2D image of the scene as the conceptual guidance. In this process, we adopt the isometric projection mode to factor out unknown camera poses while obtaining the scene layout. From the generated isometric image, we use a pre-trained image understanding method to segment the image into meaningful parts, such as off-

ground objects, trees, and buildings, and extract the 2D scene layout. These segments and layouts are subsequently fed into a procedural content generation (PCG) engine, such as a 3D video game engine like Unity or Unreal, to create the 3D scene. The resulting 3D scene can be seamlessly integrated into a game development environment and is readily playable. Extensive tests demonstrate that our method can efficiently generate highquality and interactive 3D game scenes with layouts that closely follow the user’s intention.

MULTIMEDIA MATERIAL

Project Page: https://xrvisionlabs.github.io/Sketch2Scene/ Code†: https://github.com/Tencent/Triplet_Tuning

I. INTRODUCTION

∗ Equal contribution.

Generative AI models are taking the world with storm, by enabling the automatic creation of new contents of versatile modalities (e.g. text, image, video, audio and music, etc.),

- 1 XR Vision Labs, Tencent.
- 2 Australian National University † Enhanced 2D conceptual image generation method is detailed in [1].

simply from user’s natural prompt input. AI-generated images, music, and videos can reach a level of quality close to those created by professional artists. This success has already ventured into the realm of 3D object-level asset modeling (such as LRM [2], CRM [3], and MeshLRM [4]), thanks to the growing size of massive 3D object datasets, such as Objaverse-XL [5]. Existing methods published so far have been focusing on the AI generation of small 3D assets of single object level, e.g. [2]–[4].

In contrast, the generation of high-quality 3D scenes, such as an open-world game scene, largely remains an underexplored problem. The main reason for this stems from the data efficiency issue for deep learning, namely, due to the lack of a large amount of high-quality 3D scenes to permit large scale training of powerful machine learning models. For example, so far, there is virtually no publicly available largescale game scene dataset, other than some city-scale urban driving/street-view scenes captured mainly for autonomous driving research.

In this paper, we introduce Sketch2Scene, a novel pipeline for 3D scene generation. This method automatically creates realistic and interactive virtual environments using a usercontrolled diffusion model, with input provided by a userdrawn sketch and optionally a text prompt. By leveraging casual user sketches, our approach effectively addresses the above-mentioned limitations in generating large-scale, openworld outdoor scenes. To overcome the lack of 3D scene training data, we design a method that leverages a pre-trained

- 2D denoising diffusion model (e.g. [6]) for 2D isometric image generation.

Our method first generates an illustrative 2D image (in isometric projection) depicting the intended concept of the

- 3D game scene. Then, a visual scene understanding module is designed to interpret the image, forming a background terrain (basemap) and foreground object layout map. This layout map, used as a blueprint, is fed into a procedural content generation pipeline to create 3D game scenes that are compatible hence readily playable in an existing game or rendering engine, such as Unity or Blender.

To ensure precise and adaptable sketch control, we train the ControlNet [7] using a semantic-constraint diffusion loss. Furthermore, we employ a newly developed basemap inpainting model to generate the scene’s basemap. To facilitate this process, we have curated a unique gaming isometric dataset for training both the ControlNet and the basemap inpainting networks. For achieving game-ready quality, we use high-resolution texture tiles composed with generated splat maps from the reference Bird’s Eye View (BEV) image. Our method significantly surpasses existing scene creation techniques in terms of shape quality, diversity, and controllability.

Our key contributions can be summarised as:

- • a controllable, sketch-guided 2D isometric image generation pipeline.
- • a basemap inpainting model, trained via step-unrolled denoising diffusion on a new dataset.
- • a learning-based compositional 3D scene understanding

module.

• a procedural generation pipeline to render an interactive 3D scene using the scene parameters obtained from the above scene understanding module.

II. RELATED WORKS

- A. Diffusion-based 3D scene generation

The success of diffusion models like Stable Diffusion [6], DALLE [8], and Midjourney has significantly boosted interest in developing 3D content generation tools. However, generating high-fidelity 3D scenes from text prompts or images remains challenging due to the complexity and variability in shapes and appearances. Text2Room [9] uses 2D text-toimage models and monocular depth estimation for iterative scene generation. Similar indoor-focused approaches include SceneScape [10], which renders videos of diverse scenes, and RealmDreamer [11], which uses a 3D Gaussian Splatting model [12] for wide-baseline rendering. CC3D [13] generates compositional scenes by optimizing multiple NeRFs with SDS loss [14]. Unlike CC3D, [15] jointly optimizes relative transformations between NeRFs during the SDS process for unsupervised scene decomposition. ControlRoom3D [16] and CTRL-ROOM [17] create panorama-view-based text-to3D room generation models, using 3D room layouts and a fine-tuned ControlNet [18] to edit generated rooms. SceneWiz3D synthesizes high-fidelity 3D scenes from text by using a hybrid scene representation, employing DMTets [19] for objects of interest and NeRF [20] for the environment. For large-scale, nature or city scene generation, Citygen [21] generate infinite and controllable 3D layouts by representing the 3D city layout with a semantic field and a height field. WonderJourney [22] employs ChatGPT-generated text prompts to guide the image generation process, resulting in diverse and automated scene generation. Besides generating 3D scenes from single or multi-view 2D images, another direction involves directly generating 3D scenes through text prompts or image guidance. XCube [23] uses a multiresolution coarse-to-fine shape generator with sparse voxel grid representation to generate high-resolution scenes such as streets. BlockFusion [24] leverages Tri-plane diffusion to create 3D scenes as cubic blocks, enabling large-scale unbounded scene generation with a novel tri-plane extrapolation mechanism. Frankenstein [25] extends Tri-plane diffusion for building a compositional scene generation tool.

- B. Procedural generation

Past solutions for 3D scene generation primarily focused on procedural generation methods using modifiable parameters and rule-based systems. Here we focus on combined solutions using large language models (LLMs) or diffusion models for controllable 3D scene generation, as a comprehensive listing of all works would exceed the scope of this paper. 3D-GPT [26] introduced a framework using LLMs to generate Python codes for 3D modeling, enhancing realworld flexibility of [27]. SceneX [28] improves LLM-guided scene generation by automating high-quality scene creation from textual descriptions using a large 3D asset database

Sketch guided isometric generation

#### User inputs

Visual scene understanding

#### Procedural 3D scene generation

Objects Heightmap Texture splatmap

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

2D isometric image

Sketch prompt

[Figure 18]

[Figure 19]

[Figure 20]

|An isometric game featuring a river at the center and bridges crossing the river|
|---|

[Figure 21]

Text prompt

2D empty basemap

3D interactive scene

Semantic segmentation Depth map Terrain mesh

- Fig. 2: Overview of the pipeline of the proposed method. The input user sketch and text prompt are fed into our pre-trained ControlNet that generates a 2D isometric reference image. Our Scene-Understanding module then extracts the foreground object masks. The masks are fed to a pre-trained inpainting model which generates the isometric empty basemap (i.e., the background terrain with no objects). The scene understanding module also computes the heightmap, texture splatmap and object instance pose. Finally, a procedural 3D scene generation module is employed to generate and render the 3D game scene.

and a planner for task planning, asset retrieval, and action execution.

should be able to fill in the blank regions with plausible and compatible contents. For instance, if the user draws a few houses, the model should be able to generate a road network and trees that are naturally align well with the houses, leading to a harmonic scene. To enable this flexibility in the input sketch, the model should be trained using sketches with a diverse combination. For example, the same water map associates with different roads, or the same roads combine with different buildings. Thus, we conducts sketch category filtering that augments the sketch by randomly dropping out each category. As shown in Fig. 3, the sketch of a reference image is augmented to a new one by removing other categories but road.

III. METHOD

Figure 2 provides an overview of our pipeline, which comprises three key modules: Sketch Guided Isometric Generation, Visual Scene Understanding, and Procedural 3D Scene Generation. The following subsections will describe each module in detail.

- A. Sketch Guided Isometric Generation

1) 2D Isometric Image Generation: Starting from a casual user sketch, our first task is to generate a 2D conceptual illustration of the 3D scene. To this end, we propose to use a pre-trained 2D image (denoising) diffusion model to generate an oblique view of the 3D scene using the isometric projection model. Isometric projection is a special orthographic camera projection where the coordinate axes with the same dimension have equal length, and the angle between each pair of axes is 120◦. We use this type of projection mainly for its simplicity in handling occlusions.

The training of above augmented data does not work directly since all augmented sketches correspond to the same ground truth as illustrated in Fig. 3. To address this issue, we introduce a new loss function, namely, the SketchAware Loss (SAL). A soft-mask is created for each sketch and is applied as the loss weight matrix to encourage the supervision of ControlNet to focus on valid regions in the sketch. The weight is obtained by convolving the sketch mask using a Gaussian kernel, as depicted in the middle column of Fig. 3. This means a higher weight is applied close to the user’s sketch and vice versa. Let ω = max(0.1,G (f(S))), the resulting mask is incorporated into the following loss:

We employ ControlNet [7] to provide the user with precise control in the layout of generated scene. ControlNet allows a pre-trained text-to-image diffusion model to have additional spatial conditioning during the denoising steps. We train our sketch-based conditioning with one-hot encoding with N channels, where each channel corresponds to a unique sketch category (e.g. building, road, water, bridge, etc.). Compared to the more commonly used RGB pixel-domain conditioning, one-hot representation possesses the benefit of simpler training complexity and allows category overlap.

LSAL = Ex0,t,ct,cs,ε∼N (0,1)[∥(ε −εθ(xt,t,ct,cs))·ω∥22], (1) where S is the one-hot sketch, f computes the maximum along the sketch channels (equivalent to any operator in boolean array), G is a standard Gaussian convolution with 11×11 kernel, ct is text prompt, cs is sketch condition.

2) 2D Empty Terrain Extraction: A clean reference image of the empty terrain (aka. the “basemap”) is needed to recover the corresponding 3D terrain of the scene. In the

Our method only requires the user to provide a casual guidance via a hand-drawn sketch with arbitrary number of categories. Once the sketch is provided, our method

[Figure 22]

- Fig. 3: The Sketch-Aware Loss (SAL) facilitates ControlNet’s training with a single ground truth image associated with diverse sketches generated through random category filtering, thereby enhancing its performance on flexible sketches.

generated 2D isometric image, there are still some occluded regions of the terrain due to the presence of foreground objects. For instance, the ground on the far side of a building is not visible. Unlike general inpainting tasks, this is challenging due to the requirement that the inpainted region must not contain any foreground object. Existing contextbased inpainting methods struggle with filling such large masks due to a lack of prior knowledge. While diffusionbased generative inpainting methods show its potential, current state-of-the-art (SOTA) methods such as RePaint [29], EditBench [30], and Stable Diffusion XL Inpaint (SDXLInpaint) [31] do not produce satisfactory results, even with carefully designed prompts. (cf. Fig 6)

To solve this, we fine-tune the LoRA [32] on SDXLInpaint to learn the distribution of the basemap and foreground masks. To overcome the obstacle of the lack of isometric-basemap datasets for training, we collected a training dataset from three types of data sources: isometric images with foreground objects, perspective images of empty terrain, and terrain texture images. When using isometric images with foreground objects for training, the inpainting mask is designed to have no overlap with the foreground object. On the other hand, the other two types of training data use foreground masks that are randomly extracted from other isometric images intersected with random shapes.

a) Training objective: The original SDXL-Inpaint is constructed from a 9-channel input UNet, with the loss function defined as:

Linp = Et,x0,m,ct,ε∼N (0,1)[∥ε −εθ(yt,t,ct)∥22] . (2) Here,

yt = cat(xt,m,E((1−m)·x0)) , (3) xt = √α¯tx0 + 1−α¯tε . (4)

In these equations, x0 represents the ground truth image, ct denotes text prompts, m is the binary foreground mask, ε is the random Gaussian noise, and E is an image encoder.

Compared to the standard text-based diffusion model [33], this inpainting model retains the same forward diffusion strategy, but it concatenates the mask and inverse-masked latent image into the denoising input. We employ Linp for our curated “ideal” ground truth images. We increase the size of the training dataset by also incorporating isometric images with foreground objects (full isometric), where only a partial background region can be used as training ground truth. In this case, we simply add noise and learn the denoising of the background area:

xˆt = m·(√α¯tx0 + 1−α¯tε)+(1−m)·x0 , (5)

yˆt = cat(xˆt,m,E(m·x0)) , (6)

Linppart = Et,x0,m,ct,ε∼N (0,1)[∥m·(ε −εθ(yˆt,t,ct))∥22] . (7)

During the training phase of the inpainting model, all three types of training data are thoroughly shuffled and randomly sampled.

Another obstacle that hinders the inpainting performance is caused by a distribution shift of denoising between training and inference. This shift occurs in two ways: masked regions are background during training, while masked regions are foreground during inference. Additionally, despite our efforts to mimic real foreground masks by intersecting pseudo foreground masks with random shapes, a slight discrepancy remains. Step-Unrolled Denoising (SUD) diffusion technique [34] is designed to tackle this issue. We adapted it in our inpainting process, as detailed in Algorithm 1. Note that the SUD step is applied only in the later stages of the training, as it is only effective when the prediction can produce plausible results.

Algorithm 1 Inpainting training step with SUD Input: rgb image x0, background mask mbg, text prompt ct Training loop:

t ← U(0,1) ε ← N (0,1)

mpfg ← pseudo foreground mask library m = mpfg ·mrandom ·mbg xt = √αtx0 +√1−αtε if unroll step then

mˆ ← mask foreground No gradients : εpredˆ = fθ([xt,mˆ,(1−mˆ)·x0],t) xpredˆ = (xt −

√1−αtεpredˆ )/√αt xˆt = √αtxpredˆ +√1−αtε ε¯ = (xˆt −

√αtx0)/√1−αt εpred¯ = εθ([xˆt,m,(1−m)·x0],t,ct) L = ∥ε¯ −εpred¯ ∥22

### else

εpred = εθ([xt,m,(1−m)·x0],t,ct) L = ∥ε −εpred∥22

### end if

- B. Visual Scene Understanding

We decompose the 3D scene into three main components, namely: terrain heightmap, texture splatmap, and foreground objects. The heightmap controls the terrain’s shape. The texture splatmap along with its corresponding texture tiles determines the terrain’s texture and color. Splatmaps are commonly used in game engines that act as an alpha composition of tiled texture to obtain a textured terrain. Foreground objects’ instance and pose establish the type, location and direction of 3D objects being placed into the scene.

1) Terrain HeightMap: After the basemap inpainting, there are still regions of the scene that are partially occluded, for example, the backside of a mountain. We reconstruct a coarse, but watertight 3D terrain mesh from the inpainted 2D terrain map. This mesh will be the foundation for parsing game terrain parameters, enabling high-fidelity scene generation within the game environment. Unlike previous approaches which rely on incremental scene reconstruction, our method takes advantage of the isometric perspective, which offers a comprehensive overview of the environment, with minimum occlusions. This allows us to recover the majority of colour and depth information of the scene using just a single image. To infer the scene depth, we adopt the Depth-Anything method [35], followed by reprojecting the RGB-D image into space to obtain a colored point cloud. Then, we reconstruct the complete mesh using the Poisson reconstruction technique.

Given the coarse terrain mesh in the isometric viewpoint, one can easily rotate the view to obtain a bird-eye’s view (BEV) of the terrain. This provides the depth, d of the terrain from a camera looking directly down along the gravity, and the heightmap, h is simply the reverse of the depth. Specifically, h = dmax −d.

The rough color reference also includes water regions, which are segmented out as previously described. For the water category, we not only add a water asset to the scene but also lower the terrain height in those areas to ensure the terrain is positioned below the water level.

- 2) Texture SplatMap: The rough terrain mesh provides

a rough colour reference when rotated into BEV. However, using this directly for the terrain will result in blurry, lowquality visuals in-game. Popular game engines (e.g. Unity, UE) handle terrain texturing using N texture tiles and N channels splatmap, where the splatmap acts as an alpha composition for the corresponding texture tile. Specifically, we obtain the texture splatmap by performing segmentation using Segment Everything [36] on the rendered RGB image of the terrain mesh in BEV, and use Osprey [37] to obtain the semantic category for each segmentation mask (e.g. grass, rock, road). Then, we automatically pick from a list of texture tiles from the corresponding category and assign them to the terrain. This ensures that the terrain texture remains sharp even when viewed from a close distance.

- 3) Foreground Objects: For above-ground objects like

buildings or other landmarks, we apply the instance segmentation function of the Sam model [36] to obtain the 2D masks for each of the foreground objects.

[Figure 23]

[Figure 24]

Fig. 4: Object footprint estimation, showing an illustrative example of obtaining a building footprint and height. On the left: Black region is the instance mask of a building, red box shows the homography-warped 2D object bounding box, blue box shows the estimated object footprint. On the right: Blue filled box shows the inverse-homography-warped object footprint, which can also be used to estimate the object height.

The obtained instance segmentation mask of each object helps estimate their pose within the 3D scene. Using the characteristic of isometric images, where objects are typically at a 45◦ angle from the camera, we design a method to estimate their footprints. Exploiting the specific viewpoint of an isometric projection, we warp the instance segmentation image using a homography. Then, using the homographywarped 2D object bounding box and instance segmentation from Grounded Segment Anything [38], we can estimate the object footprint in the rotated view as shown on the left of Fig. 4. The coordinates (x1,y1) is the maximum x and y coordinates of the instance mask. (x2,y1) and (x1,y2) are the intersection points of y = y1 and x = x1 with the two sides of the warped 2D object bounding box (red box) as shown in the left image of Fig. 4. Thanks to the advantages of the isometric projection, we warp the estimated object footprint back into the isometric image, and estimate the object height as shown in the right image of Fig. 4. Then, using the estimated depth, we transform the object’s footprint into its corresponding 3D location.

C. Procedural 3D Scene Generation

By leveraging the semantic and geometric understanding obtained in the previous module, we can either use 3D asset retrieval or generation, in combination with procedural generation technique for scene creation. Finally, the 3D scene is composed and rendered within the off-the-shelf 3D game engines (such as Unity or Unreal Engine). In this work, we use the Unity game engine for building our 3D interactive environment, for Unity offers valuable optimization features for terrain, vegetation, and animation, ensuring optimized runtime performance. Other game engines or 3D platforms (such as Blender) can be easily used as well.

Given the heightmap, splatmap and chosen texture tiles, it is straightforward to apply them to a Unity terrain asset. This provides us with a basic 3D terrain featuring high-resolution textures. Depending on the texture type, we can designate the vegetation and small objects that can be placed or grown on them. For instance, a grass texture may include assets like grass, flowers, and rocks, which are placed across the terrain

[Figure 25]

[Figure 26]

[Figure 27]

(a) A Pokemon-style isometric town around a crag with a river.

[Figure 28]

[Figure 29]

[Figure 30]

- (b) An isometric view of a snowy landscape with a river, a waterfall, some trees and several animals, such as deers, mooses, and bears.

[Figure 31]

[Figure 32]

[Figure 33]

- (c) A Pokemon-style isometric town on a beach with buildings and umbrellas.

Fig. 5: Results showing the generated isometric reference images (column-2), along with the inpainted basemaps (column-3). Sketch color codes: blue=water, yellow=building, orange=bridge, gray=roads, and green=trees.

using established procedural content generation techniques.

For larger objects, we use the segmented instances of the foreground objects (e.g. building, bridge) to perform either object retrieval or 3D object generation. For the former, we search the most similar instance of 3D object from the Objaverse dataset, by comparing their CLIP scores. For the latter, the 3D asset s are generated using recent 2D-to-3D asset generation AI models such as the LRM [2] or else [4], [39], [40]. These generated 3D objects are then placed into the scene following the foreground object pose estimated in the previous steps, completing the 3D scene.

IV. RESULTS A. Training and Inference Details.

We used AdamW optimizer with a learning rate of 1e-5 for training/fine-tuning both the ControlNet and inpainting models. The pre-trained diffusion models adopted in our experiments were SDXL-base model [41] for ControlNet, and SDXL-Inpaint model [31] for inpainting. We set the rank parameter of 64 in LoRA for inpainting. The ControlNet was fine-tuned on a single NVIDIA A100 GPU, completing 50K steps in around 10 hours. The inpainting model was trained on 4x V100 GPUs for 100K steps in about 60 hours. The total inference time of the entire pipeline is about 3 minutes using a single V100 GPU.

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

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Fig. 6: Basemap inpainting results of SDXL-Inpaint (middle) and ours (right) on the isometric images (left)

We collected datasets to train both the ControlNet and the Inpainting model respectively. The ControlNet dataset comprises 10,000 isometric view game scene images generated by SDXL [41], paired with corresponding text prompts from InstructBlip [42] and associated sketches. These sketches were generated by combining results obtained by several StoA foudnation models, inlcuding Grounding DINO [43], Segment Anything [36], and Osprey [37]. Since we did not have any isometric basemap as the ground truth, we curated an inpainting dataset from three sources: 5,000 isometric images with foreground objects, 4,000 manually filtered perspective images of empty terrains inpainted using [44], and 1000 pure texture images.

B. Isometric 2D Image Generation.

Fig. 5 shows representative results using our ControlNet and inpainting models with a diverse set of user sketches and prompts. These results demonstrate ControlNet’s ability to accurately follow sketch layouts and apply the scene style dictated by the prompt. The inpainting model generates clean basemaps that consistently align with the full isometric

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

- Fig. 7: Scene understanding results showing heightmap, object placement bounding boxes and object reference images for Fig. 1, 5a and 5c.

images, even when the foreground masks cover a significant portion of the image.

As shown in Fig. 5, the ControlNet offers flexibility to the user’s sketch, accommodating various categories like wateronly in Fig. 5a and 5b, and three categories in Fig. 5c respectively. With the same sketch, Fig. 5a and 5b produce distinct scenes by applying different styles from the texts.

How to balance the influences of the sketch condition and the textual prompt guidance is the key. Our SAL-enhanced ControlNet simplifies this balancing process by allowing casual (not precise) user sketches, occasionally adding extra objects or expanding patch areas to implement the user’s design intention. For example, in Fig. 5b, the river and waterfall blend coherently to meet both text and sketch requirements. In Fig. 5c, eight buildings are added to match the phrase “town with many buildings” while respecting the original user-drawn sketch.

- C. Inpainting Comparisons

We compare inpainting results with SDXL-Inpaint [31] on the isometric images in Fig. 6. We use a positive prompt of “an empty terrain map with nothing rising above the surface. This is a landscape without any buildings, vegetation, or bridges.” and a negative prompt of “buildings, vegetation, trees, bridges, artifacts, low-quality”. Our model successfully produced clean and consistent basemaps, whereas SDXL-Inpaint tended to substitute buildings and trees with artifacts.

- D. Visual Scene Understanding

Given the 2D isometric and empty basemap, our visual scene understanding module recovers the instance-level semantic segmentation of the foreground objects, estimates the isometric depth, recovers the rough terrain mesh, renders the BEV heightmap and color image, segments the splatmap and recovers the foreground object placement. Figure 7 shows examples of the generated heightmap, BEV object placement and the extracted object reference images.

- E. Procedural 3D Scene Generation

Figure 8 shows three 3D scenes generated from the isometric images from Fig. 1, 5a and 5c. It show that the layout and texture style of the 3D scenes are well aligned with the associated sketch and isometric image.

The objects in the first scene is retrieved from the Objaverse, while objects in the second and third scenes are generated by [39] using the object instance images extracted from the isometric image. These objects not only harmonize with the scene’s texture style but are also automatically and accurately scaled, oriented, and positioned in the 3D scene according to the BEV footprint. Note that variations in material composition and lighting dynamics have led to a slight discrepancy in color between the rendered images of 3D scenes and the reference image. More example results are shown in Fig. 9 and 10.

- F. Limitations

Our current implementation adopts a multi-stage pipeline involving many intermediate stages. Errors can be easily accumulated, which sometimes requires the user to restart from a different noise seed. One potential remedy is to concurrently generate multiple modalities at the same time, such as RGB, semantic, depth, surface material, and object footprint, and fuse these intermediate results until a coherent final result is obtained. Concurrently generating foreground and background layers is also a possible solution, by for exmaple applying the newly proposed LayerDiffusion method [45] . Currently, in our pipeline, terrain texture and terrain materials are obtained solely by retrieving a terrain database, which limits the diversity of terrain textures. In the future, we plan to develop diffusion based texture-generation models similar to [46], [47].

V. CONCLUSION

We have proposed a novel approach called Sketch2Scene for generating 3D interactive scenes from users’ casual sketches and text prompts. To address the main challenge of insufficient large-scale training data for 3D scenes, we leverage and improve pre-trained large-scale 2D diffusion models for the task. We provides two innovations to existing diffusion models: (1) SAL-enhanced ControlNet, and (2) step-unrolled diffusion inpainting. In contrast to other recent generative techniques for 3D scene generation (e.g., using SDS loss [9], or direct triplane regression [24]), our approach generates high quality and interactive 3D scenes with vivid 3D assets that can be seamlessly integrated into existing game engines, ready for many downstream applications. We also discussed limitations and possible remedies in the paper. The reader is invited to watch our companion video on our project page (https://xrvisionlabs.github.io/Sketch2Scene/).

REFERENCES

[1] Z. Sun, Y. Wang, Y. Ng, Y. Xu, D. Dong, H. Li, and P. Ji, “T$ˆ3$-s2s: Training-free triplet tuning for sketch to scene synthesis in controllable concept art generation,” Transactions on Machine Learning Research, 2026, featured Certification, J2C Certification. [Online]. Available: https://openreview.net/forum?id=lyn2BgKQ8F

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Prompt: An isometric game map featuring a river at the center and bridges crossing the river.

Prompt: A Pokemon-style isometric town around a crag with a river.

Prompt: A Pokemon-style isometric town on a beach with buildings and umbrellas.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

- Fig. 8: More example 3D scene generation results. From left to right: Different views of the generated 3D scenes from the isometric images of Fig. 1, Fig. 5a and 5c.

[Figure 70]

- Fig. 9: An additional example of 3D scene generation from text and sketch. The top left displays the input sketch and text, along with the generated isometric image. The other three images are rendered from the 3D scene with different viewpoints. The prompt is: “A scene in ice-age with a wooden shed, pine trees and some animals.”

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

Prompt: An isometric game map featuring a river at the center and bridges crossing the river.

|[Figure 74]|
|---|

|[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]|
|---|

- Fig. 10: 3D Scene Editing: By varying the parameters of the 3D assets, e.g. the type and color of the trees, our method can facilitate content-editing and style transfer for the same 3D game scene.

- [2] Y. Hong, K. Zhang, J. Gu, S. Bi, Y. Zhou, D. Liu, F. Liu, K. Sunkavalli, T. Bui, and H. Tan, “LRM: Large reconstruction model for single image to 3d,” ICLR, 2024.
- [3] Z. Wang, Y. Wang, Y. Chen, C. Xiang, S. Chen, D. Yu, C. Li, H. Su, and J. Zhu, “CRM: Single image to 3d textured mesh with convolutional reconstruction model,” arXiv preprint arXiv:2403.05034, 2024.
- [4] X. Wei, K. Zhang, S. Bi, H. Tan, F. Luan, V. Deschaintre, K. Sunkavalli, H. Su, and Z. Xu, “MeshLRM: Large reconstruction model for high-quality mesh,” arXiv preprint arXiv:2404.12385, 2024.
- [5] M. Deitke, R. Liu, M. Wallingford, H. Ngo, O. Michel, A. Kusupati, A. Fan, C. Laforte, V. Voleti, S. Y. Gadre et al., “Objaverse-xl: A universe of 10m+ 3d objects,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [6] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695.
- [7] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” 2023.
- [8] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in International Conference on Machine Learning. PMLR, 2021, pp. 8821–8831.
- [9] L. H¨ollein, A. Cao, A. Owens, J. Johnson, and M. Nießner, “Text2room: Extracting textured 3d meshes from 2d text-to-image models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7909–7920.
- [10] R. Fridman, A. Abecasis, Y. Kasten, and T. Dekel, “Scenescape: Textdriven consistent scene generation,” arXiv preprint arXiv:2302.01133, 2023.
- [11] J. Shriram, A. Trevithick, L. Liu, and R. Ramamoorthi, “Realmdreamer: Text-driven 3d scene generation with inpainting and depth diffusion,” arXiv preprint arXiv:2404.07199, 2024.
- [12] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 1–14, 2023.

- [13] R. Po and G. Wetzstein, “Compositional 3d scene generation using locally conditioned diffusion,” arXiv:2303.12218, 2023.
- [14] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Textto-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988, 2022.
- [15] D. Epstein, B. Poole, B. Mildenhall, A. A. Efros, and A. Holynski, “Disentangled 3d scene generation with layout learning,” arXiv preprint arXiv:2402.16936, 2024.
- [16] J. Schult, S. Tsai, L. H¨ollein, B. Wu, J. Wang, C.-Y. Ma, K. Li, X. Wang, F. Wimbauer, Z. He, P. Zhang, B. Leibe, P. Vajda, and J. Hou, “Controlroom3d: Room generation using semantic proxy rooms,” arXiv:2312.05208, 2023.
- [17] C. Fang, X. Hu, K. Luo, and P. Tan, “Ctrl-room: Controllable textto-3d room meshes generation with layout constraints,” arXiv preprint arXiv:2310.03602, 2023.
- [18] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3836–3847.
- [19] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler, “Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis,” Advances in Neural Information Processing Systems, vol. 34, pp. 6087–6101, 2021.
- [20] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.
- [21] J. Deng, W. Chai, J. Guo, Q. Huang, W. Hu, J.-N. Hwang, and G. Wang, “Citygen: Infinite and controllable 3d city layout generation,” arXiv preprint arXiv:2312.01508, 2023.
- [22] H.-X. Yu, H. Duan, J. Hur, K. Sargent, M. Rubinstein, W. T. Freeman, F. Cole, D. Sun, N. Snavely, J. Wu et al., “Wonderjourney: Going from anywhere to everywhere,” arXiv preprint arXiv:2312.03884, 2023.
- [23] X. Ren, J. Huang, X. Zeng, K. Museth, S. Fidler, and F. Williams, “Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies,” arXiv preprint arXiv:2312.03806, 2023.
- [24] Z. Wu, Y. Li, H. Yan, T. Shang, W. Sun, S. Wang, R. Cui, W. Liu, H. Sato, H. Li et al., “Blockfusion: Expandable 3d scene generation

- using latent tri-plane extrapolation,” arXiv preprint arXiv:2401.17053, 2024.
- [25] H. Yan, Y. Li, Z. Wu, S. Chen, W. Sun, T. Shang, W. Liu, T. Chen, X. Dai, C. Ma et al., “Frankenstein: Generating semantic-compositional 3d scenes in one tri-plane,” arXiv preprint arXiv:2403.16210, 2024.
- [26] C. Sun, J. Han, W. Deng, X. Wang, Z. Qin, and S. Gould, “3d-gpt: Procedural 3d modeling with large language models,” arXiv preprint arXiv:2310.12945, 2023.
- [27] A. Raistrick, L. Lipson, Z. Ma, L. Mei, M. Wang, Y. Zuo, K. Kayan, H. Wen, B. Han, Y. Wang et al., “Infinite photorealistic worlds using procedural generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12630–12641.
- [28] M. Zhou, J. Hou, C. Luo, Y. Wang, Z. Zhang, and J. Peng, “Scenex: Procedural controllable large-scale scene generation via large-language models,” arXiv preprint arXiv:2403.15698, 2024.
- [29] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11461–11471.
- [30] S. Wang, C. Saharia, C. Montgomery, J. Pont-Tuset, S. Noy, S. Pellegrini, Y. Onoe, S. Laszlo, D. J. Fleet, R. Soricut et al., “Imagen editor and editbench: Advancing and evaluating text-guided image inpainting,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18359–18369.
- [31] diffusers, “stable-diffusion-xl-1.0-inpainting-0.1,” 2024.
- [32] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.
- [33] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
- [34] S. Saxena, A. Kar, M. Norouzi, and D. J. Fleet, “Monocular depth estimation using diffusion models,” arXiv preprint arXiv:2302.14816, 2023.
- [35] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in CVPR, 2024.
- [36] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Doll´ar, and R. Girshick, “Segment anything,” arXiv:2304.02643, 2023.
- [37] Y. Yuan, W. Li, J. Liu, D. Tang, X. Luo, C. Qin, L. Zhang, and J. Zhu, “Osprey: Pixel understanding with visual instruction tuning,” arXiv preprint arXiv:2312.10032, 2023.
- [38] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.
- [39] hyperhuman, “https://hyperhuman.deemos.com/rodin,” 2024.
- [40] J. Yang, Z. Cheng, Y. Duan, P. Ji, and H. Li, “Consistnet: Enforcing 3d consistency for multi-view images diffusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 7079–7088.
- [41] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952, 2023.
- [42] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose visionlanguage models with instruction tuning,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [43] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” arXiv preprint arXiv:2303.05499, 2023.
- [44] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, and V. Lempitsky, “Resolution-robust large mask inpainting with fourier convolutions,” arXiv preprint arXiv:2109.07161, 2021.
- [45] L. Zhang and M. Agrawala, “Transparent image layer diffusion using latent transparency,” arXiv preprint arXiv:2402.17113, 2024.
- [46] Y. Wang, A. Holynski, B. L. Curless, and S. M. Seitz, “Infinite texture: Text-guided high resolution diffusion texture synthesis,” 2024.
- [47] Y.-Y. Yeh and J.-B. e. a. Huang, “Texturedreamer: Image-guided texture synthesis through geometry-aware diffusion,” arXiv preprint arXiv:2401.09416, 2024.

APPENDIX A. Dataset

a) ControlNet-Dataset: While several successful text/sketch-to-image works have already been presented, none of them focus specifically on isometric view game scenes. Since collecting a large number of isometric view game scene images for training is challenging, we created a dataset by generating these images using the SDXL model. The dataset is designed to validate the effectiveness of our method and reduce the domain gap with the original model.

We first used text prompts as input and employed the SDXL model to generate 10,000 isometric view game scene images. To label these images for training, we utilize Grounding DINO and Segment Anything to detect and segment semantic masks for elements such as buildings, trees, and boats. Additionally, we used Segment Anything along with Osprey to generate masks for irregularly shaped semantic elements such as water bodies, bridges, and roads. We manually annotated road element masks in 2,000 images for better accuracy. All images were then captioned using InstructBlip [42] to obtain detailed text prompts.

b) Inpainting-Dataset: Ideal training data for our inpainting model would be large-scale, pure isometric basemap images, but collecting or generating these is challenging. We found a viable alternative by combining three types of readily available data as mentioned in the main paper. Figure 11 shows supplementary examples of this data. The masks during the inference phase of inpainting are foreground masks of full isometric images. as illustrated in Fig. 12. Supplementary examples of masks in the training of inpainting are displayed in 11.

[Figure 78]

- Fig. 11: Examples of inpainting training data. From left to right columns: full isometric, inpainted from perspective semi-empty images, pure texture maps.

[Figure 79]

[Figure 80]

[Figure 81]

Fig. 12: Examples of inpainting inference masks, where the mask regions are foreground objects.

B. Examples of inpainting training data

To ensure a diverse range of masking scenarios and minimize the distribution discrepancy between training and inference masks, we utilize the intersection of random masks and pseudo-foreground masks for training the basemaps. These pseudo-foreground masks are randomly sampled from the foreground masks of the isometric dataset. Examples of these intersection results are shown in Fig 13, 14, and 15. For isometric images with foreground objects, only the background area can be masked and considered as inpainted ground truth. We therefore use the intersection of background masks and random masks as the training masks.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

- Fig. 13: Examples of inpainting training data: empty map.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

- Fig. 14: Examples of inpainting training data: texture images.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

- Fig. 15: Examples of inpainting training data: full isometric images.

C. Comparison with SDXL Inpainting

- Fig. 16 compares several inpainting results between our

proposed method and SDXL Inpainting. D. 2D Image Generation Results

- Fig. 17 demonstrates a variety of supplementary examples

of isometric images generated by ControlNet from texts and sketches, along with the results of basemap inpainting.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Fig. 16: Comparison of basemap inpainting between ours (right) and SDXL-Inpaint (middle) on isometric test dataset (left). From the above results we can see our method produces much cleaner empty basemaps of the terrain.

[Figure 109]

[Figure 110]

[Figure 111]

(a) An Pokemon-style isometric town around a craggy coastline

[Figure 112]

[Figure 113]

[Figure 114]

(b) A beautiful isometric world of ice and snow.

[Figure 115]

[Figure 116]

[Figure 117]

- (c) A GTA-style coastal town with charming seafront, colorful buildings, and fishing boats dotting the harbor.

[Figure 118]

[Figure 119]

[Figure 120]

- (d) The image depicts an isometric view of a mountainous landscape with a river, several houses, and a waterfall.

Fig. 17: More results of isometric image generation and basemap inpainting.

