## MAGRITTE: MANIPULATIVE AND GENERATIVE 3D REALIZATION FROM IMAGE, TOPVIEW AND TEXT

Takayuki Hara The University of Tokyo hara@mi.t.u-tokyo.ac.jp

Tatsuya Harada The University of Tokyo / RIKEN harada@mi.t.u-tokyo.ac.jp

# arXiv:2404.00345v2[cs.CV]27Nov2024

### ABSTRACT

The generation of 3D scenes from user-specified conditions offers a promising avenue for alleviating the production burden in 3D applications. Previous studies required significant effort to realize the desired scene, owing to limited control conditions. We propose a method for controlling and generating 3D scenes under multimodal conditions using partial images, layout information represented in the top view, and text prompts. Combining these conditions to generate a 3D scene involves the following significant difficulties: (1) the creation of large datasets, (2) reflection on the interaction of multimodal conditions, and (3) domain dependence of the layout conditions. We decompose the process of 3D scene generation into 2D image generation from the given conditions and 3D scene generation from 2D images. 2D image generation is achieved by fine-tuning a pretrained text-toimage model with a small artificial dataset of partial images and layouts, and 3D scene generation is achieved by layout-conditioned depth estimation and neural radiance fields (NeRF), thereby avoiding

The use of a common representation of spatial information using consideration of multimodal condition interactions and reduces the control. The experimental results qualitatively and quantitatively

|the creation of large datasets. 360-degree images allows for the domain dependence of the layout demonstrated that the proposed|
|---|

method can generate 3D scenes in diverse domains, from indoor to outdoor, according to multimodal conditions. A project website with supplementary video is here https://hara012.github.io/MaGRITTe-project.

Keywords 3D scene generation · 360-degree image generation · image outpainting · text-to-3D · layout-to-3D

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

table

view window sink

[Figure 6]

[Figure 7]

[Figure 8]

|a photography of medieval European kitchen|
|---|

Input conditions

360-degree RGB-D Perspective views

Figure 1: From a given partial image, layout information represented in top view, and text prompts, our method generates a 3D scene represented by the 360-degree RGB-D, and NeRF. Free perspective views can be rendered from the NeRF model.

図3

### 1 Introduction

3D scene generation under user-specified conditions is a fundamental task in the fields of computer vision and graphics. In particular, the generation of 3D scenes extending in all directions from the observer’s viewpoint is a promising technology that reduces the burden and time of creators and provides them with new ideas for creation in 3D applications such as VR/AR, digital twins, and the metaverse.

In recent years, 3D scene generation under user-specified conditions using generative models [31, 45, 19, 58, 51, 26] has been extensively studied. A wide range of methods exist for generating 3D scenes from parital images [14, 6, 15, 12], layout information such as floor plans and bird’s-eye views [59, 5, 29, 70, 10, 49], and text prompts [64, 50, 27, 55]. However, these methods are limited by the conditions they can take as input, making it difficult to generate the 3D scene intended by the user. This is due to the fact that each condition has its own advantages and disadvantages. For example, when partial images are given, it is possible to present a detailed appearance; however, it is difficult to create information outside the image; when a layout is given, it is possible to accurately describe object alignment but not to specify a detailed appearance; when text is given as a condition, it is suitable for specifying the overall context; however, it is difficult to determine the exact shape and appearance of objects.

Considering these problems, we propose a method for generating 3D scenes by simultaneously providing a combination of three conditions: partial images, layout information represented in the top view, and text prompts (fig. 1). This approach aims to compensate for the shortcomings of each condition in a complementary manner, making it easier to create the 3D scenes intended by the creator. That is, details of appearance from partial images, shape and object placement from layout information, and overall context can be controlled using text prompts.

Integrating partial images, layouts, and texts to control a 3D scene involves the following significant difficulties that cannot be addressed by a simple combination of existing methods: (1) creation of large datasets, (2) reflection of the interaction of multimodal conditions, and (3) domain dependence of the layout representations. To overcome these difficulties, we initially decomposed the process of 3D scene generation into two steps: 2D image generation from the given conditions and 3D generation from 2D images. For 2D image generation, our approach is to create small artificial datasets for partial images and layout conditions and fine-tune the text-to-image model trained on a large dataset. We then generated a 3D scene from a 2D image using layout-conditioned monocular depth estimation and training NeRF [40]. This approach eliminates the need to create large datasets of 3D scenes. This study aimed to improve scene consistency and reduce computational costs using 360-degree images for 2D image generation. To address the second issue, which reflects the interaction of multimodal conditions, we encoded the input conditions into a common latent space in the form of equirectangular projection (ERP) for 360-degree images. To address the third issue of domain dependence of layout representations, we present a framework for incorporating domain-specific top-view representations with less effort by converting them into more generic intermediate representations of depth and semantic maps in ERP format. This allows for generating various scenes from indoor to outdoor by simply replacing the converter.

The contributions of this study are as follows:

- • We introduce a method to control and generate 3D scenes from partial images, layouts, and texts, complementing the advantages of each condition.
- • We present a method that avoids the need for creating large datasets by fine-tuning a pre-trained large-scale text-to-image model with a small artificial dataset of partial images and layouts for 2D image generation, and by generating 3D scenes from 2D images through layout-conditioned depth estimation and training NeRF.
- • We address the integration of different modalities by converting the input information into ERP format, passing it through an encoder, and embedding the information in the same latent space.
- • We present a framework for generating various scenes from indoor to outdoor with a module for converting top view layout representations into depth maps and semantic maps in ERP format.
- • Experimental results validate that the proposed method can generate 3D scenes with controlled appearance, geometry, and overall context based on input information, even beyond the dataset used for fine-tuning.

### 2 Related Work

- 2.1 3D Scene Generation

##### 3D scene generation involves the creation of a model of a 3D space that includes objects and backgrounds, based on user-specified conditions. In recent years, the use of generative models, such as VAE [31, 45], GAN [19], autoregressive

models [58], and diffusion models [51, 26], has made rapid progress. There are methods to generate a 3D scene from random variables [38, 8], from one or a few images [14, 6, 36, 15, 12], from layout information such as floor plans [59, 5], bird’s-eye views (semantic maps in top view) [29, 70], terrain maps [10] and 3D proxies [49], and as well as from text prompts [64, 50, 27, 55, 17]. However, each method has its own advantages and disadvantages in terms of scene control characteristics, and it is difficult to generate a 3D scene that appropriately reflects the intentions. We propose a method to address these challenges by integrating partial images, layout information, and text prompts as input conditions in a complementary manner. Furthermore, while layout conditions need to be designed for each domain, the proposed method switches between converters for layout representations, enabling the generation of a variety of scenes from indoor to outdoor.

#### 2.2 Scene Generation Using 360-Degree Image

Image generation methods have been studied for 360-degree images that record the field of view in all directions from a single observer’s viewpoint. Methods to generate 360-degree images from one or a few normal images [18, 52, 3, 2, 22, 4, 23, 65] and text prompts [11, 63, 57] have been reported. Methods for panoramic three-dimensional structure prediction were also proposed [53, 54].

Studies have also extended the observer space to generate 3D scenes with six degrees of freedom (DoF) from 360-degree RGB-D. In [28, 21, 32, 62], methods were proposed for constructing a 6-DoF 3D scene by training the NeRF from 360-degree RGB-D. LDM3D [55] shows a series of pipelines that add channels of depth to the latent diffusion model (LDM) [46], generate 360-degree RGB-D from the text, and mesh it. Generating 3D scenes via 360-degree images is advantageous in terms of guaranteeing scene consistency and reducing computation. Our research attempts to generate 360-degree images from multiple conditions and 6-DoF 3D scenes by layout-conditioned depth estimation and training the NeRF.

#### 2.3 Monocular Depth Estimation

Monocular depth estimation involves estimating the depth of each pixel in a single RGB image. In recent years, deep learning-based methods have progressed significantly, and methods based on convolutional neural networks [48, 35, 33, 67, 68, 71, 39] and transformers [7, 13, 69, 56, 43] have been proposed. Monocular depth estimation for 360-degree images was also investigated [74, 34, 16, 60, 75, 61, 41, 44, 1]. However, since the accuracy of conventional monocular depth estimation is not sufficient, this study aims to improve accuracy by combining it with layout conditions.

### 3 Proposed Method

This section describes the proposed method called MaGRITTe, that generates 3D scenes under multiple conditions. fig. 2 illustrates the overview of our method. Three input conditions are considered: a partial image, layout information represented in the top view, text prompts, and outputs from a 360-degree RGB-D and NeRF model. The proposed method comprises four steps: (a) ERP conversion of partial images and layouts, (b) 360-degree RGB image generation, (c) layout-conditioned depth estimation, and (d) NeRF training. The following sections describe each step.

#### 3.1 Conversion of Partial Image and Layout

First, we describe the conversion of the partial image and layout in (a) of fig. 2. This study uses two layout representations, floor plans and terrain maps, for indoor and outdoor scenes, respectively.

#### 3.1.1 Floor Plans

A floor plan is a top-view representation of the room shape and the position/size/class of objects. The room shape comprises the two-dimensional coordinates of the corners and the height positions of the floor and ceiling, based on the assumption that the walls stand vertically. The objects are specified by 2D bounding box, height from the floor at the top and bottom, and class, such as chair or table.

#### 3.1.2 Terrain Maps

As shown in fig. 3, a terrain map describes the height of the terrain relative to the horizontal plane. This is a set RH

ter×Wter that constitutes a Hter × Wter grid with the height of the ground surface at each grid point.

|overview_3d|
|---|

MaGRITTe: Manipulative and Generative 3D Realization from Image, Topview and Text

360-degree RGB

[Figure 9]

(a)

(b)

[Figure 10]

[Figure 11]

|Converter| |
|---|---|
| | |

Generator

(d)

Partial image

Partial image

|Trainer|
|---|

[Figure 12]

(c)

[Figure 13]

Trainer

Observer position

Estimator

[Figure 14]

(a)

chair FoV

|Converter|
|---|

Coarse depth

360-degree depth

[Figure 15]

Layout

[Figure 16]

(floor plan / terrain map)

[Figure 17]

|a photography of the living room with the warmth of wood|
|---|

Semantic map

NeRF model

Text

Input Output

- Figure 2: Overview of the proposed method to generate 360-degree RGB-D and NeRF models from a partial image, layouts and text prompts. (a) The partial image is converted to an ERP image from the observer position with the specified direction and field-of-view (FoV). The layout represented the in top view is converted to a coarse depth and a semantic map in ERP format with the observer position as the projection center. (b) These ERP images and texts are combined to generate a 360-degree RGB. (c) The generated RGB is combined with the coarse depth to estimate the fine depth. (d) a NeRF model is

|trained from 360-degree|
|---|

tr e RGB-D.

図7

[Figure 18]

Observer

position FoV

|Converter| |
|---|---|
| | |

|Converter| |
|---|---|
| | |

[Figure 19]

[Figure 20]

|[Figure 21]|
|---|

Terrain map

Partial image

Coarse depth (ERP)

Partial image (ERP)

- Figure 3: The case of using a terrain map for the layout format. The partial image and the terrain map are converted into ERP images from the observer’s viewpoint, respectively.

図2

#### 3.1.3 ERP Conversion

The observer position and field of view (FoV) of the partial image are provided in the layout. Based on this information, a partial RGB P ∈ RH

ERP×WERP×3, coarse depth D ∈ RH

ERP×WERP, and semantic map S ∈ {0,1}H

ERP×WERP×C

are created in the ERP format, as shown in fig. 2 (a), where HERP and WERP are the height and width of the ERP image, respectively, and C denotes the number of classes. The semantic map takes Sijc = 1 when an object of class c exists at position (i,j) and Sijc = 0 otherwise. For floor plans, the distance from the observer’s viewpoint to the room wall is recorded, and for terrain maps, the distance from the observer’s viewpoint to the terrain surface is recorded in ERP format and used as the coarse depth. A semantic map is created for a floor plan; the regions specifying the objects are projected onto the ERP image with the observer position of the partial image as the projection center, and object classes are assigned to the locations of their presence.

#### 3.2 360-Degree RGB Generation

We combine partial images, coarse depths, and semantic maps represented in the ERP format and integrate them with text prompts to generate a 360-degree RGB image. Using the ERP format for the input and output allows the use

|generator|
|---|

MaGRITTe: Manipulative and Generative 3D Realization from Image, Topview and Text

Text prompts and time 𝑡

[Figure 22]

a photography of the living room with the warmth of wood

|Encoder|
|---|

Locked

Partial image

360-degree RGB

[Figure 23]

[Figure 24]

𝑡 = 𝑇, ⋯ , 1

ControlNet

|Encoder|
|---|

Coarse depth

Trainable

[Figure 25]

Trainable

|Decoder|
|---|

|Denoiser| |
|---|---|
| | |

𝑧

𝑧

Locked

Locked

Semantic map

- Figure 4: The pipeline of generating 360-degree RGB from a partial image, coarse depth map, semantic map, and text prompts.

図4

of text-to-image models trained on large datasets. In this study, we employ StableDiffusion (SD) [46], a pre-trained diffusion model with an encoder and decoder, as the base text-to-image model. We fine-tune the model for our purposes using ControlNet [72], which controls the diffusion model with an additional network of conditional inputs. fig. 4 shows the pipeline to generate 360-degree RGB. A partial image, coarse depth, and semantic maps are embedded in the latent space, channel merged, and provided as conditional inputs to ControlNet along with text prompts. This is an improvement on PanoDiff [63], which generates 360-degree images from partial images, and our method embeds layout information into a common latent space in ERP format as well, allowing for interaction between conditions while preserving spatial information. The encoder for partial images is from SD, and the encoder for layout information is a network with the same structure as that used in ControlNet. The weights of the network derived from SD are fixed, and only the weights of the network derived from ControlNet are updated during training.

#### 3.3 Layout-Conditioned Depth Estimation

Next, a fine depth is estimated from the coarse depth and the generated 360-degree RGB. In this study, we propose and compare two methods: end-to-end estimation and depth integration.

#### 3.3.1 End-to-End Estimation

In the end-to-end approach, the depth is estimated using U-Net [47] with a self-attention mechanism [58] with four channels of RGB-D as the input, and one channel of depth as the output. The network is trained to minimize the L1 loss between the network outputs and ground truth. Details of the network configuration are provided in appendix A.1.

#### 3.3.2 Depth Integration

In the depth integration approach, depth estimates are obtained from 360-degree RGB using the monocular depth estimation method, LeRes [71] is employed in this study, and the final depth is obtained so as to minimize the weighted squared error for the coarse depth and depth estimates. Since LeRes is designed for normal field-of-view images, the 360-degree image is projected onto N tangent images, and depth estimation and integration are performed on each

tangent image. Let dˆn ∈ RH

dWd(n = 1,2,··· ,N) be the monocular depth estimate for n-th tangent image in ERP format, where Hd and Wd are the height and width of the depth map, respectively. Since the estimated depth dˆn has unknown scale and offset, it is transformed using the affine transformation coefficient sn ∈ R2 as d˜nsn, where d˜n = (dˆn 1) ∈ RH

dWd is the coarse depth, Φn ∈ RH

dWd×2. We consider the following evaluation function Ldepth, where d0 ∈ RH

dWd×HdWd(n = 0,1,··· ,N) is the weight matrix, and x ∈ RH

dWd is the integrated depth.

N

||x − d˜nsn||2Φn, (1)

Ldepth = ||x − d0||2Φ0 +

n=1

[Figure 26]

[Figure 27]

Structured 3D dataset Our training data

- Figure 5: Semantic map. Regions related to objects are extracted, excluding regions derived from the shape of the room, such as walls, floor, and ceiling, which are enclosed in a bounding box to form a semantic map in the proposed method.

where quadratic form ||v||2Q = v⊤Qv. The fine depth x and coefficients sn(n = 1,2,··· ,N) that minimize Ldepth can be obtained in closed form from the extreme value conditions as follows:

図9

−1

N

N

Φnd˜nsn , (2)

Φn

x =

Φ0d0 +

n=0

n=1

−1 

  , (3)

  

  

  

   =

D1 U1,2 ··· U1,N U2,1 D2 ··· U2,N

- b1
- b2

- s1
- s2

 

... .

. bN

. .

. sN

UN,1 UN,2 ··· DN

where, Dk = d˜⊤k {Φ−k 1 + ( Nn=0\k Φn)−1}−1d˜k, Uk,l = −d˜⊤k Φk( Nn=0 Φn)−1Φld˜l,bk = d˜⊤k Φk( Nn=0 Φn)−1Φ0d0. The derivation of the equation and setting of weights {Φn}Nn=0 are described in appendix A.2.

#### 3.4 Training NeRF

Finally, we train the NeRF model using the generated 360-degree RGB-D. In this study, we employ a method from [21] that can train NeRF by inpainting the occluded regions from a single image.

### 4 Dataset

We fine-tune our model using the following two types of datasets for indoor and outdoor scenes, respectively. We create artificial datasets with layout annotations using computer graphics as the base dataset, whereas datasets without layout annotations are created using actual captured datasets as the auxiliary dataset.

- 4.1 Indoor Scene

For the base dataset, we modified and used a structured 3D dataset [73] containing 3500 synthetic departments (scenes) with 185,985 panoramic renderings for RGB, depth, and semantic maps. The same room had both furnished and unfurnished patterns, and the depth of the unfurnished room was used as the coarse depth. For consistency with the ERP conversion in section 3.1, the semantic map was transformed, as shown in (fig. 5). Each image was annotated with text using BLIP [37] and partial images were created using a perspective projection transformation of 360-degree RGB with random camera parameters. The data were divided into 161,126 samples for training, 2048 samples for validation, and 2048 samples for testing.

For the auxiliary dataset, we used the Matterport 3D dataset [9], which is an indoor real-world 360◦ dataset including 10,800 RGB-D panoramic images. Similar to the structured 3D dataset, partial images and text were annotated. The depth and semantic maps included in the dataset were not used, and zero was assigned as the default value for the coarse depth and semantic map during training. The data were divided into 7675 samples for training and 2174 samples for testing.

- 4.2 Outdoor Scene As the base dataset, we created the SceneDreamer dataset using SceneDreamer [10], which is a model for generating
- 3D scenes. As shown in fig. 6, a 360-degree RGB-D image was generated from random numbers via a terrain map to annotate the partial images and texts. A semantic map was not used in this study because of limited object classes. The data were divided into 12,600 samples for training, 2,052 samples for validation, and 2052 samples for testing.

Random number Generation

Random cropping

|[Figure 28]|[Figure 29]|
|---|---|

[Figure 30]

(SceneDreamer)

[Figure 31]

Generation

360-degree RGB-D

(SceneDreamer)

Partial image

|[Figure 32]|
|---|

|a photography of a group of rocks in the water with trees in the background and a blue sky|
|---|

Text captioning

Projection

Terrain map

(BLIP)

Coarse depth

Text

- Figure 6: Dataset creation for outdoor scene. SceneDreamer [10] generates a terrain map from a random number, and renders 360-degree RGB-D. The generated RGB image is annotated with text using BLIP [37], and partial images are created by a perspective projection transformation of 360-degree RGB with random camera parameters. A coarse depth is converted from the terrain maps

図10

- Table 1: Evaluation results of 360-degree RGB generation on the Modified Structured 3D dataset and the SceneDreamer dataset.

| |Structured3D dataset<br><br>|SceneDreamer dataset|
|---|---|---|
|method<br><br>|PSNR↑ (whole)<br><br>PSNR↑ (partial)<br><br>FID↓ CS↑|PSNR↑ (whole)<br><br>PSNR↑ (partial) FID↓<br><br>CS↑|

PanoDiff [63] 11.59 36.00 21.23 30.75 12.91 37.19 30.94 29.86 MaGRITTe (ours) 12.56 35.39 18.87 30.72 13.29 34.81 29.05 29.93

For the auxiliary dataset, we used the SUN360 dataset [66] which includes various real captured 360-degree RGB images. We extracted only outdoor scenes from the dataset, and partial images and text were annotated. The distance to the horizontal plane was set as the default value for the coarse depth during training. The data were divided into 39,174 training samples and 2048 testing samples.

### 5 Experimental Results

Quantitative and qualitative experiments were conducted to verify the effectiveness of the proposed method, MaGRITTe, for generating 3D scenes under multiple conditions.

#### 5.1 Implementation Details

The partial images, coarse depths, and semantic maps were in ERP format with a resolution of 512×512, and the shape of the latent variable in the LDM was 64 × 64 × 4. We trained the 360-degree RGB generation model based on the pretrained SD v2.1 using the Adam optimizer [30] with a learning rate of 1.0 × 10−5 and batch size of 16. We trained the end-to-end depth estimation model from scratch using the Adam optimizer with a learning rate of 4.5 × 10−6 and batch size of 6. The convolutional layers in the networks use circular padding [23] to resolve the left-right discontinuity in ERP.

#### 5.2 360-Degree RGB Generation

First, we evaluate 360-degree RGB generation. Because there is no comparison method that uses partial images, layouts, and text prompts as inputs to generate a 360-degree image, we compared our method with PanoDiff [63], which is a state-of-the-art 360-degree RGB image generation model that uses partial images and texts. We implemented it and used PanoDiff with the encoder of the layout information removed in MaGRITTe for a fair comparison using the same network configurations and pretrained models.

- table 1 shows the quantitative evaluation results of 360-degree RGB generation on the Structured 3D dataset and the SceneDreamer dataset. We used the peak-signal-to-noise-ratio (PSNR) as the evaluation metric: PSNR (whole) for the entire image between the ground truth and generated images, PSNR (parial) for the region of the partial image given

- by the input. We also emply the FID [25], which is a measure of the divergence of feature distributions between the ground truth and generated images, and the CLIP score (CS) [42, 24], which promptly quantifies the similarity with the input text. PanoDiff is superior in terms of PSNR (partial) and CS, which is a reasonable result since PanoDiff is a method that takes only partial images and text prompts as conditions for image generation. However, MaGRITTe is superior to PSNR (whole) and FID, which indicates that the reproducibility and plausibility of the generated images can be enhanced by considering layout information as a condition as well.
- table 2 shows the results of the evaluation of the controllability of object type and placement. Semantic segmentation [20] was performed on the 360-degree images generated for Structured3D dataset to evaluate precision, recall, and IoU for bounding boxes in the input conditions. MaGRITTe is superior to PanoDiff and produces results closer to the ground truth images, indicating that the condition-aware object placement is realized.

fig. 7 shows the examples of generating a 360-degree RGB image for the test set of the Structured 3D dataset and the SceneDreamer dataset. PanoDiff, which does not use the layout information as a condition, generates images that differ significantly from the ground truth. This may have led to the degradation of PSNR (whole) and FID. Although the image generated by MaGRITTe differs from the ground-truth image at the pixel level, it can generate images with room geometry, terrain, and object placement in accordance with the given conditions.

#### 5.3 360-Degree Depth Generation

Next, we evaluate the depth of the generated 360-degree image. Because the estimated depth has scale and offset degrees of freedom, its value was determined to minimize the squared error with the ground-truth depth, similar to the method presented in [43]. We used the root mean squared error (RMSE) and mean absolute value of the relative error,

∗ i |

AbsRel = M1 Mi=1 |zi−z

zi∗ , where M is the number of pixels, zi is the estimated depth of the ith pixel, and zi∗ is the ground-truth depth of the ith pixel. Pixels at infinity were excluded from evaluation. table 3 shows the results of the quantitative evaluation of depth generation on the Structured 3D dataset and the SceneDreamer dataset. For comparison, the results of 360MonoDepth which is a 360◦ monocular depth estimation [44] method; LeRes (ERP), which is LeRes [71] directly applied to ERP; and LeRes (multi views), which applies LeRes to multiple tangent images of a 360-degree image and integrates the estimated depths in a section 3.3 manner without using coarse depth, are also shown. In terms of RMSE and AbsRel, our method (end-to-end) was the best for the structured 3D dataset, and our method (depth integration) was the best for the SceneDreamer dataset. It was also shown that combining LeRes with coarse depth increased accuracy compared to using LeRes alone. Ours (w/o coarse depth) is an end-to-end depth estimation method that uses only RGB without the coarse depth, and we can see that the accuracy is lower than when using coarse depth in the Structured3D dataset. The end-to-end method is relatively ineffective for the SceneDreamer dataset. This may be because the number of samples in the dataset was small and the depth was estimated to be close to the coarse depth.

#### 5.4 Results in the Wild

We evaluated the results of 3D scene generation based on user-generated conditions outside the dataset used for fine-tuning. Examples of 3D scenes generated by MaGRITTe, conditioned on partial images, layouts, and text, are shown in figs. 1 and 8. These conditions were created freely by the authors. It can be seen that the generated scene contains the given partial image and conforms to the instructions of the text prompt according to the given layout. These results show that MaGRITTe can generate 3D scenes with the appearance, geometry, and overall context controlled according to the input information, even outside the dataset used for fine-tuning.

#### 5.5 Generation Results from Subset of Conditions

To verify the contribution and robustness of each condition of the proposed method, experiments were conducted to generate 360-degree RGB-D from a subset of partial images, layouts, and text prompts. Generation was performed for the test set of the structured 3D dataset. Because depth estimation in MaGRITTe requires layout information, LeRes (ERP) [71], a monocular depth estimation of ERP images, was used in the absence of layout conditions. table 4 shows

- Table 2: Evaluation results for object type and placement. Note that the object positions in the input condition are given by the bounding boxes as shown in fig. 5, therefore even in ground truth images, it doesn’t match perfectly.

Method Precision Recall IoU

|Ground truth<br><br>|0.482 0.349 0.284|
|---|---|
|PanoDiff|0.245 0.170 0.124<br><br>|
|MaGRITTe (ours)<br><br>|0.424 0.273 0.227|

[Figure 33]

cabinet bed chair sofa table door window bookshelf picture blinds desk shelves curtain dresser mirror floormat clothes books refrigerator TV paper towel shower ca. box white board person nightstand toilet sink lamp bathtub bag

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Partial

image Coarse

|[Figure 38]|
|---|

|[Figure 39]|
|---|

[Figure 40]

[Figure 41]

depth Semantic

Input

[Figure 42]

[Figure 43]

map Ground

|a photography of a room with a window and a door in it and a person standing in the doorway|
|---|

|a photography of a bathroom with a sink|
|---|

|a photography of a sand sculpture of a mountain with a lake in the background and a sky with clouds|
|---|

|a photography of a group of rocks in the middle of a body of water with a sky background|
|---|

prompts

Text

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

(ours) PanoDiff

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

truth MaGRITTe

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

(a)

(b) (c) (d)

- Figure 7: The results of generating a 3D scene for the test set of (a)(b) the Stuructured 3D dataset and (c)(d) the SceneDreamer dataset.

図9

the values of each evaluation metric for the generated results. In terms of FID, it can be seen that MaGRITTe does not significantly degrade performance when text conditions are included in the generation conditions. This is largely owing to the performance of the text-to-image model used as the base model to ensure the plausibility of the generated image. However, PSNR (whole) decreases in the absence of partial image and layout conditions, indicating that the contribution of these conditions to the composition of the overall structure is high. In addition, CS naturally decreases without the text condition. However, even without the text condition, CS is larger than that in the unconditional generation case, indicating that semantic reproduction is possible to some extent, even from partial images and layout information. For depth generation, the accuracy is significantly degraded because it is impossible to use depth estimation with a coarse depth in the absence of layout conditions. When generated from partial images and text, its performance was comparable to PanoDiff. Details of the experimental setup, additional samples, ablation studies, and limitations are described in appendices B and C.

### 6 Conclusions

We proposed a method for generating and controlling 3D scenes using partial images, layout information, and text prompts. We confirmed that fine-tuning a large-scale text-to-image model with small artificial datasets can generate 360-degree images from multiple conditions, and free perspective views can be generated by layout-conditioned depth

- Table 3: Evaluation results of 360-degree depth generation on the Modified Structured 3D dataset and the SceneDreamer dataset

| |Structured3D dataset<br><br>|SceneDreamer dataset|
|---|---|---|
|Method<br><br>|RMSE↓ AbsRel↓<br><br>|RMSE↓ AbsRel↓|

Coarse depth 8.858 0.0117 15.30 0.0200 360MonoDepth [44] 21.67 0.0138 15.30 0.0202 LeRes (ERP) [71] 19.03 0.0149 15.24 0.0187 LeRes (multi views) 21.90 0.0147 15.25 0.0188 Ours (end-to-end) 6.649 0.0056 15.29 0.0196 Ours (depth integration) 7.432 0.0119 15.20 0.0185

sample_wildOurs (w/o coarse depth) 9.837 0.0070 15.28 0.0196

[Figure 56]

[Figure 57]

sofa

|a photography of the living room with the warmth of wood|
|---|

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

view

window

|a photography of the listening room with faded red walls|
|---|

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

chair view

|a photography of a sandy beach at the foot of a big mountain|
|---|

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

view

|a photography of an old building on an isolated island with the ocean below|
|---|

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

view

Partial image Layout Text prompts 360-degree RGB Perspective views

360-degree depth Input Output

- Figure 8: Samples of the 3D scene generation based on user-generated conditions. Perspective views are rendered using the learned NeRF model. The first and fourth partial images are taken by the author using a camera, the second is a painting entitled "The Listening Room" by René Magritte and the third was downloaded from the web (https://www.photo-ac.com/).

図9

estimation and training NeRF. This enables 3D scene generation from multimodal conditions without creating a new large dataset. It is also indicated that the interaction of multiple spatial conditions can be performed using a common ERP latent space, and that both indoor and outdoor scenes can be handled by replacing the conversions.

Future studies will include the detection of inconsistent input conditions and suggestions for users on how to resolve these inconsistencies. Creating conditions under which the layout and partial images match perfectly is difficult, and a method that aligns with the approximate settings is desirable.

### Acknowledgements

This work was partially supported by JST Moonshot R&D Grant Number JPMJPS2011, CREST Grant Number JPMJCR2015 and Basic Research Grant (Super AI) of Institute for AI and Beyond of the University of Tokyo. We would like to thank Yusuke Kurose, Jingen Chou, Haruo Fujiwara, and Sota Oizumi for helpful discussions.

Table 4: Evaluation results for generation from subset of conditions.

|Conditions<br><br>|RGB<br><br>|Depth|
|---|---|---|
|Partial image<br><br>Layout Text|PSNR↑ (whole)<br><br>PSNR↑ (partial)<br><br>FID↓ CS↑<br><br>|RMSE ↓ AbsRel ↓|

|✓ ✓ ✓<br><br>|12.42 33.29 18.84 30.71<br><br>|5.05 0.0076|
|---|---|---|
|✓ ✓<br><br>|12.04 34.46 43.86 28.19|8.96 0.0100|
|✓ ✓|11.45 - 21.71 30.67|8.78 0.0056<br><br>|
|✓ ✓|11.48 33.64 21.83 30.93<br><br>|24.56 0.0172|
|✓|11.40 35.00 55.08 27.00|23.94 0.0158<br><br>|
|✓<br><br>|11.12 - 59.70 27.59|5.02 0.0086|
|✓|10.67 - 25.90 30.85<br><br>|24.53 0.0171|
| |10.43 - 87.69 24.40|24.00 0.0180|

### References

- [1] Ai, H., Cao, Z., pei Cao, Y., Shan, Y., Wang, L.: Hrdfuse: Monocular 360° depth estimation by collaboratively learning holistic-with-regional depth distributions. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- [2] Akimoto, N., Aoki, Y.: Image completion of 360-degree images by cgan with residual multi-scale dilated convolution. IIEEJ Transactions on Image Electronics and Visual Computing 8(1), 35–43 (2020)
- [3] Akimoto, N., Kasai, S., Hayashi, M., Aoki, Y.: 360-degree image completion by two-stage conditionalgans. In: IEEE International Conference on Image Processing (ICIP) (2019)
- [4] Akimoto, N., Matsuo, Y., Aoki, Y.: Diverse plausible 360-degree image outpainting for efficient 3dcg background creation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- [5] Bahmani, S., Park, J.J., Paschalidou, D., Yan, X., Wetzstein, G., Guibas, L., Tagliasacchi, A.: Cc3d: Layoutconditioned generation of compositional 3d scenes. arXiv:2303.12074 (2023)
- [6] Bautista, M.A., Guo, P., Abnar, S., Talbott, W., Toshev, A., Chen, Z., Dinh, L., Zhai, S., Goh, H., Ulbricht, D., Dehghan, A., Susskind, J.: Gaudi: A neural architect for immersive 3d scene generation. In: Advances in Neural Information Processing Systems (NeurIPS) (2022)
- [7] Bhat, S.F., Alhashim, I., Wonka, P.: AdaBins: Depth estimation using adaptive bins. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021)
- [8] Chai, L., Tucker, R., Li, Z., Isola, P., Snavely, N.: Persistent nature: A generative model of unbounded 3d worlds. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- [9] Chang, A., Dai, A., Funkhouser, T., Halber, M., Nießner, M., Savva, M., Song, S., Zeng, A., Zhang, Y.: Matterport3d: Learning from rgb-d data in indoor environments. In: International Conference on 3D Imaging, Modeling, Processing, Visualization and Transmission (3DIMPVT) (2017)
- [10] Chen, Z., Wang, G., Liu, Z.: Scenedreamer: Unbounded 3d scene generation from 2d image collections. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 45(12), 15562–15576 (2023)
- [11] Chen, Z., Wang, G., Liu, Z.: Text2light: Zero-shot text-driven hdr panorama generation. ACM Transactions on Graphics (TOG) 41(6), 1–16 (2022)
- [12] Cheng, W., Cao, Y.P., Shan, Y.: Sparsegnv: Generating novel views of indoor scenes with sparse input views. arXiv:2305.07024 (2023)
- [13] Cheng, Z., Zhang, Y., Tang, C.: Swin-depth: Using transformers and multi-scale fusion for monocular-based depth estimation. IEEE Sensors Journal (2021)
- [14] DeVries, T., Bautista, M.A., Srivastava, N., Taylor, G.W., Susskind, J.M.: Unconstrained scene generation with locally conditioned radiance fields. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2021)
- [15] Du, Y., Smith, C., Tewari, A., Sitzmann, V.: Learning to render novel views from wide-baseline stereo pairs. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- [16] Eder, M., Moulon, P., Guan, L.: Pano popups: Indoor 3d reconstruction with a plane-aware network. In: International Conference on 3D Vision (3DV) (2019)
- [17] Fridman, R., Abecasis, A., Kasten, Y., Dekel, T.: Scenescape: Text-driven consistent scene generation. arXiv:2302.01133 (2023)

- [18] Gardner, M.A., Sunkavalli, K., Yumer, E., Shen, X., Gambaretto, E., Gagné, C., Lalonde, J.F.: Learning to predict indoor illumination from a single image. ACM Transactions on Graphics (TOG) 9(4) (2017)
- [19] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Advances in Neural Information Processing Systems (NeurIPS) (2014)
- [20] Guerrero-Viu, J., Fernandez-Labrador, C., Demonceaux, C., Guerrero, J.J.: What’s in my room? object recognition on indoor panoramic images. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 567–573

(2020)

- [21] Hara, T., Harada, T.: Enhancement of novel view synthesis using omnidirectional image completion. arXiv:2203.09957 (2022)
- [22] Hara, T., Mukuta, Y., Harada, T.: Spherical image generation from a single image by considering scene symmetry. In: AAAI Conference on Artificial Intelligence (AAAI) (2021)
- [23] Hara, T., Mukuta, Y., Harada, T.: Spherical image generation from a few normal-field-of-view images by considering scene symmetry. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 45(5), 6339–6353 (2022)
- [24] Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: CLIPScore: A reference-free evaluation metric for image captioning. In: Moens, M.F., Huang, X., Specia, L., Yih, S.W.t. (eds.) Conference on Empirical Methods in Natural Language Processing (EMNLP) (2021)
- [25] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. In: Advances in Neural Information Processing Systems (NeurIPS) (2017)
- [26] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: Advances in Neural Information Processing Systems (NeurIPS) (2020)
- [27] Höllein, L., Cao, A., Owens, A., Johnson, J., Nießner, M.: Text2room: Extracting textured 3d meshes from 2d text-to-image models. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2023)
- [28] Hsu, C.Y., Sun, C., Chen, H.T.: Moving in a 360 world: Synthesizing panoramic parallaxes from a single panorama. arXiv:2106.10859 (2021)
- [29] Kim, S.W., Brown, B., Yin, K., Kreis, K., Schwarz, K., Li, D., Rombach, R., Torralba, A., Fidler, S.: Neuralfieldldm: Scene generation with hierarchical latent diffusion models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- [30] Kingma, D.P., Ba, J.L.: Adam: A method for stochastic optimization. arXiv:1412.6980 (2014)
- [31] Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv:1312.6114 (2013)
- [32] Kulkarni, S., Yin, P., Scherer, S.: 360fusionnerf: Panoramic neural radiance fields with joint guidance. arXiv:2209.14265 (2022)
- [33] Kuznietsov, Y., Stuckler, J., Leibe, B.: Semi-supervised deep learning for monocular depth map prediction. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017)
- [34] de La Garanderie, G.P., Atapour-Abarghouei, A., Breckon, T.: Eliminating the blind spot: Adapting 3d object detection and monocular depth estimation to 360° panoramic imagery. In: European Conference on Computer Vision (ECCV) (2018)
- [35] Laina, I., Rupprecht, C., Belagiannis, V., Tombari, F., Navab, N.: Deeper depth prediction with fully convolutional residual networks. In: International Conference on 3D Vision (3DV). IEEE (2016)
- [36] Lei, J., Tang, J., Jia, K.: Rgbd2: Generative scene synthesis via incremental view inpainting using rgbd diffusion models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- [37] Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International Conference on Machine Learning (ICML) (2022)
- [38] Lin, C.H., Lee, H.Y., Menapace, W., Chai, M., Siarohin, A., Yang, M.H., Tulyakov, S.: InfiniCity: Infinite-scale city synthesis. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2023)
- [39] Masoumian, A., Rashwan, H.A., Abdulwahab, S., Cristiano, J., Asif, M.S., Puig, D.: Gcndepth: Self-supervised monocular depth estimation based on graph convolutional network. Neurocomputing (2022)
- [40] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European Conference on Computer Vision (ECCV) (2020)
- [41] Pintore, G., Agus, M., Almansa, E., Schneider, J., Gobbetti, E.: SliceNet: deep dense depth estimation from a single indoor panorama using a slice-based representation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021)

- [42] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML) (2021)
- [43] Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., Koltun, V.: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 44(3), 1623–1637 (2022)
- [44] Rey-Area, M., Yuan, M., Richardt, C.: 360MonoDepth: High-resolution 360deg monocular depth estimation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- [45] Rezende, D.J., Mohamed, S., Wierstra, D.: Stochastic backpropagation and approximate inference in deep generative models. In: International Conference on Machine Learning (ICML) (2014)
- [46] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- [47] Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F. (eds.) Medical Image Computing and Computer-Assisted Intervention (MICCAI) (2015)
- [48] Roy, A., Todorovic, S.: Monocular depth estimation using neural regression forest. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2016)
- [49] Schult, J., Tsai, S., Höllein, L., Wu, B., Wang, J., Ma, C.Y., Li, K., Wang, X., Wimbauer, F., He, Z., Zhang, P., Leibe, B., Vajda, P., Hou, J.: Controlroom3d: Room generation using semantic proxy rooms. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2024)
- [50] Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X.: Mvdream: Multi-view diffusion for 3d generation. arXiv:2308.16512 (2023)
- [51] Sohl-Dickstein, J., Weiss, E.A., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamicss. In: International Conference on Machine Learning (ICML) (2015)
- [52] Song, S., Funkhouser, T.: Neural illumination: Lighting prediction for indoor environmentsk. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019)
- [53] Song, S., Zeng, A., Chang, A.X., Savva, M., Savarese, S., Funkhouser, T.: Im2pano3d: Extrapolating 360◦ structure and semantics beyond the field of view. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2018)
- [54] Srinivasan, P.P., Mildenhall, B., Tancik, M., Barron, J.T., Tucker, R., Snavely, N.: Lighthouse: Predicting lighting volumes for spatially-coherent illumination. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020)
- [55] Stan, G.B.M., Wofk, D., Fox, S., Redden, A., Saxton, W., Yu, J., Aflalo, E., Tseng, S.Y., Nonato, F., Muller, M., Lal, V.: Ldm3d: Latent diffusion model for 3d. arXiv:2305.10853 (2023)
- [56] Sun, C., Sun, M., Chen, H.: Hohonet: 360 indoor holistic understanding with latent horizontal features. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021)
- [57] Tang, S., Zhang, F., Chen, J., Wang, P., Furukawa, Y.: Mvdiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. arXiv (2023)
- [58] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 30 (2017)
- [59] Vidanapathirana, M., Wu, Q., Furukawa, Y., Chang, A.X., Savva, M.: Plan2scene: Converting floorplans to 3d scenes. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021)
- [60] Wang, F.E., Hu, H.N., Cheng, H.T., Lin, J.T., Yang, S.T., Shih, M.L., Chu, H.K., Sun, M.: Self-supervised learning of depth and camera motion from 360circ videos. In: Jawahar, C., Li, H., Mori, G., Schindler, K. (eds.) Asian Conference on Computer Vision (ACCV) (2019)
- [61] Wang, F.E., Yeh, Y.H., Sun, M., Chiu, W.C., Tsai, Y.H.: Bifuse: Monocular 360 depth estimation via bi-projection fusion. In: The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020)
- [62] Wang, G., Wang, P., Chen, Z., Wang, W., Loy, C.C., Liu, Z.: Perf: Panoramic neural radiance field from a single panorama. arXiv:2310.16831 (2023)
- [63] Wang, J., Chen, Z., Ling, J., Xie, R., Song, L.: 360-degree panorama generation from few unregistered nfov images. In: ACM International Conference on Multimedia (2023)

- [64] Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv:2305.16213 (2023)
- [65] Wu, T., Zheng, C., Cham, T.J.: Ipo-ldm: Depth-aided 360-degree indoor rgb panorama outpainting via latent diffusion model. arXiv:2307.03177 (2023)
- [66] Xiao, J., Ehinger, K.A., Oliva, A., Torralba, A.: Recognizing scene viewpoint using panoramic place representation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2012)
- [67] Xu, D., Ricci, E., Ouyang, W., Wang, X., Sebe, N.: Multi-scale continuous crfs as sequential deep networks for monocular depth estimation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

- (2017)

[68] Xu, D., Wang, W., Tang, H., Liu, H., Sebe, N., Ricci, E.: Structured attention guided convolutional neural fields for monocular depth estimation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

- (2018)

- [69] Yang, G., Tang, H., Ding, M., Sebe, N., Ricci, E.: Transformer-based attention networks for continuous pixel-wise prediction. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2021)
- [70] Yang, K., Ma, E., Peng, J., Guo, Q., Lin, D., Yu, K.: Bevcontrol: Accurately controlling street-view elements with multi-perspective consistency via bev sketch layout. arXiv:2308.01661 (2023)
- [71] Yin, W., Zhang, J., Wang, O., Niklaus, S., Mai, L., Chen, S., Shen, C.: Learning to recover 3d scene shape from a single image. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021)
- [72] Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: IEEE International Conference on Computer Vision (ICCV) (2023)
- [73] Zheng, J., Zhang, J., Li, J., Tang, R., Gao, S., Zhou, Z.: Structured3d: A large photo-realistic dataset for structured 3d modeling. In: European Conference on Computer Vision (ECCV) (2020)
- [74] Zioulis, N., Karakottas, A., Zarpalas, D., Daras, P.: Omnidepth: Dense depth estimation for indoors spherical panoramas. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) European Conference on Computer Vision (ECCV) (2018)
- [75] Zioulis, N., Karakottas, A., Zarpalas, D., Alvarez, F., Daras, P.: Spherical view synthesis for self-supervised 360o depth estimation. In: International Conference on 3D Vision (3DV) (2019)

|Sigmoid|128→1Conv2D()| |
|---|---|---|
| | | |

|GroupeNormalize|
|---|

|4→128Conv2D()| |
|---|---|
| | |

256→128ResnetBlock()

256→128ResnetBlock()

256→128ResnetBlock()

128→128ResnetBlock()

128→128ResnetBlock()

128→1Conv2D()

4→128Conv2D()

GroupeNormalize

Downsampling

Upsampling

Sigmoid

out

in

||384→128ResnetBlock()|256→128ResnetBlock()|256→128ResnetBlock()|Upsampling|
|---|---|---|---|
<br><br>384→128ResnetBlock()<br><br>256→128ResnetBlock()<br><br>256→128ResnetBlock()<br><br>Upsampling|
|---|

384→128ResnetBlock()

256→128ResnetBlock()

256→128ResnetBlock()

128→128ResnetBlock()

128→128ResnetBlock()

Downsampling

Upsampling

||512→256ResnetBlock()|512→256ResnetBlock()|384→256ResnetBlock()|Upsampling|
|---|---|---|---|
<br><br>512→256ResnetBlock()<br><br>512→256ResnetBlock()<br><br>384→256ResnetBlock()<br><br>Upsampling|
|---|

128→256ResnetBlock()

256→256ResnetBlock()

512→256ResnetBlock()

512→256ResnetBlock()

384→256ResnetBlock()

Downsampling

Upsampling

||768→256ResnetBlock()|512→256ResnetBlock()|512→256ResnetBlock()|Upsampling|
|---|---|---|---|
<br><br>768→256ResnetBlock()<br><br>512→256ResnetBlock()<br><br>512→256ResnetBlock()<br><br>Upsampling|
|---|

256→256ResnetBlock()

256→256ResnetBlock()

768→256ResnetBlock()

512→256ResnetBlock()

512→256ResnetBlock()

Downsampling

Upsampling

||1024→512ResnetBlock()|1024→512ResnetBlock()|768→512ResnetBlock()|AttentionBlock|
|---|---|---|---|
<br><br>1024→512ResnetBlock()<br><br>1024→512ResnetBlock()<br><br>768→512ResnetBlock()<br><br>AttentionBlock|
|---|

1024→512ResnetBlock()

1024→512ResnetBlock()

256→512ResnetBlock()

512→512ResnetBlock()

768→512ResnetBlock()

AttentionBlock

AttentionBlock

||512→512ResnetBlock()|AttentionBlock|512→512ResnetBlock()|
|---|---|---|
<br><br>512→512ResnetBlock()<br><br>AttentionBlock<br><br>512→512ResnetBlock()|
|---|

512→512ResnetBlock()

512→512ResnetBlock()

AttentionBlock

- Figure 9: The structure of the layout-conditioned depth estimation network. Conv2D (N → M) is a two-dimensional convolutional layer with N input channels, M output channels, and a kernel size of 3 × 3. The Resnet Block shown in fig. 10 is combined into a U-Net structure. Downsampling and upsampling are performed using a factor of 2. In the Attention Block, self-attention [58] in the form of a query, key, and value is applied in pixels.

図12

### A Details of Layout-Conditioned Depth Estimation

In this section, we describe the details of the layout-conditioned depth estimation, which generates a fine depth from the coarse depth and generated RGB.

#### A.1 End-toEnd Network Configuration

The structure of the network that generates a fine depth from a coarse depth and the generated RGB end-to-end is shown in figs. 9 and 10. The network consists of a combination of U-Net [47] and self-attention [58], with four channels of RGB-D as the input and one channel of depth as the output. The network was trained to minimize the L1 loss between the depth output from the network and the depth of the ground truth. The model was trained from scratch using the Adam optimizer with a learning rate of 4.5 × 10−6 and a batch size of six.

#### A.2 Equation Derivation for Depth Integration

Let dˆn ∈ RH

dWd(n = 1,2,··· ,N) be the monocular depth estimate for n-th tangent image in ERP format, where Hd and Wd are the height and width of the depth map, respectively. Since the estimated depth dˆn has unknown scale and offset, it is transformed using the affine transformation coefficient sn ∈ R2 as d˜nsn, where d˜n = (dˆn 1) ∈ RH

dWd is the coarse depth, Φn ∈ RH

dWd×2. We consider the following evaluation function Ldepth, where d0 ∈ RH

dWd is the integrated depth.

dWd×HdWd(n = 0,1,··· ,N) is the weight matrix, and x ∈ RH

N

||x − d˜nsn||2Φn, (4)

Ldepth = ||x − d0||2Φ0 +

n=1

where the quadratic form ||v||2Q = v⊤Qv. We find the affine transformation coefficient sn(n = 1,2,··· ,N) and fine depth x from the extreme-value conditions to minimize Ldepth. The partial differentiation of eq. (4) with x yields:

N

∂Ldepth ∂x

Φn(x − d˜nsn)

= 2Φ0(x − d0) + 2

n=1

N

N

Φnd˜nsn , (5)

Φnx − 2 Φ0d0 +

= 2

n=0

n=1

|MaGRITTe: Manipulative|
|---|

and Generative 3D Realization from Image, Topview and Text

In

|Groupe Normalize| |
|---|---|
| | |

|Sigmoid| |
|---|---|
| | |

|Conv2D (𝑁 → 𝑀)| |
|---|---|
| | |

|Conv2D (𝑁 → 𝑀)|
|---|

|Groupe Normalize| |
|---|---|
| | |

|Sigmoid| |
|---|---|
| | |

|Conv2D (𝑀 → 𝑀)| |
|---|---|
| | |

Out

- Figure 10: The structure of a Resnet Block (N → M). N is the number of input channels, and M is the number of output channels. In the groupe normalize, the number of split channels is fixed at 32. Conv2D refers to a twodimensional convolutional layer, and the numbers in parentheses indicate the conversion of the number of channels.

図13

and x satisfying the extreme-value conditions are as follows:

−1

N

N

Φnd˜nsn . (6)

x =

Φn

Φ0d0 +

n=0

n=1

Next, the partial differentiation of eq. (4) with sk yields:

∂Ldepth ∂sk

= −2d˜⊤k Φk(x − d˜ksk), (7)

and sk satisfying the extreme-value conditions are as follows:

d˜⊤k Φkd˜ksk = d˜⊤k Φkx. (8) By substituting eq. (6) into eq. (10), we obtain

−1

N

N

d˜⊤k Φkd˜ksk = d˜⊤k Φk

Φnd˜nsn . (9)

Φn

Φ0d0 +

n=0

n=1

Transposing sn on the left-hand side yields

−1 N

−1

N

N

d˜⊤k Φkd˜ksk − d˜⊤k Φk

Φnd˜nsn = d˜⊤k Φk

Φ0d0. (10)

Φn

Φn

n=0

n=1

n=0

Considering the coefficient of sk as Dk ∈ R2×2, we obtain

−1

N

Dk = d˜⊤k Φkd˜k − d˜⊤k Φk

Φkd˜k

Φn

n=0

 

 

−1

N

d˜k

= d˜⊤k Φk

I −

Φn

Φk





n=0

 

−1  

 I + Φ−k 1

 

N\k

= d˜⊤k Φk

d˜k

I −

Φn



n=0

 

 

 

 

−1

−1

N\k

= d˜⊤k Φk

d˜k

I +

Φn

Φk





n=0

 

−1  

 

 

−1

N\k

= d˜⊤k

d˜k, (11)

Φ−k 1 +

Φn



n=0

where n N=0\k Φn := Nn=0 Φn − Φk. In addition, considering the coefficient of sl(l ̸= k) as Uk,l ∈ R2×2, we obtain

−1

N

Uk,l = −d˜⊤k Φk

Φld˜l. (12)

Φn

n=0

The constant bk ∈ R2 is expressed as follows:

−1

N

bk = d˜⊤k Φk

Φ0d0. (13) Therefore, when the conditions in eq. (10) are coupled for k = 1,2,··· ,N, we obtain

Φn

n=0

  

  

  

   =

  

  . (14)

D1 U1,2 ··· U1,N U2,1 D2 ··· U2,N

- s1
- s2

- b1
- b2

... .

. .

. sN

. bN

UN,1 UN,2 ··· DN

We can then solve for sn(n = 1,2,··· ,N) as follows.

  

   =

  

  

−1 

  . (15)

D1 U1,2 ··· U1,N U2,1 D2 ··· U2,N

- b1
- b2

- s1
- s2

 

... .

. sN

. .

. bN

UN,1 UN,2 ··· DN

From the above results, we can determine x that minimizes equation eq. (4) by first calculating sn(n = 1,2,··· ,N) using eq. (15) and then substituting the value into eq. (6).

#### A.3 Weight Setting for Depth Integration

In this study, we set the weight matrix Φn(n = 0,1,··· ,N) to a diagonal matrix. By making it a diagonal matrix, the large matrix calculation in eqs. (11) to (13) can be avoided and can be attributed to element-by-element calculations. The diagonal components represent the reflected intensity at each location on each depth map. Since the weight matrices Φn(n = 1,2,··· ,N) are for depth maps that express the estimated depth for N tangent images in ERP format, the weights are increased for regions where tangent images are present, as shown in fig. 11 To smooth the boundary, we first set the following weights wij for pixel position (i,j) in the tangent image of height Htan and width Wtan.

wij = 1 −

2i Htan − 1

2

1 −

2j Wtan − 1

2

. (16)

| | | |
|---|---|---|
| | | |

[Figure 80]

- 𝑛 = 1
- 𝑛 = 2

Projection

[Figure 81]

[Figure 82]

⋮

Weight for tangent image

[Figure 83]

𝑛 = 𝑁

Weight for ERP image

図18

- Figure 11: Weights for estimated depth maps. The weights are set such that the center of the tangent image is 1, the edges of the image are 0, and the weights are converted to ERP format for each depth map (n = 1,2,··· ,N).

This weight has a maximum value of 1 at the center of the tangent image and a minimum value of 0 at the edges of the image. The weights for the tangent image are converted to ERP format and set to the diagonal components of the weight matrix Φn(n = 1,2,··· ,N). The weights of the outer regions of each tangential image are set to zero. Tangent images are created with a horizontal field of view of 90 degrees and resolution of 512 × 512 pixels, and 16 images were created with the following latitude θn and longitude ϕn shooting directions.

π

4 (1 ≤ n ≤ 4) −π4 (5 ≤ n ≤ 8) 0 (9 ≤ n ≤ 16)

(17)

θn =

πn

2 (1 ≤ n ≤ 8) πn

(18)

ϕn =

4 (9 ≤ n ≤ 16)

On the other hand, the weights for the coarse depth Φ0 are set as follows. When using floor plans for the layout format, a low-weight ηL is set for areas in the partial image or layout condition where an object is specified, and a high-weight ηH(≥ ηL) for other areas. In this study, we set ηL = 0.0, ηH = 2.0. When using the terrain map for the layout format, set the diagonal component of the weight matrix Φ0(i,j) according to the value of the coarse depth at each location (i,j) in the ERP as follows:

α d0(i,j)2 + ϵ

, (19)

Φ0(i,j) =

where α and ϵ are hyperparameters. In this study, the coarse depth is normalized to the interval [0, 1], and we set α = 1.0 × 10−3 and ϵ = 1.0 × 10−8. We set Φ0(i,j) = 0 in the region where the coarse depth is infinite. The weights are inversely proportional to the square of the coarse depth to ensure that the squared error in eq. (4) assumes values of the same scale with respect to the coarse depth. This prevents the error from being overestimated when an object is generated in the foreground of a large-depth region, such as a tree in the foreground of the sky.

### B Additional Results

#### B.1 Condition Dropout

Fine-tuning of the base model degrades image-to-text performance. To mitigate this phenomenon, we additionally use the auxiliary dataset (see Section 4) with text annotations only for fine-tuning. If one model is trained for different combinations of conditions, the learning may not be generalized to other combinations of conditions. We introduce condition dropout (CD), in which training is performed by randomly changing the combination of conditions. Each condition is dropped with a probability of 50%, with the ERP image conditions being replaced by pixel values of 0 and text replaced by an empty string.

table 5 shows the results of comparing the presence or absence of CD in the proposed method. FID tended to be slightly better when CD was present, whereas PSNR (whole), PSNR (partial), and CS were superior or inferior depending on

Table 5: Evaluation results of 360-degree RGB generation on the Modified Structured 3D dataset and the SceneDreamer dataset.

| |Structured3D dataset|SceneDreamer dataset|
|---|---|---|
|method<br><br>|PSNR↑ (whole)<br><br>PSNR↑ (partial)<br><br>FID↓ CS↑<br><br>|PSNR↑ (whole)<br><br>PSNR↑ (partial)<br><br>FID↓ CS↑|

w/o CD 12.56 35.39 18.87 30.72 12.46 34.68 29.54 29.71 w/ CD 12.42 33.29 18.84 30.71 13.29 34.81 29.05 29.93

Table 6: CS evaluation results for base model forgetting Trained on base dataset

Trained on auxiliary dataset

Condition dropout

Indoor Outdoor

|✓<br><br>|29.48<br><br>|24.75|
|---|---|---|
|✓ ✓|29.34<br><br>|26.24|
|✓ ✓ ✓|30.23|29.26|

the two datasets. The better performance of CD on the SceneDreamer dataset can be attributed to the larger number of samples in the auxiliary dataset.

Next, we

of the experiment in a setting in which the conditions were crossed between CS for generated results with the text prompt of the auxiliary dataset for the d that CS can be improved by using the auxiliary dataset and CD. fig. 12 shows the difference with and without CD. These results show that the use of CD better reflects text prompts, and the generalization of text prompts in combination with depth is possible.

|present the results of the evaluation datasets. table 6 shows the results of the depth of the base dataset. This indicates<br><br>[Figure 84]|
|---|

|a photography of a living room with a couch and a piano in it and a piano in the background|
|---|

[Figure 85]

[Figure 86]

[Figure 87]

Coarse depth Text prompts w/o Condition dropout w/ Condition dropout

- Figure 12: The difference with and without CD. In this example, "piano" in the text prompt is reflected only for the method with CD.

#### B.2 Comparison with Text2Room

Text2Room [27] is a method for generating 3D scenes as meshes by repeatedly generating images in multiple viewpoints from the input text. This method can also be used to control the layout of the generated 3D scene by changing the input text according to the viewpoint. However, layout guided generation in Text2Room is different from our setting, because it changes the text prompts for the direction of observation and cannot take geometric shapes as conditions. fig. 13 shows an example of a scene generated by Text2Room under the same conditions as in fig. 1. Text2Room is less accurate in the placement of objects and is unable to generate room shapes to suit the conditions. Conditioning the layout with semantic map and coarse depth is the advantage of our method.

図10

#### B.3 360-Degree RGB Generation

figs. 14 and 15 show additional samples of 360-degree RGB image generation for the Structured 3D dataset and SceneDreamer dataset, respectively.

#### B.4 Results in the Wild

We evaluated the results of the 3D scene generation based on user-generated conditions outside the dataset used for fine-tuning. In this experiment, the end-to-end method was used to estimate the depth in indoor scenes, whereas the depth integration method was applied to outdoor scenes because the SceneDreamer dataset is limited to natural scenery, such as mountainous areas and seashores, using monocular depth estimation models trained on an external dataset. Because CD is effective for fine-tuning with additional text annotations, we used a simpler method without CD in the

|[Figure 88]|
|---|

MaGRITTe: Manipulative and Generative 3D Realization from Image, Topview and Text

[Figure 89]

[Figure 90]

[Figure 91]

window

table sink

- (a)
- (b)

Semantic map (condition) Text2Room MaGRITTe (ours)

[Figure 92]

Condition

Text2Room

MaGRITTe (ours)

- Figure 13: Comparison with Text2Room. (a) ERP images of the generated 3D scenes, (b) Room shapes in the top view.

in-the-wild experiments described in this section. The terrain map T ∈ RH

ter×Wterwas created as a mixed Gaussian distribution in the following equation:

図1

K

- 1

- 2

(p − µk)⊤Σ−k 1(p − µk) , (20)

πk exp −

Tp =

k=1

where p ∈ {1,2,··· ,Hter} × {1,2,··· ,Wter} is the location on the 2-D map, K is the number of mixtures, and πk ∈ R, µk ∈ R2, and Σk ∈ R2×2 are the parameters of the weights, mean, and covariance matrix of the element distribution, respectively.

Additional examples of 3D scenes generated using the proposed method conditioned on text, partial images, and layouts are presented in figs. 16 to 20. In these figures, the aspect ratios of the ERP images were converted to 2:1 for display purposes. These conditions were created freely by the authors. It can be seen that the generated scene contains the given partial image and conforms to the instructions of the text prompt according to the given layout. In addition to the coarse depth created by the room shape or terrain alone, the geometry of objects such as chairs, tables, trees, and buildings can be seen. fig. 16 shows how various scenes can be generated in a controlled manner by changing the combination of layout and text for the same partial image. figs. 17 to 20 shows that our method can generate a variety of 3D scenes from photos on the web, photos taken in the real world, and fanciful paintings, taking into account the layout and text requirements we give. These results show that the proposed method can generate 360-degree RGB-D images with appearance, geometry, and overall context controlled according to the input information, even outside the dataset used for fine-tuning.

### C Discussion

- C.1 Advantages of Using 360-Degree Images

The proposed method uses a trained text-to-image model to generate a 2D image, from which the depth is generated. The proposed method is unique because it uses a 360-degree image as the 2D image for generation. Using 360-degree images is advantageous over perspective projection images in terms of scene consistency and reduced computational costs. fig. 21 shows examples of the generated scene from a partial image by the incremental multi-view inpanting and MVDiffusion [57]. Incremental multi-view inpainting is a method of repeating SD inpainting by projecting an input image from a different viewpoint. In the example shown in this figure, the road disappears, indicating that the scene is inconsistent. This is due to the fact that inpainting is performed on each perspective projection image; therefore, the overall consistency cannot be guaranteed. In addition, inpainting must be applied repeatedly, which is computationally expensive and difficult to parallelize. MVDiffusion, on the other hand, takes cross-attention among multiple views and generates multiple views that are simultaneously consistent using SD. This method is computationally expensive because it requires running SD for each view and paying cross-attention to the combinations of multiple views. The order of computational complexity is O(N2), where N is the number of viewpoints. Because the proposed method generates a single 360-degree image, it is easy to achieve scene consistency at a low computational cost. However, the resolution of the generated image using ERP is lower than that of multiview images, and a higher resolution is a future challenge.

- C.2 Limitation Although the performance of the proposed method was promising, it had several limitations.

|sample_360rgb_str3d|
|---|

[Figure 93]

cabinet bed chair sofa table door window bookshelf picture blinds desk shelves curtain dresser mirror floormat clothes books refrigerator TV paper towel shower ca. box white board person nightstand toilet sink lamp bathtub bag

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Partial

image Coarse

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

depth Semantic

Input

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

map Ground

|a photography of a room with a wooden floor and a white wall and a wooden floor and a wooden door|
|---|

|a photography of a room with a window and a picture of a city in the window and a mirror|
|---|

|a photography of a living room with a couch and a television mounted on the wall and a wooden floor|
|---|

|a photography of a room with a bed|
|---|

prompts

Text

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

w/oCD PanoDiff

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

w/CD Ours

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

truth Ours

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

図13

(a) (b) (c) (d)

Figure 14: The results of generating a 3D scene for the test set of the Stuructured 3D dataset.

- fig. 22 shows examples of problems in RGB generation. First, if the objects specified in the layout are in overlapping positions from a viewpoint, they cannot be separated and drawn in the correct number and position. This is because the 2D layout information is converted to ERP for input, which requires additional ingenuity, such as generating a 3D scene jointly from multiple viewpoints. Second, when using conditions outside the dataset, the specified conditions may not be reflected, depending on the interaction between each condition. For example, there is the phenomenon that certain text prompts do not produce certain objects. Third, it is not possible to specify the regions where objects do not exist. Except for the regions where objects are specified, object generation is controlled by other conditions such as partial image, depth, and text.
- fig. 23 shows examples of problems in 6 DoF 3D scene generation. It is difficult to synthesize plausible views when generating 3D scenes from 360-degree RGB-D images with large missing regions that exceed image completion capabilities. We hope that these limitations will be addressed in future studies.

|sample_360rgb_scdr|
|---|

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Partial

image Coarse

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

depth Ground

Input

|a photography of a group of rocks in the water with a sky background and a person in a boat|
|---|

|a photography of a lake surrounded by trees and a sky with clouds in the background and a hole in the ground|
|---|

|a photography of a picture of a view of a<br><br>mountain range with a lake in the foreground|
|---|

|a photography of a group of birds standing on top of a sandy beach next to a body of water|
|---|

prompts

Text

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

w/oCD PanoDiff

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

w/CD Ours

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

truth Ours

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

(a) (b) (c) (d)

Figure 15: The results of generating a 3D scene for the test set of the SceneDreamer dataset.

図14

|[Figure 146]|
|---|

[Figure 147]

|Text Layout|a photography of medieval European kitchen|a photography of modern kitchen|
|---|---|---|
|[Figure 148]<br><br>table<br><br>view|[Figure 149]<br><br>[Figure 150]|[Figure 151]<br><br>[Figure 152]|
|[Figure 153]<br><br>table<br><br>view<br><br>window<br><br>sink|[Figure 154]<br><br>[Figure 155]|[Figure 156]<br><br>[Figure 157]|

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Partial image

図17

- Figure 16: From a given partial image, layout, and text prompt, our method generates the 360-degree RGB space and depth. We used a painting titled "The Milkmaid" by Johannes Vermeer as a partial image. Various 3D scenes can be generated for the same partial image using different layouts and text prompts.

- (a)
- (b)
- (c)
- (d)

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

table

view

[Figure 167]

[Figure 168]

[Figure 169]

|a photography of medieval European kitchen|
|---|

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

table

view window sink

[Figure 175]

[Figure 176]

[Figure 177]

|a photography of modern kitchen|
|---|

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

window

[Figure 182]

table

view

TV

[Figure 183]

[Figure 184]

[Figure 185]

door

|a photography of a calm living room|
|---|

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

sofa

[Figure 191]

[Figure 192]

[Figure 193]

window

|a photography of a hotel lobby|
|---|

Input conditions 360-degree RGB-D Perspective views

- Figure 17: The various generated indoor 3D scenes represented by 360-degree RGB-D images and free perspective images rendered using NeRF owing to conditions outside the used dataset. (a) (b) We used a painting titled "The Milkmaid" by Johannes Vermeer as a partial image. (c) (d) A photo of sofas downloaded from the web (https://www.photo-ac.com/) was provided as a partial image.

- (a)
- (b)
- (c)
- (d)

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

sofa

[Figure 198]

view

[Figure 199]

[Figure 200]

[Figure 201]

window

|a photography of the living room with the warmth of wood|
|---|

[Figure 202]

[Figure 203]

[Figure 204]

table

[Figure 205]

[Figure 206]

view

table

[Figure 207]

[Figure 208]

[Figure 209]

|a photography of the oldfashioned cafeteria|
|---|

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

chair view

[Figure 215]

[Figure 216]

[Figure 217]

|a photography of the listening room with faded red walls|
|---|

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

view

[Figure 223]

[Figure 224]

[Figure 225]

|a photography of fruit storages|
|---|

Input conditions 360-degree RGB-D Perspective views

- Figure 18: The various generated indoor 3D scenes represented by 360-degree RGB-D images and free perspective images rendered using NeRF owing to conditions outside the used dataset. (a) (b) An image captured by the author using a camera is shown as a partial image. (e) (f) We presented a painting titled "The Listening Room" by René Magritte as a partial image.

- (a)
- (b)
- (c)
- (d)

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

view

[Figure 231]

[Figure 232]

[Figure 233]

|a photography of a sandy beach at the foot of a big mountain|
|---|

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

view

[Figure 239]

[Figure 240]

[Figure 241]

|a photography of long beaches in a big city|
|---|

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

view

[Figure 247]

[Figure 248]

[Figure 249]

|a photography of the university campus|
|---|

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

view

[Figure 255]

[Figure 256]

[Figure 257]

|a photography of an old building on an isolated island with the ocean below|
|---|

Input conditions 360-degree RGB-D Perspective views

- Figure 19: The various generated outdoor 3D scenes represented by 360-degree RGB-D images and free perspective images rendered using NeRF owing to conditions outside the used dataset. (a) (b) A photo of a sandy beach downloaded from the web (https://www.photo-ac.com/) was given as a partial image. (c) (d) An image captured by the author using a camera is shown as a partial image.

- (a)
- (b)

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

view

[Figure 263]

[Figure 264]

[Figure 265]

|a photography of a big lake|
|---|

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

view

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

|a photography of a town at the foot of the mountains under the blue sky|
|---|

- (c)
- (d)

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

view

[Figure 285]

[Figure 286]

[Figure 287]

|a photography of desert sunset|
|---|

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

view

[Figure 294]

[Figure 295]

[Figure 296]

|a photography of golf course bunkers at sunset|
|---|

Input conditions 360-degree RGB-D Perspective views

- Figure 20: The various generated outdoor 3D scenes represented by 360-degree RGB-D images and free perspective images rendered using NeRF owing to conditions outside the used dataset. (a) (b) An image captured by the author using a camera is shown as a partial image. (c) and (d) We provided a painting titled "Day after Day" by Jean-Michel Folon as a partial image.

[Figure 297]

advantage_360

[Figure 298]

[Figure 299]

[Figure 300]

Input Output

- (a) Incremental multi-view inpainting
- (b) MVDiffusion (Cross attention between multi-views)

|[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]|
|---|

[Figure 309]

Input Output

図16

- Figure 21: Examples of the scene generation from a partial image through the generation of perspective projection images. The generated scenes were displayed in ERP format. (a) In incremental multiview inpainting of the perspective image downloaded from the web (https://unsplash.com/@overture_creations/), the road disappears on the other side, indicating that the scene is not consistent. (b) MVDiffusion maintains consistency between multiple views; however, the computational cost is high because cross attention is required for each combination of multiple views.

|[Figure 310]<br><br>limitation_360rgbd|
|---|

図17

[Figure 311]

(a) Object Integration

|[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]<br><br>Partial image<br><br>[Figure 317]<br><br>table<br><br>TV<br><br>table<br><br>view Layout<br><br>|a photography of medieval European kitchen|
|---|
<br><br>Text|
|---|

Input

(b) Change in object class

(c) Appearance of objects

- Figure 22: Examples of limitations of 360-degree RGB-D generation from multimodal conditions. (a) When two tables specified in the layout condition overlap in the ERP, they are merged and generated as a single table. (b) Although the layout conditions dictate the placement of a television, it is generated and converted to a window because it does not conform to the context of “a medieval European kitchen,” which is presented in the text prompt. (c) Where nothing is specified in the layout conditions, objects may be generated automatically according to text prompts. It is impossible to specify areas where no objects exist.

|[Figure 318]|
|---|

[Figure 319]

[Figure 320]

Synthesized novel view from NeRF model

Generated 360-degree RGB image

- Figure 23: Examples of limitations of synthesized novel views from the NeRF model trained on the generated 360degree image. It is difficult to synthesize plausible views when generating 3D scenes from 360-degree RGB-D images with large missing regions that exceed image completion capabilities. In this example, the image quality is significantly reduced in the occluded region at the back of the building.

図18

