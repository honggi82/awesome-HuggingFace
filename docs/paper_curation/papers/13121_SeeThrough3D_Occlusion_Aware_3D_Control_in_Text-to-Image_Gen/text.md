## SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation

Vaibhav Agrawal1 Rishubh Parihar2 Pradhaan S Bhat2

Ravi Kiran Sarvadevabhatla1† R. Venkatesh Babu2† 1IIIT Hyderabad 2IISc Bengaluru

[Figure 1]

(a) Occlusion-aware layout following.

(b) Adherence to complex layouts with many objects.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

# arXiv:2602.23359v1[cs.CV]26Feb2026

scene_20251114_0 14048.pkl

image

[Figure 7]

[Figure 8]

scene_20251106_0 15519.pkl

image

‘A photo of bicycle and sedan and dog and puppy and chair in the backyard of a house, beautiful morning, with gleaming sunshine.

[Figure 9]

[Figure 10]

‘A photo of table and bicycle and dog in the verandah of a modern house.’

scene_20251110 _125158.pkl

(c) Camera control.

[Figure 11]

[Figure 12]

scene_20251110_1 25359.pkl

‘A photo of monitor displaying letters ‘CVPR’ in colorful and monitor displaying letters ‘CVPR’ in colorful and monitor and laptop and mouse and ofﬁce desk and plant and dog and man and ofﬁce chair and backpack in a design studio with colorful walls and colorful carpet, soft lighting, aesthetic modern environment.’

‘A photo of glass bowl and apples and shiny metal toaster and coffee mug and plate and orange and rubik’s cube on a dining table in a kitchen.’

Figure 1. We propose SeeThrough3D, a method for occlusion aware 3D scene control in text-to-image generation. Our method enables (a) occlusion-aware 3D object placement in generated images, and (b) adheres well to complex layouts featuring many objects. Additionally, our method allows for (c) control over the camera viewpoint in the generated image.

###### Abstract

We identify occlusion reasoning as a fundamental yet overlooked aspect for 3D layout–conditioned generation. It is essential for synthesizing partially occluded objects with depth-consistent geometry and scale. While existing methods can generate realistic scenes that follow in-

†Equal advising

put layouts, they often fail to model precise inter-object occlusions. We propose SeeThrough3D, a model for 3D layout conditioned generation that explicitly models occlusions. We introduce an occlusion-aware 3D scene representation (OSCR), where objects are depicted as translucent 3D boxes placed within a virtual environment and rendered from desired camera viewpoint. The transparency encodes hidden object regions, enabling the model to reason about

occlusions, while the rendered viewpoint provides explicit camera control during generation. We condition a pretrained flow based text-to-image image generation model by introducing a set of visual tokens derived from our rendered 3D representation. Furthermore, we apply masked selfattention to accurately bind each object bounding box to its corresponding textual description, enabling accurate generation of multiple objects without object attribute mixing. To train the model, we construct a synthetic dataset with diverse multi-object scenes with strong inter-object occlusions. SeeThrough3D generalizes effectively to unseen object categories and enables precise 3D layout control with realistic occlusions and consistent camera control. Project page: https://seethrough3d.github.io

###### 1. Introduction

Recent work has introduced various forms of controllability in text-to-image generation, but most methods remain limited to 2D spatial controls, such as bounding boxes or segmentation maps [23, 44, 48, 61, 62, 77, 79]. While effective for coarse control over the scene content, they offer limited control over inherently 3D scene properties, including object arrangement and camera viewpoint. Yet many practical content-creation domains such as design, gaming, and architectural visualization require precise 3D layout control, where object size, orientation, and placement must be explicitly specified. Critically, a truly 3D-aware generative model must also reason about occlusions, generate partially hidden objects with depth-consistent scale and perspective; a fundamental capability that 2D controls cannot provide.

Despite being fundamental to accurate 3D-aware generation, occlusion has been largely overlooked in recent 3D layout based methods. Existing approaches condition the generative model on depth maps derived from 3D bounding-box layouts [4, 65] or on explicit 3D attributes such as object or camera poses [7, 9, 43, 49, 55]. These methods succeed in generating simple scenes with few objects and minimal occlusion, but fail to model significant inter-object occlusions in multi-object layouts (Fig. 3(a)). A related direction represents scenes as a stack of 2D object layers [37, 76] to approximate occlusion, but this collapses the inherently 3D structure of the scene into flat planes (Fig. 3(c)), leading to generating object occlusion that violate true 3D geometry and perspective.

In this paper, we propose SeeThrough3D - an image generation model that takes 3D layout and text prompt as input and generates scenes with 3D consistent occlusions (Fig. 1). We introduce an efficient and expressive 3D scene representation, termed Occlusion-Aware 3D Scene Representation (OSCR), which jointly encodes object arrangements and camera viewpoint (Fig. 2). In OSCR, each object is modeled as a translucent 3D bounding box, where transparency reveals occluded regions, enabling explicit reasoning about

inter-object occlusions. Faces of each box are further colorcoded according to a predefined mapping to capture 3D object orientation. The final OSCR representation is obtained by rendering this layout from a specified camera viewpoint.

We build on FLUX [35] image generator, conditioning it on our OSCR scene representation. Following the success of recent works [35, 61] on controlling the diffusion transformer (DiT) [21, 54] using condition image tokens, we condition the model with tokens derived from our rendered scene representation. However, spatial conditioning alone fails to associate textual object descriptions with their corresponding box regions. To address this, we apply attention masking to bind each object to its corresponding box, ensuring accurate bounding box adherence for individual objects. Further, we extend this framework to allow 3D control of personalized objects, by conditioning on an image of the object, and binding its appearance to specific box in the OSCR representation.

To train SeeThrough3D, we create a synthetic dataset of scenes by placing diverse 3D assets in a virtual environment [12] and rendering scenes from multiple camera views. Object placement and camera parameters are controlled to induce strong inter-object occlusions in the rendered images. Despite being trained on synthetic data, SeeThrough3D generalizes well to unseen objects, backgrounds and complex scene layouts (see Fig. 1), evaluated qualitatively and through metrics, as well as a user study.

- 2. Related work
- 3D control in text-to-image generation: Previous works on 3D control in image generation trains specialized generative models conditioned on various 3D representations [2, 28, 29, 45, 46, 64, 71]. Interestingly, recent works have shown that there is inherent 3D understanding in large textto-image diffusion models [17, 18, 74]. Several works leverage this insight for enabling precise 3D aware control in generated images [3, 7, 7, 10, 16, 20, 24, 34, 38, 57]. One line of works enable 3D aware editing [47, 57, 66] using scene depth as additional input, but they are limited to manipulation of a single object at a time. Further, a recent work [51] decomposes a scene into depth-based layers, enabling depth-aware editing and scene composition. Others train implicit 3D representations such as radiance fields [33, 53, 73] or 3D Gaussian splats [8, 31, 40, 68, 78] in diffusion feature space to enable 3D aware image editing. 3D layout conditioned generation: Apart from editing, controlling the 3D layout of a scene during generation is an active research area. A recent work for layout-conditioned generation, LooseControl [4] conditions a text-to-image model using depth maps of 3D bounding boxes; however it fails to generate complex scenes with diverse objects. A follow-up work, Build-A-Scene [19] generates the scene using multiple generation-inversion cycles, each iteration

[Figure 13]

[Figure 14]

[Figure 15]

translucency captures occluded objects.

[Figure 16]

color-coding faces to indicate 3D orientation (orange for front face, blue for left, others green).

CAMERA

(c) Generated image

(a) User interface (b) OSCR

Figure 2. OSCR: We propose Occlusion-Aware Scene Representation (OSCR) for 3D layout control in text-to-image generation. OSCR describes objects as translucent 3D boxes, which exposes occluded regions, enabling the generative model to reason about occlusions. Further, each box face is color-coded with a mapping to encode its 3D orientation. (a) A user specifies the object bounding boxes (b0 and b1) and sets desired viewpoint C in an interactive graphic environment. (b) These boxes are rendered to obtain our OSCR representation, (c) which is used to condition the generation for occlusion aware 3D control.

adding a new object. However, this leads to inversion artifacts and incoherence in generated images. Another set of works provide partial control over individual 3D properties, such as object orientation [9, 43, 49], but they are limited in their extent to precisely control object placement or camera viewpoint. Another promising direction for 3D layout control is to represent the object bounding box as a set and condition the generative model using a learnable adapter [41, 50, 70]. However, they are limited to a single data domain, e.g. road scenes or indoor scenes, and are less effective than spatial conditioning approaches [55].

Occlusion awareness: Inter-object occlusions present a significant challenge in perception [22, 27, 32, 36, 42, 60, 75] and generation [37, 39, 49, 69, 76] tasks. Occlusions are particularly important for 3D aware image generation. However, it has received little attention in existing works [4, 19]. Some works model occlusions by decomposing images into flat 2D object layers [14, 37, 76], but they lack 3D awareness, resulting in geometrically inconsistent occlusions. To bridge the gap in existing works, we propose SeeThrough3D, a model that enables generalized occlusion-aware 3D layout control.

###### 3. Method

Our goal is to generate an image conditioned on a text prompt and a scene layout consisting of 3D bounding boxes. We build on a pretrained text-to-image flow model [35] and condition on the proposed Occlusion-Aware 3D Scene Representation (OSCR) (see Fig. 2).

###### 3.1. OSCR

Existing methods for 3D layout–conditioned generation represent scene layouts either by computing depth maps of 3D bounding boxes (see Fig. 3(a)) or by simplifying the scene into a finite set of 2D object layers (Fig. 3(b)). These representations, however, fail to capture true 3D structure of the scene, resulting in inaccurate occlusion modeling and

[Figure 17]

[Figure 18]

[Figure 19]

- (a) Layout depth maps

3D control

Occlusion

[Figure 20]

[Figure 21]

3D control

Occlusion

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

3D control

Occlusion

[Figure 26]

(d) Generated Image

Figure 3. Towards occlusion aware 3D scene layouts: existing methods represent scenes as (a) 3D layout depth maps [4, 19, 65], which fail to represent occluded objects (see dashed red box), or

- (b) object layers [37, 76], which are not 3D aware, hence fail to capture camera viewpoint and perspective. (c) Therefore, we propose OSCR, where objects are described using translucent 3D bounding boxes. The transparency exposes occluded regions (red box), providing cues for occlusion reasoning, while enabling 3D layout control.

(b) Object-based layering

(c) Ours: Translucent boxes

limited orientation control. To overcome this, we design OSCR, an efficient yet effective representation that encodes 3D layouts in an occlusion-aware manner.

Our input is a set of 3D bounding boxes bi, each representing an object, arranged in a 3D virtual environment

- (see Fig. 2(a)). To encode object orientation, we define a canonical color mapping across box faces, where each face is assigned a predefined color (see Fig. 2(b)). This mapping provides an explicit and interpretable encoding of 3D orientation directly in image space. To make the representation aware of spatial ordering and occlusions, we render the boxes as translucent, allowing occluded objects to remain partially visible. This simple yet expressive design compactly captures both orientation and occlusion cues
- (see Fig. 2(b)). Notably, occlusion may alter the apparent colors of some faces, causing them to deviate from the predefined mapping. However, the relative color differences between faces remain discernible, preserving reliable orientation cues. Finally, we render the composed scene from a specified camera view C using Blender [12]. The rendered image inherently embeds camera pose information, enabling precise viewpoint control in generation. The rendered image r is used as ‘OSCR condition’ to the generative model (see Fig. 4).

###### 3.2. SeeThrough3D

We build on FLUX [35], a DiT-based text-to-image model. FLUX comprises a series of multimodal DiT blocks that jointly process text and image tokens through self-attention and feed-forward layers (see Fig. 4). This architecture facilitates rich information exchange between text and image tokens, resulting in strong image-text alignment during generation. Further, this design naturally supports an effective way to condition the model on a new modality by adding condition tokens [61, 62, 79]. Leveraging this, we condi-

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

###### output tokens

||Self-Attention|LoRA<br><br>[Figure 31]|
|---|---|
<br><br>N x m m D i T b l o c k s<br><br>|Feed-forward|
|---|
|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|.|
|---|

|.|
|---|

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

text tokens image tokens OSCR tokens

[Figure 40]

[Figure 41]

'A photo ofcar anddeer in a grassy  eld'

text prompt noisy image OSCR condition

Figure 4. SeeThrough3D: We encode the rendered OSCR condition map r using the VAE to obtain OSCR tokens. These are concatenated with text prompt tokens p and noisy image tokens xt. The concatenated result is passed through the DiT based text-toimage model where they are jointly processed using self attention modules. We inject LoRA [25] onto the attention projections corresponding to OSCR tokens; this enables control while preserving prior of the base model [61, 62, 79].

tion the model on the rendered OSCR layout representation r (see Fig. 4). Specifically, we first encode r using the VAE to obtain OSCR tokens z, which are concatenated with text prompt tokens p and the noisy image tokens xt. The OSCR tokens z are assigned the same positional encodings as the noisy image tokens xt, establishing spatial correspondence between them. The combined token sequence is then processed by mmDiT blocks. To adapt the model to OSCR condition while preserving its text-to-image prior, we train a LoRA [25] only on the projection matrices associated with the newly added tokens (see Fig. 4). In line with recent work [79], we also block attention from OSCR tokens z to the image tokens xt (see Fig. 5).

###### 3.3. Object binding with attention masking

While the conditioning mechanism described above ensures spatial alignment with the given layout, it does not explicitly associate 3D bounding boxes with their corresponding object identities. This ambiguity arises because OSCR encodes geometric arrangements of objects but lacks semantic information about them, which can lead to mismatched object placements during generation. A straightforward solution would be to encode object classes as colors within the boxes, similar to semantic segmentation. However, this approach constrains the model to a fixed set of predefined categories and limits generalization. Instead, we utilize the attention mechanism to enrich OSCR tokens with corresponding object semantics. Specifically, we mask the attention so that OSCR tokens z within each bounding box only attend to corresponding object noun tokens pi in the text prompt,

(b) Masked attention from OSCR tokens to object tokens in prompt.

(a) Attention inside mmDiT block. Black regions indicate no attention.

[Figure 42]

[Figure 43]

text keys image keys OSCR keys

| | | | | | | |
|---|---|---|---|---|---|---|
| |... ...<br><br>[Figure 44]<br><br>| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

......

textqueriesimagequeriesOSCRqueries

OSCR tokens in attend to car text token

OSCR tokens in attend to deer text token

OSCR tokens in attend to both car and deer text tokens

Figure 5. (a) Inside the mmDiT block, text tokens p, image tokens xt and OSCR tokens z are jointly processed using self attention, conditioning the generation on our OSCR representation. To bind objects to corresponding boxes, we mask the attention to enable OSCR tokens within each box {bi} to attend to corresponding object tokens {pi} using a mask M (b) For this, we require spatial extent for each object box bi, which we obtain we use its amodal segmentation mask si. When multiple boxes overlap, their region of intersection (green) attends to multiple objects.

(see Fig. 5(a) M ), thus enriching the spatial OSCR tokens with corresponding object semantics. For this, we require the spatial extents for each box bi, which we obtain using its rendered segmentation mask , (see Fig. 5(b)) using Blender.

Handling overlapping objects: A challenging case for the proposed object binding arises when the rendered regions of two boxes significantly overlap. In this scenario, the OSCR tokens in the intersection region attend to multiple object tokens (see Fig. 5(b)). At first glance, it appears that attending to multiple objects would lead to semantic blending or visual artifacts at object boundaries. To investigate this, we condition our model on a complex layout with heavy occlusion (see Fig. 6(a)), and observe that the output contains precise occlusion boundaries (see Fig. 6(b)). To understand this further, we visualize attention from image tokens xt to object tokens {pi} in Fig. 6(c,d) Interestingly, the attention maps themselves reveal occlusion boundaries: inside the empty regions of the bicycle structure, attention on the van remains visible, accurately reflecting its presence behind the bicycle. This indicates that object-specific features remain distinct in the model’s latent space, and that the text-to-image model encodes necessary priors for occlusion reasoning. Our OSCR representation (see Fig. 6(a)) leverages these priors for precise control over scene layout, in an occlusion-aware manner. Further analysis of attention is provided in appendix Sec. D.

‘A photo of bicycle and van in a beautiful garden in the morning.’

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

(b) Generated image (c) Bicycle (d) Van

(a) OSCR condition

Figure 6. Visualizing object disentanglement in latent space: Given a layout with heavy occlusion like (a), our model’s outputs show precise occlusion boundaries (b). To understand this, we visualize attention from image-tokens to object tokens in prompt (bicycle and van). Interestingly, the attention maps themselves reveal occlusion boundaries: inside the empty regions of the bicycle structure, attention on the van remains visible, accurately reflecting its presence behind the bicycle. This suggests that objectspecific features remain distinct in the model’s latent space, indicating strong priors for occlusion reasoning.

- 3.4. Personalization The proposed method naturally supports layout-conditioned generation with personalized objects. Given a reference object image v, a text prompt p, and OSCR layout r, the goal is to generate the object adhering to a specific 3D bounding

box bi in the layout r. We first encode object appearance by passing the reference image v through the VAE encoder, resulting in ‘appearance tokens’ v. These are concatenated with text tokens p, target image tokens xt, and OSCR tokens z before passing through the mmDiT blocks. To bind the object’s appearance to its corresponding 3D box bi, we re-use the attention masking strategy described above. Specifically, we enable OSCR tokens inside the segmentation mask si to attend to appearance tokens v. This enables layout-aware generation of personal objects, and can be extended to multiple objects by adding separate appearance token sets for each reference image (see Fig. 11).

- 3.5. Dataset To adapt the model to OSCR representation, we require a dataset of paired images and 3D bounding boxes. While existing 3D object detection datasets [13, 58] could be used, they are often domain specific, lack occlusion scenarios, have minimal viewpoint variation and contain marginal errors in 3D annotations, making them unsuitable for our purposes. Therefore, we create a synthetic dataset using Blender [12]; where we procedurally place 3D assets in controlled configurations on the floor (x-y plane). Next, we render the paired ground truth image and OSCR representation from diverse camera viewpoints. We discard trivial scenes with minimal object overlap or very low visibility of any object, as we find such filtering crucial for maintaining occlusion consistency in the generated results (see Sec. 4.4). Augmentations: Training solely on rendered images risks overfitting to synthetic backgrounds [9, 49], due to limited realism and lack of diversity in object appearance and backgrounds. Since creating highly varied 3D scenes is

|[Figure 49]<br><br>CAMERA|
|---|

[Figure 50]

[Figure 51]

[Figure 52]

Blender

box representation rendered image

3D environment

|[Figure 53]|
|---|

[Figure 54]

[Figure 55]

[Figure 56]

FLUX.1 Depth

realistic augmentations depth map

Figure 7. Dataset creation: We place 3D assets in controlled configurations in Blender [12]. Object placements and camera viewpoint are controlled to ensure strong occlusions, while ensuring adequate visibility for each object. To generate realistic augmentations, we estimate image depth, and pass it through a depth-toimage model [35] with diverse background prompts.

an expensive process, we adopt a scalable alternative. We generate realistic augmentations for the rendered images, that follow the same layout but are rich in terms of appearance diversity. For each rendered image, we extract its depth and feed it through a depth-to-image generation pipeline (FLUX.1-Depth-dev) [35] to synthesize realistic images that preserve the same spatial layout. Although this pipeline produces high-quality results, it occasionally misaligns objects with their intended depth regions, causing incorrect placements. We mitigate this by applying objectlevel CLIP-based filtering [56] to retain only those augmentations that adhere to the original layout. Our final dataset comprises 25K rendered images and 25K augmentations. Further details about dataset pipeline and dataset statistics are provided in appendix Sec. B.

###### 4. Experiments

###### 4.1. Experimental setup

Implementation details: We use FLUX.1-dev [35] as the text-to-image model. We train for 30K steps at a learning rate of 10−4, using a LoRA rank of 128. A detailed implementation report can be found in appendix Sec. E.

Evaluation dataset: Accurate evaluation of occlusionaware 3D control requires a benchmark of paired images and 3D bounding box annotations that exhibit 1) diverse object configurations 2) challenging occlusion scenarios, and 3) wide range of camera viewpoints. To facilitate this, we introduce 3D Control with Occlusions benchmark, 3DOcBench, a dataset with 500 samples of paired 3D boundingbox layouts, rendered images, and scene text prompts. We construct the benchmark in Blender [12] by placing 3D assets on a ground plane and procedurally varying object arrangements and camera poses to produce strong occlu-

|3<br><br>[Figure 57]<br><br>A||scene_ 202511 10_013 633.pkl<br><br>[Figure 63]|[Figure 64]<br><br>[Figure 65]<br><br>B|
|---|---|
| | |
<br><br>[Figure 61]|
|---|---|
| |scene_ 202511 10_013 918.pkl<br><br>[Figure 62]|

|scene_ 202511 10_013 633.pkl<br><br>[Figure 63]|[Figure 64]<br><br>[Figure 65]<br><br>B|
|---|---|
| | |

|[Figure 66]<br><br>[Figure 67]<br><br>C||scene_2 0251105 _214126 .pkl<br><br>[Figure 72]|4 subjects<br><br>[Figure 73]<br><br>[Figure 74]<br><br>D|
|---|---|
| | |
|
|---|---|
| |scene_2 0251105 _214515 .pkl<br><br>[Figure 71]|

|scene_2 0251105 _214126 .pkl<br><br>[Figure 72]|4 subjects<br><br>[Figure 73]<br><br>[Figure 74]<br><br>D|
|---|---|
| | |

|3<br><br><br><br><br><br>C||scene_2 0251110 _134146 .pkl<br><br>|<br><br>D2<br><br><br><br><br><br>|
|---|---|
| | |
|
|---|---|
| |scene_2 0251110 _143003 .pkl<br><br>|

|scene_2 0251110 _134146 .pkl<br><br>|<br><br>D2<br><br><br><br><br><br>|
|---|---|
| | |

|[Figure 86]|
|---|

‘A photo of green parrot and blue parrot and transparent glass of water and a stack of books on an old study table.’

‘A photo of blue sedan and suv and motorbike and white sedan in a parking lot.’

‘A photo of mars rover and astronaut and alien and spaceship and blue neon robot dog on Mars, sci-ﬁ scene.’

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

###### E F

|<br><br><br><br>E||scene_2 0251105 _204820 .pkl<br><br>|4 subjects<br><br><br><br><br><br>F|
|---|---|
| | |
|
|---|---|
| |scene_2 0251105 _205338 .pkl<br><br>|

|scene_2 0251105 _204820 .pkl<br><br>|4 subjects<br><br><br><br><br><br>F|
|---|---|
| | |

scene_20251110_162424

scene_20260124_012706

6 subjects

5 subjects

.pkl

.pkl

‘A photo of woman playing guitar and man playing guitar in a music room, ultra-realistic.’

‘A photo of dog riding a bicycle in the backyard of a house’

‘A photo of red boat with letters ‘CVPR’ written on it and black boat with ‘DENVER’ written on it and sedan and man at a beautiful beach at sunset.’

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

G H

scene_20251110_162424

scene_20251110_152357

5 subjects

6 subjects

.pkl

.pkl

‘A photo of sofa and bride and groom and white horse and table scattered with ﬂowers and cat in a dreamy wedding garden at sunrise.’

‘A photo of monitor and printer and speaker and transparent water bottle and table lamp and keyboard and mouse and calendar saying ‘2026’ on a modern work desk.’

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

I J

scene_20251105_120101 .pkl

scene_20251105_075140 .pkl

7 subjects

‘A photo of sedan and sedan and suv and truck and bicycle and tow truck and ferrari and pickup truck and suv at a zebra crossing trafﬁc intersection at night.’

‘A photo of man and piano and ﬂower vase and table and guitar and woman and dog and cat in a modern living room.’

- Figure 8. Qualitative results: Our method is able to precisely follow 3D scene layouts, with high occlusion consistency. Our approach preserves the prior of text-to-image model, as evident from capabilities like see-through transparent objects (A,B,G,J), text rendering (G) and inter-object interactions (E,F). Additionally, our method enables control over viewpoint of generated image (C,D). Despite being trained on layouts with only upto 4 objects, our method is able to generalize to complex scenes with many objects (G,H,I,J).

1 2 3 4 5

‘A photo of tiger and deer on a sandy beach with waves…’

‘A photo of sedan and van and scooter and bicycle…’

‘A photo of horse and horse and motorbike and horse…’

‘A photo of ofﬁce chair, pig and dog and chair …’

‘A photo of dog and dog and chair on a picturesque…

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Layout

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

LooseControl

###### (A)

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Build-A-Scene

###### (B)

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

VODiff

###### (C)

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

- (D)

LaRender

- (E)

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Ours

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

- Figure 9. Qualitative comparison: We compare against works on 3D layout control: LooseControl [4] and Build-A-Scene [19], and on occlusion control: LaRender [76] and VODiff [37].

sions while preserving a minimum visible area for each object. We will release the benchmark for future research in occlusion-aware generation. Detailed benchmark statistics are provided in the appendix Sec. B.

Evaluation metrics: We measure the models’ performance for layout adherence, text-to-image alignment, and image quality. For text-to-image alignment, we use CLIP image-text similarity, and for image quality, we use Kernel Inception Distance (KID) [6]. Evaluating 3D layout adherence using a single metric is challenging, as the generated scene may not conform to the metric depth specified by the 3D bounding-box layout. To this end, we compute three metrics that in unison effectively evaluate 3D layout adherence. Specifically, we compute 2D bounding box adherence, relative visibility order and 3D orientation consistency. (1) For evaluating 2D layout adherence, we first obtain object masks by combining 2D layouts with Segment Anything [30]. Next, we compute CLIP similarity between the object masks and textual object descriptions, leading to CLIP objectness score. We aggregate this objectness score to evaluate the 2D layout adherence (2) For evaluating relative visibility order, we adopt a similar method as [26]: we estimate per-pixel depth [72] and obtain object depth estimates by averaging the depth within each object mask.

Baselines depth ord.↑ obj. score↑ angular err.↓ text align.↑ KID(×10−3) ↓ VODiff [37] 0.68 19.70 92.73 29.51 15.40

LooseControl [4] 0.82 20.02 89.88 28.43 14.32 Build-A-Scene [19] 0.89 21.0 91.62 28.05 20.12 LaRender [76] 1.02 21.83 89.63 30.20 13.46 Ours 1.46 22.86 47.92 31.87 5.43

Table 1. Quantitative comparison: We compute (a) depth ordering, which reflects 3D location and occlusion consistency, (b) CLIP objectness score, which indicates layout adherence and object fidelity (c) angular error, which indicates orientation correctness (d) image-text prompt alignment using CLIP [56], and (e) KID [5], which measures image fidelity.

Since all objects may not be present in the generated output, we use previously defined objectness score to filter out object masks. Finally, we compare relative depth ordering of each object pair against the ground-truth ordering, assigning a score of 1 if the ordering is correct and 0 otherwise. We aggregate this score over all such pairs 3) For assessing orientation accuracy, we employ OrientAnything [67] to estimate object orientations using filtered object segments, and compute mean absolute error against ground truth.

Baselines: We compare our method with state-of-the-art works in 3D layout control: LooseControl [4] and Build-AScene [19]. LooseControl uses layout depth maps to condition a diffusion model for scene layout control, while BuildA-Scene is an inference time method that uses pretrained LooseControl checkpoint. For fair evaluation, we train LooseControl on our dataset, and use the checkpoint to evaluate both methods. We also consider works on orientation control, Compass Control [49] and ORIGEN [43], though they do not support 3D object placement, hence not directly relevant. We compare against them in appendix Sec. G. We further evaluate against occlusion control methods, LaRender [76] and VODiff [37]. These methods decompose an image into 2D object layers to manage visibility ordering.

###### 4.2. Results

Qualitative: We present our qualitative results in Fig. 8. Our method is able to generate realistic scenes with intricate inter-object overlaps. It effectively preserves the prior of the base text-to-image model, evident from capabilities like see-through transparent objects (A,B,G,J) and text rendering (G). Additionally, our method enables control over viewpoint of generated image (C,D). Despite being trained on layouts with only upto 4 objects, our method is able to generalize to complex scenes with many objects (G,H,I,J). Even though our synthetic data consists of rigid objects in fixed canonical poses, our method is able to generate diverse poses such as sitting (H,J) and cycling (E). The model generates natural inter-object interactions (dog riding bicycle in E, person playing guitar in F), even though our synthetic data does not contain such interactions. Further, it generalizes strongly to out-of-domain objects. Notably, our training dataset does not contain any musical instruments

(F,J), electronic devices (G), transparent object (A,B,G,J) or books (A,B), but our model is able to effectively generalize to them.

Baseline comparisons: We present results in Tab. 1 and Fig. 9. 3D scene control: LooseControl [4] fails to handle complex occlusions, as layout depth fails to represent occluded objects (see Fig. 9 A1,3-5). Additionally, the objects are generated in incorrect locations, due to lack of binding (A1,3), also reflected in low objectness-score (see Tab. 1). Build-A-Scene [19] uses multiple generation and inversion cycles to sequentially add objects to the scene. While this improves upon layout adherence and occlusion consistency compared to LooseControl [4], it leads to inve artifacts (B2-3,5), and hence worse KID value. The se generation also leads to lack of coherence in the generated scene (B4), since initial generations are independent of final scene layout. Both the methods fail to provide precise orientation control, since layout depth maps can only encode orientation upto 180◦ flip, leading to high angular error. In contrast, our method is able to generate coherent images with precise 3D layout and orientation control. Occlusion control: LaRender [76] and VODiff [37] rely on 2D layouts as conditioning input, which fail to discertain exact object arrangements. For instance, in Fig. 9 (C4, D4-5), the object is generated on ‘top of the chair’, against the intended configuration ‘behind the chair’. In contrast, our OSCR representation is 3D aware, hence offers more precise control than 2D layouts. In case of large overlap between 2D bounding boxes in layout, baseline methods often fail to generate occluded objects (C1,3-4, D3-4) in contrast to SeeThrough3D, which can generate very occluded objects accurately (E).

|version sequential<br><br>scene_2 0251107 _142821 .pkl<br><br>[Figure 169]|
|---|

User study: We conducted an A/B user study where 60 participants were asked to choose between output of our method and a randomly chosen baseline. We evaluate a) image realism, b) layout adherence, and c) text prompt alignment. Results highlight high preference for our method in all evaluation categories (see Fig. 10).

###### Image realism (%)

Layout adherence (%)

Prompt following (%)

|95.92|96.71|97.71|97.93|
|---|---|---|---|
| | | | |

|96.73|96.4|98.39|96.84|
|---|---|---|---|
| | | | |

90.79 86.16 98.1 89.15

Ours LaRender VODiff LooseControl Build-A-Scene

- Figure 10. User study: Each bar indicates the % of times our method’s output was preferred over the baseline, for each category.

Ablations depth ord.↑ obj. score↑ angular err.↓ text align.↑ KID(×10−3) ↓

w/o transparency 1.20 21.67 46.15 31.39 5.90 w/o color-coding 1.36 22.23 88.77 31.57 5.93

w/o binding 0.98 20.45 57.44 31.61 6.35 w/o hard data 1.24 21.89 49.73 31.32 6.34

Ours 1.46 22.86 47.92 31.87 5.43

Table 2. Quantitative results of ablative experiments.

- 4.3. Personalization We show personalization results in Fig. 11. We adapt our training

|dataset assets, and<br><br>scene_2 0251107 _131753 .pkl<br><br>[Figure 170]<br><br>[Figure 171]|
|---|

et for personalization by applying textures to 3D a d using this textured object as reference image. Further details and results are in appendix Sec. I.

|[Figure 172]<br><br>[Figure 173]|scene_2 0 7 _ 0 .pkl<br><br>|0251107 _133050<br><br>[Figure 174]|
|---|
<br><br>[Figure 175]|[Figure 176]|
|---|---|---|
| |scene_2 0251107 _130459 .pkl<br><br>[Figure 177]<br><br>[Figure 178]| |
|[Figure 179]|scene_2 0251107 _142821 .pkl<br><br>[Figure 180]|[Figure 181]<br><br>[Figure 182]|
| |scene_2 0251107 _143524 .pkl<br><br>[Figure 183]| |

van__bic ycle__co upe/003

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

‘A photo of and man in a forest.’

<obj0>

<obj0>

<obj0>

‘A photo of <obj0> and <obj1> in a modern room.’

<obj1> <obj0>

<obj1>

Figure 11. Personalization: Our method can be extended for personalized 3D control using reference image of an object.

- 4.4. Ablations

We study the impact of key design choices, with results shown in Fig. 12 and Tab. 2. Box transparency plays a crucial role in the effectiveness of the OSCR representation, enabling reasoning about occluded objects and relative depth. Color-coding the box faces helps encode orientation and significantly reduces angular error (see Tab. 2). Interestingly, opaque boxes yield the best orientation accuracy due to a clearer color signal. The attention-based binding is essential for layout adherence—without it, objects appear at incorrect locations (see Fig. 12, 1C and 3C), resulting in lower objectness score. Finally, filtering out overly simplistic layouts in data improves performance.

- 5. Conclusion

We present SeeThrough3D, a model for occlusion aware 3D layout control. We introduce OSCR, an occlusion aware 3D scene representation. We show that our approach can faithfully model heavy occlusion scenarios, while preserving strong text-to-image prior of the model. Despite training on limited synthetic data, it exhibits strong generalization capabilities. We perform evaluations to show that our method outperforms existing baselines, and also ablate upon key design choices, providing useful insights for future research. While effective in layout adherence, our method does not preserve image consistency under layout changes. A future

‘A photo of giraffe and bicycle traversing a rocky riverbed in a deep canyon under a hot sun’

(A) w/o transparency

(B) w/o color (C) w/o binding coding

(D) occlusions in (E) Ours data

Layout

|[Figure 192]<br><br>[Figure 193]|
|---|

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

- 1

- 2

- 3

‘A photo of ofﬁce chair and chair and goat in a cozy rustic cabin’

|[Figure 203]<br><br>[Figure 204]|
|---|

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

|[Figure 214]<br><br>[Figure 215]|
|---|

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

‘A photo of elephant, table, table, jeep on a beach with crashing ocean waves and dramatic cliffs.’

- Figure 12. Ablations: We ablate upon key aspects of OSCR representation, our binding mechanism and data preparation strategy.

direction is to address this by using editing.

###### 6. Acknowledgements

We thank Harshavardhan P., Ayan Kashyap, Vansh Garg, Jainit Bafna, Abhinav Raundhal, Varun Gupta, Shivank Saxena, Akshat Sanghvi and Aishwarya Agarwal for helpful discussions and reviewing the manuscript.

###### References

- [1] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. Stable flow: Vital layers for training-free image editing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7877–7888, 2025. 6
- [2] Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, et al. Gaudi: A neural architect for immersive 3d scene generation. Advances in Neural Information Processing Systems, 35:25102–25116, 2022. 2
- [3] Edurne Bernal-Berdun, Ana Serrano, Belen Masia, Matheus Gadelha, Yannick Hold-Geoffroy, Xin Sun, and Diego Gutierrez. Precisecam: Precise camera control for textto-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2724–2733,

2025. 2

- [4] Shariq Farooq Bhat, Niloy Mitra, and Peter Wonka. Loosecontrol: Lifting controlnet for generalized depth conditioning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,

2024. 2, 3, 7, 8, 4, 5, 6, 12, 13

- [5] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 7
- [6] Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In International Conference on Learning Representations, 2018. 7
- [7] James Burgess, Kuan-Chieh Wang, and Serena Yeung. Viewpoint textual inversion: Unleashing novel view synthe-

- sis with pretrained 2d diffusion models. arXiv preprint arXiv:2309.07986, 2023. 2
- [8] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. corr abs/2311.14521

(2023), 2023. 2

- [9] Ta-Ying Cheng, Matheus Gadelha, Thibault Groueix, Matthew Fisher, Radomir Mech, Andrew Markham, and Niki Trigoni. Learning continuous 3d words for text-toimage generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6753–6762, 2024. 2, 3, 5, 1
- [10] Yen-Chi Cheng, Krishna Kumar Singh, Jae Shin Yoon, Alexander Schwing, Liang-Yan Gui, Matheus Gadelha, Paul Guerrero, and Nanxuan Zhao. 3d-fixup: Advancing photo editing with 3d priors. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 2
- [11] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1
- [12] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 2, 3, 5, 1, 4, 6, 10
- [13] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 5
- [14] Aneel Damaraju, Dean Hazineh, and Todd Zickler. Cobl: Toward zero-shot ordinal layering without user prompting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8154–8164, 2025. 3
- [15] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023. 1
- [16] Ankit Dhiman, Manan Shah, Rishubh Parihar, Yash Bhalgat, Lokesh R Boregowda, and R Venkatesh Babu. Reflecting reality: Enabling diffusion models to produce faithful mirror reflections. arXiv preprint arXiv:2409.14677, 2024. 2
- [17] Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, and Anand Bhattad. Generative models: What do they know? do they know things? let’s find out! arXiv preprint arXiv:2311.17137, 2023. 2
- [18] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3d awareness of visual foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21795–21806, 2024. 2

- [19] Abdelrahman Eldesokey and Peter Wonka. Build-a-scene: Interactive 3d layout control for diffusion-based image generation. arXiv preprint arXiv:2408.14819, 2024. 2, 3, 7, 8, 4, 5, 6, 12, 13
- [20] Alejandro Escontrela, Shrinu Kushagra, Sjoerd van Steenkiste, Yulia Rubanova, Aleksander Holynski, Kelsey Allen, Kevin Murphy, and Thomas Kipf. Neural usd: An object-centric framework for iterative editing and control. arXiv preprint arXiv:2510.23956, 2025. 2
- [21] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2

- [22] Alhussein Fawzi and Pascal Frossard. Measuring the effect of nuisance variables on classifiers. In BMVC, pages 137–1,

2016. 3

- [23] Tsu-Jui Fu, Yusu Qian, Chen Chen, Wenze Hu, Zhe Gan, and Yinfei Yang. Univg: A generalist diffusion model for unified image generation and editing. arXiv preprint arXiv:2503.12652, 2025. 2
- [24] Richard Higgins and David Fouhey. Seldom: Scene editing via latent diffusion with object-centric modifications. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 7046–7058,

2025. 2

- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 4
- [26] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 7
- [27] Kaleb Kassaw, Francesco Luzi, Leslie M Collins, and Jordan M Malof. Are deep learning models robust to partial object occlusion in visual recognition tasks? Pattern Recognition, page 112215, 2025. 3
- [28] Kunal Kathare, Ankit Dhiman, K Vikas Gowda, Siddharth Aravindan, Shubham Monga, Basavaraja Shanthappa Vandrotti, and Lokesh R Boregowda. Instructive3d: Editing large reconstruction models with text instructions. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pages 3246–3256, 2025. 2
- [29] Hyunsu Kim, Gayoung Lee, Yunjey Choi, Jin-Hwa Kim, and Jun-Yan Zhu. 3d-aware blending with generative nerfs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22906–22918, 2023. 2
- [30] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 7, 3, 6
- [31] Eunseo Koh, Sangeek Hyun, MinKyu Lee, Jiwoo Chung, Kangmin Seo, and Jae-Pil Heo. Diffusion feature field for

- text-based 3d editing with gaussian splatting. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. 2
- [32] Adam Kortylewski, Qing Liu, Huiyu Wang, Zhishuai Zhang, and Alan Yuille. Combining compositional models and deep networks for robust object classification under occlusion. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1333–1341, 2020. 3
- [33] Nupur Kumari, Grace Su, Richard Zhang, Taesung Park, Eli Shechtman, and Jun-Yan Zhu. Customizing text-to-image diffusion with camera viewpoint control. arXiv preprint arXiv:2404.12333, 2024. 2
- [34] Nupur Kumari, Grace Su, Richard Zhang, Taesung Park, Eli Shechtman, and Jun-Yan Zhu. Customizing text-to-image diffusion with camera viewpoint control. arXiv preprint arXiv:2404.12333, 2024. 2
- [35] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 2, 3, 5, 1, 6, 9, 10

- [36] Zhenyu Li, Mykola Lavreniuk, Jian Shi, Shariq Farooq Bhat, and Peter Wonka. Amodal depth anything: Amodal depth estimation in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9673–9682, 2025. 3
- [37] Dong Liang, Jinyuan Jia, Yuhao Liu, Zhanghan Ke, Hongbo Fu, and Rynson WH Lau. Vodiff: Controlling object visibility order in text-to-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18379–18389, 2025. 2, 3, 7, 8, 6, 12, 13
- [38] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023. 2
- [39] Zhengzhe Liu, Qing Liu, Chirui Chang, Jianming Zhang, Daniil Pakhomov, Haitian Zheng, Zhe Lin, Daniel Cohen-Or, and Chi-Wing Fu. Object-level scene deocclusion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
- [40] Guan Luo, Tian-Xing Xu, Ying-Tian Liu, Xiao-Xiong Fan, Fang-Lue Zhang, and Song-Hai Zhang. 3d gaussian editing with a single image. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 6627–6636,

2024. 2

- [41] L´eopold Maillard, Tom Durand, Adrien Ramanana Rahary, and Maks Ovsjanikov. Laconic: A 3d layout adapter for controllable image creation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18046– 18057, 2025. 3
- [42] Rupayan Mallick, Sibo Dong, Nataniel Ruiz, and Sarah Adel Bargal. D-feat occlusions: Diffusion features for robustness to partial visual occlusions in object recognition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1722–1731, 2025. 3
- [43] Yunhong Min, Daehyeon Choi, Kyeongmin Yeo, Jihyun Lee, and Minhyuk Sung. Origen: Zero-shot 3d orienta-

- tion grounding in text-to-image generation. arXiv preprint arXiv:2503.22194, 2025. 2, 3, 7, 4, 5, 6, 9
- [44] Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li, and Bolei Zhou. Freecontrol: Training-free spatial control of any text-to-image diffusion model with any condition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7465–7475, 2024. 2
- [45] Thu Nguyen-Phuoc, Chuan Li, Lucas Theis, Christian Richardt, and Yong-Liang Yang. Hologan: Unsupervised learning of 3d representations from natural images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7588–7597, 2019. 2
- [46] Michael Niemeyer and Andreas Geiger. Giraffe: Representing scenes as compositional generative neural feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11453–11464, 2021. 2
- [47] Karran Pandey, Paul Guerrero, Matheus Gadelha, Yannick Hold-Geoffroy, Karan Singh, and Niloy J Mitra. Diffusion handles enabling 3d edits for diffusion models by lifting activations to 3d. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7695– 7704, 2024. 2
- [48] Rishubh Parihar, Harsh Gupta, Sachidanand VS, and R Venkatesh Babu. Text2place: Affordance-aware text guided human placement. In European Conference on Computer Vision, pages 57–77. Springer, 2024. 2
- [49] Rishubh Parihar, Vaibhav Agrawal, Sachidanand VS, and Venkatesh Babu Radhakrishnan. Compass control: Multi object orientation control for text-to-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2791–2801, 2025. 2, 3, 5, 7, 1, 4, 6, 9
- [50] Rishubh Parihar, Srinjay Sarkar, Sarthak Vora, Jogendra Nath Kundu, and R Venkatesh Babu. Monoplace3d: Learning 3d-aware object placement for 3d monocular detection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6531–6541, 2025. 3
- [51] Rishubh Parihar, Sachidanand VS, and R Venkatesh Babu. Zero-shot depth aware image editing with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15748–15759, 2025. 2
- [52] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 4
- [53] Or Patashnik, Rinon Gal, Daniel Cohen-Or, Jun-Yan Zhu, and Fernando De la Torre. Consolidating attention features for multi-view image editing. arXiv preprint arXiv:2402.14792, 2024. 2
- [54] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [55] Zhenyuan Qin, Xincheng Shuai, and Henghui Ding. Scenedesigner: Controllable multi-object image generation with 9-dof pose manipulation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems. 2, 3
- [56] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 5, 7, 1, 2
- [57] Rahul Sajnani, Jeroen Vanbaar, Jie Min, Kapil Katyal, and Srinath Sridhar. Geodiffuser: Geometry-based image editing with diffusion models. arXiv preprint arXiv:2404.14403,

2024. 2

- [58] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 567–576, 2015. 5
- [59] Florian Spiess, Raphael WaltenspAˇ˜ zl, and Heiko Schuldt. The sketchfab 3d creative commons collection (s3d3c). arXiv preprint arXiv:2407.17205, 2024. 1
- [60] Wei-En Tai, Yu-Lin Shih, Cheng Sun, Yu-Chiang Frank Wang, and Hwann-Tzong Chen. Segment anything, even occluded. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29385–29394, 2025. 3
- [61] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024. 2, 3, 4, 9
- [62] Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280,

2025. 2, 3, 4, 9

- [63] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 4
- [64] Qian Wang, Yiqun Wang, Michael Birsak, and Peter Wonka. Blobgan-3d: A spatially-disentangled 3d-aware generative model for indoor scenes. arXiv preprint arXiv:2303.14706,

2023. 2

- [65] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 2, 3
- [66] Ruicheng Wang, Jianfeng Xiang, Jiaolong Yang, and Xin Tong. Diffusion models are geometry critics: Single image 3d editing using pre-trained diffusion priors. arXiv preprint arXiv:2403.11503, 2024. 2
- [67] Zehan Wang, Ziang Zhang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Orient anything: Learning robust object orientation estimation from rendering 3d models. arXiv preprint arXiv:2412.18605, 2024. 7, 5

- [68] Minghao Wen, Shengjie Wu, Kangkan Wang, and Dong Liang. Intergsedit: Interactive 3d gaussian splatting editing with 3d geometry-consistent attention prior. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26136–26145, 2025. 2
- [69] Tianhao Wu, Chuanxia Zheng, Frank Guan, Andrea Vedaldi, and Tat-Jen Cham. Amodal3r: Amodal 3d reconstruction from occluded 2d images. arXiv preprint arXiv:2503.13439,

2025. 3

- [70] Ziyi Wu, Yulia Rubanova, Rishabh Kabra, Drew A Hudson, Igor Gilitschenski, Yusuf Aytar, Sjoerd van Steenkiste, Kelsey R Allen, and Thomas Kipf. Neural assets: 3d-aware multi-object scene synthesis with image diffusion models. arXiv preprint arXiv:2406.09292, 2024. 3
- [71] Yang Xue, Yuheng Li, Krishna Kumar Singh, and Yong Jae Lee. Giraffe hd: A high-resolution 3d-aware generative model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18440–18449,

2022. 2

- [72] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024. 7
- [73] Jiraphon Yenphraphai, Xichen Pan, Sainan Liu, Daniele Panozzo, and Saining Xie. Image sculpting: Precise object editing with 3d geometry control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4241–4251, 2024. 2
- [74] Guanqi Zhan, Chuanxia Zheng, Weidi Xie, and Andrew Zisserman. What does stable diffusion know about the 3d scene? arXiv preprint arXiv:2310.06836, 2023. 2
- [75] Guanqi Zhan, Chuanxia Zheng, Weidi Xie, and Andrew Zisserman. Amodal ground truth and completion in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28003–28013, 2024. 3
- [76] Xiaohang Zhan and Dingming Liu. Larender: Training-free occlusion control in image generation via latent rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19679–19688, 2025. 2, 3, 7, 8, 6, 12, 13
- [77] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2
- [78] Qihang Zhang, Yinghao Xu, Chaoyang Wang, Hsin-Ying Lee, Gordon Wetzstein, Bolei Zhou, and Ceyuan Yang. 3ditscene: Editing any scene via language-guided disentangled gaussian splatting. arXiv preprint arXiv:2405.18424,

2024. 2

- [79] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027, 2025. 2, 3, 4, 6, 9

###### A. Overview

This appendix provides additional analysis, details about the dataset and model implementation, experimental discussion referenced in the main paper and extended qualitative results and comparisons. To skim over this material, the reader is advised to go through the figures and captions, which have been endowed with sufficient detail to understand the key content.

###### B. Dataset

###### B.1. The rendering pipeline

We collect 39 assets from Objaverse [15] and SketchFab [59] repositories from the internet. However, these assets are not aligned, making it difficult to define a canonical orientation for the objects. Hence, we align these assets manually in Blender [12] to ensure that their canonical front directions are aligned with the +Y axis. We further scale each asset to match relative real-world dimensions, for example, the size of jeep is smaller than an elephant, the scale values were obtained using Gemini 2.5 Pro [11]. We place the aligned assets in a Blender environment (upto 4 objects per scene) and add a virtual camera to render the scene from a given viewpoint. Specifically, we define a hemispherical region around the origin of a fixed radius R, within which all objects are placed. The camera lies at the surface of the hemisphere, always pointing towards the origin.

However, randomly placing the assets and the camera might result in unnatural-looking compositions, such as those where objects are colliding with each other. Additionally, as described in the ablations section (main paper), a key requirement is that the objects must be heavily occluded to ensure optimal training of our model. To cater to the above requirements, we adopt a procedural generation, where the scene configuration (camera and object placements) is first randomly sampled from a uniform distribution of parameters with some predefined constraints. This is followed by a filtering logic to remove the poor-quality examples.

Filtering based on occlusion: We filter the rendered scenes according to the extent of occlusion, to ensure heavy occlusion scenarios. For this, we require a metric to measure the extent to which an object is occluded. Therefore, we define a visibility ratio x, which is the ratio of visible area v of the object to the total area a of the object. The values v and a are measured using object segmentation masks, obtained through Blender [12]. We filter out cases where x > 0.7 for all objects in the scene, i.e., no object is occluded enough. Similarly, we filter out cases where x < 0.3 for any object, to ensure that each object is adequately visible in the image. Filtering based on object size: We filter out cases where an object is too small or too large, to avoid unnatural-looking images. We filter based on the largest side of 2D object

bounding boxes in the renderings. Specifically, we ensure that the largest side of the 2D bounding box must be within 0.125 and 0.750 of the image size.

###### B.2. Augmentations

Training solely on these rendered images from Blender risks overfitting to synthetic backgrounds [9, 49], due to limited realism and lack of diversity in object appearance and backgrounds. Since creating highly varied 3D scenes is an expensive process, we adopt a scalable alternative. We generate realistic augmentations for the rendered images that follow the same layout but are rich in terms of appearance diversity. For each rendered image, we extract its depth and pass it through a depth-to-image generation pipeline (FLUX.1-Depth-dev) [35] to synthesize realistic images that preserve the same spatial layout.

Filtering augmentation samples. Although this pipeline produces high-quality results, it occasionally misaligns objects with their intended depth regions, causing incorrect placements. For instance, on the left pane in Fig. 13(a), the depth-to-image model incorrectly generates a pigeon, instead of a crow, according to the original layout. We mitigate such issues by applying object-level CLIP-based filtering [56] to retain only those augmentations that adhere to the original layout. For this, we first obtain the object segmentation masks using Blender [12], and use these to obtain cropped object segments in the augmented image (see Fig. 13(b)). Next, we compute CLIP similarity between these object segments and corresponding text description (e.g. cat, pigeon, etc.), as shown in Fig. 13(c). If any object has a CLIP score less than the threshold value of 0.25, the augmented image is filtered out. High CLIP scores for all object segments (as shown on the right pane in Fig. 13) indicate accurate layout adherence, and these images are included in the training dataset. We visualize some examples from our training dataset in Fig. 16.

###### B.3. Statistics

We present various statistics of our training dataset in Fig. 14. (a) We plot the distribution ofthe minimum visibility ratio for any object in the scene. Since our filtering strategy favors heavy occlusion scenarios, we observe that there is a bias towards low visibility ratio cases. (b) Next, we observe that the distribution of orientation values is roughly uniform, thus avoiding any unwanted biases. (c) Interestingly, we observe that the frequency of examples with large 2D bounding dimensions shows a decreasing trend. This is because smaller object sizes enable the placement of multiple objects in a scene, while ensuring all of them are visible. (d) By common observation, high camera shots tend to have weaker occlusions compared to low camera shots. For instance, there are few occlusion scenarios in bird’s-eye-view (high camera elevation) of a scene.

|[Figure 225]|[Figure 226]|
|---|---|

- (a)

- (b)

- (c)

- Figure 13. CLIP filtering on augmentations: We use a depth-to-image model (FLUX.1-Depth-dev [35]) to generate realistic augmentations of the rendered images (a). However, the depth-to-image model occasionally misaligns objects with their intended depth regions, causing incorrect placements. For example, on the left pane, the depth-to-image model incorrectly generates a pigeon in place of a crow according to original layout. We mitigate such issues by applying object-level CLIP-based filtering [56] to retain only those augmentations that adhere to the original layout. For this, we first obtain the object segmentation masks using Blender [12], and use these to obtain cropped object segments in the augmented image (b). Next, we compute CLIP similarity between these object segments and corresponding text description (e.g. cat, pigeon, etc.), as shown in (c). If any object has a CLIP score less than the threshold value of 0.25, the augmentation is filtered out. High CLIP scores for all object segments (as shown on the right pane) indicates accurate layout adherence, and these images are included in the training dataset.

[Figure 227]

- Figure 14. Statistics of training dataset: (a) We plot the distribution of minimum visibility ratio for any object in the scene. Since our filtering strategy favors heavy occlusion scenarios, we observe that there is a bias towards cases with low visibility ratios. (b) Next, we observe that the distribution of orientation values is roughly uniform, thus avoiding any unwanted biases. (c) Interestingly, we observe that the frequency of examples with large 2D bounding box dimension shows a decreasing trend. This is because smaller object sizes enable placement of multiple objects in a scene, while ensuring all of them are visible. (d) High camera shots tend to have weaker occlusions compared to low camera shots; for instance, there are very little inter-object occlusions in bird’s-eye-view of a scene (high camera elevation). Since our data selection process favors high occlusion scenarios, renders with low camera are usually favored by the rendering pipeline algorithm, explaining the observed trend.

[Figure 228]

- Figure 15. Statistics of 3DOcBench evaluation benchmark: Similar to the training dataset, we observe that the 3DOcBench evaluation benchmark contains (a) heavy occlusion scenarios, (b) roughly uniform distribution of orientations, (c) higher frequency of smaller objects, measured using 2D bounding box dimension, and (d) large number of cases with low camera elevation and consequently high inter-object occlusion.

Since our data selection process favors high occlusion scenarios, renders with low camera are favored by the rendering pipeline algorithm, explaining the observed decreasing trend.

###### C. 3DOcBench benchmark details

For constructing our evaluation benchmark, 3DOcBench, we use the same procedural generation to prepare the training dataset (see Sec. B.1). Specifically, we construct scene layouts in Blender [12] with the 3D assets and camera placed in random locations, and filter the layouts based on whether they meet the constraints of occlusion and object size (see Sec. B.1). We present various statistics of the benchmark in Fig. 15. Similar to the training dataset, we observe that the 3DOcBench evaluation benchmark contains (a) heavy occlusion scenarios, (b) roughly uniform distribution of orientations, (c) higher frequency of smaller objects, measured using 2D bounding box dimension, and (d) a large number of cases with low camera elevation and consequently high inter-object occlusion.

###### D. Overlapping regions

In the proposed object binding strategy, the OSCR tokens at the intersection of two rendered bounding boxes attend to all participating object tokens in the text prompt. However, at first glance, it seems that attending to multiple object semantics would lead to semantic bleeding and visual artifacts at object boundaries. Upon investigation, however, we found that the generated images feature sharp occlusion boundaries without object attribute mixing. To understand this, we visualize the attention maps between image and text tokens, and find that object features are segregated in the model’s latent space. We analyse the attention maps through 8 complex scene layouts (of two ob-

jects) with heavy occlusion scenarios in Fig. Fig. 17. As we can see, the attention maps clearly distinguish between the foreground and background objects with appropriate occlusions. This indicates the inherent model’s capability in handling object occlusions, and our method provides a new interface to accurately generate such scenes, which is challenging to do with text alone.

Selecting layers for attention visualization. We performed a simple analysis for choosing the appropriate layers for attention visualization. We use Segment Anything [30] on the generated images followed by manual filtering to segment out individual object regions. Finally, we measure spatial alignment between image to object token attention and ground truth segmentation using correlation coefficient (CC). We analyze CC values across space (DiT layers) and time (denoising timesteps), results are visualized in Fig. 18. The results highlight that spatial alignment is high for very specific layers in the DiT, particularly for layers 11 to 23. Also, spatial alignment tends to emerge at around 5th denoising timestep (out of 25 timesteps). We use the resulting CC matrix to filter top-50 (layer, timestep) combinations which show highest spatial alignment, and average image to text attention for each object. The results are visualized in Fig. 17.

###### E. Implementation details

We build upon FLUX.1-dev [35] as our base model. We patch it with 128-rank LoRA adapters, applied on query, key and value projections in every attention layer. Additionally, we set the LoRA scale to 0 for the text and image tokens to preserve the strong text-to-image prior of the base model [61, 62, 79]. We train our model with a learning rate of 10−4 using the AdamW optimizer for 30K steps with an effective batch size of 2. The first 25K training steps use an image resolution of 512, followed by a resolution of 1024

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

kangaroo__chair__o ffice_chair__wolf/01 2

elephant__jeep__bu llzoder/015

Rendered image Depth flux image.

Rendered image Depth flux image.

’A photo of kangaroo and chair and ofﬁce chair and wolf on a sandy dune near the ocean with gentle waves.’

’A photo of elephant and jeep and bulldozer in a wide open desert with distant mesas and clear air.’

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

wolf__chair__pig__ dog/001

teddy__pigeon__ted dy__teddy/003

Rendered image Depth flux image.

Rendered image Depth flux image.

‘A photo of wolf and chair and pig and dog amongst blooming lavender ﬁelds under a sunny sky.’

‘A photo of teddy and pigeon and teddy and teddy on a kitchen counter.’

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

cat__crow__teddy_ _pigeon/004

jeep__tractor__mot orbike__jeep/019

Rendered image Depth flux image.

Rendered image Depth flux image.

’A photo of cat and crow and teddy and pigeon on a stack of old books in a library.’

’A photo of jeep and tractor and motorbike and jeep in a vibrant city square with historic architecture.’

- Figure 16. Samples from our training dataset: We create scenes in Blender [12] by placing 3D assets in controlled configurations and defining the rendering camera viewpoint. The object arrangements and camera viewpoint are controlled to ensure strong inter-object occlusions, while ensuring that each object is sufficiently visible in the image. Along with the main image, we render the corresponding OSCR representation, which consists of color-coded translucent 3D bounding boxes of the objects. The rendered images are further augmented using a depth-to-image pipeline to obtain realistic images that follow the same layout as shown in Fig:13.

for the next 5K steps. We found that such staged training helps improve realism in the generated images. The complete training takes around 9 hours on 2X NVIDIA H100 GPUs (one image per GPU). Our implementation is based on PyTorch [52] and Hugging Face Diffusers [63] framework.

To enable personalization, we introduce an additional ‘subject’ LoRA of the same rank (128) on the reference image tokens. Both the LoRA’s are finetuned on our personalization dataset (see Sec. I) for 7.5K iterations.

###### F. Taking control with SeeThrough3D

OSCR representation encodes various 3D attributes of a scene, such as object orientation, size and location, along with camera viewpoint as well as occluded object regions. This enables SeeThrough3D to control all the properties in jointly. Additionally, since our method preserves the strong prior of the base text-to-image model, it can generate diverse visual appearance of both objects and background, solely through text prompt control. These diverse forms of control offered by SeeThrough3D are summarized

in Figs. 19 and 20. Note that all the images in these figures are generated using the same random seed, highlighting the effectiveness of control. Notably, the model is able to preserve occlusion consistency even despite heavy overlaps, such as low camera elevation (d1, Figs. 19 and 20), (b4, Fig. 19). These results indicate the preciseness of control enabled by our method, enabling various applications in design and architecture.

###### G. Additional baseline comparisons

In the main paper, we have compared against 3D scene control methods, LooseControl [4] and Build-A-Scene [19]. These methods are directly relevant to ours, since they enable control over 3D scene layout, including object placement, orientation and camera viewpoint. Here we compare against baselines which specifically allow control over 3D object orientation only, without controlling 3D object placement or camera viewpoint. We compare against two baselines, ORIGEN [43] and Compass Control [49]. ORIGEN enables control over object orientation using a one step generative model. Specifically, they perform initial noise op-

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

scene_20251116_2 24405.pkl

scene_20251116_2 22820.pkl

(a)

(b)

‘A photo of sparrow and transparent glass containing water on a breakfast table in a modern house.’

‘A photo of giraffe and sedan under a thunderstorm sky.’

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

scene_20251116_2 15954.pkl

scene_20251117_1 43054.pkl

(c)

(d)

‘A photo of motorbike and bicycle in a modern street, sci-ﬁ, cyberpunk, neon lights, fresh after rain.’

‘A photo of cage and plant in the balcony of a house.’

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

- scene_20251116_2 35310.pkl

- scene_20251117_1 04523.pkl

scene_20251116_2 30420.pkl

(e)

(f)

‘A photo of piano and harp in a magical forest clearing at dusk, with ﬁreﬂies twinkling. ‘A photo of dog and counter stool in a softly lit modern bar.’

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

scene_20251116_2 12759.pkl

(g)

(h)

‘A photo of ﬂask with red chemical and ﬂask with blue chemical in a chemistry lab, sci-ﬁ.’

‘A photo of cow and horse in a green pasture, mountains of Denver in the background.’

- Figure 17. Visualizing object disentanglement in latent space: We use the layouts (shown in first frame) to condition our model, the outputs are shown in second frame. We store the intermediate attention maps from image tokens to object tokens in the text prompt, visualized in third and fourth images. Evidently, the attention maps reveal occlusion boundaries, and show some interesting patterns. Notably, in cases involving transparent objects like water (b) and chemical flask (g), the physically hidden regions of sparrow and flask respectively are visible in attention map. Even in case of semantically similar categories, such as cow and horse (a), flasks with differently colored chemicals (g), motorbike and bicycle (d) the attention is highly localized, with only minimal leakage.

timization according to a reward function which penalizes the mismatch between orientation of the generated object and the input orientation angle. The generated object orientation that is measured using Orient Anything [67]. However, they do not provide control over locations of the objects. Compass Control, on the other hand, enables control over object orientation along with 2D object layouts. They learn an adapter which maps object orientation to a per object compass embedding. These embeddings are then used to condition the generative process through cross attention. The cross attention maps of compass and object tokens in prompt are constrained to respective 2D bounding boxes to enable disentangled orientation control for multi-object scenes.

Analysis: We present comparison results against these baselines in Fig. 21 and Tab. 3. Since ORIGEN [43] does not allow 2D layout control, it is not compatible with our quantitative evaluation that focuses on layout adherence (see Evaluation metrics in the main paper), and we only

present qualitative comparisons. We observe that Compass Control is not able able to handle complex occlusions (A1 − 4), and mixes object attributes in case of heavy overlaps (A5 − 6), resulting in low objectness score. ORIGEN fails to generate some objects in the scene (B1-6). Additionally, its outputs contain artifacts arising from poor noise optimization (B2). Additionally, ORIGEN is limited to onestep generative models, and hence suffers from low image fidelity. In contrast, our method is able to model complex occlusions (E1-6) without attribute mixing, indicating its effectiveness.

###### H. More on angular error evaluation

Two of our baselines, LooseControl [4] and Build-AScene [19] use layout depth maps as condition for text-toimage model. While providing 3D placement cues, the layout depth representation fails to capture precise 3D orientation, leading to poor orientation accuracies, as indicated by high angular error values in Tab. 3. Specifically, we ob-

###### Correlation coefficient across space (layers) and time (timesteps) (Averaged over 8 scenes)

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24

| | | | | | | | | | | | | | | | | | | |[Figure 279]| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

Timestep

0 1 2 3 4 5 6 7 8 9 1011121314151617181920212223242526272829303132333435363738394041424344454647484950515253545556

Layer

|[Figure 280]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.6

0.5

0.4

0.3

0.2

0.1

0.0

- Figure 18. Measuring spatial alignment of image to object attention using correlation coefficient (CC): We create a dataset of 8 complex layouts containing strong occlusion scenarios (see Fig. 17). We obtain SeeThrough3D’s outputs on these layouts, and store intermediate attention maps from image tokens to object tokens in the text prompt. Next, we run Segment Anything [30] on the generated outputs to obtain object-level segmentation masks. Finally, we use correlation coefficient (CC) to measure alignment between the ground truth object segment and corresponding image to object attention maps. We compute the CC across space (layers) and time (denoising timesteps), and the obtained heatmap reveals interesting insights. For a given layer, timestep combination, a high CC value indicates strong spatial awareness. We observe that very specific layers in the DiT are spatially aware; early layers from 8 to 25, after which the spatial awareness decreases sharply. Secondly, the spatial properties in attention emerge after 5th denoising step (out of 25 steps) in these layers. The pattern of spatially aware layers is very irregular, indicating that different layers in the DiT contribute very differently to the generated image, consistent with findings from [1].

serve a large number of 180◦ flips, because bounding box depth does not encode the front-facing direction of the object. Therefore, we evaluate a relaxed angular error which does not penalize 180◦ flips in the generated objects, results tabulated in Tab. 3. We observe that 3D layout control baselines, LooseControl and Build-A-Scene show slightly improved results compared to LaRender [76] and VODiff [37], which are not orientation aware. Compass Control [49] encodes orientation value through an adapter, hence performs better than the other baselines on angular error. In contrast, our OSCR representation explicitly encodes orientation in the image space using color-coding, thus enabling precise orientation control, performing favorably compared to all existing methods.

###### I. Personalization

We show that SeeThrough3D can be finetuned for occlusion-aware 3D control of personalized objects (see Fig. 22). This is achieved by learning a separate ‘subject’ LoRA to fuse appearance attributes from personalized object image into the generation process, building upon prior work on conditioning diffusion transformers [79]. This approach achieves personalized 3D control without need for any test-time tuning. As shown in (a), we can compose objects from multiple modalities, such as dog (text) and royal chair (image). Interestingly, our model can personalize object categories not seen during training, such as bottle and glasses in (c), indicating strong gen-

Baselines depth ord.↑ obj. score↑ angular err.↓ text align.↑ KID(×10−3) ↓ 180◦ flip ang. err.↓ VODiff [37] 0.68 19.70 92.73 29.51 15.40 41.38

LooseControl [4] 0.82 20.02 89.88 28.43 14.32 37.48 Build-A-Scene [19] 0.89 21.0 91.62 28.05 20.12 32.23

LaRender [76] 1.02 21.83 89.63 30.20 13.46 41.19 CompassControl [49] 0.87 20.60 66.29 29.76 13.01 35.79

Ours 1.46 22.86 47.92 31.87 5.43 25.72

Table 3. Quantitative comparison: In the main paper, we did not compare against methods that only enable partial 3D control: Compass Control [49] and ORIGEN [43]. These baselines do not allow for 3D layout control, and primarily focus on object orientation control. While Compass Control allows for 2D layout control, ORIGEN does not, and hence it is not compatible with our quantitative evaluation (see Evaluation metrics in the main paper). Results for Compass Control are presented in yellow. It implicitly encodes orientation using an adapter, hence performs better than the other baselines in angular error. Further, we evaluate a relaxed angular error, which does not penalize 180◦ flips in the generated object (violet column). This caters to layout depth based methods LooseControl [4] and Build-A-Scene [19], which do not encode a front-facing direction for the objects, thus result in such 180◦ flips. Our OSCR representation explicitly encodes orientation in the image space using color-coding, thus enabling precise orientation control, outperforming all baselines.

eralization.

To train the personalization model, we suitably adapt our dataset for this task. Given a rendered image, we randomly choose an object and apply a texture to it in Blender [12] (see Fig. 23(a,b)). For this, we generate a small set of textures using FLUX [35] by prompting it to ensure high frequency details such as text and sharp patterns, some samples are shown in Fig. 24. We separately render the tex-

[Figure 281]

scene_20251119_2 14329.pkl

[Figure 282]

scene_20251119_2 20132.pkl

[Figure 283]

scene_20251119_2 14945.pkl

### 1 2 3 4

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

scene_20251119_2 14439.pkl

scene_20251119_2 15556.pkl

scene_20251119_2 15707.pkl

scene_20251119_2 15817.pkl

- (a) Orientation

- (b) Size

- (c) Location

- (d) Camera

- (e) Text prompt

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

scene_20251119_2 20420.pkl

scene_20251119_2 20029.pkl

scene_20251119_2 14439.pkl

scene_20251119_2 20249.pkl

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

scene_20251119_2 14439.pkl

scene_20251119_2 22759.pkl

scene_20251119_2 23105.pkl

scene_20251119_2 22539.pkl

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

scene_20251119_2 14752.pkl

scene_20251119_2 14439.pkl

scene_20251119_2 24059.pkl

scene_20251119_2 23700.pkl

‘A photo of white Rolls Royce and yellow Ferrari on a rooftop helipad overlooking a futuristic neon skyline, cyber-punk.’

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

scene_2025 1119_23245 5.pkl

[Figure 308]

scene_20251119_2 14439.pkl

scene_20251119_2 31026.pkl

scene_20251119_2 31520.pkl

‘A photo of white Rolls Royce and yellow Ferrari on a rooftop helipad overlooking a futuristic neon skyline, cyber-punk.’

‘A photo of white Rolls Royce toy and yellow Ferrari toy on a study table, gleaming sunshine.’

‘A photo of white Rolls Royce and yellow Ferrari on the Moon, sky full of stars, futuristic.’

‘A photo of white Rolls Royce and yellow Ferrari in a modern car showroom.’

[Figure 309]

scene_20251119_2 14329.pkl

- Figure 19. Taking control with SeeThrough3D: We demonstrate the individual controls that our approach offers over the scene composition, which includes 3D attributes such as (a) object orientation, (b) object size, (c) object location, (d) scene camera elevation, as well as (e) text prompt and object semantics, all while ensuring occlusion consistency. Notably, all the images in this figure were generated using the same random seed, highlighting the effectiveness of control. Disentangled control: In (a), (b) and (c), we are able to control the 3D attributes of one object (Rolls Royce), without altering the other object, indicating disentangled control. Notice how occlusion consistency is preserved even in case of heavy overlap (b4), when the white car has become very big, and in (d1), where the camera elevation is very low. The model is able to model interesting controls such as levitating objects (c4). Despite heavy overlaps, the object attributes (‘white Rolls Royce’, ‘yellow Ferrari’) remain correctly bound to respective objects without attribute mixing, highlighting effectiveness of our binding mechanism.

scene_20251119_1 81456.pkl

#### 1 2 3 4

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

scene_20251119_1 73715.pkl

scene_20251119_1 73457.pkl

scene_20251119_1 73015.pkl

scene_20251119_1 73052.pkl

- (a) Orientation

- (b) Size

- (c) Location

- (d) Camera

- (e) Text prompt

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

scene_20251119_1 73715.pkl

scene_20251119_1 73715.pkl

scene_20251119_1 74755.pkl

scene_20251119_1 74349.pkl

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

scene_20251119_1 73715.pkl

scene_20251119_1 80422.pkl

scene_20251119_1 80305.pkl

scene_20251119_1 80102.pkl

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

scene_20251119_1 81624.pkl

scene_20251119_1 82940.pkl

scene_20251119_1 73715.pkl

scene_20251119_1 82715.pkl

‘A photo of green boat and red boat resting on the edge of a calm forest lake during golden hour.’

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

##### scene_2025 1119_19303 6.pkl

[Figure 335]

scene_20251119_1 73715.pkl

scene_20251119_1 91732.pkl

‘A photo of green boat and red boat resting on the edge of a calm forest lake during golden hour.’

‘A photo of green boat and red boat at a stormy beach, lightning strikes in the sky.’

‘A photo of green paper boat and red paper boat ﬂoating in puddles on road during rainy season.’

‘A photo of green boat and red boat displayed in a cliffside open-air gallery overlooking the sea.

- Figure 20. Taking control with SeeThrough3D: We demonstrate the individual controls that our approach offers over the scene composition, which includes 3D attributes such as (a) object orientation, (b) object size, (c) object location, (d) scene camera elevation, as well as (e) text prompt and object semantics, all while ensuring occlusion consistency. Notably, all images in this figure were generated using the same random seed, highlighting the effectiveness of control. Disentangled control: In (a), (b) and (c), we are able to control the 3D attributes of one object (red boat), without altering the other object, indicating disentangled control. Notice how occlusion consistency is preserved in challenging cases like (d1), where the camera elevation is very low. Despite heavy overlaps, the object attributes (‘green boat‘, ‘red boat‘) remain correctly bound to respective objects without attribute mixing, highlighting effectiveness of our binding mechanism.

[Figure 336]

scene_20251119_1 83311.pkl

scene_20251119_1 73715.pkl

scene_20251119_1 84303.pkl

scene_20251119_1 84050.pkl

[Figure 340]

scene_20251119_1 82121.pkl

[Figure 341]

scene_20251119_1 81456.pkl

scene_20251119_1 80928.pkl

###### 1 2 3 4 5 6

|scene_2 0251119 _130514 .pkl<br><br>[Figure 343]|in a royal palace.<br><br>[Figure 344]|
|---|---|
|scene_2 0251119 _132112 .pkl<br><br>[Figure 345]| |

[Figure 346]

(a)

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

CompassControl

[Figure 356]

###### (A)

in a royal palace.

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

<obj0>

ORIGEN (E)

###### (B)

‘A photo of <obj0> and dog in a royal palace.’

|on a cobblestone street in a beautiful European city.<br><br>[Figure 364]||scene_2 0251118 _225458 .pkl<br><br>[Figure 368]|on a cobblestone street in a beautiful European city.<br><br>[Figure 369]|
|---|---|
| | |
|
|---|---|
| |scene_2 0251118 _232451 .pkl<br><br>[Figure 367]|

|scene_2 0251118 _225458 .pkl<br><br>[Figure 368]|on a cobblestone street in a beautiful European city.<br><br>[Figure 369]|
|---|---|
| | |

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

###### (b)

Ours

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

‘A photo of chair and dog on a wooden pier overlooking a serene lake at sunset.’

‘A photo of hen and cat near a droplet of water on a leaf after rain.’

‘A photo of teddy and crow hiding behind a pebble on a river bank.’

‘A photo of sheep and dog and man and chair near a bird bath in a lush green backyard.’

‘A photo of pickup truck and bulldozer in a bustling constructio n site with heavy equipment.’

‘A photo of table and horse and ferrari on an endless open highway with mountains in the background.’

<obj0> <obj1>

‘A photo of <obj0><obj0> and <obj1><obj1> on a cobblestone street in a beautiful European city.’

|on a modern window sill.<br><br>[Figure 387]|scene_2 0251118 _201714 .pkl<br><br>[Figure 388]|
|---|---|
| |scene_2 0251118 _192147 .pkl<br><br>[Figure 389]|

| |on a modern window sill.<br><br>[Figure 390]|
|---|---|
| | |

###### (c)

[Figure 391]

[Figure 392]

- Figure 21. Qualitative comparisons with additional baselines: We present comparisons with baselines Compass Control [49] and ORIGEN [43]. We observe that Compass Control is not able able to handle complex occlusions (A1-4), and mixes object attributes in case of heavy overlaps (A5-6). ORIGEN fails to generate some objects in the scene (B1-6), and its outputs contain visual artifacts arising from poor noise optimization (B2). Additionally, ORIGEN is limited to one-step generative models, and hence suffers from low image fidelity. In contrast, our method is able to model complex occlusions (E1-6) without attribute mixing, indicating its effectiveness.

<obj0> <obj1>

‘A photo of <obj0><obj0> and <obj1> on a modern window sill.’

Figure 22. Personalization: We show that SeeThrough3D can be finetuned for occlusion-aware 3D control of personalized objects. This is achieved by learning a separate ‘subject’ LoRA to fuse appearance attributes from personalized object image into the generation process, building upon prior work on conditioning diffusion transformers [61, 62, 79]. This approach achieves such personalized 3D control without need for any test-time tuning. As shown in (a), we can compose objects from multiple modalities, such as dog (text) and royal chair (image). Interestingly, our model can personalize object categories not seen during training, such as bottle and glasses in (c), indicating strong generalization.

tured 3D asset, and use it as the reference image condition

- (see Fig. 23(b)). The orientation of the reference object is slightly altered to enable the model to reason about its 3D placement, and not just copy pixels from reference image. Finally, the textured object is placed back into the original image, and used as ground truth target for training the model
- (see Fig. 23(c)), conditioned on the reference image.

###### L. User interface

One of the motives of SeeThrough3D is to enable creative artists to precisely control various 3D elements of a generated image, such as scene layout and camera viewpoint. To ease the design process, we built an intuitive web interface, which allows the user to construct 3D layouts and control camera viewpoint. The interface allows the user to add boxes for various objects, edit their dimensions, 3D placement, and specify a text description for each object. The interface also comes with pre-defined template dimensions of common objects such cars, animals, etc. which can be used. A screenshot of the interface can be seen in Fig. 26.

###### J. Additional qualitative comparisons

We present additional qualitative comparisons with the main baselines in Figs. 27 and 28. Each example has been analyzed with reference to layout adherence and occlusion consistency (red text above each example). Results indicate that SeeThrough3D generates realistic images following precise 3D layouts while maintaining occlusion consistency, and outperforms all baselines.

###### M. Limitations

###### K. Additional qualitative results

We present additional results of our method in Fig. 29. For each example, we have shown the OSCR layout alongside the generated image; the correspondence from boxes to individual objects has been omitted here for clarity.

Since our method conditions a pretrained text-to-image model (FLUX [35]), it is limited by the capabilities of the base model. For instance, FLUX sometimes fails to generate out of distribution cases, such as a parrot behind a birdcage, with realistic occlusion. Consequently, our method, which is built upon the prior of FLUX, also fails to generate

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

(a) original rendered image from training data.

(b) texture applied to object, used as reference image.

(c) textured object placed back, used as target image.

- Figure 23. Adapting the training dataset for personalization: (a) given a rendered image from the training dataset, we randomly choose an object and apply a texture to it in Blender [12] (see Fig. 24 for examples of generated textures). (b) We separately render the textured 3D asset, and use it as reference object condition. (c) Finally, the textured object is placed back into the original image, and used as ground truth target for training the model.

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

- Figure 24. Examples of generated textures: We generate textures using FLUX [35], for preparing data for the personalization task. Notice how the textures contain high frequency features, induced by text and sharp patterns.

[Figure 406]

[Figure 407]

‘A photo of parrot behind a cage in a forest environment.’

FLUX SeeThrough3D

- Figure 25. Limitations: (a) Our model is built upon FLUX [35] which fails to generate some out of distribution cases, such as a parrot outside a cage. Consequently, our model also struggles to generate such cases.

in the transformer’s context; this leads to higher VRAM requirements, especially for multi-subject personalization.

such cases, as shown in Fig. 25. Additionally, personalization requires that all the reference image tokens be present

[Figure 408]

###### Figure 26. We built an intuitive UI to enable the user to easily design layouts for using our model. The UI enables the user to add objects, edit their placement, orientation and dimensions, and provide a text description for each object. Additionally, it allows the user to set the camera parameters.

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours ‘A photo of dog and ofﬁce chair and pig and chair on a wooden pier overlooking a serene lake at sunset.'

|A, B fail to generate all objects, C,D fail to follow precise layouts (man is supposed to be standing behind the chair)|
|---|

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours ‘A photo of kangaroo and man and chair and fox next to a babbling brook in an autumnal woodland.'

|A,B fail to generate all objects, B generates an incoherent scene, C,D show object attribute entanglement and fail to generate occluded objects.|
|---|

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours ‘A photo of vw beetle and mclaren and mclaren and motorbike on a bridge spanning a wide river in a metropolitan area.'

|A, B generate incoherent scenes, C, D fail to control precise orientation.|
|---|

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours 'A photo of elephant and elephant and suv in an expansive national park with majestic red rock formations.'

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours ‘A photo of giraffe and jeep in a mountainous region with winding roads and panoramic views.'

|All baselines fail to generate a small, heavily occluded objects (the tiger). B shows visual artifacts due to inversion, and C has heavy attribute mixing.|
|---|

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours 'A photo of tiger and coupe and van and lamborghini in a vast, empty ﬁeld during a foggy morning.'

|All baselines fail to generate a small, heavily occluded object (the dog behind the man). B shows visual artifacts due to inversion.|
|---|

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours ‘'A photo of wolf and dog and man by a calm river at dawn, reﬂecting the pastel sky.'

|A, B, C generate incorrect layouts and fail to generate some objects with good fidelity. D mixes the objects van and table, leading to incorrect generation.|
|---|

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

(A) LooseControl (B) Build-A-Scene (C) VODiff (D) LaRender (E) Ours 'A photo of lamborghini and table and van traversing a rocky riverbed in a deep canyon under a hot sun.'

[Figure 489]

[Figure 490]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

layout image

layout image

layout image

[Figure 499]

‘A photo of suv and horse and jeep in a large open garage with tools and other vehicles.’

‘A photo of goat and ofﬁce chair and goat and ofﬁce chair on a rustic bridge over a clear mountain stream.’

‘A photo of lamborghini and giraffe in a mountainous region with winding roads and panoramic views.’

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

easy/3doc Subjects_comb img_idx

easy/3doc Subjects_comb img_idx

image

image layout image

‘A photo of hen and teddy and crow and rabbit near a small coin, highlighting its scale.’

‘A photo of pickup truck and motorcycle and horse on an airstrip with a control tower in the background.’

‘A photo of lamborghini and pickup truck and pickup truck and bulldozer on a snow-covered tundra with sparse trees and a frozen lake.’’

‘A photo of lamborghini and lamborghini and bear and bulldozer on a desolate, rocky planet surface with a distant nebula.’

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

easy/3doc Subjects_comb img_idx

easy/3doc Subjects_comb img_idx

[Figure 519]

image layout image

image

‘A photo of cat and cat on a grain of sand on a vast beach.’

‘A photo of pigeon and teddy and cat and crow next to a matchstick, emphasizing its small size.’

‘A photo of pigeon and teddy and shoe on a frosted windowpane, leaving tiny prints.’

‘A photo of scooter and deer on a remote forest trail with ancient trees and thick undergrowth.’

‘A photo of dog and goat and sheep and fox by a calm river at dawn, reﬂecting the pastel sky.’’

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

easy/3doc Subjects_comb img_idx

image

‘A photo of motorbike and deer and jeep on a snow-covered tundra with sparse trees and a frozen lake.’

‘A photo of tiger and elephant and table in a dense jungle with exotic foliage and ﬁltering sunlight.’

‘A photo of sheep and pig and ofﬁce chair and ofﬁce chair at the edge of a cornﬁeld with stalks reaching high.’

Figure 29. Qualitative results: We present additional results of our method. For each example, we have shown the OSCR layout alongside the generated image; the correspondence from boxes to individual objects has been omitted here for clarity.

‘A photo of pig and goat and chair in a small, traditional pottery workshop.’

easy/3doc Subjects_comb img_idx

image layout image

layout image

‘A photo of motorbike and deer and jeep on a snow-covered tundra with sparse trees and a frozen lake.’

‘A photo of tiger and elephant and table in a dense jungle with exotic foliage and ﬁltering sunlight.’

‘A photo of sheep and pig and ofﬁce chair and ofﬁce chair at the edge of a cornﬁeld with stalks reaching high.’

