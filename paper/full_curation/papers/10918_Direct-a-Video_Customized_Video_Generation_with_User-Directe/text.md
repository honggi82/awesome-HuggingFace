## Direct-a-Video: Customized Video Generation with User-Directed Camera Movement and Object Motion

Shiyuan Yang

Liang Hou

Haibin Huang

City University of Hong Kong1 Tianjin University2 1Hong Kong, 2Tianjin, China s.y.yang@my.cityu.edu.hk

Kuaishou Technology Beijing, China houliang06@kuaishou.com

Kuaishou Technology Beijing, China jackiehuanghaibin@gmail.com

Chongyang Ma

Pengfei Wan

Di Zhang

# arXiv:2402.03162v2[cs.CV]6May2024

Kuaishou Technology Beijing, China chongyangm@gmail.com

Kuaishou Technology Beijing, China wanpengfei@kuaishou.com

Kuaishou Technology Beijing, China zhangdi08@kuaishou.com

Jing Liao∗

Xiaodong Chen

City University of Hong Kong Hong Kong, China jingliao@cityu.edu.hk

Tianjin University Tianjin, China xdchen@tju.edu.cn

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Camera movement Object motion Generated Videos

Camera movement Object motion

Generated Videos

|[Figure 7]|[Figure 8]|
|---|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

[Figure 13]

[Figure 14]

+0.5 pan right ×1.8 zoom-in

×0.6 zoom-out

“A wooden house in the snow” “A waterfall in a forest with fall foliage”

|[Figure 15]|[Figure 16]|
|---|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

“A bear walking on grass”

“A leaf is flowing in the sky”

|[Figure 21]|[Figure 22]|
|---|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

“A group of ducks swimming in the lake”

“A zebra and a horse walking on grass”

|[Figure 27]|[Figure 28]|
|---|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

−0.3 pan up

[Figure 33]

[Figure 34]

×1.5 zoom-in

+0.3 pan down

×0.7 zoom-out

“A tiger is walking in the snow”

“A shark and a jellyfish swimming around the coral”

Figure 1: Direct-a-Video is a text-to-video generation framework that allows users to individually or jointly control the camera movement and/or object motion.

### ABSTRACT

to interpret quantitative camera movement parameters. We further employ an augmentation-based approach to train these layers in a self-supervised manner on a small-scale dataset, eliminating the need for explicit motion annotation. Both components operate independently, allowing individual or combined control, and can generalize to open-domain scenarios. Extensive experiments demonstrate the superiority and effectiveness of our method. Project page and code are available at https://direct-a-video.github.io/.

Recent text-to-video diffusion models have achieved impressive progress. In practice, users often desire the ability to control object motion and camera movement independently for customized video creation. However, current methods lack the focus on separately controlling object motion and camera movement in a decoupled manner, which limits the controllability and flexibility of text-tovideo models. In this paper, we introduce Direct-a-Video, a system that allows users to independently specify motions for multiple objects as well as camera’s pan and zoom movements, as if directing a video. We propose a simple yet effective strategy for the decoupled control of object motion and camera movement. Object motion is controlled through spatial cross-attention modulation using the model’s inherent priors, requiring no additional optimization. For camera movement, we introduce new temporal cross-attention layers

### CCS CONCEPTS

• Computing methodologies → Motion processing.

### KEYWORDS

Text-to-video generation, motion control, diffusion model.

∗Corresponding author.

### 1 INTRODUCTION

Text-to-image (T2I) diffusion models have already demonstrated astonishingly high quality and diversity in image generation and editing [Ho et al. 2020; Ramesh et al. 2022; Rombach et al. 2022; Saharia et al. 2022; Wang et al. 2023a; Yang et al. 2020, 2023a; Yuan et al. 2024]. The rapid development of T2I diffusion models has also spurred the recent emergence of text-to-video (T2V) diffusion models [Blattmann et al. 2023a,b; Ho et al. 2022a; Singer et al.

- 2022; Wang et al. 2023d], which are normally extended from pretrained T2I models for video generation and editing. On the other hand, the advent of controllable techniques in T2I models, such as ControlNet [Zhang et al. 2023], T2I-adapter [Mou et al. 2023] and GLIGEN [Li et al. 2023], has allowed users to specify the spatial layout of generated images through conditions like sketch maps, depth maps, or bounding boxes etc., significantly enhancing the spatial controllability of T2I models. Such spatial controllable techniques have also been successfully extended to spatial-temporal control for video generation. One of the representative works in this area is VideoComposer [Wang et al. 2023f], which can synthesize a video given a sequence of sketch or motion vector maps.

Despite the success of video synthesis, current T2V methods often lack support for user-defined and disentangled control over camera movement and object motion, which limits the flexibility in video motion control. In a video, both objects and the camera exhibit their respective motions. Object motion originates from the subject’s activity, while camera movement influences the transition between frames. The overall video motion becomes well-defined only when both camera movement and object motion are determined. For example, focusing solely on object motion, such as generating a video clip where an object moves to the right within the frame, can lead to multiple scenarios. The camera may remain stationary while the object itself moves right, or the object may be stationary while the camera moves left, or both the object and the camera may be moving at different speeds. This ambiguity in the overall video motion can arise. Therefore, the decoupling and independent control of camera movement and object motion not only provide more flexibility but also reduce ambiguity in the video generation process. However, this aspect has received limited research attention thus far.

To control camera movement and object motion in T2V generation, a straightforward approach would be to follow the supervised training route similar to works like VideoComposer [Wang et al. 2023f]. Following such kind of scheme involves training a conditional T2V model using videos annotated with both camera and object motion information. However, this would bring the following challenges: (1) In many video clips, object motion is often coupled with camera movements due to their inherent correlation. For example, when a foreground object moves to some direction, the camera typically pans in the same direction due to the preference to keep the main subject at the center of the frame. Training on such coupled camera and object motion data makes it difficult for the model to distinguish between camera movements and and object motion. (2) Obtaining large-scale video datasets with complete camera movement and object motion annotations is challenging due to the laborious and costly nature of performing frame-by-frame

object tracking and camera pose estimation. Additionally, training a video model on a large-scale dataset can be computationally expensive.

In this work, we introduce Direct-a-Video, a text-to-video framework that enables users to independently specify the camera’s pan and zoom movements and the motions of scene objects, allowing them to create their desired motion pattern as if they were directing a video (Figure 1). To achieve this, we propose a strategy for decoupling camera and object control by employing two orthogonal controlling mechanisms. In essence, we learn the camera movement through a self-supervised and lightweight training approach. Conversely, during inference, we adopt a training-free method to control object motion. Our strategy avoids the need for intensive collection of motion annotations and video grounding datasets.

In camera movement control, we train an additional module to learn the frame transitions. Specifically, we introduce new temporal cross-attention layers, known as the camera module, which functions similarly to spatial cross-attention in interpreting textual language. This camera module interprets “camera language”, specifically camera panning and zooming parameters, enabling precise control over camera movement. However, acquiring datasets with camera movement annotations can pose a challenge. To overcome this laborious task, we employ a self-supervised training strategy that relies on camera movement augmentation. This approach eliminates the need for explicit motion annotations. Importantly, we train these new layers while preserving the original model weights, ensuring that the extensive prior knowledge embedded within the T2V model remains intact. Although the model is initially trained on a small-scale video dataset, it acquires the capability to quantitatively control camera movement in diverse, open-domain scenarios.

In object motion control, a significant challenge arises from the availability of well-annotated grounding datasets for videos, curating such datasets is often a labor-intensive process. To bypass these issues, we draw inspiration from previous attention-based imagelayout control techniques in T2I models [Hertz et al. 2022; Kim et al. 2023]. We utilize the internal priors of the T2V model through spatial cross-attention modulation, which is a training-free approach, thereby eliminating the need for collecting grounding datasets and annotations for object motion. To facilitate user interaction, we enable users to specify the spatial-temporal trajectories of objects by drawing bounding boxes at the first and last frames, as well as the intermediate path. Such interaction is simpler and more userfriendly compared to previous pixel-wise control methods [Wang et al. 2023f].

Given that our approach independently controls camera movement and object motion, thereby effectively decouples the two, offering users enhanced flexibility to individually or simultaneously manipulate these aspects in video creation.

In summary, our contributions are as follows:

- • We propose a unified framework for controllable video generation that decouples camera movement and object motion, allowing users to independently or jointly control both aspects.
- • For camera movement, we introduce a novel temporal crossattention module dedicated to camera movement conditioning. This camera module is trained through self-supervision, enabling

users to quantitatively specify the camera’s horizontal and vertical panning speeds, as well as its zooming ratio.

• For objectmotion,weutilize atraining-free spatial cross-attention modulation, enabling users to easily define the motion trajectories for one or more objects by drawing bounding boxes.

- 2 RELATED WORK

- 2.1 Text-to-Video Generation

The success of text-to-image (T2I) models has revealed their potential for text-to-video (T2V) generation. T2V models are often evolved from T2I models by incorporating temporal layers. Early T2V models [Ho et al. 2022a,b; Singer et al. 2022] perform the diffusion process in pixel space, which requires multiple cascaded models to generate high-resolution or longer videos, resulting in high computational complexity. Recent T2V models draw inspiration from latent diffusion [Rombach et al. 2022] and operate in a lower-dimensional and more compact latent space [Blattmann et al. 2023b; Esser et al. 2023; Guo et al. 2023; Wang et al. 2023d; Zhou

- et al. 2022]. The most recent Stable Video Diffusion [Blattmann et al.

- 2023a] utilizes curated training data and is capable of generating high-quality videos.

On the other hand, the development of T2I editing techniques [Gal et al. 2022; Hertz et al. 2022; Kumari et al. 2023; Mokady et al. 2023; Ruiz et al. 2023] has facilitated zero/few-shot video editing tasks. These techniques convert a given source video to a target video through approaches such as weight fine-tuning [Wu et al.

- 2023b], dense map conditioning [Esser et al. 2023; Geyer et al. 2023; Yang et al. 2023b; Zhao et al. 2023b], sparse point conditioning [Gu et al. 2023; Tang et al. 2023], attention feature editing [Ceylan et al. 2023; Liu et al. 2023b; Qi et al. 2023; Wang et al. 2023c], and canonical space processing [Chai et al. 2023; Kasten et al. 2021; Ouyang et al. 2023]. Some works specifically focus on synthesizing human dance videos using source skeleton sequences and reference portraits [Chang et al. 2023; Feng et al. 2023; Hu et al. 2023; Wang et al. 2023b; Xu et al. 2023], which have yielded impressive results.

- 2.2 Video Generation with Controllable Motion

As motion is an important factor in video, research on video generation with motion control has garnered increasing attention. We can categorize the works in this field into three groups based on the type of input media: image-to-video, video-to-video, and text-to-video.

Image-to-video. Some methods focus on transforming static images into videos, and a popular approach for motion control is through key point dragging [Chen et al. 2023a; Deng et al. 2023; Yin et al. 2023]. While this interaction method is intuitive and userfriendly, it has limitations due to the local and sparse nature of the key points. Consequently, its capacity for controlling motion at a large granularity is significantly restricted.

Video-to-video. These works primarily focus on motion transfer, which involves learning a specific subject action from source videos and applying it to target videos using various techniques, including fine-tuning the model on a set of reference videos with similar motion patterns [Jeong et al. 2023; Wei et al. 2023; Wu et al. 2023a; Zhao et al. 2023a], or borrowing spatial features (e.g., sketch, depth maps) [Chen et al. 2023b; Wang et al. 2023f] or sparse features (e.g.,

DIFT point embedding) [Gu et al. 2023] from source videos. These methods highly rely on the motion priors from the source videos, which, however, are not always practically available.

Text-to-video. In the case where the source video is unavailable, generating videos from text with controllable motion is a meaningful but relatively less explored task. Our work focuses on this category. Existing approaches in this category include AnimateDiff [Guo et al. 2023], which utilizes ad-hoc motion LoRA modules [Hu et al. 2021] to enable specific camera movements. However, it lacks quantitative camera control and also does not support object motion control. VideoComposer [Wang et al. 2023f] provides global motion guidance by conditioning on pixel-wise motion vectors. However, the dense control manner offered by VideoComposer is inefficient to use and does not explicitly separate camera and object motion, resulting in inconvenient user interaction. A concurrent work, Peekaboo [Jain et al. 2023], also uses bounding boxes to control the trajectory of the object through attention masking. However, their method originally does not consider multi-object scenarios and also does not support control over camera movement, unlike our approach. MotionCtrl [Wang et al. 2023e], another concurrent work, allows for separate 2D point-driven object motion control and 3D trajectory-driven camera control by training camera and object control modules. However, its training preparation is labor-intensive, requiring the extraction of motion trajectories on large-scale video dataset. Moreover, it struggles to control multiple different objects with varied motion directions, as it lacks the explicit binding between objects and their motion trajectories during the training. In contrast, our self-supervised training scheme does not require any motion annotations and can achieve motion control over camera and multiple objects, bringing more flexibility for video synthesis.

### 3 METHOD 3.1 Overview

Task formulation. In this paper, we focus on text-to-video generation with user-directed camera movement and/or object motion. First of all, user should provide a text prompt which may optionally contain one or more object words 𝑂1,𝑂2, ...𝑂𝑁 . To determine the camera movement, user can specify an x-pan ratio 𝑐𝑥, a y-pan ratio 𝑐𝑦, and a zoom ratio 𝑐𝑧. To determine the motion of 𝑛-th object 𝑂𝑛 , user needs to specify a starting box B𝑛1, an ending box B𝑛𝐿 (𝐿 is the video length), and an intermediate track 𝜁𝑛 connecting B𝑛1 and B𝑛𝐿, our system then generates a sequence of boxes [B𝑛1, ..., B𝑛𝐿] centered along the track 𝜁𝑛 via interpolation to define the spatial-temporal journey of the object. Consequently, our model synthesizes a video that adheres to the prescribed camera movement and/or object motion, creating customized and dynamic visual narrative.

Overall pipeline. Our overall pipeline is illustrated in Figure 2. The camera movement is learned in the training stage and the object motion is implemented in the inference stage. During the training, we use video samples captured by a stationary camera, which are then augmented to simulate camera movement according to [𝑐𝑥,𝑐𝑦,𝑐𝑧]. The augmented videos are subsequently used as input to the U-Net. Additionally, the camera parameters are also encoded and injected into a newly introduced trainable temporal

#### Inference

#### Training

###### User inputs

###### Camera module

𝑐 : y-pan ratio

[Figure 35]

🔥

[Figure 36]

[Figure 37]

[Figure 38]

V

[Figure 39]

Camera embedder

V

Tempcross-attn

[Figure 40]

[Figure 41]

[Figure 42]

[𝑐 ,𝑐 ,𝑐 ]

𝑐 : x-pan ratio

𝒆𝒙𝒚 𝒆𝒛

×𝑇 timesteps

tanh(𝛼)

Camera embeds

[Figure 43]

KK🔥

𝑐 : zoom ratio

[Figure 44]

[Figure 45]

[Figure 46]

- 1

- 2

… 𝐿

- 3

Camera movement parameters

Tempself-attn

Video sample

Camera movement parameters

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Q🔥

Decode

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

###### 🔥

“A tiger and a bear walking on grass”

Camera embedder

Camera augmenter

𝝐

Frame features (𝐵×𝐿×𝐶)

Camera embeds (1×2×𝐶)

Text encode

[Figure 58]

[Figure 59]

[Figure 60]

Text prompt

[Figure 61]

[Figure 62]

[Figure 63]

Camera embeds

Text embeds

− +

[Figure 64]

[Figure 65]

|Spatial self-attention Spatial cross-attention Temporal self-attention<br><br>[Figure 66]<br><br>🔥 Trainable layers<br><br>Temporal cross-attention|
|---|

a tiger and a bear walkingon grass

[Figure 67]

||tiger|
|---|
<br><br>|bear|
|---|
<br><br>grass|
|---|

[Figure 68]

[Figure 69]

tiger bear grass

𝝐

Box interpolation

[Figure 70]

[Figure 71]

[Figure 72]

###### 🔥 🔥

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

🔥

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

###### … … …

Encode

[Figure 92]

Augmented video

Box sequence

<others>

Object motion canvas

Token-wise spatial cross-attn maps

Modulation terms

“A dog laying on the ground”

ℒ =|| 𝝐 − 𝝐 ||

Text encode

Text embeds

- Figure 2: The overall pipeline of Direct-a-Video. The camera movement is learned in the training stage and the object motion is implemented in the inference stage. Left: During training, we apply augmentation to video samples to simulate camera movement using panning and zooming parameters. These parameters are embedded and injected into newly introduced temporal cross-attention layers as the camera movement conditioning, eliminating the need for camera movement annotation. Right: During inference, along with camera movement, user inputs a text prompt containing object words and associated box trajectories. We use spatial cross-attention modulation to guide the spatial-temporal placement of objects, all without additional optimization. Note that our approach, by independently controlling camera movement and object motion, effectively decouples the two, thereby enabling both individual and joint control.

cross-attention layer to condition the camera movement (detailed in Section 3.2). During the inference, with trained camera embedder and module, users can specify the camera parameters to control its movement. Concurrently, we incorporate the object motion control in a training-free manner: given the object words from the user’s prompt and the corresponding boxes, we modulate the frame-wise and object-wise spatial cross-attention maps to redirect the object spatial-temporal size and location (detailed in Section 3.3). It is noteworthy that the modulation in inference stage does not involve additional optimization, thus the incremental time and memory cost is negligible.

right shift). Similarly, 𝑐𝑦 is the y-pan ratio, representing the total yshift of the frame center over the frame height, 𝑐𝑦 > 0 for panning downward. 𝑐𝑧 denotes the zoom ratio, defined as the scaling ratio of the last frame relative to the first frame, 𝑐𝑧 > 1 for zooming-in. We set the range of 𝑐𝑥,𝑐𝑦 to [−1, 1] and 𝑐𝑧 to [0.5, 2], which are generally sufficient for covering regular camera movement range.

In practice, for given ccam, we simulate camera movement by applying shifting and scaling to the cropping window on videos captured with a stationary camera. This data augmentation exploits readily available datasets like MovieShot [Rao et al. 2020]. Further details of this process, including pseudo code and sampling scheme of ccam are provided in the supplemental material.

- 3.2 Camera Movement Control We choose three types of camera movement: horizontal pan, vertical

Camera embedding. To encode ccam into a camera embedding, we use a camera embedder that includes a Fourier embedder, which is widely used for encoding coordinate-like data [Mildenhall et al. 2021], and two MLPs. One MLP jointly encodes the panning movement 𝑐𝑥, 𝑐𝑦, while the other encodes the zooming movement 𝑐𝑧. We empirically found that separately encoding panning and zooming helps the model distinguish between these two distinct types of camera movements effectively, and we validate this design in Section 4.5. The embedding process can be formulated as e𝑥𝑦 = MLP𝑥𝑦(F ([𝑐𝑥,𝑐𝑦])) ,e𝑧 = MLP𝑧(F (𝑐𝑧)),where F denotesFourier embedder. Both e𝑥𝑦 and e𝑧 have the same feature dimensions, By concatenating them, we obtain the camera embedding ecam = [e𝑥𝑦, e𝑧], which has a sequence length of two.

pan, and zoom, parameterized as a triplet ccam = [𝑐𝑥,𝑐𝑦,𝑐𝑧] to serve as the camera control signal. This allows for quantitative control, a feature not available in previous work [Guo et al. 2023], and is simple to use and sufficiently expressive for our needs.

Data construction and augmentation. Extracting camera movement information from existing video can be computationally expensive since the object motion needs to be identified and filtered out. As such, we propose a self-supervised training approach using camera augmentation driven by ccam, thereby bypassing the need for intensive movement annotation.

We first formally define the camera movement parameters. 𝑐𝑥 represents the x-pan ratio, and is defined as the total x-shift of the frame center from the first to the last frame relative to the frame width, 𝑐𝑥 > 0 for panning rightward (e.g., 𝑐𝑥 = 0.5 for a half-width

Camera module. We now consider where to inject the camera embedding. Previous studies have highlighted the role of temporal layers in managing temporal transitions [Guo et al. 2023; Zhao

- et al. 2023a]. As such, we inject camera control signals via temporal layers. Inspired by the way spatial cross-attention interprets textual information, we introduce new trainable temporal cross-attention layers specifically for interpreting camera information, dubbed as camera modules, which are appended after the existing temporal self-attention layers within each U-Net block of the T2V model, as depicted by the orange box in Figure 2. Similar to textual crossattention, in this module, the queries are mapped from visual frame features F, we separately map the keys and values from panning

embedding e𝑥𝑦 and zooming embedding e𝑧 for the same reason stated in the previous section. Through temporal cross-attention, the camera movement is infused into the visual features, which is then added back as a gated residual. We formulate this process as follows:

F = F + tanh(𝛼) · TempCrossAttn(F, ecam) (1)

TempCrossAttn(F, ecam) = Softmax

Q[K𝑥𝑦, K𝑧]𝑇 √

𝑑

[V𝑥𝑦, V𝑧],

(2) where [, ] denotes concatenation in sequence dimension, K𝑥𝑦,K𝑧 are key vectors, V𝑥𝑦,V𝑧 are value vectors mapped from the e𝑥𝑦, e𝑧 respectively, 𝑑 is the feature dimension of Q, and 𝛼 is a learnable scalar initialized as 0, ensuring that the camera movement is gradually learned from the pretrained state.

To learn camera movement while preserving the model’s prior knowledge, we freeze the original weights and train only the newly added camera embedder and camera module. These are conditioned on camera movement ccam, and video caption 𝑐txt. The training employs the diffusion noise-prediction loss:

L = Ex0,ccam,𝑐txt,𝑡,𝝐∼N(0,𝐼) ∥𝝐 − 𝝐𝜃 (x𝑡, ccam,𝑐txt,𝑡)∥22 , (3)

where x0 is the augmented input sample, 𝑡 denotes the diffusion timestep, x𝑡 = 𝛼𝑡x0 + 𝜎𝑡𝝐 is the noised sample at 𝑡, 𝛼𝑡 and 𝜎𝑡 are time-dependent DDPM hyper-parameters [Ho et al. 2020], 𝝐𝜃 is the diffusion model parameterized by 𝜃.

- 3.3 Object Motion Control

We choose the bounding box as the control signal for object motion as it aligns best with our method, i.e., modulating attention values within regions defined by boxes. Additionally, boxes are more efficient than dense conditions (e.g., sketch maps require drawing skills) and are more expressive than sparse conditions (e.g., key points lack the specification for object’s size).

While it is theoretically possible to train a box-conditioned T2V model similar to GLIGEN [Li et al. 2023]. However, unlike images, well-annotated video grounding datasets are less accessible, curating and training on large-scale dataset can be labor-intensive and computationally expensive. To bypass this issue, we opt to fully leverage the inherent priors of pretrained T2V models by steering the diffusion process to our desired result. Previous T2I works have demonstrated the ability to control an object’s spatial position by editing cross-attention maps [Balaji et al. 2022; Chen et al. 2024; Hertz et al. 2022; Kim et al. 2023; Ma et al. 2023; Sarukkai et al.

- 2024]. Similarly, we employ the spatial cross-attention modulation in T2V model for object motion crafting.

In cross-attention layers, the query features Q are derived from visual tokens, the key K and value features V are mapped from textual tokens. QK⊤ constitutes an attention map, where the value at index [𝑖, 𝑗] reflects the response of the i-th image token feature to the j-th textual token feature. We modulate the attention map QK⊤ as follows:

CrossAttnModulate(Q, K, V) = Softmax

###### QK⊤ + 𝜆S

√

𝑑

V, (4)

where 𝜆 represents modulation strength, 𝑑 is the feature dimension of Q, and S is the modulation term of the same size as QK⊤. It comprises two types of modulation: amplification and suppression.

Attention amplification. Considering the 𝑛-th object in the 𝑘-th frame, enclosed by the bounding box B𝑘𝑛, since we aim to increase the probability of the object’s presence in this region, we could amplify the attention values for the corresponding object words (indexed as T𝑛 in the prompt) within the area B𝑘𝑛. Note that if there exists a background word, we treat it in the same way, and its region is the complement of the union of all the objects’ regions. Following the conclusion from DenseDiff [Kim et al. 2023], the scale of this amplification should be inversely related to the area of B𝑘𝑛, i.e., smaller box area are subject to a larger increase in attention. Since our attention amplification is performed on box-shaped regions, which does not align with the object’s natural contours, we confine the amplification to the early stages (for timesteps𝑡 ≥ 𝜏,𝜏 is the cutoff timestep), as the early stage mainly focuses on generating coarse layouts. For 𝑡 < 𝜏, we relax this control to enable the diffusion process to gradually refine the shape and appearance details.

Attention suppression. To mitigate the influence of irrelevant words on the specified region and prevent the unintended dispersion of object features to other areas, we suppress attention values for unmatched query-key token pairs (except start token <sos> and end token <eos> otherwise the video quality would be compromised). Different from attention amplification, attention suppression is applied throughout the entire sampling process to prevent mutual semantic interference, an potential issue in multi-object generation scenarios where the semantics of one object might inadvertently bleed into another. We will present the results and analysis in the ablation studies (Section 4.5).

Formally, the attention modulation term for the 𝑛-th object in the 𝑘-th frame S𝑘𝑛[𝑖, 𝑗] is formulated as:



𝑘 𝑛|

1 − |B

|QK⊤|, if 𝑖 ∈ B𝑘𝑛 and 𝑗 ∈ T𝑛 and 𝑡 ≥ 𝜏 0, if 𝑖 ∈ B𝑘𝑛 and 𝑗 ∈ T𝑛 and 𝑡 < 𝜏 −∞, otherwise

(5)

S𝑘𝑛[𝑖, 𝑗] =

 

where |X| denotes the number of elements in matrix X. We perform such modulation for each object in every frame so that the complete spatial-temporal object trajectory can be determined. Note that although this modulation is independently performed in each frame, we observe that the generated videos remain continuous, thanks to the pretrained temporal layers which maintains temporal continuity.

- 4 EXPERIMENT

- 4.1 Experimental Setup

Implementation details. We adopt pretrained Zeroscope T2V model [Wang et al. 2023d] as our base model, integrating our proposed trainable camera embedder and module to facilitate camera movement learning, please refer to supplementary materials for training details. During the inference, we use DDIM sampler [Song et al. 2020] with𝑇 = 50 sampling steps and a classifier-free guidance scale of 9 [Ho and Salimans 2022]. The default attention control weight 𝜆 and cut-off timestep 𝜏 are 25 and 0.95𝑇 respectively. The output video size is 320×512×24.

Datasets. For camera movement training, we use a subset from MovieShot [Rao et al. 2020], which contains 22k static-shot movie trailers, i.e., the camera is fixed but the subject is flexible to move, ensuring that the training samples are devoid of original camera movement. Despite the limited number and category of the training data, our trained camera module is still able to adapt to general scenes. For camera control evaluation, we collected 200 scene prompts from the prompt set provided by [Chivileva et al. 2023]. For object control evaluation, we curated a benchmark of 200 box-prompt pairs, comprising varied box sizes, locations, and trajectories, with prompts primarily focusing on natural animals and objects.

Metrics. (1) To assess video generation quality, we employ FIDvid [Heusel et al. 2017] and FVD [Unterthiner et al. 2018]. The reference set consist of 2048 videos from MSRVTT [Xu et al. 2016] for the camera control task and 800 videos from AnimalKingdom [Ng et al. 2022] for the object control task. (2) To evaluate camera movement control, we introduce the flow error metric. We utilize VideoFlow [Shi et al. 2023], a state-of-the-art optical flow model, to extract flow maps from the generated videos. These are then compared against the ground truth flow maps, which are derived from the given camera movement parameters. (3) To measure the object-prompt alignment in object control task, we uniformly extract 8 frames per video sample and calculate the CLIP image-text similarity (CLIP-sim) [Hessel et al. 2021] within the box area, with a templated prompt “a photo of <obj>”, where <obj> corresponds to the object phrase. (4) To measure the object-box alignment, we employ Grounding DINO [Liu et al. 2023a] to detect object boxes in generated videos. We then calculate the mean Intersection over Union (mIoU) against the input boxes and compute the average precision score at the 0.5 IoU threshold (AP50).

Baselines. We compare our method with recent diffusion-based T2V models with the camera movement or object motion controllability, including AnimateDiff [Guo et al. 2023] (for camera movement), Peekaboo [Jain et al. 2023] (for object motion), and VideoComposer [Wang et al. 2023f] (for both).

- 4.2 Camera Movement Control

For camera movement control, we conduct comparisons with AnimateDiff and VideoComposer. For AnimateDiff, we use official pretrained LoRA motion modules, each dedicated to a specific type of camera movement but lacking support for precise control. For VideoComposer, we hand-craft a motion vector map based on the camera movement parameters, as demonstrated in its paper.

##### Table 1: Quantitative comparison for camera movement control evaluation.

FVD ↓ FID-vid↓ Flow error↓

AnimateDiff 1685.40 82.57 VideoComposer 1230.57 82.14 0.74 Direct-a-Video (ours) 888.91 48.96 0.46

Qualitative comparison. We present side-by-side visual comparison with baselines in Figure 3. As can be seen, all the methods are capable of generating videos with the single type of camera movement, but AnimateDiff does not support hybrid camera movement (e.g., pan+zoom) since its loaded motion module is dedicated to one type of camera movement only, while our method and VideoComposer can combine or switch the camera movement by altering the motion input, without the need for re-loading extra modules. In terms of precise control, both our method and VideoComposer can quantitatively control the camera speed. Specifically, VideoComposer [Wang et al. 2023f] requires a sequence of motion maps as input, while ours only requires three camera parameters. Moreover, in terms of disentanglement, our method’s camera control does not impact foreground objects, as we do not impose any motion constraints on them. In contrast, VideoComposer employs a global motion vector map, which often binds objects together with background movement. As shown in the 3rd column of Figure 3, the zebra in our results exhibits its independent motion from the camera, whereas in VideoComposer’s results (the 2nd column), the zebra is tied to the camera movement, so does the fish in the last 2nd column. Finally, our results also exhibit higher visual quality, a testament to the superiority of our base model.

Quantitative comparison. We report FVD, FID-vid, and Flow error in Table 1. Note that AnimateDiff is excluded from the flow error comparison due to its lack of quantitative control. Our results achieve the best FVD and FID-vid scores, indicating superior visual quality compared to baselines, and show more precise camera control, evidenced by a lower flow error.

### 4.3 Object Motion Control

For object motion control, we compare with VideoComposer [Wang et al. 2023f] and Peekaboo [Jain et al. 2023]. To enable VideoComposer generating object motion via boxes, we first convert object box sequences into dense flow maps, which are then processed into motion vector maps compatible with its input format. Peekaboo’s visual results are taken from their official website.

Qualitative comparison. We present visual comparison with related baselines in Figure 4. For static object generation, VideoComposer fails to generate the object in desired location (see the panda in the first column), without any motion hint, it works like a vanilla T2V model. While all methods are capable of generating a single moving object, challenges arise in multiple moving objects scenarios. Peekaboo is excluded from this comparison as its code is not implemented for multiple objects. VideoComposer does not support specifying individual motion for each object unlike our method (see the shark and jellyfish examples in the 7th and 8th

###### Pan up Zoom in

Pan right + Zoom out

###### Pan left

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Clown fish swimming in a coral reef

A zebra next to a river

A serene of Japanese garden

Sly raccoon smirks

|[Figure 97]|
|---|
|[Figure 98]|
|[Figure 99]|
|[Figure 100]|

|[Figure 101]|
|---|
|[Figure 102]|
|[Figure 103]|
|[Figure 104]|

|[Figure 105]|
|---|
|[Figure 106]|
|[Figure 107]|
|[Figure 108]|

|[Figure 109]|
|---|
|[Figure 110]|
|[Figure 111]|
|[Figure 112]|

|[Figure 113]|
|---|
|[Figure 114]|
|[Figure 115]|
|[Figure 116]|

|[Figure 117]|
|---|
|[Figure 118]|
|[Figure 119]|
|[Figure 120]|

|[Figure 121]|
|---|
|[Figure 122]|
|[Figure 123]|
|[Figure 124]|

|[Figure 125]|
|---|
|[Figure 126]|
|[Figure 127]|
|[Figure 128]|

|[Figure 129]|
|---|
|[Figure 130]|
|[Figure 131]|
|[Figure 132]|

|[Figure 133]|
|---|
|[Figure 134]|
|[Figure 135]|
|[Figure 136]|

|[Figure 137]|
|---|
|[Figure 138]|
|[Figure 139]|
|[Figure 140]|

AnimateDiff VideoComposer Ours (𝑐 :+0.2) AnimateDiff VideoComposer Ours (𝑐 : ×1.5)

VideoComposer Ours (𝑐 :+0.5,𝑐 : ×0.6)

Ours (𝑐 :−0.3)

AnimateDiff VideoComposer

##### Figure 3: Qualitative comparison on camera movement control with related baselines. Our results in the third column show that the object motion (yellow lines) can be independent from the camera movement (cyan lines) unlike results by VideoComposer [Wang et al. 2023f] in the second column.

Single moving object Static + moving objects Multiple moving objects

Static object

“A shark and a jellyfish “A panda eating bamboo on some rocks” “A horse walking on grassland” swimming in the sea”

“A tiger and a bear walking on grass”“A jeep driving by a camel in desert”

|[Figure 141]|
|---|
|[Figure 142]|
|[Figure 143]|
|[Figure 144]|

|[Figure 145]|
|---|
|[Figure 146]|
|[Figure 147]|
|[Figure 148]|

|[Figure 149]|
|---|
|[Figure 150]|
|[Figure 151]|
|[Figure 152]|

|[Figure 153]|
|---|
|[Figure 154]|
|[Figure 155]|
|[Figure 156]|

|[Figure 157]|
|---|
|[Figure 158]|
|[Figure 159]|
|[Figure 160]|

|[Figure 161]|
|---|
|[Figure 162]|
|[Figure 163]|
|[Figure 164]|

|[Figure 165]|
|---|
|[Figure 166]|
|[Figure 167]|
|[Figure 168]|

|[Figure 169]|
|---|
|[Figure 170]|
|[Figure 171]|
|[Figure 172]|

|[Figure 173]|
|---|
|[Figure 174]|
|[Figure 175]|
|[Figure 176]|

|[Figure 177]|
|---|
|[Figure 178]|
|[Figure 179]|
|[Figure 180]|

|[Figure 181]|
|---|
|[Figure 182]|
|[Figure 183]|
|[Figure 184]|

|[Figure 185]|
|---|
|[Figure 186]|
|[Figure 187]|
|[Figure 188]|

VideoComposer Peekaboo Ours VideoComposer Peekaboo Ours

VideoComposer Ours VideoComposer Ours VideoComposer Ours

##### Figure 4: Qualitative comparison on object motion control with related baselines. Our method excels in handling cases involving more than one object.

columns). Moreover, its lack of explicit binding between objects and motion leads to two extra issues: semantic mixing and absence. Semantic mixing refers to the blending of one object’s semantics with another. This is exemplified in the 9th column, where tiger’s texture leaks into bear. Semantic absence occurs when an object does not appear as anticipated, a known issue in T2I/T2V models [Chefer et al. 2023]. For instance, in the 11th column, the expected camel is missing, replaced instead by a jeep. In contrast, our method effectively addresses these issues through ad-hoc attention modulation for each object, facilitating easier control over multiple objects’ motion.

Quantitative comparison. We report quality metrics (FVD, FIDvid) and grounding metrics (CLIP-sim, mIoU, AP50) in Table 2. In terms of quality, our method is comparable to Peekaboo, as both utilize the same superior model that outperforms VideoComposer’s. For object control, our method slightly surpasses VideoComposer and significantly exceeds Peekaboo by additionally incorporating attention amplification, in contrast to Peekaboo’s reliance on attention masking alone. We believe the use of amplification plays important role in improving grounding ability, as demonstrated in our ablation study (Section 4.5).

##### Table 2: Quantitative comparison for object motion control evaluation.

FVD ↓ FID-vid↓ CLIP-sim↑ mIoU (%) ↑ AP50 (%)↑

VideoComposer 1620.83 90.57 27.35 45.24 31.01 Peekaboo 1384.62 44.49 27.03 36.55 18.77 Direct-a-Video 1300.86 43.55 27.63 47.83 31.33

### 4.4 Joint Control of Camera Movement and Object Motion

Direct-a-Video features in jointly supporting the control of both camera movement and object motion, we demonstrate such capability in Figure 5. Given the same box sequence, our method can generate videos with varied combination of foreground-background motions. For example, Figure 5(a) illustrates that a static box does not always imply a static object, by setting different camera movements, our system can generate videos of a zebra standing still (2nd column), walking right (3rd column), or walking left (4th column). Similarly, Figure 5(b) suggests that a moving box does not necessarily indicate that the object itself is in motion, it could be stationary in its position while the camera is moving (last column). Existing works focused only on object often fail to differentiate between the object’s inherent motion and apparent motion induced by camera movement. In contrast, our method enables users to distinctly specify both camera movement and object motion, offering enhanced flexibility in defining overall motion patterns. More examples are provided in Figure 11 and our project page.

### 4.5 Ablation Study

We conduct ablation studies to evaluate several key components of our work.

Attention amplification. This is crucial for object localization, the absence of attention amplification results in a decrease of grounding ability, i.e., the object is less likely to follow the boxes, as shown in the first row in Figure 6, and a decrease of metrics in Table 3.

Attention suppression. This is introduced to mitigate the unintended semantic mixing in multi-object scenarios, particularly when objects share similar characteristics. Since our attention amplification is applied only in the initial steps, and this constraint is subsequently relaxed. Without suppression, object A’s prompt feature can also attend to object B’s region, leading to semantic overlap. As shown in second row of Figure 6, where the tiger’s texture erroneously appears on the bear’s body. The third row shows that this issue can be resolved by enabling the attention suppression.

Camera embedding design. To assess the effectiveness of separately encoding panning (𝑐𝑥, 𝑐𝑦) and zooming (𝑐𝑧) movements in camera control as detailed in Section 3.2, we contrast this with a joint encoding approach. Here, [𝑐𝑥,𝑐𝑦,𝑐𝑧] are encoded into a single camera embedding vector using a shared MLP, followed by shared key-value projection matrix in the camera module. We train and evaluate the model with the same setting, we observed a reduced ability in camera movement control, with flow error increasing from

##### Table 3: Quantitative evaluation of attention amplification and suppression.

Attn amp. Attn sup. CLIP-sim ↑ mIoU (%)↑ AP50 (%)↑

× ✓ 25.82 15.35 3.46

##### ✓ × 27.49 38.87 10.25 ✓ ✓ 27.63 47.83 31.33

0.46 to 1.68. This underscores the advantages of separate encoding for distinct types of camera movements.

- 5 LIMITATIONS We consider several limitations of our method.

- (1) For joint control, while our method provides disentangled control over object and camera motion, conflicts can sometimes arise in the inputs. For instance, in the top row of Figure 7, we attempt to maintain a static object (house) within a static box while simultaneously panning the camera to the left. Given these conflicting signals, our method ends up generating a moving house, which is unrealistic. This necessitates careful and reasonable user interaction.
- (2) In camera control, due to the camera augmentation technique used in our method, which currently involves only 2D-panning and zooming, this limits the system’s ability to produce complex 3D camera movements that are out of this scope, e.g., our method cannot generate camera movements like "panning around an object". To overcome this constraint, we consider envisaging the adoption of more sophisticated augmentation algorithms, or curating a synthetic video dataset from a rendering engine given the camera movements, which we will leave in our future work.
- (3) In object control, another issue arises when handling colliding boxes. In scenarios like the one depicted in the bottom row of Figure 7, where two boxes overlap, the semantics of one object (the bear) can interfere with another (the tiger). This issue can be mitigated by modulating attention on an adaptively auto-segmented region during the diffusion sampling process, rather than relying on the initial box region.

- 6 CONCLUSION

In this work, we propose Direct-a-Video, a text-to-video framework that addresses the previously unmet need for independent and user-directed control over camera movement and object motion. Our approach effectively decouples these two elements by integrating a self-supervised training scheme for temporal cross-attention layers tailored for camera movement control, with a training-free modulation for spatial cross-attention dedicated to object motion control. Experimental evaluations demonstrate the capability of our approach in separate and joint control of camera movement and object motion. This positions Direct-a-Video as an efficient and flexible tool for creative video synthesis with customized motion.

### ACKNOWLEDGMENTS

Thisworkwassupported byaGRFgrant (Project No. CityU 11208123) from the Research Grants Council (RGC) of Hong Kong, and research funding from Kuaishou Technology.

Zoom in Pan right Pan down

Pan up Zoom in Pan left

Pan up Pan left Zoom out

Pan up Pan left Pan right

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Moving box

Static box Static Pan right

Static

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

|[Figure 197]|
|---|
|[Figure 198]|
|[Figure 199]|
|[Figure 200]|

|[Figure 201]|
|---|
|[Figure 202]|
|[Figure 203]|
|[Figure 204]|

|[Figure 205]|
|---|
|[Figure 206]|
|[Figure 207]|
|[Figure 208]|

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

“A horse walking on grassland”

“A zebra next to a river”

###### （a) （b)

Pan up

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Moving boxes Moving boxes Zoom out

Zoom in

Static Pan right Zoom out Pan left Pan down

Static Pan left Zoom out

Zoom in

|[Figure 237]|
|---|
|[Figure 238]|
|[Figure 239]|
|[Figure 240]|

|[Figure 241]|
|---|
|[Figure 242]|
|[Figure 243]|
|[Figure 244]|

|[Figure 245]|
|---|
|[Figure 246]|
|[Figure 247]|
|[Figure 248]|

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

[Figure 249]

|[Figure 250]|
|---|
|[Figure 251]|
|[Figure 252]|
|[Figure 253]|

|[Figure 254]|
|---|
|[Figure 255]|
|[Figure 256]|
|[Figure 257]|

|[Figure 258]|
|---|
|[Figure 259]|
|[Figure 260]|
|[Figure 261]|

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

（c) “A tiger and a bear walking on grass“ （d) “A tiger and a bear walking on grass“

- Figure 5: Joint control of object motion and camera movement. Given the same box sequence, by setting different camera movement parameters, our approach is capable of synthesizing videos that exhibit a diverse combination of foreground motion (yellow lines) and background motion (cyan lines). The user can create a well-defined overall video motion by distinctly specifying both object motion and camera movement using our method.

|[Figure 269]|[Figure 270]|[Figure 271]|[Figure 272]|
|---|---|---|---|

|[Figure 273]|[Figure 274]|[Figure 275]|[Figure 276]|
|---|---|---|---|

Attn amp.

“A tiger and a bear walking on grass”

|[Figure 277]|[Figure 278]|[Figure 279]|[Figure 280]|
|---|---|---|---|

Attn sup.

[Figure 281]

On Off

[Figure 282]

[Figure 283]

On

[Figure 284]

Off

[Figure 285]

On

[Figure 286]

On

Inputs

- Figure 6: Effect of attention amplification and suppression. Without amplification (the first row), the objects do not adhere to boxes; Without suppression (the second row), tiger’s texture mistakenly leaks into the bear’s body. These issues are resolved with both enabled (the third row).

[Figure 287]

##### Figure 7: Limitations of our method. Top: conflicting inputs can lead to unreal results - a moving house. Bottom: Overlapping boxes may lead to object interfere - tiger with a bear head.

Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

### REFERENCES

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. 2022. ediffi: Textto-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324 (2022).

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. 2023. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023a.

Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. 2023. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23040–23050.

Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Xiao Yang, and Mohammad Soleymani. 2023. MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer. arXiv preprint arXiv:2311.12052 (2023).

Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. Attendand-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–10.

Minghao Chen, Iro Laina, and Andrea Vedaldi. 2024. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 5343–5353.

Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, Tsung-Yi Lin, and Ming-Hsuan Yang. 2023a. Motion-Conditioned Diffusion Model for Controllable Video Synthesis. arXiv preprint arXiv:2304.14404 (2023).

Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. 2023b. Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models. arXiv preprint arXiv:2305.13840 (2023).

Iya Chivileva, Philip Lynch, Tomas Ward, and Alan Smeaton. 2023. Text prompts and videos generated using 5 popular Text-to-Video models plus quality metrics including user quality assessments. (10 2023). https://doi.org/10.6084/m9.figshare. 24078045.v2

Yufan Deng, Ruida Wang, Yuhao Zhang, Yu-Wing Tai, and Chi-Keung Tang. 2023. DragVideo: Interactive Drag-style Video Editing. arXiv preprint arXiv:2312.02216

(2023).

Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. 2023. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.

Mengyang Feng, Jinlin Liu, Kai Yu, Yuan Yao, Zheng Hui, Xiefan Guo, Xianhui Lin, Haolan Xue, Chen Shi, Xiaowen Li, et al. 2023. DreaMoving: A Human Video Generation Framework based on Diffusion Models. arXiv e-prints (2023), arXiv– 2312.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2023. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373

(2023).

Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao, Jay Zhangjie Wu, David Junhao Zhang, Mike Zheng Shou, and Kevin Tang. 2023. VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence. arXiv preprint arXiv:2312.02087 (2023).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai.

2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023).

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022).

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021).

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017). Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022a. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33 (2020), 6840–6851. Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022b. Video diffusion models. arXiv:2204.03458 (2022).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. 2023. Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation. arXiv preprint arXiv:2311.17117 (2023).

Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. 2023. PEEKABOO: Interactive Video Generation via Masked-Diffusion. arXiv preprint arXiv:2312.07509

(2023).

Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. 2023. VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models. arXiv preprint arXiv:2312.00845 (2023).

Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. 2021. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG) 40, 6 (2021), 1–12.

Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. 2023. Dense text-to-image generation with attention modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7701–7711.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu.

2023. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1931–1941.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. 2023. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22511–22521.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. 2023a. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023).

Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. 2023b. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761 (2023). Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

Wan-Duo Kurt Ma, JP Lewis, W Bastiaan Kleijn, and Thomas Leung. 2023. Directed diffusion: Direct control of object placement through attention guidance. arXiv preprint arXiv:2302.13153 (2023).

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2023. Nulltext inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6038–6047.

Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453 (2023).

Xun Long Ng, Kian Eng Ong, Qichen Zheng, Yun Ni, Si Yong Yeo, and Jun Liu. 2022. Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 19023–19034.

Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2023. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926 (2023).

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535 (2023).

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 (2022).

Anyi Rao, Jiaze Wang, Linning Xu, Xuekun Jiang, Qingqiu Huang, Bolei Zhou, and Dahua Lin. 2020. A unified framework for shot type classification based on subject centric lens. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16. Springer, 17–34.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10684– 10695.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22500–22510.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35 (2022), 36479– 36494.

Vishnu Sarukkai, Linden Li, Arden Ma, Christopher Ré, and Kayvon Fatahalian. 2024. Collage diffusion. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 4208–4217.

Xiaoyu Shi, Zhaoyang Huang, Weikang Bian, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. 2023. Videoflow: Exploiting temporal cues for multi-frame optical flow estimation. arXiv preprint arXiv:2303.08340 (2023).

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. 2023. Emergent Correspondence from Image Diffusion. arXiv preprint arXiv:2306.03881 (2023).

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018).

Jun Wang, Bohan Lei, Liya Ding, Xiaoyin Xu, Xianfeng Gu, and Min Zhang. 2023a. Autoencoder-based conditional optimal transport generative adversarial network for medical image generation. Visual Informatics (2023).

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023d. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571 (2023).

Tan Wang, Linjie Li, Kevin Lin, Yuanhao Zhai, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. 2023b. DisCo: Disentangled Control for Realistic Human Dance Generation. arXiv preprint arXiv:2307.00040 (2023).

Wen Wang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. 2023c. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599 (2023).

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. 2023f. VideoComposer: Compositional Video Synthesis with Motion Controllability. In Advances in Neural Information Processing Systems. 7594–7611.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2023e. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. arXiv preprint arXiv:2312.03641 (2023).

Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. 2023. Dreamvideo: Composing your dream videos with customized subject and motion. arXiv preprint arXiv:2312.04433 (2023).

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023b. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7623–7633.

Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, and Xiangyu Zhang. 2023a. Lamp: Learn a motion pattern for few-shot-based video generation. arXiv preprint arXiv:2310.10769 (2023).

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition. 5288–5296.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2023. MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model. arXiv preprint arXiv:2311.16498 (2023).

Han Yang, Ruimao Zhang, Xiaobao Guo, Wei Liu, Wangmeng Zuo, and Ping Luo. 2020. Towards photo-realistic virtual try-on by adaptively generating-preserving image content. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 7850–7859.

Shiyuan Yang, Xiaodong Chen, and Jing Liao. 2023a. Uni-paint: A unified framework for multimodal image inpainting with pretrained diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia. 3190–3199.

Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023b. Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation. arXiv preprint arXiv:2306.07954

(2023).

Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. 2023. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089 (2023).

Liang Yuan, Dingkun Yan, Suguru Saito, and Issei Fujishiro. 2024. DiffMat: Latent diffusion models for image-guided material generation. Visual Informatics (2024).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. 2023b. ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing. arXiv preprint arXiv:2305.17098 (2023).

Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. 2023a. Motiondirector: Motion customization of text-to-video diffusion models. arXiv preprint arXiv:2310.08465 (2023).

Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng.

2022. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018 (2022).

## Appendix

A ADDITIONAL IMPLEMENTATION DETAILS

- A.1 Camera Augmentation Details

Extracting camera movement parameters from real-world videos are computationally intensive, often requiring the cumbersome process of separating object motion from camera movement. To bypass these challenges, we propose a method of camera augmentation that simulates camera movement by algorithmically manipulating a stationary camera’s footage. In brief, the camera augmentation is implemented by altering the calculated cropping window across the video sequence captured by a stationary camera, thereby simulating the effect of camera movement in a computationally efficient manner. The detailed pseudo-code of this process is illustrated in Figure 8.

- A.2 Training Details for Camera Control

Camera movement parameters sampling. During the training, we adopt the following sampling scheme for camera movement parameters ccam = [𝑐𝑥,𝑐𝑦,𝑐𝑧]:

𝑐𝑥 ∼

0, with probability 31, Uniform(−1, 1), with probability 23,

𝑐𝑦 ∼

0, with probability 13, Uniform(−1, 1), with probability 23,

𝑐𝑧 ∼

1, with probability 13, 2𝜔, with probability 23, where 𝜔 ∼ Uniform(−1, 1).

Note that each component is sampled independently.

Training scheme. We adopt pretrained Zeroscope T2V model [Wang et al. 2023d] as our base model. To facilitate camera movement learning while retain the pretrained state, only the newly added layers are trainable, which include camera embedder and camera module. To speed up the training, we use a coarse-to-fine strategy: we first train on videos of size 256×256×8 (height × width × frames) for 100k iterations, then we resume training on videos of size 320 × 512 × 16 and 320 × 512 × 24 for 50k iterations each. The training is performed using a DDPM noise scheduler [Ho et al. 2020] with timestep 𝑡 uniformly sampled from [400, 1000], such preference to higher timesteps helps to prevents overfitting to lowlevel details, which are deemed non-essential for understanding temporal transitions. We employ an AdamW optimizer [Loshchilov and Hutter 2017] with a learning rate of 5e-5 and a batch size of 8 on 8 NVIDIA Tesla V100 GPUs.

- A.3 Inference Details for Camera Control.

In the text-to-image sampling process, classifier-free guidance [Ho and Salimans 2022] is widely used to facilitate the text response in generated images, where the predicted noise is extrapolated from the unconditional branch (which uses null-text ∅txt) towards the conditional branch (which uses normal prompt ctxt). We additionally propose a similar technique to enhance the camera control capability of our model. Specifically, our conditional branch uses

##### Table 4: Assessment of attention amplification on different parts of UNet.

E M D E+M M+D E+D E+M+D

CLIP-sim ↑ 26.20 25.93 26.75 26.35 26.74 27.62 27.63 mIOU(%) ↑ 30.90 14.28 29.25 31.71 28.98 49.06 47.83 AP50(%) ↑ 5.50 0.27 5.75 6.61 6.13 30.04 31.33

desired camera parameters ccam = [𝑐𝑥,𝑐𝑦,𝑐𝑧], while the unconditional branch uses a static camera status ∅cam = [0, 0, 1] (i.e., no panning or zooming). The predicted noise 𝜖ˆ𝜃 at each sampling step is calculated as:

𝜖ˆ𝜃 (z𝑡, ccam, ctxt,𝑡) = 𝜖𝜃 (z𝑡, ∅cam, ∅txt,𝑡)

(6)

+ 𝑠 (𝜖𝜃 (z𝑡, ccam, ctxt,𝑡) − 𝜖𝜃 (z𝑡, ∅cam, ∅txt,𝑡)) ,

where 𝑠 is the guidance scale. On the other hand, unlike text conditioning, which is applied throughout the sampling process, we found that applying the camera conditioning in only a few initial steps is sufficient for controlling camera movement, as the general temporal transitions is already determined in early stages. Formally, during the inference, we bypass the camera module when 𝑡 is less than a certain threshold, which we refer to as the camera control cut-off timestep, we empirically set this value to 0.85𝑇.

### B ADDITIONAL ABLATION STUDIES

We conduct additional ablation studies to validate the settings of our method.

Which layers for attention amplification? To determine which layers to apply the attention amplification, we divide the U-Net into three parts: encoder (E), middle layer (M), and decoder (D). We applied attention amplification to various combinations of these three and assessed their impact on the CLIP-sim, mIOU and AP50 scores. The results are presented in Table 4. We observed that applying attention amplification to either the encoder or the decoder significantly enhances object responsiveness, as evidenced by higher values across all metrics. Controllability is further strengthened when attention amplification is applied to both components. The middle layer has a comparatively smaller influence, incorporating middle layer does not bring noticeable statistic change. Consequently, we apply attention amplification across all layers.

Attention amplification hyper-parameters. In attention amplification, the strength 𝜆 and cut-off timestep𝜏 are two hyper-parameters. Generally, lower 𝜏 and higher 𝜆 will increase the strength of attentionamplification. To determine aproper choice of hyper-parameters, we conduct tests with different combinations of 𝜆 and 𝜏. Visual examples are provided in Figure 10. We observed that object responses are more sensitive to the value of𝜏 than to 𝜆. As illustrated in the 1st and 2nd rows, over-responsiveness in box regions typically occurs for 𝜏 < 0.9. This is because the early sampling stage in the diffusion model plays a significant role in determining the coarse layout of the output image or video; thus, applying amplification for an extended duration results in over-responsiveness in the box region. We also report CLIP-sim and mIOU metrics in Table 5, as can be seen, setting 𝜏 > 0.9 results in better semantic quality, as evidenced

Function aug_with_cam_motion(src_video, cx, cy, cz, h, w):

# Parameters: # src_video: Source video, a tensor with the size of [frames (f), 3, src_height, src_width] # cx: Horizontal translation ratio (-1 to 1) # cy: Vertical translation ratio (-1 to 1) # cz: Zoom ratio (0.5 to 2) # h: Height of the augmented video # w: Width of the augmented video # Returns: Augmented video, a tensor with the size of [f, 3, h, w]

# Get source frame number, width and height from src_video f, src_h, src_w = src_video.shape[0], src_video.shape[2], src_video.shape[3]

# Initialize camera boxes for frame cropping cam_boxes = zeros(f, 4) # f frames, 4: [x1,y1,x2,y2]

# Calculate dynamic cropping relative coordinates for each frame # The first frame coordinates is the reference, which is always [0,0,1,1].

- cam_boxes[:, 0] = linspace(0, cx + (1 - 1/cz) / 2, f) # x1, top-left x
- cam_boxes[:, 1] = linspace(0, cy + (1 - 1/cz) / 2, f) # y1, top-left y
- cam_boxes[:, 2] = linspace(1, cx + (1 + 1/cz) / 2, f) # x2, bottom-right x
- cam_boxes[:, 3] = linspace(1, cy + (1 + 1/cz) / 2, f) # y2, bottom-right y # Compute the minimum and maximum relative coordinates

- min_x = min(cam_boxes[:, 0::2])

- max_x = max(cam_boxes[:, 0::2])

min_y = min(cam_boxes[:, 1::2])

- max_y = max(cam_boxes[:, 1::2])

# Normalize the camera boxes normalized_boxes = zeros_like(cam_boxes)

- normalized_boxes[:, 0::2] = (cam_boxes[:, 0::2] - min_x) / (max_x - min_x)
- normalized_boxes[:, 1::2] = (cam_boxes[:, 1::2] - min_y) / (max_y - min_y)

# Initialize a tensor for the new frames augmented_frames = zeros(f, 3, h, w)

# Process each frame for i in range(f):

# Calculate the actual cropping coordinates x1, y1, x2, y2 = normalized_boxes[i] * tensor([src_w, src_h, src_w, src_h])

# Crop the frame according to the coordinates crop = src_video[i][:, int(y1):int(y2), int(x1):int(x2)]

# Resize the cropped frame and store it augmented_frames[i] = interpolate(crop, size=(h, w), mode='bilinear')

return augmented_frames

##### Figure 8: Pseudo-code for the camera augmentation function.

[Figure 288]

[Figure 289]

[Figure 290]

“A zebra next to a river”

“A man surfing in the sea”

Camera control Object control

|[Figure 291]|[Figure 292]|[Figure 293]|[Figure 294]|
|---|---|---|---|

|[Figure 295]|[Figure 296]|[Figure 297]|[Figure 298]|
|---|---|---|---|

|[Figure 299]|[Figure 300]|[Figure 301]|[Figure 302]|
|---|---|---|---|

|[Figure 303]|[Figure 304]|[Figure 305]|[Figure 306]|
|---|---|---|---|

+0.32 pan right

-0.35 pan left

[Figure 307]

×0.77 zoom-out

- ×0.63 zoom-out

[Figure 308]

[Figure 309]

Camera control Object control

[Figure 310]

[Figure 311]

-0.8 pan left

- ×1.66 zoom-in

|[Figure 312]|[Figure 313]|[Figure 314]|[Figure 315]|
|---|---|---|---|

|[Figure 316]|[Figure 317]|[Figure 318]|[Figure 319]|
|---|---|---|---|

| |
|---|

|[Figure 320]|[Figure 321]|[Figure 322]|[Figure 323]|
|---|---|---|---|

|[Figure 324]|[Figure 325]|[Figure 326]|[Figure 327]|
|---|---|---|---|

-0.2 pan up

| |
|---|

[Figure 328]

[Figure 329]

+0.5 pan right ×1.24 zoom-in

- Figure 9: Qualitative comparison of generated videos using the same prompt but different controls. 1st row: base model, i.e., no control; 2nd row: camera control only; 3rd row: object control only; 4th row: camera + object control. Adding control introduces more dynamic content without noticeable quality degradation.

by higher CLIP-sim scores and the visual results. On the other hand, setting 𝜆 ≥ 10 generally yields higher mIOU values. It is important to note that while a higher mIOU indicates better object grounding ability, it does not necessarily equate to better object quality. In summary, we empirically determine that 𝜏 ∈ [0.9𝑇, 0.95𝑇] and 𝜆 ∈ [10, 25] are generally appropriate for most cases.

|[Figure 330]|
|---|

|[Figure 331]|
|---|

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

|[Figure 344]|
|---|

|[Figure 345]|
|---|

“A rhino standing in the forest”

𝝀 = 𝟓 𝝀 = 𝟏𝟎 𝝀 = 𝟐𝟓 𝝀 = 𝟓𝟎

𝝉 = 𝟎.𝟖𝟎𝑻

𝝉 = 𝟎.𝟖𝟓𝑻

𝝉 = 𝟎.𝟗𝟎𝑻

𝝉 = 𝟎.𝟗𝟓𝑻

- Figure 10: Effect of attention amplification strength 𝜆 and cut-off timestep 𝜏 (only the first frame is showed).

##### Table 6: Quantitative evaluation for camera/object control on video quality.

Base Cam Obj Cam+Obj

FID-vid ↓ 41.12 44.95 43.55 41.20 FVD ↓ 1104.36 1204.55 1300.86 1280.88

Effect of adding control on quality. To evaluate the impact of incorporating camera/object control on the video quality, we calculate the FVD and FID-vid score under four different settings: (1) Base: no control involved, i.e., the vanilla model; (2) Cam: only camera movement control is involved (with random camera parameters); (3) Obj: only object control is involved; and (4) Cam+Obj: both camera and object control are enabled. The quality metrics are presented in Table 6. The statistic shows that adding control may have a minor influence but not so significant, as the metrics are approximately in the same level with minor fluctuations. We also present visual examples in Figure 9. As can be seen, adding control does not result in a noticeable degradation of quality; on the contrary, it introduces more dynamic content into the generated videos compared to the base model.

### C ADDITIONAL RESULTS

We show additional results in Figure 11. Please refer to our project page for dynamic results.

Table 5: CLIP-sim and mIOU metrics tested on different attention amplification hyper-parameters.

𝜆 = 5 𝜆 = 10 𝜆 = 25 𝜆 = 50

CLIP-sim mIOU CLIP-sim mIOU CLIP-sim mIOU CLIP-sim mIOU

𝜏 = 0.80𝑇 26.81 27.99 24.91 47.71 24.65 49.76 24.83 47.83 𝜏 = 0.85𝑇 26.85 25.72 25.31 42.52 25.03 47.69 25.29 47.75 𝜏 = 0.90𝑇 26.78 26.74 26.15 40.83 26.60 45.50 26.18 44.08 𝜏 = 0.95𝑇 26.48 21.82 27.49 41.61 27.63 47.83 27.32 43.92

[Figure 346]

[Figure 347]

[Figure 348]

Prompt Camera movement Object motion

Generated Videos

|[Figure 349]|[Figure 350]|[Figure 351]|[Figure 352]|[Figure 353]|[Figure 354]|
|---|---|---|---|---|---|

“A waterfall in a beautiful forest with fall foliage”

[Figure 355]

+0.3 pan right

###### CameraMovement

|[Figure 356]|[Figure 357]|[Figure 358]|[Figure 359]|[Figure 360]|[Figure 361]|
|---|---|---|---|---|---|

“A waterfall in a beautiful forest with fall foliage”

[Figure 362]

+0.8 pan right

|[Figure 363]|[Figure 364]|[Figure 365]|[Figure 366]|[Figure 367]|[Figure 368]|
|---|---|---|---|---|---|

−0.3 pan up

[Figure 369]

“A villa in a garden“

+0.3 pan right

|[Figure 370]|[Figure 371]|[Figure 372]|[Figure 373]|[Figure 374]|[Figure 375]|
|---|---|---|---|---|---|

−0.3 pan up

[Figure 376]

“A villa in a garden“

+0.8 pan right

|[Figure 377]|[Figure 378]|[Figure 379]|[Figure 380]|[Figure 381]|[Figure 382]|
|---|---|---|---|---|---|

“A UFO is flying in the sky”

###### SingleObjectMotionMultipleObjectMotionJointControl

|[Figure 383]|[Figure 384]|[Figure 385]|[Figure 386]|[Figure 387]|[Figure 388]|
|---|---|---|---|---|---|

“Firework is exploding in the night”

|[Figure 389]|[Figure 390]|[Figure 391]|[Figure 392]|[Figure 393]|[Figure 394]|
|---|---|---|---|---|---|

“A drone is landing on the grassland”

|[Figure 395]|[Figure 396]|[Figure 397]|[Figure 398]|[Figure 399]|[Figure 400]|
|---|---|---|---|---|---|

“A jeep is climbing over the rocks”

|[Figure 401]|[Figure 402]|[Figure 403]|[Figure 404]|[Figure 405]|[Figure 406]|
|---|---|---|---|---|---|

“A tiger walking by a campfire in the forest”

|[Figure 407]|[Figure 408]|[Figure 409]|[Figure 410]|[Figure 411]|[Figure 412]|
|---|---|---|---|---|---|

“A running dog and a flying UFO on the beach”

|[Figure 413]|[Figure 414]|[Figure 415]|[Figure 416]|[Figure 417]|[Figure 418]|
|---|---|---|---|---|---|

“A tiger and a zebra walking on grass”

|[Figure 419]|[Figure 420]|[Figure 421]|[Figure 422]|[Figure 423]|[Figure 424]|
|---|---|---|---|---|---|

“Clown fish and jellyfish swimming in ocean with coral reef”

|[Figure 425]|[Figure 426]|[Figure 427]|[Figure 428]|[Figure 429]|[Figure 430]|
|---|---|---|---|---|---|

[Figure 431]

“A bear walking on snow”

×0.7 zoom out

|[Figure 432]|[Figure 433]|[Figure 434]|[Figure 435]|[Figure 436]|[Figure 437]|
|---|---|---|---|---|---|

−0.3 pan up

[Figure 438]

“A leaf is flowing in the sky”

−0.3 pan left

|[Figure 439]|[Figure 440]|[Figure 441]|[Figure 442]|[Figure 443]|[Figure 444]|
|---|---|---|---|---|---|

[Figure 445]

“A shark and a jellyfish swimming in the sea”

×0.7 zoom out

|[Figure 446]|[Figure 447]|[Figure 448]|[Figure 449]|[Figure 450]|[Figure 451]|
|---|---|---|---|---|---|

−0.5 pan left

“An elephant and a flying UFO”

[Figure 452]

×0.7 zoom out

+0.3 pan down

##### Figure 11: Additional results of camera movement control and object motion control.

