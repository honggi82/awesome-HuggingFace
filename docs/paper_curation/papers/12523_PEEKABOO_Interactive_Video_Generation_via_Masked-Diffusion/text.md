## PEEKABOO: Interactive Video Generation via Masked-Diffusion

# arXiv:2312.07509v2[cs.CV]19Apr2024

Yash Jain1† Anshul Nasery2† Vibhav Vineet1 Harkirat Behl1 1Microsoft 2University of Washington

Project Webpage

[Figure 1]

[Figure 2]

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

Figure 1. Zero-training No-latency interactive video generation. PEEKABOO allows users to control the output (object size, location and trajectory) for any off-the-shelf video diffusion models, through specially designed masking modules. First row shows a panda playing PEEKABOO by following an expanding mask in left direction.

### Abstract

Modern video generation models like Sora have achieved remarkable success in producing high-quality videos. However, a significant limitation is their inability to offer interactive control to users, a feature that promises to open up unprecedented applications and creativity. In this work, we introduce the first solution to equip diffusionbased video generation models with spatio-temporal control. We present PEEKABOO, a novel masked attention module, which seamlessly integrates with current video generation models offering control without the need for additional training or inference overhead. To facilitate future research, we also introduce a comprehensive benchmark for interactive video generation. This benchmark offers a standardized framework for the community to assess the efficacy of emerging interactive video generation models. Our extensive qualitative and quantitative assessments reveal that PEEKABOO achieves up to a 3.8× improvement in mIoU over baseline models, all while maintaining the same latency. Code and benchmark are available on the webpage.

### 1. Introduction

Generating realistic videos from natural language descriptions is a challenging but exciting task that has recently made significant progress [4, 20, 35, 38, 40]. This is largely due to the development of powerful generative models and latent diffusion models (LDMs [32]), which can produce high-quality and diverse videos from text. These models have opened up new possibilities for creative applications and expression.

As the generation quality continues to improve, we can expect more innovation and potential in this domain. An important aspect is to enable more interactivity and user control over the generated videos (or better alignment), by allowing the user to control the spatial and temporal aspects of the video, such as the size, location, pose, and movement of the objects. This enables users to express their creativity and imagination through generating videos that match their vision and preferences. It can also be useful for various

†equal contribution

applications, such as education, entertainment, advertising, and storytelling, where users can create engaging and personalized video content.

While current models are capable of producing temporally and semantically coherent videos, the user cannot have spatio-temporal control [41]. Moreover, these models sometimes fail to produce the main object in the video [1]. In order to control the output of videos interactively, a model would need to incorporate inputs about spatial layouts into its generation process. One set of approaches to achieve spatial control on the network output involves training the entire network or specialized adaptors on spatially grounded data [28, 39]. However, such methods involve retraining which is resource and data intensive, limiting their access to the wider community. This raises the question Can we create a training-free technique that can introduce interactivity through desired control in videos while utilising large scale pretrained Text-to-Video (T2V) models?

In this work, we propose PEEKABOO, a training-free method to augment any off-the-shelf LDM based videogeneration model with spatial control. Further, our method has negligible inference overhead. For control over individual object generation, we propose to use local context instead of global context. We propose an efficient strategy to achieve controlled generation within the T2V inference pipeline. PEEKABOO works by refocusing the spatial-, cross-, and temporal-attention in the UNet [34] blocks.

Figures 1, 3 and 5 demonstrate outputs that our method produces for a variety of masks and prompts. Our method is able to maintain a high quality of video generation, while controlling the output spatio-temporally. To evaluate the spatio-temporal control of video generation method, we propose a new benchmark by adapting an existing dataset [26], and curating a new dataset for our task (Section 5.1.1), and proposing an evaluation strategy for further research in this space. Finally, we show the versatility of our approach on two text-to-video models [38] and a textto-image model [33]. This demonstrates the wide applicability of our method. In summary:

- • We introduce PEEKABOO which i) allows interactive video generation by inducing spatio-temporal and motion control in the output of any UNet based off-the-shelf video generation model, ii) is training-free and iii) adds no additional latency at inference time.
- • We curate and release a public benchmark, SSv2-ST for evaluating spatio-temporal control in video generation. Further, we create and release the Interactive Motion Control (IMC) dataset to evaluate interactive inputs from a human.
- • We extensively evaluate PEEKABOO on i) multiple evaluation datasets, ii) with multiple T2V models (ZeroScope and ModelScope) and iii) multiple evaluation metrics. Our evaluation shows upto 2.9× and 3.8× gain in mIoU

score by using PEEKABOO over ZeroScope and ModelScope respectively.

• We present qualitative results on spatio-temporally controlled video generation with PEEKABOO, and also showcase its ability of to overcome some fundamental failure cases present in existing models.

### 2. Related Work

#### 2.1. Video Generation

Text-based video generation using latent diffusion model has taken a significant leap in recent years [4, 8, 15, 16, 35, 44]. Make-a-video [35] introduced the 3D UNet architecture, by decomposing attention layers into spatial, cross and temporal attention layers. Further progress in this generation pipeline was made by [8, 15, 17, 44], while keeping the core three attention-layer architecture intact. Although these works focus on generating videos with high relevance to the text input, they do not provide spatiotemporal control in each frame. More recent works have tried to equip models with this ability to control generation spatio-temporally. Such methods have integrated guidance from depth maps [12], target motion [7, 18] or a combination of these modalities to generate videos [39]. However, all these works either require re-training the base model or an external adapter with aligned grounded spatio-temporal data, which is a challenging and expensive task.

On the other hand, zero-training works include Text2video-zero [20], which integrates optical flow guidance with image model to get consistent frames, ControlVideo [43], which incorporates sequence of supervising frames (depth maps, stick figures etc.) to control the motion of the video, and Free-Bloom [19], which combines a large language model (LLM) with a text-to-image model to get coherent videos. However, these methods extend specialized image models which were trained on grounded data, and cannot be used with off-the-shelf video-generation models. The closest method to our work is a concurrent work [24]. The paper uses an LLM to generate bounding box co-ordinates across scenes for an object in the prompt. They use an off-the-shelf video generation model in conjunction with a special guidance module. However, their work has a latency overhead due to extra steps in special guidance module which is absent in our method.

#### 2.2. Controllable Text to Image generation

Recent works have explored incorporating spatial and stylistic control while generating images from text using diffusion models. These methods can broadly be categorized into those requiring training of the model [22], and training-free methods [1, 5, 11, 23, 31]. The former line of works require large amounts of compute resources, as well as spatially grounded data to train their models. The latter

|[Figure 13]<br><br>[Figure 14]| | |
|---|---|---|
| | |[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]|
| | | |
|[Figure 19]<br><br>|[Figure 20]<br><br>[Figure 21]| |

- Figure 2. PEEKABOO Module: Our method proposes converting attention modules of an off-the-shelf 3D UNet into masked spatiotemporal mixed attention modules. We propose to use local context for generating individual objects and hence, guide the generation process using attention masks. For each of spatial-, cross-, and temporal-attentions, we compute attention masks such that foreground pixels and background pixels attend only within their own region. We illustrate these mask computations for an input mask ( size 2×2 and 3 frames) which changes temporally as shown on the left. Green pixels are background pixels and orange are foreground. In the attention masks, both green and orange pixels have a value of 1, and gray pixels have a value of 0. We add the colors for ease of exposition. This masking is applied for a fixed number of steps, after which free generation is allowed. Hence, foreground and background pixels are hidden from each other before being visible, akin to a game of PEEKABOO. Best viewed in color.

either try to shape the spatial and cross-attention maps using energy function guided diffusion, or through masking. Our method is hence closer to the second type of works, however, directly extending these to videos is non-trivial due to the spatio-temporal nature of video generation.

Guided Attention The idea of guiding attention maps to control the generation in the image domain has gained popularity recently. Agarwal et al. [1] focus on minimizing overlap in attention maps for different prompt words, maintaining object information across diffusion steps. Epstein et al. [11] suggest various energy functions on crossattention maps to control spatial properties of objects via guided sampling. Phung et al. [31] extend this by ensuring both cross and self-attention maps accurately represent objects, achieving this through optimized noise and segmented attention. Such optimization based methods have inference time overheads, in contrast with our method. Cao et al. [5] uses thresholded cross-attention maps of the object tokens as masks for self-attention, and ensures that foreground pixels only interact with other pixels within the foreground. Their method also requires multiple diffusion inference calls, or requires a source image as an input. Further, they apply their technique only for controlling the pose or actions of objects, which is orthogonal to our task.

### 3. Preliminaries: Video Diffusion Models

Diffusion models [36] are generative models that generate images or videos through gradually denoising random gaussian noise. The most effective amongst these are Latent Diffusion models (LDMs) [32] including Stable Diffusion. LDMs have two components: First is an image compression auto-encoder, which maps the image x to and back from a

lower dimensional latent z. Second component is a Denoising Autoencoder fθ(z) which operates in the latent space and gradually converts random noise to the image latent.

Text conditioning Most current text-to-video methods utilize a conditional latent diffusion model which takes a text query as input [29, 35, 38]. The denoising autoencoder is thus conditioned on the text caption c as

zt+1 = fθ(zt|c), (1)

where fθ is a 3D UNet [34]. During inference, the input noise is iteratively cleaned and aligned towards the desired text caption. This is achieved by including a cross attention with the text embedding.

### 4. PEEKABOO

Spatio-temporal conditioning For interactive generation, the denoising should also be conditioned on the userdesired spatial location and movement of the objects in the video. This is rather complicated, because unlike Equation 1 where the entire latent zt is conditioned on c, in this setting, parts of the video have to be conditioned on parts of the caption. Note that this would become a conditional distribution with multiple conditions.

A possible solution is to encode the extra conditions as grounding pairs (spatio-temporal volume, text embedding) and pass them as context tokens in the cross attention layer, and train accordingly, taking inspiration from the image based method Gligen [22] or even Flamingo [2]. On the other hand, we want to explore using a frozen fθ.

#### 4.1. Masked Diffusion

We draw a parallel with the segmentation problem, which is the inverse of spatio-temporal conditioned generation prob-

lem. In particular, we take inspiration from MaskFormer [9] and Mask2Former [10] who proposed to formulate segmentation as a mask classification problem. This formulation is widely used and accepted, not just for segmentation but even detection [21] and unified models [45].

Cheng et al. [10] propose to split segmentation into grouping into N regions which are represented with binary masks. Hence, Cheng et al. [10] advocate using local features for segmenting individual objects. On the other hand, text-to-video diffusion models operate on conditioning a global context, as shown in Eqn 1. Using the above insight to tackle the problem of spatio-temporal conditioned generation we also propose to use local context for generating individual objects, and then add them together. In order to control the spatial locations of objects, we propose to modify the attention computations in the transformer blocks of the diffusion model to masked attention calls similar to [10]. This enables better local generation without any additional computation or diffusion steps.

#### 4.2. Masked spatio-temporal mixed attention

Given an input bounding box for a foreground object in the video, we create a binary mask for the foreground object, and downsample it to the size of the latent. We create block sparse attention masks as described below. We use additive masking for attention, i.e. for any query Q, key K, value V a binary 2D attention mask M,

QKT d

MaskedAttention(Q,K,V,M) = softmax(

+ M)V

where M[i,j] = −∞ if M[i,j] = 0

0 if M[i,j] = 1

(2)

Here, the additive mask M is such that it has a large negative value on the masked out entries in M, leading to the attention scores for such entries being small. Note that M ∈ {0,1}d

q×dk, where dq,dk are the lengths of queries and keys respectively. We denote the length of the text prompt by ltext, the length of the video by lvideo, and the dimensions of the latents by llatents. The text input is denoted by T, and the input mask for frame f is denoted by Minputf . For the ease of notation, we assume that the input masks and the latents are flattened along their spatial dimensions. The shape of Minput is lvideo × llatents We also define the function fg(·), which takes a pixel or a text token as input, and returns 1 if it corresponds to the foreground of the video, and 0 otherwise.

By nudging the foreground token to attend only to the pixels at the desired location at each frame, we can control the position, size and movement of the object. However, naively enforcing this attention constraint only in the crossattention layer is not sufficient for spatial control. This is because the foreground and background pixels also interact

through spatial- and temporal attention. We now discuss how to effectively localise the generation context.

Masked cross attention For each frame f, we compute an attention mask MCAf , which is a 2-dimensional matrix of size llatents × ltext. For each pixel-token pair, this mask is 1 iff both the pixel and token are foreground, or if both of them are in the background. Formally

MCAf [i,j] = fg(Minputf [i]) ∗ fg(T[j])

+ (1 − fg(Minputf [i])) ∗ (1 − fg(T[j])) (3)

This ensures that the latents attend to the foreground and the background tokens at the correct locations.

Masked spatial attention For each frame f, we compute an attention mask MSAf which is a 2-dimensional matrix of size llatents ×llatents. For each pixel pair, this mask is 1 iff both the pixels are foreground, or if both of them are in the background. Formally

MSAf [i,j] = fg(Minputf [i]) ∗ fg(Minputf [j])

+ (1 − fg(Minputf [i])) ∗ (1 − fg(Minputf [j]))

(4)

This additionally focuses the attention to ensure that the foreground and background are generated at the correct locations, by encouraging them to evolve independently for the initial steps. This also helps improve the quality of generation since it leads to adequate interaction within the foreground and background regions. A similar idea in the context of image generation had been explored in MasaCtrl[5] in their self attention layer.

Masked temporal attention For each latent pixel i, we compute a mask MTAi , which is a 2D matrix of size lvideo× lvideo. For each frame pair, the value of this mask is 1 if the pixel i is a foreground pixel in both frames, or if it is a background pixel in both frames. Formally,

MTAi [f,k] = fg(Minputf [i]) ∗ fg(Minputk [i])

+ (1 − fg(Minputf [i])) ∗ (1 − fg(Minputk [i]))

(5)

This ensures temporal consistency for the generation since it provides correct local context for foreground and background latents across time.

#### 4.3. Zero-training Pipeline

Putting the selective masks in a diffusion pipeline gives us a zero-training method, dubbed PEEKABOO. PEEKABOO integrates in the attention layers of the 3D-UNet architecture

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

- Figure 3. Spatial control with PEEKABOO: Changing the bounding box while providing the same prompt leads to generated panda being faithful to the input layout in terms of size and location with our method.

of text-to-video models. We perform selective generation of foreground and background object for a fixed number of steps t and then allow free-generation for the rest of steps. This free generation enables the foreground and background pixels to cohesively integrate with each other on the same canvas as have been done by [3, 23]. In essence, our method ensures that foreground pixels cannot “see” the background pixels for some steps (and vice versa), before being visible to each other. This is akin to a game of PEEKABOO.

Unlike image control methods [3, 23], PEEKABOO does not require extra inference overhead in the form of more number of diffusion steps and works with very low value of fixed step t (refer to Appendix for more details). This ensures that there is no gain in latency during generation while providing extensive spatial control.

Further, since PEEKABOO is a zero-training off-the-shelf technique it is versatile to implement in all diffusion models and can work with present as well as future text-to-video models. Thus, PEEKABOO can give spatio-temporal control in better quality generation models which are not explicitly trained on any spatially-grounded dataset.

- 4.4. Extensions

Currently, majority of the diffusion pipelines have a UNetbased architecture. This enables PEEKABOO to become versatile and be used not only in Text-to-Video scenario, but in Text-to-Image setup with a possibility in other generation modalities too.

Automatically generated input masks Since our method is orthogonal to the choice of input masks, we can use a large language model to generate the input masks for an object corresponding to a given prompt, in a similar fashion as concurrent works [23, 25]. In Table 2, we demonstrate that doing this leads to videos with better quality than the baseline model. Moreover, it enables our method to be endto-end in terms of only requiring a text prompt from the user.

Image generation Image generation diffusion method are based of 2D-UNet architecture, with the absence of temporal attention layer. Analogous to our text-to-video setup,

we can adapt PEEKABOO for Image Diffusion models. The spatial-attention mask maintains the semantic structure of the image while the cross-attention mask focuses the attention of foreground token on desired location and vice versa for background. In Figure 7, we showcase spatial control on an off-the-shelf diffusion model and highlight the versatility of our method.

### 5. Experiments

In this section, we demonstrate the effectiveness of our method. The main focus of our technique is to generate objects in specific spatio-temporal locations in videos. We first evaluate this region level control in Sec 5.1.1. In 5.1.2, we compare the generation quality against baselines to show that grounding enables much better generation. We also demonstrate qualitative results of our method, and perform ablation analysis on our method and show the effect of each component on the final generations.

#### 5.1. Quantiative Analysis

We first present quantitative results on evaluating the spatial control and the quality of videos generated by PEEKABOO.

##### 5.1.1 Spatial Control

Evaluation Datasets Evaluating spatial control in multiple text-to-video models is a challenging task and requires creating a common benchmark for (prompt, mask) pairs. We develop a benchmark obtained from a public video dataset with high-quality masks that represent realistic locations for day-to-day subjects. Further, we also curated a set of (prompt, mask) pairs that represent an interactive input from the user in controlling a video and its subject.

- • Something-something v2-Spatio-Temporal (ssv2-ST): We use Something-Something v2 dataset [14, 26] to obtain the generation prompts and ground truth masks from real action videos. We filter out a set of 295 prompts. The details for this filtering are in the appendix. We then use an off-the-shelf OWL-ViT-large open-vocabulary object detector [27] to obtain the bounding box (bbox) annotations of the object in the videos. This set represents bbox and prompt pairs of real-world videos, serving as a test bed for both the quality and control of methods for generating realistic videos with spatio-temporal control.
- • Interactive Motion Control (IMC): We also curate a set of prompts and bounding boxes which are manually defined. We use GPT-4 to generate prompts and pick a set of 34 prompts of objects in their natural contexts. These prompts are varied in the type of object, size of the object and the type of motion exhibited. We then annotate 3 sets of bboxes for each prompt, where the location, path taken, speed and size are varied. This set of 102 promptbbox pairs serve as our custom evaluation set for spatial

DAVIS16 LaSOT ssv2-ST IMC mIoU (↑) AP50 (↑) Cov (↑) CD (↓) mIoU AP50 Cov CD (↓) mIoU AP50 Cov CD (↓) mIoU AP50 Cov CD (↓)

Method Latency

LLM-VD [24]* 2.20× 26.1 15.2 96 0.19 13.5 4.6 98 0.24 27.2 21.2 61 0.12 36.1 33.3 97 0.13 ModelScope [38] 1× 19.6 5.7 100 0.25 4.0 0.7 96 0.33 12.0 6.6 44.7 0.17 9.6 2.4 93.3 0.25 w/ PEEKABOO 1.03× 26.0 16.6 93 0.18 14.6 10.2 98 0.25 33.2 35.8 63.7 0.10 36.1 33.3 96.6 0.13 ZeroScope [38] 1× 11.7 0.1 100 0.22 3.6 0.4 100 0.3 13.9 9.3 42.0 0.22 12.6 0.6 88.0 0.26 w/ PEEKABOO 1.03× 20.6 17.9 100 0.19 11.5 11.9 100 0.28 34.7 39.8 56.3 0.17 36.3 33.8 96.3 0.12

- Table 1. Evaluation of spatio-temporal control on mIoU, AP50, Coverage and Centroid Distance (CD): We evaluate two different video generation models on spatio-temporal control against DAVIS16, LaSOT, ssv2-ST, and IMC datasets. As demonstrated by mIoU and CD, the videos generated by PEEKABOO endow the baselines with spatio-temporal control. PEEKABOO also increases the quality of the main objects in the scene, as seen by higher AP50 and Coverage scores. Further, LLM-VD[24] has higher inference cost whereas PEEKABOO does not affect latency. *: LLM-VD[24] has not released code, this is our re-implementation.

Method FVD@MSR-VTT (↓)

CogVideo (English) [17] 1294 MagicVideo [44] 1290

ModelScope [38] 868 ModelScope w/ PEEKABOO 609

- Table 2. Video quality evaluation. PEEKABOO is able to generate videos with higher quality than other baselines. We use bounding boxes generated by GPT-4 as inputs to the model.

spatio-temporal control and video quality.

Evaluation methodology. After generating videos for each (prompt, mask) pair, we pass these videos through OWLViT-large detector to compute bboxes for each generated video. We first compute the fraction of generated videos for which OwL-ViT detects bboxes in more that 50% of the generated frames. We report this fraction as the Coverage of the model in Table 1. However, the lack of a detected bbox does not necessarily imply the lack of an object generated, since OwL-ViT could fail to capture some objects correctly. Hence, to evaluate the spatio-temporal control of the generation method, we first filter out videos where less than 50% frames have a detected bbox. We then compute the Intersection-over-Union of the detected bboxes and the input mask on these filtered videos. We report the mean of these IoU (mIoU) scores for each method in Table 1. These two metrics together provide a good proxy of the quality of the generated videos as well as the spatio-temporal control imparted. We compute the Centroid Distance (CD) as the distance between the centroid of the generated object and input mask, normalized to 1. This measures control of the generation location. Finally, we report the average precision@50% (AP50) of the detected and input bboxes averaged over all videos. For generated frames with the object present, AP50 represents the spatial control provided by the method, while mIoU measures the model’s ability to match the input bboxes exactly and penalizes frames where the object cannot be detected.

control. Note that since ssv2-ST dataset has a lot of inanimate objects, we bias this dataset to contain more living objects. This dataset represents possible input pairs that real users may generate.

- • LaSOT: We repurpose a large-scale object tracking dataset –LaSOT[13]– for evaluating control in video generation. This dataset contains prompt-bbox-video triplets for a large number of classes. The videos have frame level annotations specifying the location of the object in the video. We subsample the videos to 8 FPS and then randomly pick up 2 clips per video from the test set of this dataset. This gives us 450 total clips across 70 different object categories.
- • DAVIS-16: DAVIS-16[30] is another video object segmentation dataset that we consider. We take videos from its test set, manually annotating them with prompts. We use the provided segmentation masks to create input bboxes. This gives us 40 prompt-bbox pairs in total, where each video has a different subject.

Results In Table 1, we demonstrate that our method adds control to the model. We verify that our method enables spatio-temporal control, as evidenced by the higher (upto 2.5x) AP@50 scores and lower CF on all four datasets (DAVIS16, LaSOT, ssv2-ST and IMC). This means that the generated objects are close to the true centroid of the input mask, and their shape and size are also consistent with the input mask. We observe significant jump in mIoU score with PEEKABOO across different models and LLM-VD [24], highlighting superior spatio-temporal control achieved

Experimental Setup We use two base models for our evaluation, Zeroscope and ModelScope [38]. These models are run for the default number of inference steps, with default temperature and classifier guidance parameters. We also experiment with mask guidance steps in the appendix. We provide the model with the text prompt and the set of input bboxes. The generated videos are then evaluated for

AP50 ablations

40

No MCA

No MTA

No MSA

Peekaboo

| |
|---|

| |
|---|

| |
|---|

35

30

25

AP50%

20

15

10

5

0

ZeroScope ModelScope

Performance with varying bbox sizes

| |Modelscope PEEKABOO<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.4

0.3

mIoU

0.2

0.1

0.0

0.0 0.1 0.2 0.3 0.4 0.5

Intra-Video BBox Size Std Dev

- Figure 4. (a) Ablation Studies on IMC: The performance of PEEKABOO varies as different attention masks are removed. The AP50 drops the most when cross-attention masks are removed, indicating their importance to spatial control, followed by temporal and spatial attention. (b) Performance against varying bbox sizes: mIoU score compared against varying bounding box sizes across time in videos. We observe that PEEKABOO provides better control independent of bbox size variation. Best viewed in color.

through PEEKABOO. Further, PEEKABOO has a higher coverage than the baseline models and LLM-VD, indicating that our method is also able to generate objects when the base model could not do so. Finally, we note that PEEKABOO introduce minimal latency increase to the original method compared to LLM-VD’s 2.20× increase in inference time on ModelScope.

##### 5.1.2 Quality control

While the above datasets provide evidence for PEEKABOO’s spatio-temporal control, we also benchmark our method on MSR-VTT[42] – a large scale video generation dataset – to evaluate the quality of videos generated. We benchmark PEEKABOO for evaluating quality control using Fr´echet Video Distance score (FVD) metric [37]. FVD is calculated based on I3D model trained on Kinetics-400 dataset [6]. Following previous works, we evaluate on the test-set of MSR-VTT containing 2900 videos by randomly sampling one of the 20 captions for each video. We demonstrate the versatility of our method by using bounding boxes generated by GPT-4. We query GPT-4 to generate series of locations for the foreground object depending on the prompt. We evaluate on ModelScope model and compare the scores with PEEKABOO. Table 2 shows that PEEKABOO increases the quality of generated while providing spatial control during video generation. The performance of these methods is also better than other baselines, indicating that PEEKABOO can be integrated in an automated pipeline to use GPT-4 generated bboxes and output a coherent video.

##### 5.1.3 Ablation analysis

Ablation on individual attention masks. A SpatioTemporal attention block consists of three types of attention layers– Spatial, Cross and Temporal. PEEKABOO applies masking on all three layers, however, the effect of each

mask on the generation quality is different. In this section, we experiment with PEEKABOO by disabling masking for each attention layer one-by-one. We evaluate the AP50 score for ModelScope and ZeroScope on the IMC dataset, as shown in Figure 4. The performance drops massively when any one of the attention mask is not provided. We observe that not passing MCA hurts the control the most. This is explained by the fact that main object’s text token will not focus its attention at the bbox location, leading to the object being generated at a different location. Surprisingly, not passing MTA is worse than not passing MSA. We conjecture that removing spatial attention mask leads to degraded videos, while removing the temporal attention mask leads to the loss of temporal control. Since the latter model still generates higher quality objects at incorrect locations, it has a lower AP50 score. We notice that the Coverage of the model after removing MSA is much less than the Coverage of the model after removing MTA, providing evidence in support of our hypothesis.

Performance on varying Bbox sizes. PEEKABOO can accommodate varying bbox sizes within videos. Fig 3 shows qualitative control on input bboxes with the output. Moreover, SSv2-ST dataset contain examples with varying bbox sizes. Thus, we further quantify our SSv2-ST results in Fig 4(b) by plotting mIoU scores against varying box sizes across time in videos. For each prompt of SSv2-ST, we compute the standard deviation of the bbox size (relative to the mean input) to highlight the variation across time. We observe a consistent improvement by PEEKABOO on base model independent of box size variation.

#### 5.2. Qualitative Results

In Figure 1, we present examples of videos generated by PEEKABOO applied on ZeroScope [38]. As demonstrated, the videos follow the bbox input. Through these qualitative results, we highlight the versatility of bbox input in capturing the shape, size, location and motion, and show how our method can utilize this information interactively.

Static spatial control. Figure 3 shows videos where the object is statically located in the frame. Our method can control the position of the object, and can also change the size of the object as specified by the user through a bbox.

Dynamic spatial control. Figure 5 present videos where the main subject is moving on a desired path. Our method generated realistic looking movements for various motion trajectories. The temporal masking of our method also enables it to handle cases where the mask disappears mid-way through the scene, as is the case in the first row in Figure 1, while the spatial and cross-attention masking ensures spatial coherence of the generated frames with the input bounding boxes.

Overcoming model failures. Diffusion models can have a bias on their generation capabilities depending on their

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

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- Figure 5. PEEKABOO with a moving mask: As demonstrated, our method can mimic the input mask trajectories to generated spatiotemporally controlled videos with realistic motions. For e.g. in the last row, the wolf is jumping following the mask on the left.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- (a) A spider descending on its web.

[Figure 52]

[Figure 53]

- (b) A croissant on a wooden table.

- Figure 6. Overcoming model failures: Frames on the left are generated by zero-scope, and frames on the right are generated by PEEKABOO. Inset in the first row are cross-attention map between the word “spider” and the pixels in the video frame. We can generate objects that are otherwise omitted from the video by the base model. The attention maps also show that explicit masking leads to better generation. The second row depicts a numeracy failure of the baseline where PEEKABOO can control the number of objects.

by forcing the model to generate foreground object at a specific location. In Figure 6, we present results of prompts where the original model fails to produce the foreground object however, our method can produce the object in the user specified location and motion. The inset figures in Figure 6 reveal the reason for this – while the cross-attention corresponding to the word “spider” is diffused across the entire canvas in the original model, PEEKABOO focuses this attention on the desired region. Further, Figure 6 depicts the example of hallucination by generation model where the subject was generated multiple times. Again, PEEKABOO solves this issue due to spatial-attention mask and crossattention at a specific location.

### 6. Conclusion

In this work, we explore interactive video generation. We hope that this work will inspire more research in this area. To this end, we propose a new benchmark for this task and PEEKABOO, which is a training-free, no latency overhead method to endow video models with spatio-temporal control. Future work involves exploring PEEKABOO for image-to-video generation, video-to-video generation and long form video generation.

training data. However, we observe that PEEKABOO can suppress those biases and produce high quality generation

### References

- [1] Aishwarya Agarwal, Srikrishna Karanam, K J Joseph, Apoorv Saxena, Koustava Goswami, and Balaji Vasan Srinivasan. A-star: Test-time attention segregation and retention for text-to-image synthesis, 2023. 2, 3
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning,

2022. 3

- [3] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. arXiv preprint arXiv:2302.08113, 2023. 5
- [4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 1, 2

- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing, 2023. 2, 3, 4
- [6] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 7
- [7] Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, TsungYi Lin, and Ming-Hsuan Yang. Motion-conditioned diffusion model for controllable video synthesis, 2023. 2
- [8] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models,

2023. 2

- [9] Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In NeurIPS, 2021. 4
- [10] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation, 2022. 4
- [11] Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation, 2023. 2, 3
- [12] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models,

2023. 2

- [13] Heng Fan, Hexin Bai, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Harshit, Mingzhen Huang, Juehuan Liu, Yong Xu, Chunyuan Liao, Lin Yuan, and Haibin Ling. Lasot: A high-quality large-scale single object tracking benchmark,

2020. 6

- [14] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 5, 13
- [15] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models, 2022. 2
- [16] J Ho, T Salimans, A Gritsenko, W Chan, M Norouzi, and DJ Fleet. Video diffusion models. arxiv 2022. arXiv preprint

- arXiv:2204.03458, 2022. 2

[17] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint

- arXiv:2205.15868, 2022. 2, 6

- [18] Yaosi Hu, Zhenzhong Chen, and Chong Luo. Lamd: Latent motion diffusion for video generation, 2023. 2
- [19] Hanzhuo Huang, Yufan Feng, and ChengShi LanXu JingyiYu SibeiYang. Free-bloom: Zero-shot text-to-video generator with llm director and ldm animator. arXiv preprint arXiv:2309.14494, 3, 2023. 2
- [20] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators, 2023. 1, 2
- [21] Feng Li, Hao Zhang, Huaizhe xu, Shilong Liu, Lei Zhang, Lionel M. Ni, and Heung-Yeung Shum. Mask dino: Towards a unified transformer-based framework for object detection and segmentation, 2022. 4
- [22] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation, 2023. 2, 3
- [23] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llmgrounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models,

2023. 2, 5

- [24] Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models. arXiv preprint arXiv:2309.17444, 2023. 2, 6
- [25] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning, 2023. 5
- [26] Farzaneh Mahdisoltani, Guillaume Berger, Waseem Gharbieh, David Fleet, and Roland Memisevic. On the effectiveness of task granularity for transfer learning. arXiv preprint arXiv:1804.09235, 2018. 2, 5, 13
- [27] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In European Conference on Computer Vision, pages 728–755. Springer, 2022. 5, 13

- [28] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2
- [29] John Mullan, Duncan Crawbuck, and Aakash Sastry. Hotshot-XL, 2023. 3
- [30] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Computer Vision and Pattern Recognition, 2016. 6
- [31] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing, 2023. 2, 3
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1, 3
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 2, 11
- [34] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 2, 3
- [35] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data, 2022. 1, 2, 3
- [36] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 3
- [37] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [38] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1, 2, 3, 6, 7
- [39] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability, 2023. 2
- [40] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation, 2023. 1
- [41] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu, Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, and Forrest Iandola. Cvpr 2023 text guided video editing competition, 2023. 2
- [42] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language.

- In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5288–5296, 2016. 7
- [43] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation, 2023. 2
- [44] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models, 2022. 2, 6
- [45] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15116–15127,

2023. 4

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

(a) A kite flying in the sky. (b) A ship docked at a port.

- Figure 7. Text to Image synthesis: We augment Stable-Diffusion v2.1 with PEEKABOO to produce images with spatial control including the size and location of the objects. Inset images are the masks passed to the model. Best viewed when zoomed in.

### A. Text to image synthesis

While PEEKABOO was designed for video synthesis, it can be easily modified and work for the task of Text-to-Image synthesis. Figure 7 shows the versatility of our method. We generate images using Stable-Diffusion v2.1 [33] and gained spatial control through PEEKABOO. We observe that for the same prompt and initialization seed, PEEKABOO is able to control the location of the subject making the generation process interactive. Please refer to appendix for more results.

### B. Implementation details

ModelScope - We generate videos of 256x256 resolution, and 16 frames. We fix the fixed step t to be 2 for all generations for ssv2-ST and 4 for IMC generation, and diffusion steps to be 40. For numbers on IMC, we generate 24 frames. In the quality evaluation experiments Table 2, we re-evaluated ModeScope performance on our selected set of prompts from the MSR-VTT dataset. PEEKABOO generation results are for videos generated with fixed step t equal to 2 of 40 steps.

ZeroScope - We generate videos of 320 x 576 resolution, and 24 frames. We fix the fixed step t to be 2 for all generations for ssv2-ST and 4 for IMC generation, and diffusion steps to be 40.

### C. Ablation studies C.1. Sensitivity to t

In fig 8, we present results on varying the t parameter for generation on the IMC dataset. As t increases, AP50 increases, but coverage decreases. We hypothesize that this is because of selective attention masking for longer steps affecting the diffusion process. As the base models, were not trained with selective attention the model output quality degrades with larger number of steps.

Table 3. Ablation study on PEEKABOO : We evaluate ModelScope without various attention masks on our user defined dataset. We find that each component of our method impacts the performance significantly

Model mIoU % (↑) Coverage % (↑) CD (↓) AP50 % (↑) ModelScope+PEEKABOO 36.1 96.6 0.13 33.3

- -w/o Cross Attn Mask 14.2 93.3 0.27 5.7
- -w/o Self Attn Mask 19.5 87.7 0.30 16.5
- -w/o Temp Attn Mask 19.7 96.6 0.25 9.9 ZeroScope+PEEKABOO 36.3 96.3 0.12 33.8

- -w/o Cross Attn Mask 13.3 78 0.23 6.7
- -w/o Self Attn Mask 25.6 83 0.21 28.7
- -w/o Temp Attn Mask 25.1 91 0.18 18.6

#### C.2. More results on masking

We supplement the results of Fig 4 with a full results table in Tab 3. As we can see from AP50, Cross Attn mask is responsible for the majority of the control as it determines the location of the foreground object in the canvas. While Self Attn mask is responsible for maintaining generated video quality as observed by Coverage scores.

### D. Effect of subjects on video generation

Evaluating quality across foreground subject depends heavily on the base model’s training data. However, as we show in Fig 6, using masked diffusion in PEEKABOO can overcome model failures to some extent. We further evaluate the effect of subjects on controlled video generation by PEEKABOO against LaSOT dataset. Fig 9 shows that PEEKABOO overcomes the base model’s training bias and improves the control in video generation. Interestingly, PEEKABOO performance is better for some subject classes than other, perhaps it is a factor of disentanglement of CLIP embeddings used in cross-attention. Studying the cause of this imbalance warrants further research and we encourage the community to do so.

### E. More videos

We have uploaded the videos of the results presented in the main paper with our supplementary material and are available on the project webpage. Please use VLC media player for viewing the videos. We also append more video results in the Fig 10 for the reader.

### F. Dataset Curation and filtering F.1. IMC

In this section, we provide finer details of generating IMC dataset.

F.1.1 Prompts We provide the list of prompts used in IMC dataset.

ModelScope Sensitivity

50

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

40

30

AP50

20

10

0

0 1 2 3 4 5 6 Frozen Steps

ZeroScope Sensitivity

50

40

30

AP50

20

10

0

0 1 2 3 4 5 6 Frozen Steps

- Figure 8. Sensitivity to frozen steps: We plot AP50 against number of frozen steps t for ModelScope and ZeroScope. The radius of the marker is proportional to the coverage. We find that increasing t increases AP50 at the risk of losing coverage, i.e. degrading quality.

tankcattlesquirrelgiraffeelectricfancrabfrogtigerspiderfoxbasketballbearrabbitcrocodileairplanezebrahippodeerlizardhorserobotkiteleopardelephantlionkangaroodroneyoyocatturtlesurfboardmousedoggorillagoldfishlicenseplateumbrellacoinbirdtrainflag

Subject

0.0

0.1

0.2

0.3

0.4

0.5

Meaniou

Modelscope w/ Peekaboo

Modelscope

LLM-VD

- Figure 9. Effect of subjects on video generation quality: We plot mIoU on LaSOT against different subjects present in the dataset. We observe that PEEKABOO either retains or improves generation quality for majority of subjects.

- • A woodpecker climbing up a tree trunk.
- • A squirrel descending a tree after gathering nuts.
- • A bird diving towards the water to catch fish.
- • A frog leaping up to catch a fly.
- • A parrot flying upwards towards the treetops.
- • A squirrel jumping from one tree to another.
- • A rabbit burrowing downwards into its warren.
- • A satellite orbiting Earth in outer space.
- • A skateboarder performing tricks at a skate park.
- • A leaf falling gently from a tree.
- • A paper plane gliding in the air.
- • A bear climbing down a tree after spotting a threat.

- • A duck diving underwater in search of food.
- • A kangaroo hopping down a gentle slope.
- • An owl swooping down on its prey during the night.
- • A hot air balloon drifting across a clear sky.
- • A red double-decker bus moving through London streets.
- • A jet plane flying high in the sky.
- • A helicopter hovering above a cityscape.
- • A roller coaster looping in an amusement park.
- • A streetcar trundling down tracks in a historic district.
- • A rocket launching into space from a launchpad.
- • A deer standing in a snowy field.
- • A horse grazing in a meadow.

- • A fox sitting in a forest clearing.
- • A swan floating gracefully on a lake.
- • A panda munching bamboo in a bamboo forest.
- • A penguin standing on an iceberg.
- • A lion lying in the savanna grass.
- • An owl perched silently in a tree at night.
- • A dolphin just breaking the ocean surface.
- • A camel resting in a desert landscape.
- • A kangaroo standing in the Australian outback.
- • A colorful hot air balloon tethered to the ground.

##### F.1.2 Generating the bounding Boxes

Given the set of prompts, we annotate the main subject in the prompt. Further, the prompts are classified as stationary/moving, along with the object’s aspect ratio as square, vertical rectangle, or horizontal rectangle. Specifically, the aspect ratio values are 1 : 1, 4 : 3, 3 : 4 respectively. For prompts with movement, we also classify movement into up/down, left/right or zig-zag.

Three sets of bounding boxes are generated for each prompt. The starting co-ordinate of the bounding box is chosen randomly from 9 centroids of a 3x3 grid that the canvas is divided into. The speed is randomly chosen from 5-20 for moving prompts. The movement direction is randomly flipped as well. The bonding box size is chosen as 0.25 or 0.35 of the canvas size. We then generate a bounding box for each frame according to the random parameters, adding a small jitter for each pixel is well. For moving prompts, the starting location is one of 6 centroids, omitting the centroids which align with the direction of motion. We will release the code for generating this dataset as well.

#### F.2. ssv2-ST

Filtering - We use Something-Something v2 dataset [14, 26] to obtain the generation prompts and ground truth masks from real action videos. We filter out a set of 295 prompts. The details for this filtering are in the appendix. We then use an off-the-shelf OWL-ViT-large open-vocabulary object detector [27] to obtain the bounding box annotations of the object in the videos. This set represents bounding box and prompt pairs of real-world videos, serving as a test bed for both the quality and control of methods for generating realistic videos with spatio-temporal control. We filter out the prompts such that they contain a single foreground object and obtain the bounding boxes or masks for the videos. We also further filter out videos with 0 bounding boxes.

Post-processing bounding boxes - We downsample videos in SSv2 to 5fps and 224x224 resolution. For each video, we consider the first 24 frames for computing bounding boxes. We use OwL-ViT/B16 for getting the bounding boxes of the first 24 frames. Due to frame jittering and low

resolution, we observe that obtained masks were not consistently calculated for each frame. Hence, we interpolated the masks between two successive frames. Our final test set contains 295 prompts and masks pairs. We pass the first 16 of these boxes to ModelScope, and all 24 of them to ZeroScope

### G. Limitations

In Fig 11, we depict three typical failure modes of our method. These usually happen because there is a mismatch between the prior and the input mask, ie., the bounding boxes should be of sensible size that align with the training data of the model. Further, the generation usually fails for cases where the base model is bad at the target prompt. Moreover, the movement introduced through interactive control should align with the input text prompt.

### H. Societal Impact

This is a work on controllable video generation and not video generation itself. It is possible that the base model itself reflects some societal biases of the training set which will be propagated with the work. It also inherits the potential for misuse that other such video generation works have.

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

[Figure 79]

###### Figure 10. More interactive video generation results

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- (a) A school of fish in the ocean.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

- (b) A rocket launching into space.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

(c) A grand piano in a hall.

Figure 11. Our Failure modes: Top row shows a failure mode because the mask is too small for the subject. Middle row shows a failure model where the object does not move much, since the direction of motion of the mask contradicts that of the text. Bottom row shows a case where the model inherits a bad generation of the base model.

