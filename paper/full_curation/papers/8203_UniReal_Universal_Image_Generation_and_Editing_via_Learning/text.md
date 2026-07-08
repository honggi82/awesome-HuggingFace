## UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics

# arXiv:2412.07774v2[cs.CV]11Dec2024

Xi Chen1 Zhifei Zhang2 He Zhang2 Yuqian Zhou2 Soo Ye Kim2 Qing Liu2 Yijun Li2 Jianming Zhang2 Nanxuan Zhao2 Yilin Wang2 Hui Ding2 Zhe Lin2 Hengshuang Zhao1

1The University of Hong Kong 2Adobe

|[Figure 1]<br><br>[Figure 2]<br><br>Source Image Heavy snow falls on the can. The night comes, add a candle beside the can. Remove the can. Replace the can with a bunch of roses.<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>|RES|
|---|
<br><br>|RES|
|---|
<br><br>|RES|
|---|
<br><br>|RES|
|---|
<br><br>|SOURCE|
|---|
|
|---|

|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>The woman from IMG1 is holding the dog from IMG2.<br><br>|IMG1|
|---|
<br><br>|IMG2|
|---|
<br><br>|RES|
|---|
|
|---|

|[Figure 9]<br><br>[Figure 10]<br><br>The toy from IMG1 is dancing with sunglasses.<br><br>|IMG1|
|---|
<br><br>|RES|
|---|
|
|---|

|[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>The wolf toy from IMG1 at the position of IMG3, and the duck toy of IMG2 at the position of IMG4 are placed before the gold pillows.<br><br>[Figure 15]<br><br>|IMG1|
|---|
<br><br>|IMG2|
|---|
<br><br>|IMG3|
|---|
<br><br>|IMG4|
|---|
<br><br>|RES|
|---|
|
|---|

|[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>Add the dog from IMG1 in the water of IMG2, the dog is swimming.<br><br>|IMG1|
|---|
<br><br>|IMG2|
|---|
<br><br>|RES|
|---|
|
|---|

|[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>The character from IMG1 is drawn on the paper of IMG2 in sketch.<br><br>|IMG1|
|---|
<br><br>|IMG2|
|---|
<br><br>|RES|
|---|
|
|---|

|[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>Make the vase yellow color; Generate the edit region mask in RES2.<br><br>|IMG1|
|---|
<br><br>|RES1|
|---|
<br><br>|RES2|
|---|
|
|---|

|[Figure 25]<br><br>[Figure 26]<br><br>Predict the depth map of this image.<br><br>|IMG1|
|---|
<br><br>|RES|
|---|
|
|---|

Figure 1. Demonstrations of UniReal’s versatile capabilities. As a universal framework, UniReal supports a broad spectrum of image generation and editing tasks within a single model, accommodating diverse input-output configurations and generating highly realistic results, which effectively handle challenging scenarios, e.g., shadows, reflections, lighting effects, object pose changes, etc.

### Abstract

We introduce UniReal, a unified framework designed to address various image generation and editing tasks. Existing solutions often vary by tasks, yet share fundamental principles: preserving consistency between inputs and outputs while capturing visual variations. Inspired by recent video generation models that effectively balance consistency and variation across frames, we propose a unifying approach that treats image-level tasks as discontinuous

video generation. Specifically, we treat varying numbers of input and output images as frames, enabling seamless support for tasks such as image generation, editing, customization, composition, etc. Although designed for imagelevel tasks, we leverage videos as a scalable source for universal supervision. UniReal learns world dynamics from large-scale videos, demonstrating advanced capability in handling shadows, reflections, pose variation, and object interaction, while also exhibiting emergent capability for novel applications. The page of this project is here.

### 1. Introduction

The field of visual content creation has advanced significantly with the development of diffusion models [17, 46], enabling a broad range of applications in image generation [5, 20, 36, 40, 70] and editing [1, 10, 68, 69]. However, as practical demands increase, the applications are becoming progressively specialized in both tasks and methods. This limits the ability to learn generalizable knowledge across fields and increases the workload for designing taskspecific methods and collecting domain-specific data. In this work, we explore designing a universal framework that could unify diverse tasks into a generalized formulation along with a scalable training paradigm.

Across diverse image generation and editing tasks, we observe shared core requirements, such as preserving consistency between input and output images while bringing in controlled visual variation. These requirements guide our design of a universal framework. Notably, video generation models like Sora type methods [18, 38, 41, 67, 73] effectively balance frame consistency with motion variation, aligning closely with our goals. Thus, we adapt their design principles for image-level tasks, reformulating various image generation and editing tasks into a unified framework for “discontinuous” frame generation. Specifically, we introduce UniReal, a universal solution tackling different tasks within one diffusion transformer. UniReal builds on the foundational structure of video generation models, using full attention to model relationships across frames.

UniReal treats varying numbers of input and output images as pseudo-frames, guided by a text prompt to support a wide range of applications. To unify input image types in multiple tasks, e.g., text-to-image generation, controllable generation, instructive editing, and multi-subject customization, we distinguish input images as three pivots:

- 1) the target image to edit on, 2) reference image that carries objects or visual elements to insert or preserve, and 3) condition map for layout or shape regularization. To coordinate multiple input images under a unified text prompt, we introduce image index embeddings that link each image to its corresponding prompt term. For task synergy, we design a hierarchical prompting scheme, layering context-level and image-level guidance onto the base prompt. Together, these techniques enable UniReal to seamlessly integrate and compose diverse tasks.

Instead of curating task-specific data, we explore seeking universal supervision. We find that many types of variations (e.g., add, remove, attribute changes, structural changes) are naturally covered between two discontinuous video frames. In this way, we leverage large-scale frame pairs with captions as instructive editing data, and build an automatic pipeline to construct data from videos for image customization and composition tasks.

As shown in Fig. 1, UniReal exhibits superior capabili-

ties for maintaining consistency and preserving the details with the input images. Furthermore, it demonstrates strong potential for modeling natural variations and simulating world dynamics such as lighting, reflections, and object interactions. In addition, after learning versatile tasks, UniReal shows some emerged abilities for novel applications without training data.

### 2. Related Work

Diffusion-based image editing. Image editing is a general and practical topic that covers a lot of applications. Initial works [3, 16, 23, 34, 35] explore training-free or tuning-based solutions to edit the attributes of the source image with prompts. Later, InstructPix2Pix [1] and other works [21, 66, 69, 72] train instructive image editing on constructed datasets. Meanwhile, another branch of work explores mask-based editing, e.g., regenerate the regions covered by the given mask [22, 76] or compose the region from one image to another [9, 10, 50, 51, 64]. Customized generation task [26, 27, 31, 32, 47] requires one or multiple reference images as input to generate new images with the same subject. Besides, there are also application-oriented tasks like virtual try-on [6, 25, 63], face personalization [28, 57, 71], image stylization [49, 56], etc. In general, we observe that the settings and solutions for image editing tasks vary significantly. However, there are still common requirements like keeping the consistency with the input images and modeling the visual variations according to specific conditions. In this work, we explore designing a universal framework and constructing generalizable supervision supporting versatile tasks.

Universal generative model. The omni solution for different image generation/editing tasks is a challenging topic. SEED-X [14] and Emu2 [53] use an autoregressive model to predict the next tokens and use a separated diffusion model to generate the images. Tranfusion [74] predicts discrete tokens to for text and uses continuous vectors to represent images. Show-o [61] leverages causal attention to process text tokens and uses full attention for image tokens. VILA-U [59] adopts discrete tokens for both text and image and leverages different decoders for different modalities. Nevertheless, as they mainly focus on understanding tasks, their image generation/editing abilities are not quite satisfactory as a side product. Some works explore a unified solution focusing on image generation and editing. InstructImagen [19] unifies image generation tasks together using multi-modal instructions. Pixwizard [29] proposes task embeddings for image editing and understanding tasks. OmniGen [60] tokenizes the texts and images as a long tensor, using causal attention for text tokens and bidirectional attention for image tokens. ACE [15] designs the conditioning unit to receive different kinds of input images and uses a transformer to deal with multiple inputs.

Result

[Figure 27]

|N × Transformer Blocks<br><br>[Figure 28]<br><br>Text Input Image Noise|
|---|

[Figure 29]

| | |
|---|---|
| | |

VAE Decoder

[Figure 30]

[Figure 31]

Iterative Denoising

Diffusion Transformer with Full Attention

Embed Embed Embed

Timestep Embed

| |+ Pos|
|---|---|
|[Figure 32]<br><br>IMG3| |

| |+ Pos|
|---|---|
|[Figure 33]<br><br>IMG2| |

| |+<br><br>+ Pos|
|---|---|
|[Figure 34]<br><br>RES1| |

| |+ Pos|
|---|---|
|[Figure 35]<br><br>IMG1| |

Embed

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

T5 Text Encoder

Index Embed

…

|[Figure 40]<br><br>Control|
|---|

Realistic style; Image Prompt Novel scenario; With reference object;

|[Figure 41]<br><br>Asset|
|---|

|[Figure 42]<br><br>Asset|
|---|

The dog from IMG1 is drinking the milk in the bowl of IMG2 at the position of IMG3 on a grassland.

+ Context Prompt Base Prompt

+ IMG 2 IMG 3 Noise

+

+

+

[Figure 43]

[Figure 44]

[Figure 45]

…

VAE Encoder

VAE Encoder

VAE Encoder

Visual Tokens

IMG 1

- Figure 2. Overall pipeline of UniReal. We formulate image generation and editing tasks as discontinuous frame generation. First, input images are encoded into latent space by VAE encoder. Then, we patchify the image latent and noise latent into visual tokens. Afterward, we add index embeddings and image prompt (asset/canvas/control) to the visual tokens. At the same time, the context prompt and base prompt are processed by the T5 encoder. We concatenate all the latent patches and text embeddings as a long 1D tensor and send them to the transformer. Finally, we decode the denoised results to get the desired output images.

### 3. Method

tokens for the T5 tokenizer. At the same time, we learn image index embeddings for each referring word, which are added to the tokens of the corresponding image.

UniReal leverages a video generation framework that processes arbitrary numbers of input and output images as video frames, naturally unifying multiple image generation and editing tasks. Furthermore, UniReal seeks universal supervision for these tasks from scalable video data, achieving highly realistic generation results. We will detail architecture design (Sec. 3.1), data construction (Sec. 3.2), and training schemes (Sec. 3.3).

Hierarchical prompt. Different tasks/datasets handle the same input differently. For example, image editing keeps the layout of the input image and makes local changes. However, with the same prompt, image customization generates a novel scenario and only preserves the reference object. This introduces the ambiguity for both the training and inference phases. To reduce the ambiguity when mixing multiple tasks and data sources, we propose hierarchical prompt. Besides the base prompts like “put this dog on a grassland”, we design additional context prompts and image prompts to provide detailed indications.

#### 3.1. Model Design

This section will dive into the key designs of UniReal, i.e., text-image association and hierarchical prompt based on diffusion transformer.

Context prompt provides attribute tags for different tasks and data sources, like “realistic/synthetic data”, “static/dynamic senario”, “with reference object”, etc. Different from task embeddings that are used in previous works [29, 48], some of our “keywords” can be shared among tasks, forcing those tasks to learn common features. Besides, as texts are naturally composable, we can easily compose different context prompts to realize novel functions.

Diffusion transformer. As illustrated in Fig. 2, UniReal treats input/output images as video frames and uses prompts to manage different tasks. Specifically, images are encoded to latent space by the VAE encoder, then those latent maps are Upon visual tokens, we add index embeddings to distinguish the image order and add the image prompt to indicate whether an image serves as, e.g., canvas/background or asset/foreground object. Fig. 8 illustrates the effect of image prompt. The position embeddings are added to each image/noise token, and the timestep embeddings are added to the noise tokens. At the same time, the text prompts are sent to a T5 [44] encoder to extract text tokens. We concatenate the image and noise tokens along with the text tokens as a long 1D tensor and send it to a transformer. The transformer uses full attention to model the relationship between images and the text prompt.

The image prompt indicates the specific role of input images. We split input images into three categories: canvas image, asset image, and control image. The canvas image serves as a background for the editing target with a fixed layout. The asset image provides the reference objects or visual elements for image customization or composition, for which the model should implicitly conduct segmentation, and simulate the size/position/pose changes for the object. The control image includes the mask/edge/depth map that regularizes the layout or shape. The model should take distinctive actions for images of different categories. Therefore, we design learnable category embeddings and add them to the corresponding image tokens as “image prompt”.

Text-image association. To refer to specific images in the text prompt, we construct a set of embedding pairs to associate the visual tokens with corresponding texts. Specifically, we use referring words like “IMG1” and “IMG2” to refer to the input images, and “RES1” and “RES2” for the output images. We add them as special

During inference, the context prompt and image prompt

Table 1. Statistics of datasets used for training. We mix the existing datasets (the first block) with our newly constructed video-based datasets (the second block).

|Video| | |
|---|---|---|
| | | |
| | | |

|Two Frames| |
|---|---|
| | |

|Grounding Caption|
|---|

|Video Caption| |
|---|---|
| | |

|Image Perception| |
|---|---|
| | |

|Video Segmentation|
|---|

|Two Frames + Raw Caption| |
|---|---|
| | |

No. Dataset # Samples Supporting Tasks

|MLLM|
|---|

|Frames + Masks + Image & Region Captions|
|---|

|Frames + Condition maps| |
|---|---|
| | |

- 1 InstructP2P [1] 300K Universal Instructive Editing
- 2 UltraEdit [72] 500K Universal Instructive Editing
- 3 VTON-HD [11] 10K Virtual Try-on
- 4 RefCOCO Series [24, 33] 150K Referring Segmentation
- 5 T2I Generation (in house) 300M Text to Image Generation
- 6 Instruct Editing (in house) 2M Universal Instructive Editing
- 7 Object Insertion (in house) 100K Object Insertion with Reference

- 8 Video Frame2Frame 8M Universal Instructive Editing
- 9 Video Multi-object 5M Multi-subject Customization
- 10 Video Object Insertion 1M Object Insertion with Reference
- 11 Video ObjectAdd 1M Object Insertion with Prompt
- 12 Video SEG 5M Referring Segmentation
- 13 Video Control 3M Perception, Controllable Generation

|Two Frames + Fine Instructions|
|---|

Image Customization Local Inpainting Object Insertion

Image Perception Controllable Generation

Universal Instructive Image Editing

|Frame1|
|---|

|Frame2|
|---|

|Frame1|
|---|

|Frame2|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Aboy and a girl jumping on a bed.

Add a person sitting on a swing, facing the mountains, with their back to the viewer.

- Figure 3. Data construction pipeline. Starting from raw videos, we use off-the-shelf models to construct data for different kinds of tasks. Two examples of instructive editing and image customization data (we segment objects from one frame to generate another frame) are given at the bottom of the image.

thus supporting controllable image generation and image perception (Video Control).

Training data overview. The datasets used for training are listed in Tab. 1. Besides our constructed video-based datasets, we also use open-source data [1, 11, 24, 33, 72] for specific tasks and our in-house datasets for instructive image editing and reference-based object insertion. Considering the difficulties of constructing image editing data, the public datasets are limited. Differently, our video-based data are easier to scale up.

can be automatically analyzed from the base prompt with a default value. Thus, UniReal does not require extra effort on writing prompts. Nevertheless, users can manually revise the task and image prompt to get more novel effects.

#### 3.2. Dataset Construction

Some previous works [12, 21, 66, 72] leverage complicated workflows to collect data for specific tasks. Differently, we start with video data and explore leveraging the natural consistency and variation between video frames to benefit various image generation and editing tasks.

As explained in Sec. 3.1, it is crucial to incorporate these datasets with context prompts. For example, in some videos from Video Frame2Frame, there exist global changes (e.g., background movements and camera motion) that are not captured by the instruction, which is not favored in instructive editing. In these cases, we analyze the optical flow and pixel MSE between frames to label each sample with “static/dynamic scenario”. Besides, the instructive editing datasets [1, 72] are in synthetic style, we tag them as “synthetic style” and give the context prompt of “realistic style” for real-image datasets. In addition, we give context prompts like “with reference objects” for Video Object Insertion, and give “perception task” when training the model to predict masks or depth maps, etc.

Data construction pipline. Fig. 3 illustrates the data construction pipeline from raw videos to support various tasks. We first use a caption model to get the videolevel captions. Then, we randomly pick two frames as before-/after-editing images and use video-level captions as instruction. We term this kind of data Video Frame2Frame, and we observe that this data alone is already able to train a model with basic editing abilities. We further use GPT4o mini [37] to get more precise instructions between two frames for a subset with 200K high-quality samples.

Besides, we use the grounding caption model, Kosmos-

- 2 [39] to generate image captions with the bounding boxes for the corresponding entities. Then, we use the boxes from one frame as prompts for SAM2 [45] to get the mask tracklet for the two frames. In this way, we get the captions and masks for multiple objects in two video frames. This kind of data can support image customization (Video Multiobject), object insertion (Video Object Insertion), local inpainting (Video ObjectAdd), ect. For example, for image customization, we segment each object from one frame and use them as reference images to generate another frame according to the caption. In addition, we reuse the masks and captions labeled by Kosmos-2 to support referring segmentation (Video SEG). We also leverage image perception models [2, 65] to extract the depth map and the edge map,

#### 3.3. Training schemes

Our transformer model has 5B parameters. It is first pretrained with text-to-image and text-to-video data to get the basic generation ability under a small resolution of around 256×256. Then, we train the model on all the datasets listed in Tab. 1 to learn multiple image generation/editing tasks (256 resolution). Afterward, we progressively increase the resolution to 512 and 1024. As we apply position embeddings for the image patches and the training images have different aspect ratios, UniReal could deal with different sizes and aspect ratios. We apply the learning rate of 1e-5 with a warm-up for each training stage. The training loss follows flow matching [30].

###### Source Image CosXL

###### Prompt

###### Ours OmniGen UltraEdit MGIE InstructP2P

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Add an elephant in the pool.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Remove the duck toy.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Small ants lift up the car.

- Figure 4. Comparison results for instructive image editing. We compare with the state-of-the-art methods OmniGen [60], UltraEdit [72], MGIE [13], InstructPix2Pix [1], and CosXL [52]. Our UniReal shows significant advantages in the aspects of instruction-following and generation quality. We generate multiple results for each model and pick the best ones for demonstration.

Table 2. Comparison results for instructive image editing on EMU Edit [48] and MagicBrush [69] test sets. We list the taskspecific models in the first block and some concurrent universal models in the second block.

EMU Edit Test set MagicBrush Test Set Method CLIPdir ↑ CLIPim↑ CLIPout↑ L1↓ DINO↑ CLIPdir ↑ CLIPim↑ CLIPout↑ L1↓ DINO↑

InstructPix2Pix [1] 0.078 0.834 0.219 0.121 0.762 0.115 0.837 0.245 0.093 0.767 MagicBrush [69] 0.090 0.838 0.222 0.100 0.776 0.123 0.883 0.261 0.058 0.871 PnP [55] 0.028 0.521 0.089 0.304 0.153 0.025 0.568 0.101 0.289 0.220 Null-Text Inv. [35] 0.101 0.761 0.236 0.075 0.678 0.121 0.752 0.263 0.077 0.664 UltraEdit [72] 0.107 0.793 0.283 0.071 0.844 - 0.868 - 0.088 0.792 EMU Edit [48] 0.109 0.859 0.231 0.094 0.819 0.135 0.897 0.261 0.052 0.879

ACE [15] 0.086 0.895 0.274 0.076 0.862 - - 0.284 - OmniGen [60] - 0.836 0.233 - 0.804 - - - - PixWizard [29] 0.104 0.845 0.248 0.069 0.798 0.124 0.884 0.265 0.063 0.876 UniReal (ours) 0.127 0.851 0.285 0.099 0.790 0.151 0.903 0.308 0.081 0.837

### 4. Experiments

#### 4.1. Comparisons with Existing Works

As a universal model, UniReal exhibits superior abilities for various image generation and editing tasks, even compared with existing task-specific models. We select three challenging tasks: instructive image editing, customized image generation, and object insertion as representatives to demonstrate and analyze the performance of UniReal.

Instructive image editing. Users provide a prompt to edit the input image in free forms, like adding/removing an object, changing the attributes or styles, etc. We demonstrate the qualitative comparisons in Fig. 4, where we make comparisons with several state-of-the-art public models [1, 13, 52, 60, 72]. UniReal shows a clear advantage for dealing with challenging cases like simulating the size and status of an elephant in the water and removing the duck toy along with the shadows. In the third row, UniReal successfully understands the interactions between the aunts and the car while modeling the car’s reflections.

Table 3. Quantitative results for customized generation on DreamBench [47]. We report the oracle results in the first row and compare both tuning methods and zero-shot methods.

Model CLIP-T↑ CLIP-I↑ DINO↑ Oracle (reference images) - 88.5 77.4

Textual Inversion [35] 0.255 0.780 0.569 DreamBooth [47] 0.305 0.803 0.668 BLIP-Diffusion [27] 0.302 0.805 0.670

ELITE [58] 0.296 0.772 0.647 Re-Imagen [7] 0.270 0.740 0.600 BootPIG [42] 0.311 0.797 0.674 SuTI [8] 0.304 0.819 0.741 OmniGen [60] (our test) 0.320 0.810 0.693 UniReal (ours) 0.326 0.806 0.702

The quantitative results on EMU Edit [48] and MagicBrush [69] test sets are reported in Tab. 2. CLIPdir evaluates the agreements between the changes of CLIP [43] text and CLIP image embeddings. CLIPout estimates the similarities between the editing results and the descriptions of the expected output. CLIPim, DINO, and L1 denotes the CLIP similarity, DINO [4] similarity, and L1 distance between the editing results and the source images. UniReal achieves the best performance for CLIPdir and CLIPout, and get the best CLIPim on Magicbrush test set. As we expect obvious changes between the output and input, DINO and L1 are not quite reasonable metrics. Considering our model accurately follows the instructions to make obvious edits, it is reasonable that the similarities between the editing results and the source image would be lower.

Customized image generation. Preserving the details of the reference object and following the novel text prompt simultaneously is a challenging task. In Fig. 5, we provide qualitative comparisons for customized image generation.

Reference

Prompt IP-Adapter

###### Ours OmniGen Emu2 BLIP-Diffusion ELITE

|IMG1|
|---|

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

The can from IMG1 is placed on a table full of fruits

|IMG1|
|---|

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

The cat from IMG1 is climbing the tree.

Ref. + Prompt Ours OmniGen Emu2 Ours OmniGen Emu2

Ref. + Prompt

|IMG1|
|---|

|IMG1|
|---|

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

|IMG2|
|---|

|IMG2|
|---|

The dog from IMG1 is running after the toy of IMG2 on the road. The dog from IMG1 and the dog from IMG2 are swimming in the water.

|IMG1|
|---|

[Figure 95]

|IMG1|
|---|

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

|IMG2|
|---|

|IMG2|
|---|

The duck toy from IMG1 is placed in the bowl from IMG2. The sloth plushie from IMG1 and the wolf plushie from IMG2 are fighting with each other.

- Figure 5. Qualitative comparison for image customization. For single subject, we compare with OmniGen [60], Emu2 [53], BLIPDiffusion [27], ELITE [58], and IP-Adapter [68] with Flux [62] backbone. For multiple subjects, we chose OmniGen and Emu2 as competitors. The listed prompts are in the formats of UniReal, and they are formulated according to the requirements of each method.

We select examples from DreamBench [47] to conduct both single object customization (top row) and multiple subject compositions (bottom row). UniReal demonstrate significant advantages compared with other zero-shot models [27, 53, 58, 60, 68] from all-round perspectives. Our model could precisely maintain the fine details of the logos on the can (first row) and the berry bowl (last row). Besides, UniReal could handle drastic changes like making the cat climb the tree or letting the dogs swim. Meanwhile, our method can also accurately preserve details while modeling interactions between different objects.

The quantitative results on DreamBench [47] are reported in Tab. 3. CLIP-T measures the similarity between the generated images and the text prompts, CLIP-I and DINO measure image similarity between the generated and reference images. UniReal gets the best CLIP-T showing superior instruction-following ability. As some test prompts require to edit the object’s attribute, it is a trade-off between the text following and the generation fidelity. Even though, UniReal still gets competitive DINO and CLIP-I scores.

Reference-based object insertion. In Fig. 7, we compare UniReal with representative object insertion model, AnyDoor [10]. AnyDoor requires additional masks for both the reference object and target location. Differently, UniReal does not require any masks (it could take additional masks

100

Detail Preservation

Alignment

Quality

| |
|---|

| |
|---|

| |
|---|

PreferenceRate(%)

74 77

75

73

72 72

72

68 61

66

63

63

61

56

53

50

0 vs SuTI vs OmniGen vs UltraEdit vs OmniGen vs AnyDoor

(a) Image Customization (b) Editing (c) Insertion

Figure 6. Our preference rates against other methods evaluated by user studies. We compare SuTI [8], OmniGen [60], UltraEdit [72] and AnyDoor [10] for different tasks.

to specify the target regions) but we give the corresponding text prompt like “Add the dog from IMG1 to the swimming pool of IMG2.” In this figure, we choose challenging cases that require intensive status changes, and observe that it is hard for AnyDoor [10] to naturally put the dog into the water (row 1) or automatically adjust the view of the can (row 2). Besides, as in the bottom row, a practical feature is that UniReal could strictly preserve the background pixels like the hair without any segmentation.

User study. In Fig. 6, we report the human preference rates compared to each competitor. For image customization, we evaluate OmniGen [60] using 750 samples from DreamBench [47] and SuTI [8] using 50 samples from their paper. For instruct editing and object insertion, we conduct comparisons with 100 challenging, self-collected samples.

Reference Background

AnyDoor Ours ()

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

- Figure 7. Comparison results for object insertion. Our method could automatically adjust the status of the reference object according to the environment and strictly preserve the background. Our method does not require any mask as input.

[Canvas, Asset] [ Asset, Canvas] [Canvas] [ Asset]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

“Dynamic scene” “Static scene” “Synthetic” “Realistic”

Remove the backpack. Make the dog wear a red cap.

The dog from IMG1 is biting the teddy bear in IMG2. A duck toy from IMG1 with a blue house in the background.

IMG1

IMG2

IMG1

Source Source

- Figure 8. Effects of hierarchical prompt. The same input could correspond to various types of targets when given different image prompts (row 1) and context prompts (row 2).

We assign 10 annotators to evaluate three aspects: detail preservation, alignment, and image quality. Detail preservation assesses consistency with the reference object for image customization and object insertion, or with non-edited regions for instructive editing. Alignment measures how well the edit matches user intent, focusing on instructionfollowing for customization and editing, and background coherence for object insertion. Quality evaluates the accuracy and aesthetic appeal of the generated images. Overall, UniReal shows advantages over other methods in each aspect, with further details provided in the appendix.

#### 4.2. Analysis for the Core Components

We conduct a detailed analysis of the core designs of our framework from the model and data perspectives.

Hierachical prompt. As introduced in Sec. 3.1, we expand the base prompt with the image prompt and context prompt. We illustrate their effects in Fig. 8, where the same input images and text prompt could lead to different types of generation targets given different image prompts (first row) and context prompts (second row). With the design of hi-

###### Add Remove Replacement Customization Object Insertion

###### Stylization

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

###### FullInputPromptVideoDataData

[Figure 136]

The dog from IMG1 is walking on a street.

Add a Shiba Inu biting the teddy bear.

Remove the glass. Replace the car with a blue supercar.

Make it watercolor.

Add the car from IMG1 to the road of IMG2.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

###### ExpertData

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Figure 9. Ablation study for the training data. We visualize the results for models that are trained on Video Frame2Frame dataset, task-specific expert dataset, and our multi-task full dataset. It is impressive that the model trained only on video data could master many editing tasks (e.g., add, remove, attribute/pose changing), even for tasks with multiple input images.

Table 4. Quantitative studies for our basic components on MagicBrush [69] test sets, and DreamBench [47].

MagicBrush Test set DreamBench Method CLIPdir ↑ CLIPout↑ DINO↑ CLIP-T↑ CLIP-I↑ DINO↑

w/o Context Prompt 0.144 0.294 0.769 0.315 0.781 0.683 w/o Image Prompt 0.136 0.305 0.809 0.295 0.782 0.698 only Expert Data 0.139 0.310 0.788 0.309 0.790 0.708 UniReal-full 0.151 0.308 0.837 0.326 0.806 0.702

erarchical prompts, we could effectively reduce the training ambiguity when involving multiple tasks and multiple data sources. During inference, we could automatically expand the image and context prompt according to the base prompt. At the same time, users could revise the expanded prompts to better align their intentions.

Training data. In Fig. 9, we compare the model only training on Video Frame2Frame, on task-specific data (dataset1, 2, 6 from Tab. 1 for instructive editing, dataset9 for customization, dataset10 for object insertion), and on full data. We observe that the model trained solely on video data demonstrates a broad range of capabilities, including object add/remove, color editing, and image customization. Notably, although the video data contains only single input images, the trained model can handle multiple input images (not stable), such as reference-based object insertion shown in the last column. This highlights the promising potential of video data to serve as a form of universal supervision. Although the video data is powerful, it is still hard to cover all subtasks (e.g., image stylization) and sometimes fails to precisely understand the instructions. In this case, we still need the standard task-specific data to provide standard learning examples.

Quantitative analysis. The quantitative ablations are reported in Tab. 4. As the context prompt and image prompt

###### More Tasks with Training Supervision Novel Applications with Generalized Abilities

Text-to-image generation Controllable generation

Multi-object Insertion Insertion with Editing

[Figure 155]

RES RES

[Figure 156]

RES IMG1

[Figure 157]

- IMG1

- IMG2

[Figure 158]

[Figure 159]

[Figure 160]

RES RES

[Figure 161]

[Figure 162]

IMG1

IMG3

[Figure 163]

[Figure 164]

[Figure 165]

IMG2

Add the toy from IMG1 to IMG2

Add the toy from IMG1 and the backpack from IMG2 to the desk of IMG3.

+ Make it wave its hand. + Change it to green color.

Local Reference Editing Layer-aware Editing

Cinematic portrait of a woman on top of lighthouse, next to bright orange light, ocean, detailed face, fog, dark morning light, dark azure and orange colors.

The generation result follows the depth map of IMG1; The picture draws an ice-made candle that is lit.

[Figure 166]

RES RES RES

[Figure 167]

[Figure 168]

[Figure 169]

- IMG1

- IMG2

Reference-based outpainting

[Figure 170]

Reference-based inpainting

[Figure 171]

IMG1 IMG2

[Figure 172]

[Figure 173]

RES RES

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

- IMG1 IMG1

RES

RES RES RES1 RES2

[Figure 178]

- IMG1 IMG2

IMG1 IMG2

Transfer the hairstyle from IMG1 to the person in IMG2. Add a little boy + in front of the bed. + behind the bed.

Fill the hole in IMG2 using IMG1. Fill the hole in IMG2 using IMG1.

Object Moving Object Resizing

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

IMG1 RES IMG1 IMG2

[Figure 183]

Image Understanding Multiple Outputs

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Segment the head of this animal.

Predict the depth map of this image.

The dog from IMG1 is running beside the sea; Predict the mask of this dog in RES2.

Move the can to the left side of the image. Make the toy bigger, following the mask of IMG2.

Figure 10. More applications supported by UniReal. Our method supports a large range of applications. The left block demonstrates more tasks for which we constructed corresponding training data. The right block shows some zero-shot novel applications achieved by task composition and generalization.

could reduce the ambiguity for both training and inference, they contribute to the quantitative results.

We also observe that the full version with multi-task training could give stronger performance than the taskspecific model, which is consistent with Fig. 9. We analyze that the expert data for some specific tasks may not be sufficient enough to cover all the cases or generalize well, and even could cause overfitting. Besides, existing instructive editing datasets leverage synthetic data, which may cause generation artifacts. In this case, the realistic video data helps the model retain generation quality. In general, it is not easy to collect comprehensive data, even only for a specific task. Hence, even though not aligned in target, different tasks could help each other to complement the shortage of cases, so as to improve generalization capability.

#### 4.3. More Applications

In Fig. 10, the left block shows more abilities with training samples. The right part demonstrates some novel abilities.

Trained tasks. The first row demonstrates text-to-image and controllable generation, showcasing UniReal’s ability to handle various aspect ratios and resolutions, producing highly aesthetic content. The second row illustrates reference-guided inpainting [9, 54, 75], where a reference image fills the masked region of another image, preserving reference details while adapting to the target environment. The bottom row highlights UniReal ’s capabilities in basic perception tasks like referring segmentation, depth estimation, and generating multiple output images in a single pass. Novel Abilities. UniReal demonstrates generalization abil-

ities for unseen tasks. For instance, as shown in the first row, although trained on single-object insertion, it naturally supports multi-object insertion at inference. Additionally, it can combine tasks, such as performing object insertion with pose editing or color modification. The second row highlights the ability to transfer specific local features, like hairstyles, without any input masks. When an editing region is specified with a mask, UniReal can add objects to background layers while preserving foreground elements. The final row showcases object manipulation capabilities, including moving and resizing.

### 5. Conclusion

We introduce UniReal, a universal solution for a wide range of image generation and editing tasks. To handle varying numbers of input and output images, we employ a video generation framework that treats images as individual frames. Beyond task-specific data, we leverage universal supervision from video data to learn consistency and variation across images. UniReal achieves state-of-the-art performance in multiple image generation and editing tasks and demonstrates promising capabilities in understanding real-world dynamics and generalizing to new tasks.

Limitations. While UniReal theoretically supports any number of input and output images, stability decreases, and computation becomes intensive as the number of images exceeds five. Typically, 3-4 input images are sufficient for most applications. However, to support specific tasks requiring a large number of inputs or outputs, constructing training data with more images and exploring more efficient model architectures would be necessary.

### References

- [1] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2, 4, 5
- [2] John Canny. A computational approach to edge detection. TPAMI, 1986. 4
- [3] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023. 2
- [4] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 5
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024. 2
- [6] Mengting Chen, Xi Chen, Zhonghua Zhai, Chen Ju, Xuewen Hong, Jinsong Lan, and Shuai Xiao. Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment. In ECCV, 2024. 2
- [7] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-image generator. In ICLR, 2023. 5
- [8] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. In NeurIPS, 2024. 5, 6
- [9] Xi Chen, Yutong Feng, Mengting Chen, Yiyang Wang, Shilong Zhang, Yu Liu, Yujun Shen, and Hengshuang Zhao. Zero-shot image editing with reference imitation. In NeurIPS, 2024. 2, 8
- [10] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In CVPR, 2024. 2, 6
- [11] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In CVPR, 2021. 4
- [12] Wei Cong, Xiong Zheyang, Ren Weiming, Du Xinrun, Zhang Ge, and Chen Wenhu. Omniedit: Building image editing generalist models through specialist supervision. arXiv:2411.07199, 2024. 4
- [13] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. In ICLR,

2024. 5

- [14] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv:2404.14396, 2024. 2
- [15] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. Ace: Allround creator and editor following instructions via diffusion transformer. arXiv:2410.00086, 2024. 2, 5

- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. In ICLR, 2023. 2
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [18] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2023. 2
- [19] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, et al. Instruct-imagen: Image generation with multi-modal instruction. In CVPR, 2024. 2
- [20] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. In ICML,

2024. 2

- [21] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv:2404.09990, 2024. 2, 4
- [22] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In ECCV,

2024. 2

- [23] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 2
- [24] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014. 4
- [25] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In CVPR, 2024. 2
- [26] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 2
- [27] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. In NeurIPS, 2024. 2, 5, 6
- [28] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR,

2024. 2

- [29] Weifeng Lin, Xinyu Wei, Renrui Zhang, Le Zhuo, Shitian Zhao, Siyuan Huang, Junlin Xie, Yu Qiao, Peng Gao, and Hongsheng Li. Pixwizard: Versatile imageto-image visual assistant with open-language instructions. arXiv:2409.15278, 2024. 2, 3, 5
- [30] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 4
- [31] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. In ICML, 2023. 2

- [32] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. In NeurIPS, 2023. 2
- [33] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, 2016. 4
- [34] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2
- [35] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 2, 5
- [36] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, 2024. 2
- [37] OpenAI. Gpt-4o mini: advancing cost-efficient intelligence. https://openai.com/index/gpt- 4o- miniadvancing - cost - efficient - intelligence,

2024. 4

- [38] OpenAI. Sora: Creating video from text. https:// openai.com/index/sora, 2024. 2
- [39] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. In ICLR,

2024. 4

- [40] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 2
- [41] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv:2410.13720, 2024. 2
- [42] Senthil Purushwalkam, Akash Gokul, Shafiq Joty, and Nikhil Naik. Bootpig: Bootstrapping zero-shot personalized image generation capabilities in pretrained diffusion models. arXiv:2401.13974, 2024. 5
- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 5
- [44] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 3
- [45] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv:2408.00714,

2024. 4

- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2

- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2, 5, 6, 7
- [48] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In CVPR, 2024. 3, 5
- [49] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. In NeurIPS, 2023. 2
- [50] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. Objectstitch: Object compositing with diffusion model. In CVPR, 2023. 2
- [51] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, He Zhang, Wei Xiong, and Daniel Aliaga. Imprint: Generative object compositing by learning identity-preserving representation. In CVPR, 2024. 2
- [52] StabilityAI. Cosxl model card. https : / / huggingface . co / stabilityai / cosxl, 2024. 5
- [53] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In CVPR, 2024. 2, 6
- [54] Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. Realfill: Reference-driven generation for authentic image completion. TOG, 2024. 8
- [55] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 5
- [56] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv:2404.02733,

2024. 2

- [57] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv:2401.07519, 2024. 2
- [58] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In ICCV, 2023. 5, 6
- [59] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv:2409.04429,

2024. 2

- [60] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv:2409.11340, 2024. 2, 5, 6

- [61] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv:2408.12528, 2024. 2
- [62] XLabs-AI. Flux-ip-adapter model card. https:// huggingface.co/XLabs-AI/flux-ip-adapter,

2024. 6

- [63] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv:2403.01779, 2024. 2
- [64] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In CVPR, 2023. 2
- [65] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In NeurIPS, 2024. 4
- [66] Ling Yang, Bohan Zeng, Jiaming Liu, Hong Li, Minghao Xu, Wentao Zhang, and Shuicheng Yan. Editworld: Simulating world dynamics for instruction-following image editing. arXiv:2405.14785, 2024. 2, 4
- [67] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv:2408.06072,

2024. 2

- [68] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv:2308.06721, 2023. 2, 6
- [69] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. In NeurIPS, 2024. 2, 5, 7
- [70] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2
- [71] Shilong Zhang, Lianghua Huang, Xi Chen, Yifei Zhang, ZhiFan Wu, Yutong Feng, Wei Wang, Yujun Shen, Yu Liu, and Ping Luo. Flashface: Human image personalization with high-fidelity identity preservation. arXiv:2403.17008, 2024. 2
- [72] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. In NeurIPS, 2024. 2, 4, 5, 6
- [73] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. https://github.com/hpcaitech/OpenSora, 2024. 2
- [74] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv:2408.11039, 2024. 2
- [75] Yuqian Zhou, Connelly Barnes, Eli Shechtman, and Sohrab Amirghodsi. Transfill: Reference-guided image inpainting

by merging multiple color and spatial transformations. In CVPR, 2021. 8

[76] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In ECCV, 2024. 2

