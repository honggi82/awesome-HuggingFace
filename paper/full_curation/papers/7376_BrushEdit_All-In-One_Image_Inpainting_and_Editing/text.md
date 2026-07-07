## BrushEdit: All-In-One Image Inpainting and Editing

Yaowei Li1∗ Yuxuan Bian3∗ Xuan Ju3∗ Zhaoyang Zhang2‡ Junhao Zhuang4 Ying Shan2♣ Yuexian Zou1♣ Qiang Xu3♣ 1Peking University 2ARC Lab, Tencent PCG 3The Chinese University of Hong Kong 4Tsinghua University

∗Equal Contribution ‡Project Lead ♣Corresponding Author

Project Page: https://liyaowei-stu.github.io/project/BrushEdit

### arXiv:2412.10316v3[cs.CV]5May2025

[Figure 1]

[Figure 2]

[Figure 3]

Step 1 User Edit Instruction Interpretation Editing Instruction:

[Figure 4]

[Figure 5]

Target Identification:

Wreath

[Figure 6]

Add a wreath On the head

[Figure 7]

Editing Type:

Add

[Figure 8]

[Figure 9]

[Figure 10]

Interactive Refinement

Target Caption:

A young girl with a wreath standing…

[Figure 11]

Step 2 Inpainting-based Image Editing

Curl Hair.

[Figure 12]

[Figure 13]

Editing Mask Target Caption

[Figure 14]

A young girl with a wreath standing…

Interactive Inpainting-based

Remove the Instruction Editing hedgehog.

Make the hedgehog in Italy.

Change the hedgehog to flamingo.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

All-in-one Inpainting Model

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Remove the pigeon. Replace the clothes to a delicate floral skirt. Replace the background to Alice's adventures. Add a shining necklace.

Fig. 1: BrushEdit is a cutting-edge interactive image editing framework that combines language models and inpainting techniques for seamless edits. Leveraging pre-trained multimodal language models and BrushNet’s dual-branch architecture, users can achieve diverse edits such as adding objects, removing elements, or making structural changes with free-form masks.

Abstract—Image editing has advanced significantly with the development of diffusion models using both inversion-based and instruction-based methods. However, current inversion-based approaches struggle with big modifications (e.g., adding or removing objects) due to the structured nature of inversion noise, which hinders substantial changes. Meanwhile, instruction-based methods often constrain users to black-box operations, limiting direct interaction for specifying editing regions and intensity. To address these limitations, we propose BrushEdit, a novel inpainting-based instruction-guided image editing paradigm, which leverages multimodal large language models (MLLMs) and image inpainting models to enable autonomous, user-friendly, and interactive free-form instruction editing. Specifically, we devise a system enabling free-form instruction editing by integrating MLLMs and a dual-branch image inpainting model in an agentcooperative framework to perform editing category classification, main object identification, mask acquisition, and editing area inpainting. Extensive experiments show that our framework effectively combines MLLMs and inpainting models, achieving superior performance across seven metrics including mask region

preservation and editing effect coherence.

Index Terms—Image Editing, Image Inpainting, Multimodal Large Language Model

I. INTRODUCTION

# T

HE rapid advancement of diffusion models has significantly propelled text-guided image generation [1]–[4],

delivering exceptional quality [5], diversity [6], and alignment with textual guidance [7]. However, in image editing tasks—where a target image is generated based on a source image and editing instructions—such progress remains limited due to the difficulty of collecting large amounts of paired data.

To perform image editing based on diffusion generation models, previous methods primarily focus on two strategies: (1) Inversion-based Editing: This approach leverages the structural information of noised latent derived from inversion to preserve content in non-edited regions, while manipulating

User Input Delete Results

###### User Input Add Results

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- Ⅰ
- Ⅱ

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

…

a cat is sitting on a wooden chair.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

(a) Ours

(a) Ours

[Figure 40]

[Figure 41]

[Figure 42]

a laptop is sitting on a table with coffee…

Original Image

Original Image

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

…

ⅠII

[Figure 47]

[Figure 48]

[Figure 49]

- (b) BrushNet-Ran
- (c) BrushNet-Seg

a cat is sitting on a wooden chair

- (b) BrushNet-Ran
- (c) BrushNet-Seg

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

IV

…

User Mask

User Mask

[Figure 56]

a wooden cat in the backyard with a dog.

Editing Instruction:

Editing Instruction:

(a) Mask (b) BrushNetSeg

(c) BrushNetRan

(d) Ours

Put a red dog collar…

Remove the butterfly

#### All-in-one Inpainting

#### User-friendly Editing

- Fig. 2: BrushEdit can achieve all-in-one inpainting for arbitrary mask shapes without requiring separate model training for each mask type. This flexibility in handling arbitrary shapes also enhances user-driven editing, as user-provided masks often combine segmentation-based structural details with random mask noise. By supporting arbitrary mask shapes, BrushEdit avoids the artifacts introduced by the random-mask version of BrushNet-Ran and the edge inconsistencies caused by the segmentationmask version BrushNet-Seg’s strong reliance on boundary shapes.

the latent in edited regions to achieve the desired modifications [8]–[11]. Although this method effectively maintains the overall image structure, it is often time-consuming due to multiple diffusion sampling processes. Additionally, the implicit inverse condition significantly limits editability, making large-scale edits (e.g., background replacement) and structural changes (e.g., adding or removing objects) challenging [12]. Furthermore, these methods typically require users to provide precise and high-quality source and target captions to leverage the conditional generation model’s priors for preserving backgrounds and altering foregrounds. However, in practical scenarios, users often prefer to achieve target area modifications with simple editing instructions. (2) Instruction-based Editing: This strategy involves collecting paired “source image-instruction-target image” data and finetuning diffusion models for editing tasks [13]–[16]. Due to the difficulty of obtaining manually edited paired data, training datasets are often generated using multimodal large language models (MLLMs) and inversion-based image editing methods (e.g., Prompt-to-Prompt [8] and Masactrl [9]). However, the low success rate and unstable quality of these training-free methods [11] result in noisy and unreliable datasets, leading to suboptimal performance of the trained models. Additionally, these methods often use a black-box editing process, preventing users from interactively controlling and refining edits [17].

ground preservation and text-aligned foreground generation abilities of Image Inpainting models [22], [23], inspires us to propose BrushEdit. BrushEdit is an agent-based, free-form, interactive framework for inpainting-driven image editing with instruction guidance that highlights the untapped potential in combining language understanding and image generation capabilities to enable free-form, high-quality interactive natural language-based instruction image editing. This framework requires users to input only natural language editing instructions and supports efficient, arbitrary-round interactive editing, allowing for adjustments in editing types and intensity.

Our approach consists of four main steps: (i) Editing category classification: determine the type of editing required. (ii) Identification of the primary editing object: Identify the main object to be edited. (iii) Acquisition of the editing mask and target Caption: Generate the editing mask and corresponding target caption. (iv) Image inpainting: Perform the actual image editing. Steps (i) to (iii) utilize pre-trained MLLMs [20], [21] and detection models [24] to ascertain the editing type, target object, editing masks, and target caption. Step (iv) involves image editing using the dual-branch inpainting model BrushNet, as detailed in our previous conference paper. This model inpaints the target areas based on the target caption and editing masks, leveraging the generative potential and background preservation capabilities of inpainting models. This framework enables steps (i) to (iii) to extract and summarize instructional information via MLLMs, providing clear intermediate interactive guidance for subsequent diffusion models. Meanwhile, step (iv) maximizes the inpainting models’ ability to preserve the background and generate foreground content as instructed. Meanwhile, users can interactively modify intermediate con-

Given these limitations, we pose the question: Can we develop another editing paradigm that overcomes the challenges of inference efficiency, scalable data curation, editability, and controllability? The remarkable image-text understanding capabilities of Multimodal Large Language Models (MLLMs) [18]–[21], combined with the outstanding back-

trol information (e.g., editing mask or the caption of the edited image) during steps (i) to (iv) and iteratively execute these steps as many times as needed until a satisfactory editing result is achieved. The result is a user-friendly, free-form, multi-turn interactive instruction editing system.

Moreover, we found that BrushNet’s original strategy of training separately on segmentation-based masks and random masks greatly limits its practical applicability. This is because these masks differ significantly from user-drawn masks, resulting in suboptimal performance. User-drawn masks often resemble segmentation masks in terms of object edge shapes but also contain noise and irregularities similar to random masks. To overcome this limitation, we refined, merged, and expanded the original BrushData. This allowed us to train an all-in-one inpainting model capable of handling arbitrary mask shapes, thereby facilitating versatile image editing and inpainting, as illustrated in Fig. 2.

We present a comprehensive evaluation of BrushEditthrough both qualitative and quantitative analyses. We demonstrate that our system significantly enhances image editing quality and efficiency compared to existing methods. It excels particularly in aligning with edit instructions and maintaining background fidelity, thereby validating the effectiveness of our unified inpainting-driven, instruction-guided editing paradigm.

In summary, we extend our conference version [22] by introducing several novel contributions:

- 1) We introduce BrushEdit, an advanced iteration of the previous BrushNet model. BrushEdit extends the capabilities of controllable image generation by pioneering an inpainting-based image editing approach. This unified model supports instruction-guided image editing and inpainting, offering a user-friendly, free-form, multi-turn interactive editing experience.
- 2) By integrating with existing pre-trained multimodal large language models and vision understanding models, BrushEdit significantly improves language comprehension and controllable image generation without necessitating additional training process.
- 3) We expand BrushNet into a versatile image inpainting framework that can accommodate arbitrary mask shapes. This eliminates the need for separate models for different types of mask configurations and enhances its adaptability to real-world user masks.

II. RELATED WORK

- A. Image Editing

Image editing involves modifying object shapes, colors, poses, materials, and adding or removing objects [34]. Recent advancements in diffusion models [1], [2] have notably improved visual generation tasks, outperforming GAN-based models [35]–[37] in image editing. To enable controlled and guided editing, various methods leverage modalities like text instructions [6], [13], [14], masks [15], [23], [38], layouts [8], [9], [39], segmentation maps [40], [41], and point-dragging interfaces [42], [43]. However, these methods often struggle with large structural edits due to noisy latent inversion’s

TABLE I: Comparison of BrushEdit with Previous Image Editing/Inpainting Methods. Note that we only list commonly used text-guided diffusion methods in this table.

Editing Model Plug-and-Play Flexible-Scale Multi-turn Interactive Instruction Editing Prompt2Prompt [8] ✓ ✓

MasaCtrl [9] ✓ ✓ MagicQuill [17] ✓ ✓ ✓

InstructPix2Pix [13] ✓ GenArtist [25] ✓ ✓

BrushEdit ✓ ✓ ✓ ✓

Inpainting Model Plug-and-Play Flexible-Scale Content-Aware Shape-Aware Blended Diffusion [26], [27] ✓

SmartBrush [28] ✓

SD Inpainting [5] ✓ ✓ PowerPaint [29] ✓ ✓ HD-Painter [30] ✓ ✓

ReplaceAnything [31] ✓ ✓

Imagen [32] ✓ ✓ ControlNet-Inpainting [33] ✓ ✓ ✓

BrushEdit ✓ ✓ ✓ ✓

overwhelming structural information or rely on scarce highquality “source image-target image-editing instruction” pairs. Additionally, they usually require users to operate in a blackbox manner, demanding precise inputs like masks, text, or layouts, limiting their usability for content creators. These challenges impede the development of a free-form, interactive natural language editing system.

Many Multi-modal Large Language Model (MLLM)-based methods leverage advanced vision and language understanding capabilities for image editing [15]–[17], [25], [44]. MGIE refines instruction-based editing by generating more detailed and expressive prompts. SmartEdit enhances the comprehension and reasoning of complex instructions. FlexEdit integrates MLLMs to process image content, masks, and textual inputs. GenArtist employs an MLLM agent to decompose complex tasks, guide tool selection, and systematically execute image editing, generation, and self-correction with iterative verification. However, these methods often involve costly MLLM fine-tuning, are limited to single-turn black-box editing, or face both challenges.

The recent MagicQuill [17] enables fine-grained control over shape and color at the regional level using scribbles and colors, leveraging a fine-tuned MLLM to infer editing options from user input. While it provides precise interactive control, it requires labor-intensive strokes to define regions and incurs significant training costs to fine-tune MLLMs. In contrast, our method relies solely on natural language instructions (e.g., ”remove the rose from the dog’s mouth” or ”convert the dumplings on the plate to sushi”) and integrates MLLMs, detection models, and our dual-branch inpainting mode in a training-free, agent-cooperative framework. And our framework also supports multi-round refinement, users can iteratively adjust the generated editing mask and target image caption to achieve multi-round interaction. As summarized in Tab. I, our BrushEdit overcomes the limitations of current editing methods through an instruction-based, multiturn interactive, and plug-and-play design, enabling flexible preservation of unmasked regions and establishing itself as a versatile editing solution.

B. Image Inpainting

Image inpainting remains a key challenge in computer vision, focusing on reconstructing masked regions with realistic

##### Editing Conductor

##### Editing Instructor

[Figure 57]

[Figure 58]

|[Figure 59]|
|---|

Editing Object:

Mask

###### Editing Type:

|[Figure 60]|
|---|

|[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]|
|---|

Hedgehog

Inpainting Branch

[Figure 66]

Background Edit

Target Caption:

A flamingo standing by the tree

Local Edit Addition

Dec

×scale

Mask Image Target Caption:

Masked Image

|[Figure 67]<br><br>[Figure 68]|
|---|

Remove

|[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]| |
|---|---|
| | |

|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]|
|---|

|[Figure 78]|
|---|

Enc

Frozen UNet

A flamingo standing by the tree…

[Figure 79]

64×64×4

Target Caption

[Figure 80]

[Figure 81]

MLLM

[Figure 82]

[Figure 83]

User Interactive Refinement

[Figure 84]

Editing Instruction:

[Figure 85]

Replace the hedgehog with a flamingo…

[Figure 86]

###### T= i round…

- Fig. 3: Model overview. Our model outputs an inpainted image given the mask and masked image input. Firstly, we downsample the mask to accommodate the size of the latent, and input the masked image to the VAE encoder to align the distribution of latent space. Then, noisy latent, masked image latent, and downsampled mask are concatenated as the input of BrushEdit. The feature extracted from BrushEdit is added to pretrained UNet layer by layer after a zero convolution block [33]. After denoising, the generated image and masked image are blended with a blurred mask.

single model to handle arbitrary masks seamlessly, advancing its role as a versatile inpainting solution.

and coherent content [45], [46]. Traditional methods [47], [48] and early Variational Auto-Encoder (VAE) [49], [50] or Generative Adversarial Network (GAN) [35]–[37] approaches often depend on hand-crafted features, leading to limited results.

III. PRELIMINARIES AND MOTIVATION

In this section, we will first introduce diffusion models in Sec. III-A. Then, Sec. III-B would review previous inpainting techniques based on sampling strategy modification and special training. Finally, the motivation is outlined in Section III-D.

Recently, diffusion-based models [26]–[28], [51]–[53] have gained traction for their superior generation quality, precise control, and diverse outputs [1], [5], [54]. Early diffusion approaches for text-guided inpainting [26], [27], [51], [53], [55]–[57], such as Blended Latent Diffusion, modify denoising by sampling masked regions using pre-trained models while preserving unmasked areas from input images. Despite their popularity in tools like Diffusers [58], these methods perform well on simple tasks but falter with complex masks, content, or prompts, often yielding inconsistent outputs due to limited contextual understanding of mask boundaries and surrounding regions. To overcome these shortcomings, recent works [5], [28], [29], [31], [32], [59]–[61] have fine-tuned base models for enhanced content- and shape-awareness. For example, SmartBrush [28] integrates object-mask predictions for better sampling, while Stable Diffusion Inpainting [5] processes masks, masked images, and noisy latents through the UNet architecture for optimized inpainting. Moreover, HD-Painter [30] and PowerPaint [29] improve these models for higher quality and multi-task functionality.

A. Diffusion Models

Diffusion models include a forward process that adds Gaussian noise ϵ to convert clean sample z0 to noise sample zT, and a backward process that iteratively performs denoising from zT to z0, where ϵ ∼ N (0,1), and T represents the total number of timesteps. The forward process can be formulated as follows:

zt = √αtz0 + √1 − αtϵ (1) zt is the noised feature at step t with t ∼ [1,T], and α is a

hyper-parameter.

In the backward process, given input noise zT sampled from a random Gaussian distribution, learnable network ϵθ estimates noise at each step t conditioned on C. After T progressively refining iterations, z0 is derived as the output sample:

However, many methods struggle to generalize inpainting capabilities to arbitrary pre-trained models. One prominent effort is fine-tuning ControlNet [33] on inpainting pairs, but its design remains limited in perceptual understanding, leading to suboptimal results. As summarized in Tab. I, our BrushEdit addressed these issues with a content-aware, shapeaware, and plug-and-play design, allowing flexible preservation of unmasked regions. Building on this, BrushEdit unifies training across random and segmentation masks, enabling a

zt−1 =

√αt−1 √αt

zt + √αt−1

1 αt−1 − 1 −

1 αt − 1 ϵθ (zt,t,C)

(2)

The training of diffusion models revolves around optimizing the denoiser network ϵθ to conduct denoising with condition C, guided by the objective:

0,ϵ∼N(0,I),t∼U(1,T) ∥ϵ − ϵθ (zt,t,C)∥ (3)

min

Ez

θ

- B. Image Inpainting Models

Previous image inpainting approaches can be broadly categorized into Sampling Strategy Modification and Dedicated Inpainting Models.

Sampling Strategy Modification. These methods perform inpainting by iteratively blending masked images with generated content. A representative example is Blended Latent Diffusion (BLD) [27], the default inpainting technique in popular diffusion-based libraries (e.g., Diffusers [58]). Given a binary mask m and a masked image xmasked0 , BLD extracts the latent representation z0masked using a VAE. The mask m is resized to mresized to match the latent dimensions. During inpainting, Gaussian noise is added to z0masked over T steps, producing ztmasked, where t ∼ [1,T]. The denoising starts from zTmasked, with each sampling step (eq. 2) followed by:

zt−1 ← zt−1 · 1 − mresized + ztmasked−1 · mresized (4)

Despite its simplicity, Sampling Strategy Modification often struggles to preserve unmasked regions and align generated content. These shortcomings stem from: (1) inaccuracies introduced by resizing the mask, which hinder proper blending of noisy latents, and (2) the diffusion model’s limited contextual understanding of mask boundaries and unmasked regions.

Dedicated Inpainting Models. To enhance performance, these methods fine-tune base models by adding the mask and masked image as additional UNet input channels, creating architectures specialized for inpainting. While they surpass BLD in generation quality, they face several challenges: (1) They merge noisy latents, masked image latents, and masks at the UNet’s initial convolution layer, where text embeddings globally affect all features, making it difficult for deeper layers to focus on masked image details. (2) Simultaneously handling conditional inputs and generation tasks increases the UNet’s computational load. (3) Extensive fine-tuning is required for different diffusion backbones, leading to high computational costs and limited adaptability to custom diffusion models.

- C. Image Editing Models

Recent image editing methods can fall into two types:

a) Inversion Methods: These approaches [8]–[10], [26], [62], [63] achieve editing by manipulating the latents obtained through inversion. First, they generate edit-friendly noisy latents using various inversion techniques, followed by three paradigms for preserving background regions while modifying target areas: (1) Attention Integration: They [8], [9], [42], [64]–[67] fuse attention maps linking text and image between the source and editing diffusion branches. (2) Target Embedding: They [62], [63], [68]–[74] manage to embed the editing information from the target branch and integrate it into the source diffusion branch. (3) Latent Integration: These methods [10], [26], [27], [42], [75]–[77] try to directly inject editing instructions via noisy latent features from the target diffusion branch into the source diffusion branch. Although these

methods are computationally efficient and achieve competitive zero-shot or few-shot performance, they are often limited in the diversity of supported edits (e.g., typically restricted to object interaction or attribute modification) due to simplistic generation controls. Additionally, the structural prominence in inversion latents often leads to failures when handling significant structural changes, such as object addition/removal or background replacement.

b) End-to-end Methods: These methods [13], [78]–[80] train end-to-end diffusion models for image editing, leveraging various ground-truth or pseudo-paired editing datasets. They support a broader range of edits and avoid the significant speed drawbacks of inversion methods, completing edits in a single forward pass. However, their performance is often constrained by the limited availability of ground-truth editing pairs, necessitating pseudo-pair generation via inversion methods, which hinders their upper-bound performance. Furthermore, these end-to-end models lack support for interactive, multi-round editing, preventing content creators from iterative refining or enhancing edits, thus reducing their practicality.

D. Motivation

Based on the analysis in Section III-B, a more effective inpainting architecture could incorporate an additional branch dedicated to processing masked images, enabling the backbone to recognize mask boundaries and the corresponding background without requiring modifications or retraining. Similarly, as discussed in Section III-C, there is an urgent need for a free-form, interactive natural language instruction editing model. Leveraging the exceptional multimodal understanding of MLLMs, such a model can efficiently identify the editing type, target objects, and regions to edit, as well as generate annotations for the desired output. With the support of image inpainting models, precise edits within the target masked regions can then be achieved. Moreover, this process can be iteratively refined, allowing users to create transparently and iteratively.

IV. METHOD

An overview of BrushEdit is shown in Fig. 3. Our framework integrates MLLMs with a dual-branch image inpainting model via agent collaboration, enabling free-form, multiturn interactive instruction editing. Specifically, a pre-trained MLLM, acting as the Editing Instructor, interprets user instructions to identify editing types, locate target objects, retrieve detection results for the editing region, and generate textual descriptions of the edited image. Guided by this information, the inpainting model, serving as the Editing Conductor, fills the masked region based on the target text caption. This iterative process allows users to modify or refine intermediate control inputs at any stage, supporting flexible and interactive instruction-based editing.

A. Editing Instructor

In BrushEdit, we use an MLLM as an editing instructor to interpret users’ free-form editing instructions, categorize

them into predefined types (addition, removal, local edit, background edit), identify target objects, and utilize a pre-trained detection model to find the relevant editing mask. Finally, the edited image caption is generated. In the next stage, this information is packaged and sent to the editing system to complete the task using an image inpainting approach.

The formal process is as follows: Given the editing instruction TIns and source image Isrc, we first use a pretrained MLLM ϕMLLM to identify the user’s editing type K and the corresponding target object O The MLLM then calls a pre-trained detection model ϕD to search for the target object mask Md based on O. After obtaining the mask, the MLLM combines K, O, and Isrc to generate the final edited image caption. The source image Isrc, target mask Md, and the caption are then passed to the next stage, the Editing Conductor, for image-inpainting-based editing.

- B. Editing Conductor

Our Editing Conductor, built on our previous BrushNet, employs a mixed fine-tuning strategy using both random and segmentation masks. This approach enables the inpainting model to handle diverse mask-based inpainting tasks without being restricted by mask types, achieving comparable or superior performance. Specifically, we inject masked image features into a pre-trained diffusion network (e.g., Stable Diffusion 1.5) through an additional control branch. These features include the noisy latent for enhancing semantic coherence by providing information on the current generation process, the masked image latent extracted via VAE to guide semantic consistency between the prompt foreground and the ground truth background, and the mask downsampled via cubic interpolation to explicitly indicate the position and boundaries of the foreground filling region.

To retain masked image features, BrushEdit employs a duplicate of the pre-trained diffusion model with all attention layers removed. The pre-trained convolutional weights serve as a robust prior for extracting masked image features, while excluding cross-attention layers ensures the branch focuses solely on pure background information. BrushEdit features are integrated into the frozen diffusion model layer-by-layer, enabling hierarchical, dense per-pixel control. Following ControlNet [33], zero convolution layers are used to link the frozen model with the trainable BrushEdit, mitigating noise during early training stages. The feature insertion operation is defined in Eq. 5:

ϵθ (zt,t,C)i = ϵθ (zt,t,C)i + w · Z ϵBrushNetθ zt,z0masked,mresized ,t i

(5)

, where ϵθ (zt,t,C)i represents the feature of the i-th layer in the network ϵθ, where i ∈ [1,n], and n denotes the total number of layers. The same notation is applied to ϵBrushNetθ . The network ϵBrushNetθ processes the concatenated noisy latent zt, masked image latent z0masked, and downsampled mask mresized, where concatenation is represented by [·]. Z refers to the zero convolution operation, and w is the preservation

scale that adjusts the influence of BrushEdit on the pretrained diffusion model.

Previous studies have highlighted that downsampling during latent blending can introduce inaccuracies, and the VAE encoding-decoding process has inherent limitations that impair full image reconstruction. To ensure consistent reconstruction of unmasked regions, prior methods have explored various strategies. Some approaches [29], [31] rely on copy-and-paste techniques to directly transfer unmasked regions, but these often result in outputs lacking semantic coherence. Latent blending methods inspired by BLD [5], [27] also struggle to retain desired information in unmasked areas effectively. In this work, we propose a simple pixel-space approach that applies mask blurring before copy-and-paste using the blurred mask. Although this may slightly affect accuracy near the mask boundary, the error is nearly imperceptible and significantly improves boundary coherence.

The architecture of BrushEdit is inherently designed for seamless plug-and-play integration with various pretrained diffusion models, enabling flexible preservation control. Specifically, the flexible capabilities of BrushEdit include: (1) Plugand-Play Integration: As BrushEdit does not modify the pretrained diffusion model’s weights, it can be effortlessly integrated with any community fine-tuned models, facilitating easy adoption and experimentation. (2) Preservation Scale Adjustment: The preservation scale of the unmasked region can be controlled by incorporating BrushEdit features into the frozen diffusion model with a weight w, which adjusts the influence of BrushEdit on the level of preservation. (3) Blurring and Blending Customization: The preservation scale can be further refined by adjusting the blurring scale and applying blending operations as needed. These features provide finegrained and flexible control over the editing process.

V. EXPERIMENTS A. Evaluation Benchmark and Metrics

a) Benchmark: To comprehensively evaluate the performance of BrushEdit, we conducted experiments on both image editing and image inpainting benchmarks:

- • Image Editing. We used PIE-Bench [11] (Prompt-based Image Editing Benchmark) to evaluate BrushEditand all baselines on image editing tasks. PIE-Bench consists of 700 images spanning 10 editing types, evenly distributed between natural and artificial scenes (e.g., paintings) across four categories: animal, human, indoor, and outdoor. Each image includes five annotations: source image prompt, target image prompt, editing instruction, main editing body, and editing mask.
- • Image Inpainting. Extending our prior conference work, we replaced traditional benchmarks [81]–[86] with BrushBenchfor segmentation-based masks and EditBench for random brush masks. These benchmarks span real and generated images across human bodies, animals, and indoor and outdoor scenes. EditBench includes 240 images with an equal mix of natural and generated content, each annotated with a mask and caption. BrushBench, shown in Fig. 4, contains 600 images with human-annotated

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Cartoon girl sitting at a table with pizza and drink

Cover of a book, the girl and the dog

A desk with a laptop, a mouse, and a plant

A colorful sports car is parked in a city street

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

- Ⅰ
- Ⅱ

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

- Ⅲ

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

The back of a woman in a red dress

A dog sitting on the beach with ocean in background

A blue glass pendant on a black leather cord

A stone bridge over a river in the fog

[Figure 123]

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

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

(a) BrushBench - Human (b) BrushBench - Animal (c) BrushBench - Indoor (d) BrushBench - Outdoor

[Figure 139]

[Figure 140]

[Figure 141]

The oil painting of a yellow dog next to a table with a cat under it

A white wooden shelf on the side of a wall in a living room

A paper cup next to a cell phone and some magazines

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

A wooden cat next to a real dog

A stone bird next to a window in a kitchen

A metal cd rack to the right on a desk

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

(f) EditBench - Natural

(e) EditBench - Generated

- Fig. 4: Benchmark overview. I and II separately show natural and artificial images, masks, and caption of BrushBench. (a) to (d) show images of humans, animals, indoor scenarios, and outdoor scenarios. Each group of images shows the original image, inside-inpainting mask, and outside-inpainting mask, with an image caption on the top. III show image, mask, and caption from EditBench [32], with (e) for generated images and (f) for natural images. The images are randomly selected from both benchmarks.

masks and captions, evenly distributed across natural and artificial scenes (e.g., paintings) and covering various categories such as humans, animals, and indoor/outdoor environments.

BrushEdit in edit instruction alignment and background fidelity.

- • Background Fidelity. We adopt standard metrics, including Peak Signal-to-Noise Ratio (PSNR) [89], Learned Perceptual Image Patch Similarity (LPIPS) [90], Mean Squared Error (MSE) [91], and Structural Similarity Index Measure (SSIM) [92], to evaluate the consistency between the unmasked regions of the generated and original images.
- • Text Alignment. We use CLIP Similarity (CLIP Sim) [93] to assess text-image consistency by projecting both into the shared embedding space of the CLIP model [94] and measuring the similarity of their representations.

We refined the task into two scenarios for segmentation-based mask inpainting: inside-inpainting and outside-inpainting, enabling detailed performance evaluation across distinct image regions.

Notably, BrushEdit surpasses BrushNet by leveraging unified high-quality inpainting masked images for training, enabling it to handle all mask types. This establishes BrushEditas a unified model capable of performing all inpainting and editing benchmark tasks, whereas BrushNet required separate fine-tuning for each mask type.

- b) Dataset: Building upon the BrushData proposed in

our previous conference version, we integrate two subsets of segmentation masks and random masks, and further extend the data from the Laion-Aesthetic [87] dataset, resulting in BrushData-v2. A key difference is that we select images with clean backgrounds and pair them randomly with either segmentation or random masks, effectively creating pairs that simulate deletion-based editing, significantly enhancing our framework’s removal capability in image editing. The data expansion process is as follows: We use Grounded-SAM [88] to annotate open-world masks, then filter them based on confidence scores to retain only those with higher confidence. We also consider mask size and continuity during the filtering.

- c) Metrics: We evaluate five metrics, focusing on

B. Implementation Details

We evaluate various inpainting methods under a consistent setting unless stated otherwise, i.e., using NVIDIA Tesla V100 GPUs and their open-source code with Stable Diffusion v1.5 as the base model, 50 steps, and a guidance scale of 7.5. Each method utilizes its recommended hyper-parameters across all images to ensure fairness. BrushEdit and all ablation models are trained for 430k steps on 8 NVIDIA Tesla V100 GPUs, requiring approximately 3 days. Notably, for all image editing (PnPBench) and image inpainting (BrushBench and EditBench) tasks, BrushEditachieves unified image editing and inpainting using a single model trained on BrushData-v2. In contrast, our previous BrushNet required separate training and testing for different mask types. Additional details are available in the provided code.

unedited/uninpainted region preservation and edited/inpainted region text alignment. Additionally, we conducted extensive user studies to validate the superior performance of

[Figure 161]

[Figure 162]

Remove the laptop from the table

Delete the flower in the puppy's mouth

[Figure 163]

[Figure 164]

[Figure 165]

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

[Figure 176]

- Ⅰ
- Ⅱ
- Ⅲ
- Ⅳ

[Figure 177]

[Figure 178]

[Figure 179]

Put a red dog collar on the dog's neck

Add flowers to the woman's hair

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

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

Change the jacket to a blouse

Change the wreath with a crown

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

a Replace the dumplings with sushi

a Replace the car with a motorcycle

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

(a) Ori (b) Mask (c) P2P (d) Masa (e) P2P-Z (f) PnP (g) Ours

(a) Ori (b) Mask (c) P2P (d) Masa (e) P2P-Z (f) PnP (g) Ours

###### Fig. 5: Comparison of previous editing methods and BrushEdit on natural and synthetic images, covering image editing operations such as removing objects (I), adding objects (II), modifying attributes (III), and swapping objects (IV).

- TABLE II: Comparison of BrushEdit with various editing methods in PnpBench. For editing methods Promptto-Prompt (P2P) [8], MasaCtrl [9], Pix2Pix-Zero (P2P-Zero) [9], and Plug-and-Play (PnP) [66], we evaluate two inversion techniques, DDIM Inversion (DDIM) [2] and PnP Inversion (PnP) [11], to establish stronger baselines. Red stands for the best result, Blue stands for the second best result.

Inverse Editing PSNR ↑LPIPS×103 ↓MSE×104 ↓SSIM×102 ↑ CLIP Similariy ↑ DDIM P2P 17.87 208.80 219.88 71.14 22.44

PnP P2P 27.22 54.55 32.86 84.76 22.10 DDIM MasaCtrl 22.17 106.62 86.97 79.67 21.16

PnP MasaCtrl 22.64 87.94 81.09 81.33 21.35 DDIM P2P-Zero 20.44 172.22 144.12 74.67 20.54

PnP P2P-Zero 21.53 138.98 127.32 77.05 21.05 DDIM PnP 22.28 113.46 83.64 79.05 22.55

PnP PnP 22.46 106.06 80.45 79.68 22.62 BrushEdit 32.16 17.22 8.43 97.08 22.44

- TABLE III: Comparison of inference time between our inpainting-based BrushEdit and other inversion-based methods, including Negative-Prompt Inversion (NP), Edit Friendly Inversion (EF), AIDI [98], EDICT, Null-Text Inversion (NT), and Style Diffusion added with Promptto-Prompt. BrushEdit achieves better editing results with far less inference time than all inversion-based methods.

- C. Quantitative Comparison (Image Editing)

Tab. II and Tab. III compare the quantitative image editing performance on PnPBench [11]. We evaluate the editing results of previous inversion-based methods, including four inversion techniques—DDIM Inversion [2], Null-Text Inversion [95], Negative-Prompt Inversion [96], and StyleDiffusion [97]—as well as four editing methods: Prompt-to-Prompt [8], MasaCtrl [9], pix2pix-zero [65], and Plug-and-Play [66].

The results in Tab. II confirm the superiority of BrushEdit in preserving unedited regions and ensuring accurate text alignment in edited areas. While inversion-based methods, such as DDIM Inversion (DDIM) [2] and PnP Inversion (PnP) [11], can achieve high-quality background preservation, they are inherently limited by reconstruction errors that affect background retention. In contrast, BrushEdit separately models unedited background information through a dedicated branch, while the main network generates the edited region based on the text prompt. Combined with predefined user masks and blending operations, it ensures near-lossless background preservation and semantically coherent edits.

More importantly, our method preserves high-fidelity background information without being affected by the irretrievable structural noise from inversion-based methods. It allows operations, such as adding or removing objects, that are typically impossible with inversion-based editing. Furthermore, since no inversion is required, BrushEdit only needs a single forward pass to perform the editing operation. As shown in Tab. III , the editing time of BrushEdit is significantly short, greatly improving the efficiency of image editing.

- D. Qualitative Comparison (Image Editing)

Methods BrushEdit NP EF AIDI EDICT NT Style Diffusion Inference Time (s) 3.57 18.22 19.10 35.41 35.48 148.48 382.98

including deleting objects (I), adding objects (II), modifying objects (III), and swapping objects (IV). BrushEdit consistently achieves superior coherence between the edited and unedited regions, excelling in adherence to editing instructions, smoothness at the editing mask boundaries, and overall content consistency. Notably, Fig. 5 I and II involve tasks such as deleting a flower or laptop, and adding a collar or earring. While previous methods failed to deliver satisfactory results due to persistent structural artifacts caused by inversion noise, BrushEdit successfully performs the intended operations and

The qualitative comparison with previous image editing methods is shown in Fig. 5. We present results on both artificial and natural images across various editing tasks,

TABLE IV: Quantitative comparisons between BrushEdit and other diffusion-based inpainting models in BrushBench: Blended Latent Diffusion (BLD) [27], Stable Diffusion Inpainting (SDI) [5], HD-Painter (HDP) [30], PowerPaint (PP) [29], ControlNet-Inpainting (CNI) [33], and our previous Segmentation-based BrushNet-Seg [22]. The table shows metrics on background fidelity and text alignment (Text Align) for both inside- and outside-inpainting. All models use Stable Diffusion V1.5 as the base model. Red indicates the best result, while Blue indicates the second-best result.

Inside-inpainting Outside-inpainting Metrics Masked Background Fidelity Text Align Metrics Masked Background Fidelity Text Align Models PSNR↑ MSE×103 ↓ LPIPS×103 ↓ SSIM×103 ↑ CLIP Sim↑ Models PSNR↑ MSE×103 ↓ LPIPS×103 ↓ SSIM×103 ↑ CLIP Sim↑

BLD (1) 21.33 9.76 49.26 74.58 26.15 BLD (1) 15.85 35.86 21.40 77.40 26.73 SDI (2) 21.52 13.87 48.39 89.07 26.17 SDI (2) 18.04 19.87 15.13 91.42 27.21 HDP (3) 22.61 9.95 43.50 89.03 26.37 HDP (3) 18.03 22.99 15.22 90.48 26.96 PP (4) 21.43 32.73 48.43 86.39 26.48 PP (4) 18.04 31.78 15.13 90.11 26.72 CNI (5) 12.39 78.78 243.62 65.25 26.47 CNI (5) 11.91 83.03 58.16 66.80 27.29 CNI* (5) 22.73 24.58 43.49 91.53 26.22 CNI* (5) 17.50 37.72 19.95 94.87 26.92 BrushNet-Seg* 31.94 0.80 18.67 96.55 26.39 BrushNet-Seg* 27.82 2.25 4.63 98.95 27.22 BrushEdit* 31.98 0.79 18.92 96.68 26.24 BrushEdit* 27.65 2.30 4.90 98.97 27.29

* with blending operation

produces seamless edits that blend harmoniously with the background, owing to its dual-branch decoupled inpaintingbased editing paradigm.

- E. Quantitative Comparison (Image Inpainting)

Tab. IV and Tab. V present the quantitative comparison on BrushBench and EditBench [32]. We evaluate the inpainting results of the sampling strategy modification method Blended Latent Diffusion [27], dedicated inpainting models Stable Diffusion Inpainting [5], HD-Painter [30], PowerPaint [29], the plug-and-play method ControlNet [33] trained on inpainting data, and our previous BrushNet1.

Results confirm BrushEdit’s superiority in preserving uninpainted regions and ensuring accurate text alignment in inpainted areas. Blended Latent Diffusion [27] performs the worst, primarily due to incoherent transitions between masked and unmasked regions, stemming from its disregard for mask boundaries and blending-induced latent space losses. HDPainter [30] and PowerPaint [29], both based on Stable Diffusion Inpainting [5], achieve similar results to their base model for inside-inpainting tasks. However, their performance deteriorates sharply in outside-inpainting, as they are designed exclusively for inside-inpainting. ControlNet [33], explicitly trained for inpainting, shares the most comparable experimental setup with ours. Nonetheless, its design mismatch with the inpainting task hampers its ability to maintain masked region fidelity and text alignment, requiring integration with Blended Latent Diffusion [27] for reasonable results. Even with this combination, it falls short of specialized inpainting models and BrushEdit. The performance on EditBench aligns closely with that on BrushBench, both demonstrating BrushEdit’s superior results. This suggests that our method performs consistently well across various inpainting tasks, including segmentation, random, inside, and outside inpainting masks.

It is worth noting that, compared to BrushNet, BrushEdit now surpasses BrushNet in both segmentationmask-based and random-mask-based benchmarks with a single model, achieving a more general and robust all-inone inpainting. This improvement is largely attributed to

1BrushNet fine-tunes separate models for different mask types, while BrushEdit uses a unified model and achieves state-of-the-art performance on both segmentation-based BrushBench and random-mask-based EditBench.

our unified mask types and the richer data distribution in BrushData-v2.

TABLE V: Quantitative comparisons among BrushEdit and other diffusion-based inpainting models, Random-maskbased BrushNet-Ran in EditBench. A detailed explanation of compared methods and metrics can be found in the caption of Tab. IV. Red stands for the best result, Blue stands for the second best result.

Metrics Masked Background Fidelity Text Align Models PSNR↑ MSE×103 ↓ LPIPS×103 ↓ SSIM×103 ↑ CLIP Sim↑

BLD [27] 20.89 10.93 31.90 85.09 28.62 SDI [5] 23.25 6.94 24.30 90.13 28.00 HDP [30] 23.07 6.70 24.32 92.56 28.34 PP [29] 23.34 20.12 24.12 91.49 27.80 CNI [33] 12.71 69.42 159.71 79.16 28.16 CNI* [33] 22.61 35.93 26.14 94.05 27.74 BrushNet-Ran* 33.66 0.63 10.12 98.13 28.87 BrushEdit* 32.97 0.70 7.24 98.60 29.62

* with blending operation

- F. Qualitative Comparison (Image Inpainting)

The qualitative comparison with previous image inpainting methods is shown in Fig. 6. We evaluate results on both artificial and natural images across diverse inpainting tasks, including random mask inpainting and segmentation mask inpainting. BrushEdit consistently achieves superior coherence between the generated and unmasked regions in terms of both content and color (I, II). Notably, in Fig. 6 II (left), the task involves generating both a cat and a goldfish. While all prior methods fail to recognize the existing goldfish in the masked image and instead generate an additional fish, BrushEdit accurately integrates background context, enabled by its dualbranch decoupling design. Furthermore, BrushEdit outperforms our previous BrushNet in overall inpainting performance without fine-tuning for specific mask types, achieving comparable or even better results on both random and segmentationbased masks.

- G. Flexible Control Ability

Fig. 7 and Fig. 8 demonstrate the flexible control offered by BrushEdit in two key areas: base diffusion model selection and

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

[Figure 243]

Mona Lisa by Leonardo Da Vinci A black and red mountain bike besides a building

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

- I
- II

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

A cat looking at a fish bowl with goldfish A bed with quilt and a chair with blanket

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

(a) Mask (b) BLD (c) SDI (d) HDP (e) PP (f) CNI (g) Bru (h) Ours (a) Mask (b) BLD (c) SDI (d) HDP (e) PP (f) CNI (g) Ours (h) Ours

- Fig. 6: Performance comparisons of BrushEdit and previous image inpainting methods across various inpainting tasks: (I) Random Mask Inpainting (II) Segmentation Mask Inpainting. Each group of results contains 7 inpainting methods: (b) Blended Latent Diffusion (BLD) [27], (c) Stable Diffusion Inpainting (SDI) [5], (d) HD-Painter (HDP) [30], (e) PowerPaint (PP) [29], (f) ControlNet-Inpainting (CNI) [33], (g) Our Previous BrushNet and (h) Ours.

[Figure 277]

[Figure 278]

A little girl sitting on the grass with her do g

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

(a) Mask (b) DS (c) ER (d) HR (e) MM (f) RV

[Figure 289]

A woman with her hands on her face

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

VI

[Figure 297]

(a) Mask (b) DS (c) ER (d) HR (e) MM (f) RV

[Figure 298]

[Figure 299]

A single red rose is laying on top of snow A cartoon boy driving a car

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

ⅠII

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Change the horse into a unicorn Remove the butterfly

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

ⅠI

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

Replace the cat with a bear Add a crown to the woman

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Ⅰ

[Figure 341]

[Figure 342]

[Figure 343]

(a) Image (b) DS (c) ER (d) HR (e) MM (f) RV (a) Image (b) DS (c) ER (d) HR (e) MM (f) RV

- Fig. 7: Integrating BrushEdit to community fine-tuned diffusion models. We use five popular community diffusion models fine-tuned from stable diffusion v1.5: DreamShaper (DS) [99], epiCRealism (ER) [100], Henmix Real (HR) [101], MeinaMix (MM) [102], and Realistic Vision (RV) [103]. MM is specifically designed for anime images.

scale adjustment. This flexibility extends beyond inpainting to image editing, as it is achieved by altering the backbone network’s generative prior and branch information injection strength. In Fig. 7, we show how BrushEdit can be combined with various community-finetuned diffusion models, enabling users to choose the model that best aligns with their specific editing or inpainting needs. This greatly enhances the practical value of BrushEdit. Fig. 8 illustrates the control over BrushEdit’s scale parameter, which allows users to adjust the extent of unmasked region protection during editing or inpainting, offering fine-grained control for precise and customizable results.

H. Ablation Study

We conducted ablation studies to examine the impact of different model designs on image inpainting tasks. Since BrushEdit is based on an image inpainting model, the editing

TABLE VI: Ablation on dual-branch design. Stable Diffusion Inpainting (SDI) use single-branch design, where the entire UNet is fine-tuned. We conducted an ablation analysis by training a dual-branch model with two variations: one with the base UNet fine-tuned, and another with the base UNet forzened. Results demonstrate the superior performance achieved by adopting the dual-branch design. Red is the best result.

Metrics Image Quality Masked Region Preservation Text Align Model IR×10↑ HPS×102 ↑ AS↑ PSNR↑ MSE×102 ↓ LPIPS×103 ↓ CLIP Sim↑

SDI 11.00 27.53 6.53 19.78 16.87 31.76 26.69 w/o fine-tune 11.59 27.71 6.59 19.86 16.09 31.68 26.91

###### w/ fine-tune 11.63 27.73 6.60 20.13 15.84 31.57 26.93

task is achieved through inference-only by chaining MLLMs, BrushEdit, and an image detection model as agents. The inpainting capability directly reflects our model’s training

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

###### Replace the cat with a bear

###### Add a crown to the woman

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

###### Change the horse into a unicorn

###### Remove the butterfly

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

###### (a) Image (b) 1.0 (c) 0.8 (d) 0.6 (e) 0.5 (f) 0.4 (g) 0.3 (h) 0.2 (a) Image (b) 1.0 (c) 0.8 (d) 0.6 (e) 0.5 (f) 0.4 (g) 0.3 (h) 0.2

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Book, coffee and purple flower

###### A small island with a tree and clouds

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

A cupcake with a lit candle on top

A white rabbit sitting on the ground in the grass

[Figure 417]

[Figure 418]

(a) Mask (b) 1.0 (c) 0.8 (d) 0.6 (e) 0.5 (f) 0.4 (g) 0.3 (h) 0.2

(a) Mask (b) 1.0 (c) 0.8 (d) 0.6 (e) 0.5 (f) 0.4 (g) 0.3 (h) 0.2

- Fig. 8: Flexible control scale of BrushEdit. (a) shows the given masked image, (b)-(h) show adding control scale w from 1.0 to 0.2. Results show a gradually diminishing controllable ability from precise to rough control.

outcome. Tab. VI compares the dual-branch and single-branch designs, while Tab. VII highlights the ablation study on the additional branch architecture.

The ablation studies, performed on BrushBench, average the performance for both inside-inpainting and outside-inpainting. The results in Tab. VI show that the dual-branch design significantly outperforms the single-branch design. Moreover, finetuning the base diffusion model in the dual-branch setup yields superior results compared to freezing it. However, fine-tuning may limit flexibility and control over the model. Considering the trade-off between performance and flexibility, we chose to adopt the frozen dual-branch design for our model. Tab. VII explains the reasoning behind key design choices: (1) using a VAE encoder instead of randomly initialized convolution layers for processing masked images, (2) incorporating the full UNet feature layer-by-layer into the pre-trained UNet, and (3) removing text cross-attention in BrushEditto prevent masked image features from being influenced by text.

VI. DISCUSSION

- a) Conclusion.: This paper introduces a novel

Inpainting-based Instruction-guided Image Editing paradigm (IIIE), which combines large language models (LLMs) and plug-and-play, all-in-one image inpainting models to enable autonomous, user-friendly, and interactive free-form instruction editing. Quantitative and qualitative results on PnPBench, our proposed benchmark, BrushBench, and EditBench demonstrate the superior performance of BrushEdit in terms of masked background preservation and image-text alignment in image editing and inpainting tasks.

- b) Limitations and Future Work.: However,

BrushEdit has some limitations: (1) The quality and content generated by our model heavily depend on the selected base model. (2) Even with BrushEdit, poor generation results still occur when the mask has an irregular shape or when the

TABLE VII: Ablation on model architecture. We ablate on the following components: the image encoder (Enc), selected from a random initialized convolution (Conv) and a VAE; the inclusion of mask in input (Mask), chosen from adding (w/) and not adding (w/o); the presence of cross-attention layers (Attn), chosen from adding (w/) and not adding (w/o); the type of UNet feature addition (UNet), selected from adding the full UNet feature (full), adding half of the UNet feature (half), and adding the feature like ControlNet (CN); and finally, the blending operation (Blend), chosen from not adding (w/o), direct pasting (paste), and blurred blending (blur). Red is the best result.

Metrics Image Quality Masked Region Preservation Text Align Enc Mask Attn UNet Blend IR×10↑ HPS×102 ↑ AS↑ PSNR↑ MSE×102 ↓ LPIPS×103 ↓ CLIP Sim↑

Conv w/ w/o full w/o 11.05 26.23 6.55 14.89 37.23 64.54 26.76 VAE w/o w/o full w/o 11.55 27.70 6.57 17.96 26.38 49.33 26.87 VAE w/ w/ full w/o 11.25 27.62 6.56 18.69 19.44 34.28 26.63 Conv w/ w/ CN w/o 9.58 26.85 6.47 12.15 80.91 150.89 26.88 VAE w/ w/ CN w/o 10.53 27.42 6.59 18.28 24.36 41.63 26.89 VAE w/ w/o CN w/o 11.42 27.69 6.58 18.49 24.09 36.33 26.86 VAE w/ w/o half w/o 11.47 27.70 6.57 19.01 23.77 33.57 26.87 VAE w/ w/o full w/o 11.59 27.71 6.59 19.86 16.09 31.68 26.91 VAE w/ w/o full paste 11.72 27.93 6.58 - - - 26.80

VAE w/ w/o full blur 11.76 27.94 6.58 29.88 1.53 11.65 26.81

provided text does not align well with the masked image. In future work, we aim to address these challenges.

c) Negative Social Impact.: Image inpainting models offer exciting opportunities for content creation but also present potential risks to individuals and society. Their reliance on internet-collected training data may amplify social biases, and there is a specific risk of generating misleading content by manipulating human images with offensive elements. To mitigate these concerns, responsible use and the establishment of ethical guidelines are essential, which will also be a focus in our future model releases.

REFERENCES

- [1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems (NIPS), vol. 33, pp. 6840–6851, 2020.
- [2] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.
- [3] X. Ju, A. Zeng, C. Zhao, J. Wang, L. Zhang, and Q. Xu, “Humansd: A native skeleton-guided diffusion model for human image generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15988–15998.
- [4] X. Liu, J. Ren, A. Siarohin, I. Skorokhodov, Y. Li, D. Lin, X. Liu, Z. Liu, and S. Tulyakov, “Hyperhuman: Hyper-realistic human generation with latent structural diffusion,” arXiv preprint arXiv:2310.08579, 2023.
- [5] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 10684–10695.
- [6] X. Dai, J. Hou, C.-Y. Ma, S. Tsai, J. Wang, R. Wang, P. Zhang, S. Vandenhende, X. Wang, A. Dubey et al., “Emu: Enhancing image generation models using photogenic needles in a haystack,” arXiv preprint arXiv:2309.15807, 2023.
- [7] Y. Li, X. Liu, A. Kag, J. Hu, Y. Idelbayev, D. Sagar, Y. Wang, S. Tulyakov, and J. Ren, “Textcraftor: Your text encoder can be image quality controller,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7985–7995.
- [8] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-or, “Prompt-to-prompt image editing with cross-attention control,” in International Conference on Learning Representations (ICLR), 2023.
- [9] M. Cao, X. Wang, Z. Qi, Y. Shan, X. Qie, and Y. Zheng, “Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing,” arXiv preprint arXiv:2304.08465, 2023.
- [10] C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon, “SDEdit: Guided image synthesis and editing with stochastic differential equations,” in International Conference on Learning Representations (ICLR), 2022.
- [11] X. Ju, A. Zeng, Y. Bian, S. Liu, and Q. Xu, “Pnp inversion: Boosting diffusion-based editing with 3 lines of code,” International Conference on Learning Representations (ICLR), 2024.
- [12] S. Xu, Y. Huang, J. Pan, Z. Ma, and J. Chai, “Inversion-free image editing with natural language,” arXiv preprint arXiv:2312.04965, 2023.
- [13] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 18392–18402.
- [14] Z. Geng, B. Yang, T. Hang, C. Li, S. Gu, T. Zhang, J. Bao, Z. Zhang, H. Li, H. Hu et al., “Instructdiffusion: A generalist modeling interface for vision tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 12709–12720.
- [15] Y. Huang, L. Xie, X. Wang, Z. Yuan, X. Cun, Y. Ge, J. Zhou, C. Dong, R. Huang, R. Zhang et al., “Smartedit: Exploring complex instructionbased image editing with multimodal large language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8362–8371.
- [16] T.-J. Fu, W. Hu, X. Du, W. Y. Wang, Y. Yang, and Z. Gan, “Guiding instruction-based image editing via multimodal large language models,” arXiv preprint arXiv:2309.17102, 2023.
- [17] Z. Liu, Y. Yu, H. Ouyang, Q. Wang, K. L. Cheng, W. Wang, Z. Liu, Q. Chen, and Y. Shen, “Magicquill: An intelligent interactive image editing system,” 2024.
- [18] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, B. Li, P. Luo, T. Lu, Y. Qiao, and J. Dai, “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” arXiv preprint arXiv:2312.14238, 2023.
- [19] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., “How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,” arXiv preprint arXiv:2404.16821, 2024.
- [20] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [21] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang et al., “Qwen2 technical report,” arXiv preprint arXiv:2407.10671, 2024.

- [22] X. Ju, X. Liu, X. Wang, Y. Bian, Y. Shan, and Q. Xu, “Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion,” 2024.
- [23] J. Zhuang, Y. Zeng, W. Liu, C. Yuan, and K. Chen, “A task is worth one word: Learning with task prompts for high-quality versatile image inpainting,” arXiv preprint arXiv:2312.03594, 2023.
- [24] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.
- [25] Z. Wang, A. Li, Z. Li, and X. Liu, “Genartist: Multimodal llm as an agent for unified image generation and editing,” arXiv preprint arXiv:2407.05600, 2024.
- [26] O. Avrahami, D. Lischinski, and O. Fried, “Blended diffusion for textdriven editing of natural images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 18208–18218.
- [27] O. Avrahami, O. Fried, and D. Lischinski, “Blended latent diffusion,” ACM transactions on graphics (TOG), vol. 42, no. 4, pp. 1–11, 2023.
- [28] S. Xie, Z. Zhang, Z. Lin, T. Hinz, and K. Zhang, “Smartbrush: Text and shape guided object inpainting with diffusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 22428–22437.
- [29] J. Zhuang, Y. Zeng, W. Liu, C. Yuan, and K. Chen, “A task is worth one word: Learning with task prompts for high-quality versatile image inpainting,” arXiv preprint arXiv:2312.03594, 2023.
- [30] H. Manukyan, A. Sargsyan, B. Atanyan, Z. Wang, S. Navasardyan, and H. Shi, “Hd-painter: High-resolution and prompt-faithful textguided image inpainting with diffusion models,” arXiv preprint arXiv:2312.14091, 2023.
- [31] C. Binghui, L. Chao, Z. Chongyang, X. Wangmeng, G. Yifeng, and X. Xuansong, “Replaceanything as you want: Ultra-high quality content replacement,” 2023. [Online]. Available: https: //aigcdesigngroup.github.io/replace-anything/
- [32] S. Wang, C. Saharia, C. Montgomery, J. Pont-Tuset, S. Noy, S. Pellegrini, Y. Onoe, S. Laszlo, D. J. Fleet, R. Soricut et al., “Imagen editor and editbench: Advancing and evaluating text-guided image inpainting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 18359–18369.
- [33] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
- [34] Y. Huang, J. Huang, Y. Liu, M. Yan, J. Lv, J. Liu, W. Xiong, H. Zhang, S. Chen, and L. Cao, “Diffusion model-based image editing: A survey,” arXiv preprint arXiv:2402.17525, 2024.
- [35] H. Liu, Z. Wan, W. Huang, Y. Song, X. Han, and J. Liao, “Pd-GAN: Probabilistic diverse GAN for image inpainting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 9371–9381.
- [36] H. Zheng, Z. Lin, J. Lu, S. Cohen, E. Shechtman, C. Barnes, J. Zhang, N. Xu, S. Amirghodsi, and J. Luo, “Image inpainting with cascaded modulation GAN and object-aware training,” in European Conference on Computer Vision (ECCV). Springer, 2022, pp. 277–296.
- [37] S. Zhao, J. Cui, Y. Sheng, Y. Dong, X. Liang, E. I. Chang, and Y. Xu, “Large scale image completion via co-modulated generative adversarial networks,” arXiv preprint arXiv:2103.10428, 2021.
- [38] J. Singh, J. Zhang, Q. Liu, C. Smith, Z. Lin, and L. Zheng, “Smartmask: Context aware high-fidelity mask generation for fine-grained object insertion and layout control,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6497–6506.
- [39] D. Epstein, A. Jabri, B. Poole, A. Efros, and A. Holynski, “Diffusion self-guidance for controllable image generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 16222–16239, 2023.
- [40] N. Matsunaga, M. Ishii, A. Hayakawa, K. Suzuki, and T. Narihira, “Fine-grained image editing by pixel-wise guidance using diffusion models,” arXiv preprint arXiv:2212.02024, 2022.
- [41] Y. Yang, H. Peng, Y. Shen, Y. Yang, H. Hu, L. Qiu, H. Koike et al., “Imagebrush: Learning visual in-context instructions for exemplarbased image manipulation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [42] Y. Shi, C. Xue, J. Pan, W. Zhang, V. Y. Tan, and S. Bai, “Dragdiffusion: Harnessing diffusion models for interactive point-based image editing,” arXiv preprint arXiv:2306.14435, 2023.
- [43] X. Pan, A. Tewari, T. Leimk¨uhler, L. Liu, A. Meka, and C. Theobalt, “Drag your gan: Interactive point-based manipulation on the generative image manifold,” in ACM SIGGRAPH 2023 Conference Proceedings, 2023, pp. 1–11.

- [44] T.-T. Nguyen, D.-A. Nguyen, A. Tran, and C. Pham, “Flexedit: Flexible and controllable diffusion-based object-centric image editing,” arXiv preprint arXiv:2403.18605, 2024.
- [45] W. Quan, J. Chen, Y. Liu, D.-M. Yan, and P. Wonka, “Deep learningbased image and video inpainting: A survey,” International Journal of Computer Vision (IJCV), pp. 1–34, 2024.
- [46] Z. Xu, X. Zhang, W. Chen, M. Yao, J. Liu, T. Xu, and Z. Wang, “A review of image inpainting methods based on deep learning,” Applied Sciences, vol. 13, no. 20, p. 11189, 2023.
- [47] M. Bertalmio, G. Sapiro, V. Caselles, and C. Ballester, “Image inpainting,” in International Conference and Exhibition on Computer Graphics and Interactive Techniques (SIGGRAPH), 2000, pp. 417–424.
- [48] A. Criminisi, P. P´erez, and K. Toyama, “Region filling and object removal by exemplar-based image inpainting,” IEEE Transactions on Image Processing, vol. 13, no. 9, pp. 1200–1212, 2004.
- [49] C. Zheng, T.-J. Cham, and J. Cai, “Pluralistic image completion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 1438–1447.
- [50] J. Peng, D. Liu, S. Xu, and H. Li, “Generating diverse structure for image inpainting with hierarchical vq-vae,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 10775–10784.
- [51] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “RePaint: Inpainting using denoising diffusion probabilistic models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11461–11471.
- [52] A. Razzhigaev, A. Shakhmatov, A. Maltseva, V. Arkhipkin, I. Pavlov,

I. Ryabov, A. Kuts, A. Panchenko, A. Kuznetsov, and D. Dimitrov, “Kandinsky: an improved text-to-image synthesis with image prior and latent diffusion,” arXiv preprint arXiv:2310.03502, 2023.

- [53] A. Liu, M. Niepert, and G. V. d. Broeck, “Image inpainting via tractable steering of diffusion models,” arXiv preprint arXiv:2401.03349, 2023.
- [54] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.
- [55] G. Zhang, J. Ji, Y. Zhang, M. Yu, T. Jaakkola, and S. Chang, “Towards coherent image inpainting using denoising diffusion implicit models,” 2023.
- [56] C. Corneanu, R. Gadde, and A. M. Martinez, “Latentpaint: Image inpainting in latent space with diffusion models,” in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 4334–4343.
- [57] S. Yang, L. Zhang, L. Ma, Y. Liu, J. Fu, and Y. He, “Magicremover: Tuning-free text-guided image inpainting with diffusion models,” arXiv preprint arXiv:2310.02848, 2023.
- [58] P. von Platen, S. Patil, A. Lozhkov, P. Cuenca, N. Lambert, K. Rasul, M. Davaadorj, and T. Wolf, “Diffusers: State-of-the-art diffusion models,” https://github.com/huggingface/diffusers, 2022.
- [59] S. Xie, Y. Zhao, Z. Xiao, K. C. Chan, Y. Li, Y. Xu, K. Zhang, and T. Hou, “Dreaminpainter: Text-guided subject-driven image inpainting with diffusion models,” arXiv preprint arXiv:2312.03771, 2023.
- [60] T. Yu, R. Feng, R. Feng, J. Liu, X. Jin, W. Zeng, and Z. Chen, “Inpaint anything: Segment anything meets image inpainting,” arXiv preprint arXiv:2304.06790, 2023.
- [61] S. Yang, X. Chen, and J. Liao, “Uni-paint: A unified framework for multimodal image inpainting with pretrained diffusion model,” in ACM International Conference on Multimedia (MM), 2023, pp. 3190–3199.
- [62] B. Kawar, S. Zada, O. Lang, O. Tov, H. Chang, T. Dekel, I. Mosseri, and M. Irani, “Imagic: Text-based real image editing with diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6007–6017.
- [63] D. Valevski, M. Kalman, Y. Matias, and Y. Leviathan, “Unitune: Textdriven image editing by fine tuning an image generation model on a single image,” arXiv preprint arXiv:2210.09477, 2022.
- [64] L. Han, S. Wen, Q. Chen, Z. Zhang, K. Song, M. Ren, R. Gao, Y. Chen, D. L. 0003, Q. Zhangli et al., “Improving tuning-free real image editing with proximal guidance.” CoRR, 2023.
- [65] G. Parmar, K. Kumar Singh, R. Zhang, Y. Li, J. Lu, and J.-Y. Zhu, “Zero-shot image-to-image translation,” in Special Interest Group on Computer Graphics and Interactive Techniques (SIGGRAPH), 2023, pp. 1–11.
- [66] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel, “Plug-and-play diffusion features for text-driven image-to-image translation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 1921–1930.
- [67] Y. Zhang, J. Xing, E. Lo, and J. Jia, “Real-world image variation by aligning diffusion inversion chain,” arXiv preprint arXiv:2305.18729, 2023.

- [68] B. Cheng, Z. Liu, Y. Peng, and Y. Lin, “General image-toimage translation with one-shot image guidance,” arXiv preprint arXiv:2307.14352, 2023.
- [69] Q. Wu, Y. Liu, H. Zhao, A. Kale, T. Bui, T. Yu, Z. Lin, Y. Zhang, and S. Chang, “Uncovering the disentanglement capability in text-toimage diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 1900– 1910.
- [70] M. Brack, F. Friedrich, D. Hintersdorf, L. Struppek, P. Schramowski, and K. Kersting, “Sega: Instructing diffusion using semantic dimensions,” arXiv preprint arXiv:2301.12247, 2023.
- [71] L. Tsaban and A. Passos, “LEDITS: Real image editing with ddpm inversion and semantic guidance,” arXiv preprint arXiv:2307.00522, 2023.
- [72] W. Dong, S. Xue, X. Duan, and S. Han, “Prompt tuning inversion for text-driven image editing using diffusion models,” arXiv preprint arXiv:2305.04441, 2023.
- [73] C. H. Wu and F. De la Torre, “Unifying diffusion models’ latent space, with applications to cyclediffusion and guidance,” arXiv preprint arXiv:2210.05559, 2022.
- [74] ——, “A latent space of stochastic diffusion models for zero-shot image editing and guidance,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7378–7387.
- [75] G. Couairon, J. Verbeek, H. Schwenk, and M. Cord, “Diffedit: Diffusion-based semantic image editing with mask guidance,” in International Conference on Learning Representations (ICLR), 2023.
- [76] Z. Zhang, L. Han, A. Ghosh, D. N. Metaxas, and J. Ren, “Sine: Single image editing with text-to-image diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6027–6037.
- [77] K. Joseph, P. Udhayanan, T. Shukla, A. Agarwal, S. Karanam, K. Goswami, and B. V. Srinivasan, “Iterative multi-granular image editing using diffusion models,” arXiv preprint arXiv:2309.00613, 2023.
- [78] G. Kim, T. Kwon, and J. C. Ye, “Diffusionclip: Text-guided diffusion models for robust image manipulation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 2426–2435.
- [79] A. Q. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mcgrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” in International Conference on Machine Learning (ICML). PMLR, 2022, pp. 16784–16804.
- [80] Z. Geng, B. Yang, T. Hang, C. Li, S. Gu, T. Zhang, J. Bao, Z. Zhang, H. Hu, D. Chen et al., “Instructdiffusion: A generalist modeling interface for vision tasks,” arXiv preprint arXiv:2309.03895, 2023.
- [81] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in IEEE/CVF International Conference on Computer Vision (ICCV), December 2015.
- [82] H. Huang, R. He, Z. Sun, T. Tan et al., “Introvae: Introspective variational autoencoders for photographic image synthesis,” Advances in Neural Information Processing Systems (NIPS), vol. 31, 2018.
- [83] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Ieee, 2009, pp. 248–255.
- [84] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in European Conference on Computer Vision (ECCV). Springer, 2014, pp. 740–755.
- [85] A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. PontTuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov et al., “The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale,” International Journal of Computer Vision (IJCV), vol. 128, no. 7, pp. 1956–1981, 2020.
- [86] F. Yu, A. Seff, Y. Zhang, S. Song, T. Funkhouser, and J. Xiao, “Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop,” arXiv preprint arXiv:1506.03365, 2015.
- [87] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” Advances in Neural Information Processing Systems (NIPS), vol. 35, pp. 25278–25294, 2022.
- [88] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.

- [89] Wikipedia contributors, “Peak signal-to-noise ratio — Wikipedia, the free encyclopedia,” 2024, [Online; accessed 4-March-2024]. [Online]. Available: https://en.wikipedia.org/w/index.php?title=Peak signal-to-noise ratio&oldid=1210897995

- [90] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595.
- [91] Wikipedia contributors, “Mean squared error — Wikipedia, the free encyclopedia,” 2024, [Online; accessed 4-March-2024]. [Online]. Available: https://en.wikipedia.org/w/index.php?title=Mean squared error&oldid=1207422018

- [92] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
- [93] C. Wu, L. Huang, Q. Zhang, B. Li, L. Ji, F. Yang, G. Sapiro, and N. Duan, “GODIVA: Generating open-domain videos from natural descriptions,” arXiv preprint arXiv:2104.14806, 2021.
- [94] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning (ICML). PMLR, 2021, pp. 8748–8763.
- [95] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or, “Nulltext inversion for editing real images using guided diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6038–6047.
- [96] D. Miyake, A. Iohara, Y. Saito, and T. Tanaka, “Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models,” arXiv preprint arXiv:2305.16807, 2023.
- [97] S. Li, J. van de Weijer, T. Hu, F. S. Khan, Q. Hou, Y. Wang, and J. Yang, “Stylediffusion: Prompt-embedding inversion for text-based editing,” arXiv preprint arXiv:2303.15649, 2023.
- [98] Z. Pan, R. Gherardi, X. Xie, and S. Huang, “Effective real image editing with accelerated iterative diffusion inversion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15912–15921.
- [99] Lykon, “Dreamshaper,” 2022. [Online]. Available: https://civitai.com/ models/4384?modelVersionId=128713
- [100] epinikion, “epicrealism,” 2023. [Online]. Available: https://civitai.com/ models/25694?modelVersionId=143906
- [101] heni29833, “Henmixreal,” 2024. [Online]. Available: https://civitai. com/models/20282?modelVersionId=305687
- [102] Meina, “Meinamix,” 2023. [Online]. Available: https://civitai.com/ models/7240?modelVersionId=119057
- [103] SG161222, “Realisticvision,” 2023. [Online]. Available: https://civitai. com/models/4201?modelVersionId=130072

[Figure 419]

Yaowei Li is currently pursuing his Ph.D. degree at Peking University. His research interests primarily focus on image/video generation and editing, controllable and interactive media synthesis, and multimodal processing. He has published several papers in prestigious conferences, including ICCV, ICLR, AAAI, SIGGRAPH, ACL, and EMNLP. He actively contributes to the academic community by serving

- as a reviewer for leading conferences such as ECCV, NeurIPS, AAAI, CVPR, and ICML.

[Figure 420]

Yuxuan Bian is currently pursuing a Ph.D. degree

- at The Chinese University of Hong Kong, under the supervision of Qiang Xu. His research interests include controllable image and video generation, as well as human motion generation.

Ying Shan (Senior Member, IEEE) is a distinguished scientist with Tencent, and the director of the ARC Lab, Tencent PCG. Before joining Tencent, he worked at Microsoft Research as a post-doc researcher, SRI International (Sarnoff Subsidiary) as a senior MTS, and Microsoft Bing Ads as a principal scientist manager. He has published more than 70 papers in top conferences and journals in the areas of computer vision, machine learning, and data mining, served as ACs of CVPR and senior PC of KDD, and holds a number of US/International patents. He is

[Figure 421]

currently leading R&D efforts in web search, and content AI for a suite of social media and content distribution products.

Yuexian Zou (Senior Member, IEEE) is currently a Full Professor with Peking University and the Director of the Advanced Data and Signal Processing Laboratory in Peking University and serves as the Deputy Director of Shenzhen Association of Artificial Intelligence (SAAI). She was a recipient of the award Leading Figure for Science and Technology by Shenzhen Municipal Government in 2009. She conducted more than 20 research projects including NSFC and 863 projects. She has published more than 280 academic papers in famous journals and

[Figure 422]

flagship conferences, and issued nine invention patents. Her research interests are mainly in intelligent signal and information processing, human-computer voice interaction, video and image processing, and machine learning.

[Figure 423]

Xuan Ju is a Ph.D. student at The Chinese University of Hong Kong. Her research focuses on image and video generation, multimodal image/video synthesis, and human-centric visual perception and generation. She has published papers in leading conferences, including CVPR, ECCV, ICCV, NeurIPS, ICLR, and ICML. Additionally, she has organized CVPR workshops and served as a reviewer for toptier conferences.

[Figure 424]

Qiang Xu is a Professor at The Chinese University of Hong Kong. His research interests include computer vision, large language models, and electronic design automation (EDA). He has published over 200 papers in related fields with more than 11,000 citations, including several best paper awards at prestigious conferences and an ICCAD Ten Year Retrospective Most Influential Paper award.

[Figure 425]

Zhaoyang Zhang is currently a Senior Research Scientist in ARC Lab, Tencent. He received his Ph.D. degree from The Chinese University of Hong Kong in 2024. His research interests include machine learning, visual generation, and visionlanguage processing. He has published papers in leading conferences, including CVPR, ICCV, ECCV, NeurIPS, ICML, and ICLR.

[Figure 426]

Junhao Zhuang is a Master student in Computer Technology at Tsinghua University, advised by Professor Chun Yuan. He earned his Bachelor’s degree in Computer Science and Technology from the University of Electronic Science and Technology of China. His research focuses on diffusion models, image/video generation, and editing. He has published papers at conferences such as ECCV, ACM MM, and ICASSP.

