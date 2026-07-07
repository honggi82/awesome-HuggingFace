# arXiv:2412.01824v2[cs.CV]27Aug2025

### X-Prompt: Towards Universal In-Context Image Generation in Auto-Regressive Vision Language Foundation Models

Zeyi Sun1,2, Ziyang Chu2,3, Pan Zhang2, Tong Wu4, Yuhang Zang2, Xiaoyi Dong2, Yuanjun Xiong6, Dahua Lin2,4,5, Jiaqi Wang†2

1Shanghai Jiao Tong University 2Shanghai AI Laboratory 3Tsinghua University 4The Chinese University of Hong Kong 5CPII under InnoHK 6MThreads AI.

###### Abstract

In-context generation is a key component of large language models’ (LLMs) open-task generalization capability. By leveraging a few examples as context, LLMs can perform both in-domain and out-of-domain tasks. Recent advancements in auto-regressive vision-language models (VLMs) built upon LLMs have showcased impressive performance in text-to-image generation. However, the potential of in-context learning for general image generation tasks remains largely unexplored. To address this, we introduce X-Prompt, a purely auto-regressive large-vision language model designed to deliver competitive performance across a wide range of both seen and unseen image generation tasks, all within a unified in-context learning framework. X-Prompt incorporates a specialized design that efficiently compresses valuable features from in-context examples, supporting longer in-context token sequences and improving its ability to generalize to unseen tasks. A unified training task for both text and image prediction enables XPrompt to handle general image generation with enhanced task awareness from in-context examples. Extensive experiments validate the model’s performance across diverse seen image generation tasks and its capacity to generalize to previously unseen tasks.

###### 1. Introduction

Extracting knowledge from a few examples and applying it to novel tasks at inference time has long been a major challenge in machine learning. With the significant success of large language models (LLMs) like GPT-3 [8] in purely NLP tasks, it has been demonstrated that even a few examples can lead to substantial performance improvements. Previous research has attempted to adapt this capability to computer vision tasks, achieving promising results

† Corresponding author.

in vision-only tasks [4, 22, 73, 74] with pure vision models. However, for tasks that require high-level semantics or text prompt control, like image editing and image personalization, it is important to achieve multi-modal in-context learning. With the success of multi-modal foundation models, the research focus has shifted to unified multi-modal in-context image generation.

The field of image generation is currently dominated by diffusion models [20, 28, 40, 55, 62] like SD [57], which typically rely on a text encoder alongside a diffusion network. This structure inherently complicates support for incontext learning, as it requires multi-image understanding and reasoning capabilities. Previous approaches [16, 60, 78] like Emu [64] and SEED-X [24] that bridge diffusion and LLMs often rely on predicted embeddings by LLMs, which introduce huge information loss of image conditions, limiting their abilities to preserve details in editing or lowlevel vision tasks. Recently, works like Chameleon [68], Transfusion [95] and concurrent works [75, 77, 79–81] have proposed approaches where an LLM directly predicts VQVAE [71] or VAE [33] features in an auto-regressive or diffusion manner. This approach reduces information loss during image compression, preserving more visual detail while integrating the LLM’s reasoning capabilities. However, limited research has explored in-context image generation on these foundation models.

The primary challenge of in-context learning on these foundation models is the substantial context length required during training. To retain the information in an image, VQVAE or VAE image features require a large number of tokens—typically around (18)2 or (161 )2 of the total image pixels. A single image typically requires 1024–4096 tokens. In an image-to-image in-context task, at least four images are necessary for context, leading to a prohibitive training context length. This limitation, also noted by [80], restricts support to only three images during training, making direct in-context training impractical due to the excessive context

The light yellow car in <img1> is changed into red.

[Figure 1]

Output

Output

In-context Prompt

###### In-context Prompt

Chameleon

Chameleon

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Provide same transformation on this image for me

What’s the relationship difference between <img1> and <img2> ?

Input

Input X-Prompt

X-Prompt

In-context Prompt Input Result In-context Prompt Input Result

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

A woman with long golden hair, waring a white toga

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

[Figure 44]

[Figure 45]

###### Input Result

###### Input Result

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Add the flag of <img1> to the plane in <img2>

Make the girl in <img2> dress cloth in <img1>

- Figure 1. X-Prompt can perform multi-modal generation based on in-content examples in a pure auto-regressive foundation model.

size.

Moreover, another challenge is improving the ability of foundation models to interpret the intent behind in-context prompts. Since these prompts often consist of several images that implicitly convey the target task without explicit explanations, it is crucial for foundation models to effectively identify and describe the differences or changes between each pair of images.

To address these challenges, we introduce X-Prompt method to compress the information within in-context ex-

amples. Our approach enables the model to distill the content of examples into a sequence of fixed-length compression tokens. During inference, these tokens serve as contextual information for reasoning on new images, effectively reducing the maximum context length required during training. Additionally, compressed tokens of the context enhance the interpretation of target tasks, showing improved generalizability to previously unseen tasks in experiments. Furthermore, unlike pure vision prompt or nature language prompt, Our X-Prompt supports multi-modal

ℝ ×

[Figure 52]

Img1 is dark version of img2

Incontext Example Tokens

IE

[Figure 53]

ℝ ×

ℝ ×

##### Chameleon

X-Prompt Tokens

XP

###### X-Prompt

[Figure 54]

[Figure 55]

[Figure 56]

TODO Tokens

TD

What’s the relation between two images

IE XP TD

- Figure 2. Attention masking of X-Prompt for context feature compression and unified text and image next token prediction training.

prompt for more diversified tasks like image editing and style-personalization. By constructing reversed tasks and text prediction tasks like generating descriptions of image differences, we enhance the model’s overall performance and generalization capabilities. We show some visualization results in Fig. 1, where our model can achieve high quality in-context learning on diversified tasks. Our work makes three major contributions:

- • We propose the X-Prompt method to effectively distill useful information from in-context examples into compressed tokens, improving its performance while reducing the previously prohibitive training context length.
- • By unifying image generation and image description tasks, we significantly enhance Chameleon [68]’s image generation capabilities.
- • We integrate image generation, editing, dense prediction, and low-level vision tasks into a unified in-context learning framework, demonstrating the effectiveness and generalization of in-context learning across seen and unseen tasks in a pure auto-regressive vision-language foundation model.

###### 2. Related Work

Large Vision-Language Models (LVLMs). The emergence of large language models (LLMs) [1, 9, 14, 29, 49, 70] have made remarkable breakthroughs. The domain of research has increasingly turned its attention toward Large Vision-Language Models (LVLMs). Previous advances in this field focus on the integration of vision understanding capabilities with LLMs [2, 17, 30, 37, 42, 48, 51, 52, 66, 82, 91]. Recent works start to focus on integrating vision generation abilities. One early line of these works [23, 24, 31, 64, 65, 78] compress visual features with LLMs into compressed embeddings and use diffusion decoder (like SDXL [50]) to generate visual contents, However, this suffers great information loss during LLM encoding process, leading to unsatisfying results in image editing tasks. Another line of work, pioneered by Chameleon [68], uses unified image tokens from VQ-VAE [19, 71] to unify perception and generation. In this work, we aim to fully unlock the potential of Chameleon for general image gener-

ation in a unified in-context learning paradigm for both seen and unseen tasks.

Auto Regressive based Image Generation. While previous state of the art image generation dominant by diffusion models [10, 20, 27, 28, 39, 40, 53, 55, 57, 62, 84], recent works on auto-regressive for image generation have shown promising results [41, 44, 45, 63, 69, 75, 79, 92]. However, these models only shows experiments results on textto-image generation [38, 63, 67, 77, 79, 79], with limited research on other types of image generation tasks or lack of quantitative results [64, 88]. In this work, we not only enhance text-image alignment but also extend our exploration to diverse image generation tasks, including image editing, controlled image generation, and perception tasks such as semantic segmentation and depth estimation. We demonstrate that auto-regressive models can achieve competitive results across these tasks in a unified framework.

In-Context Learning. GPT-3 [8] introduced the paradigm of in-context learning, where diverse NLP tasks are reformed as text completion tasks, enhanced through prompts containing embedded examples, which significantly boosts performance on related tasks. In-context learning has also been explored in the vision domain [4, 5, 73, 74], but these models are limited to vision-only tasks, lacking multimodal versatility. Multi-modal models like Emu-1/2 [64, 65] demonstrate in-context learning capabilities; however, their reliance on embeddings predicted by LLM from image features and the integration of an external diffusion model restrict their effectiveness in image editing and dense prediction tasks. In contrast, our approach utilizes a unified early-fusion representation based on Chameleon [68], enabling a single model to generalize across a broader array of tasks with improved performance and enhanced generalizability.

###### 3. Method

###### 3.1. In-Context Example Compression

We introduce a context-aware compression mechanism to better model in-context examples within Chameleon. As illustrated in Fig. 2. This mechanism defines three types of

The second image appears to have rain added to first image

Image Editing Controlled Image Image Composition Generation

Perception Task

Detection Style Transfer

Object Removal

Object Combination

Depth Estimation

Canny2image

difference description

[Figure 57]

Object Replacement

Season/Time Change

Attribute Assignment

Norm Estimation

Depth2image

[Figure 58]

[Figure 59]

Original Task

Object Additioin

Object Modification

Vitual Try-on

Derain

Segmentation

Norm2image

reverse input output

Low Level Vision

[Figure 60]

[Figure 61]

Reverse Task

Image Derain

Image Denoise

Image Dehase

Image Deblurring

Image SuperRes

Low Light Enhansement

Add rain

- Figure 3. Training data pair augmentation and list of training prototype tasks and subtasks. We introduce reverse task and difference description task through next text token prediction to improve the performance and generalizibility.

tokens: In-Context Example Tokens (IE), X-Prompt Tokens (XP), and TODO Tokens (TD). Given an in-context example as a prompt, Chameleon embeds it into a feature space, producing In-Context Example Tokens XIE ∈ RIE×C, where IE denotes the number of tokens and C the feature dimension. The model further includes learnable X-Prompt Tokens, represented as XXP ∈ RS×C, where S is the number of learnable tokens optimized during training.

To encourage the model to store contextual information in XXP, we disconnect the relationship between XIE and the TODO Tokens XTD through attention masking, forcing the model to rely on X-Prompt Tokens for context representation. The model then generates the TODO Tokens XTD ∈ RTD×C, with TD as the target sequence length, by sequentially maximizing the conditional probability of each token in XTD given XXP and all previously generated tokens XTD

, as formulated below:

<t

pθ(XTD | XXP) =

=

TD

t | XXP,XTD

pθ(XTD

<t

t=1

TD

softmax(f(XXP,XTD

<t

t=1

)

;θ))X

T Dt

where XTD

denotes the t-th token in the TODO sequence, and XTD

t

represents all previously generated TODO tokens with model f, parameterized by weights θ. This approach enables Chameleon to effectively compress valuable features from in-context examples, enhancing its capability for context-aware generation and reduce training context length.

<t

###### 3.2. Task Augmentation Pipeline

As Chameleon [68] possess both text and image generation ability, we also adopt unified training on interleaved text and image generation. As illustrated in Fig. 3, to further augments the training data, We construct a data generation

pipeline. Each generation task is converted into a text prediction task that focuses on describing the relationship between the input and output images. This text prediction task requires the model to interpret and articulate the relational changes between the images. Advanced vision-language models, such as GPT-4V [48] or QwenVL-2 [72], are employed for this purpose, as they can generate and interpret open-ended relational descriptions. The detailed prompts are available in Appendix C.1. Through training the model to describe differences by generating text tokens, we equip it with a deeper understanding of the relationship of input and output images, which enhances its generalization ability and improves the performance of image generation.

Additionally, we introduce a task-reversion augmentation. For each task, such as “deraining” (removing rain from an image), we introduce a reverse task—“adding rain”—by swapping the input and output. This strategy effectively doubles the task variety, enabling the model to learn transformations in both directions and deepening its comprehension of the underlying transformation processes.

###### 3.3. Retrieval-Augmented Image Editing

Following the spirit of Retrieval-Augmented Generation (RAG) [34]. We introduce Retrieval-Augmented Image Editing (RAIE) to enhances image editing by retrieving the relevant examples from a database. Given an input image Iinput and an instruction Instrcurrent, RAIE searches for the most similar instruction Instrretrieved and corresponding image editing pairPretrieved in the database. The retrieval process is defined as:

(Instrretrieved,Pretrieved) = argmin

dist(Instr,Instrcurrent).

(Instr,P)∈D

where we use the cosine similarity of CLIP [54] text features as the distance function dist. This retrieved example serves as in-context guidance for the model, which then

Type Model Single Obj. Two Obj. Counting Colors Position Color Attri. Overall

LDM [57] 0.92 0.29 0.23 0.58 0.02 0.05 0.37

- SD-1.5 [57] 0.97 0.38 0.35 0.76 0.04 0.06 0.43

- SD-2.1 [57] 0.98 0.51 0.44 0.85 0.07 0.17 0.50

Diffusion

DALL-E 2 [56] 0.94 0.66 0.49 0.77 0.10 0.19 0.52 Show-o [81] 0.95 0.52 0.49 0.82 0.11 0.28 0.53 SDXL [50] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 DALLE 3 [6] 0.96 0.87 0.47 0.83 0.43 0.45 0.67

LLamaGen [63] 0.71 0.34 0.21 0.58 0.07 0.04 0.32 Emu3Gen [75] 0.98 0.71 0.34 0.81 0.17 0.21 0.54 Chameleon [68] - - - - - - 0.39 Ours 0.97 0.69 0.28 0.71 0.14 0.15 0.49 (+0.10)

Auto-regressive

Ours (+text pred.) 0.98 0.73 0.33 0.85 0.26 0.28 0.57

∆ +0.01 +0.04 +0.05 +0.14 +0.12 +0.04 +0.08

- Table 1. Evaluation of text-to-image generation ability on GenEval [26] benchmark. Unifying image dense description task through next text token prediction can significantly improve the text-image alignment of images generated by Chameleon [68].

generates the edited output Ioutput based on both Instrcurrent and the retrieved pair (Instrretrieved,Pretrieved):

Ioutput = Model(Iinput,Instrcurrent,Instrretrieved,Pretrieved).

It is worth noticing that dist can be extended to other function beyond simply compute text feature similarity. This automated retrieval process reduces the need for manual intervention and can also be customized by users to achieve more precise, tailored image editing. RAIE is an optional choice, and we apply this method only when specifically mentioned in the experiment section.

By leveraging in-context examples, RAIE enhances the consistency and accuracy of image editing tasks. RAIE is naturally suited for a unified auto-regressive model. However, it poses significant challenge for current state-of-theart diffusion models, which rely solely on text encoders and lack comprehensive understanding capabilities.

###### 4. Experiments

Though we train a unified model across different tasks, we report the data preparation process separately in each subsection for clarity. The complete training dataset comprises approximately 5 million data pairs, expanding to around 8 million pairs after task reversion and text prediction task augmentation (detailed in Appendix B). For all tasks that take an image with text as input and produce an image with text as output, we include an in-context example of the same task type. We set the batch size to 1024 and the context window size to 5120. The learning rate is set to 1e-4 with a cosine learning rate scheduler. Training is conducted on 128 NVIDIA A100-80G GPUs over 20,000 steps.

###### 4.1. Text-to-Image Generation

Settings. To enhance Chameleon’s [68] text-to-image generation capabilities, we utilize QWen-VL2 [72] to rewrite dense descriptive captions for 500K high-quality images

filtered from the LAION-Aesthetic [59] dataset (detailed in Appendix C.4), selecting only images with an aesthetic score greater than 6. For evaluation, we adopt the GenEval [26] benchmark.

Results. As shown in Tab. 1, we significantly enhance Chameleon’s original text-to-image generation capabilities. Leveraging the image dense description task, our model further achieves competitive results compared to other autoregressive models for text-to-image generation. This experiment highlights the effectiveness of unifying dense image description and image generation tasks, resulting in notable improvements, particularly in tests involving complex multi-object and color attributes. The ability to generate images that accurately follow text prompts is essential for our downstream applications like image-editing. Qualitative visualization are available in Appendix A.

###### 4.2. Image Dense Prediction

Settings. We use representative datasets for dense prediction tasks: NYU-v2 [61] for depth and surface normal estimation, ADE-20K [94] for semantic segmentation, and Rain-13K [21], LOL [76], and GoPro [47] for corresponding low-level vision tasks. Full training data details are available in the Appendix B.

Results. We select typical specialist model and previous vision generalist for comparison. Results are shown in Tab. 2, where our model can achieve competitive results on dense prediction task and low level vision task. Our model is the first to deliver promising results using a unified, discrete token approach. The slight performance gap on low level vision task compared to continuous feature prediction models is due to the inherent information loss in the VQ-VAE discretization process, as Chameleon [68] adopt 16x compression rate. However, by discretizing images and adopting a next-token prediction approach akin to that used in large language models, our method offers promising scalability for future advancements.

Depth Est. Semantic Seg. Surface Normal Est. Lowlight Enhans. Deblur Derain RMSE↓ mIoU↑ Mean Angle Error↓ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ NYUv2 ADE20K NYU-Depth V2 LOL GoPro Rain100L

Type Methods

DepthAnything [83] 0.206 Marigold [32] 0.224

Mask DINO [36] 60.80 Mask2Former [13] 56.10

Domain Specific Model

Bae et al. [3] 14.90 InvPT [86] 19.04

AirNet [35] 18.18 0.735 24.35 0.781 32.98 0.951 InstructIR [15] 23.00 0.836 29.40 0.886 36.84 0.937

Painter [73] 0.288 49.90 × 22.40 0.872 × × 29.87 0.882 InstructCV [22] 0.297 47.23 × × × × × × ×

Unified Model (continuous)

InstructDiffusion [25] × × × × × 23.58 - 19.82 0.741 OmniGen [80] 0.480 × × 13.38 0.392 13.39 0.321 12.02 0.233

Unified-IO [43] 0.387 25.71 - × × × × × ×

Unified Model (discrete)

Lumina-mGPT [41] × 20.87 22.10 × × 17.59 0.536 16.61 0.365 Ours 0.277 31.21 19.17 19.71 0.810 21.04 0.761 25.53 0.843

- Table 2. Comparison of X-Prompt with task-specific and vision generalist baselines across six representative tasks, covering both high-level visual understanding and low-level image processing. ’×’ indicates that the method is incapable of performing the task.

Input MagicBrush w/o context w/ context context example

Input MagicBrush w/o context w/ context context example

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

make the woman wear a mask

The bed should be blue

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

The dog should have a bone with him

Can the refrigerator be blue

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

###### Replace the sticker with a smiley face sticker Remove the people

Figure 4. Qualitative Results on MagicBrush [89] testset comparing with MagicBrush results w/ and w/o context examples.

###### 4.3. Image Editing with RAIE

Settings. As we introduced Retrieval-Augmented Image Editing (RAIE) in Sec. 3.3, we also prepare training data in the same way. We use publicly available UltraEdit [93] (500K) and MagicBrush [89] (8K) For the training. We use CLIP-B/32 [54] text encoder to encode the the edit instruction for each training sample and retrieval the most similar instruction feature as its neighbor sample (excluding the sample itself). During training, we input the neighboring sample as a task prompt context and prompt the model to predict the edited image. Besides generation task, we also use QWen-VL2 [72] to describe the differences between the input image and the edited image. We add these difference description tasks into the training to help model gain better understanding of images and their variations. Preparing each editing pair with a similar editing pair for in-context learning is crucial to the success of RAIE, as we frequently observe similar editing pairs in both the UltraEdit and MagicBrush datasets. We provide detailed analysis of this in Appendix D.

For evaluation, We use MagicBrush [89] benchmark, we also encode the edit instuction using CLIP-B/32 text encoder. We only use MagicBrush training set as reference database to perform Retrieval-Augmented Image Editing (RAIE) proposed in Sec. 3.3. The testing metics and CLIP and DINO [12] models are consistent with [89, 93].

Type Methods CLIPdir↑ CLIPout↑ CLIPimg↑ DINO↑

InstructPix2Pix [7] 0.081 0.276 0.852 0.750 MagicBrush [89] 0.106 0.278 0.933 0.899 UltraEdit [93] 0.093 0.274 0.899 0.848

Continuous

Lumina-mGPT [41] 0.025 0.253 0.810 0.751 Ours (w/o text pred.) 0.067 0.263 0.823 0.785 Ours (w/ text pred.) 0.083 0.271 0.857 0.781 Ours + RAIE 0.097 0.279 0.862 0.792

Discrete

Table 3. Image Editing Results. Comparison of different methods on the MagicBrush [89] testset.

Results. As shown in Tab. 3, training solely with image editing pairs does not yield satisfactory results, as the model tends to replicate the original image rather than apply mean-

Low Light Enhancement Derain Object Addition Object Removal Depth Estimation

Settings

LOL Rain100H InstructP2P [7] NYU-v2 PSNR↑ SSIM↑ PSNR↑ SSIM↑ CLIPdir↑ CLIPout↑ CLIPdir↑ CLIPout↑ RMSE↓

OmniGen [80] (in-context) 8.923 0.243 13.14 0.411 0.054 0.243 0.031 0.233 ×

Full training 19.71 0.770 21.55 0.633 0.112 0.283 0.103 0.265 0.279 No In-context 9.140 0.253 7.924 0.212 -0.031 0.252 0.023 0.244 0.745 In-context w/o X-Prompt 17.00 0.633 18.10 0.509 0.092 0.262 0.069 0.246 0.390 In-context w/ X-Prompt 17.22 0.653 18.91 0.512 0.092 0.274 0.073 0.251 0.352

Table 4. Results of in-context learning in novel task settings. “Full training” denotes for model trained with corresponding training set. While the other settings evaluate performance on tasks not encountered during training.

###### Context example Input w/ context w/o context

###### Context example Input w/ context

w/o context

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

OmniGen

perform depth estimation

add a castle

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Ours

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

OmniGen

perform image derain

remove the sun

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Ours

Figure 5. Novel task in-context testing compared to OmniGen [80]. X-Prompt can achieve novel task generalization with a given example. While OmniGen [80] fall short in in-context learning (such as adapting to new color spectrum or preserve details when adding object to the image).

ingful edits. Incorporating the additional difference description task through training model with next text token prediction encourages the model to identify distinctions between the input and edited images. This significantly improve the overall performance. Testing with an editing pair retrieved from the training set serving as an in-context example further enhances the quality of the edits, which proves the effectiveness of RAIE. Qualitative results are presented in Fig. 4. This experiment demonstrates the effectiveness of our in-context learning strategy for image editing tasks that require advanced semantic comprehension.

###### 4.4. In-Context Learning on Novel Tasks

Settings. Following the approach of GPT-3 [46], this experiment primarily investigates the generalizability of our model on novel tasks, given only a single example as context. We choose Low light enhancement and Image derain from low-level vision, object addition and object removal from Image Editing as novel task to perform this study. During training, we remove the training data for each

novel tasks. For Image derain, we remove the training of generating derained image of Rain-13K. For low light enhancement, we remove the training of generating enhanced image of Mit-5K [11] and LOL [76]. For image editing task. we filtered out the sample in Ultra-Edit [93] and MagicBrush [89] using LLama-3-Instruct-7B [18] by querying whether the instruction involves object addition or removal. Our test is perform on Rain100H [85] (Image derain), LOLval (low light enhensement) and manually filtered 100 editing samples of Object Addition/ Removal task from publicly available instructP2P [7] dataset. For depth estimation we test a new color palette that match the distance of the pixel on image to a different color spectrum. For low level vision, we report PSNR and SSIM as image evaluation metric. For Image Editing, we report CLIP [54] score consistent with [89, 93]. For depth estimation task, we report the RMSE on the previous unseen color spectrum.

Results. We report the quantitative results in Tab. 4. “Full training” refers to the model fully trained to corresponding training set. Both “In-context” and “No In-context” settings

Style as Context

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Action as Context

[Figure 127]

[Figure 128]

[Figure 129]

An old man with a straw hat and a Hawaiian shirt A woman dressed in a burgundy velet gown and gloves A girl with blonde hair, dressed in blue uniform

Figure 6. X-Prompt can support diversified context to achieve style personalization and action preservation.

are trained with the corresponding training data completely removed. On novel task, in-context example can significantly improve results. In contrast, the model generally failed to perform the task without the in-context example. We also ablate the effectiveness of attention masking proposed in Sec. 3.1, where X-Prompt’s attention mask force the model compresses the in-context information implicitly. This achieves slightly improved result compared to standard causal masking.

We also compare our in-context learning capabilities with OmniGen, where we input three image follow the same prompt template in [80]. As shown in both Tab. 4 and Fig. 5, due to prohibitive context-length (with the image resolution already reduced to 512x512), OmniGen struggled to achieve generalized in-context learning. In depth estimation, it fails to generalize to unseen color spectrums. In image editing, OmniGen is unable to keep unchanged parts of the image consistent with the original, nor did it effectively follow contextual cues. For image deraining, the model struggled to interpret the context accurately, leading to unexpected results. In contrast, our model, leveraging unified text and image next-token prediction loss, demonstrates superior generalization to previously unseen tasks.

###### 4.5. Other In-context Form

Settings. In addition to training our model on existing datasets, we also create two small datasets for style personalization and action preservation to demonstrate XPrompt’s ability to extract diverse contextual information. For style-personalization, we use RB-Modulation [58] to generate image pairs based on style image and further filter low quality data with QWen-VL2 [72] (detailed in Appendix C.2). For action preservation, we generate diversified human actions and use pose estimation model and Con-

trolNet [90] to generate different person doing same action in the same pose. For each task, we generate 10K pairs for in-context learning. For style personalization, we give model an example transformation pair to prompt model perform similar transformation on an unseen image. For action and pose preservation, we give model two image of a person doing same action in similar pose and a new person description and prompt model to generate a new image.

Results. We show qualitative results in Fig. 6. X-Prompt can extract both the high level semantics and low level details of the context example and perform successful transformation on new image or generation based on text prompt. This experiments demonstrate that X-Prompt can achieve diversified in-context form in arbitrary multi-modal tasks.

###### 5. Discussion

In this work, we propose empowering the autoregressive foundation model Chameleon [68] to achieve unified image generation through in-context learning. We demonstrate its promising performance across tasks such as text-to-image generation, dense prediction, low-level vision, and image editing, and showcase its generalizability to previously unseen tasks when provided with in-context examples. We hope this work will pave way for this promising direction to achieve the “GPT-3 moment” in the unified multi-modal field in image generation.

While promising, our work still faces several unresolved challenges. First, the VQ-VAE in our base model, Chameleon [68], introduces substantial information loss in image reconstruction at a compression rate of 16, which is a primary reason for the model’s reduced performance on certain low-level vision tasks that demand high-quality image reconstruction. Second, X-Prompt achieves success-

ful in-context generalization only when a sub-task within a prototype task (as shown in Fig. 3) is excluded from training, with limited generalization across different prototype tasks. We believe that more comprehensive and diversified multi-modal pretraining is essential to bridge the gaps between different prototype tasks, ultimately achieve the “GPT-3 moment” in multi-modal learning.

###### 6. Acknowledgment

This project is funded in part by Shanghai Artificial lntelligence Laboratory, the National Key R&D Program of China (2022ZD0160201), the Centre for Perceptual and Interactive Intelligence (CPII) Ltd under the Innovation and Technology Commission (ITC)’s InnoHK. Dahua Lin is a PI of CPII under the InnoHK.

###### References

- [1] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Tachard Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Z. Chen, Eric Chu, J. Clark, Laurent El Shafey, Yanping Huang, Kathleen S. Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan A. Botha, James Bradbury, Siddhartha Brahma, Kevin Michael Brooks, Michele Catasta, Yongzhou Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, C Cr´epy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, M. C. D’iaz, Nan Du, Ethan Dyer, Vladimir Feinberg, Fan Feng, Vlad Fienber, Markus Freitag, Xavier Garc´ıa, Sebastian Gehrmann, Lucas Gonz´alez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, An Ren Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wen Hao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Mu-Li Li, Wei Li, Yaguang Li, Jun Yu Li, Hyeontaek Lim, Han Lin, ZhongZhong Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alexandra Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Marie Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniela Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Ke Xu, Yunhan Xu, Lin Wu Xue, Pengcheng Yin, Jiahui Yu, Qiaoling Zhang, Steven Zheng, Ce Zheng, Wei Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report. ArXiv, abs/2305.10403, 2023. 3
- [2] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Yitzhak Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Worts-

- man, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. ArXiv, abs/2308.01390, 2023. 3
- [3] Gwangbin Bae, Ignas Budvytis, and Roberto Cipolla. Estimating and exploiting the aleatoric uncertainty in surface normal estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13137–13146,

2021. 6

- [4] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan L Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22861–22872, 2024. 1, 3
- [5] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei Efros. Visual prompting via image inpainting. Advances in Neural Information Processing Systems, 35:25005–25017, 2022. 3
- [6] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 5
- [7] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 6, 7
- [8] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. 1, 3
- [9] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. ArXiv, abs/2005.14165, 2020. 3
- [10] Jiazi Bu, Pengyang Ling, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Broadway: Boost your text-to-video generation model in a training-free way. arXiv preprint arXiv:2410.06241, 2024. 3
- [11] Vladimir Bychkovsky, Sylvain Paris, Eric Chan, and Fr´edo Durand. Learning photographic global tonal adjustment with a database of input / output image pairs. In The TwentyFourth IEEE Conference on Computer Vision and Pattern Recognition, 2011. 7
- [12] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 6
- [13] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask

- transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 6
- [14] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garc´ıa, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark D´ıaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1–240:113, 2022. 3
- [15] Marcos V Conde, Gregor Geigle, and Radu Timofte. Instructir: High-quality image restoration following human instructions. In European Conference on Computer Vision, pages 1–21. Springer, 2025. 6
- [16] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499,

2023. 1

- [17] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form textimage composition and comprehension in vision-language large model, 2024. 3
- [18] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 7

- [19] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 3
- [20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1, 3

- [21] Xueyang Fu, Jiabin Huang, Delu Zeng, Yue Huang, Xinghao Ding, and John Paisley. Removing rain from single images via a deep detail network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3855–3863, 2017. 5
- [22] Yulu Gan, Sungwoo Park, Alexander Schubert, Anthony Philippakis, and Ahmed M Alaa. Instructcv: Instructiontuned text-to-image diffusion models as vision generalists. arXiv preprint arXiv:2310.00390, 2023. 1, 6
- [23] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218, 2023. 3
- [24] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024. 1, 3
- [25] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, et al. Instructdiffusion: A generalist modeling interface for vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12709–12720, 2024. 6
- [26] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 5
- [27] Runze He, Kai Ma, Linjiang Huang, Shaofei Huang, Jialin Gao, Xiaoming Wei, Jiao Dai, Jizhong Han, and Si Liu. Freeedit: Mask-free reference-based image editing with multi-modal instruction. arXiv preprint arXiv:2409.18071,

2024. 3

- [28] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 3
- [29] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and L. Sifre. Training compute-optimal large language models. ArXiv, abs/2203.15556, 2022. 3
- [30] Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Deciphering cross-modal alignment in large vision-language models with modality integration rate. arXiv preprint arXiv:2410.07167, 2024. 3
- [31] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8362– 8371, 2024. 3
- [32] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurpos-

- ing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492– 9502, 2024. 6
- [33] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1
- [34] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020. 4
- [35] Boyun Li, Xiao Liu, Peng Hu, Zhongqin Wu, Jiancheng Lv, and Xi Peng. All-in-one image restoration for unknown corruption. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17452– 17462, 2022. 6
- [36] Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M Ni, and Heung-Yeung Shum. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3041–3050, 2023. 6
- [37] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ArXiv, abs/2301.12597, 2023. 3
- [38] Xiang Li, Hao Chen, Kai Qiu, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024. 3
- [39] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338, 2024. 3
- [40] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-toimage alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 1, 3
- [41] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 3, 6
- [42] Ziyu Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Conghui He, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Mia-dpo: Multi-image augmented direct preference optimization for large vision-language models. arXiv preprint arXiv:2410.17637, 2024. 3
- [43] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In The Eleventh International Conference on Learning Representations, 2022. 6
- [44] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source

- project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 3
- [45] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-toimage generation via auto-regressive representations. arXiv preprint arXiv:2406.10797, 2024. 3
- [46] Ben Mann, N Ryder, M Subbiah, J Kaplan, P Dhariwal, A Neelakantan, P Shyam, G Sastry, A Askell, S Agarwal, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 1, 2020. 7
- [47] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3883–3891,

2017. 5

- [48] OpenAI. Gpt-4v(ision) system card. OpenAI, 2023. 3, 4
- [49] R OpenAI. Gpt-4 technical report. arXiv, pages 2303– 08774, 2023. 3
- [50] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3, 5
- [51] Zhangyang Qi, Ye Fang, Zeyi Sun, Xiaoyang Wu, Tong Wu, Jiaqi Wang, Dahua Lin, and Hengshuang Zhao. Gpt4point: A unified framework for point-language understanding and generation. arXiv preprint arXiv:2312.02980, 2023. 3
- [52] Zhangyang Qi, Ye Fang, Zeyi Sun, Xiaoyang Wu, Tong Wu, Jiaqi Wang, Dahua Lin, and Hengshuang Zhao. Gpt4point: A unified framework for point-language understanding and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26417– 26427, 2024. 3
- [53] Zhangyang Qi, Yunhan Yang, Mengchen Zhang, Long Xing, Xiaoyang Wu, Tong Wu, Dahua Lin, Xihui Liu, Jiaqi Wang, and Hengshuang Zhao. Tailor3d: Customized 3d assets editing and generation with dual-side images. arXiv preprint arXiv:2407.06191, 2024. 3
- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4, 6, 7, 2
- [55] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 1, 3
- [56] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 5

- [57] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 3, 5

- [58] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024. 8, 1
- [59] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 5, 1, 2
- [60] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871– 8879, 2024. 1
- [61] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 746–760. Springer, 2012. 5
- [62] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1, 3
- [63] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 3, 5
- [64] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In The Twelfth International Conference on Learning Representations, 2023. 1, 3
- [65] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024. 3
- [66] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Alphaclip: A clip model focusing on wherever you want, 2023. 3
- [67] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812,

2024. 3

- [68] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 1, 3, 4, 5, 8, 2
- [69] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 3

- [70] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971, 2023. 3
- [71] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 1, 3
- [72] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4, 5, 6, 8, 1, 2
- [73] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023. 1, 3, 6
- [74] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Segmenting everything in context. arXiv preprint arXiv:2304.03284, 2023. 1, 3
- [75] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 3, 5
- [76] Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. arXiv preprint arXiv:1808.04560, 2018. 5, 7
- [77] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 1, 3, 5
- [78] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023. 1, 3
- [79] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 1, 3
- [80] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 1, 6, 7, 8
- [81] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 1, 5
- [82] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large

- vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024. 3
- [83] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 6
- [84] Shuai Yang, Jing Tan, Mengchen Zhang, Tong Wu, Yixuan Li, Gordon Wetzstein, Ziwei Liu, and Dahua Lin. Layerpano3d: Layered 3d panorama for hyper-immersive scene generation. arXiv preprint arXiv:2408.13252, 2024. 3
- [85] Wenhan Yang, Robby T Tan, Jiashi Feng, Jiaying Liu, Zongming Guo, and Shuicheng Yan. Deep joint rain detection and removal from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1357–1366, 2017. 7
- [86] Hanrong Ye and Dan Xu. Inverted pyramid multi-task transformer for dense scene understanding. In European Conference on Computer Vision, pages 514–530. Springer, 2022. 6
- [87] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 4

- [88] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multimodal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2(3), 2023. 3
- [89] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36, 2024. 6, 7, 1, 2
- [90] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 8
- [91] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Haodong Duan, Songyang Zhang, Shuangrui Ding, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition, 2023. 3
- [92] Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, and Xingyu Ren. Var-clip: Text-to-image generator with visual auto-regressive modeling. arXiv preprint arXiv:2408.01181, 2024. 3
- [93] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. arXiv preprint arXiv:2407.05282, 2024. 6, 7, 1, 2
- [94] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641,

2017. 5

[95] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 1

### X-Prompt: Towards Universal In-Context Image Generation in Auto-Regressive Vision Language Foundation Models

#### Supplementary Material

[Figure 130]

Figure 7. Qualitative results of text-to-image generation. High-quality text-to-image generation cases with high aesthetic qualities after training on Laion-Aesthetics [59].

###### A. Qualitative Results of Text-to-Image Generation.

We visualize some text-to-image generation results of our model in Fig. 7 and comparison with other models in Fig. 10. Fig. 7 demonstrates that our model can generate images with high aesthetic qualities after training on filtered high qaulity data from Laion-Aesthetics [59]. Figure 10 clearly demonstrates that the incorporation image dense description task has significantly bolstered the model’s proficiency in accurately following text prompt when compared to other models such as Emu3 [75] and Janus [77].

###### B. Details of training data

Full training data statistics are reported in Tab. 6 and Tab. 7. For each of the task in Tab. 6, we use QWenVL-2 [72] to describe the transformation between the input and output images and augments with reversed task.

###### C. Details of Prompt template

###### C.1. Difference description task.

For each image editing pair in Ultra-Edit [93] and MagicBrush [89], we leverage QWenVL2 [72] to describe the difference between images using the following prompt: “Describe the differeces between two images. Use ‘input image’ describe the first image and ‘output image’ to describe the second image, describe what subtask it belongs to, choosing from [Style Transfer, Object Removal, Object Replacement, Object Addition, Object Modification, Season/ Time Change, OTHER SUBTASK]”. We also ask QWenVL2 to label a reverse editing prompt for data augmentation.

###### C.2. filtering data generated by RB-Modulation.

To generate and filter high-quality data, we first use FLUX to generate high-quality and stylized images based on the prompt templates in Tab. 8. However, RB-Modulation [58] occasionally performs correct style transformations but sometimes fails. To ensure quality, we further use QWenVL2 [72] for data filtering. Due to QWen-VL2’s current

limitations in analyzing relationships among three images, we conduct quality filtering in two stages. First, we ask QWenVL-2 to verify the consistency of the main object and semantic with the base image using the question: “Do you think the two image shares the same semantics and basic layout [Yes/No]? Provide your reasoning.”. Next, we check the success of style transfer from exemplar image by asking “Do you think the two image shares the same style [Yes/No]? Provide your reasoning.” Through this process, we filter 10K high quality image based stylepersonalization image pairs to incorporate into the training of X-Prompt.

- C.3. filtering data generated by IP-Adapter.

IP-Adapter [87] can perform layout and semantic combination on two provided images, However, the final output image can maintain different attributes (layout, semantics, texture, details) from different images in a unified but not entirely deterministic format. Given the complex attributes relationship between the input images and the output images, we employ GPT-4o to analyze and annotate these relationships. As shown in Fig. 9, GPT-4o provides high-quality, detailed descriptions of the relationships between different images. For this purpose, we annotate a dataset of 50K image pairs.

C.4. Caption Rewriting on Laion-aesthetic.

We filter high-quality data from Laion-Aesthetic [59], selecting images with an aesthetic score greater than 6. For dense caption rewriting, we use QWen-VL2 [72], focusing on the relative positions, colors, and numbers of objects. To preserve caption diversity, we retain 10% of the original captions during training.

- D. Retrieval-Augmented Image Editing

Clustering similar editing pairs during training is critical to the success of Retrieval-Augmented Image Editing (RAIE) as a form of in-context learning. Fortunately, we observe that many editing instructions in MagicBrush [89] and UltraEdit [93] are highly similar to each other. As shown in Fig. 11, by pairing each editing pair with its nearest neighbor based on CLIP [54] text feature similarity, we find that many instructions are either similar or identical. This similarity is a key factor contributing to the effectiveness of RAIE.

- E. More Qualitative Results Visualization. We provide more visualization on vision tasks in Fig. 8.

###### F. Higher Resolution Reconstruction

Model Resolution PSNR SSIM SDXL-VAE (16x)

512 27.51 0.810 1024 32.13 0.922

512 26.34 0.805 1024 29.77 0.906

Chemeleon-VQVAE (16x)

Emu3-VQVAE (8x) 512 27.78 0.833

Table 5. Reconstruction quality tested on Rain-100L. Increasing resolution can greatly enhance reconstruction quality.

We use the Rain-100L derained test set to evaluate the reconstruction abilities of different models. As shown in Tab. 5, increasing the input resolution significantly enhances reconstruction quality for both VQ-VAE and VAE models. This improvement arises from the fact that image compression inherently leads to a loss of detail, and providing higher-resolution input allows the model to recover previously lost details, resulting in better outputs. However, we are unable to implement X-Prompt with a 1024 resolution as Chameleon [68] is pretrained exclusively on a 512 resolution. We anticipate significant improvements across all tasks with the availability of higher-resolution early-fusion multi-modal foundation models in the future.

## Input Output GroudTruth Input Output GroudTruth

[Figure 131]

[Figure 132]

## ScanNet-NormADE20K-SemDerainDeBlurDeNoise

[Figure 133]

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

[Figure 144]

- Figure 8. Qualitative results of diversed tasks, such as semantic segmentation, norm estimation, image deblur, denoise and derain.

DFWB GoPro Rain13k mit5k LoL Laion Aesthetic Ultra-Edit MagicBrush NYU-v2-depth ADE20K ScannNet-Norm dep/seg/norm/hed/mlsd2img

Ori data 72K 17K 13K 5K 6K 500K 500K 1.7K 48K 20K 260K 100K × 5 Augmentation 288K 68K 52K 20K 24K 1000K 2000K 6.8K 192K 80K 1040K 100K × 20

Table 6. Detailed statistics of training data with augmentation. For each pair, we use reverse task and difference description task to augment the data.

RB-Modulation IP-Adapter Viton-Try-On Pose&Action MimicBrush Ori data 10K 50K 120K 10K 50K

- Table 7. Detailed statistics of training data without augmentation.

Rendering Sketch Poster Mosaic Cyberpunk Line Drawing Impressionism Cubism Surrealism Pop Art Minimalism Abstract Expressionism Pointillism Graffiti Neo-Futurism Baroque Melting Gold Melting Silver Gold Material Made by Gold Made by Bronze Fantasy Glow Rendering Low-Poly

- Table 8. Style key words for FLUX to generate stylised images.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Checkout the relationship between these three images. If I want to compose image1 and image 2 to generate image3(right most image), what exact prompt should I prompt the image composition to do model, please refer to source features (like style, layout, color, texture, contour, silhouette, dynamic, focus design and so on) from different image using tag <image1> and <image2>. Try to be dense and descriptive.

[Figure 149]

Combine the illuminated, detailed mushroom cap and sturdy stem from left image with the delicate, pastel pink blossoms from right image. The mushrooms should be covered with clusters of flowers, maintaining the intricate lighting and mushroom contour from left image, while adopting the soft, naturalistic texture and gentle pink hues of the blossoms. The resulting image should harmonize the vertical structure and glowing cap from left image with the floral arrangement and organic color palette from right image, creating a captivating and cohesive design.

- Figure 9. An example of conversation with GPT-4o to annotate the relationship between input images and output image produced by IP-Adapter [87]

[Figure 150]

###### Figure 10. Qualitative results of text-to-image generation. Compared to Janus [77] and Emu3 [75], our model presents marked improvement in both quality and textual alignment.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Turn the leaves into colorful ballons. Transform the leaves into colorful ballons.

Replace the clock with a giant sunflower.

Turn the zebra into a unicorn.

Replace the bananas with bundles of colorful flowers. Replace the bananas with colorful flowers.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Transform the skateboard into a magic carpet. Transform the umbrella into a rainbow-colored one. Turn the ubrella into a rainbow-colored one.

Change the powdered sugar into colorful sprinkles. Add colorful sprinkles on top of the icing.

Add a crown on the dog’s head.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Transform the umbrellas into colorful hot air balloons floating in the sky. Transform the umbrellas into colorful hot air ballons.

Turn the chair into a throne and add a crown on the cat’s head.

Replace the toy rabbit with a realistic-looking owl. Replace the stuffed bear with a small owl figurine.

Turn the puppy into a panda. Turn the dog into a panda.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Add a vibrant floral crown on the dog’s head. Add a colorful floral crown on the dog’s head.

Add fairy lights around the headboard. Turn the horse drawing into a unicorn. Surround the cat with floating bubbles.

- Figure 11. Visualization of in-context training example in RAIE. After CLIP [54] based clustering, many instruction are similar or completely the same, which is crucial to the success of RAIE.

