## SuperEdit: RectifyingandFacilitatingSupervisionforInstruction-BasedImageEditing

[Figure 1]

##### Website Code Data

Ming Li1,2,‡, Xin Gu1, Fan Chen1, Xiaoying Xing1, Longyin Wen1, Chen Chen2, Sijie Zhu1,∗

1ByteDance Intelligent Creation (USA) 2Center for Research in Computer Vision, University of Central Florida

Comparison with Existing Methods

[Figure 2]

[Figure 3]

| |SuperEdit<br><br>1.1B<br><br>1.1B<br><br>HQ-Edit<br><br>1.1B<br><br>1.1B<br><br>HIVE<br><br>1.1B<br><br>14.1B<br><br>SmartEdit<br><br>InstructDiffusion<br><br>MagicBrush<br><br>InstructPix2Pix<br><br>1.1B<br><br>ü 30×Less Training Data ü 13×Less Model Size ü 9.19% Improvements| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Real-EditBenchmarkOverallScore

# arXiv:2505.02370v1[cs.CV]5May2025

Background: Change the background to a snowy mountain landscape

Add: Add a pink tie for the boy

[Figure 4]

[Figure 5]

[Figure 6]

Training Editing Data Size (K)

- (b) Comparison with GPT-4o Evaluation

- (c) Comparison with Human Evaluation

Global: Transform the scene into a snowy winter scene

Replace: Replace the bow tie with a butterfly.

[Figure 7]

[Figure 8]

SuperEdit vs. SmartEdit

|SuperEdit vs. InstructPix2Pix| | | | |
|---|---|---|---|---|
| | | | | |

###### Human Evaluation Win Rate For Overall Score

Style: Convert the image to a watercolor style

Remove: Remove some clouds in the sky

(a) Examples of Our Method on Real & High-resolution Image

Figure 1. (a) Our editing method works well with real and high-resolution images, handling various free-form edits (left) and local edits (right); (b) Compared to the current state-of-the-art SmartEdit, our method achieves a 9.19% performance improvement with 30× less training data and 13× fewer model parameters; (c) Our method achieves better overall scores on the human evaluation results, indicating more precise editing capabilities.

##### Abstract

To this end, we further construct contrastive supervision signals with positive and negative instructions and introduce them into the model training using triplet loss, thereby further facilitating supervision effectiveness. Our method does not require the VLM modules or pre-training tasks used in previous work, offering a more direct and efficient way to provide better supervision signals, and providing a novel, simple, and effective solution for instruction-based image editing. Results on multiple benchmarks demonstrate that our method significantly outperforms existing approaches. Compared with previous SOTA SmartEdit, we achieve 9.19% improvements on the Real-Edit benchmark with 30× less training data and 13× smaller model size. All data and models are open-sourced on Github for future research.

Due to the challenges of manually collecting accurate editing data, existing datasets are typically constructed using various automated methods, leading to noisy supervision signals caused bythemismatchbetweeneditinginstructionsandoriginal-edited image pairs. Recent efforts attempt to improve editing models through generating higher-quality edited images, pre-training on recognition tasks, or introducing vision-language models (VLMs) but fail to resolve this fundamental issue. In this paper, we offer a novel solution by constructing more effective editing instructions for given image pairs. This includes rectifying the editing instructions to better align with the original-edited image pairs and using contrastive editing instructions to further enhance their effectiveness. Specifically, we find that editing models exhibit specific generation attributes at different inference steps, independent of the text. Based on these prior attributes, we define a unified guide for VLMs to rectify editing instructions. However, there are some challenging editing scenarios that cannot be resolved solely with rectified instructions.

##### 1. Introduction

In recent years, significant progress has been made in text-to-image (T2I) generation [10, 37, 40–42] due to the development of diffusion models [9, 21, 48, 49]. These T2I diffusion models can generate images that align with natural language descriptions while satisfying human perception and preferences. Consequently, numerous image editing

∗ Corresponding author, sijiezhu@bytedance.com ‡ This work was done during the internship at ByteDance, San Jose, USA

Loss

Loss

methods [4, 6, 19, 31] based on these models have been proposed to achieve various editing effects. Instruction-based methods [4, 12, 24] have become increasingly popular as they allow users to conveniently and easily modify images using language instructions without the need to provide masks, as required by mask-based methods [18, 28, 47, 53].

[Figure 9]

🔥

[Figure 10]

🔥

|T2I Pre-trained U-Net|
|---|

|T2I Pre-trained U-Net|
|---|

| | |
|---|---|
|VLM| |

| | |
|---|---|
|VAE| |

| | |
|---|---|
|CLIP| |

| | |
|---|---|
|VAE| |

[Figure 11]

[Figure 12]

[Figure 13]

🔥

❄ ❄

[Figure 14]

❄

misaligned

misaligned

Noisy Instruction

Images

Images

Noisy Instruction

(b) SmartEdit; MGIE

(a) InstructPix2Pix; HQ-Edit

The training of instruction-based editing models requires the original-edited image pairs and corresponding editing instruction, making it difficult to manually create or collect a large amount of relevant data [58]. To address the issue of scarce training data, existing efforts [13, 19, 25, 62] have attempted to develop various automated pipelines to synthesize large datasets. Specifically, most methods first use large language models (LLMs) to modify the text descriptions of original images. The original images and modified texts are then input into various pre-trained diffusion models to automatically generate edited images. However, current text-to-image diffusion models struggle to fully correspond to input text prompts [15, 59]. This often changes parts of the original images that do not require editing, leading to misaligned editing instructions and original-edited image pairs, thus resulting in noisy supervision signals. To mitigate the potential issues of noisy supervision in image editing models, existing work has attempted to introduce additional recognition pre-training tasks for U-Net [43] such as semantic segmentation [14, 45], or replace CLIP [38] text encoder with vision-language models (VLMs) [12, 24] to better understand editing instructions from noisy supervision signals. However, these methods not only introduce significant computational overhead but also overlook the issue of noisy supervision signals.

Loss Loss

[Figure 15]

🔥

[Figure 16]

🔥

|Extra Pre-trained U-Net|
|---|

|T2I Pre-trained U-Net|
|---|

| | |
|---|---|
|CLIP| |

| | |
|---|---|
|VAE| |

| | |
|---|---|
|CLIP| |

[Figure 17]

[Figure 18]

❄ ❄

| | |
|---|---|
|VAE| |

[Figure 19]

[Figure 20]

❄ ❄

misaligned aligned

Images

Images

Noisy Instruction

Rectiﬁed Instruction

###### (d) Ours

(c) InstructDiﬀusion; Emu-Edit

Figure 2. Unlike existing efforts that attempt to (a) scale up edited images with noisy supervision [4, 25], (b) introduce massive VLMs into editing model architecture [12, 24], and (c) perform additional pre-training tasks [14, 45], (d) we focus on improving the effectiveness of supervision signals, which is the fundamental issue of image editing.

stages correspond to the generation of different image attributes, independent of the input text prompt [2, 17, 19, 36, 46, 56, 61] or editing instructions. Inspired by this, we guide VLMs based on these attributes to establish a unified rectification guideline for various editing instructions, as demonstrated in Fig. 3.

When training with only rectified editing instructions, we find that the editing model can better understand the editing commands but still faces challenges in handling complex scenarios. For example, when the original image contains multiple objects, the edit model struggles to perform an accurate editing function if the instructions modify only one of these objects. Additionally, inherent issues present in pre-trained text-to-image diffusion models [15, 22–24], such as difficulty in understanding quantity, position, or object relationships, persist in the editing models. To address these issues, we propose using contrastive supervision signals to further optimize the editing model. Specifically, we first construct incorrect editing instructions based on the rectified instructions to generate positive and negative samples. We then introduce a triplet loss to guide the model, thereby enhancing the effectiveness of supervision, as shown in Fig. 5.

In this paper, we focus on addressing the fundamental challenge by introducing more effective editing instructions, as demonstrated in Fig. 2. Our data-oriented method explores a different research question: how much performance improvement can be achieved solely by focusing on supervision signal quality and optimization in image editing? Surprisingly, SuperEdit outperforms existing methods in both GPT-4o and human evaluations, despite using less data and requiring no additional modules or pretraining as shown in Fig.1. This demonstrates that high-quality supervision signals can significantly compensate for architectural simplicity, achieving results comparable to or better than methods with more complex requirements.

In summary, our contributions are summarized as follows:

- • New Insight: We aim to address the noisy supervision problem that arises from the misalignment between editing instructions and original-edited image pairs, which is a fundamental issue overlooked by previous work, as shown in Fig. 2.

- • Rectifying Supervision: We leverage diffusion generation priors to guide the vision-language model to generate betteraligned editing instructions for original-edited image pairs.

- • Facilitating Supervision: We introduce contrastive supervision using triplet loss, enabling the editing model to learn from both positive and negative editing instructions.

- • Promising Results: We achieve significant improvements on multiple benchmarks without additional pre-training or VLM. Compared to SmartEdit [24], we achieved a 9.19% improvement while reducing 30× data and 13× model parameters.

Specifically, to enhance the effectiveness of supervision signals for instruction-based image editing methods, we propose using VLMs to rectify editing instructions, creating betteralignedinstructionsfortheoriginal-editedimagepairs. However, determining which VLM to use for this task and how to establish a unified rectification method for various editing instructions remain unexplored problems. To address this, we first analyze the ability of different VLMs to understand the differences between original and edited images, showing that GPT-4o [1] is the most capable of rectifying editing instructions. Additionally, we observe that both editing models and text-to-image diffusion models share a similar prior, as shown in Fig. 4: different inference

##### 2. Related Work

###### 2.1. Image Editing with Diffusion Models

Building on advancements in text-to-image (T2I) diffusion models [10, 37, 40–42, 44], recent research has explored them for image editing [4, 19]. Training-free methods [5, 19, 29, 31, 34, 51] typically achieve this by adjusting attentions in pre-trained T2I models, but have limited performance and generalization capabilities on various editing tasks.

Training-based methods address these limitations with specialized editing models, which can be categorized into mask-based and instruction-based approaches. Mask-based methods [6, 18, 28, 47, 53, 57] enable fine-grained local edits with user-provided or predicted masks and corresponding text descriptions. However, it struggles with global image editing and is constrained by the lack of mask-based editing data [24].

Instruction-based methods directly accept textual commands, such as “add a dog”, offering better editing flexibility and generalization. InstructPix2Pix [4] pioneered this paradigm by generating instruction-based editing data and modifying the conditions of T2I diffusion models. Building on this framework, subsequent work introduces vision-language models [12, 24, 33] or additional pre-training tasks for the denoising U-Net [14, 24, 43, 45] to enhance the understanding and reasoning of input conditions. However, these methods not only introduce substantial computational overhead but also overlook the fundamental noisy supervision issue.

###### 2.2. Generating and Improving Editing Supervision

Due to the difficulty of scaling instruction-based image editing data through manual collection, existing efforts [4, 13, 25, 58, 62] aim to automatically modify text descriptions of original images and generate edited images with T2I diffusion models. However, this approach often produces synthesized images that do not align with the editing instructions, as shown in Figure 3, resulting in noisy editing supervision signals [58, 62]. To address this, MagicBrush [58] manually filters out incorrect editing data, but it is hard to scale. Unlike existing methods focusing on edited image quality, we leverage diffusion prior and vision-language model (i.e., GPT4o [1]) to create better-aligned instructions with original-edited image pairs, providing more accurate supervision.

###### 2.3. Alignment of Diffusion Models

The success of alignment training in large language models (LLMs) [26, 32, 39] has been applied to diffusion models for better image generation. This is achieved by maximizing reward scores [8, 27, 54] or the generation probability of the winner image in a pair [11, 52, 55]. In image editing, HIVE [60] and MultiReward [16] attempt to incorporate reward information into the text condition to align the editing model. In contrast, we guide the editing model by rectifying and constructing contrastive editing instructions, achieving more effective alignment.

##### 3. Method

In this section, we first introduce the most general image editing framework in Sec. 3.1. Then, we explain how to use diffusion priors to rectify editing instructions with the multimodal model (i.e., GPT-4o) in Sec. 3.2, thereby enhancing the accuracy of supervision signals. Finally, we describe how to construct contrastive supervision with both correct and incorrect editing instructions and integrate it into the editing model training using triplet loss in Sec. 3.3.

###### 3.1. Instruction-based Image Editing Framework

InstructPix2Pix [4] pioneered instruction-based image editing, performing editing tasks by simultaneously taking the original image CI and editing instructions CT as input conditions to generate the edited image x from random noise ϵ. Following the definition of DDPM [21], we randomly sample a timestep t∈T during training, and then add corresponding noise ϵt to the edited image x:

xt=√α¯tx+√1−α¯tϵt, ϵ∼N(0,I), (1) where ϵ is a noise map sampled from a Gaussian distribution,

and α¯t := ts=0αs, αt =1−βt is a differentiable function of timestep t, which is determined by the denoising sampler such

as DDPM [21]. Then the training objective of the editing model ϵθ is predicting the added noise at timestep t, which can be written as:

t,t,CI,CT,ϵ ϵθ concat(xt,CI),t,CT −ϵt 22 , (2)

Ltrain =Ex

where concat refers to concatenating the image latents of noised edited image xt and original image cI in the channel dimension.

###### 3.2. Rectifying Supervision with Diffusion Priors

As shown in Fig. 3, existing image editing datasets [4, 13, 58] typically use only Steps 1 and 2: LLMs construct editing prompts and captions, and then text-to-image diffusion models synthesize edited images. However, diffusion models often fail to accurately follow prompts while maintaining image layout, creating mismatches between original-edited pairs and editing instructions, resulting in inaccurate supervision. While better supervision signals for text-to-image diffusion models are common in image generation [3, 50], this approach remains underexplored in image editing due to two challenges: (1) VLMs trained on single-image data struggle with multi-image inputs, and (2) editing instructions vary widely, making unified rectification guidelines difficult. To address these issues, we: (1) analyzed different VLMs’ capabilities with multi-image inputs, finding GPT-4o most effective, and (2) discovered that timestep-specific roles in image generation also apply to editing, providing a foundation for a unified rectification method across various instructions (Fig. 3 and 4). Due to page limitations, our VLM analysis is in the Supplementary Material, while this section focuses on Diffusion Prior and Editing Instruction Rectification.

###### Step 3: Rectify Editing Instruction

###### Step 1: Generate Editing Instruction

###### Summarized Instruction

Diffusion Prior

||[Figure 21]|[Figure 22]|
|---|---|
|
|---|

Editing Instruction: “Make the pagoda a lighthouse” New Caption: “Chinese lighthouse HD wallpaper”

[Figure 23]

Global, Layout Local Object Style, Details

Input Caption: “Chinese Pagoda HD wallpaper”

Replace the pagoda and its surrounding garden with a lighthouse amidst rocks and the ocean. Swap the textures of trees and the pagoda with those of rocks, bricks, and water, and change the multi-tiered pagoda shape into a cylindrical lighthouse shape. Alter the color palette to contrasting cool tones and adjust the overall mood to be more serene and less dense.

LLM

###### VLM

###### Step 2: Generate Edited Image

Rectified Instruction

Only the pagoda region should be edited.

Background & layout are incorrectly changed.

Global, Layout: Replace the pagoda and surrounding garden elements with a lighthouse and surrounding rocks/ocean. Local Object: Swap the textures of trees and pagoda with rocks, bricks, and water. Replace the multi-tiered pagoda shape with a cylindrical lighthouse shape. Style, Details: Change the color palette to contrasting cool tones. Adjust the mood to be more serene and less dense.

New Caption

|[Figure 24]| |
|---|---|
| | |

|[Figure 25]|
|---|

Make summarized instruction less than 77 tokens for CLIP text encoder

Original Image Diffusion Models Edited Image

(a) Existing Data Generation Pipeline (b) Our Rectification Pipeline

- Figure 3. (a) Existing work primarily uses LLMs and diffusion models to automatically generate edited images. However, current diffusion models often fail to accurately follow text prompts while maintaining the input image’s layout, resulting in mismatches between the original-edited image pairs and the editing instructions. (b) We perform instruction rectification (Step 3) based on the images constructed in Steps 1 and 2. We show VLMs can understand the differences between the images, enabling them to rectify editing instructions to be better aligned with original-edited image pairs.

[Figure 26]

|[Figure 27]|
|---|

|[Figure 28]|
|---|

Global Change

Input Image

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

Editing Instruction 1: Change the background to the cloudy sky

Editing Instruction 2: Turn him into a

Editing Instruction 3: Transform into watercolor style

10 30 30 0 30

|[Figure 34]|
|---|

Early Stage Mid Stage Late Stage

10 20

Detail Change

[Figure 35]

Input Image

0 0 20

Local Change Detail Change

Style Change

InstructPix2Pix DDIM 30-step

[Figure 36]

Input Image

[Figure 37]

Local Change

Global Change

- Figure 4. We show that the editing model follows consistent generation attributesatdifferentsamplingstages, independentoftheeditinginstructions. The early, middle, and late sampling stages correspond to global, local, and detail changes, respectively, while style changes occur at all stages. All the generated images here are DDIM 30-step sampled final images. The orange progress bar and the grid progress bar represent the sampling stages with and without the editing instructions, respectively.

in the late stages of sampling. This finding inspires us to guide VLMs based on these four generation attributes, establishing a unified rectification method for various editing instructions. We provide more analysis and details in the Supplementary Material.

Editing Instruction Rectification. As demonstrated in Fig. 3, we extend the existing editing data generation pipeline by introducing our instruction rectification (Step 3). This process relies on the original edited image pairs obtained through Steps 1 and 2 from previous work. Specifically, we input original-edited image pairs into the vision-language model (i.e., GPT-4o) and instruct it to describe the changes in the edited image compared to the original image according to the above diffusion prior generation attributes. Finally, we use VLM to summarize the instructions and ensure that its length is less than the maximum length of CLIP text encoder, which is 77 tokens.

###### 3.3. Facilitating Supervision with Contrastive Instructions

Although using rectified editing instructions can significantly improve performance across various editing tasks, we find that editing models still struggle with closely related text instructions. For example, “add a cat on the left side of the image” and “add two cats on the right side of the image” might produce the same edited image. This indicates that inherent biases in pre-trained text-to-image diffusion models [15, 22], such as difficulties in understanding quantity, position, and spatial relationships, persist in editing models. More importantly, our experiments show that training models with rectified editing instructions does not resolve these challenges. To further facilitate supervision signal effectiveness, we drew on successful alignment experiences from large language models [1, 32, 39] and text-to-image diffusion models [7, 52, 54]: constructing positive and negative sample pairs and guiding the model to assign a higher generation probability to positive samples compared to negative ones.

Diffusion Generation Priors. Previous work has shown that different timesteps play distinct roles in image generation for text-to-image diffusion models, regardless of the text prompt [2, 17, 19, 36, 46, 56, 61]. We find that this phenomenon also exists in instruction-based editing models and present examples based on pre-trained InstructPix2Pix [4], as shown in Fig. 4. Specifically, diffusion models focus on global layout in the early stages, local object attributes in the mid stages, and image details

Change object color

|Rectified Instruction<br><br>Original Image<br><br>[Figure 38]<br><br>Edited Image<br><br>| |
|---|
<br><br>| |
|---|
<br><br>[Figure 39]<br><br>Add a pair of black sunglasses for the dog|
|---|

|[Figure 40]<br><br>Add a pair of black sunglasses on the ground<br><br>[Figure 41]<br><br>Add a pair of purple sunglasses for the dog<br><br>[Figure 42]<br><br>Add a pair of black earmuffs for the dog<br><br>Wrong Instructions<br><br>[Figure 43]<br><br>············|
|---|

[Figure 44]

[Figure 45]

|[Figure 46]|
|---|
|[Figure 47]|

Noise

[Figure 48]

[Figure 49]

[Figure 50]

###### Denoising Model

[Figure 51]

⊕

### ©

[Figure 52]

Edited Image

VLM

Attract Repel

············

[Figure 53]

Change object position

!

[Figure 54]

VAE Encoder

[Figure 55]

|[Figure 56]|
|---|

[Figure 57]

[Figure 58]

CLIP Text Encoder

VLM

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Change object category

Noise

Original Image

[Figure 65]

Wrong Instruction

Rectified Instruction

VAE Encoder

⊕Add ©Concat Freeze 🔥Trainable

[Figure 66]

[Figure 67]

VLM

(a) Contrastive Instruction Construction (b) Training Pipeline with Contrastive Instruction

- Figure 5. (a) Based on the rectified editing instruction and original-edited image pair, we utilize the Vision-Language Models (VLM) to generate various image-related wrong instructions. These involve random substitutions of quantities, spatial locations, and objects within the rectified editing instructions according to the original-edited images context; (b) During each training iteration, we randomly select one wrong instruction cTneg and input it along with the rectified instruction cTpos into the editing model to obtain predicted noises. The goal is to make the rectified instruction’s predicted noise ϵpos closer to the sampled training diffusion noise ϵ, while ensuring the noise from incorrect instructions ϵneg is farther. Best viewed in color.

Constructing Contrastive Instructions. Unlike the standard alignment process for large language models or text-to-image diffusion models, it is challenging to generate different editing results from the same instruction to create positive and negative sample pairs for image editing tasks. To address this, we construct positive and negative editing instructions for alignment, thereby generating relatively positive and negative edited images. As shown in Fig. 5 (a), we use the original image, edited image, and rectified editing instruction as input. The VLM (i.e., GPT-4o) is used to modify attributes in the rectified editing instruction, such as quantity, spatial relationships, and object types, to create different wrong instructions. Here, we require VLM to modify only a single attribute from the rectified editing instruction in each wrong instruction, keeping most of the editing text unchanged. Since only a few words are replaced between the rectified instruction and the wrong instruction, the text embeddings produced by the CLIP text encoder that serve as input to the denoising model will also be similar. This ensures the task’s learning difficulty, helping the model understand how subtle differences between the two editing instructions result in significantly different editing results.

construct positive and negative samples, respectively:

ϵpos=ϵθ concat xt,cI ,t,cTpos , (3) ϵneg=ϵθ concat xt,cI ,t,cTneg . (4)

After constructing the positive and negative sample pairs, we aim for the noise predicted by the positive editing instruction ϵpos to be closer to the true noise ϵt sampled during training, compared to the noise predicted by the wrong editing instruction ϵneg. This goal can be achieved through a triplet loss function:

Ltriplet=max{d(ϵt,ϵpos)−d(ϵt,ϵneg)+m,0}, (5)

where d(x,y)=∥x−y∥22 and margin m is a hyper-parameter. The final training loss is the combination of the original

diffusion training loss and the triplet loss:

Ltotal=Ltrain+λ·Ltriplet, where Ltrain=d(ϵt,ϵpos). (6)

Please note that the contrastive supervision signals are only used during the training phase. During inference, the editing model only requires one single input editing instruction.

##### 4. Experiment

Facilitating Editing Models with Contrastive Instructions. Our key insight is that enhancing the effectiveness of supervision signals can improve various editing tasks without introducing additional model architectures or pre-training tasks. Therefore, we adhere strictly to the InstructPix2Pix [4] model architecture and training pipeline. To be specific, the inputs including the original image cI, edited image x, the rectified instruction cTpos, and wrong editing instruction cTneg. During training, we will add a sampled timestep t ∈ T to obtain the noised edited image xt with Equation 1. Both the rectified and wrong editing instructions are fed into the denoising model to predict the final noises ϵpos and ϵneg, which are then used to

###### 4.1. Data Collection and Construction

To build a diverse dataset with various types of editing instructions, we need original and edited images from different data domains, as well as a wide variety of editing instructions. To achieve this, we sampled data from different public editing datasets to construct rectified and contrastive supervision signals. Specifically, we extracted 10,177, 8,807, and 21,016 editing pairs from InstructPix2Pix [4], MagicBrush [58], and Seed-Data-Edit [13], respectively, resulting in a total of 40,000 training samples. During extraction, we strive to ensure that the data for different types of editing tasks is as balanced as possible.

Replace the tiger with a lion, maintaining the same position in the water.

Change the background to a sandy beach with the ocean in the distance.

###### Editing Instruction

Transform the global scene to a winter setting with snow covering the houses, trees, and boat.

Change the image style to look like an impressionist painting style.

Add a pink tie for the boy. Delete the big rock in the lower left corner.

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

Original Image

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

HIVE

Scores: 4.0, 4.8, 2.5 Scores: 4.0, 4.5, 4.0 Scores: 4.8, 4.0, 4.0 Scores: 0.0, 4.5, 4.8 Scores: 4.8, 4.0, 3.5 Scores: 5.0, 3.5, 4.0

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

HQ-Edit

Scores: 4.0, 0.0, 3.0 Scores: 5.0, 0.0, 3.5 Scores: 4.0, 0.0, 3.0 Scores: 0.0, 4.5, 4.8 Scores: 4.8, 4.0, 3.0 Scores: 5.0, 1.0, 3.0

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

MGIE

Scores: 4.0, 2.8, 3.5 Scores: 4.5, 4.0, 4.0 Scores: 4.5, 4.5, 4.5 Scores: 0.0, 4.8, 4.8 Scores: 4.8, 4.5, 4.5 Scores: 4.8, 1.0, 4.0

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

KOSMOS-G

Scores: 3.0, 1.0, 3.0 Scores: 4.0, 0.0, 3.0 Scores: 3.5, 1.0, 2.0 Scores: 0.0, 3.0, 4.0 Scores: 4.0, 2.0, 3.5 Scores: 4.0, 1.0, 2.0

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

Instruct Diffusion

Scores: 2.0, 4.8, 2.5 Scores: 5.0, 4.8, 4.0 Scores: 4.5, 3.0, 4.0 Scores: 0.0, 4.5, 4.8 Scores: 4.8, 3.0, 3.5 Scores: 0.0, 4.5, 4.8

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

InstructP2P

Scores: 2.0, 4.5, 2.0 Scores: 3.5, 2.0, 3.0 Scores: 5.0, 4.0, 4.5 Scores: 0.0, 4.8, 4.8 Scores: 4.8, 4.0, 4.5 Scores: 2.0, 4.5, 4.5

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

###### MagicBrush

Scores: 2.0, 3.0, 2.0 Scores: 2.0, 4.0, 3.0 Scores: 5.0, 4.0, 4.0 Scores: 5.0, 4.5, 4.8 Scores: 3.0, 4.8, 4.8 Scores: 4.0, 3.0, 4.5

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

###### SmartEdit

Scores: 4.8, 4.8, 2.5 Scores: 5.0, 4.8, 4.0 Scores: 3.5, 4.0, 3.0 Scores: 0.0, 5.0, 4.8 Scores: 1.0, 4.8, 4.8 Scores: 2.0, 4.5, 4.5

| |
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

[Figure 127]

###### SuperEdit (Ours)

Scores: 4.8, 4.8, 4.8 Scores: 4.8, 5.0, 5.0 Scores: 5.0, 5.0, 4.8 Scores: 5.0, 5.0, 4.8 Scores: 4.8, 4.8, 4.8 Scores: 5.0, 4.8, 4.8

- Figure 6. Visual comparison with existing methods and the corresponding human-aligned GPT-4o evaluation scores (Following, Preserving, Quality Scores from left to right). We achieve better results while preserving the layout, quality, and details of the original image. Please note that we do not claim that our editing results are flawless. We provide more visual comparison results in the supplementary material.

|Method<br><br>|Extra Module<br><br>Pretrain Tasks<br><br>Edit Data<br><br>Model Size<br><br>|Following ↑ Preserving ↑ Quality ↑ Overall ↑ Acc Score Acc Score Acc Score Acc Score<br><br>|
|---|---|---|
|KOSMOS-G [33] MGIE [12] SmartEdit [24] MultiReward [16] InstructDiffusion [14] InstructPix2Pix [4] MagicBrush [58] HIVE [60] HQ-Edit [25] SuperEdit (Ours)|✓ ✓ 9.0M 1.9B ✓ ✓ 1.0M 8.1B ✓ ✓ 1.2M 14.1B<br><br>✓ ✓ 320K 1.2B<br><br>✗ ✓ 860K 1.1B<br><br>✗ ✗ 300K 1.1B<br><br>✗ ✗ 310K 1.1B<br><br>✗ ✗ 1.1M 1.1B<br><br>✗ ✗ 500K 1.1B<br><br>✗ ✗ 40K 1.1B<br>|51% 2.82 9% 1.43 27% 3.20 29.0% 2.48<br><br>40% 2.43 45% 2.79 38% 3.35 41.0% 2.86 64% 3.50 66% 3.70 45% 3.56 58.3% 3.59 63% 3.39 58% 3.43 54% 3.80 58.3% 3.54<br><br>52% 2.87 54% 3.17 45% 3.58 50.3% 3.21<br><br><br>52% 2.94 53% 3.31 50% 3.69 51.7% 3.31 51% 2.90 70% 3.85 50% 3.67 57.0% 3.47 54% 2.93 56% 3.36 53% 3.72 54.3% 3.34 51% 2.84 16% 1.63 54% 3.84 40.3% 2.77 67% 3.59 77% 4.14 65% 4.01 69.7% 3.91<br><br>|

- Table 1. Comparison with instruction-based image editing methods on Real-Edit benchmark [16]. Compared to existing work, our method achieves state-of-the-art performance across all metrics using a small amount of high-quality editing data without introducing additional models or pre-training tasks. Please note that the scores range from 0 to 5. ↑ denotes a higher result is better. All baseline results are cited from the MultiReward [16] paper.

|SuperEdit vs InstructPix2Pix| | | | |
|---|---|---|---|---|
| | | | | |

|SuperEdit vs InstructPix2Pix| | | | |
|---|---|---|---|---|
| | | | | |

|SuperEdit vs InstructPix2Pix| | | | |
|---|---|---|---|---|
| | | | | |

SuperEdit vs SmartEdit SuperEdit vs SmartEdit SuperEdit vs SmartEdit

(a) Following Score (b) Preserving Score (c) Quality Score

Figure 7. Human evaluation on three evaluation criteria for image editing effects. (a) Following: whether the edited image adhere to the editing instructions; (b) Preserving: whether the image structure outside of the editing instructions has been preserved; (c) Quality: whether the overall quality/aesthetics of the edited image has been degraded compared to the input image. Our SuperEdit achieves the best results on all of these metrics.

Following ↑ Preserving ↑ Quality ↑ Overall ↑

InstructPix2Pix [4] 2.41 2.62 2.44 2.49 SmartEdit-13B [24] 3.09 3.06 2.63 2.93 SuperEdit 3.18+1.80% 3.86+16.00% 3.37+14.80% 3.47+10.80%

- Table 2. Human evaluation results on Real-Edit [16] benchmark. All the human-evaluated scores range from 0 to 5. Overall represents the average score of Following, Preserving, and Quality scores.

###### 4.3. Experimental Results

Comparison on Real-Edit Benchmark. In Tab. 1, we present the quantitative results of editing effectiveness on the Real-Edit benchmark [16]. Without introducing additional parameters or pre-training stages, our method achieves the best results in the three GPT-4o automated evaluation metrics: Following, Preserving, and Quality, each of which includes percentage accuracy (Acc) and scores (from 0 to 5). For example, compared to SmartEdit [24], which introduces an additional 13B vision-language model (i.e., LLaVA [30]) to the 1.1B InstructPix2Pix [4] framework, we achieved improvements of 11.4% Overall Score. This suggests that given accurate and effective supervision signals, the trained editing model can understand and successfully execute the editing instructions, without the need for additional vision-language models.

We then applied our proposed methods in Sec. 3 to rectify and construct contrastive editing instructions for these training samples. Since the MagicBrush data has been manually verified, we skip the rectification step for this dataset and directly construct contrastive supervision based on the original editing instructions. For Seed-Data-Edit dataset, we only sample images from the first part of data without human editing instructions.

###### 4.2. Experimental Settings

It is worth noting that unlike existing image editing methods, which often show improvement in a single metric while others remain unchanged or worsen, our method achieves comprehensive and significant advancements across all three metrics. This indicates that improving the effectiveness of supervision signals can accurately execute editing instructions while reducing disruption to other non-edited parts of the image, and preserving the quality and aesthetics of the original images. Specifically, we surpassed the current best methods by 3%, 7%, and 11% Acc results in Following, Preserving, and Quality, respectively.

Evaluation Benchmarks and Metrics. To more accurately assess the effectiveness of various editing models, we conducted assessments on the Real-Edit benchmark [16], which is a humanaligned evaluation benchmark with GPT-4o scoring. Specifically, MultiReward [16] uses high-resolution images from the Unsplash community as a test dataset and combines them with GPT-4o [1] to create an automated evaluation method for singleturn editing. It assesses edited images in terms of accuracy (%) and scores (from 0 to 5), evaluating whether they adhere to the editing instructions (Following), whether the image structure outside of the editing instructions has been preserved (Preserving), and whether the overall quality/aesthetics of the edited image has been degraded compared to the original one (Quality).

Human Evaluation We also conduct a comprehensive human evaluation on Real-Edit benchmarks [16] in Tab. 2 and Fig. 7. The assessment involved 15 experienced evaluators who

rated edited images based on three critical metrics: instruction faithfulness (Following), preservation of irrelevant content (Preserving), and visual quality (Quality). The results of this manual evaluation demonstrate strong consistency with the GPT-4o scoring results shown in Tab. 1. This high alignment thoroughly validates that our proposed SuperEdit significantly outperforms existing methods across all evaluation criteria. Specifically, our SuperEdit surpasses the previous state-of-the-art method SmartEdit [24] by 1.8%, 16%, 14.8%, and 10.8% on Following, Preserving, Quality, and Overall scores, respectively. These substantial improvements not only confirm the effectiveness of our approach but also establish SuperEdit as a new benchmark in instruction-guided image editing, achieving superior performance while requiring significantly less training data and cost. Visual Comparison with State-of-the-art Methods. We show the visual comparison with existing image editing methods in Fig. 6. Compared to existing instruction-based editing methods, our approach not only better understands and executes editing instructions but also preserves the original image’s layout andqualitymoreeffectively, therebysignificantlyoutperforming previous methods. For example, with the instruction “Replace the tiger with a lion, maintaining the same position in the water” our SuperEdit method achieved superior results (4.8/4.8/4.8) compared to SmartEdit (4.8/4.8/2.5) and other methods. Additionally, our method improves the model’s comprehension of editing instructions. For the instruction “Change the background to a sandy beach with the ocean in the distance” our method received perfect scores (4.8/5.0/5.0) while SmartEdit only achieved(5.0/4.8/4.0). Similarly, forstyletransformationinstructions like “Change the image style to look like an impressionist painting style” SuperEdit significantly outperformed SmartEdit with scores of (4.8/4.8/4.8) versus (1.0/4.8/4.8), demonstrating our method’s superior ability to handle complex artistic transformations. Even more impressively, for scene transformation tasks like “Transform the entire scene to a winter setting with snow covering the houses, trees, and boat”, our SuperEdit achieved (5.0/4.8/4.8) while SmartEdit only obtained (2.0/4.5/4.5). We provide more visual comparisons with other instruction-based image editing methods in the Supplementary Material.

###### 4.4. Ablation Study

Ablation on the Rectified and Contrastive Instructions. Considering that the Real-Edit [16] benchmark is evaluated by GPT-4o [1], and its evaluation results closely align with human ratings [16], we choose this benchmark to conduct ablation experiments in Tab. 3. Compared to the original 300K InstructPix2Pix training data, our 40K training data with rectified editing instructions significantly improves all the performance of the editing model. Specifically, our approach improves scores by 0.95, 0.79, and 0.11, and accuracy by 21%, 22%, and 4% in these three metrics, respectively. In addition, editing performance is further enhanced by incorporating contrastive supervision signals. Compared to using only rectified editing

|Rectified Instruction<br><br>Contrastive Instructions<br><br>|Following↑ Preserving↑ Quality↑ Acc Score Acc Score Acc Score|
|---|---|
|✗ ✗ ✓ ✗ ✓ ✓<br><br>|41% 2.45 53% 3.27 61% 3.90 62% 3.40 75% 4.06 65% 4.01 67% 3.59 77% 4.14 65% 4.01|

Table 3. Ablation study on our methods. Both rectified and contrastive editing instructions achieved improvements across all metrics.

instructions, the introduction of contrastive supervision signals improves the following and preserving scores by 0.19 and 0.08, and accuracy by 5% and 2%, while maintaining the quality accuracy and score. In summary, both the introduction of rectified editing instructions and contrastive editing instructions improve the overall performance of the editing model.

Ablation on Data Scaling. We investigated the impact of training data volume on model performance by experimenting with datasets ranging from 5k to 40k samples. Tab. 4 shows consistent improvements across all metrics as training data increases. With just 5k samples, our model achieves reasonable performance (54.7% accuracy, 3.42 overall score), but scaling to 40k samples yields substantial gains (69.7% accuracy, 3.91 overall score). The most significant improvements appear in the Preserving and Quality metrics, with 10% and 15%, respectively. This upward trend across all data points demonstrates that SuperEdit effectively leverages additional training examples without performance saturation, suggesting potential for further gains with larger datasets.

|Data Size|Following ↑ Preserving ↑ Quality ↑ Overall ↑<br><br>Acc Score Acc Score Acc Score Acc Score<br><br>|
|---|---|
|5k 10k 20k 40k|49% 2.87 60% 3.71 55% 3.69 54.7% 3.42 57% 3.26 71% 3.76 58% 3.87 62.0% 3.63 64% 3.40 72% 4.02 63% 3.94 66.3% 3.79 67% 3.59 77% 4.14 65% 4.01 69.7% 3.91<br><br>|

Table 4. Ablation study on data scaling results on Real-Edit [16].

##### 5. Conclusion

In this paper, we re-examine image editing models from the perspective of enhancing supervision signals, finding that existing models have not adequately addressed this challenge, resulting in suboptimal performance. We introduce a unified editing instructionrectificationguidelinebasedondiffusionpriorsthatbetter aligns instructions with original-edited image pairs, thereby enhancing supervision effectiveness. We also construct contrastive editing instructions allowing models to learn from both positive and negative examples. Our data-oriented approach explores an important but overlooked research question: What level of performance can be achieved with minimal architectural modifications by primarily focusing on supervision quality and optimization? Remarkably, under both GPT-4o and human evaluation, our method outperforms existing approaches despite using less data and requiring no architectural modifications or additional pretraining. This shows high-quality supervision signals can effectively compensate for architectural simplicity, offering valuable new perspectives for image editing research.

##### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2, 3, 4, 7, 8
- [2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2, 4
- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2023. 3
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2, 3, 4, 5, 7, 1
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023. 3
- [6] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. In ICLR, 2023. 2, 3
- [7] Yusuf Dalva and Pinar Yanardag. Noiseclr: A contrastive learning approach for unsupervised discovery of interpretable directions in diffusion models. In CVPR, 2024. 4
- [8] Fei Deng, Qifei Wang, Wei Wei, Tingbo Hou, and Matthias Grundmann. Prdp: Proximal reward difference prediction for large-scale reward finetuning of diffusion models. In CVPR,

2024. 3

- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021. 1
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 1, 3
- [11] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. NeurIPS, 2024. 3
- [12] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. In ICLR, 2024. 2, 3, 7
- [13] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024. 2, 3, 5
- [14] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, et al. Instructdiffusion: A generalist modeling interface for vision tasks. In CVPR, 2024. 2, 3, 7, 1
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. NeurIPS, 2024. 2, 4
- [16] Xin Gu, Ming Li, Libo Zhang, Fan Chen, Longyin Wen, Tiejian Luo, and Sijie Zhu. Multi-reward as condition for instruction-based image editing. In ICLR, 2025. 3, 7, 8, 1

- [17] Qin Guo and Tianwei Lin. Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation. In CVPR, 2024. 2, 4
- [18] Qin Guo and Tianwei Lin. Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation. In CVPR, 2024. 2, 3
- [19] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In ICLR, 2023. 2, 3, 4
- [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 1, 3
- [22] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 2, 4
- [23] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. NeurIPS, 2023.
- [24] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In CVPR, 2024. 2, 3, 7, 8, 1
- [25] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024. 2, 3, 7
- [26] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023. 3
- [27] Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback. In ECCV, 2024. 3
- [28] Yuanze Lin, Yi-Wen Chen, Yi-Hsuan Tsai, Lu Jiang, and Ming-Hsuan Yang. Text-driven image editing via learnable regions. In CVPR, 2024. 2, 3
- [29] Bingyan Liu, Chengyu Wang, Tingfeng Cao, Kui Jia, and Jun Huang. Towards understanding cross and self-attention in stable diffusion for text-guided image editing. In CVPR, 2024. 3
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2024. 7
- [31] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2, 3
- [32] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022. 3, 4
- [33] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. In ICLR, 2024. 3, 7

- [34] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH, 2023. 3
- [35] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019. 1
- [36] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, and Daniel Cohen-Or. Localizing object-level shape variations with text-to-image diffusion models. In ICCV, 2023. 2, 4
- [37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2023. 1, 3
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 3
- [39] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. NeurIPS, 2024. 3, 4
- [40] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 1, 3
- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 3
- [43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 2, 3
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022. 3
- [45] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In CVPR, 2024. 2, 3, 1
- [46] Enis Simsar, Alessio Tonioni, Yongqin Xian, Thomas Hofmann, and Federico Tombari. Lime: localized image editing via attention regularization in diffusion models. arXiv preprint arXiv:2312.09256, 2023. 2, 4
- [47] Jaskirat Singh, Jianming Zhang, Qing Liu, Cameron Smith, Zhe Lin, and Liang Zheng. Smartmask: Context aware high-fidelity mask generation for fine-grained object insertion and layout control. In CVPR, 2024. 2, 3
- [48] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 1
- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 1
- [50] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024. 3

- [51] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 3
- [52] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024. 3, 4
- [53] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In CVPR, 2023. 2, 3
- [54] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. NeurIPS, 2023. 3, 4
- [55] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In CVPR, 2024. 3
- [56] Mingyang Yi, Aoxue Li, Yi Xin, and Zhenguo Li. Towards understanding the working mechanism of text-to-image diffusion model. arXiv preprint arXiv:2405.15330, 2024. 2, 4
- [57] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023. 3
- [58] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. NeurIPS, 2024. 2, 3, 5, 7, 1
- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2
- [60] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. In CVPR, 2024. 3, 7
- [61] Wentian Zhang, Haozhe Liu, Jinheng Xie, Francesco Faccio, Mike Zheng Shou, and J¨urgen Schmidhuber. Cross-attention makes inference cumbersome in text-to-image diffusion models. arXiv preprint arXiv:2404.02747, 2024. 2, 4
- [62] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. arXiv preprint arXiv:2407.05282, 2024. 2, 3

## SuperEdit: RectifyingandFacilitatingSupervisionforInstruction-BasedImageEditing WebsiteSupplementary CodeMaterialData

[Figure 128]

Could it be a river on the background?

L1↓/L2↓/DINO↑: 0.28/0.16/0.75 L1↓/L2↓/DINO↑: 0.08/0.02/0.89

##### 6. Overview of Supplementary

[Figure 129]

[Figure 130]

[Figure 131]

River here No river

The supplementary material is organized into the following sections:

- • Section 7: Implementation details.
- • Section 8: More experiments and analysis.
- • Section 9: More analysis on diffusion generation prior.
- • Section 10: Detailed prompt for generation prior.
- • Section 11: Discussion and limitation.
- • Section 12: More visualization comparison and results.

Input Image Ground Truth Result Our Result

IP2P Result

Figure 8. Existing metrics cannot reliably indicate editing quality.

###### 8.2. Evaluation on MagicBrush Benchmark

In Tab. 5, we present a quantitative comparison of various image editing methods evaluated on the MagicBrush single-turn benchmark. However, it’s important to note that these automated metrics (CLIP-I, CLIP-T, DINO, L1) should be interpreted with caution. As highlighted by previous works [16, 24, 45], such metrics often fail to fully capture human perceptual preferences, and can sometimes lead to misleading conclusions about actual editing quality. Several studies have demonstrated significant discrepancies between metric-based rankings and human evaluation results [16, 24, 45].

##### 7. Implementation Details

We implemented our editing model training based on the InstructPix2Pix PyTorch [35] code from the Diffusers repository [48], using Stable Diffusion V1.5 [42] as the pre-trained weights for the editing model. Following InstructPix2Pix’s implementation [4], we enable classifier-free diffusion guidance [20] for both the image condition and the text condition with 5% mask probability during training. The training batch size is 512 with a learning rate of 1e-4, weight decay of 1e-2, and a warm-up ratio of 100 steps. The training resolution is 512x512 by resizing input images without any crops. Margin m=5e−3 and weight λ=1.0 is used for triplet loss Ltriplet. We train the edit model for 10,000 steps and use the triplet loss after the 2,000 training steps. During inference, we keep the original image ratio and resize the shorter side to 512, with DDIM [49] sampler and 50 sampling steps, following the default settings of Multi-Reward [16]. The text guidance scale and image guidance scale we used for inference are 10.0 and 1.5, respectively.

Our proposed method adopts a data-oriented approach, contrasting with the model-oriented strategies prevalent in image editing. Remarkably, without requiring additional parameters, pretraining tasks, or extensive training data (using only 40K samples compared to 300K-1.2M in other methods), our approach achieves competitive performance across all metrics. The CLIP-T score of 30.3 is only 0.3 lower than the best results, and DINO score of 80.2 (second highest) is particularly noteworthy, suggesting strong preservation of both semantic and structural image features.

|Method|Extra Module<br><br>Pretrain Tasks<br><br>Edit Data<br><br>|CLIP-I↑ CLIP-T↑ DINO↑ L1↓|
|---|---|---|
|InstructPix2Pix [4] InstructDiffusion [14] MagicBrush [58] SmartEdit [24] SuperEdit (Ours)<br><br>|✗ ✗ 300K<br><br>✗ ✓ 860K<br><br>✗ ✗ 310K<br><br>✓ ✓ 1.2M<br><br>✗ ✗ 40K<br>|85.4 29.2 69.8 0.112<br><br>89.2 30.2 77.7 -<br>90.7 30.6 80.6 0.062<br><br><br>90.4 30.3 79.7 0.081<br><br>90.5 30.3 80.2 0.106<br><br><br>|

##### 8. More Experiments and Analysis

In this section, we provide more experiments and analysis. We first discuss limitations of current metrics in Sec. 8.1, then present the MagicBrush benchmark results in Sec. 8.2, and finally analyze GPT-4o cost and different VLMs in Sec. 8.3.

Table 5. Quantitative comparison (L1/CLIP-I/CLIP-T/DINO-I) on the MagicBrush benchmark. Our SuperEdit achieves good performance with better efficiency, without extra modules or pretrain tasks.

###### 8.1. Limitations of Existing Metrics

###### 8.3. GPT-4o Cost & Different VLMs’ Performance

Here, we show an example from MagicBrush test set in Fig. 8 to illustrate that existing metrics (e.g., L1/L2/DINO) cannot reflect actual editing quality; that is, the results of these metrics do not match human judgment. This dilemma has also been noted in previous instruction-based image editing works, including SmartEdit [24], Emu-Edit [45], and MultiReward [16].

We respectfully emphasize that our core contribution is identifying and addressing noisy supervision in existing datasets, rather than focusing on cost-effective scaling strategies. Using GPT-4o for our method costs $0.02 per 512×512 input-edited image pair, totaling $800 for 40K data, which is less expensive than existing works that require additional VLM fine-tuning or extra pre-training stages. For alternative ablation, we asked 5 annotators to evaluate rectified instructions from different VLMs. As shown in Tab. 6, existing open-source VLMs can partially substitute GPT-4o. These open-source models can be

In addition, SmartEdit’s metrics (CLIP, DINO) in Tab. 5 are worse than MagicBrush, but its human evaluation shows better results in SmartEdit paper [24]. This discrepancy further shows the rationale for our comprehensive assessment using both GPT-4o-based evaluation (RealEdit) and human evaluation.

###### No Prompt With Prompt No Prompt With Prompt With Prompt

###### With Prompt

0 20 30

0 0 20

10 30 30 0 30

10 10

10 20

20

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

[Figure 136]

###### Input Image

(a) Global Layout Change: Change the background to the sky

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

(b) Local Attributes Change: Turn the teddy bear red

Input Image

With Prompt

###### With Prompt With Prompt

###### With Prompt

###### No Prompt

###### No Prompt

20 10

0 10 20 30

10 0 20 30

0 20 30

0

10 30

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

[Figure 146]

(c) Style & Details Change: Change the image style to ink painting

Input Image

Figure 9. We show the impact of incorporating the editing prompt at different inference timesteps on the edited image. (a) The global layout changes usually occur in the early stages of inference. Adding text editing instructions to modify the global layout at the mid or late stages does not effectively impact the global layout. (b) Local object attribute changes occur in the mid-stages of sampling. Adding text editing instructions in the early or late stages may result in incorrect editing outcomes. (c) The style changes happen across all inference stages, and the detail changes happen in the late stage (Please refer to the subtle differences between the last two images). Best viewed in color.

further fine-tuned with GPT-4o data and then used for efficient scaling up, which we leave for future work.

text prompt” as cited in previous works.

Specifically, the figure illustrates three key patterns: (a) Global Layout Changes: The first row shows that changing the background to sky is most effective when prompts are introduced in the early stages (0-10 timesteps). When the same editing instruction is applied during mid (10-20) or late (20-30) stages, the model fails to properly modify the global layout, maintaining the original forest background despite the editing instructions. This validates our assertion that “diffusion models focus on global layout in the early stages.” (b) Local Object Attributes: The second row demonstrates that local attribute modifications, such as changing the teddy bear’s color to red, are optimally achieved during the mid-stages of sampling (10-20 timesteps). When the color change instruction is introduced too early or too late, the results show inconsistent or incomplete color transformation. This confirms that “local object attributes are processed in the mid stages”. (c) Style and Details: The third row reveals two important insights. First, style transformations (changing to ink painting style) can be effectively applied across all timesteps, indicating that style modifications have a more flexible temporal window. Second, subtle detail refinements are predominantly processed in the late stages (20-30), as

GPT-4o LLaVA-OV(72B) InternVL2(76B) Qwen2-VL(72B) 76.2% 50.4% 48.2% 47.8%

Table 6. Instruction rectification success rate across 100 samples

- 9. Diffusion Generation Prior As discussed in Sec. 3.2 and Fig. 4 of the main paper, editing diffusion models focus on specific generation attributes during inference, independent of the different editing instructions. Specifically, editing models focus on global layout in the early stages, local object attributes in the mid stages, image details in the late stages, and style change across all sampling stages. In this section, we further demonstrate this generation prior in Fig.9.

Fig.9 provides compelling visual evidence for the claims made in the main paper regarding how diffusion models process different aspects of image generation at specific timesteps. The experiments systematically demonstrate that this behavior is consistent across various editing tasks, reinforcing the observation that “different timesteps play distinct roles in image generation for text-to-image diffusion models, regardless of the

evidenced by the finer differences between the last two images in the bottom row. This supports our claim about “image details in the late stages of sampling.” These observations not only validate the theoretical framework presented in the main text but also provide practical insights for optimizing instruction-based image editing. The clear temporal division of editing capabilities suggests that a more nuanced approach to prompt timing could significantly improve editing outcomes. This understanding directly supports our approach of guiding Vision-Language Models based on these four generation attributes (global layout, local attributes, style, and details), enabling us to establish a unified rectification method applicable across various editing instructions as described in the main paper.

- 10. GPT-4o Prompts for Constructing Rectified and Contrastive Editing Instructions

We show the detailed prompt for GPT-4o to construct the rectified and contrastive editing instructions in Fig. 10. As discussed in Sec. 9, we input the original image and the edited image into GPT-4o and ask it to return the differences in the following four attributes: “Overall Image Layout “Local Object Attributes”, “Image Details”, and “Style Change”. When calling the GPT-4o API, we explicitly define “Overall Image Layout” as modifications to the major objects, characters, and background in the image. “Local Object Attributes” are defined as changes in the texture, motion, pose, and shape of the major objects, characters, and background. Additionally, we combine “Style” and “Details”intoasinglecategorytoreducethenumberoftokensgenerated by GPT-4o, thus saving costs. We observed that this adjustment does not reduce GPT-4o’s understanding of the style and detail changes between the original-edited image pair. In the actual training of the editing model, acknowledging that CLIP [38] text encoder can accept a maximum of 77 textual tokens as input, we ask GPT-4o to summarize and refine these rectified instructions. We then use the consolidated and refined editing instructions (“Summarized Instruction” in Fig. 10) to train the model.

- 11. Discussion and Limitation Discussion. It’s important to emphasize that our data-oriented approach is not mutually exclusive with model-oriented methods like MultiReward or SmartEdit, nor is its purpose to surpass existing work across various benchmarks or diminish their excellent contributions. Instead, our work explores a complementary yet important research question: What level of performance can be achieved with minimal architectural modifications by primarily focusing on supervision quality and optimization? Surprisingly, under both GPT-4o and human evaluation, our method significantly outperforms existing approaches despite using only a small amount of data, without modifying the model architecture, and requiring no additional pretraining. This suggests that highquality data can substantially compensate for architectural simplicity, achieving results comparable to or even better than methods with considerably more parameters and pretraining require-

ments. We believe our approach and experimental results bring new insights and novelty to the field of image editing research.

Furthermore, since our data-oriented approach is complementary and orthogonal to existing work, we can build upon current methods to further improve editing performance. Specifically, we follow the same setup as SmartEdit, retraining our model using InstructDiffusion as the pre-trained weights. The experimental results, as shown in Tab. 7, demonstrate that our method can complement existing work to achieve even better editing performance. When comparing SuperEdit with InstructDiffusion pre-trained weights against SmartEdit, we observe significant improvements across all metrics (71% vs. 64% in following instructions, 83% vs. 66% in preserving content, and 71% vs. 45% in image quality), despite using only 40K training samples compared to SmartEdit’s 1.2M.

|Method|Pre-trained U-Net<br><br>Model Size Edit Data<br><br>|Following ↑ Preserving ↑ Quality ↑ Acc Score Acc Score Acc Score<br><br>|
|---|---|---|
|SmartEdit<br><br>|InstrutDiff 14.1B/1.2M|64% 3.50 66% 3.70 45% 3.56|
|SuperEdit SuperEdit|SD1.5 1.1B/40K InstrutDiff 1.1B/40K<br><br>|67% 3.59 77% 4.14 65% 4.01 71% 3.76 83% 4.32 71% 4.17|

- Table 7. SuperEdit outperforms previous SOTA SmartEdit and achieves further improvements with InstructDiffusion pre-trained weights.

In addition, we also provide the results that trained with a lower resolution (256 × 256), the results on Real-Edit benchmark still outperforms previous SOTA method SmartEdit [24].

|Method<br><br>|Model Size Edit Data<br><br>Training Resolution<br><br>|Following ↑ Preserving ↑ Quality ↑ Acc Score Acc Score Acc Score|
|---|---|---|
|SmartEdit|14.1B/1.2M 256<br><br>|64% 3.50 66% 3.70 45% 3.56|
|SuperEdit<br><br>|1.1B/40K 256<br><br>|68% 3.56 75% 4.02 66% 4.02|

- Table 8. SuperEdit results with lower training resolution. Both SmartEdit and SuperEdit are pre-trained with InstructDiffusion here.

Limitation. Our method significantly enhances instructionbased image editing, but limitations still exist. The trained model still faces difficulties in understanding and executing complex instructions, especially with densely arranged objects and complicated spatial relationships. Although we used correction instructions and contrastive supervision signals, differences between editing results and editing instructions may still occur due to the inherent limitations of pre-trained Stable Diffusion and the challenges in fully capturing the nuances of natural language. Additionally, to fairly compare with existing methods, we chose Stable Diffusion v1.5 as the Base Model for building our editing model, which may result in worse image quality of edited images compared to state-of-the-art Text-to-Image models. Finally, ensuring the accuracy and effectiveness of correction and contrastive instructions requires the use of GPT-4o [1], which may incur additional costs as the amount of data increases.

##### 12. More Visualization Comparison and Results

We show more visual comparison with existing instructionbased image editing methods in Fig. 13 and Fig. 14. Compared to existing instruction-based editing methods, our approach not only better understands and executes editing instructions but also preserves the original image’s layout and quality more effectively, thereby significantly outperforming previous methods.

###### System Prompt for Instruction Rectification:

You are a professional image editor. I will give you two images later. The first image given is the original image, and the second is the edited image. You need to conduct a extremely detailed and step-by-step comparative analysis of the two input images according to the three independent aspects:

- 1. Overall Image Layout: Are there any changes in the composition and structure of the main content of the image, such as the number, size, focal length (zoom in/out), relative position, etc. of the main characters, main objects, and main background? Are there any entities that occupy a large space being deleted or added? In this section, please ignore the Texture, Motion, Pose, and Shape, Style, Color and Details.
- 2. Texture, Motion, Pose, and Shape: Are there any changes to the texture, motion, pose, or shape of the main characters, main objects, or main backgrounds? In this section, please ignore the Overall Image Layout, Style, Color and Details.
- 3. Style, Color and Details: Are there any changes to the color, tone, illumination, contrast, or style of all the object, background, or overall image? In this section, please ignore Overall Image Layout, and Texture, Motion, Pose, and Shape When you write editing instructions, please follow these rules:

- 1. Describe the editing instructions directly without referring to the information of the input image. For example, "Change the clothes to red", do not output "Change the clothes from black to red".
- 2. Describe the changes clearly, for example, "Darker the lighting, change the colors to blue tones, and change the style to anime style", do not output "Adjust/change the lighting, color palette, and style".
- 3. Please describe only the parts that have been changed, and ignore the parts that have not been changed. For example, do not output “maintain/remains xxx”.

Then, please summarize and combine the analysis, clearly describe how to transform from the input image to the edited image. In the end, put the instructions in a Python dictionary in order and make sure the same format as the following. Python dicts can only be output once, and they should be put in the last. ``` Instruction = {

"Overall Image Layout": "Detailed instruction", "Texture, Motion, Pose, and Shape": "Detailed instruction", "Style, Color and Details": "Detailed instruction", “Summarized Instruction”: “Combine and summarize the aforementioned details into a

comprehensive and concise transformation guide."

} ```

###### System Prompt for Contrastive Instructions:

You are a professional image editor. I will give you two images. The first one is the original image, and the second one is the edited image. Then I will give you an editing instruction, which describes how to edit from the original image to the edited image. Now you need to change the correct input editing instructions to the wrong ones, including changing the quantity, position/relation, image style, color, category and attribute of the original editing instruction. Then integrate each modified editing instruction and return it in the form of a list. Please directly output the modified editing instructions in the following format: ``` Instruction = ["instruction with wrong quantity", "instruction with wrong position/relation", "instruction with wrong image style", "instruction with wrong color", "instruction with wrong object category", "instruction with wrong object attribute"] ```

Figure 10. GPT-4o prompts for constructing rectified and contrastive editing instructions

|System Prompt for Instruction Rectification:<br><br>You are a professional image editor. I will give you two images later. The first image given is the original image, and the second is the edited image. You need to conduct a extremely detailed and step-by-step comparative analysis of the two input images according to the three independent aspects:<br><br>1. Overall Image Layout: Are there any changes in the composition and structure of the main content of the image, such as the number, size, focal length (zoom in/out), relative position, etc. of the main characters, main objects, and main background? Are there any entities that occupy a large space being deleted or added? In this section, please ignore the Texture, Motion, Pose, and Shape, Style, Color and Details.<br>2. Texture, Motion, Pose, and Shape: Are there any changes to the texture, motion, pose, or shape of the main characters, main objects, or main backgrounds? In this section, please ignore the Overall Image Layout, Style, Color and Details.<br>3. Style, Color and Details: Are there any changes to the color, tone, illumination, contrast, or style of all the object, background, or overall image? In this section, please ignore Overall Image Layout, and Texture, Motion, Pose, and Shape<br>|
|---|

|[Figure 147]|
|---|

- 1.Overall Image Layout: The primary change in the overall image layout is that the right image provides a close-up of the fox, making it appear larger in the frame. Additionally, the fox transitions from a standing position in the left image to a sitting position in the right image.
- 2.Texture, Motion, Pose, and Shape: The most significant change in this section is the pose of the fox, which shifts from standing in the left image to sitting in the right image. The texture, motion, and shape of the fox and the background remain consistent.
- 3.Style, Color, and Details: The style and color palette are consistent between the two images, with a focus on realism and a wintry setting. The most important change in this section is the increased detail in the fox's facial features in the right image due to the close-up perspective.

- 1. Overall Image Layout: The edited image zooms in on the fox, centering its upper body, and reduces the prominence of background elements like trees and shadows, whereas the original includes the entire fox and more of the forest.
- 2. Texture, Motion, Pose, and Shape: The fox in the edited image has a more detailed fur texture and is sitting, conveying calmness, while the original fox is standing, suggesting alertness.
- 3. Style, Color, and Details: The edited image has richer, more saturated colors with a cooler tone and higher contrast, making details sharper. The original image features warmer sunlight and softer contrasts, giving it a naturalistic and serene feel.

[Figure 148]

[Figure 149]

#### GPT-4o InternVL2-Pro

- 1.Overall Image Layout: The composition has changed from a standing fox to a sitting one, altering the focal point and perspective. The forest background remains consistent, but the fox's position shifts from left to center.
- 2.Texture, Motion, Pose, and Shape: The fox's pose has been altered from standing to sitting, changing its body orientation and limb positions. The texture of the fur and scarf, as well as the snow and trees, remains unchanged.
- 3.Style, Color and Details: The color palette, lighting, and overall style of the image are consistent between the two versions. The details of the fox's facial features, the scarf's knit pattern, and the snowflakes on the trees remain the same.

- 1. Overall Image Layout: Cropping the image tighter around the subject. Additionally, applying a more pronounced background blur can help to isolate the fox and draw the viewer's eye directly to it.
- 2. Texture, Motion, Pose, and Shape: Subtle texture details can be added to the fox's fur. Additionally, slight adjustments to the fox's pose can create a more dynamic and engaging composition.
- 3. Style, Color, and Details: Using dramatic lighting can help to emphasize the fox's features and create a sense of depth and dimension.

[Figure 150]

[Figure 151]

#### Gemini LLaVA-One-Vision

- Figure 11. Comparison of different vision-language models in rectifying editing instructions based on generation prior attributes. GPT-4o achieves more stable and accurate results in describing the differences between original-edited image pairs. Text in red represents incorrectly generated instructions.

[Figure 152]

[Figure 153]

Background Change: Replace the forest background with an underwater scene, incorporating elements like coral reefs and small fish to create an aquatic environment, and use gradient overlays to simulate sunlight penetration through the water. Color and Lighting Adjustments: Apply a cool blue filter for an underwater ambiance, adjust the lighting to create light rays penetrating the water, and ensure shadows and highlights fit the underwater lighting pattern. Parrot Adjustments: Retain the parrot's primary colors with slight tint adjustments, enhance its appearance to fit the underwater scene, and add reflection to blend seamlessly with the underwater light. Additional Underwater Elements: Add underwater flora, fauna, bubbles, water particles, and lightrays to increase depth and realism, making the scene more believable.

- Step 1: General Overview. The original image displays a metallic, shiny teapot with a smooth, reflective surface, while the edited image showcases a ceramic teapot adorned with a blue floral pattern.
- Step 2: Material and Texture Comparison. The original teapot is made of shiny metal, possibly stainless steel or silver, exhibiting a reflective surface. In contrast, the edited teapot is ceramic with a matte finish and intricate blue floral patterns.

...... Step 8: Artistic Style and Finish: The original teapot has a modern, minimalistic style emphasizing sleekness and simplicity, while the edited teapot boasts a traditional, decorative look with detailed floral patterns.

[Figure 154]

|System Prompt for Instruction Rectification:<br><br>You are a professional image editor. I will give you two images later. The first image given is the original image, and the second is the edited image. You need to conduct a extremely detailed and stepby-step comparative analysis.|
|---|

[Figure 155]

- Figure 12. If the predefined four generation prior attributes are not used as templates for in-context learning, the GPT-4o rectified editing instructions will contain redundant information and lack the standardization needed for scalable processes.

Replace the lighthouse with a tall, palm tree

Change the background to a snowy winter landscape

Change the background to a clear blue sky

Turn the entire scene into a spring setup, with blooming flowers and lush greenery

Remove the collar from the dog's neck

Editing Instruction

Change car paint to matte black

Add a sandcastle near the water's edge

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

Original Image

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

Ours

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

SmartEdit

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

HIVE

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

HQ-Edit

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

Instruct Diffusion

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

InstructP2P

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

MagicBrush

Figure 13. More visual comparison with existing methods.

Change the background to show a city skyline instead of mountains

Change the image style to a watercolor painting

Editing Instruction

Put a blue shirt on the boy

Change the water texture to look like lava

Remove some clouds in the sky

Add a toy car on the left side of the girl

Remove the hot air balloon

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

Original Image

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

Ours

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

SmartEdit

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

HIVE

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

HQ-Edit

|[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

Instruct Diffusion

|[Figure 254]|
|---|

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|
|---|

InstructP2P

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

MagicBrush

Figure 14. More visual comparison with existing methods.

