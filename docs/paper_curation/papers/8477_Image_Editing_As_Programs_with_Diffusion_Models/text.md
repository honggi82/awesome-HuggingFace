arXiv:2506.04158v1[cs.CV]4Jun2025

# Image Editing As Programs with Diffusion Models

Yujia Hu, Songhua Liu, Zhenxiong Tan, Xingyi Yang, and Xinchao Wang∗

National University of Singapore {yujia.hu,songhua.liu,zhenxiong,xyang}@u.nus.edu,xinchao@nus.edu.sg

[Figure 1]

[Figure 2]

Complex Instruction Editing-Step Display

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Input Change the background to the forest. Make the lady wear a white dress. Add a fox beside the lady. Change the time to autumn.

Original Instruction: What would it be like if we placed the lady in a forest during autumn, and she was wearing a white dress, with a fox beside her?

[Figure 12]

[Figure 13]

[Figure 14]

###### Simple Instruction Editing

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

Replace coffee with a cat. Change expression to not smiling. Minify the plane.

Move the plant on the cupboard.

[Figure 29]

[Figure 30]

Complex Instruction Editing-Results Only

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

Add a dog on the grass and remove the flowers.

Change the expression to smiling and make it a sketch.

Put the cat on the beach, make it decorated in a Christmas style and be in orange color.

Change the material of the sign to wood and turn the red car into a tiger.

Enlarge the unicorn on the left and move it on the lake.

Change the action of the baby to crawling and make it a stormy day.

- Figure 1: Visual results of our IEAP. Rows 1 and 3 showcase complex multi-step edits (Row 1 is further decomposed into individual instructions), while Row 2 shows single-instruction edits. Single instructions are underlined if needing to be reduced to atomic operations.

## Abstract

While diffusion models have achieved remarkable success in text-to-image generation, they encounter significant challenges with instruction-driven image editing. Our research highlights a key challenge: these models particularly struggle with structurally inconsistent edits that involve substantial layout changes. To mitigate this gap, we introduce Image Editing As Programs (IEAP), a unified image editing framework built upon the Diffusion Transformer (DiT) architecture. At its core, IEAP approaches instructional editing through a reductionist lens, decomposing complex editing instructions into sequences of atomic operations. Each operation is implemented via a lightweight adapter sharing the same DiT backbone and is specialized for a specific type of edit. Programmed by a vision-language model (VLM)-based agent, these operations collaboratively support arbitrary and structurally inconsistent transformations. By modularizing and sequencing edits in this

∗Corresponding Author Preprint. Under review.

way, IEAP generalizes robustly across a wide range of editing tasks, from simple adjustments to substantial structural changes. Extensive experiments demonstrate that IEAP significantly outperforms state-of-the-art methods on standard benchmarks across various editing scenarios. In these evaluations, our framework delivers superior accuracy and semantic fidelity, particularly for complex, multi-step instructions. Codes are available here.

## 1 Introduction

Image editing lies at the heart of a wide range of applications from photo retouching and content creation to visual storytelling and scientific visualization [42, 5, 62]. With the advent of diffusion models [23, 53, 47], the field has shifted towards highly precise and controllable manipulations

- [45, 12, 61]. The inherently progressive denoising process enables multi-stage pipelines [24, 4, 2] and localized editing methods [10, 74, 57], and its native support for multi-modal inputs has inspired unified frameworks that integrate heterogeneous signals within a single model [33, 15, 68, 18].

More recently, text-to-image pipelines based on Diffusion Transformers (DiTs) [46, 13, 31] have set new standards in generative fidelity. However, their capacity for instruction-driven editing [41, 27] remains under-explored. Notably, although there are a few existing methods [77, 37] that have extended DiTs to instruction-driven editing, they are always restricted to a narrow set of common editing operations and lack evaluation on comprehensive editing tasks.

To address this limitation, we initiate a taxonomy study of image editing instructions to systematically assess the editing capabilities of current DiT-based conditional generation methods. Our empirical analysis reveals an interesting performance dichotomy: While current methods demonstrate proficiency in structurally-consistent edits where the layouts of the input and output images remain aligned, they exhibit significant degradation when handling structurally-inconsistent operations that require layout modifications.

To overcome this issue, we introduce Image Editing As Programs (IEAP), a unified framework atop the DiT architecture which is capable of handling diverse types of editing operations efficiently and robustly in this paper. Notably, we show that structurally-inconsistent instructions can in fact be reduced to a small set of simple operations, which are called as atomic operations in our paper. Thus, instead of treating each edit as a monolithic, end-to-end task, IEAP levarages the Chain-of-Thought (CoT) reasoning [63] to break the original editing command into a sequence of atomic operations, which are namely Region of Interest (RoI) localization, RoI inpainting, RoI editing, RoI compositing and global transformation, and then executes them in a sequential manner via a neural program interpreter [49].

The five atomic operations serve as the fundamental building blocks for complex editing tasks. As such, through the sequential combination of atomic operations, IEAP can robustly handle complex, multi-step instructions that are typically confound in conventional end-to-end approaches.

Extensive experiments show that our framework demonstrates state-of-the-art performance across standard benchmarks, excelling in both structural preservation and alteration tasks through atomiclevel operation decomposition compared to other approaches. Simultaneously, the CoT reasoning and programming pipeline of IEAP enable significantly more accurate and semantically more coherent edits under complex, multi-step instructions even compared to the leading proprietary models.

Our main contributions can be summarized as follows:

- • We present a comprehensive taxonomy and empirical analysis of instruction-driven editing in DiT-based conditional generation, revealing a performance dichotomy between structurallyconsistent and -inconsistent edits.
- • We introduce Image Editing As Programs (IEAP), a unified framework on the DiT backbone that leverages CoT reasoning to parse free-form instructions into sequential atomic operations and then executes them sequentially by a neural program interpreter, thereby enabling robust handling of layout-altering and complex edits.
- • Extensive experiments demonstrate that IEAP achieves state-of-the-art performance in both structure-preserving and -altering scenarios, delivering notably higher accuracy and semantic fidelity especially on complex, multi-step instructions compared to existing methods.

## 2 Related Work

Conditional image generation. Early conditional image generation approaches like ControlNet [74] typically adopt plug-in control adapters to incorporate single condition [3, 16, 35] like segmentation mask or diverse conditional inputs [79, 48, 26, 40, 67] to guide the generation of images. Recently, the field of conditional image generation has witnessed remarkable breakthroughs through the integration of DiTs [13, 46, 31], with continuous innovations improving output quality and edit precision [45]. Some methods [66, 32, 65, 9] aim to create a unified DiT foundation for versatile conditional image generation and editing by integrating diverse inputs within a single framework. while approaches like OminiControl [59] and so on [60, 76, 38, 77, 64] leverage LoRA-based fine-tuning [25] for lightweight and effective control.

Instructional image editing. Instruction-based image editing [41, 27] enables intuitive, languagedriven modifications of existing images. Early works like InstructPix2Pix [6] establishes paired instruction–image datasets for supervised fine-tuning of diffusion models. For subsequent works, some of them focus on architectural refinement [38, 37, 78, 34, 20], which introduce specialized conditioning units and multi-stage training to improve control granularity and consistency, others concentrate on data-centric enhancements [73, 17, 55, 8], that expand instruction coverage and diversify edit examples. Moreover, some approaches [72, 28, 33, 15] has unified LLM-based [1] language reasoning with diffusion-based synthesis in a single framework, and some [69, 75] leverage CoT [63] and in-context learning [21] to enhance the reasoning ability of models for more complex editing tasks. More recently, some works [14, 77, 37] have advanced image editing with DiTs. For instance, ICEdit [77] leverages the in-context generation capabilities of large-scale DiTs to achieve flexible few-shot instruction editing, while Step1X-Edit [37] focuses on large-scale data construction and multi-modal integration to enable general-purpose image editing with performance approaching proprietary models.

## 3 Motivation

### 3.1 Preliminaries

Diffusion Transformer Fundamentals. The image generation process of text-guided DiTs [46, 13, 31] is accomplished by successively denoising input tokens in multiple steps. At step t, the model processes:

St = [Xt,CT] (1)

where Xt ∈ RN×d represents noisy image tokens and CT ∈ RM×d denotes text tokens, they share the embedding dimension d. Image tokens use Rotary Position Embedding (RoPE) [58] with spatial coordinates (i,j), while text tokens fix positions at (0,0), enabling Multi-Modal Attention (MMA) [44] mechanisms to model cross-modal interactions.

Unified Conditioning Framework. To integrate visual control signals, the prior work [59] extends the baseline formulation by incorporating encoded condition images:

#### St = [Xt,CT,CI] (2)

where CI ∈ RN×d denotes latent tokens from condition images via the pretrained VAE encoder [30, 52]. This unified sequence enables tri-modal fusion within transformer architectures, eliminating spatial misalignment inherent in feature concatenation baselines.

Moreover, an auxiliary adaptive positional encoding mechanism further preserves spatial consistency across these modalities by assigning coordinates to each token type with minimal overhead.

Gap in Instruction-Driven DiT Editing. Despite the rapid advances in DiT-based conditional image generation [59, 76, 38, 64], research on instruction-driven editing [41, 27] remains scarce. The few existing methods [77, 37] that do support instructional edits are typically confined to a small set of routine operations, and lack a comprehensive evaluation across diverse editing scenarios, leaving DiT’s true editing potential unclear. This gap motivates us to conduct a taxonomy study of DiT’s ability in instructional image editing, which is detailed in Sec. 3.2.

[Figure 43]

[Figure 44]

[Figure 45]

[Add] “Add a person on the boat.”

[Remove] “Remove the bananas.” [Replace] “Replace the dog with a backpack.”

[Figure 46]

Instruction Failthfulness

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Semantic Consistency

Scores

[Figure 53]

[Figure 54]

[Figure 55]

[Action change] “Change the action of the girl to jumping.”

[Resize] “Enlarge the bird.” [Move] “Move the couch to the left.”

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Overal Content Editing

Local Atribute Editing

Local Semantic Editing

Editing Types

(a) (b)

- Figure 2: Results of our preliminary experiments. Figure (a) shows the GPT-4o scores for three editing types across instruction faithfulness and semantic consistency, ranging from 1 to 5. Figure (b) shows the representative failure cases from local semantic editing.

### 3.2 Preliminary Experiments and Observations

To this end, we conduct a comprehensive evaluation of diffusion models for instruction-driven editing, uncovering an interesting performance dichotomy: While these methods excel at structurallyconsistent edits, they falter dramatically on structurally-inconsistent operations that demand explicit layout modifications.

Taxonomy and Experimental Setup. To enable systematic analysis [27, 70, 69], we first categorize instruction-based image editing into three main types: local semantic editing, which modifies the identity, position or size, e.g., add, remove, replace, action change, move and resize; local attribute editing, which adjusts certain properties of objects, e.g., color change, texture change, appearance change, expression change, and background change; and overall content editing, which alters the whole image consistently, e.g., tone transfer and style change.

Then we use AnyEdit dataset [70] and OminiControl [59] to train models on the above editing types, accompanied by GPT-4o [29] to rate each edit on instruction faithfulness and semantic consistency.

Results and Analysis. As shown in Fig. 2(a), both local attribute editing and overall content editing attain relatively high GPT-4o scores, whereas local semantic editing exhibits a notable performance drop. As illustrated in Fig. 2(b), the cases of “add” and “action change” alter unrelated areas like the background, and the remaining four cases demonstrate a complete failure.

We attribute this discrepancy to the fact that, unlike local attribute and overall content edits, local semantic edits require explicit spatial-layout modifications. For instance, “add” and “delete” operations necessitate instance-level scene recomposition, while “move” and “resize” further demand precise coordinate system recalibration.

Key Insight. Based on the above analysis, spatial-layout modification remains a critical challenge for diffusion-based editing models; conversely, edits that preserve the original layout demonstrate substantially better performance. We speculate that, with limited training data, it is difficult for the model to learn the complex patterns underlying layout-changing tasks. Although DiT architectures

- [46, 13, 31] employ powerful full-attention mechanisms to capture long-range dependencies, they still struggle with editing operations that require nontrivial scene reconfiguration.

Due to the combinatorial complexity of spatial-layout modifications and the empirical limitations of DiT architectures, we propose to simplify the layout-editing paradigm through decomposition, which is detailed in Sec. 4.

## 4 Methods

### 4.1 Program with Atomic Operations

The insight in Sec. 3.2 motivates us to decouple semantic and spatial reasoning. Building on this foundation, we propose a programmatic reduction framework that systematically decomposes complex editing instructions into modular atomic operations. Specifically, we first formulate instruction-driven image editing as an executable program via Chain-of-Thought (CoT) reasoning [63], and then use a neural program interpreter [49] to transcode the reasoning graph into a dynamic execution plan, sequentially invoking relevant atomic modules.

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

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Action Change

Resize Move

Add Remove Replace

[Figure 86]

[Figure 87]

[Figure 88]

Local Semantic Editing

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Expression Change

Appearance Change

Background Change

Color Change Texture Change

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Local Attribute Editing

[Figure 100]

[Figure 101]

Instruction: “Make the cat have a floral pattern, put a vase on the chair. Then, erase the pink decoration on the wall and alter the color of the spoon to yellow. Replace the biggest blue coffee cup with a cake and zoom in the pink cup next to the cat. Finally, change the time to the evening.”

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Tone Change Style Transfer

[Figure 107]

Overall Content Editing

[Figure 108]

###### IEAP

[Figure 109]

Appearance Change Add Remove Replace Resize Tone Change

Color Change

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

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

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

RoI Localization RoI Inpainting RoI Editing RoI Compositioning Global Transformation

- Figure 3: Our pipeline. The original instruction is first parsed by a VLM into atomic operations, which are then sequentially executed via a neural program interpreter.

- 4.2 General Pipeline

We abstract all editing instructions into five atomic primitives: (1) RoI Localization: Identify and isolate the relevant region in the image that the instruction refers to, serving as the spatial grounding step for subsequent localized edits; (2) RoI Inpainting: Introduce new visual content or remove existing elements within the localized region, enabling semantic-level additions, substitutions, or deletions; (3) RoI Editing: Modify visual attributes within the region, such as color, texture, or appearance, to reflect fine-grained property changes specified by the instruction; (4) RoI Compositing: Reintegrate the edited region into the full image while preserving spatial coherence and visual continuity; (5) Global Transformation: Adjust the overall content for coherent full-image modifications, such as changing the illumination, weather, or style of the whole image.

The overall pipeline is shown as Fig. 3. We reduce any editing instruction into an arbitrary combination of the five atomic operations described above, which can be formulated as:

K

T ≡

Ak, Ak ∈ {Aloc,Ainp,Aedit,Acomp,Aglobal} (3)

k=1

where T denotes the free-form editing instruction, represents the sequential program combination, K is the number of atomic operations, Aloc, Ainp, Aedit, Acomp, and Aglobal represent the five atomic primitives respectively.

RoI Localization. All problematic local semantic edits share a common first step: localizing a Region of Interest (RoI) in the image for editing. Given an image I and an editing instruction T, we first employ a Large Language Model (LLM) [1] to locate the text RoI:

ρ = MLLM(T), (4)

where ρ represents the text RoI extracted by the LLM MLLM. Subsequently, we achieve accurate localization of image RoI by:

R = Mseg(I,ρ), (5) where R denotes the image RoI segmented by the segmentation model Mseg [71].

For add operation, the instruction may not specify a text RoI, or the specification may be ambiguous. In such cases, we first derive the overall layout of all candidate objects using the capability of segmentation models [50, 71], and then prompt the LLM to determine the appropriate image RoI based on T.

Regarding move and resize, once the image RoI is obtained, we update the spatial layout of the image using an LLM [1]. Specifically, we provide the LLM with a set of in-context examples that define our layout representation and demonstrate representative editing patterns [36]. Given the current layout L and the instruction T, the LLM is prompted to produce a modified layout Ledit, as formulated below:

Tags = MLLM(I), L = Mseg(Tags), Ledit = MLLM(L,T). (6)

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

RoI Localication RoI Inpainting RoI Editing RoI Compositioning

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Input

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Annular mask

Change the action of the woman to dancing.

(a) Example Procedure of Action Change Operation.

[Figure 156]

[Figure 157]

[Figure 158]

RoI Localication RoI Inpainting RoI Compositioning Input

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

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

Layout Reconfiguration

Move the apple with the cat together so that the cat can intersect the computer.

(b) Example Procedure of Move Operation.

- Figure 4: Example procedure. Figure (a) and Figure (b) illustrate the procedures of action change and movement respectively.

We then derive the geometric differences between L and Ledit and convert them into the corresponding affine transformations, consisting of translation, scaling, and reshaping, and apply it to R to update the spatial configuration, yielding the transformed mask R′.

RoI Inpainting. Once the image RoI has been localized, we apply inpainting to seamlessly fill and complete the region. For additive and substitutive operations, which aim to introduce new objects, we employ a prompt-conditioned inpainting process to guide the generation of new content. Specifically, we first extract the semantic entity E from the instruction T via an LLM [1]:

E = MLLM(T), (7) and then construct a composite prompt P in the form: “add E on the black region”. For removal operations, which aim to eliminate existing content without introducing new semantics, we adopt a background-oriented infilling strategy, setting P as “fill in the hole of the image”. The edited image Iedit is then generated by:

Iedit = Minpaint (I ⊙ (1 − R),P), (8) where Minpaint denotes the inpainting model trained by us. RoI Editing. When operations pertain to property change are performed, we use the trained attribute editing model Mattr to perform edits in this stage to obtain Iedit:

Iedit = Mattr (I,T). (9)

RoI Compositing. To ensure seamless integration of the edited RoI with its surrounding context, we first construct an annular mask Mann by applying morphological dilation and erosion [51, 54] to the transformed RoI mask R′:

Mann = Dilate(R′, k1) \ Erode(R′, k2). (10)

Then, we employ a fusion network Mfusion, trained on ring-masked object boundaries, to refine the pre-composited image Iprep using the generated annular mask. The final edited image is obtained as:

Iedit = Mfusion (Iprep ⊙ (1 − Mann),P), (11) where P is set as “inpaint the black-bordered region so that the object’s edges blend smoothly with the background” to guide seamless boundary blending.

Global Transformation. Like RoI editing, in the scenarios involving global transformation, we use the trained global transformation model Mglobal to perform edits in this final stage to obtain Iedit.

- 5 Experiments

### 5.1 Experimental Settings

Training Settings. We train four specialized models for RoI inpainting, RoI editing, RoI compositing, and global transformation respectively. All models are fine-tuned on FLUX.1-dev [31] using LoRA

###### Input InstructP2P MagicBrush UltraEdit ICEdit IEAP(Ours)

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Add] Add a bird on the yellow stool.

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Remove] Remove the bananas.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Replace] Replace the woman with a mirror.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Action Change] Change the action of the woman to running.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Color Change] Change the color of the boy’s hair to red.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Style Change] Change the style of the image to bubbles.

- Figure 5: Comparison results of ours with baseline methods on representative editing cases. Others exhibit poor performance even on some common editing operations, while our approach demonstrates superior effectiveness across all operations.

[25], with default settings for rank 128 and alpha 128. Training is conducted with a batch size of 1 and runs for 50,000 iterations each. We use the Prodigy optimizer [39], enabling safeguard warmup and bias correction, with a weight decay of 0.01. The experiments are conducted on single NVIDIA H100 GPU (80GB).

Dataset Setup. For both the RoI editing and global transformation models, we sample from the relevant subsets of the AnyEdit [70] dataset and apply GPT-4o [29] to filter the data of some types that have numerous noisy examples. To cover facial expression edits absent in AnyEdit, we integrate the CelebHQ-FM dataset [11], which offers consistent identities and annotated expressions suitable for our instruction schema. The RoI inpainting and RoI compositing models are trained on samples from the “add”, “remove” and “replace” splits of AnyEdit. For each sample, we first obtain the image RoI according to the editing instruction. In the RoI Inpainting training setup, we set the pixels within image RoI to black as input to train. For RoI Compositing, we set k1 and k2 as 3 in default to blackout the annular mask region of image RoI as input for training.

MagicBrush test AnyEdit test CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ CLIPim ↑ L1 ↓ DINO ↑ GPT ↑

Method

InstructPix2Pix 0.838 0.229 0.112 0.758 0.801 0.110 0.765 3.83 MagicBrush 0.886 0.241 0.074 0.859 0.824 0.128 0.742 3.90 UltraEdit 0.911 0.227 0.061 0.883 0.833 0.114 0.772 3.93 ICEdit 0.913 0.236 0.058 0.885 0.847 0.110 0.765 4.13

Ours 0.922 0.247 0.060 0.897 0.882 0.096 0.825 4.41

Table 1: Quantitative results on MagicBrush and AnyEdit test set.

Local Semantic Editing Local Attribute Editing Overall Content Editing CLIPim ↑ L1 ↓ DINO ↑ GPT ↑ CLIPim ↑ L1 ↓ DINO ↑ GPT ↑ CLIPim ↑ L1 ↓ DINO ↑ GPT ↑

Method

InstructP2P 0.826 0.132 0.738 3.74 0.790 0.135 0.737 3.92 0.766 0.156 0.642 3.91 MagicBrush 0.860 0.106 0.796 3.90 0.809 0.117 0.762 4.21 0.763 0.187 0.616 3.99 UltraEdit 0.867 0.095 0.812 3.86 0.801 0.092 0.793 3.94 0.754 0.201 0.611 4.41 ICEdit 0.881 0.088 0.810 4.08 0.825 0.095 0.795 4.06 0.759 0.188 0.603 4.45

Ours 0.907 0.081 0.854 4.42 0.861 0.083 0.821 4.54 0.895 0.107 0.879 4.51

Table 2: Quantitative results on different types of editing operations.

Evaluation Settings. We evaluate our method on two benchmarks: MagicBrush test set [73], a widely used dataset spanning diverse editing types, and AnyEdit test set [70], from which we select 16 instruction-based editing categories. For MagicBrush, we follow previous works [73, 78, 15, 55] and report CLIPimg, CLIPout [22], L1, and DINO [7, 43] scores to measure the similarity between the generated results and ground-truth images. While for AnyEdit, where some categories lack reference captions required for calculating CLIPout, we instead leverage GPT-4o [29] to assign ratings on a scale from 1 to 5 across three aspects: instruction faithfulness, semantic consistency, and aesthetic quality. The final quality score is computed as the average of these three dimensions.

We first compare our method with existing state-of-the-art open-source baselines, including InstructPix2Pix [6], MagicBrush [73], UltraEdit [78], and ICEdit [77]. In addition, to demonstrate the competitiveness of our approach against powerful proprietary multimodal foundation models in complex image editing scenarios, we further make comparisons with SeedEdit (Doubao) [56], Gemini 2.0 Flash [19], and GPT-4o [29].

### 5.2 Comparisons with State of the Art.

Qualitative Comparisons. Fig. 5 shows the results of ours against other four methods [6, 73, 78, 77] on some representative editing cases, where our method demonstrates comprehensive superiority over others in accurate instruction execution, structural consistency, and instance-level fidelity.

Quantitative Comparisons. Table 1 exhibits the quantitative comparison results of our method and other approaches [6, 73, 78, 77] on MagicBrush test set [73] and AnyEdit test set [70]. The results show that our method demonstrates state-of-the-art performance on both datasets. On MagicBrush, our method achieves the best performance in terms of caption alignment, semantic consistency, and preservation of fine-grained structural details. Although it incurs a marginal increase in pixel-level deviation compared to the best [77], this is far outweighed by the substantial gains in perceptual quality and semantic fidelity. Furthermore, on AnyEdit, our approach yields significant and comprehensive improvements across all evaluation metrics, further highlighting its superiority over existing techniques.

To provide a more fine-grained analysis of editing performance, we group a subset of the instructionbased categories from the AnyEdit test set [70] into three macro-tasks: local semantic editing, local attribute editing and overall semantic editing. For local attribute editing, we augment with some CelebHQ-FM [11] test images to evaluate facial expression changes. The quantitave comparison results are shown in Tab. 2, where our method consistently outperforms other candidates across all three task categories and evaluation metrics.

Comparisons with Cutting-Edge Multimodal Models. To demonstrate the superiority of our reduction strategy on complex editing tasks, we also conduct comparative experiments against prominent closed-source multimodal models [56, 19, 29]. As illustrated in Fig. 6, our method rivals, and in most cases surpasses the performance of these leading models on intricate scenarios requiring

###### Input SeedEdit(Doubao) Gemini 2.0 Flash GPT-4o IEAP(Ours) Input SeedEdit(Doubao) Gemini 2.0 Flash GPT-4o IEAP(Ours)

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

Move the biggest orange butterfly to the center of the image and change the time to evening.

Add a cake on the table, make the table to be pink and be in a floral pattern, remove the left chair.

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Remove the necklace, add a pair of sunglasses on the dog and put on a clothes with text ‘IEAP’.

Change the action of the cat to jumping and replace the lamp and cupboard with a plant.

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

Change the material of the car to corduroy and minify the car. Put the car on the street scene.

Remove the monkeys and change the style to painting.

- Figure 6: Comparisons on Complex Instructions with Leading Multimodal Models. Our method achieves comparable or even better edit completeness and pre-post consistency.

multiple sequential edits. Unlike competing approaches, which frequently omit specified instructions or introduce extraneous alterations unrelated to the editing directives, our framework faithfully executes each instruction while maintaining superior image consistency and instance preservation.

5.3 Ablation Studies

Settings CLIPim ↑CLIPout ↑ L1 ↓ DINO ↑GPT ↑ w/o CoT & Reduction 0.873 0.241 0.117 0.795 4.10 w/o RoI Inpainting 0.861 0.218 0.124 0.775 3.65 w/o RoI Editing 0.900 0.244 0.088 0.843 4.23 w/o Layout Reconfiguration 0.900 0.245 0.088 0.848 4.31 w/o Annular Mask Integration 0.906 0.252 0.083 0.854 4.39 Full 0.907 0.252 0.081 0.854 4.42

Table 3: Ablation results on AnyEdit local semantic editing test set.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Instruction: Change the action of the dog to jumping.

Input w/o CoT & Reduction w/o RoI Inpainting

Full w/o RoI Editing w/o Annular Mask

Background change Strange filling

Inconsistent instance Unnatural edges

Figure 7: Qualitative ablation of action change operation.

Module-wise Ablation Studies. To quantify the impact of each key component in our framework, we perform a series of ablation studies on the AnyEdit local semantic editing test set as we split in Sec. 5.2. As shown in Tab. 3, we first substitute our CoT reasoning and reduction pipeline with end-to-end editing pipeline, resulting in a marked performance deterioration across all metrics. Next, we replace our specialized RoI inpainting and RoI editing models respectively with the generic inpainting model from [59], which induces performance declines of varying degrees. We then remove the LLM-guided layout reconfiguration and instead employing random layout modifications for relevant operations, which incurs a noticeable performance decline. Finally, omitting the annular mask integration produces a modest drop, underscoring its role in precise boundary delineation. Fig.

- 7 exhibits the ablation results on an example of “action change”, visually showcasing each module’s necessity. Collectively, these ablation results confirm that each component in our pipeline contributes significantly in handling robust local semantic editing tasks requiring layout changes.

## 6 Conclusions, Limitations and Future Work

In this paper, we propose Image Editing As Programs (IEAP), a unified DiT-based framework for instruction-driven image editing. By defining five atomic operations and using CoT reasoning to convert instructions into sequential programs, IEAP processes the ability to handle both simple and complex edits. Experiments demonstrate that IEAP outperforms state-of-the-art methods in both structure-preserving and structure-altering tasks, especially for complex edits.

Despite its strong overall performance, there are also some limitations. First, for complex shadow changes, our method sometimes leaves shadows inconsistent after compositing operations. Second, multiple editing iterations may induce progressive image quality decay. Future work could focus on addressing these issues via physics-aware shadow modeling and diffusion-based quality restoration.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, et al. Gpt-4 technical report, 2024.
- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM transactions on graphics (TOG), 42(4):1–11, 2023.
- [3] Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, and Xi Yin. Spatext: Spatio-textual representation for controllable image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18370– 18380, 2023.
- [4] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022.
- [5] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Trans. Graph., 28(3):24, 2009.
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [8] Tuhin Chakrabarty, Kanishk Singh, Arkadiy Saakyan, and Smaranda Muresan. Learning to follow objectcentric image editing instructions faithfully. arXiv preprint arXiv:2310.19145, 2023.
- [9] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774, 2024.
- [10] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022.
- [11] Brian DeCann and Kirill Trapeznikov. Comprehensive dataset of face manipulations for development and evaluation of forensic tools, 2022.
- [12] Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. Advances in Neural Information Processing Systems, 36:16222–16239, 2023.
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [14] Kunyu Feng, Yue Ma, Bingyuan Wang, Chenyang Qi, Haozhe Chen, Qifeng Chen, and Zeyu Wang. Dit4edit: Diffusion transformer for image editing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2969–2977, 2025.
- [15] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instructionbased image editing via multimodal large language models. arXiv preprint arXiv:2309.17102, 2023.
- [16] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scene-based text-to-image generation with human priors. In European Conference on Computer Vision, pages 89–106. Springer, 2022.
- [17] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, et al. Instructdiffusion: A generalist modeling interface for vision tasks. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 12709–12720, 2024.
- [18] Vidit Goel, Elia Peruzzo, Yifan Jiang, Dejia Xu, Xingqian Xu, Nicu Sebe, Trevor Darrell, Zhangyang Wang, and Humphrey Shi. Pair diffusion: A comprehensive multimodal object-level image editor. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8609–8618,

- 2024.

- [19] Google. Experiment with gemini 2.0 flash native image generation. Technical report, Google AI Studio,

- 2025.

- [20] Qin Guo and Tianwei Lin. Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6986–6996, 2024.
- [21] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training, 2022.
- [22] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.
- [24] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022.
- [25] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.
- [26] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023.
- [27] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. Diffusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 1–27, 2025.
- [28] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8362–8371, 2024.
- [29] Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, et al. Gpt-4o system card, 2024.
- [30] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.
- [31] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [32] Duong H Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan Mandt, Ranjay Krishna, and Jiasen Lu. One diffusion to generate them all. arXiv preprint arXiv:2411.16318, 2024.
- [33] Shufan Li, Harkanwar Singh, and Aditya Grover. Instructany2pix: Flexible visual editing via multimodal instruction following. arXiv preprint arXiv:2312.06738, 2023.
- [34] Sijia Li, Chen Chen, and Haonan Lu. Moecontroller: Instruction-based arbitrary image manipulation with mixture-of-expert controllers. arXiv preprint arXiv:2309.04372, 2023.
- [35] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22511–22521, 2023.
- [36] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models, 2024.
- [37] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [38] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling, 2025.
- [39] Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner. arXiv preprint arXiv:2306.06101, 2023.
- [40] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 4296–4304, 2024.

- [41] Thanh Tam Nguyen, Zhao Ren, Trinh Pham, Thanh Trung Huynh, Phi Le Nguyen, Hongzhi Yin, and Quoc Viet Hung Nguyen. Instruction-guided editing controls for images and multimedia: A survey in llm era. arXiv preprint arXiv:2411.09955, 2024.
- [42] Byong Mok Oh, Max Chen, Julie Dorsey, and Frédo Durand. Image-based modeling and photo editing. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 433–442, 2001.
- [43] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [44] Zexu Pan, Zhaojie Luo, Jichen Yang, and Haizhou Li. Multi-modal attention for speech emotion recognition. arXiv preprint arXiv:2009.04107, 2020.
- [45] Rishubh Parihar, VS Sachidanand, Sabariswaran Mani, Tejan Karmali, and R Venkatesh Babu. Precisecontrol: Enhancing text-to-image diffusion models with fine-grained attribute control. In European Conference on Computer Vision, pages 469–487. Springer, 2024.
- [46] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [47] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [48] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147, 2023.
- [49] Scott Reed and Nando De Freitas. Neural programmer-interpreters. arXiv preprint arXiv:1511.06279, 2015.
- [50] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks, 2024.
- [51] Jean-Francois Rivest, Pierre Soille, and Serge Beucher. Morphological gradients. Journal of Electronic Imaging, 2(4):326–336, 1993.
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2022.
- [54] Khairul Anuar Mat Said and Asral Bahari Jambek. Analysis of image processing using morphological erosion and dilation. In Journal of Physics: Conference Series, volume 2071, page 012033. IOP Publishing, 2021.
- [55] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.
- [56] Yichun Shi, Peng Wang, and Weilin Huang. Seededit: Align image re-generation to image editing. arXiv preprint arXiv:2411.06686, 2024.
- [57] Yujun Shi, Chuhui Xue, Jun Hao Liew, Jiachun Pan, Hanshu Yan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8839–8849, 2024.
- [58] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [59] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.

- [60] Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280, 2025.
- [61] Nikolaos Tsagkas, Jack Rome, Subramanian Ramamoorthy, Oisin Mac Aodha, and Chris Xiaoxuan Lu. Click to grasp: Zero-shot precise manipulation via visual diffusion descriptors. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 11610–11617. IEEE, 2024.
- [62] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Highresolution image synthesis and semantic manipulation with conditional gans. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.
- [63] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [64] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.
- [65] Bin Xia, Yuechen Zhang, Jingyao Li, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamomni: Unified image generation and editing. arXiv preprint arXiv:2412.17098, 2024.
- [66] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.
- [67] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt-free diffusion: Taking" text" out of text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8682–8692, 2024.
- [68] Shiyuan Yang, Xiaodong Chen, and Jing Liao. Uni-paint: A unified framework for multimodal image inpainting with pretrained diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 3190–3199, 2023.
- [69] Siwei Yang, Mude Hui, Bingchen Zhao, Yuyin Zhou, Nataniel Ruiz, and Cihang Xie. Complex-Edit: Cot-like instruction generation for complexity-controllable image editing benchmark, 2025.
- [70] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024.
- [71] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos, 2025.
- [72] Hong Zhang, Zhongjie Duan, Xingjun Wang, Yingda Chen, Yuze Zhao, and Yu Zhang. Nexus-gen: A unified model for image understanding, generation, and editing. arXiv preprint arXiv:2504.21356, 2025.
- [73] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.
- [74] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [75] Xinyu Zhang, Mengxue Kang, Fei Wei, Shuang Xu, Yuhe Liu, and Lin Ma. Tie: Revolutionizing text-based image editing for complex-prompt following and high-fidelity editing, 2024.
- [76] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer, 2025.
- [77] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer, 2025.
- [78] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.
- [79] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and KwanYee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:11127–11150, 2023.

## Technical Appendices and Supplementary Material

In this part, we provide additional algorithm illustration, implementation details, more comparison results, more visualization results, and more analysis and discussions of the proposed approach.

## A Algorithm Illustration

To better elaborate the details of the proposed IEAP, we provide an algorithmic illustration for the whole pipeline in Alg. 1.

Algorithm 1 IEAP: Image Editing As Programs Input:

- • I: input image path
- • T: original instruction
- • {RoI_Localization, RoI_Inpainting, ..., Global_Transformation}: editing primitives
- • cot_with_gpt(·): CoT prompt to GPT–4o
- • extract_instructions(·): parse CoT output
- • infer_with_DiT(op,·): invoke DiT for primitive op
- • roi_localization(I,instr): returns mask for region of interest
- • fusion(I1,I2): blends two intermediate outputs
- • layout_change(I,instr): compute geometric transform Output: final edited image I∗

- 1: uri ← encode_image_to_datauri(I)
- 2: (C,T ) ← cot_with_gpt(uri,T) ▷ Categories and instructions
- 3: I(0) ← I
- 4: for i = 1 to |C| do
- 5: cat ← C[i], instr ← T [i]
- 6: if cat ∈ {Add,Remove,Replace} then
- 7: M ← roi_localization(I(i−1),instr)
- 8: I′ ← infer_with_DiT(RoI Inpainting,M,instr)
- 9: I(i) ← I′
- 10: else if cat = Action Change then
- 11: M ← roi_localization(I(i−1),instr)
- 12: Ibg ← infer_with_DiT(RoI Inpainting,M,instr)
- 13: Iact ← infer_with_DiT(RoI Editing,I(i−1),instr)
- 14: I(i) ← infer_with_DiT(RoI Compositing,fusion(Ibg,Iact),instr)
- 15: else if cat ∈ {Move,Resize} then
- 16: M ← roi_localization(I(i−1),instr)
- 17: Ibg ← infer_with_DiT(RoI Inpainting,M,instr)
- 18: Ilc ← layout_change(I(i−1),instr)
- 19: I(i) ← infer_with_DiT(RoI Compositing,fusion(Ibg,Ilc),instr)
- 20: else if cat ∈ {Appearance Change,Background Change,
- 21: Color Change,Material Change,Expression Change} then
- 22: I(i) ← infer_with_DiT(RoI Editing,I(i−1),instr)
- 23: else if cat ∈ {Tone Transfer,Style Change} then
- 24: I(i) ← infer_with_DiT(Global Transformation,I(i−1),instr)
- 25: else
- 26: raise ValueError(“Invalid category: ”cat”)
- 27: end if
- 28: end for
- 29: return I(|C|)

## B Implementation Details

In this section, we present the prompts employed to leverage a VLM for CoT reasoning over complex instructions, providing further details on the layout-adjustment prompts.

Below are the detailed prompts used to invoke the VLM for the CoT process on complex instructions:

|Now you are an expert in image editing. Based on the given single image, what atomic image editing instructions should be if the user wants to {instruction}? Let’s think step by step. Atomic instructions include 13 categories as follows:<br><br>- Add: Introduce a new object, person, or element into the image, e.g.: add a car on the road<br>- Remove: Eliminate an existing object or element from the image, e.g.: remove the sofa in the image<br>- Color Change: Modify the color of a specific object, e.g.: change the color of the shoes to blue<br>- Material Change: Alter the surface material or texture of an object, e.g.: change the material of the sign like stone<br>- Action Change: Modify the pose or action of an instance, e.g.: change the action of the boy to raising hands<br>- Expression Change: Adjust the facial expression, e.g.: change the expression to smiling<br>- Replace: Substitute one object in the image with a different object, e.g.: replace the coffee with an apple<br>- Background Change: Change the background scene to another, e.g.: change the background into forest<br>- Appearance Change: Modify visual attributes such as patterns or accessories, e.g.: make the cup have a floral pattern<br>- Move: Change the spatial position of an object within the image, e.g.: move the plane to the left<br>- Resize: Adjust the scale or size of an object, e.g.: enlarge the clock<br>- Tone Transfer: Change the global atmosphere or lighting conditions, e.g.: change the weather to foggy, change the time to spring<br>- Style Change: Modify the entire image to adopt a different visual style, e.g.: make the style of the image to cartoon Respond *only* with a numbered list. Each line must begin with the category in square brackets, then the instruction. Please strictly follow the atomic categories. The operation (what) and the target (to what) are crystal clear. Do not split replace to add and remove. Always place [Tone Transfer] and [Style Change] instructions at the end of the list. For example:<br><br><br>1. [Add] add a car on the road<br>2. [Color Change] change the color of the shoes to blue<br>3. [Move] move the lamp to the left Do not include any extra text, explanations, JSON or markdown, just the list.<br>|
|---|

Below are the detailed prompts used to adjust the layout of move and resize operations:

|You are an intelligent bounding box editor. I will provide you with the current bounding boxes and the editing instruction. Your task is to generate the new bounding boxes after editing. Let’s think step by step. The images are of size 512x512. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [512, 512]. The bounding boxes should not overlap or go beyond the image boundaries. Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, bottom-right x coordinate, bottom-right y coordinate]). Do not add new objects or delete any object provided in the bounding boxes. Do not change the size or the shape of any object unless the instruction requires so. Please consider the semantic information of the layout. When resizing, keep the bottom-left corner fixed by default. When swaping locations, change according to the center point. If needed, you can make reasonable guesses. Please refer to the examples below: Input bounding boxes: [("bed", [50, 300, 450, 450]), ("pillow", [200, 200, 300, 230])] Editing instruction: Move the pillow to the left side of the bed. Output bounding boxes: [("bed", [50, 300, 450, 450]), ("pillow", [70, 270, 170, 300])]|
|---|

|Editing instruction: Input bounding boxes: [(’a car’, [21, 281, 232, 440])] Editing instruction: Move the car to the right. Output bounding boxes: [(’a car’, [121, 281, 332, 440])] Input bounding boxes: [("dog", [150, 250, 250, 300])] Editing instruction: Enlarge the dog. Output bounding boxes: [("dog", [150, 225, 300, 300])] Input bounding boxes: [("chair", [100, 350, 200, 450]), ("lamp", [300, 200, 360, 300])] Editing instruction: Swap the location of the chair and the lamp. Output bounding boxes: [("chair", [280, 200, 380, 300]), ("lamp", [120, 350, 180, 450])] Now, the current bounding boxes is {bbox}, the instruction is {instruction}.|
|---|

Below are the detailed prompts used to adjust the layout of add operations:

|You are an intelligent bounding box editor. I will provide you with the current bounding boxes and an add editing instruction. Your task is to determine the new bounding box of the added object. Let’s think step by step. The images are of size 512x512. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [512, 512]. The bounding boxes should not go beyond the image boundaries. The new box must be at least as large as needed to encompass the object. Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, bottom-right x coordinate, bottom-right y coordinate]). Do not delete any object provided in the bounding boxes. Please consider the semantic information of the layout, preserve semantic relations. If needed, you can make reasonable guesses. Please refer to the examples below: Input bounding boxes: [(’a green car’, [21, 281, 232, 440])] Editing instruction: Add a bird on the green car. Output bounding boxes: [(’a bird’, [80, 150, 180, 281])] Input bounding boxes: [(’stool’, [300, 350, 380, 450])] Editing instruction: Add a cat to the left of the stool. Output bounding boxes: [(’a cat’, [180, 250, 300, 450])] Here are some examples to illustrate appropriate overlapping for better visual effects: Input bounding boxes: [(’the white cat’, [200, 300, 320, 420])] Editing instruction: Add a hat on the white cat. Output bounding boxes: [(’a hat’, [200, 150, 320, 330])] Now, the current bounding boxes is {bbox}, the instruction is {instruction}.|
|---|

## C More Quantitative Results

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.847 0.264 0.092 0.829 4.50 4.40 4.26 4.39 MagicBrush 0.889 0.277 0.068 0.892 4.66 4.76 4.62 4.68 UltraEdit 0.897 0.274 0.056 0.909 3.36 4.24 4.22 3.94 ICEdit 0.925 0.277 0.057 0.915 4.60 4.80 4.76 4.72

IEAP(Ours) 0.928 0.278 0.056 0.917 4.68 4.84 4.60 4.71

- Table 4: Quantitative comparison results on AnyEdit Add test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.800 0.202 0.108 0.721 2.74 3.42 3.20 3.12 MagicBrush 0.853 0.211 0.083 0.800 3.08 3.60 3.18 3.29

- UltraEdit 0.846 0.211 0.066 0.802 2.50 3.54 3.44 3.16 ICEdit 0.895 0.212 0.054 0.875 4.06 4.48 4.32 4.29 IEAP(Ours) 0.916 0.230 0.057 0.886 4.18 3.88 3.66 3.91

- Table 5: Quantitative comparison results on AnyEdit Remove test set.

InstructPix2Pix 0.766 0.234 0.179 0.588 3.72 3.68 3.80 3.73 MagicBrush 0.806 0.248 0.148 0.671 4.52 4.48 4.38 4.46 UltraEdit 0.779 0.242 0.142 0.621 3.80 4.40 4.40 4.20 ICEdit 0.797 0.228 0.128 0.614 3.68 4.02 4.04 3.91

IEAP(Ours) 0.866 0.252 0.099 0.701 4.68 4.68 4.48 4.61

- Table 6: Quantitative comparison results on AnyEdit Replace test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.829 0.254 0.164 0.774 3.46 3.84 3.58 3.63 MagicBrush 0.831 0.266 0.156 0.784 2.96 4.28 4.28 3.84

- UltraEdit 0.847 0.259 0.157 0.781 2.92 4.22 4.24 3.79 ICEdit 0.827 0.255 0.152 0.745 2.68 4.04 4.04 3.59 IEAP(Ours) 0.848 0.267 0.154 0.798 4.66 4.86 4.68 4.73

Table 7: Quantitative comparison results on AnyEdit Action Change test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.881 0.219 0.127 0.771 3.82 4.44 4.36 4.21 MagicBrush 0.902 0.219 0.088 0.828 2.94 3.94 3.90 3.59 UltraEdit 0.923 0.211 0.074 0.867 3.48 4.40 4.40 4.09 ICEdit 0.944 0.213 0.063 0.868 3.28 4.64 4.30 4.07

IEAP(Ours) 0.963 0.223 0.058 0.903 3.88 4.44 4.38 4.23

- Table 8: Quantitative comparison results on AnyEdit Relation test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.831 0.241 0.124 0.746 2.94 3.56 3.62 3.37 MagicBrush 0.875 0.258 0.094 0.802 2.80 3.88 4.00 3.56 UltraEdit 0.908 0.262 0.073 0.889 3.22 4.38 4.38 4.00

- ICEdit 0.895 0.253 0.074 0.841 3.14 4.28 4.26 3.89 IEAP(Ours) 0.923 0.263 0.066 0.921 4.38 4.32 4.28 4.32

- Table 9: Quantitative comparison results on AnyEdit Resize test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.815 0.280 0.139 0.744 3.60 4.08 3.92 3.87 MagicBrush 0.852 0.294 0.094 0.815 3.96 4.32 3.98 4.09 UltraEdit 0.857 0.277 0.068 0.845 4.04 4.62 4.42 4.36 ICEdit 0.847 0.273 0.085 0.808 4.04 4.42 4.16 4.21

IEAP(Ours) 0.886 0.285 0.082 0.833 4.06 4.72 4.80 4.53

Table 10: Quantitative comparison results on AnyEdit Appearance test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.725 0.224 0.216 0.582 3.40 3.60 3.44 3.48 MagicBrush 0.746 0.230 0.228 0.567 4.58 4.38 4.46 4.47 UltraEdit 0.796 0.257 0.169 0.747 3.48 4.36 3.14 3.66 ICEdit 0.799 0.241 0.166 0.757 3.04 4.16 3.88 3.69

IEAP(Ours) 0.801 0.243 0.165 0.759 4.74 4.68 4.70 4.71

Table 11: Quantitative comparison results on AnyEdit Background Change test set.

InstructPix2Pix 0.886 0.279 0.120 0.876 3.60 4.40 4.00 4.00 MagicBrush 0.898 0.282 0.087 0.869 4.20 4.82 4.62 4.55 UltraEdit 0.890 0.280 0.065 0.87 3.80 4.40 4.20 4.13

- ICEdit 0.896 0.278 0.073 0.849 4.72 4.80 4.64 4.72 IEAP(Ours) 0.911 0.276 0.059 0.876 4.62 4.72 4.78 4.71

Table 12: Quantitative comparison results on AnyEdit Color Change test set.

Method CLIPim ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.776 0.068 0.936 3.74 4.60 4.30 4.21 MagicBrush 0.770 0.064 0.940 3.86 4.48 4.18 4.17 UltraEdit 0.699 0.073 0.907 3.14 4.10 3.80 3.68 ICEdit 0.796 0.065 0.943 3.16 4.60 4.30 4.02

IEAP(Ours) 0.882 0.052 0.945 4.34 4.72 4.50 4.52

- Table 13: Quantitative comparison results on Expression test set.

Method CLIPim ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.746 0.130 0.549 4.00 4.18 4.04 4.07 MagicBrush 0.778 0.110 0.621 3.36 4.06 3.84 3.75 UltraEdit 0.765 0.086 0.598 3.34 4.28 4.04 3.89 ICEdit 0.787 0.086 0.616 3.48 3.92 3.58 3.66

IEAP(Ours) 0.826 0.055 0.696 4.08 4.48 4.18 4.25

- Table 14: Quantitative comparison results on Material Change test set.

Method CLIPim ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.710 0.212 0.463 3.56 4.32 3.94 3.94 MagicBrush 0.692 0.214 0.440 3.12 4.64 4.00 3.92 UltraEdit 0.703 0.201 0.467 4.02 4.8 4.62 4.48 ICEdit 0.706 0.219 0.458 4.04 4.82 4.36 4.41

IEAP(Ours) 0.922 0.097 0.915 4.44 4.64 4.44 4.51

- Table 15: Quantitative comparison results on AnyEdit Style Change test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑ InstructPix2Pix 0.822 0.260 0.100 0.821 3.72 4.48 3.92 4.04

- MagicBrush 0.834 0.266 0.159 0.791 3.56 4.64 3.98 4.06 UltraEdit 0.804 0.268 0.201 0.767 4.12 4.62 4.26 4.33 ICEdit 0.812 0.260 0.157 0.748 4.06 4.88 4.56 4.50 IEAP(Ours) 0.868 0.268 0.116 0.843 4.44 4.64 4.44 4.51

Table 16: Quantitative comparison results on AnyEdit Tone Transfer test set.

Method CLIPim ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑ InstructPix2Pix 0.815 0.134 0.647 3.40 4.04 4.80 4.08

- MagicBrush 0.835 0.081 0.697 1.82 3.56 3.50 2.96 UltraEdit 0.833 0.066 0.756 2.58 4.02 4.02 3.54 ICEdit 0.906 0.042 0.842 2.98 4.40 3.40 3.59 IEAP(Ours) 0.908 0.056 0.794 3.42 4.48 4.46 4.12

- Table 17: Quantitative comparison results on AnyEdit Counting test set.

Method CLIPim ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.773 0.208 0.581 3.46 4.18 4.08 3.91 MagicBrush 0.806 0.174 0.631 2.98 3.88 4.04 3.63 UltraEdit 0.825 0.167 0.669 2.82 4.38 4.38 3.86 ICEdit 0.806 0.171 0.629 3.56 4.16 4.06 3.93

IEAP(Ours) 0.833 0.169 0.662 3.88 4.44 4.52 4.28

Table 18: Quantitative comparison results on AnyEdit Implicit Change test set.

Method CLIPim ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.887 0.111 0.858 4.30 4.50 4.30 4.37 MagicBrush 0.900 0.100 0.874 4.12 4.36 4.54 4.34 UltraEdit 0.922 0.077 0.911 3.24 4.4 4.36 4.00 ICEdit 0.898 0.079 0.864 4.16 4.46 4.20 4.27

IEAP(Ours) 0.938 0.084 0.925 4.18 4.56 4.38 4.37

Table 19: Quantitative comparison results on AnyEdit Move test set.

Method CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ GPTIF ↑ GPTFC ↑ GPTAQ ↑ GPTavg ↑

InstructPix2Pix 0.688 0.243 0.189 0.742 1.04 4.38 3.92 3.11 MagicBrush 0.680 0.255 0.156 0.786 1.02 4.48 4.10 3.20 UltraEdit 0.732 0.279 0.147 0.843 1.96 4.46 3.98 3.47 ICEdit 0.810 0.289 0.155 0.811 4.18 4.42 4.68 4.43

IEAP(Ours) 0.788 0.285 0.162 0.786 3.96 4.58 4.06 4.20

Table 20: Quantitative comparison results on AnyEdit Textual Change test set.

- D More Visualization Results In this section, we provide more visualization results, as shown below:

###### Input

###### Edited Results

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Add

Add a cat on the grass. Add a dog on the grass.

Add a laptop on the grass. Add a fish on the grass.

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Remove

Remove the hat. Remove the necklace. Remove the cat. Remove the clothes.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Replace

Replace the dog with a backpack. Replace the dog with a cat. Replace the dog with a cake. Replace the dog with flowers.

##### Figure 8: More Visualization Results.

###### Input

###### Edited Results

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Action Change

Change the action of the woman to dancing.

Change the action of the woman to squatting.

Change the action of the woman to standing.

Change the action of the woman to running.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Move

Move the bird to the left. Move the bird to the right. Move the bird upward. Move the bird downward.

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Resize

Zoom in the tiger. Minify the tiger. Enlarge the tiger to twice its original size

Minify the tiger to half its original size.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Appearance Change

Make the cat wear a scarf filled with butterflies

Make the cat wear a bow tie. Make the cat wear a hat and necklace.

Make the cat decorated in Christmas style.

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Background Change

Change the background to the forest.

Change the background to the beach.

Change the background to the mountain.

Change the background to the city street.

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Color Change

[Figure 291]

Change the color of the shirt to white.

Change the color of the hat to purple.

Change the color of the hair to red.

Change the color of the jeans to black.

##### Figure 9: More Visualization Results.

###### Input

###### Edited Results

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Material

Change the material of the car like stone.

Change the material of the car like styrofoam.

Change the material of the car like leather.

Change the material of the car like corduroy.

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Expression

Change the expression to not smiling.

Change the expression to smiling. Change the expression to surprised.

Change the appearance to old.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Tone Transfer

Change the weather to foggy. Change the weather to raining. Change the time to winter. Change the time to the evening.

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Style Change

Change the style to 8-bit. Change the style to clean line. Change the style to cartoon. Change the style to ink painting.

##### Figure 10: More Visualization Results.

[Figure 312]

[Figure 313]

- 1. [Appearance Change] make the cat have a floral pattern
- 2. [Add] add a vase on the chair
- 3. [Color Change] change the color of the spoon to yellow
- 4. [Remove] erase the pink decoration on the wall
- 5. [Replace] replace the biggest blue coffee cup with a cake
- 6. [Resize] zoom in the pink cup next to the cat
- 7. [Tone Transfer] change the time to the evening

- 1. [Background Change] Change the background to the forest.
- 2. [Appearance Change] Make the lady wear a white dress.
- 3. [Add] Add a fox beside the lady
- 4. [Tone Transfer] Change the time to autumn

[Figure 314]

[Figure 315]

[Figure 316]

CoT

CoT

[Figure 317]

Instruction: “Make the cat have a floral pattern, put a vase on the chair. Then, alter the color of the spoon to yellow and erase the pink decoration on the wall. Replace the biggest blue coffee cup with a cake and zoom in the pink cup next to the cat. Finally, change the time to the evening.”

Instruction: “What would it be like if we placed the lady in a forest during autumn, and she was wearing a white dress, with a fox beside her?”

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

Neural Program Interpreter

Neural Program Interpreter

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

RoI Localization RoI Inpainting RoI Editing RoI Compositioning Global Transformation

##### Figure 11: More Detailed Visualization Processes of the pipeline.

## E Analysis and Discussions

### E.1 Runtime Performance Analysis

We evaluate the time required for each atomic operation of IEAP on a single NVIDIA H100 GPU. Empirical measurements indicate that the RoI Localization stage requires approximately 3s to 5s per operation. Other editing primitives, including RoI Inpainting, RoI Editing, RoI Compositing, and Global Transformation, each consumes roughly 7s to 9s per operation.

Consequently, a complete multi-step edit involving k atomic operations exhibits a total latency of

Ttotal =

k

Ti with Ti =

i=1

3s to 5s, if operationi = RoI Localization, 7s to 9s, otherwise.

While this per-operation cost precludes real-time interactivity, it remains acceptable for batch-oriented workflows in digital content creation, scientific visualization, and other offline editing scenarios.

### E.2 Limitations and Future Work

Limitations. Despite its strengths, IEAP exhibits several limitations in handling dynamic scenes and complex physical interactions. First, the RoI compositing may introduce geometric distortions or texture discontinuities when editing highly dynamic or non-rigid content, such as motion-blurred instances, and fluid or smoke effects. For example, in the task of “changing the cat’s action to jumping,” in Fig. 6, the rapid motion of fur can produce blurred regions that fail to blend naturally with the background. Second, RoI compositing struggles to simulate physically consistent lighting effects in scenes with reflective or refractive surfaces, sometimes resulting in mismatched shadow directions and illumination conflicts between edited objects and their environments. For example, in the task of “change the action of the woman to dancing,” in Fig. 4, the shadows before and after editing remain the same, but the action of the woman has changed, so it is unnatural. Third, the DiT-based architecture and multi-stage atomic operations incur substantial inference latency for 5s to 9s per operation on a single H100 GPU, precluding real-time interactivity in applications such as AR/VR. Finally, the requirement for high-memory GPUs like NVIDIA H100 (80 GB) limits reproducibility for resource-constrained researchers, and multi-iteration editing can exacerbate image quality degradation over successive operations.

Future Work. As for future work, several avenues may be pursued to overcome the identified limitations. To begin with, physics-aware compositing techniques and motion-compensated inpainting could be explored to better accommodate dynamic blur and fluid effects, thereby ensuring seamless integration of non-rigid edits. Meanwhile, differentiable lighting models or neural rendering modules may be incorporated to enforce global illumination consistency, particularly in reflective and refractive contexts. On the performance front, model distillation, operation fusion, and sparse attention strategies could be investigated to reduce per-operation latency and facilitate interactive editing. To enhance accessibility, memory optimization and support for smaller-footprint architectures amenable to commodity GPUs may be implemented. Moreover, iterative refinement and error-correction mechanisms may be developed to mitigate quality degradation over successive editing steps. Furthermore, beyond still-image editing, an extension to video-based complex instruction editing could be considered, where temporal coherence and motion consistency present additional challenges and opportunities for dynamic, multi-step visual manipulation.

### E.3 Societal Impacts and Ethical Safeguards

Positive Societal Impacts. The proposed IEAP framework introduces a modular and interpretable approach to complex image editing, which holds significant potential to benefit a range of creative and technical domains. By decomposing high-level visual instructions into atomic operations, IEAP enables users to perform multi-step edits with enhanced precision and control. This capability is particularly valuable in digital content creation, advertising, and education, where fine-grained manipulation of visual content is often required. For example, IEAP’s ability to support structurally inconsistent modifications can streamline visual storytelling workflows or facilitate the generation of accurate scientific visualizations for publications and teaching materials. Furthermore, its potential extensions to fields such as medical imaging by enabling localized enhancement of diagnostic visuals,

and accessibility technology by generating descriptive visual representations for users with visual impairments, demonstrate the framework’s broader societal utility and interdisciplinary relevance.

Negative Societal Impacts and Ethical Safeguards. Despite its benefits, IEAP’s high-fidelity editing capabilities also introduce ethical risks, particularly in the domains of misinformation and privacy. The framework’s precision in altering visual content could be misused for the creation of deepfakes or manipulated images intended for disinformation, identity falsification, or reputational harm. Operations such as “Remove” or “Replace” could be exploited to tamper with sensitive or private imagery, potentially infringing on individual rights.

To address these concerns, the development and deployment of IEAP adhere to strict ethical standards. Specifically, safeguards include the implementation of data filtering pipelines, such as the use of GPT-4o-filtered subsets of AnyEdit and the compliance-oriented CelebHQ-FM dataset, to reduce harmful biases and content. Additionally, the modular nature of IEAP facilitates transparency and traceability in the editing process, supporting future content provenance systems designed to detect and flag manipulated media. All these safeguards jointly contribute to ongoing efforts in AI safety and accountability.

