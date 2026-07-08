# arXiv:2508.06905v3[cs.CV]26Aug2025

## MultiRef: Controllable Image Generation with Multiple Visual References

Dongping Chen†

Ruoxi Chen

Siyuan Wu

Zhejiang Wanli University Ningbo, China

University of Washington Seattle, USA

Huazhong University of Science and Technology Wuhan, China

Sinan Wang

Huazhong University of Science and Technology Wuhan, China

Shiyun Lang

Huazhong University of Science and Technology Wuhan, China

Peter Sushko

Allen Institute for AI Seattle, USA

Gaoyang Jiang

Huazhong University of Science and Technology Wuhan, China

Yao Wan

Huazhong University of Science and Technology Wuhan, China

Ranjay Krishna*

University of Washington Allen Institute for AI Seattle, USA

### Abstract

Visual designers naturally draw inspiration from multiple visual references, combining diverse elements and aesthetic principles to create artwork. However, current image generative frameworks predominantly rely on single-source inputs - either text prompts or individual reference images. In this paper, we focus on the task of controllable image generation using multiple visual references. We introduce MultiRef-bench, a rigorous evaluation framework comprising 990 synthetic and 1,000 real-world samples that require incorporating visual content from multiple reference images. The synthetic samples are synthetically generated through our data engine RefBlend, with 10 reference types and 33 reference combinations. Based on RefBlend, we further construct a dataset MultiRef containing 38k high-quality images to facilitate further research. Our experiments across three interleaved image-text models (i.e., OmniGen, ACE, and Show-o) and six agentic frameworks (e.g., ChatDiT and LLM + SD) reveal that even state-of-the-art systems struggle with multi-reference conditioning, with the best model OmniGen achieving only 66.6% in synthetic samples and 79.0% in real-world cases on average compared to the golden answer. These findings provide valuable directions for developing more flexible and human-like creative tools that can effectively integrate multiple sources of visual inspiration. The dataset is publicly available at: https://multiref.github.io/.

### CCS Concepts

#### • Computing methodologies → Artificial intelligence; • General and reference → Evaluation.

This work is licensed under a Creative Commons Attribution 4.0 International License. MM ’25, Dublin, Ireland

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3758292

### Keywords

Controllableimagegeneration,multi-images-to-image, unified models, Benchmark, Dataset

ACM Reference Format:

Ruoxi Chen, Dongping Chen†, Siyuan Wu, Sinan Wang, Shiyun Lang, Peter Sushko, Gaoyang Jiang, Yao Wan, and Ranjay Krishna*. 2025. MultiRef: Controllable Image Generation with Multiple Visual References. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 21 pages. https://doi.org/10.1145/3746027.3758292

### 1 Introduction

Digital artists and visual designers often create a new scene by blending elements from multiple source images: a color palette from a Monet painting, the architectural form of the Eiffel Tower from a photograph, and the texture from a hand-drawn sketch. Artists draw inspiration from multiple visual references, mixing diverse elements. This multi-reference creative process allows far more controllable image creation than relying on a single source of inspiration (Figure 1). However, current tools for this artistic process remain too primitive to be directly useful.

Despite this artistic need, today’s image generators predominantly rely on single-source conditioning—either a text prompt (i.e., text-to-image [12, 45]) or one reference image (i.e., image editing [31, 46], image translation [20, 54]) at a time. In essence, asking a modern image generative model to “paint a scene in the style of Van Gogh with the composition of a photograph” requires specific prompt engineering [18, 27] or sequential editing [26, 53]. Moreover, visual references may have inconsistent viewpoints, styles, or semantics, and merging them can produce contradictions (e.g., blending a daytime landscape with a night-time style reference). Existing approaches like ControlNet [62] excel at following one conditioning signal (i.e., edge map and depth), but they are not inherently designed to handle multiple different conditions at once.

∗Corresponding author; †Project leader

covering 10 reference types and 33 reference combinations, far surpassing any existing collection in both scale and complexity.

Text-to-image Image-to-image

[Figure 1]

[Figure 2]

"A sunset over Paris with impressionist style"

[Figure 3]

“Turn it into impressioni

[Figure 4]

[Figure 5]

We propose new protocols to evaluate the generations using our benchmark. We leverage rule-based (e.g., MSE for depth) and modelbased (e.g., ClipScore [21] for aesthetic) assessments for conditions that require precise evaluation (e.g., depth, mask and bbox) and fine-tuned MLLM-as-a-Judge [5] for semantic-level assessments (e.g., caption, sketch and semantic map) in both reference-following and overall quality with human-annotated scores.

st style"

Multi-reference Generation

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Monet color palette

Eiffel Tower

Hand drawn sketch

We evaluate three interleaved image-text generation models (e.g., OmniGen [56], ACE [19], Show-o [57]) and 6 agentic frameworks (e.g., ChatDiT [26], LLM [2, 16] + Diffusion [12, 45]). Experimental results reveal that even the most advanced “general-purpose” image generators today struggle with multi-reference conditioning. State-of-the-art diffusion and autoregressive models that claim to support arbitrary conditioning (e.g., recent unified models) often falter when actually confronted with multiple visual inputs. For instance, a model might capture the style of one reference image well but completely ignore the content from another subject reference. Quantitatively, we observe substantial performance gaps: the best existing model OmniGen achieves only 0.496 of the desired alignment score on multi-reference tasks, compared to its near-perfect performance on single-reference inputs. These results expose a clear weakness in current systems – despite their advertised flexibility, they are not truly equipped for multi-reference generation. By highlighting these shortcomings, our study provides valuable insights and direction for future research.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

“Replace Franklin with nachos on the dollar bill”

- Figure 1: Image generation conditioned on multiple visual references provides more controllable and creative digital art generation than single image or textual reference.

Additionally, naively adding more control inputs usually confuses the model, leading to jumbled or degraded outputs [63].

Given these limitations in handling multiple references, there is a growing need to benchmark current multi-reference generation models. From our investigation, most popular benchmarks in generative modeling focus on text-to-image alignment or singleimage editing. For example, IDEA-Bench [33] targets professional design scenarios but still typically deals with one reference at a time or sequential editing. Similarly, ACE [19] evaluates alignment with instructions but does not stress-test combining several images. No established benchmark yet examines models on truly multireference tasks for their integrating complexity, making it hard to quantify current research progress.

### 2 MultiRef-bench

To facilitate evaluation of multi-reference image generation models, we introduce MultiRef-bench, the first benchmark combining real-world examples and synthetic data through a dual-pipeline methodology. The first pipeline gathers authentic tasks from publicly available sources, while the second leverages computer vision techniques to generate diverse conditional features. MultiRefbench comprises 1,990 examples: 1,000 real-world tasks from Reddit’s r/PhotoshopRequest community, selected for its diverse editing tasks and active engagement, and 990 synthetic examples from MultiRef’s 38,076 samples generated using RefBlend — our framework producing diverse guidance signals including depth maps, bounding boxes, and art styles for comprehensive conditional generation scenarios. Statistics are shown in Figure 3.

In this paper, we introduce MultiRef-bench, a benchmark that rigorously evaluates multi-reference generation models with 1,000 real-world samples and 990 synthetic samples that are programmatically generated. Specifically, we compile challenging user requests from Reddit [51], where both references and ground truth images are real, to evaluate the image generation from multiple visual references. Our benchmark covers a spectrum of tasks, ranging from relatively straightforward scenarios—such as applying two independent references—to complex scenarios requiring simultaneous spatial and semantic alignment across multiple sources.

To address the scarcity of multi-reference image generation datasets, we develop a novel synthetic data engine, termed RefBlend, that efficiently creates diverse training samples. RefBlend first extracts various visual references (e.g., depth maps, Canny edge, object masks) from original images using state-of-the-art extraction models. These references are then organized into a compatibility graph structure, where nodes represent individual references and edges indicate which references can be combined without contradictions, enabling diverse and high-quality multi-reference to image samples at scale. This engine can generate synthetic samples by flexibly combining diverse reference modalities—e.g., a semantic map, human pose, and caption, each describing different aspects of the intended output—while treating the original image as the corresponding target. By controlling the data generation process, we automatically obtain rich ground-truth pairings of inputs and outputs. Finally, MultiRef-bench contains 990 synthetic samples

### 2.1 Real-World Queries Collection

To develop a robust benchmark for conditional image generation, we incorporatereal-worldtasksfrom Reddit’sr/PhotoshopRequest community, following RealEdit [51]. This platform provides authentic user-submitted editing requests requiring multiple input images. We collected 2,300 queries, selecting tasks that combine multiple images. Each datapoint includes input images, original instructions, and output images. Manual evaluation ensures data quality by verifying image necessity, instruction coherence, and output accuracy. When multiple outputs exist, annotators select the best based on clarity and instruction fidelity.

To handle noisy instructions and specify image references, we employ GPT-4o [40] to generate structured prompts with image

Conditional Modality Engine

Meta data

[Figure 15]

Conditional Modality Engine

Category balance

[Figure 16]

[Figure 17]

[Figure 18]

MultiRef: Controllable Image Generation with Multiple Visual References MM ’25, October 27–31, 2025, Dublin, Ireland

[Figure 19]

Reference Generation Engine

[Figure 20]

Original image

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

MultiRef

Alignment checking/ Data ﬁltering

[Figure 28]

[Figure 29]

[Figure 30]

Canny Semantic map

Sketch

Depth

[Figure 31]

{"References": [ "Subject", “Sketch”, "Caption"], "Enhance_prompts": [ "Create an image to uncover the intricate details of <subject_1>, Mirroring the design outlined by <sketch_image>. The image should follow the caption: A yellow rubber ducky on a grey surface."] "Image_path_mapping": { "subject_1": "/subject/duck_toy_01.jpg", "sketch_image": "/sketch/duck_toy_02.jpg"} }

"A yellow rubber ducky on a grey surface."

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Reference graph

Generated references

[Figure 36]

[Figure 37]

Subject

Bounding box

Mask

Caption

[Figure 38]

[Figure 39]

[Figure 40]

…

[Figure 41]

Step 1: Combined references

Step 2: Basic instruction

Step 3: Enhanced prompts

Generate an image, with a focus on <subject_1>, inspired by <sketch_image>, reﬂect the caption: A yellow rubber ducky on a grey surface

Create an imagec to uncover the intricate details of <subject_1>, Mirroring the design outlined by <sketch_image>.

"Subject"

"Sketch"

[Figure 42]

[Figure 43]

"Caption"

The image should follow the caption: A yellow rubber ducky on a grey surface.

(k=3)

Instruction Prompt Generation

#### Figure 2: An overview of RefBlend. It consists of reference generation (in yellow) and instruction prompt generation (in blue). First, various references (canny, depth, etc) are extracted from an original image. Then, a basic instruction prompt is formed from selected compatible references. Finally, the enhanced prompt is integrated with references to construct a sample.

- Table 1: Reference Compatibility Matrix. Rows and columns represent reference names. Yellow: Local Spatial Constraints. Green: Semantic Content Specification. Purple: Global Structural Guidance. Pink: Semantic Content Specification.

Bounding box Mask Pose Caption Subject Semantic map Depth Canny Sketch Art style

Bounding box - ➞ ➞ ✓ Mask - ➞ ➞ ✓ Pose ✓ ✓ - ✓ Caption ✓ ✓ ✓ - ✓ ✓ ✓ ✓ ✓ ✓ Subject ✓ ✓ - ✓ ✓ ✓ ✓ ✓ ✓ ✓ Semantic map ✓ ✓ - ✓ Depth ✓ ✓ - ✓ Canny ✓ ✓ - ✓ Sketch ✓ ✓ - ✓ Art style ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ -

✓ : possible, i.e., the combination of row and column is feasible but does not depend on each other. : impossible, i.e., the combination of row and column is invalid and cannot coexist. ➞: dependency, i.e., when the row is present, the corresponding column condition must also be met.

[Figure 44]

[Figure 45]

[Figure 46]

#### Figure 3: Left, Middle: Distribution analysis of textual content length and image count for synthetic and real-world parts. Right: Reference frequency in synthetic data.

reference tokens (e.g., <image1>). All generated instructions undergo manual review for clarity and consistency. Missing image references are manually corrected by annotators.

(1,000 examples) met our criteria. Each example comprises 2-6 input images, one structured instruction, and one golden output image. See Supplementary Material for more details.

We categorized datapoints using OmniEdit’s taxonomy [55]: Element Replacement, Element Addition, Style and Appearance Modifications, Spatial/Environment Modifications, and Attribute Transfer. Categorization was performed using GPT-4o. We balance the benchmark following the real-world distribution in the crawled raw user requests. After rigorous quality control, 45% of the collected data

### 2.2 RefBlend: The Synthetic Data Engine

To construct an extensive benchmark, we develop a dataset generation engine RefBlend that employs a four-step process to produce 38,076 samples across 34 reference combinations. The process

#### Table 2: Evaluation dimension and metrics of MultiRef-bench for synthetic multi-ref generation. Rule: Golden standard for evaluation criteria. Model: We leverage a fine-tuned MLLM-as-a-judge for human-aligned semantic references evaluation.

###### Evaluation Dimension Evaluation Aspect Evaluation Criteria Quantitative Metrics Rule Model

Image Quality Visual Fidelity FID ✔ Visual Attractiveness Aesthetic Appeal CLIP Aesthetic Scores ✔ Bounding Box Spatial Accuracy IoU ✔ Semantic Map Segmentation Accuracy IoU ✔ Mask Mask Alignmen t IoU ✔ Depth Map Depth Accuracy MSE ✔

General Quality

Reference Fidelity Canny Edge Edge Preservation MSE ✔ Sketch Structural Fidelity MSE ✔ Caption Text-Image Alignment CLIP Text-Image Score ✔ Pose ✽ Pose Accuracy mAP ✔ Subject Subject Consistency CLIP Image Score ✔ Art Style Style Consistency CLIP Image Score ✔

Instruction Following Instruction Adherence Instruction-Output Alignment - - ✔

✽ For pose, a single reference image may contain multiple instances (e.g., multiple poses merged in one reference image).

includes: (1) generating reference conditioning (bounding boxes, depth maps, etc.), (2) programmatically producing condition combinations based on compatibility rules, (3) aligning multiple references through text prompts, and (4) deploying filtering to eliminate low-quality results.

- Step 1: Generate Reference Conditions. Given an original image, RefBlend leverages recent advanced models (e.g. Grounded SAM2 [44], SAM2 [43], Depth Anything2 [59]), to synthesize a diverse set of conditioning inputs. These include canny edges, semantic maps, sketches, depth maps, bounding boxes, masks, poses, art styles and subjects, along with textual captions generated by GPT4o-mini [39]. These reference guidance types have proved themselves in controllable image generation in prior work [23, 42, 62, 63].

Our original images are sampled in a wide range from DreamBooth[46], CustomConcept101[29], Subjects200K[52], WikiArt[47], Human-Art [28], StyleBooth [20] and VITON-HD [10], which attach references about pose, subject, and art style within the dataset and for the diversity of metadata.

- Step 2: Combining References. References have compatibility constraints and dependencies. Some references are mutually exclusive, while others have specific dependencies that must be considered. We establish Reference Compatibility Rules that define valid combinations among different conditions, avoiding conflicts and redundancy. Based on it, we can obtain the reference compatibility matrix for better visualization (Table 1).

To ensure diversity and complexity within the dataset, we generate possible combinations of 2, 3, or 4 references per instruction while strictly adhering to compatibility rules.

Visual Reference Compatibility Rules. These rules define valid combinations among image references (style, depth, edges, segmentation), specifying compatible pairings, mutual exclusions, and dependencies. We establish three fundamental rules for references:

- • Mutual Exclusivity of Global Reference: Reference containing global information cannot be combined with each other, as this would result in information overlap.
- • Global-Local Information Incompatibility: References with local information cannot be combined with those containing global information to avoid redundancy.
- • Reference Dependencies: ➀ Universal Combinability: Style and caption can be combined with any other references; ➁ Semantic Context Requirement: Mask and bounding box references require semantic context through either subject or caption. These

spatial localization references need to be attached to specific objects or concepts.

- Step 3: Generating Instructions. Using the valid reference combinations generated in Step 2, we create two types of prompts: structured and enhanced. Structured prompts are generated using a template-based approach that maps each reference type to a standardized phrase. For example, a depth reference might use the placeholder “<depth_image>” with associated phrases such as “guided by the depth of <depth_image>.” Caption references are appended with simple introductory phrases like “following the caption:”. This method ensures that prompts are clear, consistent, and easy to parse.

To broaden the scope and realism of our dataset, we transform structured prompts into more diverse and natural instructions using GPT-4o [40]. By applying different personas from Persona Hub [15], we vary the language, tone, and style of the prompts while maintaining the reference structure and intended content. This process not only enriches the prompts with creative and contextually relevant variations but also challenges models with a wide range of linguistic expressions and scenarios. The enhanced prompts, when combined with the generated references, result in a robust and versatile dataset suitable for comprehensive model evaluation.

- Step 4: Filtering. After generating visual references, we apply a rule-based filter using metrics such as a confidence score threshold of 0.8 for the IoU (Intersection over Union) of semantic maps.

For more semantic-level visual references—such as subject, style, sketch, and canny—that do not provide confidence scores, we finetuned Qwen-2.5-VL-7B-Instruct as a scoring-based filter [5]. This model evaluates both the alignment between original images and generated references and their overall quality. Trained on 16,590 human-annotated scoring samples, the model achieves a 0.914 MAE and 0.642 Pearson correlation on a 1,750-sample test set, demonstrating performance comparable to strong proprietary models such as GPT-4o-mini. See Supplementary Materials for additional details.

### 2.3 Evaluation

Our approach combines rule-based and model-based metrics to provide a comprehensive assessment of reference following capabilities across diverse conditions. The evaluation dimension and

#### Table 3: Evaluating MLLM-as-a-Judge using Pearson similarity with cross-validated human-annotated ground truth. Human-Human shows the alignment between humans.

Realistic Synthetic IQ IF SF IQ IF SF

Model

Gemini-2.0-Flash 0.385 0.422 0.354 0.369 0.627 0.588 GPT-4o-mini 0.466 0.530 0.514 0.438 0.632 0.616 GPT-4o 0.432 0.624 0.613 0.406 0.668 0.659 Human-Human 0.589 0.665 0.571 0.629 0.721 0.694

[Figure 47]

- Figure 4: Real-world image generation conditioned on multiple image references. Most image generative models struggle with accurately following instructions and maintaining fi-

delity to source images. metrics of MultiRef-bench are shown in Table 2. All evaluation metrics are finally normalized to a [0, 1] range for consistency.

Reference Fidelity. It measures how accurately generated images preserve and incorporate the specific attributes, features, and characteristics from reference inputs. For the 10 reference types included in our benchmark, we employ specialized evaluation criteria and metrics tailored to each reference, then derive the overall fidelity score by averaging across all references involved. Notably, for aspects where rule-based metrics may not fully capture nuanced performance—particularly style consistency and subject fidelitywe supplement our evaluation with MLLM-as-a-Judge assessments by our finetuned model for complementary qualitative insights.

Image Quality. It assesses the visual quality and aesthetic appeal of generated images, independent of reference fidelity. To evaluate this dimension comprehensively, we employ two complementary metrics: FID [22] and CLIP aesthetic scores [48], to evaluate the image quality and creative aspects of the generated content.

Overall Assessment. We follow Chen et al. [9] leveraging MLLMas-a-Judge [39] to evaluate overall Image Quality (IQ), Instruction Following (IF), and Source fidelity (SF) in a holistic manner. We leverage GPT-4o-mini as our primary model for its superior alignment with human judgment shown in Table 3.

### 3 Experiments and Analysis 3.1 Experiment Setups

We conduct evaluations on three open-source unified image generation models: OmniGen [56], ACE [19], Show-o [57] 1. For ACE and Show-o, we implement multi-turn dialogues to enable image generation with multiple references, incorporating one reference image per conversational turn. Additionally, we evaluate six compositional settings that specifically leverage Gemini-2.0-Flash [16] and

1Due to computation limitation, we do not employ Emu2-Gen [49].

Claude-3.7-Sonnet [2] as preceptors,2 SD3 serves as the primary generator for dataset synthesis, with SD2.1 employed in ablation studies. See Supplementary Material for detailed configurations.

### 3.2 Empirical Results and Analysis

Compositional framework exceeds in image quality, while failing to maintain consistency on real-world cases. As shown in Figure 4, LLM+SD combinations achieve the highest image quality scores, with Claude + SD3.5 reaching 0.774, occasionally surpassing ground truth. However, all compositional frameworks consistently underperform in instruction following and source fidelity. While ground truth achieves 0.767 and 0.706 for IF and SF respectively, Claude + SD3.5 only reaches 0.589 and 0.462, indicating that a separated perceptor-generator architecture fundamentally compromises complex visual instruction execution.

Unified models struggled with generation quality and handling real-world images. Although unified models theoretically end-to-end advantage that contributes to maintaining consistency, they underperform in fidelity preservation. OmniGen’s performance in various metrics even approaches some compositional frameworks that generate images with state-of-the-art diffusion models, demonstrating its effectiveness in balancing quality with instruction adherence. However, all models still fall short when compared with the golden answer (created with professional software), highlighting significant room for improvement in real-world image generation scenarios.

Controllable image generation from multiple references is challenging. Even advanced models like ACE, despite strong performance in specific areas (Bbox: 0.219, Pose: 0.090), show substantial gaps in reference fidelity compared to Ground Truth. While unified end-to-end architectures offer greater potential than compositional frameworks, both struggle with complex reference combinations or image generation without captions, highlighting the need for improved generalization in multi-image generation.

Models show strong and varied preferences for reference formats. Figure 6 presents results from an ablation study investigating how different input formats for Bounding Box (BBox), Depth, and Mask conditions affect the generation performance of three models. There is no universally superior format that works best across all tested models. Instead, each model often exhibits a distinct preference. ACE and ChatDiT show more robust performance on the depth and mask format. For Depth MSE, ACE performs significantly better with “ori depth," whereas OmniGen and ChatDiT show slightly better or comparable performance with “color depth". Input order primarily influences specific conditional fidelities rather than global image quality. As shown in Table 5, switching the input order resulted in only minor FID improvements or no change for the models. However, this operation had more substantial and often model-specific impacts on adherence to particular conditions. For example, ACE’s depth error and sketch error both increased dramatically when the order was switched. These observations suggest that the sequence of processing conditions is more critical for controlling specific visual attributes than for overall image realism as measured by FID. The presence of captions

2Given that GPT-4o participated in most of our experiments, we select alternative models for these compositional settings to avoid bias.

#### Table 4: Comparison of model performance on the synthetic part. Although models perform well in overall assessment, they fail at generating images with multiple references.

Overall Assessment Image Quality Reference Fidelity

Model

IQ IF SF FID↓ Aesthetic↑ AVG↑ BBox↑ Semantic Map↑ Mask↑ Depth Map↓ Canny Edge↓ Sketch↓ Caption↑ Pose*↑ Subject↑ Art Style↑

Unified Model

Show-o 0.764 0.616 0.462 0.110 0.607 0.469 0.051 0.263 0.332 0.104 0.061 0.203 0.569 0.008 0.532 0.301 OmniGen 0.730 0.532 0.438 0.111 0.593 0.464 0.179 0.197 0.320 0.087 0.092 0.221 0.382 0.014 0.623 0.329 ACE 0.740 0.655 0.528 0.108 0.592 0.553 0.219 0.382 0.439 0.044 0.079 0.112 0.521 0.090 0.720 0.397

Compositional Framework

ChatDiT 0.811 0.713 0.574 0.100 0.559 0.512 0.128 0.176 0.393 0.088 0.065 0.207 0.543 0.018 0.855 0.369

- Claude + SD 2.1 0.812 0.726 0.572 0.114 0.612 0.488 0.174 0.132 0.292 0.203 0.080 0.230 0.547 0.005 0.817 0.424 Claude + SD 3 0.876 0.817 0.658 0.102 0.635 0.500 0.134 0.145 0.360 0.203 0.087 0.215 0.576 0.009 0.859 0.420

- Claude + SD 3.5 0.913 0.853 0.691 0.111 0.647 0.513 0.124 0.147 0.358 0.082 0.082 0.213 0.573 0.009 0.858 0.434

- Gemini + SD 2.1 0.791 0.708 0.547 0.113 0.615 0.477 0.161 0.133 0.255 0.202 0.092 0.239 0.550 0.003 0.791 0.406 Gemini + SD 3 0.856 0.804 0.639 0.103 0.635 0.507 0.141 0.135 0.368 0.083 0.121 0.216 0.581 0.008 0.840 0.414

- Gemini + SD 3.5 0.893 0.839 0.676 0.111 0.646 0.510 0.132 0.130 0.371 0.077 0.096 0.216 0.579 0.008 0.845 0.422 Ground Truth 0.842 0.803 0.668 0.108 0.617 0.709 0.410 0.772 0.893 0.000 0.000 0.000 0.584 0.149 0.869 0.417

#### Table 5: Ablation study on image order and caption removal using the subset of MultiRef-bench. Switch order: switch the image input order. w/o caption: delete the caption in input instructions.

Image Quality Reference Fidelity

Model Setting

FID↓ Aesthetic↑ BBox↑ Semantic Map↑ Mask↑ Depth↓ Canny↓ Sketch↓ Caption↑ Pose↑ Subject↑ Art Style↑

Original 0.114 0.588 0.267 0.272 0.273 0.062 0.098 0.216 0.014 0.193 0.735 0.565 Switch order 0.114 0.579 0.382 0.315 0.290 0.068 0.105 0.219 0.005 0.195 0.737 0.556 w/o caption 0.120 0.534 0.180 0.308 0.272 0.081 0.137 0.191 – 0.202 0.669 0.581

OmniGen

Original 0.114 0.597 0.326 0.296 0.311 0.037 0.089 0.120 0.089 0.191 0.715 0.552 Switch order 0.112 0.598 0.303 0.243 0.386 0.077 0.105 0.222 0.036 0.191 0.802 0.600 w/o caption 0.114 0.567 0.231 0.470 0.481 0.019 0.111 0.069 – 0.197 0.657 0.567

ACE

Original 0.107 0.560 0.147 0.160 0.261 0.098 0.065 0.227 0.022 0.194 0.818 0.541 Switch order 0.105 0.574 0.125 0.150 0.284 0.092 0.063 0.220 0.022 0.194 0.830 0.556 w/o caption 0.096 0.550 0.142 0.132 0.278 0.113 0.066 0.196 – 0.202 0.836 0.553

ChatDiT

Bbox Caption

Show-o OmniGen ChatDiT Claude+SD2.1 GT Subject Depth ACE ChatDiT Claude+SD2.1 Gemini+SD2.1 GT

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

A black-and-white cat with striking green eyes sits elegantly on a table, surrounded by colorful decor.

[Figure 60]

| |
|---|

Sketch Style ACE Gemini+SD2.1 Gemini+SD3 Gemini+SD3.5 GT Mask Caption ChatDiT GT

Show-o Claude+SD2.1 Claude+SD3

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

A majestic tiger with striking orange and black stripes prowls through a sunlit forest, surrounded

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

by lush greenery.

Semantic map Subject Caption Gemini+SD3.5

###### ACE Claude+SD3 GT Canny Caption

Style OmniGen ACE ChatDiT GT

A vibrant maroon backpack with

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

A serene Venetian canal scene, featuring boats, historic buildings, and a clock tower under a bright blue sky

colorful patches

rests on a rock, surrounded by a blurred mountainous landscape.

#### Figure 5: Images generated under multiple references. More are provided in the Supplementary Materials. GT: Ground Truth.

improves for all three models when captions are absent, with ACE showing a notable reduction in sketch error.

[Figure 86]

### 4 Conclusion

Our work presents the first investigation of image generation conditioned on multiple visual references. Through developing a sophisticated synthetic data engine, we have constructed MultiRef, a large-scale dataset for multi-reference image generation, from which we carefully curated a high-quality benchmark suite alongside a real-world application to MultiRef-bench. Our evaluation reveals that existing models face challenges when handling multireference generation tasks. These findings provide insights for developing models that can better handle multi-reference creative processes.

#### Figure 6: Ablation study on the impact of different reference formats. Ori/white bbox: bounding boxes drawn in black/white backgrounds. Ori/color depth: depth maps in grey/turbo style. Ori/color mask: masks in white/light colors.

also improves depth fidelity and aesthetic quality across all models. For instance, ACE’s depth error significantly increases when captions are removed. However, for semantic map fidelity, OmniGen and ACE perform better without captions. Similarly, sketch fidelity

### References

- [1] Namhyuk Ahn, Junsoo Lee, Chunggi Lee, Kunhee Kim, Daesik Kim, Seung-Hun Nam, and Kibeom Hong. 2024. Dreamstyler: Paint by style inversion with textto-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 674–681.
- [2] Anthropic. 2024. Claude 3.5: A Sonnet. https://www.anthropic.com/news/claude3-5-sonnet. Accessed: 2024-09-04.
- [3] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2023. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 843–852.
- [4] Caroline Chan, Frédo Durand, and Phillip Isola. 2022. Learning to generate line drawings that convey geometry and semantics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7915–7925.
- [5] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. 2024. Mllm-as-ajudge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning.
- [6] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. 2024. Subject-driven text-to-image generation via apprenticeship learning. Advances in Neural Information Processing Systems 36 (2024).
- [7] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao.

2024. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6593–6602.

- [8] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. 2024. UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics. arXiv preprint arXiv:2412.07774 (2024).
- [9] Zhaorun Chen, Yichao Du, Zichen Wen, Yiyang Zhou, Chenhang Cui, Zhenzhen Weng, Haoqin Tu, Chaoqi Wang, Zhengwei Tong, Qinglan Huang, et al. 2024. MJBench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation? arXiv preprint arXiv:2407.04842 (2024).
- [10] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. 2021. VITONHD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR).
- [11] Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, and Aleksander Holynski.

2023. Diffusion self-guidance for controllable image generation. Advances in Neural Information Processing Systems 36 (2023), 16222–16239.

- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Fortyfirst International Conference on Machine Learning.
- [13] Flux. 2024. Black Forest Labs. https://blackforestlabs.ai/.
- [14] Ziqi Gao, Weikai Huang, Jieyu Zhang, Aniruddha Kembhavi, and Ranjay Krishna.

2024. Generate Any Scene: Evaluating and Improving Text-to-Vision Generation with Scene Graph Programming. arXiv preprint arXiv:2412.08221 (2024).

- [15] Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. 2024. Scaling synthetic data creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094 (2024).
- [16] GeminiTeam. 2023. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805 [cs.CL]
- [17] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. 2023. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems 36 (2023), 52132–52152.
- [18] Ziyu Guo,Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. 2025. Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step. arXiv preprint arXiv:2501.13926

(2025).

- [19] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. 2024. ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer. arXiv preprint arXiv:2410.00086 (2024).
- [20] Zhen Han, Chaojie Mao, Zeyinzi Jiang, Yulin Pan, and Jingfeng Zhang. 2024. StyleBooth: Image Style Editing with Multimodal Instruction. arXiv preprint arXiv:2404.12154 (2024).
- [21] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021).
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30

(2017).

- [23] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, et al. 2024. InstructImagen: Image generation with multi-modal instruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4754–4763.

- [24] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. 2023. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 20406–20417.
- [25] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. 2023. T2icompbench: A comprehensive benchmark for open-world compositional text-toimage generation. Advances in Neural Information Processing Systems 36 (2023), 78723–78747.
- [26] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Chen Liang, Tong Shen, Han Zhang, Huanzhang Dou, Yu Liu, and Jingren Zhou. 2024. ChatDiT: A Training-Free Baseline for Task-Agnostic Free-Form Chatting with Diffusion Transformers. arXiv preprint arXiv:2412.12571 (2024).
- [27] Chengyou Jia, Changliang Xia, Zhuohang Dang, Weijia Wu, Hangwei Qian, and Minnan Luo. 2024. ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting. arXiv preprint arXiv:2411.17176 (2024).
- [28] Xuan Ju, Ailing Zeng, Jianan Wang, Qiang Xu, and Lei Zhang. 2023. Humanart: A versatile human-centric dataset bridging natural and artificial scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 618–629.
- [29] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu.

2023. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1931–1941.

- [30] Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. 2025. ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback. In European Conference on Computer Vision. Springer, 129–147.
- [31] Tianle Li, Max Ku, Cong Wei, and Wenhu Chen. 2023. Dreamedit: Subject-driven image editing. arXiv preprint arXiv:2306.12624 (2023).
- [32] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. 2023. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22511–22521.
- [33] Chen Liang, Lianghua Huang, Jingwu Fang, Huanzhang Dou, Wei Wang, ZhiFan Wu, Yupeng Shi, Junge Zhang, Xin Zhao, and Yu Liu. 2024. IDEA-Bench: How Far are Generative Models from Professional Designing? arXiv preprint arXiv:2412.11767 (2024).
- [34] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. 2024. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision. Springer, 366–384.
- [35] Xiaoyu Liu, Yuxiang Wei, Ming Liu, Xianhui Lin, Peiran Ren, Xuansong Xie, and Wangmeng Zuo. 2024. Smartcontrol: Enhancing controlnet for handling rough visual conditions. In European Conference on Computer Vision. Springer, 1–17.
- [36] Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li, and Bolei Zhou. 2024. Freecontrol: Training-free spatial control of any text-to-image diffusion model with any condition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7465–7475.
- [37] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 4296–4304.
- [38] Nithin Gopalakrishnan Nair, Jeya Maria Jose Valanarasu, and Vishal M Patel.

2024. Maxfusion: Plug&play multi-modal generation in text-to-image diffusion models. In European Conference on Computer Vision. Springer, 93–110.

- [39] OpenAI. 2024. GPT-4O Mini: Advancing Cost-Efficient Intelligence. https:// openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/. Accessed: 2024-09-04.
- [40] OpenAI. 2024. Hello GPT-4o. https://openai.com/index/hello-gpt-4o/ Accessed: 2024-06-06.
- [41] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei.

2023. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992 (2023).

- [42] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. 2024. UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild. Advances in Neural Information Processing Systems 36 (2024).
- [43] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. SAM 2: Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714 (2024). https://arxiv.org/abs/2408.00714
- [44] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. 2024. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159

(2024).

- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In

- Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [46] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22500–22510.
- [47] Babak Saleh and Ahmed Elgammal. 2015. Large-scale classification of fineart paintings: Learning the right metric on the right feature. arXiv preprint arXiv:1505.00855 (2015).
- [48] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems 35 (2022), 25278–25294.
- [49] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. 2024. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8871–8879.
- [50] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. 2024. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14398–14409.
- [51] Peter Sushko, Ayana Bharadwaj, Zhi Yang Lim, Vasily Ilin, Ben Caffee, Dongping Chen, Mohammadreza Salehi, Cheng-Yu Hsieh, and Ranjay Krishna. 2025. REALEDIT: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations. arXiv:2502.03629 [cs.CV] https://arxiv.org/abs/2502.03629
- [52] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang.

2024. OminiControl: Minimal and Universal Control for Diffusion Transformer. arXiv preprint arXiv:2411.15098 (2024).

- [53] Zhenyu Wang, Aoxue Li, Zhenguo Li, and Xihui Liu. 2024. Genartist: Multimodal llm as an agent for unified image generation and editing. Advances in Neural Information Processing Systems 37 (2024), 128374–128395.
- [54] Zhizhong Wang, Lei Zhao, and Wei Xing. 2023. Stylediffusion: Controllable disentangled style transfer via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7677–7689.
- [55] Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen.

2024. OmniEdit: Building Image Editing Generalist Models Through Specialist Supervision. arXiv:2411.07199 [cs.CV] https://arxiv.org/abs/2411.07199

- [56] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. 2024. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340 (2024).
- [57] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. 2024. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation. arXiv preprint arXiv:2408.12528 (2024).
- [58] Xingqian Xu, Zhangyang Wang, Gong Zhang, Kai Wang, and Humphrey Shi.

2023. Versatile diffusion: Text, images and variations all in one diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7754– 7765.

- [59] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024. Depth Anything V2. arXiv:2406.09414 (2024).
- [60] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, et al. 2023. Reco: Region-controlled text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14246–14255.
- [61] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. 2024. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36 (2024).
- [62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.
- [63] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. 2024. Uni-controlnet: All-in-one control to text-toimage diffusion models. Advances in Neural Information Processing Systems 36

(2024).

- [64] Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. 2023. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22490–22499.

### 5 Acknowledgments

We would like to thank Wanting Liang, Jieyu Zhang, Weikai Huang, and Zixian Ma for their insightful feedback and support.

### A Related Work

Controllable Image Generation. The emergence of controllable image generation has revolutionized artificial intelligence by enabling users to create images that precisely match their specified criteria, including composition [32, 60, 64], style [1, 54], and content elements [6, 7]. ControlNet [62] advanced this field by introducing spatially localized input conditions to pre-trained text-to-image diffusion models through efficient fine-tuning methods. Subsequent research [11, 30, 36, 37] has further enhanced image controllability by implementing additional customization layers and adaptive mechanisms, enabling more sophisticated and precise image generation processes.

Building upon these advancements, some work has studied universal guidance for image generation with diffusion models [3, 35, 38, 41, 42, 58, 63]. While early approaches often required complex, condition-specific adapters, a new generation of unified models has expanded possibilities by incorporating diverse input modalities to facilitate multi-modal controllable generation. These recent unified architectures support multiple visual features as conditions. Emu2-Gen [50] uses an autoregressive model to predict the next tokens and uses a separated diffusion model to generate images. Instruct-Imagen [23] unifies image generation tasks together using multi-modal instructions. ACE [19] introduces the condition unit designed specifically for multi-modal tasks. OmniGen [56] uses an LLM as initialization and jointly models text and images within a single model to achieve unified representations across different modalities. UniReal [8] treats image-level tasks as discontinuous video generation, enabling a wide range of image generation and editing capabilities. In parallel developments, ChatDit [26] employs a multi-agent system for general-purpose, and interactive visual generation.

Dataset for Controllable Generation. Recent controllable image generation models have succeeded largely due to extensive training datasets like MultiGen-20M [42], which spans nine tasks across five categories with condition-specific instructions, while X2I dataset [56] incorporates flexible multi-modal instructions yet these approaches still predominantly address single or dual conditions rather than complex, multi-reference combinations.

Previous work has established benchmarks for evaluating image generation, primarily focused on text-to-image quality and alignment [14, 17, 24, 25, 34] or image editing tasks [31, 49, 61]. Existing benchmarks like IDEA-Bench [33] and ACE benchmark [19] are limited in scope, with the former including images-to-image tasks but focusing primarily on editing operations like font transfer, while the latter only evaluates alignment with textual instructions—both failing to address complex scenarios involving multiple image references and their combinations.

### B Details of Collecting MultiRef

We provide further details on the collection of MultiRef. Our dataset contains 38,076 samples in total, where 990 samples are split into test set for evaluation. Using RefBlend, we generate more than 100k raw data, and finally gain 38k dataset of highquality after filtering. See Table 6 for detailed statistics and Figure

- 7 for reference distribution in MultiRef.

Table 6: Distribution of Combinations by Count and Percentage in MultiRef.

Combination Count (%) caption+depth+subject 2,580 6.78 caption+mask+subject 2,487 6.53 caption+depth 2,470 6.49 bbox+caption+subject 2,461 6.46 canny+caption+subject 2,455 6.45 caption+sketch+subject 2,422 6.36 canny+caption 2,413 6.34 caption+sketch 2,404 6.31 caption+semantic map+subject 2,125 5.58 caption+semantic map 2,009 5.28 bbox+caption 1,961 5.15 caption+mask 1,946 5.11 caption+pose+subject 1,918 5.04 depth+subject 1,003 2.63 caption+depth+style 700 1.84 canny+caption+style 652 1.71 caption+sketch+style 644 1.69 bbox+subject 578 1.52 bbox+caption+style 544 1.43 mask+subject 542 1.42 caption+mask+style 535 1.41 sketch+subject 508 1.33 canny+subject 490 1.29 caption+semantic map+style 480 1.26 semantic map+subject 463 1.22 caption+subject 257 0.67 pose+subject 248 0.65 caption+pose 215 0.56 canny+style 131 0.34 depth+style 131 0.34 sketch+style 124 0.33 semantic map+style 93 0.24 Total 38,076 100.00%

[Figure 87]

Figure 7: Reference Distribution in MultiRef.

### B.1 Reference Generation

We choose the guidance of image generation from prior work [23, 42, 62, 63], combining commonly used references, standardizing their names and adapting them to work with flexible input and output formats. Our final set of references includes edge maps (Canny), semantic maps, sketches, depth maps, bounding boxes, masks, poses, art styles and subjects, along with textual captions.

Bounding box. A bounding box is a small possible rectangular box that can completely enclose an object in an image, typically defined by the (x,y) coordinates of its top-left and bottom-right corners. We utilize phrase grounding in Grounded SAM2 [44] to identify and localize the main objects in a given image. The bounding box is visualized by drawing it on a black background of the same dimensions as the input image.

Mask. A mask is a binary image representation where the object of interest is separated from the background. It precisely outlines the shape and contour of the target object, creating a silhouette that exactly matches the object’s boundaries rather than using a rectangular bounding box. We use Grounded SAM2 to generate masks, with one object corresponding to one mask. The mask is typically visualized as a binary image, where the background is represented by black pixels (value 0), and the object mask is represented by white pixels (value 1).

Pose. A pose refers to the spatial arrangement of key body parts (such as head, shoulders, elbows, wrists, hips, knees, and ankles) in a human figure, typically represented as a skeleton structure with joints and connections. The pose reference is visualized on a black background, with colored joints and connections highlighting the body’s key positions and movements.

Semantic map. A semantic map, is a visual representation where each object class or semantic category is assigned a unique color or label, showing the location and boundaries of different semantic concepts in an image. We use AutomaticMaskGenerator in SAM2 [43] to generate the semantic map.

Depth map. A depth map is a grayscale image where each pixel’s intensity represents the distance between the camera and the corresponding point in the scene. Typically, lighter/brighter pixels indicate points closer to the camera while darker pixels represent points that are farther away, creating a visual representation of the scene’s 3D structure in a 2D format. We use Depth Anything V2 [59] to generate the depth map with default parameters.

Canny edge. A Canny edge map is a binary image that shows the boundaries and edges detected in an original image using the Canny edge detection algorithm. It identifies edges by looking for areas of rapid intensity change in the image, producing a clean, thin outline where white pixels represent detected edges against a black background. We use the Canny operation in OpenCV with thresholds in [100, 200].

Sketch. A sketch of an image is a simplified, line-based representation that captures the original image’s essential contours and structural elements using only black lines on a white background. It focuses on preserving the key visual information while removing details like color, texture, and shading, similar to a hand-drawn outline. We use the line drawing method by Chan et al. [4] to generate the sketch reference, with contour_style and resize_and_crop process.

Art style. An art style of an image refers to the distinctive visual aesthetic, technique, or artistic treatment applied to transform the original image into a specific artistic rendering - such as watercolor, oil painting, cartoon and impressionist.

Subject. A subject reference image provides the main content or subject matter that needs to be transformed or recreated. It serves as the primary visual input that specifies what object or subject should be generated in the new image while maintaining its key characteristics and identity.

Caption. A caption of an image is a concise textual description that explains what is shown in the image, often describing the key subjects, actions, or notable elements present in the visual content. We use GPT-4o-mini [39] to describe the input image with prompts as follows.

Generate image caption

System prompt: You are a helpful assistant that can analyze images and provide detailed descriptions. Here is the image: [INSERT_IMAGES]

For subject-related images: Describe this image in detail using no more than 20 words. Focus on the main subject in the image. Do not include any other unrelated information.

For other images: Describe this image in detail using no more than 20 words. Do not include any other unrelated information.

### B.2 Details of Metadata

Originalimagesusedfor referencegeneration are from seven datasets, as follows.

DreamBooth [46]. It is a collection of images used for finetuning text-to-image diffusion models for subject-driven generation. It includes 30 subjects from 15 different classes. Images of the subjects are usually captured in different conditions, environments, and under different angles. While DreamBooth offers subject references, it does not include art style or pose references.

Subjects200K [52]. It is a large-scale dataset containing 200,000 paired images. Each image pair maintains subject consistency while presenting variations in the scene context. The dataset does not include art style or pose references. We leverage subject references provided by the dataset itself.

CustomConcept101 [29]. It is a dataset consisting of 101 concepts with 3-15 images in each concept. The categories include toys, plushies, wearables, scenes, transport vehicles, furniture, home decor items, luggage, human faces, musical instruments, rare flowers, food items, pet animals. While it offers subject references, it does not include art style or pose references.

Human-Art [28]. It is a versatile human-centric dataset to bridge the gap between natural and artificial scenes. It includes twenty high-quality human scenes, including natural and artificial humans in both 2D representation and 3D representations. It includes 50,000 images in 20 scenarios, with annotations of human bounding box and human keypoints. From this dataset, we utilize

two subsets: 2D_virtual_human and real_human, containing 22,000 and 10,000 images, respectively. Specifically, 2D_virtual_human provides art style and pose references while real_human provides pose references. Additionally, we leverage the art style and pose annotations provided within the dataset.

WikiArt [47]. WikiArt contains art paintings from 195 different artists. The dataset has 42,129 images for training and 10,628 images for testing. It does not include the subject reference or pose reference. We use images that share the same style as the art style references.

StyleBooth [20]. It is a high-quality style editing dataset accepting 67 prompt formats and 217 diverse content prompts, ending up with 67 different styles and 217 images per style. We use images that share the same style as the art style references.

VITON-HD [10]. It is a high-resolution virtual try-on dataset consisting of 11,647 person images and 11,647 corresponding clothing images. Each sample contains a person image, a target clothing image, and a pose representation. We use images as subject and pose references.

### B.3 Prompt Template

For each reference, we generate 10 structured basic instructions, as shown below.

Basic instructions for Art Style

- • Inspired by the essence of ⟨style_image⟩, this reflects its distinctive flair
- • Crafted in the characteristic tone of ⟨style_image⟩
- • Modeled with the unique influence of ⟨style_image⟩
- • Echoing the artistic spirit of ⟨style_image⟩
- • Infused with the signature style of ⟨style_image⟩
- • Reflecting the aesthetic nuances of ⟨style_image⟩
- • A reinterpretation influenced by ⟨style_image⟩
- • Harmonizing with the thematic essence of ⟨style_image⟩
- • Inspired by and shaped in the vein of ⟨style_image⟩
- • Capturing the creative vision embodied by ⟨style_image⟩

Basic instructions for Sketch

- • Following the sketch of ⟨sketch_image⟩, this mirrors its essence.
- • Designed in alignment with the sketch of ⟨sketch_image⟩.
- • Echoing the framework drawn by ⟨sketch_image⟩.
- • Guided by the outline of ⟨sketch_image⟩, it retains its authenticity.
- • Reflecting the initial strokes of ⟨sketch_image⟩.
- • Infused with the skeletal form of ⟨sketch_image⟩.
- • Shaped under the influence of ⟨sketch_image⟩’s sketch.
- • Structured around the design of ⟨sketch_image⟩.
- • Capturing the structural integrity of ⟨sketch_image⟩.
- • Crafted to reflect the framework of ⟨sketch_image⟩.

Basic instructions for Depth

- • Following the depth of ⟨depth_image⟩, this delves into its essence.
- • Inspired by the dimensionality of ⟨depth_image⟩, it captures its core.
- • Reflecting the profound layers of ⟨depth_image⟩.
- • Echoing the spatial depth of ⟨depth_image⟩, it retains its integrity.
- • Infused with the visual perspective of ⟨depth_image⟩.
- • Guided by the textured depth of ⟨depth_image⟩.
- • Structured to align with the depths captured by ⟨depth_image⟩.
- • Modeled after the layered depth of ⟨depth_image⟩.
- • Harmonizing with the multi-dimensional feel of ⟨depth_image⟩.
- • Crafted to embrace the depth portrayed by ⟨depth_image⟩.

Basic instructions for Canny

- • Following the edge of ⟨canny_image⟩, this captures its sharpness.
- • Inspired by the contours of ⟨canny_image⟩, it traces its form.
- • Reflecting the defined edges of ⟨canny_image⟩.
- • Echoing the precision lines of ⟨canny_image⟩, it retains its clarity.
- • Infused with the sharp boundaries of ⟨canny_image⟩.
- • Guided by the linear features of ⟨canny_image⟩.
- • Structured to follow the contours highlighted by ⟨canny_image⟩.
- • Modeled after the crisp edges of ⟨canny_image⟩.
- • Harmonizing with the boundary lines of ⟨canny_image⟩.
- • Crafted to reflect the edge details of ⟨canny_image⟩.

Basic instructions for Semantic Map

- • Following the semantic map in ⟨semantic_image⟩, this aligns with its meaning.
- • Inspired by the structure of ⟨semantic_image⟩, it conveys its intent.
- • Reflecting the mapped semantics of ⟨semantic_image⟩.
- • Echoing the visual language of ⟨semantic_image⟩, it captures its essence.
- • Infused with the meaningful contours of ⟨semantic_image⟩.
- • Guided by the symbolic layout of ⟨semantic_image⟩.
- • Structured around the semantics depicted in ⟨semantic_image⟩.
- • Modeled to align with the conceptual map of ⟨semantic_image⟩.
- • Harmonizing with the thematic essence of ⟨semantic_image⟩.
- • Crafted to reflect the semantic details of ⟨semantic_image⟩.

Basic instructions for Bounding Box

- • Following the bounding box in ⟨bbox_image⟩, this outlines its structure.
- • Inspired by the box constraints of ⟨bbox_image⟩, it defines its scope.
- • Reflecting the encapsulated regions of ⟨bbox_image⟩.
- • Echoing the boundary lines of ⟨bbox_image⟩, it retains its precision.
- • Infused with the spatial framework of ⟨bbox_image⟩.
- • Guided by the rectangular limits of ⟨bbox_image⟩.
- • Structured to follow the defined areas in ⟨bbox_image⟩.
- • Modeled after the bounding parameters of ⟨bbox_image⟩.
- • Harmonizing with the enclosed regions of ⟨bbox_image⟩.
- • Crafted to reflect the boundary specifications of ⟨bbox_image⟩.

Basic instructions for Single Mask

- • Following the mask in ⟨mask_image⟩, this captures its shape.
- • Inspired by the masked outline of ⟨mask_image⟩, it defines its form.
- • Reflecting the contours covered by ⟨mask_image⟩.
- • Echoing the masked regions of ⟨mask_image⟩, it retains its detail.
- • Infused with the coverage specified by ⟨mask_image⟩.
- • Guided by the spatial coverage of ⟨mask_image⟩.
- • Structured to align with the masked features in ⟨mask_image⟩.
- • Modeled after the outlined mask of ⟨mask_image⟩.
- • Harmonizing with the masked boundaries of ⟨mask_image⟩.
- • Crafted to reflect the regions defined by the mask in ⟨mask_image⟩.

Basic instructions for Pose

- • Following the pose in ⟨pose_1⟩, this mirrors its stance.
- • Inspired by the posture captured in ⟨pose_1⟩, it reflects its form.
- • Reflecting the alignment depicted in ⟨pose_1⟩.
- • Echoing the position shown in ⟨pose_1⟩, it retains its essence.
- • Infused with the dynamic structure of ⟨pose_1⟩.
- • Guided by the articulated motion of ⟨pose_1⟩.
- • Structured around the pose outlined in ⟨pose_1⟩.
- • Modeled to replicate the position in ⟨pose_1⟩.
- • Harmonizing with the posture embodied in ⟨pose_1⟩.
- • Crafted to reflect the expressive pose of ⟨pose_1⟩.

- Basic instructions for Subject
- • featuring ⟨subject_1⟩.
- • showcasing ⟨subject_1⟩.
- • focusing on ⟨subject_1⟩.
- • while emphasizing ⟨subject_1⟩
- • with a focus on ⟨subject_1⟩.
- • centered on ⟨subject_1⟩.
- • highlighting ⟨subject_1⟩.
- • to better display ⟨subject_1⟩.
- • while emphasizing ⟨subject_1⟩.
- • to reveal finer details of ⟨subject_1⟩.

We use the prompt Diversity enhancement to write enhanced instructions, shown as below.

Diversity enhancement

You will adopt the persona of selected_persona. You will be given a text and your task is to rewrite and polish it in a more diverse and creative manner that reflects the persona’s style. Do not include any direct references to the persona itself. You may alter sentence structure, wording, and tone. Do not modify text enclosed in angle brackets ”. If there is a ’caption:’ section in the text, do not change anything following ’caption:’ Here is the text: basic_instruction Please provide the revised text directly without any additional commentary.

Table 7 illustrates a side-by-side comparison of the basic instructions and their enhanced versions used in our prompt design pipeline. Each basic instruction provides a concise description of the target visual generation goal, typically referencing a specific condition input (e.g., <bbox_image> or <mask_image>). The enhanced prompts further enrich this guidance by incorporating more descriptive verbs, clarifying intentions, and explicitly reinforcing alignment with auxiliary conditions or captions. Overall, the enhanced prompts are not only longer but also more expressive and instructive.

### B.4 Data Filtering

To evaluate the complex outputs of free-form image generation, we assess both image quality and reference alignment using scoringbased MLLM as Filter [5], which has gained widespread adoption in the field [33].

For each reference, the multimodal large language model examines both the original and generated images, evaluating alignment between them and assessing the quality of the generated reference (if applicable). The evaluation produces numerical scores on a 5point scale (1-5), following specific scoring rubrics detailed below. References with a score under 3 will be filtered in the checking process. Pose, subject, and art style references are manually verified, as they are provided by the dataset and contain minimal annotation errors.

Table 7: Basic Instructions and their enhanced prompts. # Basic Instruction Enhanced Prompt

- 1 Generate an image featuring <subject_1>. Reflecting the en-

capsulated regions of <bbox_image>.

Create an image showcasing <subject_1>, capturing the essence of the defined areas within <bbox_image>.

- 2 Generate an image following the edge of <canny_image>, capturing its sharpness. Harmonizing with the thematic essence of <style_image> following the caption: The image features a woman with a neatly styled hair bun, dressed in a simple, elegant garment.

Create an image that traces the contours of <canny_image>, highlighting its precision. It should resonate with the thematic core of <style_image> while adhering to the caption: The image features a woman with a neatly styled hair bun, dressed in a simple, elegant garment.

- 3 Generate an image crafted to reflect the regions defined by the mask in <mask_image>. Following the caption: A woman in a pink dress stands confidently at a vibrant outdoor market, surrounded by colorful produce and stalls.

Generate an image designed to capture the essence of the areas delineated by the mask in <mask_image>, following the caption: A woman in a pink dress stands confidently at a vibrant outdoor market, surrounded by colorful produce and stalls.

Alignment: bbox

Alignment: canny

Alignment: caption

Alignment: depth

Alignment: pose

Alignment: semantic_map

60

60

60

60

60

60

| |364| | | | | |
|---|---|---|---|---|---|---|
| |(45.5%)| | | | | |
| | | | | | | |
| |160 (20.0%)<br><br>110| | | | | |
| |60 (7.5%)<br><br>106 (13.2%)(13.8%)| | | | | |
| | | | | | | |
| | | | | | | |

| |362| | | | | |
|---|---|---|---|---|---|---|
| |(45.4%)| | | | | |
| | | | | | | |
| |124<br><br>162 (20.3%)| | | | | |
| |52<br><br>98 (12.3%)<br><br>(15.5%)| | | | | |
| |(6.5%)| | | | | |
| | | | | | | |

| |804 (50.4%)| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| |312 (19.6%)<br><br>346 (21.7%)| | | | | |
| |114 (7.2%)| | | | | |
| |18 (1.1%)| | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |674 (42.6%)| | | | | |
| | | | | | | |
| |336 (21.2%)<br><br>362 (22.9%)| | | | | |
| |54 (3.4%)<br><br>156 (9.9%)| | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |604 (37.8%)| | | | | |
| | | | | | | |
| |334 (20.9%)<br><br>352 (22.0%)| | | | | |
| |108<br><br>200 (12.5%)| | | | | |
| |(6.8%)| | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |220 (27.6%)| | | | | |
| |158 (19.8%)<br><br>154 (19.3%) 132 134| | | | | |
| |(16.5%)(16.8%)| | | | | |
| | | | | | | |
| | | | | | | |

50

50

50

50

50

50

40

40

40

40

40

40

Percentage

Percentage

Percentage

Percentage

Percentage

Percentage

30

30

30

30

30

30

20

20

20

20

20

20

10

10

10

10

10

10

0

0

0

0

0

0

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

Score (1-5)

Score (1-5)

Score (1-5)

Score (1-5)

Score (1-5)

Score (1-5)

Alignment: single_mask

Alignment: sketch

Quality: sketch

Quality: canny

Quality: semantic_map

Quality: depth

428

60

60

60

60

60

60

| |(53.6%)| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| |144 (18.0%)| | | | | |
| |52<br><br>72 (9.0%)<br><br>102 (12.8%)| | | | | |
| |(6.5%)| | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |596 (37.6%)| | | | | |
| | | | | | | |
| |340 (21.4%)<br><br>320 (20.2%)| | | | | |
| |126 (7.9%)<br><br>204 (12.9%)| | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |546| | | | | |
| |448 (28.3%)<br><br>(34.5%)| | | | | |
| |230| | | | | |
| |212 (13.4%)(14.5%)<br><br>146 (9.2%)| | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |342 (43.0%)| | | | | |
| | | | | | | |
| |118<br><br>148 (18.6%) 122| | | | | |
| |(14.8%) (15.3%)<br><br>66 (8.3%)| | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |216 208| | | | | |
| |(27.0%)<br><br>128<br><br>144 (18.0%)<br><br>(26.0%)| | | | | |
| |(16.0%) 104 (13.0%)| | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |390<br><br>516 (32.6%)| | | | | |
| |(24.7%) 310 (19.6%)| | | | | |
| |202 (12.8%) 164 (10.4%)| | | | | |
| | | | | | | |
| | | | | | | |

50

50

50

50

50

50

40

40

40

40

40

40

Percentage

Percentage

Percentage

Percentage

Percentage

Percentage

30

30

30

30

30

30

20

20

20

20

20

20

10

10

10

10

10

10

0

0

0

0

0

0

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

Score (1-5)

Score (1-5)

Score (1-5)

Score (1-5)

Score (1-5)

Score (1-5)

Figure 8: Distribution of human preference scores (1-5) for image-text alignment and image quality across various conditioning types in the MLLM-as-a-Judge fine-tuning dataset. The top row shows alignment scores for conditions like BBox, Canny, Caption, Depth, Pose, and Semantic Map. The bottom row shows quality scores for conditions such as Single Mask, Sketch, Canny, Semantic Map, and Depth. Percentages indicate the proportion of samples receiving each score within that specific conditioning type.

For more semantic-level visual references, such as subject, style, sketch, canny, and edge that do not provide confidence scores, we establish a fine-tuned MLLM-as-a-Judge [5] that verifies both the alignment between original images and generated references and their quality. Specifically, we collect a subset with 6,400 original images and their corresponding references, constructing them into (image, reference) pairs and subsequently collect cross-validated humanscoring from 1-5foralignment and quality score individually. Finally, we split it into train/test sets, each with 16,590 and 1,750 samples, and fine-tune Qwen-2.5-7B-VL. Figure 8 illustrates the distribution of these scores across different conditioning types and assessment criteria (alignment and quality).

aspects: the effectiveness of fine-tuning MLLM-as-a-Judge (by comparing fine-tuned "-FT" versions of Qwen models to their zero-shot base versions) and benchmarking against VQA scores [34] from GPT-4o-mini.

In summary, the results in Table 8 robustly validate the finetuning strategy for the MLLM-as-a-Judge, with Qwen2-7B-VL-FT achieving the best overall alignment with human preferences. While this fine-tuned judge performs well across many conditions, particularly in comparison to a general VQA model like GPT-4o-mini, challenges persist in accurately judging highly nuanced spatial conditions such as pose and bounding box accuracy. The fine-tuned model with human-annotated scores as ground truth across Pearson similarity and MAE in Table 8 reveals close alignment with human annotators, validating it as a good judge for filtering.

- Table 8 presents a comprehensive performance comparison of

various models, evaluating their ability to align with human judgments across diverse condition types. The metrics used are Pearson correlation (higher indicates better alignment) and Mean Absolute Error (MAE, lower is better). This analysis focuses on two key

#### Table 8: Performance comparison across different models and condition types. ZS: zero-shot, FT: finetuned.

Overall Pose Subject Depth Caption Single Mask Style Sketch Semantic Map BBox Canny Pearson MAE Pearson MAE Pearson MAE Pearson MAE Pearson MAE Pearson MAE Pearson MAE Pearson MAE Pearson MAE Pearson MAE Pearson MAE

Model

Qwen2-2B-VL-ZS 0.122 1.550 0.028 1.596 0.231 1.475 0.044 1.631 0.864 0.491 0.051 1.987 0.173 2.671 0.239 1.446 0.209 1.338 -0.006 2.200 0.196 1.253 Qwen2-2B-VL-FT 0.602 0.998 0.102 1.513 0.787 0.633 0.695 0.818 0.874 0.462 0.308 1.068 0.473 0.975 0.617 1.025 0.497 1.461 0.093 1.177 0.270 1.228 Qwen2-7B-VL-ZS 0.364 1.397 0.104 1.603 0.694 0.892 0.148 1.757 0.869 0.515 0.495 1.757 0.293 1.171 0.270 1.515 0.618 1.117 0.118 2.588 0.378 1.309 Qwen2-7B-VL-FT (Ours) 0.642 0.914 0.257 1.397 0.838 0.557 0.635 0.894 0.865 0.485 0.605 0.797 0.629 0.728 0.634 0.946 0.605 1.162 0.404 1.129 0.402 1.124

GPT-4o-mini 0.424 1.307 0.272 1.789 0.759 0.779 0.367 1.592 0.782 0.580 0.390 1.662 0.457 0.949 0.490 1.290 0.404 1.188 0.046 2.541 0.197 1.167

Eval rubrics for canny Definitions:

- • Partially Aligned (Score 3) -Some parts of the caption correctly describe the image while other described elements are missing or different, or the caption captures the general scene but misses key elements.
- • Mostly Aligned (Score 4) - The caption describes most main elements and the overall scene with minor inaccuracies or omissions.
- • Well Aligned (Score 5) - The caption accurately describes the main elements and scene in the image.

Canny Edge Map is a visual representation that highlights the edges and contours of objects in an image, where white lines represent detected edges and black represents non-edge regions.

#### Alignment:

- • Not Aligned (Score 1) - Major object contours are unrecognizable or wrongly placed compared to the target image.
- • Minimally Aligned (Score 2) - Few contours match the target image, with significant placement issues.
- • Partially Aligned (Score 3) - Some major contours match while others are missing or misplaced.
- • Mostly Aligned (Score 4) - Most main contours are recognizable and properly placed with minor misalignments.
- • Well Aligned (Score 5) - Main object contours are recognizable and properly placed throughout the image.

Eval rubrics for sketch Definitions:

A sketch is a simplified, hand-drawn representation of an image, typically in black and white or grayscale, focusing on the main outlines and shapes of objects.

#### Alignment:

- • Not Aligned (Score 1) - The basic object or scene structure is not captured at all.
- • Minimally Aligned (Score 2) - Vague resemblance to the original image with major structural inaccuracies.
- • Partially Aligned (Score 3) -The main concept is recognizable but with significant structural deviations.
- • Mostly Aligned (Score 4) - Basic shapes and composition generally match with minor proportional variations.
- • Well Aligned (Score 5) - The basic shapes and composition match accurately to the original image.

#### Quality:

- • Poor Quality (Score 1) - Excessive noise or breaks prevent object recognition entirely.
- • Below Average Quality (Score 2) - Significant noise or breaks make most objects difficult to recognize.
- • Average Quality (Score 3) - Key objects are recognizable despite moderate noise or breaks in contours.
- • Good Quality (Score 4) - Main edges form clear object contours with minimal noise or breaks.
- • High Quality (Score 5) - Main edges form recognizable object contours with the appropriate level of detail.

#### Quality:

- • Poor Quality (Score 1) - Excessive noise or unclear lines make it difficult to interpret the intended subject.
- • Below Average Quality (Score 2) - Substantial noise or rough elements that significantly detract from the subject.
- • Average Quality (Score 3) -The sketch shows the subject but includes noticeable noise, scattered marks, or rough elements while maintaining recognizable forms.
- • Good Quality (Score 4) - Clear lines with minimal noise that effectively represent the subject.
- • High Quality (Score 5) - Clean, clear lines that effectively convey the subject with minimal noise or distraction.

Eval rubrics for caption

Definitions: Caption is a textual description that describes the content, context, objects, actions, or scene depicted in an image.

#### Alignment:

- • Not Aligned (Score 1) - The caption describes elements that aren’t present in the image, or fails to describe the main elements that are clearly visible.
- • Minimally Aligned (Score 2) - The caption has minimal connection to the image content, with only one or two elements correctly identified.

Eval rubrics for semantic map

Definitions: A semantic map is a visual representation where an image

is divided into distinct regions to represent different objects, areas, or elements of the scene, using any colors or styles to distinguish between regions.

Alignment:

- • Not Aligned (Score 1) - The basic scene structure or main objects are unrecognizable.
- • Minimally Aligned (Score 2) - Only a few elements are recognizable, with significant missing or misplaced components.
- • Partially Aligned (Score 3) - Some key elements are recognizable but others are missing or unclear.
- • Mostly Aligned (Score 4) - Most elements capture recognizable objects and scene layout with minor inaccuracies.
- • Well Aligned (Score 5) -The map captures recognizable objects and scene layout appropriately (simplified shapes are acceptable, textures and fine details not required).

#### Quality:

- • Poor Quality (Score 1) - Semantic regions are too sparse or scattered to identify main objects; regions are too minimal to understand scene content.
- • Below Average Quality (Score 2) - Main elements are barely distinguishable with significant noise, artifacts, or fragmented segments that impair understanding.
- • Average Quality (Score 3) - Main elements are clearly visible but with noticeable noise/artifacts or scattered segments, while still maintaining recognizable object shapes.
- • Good Quality (Score 4) - Key objects/regions are welldefined with limited noise or artifacts; segmentation is generally clean with only minor issues.
- • High Quality (Score 5) - Main objects/regions are clearly visible and distinguishable, with clean segmentation of major elements; minimal artifacts or noise around edges.

Eval rubrics for mask Definitions:

Mask Image is a binary image where white regions indicate areas of interest or target regions for object placement/generation, while black regions represent background or non-target areas.

#### Alignment:

- • Not Aligned (Score 1) - Main parts of the main object are not covered by the mask, or the mask position doesn’t correspond to the object location.
- • Minimally Aligned (Score 2) - The mask covers only a small portion of the main object or is significantly misplaced.
- • Partially Aligned (Score 3) - The mask covers most but not all of the main object, or if positioning is noticeably off.

- • Mostly Aligned (Score 4) - The mask covers the main object with minor positioning issues or slight shape inaccuracies.
- • Well Aligned (Score 5) - The mask captures the general outline and position of the main object accurately.

### B.5 Human Annotation

The annotation process was conducted by three independent evaluators: two authors of this paper and one volunteer. Recognizing that annotator diversity is essential for minimizing bias and maximizing dataset reliability, we selected annotators with varying demographic characteristics (gender, age, and educational background) while ensuring all possessed domain expertise in image generation evaluation.

To establish annotation consistency and objectivity, all evaluators underwent comprehensive training sessions before beginning the task. These sessions included detailed tutorials on objective image assessment techniques, familiarization with reference rubrics, and instruction on the specific criteria used in our Score Evaluation framework.

The annotation platform is shown in Figure 9.

[Figure 88]

Figure 9: Human Annotation Platform

C Details of MultiRef-bench

MultiRef-bench consists of 1,990 examples. The 1,000 examples from the real-world part represent real-world tasks sampled from the Reddit community r/PhotoshopRequest. The synthetic 990 examples are a test set split from MultiRef generated using RefBlend. We show some examples in Figure 10.

### C.1 Real-world

We collect 2,300 user queries from r/PhotoshopRequest community on Reddit, explicitly selecting tasks that require combining multiple input images to fulfill the requested edits. For each query, we gather all associated input images, the original text-based user instructions, and corresponding output images. To ensure data integrity and quality, each datapoint undergoes manual evaluation

Instruction: Edit image <image3> by replacing the face of the left caroler with the face from <image1> and the face of the right caroler with the face from <image2>

Instruction: Instruction: Edit image <image1> by placing the child from <image2> into the arms of the person, ensuring the child appears to be sitting naturally and is proportionate to the person holding them.

[Figure 89]

[Figure 90]

Sourced Ground Truth

Image 1 Image 2 Image 3

Sourced Ground Truth

Image 1 Image 2

Instruction: Create an image influenced by the symbolic arrangement of <semantic_image>. Drawing inspiration from the heart of <style_image>, this piece captures its unique charm beautifully.

Instruction: Craft an image that draws inspiration from the elegant posture embodied in <pose_image>, as it beautifully encapsulates the essence of movement. Follow the caption: Two people are dancing together.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Semantic map Style Ground Truth

Pose Ground Truth

Instruction: Create an image that intertwines with the skeletal structure of <sketch_image>, illuminating the intricate nuances of <subject_1>. following the caption: A simple, light-colored wooden table with four straight legs, placed on a wooden floor.

Instruction: Create a visual masterpiece inspired by the intricate nuances of <depth_image>, unveiling the subtle intricacies of <subject_1>, following the caption: A green bottle of Cascade detergent is prominently displayed

amidst neatly folded laundry in a cozy, domestic setting.

|[Figure 96]|
|---|

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Sketch Subject Ground Truth

Depth Subject Ground Truth

Figure 10: Samples in MultiRef-bench. The first row present real-world examples while others are from the synthetic part.

- Table 9: Distribution of examples across different categories in real-world samples.

Prompt of generating taxonomy for real-world queries

You are tasked with classifying image editing instructions into one of the following 5 categories:

Category Num. Element Replacement 529 Element Addition 246 Spatial/Environment Modifications 111 Attribute Transfer 73 Style and Appearance Modifications 41 Total 1,000

- 1. Element Replacement - Face swaps - Object substitutions Background replacements - Text replacements - Component swaps (wheels, screens, etc.)
- 2. Element Addition - Adding people to scenes - Adding objects to environments - Adding details or elements to objects - Adding text or graphics - Adding visual effects
- 3. Style and Appearance Modifications - Color adjustments

- Lighting modifications - Artistic style transfers - Texture changes - Visual quality enhancements

- 4. Spatial/Environment Manipulations - Repositioning elements - Combining multiple images into layouts - Changing scale or proportion - Adjusting orientation or alignment Creating composite images
- 5. Attribute Transfers - Transferring expressions between faces - Applying visual characteristics across images - Maintaining specific features while changing others - Matching visual properties (lighting, color) - Transferring specific details while preserving context

according to rigorous criteria. These criteria include verifying the necessity and appropriateness of each input image, assessing the logical coherence and relevance of instructions, and confirming accurate adherence to the instructions in the output image. In cases where multiple output images are provided for a single query, annotators select only one based on clarity, fidelity to the instruction, and overall quality.

Taxonomy creation. We adopted the taxonomy structure introduced in OmniEdit [55] to categorize the types of edits represented in our benchmark. We utilized GPT-4o with the following prompt to generate the taxonomy for our dataset. The distribution of edit types in the real-world part is shown in Table 9.

To produce the meta-style prompts from noisy user instructions, we used the prompt with GPT-4o. We supplied all input images, the corresponding output image, as well as the original user instructions.

#### Table 10: Evaluating MLLM-as-a-Judge in scoring with cross-validated human-annotated ground truth. GPT-4o and 4o-mini aligns closely with human scores in overall assessment. Human-Human shows the alignment between human annotators.

Image Quality Instruction Following Source Fidelity Pearson Spearman MSE MAE Pearson Spearman MSE MAE Pearson Spearman MSE MAE

Model

##### Realistic

Gemini-2.0-Flash 0.385 0.403 2.220 1.118 0.422 0.447 2.750 1.216 0.354 0.356 3.747 1.409 GPT-4o-mini 0.466 0.466 1.676 0.986 0.530 0.569 1.493 0.858 0.514 0.518 1.193 0.733 GPT-4o 0.432 0.420 2.486 1.223 0.624 0.616 1.405 0.764 0.613 0.513 1.216 0.736 Human-Human 0.589 0.573 1.611 0.936 0.665 0.590 1.152 0.720 0.571 0.441 1.473 0.824

##### Synthetic

Gemini-2.0-Flash 0.369 0.347 2.078 1.052 0.627 0.592 1.662 0.855 0.588 0.574 2.057 0.960 GPT-4o-mini 0.438 0.410 1.680 1.013 0.632 0.552 1.503 0.870 0.616 0.615 2.173 1.140 GPT-4o 0.406 0.374 2.350 1.083 0.668 0.608 1.537 0.843 0.659 0.626 1.573 0.860 Human-Human 0.629 0.648 1.823 0.930 0.721 0.735 1.820 0.867 0.694 0.708 1.840 0.840

ORIGINAL INSTRUCTION: {{description}}

Given the following image editing instruction, classify it into exactly one of these 5 categories. Respond with a JSON object with a single key "category" and the value being the category number (1-5).

Please provide a well-structured, clear editing prompt that precisely describes the transformation shown in the images.

### C.2 Synthetic

Following established compatibility rules, we identified 33 different reference combinations. To ensure statistical robustness while maintaining a manageable evaluation scope, we selected 30 samples for each combination, resulting in a comprehensive set of 990 evaluation instances in the synthetic portion of MultiRef-bench. Some examples are shown in row 2-3 in Figure 10.

GPT-4o prompt for rewriting instructions

You are an expert at image editing. Your job is to write a prompt that would help machine learning models to edit images.

I’m showing you:

- 1. First, the INPUT IMAGE(S) that the user wants to edit.
- 2. Then, the user’s ORIGINAL INSTRUCTION (which might be noisy or unclear).
- 3. Finally, the OUTPUT IMAGE after editing. Based on comparing these, please:

### C.3 Evaluation

For overall assessment, we leverage MLLM-as-a-Judge using GPT4o-mini. We validate the correlation of MLLM-as-a-Judge and human with a selected test set of 300 samples for either Realistic and Synthetic dataset. Our experiment in Table 10 reveals that GPT-4omini surpasses other models in aligning with humans. Therefore, we use GPT-4o-mini for overall assessment.

- 1. Infer what specific edit was performed
- 2. Write a clear, precise prompt that would help an AI model achieve this exact edit

### D Details of Experiment D.1 Model Settings

Your prompt should follow this format: "Edit image <image1> by [specific editing instruction using clear terminology]"

In this section, we will introduce the hyper-parameters of image generative models to facilitate experiment reproducibility and transparency. All our experiments were conducted on a server equipped with two A800 and two 4090 GPUs.

Here are some examples of good output prompts:

- - "Edit image <image1> by taking the person from <image2>, person from <image3> and adding them to <image1>."
- - "Edit image <image2> by transferring the background from <image1> and replacing the person with the person from <image3>"
- - "Edit image <image1> by faceswapping the person from <image2> into <image1>" Now, analyze the following:

Open-source Unified Models. We employed four open-source unified models. All hyper-parameters are detailed as follows:

- • OmniGen[56].We setheight=1024,width=1024,guidance_scale=2.5, img_guidance_scale=1.6, seed=0 as default settings.
- • ChatDit [26]. We use the images-to-image API call provided in the GitHub.
- • ACE [19]. We use the ACE-0.6B-512px as ACE-chat model for multi-reference image generation in multi-turn. We set sampler=’ddim’,sample_steps=20,guidance_scale=4.5, guide_rescale=0.5.

• Show-o [57]. We use multi-turn dialogue for multi-reference image generation. Wesetguidance_scale=1.75, generation timesteps=18, temperature=0.7, resolution: 256 × 256.

As reported in GitHub, Emu2-Gen [49] needs at least 75GB of memory. Due to the limitation of computation, it is not employed in our experiments.

Other Models. We utilize three proprietary models, GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro-latest as multimodal preceptors and Flux-dev, SD3, SD2.1 as image generators, with detailed settings as follows:

- • Gemini-1.5-pro-latest [16]. Temperature=1, top_p= 0.95.
- • Claude-3.5-Sonnet [2]. Temperature=0.9.
- • GPT-4o [40]. Temperature=1, top_p=1.
- • Flux1-dev [13]. guidance scale=3.5, num inference steps=50.
- • Stable Diffusion 3 [12]. guidance scale=7.0, num inference steps=28.
- • Stable Diffusion 2.1 [45]. guidance scale=7.5, num inference steps=25.

### E Additional Experiment Results

- E.1 Full Results We present full results of model generation in Table 11.
- E.2 More Visualizations

#### General Results.

We show images generated by models under a combination of two references and three references in Figure 11 and Figure 12, respectively. “GT" means ground truth image.

Figure 11 illustrates model performance when handling dual conditioning inputs. The results reveal significant variations in how different models interpret and integrate these multiple reference signals. Notably, most models perform poorly in maintaining the spatial accuracy of bounding boxes. For instance, ACE and ChatDiT inadequately follow sketch references, while compositional frameworks struggle with adherence to depth references. As an example, the Subject + Depth combination for flower generation, which tests the handling of both object appearance and spatial depth information, shows ACE maintaining depth relationships more effectively than other models. Encouragingly, all models demonstrate good performance in preserving object identity and art styles.

- Figure 12 extends this evaluation to three-reference scenarios,

introducing more complex conditioning challenges. In the Canny + Style combination for Venetian canal scenes, for example, OmniGen and ACE exhibit better preservation of architectural details, whereas other models struggle with structural consistency. Furthermore, ACE demonstrates superior performance in spatial scene composition compared to other models, as evidenced by the Semantic map + Subject + Caption combination.

#### Ablation study on reference formats.

To evaluate the effectiveness of different conditioning formats, we conduct comprehensive ablation studies on depth map, mask, and bounding box inputs. Figure 13 reveals distinct model behaviors under depth and mask conditioning. Depth-based conditioning (left panel) shows that models vary considerably in their ability to

respect spatial depth relationships, with ACE can generate plausible scenes while OmniGen and ChatDiT struggle with geometric consistency. Mask-based conditioning (right panel) demonstrates each model’s capacity for precise object placement and boundary adherence. ChatDiT shows more robust performance on different colors of mask references than others.

Furthermore, Figure 15a demonstrates that while all models can incorporate bounding box constraints, significant variations exist in spatial accuracy and semantic fidelity. ACE and ChatDiT exhibit different strengths in scene composition and detail preservation, but both perform poorly on spatial accuracy.

#### Ablation study on input image orders.

Figure 15b presents a grid of images demonstrating an ablation study on the effect of input order for different reference images. The results indicate that while all models can process combined visual inputs, the sequence in which these conditions are provided can noticeably influence the final output characteristics. For example, across OmniGen, ACE, and ChatDiT, variations in adherence to depth cues versus stylistic elements are observable when comparing “Depth+Style" to “Style+Depth" generations for the garlic scene. Similarly, the interplay between semantic maps and subject guidance, as shown in the bottom row of examples, yields distinct visual differences based on their ordering for all three models. This suggests that the sequential integration of features is not always commutative; the chosen order can be a significant factor in achieving the desired emphasis and balance between multiple visual constraints, affecting aspects like textural detail, spatial definition, and overall compositional fidelity.

#### Ablation study on captions.

Figure 14 highlights the significant impact of captions on the fidelity and semantic accuracy of generated images. When provided with a descriptive caption, models like OmniGen, ACE, and ChatDiT generally succeed in producing images that align well with the specified content and scene description, as demonstrated with the “modern gray leather sectional sofa" and the “vibrant green potted plant stands beside a cozy armchair" examples. However, in the absence of captions (“w/o caption"), the generated outputs often exhibit a marked degradation in quality and relevance across all tested models. These uncaptioned results can range from abstract or distorted representations of the intended subject, as seen with OmniGen and ACE in the sofa examples, to the generation of entirely unrelated subject matter, notably where ChatDiT produced an image of a person wearing a mask instead of a sofa when no caption was provided for that scene. This underscores the crucial role of captions in guiding the models towards specific, coherent, and contextually appropriate image synthesis.

Gemini+ SD 3.5 GT

Claude+ SD 2.1

###### Bbox Caption

Show-o OmniGen ACE ChatDiT

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

A black-and-white cat with striking green eyes sits elegantly on a table, surrounded by colorful decor.

[Figure 109]

| |
|---|

Claude+ SD 2.1

Claude+ SD 3.5

Gemini+ SD 2.1

Gemini+ SD 3 GT

Canny Style ACE ChatDiT

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Claude+

Gemini+

Gemini+ SD 3.5 GT

Subject Depth OmniGen ACE ChatDiT

SD 2.1

SD 2.1

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Claude+ SD 3.5 GT

Claude+ SD 3

###### Semantic map Caption

Show-o OmniGen ACE ChatDiT

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

The Petronas Towers rise majestically, illuminated against the night sky, flanked by modern skyscrapers and a bustling cityscape.

Gemini+ SD 3.5 GT

Claude+

Gemini+

Gemini+

Sketch Style ACE ChatDiT

SD 3.5

SD 2.1

SD 3

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

#### Figure 11: Image generation conditioned on a combination of two references.

Claude+ Style SD 2.1

Show-o OmniGen ACE ChatDiT GT Gemini+ SD 3

###### Canny Caption

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

A serene Venetian canal scene, featuring boats, historic buildings, and a clock tower under a bright blue sky

Claude+ SD 2.1

Gemini+ SD 2.1

Gemini+

###### Mask Subject Caption SD 3.5 GT

Show-o OmniGen ChatDiT

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

The image features a tube of Eucerin Advanced Repair Cream, showcasing its sleek design and clear labeling.

Gemini+ Caption OmniGen ACE ChatDiT SD 3.5

Gemini+ SD 2.1

Claude+

###### Pose Style SD 3.5 GT

A formally dressed man holds a card, wearing a black coat and bow tie, with a green drape in the background..

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Claude+ SD 3

Claude+ SD 2.1

Gemini+ SD 2.1

Gemini+

###### Semantic map Subject Caption SD 3.5 GT

ACE ChatDiT

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

A vibrant maroon backpack with colorful patches rests on a rock, surrounded by a blurred mountainous landscape.

#### Figure 12: Image generation conditioned on a combination of three references.

#### Table 11: Real-world image generation conditioned on multiple image references. Although today’s image generative models produce high-quality outputs, most struggle with accurately following instructions and maintaining fidelity to source images. IQ - Image Quality, IF - Instruction Following, SF - Source Fidelity.

Element Add. Spatial Mani. Element Rep. Attribute Tran. Style Modi. Overall

Model

IQ IF SF IQ IF SF IQ IF SF IQ IF SF IQ IF SF IQ IF SF

Unified Model

Show-o 0.511 0.290 0.253 0.525 0.300 0.258 0.508 0.268 0.240 0.548 0.301 0.260 0.473 0.307 0.259 0.513 0.293 0.254 OmniGen 0.553 0.498 0.429 0.553 0.461 0.422 0.484 0.450 0.379 0.567 0.479 0.408 0.620 0.590 0.468 0.555 0.496 0.421 ACE 0.254 0.207 0.205 0.260 0.207 0.205 0.255 0.207 0.203 0.234 0.200 0.200 0.265 0.205 0.200 0.254 0.205 0.203

###### Compositional Framework

ChatDiT 0.629 0.390 0.345 0.643 0.411 0.352 0.643 0.434 0.360 0.682 0.466 0.395 0.688 0.522 0.424 0.657 0.445 0.375 Gemini+SD2.1 0.611 0.372 0.329 0.620 0.404 0.324 0.574 0.391 0.339 0.605 0.397 0.332 0.660 0.495 0.385 0.614 0.412 0.342 Claude+SD2.1 0.620 0.402 0.330 0.625 0.416 0.339 0.555 0.371 0.322 0.674 0.419 0.345 0.717 0.507 0.390 0.638 0.423 0.345 Gemini+SD3 0.764 0.590 0.478 0.729 0.589 0.453 0.725 0.540 0.452 0.715 0.556 0.452 0.785 0.640 0.485 0.744 0.583 0.464 Claude+SD3 0.744 0.578 0.454 0.751 0.586 0.456 0.675 0.497 0.408 0.745 0.556 0.441 0.795 0.629 0.478 0.742 0.569 0.447 Gemini+SD3.5 0.786 0.615 0.500 0.756 0.591 0.473 0.759 0.558 0.459 0.789 0.564 0.441 0.780 0.610 0.460 0.774 0.588 0.467 Claude+SD3.5 0.767 0.563 0.469 0.777 0.598 0.472 0.700 0.506 0.406 0.789 0.625 0.466 0.790 0.654 0.498 0.765 0.589 0.462

Ground Truth 0.711 0.797 0.712 0.751 0.780 0.748 0.651 0.714 0.624 0.772 0.722 0.692 0.780 0.820 0.756 0.733 0.767 0.706

OmniGen ACE ChatDiT

Depth

OmniGen ACE ChatDiT

Mask

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

GT

Subject

GT

Subject

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Mask OmniGen ACE ChatDiT

Depth OmniGen ACE ChatDiT

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Caption

###### Caption

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

GT

GT

A majestic tiger with striking orange and black stripes prowls through a sunlit forest, surrounded by lush greenery.

A serene riverscape featuring a bridge, distant buildings, and a sailboat against a muted sky.

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

#### Figure 13: Ablation study on depth map and mask conditioning formats. Left: Depth-conditioned generation results comparing model performance. Right: Mask-conditioned generation result. All models receive the same conditioning inputs but demonstrate varying adherence to spatial and semantic constraints.

OmniGen ACE ChatDiT

Semantic map w/ caption

[Figure 219]

[Figure 220]

###### Style

Caption

[Figure 221]

A young boy in white clothing pulls

[Figure 222]

GT

[Figure 223]

[Figure 224]

on vines,

surrounded by a serene landscape with water and trees.

[Figure 225]

[Figure 226]

[Figure 227]

w/o

caption

OmniGen ACE ChatDiT

[Figure 228]

[Figure 229]

[Figure 230]

###### Mask Subject Caption

w/ caption

GT

A modern gray leather sectional sofa with decorative pillows, accompanied by a small wooden table on a hardwood floor..

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

w/o caption

OmniGen ACE ChatDiT

[Figure 237]

[Figure 238]

[Figure 239]

w/ caption

###### Canny Subject Caption

GT

[Figure 240]

[Figure 241]

[Figure 242]

A vibrant green potted

plant stands beside a cozy armchair, framed by sunlight streaming through large windows.

[Figure 243]

[Figure 244]

[Figure 245]

w/o

caption

- Figure 14: Ablation study on the impact of captions. This figure illustrates the differences in image generation when a text caption is provided ("w/ caption") versus when it is omitted ("w/o caption").

OmniGen ACE ChatDiT

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Depth+ GT Style

Depth

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Style+

OmniGen ACE ChatDiT

Bbox

Style

Depth

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

OmniGen ACE ChatDiT

Subject

| |
|---|

GT

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

|[Figure 263]|
|---|

[Figure 264]

Subject+

GT

Subject

|[Figure 265]|
|---|

[Figure 266]

Sketch

[Figure 267]

[Figure 268]

[Figure 269]

|[Figure 270]|
|---|

[Figure 271]

[Figure 272]

[Figure 273]

Sketch+ Subject

Sketch

Bbox OmniGen ACE ChatDiT Subject

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

###### Caption

GT

OmniGen ACE ChatDiT

An electric guitar stands upright on a rock, surrounded by vibrant foliage and majestic mountains in the background.

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Semantic map+ Subject

Semantic

GT

|[Figure 284]|
|---|

[Figure 285]

[Figure 286]

[Figure 287]

map

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Subject+ Semantic map

Subject

(a) Ablation study on bounding box conditioning formats. We compare different models using bounding boxes in black and white backgrounds for conditional image generation.

(b) Ablation study on the impact of input reference order.

#### Figure 15: Ablation studies: (a) bounding box conditioning formats; (b) impact of input reference order.

