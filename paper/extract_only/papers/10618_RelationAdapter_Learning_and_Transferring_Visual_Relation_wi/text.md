arXiv:2506.02528v1[cs.CV]3Jun2025

## RelationAdapter: Learning and Transferring Visual Relation with Diffusion Transformers

#### Yan Gong1 Yiren Song2 Yicheng Li1 Chenglin Li1 Yin Zhang1∗

1Zhe Jiang University 2National University of Singapore zhangyin98@zju.edu.cn

(a) Low Level Image Processing

(b) Style Transfer

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Input Result

Input Result

(c) Image Edit

(d) Customized Generation

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Input Result Input Result

- Figure 1: Our framework, RelationAdapter, can effectively perform a variety of image editing tasks by relying on exemplar image pairs and the original image. These tasks include (a) low-level editing, (b) style transfer, (c) image editing, and (d) customized generation.

### Abstract

Inspired by the in-context learning mechanism of large language models (LLMs), a new paradigm of generalizable visual prompt-based image editing is emerging. Existing single-reference methods typically focus on style or appearance adjustments and struggle with non-rigid transformations. To address these limitations, we propose leveraging source-target image pairs to extract and transfer content-aware editing intent to novel query images. To this end, we introduce RelationAdapter, a lightweight module that enables Diffusion Transformer (DiT) based models to effectively capture and apply visual transformations from minimal examples. We also introduce Relation252K, a comprehensive dataset comprising 218 diverse editing tasks, to evaluate model generalization and adaptability in visual prompt-driven scenarios. Experiments on Relation252K show that RelationAdapter significantly improves the model’s ability to understand and transfer editing intent, leading to notable gains in generation quality and overall editing performance. Project page: https://github.com/gy8888/RelationAdapter

### 1 Introduction

Humans excel at learning from examples. When presented with just a single pair of images, comprising an original and its edited counterpart, we can intuitively infer the underlying transformation and apply it to new, unseen instances. This paradigm, known as edit transfer or in-context visual learning

∗Corresponding author.

Preprint. Under review.

[6, 46, 22, 51], provides an intuitive and data-efficient solution for building flexible visual editing systems. Unlike instruction-based editing methods [14, 48, 19] that rely on textual prompts—where ambiguity and limited expressiveness can hinder precision—image pairs inherently encode rich, implicit visual semantics and transformation logic that are often difficult to articulate in language. By directly observing visual changes, models and users alike can grasp complex edits such as stylistic shifts, object modifications, or lighting adjustments with minimal supervision. As a result, this paradigm offers a highly intuitive and generalizable modality for a wide range of image manipulation tasks, from creative design to personalized photo retouching.

In-context learning-based methods [51, 46, 6, 22] have proven effective in extracting editing intent from image pairs. However, inputting image pairs into the model by concatenating them with the original image leads to several issues, including high memory consumption during inference and degraded performance of text prompts. To address these issues, we aim to develop a dedicated bypass module that can efficiently extract and inject editing intent from example image pairs, thereby facilitating image editing tasks. Nevertheless, building a scalable and general-purpose framework for image-pair-driven editing still presents several fundamental challenges: (1) accurately extracting visual transformation signals from a single image pair, including both semantic modifications (e.g., object appearance, style) and structural changes (e.g., spatial layout, geometry); (2) effectively applying these transformations to novel images while maintaining layout consistency and high visual fidelity; and (3) achieving strong generalization to unseen editing tasks—such as new styles or unseen compositional edits—without requiring retraining.

In this paper, we propose a unified framework composed of modular components that explicitly decouples the extraction of editing intent from the image generation process and enables more interpretable and controllable visual editing. First, we introduce RelationAdapter, a dual-branch adapter designed to explicitly model and encode visual relationships between the pre-edit and postedit images. It utilizes a shared vision encoder [32, 52] (e.g., SigLIP) to extract visual features, subsequently injecting these pairwise relational features into the Diffusion Transformer (DiT) [29] backbone to effectively capture and transfer complex edits. As a result, our framework robustly captures transferable edits across semantic, structural, and stylistic dimensions.

Second, we design an In-Context Editor that performs zero-shot image editing by integrating clean condition tokens with noisy query tokens. This mechanism enables the model to effectively align spatial structures and semantic intentions between the input and its edited version. A key innovation introduced in this method is positional encoding cloning, which explicitly establishes spatial correspondence by replicating positional encodings from condition tokens to target tokens, thus ensuring precise alignment during the editing process.

Third, to facilitate robust generalization across a wide range of visual editing scenarios [38, 17, 4], we construct a large-scale dataset comprising 218 diverse editing tasks. These scenarios span from low-level image processing to high-level semantic modifications, user-customized generation, and style-guided transformations. The dataset consists of 33K image pairs, which we further perform permutation to obtain a total of 252K training instances. This extensive and heterogeneous dataset improves the model’s generalization to unseen styles and edits.

Our main contributions are summarized as follows:

- • We propose RelationAdapter, the first DiT-based adapter module designed to extract visual transformations from paired images, enabling efficient conditional control for generating high-quality images with limited training samples.
- • We introduce In-Context Editor, a consistency-aware framework for high-fidelity, semantically aligned image editing with strong generalization to unseen tasks.
- • We establish a comprehensive benchmark dataset covering 218 task types with 251,580 training instances, addressing crucial gaps in task diversity, scale, and evaluation standards within image-pair editing research. This dataset provides a unified and scalable foundation for training and evaluating future image-pair editing models.

### 2 Related Work

- 2.1 Diffusion Models

Diffusion models have emerged as a dominant paradigm for high-fidelity image generation [34, 56, 57], image editing[24, 60, 58], video generation [40, 41, 45] and other applications [43, 37, 7, 44]. Foundational works such as Denoising Diffusion Probabilistic Models [15] and Stable Diffusion [34] established the effectiveness of denoising-based iterative generation. Building on this foundation, methods like SDEdit [24] and DreamBooth [35] introduced structure-preserving and personalized editing techniques. Recent advances have shifted from convolutional U-Net backbones to Transformerbased architectures, as exemplified by Diffusion Transformers (DiT) [29, 61] and FLUX [1]. DiT incorporates adaptive normalization and patch-wise attention to enhance global context modeling, while FLUX leverages large-scale training and flow-based objectives for improved sample fidelity and diversity. These developments signal a structural evolution in diffusion model design, paving the way for more controllable and scalable generation.

- 2.2 Controllable Generation

Controllability in diffusion models has attracted increasing attention, with various approaches enabling conditional guidance. ControlNet [55], T2I-Adapter [25], and MasaCtrl [5] inject external conditions—such as edges, poses, or style cues—into pretrained models without altering base weights. These zero-shot or plug-and-play methods offer flexibility in structure-aware generation. In parallel, layout- and skeleton-guided frameworks such as GLIGEN [21] and HumanSD [18] enable high-level spatial control. Fine-tuning-based strategies, including Concept Sliders [11] and Finestyle [53], learn attribute directions or attention maps to enable consistent manipulations. In the era of Diffusion Transformers, some methods concatenate condition tokens with denoised tokens and achieve controllable generation through bidirectional attention mechanisms or causal attention mechanisms [59, 12, 41, 17, 39, 42]. Despite their success, many of these methods rely on fixed condition formats or require significant training overhead.

- 2.3 Image Editing

Text-based and visual editing with diffusion models has seen rapid development. Prompt-to-Prompt [14] and InstructPix2Pix [4] allow fine-grained edits using prompt modifications or natural language instructions. Paint by Example [50] and LayerDiffusion [54] exploit visual references and layered generation to perform localized, high-quality edits. Versatile Diffusion [49] supports joint conditioning on text and image modalities, expanding the space of compositional control. Complementary to existing methods that often introduce a substantial number of additional parameters, our proposed RelationAdapter provides a lightweight yet effective solution that leverages DiT’s strong pretrained visual representation and structural modeling capacity, enabling few-shot generalization to novel and complex editing tasks. By injecting learned edit intent into DiT’s attention layers, our method supports fine-grained structural control and robust style preservation.

### 3 Methods

In this section, we present the overall architecture of our proposed methods in Section 3.1. Next, Section 3.2 outlines our RelationAdapter module, which serves as a visual prompt mechanism to effectively guide image generation. We then integrate the In-Context Editor module (Section 3.3) by incorporating the Low-Rank Adaptation (LoRA) [16] fine-tuning technique into our framework. Finally, Section 3.4 presents a novel dataset of 218 in-context image editing tasks to support a comprehensive evaluation and future research.

- 3.1 Overall Architecture As shown in Figure 2, our method consists of two main modules:

RelationAdapter. RelationAdapter is a lightweight module built on the DiT architecture. By embedding a novel attention processor in each DiT block, it captures visual transformations and

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Legends

Image Prompt Tokens

“Apply a crayon drawing style to the real-world image.”

Image Reference Tokens

Text Tokens

Image Souce Tokens

[Figure 13]

[Figure 14]

[Figure 15]

Noised Latent Tokens

[Figure 16]

[Figure 17]

Text Encoder

SigLip SigLip

VAE VAE

Position Embedding Frozen Learnable

|Linear<br><br>[Figure 18]| |
|---|---|
| | |

|Linear<br><br>[Figure 19]|
|---|

| | | | |
|---|---|---|---|
| | | | |

[Figure 20]

[Figure 21]

... ...

... ... ...

| | |
|---|---|
| | |

||KV Projector|
|---|
<br><br>K V Q<br><br>RA Attention<br><br>[Figure 22]|
|---|

[Figure 23]

###### Position Encoding

DiT Block DiT Block

MM Attention

[Figure 24]

|QKV Projector|
|---|

| | |
|---|---|
| | |

+

LoRA V

Clone

...

Q K

+

DiT Block

Relation Adapter

... ... ...

|Out Projector|
|---|

In-Context Editor

- Figure 2: The overall architecture and training paradigm of RelationAdapter. We employ the RelationAdapter to decouple inputs by injecting visual prompt features into the MMAttention module to control the generation process. Meanwhile, a high-rank LoRA is used to train the In-Context Editor on a large-scale dataset. During inference, the In-Context Editor encodes the source image into conditional tokens, concatenates them with noise-added latent tokens, and directs the generation via the MMAttention module.

injects them into the hidden states. This enhances the model’s relational reasoning over image pairs without modifying the core DiT structure.

In-Context Editor. In-Context Editor frames image editing as a conditional generation task during training. It jointly encodes the images and textual description, enabling bidirectional attention between the denoising and input branches. This facilitates precise, instruction-driven editing while preserving the pre-trained DiT architecture for compatibility and efficiency.

#### 3.2 RelationAdapter

Our method can be formulated as a function that maps a set of multimodal inputs, namely, a visual prompt image pair (Iprm,Iref), a source image Isrc, and a textual prompt Tprm to a post-edited image as a target image Itar:

Itar ≡ E(Iprm,Iref,Isrc,Tprm) ≡ D (R(Iprm,Iref),Isrc,Tprm) (1)

where D denotes the Diffusion Transformer, and R refers to the RelationAdapter module integrated into the Transformer encoder blocks of the DiT architecture.

Image Encoder. Most personalized generation methods use CLIP [32] as an image encoder, but its limited ability to preserve fine-grained visual details hinders high-fidelity customization. To overcome this, we adopt the SigLIP-SO400M-Patch14-384 [52] model for its superior semantic fidelity in extracting visual prompt features from paired visual prompts Iprm and Iref. Let cP and cR denote the representations of the sequence of features of Iprm and Iref, respectively. The visual prompt representation cV is constructed by concatenating cP and cR.

Revisiting Visual Prompt Integration. To enhance the representational flexibility of the DiT based model, we revisit the current mainstream image prompt based approaches (e.g., FLUX.1 Redux [3], which directly appends visual features to the output of the T5 encoder [23]).

Given the visual prompt features cV and the backbone DiT input features cB, FLUX.1 Redux applies a bidirectional self-attention mechanism over the concatenated feature sequence. The resulting attention output Z′ is computed as:

Z′ = Attention(Q,K,V) = Softmax

#### QK⊤

√

d

V (2)

Q = cB,V Wq, K = cB,V Wk, V = cB,V Wv (3) and cB,V denotes the concatenation of backbone DiT input features cB and visual features cV .

Decoupled Attention Injection. A key limitation of current approaches is that visual feature embeddings are typically much longer than textual prompts, which can weaken or even nullify text-based guidance. We design a separate key-value (KV) attention projection mechanism, Wk′ and Wv′ , for the visual prompt features. Crucially, the cross-attention layer for visual prompts shares the same query Q with the backbone DiT branch:

ZV = Attention(Q,K′,V′) = Softmax

#### Q(K′)⊤

√

d

V′ (4)

#### Q = cBWq, K′ = cV Wk′ , V′ = cV Wv′ (5)

Then, we fuse the visual attention output ZV (from the RelationAdapter) with the original DiT attention output ZB before passing it to the Output Projection module:

Znew = ZB + α · ZV (6) where α is a tunable scalar coefficient that controls the influence of visual prompt attention.

#### 3.3 In-Context Editor

In-Context Editor builds upon a DiT-based pretrained architecture, extending it into a robust incontext image editing framework. Both the source image Isrc and the target image Itar are encoded into latent representations, cS and z respectively, via a Variational Autoencoder (VAE) [20]. After cloning positional encodings, the latent tokens are concatenated along the sequence dimension to enable Multi-modal Attention [28], formulated as:

MMA([z;cS;cT]) = softmax

QK⊤ d

V (7)

Here, Z = [z;cS;cT] denotes the concatenation of noisy latent tokens z, source image tokens cS, and text tokens cT, where z is obtained by adding noise to target image tokens.

Position Encoding Cloning. Conventional conditional image editing models often struggle with pixel-level misalignment between source and target images, leading to structural distortions. To address this, we propose a Position Encoding Cloning strategy that explicitly embeds latent spatial correspondences into the generative process. Specifically, we enforce alignment between the positional encodings of the source condition representation cS and the noise variable z, establishing a consistent pixel-wise coordinate mapping throughout the diffusion process. By sharing positional encodings across key components, our approach provides robust spatial guidance, mitigating artifacts such as ghosting and misplacement. This enables the DiT to more effectively learn fine-grained correspondences, resulting in improved editing fidelity and greater theoretical consistency.

LoRA Fine-Tuning. To enhance the editing capabilities and adaptability of our framework to diverse data, we constructed a context learning–formatted editing dataset comprising 2,515,800 samples (see Section 3.4). We then applied LoRA fine-tuning to the DiT module for parameterefficient adaptation. Specifically, we employed high-rank LoRA by freezing the pre-trained weights W0 and injecting trainable low-rank matrices A ∈ Rr×k and B ∈ Rd×r into each model layer.

Noise-Free Paradigm for Conditional Image Features. Existing In-Context Editor frameworks concatenate the latent representations of source and target images as input to a step-wise denoising process. However, this often disrupts the source features, causing detail loss and reduced pixel fidelity.

To address this, we propose a noise-free paradigm that preserves the source features cS from Isrc throughout all denoising stages. By maintaining these features in a clean state, we provide a stable

and accurate reference for generating the target image Itar. Combined with position encoding cloning and a Multi-scale Modulation Attention (MMA) mechanism, this design enables precise, localized edits while minimizing unintended modifications.

#### 3.4 Relation252K Dataset

We present a large-scale image editing dataset encompassing 218 diverse tasks, categorized into four main groups based on functional characteristics: Low-Level Image Processing, Image Style Transfer, Image Editing, and Customized Generation. The dataset contains 33,274 images and 251,580 editing samples generated through image pair permutations. Figure 3 provides an overview of the four task categories. We will fully open-source the dataset to encourage widespread usage and further research in this field.

Low Level Style Transfer

Depth Surface Normal Origami Eﬀect

Van Gogh Style

...

...

Cloud Removal

Woodcut Eﬀect

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Image Editing

Customized Generation

Close Eyes Add Object

Lego Form

3D PhotoFrame

...

BallonMorphic ...

Raise Arms

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

- Figure 3: Overview of the four main task categories in our dataset. Each block lists representative sub-tasks (with ellipses indicating more), along with image-pair examples.

Source Caption: A person wearing a white shirt with rolledup sleeves, posing casually.

Target Caption: A person wearing a white shirt with buttoned sleeves against a red background.

Edit Instruction: Apply an ID photo transformation to the portrait, front-facing, red background.

[Figure 33]

Unlabeled Image Pair

[Figure 34]

GPT-4o

You are an image edit expert, There is a comparison image (left and right). Please describe what you see on the left and right side ....

Figure 4: Overview of the annotation pipeline using GPT-4o. GPT-4o generates a set of source caption, target caption, and edit instruction describing the transformation from Isrc to Itar.

Automatic Image Pairs Generation. We introduce a semi-automated pipeline for constructing a high-quality dataset. A custom script interfaces with a Discord bot to send /imagine commands to MidJourney, generating high-fidelity images. Also using the GPT-4o [27] multimodal API, we generate context-aware images from original inputs and edits. For low-level tasks, we additionally curate a subset of well-known benchmark datasets[31, 30, 10, 36, 26, 13, 9, 8] through manual collection to ensure coverage of classic image processing scenarios. To improve annotation efficiency and scalability, we leverage the multimodal capabilities of GPT-4o to automatically generate image captions and editing instructions. Specifically, we concatenate the source image (Isrc) and the corresponding edited image (Itar) as a joint input to the GPT-4o API. A structured prompt guides the model to produce three outputs: (1) a concise description of Isrc; (2) a concise description of Itar; and (3) a human-readable editing instruction describing the transformation from Isrc to Itar. An example illustrating the pipeline is shown in Figure 4. To conform with the model’s input specification, image pairs are sampled and arranged via rotational permutation, with up to 2,000 instances selected per task to ensure distributional balance. In each sample, the upper half is used as visual context for the RelationAdapter, and the lower half is input to the In-Context Editor module. Directional editing instruction (Isrc → Itar) are provided solely as text prompt, without detailed content descriptions.

- 4 Experiments

#### 4.1 Settings

We initialize our model with FLUX.1-dev [2] within the DiT architecture in training. To reduce computational overhead while retaining the pretrained model’s generalization, we fine-tune the In-Context Editor using LoRA, with a rank of 128. Training spans 100,000 iterations on 4 H20 GPUs, with an accumulated batch size of 4. We use the AdamW optimizer and bfloat16 mixed-precision training, with an initial learning rate of 1×10−4. The total number of trainable parameters is 1,569.76 million. Training takes 48 hours and consumes ∼74GB of GPU memory. At inference, the model requires ∼40 GB of GPU memory on a single H20 GPU. The RelationAdapter employs a dual-branch SigLIP visual encoder, where each branch independently processes one image from the input pair and outputs a 128-dimensional feature token via a two-layer linear projection network. The attention fusion coefficient α is fixed to 1. To balance computational efficiency, input images are resized such that their area corresponds to a maximum long side of 512 pixels before encoding.

#### 4.2 Benchmark

We selected 2.6% of the dataset (6,540 samples) as a benchmark subset, covering a diverse range of 218 tasks. Among these, 6,240 samples correspond to tasks seen during training, while 300 represent unseen tasks used to evaluate the model’s generalization capability.

#### 4.3 Baseline Methods

To assess the performance of our method, we compare it against two baselines: Edit Transfer [6] and VisualCloze [22]. Both baselines follow an in-context learning setup and are evaluated within the shared training task space to ensure a fair comparison, using the official implementation and recommended hyperparameters to ensure reproducibility.

#### 4.4 Evaluation Metrics

We evaluate model performance using four key metrics: Mean Squared Error (MSE), CLIP-based Image-to-Image Similarity (CLIP-I), Editing Consistency (GPT-C), and Editing Accuracy (GPT-

- A). MSE [47] quantifies low-level pixel-wise differences between the generated and ground-truth images, while CLIP-I [33] captures high-level semantic similarity by measuring the CLIP-based distance between generated and ground-truth images. To further assess editing quality from a humancentered perspective, we leverage GPT-4o to interpret the intended transformation from the prompt image Iprm to the reference image Iref, and evaluate the predictions based on two dimensions: Editing Consistency (GPT-C), which measures alignment with the source image Isrc, and Editing Accuracy (GPT-A), which assesses how faithfully the generated image reflects the intended edit.

#### 4.5 Comparison and Evaluation

Quantitative Evaluation. As shown in Table 1, our method consistently outperforms the baselines in both MSE and CLIP-I metrics. Compared to Edit Transfer, our model achieves a significantly lower MSE (0.020 vs. 0.043) and a higher CLIP-I score (0.905 vs. 0.827), indicating better pixellevel accuracy and semantic consistency with the ground truth. Similarly, when compared with VisualCloze, our method achieves a notable improvement, reducing the MSE from 0.049 to 0.025 and boosting CLIP-I from 0.802 to 0.894. These results demonstrate the effectiveness of our approach in producing both visually accurate and semantically meaningful image edits. Our method consistently outperforms two state-of-the-art baselines in both GPT-C and GPT-A metrics.

Qualitative Evaluation. As shown in Figure 5, our method demonstrates strong editing consistency and accuracy in both seen and unseen tasks. Notably, in the unseen task of adding glasses to a person, our approach even outperforms Edit Transfer, which was explicitly trained on this task. In contrast, Edit Transfer shows instability in low-level color control (e.g., clothing color degradation). Compared to VisualCloze, our method is less affected by the reference image Iref, especially in tasks like depth prediction and clothes try-on. VisualCloze tends to overly rely on Iref, reducing transfer accuracy, while our method more reliably extracts key editing features, enabling stable transfer. On unseen tasks, VisualCloze often shows inconsistent edits, such as foreground or background shifts. Our method better preserves structural consistency. This may be due to VisualCloze’s bidirectional attention causing feature spillover. Although our method retains some original color in style transfer, it produces more coherent edits overall, indicating room to further improve generalization.

#### 4.6 Ablation Study

To assess the effectiveness of our proposed RelationAdapter module, we conducted an ablation study by directly concatenating the visual prompt features with the condition tokens cS. For a fair comparison, this baseline was trained for 100,000 steps, identical to RelationAdapter. As shown in Table 2, our model consistently outperforms the in-context learning baseline across all four evaluation metrics on both seen and unseen tasks. This improvement is attributed to the RelationAdapter, which enhances performance by decoupling visual features and reducing redundancy.

Although latent-space concatenation (i.e., directly merging four input images before VAE encoding) is effective, it incurs high GPU memory usage. This limitation restricts the resolution of generated images and compromises fine-grained details during inference. In contrast, our lightweight

###### Seen Tasks Unseen Tasks

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

Edit Transfer

Edit Transfer

Edit Transfer

Input Images Input Images Input Images

Ours

Ours

Ours

[Figure 59]

[Figure 60]

[Figure 61]

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

Input Images Ours VisualCloze Input Images Ours VisualCloze Ours VisualCloze

Input Images

Figure 5: Compared to baselines, RelationAdapter demonstrates outstanding instruction-following ability, image consistency, and editing effectiveness on both seen and unseen tasks.

Table 1: Quantitative Comparison of Baseline Methods Trained on a Common Task (ET: Edit Transfer, VC: VisualCloze). The best results are denoted as Bold.

Method MSE ↓ CLIP-I ↑ GPT-C ↑ GPT-A ↑

EditTransfer 0.043 0.827 4.234 3.508 Ours ∩ ET 0.020 0.905 4.437 4.429 VisualCloze 0.049 0.802 3.594 3.411 Ours ∩ VC 0.025 0.894 4.084 3.918

Table 2: Ablation Study on the Effectiveness of the RelationAdapter(RA) in Seen and Unseen Tasks (-S for Seen, -U for Unseen). The best results are denoted as Bold.

Method MSE ↓ CLIP-I ↑ GPT-C ↑ GPT-A ↑

w/o RA -S 0.055 0.787 3.909 3.597 Ours -S 0.044 0.852 4.079 4.106 w/o RA -U 0.061 0.778 3.840 3.566 Ours -U 0.053 0.812 4.187 4.173

RelationAdapter provides a more efficient alternative, enabling the model to capture and apply the semantic intent of editing instructions with minimal computational cost. Figure 6 demonstrates that our approach yields higher editing accuracy and consistency in both task settings.

###### Seen Tasks Unseen Tasks

[Figure 83]

[Figure 84]

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

In-Context Learning

In-Context Learning

In-Context Learning

Input Images Input Images Input Images

Ours

Ours

Ours

Figure 6: Ablation study results. Our strategy shows better editorial consistency.

### 5 Discussion

As shown in Figure 7, RelationAdapter demonstrates superior performance in various image editing tasks. This performance can be attributed to the integration of a lightweight module that performs weighted fusion with attention, leading to more precise edits. Notably, this suggests that leveraging visual prompt relation can be effectively decoupled from conditional generation through attention

fusion, without the need for full bidirectional self-attention. This finding reveals a promising direction for designing more efficient and scalable editing models.

- Table 3: Quantitative comparison of evaluation metrics (mean ± std) across four image generation tasks. Best results are shown in bold.

Tasks MSE ↓ CLIP-I ↑ GPT-C ↑ GPT-A ↑

Low-Level (n=32) 0.028 ± 0.038 0.885 ± 0.067 3.943 ± 0.383 3.822 ± 0.406 Style Transfer (n=84) 0.051 ± 0.032 0.846 ± 0.036 4.077 ± 0.198 4.246 ± 0.285 Image Editing (n=63) 0.031 ± 0.023 0.861 ± 0.055 4.173 ± 0.229 4.100 ± 0.400 Customized Generation (n=39) 0.065 ± 0.048 0.816 ± 0.073 4.071 ± 0.224 4.064 ± 0.313

We evaluated RelationAdapter on four classification tasks of varying complexity. As shown in Table 3, it excels in complex tasks like style transfer and customized generation, showing strong semantic alignment and text-image consistency. In editing tasks, it balances reconstruction and semantics well. While GPT scores slightly drop in low-level tasks, further low-level evaluations and a user study (see supplementary materials) provide a more complete assessment.

###### Seen Tasks Unseen Tasks

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

Input Images Result Input Images Result Input Images Result

- Figure 7: The generated results of RelationAdapter. RelationAdapter can understand the transformations in example image editing pairs and apply them to the original image to achieve high-quality image editing. It demonstrates a certain level of generalization capability on unseen tasks.

### 6 Limitation

Although our model performs well across various editing tasks, it sometimes fails to accurately render text details in the generated images. This is a common problem with current Diffusion models. In addition, the model may perform slightly differently on rare or previously unseen tasks, suggesting that it is sensitive to task-specific nuances.

### 7 Conclusion

In this work, we proposed RelationAdapter, a novel visual prompt editing framework based on DiT, which strikes a previously unattained balance between efficiency and precision. We began by revisiting the limitations of existing in-context learning approaches and introduced a decoupled strategy for re-injecting visual prompt features. Leveraging the inherent editing capabilities of DiT, our method enhances both the stability and the generative quality of the model in transformation learning scenarios. To support our approach, we constructed a large-scale dataset comprising 218 visual prompt-based editing tasks. We further introduced two training paradigms—position encoding cloning and a noise-free conditioning scheme for In-Context Editor, which significantly improve the model’s editing capability. Extensive experiments validate the effectiveness of our method and demonstrate its superior performance across diverse editing scenarios. We believe this efficient and accurate framework offers new insights into visual prompt-based image editing and lays the groundwork for future research.

### References

- [1] Black Forest Labs. Flux: Official inference repository for flux.1 models. https://github.com/ black-forest-labs/flux, 2024. Accessed: 2025-05-14.
- [2] Black Forest Labs. Flux.1-dev: A 12b parameter rectified flow transformer for text-to-image generation. https://huggingface.co/black-forest-labs/FLUX.1-dev, 2024. Accessed: 2025-05-14.
- [3] Black Forest Labs. Flux.1 redux-dev. https://huggingface.co/black-forest-labs/FLUX. 1-Redux-dev, 2024. Accessed: 2025-05-14.
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22560–22570, 2023.
- [6] Lan Chen, Qi Mao, Yuchao Gu, and Mike Zheng Shou. Edit transfer: Learning image editing via vision in-context relations. arXiv preprint arXiv:2503.13327, 2025.
- [7] Hai Ci, Pei Yang, Yiren Song, and Mike Zheng Shou. Ringid: Rethinking tree-ring watermarking for enhanced multi-key identification. In European Conference on Computer Vision, pages 338–354. Springer, 2024.
- [8] Peng Dai, Xin Yu, Lan Ma, Baoheng Zhang, Jia Li, Wenbo Li, Jiajun Shen, and Xiaojuan Qi. Video demoireing with relation-based temporal consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [9] Egor Ershov, Alexey Savchik, Illya Semenkov, Nikola Bani´c, Alexander Belokopytov, Daria Senshina, Karlo Košˇcevi´c, Marko Subaši´c, and Sven Lonˇcari´c. The cube++ illumination estimation dataset. IEEE Access, 8:227511–227527, 2020.
- [10] Hao Feng, Wengang Zhou, Jiajun Deng, Yuechen Wang, and Houqiang Li. Geometric representation learning for document image rectification. In Proceedings of the European Conference on Computer Vision (ECCV), 2022.
- [11] Rohit Gandikota, Joanna Materzy´nska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. In European Conference on Computer Vision, pages 172–188. Springer, 2024.
- [12] Hailong Guo, Bohan Zeng, Yiren Song, Wentao Zhang, Chuang Zhang, and Jiaming Liu. Any2anytryon: Leveraging adaptive position embeddings for versatile virtual clothing tasks. arXiv preprint arXiv:2501.15891, 2025.
- [13] Yun Guo, Xueyao Xiao, Yi Chang, Shumin Deng, and Luxin Yan. From sky to the ground: A large-scale benchmark and simple baseline towards real rain removal. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12097–12107, October 2023.
- [14] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-toprompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [17] Shijie Huang, Yiren Song, Yuxuan Zhang, Hailong Guo, Xueyin Wang, Mike Zheng Shou, and Jiaming Liu. Photodoodle: Learning artistic image editing from few-shot pairwise data. arXiv preprint arXiv:2502.14397, 2025.
- [18] Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, and Qiang Xu. Humansd: A native skeleton-guided diffusion model for human image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15988–15998, 2023.
- [19] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6007–6017, 2023.

- [20] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.
- [21] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22511–22521, 2023.
- [22] Zhong-Yu Li, Ruoyi Du, Juncheng Yan, Le Zhuo, Zhen Li, Peng Gao, Zhanyu Ma, and Ming-Ming Cheng. Visualcloze: A universal image generation framework via visual in-context learning. arXiv preprint arXiv:2504.07960, 2025.
- [23] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens Van Der Maaten. Exploring the limits of weakly supervised pretraining. In Proceedings of the European conference on computer vision (ECCV), pages 181–196, 2018.
- [24] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
- [25] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 4296–4304, 2024.
- [26] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3883–3891, 2017.
- [27] OpenAI. Gpt-4o technical report. https://openai.com/index/gpt-4o, May 2024. Accessed: 202505-14.
- [28] Zexu Pan, Zhaojie Luo, Jichen Yang, and Haizhou Li. Multi-modal attention for speech emotion recognition. arXiv preprint arXiv:2009.04107, 2020.
- [29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [30] Eduardo Pérez-Pellitero, Sibi Catley-Chandar, Ales Leonardis, and Radu Timofte. Ntire 2021 challenge on high dynamic range imaging: Dataset, methods and results. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 691–700, 2021.
- [31] Xavier Soria Poma, Edgar Riba, and Angel Sappa. Dense extreme inception network: Towards a robust cnn model for edge detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1923–1932, 2020.
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.
- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [35] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.
- [36] Thomas Schöps, Johannes L. Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multicamera videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [37] Wenda Shi, Yiren Song, Dengming Zhang, Jiaming Liu, and Xingxing Zou. Fonts: Text rendering with typography and style controls. arXiv preprint arXiv:2412.00136, 2024.

- [38] Wensong Song, Hong Jiang, Zongxing Yang, Ruijie Quan, and Yi Yang. Insert anything: Image insertion via in-context editing in dit. arXiv preprint arXiv:2504.15009, 2025.
- [39] Yiren Song, Danze Chen, and Mike Zheng Shou. Layertracer: Cognitive-aligned layered svg synthesis via diffusion transformer. arXiv preprint arXiv:2502.01105, 2025.
- [40] Yiren Song, Shijie Huang, Chen Yao, Xiaojun Ye, Hai Ci, Jiaming Liu, Yuxuan Zhang, and Mike Zheng Shou. Processpainter: Learn painting process from sequence data. arXiv preprint arXiv:2406.06062, 2024.
- [41] Yiren Song, Cheng Liu, and Mike Zheng Shou. Makeanything: Harnessing diffusion transformers for multi-domain procedural sequence generation. arXiv preprint arXiv:2502.01572, 2025.
- [42] Yiren Song, Cheng Liu, and Mike Zheng Shou. Omniconsistency: Learning style-agnostic consistency from paired stylization data. arXiv preprint arXiv:2505.18445, 2025.
- [43] Yiren Song, Xiaokang Liu, and Mike Zheng Shou. Diffsim: Taming diffusion models for evaluating visual similarity. arXiv preprint arXiv:2412.14580, 2024.
- [44] Yiren Song, Shengtao Lou, Xiaokang Liu, Hai Ci, Pei Yang, Jiaming Liu, and Mike Zheng Shou. Anti-reference: Universal and immediate defense against reference-based generation. arXiv preprint arXiv:2412.05980, 2024.
- [45] Cong Wan, Xiangyang Luo, Zijian Cai, Yiren Song, Yunlong Zhao, Yifan Bai, Yuhang He, and Yihong Gong. Grid: Visual layout generation. arXiv preprint arXiv:2412.10718, 2024.
- [46] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023.
- [47] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.
- [48] Sam Witteveen and Martin Andrews. Investigating prompt engineering in diffusion models. arXiv preprint arXiv:2211.15462, 2022.
- [49] Xingqian Xu, Zhangyang Wang, Gong Zhang, Kai Wang, and Humphrey Shi. Versatile diffusion: Text, images and variations all in one diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7754–7765, 2023.
- [50] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18381–18391, 2023.
- [51] Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike, et al. Imagebrush: Learning visual in-context instructions for exemplar-based image manipulation. Advances in Neural Information Processing Systems, 36:48723–48743, 2023.
- [52] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [53] Gong Zhang, Kihyuk Sohn, Meera Hahn, Humphrey Shi, and Irfan Essa. Finestyle: Fine-grained controllable style personalization for text-to-image models. Advances in Neural Information Processing Systems, 37:52937–52961, 2024.
- [54] Lvmin Zhang and Maneesh Agrawala. Transparent image layer diffusion using latent transparency. arXiv preprint arXiv:2402.17113, 2024.
- [55] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847,

- 2023.

[56] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078,

- 2024.

- [57] Yuxuan Zhang, Yiren Song, Jinpeng Yu, Han Pan, and Zhongliang Jing. Fast personalized text to image synthesis with attention injection. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6195–6199. IEEE, 2024.

- [58] Yuxuan Zhang, Lifu Wei, Qing Zhang, Yiren Song, Jiaming Liu, Huaxia Li, Xu Tang, Yao Hu, and Haibo Zhao. Stable-makeup: When real-world makeup transfer meets diffusion model. arXiv preprint arXiv:2403.07764, 2024.
- [59] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027, 2025.
- [60] Yuxuan Zhang, Qing Zhang, Yiren Song, and Jiaming Liu. Stable-hair: Real-world hair transfer via diffusion model. arXiv preprint arXiv:2407.14078, 2024.
- [61] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024.

# Appendices

The Appendices provide a comprehensive overview of the experimental framework used to develop and evaluate our method. It includes implementation details (Section A), comparisons with baselines (Section B), failure case analysis (Section C), user study design (Section D), and additional results (Section E).

### A Implementation Details

- A.1 Data Annotation

We leverage the multimodal capabilities of GPT-4o to automatically generate image captions and editing instructions. Specifically, we concatenate the source image Isrc and the corresponding target image Itar as a single input to the GPT-4o API. A structured text prompt—illustrated in Figure 9—is provided to guide the model in producing three outputs: a concise caption for Isrc; a concise caption for Itar; a human-readable instruction describing the transformation from Isrc to Itar. Notably, the editing instruction is provided solely in textual form, without detailed descriptions of image content.

- A.2 Inference Details

During inference, we set the guidance_scale to 3.5, the number of denoising steps to 24, and the attention fusion weight α to 1.0. A fixed random seed of 1000 was used to ensure reproducibility.

### B Details of Comparisons with Baselines

B.1 Baseline and Ablation Study Settings

We adopt the official implementations and default configurations for both VisualCloze and Edit Transfer. During inference, since VisualCloze supports layout prompts, we specify the layout as: "4 images are organized into a grid of 2 rows and 2 columns." Before concatenating the images into the grid layout, each individual image is resized to a square region with an area of 512 × 512 pixels to ensure consistent resolution and layout compatibility. We fix the random seed to 1000 and use the default 30 denoising steps. For Edit Transfer, we similarly set the random seed to 1000, while keeping all other parameters at their default values.

In the ablation study, we remove all components related to the RelationAdapter module and directly feed the prompt image Iprm and the reference image Iref into the In-Context Editor. Additionally, we apply Position Encoding Cloning to each input image to retain spatial correspondence. All other configurations are kept unchanged to ensure fair comparison.

- B.2 Evaluation Details

We leverage the multimodal reasoning capabilities of GPT-4o to interpret the intended transformation from the prompt image Iprm to the reference image Iref, and evaluate model predictions from a human-centered perspective along two key dimensions: Editing Consistency (GPT-C) and Editing Accuracy (GPT-A).

To facilitate this evaluation, we construct composite inputs consisting of five concatenated images: the prompt image Iprm, the reference image Iref (representing the desired attribute or change), the source image Isrc, and two generated results Ipred1 and Ipred2. GPT-4o is then prompted to interpret the intended edit and assess each prediction based on the above criteria. The specific text prompt provided to GPT-4o is illustrated in Figure 10.

- B.3 Perceptual Capability Evaluation

We evaluate the model’s perceptual capability across a series of low-level image editing tasks, including depth estimation, surface normal prediction, edge detection, and semantic segmentation. We further compare its performance against the current state-of-the-art general-purpose image generation framework, VisualCloze, using multiple evaluation metrics. Detailed results are provided in Tables 4, 5, 6, and 7.

### C Failure Cases

Figure 11 illustrates a set of challenging editing tasks. While the model successfully captures edit intentions in several cases, it struggles with fine-grained spatial alignment and the restoration of detailed textual elements. A future solution could involve training on higher-resolution data to better capture spatial nuances.

- Table 4: Edge detection performance on the BSDS500 dataset.

Metric VisualCloze Ours

Precision ↑ 0.3476 0.2266 Recall ↑ 0.0837 0.3134 F1-score ↑ 0.1227 0.2150

Table 5: Segmentation performance on the COCO dataset.

Metric VisualCloze Ours

Pixel Acc. ↑ 0.7817 0.7810 Mean Acc. ↑ 0.3959 0.4722 Mean IoU ↑ 0.3143 0.3642

Table 6: Depth estimation (δ1) on multiple datasets.

Dataset VisualCloze Ours

BSDS500 0.1492 0.1833 COCO 0.1515 0.1750 BIPED 0.2954 0.3088

Table 7: Surface normal estimation results. Lower error and higher accuracy indicate better performance. Mean/Median Angular Error measure deviation from ground truth (°), while Accuracy@X° reports the percentage of predictions within X degrees. Best results are highlighted in bold.

Metric / Dataset BSDS500 COCO BIPED NYUv2

VisualCloze Ours VisualCloze Ours VisualCloze Ours VisualCloze Ours

Mean Angular Error (◦) 52.29 27.15 63.30 35.62 53.15 29.83 43.19 31.69 Median Angular Error (◦) 49.15 24.59 61.00 32.66 51.01 28.39 37.78 28.30 Accuracy (<11.25◦) 6.76 16.12 2.21 11.90 4.09 12.28 4.47 11.46 Accuracy (<22.5◦) 22.10 47.89 9.59 36.30 15.06 38.57 20.74 37.33 Accuracy (<30◦) 33.52 65.88 18.13 51.68 25.56 54.87 36.38 53.75

### D User Study

We conducted a user study to evaluate our method. Thirty volunteers were recruited to complete assessment questionnaires. In each task, participants were presented with a pair of task prompt images (representing the intended edit), one source image, and two edited results: one generated by our proposed method and the other by a baseline method. For the in-context learning baseline, we used the model variant from our ablation study with the RelationAdapter module removed. All images were randomly sampled to ensure fairness across tasks. To mitigate potential bias, the order of the two edited images was randomized for each task.

Participants were instructed to interpret the intended transformation from the prompt pair and answer the following three questions:

- 1. Edit Accuracy: Which image better aligns with the editing intent implied by the prompt pair?
- 2. Edit Consistency: Which image better preserves the structure and identity of the source image?
- 3. Overall Preference: Which image do you prefer overall?

The aggregated results of the user study are summarized in Figure 8. When compared with an in-context learning-based method, our approach was preferred for tasks included in training in 73.19% of cases for Edit Accuracy, 80.08% for Edit Consistency, and 79.58% for Overall Preference. Even on tasks unseen during training, users still favored our method in 57.67%, 57.00%, and 66.33% of cases, respectively.

We also conducted comparisons against other representative baselines. Against VisualCloze, our method was preferred in 70.98% of cases for Edit Accuracy, 72.55% for Edit Consistency, and 69.22% for Overall Preference. When compared to Edit Transfer, the preference gap widened further, with our method selected in 97.11% of cases for Edit Accuracy, 78.89% for Edit Consistency, and 75.78% for Overall Preference.

| |
|---|

Figure 8: User study results comparing our method with baselines (in-context learning, VisualCloze and Edit Transfer) across evaluation criteria: edit accuracy, edit consistency, and overall preference.

### E Additional Results

As shown in Figures 12, 13, and 14, our method demonstrates strong performance across diverse editing tasks, effectively handling spatial transformations and capturing complex semantic modifications with high fidelity.

#### Text Prompts

This is a side-by-side comparison image (left and right). Please describe what you see on the left and right side respectively, and provide a transformation or edit instruction from left to right. Return only a JSON object with the following fields:

- 1. ’left_image_description’
- 2. ’right_image_description’
- 3. ’edit_instruction’ Do not include any other text or explanation.

- Figure 9: Structured prompt used for labeling image pairs and extracting transformation instructions. Text Prompts

You are given a composite image with two columns. The left column contains three images arranged vertically: Left Column: A (original image), A1 (edited version of A), B (another original image).

The right column contains two images: B1 and B2, which are two independently edited versions of image B.

Your task is to independently score B1 and B2 based on two dimensions:

- 1. Edit Consistency (1–5): How visually consistent is the edited image (B1 or B2) with the original image B? Focus: Are key objects, colors, and structures consistent with the source?
- 2. Edit Accuracy (1–5): Assess how accurately the editing operation applied to B (to produce B1 or B2) mirrors the transformation seen from A → A1. Focus: Did the editor apply similar changes, in the correct location, with the same degree of modification?

Avoid giving tied scores unless B1 and B2 are truly indistinguishable. Ensure scores reflect nuanced differences in both consistency and accuracy between B1 and B2. Be critical. Reserve scores of 4–5 for highly consistent/accurate edits. Be objective and concise in your assessment.

Return your answer in the following JSON format: { "B1": {"consistency":<1-5>, "accuracy":<1-5>}, "B2": {"consistency":<1-5>, "accuracy":<1-5>} }

- Figure 10: Evaluation prompt used to assess edit consistency and accuracy between two generated outputs, leveraging GPT-4o for interpretation and scoring.

[Figure 131]

Input Result Input Result

- Figure 11: Failure cases on gesture editing, background pedestrian removal, document rectification, and image-to-sketch conversion. The model shows partial success with room for improvement.

[Figure 132]

Input Result Input Result

- Figure 12: Additional experimental results of RelationAdapter.

[Figure 133]

Input Result Input Result

- Figure 13: Additional experimental results of RelationAdapter.

[Figure 134]

Input Result Input Result

- Figure 14: Additional experimental results of RelationAdapter.

