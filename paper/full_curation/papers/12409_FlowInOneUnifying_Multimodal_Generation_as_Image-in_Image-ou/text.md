# arXiv:2604.06757v2[cs.CV]14May2026

[Figure 1]

## FLOWINONE: UNIFYING MULTIMODAL GENERATION AS IMAGE-IN, IMAGE-OUT FLOW MATCHING

Junchao Yi 1∗, Rui Zhao 3∗, Jiahao Tang 2∗, Weixian Lei3, Linjie Li 5, Qisheng Su 4, Zhengyuan Yang 5, Lijuan Wang 5, Xiaofeng Zhu 1†, Alex Jinpeng Wang 2† 1 University of Electronic Science and Technology of China 2 Central South University 3 National University of Singapore 4 University of Science and Technology of China 5 Microsoft

Homepage: https://csu-jpg.github.io/FlowInOne.github.io/

#### ABSTRACT

Multimodal generation has long been dominated by text-driven pipelines where language dictates vision but cannot reason or create within it. We challenge this paradigm by asking whether all modalities, including textual descriptions, spatial layouts, and editing instructions, can be unified into a single visual representation. We present FlowInOne, a framework that reformulates multimodal generation as a purely visual flow, converting all inputs into visual prompts and enabling a clean image-in, image-out pipeline governed by a single flow matching model. This vision-centric formulation naturally eliminates cross-modal alignment bottlenecks, noise scheduling, and taskspecific architectural branches, unifying text-to-image generation, layout-guided editing, and visual instruction following under one coherent paradigm. To support this, we introduce VisPrompt-5M, a large-scale dataset of 5 million visual prompt pairs spanning diverse tasks including physics-aware force dynamics and trajectory prediction, alongside VP-Bench, a rigorously curated benchmark assessing instruction faithfulness, spatial precision, visual realism, and content consistency. Extensive experiments demonstrate that FlowInOne achieves state-of-the-art performance across all unified generation tasks, surpassing both open-source models and competitive commercial systems, establishing a new foundation for fully vision-centric generative modeling where perception and creation coexist within a single continuous visual space.

#### 1 Introduction

Multimodal generation has long operated under a text-dominant assumption: language encodes intent, and vision executes it. Models such as diffusion- and transformer-based text-to-image systems rely on linguistic embeddings as the central conditioning source. While this design achieves impressive visual fidelity, it introduces a fundamental asymmetry: language controls vision, but vision cannot reason or generate on its own. This fragmentation of representation space makes it inherently difficult to unify understanding, editing, and generation within a single coherent model.

Recently, a growing trend of vision-centric models has emerged Salesky et al. [2021], Rust et al. [2022], Xiao et al. [2024], suggesting that textual information can be processed in a purely visual manner by rendering text into pixel space. These studies collectively suggest that the visual modality itself is expressive enough to serve as the foundation for multimodal understanding. Together, these studies show that representing language visually enables unified perception and alignment within a single modality. Yet they remain fundamentally perception-oriented, leaving the generative potential of this vision-first formulation largely unexplored. This raises an important question: can we build a large model that both reasons and generates entirely within the visual space?

Flow matching Lipman et al. [2023] provides a principled answer to this question. Compared with diffusion, it directly learns the underlying velocity field of transformation, offering higher sampling efficiency and stable optimization. By learning visual flows instead of stochastic noise removal, it connects perception and generation under a single deterministic principle.

∗Equal contribution, †Corresponding author

In this work, we take a decisive step toward this goal and introduce FlowInOne, a framework that redefines multimodal generation as a purely visual flow. In FlowInOne, text, layout, and instruction inputs are first transformed into visual prompts, forming the input image state. The model then learns a continuous transport process that evolves this state into the target visual output using flow matching. This formulation enables a simple and general training pipeline that eliminates noise scheduling, diffusion sampling, and task-specific condition heads.

Text-to-image

Text-image-to-image Ours Image-in-image-out

[Figure 2]

|[Figure 3]|
|---|

Text

Text

Visual Encoder

Text Encoder

Text Encoder

Image Encoder

LDM

Flow Matching Model

LDM

Image Decoder

Image Decoder

Image Decoder

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

As shown in Figure 1, FlowInOne departs from the traditional text-conditioned pipeline. Conventional text-toimage or image editing models use text encoders (e.g., Flan-T5 Chung et al. [2024]) to condition a latent diffusion model, while text-image-to-image setups require two encoders for joint conditioning. In contrast, FlowInOne unifies all input conditions as visual prompts, forming a simple image-in, image-out pipeline with a single model. This design not only simplifies architecture but also ensures consistent alignment between semantic content and spatial control across diverse tasks such as text-to-image generation, layout-guided editing, and visual instruction following.

Figure 1: Comparison of generation paradigms. Left: Traditional T2I only uses the text encoder to condition the Latent Diffusion Model(LDM); Middle: Traditional TI2I requires the joint conditioning of both the text and image encoders; Right: We unify the conditions as visual input and form a simple image-in, image-out framework with a single model.

To support this unified paradigm, we construct VisPrompt-5M, a large-scale Visual Prompt Dataset that spans text-inimage generation, versatile visual editing and physics-aware instruction following. Each sample pairs a visual prompt canvas with its corresponding target image, providing supervision as continuous visual evolution without task-specific modules or auxiliary channels. We further introduce VP-Bench, a carefully curated evaluation benchmark that assesses model performance across four dimensions: instruction faithfulness, content consistency, visual realism, and spatial precision.

Our main contributions are as follows. i. We reformulate multimodal generation into a vision-centric image-in, image-out paradigm, eliminating the text encoders and modality-specific bridges. ii. We propose FlowInOne, a unified flow matching framework that models multimodal transformation as continuous visual evolution within a shared latent space. iii. We build VisPrompt-5M, a comprehensive dataset of visual prompts that enables unified training and strong generalization across text-to-image, image-to-image, and instruction-guided generation tasks. Extensive experiments demonstrate that FlowInOne achieves state-of-the-art performance across unified generation, precise image editing, and physics-aware instruction following, establishing a new foundation for fully vision-centric generative modeling.

#### 2 Related Works

Diffusion and Flow Matching. While diffusion models, from DDPM Ho et al. [2020], Song et al. [2021] to LDM Rombach et al. [2022] and DiT Peebles and Xie [2023], dominate image generation via progressive denoising, Flow Matching Lipman et al. [2023], Liu et al. [2023a], Geng et al. [2025], Zhang et al. [2025] learns a continuous transport map between distributions. This approach reduces reliance on complex noise schedules while enhancing sampling efficiency and stability Gat et al. [2024]. Building on this, FlowInOne directly models the continuous evolution within a shared latent space, entirely eliminating additional conditioning or noise injection.

Text- and Image-Conditioned Generation. Current T2I models Ramesh et al. [2021], Zhou et al. [2023], Podell et al. [2023], Chen et al. [2024], Sun et al. [2024] typically inject discrete text tokens via cross-attention Esser et al. [2024], Polyak et al. [2025], leaving control signals disjointed from the visual space. Similarly, conventional image-to-image translation Xiao et al. [2025], Nobis et al. [2024], Liu et al. [2023b,a], Zhou et al. [2024], restoration Wang et al. [2025a], Liu et al. [2025a], and controllable editing Brooks et al. [2023], Hertz et al. [2023], Li et al. [2023], Zhang et al. [2023], Chen et al. [2023], Yang et al. [2024], Pan et al. [2023] rely heavily on adversarial frameworks Mirza and Osindero [2014], Zhu et al. [2020], diffusion priors Meng et al. [2022], or external control channels. A common limitation across these methods is the dependence on explicit masks or task-specific interfaces for geometric and semantic control. In contrast, FlowInOne entirely bypasses specialized conditioning branches. By rendering heterogeneous constraints—such as text and arrows, we converge multiple forms of control into a single image input, natively aligning semantics and geometry within the visual domain.

Color Alteration Compound Edit Object Remove Expression Modification

Relationship Edit

Object Remove

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>|
|---|

|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>|
|---|

|[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>|
|---|

|[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>|
|---|

Text in

Canny to Image Depth to Image Restoration

Inpainting

image edit

|[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]|
|---|

|[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]|
|---|

|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]|
|---|

|[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]|
|---|

…

…

Attribute modification

Object Swap Object Remove

Object Remove

Position and Size Control

|[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]|
|---|

|[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|
|---|

|[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]|
|---|

|[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]|
|---|

Visual marker edit

Textbbox

…

…

edit

edit

Trajectory prediction

Class to Image

|[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|
|---|

|[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]|
|---|

|[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]|
|---|

|[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]|
|---|

Trajectory understand

understand

Class to image

[Figure 85]

[Figure 86]

Global Force Point Force

|[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]|
|---|

|[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]|
|---|

|[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]|
|---|

Force understand

understand

Text to Image

|[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]|
|---|

|[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]|
|---|

|[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]|
|---|

Text to image

Single Doodle edit More Doodles edit

|[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]|
|---|

|[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]|
|---|

|[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]|
|---|

Doodles edit

edit

- Figure 2: VisPrompt-5M is a comprehensive dataset that comprises eight distinct data types, including class-toimage generation, text-to-image generation, text-in-image editing, text bounding box editing, visual marker editing, doodles editing, force understanding, and trajectory understanding. The dataset covers a wide spectrum of image-toimage generation, ranging from basic text-in-image generation to compositional editing, and further to physics-aware instruction following.

Modal mapping. While standard diffusion models map discrete text to images across divergent modalities Chen

- et al. [2024], Fan et al. [2024], Sucheng et al. [2025], Yu et al. [2024], Weber et al. [2024], Kim et al. [2025], Bai et al. [2025a], Chen et al. [2025a], Zheng et al. [2025], Chang et al. [2025], inherent modality gaps, tokenization artifacts, and reliance on Gaussian noise severely limit spatial precision Jia et al. [2025]. Moving beyond generic image-to-image translation Liu et al. [2023b,a], Zhou et al. [2024], we frame generation fundamentally as an intra-modal transport problem. By encoding both the visually-instructed input and the target image into a shared, isomorphic latent space, we learn a direct, noise-free flow between them. This pure single-modality mapping resolves structural mismatches at the latent level, demonstrating exceptional scalability across diverse editing and generation tasks.

#### 3 Dataset and Benchmark

In this section, we detail the construction of the VisPrompt-5M dataset (Sec. 3.1) and our evaluation benchmark, VP-Bench (Sec. 3.2).

###### 3.1 VisPrompt-5M

Illustrated in Figure 2, VisPrompt-5M enables a unified image-in, image-out paradigm. Training pairs (Iv,I⋆) embed all textual and spatial instructions directly into the input canvas Iv. Eliminating auxiliary text channels mitigates ambiguity and enforces strict geometric alignment, empowering a single model to handle diverse tasks within one modality.

Fundamental Generation. We render textual prompts from text-to-image-2M zk [2024] and class labels from an 860K high-quality ImageNet Russakovsky et al. [2014] subset directly as input canvases to match their corresponding target images.

Text-in-Image Editing. This category unifies diverse editing and condition-to-image tasks by overlaying textual instructions directly onto the input image canvas. Drawing from GPT-Image-Edit Wang et al. [2025b], Pico-Banana Qian et al. [2025], and UnicEdit Ye et al. [2025], we filter out complex or inconsistent pairs to retain approximately 1.6M examples, spanning a diverse range of edits including additions, deletions, and attribute/environment changes. Furthermore, we seamlessly integrate 315K structured image pairs from PixWizard Lin et al. [2024], which encompasses a broad spectrum of tasks including Canny-to-image, depth-to-image, inpainting, and image restoration. In total, this yields nearly 1.9M curated pairs where all operational intents are explicitly embedded as visual text prompts.

Text Bounding Box Editing. For precise object insertion with explicit geometric constraints, we extract a highquality subset of 45K examples from GPT-Image-Edit. Leveraging the combined priors of Qwen Image Edit Wu et al. [2025a] and Qwen3-VL Bai et al. [2025b] to guide I2I generation, we synthesize pairs where text and bounding boxes jointly dictate the target category, scale, and location. Automated filtering via Qwen3-VL 7B ultimately retains 24K high-quality pairs.

Visual Marker Editing. Leveraging visual understanding capabilities of Qwen3-VL, we generate 250K pairs where arrow annotations serve as salient cues without requiring explicit object names in instruction. This subset supports operations such as deletion, replacement, implicitly representing semantics and spatial relationships through visual markers.

Doodles Editing. Using a two-stage synthesis pipeline with Qwen Image Edit Wu et al. [2025a] on 5K images crawled from the web, we first add doodles to create input images, then transform them into photo-realistic objects to form target images. Rigorous manual inspection mitigates generative instability, yielding 1K high-quality pairs where doodle lines explicitly serve as shape priors.

Force & Trajectory Understanding. VisPrompt-5M supports physics-aware generation. For motion trajectories, we manually annotated the Blender-rendered videos of car and ball movements, yielding 1.5K strictly curated image pairs. For force understanding, we leverage the Force Prompting dataset Gillman et al. [2025] that covers aerodynamics, oscillations, and linear motion. By extracting keyframes and superimposing text and arrows to denote precise force magnitude and direction, we explicitly visualize object dynamics within the input image.

Across the entire pipeline, we enforce standardized data formats while strictly maintaining task metadata and geometric attributes. We also incorporate automated quality control measures via MLLMs to ensure semantic consistency, visual fidelity, and visual text readability.

###### 3.2 Benchmark and Evaluation

To evaluate our pure Image-in, Image-out paradigm, we curate VP-Bench, a comprehensive and manually filtered benchmark covering diverse visual instructions (details provided in Appendix B).

Since conventional metrics like FID Heusel et al. [2018] struggle with complex visual instructions, we follow recent studies Labs et al. [2025], Liu et al. [2025b] by adopting VLMs as our primary evaluators. A generation is deemed successful only if it simultaneously satisfies four criteria: (1) Instruction Faithfulness, (2) Content Consistency, (3) Visual Realism, and (4) Spatial Precision. Alongside the VLM assessment, we conduct rigorous human evaluation based on these exact same criteria to compute the overall pass rate. Since standard VLMs may miss implicit constraints rendered on the image canvas, we manually extract the textual instructions and supply them as supplementary text prompts to ensure fair assessment (refer to Appendix F for VLM evaluation prompts).

To comprehensively evaluate visual quality and editing accuracy, we additionally tailor four quantitative metrics to our paradigm: (1) CLIP-IQA Wang et al. [2022a] to measure overall visual realism; (2) CLIP Score Wang et al. [2022b] to evaluate semantic alignment, for which we directly extract the rendered text instructions from the input image and compute their similarity with the generated image; (3) Directional CLIP Similarity Gal et al. [2022] to assess semantic consistency in marker-based editing. Captions are manually generated for the input and generated images, and subsequently computing the directional alignment between the image transition and the corresponding caption pairs; and (4) DINOv3 Directional Similarity (DINOv3 Sim) to accurately capture fine-grained spatial and

physical structural changes, by directly computing the cosine similarity of the edit displacement vectors among the input, generated, and ground-truth images within the dense DINOv3 Siméoni et al. [2025] feature space.

#### 4 Method

In this section, we first briefly review the preliminaries of Flow Matching (Sec. 4.1) and introduce our core strategy for encoding visual instructions into a unified visual semantic space (Sec. 4.2). Finally, we present the detailed architecture of FlowInOne (Sec. 4.3), featuring a novel Dual-Path Spatially-Adaptive Modulation mechanism to balance structural preservation and instruction adherence.

###### 4.1 Preliminaries: Flow Matching

Flow Matching (FM) Lipman et al. [2023], Liu et al. [2023a, 2024] formulates generative modeling as a continuous transport from a source distribution p0 to a target distribution p1 over t ∈ [0,1]. Unlike traditional diffusion models Ho et al. [2020], Song et al. [2021], FM does not rely on complex noise scheduling and permits non-Gaussian source distributions, provided they are isomorphic to the target. During training, FM constructs a differentiable probability path zt between a sample pair (z0,z1), which directly yields the ground-truth velocity vt⋆ for supervision:

∂zt ∂t

zt = tz1 + (1 − (1 − σmin)t)z0, vt⋆ =

= z1 − (1 − σmin)z0 (1)

The network learns a time-dependent velocity field vθ(zt,t) by minimizing the Mean Squared Error (MSE) against vt⋆. In our FlowInOne framework, we explicitly define a non-Gaussian source distribution: z0 represents the latent state of the unified visual instruction extracted via a visual encoder and text-image VAE, while z1 represents the target image latent. Since both latents are isomorphic within a shared space, inference is intuitively performed by solving the Ordinary Differential Equation (ODE) dzt

dt = vθ(zt,t) from t = 0 to t = 1, deterministically evolving the visual instruction into the final target image.

###### 4.2 Unified processing of text in visual semantic space

The inherent heterogeneity between discrete linguistic symbols and continuous visual textures poses significant alignment challenges in flow matching. To alleviate this, we propose a paradigm shift: rendering textual instructions and diverse visual cues directly onto the image canvas. This explicitly preserves spatial layouts and structural priors without relying on complex cross-modal alignment modules.

By treating text as an integral part of the visual geometry, we circumvent the semantic fragmentation typically introduced by textual tokenizers. To extract robust representations from this unified image Iv, we leverage the visual encoder of Janus-Pro-1B Chen et al. [2025b]. The input is processed by a SigLIP Vision Transformer to extract patch-level semantic features, which are then mapped into the target embedding space via an MLP projector:

Xfuse = MLP(SigLIP(Iv)) ∈ RN×D (2)

where N is the number of patches and D denotes the embedding dimension. This sequence, Xfuse, encapsulates both textual semantics and visual geometry.

To perform continuous flow matching, we map the unified visual tokens to a source latent space via a text-image VAE. Rather than predicting it directly, it parameterizes a distribution to sample the source state ZTI ∼ N(¯µz

,diag(¯σz2

)) ∈

0

0

RH×W×C. Symmetrically, a frozen image VAE encodes the target image into an isomorphic latent ZI. Our generative process is thus elegantly formulated as modeling the time-dependent velocity field that continuously transports the

source latent ZTI to the target image latent ZI within the shared latent space.

###### 4.3 FlowInOne

Dual-Path Spatially-Adaptive Modulation. Within the unified Flow Matching framework proposed in this work, we formulate image generation as a deterministic trajectory evolution from a starting distribution p0 to a target distribution p1. However, due to the information compression inherent in the visual encoding stage, the initial latent ZTI often fails to fully capture the fine-grained structural features of the source image Isrc. To address this, we introduce a Dual-Path Spatially-Adaptive Modulation mechanism. This mechanism is designed to dynamically compensate for the missing structural manifold while switching computational paths based on the specific task type. As shown in Figure 3, let H(l) ∈ RN×D denote the hidden state of the l-th Transformer layer, where N is the token sequence length and D is the

input Image with Visual instruction

Image with

|[Figure 137]<br><br>[Figure 138]|
|---|

[Figure 139]

[Figure 140]

###### DiT variant

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

###### Transformer Block

Janus Visual Encoder

###### Text-Image VAE

-Image

with - -

with Dual-Path SpatiallyAdaptive Modulation

Encoder

Self-Attention

-Attention

q

+

+

Attention

CrossAttention

Target image

image

|[Figure 146]<br><br>[Figure 147]|
|---|

k,v

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

C

ImageEncoderVAE +

###### ImageEncoderVAE +

Image VAE

Image VAE Decoder

Adaptive gating network

α

Z

ZTI

·

Timestep

Input image

image

+

+

|[Figure 154]<br><br>[Figure 155]|
|---|

[Figure 156]

[Figure 157]

[Figure 158]

MLP

Image VAE Encoder

Image VAE Encoder

+

+

Z

ZI

Input image latent

- Figure 3: Overview of the FlowInOne architecture, a general and simple framework using flow matching for continuous evolution in only one modality. FlowInOne employs a Dual-Path Spatially-Adaptive Modulation to adapt computation by modality. For input image rendering with only text, the structural branch is bypassed to strictly follow semantic evolution. Conversely, for image editing, a spatially-adaptive gated network and cross attention activates to selectively inject source priors, dynamically balancing original image preservation with instruction-driven reconstruction.

feature dimension. This state is first updated via a self-attention Aself to capture the global contextual information:

H˜ (l) = LayerNorm(H(l) + Aself(H(l))) (3)

Upon obtaining the features enhanced by self-attention, the model follows a dual-path conditional branch defined by the two different tasks. For text-to-image generation in visual input, where no external structural prior needs to be maintained, the model bypasses the cross-attention layer to prevent the introduction of irrelevant noise. This ensures that the generation trajectory strictly follows the evolution dictated by the text semantics. Conversely, for image editing involving a source image, we employ an Image VAE Encoder to map Isrc into the latent space, resulting in the latent zsrc ∈ RH×W×C. Since zsrc captures the visual structure of the image, we reshape it into a reference sequence S ∈ RN×C, where N = H × W. Subsequently, the structural increment ∆Hstruct is computed via a cross-attention mechanism, where the intermediate state H˜ (l) serves as the Query and the reference sequence S acts as the Key-Value pair:

(H˜ (l)WQ)(SWK)⊤ √dk

∆Hstruct = Softmax

###### (SWV ) (4)

To achieve an optimal trade-off between source image fidelity and instruction-following alignment, we design a lightweight adaptive gating network. By concatenating the current denoising state with the extracted source information, the network predicts an anisotropic token-level weight vector Λ ∈ [0,1]N. This design enables the model to identify spatial heterogeneity at a pixel-level granularity—specifically, distinguishing between regions belonging to the background manifold that require strict preservation and regions targeted for editing that require reconstruction. The derivation of the gating coefficient matrix is as follows:

Λ = σ MLPθ [H˜ (l) ∥ ∆Hstruct] (5) where ∥ denotes feature concatenation along the channel axis and σ represents the Sigmoid activation function. The final layer output H(outl) is integrated via a conditional formulation controlled by a task indicator Iedit:

H(outl) = H˜ (l) + Iedit · (Λ ⊙ ∆Hstruct) (6)

Here, Iedit ∈ {0,1} is a binary indicator: for pure text-to-image inputs, Iedit = 0, and the modulation term is nullified to sever structural dependencies; for inputs containing visual instruction and source images, Iedit = 1, activating the spatially-adaptive refinement. By precisely controlling the infiltration of the structural manifold via Λ, this method effectively mitigates editing conflicts caused by over-preservation. Furthermore, by leveraging explicit structural compensation, it significantly reduces the evolution error of Flow Matching in complex image editing scenarios.

Flowing in the unified modality. We model visual instruction images to target images as a flow matching process in a shared latent space. Given the instruction sequence X ∈ RN×C obtained from a unified visual encoder, a variational posterior is defined using a Variational Autoencoder.

qψ(z |X) = N µ ¯X, diag σ ¯X2 , z0 ∼ qψ(z |X), z0 ∈ RN×D.

(7)

Image-side training and sampling are conducted in a latent space: the pre-trained and frozen VAE from LDM Rombach et al. [2022] Encimg is used to map the target image I⋆ to the target latent variable.

z1 = Encimg(I⋆) ∈ RN×D. (8) During training, vanilla flow matching is adopted: time is sampled from t ∼ U(0,1) and linear interpolation is constructed in Equation 1. The instantaneous velocity is represented by the vector field vθ(z,t), minimizing

0,z1),t vθ(zt,t) − (z1 − z0) 22. (9) During inference, given only the visual instruction image Iv, first take z0 ∼ qψ(·|Ev(Iv)), and then solve the ordinary differential equation. Obtaining zˆ1 = zt=1, the final image Iˆ = Decimg(ˆz1) is generated through the frozen image VAE decoder. This process achieves continuous transportation from the instruction state to the image state within a unified one-dimensional sequence modality, avoiding additional noise scheduling and conditional branching, and maintaining an isomorphic representation to the visual instructions.

LFM = E(z

#### 5 Experiment Results

In this section, we first provide the implementation details of FlowInOne (Sec. 5.1), and then present the main results on our carefully curated VP-Bench (Sec. 5.2). Finally, we conduct ablation studies to better understand the design choices of FlowInOne for image generation (Section 5.3).

###### 5.1 Implementation Details

Model architecture. Based on the CrossFlow framework Liu et al. [2025c], we encode unified image input via Janus-pro-1B Chen et al. [2025b] and a frozen LDM VAE Rombach et al. [2022], projecting them through a stacked Transformer text-image VAE. We augment the Transformer blocks with additional cross-attention layers. The concatenated self- and cross-attention outputs are then fed into a lightweight network to predict spatially adaptive weights for input feature modulation.

Training details. The 1.2B FlowInOne is initialized from CrossFlow Liu et al. [2025c] and trained at 256 × 256 resolution for 240k steps via a balanced WebDataset Aizman et al. [2020], optimizing a combined Flow matching, KL divergence, and CLIP contrastive loss (see Appendix G for detailed configurations and more ablations).

Evaluation metrics. We evaluate FlowInOne on a high-quality subset of VP-Bench. Following Section 3.2, we report pass rates assessed by Gemini 3 Google [2025a], GPT 5.2 OpenAI [2025], and Qwen3.5 Qwen Team [2026] and compute four quantitative metrics. Additionally, for qualitative verification, ten independent evaluators cross-checked 250 stratified random samples (25 each) to ensure judgment reliability.

###### 5.2 Evaluation on Image-to-image generation State-of-the-art Comparison. We compare FlowInOne against several competitive open-source frameworks such

- as OmniGen2 Wu et al. [2025b], Qwen-Image-Edit-2509 Wu et al. [2025a], and FLUX.1-Kontext-dev Labs et al. [2025], as well as the commercial model Nano Banana Google [2025b]. To ensure fairness, we evaluate each baseline models through its optimal native interface, using detailed text prompts expanded by Qwen3-VL Bai et al. [2025b] to supplement the image inputs. Quantitative results demonstrate that FlowInOne possesses a significant advantage in unified generation tasks.

- Table 1: Evaluation on the VP-Bench visual instruction benchmark using three VLM evaluators and human judges. We use Gemini3 Google [2025a], GPT5.2 OpenAI [2025] and Qwen3.5 Qwen Team [2026] to evaluate the success ratio of FlowInOne on all kinds of visual instruction tasks. C2I: class to image, T2I: text to image, TIE: text in image edit, FU: force understanding, TBE: text & bbox edit, TU: trajectory understanding, VME: visual marker edit, DE: doodles edit. Total denotes the average success rate across all sub-categories.

Method C2I T2I TIE FU TBE TU VME DE Total Evaluator: Gemini3

Nano Banana Google [2025b] .810 .980 .521 .500 .600 .020 .537 .740 .589 Omnigen2 Wu et al. [2025b] .720 .760 .313 .013 .020 .000 .020 .140 .248 Kontext Labs et al. [2025] .620 .700 .363 .027 .163 .020 .096 .180 .271 Qwen-IE-2509 Wu et al. [2025a] .680 .690 .383 .047 .060 .000 .040 .160 .258

|FlowInOne (Ours)|.890 .700 .355 .727 .302 .520 .292 .535|.540<br><br>|
|---|---|---|
| |Evaluator: GPT5.2<br><br>| |
|Nano Banana Google [2025b]|.760 .960 .402 .163 .100 .020 .227 .495|.391|

Omnigen2 Wu et al. [2025b] .660 .820 .203 .001 .000 .000 .001 .160 .231 Kontext Labs et al. [2025] .620 .690 .266 .013 .093 .000 .056 .160 .237

|Qwen-IE-2509 Wu et al. [2025a] FlowInOne (Ours)<br><br>|.640 .680 .286 .040 .020 .020 .020 .140 .850 .800 .079 .500 .116 .240 .083 .465|.231 .392|
|---|---|---|
| |Evaluator: Qwen3.5| |
| | | |

Nano Banana Google [2025b] .780 .960 .446 .427 .260 .040 .395 .760 .508 Omnigen2 Wu et al. [2025b] .740 .790 .257 .027 .020 .000 .010 .160 .251 Kontext Labs et al. [2025] .720 .690 .322 .020 .133 .040 .083 .140 .269 Qwen-IE-2509 Wu et al. [2025a] .780 .710 .345 .107 .060 .000 .043 .180 .278 FlowInOne (Ours) .859 .720 .354 .713 .272 .320 .306 .481 .503

Evaluator: Human

Nano Banana Google [2025b] .790 .940 .372 .287 .220 .020 .306 .740 .459 Omnigen2 Wu et al. [2025b] .720 .710 .268 .013 .020 .000 .010 .120 .233 Kontext Labs et al. [2025] .640 .680 .317 .013 .080 .020 .048 .120 .240 Qwen-IE-2509 Wu et al. [2025a] .700 .665 .331 .047 .020 .000 .023 .160 .243 FlowInOne (Ours) .800 .645 .242 .705 .255 .280 .255 .400 .449

As presented in Table 1, FlowInOne consistently achieves the best performance among all open-source baselines across different evaluators. Specifically, FlowInOne obtains total success rates of 54.0%, 39.2%, 50.3%, and 44.9% under Gemini3, GPT5.2, Qwen3.5, and Human evaluation, respectively, substantially outperforming OmniGen2, FLUX.1-Kontext-dev, and Qwen-IE-2509. Notably, FlowInOne also remains highly competitive with the commercial model Nano Banana, achieving the highest total score under GPT5.2 and only slightly trailing Nano Banana under Gemini3, Qwen3.5, and Human evaluation. These results demonstrate that FlowInOne establishes a strong open-source baseline and approaches commercial-level performance in the image-in, image-out generation paradigm.

- Table 2 further provides a fine-grained four-dimensional analysis. FlowInOne achieves consistently strong performance in instruction faithfulness and content consistency, showing that it can effectively follow visual instructions while preserving the required semantic content. More importantly, FlowInOne obtains the best spatial precision scores across all three evaluators, i.e., 3.42 under Gemini3, 3.24 under GPT5.2, and 3.30 under Qwen3.5, demonstrating its advantage in spatially grounded image-in, image-out generation. Although Nano Banana achieves higher visual realism, FlowInOne shows a better balance between instruction following and spatial control, especially compared with existing open-source baselines.

Although minor discrepancies exist between MLLMs and human assessments due to limitations in grounding finegrained visual markers Dong et al. [2025], the overall ranking trends consistently align, verifying the reliability of our automated metrics.

- Table 2: Overall four-dimensional breakdown. Each dimension has a maximum score of 5 points. IF: Instruction Faithfulness, CC: Content Consistency, VR: Visual Realism, SP: Spatial Precision.

Method

Gemini3 GPT5.2 Qwen3.5

| |IF CC VR SP|IF CC VR SP|IF CC VR SP<br><br>|
|---|---|---|---|
|Nano Banana Omnigen2 Kontext Qwen-IE-2509 FlowInOne|3.43 3.10 4.43 2.99 2.15 1.39 3.46 1.69 2.31 1.62 3.15 2.25<br><br>2.96 2.51 3.28 1.91<br>3.38 2.94 3.12 3.42<br><br><br>|3.28 2.78 4.06 2.86 2.29 1.53 3.31 1.92 2.40 1.71 3.25 2.21<br><br>2.76 2.39 3.58 1.94<br>3.16 2.81 2.96 3.24<br><br><br>|3.30 3.01 4.43 2.94 1.72 0.91 3.13 1.27<br><br>1.97 1.87 3.11 1.76<br>2.33 1.98 3.23 1.39<br><br>3.31 2.87 3.20 3.30<br><br><br>|

- Table 3: Quantitative evaluation on VP-Bench across five models. CLIP-IQA is averaged over all categories to measure overall visual realism. CLIP Score evaluates semantic alignment for pure generation tasks. Directional CLIP Similarity assesses instruction-driven consistency in marker-based editing. DINOv3 Similarity measures fine-grained spatial and physical structural accuracy.

IQA ↑ CLIP Score ↑ Dir CLIP ↑ DINOv3 Sim ↑

Method

| |Total|C2I T2I Avg.|TIE VME Avg.|DE FU TBE TU Avg.<br><br>|
|---|---|---|---|---|
|Nano Banana Google [2025b] Omnigen2 Wu et al. [2025b] Kontext Labs et al. [2025] Qwen-IE-2509 Wu et al. [2025a] FlowInOne (Ours)|0.688 0.603 0.621 0.646 0.684|0.281 0.302 0.291 0.173 0.208 0.191 0.164 0.172 0.168 0.231 0.216 0.224 0.290 0.276 0.283|0.106 0.103 0.105 0.001 0.005 0.003 0.010 0.008 0.009 0.005 0.011 0.008 0.092 0.101 0.097|0.430 0.474 0.501 0.486 0.473 0.224 0.066 0.004 0.207 0.125 0.283 0.240 0.118 0.004 0.161 0.133 0.215 0.250 0.207 0.201 0.335 0.536 0.506 0.570 0.487|

Beyond pass rates, Table 3 further substantiates our findings. Crucially, FlowInOne excels in fine-grained spatial and physical controls, achieving the highest average DINOv3 Sim score of 48.7% (outperforming Nano Banana’s 47.3%), with notable margins in force & trajectory understanding and text bbox editing. Furthermore, in overall visual realism and semantic alignment, FlowInOne significantly surpasses all open-source baselines and performs comparably to the commercial model, underscoring its robust generation quality and precise instruction-following capabilities.

Qualitative Comparison. Figure 4 visually compares FlowInOne and baselines across five VP-Bench tasks: force, trajectory, text & bbox, visual marker, and doodle editing. Unlike traditional pipelines, FlowInOne directly processes a unified canvas containing textual instructions, spatial layouts, and visual cues (arrows, markers, doodles). For fairness, we evaluate baselines via their optimal native interfaces: instructions are extracted and expanded into detailed prompts by Qwen3-VL before being provided alongside the processed image.

Despite these enhanced prompts, baselines often fail to translate visual cues into precise edits. In force and trajectory tasks, they struggle to convert arrows into physically plausible motions. For text & bbox editing, they frequently violate spatial constraints or size specifications. In marker and doodle tasks, baselines often misinterpret localized cues as scene elements, resulting in inaccurate synthesis or artifact retention. Conversely, FlowInOne accurately executes modifications while preserving background consistency. This success confirms that our image-in, image-out paradigm enables more reliable grounding of fine-grained spatial and physical intents than conventional text-driven interfaces.

###### 5.3 Ablation Studies

We conduct a series of ablation studies to validate the architectural designs of our model. Due to computational constraints, all models are trained for 100k steps with a batch size of 512 unless otherwise specified. To accurately reflect model performance, the reported Pass Rate averages the Gemini and GPT evaluations. Furthermore, we adopt a progressive strategy, building upon the optimal configuration from each preceding part.

Different compression methods. Since our model leverages pre-trained weights, aligning the sequence length and feature dimensions of visual encoder to the pre-trained latent space is critical. We explore three distinct compression and mapping strategies: (1) MLP + truncation: The feature dimension is mapped via an MLP, while the sequence length is directly truncated; (2) VAE expansion: The number of Transformer layers in the VAE is directly increased to naturally match dimensions within the latent space; (3) MLP + MLP: Both the sequence length and feature dimension are projected via MLPs. As shown in Table 4(a), the MLP + MLP strategy performs best (18.12%). We hypothesize that simple truncation discards critical edge information, while merely expanding VAE layers increases optimization

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

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

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

Inputs Kontext OmniGen2 Qwen-Image-

- -

Nano Banana FlowInOne

Ground Truth

Edit-2509

-

(Ours)

###### Figure 4: Visual instruction editing comparison across methods.

- Table 4: Ablation study on compression method, modulation method, and training strategy. The average scores across Gemini3 Google [2025a], GPT5.2 OpenAI [2025] and Qwen3.5 Qwen Team [2026] are used as the evaluation metric.

(a) Compression Method

Method Gemini↑ GPT↑ Qwen↑ Avg. MLP + truncation 0.179 0.153 0.176 0.169 VAE expansion 0.169 0.147 0.155 0.157 MLP + MLP 0.192 0.170 0.185 0.182

(b) Modulation Method

Cross attention Gemini↑ GPT↑ Qwen↑ Avg. Wo CA 0.192 0.170 0.185 0.182 W Dual-Path CA 0.227 0.185 0.229 0.214 Dual-Path SAM 0.242 0.214 0.238 0.231

(c) Training strategy

Train strategy Gemini↑ GPT↑ Qwen↑ Avg. 2-stage training 0.336 0.256 0.283 0.291 Joint training 0.540 0.392 0.503 0.478

###### difficulty. Conversely, dual MLP projection effectively preserves the semantic and spatial structure of the input visual prompt while maintaining pre-trained priors.

Token gated cross attention. Next, we investigate modulation mechanisms to unify generation and editing, which exhibit distinct structural dependencies. We compare: (1) Wo CA: self-attention only; (2) W Dual-Path CA: dualpath cross-attention without adaptive gating; and (3) Dual-Path SAM: our proposed Dual-Path Spatially-Adaptive Modulation. Table 4(b) shows that lacking cross-attention (Wo CA) yields the poorest results (18.12%) due to insufficient utilization of source image structural priors. Adding cross-attention improves performance to 21.40%, while

Dual-Path SAM reaches 23.1%. This demonstrates that the adaptive gating mechanism dynamically balances content consistency and instruction adherence, optimizing performance within a unified framework.

Joint training vs. two-stage training. Based on the optimal architecture described above, we evaluate data training strategies. We compare: (1) Two stage training: 100k steps on 3M samples (T2I, C2I, coarse-grained editing), followed by 140k steps on the remaining 2M samples; and (2) Joint training: mixing all 5M samples for 240k steps. As Table 4(c) shows, joint training dominates with a 47.8% pass rate, far surpassing the two-stage approach (29.1%). Two-stage training likely suffers from catastrophic forgetting across varying tasks. Instead, joint training forces the model to simultaneously learn semantic generation, geometric transformation, and physical laws within a shared visual flow space, yielding stronger generalization and instruction adherence.

#### 6 Conclusion

We presented FlowInOne, a unified framework that redefines multimodal generation as a purely visual flow. By embedding all modalities into a shared visual space and learning continuous transport between visual instruction and image states, FlowInOne achieves efficient and consistent generation across diverse tasks. To support this paradigm, we introduced the large-scale VisPrompt-5M dataset, enabling cross-task generalization under a single visual interface. FlowInOne achieves state-of-the-art performance across all evaluated tasks, surpassing existing open-source models and competitive commercial systems in both automated and human evaluations. Extensive experiments demonstrate that FlowInOne establishes a foundation for future vision-centric multimodal models that unify perception and generation under a single deterministic principle. We believe this work marks a step toward closing the gap between visual understanding and creation within a continuous visual domain.

#### References

Elizabeth Salesky, David Etter, and Matt Post. Robust open-vocabulary translation from visual text representations. arXiv preprint arXiv:2104.08211, 2021.

Phillip Rust, Jonas F Lotz, Emanuele Bugliarello, Elizabeth Salesky, Miryam de Lhoneux, and Desmond Elliott. Language modelling with pixels. arXiv preprint arXiv:2207.06991, 2022.

Chenghao Xiao, Zhuoxu Huang, Danlu Chen, G Thomas Hudson, Yizhi Li, Haoran Duan, Chenghua Lin, Jie Fu, Jungong Han, and Noura Al Moubayed. Pixel sentence representation learning. arXiv preprint arXiv:2402.08183, 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, June 2022.

William S. Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2023.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023a.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Qing Qu, and Ivan Skorokhodov. Alphaflow: Understanding and improving meanflow models. arXiv preprint arXiv:2510.20771, 2025.

Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. In Advances in Neural Information Processing Systems, volume 37, pages 133345–133385, 2024.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pages 8821–8831, 2021.

Yufan Zhou, Bingchen Liu, Yizhe Zhu, Xiao Yang, Changyou Chen, and Jinhui Xu. Shifted diffusion for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10157–10166, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2024.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Adam Polyak et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2025.

Jiuhong Xiao, Roshan Nayak, Ning Zhang, Daniel Toertei, and Giuseppe Loianno. Thermalgen: Style-disentangled flow-based generative models for RGB-to-thermal image translation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Gabriel Nobis, Maximilian Springenberg, Arina Belova, Rembert Daems, Christoph Knochenhauer, Manfred Opper, Tolga Birdal, and Wojciech Samek. Fractional diffusion bridge models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, and Anima Anandkumar. I2sb: Image-to-image schrödinger bridge. In International Conference on Machine Learning, 2023b.

Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon. Denoising diffusion bridge models. In The Twelfth International Conference on Learning Representations, 2024.

Hebaixu Wang, Jing Zhang, Haoyang Chen, Haonan Guo, Di Wang, Jiayi Ma, and Bo Du. Residual diffusion bridge model for image restoration. arXiv preprint arXiv:2510.23116, 2025a.

Yidi Liu, Xueyang Fu, Jie Huang, Jie Xiao, Dong Li, Wenlong Zhang, Lei Bai, and Zheng-Jun Zha. Latent harmony: Synergistic unified uhd image restoration via latent space regularization and controllable refinement. arXiv preprint arXiv:2510.07961, 2025a.

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, June 2023.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. CVPR, 2023.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3813–3824, 2023.

Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters. arXiv preprint arXiv:2305.10855, 2023.

Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, and Kai Chen. Glyphcontrol: Glyph conditional control for visual text generation. Advances in Neural Information Processing Systems, 36, 2024.

Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH 2023 Conference Proceedings, 2023.

Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014. Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-

consistent adversarial networks. arXiv preprint arXiv:1703.10593, 2020.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024.

Ren Sucheng, Yu Qihang, He Ju, Shen Xiaohui, Yuille Alan, and Chen Liang-Chieh. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025.

Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Randomized autoregressive visual generation. arXiv preprint arXiv:2411.00776, 2024.

Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024.

Dongwon Kim, Ju He, Qihang Yu, Chenglin Yang, Xiaohui Shen, Suha Kwak, and Liang-Chieh Chen. Democratizing text-to-image masked generative models with compact text-aware one-dimensional tokens. arXiv preprint arXiv:2501.07730, 2025.

Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng YAN. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. In The Thirteenth International Conference on Learning Representations, 2025a.

Zhi-Kai Chen, Jun-Peng Jiang, Han-Jia Ye, and De-Chuan Zhan. Hawk: Leveraging spatial context for faster autoregressive text-to-image generation. arXiv preprint arXiv:2510.25739, 2025a.

Youwei Zheng, Yuxi Ren, Xin Xia, Xuefeng Xiao, and Xiaohua Xie. Dense2moe: Restructuring diffusion transformer to moe for efficient text-to-image generation. arXiv preprint arXiv:2510.09094, 2025.

Yu Chang, Jiahao Chen, Anzhe Cheng, and Paul Bogdan. Maskattn-sdxl: Controllable region-level text-to-image generation. arXiv preprint arXiv:2509.15357, 2025.

Jian Jia, Jingtong Gao, Ben Xue, Junhao Wang, Qingpeng Cai, Quan Chen, Xiangyu Zhao, Peng Jiang, and Kun Gai. From principles to applications: A comprehensive survey of discrete tokenizers in generation, comprehension, recommendation, and information retrieval. arXiv preprint arXiv:2502.12448, 2025.

zk. text-to-image-2m, 2024. URL https://huggingface.co/datasets/jackyhate/text-to-image-2M. Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,

Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211 – 252, 2014.

Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-edit-1.5m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025b.

Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing. arXiv preprint arXiv:2510.19808, 2025.

Keming Ye, Zhipeng Huang, Canmiao Fu, Qingyang Liu, Jiani Cai, Zheqi Lv, Chen Li, Jing Lyu, Zhou Zhao, and Shengyu Zhang. Unicedit-10m: A dataset and benchmark breaking the scale-quality barrier via unified verification for reasoning-enriched edits. arXiv preprint arXiv:2512.02790, 2025.

Weifeng Lin, Xinyu Wei, Renrui Zhang, Le Zhuo, Shitian Zhao, Siyuan Huang, Junlin Xie, Yu Qiao, Peng Gao, and Hongsheng Li. Pixwizard: Versatile image-to-image visual assistant with open-language instructions. arXiv preprint arXiv:2409.15278, 2024.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Nate Gillman, Charles Herrmann, Michael Freeman, Daksh Aggarwal, Evan Luo, Deqing Sun, and Chen Sun. Force prompting: Video generation models can learn and generalize physics-based control signals. arXiv preprint arXiv:2505.19386, 2025.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. arXiv preprint arXiv:1706.08500, 2018.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025b.

Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI Conference on Artificial Intelligence, 2022a.

Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. arXiv preprint arXiv:2207.12396, 2022b.

Rinon Gal, Or Patashnik, Haggai Maron, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clip-guided domain adaptation of image generators. ACM Trans. Graph., 41(4), 2022.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. arXiv preprint arXiv:2309.06380, 2024.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025b.

Qihao Liu, Xi Yin, Alan Yuille, Andrew Brown, and Mannat Singh. Flowing from words to pixels: A noise-free framework for cross-modality evolution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2755–2765, 2025c.

Alex Aizman, Gavin Maltby, and Thomas Breuel. High performance i/o for large scale deep learning. arXiv preprint

arXiv:2001.01858, 2020. Google. Gemini. https://deepmind.google/models/gemini/, 2025a. OpenAI. Chatgpt. https://chatgpt.com/, 2025. Qwen Team. Qwen3.5: Towards native multimodal agents, 2026. URL https://qwen.ai/blog?id=qwen3.5.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Google. Nano banana. https://aistudio.google.com/models/gemini-2-5-flash-image, 2025b. Zhuobai Dong, Junchao Yi, Ziyuan Zheng, Haochen Han, Xiangxi Zheng, Alex Jinpeng Wang, Fangming Liu, and

Linjie Li. Seeing is not reasoning: Mvpbench for graph-based evaluation of multi-path visual physical cot. arXiv preprint arXiv:2505.24182, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pages 8748–8763, 2021.

Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. arXiv preprint arXiv:2407.05282, 2024.

Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. arXiv preprint arXiv:2411.07199, 2024.

#### Appendix Overview

In the appendix, we provide additional information as listed below:

- Appendix A More experiment results p. 17

- A.1 Error Analysis on human evaluation 17
- A.2 Robustness test 18
- A.3 Visual instruction Ablation 19
- A.4 Inter-evaluator agreement statistics 19

- Appendix B More Benchmark Details p. 20

- B.1 Leakage prevention 20
- B.2 Data Analysis 21

- Appendix C Additional Qualitative Examples p. 22
- Appendix D More Dataset Details p. 22

- D.1 Overview of VisPrompt-5M 22
- D.2 Data construction 26
- D.3 Visual Text Rendering Pipeline 28
- D.4 Automated Quality Control and Filtering Pipeline 30
- D.5 Dataset Composition and Detailed Statistics 31

- Appendix E Limitations and future work p. 32
- Appendix F More evaluation details p. 34

- F.1 VLM evaluation 34
- F.2 Quantitative and human evaluation 35

- Appendix G Model details p. 37

- G.1 Parameter breakdown 37
- G.2 Loss function for image in-image out generation 37
- G.3 Hyperparameter ablation 38
- G.4 Experimental details 38

#### A More experiment results

###### A.1 Error Analysis on human evaluation

To complement our automated VLM-based evaluation and gain deeper insights into the failure modes of our model, we conducted a rigorous human evaluation. Ten independent expert evaluators independently assessed, and subsequently cross-checked, a stratified random subset of 250 generated samples across the benchmark (25 samples per subset). Note that the methodological difference between our human evaluation and the VLM evaluation: while the VLM assigns a continuous score (1-5) for every sample across all four dimensions, the human evaluators adopted a strict visual-inspection approach.

[Figure 229]

[Figure 230]

Specifically, evaluators first made a binary judgment on whether the generated image was qualified. If a sample was deemed a “FAIL,” the evaluator then recorded which specific dimensions contributed to the failure, allowing for multi-label tagging per sample. Please note that for brevity in the figures, the labels Fidelity, Spatial, Realism, and Consistency correspond strictly to Instruction Fidelity, Spatial Precision, Visual Realism, and Content Consistency, respectively.

Overall error types in VPBench (multi-label per sample)

As illustrated in Figure 5, the global error distribution across the entire VP-Bench reveals that Instruction Fidelity is the most prominent bottleneck, accounting for 56.2% of all tagged errors. This indicates that fully capturing the nuanced semantics of complex visual instructions remains the primary challenge. Spatial Precision constitutes the second largest error source

- at 20.5%, followed by Visual Realism (13.5%) and Content Consistency (9.7%).

Figure 5: Overall error types in VP-Bench. For brevity in the figures, the labels Fidelity, Spatial, Realism, and Consistency correspond strictly to Instruction Fidelity, Spatial Precision, Visual Realism, and Content Consistency, respectively.

- A more granular breakdown of error types by subset category is presented in Figure 6. The distribution of errors varies significantly depending on the nature of the specific task. For instance, in semantics-driven tasks that require strict adherence to explicit textual concepts and precise content generation, such as class-to-image and text bbox control, Instruction Fidelity errors overwhelmingly dominate, taking up 85.7% and 80.0% respectively. Conversely, in tasks requiring strict spatial grounding and geometric reasoning—such as doodles, force, trajectory, and vismarker—the proportion of Spatial Precision errors significantly increases, reaching up to 45.5% in doodle-guided tasks. Additionally, Visual Realism emerges as a more noticeable issue in from-scratch generation tasks like text-to-image (33.3%).

[Figure 231]

[Figure 232]

Error types for each subset in VPBench (multi-label per sample)

###### Figure 6: Error types for each subset in VP-Bench.

###### A.2 Robustness test

Origin Text style Text length Text blur Text random error

Origin Text style Text length Text blur Text random error Origin Text style Text length Text blur Text random error

Origin Text style Text length Text blur Text random error

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

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Input

Input

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

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Output

Output

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

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

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Input

Input

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

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

Output

Output

Figure 7: Qualitative results of the robustness analysis against visual instruction perturbations. We evaluate the model’s generation stability under five distinct conditions: the original unmodified input, variations in text style (e.g., size, color, font, and layout), changes in text length, text blurring, and random text corruption.

To evaluate the stability and reliability of our model in real-world scenarios, we conduct comprehensive robustness tests under various challenging conditions.

First, we assess the resilience of model to perturbations in the textual components of the visual instructions. As shown in Figure 7, we apply four distinct types of interference to the original input: changes in text style (including size, color, font, and spatial layout), variations in text length, severe text blurring, and random text corruption (replacing valid instructions with meaningless or incorrect text). The generation results demonstrate that our model maintains highly stable and accurate performance across style changes, length variations, and strong blurring. This indicates that the model effectively extracts the underlying semantic intent rather than simply memorizing superficial formatting. However, as expected, the performance of model significantly deteriorates when subjected to random text errors. This failure case actually serves as positive confirmation that our model strictly follows the explicit semantic guidance provided within the image, rather than hallucinating edits based on visual context alone.

Input 128×128 256×256 384×384 512×512

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

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

[Figure 351]

[Figure 352]

###### Figure 8: Generation results across different input image resolutions.

Furthermore, we evaluate the robustness of model across different resolutions. We test the generation quality using input images scaled to 128 × 128, 256 × 256, 384 × 384, and 512 × 512 pixels. As illustrated in Figure 8, the model is largely capable of producing high-quality generation at resolutions of 256 × 256 and above, despite occasional failure cases. A noticeable drop in visual fidelity and instruction-following capability is only observed at the extreme low resolution of 128 × 128. This is primarily because extreme downsampling severely compresses the visual markers and text, making them illegible for accurate feature extraction.

###### A.3 Visual instruction Ablation

To thoroughly investigate the individual contributions of textual and visual components within our unified visual instructions, we conducted an ablation study across four representative samples. Specifically, we evaluated the generation performance under four distinct input configurations: (1) Blank, where both text and visual prompts are removed, leaving only the original source image; (2) Text, where visual prompts are removed; (3) Visual prompt, where textual instructions are removed; and (4) Text + visual prompt, representing the complete visual instruction. As

Blank Text Visual prompt Text + visual prompt Blank Text Visual prompt Text + visual prompt

Blank Text Visual prompt Text + visual prompt Blank Text Visual prompt Text + visual prompt

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

[Figure 368]

Input

Input

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

[Figure 384]

Output

Output

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

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

Input

Input

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

Output

Output

- Figure 9: Qualitative results of the visual instruction ablation study. We compare generation outcomes across four input configurations: Blank (original image only), Text only, Visual prompt only, and the complete Text & visual prompt.

illustrated in Figure 9, the results clearly demonstrate the necessity of combining both instructional elements within the unified image canvas. When provided with the “Blank” image, the model predictably makes no modifications. In the “Text only” setting, the model primarily acts to erase the overlaid text but fails to execute the core editing intent, as it lacks precise spatial and operational grounding. Conversely, the “Visual prompt only” setting leads to chaotic and disordered generation results. This strongly proves that the in-image text is indispensable, serving as a lightweight semantic trigger that assigns explicit functional meaning to the visual prompts. Finally, only the complete “Text + visual prompt” configuration successfully and consistently generates the correct, high-quality target images, confirming that the synergy between rendered textual semantics and graphical spatial cues is essential for accurate visual instruction following.

###### A.4 Inter-evaluator agreement statistics

As shown in Table 5, despite absolute score variations caused by different VLM evaluators’ judging preferences and biases, the inter-evaluator agreement remains reasonably strong. Specifically, the Kendall’s W score of 0.7422, which measures the consistency of ordinal model rankings across evaluators, indicates that different evaluators produce largely aligned relative rankings among the compared models. Meanwhile, the Fleiss’ κ score of 0.6132, which reflects the agreement on binary pass/fail judgments, further suggests substantial consensus at the instance level. These results

demonstrate that although individual evaluators may assign different absolute success rates, the overall comparative conclusions are stable and not dominated by evaluator-specific biases.

###### Table 5: Inter-evaluator Agreement.

###### Model Fleiss’ κ Kendall’s W

FlowInOne 0.6537 0.7935 FLUX.1 Kontext 0.4540 0.5189 Nano Banana 0.4537 0.5251 OmniGen2 0.5361 0.5082 Qwen-IE-2509 0.4515 0.6115

Overall 0.6132 0.7422

#### B More Benchmark Details

- Table 6: Detailed statistics of the human-curated VP-Bench. To provide a structured overview, highly fine-grained editing sub-categories are conceptually grouped into higher-level semantic macro-categories.

Top-Level Task Sub-Category / Grouping Pairs Total Class-to-Image - - 100 Doodles Editing - - 50

Point Force 100

Force Understanding

150 Global Force 50

Text-to-Image - - 50 Text BBox Control - - 50

Semantic Operations (Add, Remove, etc.) 65

Attribute & Environment Modifications 70 Style & Artistic Transfer 20 Structural Tasks (Condition-to-Image) 65 Complex Spatial & Reasoning 70

Text-in-Image Editing

290

Trajectory Understanding - - 50

Removal Operations 40 Replacement & Swapping 120 320 Attribute & Local Changes 160

Visual Marker Editing

##### Total Benchmark Size 1,060 Pairs

###### B.1 Leakage prevention

Since VP-Bench is derived via random sampling from our eight task categories, we implemented a strict two-fold leakage prevention protocol to ensure a genuine zero-shot evaluation and defend against data contamination:

- • Root-Image Level Partitioning: Instead of splitting the dataset at the final instruction-pair level, the split is strictly executed based on the underlying root images (the unedited base canvases). This ensures the model has never observed the semantic backgrounds, objects, or spatial layouts of the benchmark during optimization.
- • Variant and Augmentation Exclusion via Visual Deduplication: To technically enforce this mutual exclusivity, we employed a rigorous visual feature deduplication pipeline. Specifically, we extracted deep visual

embeddings via CLIP Radford et al. [2021] for all root images sampled for VP-Bench. We then computed the cosine similarity against the entire 5M training pool. Any training pair exhibiting a visual similarity score above a highly conservative threshold was aggressively discarded. This feature-level filtering mathematically guarantees that no differently instructed pairs, cropped variants, or intermediate edits originating from the benchmark’s base canvases remain in the training phase.

###### B.2 Data Analysis

Overview of VP-Bench. As summarized in Table 6 and Figure 10, VP-Bench comprises a total of 1,060 meticulously curated image pairs spanning eight distinct task categories. The dataset is heavily anchored by two core visual instruction tasks: Visual Marker Editing (320 pairs, 30.19%) and Text-in-Image Editing (290 pairs, 27.36%). Together, these constitute over 57% of the benchmark, providing highly fine-grained evaluations across semantic operations, structural tasks, and complex Spatial Reasoning. Beyond general editing, VP-Bench is strategically structured to evaluate

- Table 7: Textual prompt statistics across different task categories in VP-Bench. The uniquely low average word counts prove the benchmark’s reliance on explicit visual markers rather than lengthy textual descriptions.

### Task Category Pairs Avg. Words Word Range Avg. Chars

Class-to-Image 100 3.0 1 – 6 22.4 Text BBox Control 43 3.5 3 – 5 16.0

Doodles Editing 43 12.0 12 – 12 63.0 Trajectory Understanding 50 14.0 14 – 14 79.0 Visual Marker Editing 312 14.1 3 – 32 74.3 Force Understanding 150 17.7 13 – 24 96.0 Text-in-Image Editing 292 17.9 2 – 44 105.0 Text-to-Image (Baseline) 50 23.7 8 – 46 137.1

specific advanced capabilities. To rigorously test physics-aware reasoning, we specifically include Force Understanding (150 pairs, 14.15%) and Trajectory Understanding (50 pairs, 4.72%). Additionally, the benchmark evaluates explicit spatial layout and sketch-based control through Text BBox Control (50 pairs, 4.72%) and Doodles Editing (50 pairs, 4.72%). Finally, to ensure a comprehensive evaluation spectrum, we incorporate Class-to-Image (100 pairs, 9.43%) and the baseline Text-to-Image (50 pairs, 4.72%) purely as foundational tasks. This precise proportional distribution ensures that the benchmark prioritizes complex visual grounding and physical reasoning over conventional text-driven generation. We deliberately designed this proportional distribution to ensure that subsets driven by explicit visual prompts heavily dominate the benchmark. This structural emphasis highlights our core objective: rigorously evaluating a model’s capacity for genuine visual-centric instruction following, thereby preventing models from bypassing visual grounding through conventional text-driven priors.

Additional statistics. A core motivation of VP-Bench is to shift the cognitive load from complex, dense textual descriptions to intuitive, explicit visual instructions. To quantitatively demonstrate that our benchmark evaluates genuine visual reasoning rather than textual comprehension, we analyze both the linguistic length and the semantic distribution of the textual prompts extracted in all images. First, as detailed in Table 7, the overall textual requirement is extremely lightweight. TTasks driven by straightforward semantic categories (e.g., Class-to-Image) or explicit spatial bounding boxes (e.g., Text BBox Control) utilize highly concise textual tags, averaging just 3.0 and 3.5 words, respectively. Even for complex physics-aware reasoning (Trajectory Understanding and Force Understanding) and intricate manipulation (Visual Marker Editing), the text remains remarkably brief (averaging between 14 to 18 words). Because such a brief text alone is insufficient to describe complex spatial configurations, the model is forced to extract the precise operational intent directly from the spatial geometry of the rendered boxes, doodles, and superimposed arrows. In contrast, the traditional Text-to-Image baseline relies on significantly longer descriptive prompts (averaging 23.7 words).

Furthermore, the semantic composition of the instructions corroborates this visual reliance. As illustrated in Figure 11, the top-20 global keywords diverge sharply from traditional generation prompts. Instead of dense descriptive adjectives, the vocabulary is dominated by abstract references (e.g., “object”, “image”), explicit visual pointers (e.g., “arrow”,

[Figure 417]

- Figure 10: Data distribution across the eight distinct subsets within VP-Bench.VP-Bench is a comprehensive benchmark that comprises eight distinct data types.

###### Top-20 Global Keywords in Benchmark Instructions

385

object

345

image

310

arrow

262

pointed

202

generate

[Figure 418]

350

137

change

124

turn

300

107

apply

- 100
- 101
- 102

according

250

Frequency

wind

final

200

89

swap

150

77

objects

69

color

100

69

replace

67

white

- 55
- 56
- 57
- 58

blue

add

pose

remove

0 50 100 150 200 250 300 350 400

Frequency

Figure 11: Top-20 global keywords in VP-Bench instructions. The distribution reveals a dominance of operational verbs (e.g., “change”, “turn”) and visual pointers (e.g., “arrow”), highlighting the minimal text dependency of our visual-centric evaluation paradigm.

“pointed”), and operational verbs (e.g., “change”, “turn”, “swap”). This stark statistical and linguistic contrast robustly validates that VP-Bench minimizes text dependency, utilizing text merely as a lightweight operational trigger while rigorously testing “image-in, image-out” visual instruction following.

#### C Additional Qualitative Examples

In this section, we provide additional qualitative examples to further demonstrate the versatile generation capabilities of our model across various visual instruction categories. Specifically, we group these supplementary results into three main aspects: (1) Figure 12 showcases extended results for text-to-image generation and text-in-image editing tasks. (2) Figure 13 provides further visual examples focusing on text bounding box (bbox) editing, doodle-guided editing, and visual marker-based editing. (3) Figure 14 illustrates additional generation outcomes that emphasize the model’s capacity for physical force understanding and trajectory understanding.

#### D More Dataset Details

###### D.1 Overview of VisPrompt-5M

To support the training of FlowInOne under a purely vision-centric paradigm, we constructed VisPrompt-5M, a meticulously curated large-scale dataset comprising approximately 5 million pairs of visual instructions. Unlike traditional multimodal datasets that rely on disjointed textual captions, VisPrompt-5M unifies diverse control signals by explicitly rendering task-specific instructional elements—such as text, spatial layouts, or physical constraints—directly onto the input image canvas.

- Table 8 provides a comprehensive statistical breakdown of VisPrompt-5M. To ensure FlowInOne acquires a versatile and generalizable visual reasoning capability, we scale the dataset across eight distinct fine-grained task categories. These categories can be conceptually grouped into three major multimodal capabilities:

- • Fundamental Generation (Text to Image, Class to Image): Constructed via direct rendering of textual prompts and class labels from massive datasets, establishing the basic semantic-to-visual mapping directly on the input images.
- • Unified Image Editing (Text in Image Edit, Text Bounding Box Edit, Visual Marker Edit, Doodles Edit): This overarching category spans a broad spectrum of manipulations. It ranges from general operational intents (e.g., attribute changes, inpainting, and condition-to-image tasks extracted from large-scale datasets) to precise,

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

[Figure 429]

[Figure 430]

Input

Input

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

Output

Output

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

Input

Input

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

Output

Output

[Figure 467]

[Figure 468]

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

Input

Input

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

[Figure 489]

[Figure 490]

Output

Output

###### Figure 12: Additional qualitative results for text-to-image generation and text-in-image editing tasks.

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

Input

Input

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

Output

Output

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

Input

Input

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

Output

Output

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

Input

Input

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

Output

Output

###### Figure 13: Additional qualitative results for text bounding box (bbox) editing, doodle-guided editing, and visual marker-based editing.

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

Input

Input

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

Output

Output

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

Input

Input

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

Output

Output

###### Figure 14: Additional qualitative results demonstrating physical force understanding and trajectory understanding.

spatially-constrained modifications. Notably, the subsets requiring explicit geometric priors and implicit visual cues (BBox, Visual Markers, and Doodles) are specifically synthesized and heavily curated through advanced MLLMs (e.g., Qwen3-VL) and rigorous manual inspection to guarantee high spatial fidelity.

• Physics Understanding (Trajectory Understanding, Force Understanding): The most challenging subset, pushing the boundary of physics-aware image-in, image-out generation. It incorporates manually annotated trajectory lines and explicit force arrows, visualizing precise object dynamics.

As illustrated in Table 8, our rigorous construction and filtering pipeline, spanning from raw data extraction to multistage MLLM synthesis and manual annotation—ensures that the model learns accurate spatial-visual logic rather than exploiting noisy dataset correlations.

Table 8: Comprehensive Overview of the VisPrompt-5M Dataset. We detail the task categories, original data sources, data scale before and after the filtering pipeline, construction methodology, and the explicitly embedded visual instruction formats.

Task Category Raw Source(s) Raw Size Retained Size Construction Pipeline Visual Instruction Format Text to Image text-to-image-2M zk [2024] 2.26M 2.26M Direct Rendering Text Class to Image ImageNet (Subset) Russakovsky et al. [2014] 2M 860K Direct Rendering Text (Class Labels)

GPT-Image-Edit Wang et al. [2025b] 1.5M 1.04M

Pico-Banana Qian et al. [2025] 400K 10K UnicEdit Ye et al. [2025] 2M 585K PixWizard Lin et al. [2024] 500K 315K Extracted

Extracted & Filtered

Text in Image Edit

Text

Text & BBox Edit GPT-Image-Edit subset Wang et al. [2025b] 45K 24K Synthetic Editing (MLLM guided) Text + Bounding Box Visual Marker Edit UltraEdit subset Zhao et al. [2024] and OmniEdit Wei et al. [2024] 400K 250K Synthetic Editing & Filtered Text + Arrow Marker Doodles Edit Web Crawled Images 5K 1K Two-stage Synthesis & Manual Text + Doodle Trajectory Understanding Blender-rendered Videos - 1.5K Manual Annotation Text + Trajectory Line Force Understanding Force Prompting Dataset Gillman et al. [2025] 36K 32K Keyframe Extracted & Annotated Text + Force Arrow

###### D.2 Data construction.

Fundamental Generation To align with our unified image-in, image-out paradigm, we must convert traditional text-image pairs into unified image-image pairs. For the text-to-image task, we directly render the corresponding textual prompts onto a blank input canvas. To ensure the model acquires robust visual text comprehension rather than overfitting to specific typographical layouts, we introduce extensive data augmentation during the rendering process. Specifically, the font style, font size, font color, text position, and the background color of the canvas are all randomly sampled. We also enforce a strict boundary check to ensure that all generated text remains entirely within the canvas limits. Through this automated pipeline, the 2M text-image pairs from text-to-image-2M zk [2024] are seamlessly transformed into image-image pairs.

Similarly, for the class-to-image generation task, we utilize a high-quality subset of ImageNet Russakovsky et al. [2014]. The discrete class labels (e.g., “golden retriever”) are extracted and rendered onto the input canvas using the identical randomized rendering strategy described above. Because ImageNet inherently contains multiple diverse target images for each class, a single rendered label canvas can be paired with various target images from the same category. This one-to-many pairing strategy naturally encourages the model to capture and generate intra-class diversity.

Text-in-Image Editing. This category serves as the cornerstone for our model’s instruction-following capabilities, systematically unifying semantic image manipulations and structural condition-to-image tasks into a single learning objective.

To cover a comprehensive spectrum of operational intents, we aggregate data from four major sources: GPT-ImageEdit Wang et al. [2025b], Pico-Banana Qian et al. [2025], UnicEdit Ye et al. [2025], and PixWizard Lin et al. [2024]. Instead of enumerating all fine-grained sub-tasks, we conceptually group the diverse editing capabilities into several core dimensions:

- • Semantic Operations: Including subject addition, removal, replacement, and object swapping.
- • Attribute & Environment Modifications: Covering local property changes (color, material, age/gender, facial expressions) and global atmospheric adjustments (lighting, weather conditions, background swapping).
- • Artistic & Style Transfer: Ranging from fundamental stylization to highly specific domain translations (e.g., 2D anime, Pixar-like 3D, sketch, line-art, and Western comic styles).
- • Complex Spatial Reasoning: Encompassing multi-object coordination, counting changes, outpainting, and pose adjustments.

While traditional paradigms treat structural condition-to-image generation (e.g., Depth-to-image and segmentation-toimage) or restoration tasks (inpainting, face/nature restoration) as distinct architectural branches, we argue that they naturally fall under the umbrella of “image editing”. Specifically, a structural condition map (such as a Canny edge map) is inherently a source image. By treating these spatial conditions as the starting canvas, we seamlessly integrate 315K high-quality pairs from PixWizard into our unified training pipeline.

To fully align with the purely vision-centric FlowInOne architecture, all operational intents must be explicitly embedded as visual text prompts. We transform the conventional heterogeneous triplet ⟨text instruction, source image, target image⟩ into a strictly visual ⟨input image, target image⟩ pair. This is achieved by directly rendering the textual instruction onto the source image (or condition map) being edited. Consistent with our fundamental generation strategy, we apply extensive randomized augmentations during this rendering process—randomly sampling the font style, text size, font color, and spatial placement on the canvas. This guarantees that the model learns to robustly perceive and parse visual text commands in varied scenarios rather than relying on fixed typographical shortcuts.

To maintain a high-quality optimization landscape for flow matching, we implement a strict filtering mechanism on the combined 2.5M raw semantic editing pairs, discarding complex or inconsistent pairs. Specifically, we filter out examples exhibiting: (1) highly ambiguous or overly convoluted textual instructions that lack explicit visual targets; (2) source images with severe generative artifacts or extreme aspect ratios; and (3) poor visual-semantic alignment, where the target image fails to faithfully reflect the specific operational intent dictated by the text. After filtering, we retain approximately 1.6M highly curated semantic editing pairs, yielding a total of 1.9M unified pairs when combined with PixWizard.

Text Bounding Box Editing. To endow the model with precise spatial control, we construct a high-quality subset where text and bounding boxes jointly dictate the generation. Rather than utilizing existing imperfect pairs, we sample 45K high-resolution, artifact-free, and aesthetically pleasing images from GPT-Image-Edit. These images serve solely as the initial unedited source images.

Two-Stage Vision-Guided Synthesis: The generation of valid image pairs follows a customized pipeline leveraging the combined priors of Qwen Image Edit Wu et al. [2025a] and Qwen3-VL Bai et al. [2025b]. First, we randomly define a set of target objects to be added. We then prompt Qwen Image Edit to seamlessly insert the specified object into the source canvas, generating the final target image. Subsequently, we employ Qwen3-VL as a visual grounding agent to precisely localize the newly inserted object within the target image. Based on these spatial coordinates, we draw a prominent bounding box directly onto the original source images and render text label of the object adjacent to the box. In this purely visual prompt configuration, the textual label explicitly dictates the semantic category, while the bounding box imposes strict geometric constraints defining the scale and location.

Automated Logit-Based Filtering: To ensure the highest data quality and eliminate the need for laborious manual inspection, we design an automated auditing mechanism powered by Qwen3-VL 7B. The auditor evaluates the generated pairs across three critical dimensions: instruction faithfulness, local visual consistency, and global background preservation. Crucially, rather than relying on simple binary outputs, the auditor computes a continuous confidence score based on output logits:

P(Yes) − P(No) P(No)

(10)

Score =

[Figure 611]

[Figure 612]

This formulation allows us to establish a strict, customizable threshold for data retention. When an edit is deemed unsuccessful (i.e., falls below the threshold), the auditor activates a Refinement Protocol. It identifies specific synthesis issues (e.g., color mismatch or texture corruption) and outputs a prompt starting with ROP (Refinement Output Prompt) to provide feedback for the generation pipeline. By applying this rigorous auditing standard, we filter out sub-optimal generations and ultimately retain 24K high-fidelity pairs that exhibit perfect spatial-visual alignment.

[Figure 613]

[Figure 614]

|[Figure 615]<br><br>[Figure 616]|
|---|

|[Figure 617]<br><br>[Figure 618]|
|---|

|[Figure 619]<br><br>[Figure 620]|
|---|

|[Figure 621]<br><br>[Figure 622]|
|---|

|[Figure 623]<br><br>[Figure 624]|
|---|

|[Figure 625]<br><br>[Figure 626]|
|---|

###### Figure 15: General data construct process.

Visual Marker Editing. Visual markers, such as arrows, offer a highly intuitive and efficient interface for human users. They enable precise spatial referencing without requiring explicit object names, which is particularly advantageous when the target object is ambiguous or difficult to describe. To construct this subset, we curate high-quality image pairs from UltraEdit Zhao et al. [2024] and OmniEdit Wei et al. [2024], focusing specifically on operation-centric sub-categories including removal, replacement, object swap, attribute modification, and color/local changes.

To automate the visual marker annotation, we first parse the original textual instructions to extract the specific target subject. We then employ Qwen3-VL to precisely ground this subject within the input image. Based on the spatial coordinates, a script dynamically renders an arrow on the source canvas. To prevent the model from overfitting to specific marker styles, the arrow’s color, size, and starting position are completely randomized, with the strict constraint

that its tip must point directly at the target object. Finally, the explicit object name in the instruction is systematically replaced with a generic visual reference (e.g., “the object pointed by the arrow”). To guarantee data quality, we subject these synthesized pairs to the exact same automated logit-based auditing mechanism detailed in the Text & BBox Guided Editing section. This rigorous filtering process ultimately yields 250K highly reliable pairs, forcing the model to explicitly comprehend semantics and spatial relationships through visual cues alone.

Doodles Editing. Doodles provide an intuitive interface for users to explicitly specify shape priors and spatial layouts. To construct this subset, we collect 5K high-quality web images to serve as unedited base canvases and predefine ten diverse object categories. We then employ a two-stage synthesis pipeline powered by Qwen Image Edit Wu et al. [2025a]. In the first stage, the model is prompted to insert 1 or 2 simple, hand-drawn style doodles from the predefined categories into the base canvas, forming the input image. In the second stage, Qwen Image Edit transforms these abstract doodles into photo-realistic objects, yielding the corresponding target image.

Due to the inherent generative instability of multi-stage image-to-image translation, a significant portion of the initial outputs suffer from structural errors. Therefore, we conduct a rigorous manual inspection. The retention criteria strictly dictate that: (1) there must be an absolute absence of generative artifacts; (2) the doodles must remain structurally simple and abstract, rather than prematurely resembling real objects; (3) the synthesized photo-realistic objects must perfectly align with the shape priors defined by the doodles; and (4) the non-edited regions must maintain perfect pixel-level consistency with the input image. This meticulous manual curation filters out the majority of failure cases, ultimately yielding 1K high-fidelity pairs.

Force Understanding. To endow FlowInOne with physics-aware reasoning, we utilize the Force Prompting dataset Gillman et al. [2025], which encompasses two distinct physical paradigms: point forces and global forces. Specifically, the point force subset captures the linear kinematics of spherical objects and the harmonic oscillations of plants, while the global force subset simulates aerodynamic effects, such as wind acting on flags. To transform these continuous video dynamics into static image-to-image reasoning pairs, we implement a systematic, physics-driven keyframe extraction strategy. We designate the first frame of each video as the initial state canvas. For the target outcome, rather than arbitrary sampling, we employ specific analytical strategies (e.g., optical flow analysis) to determine the temporal point of kinematic convergence—a statistically determined frame where the dynamic systems reliably reach a terminal state, steady equilibrium, or maximum physical displacement.

Finally, we explicitly visualize the underlying physical parameters. The original dataset provides precise annotations: the exact spatial coordinate of the applied force, the force angle, and a normalized magnitude scalar m ∈ [0,1]. Utilizing an automated script, we render explicit force arrows directly onto the initial frame. For point forces, the arrow originates precisely at the application coordinate. For global forces, it is rendered as a global environmental indicator. The arrow’s geometric length is strictly proportional to the magnitude m, and its orientation aligns with the physical angle. This fully annotated initial frame forms our input image, explicitly challenging the model to predict the corresponding steady-state target image strictly governed by the visualized force dynamics.

Trajectory Understanding. To further explicitly model physics-aware motion priors, we construct a highly curated trajectory understanding subset. We first manually render a collection of high-fidelity dynamic videos featuring car and ball movements using the Blender 3D engine, encompassing both linear and complex curved kinematics.

To formulate the static image-to-image pairs, we extract the initial and terminal frames of each video. Our manual annotation protocol requires human annotators to draw a solid, continuous line directly onto the initial frame, accurately tracing the exact future geometric path of the moving object. This meticulously annotated frame serves as the input image, while the original terminal frame (showing the object at its final destination) acts as the corresponding target image. Crucially, to ensure the model strictly learns the spatial geometric path rather than overfitting to specific visual artifacts, the thickness and color of the drawn trajectory lines are entirely randomized during the annotation process. This rigorous manual pipeline ultimately yields 1.5K highly precise pairs for motion trajectory generation.

###### D.3 Visual Text Rendering Pipeline

To synthesize high-fidelity and diverse visual instruction data, we propose a robust, automated text-rendering engine. The pipeline is designed to dynamically adapt to various text lengths and underlying background constraints while strictly preserving legibility and geometric alignment. The rendering process comprises five pivotal stages:

- 1. Robust Font Selection and Glyph Validation. To ensure the generative robustness of the text rendering, we implement a dynamic font-picking mechanism. Given an input text sequence, the engine first validates character support by parsing the TrueType font’s cmap tables. To prevent the rendering of corrupted or “missing glyph” boxes (often caused by incomplete font files), we introduce an empirical glyph-area validation threshold. Let Ac be the bounding

Algorithm 1 Adaptive Bounding-Box Layout via Binary Search Input: Token sequence T, Target bounding box dimensions W × H, Font size search space [Smin,Smax] Output: Optimal font size S∗, Layout configuration L∗ (or Null if infeasible)

- 1: low ← Smin, high ← Smax
- 2: S∗ ← Null, L∗ ← Null
- 3: while low ≤ high do
- 4: mid ← ⌊(low + high)/2⌋
- 5: Ltmp ← TryWordWrap(T,W,mid) ▷ Greedy wrap within width W
- 6: if Ltmp ̸= Null then ▷ Valid width: No single token exceeds W
- 7: Htmp ← CalculateTotalHeight(Ltmp,mid)
- 8: if Htmp ≤ H then ▷ Valid height: Text fits entirely within H
- 9: S∗ ← mid ▷ Update optimal states
- 10: L∗ ← Ltmp
- 11: low ← mid + 1 ▷ Attempt to maximize readability (larger font)
- 12: else
- 13: high ← mid − 1 ▷ Height overflow, require smaller font
- 14: end if
- 15: else
- 16: high ← mid − 1 ▷ Width overflow, require smaller font
- 17: end if
- 18: end while
- 19: return S∗,L∗

box area of a rendered character c and s be the font size. A font is deemed robust and selected only if the average active pixel ratio N1 c∈T Ac

s2 exceeds a predefined minimum threshold, guaranteeing high-quality typographic representation across millions of synthesized pairs.

- 2. Semantic-Aware Tokenization. Handling multi-lingual instructions requires precise line-breaking strategies. We utilize a custom tokenization algorithm tailored for visual layouts. Characters are isolated as individual tokens to allow flexible word wrapping, whereas Latin alphanumeric sequences and symbols are grouped as cohesive whole-word tokens. This strategy prevents improper truncations of Western words at the end of a line, strictly preserving the semantic readability of the visual prompt.
- 3. Adaptive Bounding-Box Layout Algorithm. To automatically determine the optimal typographic layout within a constrained visual canvas, we model the layout generation as a constrained optimization problem. Given a target

bounding box with dimensions W × H, our goal is to find the maximum font size smax that accommodates the tokenized sequence T without overflow. We solve this efficiently in O(log N) time using a binary search algorithm

over the font size space [smin,smax]. For instances with extensive token counts, the engine defaults to utilizing the maximum available canvas margin. For shorter instructions, we introduce spatial diversity by randomizing the location and dimensions of the localized bounding boxes, thereby forcing the generative model to understand text instructions across arbitrary spatial distributions.

- 4. Context-Aware Stylization and Alpha Compositing. To guarantee text legibility regardless of the underlying visual content, we integrate a context-aware color contrast mechanism. Before rendering, the engine calculates the perceptual luminance L of the underlying image region bounded by the text block:

L = 0.299µR + 0.587µG + 0.114µB (11)

where µR,µG,µB denote the mean channel intensities of the cropped background. If the local background is heavily illuminated (L > 128), the engine applies dark text fill coupled with a thick white stroke; conversely, it utilizes bright text with a dark stroke for low-luminance regions. The stroke width is dynamically scaled based on the calculated line height. Finally, the text is rendered onto a dedicated transparent RGBA layer and seamlessly merged with the base canvas using alpha compositing, eliminating visual artifacts along the font anti-aliasing edges.

- 5. Analysis of the Layout Algorithm. The proposed adaptive layout strategy (Algorithm 1) provides several critical advantages for large-scale data synthesis:

- • Computational Efficiency: Traditional text rendering engines often rely on a linear step-down approach (iteratively decreasing font size until the text fits), yielding a time complexity of O(Smax − Smin). By

- formulating the layout process as a binary search optimization, we reduce the complexity to O(log(|Smax − Smin|)). This logarithmic efficiency is paramount when dynamically rendering over 5 million high-resolution image pairs, significantly accelerating the data generation pipeline.
- • Robust Fallback Mechanism: For extreme edge cases—such as exceptionally long instructions or single

words that exceed the randomized bounding box width W even at Smin—the algorithm seamlessly triggers a global fallback. Instead of discarding these valuable data points, the engine automatically re-initializes the target dimensions W ×H to the maximum safe canvas margins. This hierarchical container strategy guarantees a near 100% layout success rate, preventing long-tailed complex instructions from being systematically filtered out.

- • Spatial Variance as Implicit Augmentation: By stochastically sampling the initial dimensions (W,H) and the starting anchor coordinates (X,Y ) rather than always utilizing the full canvas, we introduce vast spatial diversity. This design forces the downstream generative model to learn robust spatial grounding and positional alignment, ensuring that the model adheres to precise geometric constraints rather than simply memorizing centered, full-screen text overlays.

###### D.4 Automated Quality Control and Filtering Pipeline

This section provides a granular breakdown of the automated filtering pipelines, the statistical properties of the curated data, and the rigorous evaluation protocols employed in our benchmark.

Given that our raw image pairs are sourced from diverse public datasets (yielding highly variable initial quality), we implement a rigorous, multi-stage filtering pipeline. Following the large-scale visual text rendering detailed in Section D.3, the generated images undergo a comprehensive inspection to guarantee visual fidelity, text legibility, and data diversity.

- 1. OCR-based Legibility Verification. To ensure the synthesized text is completely legible and free from truncation or rendering artifacts (e.g., overlapping bounding boxes or corrupted glyphs), we deploy an Optical Character Recognition (OCR) engine as the first filter. Let Tsrc denote the original instruction text and Tocr denote the text extracted from the rendered canvas Iv. We compute the Character Error Rate (CER) and filter out pairs where the error exceeds a stringent threshold τocr:

CER(Tsrc,Tocr) =

S + D + I

N ≤ τocr (12) where S,D, and I are the number of substitutions, deletions, and insertions, respectively, and N is the total number of characters in Tsrc. Images failing this check are discarded to prevent the model from learning corrupted visual instructions.

- 2. Task-Specific VLM Quality Inspection. Images that pass the OCR check are subsequently evaluated by an advanced Multimodal Large Language Model (MLLM, e.g., Qwen3-VL). To handle the diverse nature of our generative tasks, we design task-specific prompts. The VLM acts as a judge, outputting a boolean decision based on customized criteria:

- • Fundamental Generation: “Does the main subject in the image perfectly align with the embedded text prompt: [PROMPT]?”
- • Spatial Constraints (BBoxes/Markers): “Is the object precisely located within the red bounding box/indicated by the visual arrow?”
- • Physics-Aware Operations: “Does the motion blur or trajectory accurately reflect the directional force specified by the vector arrows?”

Only pairs that receive a positive confirmation across both semantic alignment and visual realism are retained.

- 3. Diversity-Oriented Deduplication. To maximize the informational entropy of the dataset and prevent mode collapse during training, we apply a diversity-oriented filtering mechanism. We extract CLIP image embeddings Eclip(I) for all candidates within a specific sub-task. A candidate Ii is retained only if its cosine similarity with all previously accepted images Ij in the active pool P remains below a diversity threshold τdiv:

Eclip(Ii) · Eclip(Ij) ∥Eclip(Ii)∥∥Eclip(Ij)∥

< τdiv (13)

max

Ij∈P

This strategy effectively prunes redundant concepts, ensuring a highly diverse data distribution.

###### Force & Trajectory Generation Tasks

###### Text-in-Image Editing Tasks(GPT-Image-Edit)

199,436

Attribute Modification

180,478

Addition

130,771

Env

Balls Poke

11,950

115,605

Object Swap

88,998

Removal

58,474

Generate

56,281

Swap

54,460

Edit

36,292

Complex Edit

17,735

Background Swap

Plants Poke

10,948

10,913

Others

- 10,875
- 10,876
- 10,877

Change Color

Replace

Turn

10,873

Transform Local

10,861

Change Global

10,842

Transform Global

Wind

9,881

10,807

Change Local

10,572

Add

7,166

Style

0 2000 4000 6000 8000 10000 12000 14000

0 50000 100000 150000 200000

Number of Image Pairs

Number of Image Pairs

- Figure 16: Detailed instance distribution of VisPrompt-5M. Left: The Force & Trajectory Generation subset, highlighting strictly curated physics-aware categories (e.g., wind, object poking) designed to impart dynamic kinematic priors. Right: The Text-in-Image Editing subset (derived from GPT-Image-Edit), demonstrating a natural long-tailed distribution of semantic operations. Ranging from high-frequency attribute modifications to specialized style transfers, this structural diversity ensures robust generalization across both complex stylistic changes and physical constraints.

###### D.5 Dataset Composition and Detailed Statistics

Through the aforementioned rendering and rigorous filtering pipeline, we curated a final dataset of approximately 5M high-quality image pairs. Table 9 details the macro-level composition, primary sources, and final retention volumes for each major task category. To further illustrate our stringent quality control, Table 10 provides a breakdown of the filtering survival rates across selected complex generative categories.

Beyond macroscopic volumes, analyzing the intra-category distributions is crucial for understanding the structural diversity of VisPrompt-5M. As illustrated in our distribution figures (refer to Figure 16, 17, 18, 19), the curated dataset exhibits a multi-granularity structure tailored to impart distinct generative priors to the model:

- 1. Broad Semantic and Stylistic Coverage (Long-Tailed Nature). Our text-in-image editing subsets—derived heavily from UnicEdit, GPT-Image-Edit, and PicoBanana—encompass a massive spectrum of user intents. UnicEdit and GPT-Image-Edit contribute the bulk of the volume, dominated by high-frequency operations such as Color Alteration (∼203K), Attribute Modification (∼199K), and Subject Addition (∼150K). Interestingly, the data naturally exhibits a long-tailed distribution (e.g., rare tasks like Object Extraction or Counting Change in UnicEdit contain fewer than 100 pairs). Rather than artificially flattening this curve, we intentionally preserve this long-tailed property to reflect real-world human editing priors. Furthermore, PicoBanana injects extreme stylistic diversity, contributing 35 highly specialized, fine-grained categories (e.g., Simpsonize, Vintage Filter, Outpainting), ensuring the model’s robustness against complex, composite textual instructions.
- 2. Spatial Reasoning and Region-Aware Constraints. While text instructions govern semantic changes, visual and geometric inputs dictate spatial precision. Our structured editing subsets (PixWizard and VisMarker) serve this exact purpose. The VisMarker subset provides highly balanced, region-aware supervision across 8 core categories (e.g., Object Swap, Removal, ranging consistently between 33K and 60K pairs), forcing the model to strictly respect local visual markers rather than applying global stylistic shifts. Concurrently, the PixWizard subset injects dense structural conditions, containing robust distributions of Face Restoration (∼49K) and Image-to-Sketch (∼52K), which train the model to understand dense spatial mappings like bounding boxes and segmentation masks unified within the input canvas.

###### Text-in-Image Editing Tasks(PicoBanana)

###### Text-in-Image Editing Tasks(PixWizard)

1,261

Add New Scene Context Background

52,117

Image2Sketch

1,032

Strong Artistic Style Transfer E G Van Gog...

884

Photo To Cartoon Sketch Comic

49,296

Face Restoration

833

Zoom In

761

Outpainting Extend Canvas Beyond Boundaries

42,876

Image2Canny

684

Modern Historical Style Look

676

Replace One Object Category With Another

40,604

Face Inpaint

546

Apply Seasonal Transformation Summer Winter

466

Add A New Object To The Scene

26,440

Image2Seg

455

Change Overall Color Tone Warm Cool

432

Add Film Grain Or Vintage Filter

18,255

Grounding Seg

417

Change The Size Shape Orientation Of An Ob...

409

Change An Object S Attribute E G Color Mat...

18,194

Grounding Seg Mask

347

Adjust Global Lighting Golden Hour Fluores...

16,825

336

Change Weather Conditions Sunny Rainy Snowy

Grounding Box

315

Remove An Existing Object

16,018

Nature Restoration

177

Replace Text In Signs Posters Billboards

169

Relocate An Object Change Its Position Spa...

- 7,630
- 8,667

Lineart2Image

139

Add New Handwritten Printed Etc Text

99

Funko Pop Style Toy Figure Of The Person

Normal2Image

94

Change Font Style Or Color Of Visible Text...

81

Translate Written Text Into Other Languages

- 4,018
- 5,780

Canny2Image

67

Pose Tweak Minor Plausible Change

60

Change Age Gender

Hed2Image

57

Simpsonize The Person Yellow Skin Cartoon ...

37

Convert Person To Pixar Disney Like 3D Car...

2,947

Seg2Image

35

Caricature With Mild Feature Exaggeration ...

33

Lego Minifigure Rendition Of The Person

2,600

Depth2Image

28

Clothing Edit Change Color Outfit

22

Modify Expressions Smile Frown Neutral

2,396

Pose2Image

20

Convert Person To Western Comic Cel Shaded...

17

Sticker Ify The Person With Bold Outline A...

489

Inpainting

- 16
- 17

Add Remove Replace Accessories Glasses Hat...

Convert Person To 2D Anime Manga Style Ide...

121

Outpainting

13

Line Art Ink Sketch Of The Person

0 200 400 600 800 1000 1200 1400

0 10000 20000 30000 40000 50000 60000

Number of Image Pairs

Number of Image Pairs

- Figure 17: Fine-grained structure and stylized distributions of VisPrompt-5M. Left: The Structured Editing subset (derived from PixWizard), highlighting dense spatial translations such as Image-to-Sketch and Face Restoration. This subset trains the model to strictly adhere to geometric and structural conditions. Right: The Text-in-Image Editing subset (derived from PicoBanana), detailing 35 highly specialized, long-tailed aesthetic operations. This extreme categorical diversity ensures the model’s proficiency in handling nuanced, localized, and composite textual instructions.

- 3. Physics-Aware and Kinematic Dynamics. A uniquely challenging component of VisPrompt-5M is the Force & Trajectory generation subset. While smaller in scale compared to semantic edits (comprising specifically curated classes like balls_poke at ∼11K and wind at ∼9K), this subset is of exceptionally high fidelity. It forces the image-to-image paradigm to step beyond static pixel manipulation and understand dynamic kinematic priors, translating explicit visual force vectors (arrows and magnitudes) into physically plausible consequences like motion blur, structural deformation, and trajectory extrapolation.

In summary, the statistical distribution of VisPrompt-5M is purposefully engineered. The massive text-in-image editing pairs provide a robust semantic foundation, the structured marker datasets enforce spatial discipline, and the curated physics subset unlocks novel dynamic capabilities, collectively empowering a single model to master multi-modal, instruction-driven image generation.

#### E Limitations and future work

While our model introduces a promising unified paradigm for visual instruction following, we acknowledge several limitations in the current framework. First, although the model demonstrates strong performance on our benchmark, its generalization capabilities in highly complex, unconstrained scenarios remain somewhat limited. This is primarily bounded by our current model capacity (1.2B parameters) and the scale of the training dataset. Second, due to computational constraints during training, the output generation is currently restricted to a fixed spatial resolution of 256 × 256 pixels, which may not fully satisfy the demands of high-fidelity creative workflows. Finally, our approach is currently optimized for single-turn instruction execution, and its potential for continuous, multi-turn interactive editing has yet to be fully explored. In future work, we aim to scale up both the model parameters and the training data to handle increasingly complex scenarios, further optimize our framework to support high-resolution generation, and extend our visual-centric paradigm to facilitate seamless multi-turn visual editing scenarios.

###### Text-in-Image Editing Tasks(UnicEdit10M)

###### VisMarker Region-Aware Edits

203,326

Color Alteration

Object Swap

59,960

150,653

Subject Addition

74,001

Background Change

57,037

Spatial Reasoning Edits

Removal

59,633

30,155

Material Modification

20,312

Portrait Editing

Swap

54,767

17,637

Texture Editing

13,068

Multi Object Coordination

8,676

Shape Size Alteration

Attribute Modification

33,439

3,581

Compound Operation Edits

3,543

Subject Replacement

Replace

11,072

1,230

Text Modification

1,169

Motion Change

1,060

Tone Transformation

Change Color

11,072

158

Subject Removal

158

Style Transfer

Turn

- 10,969
- 11,071

58

Relation Change

21

Counting Change

9

Viewpoint Transformation

Change Local

4

Object Extraction

0 50000 100000 150000 200000 250000

0 10000 20000 30000 40000 50000 60000 70000

Number of Image Pairs

Number of Image Pairs

- Figure 18: Semantic diversity and region-aware distributions of VisPrompt-5M. Left: The UnicEdit10M Diverse Edits subset, showcasing a profound long-tailed distribution. It spans high-frequency semantic operations like Color Alteration and Subject Addition to rare edge cases such as Object Extraction. Right: The VisMarker Region-Aware Edits subset, featuring robust, high-volume spatial operations like Object Swap and Removal. This combination ensures the model learns broad semantic reasoning while maintaining strict adherence to localized visual markers.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- Figure 19: Volume of ancillary generation and structural datasets. This logarithmic lollipop chart illustrates the extreme variance in scale and specialization across our supplementary sources. It highlights the massive foundation of over 2.26 million general Text-to-Image pairs, balanced by broad categorical coverage (ImageNet21K) and highly specialized, precise structural tasks such as Text Bbox Edit and Doodles. This multi-scale composition ensures the model achieves both robust open-domain generative capabilities and fine-grained execution of explicit visual constraints.

- Table 9: Comprehensive breakdown of the VisPrompt-5M composition. The dataset aggregates various public sources, strictly refined through our multi-stage OCR and VLM filtering pipeline.

Task Category Primary Source(s) Initial Volume Curated Pairs Fundamental Gen. (Text) Text-to-Image-2M zk [2024] 2,269,892 2,269,892 Fundamental Gen. (Class) ImageNet Subset Russakovsky et al. [2014] 2,000,000 860,000 Text-in-Image Editing GPT-Image-Edit, Pico-Banana, UnicEdit 2,100,000 1,601,000 Structured Editing PixWizard Lin et al. [2024] 350,000 315,000 Bounding Box Editing GPT-Image-Edit (Filtered) 45,000 24096 Visual Marker Editing Qwen3-VL Synthesized 300,000 250,000 Doodles Editing Web Crawled Images 5,000 1,007 Trajectory Understanding Blender Renders 1,600 1,513 Force Understanding Force Prompting Gillman et al. [2025] 36,000 25,820 Total - ∼7.12M ∼5.36M

- Table 10: Filtering survival rates across selected complex generative tasks, demonstrating the rigorousness of our automated OCR and VLM inspection.

###### Category Primary Rejection Reason VLM/OCR Filtering Retention Rate

Text-in-Image Editing Semantic Inconsistency VLM Semantic Check 78.2% Bounding Box Editing Geometric Misalignment VLM Spatial Check 53.3% Doodles Editing Generative Instability VLM Realism Check 20.0% Visual Marker Editing Generative Instability VLM Spatial Check 83.3%

#### F More evaluation details

###### F.1 VLM evaluation

To ensure a comprehensive, objective, and reproducible assessment of visual instruction following, we design a systematic evaluation pipeline driven by Vision-Large Language Models (VLMs). As illustrated in Figure 20, our evaluation process takes three primary inputs: the source image (categorized into Case A for text-only canvases and Case B for annotated real-world images), the generated output image, and a plain text generation instruction. This text instruction is explicitly extracted from the source image to prevent the VLM judge from making incorrect judgments due to inherent Optical Character Recognition (OCR) errors in the image.

The VLM evaluates the generated images across four distinct criteria on a 1-5 scale:

- • Instruction Fidelity: Measures the semantic precision of the generated result (e.g., matching objects, attributes, and actions) in responding to the core generation instruction.
- • Content Consistency: For generation tasks (Case A), this evaluates canvas cleanliness. For editing tasks (Case B), it strictly checks for the preservation of unedited background regions and the successful removal of the original visual markers and text instructions.
- • Visual Realism: Assesses the overall image quality, penalizing conspicuous artifacts, blurriness, or jagged edges to ensure natural blending.
- • Spatial Precision: Evaluates whether the generated objects are complete and strictly confined within the spatial boundaries indicated by visual markers (e.g., bounding boxes) or follow specific directional arrows.

Based on these four scores, the evaluator outputs a structured JSON response containing a concise analysis and a final verdict of either PASS or FAIL. To achieve a PASS, a generated image must meet three strict conditions: an Instruction Fidelity score of ≥ 3.0, an overall average score of ≥ 3.0, and no single dimension scoring ≤ 2.0.

For full transparency, the exact evaluation prompts detailing the role, criteria, and scoring rubrics used by the VLM evaluators are provided in Figure 21. Furthermore, to guarantee a fair and standardized comparison across different models during the initial image generation phase, e evaluate each baseline using its optimal native interface, where the image inputs are supplemented with detailed text prompts expanded by Qwen3-VL.

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

arXiv Template A PREPRINT

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

Inputs

###### Scoring & Output Format

Evaluation Criteria

Source Image Case A Case B

- 1. Instruction Fidelity

- 2. Content Consistency

- 3. Visual Realism

- 4. Spatial Precision

Requirements

Case A Case B

The semantic precision of the generated result in

Analysis: Describe whether the

|[Figure 654]<br><br>[Figure 655]<br><br>[Figure 656]|
|---|

|[Figure 657]<br><br>[Figure 658]<br><br>[Figure 659]|
|---|

responding to the generation instruction.

generated image follows the instruction fidelity, content consistency, visual realism and spatial precision

Checkpoints

###### Checkpoints

###### Checkpoints

Objects Attributes Actions

[Figure 660]

[Figure 661]

[Figure 662]

Object that matches

[Figure 663]

[Figure 664]

[Figure 665]

instruction description

……

……

……

Score: 1-5 scale in four dimension

-

Text-Only Visual prompt

-

###### Case A Case B Checkpoints

###### Case A Case B Checkpoints

###### Case A

###### Case B

###### Checkpoints

Final Verdict Rules

Preservation of the background and nonedited areas Removal of

Generation Instruction

-

PASS Or FAIL

Text descriptions extracted from the source image help prevent the model from making incorrect judgments due to text recognition errors in the image.

Canvas

Background

[Figure 666]

[Figure 667]

[Figure 668]

instructions/markers

cleanliness

preservation

……

……

……

Instruction Fidelity >= 3.0 Average Score >= 3.0

No score <= 2.0

###### The quality of the image

###### Checkpoints

###### Checkpoints

###### Checkpoints

Conspicuous artifacts Blurriness Jagged edges ……

###### Output Image

Natural Blurry

The result generated by the model.

……

……

{ "analysis": "Concise analysis",

|[Figure 669]<br><br>[Figure 670]<br><br>[Figure 671]|
|---|

|[Figure 672]<br><br>[Figure 673]<br><br>[Figure 674]|
|---|

"fidelity_score": 0,

"consistency_score": 0, "realism_score": 0, "spatial_score": 0, "final_verdict": "PASS" or "FAIL" }

###### Case A Case B

###### Case A Case B Checkpoints

###### Case A

###### Case B

###### Checkpoints

###### Checkpoints

|[Figure 675]<br><br>[Figure 676]<br><br>[Figure 677]|
|---|

The generated content strictly confined within the boundaries of the visual markers ……

The object is complete and within the frame

Object confined in the bbox

Object follow the arrow

……

……

Figure 20: Evaluation process.

###### F.2 Quantitative and human evaluation

Building upon Section 3.2, we elaborate on the formulation of our tailored quantitative metrics and the human evaluation protocol.

###### F.2.1 Quantitative Metrics Formulation

Standard image generation metrics are inadequate for evaluating fine-grained, instruction-following visual edits. Therefore, we rigorously formulated the following similarity constraints:

Directional CLIP Similarity: To capture the semantic transition from the input to the generated image, we utilize the directional CLIP feature space. Let csrc and ctgt be the MLLM-generated captions for the input image Iin and the generated image Igen, respectively. Using the CLIP text encoder ET and image encoder EI, the metric is defined as:

(ET(ctgt) − ET(csrc)) · (EI(Igen) − EI(Iin)) ||ET(ctgt) − ET(csrc)|| ||EI(Igen) − EI(Iin)||

(14) This measures how well the visual change aligns with the textual description of the edit.

Simdir =

DINOv3 Directional Similarity (DINOv3 Sim): While CLIP captures high-level semantics, DINOv3 is highly sensitive to dense spatial and physical structural changes. Let ϕ(·) denote the dense feature extractor of DINOv3. We compute the cosine similarity of the edit displacement vectors between the generated output and the ground truth Igt:

(ϕ(Igen) − ϕ(Iin)) · (ϕ(Igt) − ϕ(Iin)) ||ϕ(Igen) − ϕ(Iin)|| ||ϕ(Igt) − ϕ(Iin)||

(15)

SimDINOv3 =

A higher SimDINOv3 strictly indicates that the model has accurately applied the spatial and physical transformations requested by the visual prompt, without introducing unintended background artifacts.

###### F.2.2 Human Evaluation Protocol

To complement the automated VLM-based assessments, we conducted a rigorous human evaluation involving 10 independent expert evaluators. We selected a stratified random subset of 250 diverse pairs from VP-Bench, ensuring

Evaluation Prompts

[Figure 678]

###### Role

You are an insightful expert judge in the fields of computer vision and image generation. Your task is to evaluate the quality of "visual instruction-based image generation" tasks.

Inputs

You will be provided with the following three pieces of information:

- • Source Image: The original input image. This primarily falls into two categories:

- • Case A (Text-Only Source Image): The image has a uniform background and contains only text (e.g., "draw a cat"). In this scenario, please disregard the source image entirely.
- • Case B (Annotated Real-World Image): The image is a natural scene overlaid with textual instructions or visual markers (e.g., bounding boxes, arrows, scribbles, highlights). Please note that the overlaid text serves as the editing instruction. When evaluating consistency, ignore the disappearance of the text itself and focus on the coherence of the underlying image content.

- • Generation Instruction: This is a plain text instuction extracted from the source image.

• It serves as your sole semantic basis for judging the generated content.

- • Output Image: The final result generated by the model.

Evaluation Criteria

Please conduct a flexible evaluation based on the source image type (Case A or Case B) across the following four dimensions:

- 1. Instruction Fidelity

- •General Objective: The semantic precision of the generated result in responding to the generation instruction.
- •Checkpoints: Do the core objects, attributes (color, material), and actions described in the instruction accurately appear in the generated image?
- •Critical Judgment: If the generated content is irrelevant to the text description, this dimension must receive a strictly low score.

- 2. Content Consistency (Non-Edited Areas)

- •Case A (Text-Only Source Image):

- • Objective: Canvas cleanliness.
- • Checkpoints: Is the generated subject clear? Is the background clean or logical? (i.e., it should not produce messy, hallucinated objects). As long as the generated image is not chaotic, this metric can receive a high score.

- •Case B (Annotated Real-World Image):
- • Objective: Preservation of the background and non-edited areas; removal of instructions/markers.
- • Checkpoints: With the exception of the regions targeted for modification by instructions and markers, the background, lighting, shadows, and unrelated objects must be preserved at the pixel level (or highly approximated). Modifying unannotated areas is strictly prohibited.
- • Critical Judgment: The source image may contain three types of overlaid elements: (1) instructional text, (2) visual markers (bounding boxes, arrows, scribbles, highlights), and (3) original background text within the scene. The generated image must strictly remove all instructional text and visual markers (Types 1 and 2). Original background text (Type 3) is irrelevant and can be ignored. If instructional text or visual markers remain in the generated image, this dimension should receive a low score.

- 3. Visual Realism

- •General Objective: The naturalness of the image and the effective suppression of artifacts.
- •Checkpoints:
- • Are there conspicuous artifacts, blurriness, jagged edges, or anatomical distortions (e.g., twisted limbs)?
- • For Case B, is the blending between the edited region and the original background natural?

- 4. Spatial Control Precision

###### •Case A (Text-Only Source Image):

- • Objective: Compositional rationality.
- • Checkpoints: If there are no explicit visual markers, assign a default score of 5 (provided the object is complete and within the frame).

###### •Case B (Annotated Real-World Image):

- • Objective: Marker alignment.
- • Checkpoints: Is the generated content strictly confined within the boundaries of the visual markers (boxes, scribbles, etc.)? Is there any "spillover" or "under-filling"? Does the object follow the direction indicated by an arrow? Is the specific object pointed to by an arrow edited correctly?

###### Scoring & Output Format

Based on the analysis above, please output the results in a standardized format. Requirements: The analysis field must be concise.

- For Case A: Describe whether the generated object follows the instruction and briefly assess the visual quality.
- For Case B: Describe whether it follows the instruction, assess the visual quality, note if the non-edited areas maintain consistency, and evaluate the precision of spatial control. Scores are based on a 1-5 scale.

Final Verdict Rules:

PASS: The generated result meets acceptable quality standards. Criteria (must meet all):

Instruction Fidelity >= 3 (Mandatory - if < 3, it automatically fails) Average of the four scores >= 3.0 No single score < 2 (completely unacceptable) FAIL: The generated result has major quality issues.

Figure 21: Evaluation prompts used in VLM evaluators.

a balanced representation of 25 samples per subset. During the evaluation, annotators were presented with the input visual instruction and the generated output. To avoid the inherent subjectivity of Likert scales, evaluators adopted a strict visual-inspection approach. Specifically, they first made a strict binary judgment on whether the generation overall qualified as a “Pass” or “Fail”. If a sample was deemed a “Fail”, the annotators were then required to explicitly record which specific dimensions out of the four criteria (Instruction Fidelity, Content Consistency, Visual Realism, Spatial Precision) contributed to the failure, allowing for multi-label tagging. To ensure a highly robust standard, all 250 samples were cross-checked by the independent evaluators to reach a reliable consensus.

#### G Model details

###### G.1 Parameter breakdown

Table 11 reports the parameter breakdown of FlowInOne. The main trainable component is the Flow Backbone, which contains 1108.40M parameters and provides the core generative capacity. In addition, the Text-Image VAE contains 103.65M trainable parameters for mapping textual visual prompts into the image-like representation space. The Visual Encoder and Image VAE are kept frozen, with 322.58M and 83.65M parameters respectively, which reduces training cost while preserving strong visual representation and reconstruction priors.

Table 11: Parameter Breakdown of FlowInOne.

Component Params (M) Train.

Flow Backbone 1108.40 Yes Text-Image VAE 103.65 Yes Visual Encoder 322.58 No Image VAE 83.65 No

###### G.2 Loss function for image in-image out generation

We jointly train our model for the image in-image out generation task using the following comprehensive optimization objective:

L = Lfm + β1Lkld + β2Lclip (16)

where β1 and β2 are the scaling weights for the KL-divergence loss and the contrastive loss, respectively. For the Flow Matching loss Lfm, we compute the Mean Squared Error (MSE) between the predicted velocity field vθ(zt,t) at time-step t and the ground-truth velocity vˆ.

To explicitly enforce semantic alignment within our visual-centric paradigm, we adapt a CLIP-style contrastive loss. Unlike traditional text-to-image models that align text and images, our model specifically aligns the unified visual instruction embedding with the generated image representation. Specifically, given a mini-batch of N instruction-target pairs, we obtain the visual instruction latents zTI and extract the corresponding target image features zI. We then compute the cosine similarity between all pairs of zTI and zI in the batch. This results in an N × N similarity matrix S, where each element si,j represents the cosine similarity between the i-th instruction latent zTI and the j-th target image feature zI.

These similarity scores are subsequently scaled by a learnable temperature parameter τ, denoted as logitsi,j = si,j/τ. Following this, a symmetric cross-entropy loss over the similarity scores is computed along both the instruction-to-image

and image-to-instruction directions:

N

1 N

LTI→I = −

i=1

exp(logitsi,i)

log

N j=1 exp(logitsi,j)

N

1 N

LI→TI = −

i=1

exp(logitsi,i) N j=1 exp(logitsj,i)

log

We compute the average of these two components to obtain the final contrastive semantic alignment loss:

(17)

(18)

Lclip = CLIP(zTI,zI) =

- 1

- 2

(LTI→I + LI→TI) (19)

For the KL divergence loss Lkld, we regularize the visual instruction tokens towards a standard normal distribution N(0,1) to prevent latent space collapse. Based on ablation study, we set the hyperparameters to β1 = 1 × 10−2 and β2 = 1.

###### G.3 Hyperparameter ablation

Table 12: Hyperparameter ablation study on VP-Bench. We report the average PASS Rate (%) evaluated by three VLMs. Default optimal settings (CFG = 7, Steps = 50, β1 = 0.01, β2 = 1) are marked in bold. The control variable method is applied; parameters not actively ablated are fixed to their default values.

(a) CFG Scale (b) Sampling Steps (c) Loss Weights

|CFG PASS (%)|Steps PASS (%)<br><br>|(Lclip,Lkld) PASS (%)|
|---|---|---|
|1.1 38.5 1.5 44.8 3.0 47.7 5.0 51.5 7.0 54.0 9.0 44.2|10 25.7 20 40.3 30 46.3 40 48.7 50 54.0 - -|(0.1, 0.001) 46.7 (0.1, 0.01) 45.2 (0.1, 0.1) 44.2 (1, 0.001) 46.0 (1, 0.01) 54.0 (1, 0.1) 45.8|

Ablation on CFG scale. The CFG scale dictates the degree to which the generated image aligns with the visual and textual instructions. We sweep the CFG scale from 1.1 to 9. As shown in Table 12, the PASS Rate exhibits an inverted U-shape trajectory. At lower scales (e.g., 1.1 and 1.5), the model struggles to strictly follow the editing constraints, yielding sub-optimal PASS rates (38.5% and 44.8%). The performance peaks at 54.0% with a CFG scale of 7. However, excessively high guidance (CFG = 9) degrades the generation quality, likely due to visual artifacts and color saturation typical of diffusion models, causing the PASS Rate to drop to 44.2%. Consequently, we adopt 7 as the optimal CFG scale.

Ablation in the sampling steps. We evaluate the denoising process across various sampling steps ranging from 10 to 50. The results demonstrate a clear positive correlation between sampling steps and instruction adherence. Extreme low-step regimes (e.g., 10 steps) yield a poor PASS Rate of 25.7%, indicating insufficient structural synthesis. Performance improves substantially and begins to converge as steps increase, reaching our best result (54.0%) in steps 50. To balance computational efficiency and high-fidelity generation, 50 sampling steps are selected for our primary evaluations.

Ablation on loss weights. The overall optimization objective is defined as L = Lfm + β1Lkld + β2Lclip, where Lfm (weight fixed to 1) ensures structural integrity. We conduct a grid search over the KL divergence penalty β1 and the CLIP semantic alignment weight β2. The experiments reveal that the model is highly sensitive to this balance. Setting β2 = 1 and β1 = 0.01 achieves the superior PASS Rate of 54.0%. Lowering the CLIP weight (e.g., β2 = 0.1) universally harms performance (hovering around 45% − 46%), as it weakens the semantic alignment between the visual instruction embedding zTI and the image embedding zI. Conversely, over-penalizing the KL divergence (β1 = 0.1) overly restricts the latent space, hindering the model’s expressive capacity.

###### G.4 Experimental details

We utilize the WebDataset Aizman et al. [2020] format for highly efficient, streaming-based I/O processing of large-scale image-instruction pairs. To prevent the model from overfitting to dominant task categories and to ensure stable gradient descent, we employ a strictly balanced mini-batch sampling strategy. Specifically, this strategy guarantees that the eight distinct dataset sub-categories are uniformly distributed within each training batch, effectively mitigating task-level bias and ensuring balanced optimization across all instruction types. Initialized with pre-trained weights from Crossflow Liu et al. [2025c], our 1.2B-parameter model, FlowInOne, benefits from strong prior knowledge of visual semantics. During training, the resolution of the output images is 256 × 256. We employ the AdamW optimizer with a base learning rate of 1 × 10−4 alongside a standard cosine decay schedule with a linear warmup. The entire training process was conducted for 240,000 steps with a global batch size of 512, requiring approximately 240 A100 GPU hours to reach convergence. During the evaluation phase, when generating images on VP-Bench for comparison, all baseline models are executed using their official default inference parameters to ensure a standardized and fair assessment.

