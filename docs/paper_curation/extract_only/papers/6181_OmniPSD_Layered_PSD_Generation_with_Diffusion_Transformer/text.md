## OmniPSD: Layered PSD Generation with Diffusion Transformer

# arXiv:2512.09247v1[cs.CV]10Dec2025

Cheng Liu1,* Yiren Song1,* Haofan Wang2 Mike Zheng Shou1† 1National University of Singapore 2Lovart AI

[Figure 1]

[Figure 2]

I want to generate a minimalist eco poster: deep teal background, a cloud-ringed Earth centered, a light blue wave on top, with a few plants and bubbles as accents.

Extract the text, all foreground elements and

background

from this poster.

###### Generating Hierarchical Captioning/Prompt

Rendering Text

“poster”: “Features a stylized representation of Earth surrounded by clouds, …", “foreground”: "consists of a circular depiction of Earth, prominently placed …”, “midground”: "includes a wavy, light blue pattern that represents the ocean …”, “background”: "a solid deep teal color that fills the entire poster. It serves … ”

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Figure 1. OmniPSD is a Diffusion-Transformer framework that generates layered PSD files with transparent alpha channels. Our system supports both Text-to-PSD multi-layer synthesis and Image-to-PSD reconstruction, producing editable layers that preserve structure, transparency, and semantic consistency.

### Abstract

https://showlab.github.io/OmniPSD/.

Recent advances in diffusion models have greatly improved image generation and editing, yet generating or reconstructing layered PSD files with transparent alpha channels remains highly challenging. We propose OmniPSD, a unified diffusion framework built upon the Flux ecosystem that enables both text-to-PSD generation and imageto-PSD decomposition through in-context learning. For text-to-PSD generation, OmniPSD arranges multiple target layers spatially into a single canvas and learns their compositional relationships through spatial attention, producing semantically coherent and hierarchically structured layers. For image-to-PSD decomposition, it performs iterative in-context editing—progressively extracting and erasing textual and foreground components—to reconstruct editable PSD layers from a single flattened image. An RGBAVAE is employed as an auxiliary representation module to preserve transparency without affecting structure learning. Extensive experiments on our new RGBA-layered dataset demonstrate that OmniPSD achieves high-fidelity generation, structural consistency, and transparency awareness, offering a new paradigm for layered design generation and decomposition with diffusion transformers. Project page:

### 1. Introduction

Layered design formats such as Photoshop (PSD) files are essential in modern digital content creation, enabling structured editing, compositional reasoning, and flexible element-level manipulation. However, most generative models today can only output flat raster images [20, 46, 52], lacking the layer-wise structure and transparency information that are crucial for professional design workflows. To bridge this gap, we introduce OmniPSD, a unified diffusion-based framework that supports both text-to-PSD generation and image-to-PSD decomposition under a single architecture. It enables bidirectional transformation between textual or visual inputs and fully editable, multi-layer PSD graphics.

At the core of OmniPSD lies a pre-trained RGBAVAE, designed to encode and decode transparent images into a latent space that preserves alpha-channel information [24, 48, 74]. This RGBA-VAE serves as a shared foundation across both sub-tasks. On top of it, we leverage the Flux ecosystem, which consists of two complementary diffusion transformer models: Flux-dev [69], a text-to-image generator for creative synthesis, and Flux-Kontext [3], an image editing model for in-context refinement and recon-

* Equal contribution. † Corresponding author.

struction. By integrating these components, OmniPSD provides a unified, transparency-aware solution for both generation and decomposition.

- (1) Text-to-PSD Generation. Given a textual description, OmniPSD generates a layered PSD representation directly from text. Instead of producing a single flat image, we spatially arrange multiple semantic layers (e.g., background, foreground, text, and effects) into a 2×2 grid, and generate

- them simultaneously through the Flux-dev backbone. Each generated layer is then decoded by the shared RGBA-VAE to recover transparency and alpha information, producing semantically coherent, editable, and compositional layers.

- (2) Image-to-PSD Decomposition. For reverseengineering real or synthetic posters into editable PSDs, OmniPSD extends the Flux-Kontext model by replacing its standard VAE with our pre-trained RGBA-VAE, enabling transparency-aware reasoning in image editing. The decomposition process is iterative: we first extract text layers through in-context editing, then erase them to reconstruct the clean background, and finally segment and refine multiple foreground layers. All decomposed outputs are in RGBA format, ensuring accurate transparent boundaries and realistic compositional relationships.

By unifying generation and decomposition within a single diffusion-transformer architecture, OmniPSD demonstrates that both creative synthesis and structural reconstruction can be achieved under a transparency-aware, in-context learning framework.

Our main contributions are summarized as follows:

- • We present OmniPSD, a unified diffusion-based framework that supports both text-to-PSD generation and image-to-PSD decomposition within the same architecture, bridging creative generation and analytical reconstruction.
- • We pre-train a transparency-preserving RGBA-VAE and integrate it with Flux-dev and Flux-Kontext through incontext learning, achieving high-fidelity image generation and reconstruction with accurate alpha-channel representation.
- • We construct a large-scale dataset with detailed RGBA layer annotations and establish a new benchmark for editable PSD generation and decomposition. Extensive experiments demonstrate the effectiveness and superiority of our proposed approach.

### 2. Related Works

#### 2.1. Diffusion Models

Diffusion probabilistic models have rapidly become the dominant paradigm for high-fidelity image synthesis, largely replacing GANs due to their stable training, strong mode coverage, and ability to model complex data distributions via reversed noising processes [17, 20, 55]. They

now underpin a broad range of visual tasks, including text-to-image generation [58, 78–80], image editing [16, 26, 29, 59], and video synthesis [4, 21, 40–43, 52, 56]. To better support design and editing applications, subsequent work augments diffusion models with grounded conditions and spatial controls—for example, grounded textto-image generation, conditional control branches, imageprompt adapters, cross-attention-based prompt editing, selfguided sampling, inpainting modules, as well as instancelevel and multi-subject layout control [6, 13, 18, 35, 62, 63, 72, 73, 76]—thereby improving layout consistency and local editability. Early work predominantly relied on UNet-based denoisers in pixel or latent space, as popularized by latent diffusion models such as Stable Diffusion and SDXL [47, 52]. More recently, Transformer-based denoisers have become the de facto backbone, with Diffusion Transformers (DiT) driving models like Stable Diffusion 3, FLUX, HunyuanDiT, and PixArt, leveraging global attention and scalability to improve visual fidelity and prompt alignment [8, 14, 36, 46, 69]. In parallel, flow-matching and ODE-based formulations recast diffusion as learning continuous deterministic flows between distributions, enabling more efficient sampling and deterministic trajectories [37, 38].

#### 2.2. Layer-wise Image Generation

Layered image representations are fundamental to graphics and design, as they enable element-wise editing, compositional reasoning, and asset reuse [32]. Early work mainly decomposes a single image into depth layers, alpha mattes, or semantic regions under simplified foreground–background assumptions [7, 33, 53, 68], which helps matting and segmentation but falls short of the rich, editable layer structures used in professional tools. With diffusion models, newer methods explicitly target layered generation. Some methods still rely on post-hoc detection, segmentation, and matting from a flat RGB output, or adopt a two-stage “generate-then-decompose” pipeline, where a composite RGB image is first synthesized and then separated into foreground/background layers or RGBA instances [15, 30, 39, 65, 77]. Such designs often accumulate errors between stages and offer limited control over global layout and inter-layer relationships.

More recent approaches generate multi-layer content directly in a diffusion framework. LayerDiff, LayerDiffuse, and LayerFusion explore layer-collaborative or multibranch architectures to jointly synthesize background and multiple foreground RGBA layers while modeling occlusion relationships [11, 25, 75]. ART and LayerTracer further introduce region-based transformers and vector-graphic decoders for variable multi-layer layouts and object-level controllability [48, 57]. In parallel, multi-layer datasets such as MuLAn provide high-quality RGBA anno-

[Figure 18]

###### Legends

Text Erasure Object Erasure Object Erasure

[Figure 19]

🔥 Learnable

Text Prompt

Text Tokens

[Figure 20]

GPT4

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Noised Latent Tokens

[Figure 28]

Video Condi on Tokens

[Figure 29]

Hierarchical Prompt

DiT

- (a) RGBA VAE Training

[Figure 30]

[Figure 31]

[Figure 32]

🔥 🔥

[Figure 33]

[Figure 34]

RGBA VAE

[Figure 35]

DiT Block

[Figure 36]

🔥

[Figure 37]

[Figure 38]

- (b) Image Edit Training

[Figure 39]

DiT Block

[Figure 40]

🔥

[Figure 41]

[Figure 42]

- (c) Image Generate Training

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Render Text

[Figure 48]

Condi on

Object Extrac on

Object Extrac on

Text Extrac on

latent

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

latent

(d) Image-to-PSD Inference

(e) Text-to-PSD Inference

Figure 2. OmniPSD overview. A unified Diffusion-Transformer with a shared RGBA-VAE enables both text-to-PSD layered generation (left) and image-to-PSD decomposition (right). Text-to-PSD leverages spatial in-context learning with hierarchical captions, while Imageto-PSD performs iterative flow-guided foreground extraction and background restoration. Our method produces fully editable PSD layers with transparent alpha channels.

tations and occlusion labels to support controllable multilayer generation and editing [60]. Recent works like PSDiffusion explicitly harmonize layout and appearance across foreground and background layers [24], and our method follows this line while additionally targeting PSD-style layer structures and workflows tailored for poster and graphic design.

ing the resulting assets directly editable and composable in professional design tools.

#### 2.3. RGBA Image Generation

Generating transparent or layered RGBA content is crucial for compositing and design, yet has long been underexplored compared to standard RGB image synthesis. Traditional workflows typically rely on first generating opaque RGB images and then applying separate matting, segmentation, or alpha-estimation networks [7, 23, 33, 34, 53, 68], which often leads to inconsistent boundaries, halo artifacts, and limited control over transparency. Recent diffusionbased methods begin to treat transparency as a first-class signal. One representative line augments latent diffusion models with “latent transparency”, learning an additional latent offset that encodes alpha information while largely preserving the original RGB latent manifold, so that existing text-to-image backbones can natively produce transparent sprites or multiple transparent layers without retraining from scratch [75]. Building on this idea, RGBA-aware generators produce isolated transparent instances or sticker-like assets that can be flexibly composed for graphic design and poster layouts [15, 49].

Orthogonal to transparent-layer modeling, another line of work in automatic graphic design and poster generation emphasizes layout- and template-level generation. COLE and OpenCOLE propose hierarchical pipelines that decompose graphic design into planning, layer-wise rendering, and iterative editing [27, 28]. Graphist formulates hierarchical layout generation for multi-layer posters with a large multimodal model that outputs structured JSON layouts for design elements [10]. Visual Layout Composer introduces an image–vector dual diffusion model that jointly generates raster backgrounds and vector elements for design layouts [54]. MarkupDM and Desigen treat graphic documents as multimodal markup or controllable design templates, enabling completion and controllable template generation from partial specifications [31, 66]. PosterLLaVa further leverages multimodal large language models to generate poster layouts and editable SVG designs from naturallanguage instructions [71]. These systems focus on highlevel layout synthesis but typically output flattened renders or coarse vector structures, whereas our approach targets PSD-style RGBA layers with explicit alpha channels, mak-

Complementary work focuses on the representation side, proposing unified RGBA autoencoders that extend pretrained RGB VAEs with dedicated alpha channels and introducing benchmarks that adapt standard RGB metrics to four-channel images via alpha compositing, thereby stan-

dardizing evaluation for RGBA reconstruction and generation [64]. Building on these ideas, multi-layer generation systems increasingly adopt autoencoders that jointly encode and decode stacked RGBA layers and couple them with diffusion transformers that explicitly model transparency and inter-layer effects [9, 11, 24, 48, 65, 70, 74], often trained or evaluated on matting-centric multi-layer datasets such as MAGICK and MuLAn [5, 60], yielding more accurate alpha boundaries, coherent occlusions, and realistic soft shadows in complex layered scenes.

### 3. Method

In this section, we first introduce the unified OmniPSD architecture in Section 3.1. Next, Section 3.2 presents the RGBA-VAE module, which enables alpha-aware latent representation shared across both pathways. Then, Section 3.3 discusses the Image-to-PSD process based on iterative incontext editing and structural decomposition. After that, Section 3.4 describes the Text-to-PSD process, where layered compositions are generated via spatial in-context learning and cross-layer attention. Finally, Section 3.5 introduces the Layered Poster Dataset.

#### 3.1. Overall Architecture

We propose OmniPSD, a unified diffusion-based framework designed to reconstruct and generate layered PSD structures from either raster images or textual prompts. The framework is built upon the Flux model family [3, 69], combining Flux-Dev for text-to-image generation and FluxKontext for image editing within an in-context learning paradigm. At its core, a shared RGBA-VAE provides an alpha-aware latent space, enabling consistent representation of transparency and compositional hierarchy across both generation and decomposition tasks.

Specifically, the Image-to-PSD branch iteratively decomposes a given poster into text, foreground, and background layers through LoRA-based editing under the FluxKontext backbone, ensuring accurate structural separation with preserved alpha channels. In contrast, the Text-toPSD branch arranges layers spatially within a single generation canvas, where the model learns inter-layer relations via spatial attention under the Flux-DEV backbone. Together, these two pathways form a cohesive framework capable of bidirectional conversion between design images and editable PSD layers, supported by our large-scale Layered Poster Dataset for training and evaluation.

#### 3.2. RGBA-VAE

To accurately represent transparency and compositional relationships in layered design elements, we adopt and extend the AlphaVAE [64], a unified variational autoencoder for RGBA image modeling. While AlphaVAE provides a

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Foreground Extraction Foreground Erasure

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

{"poster": "The poster features a professional setting with a focus on

healthcare. It includes a laptop, medical tools, and abstract …...", "foreground": "The foreground content is positioned in the top-right corner of the poster. It features a close-up of a person's hands …...", "midground": "The midground content is located in the bottom-left corner of the poster. It includes a stethoscope and a clipboard, …...", "background": "The background content consists of a soft beige color with abstract shapes in light blue and peach tones. The …..."}

Text Erasure Text Erasure

(a) Image-to-PSD Reconstruction Dataset (b) Text-to-PSD Generation Dataset

Figure 3. OmniPSD’s layered dataset. Image-to-PSD is trained on paired samples, while Text-to-PSD uses a 2 × 2 grid that presents the full poster and its decomposed layers for in-context learning.

strong foundation for alpha-aware reconstruction, its pretraining on limited natural transparency data causes severe degradation when applied to design scenarios such as semitransparent text, shadow overlays, and soft blending effects. To address this, we retrain the model on our curated dataset of real-world design samples, enabling stable reconstruction of both alpha and color layers. We refer to this retrained version as RGBA-VAE.

Following the formulation in the original AlphaVAE paper, our training objective jointly optimizes pixel fidelity, patch-level consistency, perceptual alignment, and latent regularization as:

L = λpix E ∥Iˆ− I∥1 + λpatch E ∥ϕ(Iˆ) − ϕ(I)∥1

+ λperc E ∥ψ(Iˆ) − ψ(I)∥22

(1)

+ λKL KL(q(zRGB|·)∥p) + KL(q(zA|·)∥p) ,

where I and Iˆdenote the ground-truth and reconstructed images, respectively. ϕ(·) represents a patch-level feature extractor enforcing local structure consistency, and ψ(·) denotes a perceptual encoder (e.g., VGG) that maintains semantic fidelity. zRGB and zA correspond to the latent variables for color and alpha channels, and p is the Gaussian prior. The coefficients λpix, λpatch, λperc, and λKL balance pixel accuracy, local consistency, perceptual alignment, and latent regularization, respectively.

This retraining procedure effectively bridges the gap between natural transparency modeling and design-layered imagery. The resulting RGBA-VAE thus provides a shared latent space for both our text-to-PSD and image-to-PSD modules, enabling high-fidelity, alpha-preserving decomposition and generation.

#### 3.3. Image-to-PSD Reconstruction

We formulate the Image-to-PSD reconstruction task as a multi-step, iterative image-editing process, analogous to how professional designers manually decompose visual elements into layers in Photoshop. Instead of predicting all

layers in a single pass, we progressively extract text and foreground objects, while recovering occluded background content. Each step outputs an RGBA PNG layer with accurate transparency. This iterative design ensures pixel-level fidelity, precise alpha recovery, and structural composability for final PSD reconstruction.

Concretely, we train two expert models: one specialized for foreground extraction and another for foreground removal and background restoration. After each extraction, the background-restoration model reconstructs clean background content, enabling the system to reveal deeper visual layers over iterations. Through this alternating “extractforeground → erase-foreground” process, a flattened input image is gradually decomposed into a stack of text, foreground, and background layers suitable for PSD editing. This pipeline is built on the Flux Kontext diffusion backbone with task-specific LoRA adapters. The decomposition process is formulated as a conditional flow-matching problem, where the flattened image is treated as a conditioning input and the model learns a deterministic flow field that maps noisy latent states toward their target decomposed layer representations.

Formulation. Let I0 ∈ RH×W×4 denote the flattened input poster image, and y ∈ {foreground,background} denote the target layer type. We define latent variables z0 = Eα(I0) and z1 = Eα(Iy), where Eα is the RGBAVAE encoder. Flux models the continuous transformation between z0 and z1 as a flow field vθ(zt,t | z0) governed by an ODE:

dzt dt

= vθ(zt,t | z0), t ∈ [0,1], (2)

where zt = (1 − t)z0 + tz1 represents intermediate latent states.

The training objective follows the standard Flow Matching Loss [37, 69]:

0,z1) ∥vθ(zt,t | z0) − (z1 − z0)∥22 ,

Lflow = Et∼U(0,1),(z

(3) which enforces the learned flow field to align with the true displacement between input and target latents. This formulation avoids stochastic noise injection, leading to faster convergence and deterministic inference.

Foreground Extraction Model. Given I0, the model detects salient regions and generates RGBA layers for each foreground instance. Each LoRA adapter is trained on triplets (I0,m,Ifg), where m denotes a binary or boundingbox mask, and Ifg is the corresponding RGBA foreground target. Both conditional and target images are encoded into latent sequences:

zcond = Eα(I0), ztarget = Eα(Ifg), (4)

- then concatenated into a unified token sequence: Z = [zcond;ztarget]. (5)

The transformer backbone applies Multi-Modal Attention (MMA) [45] with bidirectional context:

QK⊤ √

Z′ = MMA(Z) = Softmax

V, (6) capturing pixel-level and semantic correlations between

d

input and decomposed regions.

Foreground Erasure Model. After extraction, we employ an erasure module trained to reconstruct occlusion-free backgrounds Ibg given the same condition I0 and mask m. At each iteration k, the model removes the current foreground, restores the occluded background I(bgk), and stores the removed content I(fgk) as an independent RGBA layer:

{I(1)fg ,...,I(fgK),Ibg} → PSD Stack. (7) All LoRA modules share the same latent flow space of Flux Kontext, ensuring modular composability across text removal, object extraction, and background inpainting subtasks.

Editable Text Layer Recovery. To transform rasterized text regions into editable design layers, we reconstruct vector-text through a unified OCR–font-recovery–rendering pipeline. We detect and recognize textual content from pixel-level inputs using a transformer-based OCR module, implemented via the open-source PaddleOCR toolkit [2], which provides state-of-the-art scene and documenttext recognition with multilingual and layout-aware support. The recognized text regions are then associated with the most plausible typeface from a curated font bank through semantic font embedding retrieval, achieved using the lightweight font classify system [1], which enables efficient deep-learning-based font matching across large-scale font libraries. The recovered text content together with its inferred font attributes is subsequently re-rendered as resolution-independent vector layers, yielding editable PSD text objects that faithfully preserve the original typography and layout structure.

#### 3.4. Text-to-PSD Generation

While Image-to-PSD is highly effective at decomposing an existing image into layered RGBA components, in many real scenarios no reference image is available. Instead, users may wish to generate a fully layered PSD file directly from textual descriptions. To meet this need, we introduce the Text-to-PSD model, which leverages hierarchical textual prompts, cross-modal feature alignment, and an in-context layer reasoning mechanism.

In-Context Layer Reasoning via a 2×2 Grid. Our key idea is to enable different layers to “see” each other without modifying the backbone or introducing explicit cross-layer attention modules [61]. We arrange four images—the full poster Ifull, foreground Ifg, middle-ground Imid, and back-

ground Ibg—into a 2×2 grid:

Ifull Ifg Imid Ibg

G =

##### .

This grid serves as an in-context visual canvas, enabling the model’s native spatial attention to implicitly learn layer relationships such as layout consistency, occlusion ordering, color harmony, and transparency boundaries. During inference, the model generates all PSD layers jointly in a single pass.

Hierarchical Text Prompts. To provide structured semantic grounding, we annotate each sample with a JSON record that assigns a dedicated description to the full poster and each semantic layer, e.g., {"poster": "...", "foreground": "...", "midground": "...", "background": "..." }. Here, poster captures the global scene, while the remaining fields describe the corresponding layers.

Grid Spatial In-Context Learning. The 2×2 grid G is encoded by the RGBA-VAE and processed by the DiT backbone in a single forward pass. Spatial self-attention over this grid lets layer tokens attend to the full-poster tokens, so the model learns cross-layer correspondences and compositional relationships without any extra cross-layer modules.

Training Objective. We retain the standard flow-matching objective of the diffusion transformer and introduce no additional losses, allowing the model to learn layered semantics purely from the hierarchical prompts and the in-context 2 × 2 grid formulation.

#### 3.5. Dataset Construction

To support training and evaluation, we construct the Layered Poster Dataset, comprising over 200,000 real PSD files collected from online design repositories. These files are manually authored by professional designers and contain rich semantic groupings, font layers, shape groups, and effect overlays. We perform automated parsing to extract group-level and layer-level metadata, then apply postfiltering to retain only PSDs with valid RGBA structure. Each sample is annotated into structured groups—text, foreground, background—with each layer saved as an RGBA png and associated with editable metadata (e.g., bounding box, visibility, stacking order).

To further support training across different subtasks, we organize the data with task-specific structures. For the Textto-PSD generation task, we intentionally remove all text layers during dataset construction, since text should be rendered last rather than generated. This preserves authentic typography, font fidelity, and editability. The data is arranged in a four-panel grid: the top-left contains the full poster, while the remaining three panels provide semantic decomposition—top-right: foreground layer 1, bottom-left: foreground layer 2, and bottom-right: background layer.

This format encourages the model to learn how text conditions map to layered design structures.

For the Image-to-PSD task, we adopt a triplet data strategy that mirrors the iterative layer editing process at inference time. Each triplet consists of (i) an input image, (ii) the extracted foreground content, and (iii) the corresponding background after foreground removal. This setup simulates the step-by-step editing workflow used in practical design software—first isolating editable regions, then erasing them from the scene—enabling the model to learn realistic PSD-style layer decomposition and reconstruction.

### 4. Experiments

#### 4.1. Experiment Details.

Experiment Details. During the Text-to-PSD training, we employed the Flux 1.0 dev model [69] built upon the pretrained Diffusion Transformer (DiT) architecture. The training resolution was set to 1024×1024 with a 2×2 grid layout. We adopted the LoRA fine-tuning strategy [22] with a LoRA rank of 128, a batch size of 8, a learning rate of 0.001, and 30,000 fine-tuning steps.

For the Image-to-PSD model training, we fine-tuned LoRA adapters on the Flux Kontext backbone [3] at a resolution of 1024×1024. Specifically, we separately trained two types of modules—foreground extraction (for text and non-text elements) and foreground erasure (for text and non-text elements)—each for 30,000 steps. For tasks that require transparency channels (e.g., Text-to-PSD, text extraction, and object extraction), we used the RGBA-VAE as the variational autoencoder. For other tasks without transparency needs, we used the original VAE backbone.

Baseline Methods. For the Text-to-PSD task, we benchmark against LayerDiffuse [74] and GPT-Image-1 [44], the most relevant publicly available layered poster generation systems. For the Image-to-PSD task, to the best of our knowledge, this is the first work enabling editable PSD reconstruction from a single flattened image, and thus no prior method exists for direct comparison. Thus, we evaluate several commercial systems capable of producing RGBA layers [44], as well as a non-RGBA baseline [3, 12] where foregrounds are generated on a white canvas and transparency masks are derived using SAM2 segmentation [51], representing a proxy solution without alpha-aware modeling.

Metrics. We evaluate OmniPSD using four metrics. FID [19] is computed on each generated layer and composite output to measure visual realism. For the Text-to-PSD task, we report layer-wise CLIP Score [50] to assess semantic alignment between each generated layer and its textual prompt. For the Image-to-PSD task, we compute reconstruction MSE by re-compositing predicted layers into a flattened image and measuring pixel error against the input. Together, these metrics capture realism, semantic fi-

|[Figure 64]|[Figure 65]|[Figure 66]|[Figure 67]|
|---|---|---|---|

|[Figure 68]|[Figure 69]|[Figure 70]|[Figure 71]|
|---|---|---|---|

|[Figure 72]|[Figure 73]|[Figure 74]|[Figure 75]|
|---|---|---|---|

|[Figure 76]|[Figure 77]|[Figure 78]|
|---|---|---|

|[Figure 79]|
|---|

|[Figure 80]|[Figure 81]|[Figure 82]|[Figure 83]|
|---|---|---|---|

|[Figure 84]|[Figure 85]|[Figure 86]|[Figure 87]|
|---|---|---|---|

|[Figure 88]|[Figure 89]|[Figure 90]|[Figure 91]|
|---|---|---|---|

|[Figure 92]|[Figure 93]|[Figure 94]|[Figure 95]| |
|---|---|---|---|---|

|[Figure 96]|[Figure 97]|[Figure 98]|[Figure 99]|
|---|---|---|---|

|[Figure 100]|[Figure 101]|[Figure 102]|[Figure 103]|
|---|---|---|---|

- (a) Image-to-PSD Reconstruction
- (b) Text-to-PSD Generation

|[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]|[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]|[Figure 110]|[Figure 111]|[Figure 112]|
|---|---|---|---|---|

|[Figure 113]<br><br>[Figure 114]|[Figure 115]<br><br>[Figure 116]|[Figure 117]|[Figure 118]|[Figure 119]|
|---|---|---|---|---|

{"poster": "The poster features a serene landscape scene ...", "foreground": "The foreground content includes various green botanical ...", "midground": “Showcases a stunning view of mountains ...", "background": “Consists of a textured green pattern that fills remaining ..."}

{"poster": "The poster radiates a cheerful, adventurous summer theme, …", "foreground": "there is a silhouette of several domed and towered …", "midground": "Colorful hanging flags in white, green, and ...", "background": "The background presents a glowing golden ..."}

|[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]|[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]|[Figure 126]|[Figure 127]|[Figure 128]|
|---|---|---|---|---|

|[Figure 129]|[Figure 130]|[Figure 131]|
|---|---|---|

|[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]|
|---|

|[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]|
|---|

{"poster": "The poster features a romantic theme with a couple in love ...", "foreground": “Showcases a circular image of a couple, positioned centrally ...", "midground": “Consists of various abstract shapes...", "background": “Features a smooth gradient transitioning from ..."}

{“poster”: “The poster features a nature-themed design ...", "foreground": "The foreground content is centered in the poster, featuring a pair of hands ...", "midground": “Consists of abstract green shapes ...", "background": “Features a textured, crumpled paper design in ..."}

- Figure 4. Generation results of OmniPSD. (a) Image-to-PSD reconstruction decomposes an input poster into editable text layers, multiple foreground layers, and a clean background layer. (b) Text-to-PSD synthesis uses hierarchical captions to generate background and foreground layers, followed by rendering the corresponding editable text layers.

delity, structural coherence, and reconstruction accuracy. To evaluate cross-layer structure and layout coherence, we employ GPT-4 [44] as a vision-language judge, scoring spatial arrangement and design consistency. The detailed GPT-

- 4 score metrics are provided in the supplementary materials A.

Benchmarks. For the Text-to-PSD task, we prepare a test set of 500 layer-aware prompts (two foreground, one background, and one global layout description), all derived from

real PSD files to ensure realistic evaluation. For the Imageto-PSD task, we curate 500 real PSD files as the test set, which are flattened into single images for evaluating PSD reconstruction quality.

User Study. We conducted a user study with 18 participants to evaluate the usability and perceptual quality of the layers generated by OmniPSD. The detailed study procedures and results are provided in the supplementary materials B.

Poster Foreground Background Original Poster

Reconstructed Poster

Text Extraction

Text Erasure

Foreground Extraction

Foreground Erasure

Poster Foreground Background

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

[Figure 149]

[Figure 150]

[Figure 151]

LayerDiffuse SDXL

Kontext & Segmentation

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Nano-Banana & Segmentation

[Figure 162]

GPTImage-1

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

GPT-Image-1

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

OmniPSD

OmniPSD

###### (a) Text-to-PSD Comparative Results (b) Image-to-PSD Comparative Results

- Figure 5. Compare with baselines on text-to-PSD and image-to-PSD. OmniPSD matches the visual quality of leading diffusion and visionlanguage models while uniquely supporting multi-layer PSD generation with transparent alpha channels. Compared to existing layered synthesis baselines, it achieves clearly superior visual fidelity and more coherent, logically structured layers.

- Table 1. Image-to-PSD generation results across methods. Lower is better for MSE; higher is better for PSNR, SSIM, CLIP-I (CLIP image score), and GPT-4-score. Bold numbers indicate the best performance for each metric.

Method MSE ↓ PSNR ↑ SSIM ↑ CLIP-I ↑ GPT-4-score ↑

Kontext [3] 1.10e-1 9.59 0.653 0.692 0.64 Nano-Banana [12] 2.06e-2 16.9 0.816 0.916 0.86 GPT-Image-1 [44] 2.48e-2 16.1 0.761 0.837 0.84 OmniPSD (ours) 1.14e-3 24.0 0.952 0.959 0.92

- Table 2. Text-to-PSD generation results across methods. Lower is better for FID; higher is better for CLIP and GPT-4 scores. Bold numbers indicate the best performance for each metric.

Method FID ↓ CLIP Score ↑ GPT-4 Score ↑

LayerDiffuse SDXL [74] 89.35 24.78 0.66 GPT-Image-1 [44] 53.21 35.59 0.84 OmniPSD (ours) 30.43 37.64 0.90

- Table 3. Image-to-PSD evaluation. Lower is better for FID and MSE; higher is better for PSNR and GPT-4 scores. We evaluate two sub-tasks—foreground extraction and foreground erasure—as well as the full reconstruction pipeline.

Task FID ↓ MSE ↓ PSNR ↑ GPT-4 Score ↑

Text Extraction 11.42 1.34e-3 26.86 0.86 Text Erasure 19.38 1.15e-3 26.37 0.94 Foreground Extraction 33.35 2.26e-3 19.27 0.84 Foreground Erasure 27.14 2.13e-3 29.41 0.92

Full Image-to-PSD 24.71 1.14e-3 23.98 0.90

- 4.2. Comparison and Evaluation

produces plausible foregrounds and layouts but unstable, artifact-prone backgrounds, while GPT-Image-1, despite strong visual quality, often loses or alters background elements, harming global consistency. OmniPSD, by contrast, yields high-quality foreground and background layers with coherent overall posters. For image-to-PSD, baselines do not output true RGBA layers and thus cannot provide checkerboard visualizations. OmniPSD accurately performs text extraction, foreground extraction/removal, and background reconstruction, whereas other methods struggle to recover text and maintain consistency between extracted and erased regions, limiting their usability for PSD-style editing.

Quantitative Evaluation. This section presents quantitative analysis results. Table 1 and 2 summarize the comparison results. Table 3 further reports the performance of each component in the image-to-PSD pipeline. Compared with strong baselines, OmniPSD achieves visual generation quality on par with state-of-the-art large diffusion and vision-language models. More importantly, our method uniquely supports multi-layer PSD generation with transparent alpha channels, a capability that existing approaches are far from achieving. Relative to prior layered synthesis systems, OmniPSD also demonstrates significant advantages in visual fidelity, semantic coherence, and logical layer structure, producing clean, editable layers that better reflect real design workflows.

#### 4.3. Ablation Study.

In this section, we present a detailed ablation study. We first compare our RGBA-VAE with other VAEs capable of encoding and decoding alpha channels. As shown in Table 4 and 6, models trained primarily on natural images perform

Qualitative Evaluation. Figure 4 and 5 show the qualitative comparison. For text-to-PSD, LayerDiffuser-SDXL

### References

Table 4. RGBA reconstruction results. Lower is better for MSE and LPIPS; higher is better for PSNR and SSIM. Bold numbers indicate the best performance for each metric.

- [1] Storia AI. font-classify: Lightweight deep-learning-based font recognition. https://github.com/StoriaAI/font-classify, 2024. Accessed: 2025-03-10. 5
- [2] PaddlePaddle Authors. Paddleocr: Open-source ocr toolkit. https://github.com/PaddlePaddle/ PaddleOCR, 2023. Accessed: 2025-03-10. 5
- [3] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 1, 4, 6, 8

- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [5] Ryan D. Burgert, Brian L. Price, Jason Kuen, Yijun Li, and Michael S. Ryoo. Magick: A large-scale captioned dataset from matting generated images using chroma keying. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 4
- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 2
- [7] Guowei Chen, Yi Liu, Jian Wang, Juncai Peng, Yuying Hao, Lutao Chu, Shiyu Tang, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang, Xiaoguang Hu, and Dianhai Yu. Pp-matting: High-accuracy natural image matting. arXiv preprint arXiv:2204.09433, 2022. 2, 3
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2
- [9] Xuewei Chen, Zhimin Chen, and Yiren Song. Transanimate: Taming layer diffusion to generate rgba video. arXiv preprint arXiv:2503.17934, 2025. 4
- [10] Yutao Cheng, Zhao Zhang, Maoke Yang, Hui Nie, Chunyuan Li, Xinglong Wu, and Jie Shao. Graphic design with large multimodal model. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025. 3
- [11] Yusuf Dalva, Yijun Li, Qing Liu, Nanxuan Zhao, Jianming Zhang, Zhe Lin, and Pinar Yanardag. Layerfusion: Harmonized multi-layer text-to-image generation with generative priors. arXiv preprint arXiv:2412.04460, 2024. 2, 4
- [12] Google DeepMind. Nano-banana (gemini 2.5 flash image): Google deepmind’s image generation and editing model. https://aistudio.google.com/models/ gemini-2-5-flash-image, 2025. Accessed: 202511-14. 6, 8
- [13] Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for control-

Method MSE ↓ PSNR ↑ SSIM ↑ LPIPS ↓

LayerDiffuse VAE [74] 2.54e-1 8.06 0.289 0.473 Red-VAE [67] 2.52e-1 8.53 0.300 0.451 Alpha-VAE [64] 4.15e-3 26.9 0.739 0.120 RGBA-VAE (ours) 9.82e-4 32.5 0.945 0.0348

Table 5. Ablation study results in Text-to-PSD task.

Method FID ↓ CLIP Score ↑ GPT-4 Score ↑

w/o layer-specific prompt 38.56 34.31 0.78 OmniPSD full 30.43 37.64 0.90

poorly in the design-poster setting, exhibiting inconsistent reconstruction, noticeable artifacts, and blurred text. Table

- 5 further highlights the importance of structured, layer-wise prompts in the text-to-PSD task: when using naive prompts, the generation quality degrades significantly.

Ground Truth LayerDiffuse VAE Red-VAE Alpha-VAE RGBA-VAE

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

- Figure 6. OmniPSD’s RGBA-VAE. Compared to existing VAE methods which compatible with image alpha channels.

### 5. Conclusion

In this paper, we present OmniPSD, a unified framework for layered and transparency-aware PSD generation from a single raster image. Built upon a Diffusion Transformer backbone, OmniPSD decomposes complex poster-style images into structured RGBA layers through an iterative, in-context editing process. Our framework integrates an RGBAVAE for alpha-preserving representation and multiple taskspecific Kontext-LoRA modules for text, object, and background reconstruction. We further construct a large-scale, professionally annotated layered dataset to support training and evaluation. Extensive experiments demonstrate that OmniPSD achieves superior structural fidelity, transparency modeling, and semantic consistency, establishing a new paradigm for design-aware image decomposition and editable PSD reconstruction.

- lable image generation. Advances in Neural Information Processing Systems, 36, 2023. 2
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 2
- [15] Alessandro Fontanella, Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, and Sarah Parisot. Generating compositional scenes via text-to-image rgba instance generation. Advances in Neural Information Processing Systems, 2024. 2, 3
- [16] Yan Gong, Yiren Song, Yicheng Li, Chenglin Li, and Yin Zhang. Relationadapter: Learning and transferring visual relation with diffusion transformers. arXiv preprint arXiv:2506.02528, 2025. 2
- [17] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross-attention control. In Proceedings of the International Conference on Learning Representations, 2023. 2
- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, 2017. 6
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2
- [21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 2
- [22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 6
- [23] Xiaobin Hu, Xu Peng, Donghao Luo, Xiaozhong Ji, Jinlong Peng, Zhengkai Jiang, Jiangning Zhang, Taisong Jin, Chengjie Wang, and Rongrong Ji. Diffumatting: Synthesizing arbitrary objects with matting-level annotation. arXiv preprint arXiv:2403.06168, 2024. 3
- [24] Dingbang Huang, Wenbo Li, Yifei Zhao, Xinyu Pan, Yanhong Zeng, and Bo Dai. Psdiffusion: Harmonized multilayer image generation via layout and appearance alignment. arXiv preprint arXiv:2505.11468, 2025. 1, 3, 4
- [25] Runhui Huang, Kaixin Cai, Jianhua Han, Xiaodan Liang, Wei Zhang, Songcen Xu, and Hang Xu. Layerdiff: Exploring text-guided multi-layered composable image synthesis via layer-collaborative diffusion model. arXiv preprint arXiv:2403.12036, 2024. 2

- [26] Shijie Huang, Yiren Song, Yuxuan Zhang, Hailong Guo, Xueyin Wang, and Jiaming Liu. Arteditor: Learning customized instructional image editor from few-shot examples. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17651–17662, 2025. 2
- [27] Naoto Inoue, Kento Masui, Wataru Shimoda, and Kota Yamaguchi. Opencole: Towards reproducible automatic graphic design generation. arXiv preprint arXiv:2406.08232, 2024. 3
- [28] Peidong Jia, Chenxuan Li, Zeyu Liu, Yichao Shen, Xingru Chen, Yuhui Yuan, Yinglin Zheng, Dong Chen, Ji Li, Xiaodong Xie, Shanghang Zhang, and Baining Guo. Cole: A hierarchical generation framework for graphic design. arXiv preprint arXiv:2311.16974, 2023. 3
- [29] Yuxin Jiang, Yuchao Gu, Yiren Song, Ivor Tsang, and Mike Zheng Shou. Personalized vision via visual in-context learning. arXiv preprint arXiv:2509.25172, 2025. 2
- [30] Kyoungkook Kang, Gyujin Sim, Geonung Kim, Donguk Kim, Seungho Nam, and Sunghyun Cho. Layeringdiff: Layered image synthesis via generation, then disassembly with generative knowledge. arXiv preprint arXiv:2501.01197,

2025. 2

- [31] Kotaro Kikuchi, Ukyo Honda, Naoto Inoue, Mayu Otani, Edgar Simo-Serra, and Kota Yamaguchi. Multimodal markup document models for graphic design completion. In Proceedings of the ACM International Conference on Multimedia, 2025. 3
- [32] Wei-Cheng Lee, Chih-Peng Chang, Wen-Hsiao Peng, and Hsueh-Ming Hang. A hybrid layered image compressor with deep-learning technique. In IEEE International Workshop on Multimedia Signal Processing (MMSP), 2020. 2
- [33] Jizhizi Li, Jing Zhang, Stephen J. Maybank, and Dacheng Tao. Bridging composite and real: Towards end-to-end deep image matting. International Journal of Computer Vision, 130(2):246–266, 2022. 2, 3
- [34] Jiachen Li, Jitesh Jain, and Humphrey Shi. Matting anything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 1775– 1785, 2024. 3
- [35] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2
- [36] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 2

- [37] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2, 5
- [38] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2
- [39] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Learning to see through obstructions with layered decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
- [40] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4117–4125, 2024. 2
- [41] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024.
- [42] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Leqi Shen, Chenyang Qi, Jixuan Ying, Chengfei Cai, Zhifeng Li, Heung-Yeung Shum, et al. Follow-your-click: Open-domain regional image animation via motion prompts. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6018–6026, 2025.
- [43] Yue Ma, Yulong Liu, Qiyuan Zhu, Ayden Yang, Kunyu Feng, Xinhua Zhang, Zhifeng Li, Sirui Han, Chenyang Qi, and Qifeng Chen. Follow-your-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning. arXiv preprint arXiv:2506.05207, 2025. 2
- [44] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6, 7, 8
- [45] Zexu Pan, Zhaojie Luo, Jichen Yang, and Haizhou Li. Multimodal attention for speech emotion recognition. arXiv preprint arXiv:2009.04107, 2020. 5
- [46] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 1, 2

- [47] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [48] Yifan Pu, Yiming Zhao, Zhicong Tang, Ruihong Yin, Haoxing Ye, Yuhui Yuan, Dong Chen, Jianmin Bao, Sirui Zhang, Yanbin Wang, et al. Art: Anonymous region transformer for variable multi-layer transparent image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7952–7962, 2025. 1, 2, 4
- [49] Fabio Quattrini, Vittorio Pippi, Silvia Cascianelli, and Rita Cucchiara. Alfie: Democratising rgba image generation with no $$$. arXiv preprint arXiv:2408.14826, 2024. 3
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. 6
- [51] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 6
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [53] Soumyadip Sengupta, Vivek Jayaram, Brian Curless, Steven M Seitz, and Ira Kemelmacher-Shlizerman. Background matting: The world is your green screen. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2291–2300, 2020. 2, 3
- [54] Mohammad Amin Shabani, Zhaowen Wang, Difan Liu, Nanxuan Zhao, Jimei Yang, and Yasutaka Furukawa. Visual layout composer: Image-vector dual diffusion model for design layout generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2024. 3

- [55] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [56] Yiren Song, Shijie Huang, Chen Yao, Xiaojun Ye, Hai Ci, Jiaming Liu, Yuxuan Zhang, and Mike Zheng Shou. Processpainter: Learn painting process from sequence data. arXiv preprint arXiv:2406.06062, 2024. 2
- [57] Yiren Song, Danze Chen, and Mike Zheng Shou. Layertracer: Cognitive-aligned layered svg synthesis via diffusion transformer. arXiv preprint arXiv:2502.01105, 2025. 2
- [58] Yiren Song, Cheng Liu, and Mike Zheng Shou. Makeanything: Harnessing diffusion transformers for multidomain procedural sequence generation. arXiv preprint arXiv:2502.01572, 2025. 2
- [59] Yiren Song, Cheng Liu, and Mike Zheng Shou. Omniconsistency: Learning style-agnostic consistency from paired stylization data. arXiv preprint arXiv:2505.18445, 2025. 2
- [60] Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, Fei Chen, Steven McDonagh, Gerasimos Lampouras, Ignacio Iacobacci, and Sarah Parisot. Mulan: A multi layer annotated dataset for controllable text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3, 4
- [61] Cong Wan, Xiangyang Luo, Zijian Cai, Yiren Song, Yunlong Zhao, Yifan Bai, Yuhang He, and Yihong Gong. Grid: Visual layout generation. arXiv preprint arXiv:2412.10718, 2024. 5
- [62] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instancelevel control for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2

- [63] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. In Proceedings of the International Conference on Learning Representations, 2025. 2
- [64] Zile Wang, Hao Yu, Jiabo Zhan, and Chun Yuan. Alphavae: Unified end-to-end rgba image reconstruction and generation with alpha-aware representation learning. arXiv preprint arXiv:2507.09308, 2025. 4, 9
- [65] Zitong Wang, Hang Zhao, Qianyu Zhou, Xuequan Lu, Xiangtai Li, and Yiren Song. Diffdecompose: Layer-wise decomposition of alpha-composited images via diffusion transformers. arXiv preprint arXiv:2505.21541, 2025. 2, 4
- [66] Haohan Weng, Danqing Huang, Yu Qiao, Zheng Hu, ChinYew Lin, Tong Zhang, and C. L. Philip Chen. Desigen: A pipeline for controllable design template generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3
- [67] Qiang Xiang and Shuang Sun. Layerdiffuse-flux: Flux version implementation of layerdiffusion. https : //github.com/FireRedTeam/LayerDiffuseFlux, 2025. Code repository, accessed 2025-11-13. 9
- [68] Ning Xu, Brian Price, Scott Cohen, and Thomas Huang. Deep image matting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2970– 2979, 2017. 2, 3
- [69] Chenglin Yang, Celong Liu, Xueqing Deng, Dongwon Kim, Xing Mei, Xiaohui Shen, and Liang-Chieh Chen. 1.58-bit flux. arXiv preprint arXiv:2412.18653, 2024. 1, 2, 4, 5, 6
- [70] Jinrui Yang, Qing Liu, Yijun Li, Soo Ye Kim, Daniil Pakhomov, Mengwei Ren, Jianming Zhang, Zhe Lin, Cihang Xie, and Yuyin Zhou. Generative image layer decomposition with visual effects. arXiv preprint arXiv:2411.17864, 2024. 4
- [71] Tao Yang, Yingmin Luo, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. Posterllava: Constructing a unified multi-modal layout generator with llm. arXiv preprint arXiv:2406.02884, 2024. 3
- [72] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2

- [73] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023. 2
- [74] Lvmin Zhang and Richard Zhang. Transparent image layer diffusion using latent transparency. arXiv preprint

- arXiv:2402.17113, 2024. 1, 4, 6, 8, 9

[75] Lvmin Zhang and Richard Zhang. Transparent image layer diffusion using latent transparency. arXiv preprint

- arXiv:2402.17113, 2024. 2, 3

- [76] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2
- [77] Xinyang Zhang, Wentian Zhao, Xin Lu, and Jeff Chien. Text2layer: Layered image generation using latent diffusion model. arXiv preprint arXiv:2307.09781, 2023. 2

- [78] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024. 2
- [79] Yuxuan Zhang, Lifu Wei, Qing Zhang, Yiren Song, Jiaming Liu, Huaxia Li, Xu Tang, Yao Hu, and Haibo Zhao. Stablemakeup: When real-world makeup transfer meets diffusion model. arXiv preprint arXiv:2403.07764, 2024.
- [80] Yuxuan Zhang, Qing Zhang, Yiren Song, Jichao Zhang, Hao Tang, and Jiaming Liu. Stable-hair: Real-world hair transfer via diffusion model. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10348–10356, 2025. 2

### Supplementary

In the supplementary material, we provide additional details on the GPT-4-based automatic evaluation protocol, describe the design and results of our user study in both textto-PSD and image-to-PSD settings, present more qualitative examples of OmniPSD’s layered poster generation and reconstruction, and showcase the interactive user interface and typical editing workflows supported by our system.

### A. GPT-4 Evaluation

In this section, we provide additional details about the automatic GPT-4-based evaluation protocol used in our experiments. We describe how candidate layered posters from different methods are jointly presented to GPT-4, the discrete 1–5 scoring rubric, the JSON output format, and how we aggregate and normalize these scores to obtain the final quantitative metric reported in the main paper.

Implementation details of the GPT-4 evaluation. We adopt GPT-4 as an automatic visual judge to assess the quality of layered posters produced by different methods. For each input (either a text description or an image), we collect all candidate outputs from the compared methods and submit them together in a single query. GPT-4 then gives each method an independent score, which allows a fair, side-byside comparison under exactly the same context.

The assistant evaluates the layered poster results of different methods according to a 1–5 scale:

- • 1 = very poor (severely unreasonable, chaotic structure, and strong visual inconsistency),
- • 2 = poor,
- • 3 = fair / acceptable but with clear flaws,
- • 4 = good,
- • 5 = very good (clear structure, reasonable layer relationships, and visually coherent as a whole).

Scores are output in JSON format, for example: {

"Method1": 4, "Method2": 5, "Method3": 4

}

For each method in a given query, GPT-4 assigns one integer score within this range based on the overall visual quality of the layered poster, including the consistency between layers, the plausibility of occlusion and depth, and the readability and layout of the final composed poster. The reported GPT-4 score in the main paper is obtained by averaging these integer scores over all test cases and then linearly normalizing the result to [0,1].

Example of task prompt and evaluation. Prompt: “You will evaluate layered poster results produced by multiple methods under the same input. For each method, inspect

the full composited poster, the text layer, the foreground layer(s), and the background layer, and assign a single integer score in {1,2,3,4,5} based on visual consistency between layers, plausibility of occlusion and depth, and readability and layout of the final composed poster.”

Images: [Upload the layered poster results] Evaluation: The assistant scores each method from 1 to 5 and returns the result in JSON format.

### B. User Study

This subsection gives additional information about the user study conducted to evaluate OmniPSD in both text-to-PSD and image-to-PSD settings. We describe the participant pool, the evaluation criteria, and the 5-point Likert rating protocol, and we summarize how subjective feedback from designers and students supports the quantitative improvements reported in the main paper.

Text-to-PSD. In the text-to-PSD setting, participants compared OmniPSD with LayerDiffuse-SDXL and GPTImage-1 on 50 text prompts. For each generated layered poster, they rated two criteria on a 5-point Likert scale: (1) layering reasonableness (whether foreground, background, and text are separated in a semantically meaningful way with plausible occlusion and depth), and (2) overall preference (the overall visual appeal and usability of the final composed poster, including readability and layout). As summarized in Tab. 6, OmniPSD achieves the highest mean scores on both criteria, clearly outperforming the baselines.

Table 6. User study results for the text-to-PSD setting.

Metric LayerDiffuse-SDXL GPT-Image-1 OmniPSD

Layering reasonableness 3.33 3.89 4.39 Overall preference 3.39 3.78 4.50

Image-to-PSD. In the image-to-PSD setting, participants evaluated 50 poster images decomposed by OmniPSD and the same two baselines. For each decomposed result, they rated three criteria on a 5-point Likert scale: (1) reconstruction consistency (how well the recomposed poster from the predicted layers matches the original input image in content and structure), (2) layering reasonableness (whether the recovered layers form a clean and plausible decomposition with correct occlusion and depth), and (3) overall preference (the perceived quality and practical usability of the layered result as a design asset). Tab. 7 shows that OmniPSD again obtains the highest mean scores on all three criteria, with consistent gains over the baselines.

Across both settings, designers particularly praised OmniPSD for its “clear layer separation” and “realistic transparency,” which enables direct reuse in professional editing workflows. These results confirm that OmniPSD provides superior structural consistency and practical value for real-

Table 7. User study results for the image-to-PSD setting.

Metric Kontext Nano-Banana GPT-Image-1 OmniPSD

Reconstruction consistency 3.05 4.06 4.11 4.56 Layering reasonableness 3.44 4.16 4.22 4.61 Overall preference 3.39 4.33 4.28 4.72

world design generation and reconstruction.

### C. More Results

In this subsection, we present additional qualitative results of OmniPSD in both image-to-PSD reconstruction and textto-PSD synthesis. These visual examples cover diverse layouts and contents, illustrating the clarity of the recovered layers, the realism of transparency and occlusion, and the overall visual quality of the final composed posters.

Image-to-PSD reconstruction. Figure 7 shows more examples where OmniPSD decomposes input poster images into layered PSD files and then recomposes them. The reconstructions exhibit high fidelity to the original designs while preserving clean layer boundaries that are convenient for downstream editing.

Text-to-PSD synthesis. Figure 8 presents additional OmniPSD results in the text-to-PSD setting. Given only textual descriptions, our method synthesizes layered posters with coherent foreground elements, legible text, and visually consistent backgrounds, demonstrating its versatility as a generative design tool.

### D. User Interface

In this subsection, we present the interactive user interface of OmniPSD and demonstrate typical editing workflows on a representative poster example. Starting from a useruploaded image, OmniPSD automatically infers a layered representation that separates text, foreground objects, and background regions into editable components. Through intuitive point-and-click operations, users can modify textual content, remove or replace the background, and delete or adjust individual graphical elements while preserving the overall layout and visual coherence of the design. This interface illustrates how OmniPSD couples high-quality layer decomposition with practical, user-friendly tools for realworld poster editing and creation.

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

|[Figure 197]|
|---|
|[Figure 198]|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|
|[Figure 202]|

|[Figure 203]|
|---|
|[Figure 204]|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|
|[Figure 208]|

|[Figure 209]|
|---|
|[Figure 210]|

|[Figure 211]| | |
|---|---|---|
| |[Figure 212]| |

|[Figure 213]|
|---|
|[Figure 214]|

###### Figure 7. More generation results of OmniPSD image-to-PSD reconstruction.

|[Figure 215]<br><br>[Figure 216]|
|---|

|[Figure 217]|[Figure 218]|[Figure 219]|
|---|---|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

{“poster”: “The complete poster features a romantic theme centered around a heart-shaped ...”, “foreground”: “The foreground content includes a heart-shaped gift box with a pink ribbon and several red roses ...”, “midground”: “The midground content consists of abstract shapes, including a large diamond shape with ...”, “background”: “The background content features a soft pink hue with a subtle texture of roses and ..."}

|[Figure 222]|[Figure 223]|[Figure 224]|
|---|---|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]<br><br>[Figure 228]|
|---|

{“poster”: “The complete poster features a vibrant and modern design centered around a striking red rose …”, “foreground”: “The foreground consists of a detailed red rose, prominently placed in the center of ...”, “midground”: “The midground features a series of geometric shapes in shades ...”, “background”: “The background is a solid blue with a large yellow circle in the center, complemented by a pink triangular ..."}

|[Figure 229]<br><br>[Figure 230]|
|---|

|[Figure 231]|[Figure 232]|[Figure 233]|
|---|---|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

{“poster”: “The complete poster features a vibrant and modern design centered around virtual reality. It showcases two ...”, “foreground”: “The foreground content includes a large percentage symbol in orange ...”, “midground”: “The midground features two individuals, a man and a woman, both wearing VR headsets ...", "background": "The background consists of a smooth gradient transitioning from dark purple to a ..."}

|[Figure 236]<br><br>[Figure 237]|
|---|

|[Figure 238]|[Figure 239]|[Figure 240]|
|---|---|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

{"poster": "The complete poster features a vibrant blue abstract design with swirling patterns and bubbles, set against a dark ...", "foreground": "The foreground content includes two blue star-shaped elements and a vertical blue bar, ...", "midground": "The midground consists of a large

blue abstract shape ...", "background": "The background is a smooth gradient transitioning from dark blue at the top to a lighter shade at ..."}

|[Figure 243]<br><br>[Figure 244]|
|---|

|[Figure 245]|[Figure 246]|[Figure 247]|
|---|---|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

{“poster”: “The poster features a vibrant and cheerful theme, showcasing a joyful individual surrounded by colorful ...”, “foreground”: “The foreground content includes a circular image of a smiling person with long, wavy hair, adorned ...”, “midground”: “The midground consists of two large, stylized green flowers …”, “background”: “The background features a soft, blended gradient of colors, including shades of orange …"}

Figure 8. More generation results of OmniPSD text-to-PSD synthesis.

[Figure 250]

[Figure 251]

[Figure 252]

(a) User Inference (b) Eidt Text (c) Remove background

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

(d) Remove Text (e) Remove element

Figure 9. User interface and functional demonstration of OmniPSD. Given a user-uploaded poster image, OmniPSD enables the addition, removal, and editing of textual and graphical elements.

