## Controllable Layer Decomposition for Reversible Multi-Layer Image Generation

Zihao Liu1*, Zunnan Xu1*, Shi Shu1, Jun Zhou1†, Ruicheng Zhang1,2, Zhenchao Tang2, Xiu Li1†

1Tsinghua University, 2Sun Yat-sen University

Origin Bbox Result Origin Bbox Result

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

# arXiv:2511.16249v2[cs.GR]25Nov2025

[Figure 13]

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

Figure 1. Given an image and its bounding boxes, our framework generates a clean foreground layer for each box and a single, coherent background. Our method can cleanly separate every foreground layer even when objects are crowded and heavily overlapping, while keeping boundaries sharp, depth order intact, and the composite visually coherent.

### Abstract

fine-grained control; and Multi-Layer Conditional Adapter (MLCA), which injects target image information into multilayer tokens to achieve precise conditional generation. To enable a comprehensive evaluation, we build a new benchmark and introduce tailored evaluation metrics. Experimental results show that CLD consistently outperforms existing methods in both decomposition quality and controllability. Furthermore, the separated layers produced by CLD can be directly manipulated in commonly used design tools such as PowerPoint, highlighting its practical value and applicability in real-world creative workflows. Our project is available at CLD.

This work presents Controllable Layer Decomposition (CLD), a method for achieving fine-grained and controllable multi-layer separation of raster images. In practical workflows, designers typically generate and edit each RGBA layer independently before compositing them into a final raster image. However, this process is irreversible: once composited, layer-level editing is no longer possible. Existing methods commonly rely on image matting and inpainting, but remain limited in controllability and segmentation precision. To address these challenges, we propose two key modules: LayerDecompose-DiT (LD-DiT), which decouples image elements into distinct layers and enables

*Equal Contribution. †Corresponding author.

### 1. Introduction

In the design of posters, advertisements, and other visual media, designers typically do not directly work on a single raster image. Instead, they create and edit foreground elements across multiple RGBA layers. Specifically, designers use tools such as Adobe Photoshop or PowerPoint to construct foreground elements at the layer level and then overlay multiple foreground layers onto a background layer to produce the final image. However, this compositing process is irreversible: once multiple RGBA layers are merged into a single raster image, the original multi-layer information cannot be recovered. Consequently, when only a raster image is available, precise layer-level editing and adjustments become extremely challenging. Accurate layer decomposition from raster images can effectively address this issue, enabling designers to perform efficient and controllable edits on the separated layers.

In this work, we investigate the problem of usercontrollable layer decomposition, aiming to split a single raster image into a set of independent layers based on userprovided bounding boxes. Existing approaches, such as decomposition methods for natural images [1, 2, 16, 43], often suffer from incomplete foreground extraction or unwanted artifacts, including messy edges or background color leaking into foreground layers. Some studies [4, 45, 60] adopt a modular, multi-stage pipeline, dividing the task into object detection [38–40], segmentation [12, 13, 21, 48, 49], image matting [20, 41, 54], and image inpainting [5, 17, 31], with each stage handled by a dedicated pre-trained model. However, such pipelines are prone to error propagation: mistakes in early stages can adversely affect subsequent stages, degrading overall layer decomposition quality. To address these limitations, LayerD [42] proposed a fully automatic graphic layer decomposition method, iteratively performing image matting for top-layer extraction and image inpainting for background completion. However, LayerD entirely relies on the matting model to identify top-layer elements, leaving users with no control over the final decomposition results.

To overcome these limitations, we propose a controllable layer decomposition method, which allows users to specify desired decomposition outcomes via bounding boxes. Our core model uses pre-trained DiT models (e.g., FLUX.1[dev] [23]), fully exploiting their strong image generation capabilities to produce layer decomposition results of higher quality. Figure 1 shows the results of our method. It can be seen that the model is able to generate the corresponding foreground and background layers based on the provided bounding boxes, achieving excellent overall visual quality. Moreover, previous evaluation metrics for layer decomposition were limited by uncontrollable generated results, lacking standardized benchmarks. By introducing bounding boxes, we can define ideal target layers

and propose a new benchmark. Specifically, we establish a benchmark for controllable layer decomposition, built on the PrismLayersPro [3] dataset, with new metric dimensions tailored for controllable settings. Our main contributions are summarized as follows:

- • We propose CLD (Controllable Layer Decomposition), a controllable framework for raster image layer decomposition that leverages DiT’s generation and reasoning capabilities to achieve high-quality layer separation.
- • We introduce a new benchmark to assess the performance of layer decomposition from multiple dimensions.
- • Experimental results demonstrate that CLD outperforms baseline methods in both controllability and generation quality, and separated layers can be applied directly to downstream graphic design and editing tasks.

### 2. Related work

Image Layer Decomposition. Image layer decomposition aims to divide an image into a set of composable layers that can be re-synthesized to reconstruct the original image. Early approaches are mostly color-based, grouping pixels by color similarity and focusing on digital painting or natural images [1, 2, 9, 22, 43, 44]. Other studies explore objectlevel decomposition in natural scenes [15, 27, 28, 32, 50, 52, 55, 58, 59]. Recent studies have explored multi-layer image generation, using generative models to directly synthesize and disentangle foreground and background layers in a unified framework [3, 11, 14, 29, 30, 35, 36, 46, 56]. For example, Huang et al. [14] introduced DreamLayer, which explicitly models the interaction between transparent foreground and background layers for coherent text-driven generation. However, this method is designed for text-toimage synthesis and is not directly applicable to our layer decomposition task. In design images such as posters and advertisements, multiple visual elements coexist, making decomposition challenging in terms of fine-grained separation and controllability. Recent works [4, 42] have explored this direction. Chen et al.[4] proposed a pipeline combining visual language models (VLM) with SAM[21], while Suzuki et al. [42] introduced LayerD, which separates layers via image matting and inpainting. However, LayerD relies entirely on the matting model to determine top-layer elements, offering no user control and tending to produce coarse, element-level separations. To overcome these limitations, we propose a DiT-based framework that enables controllable, fine-grained layer generation guided by userprovided bounding boxes.

Image matting and Element extraction. Natural image matting optimizes the compositing equation I = αF + (1 − α)B for α,F,B using inputs like trimaps or scribbles [25, 34, 47]. Although trimap-free methods [19, 53] reduce manual effort, they are optimized for photographic statistics and fail on the clean edges, solid colors, and repet-

itive textures common in graphic designs.

Zero-shot segmentation foundation models offer an alternative. SAM [21] and SAM 2 [37] produce binary masks from prompts but ignore partial transparency. HQSAM [18] refines boundaries but remains at the instance level, failing to model soft transitions. ZIM [20] adapts SAM for zero-shot matting, but it is trained for natural objects and hallucinates alpha on solid-color graphics. Furthermore, none of these models support sequential decomposition; users cannot extract nested elements (e.g., “segment the red subtitle”) without providing a new manual trimap. Our framework solves this. We treat the bounding box as a layer request rather than a rough separator. The network predicts a hard binary mask (for crisp boundaries) and a residual alpha map (for optional soft transitions), enabling the extraction of both solid text and semi-transparent shadows. Because the model is conditioned on the box, it can be applied recursively: after removing the top element, the model can process the revealed region, yielding a full layer stack. Extensive experiments (Sec. 4) show our strategy outperforms matting-based and SAM-based baselines on graphic-design benchmarks in edge accuracy, layer consistency, and user controllability.

### 3. Method

#### 3.1. Preliminaries

Flow Matching [7, 26] is an alternative to diffusion models by directly learning the velocity field that transports noise to data. Unlike noise-prediction diffusion models (e.g., DDPM [8]), it formulates generation as a continuous dynamical system, enabling a more stable and efficient sampling process. Specifically, given a real sample x0 and a noise sample x1, Flow Matching learns a velocity field vθ(·) to approximate the optimal transport between them. The objective aligns the predicted velocity with the true derivative dxt

dt along the interpolated path xt = (1 − t)x0 + tx1. By modeling the velocity rather than the noise residual, it achieves smoother optimization and more stable gradients in high-dimensional spaces. Due to its faster convergence and higher generation quality, we adopt Flow Matching as the training paradigm for CLD and fine-tune the backbone via LoRA [10]. The overall training objective is:

LFM = Et∼U(0,1) |vθ(xt,t,ctext,cimg) − (x1 − x0)|2 ,

(1) where ctext and cimg denote the text and image conditions.

#### 3.2. Problem Formulation

Given an raster image I ∈ RH×W×3, such as a composited design image, we regard it as consisting of a background layer Ibg ∈ RH×W×4 and multiple foreground layers {Iifg ∈ RH

i×Wi×4}Ni=1−1, where N denotes the total

number of layers. Each layer is represented in RGBA format to preserve transparency information. To enable controllable layer decomposition, the user can provide a set of bounding boxes Bu = {B1,...,BN−1}, where each Bi = (xli,yil,xri,yir) specifies the top-left and bottom-right coordinates of a target region. The background layer is defined by B0 = (0,0,H,W), and together they form the complete bounding box set B = concat(B0,Bu). Our objective is to generate a collection of N disentangled layers D = {D0,D1,...,DN−1}, where each Di represents an independent RGBA layer aligned with its corresponding bounding box Bi. The decomposition aims to ensure spatial alignment and visual consistency across all layers, thereby achieving fine-grained and controllable layer separation.

#### 3.3. LayerDecompose-DiT

LayerDecompose-DiT (LD-DiT) is designed to simultaneously generate visual tokens for the background layer and multiple foreground layers, enabling fine-grained disentanglement and hierarchical modeling of image structures. At the architectural level, we build upon FLUX.1[dev] [23], which adopts the Multimodal Diffusion Transformer (MMDiT) as its backbone. MMDiT employs two distinct sets of network weights to separately process text tokens and image tokens, enabling efficient multimodal interaction. However, the original MMDiT is limited to singleimage generation and cannot directly support multi-layer synthesis. To address this limitation, we introduce several key enhancements. Specifically, the input visual tokens are cropped according to user-provided bounding boxes, and the target regional tokens are concatenated into a unified token sequence, which is then processed by MMDiT for denoising multi-layer generation. Moreover, to enable the model to better capture hierarchical dependencies within this sequence, we incorporate a Layer-Aware Rotary Position Embedding (LA-RoPE) to jointly encode spatial and inter-layer positional relationships. As shown in Figure 2, the conditional inputs to LD-DiT include the bounding boxes B, the original image I, and a text prompt T. The text prompt can either be user-provided to semantically guide the generation process or automatically extracted from the original image using a VLM. The noisy inputs are obtained by adding Gaussian noise to the cropped multi-layer latent sequence, which is then denoised by MMDiT and decoded into a set of layered RGBA images. To enhance global consistency in multi-layer generation, we introduce the composite image as an auxiliary generation target. Unlike layerwise blending, it serves as a direct reconstruction of the input image. During training, the input image is embedded using the same visual encoder as the layer tokens and prepended to the layer-specific sequence, with a bounding box (0,0,H,W) covering the entire image. Formally, given layer tokens Xlayers = {x0,x1,...,xN−1} and the com-

Hidden States

Hidden States

[Figure 83]

[Figure 84]

Origin Image （Latent Space)

[Figure 85]

Generate random noise

[Figure 86]

[Figure 87]

Flatten

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

LoRA Layer

LoRA Layer

[Figure 95]

[Figure 96]

[Figure 97]

Region Layout

Linear

[Figure 98]

LA-RoPE

Bboxwised Cropping

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

MultiLayer Decoder

MMDiT Layers

MMDiT Layers

[Figure 104]

[Figure 105]

###### MLCA

[Figure 106]

[Figure 107]

Guidance Tokens

[Figure 108]

[Figure 109]

###### VAE

Guidance Tokens

[Figure 110]

[Figure 111]

[Figure 112]

Flatten

Origin Image

[Figure 113]

CLIP & T5

[Figure 114]

[Figure 115]

“This is a furry style image. VLM The image presents .”

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Region Layout

Text Descrition

(a) Overall architecture of the Controllable Layer Decomposition (CLD) framework.

(b) The Multi-Layer Conditional Adapter (MLCA).

- Figure 2. Our framework utilizes a main backbone and a parallel control module for precise layer decomposition. (a) The overall CLD architecture, showing the LayerDecompose-DiT (LD-DiT) backbone responsible for generating the multi-layer latent. (b) The detailed structure of the Multi-Layer Conditional Adapter (MLCA). MLCA additively fuses features from the conditional image with the LD-DiT’s hidden states, then performs hierarchical cropping based on the input bounding boxes to create a multi-layer guidance token sequence.

[Figure 120]

[Figure 121]

InputLinear

Vision Transformer

OutputLinear

Denoised Latent

Reshape

Multi-Layer Image

[Figure 122]

[Figure 123]

- (0, 0, 0) (0, 0, 1)
- (0, 1, 0) (0, 1, 1)

[Figure 124]

[Figure 125]

- (1, 0, 0) (1, 0, 1)
- (1, 1, 0) (1, 1, 1)

Width

Height

[Figure 126]

[Figure 127]

- (2, 0, 0) (2, 0, 1)
- (2, 1, 0) (2, 1, 1)

Image Layers

Position index (a) Decoder Architecture Adopted in Our Work (b) LA-RoPE

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- Figure 3. Overview of our adapted Multi-Layer RGBA image Decoder Architecture and Layer-Aware Rotary Position Encoding.

and stronger inter-layer coherence. Based on these observations, we choose to adapt the Multi-Layer Transparent Image Autoencoder to fit our multi-layer generation pipeline, enabling LD-DiT to accurately maintain consistent structures across layers. Figure 3 presents the decoder architecture used in our framework.

LA-RoPE. Traditional positional encodings typically model spatial relations within a single layer and fail to capture hierarchical dependencies across layers. To address this, we propose Layer-Aware Rotary Position Embedding (LA-RoPE), which jointly encodes both spatial and interlayer positional information in a unified hierarchical form (see Fig. 3). Each visual token is indexed by [l,h,w], denoting its layer, height, and width positions. By integrating these hierarchical indices into the queries and keys of selfattention, LA-RoPE enables reasoning across layers and enhances structural coherence and controllability in multilayer generation. Formally, let the n-th query and m-th key be qn,km ∈ Rd

posite token xcomp, the input to MMDiT is:

##### Xinput = [xcomp;x0;x1;...;xN−1], (2)

where [;] denotes sequence concatenation. This setup enables the model to propagate global context through the composite token, reinforcing visual and structural coherence across layers.

head, each split along the channel dimension into three parts:

qn = {qnl ,qnh,qnw}, km = {kml ,kmh ,kmw}. (3) The (n,m)-th element of the attention matrix is:

Latent Decoder. Conventional VAEs are limited to processing RGB inputs and are unable to capture transparency information. To address this limitation, our model explicitly incorporates transparency-aware encoding. We adapt two alternative designs into our framework: the MultiLayer Transparent Image Autoencoder from ART [35] and the Latent Transparency module proposed in LayerDiffuse [56]. Through systematic comparison, we find that the former achieves better preservation of alpha information

c

##### n−pcm)θ , (4)

Re qnc(kmc )∗ei(p

A(n,m) =

c∈{l,h,w}

where pn = {ln,hn,wn} denotes the 3D position index of the n-th token, (kmc )∗ is the complex conjugate of kmc , θ ∈ R is a preset nonzero constant, and Re[·] represents the real part of a complex number.

#### 3.4. Multi-Layer Conditional Adapter

Since the input to MMDiT consists of a multi-layer sequence of visual tokens, existing conditional control frameworks, such as ControlNet [57], are primarily designed for single-layer image generation and therefore struggle to capture the complex dependencies among multiple visual layers. In particular, ControlNet performs conditioning through a global residual pathway, which offers only coarse-level guidance and lacks the ability to maintain structural alignment and inter-layer consistency.

To overcome these limitations, we introduce the MultiLayer Conditional Adapter (MLCA), a dedicated conditioning module tailored for multi-layer visual generation. MLCA explicitly models hierarchical relationships between layers and dynamically injects layer-specific conditions into the diffusion process. By aligning conditional features with each layer’s token representation, MLCA enables precise, fine-grained control while preserving inter-layer coherence throughout generation. Specifically, the input conditional image I is first encoded into its latent representation zimg via a VAE encoder EVAE(·):

′×W′×C. (5)

zimg = EVAE(I) ∈ RH

Next, a linear mapping network Linear(·) projects the latent feature into the same feature space as the hidden states h ∈ RL×D of the main MMDiT model, obtaining zˆimg:

##### ′×W′×D. (6)

zˆimg = Linear(zimg) ∈ RH

Subsequently, we crop zˆimg according to the input bounding boxes B, slicing the corresponding regions and flattening

- them into a multi-layer guidance token sequence himg. This sequence is structurally aligned with the MMDiT input, enabling parallel modeling across layers within the Transformer’s self-attention mechanism for both explicit alignment and implicit coordination:

himg = Flatten(Crop(zˆimg,B)) ∈ RL×D. (7)

We then perform additive conditioning by summing the mapped conditional features with the MMDiT hidden states to obtain the fused feature representation hˆ:

##### hˆ = h + himg. (8)

Afterward, hˆ is fed into LD-DiT as a crucial guidance signal during the denoising process. This design allows the Transformer to dynamically coordinate inter-layer interactions while maintaining the individual objectives of each layer. By decoupling layer-wise conditioning and enabling effective cross-layer communication, MLCA delivers precise control, stronger inter-layer coherence, and more consistent multi-layer decomposition compared to traditional single-layer control frameworks.

#### 3.5. Dual-Condition Classifier-Free Guidance

Building on the fine-grained guidance from MLCA, we further introduce a dual-condition Classifier-Free Guidance (CFG) strategy to improve inter-layer consistency and semantic controllability. The model takes two conditional inputs: a text description ctext and a reference image cimg. During inference, the process splits into a conditional and an unconditional branch. Unlike conventional CFG that discards all conditions in the unconditional branch, we retain cimg while nullifying ctext.

The reason for retaining cimg in the unconditional branch is twofold. (1) It provides a structural anchor, ensuring that both conditional and unconditional branches remain aligned in terms of spatial layout and hierarchical structure. (2) It enables semantic incremental separation: by subtracting the unconditional prediction from the conditional one, shared structural information is canceled, isolating the text-driven semantic contribution.

The predicted flow velocity vˆ is computed as:

vˆ = vθ(xt,t,∅,cimg)

+ s · vθ(xt,t,ctext,cimg) − vθ(xt,t,∅,cimg) ,

(9)

where s denotes the CFG scaling factor, and vθ is the flow velocity predicted by the model under different conditions.

### 4. Experiment 4.1. Experiment Setting

Dataset Preparation. We employ PrismLayersPro [3], one of the latest and largest high-quality multi-layer image datasets, as the primary source for both training and evaluation. This dataset contains approximately 20K groups of high-quality layered samples, where each group consists of a complete composite image, multiple corresponding transparent layers (in RGBA format), and an associated textual description. PrismLayersPro spans 21 distinct categories, encompassing a wide range of visual styles such as 3D, cartoon, and others, exhibiting high diversity and visual complexity. For data partitioning, we divide PrismLayersPro into training, validation, and testing sets with a ratio of 90% / 5% / 5%, respectively. Based on this partitioning, we define a set of dedicated evaluation metrics and construct a new benchmark for multi-layer decomposition.

Implementation details. Our backbone is built upon the latest FLUX.1[dev] model, a state-of-the-art Multimodal Diffusion Transformer (MMDiT) architecture. We fine-tune this model on the partitioned PrismLayersPro training set using LoRA adaptation, with the LoRA rank set to 64. The training is performed with the Prodigy optimizer, a learning rate of 1, and a batch size of 4. The model is trained for 25K iterations on images with a resolution of 1024 × 1024.

- Table 1. Comparison with LayerD [42] on the Crello [51] dataset test set. LayerD Metrics refers to the evaluation metrics proposed in LayerD, while Q-Insight Metrics denotes the evaluation metrics we introduce based on the Q-Insight [24] model.

Method

|LayerD Metrics<br><br>|Q-Insight Metrics<br><br>|User Study|
|---|---|---|
|RGB L1↓<br><br>Alpha Soft IoU↑<br><br>Unified Score↓<br><br>|Semantic Consistency↑<br><br>Visual Fidelity↑ Editability↑|Content Completeness↑<br><br>Semantic Consistency↑<br><br>Visual Quality↑|

LayerD[ICCV 25] 0.0653 0.7055 0.1799 3.8658 3.6773 4.0011 19% 24% 9% Ours 0.0474 0.7771 0.1352 3.9157 3.7334 4.0462 81% 76% 91%

- Table 2. Ablation study on different model variants. We examine four configurations: (1) using the decoder adapted from LayerDiffuse [56],

(2) disabling the image condition in the CFG unconditional branch, (3) removing the composite image prediction objective, and (4) the full model. The ART [35] adapted decoder is used for all experiments except the first configuration.

|Layer-level<br><br>|Mask-level<br><br>|Reconstruction|
|---|---|---|
|PSNR↑ SSIM↑ FID↓<br><br>|IoU↑ F1↑<br><br>|PSNR↑ SSIM↑ FID↓|

###### Variant

- (1) Using adapted LayerDiffuse decoder 24.429 0.822 24.016 0.847 0.902 22.570 0.824 36.127

- (2) w/o image condition in CFG-Uncond 21.691 0.767 69.910 0.576 0.692 20.658 0.819 103.250

- (3) w/o composite image prediction 26.211 0.845 21.453 0.867 0.918 27.240 0.926 15.638

- (4) Full Model 27.646 0.874 19.413 0.867 0.920 29.825 0.945 11.464

#### 4.2. Metrics

Since there is currently no established benchmark for evaluating deterministic multi-layer image separation tasks, we propose a comprehensive, multi-dimensional evaluation to assess model performance across layer quality, transparency accuracy, and overall reconstruction fidelity. Our evaluation system is divided into three main categories:

- • Layer-level: These metrics assess the visual quality of each individual layer, using PSNR and SSIM for pixellevel and structural consistency, and FID [6] for perceptual similarity between generated and real images.
- • Mask-level: We use IoU (Intersection over Union) and F1 score to evaluate the alignment between generated and ground-truth alpha masks, reflecting accuracy in modeling transparent regions and boundaries.
- • Reconstruction: All generated RGBA layers are composited back into an RGB image. We then evaluate this reconstructed image with PSNR, SSIM, and FID to assess global coherence and overall fidelity restoration.

All generated images are in RGBA format, while evaluation metrics (PSNR, SSIM, FID) are defined only for RGB images. To ensure metric compatibility and fair comparison, we first convert the RGBA image into RGB format. Specifically, We use a fixed neutral gray background Igrey with pixel values (0.5,0.5,0.5) within the range [0,1], and

- then composite it into Itgt with the RGBA image’s RGB channels Irgb and alpha channel Iα as follows:

Itgt = Irgb × Iα + Igrey × (1 − Iα). (10) The use of a neutral gray background ensures robust blend-

ing with minimal color distortion, preserving transparent regions and enabling consistent evaluation across methods.

#### 4.3. Quantitative Result

Currently, there is no existing work that performs a fully comparable multi-layer generative decomposition under user-guided conditions. Therefore, we select LayerD [42], a state-of-the-art method for automatic layer separation, as our primary baseline for comparison since it is the most recent and directly comparable approach in both task definition and output representation. LayerD [42] performs layer separation using image matting and inpainting. Operating on a single composite image, it relies entirely on the model’s inference, offering no explicit control over layer structure. To address layer correspondence, LayerD employs order-aware alignment via DTW [33] and computes evaluation metrics based on the matched layers. It is important to note that our method differs from LayerD in that it incorporates user-provided bounding boxes to guide the layer decomposition. This bounding-box guidance introduces explicit structural control over the generated layers, which LayerD does not utilize. Consequently, the two methods may produce different generation styles due to the additional structural guidance in our approach. We perform this comparison to demonstrate how the combination of bounding-box guidance and the proposed model architecture affects controllability, precision, and visual quality.

Table 1 presents a comparison between LayerD and our proposed method under the evaluation metrics defined by LayerD, using the Crello [51] dataset as the test set, which we adopt for fair comparison since it was also used in

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

OriginLayerDOurs

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

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

- Figure 4. Unlike LayerD [42], which offers coarse separation and lacks user control, our method uses bounding boxes to guide a more fine-grained and controllable process. This results in better precision, visual quality, and hierarchical consistency in complex scenarios.

#### 4.4. Qualitative Result

LayerD’s experiments. To adapt to our task, the RGB L1 metric is calculated by first converting RGBA images into RGB format according to Equation 10, and then applying the L1 distance calculation. The Unified Score metric jointly considers the performance of both RGB L1 and Alpha soft IoU, and is formulated as follows:

Comparison with LayerD. Figure 4 shows a visual comparison of layer separation between our method and LayerD, and the results of the user study are presented in Table 1. Our model produces more detailed and fine-grained layer structures, benefiting from user-specified bounding boxes that guide the decomposition process. This additional guidance allows higher controllability and flexibility, particularly in complex scenes. Compared to LayerD’s automatic separation, our method preserves semantic and structural alignment across layers, and produces results that better align with user preferences.

RGB L1 + (1 − Alpha soft IoU) 2

Unified Score =

. (11)

Experimental results show that our method achieves better results than LayerD across all evaluation metrics, benefiting from the introduction of bounding-box guidance and the powerful image generation capability of DiT.

Comparison with Image Matting and Segmentation Methods. Recent layer separation methods, including our baseline LayerD [42], largely rely on image matting or segmentation models, whose performance directly impacts subsequent layer separation and background inpainting. These models often struggle in scenarios containing with numerous foreground objects or complex hierarchical relationships, limiting the quality of the decomposition. To illustrate this, we compare our approach with widely used matting and segmentation methods, ZIM [20] and SAM2 [37]. Figure 6 shows that in scenes with overlapping elements, complex layouts, or text-rich regions, these methods tend to produce blurred boundaries, fragmented regions, or mis-segmented objects. In contrast, our method performs multi-layer decomposition in a unified generative process. Leveraging the global reasoning and contextual

To further assess our model’s performance, we employ Q-Insight [24], a state-of-the-art image quality evaluation model that leverages MLLM reasoning and reinforcement learning to provide interpretable, zero-shot perceptual quality assessment. We design three evaluation dimensions: Semantic Consistency, Visual Fidelity, and Editability. As shown in Table 1, our method achieves better performance than LayerD across all three dimensions, confirming its comprehensive advantages in semantic preservation, visual quality, and editability.

It is worth noting that the LayerD model is trained on the Crello [51] training set, while our model is trained solely on the PrismLayerPro [3] dataset. Despite this, our model still achieves better results on the Crello test set, further demonstrating its strong generalization ability and robustness across datasets.

Origin Ours SAM2 ZIM

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

OriginGTOurs(Fullmodel)cfg:w/oimage-guideDecoder:LayerDiffuse

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

- Figure 5. Ablation study on the impact of decoder choices and the CFG unconditional image condition.

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

modeling of DiT, it accurately infers layer boundaries and maintains visual and semantic coherence, even under heavy occlusion. The comparison results indicate that combining generative end-to-end decomposition with user-guided structural control provides inherent advantages over existing layer decomposition methods based on image matting or segmentation, offering higher fidelity, clearer layer separation, and stronger inter-layer consistency.

Figure 6. Unlike segmentation (SAM2 [37]) and matting (ZIM [20]) models, which fail on complex design images, our generative approach produces clearer, more coherent layering.

and layer fidelity.

#### 4.5. Ablation Study

CFG unconditional branch. Our framework uses two main conditions: a text description and the original image. During inference, we apply Classifier-Free Guidance (CFG), dropping the text condition in the unconditional branch but retaining the original image. Keeping the image in the unconditional branch significantly improves results (Table 2) and prevents background leakage in foreground layers (Figure 5), highlighting its importance for spatial consistency and semantic coherence.

We conducted systematic ablation studies on our proposed layer decomposition model to evaluate the contribution of each component to the overall performance. The experiments were performed on a test set derived from the PrismLayersPro [3] dataset, and model performance was analyzed using the multi-dimensional evaluation metrics defined in Sec. 4.2.

Incorporating the composite image into the generation target. In our model design, we include the composite image as an auxiliary generation target to enable information flow between the overall image and individual RGBA layers, improving global consistency and reconstructability. As shown in Table 2, this design enhances Layer-level and Mask-level metrics, with the largest gains in the Reconstruction metric, demonstrating improved visual coherence

RGBA image decoder. Traditional VAE decoders only handle RGB. We adapted two RGBA strategies for CLD: (1) LayerDiffuse [56], which reconstructs RGB and alpha separately, and (2) ART [35], which directly generates RGBA from a ViT-based latent. As shown in Figure 5 and Table 2, the ART-adapted decoder achieves better metrics and cleaner foreground boundaries.

Origin Bbox Decomposed Layers Remove Layers

- [2] Ya˘giz Aksoy, Tun¸c Ozan Aydin, Aljoˇsa Smoli´c, and Marc Pollefeys. Unmixing-based soft color segmentation for image manipulation. ACM Transactions on Graphics (TOG), 36(2):1–19, 2017. 2
- [3] Junwen Chen, Heyang Jiang, Yanbin Wang, Keming Wu, Ji Li, Chao Zhang, Keiji Yanai, Dong Chen, and Yuhui Yuan. Prismlayers: Open data for high-quality multilayer transparent image generative models. arXiv preprint arXiv:2505.22523, 2025. 2, 5, 7, 8
- [4] Jingye Chen, Zhaowen Wang, Nanxuan Zhao, Li Zhang, Difan Liu, Jimei Yang, and Qifeng Chen. Rethinking layered graphic design generation with a top-down approach. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16861–16870, 2025. 2, 1
- [5] Ciprian Corneanu, Raghudeep Gadde, and Aleix M Martinez. Latentpaint: Image inpainting in latent space with diffusion models. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 4334– 4343, 2024. 2
- [6] DC Dowson and BV666017 Landau. The fr´echet distance between multivariate normal distributions. Journal of multivariate analysis, 12(3):450–455, 1982. 6
- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 3

- [8] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [9] Daichi Horita, Kiyoharu Aizawa, Ryohei Suzuki, Taizan Yonetsuji, and Huachun Zhu. Fast nonlinear image unblending. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2051–2059, 2022. 2
- [10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 3
- [11] Dingbang Huang, Wenbo Li, Yifei Zhao, Xinyu Pan, Yanhong Zeng, and Bo Dai. Psdiffusion: Harmonized multilayer image generation via layout and appearance alignment. arXiv preprint arXiv:2505.11468, 2025. 2
- [12] Jiaqi Huang, Zunnan Xu, Ting Liu, Yong Liu, Haonan Han, Kehong Yuan, and Xiu Li. Densely connected parameterefficient tuning for referring image segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3653–3661, 2025. 2
- [13] Jiaqi Huang, Zunnan Xu, Jun Zhou, Ting Liu, Yicheng Xiao, Mingwen Ou, Bowen Ji, Xiu Li, and Kehong Yuan. Sam-r1: Leveraging sam for reward feedback in multimodal segmentation via reinforcement learning. arXiv preprint arXiv:2505.22596, 2025. 2
- [14] Junjia Huang, Pengxiang Yan, Jinhang Cai, Jiyang Liu, Zhao Wang, Yitong Wang, Xinglong Wu, and Guanbin Li. Dreamlayer: Simultaneous multi-layer generation via diffusion mode. arXiv preprint arXiv:2503.12838, 2025. 2, 1

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

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Copy & Add Layers

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

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

Change Layout

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

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

Figure 7. Application: Direct Editing on Decomposed Layers. This example showcases direct editing operations on separated layers, including Remove Layers, Copy & Add Layers, and Change Layout. All operations are performed within PowerPoint.

#### 4.6. Application

As shown in Figure 7, we demonstrate a practical application of our method. By manipulating the separated layers, users can conveniently perform various editing operations in common office software such as PowerPoint, including removing layers, copying and adding layers, and adjusting layouts. This layer-based editing paradigm greatly enhances the editability and flexibility of image content, enabling even non-professional users to easily accomplish complex element rearrangement and layout adjustments. This highlights the practical value of our method in realworld design and content creation scenarios.

### 5. Conclusion

In this paper, we present CLD, a method to separate raster images into fine-grained layers, solving a key limitation in post-production. We use bounding boxes for precise, userguided separation and introduce a new multi-dimensional benchmark for this task. Experiments show CLD outperforms existing methods in decomposition quality, controllability, and visual fidelity. The generated layers are directly usable for downstream editing (e.g., removal, recomposition), offering a practical and high-quality solution for creative design and future multi-layer extensions.

### References

[1] Naofumi Akimoto, Huachun Zhu, Yanghua Jin, and Yoshimitsu Aoki. Fast soft color segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8277–8286, 2020. 2

- [15] Phillip Isola and Ce Liu. Scene collaging: Analysis and synthesis of natural images with semantic layers. In Proceedings of the IEEE International Conference on Computer Vision, pages 3048–3055, 2013. 2
- [16] Xiaoyu Jin, Zunnan Xu, Mingwen Ou, and Wenming Yang. Alignment is all you need: A training-free augmentation strategy for pose-guided video generation. arXiv preprint arXiv:2408.16506, 2024. 2
- [17] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In European Conference on Computer Vision, pages 150–168. Springer,

2024. 2

- [18] Lei Ke, Mingqiao Ye, Martin Danelljan, Yu-Wing Tai, ChiKeung Tang, Fisher Yu, et al. Segment anything in high quality. Advances in Neural Information Processing Systems, 36: 29914–29934, 2023. 3
- [19] Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, and Rynson WH Lau. Modnet: Real-time trimap-free portrait matting via objective decomposition. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1140– 1147, 2022. 2
- [20] Beomyoung Kim, Chanyong Shin, Joonhyun Jeong, Hyungsik Jung, Se-Yun Lee, Sewhan Chun, Dong-Hyun Hwang, and Joonsang Yu. Zim: Zero-shot image matting for anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23828–23838, 2025. 2, 3, 7, 8
- [21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 2, 3
- [22] Yuki Koyama and Masataka Goto. Decomposing images into layers with advanced color blending. In Computer Graphics Forum, pages 397–407. Wiley Online Library, 2018. 2
- [23] Black Forest Labs. Flux.1 [dev]. https : / / huggingface.co/black-forest-labs/FLUX. 1-dev, 2024. Last accessed 7 March, 2025. 2, 3
- [24] Weiqi Li, Xuanyu Zhang, Shijie Zhao, Yabin Zhang, Junlin Li, Li Zhang, and Jian Zhang. Q-insight: Understanding image quality via visual reinforcement learning. arXiv preprint arXiv:2503.22679, 2025. 6, 7, 1
- [25] Yaoyi Li and Hongtao Lu. Natural image matting via guided contextual attention. In Proceedings of the AAAI conference on artificial intelligence, pages 11450–11457, 2020. 2
- [26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations. 3
- [27] Zhengzhe Liu, Qing Liu, Chirui Chang, Jianming Zhang, Daniil Pakhomov, Haitian Zheng, Zhe Lin, Daniel Cohen-Or, and Chi-Wing Fu. Object-level scene deocclusion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [28] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Pose-

guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4117–4125, 2024. 2

- [29] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024. 2
- [30] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869,

2025. 2

- [31] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Leqi Shen, Chenyang Qi, Jixuan Ying, Chengfei Cai, Zhifeng Li, Heung-Yeung Shum, et al. Follow-your-click: Open-domain regional image animation via motion prompts. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6018–6026, 2025. 2
- [32] Tom Monnier, Elliot Vincent, Jean Ponce, and Mathieu Aubry. Unsupervised layered image decomposition into object prototypes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8640–8650,

2021. 2

- [33] Meinard M¨uller. Information retrieval for music and motion. Springer, 2007. 6
- [34] GyuTae Park, SungJoon Son, JaeYoung Yoo, SeHo Kim, and Nojun Kwak. Matteformer: Transformer-based image matting via prior-tokens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11696–11706, 2022. 2
- [35] Yifan Pu, Yiming Zhao, Zhicong Tang, Ruihong Yin, Haoxing Ye, Yuhui Yuan, Dong Chen, Jianmin Bao, Sirui Zhang, Yanbin Wang, et al. Art: Anonymous region transformer for variable multi-layer transparent image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7952–7962, 2025. 2, 4, 6, 8, 1, 3
- [36] Guocheng Gordon Qian, Ruihang Zhang, Tsai-Shien Chen, Yusuf Dalva, Anujraaj Argo Goyal, Willi Menapace, Ivan Skorokhodov, Meng Dong, Arpit Sahni, Daniil Ostashev, et al. Layercomposer: Interactive personalized t2i via spatially-aware layered canvas. arXiv preprint arXiv:2510.20820, 2025. 2
- [37] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In The Thirteenth International Conference on Learning Representations. 3, 7, 8
- [38] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018. 2
- [39] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016.
- [40] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015. 2

- [41] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Semantic image matting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11120– 11129, 2021. 2
- [42] Tomoyuki Suzuki, Kang-Jun Liu, Naoto Inoue, and Kota Yamaguchi. Layerd: Decomposing raster graphic designs into layers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17783–17792, 2025. 2, 6, 7, 1, 3
- [43] Jianchao Tan, Jyh-Ming Lien, and Yotam Gingold. Decomposing images into layers via rgb-space geometry. ACM Transactions on Graphics (TOG), 36(1):1–14, 2016. 2
- [44] Jianchao Tan, Jose Echevarria, and Yotam Gingold. Efficient palette-based decomposition and recoloring of images via rgbxy-space geometry. ACM Transactions on Graphics (ToG), 37(6):1–10, 2018. 2
- [45] Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, Fei Chen, Steven McDonagh, Gerasimos Lampouras, Ignacio Iacobacci, and Sarah Parisot. Mulan: A multi layer annotated dataset for controllable text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22413–22422, 2024. 2
- [46] Zitong Wang, Hang Zhao, Qianyu Zhou, Xuequan Lu, Xiangtai Li, and Yiren Song. Diffdecompose: Layer-wise decomposition of alpha-composited images via diffusion transformers. arXiv preprint arXiv:2505.21541, 2025. 2
- [47] Ning Xu, Brian Price, Scott Cohen, and Thomas Huang. Deep image matting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2970– 2979, 2017. 2
- [48] Zunnan Xu, Zhihong Chen, Yong Zhang, Yibing Song, Xiang Wan, and Guanbin Li. Bridging vision and language encoders: Parameter-efficient tuning for referring image segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 17503–17512, 2023. 2
- [49] Zunnan Xu, Jiaqi Huang, Ting Liu, Yong Liu, Haonan Han, Kehong Yuan, and Xiu Li. Enhancing fine-grained multimodal alignment via adapters: a parameter-efficient training framework for referring image segmentation. In 2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ ICML 2024), 2024. 2
- [50] Zunnan Xu, Zhentao Yu, Zixiang Zhou, Jun Zhou, Xiaoyu Jin, Fa-Ting Hong, Xiaozhong Ji, Junwei Zhu, Chengfei Cai, Shiyu Tang, et al. Hunyuanportrait: Implicit condition control for enhanced portrait animation. In Proceedings of the

- Computer Vision and Pattern Recognition Conference, pages 15909–15919, 2025. 2
- [51] Kota Yamaguchi. Canvasvae: Learning to generate vector graphic documents. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5481–5489,

2021. 6, 7, 2

- [52] Jinrui Yang, Qing Liu, Yijun Li, Soo Ye Kim, Daniil Pakhomov, Mengwei Ren, Jianming Zhang, Zhe Lin, Cihang Xie, and Yuyin Zhou. Generative image layer decomposition with visual effects. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7643–7653, 2025. 2
- [53] Jingfeng Yao, Xinggang Wang, Shusheng Yang, and Baoyuan Wang. Vitmatte: Boosting image matting with pretrained plain vision transformers. Information Fusion, 103: 102091, 2024. 2
- [54] Qihang Yu, Jianming Zhang, He Zhang, Yilin Wang, Zhe Lin, Ning Xu, Yutong Bai, and Alan Yuille. Mask guided matting via progressive refinement network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1154–1163, 2021. 2
- [55] Xiaohang Zhan, Xingang Pan, Bo Dai, Ziwei Liu, Dahua Lin, and Chen Change Loy. Self-supervised scene deocclusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3784–3792,

2020. 2

- [56] Lvmin Zhang and Maneesh Agrawala. Transparent image layer diffusion using latent transparency. ACM Transactions on Graphics (TOG), 43(4):1–15, 2024. 2, 4, 6, 8, 1
- [57] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 5
- [58] Ruicheng Zhang, Jun Zhou, Zunnan Xu, Zihao Liu, Jiehui Huang, Mingyang Zhang, Yu Sun, and Xiu Li. Zero-shot 3daware trajectory-guided image-to-video generation via testtime training. arXiv preprint arXiv:2509.06723, 2025. 2
- [59] Chuanxia Zheng, Duy-Son Dao, Guoxian Song, Tat-Jen Cham, and Jianfei Cai. Visiting the invisible: Layer-bylayer completed scene decomposition. International Journal of Computer Vision, 129(12):3195–3215, 2021. 2
- [60] Jun Zhou, Jiahao Li, Zunnan Xu, Hanhui Li, Yiji Cheng, Fa-Ting Hong, Qin Lin, Qinglin Lu, and Xiaodan Liang. Fireedit: Fine-grained instruction-based image editing via region-aware vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13093–13103, 2025. 2

## Controllable Layer Decomposition for Reversible Multi-Layer Image Generation Supplementary Material

### 6. Further Explanation of the Baseline

state-of-the-art image evaluation model, for automated assessment. We evaluate the generated images from multiple perspectives, complementing traditional metrics by assessing semantic alignment and editability. Specifically, we structure the assessment into three dimensions:

Currently, no existing work fully matches our task setting. Recent approaches [4, 42] are mostly based on image matting or vision-language models (VLMs), but these methods cannot handle user-specified bounding box constraints. Other related works [14, 35, 56] that use DiT focus on textto-image generation rather than layer decomposition, making them unsuitable as direct baselines. Figure 9 illustrates the differences between our task setup and those of related works.

- • Semantic Consistency: Measures the alignment between the generated image and the textual description or target semantics;
- • Visual Fidelity: Assesses visual quality, detail preservation, and overall realism;
- • Editability: Evaluates the manipulability of the image for subsequent editing, modifications, and local adjustments.

Among existing research, the most relevant method is the recently proposed LayerD [42]. This work investigates three architectural variants: the matting-base model, the YOLO-base model, and the VLM-base model. Among them, the matting-base design achieves the strongest performance. In addition, LayerD systematically examines multiple combinations of matting models and inpainting models and selects the best-performing configuration. As a result, LayerD can be regarded as the current state of the art among traditional approaches based on matting/segmentation/VLM.

The evaluation prompts used are as follows.

###### Q-Insight Prompt

You are evaluating the quality of a layered image decomposition. The input consists of multiple images: [Original Image], [Layer 1], [Layer 2], ..., [Reconstructed Image]. # — THINKING STEP First, reason about the decomposition quality step-by-step inside the <think> tags. Analyze the semantic completeness of each layer, the visual fidelity of the reconstruction, and the practical editability of the layer structure based on the criteria below. # — RATING STEP Second, based on your reasoning, provide three separate ratings. All ratings should be floats between 1.00 and 5.00, rounded to two decimal places. # — CRITERIA —

For these reasons, we adopt LayerD as our primary baseline. Experimental results demonstrate that our method significantly outperforms LayerD in both quantitative metrics and visual quality. This highlights the advantages of our diffusion-based layered generation framework, which provides stronger representational capacity and better disentanglement under complex scenarios compared to traditional matting/segmentation/VLM-based solutions.

- 1. **semantic consistency:** (1.00 = fragmented, meaningless layers; 5.00 = all layers are semantically whole and independent). Each layer should represent a complete, logical object or part (e.g., a whole text block, a distinct background element, a person).

- 2. **visual fidelity:** (1.00 = major artifacts, poor reconstruction; 5.00 = layers reconstruct the original perfectly).

**Note: You must ignore the inherent artistic style of the image (e.g., photo vs. cartoon).** Your score should *only* reflect artifacts from the decomposition process, such as halos, incorrect background inpainting, color bleeding, or missing pixels.

- 3. **editability:** (1.00 = useless, under- or over-decomposed; 5.00 = perfect granularity for editing).

RGBA RGB RGBA RGB

[Figure 396]

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

**Preference: A finer-grained decomposition (more layers) is preferred and should receive a higher score,** as long as the individual layers still represent complete semantic parts. Do not penalize for ‘over-decomposition’ if the resulting finegrained layers are logical and useful for an editor (e.g., separating a title from a body text is better than keeping them as one layer). # — FORMATTING Return the result in JSON format with the following keys: { “semantic consistency”: <score>, “visual fidelity”: <score>, “editability”: <score> } )

Figure 8. Visualization of RGBA–RGB Conversion.

### 7. Q-Insight Additional Details

To provide a more objective and comprehensive comparison of our method with LayerD [42] in terms of generation quality, we further employ Q-Insight [24], the current

### 8. Additional Dataset Specifications

The dataset used for training and in the proposed benchmark is based on PrismLayersPro, constructed by Chen et al[3]. It contains 21 visual style categories, including 3D, Pokemon, anime, cartoon, doodle art, furry, ink, kid crayon drawing, line draw, melting gold, melting silver, metal textured, neon graffiti, papercut art, pixel art, pop art, sand painting, steampunk, toy, watercolor painting, and wood carving. The dataset comprises 20K samples, each providing complete layer information along with corresponding textual annotations. For each data sample, the dataset contains:

- 1. A global textual description: Tglobal;
- 2. A composite image: Icomp;
- 3. n RGBA layers: {I(0),I(1),...,I(n−1)}, (12)

where I(0) is the background layer, and I(1) ∼ I(n−1) are foreground layers;

- 4. Text descriptions for each layer: {T(0),T(1),...,T(n−1)}; (13)
- 5. Bounding boxes for foreground layers: For the k-th foreground layer (k ≥ 1), the bounding box is

b(k) = (x(lk), yl(k), x(rk), yr(k)), (14)

while the background layer does not provide a bounding box, defaulting to cover the entire image:

b(0) = (0,0,H,W). (15)

PrismLayersPro provides sufficient data, complete layer annotations, and textual descriptions, allowing direct use for training and evaluation. We split each style category independently for our experiments: 0–90% for training, 90–95% for testing, and 95–100% for validation, ensuring balanced and consistent partitions.

### 9. RGBA-to-RGB Conversion Method

We describe our RGBA-to-RGB conversion process in Sec. 4.2. A fixed solid background is used, and the input RGBA image is decomposed into RGB channels Irgb and the alpha channel Iα, with the final RGB image obtained via linear color compositing (Eq. 10). In this paper, we adopt neutral gray (pixel values (0.5, 0.5, 0.5) in the normalized range [0, 1]) as the default background color.

Since RGBA layers contain no “real background,” transparent regions must be filled with a placeholder color solely for computing standard RGB metrics (PSNR, SSIM, FID). The placeholder should satisfy two criteria: (1) it should not introduce additional color bias, and (2) it should minimize the evaluation error induced by background filling in

a global statistical sense. Neutral gray, as the midpoint of the color space, naturally satisfies both criteria and provides a balanced fill, reducing systematic color shifts and worstcase error, leading to stable and fair evaluations.

From a visual perspective, neutral gray avoids artificial color or brightness shifts in transparent areas, ensuring comparability across methods. Moreover, in opaque regions (where α is close to 1), the influence of the background becomes negligible, and thus this choice does not interfere with foreground content. Figure 8 illustrates the visual results after conversion.

In summary, using neutral gray (0.5, 0.5, 0.5) as a placeholder is statistically robust, visually neutral, and practically convenient, reducing metric bias and enabling consistent, reliable RGB-based comparisons.

### 10. Additional Results

In Figure 10 and 11, we present additional examples of layer decomposition results. All samples shown in this figure are taken from the test split of our curated PrismLayersPro dataset. These examples further illustrate the effectiveness and consistency of our decomposition pipeline across diverse scenes and appearance variations.

### 11. In-The-Wild Case

In Figure 12 and 13, we present the layer decomposition results of our method on in-the-wild examples. These samples are drawn from the test split of the Crello [51] dataset, which contains diverse, professionally designed graphic layouts with rich stylistic variations. As shown in the figure, our model generalizes well beyond the controlled training distribution and is able to separate complex visual elements, including icons, text segments, and composite shapes. This demonstrates the robustness of our framework and its ability to handle real-world design assets encountered in practical editing scenarios.

### 12. Failure Case

Figure 14 shows representative failure cases of our method, which fall into two main categories. The first occurs with very small, thin, or intricate content, such as fine text or tiny details, where limited spatial resolution prevents the model from capturing sufficient local cues. We believe that increasing input resolution or using multi-scale features could mitigate this issue. The second involves complex hierarchical occlusions. When objects are heavily occluded, the image provides insufficient information for accurate reconstruction. Although textual descriptions are used as auxiliary guidance, they are often insufficient to recover hidden content. Addressing this may require more advanced use of text cues or explicit structural priors.

Input

###### ART

Background Foreground 0 Foreground 1

Bounding Box

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

Foreground 2 Foreground 3 Foreground 4

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

Text

This is a furry style image. The image presents a vibrant and playful composition, characterized by a bright yellow background ...

[Figure 420]

Input

###### LayerD

Background Foreground 0 Foreground 1

Origin Image

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

Input

CLD (ours)

Background Foreground 0 Foreground 1

Bounding Box Origin Image

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

Foreground 2 Foreground 3 Foreground 4

[Figure 432]

[Figure 433]

[Figure 434]

Text

This is a furry style image. The image presents a vibrant and playful composition, characterized by a bright yellow background ...

[Figure 435]

Figure 9. Comparison with related work. The figure illustrates the distinctions between our task and prior methods: ART [35] generates multilayer images without target-image constraints, and LayerD [42] decomposes layers without fine control. Our approach, by incorporating user-provided bounding boxes, achieves accurate and controllable layer separation.

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

###### Figure 10. More qualitative results.

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

###### Figure 11. More qualitative results.

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

###### Figure 12. In-The-Wild Case.

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

Origin Image (Input)

Bounding Box (Input) Background Foreground 0 Foreground 1 Foreground 2

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

Foreground 3 Foreground 4 Foreground 5 Foreground 6 Foreground 7 Foreground 8

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

###### Figure 13. In-The-Wild Case.

Origin Image Ground Truth Predicted Origin Image Ground Truth Predicted

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

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

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

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

Figure 14. Failure case.

