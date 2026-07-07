# arXiv:2412.03069v2[cs.CV]7Aug2025

## TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation

Liao Qu*, Huichao Zhang*, Yiheng Liu, Xu Wang†, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, Xinglong Wu ByteDance https://github.com/ByteVisionLab/TokenFlow

#### Abstract

We present TokenFlow, a novel unified image tokenizer that bridges the long-standing gap between multimodal understanding and generation. Prior research attempt to employ a single reconstruction-targeted Vector Quantization (VQ) encoder for unifying these two tasks. We observe that understanding and generation require fundamentally different granularities of visual information. This leads to a critical trade-off, particularly compromising performance in multimodal understanding tasks. TokenFlow addresses this challenge through an innovative dual-codebook architecture that decouples semantic and pixel-level feature learning while maintaining their alignment via a shared mapping mechanism. This design enables direct access to both high-level semantic representations crucial for understanding tasks and fine-grained visual features essential for generation through shared indices. Our extensive experiments demonstrate TokenFlow’s superiority across multiple dimensions. Leveraging TokenFlow, we demonstrate for the first time that discrete visual input can surpass LLaVA-1.5 13B in understanding performance, achieving a 7.2% average improvement. For image reconstruction, we achieve a strong FID score of 0.63 at 384×384 resolution. Moreover, TokenFlow establishes state-of-the-art performance in autoregressive image generation with a GenEval score of

- 0.55 at 256×256 resolution, achieving comparable results to SDXL.
- 1. Introduction

Large Language Models (LLMs) have revolutionized natural language processing through their unified autoregressive framework, demonstrating remarkable capabilities across diverse tasks [1, 2]. However, in the multimodal domain of vision and language, a fundamental divide persists

*Equal contribution †project lead

SEEDBench

MME-Perception

MMVet

70.0

1525.0

45.0

65.0

1450.0

40.0

60.0

1375.0

35.0

MMBench

VQAv2

75.0

75.0

68.0

60.0

61.0

45.0

32.5

83.0

20.0

53.0

37.0

85.0

41.5

87.0

40.0

57.0

MMMU

POPE

60.0

61.0

LLaVA-1.5 (13B)

Janus (1.3B)

VILA-U (7B)

EMU3 (8B)

TextVQA GQA

TokenFlow-XL (Ours, 14B)

Figure 1. Multimodal Understanding Results with TokenFlow. We demonstrate for the first time that discrete visual input can surpass LLaVA-1.5 13B in understanding performance, achieving a 7.2% average improvement.

between perception and generation paradigms. Current approaches address them through distinct architectures: multimodal understanding models leverage vision encoders and projection layers to align visual representations with pretrained LLMs [29, 52], while visual generation relies on either diffusion-based methods [39, 41] or discrete image tokens for autoregressive generation [38, 44, 51, 65]. This divergence motivates the pursuit of unified approaches capable of both understanding and generation.

The advent of GPT-4o [59] has greatly boosted interest in developing more generalist multimodal models. Early efforts to unify perception and generation capabilities [27, 46] have primarily focused on equipping LLMs with the power of diffusion models. However, these ap-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

A bird flying through the air while flapping it's wings.

A hotel room with a large bed, lamp, and window view.

Ocean waves under a vibrant sunset sky with clouds and birds.

A portrait of a woman.

Intricate origami of a fox and a unicorn in a snowyforest.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

A photo of a hamburger.

Woman in a flowing dress with long hair, surrounded by orange autumn trees and sunset.

Person standing in a mystical forest under a bright moon, with colorful trees and reflections.

A dog is lying on the carpet of the living room.

A graffiti wall with the words 'TOKEN FLOW' written on it.

- Figure 2. Visual Generation Results with TokenFlow. We present diverse 256×256 results across various styles, subjects, and scenarios.

proaches introduce substantial architectural complexity and computational overhead, highlighting the need for a more elegant unified solution. Recent efforts have explored one promising direction: using a single transformer architecture to unify visual and textual information within the next-token prediction framework [48, 55]. This approach relies on VQ encoders to convert visual inputs into discrete tokens that can be processed alongside text, offering a potentially simpler and more efficient framework. By treating both modalities as sequences of discrete tokens, this framework enables end-to-end training within a single architecture.

However, a fundamental challenge exists in such unified approaches. Multimodal understanding demands rich semantic representations to support complex reasoning, while visual generation, on the other hand, requires precise encoding of spatial structure and textural details. Current methods predominantly employ reconstruction-targeted VQ encoders [13, 73], which are primarily optimized for reconstruction fidelity. While this optimization makes them wellsuited for generation tasks, it potentially limits their ability to capture the high-level semantic features crucial for understanding tasks. While Janus [57] attempts to address this conflict by employing separate encoders for understanding and generation tasks, this increases model complexity without fundamentally resolving the underlying representation disparity. These limitations underscore a critical gap in the field: the absence of a unified visual encoding mechanism that can effectively serve both perception and generation objectives. This motivates our central research question: Can one single image tokenizer derive representations suitable for both multimodal understanding and generation?

To address this challenge, we propose TokenFlow, a novel unified image tokenizer that bridges the gap between understanding and generation through a unique dual-flow design. The key insight is to decouple the learning of semantic and pixel-level features while maintaining their alignment through a shared index mapping. By mapping patches with both semantic and pixel-level similarities to identical indices, the quantized features can be directly applied to both autoregressive visual generation and multimodal understanding. Unlike concurrent approach that constrains different feature levels within a single codebook [60], TokenFlow’s dual-codebook design enables specialized learning while maintaining cross-level correlations through shared indices. This innovation allows simultaneous access to both semantic and pixel-level representations without compromising either aspect. Specifically, TokenFlow adopts a dual-encoder architecture coupled with corresponding specialized codebooks. The semantic encoder, learned from a CLIP-style teacher, provides strong semantic priors, while the pixel encoder captures detailed visual information. The extracted features are then quantized by minimizing the weighted summation of semantic and pixellevel distances, creating a joint representation space.

Our framework exhibits remarkable scalability, maintaining exceptional codebook utilization (95%+) even with large-scale codebooks of over 130K entries - substantially advancing beyond prior approaches [13] in both capacity and efficiency. TokenFlow also achieves a strong FID score of 0.63 at 384×384 resolution. For text-to-image synthesis, we establish a new state-of-the-art GenEval score of 0.55 at 256×256 resolution in the autoregressive paradigm

while requiring significantly fewer sampling steps compared to existing methods like EMU3 [55] and LlamaGen [44]. On multimodal understanding benchmarks, TokenFlow achieves new state-of-the-art performance with minimal training overhead, surpassing LLaVA-1.5 13B by 7.2% on average - for the first time discrete visual inputs can outperform this strong baseline. These results validate TokenFlow’s effectiveness as a unified visual tokenizer that bridges the long-standing gap between understanding and generation tasks.

#### 2. Related Work

##### 2.1. Tokenization for Visual Generation.

Vector quantized (VQ) image tokenizers have played a crucial role in recent advancements in autoregressive image generation [28, 34, 44, 51, 65]. [54] proposed the VQVAE, quantizing patch-level features using the nearest codebook entry, with the codebook learned with the encoder-decoder structure through reconstruction loss. VQVAE-2 [40] advanced this framework through exponential moving average updates and a hierarchical multi-scale approach. VQGAN [13] further enhanced the architecture by incorporating adversarial and perceptual losses, yielding more precise and detailed representations. Recent advances in VQ tokenizers have focused on three main directions: improving reconstruction fidelity and generation quality [21, 64, 73], enhancing codebook utilization [64, 70, 76], and exploring novel architectures such as the multi-scale VQVAE [25, 51] for next-scale prediction of images. While these methods effectively preserve local details after quantization, they often struggle to capture semantic-level information, limiting their effectiveness in autoregressive multi-modal image understanding tasks. Our proposed TokenFlow addresses this limitation by introducing dual codebooks with shared mapping, achieving state-of-the-art performance in both autoregressive generation and multimodal understanding.

##### 2.2. Tokenization for Unified Multimodal Understanding and Generation

Recent efforts have emerged to bridge the gap between multimodal understanding and generation [23, 48, 55, 57, 60, 62]. Approaches like Chameleon [48], EMU3 [55] and Show-o [62] employ VQ tokenizers [13, 66, 73] to encode images for both tasks. However, these methods typically require multimodal training from scratch and often suffer performance degradation in visual perception tasks due to limited semantic representation in their tokenized features. SEED-LLaMA [23] introduced a novel VQ tokenizer incorporating high-level semantics for understanding and utilize SD [41] as generation decoder. Janus [57] attempted to address the modality gap by employing separate tokenizers for understanding [69] and generation [44], though this

leads to increased model complexity without fundamentally resolving the underlying challenge. Concurrent work [60] proposed a unified vision tower aligning discrete visual features with text during pre-training. However, their approach constrains low-level and high-level representations within a single flow, limiting the upper bound of downstream performance. In contrast, our work posits that the key to unifying understanding and generation lies in learning a universal mapping. By defining dual codebooks with shared mapping, TokenFlow enables flexible combinations of low and high-level features, resulting in superior performance across all downstream tasks.

#### 3. Method

##### 3.1. Motivation

Table 1. Comparison of various visual encoders on multimodal understanding [14, 23, 43] within the LLaVA-1.5 framework. VQKD is distilled from CLIP ViT-B/14. ”Sem.” refers to semantic encoders that learn semantic-level representations, while ”Pix.” indicates pixel-level tokenizers that focus on low-level visual features.

# Exp. Visual Encoder Type MME-P ↑ SEEDB ↑ TQA ↑ Continuous:

- 1 CLIP ViT-B/14 [37] Sem. 1460.9 64.1 53.4

Discrete:

- 2 VQGAN [13] Pix. 756.1 38.2 46.8
- 3 VQGAN-LC [76] Pix. 744.8 38.2 45.7
- 4 LFQ [66] Pix. 889.5 41.1 46.4
- 5 VQKD [35] Sem. 1252.4 57.8 48.2

Unifying multimodal understanding and generation into a cohesive next-token prediction paradigm requires a VQ tokenizer for extracting indices from input images. While traditional VQ tokenizers [13, 54, 66, 76] excel at pixellevel image reconstruction, our investigation reveals a significant limitation in their image understanding capabilities. We conducted experiments utilizing these tokenizers as feature extractors within the LLaVA-1.5 [29] framework. As shown in Exp. 2-4 of Tab. 1, the performance of these discrete tokenizers consistently lags behind that of the continuous tokenizer CLIP ViT-B/14 [37]. We posit that this performance gap stems from their pre-training objectives, which primarily optimize towards better low-level reconstruction quality. Consequently, the extracted features mainly encode low-level information, lacking the semantic-level understanding, which is crucial for complex visual reasoning.

Another straight forward solution for unified understanding and generation can be distill discrete tokens from pretrained CLIP [8, 37, 45, 69], and then equip it with image reconstruction capability. As demonstrated in Exp. 5, VQKD, distilled from CLIP ViT-B/14, substantially reduces the performance gap compared to other discrete tokenizers. We

Semantic Codebook Embeddings

Norm Data Flow Supervision

[Figure 11]

[Figure 12]

| ||1|
|---|
<br><br>1<br><br>|2|
|---|
<br><br>2<br><br>|3|
|---|
<br><br>3<br><br>|𝑘|
|---|
<br><br>𝑘<br><br>|𝑘−2|
|---|
<br><br>𝑘−2<br><br>|𝑘−1|
|---|
<br><br>𝑘−1|
|---|---|
| | |

ℒ

Semantic Encoder

Semantic Decoder

𝑘−1

𝑘−1 𝑘−2

𝒅𝒔𝒆𝒎

ℕ

ℕ

1

2

3

𝑘

|1|
|---|

|1|
|---|

|1|
|---|

[Figure 13]

|𝑘 − 1|
|---|

|𝑘 − 1|
|---|

|𝑘 − 1|
|---|

[Figure 14]

Shared Mapping

quantization

query

|2|
|---|

|2|
|---|

|2|
|---|

[Figure 15]

|𝑘|
|---|

|𝑘|
|---|

|𝑘|
|---|

| ||1|
|---|
<br><br>1<br><br>|2|
|---|
<br><br>2<br><br>|3|
|---|
<br><br>3<br><br>|𝑘−2|
|---|
<br><br>𝑘−2<br><br>|𝑘|
|---|
<br><br>𝑘 𝑘−1<br><br>|𝑘−1|
|---|
|
|---|---|
| | |

ℒ

Pixel Encoder

Pixel Decoder

𝑘−2

index

ℕ 𝒅𝒑𝒊𝒙

ℕ

1

2

3

𝑘

###### Downstream tasks

Pixel Codebook Embeddings

- Figure 3. Overview of TokenFlow. We incorporate dual encoders and codebooks with a shared mapping, enabling the joint optimization of high-level semantics and low-level pixel details. For a given input image, distances dsem and dpix are calculated from the pixel-level and semantic-level codebooks, respectively, with the final codebook index and features determined by minimizing the weighted sum

dsem + wdis · dpix. The resulting quantized features are independently decoded for both semantic alignment and image reconstruction training, and then concatenated to provide a unified representation for downstream tasks in understanding and generation.

- (a)
- (b)

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

- (c)

- Figure 4. Visualization of images clustered by (a) VQKD [35], (b) VQGAN [13], and (c) Our TokenFlow. VQKD clusters exhibit semantic similarity, while VQGAN clusters exhibit low-level similarity (i.e. color). Our TokenFlow can successfully combine both semantic and low-level similarity. Implementation details of image clustering can be found in Appendix A.1.

gregated by VQKD becomes extremely challenging.

These observations highlight the necessity of developing a novel tokenization approach that can effectively handle high-level semantic understanding and low-level visual reconstruction tasks.

##### 3.2. Unified Image Tokenizer

To bridge this gap, we propose TokenFlow (Fig. 3), a novel unified image tokenizer that enables joint representation learning at both semantic and pixel level. We find the key to unifying understanding and generation lies in learning an universal mapping. If the tokenizer can map patches that are both high-level and low-level similar to the same codebook index, then the quantized features can be easily decoded and directly applied to both autoregressive visual generation tasks and multimodal understanding tasks.

Encoder. Unlike previous approaches that utilize one single encoder to extract low-level image information, we propose a dual-encoder architecture comprising a semantic encoder Esem and a pixel encoder Epix. This design enables the extraction of two distinct types of image features. For the semantic encoder, we initialize it with a pre-trained textaligned vision encoder (e.g., CLIP ViT-B/14). This initialization strategy facilitates better learning of high-level textaligned embeddings in the semantic codebook, ultimately enhancing the model’s multimodal understanding capabilities. For brevity here, we omit the spatial indices of feature representations, where zˆsem = Esem(x) ∈ Rd

further conducted an experiment to reconstruct the original image from quantized features extracted by VQKD. The reconstructed images exhibited significant blurring and a evident loss of high-frequency details, as shown in Fig. 8. We attribute this outcome to the nature of VQKD’s encoder, which maps semantically close patches into same codebook index. As visualized in Fig. 4 (a), it tends to map images with same semantical meaning to the same codebook index, while VQGAN (Fig. 4 (b)) tends to map visually similar images to the same codebook index, prioritizing low-level features over semantic content. Therefore, the reconstruction of fine-grained details from low-level dissimilar patches ag-

sem and zˆpix = Epix(x) ∈ Rd

pix are the encoded features from semantic and pixel encoder.

Quantization. We introduce an innovative quantization approach that employs dual codebooks: semantic-level embeddings Zsem = {zsem,i}Ki=1 ∈ RK×d

sem and pixel-level embeddings Zpix = {zpix,i}Ki=1 ∈ RK×d

pix, where K is the

number of codebook entries. These two codebooks share a unified mapping, enabling simultaneous consideration of high-level semantic information and low-level pixel details during the quantization process. Given the encoded feature representations zˆsem and zˆpix, we compute the distances to their respective codebook embeddings after l2-norm [64]:

dsem,i = ∥zˆsem − zsem,i∥22,for i = 1,...,K (1)

dpix,i = ∥zˆpix − zpix,i∥22,for i = 1,...,K (2)

i∗ = arg min

(dsem,i + wdis · dpix,i) (3)

i

The optimal quantization index i∗ is determined by minimizing the weighted sum of these two distances, where wdis is the distance balance weight, as shown in Eq. (3). This joint optimization approach differs significantly from previous VQ methods that typically focus on learning the distribution of a single feature type. We further adopt the multiscale VQ (MSVQ) structure [51] to to enhance the richness of the codebook representation. Our shared mapping strategy enables the codebook to learn the joint distribution of high-level semantics and low-level features, resulting in several key advantages:

❶ Scalability: Our approach demonstrates consistent performance improvements in both generative and understanding tasks as the codebook size increases, since large codebook size offers more high- and low-level feature combination possibilities. With an expanded codebook size of 131,072, it can still maintain a remarkably high utilization rate of over 95% while achieving best image reconstruction quality and multimodal understanding performance.

❷ Multi-task Capabilities: By learning the joint distribution of semantic and pixel-level features, our method bridges the gap between generation and understanding tasks. This unified representation enables a single tokenizer to excel in both domains. This design also allows seamless integration of more codebooks to embed other type of feature representations, enabling extensibility to more downstream tasks without architectural modifications.

Decoder and Training Objective. Our architecture incorporates two distinct decoders, including semantic decoder Dsem and pixel decoder Dpix for reconstructing semantic features and original image. We employ a teacher model [35] (identical to the semantic encoder’s initialization) for target feature extraction. The semantic loss Lsem is computed as the l2 distance between decoded and teacherextracted features. The reconstruction loss is formulated as:

Lpix = ℓ2(x,xˆ) + LP(x,xˆ) + λGLG(ˆx) (4)

where xˆ = Dpix(z), ℓ2 represents pixel-wise reconstruction loss, LP(·) denotes perceptual loss using LPIPS, and LG(·)

represents adversarial loss with λG as its weight coefficient. Following vector quantization conventions, we employ a straight-through gradient estimator: z = sg[z−zˆ]+ˆz where sg[·] denotes the stop-gradient operation. The codebook learning objective is: LVQ = ||sg[ˆz]−z||22 +β||zˆ−sg[z]||22 where the second term represents commitment loss with balancing factor β. The total training objective is the sum of all losses: Ltotal = Lsem + LVQ + Lpix.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

- (a)
- (b)

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Figure 5. Qualitative comparison of different sampling strategies in our framework. (a) Single-pass top-k (k=1200) and top-p (p=0.8) sampling exhibits inconsistent patterns and artifacts. (b) Our proposed multi-step sampling strategy produces more coherent and visually appealing results. Best zoomed in for details.

##### 3.3. Visual Generation with TokenFlow

TokenFlow helps us achieve SOTA performance in autoregressive text-to-image generation using the next-scale prediction paradigm. Below, we detail our training and inference strategy for high-quality image synthesis.

Training Strategy. Our visual generation architecture builds upon a pre-trained LLM model [53]. For text encoding, we leverage the model’s native BPE tokenizer to transform input text into discrete token sequences and extract feature representations. The original vocabulary is extended with specialized visual tokens. We extract the image tokens using TokenFlow, pass it through a MLP, and concatenate it with text tokens for training. Given the model’s autoregressive nature, we employ cross-entropy loss computed exclusively on image tokens. To enable classifier-free guidance [17] during inference, we randomly replace conditioned text with an empty string with probability pdrop = 0.1 during training. Following [11, 48, 56], we incorporate QKnormalization and norm re-ordering to enhance training stability and prevent loss spikes.

Inference Strategy. We observed that conventional topk-top-p sampling strategies, when employed in the nextscale paradigm, often lead to image collapse and repetitive local patterns. This can be attributed to the cross-entropy training objective, which establishes attention-based relationships primarily with the top-1 prediction. Independent top-k sampling for each token during inference can result in tokens lacking direct correlations, leading to inconsistent or repetitive patterns that can only be partially remedied

through subsequent scales’ attention. This issue becomes more severe particularly with limited inference steps.

To address this fundamental limitation, we propose a novel multi-step sampling approach: (i) Initial sampling: Perform top-k top-p sampling with parameters k1 and p1. (ii) Refinement: Use the sampled output as input for a second round of sampling in the same scale with reduced parameters k2 < k1 and p2 < p1. This progressive narrowing of the sampling space maintains creative diversity while enforcing consistency through refinement steps. Empirical results demonstrate significantly more coherent and visually appealing generations compared to single-pass sampling methods (see Fig. 5 and detailed ablation in Appendix B.1).

##### 3.4. Multimodal Understanding with TokenFlow

TokenFlow functions as a multi-scale VQ tokenizer, where the quantized multi-scale features can be directly fed into a pre-trained LLM for multimodal understanding training, following the LLaVA-1.5 [29] paradigm. The joint feature representations from dual flow serve as input to the model. We validate multiple feature input strategies: (i) Feature from all scales (ii) Final-scale feature only (iii) Residual features from all scales. We discover that features from the final scale achieves best overall performance, as detailed in Appendix B.1. This suggests that the final scale captures the most relevant semantic information for multimodal understanding, while additional scale features or residual features may introduce noise that compromises performance. Our model demonstrates substantial improvements over existing discrete multimodal methods. Notably, the performance gains can be achieved with minimal computational overhead, requiring less than 24 hour training on 8×A100 GPUs using LLaVA 1.5 training data.

#### 4. Experiments 4.1. Experimental Setup

Datasets. TokenFlow is trained on LAION [42] and COYO-700M [5] and evaluate it on ImageNet [12]. To enhance face generation quality, we follow [48] and upsample the percentage of images with faces during tokenizer training by 2 times. For ablation studies, we train the tokenizer for 50 epochs on ImageNet-1K with CLIP ViT-B/14-224 [37]. For visual generation with TokenFlow, we trained it on a curated dataset of 60M high-quality images, with captions generated using Qwen-VL [3].

Implement Details. We employ three variants of TokenFlow (B/L/XL), using CLIP ViT-B/14-224 [37], ViTaminXL-256 [8], and SigLIP-SO400M-patch14-384 [69] as respective teacher models and semantic encoder initializations. Detailed configurations are provided in Appendix A.2. For multimodal understanding, we employ Vicuna-v1.5-13B [10] and Qwen-2.5-14B [50] as the lan-

guage backbone. For 256×256 visual generation training, we truncate captions to first sentence with 0.2 probability to enhance short prompt generation capabilities. The model is initialized with Llama-2-7b [53], and being trained for 2 epochs. At inference, we apply classifier-free guidance [17] with a scale factor of 7.5.

Evaluation Metrics. We assess reconstruction quality using rFID, PSNR, and SSIM on the ImageNet-1K validation set [12]. For multimodal understanding, we evaluate on a comprehensive suite of vision-language benchmarks: SEEDBench [22], MMVet [67], POPE [26], VQAv2 [16], GQA [19], TextVQA [43], AI2D [20], RealWorldQA [61], MMMU [68], MMBench [32], and MME [14]. Visual generation capabilities are evaluated using GenEval [15] and DPG-Bench [18]. We opt not to include FID scores as argued that it does not correlate well with human assessment of the overall performance of generative models [7, 36, 46].

##### 4.2. Unified Image Tokenizer

Table 2. Comparison of reconstruction quality on the ImageNet 50k validation set. “#Lvls.” represents the number of residual levels used. For 384×384 resolution, the downsample ratio of 14.2 is derived from 384/27.

Model Res. ratio #Lvls. rFID ↓ PSNR ↑ SSIM ↑ VQ-GAN [13] 256 16 1 4.98 20.00 0.629 LlamaGen [44] 256 16 1 2.19 20.79 0.675 RQ-VAE [21] 256 32 4 3.20 – – RQ-VAE [21] 256 16 4 1.30 – – VAR [51] 256 16 10 1.00 22.63 0.755 VILA-U [60] 256 16 4 1.80 – – Ours 256 16 9 1.37 21.41 0.687 LlamaGen [60] 384 14.2 1 0.94 21.94 0.726 VILA-U [60] 384 14.2 16 1.25 – – VAR [51] 384 16 13 2.09 22.73 0.774 Ours 384 14.2 15 0.63 22.77 0.731

In Tab. 2, we present reconstruction metrics of TokenFlow on 256×256 and 384×384 resolutions. The metric of VAR [51] is tested with the released checkpoint. At 256×256 resolution with a 16× compression ratio, TokenFlow achieves competitive performance with an rFID of 1.37, comparable to RQ-VAE while significantly outperforming previous methods such as VQ-GAN and LlamaGen. TokenFlow demonstrates superior reconstruction quality across all metrics in 384×384 resolution—a standard size in multimodal understanding tasks. These results validate the effectiveness of dual codebook design in preserving fine-grained visual details. Moreover, the incorporation of shared mapping enables TokenFlow to maintain high-level semantic features, as verified in Sec. 4.3.

##### 4.3. Multimodal Understanding

TokenFlow, as a discrete visual encoder, demonstrates state-of-the-art performance across a comprehensive suite

- Table 3. Evaluation on multimodal understanding benchmarks. We collect evaluations including: SEEDB: SEED Bench-Img [22]; MMV: MM-Vet [67]; POPE [26]; VQAv2 [16]; GQA [19]; TQA: TextVQA [43]; AI2D [20]; RWQA: RealWorldQA [61]; MMMU [68]; MMB: MMBench [32]; MME [14] and MME-P: MME-Perception. We include approaches with continuous visual inputs (top) versus discrete visual inputs (bottom). The best results among approaches with discrete visual input are highlighted in bold. * results are not reported in original paper and tested with lmms-eval [71] using the released checkpoint. When calculating average, we use MME-P and divide it by 20 to have the same scale with other benchmarks.

Method # Params Res. SEEDB MMV POPE VQAv2 GQA TQA AI2D RWQA MMMU MMB MME MME-P Avg. Continuous Visual Input

InstructBLIP [30] Vicuna-13B 224 58.8 25.6 78.9 – 49.5 50.7 – – – 36.0 – 1212.8 – MiniGPT-4 [75] Vicuna-13B 224 – – – – – – – – – – 1158.7 866.6 – BLIP-2 [24] Vicuna-13B 224 46.4 22.4 – – – 42.5 – – 26.6 – – 1293.8 – ShareGPT4V [9] Vicuna-7B 336 69.7 37.6 – 80.6 63.3 60.4 58.0 54.9 37.2 68.8 1943.8 1567.4 – NExT-GPT [58] Vicuna-7B 224 57.5 – – 66.0 – – – – – 58.0 – – – Qwen-VL-Chat [3] Qwen-7B 448 57.7 – – 78.2 57.5 – – – – – 1848.3 1487.5 – Janus [57] DeepSeek-LLM-1.3B 384 63.7 34.3 87.0 77.3 59.1 – – – 30.5 69.4 – 1338.0 – LLaVA-1.5 [29] Vicuna-13B 336 68.1 36.1 85.9 80.0 63.3 61.3 61.1 55.3 36.4 67.7 1826.7 1531.3 62.9

Discrete Visual Input

Gemini-Nano-1 [49] 1.8B from scratch – – – – 62.7 – – – – 26.3 – – – – Chameleon [48] 34B from scratch 256 – – – 69.6 – – – – – – – – – LWM [31] LLaMA-2-7B 256 – 9.6 75.2 55.8 44.8 18.8 – – – – – – – SEED-LLaMA [23] LLaMA-2-13B 224 53.7 – – 63.4 – – – – – – – – – Show-o [62] Phi-1.5-1.3B 256 – – 80.0 69.4 58.0 – – – 26.7 – – 1097.2 – VILA-U [60] LLaMA-2-7B 256 56.3 27.7 83.9 75.3 58.3 48.3 – – – – – 1336.2 – VILA-U [60] LLaMA-2-7B 384 59.0 33.5 85.8 79.4 60.8 60.8 – – – – – 1401.8 – EMU3 [55] 8B from scratch 512 68.2 37.2 85.2 75.1 60.3 64.7 70.0 57.4 31.6 58.5 1509.9* 1243.8* 60.9 TokenFlow-B Vicuna-13B 224 60.4 22.4 84.0 70.2 59.3 49.8 54.2 49.4 34.2 55.3 1660.4 1353.6 55.2 TokenFlow-L Vicuna-13B 256 62.6 27.7 85.0 73.9 60.3 54.1 56.6 49.2 34.4 60.3 1622.9 1365.4 57.5 TokenFlow-XL Vicuna-13B 384 68.7 40.7 86.8 77.9 62.7 61.5 66.7 53.7 38.7 68.9 1840.9 1545.9 64.0 TokenFlow-XL Qwen-2.5-14B 384 72.6 48.2 87.8 77.6 62.5 62.3 75.8 56.6 43.2 76.8 1922.2 1551.1 67.4

of multimodal understanding benchmarks. Following LLaVA-1.5’s training pipeline, we train TokenFlow-B and TokenFlow-L using LLaVA-Pretrain558K for adapter pretraining and LLaVA-v1.5-mix-665K for instruction tuning. For TokenFlow-XL, inspired by recent findings in [52], we leverage Cambrian-Alignment and Cambrian-10M for pretraining and instruction tuning respectively, as the teacher model SigLIP-SO400M benefits significantly from increased training data. As evidenced in Tab. 3, TokenFlowXL achieves competitive or superior results compared to leading approaches with continuous inputs from CLIPstyle encoders. Using the same language backbone (Vicuna 13B), TokenFlow-XL outperforms LLaVA-1.5 13B by 1.7% on average, for the first time demonstrates that model with discrete visual input can surpass this strong baseline. By simply changing the LLM backbone to Qwen-2.5-14B

- [50], we further surpass LLaVA-1.5 by 7.2%.

When compared to methods using discrete inputs, our approach demonstrates superior performance while maintaining training efficiency. Unlike models trained from scratch such as Chameleon and EMU3, our method requires less than 24 hour of training on 8×A100 GPUs using LLaVA 1.5 data. TokenFlow-XL 14B significantly outperforms EMU3 with an overall improvement of 10.7%. Given these promising empirical results, we position TokenFlow as a potential next-generation vision tokenizer for unified understanding and generation tasks. Our findings suggest that discrete visual representations can not only match but

exceed the performance of continuous counterparts while maintaining practical training requirements.

##### 4.4. Visual Generation

We evaluate our model’s generation capabilities against state-of-the-art methods including diffusion-based, autoregressive-based, and hybrid approaches on standard benchmarks GenEval [15] and DPG-Bench [18]. As shown in Tab. 4, our approach achieves competitive performance while requiring significantly fewer generation steps.

For 256×256 image generation, we employ a multistep sampling strategy instead of the original 9-step sampling (one per tokenizer scale). Specifically, we apply three steps per scale with top-k=[1200,100,1] and topp=[0.8,0.8,1.0] across all scales except the first, totaling 25 steps. Under this inference scheme, our model achieves a GenEval score of 0.55, surpassing prominent diffusion models like Stable Diffusion v2.1 and PixArtalpha. More significantly, it surpasses autoregressive methods such as Chameleon, LlamaGen, and EMU3, which require thousands of inference steps. With prompt rewriting, our model achieves 0.63, approaching DALL-E 3’s performance. On DPG-Bench, it achieves an average score of 72.9, outperforming LlamaGen, Show-o, SD v1.5, and PixArt-alpha. Moreover, our model only requires 2.7 seconds to infer one image with 1×A100 GPU, which is significantly faster than other autoregressive-based methods.

We further conduct additional text-to-image compari-

- Table 4. Comparison of generation quality on GenEval [15] and DPG-Bench [18]. ”#Step”: the number of model runs needed to generate an image. † result is with rewriting.

GenEval DPG-Bench

Model Text Pretrain Res. #Steps

Overall ↑ Average ↑

Diffusion-based SD v1.5 [41] CLIP ViT-L/14 512 50 0.43 63.18

- DALL-E 2 [39] CLIP ViT-H/16 1024 – 0.52 – SD v2.1 [41] CLIP ViT-H/14 768 50 0.50 – SDXL [36] CLIP ViT-bigG 1024 40 0.55 74.65 PixArt-alpha [7] Flan-T5-XXL 512 20 0.48 71.11

- DALL-E 3 [4] Flan-T5-XXL 1024 – 0.67† 83.50 Autoregressive meets diffusion

Show-o [62] Phi-1.5 256 16 0.53 67.27 Transfusion [74] – 256 250 0.63 –

Autoregressive-based

Chameleon [48] – 512 1024 0.39 – LlamaGen [44] Flan-T5-XL 512 1024 0.32 64.84 EMU3 [55] – 512 4096 0.54 / 0.66† 80.60 VAR [51] – 256 28 0.53 71.08 Ours – 256 25 0.55 / 0.63† 73.38

- 2.1

2.15

- 2.2

2.25

- 2.3

ReconstructionFID

Reconstruction FID

Generation FID

Codebook Usage

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

212 213 214 215 216

Codebook Size

80%

90%

100%

CodebookUsage

4.5

5.0

5.5

6.0

GenerationFID

- 55

- 56

- 57

- 58

- 59

- 60

- 61

VLMScore

SEED-Bench

MME

Text-VQA

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

212 213 214 215 216

Codebook Size

- 48
- 49
- 50

Figure 6. Impact of codebook size on reconstruction quality, class-conditional generation, and multimodal understanding benchmarks. MME is divide by 28 to have the same scale.

son between TokenFlow and the released VAR tokenizer [51]. Under identical training configurations and dataset settings, our model consistently demonstrates better performance across all benchmark metrics, this further showcasing the effectiveness of our unified tokenization approach.

- 4.5. Ablation Studies

Effect of Codebook Size. In Fig. 6, we experimented the impact of codebook size in our unified tokenizer, varying from 8,192 to 131,072. Our evaluation spans reconstruction quality, class-conditional generation, and multimodal understanding capabilities. For class-conditional generation, we employ the VAR transformer [51] with d=16, resulting in approximately 310M parameters. Notably, our approach maintains a consistently high codebook utilization rate exceeding 95% even with codebook size of 131,072, attributed to our shared mapping design. The shared mapping allows for effective combinations of high-level semantic features and low-level details, addressing a common limitation of

Table 5. Impact of key design choices on reconstruction quality and multimodal understanding benchmarks. Best results for each metric are highlighted in bold.

Shared Mapping MSVQ CLIP Init. rFID ↓ MME-P ↑ SEEDB ↑ TQA ↑

8.07 1252.38 57.84 49.16 3.96 1212.51 55.97 47.42 2.18 1209.90 56.08 47.40 2.16 1312.09 58.99 49.29

conventional VQ tokenizers [13] that typically suffer from deteriorating utilization rates at larger scales.

Our results reveal that increasing codebook size enhances performance across multimodal understanding benchmarks and reconstruction quality. However, when codebook size exceeds 32,768, we observe a slight degradation in class-conditional generation performance. This phenomenon can be attributed to the increased complexity of learning for autoregressive generation with larger codebooks. Based on this finding, we adopt a codebook size of 32,768 for our text-to-image generation experiments.

Effect of Key Design Choice. We validate the effectiveness of our key design choices in TokenFlow: shared mapping, multi-scale vector quantization (MSVQ), and CLIP initialization for the semantic encoder. As shown in Tab. 5, we start with a baseline that uses one single codebook distilled from CLIP ViT-B/14, coupled with a pixel decoder for direct image reconstruction from semantic features. This baseline yields a high reconstruction FID of 8.07, primarily due to the challenge of reconstructing fine-grained pixel details solely from semantic features, as visualized in Fig. 8. The introduction of shared mapping (Row 2) enables the two codebooks to capture high-level and low-level features simultaneously. By weighted distance computation, we quantize the input with optimal combinations of high-level and low-level features. This design significantly improves reconstruction quality (-4.11 rFID) while maintaining comparable understanding capabilities.

We further find that incorporating MSVQ [51] (Row 3) introduces multi-granular information into the codebook embeddings, which results in enhanced reconstruction performance, with rFID of 2.18. Moreover, this hierarchical design enables a next-scale prediction paradigm in downstream text-to-image generation tasks, offering significant inference speed advantages over traditional next-token prediction approaches [47, 51]. Initializing the semantic encoder with pretrained CLIP weights (Row 4) while making it unfrozen during tokenizer training provides strong semantic priors for codebook embeddings. This results in substantial improvements across all understanding metrics (+8.4% in MME-Perception, +5.2% in SEED-Bench, and +4.0% in TextVQA). Given these empirical results, we adopt this configuration as our final model architecture and extend our experiments with stronger teacher models, additional train-

ing data, and longer training iterations.

#### 5. Conclusion

In this work, we introduce TokenFlow, a novel unified image tokenizer that effectively bridges the gap between multimodal understanding and generation through its innovative dual-codebook architecture. By decoupling semantic and pixel-level feature learning while maintaining their alignment via shared mapping, TokenFlow successfully addresses the fundamental issue between different granularities of visual information required for understanding and generation tasks. Our comprehensive experiments demonstrate its effectiveness across multiple dimensions: superior reconstruction quality at different resolutions, state-of-theart performance in multimodal understanding with minimal training costs, and competitive visual generation capabilities with substantially fewer inference steps. These results validate that decoupled yet aligned feature learning through our shared mapping can effectively unify understanding and generation while maintaining superior performance in both domains, suggesting TokenFlow as a promising next-era foundation tokenizer for vision-language systems.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 6, 7
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 8, 4
- [5] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 6
- [6] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 3
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion

- transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 6, 8, 4
- [8] Jieneng Chen, Qihang Yu, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Vitamin: Designing scalable vision models in the vision-language era. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12954–12966, 2024. 3, 6, 1, 4, 7
- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 7
- [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6,

2023. 6

- [11] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pages 7480–7512. PMLR, 2023. 5
- [12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 6
- [13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2, 3, 4, 6, 8, 1, 5
- [14] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 3, 6, 7
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 6, 7, 8, 2, 4
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 6, 7
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5, 6
- [18] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 6, 7, 8, 4
- [19] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 6, 7

- [20] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 6, 7

- [21] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022. 3, 6
- [22] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 6, 7
- [23] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 3, 7
- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 7

- [25] Xiang Li, Hao Chen, Kai Qiu, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024. 3
- [26] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 6, 7
- [27] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 1

- [28] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 3
- [29] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 3, 6, 7
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 7
- [31] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024. 7
- [32] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an

- all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025. 6, 7
- [33] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 1
- [34] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-toimage generation via auto-regressive representations. arXiv preprint arXiv:2406.10797, 2024. 3
- [35] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366, 2022. 3, 4, 5, 1, 2
- [36] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 6, 8, 4
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 6, 1, 4, 7
- [38] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 1
- [39] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1, 8, 4

- [40] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019. 3
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1, 3, 8, 4
- [42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 6
- [43] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 3, 6, 7
- [44] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 3, 6, 8, 4
- [45] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 3

- [46] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023. 1, 6
- [47] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812,

2024. 8

- [48] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2, 3, 5, 6, 7, 8, 4
- [49] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 7
- [50] Qwen Team. Qwen2.5: A party of foundation models, 2024. 6, 7
- [51] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 1, 3, 5, 6, 8, 4
- [52] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 1, 7
- [53] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 5, 6, 2
- [54] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3
- [55] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2, 3, 7, 8, 4
- [56] Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for large-scale transformer training instabilities. arXiv preprint arXiv:2309.14322, 2023. 5
- [57] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 2, 3, 7
- [58] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023. 7
- [59] Yiqi Wu, Xiaodan Hu, Ziming Fu, Siling Zhou, and Jiangong Li. Gpt-4o: Visual perception performance of multimodal large language models in piglet activity understanding. arXiv preprint arXiv:2406.09781, 2024. 1

- [60] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 2, 3, 6, 7
- [61] XAI. Realworldqa, 2024. 6, 7
- [62] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3, 7, 8, 4
- [63] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36, 2024. 2
- [64] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 3, 5
- [65] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 1, 3
- [66] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 3, 1
- [67] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 6, 7
- [68] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 6, 7
- [69] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 3, 6, 1, 4, 7
- [70] Jiahui Zhang, Fangneng Zhan, Christian Theobalt, and Shijian Lu. Regularized vector quantization for tokenized image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18467– 18476, 2023. 3
- [71] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 7

- [72] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model, 2024. 2
- [73] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for highfidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022. 2, 3
- [74] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 8, 4
- [75] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 7
- [76] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. arXiv preprint arXiv:2406.11837, 2024. 3, 1

## TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation

### Supplementary Material

#### A. Implementation Details

##### A.1. Motivation

Experimental Setup for Multimodal Understanding. To evaluate the multimodal understanding capabilities of current VQ tokenizers, we conduct experiments as detailed in Tab. 1. For LFQ [66], we utilize the open-source implementation [33], which demonstrates comparable performance to the original paper. The codebook size of LFQ is 262,144. For VQGAN-LC [76], we employ features before its projection layer, which is clustered from the pretrained CLIP image encoder, with a codebook size of 100,000.

Experimental Setup for Visual Comparison of VQKD, VQGAN and TokenFlow. To generate the visualizations in Fig. 4, we perform an experiment using 50,000 images from the ImageNet-1k validation set. We process these images through the encoders of VQKD, VQGAN and TokenFlow, applying average pooling to the extracted features to obtain a 1 × 1 representation. Subsequently, we identify the closest index in their respective codebooks using l2 distance. We provide more visualizations in Fig. 11, and visualize the cluster size distribution in Fig. 7.

Experimental Setup for Image Reconstruction from Quantized Semantic Feature. We conducted an experiment to reconstruct original images from quantized features extracted by VQKD [35]. In this setup, we maintained the original encoder and quantizer of VQKD, while introducing an additional decoder aimed at reconstructing the input image. The architecture of this decoder is identical to the pixel decoder employed in our TokenFlow. We trained this decoder on the ImageNet-1K dataset for 100 epochs. Fig. 8 presents a visual comparison between the original and the reconstructed images. As observed, while the reconstructed images maintain the overall semantic content, they exhibit a noticeable loss of high-frequency details. This phenomenon suggests that the quantized semantic features cannot fully preserve fine-grained visual details, which is crucial for visual generation.

##### A.2. Tokenizer Training Details

We provide detailed training configurations for TokenFlow-B, TokenFlow-L, and TokenFlow-XL variants in Tab. 11. All models share common hyperparameters including learning rate, batch size, commitment loss factor, adversarial loss factor and distance balance weight. The models primarily differ in their input resolution (224, 256, and 384) and semantic teacher models, utilizing CLIP

###### VQKD

3000

Total non-empty clusters: 2224 (27.1%) Average images per cluster: 22.4 Maximum images in a cluster: 3077

NumberofImages

2000

1000

0

0 500 1000 1500 2000

###### VQGAN

6000

Total non-empty clusters: 207 (2.5%) Average images per cluster: 238.0 Maximum images in a cluster: 6245

NumberofImages

4000

2000

0

0 50 100 150 200

TokenFlow

Total non-empty clusters: 7161 (87.4%) Average images per cluster: 7.0 Maximum images in a cluster: 118

100

NumberofImages

75

50

25

0

0 1000 2000 3000 4000 5000 6000 7000

Cluster Index (Sorted by Size)

Figure 7. Comparison of cluster size distributions between VQKD [35], VQGAN [13], and TokenFlow (ours), with a fixed codebook size of 8,192. Analysis performed on 50,000 images from the ImageNet-1k validation set. TokenFlow exhibits significantly smoother distribution compared to others, attributed to our shared mapping design that learns joint distributions of semantic and pixel-level features. This joint learning approach helps maintain high codebook utilization (95%+) even with large-scale codebooks containing over 131K entries.

ViT-B/14 [37], ViTamin-XL [8], and SigLIP-SO400M [69].

#### B. Additional Results

##### B.1. Additional Ablation Study

Effect of Sampling Strategy to Visual Generation. We conduct comprehensive ablation studies to analyze the im-

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Original

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Reconstructed

- Figure 8. Comparison of original images and their reconstructions from quantized semantic features extracted by VQKD [35]. The reconstructed images preserve the semantic content but exhibit significant loss of high-frequency details.

pact of different sampling strategies on generation quality. As shown in Table 6, we evaluate various configurations using GenEval [15] and ImageReward [63] metrics. We choose ImageReward for ablation due to its strong correlation with human preferences, particularly in capturing local artifacts and overall visual quality. The ImageReward is average over 10k prompts from the MS-COCO validation set. For multi-step configurations, we denote the top-p and topk values for each step using bracket notation [x1, ..., xn].

Our multi-step approach with a two-step strategy (topk=[1200, 1], top-p=[0.8, 0]) significantly improves generation quality, yielding gains of +0.039 in GenEval and +0.084 in ImageReward compared to single-step sampling. This validates our hypothesis that progressive refinement helps maintain global consistency. When increasing the second-step k value to 10 or 100 while maintaining top-p, we observe slightly degraded performance. This degradation suggests that excessive sampling freedom in refinement steps can lead to increased artifacts and local inconsistencies.

Most notably, three-step strategy (top-k=[1200, 100, 1], top-p=[0.8, 0.8, 0]) achieves the best performance across both metrics. This represents substantial improvements of 10.2% and 14.3% over traditional single-step sampling, respectively. The gradual narrowing of sampling space (1200→100→1) strikes a balance between generation diversity and local consistency. As illustrated in Figure 5, our multi-step approach produces more coherent and visually appealing results. These quantitative and qualitative results demonstrates that progressive refinement in top-p topk sampling is crucial for high-quality generation in nextscale prediction frameworks.

Effect of Model Size to Visual Generation. We conduct ablation studies to investigate the impact of model size on our decoder-only visual generation architecture. Specifically, we initialize our framework with two different backbone models: TinyLlama-1B [72] and Llama-2-7B

Table 6. Impact of sampling strategy to visual generation. We compare single-step v.s. multi-step sampling strategy using GenEval and ImageReward. For multi-step approaches, values in brackets indicate parameters for successive sampling steps.

Strategy Top-k Top-p GenEval ↑ ImageReward ↑ Single Step 1200 0.8 0.502 0.722

[1200, 1] [0.8, 0] 0.541 0.806 [1200, 10] [0.8, 0.8] 0.531 0.799

Multi Step

[1200, 100] [0.8, 0.8] 0.529 0.745 [1200, 100, 1] [0.8, 0.8, 0] 0.553 0.825

Table 7. Impact of model size to visual generation.

Model size Training epoches GenEval ↑ ImageReward ↑

1B 4 0.485 0.677 7B 2 0.553 0.825

Table 8. Impact of different input strategies on multimodal understanding. Best results for each metric are highlighted in bold.

Input strategy MME ↑ MME-P ↑ SEEDB ↑ TQA ↑

Full scale 1610.1 1315.1 59.6 49.5 Full scale residual 1527.5 1216.5 57.0 48.1 Last scale semantic feat. only 1580.3 1315.6 60.1 49.7 Last scale 1634.3 1356.5 59.9 49.1

[53]. Experiments demonstrate that model size plays a crucial role in generation performance. As shown in Tab. 7 and Fig. 9, under identical sampling strategies and training dataset configurations, the 1B model significantly underperforms compared to its 7B counterpart, even with doubled training epochs.

Effect of Input Strategy to Multimodal Understanding. We validate different feature input strategies for multi-

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

1B

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

7B

- Figure 9. Qualitative comparison of visual generation capabilities between 1B and 7B models. Prompts (from left to right): (1) ”A pizza sitting on top of a wooden cutting board”, (2) ”Television set being held by a hand”, (3) ”The guy is nicely dressed in a suit and tie”, and (4) ”A sailing ship rests on waters”. The 7B model demonstrates enhanced quality compared to its 1B counterpart.

modal understanding with TokenFlow. As shown in Tab. 8, final-scale features consistently outperform both full-scale features and full-scale residual features across all benchmarks. This suggests that the final scale captures the most relevant semantic information for multimodal understanding, while additional scale features or residual features may introduce noise that compromises performance. Our experiments also reveal that utilizing semantic features only does not improve the overall understanding performance.

Effect of Tokenizer Decoder Finetuning. To further improve our model’s ability to generate fine details, we follow [6] and double both the number of residual layers and channel dimensions in the decoder. We exclusively finetune these enhanced decoder layers while keeping all other components frozen, thereby preserving the learned visual token mappings. This enables us to improve reconstruction fidelity without compromising perception ability of TokenFlow. As shown in Fig. 10, the enhanced decoder yields notable improvements in reconstruction quality. It demonstrates superior preservation of high-frequency details, particularly in facial details and text elements.

##### B.2. More Analysis of TokenFlow

Analysis of Joint Distribution Learning. To evaluate the effectiveness of our shared mapping mechanism, we conduct comparative experiments against VQKD [35] and VQGAN [13]. All models are configured with identical codebook sizes of 8,192 tokens for fair comparison. For baseline models, we utilize the official pretrained checkpoints from [35] and [48], respectively. Our TokenFlow model is trained on ImageNet-1K for 50 epochs. We deliberately excludes the multi-scale VQ design [51] to isolate the effects of the shared mapping in this experiment.

For evaluation, we process 50,000 images from the ImageNet-1K validation set through each model’s encoder.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- (a)
- (b)
- (c)

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Figure 10. Comparison of image reconstruction quality. (a) Original images. (b) Reconstructions using the base pixel decoder. (c) Reconstructions using the enhanced (2× capacity) decoder. The enhanced decoder demonstrates superior preservation of finegrained details, particularly in facial details and textual elements.

We apply average pooling to the extracted features to obtain a 1 × 1 representation, and then identify the closest index in their respective codebooks using l2 distance. As shown in Fig. 7, TokenFlow exhibits significantly smoother distribution against compared to others. The total non-empty clusters of TokenFlow are 7161/8192 (87.4%), which is significantly larger than that of VQGAN (2.5%) and VQKD (27.1%). These results demonstrate that our shared mapping design enables effective learning of joint distributions across high-level semantic and low-level pixel representations. By simultaneously encoding multiple levels of visual information, we induces a joint representation space compared to single-representation architectures. This directly contributes to the superior codebook utilization observed in our experiments. Even when expanding the codebook to over 131K entries, TokenFlow maintains an exceptional utilization ratio exceeding 95%. The clustered results is shown in Fig. 11.

Automatic Balancing between Semantic Distance and Pixel Distance. In our structure, the optimal quantize index is determined by arg mini(dsem,i + wdis · dpix,i). There exists an automatic balancing mechanism between semantic distance and pixel distance. For instance, when encountering a case where dsem,i is relatively small while dpix,i is large, during backpropagation, both commit loss and perceptual loss will contribute to reducing the distance between the encoded features and their quantized counterparts. This mechanism naturally narrows the gap between these two distance metrics. Therefore, we set wdis to 1.0 across all experiments.

Comparison between TokenFlow and their corresponding semantic teachers. Table 9 presents a fair

Table 9. Quantitative comparison of multimodal understanding capabilities between our discrete TokenFlow and their corresponding continuous semantic teachers. All experiments are trained with LLaVA-1.5 data for fair comparison. When calculating average, we use MME-P and divide it by 20 to have the same scale with other benchmarks.

Method # Params Visual Encoder Res. SEEDB MMV POPE VQAv2 GQA TQA AI2D RWQA MMMU MMB MME MME-P Avg. Continuous Visual Input

CLIP ViT-B/14 [37] 224 64.1 30.8 85.1 73.8 61.3 53.4 57.8 50.9 35.1 62.0 1737.0 1460.9 58.9 ViTamin-XL [8] 256 65.7 34.6 85.8 76.8 62.6 57.4 59.4 54.4 35.0 66.4 1839.1 1514.5 61.3

LLaVA-1.5 Vicuna-13B

SigLIP-SO400M [69] 384 67.5 38.1 86.5 78.6 63.8 62.2 59.5 57.4 35.4 68.3 1802.1 1488.2 62.9 Discrete Visual Input

TokenFlow-B 224 60.4 22.4 84.0 70.2 59.3 49.8 54.2 49.4 34.2 55.3 1660.4 1353.6 55.2 (93.7%) TokenFlow-L 256 62.6 27.7 85.0 73.9 60.3 54.1 56.6 49.2 34.4 60.3 1622.9 1365.4 57.5 (93.8%)

Ours Vicuna-13B

TokenFlow-XL 384 65.3 41.2 86.2 76.6 63.0 57.5 56.8 53.3 34.7 62.7 1794.4 1502.3 61.1 (97.1%)

Table 10. Comparison of generation quality on GenEval and DPG-Bench. Obj.: Object. Attri.: Attribute. † result is with rewriting.

GenEval DPG-Bench Method Overall Single Obj. Two Obj. Counting Colors Position Color Attri. Overall Global Entity Attribute Relation Other Diffusion-based SDv1.5 [41] 0.43 0.97 0.38 0.35 0.76 0.04 0.06 63.18 74.63 74.23 75.39 73.49 67.81

- DALL-E 2 [39] 0.52 0.94 0.66 0.49 0.77 0.10 0.19 – – – – – – SDv2.1 [41] 0.50 0.98 0.51 0.44 0.85 0.07 0.17 – – – – – – SDXL [36] 0.55 0.98 0.74 0.39 0.85 0.15 0.23 74.65 83.27 82.43 80.91 86.76 80.41 PixArt-alpha [7] 0.48 0.98 0.50 0.44 0.80 0.08 0.07 71.11 74.97 79.32 78.60 82.57 76.96

- DALL-E 3 [4] 0.67† 0.96† 0.87† 0.47† 0.83† 0.43† 0.45† 83.50 90.97 89.61 88.39 90.58 89.83 Autoregressive meets diffusion

Show-o [62] 0.53 0.95 0.52 0.49 0.82 0.11 0.28 67.27 79.33 75.44 78.02 84.45 60.80 Transfusion [74] 0.63 – – – – – – – – – – – –

Autoregressive-based

Chameleon [48] 0.39 – – – – – – – – – – – – LlamaGen [44] 0.32 0.71 0.34 0.21 0.58 0.07 0.04 64.84 81.76 75.43 76.17 84.76 58.40 EMU3 [55] 0.54 0.98 0.71 0.34 0.81 0.17 0.21 80.60 85.21 86.68 86.84 90.22 83.15 VAR [51] 0.53 0.95 0.60 0.41 0.81 0.16 0.24 71.08 77.51 78.17 77.80 85.80 62.00

0.55 0.97 0.66 0.40 0.84 0.17 0.26

73.38 78.72 79.22 81.29 85.22 71.20

Ours

0.63† 0.93† 0.72† 0.45† 0.82† 0.45† 0.42†

comparison between our discrete TokenFlow variants and their corresponding semantic teachers under the LLaVA-1.5 training paradigm. TokenFlow exhibits a relative performance gap compared to its semantic teachers due to vector quantized distillation. However, this gap diminishes as resolution increases: from 6.3% at 224×224 to 6.2% at 256×256, and finally to 2.9% at 384×384. This improvement can be attributed to the increased number of discrete tokens and additional scales supplementing the residual features at higher resolutions.

##### B.3. More Visual Generation Results

Quantitative Results. In Tab. 10, we present the complete scores for both GenEval [15] and DPG-Bench [18]. Following DALL-E 3 [4], we report our GenEval results using GPT-4V as a rewriter. For DPG-Bench, we tested the results of LlamaGen and Show-o using their released checkpoints. We compare against VAR [51] by using their released tokenizer and training the visual generation model under identical settings to ensure fair comparison.

Qualitative Results. We present additional visual generation results in Fig. 12. Our method can generate images

with various styles, subjects, and scenarios.

#### C. Limitation and Future Work

A primary limitation of TokenFlow lies in the performance gap in multimodal understanding between our discrete tokenizer and its continuous semantic teacher, which stems from the vector quantization distillation process. While this gap narrows to 2.9% at 384×384 resolution, several methods remain for further improvement, such as incorporating text alignment loss during tokenizer training.

In this work, we primarily focused on designing TokenFlow and validating its effectiveness separately in multimodal understanding and visual generation tasks. A natural extension of this work is the development of a fully unified model for both multimodal understanding and generation. This unification can be achieved through joint training on interleaved vision-language data. This is currently in our high priority for exploration.

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

[Figure 139]

[Figure 140]

[Figure 141]

VQKD

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

VQGAN

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

TokenFlow

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

- Figure 11. Qualitative comparison of images clustered by VQKD [35], VQGAN [13] and our TokenFlow. VQKD clusters exhibit semantic similarity, while VQGAN clusters exhibit low-level similarity (i.e. color and texture). Our TokenFlow can successfully combine both semantic and low-level similarity (e.g. birds with different background can be mapped into two different index).

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

A picture of the head of a brown cow wearing a halter.

A bedroom with a white bed on a frame next to a window.

A man with a bald head wearing a pair of glasses.

A duck floating on a lake with gray and black feathers.

A photo of a man holding a sign with text 'FLOW'.

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

A toy smiley face in the middle of a doughnut.

A couple of vehicles are side by side.

A breakfast of croissant and coffee sits on a table.

Aman with long hair with a pizza in front of him on the table.

An elephant walking under the sea.

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

A photo of a potted plant. A photo of two wine glasses. A photo of a yellow tv remote. A photo of a red apple.

A photo of a purple backpack and a white umbrella.

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

A handsome 24 years old boy in the middle with sky color background wearing eye glasses, it's super detailed with anime style.

Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.

Crocodile in a sweater. A deep forest clearing with a

A realistic landscape shot of the Northern Lights dancing over a snowy mountain range in Iceland.

mirrored pond reecting a galaxylled night sky.

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

A vivid green iguana is perched motionlessly atop a worn wooden log, its intricate scales exhibiting various shades of green and black.

An intricately detailed representation of the Marvel character Ghost Rider featuring a human skull, with flames licking around the contours of the skull and rising above it in a fierce expression of fiery vengeance.

A vibrant yellow 2017 Porsche 911 is captured in motion, navigating a winding mountain road with its sleek body hugging the curve.

An astronaut riding a horse on the moon, oil painting by Van Gogh.

A lighthouse in a giant wave, origami style.

###### Figure 12. More Visual Generation Results with TokenFlow. We present diverse 256×256 results across various styles, subjects, and scenarios.

Table 11. Detail settings of TokenFlow-B, TokenFlow-L and TokenFlow-XL.

Tokenizer TokenFlow-B TokenFlow-L TokenFlow-XL Tokenizer settings:

Input resolution 224 256 384 Codebook size 32,768 32,768 32,768 Semantic teacher CLIP ViT-B/14-224 [37] ViTamin-XL-256 [8] SigLIP-SO400M-patch14-384 [69] Multi-scale settings [1, 2, 4, 6, 8, 10, 12, 14] [1, 2, 3, 4, 6, 8, 10, 12, 14, 16] [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 22, 27] Semantic codebook embedding dimension 32 32 32 Pixel codebook embedding dimension 8 8 8

Training settings:

Learning rate 1e-4 1e-4 1e-4 Batch size 256 256 256 Training steps 1,000,000 500,000 500,000

Distance balance weight wdis 1.0 1.0 1.0 Commitment loss factor β 0.25 0.25 0.25

Adversarial loss factor λG 0.5 0.5 0.5 Max gradient norm 1.0 1.0 1.0

