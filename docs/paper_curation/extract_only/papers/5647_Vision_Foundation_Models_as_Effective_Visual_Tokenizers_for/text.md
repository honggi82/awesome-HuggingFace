# arXiv:2507.08441v2[cs.CV]25Oct2025

## Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Generation

Anlin Zheng1 Xin Wen1 Xuanyang Zhang2∗ Chuofan Ma1 Tiancai Wang3 Gang Yu2 Xiangyu Zhang2,4 Xiaojuan Qi1∗† 1The University of Hong Kong 2StepFun 3Dexmal 4MEGVII Technology

### Abstract

In this work, we present a novel direction to build an image tokenizer directly on top of a frozen vision foundation model, which is a largely underexplored area. Specifically, we employ a frozen vision foundation model as the encoder of our tokenizer. To enhance its effectiveness, we introduce two key components: (1) a region-adaptive quantization framework that reduces redundancy in the pre-trained features on regular 2D grids, and (2) a semantic reconstruction objective that aligns the tokenizer’s outputs with the foundation model’s representations to preserve semantic fidelity. Based on these designs, our proposed image tokenizer, VFMTok, achieves substantial improvements in image reconstruction and generation quality, while also enhancing token efficiency. It further boosts autoregressive (AR) generation—achieving a gFID of 1.36 on ImageNet benchmarks, while accelerating model convergence by three times, and enabling high-fidelity class-conditional synthesis without the need for classifier-free guidance (CFG). The code is available at https://github.com/CVMI-Lab/VFMTok.

### 1 Introduction

GPT’s success in language generation has spurred interest in autoregressive (AR) image generation [43, 44, 50], which relies on visual tokenizers like VQGAN [13, 39, 43, 48, 52] to map images into compact, discrete latent spaces. However, these tokenizers, typically trained from scratch and optimized for reconstruction, often yield latent spaces rich in low-level details but poor in highlevel semantics and laden with redundancy. Such flawed latent spaces not only prolong AR model training (Fig. 1) but also necessitate techniques like classifier-free guidance (CFG) for high-fidelity, class-conditional image generation, which in turn increases inference time.

In parallel, within the field of computer vision, pre-trained vision foundation models such as DINOv2 and CLIP [9, 34, 37, 46, 58] have demonstrated strong capabilities in extracting semantically rich and generalizable visual features. Early explorations in diffusion-based image generation—e.g., REPA [57]—suggest that the semantic representations learned by these models can facilitate the training of generative models. This leads to a natural and compelling question: Can the latent features from vision foundation models, originally designed for visual understanding, also serve as robust and structured representations for image reconstruction and generation?

Recent studies [63, 65] have started exploring this direction by leveraging features from vision foundation models to initialize quantizer codebooks [63, 65], augment VQGAN architectures with additional branches [36], or distill these features to guide latent space learning [41]. Although these approaches show promise, they typically treat foundation model features as auxiliary components rather than fully capitalizing on their potential as generative priors. As a result, these methods often suffer from inefficiencies and fail to fully utilize the rich semantic information embedded in foundation model features, leaving their generative capabilities largely underexplored.

† Corresponding author: xjqi@eee.hku.hk. ∗ Project lead.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

[Figure 1]

[Figure 2]

[Figure 3]

3.0× faster

(a) Region-adaptive quantization (b) Converge speed improvement

- Figure 1: VFMTok introduces novel features, including: a).region-adaptive quantization— where it adaptively samples regions of similar patterns and extracts their VFM features for quantization; b).convergence speed improvement compared with vanilla VQGAN [43] for AR image synthesis.

Can VFMs be effective tokenizers? To address this, we initialized the encoder of a VQGAN with different frozen pre-trained foundation models to reconstruct images. Once trained, the tokenizer is integrated on top of an AR model for image synthesis (implementation details depicted in Sec. 3.2) As shown in Tab. 1 (middle rows), our results demonstrate that the semantically rich features from these foundation models not only support effective image reconstruction but also achieve generative performance comparable to—or even surpassing—that of a fully trained VQGAN encoder optimized for both reconstruction and generation. These findings highlight the strong potential of pre-trained vision foundation models to serve as efficient and effective tokenizers for image generation tasks, eliminating the need for extensive encoder training while improving qualities.

Can we improve token efficiency for VFMs? Building on this pilot study, we are further motivated by the observation that natural images often consist of irregular regions that exhibit recurring visual patterns. For example, as illustrated in Fig. 1(a), the upper portion of the crystal ball exhibits consistent patterns such as texture and transparency; similarly, the moss in the stone possesses similar textural structure. When such images are represented using a regular 2D feature grid extracted from foundation models, this structure-agnostic representation may introduce significant redundancy. Exploiting redundancy within semantically coherent regions offers a promising direction for improving tokenization efficiency. Motivated by this insight, we propose a region-adaptive strategy to refine the latent space that aims to enhance both image reconstruction and generation quality while significantly improving token representation efficiency.

Our solution and results. Guided by the preceding experimental analysis and insights, we introduce VFMTok, an image tokenizer that leverages a frozen pre-trained vision foundation model for adaptive region-level tokenization. VFMTok is designed to achieve high reconstruction and generation quality with improved token efficiency. Specifically, VFMTok employs a frozen pre-trained VFM as an encoder to extract multi-level semantic features. A set of learnable anchor queries performs region-level sampling on these features via deformable attention [64], producing region-adaptive tokens that are subsequently quantized into discrete tokens representing the image’s latent representation. These contextual tokens are then processed by a lightweight Vision Transformer [12](ViT) in a BERT-style framework [11, 15] with two primary reconstruction objectives. First, the original image pixels are reconstructed after dequantization using a VQGAN [43] decoder. Then, the model reconstructs the features from the frozen foundation model itself, allowing VFMTok to retain the semantic richness and discriminative power of the original representations. Once trained, VFMTok enables standard autoregressive Transformers (e.g., Llama [45]) to generate contextual token sequences, which are decoded back into images via the VQGAN decoder, facilitating high-quality image synthesis with compact and semantically mean-

Table 1: Pilot study of image reconstruction and generation on ImageNet [10]. Relative wall-clock inference time for the tokenizer (compared to VFMTok) is reported. L.P. denotes linear probing results on the ImageNet validation set, used to estimate the semantic quality of latent tokens.

Image Recon. AR Generation L.P.

Setup

#Tok. rFID↓ rIS↑ gFID↓ gIS↑ Time (%) VQGAN [43] 576 0.95 197.3 3.71 228.3 4.3 23.1 VQGAN (CLIP) 1.47 182.0 3.45 221.2 4.0 59.5 VQGAN (SigLIP2) 0.96 198.4 3.39 267.8 4.0 55.5 VQGAN (DINOv2)

576

0.99 206.3 3.34 268.6 4.0 56.4

VFMTok (CLIP) 0.99 200.1 3.40 274.7 1.0 63.9 VFMTok (SigLIP2) 0.94 218.7 3.01 280.8 1.0 78.5 VFMTok (DINOv2)

256

0.89 215.4 3.08 274.2 1.0 69.4

ingful representations. As shown in Tab. 1 (bottom rows), VFMTok achieves superior reconstruction and generation performance while using fewer than half the original number of tokens (256 vs. 576).

Extensive experiments validate that VFMTok, by combining the representational power of visual foundation models with a novel region-adaptive tokenization strategy based on irregular sampling and learnable anchor queries, enables both high-quality and efficient image reconstruction and autoregressive (AR) generation. First, VFMTok achieves superior reconstruction quality and captures richer semantics using significantly fewer tokens compared to prior methods (e.g., 256 vs. 576 in [43]), resulting in a structured, semantic-aware, and compact latent space. As shown in Tab. 1, VFMTok, with only 256 tokens, outperforms other tokenizers using the same VFM encoder by delivering superior reconstruction quality and stronger semantic representation (as indicated by linear probing). Second, the high-quality latent space produced by VFMTok facilitates effective AR training using a simple LLaMA-based model, leading to faster convergence (see Fig. 1(b)) and improved generation performance. Notably, the 1.4B AR model surpasses the performance of LlamaGen-3B despite having fewer parameters and requiring fewer training iterations. The 1.5B advanced AR model achieves a new state-of-the-art with a gFID of 1.36 on ImageNet [10] 256 × 256, outperforming widely-used diffusion models. Third, due to the compact token space and the reduced number of tokens, VFMTok significantly improves the inference speed of AR models (see Tab. 1). Moreover, the rich semantic content embedded in the latent tokens enables effective class-conditional image synthesis with high fidelity—without the need for classifier-free guidance—further reducing inference time.

Our contributions can be summarized as follows:

- • We demonstrate that frozen vision foundation models—ranging from self-supervised to languagesupervised—are effective for image reconstruction and generation. Leveraging their semantic richness enhances the tokenizer and enables AR generation models to converge faster and perform high-fidelity, CFG-free image synthesis, without bells and whistles.
- • We propose a region-adaptive tokenization framework that effectively leverages inherent redundancies in image regions to achieve compact tokenization. This approach reduces the number of visual tokens while enhancing performance, enabling efficient AR generation without sacrificing quality.
- • Extensive experiments validate the effectiveness of our approach in both image reconstruction and AR generation, establishing pre-trained vision foundation models as powerful tokenizers for high-quality and efficient image generation.

### 2 Related Work

Image Tokenization using Autoencoders. Pixel-space images are highly redundant. Autoencoderbased tokenizers [30, 43, 44, 54] create compact latent tokens to reduce redundancy. VQVAEs [21, 39, 48] and their derivatives evolved using adversarial losses [13], Transformers [52], multistage quantization [24, 60], lookup-free methods [30, 32], and codebook initialization from pre-trained features [63, 65]). These 2D tokenizers map features to a static 2D grid, which limits redundancy exploration. Recent 1D tokenizers [1, 33, 49, 56] offer superior compression, reconstruction, and redundancy removal, but often require complex and lengthy training. For example, TiTok [56] requires a two-stage process (warming up and fine-tuning) for 200 epochs. Our VFMTok adopts a novel region-adaptive tokenization framework to reduce redundancy. With a simpler training strategy for only 50 epochs, VFMTok exhibits discriminative semantics and excellent generation results.

Vision Foundation Models. Vision Foundation Models (VFMs) [3, 6, 14, 16, 17, 20, 34, 37, 46, 58] aim to learn general, transferable visual representations from large-scale, diverse data. The training of these versatile models has shifted from early supervised approaches to more scalable self-supervised learning [3, 6, 9, 11, 14–16, 34], which leverages inherent data structures. More recently, language-supervised pre-training [20, 46, 58] on vast image-text pairs has enabled VFMs to learn rich, semantically grounded representations. Pre-trained VFMs serve as powerful backbones for a wide array of downstream tasks. In this work, we utilize pre-trained VFMs directly as image tokenizers for AR image generation, surpassing other methods [63, 65] with superior performance. Furthermore, using VFMs as tokenizers enables the removal of classifier-free guidance.

Autoregressive Image Generation. GPT-style Transformers [5, 24, 38, 43, 44, 47] have spurred interest in autoregressive (AR) image generation, which predicts visual token sequences. While early AR models operated in pixel space [5, 47], current methods [24, 43, 44, 52] generate discrete latent tokens via next-token prediction, then decode them to pixels using a tokenizer’s decoder [13, 39, 48, 52]. To improve the generation quality, recent works [25, 44, 50] add bidirectional attention (e.g., VAR’s

next-scale prediction [44], MAR’s BERT-style framework [25], Show-o’s hybrid attention [50]). These innovations, however, complicate designing universal, multi-modal Transformers adhering to next-token prediction. Instead, our VFMTok enables standard AR transformers to generate contextual token sequences for subsequent decoding, eliminating complex structural modifications.

### 3 Method

In this section, we first provide preliminary background on quantized image tokenizers. We then present our pilot studies exploring the use of vision foundation models for tokenization. Finally, we introduce VFMTok, a novel tokenizer built upon frozen vision foundation models, incorporating region-adaptive strategies to enhance both the efficiency and effectiveness of the tokenization process.

#### 3.1 Preliminary

Quantized Image Tokenizer. To apply autoregressive modeling to visual generation, existing methods [43, 44, 52, 53] necessitate an image tokenizer to map a 2D image into discrete token sequences for AR generation. To achieve this, quantized autoencoders, such as VQVAEs [13, 39, 43, 44, 48, 52, 63], are widely used. Typically, an image tokenizer consists of an encoder E(·), a quantizer VQ(·), and a decoder D(·). Given an input image I ∈ RH×W×3, the encoder E(·) first

f ×Wf ×D with spatial down-sampling factor f.

H

convert an image into patch embeddings Z2D ∈ R

Then, Z2D is fed into the quantizer VQ(·) that includes a learnable codebook C ∈ RN×D with N vectors. Each feature vector zi ∈ RD is mapped into its nearest vector ci ∈ RD in the codebook C.

Z2D = E(I), VQ(zi) = ci, where i = arg min

∥zi − cj∥2 , (1)

j∈{1,2,...,N}

where H and W denote the input image’s height and width, respectively. D depicts the latent feature dimension. Once discrete tokens are acquired, they can be de-quantized into corresponding code and converted back to image pixels by the decoder D(·), as depicted in Eq. (2).

ˆI = D(VQ(Z2D)). (2) To optimize the codebook, the training objective is Lvq = ∥sg(zi) − ci∥22 + β · ∥sg(ci) − zi∥22, where sg(·) is a stop-gradient function [2, 48]. The second term is a commitment loss weighted by β to align extracted features with codebook vectors. For image reconstruction, the loss function is LAE = L2(I,ˆI) + LP(I,ˆI) + λG · LG(ˆI), where L2 is a pixel-wise reconstruction loss, LP is perceptual loss from LPIPS [59], and LG is adversarial loss from PatchGAN [19] with weight λG.

#### 3.2 Pilot Study: Pre-trained Vision Foundation Models as Tokenizers for AR Generation

To assess whether a pre-trained VFM can serve as a tokenizer for image reconstruction and benefit image generation, we performed a pilot study. In our setup, we extract the final 2D grid features from images of size 336 × 336 using a frozen VFM, such as DINOv2, CLIP, and SigLIP2. These features, after quantization, are fed into a VQGAN [43] decoder for image reconstruction. Once trained, the tokenizer is integrated on top of a Llama-based AR model for image synthesis. Additionally, the training duration for VQGANs and AR models is 50 and 100 epochs, respectively.

As Tab. 1 illustrates, directly using features from pre-trained VFMs yields decent image reconstruction and generation performance compared to vanilla VQGANs. Notably, these VFM-based tokenizers consistently exhibit stronger semantic representation capabilities (as indicated by the linear probing experiment in Tab. 1). For instance, VQGAN (SigLIP2) achieves reconstruction performance on par with vanilla VQGAN, while exhibiting better semantic representation and superior generation quality. Nevertheless, variations in image reconstruction and generation quality arise when different VFMs are used to initialize the tokenizer’s encoder. Specifically, VQGAN (DINOv2) and VQGAN (SigLIP2) demonstrate similar reconstruction and generation quality, both outperforming vanilla VQGAN, while the reconstruction quality of VQGAN (CLIP) trails that of vanilla VQGAN. One contributing factor is that different learning objectives used to train VFMs influence their ability to extract detailed and semantic features from images, thereby affecting downstream image reconstruction and generation quality. As evidence, both DINOv2 [9] and SigLIP2 [46] employed a masked prediction objective to optimize their VFMs, whereas CLIP [37] did not.

#### 3.3 VFMTok

Building upon the semantically rich features provided by vision foundation models—typically structured as regular 2D grids—we introduce VFMTok, a region-adaptive tokenizer that identifies semantically coherent, irregular local regions to produce region-adaptive tokens. These tokens are sequentially quantized for decoding, with tailored learning objectives to enhance performance. In the following, we detail the architecture of VFMTok, including its region-adaptive token generation module and dedicated decoder for both image and feature reconstruction. We further describe the training objectives, which combine a pixel-level reconstruction loss for image synthesis with a feature reconstruction loss that preserves the semantic content of the foundation model’s representations.

𝐿

###### Encoding Quantization Decoding

[Figure 4]

•••

[Figure 5]

[Figure 6]

multi-level feature

❆

[Figure 7]

sharedViT

[Figure 8]

+

[Figure 9]

FoundationModel

•••

DeformableTransformer

••••••

[Figure 10]

[Figure 11]

[Figure 12]

••••

••••

Quantizer

share weights

[Figure 13]

Decoder

[Figure 14]

[Figure 15]

sharedViT

Reshape

•••

+

•••

•••

•••

•••

•••

| |
|---|

region-adaptive tokens reconstructed foundation model features

feature mask embedding image latent features

learnable anchor queries image mask embedding

foundation model features shared positional embeddings

- Figure 2: The framework of VFMTok. VFMTok utilizes a frozen VFM to extract multi-level image features. A deformable Transformer then processes these features with learnable grid queries to generate region-adaptive tokens. After quantization, these tokens are fed into a shared ViT for dual reconstruction: 1) VFM features, targeting similarity with the VFM’s last-layer outputs, and 2) image latent features, which are reshaped to a 2D grid and decoded into pixels.

Region-adaptive Token Generation. Following our pilot study, we utilize a frozen pre-trained vision foundation model (VFM) to encode an input image I into latent embeddings F. Since features extracted from VFMs contain rich details in shallower layers and high-level semantics in deeper layers [7, 27, 28]—both of which are critical for image reconstruction—we extract multi-level features Fm from the VFM. These multi-level features are then projected to a uniform embedding dimension using a two-layer MLP. Next, as shown in Fig. 2, based on the multi-level features Fm, we introduce a region-adaptive sampling mechanism using deformable cross-attention layers [8, 64]. A set of learnable anchor queries, initialized as a 2D grid, are iteratively refined through multiple deformable attention layers. In each layer, an anchor query predicts sampling offsets for each VFM feature level via a linear layer, enabling sampling from irregular, data-dependent positions. These sampled features are then weighted using attention scores—computed through another linear layer—and aggregated to update the query. Through this process, the anchor queries are progressively refined to capture semantically coherent, region-specific information. The final refined queries are referred to as region-adaptive tokens Zr, which are subsequently quantized into discrete tokens Z˜r. Compared to a fixed 2D feature grid, VFMTok adaptively aggregates features from semantically coherent, irregular regions. This substantially reduces redundancy, enabling the use of fewer tokens while maintaining superior image reconstruction and generation performance. As shown in Tab. 1, just 256 semantically rich tokens from VFMTok are sufficient to achieve high-fidelity reconstruction and generation.

Vector Quantization. Once the continuous region-adaptive tokens Zr are obtained, a quantizer Qc(·) is applied to discretize them into region-adaptive discrete tokens Z˜r. Given that the design of the codebook plays a critical role in the performance of an image tokenizer, we follow the practices in [43, 52] by applying ℓ2-normalization to the codebook vectors. Additionally, we adopt a lowdimensional embedding space with a large codebook size to enhance both reconstruction quality and codebook utilization following [43, 52].

Decoder of VFMTok for Image and VFM Feature Reconstruction. After de-quantization, the region-adaptive tokens Z˜r are used for image reconstruction. Since these tokens represent irregular,

region-level features, decoding them into a regular 2D image grid requires alignment. To achieve this, we introduce a set of mask tokens MI, representing a 2D feature map of size Hm × Wm with channel dimension C. The mask tokens are initialized by replicating a single learnable token Hm×Wm times. Position embeddings E, encoding spatial locations, are then added to form position-aware masked tokens. Next, the de-quantized region-adaptive tokens Z˜ are concatenated with MI, and the combined sequence is processed by a lightweight Transformer EViT(·), which propagates information from the region-adaptive tokens to the masked image tokens. This Transformer employs causal self-attention, aligning its latent space with the structure of autoregressive models. Following DINOv2 [9], we further enrich the input sequence by appending a [CLS] token and several register tokens to improve representation learning and capture global context—though these are not used for reconstruction. The output of this Transformer is a refined set of mask tokens FI representing a regular 2D grid structure. These are reshaped into a spatial grid and passed into a decoder D(·) to reconstruct the image.

To preserve the semantic integrity of the VFMTok tokens, we also reconstruct high-level features (specifically, from the final layer) of the vision foundation model (VFM). This process mirrors image reconstruction: a new set of mask tokens Mf is initialized and augmented with positional embeddings E, shared with those used in image reconstruction. The concatenation of Zr and Mf is then processed by the same shared Transformer EViT(·) to produce FP, the reconstructed high-level VFM feature map. By sharing EViT(·) between image and feature reconstruction, we reduce the model’s parameter footprint while ensuring the semantic fidelity of the latent tokens. Note that the VFM feature reconstruction is only applied during tokenizer training.

Training Objective. For tokenizer optimization, we follow the training objectives of VQGAN [13, 43], with one key modification: we replace its original discriminator with a pre-trained DINOv1-S [3] model. This substitution provides adversarial training guidance in a more semantically meaningful way compared to conventional discriminators such as PatchGAN [19], and we find it consistently improves reconstruction quality. In addition to image reconstruction, we incorporate a feature reconstruction objective by computing the cosine similarity loss between the reconstructed features and the corresponding frozen features from the pre-trained vision foundation model (VFM). The overall training loss is defined as: L = α·LAE+λ·Lsim, where LAE denotes the image reconstruction loss and Lsim is the feature reconstruction loss. In our experiments, we set both α and λ to 1.

#### 3.4 Autoregressive Image Generation

Once VFMTok is trained, the optimized discrete region-adaptive tokens Z˜r can be integrated into an autoregressive (AR) Transformer, where they are generated sequentially via a next-token prediction mechanism, conditioned on a class or text embedding c. The generated tokens are then passed through the Transformer encoder EViT(·) to produce latent image features FI, which are subsequently decoded into images using the decoder D(·). In the AR model, the region-adaptive tokens Z˜r are augmented with positional embeddings—specifically 2D Rotary Position Embeddings (RoPE) [42]—to better capture their spatial locality and structure.

### 4 Experiment

#### 4.1 Setup

Image Tokenizer. In the main experiment, we initialize the encoder of VFMTok with a frozen pre-trained DINOv2-L [9]. Considering its composition of 24 Transformer layers, we extract features from the 6th, 12th, 18th, and 24th layers to create multi-level features. Consistent with [43, 56], we set the codebook vector dimension of the quantizer to 12 with a codebook size of 16384, to achieve a better reconstruction quality and efficient codebook utilization. Meanwhile, VFMTok utilizes 256 tokens to represent an image. Besides, the depth of the Transformer is set to 6 (following [64]). The model is trained on the ImageNet [10] training set and evaluated on its validation set.

Given that the resolution of vision foundation models (VFMs) [9, 20, 37, 46, 58] is typically 336×336, while VFMTok represents images with fixed 256 tokens by default, it’s comparable to vanilla tokenizers [13, 43, 63]. Thus, we train the tokenizer on 336 × 336 images. Except this, we keep the training settings unchanged as LlamaGen [43]. During evaluation, the reconstructed images of 336 × 336 are resized to 256 × 256 for evaluation, which is consistent with the evaluation procedure in LlamaGen [43].

Class-conditional Autoregressive Image Generation. Following the generation procedure in LlamaGen [43], the AR models first generate images of 336 × 336 and then resize them to 256 × 256 for evaluation. Considering computational costs, we set the training duration based on the number of models’ parameters. Models with fewer than 1B parameters are trained for 300 epochs, while the remaining models are trained for 200 epochs. Beyond the resolution and training duration, all models are trained with the same settings as LlamaGen [43]. Furthermore, we also incorporated the same VFMTok into the RAR [55] autoregressive generation framework, with all training settings remaining consistent with RAR [55]. Additionally, in our experiments, AR generation is conducted with both classifier-free guidance (CFG) and a CFG-free protocol.

Evaluation metrics. To evaluate image generation performance, we use Fréchet inception distance (FID) [18] and Inception Score (IS) [40] as the main metrics to measure the generation quality of different models. In addition, sFID, Precision, and Recall [23] are also reported following [43].

#### 4.2 Main Results

Table 2: Comparison with other image tokenizers. oim. indicates trained on OpenImages [22]. Qc/QP denotes the codebook usage in contextual and patch-level quantizers, respectively.

Image Reconstruction. We compare VFMTok against representative 2D image tokenizers, VQGAN [13], MaskGiT [4], ViT-VQGAN [52], and 1D tokenizer, TiTok [56]. As shown in Tab. 2, our tokenizer represents an image with just 256 tokens, considerably fewer than some counterparts. For instance, the VQGAN variant LlamaGen [43] uses 576 tokens, while VQGAN [13] and ViT-VQGAN [52] even utilize up to 1024 tokens. Despite this efficiency, VFMTok achieves a strong rFID of 0.89, and further demonstrates 100% utilization of the codebook.

Tokenizer Setup Image Recon. Usage (%)↑

Method

f Size Dim. #Tok. rFID↓ rIS↑ QC QP

TiTok [56] – 8192 64 256 1.05 191.5 100 – ImageFolder [26] – 32768 32 286 0.69 201.5 100 –

VQGANoim. [13]

1.44 – – – VQGAN [13] 8192 256 1.49 – – – ViT-VQGAN [52] 8192 32 1.28 192.3 – 95.0 VQGANoim. [13] 16384 4 1.19 – – –

256 4

1024

8

7.94 – – – MaskGiT [4] 2.28 – – – VAR [44] 4096 32 680 0.92 196.0 – 100

VQGAN [13]

1024 256 256

16

RQ-VAE [24] 32 16384 256 1024 1.83 – – – VQGAN [13]

256 256 4.98 – – – VQGAN [43]

16 16384

441 1.21 189.1 – 99.2 VQGAN [43] 576 0.95 197.3 – 99.7 VFMTok (Ours) – 16384 12 256 0.89 215.4 100 –

8

The rIS score of 215.4 achieved by VFMTok significantly outperforms other methods, e.g., TiTok [56] and the VQGAN series. The rIS metric quantifies the KL-divergence between the original label distribution and the logit distribution of reconstructed images after softmax normalization, thereby measuring the semantic consistency between reconstructed and original images. The higher rIS confirms VFMTok is more effective at preserving semantic consistency during reconstruction.

Class-conditional Image Generation. We evaluate VFMTok on vanilla autoregressive models – LlamaGen [43], and advanced generative model – RAR [55] with different scales by conducting 256 × 256 class-conditional image generation task on ImageNet [10], where comparing them with the mainstream generation models, including diffusion models (Diff.) [31, 35, 51, 61], BERT-style masked-prediction models (Mask.) [4], and AR generation models (AR) [13, 24, 39, 43, 52, 56].

- As shown in Tab. 3, our models exhibit competitive performance across all metrics compared to mainstream image generation models. Notably, VFMTok beats BERT-style models [4] in terms of gFID without the requirement of complicated sampling tuning. With comparable or even fewer parameters, our method surpasses most AR generative models [13, 24, 39, 52, 56] in both gFID and gIS metrics. Under the same training setting, VFMTok surpasses LlamaGen [43] by significant gFID gains and notable gIS improvements. Specifically, VFMTok-B outperforms LlamaGen-B [43] with gains of 2.56 in gFID and 69.7 in gIS. Besides, our VFMTok-L model achieves a gFID of 2.75 at 300 epochs, also obtaining a gain of 22.7 in gIS. Notably, when compared with LlamaGen-3B with

##### 3B parameters, our VFMTok-XXL achieves even better generation performance with less than half the number of parameters and fewer training iterations. Futhermore, when VFMTok is incorporated into RAR [55], it achieves a generative performance with gFID of 1.36, which is the state-of-the-art generation performance at present. Additionally, class-conditional image generation results are visualized in the Appendix.

Table 3: Class-conditional image generation quality estimated on ImageNet [10] validation benchmark. † indicates it is implemented by us, and ‘-re’ indicates using rejection sampling.

Generation w/ CFG Generation w/o CFG gFID sFID gIS Pre. Rec. gFID sFID gIS Pre. Rec.

Type Method #Epoch #Para. #Tok.

MaskDiT [61] 1600 675M

2.28 5.67 276.6 0.80 0.61 5.69 10.34 177.9 0.74 0.60 DiT [35] 1600 675M 2.27 4.60 278.2 0.83 0.57 9.62 6.85 121.5 0.67 0.67 SiT [31] 1600 675M 2.06 4.50 270.3 0.82 0.59 8.61 6.32 131.7 0.68 0.67 FasterDIT [51] 400 675M 2.03 4.63 264.0 0.81 0.60 7.91 5.45 131.3 0.67 0.69

Diff.

–

MaskGiT [4]

– – – – – 6.18 – 182.1 0.80 0.51

Mask.

555 227M 256

MaskGiT-re 4.02 – 355.6 – – – – – – –

VAR [44] 350 310M 680 3.30 – 274.4 0.84 0.51 – – – – – TiTok-B† [56]

6.76 7.82 175.3 0.85 0.43 19.6 57.9 7.54 0.64 0.60

111M

300

256

TiTok-L† [56] 343M 4.03 6.93 219.5 0.84 0.52 11.4 88.8 7.14 0.68 0.64 LlamaGen-B

6.09 7.24 182.5 0.85 0.42 32.2 11.84 39.9 0.57 0.61 LlamaGen-L 343M 3.07 6.09 256.1 0.83 0.52 19.1 8.67 64.3 0.61 0.67 LlamaGen-XXL 1.4B 2.34 6.00 253.9 0.81 0.60 14.6 8.69 86.3 0.63 0.68 LlamaGen-3B 3.1B 2.19 5.97 263.3 0.82 0.58 9.38 8.24 112.9 0.69 0.67

111M

300

576

RAR-L [55]

1.70 – 299.5 0.82 0.58 6.72 5.56 129.2 0.74 0.61 RAR-XL [55] 955M 1.50 – 306.9 0.80 0.62 4.62 5.27 158.3 0.77 0.62 RAR-XXL [55] 1.5B 1.48 – 326.0 0.80 0.63 3.85 5.18 176.2 0.78 0.61

461M

AR

400

256

VFMTok-B 111M 3.43 5.88 252.2 0.85 0.53 3.09 5.67 173.6 0.80 0.58 VFMTok-L

300

343M 2.75 5.58 278.8 0.84 0.57 2.11 5.46 230.1 0.82 0.60 VFMTok-XXL 200 1.4B 2.19 5.53 278.0 0.83 0.60 1.95 5.65 259.3 0.82 0.62 VFMTok-3B 200 3.1B

256

2.07 6.23 280.4 0.81 0.62 2.04 5.43 267.6 0.82 0.61

RAR-L(VFMTok) 461M 1.44 6.03 312.8 0.78 0.66 2.02 5.51 210.4 0.79 0.63 RAR-XL(VFMTok) 955M 1.38 5.86 310.2 0.78 0.65 1.74 5.33 233.0 0.80 0.63 RAR-XL(VFMTok)

400

256

1.36 5.85 301.3 0.78 0.66 1.65 5.55 253.7 0.80 0.63 VFMTok-L(SigLIP2) 300 343M

1.5B

2.69 5.31 273.4 0.84 0.56 2.11 5.39 225.6 0.81 0.60 VFMTok-XXL(SigLIP2) 200 1.4B 2.16 5.45 272.0 0.83 0.60 1.98 5.53 265.3 0.82 0.62 VFMTok-2B(SigLIP2) 200 2.2B 2.17 5.43 281.4 0.83 0.60 1.98 5.41 269.7 0.82 0.62

256

Furthermore, we conducted experiments by removing classifier-free guidance (CFG). Remarkably, the generation results without CFG show that most evaluation metrics—such as sFID, Precision, and Recall—remain comparable to those obtained with CFG. While gIS experiences a slight decline, gFID improves compared to its CFG-enabled counterpart. Similar trends are observed when VFMTok’s encoder is replaced with other frozen pre-trained vision foundation models like SigLIP2 [46]. These results demonstrate that our method supports high-fidelity autoregressive image generation even without CFG, which significantly accelerates inference. In contrast, baseline methods suffer substantial performance degradation without CFG—for example, LlamaGen-3B model sees gFID worsen to 9.38, whereas our 1.4B model VFMTok-XXL achieves a gFID of 1.95 without CFG.

#### 4.3 Ablation Study and More Analysis

Component study. To assess the contribution of each proposed component to image reconstruction and synthesis, we conduct a step-by-step component analysis using a baseline tokenizer built on vanilla VQGAN [43]. We incrementally add the following components: (1) replace the VQGAN encoder with a frozen pre-trained foundation model (DINOv2-L [9]); (2) introduce learnable queries and a deformable attention for region-adaptive tokenization, using only single-level features from the final layer; (3) incorporate multi-level features to enrich representations with both low-level detail and high-level semantics; and (4) add a feature reconstruction objective based on pre-trained VFM outputs. After training each tokenizer, we integrate it with our AR generation model, VFMTok-L, for autoregressive image synthesis. Both the tokenizer and AR model are trained for 50 epochs. Additionally, we perform linear probing on the [CLS] token, following the MAE [15] protocol.

- As shown in Tab. 4, replacing VQGAN’s encoder with a frozen pre-trained vision foundation model yields reconstruction and generation performance on par with a VQGAN trained specifically for

visual reconstruction using 576 tokens. This substitution also significantly enhances the semantic quality of the tokenizer’s representations. To further improve token efficiency, we introduce regionadaptive tokenization using deformable attention to exploit the spatial redundancy inherent in regular 2D grid features. This reduces the number of visual tokens to 256. However, this performance gain comes at a cost: reconstruction and generation quality degrade slightly due to two factors: (1) fewer visual tokens limit representational capacity, and (2) the absence of explicit supervision hinders the effective optimization of the region-adaptive tokens. To address this, we incorporate multi-level feature extraction, which improves the reconstruction capability by leveraging both lowand high-level information. However, without additional guidance, the semantic consistency of the learned tokens may still degrade. Finally, we introduce a pre-trained feature reconstruction objective, which significantly boosts both image reconstruction and generation quality. This objective encourages alignment with semantic features from the frozen VFM and effectively balances the contributions of low- and high-level features to the contextual tokens—thereby preserving semantic fidelity.

Table 4: Ablation study on VFMTok’s components.

With these three key components(1) deformable attention for regionadaptive tokenization to reduce redundancy, (2) multi-level features for enhanced reconstruction, and (3) feature reconstruction loss for semantic alignment—VFMTok produces compact, semantically rich, and efficient tokens. Using only 256 tokens, VFMTok outperforms its VQGAN baseline with 576 tokens in reconstruction quality, generative performance, and semantic representation. Supplemental ablations are discussed in the Appendix.

Image Recon. Usage AR Gen. L.P.

Setup

#Tok. rFID↓ rIS↑ QC ↑ gFID↓ gIS↑ (%) VQGAN

0.95 197.3 99.7% 3.71 228.3 23.1 + Frozen VFM 0.99 206.3 100% 3.69 267.5 56.4

576

+ Region Adapt.

3.98 241.6 41.5 + Multi-level Feat. 0.92 199.5 3.71 251.1 22.7 + Reconstruct Feat. 0.89 215.4 3.42 277.3 69.4

1.20 199.0

256

100%

- Frozen VFM 256 0.95 196.3 100% 3.73 248.7 59.1

Convergence and efficiency analysis. Beyond above analysis, we experiment VFMTok with a randomly initialized encoder instead of a pre-trained VFM with other components remaining unchanged. As shown in Tab. 4 (last row), its reconstruction quality dropped to the level of VQGAN. Meanwhile, both its semantic representation capability and generation performance also decreased. This indicates a frozen VFM benefits tokenizer training as it provides a latent space advantageous for image reconstruction and generation. Besides, those semantic-rich, structured latent tokens accelerate AR model training convergence. As evidenced in Fig. 1(b), VFMTok enables AR models to achieve a 3× speedup in convergence compared to VQGAN. Moreover, an AR model’s generation time is quadratically proportional to the number of tokens. At the same resolution, VFMTok uses approximately half the tokens for image representation compared to counterparts like DINOv2VQGAN and CLIP-VQGAN. Consequently, VFMTok achieves a 4× generation speedup over these counterparts depicted in Tab. 1. This acceleration can be further enhanced with CFG-free generation.

### 5 Conclusion

In this work, we have demonstrated that frozen pre-trained vision foundation models (VFMs)ranging from self-supervised to language-supervised– are sufficient for high-quality image reconstruction and generation. To fully exploit their potential while addressing the redundancy inherent in 2D feature grids, we introduce VFMTok, a novel image tokenizer that incorporates region-adaptive tokenization to enhance token efficiency. By reducing feature redundancy, integrating multi-level feature representations, and introducing a semantic-preserving feature reconstruction objective, VFMTok yields a compact and semantically rich latent space. This facilitates high-quality image reconstruction and generation, accelerates convergence in autoregressive (AR) models, and enables efficient, high-fidelity, classifier-free (CFG-free) image synthesis—without the need for additional heuristics. Furthermore, the reduced number of tokens significantly lowers the computational cost of AR inference, making the approach both scalable and effective. Looking forward, the rich semantic structure of the learned latent space offers exciting potential for extending this work toward unified visual generation and understanding.

### 6 Acknowledgments

This work has been supported by the National Key R&D Program of China (Grant No. 2022YFB3608300), Hong Kong Research Grant Council - Early Career Scheme (Grant No. 27209621), General Research Fund Scheme (Grant No. 17202422, 17212923, 17215025) Themebased Research (Grant No. T45-701/22-R) and Shenzhen Science and Technology Innovation Commission (SGDX20220530111405040). Part of the described research work is conducted in the JC STEM Lab of Robotics for Soft Materials funded by The Hong Kong Jockey Club Charities Trust. We are deeply grateful to Lufan Ma for the contribution in polishing up this paper. We would also appreciate Tong Yang for providing the DINO discriminator script.

### 7 Author Contribution Statement

X.Q. proposed the initial concept of region-adaptive quantization. Based on this, A.Z. built the VFMTok, conducted the overall experiments, and led the writing of the initial draft. X.W., X.Q., and C.M. were deeply involved in the project progress and manuscript writing. X(iangyu).Z., X(uanyang).Z., G.Y., and T.W., provided sufficient computational resources. X(uanyang).Z. joint discussion where suggested ablation studies along with T.W., and discussed the writing of the draft. All authors contributed critical feedback, shaping the research, analysis, and the final manuscript.

### References

- [1] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, O˘guzhan Fatih Kar, Elmira Amirloo, Alaaeldin El-Nouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. arXiv preprint arXiv:2502.13967, 2025.
- [2] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021.
- [4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022.
- [5] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020.
- [6] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.
- [7] Xuangeng Chu, Anlin Zheng, Xiangyu Zhang, and Jian Sun. Detection in crowded scenes: One proposal, multiple predictions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12214–12223, 2020.
- [8] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proceedings of the IEEE International Conference on Computer Vision, pages 764–773, 2017.
- [9] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023.
- [10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.
- [11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, 2019.

- [12] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12873–12883, 2021.
- [14] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.
- [15] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022.
- [16] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. arXiv preprint arXiv:1911.05722, 2019.
- [17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems, 30, 2017.
- [19] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1125–1134, 2017.
- [20] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021.
- [21] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [22] Alina Kuznetsova, Mohamad Hassan Mohamad Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020.
- [23] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems, 32, 2019.
- [24] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022.
- [25] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024.
- [26] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024.
- [27] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017.
- [28] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision, pages 2980–2988, 2017.
- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

- [30] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An opensource project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.
- [31] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.
- [32] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023.
- [33] Keita Miwa, Kento Sasaki, Hidehisa Arai, Tsubasa Takahashi, and Yu Yamaguchi. One-d-piece: Image tokenizer meets quality-controllable compression. arXiv preprint arXiv:2501.10064, 2025.
- [34] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [36] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024.
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.
- [38] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training, 2018.
- [39] Ali Razavi, Aäron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in Neural Information Processing Systems, 32, 2019.
- [40] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in Neural Information Processing Systems, 29, 2016.
- [41] Wei Song, Yuran Wang, Zijia Song, Yadong Li, Haoze Sun, Weipeng Chen, Zenan Zhou, Jianhua Xu, Jiaqi Wang, and Kaicheng Yu. Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324, 2025.
- [42] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [43] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [44] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024.
- [45] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [46] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.
- [47] Aäron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelcnn decoders. Advances in Neural Information Processing Systems, 29, 2016.

- [48] Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. Advances in Neural Information Processing Systems, 30, 2017.
- [49] Xin Wen, Bingchen Zhao, Ismail Elezi, Jiankang Deng, and Xiaojuan Qi. “Principal components” enable a new language of images. arXiv preprint arXiv:2503.08685, 2025.
- [50] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [51] Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster diffusion transformers training without architecture modification. Advances in Neural Information Processing Systems, 37:56166–56189, 2024.
- [52] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.
- [53] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022.
- [54] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459– 10469, 2023.
- [55] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Randomized autoregressive visual generation. arxiv, 2024.
- [56] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. arXiv preprint arXiv:2406.07550, 2024.
- [57] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.
- [58] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [59] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.
- [60] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022.
- [61] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023.
- [62] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. International Conference on Learning Representations (ICLR), 2022.
- [63] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. arXiv preprint arXiv:2406.11837, 2024.
- [64] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.
- [65] Yongxin Zhu, Bocheng Li, Hang Zhang, Xin Li, Linli Xu, and Lidong Bing. Stabilize the latent space for image autoregressive modeling: A unified perspective. arXiv preprint arXiv:2410.12490, 2024.

### Appendix

This section first illustrates the implementation of VFMTok and AR generation models, covering their learning rates, optimization approaches, and training demands. Subsequently, we present more ablation studies on VFMTok, investigating the effects of different architectural designs on image reconstruction and generation. Besides, all VFMToks in this study utilize a frozen, pretrained DINOv2-L [9] as their encoder. Next, we show AR image generation with the other VFMs. Subsequently, we answer the question of what makes VFMs a good visual tokenizer. Finally, we demonstrate additional visualization samples of class-to-image generation using VFMTok.

### A VFMTok Implementation

Tokenizer training. VFMTok is trained on the ImageNet [10] training set at 336 × 336 resolution with random crop augmentation. All models share identical training settings: a constant learning rate of 10−4, AdamW optimizer [29] (β1 = 0.9,β2 = 0.95, weight decay = 0.05), a batch size of 256, and 50 training epochs. For training losses, the commitment loss weight is 0.25 and the adversarial loss weight is 0.5, with the adversarial loss activated after 20,000 iterations. Besides, VFMTok requires 1.5 days of training on 16 Nvidia H800 GPUs.

AR model optimization. The AR model training configuration aligns with LlamaGen’s [43], except our training resolution is 336×336 and the duration depends on model parameters. Other key settings include: a base learning rate of 10−4 per 256 batch size; AdamW optimizer [29] (β1 = 0.9, β2 = 0.95, weight decay = 0.05, gradient clipping of 1.0); a dropout rate of 0.1 for input token embeddings, attention, and FFN modules; and a 0.1 class condition embedding dropout for classifier-free guidance. Besides, A VFMTok-L requires 19.4 hours of training on 8 NVIDIA H800 GPUs.

### B Supplemental Ablation Study

In this subsection, we conduct more ablation studies on the design of our approach, including the AR image generation with resolution of 256 × 256, the effect of single-level v.s. multi-level features, the effect of shared ViT v.s. unshared ViT, the impact of different components, and the number of tokens to represent an image. Additionally, the effect of different VFMs on image reconstruction and generation is also presented.

#### B.1 AR Generation with Resolution of 256×256

In the main paper, given that the input resolution of the vision foundation model is 336 × 336, we adjust the resolution of reconstructed and generated images to 336 × 336 by default, thus avoiding changing the number of tokens for image representation. Following the common practices [43, 44], we also train the image tokenizer and the AR generation model with the resolution of 256 × 256, respectively. We first initialize and train VFMTok tokenizer for 50 epochs, then integrate it with AR generative models. Considering computational costs, models with fewer parameters like VFMTok-B and VFMTok-L are trained for 300 epochs, while larger AR models for 50 epochs. This setting aligns with LlamaGen [43] for 256 × 256 image generation. Furthermore, to ensure a fair comparison with the advanced autoregressive generation framework, RAR [55], we also incorporated VFMTok into RAR [55], maintaining the same setup during the training phase. As shown in Tab. 5, VFMTok not only achieves a decent reconstruction performance but also improves the generation quality compared to its counterparts [43, 55] by a large margin. It is worth noting that VFMTok also accelerates the convergence speed during AR model training and significantly improves synthesis quality.

#### B.2 Effect of Single-level v.s. Multi-level Features

In this paragraph, we investigate the separate impacts of each single-level or multi-level features on image reconstruction and generation. Specifically, we begin by conducting an ablation study to evaluate the impact of each single-layer feature on image reconstruction and generation. Subsequently, we cumulatively add each feature layer to observe the combined effect.

Setup. We initialize and train each tokenizer for 50 epochs. Once the tokenizer is optimized, it is integrated with an AR generation model, VFMTok-L, for a total of 100 training epochs. During evaluation, the image reconstruction and generation performance are reported.

Observation. The results, as presented in the Table 6 and Table 7, show that while a single feature layer offers no significant advantage, the quality of both image reconstruction and generation markedly improves as more layers are incorporated.

- Table 5: VFMTok performs image reconstruction and AR generation with the size of 256 × 256. Approach

Image recon. code usage↑ AR gen.

| |#Toks|rFID↓ rIS↑<br><br>|QC QP|#Epochs<br><br>|Para.|gFID↓ gIS↑<br><br>|
|---|---|---|---|---|---|---|
|LlamaGen-B LlamaGen-L LlamaGen-XL LlamaGen-XXL LlamaGen-3B|256|2.22 169.8<br><br>|– 95.2%<br><br>|300<br><br>|111M 343M 775M 1.4B 3.1B<br><br>|5.46 193.6 3.81 248.3 3.39 227.1 3.09 253.6 3.06 279.7|
| | | | |50| | |

RAR-L [55]

256 2.12 171.4 – 100.0% 400

461M 1.70 299.5 RAR-XL [55] 955M 1.50 306.9 RAR-XXL [55] 1.5B 1.48 326.0

VFMTok-B

256 1.02 213.2 100.0% –

100

111M 3.95 248.4 VFMTok-L 343M 3.02 271.6 VFMTok-B

300

111M 3.61 247.6 VFMTok-L 343M 2.79 276.0 VFMTok-XL

50

775M 2.79 277.1 VFMTok-XXL 1.4B 2.62 279.7 VFMTok-2B 2B 2.64 284.0

VFMTok(RAR-L)

256 0.88 216.2 – 100.0% 400

461M 1.47 316.2 VFMTok(RAR-XL) 955M 1.38 303.3 VFMTok(RAR-XXL) 1.5B 1.30 300.0

- Table 6: Performance of each single-level feature. Fi represents the indexed feature level.

Table 7: Performance of cumulatively added features. Fi represents the indexed feature level.

Image recon. AR gen. #Toks rFID↓ rIS↑ QC gFID↓ gIS↑

Image recon. AR gen.

Fi

Fi

#Toks rFID↓ rIS↑ QC gFID↓ gIS↑ F1

1.04 186.4

3.84 257.9

- F1

256

1.04 186.4

100.0%

3.84 257.9

- F2 0.95 200.6 3.79 272.7

- F3 1.03 208.5 3.69 274.9

- F4 1.23 214.8 3.64 277.7

- +F2 0.94 205.0 3.69 274.4

- +F3 0.94 210.8 3.27 272.5

- +F4 0.89 215.4 3.09 274.2

256

100.0%

- B.3 Effect of Shared ViT v.s. Unshared ViT Table 8: Effect of shared ViT v.s. unshared ViT.

Type

Image recon.

QC

AR gen.

L.P. #Toks rFID↓ rIS↑ #E gFID↓ gIS↑

unshared ViT

256 shared ViT

|0.91 214.9|100.0%<br><br>|50|3.52 277.4<br><br>|61.1|
|---|---|---|---|---|
|0.89 215.4<br><br>|100.0%<br><br>|50|3.42 277.3|69.4|

In this work, VFMTok utilizes a shared ViT to generate latent features for pixel rendering and highlevel VFM feature (specifically, from the last layer) reconstruction, respectively. However, it is uncertain if the sharing parameter is optimal. To this end, we experimented with another unshared ViT of the same architecture to generate the high-level VFM feature. Following the training setup, we train the tokenizer and AR model – VFMTok-L for 50 epochs. As shown in Tab. 8, the shared ViT ensures a better image reconstruction and synthesis quality with enriched semantics. Therefore, we utilize shared ViT in the VFMTok design by default.

- B.4 Effect of Different Components

In VFMTok, we introduce three key features: a frozen foundation (e.g., DINOv2 [9]) model as the encoder, multi-level plain feature interaction, and reconstructing the pre-trained feature, to achieve a notable performance in image reconstruction and generation. To achieve this, we incorporate several learnable modules into VFMTok. Namely, the mask tokens MI and Mf for image and its pre-trained feature reconstruction, the deformable attention layer adopted in the deformable transformer, and a set of learnable register tokens to address the potential artifacts in the latent feature. Thus, some questions naturally emerge: 1) Whether the mask token could be shared between MI and Mf; 2) Could the deformable attention layer be replaced with the vanilla cross-attention layer; 3) Can the register tokens be discarded from VFMTok?

Setup. To answer the above questions, we conducted 3 different experiments: 1) MI and Mf share the same mask token Mshare; 2) Replacing deformable attention with cross-attention in deformable transformer. To address the memory, computational demands, and fairness considerations of crossattention, we first concatenate multi-level features from a VFM along the channel dimension, then apply a single MLP for dimensionality reduction before these features are fed to the cross-attention transformer for interaction with queries. Besides, each query interacts with VFM features within a 16×16 window to simulate region-adaptive behavior; 3) Removing the register tokens from VFMTok. Additionally, linear probing is carried out on the [CLS] token to estimate the semantic representation capability of VFMTok.

Observation. As shown in Tab. 9, unshared mask tokens seldom affect image reconstruction or generation quality but significantly degrade VFMTok’s overall semantic representation. This indicates that image reconstruction requires a certain amount of semantic information, but this semantic information is not as strong as that requested for VFM’s feature reconstruction. Besides, Tab. 9 also reveals that shared mask tokens can enhance the semantic representation of VFMTok, potentially benefiting downstream comprehension tasks. Hence, we use shared mask tokens by default in VFMTok. Introducing cross-attention for learnable queries and multi-level feature interaction shows an evident decline in image reconstruction, generation, and semantic level. Compared to deformable attention that focuses on local regions, using cross-attention may introduce redundant information, this redundancy weakens the overall semantic representation by complicating quantized visual token distribution, thereby hindering image generation. Additionally, a slight decrease in image reconstruction, generation, as well as the semantic representation is also observed when register tokens Tokreg are removed from VFMTok. This suggests that introducing register tokens into VFMTok is reasonable since they can eliminate some artifacts in the features.

Table 9: Impact study on mask embeddings, attention, and registered tokens. Mshared

Attention

Image Recon. AR gen.

Tokreg

L.P. Cross-Attn Deform-Attn #Toks rFID↓ rIS↑ gFID↓ gIS↑

✓ ✓ ✓

1.00 211.5 3.89 271.5 34.1 ✓ ✓ 0.91 215.8 3.45 276.5 68.3 ✓ ✓ 0.89 216.1 3.50 276.0 64.7 ✓ ✓ ✓ 0.89 215.4 3.42 277.3 69.4

256

#### B.5 Effect of the Number of Tokens to Represent an Image

In this work, we utilize a set of region-adaptive tokens to represent an image. The number of these tokens is empirically fixed at 256 throughout our experiments. However, the impact of tokens’ cardinality on image reconstruction and generation quality remains unexplored.

Setup. To investigate this, we designed a controlled setup to isolate the impact of query quantity while maintaining architectural consistency across experiments. Specifically, we parameterize the image tokenizer with varying numbers of learnable queries. Each tokenizer variant is trained on the ImageNet [10] dataset for 50 epochs. Subsequently, the trained tokenizer is integrated into an AR image generation model, LlamaGen-L, and trained for 50 epochs. Both image reconstruction and generation quality are estimated on ImageNet [10] dataset with FID and IS, respectively.

Table 10: The effect of the number of tokens on image reconstructions and generation

Image recon. AR gen. rFID↓ rIS↑ gFID↓ gIS↑

#Tokens

36 2.61 175.4 3.93 222.4 64 2.09 188.3 3.59 250.5

100 1.44 204.4 3.54 270.8 121 1.36 202.5 3.59 267.1 144 1.20 204.6 3.46 274.9 169 1.11 212.4 3.42 275.5 196 1.08 213.7 3.32 271.2 225 0.96 215.5 3.45 279.2 256 0.89 215.4 3.42 277.3 289 0.85 217.2 3.41 272.3 361 0.80 218.8 3.64 277.9 400 0.79 221.3 3.36 272.3 441 0.73 221.8 3.61 275.5 576 0.60 222.8 3.57 278.8

Observation. As show in Tab. 10, image reconstruction exhibits a positive correlation with the tokens’ cardinality. As the number of tokens increases, metrics for estimating image reconstruction, namely rFID and rIS, both show gradual improvement. However, this trend does not hold for generation: as the number of tokens increases to higher values, there is no significant improvement in generation quality. Therefore, to balance generation quality with computational cost, and maintain fairness with vanilla counterparts [13, 43, 63], we fix the number of tokens at 256 across our experiments. Actually, it’s observed that 144 visual tokens suffice for representing

images in ImageNet [10]. This finding indicates VFMTok can further eliminate redundancy within image representations, yielding more compact and more effective image compression.

#### B.6 Effect of Different VFMs

Here, we explore employing different frozen VFMs [9, 46, 58] as the encoder of VFMTok, and incorporate them into different AR generative frameworks – LlamaGen [43] and RAR [55]. As shown in Tab. 11, VFMTok implemented with SigLIP [58] also consistently yields decent image reconstruction and generation quality based on AR models of different scales. Additionally, we also observe a potential correlation between the AR synthesis quality and different VFMs. VFM with stronger representation capabilities will achieve higher generation performance.

Table 11: Class-conditional image generation with different VFMs.

Generation w/ CFG Generation w/o CFG gFID sFID gIS Pre. Rec. gFID sFID gIS Pre. Rec.

Type Method #Epoch #Para. #Tok.

VFMTok-B(SigLIP)

3.53 5.76 254.4 0.85 0.51 3.75 5.80 156.3 0.79 0.58 VFMTok-L(SigLIP) 343M 2.61 5.54 272.1 0.84 0.56 2.11 5.65 214.0 0.81 0.61

111M

300

256

VFMTok-XXL(SigLIP)

1.4B 2.09 5.75 272.6 0.82 0.60 1.85 5.78 251.2 0.81 0.61

200

VFMTok-2B(SigLIP) 2.2B 2.05 5.77 271.4 0.82 0.61 1.92 5.78 260.5 0.82 0.61 VFMTok-XL(DINOv2)

2.41 5.53 276.8 0.83 0.59 2.10 5.54 258.1 0.82 0.60 VFMTok-XL(SigLIP) 2.42 5.70 275.1 0.83 0.59 1.99 5.75 241.9 0.81 0.61 VFMTok-XL(SigLIP2) 2.20 5.38 272.1 0.83 0.59 1.93 5.43 253.3 0.81 0.60

300 775M 256

AR.

1.46 6.18 312.9 0.78 0.64 2.26 5.78 204.3 0.79 0.62 RAR-XL(SigLIP) 955M 1.40 6.39 311.7 0.78 0.65 1.87 5.77 226.7 0.79 0.63 RAR-XXL(SigLIP) 1.5B 1.38 6.18 298.4 0.78 0.65 1.68 5.64 239.5 0.79 0.63

RAR-L(SigLIP)

461M

400

256

RAR-XL(SigLIP2)

1.50 6.37 292.7 0.77 0.66 2.05 5.50 217.7 0.79 0.62 RAR-XL(SigLIP2) 955M 1.46 6.02 288.4 0.78 0.65 1.72 5.41 244.8 0.80 0.63 RAR-XXL(SigLIP2) 955M 1.43 6.07 291.0 0.79 0.65 1.65 5.44 266.5 0.81 0.63

461M

400

256

#### B.7 What makes a VFM a good visual tokenizer?

To answer this problem, we first leverage existing vision foundation models, which were supervised with different learning objectives, e.g. masked image modeling in pixel or latent space (Pixel-MIM and Latent-MIM) and contrastive learning (C.L) – to construct VFMTok for image reconstruction and generation, and subsequently provide more insights on this question.

Setup. The encoder of vanilla VQGAN [43] is substituted with different vision foundation models (VFMs), which are supervised with distinct learning objectives. All of these tokenizers are trained on the ImageNet [10] training set for 50 epochs. Subsequently, these tokenizers are incorporated into an AR generative model – LlamaGen-L [43] and trained for 100 epochs. Both image reconstruction and generation quality are evaluated on the ImageNet validation set with FID and IS, respectively. Additionally, the top-1 accuracy of linear probing is also reported.

Observation As shown in the Table. 12, Pixel-MIM (MAE [15]) aligns best to reconstruction objectives, thus, VQGAN(MAE) achieves the optimal image reconstruction quality. However, it does not provide a feature space that is friendly for generation tasks. Its semantics is also inferior to the other VFMs. Contrastive learning-only models, VQGAN(CLIP) and VQGAN(SigLIP) achieve the best semantics (top-1 accuracy in linear probing), but modest image reconstruction and generation performance. Without a mask image modeling objective, they lack sufficient capability to model the local structures. Latent-MIM and contrastive learning co-optimized approaches, VQGAN(SigLIP2), and VQGAN(DINOv2) achieve the best overall performance.

Discussion The training objectives of vision foundation models(VFMs) affect the image reconstruction and generation of VFMTok. CLIP [37] and SigLIP [58] apply contrastive learning to image-text pairs, whereas DINO [3] applies a contrastive learning-like clustering objective to images. MAE [15] performs mask image modeling on image patches in pixel space, and iBOT [62] predicts DINO-like cluster assignments of image patches in latent space. DINOv2 [9] basically is the combination of DINO [3] and iBOT [62], and SigLIP2 [46] is also essentially the integration of SigLIP [58] and iBOT [62] (termed TIPS loss in their paper), omitting other regularizing losses. We hereby provide a holistic comparison of the training objectives of different VFMs in Table 13.

Table 12: Image reconstruction and generation with VFM of a distinct learning objective.

Tokenizer Pixel-MIM Latent-MIM C.L. #tok. rFID↓ rIS↑ gFID ↓ gIS↑ L.P.(%)↑ VQGAN(CLIP [37]) ✗ ✗ ✓

1.47 182.0 3.45 221.2 59.5 VQGAN(SigLIP [58]) ✗ ✗ ✓ 1.26 190.8 3.50 246.1 60.3 VQGAN(SigLIP2 [46]) ✗ ✓ ✓ 0.96 198.4 3.39 267.8 55.5 VQGAN(DINOv2 [9]) ✗ ✓ ✓ 0.99 206.3 3.34 268.6 56.4 VGQGAN(MAE [15]) ✓ ✗ ✗ 0.67 207.6 3.40 265.5 39.0

576

Notably, iBOT [62], DINOv2 [9], and SigLIP2 [46] are performing codebook learning resemble that in VQGAN on image patches, despite codes being soft instead of one-hot, and their codebook vectors being high-dimensional (e.g., 256). Following DINO [3], they learn a set of 8,192 or 65,536 prototypes, and require two augmented versions of the same patch (or image) to have identical prototype assignments (soft codes). They also predict the soft codes of masked patches, thus performing mask image modeling(Latent-MIM) in latent space. This formulation is termed self-distillation in DINO [3], and then extended to patches with Laent-MIM by iBOT [62], which was subsequently inherited by DINOv2 [9] and SigLIP2 [46].

Table 13: The category of learning objectives adopted in VFMs.

VFM P-MIM L-MIM C.L.

CLP [37] ✗ ✗ ✓ DINO [3] ✗ ✗ ✓ SigLIP [58] ✗ ✗ ✓ MAE [15] ✓ ✗ ✗ iBOT [62] ✗ ✓ ✗ SigLIP2 [46] ✗ ✓ ✓ DINOv2 [9] ✗ ✓ ✓

The connection between Latent-MIM and VQGAN makes it natural to convert these VFMs to tokenizers and expect decent image reconstruction and generation performance. Additionally, the global (image-level) contrastive learning objectives of DINOv2 [9] and SigLIP2 [46] also promise better high-level semantics (better performance on understanding tasks).

To summarize, the conclusion can be drawn as: masked image modeling(MIM) objectives primarily help reconstruction and generation. MIM in pixel space helps reconstruction more, but is less beneficial for generation than MIM in latent space. Contrastive learning(C.L.) objectives are less helpful for reconstruction and generation, but are important for understanding abilities (e.g., top-1 accuracy on ImageNet [10]). Best VFMs for visual tokenization are trained with both mask image modeling in latent space and contrastive objectives, namely DINOv2 [9] and SigLIP2 [9].

### C Limitations

Beyond the decent improvement achieved by VFMTok the optimal architectural design of VFMTok and its scalability on large-scale datasets are still under exploration. Currently, VFMTok’s image reconstruction is not yet optimal, primarily due to limitations in its codebook design and model size. For fair comparison and maintaining a concise design, we adopt a vanilla codebook and keep VFMTok’s size comparable to VQGAN’s [43]. While advanced codebook designs and increased model size could further enhance reconstruction quality. Beyond architecture design, exploring the scalability of VFMTok on large-scale datasets is also crucial for developing a superior tokenizer and advancing towards unified generation and understanding tasks. These aspects, however, outline promising directions for our future research.

### D Broader Impacts

The advancements in image tokenizers and autoregressive (AR) image generation present significant broader impacts. Positively, these technologies can democratize content creation across art, design, and media, accelerate scientific research through enhanced data augmentation and visualization, and enable novel forms of personalized digital experiences. However, this transformative potential is accompanied by substantial ethical challenges. Key concerns include the proliferation of realistic misinformation (deepfakes), the potential for misuse in creating non-consensual or harmful content, the amplification of societal biases embedded in training data, and complex issues surrounding intellectual property and copyright. Furthermore, the computational resources required for training state-of-the-art models raise environmental considerations. Therefore, the responsible development and deployment of these powerful tools necessitate robust frameworks for ethical guidelines, bias detection and mitigation, provenance tracking, and public awareness to harness their benefits while minimizing societal risks.

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

##### Figure 3: Class-conditional image generation with CFG.

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

##### Figure 4: Class-conditional image generation without CFG.

