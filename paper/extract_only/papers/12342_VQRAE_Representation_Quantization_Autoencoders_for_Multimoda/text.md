# arXiv:2511.23386v1[cs.CV]28Nov2025

## VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction

Sinan Du1,3∗‡, Jiahao Guo2,3∗‡, Bo Li3 , Shuhao Cui3, Zhengzhuo Xu1, Yifu Luo1, Yongxian Wei1 Kun Gai3, Xinggang Wang2, Kai Wu3†, Chun Yuan1

1Tsinghua University, 2Huazhong University of Science and Technology (HUST), 3Kolors Team, Kuaishou Technology

##### Abstract

Unifying multimodal understanding, generation and reconstruction representation in a single tokenizer remains a key challenge in building unified models. Previous research predominantly attempts to address this in a dual encoder paradigm, e.g., utilizing the separate encoders for understanding and generation respectively or balancing semantic representations and low-level features with contrastive loss. In this paper, we propose VQRAE, a Vector Quantization version of Representation AutoEncoders, which pioneers the first exploration in unified representation to produce Continuous semantic features for image understanding and Discrete tokens for visual generation within a unified tokenizer. Specifically, we build upon pretrained vision foundation models with a symmetric ViT decoder and adopt a two-stage training strategy: first, it freezes the encoder and learns a high-dimensional semantic VQ codebook with pixel reconstruction objective; then jointly optimizes the encoder with self-distillation constraints. This design enables negligible semantic information for maintaining the ability of multimodal understanding, discrete tokens that are compatible for generation and fine-grained reconstruction. Besides, we identify the intriguing property in quantizing semantic encoders that rely on high-dimensional codebook in contrast to the previous common practice of lowdimensional codebook in image reconstruction. The semantic VQ codebook can achieve a 100% utilization ratio at a dimension of 1536. VQRAE presents competitive performance on several benchmarks of visual understanding, generation and reconstruction with promising scaling property in the autoregressive paradigm for its discrete merits.

##### 1. Introduction

The advent of GPT-4o [44] indicates the significant potential of Multimodal Large Language Models (MLLMs) [63, 69] to unify visual understanding and generation within a single augoregressive architecture [11, 17, 26, 80, 97].

∗ Equal Contibution. † Project Lead. Corresponding Authors. ‡ Work done during internship in Kolors Team, Kuaishou Technology.

[Figure 1]

(a) (b) (c)

Figure 1. Comparions of different unified tokenizers. (a) Janus [7, 76] series adopt dual-encoder paradigm. (b) QLIP [95] and UniTok [41] supervise dicrete tokens with CLIP loss. (c) Our VQRAE can produce continuous and discrete tokens for different tasks.

These unified models deliver precise response in multimodal interaction [11, 26, 44], emergent in-context reasoning properties [12, 24] and synergistic benefit across tasks [81, 86]. However, a fundamental dilemma remains in selecting visual tokenizers to obtain appropriate representations to achieve a trade-off between understanding, generation, and reconstruction tasks.

Discrete tokenizers [15, 66] were widely adopted in early unified models [36, 60, 70, 80] due to their compatibility with the next token prediction (NTP) paradigm, scalability potential, and training efficiency enabled by highly optimized AI infrastructure. However, discrete tokenizers trained with reconstruction objective (pixel-level) tend to produce fine-grained details, which conflict with the semantic-level features required for visual understanding tasks such as CLIP [50, 65, 93], leading to performance degradation. Dual encoder methods attempt to address this by utilizing separate encoders [7, 43, 76] for multimodal understanding and visual generation, or balancing semantic representations and low-level features with contrastive loss. These dual encoder approaches increase model complexity [49, 82], hinder deeper interaction between representations [7, 43, 76], and demand an immense batch size to balance loss conflicts effectively [41, 78, 95]. Can we design a unified tokenizer to produce continuous semantic features and discrete fine-grained tokens simultaneously?

Inspired by the practice of RAE [96] in generative modeling, which replaces the VAE [28] with pretrained vision encoders [2, 93] paired with trained decoders and demonstrates the structured semantic space benefits convergence

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

[Figure 13]

###### ReconstructionUnderstandingGeneration

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

User:

User:

User:

Analyze what type of climate predominates in the orange-red areas of the image.

Describe the content of this image in detail.

Tel me why are these grapes yelow?

The image features a majestic deer with large antlers standing in a snowy field. The deer is positioned towards the center of the image, surrounded by a cold and snow-covered environment. The scene appears to be set in a forest, as the deer is standing in front of a tree. The overall atmosphere of the image is serene and captures the beauty of the animal in its natural habitat.

The grapes in the image are yellow because they are unripe. As grapes ripen, they typically change color from green to yellow, and eventually to red, depending on the variety. In this case, the grapes are still in the early stages of ripening, which is why they appear yellow. The image shows a bunch of green grapes hanging from a tree, indicating that they are not yet fully ripe and ready for consumption.

The orange-red areas in the image, which represent the African continent, are predominantly characterized by a tropical climate. Tropical climates are typically warm and humid, with high temperatures and abundant rainfall throughout the year. These regions are known for their lush vegetation, diverse wildlife, and rich biodiversity.

Figure 2. Showcase of the visual understanding, generation and reconstruction ability of our VQRAE model.

of diffusion transformers [46]. We propose VQRAE, a vector quantization version of RAE [96], which pioneers the unified tokenizer to produce continuous features for visual understanding and discrete tokens for generation and reconstruction, effectively reducing the complexity of dual encoder design and reliance on convolutional pixel encoders. VQRAE comprises pretrained vision foundation models (VFMs), a high-dimensional VQ codebook and a symmetric ViT decoder. We adopt a two-stage training strategy: in the first stage, built upon the frozen VFMs, we jointly optimize the codebook to learn discrete semantic representation and the ViT decoder with pixel reconstruction objective; the second stage unfreezes the VFMs with self-distillation loss to maintain semantic features and reconstruction loss to supplement fine-grained details. Besides, to our knowledge, we are the first work to train a 100% utilization ratio semantic codebook for reconstruction at a high dimension (e.g., 1536), while previous studies mainly at a low dimension (e.g., 8-256) [15, 53, 57, 99].

Our key contributions are summarized as follows:

- • We propose VQRAE, a VQ version of RAE, which is the first work to train a 100% utilization ratio semantic codebook at a high dimension for reconstruction.
- • VQRAE pioneers the unified tokenizer without convolution blocks, producing continuous features and discrete tokens simultaneously for understanding, generation and reconstruction under the AR-only paradigm.
- • Extensive experiments on downstream benchmarks present competitive performance and scaling promise, opening new avenues for exploring unified models.

##### 2. Related Work

###### 2.1. Visual Tokenizer for Generation

Visual tokenizer plays a crucial role in encoding raw pixels into compact latent representations for generative modeling. Inspired by the success of next token prediction (NTP) paradigm, vector quantization (VQ) tokenizer [15, 53, 66, 99, 100] is widely explored for its discrete property and compatibility with autoregressive and masked generative models [3, 51, 57, 64, 88]. However, discrete methods such as Chameleon [60], EMU-3 [70], and Show-o [80] suffer performance degradation in visual understanding tasks due to quantization errors [36, 67, 72]. In addition, the VQ tokenizer optimized with the reconstruction objective tends to provide pixel-level tokens for fine-grained features, which introduce significant alignment costs with LLMs.

###### 2.2. Unified Tokenizer

To address the representation dilemma (continuous vs. discrete, pixel vs. semantic) above, Janus series [7, 43, 76] propose to disentangle representations for understanding and generation with separate visual encoder. TokenFlow [49]

and MUSE-VL [82] adopt a dual encoder training paradigm and decouple semantic and pixel-level feature learning while maintaining their alignment via a shared mapping mechanism. QLIP [95], VILA-U [78] and UniTok [41] supervise the latent features from vision foundation models [50, 93] with contrastive learning loss [50], which demand an immense batch size to train effectively and balance loss conflicts. However, this dual encoder paradigm leads to training inefficiency and hinders the deeper alignment and interaction between different representations, which is critical for unified models [11, 12, 24, 33, 45, 59, 61, 68, 73, 77, 81]. Diffusion-based tokenizers [6, 58, 92] typically employ continuous representations for reconstruction, which are difficult to converge in the autoregressive paradigms due to the high dimensional CLIP feature [31, 50]. Tar [21] and X-Omni [19] propose VQ tokenizers trained with semantic supervision but discarded reconstruction capabilities, thus precluding their nature as autoencoders. In contrast, VQRAE achieves a superior trade-off between visual understanding and reconstruction and keeps its discrete merits for autoregressive generation.

##### 3. Method

Our VQRAE is a vector quantization version of representation autoencoder designed to achieve unified multimodal understanding, generation and reconstruction. In Sec. 3.1, we conduct pilot experiments to clarify the motivation behind our design and highlight the distinctive features of VQRAE compared to other approaches. In Sec. 3.2, we provide a detailed description of the VQRAE architecture, which adopts a symmetric ViT encoder-decoder structure and a two-stage training strategy to optimize the discrete codebook. In Sec. 3.3, we train a high-dimensional codebook when quantizing VFMs, which contrasts with previous findings that low-dimensional codebook is crucial for reconstruction and generation [15, 53, 57]. In Sec. 3.4 and 3.5, we train understanding and generation with VQRAE.

###### 3.1. Pilot Experiments

###### # Exp. Method Und. Gen. Type rFID↓ PSNR↑ SSIM↑ MME-P↑ SEED↑ TQA↑

- 1 VQGAN† [15] D D single 4.98 20.00 0.629 756.1 38.2 46.8

- 2 VQKD† [47] D - single - - - 1252.4 57.8 48.2

- 3 RAE† [96] C C single 0.49 19.23 0.620 1544.3 70.0 61.7

- 4 TokenFlow† [49] D D dual 1.37 21.41 0.690 1365.4 62.6 54.1

- 5 Janus [76] C D dual 2.19 20.79 0.675 1338.0 63.7 -

- 6 VQRAE† (ours) C D single 1.31 22.23 0.762 1543.3 70.0 61.7

Table 1. Comparisons of various methods in multimodal understanding and reconstruction. “Und.” and “Gen.” refer to Understanding and Generation. “C” and “D” indicate Continuous and Discrete representations used for specific tasks. † denotes training on LLaVA-v1.5 [37] setting. Reconstruction quality is evaluated on the 256 × 256 ImageNet 50k validation set. MMEP: MME-Perception [18]; SEED: SEEDBench-Img [29]; TQA: TextVQA [54]; TokenFlow: TokenFlow-L-13B [49].

[Figure 35]

Continuous token Discrete token Text token Mask token

VQ Codebook

Reconstruction Loss

ViTBlock

ViTBlock

Pixel Dec.

Pixel Dec.

(a)

Self-Distillation Loss

Semantic Enc.

Semantic Enc.

Stage 1 Stage 2

Understanding Generation

(c)

(b)

Figure 3. Illustration of our unified tokenizer VQRAE. (a) Our VQRAE is built on pretrained VFMs (e.g., SigLIP2 [65]), which can simultaneously produce continous semantic features for multimodal understanding tasks and discrete tokens for visual generation and reconstruction tasks. (b) Training pipeline of VQRAE. We adopt a two-stage training paradigm. In the first stage, the pretrained semantic encoder remains frozen, while a high-dimensional vector quantization codebook and a pixel decoder are trained using an image reconstruction loss. In the second stage, the encoder, codebook, and decoder are jointly optimized to achieve fine-grained reconstruction. Additionally, the encoder outputs are regularized via a self-distillation loss to maintain semantic understanding performance. (c) VQRAE achieves a superior trade-off with the unified encoder in the autoregressive style.

To further elucidate the design rationale behind VQRAE for achieving task-level trade-off within a unified tokenizer, we conduct pilot experiments utilizing the LLaVA-1.5 [37] settings for image understanding and the ImageNet dataset for reconstruction. In Tab. 1 Exp. 1-2, the training objective of pixel reconstruction yields fine-grained features that underperform in multimodal understanding tasks; Although distilling VFMs [47] for discrete tokens can narrow this performance gap, the resulting representations still lag behind those produced by continuous tokenizers, which are attributed to the quantization errors in discretization. TokenFlow [49] addresses this with semantic and pixel encoder sharing a mapping network, while Janus [76] directly utilizes separate encoders. In contrast, VQRAE, is capable of generating both continuous features for visual understanding and discrete tokens for generation and reconstruction within a single tokenizer. This differs from the dual encoder architecture employed by TokenFlow and Janus, and enables a more favorable trade-off as presented in Exp. 4-6.

###### 3.2. A Unified Tokenizer: VQRAE

As illustrated in Fig.3a, our VQRAE is composed of a unified tokenizer for both low-level and semantic encoding with well-pretrained VFMs, a high-dimensional semantic VQ codebook with 100% utilization ratio, and a symmetric ViT decoder for pixel reconstruction.

VFMs as Unified Encoder. Unlike prior works in dual encoder paradigm [49, 76, 82] that employ a seman-

tic encoder (ViT-based) and a pixel encoder (CNN-based) for separate representations, this approach introduces additional model complexity and training overhead, while also leading to a disparity in representation interaction. We adopt the pretrained VFMs (e.g., CLIP [50, 65, 98]) as unified encoder E for encoding visual information. We find that the continuous features generated by the frozen semantic encoder can be directly utilized for image reconstruction, albeit with some loss of detail—such as in color and texture as evidenced in [59, 96]. However, slight fine-tuning of the encoder can refine the representations to recover these missing details, while causing almost no degradation but even stronger in semantic understanding. Given an input image X ∈ Rh×w×3 and the unified encoder E with patch size p

hw

p2 ×d. These intermediate features are used for semantic quantization and multimodal understanding task.

and hidden size d, we obtain latent features ZI ∈ R

High Dimensional VQ. Vector Quantization [15, 66] is a well-studied technique used to translate continuous representations into a set of discrete tokens. Unlike previous discrete unified tokenizers [49, 82] built upon pixel features, we only exploit the semantic features from VFMs for quantization. Specifically, we utilize SimVQ [100] method with initialized VQ codebook C ∈ Rk×e = {ci}|ki=1 and learnable projection matrix W ∈ Re×e = {wi}|ei=1 for quantization, where n is the codebook size and e denotes the codebook dimension. Semantic features ZI from VFMs

hw

p2 ×e and then quantized vectors Zq ∈ R

are projected to Zˆc ∈ R

hw

p2 ×e are selected from codebook C according to the l2-norm distances:

||Zˆc −ciwi|| ,for i = 1,...,k (1)

Zq = lookup argmin

i

We highlight that the codebook in our VQRAE performs effectively in high-dimensional formulation, with its dimensionality necessarily matching at least that of the VFMs encoder E. This observation contrasts with previous studies, which suggest that codebooks for training reconstruction objectives should operate in low-dimensional spaces [15, 53, 57]. This discrepancy will be presented in our ablations.

Symmetric Decoder. We replace the previous CNN-like pixel decoder [15, 28, 59] with a ViT-based decoder that mirrors the encoder structure, thereby mapping latent features back to pixel space as in RAE [96], while our approach focuses on reconstructing images from discrete tokens. Specifically, quantized vectors Zq are projected to the bottleneck features Zbot ∈ R

hw p2 ×d to align with dimension of the symmetric decoder D. We simply set the patch size of the decoder as 1 and project the decoded features D(Zbot) from decoder to X′ ∈ R

p ×wqp′×3, where q and q′ are hyperparameters to adjust the resolution of the reconstructed images. We set q = q′ = p to keep the resolution constant.

hq

Two-Stage Training. In the original RAE [96] paper, VFMs encoder keeps frozen to maintain the structure of semantic features. However, VFMs are not meant to be optimized for fine-grained reconstruction and may cause blur in this setting. Our pilot experiments in Sec. 3.1 validate that appropriate finetuning can strengthen the reconstruction ability and VFMs encoder are robust to maintain the performance of visual understanding. As detailed in Fig. 3b, in the first stage, we freeze the VFMs encoder E and jointly optimize the VQ codebook C and decoder D with the pixel reconstruction and vector quantization objectives:

Lrec = ℓ2(X,X′) + LP(X,X′) + λGLG(X′) (2) Lquant = ||sg(C) − Zq||22 + β · ||Zq − sg(C)||22 (3)

Lstage1 = Lrec + Lquant (4) where ℓ2 represents pixel-wise reconstruction loss, LP(·) denotes perceptual loss using LPIPS, sg[·] denotes the stopgradient operation, and LG(·) represents adversarial loss with λG as its weight coefficient. β is set to 0.25 by default.

As depicted in Fig. 3b, in the second stage, we unfreeze the VFMs encoder with self-distillation loss constraints to augment reconstruction quality and maintain the performance of multimodal understanding. Note that, we directly supervise the continuous features ZI without quantization errors as in [21, 47]. The loss function in this stage is:

Ldistill = ||ZI − T(X)||22 (5)

Lstage2 = Lrec + Lquant + λdLdistill (6)

where T denotes the teacher model initialized from E and keeps frozen, while encoder, VQ codebook and decoder are jointly optimized. λd is the coeffienct of distillation loss.

###### 3.3. Semantic VQ with 100% Utilization Ratio.

Previous VQ methods (e.g., VQVAE [66] and VQGAN [15, 57]) typically train a discrete codebook at low dimensional entries (e.g., 8-256) with the pixel reconstruction objective with CNN extracted features. As discussed in [15, 53, 57], it remains a fundamental challenge to train a high dimensional codebook without collapse risks and with a high utilization ratio. RAE [96] illustrates that continuous high dimensional latent space is more structured and can facilitate the convergence of state-of-the-art generative models [42, 46, 90]. To our knowledge, our VQRAE represents the first attempt to successfully train a high dimensional codebook at a 100% utilization ratio (4k-16k entries). Contrary to the findings in previous CNN-based VQ codebook practices [15, 53, 57], our empirical results indicate that a semantic codebook generally requires a larger dimension; otherwise, it may lead to non-convergence in reconstruction training and codebook collapse.

###### 3.4. Multimodal Understanding with VQRAE

Different from previous works [21, 47] which utilize self-distillation loss to supervise the discrete tokens, our VQRAE can produce intermediate continuous features for image understanding without quantization errors. Furthermore, since our VQRAE is built upon pretrained VFMs, it can be seamlessly integrated into existing MLLMs, significantly reducing training overhead and the need for separate evaluation of the performance of the tokenizer. This approach eliminates the previous requirement to complete tokenizer and MLLM training before assessing the visual comprehension capabilities of the model. For instance, by employing the VFMs from existing MLLMs as the encoder, the tokenizer obtained through our two-stage training can be directly incorporated into MLLMs without requiring additional pretraining or supervised fine-tuning. Consequently, the MLLMs presented in this paper have not been specifically trained for the proposed unified tokenizer.

###### 3.5. Visual Generation with VQRAE

Although MAR [31, 61] methods demonstrate the ability of continuous autoregressive for generative modeling, our discrete VQ codebook is more elegant and compatible with highly optimized AI infrastructure for training acceleration. We leverage the text tokenizer for text encoding and VQRAE for image encoding. Building on the backbone of Qwen3 [87], we expand the vocabulary size for visual tokens and train the LLMs with NTP loss exclusively on

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

- (a)
- (b)

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

- Figure 4. We perform K-means clustering on the ImageNet-1K validation set using continuous features and discrete tokens. The visualization illustrates images grouped by (a) continuous features and (b) discrete tokens, both derived from our VQRAE. VQRAE is capable of producing discriminative features for multimodal understanding and discrete visual tokens for fine-grained reconstruction and generation simultaneously within a unified tokenizer. It indicates the redundancy in the dual-encoder paradigm.

visual tokens. We highlight the property of our disentangled representations with a unified tokenizer, as visualized in Fig. 4, the semantic features tend to cluster similar objects and animals, while fine-grained features cluster same textures. We use the VQRAE-InternViT version with codebook size of 16k and dimension of 1536.

##### 4. Experiments

###### 4.1. Experimental Setups

Datasets. VQRAE is pretrained on BLIP3-o [6] opensourced data, which consists of 27M samples recaptioned by Qwen2.5-VL-7B [1], 5M samples from CC12M [4], and 4M synthesized images from JourneyDB [56]. For image understanding, we follow the setups in LLaVA-1.5 [37], which uses LLaVA-Pretrain-595K and LLaVA-v1.5-mix665K for pretraining and SFT. For visual generation, we train it on BLIP3-o [6] data and additional 80M high quality images. We conduct ablation studies on VQ codebook on ImageNet-1K with 20 epochs training for efficiency.

Implementation Details. We implement VQRAE with SigLIP2-so400m-p16-256px [65], SigLIP2-so400m-p16512px [65] and InternViT-300M-448px [98] as unified encoders. Detailed hyperparameters are provided in the Appendix. For image understanding, we employ Vicunav1.5 [10] and Qwen2.5-7B [62] as LLM backbone. Note that, we did not conduct training specifically for our VQRAE as clarified in Sec. 3.4. For visual generation, we use Qwen3-0.6B [87] for efficient training.

Evaluation Metrics. We assess reconstruction quality using rFID, PSNR, and SSIM on the ImageNet-1K validation set. For multimodal understanding, we evaluate on MME-Perception [18], GQA [25], POPE [32], MMBench-en [38], SEEDBench-Img [29], MMMU [91], TextVQA [54] and AI2D [27]. For image generation, we evaluate on GenEval [20] and DPG-Bench [22].

Method Ratio rFID↓ PSNR↑ SSIM↑

Generative Only Tokenizer

VQGAN [15] 16 4.98 20.00 0.629 LlamaGen [57] 16 2.19 20.79 0.675

VAR [64] 16 1.00 22.63 0.755 Open-MAGVIT2 [40] 16 1.67 22.70 0.640

RAE [96] 16 0.49 19.23 0.620 Unified Tokenizer

Show-o [80] 16 3.50 21.34 0.590 TokenFlow [49] 16 1.37 21.41 0.690 DualViTok [23] 16 1.37 22.53 0.740 MUSE-VL [82] 16 2.26 20.14 0.646

VQRAE (SigLIP2) 16 1.31 22.23 0.762 VQRAE (InternViT) 14 1.39 22.88 0.784

Table 2. Comparisons of reconstruction quality on the 256 × 256 ImageNet 50k validation set. Ratio: downsample ratio.

###### 4.2. Unified Visual Tokenizers

As presented in Tab. 2, we evaluate reconstruction metrics on generative-only tokenizers and unified tokenizers. Our experiments demonstrate that employing pre-trained VFMs as unified encoders, ViT-based decoder, along with the discretization of semantic features, can achieve competitive reconstruction quality without any convolution blocks. This finding aligns with the observation in [59, 96] that continuous features generated by a semantic encoder can be utilized for reconstruction tasks, while our work further validates the feasibility of operating in a discrete space. In addition, our VQRAE surpasses dual-encoder methods such as TokenFlow and MUSE-VL with a more efficient unified encoder style. We provide visualizations results in Fig. 5.

###### 4.3. Multimodal Understanding

As shown in Tab. 3, our VQRAE consistently outperforms other unified tokenizers on downstream multimodal understanding benchmarks under the same training settings as LLaVA-1.5 [37], both on 7B and 13B scales. It can be observed that the semantic representations provided by other unified tokenizers often cause information loss, manifested as varying degrees of performance degradation compared to the LLaVA-1.5 baseline [37]. Moreover, our approach is more efficient, as it does not require multimodal alignment or instruction tuning for the pre-trained VQRAE tokenizers (as presented in Sec. 3.4). By simply replacing the ViT encoder in the base model (e.g., InternVL3 [98]), our method can be directly applied to downstream tasks without performance degradation—in fact, performance may even improve. This suggests that our two-stage training approach introduced in Sec. 3.2 effectively preserves understanding performance while training for image reconstruction. Our method yields a significant improvement on the MME-P [18] benchmark compared to the dual-encoder

###### Method Vision Encoder LLM Res. POPE GQA TQA MMB MME-P SEED MMMU AI2D

Understanding Only MLLM

Emu3-Chat [70] MoVQGAN 8B from scratch 512 85.2 60.3 64.7 58.5 1243.8 68.2 31.6 70.0 LLaVA-1.5† [37] CLIP-L Vicuna-7B 336 85.9 62.0 46.1 64.3 1510.7 58.6 35.4 55.3 LLaVA-1.5† [37] CLIP-L Vicuna-13B 336 85.9 63.3 61.3 67.7 1531.3 68.1 36.4 61.1

InternVL2.5 [8] InternViT-300M InternLM2.5-7B 448 90.6 - 79.1 84.6 - - 56.0 84.5 InternVL3 [98] InternViT-300M Qwen2.5-7B 448 91.1 - 80.2 83.4 1748.4 77.1 62.7 85.2

Qwen2.5-VL [1] QwenViT Qwen2.5-7B 448 85.9 - 84.9 83.5 1698.1 77.0 58.6 83.9

MLLM with Unified Tokenizer VILA-U† [78] SigLIP-so400m Vicuna-7B 256 81.6 - - - 1311.6 - - -

UniTok† [41] Vitamin-L Vicuna-7B 256 81.7 - - - 1448.0 - - SemHiTok† [9] SigLIP-L Vicuna-7B 256 84.2 61.0 - 60.3 1400.6 - - -

QLIP† [95] CLIP-L Vicuna-7B 392 86.1 61.8 55.2 - 1498.3 - - -

TokenFlow-L† [49] ViTamin-XL Vicuna-13B 256 85.0 60.3 54.1 60.3 1365.4 62.6 34.4 56.6 TokenFlow-XL [49] SigLIP-so400m Vicuna-13B 384 86.8 62.7 61.5 68.9 1545.9 68.7 38.7 66.7

TokLIP† [34] ViT-so400m Qwen2.5-7B 384 82.7 59.3 - - 1410.2 65.2 42.1 -

Tar [21] SigLIP2-so400m Qwen2.5-7B 384 87.8 61.3 - 74.4 1571.0 73.0 39.0 VQRAE† SigLIP2-so400m Vicuna-7B 256 84.4 62.4 44.4 65.3 1445.7 66.4 31.3 53.1 VQRAE† SigLIP2-so400m Vicuna-13B 256 85.1 63.4 46.5 65.5 1491.1 66.8 33.3 57.0 VQRAE† SigLIP2-so400m Vicuna-7B 512 88.2 63.6 58.8 67.6 1494.2 62.8 33.9 55.3 VQRAE† SigLIP2-so400m Vicuna-13B 512 88.2 64.8 61.7 67.3 1543.3 69.9 37.4 59.8

VQRAE InternViT-300M Qwen2.5-7B 448 90.5 - 80.6 85.1 1746.8 77.0 61.6 84.8

Table 3. Evaluation on multimodal understanding benchmarks. We collect evaluations including: POPE [32]; GQA [25]; TQA: TextVQA [54]; MMB: MMBench-En [38]; MME-P: MME-Perception [18]; SEED: SEEDBench-Img [29]; MMMU [91]; AI2D [27]. † denotes training on LLaVA-v1.5 [37] setting. “Res.” is an abbreviation of resolution. Our pretrained VQRAE can be directly comparable with SOTA open-sourced MLLMs without specific fine-tuning as detailed in Sec. 3.4.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

- Figure 5. Visualization of reconstruction results from VQRAEInternViT version. Left: input image; Right: output image.

method TokenFlow-L under the same setting of 13B parameters (1491.1 vs. 1365.4). Also, we surpass Tar [21] which performs semantic distillation directly on discrete tokens under the Qwen2.5-7B setting. It indicates the performance degradation on multimodal understanding due to quantization errors [15, 47].

###### 4.4. Visual Generation

We evaluate the performance of visual generation with our VQRAE on the GenEval [20] and DPG-Bench [22] benchmarks as presented in Tab. 4. Our lightweight visual generation models with only 0.6B parameters showcase competitive capabilities with models of similar parameter size. It indicates that semantic high dimensional latent space built on VFMs not only accelerates the convergence of diffusion-

based models for generative modeling [90, 96], but also benefits the training dynamics of the autoregressive models in discrete scaling paradigm.

###### 4.5. Ablation Studies

Codebook Dim. Previous studies [15, 53, 57] found that when quantizing features extracted from CNN-based encoders to learn a VQ codebook, a relatively small dimension (ranging from 8 to 256) should be used—a choice that has been interpreted as necessary for reconstruction requiring more fine-grained details. These approaches often encounter issues such as codebook collapse or a sharp decline in utilization when further scaling up to the typical dimensions of CLIP-based encoders (e.g., 1152). However, as shown in Tab. 5, it presents a contrary conclusion: when quantizing features extracted from VFMs (ViT-based), the dimension of the codebook should be higher; otherwise, it leads to training non-convergence and codebook collapse. In particular, our VQRAE maintains a high utilization ratio (nearly 100%) in a high dimension range.

Codebook Size. We also ablate the effect of codebook size on our unified tokenizer VQRAE in Tab. 5. We observe that the reconstruction quality continually improves with increasing codebook size. However, when codebook size exceeds 16K, we observe a slight degradation which is attributed to the slow convergence of the training process.

Training Strategies. We conduct ablation experiments on the training strategy of unifying understanding, generation

GenEval [20] DPG-Bench [22] Single Obj. Two Obj. Counting Colors Position Color Attri. Overall↑ Global Entity Attribute Relation Other Overall↑

Method # Params

Diffusion-based Model

SDv1.5 [52] 0.9B 0.97 0.38 0.35 0.76 0.04 0.06 0.43 74.63 74.23 75.39 73.49 67.81 63.18 PixArt-α [5] 0.6B 0.98 0.50 0.44 0.80 0.08 0.07 0.48 74.97 79.32 78.60 82.57 76.96 71.11 SDv2.1 [52] 0.9B 0.98 0.51 0.44 0.85 0.07 0.17 0.50 - - - - - -

SDXL [48] 2.6B 0.98 0.74 0.39 0.85 0.15 0.23 0.55 83.27 82.43 80.91 86.76 80.41 74.65 DALLE3 [35] - 0.96 0.87 0.47 0.83 0.43 0.45 0.67 90.97 89.61 88.39 90.58 89.83 83.50

SD3-Medium [16] 2B 0.99 0.94 0.72 0.89 0.33 0.60 0.74 87.90 91.01 88.83 80.70 88.68 84.08

SANA-1.5 [79] 4.8B 0.99 0.93 0.86 0.84 0.59 0.65 0.81 - - - - - 84.70 Autoregressive-based Model

Chameleon [60] 7B - - - - - - 0.39 - - - - - LlamaGen [57] 0.8B 0.71 0.34 0.21 0.58 0.07 0.04 0.32 81.76 75.43 76.17 84.76 58.40 64.84 EMU3-Gen [70] 8B 0.98 0.71 0.34 0.81 0.17 0.21 0.54 85.21 86.68 86.84 90.22 83.15 80.60 TokenFlow [49] 13B 0.97 0.66 0.40 0.84 0.17 0.26 0.55 78.72 79.22 81.29 85.22 71.20 73.38

Janus [76] 1.3B 0.97 0.68 0.30 0.84 0.46 0.42 0.61 82.33 87.38 87.70 85.46 86.41 79.68 SimpleAR [67] 1.5B - 0.90 - - 0.28 0.45 0.63 87.97 - - 86.33 - 81.97

Janus-Pro [7] 1B 0.98 0.82 0.51 0.89 0.65 0.56 0.73 87.58 88.63 88.17 88.98 88.30 82.63 VQRAE 0.6B 0.96 0.82 0.64 0.80 0.73 0.58 0.76 89.78 93.14 89.92 90.34 91.27 86.67

Table 4. Comparisons of visual generation quality on GenEval [20] and DPG-Bench [22]. Obj.: Object. Attri.: Attribute.

Dim Size rFID↓ PSNR↑ SSIM↑ Ratio↑ ≤ 256

NA NA NA NA

384 7.69 8.24 0.261 64% 768 5.38 13.76 0.398 69%

16384

1152 3.51 17.22 0.569 83% 1536 2.65 20.14 0.668 100% 1920 2.69 20.07 0.664 98%

4096 7.07 8.02 0.253 100% 8192 3.74 17.02 0.548 100% 16384 2.65 20.14 0.668 100% 32768 2.78 19.94 0.645 96%

1536

- Table 5. Ablation on the hyperparameters of the VQ codebook. Two Self-

Reconstruction Understanding Stage Distillation rFID↓ PSNR↑ SSIM↑ MME-P↑ MMB↑ AI2D↑ TQA↑

- ✗ ✗ 2.69 21.35 0.704 608.9 22.3 48.6 7.0

- ✗ ✓ 2.84 19.68 0.644 1435.2 64.9 52.8 42.6

✓ ✓ 2.71 20.52 0.680 1439.1 65.8 53.1 44.0

- Table 6. Ablation on the effect of training strategy on image understanding and reconstruction. and reconstruction in a single tokenizer in Tab. 6. The first row represents end-to-end training all the components without self-distillation, which achieves the best reconstruction quality and fails to maintain the performance of visual understanding. The second row utilizes semantic constraints and alleviates the degradation of the ability of visual understanding. It is advisable to minimize the update of wellpretrained encoder in the initial training for image reconstruction. As shown in Fig. 6, with the two-stage training strategy and self-distillation loss, we achieve a trade-off on image reconstruction and understanding.

##### 5. Conclusion

In this paper, we propose VQRAE, a vector quantization variant of RAE designed for unified tokenizers, which pioneers the first attempt to produce continuous semantic rep-

Original Stage 1 Stage 2 E2E

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

Figure 6. Visualization results on ablation study of training strategies. As indicated in Tab. 6, the second training stage adds more fine-grained details on reconstruction and retains semantics, while end-to-end training without distillation constraints fails to achieve a trade-off between them.

resentations for multimodal understanding and fine-grained discrete tokens for visual generation. We eliminate the dependency on pixel encoders during the training of the unified tokenizer, employing a pure ViT-based model and a two-stage training strategy to achieve the integration of visual understanding, generation, and reconstruction. Built upon pre-trained VFMs, VQRAE is the first to realize highutilization, high-dimensional codebooks for discrete autoregressive modeling. Extensive experiments on the multimodal understanding, generation and reconstruction bench-

marks demonstrate the competitive performance compared to both diffusion-based generative models and autoregressive models with unified tokenizers.

##### 6. Acknowledgements

This work was supported by Kuaishou Technology. We extend our sincere gratitude to all collaborators involved in this project. Moreover, this work is also supported by the National Key R&D Program of China (2022YFB4701400/4701402), SSTIC Grant (KJZD20230923115106012, KJZD20230923114916032, GJHZ20240218113604008).

##### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6, 7, 1
- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 1
- [3] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315– 11325, 2022. 3
- [4] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3558–3568, 2021. 6, 1
- [5] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024. 8
- [6] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 3, 6, 1
- [7] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 1, 3, 8

- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 7
- [9] Zisheng Chen, Chunwei Wang, Xiuwei Chen, Hongbin Xu, Runhui Huang, Jun Zhou, Jianhua Han, Hang Xu, and Xiaodan Liang. Semhitok: A unified image tokenizer via semantic-guided hierarchical codebook for multimodal understanding and generation. arXiv preprint arXiv:2503.06764, 2025. 7, 1
- [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 6, 1

- [11] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025. 1, 3, 6
- [12] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 3
- [13] Sinan Du, Guosheng Zhang, Keyao Wang, Yuanrui Wang, Haixiao Yue, Gang Zhang, Errui Ding, Jingdong Wang, Zhengzhuo Xu, and Chun Yuan. Alore: Efficient visual adaptation via aggregating low rank experts. arXiv preprint arXiv:2412.08341, 2024. 3
- [14] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024. 2
- [15] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 1, 3, 4, 5, 6, 7
- [16] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 8

- [17] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024. 1
- [18] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 3, 6, 7
- [19] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025. 3, 1, 6
- [20] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 6, 7, 8, 2
- [21] Jiaming Han, Hao Chen, Yang Zhao, Hanyu Wang, Qi Zhao, Ziyan Yang, Hao He, Xiangyu Yue, and Lu Jiang. Vision as a dialect: Unifying visual understanding and generation via text-aligned representations. arXiv preprint arXiv:2506.18898, 2025. 3, 5, 7, 1

- [22] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 6, 7, 8, 2
- [23] Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, et al. Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement. arXiv preprint arXiv:2504.01934, 2025. 6, 1
- [24] Ziyuan Huang, DanDan Zheng, Cheng Zou, Rui Liu, Xiaolong Wang, Kaixiang Ji, Weilong Chai, Jianxin Sun, Libin Wang, Yongjie Lv, et al. Ming-univision: Joint image understanding and generation with a unified continuous tokenizer. arXiv preprint arXiv:2510.06590, 2025. 1, 3
- [25] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 6, 7
- [26] Kat Kampf and Nicole Brichtova. Experiment with gemini 2.0 flash native image generation, 2025. URL https://developers. googleblog. com/en/experiment-withgemini-20-flash-native-image-generation, 3, 2025. 1
- [27] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images, 2016. 6, 7
- [28] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1, 5
- [29] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 3, 6, 7
- [30] Bo Li, Peiyuan Zhang, Kaichen Zhang, Fanyi Pu, et al. Lmms-eval: Accelerating the development of large multimoal models, 2024. 1
- [31] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 3, 5
- [32] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 6, 7
- [33] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 3
- [34] Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422,

2025. 7, 1

- [35] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-

- to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024. 8
- [36] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yi Xin, Xinyue Li, Qi Qin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic textto-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 1, 3
- [37] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 3, 4, 6, 7, 1
- [38] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multimodal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 6, 7
- [39] Yifu Luo, Penghui Du, Bo Li, Sinan Du, Tiantian Zhang, Yongzhe Chang, Kai Wu, Kun Gai, and Xueqian Wang. Sample by step, optimize by chunk: Chunklevel grpo for text-to-image generation. arXiv preprint arXiv:2510.21583, 2025. 3
- [40] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 6
- [41] Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025. 1, 3, 7
- [42] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer,

2024. 5

- [43] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai yu, Liang Zhao, Yisong Wang, Jiaying Liu, and Chong Ruan. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation, 2024. 1, 3
- [44] OpenAI. Introducing 4o image generation, 2025. 1
- [45] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 3
- [46] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748,

2022. 3, 5, 1

- [47] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366, 2022. 3, 4, 5, 7
- [48] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 8

- [49] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025. 1, 3, 4, 6, 7, 8
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1, 3, 4
- [51] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 3
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 8
- [53] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16037–16046, 2025. 3, 5, 7
- [54] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 3, 6, 7
- [55] Wei Song, Yuran Wang, Zijia Song, Yadong Li, Haoze Sun, Weipeng Chen, Zenan Zhou, Jianhua Xu, Jiaqi Wang, and Kaicheng Yu. Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324, 2025. 1
- [56] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in neural information processing systems, 36:49659–49678, 2023. 6, 1
- [57] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 3, 5, 6, 7, 8
- [58] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024. 3
- [59] Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025. 3, 4, 5, 6
- [60] Chameleon Team. Chameleon: Mixed-modal early-fusion

foundation models. arXiv preprint arXiv:2405.09818,

2024. 1, 3, 8

- [61] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025. 3, 5
- [62] Qwen Team. Qwen2.5: A party of foundation models,

2024. 6, 1

- [63] Qwen Team. Qwen3 technical report, 2025. 1
- [64] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 3, 6
- [65] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 1, 4, 6
- [66] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 1, 3, 4, 5
- [67] Junke Wang, Zhi Tian, Xun Wang, Xinyu Zhang, Weilin Huang, Zuxuan Wu, and Yu-Gang Jiang. Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl. arXiv preprint arXiv:2504.11455,

2025. 3, 8, 6

- [68] Peiyu Wang, Yi Peng, Yimeng Gan, Liang Hu, Tianyidan Xie, Xiaokun Wang, Yichen Wei, Chuanxin Tang, Bo Zhu, Changshi Li, et al. Skywork unipic: Unified autoregressive modeling for visual understanding and generation. arXiv preprint arXiv:2508.03320, 2025. 3
- [69] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1
- [70] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 3, 7, 8
- [71] Yuanrui Wang, Cong Han, Yafei Li, Zhipeng Jin, Xiawei Li, SiNan Du, Wen Tao, Shuanglong Li, Yi Yang, Chun Yuan, et al. Uniglyph: Unified segmentation-conditioned diffusion for precise visual text synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18335–18344, 2025. 3
- [72] Hongyang Wei, Shuaizheng Liu, Chun Yuan, and Lei Zhang. Perceive, understand and restore: Real-world image super-resolution with autoregressive multimodal generative models. arXiv preprint arXiv:2503.11073, 2025. 3
- [73] Hongyang Wei, Baixin Xu, Hongbo Liu, Cyrus Wu, Jie Liu, Yi Peng, Peiyu Wang, Zexiang Liu, Jingwen He, Yidan Xietian, et al. Skywork unipic 2.0: Building kontext model

- with online rl for unified multimodal model. arXiv preprint arXiv:2509.04548, 2025. 3
- [74] Yongxian Wei, Runxi Cheng, Weike Jin, Enneng Yang, Li Shen, Lu Hou, Sinan Du, Chun Yuan, Xiaochun Cao, and Dacheng Tao. Unifying multimodal large language model capabilities and modalities via model merging. arXiv preprint arXiv:2505.19892, 2025. 3
- [75] Yongxian Wei, Yilin Zhao, Li Shen, Xinrui Chen, Runxi Cheng, Sinan Du, Hao Yu, Gang Liu, Jiahong Yan, Chun Yuan, et al. Learning to pose problems: Reasoning-driven and solver-adaptive data synthesis for large reasoning models. arXiv preprint arXiv:2511.09907, 2025. 3
- [76] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 1, 3, 4, 8
- [77] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871,

2025. 3

- [78] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 1, 3, 7
- [79] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025. 8
- [80] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528,

2024. 1, 3, 6

- [81] Ji Xie, Trevor Darrell, Luke Zettlemoyer, and XuDong Wang. Reconstruction alignment improves unified multimodal models, 2025a. URL https://arxiv.org/abs/2509.07295, 2025. 1, 3
- [82] Rongchang Xie, Chen Du, Ping Song, and Chang Liu. Muse-vl: Modeling unified vlm through semantic discrete encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24135– 24146, 2025. 1, 3, 4, 6
- [83] Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, and Jian Guo. Chartbench: A benchmark for complex visual reasoning in charts. arXiv preprint arXiv:2312.15915, 2023. 3
- [84] Zhengzhuo Xu, Bowen Qu, Yiyan Qi, Sinan Du, Chengjin Xu, Chun Yuan, and Jian Guo. Chartmoe: Mixture of diversely aligned expert connector for chart understanding. arXiv preprint arXiv:2409.03277, 2024.
- [85] Zhengzhuo Xu, SiNan Du, Yiyan Qi, Siwen Lu, Chengjin Xu, Chun Yuan, and Jian Guo. Chartpoint: Guiding mllms

- with grounding reflection for chart reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 426–436, 2025. 3
- [86] Zhiyuan Yan, Kaiqing Lin, Zongjian Li, Junyan Ye, Hui Han, Zhendong Wang, Hao Liu, Bin Lin, Hao Li, Xue Xu, et al. Can understanding and generation truly benefit together–or just coexist? arXiv e-prints, pages arXiv–2509,

2025. 1

- [87] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 5, 6, 1
- [88] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 3
- [89] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. NeurIPS,

2024. 1

- [90] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 5, 7
- [91] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024. 6, 7
- [92] Zhengrong Yue, Haiyu Zhang, Xiangyu Zeng, Boyu Chen, Chenting Wang, Shaobin Zhuang, Lu Dong, KunPeng Du, Yi Wang, Limin Wang, et al. Uniflow: A unified pixel flow tokenizer for visual understanding and generation. arXiv preprint arXiv:2510.10575, 2025. 3
- [93] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 1, 3
- [94] Kaichen Zhang, Bo Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024. 1
- [95] Yue Zhao, Fuzhao Xue, Scott Reed, Linxi Fan, Yuke Zhu, Jan Kautz, Zhiding Yu, Philipp Kr¨ahenb¨uhl, and De-An Huang. Qlip: Text-aligned visual tokenization unifies autoregressive multimodal understanding and generation. arXiv preprint arXiv:2502.05178, 2025. 1, 3, 7
- [96] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025. 1, 3, 4, 5, 6, 7
- [97] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe

- Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 1
- [98] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 4, 6, 7, 1
- [99] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vq-gan to 100,000 with a utilization rate of 99%. Advances in Neural Information Processing Systems, 37:12612–12635, 2024. 3
- [100] Yongxin Zhu, Bocheng Li, Yifei Xin, Zhihua Xia, and Linli Xu. Addressing representation collapse in vector quantized models with one linear layer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22968–22977, 2025. 3, 4

## VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction

### Supplementary Material

In the supplementary materials, Section A provides a concise overview of the motivation underlying this study and highlights the distinctions between VQRAE and other related works; Section B illustrates the detailed training configurations of the tokenizer (Section B.1) and autoregressive models for understanding and generation (Section B.2) and their evaluation setups (Section B.3). Section C presents more qualitative results in image reconstruction (Section C.1), visual generation (Section C.2) and failure cases (Section C.3). Finally, we summarize the limitations of our work and future exploration in Section D.

##### A. Overview

Related Work. Reviewing the evolution of unified models, pioneering works such as Chameleon [60] and EMU-3 [70] employed VQGAN [15] as the encoder for both understanding and generation tasks, albeit at the expense of semantic understanding. The Janus series [7, 43, 76] adopted a dual-encoder architecture, utilizing a semantic encoder like CLIP [50] for multimodal understanding and a pixel encoder such as VQGAN [15] for image generation. This approach, however, hindered interaction and alignment between the two types of representations. The unified tokenizer aims to employ a single encoder-decoder framework to maintain performance on understanding tasks while enabling generation. Nevertheless, many previous methods [9, 23, 34, 41, 49, 55, 78, 82, 95] were overly complex in design or relied on simple contrastive learning to provide weak semantic supervision on latent tokens, making it difficult to achieve a satisfactory trade-off between comprehension and reconstruction. These tokenizers generally still underperform compared to classic understanding-only baseline models like LLaVA-1.5 [37]. Tar [21] and X-Omni [19] were among the first works to utilize pretrained Visual Foundation Models (VFMs) as encoders while discretizing representations to preserve multimodal understanding capabilities. Although they narrowed the performance gap of earlier discrete tokenizers [60, 70] on understanding tasks, they still suffer from quantization errors and lack inherent autoencoder properties. Recent work, RAE [96], discovered that high-dimensional ViT encoders can be directly applied to reconstruction and have been used to replace original VAEs [28] in diffusion-based generative models [46].

Contributions. The contribution of our paper can be summarized in three folds: (i) We propose a vector quantization version of RAE, namely VQRAE, which pioneers the first attempt in unifying understanding, generation and recon-

struction. (ii) VQRAE is the first unified tokenizer to produce continuous semantic features for understanding and fine-grained discrete tokens for generation and reconstruction simultaneously. (iii) VQRAE features the first highdimensional VQ codebook (comparable to CLIP encoders) with a nearly 100% utilization ratio, which is contrary to previous explorations in this field.

##### B. Implementation Details

###### B.1. Tokenizer Training Details

VQRAE is pretrained on BLIP3-o [6] open-sourced data, which consists of 27M samples recaptioned by Qwen2.5VL-7B [1], 5M samples from CC12M [4], and 4M synthesized images from JourneyDB [56] (Table 1 & 2). We train three variants of encoder: InternViT-300M448px [98], SigLIP2-so400m-256px [65] and SigLIP2so400m-512px [65]. The decoder adopts the symmetric design with encoder. Specific training configurations are provided in Table 7. The code implementation is adapted from TiTok [89] repository.

###### B.2. Autoregressive Training Details

For image understanding, we follow the setups in LLaVA1.5 [37], which uses LLaVA-Pretrain-595K and LLaVAv1.5-mix-665K for pretraining and SFT (Table 1 & 3 & 6). For visual generation, we train it on BLIP3-o [6] data and additional 80M high quality images (Table 4). We conduct ablation studies on VQ codebook on ImageNet1K with 20 epochs training for efficiency (Table 5 & 6). Note that, the results in Table 6 are attained through training on stage 1. The LLM backbones include: Vicuna-v1.57B [10], Vicuna-v1.5-13B [10] and Qwen2.5-7B [62] for understanding; Qwen3-0.6B [87] for generation. Besides, we did not conduct specific training on understanding tasks for our pretrained tokenizer. Specific training configurations are provided in Table 8 & 9. The code implementation is adapted from the LLaVA [37] repository.

###### B.3. Evaluation Details

For image reconstruction, we evaluate on the 256 x 256 ImageNet 50k validation set to compute the rFID, PSNR and SSIM as in TiTok [89] official codebase in Table 1 & 2. For multimodal understanding, we evaluate the SigLIP2-Vicuna variants using the LMMs-Eval codebase [30, 94]. We directly replace the ViT with our tokenizer without specific training. Due to the compatibility with InternVL3 [98], we

Stage1 Stage2 Model SigLIP2-so400m SigLIP2-so400m InternViT-300M SigLIP2-so400m SigLIP2-so400m InternViT-300M

resolution 256px 512px 448px 256px 512px 448px freeze encoder true true true false false false codebook size 16384 16384 16384 16384 16384 16384 codebook dim 1536 1536 1536 1536 1536 1536

discriminator start steps NA NA NA 50000 50000 30000 discriminator weight NA NA NA 0.1 0.1 0.1

distillation weight NA NA NA 1.0 1.0 1.0 perceptual loss weight 1.1 1.1 1.1 1.1 1.1 1.1

perceptual model convnext-s convnext-s convnext-s convnext-s convnext-s convnext-s augmentation random crop & random flip random crop & random flip random crop & random flip random crop & random flip random crop & random flip random crop & random flip

encoder lr NA NA NA 1e-5 1e-5 1e-5 decoder lr 4e-4 4e-4 4e-4 1e-4 1e-4 1e-4

end lr 1e-4 1e-4 1e-4 1e-5 1e-5 1e-5 scheduler cosine cosine cosine cosine cosine cosine weight decay 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4

discriminator lr NA NA NA 4e-5 4e-5 4e-5 optimizer AdamW AdamW AdamW AdamW AdamW AdamW (β1,β2) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999)

warmup steps 2000 2000 2000 2000 2000 2000 mixed precision bf16 bf16 bf16 bf16 bf16 bf16 max grad norm 1.0 1.0 1.0 1.0 1.0 1.0

global batch size 1024 1024 768 1024 1024 768 total steps 100000 100000 45000 70000 70000 60000

Table 7. The detailed training configurations of VQRAE tokenizer.

Hyperparameters Pretrain SFT freeze backbone true false freeze ViT true true freeze connector false false

mm projector type mlp2x gelu mlp2x gelu mm vision select layer -1 -1

deepspeed stage ZeRO-2 ZeRO-3

scheduler cosine cosine learning rate 1e-3 2e-5 weight decay 0.0 0.0

optimizer AdamW AdamW (β1,β2) (0.9, 0.999) (0.9, 0.999)

warmup ratio 0.03 0.03 mixed precision bf16 bf16 max grad norm 1.0 1.0

global batch size 256 128 model max length 2048 2048

Table 8. The detailed training configurations of multimodal understanding, SigLIP2-Vicuna-v1.5-7B / 13B.

utilize the OpenCompass VLMEvalKit [14] for evaluation of InternViT-Qwen2.5-7B in Table 3. For visual generation, we use the official evaluation toolkit from GenEval [20] and DPG-Bench [22] in Table 4.

##### C. Additional Qualitative Results

We provide more qualitative results on image reconstruction and generation in this section. Also, we present some failure cases in our tokenizer and AR model.

#### Hyperparameters Generation Tuning

augmentation center crop deepspeed stage ZeRO-1

scheduler constant learning rate 1e-4 weight decay 0.0

optimizer AdamW (β1,β2,ϵ) (0.9, 0.999, 1e-15)

warmup steps 4000 mixed precision bf16 max grad norm 1.0

global batch size 512 model max length 1536

Table 9. The detailed training configurations of visual generation.

###### C.1. Reconstruction Results

As shown in Figure 7, our VQRAE can achieve fine-grained reconstruction in human faces, scenes and objects.

###### C.2. Generation Results

We present additional visual generation results in Figure 8. Our method can generate images with various styles, subjects, and scenarios.

###### C.3. Failure Cases

As shown in Figure 9, our tokenizer remains flawed in text reconstruction and high-density scenarios, which is likely

attributable to the trade-off between semantic representation and reconstruction performance and specific text data tuning. Moreover, in terms of image generation, in Figure 10, certain artifacts persist in fingers and human faces, issues that may primarily necessitate resolution through post-training [6, 19, 67].

##### D. Limitation and Future Work

The primary limitation of VQRAE lies in the lack of exploration of alternative and more effective methods to balance understanding and reconstruction performance to minimize the compromise on understanding capability. The potential for reconstruction and generation to enhance understanding remains underexplored. Additionally, the quantization loss inherent in the discrete tokenizer makes it challenging for VQRAE to compete with state-of-the-art continuous VAEs. There is still room for improvement in generation quality, particularly in understanding spatial relationships, texture rendering, and addressing artifacts in faces and fingers.

Looking ahead, several directions [13, 19, 39, 61, 70, 71, 74, 75, 83–85] appear particularly promising for future research. In this work, our main objective is to develop a unified tokenizer that provides more effective representations for understanding, generation, and reconstruction tasks. However, leveraging such representations to integrate various tasks into a single model requires further investigation. Issues such as conflicts and synergies among different tasks, as well as efficient model scaling, are left for future work.

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

###### Figure 7. Additional visualization of reconstruction results from VQRAE-InternViT. Left: input image; Right: output image.

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

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

###### Figure 8. Additional visualization of generation results at 512 x 512 px.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Figure 9. Failure reconstruction cases from VQRAE-InternViT. Left: input image; Right: output image.

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

Figure 10. Failure generation cases. Our model still has certain artifacts in generating fine-grained text, small human faces and fingers, which can be addressed with extensive training data and reinforcement learning as explored in [11, 19, 67].

