# arXiv:2604.07340v1[cs.CV]8Apr2026

## TC-AE: Unlocking Token Capacity for Deep Compression Autoencoders

Teng Li1,2,∗, Ziyuan Huang1,∗,†, Cong Chen1,3,∗, Yangfu Li1,4, Yuanhuiyi Lyu1,5, Dandan Zheng1, Chunhua Shen3, Jun Zhang2,†

1Inclusion AI, Ant Group, 2HKUST, 3ZJU, 4ECNU, 5HKUST (GZ) ∗Equal contribution, †Corresponding authors

We propose TC-AE, a Vision Transformer (ViT)– based architecture for deep compression autoencoders. Existing methods commonly increase the channel number of latent representations to maintain reconstruction quality under high compression ratios. However, this strategy often leads to latent representation collapse, which degrades generative performance. Instead of relying on increasingly complex architectures or multi-stage training schemes, TC-AE addresses this challenge from the perspective of the token space, the key bridge between pixels and image latents, through two complementary innovations: (1) We study token number scaling by adjusting the patch size in ViT under a fixed latent budget, and identify aggressive token-to-latent compression as the key factor that limits effective scaling. To address this issue, we decompose token-to-latent compression into two stages, reducing structural information loss and enabling effective token number scaling for generation. (2) To further mitigate latent representation collapse, we enhance the semantic structure of image tokens via joint self-supervised training, leading to more generative-friendly latents. With these designs, TC-AE achieves substantially improved reconstruction and generative performance under deep compression. We hope our research will advance ViT-based tokenizer for visual generation.

[Figure 1]

Date: April 9, 2026 : https://github.com/inclusionAI/TC-AE

### 1 Introduction

Latent diffusion models (LDMs) (Rombach et al., 2022) have emerged as the dominant paradigm for efficient image generation. They typically adopt a two-stage framework: a tokenizer compresses images into latent representations and a diffusion model operates in the latent space. To further improve efficiency, recent works (Chen et al., 2024; Huang et al., 2025) increasingly push tokenizers toward deep compression by aggressively reducing the spatial resolution of the latent representation, while compensating with a larger channel number to preserve reconstruction quality. However, this combination of extreme spatial compression and channel expansion often leads to latent representation collapse, resulting in degraded generative performance (Chen et al., 2025c). In this work, we study this challenge in the context of ViT-based tokenizers, due to their strong potential in scalability and representation capacity (Hansen-Estruch et al., 2025; Xiong et al., 2025).

In ViT-based tokenizers (Hansen-Estruch et al., 2025; Chen et al., 2025a; Huang et al., 2025), images are firstly patchified into a sequence of image tokens, which are then processed by Transformer blocks, and finally compressed into compact latents through a bottleneck. Image tokens therefore serve as a critical bridge between raw pixels and the latent space, and their capacity and structure directly influence the quality of the resulting latent representations. Existing approaches primarily

Spatial compression

Image latent

Low token resolution

High token resolution

Enc. Dec. Enc. Dec.

Enc. Enc. Dec. Dec.

[Figure 2]

[Figure 3]

[Figure 4]

High FLOPs Good Gen.

Small FLOPs Poor Gen.

Small FLOPs Good Gen.

[Figure 5]

[Figure 6]

[Figure 7]

(a) CNN (DC-AE) (b) Vanilla ViT (MingTok) (c) Ours: ViT with staged compression

- Figure 1 Architectural Comparasion of existing methods. (a) CNN-based tokenizers achieve strong generation with high FLOPs. (b) Vanilla ViT-based tokenizers are computationally efficient but suffer from severe information loss in the patchify layer, as spatial resolution is not gradually reduced as in CNNs. (c) TC-AE introduces staged token compression in the ViT encoder, balancing generative performance and efficiency.

focus on scaling model parameters (Hansen-Estruch et al., 2025; Xiong et al., 2025) or directly optimizing the final latent representations (Yao et al., 2025b; Huang et al., 2025) to improve generative performance, while overlooking the optimization of the intermediate token space. Meanwhile, studies on ViT-based representation learning (Caron et al., 2021; Wang et al., 2025) have shown that increasing the number of tokens can substantially improve downstream performance and lead to more semantically structured representations. However, whether and how scaling the token space translates into improved latent generatability in LDM frameworks remains an open problem.

In our preliminary experiments, we observe that naively increasing the number of image tokens consistently improves reconstruction quality, but does not necessarily lead to better generation performance (Fig. 3). We further identify the token-to-latent compression as the root cause of this phenomenon (Fig. 2 and Sec. 3.3). Under a fixed pixel-to-latent compression ratio, increasing the intermediate token number inevitably amplifies the compression ratio at the token-to-latent compression bottleneck. Such aggressive compression causes substantial structural information loss in the latent space, which in turn degrades generative performance. To address this issue, we propose staged token compression, which decomposes the token-to-latent bottleneck into two stages: an intermediate compression stage applied within the ViT encoder to reduce token resolution, followed by a final bottleneck that further compresses the tokens into compact image latent. Unlike vanilla ViT-based tokenizers that rely on a single abrupt bottleneck, this design preserves latent structure under large token spaces and restores positive scaling trends for both reconstruction and generation.

Beyond increasing token capacity, we further investigate improving the semantic structure of the token space to enhance latent generatability. Recent studies (Yao et al., 2025b; Xiong et al., 2025; Huang et al., 2025) have shown that semantically regularized latent representations can significantly benefit generative modeling, often achieved by aligning latents with representations from pretrained foundation models (Oquab et al., 2023; He et al., 2022). However, such approaches rely on external large-scale pretraining and implicitly introduce strong data priors. In contrast, we introduce a joint self-supervised objective directly into tokenizer training, enabling the ViT encoder to produce semantically structured image tokens. This design consistently improves generation performance across different token numbers.

Coupling these designs, we propose TC-AE, a ViT-based tokenizer powered by a large and semantically structured token space for image generation. Fig. 1 presents a high-level framework comparison between TC-AE and other deep compression tokenizers. Fig. 8 shows that token number

scaling and model parameter scaling are complementary, jointly improving generative performance. This highlights token space optimization as a new and effective direction for ViT-based tokenizers. Our main contributions are summarized as follows:

- • Novel perspective. We are the first to address representation collapse in deep compression autoencoders from the perspective of the token space, and identify the token-to-latent compression

- as a key factor that limits effective token number scaling.

- • TC-AE. We introduce staged token compression to alleviate structural information loss during token-to-latent mapping, and further improve token structure via joint self-supervised training.
- • Strong performance. TC-AE achieves consistently better reconstruction and generative performance than existing tokenizers, and shows that token number scaling offers a more effective and complementary scaling dimension than parameter scaling.

### 2 Related Work

##### 2.1 Deep Compression Visual Tokenizers

The primary goal of visual tokenizers is to compress images into compact latent representations for efficient generative modeling (Rombach et al., 2022). Recently, achieving extremely high compression ratios (Chen et al., 2024, 2025c; Huang et al., 2025) has become an active research focus to further improve generation efficiency (He et al., 2025; Xie et al., 2025). However, such deep compression typically reduces the spatial resolution of the latent representation, often leading to severe reconstruction degradation and representation collapse. To alleviate these issues, prior works (Chen et al., 2024, 2025c) commonly increase the latent channel dimension to compensate for the loss of spatial capacity, and employ multi-stage training strategies to encourage more structured latent representations. While effective to some extent, these approaches introduce additional architectural and training complexity. In contrast, our work addresses the challenges of deep compression from a different perspective: scaling the token space. We investigate how both reconstruction and generation performance scale with the number of image tokens, and propose a staged token compression architecture that mitigates the trade-off between compression efficiency and representation quality.

##### 2.2 Scaling Token Space in ViT

In ViT-based tokenizers (Hansen-Estruch et al., 2025; Huang et al., 2025; Chen et al., 2025a), the image token acts as a critical bridge connecting raw RGB inputs to the latent space. Increasing the number of tokens theoretically provides a more comprehensive representation of the visual content. This benefit of increasing token granularity has been widely observed in visual representation learning. In self-supervised ViTs, increasing the number of tokens leads to richer semantic representations and improved transfer performance, as demonstrated by DINO (Caron et al., 2021) and subsequent patch-level representation analysis (Yun et al., 2022). More recent work further establishes systematic scaling laws for patchification, showing consistent gains across downstream tasks as token counts increase (Wang et al., 2025). Similar trends have also been reported in generative modeling (Peebles and Xie, 2023), where smaller patch size yields better generation performance. However, we are the first to systematically introduce this token scaling perspective into the design of tokenizers specifically for generation. Our investigation reveals a critical trade-off

Less token number, low compression (N/n)

Compression

Image token Image latent

Token score Latent score

| | |
|---|---|
| | |

𝐻𝑊 64

𝑁 =

29.5 6.3

𝑁

[Figure 8]

###### gFID 65.0

Constant budget

|-79%|
|---|

Compression

| | | |
|---|---|---|
| | | |
| | | |

𝐻𝑊 32

Compression

𝑁 =

[Figure 9]

41.9 13.8

22.9

𝑁

|-66%|
|---|

| | | |
|---|---|---|
| | | |
| | | |

𝐻𝑊 16

[Figure 10]

58.4

𝑁 =

17.2

𝑁

|-69%|
|---|

18.1

𝐻𝑊 32

𝑛 =

𝐻𝑊 8

62.9 5.3

𝑁 =

[Figure 11]

𝑁

25.4

|-92%|
|---|

ViT Encoder

Latent structure

More token number, high compression (N/n)

collapse

Patchify

Compression

Compression

| | | |
|---|---|---|
| | | |
| | | |

𝑁 (ours)

12.1 39.1 16.4

[Figure 12]

|-69%|
|---|

[Figure 13]

Linear probing accuracy

Staged token compression

(a) Encoder structure (b) Token-to-latent compression analysis

- Figure 2 Latent structure collapse caused by token-to-latent compression when scaling token numbers. Although increasing token numbers improves the token-level structure, the fixed latent resolution forces more aggressive token-to-latent compression, causing severe information loss and latent structure collapse at small patch sizes (e.g., p = 8).

between reconstruction and generation in naive scaling strategies, and we propose a staged token compression strategy that resolves this conflict.

- 2.3 Representation Learning in Generation

The integration of representation learning into generative frameworks has recently achieved remarkable success. For example, REPA (Yu et al., 2024) and iREPA (Singh et al., 2025) improve generative performance by aligning intermediate representations in the SiT (Ma et al., 2024) framework with features from DINOv2 (Oquab et al., 2023). Similarly, in the context of tokenizer design, approaches such as VA-VAE (Yao et al., 2025b), MAETok (Chen et al., 2025b), and GigaTok (Xiong et al., 2025) incorporate pretrained visual foundation models to guide latent optimization, while RAE (Zheng et al., 2025) directly adopts representations from foundation models for generation. Despite their effectiveness, these strategies typically rely on heavy teacher models and implicitly introduce strong priors from large-scale pretraining. In contrast, we adopt a self-supervised joint training mechanism that structures the token space during tokenizer training, enabling us to demonstrate the intrinsic value of representation learning for deep compression tokenizers.

- 3 Method

##### 3.1 Preliminary: Compression in ViT-based Image Autoencoder

Recent image tokenizers (Hansen-Estruch et al., 2025; Xiong et al., 2025; Chen et al., 2025a; Yao et al., 2025a) increasingly adopt the Vision Transformer (ViT) (Dosovitskiy, 2020) architecture due to its strong scalability and representation capacity. A vanilla ViT-based image autoencoder consists of an encoder E and a decoder D, which compress an input image into a latent representation and reconstruct it back to the pixel space.

Given an input image X ∈ RH×W×3, the encoder applies a patch embedding layer ϕp(·) that partitions the image into a 2D grid of non-overlapping patches with size p × p, projects each patch

[Figure 14]

- Figure 3 Naive scaling of token numbers for reconstruction and generation. We increase the number of image tokens by reducing the ViT patch size. While reconstruction quality consistently improves with more tokens (shown in the left two figures), generative performance does not exhibit corresponding gains (shown in the right two figures).

to a d-dimensional embedding, and flattens the grid into a token sequence:

T = ϕp(X) ∈ RN×d, N =

H · W p2

. (1)

The image tokens T are processed by a stack of Transformer layers TF(·) and subsequently compressed by a bottleneck layer B(·) into a latent tensor:

z = B(TF(T)) ∈ Rh×w×c, (2)

where (h,w) denotes the spatial resolution of the latent representation z and c is the channel dimension. The latent z then serves as the learning objective for downstream generative models.

Image tokens as the information bridge. In a ViT-based autoencoder, the information flow from the input image to the latent representation involves two consecutive compression stages: pixel-totoken compression in the patch embedding layer, and token-to-latent compression at the bottleneck. The overall spatial compression ratio fpix→lat can therefore be decomposed as:

N h · w

, (3)

fpix→lat = fpix→tok × ftok→lat = p2 ·

where fpix→tok and ftok→lat denote the spatial compression introduced at the patch embedding and bottleneck stages, respectively.

Image tokens therefore act as an information bridge between the input image and the latent space. Their capacity and semantic structure directly determine how effectively information is preserved through compression. In the following sections, we study how improving the token space impacts the quality of the latent representation and, consequently, the reconstruction and generative performance.

##### 3.2 Naive Scaling Token Numbers Fails to Improve Generation

Increasing the number of image tokens in ViT has been shown to improve model capacity (Caron et al., 2021; Wang et al., 2025). Motivated by this observation, we investigate whether token number scaling can similarly benefit deep compression autoencoders.

We conduct controlled experiments under a fixed latent budget. Images are spatially compressed by a factor of fpix→lat = 32, and the latent channel dimension is fixed to c = 128 across all settings, following common practice in deep compression autoencoders (Chen et al., 2025c). The number of image tokens is increased by sweeping the patch size p ∈ {32,16,8} for different tokenizer model

[Figure 15]

- Figure 4 Token number scaling with staged token compression (STC). This strategy mitigates structural information loss at the bottleneck under large token spaces, enabling generative performance to scale with increasing token numbers.

sizes (ViT-S, ViT-B, and ViT-L). As shown in Fig. 3, reconstruction quality consistently improves as the number of image tokens increases, indicating that higher token resolution enables the tokenizer to capture finer-grained visual details. In contrast, generation performance does not improve with token number scaling. While similar limitations have been reported in prior works when scaling model parameters (Hansen-Estruch et al., 2025; Xiong et al., 2025), our results reveal that this issue persists when scaling along the token space dimension via patch size reduction.

Observation 1: Scaling the number of image tokens improves reconstruction quality, but it does not improve generative performance under a fixed latent budget.

##### 3.3 Scaling Token Numbers with Staged Token Compression

Prior works (Yao et al., 2025b; Xiong et al., 2025; Zheng et al., 2025; Shi et al., 2025) have shown that enriching latent representations with semantic structure improves their generative quality. Motivated by this observation, we analyze the failure of token number scaling from the perspective of semantic preservation.

Specifically, we examine how semantic information carried by image tokens is retained when compressed into the latent representation using linear probing (Xiong et al., 2025). We train linear classifiers on spatially average-pooled image tokens (before bottleneck) and latent features (after bottleneck), and denote the resulting classification accuracies as A1 and A2, respectively. The ratio A2/A1 serves as a quantitative measure of semantic information preserved across the compression bottleneck B(·), while (1 − A

A1) represents the structure loss in this process.

2

Bottleneck-induced latent structure loss. Results are reported in Table 1. As the patch size decreases, the semantic structure of image tokens improves as indicated by increasing A1. However, this improvement does not carry over to the latent space: at smaller patch sizes (e.g., p = 8), A2 drops sharply, resulting in a very high information loss (1 − A

A1 = 0.92). This indicates severe semantic loss during token-to-latent compression.

2

This structure loss can be explained by the compression budget constraint in Eq. 3. Under a fixed pixel-to-latent compression ratio fpix→lat, decreasing the patch size reduces the pixel-to-token compression factor fpix→tok, but necessarily increases the token-to-latent compression factor ftok→lat. Consequently, although image tokens become increasingly semantically structured, the aggressive compression at the bottleneck destroys this structure, preventing generative performance from scaling with the number of tokens.

Staged token compression. To address this issue, we propose staged token compression to redistribute the compression process from the bottleneck into the ViT Transformer layers. Specifically, we divide the layers into two stages. The first stage operates on a large number of tokens to learn

Table 1 Latent structure loss under token-to-latent compression. Increasing token numbers improves semantics of image tokens (A1), but aggressive bottleneck compression causes severe semantic loss in the latent space. Bottl.: bottleneck. Str.: structure.

Linear Probing Accuracy Before Bottl. (A1)↑ After Bottl. (A2)↑ Str. Loss (1 − A

Patch Size gFID↓

###### A1)↑

2

64 64.96 29.5 6.3 0.79 32 22.93 41.9 13.8 0.67 16 17.22 58.4 18.1 0.69

8 25.36 62.9 5.33 0.92 8 (w/ STC) 16.39 39.1 12.1 0.69

rich semantic structure, followed by an intermediate compression layer that aggregates semantics into a small set of tokens. Then, a second compression is applied at the bottleneck to produce the final latent representation. By decomposing a single aggressive compression step into multiple steps, image latent preserves semantic structure more effectively during token-to-latent transformation.

As shown in Table 1, staged token compression substantially increases the semantic preservation ratio A2/A1 from 0.08 to 0.31. Consequently, as shown in Fig. 4, it achieves the strongest generative performance and enables generation quality to continue improving with increasing numbers of image tokens.

Observation 2: We identify the token-to-latent compression bottleneck as the primary factor limiting token number scaling, and it can be addressed by with staged token compression.

##### 3.4 Enhancing Token Structure via Self-Supervision

Beyond increasing the number of tokens, we investigate enhancing the semantic structure of the token space to improve the generative quality of the latent. Prior works (Yao et al., 2025b; Xiong et al., 2025; Zheng et al., 2025; Shi et al., 2025) typically achieve this by aligning latent with pretrained models or initializating encoder from their weights. However, such approaches rely on external large-scale pretraining and require the pretraining configuration (e.g., image resolution and compression ratio) to be tightly coupled with the tokenizer design. In contrast, we introduce joint self-supervised learning (SSL) as an auxiliary objective to structure the token space during tokenizer training.

Self-supervised auxiliary training with iBOT. We incorporate iBOT (Zhou et al., 2021) as an auxiliary training objective. iBOT employs a student–teacher distillation framework. The teacher is an exponential moving average (EMA) of the student.

Given an input image X, teacher and student inputs are generated via two augmentation pipelines, Augtea(·) and Augstu(·). The teacher pipeline Augtea(X) produces two global crops. The student pipeline Augstu(X) produces the same global crops with random patch masking, along with multiple additional local crops. For the masked global crops, the student is trained to predict the teacher patch-token outputs, corresponding to a masked image modeling objective LMIM. For the local crops, the student class-token predictions are aligned with those of the teacher to enforce semantic consistency across different views, yielding the class-token distillation loss L[CLS]. Together, these objectives encourage both local-level and global-level semantic structure. The overall self-supervised objective is given by:

LiBOT = LMIM + L[CLS]. (4)

[Figure 16]

Compression

[Figure 17]

Compression

×𝑁

×𝑀

ViTBlock

ViTBlock

ViTBlock

ViTBlock

Pachify

###### Decoder

𝐿

𝐴𝑢𝑔

Stage2

Stage1

iBOT Head

###### ViT Encoder

Rec. forward iBOT forward

EMA 𝐿

𝐴𝑢𝑔

Rec. class token iBOT class token

Rec. patch token iBOT patch token Image latent Mask token

iBOT Online ViT Encoder Head

- Figure 5 Architecture of TC-AE. TC-AE introduces staged token compression to alleviate structural information loss at the bottleneck, enabling generative performance to scale with the number of image tokens. In addition, self-supervised training is incorporated to further structure the token space.

##### 3.5 TC-AE

Based on the above studies, we propose TC-AE, a ViT-based image tokenizer powered by a large and semantically structured token space. An overview of the architecture is shown in Fig. 5.

TC-AE consists of a ViT encoder, a latent bottleneck, and a structurally symmetric decoder. The first innovation lies in the encoder design. Concretely, the encoder begins with a patch embedding layer using a small patch size p to produce high-resolution image tokens, reducing information loss at the pixel-to-token compression. These fine-grained tokens are processed by the first M ViT blocks to capture rich visual details and semantic structure. An intermediate bottleneck then compresses the token sequence to 1/4 length, yielding a compact and structured intermediate representation. The compressed tokens are further processed by the remaining N ViT blocks, after which a second bottleneck produces the final compact latent representation for downstream generative modeling.

The second innovation in TC-AE lies in its training scheme, which jointly optimizes the tokenizer with a self-supervised objective. This encourages the ViT encoder to learn latent representations with stronger semantic regularization, without requiring external large-scale pretraining for foundation models. Compared to VTP (Yao et al., 2025a), TC-AE adopts a lightweight training scheme, making it practical to deploy under limited computational resources (see Sec. 4.5).

The overall training objective of TC-AE combines the standard reconstruction loss with the selfsupervised objective:

LTC-AE = αLrec + LiBOT. (5) The reconstruction loss LAE is defined as:

Lrec = Lpix + λpLp + λgLg, (6)

where Lpix denotes the pixel-level reconstruction loss, implemented as a ℓ1 term between original image and reconstructed image. Lp is a perceptual loss that captures high-level semantic discrepancies, and Lg is an adversarial loss that encourages the realism of reconstructed images.

3.5× faster

2.7× faster

1.9× faster

4.7× faster

- Figure 6 Diffusion model convergence comparison. Both staged token compression strategy and joint self-supervised training accelerate the diffusion convergence.

### 4 Experiments

##### 4.1 Implementation Details

Tokenizer training. All experiments are conducted on ImageNet-1K (Deng et al., 2009) with an input resolution of 256 × 256. Compression layers within the ViT encoder are implemented using two-layer convolutional modules. The final token-to-latent bottleneck is realized via a pixel-shuffle operation followed by an MLP. For ablation studies, the tokenizer is trained for 50 epochs with a batch size of 512, using the Lpixel, Lp, and LiBOT. For scale-up experiments, we further incorporate the adversarial loss Lg and finetune the decoder for additional 16 epochs. More details and ablations are provided in the appendix.

Evaluation. The reconstruction quality is evaluated using SSIM (Wang et al., 2004), PSNR (Hore and Ziou, 2010), FID (Heusel et al., 2017), and LPIPS (Zhang et al., 2018) on ImageNet-50k validation set. We adopt LightningDiT from RAE (Zheng et al., 2025) to evaluate the generative capability of the tokenizer. The model is trained for 80 epochs with a batch size of 1024 for ablation studies. For system-level comparason, the model is trained for 800 epochs. The generative performance is measured using FID and Inception Score (IS) (Barratt and Sharma, 2018).

##### 4.2 Synergistic Effects of Token Number Scaling and Structuring

Accelerating diffusion model convergence. We evaluate the convergence behavior of DiT trained on latents produced by different tokenizers under identical training settings. As shown in Fig. 6, introducing staged token compression already leads to a noticeable acceleration compared to the baseline: measured by gFID and IS, convergence is accelerated by 2.7× and 1.9×, respectively. When further incorporating self-supervised token structuring, the full TC-AE model achieves substantially faster convergence, reaching the same gFID with 4.7× fewer training iterations and comparable IS with 3.5× fewer steps.

We further provide some qualitative comparison between TC-AE and baseline in Fig. 7. These results indicate that staged token compression and self-supervised training contribute complementary benefits. By reducing structural information loss during token-to-latent compression and learning a more semantically organized latent space, TC-AE provides a more favorable representation for diffusion model optimization, leading to significantly improved training efficiency.

Effectiveness of staged token compression. Table 2 studies the effect of staged token compression, both with and without self-supervision.

200K 400K 600K 200K 400K 600K 200K 400K 600K

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

TC-AEBaselineTC-AEBaseline

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

- Figure 7 Qualitative comparison between TC-AE and the baseline at 256×256 resolution. Images are sampled from DiT checkpoints

- at different training steps using the same initial noise. TC-AE produces more visually appealing results compared to the baseline.

Table 2 Staged token compression improves generation performance.

Staged token compression rFID↓ PSNR↑ LPIPS↑ SSIM↑ gFID↓ IS↑ w/o SSL

- ✗ 1.33 25.13 0.068 0.794 44.72 33.62

✓ 0.75 25.72 0.072 0.807 32.92 43.58 w/ SSL

- ✗ 0.90 24.84 0.085 0.772 25.36 52.38

✓ 0.90 24.21 0.098 0.749 16.39 71.33

Without SSL, introducing staged token compression consistently improves both reconstruction and generation quality. Specifically, rFID is significantly reduced from 1.33 to 0.75, while improving PSNR and SSIM. More importantly, it leads to a substantial improvement in generative performance, reducing gFID from 44.72 to 32.92 and increasing IS from 33.62 to 43.58. These results indicate that staged token compression alone effectively alleviates the token-to-latent bottleneck and enhances the generatability of the latent representation. When combined with SSL, staged token compression yields even larger gains in generation quality. While reconstruction metrics remain comparable, gFID is further reduced from 25.36 to 16.39, and IS increases markedly from 52.38 to 71.33. This suggests that staged token compression and semantic token structuring via SSL are complementary: staged token compression preserves semantic information during compression, while SSL further improve the latent structure to benefit downstream generation.

Effectiveness of self-supervision. We validate its effectiveness by comparing models trained with and without SSL, and report results under different token numbers by sweeping patch sizes.

As shown in Table 3, SSL consistently improves generative performance across all patch sizes, substantially reducing gFID and increasing IS. This confirms that SSL is effective for enhancing the generatability of the latent representation, regardless of token resolution. On the other hand, incorporating SSL generally degrades reconstruction quality, consistent with prior works that directly

Table 3 Self-supervision improves generative performance across token numbers.

ViT patch size SSL rFID↓ PSNR↑ LPIPS↑ SSIM↑ gFID↓ IS↑ 64

✗ 14.97 21.93 0.302 0.602 65.23 20.84 ✓ 30.96 20.25 0.383 0.501 64.96 24.89

✗ 5.31 23.72 0.177 0.721 37.09 38.51 ✓ 4.63 22.15 0.195 0.632 22.93 57.22

32

✗ 1.12 25.03 0.080 0.779 31.37 45.49 ✓ 1.27 23.76 0.104 0.723 17.22 69.00

16

✗ 1.33 25.13 0.068 0.794 44.72 33.62 ✓ 0.90 24.84 0.085 0.772 25.36 52.38

8

align latent with pretrained foundation models (Yao et al., 2025b; Xiong et al., 2025). We further observe that this effect is more pronounced when the number of tokens is small (e.g., p = 64), but progressively diminishes as the token 68esolution increases. At smaller patch sizes (e.g., p = 8), reconstruction performance with SSL becomes comparable to the baseline, while generative performance remains significantly improved.

##### 4.3 Synergistic Effects of Token Number Scaling and Parameter Scaling

Prior work (Hansen-Estruch et al., 2025) observes that increasing decoder parameters can improve the generatability of the latent representation. To compare the effect between model capacity and token capacity, we vary the decoder size across S-{S, B, L} and sweep the patch size p ∈ {32,16,8} to control the number of image tokens. For each configuration, we train the downstream DiT model under identical settings.

|S-L (𝑝=8)|
|---|
|TC-AE<br><br>S-S (𝑝=8) (dual scaling)|
|Token number scaling<br><br>S-S (𝑝=16)|
|S-B (𝑝=32)<br><br>S-L (𝑝=32)|
|Parameter scaling<br><br>S-S (𝑝=32)|

As shown in Fig. 8, increasing the number of tokens or enlarging the decoder consistently improves generative performance. However, for comparable computational cost (GFLOPs), token number scaling yields stronger improvements than parameter scaling. More importantly, the two scaling dimensions are complementary: combining larger token spaces with higher decoder capacity leads to further gains, demonstrating a synergistic effect between token number and model size.

Figure 8 Synergy of token number scaling and parameter scaling.

##### 4.4 System-Level Comparison

We compare TC-AE with existing tokenizers and downstream generative models on ImageNet 256×256 in Table 4.

Among deep compression tokenizers operating with only 64 tokens, TC-AE significantly outperforms prior methods in generative quality. When paired with the same DiT backbone, TC-AE reduces gFID from 26.44 (DC-AE) and 17.31 (DC-AE-1.5) to 7.16 without classifier-free guidance (CFG), and further to 2.57 with CFG enabled. Notably, this improvement is achieved with

Table 4 System-level comparison of tokenizers and downstream generative models on ImageNet 256×256. The CFG scale is set to 1.4 for TC-AE. †: Results are taken from the original paper; additional comparisons are provided in the appendix.

Tokenizer Latent Tokens rFID ↓ Generative Models Param gFID (w/o CFG) ↓ gFID (w/ CFG) ↓

low compression tokenizer MaskGIT (Chang et al., 2022) 256 2.28 MaskGIT (Chang et al., 2022) 227M 6.18 -

VQGAN (Esser et al., 2021) 256 0.59 LlamaGen (Sun et al., 2024) 3B 9.38 2.18 VAR-Tok. (Tian et al., 2024) 680 1.00 VAR-d20 (Tian et al., 2024) 600M - 2.95

LlamaGen-Tok. (Sun et al., 2024) 256 2.19 LlamaGen (Sun et al., 2024) 1.4B - 3.09 VAE (Rombach et al., 2022) 256 0.53 MAR (Li et al., 2024) 943M 2.35 1.55 VAE (Rombach et al., 2022) 4096 0.27 LDM-4 (Rombach et al., 2022) 400M - 3.60

SD-VAE (sdv, 2023) 256 0.61 DiT (Peebles and Xie, 2023) 600M 9.62 2.27 ViTok S-B (Hansen-Estruch et al., 2025) 256 0.18 DiT (Peebles and Xie, 2023) 675M - 2.45 MAETok (Chen et al., 2025b) 256 0.48 LightningDiT (Yao et al., 2025b) 675M 2.21 1.73 GigaTok-B-L (Xiong et al., 2025) 256 0.51 LlamaGen (Sun et al., 2024) 111M - 3.33

deep compression tokenizer

DC-AE-1.5† (Chen et al., 2025c) 64 0.26 DiT (Peebles and Xie, 2023) 675M 17.31 DC-AE† (Chen et al., 2024) 64 0.26 DiT (Peebles and Xie, 2023) 675M 26.44 -

TC-AE 64 0.35 DiT (Zheng et al., 2025) 675M 7.16 2.57

Table 5 Layer depth for staged token compression.

Layer depth (M) rFID↓ SSIM↑ IS↑ (w/o cfg) IS↑ (w/ cfg)

0 1.27 0.723 69.00 312.45 3 0.93 0.728 66.68 311.45 6 0.90 0.749 71.33 326.47 9 0.92 0.756 62.61 304.26

12 0.90 0.772 52.38 257.93

###### Table 6 Training with other self-supervised learning methods.

SSL method rFID↓ PSNR↑ LPIPS↑ SSIM↑ gFID↓ IS↑

DINO 1.34 23.94 0.101 0.732 20.38 62.12 DINOv2 1.34 24.48 0.103 0.759 28.22 49.19

iBOT 1.27 23.76 0.104 0.723 17.22 69.00

substantially lower computational cost: TC-AE requires only about 164 GFLOPs, compared to 607 GFLOPs for DC-AE. This substantial margin demonstrates that TC-AE effectively mitigates representation collapse and preserves generative structure under deep compression, while being significantly more computationally efficient.Compared with low-compression tokenizers, TC-AE achieves strong generative performance. For instance, TC-AE with only 64 tokens matches or outperforms methods using 256 tokens such as ViTok (2.45) and VAR (2.95), while also exceeding several billion-parameter generator setups in LlamaGen.

Overall, these results demonstrate that TC-AE bridges the performance gap between deep compression and low-compression tokenizers, enabling high-quality image generation while retaining the efficiency benefits of deep compression.

##### 4.5 Design Space of TC-AE

Layer depth for each stage. We study where to apply staged token compression in the ViT encoder by varying the split between the two stages. The encoder contains 12 Transformer layers.

- As shown in Table 5, reconstruction quality (SSIM) consistently improves as M increases. This trend indicates that allocating more layers to process high-resolution tokens benefits reconstruction by preserving finer visual details. In contrast, TC-AE achieves the best generative performance M = 6. We hypothesize that this behavior reflects a trade-off between semantic structure learning and token-to-latent compression. The first stage primarily learns rich semantic structure from the large token space, while the second stage, together with the final bottleneck, acts as a compression buffer between high-resolution tokens and the compact latent. Increasing M strengthens semantic representation learning but reduces the capacity available for compression, which may amplify structural information loss during token-to-latent transformation. At M = 6, this trade-off reaches an optimal balance, resulting in the strongest generative performance. Based on these observations,

we adopt M = 6 as the default configuration in the scale-up experiment.

Joint training with different SSL methods. We further compare different self-supervised objectives for joint tokenizer training. In addition to iBOT, we evaluate other representative selfdistillation methods, including DINO (Caron et al., 2021) and DINOv2 (Oquab et al., 2023), where DINOv2 is also adopted in the concurrent work VTP (Yao et al., 2025a).

- As shown in Table 6, iBOT consistently achieves stronger generative performance than DINO, while maintaining comparable reconstruction quality. Furthermore, joint training with DINOv2 leads to notably weaker gains under our training setting with a batch size of 512 on 8 NVIDIA GPUs, indicating limited effectiveness in the moderate-batch regime. Overall, these results suggest that iBOT offers a lightweight, stable, and practical self-supervised objective for joint tokenizer training under realistic computational budgets.

### 5 Conclusion

In this work, we study deep compression autoencoders from the perspective of the token space. We show that naively scaling the number of image tokens fails to benefit generation due to severe structural information loss at the compression bottleneck. To address this issue, we propose to redistribute token-to-latent compression across encoder stages, enabling effective token number scaling for generation. In addition, we introduce self-supervised objective to further structure the token space, leading to more generative-friendly latent representations. Extensive experiments on ImageNet demonstrate that TC-AE significantly improves both reconstruction and generative performance under deep compression, accelerates diffusion model convergence, and achieves a favorable efficiency–quality trade-off compared to existing tokenizers.

#### 6 Acknowledgement This work was supported by Ant Group Research Intern Program.

### References

stabilityai/sd-vae-ft-ema. https://huggingface.co/stabilityai/sd-vae-ft-ema, 2023. Shane Barratt and Rishi Sharma. A note on the inception score. arXiv preprint arXiv:1801.01973, 2018. Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties

in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022.

Cong Chen, Ziyuan Huang, Cheng Zou, Muzhi Zhu, Kaixiang Ji, Jiajia Liu, Jingdong Chen, Hao Chen, and Chunhua Shen. Hieratok: Multi-scale visual tokenizer improves image reconstruction and generation. arXiv preprint arXiv:2509.23736, 2025a.

Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are effective tokenizers for diffusion models. In Forty-second International Conference on Machine Learning, 2025b.

Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.

Junyu Chen, Dongyun Zou, Wenkun He, Junsong Chen, Enze Xie, Song Han, and Han Cai. Dc-ae 1.5: Accelerating diffusion model convergence with structured latent space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19628–19637, 2025c.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

Philippe Hansen-Estruch, David Yan, Ching-Yao Chung, Orr Zohar, Jialiang Wang, Tingbo Hou, Tao Xu, Sriram Vishwanath, Peter Vajda, and Xinlei Chen. Learnings from scaling visual tokenizers for reconstruction and generation. arXiv preprint arXiv:2501.09755, 2025.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

Wenkun He, Yuchao Gu, Junyu Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Haocheng Xi, Muyang Li, Ligeng Zhu, Jincheng Yu,

et al. Dc-gen: Post-training diffusion acceleration with deeply compressed latent space. arXiv preprint arXiv:2509.25180, 2025. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale

update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010.

Ziyuan Huang, DanDan Zheng, Cheng Zou, Rui Liu, Xiaolong Wang, Kaixiang Ji, Weilong Chai, Jianxin Sun, Libin Wang, Yongjie Lv, et al. Ming-univision: Joint image understanding and generation with a unified continuous tokenizer. arXiv preprint arXiv:2510.06590, 2025.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. arXiv preprint arXiv:2510.15301, 2025.

Jaskirat Singh, Xingjian Leng, Zongze Wu, Liang Zheng, Richard Zhang, Eli Shechtman, and Saining Xie. What matters for representation alignment: Global information or spatial structure? arXiv preprint arXiv:2512.10794, 2025.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

Feng Wang, Yaodong Yu, Wei Shao, Yuyin Zhou, Alan Yuille, and Cihang Xie. Scaling laws in patchification: An image is worth 50,176 tokens and more. In Forty-second International Conference on Machine Learning, 2025.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.

Tianwei Xiong, Jun Hao Liew, Zilong Huang, Jiashi Feng, and Xihui Liu. Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. arXiv preprint arXiv:2504.08736, 2025.

Jingfeng Yao, Yuda Song, Yucong Zhou, and Xinggang Wang. Towards scalable pre-training of visual tokenizers for generation. arXiv preprint arXiv:2512.13687, 2025a.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025b.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

Sukmin Yun, Hankook Lee, Jaehyung Kim, and Jinwoo Shin. Patch-level representation learning for self-supervised vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8354–8363, 2022.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021.

- A TC-AE Implementation Details The detailed training configurations of TC-AE are provided in Table 7.

Self-supervision training. We adopt the same data augmentation pipelines as in the corresponding self-supervised learning methods (Caron et al., 2021; Zhou et al., 2021; Oquab et al., 2023). To ensure stable joint optimization with the reconstruction objective, we reduce the learning rate.

Adversarial training. We adopt the adversarial training setup from RAE (Zheng et al., 2025). The discriminator is built upon a frozen DINO-S/8 backbone (Caron et al., 2021), which has been shown to stabilize training and prevent the decoder from exploiting adversarial patch-level artifacts. All inputs are resized to 224 × 224 before being fed into the discriminator.

Table 7 TC-AE training details.

Component Joing Training Decoder Finetuning Training data ImageNet ImageNet

Training epochs 50 16 Batch size 512 256 Learning Rate schedule Cosine Decay Cosine Decay Optimizer betas (0.9, 0.95) (0.9, 0.95)

Weight decay 0.04 0 Base learning rate 1e-3 2e-4

Minimum learning rate 1e-6 2e-5 Warmup epoch 5 1

Optimizer AdamW AdamW α 0.1 0 λp 1 1 λg 0 0.75

Discriminator start epoch - 6 Adversarial loss start epoch - 8

- B More Ablations and Results

Compression module design in ViT. We compare two alternative designs for the token compression module within the ViT encoder: a pixel-shuffle followed by an MLP, and a lightweight two-layer convolution. All experiments are conducted with a patch size of p = 8. For the pixel-shuffle-based design, token compression is implemented by first merging each 2 × 2 group of neighboring tokens into a single token via pixel shuffle, followed by an MLP that downsamples the channel dimension by a factor of four, resulting in a 4× reduction in token count.

Table 8 Compression module design in ViT.

Compression module rFID↓ PSNR↑ LPIPS↑ SSIM↑ gFID↓ IS↑ Pixel shuffle + MLP 0.89 25.72 0.070 0.808 37.07 38.91

Two-layer convolution 0.75 25.72 0.072 0.807 32.92 43.58

- As shown in Table 8, the convolutional module consistently yields better generative performance in terms of gFID and IS. We conjecture that this improvement stems from aggregating local neighbor-

hood information, as convolution explicitly models spatial locality. Based on these observations, we adopt the two-layer convolution as the default compression module in TC-AE.

Weight of α. We examine the effect of the weighting factor α, which controls the trade-off between reconstruction and SSL objectives during tokenizer training.

Table 9 Trainging weights between reconstruction and self-supervision.

α rFID↓ PSNR↑ LPIPS↑ SSIM↑ gFID↓ IS↑

- 0.1 1.27 23.76 0.104 0.723 17.22 69.00

- 1.0 1.31 24.79 0.092 0.768 27.74 48.92 10 1.26 25.22 0.086 0.789 34.10 40.62

- As shown in Table 9, increasing α improves reconstruction at the expense of generative quality. This reflects a trade-off between emphasizing high-frequency details and learning semantically structured latents. We therefore set α = 0.1 as the default in all experiments.

Expanding latent channel number to 256. Table 10 further examines the effect of increasing the latent channel dimension from c = 128 to c = 256. Overall, both staged token compression (STC) and self-supervised learning (SSL) consistently improve reconstruction and generative performance relative to their corresponding baselines when the channel dimension is expanded.

Despite these gains, models with c = 256 exhibit inferior generative quality compared to their c = 128 counterparts, as indicated by higher gFID and lower IS. This observation suggests that simply increasing the latent channel dimension may aggravate representation collapse. In practice, c = 128 provides a more favorable balance between reconstruction fidelity and latent generatability.

Table 10 Expanding latent channel number to 256.

Channel SSL STC rFID↓ PSNR↑ LPIPS↑ SSIM↑ gFID↓ IS↑

1.33 25.13 0.068 0.794 44.72 33.62 ✓ 0.75 25.72 0.072 0.807 32.92 43.58

128

- ✓ 0.57 27.98 0.037 0.875 39.93 36.49

256

0.73 28.06 0.040 0.879 47.33 30.73 ✓ 0.57 27.98 0.037 0.875 39.93 36.49

- ✓ 1.23 24.99 0.091 0.780 24.27 55.20

Additional comparison with DC-AE. We download the official checkpoint1 from the authors of DC-AE (Chen et al., 2024) and use it as the tokenizer. We then train diffusion models using DC-AE and TC-AE under same configurations. The latent shape of TC-AE is kept the same as DC-AE for a fair comparison. The results are reported in Table 11. TC-AE consistently achieves better generative performance than DC-AE. These results further demonstrate that the proposed staged token compression and joint self-supervised training effectively improve latent generatability.

Table 11 Additional comparison with DC-AE. The CFG scale is set to 1.4.

Tokenizer Latent shape GFLOPS rFID↓ gFID (w/o CFG) ↓ gFID (w/ CFG) ↓

DC-AE 8 × 8 × 32 607 0.71 15.90 5.59 TC-AE 8 × 8 × 32 164 0.81 10.72 4.77

1https://huggingface.co/mit-han-lab/dc-ae-f32c32-in-1.0-256px

Additional visualization. We provide more generation results in Fig. 9.

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

Figure 9 Image generated by diffusion models using TC-AE at 256 × 256 resolution.

