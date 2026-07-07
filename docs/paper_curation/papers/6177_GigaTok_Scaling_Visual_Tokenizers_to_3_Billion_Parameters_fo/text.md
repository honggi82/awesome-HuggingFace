## GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters for Autoregressive Image Generation

Tianwei Xiong1 Jun Hao Liew2 Zilong Huang2 Jiashi Feng2 Xihui Liu1† 1The University of Hong Kong 2ByteDance Seed Project page: https://silentview.github.io/GigaTok/

# arXiv:2504.08736v2[cs.CV]24Aug2025

#### Abstract

Tokenizer: rFID ↓

Tokenizer Params 136M 622M 2.9B

###### ReconstructionGeneration

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

In autoregressive (AR) image generation, visual tokenizers compress images into compact discrete latent tokens, enabling efficient training of downstream autoregressive models for visual generation via next-token prediction. While scaling visual tokenizers improves image reconstruction quality, it often degrades downstream generation quality—a challenge not adequately addressed in existing literature. To address this, we introduce GigaTok, the first approach to simultaneously improve image reconstruction, generation, and representation learning when scaling visual tokenizers. We identify the growing complexity of latent space as the key factor behind the reconstruction vs. generation dilemma. To mitigate this, we propose semantic regularization, which aligns tokenizer features with semantically consistent features from a pre-trained visual encoder. This constraint prevents excessive latent space complexity during scaling, yielding consistent improvements in both reconstruction and downstream autoregressive generation. Building on semantic regularization, we explore three key practices for scaling tokenizers: (1) using 1D tokenizers for better scalability, (2) prioritizing decoder scaling when expanding both encoder and decoder, and (3) employing entropy loss to stabilize training for billion-scale tokenizers. By scaling to 3 billion parameters, GigaTok achieves stateof-the-art performance in reconstruction, downstream AR generation, and downstream AR representation quality.

Original image Better reconstruction with larger tokenizer

111M AR model with

Tokenizer Params 136M 622M 2.9B

different tokenizers: gFID ↓

|Tokenizer Fails to Converge|
|---|

[Figure 5]

[Figure 6]

AR generation with Baseline Tokenizer

Worse generation with larger tokenizer

[Figure 7]

[Figure 8]

[Figure 9]

AR generation with GigaTok

Better generation with larger tokenizer

Figure 1. Reconstruction vs. generation dilemma: Naively scaling visual tokenizers achieves better reconstruction but degrades downstream autoregressive (AR) generation. In contrast, GigaTok achieves better performance for both reconstruction and generation as tokenizers scale up.

with image reconstruction supervision, while the AR generator models the distribution of these discrete tokens through next-token prediction. The image tokenizer plays a pivotal role in AR visual generation, providing a compact and expressive latent space that enables effective generative modeling by downstream AR models.

Despite its pivotal role, scaling of visual tokenizer is rarely explored in the literature. In fact, unlike the downstream AR models whose scalability has been widely validated [12, 30, 60, 62], scaling the visual tokenizer presents a significant challenge. Specifically, there exists a reconstruction vs. generation dilemma, where scaling tokenizer improves reconstruction fidelity but degrades downstream generation quality, as shown in Fig. 1. This dilemma is also observed in prior works [13, 21]. In this work, we seek to overcome this limitation and explore strategies for effectively scaling tokenizers to enhance both reconstruction and generation performance.

#### 1. Introduction

Autoregressive (AR) language models (LM) have emerged as a promising approach for visual generation [15, 50, 66, 69], driven by their proven scalability [2, 5, 14, 19, 37, 51, 52, 54, 55] and the potential for unified multimodal modeling [12, 45, 62]. The AR image generation framework consists of a visual tokenizer and a downstream AR generator. The tokenizer encodes images into discrete tokens, trained

To investigate the root cause of this dilemma, we propose

† Corresponding Author.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Figure 2. The 2.9B GigaTok achieves SOTA autoregressive image generation with a 1.4B AR model on ImageNet 256×256 resolution.

tecture, asymmetric encoder-decoder scaling, and entropy loss for billion-scale tokenizers.

an AR probing scheme that trains a lightweight downstream generative AR model to monitor the tokenizer’s training process. Surprisingly, we find that as tokenizers scale, the downstream AR model struggles more to learn the resulting token distribution, as evidenced by the increasing AR generation loss. This suggests that the larger tokenizers produce a more complex token space, making it increasingly difficult for AR models to learn effectively.

• Our GigaTok is the first tokenizer scaled to 3B, achieving state-of-the-art reconstruction, downstream AR generation, and downstream AR representation on ImageNet.

#### 2. Related Work

Image tokenizers. Image tokenizers map image inputs into discrete [15, 56, 66] or continuous [28] tokens which can be modeled by downstream generative models. For discrete tokenizers, Vector Quantization (VQ) [15, 56, 66] is dominantly adopted. Recently, new quantization methods [49, 69, 75, 76] have also been proposed for better scaling of codebook size. However, how to properly scale up tokenizer models is insufficiently studied in existing literature. ViT-VQGAN [66] and TiTok [70] utilize transformer architecture to enable convenient scaling of tokenizers, but end up training their best generative models on smaller tokenizer versions. A concurrent work, ViTok [76], suggests de-prioritizing VAE scaling due to its less predictable effect for downstream diffusion models. We observe a similar reconstruction vs. generation dilemma in scaling discrete tokenizers, and provide our analysis and solution to it.

To address this challenge, we introduce pre-trained visual representation models (e.g. DINOv2 [43]) to regularize tokenizers. Specifically, we leverage a semantic regularization loss during tokenizer training, encouraging high similarity between tokenizer features and the pre-trained model features. Such regularization helps constrain the latent space complexity, preventing the tokenizer from learning overly complicated latent token dependencies that hinder downstream AR generative modeling. Moreover, we design a vector-quantized (VQ) tokenizer with a hybrid CNNTransformer architecture as the backbone, suitable for both

- 1D and 2D tokenizers, and explore best practices for scaling tokenizers: (1) 1D tokenizers exhibit better scalability compared to 2D tokenizers; (2) Asymmetric model scaling, prioritizing decoder scaling over encoder scaling, proves effective; (3) Entropy loss [69] becomes crucial for convergence when training tokenizers with billion-level parameters. With our semantic regularization and three key scaling strategies, we effectively scale GigaTok to 3 billion parameters, overcoming the reconstruction vs. generation dilemma.

Autoregressive Visual Generation. Autoregressive visual generative models [33, 38, 40, 49, 50, 56, 58, 60, 66] follow the next-token-prediction (NTP) approach of LLMs, enabling the leverage of advancements in LLMs and simplifying the path to unified multi-modal generation. Other variants utilize visual-specific paradigms such as mask image modeling [8, 61, 69, 70] and next-scale-prediction [36, 53] for better performance. We reveal that scaling tokenizers helps NTP AR models to be comparable to these variants.

We summarize our contributions as follows:

- • We identify that the reconstruction vs. generation dilemma in tokenizer scaling stems from increased latent space complexity in larger tokenizers. To address this, we propose semantic regularization, effectively mitigating the dilemma and enabling tokenizer scaling.
- • We explore best practices for scaling tokenizers, including 1D tokenizers with hybrid CNN-Transformer archi-

Semantic Guidance for Visual Generative Models and Tokenizers. The guidance from visual foundation models [7, 23, 43, 46, 72] has been used to improve training

convergence speed and quality [65, 71] of visual generative models, as well as enhancing representation quality or downstream performance of visual tokenizers [9, 10, 18, 36, 41, 59, 63–65, 68, 73, 76, 77]. REPA [71] presents impressive performance improvements brought by a simple representation alignment strategy, and recently, VA-VAE [65] shows the significant benefits of semantic guidance to the reconstruction-generation Pareto Frontier of VAEs. Different from existing work, GigaTok novelly reveals the critical role of semantic regularization for resolving the reconstruction vs. generation dilemma in scaling visual tokenizers.

#### 3. Pilot Study

We first introduce AR Probing as a proxy to effectively monitor the tokenizer’s effectiveness for downstream generation (Sec 3.1), followed by a pilot experiment that investigates the reconstruction vs. generation challenges when naively scaling visual tokenizers (Sec 3.2).

##### 3.1. AR Probing for Tokenizer Evaluation

In autoregressive visual generation, the training of the tokenizer and downstream AR model are performed in separate stages. In the first stage, a visual tokenizer is trained to compress images into discrete tokens, optimized with reconstruction objective. In the second stage, the downstream generative model is trained based on the discrete tokens from the pre-trained tokenizer. However, a tokenizer that performs well in terms of reconstruction fidelity in the first stage may not necessarily lead to better performance for downstream generative models. Thus, it is crucial to evaluate the effectiveness of the trained tokenizers for downstream generation alongside its reconstruction quality.

Despite its importance, assessing how a tokenizer influences downstream generation models can be computationally expensive. For example, sufficiently training a 343M parameter downstream AR generator takes 170 hours on 64 V100 GPUs. To address this challenge, we introduce AR Probing, inspired by Linear Probing in representation learning literature [11, 23]. The key idea is to use the performance of a small AR model as a proxy to reflect the performance trends of large-scale AR models.

Specifically, we use the tokenizer to train a small Llamastyle model [50, 54] (111M parameters) for 50 epochs, and evaluate its gFID [24], validation loss, and linear probing accuracy [11, 23] for a fair comparison between different tokenizers. Training the proposed AR Probing model for evaluating tokenizers is 10× more efficient than training the original 343M downstream AR model. Our experiments in Sec. 5.1 (Fig. 6) demonstrate that the trends observed with AR Probing align with the performance of the large-scale AR models after sufficient training.

gFID. The generation FID [24] of AR probing indicates the overall image generation performance of the two-stage

###### 1.6 Tokenizer: rFID

###### AR Probing: Validation Loss

AR Probing: gFID

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

- 10
- 11

8.6

1.4

8.4

1.2

8.2

200 400 600

200 400 600

200 400 600

Tok. Params (M)

Tok. Params (M)

Tok. Params (M)

Figure 3. Scaling trend for vanilla 1D tokenizers. As the model size increases, the reconstruction quality of vanilla tokenizers improves but the downstream AR Probing gFID consistently degrades. The increasing AR Probing validation loss indicates that scaling vanilla tokenizers results in a more complex latent space, making it difficult for AR models to learn effectively.

framework. It reflects both the reconstruction fidelity of the tokenizer and how well the downstream AR probing model can learn the dependency of the visual tokens (i.e., learnability of the token distribution).

Validation loss. We use the validation loss of the AR probing model to measure the learnability of the latent tokens as a disentangled factor. The validation loss is calculated as an average of the token-wise cross-entropy loss in the next-token-prediction paradigm on ImageNet [48] 50k validation set. With the same vocabulary size, the same number and structure of visual tokens, and the same AR probing model, larger validation loss indicates a latent space that is more difficult for the AR model to learn. Therefore, we use validation loss to reflect the latent space complexity and learnability for AR models.

Linear probing accuracy. Beyond visual generation quality, we also investigate whether scaling tokenizers will lead to better visual representations of AR models, which may provide inspiration for future research in unified multimodal understanding and generation with AR models. To assess the representation quality, we adopt the standard practice [11, 23] of linear probing accuracy using features from the middle Transformer layer of the AR probing model.

##### 3.2. Naively Scaling Tokenizers Does Not Work

To study the challenges when naively scaling visual tokenizers, we train three vector-quantized tokenizers1 on ImageNet [48] at 256×256 resolution with increasing model sizes. As shown in Fig. 3, as the tokenizer size increases, although the reconstruction quality (rFID) consistently improves, the AR generation performance (gFID) significantly degrades. This highlights the reconstruction vs. generation dilemma in tokenizer scaling. Moreover, we observe that the validation loss of AR Probing consistently increases as the tokenizers scale, indicating that larger tokenizers lead to complicated token dependencies that are more difficult for the AR model to learn. This observation motivates us to design the semantic regularization to constrain the latent space complexity of the tokenizer and therefore break the

The tokenizer architectures are described in Sec. 4.1

Transformer Decoder

reconstruction vs. generation dilemma in Sec. 4.2.

Input Image

Reconstructed Image

TransformerEncoder

#### 4. GigaTok

CNNDecoder

CNNEncoder

[Figure 22]

[Figure 23]

Quantizer

𝑙+1Layer

𝑙Layer

In this section, we introduce the model structure and training strategies for our scalable visual tokenizer, GigaTok. In Sec. 4.1, we present a tokenizer backbone supporting 1D and 2D token structures, and discuss the asymmetric scaling strategies for the encoder and decoder. In Sec. 4.2, we introduce semantic regularization, which breaks the reconstruction vs. generation dilemma by regularizing the complexity of the latent space with pre-trained visual representations. In Sec. 4.3, we show how entropy loss [69] facilitates the convergence of billion-scale tokenizers.

Pre-trained DINOv2

Semantic Regularization

MLP

[Figure 24]

Figure 4. GigaTok architecture and semantic regularization. Top: We use a hybrid CNN-Transformer design for our visual tokenizer. The transformer layers are implemented with ViT for 2D tokenizer and Q-Former for 1D tokenizer. Bottom: We use a frozen DINOv2 [43] image encoder for semantic regularization.

##### 4.1. Architecture

The CNN [32] architectures have been the dominant choices for image tokenizers [15, 40, 69, 76] due to their effectiveness in capturing fine-grained local details. Yet, Transformers are more scalable architectures with less inductive bias. Thus, we design a vector quantized tokenizer backbone with a hybrid architecture that combines CNN [15, 32] and Transformer [6, 13, 57] for encoder and decoder (Fig. 4). Specifically, our encoder consists of a series of CNN blocks that progressively downsamples the input image by a factor of p, followed by Transformer layers and a vector quantizer to produce discrete latent codes. Similarly, our decoder consists of multiple Transformer layers, followed by CNN decoders which upsamples the features to obtain the reconstructed image2. Our tokenizer architecture can be adapted to both 1D and 2D tokenizers by using different Transformer designs introduced in the next two paragraphs.

the decoders are always larger than the encoders. In practice, we maintain the same and fixed size for the CNN encoder/decoder and only increase the depth and width of the Transformer modules for scaling.

##### 4.2. Semantic Regularization

In our pilot study (Sec. 3.2), the latent space complexity significantly increases as the tokenizer scales, which potentially leads to worse downstream AR generation for larger tokenizers. We hypothesize that larger tokenizers tend to capture excessive fine-grained low-level details for better reconstruction, resulting in overly complex latent token distributions, which makes it harder for AR models to learn the token dependencies effectively.

- 2D tokenizers with ViT. For 2D tokenizers, the Transformers in both tokenizer encoder and decoder are implemented by ViT [13] architecture. 2D structures of the latent features and tokens are preserved throughout the tokenizer.

To address this, we introduce semantic regularization to guide the tokenizer to encode a more semantically consistent latent space, which is less complex and easier for downstream generative modeling. Specifically, we introduce a simple semantic regularization term alongside the tokenizer training objective. The regularization aligns the intermediate features of the tokenizer decoder with the feature representations extracted from pre-trained frozen DINOv2 [43].

1D tokenizers with Q-Former. For 1D tokenizers, we implement the Transformer modules in both encoder and decoder as Q-Formers [6, 34]. The Q-Former in the encoder employs 1D queries, transforming 2D input features into 1D latent tokens. The Q-Former in the decoder utilizes 2D queries to transform 1D latent tokens back to 2D features, which are then passed to the CNN decoder to reconstruct images. The 1D tokenizers remove the 2D inductive bias and demonstrate better scalability than 2D tokenizers in our experiments (Sec. 5.5).

Mathematically, let fdec,l be the output feature of the lth layer of the Transformer decoder, fDINO be the semantic features of a pretrained image encoder (here DINOv2B [43]). The semantic regularization can be represented as:

N

1 N

Asymmetric encoder-decoder scaling. Since the decoder faces the more challenging task of reconstructing images from lossy latent codes, we adopt an asymmetric design for more efficient parameter allocation. Specifically, we scale both the encoder and decoder, while ensuring that

sim fndec,l,ϕ(fnDINO) (1)

Lreg =

n=1

where N is the batch size, n is the image index, sim(·,·) is a cosine similarity function, and ϕ(·) is an MLP that projects decoder feature fdec,l to match the channel dimension of fDINO. When training VQ tokenizers, we add the semantic regularization to the original VQGAN [15, 50] objectives:

Throughout this work, we use downsample ratio p = 16, codebook dimension D = 8, and codebook size 16384 by default.

w/o Entropy Loss w/ Entropy Loss

###### Perceptual Loss

Codebook Usage

- 0

- 1

0.50

0.25

0 20 40 60 80

0 20 40 60 80

Tok. Training Iters (k)

Tok. Training Iters (k)

Figure 5. Training curves for 2.9B XL-XXL tokenizers with and without entropy loss. A 2.9B tokenizer does not converge without entropy loss. The entropy loss encourages high codebook usage and stabilizes training loss.

Ltotal = Lvqgan + λLreg, (2)

and we empirically set λ = 0.5 in this work. Here Lvqgan is a combination of multiple losses , including Lrecon, the l2 reconstruction loss on image pixels, Lpercp, the perceptual loss [27, 74], LGAN, PatchGAN [26] adversarial loss, and LVQ [15, 66] the VQ codebook loss.

- 4.3. Entropy Loss for Billion-Level Tokenizers

When training a 2.9B tokenizer, we find that using the same training recipe as the 622M tokenizer leads to convergence failure for both perceptual loss and reconstruction loss, and consistently low codebook usage. We hypothesize that low codebook usage accounts for the convergence difficulty. To address this, we incorporate entropy penalty [67, 69] to encourage higher codebook utilization:

Lentropy = Ez [H(zˆ|z)] − H(zˆ) (3)

where H(·) denotes the Shannon entropy, z ∈ RD is the input for quantizer to be quantized to ˆz = ci ∈ RD and ci is the i-th codebook vector. Ez [H(zˆ|z)] penalizes the uncertainty in quantization to reduce quantization error, and −H(zˆ) encourages the codebook vectors to be selected more uniformly across the entire codebook. The detailed derivation can be found in our supp. We find that the entropy penalty addresses the convergence difficulty of large tokenizers. As shown in Fig. 5, introducing entropy loss to the 2.9B tokenizer enables the codebook usage to quickly reach a high level, and the loss converges properly3.

- 5. Experiments

- 5.1. Settings

For scaling up visual tokenizers, we follow the architecture configurations for the Transformers in GigaTok tokenizers as summarized in Tab. 1. We evaluate the tokenizers from three perspectives: reconstruction, downstream AR generation, and downstream AR representation quality. We use

We take perceptual loss as an example, and reconstruction loss shows a similar pattern

Type Enc./Dec. Params. Blocks Heads Dim.

- 1D Tok.

S 26M 6 8 512 B 115M 12 12 768 L 405M 24 16 1024

XL 948M 36 20 1280 XXL 1870M 48 24 1536

- 2D Tok.

S 19M 6 8 512 B 86M 12 12 768 L 329M 24 16 1024

Table 1. Architectures of the transformer variants for tokenizer encoder/decoder parts in our experiments. We use QFormer [6, 34] for 1D tokenizers and ViT [13] for 2D tokenizers.

gFID: AR Probing v.s. LlamaGen-XL

Lin Acc. : AR Probing v.s. LlamaGen-XL

LlamaGen-XL

LlamaGen-XL

| | | | |
|---|---|---|---|
| | | | |
| | |R2 = 0.99| |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | |R2 = 0|.80|

2.50

0.70

2.25

0.68

5.0 5.5 6.0

0.58 0.60 0.62 0.64 0.66

AR Probing

AR Probing

Figure 6. Correlation between AR Probing Performance and Larger AR models. For 3 tokenizers: S-S, S-L, and B-L, we present that as the tokenizer improves, the performance improvements of AR Probing correlate to the performance improvements of larger AR models. Therefore, the AR Probing can effectively indicate how the tokenizer affects downstream larger AR models with limited computational costs.

rFID and LPIPS [74] to evaluate reconstruction fidelity, gFID to evaluate generation performance, and linear probing to evaluate the representation quality of the downstream AR model. Our downstream AR models are LlamaGen [50] with 1D absolute positional embedding. Our scaling experiments (Sec. 5.2) and ablation study (Sec. 5.3) use AR Probing (111M AR model described in Sec.3.1) validation loss, gFID, and linear probing to reflect the learnability of tokens, generation performance, and representation quality, respectively. While in the system-level comparison (Sec. 5.4), we train larger 1.4B AR models for comparison with previous work. More details are in the supplementary material.

Effectiveness of AR Probing. As shown in Fig. 6, AR Probing performances including gFID and linear probing accuracy align with the larger LlamaGen-XL [50] model results. Therefore, we use AR Probing throughout the following experiments except for the system-level comparison.

##### 5.2. Scaling with Semantic Regularization

We demonstrate that our proposed semantic regularization resolves the reconstruction vs. generation dilemma in scaling tokenizers.

Model scaling with semantic regularization. Results are shown in Fig. 7. (1) Semantic regularization improves the reconstruction fidelity, indicated by lower rFID. (2) More importantly, the AR Probing validation loss and gFID de-

w/o semantic regularization w/ semantic regularization

Tokenizer: rFID

###### Tokenizer: LPIPS

###### AR Probing: Validation Loss

###### AR Probing: g

###### AR Probing: gFID

###### AR Probing: in Acc

AR Probing: Lin Acc

8.75

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

6.0

1.50

0.220

0.660

- 10
- 11

0.425

8.50

1.25

0.215

5.5

8.25

0.640

0.400

0.210

1.00

8.00

5.0

0.375

0.620

0.205

200 400 600

200 400 600

200 400 600

200 400 600

200 400 600

Tok. Params (M)

Tok. Params (M)

Tok. Params (M)

Tok. Params (M)

Tok. Params (M)

- Figure 7. Scaling trends of tokenizers for reconstruction, downstream generation and representation quality with and without semantic regularization. By semantic regularization, GigaTok resolves the reconstruction vs. generation dilemma for tokenizer scaling in contrast to the vanilla version without semantic regularization. Moreover, GigaTok consistently improves the representation quality of downstream AR models by scaling up visual tokenizers. Note that in the last two figures, the red and blue curves correspond to different scales on the y-axis.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

original w/o sem. reg. w/ sem. reg. original w/o sem. reg. w/ sem. reg.

- Figure 8. Visualization of tokenizer features with and without semantic regularization. We compute PCA among the tokenizer features of a group of images of the same “golden retriever” class and visualize the first 3 PCA components. We observe that the latent space of vanilla tokenizers shows inconsistent features both within a single image or across multiple semantically similar images. In contrast, GigaTok encodes images with semantic consistency and thus reduces the latent space complexity for AR models.

Enc./Dec. Size rFID↓ LPIPS↓ gFID↓ Lin Acc.↑

B-S 0.98 0.221 6.56 64.5 S-B 0.94 0.214 5.65 59.8

S-L 0.83 0.206 5.19 60.6 B-L 0.81 0.206 4.82 66.9

Table 2. The results for scaling encoder/decoder. Prioritizing the scaling of decoders benefits downstream generation more than scaling encoders (S-B v.s. B-S). But scaling encoders can still bring significant improvements (S-L v.s. B-L).

2D Tok. 1D Tok.

###### AR Probing: gFID

AR Probing: Lin Acc.

###### Tokenizer: rFID

grades for larger tokenizers without semantic regularization, showing the reconstruction vs. generation dilemma. The dilemma is addressed with semantic regularization, evidenced by the relatively constrained validation loss and consistently decreasing gFID. (3) The Linear Probing results show that semantic regularization helps AR models to learn better representations as the tokenizer model scales up.

6.0

1.2

0.65

5.5

1.0

0.60

5.0

0.8

200 400 600

200 400 600

200 400 600

Tok. Params (M)

Tok. Params (M)

Tok. Params (M)

Figure 9. Scalability comparison for 1D and 2D tokenizers. Using the same training setting, 1D tokenizers shows better reconstruction (rFID) and downstream representation quality (AR Probing: Lin Acc.). For downstream generation (gFID), 1D tokenizers present a steeper improving trend than 2D tokenizers.

Visualization for the tokenizer feature space. We visualize the first 3 PCA components of the tokenizer features from the first Transformer decoder layer for a group of images. As shown in Fig. 8, we find the vanilla tokenizer encodes a latent space with limited semantic consistency, which potentially impairs its learnability for downstream AR models. In contrast, GigaTok presents semantically consistent patterns (Fig. 8), indicating a meaningful and consistent latent space.

epochs. Our results show that scaling decoders, rather than encoders, leads to greater improvements in both reconstruction and downstream generation, suggesting that decoder scaling should be prioritized.

Scaling tokenizer encoder is also important. While prioritizing the scaling of tokenizer decoders yields significant benefits, we also find that scaling tokenizer encoders can further enhance downstream models. In Tab. 2, we show that a B-L tokenizer gains significant improvements compared to an S-L tokenizer. Therefore, we recommend scaling both encoders and decoders while maintaining a larger decoder than the encoder for optimal performance.

##### 5.3. Asymmetric 1D Tokenizer is More Scalable

Tokenizer decoder deserves more parameters. To determine whether the decoder or encoder should be prioritized when scaling up, we compare S-B4 and B-S tokenizers in Tab. 2, both trained under the same setting for 100

X-Y tokenizer denotes X-sized encoder and Y-sized decoder. For example, S-B indicates Small encoder-Base decoder structure

Tokenizer Tok. Type/Param. #Tokens rFID↓ Generator Model/Param. Type gFID↓ Acc.↑ Continuous token modeling

VAE [47] KL† 55M 4096 0.27 LDM-4 [47] 400M Diff. 3.60 -

DiT-XL/2 [44] 675M Diff. 2.27 SiT-XL/2 [42] 675M Diff. 2.06 SiT-XL/2 + REPA [71] 675M Diff. 1.42 74.6

SD-VAE [1] KL† 84M 1024 0.62

VA-VAE [65] KL 70M 256 0.28 LightningDiT [65] 675M Diff. 1.35 VAE [35] KL 66M 256 0.53 MAR-H [35] 943M AR+Diff. 1.55 60.0⋄

###### Discrete token modeling

VQGAN [8] VQ 66M 256 2.28 MaskGIT [8] 227M Mask. 6.18⋆ TiTok-S [70] VQ 72M 128 1.71 MaskGIT-UViT-L [4, 8] 287M Mask. 1.97 TiTok-L [70] VQ 641M 32 2.21 MaskGIT-ViT [8] 177M Mask. 2.77 -

BiGR-XXL-d32 [22] 1.5B AR+Diff 2.36 -

B-AE-d32 [22] LFQ 66M 256 1.69

BiGR-XL-d32 [22] 799M AR+Diff - 69.8 VAR-Tok. [53] MSRQ† 109M 680 1.00‡ VAR-d24 [53] 1.0B VAR 2.09 -

VAR-d30 [53] 2.0B VAR 1.92 ImageFolder [36] MSRQ 176M 286 0.80‡ ImageFolder-VAR [36] 362M VAR 2.60 VQGAN [15] VQ 23M 256 4.98 Taming-Tran. [15] 1.4B AR 15.78⋆ ViT-VQGAN [66] VQ 64M 1024 1.28 VIM-Large [66] 1.7B AR 4.17⋆ RQ-VAE [33] RQ 66M 256 3.20 RQTran. [33] 3.8B AR 7.55⋆ Open-MAGVIT2 [40] LFQ 133M 256 1.17 Open-MAGVIT2-XL [40] 1.5B AR 2.53 IBQ [49] IBQ 128M 256 1.37 IBQ-XXL [49] 2.1B AR 2.05 LlamaGen-Tok. [50] VQ 72M 256 2.19

LlamaGen-L [50] 343M AR 3.81 40.5⋄

LlamaGen-XXL [50] 1.4B AR 3.09 LlamaGen-Tok. [50] VQ 72M 576 0.94 LlamaGen-XXL [50] 1.4B AR 2.34 -

GigaTok-B-L VQ 622M 256 0.51‡ LlamaGen-B (1d) [50] 111M AR 3.33 67.7 GigaTok-S-S VQ 136M 256 1.01 LlamaGen-B (1d) [50] 111M AR 4.05 62.6 GigaTok-S-B VQ 232M 256 0.89 LlamaGen-B (1d) [50] 111M AR 3.83 62.9

LlamaGen-B (1d) [50] 111M AR 3.26 67.6 LlamaGen-XXL (1d) [50] 1.4B AR 2.03⋆ 69.4

GigaTok-B-L VQ 622M 256 0.81

LlamaGen-B (1d) [50] 111M AR 3.15 72.0 LlamaGen-XXL (1d) [50] 1.4B AR 1.98⋆ 74.0

GigaTok-XL-XXL VQ 2.9B 256 0.79

- Table 3. System-level comparison for tokenizers and downstream generation models on ImageNet 256×256. For gFID, we present the lowest value between w/ or w/o CFG scenarios. †: Training set includes data besides ImageNet. ‡: Using frozen DINO [7] for discriminator, which largely improves rFID. ⋆: Without classifier-free-guidance. ⋄: Data from BiGR [22].

1D tokenizers are more scalable than 2D tokenizers. We train S-S, S-B and B-L 1D/2D tokenizers with the same setting with semantic regularization. As shown in Fig. 9, 1D tokenizers consistently achieve better rFID and AR Probing linear probing accuracy than 2D tokenizers. For AR Probing gFID, the 1D tokenizers exhibit a steeper scaling trend, eventually surpassing 2D tokenizers as the model scales. We attribute the superior scalability of 1D tokenizers to the reduced inductive bias.

##### 5.4. System-level Comparison

Experiment Settings. Using GigaTok for tokenization, we scale the training of LlamaGen [50] AR models on 256 × 256 ImageNet training set for 300 epochs to compare with other methods. We do not use AdaLN [44, 53] as it is specific for class-conditional generation. We provide

the results of a B-L tokenizer trained with DINO discriminator [36, 53] to fairly compare rFID. But in practice we find DINO discriminator provides limited improvement for LPIPS and may affect the training stability of billion-scale tokenizers. Therefore, we exclude it from our main design.

Results. As shown in Tab. 3, our 2.9B GigaTok achieves state-of-the-art reconstruction performance (rIFD) among all discrete tokenizers. Furthermore, with our 2.9B tokenizer, the downstream 1.4B AR model achieves stateof-the-art image generation performance (gFID) among LLM-style autoregressive next-token-prediction models. VAR [53] predicts images with next-scale prediction rather than next-token-prediction, which is less compatible with language models. Our model achieves comparable gFID to VAR [53] with a simple LLM-style downstream AR genera-

Decoder\AR Model Size B L XXL

B 3.7% 2.3% 1.3% L 11.2% 7.0% 3.4%

XXL 32.4% 20.3% 9.9%

- Table 4. Ratio of time consumptions for tokenizer decoding during image generation. When we use a 2.9B XLXXL tokenizer for a 1.4B LlamaGen-XXL AR model, the tokenizer decoding only takes 9.9% of the total inference time.

tor without incorporating vision-specific designs like VAR. Moreover, this 1.4B AR model trained on the 2.9B tokenizer achieves state-of-the-art linear probing accuracy via visual generative pretraining5. This indicates that our GigaTok helps the downstream generation model to learn better representations. The high-quality representation learned from generative pre-training may also help unify generation and understanding for future native multimodal models.

- 5.5. Discussion and Ablation Study

Align. Layer l rFID↓ LPIPS↓ gFID↓ Lin Acc.↑

- 2 1.06 0.224 6.26 63.4

- 3 1.01 0.223 6.10 61.9

- 4 1.07 0.223 6.07 58.6

- Table 5. Layer l for semantic regularization (S-S tokenizer). Smaller l brings better downstream AR model representations but can sacrifice reconstruction and downstream generation quality. We choose l=3 by default for more balanced performance.

Sem. Enc. rFID↓ LPIPS↓ gFID↓ Lin Acc.↑

CLIP [16, 46] 0.91 0.210 6.35 61.4 SigLIP [72] 0.92 0.210 6.20 56.7 DINOv2-B [43] 0.85 0.212 5.55 64.4

- Table 6. Ablation study for the choice of pretrained semantic encoders (S-B tokenizer). DINOv2-B delivers the best performance among all models.

Discussion on generation costs. When generating an image, AR models take multiple passes to predict tokens, while tokenizers only need one forward pass. Therefore, the time consumption for decoding tokens to images is relatively small compared to AR models. We record the ratio of time spent on tokenizer decoding for different tokenizer/AR models in Tab. 4. For a 1.4B AR model, our largest 2.9B tokenizer takes only ∼10% of the total inference time.

REPA [71] achieves better representation by directly distilling pretrained representations to the generation model, which is not a fair comparison with ours as we do not leverage the supervision for AR training.

Sem. Reg. λ rFID↓ LPIPS↓ gFID↓ Lin Acc.↑

0.25 1.28 0.226 6.27 57.0 0.50 1.22 0.228 6.39 58.6

- 0.75 1.27 0.236 6.29 58.6

- 1.00 1.38 0.239 6.27 62.5

Table 7. Ablation Study for the semantic regularization weight (S-S tokenizer). A strong semantic regularization weight leads to worse reconstruction but better downstream representation. We choose λ = 0.5 by default for more balanced performance.

Searching the best layer for semantic regularization. We search l, the layer’s index in the Transformer decoder before intermediate features are extracted to calculate semantic regularization in Eq. 1. As shown in Tab. 5, varying l presents a trade-off between gFID and the Lin Acc. for AR Probing. Smaller l means stricter regularization for the latent space so that the downstream generation models learn better representation. However, smaller l also sacrifices generation quality. We choose l = 3 for a more balanced rFID, gFID, and linear probing accuracy for all tokenizers.

Exploring pretrained semantic encoder choices. We compare CLIP (DFN) [16, 46], SigLIP-400M [72] and DINOv2-B [43] as the source of semantic regularization for S-B tokenizers. As shown in Tab. 6, utilizing DINOv2-B as the semantic encoder for regularization produces the best tokenizer for reconstruction, downstream class conditional generation and representation quality.

Exploring weights for semantic regularization. We study the effects of different regularization weights λ (Eq. 2), from 0.25 to 1.00. As shown in Tab. 7, a large λ (0.75,1.00) will damage the reconstruction quality but benefits the linear probing accuracy, whereas smaller λ (0.25) results in suboptimal rFID and linear probing accuracy. We choose the more balanced λ = 0.5 as a default for all tokenizers.

#### 6. Conclusion

In this work, we study and address the reconstruction vs. generation dilemma for scaling visual tokenizers. We identify that the dilemma stems from increasing latent space complexity in larger tokenizers. We propose semantic regularization to effectively regularize the tokenizer latent space by injecting pre-trained representations to align with tokenizer features in training. The semantic regularization, together with several key practices we explored, lead to the first 3B tokenizer, GigaTok, that achieves state-of-the-art reconstruction, downstream AR generation, and downstream AR representation quality. Please refer to discussions on limitations and future work in supplementary materials.

#### Acknowledgments

This work is partially supported by the National Nature Science Foundation of China (No. 62402406).

The authors also sincerely thank Qihang Yu and LiangChieh Chen for their valuable discussions during the development of GigaTok.

#### References

- [1] stabilityai/sd-vae-ft-ema. https://huggingface.co/ stabilityai/sd-vae-ft-ema, 2023. 7
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [3] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, O˘guzhan Fatih Kar, Elmira Amirloo, Alaaeldin ElNouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. arXiv preprint arXiv:2502.13967, 2025. 3
- [4] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023. 7
- [5] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling opensource language models with longtermism. arXiv preprint arXiv:2401.02954, 2024. 1
- [6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 4, 5, 1
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 2, 7, 3
- [8] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022. 2, 7
- [9] Hao Chen, Ze Wang, Xiang Li, Ximeng Sun, Fangyi Chen, Jiang Liu, Jindong Wang, Bhiksha Raj, Zicheng Liu, and Emad Barsoum. Softvq-vae: Efficient 1-dimensional continuous tokenizer. arXiv preprint arXiv:2412.10958, 2024. 3
- [10] Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are effective tokenizers for diffusion models. arXiv preprint arXiv:2502.03444, 2025. 3
- [11] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pre-

- training from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020. 3
- [12] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 1

- [13] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 4, 5
- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1

- [15] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 1, 2, 4, 5, 7
- [16] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 8
- [17] Christopher Fifty, Ronald G Junkins, Dennis Duan, Aniketh Iger, Jerry W Liu, Ehsan Amid, Sebastian Thrun, and Christopher R´e. Restructuring vector quantization with the rotation trick. arXiv preprint arXiv:2410.06424, 2024. 2
- [18] Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041, 2023. 3
- [19] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1
- [20] Alexander H¨agele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, and Martin Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. arXiv preprint arXiv:2405.18392, 2024. 1
- [21] Philippe Hansen-Estruch, David Yan, Ching-Yao Chung, Orr Zohar, Jialiang Wang, Tingbo Hou, Tao Xu, Sriram Vishwanath, Peter Vajda, and Xinlei Chen. Learnings from scaling visual tokenizers for reconstruction and generation. arXiv preprint arXiv:2501.09755, 2025. 1, 4
- [22] Shaozhe Hao, Xuantong Liu, Xianbiao Qi, Shihao Zhao, Bojia Zi, Rong Xiao, Kai Han, and Kwan-Yee K Wong. Bigr: Harnessing binary latent codes for image generation and improved visual representation capabilities. arXiv preprint arXiv:2410.14672, 2024. 7
- [23] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 2, 3
- [24] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilib-

- rium. Advances in neural information processing systems, 30, 2017. 3
- [25] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024. 1
- [26] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134,

2017. 5

- [27] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 694–711. Springer, 2016. 5
- [28] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2, 4
- [29] Diederik P Kingma, Max Welling, et al. An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4):307–392, 2019. 4
- [30] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Joshua V. Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, MingHsuan Yang, Irfan Essa, Huisheng Wang, David A Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation. In Proceedings of the 41st International Conference on Machine Learning, 2024. 1
- [31] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. arXiv preprint arXiv:2404.07724, 2024. 1
- [32] Yann LeCun, Yoshua Bengio, et al. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10):1995, 1995. 4
- [33] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022. 2, 7
- [34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 4, 5, 1

- [35] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024. 7
- [36] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregres-

- sive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024. 2, 3, 7, 4
- [37] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 1
- [38] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 2
- [39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 2
- [40] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 2, 4, 7
- [41] Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025. 3
- [42] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 7
- [43] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 4, 8, 3
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 7, 1

- [45] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. 1
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2, 8
- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 7, 1
- [48] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015. 3

- [49] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Taming scalable visual tokenizer for autoregressive image generation. arXiv preprint arXiv:2412.02692, 2024. 2, 7
- [50] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. 2024. 1, 2, 3, 4, 5, 7
- [51] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
- [52] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [53] Keyu Tian, Yi Jiang, Zehuan Yuan, BINGYUE PENG, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. 2, 7, 1, 3
- [54] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 3
- [55] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [56] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2
- [57] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 4
- [58] Hanyu Wang, Saksham Suri, Yixuan Ren, Hao Chen, and Abhinav Shrivastava. Larp: Tokenizing videos with a learned autoregressive generative prior. In ICLR, 2025. 2
- [59] Luting Wang, Yang Zhao, Zijian Zhang, Jiashi Feng, Si Liu, and Bingyi Kang. Image understanding makes for a good tokenizer for image generation. arXiv preprint arXiv:2411.04406, 2024. 3
- [60] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 2
- [61] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv

- preprint arXiv:2409.16211, 2024. 2

[62] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv

- preprint arXiv:2410.13848, 2024. 1

- [63] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 3
- [64] Wanghan Xu, Xiaoyu Yue, Zidong Wang, Yao Teng, Wenlong Zhang, Xihui Liu, Luping Zhou, Wanli Ouyang, and Lei Bai. Exploring representation-aligned latent space for better generation. arXiv preprint arXiv:2502.00359, 2025.
- [65] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025. 3, 7, 4
- [66] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 1, 2, 5, 7
- [67] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 5, 1
- [68] Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, et al. Spae: Semantic pyramid autoencoder for multimodal generation with frozen llms. Advances in Neural Information Processing Systems, 36:52692–52704, 2023. 3
- [69] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 1, 2, 4, 5
- [70] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. arXiv preprint arXiv:2406.07550, 2024. 2, 7, 3
- [71] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 3, 7, 8, 2
- [72] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 2, 8
- [73] Baoquan Zhang, Huaibin Wang, Chuyao Luo, Xutao Li, Guotao Liang, Yunming Ye, Xiaochen Qi, and Yao He. Codebook transfer with part-of-speech for vector-quantized image modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7757–7766, 2024. 3
- [74] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5

- [75] Yue Zhao, Yuanjun Xiong, and Philipp Kr¨ahenb¨uhl. Image and video tokenization with binary spherical quantization. arXiv preprint arXiv:2406.07548, 2024. 2
- [76] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. arXiv preprint arXiv:2406.11837, 2024. 2, 3, 4
- [77] Yongxin Zhu, Bocheng Li, Hang Zhang, Xin Li, Linli Xu, and Lidong Bing. Stabilize the latent space for image autoregressive modeling: A unified perspective. arXiv preprint arXiv:2410.12490, 2024. 3

## GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters for Autoregressive Image Generation

### Supplementary Material

#### A. Limitations and Future Work

Input Image

Reconstructed Image

This study primarily focuses on scaling tokenizers for classconditional image generation. While we have demonstrated the effectiveness of GigaTok for downstream classconditional generation, expanding the scope to include textconditional image generation or video generation remains an open avenue for future work. Additionally, unlike CNNbased 2D tokenizers, 1D Transformer-based tokenizers are not directly applicable to multiple resolutions without additional training adjustments. This challenge presents an important direction for further exploration. Besides scaling the model sizes of tokenizers, the effect of scaling training data, codebook dimension and codebook size for downstream autoregressive generation are left for future research.

Q-FormerEncoder

Q-FormerEncoder

CNNEncoder

CNNDecoder

[Figure 31]

[Figure 32]

Quantizer

ℒ

Semantic Regularization

1D queries

2D queries

- Figure 10. The architecture of GigaTok with Q-Former.

| | | | | |
|---|---|---|---|---|
| | | | | |

2 = (1 + 2 ) + 2 + ⋯ + 2

An 1D token sequence with 2 length can be initialized with 𝐿 levels from a 2D feature map

Avg Pooling for Each Region

| |
|---|

Level 0 Level 1

| | |
|---|---|

Level 2

| | |
|---|---|
| | |

Level 3

| | | | |
|---|---|---|---|
| | | | |

×2 Flatten and Concatenate

| | | | | |
|---|---|---|---|---|
| | | | | |

2D Input Features from the CNN Encoder

- Figure 11. Initialization of 1D queries in Q-Former modules.

#### B. Configurations for AR models

Size Params. Blocks Heads Dim.

B 111M 12 12 768 L 343M 24 16 1024

XL 775M 36 20 1280 XXL 1.4B 48 24 1536

Table 8. Architectures of the LLamaGen models in our experiments.

AR model training. We scale up the training of downstream Llama-style [50, 54] AR models to compare generation performance with other models. For model training, we use WSD learning rate scheduler [20, 25] with 1×10−4 base learning rate, 0.2 decay ratio and 1 epoch warm-up. We do not use AdaLN [44, 53] as it is specific for class-conditional generation. We use a batch size of 256 for training the B, L and XL models and a 512 batch size for training the XXL model. Our AR models are trained for 300 epochs on the 256 × 256 ImageNet training set.

for better visual quality. Interestingly, we find that the 1.4B LlamaGen model achieves the best gFID without CFG.

#### C. Detailed GigaTok Implementation

Please refer to Tab. 9 for training details.

Q-Fomrer in GigaTok. GigaTok utilizes Q-Former [6, 34] to build 1D tokenizers, as shown in Fig. 10. For Q-Former encoder in GigaTok, we initialize the 1D queries initialized from the 2D input features of the CNN encoder using a multi-level average pooling strategy, as shown in Fig. 11. Specifically, for the same 2D input features, we spatially divide them with different granularity at different levels, and perform average pooling for every divided region at each level. The pooled features are flattened and concatenated from level 0 to the last level. Therefore, a 1D token sequence with 2L length can be initialized with L levels from 2D input features. At the decoding stage, the 2D queries are all initialized from the first 1D latent feature.

CFG for gFID. Since gFID of GPT models can be largely affected by classifier free guidance (CFG) [47, 50] and often has an optimal CFG [50], for fair comparison, we search the optimal CFG using zero-order search with a step of 0.25 and use the lowest gFID as the final value. For AR Probing, we use constant CFG scheduling for simplicity. For system-level comparison, we use a step function for CFG scheduling inspired by [31]. Specifically, the AR models predict the first 18% tokens without CFG, i.e., CFG = 1 for better diversity, and use CFG for the remaining tokens

###### Entropy Loss for VQ Tokenizers. While entropy loss [67,

###### Configuration S-S S-B S-L B-L XL-XXL

Q-Former Encoder depth 6 6 6 12 36 Q-Former Encoder heads 8 8 8 12 20 Q-Former Encoder dim. 512 512 512 768 1280 Q-Former Decoder depth 6 12 24 24 48 Q-Former Decoder heads. 8 12 16 16 24 Q-Former Decoder dim. 512 768 1024 1024 1536 Params (M) 136 232 533 622 2896

Codebook size 16384 Codebook dimension 8 #Tokens 256

Training epochs 100 200 200 200 300 Batch size 128 128 256 256 256 Alignment Layer l 3 Learning rate schedule Cosine Decay Base learning rate 1 × 10−4 Minimum learning rate 1 × 10−5 LR warm-up iterations 0 0 0 0 5000 Optimizer AdamW[39] Opt. momentum β1 = 0.9,β2 = 0.95 Entropy Loss weight 0 0 0 0 5 × 10−3

Table 9. GigaTok configuration and default training details

69] is discussed for LFQ [69], its application to VQ tokenizers is less commonly explained. We provide a detailed derivation of the entropy loss specifically for VQ tokenizers. Mathematically, for quantization process from continuous vector z ∈ RD to quantized vector ˆz = ci ∈ RD where ci is the i-th codebook vector from codebook C ∈ RN×D, we assume this process is statistical and follows the following distribution:

p(zˆ = ci|z) ≜ softmax(−l2(z,C))[i] (4)

where l2(z,C) ∈ RN is the L2 distance between z and all the codebook vectors. Then, minimization of the quantization error can be partially achieved by minimizing the expectation of entropy Ez [H(zˆ|z)], which can be understood as maximizing the prediction confidence for p(zˆ|z). To encourage higher codebook utilization, we aim to make the average appearance probability of codebook vectors more uniform. This is achieved by maximizing the entropy H(zˆ), Therefore, the optimization of the two entropy terms leads to the final entropy loss equation:

Lentropy = Ez [H(zˆ|z)] − H(zˆ) (5)

In practice, to calculate H(zˆ), we estimate p(zˆ = ci) by p(zˆ = ci) = Ez [p(zˆ = ci|z)]. Note that entropy loss is not our contribution. We only provide a detailed definition of entropy loss in VQ scenarios for better understanding.

Additional implementation details. To stabilize the training of our tokenizer with a hybrid architecture, we initially use a shortcut feature reconstruction trick at the first 15k iterations of the tokenizer training. But we later found that this trick can be replaced with a simple 1-epoch learning rate warmup combined with entropy loss [15, 69]. Specifically for this trick, we additionally give the output feature of the CNN encoder to the CNN decoder directly to be trained for reconstruction, and also align the output feature of the Transformer decoder to the output feature of the CNN encoder, besides the original training objectives. Note that this strategy is complex and can even hinder performance for XL-XXL tokenizers. We recommend using the learning rate warmup combined with entropy loss [15, 69] instead, for both XL-XXL tokenizer and the smaller ones. Additionally, we utilize the rotation trick [17] for all tokenizers, though we observe its effect on performance to be limited for our tokenizer. The implementation of the semantic regularization is partially inspired by REPA [71].

#### D. Full Evaluation Results and Analysis

Here we present the full evaluation results for the tokenizers and downstream AR models, as summarized in Tab. 10. We observe that scaling up visual tokenizers consistently improves the reconstruction quality across multiple metrics. Interestingly, for the 1.4B AR model, the lowest gFID is obtained without applying any CFG. This phenomenon is

Tokenizer Param. rFID↓ LPIPS↓ PSNR↑ SSIM↑ AR Model Param. gFID↓ Acc.↑ IS↑ Precision↑ Recall↑ LlamaGen-Tok. [50] 72M 2.19 - 20.79 0.675 LlamaGen-B [50] 111M 5.46 - 193.61 0.83 0.45 GigaTok-S-S 136M 1.01 0.2226 20.74 0.670 LlamaGen-B (1d) [50] 111M 4.05 62.6 240.61 0.81 0.51 GigaTok-S-B 232M 0.89 0.2121 20.93 0.677 LlamaGen-B (1d) [50] 111M 3.83 62.9 233.31 0.83 0.51 GigaTok-B-L 622M 0.81 0.2059 21.21 0.685

LlamaGen-B (1d) [50] 111M 3.26 67.6 221.02 0.81 0.56

LlamaGen-XXL (1d) [50] 1.4B 2.03⋆ 69.4 238.52 0.80 0.63 GigaTok-B-L 622M 0.51‡ 0.206 21.32 0.691 LlamaGen-B (1d) [50] 111M 3.33 67.7 265.43 0.80 0.56 GigaTok-XL-XXL 2.9B 0.79 0.1947 21.65 0.699

LlamaGen-B (1d) [50] 111M 3.15 72.0 224.28 0.82 0.55 LlamaGen-XXL (1d) [50] 1.4B 1.98⋆ 74.0 256.76 0.81 0.62

Table 10. Full results for our tokenizers and AR models on ImageNet 256×256. For gFID, we present the lowest value between w/ or w/o CFG scenarios. ‡: Using frozen DINO [7] for discriminator, which largely improves rFID. ⋆: Without classifier-free-guidance.

also observed in the concurrent work FlexTok [3], despite significant differences between GigaTok and FlexTok. We hypothesize that semantic regularization might be the primary contributing factor for this phenomenon.

Discussion on Scaling and Enhancing the Discriminator. Recently, VAR [53], ImageFolder [36], and the concurrent work UniTok [41] have begun leveraging DINO-based discriminators [7, 43] to enhance tokenizer training, achieving impressive improvements in rFID scores. We have also experimented with the same DINO discriminator configuration as VAR. Our results indicate that although rFID scores improve, the downstream generation quality improvements are less significant, as detailed in Tab. 10. Furthermore, when applying the DINO discriminator to XL-XXL tokenizers, we observed that adversarial training frequently encounters instability. Specifically, a strong discriminator quickly learns to distinguish reconstructed samples, diminishing the benefits of adversarial training and leading to blurry artifacts. We leave further exploration of discriminator scaling and enhancement strategies for future work.

#### E. Training Tokenizers for More Iterations

While we largely resolve the reconstruction vs. generation dilemma regarding tokenizer model scaling, this challenge persists for tokenizer training duration scaling. To illustrate this phenomenon, we train five S-S tokenizers ranging from 40 to 120 epochs using a cosine learning rate scheduler, as detailed in Tab. 9. The results are presented in Fig. 12.

When extending tokenizer training iterations, reconstruction quality consistently improves. However, downstream generation quality initially improves but subsequently degrades with further increases in tokenizer training duration. Additionally, the validation loss of AR probing continuously rises with longer tokenizer training, regardless of semantic regularization. This trend suggests an increasing complexity in the tokenizer’s latent space as the training duration extends.

We hypothesize that data scaling may alleviate this is-

sue, and leave it for future exploration. In practice, allocating computational resources toward model scaling rather than extended training duration may yield better tokenizer performance.

#### F. Linear Probing Accuracy of Tokenizers

We show that the linear probing accuracy of the tokenizer encoders may not necessarily indicate the performance of downstream AR models. We utilize the intermediate checkpoints during the training of B-L and XL-XXL tokenizers for evaluation. As shown in Fig. 13, the XL-XXL tokenizer encoder presents an overfitting trend in terms of tokenizer encoder linear probing accuracy. However, this overfitting trend is not reflected in AR Probing linear probing accuracy or gFID. Therefore, the linear probing accuracy of the tokenizer encoders may not be a good indicator of downstream model performance. Similarly, a concurrent work UniTok [41], also points out that the performance of the tokenizer encoder in terms of zero-shot ImageNet classification accuracy may not necessarily reflect the visual understanding ability of downstream LLMs trained on the tokenizer.

The abnormality for large tokenizers reveals that the linear probing accuracy of the tokenizer is not necessarily a good indicator for downstream generation models. Since we care more about the representation learning for downstream models than for the tokenizers, using AR Probing as a direct evaluating method is better than indirect tokenizer linear probing accuracy.

#### G. More Discussions About Related Work

TiTok [70] explores the use of 1D Transformer-based tokenizers under a high compression rate setting. TiTok seminally explores the model scaling of visual tokenizers and uses larger tokenizers for higher compression rate. However, the reconstruction vs. generation dilemma for scaling tokenizers is not solved in TiTok. As a result, the best generation model in TiTok is still trained on its smallest tokenizer variant.

w/o semantic regularization w/ semantic regularization

Tokenizer: rFID

###### Tokenizer: LPIPS

###### AR Probing: Validation Loss

- 9.2
- 9.3
- 9.4
- 9.5AR Probing: g

###### AR Probing: gFID

###### AR Probing: in. Acc.

AR Probing: Lin. Acc.

0.230

8.2

6.4

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

0.460

1.75

0.620

6.2

8.0

1.50

0.450

0.225

0.600

1.25

0.440

6.0

7.8

1.00

0.220

500 1000

500 1000

500 1000

500 1000

500 1000

Tok. Training Iter. (k)

Tok. Training Iter. (k)

Tok. Training Iter. (k)

Tok. Training Iter. (k)

Tok. Training Iter. (k)

- Figure 12. Training duration scaling trends of tokenizers for reconstruction, downstream generation and representation quality with and without semantic regularization. Note that in the last two figures, the red and blue curves correspond to different scales on the y-axis.

250 500 750

Tok. Training Iter./k

0.66

0.68

0.70

Tokenizer: Lin Acc.

250 500 750

Tok. Training Iter./k

0.650

0.675

0.700

AR Probing: Lin Acc.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

250 500 750

Tok. Training Iter./k

4.6

4.8

5.0

5.2

AR Probing: gFID

B-L XL-XXL

- Figure 13. The linear probing accuracy of tokenizer encoders does not necessarily reflect downstream model performance. As the training proceeds, the XL-XXL tokenizer encoder presents an overfitting trend measured by linear probing accuracy, but downstream model performances consistently improve.

visual details respectively. It seminally utilizes semantic alignment to enhance the learned representation of tokenizers.

VA-VAE [65] tames the reconstruction vs. generation dilemma in increasing latent dimensions for continuous VAE [28, 29]. VA-VAE improves the reconstructiongeneration Pareto Frontier by introducing vision foundation model alignment loss. In contrast, we seek continuous improvements in both reconstruction and generation by scaling tokenizers. Semantic regularization serves different purposes in the two works.

ViTok [21] is a concurrent work which has explored the effect of model scaling for VAE [28]. ViTok evaluates its VAE models in terms of both reconstruction and downstream diffusion generation performance. While having a very different setting from GigaTok, ViTok similarly finds that asymmetric design is better for VAEs. While ViTok suggests that small encoders are optimal, we point out that in our setting scaling encoders is also beneficial. Notably, the reconstruction vs. generation dilemma for scaling visual tokenizers is not solved in ViTok. We hypothesize that adding semantic regularization may similarly help solve the tokenizer scaling dilemma for VAEs, but leave it for future study.

MAGVIT-v2 [69] introduces LFQ to enhance discrete tokenizers. It also introduces the entropy penalty for tokenizer training, which is shown to be important for training largescale tokenizers in our work. Instead of tokenizer model scaling, MAGVIT-v2 focuses more on scaling the codebook size of tokenizers. While codebook dimension and codebook size are important bottlenecks for visual tokenizers, we point out that model size scaling is also an important way for improving visual tokenizers.

ImageFolder [36] utilizes two branches for image encoding to handle high-level semantic information and low-level

