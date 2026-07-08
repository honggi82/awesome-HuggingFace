# arXiv:2412.02632v2[cs.CV]4Dec2024

## Scaling Image Tokenizers with Grouped Spherical Quantization

Jiangtao Wang1 , Zhen Qin2, Yifan Zhang3, Vincent Tao Hu4, Björn Ommer4, Rania Briq1, Stefan Kesselheim1

Jülich Supercomputing Centre1, TapTap2, Tsinghua University3, CompVis @ LMU Munich, MCML4

Training Code & Checkpoints

### Abstract

Vision tokenizers have gained a lot of attraction due to their scalability and compactness; previous works depend on old school GAN-based hyperparameters, biased comparisons, and a lack of comprehensive analysis of the scaling behaviours. To tackle those issues, we introduce Grouped Spherical Quantization (GSQ), featuring spherical codebook initialization and lookup regularization to constrain codebook latent to a spherical surface. Our empirical analysis of image tokenizer training strategies demonstrates that GSQ-GAN achieves superior reconstruction quality over state-of-the-art methods with fewer training iterations, providing a solid foundation for scaling studies. Building on this, we systematically examine the scaling behaviours of GSQ—specifically in latent dimensionality, codebook size, and compression ratios—and their impact on model performance. Our findings reveal distinct behaviours at high and low spatial compression levels, underscoring challenges in representing high-dimensional latent spaces. We show that GSQ can restructure high-dimensional latent into compact, low-dimensional spaces, thus enabling efficient scaling with improved quality. As a result, GSQ-GAN achieves a 16× down-sampling with a reconstruction FID (rFID) of 0.50.

### 1 Introduction

Recent advancements in generative models for images and videos have seen substantial success, with approaches like autoregressive models Sun et al. (2024); Kondratyuk et al. (2024); Wang et al. (2024b), masked language models Yu et al. (2024b; 2023); Chang et al. (2022); Weber et al. (2024), and diffusion-based methods (including score-matching and flow-matching) Rombach et al. (2022); Yang et al. (2024); Hu et al. (2024); Gao et al. (2024) and surpass GAN-based Kang et al. (2023); Sauer et al. (2023) models. A common factor in many of these models is the reliance on latent discrete representations of images, especially within language-model-based, where continuous feature maps are quantized into discrete tokens. This quantization has become critical for high-fidelity generation, as tokenized images facilitate model efficiency and enhance generative quality, avoiding the need to work on high-resolution images directly. Recent studies Yu et al. (2024b); Wang et al. (2024b) confirm image tokenizer directly translates to generative quality, and the effectiveness of generative models is closely tied to the performance of image tokenizers.

The fundamental challenge in training image tokenizers is balancing compression efficiency with reconstruction accuracy. While recent methods show progress, several critical issues remain unresolved: (1) Many current tokenizers still depend on outdated GAN-based hyperparameters, often resulting in suboptimal, even negative, performance due to inconsistencies between generation and reconstruction objectives. (2) Benchmarking efforts frequently rely on legacy VQ-GAN implementations with outdated configurations, leading to biased comparisons and limited assessment accuracy. (3) Although various quantization models

2.62

2.5

2.28

2.19

2.0

1.70 1.66 1.63

1.5

rFID

1.17 1.15

1.0

0.82

0.74

0.52 0.50

0.5

VQGAN-LCMASKGITLLamaGEN Titok-BMASKBITOurs(G=1)Open-MAGVIT2MAGVIT2Ours(G=2)Ours(G=4)Ours(G=16)Ours(G=8)

Methods

- (a) Reconstruction performance of GSQ with a latent dimension of 16 at 16ˆ spatial compression, compared to the state-of-the-art.

D16 D32 D64 D128 Latent Dimension

0.1

- 0.5

- 1

- 2

6

rFID

0.39

0.23

0.14

1.63

0.82

0.45

7.09

3.31

1.63

0.83

Downsample Factor

F-8

F-16 F-32

- (b) Scaling behaviour of the latent dimension v.s. spatial compression factor in GSQ; d “ 16 is fixed while groups G increase to expand latent space.

- Figure 1: The top figure shows GSQ-GAN’s reconstruction performance compared to state-of-the-art methods, demonstrating superior results even without latent decomposition. Training with larger G, which is more composed of groups, can further optimize the use of latent space, enhancing reconstruction quality. The bottom figure illustrates GSQ-GAN’s efficient scaling behaviour, where expanded latent capacity effectively manages increased spatial compression, thus achieving higher fidelity reconstructions on highly spatial compressed latent. Notably, GSQ-GAN achieves these results with only 20 training epochs on ImageNet at 2562 resolution, while methods, such as Luo et al. (2024); Yu et al. (2024b), require over 270 epochs.

have been introduced, comprehensive analyses of their relative performance and scalability are limited, hindering the development of efficient, streamlined training methodologies for image tokenizers. Additionally, some methods, such as FSQ Mentzer et al. (2024) and LFQ Yu et al. (2024b), rigidly bind latent dimension and codebook size, making independent scaling of either latent dimension or codebook size infeasible. To address these challenges, we propose the following contributions:

- 1. Grouped Spherical QuAnTization (GSQ): We introduce a novel approach featuring spherical codebook initialization and lookup regularization. With optimised configurations, GSQ outperforms state-of-the-art image tokenizers, achieving high performance with fewer training steps and without the need for auxiliary losses or GAN regularization.
- 2. Efficient Latent Space Utilization: GSQ achieves superior reconstruction performance with compact latent dimensions and large codebook sizes. Scaling studies reveal that latent space is often un-

- derutilized in lower spatial compression scenarios, underscoring the need for efficient latent space usage, which GSQ can address.
- 3. Scalability with Latent Dimensions: GSQ scales effectively with increasing latent dimensions by decomposing and grouping latents. Our spatial scaling studies indicate that latent space saturation occurs at larger spatial reduction scenarios. GSQ enables greater spatial reductions and leverages an expanded latent space to maximize the quantizer’s capacity.

These insights lay a foundation for more efficient and scalable training protocols in image tokenizers, advancing the potential of downstream tasks such as generative models for high-fidelity image generation tasks. We also demonstrate that our training approach can easily train up to 32ˆ spatial downsampling image tokenizer.

### 2 Related Work

The Variational AutoEncoder Kingma (2013) is the foundational approach for image tokenization, initially developed to compress images into a continuous latent space, while later one more work focuses on refining continuous representations Higgins et al. (2017); Vahdat & Kautz (2020); Kim et al. (2019); Luhman & Luhman (2022); Bhalodia et al. (2020); Egorov et al. (2021); Su & Wu (2018); Qin & Huang (2024). Despite their strengths, however, these image encodings, often constrained by strong KL regularization, are rarely applied as image tokenizers within generative models. Instead, the VAE with vector quantization (VQ-VAE) Van Den Oord et al. (2017); Razavi et al. (2019) have become the preferred choice due to their effective use of a codebook for latent distribution regularization. Alternative variance is Residual Vector Quantizer (RVQ) Zeghidour et al. (2021) that can achieve image compression and discrete quantization simultaneously.

Building on the success of VQ-VAE, the VQ-GAN model Esser et al. (2021) further advanced image tokenizer training by incorporating a perceptual loss Zhang et al. (2018) and an adversarial loss, enhancing the quality of generated images. Subsequent research has extended VQ-GAN through (1) architectural improvements, such as transformer-based structures Yu et al. (2022) and Layer Normalization Chang et al. (2022); (2) novel vector quantizers like Finite Scalar Quantization Mentzer et al. (2024), Lookup-Free Quantizer Yu et al. (2024b) and so on Zhao et al. (2024); Zheng et al. (2022a); Zhu & Soricut (2024); Sadat et al. (2024); Adiban

- et al. (2023); Yu et al. (2024a); Cao et al. (2023); You et al. (2022); Lee et al. (2022); Adiban et al. (2022); Kumar et al. (2024); Zheng et al. (2022b); Kumar et al. (2024); Li et al. (2024); Luo et al. (2024); Tian et al.

(2024); Fifty et al. (2024); and (3) refined loss functions with perceptual enhancements, for example, using ResNet-based perceptual loss Weber et al. (2024); Yu et al. (2023) and incorporating StyleGAN discriminators Yu et al. (2022; 2024b). Our work primarily focuses on this stream of compression-oriented image tokenizer training, examining scaling behaviours and their influence on reconstruction quality.

An alternative line of research in image tokenization focuses on embedding semantic visual representations in the latent space, rather than maximising compression rates. This approach typically leverages pretrained visual foundation models, such as DINO Oquab et al. (2024), CLIP Radford et al. (2021), and MAE He et al. (2022), by transferring their learned representations into the latent space of image tokenizers or quantizing their latent representations. Early studies Peng et al. (2022); Hu et al. (2023); Park et al. (2023) demonstrated the feasibility of this strategy, though these models traditionally underperform in reconstruction quality compared to compression-driven tokenizers. Recent advancements have narrowed this gap by optimizing codebook initialization, refining network architectures, and employing advanced knowledge distillation methods, resulting in models that achieve competitive reconstruction fidelity while preserving strong semantic representation capabilities Yu et al. (2024c); Zhu et al. (2024a;b); Li et al. (2024).

### 3 Methodology

- 3.1 Preliminary: VQ Image Tokenizer

The image tokenizer consists of an encoder Enc and a decoder Dec. The encoder compresses the highresolution input image I P RHˆWˆ3 into continuous latent maps:

Z “ EncpIq “ tzi P RDuhi“ˆ1w. (1)

and the decoder reconstructs the image from the latent representation, ˆI “ DecpZq. The down-sampling factor f “ Hh “ Ww denotes spatial reduction, and the compression ratio is given by R “ 3Df2 , where H, W is the height and width of input image I and h, w, D is the height, width and dimension of latent.

With a vector quantizer, the latent space is discretised by mapping Z to indices in the codebook C “ tci P RDuVi“1, where V is the vocabulary size. Each latent vector zi from Z is quantized to the nearest codebook entry using a look-up operation, often based on Euclidean distance:

VQpziq “ lookuppzi,Cq “ arg min

j

||zi ´ cj||2. (2)

- 3.2 Simple Scaling with GSQ

Pursuing higher spatial reduction f requires increasing the latent dimensionality D to maintain R, thus preserving reconstruction fidelity. However, increasing D introduces high-dimensionality challenges, making distance computations less effective and limiting achievable compression ratios. One of the solutions is using a product quantizer Vahdat & Kautz (2020); Zheng et al. (2022a;b); Jegou et al. (2010), hence we decompose each latent vector zi into G groups:

GSQpziq “ tlookup˚pzipgq,CpgqquGg“1, (3)

Here, each zipgq represents a sub-group of zi with d channels, where G ˆ d “ D enables efficient compression without compromising reconstruction fidelity. To improve stability and performance, we propose to initialize

codebook entries from a spherical uniform distribution and same as Yu et al. (2022); Zhao et al. (2024), apply ℓ2 normalization during lookup:

cpjgq „ ℓ2pNp0,1qq, (4) lookup˚pzi,Cq “ arg min

||ℓ2pziq ´ ℓ2pcjq||2. (5)

j

We employ a shared codebook among all groups and omit ℓ2 when G{D P t1,2u, in which case GSQ reduces to LFQ Yu et al. (2024b), and the spherical space significantly collapsed, which requires additional entropy loss during training Yu et al. (2024b); Zhao et al. (2024). Further discussion is provided in Appendix C.

### 4 Experiments

#### 4.1 Optimized Training for GSQ-VAE

We first investigate the efficacy of our proposed improvements to GSQ on VAE-based tokenizers, including impacts of training configurations, auxiliary losses, model architecture, and hyperparameter settings. We set G “ 1 for all modes, they were trained on 1282 resolution ImageNet Deng et al. (2009) with a down-sampling factor f “ 8, vocabulary size V “ 8,192, latent dimensionality D “ 8, with batch size 256, and learning rate of 1e´4 for 100k steps (20 epochs). Specific hyperparameters are reported in Appendix D. All tokenizers adopted an exponential moving average with a decay rate of 0.999. We utilized the LPIPS perceptual loss Zhang et al. (2018) as proposed in Esser et al. (2021) with a weight of 1.0 in training.

|Codebook Init Norm|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò Usage Ò PPL Ò<br><br>|
|---|---|
|Up´1{V,1{V q Up´1{V,1{V q ℓ2<br><br>ℓ2pNp0,1qq<br><br>ℓ2pNp0,1qq ℓ1<br><br>ℓ2pNp0,1qq ℓ2<br><br><br>|11.37 84 0.12 22.3 0.64 3.38% 237 5.343 113 0.10 23.7 0.71 100% 8077 5.343 113 0.12 23.9 0.72 100% 7408 8.312 94 0.12 22.1 0.66 33.9% 566 5.375 113 0.11 23.59 0.71 100% 8062<br><br>|
| | |

- Table 1: Ablation of spherical codebook initialization and lookup normalization for GSQ-VAE-F8 models, trained on ImageNet with 1282 resolution for 20 epochs. PPL is the perplexity.

#### 4.1.1 Effectiveness of Spherical Quantization

Baseline and codebook initialization. table 1 demonstrates that our spherical uniform distribution codebook initialization significantly improved codebook usage to nearly 100% during training. Using ℓ2 normalization, mentioned with previous studies Yu et al. (2022); Zhao et al. (2024), is crucial for stabilizing codebook usage (especially in larger codebooks) and ensuring all codes are usually equal. As illustrated in fig. 7, our approach maintained approximately 100% codebook utilization throughout training, which enabled the reduction of the rFID from 11.37 to 5.375, and with ℓ2 the perplexity of codebook usage is close to the vocabulary size.

20

15

rFID

10

GSQ(G=8)

5

RVQ FSQ VQ

GSQ(G=1)

0

10k 20k 30k 40k

Step

Figure 2: Comparisons of quantizers for VAE-F8 training. VQ is initialized with uniform distribution; all models have the same backbone, latent dimension, and vocabulary size.

Quantizer Comparisons. Taking the proposed spherical codebook initialization method and ℓ2 normalized lookup, GSQ (similar to VQ, when G is 1) can outperform FSQ Mentzer et al. (2024), and by scaling G to 8, GSQ can beat RVQ Zeghidour et al. (2021), as we reported in fig. 2, all model here has same latent dimension eight and vocabulary size 8,192.

Codebook auxiliary loss. We investigated the effectiveness of codebook auxiliary losses, e.g. entropy loss Yu et al. (2024b); Luo et al. (2024) and TCR loss Zhang et al. (2023). table 2 reveals that these losses negatively impacted the tokenizer performance and impeded codebook usage. Entropy loss only provided a marginal improvement with a minimal weight (0.01). Given their limited utility and computational cost on large vocab size during training, we opted not to use them. Also, the later results show that our method maintained 100% codebook usage for vocabulary sizes up to 512k without these losses.

#### 4.1.2 Ablation of Network Backbone

We explored variations in baseline architectures, including the effect of Adaptive Group Normalization (as known as AdaLN) Huang & Belongie (2017) and Depth2Scale Yu et al. (2024b). As detailed in table 3, surprisingly, these modules degraded the reconstruction’s perceptual quality, increasing the rFID but decreasing the pixel-wise error. We use Adaptive Group Normalization as the default and further invested Depth2Scale in GAN’s training in section 4.2.4.

|Entropy Loss TCR Loss<br><br>|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò Usage Ò PPL Ò|
|---|---|
|0.01 0.1 0.5<br><br>0.01<br><br>✗ ✗<br><br>|5.281 114 0.12 23.9 0.72 99.8% 7397 5.687 112 0.12 23.7 0.71 73.5% 5399 7.906 97 0.11 22.8 0.67 8.83% 620 9.937 82 0.15 22.5 0.65 81.1% 830 5.375 113 0.11 23.59 0.71 100% 8062<br><br>|
| | |

- Table 2: Ablation of codebook auxiliary loss for GSQ-VAE-F8. Our methods enable the codebook usage to always be full; there is no need to use this auxiliary loss for training.

|AGN Depth2Scale|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò Usage Ò PPL Ò|
|---|---|
|✗ ✗<br><br>✓<br><br>✓ ✓ ✓|5.375 113 0.11 23.59 0.71 100% 8062 5.406 113 0.10 23.85 0.71 100% 7457 5.562 113 0.11 23.93 0.72 100% 7410 5.531 112 0.11 23.94 0.72 100% 7452<br><br>|
| | |

Table 3: Ablation of using Adaptive Group Norm (AGN) and Depth2Scale for GSQ-VAE-F8.

|Type λp λrec|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò Usage Ò PPL Ò<br><br>|
|---|---|
|LPIPS<br><br>0.1 1.0<br><br>0.1 5.0<br><br>1.0 1.0<br><br><br>1.0 5.0 10 1.0<br>|7.062 98 0.12 25.26 0.75 100% 7013 12.18 73 0.14 25.68 0.75 87% 5673<br><br>5.406 113 0.10 23.85 0.71 100% 7457<br><br>6.156 105 0,11 24.93 0.74 100% 7192 6.093 115 0.11 22.41 0.68 99% 7417<br><br><br>|
|Dino<br><br>0.1 1.0 0.1 5.0 0.7 4.0|7.312 90 0.15 24.91 0.72 100% 6457 4.250 112 0.12 23.12 0.65 100% 7004 4.343 110 0.13 23.66 0.67 100% 6887<br><br>|
|ResNet<br><br>0.1 1.0 0.1 5.0 0.7 4.0<br><br>|31.37 53 0.19 21.70 0.57 37% 2657 9.625 84 0.15 23.91 0.68 73% 5001<br><br>204 1.60 0.56 20.16 0.41 77% 5028|
|VGG-16<br><br>0.1 1.0 0.1 5.0 0.7 4.0|4.468 112 0.14 22.64 0.63 100% 6926<br>5.031 111 0.14 21.97 0.61 100% 6986 4.906 103 0.15 24.17 0.69 100% 6759<br>|
| | |

- Table 4: Ablation of perceptual loss and weights for VAE-F8 training. λp and λrec are weights of perceptual and reconstruction loss.

#### 4.1.3 Ablation of Perceptual Loss Selection

We explored various perceptual loss configurations, including LPIPS Zhang et al. (2018) and logit-based perceptual loss with different backbone architectures: ResNet He et al. (2016), VGG Simonyan & Zisserman (2015), and Dino Oquab et al. (2024). As presented in table 4, our findings indicate that ResNet-based logit loss is ineffective as a perceptual loss, which contradicts earlier findings Weber et al. (2024). In contrast, Dino and VGG-based logit losses yielded lower rFID scores, demonstrating their potential. However, we opted for LPIPS due to its ability to effectively balance rFID and pixel-wise error. We anticipate that further optimisation through detailed hyperparameter tuning could enhance the performance of stronger perceptual losses.

#### 4.1.4 Hyper-parameters optimization for GSQ-VAE

Optimizers. The choice of hyper-parameters specifically β in Adam, significantly affects training dynamics. We evaluated combinations of β values, ranging from 0 to 0.9, and reported results in table 5. Our experiments reveal higher β always brings better reconstruction performance by promoting stable training.

We also assessed weight decay values of 5e´2 and 1e´4, and results show that when higher β is used, weight decay with 5e´2 performing best overall. Therefore, we use β “ r0.9,0.99s with a weight decay of 0.05 for optimal training stability.

|β Weight Decay<br><br>|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò Usage Ò PPL Ò|
|---|---|
|(0, 0.99)<br><br>5e´2 1e´4|5.562 113 0.11 23.9 0.72 100% 7410 5.812 107 0.11 23.9 0.71 100% 7393<br><br>|
|(0.5, 0.99)<br><br>5e´2 1e´4<br><br>|5.750 111 0.10 23.85 0.71 100% 7492 5.375 109 0.09 23.85 0.71 100% 7421|
|(0.9, 0.95)<br><br>5e´2 1e´4|5.406 113 0.10 23.85 0.71 100% 7457 5.562 113 0.10 23.85 0.71 100% 7407<br><br>|
|(0.9, 0.99)<br><br>5e´2 1e´4|5.343 113 0.10 23.89 0.71 100% 7462 5.562 112 0.10 23.86 0.71 100% 7404<br><br>|
|(0.9, 0.999)<br><br>5e´2 1e´4|5.406 112 0.10 23.87 0.71 100% 7472 5.468 111 0.10 23.88 0.71 100% 7411|
| | |

- Table 5: Optimizer’s β and weight decay ablations for GSQ-VAE-F8 training. The codebook usage is 100% for all models.

|Warm-up Decay Final L.R.|rFID IS LPIPS PSNR SSIM Usage PPL Ó Ò Ó Ò Ò Ò Ò|
|---|---|
|0 ✗ 1e´4 5k ✗ 1e´4 5k 75k 1e´5 5k 95k 1e´5 5k 95k 0 5k 10% at 75k 1e´5<br><br>|5.343 113 0.10 23.89 0.71 100% 7462 5.406 114 0.10 23.78 0.72 100% 7429 5.750 110 0.10 23.67 0.71 100% 7344 5.781 109 0.09 23.76 0.71 100% 7355 5.625 111 0.10 23.73 0.71 100% 7343 5.468 112 0.10 23.83 0.71 100% 7389<br><br>|
| | |

- Table 6: Learning rate scheduler ablations for GSQ-VAE-F8 training, the maximal learning rate is 1e´4. The codebook usage is 100% for all models.

Learning rate scheduler. Recent studies used diverse learning rate schedulers for training tokenizers. We compared fixed learning rate training against the other five schedulers, each with a 5k steps warm-up period and varied decay strategies, as plotted in fig. 9 in Appendix D. The results are reported in table 6, showing that substantial learning rate decay negatively impacted model performance, and there are no advantages from warm-up training. Therefore, we opted for a constant learning rate throughout training to maintain the GAN training and simplicity of hyper-parameter optimization.

#### 4.2 Optimized Training for GSQ-GAN

Next, we incorporated a discriminator and adversarial loss to ablate training configurations for GSQ-GAN training on ImageNet Deng et al. (2009) at 1282 resolution for up to 80k steps; the VAE and discriminator have a learning rate of 1e´4. Detailed hyperparameters are reported in Appendix E.

#### 4.2.1 Ablations of Discriminator and Combinations of Adversarial Loss

We evaluated three types of discriminator: N-Layer Discriminator (NLD) Isola et al. (2017), StyleGAN Discriminator (SGD) Karras et al. (2019), and Dino Discriminator (DD) Sauer et al. (2023). We also compared three adversarial loss types: vanilla non-saturating (V), hinge (H), and improved non-saturating (N), resulting in six combinations of adversarial-discriminator loss setups.

Choosing an improper GAN loss led to negative performance for N-Layer and Dino Discriminators. As shown in table 7. All GAN models trained with Dino Discriminators consistently outperformed GAN with

|Discriminator Adv. Discr. loss loss|rFID IS PSNR SSIM Usage PPL Ó Ò Ò Ò Ò Ò<br><br>|
|---|---|
|✗ ✗ ✗|5.343 113 23.89 0.71 100% 7462<br><br>|
|NLD Isola et al. (2017)<br><br>Hinge Vanilla Hinge Hinge Hinge Non-Sat.<br><br>Non-Sat. Vanilla Non-Sat. Hinge Non-Sat. Non-Sat.<br><br>|45.2 25 20.6 0.58 96.4% 6976 24.0 49 21.4 0.62 98.5% 7424 68.5 14 19.3 0.51 58.2% 4069<br><br>9.562 86 22.08 0.66 100% 7558 11.3 80 22.0 0.66 100% 7516 23.7 50 21 0.62 99.0% 7451<br><br>|
|SGD (1k) Karras et al. (2019)<br><br>Hinge Hinge Non-Sat. Vanilla Non-Sat. Hinge<br><br>|18.1 63 21.65 0.64 100% 6104<br>19.1 62 21.57 0.64 100% 6061 27.1 46 21.42 64.96 100% 5514<br>|
|DD Sauer et al. (2023)<br><br>Hinge Hinge Non-Sat. Vanilla Non-Sat. Hinge<br><br>|1.976 116 21.78 0.64 100% 7546 1.906 117 22.01 0.65 100% 7533 1.867 117 22.12 0.66 100% 7525<br><br>|
|OpenMagViT2 w/ 1.75M steps|1.180 Luo et al. (2024)|
| | |

- Table 7: GSQ-GAN-F8 model trained on 1282 ImageNet, 80k training step. The SGD-GAN model is evaluated at the 1k training step due to the failure of NaN loss in training.

the N-Layer one. The best losses for N-Layer Discriminator are with NV losses, achieving an rFID of 9.562, and NH for Dino Discriminator, which reached 1.867 rFID. Additionally, we ablate the data augmentation Sauer et al. (2023) in Dino Discriminator, as shown in table 8, using a combination of colour augmentation, translation, and cutout led to improved reconstruction performance.

2 2

|Discr. Data Aug.|rFID-128 Ó rFID-256 Ó<br><br>|
|---|---|
|✗ Color+Trans Cutout+Color+Trans Resize+Color+Trans<br><br>|1.953 0.824<br>2.000 0.783<br><br><br>1.867 0.824<br><br>2.000 0.832<br>|
| | |

Table 8: Ablation on data augmentation in Dino-Discriminator.

#### 4.2.2 Hyper-parameters Optimization for GSQ-GAN

Discriminator optimizer and adversarial loss weights. We performed ablation studies on optimizer hyper-parameters (β) for N-Layer and Dino Discriminator. The results, presented in table 9, indicate that higher β values (β “ r0.9,0.99s) led to more stable training dynamics for both discriminator types. We used this configuration for the remainder of the experiments. Additionally, varying the weight of adversarial loss did not show significant benefits, leading us to set the adversarial loss weight to 0.1.

Learning Rates and Batch Size. We investigated the batch size and learning rate configurations, comparing three different batch sizes and learning rates. The results, shown in table 10, indicate that larger batch sizes and increased learning rates improved stability and convergence speed and thus allowed us to speed up GAN training with larger batch sizes.

#### 4.2.3 GAN Regularization Ablations

We explored several regularization techniques for stabilizing discriminator training: gradient penalty Gulrajani et al. (2017), LeCAM regularisation Yu et al. (2023), and autoencoder warm-up, as well as adaptive discriminator loss weights Yu et al. (2022), weight decay, and gradient clipping. table 11 summaries our findings.

|Discr. Loss β λadv|rFID Ó IS Ò PSNR Ò SSIM Ò<br><br>|
|---|---|
|NLD NH (0, 0.99) 0.1 NLD NH (0.5, 0.9) 0.1 NLD NH (0.5, 0.9) 0.5 NLD NH (0.9, 0.95) 0.1 NLD NH (0.9, 0.99) 0.1 NLD NH (0.9, 0.99) 0.5<br><br>|6.687 96.5 22.35 0.67 11.31 80.0 22.01 0.66<br><br>106 8.68 15.40 0.29 3.578 114 22.74 0.69 3.515 114 22.85 0.69 3.718 114 22.83 0.69|
|NLD NV (0.5, 0.9) 0.1 NLD NV (0.9, 0.99) 0.1 NLD NV (0.9, 0.99) 0.5<br><br>|9.562 86 22.08 0.66 3.390 102 22.88 0.69 3.515 114 22.86 0.69<br><br>|
|DD NH (0.5, 0.9) 0.1 DD NH (0.9, 0.99) 0.1 DD NH (0.9, 0.99) 0.5 DD NV (0.5, 0.9) 0.1 DD NV (0.9, 0.99) 0.1 DD NV (0.9, 0.99) 0.5<br><br>|1.867 117 22.12 0.66<br><br>1.859 118 22.12 0.66<br><br>2.453 106 20.66 0.59 1.906 117 22.01 0.65<br><br><br>1.820 117 22.02 0.65<br>2.671 102 20.28 0.57<br>|
| | |

- Table 9: Ablation of Adam’s β and adversarial loss weights for GSQ-GAN-F8 training. λadv is the weight of adversarial loss.

|Batch size Learning rate|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò Usage Ò PPL Ò<br><br>|
|---|---|
|256 1e´4<br><br>256 2e´4<br><br>256 3e´4<br><br><br>512 1e´4<br><br>512 2e´4<br><br><br>768 2e´4<br><br>768 3e´4<br>|1.859 118 0.08 22.12 0.66 100% 7528 1.796 119 0.07 22.28 0.66 100% 7525 1.890 118 0.07 22.36 0.67 100% 7544 1.671 120 0.08 22.08 0.66 100% 7494 1.578 122 0.07 22.25 0.66 100% 7538 1.593 121 0.07 22.32 0.67 100% 7513 1.648 122 0.07 22.31 0.67 100% 7520<br><br>|
| | |

- Table 10: Batch size and learning rate ablations of GSQ-GAN-F8 training, with DD-NH discriminator and loss combination.

Using constant λadv performed best, with no advantages observed from adaptive weighting Esser et al. (2021). The Gradient penalty added for N-Layer Discriminator was ineffective, and LeCAM only slightly improved results. Autoencoder warm-up (discriminator training starts after 20k steps) did not improve stability or performance; gradient clipping at 2.0 (by default) was more effective than at 1.0, and weight decay of 1e´4 improved the N-Layer Discriminator but slightly degraded the Dino Discriminator.

Training StyleGAN Discriminator with regularization could not address NaN issues. We also tested a combination of StyleGAN Discriminator and gradient penalty. But training with gradient penalty was also roughly four times slower, so we could not finish the training within 80k step training wall time, see more details of StyleGAN Discriminator in Appendix E).

#### 4.2.4 Analysis of Attention Integration

We conducted ablation studies on the attention module and Depth2Scale layers. Recent works such as Luo

- et al. (2024); Yu et al. (2024b) omit attention layers, but as seen in table 12, incorporating attention into mid-blocks improved model performance. We also re-evaluated Depth2Scale, observing that it enhanced GAN’s performance under adversarial training. The model’s performance across different resolutions is also reported in table 12, showing the model’s cross-resolution inference capabilities. We take Depth2Scale, as it generally benefits the GAN training; the model trained with Depth2Scale has rFID 1.53 with 80k training steps. Including attention modules can further boost reconstruction, though it may introduce instability during training.

|Discr. WD AW|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò PPL Ò|
|---|---|
|NLD-NV 5e´2 NLD-NV + GC 1.0 5e´2 NLD-NV 1e´4 NLD-NV 5e´2 ✓ NLD-NV + GP 5e´2 NLD-NV + LeCAM 5e´2<br><br>|3.390 114 0.06 22.8 0.69 7594 3.453 114 0.06 22.8 0.69 7483<br><br>3.296 115 0.06 22.86 0.69 7494<br>4.437 112 0.07 23.34 0.70 7476<br>5.750 110 0.09 23.78 0.71 7447 3.546 113 0.07 22.89 0.69 7455<br>|
|DD-NH 5e´2 DD-NH 1e´4 DD-NH 5e´2 ✓ DD-NH + AE-warmup 5e´2 DD-NH + LeCAM 5e´2<br><br>|1.859 118 0.08 22.12 0.66 7528<br><br>1.914 118 0.08 22.12 0.66 7514<br>2.687 117 0.07 23.40 0.70 7464<br><br><br>2.000 116 0.08 22.22 0.66 7484 5.250 111 0.08 23.79 0.71 7437<br>|
|SGD-NH 5e´2 ✓<br><br>|3.593 110 0.07 23.61 0.70 7470<br><br>|
| | |

- Table 11: Ablation studies of GAN’s regularization technologies for GSQ-GAN-F8 training, WD is weight decay, AW is adversarial loss adaptive weight Esser et al. (2021), GC is gradient clip. All modes are trained with gradient clip 2.0 by default, GP is gradient penalty, and LeCAM’s weight is 0.001 if enabled; when warmup is used, the discriminator starts to be updated after 20k iterations.

Data Aug D2S Attention

rFIDÓ 128

rFIDÓ 256

1.609 0.675 ✓ 1.578 0.652 ✓ 1.570 0.660 ✓ ✓ 1.531 0.605 ✓ 1.421 0.605 ✓ ✓ 1.539 0.585

✓ ✓ 1.523 0.660 OpenMagViT2 Luo et al. (2024) w/ 1.75M steps 1.180 0.34

- Table 12: Ablation of discriminator data augmentation, integration of attention and Depth2Scale for GSQGAN-F8 training. D2S is the short for Depth2scale.

#### 4.3 Scaling Behaviors of GSQ-GAN

This section investigates how variations influence reconstruction quality in latent dimensions and codebook vocabulary size. All models in this study were trained at a 2562 resolution with a batch size of 512 over 50k steps (20 epochs). Detailed hyper-parameters are provided in Appendix F.

1.7

| |1.633 (77.53M)<br><br>1.548 (83.30M)<br><br>1.430 (100.58M)<br><br>1.273 (166.26M)<br><br>0.389 (75.59M)<br><br>0.369 (77.70M)<br><br>0.375 (96.64M)<br><br>0.363 (100.74M)<br><br>80<br><br>100<br><br>120<br><br>140<br><br>160<br><br>Params.(M)<br><br>f16 - rFID<br><br>f8 - rFID<br><br>Params (M)| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.390

1.6

###### f16-rFID

f8-rFID

Params.(

1.5

0.375

1.4

1.3

1.2

0.360

f16-11224f16-12224f16-11244f16-12244-Deeper F8-1224f8-1224-attnF8-1244f8-1244-attn

- Figure 3: GSQ-GAN ablations on wider and deeper networks w/ and w/o attention blocks. Models are trained on 2562 resolution on ImageNet.

#### 4.3.1 Network Capacity.

We examine the effects of network capacity on reconstruction fidelity, specifically looking at the width and depth. Width scaling was implemented by increasing the number of channels in convolution layers, while depth scaling involved adding additional convolution blocksYu et al. (2024b). The results, summarized in fig. 3, demonstrate consistent improvements in reconstruction as network width and depth increase. Integrating attention modules within wider networks yielded further gains as used in Esser et al. (2021).

0.65

0.637

Vocabulary Sizes

0.621

V=8192

V=16384 V=65536

0.60

0.586

0.586

V=262144 V=524288 Open-MAGVIT2

0.555

0.55

0.519

0.512

0.50

0.488

0.486

rFID

0.475

0.453

0.449

0.443

0.45

0.426

0.397

0.40

0.389

0.389

0.365

0.355 0.357

0.340

0.35

2³ 2 2 2 Latent Dimension

- (a) Scaling of latent dimension and vocabulary size for GSQ at 8ˆ spatial compression.

13 14 15 16 17 18 19 Log(Vocab Size)

0.35

0.40

0.45

0.50

0.55

0.60

0.65

rFID

0.488

0.443

0.389

0.365

0.355

0.586

0.519

0.449

0.389

0.357

0.621

0.555

0.486

0.426

0.397

0.637

0.586

0.512

0.475 0.453

Dimension: 8

Dimension: 16 Dimension: 32 Dimension: 64

- (b) Same scaling behaviour as the top figure with vocabulary size in logarithmic scale.

Figure 4: The top figure illustrates the scaling of latent dimension and codebook size for GSQ at 8ˆ spatial compression, where a smaller latent dimension improves reconstruction, suggesting the latent space is not saturated for F8 downsampling. Optimising latent space size further enhances performance. The bottom figure shows the same trend with vocabulary size in logarithmic scale, indicating effective scaling as vocabulary size increases. All models are trained with G “ 1 and no latent decomposition, making this equivalent to VQ-based methods. All models are trained on ImageNet at 2562 resolution.

#### 4.3.2 Scaling of Latent Space and Vocabulary.

Next, we investigate the impact of scaling latent dimensionality and codebook vocabulary size. Models were trained with latent dimensions of 23, 24, 25, and 26, each paired with vocabulary sizes of 8k, 16k, 64k, 256k, and 512k. Results in fig. 4a and fig. 4b indicate that larger vocabulary sizes, combined with lower latent dimensions, consistently yielded superior reconstruction performance. Remarkably, a model with a latent

dimension of 8 and a vocabulary size 512k outperformed the state-of-the-art image tokenizers, achieving notable results within just 50k training steps (20 epochs).

These findings underscore the significance of a large codebook vocabulary in enhancing quantizer representational capacity. This trend aligns with theoretical expectations, as the representational capacity of GSQ-GAN is fundamentally bounded by log V as shown in fig. 4b, where V is the vocabulary size. The pattern holds consistently across configurations and provides a point of contrast with prior studies with VQ (e.g., Yu et al. (2024b) Yu et al. (2022) Sun et al. (2024)), as they did not employ optimized configurations for VQ-GAN training that the model training degradation has a bias on their scaling behaviours observation.

Our experiments reveal that lower-dimensional latent spaces result in improved reconstruction fidelity. As detailed in Appendix C, low-dimensional latent spaces are advantageous for computing precise Euclidean distances used for codebook updates. This insight supports the success of decomposed vector quantization approaches, such as LFQ Yu et al. (2024b), FSQ Mentzer et al. (2024), and our own proposed GSQ.

Interestingly, one might intuitively expect a larger latent dimension to yield better performance because of the huge latent space. Our results suggest that high-dimensional spaces are often underutilized. This is important since effective compression at higher spatial down-sampling ratios requires larger latent dimensionality. However, normal VQ-like models cannot effectively scale latent dimensions against high spatial compression challenges. As illustrated in fig. 5, increasing latent dimensionality enhances reconstruction quality when moving from F8 to F16. However, beyond a certain point (here is F16-D16), the model encounters the well-known limitations imposed by the curse of dimensionality. By contrast, when using the dimension decomposition in GSQ, even with G “ 2, the reconstruction performance gains fascinating improvement.

#### 4.3.3 Latent Space and Downsample Factor, and Better Scaling with GSQ

To address the limitations regarding the difficulty of scaling attend dimension. We use GSQ to decompose large latent dimensions into low dimensions, thus maximizing reconstruction fidelity more effectively. As demonstrated in table 13, by decomposing latent vectors into multiple groups, GSQ significantly enhances reconstruction performance without changing the overall latent dimensionality or vocabulary size. This result confirms GSQ’s ability to harness the representational power of high-dimensional latent spaces, leading to substantial gains in model fidelity.

- 1

- 2

- 3

- 4

- 5

GSQ(G=1) D=8

GSQ(G=1) D=16

- GSQ(G=1) D=32

- GSQ(G=2) D=32

rFID

10k 20k 30k 40k 50k

Step

- Figure 5: Latent dimension scaling for GSQ-GAN-F16 training, the latent space is saturated for F16 spatial compression; we expect to enhance reconstruction performance by increasing the latent dimension to increase the latent capacity. Only GSQ with latent decomposition can scale to a higher latent dimension.

Notably, the model achieves near-lossless reconstruction with D “ 64 and G “ 16, approaching theoretical maximum performance. Although the compression ratio is very low and lacks practical value, it highlights GSQ’s remarkable scalability and representational power.

Scaling Down-sample Factor. With GSQ optimizing latent space utilization, we further investigate the impact of varying down-sampling factors on reconstruction quality. We conducted experiments across

|Models<br><br>|G ˆ d|rFID Ó IS Ò LPIPS Ó PSNR Ò SSIM Ò Usage Ò PPL Ò|
|---|---|---|
|Luo et al. (2024) LFQ F16-D18 V “ 256k<br><br>|18 ˆ 1|1.17|
|GSQ F8-D64 V “ 8k|1 ˆ 64<br>2 ˆ 32 4 ˆ 16 16 ˆ 4<br><br><br>|0.63 205 0.08 22.95 0.67 99.87% 8,055 0.32 220 0.05 25.42 0.76 100% 8,157 0.18 226 0.03 28.02 0.08 100% 8,143 0.03 233 0.004 34.61 0.91 99.98% 6,775<br><br>|
|GSQ F16-D16 V “ 256k<br><br>|1 ˆ 16<br>2 ˆ 8 4 ˆ 4 8 ˆ 2<br><br><br>16 ˆ 1 16 ˆ 1˚|1.63 179 0.13 20.70 0.56 100% 254,044 0.82 199 0.09 22.20 0.63 100% 257,273 0.74 202 0.08 22.75 0.63 62.46% 43,767<br><br>0.50 211 0.06 23.62 0.66 46.83% 22,181 0.52 210 0.06 23.54 0.66 50.81% 181<br>0.51 210 0.06 23.52 0.66 52.64% 748<br>|
|GSQ F32-D32 V “ 256k|1 ˆ 32<br>2 ˆ 16 4 ˆ 8 8 ˆ 4<br><br><br>16 ˆ 2 32 ˆ 1|6.84 95 0.24 17.83 0.40 100% 245,715 3.31 139 0.18 19.01 0.47 100% 253,369 1.77 173 0.13 20.60 0.53 100% 253,199 1.67 176 0.12 20.88 0.54 59% 40,307 1.13 190 0.10 21.73 0.57 46% 30,302 1.21 187 0.10 21.64 0.57 54% 247|
| | | |

- Table 13: Ablation studies of group decomposition with 8, 16 and 32 spatial downsample, vocabulary size is 8k, 256k and 256k respectively. GSQ outperforms LFQ with 3ˆ lower rFID. G is the number of groups, and d is a latent dimension in each group. 16 ˆ 1˚ is trained with clip instead of ℓ2 normalization.

different configurations of latent dimensions and down-sampling factors. As illustrated in 1b, models trained with a down-sampling factor of f “ 8{16{32 showed a consistent improvement in reconstruction as latent dimensions increased (with d “ 16 and group count G adjusted accordingly). These results align with theoretical expectations and further validate the effectiveness of GSQ in fully utilizing the latent space.

### 5 Conclusion

We introduce a novel quantization method, Grouped Spherical Quantization(GSQ), incorporating spherical codebook initialization, lookup normalization, and latent decomposition. We systematically investigate training strategies and optimizations for the proposed GSQ-GAN, identifying key configurations that enhance reconstruction quality with significantly fewer training iterations. We highlight critical scaling behaviours related to the model, latent space, and codebook vocabulary size, emphasizing the role of compact latent spaces in achieving high-fidelity reconstruction. Our results demonstrate that GSQ efficiently scales in high-dimensional latent spaces, leverating latent decomposition and spherical normalization for improved compression and reconstruction.

### 6 Acknowledgment

In alphabetical order, we thank Erik, Ismail, Jan, Lijun, and Oleg for their insightful input and feedback on this manuscript. This work was supported by the German Federal Ministry for Economic Affairs and Climate Action under the project “NXT GEN AI METHODS: Generative Methods for Perception, Prediction, and Planning,” the bidt project KLIMA-MEMES, Bayer AG, and the German Research Foundation (DFG) project 421703927. We also appreciate the Gauss Centre for Supercomputing e.V. for granting access to computing resources on the JUWELS and JURECA supercomputers at the Jülich Supercomputing Centre (JSC). The German AI Service Centre WestAI provided additional computational resources.

### References

Mohammad Adiban, Marco Siniscalchi, Kalin Stefanov, and Giampiero Salvi. Hierarchical residual learning based vector quantized variational autoencorder for image reconstruction and generation. In 33rd British Machine Vision Conference, 2022.

Mohammad Adiban, Kalin Stefanov, Sabato Marco Siniscalchi, and Giampiero Salvi. S-hr-vqvae: Sequential hierarchical residual learning vector quantized variational autoencoder for video prediction. arXiv preprint arXiv:2307.06701, 2023.

Riddhish Bhalodia, Iain Lee, and Shireen Elhabian. dpvaes: Fixing sample generation for regularized vaes. In Proceedings of the Asian Conference on Computer Vision, 2020.

Shiyue Cao, Yueqin Yin, Lianghua Huang, Yu Liu, Xin Zhao, Deli Zhao, and Kaigi Huang. Efficientvqgan: Towards high-resolution image generation with efficient vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7368–7377, 2023.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325, 2022.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Evgenii Egorov, Anna Kuzina, and Evgeny Burnaev. Boovae: Boosting approach for continual learning of vae. Advances in Neural Information Processing Systems, 34:17889–17901, 2021.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iger, Jerry W. Liu, Ehsan Amid, Sebastian Thrun, and Christopher Ré. Restructuring vector quantization with the rotation trick, 2024.

Peng Gao, Le Zhuo, Chris Liu, , Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans, 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.

Irina Higgins, Loic Matthey, Arka Pal, Christopher P Burgess, Xavier Glorot, Matthew M Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. ICLR (Poster), 3, 2017.

Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving, 2023.

Vincent Tao Hu, Stefan Andreas Baumann, Ming Gui, Olga Grebenkova, Pingchuan Ma, Johannes Fischer, and Björn Ommer. Zigma: A dit-style zigzag mamba diffusion model. In ECCV, 2024.

Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision, pp. 1501–1510, 2017.

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117–128, 2010.

Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10124–10134, 2023.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

Minyoung Kim, Yuting Wang, Pritish Sahu, and Vladimir Pavlovic. Bayes-factor-vae: Hierarchical bayesian deep auto-encoder models for factor disentanglement. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2979–2987, 2019.

Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Joshua V. Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation. In Forty-first International Conference on Machine Learning, 2024.

Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, and Kundan Kumar. High-fidelity audio compression with improved rvqgan. Advances in Neural Information Processing Systems, 36, 2024.

Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11523–11532, 2022.

Xiang Li, Hao Chen, Kai Qiu, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024.

Eric Luhman and Troy Luhman. Optimizing hierarchical image vaes for sample quality. arXiv preprint arXiv:2210.10205, 2022.

Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An opensource project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.

Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: VQ-VAE made simple. In The Twelfth International Conference on Learning Representations, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt.

Song Park, Sanghyuk Chun, Byeongho Heo, Wonjae Kim, and Sangdoo Yun. Seit: Storage-efficient vision training with tokens using 1% of pixel storage. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17248–17259, October 2023.

Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers, 2022.

Tian Qin and Wei-Min Huang. Epanechnikov variational autoencoder. arXiv preprint arXiv:2405.12783, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M Weber. Litevae: Lightweight and efficient variational autoencoders for latent diffusion models. arXiv preprint arXiv:2405.14477, 2024.

Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pp. 30105–30118. PMLR, 2023.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. ICLR, 2015.

Jianlin Su and Guang Wu. f-vaes: Improve vaes with conditional flows. arXiv preprint arXiv:1809.05861, 2018.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. NeurIPS, 2024.

Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. Advances in neural information processing systems, 33:19667–19679, 2020.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation, 2024a. URL https://arxiv.org/abs/2406.09399.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Tackgeun You, Saehoon Kim, Chiheon Kim, Doyup Lee, and Bohyung Han. Locally hierarchical autoregressive modeling for image generation. Advances in Neural Information Processing Systems, 35:16360– 16372, 2022.

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=pfNyExj7z2.

Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10459–10469, 2023.

Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, et al. Spae: Semantic pyramid autoencoder for multimodal generation with frozen llms. Advances in Neural Information Processing Systems, 36, 2024a.

Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview. net/forum?id=gzqrANCF4g.

Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. NeurIPS, 2024c.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30: 495–507, 2021.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Yifan Zhang, Zhiquan Tan, Jingqin Yang, Weiran Huang, and Yang Yuan. Matrix information theory for self-supervised learning. arXiv preprint arXiv:2305.17326, 2023.

Yue Zhao, Yuanjun Xiong, and Philipp Krähenbühl. Image and video tokenization with binary spherical quantization. arXiv preprint arXiv:2406.07548, 2024.

Chuanxia Zheng, Guoxian Song, Tat-Jen Cham, Jianfei Cai, Dinh Phung, and Linjie Luo. High-quality pluralistic image completion via code shared vqgan. arXiv preprint arXiv:2204.01931, 2022a.

Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022b.

Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of VQ-GAN to 100,000 with a utilization rate of 99%. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. URL https://openreview.net/forum?id=RbU10yvkk6.

Yongxin Zhu, Bocheng Li, Hang Zhang, Xin Li, Linli Xu, and Lidong Bing. Stabilize the latent space for image autoregressive modeling: A unified perspective, 2024b.

Zhenhai Zhu and Radu Soricut. Wavelet-based image tokenizer for vision transformers. arXiv preprint arXiv:2405.18616, 2024.

### Contents

- 1 Introduction 1
- 2 Related Work 3
- 3 Methodology 4

- 3.1 Preliminary: VQ Image Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2 Simple Scaling with GSQ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 4 Experiments 4

- 4.1 Optimized Training for GSQ-VAE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 4.1.1 Effectiveness of Spherical Quantization . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 4.1.2 Ablation of Network Backbone . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 4.1.3 Ablation of Perceptual Loss Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.1.4 Hyper-parameters optimization for GSQ-VAE . . . . . . . . . . . . . . . . . . . . . . . 6

- 4.2 Optimized Training for GSQ-GAN . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 4.2.1 Ablations of Discriminator and Combinations of Adversarial Loss . . . . . . . . . . . . 7
- 4.2.2 Hyper-parameters Optimization for GSQ-GAN . . . . . . . . . . . . . . . . . . . . . . 8
- 4.2.3 GAN Regularization Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.2.4 Analysis of Attention Integration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4.3 Scaling Behaviors of GSQ-GAN . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 4.3.1 Network Capacity. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.3.2 Scaling of Latent Space and Vocabulary. . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.3.3 Latent Space and Downsample Factor, and Better Scaling with GSQ . . . . . . . . . . 12

- 5 Conclusion 13
- 6 Acknowledgment 13

- A Performance of State-of-the-Art Image Tokenizers 20
- B Networks 20
- C GSQ and Other Quantizers 21

- C.1 Discussion of Euclidean Distance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- C.2 Scaling Without Dimension Decomposition . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- D Ablation Studies of VAE 24

- D.1 VAE Training configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.2 Usage of Codebook Initialization Ablation Studies. . . . . . . . . . . . . . . . . . . . . . . . 24

- D.3 Learning Rate Scheduler . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- D.4 VAE Reconstruction Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

#### E Ablation Studies of GAN 27

- E.1 GAN Training configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- E.2 Discriminator Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E.3 Adversarial and Discriminator Loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E.4 Failed Style-GAN Discriminator GAN’s Training . . . . . . . . . . . . . . . . . . . . . . . . . 29
- E.5 GAN Reconstruction Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

#### F Scaling Behaviors 32

### A Performance of State-of-the-Art Image Tokenizers

We list additional comparisons of reconstruction performance among various state-of-the-art image tokenizers, including the model trained in this study. The evaluation is conducted on ImageNet at a resolution of 256 ˆ 256.

f Latent-size D V rFID

Ours-GSQ 8 32 ˆ 32 8 (d “ 8,G “ 1) 8k 0.48 Ours-GSQ 8 32 ˆ 32 8 (d “ 8,G “ 1) 256k 0.36 Ours-GSQ 8 32 ˆ 32 16(d “ 16,G “ 1) 256k 0.51 Ours-GSQ 8 32 ˆ 32 64(d “ 4,G “ 16) 8k 0.03 VQ-GAN Esser et al. (2021) 8 32 ˆ 32 8k 1.49 VQGAN-LC Zhu et al. (2024a) 8 1024 8 100,000 1.29 VIT-VQGAN_SL Yu et al. (2022) 8 32 ˆ 32 32 8k 1.28 OmniTokenizer Wang et al. (2024a) 8 32 ˆ 32 8 8k 1.11 OmniTokenizer Wang et al. (2024a) 8 32 ˆ 32 8 8 0.69 LlamaGen Sun et al. (2024) 8 32 ˆ 32 8 16k 0.59 BSQ Zhao et al. (2024) 8 32 ˆ 32 36 236 0.41 Open-MAGVIT2 Luo et al. (2024) 8 32 ˆ 32 18 256k 0.34

Ours-GSQ w/ attention 16 16 ˆ 16 8(d “ 1,G “ 8) 512k 0.95 Ours-GSQ 16 16 ˆ 16 16(d “ 16,G “ 1) 256k 1.42 Ours-GSQ 16 16 ˆ 16 16(d “ 1,G “ 16) 256k 0.52 VQGAN-LC Zhu et al. (2024a) 16 256 8 100,000 2.62 MASKGIT Chang et al. (2022) 16 16 ˆ 16 256 1k 2.28 LlamaGen Sun et al. (2024) 16 16 ˆ 16 8 16k 2.19 Titok-B Yu et al. (2024c) 16 128 4k 1.70 MASKBIT Weber et al. (2024) 16 16 ˆ 16 256 1024 1.66 ImageFolder Li et al. (2024) 16 265 4k 1.57 MAGVIT2 Yu et al. (2024b) 16 16 ˆ 16 18 256k 1.15 Open-MAGVIT2Luo et al. (2024) 16 16 ˆ 16 18 256k 1.17

Table 14: Reconstruction performance comparison of the proposed model against other state-of-the-art methods on ImageNet (256 ˆ 256 resolution).

### B Networks

The network backbone is derived from VQ-GAN Yu et al. (2022), and MagVit2 Yu et al. (2024b). The encoder and decoder backbones are classified into two primary components: up/down-sampling resolution blocks (grey blocks in fig. 6) and mid-blocks (green blocks in fig. 6). We build down-sampling resolution blocks in the encoder with such rules: 1) For a spatial down-sampling factor of f “ 2N, the encoder includes N ` 1 down-sampling blocks, each containing a ResBlock. The first N blocks are followed by a down-sampling operation using stride convolutions. 2) the convolutional channels in each ResBlock and the number of ResBlocks within each down-sampling block are determined by the Channel Multipliers and Encoder Layer Configurations. 3) an additional ResBlock is introduced to match the channel dimensions if the channel multiplier doubles at a specific layer. The decoder follows analogous principles, adding Adaptive GroupNorm layers before each up-sampling operation.

For mid-blocks each of the mid-blocks consists of a specified number of ResBlocks, with their channel dimensions determined by the output channels of the preceding layer. When mid-block attention mechanisms are used, attention is inserted between any two consecutive ResBlocks within the mid-blocks.

|ResBlock<br><br>512<br><br>Conv 3x3 3 → 128<br><br>GroupNorm + SiLU + Conv 1x1, 512 → 16<br><br>ResBlock<br><br>128 Down-Sample<br><br>ResBlock<br><br>256<br><br>Down-Sample<br><br>ResBlock<br><br>256 → 512<br><br>Encoder<br><br>2 ×<br><br>2 ×<br><br>2 ×<br><br>ResBlock 128 → 256<br><br>ResBlock<br><br>256 Down-Sample<br><br>2 ×<br><br>2 ×<br><br>ResBlock 512<br><br>Attention<br><br>ResBlock 512<br><br>|Conv 3x3<br><br>16 → 512<br><br>Adaptive<br><br>GroupNorm<br><br>ResBlock<br><br>512<br><br>UP-Sample<br><br>Adaptive<br><br>GroupNorm<br><br>ResBlock 256<br><br>UP-Sample<br><br>ResBlock<br><br>512 → 256<br><br>Adaptive<br><br>GroupNorm<br><br>ResBlock 256<br><br>UP-Sample<br><br>ResBlock 128<br><br>ResBlock<br><br>256 → 128<br><br>Adaptive<br><br>GroupNorm<br><br>GroupNorm + SiLU + Conv 3x3, 128 → 3<br><br>Decoder<br><br>2 ×<br><br>2 ×<br><br>2 ×<br><br>2 ×<br><br>2 ×<br><br>ResBlock<br><br>512<br><br>Attention<br><br>ResBlock<br><br>512<br><br>|
|---|---|
|Quantizer| |

ResBlock 𝑋

GroupNorm+SiLU

Conv 3x3

𝑋 → 𝑋

GroupNorm+SiLU

Conv 3x3

𝑋 → 𝑋

ResBlock X → 𝑌

GroupNorm+SiLU

Conv 3x3

𝑋 → 𝑌

Conv 3x3 X→ 𝑌

GroupNorm+SiLU

Conv 3x3 Y→ 𝑌

Up-Sample

Conv 3x3 X→ 4𝑋

Depth2Scale 2x2

- Figure 6: Architecture of the GSQ tokenizer. The backbone follows the 2D convolutional version of MagVit2 Yu et al. (2024b), with variations in the number of blocks.

### C GSQ and Other Quantizers

This section discusses the relationship between GSQ and other tokenizers. GSQ provides a unified framework for tokenizers, excluding the specific spherical codebook initialization proposed in this work. Other tokenizers can be derived by appropriate configurations, as outlined in table 15.

| |D<br><br>|d|g|V|Finite|Codebook-Sharing|ℓ2|Fixed-Codebook|Effective V|
|---|---|---|---|---|---|---|---|---|---|
|VQ|D|D|1|V|✗|✗|✗<br><br>|✗|V|
|VQGAN-ViT|D<br><br>|D|1|V|✗|✗|✓|✗|V|
|LFQ<br><br>|D|1|D|2|t´1,1u|✗|✓|✓|2D|
|FSQ<br><br>|D|1|D||Cpgq||✓|✗|✓|✓|gPG |Cpgq||
|BSQ|D|2|D 2<br><br>|V|✗|✓|✓|✓<br><br>|ś<br><br>V D2<br><br>|
|GSQ|d ˆ g|d|g|V|✗|d ą 2|✓|✗|V g|

Table 15: The effective configurations of other tokenizers in GSQ’s view.

VQ VQ Van Den Oord et al. (2017) and GSQ are identical when the latent space is not decomposed into groups (G “ 1) and without ℓ2 normalization.

BSQ BSQ Zhao et al. (2024) represents the d “ 2 case of GSQ, where the number of groups is set as

- G “ D2 . Codebooks are shared across groups, and BSQ’s codebook is fixed.

FSQ FSQ Mentzer et al. (2024) is a specific case of GSQ, where G “ D, and each group has its own unshared, finite codebook. The term "finite" here refers to a small vocabulary size V . with each latent variable z in the codebook Cpgq constrained as follows:

1 V ´ 1

2 V ´ 1

Sigmoidpzq P t0,

,...,1u (6)

,

In FSQ, typical values for V pgq are 5, 6, 7, or 8, representing a very small vocabulary size.

LFQ LFQ Mentzer et al. (2024) can be interpreted from multiple perspectives. Within the GSQ framework, the simplest interpretation is to set d “ 1. For any 1-dimensional latent variable zi, the ℓ2 normalization reduces to two possible outputs, ´1 or 1:

# 1,if zi ą 0 ´1,if zi ă 0

zi ||zi||2

(7)

ℓ2pziq “

“

Special cases, such as zi “ 0, are handled by setting ℓ2pziq “ ´1 in alignment with Mentzer et al. (2024). In this scenario, the 1-dimensional sphere degenerates into two discrete points, reducing the vocabulary size V to 2. Prior studies Yu et al. (2024b); Luo et al. (2024); Zhao et al. (2024) have shown the necessity of additional auxiliary objectives, such as entropy loss, to ensure effective codebook usage during training. However, in LFQ, codebook indices are not explicitly used; instead, the computational cost is transferred to entropy calculations. For large codebooks, even modern entropy computation kernels introduce significant memory and computational overhead.

There are two possible ways to address these challenges for d “ 1 with a shared codebook: Avoid applying ℓ2 normalization, thereby eliminating the vocabulary size degradation and the need for entropy loss and expensive entropy computations in large codebooks. Alternatively, we can enable ℓ2 normalization but use different codebooks among groups (very similar to LFQ). Both approaches generalize the 1-dimensional sphere into a 1-dimensional manifold, equivalent to the d “ 2 case of GSQ without ℓ2 normalization. We take the first solution in for d “ 1 case.

#### C.1 Discussion of Euclidean Distance

The squared Euclidean distance between an n-dimensional vector z and a vector C in the codebook is given by:

||z ´ C||22 “ ||z||22 ` ||C||22 ´ 2pz ¨ Cq, (8)

where z ¨ C denotes the dot product. In high-dimension spaces, (assuming z and C are drawn from Np0,σqboth the mean and variance of the distances scale linearly with dimension n:

Er||z ´ C||22s “ 2nσ2 (9) Varr||z ´ C||22s “ 4nσ4. (10)

By normalizing both z and C with ℓ2 normalization (i.e., ||z||2 “ ||C||2 “ 1), the distance calculation simplifies to:

||ℓ2pzq ´ ℓ2Cq||22 “ 2p1 ´ cosθq (11) where cosθ represents the cosine similarity between z and C. For ℓ2-normalized vectors, the expectation and variance of the squared Euclidean distance are as follows:

Er||ℓ2pzq ´ ℓ2Cq||22s “ 2 (12) Varr||ℓ2pzq ´ ℓ2Cq||22s “

4 n ´ 1 “ Op

1 nq. (13)

In high-dimensional spaces, most vectors in the codebook become nearly orthogonal to the query vector z. This results in similar distances from z to most codebook vectors, converging towards 2 as the dimension increases.

However, the rate of this convergence is relatively slow. As dimensionality increases, the differences between the query vector and the vectors in the codebook become centralized around 2, with variance proportional to n1. This highlights the inefficiency of directly quantifying high-dimensional vectors. Instead, quantifying individual components of high-dimensional vectors separately is more effective in preserving representational diversity and accuracy.

#### C.2 Scaling Without Dimension Decomposition

When dimension decomposition is not applied (i.e., GSQ with G “ 1), we explored the relationship between vocabulary size (V ) and latent dimensionality (D) by tuning these parameters (fig. 4). The relationship between the rFID and the parameters log V and D can be modeled as:

B

rFID “

log V α ` C ¨ Dβ (14) “

411.63 plog V q2.8375

` 0.1601 ¨ D0.1956 (15) (16)

### D Ablation Studies of VAE

#### D.1 VAE Training configurations

We list full training parameters here and highlight the optimized parameters that can improve the models’ performance.

Parameter Value Training Parameters Image Resolution 128ˆ 128 Num Train Steps 100,000 (20 epochs) Gradient Clip 2 Mixed Precision BF16 Train Batch Size 256 Exponential Moving Average Beta 0.999 Model Configuration Down-sample-factor (f) 8 Hidden Channels 128 Channel Multipliers [1, 2, 2, 4] Encoder Layer Configs [2, 2, 2, 2, 2] Decoder Layer Configs [2, 2, 2, 2, 2] Quantizer Settings Embed Dimension (D) 8 Codebook Vocabulary (V ) 8192 Group (G) 1 Codebook Initialization ℓ2pNp0,1qq Look-up Normalization ℓ2 Loss weights

Reconstruction Loss 1.0 Perceptual Loss (LPIPS) 1.0 Commitment Loss 0.25

VAE Optimizer Base Learning Rate 1 ˆ 10´4 Learning Rate Scheduler Fixed Weight Decay 0.05 Betas [0.9, 0.95] Ñ [0.9, 0.99] Epsilon 1 ˆ 10´8

Table 16: VAE-F8 Training Hyperparameters

#### D.2 Usage of Codebook Initialization Ablation Studies.

In section 4.1.1, we compared various codebook initialization methods and observed that ℓ2-normalized lookup in VAE achieves superior reconstruction performance and higher codebook usage. The detailed codebook usage during training is shown in fig. 7. Notably, the proposed spherical initialisation ensures 100% codebook usage throughout the training process, unlike uniform initialisation.

To further analyse the impact, we trained an additional model, GSQ-GAN-F16, with G “ 4 and a 256k vocabulary size, using a codebook initialised with a uniform distribution. As summarised in table 13, the rFID of our proposed method is 0.52, while the uniform distribution case exhibits a degraded rFID of 0.66. More critically, the codebook usage drops significantly to just 3.68% with uniform initialisation, as illustrated in fig. 8.

[Figure 1]

- Figure 7: Codebook usage during training for GSQ-VAE-F8. Our proposed ℓ2pNp0,1qq codebook initialisation, both with and without ℓ2, ensures consistent full codebook usage.

[Figure 2]

- Figure 8: Codebook usage for GSQ-GAN-F16-D16G4 training with a uniformly initialised codebook.

#### D.3 Learning Rate Scheduler

[Figure 3]

Figure 9: The learning rate schedules for GSQ-VAE-F8 training.

In section 4.1.4, we compared five different learning rate schedulers against a constant learning rate for GSQ-VAE-F8. The detailed learning rate schedules relative to training steps are depicted in fig. 9.

#### D.4 VAE Reconstruction Visualization

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

- (a) Original images (128ˆ128 resolution)

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

- (b) Reconstruction results by VAE-F8

[Figure 14]

[Figure 15]

(c) With Depth2Scale

[Figure 16]

[Figure 17]

(d) With Adaptive Normalization

[Figure 18]

[Figure 19]

(e) With Depth2Scale and Adaptive Normalization

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Table 17: Reconstruction results of the VAE-F8 model (in Section 4.1.2) with ablation of Depth2Scale and Adaptive Normalization.

### E Ablation Studies of GAN

#### E.1 GAN Training configurations

We list the full training parameters for GAN in table 18, with highlighted optimised parameters that significantly improve the model’s performance. To achieve a high rFID without group decomposition, ℓ2 normalisation was omitted.

Parameter Value Training Parameters Image Resolution 128ˆ 128 Num Train Steps 80,000 (16 epochs) Gradient Clip 2 Mixed Precision BF16 Train Batch Size 256 Exponential Moving Average Beta 0.999 Model Configuration Down-sample-factor (f) 8 Hidden Channels 128 Channel Multipliers [1, 2, 2, 4] Encoder Layer Configs [2, 2, 2, 2, 2] Decoder Layer Configs [2, 2, 2, 2, 2] Quantizer Settings Embed Dimension (D) 8 Codebook Vocabulary (V ) 8192 Group (G) 1 Codebook Initialization ℓ2pNp0,1qq Look-up Normalization Discriminator Name Dino Discriminator Generator Loss Non-Saturate Discriminator Loss Hinge Dino-D Data Augmentation Cutout+Color+Translation Loss weights Reconstruction Loss 1.0 Perceptual Loss (LPIPS) 1.0 Commitment Loss 0.25 Adversarial Loss 0.1 Discriminator Loss 1.0 VAE Optimizer Base Learning Rate 1 ˆ 10´4 Ñ 2 ˆ 10´4 Learning Rate Scheduler Fixed Weight Decay 0.05 Betas [0.9, 0.99] Epsilon 1 ˆ 10´8 Discriminator Optimizer Base Learning Rate 1 ˆ 10´4 Ñ 2 ˆ 10´4 Learning Rate Scheduler Fixed Weight Decay 0.05 Betas [0.5, 0.9] Ñ [0.9, 0.99] Epsilon 1 ˆ 10´8

Table 18: GAN-F8 Training Hyperparameters

#### E.2 Discriminator Architecture

We listed the network configurations of N-Layer, Dino and StyleGAN discriminators we used in GAN’s ablation studies as follows:

Parameter Value N-Layer Discriminators (NLD) Input Channels 3 Number of Channels 64 Number of Layers 3 Style-GAN Discriminators (SGD) Input Channels 3 Number of Channels 128 Channels Multiplier [2, 4, 4, 4, 4] DINO Discriminators (DD) Base Model DinoV2_vits14_reg Channels Multiplier [2, 4, 4, 4, 4] Features from layer [2, 5, 8, 11]

Table 19: Discriminator configurations

#### E.3 Adversarial and Discriminator Loss

We define the adversarial and discrimination loss as follows: the ℓreal and ℓfake are logits of real and reconstructed images obtained by passing corresponding images to the discriminator.

#### Vanilla Discriminator Loss

- 1

- 2 `

“

q‰ ` E

“

q‰˘

logp1 ` e´ℓ

logp1 ` eℓ

(17)

E

Lvanilla_discr “

real

fake

#### Vanilla Generator Loss

“

q‰

(18) Hinge Generator Loss

logp1 ` e´ℓ

Lvanilla_gen “ E

fake

Lhinge_gen “ ´Erℓfakes (19) Hinge Discriminator Loss

- 1

- 2 pErmaxp0,1 ´ ℓrealqs ` Ermaxp0,1 ` ℓfakeqsq (20)

Lhinge_discr “

#### Non-Saturate Generator Loss

Lnon_saturate_gen “ E”ReLUpℓfakeq ´ ℓfake ¨ 1 ` log ´1 ` e|ℓ

¯ı (21)

fake|

#### Non-Saturate Discriminator Loss

Lreal “ E”ReLUpℓrealq ´ ℓreal ¨ 1 ` log ´1 ` e|ℓ

¯ı (22)

real|

Lfake “ E”ReLUpℓfakeq ´ ℓfake ¨ 0 ` log ´1 ` e|ℓ

¯ı (23)

fake|

- 1

- 2 pLreal ` Lfakeq (24)

Lnon_saturate_discr “

#### E.4 Failed Style-GAN Discriminator GAN’s Training

As discussed in the main paper, extensive ablations were conducted on Style-GAN Discriminator (SGD) training. However, most experiments encountered numerical instability, resulting in NaN errors. We provide a qualitative analysis of these failed runs by plotting training loss and evaluation rFID. We compare three combinations of discriminator losses: NV, HH, and NH. These combinations were chosen based on their relatively better performance in the NLD ablation studies (see table 7).

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>SGD_NV SGD_HH SGD_NH SGD_NH_0.9<br><br>SGD_NH_0.9_AW<br><br>SGD_NH_0.9_GP<br><br>SGD_NH_0.9_GP_AW SGD_NH_0.9_LeCAM SGD_NH_0.9_warmup_LeCAM<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.2

1.0

0.8

AELoss

0.6

0.4

0.2

0k 10k 20k 30k 40k 50k Step

- (a) The summation of the generator (VAE) training loss of GSQGAN training with Style-GAN Discriminator.

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>SGD_NV SGD_HH SGD_NH SGD_NH_0.9<br><br>SGD_NH_0.9_AW<br><br>SGD_NH_0.9_GP<br><br>SGD_NH_0.9_GP_AW SGD_NH_0.9_LeCAM SGD_NH_0.9_warmup_LeCAM<br><br>| | | | |
|---|---|---|---|---|
| | | | | |

5k 10k 15k 20k 25k 30k

Step

5

10

15

20

25

30

rFID

- (b) The RFID of GSQ-GAN trained with Style-GAN Discriminator and different combinations of discriminator loss and regularization.

Figure 10: Style-GAN Discriminator training models’ training loss and rFID are trained with different discriminator loss combinations and GAN regularization technologies.

As shown in fig. 10, training with NV achieves the lowest rFID and exhibits more stable numerical behaviour than the other combinations. During this short training period, NV performs better than NH, achieving both lower rFID and lower training loss, consistent with the results of NLD. However, SGD-NV training fails abruptly at 10k steps due to NaN errors. Training with NH using the optimizer configuration β “ r0.9,0.99s also fails before reaching the 10k step, previous studies (NLD and DD) suggesting that higher β values boost model performance.

We further conducted ablation studies on GAN regularization techniques, including adaptive discriminator loss weights, LeCAM regularization, gradient penalty, and generator warmup. The results are presented in fig. 10. Training with gradient penalty regularization demonstrates a robust and stable dynamic, with the model’s loss decreasing smoothly and achieving lower rFID than other methods. In contrast, training with LeCAM regularization shows significantly unstable behaviour, as reflected by sharp peaks in the loss curves.

Gradient penalty and adaptive Weights perform best for Style-GAN Discriminator training among all the regularization methods, but when these two work together, the training will be highly unstable. Meanwhile,

##### due to the high parameter count and computational FLOPs of SGD, gradient penalty regularization and adaptive weights become computationally expensive, requiring additional backward passes during training. Consequently, it makes SGD an impractical choice for efficient GAN training.

#### E.5 GAN Reconstruction Visualization

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

(a) Orignal images (128ˆ128 resolution)

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

- (b) Reconstruction results by with NLD-NV discriminators

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

- (c) Reconstruction results by with DD-NH discriminators
- (d) Reconstruction results by with NLD-NV discriminators and β “ r0.9,0.99s

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

- (e) Reconstruction results by with DD-NH discriminators and β “ r0.9,0.99s

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Table 20: Reconstruction results of the GAN-F8 models (see Section 4.2.1) , trained with different discriminators.

### F Scaling Behaviors

This section details the training parameters for the GAN scaling experiments. All models were trained on the 256ˆ256 resolution ImageNet dataset. Each scaling ablation study focuses on the latent dimension and codebook vocabulary size.

Parameter Value Training Parameters Image Resolution 256ˆ 256 Num Train Steps 50,000 (20 epochs) Gradient Clip 2 Mixed Precision BF16 Train Batch Size 512 Exponential Moving Average Beta 0.999 Model Configuration Down-sample-factor (f) 8 Hidden Channels 128 Channel Multipliers [1, 2, 2, 4] Encoder Layer Configs [2, 2, 2, 2, 2] Decoder Layer Configs [2, 2, 2, 2, 2] Discriminator Name Dino Discriminator Generator Loss Non-Saturate Discriminator Loss Hinge Dino-D Data Augmentation Cutout+Color+Translation Loss weights Reconstruction Loss 1.0 Perceptual Loss (LPIPS) 1.0 Commitment Loss 0.25 Adversarial Loss 0.1 Discriminator Loss 1.0 VAE and Discriminator Optimizer Base Learning Rate 2 ˆ 10´4 Learning Rate Scheduler Fixed Weight Decay 0.05 Betas [0.9, 0.99] Epsilon 1 ˆ 10´8

Table 21: GAN-F8 Training Hyperparameters

In the network capacity scaling experiments described in section 4.3.1, the model names correspond to their respective Channel Multipliers. The default depth of the network is set to two for each block (Encoder Layer Configs and Decoder Layer Configs). For the Deeper network configuration, the Encoder Layer Configs are set to r4,3,4,3,4,4s and the Decoder Layer Configs to r3,4,3,4,4,4s, following the architectural design principles outlined in MagVit2 Yu et al. (2024b).

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

(b) GSQ-GAN-F8, D “ 16, G “ 1

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

(b) GSQ-GAN-F8, D “ 32, G “ 2

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

- (b) GSQ-GAN-F8, D “ 64, G “ 4

- Table 22: Scaling latent dimension for GSQ-GAN-F8 model. The models are detailed in fig. 1b.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

- (b) GSQ-GAN-F16, D “ 16, G “ 1

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

- (c) GSQ-GAN-F16, D “ 32, G “ 2

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

- (c) GSQ-GAN-F16, D “ 64, G “ 4

- Table 23: Scaling latent dimension for GSQ-GAN-F16 model. The models are detailed in fig. 1b.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

- (b) GSQ-GAN-F32, D “ 16, G “ 1

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

- (c) GSQ-GAN-F32, D “ 32, G “ 2

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

- (d) GSQ-GAN-F32, D “ 64, G “ 4

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

- (e) GSQ-GAN-F32, D “ 128, G “ 8

- Table 24: Scaling latent dimension for GSQ-GAN-F32 model. The models are detailed in fig. 1b.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- (b) GSQ-GAN-F8, D “ 64, G “ 1

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

- (c) GSQ-GAN-F8, D “ 64, G “ 2

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

(c) GSQ-GAN-F8, D “ 64, G “ 4

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

(c) GSQ-GAN-F8, D “ 64, G “ 16

Table 25: Scaling latent dimension for GSQ-GAN-F8-D64 model. The models are detailed in Section 4.3.3.

