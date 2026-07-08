# arXiv:2601.15369v2[eess.IV]12Mar2026

## OpenVision 3: A Family of Unified Visual Encoder for Both Understanding and Generation

Letian Zhang1⋆, Sucheng Ren2⋆, Yanqing Liu1, Xianhang Li1, Zeyu Wang1, Yuyin Zhou1, Huaxiu Yao3, Zeyu Zheng4, Weili Nie5, Guilin Liu5, Zhiding Yu5, and Cihang Xie1

1 UC Santa Cruz 2 JHU 3 UNC-Chapel Hill 4 UC Berkeley 5 NVIDIA

[Figure 1]

##### Project Page: https://ucsc-vlaa.github.io/OpenVision3/

###### OpenVision 3

[Figure 2]

[Figure 3]

Generation

Understanding

Reconstruction

Reconstruction Understanding

###### rFID gFID

2.51 2.54

1.06

[Figure 4]

[Figure 5]

Image caption

2.06

0.36

0.19

1.87

[Figure 6]

Unified Tokens

Addnoise Text Encoder

[Figure 7]

###### Contrastive Learning

[Figure 8]

ViT Decoder

[Figure 9]

[Figure 10]

Understanding

[Figure 11]

[Figure 12]

85.2

[Figure 13]

[Figure 14]

61.0

1468 1405

313

84.7

73.9 65.8

[Figure 15]

VAE Decoder

60.4

[Figure 16]

65.4

72.4

ViT Encoder

292

[Figure 17]

Unified Encoder

Text Decoder

[Figure 18]

[Figure 19]

[Figure 20]

VAE Encoder

Reconstruction Loss

MME-P MME-C

SciQA Seed GQA POPE

Captioning Loss

OpenVision 3-L CLIP-L/14

Fig. 1: An overview of OpenVision 3’s architecture design and performance highlight. Left panel: The architecture of OpenVision 3. We employ a frozen VAE and a trainable ViT as the unified tokenizer, which produces tokens that are fed simultaneously into both the reconstruction and understanding branches. Middle panel: The learning objectives of the two branches. For the reconstruction branch, we focus on pixel-level image reconstruction; concurrently, the understanding branch is optimized via joint contrastive and captioning objectives. Right panel: The performance summarization. The result shows that OpenVision 3 outperforms other unified tokenizers and semantics-based encoders in reconstruction and generation, while remaining competitive with CLIP in multimodal understanding ability.

Abstract. This paper presents a family of advanced vision encoder, named OpenVision 3, that learns a single, unified visual representation that can serve both image understanding and image generation. Our core architecture is simple: we feed VAE-compressed image latents to a ViT encoder and train its output to support two complementary roles. First, the encoder output is passed to the ViT-VAE decoder to reconstruct the original image, encouraging the representation to capture generative structure. Second, the same representation is optimized with contrastive

⋆ Equal contribution

learning and image-captioning objectives, strengthening semantic features. By jointly optimizing reconstruction- and semantics-driven signals in a shared latent space, the encoder learns representations that synergize and generalize well across both regimes. We validate this unified design through extensive downstream evaluations with the encoder frozen. For generation, we test it under the RAE framework: ours substantially surpasses the standard CLIP-based encoder (e.g., gFID: 1.87 vs. 2.54 on ImageNet). For multimodal understanding, we plug the encoder into the LLaVA-1.5 and LLaVA-NeXT framework: it performs comparably with a standard CLIP vision encoder (e.g., 63.3 vs. 61.2 on SeedBench, and 59.2 vs. 58.1 on GQA). We provide empirical evidence that generation and understanding are mutually beneficial in our architecture, while further underscoring the critical role of the VAE latent space. We hope this work can spur future research on unified modeling.

Keywords: Unified Image Tokenizer · Unified Multimodal Model · Vision-Language Pretraining

### 1 Introduction

Unified Multimodal Models (UMMs) has emerged as a cornerstone of multimodal research, driven by the need for systems that seamlessly integrate visual understanding and generation. Their development is grounded in the Platonic Representation Hypothesis [16], which posits that different data modalities reflect a shared underlying reality, and that learning a unified multimodal representation enables mutual benefits across modalities while improving generalization. The success of representative proprietary UMMs such as GPT-4o [17] and Gemini-

- 2.5 Flash [4], as well as public models like BAGEL [6], further supports this view, showcasing strong capabilities in dialogue-based multi-turn generation, multimodal in-context learning, and fine-grained control over generated content.

A key challenge in developing native UMMs lies in how visual representations are encoded. Owing to the representational discrepancy between visual understanding and visual generation, a common UMM design, exemplified by UniFluid [10], BAGEL [6], and MOGAO [27], employs two distinct visual tokenizers that encode the same image twice, producing one set of high-level semantic tokens and another set of low-level, pixel-reconstructable tokens. While effective, this approach increases system complexity and may hinder deeper synergy between understanding and generation.

Another line of work attempts to bridge this gap through shared visual tokenizers. However, these approaches typically rely on quantized hidden representations, which inevitably introduce discretization errors and limit generation quality (e.g., TokenFlow [37], UniTok [34], and EMU3.5 [5]). As a result, developing a simple yet effective continuous visual tokenizer that naturally supports both visual understanding and generation remains an open and practically challenge.

This paper presents OpenVision 3 as a step toward mitigating this challenge. Concretely, we build our tokenizer by stacking a ViT encoder on top of a

well-trained VAE encoder. The output of the ViT encoder is further fed into two separate branches, one generation decoder that is trained to reconstruct the original image and enforce preservation of low-level visual information, and another understanding decoder that is trained by contrastive and captioning objectives, enhancing semantic supervision. Intriguingly, as analyzed in Sec. 5.1, this design choice can non-trivially synergize the learning of both fine-grained details and high-level semantics, e.g., even optimizing understanding loss alone can lead to better reconstruction performance, and conversely, optimizing reconstruction alone can benefit semantic alignment. This behavior is also consistent with recent evidence that semantically informed tokenization can facilitate low-level reconstruction learning [20,50,52], and may even serve as a direct drop-in replacement for purely reconstruction-oriented tokenizers [54].

Our experiments validate the effectiveness of OpenVision 3 across understanding, reconstruction and generation. Crucially, in all downstream evaluations we keep the tokenizer/encoder frozen, ensuring that the reported gains reflect the quality and transferability of the learned visual representation rather than task-specific fine-tuning. For understanding evaluation, we integrate our tokenizer into the LLaVA-1.5 and LLaVA-NeXT framework for training and evaluate its performance across various standard multimodal benchmarks. For the reconstruction evaluation, we evaluate the quality of reconstructed images OpenVision 3 on COCO [29] and ImageNet [7]. For the generation evaluation, we train flow matching model following RAE [54] on ImageNet. The results demonstrate that OpenVision 3 is comparable to CLIP in terms of understanding capabilities(e.g., 63.3 vs. 61.2 on SeedBench, and 59.2 vs. 58.1 on GQA), while surpassing existing unified tokenizers in image reconstruction (e.g., rFID: 0.187 vs. 0.362 on ImageNet). For image generation, our tokenizer outperforms standard CLIP-based encoder under the RAE framework by a large margin (e.g., gFID: 1.87 vs. 2.54 on ImageNet). Moreover, Through the ablation study, we observed that our tokenizer achieves a mutual promotion between understanding and generation during training. The utilization of the VAE latent space has also been empirically validated as both effective and indispensable through our experiments. We hope that releasing OpenVision 3 will catalyze further research into more advanced unified vision tokenizers.

### 2 Related Work

#### 2.1 Vision-Language Pretraining

Vision-Language pretraining serves as the cornerstone of multimodal representation learning. Pioneering works, exemplified by CLIP, adopt contrastive learning

- as their core methodology to extract and align visual and textual features. This training paradigm was subsequently adopted by a wide range of studies, such as LAION [42], DataComp [13], DFN [11], OpenCLIP [2], MetaCLIP [3,49] and CLIPA [24,25]. These research focuses primarily on efficient and open-sourced data and scaling methodologies. Follow-up works have continuously explored alternative training regimes. CoCa [51] adds a captioning loss on the multimodal

decoder outputs which predicts text tokens autoregressively. SigLip [53] proposes to replace contrastive loss with pairwise Sigmoid loss. SigLip2 [44] further extends this by incorporating captioning-based pretraining, self-distillation and masked prediction. The AM-RADIO [14,39] series of works is dedicated to multi-resolution training and knowledge distillation from multiple teacher models. More recently, CLIPS [31], OpenVision [22], and OpenVision 2 [32] have focused on the efficient utilization of captioning loss in vision-language pretraining, demonstrating it to be a low-cost yet high-performance approach. Our work builds upon this line of research and extend this efficient paradigm to unified multimodal learning. By combining contrastive, captioning, and reconstruction losses, we simultaneously supervise semantic and generative learning, resulting in reciprocal performance gains for both parts.

- 2.2 Unified Tokenizer

Extracting representative feature for both generation and understanding has been a bottleneck for the development of unified modeling. Previous works mostly adapt separate encoder for the two kinds of features, and then combine them. For example, BAGEL takes FLUX-VAE [18,19] for low-level features and SigLIP2 for semantic features. UniWorld-V1 [28] also computes the two types of features separately and then concatenates them.

Contrast to above work, another line of studies focuses more on developing unified tokenizers that fuse semantic and pixel-level features. Inspired by the success of VQGAN [9], early unified tokenizers predominantly adapt a discrete token design. Discrete tokenizers rely on vector quantization(VQ) to train representative unified codebooks. For example, TokenFlow [37] jointly optimizes semantic and pixel-level features by incorporating dual codebooks with a shared mapping. VILA-U [47] discretizes the features extracted by the SigLIP with residual quantization. UniTok [34] uses multi-codebook quantization to construct unified discrete representations. Lately, the prevalent trend has gradually shifted toward continuous tokenizers. UniLIP [43] tailors CLIP features for image generation by incorporating self-distillation constraints to bridge the domain gap. Show-o2 [48] applies semantic and low-level projection to VAE latents and fuse dual features to produce unified feature space. More recently, the concurrent work, TUNA [33], further simplifies this by connecting a VAE and a ViT as a unified tokenizer, which is most related to our work. However, TUNA relies on pretrained ViT checkpoints and it remains non-transparent how to train such a tokenizer. In our work, we train the ViT from scratch and propose an effective training paradigm for the unified tokenizer with unified representations.

- 3 Method

- 3.1 Motivation

Developing unified tokenizer is a pivotal step toward unifying generation and understanding, but it is often hindered by the difficulty of establishing a unified

feature space and high-efficient training. Previous studies have presented impressive methods to eliminate these obstacles. However, explorations into constructing unified representations remain in their preliminary stages, and the associated training pipelines still remain non-transparent to the community. In the following, we present our model, which constructs unified vision representation space through a VAE and a ViT in a effective and straightforward way. We demonstrate to the research community how to train a unified tokenizer efficiently from scratch within the VAE latent space.

#### 3.2 OpenVision 3: A unified tokenizer

OpenVision 3 uses a VAE encoder and a vision transformer(ViT) to extract unified vision features, as depicted in Figure 1. The input image x ∈ RH×W×C is first encoded by the VAE encoder Evae from FLUX.1-dev into VAE latents zvae, and the following training process is completely under the VAE latent space. Next, the VAE latents are fed into the ViT encoder Evit to extract the unified representations zu for both understanding tasks and generation tasks. During the VAE stage, the FLUX.1 VAE downsamples the image height and width by 8×, respectively. Therefore, we adjust the patch size of the ViT to 2×2 so that the whole compression ratio is 16×, which aligns with common settings. Formally,

H

8 ×W8 ×Dvae (1)

zvae = Evae(x) ∈ R

16×W16×Du (2)

H

zu = Evit(zvae) ∈ R

where Dvae is the VAE latent channels, Du is the ViT dimensions. The encoded unified feature zu then goes into the reconstruction branch and the understanding branch to do decoding. OpenVision 3 employs two distinct branches to cultivate its ability to extract both generative and interpretive vision representations. The two branches are completely separate, and their respective architectures will be elaborated upon below.

Reconstruction branch. The reconstruction decoding part mirrors the structure of the tokenizer, maintaining a near-symmetrical configuration. Before the decoding, we first add noise to the unified representations in order to improve the generalization of generation ability. The perturbed feature z˜u is generated by adding Gaussian noise scaled by a sample-specific intensity:

z˜u = zu + σ ⊙ ϵ, ϵ ∼ N(0,I) (3) where σ is uniformly sampled from [0,τ] for each instance in the batch, τ is a constant. Then we use a ViT decoder with patch size 1×1 and a linear layer to convert the noised unified feature z˜u back into VAE latents zˆvae. Next, the VAE decoder is applied to decode the zˆvae into reconstruction image xˆ. The whole reconstruction loss includes the reconstruction loss of image xˆ and VAE latents zˆvae, and a perceptual loss based on LPIPS. The whole reconstruction loss can be formulated as:

Lrec = ℓ1(x,xˆ) + βℓ1(zvae,zˆvae) + λLLPIPS(x,xˆ) (4)

Understanding branch. The paradigm of understanding branch generally follows OpenVision [22], where we do contrastive learning and image captioning. As shown in Figure 1, we use a text encoder to extract the caption feature ztxt to calculate contrastive loss with the unified visual feature zu. In parallel, we utilize a text decoder to perform autoregressive prediction of synthetic captions from the unified representations and calculate the corresponding captioning loss. Formally, the understanding loss can be formulated as:

Lund = Lcaption + αLcontrastive(zu,ztxt) (5) The overall training objective is:

Loverall = ωrecLrec + ωundLund (6)

We configure ωund as double that of ωrec during the training process. Reducing ωrec helps to preserve generative quality while ensuring that the understanding capability remains unimpaired.

#### 3.3 Training settings

Training stages and resolution. In accordance with the conclusions drawn in CLIPA [25], we employ a progressive training strategy for the tokenizer, transitioning from low-resolution to high-resolution inputs. We first pretrain the tokenizer at 128×128, and then finetune it with 224×224 or 256×256. The epoch distribution for the two training stages is maintained

- at around a 10:1 ratio. By focusing most of the computation on lowresolution stages, our tokenizer attains superior performance while significantly reducing the computational overhead typically associated with high-resolution training.

Table 1: Parameter configs for two training stages. The epoch number is defined as ImageNet-equivalent epochs (1 epoch ≈ 1.3M samples).

Parameter Pretraining Finetune Resolution 128 224/256 Global batch size 8192 4096 Base learning rate 8 × 10−6 4 × 10−7 Epochs 4000 400 Warmup Epochs 40 20 LPIPS loss weight λ 0 0.5 VAE latents β 0.4 Contrastive α 1.0 Rec. loss ωrec 0.5 Und. loss ωund 1.0

Training details. As depicted in Figure 1, we use pre-trained FLUX.1 VAE and freeze it during the whole training process. All other components (including ViT encoder, ViT decoder, text encoder, text decoder, and linear layer) are randomly initialized and remain unfrozen throughout the training. For the two training stages, the global batch sizes are 8K and 4K, with cosine-decayed base learning rates of 8 × 10−6 and 4 × 10−7. We disable LPIPS loss during the pretraining stage to prevent loss conflicts due to varying resolutions. For complete parameter and configuration details, please

- Table 2: Reconstruction performance of visual tokenizers. Evaluations are performed on the ImageNet and COCO validation sets. Images are resized and centercropped to 256×256. Metrics includes Peak signal-to-noise ratio (PSNR), Structural Similarity Index Measure(SSIM), Learned Perceptual Image Patch Similarity (LPIPS) and reconstruction Fréchet inception distance (rFID).

ImageNet COCO

Model

PSNR↑ SSIM↑ LPIPs↓ rFID↓ PSNR↑ SSIM↑ LPIPs↓ rFID↓ Generation-oriented Tokenizer

SD-VAE 26.26 0.745 0.133 0.606 25.99 0.759 0.130 4.142 SD3-VAE 31.29 0.886 0.059 0.201 31.18 0.894 0.056 1.671 Cosmos 25.07 0.700 0.167 0.959 24.74 0.711 0.165 5.063 FLUX-VAE 32.86 0.917 0.044 0.176 32.73 0.923 0.041 1.343 Wan2.1-VAE 31.34 0.886 0.058 0.945 31.19 0.895 0.055 3.449

Unified Tokenizer RAE (CLIP) 17.44 0.403 0.324 1.06 16.98 0.394 0.345 10.119 UniTok 25.34 0.742 0.132 0.362 24.95 0.750 0.131 3.918 OmniTokenizer 24.69 0.771 0.138 1.411 24.31 0.779 0.137 6.292 Vila-U 22.24 0.612 0.228 4.231 21.89 0.620 0.227 10.997 OpenVision 3 30.92 0.902 0.053 0.187 30.89 0.907 0.050 1.601

refer to Tab. 1. The model is trained on the DataComp dataset recaptioned by LLaVA-Llama-3 [23], which ensures the high quality of the training data and highly efficient multimodal learning.

### 4 Experiments

#### 4.1 Evaluation settings

To comprehensively evaluate the performance of our unified tokenizer, we evaluate the reconstruction, generation and understanding performance and report their results in Sec. 4.2. For the generation side, we follow RAE configs to train a generative model with DiT and a wide DDT head, and evaluate the generation fedelity of OpenVision 3 on ImageNet. For the understanding side, we train vision-language models with our tokenizer under LLaVA-1.5 and LLaVA-NeXT frameworks [30], and evaluate the understanding performance across a range of downstream multimodal benchmarks.

#### 4.2 Reconstruction performance

As shown in Tab. 2, OpenVision 3 significantly outperforms existing unified tokenizers across all metrics. Previous unified models [34,46,47,54] often struggle to maintain high reconstruction quality due to the trade-off required to align with semantic objectives (e.g., SigLIP alignment). For instance, on ImageNet,

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

Original SD RAE OpenVision 3

Fig. 2: Reconstruction visualization. We present three cases of image reconstruction to compare OpenVision 3 with SD-VAE and RAE. Our tokenizer excels at preserving textual content (Case 1) and intricate image details (Case 2 and 3). There is hardly any perceptible difference between our reconstructions and the ground truth.

OpenVision 3 achieves a PSNR of 30.92 dB, surpassing UniTok (25.34 dB) and Vila-U (22.24 dB) by a wide margin. Similarly, in terms of perceptual quality, our model achieves an LPIPS score of 0.053, whereas the closest unified competitor, UniTok, lags behind at 0.132. This demonstrates that our architecture—utilizing the VAE-ViT hybrid design—successfully mitigates the information loss typically associated with semantic compression. On the COCO dataset, our tokenizer keeps a substantial performance advantage. For instance, our rFID significantly outperforms UniTok (1.601 vs. 3.918) and other unified tokenizers by a large margin, further demonstrating our robust generalizability. Notably, even in comparison with specialized generation-oriented tokenizers [1,8,19,40,45], our model maintains competitive or better results.

Visualization. To provide a visual demonstration of our reconstruction performance, we present several qualitative results in Figure 2. Compared to SD-VAE and RAE, OpenVision 3 exhibits superiority in preserving intricate image details and faithfully reconstructing text. In the first case, our tokenizer fully recovers the characters on the hull, whereas both SD-VAE and RAE fail. In the

- Table 3: Class-conditional image generation on ImageNet 256x256. We report gFID, Inception Score (IS), Precision (Pre.), and Recall (Rec.). Our tokenizer achieves the best performance across all metrics.

Tokenizer Generator gFID↓ IS↑ Pre.↑ Rec.↑

SD-VAE DiT 2.27 278.2 0.83 0.57 SD-VAE SiT 2.06 270.3 0.82 0.59 UniTok LlamaGen 2.51 216.7 0.82 0.57 CLIP RAE 2.54 256.4 0.80 0.54 OpenVision RAE 2.44 262.2 0.80 0.53 OpenVision 3 RAE 1.87 290.0 0.84 0.59

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

Fig. 3: Qualitative results of class-conditional ImageNet256 generation. Under the RAE framework, OpenVision 3 is able to generate high quality images.

second and third cases, our tokenizer exhibits superiority in capturing fine-grained details, such as the textures of flowers and foliage and the intricacies of the tiger’s eyes, demonstrating a higher-fidelity reconstruction capability.

#### 4.3 Generation performance

As shown in we report generation Fréchet inception distance (gFID), Inception Score (IS), Precision (Pre.), and Recall (Rec.) as evaluation metric. We present the generative performance of each tokenizer when paired with its respective compatible generator. For low-level tokenizer, we evaluate SD-VAE with traditional diffusion-based generative models (DiT and SiT) [35,36]. For semantic tokenizer, we select CLIP with RAE generator for fair comparison with our tokenizer. According to Tab. 3, OpenVision 3 outperforms these tokenizers across all the metrics. For example, we achieve better gFID when compared to SD-VAE with improved generator SiT(1.87 vs. 2.06). Our tokenizer also surpasses semantic encoders like CLIP [38] by a large margin in generation(1.87 vs. 2.54).

Visualization. As shown in Figure 3, we visualize some class-conditional ImageNet256 generation results with OpenVision 3 under RAE framework. The images generated by our tokenizer exhibit structurally coherent objects with rich stylistic details, which shows the strong capability of our tokenizer in generating highquality samples with great fidelity and diversity.

#### 4.4 Understanding performance

To evaluate the semantic representation capability of OpenVision 3, we integrate it into the LLaVA-1.5 and LLaVA-NeXT framework and conduct training following their standard training configurations. Due to the fixed downsample size of VAE, we keep the same encoded token numbers with OpenAI CLIP for fair comparison. In Tab. 4 and Tab. 5, we compare our tokenizer with CLIP and

- Table 4: Comparison of OpenVision 3 with OpenAI CLIP under LLaVA-1.5 framework. We evaluate the understanding performance of our tokenizer and CLIP on multiple multimodal benchmarks. With the same image token numbers, our unified tokenizer performs on par with OpenAI CLIP under both Base and Large size, while surpassing CLIP across some specific benchmarks.

Method Vision Encoder # Tokens # Res. MME-P MME-C SeedBench ScienceQA GQA POPE OpenAI-CLIP B/16 196 224 1399 318 62.2 73.7 58.6 82.9

OpenVision 3 VAE + B/2 196 224 1388 275 63.1 73.1 58.9 83.7 OpenAI-CLIP L/14 256 224 1468 292 65.4 73.9 60.6 84.7 OpenVision 3 VAE + L/2 256 256 1405 313 65.8 72.4 61.0 85.2

- Table 5: Comparison of OpenVision 3 with OpenAI CLIP under LLaVANeXT framework. We train OpenVision 3 and OpenAI CLIP under LLaVANeXT setting. The results reveal that our tokenizer show competitive understanding performance with CLIP.

Method Vision Encoder # Tokens # Res. MME-P MME-C SeedBench ScienceQA GQA POPE OpenAI-CLIP B/16 196 224 1368 316 61.2 72.8 58.1 84.7

OpenVision 3 VAE + B/2 196 224 1340 337 63.3 68.9 59.2 84.9 OpenAI-CLIP L/14 256 224 1446 319 61.8 75.3 59.4 84.1 OpenVision 3 VAE + L/2 256 256 1449 280 68.6 73.6 62.0 86.6

present the results on multiple multimodal benchmarks, including MME [12], ScienceQA [41], SeedBench [21], GQA [15] and POPE [26]. According to the tables, OpenVision 3 can match or exceed the understanding performance of CLIP on different model sizes (Base and Large). For example, our tokenizer consistently surpasses CLIP on SeedBench (63.1 vs. 62.2 and 65.8 vs. 65.4) and POPE (83.7 vs. 82.9 and 85.2 vs. 84.7) under LLaVA-1.5 framework. In the stronger LLaVA-NeXT setting, OpenVision 3 maintains its performance advantage over CLIP. For instance, we present pronounced performance gap over the CLIP baseline on SeedBench (63.3 vs. 61.2 and 68.6 vs. 61.8) and GQA (59.2 vs. 58.1 and 62.0 vs. 59.4), exhibiting its exceptional robustness.

The results strongly demonstrate that our unified tokenizer is comparable to the understanding-oriented tokenizer CLIP in terms of semantic comprehension, and even displays a clear advantage in certain aspects.

### 5 Discussion

#### 5.1 Reciprocal synergy between understanding and reconstruction

For unified tokenizers, balancing the capabilities of understanding and generation remains a long-standing challenge. To investigate the mutual influence of these two objectives within our tokenizer, we conduct experiments by training the model exclusively with the understanding loss and exclusively with the reconstruction loss, respectively.

- Fig. 4: Loss visualization with only semantic loss. We trained our tokenizer with and without the reconstruction loss, respectively. In Figures (a) and (b), both pixel-level and latent-level reconstruction losses decrease significantly even in the absence of explicit reconstruction signals. Figures (c) and (d) demonstrate that the incorporation of the reconstruction loss has no adverse impact on the losses of the understanding branch.

[Figure 49]

(a) pixel recon loss (b) latents recon loss (c) caption loss (d) contrastive loss

[Figure 50]

[Figure 51]

[Figure 52]

- Fig. 5: Loss visualization with only reconstruction loss. We trained our tokenizer with and without the understanding loss, respectively. In Figure (a), the inclusion of semantic loss leads to a lower image reconstruction loss, suggesting that semantic supervision can, in turn, enhance reconstruction performance. Figures (c) and (d) reveal that the caption loss exhibits a slight downward trend despite the lack of direct semantic signals, and the contrastive loss remains almost stagnant.

Remove reconstruction loss. In Figure 4, we remove reconstruction loss and train only with semantic loss. The blue curve represents the baseline loss, while the red curve denotes the model trained without the reconstruction loss. According to the loss curves in Figure 4a and Figure 4b, even in the absence of reconstruction objectives, the reconstruction loss still exhibits a substantial decline, suggesting that our semantic objectives contribute significantly to image reconstruction. Furthermore, comparing the red and blue curves in Figure 4c and Figure 4d, it is evident that the incorporation of the reconstruction loss leads to no significant change in either caption or contrastive loss. These observations collectively indicate a mutually beneficial synergy between the two types of losses.

Remove understanding loss. In Figure 5, we remove understanding loss and train only with reconstruction-driven signals. The red curves here denote the loss without reconstruction-driven signals. Figure 5c and Figure 5d show that in the absence of semantic supervision, the contrastive loss remains almost stagnant, whereas the caption loss exhibits a marginal decline. This indicates that the reconstruction task intrinsically facilitates semantic tasks that are also generative in nature. Moreover, as seen in Figure 5a, the addition of semantic loss

- Table 6: Effectiveness of VAE in reconstruction. We study the influence of VAE latents in reconstruction performance. Compared to the variant without a VAE, OpenVision 3 demonstrates a significant advantage in rFID, achieving superior reconstruction quality. Our tokenizer maintains competitive results on PSNR, SSIM, and LPIPS with slight differences.

###### Table 7: Effectiveness of VAE in Generation.

Under the VAE latent space, OpenVision 3 outperforms its counterpart by 1.23 gFID, showing better generation ability.

Model PSNR↑ SSIM↑ LPIPs↓ rFID↓

w/o VAE 32.82 0.935 0.060 0.980 OpenVision 3 30.33 0.885 0.061 0.216

Model gFID↓

w/o VAE 9.68 OpenVision 3 8.45

- Table 8: Understanding performance comparison of unified tokenizer with and without VAE. We evaluate the multimodal understanding capabilities of OpenVision 3 and its non-VAE counterpart under the LLaVA-1.5 and LLaVA-NeXT architectures. Under identical training settings, it can be observed that OpenVision

- 3 consistently outperforms the non-VAE version. Under the LLaVA-1.5 framework, our tokenizer outperforms the counterpart in four out of six metrics. Notably, under the LLaVA-NeXT framework, our model achieves a comprehensive lead across all metrics.

Method Vision Encoder # Tokens # Res. MME-P MME-C SeedBench ScienceQA GQA POPE LLaVA-1.5 Framework

- OpenVision 3 VAE + B/2 196 224 1382 287 62.4 73.0 58.0 83.7 w/o VAE B/16 196 224 1366 307 62.2 72.7 57.6 83.8

LLaVA-NeXT Framework

- OpenVision 3 VAE + B/2 196 224 1383 348 63.6 70.5 59.1 84.4 w/o VAE B/16 196 224 1346 296 62.0 67.8 57.7 83.8

paradoxically improves reconstruction performance, providing further evidence of the synergistic relationship between these two branches.

#### 5.2 VAE latents help unified learning

A core design of our tokenizer involves feeding VAE latents into a ViT for encoding. VAEs are originally designed for low-level visual encoding to capture pixel-level image features. However, we found that performing unified modeling with VAE latents substantially enhances image generation capabilities without compromising multimodal understanding performance, as shown in Tab. 6, Tab. 7 and Tab. 8. We argue that high-quality VAE latents enable a single ViT encoder to generate a single, unified visual representation that supports both generation and understanding. This avoids the need for separate encoding processes followed by concatenation or feature fusion. Furthermore, within this VAE latent space, we can achieve superior alignment between features for understanding and generation, leading to high performance in both capabilities.

To evaluate the necessity of VAE, we conducted a comparative analysis by training 2 tokenizers with raw image tokens and VAE latents, respectively.

- Fig. 6: Loss visualization with different ViT decoder sizes. We use M/1, B/1 and L/1 variants as ViT decoder and draw the loss curves. As shown in Figures (a) and (b), employing a Large-sized decoder leads to significant training instability. While varying the decoder size has negligible impact on understanding performance, the Base-sized decoder effectively reduces reconstruction loss while maintaining training stability.

Specifically, since the downsample rate of FLUX-VAE is 8, we separately train VAE+ViT-B/2 and ViT-B/16 as the visual tokenizers to keep the same image token number. Apart from the difference in the encoder architecture, all other structural components and hyperparameters were kept strictly identical. For both sides, we pretrain at 128×128 resolution for 1000 epochs, and then finetune for 200 epochs at 256×256 for generation, and 224×224 for understanding.

In Tab. 6, we report the reconstruction performance with and without VAE. Compared to the standalone ViT, our tokenizer outperforms it by a large margin in rFID (0.216 vs. 0.980), demonstrating our high-quality reconstruction results. Meanwhile, there are marginal gaps in the other three metrics: PSNR, SSIM, and LPIPS. Furthermore, we train both tokenizers under RAE framework with 100 epochs to study the effectiveness of VAE in generation. As shown in Tab. 7, our model achieves a gFID of 8.45 on ImageNet with the VAE, representing a 1.23 improvement over the 9.68 gFID obtained without it. In Tab. 8, we evaluated the understanding capabilities of the tokenizer when the VAE is removed under both LLaVA-1.5 and LLaVA-NeXT frameworks. The results show a noticeable performance gap compared to OpenVision 3. Our tokenizer outperforms the baseline in 4 out of 6 metrics under the LLaVA-1.5 setting; notably, when using the stronger LLaVA-NeXT baseline, our advantage becomes even more pronounced. This suggests that although the VAE primarily extracts low-level visual features, it also contributes to the enhancement of multimodal understanding capabilities. These findings demonstrate that unifying tasks in the VAE latent space yields simultaneous gains in reconstruction fidelity, generative quality, and comprehensive understanding, validating the necessity of the VAE in our unified learning framework.

#### 5.3 The selection of ViT decoder size

In the reconstruction branch, we use a ViT and a linear layer as the pixel decoder. In Figure 6, we investigate the impact of the size of the ViT decoder on reconstruction and understanding. Varying the ViT decoder variants for pixellevel decoding has minimal impact on the understanding branch. As illustrated

###### Table 9: Effectiveness of encoder size in reconstruction and generation. B size and L size show similar performance in reconstruction and generation ability.

Reconstruction Generation PSNR↑ SSIM↑ LPIPs↓ rFID↓ gFID↓ IS↑ Pre.↑ Rec.↑

Model

OpenVision 3-B 30.92 0.902 0.053 0.187 8.45 144.7 0.78 0.36 OpenVision 3-L 30.96 0.902 0.052 0.186 8.89 148.3 0.78 0.36

in Figure 6c and Figure 6d, the semantic signals remains nearly invariant across different ViT decoder sizes. Regarding reconstruction learning, increasing the decoder size from M/1 to B/1 results in a significant reduction in pixel-level loss, as shown in Figure 6a. However, employing the larger L/1 variant leads to highly unstable training, as depicted by the brown curve. In terms of training time, we conduct experiments for 200 epochs for each size. The total training durations for the M/1, B/1, and L/1 decoder on a TPU v4-256 pod are 6.1, 6.0, and 6.9 hours, respectively. Taking into account both performance and computational efficiency, employing Base-sized ViT as the decoder strikes an optimal balance.

#### 5.4 The selection of ViT encoder size

In Tab. 9, we explore the scaling behavior of the ViT encoder on the image reconstruction and generation. Generation results are evaluated under RAE pipeline with 100 training epochs. The results indicate that B-size and L-size tokenizers yield comparable results in reconstruction and generation. In contrast, Tab. 4 and Tab. 5 demonstrate that scaling up to L-size provides substantial gains in multimodal understanding. We argue that although increasing the size leads to a more representative tokenizer naturally, the outer VAE architecture imposes a theoretical ceiling on the generative performance. This bottleneck marginalizes the gains derived from larger encoders, presenting an open problem that requires further investigation in future research.

### 6 Conclusion

This work introduces OpenVision 3, a unified vision encoder for both understanding and generation. We innovatively couple a VAE with a ViT to form a unified architecture, and generate a single, unified representation for different downstream tasks. For the efficient training of our tokenizer, we propose a new training paradigm with both reconstruction- and semantics-driven signals for joint learning. Comprehensive evaluations reveal that our model yields superior results across generative and understanding tasks. OpenVision 3 outperforms current other unified tokenizer in reconstruction and generation, and shows competitive ability with CLIP on semantic tasks. Furthermore, our architecture enables a mutual promotion relationship between image understanding and generation. To facilitate future research in the community, we will fully open-source our training code, data, and tokenizer checkpoints.

### References

- 1. Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al.: Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575 (2025)
- 2. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2818–2829 (2023)
- 3. Chuang, Y.S., Li, Y., Wang, D., Yeh, C.F., Lyu, K., Raghavendra, R., Glass, J., Huang, L., Weston, J., Zettlemoyer, L., et al.: Meta clip 2: A worldwide scaling recipe. arXiv preprint arXiv:2507.22062 (2025)
- 4. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,

I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

- 5. Cui, Y., Chen, H., Deng, H., Huang, X., Li, X., Liu, J., Liu, Y., Luo, Z., Wang, J., Wang, W., et al.: Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583 (2025)
- 6. Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al.: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025)
- 7. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)
- 8. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)
- 9. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021)
- 10. Fan, L., Tang, L., Qin, S., Li, T., Yang, X., Qiao, S., Steiner, A., Sun, C., Li, Y., Zhu, T., et al.: Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436 (2025)
- 11. Fang, A., Jose, A.M., Jain, A., Schmidt, L., Toshev, A., Shankar, V.: Data filtering networks. arXiv preprint arXiv:2309.17425 (2023)
- 12. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023)
- 13. Gadre, S.Y., Ilharco, G., Fang, A., Hayase, J., Smyrnis, G., Nguyen, T., Marten, R., Wortsman, M., Ghosh, D., Zhang, J., et al.: Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems 36, 27092–27112 (2023)
- 14. Heinrich, G., Ranzinger, M., Yin, H., Lu, Y., Kautz, J., Tao, A., Catanzaro, B., Molchanov, P.: Radiov2. 5: Improved baselines for agglomerative vision foundation models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22487–22497 (2025)
- 15. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: CVPR (2019)

- 16. Huh, M., Cheung, B., Wang, T., Isola, P.: The platonic representation hypothesis. arXiv preprint arXiv:2405.07987 (2024)
- 17. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)
- 18. Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024)
- 19. Labs, B.F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al.: Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742 (2025)
- 20. Leng, X., Singh, J., Hou, Y., Xing, Z., Xie, S., Zheng, L.: Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483 (2025)
- 21. Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., Shan, Y.: Seed-bench: Benchmarking multimodal large language models. In: CVPR (2024)
- 22. Li, X., Liu, Y., Tu, H., Xie, C.: Openvision: A fully-open, cost-effective family of advanced vision encoders for multimodal learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3977–3987 (2025)
- 23. Li, X., Tu, H., Hui, M., Wang, Z., Zhao, B., Xiao, J., Ren, S., Mei, J., Liu, Q., Zheng, H., et al.: What if we recaption billions of web images with llama-3? arXiv preprint arXiv:2406.08478 (2024)
- 24. Li, X., Wang, Z., Xie, C.: Clipa-v2: Scaling clip training with 81.1% zero-shot imagenet accuracy within a $10,000 budget; an extra $4,000 unlocks 81.8% accuracy. arXiv preprint arXiv:2306.15658 (2023)
- 25. Li, X., Wang, Z., Xie, C.: An inverse scaling law for clip training. Advances in Neural Information Processing Systems 36, 49068–49087 (2023)
- 26. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: EMNLP (2023)
- 27. Liao, C., Liu, L., Wang, X., Luo, Z., Zhang, X., Zhao, W., Wu, J., Li, L., Tian, Z., Huang, W.: Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472 (2025)
- 28. Lin, B., Li, Z., Cheng, X., Niu, Y., Ye, Y., He, X., Yuan, S., Yu, W., Wang, S., Ge, Y., et al.: Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147 (2025)
- 29. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)
- 30. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 26296–26306 (2024)
- 31. Liu, Y., Li, X., Wang, Z., Zhao, B., Xie, C.: Clips: An enhanced clip framework for learning with synthetic captions. arXiv preprint arXiv:2411.16828 (2024)
- 32. Liu, Y., Li, X., Zhang, L., Wang, Z., Zheng, Z., Zhou, Y., Xie, C.: Openvision 2: A family of generative pretrained visual encoders for multimodal learning. arXiv preprint arXiv:2509.01644 (2025)
- 33. Liu, Z., Ren, W., Liu, H., Zhou, Z., Chen, S., Qiu, H., Huang, X., An, Z., Yang, F., Patel, A., et al.: Tuna: Taming unified visual representations for native unified multimodal models. arXiv preprint arXiv:2512.02014 (2025)
- 34. Ma, C., Jiang, Y., Wu, J., Yang, J., Yu, X., Yuan, Z., Peng, B., Qi, X.: Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321 (2025)

- 35. Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In: European Conference on Computer Vision. pp. 23–40. Springer

(2024)

- 36. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)
- 37. Qu, L., Zhang, H., Liu, Y., Wang, X., Jiang, Y., Gao, Y., Ye, H., Du, D.K., Yuan, Z., Wu, X.: Tokenflow: Unified image tokenizer for multimodal understanding and generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2545–2555 (2025)
- 38. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)
- 39. Ranzinger, M., Heinrich, G., Kautz, J., Molchanov, P.: Am-radio: Agglomerative vision foundation model reduce all domains into one. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12490–12500

(2024)

- 40. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 41. Saikh, T., Ghosal, T., Mittal, A., Ekbal, A., Bhattacharyya, P.: Scienceqa: A novel resource for question answering on scholarly articles. In: IJDL (2022)
- 42. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems 35, 25278–25294 (2022)
- 43. Tang, H., Xie, C., Bao, X., Weng, T., Li, P., Zheng, Y., Wang, L.: Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278 (2025)
- 44. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025)
- 45. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)
- 46. Wang, J., Jiang, Y., Yuan, Z., Peng, B., Wu, Z., Jiang, Y.G.: Omnitokenizer: A joint image-video tokenizer for visual generation. Advances in Neural Information Processing Systems 37, 28281–28295 (2024)
- 47. Wu, Y., Zhang, Z., Chen, J., Tang, H., Li, D., Fang, Y., Zhu, L., Xie, E., Yin, H., Yi, L., et al.: Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429 (2024)
- 48. Xie, J., Yang, Z., Shou, M.Z.: Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564 (2025)
- 49. Xu, H., Xie, S., Tan, X.E., Huang, P.Y., Howes, R., Sharma, V., Li, S.W., Ghosh, G., Zettlemoyer, L., Feichtenhofer, C.: Demystifying clip data. arXiv preprint arXiv:2309.16671 (2023)
- 50. Yao, J., Yang, B., Wang, X.: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15703–15712 (2025)

- 51. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917 (2022)
- 52. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940 (2024)
- 53. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pretraining. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)
- 54. Zheng, B., Ma, N., Tong, S., Xie, S.: Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690 (2025)

