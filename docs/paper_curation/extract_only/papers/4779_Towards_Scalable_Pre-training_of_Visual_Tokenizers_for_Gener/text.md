## arXiv:2512.13687v2[cs.CV]6Mar2026

[Figure 1]

# Towards Scalable Pre-training of Visual Tokenizers for Generation

###### Jingfeng Yao1, Yuda Song2, Yucong Zhou2 and Xinggang Wang1,*

1Huazhong University of Science and Technology, 2MiniMax

The quality of the latent space in visual tokenizers (e.g., VAEs) is crucial for modern generative models. However, the standard reconstruction-based training paradigm produces a latent space that is biased towards low-level information, leading to a foundational flaw: better pixel-level reconstruction accuracy does not lead to higher-quality generation. This implies that pouring extensive compute into visual tokenizer pre-training translates poorly to improved performance in generation. We identify this as the “pre-training scaling problem” and suggest a necessary shift: to be effective for generation, a latent space must concisely represent high-level semantics. We present VTP, a unified visual tokenizer pre-training framework, pioneering the joint optimization of image-text contrastive, self-supervised, and reconstruction losses. Our study reveals that perception-oriented tokenizer pre-training unlocks a new scaling law for generation, where generative performance scales effectively with compute, parameters, and data allocated to the pre-training of the visual tokenizer. Our large-scale pre-training experiments demonstrate the following results: (1) Without modifying DiT training specs and FLOPs, solely scaling VTP pre-training consistently achieves gains in both ImageNet class-conditional and LAION text-toimage generation, while conventional autoencoders stagnate very early at 1/10 of the FLOPs. (2) VTP achieves 0.36 rFID while simultaneously delivering 78.2% zero-shot accuracy and 85.7% linear probing accuracy, surpassing prior unified tokenizers such as VILA-U and UniTok. (3) Furthermore, the VTP-based diffusion model exhibits exceptionally fast convergence—reaching 2.03 gFID in only 80 epochs without guidance tricks, outperforming previous methods like VA-VAE and RAE—and ultimately scales to achieve a remarkable 1.11 gFID on ImageNet 256 × 256 generation. Our code and models are publicly available at https://github.com/MiniMax-AI/VTP.

### 1. Introduction

Latent Diffusion Models (LDMs) (Rombach et al., 2022) employ a visual tokenizer, such as a VAE (Kingma and Welling, 2013), to compress visual signals into a latent space. Typically, visual tokenizers are pre-trained in a separate stage using a reconstruction objective.

However, a clear paradox has emerged: better reconstruction does not guarantee better generation. Instead, a noticeable trade-off between the two objectives is widely observed (Esser et al., 2024; Xie et al., 2024; Yao et al., 2025). It implies that scaling the computational investment in pre-training, while potentially further improving reconstruction performance, carries the risk of compromising generation performance (see fig. 1 (c)), which is consistent with prior work (Hansen-Estruch et al., 2025). We note that this limitation of reconstruction-only training may arise because the objective biases the latent space toward low-level information and, as training scales up, increasingly drives it away from the structured latent space we ultimately desire, thereby motivating the search for tokenizer pre-training schemes that genuinely scale—a challenge we term the “pre-training scaling problem”.

Unlike conventional approaches that emphasize low-level information, we propose that an effective latent space for generation should efficiently encode the core visual semantics. Early explorations have already demonstrated the value of this principle through two primary pathways. Some works explicitly

Corresponding author(s): xgwang@hust.edu.cn © 2026 MiniMax. All rights reserved

(c) Pre-training improves Generation

- (a) Visual Tokenizer Pre-training
- (b) Downstream Gen. Model Training Diffusion Model Z

auto-encoder representationlearning ...

[Figure 2]

REC SSL CLIP

Visual Tokenizer

better generation

no grad Scaling Pre-training

Figure 1 | Visual Tokenizer Pre-training. We revisit the visual tokenizer pre-training in LDM (Rombach et al., 2022) from a representation learning perspective. Critically, while keeping the diffusion model (e.g., DiT (Peebles and Xie, 2023)) training configuration and FLOPs fixed, our method improves generation solely by scaling the tokenizer’s pre-training to learn a better-structured latent space.

enrich the latent space with specific semantic objectives—for instance, by concatenating optical flow for motion or leveraging powerful pre-trained features to the latent space, like VideoJAM (Chefer et al.) and ReDi (Kouzelis et al., 2025). Others implicitly structure the space using semantic constraints, an approach seen in methods like VA-VAE (Yao et al., 2025) and REPA-E (Leng et al., 2025), which regularize the VAE’s feature space with representational priors. While promising, these efforts remain preliminary and do not explore broader scaling properties.

To address this challenge, we present VTP, a novel pre-training framework for visual tokenizers. The core contribution of our work is a redesigned, scalable paradigm for visual tokenizer pre-training that benefits generation. This is achieved by jointly optimizing the model across a spectrum of visual representation tasks, including cross-modal alignment, global semantic understanding, local spatial perception, and low-level pixel reconstruction. Technically, our framework is built upon a vision transformer (ViT) (Dosovitskiy, 2020) based Auto-Encoder. Building on the flexibility of the ViT architecture for representation learning, we integrate a suite of diverse learning objectives. First, cross-modal image-text contrastive learning is employed to instill a global semantic understanding (Ilharco et al., 2021;

[Figure 3]

better gen.

stagnated und.&gen.

better und.

Figure 2 | Understanding is a key driver of generation. We observe a strong positive correlation between the comprehension and generative capabilities of the latent space during visual tokenizer pretraining.

Radford et al., 2021). This is complemented by integrating established self-supervised learning techniques, notably self-distillation and mask image modeling (Caron et al., 2021; He et al., 2022; Oquab et al., 2023; Zhou et al., 2021), to enhance the model’s spatial-semantic perception. Throughout this multi-task learning process, the pixel-level reconstruction objective is consistently applied to preserve fine-grained visual details for generation. We posit that this holistic training paradigm encourages the latent space to form a unified and rich representation of visual information, which is instrumental in boosting the fidelity and semantic coherence of the generated outputs. (section 3)

Through extensive experiments, we reveal that perception-oriented tokenizer pre-training unlocks a new scaling frontier for generation: (1) Understanding is a key driver of generation: The introduction of semantic understanding and perception tasks enhances the generative capability of models initially

pre-trained solely on reconstruction. We observe a strong positive correlation between the semantic quality of the latent space and its generative performance. While these tasks differ in paradigm, they consistently inject more meaningful representations, leading to significant gains in downstream generation. (see fig. 2) (2) Superior Scalability for Generation: VTP is the first visual tokenizer to demonstrate scaling properties. Its generative performance improves steadily as we scale up training compute (FLOPs), model parameters, and dataset size of the visual tokenizer. This stands in stark contrast to traditional tokenizers pre-trained only on reconstruction, whose performance rapidly saturates and shows negligible gains with increased scale.

We conduct extensive experiments on ImageNet (Deng et al., 2009) class-conditional generation and LAION (Schuhmann et al., 2021) text-to-image generation. (1) We demonstrate new scalability for visual tokenizer pre-training: increasing pre-training compute accelerates the convergence of downstream VTP-based diffusion models. Specifically, scaling compute by 10× yields a 65.8% FID improvement for DiT on ImageNet, overcoming the early saturation of conventional autoencoders. (2) A systematic analysis reveals that diverse perception losses consistently improve general generation quality (section 4), and (3) specific losses offer targeted benefits for downstream tasks; e.g., CLIP loss significantly enhances text-to-image generation (section 5). (4) Our final model gets a strong performance, achieving 0.36 rFID on ImageNet alongside 78.2% zero-shot and 85.7% linear probing accuracy, outperforming unified tokenizers like VILA-U (Wu et al., 2024) and UniTok (Ma et al., 2025). Coupled with a DiT (Peebles and Xie, 2023; Yao et al., 2025), it enables exceptionally fast convergence (2.03 FID in just 80 epochs, unguided), surpassing prior approaches such as VA-VAE (Yao et al., 2025) and RAE (Zheng et al., 2025), and ultimately reaches a remarkable 1.11 gFID on generation tasks.

To sum up, our contribution could be summarized as follows:

- • We propose Visual Tokenizer Pre-training (VTP), a framework integrating contrastive, selfsupervised, and reconstruction objectives to build a perception-oriented tokenizer pre-training for generative models.
- • We demonstrate a new scaling property for visual tokenizers: overcoming the early saturation of reconstruction-only training, downstream generation quality consistently improves as VTP pre-training scales in compute, parameters, and data.
- • VTP pushes the boundaries of unified understanding and generation of visual encoder, achieving an impressive 1.11 gFID on ImageNet generation alongside 0.36 rFID, 78.2% zero-shot, and 85.7% linear probing accuracy.

### 2. Related Work

#### 2.1. Pre-training and Representation Learning

Pretraining is typically a scalable paradigm for boosting downstream task performance by first optimizing models on large-scale data with specific objectives. The early paradigm relied on supervised pretraining—such as ImageNet classification (Dosovitskiy, 2020; He et al., 2016)—and transferred weights to downstream tasks like detection (Ren et al., 2015) and segmentation (Xiao et al., 2018). A recent paradigm shift has focused on weakly-supervised and unsupervised methods to enable pretraining at larger scales. For instance, CLIP (Radford et al., 2021) uses image-text contrastive learning with weak supervision by minimizing the distance between image and text features. SigLIP (Zhai et al., 2023) further optimizes this process via a sigmoid loss for large-scale training. Another branch, self-supervised learning (SSL), learns directly from unlabeled data. Methods like MAE (He et al., 2022) and BEiT (Bao et al.) adopt masked image modeling (MIM), training models to reconstruct masked patches. DINO (Caron et al., 2021) utilizes self-distillation to enforce multi-view classifi-

cation consistency. iBOT (Zhou et al., 2021) and DINOv2 (Oquab et al., 2023) combine MIM with self-distillation to learn more generalized representations.

Despite these advances, within the explicitly decoupled, two-stage framework of LDMs (Rombach

- et al., 2022)—comprising a visual tokenizer followed by a generative model—how to pretrain the first-stage tokenizer to enhance second-stage generative performance has not been systematically explored.

2.2. Latents with Pre-trained Representations

Previous work has explored the use of visual representations to structure the latent space, which falls into two categories. The first employs a distillation objective: VA-VAE (Yao et al., 2025) aligns its latent space with features of visual foundation models to alleviate the trade-off between reconstruction and generation. ImageFolder (Li et al., 2024) decouples semantic and pixel-level feature spaces to improve autoregressive generation. MAETok (Chen et al., 2025a) enhances latent representations by incorporating DINOv2 features into its MIM pre-training objective. REPA-E (Leng et al., 2025)concurrently optimizes the feature space of the VAE during DiT training by leveraging supervision from a pre-trained foundation model. I-DeTok (Yang et al., 2025) improves the latent space’s suitability for both autoregressive and diffusion models generation via a joint pre-training strategy that employs MIM and noise injection. The second strand directly utilizes pre-trained representations for generation. For instance, BLIP3-o (Chen et al., 2025b) regresses SigLIP features and employs an SD-XL-based (Podell

- et al., 2023) decoder to boost efficiency. Recently, RAE (Zheng et al., 2025) leverages DINOv2 features and trains a separate pixel decoder for reconstruction.

However, these methods are inherently limited by existing foundational models, often leading to a low performance ceiling or substantial reconstruction loss. Concurrently, while previous studies achieved performance improvements under specific configurations, the scalability of the proposed methods generally remains unverified.

### 3. Visual Tokenizer Pre-training

Our work introduces a scalable visual tokenizer pre-training paradigm that benefits generation. To this end, we integrate representation learning objectives with the conventional reconstruction loss to learn visual representations that are semantically rich, accurate in reconstruction, and generation-friendly (see fig. 3).

#### 3.1. Architecture

Leveraging its flexibility in learning visual representations, our visual tokenizer uses a fully Vision Transformer (ViT) architecture. In line with standard autoencoder designs, we introduce a bottleneck that maps visual information into a 𝑑-dimensional latent space. Encoder features are leveraged by the text encoder, EMA teacher, and pixel decoder to facilitate their distinct training objectives.

#### 3.2. Visual Reconstruction

Given an image 𝐼 ∈ ℝ3×𝐻×𝑊, we compress it into a latent space ℝ𝑑×𝐻/16×𝑊/16 using a visual tokenizer and subsequently reconstruct into 𝐼′ via a pixel decoder, which lifts latents back to the feature space, refines them with N ViT blocks, and reconstructs images in pixel space through a final pixel-shuffle layer.

Contrastive Learning Reconstruction Self-Supervised Learning

global patch tokens

global patch tokens

global cls tokens

local cls tokens

global cls tokens

Pixel Decoder

.

text features cls tokens

###### EMA

##### Text Encoder Vision Transformer Teacher

masking

I

I GG

Image Caption

.

global crops

GG

local images crops

globalcrops L

images

Visual Tokenizer Pre-training

[Figure 4]

[Figure 5]

Pixel Decoder

Diffusion

Diffusion Transformer

Vision Transformer

Z Transformer

Z

generated latents

Diffusion Training Diffusion Sampling

add noise

Figure 3 | Overwiew of Visual Tokenizer Pre-training (VTP). By integrating representation learning (image-text contrastive (Radford et al., 2021) and self-supervised learning (Oquab et al., 2023)) with reconstruction within a Vision Transformer Auto-Encoder, we find that VTP exhibits a well-behaved scaling property for generative performance.

The reconstruction task is challenged by the poor compatibility of GAN loss (Esser et al., 2021) with the ViT architecture, which causes large gradient norms and low training stability. To address this, we employ a two-stage training strategy. In the first stage (i.e., the pre-training stage), all parameters are jointly optimized by minimizing a composite loss function comprising the L1 loss and a perceptual loss Lperceptual (Zhang et al., 2018) between 𝐼 and 𝐼′. During the second stage, the visual tokenizer remains frozen while the pixel decoder is fine-tuned with a GAN objective to improve fidelity.

The overall reconstruction loss Lrec during pre-training is defined as:

Lrec = L1 + Lperceptual (1)

#### 3.3. Self-Supervised Learning

Following DINOv2 (Oquab et al., 2023), our self-supervised learning framework comprises two components: masked image modeling (MIM) (He et al., 2022; Zhou et al., 2021) and self-distillation (Caron et al., 2021).

For a given image 𝐼, we apply data augmentation to obtain global and local views 𝐼global and 𝐼local. In MIM, adhering to (Zhou et al., 2021), 𝐼global is patch-embedded and fed directly to an EMA teacher, while its masked version is processed by the visual tokenizer, optimizing the complementary

masking loss Lmim. For self-distillation, similar to (Caron et al., 2021), 𝐼global and 𝐼local are passed to the visual tokenizer, and 𝐼global to the EMA teacher, with the cross-entropy loss Ldino applied to their pseudo-label predictions.

Therefore, the overall self-supervised learning loss is defined as:

L𝑠𝑠𝑙 = L𝑚𝑖𝑚 + L𝑑𝑖𝑛𝑜 (2)

#### 3.4. Contrastive Learning

Given a batch of image-text pairs, we encode the image 𝐼 and text 𝑇 using a visual tokenizer and a text encoder, respectively, to obtain their visual and textual features. Following CLIP, we then maximize the similarity of the corresponding (positive) image-text pairs while minimizing the similarity of the remained non-corresponding (negative) pairs. This objective is formulated as the contrastive loss Lclip.

#### 3.5. Overall Objective

Building upon the preceding components, we integrate them into a unified pre-training framework. The overall training objective for our visual tokenizer pre-training is formulated as a weighted combination of the aforementioned losses:

Ltotal = 𝜆recLrec + 𝜆sslLssl + 𝜆clipLclip (3)

where 𝜆rec > 0, 𝜆ssl ≥ 0, and 𝜆clip ≥ 0 are balancing coefficients that control the contribution of each objective. This multi-task learning scheme enables the model to concurrently develop high-fidelity reconstruction capability, semantically rich representation learning, and cross-modal alignment, thereby establishing a robust and scalable visual tokenizer for diverse generation tasks.

#### 3.6. Batch Sampling

We observe a significant disparity in optimal batch sizes across different training paradigms. Contrastive learning frameworks like CLIP demand extremely large batches (e.g., 16k or 32k), while self-supervised and reconstruction objectives are typically effective with orders of much smaller batches (e.g., 4k).

Given an input batch of 𝐵 image-caption pairs, all samples are used for CLIP training, e.g. 𝐵clip = 𝐵. 𝐵ssl and 𝐵rec are random sampled from 𝐵 to accommodate the divergent batch size requirements of self-supervised learning and reconstruction.

### 4. Experiments

#### 4.1. Implementation Details

Pre-training Our model architecture builds upon the Vision Transformer (ViT) implemented in (Siméoni et al., 2025). We incorporate QKNorm (Henry et al., 2020) to enhance training stability. We employ a 12-layer transformer with a hidden dimension of 768 as the text encoder, and a 4-layer ViT-Large layer as the pixel decoder for fast experimentation. In designing the latent bottleneck, we primarily adopt a dimension of 64, following (Ma et al., 2025), to balance semantic comprehension with reconstruction quality. An ablation study on this configuration is conducted by varying the dimension to 256. We use an internally filtered version of DataComp-1B (Deng et al., 2009) with 277M samples for tokenizer pretraining, and ImageNet (Deng et al., 2009) for downstream DiT training. We set 𝐵clip = 16𝑘, 𝐵ssl = 4𝑘 and 𝐵rec = 2𝑘. For weighting, we set 𝜆rec = 0.1, while 𝜆clip and 𝜆ssl are set to either 0 or 1. We find that a smaller reconstruction weight contributes to improved generative performance. For self-supervised and contrastive pretraining implementation, we closely follow the established practices of DINOv2 (Oquab et al., 2023) and OpenCLIP (Ilharco et al., 2021).

Arch. FLOPs #Params rPSNR gFID CNN (Rombach et al., 2022) 389.4G 70.3M 30.63 59.53 ViT-B (Dosovitskiy, 2020) 87.7G 171.2M 30.72 58.40 ViT-L (Dosovitskiy, 2020) 311.1G 607.2M 31.28 53.51

- Table 1 | AutoEncoder Performance with Different Architectures. ViT serves as a comparable alternative to CNNs, enabling simpler pre-training scaling experiments.

Figure 4 | Reconstruction Only Training Target CANNOT Lead to Effective Scaling for Downstream Diffusion Models. As training progresses, the tokenizer’s reconstruction performance improves, while its generative performance degrades concurrently. It reveals the inadequacy of pure reconstruction tasks for scalable tokenizer pre-training.

Downstream DiT training & evaluation We train the Diffusion Transformer (DiT) (Peebles and Xie, 2023) under a fixed configuration to evaluate the generative capability of our visual tokenizer. Specifically, we follow LightningDiT (Yao et al., 2025) as a strong baseline. We report FID-10k scores obtained with a LightningDiT-B (Yao et al., 2025) model trained on ImageNet (Deng et al., 2009) for 80 epochs under a consistent protocol. For the reconstruction evaluation, the performance of all tokenizers is assessed on the standard ImageNet validation set at a resolution of 256. For most cases, we report rFID as the reconstruction metric. For understanding evaluation, we evaluate representation performance on ImageNet using linear probing. We do not employ the feature enhancements common in the DINO (Oquab et al., 2023; Siméoni et al., 2025) series, which can substantially increase linear probing scores by leveraging multi-layer features. Instead, we probe only the reduced-dimensionality features from the bottleneck, thereby directly evaluating the inherent properties of the latent features, and report ImageNet Top-1 acc. as the understanding metric.

#### 4.2. Auto-Encoder with Vision Transformers

We begin by demonstrating that a Vision Transformer (ViT) can serve as an effective substitute for CNNs in standard reconstruction tasks. We construct a ViT visual tokenizer with a symmetric encoder and decoder. It has the specification of f16d64, where ‘𝑓’ denotes downsample ratio (or patch size for ViT) and ‘𝑑’ denotes bottleneck dimension. A two-stage training pipeline discussed in section 3.2 is adopted to enhance training stability. Then, we implement the convolutional LDM architecture (Rombach et al., 2022) under the same specifications. We use ImageNet at 256 resolution for training and testing.

As illustrated in table 1, we evaluate their reconstruction and generation performance, observing that ViT-L achieves a reconstruction PSNR of 31.28 and a gFID of 53.51, on par with LDM. While it utilizes more parameters, it requires lower computational cost. These findings are consistent with previous observations (Hansen-Estruch et al., 2025; Teng et al., 2025), suggesting that the simple design of this ViT tokenizer architecture is effective.

#### 4.3. Scaling up Visual Tokenizer Pre-training

Our work focuses on how to scale up the tokenizer pre-training to improve the model’s capabilities for downstream generative training.

We conduct three distinct scaling-up experiments. For these experiments, we employ a ViT-L backbone as the encoder and a lightweight decoder composed of 4 ViT-L layers to facilitate rapid training and inference. All models are trained on the 277M DataComp-filtered dataset introduced

1010 1011

1010 1011

- Figure 5 | Scalability of CLIP+AE & SSL+AE Visual Tokenizer Pre-training. Scaling properties under different strategies and bottleneck dimensions. Our method shows correlated growth in generation and comprehension with compute, while VAE-based tokenizer performance rapidly saturates.

above.

Scaling with reconstruction only CANNOT help generation. Initially, we scale up the training computation for a standard reconstruction tokenizer. As illustrated in fig. 4, we observe a scaling paradox: the model’s reconstruction performance improves substantially with increased training compute, with rFID improving from 2.0 to 0.5. However, its generative performance in fact slightly degrades, as indicated by the gFID rising from 55.04 to 58.56. We posit that this phenomenon occurs because the reconstruction objective effectively guides the model to capture low-level details but provides insufficient incentive for learning high-level semantic representations, which are crucial for generation. Reconstruction task itself does not exhibit scalability in pretraining for downstream generation.

Scaling with different understanding tasks helps generation in a similar way. Then, we scale up visual tokenizer pre-training with the assistance of representation tasks. As described in section 3, we integrate the reconstruction task with either image-text contrastive learning (CLIP) (Radford et al., 2021) or self-supervised learning (SSL, specifically DINOv2) (Oquab et al., 2023) in a joint training framework. These two hybrid approaches are denoted as CLIP+AE and SSL+AE, respectively. To further substantiate the robustness of our conclusions, we include an additional experimental configuration with a latent dimension of 𝑑 = 256. For a fair comparison, all Autoencoders (AEs) across different latent dimensions were trained under an identical computational budget. We concurrently monitor four key metrics: understanding performance, reconstruction fidelity, generation quality, and

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1010 1011

- Figure 6 | Scalability of CLIP+SSL+AE Visual Tokenizer Pre-training. Under the same computational budget, the f16d64 tokenizer trained with joint CLIP and SSL representation learning achieves the best performance in both generation and comprehension.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

(a) Data Scaling. (b) Encoder Scaling. (c) Decoder Scaling.

- Figure 7 | Scalability of data and parameters. We observe a new scaling property: the DiT generation performance within fixed training FLOPs increases while its tokenizer has larger model sizes and training data.

training FLOPs for a comprehensive evaluation. Our experimental results, summarized in fig. 5, lead to the following key observations:

(1) Feasibility of Hybrid Objectives: Hybrid training combining representation learning with reconstruction is viable. As evidenced by the fig. 5 (c)&(f), traditional autoencoders (AEs) trained solely on reconstruction maintain low understanding performance. In contrast, when augmented with representation learning objectives—either CLIP or SSL—both understanding and reconstruction metrics exhibit stable, simultaneous improvement.

- (2) Negative Impact of Pure Reconstruction: Solely relying on reconstruction proves counterproduc-

tive for downstream generation tasks. The fig. 5 (b)&(e) illustrates a negative yield in reconstructiononly AEs: as the computational budget increases, reconstruction performance improves but the generation performance degrades.

- (3) Understanding as the Key Driver: The integration of semantic understanding tasks counteracts

this negative effect and emerges as the dominant factor for improving generation. The reversal of this trend is visible in the fig. 5 (a)&(d) and (b)&(e). Specifically, our visual tokenizer pre-training, which jointly optimizes for reconstruction and representation learning, enables continuous, concurrent improvement in reconstruction, understanding, and generation as pre-training scales. Conversely, focusing exclusively on reconstruction optimization yields superior reconstruction, but leads to the stagnation of both understanding and generation performance. These observations imply that understanding is the key driving force necessary for effective generation.

- (4) Generality of Representation Learning: Diverse representation learning paradigms, including

CLIP and SSL, consistently enhance generation performance. Despite significant differences in their training frameworks, they share a critical mechanism: enriching the semantic understanding within the latent space. Although their scaling behaviors differ slightly, both methods substantially improve the efficacy of visual tokenizer pre-training for downstream generative tasks. This observation also suggests that new and emerging representation learning techniques can be seamlessly integrated to establish even better performance bounds.

Scaling with multiple understanding tasks gets better performance. Subsequently, we introduce, to the best of our knowledge, the first integration of contrastive, self-supervised, and reconstruction objectives (CLIP+SSL+AE) for visual tokenizer pre-training. Our experiments demonstrate that this training paradigm is feasible and stable. This multi-objective framework enables the tokenizer to capture multi-scale features, enhancing both semantic alignment and spatial fidelity. As shown in fig. 6, our method under the f16d64 setting achieves a higher generative upper bound (gFID=27.8) alongside better understanding performance (74.9% linear probing accuracy) under a fixed computational budget. All subsequent data and parameter scaling experiments are based on this pre-training configuration.

#### 4.4. Scaling Properties with Parameters

VTP demonstrates a new interesting parameter scalability, as its downstream DiT generative performance improves consistently with increased previous tokenizer model size.

We first investigate encoder scaling by training three ViTs of varying sizes using CLIP+SSL+AE and a baseline AE. As shown in fig. 7 (b), the generative performance of the AE remains stagnant at about 57, regardless of the model capacity (from 20M to 300M parameters). In contrast, VTP exhibits a clear scaling trend: its gFID improves steadily from 31.28 to 26.12 as the model size grows, forming a welldefined parameter scaling curve. We then proceed to scale up the pixel decoder. Our findings indicate that this architectural expansion also leads to a correlated improvement in generative performance, with the gFID score decreasing from 26.12 to 24.08.

#### 4.5. Scaling Properties with Data

The scale of training data is also crucial for the generalization ability of the tokenizer. To validate this, we constructed four subsets of varying scales—100K, 1M, 10M, and 100M—by randomly sampling from the Datacomp-1B dataset. We trained both VTP-ViT-Large and AE-ViT-Large architectures on these subsets for 1.1 billion samples each and evaluated their generation performance. The results are summarized in fig. 7 (a).

We draw the following observations from the results: First, VTP consistently outperforms the conventional autoencoder across all data scales. More importantly, the generative performance of the autoencoder shows negligible improvement with increased data, with its FID score merely decreasing from 58.37 to 56.71. In stark contrast, the performance of VTP improves significantly as the volume of training data grows, with its FID score substantially improving from 47.59 to 27.45. Notably, the downstream DiT training FLOPs remained strictly identical. This compellingly demonstrates that the introduced representation learning effectively enhances the data scalability of the visual tokenizer.

[Figure 6]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### adding the CLIP loss to VTP makes a better text generation

(a) VTP scaling properties generalize to T2I generation. VTP pretrained with representation learning objectives consistently improves with increased compute, achieving better generation performance than reconstruction-only tokenizers.

(b) Ablation on Semantic losses. Adding CLIP loss to VTP significantly improves the text rendering ability in T2I.

- Figure 8 | Text-to-Image Experiments on LAION.

### 5. Scaling to Text-to-Image Generation

To validate the generalizability of VTP’s scaling properties beyond class-conditional generation on ImageNet, we extend our evaluation to text-to-image (T2I) generation tasks on the LAION dataset (Schuhmann et al., 2021).

We train VTP tokenizers and evaluate T2I generation performance across different pre-training compute budgets. For the downstream generative model, we adopt a DiT-XL architecture for generation. During training, images are resized with a short edge randomly sampled between 256 and 512. The results are shown in fig. 8a. More details will be provided in the supplementary materials.

VTP scaling properties generalize to T2I. We observe two key findings: (1) Tokenizer with representation learning objectives achieves significantly faster convergence than the reconstructiononly AE baseline, demonstrating that semantic-aware tokenizer pre-training substantially improves training efficiency for downstream T2I generation. (2) As the tokenizer pre-training compute increases, the downstream generative model continues to improve, confirming that VTP’s scaling properties are not limited to class-conditional ImageNet generation but generalize to the more challenging T2I setting.

CLIP loss improves text rendering. We further ablate the effect of different semantic losses on T2I generation quality (fig. 8b). Similar to the results in fig. 6 on ImageNet, as the variety of perception losses incorporated into VTP pre-training increases, we observe a steady improvement in downstream T2I generation performance. Furthermore, we observe that the tokenizer trained with the CLIP loss exhibits a significant advantage in text rendering for text-to-image generation, substantially outperforming both AE and SSL+AE and tokenizers.

### 6. Further Scaling and Comparison

We further scale VTP pre-training with larger training compute across three symmetric ViT architectures (VTP-S, VTP-B, VTP-L) and summarize results in table 3 and table 2.

Generation w/o guidance Generation w/ guidance rFID↓

Tokenizer Gen Model Params

Method

Epochs

ZeroShot↑

Linear Probe↑ gFID↓ IS↑ Prec.↑ Rec.↑ gFID↓ IS↑ Prec.↑ Rec.↑

Perception or Unified Tokenizer Baselines

SigLIP (Zhai et al., 2023) - 80.5 - - - - - - - - - - MAE (He et al., 2022) - - 85.9 - - - - - - - - - DINOv2 (Oquab et al., 2023) - - 86.7 - - - - - - - - - VILA-U (Wu et al., 2024) 1.80 73.3 - - - - - - - - - - UniTok (Ma et al., 2025) 0.41 70.8 - 1.4B - 2.51 216.7 0.82 0.57 2.77 227.5 0.81 0.57

###### Convergence Efficiency

REPA (Yu et al., 2024) 0.61 - - 675M 80 7.90 122.6 0.70 0.65 - - - DDT (Wang et al., 2025) 0.61 - - 675M 80 6.62 135.2 0.69 0.67 1.52 263.7 0.78 0.63 VA-VAE (Yao et al., 2025) 0.28 - - 675M 80 4.29 - - - - - - REPA-E (Leng et al., 2025) 0.28 - - 675M 80 3.46 159.8 0.77 0.63 1.67 266.3 0.80 0.63

- RAE (Zheng et al., 2025) 0.57 - 84.5 675M 80 4.28 - - - - - - RAE (Zheng et al., 2025) 0.57 - 84.5 835M 80 2.16 214.8 0.82 0.59 - - - VTP (Ours) 0.36 78.2 85.7 675M 80 2.62 197.8 0.79 0.62 1.44 238.2 0.80 0.63 VTP (Ours) 0.36 78.2 85.7 1.0B 80 2.03 219.4 0.80 0.62 - - - -

Long Period Training

DiT (Peebles and Xie, 2023) - - - 675M 1400 9.62 121.5 0.67 0.67 2.27 278.2 0.83 0.57 SiT (Ma et al., 2024) - - - 675M 1400 8.61 131.7 0.68 0.67 2.06 270.3 0.82 0.59 VA-VAE (Yao et al., 2025) 0.28 - - 675M 800 2.17 205.6 0.77 0.65 1.35 295.3 0.79 0.65 REPA (Yu et al., 2024) 0.61 - - 675M 800 5.78 158.3 0.70 0.68 1.29 306.3 0.79 0.64 DDT (Wang et al., 2025) 0.61 - - 675M 400 6.27 154.7 0.68 0.69 1.26 310.6 0.79 0.65 REPA-E (Leng et al., 2025) 0.28 - - 675M 800 1.70 217.3 0.77 0.66 1.15 304.0 0.79 0.66

- RAE (Zheng et al., 2025) 0.57 - 84.5 676M 800 1.87 209.7 0.80 0.63 1.41 309.4 0.80 0.63 RAE∗ (Zheng et al., 2025) 0.57 - 84.5 839M 800 1.51 242.9 0.79 0.63 1.13 262.6 0.78 0.67 VTP (Ours) 0.36 78.2 85.7 675M 600 1.85 232.3 0.79 0.63 1.11 279.5 0.79 0.67

#### Table 2 | Generation Performance on ImageNet 256×256.

Scalability Compared to Fixed Representation AutoEncoders. Compared to concurrent work RAE (Zheng et al., 2025). A distinctive advantage of VTP is its scalability. As shown in Table 3, under identical DiT training configurations, VTP’s downstream generation improves consistently as the tokenizer scales from S to L, whereas RAE’s performance degrades at larger scales. Moreover, since VTP always involves reconstruction during training, it preserves fine-grained details significantly better than RAE (see fig. 9a).

Tokenizer RAE VTP Small 3.50 5.46 Base 4.28 3.88 Large 6.09 2.81

Unified Performance Frontier. Our final model achieves 0.36 rFID, 78.2% zero-shot accuracy, and 85.7% linear probing accuracy on ImageNet, surpassing prior unified tokenizers VILA-U (Wu et al., 2024) and UniTok (Ma et al., 2025).

Table 3 | Scalability comparison. LightningDiT gFID↓ at 80 epochs (w/o guidance) with identical DiT training. We take RAE’s results from its original paper.

Superior Generation Without Architecture Modification. Built upon the standard LightningDiT (Yao et al., 2025), VTP-L achieves superior 1.11 gFID with

guidance, surpassing all prior methods (table 2). Moreover, VTP attains 2.60 and 2.03 gFID without guidance in only 80 epochs, demonstrating remarkably fast convergence. More details will be provided in supplementary materials.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Input

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

VTP

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

RAE

(a) Reconstruction Comparison with RAE. Visual tokenizer pre-training with reconstruction target enables better reconstruction capacity than direct representation encoder transfer.

(b) Convergence on ImageNet. With 80 epochs of training, VTP achieves 2.61 gFID, outperforming VA-VAE (4.29) and RAE (4.28).

- Figure 9 | Visualizations of Reconstruction and Generation.

### 7. Conclusion

In this work, we redesign a scalable paradigm for visual tokenizer pre-training that substantially enhances generative performance and propose VTP. We demonstrate that perception-oriented tokenizer pre-training unlocks a new scaling law for generation: (1) semantic understanding is the key driver for improving generation, and (2) with this understanding, the visual tokenizer achieves scalable performance on generative tasks. In contrast to traditional tokenizers pre-trained solely on reconstruction—whose performance saturates with a small scale—our approach consistent attains a significant gain in generative performance when scaling the compute budget, model size, data scales. Hope VTP could inpire the following research on visual tokenizers.

### References

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In International Conference on Learning Representations.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. In Forty-second International Conference on Machine Learning.

Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are effective tokenizers for diffusion models. In Forty-second International Conference on Machine Learning, 2025a.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal modelsarchitecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025b.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale

hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Philippe Hansen-Estruch, David Yan, Ching-Yao Chung, Orr Zohar, Jialiang Wang, Tingbo Hou, Tao Xu, Sriram Vishwanath, Peter Vajda, and Xinlei Chen. Learnings from scaling visual tokenizers for reconstruction and generation. arXiv preprint arXiv:2501.09755, 2025.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. arXiv preprint arXiv:2010.04245, 2020.

Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. Computer software. An open-source implementation of CLIP.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Theodoros Kouzelis, Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. Boosting generative image modeling via joint image-feature synthesis. arXiv preprint arXiv:2504.16064, 2025.

Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.

Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024.

Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer vision (ECCV), pages 418–434, 2018.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.

Jiawei Yang, Tianhong Li, Lijie Fan, Yonglong Tian, and Yue Wang. Latent denoising makes good visual tokenizers. arXiv preprint arXiv:2507.15856, 2025.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021.

