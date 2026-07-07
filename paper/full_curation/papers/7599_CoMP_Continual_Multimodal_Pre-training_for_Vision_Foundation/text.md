# arXiv:2503.18931v2[cs.CV]16May2025

## COMP: Continual Multimodal Pre-training for Vision Foundation Models

Yitong Chen1,2∗ Lingchen Meng1∗ Wujian Peng1,2 Zuxuan Wu1,2† Yu-Gang Jiang1 1Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University 2Shanghai Innovation Institute https://slimm-x.github.io/comp

### Abstract

Pre-trained Vision Foundation Models (VFMs) provide strong visual representations for a wide range of applications. In this paper, we continually pre-train prevailing VFMs in a multimodal manner such that they can effortlessly process visual inputs of varying sizes and produce visual representations that are more aligned with language representations, regardless of their original pre-training process. To this end, we introduce COMP, a carefully designed Continual Multimodal Pre-training pipeline. COMP uses a Continual Rotary Position Embedding to accommodate visual inputs with different resolutions, and an Alignment Loss between visual and textual features for better cross-modal alignment. After continual pre-training, leading VFMs like DINOv2, SigLIP and AIMv2 achieve remarkable improvements not only in multimodal understanding tasks but also in generic classification and segmentation tasks. Remarkably, COMP-AIMv2 achieves scores of 64.9 on ChartQA with a 0.5B LLM, while maintaining an 87.3% accuracy on ImageNet-1K and a 51.8 mIoU on ADE20K under frozen chunk evaluation.

### 1 Introduction

Pre-training Vision Foundation Models (VFMs) capable of extracting transferable representations for various downstream tasks has been a long pursuit of the computer vision community. The key to pre-training is to scale up models and data through constructing strong supervisory signals with weak-strong augmentations in vision-only pretraining [1, 2, 3, 4] or cross-modality alignment in vision-language pre-training [5, 6, 7, 8]. These VFMs often demonstrate strong performance for a variety of downstream tasks, and can be easily combined with Large Language Models (LLMs) by designing lightweight adapters that project visual features into text space.

In this paper, we revisit these widely used VFMs such as vision-only pre-training DINOv2 [2], visionlanguage pre-training SigLIP [6] and AIMv2 [8]. We argue that these prevailing VFMs, regardless of their pre-training procedures, can be further boosted through continual multimodal pre-training. This allows VFMs to (1) better process visual inputs of arbitrary sizes when used as vision encoders of LMMs; (2) produce outputs that are more aligned with language representations, thereby improving multimodal understanding, significantly benefiting encoders from vision-only pre-training.

On the one hand, equipping VFMs with the ability to deal with images of varying sizes is the core of visual understanding, as image resolutions directly impact the richness of the information within an image. However, existing approaches treat “an image as worth 16×16 words"[9], resizing all images to a predefined size. This one-size-fits-all strategy results in the loss of critical details, impeding the model’s ability to perceive fine-grained information. This is particularly detrimental in tasks that

∗ Equal contributions. † Corresponding author.

Preprint. Under review.

detailed image caption [EOS]

|Alignment Loss|
|---|

|Decoding Loss|
|---|

P✓(Xt) P✓(Xt|Hv)

###### LLM P✓

|Adapter F|
|---|

Visual Tokens Hv

Text Tokens Ht

Visual Features Z

[Figure 1]

VFM V 

Native Res. Image Xv

Text Xt

- Figure 1: Overview of COMP. Our method accepts an image at native resolution and its corresponding text. Then, in addition to training through text decoding in next-token prediction paradigm, we also explicitly project the visual features into the language space of LLM using Alignment Loss.

demand high-resolution inputs, such as chart understanding[10], document parsing [11], and finegrained recognition [12]. Recent works [13, 14, 2, 6, 7, 8] have attempted to address this challenge by improving bilinear interpolation of positional embeddings and incorporating multi-resolution training. However, they still struggle with real-world scenarios involving diverse input resolutions, as they remain constrained by the limitations of resolution extrapolation.

On the other hand, we believe there is still a representation gap between VFMs and LLMs, which results from their distinct training objectives and data modalities during pre-training [15]. To bridge this gap and enable LMMs to better understand visual inputs, the mainstream approach [16, 17, 18] involves training an adapter that projects the visual embeddings of VFMs into the textual embedding space of LLMs, typically through next-token prediction for text tokens. However, relying solely on text-based supervision is insufficient to effectively and directly reduce this gap [19], particularly when VFMs have not undergone vision-language alignment pretraining [20, 21].

To address these challenges, we introduce COMP, a continual pre-training pipeline that is carefully designed to enhance exisiting VFMs. Specifically, COMP builds upon (1) C-ROPE, a Continual Rotary Position Embedding for vision models, which is operated by adding the standard RoPE2D [22] with the learned 1D position embedding, to support native resolution continual pre-training; (2) Alignment Loss, a cross-entropy loss between visual and textual features through language prototypes, to align multimodal representations between pre-trained VFMs and LMMs.

As shown in Fig. 1, our method accepts an image with native resolution and its corresponding text. Then, in addition to training through next-token prediction on the text, we also explicitly project the visual features into the language space using Alignment Loss by word embedding of LLM. With a three-stage continual pre-training, our models excel not only on multimodal understanding, but also in downstream tasks such as classification and segmentation. Our contribution can be summarized as:

- 1. We propose a continual multimodal pre-training pipeline COMP, with C-ROPE and Alignment Loss, to enable VFMs supporting native resolution, and aligning representation space of LLM.
- 2. With the COMP, we present COMP-SigLIP, COMP-AIMv2 and COMP-DINOv2, which achieve remarkable improvements not only in multimodal understanding but also in traditional visual tasks.
- 3. Based on COMP-VFMs, we introduce COMP-MM-1B and COMP-MM-7B, which significantly outperforms all other methods under the similar training data size.
- 4. We conduct comprehensive experiments and ablation studies on different models and different tasks, which provide useful insights behind the design choices.

### 2 COMP

Our goal is to empower a pre-trained vision foundation model (VFM) with the ability to process images at their native resolution while aligning its encoder features with the representation space

pv loss : pt

######  pt · log pv

L⇥ Visual Features Z

sg

Softmax S.K.

Feedforward

Cv Ct

|WordPEmbeddingrototype asMPrototypeapping W|
|---|

MatMul

Softmax

Add

Fv Ft

Rotary R

sg

q k

v

Pooling Pooling

|V  & F|
|---|

|P✓|
|---|

Interpolation

PatchEmbed

Xv Xt

Image-text pair

Image Xv Pos. Embed Epos

- Figure 2: Left: C-ROPE. For ease of visualization, the projection layers Projq,k,v,o and scale operators are omitted. We leverage both absolute learned position embedding and relative RoPE-

- 2D [22] to capture richer positional information. Right: Alignment Loss. We illustrate it in the case

of one single pair of global vision and text features Fv and Ft for simplicity. Fv and Ft are mapped by frozen learned prototype W, i.e., the word embedding of LLMs. Then, they are converted into normalized probabilities using the Softmax function and iterative Sinkhorn-Knopp algorithm [23], respectively. Finally, cross-entropy is applied as the loss. To prevent information leakage, the text features are extracted without image prefixes.

of a pre-trained LLM. To achieve this, as shown in Fig. 1, we propose a continual multimodal pretraining pipeline that improves existing VFMs so that they can handle native resolution inputs using C-ROPE (Sec. 2.1) and better align with language space through carefully designed loss functions: the text decoding loss (Sec. 2.2) and the cross-modal alignment loss (Sec. 2.3). These components are integrated into a three-stage training (Sec. 2.4), ensuring effective adaptation and alignment.

#### 2.1 Native resolution adaptation

Vision encoders often use fixed-size inputs during the pre-training stage, and thus struggle to handle images with varying resolutions, particularly high-resolution images for fine-grained visual understanding. While training on images of different sizes is a straightforward solution, it is particularly challenging due to the predefined shape of position embeddings in vision transformers. A common approach is to interpolate the original position embeddings in an online manner to accommodate different input resolutions, yet the results are unsatisfactory [13, 14].

Inspired by the success of Rotary Position Embedding (RoPE) [24] that demonstrates strong extrapolation capabilities in NLP [25, 26] and CV [27, 28, 29, 30, 31], we aim to build upon RoPE-2D [22] to handle a sequence of visual tokens. Unlike previous methods that only rely on RoPE-2D, which is neither a data-efficient nor a training-friendly approach (see Tab. 5a), we leverage both absolute and relative embeddings to ensure a smooth transition from the pre-trained vanilla ViT to arbitrary resolutions, and capture richer positional information so as to handle a variety of high-resolution inputs. We refer to this combination as C-ROPE.

Specifically, as shown in Fig. 2 (left), a 2D image of resolution (H,W) is patchified into N = HW/P2 patches xp ∈ RN×(P

2·C), where P denotes the patch size of vision encoder and C indicates the number of image channel. The image encoding processes can be expressed as:

z0 = [x1pE;x2pE;··· ;xNp E] + Int(Epos), (1) qi,ki,vi = Projq(zi),Projk(zi),Projv(zi), (2)

(Rqi)⊤(Rki) c

)v), (3)

yi = zi + Projo(Softmax(

zi+1 = yi + FFN(yi), where i = 0,··· ,L − 1, (4) where E ∈ R(P

2·C×Dv) and Epos ∈ RN×D

v indicate patch embedding and learnable position embedding, Dv indicates the hidden dimension of vision encoder, Int(·) represents bilinear interpolation,

Proj(·) is the projection layer, FFN(·) denotes standard feed-forward network, and L denotes the number of encoder layers. In particular, R in Eq. (3) is the 2D rotary matrix[22]:

  

  .

cosxθ −sinxθ 0 0 sinxθ cosxθ 0 0

Rx,y =

0 0 cosyθ −sinyθ 0 0 sinyθ cosyθ

#### 2.2 Text-supervised generative pre-training

Text-supervised generative pre-training is widely used in large multimodal models (LMMs). It extends the standard text-only autoregressive next-token prediction framework [32, 33, 34] to visual inputs, mapping visual features into the input layer of a large language model via query-driven cross-attention [35, 36, 37] or projection [16]. We adopt the projection-based multimodal framework due to its simplicity and effectiveness. Formally, the projection Fψ can be defined as follows:

Hv = Fψ(Z), (5) where Z = zL in Eq. (4). Then, the image tokens Hv are fed into the input layer of the LLM Pθ, serving as the condition for the corresponding text Xt. The text decoding loss can be expressed as:

V +T

1 T

log Pθ(Xi|X<i,Hv), (6)

Ldec = −

i=V +1

where V and T denote the number of visual and textual tokens, respectively. To support the autoregressive generation, the image-grounded text decoder utilizes causal self-attention mechanisms.

#### 2.3 Vision-language representation alignment

Thanks to text-supervised generative pre-training and decoding loss, the vision encoder can be optimized using paired images and texts. However, the supervision is too distant for directly optimizing the vision model, especially when the original pre-training of vision encoder does not involve vision-language alignment, such as in image-only SSL of DINOv2 [2] (see Tab. 4). To bridge the modality gap between the vision and language models, we utilize direct vision-language representation alignment by encoding visual and textual features through the VFMs and LLMs, respectively.

Some previous approaches [38, 39, 40] employ a contrastive loss for cross-modal alignment, which usually requires large batch sizes and even an auxiliary text encoder. To avoid these constraints and inspired by the knowledge distillation in DINO [1, 2], we align visual and textual representations by treating the word embeddings W of the LLM as prototypes [19]. Specifically, as shown in Fig. 2 (right), we first obtain global visual and its corresponding text features Fv and Ft ∈ RB×D

t through the VFM and LLM by parameter-free global average pooling, respectively, as follows:

Fv = Pool(Hv),Ft = Pool(Pθ(Xt)), (7) where B denotes the mini-batch size, Dt represents the hidden dimension of LLM and Xt is a piece of corresponding text to Hv. And then, we map Fv, Ft into language space by the prototype W ∈ RD

t×K as follows:

Cv = W⊤Fv,Ct = W⊤Ft, (8) where K indicates the vocabulary size of LLM. To prevent information leakage during training, the text features are extracted without image prefixes. Additionally, we detach the gradient of the word embedding to avoid training collapse.

Moreover, for adapting the learned prototype, we replace the Softmax function in DINO, which assumes a uniform distribution, with iterative Sinkhorn-Knopp algorithm [23] that explores the prior distribution of word embeddings [19] to obtain soft normalized probabilities of Ct:

Ct ϵ

)Diag(v), (9)

pt = Diag(uW)exp(

where uW ∈ RK is the prior marginal distribution of words, and v ∈ RB are renormalization vectors. Thus, the alignment loss can be formally expressed as:

Lalign = −pt · log pv, (10) where, pv = Softmax(Cv). Additionally, we stop the gradient propagation for LLM, ensuring that Lalign updates only the parameters of VFM.

- 2.4 Training recipe Our continual pre-training is divided into three stages:

- Stage-I: Vision-language adapter warming up. We freeze the VFM and the LLM, and only train the adapter at fixed low image resolution without RoPE-2D.
- Stage-II: Native resolution adaptation. We train the entire model using RoPE-2D for a period at a fixed high image resolution, followed by a subsequent period at native resolution.
- Stage-III: Instruction tuning (optional). We fine-tune the entire model on the instruction dataset at native resolution with RoPE-2D, to accommodate different types of data inputs. The training objective can be formally expressed as:

L =

Ldec + α · Lalign in Stage-I & II, Ldec in Stage-III,

where α is loss weight to balance Ldec and Lalign.

(11)

### 3 Empirical results

Experiments are conducted on different mainstream VFMs to verify the applicability of our method across models with different pre-training objectives. We evaluated the performance of our COMPMM on multimodal benchmarks with other LMMs. Additionally, we conducted detailed comparisons with other VFMs across various visual downstream tasks, including multimodal understanding, image classification, and semantic segmentation.

- Table 1: Main results of COMP-MM on multimodal understanding benchmarks. #PT indicates the size of pre-training dataset. #IT indicates the size of intrcution tuning dataset. N/A indicates the size is unknown. † denotes we report the performance on validation sets. ⋆ denotes we replace original SigLIP with our COMP-SigLIP for initialization.

Text-rich and Fine-grained General and Real-world CQA DVQA† AI2D INST GQA MMMU† MMB† RWQA

Model #PT #IT

∼1B models Deepseek-VL [41] 3.75M N/A - - 51.5 - - 32.2 - LLaVA-OV (SI) [18] 4.6M 3.2M 61.0 75.0 54.2 44.2 - 31.2 43.8 53.7 LLaVA-OV [18] 4.6M 4.8M 61.4 73.7 57.1 47.8 - 31.4 52.1 55.6 COMP-MM-DINOv2 4.6M 3.2M 66.1 71.0 60.0 48.7 59.3 30.4 44.7 57.3 COMP-MM-SigLIP 4.6M 3.2M 66.7 75.9 61.9 50.1 60.4 33.0 56.4 58.3 COMP-MM-AIMv2 4.6M 3.2M 68.3 75.8 64.9 50.3 61.6 35.7 61.6 58.7

∼7B models LLaVA-1.5 [42] 558K 665K 18.2 28.1 - 32.1 62.0 35.3 - LLaVA-NeXT [17] 558K 765K 54.8 74.4 - 42.4 64.2 35.1 - Deepseek-VL [41] 3.75M N/A - - 65.3 - - 36.6 - Cambrian-1 [20] 1.2M 7.0M 73.3 77.8 73.0 - 64.6 42.7 - 64.2 LLaVA-OV (SI) [18] 4.6M 3.2M 78.8 89.3 81.6 61.8 - 47.3 81.7 65.5 LLaVA-OV [18] 4.6M 4.8M 80.0 90.2 81.4 71.7 - 48.8 80.8 66.3 COMP-MM-SigLIP 4.6M 3.2M 79.6 91.0 81.4 65.0 65.9 48.9 81.4 66.4 COMP-MM-SigLIP⋆ 4.6M 3.2M 81.8 92.3 81.7 68.2 65.8 47.8 81.4 66.2

#### 3.1 COMP-MM

Setup. For 1B models, we utilize DINOv2-L [2], SigLIP-So400M [6], AIMv2-L-336px [8] as pre-trained VFMs and Qwen2.5-0.5B [43] as pre-trained LLM; for 7B models, we utilize original SigLIP-So400M and COMP-SigLIP-So400M obtained by 1B model as VFM, and Qwen2.5-7B [43] as LLM. The cross-modality adapter is a 2x2 downsampling MLP. For Stage-I, we train the adapter on LLaVA-Pretrain data [16] at pre-trained visual resolution. For Stage-II, we train the full model on

LLaVA-Mid-Stage data [18] at 1024px and native resolution. To support high resolution inputs, we replace 1M data in CC3M [44] with Densefusion-1M [45]. For Stage-III, we train the full model on LLaVA-OV-SI SFT data [18] at native resolution. All experiments are conducted on 8 × H100.

Results. As shown in Tab. 1, under the slimilar pre-training data size, our model significantly outperforms all other methods and achieves state-of-the-art performance among open-data-source models across multiple benchmarks and on both 1B models and 7B models. Specifically, COMP outperforms LLaVA-OV-SI [18], a strong baseline that employs the AnyRes technique for highresolution input, not only on text-rich and fine-grained understanding tasks such as ChartQA [10], DocVQA [11], AI2D [46], and Inst-IT [12], but also on various general and real-world multimodal understanding tasks like GQA [47], MMMU [48], MMBench [49] and RealWorldQA [50].

- Table 2: Evaluation on multimodal understanding benchmarks. We conduct extenseive experiments on COMP-DINOv2-Large, COMP-SigLIP-So400M and COMP-AIMv2-Large with LLaMA3.0 8B [51], freezing the vision encoder and directly tuning on LLaVA SFT data [16] for one epoch. #Patch indicates the number of input visual patches for the LLM. † denotes we report the performance on validation sets. We also report the performance of the SigLIP 2 [7] NaFlex variant.

Model ViT #PatchOKVQA†TextVQA† DocVQAInfoVQAChartQASEEDMME DINOv2 [2] L/14 576 54.1 13.4 7.3 21.3 10.8 57.0 1345 DINOv2 [2] G/14 3034 56.9 15.1 8.2 19.7 12.0 68.9 1423 COMP-DINOv2 L/14 576 59.0 53.6 24.7 22.8 23.8 72.8 1484 CLIP [5] L/14 576 60.0 47.5 25.6 21.8 19.2 70.1 1481 SigLIP [6] L/14 576 59.3 44.1 16.9 20.7 14.4 66.8 1416 SigLIP [6] So/14 729 60.1 47.5 19.2 21.0 14.7 67.5 1433 SigLIP 2 (NaFlex) [7]So/16 576 60.6 59.9 28.9 25.0 18.4 73.1 1536 COMP-SigLIP So/14 576 61.0 62.5 34.0 26.0 25.0 74.3 1543 AIMv2 (336px) [8] L/14 576 60.8 53.6 26.6 22.8 19.2 71.8 1472 AIMv2 (336px) [8] H/14 576 61.3 55.5 27.8 23.1 19.9 72.1 1545 COMP-AIMv2 L/14 576 60.9 60.7 33.4 24.9 23.6 73.2 1520

#### 3.2 Multimodal understanding

Setup. To further quantify the performance of COMP for LMMs, we compare it with other mainstream VFMs following the settings and hyperparameters in [8]. Specifically, we reinitialize an adapter between COMP vision encoder and LLM, e.g. LLaMA 3.0 8B [51], and freeze the parameters of vision encoder all the time. We train the adapter and the LLM jointly in a single stage on LLaVA SFT data [16] for one epoch, and scale up the learning rate of adapter by a factor of 8. To ensure fairness, we used the checkpoint before instructing tuning (Stage III) to confirm that the model had not been exposed to instruction tuning data and fixed the number of patches input to the LLM at 576.

Results. We evaluate COMP across various benchmarks covering general knowledge (OKVQA [52], SEED-Bench [53], MME [54]) and text-rich (TextVQA [55], DocVQA [11], InfoVQA [56], ChartVQA [10]) tasks. As presented in Tab. 2, our models outperform DINOv2, SigLIP and AIMv2 by a significant margin. Notably, our COMP-AIMv2-L outperforms AIMv2-H on most tasks, and our COMP-DINOv2-L also surpasses DINOv2-G, demonstrating the effectiveness of our COMP.

#### 3.3 Image recognition

Setup. We evaluate the global-view semantic quality of COMP by image classification on ImageNet1K [57]. In detail, we utilize attentive pooling probing, i.e. adding an attention pooling layer on top of the frozen features, to train our method at fixed 224px and 448px. To further analyze the recognition performance of our model, we evaluate a SigLIP variant from the checkpoint of LLaVA-OV-SI-0.5B model, denoting it as LLaVA-SigLIP. All our vision models are from Stage-II of COMP-MM-1B.

Results. As shown in 3a, our model preserves the rich global features of the original VFMs while supporting native resolution inputs. Notably, our COMP-SigLIP also outperforms LLaVA-SigLIP

#### Table 3: Results on visual downstream tasks.

(a) Evaluation on frozen trunk classification. All experiments are conducted on ImageNet-1K [57] at 224px and 448px by utilizing attentive pooling probing.

Model ViT 224px 448px MAE [58] H/14 78.5 DINOv2 [2] L/14 86.3 87.6 COMP-DINOv2 L/14 85.7 86.5

CLIP [5] L/14 84.4 83.8 SigLIP [6] So/14 87.1 88.2 LLaVA-SigLIP [18] So/14 83.2 84.4 COMP-SigLIP So/14 86.5 87.4

AIMv2 (224px) [8] L/14 86.6 84.8 AIMv2 (448px) [8] L/14 78.9 87.9 COMP-AIMv2 L/14 86.1 87.3

(b) Evaluation on semantic segmentation. All experiments are conducted on ADE20K [59] at 504px and 672px by freezing the backbone and only train the UperNet [60] head.

Model ViT 504px 672px

DINOv2 [2] L/14 55.3 55.9 COMP-DINOv2 L/14 52.7 53.0

SigLIP [6] So/14 35.2 31.6 SigLIP 2 (NaFlex) [7] So/16 35.3 34.8 LLaVA-SigLIP [18] So/14 39.9 36.5 COMP-SigLIP So/14 49.5 49.1

AIMv2 (336px) [8] L/14 51.5 50.2 COMP-AIMv2 L/14 51.0 51.8

from LLaVA-OV-SI-0.5B checkpoint by a large margin, indicating our approach can better preserve the classification ability after the continual pre-training.

#### 3.4 Semantic segmentation

Setup. We evaluate the local-view semantic quality of COMP by semantic segmentation on ADE20K [59], utilizing UperNet [60] by head tuning. Specifically, we freeze the backbone and only train the UperNet head at fixed 504px and 672px. To further analyze the segmentation performance of our model, we evaluate SigLIP 2 NaFlex variant [7], which also support native resolution inputs, and LLaVA-SigLIP which is from the checkpoint of LLaVA-OV-SI-0.5B model. All our vision models are from Stage-II of COMP-MM-1B.

Results. As shown in Tab. 3b, for vision-only pre-training DINOv2, COMP preserves the segmentation capability; for vision-language pre-training SigLIP, COMP significantly enhances its pixel-level understanding ability; for AIMv2 which has visual reconstruction process, COMP-AIMv2 brings the ability of resolution extrapolation, and outperforms AIMv2 on 672px by a large margin.

### 4 Ablation study

In this section, we investigate various design choices of the training recipe and components of the model in detail. We select AI2D [46], ChartQA [10], and DocVQA [11] as the main evaluation sets for ablation studies, as these widely adopted, text-rich and fine-grained benchmarks provide a direct assessment of the vision model while minimizing interference from the prior knowledge of the language model [20]. We put more results and full set evaluation in the Appendix A.

Improved training recipe. Tab. 4 presents the roadmap from the commonly used LLaVA-1.5-like architecture to our proposed COMP. We begin our experiments with a data combination strategy basically following LLaVA-OV [18], while replacing the instruction-tuning data with 0.8M LLaVANeXT-SFT data [17]. We adopt SigLIP-So400M [6] as the vision encoder and Qwen2.5-0.5B [43] as the language decoder. Additionally, we progressively increase the input resolution across stages, using 384px, 576px, and 768px, respectively, by interpolation to the fixed-size position embeddings.

As shown in line #1 of Tab. 4, the baseline model struggles with text-rich and fine-grained tasks. To support native resolution, we incorporate ROPE-2D while retaining learned fixed-size position embeddings. However, introducing ROPE-2D in Stage-I destabilizes training, as this interferes with the adapter warming up. Unlike prior work [29], we find it essential to unfreeze the language model during pre-training (see line #4) to enable effective multimodal alignment. Building on stable high-resolution training, we further introduce native-resolution pre-training in Stage II and III, which consistently improves performance across benchmarks. Notably, replacing SigLIP with DINOv2

Table 4: Ablation on training recipe. The first row presents the baseline three-stage training recipe, with progressive resolutions of 384px, 576px, and 768px. In the first stage, only the adapter is unfrozen, while in the subsequent stages, the full model is fine-tuned. Each following row modifies strategy from the last non-italicized row, and our final recipe is highlighted in blue . Experiments are run using SigLIP-So400M and DINOv2-Large with Qwen2.5-0.5B. †: Unfreezing vision encoder for RoPE training; ∆: The average of the difference from previous row.

# AI2D ChartQA DocVQA ∆

- 1 Baseline Recipe 50.2 29.9 24.1 -

- 2 + RoPE-2D from Stage-I† 51.5 11.0 11.4 ↓ 10.1

- 3 + RoPE-2D from Stage-II 55.4 56.9 61.6 ↑ 23.2

- 4 + Freezing LM in Stage-II 54.6 55.4 55.3 ↓ 2.87

- 5 + Native resolution training 56.7 59.5 67.7 ↑ 3.33
- 6 + Increase resolution to 1024px in Stage-II 56.2 59.9 68.8 ↑ 0.33
- 7 + Scale up training data 62.0 65.2 75.0 ↑ 5.77

- 8 + Alignment Loss 61.9 66.7 75.9 ↑ 0.77

- 9 Replace with DINOv2-Large in #7 58.9 61.5 68.5 -

- 10 + Alignment Loss 60.0 66.1 71.0 ↑ 2.73

#### Table 5: Ablation on technique designs.

(a) Ablation on C-ROPE.

###### (b) Ablation on Alignment Loss.

# Res. AI2DChartQADocVQA

Learned PE 384 48.8 22.8 24.3 Learned PE 768 47.2 28.8 29.9 RoPE-2D 768 47.5 8.24 11.9 C-ROPE 768 48.0 32.2 33.2

Scale up training data to 8M

RoPE-2D Native 57.8 29.5 34.3 C-ROPE Native 61.9 66.7 75.9

# AI2D ChartQA DocVQA Contrastive Loss 51.1 45.1 39.7 Alignment Loss 52.3 46.4 43.4 w/o uW in Eq. (9) 51.0 44.0 35.4 w/o word embedding 50.8 44.3 35.9 w/o Sinkhorn-Knopp 50.7 45.6 37.3 w/o Alignment Loss 51.0 45.0 36.9

(line #9) yields slightly lower performance, which can be recovered by incorporating Alignment Loss—highlighting the importance in models lacking explicit vision-language pre-training.

Effectiveness of C-ROPE. We utilize Qwen2-0.5B [61] and SigLIP-So400M [6] to directly be fine-tuned on LLaVA-NeXT-SFT data [17] for one epoch in three different settings: (1) Only learned position embedding with interpolation at pre-trained resolution 384px and higher resolution 768px, (2) only RoPE-2D and directly removing original position embedding [29] at 768px, and (3) CROPE at 768px. As shown in Tab. 5a, although the learned position embedding can obtain a certain capability to process high-resolution inputs through interpolation, C-ROPE can further unleash the performance by a large margin. Moreover, the performance degradation is observed when directly using RoPE-2D like previous methods, e.g. Qwen2VL [29]. To further investigate the failure of RoPE-2D without learned positional embedding, we scale the training data to 8M and replace LLM with Qwen2.5-0.5B [43], yet the model still exhibits significant performance degradation, suggesting that it is neither a data-efficient nor a training-friendly approach compared to C-ROPE.

Design choices of Alignment Loss. We train Qwen2.5-0.5B [43] and SigLIP-So400M [6] with LLaVA-NeXT [17] dataset combination to evaluate different auxiliary losses and components in Alignment Loss. As shown in Tab. 5b, contrastive loss typically demands large batch sizes for effective optimization (e.g., 65K in CoCa [39], 164K in InternVL [38], and for comparison, our mini-batch size is 32), resulting in substantial computational overhead. In addition, we investigate three variants of our Alignment Loss: (1) Without the prior distribution of word embeddings; (2) Replacing the word embedding prototypes with learnable prototypes following DINO [1]; (3) Replacing the iterative Sinkhorn-Knopp algorithm with simple softmax function. The results indicate that the Sinkhorn-Knopp algorithm, when combined with word embedding prototypes and their prior distribution, effectively aligns visual features with the LLM space.

### 5 Related work

Vision foundation models. Large-scale vision pre-training has achieved remarkable breakthroughs [3, 4, 5, 6, 7, 8], particularly with vision transformers [9] as the backbone, forming the foundation of visual understanding. These pre-training approaches can be broadly categorized into two main directions: vision-only pre-training and vision-language pre-training. In vision-only pre-training, models are trained by either distinguishing image-or patch-level entities from different views using contrastive learning [1, 2, 3, 4] or reconstructing masked patterns back to the raw image [58, 62, 63]. In vision-language pre-training [5, 6, 7, 8], models are encouraged to align visual and linguistic features into a joint semantic space by leveraging web-scale image-text pairs as the pre-training corpus. More recently, models like AIMv2 [8] utilize the multimodal framework to pre-train vision encoders. However, its native variants struggle to generalize to real-world scenarios with diverse input resolutions and lack mechanisms for cross-modal alignment.

Large language models. Leveraging the strong representational power of Transformers [64], Large Language Models (LLMs) can be pre-trained on large-scale unlabeled text corpora. Specifically, BERT [65] adopts an encoder-decoder architecture and introduces a masked language modeling paradigm, where parts of a sentence are randomly masked and then predicted by the model. This approach has proven effective for representation learning and has demonstrated strong performance on downstream tasks after fine-tuning. Subsequent works [66, 67, 68] further improve the performance through multi-task training and data scaling. Meanwhile, GPT series [32, 33, 34] utilize decoder-only Transformers and optimize under the next-token prediction paradigm with a causal mask, enabling emergent text generation capabilities. Building on this foundation, InstructGPT [69] and ChatGPT enhance instruction-following capabilities, making LLMs more suitable for real-world applications. In response to ChatGPT, recent open-source projects such as LLaMA [25, 51], Qwen [70, 61, 43], and DeepSeek [71], have attracted significant attention, driving further advancements in the community. In this paper, we leverage pre-trained LLMs as an interface to process various forms of text supervision by captioning the given image, enhancing vision backbones for diverse scenarios.

Multimodal pre-training CLIP [5] and its follow-ups [72, 73, 74, 75] demonstrate the effectiveness of aligning vision and language modalities into a unified semantic feature space using paired imagetext supervision, showcasing promising capabilities in open-set image-level classification and retrieval. However, this vision-language approach faces challenges when dealing with fine-grained tasks, e.g. segmentation and detailed caption [76, 77], due to its holistic representation and resolution. More recently, models like Flamingo [37], CoCa [39], and BLIP [78] introduced a cross-attention mechanism to handle image-grounded captions via image-to-text cross-attention, enabling pre-trained models to generate captions for visual inputs. Additionally, with the rapid advancements in large language models (LLMs), recent works leverage pre-trained LLMs as interfaces alongside pre-trained vision encoders to build powerful large multimodal models capable of addressing complex visual question-answering tasks. BLIP-2 [79] and InstructBLIP [36] utilize the previous cross-attention to deal with visual tokens, while LLaVA [16, 42] series project visual features as a sequence of visual tokens and feed them as inputs of LLM. The follow-up works [17, 80, 81, 82, 38, 83, 29, 84] focus on improving multimodal understanding through image tiling, data scaling, and improved vision token processing. In contrast to these approaches, we focus on improving the foundational visual capabilities through a text-supervised generative pre-training paradigm. Furthermore, we enhance native resolution capabilities by incorporating RoPE-2D into visual encoding and introducing an additional training objective to better align vision-language features. As a result, our method demonstrates strong performance in multimodal understanding while also improving traditional vision tasks like image classification and semantic segmentation.

### 6 Conclusion

We introduced COMP, a continual multimodal pre-training pipeline to tackle fixed resolution and modality gap problems under LMM framework, for both vision-language pre-training and vision-only pre-training models. Specifically, we proposed C-ROPE and Alignment Loss, which efficiently adapt to native resolution inputs with light continual training data and are better suited for the LLM text space. Through extensive experiments, we demonstrated that our approach achieved state-of-the-art performance on multimodal understanding benchmarks, and the performance of the VFMs in other downstream visual tasks is preserved.

### References

- [1] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In CVPR, 2021. 1, 4, 8, 9
- [2] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. TMLR, 2024. 1, 2, 4, 5, 6, 7, 9, 15, 17, 18
- [3] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 1, 9
- [4] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 1, 9
- [5] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 6, 7, 9, 15
- [6] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 1, 2, 5, 6, 7, 8, 9, 15, 17, 18
- [7] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 1, 2, 6, 7, 9
- [8] Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, and Alaaeldin El-Nouby. Multimodal autoregressive pre-training of large vision encoders. arXiv preprint arXiv:2411.14402, 2024. 1, 2, 5, 6, 7, 9, 15, 17, 18
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR,

2021. 1, 9

- [10] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL Findings, 2022. 2, 6, 7, 15
- [11] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021. 2, 6, 7, 15
- [12] Wujian Peng, Lingchen Meng, Yitong Chen, Yiweng Xie, Yang Liu, Tao Gui, Hang Xu, Xipeng Qiu, Zuxuan Wu, and Yu-Gang Jiang. Inst-it: Boosting multimodal instance understanding via explicit visual prompt instruction tuning. arXiv preprint arXiv:2412.03565, 2024. 2, 6
- [13] Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, and Filip Pavetic. Flexivit: One model for all patch sizes. In CVPR, 2023. 2, 3
- [14] Rui Tian, Zuxuan Wu, Qi Dai, Han Hu, Yu Qiao, and Yu-Gang Jiang. Resformer: Scaling vits with multi-resolution training. In CVPR, 2023. 2, 3
- [15] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In NeurIPS, 2022. 2
- [16] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 4, 5, 6, 9
- [17] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2, 5, 7, 8, 9, 15
- [18] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. TMLR, 2024. 2, 5, 6, 7, 16, 17
- [19] Jungin Park, Jiyoung Lee, and Kwanghoon Sohn. Bridging vision and language spaces with assignment prediction. In ICLR, 2024. 2, 4
- [20] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 2, 5, 7

- [21] Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, and Yu Liu. Mova: Adapting mixture of vision experts to multimodal context. arXiv preprint arXiv:2404.13046,

2024. 2

- [22] Jianlin Su. Transformer upgrade path: 17. insights into multimodal positional encoding, 2024. 2, 3, 4
- [23] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In NeurIPS, 2013. 3, 4
- [24] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024. 3
- [25] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3, 9
- [26] Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. Effective long-context scaling of foundation models. In NAACL, 2024. 3
- [27] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. In NeurIPS, 2023. 3
- [28] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, Albert Q. Jiang, Kartik Khandelwal, Timothée Lacroix, Guillaume Lample, Diego Las Casas, Thibaut Lavril, Teven Le Scao, Andy Lo, William Marshall, Louis Martin, Arthur Mensch, Pavankumar Muddireddy, Valera Nemychnikova, Marie Pellat, Patrick Von Platen, Nikhil Raghuraman, Baptiste Rozière, Alexandre Sablayrolles, Lucile Saulnier, Romain Sauvestre, Wendy Shang, Roman Soletskyi, Lawrence Stewart, Pierre Stock, Joachim Studnia, Sandeep Subramanian, Sagar Vaze, Thomas Wang, and Sophia Yang. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024. 3
- [29] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 3, 7, 8, 9
- [30] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 3
- [31] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In ECCV, 2024. 3
- [32] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 4, 9
- [33] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019. 4, 9
- [34] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 4, 9
- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023. 4
- [36] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. NeurIPS, 2024. 4, 9
- [37] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022. 4, 9
- [38] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2025. 4, 8, 9

- [39] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. In Trans. Mach. Learn. Res., 2022. 4, 8, 9
- [40] Chenyu Yang, Xizhou Zhu, Jinguo Zhu, Weijie Su, Junjie Wang, Xuan Dong, Wenhai Wang, Bin Li, Jie Zhou, Yu Qiao, et al. Vision model pre-training on interleaved image-text data via latent compression learning. In NeurIPS, 2024. 4
- [41] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseekvl: Towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024. 5
- [42] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. ArXiv, abs/2310.03744, 2023. 5, 9
- [43] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2025. 5, 7, 8, 9, 16
- [44] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018. 6
- [45] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Ling-Yu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. In NeurIPS, 2024. 6
- [46] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016. 6, 7
- [47] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 6
- [48] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 6
- [49] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024. 6
- [50] xAI Team. Grok-1.5 vision preview, 2024. 6
- [51] Avijit Dubey, Aaron Grattafiori, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 6, 9, 17
- [52] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In ECCV, 2022. 6
- [53] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 6
- [54] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 6
- [55] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019. 6
- [56] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, 2022. 6
- [57] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 6, 7, 17
- [58] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 7, 9
- [59] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017. 7, 18
- [60] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018. 7
- [61] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He,

- Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 8, 9, 15, 16
- [62] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2021. 9
- [63] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. NeurIPS, 2022. 9
- [64] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 9
- [65] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 9
- [66] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 2020. 9
- [67] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019. 9
- [68] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942,

2019. 9

- [69] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022. 9
- [70] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, et al. Qwen Technical Report. arXiv preprint arXiv:2309.12345, 2023. 9
- [71] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. Zhang, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437,

2024. 9

- [72] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. In ICML, 2021. 9
- [73] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-CLIP: Unlocking the Long-Text Capability of CLIP. arXiv preprint arXiv:2403.15378, 2024. 9
- [74] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022. 9
- [75] Wujian Peng, Sicheng Xie, Zuyao You, Shiyi Lan, and Zuxuan Wu. Synthesize diagnose and optimize: Towards fine-grained vision-language understanding. In CVPR, 2024. 9
- [76] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Xiangtai Li, Wentao Liu, and Chen Change Loy. Clipself: Vision transformer distills itself for open-vocabulary dense prediction. In ICLR, 2024. 9
- [77] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 9
- [78] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022. 9
- [79] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023. 9
- [80] Lingchen Meng, Jianwei Yang, Rui Tian, Xiyang Dai, Zuxuan Wu, Jianfeng Gao, and Yu-Gang Jiang. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms. In NeurIPS, 2024. 9

- [81] Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935, 2024. 9
- [82] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. arXiv preprint arXiv:2312.07533,

2023. 9

- [83] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 9
- [84] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv preprint arXiv:2311.07574, 2023. 9
- [85] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 17

### A More results

- Table 6: Ablation on the effectiveness of Alignment loss. We utilize Qwen2-0.5B [61] and DINOv2Large [2], and employ LLaVA-NeXT-SFT [17] as training data of Stage-III.

Recipe AI2D ChartQA DocVQA

No Alignment Loss 52.8 55.6 60.9 Only fixed res. 54.0 59.6 68.4 Both fixed & native res. 54.1 60.2 67.5

Effectiveness of Alignment Loss. To further evaluate the effectiveness of Alignment Loss, we utilize Qwen2.5-0.5B [61] and DINOv2-Large [2], and replace the training data of Stage-III with LLaVANeXT-SFT data [17] for rapid evaluation in three different settings: (1) without Alignment Loss, (2) with Alignment Loss when the inputs are fixed resolution in Stage-I, II and (3) with Alignment Loss during entire Stage-I, II, including native resolution training. As shown in Tab. 6, Alignment Loss is particularly beneficial for text-rich tasks, and generally, the longer it is applied, the better the performance. Therefore, we use Alignment Loss in both entire Stage I and Stage II by default.

- Table 7: Ablation on vision foundation models. We ablate the vision foundation models of different model sizes and pretraining tasks.

Model AI2D ChartQA DocVQA MMMU

SigLIP-Base [6] 60.2 61.6 71.0 33.6 SigLIP-So400M [6] 61.9 66.7 75.9 33.0 DINOv2-Large [2] 60.0 66.1 71.0 30.4 CLIP-Large [5] 61.6 67.3 78.6 33.1 AIMv2-Large [8] 64.9 68.3 75.8 35.7

Comparisons with different VFMs. As shown in Tab. 7, we explore different VFMs with SigLIPBase, SigLIP-So400M, DINOv2-Large, CLIP-Large-336px and AIMv2-Large-336px. The results demonstrate that our method is applicable not only to different pre-training objectives but also to various model sizes.

DocVQA ChartQA

Max Visual Tokens to LLM

Figure 3: Varying the image resolution during inference. We investigate the impact of image resolution on DocVQA [11] and ChartQA [10] by our COMP-MM-1B.

Varying the resolution during inference. To investigate the impact of image resolution on resolutionsensitive tasks, we observe the changes in scores on DocVQA [11] and ChartQA [10] by limiting the max number of visual tokens to LLM. As shown in Fig. 3, as the input resolution increases, the performance of both tasks gradually improves, which demonstrates the effectiveness of our approach for varying resolutions, particularly the high resolution.

- Table 8: Alignment Loss for visual downstream task. We train SigLIP-So400M with and without Alignment Loss and test on classification and segmentation.

Classification Segmentation 224px 448px 504px 672px

COMP-SigLIP 86.5 87.4 49.5 49.1 w/o Alignment Loss 86.0 87.0 49.2 48.9

- Table 9: Ablation on different language decoders. We utilize Qwen2-0.5B [61] and Qwen2.5-0.5B as different language decoders. L.D. denotes language decoder.

Method L.D. ChartQA DocVQA

LLaVA-OV (SI) [18] Qwen2 [61] 61.0 75.0 LLaVA-OV [18] Qwen2 [61] 61.4 73.7 COMP-MM Qwen2 [61] 63.9 76.7 COMP-MM Qwen2.5 [43] 65.2 75.0

Alignment Loss for visual downstream task. As shown in Tab. 7, we explore the effectiveness of Alignment Loss on visual downstream task like classification and segmentation. As shown in Tab. 8, it consistently improves performance across tasks.

Different Language Decoders. We use different LLMs, Qwen2-0.5B [61] and Qwen2.5-0.5B [43], as the language decoder to verify the generalizability of our method. As shown in Tab. 9, on text-rich tasks, Qwen2 and Qwen2.5 show no significant differences. Moreover, compared to LLaVA-OV [18], COMP-MM-Qwen2 still demonstrates strong performance. Notably, Alignment Loss is not utilized in these experiments.

- Table 10: Full set results of Tab. 4. Our final recipe is highlighted in blue . Experiments are run using SigLIP-So400M with Qwen2.5-0.5B.

# AI2D ChartQA DocVQA Inst-IT GQA MMMU MMBench RealWorldQA

- 1 50.2 29.9 24.1 41.3 56.6 31.6 46.5 49.2

- 2 51.5 11.0 11.4 37.0 39.0 31.4 19.8 42.0

- 3 55.4 56.9 61.6 44.7 60.3 32.0 53.2 53.9

- 4 54.6 55.4 55.3 43.5 59.6 32.9 51.9 53.2

- 5 56.7 59.5 67.7 44.2 60.2 34.3 51.2 57.8

- 7 62.0 65.2 75.0 49.8 60.4 32.6 55.8 56.6

- 8 61.9 66.7 75.9 50.1 60.4 33.0 56.4 58.3

Full set evaluation. We report the full set results of Tab. 4 in Tab. 10. For clarity and consistency, we retain the original numbering.

### B Limitations and Future Work

In this work, we do not systematically investigate the scaling laws of models and data. Our main experiments are conducted on 1B and 7B models, with the largest vision foundation model (VFM) being only 400M parameters and the training set limited to 8M samples. While the results are promising, it remains an open question whether the observed trends hold at scale. We leave these as promising directions to explore in the future.

### C Hyperparameters

COMP-MM. We outline the optimization hyperparameters used during COMP-MM continual pretraining in Tab. 11, and the hyperparameters not listed remain consistent with LLaVA-OneVision [18].

Table 11: Detailed configuration for each training stage of our COMP-MM models.

Stage-II

Stage-I

Stage-III Fixed Native

Trainable Adapter Full Model Full Model Full Model Batch Size 32×8 32×8 32×8 16×8

LRAdapter 1 × 10−3 5 × 10−3 5 × 10−3 1 × 10−5 LRV FM - 1 × 10−4 1 × 10−4 2 × 10−5 LRLLM - 2 × 10−5 2 × 10−5 1 × 10−5 Epoch 1 1 1 1

Besides, uW(k) in Eq. (9) is Nk

i Ni , where Nk is the number of the k-th word in Stage-II dataset, α in Eq. (11) is set to 0.05, and all temperature coefficients are 0.005.

Multimodal Understanding. The hyperaparmeters used for the instruction tuning are detailed in Tab. 12. We tune COMP-SigLIP and COMP-DINOv2 with LLama 3.0 8B [51] on LLaVA SFT data [85] for one epoch. In addition, we used a 2 × 2 downsampling adapter to unleash the highresolution perception capability of COMP-SigLIP and COMP-DINOv2, while keeping the number of tokenens to LLM at 576 for a fair comparison.

Table 12: Detailed configuration of COMP-SigLIP, COMP-AIMv2 and COMP-DINOv2 in instruction tuning for multimodal understanding.

Training Config

Optimizer AdamW Decoder peak learning rate 1 × 10−5 Decoder peak learning rate 1 × 10−5 Adapter peak learning rate 8 × 10−5 Minimum learning rate 0 Learning rate schedule cosine decay Batch size 128 Iterations 5197 Warmup ratio 0.05 Transformations PadToSquare, Resize

Image Recognition. The hyperaparmeters used for frozen trunk classification on ImageNet-1k [57]

- are detailed in Tab. 13. The mean and std in Normalization of COMP-SigLIP and COMP-DINOv2 are [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)] and [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)] respectively, following the original SigLIP [6] and DINOv2 [2]. We use the same hyperaparmeters for all models and baselines. For AIMv2 [8], the mean and std in Normalization are [(0.481, 0.458, 0.408), (0.269, 0.261, 0.276)], and the learning rate is 8 × 10−5 for 224px and 2 × 10−5 for 448px.

Table 13: Detailed configuration of COMP-SigLIP and COMP-DINOv2 for classification.

Training Config 224px 448px Optimizer AdamW Peak learning rate 1 × 10−4 1 × 10−5 Minimum learning rate 2 × 10−5 5 × 10−6 Learning rate schedule cosine decay Batch size 1024 256 Weight decay 0.05 Epochs 10 Warmup epochs 1 Augmentations:

RandomResizedCrop size 224px 448px scale (0.08, 1.0) ratio (0.75, 1.33) interpolation Bicubic

RandomHorizontalFlip p = 0.5 ToTensor Normalize follows SigLIP or DINOv2

Table 14: Detailed configuration of COMP-SigLIP and COMP-DINOv2 for semantic segmentation.

Training Config 504px 672px Optimizer AdamW Weight decay 0.05 Peak learning rate 4 × 10−5 Minimum learning rate 0 Learning rate schedule poly decay Batch size 16 Iterations 80K Warmup iters 1500 Augmentations:

RandomResizedCrop 504px 672px RandomFlip p = 0.5 PhotoMetricDistortion Normalize follows SigLIP or DINOv2

Semantic Segmentation. The hyperaparmeters used for semantic segmentation on ADE20K [59]

- are detailed in Tab. 14. The mean and std in Normalization of COMP-SigLIP and COMP-DINOv2 follow the original SigLIP [6] and DINOv2 [2]. The mean and std of AIMv2 [8] are [(0.481, 0.458, 0.408), (0.269, 0.261, 0.276)].

