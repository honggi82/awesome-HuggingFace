## Multimodal Autoregressive Pre-training of Large Vision Encoders

# arXiv:2411.14402v1[cs.CV]21Nov2024

Enrico Fini⋆ Mustafa Shukor⋆† Xiujun Li Philipp Dufter Michal Klein David Haldimann Sai Aitharaju Victor G. Turrisi da Costa Louis B´ethune Zhe Gan Alexander Toshev Marcin Eichner Moin Nabi Yinfei Yang Joshua Susskind Alaaeldin El-Nouby⋆

Apple

https://github.com/apple/ml-aim

### Abstract

We introduce a novel method for pre-training of large-scale vision encoders. Building on recent advancements in autoregressive pre-training of vision models, we extend this framework to a multimodal setting, i.e., images and text. In this paper, we present AIMV2, a family of generalist vision encoders characterized by a straightforward pre-training process, scalability, and remarkable performance across a range of downstream tasks. This is achieved by pairing the vision encoder with a multimodal decoder that autoregressively generates raw image patches and text tokens. Our encoders excel not only in multimodal evaluations but also in vision benchmarks such as localization, grounding, and classification. Notably, our AIMV2-3B encoder achieves 89.5% accuracy on ImageNet-1k with a frozen trunk. Furthermore, AIMV2 consistently outperforms state-of-the-art contrastive models (e.g., CLIP, SigLIP) in multimodal image understanding across diverse settings.

### 1. Introduction

Research on pre-training of vision models has evolved significantly over time. Initially, specialist models were designed to maximize performance on specific tasks [14, 36, 45, 46, 56, 114]. Gradually, general-purpose models emerged that can be deployed for a number of predefined downstream tasks with minimal adaptation [54, 87, 94, 133]. However, the remarkable success of Large Language Models (LLMs) [1, 5, 96, 116] has introduced new paradigms for utilizing vision models [3, 73, 85, 115]. Unlike the rigid predefined settings where vision models were previously employed, LLMs enable more effective exploration of the pre-trained model capabilities. This shift demands rethinking pre-training methods for vision models.

Generative pre-training is the dominant paradigm for language modeling [23, 92, 93] and has shown remark-

⋆Equal technical contribution. † Work done during an Internship.

Correspondence:{alaaeldin ali,efini}@apple.com

able performance and scalability [50, 55]. Generative pretraining has been extensively explored in computer vision [8, 29, 33, 48, 118], but its performance still lags behind that of discriminative methods [87, 94, 133, 137]. For instance, a formulation highly reminiscent of LLMs pretraining was proposed by El-Nouby et al. [33] and demonstrated encouraging scaling properties. However, it requires much higher capacity models to match the performance of its discriminative counterparts. In contrast, while contrastive techniques are often more parameter efficient, they are notably challenging to train and scale. Although significant progress has been made to mitigate these issues, there remains a gap in developing methods that combine the simplicity and scalability of generative pre-training with the parameter efficiency of discriminative approaches.

In this paper, we introduce AIMV2, a family of open vision models pre-trained to autoregressively generate both image patches and text tokens. During pre-training, AIMV2 uses a causal multimodal decoder that first regresses image patches and then decodes text tokens in an autoregressive manner, as illustrated in Figure 1. Such a simple approach offers several advantages. First, AIMV2 is straightforward to implement and train without requiring excessively large batch sizes [35, 94] or specialized interbatch communication methods [133]. Second, the architecture and pre-training objectives of AIMV2 align well with LLM-powered multimodal applications, enabling seamless integration. Finally, AIMV2 extracts a training signal from every image patch and text token, providing denser supervision compared to discriminative objectives.

Our AIMV2 models are strong generalists that exhibit remarkable performance across various vision and multimodal tasks. In particular, AIMV2 performs favorably on multimodal understanding benchmarks compared to stateof-the-art vision-language pre-trained methods [35, 133]. It outperforms DINOv2 [87] on open-vocabulary object detection and referring expression comprehension, and attains strong recognition performance with a frozen trunk, out-

Pseudo-code for AIMV2 pre-training

- 1 2 3 4

[Figure 1]

[Figure 2]

[Figure 3]

- 2 3 4

|Grayscale photo of the Eiffel Tower. [EOS]|
|---|

# img, cap: input image patches, caption # I, T: number of patches, number of text tokens # f_enc, f_dec: Transformer Encoder, Transformer Decoder

[Figure 4]

[Figure 5]

Pixel MSE Loss

Cross-entropy Loss

def forward(img, cap): # prepare targets by shifting the input pixel_target, cap_target = shift_left(img), shift_left(cap)

[Figure 6]

Causal Multimodal Decoder

# sample a prefix len and build attn mask attn_mask = build_attn_mask(prefix_len=randint(0, I-1))

[Figure 7]

|[BOS] Grayscale photo of the Eiffel Tower.|
|---|

# extract image features img_feats = f_enc(I, attn_mask)

Prefix Vision Encoder

[Figure 8]

# concatenate image features and text embedding mm_input = concat(img_feats, cap)

[Figure 9]

[Figure 10]

# decode image patches and caption mm_out = f_dec(mm_input, is_causal=True)

Patchify

[Figure 11]

[Figure 12]

# compute loss pixel_loss = normalized_mse_loss(mm_out[:I], pixel_target) cap_loss = cross_entropy_loss(mm_out[-T:], cap_target) loss = cap_loss + alpha * pixel_loss return loss

- Figure 1. AIMV2 pre-training Overview. (Left) Image patches are processed by a vision encoder trained with prefix attention [33, 95]. The resulting visual representations are concatenated with the text embeddings of their corresponding captions. This combined multimodal sequence is then processed by a joint decoder. The model is pre-trained to autoregressively reconstruct the shifted input. (Right) Pseudocode for the forward pass during AIMV2 pre-training. The pre-training process of AIMV2 is straightforward to implement, resembling that of AIM and LLMs as it relies solely on a simple autoregressive objective.

performing a number of strong baselines. Furthermore, AIMV2 enjoys strong scalability, similar to its languageonly and vision-only counterparts, improving consistently when scaling data or parameters. Moreover, we demonstrate the compatibility of AIMV2 with several modern tools, including support for native image resolution and adaptation to zero-shot recognition [132]. We discuss related works in more detail in Sec. 6.

quence, regardless of what modality it is currently processing. Our pre-training setup consists of a dedicated vision encoder that processes the raw image patches, which are then passed to a multimodal decoder alongside the embedded text tokens, as illustrated in Figure 1. The decoder subsequently performs next-token prediction on the combined sequence, following the factorization above. To support the autoregressive generation process, the vision encoder and multimodal decoder employ prefix and causal self-attention operations, respectively.

### 2. Approach

Objective function. We define separate loss functions for the image and text domains as follows:

#### 2.1. Pre-training

Our model extends the standard unimodal autoregressive framework to multimodal settings that integrate both images and text into a unified sequence. Specifically, an image x is partitioned into I non-overlapping patches xi, i ∈ [1,I], forming a sequence of tokens. Similarly, a text sequence is broken down into subwords xt, t ∈ [I,I + T]. These sequences are then concatenated, allowing text tokens to attend to image tokens. While both concatenation directions (image → text and text → image) are possible, we focus on training a strong vision encoder by always prepending the image first, thereby enabling stronger conditioning on the visual features. This results in a unified multimodal autoregressive modeling process, where the sequence is factorizatized as follows:

I

1 I

∥xˆi(x<i;θ) − xi∥22,

Limg =

i=1

I+T

1 T

Ltext = −

log P(xt|x<t;θ).

t=I+1

The overall objective is to minimize Ltext + α ∗ Limg with respect to model parameters θ. For the text domain, Ltext is a standard cross-entropy loss that measures the negative log-likelihood of the ground truth token at each step. For the image domain, Limg is an ℓ2 pixel-level regression loss, where the model’s predicted patch xˆi(θ) is compared to the true patch xi. We normalize the image patches following He et al. [48]. In practice, we use separate linear layers to map the final hidden state of the multimodal decoder to the appropriate output dimensions for image patches and vocabulary size for vision and language, respectively.

I+T

P(Sj|S<j),

P(S) =

j=1

where Sj represents the j-th token in the concatenated sequence of image patches and text tokens, and S<j includes all preceding tokens. This unified factorization allows the model to autoregressively predict the next token in the se-

#### 2.2. Architecture

For the vision encoder of AIMV2, we adopt the Vision Transformer (ViT) architecture [30]. We train a series of

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |ram|et|ers|
| | | | | | | | | | | | | |3|e8| |
| | | | | | | | | | | | | |6|e8| |
| | | | | | | | | | | | | |1|e9| |
| | | | | | | | | | | | | |3|e9| |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | |parameters|
| | | | |3e8|
| | | | | |
| | | | |6e8 1e9|
| | | | |3e9|
| | | | | |
| | | | | |
| | | | | |

ValidationCross-Entropy

ValidationCross-Entropy

2.8

2.8

2.7

2.7

2.6

2.6

1E9 2E9 3E9 4E9 5E9 6E9

5E19 15E19 5E20 15e21 45E21

Samples Seen

FLOPs (log scale)

- Figure 2. Scaling properties of AIMV2. (Left) Given a fixed pre-training data size, increasing the number of parameters always leads to an improvement in the validation loss. (Right) The optimal model size varies based on the pre-training compute budget. Larger models perform worse than smaller ones when severely undertrained but improves consistently as the compute budget increases. This behavior is consistent with that reported by Hoffmann et al. [50] for text-only autoregressive models.

model #params denc Lenc ddec Ldec AIMV2-L 0.3B 1024

AIMV2-H 0.6B 1536 AIMV2-1B 1.2B 2048 AIMV2-3B 2.7B 3072

24 12 1024

Table 1. AIMV2 family of models. We detail the architectural specifications of AIMV2 models including the embedding dimension d, number of layers L for the vision encoder and the mutlimodal decoder, and the total number of encoder parameters.

vision encoders ranging between 300M and 3B parameters. Detailed model specifications are provided in Table 1.

Prefix Attention. Following El-Nouby et al. [33], we constrain the self-attention mechanism within the vision encoder by applying a prefix attention mask [95]. This strategy facilitates the use of bidirectional attention during inference without additional tuning. Specifically, we randomly sample the prefix length as M ∼ U{1,2,...,I − 1}. The pixel loss is computed exclusively for non-prefix patches, defined as {xi | i > M }.

SwiGLU and RMSNorm. Our vision encoder and multimodal decoder incorporate SwiGLU [102] as the feedforward network (FFN) and replace all normalization layers with RMSNorm [134]. These modifications leverage the recent successes of SwiGLU and RMSNorm in language modeling [116, 117].

Multimodal Decoder. We adopt a unified multimodal decoder that performs autoregressive generation for both image and text modalities concurrently. Image features and raw text tokens are each linearly projected and embedded into Rd

dec. The decoder receives concatenated sequences of image and text features as input and employs causal attention in the self-attention operations. The outputs of the decoder are processed through two separate linear heads—one for image tokens and another for text tokens—to predict the next token in each modality respectively. We use the same decoder capacity for all the AIMV2 variants.

The optimization hyperparameters used during pre-training of all AIMV2 models are outlined in Table A1.

dataset public caption #images-text pairs sampling prob. DFN [35]

✓ alt-text 1,901,228,573 30% ✗ synthetic 3,802,457,146 30%

COYO [13] ✓ alt-text 560,171,533 9% HQITP

- ✗ alt-text 564,623,839 28%
- ✗ synthetic 431,506,953 3%

Table 2. Pre-training data mixture. AIMV2 is pre-trained using a large-scale collection of image and text pairs. For the paired captions, we utilize alt-text as well as synthetic text generated from a pre-trained captioner. In this table we list the datasets as well the sampling probabilities we used for each data source.

#### 2.3. Data

We pre-train AIMV2 models using a combination of public and private datasets containing paired images and text. We use the publicly available DFN-2B [35] and COYO [13] datasets, along with a proprietary dataset of High Quality Image-Text Pairs (HQITP). In addition to alt-text, we use synthetic captions following the approach of Lai et al. [63]. Details regarding the datasets, including their sizes and the sampling probabilities used for each dataset, are provided in Table 2. Unless mentioned otherwise, all AIMV2 models were pre-trained using 12 billion image-text samples.

#### 2.4. Post-Training

While the initial pre-training stage of AIMV2 yields highly performant models, we explore methods to further enhance the capabilities through various post-training strategies.

High-resolution Adaptation. In the initial pre-training stage, we use image data with a fixed resolution of 224px. However, many downstream task, such as detection, segmentation, and multimodal LLMs, benefit from models adapted to handle higher resolution images. Therefore, we finetune AIMV2 models for 336 and 448 pixel resolutions. The high-resolution adaptation stage utilizes 2 billion image-text pairs sampled from the same pool as the pretraining stage, except that we do not use synthetic captions at this stage. Consistent with the observations of Zhai et al. [133], we find that weight decay of zero is important for maintaining stable optimization.

- 86
- 87
- 88
- 89
- 90

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | |22|4px| |
| | | | | | | | | | | |
| | | | | | | | |33|6px| |
| | | | | | | | |44|8px| |
| | | | | | | | | | | |
| | | | | | | | | | | |

IN-1kAccuracy(%)

L H 1B 3B Model Size

- Figure 3. Scaling capacity and resolution. AIMV2 shows strong scalability with respect to model parameters, measured in frozentrunk top-1 accuracy for IN-1k. This behavior is consistent when scaling image resolution.

- 84
- 85
- 86
- 87

- 85

- 85.5
- 86

- 86.5
- 87

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | |AIMV2<br><br>Cap| |
| | | | | |

IN-1kAccuracy(%)

IN-1kAccuracy(%)

AIMV2

Cap

500M 1B 2B 4B Image-text Pairs

L H 1B Model Size

Figure 4. AIMV2 vs. Captioning. We investigate the role of the image-level objective in comparison to a captioning-only baseline, particularly as we scale data and model size. Our findings indicate that AIMV2 consistently outperforms the captioning baseline across all dataset and model sizes. Notably, AIMV2 exhibits fewer signs of saturation when scaling data compared to the captioning-only approach.

Native Resolution Fine-tuning. Training models for a dedicated resolution and aspect ratio can be inflexible for many applications that require processing images in their original shapes. Prior works such as FlexiViT [9] and NaViT [26] have tackled these limitation. We adopt a different approach for training with variable aspect ratios and resolutions. Specifically, we define Bi as the number of images in a mini-batch, Ai as the number of patches per image, and C as the total number of image patches in the mini-batch. For a mini-batch i, we randomly sample an area A and resize the images to fit within this area while maintaining their aspect ratios.1 We then adjust the mini-batch size Bi such that C = Ai × Bi. This strategy is analogous to the approach proposed by Pouransari et al. [91] for training LLMs with variable context lengths. Our implementation does not require heuristics for sequence packing, attention masking, or custom pooling operations. We choose A = 2n, where n is sampled from a truncated normal distribution N(0,1) within the range [−1,1] and linearly mapped to [7,12].

### 3. Analysis

One of the main advantages of AIMV2 is its simplicity; it is easy to implement and scale. Therefore, we investigate the scaling properties of the AIMV2 family of models.

#### 3.1. Scaling AIMv2

First, we investigate the impact of scaling data size and model capacity on the validation performance of AIMV2. We fix the model size and vary the number of samples seen during pre-training. This analysis is similar to “Approach 1” in the study of Hoffmann et al. [50]. The results of this study are illustrated in Figure 2.

Setup. We train four model capacities, ranging from 300 million to 3 billion parameters, and vary the number of

1We use zero padding if the image cannot be perfectly fitted into the desired area.

samples seen between 500 million to 6.4 billion image-text pairs. All models are trained to convergence with no early stopping. To achieve this with minimal computational cost, we train a single model for each capacity using 5 billion images with a half-cosine learning rate schedule (i.e., the final learning rate is half the peak learning rate). We select seven intermediate checkpoints from this run and apply a linear cooldown to 1 × 10−6. The length of the cooldown stage is 20% of the initial pre-training stage.

Results. We observe a consistent improvement in performance with scaling data or parameters. However, diminishing returns appear when scaling data for the lower-capacity models. Additionally, we find that the optimal model size changes as the compute budget varies. At smaller compute budgets, larger-capacity models are undertrained and underperform compared to their lower-capacity counterparts.

#### 3.2. AIMv2 vs. Captioning

We study the role of the image-level autoregressive objective in the pre-training of AIMV2. We compare the performance of models trained with the multimodal autoregressive objective to ones trained only with language supervision. The results are illustrated in Figure 4.

Setup. Unless specified otherwise, this investigation uses a ViT-H backbone and 2 billion image-text pairs for pretraining. All models are trained to convergence with a cosine learning rate schedule. We measure the IN-1k top-1 accuracy after attentive probe with a frozen trunk.

Results. First, the image-level objective of AIMV2 consistently improves performance compared to the captioningonly baseline. This is true even when changing the model capacity and the size of the pre-training data. Moreover, we see both approaches improving consistently when increasing the data size or model capacity; however, we observe signs of plateauing for the captioning baselines with scaling data which we do not observe with AIMV2.

Infographic

EuroSAT

Cifar100

Food101

iNAT-18

CAM17

Cifar10

PCAM

RxRx1

fMoW

IN-1k

DTD

Cars

Pets

model architecture

MAE [48] ViT-2B/14 82.2 70.8 97.5 87.3 93.4 81.2 95.1 94.9 94.4 90.3 7.3 98.2 60.1 50.2 AIMV1 [33]

ViT-H/14 78.5 64.0 97.2 86.8 90.1 80.1 93.0 93.0 94.3 90.0 7.8 98.4 58.3 45.2 ViT-7B/14 84.0 75.5 98.9 91.8 94.1 85.6 95.4 95.0 94.2 90.5 8.4 98.5 63.5 57.7

DINOv2 [87] ViT-g/14 87.2 83.0 99.7 95.6 96.0 86.9 96.8 94.9 95.8 90.1 9.0 98.8 65.5 59.4 OAI CLIP [94] ViT-L/14 85.7 73.5 98.7 89.7 95.4 83.5 96.2 94.5 94.4 89.2 5.7 98.0 62.0 66.9 DFN-CLIP [35]

VIT-L/14 86.5 75.5 99.2 93.2 96.2 85.8 96.3 96.4 95.0 89.8 5.8 98.3 63.1 66.8

- ViT-H/14 86.9 76.4 99.3 93.9 96.3 87.0 96.8 96.7 95.7 90.5 6.1 98.8 63.4 68.1

SigLIP [133]

ViT-L/16 86.5 75.1 98.5 90.4 96.1 86.7 96.7 96.5 93.1 89.5 4.5 98.3 61.7 71.0 ViT-So400m/14 87.3 77.4 98.8 91.2 96.5 87.7 96.7 96.6 93.3 90.0 4.6 98.6 64.4 72.3

ViT-L/14 86.6 76.0 99.1 92.2 95.7 87.9 96.3 96.3 93.7 89.3 5.6 98.4 60.7 69.0

- ViT-H/14 87.5 77.9 99.3 93.5 96.3 88.2 96.6 96.4 93.3 89.3 5.8 98.5 62.2 70.4 ViT-1B/14 88.1 79.7 99.4 94.1 96.7 88.4 96.8 96.5 94.2 89.0 6.7 98.8 63.2 71.7 ViT-3B/14 88.5 81.5 99.5 94.3 96.8 88.9 97.1 96.5 93.5 89.4 7.3 99.0 64.2 72.2

AIMV2

ViT-3B/14448px 89.5 85.9 99.5 94.5 97.4 89.0 97.4 96.7 93.4 89.9 9.5 98.9 66.1 74.8

Table 3. Frozen trunk evaluation for recognition benchmarks. We report the recognition performance of the AIMV2 family models when compared to a number of self-supervised and weakly-supervised state-of-the-art models. All models are evaluated using attentive probing with a frozen backbone. Unless otherwise specified, all AIMV2 models are trained at 224px resolution on 12B samples.

### 4. Results

AIMV2 is a generalist vision encoder that can be leveraged off-the-shelf for a wide range of downstream tasks. We evaluate the performance of the AIMV2 family across various tasks, including recognition, detection, captioning, and multiple multimodal benchmarks.

#### 4.1. Image Recognition

Attentive probing. We assess the quality of the AIMV2 models as off-the-shelf backbones for recognition benchmarks which are outlined in Table B1. To this end, we adopt the attentive probing setting proposed by Yu et al. [129], where the vision encoder remains frozen, and only an attentive probe classifier is trained on top of the last layer features. The results are presented in Table 3. Detailed hyperparameters used for the probing experiments are provided in Table A2. First, we observe that AIMV2 significantly outperforms generative unsupervised methods such as MAE [48] and AIM [33], even with much smaller capacity models. Compared to DINOv2 [87], we find that both the similarly sized AIMV2-1B and the smaller AIMV2-H provide competitive performance, outperforming DINOv2 on several benchmarks including IN-1k, Food101, DTD, Cars, and with a particularly large margin on Infographic. However, DINOv2 offers exceptional performance for iNaturalist and fMoW. Furthermore, we find the performance of self-supervised models on medical imaging benchmarks (e.g., RxRx1 and CAM17) noteworthy, as they exhibit stronger performance compared to their weakly supervised counterparts. This affirms the importance of self-supervised learning methods, particularly in low-resource domains.

Second, when compared to other vision-language pretrained baselines, AIMV2 exhibits highly competitive performance. For instance, at the ViT-Large capacity, AIMV2 outperforms OAI CLIP on the majority of benchmarks and achieves stronger performance than DFN-CLIP and SigLIP

open-vocabulary referring expression model COCO LVIS RefC RefC+ RefCg

OAI CLIP 59.1 31.0 92.2 86.2 88.3 DFN-CLIP 59.8 30.7 92.5 85.8 88.3 SigLIP 58.8 30.5 92.3 86.1 88.4 DINOv2 60.1 30.8 92.2 85.9 89.1 AIMV2 60.2 31.6 92.6 86.3 88.9

- Table 4. Evaluation after finetuning on grounding dataset mixture. We report the performance on mean average precision (AP) for open-vocabulary detection and Precision @1 for referring expression comprehension tasks.

|model → Cap AIMV2|CapPa [118] AIMV2|OAI CLIP SigLIP<br><br>|
|---|---|---|
|Pre-train/LiT 2B/3B 2B/3B IN-1k top-1 75.0 75.3|9B/3B 12B/3B 76.4 77.0<br><br>|13B/- 40B/75.5 80.4|

- Table 5. Zero-shot performance. Comparison of different models with varying amounts of pre-training and LiT pairs, and their performance on IN1k. For CapPa, we compare to the number reported by Tschannen et al. [118].

test resolution → 224×224 448×448 native

AIMV2-L224px 86.6 84.8 AIMV2-L448px 78.9 87.9 AIMV2-Lnative 86.1 87.1 87.3

- Table 6. AIMV2 with native apsect ratio and resolution. We report the IN-1k top-1 performance of the native resolution AIMV2-L model as compared to AIMV2-L models that are pretrained/finetuned for single dedicated resolution.

on several key benchmarks, including IN-1k, iNaturalist, DTD, and Infographic. These results are particularly impressive given that AIMV2 is trained using nearly a quarter of the data required for training DFN-CLIP and SigLIP (12B vs. 40B), while also being easier to train and scale. Finally, we find that scaling the capacity of AIMV2 models consistently leads to a stronger performance with AIMV23B exhibiting the strongest result, in particular its variant finetuned for 448px images which achieves 89.5% top-1 accuracy on IN-1k with a frozen trunk. Finally, in Figure 3 we observe a clear improvement to the performance of IN-1k when scaling the model capacity and the image resolution

model architecture # patches VQAv2 GQA OKVQA TextVQA DocVQA InfoVQA ChartQA ScienceQA COCO TextCaps NoCaps SEED MMEp OpenAI CLIP ViT-L/14 576 78.0 72.0 60.0 47.5 25.6 21.8 18.5 73.8 94.9 75.3 93.3 70.1 1481

ViT-L/14 576 76.9 70.3 59.3 44.1 16.9 20.7 14.4 74.7 93.0 69.9 92.7 66.8 1416 ViT-So400M/14 752 77.7 71.0 60.1 47.5 19.2 21.0 14.7 74.9 94.6 70.8 94.5 67.5 1433

SigLIP

DINOv2 VIT-g/14 3034 76.7 72.7 56.9 15.1 8.2 19.7 12.0 69.5 93.4 42.1 89.1 68.9 1423

ViT-L/14 576 79.7 72.5 60.8 53.6 26.6 22.8 19.2 74.1 96.9 81.1 99.9 71.8 1472 ViT-H/14 576 80.2 72.8 61.3 55.5 27.8 23.1 19.9 76.8 99.6 80.7 102.7 72.1 1545 ViT-1B/14 576 80.5 73.0 61.5 56.8 28.5 22.1 20.5 76.4 98.4 82.6 101.5 72.7 1508

AIMV2

ViT-3B/14 576 80.9 73.3 61.7 58.2 30.4 23.0 22.6 77.3 100.3 83.8 102.9 72.9 1545

- Table 7. Mutlimodal Evaluations. We compare AIMV2 to state-of-the-art visual backbones for multimodal instruction tuning. Under comparable capacities, AIMV2-L outperforms its counterparts on the majority of benchmarks. Additionally, scaling to the larger AIMV2-

3B model results in clear improvements, achieving the highest scores on nearly all benchmarks. All AIMV2 models use 336px resolution.

model architecture 0-shot 4-shot 8-shot

OAI CLIP [85] ViT-L/14 39.3 62.2 66.1 DFN-CLIP [85] ViT-H/14 40.9 62.5 66.4 AIMV2 ViT-L/14 39.6 63.8 67.2

- Table 8. ICL few-shot performance. We report the in-context few-shot performance averaged across a wide range of benchmarks as detailed in § 4.3.2. The results for DFN-CLIP and OAI CLIP are as reported by McKinzie et al. [85].

TextVQA

VQAv2

70

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | |AI|MV|2| |
| | | | | | | | |
| | | | |OA|I C|LIP| |
| | | | |Si|gLIP| | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | |AI|MV|2| |
| | | | |OA|I C|LIP| |
| | | | | | | | |
| | | | |Si|gLIP| | |

82

Accuracy(%)

60

80

50

78

in conjunction. We provide more detailed results for the high-resolution finetuned backbones in Appendix B.

76

40

336px 672px 1008px

336px 672px 1008px

Zero-shot via LiT Tuning. We investigate the compatibility of AIMV2 backbones with LiT [132], extending its application to zero-shot settings. We report the IN-1k zeroshot performance in Table 5. First, we observe that AIMV2, with the multimodal autoregressive objective, shows a modest improvement compared to the captioning-only baseline, even in this setting. Furthermore, an AIMV2-L model trained for a longer duration exhibits favorable performance compared to the results reported by Tschannen et al. [118] for CapPa. Overall, our model demonstrates strong zero-shot performance, outperforming OAI CLIP [94], yet still lagging behind dedicated models like SigLIP that are trained for a longer schedule with 40B image-text pairs.

Resolution

Resolution

Figure 5. Impact of Scaling Resolution. The performance boost achieved byAIMV2 persists after scaling input resolution via tiling Lin et al. [72], Liu et al. [73] compared to popular vision backbones for VLMs such as OAI CLIP and SigLIP.

MM-Grounding-DINO [74, 136] but adapt ViT-L through the ViTDet [68] formulation as the vision backbone. Our results are presented in Table 4. For OVD capabilities we evaluate on COCO [71] and LVIS [43], while for REC, we evaluate on RefCOCO (RefC) [57], RefCOCO+ (RefC+) [130], and RefCOCOg (RefCg) [79]. All models were trained on the following datasets: Objects365v1 [101], Flickr-30k Entities [90, 127], GQA [52], COCO17 [71], and RefCOCO [57, 79, 130]. During DINOv2 training we fix the window size to 16 [69] to ensure fixed compute cost across backbones. Our results indicate that AIMV2 outperforms DINOv2 as well as other vision-language pre-trained models on all benchmarks but one, showing particularly strong performance on LVIS. We present additional localization and grounding results including closed-vocabulary detection and instance segmentation as well as ablations on varying window sizes in Appendix D.

Native resolution. We finetune AIMV2 to process images with a wide range of resolutions and aspect ratios as detailed in § 2.4. In order to assess the quality of this stage of post-training, we compare the performance of an AIMV2 encoder adapted for native resolution to models that are tuned for one specific resolution in Table 6. We observe that AIMV2-Lnative provides a strong performance across a wide range of resolutions off-the-shelf, experiencing only a minor degradation in performance to the dedicated models. Additionally, evaluating our model using the original native resolutions of the IN-1k validation set images yields a robust accuracy of 87.3%, confirming that AIMV2 maintains exceptional recognition performance while offering high flexibility in both aspect ratio and resolution.

#### 4.3. Multimodal Understanding

Vision encoders play a crucial role in advancing large multimodal models [6, 73, 85, 115, 138]. To quantify the performance of AIMV2 models in this setting, we perform a multimodal instruction tuning stage similar to Liu et al. [73]. Additionally, we explore the few-shot In-Context Learning (ICL) setting after large-scale multimodal pretraining similar to McKinzie et al. [85].

#### 4.2. Object Detection and Grounding

To further demonstrate the capabilities of AIMV2, we evaluate its performance on tasks such as Open-Vocabulary Detection (OVD) and Referring Expression Comprehension (REC). We follow the model architecture introduced by

OAI CLIP SigLIP AIMV2

100

97

93

93

91 76 69 70 57 48

91

79 71

79

77 70

74

72

72

69

69

69

68

58 53

57

46

28 23 21

25 23 18

23 22 15

VQAv2 GQA OKVQA TextVQA DocVQA InfoVQA ChartQA ScienceQA COCO TextCaps NoCaps MMEp

###### (a) Vicuna 1.5 + Llava mixture

98

96 75 71 80 61 56 51

96 76 72 78 61 58 50

96

95

94

82

80

80

80

80

79

78

74 71

60 53 51

52

51

48

35

34

34

VQAv2 GQA OKVQA TextVQA DocVQA InfoVQA ChartQA ScienceQA COCO TextCaps NoCaps MMEp

(b) Llama 3.0 + Cambrian 7M

Figure 6. Instruction tuning under different settings. We evaluate instruction-tuned models across combinations of LLM decoders and tuning data mixtures. In all settings, AIMV2 consistently outperforms or matches SigLIP and OAI CLIP on most benchmarks. All models use a ViT-L backbone with 336px images. For better readability, we present normalized MMEp scores by dividing the raw scores by 2000.

##### 4.3.1. Multimodal Instruction Tuning

Setup. We place a 2-layer MLP connector between the vision encoder (e.g., AIMV2-L) and the LLM (e.g., Llama 3.0). The parameters of the vision encoder are frozen during this stage. Contrary to Liu et al. [73], we train the connector and the LLM jointly in a single stage. However, we scale up the learning rate for the connector by a factor of 8. We found this strategy to be simpler while leading to comparable results. We detail the evaluation datasets, task prompts, and hyperparameters used during this stage in Appendix C. Unless mentioned otherwise, we use the Llava SFT mixture [73] and Llama-3.0 8B LLM decoder [32]. We train all models for a single epoch.

Evaluation. We evaluate the instruction-tuned models across various question answering benchmarks covering general knowledge, text-rich images, scientific domains, and captioning. The results for AIMV2 and several baselines are presented in Table 7. Notably, our smallest model, AIMV2-L, outperforms OAI CLIP, SigLIP, and DINOv2 on most benchmarks, even when the baselines use larger capacities or higher input resolutions. Furthermore, performance consistently improves with increasing the AIMV2 backbone capacity, with the AIMV2-3B model achieving the best performance across all benchmarks except one.

Varying the LLM and Data Mixture. In addition to the canonical setting reported in Table 7, we evaluate whether AIMV2 can provide similar gains compared to popular vision encoders across various combinations of LLM decoders and instruction tuning data mixtures. Specifically, we perform the instruction tuning stage under the following settings: (1) Llama 3.0 with the Cambrian data mixture [115] and (2) Vicuna 1.5 [22] with the Llava SFT mixture. We present the results for AIMV2-L alongside similarly sized OAI CLIP and SigLIP backbones in Figure 6. Across all settings, AIMV2 provides a stronger, or at worst on par, performance compared the OAI CLIP and SigLIP. These findings further affirm the robustness and compatibility of AIMV2 within diverse multimodal pipelines.

High-Resolution via Tiling. One of the most popular strategies to enhance the performance of vision-language models is increasing the image resolution. This can be achieved through a tiling strategy [72, 73, 103], where a high-resolution image is divided into a number of equally sized crops that match the pre-training resolution of the available backbones (e.g., 224px or 336px). We investigate the compatibility of AIMV2 with this strategy. Specifically, we use a crop size of 336px and evaluate our pipeline on 672px and 1008px images corresponding to 2×2 and 3×3 grids respectively. The results for AIMV2, SigLIP, and OAI CLIP are presented in Figure 5. We observe that the performance of all methods improves with higher resolutions, with a significant improvement for TextVQA. Notably, AIMV2 maintains its advantage over the baselines in high-resolution tiling settings, demonstrating its versatility.

##### 4.3.2. Multimodal In-Context Learning

We also evaluate AIMV2 in a large-scale multimodal pre-training setting. Following the pre-training recipe as MM1 [85], we simply replace the vision encoder with AIMV2. Given that this model is pre-trained using interleaved image-text documents, it enables in-context evaluations [3]. We report the ICL performance in Table 8. Specifically, we report the average 0-shot, 4-shot, and 8-shot performance across the following benchmarks: COCO [21], NoCaps [2], TextCaps [105], VQAv2 [40], TextVQA [107], VizWiz [44], GQA [53], and OK-VQA [80]. Our results demonstrate that AIMV2 achieves the best performance in the 4-shot and 8-shot settings, surpassing the higher capacity DFN-CLIP adopted by the MM1 series. This highlights the compatibility and effectiveness of AIMV2 in leveraging ICL in a large-scale multimodal setup.

### 5. Ablation Study

In this section, we investigate various design choices and present the trade-offs associated with each. The results of our study are summarized in Table 9.

model bsz IN-1k VQAv2 TextVQA CLIP

model pre-train attn. IN-1k VQAv2 TextVQA AIMV1 prefix 72.0 65.4 12.7

α IN-1k VQAv2 TextVQA

8k 84.6 74.1 24.6

0.2 85.6 76.7 37.4 0.4 85.6 76.9 37.5 0.6 85.6 76.7 37.4

bidir 85.1 76.2 34.4 prefix 85.4 76.8 36.5

16k 85.2 74.8 26.3 CapPa 8k 84.7 75.1 30.6 AIMV2 8k 85.6 76.9 37.5

Cap

AIMV2 prefix 85.6 76.9 37.5

(b) AIMV2 vs. CLIP..

(a) Objective.

(c) Criteria Weights.

IN-1k VQAv2 TextVQA

width IN-1k VQAv2 TextVQA

depth IN-1k VQAv2 TextVQA

512 85.3 76.2 35.9 1024 85.6 76.9 37.5 1536 85.1 76.9 36.9

8 85.5 76.7 37.0 12 85.6 76.9 37.5 16 85.6 76.9 36.6

separate 85.6 77.1 37.2 joint 85.6 76.9 37.5

(d) Decoder Architecture.

(e) Decoder Width.

(f) Decoder Depth.

- Table 9. Ablations. We ablate a number of design choices for AIMV2 and how they impact performance on key recognition and multimodal benchmarks. This includes (a) the contribution of the visual and textual objectives, (b) comparison to other popular objectives, (c) the optimal balancing between the losses, (d-f) the architecture of the multimodal decoder. All models are trained at 224px resolution.

Setup. The default setting for this ablation study utilizes a ViT-Large vision encoder and 2 billion image-text pairs during pre-training. We measure the IN-1k top-1 accuracy after attentive probing, as well as the questionanswering accuracy on the validation sets of VQAv2 [40] and TextVQA [106] following instruction tuning, as described in § 2.4. All experiments reported in this ablation study employ images with 224 × 224 resolution. The metrics selected for this study provide a comprehensive view of the models’ capabilities, encompassing recognition, general question answering, and text-rich question answering.

Pre-training Objective. The pre-training objective of AIMV2 comprises a combination of image-level and textlevel autoregressive objectives. We evaluate the performance of each objective independently and assess the impact of combining them, as presented in Table 9a. First, we observe that utilizing only the image-level objective (i.e., AIMv1) results in weaker performance compared to models that incorporate the captioning objective. This is expected, given that AIMv1 operates in an unsupervised manner and demands higher-capacity models to achieve optimal performance, as highlighted by El-Nouby et al. [33]. Second, for the captioning-only model, using prefix attention within the vision encoder yields superior performance compared to fully bidirectional attention. We hypothesize that prefix attention facilitates the encoding of maximally informative contexts even from partial images, as such contexts are utilized by subsequent visual and textual tokens. However, this hypothesis warrants further investigation, which is beyond the scope of this work and is reserved for future research. Finally, we find that combining the image-level and text-level objectives in AIMV2 leads to an improved performance, particularly noticeable for TextVQA.

AIMV2 vs. CLIP vs. CapPa. In Table 9b, we evaluate the performance of models trained with the AIMV2 objective in comparison to other popular vision-language pretraining objectives, specifically CLIP [94] and CapPa [118]. All models are trained using identical architectures, incorporating SwiGLU and RMSNorm, and are pre-trained using the same dataset of image-text pairs. Notably, since CLIP pre-training benefits from larger batch sizes, we re-

port CLIP results using both 8k and 16k batch sizes to ensure a fair comparison. Our findings indicate that, under comparable settings, AIMV2 consistently outperforms both CLIP and CapPA by a significant margin, particularly on the TextVQA benchmark. This performance is especially noteworthy given the simplicity and scalability AIMV2.

Multi-task balancing. We examine whether the pretraining of AIMV2 is sensitive to the balancing between the image-level and text-level objectives in Table 9c. We vary the hyperaparmeter α, as described in § 2.1, and we observe only minor fluctuations in performance around the optimal value of 0.4 across the three benchmarks.

Joint vs. Separate Decoders. In the AIMV2 architecture, we opt for a multimodal joint decoder instead of employing dedicated decoders for each modality. In Table 9d, we examine the performance of an AIMV2 variant that utilizes two dedicated decoders. Using a single joint decoder achieves comparable results to using separate decoders while offering greater simplicity and efficiency during pre-training. This advantage proves valuable when scaling data and model capacity.

Decoder architecture. Finally, we examine the capacity of the multimodal decoder, as detailed in Table 9e and Table 9f. We find that performance is more sensitive to changes in decoder capacity when scaling the width compared to scaling the depth. Additionally, we observe that increasing the decoder capacity beyond a certain threshold, whether by scaling width or depth, leads to a decline in performance. This observation is consistent with the findings of Tschannen et al. [118] for captioning-only models.

### 6. Related Works

Autoregressive pre-training. Autoregressive modeling has been a foundational idea in machine learning and statistics for decades, long before deep learning [11]. However, it has been popularized and scaled by works such as GPT [12, 92, 93], and LLaMAs [31, 116, 117] which have demonstrated the power of autoregressive pre-training in natural language processing tasks. In vision, autoregressive principles have been applied through models like

iGPT [19], which flattens images into a sequence of discretized pixels and then treats them analogously to language tokens. Similarly, Yu et al. [128] also discretize the patches with a VQGAN model [34] and then predicts them autoregressively. AIM [33] brings back the more practical continuous approach and scales to very large vision models. However, AIM still lags behind other state of the art models in sheer performance, as it uses vision-only data and requires large model capacities to perform optimally. This paper addresses these limitations by introducing multimodal pre-training in the AIMV2 family. Concurrent works [77, 104, 113, 122, 124, 125, 131] have also investigated similar multimodal autoregressive approaches that predict text and images. However, they often focus on multimodal generation quality rather than representation quality, and therefore use discrete tokens or leverage diffusion models [98] as decoders [70, 110, 111].

Pre-training in vision. For many years, the computer vision community predominantly focused on supervised pretraining [58, 97, 108], with ImageNet [61] checkpoints serving as the backbone for most visual tasks. This eventually hit a wall in terms of scalability, as labels are expensive to acquire. The community therefore focused on self-supervised methods. Earlier models used pretext tasks such as rotation prediction and patch deshuffling [39, 86, 135]. More sophisticated models like SimCLR [20], BYOL [42], SwAV [15] and DINO [16] leverage variations of contrastive learning to train models that are quasi-invariant to a broad range of image augmentations. This turns out to learn strong and general visual representations without supervision. However they require carefully handcrafted data augmentations, which also makes them computationally expensive, especially at scale. On the other hand, MAE and BEiT [8, 48] introduced masking strategies to reconstruct input data, reducing the reliance on augmentations and increasing efficiency but sacrificing performance. In practice, the best performing self-supervised vision-only models use a mixture of augmentations and masking [4, 87, 137]. Unfortunately, they are challenging to scale as they still need multiple forward passes for each training step. AIM [33] departs from these methods by employing a reconstruction-based autoregressive framework that exhibits strong scalability but requires high capacity models to attain optimal performance. Leveraging large-scale, noisy, weakly supervised datasets from the internet [13, 35, 100], an efficient paradigm emerged that aligns vision and text features through contrastive learning [54, 94]. Nevertheless, CLIP-like models require large batch sizes and meticulous dataset filtering [35, 100]. Subsequent works, such as SigLIP [133], EVA CLIP [109], and Fini et al. [37], have addressed these issues by optimizing training processes and improving data filtering [35]. Unlike these approaches, AIMV2 does not perform explicit feature

space alignment but aligns training objectives through autoregressive modeling for better multimodal synergy.

Captioning. Image captioning has been extensively studied prior to the computer vision literature. Early works [56, 121, 126] focused on aligning visual features with text to generate descriptions using CNNs and RNNs. VirTex [28] and ICMLM [99] utilize captioning for visual pre-training. SimVLM [123] employs a PrefixLM approach, encoding images and partial text tokens with a multimodal encoder and decoding the remaining text. LEMON [51] scales the language model in both parameters and dataset size. Approaches such as [65, 66] combines generative captioning with discriminative contrastive objectives to enhance multimodal learning, which led to scaling to billion-parameter models [62, 67, 129]. Similarly, CapPa [118] trains a captioning model that functions as both a masked and causal decoder, and Caron et al. [17] re-purposes a captioning model for web-scale entity recognition. Different from most previous approaches, AIMV2 does not use cross-attention and treats vision and text tokens symmetrically, similar to large multimodal models (e.g. LLaVA [73] and MM1 [85]). Additionally, AIMV2 incorporates an autoregressive image modeling loss on vision tokens, further enhancing performance beyond captioning-only methods.

### 7. Conclusion

This paper introduce AIMV2, a family of vision encoders pre-trained with a multimodal autoregressive objective that reconstructs image patches and text tokens. This unified objective enables AIMV2 to excel in diverse tasks, including image recognition, grounding, and multimodal understanding. The superior performance of AIMV2 is attributed to its ability to leverage signals from all input tokens and patches, facilitating efficient training with fewer samples compared to other methods. AIMV2 consistently outperforms or matches existing self-supervised and vision-language pretrained models, demonstrating its strength and versatility as a vision encoder. Additionally, the straightforward pretraining process of AIMV2 ensures easy scalability, paving the way for further advancements in vision model scaling.

### Acknowledgement

We thank Shuangfei Zhai, Miguel Bautista, Jason Ramapuram, Chun-Liang Li, Seyed Mohsen Moosavi Dezfooli, David Mizrahi, Roman Bachmann, and Jesse Allardice for many fruitful discussions. We thank Vaishaal Shankar, Peter Grasch, Vasileios Saveris, and Jeff Lai for their support on data collection and synthetic captioning. We thank Denise Hui, Dan Busbridge, and Samy Bengio for infra and compute support. Finally, we thank Marco Cuturi, James Thornton, Pierre Ablin, Eugene Ndiaye and the whole MLR team at Apple for their support throughout the project.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [2] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In ICCV, 2019. 7, 16
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 1, 7
- [4] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. arXiv preprint arXiv:2301.08243, 2023. 9
- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966,

2023. 6

- [7] Peter Bandi, Oscar Geessink, Quirine Manson, Marcory Van Dijk, Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae Lee, Kyunghyun Paeng, Aoxiao Zhong, et al. From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge. IEEE Transactions on Medical Imaging, 2018. 15
- [8] Hangbo Bao, Li Dong, and Furu Wei. BEiT: Bert pretraining of image transformers. In ICLR, 2022. 1, 9
- [9] Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, and Filip Pavetic. Flexivit: One model for all patch sizes. In CVPR, 2023. 4
- [10] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In ECCV, 2014. 15
- [11] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. Time series analysis: forecasting and control. John Wiley & Sons, 2015. 8
- [12] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,

- et al. Language models are few-shot learners. preprint arXiv:2005.14165, 2020. 8
- [13] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset, 2022. 3, 9
- [14] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV,

2020. 1

- [15] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020. 9
- [16] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 9
- [17] Mathilde Caron, Ahmet Iscen, Alireza Fathi, and Cordelia Schmid. A generative approach for wikipedia-scale visual entity recognition. In CVPR, 2024. 9
- [18] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, and Dahua Lin. Mmdetection: Open mmlab detection toolbox and benchmark. 2019. 18
- [19] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 9
- [20] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 9
- [21] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 7
- [22] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 7
- [23] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arxiv 2022. arXiv preprint arXiv:2204.02311,

2022. 1

- [24] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. 15
- [25] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi. Describing textures in the wild. In CVPR, 2014. 15
- [26] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M

- Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36, 2024. 4
- [27] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, 2009. 15
- [28] Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021. 9
- [29] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015. 1
- [30] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 2
- [31] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 8

- [32] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 7

- [33] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. arXiv preprint arXiv:2401.08541, 2024. 1, 2, 3, 5, 8, 9
- [34] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 9

- [35] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 1, 3, 5, 9
- [36] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. 2019 ieee. In ICCV, 2018. 1
- [37] Enrico Fini, Pietro Astolfi, Adriana Romero-Soriano, Jakob Verbeek, and Michal Drozdzal. Improved baselines for vision-language pre-training. arXiv preprint arXiv:2305.08675, 2023. 9
- [38] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 16
- [39] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018. 9

- [40] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017. 7, 8
- [41] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2017. 16
- [42] Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. NeurIPS, 2020. 9
- [43] Agrim Gupta, Piotr Doll´ar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation, 2019. 6
- [44] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, 2018. 7
- [45] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR,

2016. 1

- [46] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In ICCV, 2017. 1
- [47] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. 2018. 18
- [48] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 1, 2, 5, 9
- [49] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification,

2017. 15

- [50] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 1, 3, 4
- [51] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In CVPR, 2022. 9
- [52] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019. 6, 16
- [53] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 7
- [54] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 1, 9

- [55] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 1

- [56] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In CVPR,

2015. 1, 9

- [57] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 2014. 6
- [58] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. In ECCV, 2020. 9
- [59] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In 4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13), 2013. 15
- [60] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 15
- [61] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. NeurIPS, 2012. 9
- [62] Weicheng Kuo, AJ Piergiovanni, Dahun Kim, Xiyang Luo, Ben Caine, Wei Li, Abhijit Ogale, Luowei Zhou, Andrew Dai, Zhifeng Chen, et al. Mammut: A simple architecture for joint learning for multimodal tasks. arXiv preprint arXiv:2303.16839, 2023. 9
- [63] Zhengfeng Lai, Vasileios Saveris, Chen Chen, Hong-You Chen, Haotian Zhang, Bowen Zhang, Juan Lao Tebar, Wenze Hu, Zhe Gan, Peter Grasch, et al. Revisit large-scale image-caption data in pre-training multimodal foundation models. arXiv preprint arXiv:2410.02740, 2024. 3
- [64] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 16
- [65] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. NeurIPS, 2021. 9
- [66] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In

- ICML, 2022. 9

[67] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In

- ICML, 2023. 9

- [68] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In European conference on computer vision, 2022. 6, 17, 18

- [69] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection, 2022. 6
- [70] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 9

- [71] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, 2014. 6, 16
- [72] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 6, 7
- [73] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024. 1, 6, 7, 9, 16, 17
- [74] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. 2024. 6
- [75] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In ICLR, 2017. 15
- [76] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 15, 16
- [77] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In CVPR, 2024. 9
- [78] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 2022. 16
- [79] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016. 6
- [80] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, 2019. 7
- [81] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 16
- [82] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question

- answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 16
- [83] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021. 16
- [84] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022. 16
- [85] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024. 1, 6, 7, 9
- [86] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV,

2016. 9

- [87] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 1, 5, 9
- [88] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. Cats and dogs. In CVPR, 2012. 15
- [89] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In ICCV, 2019. 15
- [90] Bryan A. Plummer, Liwei Wang, Christopher M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. IJCV,

2017. 6

- [91] Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, and Oncel Tuzel. Dataset decomposition: Faster llm training with variable sequence length curriculum. arXiv preprint arXiv:2405.13226, 2024. 4
- [92] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. 1, 8
- [93] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019. 1, 8
- [94] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 5, 6, 8, 9, 16
- [95] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1), 2020. 2, 3

- [96] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [97] Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses. arXiv preprint arXiv:2104.10972, 2021. 9
- [98] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 9
- [99] Mert Bulent Sariyildiz, Julien Perez, and Diane Larlus. Learning visual representations with caption annotations. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16. Springer, 2020. 9
- [100] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion400m: Open dataset of clip-filtered 400 million image-text pairs. In NeurIPS Workshop, 2021. 9
- [101] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, 2019. 6
- [102] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 3
- [103] Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. When do we not need larger vision models? arXiv preprint arXiv:2403.13043, 2024. 7
- [104] Mustafa Shukor, Corentin Dancette, Alexandre Rame, and Matthieu Cord. Unival: Unified model for image, video, audio and language tasks. Transactions on Machine Learning Research Journal, 2023. 9
- [105] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In ECCV, 2020. 7, 16
- [106] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019. 8, 16
- [107] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019. 7
- [108] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In ICCV, 2017. 9
- [109] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 9
- [110] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023. 9

- [111] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In CVPR, 2024. 9
- [112] J. Taylor, B. Earnshaw, B. Mabey, M. Victors, and J. Yosinski. Rxrx1: An image set for cellular morphological variation across many experimental batches. In ICLR, 2019. 15
- [113] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818,

2024. 9

- [114] Giorgos Tolias, Ronan Sicre, and Herv´e J´egou. Particular object retrieval with integral max-pooling of cnn activations. arXiv preprint arXiv:1511.05879, 2015. 1
- [115] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv:2406.16860, 2024. 1, 6, 7, 16, 17
- [116] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 3, 8
- [117] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3, 8
- [118] Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, and Lucas Beyer. Image captioners are scalable vision learners too. NeurIPS, 2024. 1, 5, 6, 8, 9
- [119] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In CVPR, 2018. 15
- [120] Bastiaan S Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant cnns for digital pathology. In Medical Image Computing and Computer Assisted Intervention, 2018. 15
- [121] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2015. 9
- [122] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International conference on machine learning, pages 23318–23340. PMLR, 2022. 9
- [123] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021. 9
- [124] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu

- Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 9
- [125] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528,

2024. 9

- [126] Kelvin Xu. Show, attend and tell: Neural image caption generation with visual attention. arXiv preprint arXiv:1502.03044, 2015. 9
- [127] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. TACL, 2, 2014. 6
- [128] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. ArXiv, 2021. 9
- [129] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. TMLR, 2022. 5, 9
- [130] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions, 2016. 6
- [131] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2023. 9
- [132] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In CVPR, 2022. 2, 6
- [133] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 1, 3, 5, 9, 15, 16
- [134] Biao Zhang and Rico Sennrich. Root mean square layer normalization. NeurIPS, 2019. 3
- [135] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016. 9
- [136] Xiangyu Zhao, Yicheng Chen, Shilin Xu, Xiangtai Li, Xinjiang Wang, Yining Li, and Haian Huang. An open and comprehensive pipeline for unified object grounding and detection, 2024. 6
- [137] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pretraining with online tokenizer. In ICLR, 2022. 1, 9
- [138] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024. 6

### A. Hyperparamters

Pre-training. We outline the optimization hyperaparmeters and data augmentations used during AIMV2 pre-training in Table A1. For the captions, we adopt the tokenizer used by SigLIP [133] and truncate any text longer than 77 tokens.

config ViT-L ViTs-H ViT-1B ViT-3B Optimizer Fully decoupled AdamW [76] Optimizer Momentum β1 = 0.9,β2 = 0.95 Peak learning rate 1e-3 8e-4 8e-4 4e-4 Minimum Learning rate 1e-5 Weight decay 1e-4 Batch size 8192 Patch size (14, 14) Gradient clipping 1.0 Warmup iterations 12,500 Total iterations 1,500,000 Learning rate schedule cosine decay [75] Augmentations:

RandomResizedCrop size 224px scale [0.4, 1.0] ratio [0.75, 1.33] interpolation Bicubic

RandomHorizontalFlip p = 0.5

Table A1. Pre-training hyperparameters We detail the hyperaparmeters used for pre-training all AIMV2 variants.

Attentive probing. The optimization and data augmentations hyperaparmeters for the attentive probing stage are detailed in Table A2. We use the same set of hyperaparmeters for all AIMV2 capacities and the baselines. To ensure a fair comparison, we sweep the learning rate and weight decay using the ranges detailed in Table A2 and report the strongest results for each model.

### B. Image Recognition

Evaluation benchmarks. In Table 3, we evaluate the recognition performance of AIMV2 and other baselines on a diverse set of benchmarks that encompass fine-grained recognition, medical imaging, satellite imagery, natural environment imagery, and infographic analysis. We detail the datasets, the splits and their sizes in Table B1.

High-resolution adaptation. In Table B2, we show the performance of AIMv2 models with varying image resolutions (224px, 336px, and 448px) across a broad set of recognition benchmarks. These results extend the main paper, which primarily focuses on the 224px resolution and the 3B model at 448px. We observe that scaling both the model capacity and image resolution leads to consistent improvements across most tasks.

config IN-1k Others Optimizer Pytorch’s AdamW [76] Optimizer Momentum β1 = 0.9,β2 = 0.999 Peak learning rate grid [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3] Minimum Learning rate 1e-5 Weight decay [0.05, 0.1] Batch size 1024 512 Gradient clipping 3.0 Warmup epochs 5 0 Epochs 100 Learning rate schedule cosine decay Augmentations:

RandomResizedCrop size 224px scale [0.08, 1.0] ratio [0.75, 1.33] interpolation Bicubic

RandomHorizontalFlip p = 0.5 Color Jitter 0.3 AutoAugment rand-m9-mstd0.5-inc1

Table A2. Attentive probe hyperparameters. We detail the hyperparameters used during attentive probing of AIMV2 and the baselines. For AIMV2 and the baselines we sweep over the learning rates and weight decay and report the best performance for each model.

Dataset train test classes Imagenet-1k [27] 1,281,167 50,000 1000 iNAT-18 [119] 437,513 24,426 8142 CIFAR-10 [60] 50,000 10,000 10 CIFAR-100 [60] 50,000 10,000 100 Food101 [10] 75,750 25,250 101 DTD [25] 3,760 1,880 47 Pets [88] 3,680 3,669 37 Cars [59] 8,144 8,041 196 Camelyon17 [7] 302,436 34904 2 PCAM [120] 262,144 32768 2 RxRx1 [112] 40,612 9854 1139 EuroSAT [49] 16,200 5400 10 fMoW [24] 76,863 19915 62 Infograph [89] 36,023 15,582 345

Table B1. Recognition benchmarks. We outline the recognition benchmarks, the number of train and test images for each dataset, and the number of categories.

### C. Multimodal understanding

#### C.1. Instruction Tuning Setup

Evaluation benchmarks. We list the multimodal benchmarks we use for assessing the performance of our models and the baselines in Table C2, together with the splits, prompts, and evaluation metric utilized for each dataset.

Hyperparamters. The hyperaparmeters used for the instruction tuning stage are detailed in Table C1. We use the same hyperaparmeters for all language decoders, AIMV2 capacities, and the baselines.

#### C.2. Additional Results

Instruction tuning with Cambrian. Table C3 evaluates AIMv2, fine-tuned on Cambrian, across different res-

Infographic

EuroSAT

Cifar100

Food101

iNAT-18

CAM17

Cifar10

PCAM

RxRx1

fMoW

IN-1k

DTD

Cars

Pets

model architecture

ViT-L/14 86.6 76.0 99.1 92.2 95.7 87.9 96.3 96.3 93.7 89.3 5.6 98.4 60.7 69.0 ViT-H/14 87.5 77.9 99.3 93.5 96.3 88.2 96.6 96.4 93.3 89.3 5.8 98.5 62.2 70.4 ViT-1B/14 88.1 79.7 99.4 94.1 96.7 88.4 96.8 96.5 94.2 89.0 6.7 98.8 63.2 71.7

AIMV2 224px

ViT-3B/14 88.5 81.5 99.5 94.3 96.8 88.9 97.1 96.5 93.5 89.4 7.3 99.0 64.2 72.2 ViT-L/14 87.6 79.7 99.1 92.5 96.3 88.5 96.4 96.7 93.8 89.4 6.7 98.4 62.1 71.7 ViT-H/14 88.2 81.0 99.3 93.6 96.6 88.8 96.8 96.4 93.3 89.4 7.2 98.7 63.9 73.4

- ViT-1B/14 88.7 82.7 99.4 93.9 97.1 88.9 96.9 96.5 94.2 89.5 8.4 98.9 65.1 73.7

AIMV2 336px

ViT-3B/14 89.2 84.4 99.5 94.4 97.2 89.3 97.2 96.6 93.2 89.3 8.8 99.0 65.7 74.0 ViT-L/14 87.9 81.3 99.1 92.4 96.6 88.9 96.5 96.6 94.1 89.6 7.4 98.6 62.8 72.7 ViT-H/14 88.6 82.8 99.4 93.6 97.0 88.9 96.8 96.5 93.4 89.6 7.8 98.7 64.8 74.5

- ViT-1B/14 89.0 83.8 99.4 94.1 97.2 88.9 97.1 96.6 93.5 89.9 9.2 99.1 65.9 74.4

AIMV2 448px

ViT-3B/14 89.5 85.9 99.5 94.5 97.4 89.0 97.4 96.7 93.4 89.9 9.5 98.9 66.1 74.8

Table B2. Frozen trunk evaluation for recognition benchmarks, high resolution AIMV2 models. We report the recognition performance of the AIMV2 high resolution family of models when compared to the base 224px models shown in the main paper. All models are evaluated using attentive probing with a frozen backbone.

config Llava SFT mixture Cambrian Optimizer Pytorch’s AdamW [76] Optimizer Momentum β1 = 0.9,β2 = 0.999 Decoder peak learning rate 1e-5 2e-5 Connector peak learning rate 8e-5 1.6e-4 Minimum Learning rate 0 Weight decay 0.01 Batch size 128 512 Gradient clipping 1.0 Warmup iterations 250 700 iterations 5000 14,000 Learning rate schedule cosine decay Transformations [PadToSquare, Resize]

Table C1. Instruction tuning hyperaparmeters. We detail the hyperparameters of the instruction tuning stage, both for the Llava SFT mixture [73] and Cambrian [115].

olutions using a tiling strategy. Unlike the main paper, which uses Llava SFT, Cambrian offers a less in-domain data mix and achieves stronger results on text-rich benchmarks. Starting with a base resolution of 336px (matching the encoder’s pretraining resolution), higher resolutions (672px and 1008px) are obtained with tiling; by splitting high-resolution images into 2×2 and 3×3 grids. AIMv2 paired with tiling shows consistent improvements on textrich benchmarks such as InfoVQA, ChartQA, DocVQA, and TextVQA. However, on benchmarks like COCO, NoCaps, TextCaps, and MMEp, no significant gains are observed with increased resolution.

Instruction tuning with DCLM-1B decoder. In Figure C2, we present the same comparison between OAI CLIP, SigLIP, and AIMV2 as in the main paper, but this time using the Llava SFT mixture paired with a DCLM 1B decoder. These results demonstrate that AIMV2 consistently outperforms the baselines, regardless of the decoder’s capacity. Notably, in the practical setting of a small decoder, AIMV2 maintains its position as the preferred choice for multimodal understanding tasks.

Benchmark Split Prompt Evaluation Metric VQAv2 [41] Val

Accuracy GQA [52] Val Accuracy OKVQA [81] Val Accuracy TextVQA [106] Val Accuracy DocVQA [83] Test ANLS InfoVQA [84] Test ANLS ChartQA [82] Test Relaxed accuracy SEED [64] Test (image split) Accuracy

Answer the question using a single word or phrase.

ScienceQA [78] Test (image split) Answer with the option’s letter from the given choices directly.

Accuracy MME [38] Test (image split) Accuracy COCO [71] Val

CIDEr TextCaps [105] Val CIDEr NoCaps [2] Val CIDEr

Provide a one-sentence caption for the provided image.

Table C2. Multimodal benchmarks. We provide the list of benchmarks used during the multimodal evaluation including the reference, split, prompt, and the evaluation metric.

#### C.3. Qualitative Results

The qualitative results in Figure C1 highlight AIMv2’s strengths on multimodal evaluations compared to SigLIP [133] and OAI CLIP [94] after instruction tuning on Cambrian. In the first three examples, AIMv2 excels in text-rich tasks by correctly localizing and extracting the relevant textual information. For instance, in the example on the left, AIMv2 is able to identify the correct value for “supreme gasoline” and outputs the correct operation for finding the solution (“Divide 50 by 3.65”). This contrasts with OAI CLIP and SigLIP, which provide generic and incomplete answers that fail to focus on the relevant information. Similarly, AIMv2 successfully identifies the license plate number (“AED-632”) in a blurry image, demonstrating robust localization and reading capabilities in challenging settings. In the luggage example, AIMv2 accurately reads the weight (“30.7”), despite the presence of multiple distracting objects in the image, while the other models make mistakes. Finally, in the calorie estimation example, AIMv2 provides a more plausible response (“1000 calories”) based on its knowledge, whereas SigLIP and OAI CLIP offer less contextually plausible answers.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

: How many gallons of supreme gasoline can I get with $50?

: What might be the weight of this luggage?

: How many approximate calories is in this meal?

: What is the license plate number?

OAI CLIP: To calculate how many gallons of supreme gasoline you can get with $50, you would divide the total amount of money you

OAI CLIP: The license plate number is partially obscured and not fully legible due to the blurriness of the image.

###### OAI CLIP: 3.7

###### OAI CLIP: 500

SigLIP: 30 pound

SigLIP: 500

SigLIP: To calculate how many gallons of supreme gasoline you can get with $50, you would divide $50 by the price per gallon

SigLIP: The license plate number is not clearly visible in the image provided.

AIMv2: 30.7

AIMv2: 1000

AIMv2: The license plate number is AED-632

AIMv2: Divide 50 by 3.65.

Figure C1. Qualitative comparison of AIMv2, SigLIP, and OAI CLIP on multimodal tasks after instruction tuning on Cambrian. AIMv2 demonstrates superior performance in both text-rich (e.g. extracting relevant information or reading text in cluttered scenes) and knowledge-based scenarios (e.g., estimating caloric content), showcasing its ability to focus on relevant information, accurately localize text, and provide contextually appropriate answers.

data mix decoder resolution VQAv2 GQA OKVQA TextVQA DocVQA InfoVQA ChartQA ScienceQA COCO TextCaps NoCaps MMEp Cambrian Llama 3.0 336px 75.5 71.5 61.1 58.3 50.2 35.1 51.7 78.7 95.5 82.3 98.1 1594 Cambrian Llama 3.0 672px 77.5 72.8 62.0 69.1 76.4 48.3 64.7 79.4 92.6 80.6 95.4 1482 Cambrian Llama 3.0 1008px 77.7 73.2 62.0 72.2 79.2 53.5 65.1 81.6 93.7 81.6 97.6 1507

Table C3. Additional multimodal evaluations. We compare the performance of AIMV2 with different SFT data mixtures (Llava [73] and Cambrian [115]), and resolutions (336px, 672px and 1008px).

COCO LVIS Val

Model APall APs APm APl APall APr APc APf OpenAI CLIP 59.1 43.5 63.5 74.8 31.0 17.6 27.2 41.2 DFN-CLIP 59.8 44.0 63.8 75.3 30.7 17.2 26.4 41.5 SigLIP 58.8 41.7 62.8 75.7 30.5 16.5 26.5 41.1 DINOv2 60.1 43.7 64.2 75.8 30.8 18.5 26.1 41.4 AIMV2 60.2 44.5 64.3 75.4 31.6 18.0 27.0 42.8

Table D1. Performance on OVD Benchmarks. We report the performance on mean average precision (AP) for COCO and LVIS. For COCO, we also report AP for the small, medium, and large subsets, while for LVIS, we report on rare, medium, and frequent subsets.

### D. Detection, Segmentation and Grounding

#### D.1. Open Vocabulary Detection and Grounding

Performance on Small Objects. In Table D1 we report the breakdowns of COCO between classes that are either small, medium, or large. We can observe that AIMV2 consistently outperforms on the small classes by +0.5 AP, compared to DFN-CLIP, the second best performing model in that breakdown. This is further emphasized by the results reported on LVIS val, as objects in LVIS are more likely to be small. There we observe an improvement of +1.3 AP on the frequent subset against DFN-CLIP.

Window COCO LVIS Val RefCOCO RefCOCO+ RefCOCOg Model Size APall APall Val P@1 Val P@1 Val P@1 DINOv2

60.1 30.8 92.2 85.9 89.1 AIMV2 60.2 31.6 92.6 86.3 88.9

16

- DINOv2 59.6 29.6 92.1 85.0 88.7

AIMV2

24

59.8 31.2 92.3 85.8 89.1

- DINOv2 60.2 30.7 92.5 86.1 89.5

32

60.3 32.9 92.5 86.3 88.9 DINOv2 37 60.2 31.1 92.2 85.9 88.4

AIMV2

Table D2. Ablation across window sizes. We report the performance on mean average precision (AP) for COCO and LVIS. For RefCOCO* we report Precision @1 on the respective validation splits.

Window Size Ablation. Due to varying input resolutions and feature map sizes used during pre-training, we ablate the effect of window size [68] for AIMV2 and DINOv2 in Table D2. For AIMV2 the input image resolution is scaled during pre-training such that the feature map size matches the window size during finetuning, while for DINOv2 the window size is fixed to match AIMV2. For comparison we also add DINOv2 trained with a window size of 37, which matches its pre-training feature map size. Across the window sizes, AIMV2 outperforms DINOv2 across all OVD and for two out of three referring comprehension benchmarks. When comparing our best performing AIMV2 with the best performing DINOv2 across all

OAI CLIP SigLIP AIMV2

96

94

93

90

88

83

74

73

71

65

65

64

64

64

64

64

63

63

63

59

57

49 44

48

47

41

38

21 18 13

21 17 13

21 17 13

VQAv2 GQA OKVQA TextVQA DocVQA InfoVQA ChartQA ScienceQA COCO TextCaps NoCaps MMEp

(a) DCLM 1B + Llava SFT mixture

Figure C2. Instruction with a small decoder (DCLM). Performance comparison of OAI CLIP, SigLIP, and AIMv2 across 12 multimodal benchmarks using the Llava SFT mixture paired with a DCLM 1B decoder. AIMV2 exhibits superior performance across most benchmarks, even with the constrained capacity of a small decoder.

detection mAP50:95 mask mAP50:95

Model APall APs APm APl APall APs APm APl OAI CLIP 53.6 37.2 58.5 69.2 46.7 26.6 50.9 66.2 DFN-CLIP 53.4 37.1 58.3 69.3 46.2 26.4 50.8 66.4 SigLIP 53.3 37.2 57.6 69.7 46.6 27.1 50.5 66.3 DINOv2 55.5 39.5 59.9 70.6 48.3 29.4 52.3 67.4 AIMV2 54.0 37.4 58.8 70.0 46.7 26.7 51.1 66.5

Table D3. COCO17 detection and segmentation benchmarks. We report overall detection and segmentation scores along with the small, medium, and large subset breakdowns.

benchmarks, we observe that AIMV2 strongly outperforms on LVIS Val while outperforming on all except one benchmark against DINOv2.

#### D.2. Detection and Segmentation via ViTDet MaskRCNN

To compare vision only capabilities of the encoders we incorporate them into a Mask-RCNN[47] detection model as backbones by utilizing a ViTDet formulation to accommodate for high resolution (1024) detector training / testing input size. We ensure that ViTDet [68] backbone forward pass outputs match the respective ViT-L implementations before the training. We utilize the same set of hyperparameters for training all compared detectors: consistent windowed attention size (16) ensuring comparable compute, AdamW optimizer, cosine decay learning rate schedule, layer-wise learning rate, and weight decay. All detectors are finetuned on coco17 train split for 100 epochs with a global batch size of 64 following the default recipe from MMDetection [18]. We report results from the coco17-val split in Table D3. AIMV2 consistently outperforms encoders pre-trained on contrastive objectives, falling slightly behind DINOv2 which provides the strongest performance.

