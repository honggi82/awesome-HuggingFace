## arXiv:2512.21004v1[cs.CV]24Dec2025

##### Learning from Next-Frame Prediction: Autoregressive Video Modeling Encodes Effective Representations

Jinghan Li Yang Jin* Hao Jiang Yadong Mu Yang Song Kun Xu* Peking University {li.jh, jiny}@stu.pku.edu.cn, syxu828@gmail.com https://github.com/Singularity0104/NExT-Vid

#### Main idea

###### Abstract

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

### Main idea

Recent advances in pretraining general foundation models have significantly improved performance across diverse downstream tasks. While autoregressive (AR) generative models like GPT have revolutionized NLP, most visual generative pretraining methods still rely on BERT-style masked modeling, which often disregards the temporal information essential for video analysis. The few existing autoregressive visual pretraining methods suffer from issues such as inaccurate semantic localization and poor generation quality, leading to poor semantics. In this work, we propose NExT-Vid, a novel autoregressive visual generative pretraining framework that utilizes masked next-frame prediction to jointly model images and videos. NExT-Vid introduces a context-isolated autoregressive predictor to decouple semantic representation from target decoding, and a conditioned flow-matching decoder to enhance generation quality and diversity. Through context-isolated flowmatching pretraining, our approach achieves strong representations. Extensive experiments on large-scale pretrained models demonstrate that our proposed method consistently outperforms previous generative pretraining methods for visual representation learning via attentive probing in downstream classification.

Autoregressive Video Modeling

| |
|---|
| |

| | |
|---|---|
| | |
| | |
| | |

| |[Figure 6]| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| |
|---|
| |

| | |
|---|---|
| | |
| | |
| | |

| |
|---|
| |
| |
| |

| | |
|---|---|
| | |
| | |
| | |

| |[Figure 7]| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

[Figure 8]

[Figure 9]

[Figure 10]

| |
|---|

| |
|---|

(a) Pretraining paradigm using masked next frame generation

Context-Isolated

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

[Figure 22]

Flow Decoder

Autregressive Predictor

Encoder

Representation Alignment

: visible token : query token

: predicted token

(b) Overview of our proposed framework

Figure 1. Conceptual illustration of context autoregressive flowmatching pretraining. a) Each frame in the video clip is generated based on masked previous ones. b) The encoder output is isolated to strengthen its semantic representation, and the conditioned flow-matching decoder boosts the generation quality and diversity.

studies in visual generative pretraining continue to follow the BERT-style masked modeling paradigm [5, 53]. Given the tremendous potential of autoregressive generative models, we aim to explore unified visual autoregressive generative pretraining for images and videos.

###### 1. Introduction

Recently, pretraining general foundation models [1, 37, 40] for a wide range of downstream tasks has attracted a lot of attention. Trained on large-scale datasets, they can produce high-quality representations and exhibit superior generalization compared to task-specific models. In NLP, generative pretrained foundation models, from BERT [13] to GPT [39], have gradually become mainstream. Notably, GPT leverages autoregressive pretraining to achieve powerful semantic of sequential dependencies. In contrast, most

Most existing visual generative pretraining methods, such as MAE [20], rely on masked modeling that recovers the missing portions of input data. However, when applied to videos [48, 52], they often neglect the crucial temporal information, which is vital for accurate action and motion understanding. Recently, some studies start to explore visual generative autoregressive pretraining. For instance, iGPT [9] leverages autoregressive next-pixel prediction, while AIM [14] adopts next-patch embedding predic-

*Project leader

tion. Toto [41] further extends these concepts to video pretraining by using VAEs to compress video frames. Despite these advances, several problems remain. First, target-totarget pretraining objective tends to embed semantic representations deep within the intermediate layers [41], necessitating layer-by-layer probing for extraction. Second, the direct regression objective used to generate visual patches often struggles to capture the diversity of input data, particularly in complex natural scenes. This typically results in blurry or averaged output with unclear semantic quality.

In this work, we further explore autoregressive generative pretraining and propose NExT-Vid, a novel visual autoregressive generative pretraining method that adopts a masked next-frame prediction paradigm in Fig. 1a. Our core insight is to decouple semantic representation from target decoding. In other words, the semantic representations serve only as references to guide target generation, rather than acting as the starting point and participating in the hidden state transformation during decoding. We find that conditioned denoising models, such as conditioned flowmatching [34], possess inherent advantages. They accept conditional information and generation targets originating from two different domains (e.g. text-to-image models [16] where conditions come from text and targets are images), which makes them excellent isolators between representation and decoding. Furthermore, flow-matching models generate samples through multi-step denoising, leading to better generation quality and diversity compared to deterministic regression.

Based on the above motivation, our core design comprises two main components and the pipeline is shown in Fig. 1b. 1) Context-isolated autoregressive predictor. We decouple the core prediction module from conventional autoregressive models, which predicts the latent features of next frame from the previous ones. Unlike end-to-end GPT models that align output directly with the targets [9], our autoregressive predictor generates implicit conditions that guide the generative decoding process. Besides, the predictor employs cross attention and representation alignment to further isolate and stabilize the encoder output. 2) Conditioned flow-matching decoder. It takes the latent features produced by the autoregressive predictor as conditions and generates the corresponding target for the next frame. There are many methods for conditioned generation, such

- as AdaLN in DiT [38], or sequence concatenation in textto-image generative models [16]. However, these injection methods are typically used in unbalanced scenarios (e.g., sparse text versus dense images). In our case, the conditions predicted by the autoregressive predictor have the same spatial structure as the target feature maps. Therefore, we perform spatially aligned concatenation, so that each noisy target token is denoised in conjunction with the predicted condition of its corresponding patch.

Our main contributions are as follows:

- • We propose NExT-Vid, a novel visual generative autoregressive approach that effectively models temporal information between video frames, and enhances video understanding through high-quality generative modeling.
- • The proposed context isolation design effectively decouples semantic representation and target decoding, enabling the encoder to output strong semantic representations without being affected by transformations in the decoder hidden state.
- • We train ViT encoders of various scales, up to 1B parameters, and evaluate them via attentive probe methods on K400 [27], ImageNet [43], SSv2 [18] and Diving48 [26]. Our ViT-G model surpasses existing generative pretraining methods particularly on video understanding.

###### 2. Related Works

Visual Representation Pretraining. Recently, the rise of multi-modal models [35] has demanded vision models with enhanced semantic extraction and representation capabilities. Approaches such as CLIP [40], Siglip [49, 62], Blip [31, 32] and Internvideo [55, 57] address this by directly aligning visual and textual representations. As for self-supervised learning, DINO [7, 37, 44] performs localglobal alignment between teacher and student models, and JEPA [3, 4, 6, 30] introduces embedding space alignment.

Notably, Mask-based visual modeling, inspired by NLP [13], adopts mask prediction as the self-supervised learning objective, as demonstrated by significant advances. BEIT [5] and BEVT [53] use discrete token prediction for generative pretraining, while MAE [20] employs a maskbased autoencoder to perform pixel-level regression. Building on these approaches, CAE [10] introduces context autoencoder designs, and VideoMAE [48, 52] extends them to video pretraining. Subsequent methods like DiffMAE [58] further enhance pretraining by integrating denoising-based objectives. All of these techniques draw on the BERT [13] pretraining paradigm with masked generative objectives.

Building on the success of GPT [1, 39] in NLP, some studies explore autoregressive generation for visual pretraining. iGPT [9] performs pixel sequence prediction, while Toto [41] adopts discrete token prediction and extends it to video. In this paper, we further apply autoregressive methods to visual pretraining and introduce a more powerful generator as the training objective.

Visual Generative Modeling. Visual generative models have long been a research focus. Significant progress has been made in denoising-based image generation [21, 45]. Notably, latent-space diffusion models [42] leverage pretrained variational autoencoders (VAEs) [28] to compress images into compact representations, allowing for efficient denoising and realistic image generation. Further adaptations of denoising-based models to videos [22] enable high-

# Architecture

[Figure 23]

Denoising Target

…

###### …

: Visible Cube : Masked Cube : Learnable Query

: Encoder Representation : Reference Representation : Predicted Representation

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]|
|---|

|[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]|
|---|

|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]|
|---|

KV Cross Attention KV Cross Attention KV Cross Attention

DiT Block

× 𝑛𝑛𝑑𝑑

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Flow-Matching Decoder

: Frame-Wise Causal Attn Mask : Autoregressive Attn Mask : Frame-Isolated Attn Mask

…

Add Noise

× 𝑛𝑛𝑝𝑝

###### …

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Autoregressive Predictor

[Figure 47]

…

Concatenation as Condition

[Figure 48]

Representation Alignment

M

Stop-Grad

Decoder Mask

…

…

EMA

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

…

ViT Block Encoder

× n𝑒𝑒

[Figure 52]

Reference Encoder

[Figure 53]

[Figure 54]

…

…

[Figure 55]

[Figure 56]

[Figure 57]

…

[Figure 58]

[Figure 59]

[Figure 60]

VAE Encoder

VAE Encoder

VAE Encoder

[Figure 61]

[Figure 62]

[Figure 63]

Masked Sequence

Full Sequence

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

M

…

Encoder Mask

0 𝑡𝑡 − 1

- Figure 2. Overview of the proposed pretraining pipeline. The input video is split into two branches. One branch is masked and fed into the encoder to obtain local representations, while the other branch with the entire sequence is processed by the reference encoder for representation alignment. The autoregressive predictor employs cross-attention to aggregate local features and predict the representations for the next frame. The predicted representations is then aligned with the reference encoder and passed to the flow-matching decoder to generate the VAE latent features of the next frame. The custom attention masks for autoregressive modeling are shown in Fig. 3.

quality video generation.

Correspondingly, there are autoregressive visual generation methods with discrete visual representations. VQGAN [15, 61] uses tokenizers with vector quantization for discrete image encoding and performs autoregressive generation in the latent space. LlamaGen [47] successfully scales autoregressive models to generate high-quality images. Other studies [23, 25, 29, 59], use autoregressive approaches for video generation. Additional studies [8, 50] combine autoregressive models with denoising models to support long-duration video generation. PyramidFlow [24] introduces image pyramids to speed up the denoising process. Building on this, we will further expand generative pretraining by incorporating autoregressive denoising.

###### 3. Method

###### 3.1. Masked Next-Frame Generative Pretraining

We focus on modeling visual representations that are applicable to both videos and images. Images lack inherent spatial autoregressive properties, whereas videos have sequential properties in the temporal dimension. We perform

autoregressive generative pretraining based on next-frame prediction. We start with abstracting the entire generation pipeline as G and focusing on the generation of one frame ft. The generator G receives the previous t frames and generates the next frame ft:

ft = G(ft−1,ft−2,··· ,f0). (1)

Unlike visual generative models, our goal is not to generate sufficiently detailed video frames ft, but rather to obtain a good representation of existing videos through generative pretraining. Videos contain significant temporal redundancy compared to the appropriate sequential mutual information of natural language. Imagine a fixed-scene video where most patches have minimal temporal variation. This is ideal for generation, but not for representation learning. In extreme cases, the autoregressive generator can simply copy most of the previous frame’s content ft−1 to create the next frame ft, regardless of the quality of the representation. To this end, we propose the masked next-frame generative pretraining, with the whole pipeline becoming:

ft = G(M(ft−1,ft−2,··· ,f0)), (2)

encoder Eref processes the full unmasked sequence from f0 through ft, yielding the reference representation c′t for ft to align with the predicted representation zt.

: Attended : Masked

KV

KV

KV

0 1 … t - 2 t - 1

0 1…t-2 t-1

0 1 … t - 2 t - 1

| |
|---|

| |
|---|

t-2t-1…10

t-2t-1…10

t-2t-1…10

| |
|---|

| |
|---|

In practice, to ensure that context c remains unchanged during prediction, the predictor is implemented by Qformer [32] style autoregressive blocks with causal cross attention, where the historical frame context c0···t−1 is always used as the key and value to calculate the predict zt and is truncated here.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Q

Q

Q

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

Frame-Wise Causal Autoregressive Frame-Isolated

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

(Reference) Encoder Autoregressive Predictor Flow-Matching Decoder

###### 3.3. Conditioned Flow-Matching Target

- Figure 3. Custom attention masks for autoregressive modeling. The frame-wise causal mask temporally aggregates video frames, while the autoregressive mask ensures that a frame can only see previous frames. Frame-isolated mask enables individual frame generation, effectively preventing information leakage.

The choice of generator G is crucial for both the training objective and the quality of the generation. Here, we adopt the conditioned flow-matching model for high-quality and diverse generation, where zt serves as condition information in generative decoding and remains distinct from the target domain, such as pixel space or VAE latent features.

where the mask M acts as a powerful information attenuator, making it difficult to generate the next frame but enhancing the quality of semantic representation.

Revisiting flow-matching modeling [36] under condition y, it aims to transform the noise distribution pnoise(x) into the conditioned data distribution pdata(x|y) by training gθ to fit a vector field:

###### 3.2. Context-Isolated Autoregressive Prediction

d dτ

Directly modeling the entire generation process G with a causal transformer is not ideal. Previous studies [41] shows that semantic representations exist in the intermediate layers of GPT-style generative models. This implies that the model autonomously learns a two-stage process from semantic prediction to target decoding. Inspired by CAE [10], we further divide the generation process G into the semantic representation prediction with semantic encoder E and autoregressive representation predictor AR, followed by generative decoding with generator G. In the first stage, the model predicts the implicit representation zt of the current frame from embeddings of previous frames:

gθ(x,y,τ) ≈

γ(x0,x1;τ), (6)

where the noisy sample x0 ∼ pnoise(x), the data sample x1 ∼ pdata(x|y) and timestep τ ∼ Uniform(0,1). γ(x0,x1;τ) = (1 − τ)x0 + τx1 is an effective choice of the conditional probability path with linear interpolation of data and noise. The training objective is in the form of velocity regression:

0,x1,y,τ ∥gθ(γ(x0,x1;τ),y,τ) − v∥2 , (7)

Lcfm = Ex

where v = x1−x0. Additionally, we can sample τ multiple times under the same condition y to enhance the training efficiency.

zt = AR(ct−1,ct−2,··· ,c0), ci = E(M(f0...i)), (3)

Returning to the generation of frame ft, we follow latent diffusion models, using the latent features compressed by VAE encoder E as the generation target x1, while the condition y comes from the prediction of autoregressive predictor. The training target becomes:

where ci is the frame-wise causal semantic representation obtained by the encoder E, while zt is the latent feature of the next frame predicted by autoregressive predictor AR. After prediction, zt is artificially separated from c and used in generative decoding by G, while c tends to remain fixed and isolated:

0,h1,zt,τ ∥gθ(γ(h0,h1;τ),zt,τ) − vh∥2 ,

Ltflow = Eh

ft = G(zt). (4)

(8) where h0 ∼ N(0,I), h1 ∼ E(ft) and vh = h1 − h0.

However, without any constraints, zt could become a substitute for previous c, indirectly involving c in the target generation and decoding process. To further enhance the representation of the t-th frame in zt and prevent c from participating in subsequent decoding, we incorporate representation alignment regularization:

Following the conditioned decoder in [51], zt and the noisy target are concatenated along the channel dimension, while being aligned in the spatial-temporal dimension.

Pretraining Objective. Given a video clip with frames f0 to fT, the final training objective is as follows. The balance factor β controls the strength of alignment regularization.

Ltalign = MSE(zt,sg[c′t]), c′t = Eref(f0...t), (5)

T

1 T − 1

where Eref denotes the reference encoder updated by EMA, and sg indicates the stop-gradient operation.The reference

(Ltflow + βLtalign). (9)

L =

t=1

###### 3.4. Architecture and Implementation

We implement the overall framework shown in Fig. 2.

Encoder. The encoder adopts a standard ViT and accepts multi-frame visual input (images are expanded into frame-invariant videos). The input video is divided into non-overlapping 3D patches along spatial-temporal dimension, which are then masked and fed into the encoder with 3D RoPE [46]. To accommodate autoregressive training, frame-wise causal attention (see Fig. 3) is incorporated into the encoder, so that each patch can only attend to content within its own frame and previous frames. Additionally, we maintain a reference encoder updated by EMA [19], which receives the full sequence of videos for the alignment loss.

Masking Strategy. Following the strategies in [4, 52], we randomly mask multi-frame visual inputs before feeding them into the encoder to extract context c. To this end, we employ a temporally consistent masking strategy, whereby the same spatial patch locations are masked simultaneously across multiple frames within the same video. Since patches

- at identical spatial locations in different frames are typically highly correlated, this method helps to mitigate the complementary effect caused by redundant information among frames. In addition, we apply multiple mask strategies simultaneously during pretraining. Autoregressive Predictor. The autoregressive predictor consists of a set of learnable queries and multiple crossattention blocks. The lenrnable queries serve as mask tokens, and interact with the visible context token (i.e., the encoder’s output) through cross-attention with autoregressive mask (see Fig. 3). Specifically, the context serves as both the key and value for each cross-attention layer, which ensures that the context is isolated. Therefore, the context does not participate in embedding transformations within the autoregressive predictor, preserving stable semantics. Generation Target. To reduce computational cost during generative training, we follow latent generation models [42], which take latent frames extracted by VAEs [28] as generation targets. We directly apply image VAEs [60], which are designed for high-fidelity image compression, to extract frame-wise latent from videos. To align latent

frames spatially and temporally with the condition zt, latent frames are reorganized as target latent with stacked channels. After alignment, the corresponding tokens in the target sequence and the computed sequence represent the same cube of the original video. We also explored other targets, such as video VAEs [2], semantic encoders [49], and pixels. Flow-Matching Decoder. The flow-matching decoder adopts a standard DiT [38] for training and generation of conditioned flow-matching. Following [51], the decoder receives the concatenated input of the condition zt and the noisy target with timestep τ, and outputs the prediction of velocity vh. The timestep is injected by AdaLN [38]. Since condition zt results from the next frame prediction of

the autoregressive predictor, which has already completed feature prediction, the decoder only needs to generate the latent frame separately. Therefore, the decoder incorporates a frame-isolated attention mask (see Fig. 3), which enables full attention within each frame and prevents attention across frames, allowing separate generation of each frame.

###### 4. Experiments

###### 4.1. Pretraining Setting

Models. We pretrain three variants with different encoder sizes: ViT-L (24 layers, 1024 width), ViT-H (32 layers, 1280 width), and ViT-G (40 layers, 1048 width). The video frames are divided into 16×16×2 patches with 3D RoPE.

All encoders have the same autoregressive predictor and flow decoder configurations. The autoregressive predictor consists of 12 cross-attention layers with a hidden size of 384 and applies autoregressive attention masks. The flow decoder consists of 12 DiT blocks with a hidden size of 384 and applies frame isolation attention masks.

Dataset. The scale and diversity of the pretraining data significantly impact model performance. To this end, we construct a mixed dataset including videos and images. The dataset comprises 2.4M hours of videos and 1.28M images. The image data originates from ImageNet-1k [12], which contains a wide variety of everyday objects and scenes. The video data comes from various sources, including human action videos from Something-Something-V2 [18] and Kinetics-400 [27], as well as daily life footage from InternVid [56] and in house video data.

Training Strategy. We pre-train our models for 132K steps across 96 H100 GPUs, using a batch size of 3072, which amounts to a total of 405M samples and 830B visual tokens. Pretraining begins with a 12K-step warm-up phase, followed by a formal training phase of 108K steps and a cool-down stage of 12K steps, lasting 146 hours.

###### 4.2. Probe-Based Classification

Benchmarks. We create downstream visual classification tasks on four widely used benchmarks: Kinetics-400 [27] (K400), ImageNet-1K [12] (IN1K), Something-SomethingV2 [18] (SSv2), and Diving48 [26]. For image evaluation, IN1K covers 1,000 categories of common things, which requires model to extract semantic information about objects and scenes. For video evaluation, SSv2 contains 174 categories of human actions and Diving48 contains 48 categories of diving videos, focusing on understanding of gestures and movements. K400 comprises 400 categories of everyday videos, evaluating the joint comprehension of both scenes and actions.

Attentive Probe. For probe-based visual classification, we train single attentive pooling layer and a linear classification layer on top of the frozen encoder using training data

Table 1. Attentive probing results on K400, IN1K, SSv2 and Diving48. We report top-1 accuracy with the encoder frozen and only a single layer of attentive probe trained. Our ViT-H model outperforms other video generative pretraining models. Our ViT-G model achieves state-of-the-art performance among generative pretraining methods, while remaining competitive with discriminative pretraining methods.

Type Method Arch. Param. K400 IN1K SSv2 Diving48

Methods Pretrained on Images IJEPA [3] ViT-H 630M 79.7 84.4 50.0 DINOv2 [37] ViT-G 1.1B 83.4 86.2 50.6 82.5 Siglip2 [49] ViT-G 1.2B 87.3 88.0 49.9 75.3 OpenCLIP [11] ViT-G 1.8B 81.8 85.3 34.8 -

Discriminative Pretraining

Methods Pretrained on Videos

VJEPA [6] ViT-H 630M 82.0 75.9 71.4 87.9 VJEPA2 [4] ViT-G 1B 86.6 84.6 75.3 90.1 InternVideo2 [57] ViT-G 1B 87.9 85.8 67.3 86.4 VideoPrism [63] ViT-G 1B 87.6 - 68.5 71.3

Methods Pretrained on Images

CAE [10] ViT-L 300M - 81.2 - BEiT [5] ViT-L 300M - 62.2 - MAE [20] ViT-H 630M - 80.9 - iGPT [9] GPT-2 6.8B - 72.0 - -

Methods Pretrained on Videos MVD [54] ViT-L 200M 79.4 73.3 66.5 OmniMAE [17] ViT-H 630M 71.4 76.3 65.4 VideoMAE [48] ViT-H 630M 79.8 72.3 66.2 VideoMAEv2 [52] ViT-G 1.1B 71.2 71.4 61.2 Toto [41] LLaMA 1B 74.4 75.3 - NExT-Vid-L ViT-L 300M 78.5 76.3 63.9 82.7 NExT-Vid-H ViT-H 600M 80.6 79.0 67.0 84.5 NExT-Vid-G ViT-G 1.1B 83.1 81.4 69.5 87.2

Generative Pretraining

from four benchmarks. Attentive pooling employs a learnable query token that interacts with the encoder’s output via cross-attention, which dynamically allocates aggregation weights to different visual tokens.

Main Results The experimental results are summarized in Tab. 1. Our model achieves state-of-the-art performance among generative pretraining approaches. 1) Compared with other video pretraining methods, our compact ViT-L model (300M parameters) achieves the best results on IN1K with top-1 accuracy of 76.3. It also performs competitively on the K400 set, achieving 78.5 accuracy, with the best result being 79.8. The larger ViT-H model (600M parameters) outperforms all other methods across three benchmarks, achieving a notable improvement of 2.7 points on IN1K. 2) Compared with generative pretraining methods both on images and videos, our ViT-G model achieves leading performance across all benchmarks, with improvements of 3.3 on K400, 0.2 on IN1K, and 3.0 on SSv2. On Diving48, it outperforms the ViT-L model by 4.5. Notably, our model delivers consistent increases across three benchmarks when compared to VideoMAEv2, highlighting the superior semantic representation of autoregressive flow-matching generative training over direct regression. Against Toto, another au-

toregressive model, our approach yields improvements of 8.7 on K400 and 6.1 on IN1K, demonstrating the advantage of next-frame generation over token-by-token training. When compared to CAE, which also uses context isolation, our model does not outperform CAE on IN1K at the same scale—likely because CAE is trained exclusively on IN1K, while we use mixed image and video data. However, when scaled to 1B parameters, our model surpasses the best performance of CAE.

When extending the comparison to include discriminative pretraining methods, our model remains competitive. 1) Compared with discriminative image pretraining methods, our model significantly outperforms them on SSv2 and Diving48, demonstrating the effectiveness of leveraging temporal information in video pretraining for downstream action recognition tasks. On K400, we outperform IJEPA and OpenCLIP. While image-based methods are better suited for IN1K, we still achieve over 80 accuracy. 2) Compared with discriminative video pretraining models, we are highly competitive. On K400 and IN1K, we outperform VJEPA; on SSv2, we outperform both InternVideo2 and VideoPrism; and on Diving48, we surpass VideoPrism and InternVideo2 and perform similarly to VJEPA.

(a) Data scaling (b) Model scaling

- Figure 4. Scaling curves both on training data amount and model parameters. a) As training data increases, model performance initially grows rapidly before stabilizing, and then improves further after a cool-down stage. b) Increasing model parameters also improves performance, especially from ViT-L (300M) to ViT-H (600M). After the cool-down stage, ViT-G (1100M) significantly outperforms ViT-H.

[Figure 81]

[Figure 82]

50%

Masked Ori.Ori.

[Figure 83]

50%

Masked

Seed Image Autoregressive Generation

| |
|---|

- Figure 5. Visualization of generation. Top and middle: Masked generation on ImageNet and SSv2, where “ori” denotes the original samples. Bottom: Autoregressive generation based on a seed image from ImageNet.

Scaling on Data Size. We plot the scaling curves (Fig. 4a) of the ViT-G model on downstream tasks as the training data increases. The model’s performance grows rapidly during the early stages of training, then stabilizes after about 100 million video clips. In the later stages, performance on IN1K, K400, and SSv2 continues to increase slowly, while performance on diving48 begins to fluctuate. This may be related to the relatively small size of Diving48, which is prone to overfitting. Notably, during the final cooldown phase, the model’s performance improves further, particularly on SSv2 and Diving48, which require strong action recognition capabilities. This is associated with the use of more frames during the cooldown stage, enhancing the model’s understanding of motion.

Scaling on Model Size We also plot the scaling curves (Fig. 4b) as the model parameters increase. We find that there is a significant performance improvement from ViT-L (300M) to ViT-H (600M), whereas the improvement from ViT-H (600M) to ViT-G (1100M) is much smaller, which is consistent with observations in VideoMAEv2 [52]. However, we further observe that adding the cool-down stage to ViT-G significantly improves the model’s performance.

###### 4.3. Visualization

Fig. 5 shows the generation visualization results. Even with a compact flow decoder (only 30M) during representation learning, our model generates high-quality, diverse samples. Additionally, the autoregressive generation allows it to ef-

fectively capture rich temporal information from the video.

###### 4.4. Ablation Study

To facilitate rapid performance comparison and validation in the ablation experiments, we employ an early stopping strategy by fixing encoder training steps at 15K. The evaluation of the attentive probe is performed on both IN1K and SSv2, following the same protocol as Sec. 4.2.

Model Designs. We conduct ablation studies on key components of our model, as summarized in Tab. 2. The results demonstrate that our complete design achieves the highest performance (fifth row).

In particular, Focusing on the first, second, and fifth rows: The experiments show that flow-matching generative pretraining alone yields highly competitive results, achieving the highest IN1K accuracy (first row). However, introducing context-isolation reduces performance (second row), suggesting that simply fixing the encoder output makes it too closely to the latent space, which hinders semantic representation learning. It is only when context-isolated is combined with semantic alignment that the encoder output is effectively decoupled from the hidden state transformation in the flow-matching decoder, resulting in optimal average performance (fifth row).

Generation Targets. We use different generation targets for pretraining, including pixel, SigLIP2, Cosmos VAE, and VAVAE, with the results shown in Tab. 3. Experiments demonstrate that VAVAE achieves the best performance. In addition, direct pixel generation also shows competitive results. In contrast, Cosmos VAE performs poorly, possibly due to the less precise reconstruction of video VAEs.

Empirical Study. We use a simplified generative pretraining model (see Tab. 2, second row) to empirically investigate key factors in representation learning with generative models. Our results highlight the importance of challenging training objectives: difficult objectives lead to strong representations.

- Tab. 4 compares pretraining with and without masking.

Without masking, the high temporal redundancy in videos makes generation easier, resulting in lower-quality representations. Introducing masking to occlude information increases the difficulty of next-frame prediction, enabling the training of more robust semantic representations.

- Tab. 5 further supports this finding. We implement flow-

matching decoders based on cross-self attention layers and cross-attention layers. In cross-self attention, the noisy VAE targets can self-attend, while in cross-attention, they can only attend to encoder output. Since cross-self attention reduces generation difficulty, it also diminishes representation quality, which further supports the frame-isolated mask.

- Table 2. Ablation study of model designs. “Gen.” denotes the flow-matching generative objective, “Align.” indicates the feature alignment regularization, and “C-Iso.” refers to context-isolated prediction. While applying context isolation alone does not improve performance, our full design achieves optimal results.

Gen. Align. C-Iso. IN1K SSv2 Avg.

✓ 76.8 61.1 68.9 ✓ ✓ 75.0 58.7 66.9

✓ 74.7 62.8 68.8 ✓ ✓ 73.1 61.9 67.5

✓ ✓ ✓ 75.1 63.8 69.5

- Table 3. Ablation study of different generation targets. Generating VAVAE latent features, pixels or Siglip representations produces better results, while Cosmos VAE leads to inferior performance.

IN1K SSv2 Avg.

Pixel 73.5 62.4 68.0 Siglip2 [49] 73.2 62.5 67.9 Cosmos VAE [2] 71.7 54.8 63.3 VAVAE [60] 75.1 63.8 69.5

- Table 4. Empirical study of video masking. Due to the high temporal redundancy in videos, masking strategies are essential for video representation learning, significantly improving semantic quality.

IN1K SSv2 Avg.

w/o masking 31.0 14.7 22.9 with masking 75.0 58.7 66.9

- Table 5. Empirical study of information isolation in denoising. Disabling self-attention helps prevent information leakage between noisy targets, enhancing semantic representation quality.

IN1K SSv2 Avg.

Cross-Self Attention 72.6 54.2 63.4 Cross Attention 75.0 58.7 66.9

###### 5. Conclusion

In this paper, we present NExT-Vid, a novel autoregressive flow-matching visual pretraining method based on masked next-frame prediction. Our context-isolated design features two main components, the autoregressive predictor and the flow decoder, which decouple representation prediction from target decoding, enabling full utilization of generative objectives and precise semantic localization. Compared to other generative pretraining methods, our approach achieves state-of-the-art results on multiple benchmarks.

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1, 2

- [2] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 5, 8
- [3] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. 2, 6
- [4] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Selfsupervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 2, 5, 6, 1
- [5] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 1, 2, 6
- [6] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024. 2, 6
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 2
- [8] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 3
- [9] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020. 1, 2, 6
- [10] Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. Context autoencoder for selfsupervised representation learning. International Journal of Computer Vision, 132(1):208–223, 2024. 2, 4, 6
- [11] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2818–2829, 2023. 6

- [12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5, 1, 2, 3
- [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171– 4186, 2019. 1, 2
- [14] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pretraining of large autoregressive image models. arXiv preprint arXiv:2401.08541, 2024. 1
- [15] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 3
- [16] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2

- [17] Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Omnimae: Single model masked pretraining on images and videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10406–10417,

2023. 6

- [18] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 2, 5, 1, 3
- [19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020. 5, 4
- [20] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 1, 2, 6
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [22] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 2
- [23] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for

- text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [24] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 3

- [25] Yang Jin, Zhicheng Sun, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, et al. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. arXiv preprint arXiv:2402.03161, 2024. 3
- [26] Gagan Kanojia, Sudhakar Kumawat, and Shanmuganathan Raman. Attentive spatio-temporal representation learning for diving classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019. 2, 5, 3
- [27] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950,

2017. 2, 5, 1, 3

- [28] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2, 5
- [29] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 3
- [30] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62,

2022. 2

- [31] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 2
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 2, 4

- [33] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 3
- [34] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 2
- [36] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [37] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

- Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1, 2, 6
- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2, 5, 1

- [39] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 1, 2
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1, 2
- [41] Jathushan Rajasegaran, Ilija Radosavovic, Rahul Ravishankar, Yossi Gandelsman, Christoph Feichtenhofer, and Jitendra Malik. An empirical study of autoregressive pretraining from videos. arXiv preprint arXiv:2501.05453,

2025. 2, 4, 6

- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 5
- [43] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015. 2
- [44] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 2
- [45] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [46] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 5, 1

- [47] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 3
- [48] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 1, 2, 6
- [49] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 2, 5, 6, 8

- [50] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024. 3
- [51] Th´eophane Vallaeys, Jakob Verbeek, and Matthieu Cord. Ssdd: Single-step diffusion decoder for efficient image tokenization. arXiv preprint arXiv:2510.04961, 2025. 4, 5
- [52] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14549–14560, 2023. 1, 2, 5, 6, 7
- [53] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, and Lu Yuan. Bevt: Bert pretraining of video transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14733–14743, 2022. 1, 2
- [54] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, and Yu-Gang Jiang. Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6312–6322, 2023. 6
- [55] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 2
- [56] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 5, 1
- [57] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024. 2, 6
- [58] Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, and Christoph Feichtenhofer. Diffusion models as masked autoencoders. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16284– 16294, 2023. 2
- [59] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3
- [60] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025. 5, 8
- [61] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 3

- [62] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 2
- [63] Long Zhao, Nitesh B Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, Jennifer J Sun, Luke Friedman, Rui Qian, Tobias Weyand, Yue Zhao, et al. Videoprism: A foundational visual encoder for video understanding. arXiv preprint arXiv:2402.13217, 2024. 6

##### Learning from Next-Frame Prediction: Autoregressive Video Modeling Encodes Effective Representations

###### Supplementary Material

Table 6. Pretraining dataset details. We construct a large-scale mixed dataset comprising both images and videos with different sampling weights for balance pretraining.

Source Type Samples Hours weight

ImageNet-1K [12] Image 128M - 0.25 Sth-Sth-V2 [18] Video 169K 168 0.05 Kinetics-400 [27] Video 241K 669 0.18 InternVid [56] Video 234M 760K 0.2 Self-collected Videos Video 115M 958K 0.32

###### A. Pretraining Details

###### A.1. Datasets

Dataset Composition. To enable large-scale visual pretraining, we build a hybrid dataset by combining multiple image and video sources, as shown in Tab. 6. The image portion is from ImageNet-1K [12], with 1.28M images across 1,000 everyday object and scene categories. These carefully curated and cropped images provide rich semantics and textures, which are important for image pretraining.The video portion comes from several datasets. Something-Something-V2 [18] contains 168K short clips focused on hand-centric human actions from a largely firstperson perspective. Its simple backgrounds and clear motion patterns emphasize temporal dynamics, which is key for learning action understanding. Kinetics-400 [27], curated from YouTube, includes 400 categories of everyday actions in diverse scenes, helping the model learn complex spatiotemporal patterns. In addition, we use two largescale “in-the-wild” video datasets, InternVid [56] and selfcollected videos, sampled from popular E-commerce websites. Together, they broadly represent contemporary online video formats and content, covering most of what people watch in daily life and helping the model build a more comprehensive understanding of the visual world.

Dataset Sampling. Previous studies [4] show that sampling data with different weights can mitigate the impact of data imbalance and enable models to better exploit training data. We sample from each source according to the weights in Tab. 6. ImageNet-1K contributes 0.25 of the total samples, which is helpful for the model’s understanding of texture and appearance. Among the video datasets, TTVideo and InternVid have the largest sampling weights and serve as the primary sources for video understanding, while Kinetics-400 and Something-Something-V2 are included mainly to strengthen action recognition.

Table 7. Model Architectures. We pretrain the ViT-L, ViT-H, and ViT-G encoders, which share the same autoregressive predictor and flow-matching decoder configurations.

ViT-L ViT-H ViT-G

Encoder Architecture Block ViT Block Depth 24 32 40 Width 1024 1280 1408 Num Heads 16 16 22 MLP Ratio 4 4 4.36 Parameters 300M 600M 1.1B Patch Size 16×16 Tubelet Size 2 Norm LayerNorm Position Embedding 3D RoPE [46] Attention Mask Frame-Wise Causal

Autoregressive Predictor Architecture Block Cross Attention Block Depth 12 Width 384 Num Heads 12 Norm LayerNorm Position Embedding 3D RoPE [46] Attention Mask Autoregressive Query Tokens Init Zero Init Parameters 22M

Flow-Matching Decoder Architecture Block DiT Block Depth 12 Width 384 Num Heads 12 Norm LayerNorm Position Embedding 3D RoPE [46] Attention Mask Frame-Isolated Timestep Condition AdaLN [38] Parameters 33M

###### A.2. Models

The detailed model architectures are summarized in Tab. 7. We train three variants (ViT-L, ViT-H, and ViT-G, with 300M, 600M, and 1.1B parameters respectively). These models differ mainly in terms of depth and width, while all other settings are shared. Our model largely follows the standard ViT design, with two key differences: 3D RoPE [46] and frame-wise causal masks(see Fig. 3). For pretraining, all three variants use the same autoregressive predictor and flow-matching decoder, with 22M and 33M

Figure 6. Loss curve of the pretraining process. The training stages highlighted with different background colors correspond to Tab. 8.

parameters respectively. Notably, despite being small compared with the large encoder, these two components still produce high-quality images and videos while preserving rich semantic.

###### A.3. Training Strategies

Through extensive experimentation, we designed a fourstage pretraining schedule summarized in Tab. 8, with the loss curve shown in Fig. 6.

- 1) Warm-up stage (12K steps). The model starts with a

small learning rate that is gradually increased. In this stage, it primarily builds basic patterns and stable representations, which makes the autoregressive predictions progressively more challenging. Besides, the flow-matching decoder converges quickly. As shown in Fig. 6, the alignment loss drops sharply at first, then slowly rises to its peak, while the flow loss rapidly decreases and then flattens.

- 2) Stable stage 1 (28K steps). The learning rate is grad-

ually decayed from its peak. The model enters a representation searching stage. The flow-matching decoder updates steadily, and the autoregressive predictor progressively align with the reference representations.

- 3) Stable stage 2 (80K steps): The learning rate is further

reduced, but the flow-matching decoder is assigned a separate and fixed learning rate. Meanwhile, the τ-sampling rate is changed from 4 to 1. During this stage, the model enters a non-stationary period, requiring careful updates. We find that keeping a large, fixed learning rate for the flow-matching decoder and using single-step τ-sampling improves robustness and leads to stable updates throughout this stage.

- 4) Cool-down stage (12K steps). Finally, we finetune the

model with a smaller learning rate and increase the input to 64 video frames. This stage aims to consolidate semantic representations and improve understanding of longer

videos. After cooling, the model shows markedly stronger video semantic understanding.

###### B. Evaluation Details

###### B.1. Benchmarks

We conduct attentive probing on four benchmarks, summarized in Tab. 9. ImageNet-1K [12] comprises 1.28M training and 50K validation images across 1000 everyday object categories, evaluating the model’s understanding of appearance and texture. Kinetics-400 [27] includes 241K training and 20K validation video clips over 400 action categories, featuring complex scenes and diverse actions to evaluate spatial-temporal comprehension. Something-SomethingV2 [18] contains 169K training and 24K validation clips across 174 egocentric action categories, and Diving-48 [26] provides 15K training and 2K validation clips over 48 diving categories. These two benchmarks primarily evaluate the model’s understanding of motion.

###### B.2. Evaluation Strategies

We use a single-layer attentive probe for visual classification, with the classifier and training settings detailed in Tab. 10. The probe aggregates encoder visual features via a cross-attention layer and a learnable query token, and the resulting representation is fed into a single-layer linear classifier. During probe training, the encoder is frozen and only the probe parameters are updated. For each dataset, we adopt different data hyperparameters, training the attentive probe for multiple epochs on the training set and evaluating on the validation set.

- Table 8. Multi-stage pretraining strategies. Our pre-training process has four stages. 1) Warm-up stage begins with a low learning rate. 2) Stable stage 1 and stable stage 2 stabilize training by careful designs of learning rate and τ sample. 3) Cool-down stage finetunes the model at high frames.

WarmUp Stable1 Stable2 CoolDown

Data Hyperparameters Frames 16 16 16 64 FPS 4 Resolution 256 Normalization [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

Optimization Hyperparameters Steps 12K 28K 80K 12K Start LR 1e-4 5e-4 4.5e-4 1e-4 Final LR 5e-4 4.5e-4 1e-4 1e-6 Flow LR - - 8e-4 3e-4 Batch Size 3072 3072 3072 768 Grad Clipping 1.0 Betas [0.9, 0.95] Weight Decay 0.04 Flow Weight 0.5 Align Weight 1.0 EMA 0.99925

Flow-Matching Hyperparameters τ sample 4 4 1 1 Timesteps 1000 Mode Random

Mask Hyperparameters Spatial Scale [0.15, 0.7] Temporal Scale [1.0, 1.0] Aspect Ratio 0.75∼1.5 Mask Sample [8, 2]

- Table 9. Evaluation benchmark details. We perform attentive probe evaluations on four benchmarks by visual classification.

Benchmarks Type Train Val Classes ImageNet-1K [12] Image 1.28M 50K 1000 Kinetics-400 [27] Video 241K 20K 400 Sth-Sth-V2 [18] Video 169K 24K 174 Diving-48 [26] Video 15K 2K 48

###### C. Additional Results

###### C.1. Ablation Studies

This section presents further ablations on several designs for pretraining stability, which help inform the four-stage training strategy described in Sec. A.3.

Timestep Sampling To fully train the small flow-matching decoder, we follow MAR [33] and sample τ for k times of each sample, effectively increasing the decoder’s batch size by a factor of k. However, we find that this strategy

[Figure 84]

Figure 7. t-SNE visualization of ImageNet validation set representations. Different colors indicate different image categories.

Table 10. Attentive probe strategies. We train a single-layer attentive probe classifier on four benchmarks.

K400 IN1K SSv2 Diving48

Data Hyperparameters Segments 8 - 2 4 Views 3 - 3 3 Frames 16 16 16 32 FPS 4 - 4 2 Resolution 256 Normalization [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

Classifier and Training Hyperparameters Batch Size 192 384 96 48 Epochs 20 20 20 100 Encoder Layers 1 1 1 4 Classifier Heads 16 Classifier Depths 1

introduces instability in the later stages of pretraining, so we conduct the ablations reported in Tab. 11. Specifically, we compare single-fold sampling with 4-fold sampling. At 15K steps, 4-fold sampling performs worse than single-fold sampling, whereas at 30K steps, 4-fold sampling outperforms single-fold sampling. This suggests that although extensive generative training initially impedes semantic learning, it ultimately offers greater potential. Therefore, we do not discard this design solely due to its instability. Instead, as shown in Tab. 8, we adopt the Stable-1 and Stable2 stages: we first fully exploit the model’s potential with multi-fold τ sampling, and then switch to single-fold τ sam-

- Table 11. Ablation study of τ sample. Although extensive flowmatching training by multi-fold τ sample may slow semantic learning, it has better representational potential.

IN1K SSv2 Avg. Training 15K Steps

τ sample = 1 75.3 63.8 69.6 τ sample = 4 75.1 63.8 69.5

Training 30K Steps

τ sample = 1 79.3 67.1 73.2 τ sample = 4 79.4 67.7 73.6

- Table 12. Ablation study of momentum reference encoder. Using momentum reference model greatly stabilizes training.

IN1K SSv2 Avg.

w/o EMA < 10 < 10 with EMA 75.1 63.8 69.5

tation typically requires challenging targets, which makes it difficult to simultaneously train a high-quality generative model. Future work will continue exploring these aspects.

pling for more stable later-stage training.

Momentum Reference Encoder Another crucial design for stabilizing training is the momentum reference model with EMA (Exponential Moving Average) [19] updating. As shown in Tab. 12, without EMA, there are large fluctuations in the pretraining loss, which are often followed by crashes that completely degrade the performance of the model. By contrast, the slower, smoothed updates provided by EMA make the reference features more stable, thereby preventing such crashes in the early stages of pretraining.

###### C.2. Visualization

t-SNE Decomposition To intuitively visualize the encoder’s feature space, we extract representations of the 50k images in the ImageNet validation set. We then apply average pooling and perform t-SNE for dimensionality reduction, and plot the results in a 2D plane. We use different colors to denote images from different classes. As shown in Fig. 7, we observe that features from the same class are well clustered, while features from different classes are clearly separable. Due to the semantic continuity of ImageNet classes, clusters of semantically related classes tend to lie close to each other. This demonstrates the high quality of the semantic features extracted by our model.

###### D. Limitations

We would like to discuss the limitations of our approach. First, although our method performs autoregressive pretraining to model temporal semantics, it still relies on masking and thus cannot fully exploit the efficiency of GPT-style pretraining. Second, our method faces an inherent trade-off between generation and representation. Effective represen-

