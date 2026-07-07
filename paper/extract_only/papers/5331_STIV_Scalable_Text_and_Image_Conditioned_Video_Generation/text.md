# arXiv:2412.07730v2[cs.CV]6Oct2025

### STIV: Scalable Text and Image Conditioned Video Generation

Zongyu Lin1⋆* Wei Liu1⋆ Chen Chen2⋆ Jiasen Lu2⋆ Wenze Hu2⋆ Tsu-Jui Fu2⋆ Jesse Allardice2⋆ Zhengfeng Lai2⋆ Liangchen Song2⋆ Bowen Zhang2⋆ Cha Chen2⋆ Yiran Fei⋆ Lezhi Li⋆ Yizhou Sun⋄† Kai-Wei Chang⋄† Yinfei Yang⋄⋆ ⋆Apple †University of California, Los Angeles

#### Abstract

The field of video generation has made remarkable advancements, yet there remains a pressing need for a clear, systematic recipe that can guide the development of robust and scalable models. In this work, we present a comprehensive study that systematically explores the interplay of model architectures, training recipes, and data curation strategies, culminating in a simple and scalable text-image-conditioned video generation method, named STIV. Our framework integrates image condition into a Diffusion Transformer (DiT) through frame replacement, while incorporating text conditioning via a joint image-text conditional classifier-free guidance. This design enables STIV to perform both text-to-video (T2V) and text-image-to-video (TI2V) tasks simultaneously. Additionally, STIV can be easily extended to various applications, such as video prediction, frame interpolation, multi-view generation, and long video generation, etc. With comprehensive ablation studies on T2I, T2V, and TI2V, STIV demonstrate strong performance, despite its simple design. An 8.7B model with 5122 resolution achieves 83.1 on VBench T2V, surpassing both leading open and closed-source models like CogVideoX-5B, Pika, Kling, and Gen-3. The same-sized model also achieves a state-of-the-art result of 90.1 on VBench I2V task at 5122 resolution. By providing a transparent and extensible recipe for building cutting-edge video generation models, we aim to empower future research and accelerate progress toward more versatile and reliable video generation solutions.

#### 1. Introduction

The field of video generation has witnessed a significant progress with the introduction of Sora [42], a video generation model based on Diffusion Transformer (DiT) [43] architecture. Researchers have been actively exploring optimal methods to incorporate text and other conditions into the DiT architecture. For example, PixArt-α [8] leverages cross attention, while SD3 [19] concatenates text with the noised patches and applies self-attention using the MMDiT block. Several video generation models [21, 46, 65] adopt similar approaches and have made substantial progress in the text-to-video (T2V) task. Pure T2V approaches often struggle with producing coherent and realistic videos, as their outputs are not grounded in external references or contextual constraints [13]. To address this limitation, text-image-to-video (TI2V) introduce an initial image frame along with the textual prompt, providing a more concrete grounding for the generated video.

*This work was done during an internship at Apple. 1First authors 2Core authors ⋄Senior authors

Despite substantial progress in video generation, achieving Sora-level performance for T2V and TI2V remains challenging. A central challenge is how to seamlessly integrate image-based conditions into the DiT architecture, calling for innovative techniques blend visual inputs smoothly with textual cues. Meanwhile, there is a pressing need for stable, efficient large-scale training strategies, as well as improving the overall quality of training datasets. To address these issues, a comprehensive, step-by-step “recipe” would greatly assist in developing unified models that handle both T2V and TI2V task under one framework. Overcoming these challenges is essential for advancing the field and fully realizing the potential of video generation models.

[Figure 1]

Figure 1. Performance comparison of our Text-to-Video model against both open-source and closed-source state-ofthe-art models on VBench [31].

Although various studies [2, 6, 11, 14, 49, 62, 70] have examined methods of integrating image conditions into the U-Net architectures, how to effectively incorporate such conditions into the DiT architecture remains unsolved. Moreover, existing studies in video generation often focuses on individual aspects independently, overlooking the how their collective impact on overall performance. For instance, while stability tricks like QK-norm [19, 28] have been introduced, they prove insufficient as models scale to larger sizes [57], and no existing approach has successfully unified T2V and TI2V capabilities within a single model. This lack of systematic, holistic research limits progress toward more efficient and versatile video generation solutions.

In this work, we first present a comprehensive study of model architectures and training strategies to establish a robust foundation for T2V. Our analysis reveals three key insights: (1) stability techniques such as QK-norm and sandwich-norm [17, 25] are critical for effectively scaling larger video generation models; (2) employing factorized spatial-temporal attention [1], MaskDiT [73], and switching to AdaFactor [54] significantly improve training efficiency and reduce memory usage with minimal impact on performance loss; (3) progressive training, where spatial and temporal layers are initialized from separate models, outperforms using a single model under the same compute constraints. Starting from a PixArt-α baseline architecture, we address scaling challenges with these stability and efficiency measures, and further enhance performance with Flow Matching [41], RoPE [56], and micro conditions [45]. As a result, our largest T2V model (8.7B parameters) achieves state-of-the-art semantic alignment and a VBench score of 83.1.

We then identify the optimal model architecture and hyperparameters established in the T2V setting and apply them to the TI2V task. Our results show that simply replacing the first noised latent frame with the un-noised image condition latent yields strong performance. Although ConsistI2V [49] introduced a similar idea in a U-Net setting, it required spatial self-attention for each frame and window-based temporal self-attention to match our quality. In contrast, the DiT architecture natively propagates the image-conditioned first frame through stacked spatial-temporal attention layers, eliminating the need for these additional operations. However, as we scale up spatial resolution, we observe the model producing slow or nearly static motion. To solve this, we introduce random dropout of the image condition during training and apply joint image-text conditional classifier-free guidance (JIT-CFG) for both text and image conditions during inference. This strategy resolves the motion issue and also enables a single model to excel at both T2V and TI2V tasks.

With all these changes, we finalize our model and scale it up from 600M to 8.7B parameters. Our best STIV model achieves a state-of-the-art result of 90.1 in the VBench I2V task at 5122 resolution. Beyond enhancing video generation quality, we demonstrate the potential of extending our framework to various downstream applications, including video prediction, frame interpolation, multi-view generation and long video generation. These results

|Text-to-Video|
|---|

Prompt: An adorable kangaroo wearing blue jeans and a white t shirt taking a pleasant stroll in Johannesburg South Africa during a beautiful sunset.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Prompt: A swan with wings tipped in gold gliding across a misty lake, leaving a trail of soft, shimmering light that fades as the sun rises.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|Text-Image-to-Video|
|---|

Prompt: The video presents a sequence of frames that depict a space scene with a large, green and yellow planet at the center, surrounded by smaller celestial bodies. The background is a deep blue, speckled with stars.

|[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Prompt: Robots move efficiently through a futuristic laboratory, adjusting holographic displays and conducting experiments, while scientists observe and interact with the high-tech equipment.

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 2. Text-to-Video and Text-Image-to-Video generation samples by T2V and STIV models. The text prompts and first frame image conditions are borrowed from Sora’s demos [42] and MovieGenBench [46].

[Figure 20]

Scaled Dot-Product Attention

Timestep Projection norm Projection norm

[Figure 21]

[Figure 22]

Shared AdaLN

[Figure 23]

+

Micro Conditions

|Text Projection norm| |
|---|---|
| | |

Pooled T Embedding

RMSNorm

RMSNorm

Linear

Linear

Linear

[Figure 24]

V

K

Q

VAE Enc

Scale & Shift Gate

Gate

Scale & Shift

Frame Replacement + MHA with QK-Norm Image Condition Dropout

[Figure 25]

[Figure 26]

[Figure 27]

x N

[Figure 28]

[Figure 29]

Cross Attention

Spatial Attention MHA with QK-norm

Temporal Attention

norm

norm

norm

norm

norm

norm

+

FFN

+ +

+

MHA With QK-norm

MHA With QK-norm

|A pirate ship sailing through a thunderstorm with enormous waves.| |
|---|---|
| | |

[Figure 30]

CLIPText Enc

STIV Block

Sequence Text Embedding

- Figure 3. We replace the first frame of the noised video latents with the ground truth latent and randomly drop out the image condition. We use cross attention to incorporate the text embedding, and use QK-norm in multi-head attention, the sandwich-norm in both attention and feedforward, and stateless layernorm after singleton conditions to stabilize the training.

validate the scalability and versatility of our approach, showcasing its ability to address diverse video generation challenges. We summarize our contributions as follows:

- • We present STIV, a single model capable of performing both T2V and TI2V tasks. At its core, we replace the noised latent with the un-noised image condition latent and introduce joint image-text conditioned CFG.
- • We conduct a systematic study for T2I, T2V and TI2V, covering model architectures, efficient and stable training techniques, and progressive training recipes to scale up the model size, spatial resolution, and duration.
- • These design features make it easy to train and adaptable to various tasks, including video prediction, frame interpolation, and long video generation.
- • Our experiments include detailed ablation studies on different design choices and hyperparameters, evaluated on VBench, VBench-I2V and MSRVTT. These studies demonstrate the effectiveness of the proposed model compared with a range of recent state-of-the-art open-source and closed-source video generation models. Some of the generated videos are shown in Fig. 2. More examples can be found in the Sec. K in the Appendix.

#### 2. Basics for STIV

This section describes our key components of our proposed STIV method for text-image-to-video (TI2V) generation, which is illustrated in Fig. 3. Afterward, Sec. 3 and 4 presents detailed experimental results.

##### 2.1. Base Model Architecture

The STIV model is based on PixArt-α [8], which converts the input frames into spatial and temporal latent embeddings using a frozen Variational Autoencoder (VAE). These embeddings are then processed by a stack of learnable DiT-like blocks. We employ the T5 [48] tokenizer and an internally trained CLIP [47] text encoder to process text prompts. The overall framework is illustrated in Fig. 3. For more details, please refer to the appendix. The other significant architectural changes are outlined below.

Spatial-Temporal Attention We employ factorized spatial and temporal attention [1] to handle video frames. We first fold the temporal dimension into the batch dimension and perform spatial self-attention on spatial tokens. Then, we permute the outputs and fold the spatial dimension into the batch dimension to perform temporal

self-attention on temporal tokens. By using factorized spatial and temporal attention, we can easily preload weights from a text-to-image (T2I) model, as images are a special case of videos with only one temporal token and only need spatial attention.

Singleton Condition We use the original image resolution, crop coordinates, sampling stride, and number of frames as micro conditions to encode the meta information of the training data. We first use a sinusoidal embedding layer to encode these properties, followed by an MLP to project them into a d-dimensional embedding space. These micro condition embeddings, along with the diffusion timestep embedding and the last text token embedding from the last layer of the CLIP model, are added to form a singleton condition. We also apply stateless layer normalization to each singleton embedding and then add them together. This singleton condition is used to produce shared scale-shift-gate parameters that are utilized in the spatial attention and feed-forward layers of each Transformer layer.

Rotary Positional Embedding Rotary Positional Embeddings (RoPE) [56] are used so that the model has a strong inductive bias for processing relative temporal and spatial relationships. Additionally, RoPE can be made compatible with the masking methods used in high compute applications and are highly adaptable to variations in resolution [76]. We apply 2D RoPE [39] for the spatial attention and 1D RoPE for the temporal attention inside the factorized Spatial-Temporal attention.

Flow Matching Instead of employing the conventional diffusion loss, we opt for a Flow Matching training objective. This objective defines a conditional optimal transport between two examples drawn from a source and target distribution. In our case, we assume the source distribution to be Gaussian and utilize linear interpolates [41] to achieve this.

xt = t · x1 + (1 − t) · ϵ. (1) The training objective is then formulated as

Ex,ϵ∈N(0,I),c,t ∥Fθ(xt,c,t) − vt∥22 (2) where the velocity vector field vt = x1 − ϵ.

min

θ

In inference time, we solve the corresponding reverse-time SDE, from timestep 0 to 1, to generate images from a randomly sampled Gaussian noise ϵ.

##### 2.2. Model Scaling

As we scale up the model, we encounter training instability and infrastructure challenges in fitting larger models into memory. In this section, we outline the methods to stabilize the training and enhance training efficiency.

Stable Training Recipes We discovered that QK-Norm — applying RMSNorm [68] to the query and key vectors prior to computing attention logits — significantly stabilizes training. This finding aligns with the results reported in SD3 [19]. We also change from pre-norm to sandwich-norm [17] for both MHA and FFN, which involves adding pre-norm and post-norm with stateless layer normalization [37] to both the layers within the STIV block.

MHA(x) = x + gate · norm(Attn(scale · norm(x) + shift)) FFN(x) = x + gate · norm(MLP(scale · norm(x) + shift))

Efficient DiT Training We follow MaskDiT [73] by randomly masking 50% of spatial tokens before passing

- them into the major DiT blocks. After unmasking, we add two additional DiT blocks. We also switch from AdamW to AdaFactor optimizer and employ gradient checkpointing to only store the self-attention outputs. These modifications significantly enhance efficiency and reduce memory consumption, enabling the training of larger models at higher resolution and longer duration.

##### 2.3. Image Conditioning

- 2.3.1 Frame Replacement

During training, we replace the noised first frame latent with the un-noised latent of the image condition before passing the latents into the STIV blocks, and masking out the loss of the replaced frame. During inference, we use the un-noised latent of the original image condition for the first frame at each TI2V diffusion step.

The frame replacement strategy offers flexibility in extending STIV to various applications. For instance, if cI = ∅, it defaults to text-to-video (T2V) generation. Conversely, if cI is the initial frame, it becomes the typical text-image-to-video (TI2V) generation. Moreover, if multiple frames as cI are provided, they can be used for video prediction even without cT. Additionally, supplying the first and last frames as cI enables the model to learn a frame interpolation, generating frames between them. Furthermore, combining T2V and frame interpolation allows for the generation of long-duration videos: T2V generates keyframes, and frame interpolation frames then fills in frames between each pair of consecutive keyframes. Ultimately, a single model can be trained to perform all tasks by randomly selecting the appropriate conditioning strategy.

- 2.3.2 Image Condition Dropout

As discussed previously, the frame replacement strategy offers substantial flexibility for training various types of models. Here, we demonstrate a specific application in which we train a model to perform both T2V and TI2V tasks. In this case, we randomly drop out cI and cT during training, similar to how T2V models employ random dropout to text condition alone.

Classifier-free guidance (CFG), commonly used in text-to-image generation, has proven to be highly beneficial in enhancing the quality of generated images by directing the probability mass toward the high-likelihood regions given the condition. Building on this concept, we introduce a Joint Image-Text Classifier-Free Guidance (JIT-CFG) approach, which leverages both text and image conditions. It modifies the velocity estimates as

Fˆθ(xt,cT,cI,t) = Fθ(xt,∅,∅,t)

+ s · (Fθ(xt,cT,cI,t) − Fθ(xt,∅,∅,t))

where s is the guidance scale. When cI = ∅, it reduces to standard CFG for T2V generation. Although it is possible to introduce two separate guidance scales, as done in [4], to balance the strength of the image and text conditions, we found that our two-pass approach yields strong results. Additionally, using two scales would require three forward passes, increasing the inference cost.

Empirical observations 3.4.2 suggest that applying image condition dropout with JIT-CFG effectively not only achieves multi-task training in a natural way, but also resolves the staleness issue for a 5122 STIV model. We hypothesize that image condition dropout prevents the model from passively overfitting to the image condition, allowing it to more effectively capture the motion information from the underlying video training data.

##### 2.4. Progressive Training Recipe

We employ a progressive training recipe as illustrated in Figure 4. The process begins by training a text-to-image (T2I) model, which serves to initialize a text-to-video (T2V) model. Next, the T2V model serves as the starting point for initializing the STIV model. To facilitate rapid adaptation to higher resolutions and longer durations training, we incorporate interpolated RoPE embeddings in both the spatial and temporal dimensions, while initializing the model weights using those from the lower-resolution, shorter-duration models.

Low-resolution T2I

Low-resolution T2V

Low-resolution STIV

High-resolution T2I

High-resolution T2V

High-resolution STIV

full init. init. of temporal attention

full init. except for temporal attention

- Figure 4. Progressive training pipeline of the STIV model. The T2I model is first trained to initialize the T2V model, which

- then initializes the STIV model at both low and high resolutions. Notably, the high-res T2V model is initialized using both the high-res T2I model and the low-res T2V model.

###### Architecture Ablation

- T2V (+ Temporal):
- – Spatial Temporal Attention
- – Temporal scale_shift_gate
- – Temporal Masking
- – …

- TI2V (+ Image condition):
- – Frame Replacement
- – Image conditioning dropout
- – Joint Image-Text CFG
- – …

###### T2I (Base Model):

- – QK-Norm
- – Sandwidth Norm
- – MaskDiT (Spatial Masking)
- – …

Figure 5. Ablation study of the STIV model, from the base T2I model to the temporally-aware T2V model, and finally to the image-conditioned TI2V model.

###### Video Data Engine

###### Pre-processing:

- Per Segment Filtering:
- – Motion Score
- – Aesthetic Score
- – Video Orientation
- – Resolution …

- Captioning:
- – Richness
- – Accuracy
- – Hallucination
- – …

- Analysis:
- – Video Duration
- – Category Distribution
- – Temporal Dynamics
- – …

- – Segment Video
- – Remove Insistent Segments
- – …

#### 3. Recipe Study for STIV

##### 3.1. Basic Setup

Before we dive into the studies of architecture and data for video generation models, we first introduce the training, data and evaluation setup before introducing our model and studies as follows:

Training Unless otherwise specified, we use the AdaFactor optimizer (β1 = 0.9, β2 = 0.999) [54] without any weight decay. We also clip the gradient norm if the gradient norm exceeds 1.0. We use a constant learning rate schedule with a 1k step linear warmup with a maximum learning rate of 2 × 10−4. For T2I models, we train each model for 400k steps with a batch size of 4,096. This is approximately 1.4 epochs on our internal T2I datasets. For T2V and TI2V models, we train each model for 400k steps with a batch size of 1,024. This is roughly 5.5 epochs on our internal video datasets. For all models, exponential moving average weights are gathered by a decay rate of 0.9999 and are then used for evaluation. When MaskDiT is used, we train with 50% spatial random masking during the initial 400k steps. Subsequently, we perform unmasked fine-tuning using all tokens. We use 50k steps of unmasked fine-tuning for T2I models and 100k steps for T2V and TI2V models.

Data We build a video data engine pipeline that includes video pre-processing, captioning, and filtering to accelerate the model’s development when handling large-scale videos is required. Specifically, we apply PySceneDetect 1 to analyze video frames, detect and segment scenes based on abrupt transitions and gradual fades. This segmentation is followed by the feature extractions for filtering, including motion score, aesthetic score, text area, frame

1https://github.com/Breakthrough/PySceneDetect

dimensions, clarity score, temporal consistency, and video orientation, among others. For each video segment, we perform dense captioning and categorization to gain a comprehensive understanding of video distribution. To further enhance caption quality, we adapt DSG [12] and propose DSG-Video, a metric designed to assess hallucination rates and overall caption quality. This data engine is integral in filtering videos and preparing tailored datasets for different training stages: our data sources include Panda-70M [10] and an internally curated high-quality dataset of 42M videos. Using our data engine, we curate over 90M high-quality video-caption pairs 2. Next we are going to dive into more fine-grained modulation studies. As illustrated in figure 5, We follow the principle of studying from base T2I model to the temporally-aware T2V model, and finally to the TI2V model by adding image conditioning.

Evaluation We mainly use VBench [31], VBench-I2V and MSRVTT [63] to evaluate T2V and TI2V models. For VBench, we mainly report Quality (temporal quality and frame-wise quality), Semantic (semantic alignment with different perspectives of the input text prompt) and Total score (weighted average of Quality and Semantic), and they can be actually decomposed into 16 dimensions in total. VBench-I2V builds upon the VBench with three new Video-Image Alignment metrics: Subject Consistency, Background Consistency, and Camera Motion Control. These additional metrics provide a more comprehensive evaluation by focusing on how well the generated video aligns with the input image and specified prompt instructions. More details about the detailed dimensions are presented in Section F.1. We present three model scales: XL, XXL, and M with their configuration detailed in Tab. 1. In the following section, we use the notation X-S-T to represent an X-size model with an S2 resolution and T frames. If unspecified, the default configuration is a 2562 resolution with 20 frames. More detailed model and training configurations are provided in the Appendix.

|Model Size|# of STIV Blocks|Hidden Dim<br><br>|# of Attn Heads|
|---|---|---|---|
|XL (600M) XXL (1.5B) M (8.7B)<br><br>|28 38 46|1,152 1,536 3,072<br><br>|18 24 48|

Table 1. Model Configurations

##### 3.2. Ablation Studies for Key Changes on T2I

We conduct a comprehensive ablation study to understand the impact of various model architecture designs and training strategies mentioned in Sec. 2 on the text-to-image generation task. To evaluate generation quality, we use a suite of popular automated metrics, including FID score [29], Pick Score [33], CLIP Score, GenEval [23], and DSGEval [12], Human Preference Score (HPSv2) [61], Image Reward [64].

We began with a base T2I-XL model, a DiT [43] model augmented with cross-attention layers to integrate with text embeddings. Initially, we applied a series of stabilization techniques, including QK-norm, sandwich-norm and singleton condition norm, which yielded comparable results to the baseline. Notably, these techniques enabled us to train models stably even with a learning rate increased from 1e-4 to 2e-4. We demonstrated that incorporating Flow Matching during training and employing CFG-Renormalization3 during inference improved all the metrics substantially. Subsequently, we explored techniques to reduce training memory, such as AdaFactor Optimizer, MaskDiT, and Shared AdaLN, which maintained similar performance. Utilizing micro conditions and RoPE further reduced the FID score and improved DSGEval and Image Reward. Finally, incorporating an internally trained bigG CLIP model improved on all metrics even more. Notably, combining synthetic recaption with original caption following [35] achieved the best results in almost all metrics. For more details, refer to the Appendix D.

We use the optimal model architecture and training hyperparameters based on the T2I ablation study as our starting point for the remaining T2V and TI2V experiments.

2The details and effectiveness of our data engine are studied in the Appendix. 3Detailed description in Appendix.

|Model<br><br>|COCO COCO COCO Gen DSG HPSv2 Image FID↓ PICK↑ CLIP↑ Eval↑ Eval↑ Eval↑ Reward↑|
|---|---|
|Baseline<br><br>+ QK norm<br><br>+ Sandwich norm<br><br>+ Cond. norm<br><br>+ LR to 2E-4<br><br>|26.17 20.91 32.03 0.358 0.571 26.33 -0.25 25.60 20.92 32.08 0.372 0.574 26.32 -0.22 25.76 20.97 32.13 0.366 0.577 26.32 -0.23<br><br>25.58 21.05 32.27 0.393 0.583 26.43 -0.22<br>26.35 21.03 32.28 0.379 0.586 26.40 -0.12<br>|
|+ Flow<br><br>+ Renorm<br><br>|24.96 21.45 32.90 0.457 0.639 26.95 0.15 21.16 21.46 32.93 0.471 0.668 27.27 0.32|
|+ AdaFactor<br><br>+ MaskDiT<br><br>+ Shared AdaLN<br><br>|20.26 21.47 32.97 0.474 0.661 27.26 0.32 23.85 21.51 33.07 0.499 0.663 27.28 0.30 22.83 21.44 33.12 0.496 0.658 27.27 0.24|
|+ Micro cond.<br><br>+ RoPE<br><br>|20.02 21.50 33.09 0.498 0.673 27.27 0.41 18.40 21.46 33.11 0.502 0.680 27.26 0.48|
|+ Internal VAE<br><br>+ Internal CLIP<br><br>+ Synth. captions|19.57 21.79 33.26 0.492 0.668 27.26 0.52<br><br>17.97 21.89 33.62 0.607 0.717 27.40 0.65<br><br>18.04 22.10 33.65 0.685 0.751 27.65 0.81<br>|

Table 2. Text-to-image model ablation studies.

##### 3.3. Ablation Studies on Key Designs for T2V

Key Modulation We make some design choices in our model based on the evaluations on VBench, as shown in Fig. 6a. The base model uses a temporal path size of 2, non-causal temporal attention, and a spatial masking ratio of 0.5. As expected, the model with temporal patch=1 performs the best, but it is only slightly better with 2x compute. However, the model with temporal patch=4 leads to a noticeable performance drop. Using causal temporal attention also results in a significant drop in both quality and total scores. Adding a scale-shift-gate to the temporal attention layer 4 is slightly worse than the baseline, despite having more parameters. Furthermore, removing the spatial masking results in a slight decrease in the Semantic score and an improvement in the Quality and Total scores. However, this comes at the cost of requiring more compute as the length of tokens are doubled. On the other hand, using temporal masking significantly degrades model performance, with large drops observed in the VBench quality and final scores.

Model Initialization We investigate how initialization impacts the performance of T2V-XL models. We train 5122 T2V models by four different paths under a controlled total FLOPs setting: from scratch, initializing from a lower resolution T2V-256 model, initializing from a T2I-512 model, and loading both the temporal and spatial weights from T2V-256 and T2I-512 models respectively (Fig. 6b). We find that jointly initializing from both a low resolution T2V model and a high resolution T2I model can achieve better VBench metrics. This joint initialization method yields slightly improved FVD values compared to training from scratch and offers benefits in terms of efficient experimentation and cost when low resolution models are already present. Under a similar methodology we additionally explore the effects of training T2V models with higher numbers of frames (40 frames) by initializing from shorter T2V models (20 frames). Fig. 6c shows that when training models with a higher number of frames initializing from a low frame count model achieves improved metrics over initializing directly from a T2I model. Using interpolation of the RoPE embeddings yields improved VBench scores compared to extrapolation. Additionally we find that initializing the high frame count training from a T2V model trained with a proportionally lower frame rate (higher frame sub-sampling stride) can improve the VBench metrics, particularly the motion smoothness and dynamic degree.

4See Fig. 3 for the diagram of the model.

|Module<br><br>|VBench| | |
|---|---|---|---|
| |Quality ↑<br><br>|Semantic ↑<br><br>|Total ↑|
|Base model<br><br>w/ temp. patch=1 w/ temp. patch=4 w/ causal temp._atten<br><br>+ temp. scale_shift_gate<br><br>+ temp. mask - spatial mask|80.19 80.92 79.72 74.59 80.32 77.58 80.57<br><br>|70.51 71.69 69.15 73.13 68.94 65.95 70.31<br><br>|78.25 79.07 77.61 74.30 78.04 75.25 78.52|

(a) Ablation Study of T2V model design using T2V-XL. The base model uses temporal path size 2, non-causal temporal attention, spatial masking ratio 0.5, and no temporal masking.

|Init.<br><br>|MSRVTT<br><br>|VBench| | |
|---|---|---|---|---|
| |FVD ↓<br><br>|Quality ↑|Semantic ↑|Total ↑<br><br>|
|Scratch T2V-256 T2I-512 Both|417.98 415.63 401.83 405.14<br><br>|80.27 80.28 79.77 80.45<br><br>|67.84 71.29 71.58 72.37|77.78 78.49 78.13 78.83<br><br>|

- (b) Different model initialization for T2V-XL-512.

|Init.<br><br>|MSRVTT<br><br>|VBench| | |
|---|---|---|---|---|
| |FVD ↓|Quality ↑<br><br>|Semantic ↑|Total ↑|
|T2I T2V (inter.) T2V (extra.) T2V 2x (inter.)|549.13 407.86 397.90 401.94<br><br>|78.71 79.56 79.18 79.59<br><br>|65.69 65.42 64.63 66.24<br><br>|76.10 76.73 76.27 76.92|

- (c) Different initialization for T2V-XL 40 frames.

Figure 6. Ablation studies of key designs for T2V.

##### 3.4. Ablation Studies on Key Designs for TI2V

To integrate the image condition with the text condition, we reformulate the model as Fθ(xt,cT,cI,t), where cT and cI are the text and image conditions. Then, we studied each design component in TI2V framework and tackled multi-task learning and staleness issue encountered when training high resolution TI2V models.

###### 3.4.1 The Effectiveness of Frame Replacement

We ablate several model variants for TI2V on STIV-XL model, by combining the following key components: Frame Replacement (FR), Cross Attention (CA), Large Projection (LP), and First Frame Loss (FFL) 5. As shown in Tab. 3, notably, adding a large projection layer enhances the information passed by the cross-attention, resulting in improvements in both subject and background consistency. However, this approach may overly constrain the model, as evidenced by a reduction in the dynamic degree score (22.36 for FR + CA + LP compared to 35.4 for FR + CA), indicating that the model might exert excessive control over the generated output. Additionally, adding a first-frame loss, though seemingly beneficial, has shown to reduce overall scores, particularly in aspects of motion quality, suggesting that this loss might inadvertently constrain the model’s temporal dynamics. In contrast, frame replacement alone has proven to be a robust and effective approach, yielding consistent improvements without negatively impacting other dimensions of video quality. The frame replacement (FR) model achieves high scores in I2V average scores (the average of I2V Subject, I2V Background and Camera Motion) and total average scores. These results underline the advantage of frame replacement as a foundational component, providing a stable backbone for maintaining quality across diverse dimensions.

###### 3.4.2 The Effectiveness of Image Condition Dropout

Our experiments show that image condition dropout with JIT-CFG not only supports multi-task training but also resolves staleness in a 5122 STIV model.

Multi-task training By using image-conditioning dropout during STIV training, we effectively enable both T2V and TI2V capability. As shown in Tab. 4, models trained exclusively on T2V or TI2V task alone cannot perform the other task, while STIV with image condition dropout can easily handles both two task well, achieving performance comparable to the best single-task models.

5(1) FR uses frame replacement alone for strong image-video alignment. (2) CA uses cross attention alone to align features between the input image and generated video. (3) FFL removes the first frame loss mask introduced in Section 3.4 to constrain the initial frame of the video. (4) LP employs a more powerful ResNet 2D encoder as the projection layer here.

|Models<br><br>|Subj Cons|Bg Cons<br><br>|Temp Flick|Mot Smooth<br><br>|Dyn Deg|Aesth Qual<br><br>|Img Qual<br><br>|I2V Subj<br><br>|I2V Bg<br><br>|Cam Mot|I2V Avg Scores<br><br>|Avg Scores|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|CA CA + FFL CA + LP FR FR + CA FR + CA + LP FR + CA + LP + FFL|82.2 84.5 95.2 94.5 95.1 95.3 95.2<br><br>|92.8 95.6 98.7 98.3 98.6 98.5 98.7<br><br>|95.7 96.1 97.4 96.6 97.0 97.3 97.4<br><br>|96.3 96.7 98.1 97.8 98.1 98.2 98.1|42.4 29.7 22.2 36.6 35.4 22.4 22.2<br><br>|48.8 48.7 57.3 58.0 58.0 57.3 57.3<br><br>|65.5 64.7 66.8 66.1 66.2 66.3 66.8|88.9 91.5 96.9 96.8 96.9 97.0 96.9<br><br>|90.9 94.7 97.3 97.1 97.3 97.4 97.3<br><br>|26.9 17.6 22.7 31.5 28.8 25.8 22.7<br><br>|68.2 67.2 72.3 75.8 74.4 73.4 72.3<br><br>|73.0 72.0 75.3 77.3 77.1 75.6 75.3|

- Table 3. Ablation Study Results for Different Model Components for Text-Image-To-Video (TI2V) task on VBench-I2V.

|Model|VBench-T2V<br><br>|VBench-I2V|
|---|---|---|
| |Q ↑ S ↑ T ↑<br><br>|I ↑ Q ↑ T ↑|
|T2V-M-512 STIV-M-512 STIV-M-512-JIT STIV-M-512-JIT-TUP<br><br>|82.2 77.0 81.2 74.6 31.9 66.1<br>82.3 74.1 80.7 83.0 73.1 81.0<br>|/ / / 98.0 82.1 90.1 97.6 81.9 89.8 97.2 82.3 89.7<br><br>|

- Table 4. Comparison of T2V, STIV and STIV with JIT-CFG on VBench and VBench-I2V I2V Score, Quality, Total scores.

Greater motion In practice, we have observed that while STIV-M-512 performs well on VBench-I2V, it sometimes generates static frames. The VBench-I2V metric tends to favor videos with less motion, prioritizing smoothness and consistency. As shown in Tab. 5, STIV with JIT-CFG achieves higher dynamic degrees at the cost of a slight reduction in consistency and smoothness scores. We also show visual comparisons from Fig. 14 to Fig. 16 in the Appendix.

|Model|Dynamic Degree<br><br>|Motion Smoothness<br><br>|Temporal Consistency|Background Flickering<br><br>|
|---|---|---|---|---|
|STIV-M-512 STIV-M-512-JIT|10.2 24.0<br><br>|99.6 99.1<br><br>|99.3 98.6|99.1 98.6<br><br>|

Table 5. Effect of JIT-CFG on motion-related scores.

JIT-CFG and the Variant It is natural to think about extending the traditional classifier-free guidance (CFG) to a three-copy weighted approach, where three conditioning modes are considered: (1) Null condition: Both the image (cI) and text (cT) conditions are null (∅). (2) Image-only condition: The image condition is the source image, while the text condition is null. (3) Joint condition: Both the image and text conditions are provided. The velocity estimates are combined as:

Fˆθ(xt,cT,cI,t) =Fθ(xt,∅,∅,t)

- + s1 · (Fθ(xt,cI,∅,t) − Fθ(xt,∅,∅,t))
- + s2 · (Fθ(xt,cI,cT,t) − Fθ(xt,cI,∅,t)).

Here, s1 and s2 are guidance scales for the image-only (CFG-I) and joint conditions (CFG-T), respectively. We dub it separate image and text classifier-free guidance (SIT-CFG). We achieved FV D = 94.1 on the MSRVTT test set using STIV-M-512-JIT by setting s = 7.5 in JIT-CFG. Meanwhile, we conducted experiments on our STIV-M-512-JIT, performing a grid search on s1 and s2 by taking the Cartesian product (1.1,1.5,4.5,7.5,10.5)× (1.1,1.5,4.5,7.5,10.5).

As shown in the Fig. 7, we observed: (1) Fixing CFG-T, as CFG-I increases, FVD first decreases and then increases; (2) Fixing CFG-I, as CFG-T increases, FVD continuously decreases, except when CFG-I is very small (1.1, 1.5), where it first decreases and then increases; (3) The best configuration occurs at CFG − T = 7.5 and CFG − I = 1.5, yielding FV D = 95.2. However, overall, SIT-CFG does not show significant advantages compared to JIT-CFG, and using two copies for inference is significantly less efficient. Note that this search was optimized for MSRVTT, and for other prompts requiring stronger dependence on the first-frame subject, a larger CFG-I might be needed.

CFG-I

1.1 1.5 4.5 7.5 10.5

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
|146|.8|125|.3|218.1| |316|.6|452|.9|
| | | | | | | | | | |
|139|.6|120|.8|217.5| |321|.2|449|.1|
| | | | | | | | | | |
|107|.2|97|.1|193.6| |310|.1|439|.9|
| | | | | | | | | | |
|105|.7|95|.2|164.5| |288|.4|424|.9|
| | | | | | | | | | |
|117|.3|106|.0|145.3| |269|.3|409|.6|
| | | | | | | | | | |

[Figure 31]

1.11.54.57.510.5

400

300

CFG-T

FVD

200

100

Figure 7. Grid search the optimal FVD for CFG-I s1 and CFG-T s2 on MSRVTT [63] test set.

###### 3.4.3 Model Initialization

We also studied how the performance of TI2V models is affected by the initialization methods, including starting from T2I or T2V. We run the same total number of steps to check the final performance on VBench-I2V. From Tab. 6, we observe that initializing from T2V model can achieve a better camera motion score and slightly better dynamic degree, while comparable to initializing from T2I on all other dimensions.

|Initialization|Subj Cons<br><br>|Bg Cons<br><br>|Temp Flick|Mot Smooth<br><br>|Dyn Deg<br><br>|Aesth Qual|Img Qual<br><br>|I2V Subj<br><br>|I2V Bg|Cam Mot<br><br>|Avg Scores<br><br>|
|---|---|---|---|---|---|---|---|---|---|---|---|
|T2V T2I<br><br>|94.1 94.5<br><br>|98.2 98.7|96.5 96.9<br><br>|97.7 97.9<br><br>|37.1 36.5|57.9 57.4<br><br>|65.5 66.1<br><br>|96.6 96.6|96.9 97.3<br><br>|38.0 29.8<br><br>|77.9 77.2|

Table 6. Results for different model initialization on VBench-I2V.

##### 3.5. Video Data Engine

Data quality is pivotal for video generation models. However, curating large-scale, high-quality datasets remains challenging due to issues like noisy captions, hallucinations, and limited diversity in video content and duration. To address these concerns, we propose a Video Data Engine (Fig. 8)—a comprehensive pipeline that improves dataset quality and reduces hallucinations, ultimately enhancing model performance. More details can be found in Sec. I in the appendix.

Our approach focuses on three key questions: (1) How to preprocess raw videos for better consistency? (2) What is the effect of data filtering on model performance? (3) How can advanced video captioning reduce hallucinations and improve outcomes? We use Panda-70M [10] as a working example and produce a curated subset, Panda-30M, via our pipeline.

Video Pre-processing and Feature Extraction. We employ PySceneDetect6 to remove abrupt transitions and inconsistent segments, yielding more coherent clips. We then extract key features (e.g., motion and aesthetic scores) to guide subsequent filtering.

6https://github.com/Breakthrough/PySceneDetect

- Per Segment Filtering:
- – Motion Score
- – Aesthetic Score
- – Video Orientation
- – …

Captioning: Raw Video

###### Pre-processing:

- – Richness
- – Accuracy
- – Hallucination
- – ..

- – Segment Video
- – Remove insistent segments
- – ..

[Figure 32]

[Figure 33]

Processed Video

Figure 8. An overview of our video data engine, including video pre-processing, filtering, and video captioning.

[Figure 34]

[Figure 35]

###### Object Existence Questions

###### Detected ?

[Figure 36]

###### Caption

Are there hands? Is there a meal? Is there rice? Is there meat? Are there vegetables? Is there a container? Is there an insect? Is there a box?

[Figure 37]

A person is preparing a meal with rice, meat, and vegetables. They are placing the food items in a black rectangular container. Next to the container, there is a small insect in a plastic box, and a yellow plastic box with a cat illustration, hinting at the meal's intended recipient. Hands reach out to grab the containers.

[Figure 38]

[Figure 39]

LLM MLLM

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Figure 9. An overview of DSG-Video’s approach to detecting object hallucinations in captions: we use an LLM to generate questions and another MLLM to validate the presence of the object across frames. If the MLLM fails to detect the object in all frames, the object is classified as a hallucination.

Data Engine for Filtering Effective data filtering is crucial for improving dataset quality and reducing hallucinations. We develop an automated filtering infrastructure that supports efficient data selection, quality control, and continuous improvement throughout the model’s development lifecycle. For instance, we can sample high-quality videos with predefined resolutions / motion scores for the fine-tuning stage. This filtering system allows us to systematically remove low-quality videos and focus on data that enhances model performance. From Panda-30M, we further apply filtering based on motion score and aesthetic score to obtain Panda-10M, named as a high-quality version of Panda-30M. The results are summarized in Tab. 7: instead of pursuing data volume, higher-quality videos have the potential to achieve more promising results.

|Data<br><br>|MSRVTT<br><br>|VBench| | |
|---|---|---|---|---|
| |FVD ↓|Quality ↑<br><br>|Semantic ↑<br><br>|Total ↑|
|Panda-30M Panda-10M|770.9 759.2<br><br>|80.4 80.8<br><br>|73.6 73.4<br><br>|65.6 66.2|

Table 7. Compare Panda-30M and Panda-10M (high-quality) using XL T2V model.

Video Captioning Model High-quality video-text pairs are essential for training text-to-video models. Existing datasets often suffer from noisy or irrelevant captions, limited in describing temporal dynamics. We initially attempted a frame-based captioning approach followed by LLM summarization [2], but found that single-frame captions fail to represent motion, and LLM summarization can induce hallucinations. To improve caption quality while balancing cost, we employ LLaVA-Hound-7B [69]—a video LLM capable of producing more coherent and motion-aware descriptions.

Caption Evaluation and Ablations To objectively assess caption accuracy, we introduce DSG-Video(Fig. 9), a module inspired by DSG [12], that detects hallucinated objects by probing captions with LLM-generated questions and verifying object presence in sampled video frames using a multimodal LLM. This yields two metrics, DSG-Videoi and DSG-Videos 7, reflecting hallucination at the object and sentence levels, respectively. We compare two captioning strategies—frame-based plus LLM summarization (FCapLLM) and direct video

7More details are illustrated in Appendix I.

captioning (VCap)—on the Panda-30M dataset. As shown in Tab. 8, VCap reduces hallucinations and increases the diversity of described objects, leading to improved T2V model performance. These results demonstrate that richer, more accurate video descriptions can significantly enhance downstream generation quality.

|Caption|Total Object<br><br>|DSG-Videoi(↓) DSG-Videos(↓)<br><br>|MSRVTT FVD (↓) VBench (↑)|
|---|---|---|---|
|FCapLLM VCap<br><br>|1249 1911<br><br>|6.4 24.0 5.3 15.0<br><br>|808.1 64.2 770.9 65.6<br><br>|

Table 8. Compare different captions using XL T2V model. DSG-Video metrics are calculated from 100 random captions.

#### 4. Results

Based on all of these studies, we scale our T2V and STIV model from 600M to 8.7B. We show the main results in Tab. 9 and Tab. 10, comparing our models with state-of-the-art open sourced and close sourced models, which demonstrates the effectiveness of our recipes. Specifically, we do finetuning on top of the pretrained video generation models (SFT), based on the 20,000 videos filtered from Panda-70M [10] using the method mentioned in Section 3.5. Since we adopt MaskDiT technique in our pretraining stage, we try finetuning our model in an unmask manner (UnmaskSFT). We also finetuned our STIV model to become a temporal upsampler to interpolate the videos generated by our main T2V and STIV models to boost the motion smoothness (+ TUP).

T2V Performance We first showcase the effectiveness of our T2V model as the foundation for STIV. Tab. 9 presents a comparison of different T2V model variants on VBench, including the VBench-Quality, VBenchSemantic, and VBench-Total scores. Our analysis reveals that scaling up model parameters in our T2V model improves semantic following capability. Specifically, as model size increase from XL to XXL and M, VBenchSemantic scores rise from 72.5 to 72.7 and then to 74.8. This explicit emergence (from XL, XXL to M), suggesting larger models are better at capturing semantic information. However, the impact on video quality, measured by VBench-Quality, remains modest, with only a slight increase from 80.7 to 81.2 and then to 82.1. This finding suggests that scaling has a greater effect on the model’s semantic capabilities than on video quality. Furthermore, increasing the spatial resolution from 256 to 512 significantly boosts the VBench-Semantic score from 74.8 to 77.0. Detailed results can be found in Tab. 11.

The Influence of SFT Additionally, fine-tuning the model with high-quality SFT data markedly enhances the VBench-Quality score from 82.2 to 83.9. Finetuning our model without any masked tokens slightly increases the performance of model on the semantic score. Our best model achieves a VBench-Semantic score of 79.5, outperforming prominent closed source models such as KLING, PIKA, and Gen-3. With the temporal upsampler, our model can achieve the state-of-the-art quality score compared with all other models.

TI2V Performance As shown in Tab. 10, our model delivers competitive performance compared to state-of-the-art approaches. It also reveals that while scaling up improves the I2V score, it has minimal impact on quality. In contrast, increasing the resolution leads to noticeable improvements in both quality and I2V scores. We provide complete results for the decomposed dimensions in Tab. 12.

#### 5. Flexible Applications

Here, we demonstrate how to extend our STIV to various applications, such as video prediction, frame interpolation, multi-view generation, and long video generation.

Video Prediction We initialize from a STIV-XXL model to train a text-video-to-video model conditioned on the first four frames. As shown in Fig. 10a, the video-to-video model (STIV-V2V) shows significantly lower FVD scores compared to the text-to-video model (T2V) on MSRVTT [63] test set and MovieGen Bench [46]. This

Model Quality ↑ Semantic ↑ Total ↑ Open Sourced Models

OpenSora V1.2 [74] 81.4 73.4 79.8 AnimateDiff-V2 [26] 82.9 69.8 80.3 VideoCrafter-2.0 [7] 82.2 73.4 80.4 T2V-Turbo [38] 82.2 74.5 80.6 CogVideoX-2B [65] 82.2 75.8 80.9 Allegro [75] 83.1 73.0 81.1 CogVideoX-5B [65] 82.8 77.0 81.6 LaVie-2 [60] 83.2 75.7 81.8

Close Sourced Models

- Gen-2 [51] 82.5 73.0 80.6 PIKA [44] 82.9 71.8 80.7 EMU3 [24] 84.1 68.4 81.0 KLING [34] 83.4 75.7 81.9

- Gen-3 [52] 84.1 75.2 82.3 Ours

XL 80.7 72.5 79.1 XXL 81.2 72.7 79.5 M 82.1 74.8 80.6 M-512 82.2 77.0 81.2 M-512 SFT 83.9 78.3 82.8 M-512 SFT + TUP 84.2 78.5 83.1 M-512 UnmaskSFT 83.7 79.5 82.9 M-512 UnmaskSFT + TUP 84.4 77.2 83.0

Table 9. Performance comparison of T2V variants with open-sourced and close-sourced models on VBench.

|Model<br><br>|Quality ↑|I2V ↑<br><br>|Total ↑|
|---|---|---|---|
|VideoCrafter-I2V [6] Consistent-I2V [49] DynamicCrafter-256 [62] SEINE-512 [11] I2VGen-XL [70] DynamicCrafter-512 [62] Animate-Anything [14] SVD [2]|81.3 78.9 80.2 80.6 81.2 81.6 81.2 82.8<br><br>|89.0 94.8 96.6 96.3 95.8 96.6 98.3 96.9<br><br>|85.1 86.8 88.4 88.4 88.5 89.1 89.8 89.9|
|STIV-XL STIV-M STIV-M-512 STIV-M-512-JIT<br><br>|79.1 78.8 82.1 81.9<br><br>|95.7 96.3 98.0 97.6<br><br>|87.4 87.6 90.1 89.8|

Table 10. Performance comparison of STIV-TI2V variants with open-sourced and close-sourced models on VBench-I2V.

result indicates that video-to-video models can achieve superior performance, which is promising for applications in autonomous driving and embodied AI where high fidelity and consistency in generated video frames are crucial.

Frame Interpolation We propose STIV-TUP, a temporal upsampler initialized from an STIV-XL model, and continue train conditioned on consecutive frames sampled by stride of 2 with the text conditioning. Fig. 10b shows that our STIV can also be used to do decent frame interpolation conditioned on both text and image. We

|Model<br><br>|use text<br><br>|MSRVTT FID ↓ FVD ↓|
|---|---|---|
|STIV-TUP STIV-TUP|No Yes<br><br>|2.2 6.3 2.0 5.9|

|Model<br><br>|MSRVTT FVD ↓|MovieGen FVD ↓<br><br>|
|---|---|---|
|T2V STIV-V2V<br><br>|536.2 183.7|347.2 186.3<br><br>|

(a) Comparison of T2V and V2V models.

(b) Performance of STIV-TUP.

|Model|GSO [18] PSNR ↑ SSIM ↑ LPIPS ↓<br><br>|
|---|---|
|Zero123++ [55] STIV-TI2V-XL|21.200 0.723 0.143 21.643 0.724 0.156<br><br>|

(c) Multiview generation comparison.

Figure 10. Demonstration of flexible applications of STIV framework.

observe that using text conditions is slightly better in FID and FVD on the MSRVTT test set. We also cascade the temporal upsampler with our main model to explore whether it can boost the main performance. As shown in Tab. 9 and Tab. 4, using a temporal upsampler on the top of the main models can improve the quality performance while maintaining other scores.

Multi-View Generation Multi-view generation is a specialized task focused on creating novel views from a given input image. This task places demands on view consistency and can greatly benefit from a well-pretrained video generation model. By adapting a video generation model for multi-view generation, we can evaluate whether the pretraining has effectively captured underlying 3D information, which would enhance multi-view generation.

Here, we adopt the novel view camera definitions outlined in Zero123++ [55], which specifies six novel view cameras for each input image. The initial frame in our TI2V model is set as the given image, and the next six frames, representing novel views, are predicted as future frames within TI2V. For training, we began with our TI2V-XL checkpoint trained with a 256 resolution, fine-tuning it for 110k steps on Objaverse [16]. For a fair comparison, we increased the image resolution to 320 during finetuning, aligning with the settings used in Zero123++. Our evaluation used objects from the Google Scanned Objects dataset [18], where we compared the output multi-view images against ground-truth renderings. As shown in Fig. 10c, despite only using temporal attention for cross-view consistency, our approach achieves comparable performance to Zero123++ which uses full attention to all the views. This outcome validates the effectiveness of our spatiotemporal attention in maintaining 3D consistency. A visual comparison between our approach and Zero123++ is shown in Figure 11.

Long Video Generation We develop an effective and efficient framework to generate long videos. Specifically, we propose a hierarchical framework, including training our STIV on two different modes: (1) key frame prediction by learning uniformly sampled video frames with stride of 20 with image conditioning dropout and (2) interpolated frame generation by learning consecutive video frames with first and last frame as image conditions. During sampling stage, we change the image and micro conditions, and first use the first mode to generate key frames and then generate the interpolated frames use the second mode, leading to a long video. It is natural to reuse the STIV model to autoregressively generate the videos conditioned on previous generated one. However, in practice, we found this rollout approach can be hurt by error propagation in the previous video, and lacks some global consistency between frames. Therefore, we propose a simple yet effective baseline, purely based on our STIV framework. As mentioned in the main text, we design a hierarchical framework, train our STIV on two different modes: (1) key frame prediction by learning uniformly sampled video frames with a stride of 20 with image conditioning dropout, and (2) interpolated frame generation by learning consecutive video frames with the first and last frame as image conditions. During the sampling stage, we change the image and micro conditions, and first use the first mode to generate key frames and then generate the interpolated frames using the second mode, leading to a long video. We showcase one long T2V and one TI2V example in Figure 12; we achieve (20 − 1) × 20 = 380 frames in total. We uniformly sampled 8 frames out of the 380 frames. Noted that this is only an early exploration of long video generation, and we do not have many long enough videos in our training distribution, so we leave it as future work to further explore the architecture to boost long video synthesis.

|Multi-View Generation|
|---|

[Figure 46]

|[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]|
|---|

|[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 59]

|[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]|
|---|

|[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]|
|---|

[Figure 78]

|[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]|
|---|

[Figure 91]

|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]|
|---|

[Figure 104]

|[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]|
|---|

[Figure 117]

|[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]|
|---|

[Figure 130]

|[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]|
|---|

[Figure 143]

|[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]|
|---|

[Figure 156]

|[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Figure 11. The visual comparison between our STIV-XL with Zero123++ [55] on GSO [18].

|Long Video Generation|
|---|

Prompt: A drone camera circles around a beautiful historic church built on a rocky outcropping along the Amalfi Coast, the view showcases historic and magnificent architectural details and tiered pathways and patios...

|[Figure 163]|
|---|

###### Reference Image

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Prompt: A slow cinematic push in on an ostrich standing in a 1980s kitchen.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Figure 12. Visualizations of long video generation framework.

#### 6. Related Work

Text-To-Video Generation In recent years, diffusion-based methods have emerged as the dominant approach in text-to-video generation, both for close-source models [42, 44, 46] and open-source models [66, 74]. [6, 7, 27] leverages latent diffusion models (LDMs) [50] to enhance training efficiency. VideoLDM [3] integrates temporal convolution and attention mechanisms into the LDM U-Net for video generation. Recently, there has been a shift from UNet to diffusion transformer-based architectures [21, 46, 66, 75]. CogVideoX [65] adopts the framework from SD3 [19] to incorporate self-attention on the entire 3D video sequence with text conditions. Lumina-T2X [39] employs zero-init attention to transform noise into different modalities. In contrast to previous models, our focus is to scale our diffusion transformer-based video generation model with spatial, temporal, and cross attention to over 8B parameters using various techniques. This model achieves good performance on VBench and serves as a strong baseline for the development of our text-image-to-video model: STIV.

Text-Image-To-Video Generation Controlling video content solely through text poses significant challenges in achieving satisfactory alignment between the video and the input text, as well as fine-grained control over the video generation process. To address this issue, recent approaches have integrated both the first frame and text to enhance control over video generation [6, 24, 49, 62, 70], mostly based on U-Net architecture. I2VGen-XL [70] builds upon the SDXL and employs a cascading technique to generate high-resolution video. DynamiCrafter [62] and VideoCrafter [6] use cross-attention to incorporate image condition. ConsistentI2V [49] employs a similar frame replacement strategy, but it also requires spatial temporal attention over the initial frame and special noise initialization to enhance consistency. Animate Anything [15] also employs the frame replacement technique, but it requires the use of motion strength loss to enhance the motion. However, their Dynamic Degree on VBench-I2V is relatively low, at 2.7%. We apply frame replacement on the DiT architecture, along with our proposed image condition dropout method, and JIT-CFG can generative high quality I2V videos while effectively addresses the motion staleness issue.

#### 7. Conclusion

In conclusion, we conduct a comprehensive study on how to build a good video generation model, and present a scalable and flexible approach for integrating text and image conditioning within a unified video generation framework. Our model not only demonstrates good performance on public benchmarks, but also shows versatility in downstream applications, supporting controllable video generation, video prediction, frame interpolation, long video generation, and multi-view generation, which collectively highlight its potential as a foundation for the broad research community.

#### 8. Acknowledgement

We thank Yifan Jiang,Alex Schwing, Monica Zuendorf, Saeed Khorram, Pengsheng Guo and Zhe Gan for their regular discussions on model design and training recipes. We also acknowledge the indirect influence of these discussions on our approach. Furthermore, we thank Meng Cao for data and Ruoming Pang for training infrastructure support. We also acknowledge the leadership guidance from Yang Zhao throughout the project. Finally, we thank the Axlearn team 8 at Apple for providing training infrastructure, which greatly facilitated our experiments.

8https://github.com/apple/axlearn

#### References

- [1] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, page 4, 2021. 2, 4
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 13, 15, 9, 11
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575,

2023. 19

- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 6
- [5] Joao Carreira and Andrew Zisserman. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 10
- [6] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. VideoCrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2, 15, 19, 8
- [7] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310–7320, 2024. 15, 19, 8, 9
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 1, 4, 2
- [9] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 11
- [10] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70M: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 8, 12, 14
- [11] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. arXiv preprint arXiv:2310.20700, 2023. 2, 15, 9
- [12] Jaemin Cho, Yushi Hu, Jason Baldridge, Roopal Garg, Peter Anderson, Ranjay Krishna, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In ICLR, 2024. 8, 13, 12
- [13] Zhixuan Chu, Lei Zhang, Yichen Sun, Siqiao Xue, Zhibo Wang, Zhan Qin, and Kui Ren. Sora detector: A unified hallucination detection for large text-to-video models. arXiv preprint arXiv:2405.04180, 2024. 1
- [14] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Animateanything: Fine-grained open domain image animation with motion guidance. arXiv e-prints, pages arXiv–2311,

2023. 2, 15, 9

- [15] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Fine-grained open domain image animation with motion guidance. arXiv preprint arXiv:2311.12886, 2023. 19
- [16] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153,

2023. 16

- [17] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34:19822–19835, 2021. 2, 5
- [18] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE,

2022. 16, 17

- [19] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1, 2, 5, 19
- [20] Tsu-Jui Fu, Licheng Yu, Ning Zhang, Cheng-Yang Fu, Jong-Chyi Su, William Yang Wang, and Sean Bell. Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 11
- [21] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024. 1, 19
- [22] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic VQGAN and time-sensitive transformer. In European Conference on Computer Vision, pages 102–118, 2022. 10, 11
- [23] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 8
- [24] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. In European Conference on Computer Vision, 2023. 15, 19, 8
- [25] Xinyu Gong, Wuyang Chen, Tianlong Chen, and Zhangyang Wang. Sandwich Batch Normalization: A Drop-In Replacement for Feature Distribution Heterogeneity. In Winter Conference on Applications of Computer Vision (WACV), 2022. 2
- [26] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 15, 8
- [27] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 19
- [28] Alex Henry, Prudhvi Raj Dachapally, Shubham, and Yuxuan Chen. Query-Key Normalization for Transformers. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. 2
- [29] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. 8
- [30] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 8, 11
- [31] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 2, 8, 7
- [32] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The Kinetics Human Action Video Dataset. In arXiv:1412.0767, 2014. 10

- [33] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:36652–36663, 2023. 8
- [34] Kuaishou. Kling. https://kling.kuaishou.com, 2024. 15, 8
- [35] Zhengfeng Lai, Vasileios Saveris, Chen Chen, Hong-You Chen, Haotian Zhang, Bowen Zhang, Juan Lao Tebar, Wenze Hu, Zhe Gan, Peter Grasch, et al. Revisit large-scale image-caption data in pre-training multimodal foundation models. arXiv preprint arXiv:2410.02740, 2024. 8, 3
- [36] Zhengfeng Lai, Haotian Zhang, Bowen Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan, Jiulong Shan, Chen-Nee Chuah, et al. Veclip: Improving clip training via visual-enriched captions. In European Conference on Computer Vision, pages 111–127. Springer, 2025. 3
- [37] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. ArXiv e-prints, pages arXiv–1607, 2016. 5
- [38] Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 15, 8
- [39] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024. 5, 19
- [40] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 11
- [41] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024. 2, 5, 1
- [42] OpenAI. Sora. https://openai.com/index/sora, 2024. 1, 3, 19, 13
- [43] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023. 1, 8, 2
- [44] Pika. Pika 1.0. https://pika.art, 2023. 15, 19, 8
- [45] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3
- [46] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 1, 3, 14, 19, 13
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4
- [48] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 4
- [49] Weiming Ren, Huan Yang, Ge Zhang, Cong Wei, Xinrun Du, Wenhao Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-to-video generation. arXiv preprint arXiv:2402.04324, 2024. 2, 15, 19, 9
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 19

- [51] Runway. Gen-2. https://research.runwayml.com/gen2, 2023. 15, 8
- [52] Runway. Gen-3 alpha. https://runwayml.com/research/introducing-gen-3-alpha,

2024. 15, 8

- [53] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved Techniques for Training GANs. In Conference on Neural Information Processing Systems (NeurIPS), 2016. 10
- [54] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR, 2018. 2, 7
- [55] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 16, 17
- [56] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 2, 5
- [57] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2
- [58] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning Spatiotemporal Features with 3D Convolutional Networks. In International Conference on Computer Vision (ICCV), 2015. 10
- [59] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 8
- [60] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 15, 8
- [61] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023. 8
- [62] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190,

2023. 2, 15, 19, 9

- [63] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. MSR-VTT: A large video description dataset for bridging video and language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5288–5296, 2016. 8, 12, 14
- [64] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation, 2023. 8
- [65] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In arXiv:2408.06072, 2024. 1, 15, 19, 8
- [66] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, JiaZheng Xu, Yuanming Yang, Xiaohan Zhang, Xiaotao Gu, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 19
- [67] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, and Lu Jiang Irfan Essa. MAGVIT: Masked Generative Video Transformer. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 11
- [68] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 5

- [69] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024. 13, 11
- [70] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2, 15, 19, 9
- [71] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, 2024. 11
- [72] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 11
- [73] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023. 2, 5
- [74] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-Sora: Democratizing Efficient Video Production for All, 2024. 15, 19, 8
- [75] Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv:2410.15458, 2024. 15, 19, 8
- [76] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024. 5

## STIV: Scalable Text and Image Conditioned Video Generation Appendix

#### A. Joint Image-Text Classifier-free Guidance

We introduce a novel framework, Joint Image-Text Classifier-Free Guidance (JIT-CFG), in Section 3.4, which facilitates the seamless integration of text and image conditions to enhance the modeling performance. This is accomplished through a modified velocity estimate, expressed as:

Fˆθ(xt,cT,cI,t) = Fθ(xt,∅,∅,t)

+ ω · (Fθ(xt,cT,cI,t) − Fθ(xt,∅,∅,t))

The approach employs text and image condition dropout, which is also critical for unifying T2V and TI2V tasks.

Probability mass shift Our model learns P(x|cT,cI), the probability distribution of generating video x given the text prompt cT and image condition cI. Here, we demonstrate how JIT-CFG shifts the probability mass toward regions of higher likelihood, conditioned on cT and cI. First, consider a score-matching model with JIT-CFG

sˆθ(xt,cT,cI,t) = sθ(xt,t)

+ ω · (sθ(xt,cT,cI,t) − sθ(xt,t)) Using the definition of score and Bayes’ Rule, we derive

sˆθ(xt,cT,cI,t)

=∇logPt(xt) + ω · (∇logPt(xt,cT,cI) − ∇logPt(xt))

=∇logPt(xt) + ω · ∇logP(cT,cI|xt)

=∇log (Pt(xt)Ptω(cT,cI|xt))

=∇log Pt1−ω(xt)Ptω(xt|cT,cI) ,

where ω determines the influence of the text and image conditions during sampling from the tempered distribution. For a flow-matching model employing linear interpolants, the velocity and score are related as [41]:

1 1 − t

t 1 − t

Fˆθ(xt,cT,cI,t) −

sˆθ(xt,cT,cI,t) =

###### xt,

It implies ∂∂Fsˆˆ = 1−t t > 0, meaning that the JIT-CFG-guided velocity Fˆ shifts the probability mass in alignment with the modified score sˆ by adjusting the tempered distribution.

CFG-Renormalization Empirically, we observed that the magnitude of the modified velocity, ||Fˆθ(xt,cT,cI,t)|| tends to be very large during the early stages of integration in inference (i.e. when t is small). This behavior sometimes leads to overshooting beyond the learned latent distribution, resulting in artifacts in the generated output. We identified this issue as primarily due to the significant difference between the conditional velocity, Fθ(xt,cT,cI,t), and the unconditional velocity, Fθ(xt,∅,∅,t) when t is small.

To mitigate this, we propose a simple yet effective renormalization method that re-scales the magnitude of the modified velocity to ||Fθ(xt,cT,cI,t)|| while preserving its direction. Formally, this is defined as:

Fˆθ(xt,cT,cI,t) ||Fˆθ(xt,cT,cI,t)||

F˜θ(xt,cT,cI,t) = ||Fθ(xt,cT,cI,t)||

As shown in Table 2, this technique significantly improves performance across various T2I evaluation benchmarks.

#### B. Implementation Details for T2V and STIV

Given that we use spatial-temporal attention, we first pretrain the T2I model using only an image dataset. Subsequently, we load the EMA weights from the T2I model, excluding the temporal attention. In our work, we use the per-frame VAE, which is the same one used in the T2I model. On top of that, we use a temporal patch of size 2 in the DiT part for video models. We modify the T2I cubify weights by inflating the 3D convolution weight in the temporal dimension. For video training data, we select one frame from every three frames and add independent and identically distributed Gaussian noise to each frame. Following standard practice, we randomly replace text prompt with empty string 10% during training. In our JIT-CFG setting, we also independently randomly drop image condition 8% during training. For both T2V and STIV models, the JIT-CFG scale is set to 7.5. The training schedule follows the progressive training recipe described in section 2.2.

#### C. Implementation Details of Text Encoders

We used our internal CLIP text encoder to encode text into embeddings. Concretely, a text is first tokenized via a T5 tokenizer. The tokenized text is mapped into embeddings via an embedding lookup table and further encoded via 32 layers of transformer with casual attention. Each transformer layer contains 20 attention heads. Each attention head has 64 hidden dimensions. The output text embedding has a dimension of 1280.

#### D. Ablation Study on Text-to-Image Generation

Baseline Setup For our base model, we employed the PixArt-α architecture [8], which builds on the DiT [43] model with added cross-attention layers to integrate image tokens with text embeddings. As pre-trained components, we used the open-source sd-vae-ft-ema model9 and OpenAI CLIP L14 model10, both of which are widely adopted in the community. We conduct our experiments using the XL model configuration with a 2562 image size. The full baseline model, which includes the VAE and CLIP text encoder, has approximately 1.06 billion parameters. For noise generation and denoising, we used a diffusion-based approach with Stable Diffusion’s default noise schedule. The training was conducted with a batch size of 4,096 over 400k steps, which corresponds to approximately 1.4 epochs on our internal text-to-image dataset.

Table 2 summarizes the results of our ablation study, focusing on the following aspects:

Stabilized Training Leveraging recent advancements in LLM and diffusion model architectures, we integrated QK-Norm [28] to manage the activation scale within attention layers. Additionally, we applied sandwich-norm [25] to both the inputs and outputs of the attention layer and the feedforward layer. Projected conditions, including timestep embeddings, pooled CLIP text embeddings, and micro condition embeddings, were normalized before being input to AdaLN. These normalization techniques enhanced training stability, allowing us to increase the learning rate from 1 × 10−4 to 2 × 10−4, and also resulting in quality improvements.

Noising and Denoising Process Formulation We explored optimized noising/denoising formulations by replacing the diffusion process with a flow-based linear interpolant[41]. Additionally, we applied renormalization at each inference step to counteract potential side effects from high classifier-free guidance (CFG) values. Here, the norm of the prediction with CFG was linearly scaled to match the conditional prediction norm, as explained in Sec. A.

- 9https://huggingface.co/stabilityai/sd-vae-ft-ema
- 10https://huggingface.co/openai/clip-vit-large-patch14

|25.92%<br><br>30.94%<br><br>28.98%<br><br>21.51%<br><br>26.04%<br><br>32.78%<br><br>52.57%<br><br>43.01%<br><br>38.24%<br><br>DIFFUSION TO FLOW<br><br>INTERNAL CLIP<br><br>SYNTH. CAPTION<br><br>Baseline Win Tie New Config Win<br><br>|
|---|

Figure 13. Human evaluation results on significant changes in T2I ablation study Tab. 2.

Training Cost Optimization To reduce training costs, we evaluated three strategies: (1) switching from the AdamW optimizer to Adafactor, (2) applying MaskDiT training with a 50% masking ratio, and (3) using a shared AdaLN module across layers instead of unique instances per layer. These changes reduced per-device HBM usage from approximately 28GB to 11GB, allowing us to train on v5e TPUs instead of the more costly v5p TPUs. Notably, as shown in Table 2, masked training may adversely affect metrics such as FID and HPS. However, we found additional unmask finetuning for a short duration (e.g. 50k steps) can fix the artifacts causing these score drops. However, this additional training phase was not included in the final configuration, as further training on video generation can address this issue as well.

Enhanced Pre-trained Models and Conditioning We evaluated improvements from advanced pre-trained models and additional conditioning techniques. Specifically, we upgraded from the OpenAI CLIP L14 to an internally trained CLIP-bigG model [36] and from a 4-channel to an 8-channel VAE. We also introduced 2D RoPE to support masked training and added micro-conditions, inspired by SDXL [45], to mitigate cropping artifacts in elongated objects. Finally, synthetic captions generated via [35] were included in our training data, resulting in notable performance gains.

Human Evaluation of Model Changes To validate improvements observed in automated metrics, we conducted human evaluations for key modifications, including the addition of synthetic captions, upgrade of CLIP model, and transition from diffusion to flow matching based objective. Human raters are asked to asses image fidelity, text-image alignment, and visual appeal, and give 5 level preference ratings for image pairs. Each pair is sent to 5 raters for rating and the image pair will be considered tie of combined voting is neutral. Results from Figure 13 demonstrate clear alignment between automated metrics and human judgments. This justifies our usage of automatic evaluation as daily development metrics to maintain generation quality and prevent regressions leading to significant quality losses.

#### E. Detailed Results for Imaging Dropout

As mentioned in Section 3.4.2, after adding imaging dropout. We observe this phenomenon happens when we scale our model to 8B with >= 512 resolutions, probably due to the model being more easily overfitting to follow the first frame with a larger model capacity, and it becomes worse under the higher resolution. Specifically, we showcase some examples to see the different between generated videos without image dropout and videos with image dropout (STIV-M-512). We generate the videos conditioned on the first frame and text prompt borrowed from MovieGenBench [46] As shown in Fig. 14 to 16, using image condition dropout in general achieves better performance than the baseline in terms of motion quality.

|STIV-M-512 V.S. STIV-M-512-JIT|
|---|

|[Figure 181]|
|---|

###### Reference Image

Prompt: A red panda taking a bite of a pizza.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

|[Figure 186]|
|---|

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Reference Image Prompt: A rocket blasting off from the launch pad, accelerating rapidly into the sky.

|[Figure 191]|
|---|

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Reference Image

|[Figure 196]|
|---|

Reference Image

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

- Figure 14. Visualization of STIV-M-512 V.S. STIV-M-512-JIT. (Given the same prompt, the figures in the top row are generated by STIV-M-512, while the figures in the bottom row are generated by STIV-M-512-JIT.)

|STIV-M-512 V.S. STIV-M-512-JIT|
|---|

Prompt: A sports car accelerating rapidly on an open highway, the engine roaring.

|[Figure 201]|
|---|

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Reference Image

|[Figure 206]|
|---|

Reference Image

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Prompt: A glass of iced coffee condensing water on the outside, with droplets forming and sliding down the glass in slow motion.

|[Figure 211]|
|---|

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Reference Image

|[Figure 216]|
|---|

Reference Image

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

- Figure 15. Visualization of STIV-M-512 V.S. STIV-M-512-JIT. (Given the same prompt, the figures in the top row are generated by STIV-M-512, while the figures in the bottom row are generated by STIV-M-512-JIT.)

|STIV-M-512 V.S. STIV-M-512-JIT|
|---|

Prompt: Cars and pedestrians move through a bustling downtown street lined with skyscrapers, their lights reflecting off the windows of the towering buildings as day turns to dusk.

|[Figure 221]|
|---|

Reference Image

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

|[Figure 226]|
|---|

Reference Image

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Prompt: Robots move efficiently through a futuristic laboratory, adjusting holographic displays and conducting experiments, while scientists observe and interact with the high-tech equipment.

|[Figure 231]|
|---|

Reference Image

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

|[Figure 236]|
|---|

Reference Image

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

- Figure 16. Visualization of STIV-M-512 V.S. STIV-M-512-JIT. (Given the same prompt, the figures in the top row are generated by STIV-M-512, while the figures in the bottom row are generated by STIV-M-512-JIT.)

#### F. Detailed Results for T2V and STIV

- F.1. Details of VBench and VBench-I2V Evaluation Metrics We follow the same as the evaluation protocol provided by VBench [31].

- F.1.1 Video Quality

Video Quality is divided into two aspects: Temporal Quality and Image Quality. Temporal Quality evaluates cross-frame consistency, including (1) Subject Consistency, ensuring that subjects maintain a consistent appearance across frames; (2) Background Consistency, assessing stability in the background using feature similarity; (3) Temporal Flickering, measuring smooth transitions in both static and dynamic areas; (4) Motion Smoothness, evaluating the fluidity and realism of motion; and (5) Dynamic Degree, analyzing the presence of large-scale dynamics or motions. Image Quality focuses on individual images and evaluates (1) Aesthetic Quality, considering artistic appeal and visual richness, and (2) Imaging Quality, measuring clarity, noise, and other distortions.

- F.1.2 Video-Condition Consistency

Video-Condition Consistency ensures alignment with the input prompt and is categorized into Semantics and Style, each with finer-grained dimensions.

Semantics (1) Object Class: Measures the success of generating specific objects described in the text prompt. (2) Multiple Objects: Evaluates the ability to compose multiple objects from different classes in a single frame. (3) Human Action: Assesses whether the generated video accurately captures actions described in the prompt. (4) Color: Ensures synthesized object colors align with the text description. (5) Spatial Relationship: Checks whether spatial relationships between objects align with the prompt. (6) Scene: Evaluates consistency between generated scenes and the intended description (e.g., “ocean” versus “river”).

Style (1) Appearance Style: Measures consistency of styles mentioned in the prompt, such as “oil painting” or “cyberpunk” (2) Temporal Style: Assesses temporal continuity of styles across frames, ensuring smooth transitions.

Overall Consistency We further evaluate Overall Consistency using metrics that combine semantic and style alignment, reflecting both the accuracy and coherence of generated videos.

VBench-I2V builds upon the VBench with three new Video-Image Alignment metrics: Subject Consistency, Background Consistency, and Camera Motion Control. These additional metrics provide a more comprehensive evaluation by focusing on how well the generated video aligns with the input image and specified prompt instructions. Specifically, Subject Consistency evaluates the alignment between the subject in the input image and the generated video, ensuring coherence in character or object representation. Background Consistency assesses the continuity of the background scene between the input image and the video, highlighting the model’s ability to maintain a consistent environment. Camera Motion Control, under Video-Text Alignment, examines the adherence to camera control directions as described in the prompt, which is crucial for generating realistic video sequences that respond to specified dynamic instructions.

- F.2. Detailed Results on VBench and VBench-I2V We showcase the detailed version of the performance shown in Tab. 11 and Tab. 12.

|Model<br><br>|Subject Cons.|Back. Cons.<br><br>|Temporal Flickering<br><br>|Motion Smooth.|Dynamic Degree<br><br>|Aesthetic Quality<br><br>|Imaging Quality<br><br>|Object Class|Multiple Objects<br><br>|Human Action|
|---|---|---|---|---|---|---|---|---|---|---|
|CogVideoX-5B [65] CogVideoX-2B [65] Allegro [75] AnimateDiff-V2 [26] OpenSora V1.2 [74] T2V-Turbo [38] VideoCrafter-2.0 [7] LaVie-2 [60] LaVIE [60] ModelScope [59] VideoCrafter [6] CogVideo [30]|96.2 96.8 96.3 95.3 96.8 96.3 96.9 97.9 91.4 89.9 86.2 92.2<br><br>|96.5 96.6 96.7 97.7 97.6 97.0 98.2 98.5 97.5 95.3 92.9 95.4<br><br>|98.7 98.9 99.0 98.8 99.5 97.5 98.4 98.8 98.3 98.3 97.6 97.6|96.9 97.7 98.8 97.8 98.5 97.3 97.7 98.4 96.4 95.8 91.8 96.5<br><br>|80.0 59.9 55.0 40.8 42.4 49.2 42.5 31.1 49.7 66.4 89.7 42.2<br><br>|62.0 60.8 63.7 67.2 56.9 63.0 63.1 67.6 54.9 52.1 44.4 38.2|62.9 61.7 63.6 70.1 63.3 72.5 67.2 70.4 61.9 58.6 57.2 41.0<br><br>|85.2 83.4 87.5 90.9 82.2 94.0 92.6 97.5 91.8 82.2 87.3 73.4|62.1 62.6 59.9 36.9 51.8 54.7 40.7 64.9 33.3 39.0 25.9 18.1<br><br>|99.4 98.0 91.4 92.6 91.2 95.2 95.0 96.4 96.8 92.4 93.0 78.2|
|PIKA [44] Gen-3 [52] Gen-2 [51] KLING [34] EMU3 [24]|96.9 97.1 97.6 98.3 95.3<br><br>|97.4 96.6 97.6 97.6 97.7|99.7 98.6 99.6 99.3 98.6<br><br>|99.5 99.2 99.6 99.4 98.9<br><br>|47.5 60.1 18.9 46.9 79.3<br><br>|62.4 63.3 67.0 61.2 59.6|61.9 66.8 67.4 65.6 62.6<br><br>|88.7 87.8 90.9 87.2 86.2<br><br>|43.1 53.6 55.5 68.1 44.6<br><br>|86.2 96.4 89.2 93.4 77.7|
|XL XXL M-256 M-512 M-512-SFT M-512-SFT+TUP M-512-UnMSFT M-512-UnMSFT+TUP|96.0 97.5 96.0 95.9 96.7 94.8 94.3 95.2<br><br>|98.5 98.9 98.5 96.9 97.4 95.9 96.9 95.8<br><br>|98.4 99.1 98.6 98.8 98.7 98.7 98.8 98.8|96.5 98.2 97.2 98.0 98.3 99.2 96.7 99.2<br><br>|62.5 48.6 68.1 59.7 70.8 70.8 77.8 70.8<br><br>|56.3 56.2 57.0 60.6 61.7 63.7 61.4 63.6|59.3 59.7 60.8 62.5 63.9 65.0 68.6 65.9<br><br>|91.5 91.1 88.8 85.9 88.1 88.9 90.0 90.0|41.3 49.1 62.1 72.4 67.7 70.3 72.3 69.8<br><br>|98.0 99.0 98.0 96.0 97.0 95.0 97.0 94.0|

|Model|Color<br><br>|Spatial Rel.<br><br>|Scene<br><br>|App. Style|Temp. Style<br><br>|Overall Cons.<br><br>|Quality Score<br><br>|Semantic Score|Total Score<br><br>|Averaged Scores|
|---|---|---|---|---|---|---|---|---|---|---|
|CogVideoX-5B [65] CogVideoX-2B [65] Allegro [75] AnimateDiff-V2 [26] OpenSora V1.2 [74] T2V-Turbo [38] VideoCrafter-2.0 [7] LaVie-2 [60] LaVIE [60] ModelScope [59] VideoCrafter [6] CogVideo [30]<br><br>|82.8 79.4 82.8 87.5 90.1 89.9 92.9 91.7 86.4 81.7 78.8 79.6|66.4 69.9 67.2 34.6 68.6 38.7 35.9 38.7 34.1 33.7 36.7 18.2<br><br>|53.2 51.1 46.7 50.2 42.4 55.6 55.3 49.6 52.7 39.3 43.4 28.2<br><br>|24.9 24.8 20.5 22.4 24.0 24.4 25.1 25.1 23.6 23.4 21.6 22.0<br><br>|25.4 24.4 24.4 26.0 24.5 25.5 25.8 25.2 25.9 25.4 25.4 7.8|27.6 26.7 26.4 27.0 26.9 28.2 28.2 27.4 26.4 25.7 25.2 7.7<br><br>|82.8 82.2 83.1 82.9 81.4 82.6 82.2 83.2 78.8 78.1 81.6 72.1<br><br>|77.0 75.8 73.0 69.8 73.4 74.8 73.4 75.8 70.3 66.5 72.2 46.8<br><br>|81.6 80.9 81.1 80.3 79.8 81.0 80.4 81.8 77.1 75.8 79.7 67.0|70.0 68.3 67.5 64.7 66.0 67.4 66.0 67.6 63.8 62.4 62.3 52.3<br><br>|
|PIKA [44] Gen-3 [52] Gen-2 [51] KLING [34] EMU3 [24]<br><br>|90.6 80.9 89.5 89.9 88.3<br><br>|61.0 65.1 66.9 73.0 68.7<br><br>|49.8 54.6 48.9 50.9 37.1<br><br>|22.3 24.3 19.3 19.6 20.9|24.2 24.7 24.1 24.2 23.3<br><br>|25.9 26.7 26.2 26.4 24.8<br><br>|82.9 84.1 82.5 83.4 84.1<br><br>|71.8 75.2 73.0 75.7 68.4<br><br>|80.7 82.3 80.6 81.9 81.0|66.1 68.5 66.1 68.8 66.7<br><br>|
|XL XXL M-256 M-512 M-512-SFT M-512-SFT+TUP M-512-UnMSFT M-512-UnMSFT+TUP|86.4 90.8 83.6 91.2 93.7 94.7 92.0 87.7<br><br>|42.4 45.1 44.5 51.0 58.0 50.6 59.8 46.9|54.4 45.5 54.7 53.6 52.8 57.3 53.1 57.1<br><br>|22.4 22.1 22.5 23.9 24.6 24.5 24.8 24.5|26.3 26.1 26.6 25.8 26.2 26.7 26.7 26.6<br><br>|27.8 27.4 28.4 27.8 28.5 28.6 28.8 28.5<br><br>|80.7 81.2 82.7 82.2 83.9 84.2 83.7 84.4<br><br>|72.5 72.7 74.8 77.0 78.3 78.5 79.5 77.2|79.1 79.5 80.6 81.2 82.8 83.1 82.9 83.0<br><br>|66.1 65.9 67.9 68.8 70.3 70.3 71.2 69.7|

###### Table 11. Detailed Evaluation Results for Text-To-Video Generation Models.

|Model|Subject Consistency<br><br>|Background Consistency|Temporal Flickering<br><br>|Motion Smoothness<br><br>|Dynamic Degree<br><br>|Aesthetic Quality|
|---|---|---|---|---|---|---|
|DynamicCrafter-256 [62] DynamicCrafter-512 [62] Animate-Anything [14] SVD [2] SEINE-512 [11] VideoCrafter-I2V [7] Consistent-I2V [49] I2VGen-XL [70]<br><br>|94.7 93.8 98.9 95.5 95.3 97.9 95.3 94.2|98.3 96.6 98.2 96.6 97.1 98.8 98.3 97.1<br><br>|98.1 95.6 98.1 98.1 97.3 98.2 97.6 98.3|97.8 96.8 98.6 98.1 97.1 98.0 97.4 26.1<br><br>|40.6 69.7 2.7 52.4 27.1 22.6 18.6 26.1|58.7 60.9 67.1 60.2 64.6 60.8 59.0 64.8<br><br>|
|STIV-M STIV-M-512 STIV-M-512-JIT<br><br>|95.4 99.5 98.1|98.9 99.3 98.6<br><br>|97.2 99.5 98.7<br><br>|98.1 99.6 99.1<br><br>|32.1 10.2 24.0|59.0 62.5 65.4<br><br>|

|Model<br><br>|Imaging Quality<br><br>|I2V Subject|I2V Background<br><br>|Camera Motion|I2V Quality<br><br>|Final Score<br><br>|
|---|---|---|---|---|---|---|
|DynamicCrafter-256 [62] DynamicCrafter-512 [62] Animate-Anything [14] SVD [2] SEINE-512 [11] VideoCrafter-I2V [7] Consistent-I2V [49] I2VGen-XL [70]|62.3 68.6 72.1 69.8 71.4 71.7 66.9 69.1<br><br>|97.1 97.2 98.8 98.8 97.2 91.2 95.8 96.5|97.6 97.4 98.6 98.6 96.9 91.3 96.0 96.8<br><br>|20.9 32.0 13.1 62.3 21.0 33.6 33.9 18.5<br><br>|80.2 81.6 81.2 82.8 80.6 81.3 78.9 81.2<br><br>|88.4 89.1 89.8 89.9 88.4 85.1 86.8 88.5|
|STIV-M STIV-M-512 STIV-M-512-JIT|66.1 71.5 71.0<br><br>|97.0 99.2 98.8<br><br>|97.4 97.3 97.5<br><br>|22.7 13.2 15.1|78.8 82.1 81.9<br><br>|87.6 90.1 89.8|

Table 12. Detailed Evaluation Results for Text-Image-To-Video Generation Models.

#### G. Details of Model Initialization Ablations

To facilitate a fair comparison for different initialization methods we estimate the FLOPs associated with spatialtemporal computation in the transformer for various model training steps (Tables 13 and 14). When controlling for FLOPs we take into account, the compute used to pretrain the intermediate models, the reduction in an effective number of tokens due to masking in the relevant attention blocks, the increased parameter count when temporal attention is included, and the increased number of tokens passed to the model during high resolution training. For both the high resolution and higher frame count experiments we attebyto keep the compute budget across model initialization ablations similar. Tables 15 and 16 show the VBench quality metrics for high resolution and high frame count XL sized models respectively.

|Init. Method<br><br>|Models|Stage 1<br><br>|Stage 2<br><br>|Stage 3|Stage 4<br><br>|Total|
|---|---|---|---|---|---|---|
|Scratch T2V-256<br><br>T2I-512 Both|T2V-512<br><br>T2I-256, T2V-256, T2V-512<br><br>T2I-256, T2I-512, T2V-512<br><br>T2I-256, T2V-256, T2I-512, T2V-512<br><br>|5.93 1.11 1.11 1.11|2.05 8.43 2.05<br><br>|2.84 4.02 8.43|1.98<br><br>|5.93 6.00 5.97 5.98|

- Table 13. A breakdown of FLOPs for training high resolution T2V models. Unit 1021.

|Init. Method<br><br>|Models|Stage 1|Stage 2<br><br>|Stage 3|Total|
|---|---|---|---|---|---|
|T2I T2V (int.) T2V (ext.) T2V 2x (int.)|T2I-256, T2V-256-40<br><br>T2I-256, T2V-256-20, T2V-256-40 T2I-256, T2V-256-20, T2V-256-40 T2I-256, T2V-256-20 2x stride, T2V-256-40<br><br>|1.11 1.11 1.11 1.11<br><br>|2.05 1.02 1.02 1.02<br><br>|1.02 1.02 1.02<br><br>|3.16 3.16 3.16 3.16|

- Table 14. A breakdown of FLOPs for training high frame count T2V models. Unit: 1021.

|Initial Method|Subject Cons.<br><br>|Background Cons.<br><br>|Temporal Flickering|Motion Smoothness<br><br>|Dynamic Degree<br><br>|Aesthetic Quality<br><br>|Imaging Quality|Object Class<br><br>|
|---|---|---|---|---|---|---|---|---|
|Scratch T2V-256<br><br>T2I-512 Both|93.1 91.9 92.3 92.4<br><br>|97.1 97.1 97.2 97.3|97.9 98.0 98.2 98.3<br><br>|97.3 97.5 97.0 97.4|61.4 58.6 52.2 53.9<br><br>|58.6 59.4 60.0 60.7|58.6 59.7 59.3 60.6<br><br>|87.0 91.2 88.8 88.2|

|Initial Method|Multiple Objects<br><br>|Human Action|Color<br><br>|Spatial Relationship|Scene<br><br>|App. Style|Temp. Style<br><br>|Overall Cons.|
|---|---|---|---|---|---|---|---|---|
|Scratch T2V-256<br><br>T2I-512 Both<br><br>|29.7 45.7 47.4 49.7|95.4 95.8 96.4 96.0<br><br>|88.3 89.0 87.9 88.1|33.8 36.3 37.0 36.7<br><br>|46.9 50.0 49.1 52.3<br><br>|21.6 21.9 22.5 22.8<br><br>|25.8 25.8 26.2 26.3|26.4 27.3 27.8 28.0<br><br>|

- Table 15. Detailed VBench metrics of different model initialization methods for higher resolution T2V model training.

|Initial Method|Subject Cons.<br><br>|Background Cons.<br><br>|Temporal Flickering|Motion Smoothness<br><br>|Dynamic Degree|Aesthetic Quality<br><br>|Imaging Quality<br><br>|Object Class|
|---|---|---|---|---|---|---|---|---|
|T2I T2V (int.) T2V (ext.)<br><br>T2V 2x (int.)|93.2 91.7 91.3 91.0<br><br>|98.1 97.7 97.5 97.3|98.7 97.7 97.8 97.2<br><br>|95.2 96.8 96.9 97.0<br><br>|57.8 64.7 58.6 70.3<br><br>|54.2 54.7 54.6 54.1|58.2 59.2 60.0 59.4<br><br>|84.6 86.9 86.1 85.8<br><br>|

|Initial Method<br><br>|Multiple Objects<br><br>|Human Action|Color<br><br>|Spatial Relationship<br><br>|Scene<br><br>|App. Style<br><br>|Temp. Style<br><br>|Overall Cons.|
|---|---|---|---|---|---|---|---|---|
|T2I T2V (int.) T2V (ext.)<br><br>T2V 2x (int.)<br><br>|30.8 25.5 28.5 29.3|92.2 95.4 95.2 94.0<br><br>|85.0 85.3 84.2 87.7<br><br>|29.9 28.6 25.9 28.6<br><br>|45.2 41.4 36.8 44.2|21.1 21.2 20.9 20.9<br><br>|25.0 25.3 25.6 25.7|26.0 26.6 26.8 26.7<br><br>|

- Table 16. Detailed VBench metrics of different model initialization methods for higher frame count T2V model training.

#### H. Study of Class-to-Video on UCF-101

UCF-101 is an action recognition dataset, which contains 101 classes over 9.5K training videos. Here we train STIV from scratch and perform label-to-video (L2V) generation with 16 frames and 1282 resolution. We follow TATS [22] to adopt the Inception Score (IS) [53] and FVD for the evaluation11.

Tab. 17 shows that our L2V-XL achieves significant improvements, leading to +12% IS and -22% FVD over MAGVIT. This also highlights the effectiveness of our model design for convention video generation. From the ablation study over different modulations, only without spatial mask makes a lower FVD but degrades IS, while all other settings hurt the performance.

11Following our baselines (https://github.com/songweige/TATS/issues/13), we apply C3D [58] pre-trained on UCF-101 for the IS logits. For FVD, we adopt I3D [5] pre-trained on Kinetics-400 [32] to calculate the video embeddings.

|Method<br><br>|IS ↑<br><br>|FVD ↓|VBenchQuality ↑<br><br>|
|---|---|---|---|
|CogVideo [30] TATS [22] MMVG [20] VideoFusion [40] MAGVIT [67]|50.5 79.3 73.7 80.0 83.6<br><br>|626 332 328 173 159<br><br>|-|
|XL-128<br><br>- Spatial Mask<br><br>+ Temporal Mask<br><br>+ Temporal ScaleShiftGate<br><br>+ Causal TemporalAttention<br><br>|93.4 88.5 94.9 78.9 86.9<br><br>|124 102 167 141 106<br><br>|69.9 70.6 68.1 69.1 70.3|

Table 17. Performance of Class-to-Video Generation on UCF-101.

#### I. Details of Video Data Engine

Details of Video Pre-processing and Feature Extraction To ensure high-quality input data, we first address the issue of inconsistent motions and unwanted transitions like cuts and fades in raw videos. Using PySceneDetect12, we analyze video frames to identify and segment scenes with abrupt transitions or gradual fades. This process isolates and removes inconsistent segments, resulting in video clips that maintain visual consistency, reducing artifacts and improving the overall quality. After that, we extract several initial features for future filtering, including motion score, aesthetic score, text area, frame height, frame width, clarity score, temporal consistency, and video orientation, et al.

Details of Video Captioning and Categorization Video-text pairs play a crucial role in training text-to-video generation models. However, many video datasets lack well-aligned, high-quality captions and often include noisy or irrelevant content. Therefore, we’ve incorporated an additional video captioning module in our pipeline to generate comprehensive textual descriptions.

[Figure 241]

We mainly explore two directions: (1) sample a few frames, apply an image captioner, and then use an LLM to summarize the resulting captions [2]; (2) apply a video LLM to generate captions. We initially explored the first direction but found two major limitations. Firstly, the image captioner can only capture visual details in a single frame, resulting in a lack of descriptions of video motions. Secondly, the LLM may hallucinate when prompted to generate a dense caption based on multiple frame captions. Recent works [9, 69, 71, 72] use GPT-4V or GPT-4o to curate a fine-tuning dataset and train their video LLMs. To balance quality and cost in large-scale captioning, we select LLaVA-Hound-7B [69] as our video captioner. Then, we use an LLM to categorize the generated captions and obtain the distribution of videos. Figure 17. Category distribution of our curated Panda-

30M dataset.

12https://github.com/Breakthrough/PySceneDetect

Details of DSG-Video: Hallucination Evaluation To compare various captioning techniques, we develop an evaluation module that assesses both caption richness and accuracy. We quantify caption richness by measuring the diversity of unique objects referenced across captions. And we identify hallucinations to assess accuracy. Inspired by DSG [12] for fine-grained text-to-image evaluation, we introduce DSG-Video to validate the presence of objects referenced in captions within the video content (Fig. 9). First, we use an LLM to automatically generate questions that probe key details of captions, such as object identity, actions, and context. For instance, given a caption mentioning "a cat sitting on a couch", the LLM will generate questions like "Is there a cat in the video?" and "Is the cat on a couch?". Second, we utilize a multimodal LLM to answer these object verification questions by evaluating the presence of each referenced object in N uniformly sampled frames from the video. For each question generated (e.g., "Is there a cat in this frame?"), the multimodal LLM examines each sampled frame and provides a response. If, for a given question, all frame-based responses suggest that the entity in question is absent, we classify the object as hallucinated. This approach ensures a thorough and frame-by-frame validation for each object in the video. Based on this, we have two metrics: one that measures the fraction of hallucinated object instances (referred to as DSG-Videoi), and the other that calculates the fraction of sentences containing hallucinated objects (referred to as DSG-Videos). We use these two metrics to assess caption quality.

DSG-Videoi = |{hallucinated objects}|

,

|{all objects mentioned}|

DSG-Videos = |{sentences with hallucinated object}|

.

|{ all sentences}|

#### J. Details for Flexible Applications

##### J.1. Video Prediction

We start from a STIV-XXL model to train a text-video-to-video model. Specifically, we replace the first four frames with ground truth frame latents to train additional 400K steps. The results are shown in the main text.

##### J.2. Frame Interpolation

We start from a STIV-XL model to train a text-video-to-video model. Specifically, we use a stride of two to sample the frames with ground truth frame latents to train additional 400K steps. The results are shown in the main text.

##### J.3. Multi-view Generation

Our model employs temporal attention to model cross-frame correspondence. However, this approach may be less effective than full-attention mechanisms, as temporal attention focuses solely on the same spatial position across all frames. To investigate this limitation, we conducted experiments on multiview generation, which involves predicting novel views of a specific object. For evaluation, we adopted the six-view scheme from Zero123++, consisting of elevation angles of 30◦,−20◦,30◦,−20◦,30◦,−20◦ and azimuth angles of 30◦,90◦,150◦,210◦,270◦,330◦. This scheme was chosen because the large camera changes between adjacent views simulate a challenging scenario where the camera rotates around an object with significant motion, providing a robust test for the temporal attention mechanism’s ability to capture large inter-frame motions.

For implementation, we redefined the input frames in our text-image-to-video model as the given views of the object, with an empty string as the text conditioning. The negative input views were set to zero latent vectors during classifier-free guidance training. For the generation, we used a guidance scale of 3.5. A visual comparison between our approach and Zero123++ is shown in Fig. 11.

#### K. More Examples

We show more examples at the end of the Appendix using the text prompts and image as first frame condition borrowed from MovieGenBench [46] and Sora [42].

|Text-to-Video|
|---|

Prompt: A pirate ship sailing through a storm with enormous waves crashing against the sides, its crew fighting against the wind as lightning illuminates the scene.

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Prompt: A samurai on horseback charging across a field of cherry blossoms, slicing petals in mid-air as they fall, leaving a trail of pink in their path.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Prompt: Two pigs are eating a hotpot.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Prompt: Giant Pandas are eating hot noodles in a Chinese restaurant.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

|Text-to-Video|
|---|

Prompt: A zoom-in on a clock face, focusing on the intricate movement of the hands and the ticking mechanism inside.

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Prompt: Robots move efficiently through a futuristic laboratory, adjusting holographic displays and conducting experiments, while scientists observe and interact with the high-tech equipment.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Prompt: A robotic arm wielding a glowing sword, battling a shadowy figure in a high-tech dojo, each strike creating sparks that light up the space.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Prompt: A city skyline reflected in the water, but the reflection shows an alternate world with flying cars, towering robots, and futuristic architecture.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

|Text-to-Video|
|---|

Prompt: A dog dressed as a chef, expertly flipping pancakes in a kitchen.

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Prompt: A motocross bike accelerating out of a tight turn on a dirt track.

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Prompt: A snowboarder performing a dramatic backflip over a frozen lake, landing gracefully and leaving a trail of sparkling ice dust in the air.

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Prompt: A surfer accelerating on a wave, carving through the water.

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

|Text-to-Video|
|---|

Prompt: A person dancing with their own shadow, which has come to life.

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Prompt: A bobsled team accelerating down an icy track.

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Prompt: A cyclist accelerating out of the saddle during a steep climb.

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Prompt: A speed skater accelerating during a short track race.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

|Text-Image-to-Video|
|---|

Prompt: Reflections in the window of a train traveling through the Tokyo suburbs.

|[Figure 306]|
|---|

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

###### Reference Image

Prompt: The Glenfinnan Viaduct is a historic railway bridge in Scotland, UK, that crosses over the west highland line between the towns of Mallaig and Fort William. It is a stunning sight as a steam train leaves the bridge...

|[Figure 311]|
|---|

###### Reference Image

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Prompt: The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV...

|[Figure 316]|
|---|

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Reference Image Prompt: Photorealistic closeup video of two pirate ships battling each other as they sail inside a cup of coffee.

|[Figure 321]|
|---|

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

###### Reference Image

|Text-Image-to-Video|
|---|

Prompt: A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in.

|[Figure 326]|
|---|

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Reference Image Prompt: An adorable happy otter confidently stands on a surfboard wearing a yellow lifejacket, riding along turquoise tropical waters near lush tropical islands, 3D digital render art style.

|[Figure 331]|
|---|

###### Reference Image

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Prompt: A dog dressed as a chef, expertly flipping pancakes in a kitchen.

|[Figure 336]|
|---|

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Reference Image Prompt: A skeleton wearing a flower hat and sunglasses dances in the wild at sunset.

|[Figure 341]|
|---|

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

###### Reference Image

Prompt: The video features a central spacecraft with a predominantly white and gray color scheme, accented with red and black details. It has a sleek, angular design with multiple protruding elements that suggest advanced technology...

|[Figure 346]|
|---|

###### Reference Image

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Prompt: The video begins with a dark space background, dotted with stars, and a central object that appears to be a spacecraft with a glowing blue light at its core. The spacecraft is detailed with various components...

|[Figure 351]|
|---|

###### Reference Image

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Prompt: Robots move efficiently through a futuristic laboratory, adjusting holographic displays and conducting experiments, while scientists observe and interact with the high-tech equipment.

|[Figure 356]|
|---|

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

Reference Image Prompt: The video presents a serene scene with a group of camels walking in a line across a desert landscape.

The camels are adorned with colorful saddles and are led by a person wearing a green garment. The background features a clear sky...

|[Figure 361]|
|---|

###### Reference Image

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

|Text-Image-to-Video|
|---|

Prompt: A crab made of different jewlery is walking on the beach. As it walks, it drops different jewelry pieces like diamonds, pearls, etc.

|[Figure 366]|
|---|

###### Reference Image

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Prompt: The video captures a single sea turtle with a patterned shell and flippers, swimming in a clear blue underwater environment. The turtle moves gracefully over a bed of coral reefs, which exhibit a variety of colors...

|[Figure 371]|
|---|

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

Reference Image Prompt: A mesmerizing video of a jellyfish moving through water, with its tentacles flowing gracefully.

|[Figure 376]|
|---|

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Reference Image Prompt: A video of a diver creating bubbles underwater, with bubbles rising and interacting with each other.

|[Figure 381]|
|---|

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

###### Reference Image

Prompt: The individual in the video is dressed in a blue protective suit with a hood, a mask with a filter, and white gloves. They are holding a spray bottle in one hand and a spray nozzle in the other...

|[Figure 386]|
|---|

###### Reference Image

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

Prompt: The video captures a bustling city street scene during the evening. The sky is overcast, and the street is wet, reflecting the lights from the vehicles and buildings. The buildings are tall with modern architecture...

|[Figure 391]|
|---|

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

Reference Image Prompt: The video presents a series of images capturing the Colosseum from an aerial perspective during the evening. The ancient amphitheater is illuminated by artificial lighting, which highlights its circular shape and the arches...

|[Figure 396]|
|---|

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

Reference Image Prompt: The video features two dogs, one with a predominantly white coat and the other with a mix of black, brown, and white fur. Both dogs are adorned with accessories; the white dog wears a red tie, while the other sports a purple bow tie...

|[Figure 401]|
|---|

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

###### Reference Image 22

