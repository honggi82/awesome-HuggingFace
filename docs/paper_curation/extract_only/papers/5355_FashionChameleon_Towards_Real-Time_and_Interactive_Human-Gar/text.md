# arXiv:2605.15824v2[cs.CV]17Jun2026

Taobao Xiamen University

[Figure 1]

[Figure 2]

## FashionChameleon: Towards Real-Time and Interactive Human-Garment Video Customization

Quanjian Song1, 2, Yefeng Shen2, Mengting Chen2,∗, Hao Sun2,

Jinsong Lan2, Xiaoyong Zhu2, Bo Zheng2, Liujuan Cao1 1Xiamen University 2Alibaba Group

Project Page:

[Figure 3]

### Abstract

Human-centric video customization, particularly at the garment level, has shown significant commercial value. However, existing approaches cannot support low-latency and interactive garment control, which is crucial for applications such as e-commerce and content creation. This paper studies how to achieve interactive multi-garment video customization while preserving motion coherence using only single-garment video data. We present FashionChameleon, a real-time and interactive framework for human-garment customization in autoregressive video generation, where users can interactively switch garment during generation. FashionChameleon consists of three key techniques: (i) Instead of training on multi-garment video data, we train a Teacher Model with In-Context Learning on a single reference–garment pair. By retaining the image-to-video training paradigm while enforcing a mismatch between the reference and garment image, the model is encouraged to implicitly preserve coherence during single-garment switching. (ii) To achieve consistency and efficiency during generation, we introduce Streaming Distillation with In-Context Learning, which fine-tunes the model with in-context teacher forcing and improves extrapolation consistency via gradient-reweighted distribution matching distillation. (iii) To extend the model for interactive multi-garment video customization, we propose Training-Free KV Cache Rescheduling, which includes garment KV refresh, historical KV withdraw, and reference KV disentangle to achieve garment switching while preserving motion coherence. Our FashionChameleon uniquely supports interactive customization and consistent long-video extrapolation, while achieving real-time generation at 23.8 FPS on a single GPU, 30-180× faster than existing baselines.

Date: May 18, 2026

Garment1 Garment2

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

[Figure 14]

[Figure 15]

Reference

|[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

… … … … … …

𝑡 = 0 𝑡 = 𝑇

Start

Switch

Coherent Motion

Coherent Motion

[Figure 25]

♾

Streaming

23.8 FPS

Garment3

Garment4

[Figure 26]

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

|[Figure 39]|
|---|

|[Figure 40]<br><br>[Figure 41]|
|---|

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

… …

… … … …

𝑡 = 𝑇

𝑡 = 𝑇

Switch

Switch

Coherent Motion

Coherent Motion

Figure 1: Given a reference image and a sequence of garment images, FashionChameleon generate customized videos in a streaming and interactive manner, where users can interactively switch garments during generation while preserving coherent motion, achieving 23.8 FPS real-time generation.

*Project leader.

### 1 Introduction

Driven by advances in diffusion models [1, 2], text-to-video and image-to-video generation [3, 4, 5] have become prominent directions. However, these approaches condition only on a simple prompt or an initial frame, which limits their applicability in real-world scenarios [6, 7, 8]. To overcome this limitation, recent work has explored various customized video generation, in which visual concepts are injected into the generation process through user-provided reference images. One representative setting is subject-to-video (S2V) [9, 10, 11, 12, 13, 14, 15] customization, which aims to ensure that subjects in generated videos remain consistent with the given reference images. With the advances of Diffusion Transformers (DiT) [16, 3, 4, 5], subsequent works [17, 18, 19, 20] extend S2V customization to multi-reference settings, enabling more flexible control in complex scenes.

Despite this progress, existing customization methods mainly focus on human-centric subject consistency, with comparatively less emphasis on fine-grained human attributes. Among these attributes, garment-level customization is particularly desirable in practical applications such as filmmaking [21, 22], e-commerce [23] and entertainment [24, 25, 26], where users often require low-latency, streaming, and interactive control over garments. Given the recent success of hybrid autoregressive generation [27, 28, 29] in diverse domains [30, 31, 32], we are inspired to ask: Can this paradigm be extended to the customization domain? In this work, we formulate streaming and

0.85

AveragePerformance(Score)

| | | | | | | | |High Perf / High (Target Region)|FPS|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.80

0.75

0.70

0.65

0.60

0.55

0.50

0.45

0.1 0.2 0.5 1 5 10 30

Inference Speed (FPS)

Figure 2: Average performance (Cur., GME, Amp., Smoo., and VQ) and inference speed comparison across different approaches.

interactive human-garment video customization and pinpoint three key challenges: (i) Single-tomultiple generalization. Video data with multi-garment switching are typically difficult to obtain. How to effectively exploit single-garment data for interactive multi-garment video customization remains a significant challenge. (ii) Consistency and efficiency. Although distillation from bidirectional to autoregressive generation improves inference efficiency, it also introduces error accumulation during self-rollout. In human-centric scenarios, it is important to maintain identity and motion consistency while achieving efficiency during streaming generation. (iii) Coherent interaction. Interactive video customization requires dynamically switching a character’s garments during generation. Ensuring seamless garment transitions while preserving continuous human motion remains challenging.

In this paper, we introduce FashionChameleon, a real-time and interactive framework that enables human-garment customization in autoregressive video generation (see Figure 1), where users can interactively switch garments during generation while maintaining coherent human motion.

- (i) Rather than directly training a teacher model on multi-garment video data, we train a Teacher Model with In-Context Learning to process a reference image paired with a garment image. Notably, we retain the image-to-video training paradigm while ensuring that the garment worn by the reference person differs from the target garment. This enables the model to implicitly preserve coherence during single-garment switching, laying the foundation for interactive multi-garment switching.
- (ii) To achieve consistency and efficiency during streaming video generation, we introduce Streaming Distillation with In-Context Learning. Specifically, it fine-tunes the model with in-context teacher forcing to eliminate the data-intensive ODE initialization, and incorporates gradient-reweighted distribution matching distillation to improve consistency in long-video extrapolation.
- (iii) To extend the model for interactive multi-garment video customization, we propose TrainingFree KV Cache Rescheduling. Specifically, it first perform garment KV refresh to switch garments during inference, then apply historical KV withdraw to suppress outdated garment in historical frames, and utilize reference KV disentangle to preserve coherent human motion during garment-switching.

To further support teacher model pre-training and streaming distillation post-training, we propose a high-quality data curation pipeline with four stages: general coarse-to-fine video filtering, staticdynamic video captioning, fine-grained garment image extraction, and adaptive reference image extraction. Qualitative and quantitative experiments on the proposed HGC-Bench show that our FashionChameleon is superior to existing baselines while achieving real-time 720p customization at

23.8 FPS on a single H200 GPU (see Figure 2). Additional experiments on interactive multi-garment video customization and consistent long-video extrapolation further highlight its unique capabilities.

### 2 Related Works

Subject-to-Video Customization. Subject-to-Video (S2V) aims to preserve subjects specified by reference images for customized video generation. Early approaches [9, 10] rely on few-shot tuning, while later works [11, 12] improve generalization by fine-tuning U-Net-based models. With the rise of diffusion transformers (DiT) [16, 33], subsequent methods [14, 19, 15, 13] focus on human-centric customization, with improved identity preservation, editing flexibility, and text-image alignment. Recent works extend this paradigm to multi-reference customization: MAGREF [18] supports anyreference generation via subject disentanglement, while BindWeave [17] and Kaleido [20] improve multi-entity grounding and reference integration in complex scenes. Despite this progress, they suffer from high inference latency and limited interactivity, which are crucial for practical user experience. In contrast, our FashionChameleon achieves real-time and interactive customization.

Hybrid Autoregressive Video Generation. Recent hybrid autoregressive video generation methods [34, 27, 28, 29] combine diffusion-based frame modeling [4, 3, 5] with autoregressive prediction across frames [35, 36], balancing fidelity and efficiency. CausVid [27] leverages distribution matching distillation (DMD) [37] to distill a slow bidirectional teacher into a few-step autoregressive student, avoiding training from scratch. Furthermore, Self Forcing [28] conditions the model on its own rolled-out frames instead of ground-truth frames, thereby fundamentally solving the traininginference mismatch. Building on this paradigm, Rolling Forcing [38] accelerates inference, Reward Forcing [39] improves motion dynamics, Infinity-RoPE [40] enables stable long-video generation, and Causal Forcing [29] reduces distribution mismatch during ODE initialization.

Applications of Streaming Video Generation. Benefiting from low latency and interactive inference, hybrid autoregressive generation has been adopted in various downstream tasks. LiveAvatar [31], FlashVSR [30], MotionStream [32], and LongLive [41] extend this paradigm to audio-driven avatar generation, video super-resolution, interactive motion-controlled generation, and interactive prompt-controlled generation, respectively. More recently, popular video world models, such as Vid2World [42], Yume [43], WorldPlay [44], and Matrix-Game [45] further exploit it for interactive virtual worlds. However, these works mainly consider continuous control signals such as audio, motion, or mouse/keyboard inputs. To the best of our knowledge, no research has yet explored streaming applications in customized video generation tasks, particularly those involving discrete control signals like garment manipulation. Our work seeks to address this gap.

### 3 Preliminary

Video Diffusion Models. The advanced video diffusion generation typically consists of a variational encoder–decoder pair ⟨E,D⟩ along with a transformer-based predict network vθ. During training, the encoder E transforms a video with F frames into a latent sequence z1:0 f with f frames, where f = F−4 1 +1. According to flow matching [2], the forward process is defined as a linear interpolation between the data distribution and a standard normal distribution, as follows:

zt1:f = (1 − t) · z01:f + t · ϵ1:f, (1)

where t is a random timestep and ϵ1:f ∼ N(0,I). For the noisy latent zt1:f, we utilize the predict network vθ to regress the conditional vector field via conditional flow matching [2] loss:

Et∼U(0,1)∥vθ(zt1:f,t,c) − v∥22, (2)

min

θ

where v = ϵ1:f − z01:f denotes the target vector field, and c represents the conditional signals.

Hybrid Autoregressive Video Generation. Given a video with F frames V1:F = ⟨V1,V2,...,VF⟩, CausVid [27] proposes to factorizes the joint distribution as p(V1:F) = Fi=1 p(Vi | V<i), where each conditional distribution p(Vi | V<i) is modeled by the diffusion models where each frame/chunk

Legends

Teacher Forcing with In-Context Learning

KV Cache Manage

Condition

| |
|---|

🔥 Trainable

[Figure 48]

Reference Token/KV

Noisy Token

Sink First-In and First-Out

[Figure 49]

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Garment Token/KV

Mask Frame/Chunk

Frozen

CleanNoisy

Reference

Garment

History

Video Token/KV

Unmask Frame/Chunk

SharedVAE

…

Garmen

|t<br><br>[Figure 50]| |
|---|---|
| | |

Garment KV Refresh

Refresh

New

VAE

t

Reference Image Garment Image Video

|[Figure 51]|[Figure 52]|[Figure 53]|
|---|---|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

[Figure 56]

🔥

Diffusion Transformer

In-Context Teacher Forcing Mask

New Garment KV Historical KV with Old Garment

Suppress

[Figure 57]

Low Attention Weight High Attention Weight

Historical KV Withdraw

Gradient-Reweighted Distribution Matching Distillation

High Attention Weight

Shared VAE

[Figure 58]

| |
|---|

∇𝓛Reweight−DMD

[Figure 59]

🔥

Real Score

[Figure 60]

+Noise

Generator

Directly Replace to Preserve Coherence

DiffusionTransformer

- ·

[Figure 61]

Self Rolling

Mismatch with Training Distribution

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

+Noise

Reference KV Disentangle

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

🔥

Reweight

…

Fake Score

[Figure 71]

🔥

[Figure 72]

Match with Training Distribution Preserve Coherence

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

VAE

VAE

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Reward Model Normalize

Multi-Model Attention

[Figure 83]

Frame/Chunk

Teacher Model with In-Context Learning Streaming Distillation with In-Context Learning Training-Free KV Cache Rescheduling

- Figure 3: Overall pipeline of FashionChameleon: Teacher Model with In-Context Learning, Streaming Distillation with In-Context Learning, and Training-Free KV Cache Rescheduling.

is generated autoregressively. Self-Forcing [28] further improves this paradigm with self-rolling, conditioning on self-generated rather than ground-truth history to better align training with inference. To avoid training from scratch, most methods distill multi-step bidirectional teacher models into fewstep autoregressive student models via Distribution Matching Distillation (DMD) [37]. Specifically, DMD minimizes an approximate KL divergence between the student distribution estimated by sfake and the data distribution estimated by sreal. This process can be formulated as follows:

dGθ(ϵ) dθ

dϵ , (3)

∇LDMD = −Et sreal(ϕ(G(ϵ),t),t) − sfake(ϕ(G(ϵ),t),t) ·

where ϵ ∼ N(0,I), Gθ denotes student model, and ϕ(·,t) represents forward diffusion at timestep t defined in Eq.1. During distillation, Gθ and sfake are updated while sreal remains frozen.

### 4 Methodology

In this work, we propose FashionChameleon, a real-time and interactive framework that enables human-garment customization in autoregressive video generation. Given a reference image Isrc and a sequence of N garment images ⟨Igar1,...,IgarN

⟩, our goal is to generate videos in a streaming manner, where each garment is applied to the character at different moments while ensuring coherent human motion. In Sec.4.1, we first train a Teacher Model with In-Context Learning conditioned on a reference image and a single garment image. In Sec. 4.2, we introduce Streaming Distillation with In-Context Learning, featuring an in-context teacher forcing mask technique for stable training and a gradient-reweighted distribution matching distillation strategy to improve extrapolation consistency. In Sec. 4.3, we propose Training-Free KV Cache Rescheduling, which consists of garment KV refresh, historical KV withdraw, and reference KV disentangle, enabling seamless garment switching while maintaining motion coherence. In Sec.4.4, we develop a High-Quality Data Curation Pipeline to further support training. The overall pipeline of FashionChameleon is shown in Figure 3.

#### 4.1 Teacher Model with In-Context Learning

To enable real-time and interactive human-garment video customization, we first train a bidirectional teacher model conditioned on a reference image and a single garment image. Unlike prior works [31, 30, 32] that rely on auxiliary encoders to process continuous signals, we adopt in-context learning within a unified backbone network to process discrete reference and garment images, eliminating the auxiliary encoders. Notably, we retain the image-to-video (I2V) training property, such that the first generated frame stays consistent with the reference frame, except for the garment information. Meanwhile, we ensure that the garment worn by the reference person differs from the target garment. This implicitly enables the model to learn single-garment switching while maintaining coherence.

Shared Latent Space with Varying Noise Levels. During training process, a given video V is encoded into a latent representation z0v by the VAE encoder E. Instead of introducing an additional encoder, we reuse E to separately encode the reference image Isrc and the garment image Igar into latent representations z0src and z0gar. The whole process can be formulated as follows:

z0v = E(V); z0src = E(Isrc); z0gar = E(Igar). (4) In this way, all latents can share semantic space without introducing additional parameters. Subsequently, the video latent z0v is noised according to the flow-matching defined in Eq.1, while the reference latent z0src and garment latent z0gar remain noise-free as conditional inputs.

Multi-Modal Attention. To enable multi-modal interaction within a single backbone, the clean reference latent zsrc0 , clean garment latent zgar0 , and noisy video latent zvt are concatenated along the token dimension. The resulting sequence ztuni is then projected via learnable matrices Wq, Wk, and Wv, followed by multi-modal attention interaction. The attention output O can be formulated by:

(Wq · ztuni)(Wk · ztuni)⊤ √dk

)(Wv · ztuni), (5)

O = Softmax(

where dk denotes the feature dimension. These shared projection matrices enables global interaction between conditional and video latents without introducing additional parameters. Finally, the model output retains only the video latent, discarding the reference latent and garment latent.

#### 4.2 Streaming Distillation with In-Context Learning

In this section, we distill the pretrained teacher into a few-step autoregressive student for streaming generation. Prior works [27, 28, 29] show that direct distillation is challenging and adopt a two-stage strategy comprising ODE initialization and distribution matching distillation [37]. To better adapt to our setting, we instead adopt teacher forcing [46, 47, 48] to initialize the student model, followed by gradient-reweighted distribution matching distillation to improve extrapolation consistency.

In-Context Teacher Forcing Mask. The teacher forcing fine-tunes the pretrained multi-step bidirectional model into a multi-step autoregressive model using clean data. However, unlike prior approaches [31, 30, 32] that inject control signals via adapters, our model incorporates these signals through in-context token concatenation, making standard teacher forcing inapplicable. To address this, we design an in-context teacher forcing mask for training, with the toy examples shown in Figure 3. Specifically, in addition to the noisy sequence ⟨z0src,z0tar,ztv⟩, we symmetrically concatenate its clean counterpart ⟨z0src,z0tar,z0v⟩ and feed the resulting sequence into the model. For the conditioning signals z0src and z0tar, we apply a dedicated masking strategy such that all generated frames can attend to them, while z0src and z0tar cannot access any future generated frames. In this way, when predicting the next frame (chunk), model conditions on ground-truth historical frames and conditional signals.

Gradient-Reweighted Distribution Matching Distillation. Based on the autoregressive model finetuned with teacher forcing, we further apply distribution matching distillation (DMD) for few-step generation and combine it with Self-Forcing [28] to better align training with inference. However, we observe that directly applying DMD often leads to distorted human motions during extrapolation. We attribute this to the unequal difficulty of frames in self-rolling generation: errors accumulate over time, making later frames more prone to drift, whereas vanilla DMD weights all frames equally. To resolve this, we propose an adaptive gradient reweighting strategy that increases the weights of low-quality frames while decreasing those of high-quality ones during distillation. Specifically, we use an aesthetic reward model R to estimate frame quality during distillation and normalize the resulting scores into frame-wise gradient weights. In this way, the Eq.3 can be rewritten as:

dGθ(ϵ) dθ · dϵ ,

∇LReweight-DMD = −Et A1:f(G(ϵ)) · s1:realf(ϕ(G(ϵ),t),t) − s1:fakef (ϕ(G(ϵ),t),t) ·

exp(−R(Gi(ϵ))/τ)

Ai(G(ϵ)) =

, i = 1,...,f,

f j=1 exp(−R(Gj(ϵ))/τ)

(6) where τ denotes the temperature coefficient that controls the relative weight. Note that this approach is not restricted to aesthetic rewards and can naturally accommodate other reward models.

Reference Generated Sequences Reference History

Directly Refresh Ours Garment

|[Figure 84]|
|---|

|[Figure 85]|
|---|

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

h

###### ...

Garment

Switch

|[Figure 91]|
|---|

w Attention Map across Frames

(k+1)-th frame

1-st frame k-th frame

- Figure 4: (Left) Generated sequences during garment switching. Directly refreshing the garment KV fails to change the subject’s clothing, while our KV cache rescheduling enables garment-switching and motion coherence. (Right) Average attention visualization of newly generated frames over historical and conditional KV. The model attends more to historical KV than to conditional KV.

#### 4.3 Training-Free KV Cache Rescheduling

Given the distilled few-step autoregressive models, we manage KV cache to enable stable long-video extrapolation. In detail, the reference KV entry KV src and garment KV entry KV gar are persistently stored in the KV cache as conditioning signals. Following prior work [41, 40], we also retain the KV entries of the initial frame (chunk), KV 0, as an attention sink to improve stability during extrapolation. All remaining KV entries follow a first-in and first-out policy when the cache exceeds its maximum size. Formally, at the generation of k-th frame, the KV cache is defined as:

KV Cache := ⟨KV src,KV gar,KV 0,KV Max(1, k - M + 4),...,KV k⟩, (7)

where M is the maximum KV cache size. To enable interactive multi-garment switching while maintaining coherence, we reschedule the KV cache via three mechanisms: Garment KV Refresh, Historical KV Withdraw, and Reference KV Disentangle, as illustrated in Figure 3 (right).

Garment KV Refresh. To switch the character with a new garment Igar2 during generation, we refresh the garment KV in the cache. Specifically, Igar2 is encoded into zgar2 by VAE, and the corresponding KV gar2 are obtained via a forward pass. We then replace the old KV gar in the cache with new new KV gar2, so that subsequent frames are generated conditioned on the updated garment.

Historical KV Withdraw. However, as shown in Figure 4 (left), directly refreshing garment KV is insufficient to change the garment in subsequent generated frames. To analyze this phenomenon, we visualize the average attention weights of newly generated latents over conditional and historical KV. In Figure 4 (right), attention is more concentrated on historical KV rather than conditional KV. This indicates that, under streaming eneration with in-context learning, the model relies more on historical context than on conditional signals. Consequently, the old garment from historical frames tends to persist in newly generated frames, rendering the new garment signal ineffective. Therefore, we withdraw the historical KV, encouraging the model to focus on the new garment KV.

Reference KV Disentangle. While withdrawing historical KV enables garment switching, it weakens temporal coherence across the switching frame. Recall that we deliberately I2V property during pre-training, in which the first generated frame remains consistent with the reference frame except for garment information. This endows the model with an implicit capability to maintain temporal coherence during single-garment switching. To enable multi-garment switching during generation, the key is to align the distribution of the new conditioning signal with that of the original conditioning signal. To this end, we replace old KV src with the KV k extracted from the last historical frame. Notably, the new reference KV corresponds to four decoded frames, mismatching with the old reference KV that corresponds to single-frame. We thus perform a VAE decode-encode process to disentangle the last decoded frame, followed by an additional forward to obtain new reference KV.

#### 4.4 High-Quality Data Curation Pipeline

To further support teacher model pre-training and streaming distillation post-training, we design a data curation pipeline to construct samples of the reference image Isrc, garment image Igar, video sequence V and corresponding prompt. The pipeline consists of four stages: 1. General Coarse-to-Fine Video Filtering, 2. Static-Dynamic Video Captioning, 3. Fine-Grained Garment Images Extraction, and 4. Adaptive Reference Images Construction. We provide implementation details in the Appendix.

- Table 1: Quantitative comparison of different methods for short (81 frames) video customized generation. The best results are highlighted in bold and the second best are underlined. Note that the frames per second (FPS) of all methods are evaluated on an H200 GPU.

Methods Params ↓ Cur. ↑ GME ↑ Amp. ↑ Smoo. ↑ VQ ↑ HGC ↑ LGC ↑ NTP ↑ FPS ↑

Edit [49]+I2V [5] 20B+5B 0.4094 0.6741 0.8636 0.9898 0.7482 4.5417 3.9167 4.4583 0.76 VACE [14] 14B 0.2746 0.6962 0.4054 0.9764 0.7409 4.3708 3.5458 4.6417 0.23 Kaleido [20] 14B 0.3676 0.6882 0.2675 0.9935 0.7478 4.1708 3.5500 4.7167 0.13 MAGREF [18] 14B 0.0459 0.7138 0.2571 0.9436 0.7301 3.6000 2.2000 2.6875 0.27 SkyReels-A2 [19] 14B 0.3689 0.6550 0.5205 0.9424 0.7241 3.3625 2.6958 4.6458 0.54 Phantom [13] 1.3B 0.5507 0.6855 0.1144 0.9668 0.7338 4.3292 3.6417 4.6875 0.77 Phantom [13] 14B 0.4911 0.6972 0.2086 0.9932 0.7446 4.5375 3.8333 4.6417 0.15 FashionChameleon 5B 0.4911 0.6839 0.7771 0.9969 0.7483 4.6833 3.9250 4.7625 23.8

### 5 Experiments

#### 5.1 Experimental Details.

Implementation Details. Our teacher model is initialized with WAN2.2-5B-TI2V [5]. During streaming distillation, we use an aesthetic scorer as the reward model, with the temperature coefficient τ set to 0.2. During inference, the KV cache size M = 23. We adopt a chunk-wise generation strategy, where each chunk consists of 3 latent frames. All experiments are conducted on NVIDIA A100 GPUs. Due to space limitations, we provide additional training details in the Appendix.

Evaluation Settings. The task most closely related to ours is multi-reference customized video generation. Accordingly, we select several representative baselines: VACE [14], Kaleido [20], MAGREF [18], SkyReels-A2 [19] and Phantom [13]. Moreover, we compare with a first-frame editing + Image-to-Video (I2V) pipeline, where Qwen-Image-Edit [49] performs editing, followed by WAN-5B-TI2V [5] for I2V generation. Note that all baselines generate videos at their respective native resolutions and durations. To evaluate different methods on the human-garment video customization task, we construct a benchmark termed HGC-Bench. HGC-Bench contains 240 samples, each consisting of a reference character image, a garment image, and a corresponding prompt, covering a wide range of characters, scenarios, and garments. We provide additional details in the Appendix.

#### 5.2 Main Results

Quantitative Comparisons. Inspired by prior works [18, 13, 15], we adopt several evaluation metrics, including ID consistency (Cur Score), text alignment (GME Score), motion magnitude (Amplitude), and temporal smoothness (Smoothness) following OpenS2V-Nexus [50], as well as overall visual quality (VQ Score) following VBench [51]. To assess garment consistency, we use Gemini-3.0 to evaluate the generated results from three aspects: high-level garment consistency (HGC), low-level garment consistency (LGC), and non-target garment preservation (NTP). In addition, we report the frames per second (FPS) of each method to measure efficiency. See Appendix for details. In Table 1, FashionChameleon outperforms all baselines in temporal consistency, video quality, and three garment consistency metrics. For ID consistency and motion magnitude, our method ranks second, following the Phantom(1.3B) [13] and Edit [49]+I2V [5], respectively. Notably, FashionChameleon significantly outperforms all baselines in efficiency, enabling real-time generation at 23.8 FPS.

Qualitative Comparisons. We further provide qualitative comparisons to assess ID consistency, garment consistency, and overall visual fidelity across different methods. As shown in Figure 5, existing approaches often struggle to simultaneously maintain subject identity, garment details, and natural motions. In cases involving large pose variations or with complex garments, these methods tend to exhibit noticeable degradation in appearance and garment preservation. Moreover, several baselines exhibit garment mismatch or unintended modifications to non-target garments, which degrade overall realism and temporal consistency across frames. See Appendix for more results.

Long-Video Extrapolation. Existing multi-reference customization methods rely on bidirectional architectures that synthesize all frames jointly, making them unsuitable for long-video customized

Ours(5B) VACE(14B) Kaleido(14B) MAGREF(14B) SkyReels-A2(14B) Phantom(1.3B) Phantom(14B)

Edit(20B)+I2V(5B)

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Reference

|[Figure 100]|
|---|

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Garment

|[Figure 109]|
|---|

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Reference

|[Figure 118]|
|---|

Garment

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

|[Figure 127]|
|---|

- Figure 5: Qualitative comparison of our FashionChameleon with other baselines. Due to space limitations, we omit the input prompts here; please refer to the Appendix for details.

Reference

Garment

Reference

Garment

1-st Frame 18-th Frame

Interactive Multi-Garment Customization

Long-Video Extrapolation

35-th Frame 52-th Frame 69-th Frame 86-th Frame 103-th Frame 120-th Frame 137-th Frame 154-th Frame

|[Figure 128]|
|---|

|[Figure 129]|
|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

|[Figure 139]|
|---|

Switch

|[Figure 140]|
|---|

Switch

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

|[Figure 151]|
|---|

|[Figure 152]|
|---|

Reference

Garment

Switch Switch

|[Figure 153]|
|---|

|[Figure 154]|
|---|

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

|[Figure 161]|
|---|

[Figure 162]

[Figure 163]

[Figure 164]

|[Figure 165]|
|---|

- Figure 6: Additional applications of FashionChameleon. It supports both long-video extrapolation and interactive multi-garment customization. We omit prompts for brevity; see Appendix for details.

generation. In contrast, the autoregressive generation paradigm of FashionChameleon naturally supports long-video extrapolation. As shown in Figure 6, FashionChameleon can maintain character consistency and garment consistency across long temporal ranges. See Appendix for more results.

Interactive Customization. Benefiting from proposed KV Cache Rescheduling, FashionChameleon further enables interactive multi-garment customized generation, which is beyond the capability of existing methods. As shown in Figure 6, FashionChameleon supports interactive garment-switching during generation while preserving coherent human motion. See Appendix for more results.

1-st Frame 100-th Frame 110-th Frame 120-th Frame 1-st Frame 100-th Frame 110-th Frame 120-th Frame

Reference

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

|[Figure 174]|
|---|

Garment

|[Figure 175]|
|---|

Native DMD

Gradient-Reweighted DMD

Reference

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

|[Figure 184]|
|---|

|[Figure 185]|
|---|

Garment

|[Figure 186]|
|---|

Switch

Random Reference Reference KV w/o Disentangle

Reference KV Disentangle

- Figure 7: Qualitative ablation of Gradient-Reweighted Distribution Matching Distillation (DMD) and Reference KV Disentangle. Gradient-Reweighted DMD alleviates motion collapse during extrapolation, while Reference KV Disentangle further enhances consistency during garment switching.

- Table 2: Quantitative ablation of teacher training strategies for short (81 frames) video customized generation. The best results are highlighted in bold and the second best are underlined.

Variants Cur. ↑ GME ↑ Amp. ↑ Smoo. ↑ VQ ↑ HGC ↑ LGC ↑ NTP ↑ Chan.-Concat + Full FT 0.1811 0.6874 0.3748 0.9266 0.7404 4.4917 3.1667 4.4667 Ours + Full FT 0.4602 0.6972 0.5625 0.9936 0.7473 4.8583 4.1583 4.7792 Ours + Attn FT 0.4348 0.6900 0.6350 0.9881 0.7471 4.8500 4.0625 4.7750 Ours + LoRA [52] FT 0.4046 0.6928 0.6448 0.9777 0.7437 4.7292 3.9458 4.7042

- Table 3: Quantitative ablation of Gradient-Reweighted Distribution Matching Distillation (GR-DMD) for long (165 frames) video customized generation. The best results are highlighted in bold.

Variants Cur. ↑ GME ↑ Amp. ↑ Smoo. ↑ VQ ↑ HGC ↑ LGC ↑ NTP ↑ Naive DMD 0.4232 0.6700 0.8026 0.9932 0.7419 4.6958 3.8958 4.7125

- GR-DMD (τ = 0.2) 0.4265 0.6732 0.8395 0.9975 0.7480 4.7000 3.9042 4.7333

- GR-DMD (τ = 0.3) 0.4111 0.6786 0.5106 0.9933 0.7465 4.7583 3.9375 4.6958
- GR-DMD (τ = 0.4) 0.4047 0.6696 0.7869 0.9872 0.7424 4.7125 3.9022 4.7208
- GR-DMD (τ = 0.5) 0.4252 0.6774 0.7907 0.9953 0.7421 4.7083 3.8833 4.7058

#### 5.3 Ablation Studies

In this section, we conduct three groups of ablation studies: Teacher Model, Streaming Distillation, and KV Cache Rescheduling. Additional ablation results are provided in the Appendix.

Ablation with Teacher Model. To validate the effectiveness of In-Context Learning, we compare it with channel-wise concatenation. In Table 2, our designed in-context learning outperforms simple channel-wise concatenation across several metrics. Moreover, we compare different fine-tuning (FT) strategies, including Full FT, Attn FT, and LoRA [52] FT, with the results shown in Table 2. Full FT performs best overall, so we adopt this version of the teacher model for streaming distillation.

Ablation with Streaming Distillation. We first analyze the effectiveness of Gradient-Reweighted Distribution Matching Distillation (GR-DMD) in long-video (165 frames) extrapolation through qualitative and quantitative evaluations, as shown in Table 3 and Figure 7. Intuitively, naive DMD tends to produce distorted or duplicated human limbs during extrapolation. In contrast, our GradientReweighted DMD generates coherent and anatomically consistent human structures during extrapolation. Moreover, we further investigate the effect of the temperature coefficient τ on long-video extrapolation. In Table 3, the hyper-parameter τ = 0.2 yields the best overall performance.

Ablation with KV Cache Rescheduling. We now analyze the choice of reference KV and the effectiveness of disentanglement, as visualized in Figure 7. Clearly, randomly selecting reference KV leads to inconsistencies with previous frames. This phenomenon stems from the image-to-video

prior, where the generated initial frame aligns with the reference image; thus, mismatched reference KV breaks temporal coherence. Moreover, without disentangling the last historical KV, distribution mismatch arises: the reference frame is independently VAE-encoded during training, while the non-disentangled historical KV corresponds to multiple decoded frames (e.g., four).

### 6 Conclusion

In conclusion, we present FashionChameleon, a real-time and interactive framework for humangarment customization in autoregressive video generation, where users can interactively switch garment during generation. FashionChameleon consists of three key techniques: (i) We develop a Teacher Model with In-Context Learning to encourage the model to implicitly preserve coherence during single-garment switching. (ii) We introduce Streaming Distillation with In-Context Learning to enable efficient inference and consistent long-video extrapolation. (iii) We propose Training-Free KV Cache Rescheduling to support interactive multi-garment video customization while preserving coherent human motion. Extensive experiments show that our FashionChameleon demonstrates superiority over existing approaches while achieving real-time 720p video generation at 23.8 fps on a single GPU. Additional experiments on interactive customization and long-video extrapolation showcase its practical value in human-centric applications such as e-commerce and content creation.

### References

- [1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Conference on Neural Information Processing Systems, 2020.

- [2] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

- [3] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.

- [4] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang et al., “Hunyuanvideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.

- [5] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

- [6] T. Li, G. Zheng, R. Jiang, S. Zhan, T. Wu, Y. Lu, Y. Lin, C. Deng, Y. Xiong, M. Chen et al., “Realcam-i2v: Real-world image-to-video generation with interactive complex camera control,” in International Conference on Computer Vision, 2025.

- [7] Y. Fu, A. Jain, X. Chen, Z. Mo, and X. Di, “Drivegenvlm: Real-world video generation for vision language model based autonomous driving,” in IEEE International automated vehicle validation conference, 2024.

- [8] Q. Song, Y. Song, K. Peng, Y. Gao, and M. Z. Shou, “Worldwander: Bridging egocentric and exocentric worlds in video generation,” arXiv preprint arXiv:2511.22098, 2025.

- [9] Z. Wang, A. Li, L. Zhu, Y. Guo, Q. Dou, and Z. Li, “Customvideo: Customizing text-to-video generation with multiple subjects,” IEEE Transactions on Multimedia, 2026.

- [10] H. Chen, X. Wang, Y. Zhang, Y. Zhou, Z. Zhang, S. Tang, and W. Zhu, “Disenstudio: Customized multi-subject text-to-video generation with disentangled spatial control,” in ACM International Conference on Multimedia, 2024.

- [11] X. He, Q. Liu, S. Qian, X. Wang, T. Hu, K. Cao, K. Yan, and J. Zhang, “Id-animator: Zero-shot identity-preserving human video generation,” arXiv preprint arXiv:2404.15275, 2024.

- [12] S. Yuan, J. Huang, X. He, Y. Ge, Y. Shi, L. Chen, J. Luo, and L. Yuan, “Identity-preserving textto-video generation by frequency decomposition,” in IEEE Conference on Computer Vision and Pattern Recognition, 2025.

- [13] L. Liu, T. Ma, B. Li, Z. Chen, J. Liu, G. Li, S. Zhou, Q. He, and X. Wu, “Phantom: Subjectconsistent video generation via cross-modal alignment,” arXiv preprint arXiv:2502.11079, 2025.

- [14] Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu, “Vace: All-in-one video creation and editing,” in International Conference on Computer Vision, 2025.

- [15] B. Xue, Z.-P. Duan, Q. Yan, W. Wang, H. Liu, C.-L. Guo, C. Li, C. Li, and J. Lyu, “Stand-in: A lightweight and plug-and-play identity control for video generation,” in IEEE Conference on Computer Vision and Pattern Recognition, 2026.

- [16] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in International Conference on Computer Vision, 2023.

- [17] Z. Li, D. Qian, K. Su, Q. Diao, X. Xia, C. Liu, W. Yang, T. Zhang, and Z. Yuan, “Bindweave: Subject-consistent video generation via cross-modal integration,” arXiv preprint arXiv:2510.00438, 2025.

- [18] Y. Deng, Y. Yin, X. Guo, Y. Wang, J. Z. Fang, S. Yuan, Y. Yang, A. Wang, B. Liu, H. Huang et al., “Magref: Masked guidance for any-reference video generation with subject disentanglement,” arXiv preprint arXiv:2505.23742, 2025.

- [19] Z. Fei, D. Li, D. Qiu, J. Wang, Y. Dou, R. Wang, J. Xu, M. Fan, G. Chen, Y. Li et al., “Skyreelsa2: Compose anything in video diffusion transformers,” arXiv preprint arXiv:2504.02436, 2025.

- [20] Z. Zhang, J. Teng, Z. Yang, T. Cao, C. Wang, X. Gu, J. Tang, D. Guo, and M. Wang, “Kaleido: Open-sourced multi-subject reference video generation model,” arXiv preprint arXiv:2510.18573, 2025.

- [21] Z. Wang, Z. Yuan, X. Wang, Y. Li, T. Chen, M. Xia, P. Luo, and Y. Shan, “Motionctrl: A unified and flexible motion controller for video generation,” in ACM SIGGRAPH Conference on Computer Graphics and Interactive Techniques, 2024.

- [22] Q. Song, Z. Lin, Z. Zeng, Z. Zhang, L. Cao, and R. Ji, “Lightmotion: A light and tuning-free method for simulating camera motion in video generation,” arXiv preprint arXiv:2503.06508, 2025.

- [23] H. A. Lari, K. Vaishnava, and K. Manu, “Artifical intelligence in e-commerce: Applications, implications and challenges,” Asian Journal of Management, 2022.

- [24] Z. Li, M. Cao, X. Wang, Z. Qi, M.-M. Cheng, and Y. Shan, “Photomaker: Customizing realistic human photos via stacked id embedding,” in IEEE Conference on Computer Vision and Pattern Recognition, 2024.

- [25] Q. Song, M. Lin, W. Zhan, S. Yan, L. Cao, and R. Ji, “Univst: A unified framework for training-free localized video style transfer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

- [26] Z. Zhang, M. Lin, Q. Song, Y. Zhang, and R. Ji, “Objectadd: adding objects into image via a training-free diffusion modification fashion,” Pattern Recognition, 2025.

- [27] T. Yin, Q. Zhang, R. Zhang, W. T. Freeman, F. Durand, E. Shechtman, and X. Huang, “From slow bidirectional to fast autoregressive video diffusion models,” in IEEE Conference on Computer Vision and Pattern Recognition, 2025.

- [28] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman, “Self forcing: Bridging the train-test gap in autoregressive video diffusion,” arXiv preprint arXiv:2506.08009, 2025.

- [29] H. Zhu, M. Zhao, G. He, H. Su, C. Li, and J. Zhu, “Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation,” arXiv preprint arXiv:2602.02214, 2026.

- [30] J. Zhuang, S. Guo, X. Cai, X. Li, Y. Liu, C. Yuan, and T. Xue, “Flashvsr: Towards real-time diffusion-based streaming video super-resolution,” arXiv preprint arXiv:2510.12747, 2025.

- [31] Y. Huang, H. Guo, F. Wu, S. Zhang, S. Huang, Q. Gan, L. Liu, S. Zhao, E. Chen, J. Liu et al., “Live avatar: Streaming real-time audio-driven avatar generation with infinite length,” arXiv preprint arXiv:2512.04677, 2025.

- [32] J. Shin, Z. Li, R. Zhang, J.-Y. Zhu, J. Park, E. Shechtman, and X. Huang, “Motionstream: Real-time video generation with interactive motion controls,” arXiv preprint arXiv:2511.01266, 2025.

- [33] F. Bao, S. Nie, K. Xue, Y. Cao, C. Li, H. Su, and J. Zhu, “All are worth words: A vit backbone for diffusion models,” in IEEE Conference on Computer Vision and Pattern Recognition, 2023.

- [34] B. Chen, D. Martí Monsó, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann, “Diffusion forcing: Next-token prediction meets full-sequence diffusion,” in Conference on Neural Information Processing Systems, 2024.

- [35] D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, G. Schindler, R. Hornung, V. Birodkar, J. Yan, M.-C. Chiu et al., “Videopoet: A large language model for zero-shot video generation,” arXiv preprint arXiv:2312.14125, 2023.

- [36] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan, “Autoregressive model beats diffusion: Llama for scalable image generation,” arXiv preprint arXiv:2406.06525, 2024.

- [37] T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and W. T. Freeman, “Improved distribution matching distillation for fast image synthesis,” in Conference on Neural Information Processing Systems, 2024.

- [38] K. Liu, W. Hu, J. Xu, Y. Shan, and S. Lu, “Rolling forcing: Autoregressive long video diffusion in real time,” arXiv preprint arXiv:2509.25161, 2025.

- [39] Y. Lu, Y. Zeng, H. Li, H. Ouyang, Q. Wang, K. L. Cheng, J. Zhu, H. Cao, Z. Zhang, X. Zhu et al., “Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation,” arXiv preprint arXiv:2512.04678, 2025.

- [40] H. Yesiltepe, T. H. S. Meral, A. K. Akan, K. Oktay, and P. Yanardag, “Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout,” arXiv preprint arXiv:2511.20649, 2025.

- [41] S. Yang, W. Huang, R. Chu, Y. Xiao, Y. Zhao, X. Wang, M. Li, E. Xie, Y. Chen, Y. Lu et al., “Longlive: Real-time interactive long video generation,” arXiv preprint arXiv:2509.22622, 2025.

- [42] S. Huang, J. Wu, Q. Zhou, S. Miao, and M. Long, “Vid2world: Crafting video diffusion models to interactive world models,” arXiv preprint arXiv:2505.14357, 2025.

- [43] X. Mao, Z. Li, C. Li, X. Xu, K. Ying, T. He, J. Pang, Y. Qiao, and K. Zhang, “Yume-1.5: A text-controlled interactive world generation model,” arXiv preprint arXiv:2512.22096, 2025.

- [44] W. Sun, H. Zhang, H. Wang, J. Wu, Z. Wang, Z. Wang, Y. Wang, J. Zhang, T. Wang, and C. Guo, “Worldplay: Towards long-term geometric consistency for real-time interactive world modeling,” arXiv preprint arXiv:2512.14614, 2025.

- [45] Y. Zhang, C. Peng, B. Wang, P. Wang, Q. Zhu, F. Kang, B. Jiang, Z. Gao, E. Li, Y. Liu et al., “Matrix-game: Interactive world foundation model,” arXiv preprint arXiv:2506.18701, 2025.

- [46] K. Gao, J. Shi, H. Zhang, C. Wang, J. Xiao, and L. Chen, “Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing,” arXiv preprint

- arXiv:2411.16375, 2024.

[47] J. Hu, S. Hu, Y. Song, Y. Huang, M. Wang, H. Zhou, Z. Liu, W.-Y. Ma, and M. Sun, “Acdit: Interpolating autoregressive conditional modeling and diffusion transformer,” arXiv preprint

- arXiv:2412.07720, 2024.

- [48] T. Zhang, S. Bi, Y. Hong, K. Zhang, F. Luan, S. Yang, K. Sunkavalli, W. T. Freeman, and H. Tan, “Test-time training done right,” arXiv preprint arXiv:2505.23884, 2025.

- [49] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen et al., “Qwen-image technical report,” arXiv preprint arXiv:2508.02324, 2025.

- [50] S. Yuan, X. He, Y. Deng, Y. Ye, J. Huang, B. Lin, J. Luo, and L. Yuan, “Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation,” arXiv preprint arXiv:2505.20292, 2025.

- [51] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in IEEE Conference on Computer Vision and Pattern Recognition, 2024.

- [52] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” in International Conference on Learning Representations, 2022.

- [53] R. Li, M. Li, W. Liu, Y. Zhou, X. Zhou, Y. Yao, Q. Zhang, and H. Chen, “Unimatch: Universal matching from atom to task for few-shot drug discovery,” arXiv preprint arXiv:2502.12453, 2025.

- [54] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun et al., “Q-align: Teaching lmms for visual scoring via discrete text-defined levels,” arXiv preprint arXiv:2312.17090, 2023.

- [55] H. Wu, C. Chen, J. Hou, L. Liao, A. Wang, W. Sun, Q. Yan, and W. Lin, “Fast-vqa: Efficient end-to-end video quality assessment with fragment sampling,” in European Conference on Computer Vision, 2022.

- [56] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in IEEE Conference on Computer Vision and Pattern Recognition, 2019.

- [57] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

- [58] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “Musiq: Multi-scale image quality transformer,” in International Conference on Computer Vision, 2021.

❌ Low Aesthetics ❌ Low Quality

❌ No Human ❌ Multiple Human

[Figure 187]

❌ Transition Abruptly ❌ Discontinuous Frames

[Figure 188]

[Figure 189]

❌ Low Motion Strength

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

✅ Scene Layout ✅ Human State

✅ Environment Atmosphere ✅ Garment Details

[Figure 195]

[Figure 196]

Static caption:

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

VLM ✅ Human Evolution

Human Detection

Shot Segmentation

Overall Assessment

Raw Video Database

[Figure 207]

Optical-Flow Estimate

✅ Human Actions ✅ Scene Transitions

Filtered Video Database

[Figure 208]

Dynamic caption:

✅ Camera Motion

[Figure 209]

[Figure 210]

Extract

Stage1: General Coarse-to-Fine Video Filtering Stage2: Static-Dynamic Video Captioning

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Garment Image Initial Frame Reference Images

[Figure 217]

Try-Off Model

Try-On Model

❓Upper

[Figure 218]

Re-Generation Invalid

[Figure 219]

❓Lower

[Figure 220]

❓Semantic Consistency ❓Textural Consistency ❓Non-Garment Context

❓Non-Edited Preservation

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

VLM

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

❓Full

[Figure 235]

[Figure 236]

[Figure 237]

Invalid

[Figure 238]

Accepted

Retrieval

[Figure 239]

[Figure 240]

Valid

[Figure 241]

[Figure 242]

Re-

[Figure 243]

[Figure 244]

[Figure 245]

Valid

Check Types

Retrieval

[Figure 246]

Accepted

VLM

VLM

Garment Images Database

Stage3: Fine-Grained Garment Images Extraction Stage4: Adaptive Reference Images Construction

- Figure 8: The high-quality data curation pipeline of FashionChameleon. It consists of four stages:

(1) General Coarse-to-Fine Video Filtering, (2) Static-Dynamic Video Captioning, (3) Fine-Grained Garment Image Extraction, and (4) Adaptive Reference Image Construction.

### A Data Curation Pipeline Details

Recall that we briefly introduce our high-quality data curation pipeline in the main paper, which comprises four stages: 1. General Coarse-to-Fine Video Filtering, 2. Static-Dynamic Video Captioning,

- 3. Fine-Grained Garment Image Extraction, and 4. Adaptive Reference Image Construction. The overall curation pipeline is illustrated in Figure 8, and we detail each stage as follows:

- 1. General Coarse-to-Fine Video Filtering. We collected a large set of raw videos from the Internet and filtered them in a coarse-to-fine manner using Shot Segmentation, Human Detection, Optical-Flow Estimation, and Overall Assessment to retain only qualified videos:

- • Shot Segmentation. The raw videos are first processed with PySceneDetect to identify scene transitions and split into separate scene clips. These clips are then further divided into 3-5 second subclips, while discontinuous or overly short subclips are removed.
- • Human Detection. We apply YOLOv8-Seg to each subclip to detect human presence and retain only single-person clips. Clips without humans or with multiple prominent humans are removed. Note that a clip is still considered single-person if one person occupies most of the frame and any other visible people appear only as small, blurred background figures.
- • Optical-Flow Estimation For each subclip containing one human, we estimate optical flow using UniMatch [53] to measure motion magnitude. We then retain clips with moderate to large motion and discard clips with little or slow motion based on a predefined threshold.
- • Overall Assessment. Finally, we evaluate each subclip using Q-Align [54] for aesthetics and FAST-VQA-M [55] for overall visual quality. We retain clips with high aesthetic and quality scores according to predefined thresholds, and remove those with low scores.

- 2. Static-Dynamic Video Captioning. For the filtered videos, we use the vision-language model (VLM) Gemini-3.1 to generate captions. Specifically, we adopt a static-dynamic decoupling strategy:

- • Static Caption. We prompt the VLM to focus on the static content in each video, including the scene layout, environmental atmosphere, human attributes (e.g., appearance), and garment details. These elements are intrinsic to the video and remain unchanged over time.

- • Dynamic Caption. We then prompt the VLM to capture the dynamic content of each video, including human evolution (e.g. facial expressions), human action, camera motion, and scene transitions. These elements are inherently temporal and typically change over time.

The system prompt for Gemini-3.1 is presented in Sec.N.

- 3. Fine-Grained Garment Images Extraction. For each filtered video, we extract the initial frame and apply the image try-off model Qwen-Image-Edit [49] to extract corresponding garment images. Since try-off is not always reliable in practice, we further introduce a VLM to verify the extracted garments. In detail, for each extracted garment, the VLM performs a three-stage validity check:

- • Semantic Consistency. The VLM will check whether the extracted garment matches the clothing in the initial frame at a high level, such as garment category and color.
- • Textural Consistency. The VLM will check whether the extracted garment matches the clothing in the initial frame at a low level, such as texture and logos.
- • Non-Garment Context. The VLM will check whether the extracted garment contains information beyond the garment itself, such as irrelevant scene content or other artifacts.

We reapply the image try-off model until the extracted result passes all VLM-based validity checks. If extraction fails repeatedly, we discard the corresponding sample.

- 4. Adaptive Reference Images Construction. In the final stage, we construct the reference image. To improve training robustness, the garment worn by the person in the reference image should differ from the extracted garment. We note that the garment information extracted in the previous stage may be incomplete, for example, including only the upper-body or lower-body clothing. To fully utilize the available garment information, we employ the VLM Gemini-3.1 to guide the accurate construction of the reference image. In detail, the overall process is formulated as follows:

- • Garment Type Classification. For the garment extracted from each video, the VLM first determines whether it corresponds to upper-body, lower-body, or full-body clothing.
- • Garment Type Retrieval. Based on the predicted garment category, the VLM will retrieve a visually compatible garment of the same type from the garment database.
- • Accurate Image Try-On. Given the retrieved garment and the extracted first frame, we apply an image try-on model to construct the reference image. This enables fine-grained customization, where the specified garment is changed while other regions remain unchanged.
- • Validity Check. We use a VLM to verify each reference image by checking whether the non-edited regions remain unchanged. If not, we reconstruct the reference image using the image try-on mode. We discard the corresponding sample if reconstruction fails repeatedly.

In total, we curate about 82K triplets, each consisting of a reference image, a garment image, and the corresponding video. After manual verification, about 62K triplets are retained in the training dataset.

### B Training Details

Pre-training Configuration. During teacher model pre-training, we keep the VAE in float32 precision and fully fine-tune the transformer in bfloat16. To further improve GPU utilization, we adopt a Fully Sharded Data Parallel (FSDP) training strategy with a global batch size of 64. We optimize the model using AdamW with β1 = 0.9, β2 = 0.999, and a weight decay of 0.01. We further employ a learning rate schedule with a warm-up of 200 steps, followed by a two-stage decay: the learning rate is set to 1 × 10−5 until step 1100 and then decayed to 5 × 10−6 until step 2300.

Post-training Configuration. During streaming distillation post-training, we maintain both the VAE and transformer in bfloat16 and also adopt FSDP training strategy with a global batch size of 64. For teacher forcing, the generator is initialized from the pre-trained teacher model and then fully fine-tuned for 4000 steps using AdamW with a learning rate of 1 × 10−6, β1 = 0.0, β2 = 0.999, and a weight decay of 0.01. For gradient-reweighted distribution distillation matching, the generator is initialized from the model fine-tuned with teacher forcing, while both the real score and fake score networks are initialized from the pre-trained teacher model. The few-step generator uses a timestep schedule of [1000,750,500,250]. We fully fine-tune the generator and the fake score network with a 1:5 update ratio, while keeping the real score network frozen. We optimize both generator and fake score network with AdamW for 400 steps, using learning rates of 2 × 10−6 for the generator and

- 4 × 10−7 for the fake score network, with β1 = 0.0, β2 = 0.999, and a weight decay of 1 × 10−2.

[Figure 247]

(a) (b)

|[Figure 248]|
|---|

In a classroom, a girl with black hair bows sits at a desk with a microscope and a physics book. She wears a denim corset top with white lace trim, paired with denim wide-leg jeans featuring a lace belt and tall black boots. She slowly rises from her seat to stand upright, showcasing the garment silhouette as the camera steadily pulls back. She then tilts her head slightly to the right, shifting her gaze from the microscope to look directly into the lens, her long hair swaying naturally.

|[Figure 249]|
|---|

Reference

Garment

(c)

- Figure 9: Data analysis and representative samples of HGC-Bench. (a) A word cloud generated from the input prompts, illustrating the diversity of scenarios and semantic content. (b) The distribution of garment categories, showing the proportions of different garment types. (c) Representative samples from HGC-Bench, each comprising a reference image, a garment image, and an input prompt.

Dataset Configuration. For both pre-training and post-training, we use a carefully curated paired dataset of 62K samples, each consisting of a reference image, a garment image, and a video sequence. We sample sequences of 81 frames to align with existing customization methods. The video and reference image are resized to 1280 × 704 while preserving aspect ratio, whereas the garment image is center-padded to 1280 × 704 with aspect ratio preserved, following the standard resolution of WAN2.2-5B-TI2V [5]. During pre-training, since the reference image already contains rich static information, we use only the dynamic content with a probability of 70%, and use the full caption (static-dynamic contents), in the remaining 30% of cases. This encourages the model to infer static attributes directly from the reference image, reducing its reliance on textual descriptions. During post-training, we observe that using full captions, which include both static and dynamic content, leads to improved performance. We provide a more comprehensive analysis in Sec. D. During interactive inference, we intentionally avoid including garment-related descriptions in the input prompt, since the character’s outfit is determined by the input garment image and may vary over time, which could otherwise conflict with fixed textual descriptions.

### C HGC-Bench Details

We propose HGC-Bench, a dedicated benchmark for comprehensive evaluation. Specifically, we curate high aesthetic reference images from the Internet, anonymize identifiable facial information via face swapping, and pair them with corresponding garment images from our collected garment database. Given the reference image and the garment image, we employ Gemini-3.0 to generate the corresponding prompt, which consists of a concise static description (e.g., human accessories and scene information), and a detailed dynamic description (e.g., human motions, camera movements). In total, we curate 240 samples, where each sample consists of a reference image, a garment image, and the corresponding prompt. Figure 9 presents the data analysis and representative samples of HGC-Bench. The system prompt for Gemini-3.0 is presented in Sec.N:

### D Additional Ablation Studies on Distillation Prompts

Recall that we adopted the hybrid caption strategy (70% dynamic content and 30% static-dynamic contents) during the teacher model training, to facilitate the extraction of static information from reference images. During the streaming distillation (teacher forcing and gradient reweighted DMD)

Table 4: Additional quantitative ablation on different distillation captions with τ = 0.2. Variants Cur. ↑ GME ↑ Amp. ↑ Smoo. ↑ VQ ↑ HGC ↑ LGC ↑ NTP ↑ Naive DMD (Mixed Caption) 0.4237 0.6564 0.7164 0.9797 0.7349 4.6234 3.8703 4.6444 Naive DMD (Long Caption) 0.4232 0.6700 0.8026 0.9932 0.7419 4.6958 3.8958 4.7125 GR-DMD (Mixed Caption) 0.4102 0.6692 1.1699 0.9955 0.7473 4.6583 3.9000 4.7369 GR-DMD (Long Caption) 0.4265 0.6732 0.8395 0.9975 0.7480 4.7000 3.9042 4.7333

Edit(20B)+I2V(5B) VACE(14B)

| |
|---|

| |
|---|

Kaleido(14B) MAGREF(14B)

| |
|---|

| |
|---|

SkyReels-A2(14B) Phantom-1.3b

| |
|---|

| |
|---|

Phantom-14b Ours

| |
|---|

| |
|---|

| | | | | | |32%|
|---|---|---|---|---|---|---|

ID Consistency

| | | | | | | |43%|
|---|---|---|---|---|---|---|---|

Garment Consistency

| | | | | | | |44%|
|---|---|---|---|---|---|---|---|

Temporal Coherence

| | | | | | |35%|
|---|---|---|---|---|---|---|

Visual Quality

- Figure 10: Quantitative results of the human evaluation. We compare our FashionChameleon with other baselines across four key dimensions: ID Consistency, Garment Consistency, Temporal Coherence, and Visual Quality. Our FashionChameleon achieves superior human preference rates.

process, we find that using different types of captions can lead to different distilled results. We quantify this effect, and the comparison results are reported in Table 4. Experimental results demonstrate that employing long caption (static-dynamic contents) yields superior performance.

### E Additional User Study

To evaluate user preference over videos generated by our method FashionChameleon and other baselines, we conduct a user study. In detail, for each comparison group, participants are shown videos generated by different methods and are asked to select the one with the best ID Consistency, the best Garment Consistency, the best Temporal Coherence, and the best Visual Quality. In total, we collected 672 valid responses, and the results are shown in Figure 10. Our method achieves superior performance in id consistency, garment consistency, temporal coherence, and visual quality.

### F Evaluation Details

In this section, we provide a detailed clarification of the quantitative metrics used in the main paper. ID Consistency (Cur Score) The Cur Score measures the consistency between the reference image and generated video. Specifically, we extract facial embeddings from the reference image and each video frame using ArcFace [56] and compute the cosine similarity between the resulting embeddings. Text Alignment (GME Score) The Gme Score is used to assess the semantic alignment between the generated video and the input prompt. In detail, we utilize a vision-language model fine-tuned from Qwen2-VL [57] to provide stronger capability in handling long and complex text descriptions.

Motion Magnitude (Amplitude) The Amplitude score measures motion amplitude in the generated video. Specifically, we compute forward and backward optical flow between adjacent frames, calculate the flow magnitude, and average it over all pixels and frames to obtain the final score.

Temporal Smoothness (Smoothness) The Smoothness score evaluates the overall fluidity of motion in the generated video. In particular, we utilize Q-Align [54] to measure the temporal coherence and the smoothness of motion transitions between consecutive frames.

Visual Quality (VQ Score) The VQ Score evaluates the overall visual quality of a video. Specifically, we apply the no-reference image quality assessment model MUSIQ [58] to predict a quality score for each frame, and then average the frame-level scores to obtain the final video-level score.

Inference Efficiency (FPS) The frames per second (FPS) measures the inference efficiency of a model. Specifically, we compute it as the total number of frames generated by the backbone network divided by the corresponding inference time.

Garment Consistency Besides the metrics above, we further evaluate the consistency between the garment worn by the character in the generated video and the given garment image. As no established metric is available for this purpose, we employ the vision-language model Gemini-3.0 to assess this consistency from three dimensions: high-level garment consistency, low-level garment consistency, and non-target garment preservation. System prompt for Gemini-3.0 is provided in Sec.N.

### G Limitations and Future Work

While FashionChameleon shows strong efficiency and interactivity in human-centric applications, several limitations remain: (i) Despite the curated data pipeline, the current training data still has limited garment categories and variations, which may restrict its generalization to complex scenarios. (ii) The model remains challenged by complex human motions and camera movements, largely due to the imperfect performance of current open-source video generation backbones like Wan [5].

Therefore, future work could focus on developing a more efficient data curation pipeline, scaling up training datasets, and exploring stronger video generation backbones to address these limitations.

### H Potential Negative Societal Impact

Our FashionChameleon is intended for human-garment customized video generation in human-centric content creation scenarios. Nevertheless, we acknowledge that current models for human-garment video customization can introduce nontrivial societal risks when deployed irresponsibly or used with malicious intent. We summarize our discussion in the following three points:

- • Sexually Explicit or Violent Content. Without proper safeguards, generated content may include sexually explicit, violent, or otherwise inappropriate material, potentially causing psychological or emotional harm to diverse audiences.
- • Stereotypes and Bias. Unintended biases in character and garment information in the training data may be reflected or amplified in generated content, potentially reinforcing harmful cultural stereotypes or discriminatory visual representations.
- • Misleading Content. Human-garment video customization models may be misused to create realistic yet false video advertisements, increasing the risk that misleading information spreads quickly and widely at scale.

We include these considerations to make clear that the method should be deployed responsibly and always accompanied by appropriate protections against misuse.

### I Additional Qualitative Comparison

To further validate the effectiveness of our FashionChameleon and its advantages over competing baselines, we provide additional qualitative comparisons in Figure 11 and Figure 12. Visually, FashionChameleon demonstrates better character consistency and garment consistency, while producing more coherent and higher-quality results.

Ours(5B) VACE(14B) Kaleido(14B) MAGREF(14B) SkyReels-A2(14B) Phantom(1.3B) Phantom(14B)

Edit(20B)+I2V(5B)

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Reference

|[Figure 258]|
|---|

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Garment

|[Figure 267]|
|---|

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Reference

|[Figure 276]|
|---|

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Garment

|[Figure 285]|
|---|

- Figure 11: Additional qualitative comparison between our FashionChameleon and other baselines.

Edit(20B)+I2V(5B)

Reference

Garment

Reference

Garment

Ours(5B) VACE(14B) Kaleido(14B) MAGREF(14B) SkyReels-A2(14B) Phantom(1.3B) Phantom(14B)

|[Figure 286]|
|---|

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

|[Figure 303]|
|---|

[Figure 304]

|[Figure 305]|
|---|

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

- Figure 12: Additional qualitative comparison between our FashionChameleon and other baselines.

### J Additional Examples of Short Video Customization

Our FashionChameleon is trained on 81-frame video clips and therefore supports customized generation of short videos of the same length. We provide additional examples, as shown in Figure 13 and Figure 14. Notably, FashionChameleon can produce coherent and high-fidelity human-garment customized videos, further highlighting its superiority.

Reference 1-st Frame 9-th Frame 18-th Frame 27-th Frame 36-th Frame 45-th Frame 54-th Frame 63-th Frame 72-th Frame 81-th Frame

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

|[Figure 332]|
|---|

Garment

|[Figure 333]|
|---|

Reference

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

|[Figure 343]|
|---|

[Figure 344]

Garment

|[Figure 345]|
|---|

Reference

|[Figure 346]|
|---|

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

Garment

|[Figure 357]|
|---|

###### Figure 13: Additional results for short video customization using our FashionChameleon.

Reference 1-st Frame 9-th Frame 18-th Frame 27-th Frame 36-th Frame 45-th Frame 54-th Frame 63-th Frame 72-th Frame 81-th Frame

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

|[Figure 368]|
|---|

Garment

|[Figure 369]|
|---|

Reference

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Garment

|[Figure 381]|
|---|

Reference

|[Figure 382]|
|---|

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

Garment

|[Figure 393]|
|---|

Figure 14: Additional results for short video customization using our FashionChameleon.

### K Additional Examples of Interactive Customization

Thanks to the proposed KV cache rescheduling strategy, our FashionChameleon supports interactive multi-garment customized generation, with the additional examples shown in Figure 15 and Figure 16. Unlike conventional methods that require a reference image to be specified in advance, FashionChameleon allows users to freely switch reference images at different stages of generation while preserving motion continuity, enabling interactive customization. This further demonstrates the superiority of FashionChameleon in the interactive generation domain.

Reference

|[Figure 394]|
|---|

|[Figure 395]|
|---|

|[Figure 396]|
|---|

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Garment Switch Switch

|[Figure 406]|
|---|

Reference

|[Figure 407]|
|---|

|[Figure 408]|
|---|

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

Switch Switch

Garment

|[Figure 419]|
|---|

Reference

|[Figure 420]|
|---|

|[Figure 421]|
|---|

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

|[Figure 431]|
|---|

Switch Switch

Garment

|[Figure 432]|
|---|

- Figure 15: Additional visualizations for interactive multi-garment video customization using our FashionChameleon.

Garment

Reference

Switch Switch

Reference

Garment Switch Switch

[Figure 433]

|[Figure 434]|
|---|

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

|[Figure 439]|
|---|

[Figure 440]

[Figure 441]

[Figure 442]

|[Figure 443]|
|---|

[Figure 444]

[Figure 445]

|[Figure 446]|
|---|

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

|[Figure 456]|
|---|

|[Figure 457]|
|---|

|[Figure 458]|
|---|

Garment

Reference

|[Figure 459]|
|---|

[Figure 460]

|[Figure 461]|
|---|

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

|[Figure 470]|
|---|

Switch

|[Figure 471]|
|---|

Switch

- Figure 16: Additional visualizations for interactive multi-garment video customization using our FashionChameleon.

### L Additional Examples of Long Video Customization.

Benefiting from our dedicated autoregressive design, FashionChameleon can generalize beyond the training sequence length, thereby enabling customized generation of longer videos. Additional qualitative results are provided in Figure 17 and Figure 18. The qualitative results show that FashionChameleon maintains long-range character consistency and garment consistency.

### M Prompt List of Figures

For reproducibility, we list the prompts used to generate Figure 1 in the main paper:

Reference 1-st Frame 18-th Frame 35-th Frame 52-th Frame 69-th Frame 86-th Frame 103-th Frame 120-th Frame 137-th Frame 154-th Frame

|[Figure 472]|
|---|

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

Garment

|[Figure 483]|
|---|

Reference

|[Figure 484]|
|---|

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

Garment

|[Figure 495]|
|---|

Reference

[Figure 496]

[Figure 497]

|[Figure 498]|
|---|

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

Garment

|[Figure 507]|
|---|

###### Figure 17: Additional long video extrapolation visualizations of our FashionChameleon.

Reference 1-st Frame 18-th Frame 35-th Frame 52-th Frame 69-th Frame 86-th Frame 103-th Frame 120-th Frame 137-th Frame 154-th Frame

|[Figure 508]|
|---|

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

Garment

|[Figure 519]|
|---|

Reference

|[Figure 520]|
|---|

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

Garment

|[Figure 531]|
|---|

Reference

|[Figure 532]|
|---|

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

Garment

|[Figure 543]|
|---|

Figure 18: Additional long video extrapolation visualizations of our FashionChameleon.

1. “A woman wearing a blue beret, earrings, and a watch stands on a floral garden path. She takes light steps forward, with her arms swinging naturally. Her gaze shifts from downward to focusing on the lens with a gentle smile, then smoothly transitions into a still pose, ensuring the movement is continuous and physically realistic.”

- For reproducibility, we list the prompts used to generate Figure 4 in the main paper:

1. “A woman performs a series of poses in an indoor setting while holding a white handbag in her right hand. Initially facing the camera, she subtly shifts her body to the left and places her left hand into her pocket. She then moves her left hand to rest lightly on a black shelving unit behind her. Throughout the video, she maintains a friendly smile and steady eye contact with the camera, with subtle changes in her stance and orientation. The video is

filmed in a minimalist indoor studio featuring plain white walls and a light grey carpeted floor. To the right, a sleek black shelf displays decorative items such as vinyl records and magazines, while the corner of a white sofa is partially visible on the left. The lighting is bright and diffused, creating a clean and modern aesthetic. The camera remains stationary in a full-body composition, ensuring a consistent visual style.”

- For reproducibility, we list the prompts used to generate Figure 5 in the main paper:

- 1. “A man strolls along an outdoor brick path, wearing a brown turtleneck long-sleeved knit sweater paired with white shorts and beige sandals. He maintains a steady forward gait, his arms swinging naturally to showcase the drape of the new garment. The camera performs a smooth tracking shot, moving backward to keep him centered in the frame. Initially looking to the side, he slowly turns his head forward, shifting his gaze naturally and smoothly to look directly into the lens.”
- 2. “A young woman stands in a room, wearing a red short-sleeved t-shirt paired with a long floral skirt, with a red string bracelet on her left wrist. She initially tilts her head slightly to the side, then naturally shifts her gaze back to the lens with a soft smile. She performs a subtle turn to the left, causing the hem of the long skirt to sway with natural physics. The camera pans slowly to the right to keep her centered as she turns back to face forward, showcasing the elegant silhouette of the outfit.”

- For reproducibility, we list the prompts used to generate Figure 6 in the main paper:

- 1. “A young woman walks near park flowers, wearing a blue zippered crop top and lace-up distressed denim shorts, accented with a white cap, necklace, and a bag featuring a teddy bear charm. She walks forward with an elegant catwalk stride, her arms swinging naturally while her platform sneakers land steadily. The camera performs a steady tracking shot, keeping her centered. She shifts her gaze from forward to the lens, blinking with a smile and tilting her head slightly.”
- 2. “A young woman stands against a pink and blue background. She wears purple flower earrings and carries a pink woven bag on her shoulder. She walks forward with light steps, her arms swinging naturally, while the bag strap bounces slightly. She then tilts her head toward the camera with a bright smile and a natural blink. The movement is smooth and consistent, ending in a frozen mid-stride pose.”
- 3. “A young woman stands in a room filled with books and vintage items. She wears a baseball cap with text and has one hand in her pocket. She slowly lowers her hand from the cap, shifts her weight, and turns slightly to the right. Her gaze shifts from the lens toward the stack of books before turning back to blink and smile naturally. The movement is smooth and consistent, ending with her holding a slightly turned pose.”

- For reproducibility, we list the prompts used to generate Figure 7 in the main paper:

- 1. “In the video, a young woman slowly enters from the right side of the frame and stops near a table. She initially looks down in reflection, then gracefully turns her head to the left, gazing into the distance. The camera remains in a fixed position, capturing the scene through a transparent glass door, with subtle reflections on the glass shifting as she moves. The setting is an interior space with soft lighting, likely a cafe or restaurant, featuring wooden tables and chairs with a warm texture. The overall visual style is realistic and cinematic, using the glass door in the foreground to create an observational perspective within a warm and tranquil atmosphere.”
- 2. “Captured from a static camera angle, a young woman with long, flowing black hair sways her body gracefully to a rhythmic beat. She raises her left hand to touch and adjust her hair, tossing it over her shoulder while her arms move naturally in sync with her shifting posture. Throughout the sequence, she maintains direct eye contact with the camera, exhibiting a series of fluid and confident movements. The setting is a minimalist and elegant indoor environment featuring large beige pleated curtains in the background and a brown striped carpet on the floor. To the right stands a contemporary white floor lamp with a decorative stem made of transparent spherical crystals. The lighting is soft and diffused, dominated by a warm color palette of beige and tan, creating a cozy, high-quality lifestyle aesthetic.”

For reproducibility, we list the prompts used to generate Figure 11 and Figure 12 in the Appendix:

- 1. “On a lush tree-lined path, a woman wears a black and white checkered vest paired with a blue mini skirt featuring a cherry graphic, accented by a pearl necklace and white boots. She slowly lowers her raised right arm and turns her body slightly to the left to showcase the skirt’s silhouette. The camera orbits steadily around her in an arc. She shifts her gaze from the side back to the lens, her long hair swaying naturally over her shoulders as she moves.”
- 2. “A woman in a blue cap and sunglasses stands by a white tiled wall, wearing a light grey multi-pocket hooded jacket, white wide-leg pants, and beige shoes, holding a brown bag with a bear charm. She slowly lowers her raised right arm and takes a natural step forward to showcase the outfit. The camera pans horizontally to the right; she turns her head from the side to face forward, gazing into the lens through her sunglasses with a relaxed posture.”
- 3. “A young woman stands outdoors wearing white headphones and sunglasses, dressed in a black short-sleeved T-shirt and a dark green button-front maxi skirt, carrying a black backpack with white socks and sneakers. She walks steadily toward the camera, the long skirt’s hem swaying naturally and gracefully with her steps. The camera pulls back smoothly to reveal the full silhouette of the outfit; she shifts her gaze from the side to the lens, smiling faintly and blinking.”
- 4. “On a city street, a black-haired man wearing sunglasses is dressed in a black U-neck tank top paired with ripped blue jeans and a black belt, holding a brown leather bag in his right hand with a watch and bracelet on his wrists. He walks forward with steady steps, his body swaying naturally to showcase the fit of the tank top. The camera slowly zooms out from a close-up to a full-body view. He shifts his gaze from downward to looking straight ahead with a calm expression.”

For reproducibility, we list the prompts used to generate Figure 13 and Figure 14 in the Appendix:

- 1. “A young woman stands by the poolside with city buildings in the background, wearing a turquoise long-sleeved shirt and a white tiered ruffled long skirt, holding a small creamcolored handbag. She walks toward the camera with light catwalk steps, the layered hem swaying naturally. The camera slowly zooms out to reveal the full silhouette; her gaze shifts from the side back to the lens as she gives a slight, steady nod.”
- 2. “A young woman stands in a minimalist gray indoor setting, wearing a white puff-sleeved blouse and a red phoenix-embroidered vest paired with a red patterned pleated skirt, with a thin bracelet on her left wrist. She looks down initially, then raises her gaze to the camera while turning slightly to the left, allowing the skirt to drape naturally. The camera smoothly pulls back from a close-up to reveal the full-length silhouette of the traditional outfit.”
- 3. “In front of a white wall, a man wearing black-rimmed glasses holds a coffee cup, dressed in a tan sports bra and dark brown leggings with white sneakers. He slowly transitions from a leaning pose to a steady upright stance, balancing his weight on both feet to showcase the silhouette. The camera zooms out smoothly to capture the full outfit; he tilts his head slightly, shifting his gaze from the side back to the lens with a calm expression.”
- 4. “A young man wearing a baseball cap and black glasses stands before a dark rolling shutter, dressed in a white long-sleeved top and dark blue wide-leg trousers. He moves his hands out of his pockets to his sides and walks forward toward the camera, the loose pant legs creating natural folds and swaying with each step. The camera tracks him steadily; he tilts his head slightly upward, shifting his gaze from the side to the lens with a composed expression.”
- 5. “A young woman stands by an outdoor road, wearing a red and blue striped tie-front top with light-blue denim shorts and carrying a large pink canvas bag. Transitioning from an open-arm pose, she naturally lowers her hands and walks forward toward the camera with a brisk, steady gait. The camera tracks backward smoothly, keeping her centered in the frame. She briefly looks down before raising her head, shifting her gaze from the side to the lens with a bright smile, her long hair swaying naturally as she moves.”
- 6. “A Black man sits on an outdoor hay bale, wearing a brown long-sleeved shirt with double chest pockets, paired with wide-leg white trousers, brown boots, and olive socks. Resting his hands on his knees, he slowly stands up from the bale, smoothing the shirt front to showcase the drape. The camera pulls back slowly to reveal the full outfit. He tilts his head slightly, shifting his gaze from the side back to the lens with a calm expression.”

For reproducibility, we list the prompts used to generate Figure 15 and Figure 16 in the Appendix:

- 1. “The woman stands against a white backdrop holding an exquisite bouquet of lilies and greenery. Starting with a direct gaze, she blinks and transitions into a natural smile with gentle eyes. She then turns her body slowly to the right while holding the bouquet, showcasing her side profile with smooth movements. Her hair and the flower petals sway slightly following the physics of the motion. Finally, she holds a graceful side-facing posture with a relaxed expression.”
- 2. “A woman strolls through an urban street. She carries a brown leather tote bag on her right shoulder and holds an iced coffee in her left hand, with her gold necklace and hair clip glinting. She walks forward toward the camera with light steps, her arms swinging naturally and the bag swaying slightly with her rhythm. Initially laughing and looking aside, she then turns her gaze to the camera with bright eyes, eventually pausing while maintaining a natural walking posture.”
- 3. “A woman stands against a simple background, cradling a woven basket of white daisies in her right arm and wearing a watch on her left wrist. She initially looks down at the flowers, then slowly turns her body to the left with smooth movements, her arms swinging naturally. She then shifts her gaze to the camera with a gentle smile and a slight head tilt, ensuring a fluid transition before returning to a stable forward-facing pose.”
- 4. “A young man stands against a clean light blue background. He shifts his center of gravity and takes a natural small step forward, with his arms swinging slightly and naturally. He blinks and tilts his head down slightly before looking up to gaze at the lens with a confident and gentle smile. His head turns slightly in coordination with his body, and the entire movement is smooth, consistent, and physically natural.”
- 5. “A young man stands in a minimalist studio with a wooden cabinet nearby, holding a pair of headphones in his right hand. Wearing orange sunglasses and a silver chain, he begins by taking a steady step forward. As his weight shifts, he transitions from a slight head tilt to looking directly into the lens with a relaxed expression. The headphones sway gently with his movement, which is smooth and physically natural, ending in a stable standing pose.”
- 6. “The woman stands in the center of a leafy street, wearing hoop earrings. She tilts her head slightly to showcase her accessories, then begins walking slowly toward the camera with her arms swinging naturally and her weight shifting steadily. During the walk, she turns her gaze from the side back to the lens, blinking naturally with a confident smile, before coming to a smooth stop.”

For reproducibility, we list the prompts used to generate Figure 17 and Figure 18 in the Appendix:

- 1. “On a sunlit park path, a long-haired woman with a red flower hair accessory wears a black V-neck sweater paired with a long blue traditional skirt featuring gold patterns and a delicate necklace. She takes elegant catwalk steps toward the camera, the heavy blue hem swaying naturally with her stride. The camera moves backward smoothly to track her, maintaining a consistent frame. She tilts her head slightly, shifting her gaze upward from the ground to fixate on the lens with a gentle smile.”
- 2. “On an outdoor park path, a long-haired woman wearing sunglasses is dressed in a white short-sleeved T-shirt with a blue bow and a pink tie-dye denim mini skirt. She carries a white mini handbag in her left hand and holds a phone in her right. She walks toward the camera with a graceful catwalk gait, her movements fluid and natural. The camera performs a steady tracking shot as she tilts her head slightly to the right, shifting her gaze from the side back to the lens with a smile, her hair swaying gently with her steps.”
- 3. “A silver-haired elderly woman stands by a traditional wooden chair, wearing a beige standcollar jacket with plaid cuffs and a cinched hem, paired with red printed trousers and a pearl necklace. She lowers her raised right arm and gently turns to the left to display the jacket’s side profile. The camera pulls back steadily to capture the full ensemble; the woman turns her head to shift her gaze from the side back to the lens with a kind and composed expression.”
- 4. “A young woman stands against a light blue background, wearing a navy blue camisole paired with a long dark blue denim skirt featuring a brown belt, along with white socks and

- sneakers. She slowly turns her body to the left, showcasing the drape of the long skirt and the belt details with fluid movements. The camera performs a subtle orbital rotation around her; she tilts her chin slightly and shifts her gaze naturally from the side back to the lens with a gentle smile.”
- 5. “A young woman stands in a clothing store wearing a light purple ruffled short-sleeve shirt and cream-colored wide-leg pants, with a gold bracelet on her right wrist and white sneakers. She walks toward the camera with light steps, the wide pant legs swaying naturally with her movement. The camera tracks her steadily; she initially looks toward the side shelves before gently turning her head to shift her gaze back to the lens with a smile.”
- 6. “A long-haired woman wearing sunglasses, a colorful necklace, and an orange bracelet, dressed in a brown turtleneck sweater and black trousers with white sneakers, walks outdoors. She maintains a steady gait approaching the camera, her arms swinging naturally to showcase the drape of the sweater and trousers. The camera tracks her movement, keeping her centered in the frame. She tilts her head slightly to the left, shifting her gaze from the side back to the lens with a relaxed expression.”

### N System Prompts of VLM

We present the system prompt for Gemini-3.1 to generate prompts in training datasets below:

System Prompt: Please generate a structured multilingual description based on the input video content, strictly following the specifications below:

- 1. Dynamic element description: Focus on content that changes over time in the video, such as:

- (a) The subject (e.g., person, animal, object, etc.) and changes in its behavior, state, or position;
- (b) Specific actions (e.g., running, waving, opening a door, vehicles moving, etc.); (c) Scene transitions (e.g., switching from a street to an indoor setting, weather changing from sunny to rainy, etc.); (d) Camera movement (e.g., push-in, pull-out, pan, tracking shot, fixed shot, zoom, etc.).

- 2. Static element description: Accurately identify visual features that remain unchanged throughout the video, such as: (a) The inherent appearance of the subject or environment (e.g., clothing style, architectural style, object color and material, indoor layout, etc.); b The overall aesthetic style (e.g., realistic, animated, film-like, cyberpunk, minimalist, etc.); (c) Consistent visual style elements such as color tone, lighting atmosphere, and composition principles; d. The character’s clothing, styling, and accessories.
- 3. Additional notes: In the dynamic element description, completely ignore any description of the character’s clothing, styling, or accessories, but handheld items (if any) may be briefly described. Please output in standard JSON format, containing the following four fields. The value of each field must be an array of two strings: (a) The first string: description of dynamic elements; (b) The second string: description of static elements. The field definitions are as follows: “cn long”: two detailed Chinese descriptions (the first for dynamics, the second for statics); “cn short”: two concise Chinese descriptions (the first for dynamics, the second for statics); “en long”: two detailed English descriptions, semantically corresponding to cn long; “en short’: two concise English descriptions, semantically corresponding to cn short. Output requirements:

- 1. Use natural, objective, and accurate language, based only on visible video content, without adding speculation or fabricated details;
- 2. The Chinese and English descriptions should be semantically aligned, but do not need to be word-for-word translations;
- 3. Long descriptions should be comprehensive and detailed, while short descriptions should be concise and focused on the core information.

We present the system prompt for Gemini-3.0 to generate prompts in HGC-Bench below:

System Prompt: Role Definition: You are a senior prompt expert specializing in video generation (I2V). Your core task is to write coherent, dynamic, and physically plausible video-generation description scripts based on the characteristics of the first-frame image [Image1] and the target garment image [Image2]. The script mainly includes a static description of the first frame and a subsequent dynamic description. The static description should focus on the person wearing the new garment in the original scene. The dynamic description should be coherent and natural, avoiding motion collapse.

##### I. Static Description:

1. Scene Description: Since the first frame largely provides the scene information, only a brief description is needed here. 2. Garment Description: The person’s original clothing in the firstframe image [Image1] must be used as an anchor, with its type described but without detailed description, and integrated with the new garment from the target garment image [Image2], whose type should be described but without unnecessary details. In the description, directly assign the new garment to the person. It is strictly forbidden to use words that describe a dynamic transformation process, such as “changed into,” “switched to,” “turned into,” or similar expressions. The new garment should already be part of the person’s outfit in the initial state. 3. Detail Description: Pay attention to describing accessories, backpacks, handheld items, and similar details of the person in the first-frame image [Image1]. Ignoring these details may degrade the performance of the I2V task. The description should focus on the person wearing the new garment in the original scene. 4. Consistency Preservation: When the target garment image [Image2] provides only part of the outfit, the description must logically and coherently match it with the remaining clothing from [Image1], ensuring overall character consistency. Do not describe the old garment in the first-frame image [Image1] that has been replaced, because it has already been discarded and replaced by the new garment. 5. Hallucination Avoidance: It is strictly forbidden to use any adjectives describing visual style, such as “cinematic,” “highdefinition,” or “hyper-realistic.” It is strictly forbidden to fabricate objects that do not exist in the first-frame image.

##### II. Dynamic Description (one item must be selected from each category):

- 1. Overall Body Movement: One of the following must be randomly included: (a) Runway Walk / Walking: Walking forward toward the camera, walking away with the back facing the camera, catwalk, etc. Reasonable imagination is allowed, but the motion must be coherent and should avoid causing collapse. (b) Turning / Spinning: Slightly turning to the left, turning backward, etc. Reasonable imagination is allowed, but the motion must be coherent and should avoid causing collapse. (c) Posture Transition: Standing up from a seated pose, sitting down from a standing pose, shifting from leaning to upright, etc. Reasonable imagination is allowed, but the motion must be coherent and should avoid causing collapse. (d) Stretching / Extending: Raising both arms horizontally, stretching upward, turning sideways to show the back or side cut, etc. Reasonable imagination is allowed, but the motion must be coherent and should avoid causing collapse.
- 2. Facial Expression and Head Movement: One of the following must be randomly included: (a) Gaze Shift: From looking down to looking at the camera, glancing sideways and then turning back, blinking with a slight smile, etc. Reasonable imagination is allowed, but the motion must be coherent and should avoid causing collapse. (b) Head Movement: Tilting the head left and right for display, hair-swaying motion, etc. Reasonable imagination is allowed, but the motion must be coherent and should avoid causing collapse. (c) Notes: (i) Motion Stability: The description must logically include an “initial transition phase,” a “dynamic display phase,” and a “final freeze phase,” but these do not need to be explicitly written out. Reasonable imagination is allowed, but the motion must be coherent and should avoid causing collapse. (ii) Physical Plausibility: The range of motion must be kept within a reasonable scope. For example, turning should preferably not exceed 90 degrees to avoid limb deformation. The motion must follow gravity, such as the hem of the garment moving with the body during turning and the arms swinging naturally. (iii) First-Frame Continuity: If there is a large difference between the pose in the first frame and the subsequent pose, the description should provide a gradual, coherent, and smooth transition to avoid collapse. (iv) Hallucination Avoidance: It is strictly forbidden to use any adjectives describing visual style, such as “cinematic,” “high-definition,” or “hyper-realistic.”

##### III. Output Requirements:

- 1. Each set of descriptions must be strictly limited to within 150 words.
- 2. Output format: JSON, generating both Chinese (cn short) and English (en short) descriptions.

We present the system prompt for Gemini-3.0 to evaluate garment consistency below:

System Prompt: Task Objective: Evaluate the quality of AI-customized video generation. Input Description: [Image 1]: Original image of the model (reference for person identity, pose, non-target clothing, and original background). [Image 2]: Target garment image (reference for the clothing to be virtually tried on). [Video 1]: AI-generated video sequence (evaluation target). Scoring Dimensions: 1–5, where 1 is the worst and 5 is the best:

- 1. High-level garment consistency Evaluate how well the target garment in [Image 2] matches the garment worn by the model in the corresponding [Video 1] sequence at a high-level semantic level. Checkpoints: whether the category, overall silhouette (e.g. fit, cut), large color block distribution, and overall style are consistent. Execution focus: Only evaluate the match of the target garment from [Image 2]; ignore other non-target garments worn by the model in [Video 1].
- 2. Low-level garment consistency Evaluate how well the target garment in [Image 2] matches the garment worn by the model in the corresponding [Video 1] sequence at the pixel-detail level. Checkpoints: whether fine-grained features are accurately reproduced, such as patterns (prints, stripes), logos, embroidery, fabric texture, surface gloss, etc. Hard constraint: If there is obvious pattern distortion, blurred logos, or completely incorrect fabric texture, this score must be <= 2. Execution focus: Only evaluate the match of the target garment from [Image 2]; ignore other non-target garments worn by the model in [Video 1].
- 3. Non-target garment preservation Evaluate the consistency between the non-target garments in [Video 1] (i.e. all other clothing items, accessories, etc. worn by the model besides the target garment) and those in [Image 1]. Checkpoints: whether the style, color, and texture of all non-target parts in [Video 1] have been incorrectly modified or removed. Logical focus: Apart from virtually trying on the target garment, all other garments on the model in [Video 1] should preserve the original appearance from [Image 1] as much as possible. Output Requirements: Please output only a single JSON object, without any explanatory text: { “high-level garment consistency”: 0-5, “low-level garment consistency”: 0-5, “non-target garment preservation”: 0-5 }

