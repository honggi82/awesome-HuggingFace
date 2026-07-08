# arXiv:2506.19838v3[cs.CV]28Sep2025

## SimpleGVR: A Simple Baseline for Latent-Cascaded Generative Video Super-Resolution

#### Liangbin Xie1,2∗ Yu Li3 Shian Du3 Menghan Xia4† Xintao Wang4 Fanghua Yu2 Ziyan Chen2 Pengfei Wan4 Jiantao Zhou1† Chao Dong2,5

- 1State Key Laboratory of Internet of Things for Smart City, University of Macau
- 2Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

3Tsinghua University 4Kuaishou Technology 5Shenzhen University of Advanced Technology

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

“An adorable giant panda with soft, fluffy fur and a sweet expression …”

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

“A cute little white chick wearing black-framed glasses …”

(b) Large T2V (384×672) + SimpleGVR (c) Large T2V (384×672) (d) FlashVideo (e)SimpleGVR

(a) Large T2V (1080p)

Figure 1: Built upon the low-resolution latent outputs (e.g., 384 × 672 resolution) from the first-stage Large T2V model, SimpleGVR generates high-quality results that even surpass the 1080p outputs of the Large T2V model. Compared to FlashVideo, which also adopts a cascaded architecture, SimpleGVR produces more realistic and finer details.

### Abstract

Latent diffusion models have emerged as a leading paradigm for efficient video generation. However, as user expectations shift toward higher-resolution outputs, relying solely on latent computation becomes inadequate. A promising approach involves decoupling the process into two stages: semantic content generation and detail synthesis. The former employs a computationally intensive base model at lower resolutions, while the latter leverages a lightweight cascaded video superresolution (VSR) model to achieve high-resolution output. In this work, we focus on studying key design principles for latter cascaded VSR models, which are underexplored currently. First, we propose two degradation strategies to generate traininpairs that better mimic the output characteristics of the base model, ensuring alignment between the VSR model and its upstream generator. Second, we provide critical insights into VSR model behavior through systematic analysis of (1) timestep sampling strategies, (2) noise augmentation effects on low-resolution (LR) inputs. These findings directly inform our architectural and training innovations. Finally, we introduce interleaving temporal unit and sparse local attention to achieve efficient training and inference, drastically reducing computational overhead. Extensive experiments demonstrate the superiority of our framework over existing methods, with ablation studies confirming the efficacy of each design choice. Our work establishes a simple yet effective baseline for cascaded video super-resolution generation, offering practical insights to guide future advancements in efficient cascaded synthesis systems.

∗ Work done during an internship at KwaiVGI, Kuaishou Technology. † Corresponding authors

Preprint.

### 1 Introduction

Recent advancements [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in diffusion-based text-to-video (T2V) generation have markedly enhanced the visual quality and coherence of synthesized videos. Leading models, such as Hunyuan [11], CogXVideo [14] and Wan [13], rely on DiT backbones with full self-attention to fuse spatial, temporal, and textual cues, producing coherent clips with rich detail. However, their computational cost grows quadratically with spatial resolution: directly generating 1080p output in a single stage demands prohibitive computation and incurs long inference times. Given their multi-resolution training paradigm, running these models at lower resolution (e.g., 5122) strikes an optimal balance between visual fidelity and computational efficiency. We define 512p as a resolution whose total pixel area is nearly 5122. Since 512p outputs capture motion and structure well, applying a lightweight cascaded VSR model to enhance them to 1080p offers a promising solution for efficient high-resolution video synthesis with low computational cost and latency.

Existing works, such as VEnhancer [16], SeedVR [17], and FlashVideo [18], have demonstrated strong performance in video enhancement area. However, all these methods cannot operate directly on low-resolution latent representations. To apply these methods, users must first decode the lowresolution latent into a low-resolution video, apply upsampling, and then re-encode the video into a high-resolution latent before performing detail refinement. Consequently, this process introduces significant computational overhead. In this work, we propose SimpleGVR, the first method to support upsampling directly on the low-resolution latent representations produced by upstream T2V models, thereby eliminating redundant decoding and re-encoding steps and improving overall efficiency.

SimpleGVR adopts a simple yet effective architecture. It incorporates conditional information via concatenation and enables full-parameter finetuning. To improve SimpleGVR’s performance, we focus on several key aspects. One aspect is the design of degradation pipeline, which is critical and has been extensively explored in real-world image and video restoration literature. In contrast, the degradation pipeline in AIGC remains underexplored. Unlike real-world degradations, which often stem from well-defined physical processes like image signal processing (ISP) pipelines, degradations in AIGC content lack clear physical analogs. Directly applying two-stage degradation strategies, like those in RealBasicVSR [19], leads to severe artifacts and impaired depth perception when enhancing AIGC videos. To mimic the output characteristics of the computationally intensive base model (Large T2V model), we propose two degradation strategies: (1) Flow-based degradation, where optical flow guides motion-aware color blending and adaptive blurring, and (2) Model-guided degradation, where noise is added to low-resolution video frames and partially denoised using the base T2V model. These strategies generate training pairs that more faithfully reflect the characteristics of base T2V model output, narrowing the degradation gap compared to second-order degradation approach.

Another crucial aspect lies in the training configuration, which encompasses both timestep sampling strategy and noise augmentation to the low-resolution (LR) inputs. These components significantly affect SimpleGVR’s ability to enhance fine details and correct structural errors. Specifically, motivated by the observed differences in SimpleGVR’s capacity to reconstruct high-frequency details across different timestep, we propose a detail-aware sampler that outperforms the commonly used uniform sampler. Besides, the range of noise augmentation during training modulates the model’s capacity for structural modification to the input LR, in addition to details generation. We find that using a middle-range noise level (e.g., 0.3 ∼ 0.6) strikes a good balance between enhancing high-frequency details and correcting structural errors in the input frames.

To enable SimpleGVR to handle more practical scenarios, such as processing 5-second videos (i.e., 77 frames), we further explore strategies for efficient training and inference. Due to GPU memory constraints, training directly on full 77-frame sequences is impractical. To address this, we adopt an efficient training strategy: starting with training on shorter 17-frame clips, then extending SimpleGVR to 77 frames through the interleaving temporal unit mechanism. Expanding SimpleGVR’s capability to handle 77 frames requires only 5K additional iterations. To support efficient inference, we further replace the full self-attention with the sparse local attention. This reduces computational cost of full self-attention by 80% while maintaining comparable performance. Compared to swin attention [20], the sparse local attention mechanism achieves better detail reconstruction with less computation, striking a good balance between efficiency and quality.

In summary, our main contributions are as follows: 1) We present SimpleGVR, the first lightweight model that performs video super-resolution directly on the latent representations of large T2V

models. When applied to 512p outputs, SimpleGVR produces 1080p videos with higher quality than directly generated 1080p outputs from the same base Large T2V model. 2) We design two degradation schemes, namely flow-based degradation and model-guided degradation synthesis, to simulate the degradation characteristics of the base model’s outputs. This ensures better alignment between the VSR model and its upstream generator. 3) We analyze the timestep sampler and noise augmentation, and introduce a detail-aware sampler along with an appropriate noise augmentation range, which together enhance the model’s ability to recover fine details and correct structural errors. 4) We introduce the interleaving temporal unit mechanism and incorporate the sparse local attention mechanism into SimpleGVR, enabling efficient training and inference while supporting long-sequence, high-resolution video generation.

### 2 Related Work

- 2.1 Cascade Diffusion Models.

Cascade architectures have been widely explored in text-to-image and text-to-video generation [21, 22, 18], where multi-stage designs are employed to address the challenge of generating high-resolution outputs. Typically, a low-resolution sample is first generated, followed by a specialized model to progressively refine details. Among these, FlashVideo [18] is most related to our work. It begins second-stage generation from low-quality video inputs rather than pure guassian noise, enabling efficient high-resolution synthesis with only 4 function evaluations. However, SimpleGVR differs in two key aspects. First, we concatenate the low-resolution latent as a condition, allowing the model not only to leverage coarse content but also to correct structural errors within it. This design leads to better performance under the same 50-step inference setting. Second, we introduce two degradation strategies that explicitly simulate the characteristic degradations from the first-stage T2V generator.

- 2.2 Degradation Models in Restoration.

Degradation modeling is important for effective image and video restoration. Traditional models [23, 24, 25] often rely on simple assumptions like bicubic downsampling or gaussian blur, which fail to capture complex real-world degradations. Prior works such as BSRGAN [26], RealESRGAN [27], and video-oriented methods like RealBasicVSR [19] simulate more complicated degradations, including blur, noise, and compression, improving robustness on real-world low quality images and videos. However, these models are designed for real-world scenarios and do not account for the unique distortions in AIGC-generated videos, such as motion blur and color blending. These AIGC-specific artifacts require specialized degradation modeling. To this end, we propose a heuristic degradation strategy consisting of two complementary components: a flow-based degradation scheme and a model-guided degradation scheme via SDEdit. Together, they enable the generation of training pairs that more faithfully mimic the output characteristics of the first-stage T2V generator.

- 2.3 Video Restoration.

Early video restoration (VR) methods [28, 29, 30, 31, 32, 33, 34] rely on synthetic data, limiting real-world performance. Later works [19, 35, 36] shift toward real scenarios but still struggle with texture realism. Diffusion-based approaches [16, 21, 37, 17, 18] leverage generative priors to achieve more realistic and coherent video restoration. However, all these methods require decoded RGB frames and cannot operate directly on latent representations, making them less suitable for T2V pipelines. In contrast, our SimpleGVR performs upsampling and refinement directly in the latent space of the upstream generator, enabling seamless integration with generative video models. Unlike SeedVR, which uses a fixed window attention mechanism, SimpleGVR adopts a more advanced local attention strategy that achieves better performance under the lower computational budget.

### 3 Preliminary

SimpleGVR is conducted over an internal pre-trained text-to-video foundation model, which is composed of a 3D Variational Autoencoder (VAE) [38], a T5-based text encoder [39], and a transformerbased latent diffusion module [40, 41]. The generative transformer operates over latent representations and is structured with a repeated stack of components: 2D spatial self-attention, 3D spatiotemporal

attention, text-guided cross-attention, and feed-forward layers. The input text prompt is encoded by the T5 model into a conditioning vector ctext, which guides the generation model. We follow the Rectified Flow framework [42] to define a linear path between the clean latent z0 and its noisy counterpart zt at timestep t:

zt = (1 − t)z0 + tϵ, ϵ ∼ N(0,I) (1) The denoising process is formulated as ordinary differential equation (ODE) that maps zt back to z0:

dzt = vΘ(zt,t,ctext)dt, (2)

where the velocity field v is modeled by a neural network with parameters vΘ. During training, Conditional Flow Matching (CFM) [43] is used to regress the velocity via the following objective:

0 ∥(z1 − z0) − vΘ(zt,t,ctext)∥22 . (3)

LCFM = Et,ϵ∼N(0,I),z

### 4 Methodology

Our cascaded video generation framework operates within a latent space defined by a pre-trained VAE. The framework comprises two core components: (i) A computationally intensive base Textto-Video (T2V) model, which employs a Diffusion Transformer (DiT [44]) architecture to generate low-resolution video latent representations. (ii) A cascaded latent video super-resolution model, termed SimpleGVR, which adopts a lightweight architecture to efficiently enhance the base model’s output into high-resolution video latent representations. The overall framework structure is illustrated in Fig. 2(b). As the primary focus of this paper, our method addresses the task of the latter component.

In the following sections, we first formalize the problem addressed by SimpleGVR (Sec. 4.1). We then investigate this critical yet underexplored challenge through three key perspectives: Degradation simulation for synthesizing training pairs (Sec. 4.2); Training configurations to promote faithful detail generation (Sec. 4.3); Efficient designs to address high-resolution video computation demands (Sec. 4.4). Note that the 512p is defined as a resolution whose total pixel area is approximately 5122.

#### 4.1 Formulation of SimpleGVR

Fig. 2 (a) and (b) illustrate the training and inference pipelines of SimpleGVR, respectively. To optimize SimpleGVR, we adopt a diffusion model. During training, the 512p low-resolution (LR) video and the corresponding 1080p high-resolution (HR) video are first encoded into latent representations via a 3D VAE [45], yielding LR and HR latents. To align the spatial dimensions of the LR latent with the HR latent, we apply a 3D CNN followed by a bilinear upsampling operator, and then another 3D CNN layer. Two independent random noises are then injected into both latents with different magnitudes, yielding noisy representations ct and zt. Here, zt refers to the noisy latent in the diffusion process, while ct denotes the noisy LR latent that serves as the conditioning input. To distinguish the roles of these two noise injections, we refer to the perturbation applied to the LR latent as noise augmentation [16, 17], which plays a crucial role in enhancing the model’s capacity for detail generation and correcting structural errors. This effect is further analyzed in later sections. The noisy LR and HR latents are then fused via channel-wise concatenation and fed into a series of DiT blocks. All parameters, including those of the 3D CNN and DiT blocks, are optimized jointly in an end-to-end fashion during training.

Once trained, SimpleGVR can be directly applied to the low-resolution latents generated by the large T2V model. Specifically, given a random gaussian noise cT, the large T2V model performs multiple denoising steps to produce a clean low-resolution latent c0. This latent is then upsampled and perturbed with a fixed level of random noise to yield c. Concurrently, a high-resolution gaussian noise sample zT is randomly initialized. The noisy high-resolution latent zT and the conditioned latent c are concatenated along the channel dimension and fed into the DiT blocks of SimpleGVR. Notably, the conditioning latent c remains fixed throughout the denoising process. After the iterative refinement, the final clean high-resolution latent z0 is decoded to obtain a high-fidelity 1080p video.

[Figure 11]

Text Prompt

Text Encoder

“A young tiger cub is lying on the ground.”

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Temporal Attention

Self Attention

Cross Attention

HR Video (1080p)

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

+𝑛𝑜𝑖𝑠𝑒

3D VAE Encoder

FFN

| |
|---|

| |
|---|

C

𝑧̂

𝑧

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

DiT Block × 𝑁

3DCNN

3DCNN

[Figure 25]

[Figure 26]

[Figure 27]

bilinear

+𝑛𝑜𝑖𝑠𝑒

[Figure 28]

LR Video (512p)

3D VAE Encoder

| |
|---|

𝑐

###### (a) Training pipeline of SimpleGVR at timestep 𝑡

“A cat is playing chess, focusing on the cat's playful interaction with the game pieces.”

Text Encoder

Text Prompt

[Figure 29]

[Figure 30]

𝑇 𝑇

[Figure 31]

+fix noise

Temporal Attention

Self Attention

Cross Attention

3D VAE Decoder

[Figure 32]

[Figure 33]

3DCNN

[Figure 34]

[Figure 35]

3DCNN

FFN

bilinear

| |
|---|

C

| |
|---|

Large T2V Model

𝑧

𝑐 c

𝑐

[Figure 36]

[Figure 37]

[Figure 38]

DiT Block × #

| |
|---|

𝑧

(b) Inference pipeline of SimpleGVR

- Figure 2: The training and inference pipeline of SimpleGVR. Unlike previous methods, SimpleGVR

performs upsampling directly at the latent level. This design allows the latent representation c0 produced by the upstream Large T2V model to be directly processed by SimpleGVR during inference, eliminating redundant decoding and re-encoding steps.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

- Figure 3: Visual artifacts in decoded videos from the Large T2V model. Dynamic regions exhibit noticeable local motion blur and color blending distortions.

- 4.2 Degradation Modeling

- 4.2.1 Flow-based degradation

Upon inspecting the first-stage T2V outputs (see Fig. 3), we observe that unlike real-world lowquality videos, these video sequences do not exhibit severe degradations such as severe blur, noise, or compression. Instead, they present two distinctive characteristics: 1) localized motion-dependent blur and (2) frame-to-frame color blending distortion (current-frame colors smeared by previous-frame hues). These two phenomena are closely entangled and cannot be effectively replicated using traditional degradation model [19]. To synthesize color blending distortion, we first estimate the motion field between adjacent frames using Dense Inverse Search (DIS) [46] optical flow algorithm, mapping each point (x,y) in the current frame to its position (x′,y′) in the previous frame. Based on the optical flow results, we generate motion masks to identify regions with significant movement and create elliptical patterns within these regions. For each elliptical region, we apply gaussian sampling to extract colors from the previous frame, blend them in RGB space, and generate the colors. We then blend these colors into the current frame using a distance-based weight map, where pixels closer to the ellipse center receive stronger effects.

After applying color blending, we synthesize motion blur effects on the processed frames. Based on the motion field computed earlier, for each block in the frame, we generate adaptive blur kernels whose parameters are determined by the local motion characteristics. The kernel size and shape vary

Text Encoder

“A young tiger cub is lying on the ground.”

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

1-𝛼 3DVAE Decoder

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

3D VAE Encoder

Large T2V Model

𝛼

𝑐 c

[Figure 56]

[Figure 57]

HR Video (1080p)

Synthesized LR Video (512p)

Downsampled LR Video (512p)

Gaussian noise

- Figure 4: Model-guided degradation synthesis pipeline. The parameter α controls the strength of the added Gaussian noise, which also affects the structural alignment between cˆ0 and c0.

according to the motion magnitude, with larger motion leading to longer blur kernels aligned with the motion direction. We apply these kernels through a weighted convolution operation, where the blur intensity is proportional to the local motion magnitude. This block-wise processing ensures that static regions maintain their sharpness while areas with significant movement exhibit realistic motion blur.

- 4.2.2 Model-guided degradation

SimpleGVR is fundamentally designed to learn a mapping from the output domain of large T2V models to high-quality video data. By constructing paired training samples where the low-resolution inputs are directly sourced from the T2V model outputs, SimpleGVR can be better aligned with the distribution and artifacts specific to the T2V model. Inspired by SDEdit [47], which injects noise into the input and leverages a diffusion model as the structural prior to produce outputs that are both visually realistic and structurally faithful, we adopt a similar approach in our framework. As shown in Fig. 4, we begin by downsampling a high-quality 1080p video to 512p and encoding it via a 3D VAE to obtain the latent c0. This latent is blended with a gaussian noise under a predefined ratio α, and the noisy latent is then partially denoised using the large T2V model to generate cˆ0. A higher α pushes cˆ0 closer to the T2V distribution but weakens its structural alignment with the original video. To balance realism and fidelity, we set α ∈ [0.3,0.4], ensuring that cˆ0 retains the overall layout of the source video while approximating the output domain of the Large T2V model. The synthesized LR video produced by the 3D VAE encoder can be used as the LR input during training (see Fig. 2(a)). In practice, to reduce the overhead of decoding and re-encoding, we store only the latent representation cˆ0, which is directly used as input to the 3D CNN.

- 4.3 Training Configuration

To further enhance SimpleGVR’s ability to recover fine details and correct structural errors in the input frames, we analyze two key training configurations: the timestep sampling scheduling and the noise augmentation applied to the low-resolution (LR) branch.

[Figure 58]

- Figure 5: High-frequency variation curve over timesteps during inference.

|Sampler|MUSIQ|DOVER Technical Aesthetic Overall<br><br>|
|---|---|---|
| | | |
|Uniform|62.04<br><br>|18.58 98.78 68.94|
|Detail-aware<br><br>|62.19|18.92 98.83 69.64|

Figure 6: Quantitative comparison between the uniform sampler and the detail-aware sampler on the AIGC100 dataset, demonstrating that the detail-aware sampler outperforms the uniform sampler in most metrics. These experiments are conducted on 17-frame inputs for 20K iterations.

Timestep Sampling Scheduling. The uniform sampler, where each timestep is sampled with equal probability, is commonly used for training diffusion-based models. However, since SimpleGVR

Train NA: 0.3~0.6 Train NA: 0.6~0.9

Train NA: 0.0~0.3

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Test NA: 0.0

Test NA: 0.3

Test NA: 0.6

[Figure 63]

[Figure 64]

[Figure 65]

Input LR Test NA: 0.8

Test NA: 0.2

Test NA: 0.5

Figure 7: Visual results of SimpleGVR trained with different noise augmentation (NA) ranges. When the NA during training is small, SimpleGVR struggles to remove artifacts from the low-resolutn video. When the NA is large, SimpleGVR has difficulty preserving the original structure of the low-resolution video. Only when the NA is in a moderate range does SimpleGVR strike a good balance between correcting structural errors and enhancing high-frequency details.

focuses on detail synthesis, understanding which timesteps contribute most to enhancing visual details during denoising is crucial. To this end, we analyze high-frequency detail changes in the predicted zˆt0 at each denoising step. Specifically, we sample 200 low-resolution 512p test videos and perform 50-steps inference using the SimpleGVR model trained with a uniform sampler. At each denoising timestep t, we obtain the latent zt and directly predict its corresponding clean signal zˆt0. To quantify the high-frequency content of zˆt0, we apply discrete cosine transform (DCT) and extract its high-frequency coefficients H(ˆzt0). We then compute the pairwise differences of these high-frequency components across timesteps to derive the detail variation curve shown in Fig. 5. The figure shows that significant detail gains primarily occur in the high-noise and mid-noise regions, while the lownoise region contributes minimally. Based on this observation, we normalize the weights to create a probability distribution, which we then adopt as the detail-aware sampler. Replacing the uniform sampler with this detail-aware sampler during training leads to improved performance, as evidenced by the results in Table 6.

Noise augmentation Effect. As noted in prior work [16, 17, 18], introducing noise in the LR branch during training not only enhances the generative capacity of video enhancement models but also facilitates controllable behavior at inference time. To identify a suitable noise range for SimpleGVR, we divide the noise scale into three intervals: small (0.0 ∼ 0.3), middle (0.3 ∼ 0.6), and large (0.6 ∼ 0.9). By training and testing the model separately within each interval, we observe that different noise levels lead to distinct behaviors. When trained with noise in the small interval, the model exhibits limited ability in refining fine details. In particular, when the input video frames contain structural errors, increasing the noise level in the LR branch tends to introduce more artifacts rather than meaningful corrections, as illustrated by the red box in Fig. 7. In contrast, training with large noise results in significant deviations in structural content, causing the output to diverge from the input. Notably, only the middle interval strikes a good balance: the model is capable of enhancing high-frequency details while still being able to correct structural errors in the input frames.

17 frames

#### 4.4 Efficient Pseudo-Global Computation

5

Stage2 latent 5f

17 − 1 4

+ 1 = 5

Interleaving Temporal Unit. Fig. 8 illustrates the design of the interleaving temporal unit mechanism for a 77-frame input. When the input frame count is 17 or 77, the latent vectors obtained via the VAE are 5 and 20 along the time dimension, respectively. Due to GPU memory limitations, we are unable to process the entire 77-frame data at once. Therefore, we first train a SimpleGVR model capable of handling 17 frames, and then extend it to 77 frames. During this extension, we load all 77 frames at once, then the corresponding latent is sliced along the

𝑙

5 5 5 5

shift half window size

𝑙

77 frames

5 5 5

2 3

Stage3 latent 20f

shift half window size

77 − 1 4

+ 1 = 20

𝑙

5 5 5 5

Figure 8: Visualization of the interleaving temporal unit mechanism.

time dimension into units of 5, with attention computations performed within each unit. To facilitate information exchange between units, inspired by Swin Attention [20], we apply a shift operation on the entire latent at the layer l2k+1, and then perform another shift at the layer l2k+2. Through this interleaved temporal unit design, we achieve highly efficient training, and the final model can effectively handle 77-frame video sequences.

Sparse Local Attention. To reduce computational overhead during inference, we further explore replacing the full attention in SimpleGVR with swin attention. However, despite the use of the shifted window mechanism, this modification leads to noticeable reduction in fine details. We hypothesize that this is due to its limited receptive field and lack of dynamic cross-window interactions. Inspired by MoBA [48] and swin attention’s window partitioning strategy, we divide tokens into 2D window units of fixed size. Each unit performs self-attention within itself and dynamically attends to the top-k most relevant window units across the entire video, including future frames, based on computed relevance scores. This mechanism enables effective long-range modeling and detail preservation, even with small window sizes and small top-k values (e.g., 1).

### 5 Experiments

- 5.1 Implementation Details

- 5.1.1 Training Dataset

We collect approximately 840K high-quality video clips from the Internet to construct our training dataset. Given the highly variable quality of online videos, we design an automated filtering pipeline to retain only visually high-quality content. Specifically, we first discard videos that are overly bright or dark. Then, for each video, we uniformly sample 10 frames and compute two metrics: the average MUSIQ score [49] and the Laplacian variance, which reflects the level of spatial detail or sharpness. Videos with an average MUSIQ score below 40 or a Laplacian variance below 30 are discarded.

- 5.1.2 Testing Dataset

To ensure diversity in our test samples, we first decode the latent representations from the T2V base model into pixel space. Based on this output, we collect a set of 100 video clips across diverse scenarios, including human, animal, object, motion, and defocus scenes. We refer to this curated set as AIGC100, with each clip containing 77 frames.

- 5.1.3 Metrics

Since AIGC100 is generated by the first-stage Large T2V model, ground-truth references are not available for full-reference evaluation. Consequently, we adopt a suite of no-reference metrics to assess both frame-level and video-level quality. Specifically, we use MUSIQ [49] for single-frame perceptual quality, DOVER [50] for overall video quality, and a set of metrics from VBench [51], which evaluate diverse aspects of AIGC videos, including background consistency, subject consistency, aesthetic quality, imaging quality, and motion smoothness.

- 5.1.4 Training Details

SimpleGVR is trained on 16 NVIDIA H800 GPUs (80GB each) with a total batch size of 32. We use the AdamW optimizer [52] with a learning rate of 5 × 10−5, and randomly replace the text prompt with a null prompt in 10% of cases to enhance robustness. The training pipeline is divided into three stages. In the first stage, initialized from a pretrained 1B T2V model, SimpleGVR is trained for 20K iterations on 17-frame inputs using training pairs constructed via a one-order degradation process. In the second stage, we fine-tune the model for an additional 5K iterations on a dataset (30K) generated using the proposed degradation strategies. In the third stage, based on the dataset synthesized in the previous two stages, we continue fine-tuning SimpleGVR and extend the temporal range to 77 frames by using the temporal window attention training strategy. Note that we also implement the sparse local attention mechanism in this stage. During the whole training pipeline, the LR branch is injected with noise sampled from the range [0.3,0.6], corresponding to diffusion timesteps randomly selected from [300,600].

- 5.2 Comparison with SOTA methods

We compare SimpleGVR with existing state-of-the-art methods, RealBasicVSR [19], Upscale-AVideo [53], VEnhancer [16], STAR [54] and FlashVideo [18]. For fair comparison, we set the inference steps of FlashVideo to 50. As shown in Table 1, SimpleGVR achieves the best performance on both MUSIQ and DOVER. Moreover, regarding the comprehensive metrics proposed in VBench, SimpleGVR also achieves the highest average score. Qualitative comparisons are presented in Fig. 9. Compared to other methods, SimpleGVR adds realistic details while maintaining the original style and semantics. For example, while VEnhancer introduces unnatural textures around the panda’s eyes and alters the style, SimpleGVR maintains more natural details. Similarly, for human faces, SimpleGVR produces finer and more realistic details. In contrast, other methods either struggle to generate sufficient detail or create noticeable artifacts, resulting in outputs less realistic than those produced by SimpleGVR.

Table 1: Quantitative comparison on AIGC100 dataset. Bold and underline indicate the best and second best performance.

|Method|MUSIQ|DOVER|Vbench Metrics Background Subject Aesthetic Imaging Motion Average<br><br>|
|---|---|---|---|
| | |Technical Aesthetic Overall|Consistency Consistency Quality Quality Smoothness Score|
|RealBasicVSR VEnhancer Upscale-A-Video STAR Flashvideo|57.55 40.03 36.35 46.73 53.65<br><br>|12.27 98.66 61.84 15.38 98.32 62.54 12.43 98.29 59.04 18.17 98.66 67.76 15.97 98.61 65.38<br><br>|93.73 93.98 61.63 72.76 98.70 84.16<br><br>94.59 94.44 59.98 64.22 99.16 82.48<br>95.96 94.41 61.26 63.85 98.99 82.89<br>96.17 94.43 62.24 67.24 99.01 83.82 95.49 94.75 60.76 69.11 98.45 83.71<br>|
|Ours<br><br>|62.35|20.44 98.88 71.34<br><br>|95.35 94.32 62.84 71.91 98.74 84.63|

“A cute white chick wearing black-framed glasses perched on the end of its beak, …”

[Figure 66]

[Figure 67]

“A young lady with long, curly brown hair …”

STAR

RealBasicVSR

[Figure 68]

[Figure 69]

RealBasicVSR

STAR

[Figure 70]

[Figure 71]

FlashVideo

Upscale-A-Video

[Figure 72]

[Figure 73]

Upscale-A-Video

FlashVideo

[Figure 74]

[Figure 75]

SimpleGVR

VEnhancer

[Figure 76]

[Figure 77]

VEnhancer

SimpleGVR

[Figure 78]

[Figure 79]

- Figure 9: Qualitative comparisons on AIGC100 dataset. Our SimpleGVR is capable of generating more realistic details than our methods.

- 5.3 Ablation Study

- 5.3.1 Effectiveness of Degradation Strategies

In Fig. 10, we demonstrate the effectiveness of the proposed degradation strategies. As shown in the first and third rows, the SimpleGVR trained solely with one-order degradation exhibits noticeable

temporal inconsistencies in motion areas, such as the panda’s paw, and suffers from color blending artifacts, particularly evident in the human arm. After fine-tuning with training pairs generated by the two proposed degradation schemes, SimpleGVR effectively mitigates abrupt changes in motion regions across consecutive frames and significantly reduces color blending distortions.

[Figure 80]

[Figure 81]

[Figure 82]

w/odegradationFTw/odegradationFTwithdegradationFTwithdegradationFT

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

- Figure 10: Visualization of three consecutive frames generated by SimpleGVR. "w/o degradation FT" indicates SimpleGVR is trained only with one-order degradation, without fine-tuning using the proposed degradation strategies.

#### 5.3.2 Effectiveness of Sparse Local Attention

To verify the effectiveness of sparse local attention, we train SimpleGVR with different attention operations on 17-frame inputs for 20K iterations. As shown in Table 2, compared to Swin Attention, we observe that sparse local attention achieves better performance with lower computational cost. In our setting, the block size is set to 12 × 9, and each block attends to only one additional neighboring block. Under this configuration, sparse local attention reduces computational cost by approximately 80% compared to full attention, while maintaining nearly the same performance. These results show the promising potential of sparse local attention mechanisms for video super-resolution tasks.

Table 2: Computational cost and performance comparison.

|Attention Operator<br><br>|FLOPS (G)|MUSIQ<br><br>|DOVER Technical Aesthetic Overall<br><br>|
|---|---|---|---|
| | | | |
|Full Attention Swin Attention Sparse Local Attention|1612.5 452.6 377.1<br><br>|62.19 59.33 61.64|18.92 98.83 69.64<br><br>17.80 98.73 67.95<br><br>18.51 98.81 69.11<br>|

#### 5.4 Comparison in T2V: End-to-End vs. Cascaded

We also compare the performance of two different T2V paradigms: a large T2V model that directly generates 1080p videos (i.e., end-to-end), versus a large T2V model that first produces 512p latent representations followed by the SimpleGVR module to generate 1080p outputs (i.e., cascaded). As shown in Tab. 3, the cascaded paradigm achieves significantly higher performance on quality metrics than the end-to-end paradigm. On other metrics that measure diverse aspects of videos (i.e., smoothness and consistency), the results under both paradigms are comparable.

Table 3: Quantitative comparison between two different T2V paradigms on AIGC100 dataset.

|Method|MUSIQ<br><br>|DOVER|Vbench Metrics Background Subject Aesthetic Imaging Motion Average<br><br>|
|---|---|---|---|
| | |Technical Aesthetic Overall<br><br>|Consistency Consistency Quality Quality Smoothness Score|
|End-to-End Cascaded<br><br>|56.77 62.35<br><br>|18.82 97.27 62.32 20.44 98.88 71.34|96.04 95.16 63.45 67.69 98.89 84.25 95.35 94.32 62.84 71.91 98.74 84.63<br><br>|

### 6 Conclusions

In this work, we propose SimpleGVR, a cascaded latent video super-resolution model designed to efficiently enhance the output of the Large T2V model into high-resolution video latent representations. To better align SimpleGVR with the base generator, we introduce two degradation strategies for synthesizing training pairs. We also investigate the effects of noise augmentation and timestep sampling to promote more accurate detail generation. In addition, to address the high computational cost of global full-attention, we propose two efficient design adaptations. Experimental results demonstrate the superiority of our framework, providing a strong baseline for future advancements in efficient cascaded VSR systems.

### References

- [1] X. Chen, Y. Wang, L. Zhang, S. Zhuang, X. Ma, J. Yu, Y. Wang, D. Lin, Y. Qiao, and Z. Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023.
- [2] R. Fridman, A. Abecasis, Y. Kasten, and T. Dekel. Scenescape: Text-driven consistent scene generation. volume 36, 2024.
- [3] S. Yin, C. Wu, H. Yang, J. Wang, X. Wang, M. Ni, Z. Yang, L. Li, S. Liu, F. Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023.
- [4] V. Voleti, A. Jolicoeur-Martineau, and C. Pal. Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. volume 35, pages 23371–23385, 2022.
- [5] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023.
- [6] D. J. Zhang, J. Z. Wu, J.-W. Liu, R. Zhao, L. Ran, Y. Gu, D. Gao, and M. Z. Shou. Show1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023.
- [7] O. Bar-Tal, H. Chefer, O. Tov, C. Herrmann, R. Paiss, S. Zada, A. Ephrat, J. Hur, Y. Li, T. Michaeli, et al. Lumiere: A spacetime diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024.
- [8] Y. Jiang, S. Yang, T. L. Koh, W. Wu, C. C. Loy, and Z. Liu. Text2performer: Text-driven human video generation. arXiv preprint arXiv:2304.08483, 2023.
- [9] A Polyak, A Zohar, A Brown, A Tjandra, A Sinha, A Lee, A Vyas, B Shi, CY Ma, CY Chuang, et al. Movie gen: A cast of media foundation models, 2025. URL https://arxiv. org/abs/2410.13720, page 51.
- [10] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.
- [11] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [12] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025.
- [13] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [14] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [15] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [16] Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667, 2024.
- [17] Jianyi Wang, Zhijie Lin, Meng Wei, Yang Zhao, Ceyuan Yang, Fei Xiao, Chen Change Loy, and Lu Jiang. Seedvr: Seeding infinity in diffusion transformer towards generic video restoration. arXiv preprint arXiv:2501.01320, 2025.
- [18] Shilong Zhang, Wenbo Li, Shoufa Chen, Chongjian Ge, Peize Sun, Yida Zhang, Yi Jiang, Zehuan Yuan, Binyue Peng, and Ping Luo. Flashvideo: Flowing fidelity to detail for efficient high-resolution video generation. arXiv preprint arXiv:2502.05179, 2025.
- [19] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5962–5971, 2022.
- [20] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.
- [21] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, 133(5):3059–3078, 2025.
- [22] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence, 45(4):4713–4726, 2022.
- [23] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part IV 13, pages 184–199. Springer, 2014.
- [24] Chao Dong, Chen Change Loy, and Xiaoou Tang. Accelerating the super-resolution convolutional neural network. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 391–407. Springer, 2016.
- [25] Jinjin Gu, Hannan Lu, Wangmeng Zuo, and Chao Dong. Blind super-resolution with iterative kernel correction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1604–1613, 2019.
- [26] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4791–4800, 2021.
- [27] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1905–1914, 2021.

- [28] Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Basicvsr: The search for essential components in video super-resolution and beyond. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4846–4855,

- 2021.

[29] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Basicvsr++: Improving video superresolution with enhanced propagation and alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1627–1636,

- 2022.

- [30] Zhikai Chen, Fuchen Long, Zhaofan Qiu, Ting Yao, Wengang Zhou, Jiebo Luo, and Tao Mei. Learning spatial adaptation and temporal coherence in diffusion models for video superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 432–441, 2024.
- [31] Dasong Li, Xiaoyu Shi, Yi Zhang, Ka Chun Cheung, Simon See, Xiaogang Wang, Hongwei Qin, and Hongsheng Li. A simple baseline for video restoration with grouped spatial-temporal shift. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1123–1132, 2023.
- [32] Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, and Luc V Gool. Recurrent video restoration transformer with guided deformable attention. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [33] Xintao Wang, Kelvin CK Chan, Ke Yu, Chao Dong, and Chen Change Loy. Edvr: Video restoration with enhanced deformable convolutional networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (CVPR-W), pages 195–204, 2019.
- [34] Geunhyuk Youk, Jihyong Oh, and Munchurl Kim. Fma-net: Flow-guided dynamic filtering and iterative feature refinement with multi-attention for joint video super-resolution and deblurring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2302–2311, 2024.
- [35] Liangbin Xie, Xintao Wang, Shuwei Shi, Jinjin Gu, Chao Dong, and Ying Shan. Mitigating artifacts in real-world video super-resolution models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2956–2964, 2023.
- [36] Yuehan Zhang and Angela Yao. Realviformer: Investigating attention for real-world video super-resolution. In European Conference on Computer Vision, pages 412–428. Springer, 2024.
- [37] Xiaohui Li, Yihao Liu, Shuo Cao, Ziyan Chen, Shaobin Zhuang, Xiangyu Chen, Yinan He, Yi Wang, and Yu Qiao. Diffvsr: Enhancing real-world video super-resolution with diffusion models for advanced visual quality and temporal consistency. arXiv preprint arXiv:2501.10110, 2025.
- [38] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.
- [39] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [40] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [41] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [42] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [43] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [45] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.
- [46] Till Kroeger, Radu Timofte, Dengxin Dai, and Luc Van Gool. Fast optical flow using dense inverse search. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 471–488. Springer, 2016.
- [47] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
- [48] Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, et al. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025.
- [49] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021.
- [50] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154, 2023.
- [51] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [52] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [53] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscalea-video: Temporal-consistent diffusion model for real-world video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2535–2545, 2024.
- [54] Rui Xie, Yinhong Liu, Penghao Zhou, Chen Zhao, Jun Zhou, Kai Zhang, Zhenyu Zhang, Jian Yang, Zhenheng Yang, and Ying Tai. Star: Spatial-temporal augmentation with text-to-video models for real-world video super-resolution. arXiv preprint arXiv:2501.02976, 2025.

