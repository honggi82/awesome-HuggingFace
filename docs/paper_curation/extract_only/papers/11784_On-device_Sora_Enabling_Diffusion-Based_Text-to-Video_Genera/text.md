# arXiv:2502.04363v2[cs.CV]31Mar2025

## On-device Sora: Enabling Training-Free Diffusion-based Text-to-Video Generation for Mobile Devices

Bosung Kim*, Kyuhwan Lee†, Isu Jeong, Jungmin Cheon, Yeojin Lee, Seulki Lee‡, Ulsan National Institute of Science and Technology

#### Abstract

We present On-device Sora, the first model training-free solution for diffusion-based on-device text-to-video generation that operates efficiently on smartphone-grade devices. To address the challenges of diffusion-based text-to-video generation on computation- and memory-limited mobile devices, the proposed On-device Sora applies three novel techniques to pre-trained video generative models. First, Linear Proportional Leap (LPL) reduces the excessive denoising steps required in video diffusion through an efficient leap-based approach. Second, Temporal Dimension Token Merging (TDTM) minimizes intensive token-processing computation in attention layers by merging consecutive tokens along the temporal dimension. Third, Concurrent Inference with Dynamic Loading (CI-DL) dynamically partitions large models into smaller blocks and loads them into memory for concurrent model inference, effectively addressing the challenges of limited device memory. We implement On-device Sora on the iPhone 15 Pro, and the experimental evaluations show that it is capable of generating high-quality videos on the device, comparable to those produced by high-end GPUs. These results show that Ondevice Sora enables efficient and high-quality video generation on resource-constrained mobile devices. We envision the proposed On-device Sora as a significant first step toward democratizing state-of-the-art generative technologies, enabling video generation on commodity mobile and embedded devices without resource-intensive re-training for model optimization (compression). The code implementation is available at a GitHub repository1.

#### 1. Introduction

Recent advancements in generative models [60] have significantly expanded capabilities in data generation across various modalities, including text [27], image [55], and

*Co-first author. †Co-first author. ‡Corresponding author. 1https://github.com/eai-lab/On-device-Sora

video [63]. In particular, diffusion-based models for image tasks [14, 23, 46, 47, 50, 53, 58, 78] have emerged as foundational tools for a wide range of applications, such as image generation [25], image editing [30], and personalized content creation [79]. Further extending these technologies, diffusion-based generative models are now driving remarkable progress in video generation tasks [4, 5, 21, 24, 26, 31, 41, 42, 56, 65, 68], including video synthesis [37] and real-time video analysis [44].

With the rapid expansion of mobile and embedded devices, there is an increasing demand for executing generative applications directly on-device. In response, extensive research has focused on developing on-device image generation methods [8, 12, 13, 35, 64, 81], with recent advancements extending to on-device video generation [70, 71]. However, generating lightweight and compressed on-device video generation models necessitates additional model optimization (training), such as distillation [36, 57, 66], quantization [62, 80], and pruning [69], to reduce model size and computational complexity. Furthermore, mitigating performance degradation in the compressed models requires a time-intensive and iterative model re-training. This process usually demands substantial computational and memory resources, e.g., SnapGen-V [70] takes over 150K training iterations on 256 NVIDIA A100 80GB GPUs to generate mobile-deployable compressed video generation models.

We introduce On-Device Sora, the first training-free framework that enables diffusion-based text-to-video generation on mobile devices, which allows video generative models to run directly on-device, enabling the production of high-quality videos on smartphone-grade devices. While previous approaches [70] require additional training and substantial GPU resources to optimize video generation models for mobile environments, the proposed On-Device Sora eliminates the need for training and directly enables efficient on-device execution. Using pre-trained video generative models, e.g., Open-Sora [83], Pyramidal Flow [29], On-device Sora significantly enhances their efficiency, enabling on-device video generation with limited resources.

To achieve this, we address key challenges of enabling on-device text-to-video generation: 1) Linear Proportional

Leap (LPL) reduces the iterative denoising steps in the diffusion process by leaping through a significant portion of steps using the Euler’s method [3] along an estimated direct trajectory, 2) Temporal Dimension Token Merging (TDTM) lowers computational complexity of STDiT (Spatial-Temporal Diffusion Transformer) [83] by merging consecutive tokens [6, 7] in the attention layers, and 3) Conference Inference and Dynamic Loading (CI-DL) enables video generation with limited memory capacity of mobile devices by integrating concurrent model inference with the dynamic loading of models into memory. With these three proposed methods, high-quality video generation becomes feasible on smartphone-grade devices with limited computing resources, overcoming the requirements for substantial computational power, such as high-end GPUs. To the best of our knowledge, On-device Sora proposed in this work is the first training-free solution that enables the efficient generation of video directly on the device. The advantages of On-device Sora, i.e., directly deploying well-trained video generative models onto resource-constrained devices without time-consuming model modification and/or re-training, are expected to become even more pronounced with the active development of compact DiT-based text-to-video generative models [46].

We implement On-device Sora on the iPhone 15 Pro [1] using Open-Sora [83]. The full implementation is available as open-source code in an anonymous GitHub1. The extensive experiments are conducted to evaluate the performance of On-device Sora using the state-of-the-art video benchmark tool, i.e., VBench [28], compared with NVIDIA A6000 GPUs. We also evaluate the proposed methods on additional text-to-video generative models, including Pyramidal Flow [29]. The experimental results demonstrate that On-device Sora maintains comparable video quality while accelerating generation speed with the proposed methods. While the iPhone 15 Pro has a GPU of 143 times less computational power [1] and 16 times smaller memory compared to the NVIDIA A6000 GPU, the evaluation results show that On-device Sora significantly improves the efficiency of video generation by effectively compensating for the limited computing resources of the device.

#### 2. Background and Challenges

We first provide a background of diffusion-based text-tovideo generation (e.g. Open-Sora [83]), which is the backbone of the proposed On-device Sora, and key challenges in realizing on-device video generation for mobile devices.

##### 2.1. Background: Diffusion-based Text-to-Video Generation

In general, diffusion-based text-to-video generation models, such as Open-Sora [83], generate videos from user prompts

(texts) through the three stages: 1) prompt (text) embedding, 2) latent video generation, and 3) video decoding.

- 1) Prompt (Text) Embedding. The first stage of diffusionbased text-to-video generation is to map a user prompt, a textual description of the desired video, to an embedding vector, which is used as input for the subsequent video generation stage. For example, to produce prompt embeddings from user texts, Open-Sora employs T5 (Text-to-Text Transformer) [48], a language model specifically fine-tuned to support video generation tasks.
- 2) Latent Video Generation. The next stage is to generate the latent video representation conditioned on the prompt embedding obtained from language models, e.g., T5 (Textto-Text Transformer) [48]. To this end, for instance, OpenSora employs STDiT (Spatial-Temporal Diffusion Transformer) [83], a diffusion-based text-to-video model using the Markov chain [77]. Since maintaining temporal consistency across video frames is essential in video, STDiT [83] applies the spatial-temporal attention mechanism [72] to the patch representations. It allows effective learning of temporal features across video frames through the temporal attention, enhanced by incorporating rope embeddings [59]. During the forward process of STDiT, the Gaussian noise

ϵk is iteratively added to the latent video representation xk over K steps, transforming the intact video representation x0 into the complete Gaussian noise xK in the latent space with the forward distribution q(xk|xk−1) as:

xk = 1 − βkxk−1 + βkϵk

q(xt|xk−1) = N(xk; 1 − βkxk−1,βkI)

(1)

where βk is the parameter determining the extent of noise.

To generate the latent representation x0, the noise ϵk is repeatedly removed (denoised) from the complete Gaussian noise xK through the reverse process using the estimated noise with the reverse distribution pθ(xk−1|xt) modeled by STDiT [83] with the parameter set θ, as follows:

xk−1 = µθ(xk,k) + βkϵ where ϵ ∼ N(0,I) pθ(xk−1|xk) = N(xk−1;µθ(xk,k),Σθ(xk,k))

(2)

where µθ(xk,k) and Σθ(xk,k) is the mean and variance of xk estimated by STDiT training, respectively. This reverse process repeatedly denoises xk into xk−1 over a large number of 1≤k≤K denoising steps (i.e., from dozens to thousands of steps), eventually generating the de-noised latent video representation x0 close to the intact representation.

- 3) Video Decoding. Finally, the latent video representa-

tion x0 generated from STDiT is decoded and up-scaled into the human-recognizable video through the VAE (Variational Autoencoder) [15]. For example, the VAE employed in Open-Sora [83] utilizes both 2D and 3D structures; the

“A serene underwater scene featuring Denoising Process a sea turtle swimming through […] ” Prompt (Text)

…

VAE Decoder Blocks

STDiT Inference

Linear Proportional Leap Compute Similarity

Concurrent Inference and Dynamic Loading

Concurrent Inference and Dynamic Loading

MLP

[Figure 1]

TemporalDimension

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Unmerge

[Figure 9]

|Thread for Load Queue<br><br>|S T S T|
|---|
|
|---|

T5 Blocks …

[Figure 10]

STDiT Blocks

TokenMerging

Cross Attn

[Figure 11]

Merge

|Thread for Inference Queue<br><br>|S T|
|---|
|
|---|

[Figure 12]

[Figure 13]

Unmerge

Self Attn

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

Memory(Optimally Loaded) S T S T

Video Frames

[Figure 14]

Merge

Prompt (Text) Embedding

Figure 1. On-device Sora enables training-free text-to-video generation directly on the device by employing three key methods: 1) Linear Proportional Leap (LPL), 2) Temporal Dimension Token Merging (TDTM), and 3) Concurrent Inference with Dynamic Loading (CI-DL).

2D VAE is based on SDXL [47], while the 3D VAE adopts the architecture of MAGVIT-v2 [76].

sion Transformer models, which progressively merges similar visual tokens within the transformer to accelerate the model inference latency by reducing the size of tokens to be processed. While token merging has been applied to diffusion models, it is only applied to spatial tokens [6, 7] and has not been applied to the temporal tokens in video diffusion models, such as STDiT [83]. Thus, a novel token merging method is required to improve the computational efficiency of token processing within well-trained video generation models, while preserving high video quality.

##### 2.2. Key Challenges in On-device Video Generation

- C1) Excessive Denoising Steps. Tab. 1 shows the latency of each model component in Open-Sora [83], where the latent video generation process (denoising process) performed by STDiT is the most time-consuming. That is because a substantial number of denoising steps is required to remove the noise ϵk from xK to obtain x0 during latent video generation [58]. Such extensive denoising iterations presents considerable challenges on mobile devices with constrained computational capabilities. While the numerous denoising steps are manageable in server-level environments—where the complete denoising process typically finishes within one minute—on mobile devices, it may take several tens of minutes. Accordingly, an effective approach to reduce denoising steps without model modification or retraining is essential to enable on-device video generation.

- Table 1. The number of executions (iterations) of each model component (i.e., T5 [48], STDiT [83], and VAE [15]) in OpenSora [83] and their total latencies on the iPhone 15 Pro [1].

|Component|Iterations Inference Time (s) Total Latency (s)|
|---|---|
|T5 [48] STDiT [83] VAE [15]<br><br>|1 110.505 110.505 50 35.366 1768.320<br><br>1 135.047 135.047|

- C2) Intensive Token Processing. While a large number of denoising steps poses a significant challenge to video generation on mobile devices, even a single denoising step itself is computationally intensive. The primary reason is that the computational complexity of the attention mechanism [43] in STDiT [83] grows quadratically with the token size, which significantly increases the computational load for token processing and, consequently, the model’s inference latency. To address this challenge, Token merging [7] has been proposed to improve the throughput of vi-

C3) High Memory Requirements. Fig. 2 shows the memory requirements of model components in Open-Sora [83], i.e., VAE [15], T5 [48], and STDiT [83], where their cumulative memory demand, i.e., 23 GB, surpasses the memory capacity of many mobile devices. For instance, the iPhone 15 Pro [1], with 8 GB of memory, restricts the available memory for a single application to 3.3 GB for system stability. Furthermore, the individual memory requirements of T5 and STDiT exceed 3.3 GB, creating challenges in loading them into memory. In addition, some memory must be reserved for model execution (inference), exacerbating memory shortages on mobile devices. Thus, limited device memory is another challenge that should be addressed to enable on-device video generation.

Model Size (GB)

| | |
|---|---|
| | |
| | |

VAE STDiT

T5 Memory Constraint (~3.3GB)

| |
|---|

| |
|---|

0 5 10

15 20

Figure 2. The size of Open-Sora models: T5 [48] (18.00 GB), STDiT [83] (4.50 GB), and VAE [15] (0.82 GB), which exceeds the available memory capacity of the iPhone 15 Pro [1] (3.3 GB).

#### 3. Overview: On-device Sora

Fig. 1 shows an overview of On-Device Sora, which takes Open-Sora [83] outlined in Sec. 2.1 as its backbone. On-

[Figure 15]

[Figure 16]

device Sora enables efficient diffusion-based text-to-video generation on mobile devices by addressing three key challenges (Sec. 2.2) through three proposed methods: 1) Linear Proportional Leap (LPL) (Sec. 4), 2) Temporal Dimension Token Merging (TDTM) (Sec. 5), and 3) Concurrent Inference with Dynamic Loading (CI-DL) (App. A.1). These methods can also be applied to other diffusion-based text-to-video generation models, e.g., Pyramidal Flow [29].

- (a)
- (b)

𝔃

𝔃

Prompt : “A beautiful waterfall”

[Figure 17]

[Figure 18]

𝔃

𝔃

[Figure 19]

[Figure 20]

𝒗 𝑃 , 𝑡 𝑡 (c)

𝔃

𝔃 𝔃 ≈ 𝔃

#### 4. Linear Proportional Leap

Regular Denoising Step

Linear Proportional Leap

On-device Sora reduces the excessive number of denoising steps performed by STDiT [83] by introducing Linear Proportional Leap (LPL), which leverages the trajectory properties of Rectified Flow [38]. It allows early stop of denoising steps through proportionally scaled linear leaps, without extra model training or modification of STDiT architecture.

Figure 3. An abstracted illustration of trajectories and latent visualizations for K = 30 and n = 15: (a) Rectified Flow [38] with full k = 30 denoising steps, generating intact and complete video data, (b) Rectified Flow [38] with n + 1 = 16 denoising steps without applying Linear Proportional Leap, resulting in lowquality video data generation from variance with high step sizes (dtk), and (c) Linear Proportional Leap with n + 1 = 15 + 1 denoising steps, producing video data nearly equivalent to (a).

##### 4.1. Rectified Flow

The reverse diffusion process is performed through multiple denoising steps, transforming an initial Gaussian distribution into a desired distribution corresponding to the input prompt. Several ODE-based methods [40, 82] reformulate this process by training models to predict the drift at each time point, building the distribution trajectories from the initial to the target point by using ODE solvers [52, 73].

##### 4.2. Linear Proportional Leap

Based on Rectified Flow [38], the proposed Linear Proportional Leap reduces denoising steps, as illustrated in Fig. 3.

If the nth data distribution is sufficiently close to the Kth target distribution in the denoising process, the trajectories zn+1...K would be approximately straight for the remaining time steps tn+1...K, making the drift estimation v(Pk,tk)dtk in Eq. (3) unnecessary for k>n. Consequently, v(Pk,tk)dtk is estimated only for 1≤k≤n, allowing the denoising process to stop early at the (n + 1)th step, not to the full Kth step. For the remaining steps tn+1...K, the trajectories zn+1...K linearly leap towards the target data distribution, with the straight direction of v(Pn+1,tn+1) and dtn+1 scaled proportionally to tn+1...K.

Rectified Flow [38] simplifies the transition from the initial point to the target point by training the model to predict a drift aligned with the direct linear trajectory connecting these two points. Using the Euler method [10], the kth trajectory zk is derived by updating the previous trajectory zk−1 with the estimated drift v(Pk,tk) and step size dtk defined by two sampled time steps tk and tk+1, as follows:

By assuming k = K, zk is derived from Eq. (3) as: zk = zk−1 + v(Pk,tk)dtk

zk = zk−1 + v(Pk,tk)dtk ∀1 ≤ k ≤ K

(3)

tk−tk+1, if tk ̸= tK tk, if tk = tK

where tk ∈ [0,1] and dtk =

k−1

(4)

v(Pi,ti)(ti − ti+1) + v(Pk,tk)tk

= z0 +

i=1

Here, the time step tk∈[0,1] corresponds to the normalized reverse process at the kth denoising step, with tk=1 representing the time step at which data is fully noisy (start of denoising) and tk=0 corresponding to the time step when data reaches the desired distribution (end of denoising). The drift v(Pk,tk) is predicted from STDiT [83] given the kth position on the trajectory, Pk = tkP1 + (1 − tk)P0, computed from linear interpolation with sampled time step, tk [38].

If the denoising process stops at the step n, the nth trajectory is represented as zn = z0+ ni=1 v(Pi,ti)(ti − ti+1). Then, if we apply the identical drift v(Pn+1,tn+1)dtn+1 to the remaining n + 1 ≤ i ≤ k steps, Eq. (4) becomes:

k−1

(ti−ti+1)+v(Pn+1,tn+1)tk

zk=zn+v(Pn+1,tn+1)

i=n+1

In Rectified Flow [38], the model is trained to predict the direct direction toward target point at any point on the trajectory. This allows diffusion-based generation models for achieving a denoising process in few steps without significant performance degradation. With Rectified Flow [38], Open-Sora [83] generates video with K = 30 or 50 steps.

=zn+v(Pn+1,tn+1)(tn+1−tn+2+···+tk−1−tk+tk)

=zn+v(Pn+1,tn+1)tn+1

(5)

Thus, the required denoising steps are reduced to n + 1 out of the total k steps, allowing STDiT to be executed only

1.00

Figure 4. An example of cosine similarities between two adjacent drifts estimated from STDiT [83], i.e., v(Pn, tn) and v(Pn−1, tn−1) for 30 (red) and 50 steps (blue).

STDiT3

CosineSimilarity

0.95

MLP

MLP

TemporalBlock

0.90

SpatialBlock

Unmerge

Unmerge

0.85

30 steps 50 steps

Cross Attention

Cross Attention

0.80

Merge Unmerge

Merge Unmerge

0 10 20 30 40 50 Denoising steps

Self Attention

Self Attention

Merge

Merge

|Merge (BS) x 2 x C|
|---|

|Unmerge (BS) x 4 x C|
|---|

n + 1 times, with the last (n + 1)th trajectory applied to its time step tn+1, which enables Linear Proportional Leap. Replacing dtk with tn+1 in Eq. (5), instead of computing difference between the sampled time steps, can invoke an identical effect under assumption that later steps tend to sustain drift directions. It enables the immediate completion of the denoising process, as tn+1 is equivalent to the remaining denoising steps required to reach the end of denoising.

Figure 5. In attention layers of STDiT [83], two consecutive tokens are merged along the temporal dimension and subsequently unmerged after processing, reducing the token size by half and the computational complexity up to a quarter.

In STDiT [83], the attention bias influenced by input tokens in cross-attention layers leads to higher computational demands than in self-attention layers. As a result, developing an effective token merging method for cross-attention is crucial in video generation. However, existing token merging [6, 7, 34] applied to the cross-attention have shown suboptimal performance, and applying them to the selfattention in STDiT has observed video quality drops [6, 7].

Linear Proportional Leap can be dynamically applied by measuring the cosine similarity between two consecutive drifts v at runtime. When the cosine similarity appears that the current trajectory is sufficiently linear, a linear leap is made proportionally to the remaining steps. Fig. 4 visualizes the cosine similarities between consecutive drifts, stabilizing after a certain number of steps, suggesting that the trajectory toward the target data distribution is nearly linear. This enables fewer steps by utilizing the leap with the larger step size to efficiently progress to the desired direction.

##### 5.2. Temporal Dimension Token Merging

Fig. 5 illustrates Temporal Dimension Token Merging. Based on the hypothesis that successive video frames exhibit similar values, two consecutive frames are merged over the temporal dimension by averaging, creating a single token without the overhead of calculating frame similarity. This reduces the size of tokens by half while preserving the essential temporal information. Consequently, it decreases the computation of self-attention by a factor of four, according to the self-attention’s quadratic complexity, O(n2). Similarly, it reduces the computation of cross-attention by half, based on its linear complexity, O(nm). Then, the output token processed through attentions replicates the dimensions for each frame, restoring them to their original size.

#### 5. Temporal Dimension Token Merging

On-device Sora reduces the computational complexity of the denoising process by introducing Temporal Dimension Token Merging (TDTM), which halves the size of tokens in STDiT [83] over the temporal dimension, decreasing the computation of self-attention quadratically and crossattention in half. Unlike existing token merging applying to self-attentions over the spatial dimension, which exhibits suboptimal performance [6, 7, 17, 34], Temporal Dimension Token Merging leverages the temporal aspect of video frames to reduce computation while ensuring video quality.

Given the token Tin as an input for a self- or crossattention with the dimension [B, ST, C], where B denotes the batch size, S is the number of pixel patches, T is the number of frames, and C is the channel dimension, the input token Tin is merged into Tmerged, with the index i, as:

##### 5.1. Token Merging

STDiT [83] consists of multiple attention layers, i.e., crossand self-attention, of the linear and quadratic complexity, respectively. In video generation, these attentions extend across two dimensions, i.e., spatial and temporal dimension. General model optimization techniques, e.g., pruning [49], quantization [20], and distillation [22], may reduce STDiT’s attention computation. However, they necessitate model training (fine-tuning) or specialized hardware for implementation, and more importantly, the performance of video generation can hardly be preserved. In contrast, token merging [7] reduces the token size processed in attentions, decreasing computational complexity without requiring model re-training or hardware-specific adaptations.

Tmerged = TDTMmerge(Tin) (6) TDTMmerge(T)[i] =

- 1

- 2

(T[:,i,:] + T[:,i + 1,:]) (7)

From this, two adjacent tokens are merged along the temporal dimension, producing the merged token Tmerged of [B, ST/2, C], which is processed by self- or cross-attention.

After being processed by each attention, Tmerged is unmerged into Tunmerged of the dimension [B, ST, C] as:

Tunmerged = TDTMunmerge(Attention(Tmerged)) (8)

TDTMunmerge(T)[2i] = T[:,i,:] (9) where Attention(·) is either the self- or cross-attention.

Temporal Dimension Token Merging can be selectively applied during the denoising process to minimize potential negative impacts on video quality that may arise from processing merged tokens. Specifically, out of a total of K denoising steps, tokens can be merged only for the initial k steps, while the tokens for the remaining K−k steps remain unmerged. This is based on the observation that, when tokens are merged along the temporal dimension, the noise values vary slightly across frames—a phenomenon not observed in image diffusion [53] that does not involve a temporal dimension in the generation process. However, because noise values in the early denoising steps are less critical, it is expected that applying token merging exclusively for initial steps does not substantially drop video quality.

#### 6. Implementation

We implement the proposed On-device Sora on iPhone 15 Pro [1], leveraging its GPU of 2.15 TFLOPS and 3.3 GB of available memory, with the two methods proposed in Sec. 4 and 5. In addition, to execute large video generative models (i.e., T5 [48] and STDiT [83]) with the limited device memory, we devise and implement Concurrent Inference with Dynamic Loading (CI-DL), which partitions the models into smaller blocks that can be loaded into the memory and executed concurrently. The details of CI-DL is described in App. A.1. The model components—T5 [48], STDiT [83], and VAE [15]—in PyTorch [45] are converted to MLPackage, an Apple’s CoreML framework [54] for machine learning apps. Since current version of CoreML [1] lacks support for certain diffusion-related operations in text-to-video generation, we develop custom solutions like xFormer [33] and cache-based acceleration. We implement denoising scheduling, sampling pipeline, and tensor-to-video conversion in Swift [2] using Apple-provided libraries. To optimize models, T5 [48], the largest in video generation, is quantized to int8, while others models (STDiT [83] and VAE [15]) run in float32; we found that they are challenging to quantize due to sensitivity and performance degradation.

#### 7. Experiment

We evaluate the performance of On-device Sora on 68frame videos at 256×256 resolution, using 800 text prompts sampled from VBench [28] and 1,000 each from Pandas70M [11], and VidGen [61]. To assess both temporal and frame-level video quality, we utilize VBench [28], the state-of-the-art benchmark for text-to-video generation, which provides comprehensive metrics, including subject

consistency, background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality, and imaging quality. Additional experiments on another textto-video generation model, i.e., Pyramidal Flow [29], are provided in Fig. 15 and Tab. 7 in App. B.

##### 7.1. Video Generation Performance

We evaluate the quality of videos generated by On-device Sora on the iPhone 15 Pro [1], in comparison to videos produced by Open-Sora [83] running on NVIDIA A6000 GPUs. Tab. 2 summarizes the averaged results that compare those videos evaluated with VBench [28], including categories such as animals, humans and lifestyle. Tab. 6 and Fig. 11 in App. B provide detailed full per-category results. The results demonstrate that On-device Sora generates videos with quality nearly equivalent to Open-Sora in most metrics, exhibiting only a slight drop in frame-level quality, averaging 0.03, while achieving an average 0.06 improvement in dynamic degree.

Fig. 6 shows example videos, compared with OpenSora [83]. For the prompt “a stack of dried leaves burning in a forest”, both On-device Sora and Open-Sora generate visually plausible videos, illustrating stack of burning dry leaves and forest in the background. Similarly, for “closeup of a lemur”, both produce descriptive videos: On-device Sora shows a lemur turning its head, while Open-Sora delivers a less dynamic yet visually comparable depiction.

##### 7.2. Linear Proportional Leap

Tab. 3 presents the video generation performance and speedup of On-device Sora when applying Linear Proportional Leap (LPL) proposed in Sec. 4. In the table, ‘LPL Setting’ indicates the number of denoising steps used for video generation out of a total of 30 steps, while the remaining steps are omitted by LPL. We also evaluate a dynamic version of LPL, referred to as ‘Dynamic’ in Tab. 3, which determines the number of denoising steps at runtime based on the cosine similarities between two adjacent drifts estimated using STDiT [83]. The dynamic LPL halts denoising when a predefined number of cosine similarity measurements fail to improve beyond a 10−4 tolerance, after a designated minimum number of denoising steps (50%).

Overall, LPL enables video generation with quality comparable to Open-Sora [83], without noticeable visual degradation (e.g., 0.743 vs. 0.736 in average of VBench [28]), while accelerating video generation up to 1.94× without any model optimization or re-training. LPL reduces denoising latency linearly without extra computation, enabling efficient video generation while retaining robust performance.

Fig. 7 presents example videos generated using LPL, where all LPL settings consistently produce semantically identical target videos, with most video quality remaining stable across various numbers of denoising steps.

Open-Sora

Open-Sora

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

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

On-device Sora

On-device Sora

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

“a stack of dried leaves burning in a forest”

“close up of a lemur”

Figure 6. Example videos generated by On-device Sora and Open-Sora [83] (68 frames, 256×256).

- Table 2. The VBench [28] evaluation summary: On-device Sora vs. Open-Sora [83] (68 frames, 256×256), full results in Tab. 6 in App. B.

|Method<br><br>|Temporal Quality↑|Frame-Wise Quality↑|
|---|---|---|
| |Subject Consistency<br><br>Background Consistency<br><br>Temporal Flickering<br><br>Motion Smoothness<br><br>Dynamic Degree|Aesthetic Quality<br><br>Imaging Quality|
|Open-Sora On-device Sora<br><br>|0.97 0.97 0.99 0.99 0.21 0.96 0.97 0.99 0.99 0.27<br><br>|0.50 0.56 0.47 0.53|

Table 3. The video quality and generation speedup under different settings of LPL (Linear Proportional Leap).

|Dataset|LPL Setting<br><br>|Temporal Quality↑|Frame-Wise Quality↑<br><br>|Speedup↑|
|---|---|---|---|---|
| | |Subject Consistency<br><br>Background Consistency<br><br>Temporal Flickering<br><br>Motion Smoothness<br><br>Dynamic Degree|Aesthetic Quality<br><br>Imaging Quality| |
|VBench|Dynamic (µ:17.73/30)|0.97 0.97 0.99 0.99 0.20<br><br>|0.50 0.56<br><br>|1.53×|
| |16/30 (53%) 30/30 (100%)<br><br>|0.97 0.97 0.99 0.99 0.18 0.97 0.97 0.99 0.99 0.21|0.50 0.55 0.50 0.57<br><br>|1.94× 1.00×|

|Pandas70M<br><br>|Dynamic (µ:17.66/30)|0.97 0.97 0.99 0.99 0.22|0.47 0.60<br><br>|1.92×|
|---|---|---|---|---|
| |16/30 (53%) 30/30 (100%)|0.97 0.97 0.99 0.99 0.19 0.97 0.97 0.99 0.99 0.22<br><br>|0.47 0.60 0.45 0.58<br><br>|1.62× 1.00×<br><br>|

|VidGen<br><br>|Dynamic (µ:18.08/30)|0.95 0.96 0.98 0.99 0.36|0.47 0.57<br><br>|1.94×|
|---|---|---|---|---|
| |16/30 (53%) 30/30 (100%)|0.95 0.96 0.98 0.98 0.33<br><br>0.96 0.97 0.99 0.99 0.39<br>|0.46 0.56 0.48 0.57<br><br>|1.53× 1.00×|

Table 4. The video quality and speedup under different merging steps of TDTM (Temporal Dimension Token Merging).

|Dataset|Merging Steps|Temporal Quality↑<br><br>|Frame-Wise Quality↑<br><br>|Speedup↑|
|---|---|---|---|---|
| | |Subject Consistency<br><br>Background Consistency<br><br>Temporal Flickering<br><br>Motion Smoothness<br><br>Dynamic Degree|Aesthetic Quality<br><br>Imaging Quality| |
|VBench<br><br>|30/30 (100%) 15/30 (50%) 0/30 (0%)<br><br>|0.97 0.97 0.99 0.99 0.06 0.97 0.97 0.99 0.99 0.12 0.96 0.97 0.99 0.99 0.23<br><br>|0.50 0.56<br>0.50 0.57<br>0.50 0.58<br>|1.27× 1.13× 1.00×<br><br>|

30/30 (100%) 0.98 0.97 0.99 0.99 0.09 0.48 0.59 1.23× 15/30 (50%) 0.98 0.97 0.99 0.99 0.13 0.47 0.58 1.11× 0/30 (0%) 0.97 0.97 0.99 0.99 0.22 0.45 0.58 1.00×

Pandas70M

30/30 (100%) 0.97 0.97 0.99 0.99 0.14 0.50 0.59 1.70× 15/30 (50%) 0.97 0.98 0.99 0.99 0.16 0.47 0.58 1.36× 0/30 (0%) 0.96 0.97 0.99 0.99 0.39 0.48 0.57 1.00×

VidGen

##### 7.3. Temporal Dimension Token Merging

- Tab. 4 presents the video quality and video generation speedup achieved when varying numbers of denoising steps

to which Temporal Dimension Token Merging (TDTM) (Sec. 5) is applied, indicated as ‘Merging Steps’ in the table, out of a total of 30 denoising steps. The results in-

“close up of wolf”

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

30/30 (100%) 23/30 (76%) 16/30 (53%) Dynamic

“a sleeping orangutan”

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

30/30 (100%) 23/30 (76%) 16/30 (53%) Dynamic

Figure 7. The snapshots of videos (68 frames, 256×256 resolution) applied with various LPL settings (Tab. 3).

dicate that increasing the number of merging steps consistently accelerates video generation, ranging from 1.11× to 1.70× speedups, while maintaining stable quality metrics; the average scores for VBench remain at 0.736. Nonetheless, some declines in the dynamic degree metric are observed, revealing a trade-off between maintaining visual dynamics and reducing token processing complexity. This indicates the importance of striking a balance between video dynamics and speedup. We found that selectively applying TDTM to specific denoising steps can effectively reduce visual noises. For instance, limiting TDTM to the first 15 denoising steps while not applying it to the rest of the denoising steps tends to result in a less severe quality drop compared to applying TDTM to all steps, which can mitigate issues like flickering or dynamic degree to provide improved video quality.

Fig. 8 shows examples of video frames generated with varying numbers of denoising steps to which TDTM is applied, out of 30 steps. The results show that even as TDTM is applied to an increasing number of denoising steps, the quality of the video frames seems to remain consistent.

##### 7.4. Video Generation Latency

- Tab. 5 shows video generation latencies of two resolutions when each of the proposed methodologies—LPL, TDTM, and CI-DL—is applied individually, as well as the latency when all of them are applied together (‘All’). Latencies are measured with LPL activated at the 15th denoising step and TDTM applied throughout all steps, reported as the mean of three independent experiments. ‘STDiT’ and ‘Total’ specify whether the latency is measured solely for STDiT [83] or for end-to-end video generation, including T5 [48] and VAE [15]. The results demonstrate substantial latency reductions for each methodology compared to the case without using the proposed methods, i.e., 293.51 vs. 1768.32 seconds (Tab. 1) for STDiT. For the 192×192 resolution video, STDiT (denoising process) takes less than five min-

“grilling a meat on a charcoal grill”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

0/30 (0%) 10/30 (33%) 15/30 (50%)

20/30 (66%)

“scenery of a relaxing beach”

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

0/30 (0%) 10/30 (33%) 15/30 (50%) 20/30 (66%)

Figure 8. The snapshots of videos (68 frames, 256×256 resolution) applied with various merging steps of TDTM (Tab. 4).

utes when all three methodologies are applied. Additionally, it indicates that the methodologies do not interfere with each other, instead work synergistically to enhance latency.

Table 5. Ablation study on video generation latency (s). ‘All’ denotes the combined application of LPL, TDTM, and CI-DL.

|Resolution<br><br>|Measurement|LPL TDTM CI-DL All|
|---|---|---|
|192×192<br><br>|STDiT Total|390.72 696.03 566.81 293.51 514.24 823.03 691.25 416.50<br><br>|
|256×256<br><br>|STDiT Total|573.88 965.40 947.67 454.48 754.08 1148.63 1127.88 638.09<br><br>|

#### 8. Related Work

On-device Video Generation. On-device video generation has recently gained attention due to the growing demand for real-time, efficient, and privacy-preserving content creation, and recently begun to be studied. MobileVD [71], based on UNet [51] in Stable Video Diffusion [4], reduces computational costs by employing pruning, including low-resolution fine-tuning, temporal multi-scaling, and optimizations in channel and time blocks. SnapGen-V [70] utilizes Stable Diffusion v1.5 [50] for processing image information while efficiently handling temporal information through Conv3D and attention-based layers. Furthermore, with fine-tuning, the denoising step is reduced. However, these approaches require substantial GPU resources for model optimization, e.g., MobileVD gone 100K training iterations on four A100 GPUs, and SnapGen-V utilized more than 150K iterations on 256 NVIDIA A100 GPUs. In contrast, On-Device Sora instantly applies without additional model training or optimization, removing the requirement for GPU resources.

Rectified Flow. While Open-Sora [83] reduces the number of denoising steps by leveraging Rectified Flow [38], most related approaches [16, 84] require conditioned model training and/or distillation [84]. In contrast, the proposed

Linear Proportional Leap effectively reduces the denoising steps without a significant performance drop, as validated using VBench [28], without requiring model re-training or distillation. Notably, it can be easily activated at runtime by just calculating the cosine similarities of drifts between consecutive denoising steps.

Token Merging. Most token merging methods [6, 7, 17] primarily focus on image generation, where tokens are merged based on the spatial similarity rather than temporal similarity. Although some temporal token merging have been proposed, they are applied to models in other domains [9, 18, 34], not in video generation. As such, Temporal Dimension Token Merging is the first to apply token merging based on the successive similarities between frames in video generation. Additionally, while previous works apply token merging to self-attention due to performance degradation [6, 7, 17, 34], On-device Sora shows that token merging can be applied to cross-attention with minimal performance loss, achieving 50% merging ratio.

#### 9. Conclusion

We propose On-device Sora, the first training-free solution for generating videos on mobile devices using diffusionbased models. It addresses key challenges in video generation to enable efficient on-device operation. The issue of extensive denoising steps is addressed through Linear Professional Leap, the challenge of handling large tokens is mitigated with Temporal Dimension Token Merging, and memory limitations are overcome through Concurrent Inference with Dynamic Loading. These proposed efficient on-device video generation methodologies are not limited to On-device Sora but are broadly applicable to various applications, providing advancements in on-device video generation without additional model re-training.

#### References

- [1] Apple. iphone 15 pro—technical specifications, 2023. [Online]. Available: https://support.apple.com/enus/111829. 2, 3, 6, 14, 16
- [2] Apple. Swift, 2024. https://developer.apple. com/swift/. 6
- [3] BN Biswas, Somnath Chatterjee, SP Mukherjee, and Subhradeep Pal. A discussion on euler method: A review. Electronic Journal of Mathematical Analysis and Applications, 1(2):2090–2792, 2013. 2
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 8
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with la-

- tent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 1
- [6] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022. 2, 3, 5, 9
- [7] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In International Conference on Learning Representations, 2023. 2, 3, 5, 9
- [8] Thibault Castells, Hyoung-Kyu Song, Tairen Piao, Shinkook Choi, Bo-Kyeong Kim, Hanyoung Yim, Changgwun Lee, Jae Gon Kim, and Tae-Ho Kim. Edgefusion: On-device text-to-image generation. arXiv preprint arXiv:2404.11925,

2024. 1

- [9] Jialin Chen and Rex Ying. Tempme: Towards the explainability of temporal graph neural networks via motif discovery. Advances in Neural Information Processing Systems, 36,

2024. 9

- [10] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018. 4
- [11] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. arXiv preprint arXiv:2402.19479, 2024. 6
- [12] Yu-Hui Chen, Raman Sarokin, Juhyun Lee, Jiuqiang Tang, Chuo-Ling Chang, Andrei Kulik, and Matthias Grundmann. Speed is all you need: On-device acceleration of large diffusion models via gpu-aware optimizations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4651–4655, 2023. 1
- [13] Jiwoong Choi, Minkyu Kim, Daehyun Ahn, Taesu Kim, Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Jae-Joon Kim, and Hyungjun Kim. Squeezing large-scale diffusion models for mobile. arXiv preprint arXiv:2307.01193, 2023. 1
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 1
- [15] Carl Doersch. Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908, 2016. 2, 3, 6, 8, 16
- [16] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 8
- [17] Zhanzhou Feng and Shiliang Zhang. Efficient vision transformer via token merger. IEEE Transactions on Image Processing, 2023. 5, 9
- [18] Leon G¨otz, Marcel Kollovieh, Stephan G¨unnemann, and Leo Schwinn. Efficient time series processing for transformers and state-space models through token merging. arXiv preprint arXiv:2405.17951, 2024. 9

- [19] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129(6):1789–1819, 2021. 16
- [20] Robert M. Gray and David L. Neuhoff. Quantization. IEEE transactions on information theory, 44(6):2325–2383, 1998. 5
- [21] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. Advances in Neural Information Processing Systems, 35:27953–27965, 2022. 1
- [22] Geoffrey Hinton. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 5
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [24] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1
- [25] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022. 1
- [26] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 1
- [27] Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, and Eric P Xing. Toward controlled generation of text. In International conference on machine learning, pages 1587–1596. PMLR, 2017. 1
- [28] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 2, 6, 7, 9, 14
- [29] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 1, 2, 4, 6, 15, 16

- [30] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 1
- [31] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023. 1
- [32] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 15

- [33] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/ facebookresearch/xformers, 2022. 6
- [34] Xirui Li, Chao Ma, Xiaokang Yang, and Ming-Hsuan Yang. Vidtome: Video token merging for zero-shot video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7486–7495, 2024. 5, 9
- [35] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024. 1
- [36] Shanchuan Lin and Xiao Yang. Animatediff-lightning: Cross-model diffusion distillation. arXiv preprint arXiv:2403.12706, 2024. 1
- [37] Ming-Yu Liu, Xun Huang, Jiahui Yu, Ting-Chun Wang, and Arun Mallya. Generative adversarial networks for image and video synthesis: Algorithms and applications. Proceedings of the IEEE, 109(5):839–862, 2021. 1
- [38] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4, 8, 15, 17
- [39] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 13
- [40] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 4
- [41] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7038– 7048, 2024. 1
- [42] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 1
- [43] Zhaoyang Niu, Guoqiang Zhong, and Hui Yu. A review on the attention mechanism of deep learning. Neurocomputing, 452:48–62, 2021. 3
- [44] Xavier Orriols. Generative models for video analysis and 3D range data applications. Universitat Aut`onoma de Barcelona,, 2004. 1
- [45] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im-

- perative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 6
- [46] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 1, 2

- [47] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 3
- [48] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 2, 3, 6, 8, 13, 15, 16
- [49] Russell Reed. Pruning algorithms-a survey. IEEE transactions on Neural Networks, 4(5):740–747, 1993. 5, 16
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 8
- [51] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 8
- [52] Carl Runge. Uber¨ die numerische aufl¨osung von differentialgleichungen. Mathematische Annalen, 46(2):167–178, 1895. 4
- [53] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10,

2022. 1, 6

- [54] Ozg¨¨ ur Sahin and Ozg¨¨ ur Sahin. Introduction to apple ml tools. Develop Intelligent iOS Apps with Swift: Understand Texts, Classify Sentiments, and Autodetect Answers in Text Using NLP, pages 17–39, 2021. 6
- [55] Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli. Singan: Learning a generative model from a single natural image. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4570–4580, 2019. 1
- [56] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 1

- [57] Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video editing via factorized diffusion distillation. In European Conference on Computer Vision, pages 450–466. Springer, 2024. 1

- [58] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1, 3
- [59] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 2

- [60] Masahiro Suzuki and Yutaka Matsuo. A survey of multimodal deep generative models. Advanced Robotics, 36(5-6): 261–278, 2022. 1
- [61] Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, and Hao Li. Vidgen-1m: A large-scale dataset for text-to-video generation. arXiv preprint arXiv:2408.02629, 2024. 6
- [62] Shilong Tian, Hong Chen, Chengtao Lv, Yu Liu, Jinyang Guo, Xianglong Liu, Shengxi Li, Hao Yang, and Tao Xie. Qvd: Post-training quantization for video diffusion models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10572–10581, 2024. 1
- [63] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 1
- [64] Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Mobileone: An improved one millisecond mobile backbone. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7907–7917, 2023. 1
- [65] Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. Advances in neural information processing systems, 35:23371–23385, 2022. 1
- [66] Fu-Yun Wang, Zhaoyang Huang, Weikang Bian, Xiaoyu Shi, Keqiang Sun, Guanglu Song, Yu Liu, and Hongsheng Li. Animatelcm: Computation-efficient personalized style video generation without personalized video data. In SIGGRAPH Asia 2024 Technical Communications, New York, NY, USA,

2024. Association for Computing Machinery. 1

- [67] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, et al. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pages 38–45,

2020. 13

- [68] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 1
- [69] Yiming Wu, Huan Wang, Zhenghao Chen, and Dong Xu. Individual content and motion dynamics preserved pruning for video diffusion models. arXiv preprint arXiv:2411.18375,

2024. 1

- [70] Yushu Wu, Zhixing Zhang, Yanyu Li, Yanwu Xu, Anil Kag, Yang Sui, Huseyin Coskun, Ke Ma, Aleksei Lebedev, Ju Hu, et al. Snapgen-v: Generating a five-second

- video within five seconds on a mobile device. arXiv preprint arXiv:2412.10494, 2024. 1, 8
- [71] Haitam Ben Yahia, Denis Korzhenkov, Ioannis Lelekas, Amir Ghodrati, and Amirhossein Habibian. Mobile video diffusion. arXiv preprint arXiv:2412.07583, 2024. 1, 8
- [72] Chenggang Yan, Yunbin Tu, Xingzheng Wang, Yongbing Zhang, Xinhong Hao, Yongdong Zhang, and Qionghai Dai. Stat: Spatial-temporal attention mechanism for video captioning. IEEE transactions on multimedia, 22(1):229–241,

2019. 2

- [73] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and MingHsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4): 1–39, 2023. 4
- [74] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 17
- [75] Zilyu Ye, Zhiyang Chen, Tiancheng Li, Zemin Huang, Weijian Luo, and Guo-Jun Qi. Schedule on the fly: Diffusion time prediction for faster and better image generation. arXiv preprint arXiv:2412.01243, 2024. 17
- [76] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 3
- [77] Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon. Text-to-image diffusion models in generative ai: A survey. arXiv preprint arXiv:2303.07909, 2023. 2
- [78] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 1
- [79] Xulu Zhang, Xiao-Yong Wei, Wengyu Zhang, Jinlin Wu, Zhaoxiang Zhang, Zhen Lei, and Qing Li. A survey on personalized content synthesis with diffusion models. arXiv preprint arXiv:2405.05538, 2024. 1
- [80] Tianchen Zhao, Tongcheng Fang, Enshu Liu, Rui Wan, Widyadewi Soedarmadji, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, Huazhong Yang, et al. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. arXiv preprint arXiv:2406.02540,

2024. 1

- [81] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023. 1
- [82] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpmsolver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36:55502–55542, 2023. 4, 17
- [83] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16

[84] Yuanzhi Zhu, Xingchao Liu, and Qiang Liu. Slimflow: Training smaller one-step diffusion models with rectified flow. In European Conference on Computer Vision, pages 342–359. Springer, 2025. 8

#### A. Appendix - Implementation

Given the number of model blocks b, block loading latency l, and inference latency of block e, the latency reduction r achieved through Concurrent Inference is given by:

##### A.1. Concurrent Inference with Dynamic Loading

On-device Sora tackles the challenge of limited device memory in text-to-video generation, which restricts the ondevice inference of large diffusion-based models, by introducing Concurrent Inference with Dynamic Loading (CIDL), which partitions models and executes them in a concurrent and dynamic manner.

r = b · min(l,e) − α (10) where α is the overhead caused by the block loading.

Given the large number of denoising steps performed by STDiT, which is partitioned into multiple blocks for execution, similar to T5 [48], the number of blocks b is expected to be large, leading to a significant reduction in latency. It is expected to accelerate the overall model inference effectively regardless of the device’s memory capacity. When the available memory is limited, then b increases, while with larger memory, both l and e increase. In either case, it can result in an almost constant latency reduction r in Eq. (10). Dynamic Loading To further enhance the model inference latency, we propose Dynamic Loading, which is applied in conjunction with Concurrent Inference. It maintains a subset of model blocks in memory without unloading them, with the selection for the subset of blocks to be retained in memory dynamically determined based on the device’s available memory at runtime.

Concurrent Inference The model components of Ondevice Sora [39], i.e., STDiT [83] and T5 [48], easily exceed the available memory of many mobile devices, e.g., 3.3 GB RAM of iPhone 15 Pro, as shown in Fig. 2. Given that the Transformer architecture [67], which is the backbone for both T5 [48] and STDiT [83], we partition these models into smaller blocks (segments) and load them into memory accordingly for model inference.

To execute video model inference using the model partitioning, each block must be loaded onto memory sequentially before execution, increasing the overall latency of video generation by incurring block loading time. Fig. 9(a) shows the sequential block load and inference cycle of STDiT [83], where GPU remains idle intermittently, waiting for each block to be loaded into memory, and only begins execution after the loading is complete. This sequential loading and execution process significantly increases the overall latency of model inference.

The available memory of the device can vary at runtime based on the system status and configurations of applications running on the mobile device. By retaining a subset of model blocks in memory, the overhead of reloading these blocks during subsequent steps of Concurrent Inference can be eliminated, enabling reductions in model inference latency. To achieve this, we measure the device’s run-time memory capacity and the memory required for inferring a single model block during the initial step of Concurrent Inference. Next, the memory allocated for retaining certain model blocks is dynamically determined as the difference between the available memory and the memory required for inferring a model block. Then, a series of model blocks that fit within this allocated memory is loaded in a retained state.

[Figure 61]

Figure 9. The block loading and inference cycles for (a) sequential loading and inference, and (b) concurrent inference.

Fig. 10 depicts Dynamic Loading; the first four model blocks are loaded in a retrained state. Unlike other blocks, e.g., the 5th block, these blocks are not unloaded to memory after the initial step, reducing block loading overhead.

To minimize the increase in model inference latency caused by sequential block loading and execution, we propose Concurrent Inference, which leverages both the CPU and GPU for parallel block loading and execution; CPU loads the (i + 1)th block, while GPU concurrently executes the ith block. Initially, the first and second blocks are loaded into memory concurrently, with the first block completing its loading first. Subsequently, the inference of the first block and the loading of the second block occur simultaneously. This process continues such that the inference of the ith block and the loading of the (i + 1)th block overlap, ensuring continuous parallelism until the final block. Fig. 9-(b) depicts the load and inference cycle of STDiT with Concurrent Inference, which shows that GPU is active without almost no idle time by performing block loading and inference in parallel.

Loaded on Memory

1 st block 2 nd block 3 rd block 4 th block 5 th block

Load Inference (GPU)

Figure 10. The block loading and inference cycle for Dynamic Loading applied with Concurrent Inference.

Applying Dynamic Loading, the latency reduction r in Eq. (10) for Concurrent Inference is updated as:

r = b · min(l,e) + d · max(0,l − e) − α · (1 − d/b) (11)

where d is the number of blocks maintained in memory. As the number of model blocks retained in memory increases

Table 6. The VBench [28] evaluation by category: On-device Sora vs. Open-Sora [83] (68 frames, 256×256 resolution).

|Category|Method|Temporal Quality↑<br><br>|Frame-Wise Quality↑|
|---|---|---|---|
| | |Subject Consistency<br><br>Background Consistency<br><br>Temporal Flickering<br><br>Motion Smoothness<br><br>Dynamic Degree<br><br>|Aesthetic Quality<br><br>Imaging Quality|
|Animal|Open-Sora<br><br>On-device Sora<br><br>|0.97 0.98 0.99 0.99 0.15 0.95 0.97 0.99 0.99 0.28<br><br>|0.51 0.56 0.48 0.55|
|Architecture|Open-Sora<br><br>On-device Sora<br><br>|0.99 0.98 0.99 0.99 0.05 0.98 0.98 0.99 0.99 0.12<br><br>|0.53 0.60 0.49 0.56|
|Food|Open-Sora<br><br>On-device Sora<br><br>|0.97 0.97 0.99 0.99 0.26 0.95 0.97 0.99 0.99 0.38|0.52 0.60 0.48 0.53<br><br>|
|Human|Open-Sora<br><br>On-device Sora<br><br>|0.96 0.97 0.99 0.99 0.38 0.96 0.96 0.99 0.99 0.43|0.48 0.57 0.48 0.55<br><br>|
|Lifestyle<br><br>|Open-Sora On-device Sora|0.97 0.97 0.99 0.99 0.23 0.96 0.97 0.99 0.99 0.25<br><br>|0.45 0.56 0.45 0.53|
|Plant|Open-Sora<br><br>On-device Sora<br><br>|0.98 0.98 0.99 0.99 0.15 0.97 0.98 0.99 0.99 0.16|0.50 0.58 0.46 0.55<br><br>|
|Scenery|Open-Sora<br><br>On-device Sora|0.98 0.98 0.99 0.99 0.10 0.97 0.98 0.99 0.99 0.17<br><br>|0.50 0.50 0.48 0.47|
|Vehicles|Open-Sora<br><br>On-device Sora<br><br>|0.97 0.97 0.99 0.99 0.37 0.94 0.96 0.98 0.99 0.44<br><br>|0.48 0.54 0.47 0.49|

###### Food

###### Human

###### Lifestyle

###### Plant

###### Vehicles

Subject Consistency

Subject Consistency

Subject Consistency

Subject Consistency

Subject Consistency

Imaging Quality

Background Consistency

Imaging Quality

Background Consistency

Imaging Quality

Background Consistency

Imaging Quality

Background Consistency

Imaging Quality

Background Consistency

Aesthetic Quality

Temporal Flickering

Aesthetic Quality

Temporal Flickering

Aesthetic Quality

Temporal Flickering

Aesthetic Quality

Temporal Flickering

Aesthetic Quality

Temporal Flickering

0 0.5 1

0 0.5 1

0 0.5 1

0 0.5 1

0 0.5 1

Dynamic Degree

Motion Smoothness

Dynamic Degree

Motion Smoothness

Dynamic Degree

Motion Smoothness

Dynamic Degree

Motion Smoothness

Dynamic Degree

Motion Smoothness

Open-Sora On-device Sora

Figure 11. A visual comparison of videos generated by On-device Sora and Open-Sora [83], evaluated using VBench [28].

(i.e., a larger d), the overhead of loading unloading blocks is further minimized, fully accelerating the overall model inference under the device’s run-time memory constraints. Dynamically keeping some model blocks in memory with a retrained state is particularly advantageous for STDiT [83] that is iteratively executed for denoising steps, which entails loading and reusing the same blocks for each step.

#### B. Appendix - Additional Experiments

##### B.1. Video Generation Performance

In Sec. 7.1, we evaluate the quality of videos generated on an iPhone 15 Pro [1] and compare them to videos produced by Open-Sora running on NVIDIA A6000 GPUs using VBench [28] as the evaluation benchmark. The evaluation is conducted on 68-frame videos at 256×256 resolution, using text prompts provided by VBench [28], consisting of 100 examples each across eight categories: animals, architecture, food, humans, lifestyle, plants, scenery, and vehicles. Tab. 6 and Fig. 11 present the comprehensive comparison of generated videos on all categories.

##### B.2. Concurrent Inference with Dynamic Loading

Fig. 12-(b) illustrates the model block loading and inference cycles of STDiT [83] when applying the proposed Concurrent Inference (Sec. A.1), whereas Fig. 12-(a) depicts the case without its application. It can be observed that, with Concurrent Inference, the GPU executes each model block for inference without almost no idle time in parallel with the model block loading, indicated by the overlap between the red (loading) and black (inference) boxes in Fig. 12-(b), resulting in a block inference latency reduction from 29s to 23s. In contrast, without Concurrent Inference, each model block inference is executed only after the corresponding block is fully loaded to memory, indicated by the lack of overlap between the red (loading) and black (inference) boxes in Fig. 12-(a). Consequently, the total latency of the full denoising process using multiple executions of STDiT [83] is reduced by approximately 25%, decreasing from 1,000 to 750 seconds for 30 denoising steps. Given that STDiT is executed multiple times to perform numerous denoising steps, it significantly accelerates the STDiT’s overall inference. In addition, when applied with Dynamic Loading (Sec. A.1), it achieves an additional average speed improvement of 17 seconds as reloading is not required for

certain model blocks that are retained in memory, provided in Eq. (11).

tably, Pyramidal Flow employs a cascading model procedure in which subtle variances are treated as noise to be subsequently denoised. Since these variances become negligible in the final output, Linear Proportional Leap is particularly well-suited to this architecture. Our experiments involving multiple prompts (as shown in Fig. 15) confirmed that removing nearly half of the total denoising steps does not compromise performance while preserving both visual quality and temporal consistency. Furthermore, we observed that additional steps at later units can also be omitted due to the high straightness of Pyramidal Flow. Future work will further investigate this phenomenon by exploring the compatibility of flow-based models with the current LPL algorithm, developing the algorithm to be more compatible to other type of models.

이 슬

⋯

기

(a) Sequential Loading and Inference

⋯

|29s|
|---|

[Figure 62]

67s 29s 23s 24s 24s

28s 28s 28s

[Figure 63]

| | | | | | | |[Figure 64]| | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

25s 24s 24s 24s 24s 25s 25s 25s⋯

(b) Concurrent Inference

### ⋯

[Figure 65]

|23s|
|---|

63s 25s 19s 24s 23s

| |6|7s| | |29s|
|---|---|---|---|---|---|
| | | | |[Figure 66]| |

[Figure 67]

Load Inference

| |
|---|

| |
|---|

- Figure 12. The block loading and inference cycles of STDiT [83] without (a) and with (b) Concurrent Inference. The red box repr block inference on the iPhone 15 Pro’s GPU.

[Figure 68]

23s 24s 24s

|29s|
|---|

28s 28s 28s

| | | | | | | | | |[Figure 69]|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

[Figure 70]

⋯

⋯

[Figure 71]

⋯

[Figure 72]

⋯

(a) Sequential Loading and Inference

| | | | |[Figure 73]|
|---|---|---|---|---|
| | |(b) Concurrent Inference| | |

| | | | | |
|---|---|---|---|---|
| |[Figure 74]<br><br>| | | |

Load Inference

| |
|---|

| |
|---|

Load Inference

(a) Sequential Loading and Inference

(b) Concurrent Inference

이 슬

기

- Figure 13. The block loading and inference cycles of T5 [48] without (a) and with (b) Concurrent Inference.

(a) Sequential Loading and Inference

|ese|n|63sts the loading25s 19s cycle,24s 23sw|23shile|25sthe black24s 24sbox24sindicates24s 25s the25smo25|ds|[Figure 75]<br><br>el<br><br>[Figure 76]|
|---|---|---|---|---|---|---|
| | |(b) C|oncu|rrent Inference| | |

[Figure 77]

| |
|---|

| |
|---|

Models Time (s) Resolution DiT Steps Speedup

Load Inference

|Pyramidal Flow - 2B<br><br>|262.03<br><br>|1280 × 768|270<br><br>|1.00×|
|---|---|---|---|---|
|Pyramidal Flow - 2B - LPL (2)<br><br>|165.15<br><br>|1280 × 768|159<br><br>|1.59×|
|Pyramidal Flow - 2B<br><br>|48.88|640 × 384<br><br>|270|1.00×|
|Pyramidal Flow - 2B - LPL (2)<br><br>|30.54<br><br>|640 × 384|159<br><br>|1.60×|

Table 7. Ablation study conducted on Pyramidal Flow [29] based on the miniFLUX [32] architecture. ‘DiT steps’ represent the total number of DiT forward computations required for the denoising process. Speedups are measured relative to the corresponding model, with execution time computed as the average of three independent runs. Notably, LPL does not degrade video quality, even when using nearly half of its total steps.

Fig. 13 shows the case of T5 [48]. Unlike STDiT, which exhibits similar latencies for both block loading and inference execution, T5 exhibits a much longer block loading latency relative to inference latency. Consequently, the overlap between the model loading and inference is minimal, as shown by the small region of overlap between the red (loading) and black (inference) boxes. As a result, the latency improvement is expected to be less substantial, as in Eq. (10). Nevertheless, the inference latency is reduced from 164 to an average of 137 seconds, achieving a reduction of 16%. This result implies the effectiveness of concurrent loading and inference, even in cases when there is an imbalance between the latencies of model block loading and inference.

1.0

1.0

| |Stage 1<br><br>Stage 2<br><br>Stage 3<br>| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| |Stage 1<br><br>Stage 2<br><br>Stage 3<br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

CosineSimilarity

0.8

0.8

0.6

0.6

0.4

0.4

0 2 4 6 8 10 12 14 16 18 Time Step

72 74 76 78 80 Time Step

(a)

(b)

Figure 14. Cosine similarities between adjacent drifts estimated by Pyramidal Flow [29] at each stage. (a) and (b) show cosine similarities computed at the first and last units, respectively. The values approach 1.0 at the last unit across all stages. Computations follow the experimental settings in [29], using 20 denoising steps in (a) and 10 in (b).

##### B.3. Ablation Study of Linear Proportional Leap (LPL) on Other Text-to-Video Models

The primary constraint for applying Linear Proportional Leap (LPL) is that the model must be a flowmatching–based model trained with a rectified flow target [38]. Since Pyramidal Flow [29], one of the state-ofthe-art open-source video generation models, satisfies this condition, we conduct experiments on it.

#### C. Appendix - Discussions and Limitations

Latency Improvement. Although On-device Sora enables efficient video generation, the generation latency remains higher compared to utilizing high-end GPUs; it requires several minutes to generate a video, whereas an NVIDIA A6000 GPU takes one minute. This discrepancy is evident due to the substantial disparity in computational resources between them. For instance, the iPhone 15 Pro’s GPU features up to 2.15 TFLOPS with 3.3 GB of available memory, compared to the NVIDIA A6000, which offers up to 309

In a manner similar to Open-Sora with Rectified Flow [38], Pyramidal Flow consistently demonstrates a high degree of straightness in its generation process following the initial generation unit (as illustrated in Fig. 14). As demonstrated in Tab. 7, the application of LPL during generation reduces the overall generation time by nearly 38%, decreasing from 264.60 seconds to 165.15 seconds. No-

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Prompt: ”A movie trailer featuring the adventures of a 30-year-old spaceman wearing a red wool-knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors.”

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Prompt: ”A cybernetic humanoid scans the streets with red eyes as holographic screens flicker around its head, displaying futuristic data.”

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Prompt: ”A green-eyed woman with auburn hair and a mysterious smile, wearing a flowing red dress.” Figure 15. Videos generated using naive Pyramidal Flow [29] (top) and Pyramidal Flow with the proposed Linear Proportional Leap (bottom) under identical prompts. Linear Proportional Leap (LPL) reduces the total number of denoising steps by nearly half (i.e., from 270 to 159), while maintaining video generation results that are highly comparable to those achieved using the full set of denoising steps and preserving most details.

TFLOPS and 48 GB of memory. Despite this significant resource gap, On-device Sora achieves exceptional efficiency in video generation. Currently, it utilizes only the iPhone 15 Pro’s GPU. We anticipate that the latency could be significantly enhanced if it can leverage NPU (Neural Processing Unit), e.g., the iPhone 15 Pro’s Neural Engine [1], which delivers a peak performance of 35 TOPS. However, the current limitations in Apple’s software and SDK support for state-of-the-art diffusion-based models [83] make the iPhone’s NPU challenging to utilize effectively. We look forward to the development of software support for NPUs

and leave the exploration of this for future work. Also, we plan to investigate the potential of NPUs on a variety of mobile devices, such as Android smartphones.

Model Optimization. In On-device Sora, only T5 [48] is quantized to int8, reducing its size from 18 GB to 4.64 GB, while STDiT [83] and VAE [15] are executed with float32 due to their performance susceptibility, which has a significant impact on video quality. Additionally, we do not apply pruning [49] or knowledge distillation [19], as these methods also drop visual fidelity. In particular, we observe that naively shrinking STDiT [83] leads to significant vi-

sual loss, caused by iterative denoising steps, where errors propagate and accumulate to the final video. Another practical difficulty in achieving lightweight model optimization is the lack of resources required for model optimization; both re-training and fine-tuning state-of-the-art diffusionbased models typically demand several tens of GPUs, and the available datasets are often limited to effectively apply optimization methods. To tackle these challenges, we propose model training-free acceleration techniques for video generation in this work, i.e., Linear Proportional Leap (Sec. 4) and Temporal Dimension Token Merging (Sec. 5).

Straightness Constraints. For video generation models that do not exhibit straightness during their denoising procedures, such as CogVideoX [74], which employs DPM-Solver [82], Linear Proportional Leap (LPL) is currently inapplicable. In such cases, as described in previous work [75], an alternative approach involves predicting modifications to the model’s noise schedule by employing additional methodologies, such as reinforcement learning. However, these kind of methods require additional training which may invoke extra costs, and necessitates retraining whenever the target model architecture changes. By contrast, LPL is a plug-and-play algorithm that can be applied in a model-agnostic manner, as long as a sufficient level of straightness is maintained during the denoising process, thereby eliminating the need for additional training. In future work, we plan to investigate the optimal reduction point achievable with this algorithm, and we anticipate that it will evolve into a widely applicable methodology for all Rectified Flow-based models [38].

