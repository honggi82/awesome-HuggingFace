# arXiv:2406.16260v1[cs.CV]24Jun2024

## Video-Infinity: Distributed Long Video Generation

https://video-infinity.tanzhenxiong.com Zhenxiong Tan ∗ Xingyi Yang ∗ Songhua Liu Xinchao Wang † National University of Singapore zhenxiong@u.nus.edu xinchao@nus.edu.sg

###### Sync

###### GPU(0)

###### GPU(1)

###### GPU(2)

###### GPU(8)

[Figure 1]

[Figure 2]

[Figure 3]

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

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

• • •

N/8 frames

N/8 frames

N/8 frames

N/8 frames

A loyal golden retriever is waiting his owner back home, in the morning

A loyal golden retriever is waiting his owner back home, at the night

A loyal golden retriever is waiting his owner back home, in the rain

A loyal golden retriever is waiting his owner back home, in the snow

2304 frames (N)

|[Figure 21]| |[Figure 22]<br><br>[Figure 23]| | | | | |[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]| |[Figure 30]<br><br>[Figure 31]| | | | | | | | | | |[Figure 32]| | |[Figure 33]<br><br>[Figure 34]| | | | | | |[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]| | | | | | | | | | |[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]|[Figure 46]| | | | | | | | | | | |[Figure 47]| | | |[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]| | | | | |[Figure 54]<br><br>[Figure 55]| | |[Figure 56]<br><br>[Figure 57]| | | | | | | |[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]| | | | | | |[Figure 64]|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

0:00 0:19 0:38 0:57 1:16 1:35

GPU Time: 5mins

Figure 1: Multiple GPUs parallelly generate a complete video, producing 2300 frames in 5 minutes.

### Abstract

Diffusion models have recently achieved remarkable results for video generation. Despite the encouraging performances, the generated videos are typically constrained to a small number of frames, resulting in clips lasting merely a few seconds. The primary challenges in producing longer videos include the substantial memory requirements and the extended processing time required on a single GPU. A straightforward solution would be to split the workload across multiple GPUs, which, however, leads to two issues: (1) ensuring all GPUs communicate effectively to share timing and context information, and (2) modifying existing video diffusion models, which are usually trained on short sequences, to create longer videos without additional training. To tackle these, in this paper we introduce Video-Infinity, a distributed inference pipeline that enables parallel processing across multiple GPUs for long-form video generation. Specifically, we propose two coherent mechanisms: Clip parallelism and Dual-scope attention. Clip parallelism optimizes the gathering and sharing of context information across GPUs which minimizes communication overhead, while Dual-scope attention modulates the temporal self-attention to balance local and global contexts efficiently across the devices. Together, the two mechanisms join forces to distribute the workload and enable the fast generation of long videos. Under an 8 × Nvidia 6000 Ada GPU (48G) setup, our method generates videos up to 2,300 frames in approximately 5 minutes, enabling long video generation at a speed 100 times faster than the prior methods.

∗Equal Contribution †Corresponding Author

Preprint. Under review.

### 1 Introduction

A long-standing pursuit of human being is to replicate the dynamic world we live in, in the digital system. Traditionally dominated by physics and graphics, this effort has recently been enhanced by the emergence of data-driven generative models [20, 9, 5, 7], which can create highly realistic images and videos indistinguishable from reality. However, these models typically produce very short video segments, with most limited to 16-24 frames [4, 1, 2]. Some models extend to 60 or 120 frames [30, 10], but compromise heavily on resolution and visual quality.

Generating long video poses substantial challenges, primarily due to the extensive resource demands for model training and inference. Current models, constrained by available resources, are often trained on brief clips, making it difficult to sustain quality over longer sequences. Moreover, generating a minute-long video in one go can overwhelm GPU memory, making the task seem elusive.

Existing solutions, including autoregressive, hierarchical, and short-to-long methods, offer partial remedies but have significant limitations. Autoregressive methods [6, 29] produce frames sequentially, dependent on preceding ones. Hierarchical methods [3, 29, 31] create keyframes first, then fill in transitional frames. Furthermore, some approaches treat a long video as multiple overlapping short video clips [19, 25]. These methods are not end-to-end; they often miss global continuity, require extensive computation, especially in regions of overlap, and struggle with consistency across segments.

To bridge these gaps, we introduce a novel framework for distributed long video generation, termed Video-Infinity. On the high level, it work in a divide-and-conquer principle. It breaks down the task of long video generation into smaller, manageable segments. These segments are distributed across multiple GPUs, allowing for parallel processing. All clients should work collaboratively to ensure the final video is coherent in semantics.

This setup, while straightforward, faces two principal challenges: ensuring effective communication among all GPUs to share and contextual information, and adapting existing models—typically trained on shorter sequences—to generate longer videos without requiring additional training.

To overcome these challenges, we introduce two synergistic mechanisms: Clip parallelism and Dual-scope attention. Clip parallelism enables efficient collaboration among multiple GPUs by splitting contextual information into three parts. It uses an interleaved communication strategy to complete the sharing in three steps. Building on the capabilities of Clip parallelism, Dual-scope attentionmeticulously adjusts the temporal self-attention mechanisms to achieve an optimal balance between local and global contexts across devices. This balance allows a model trained on short clips to be extended to long video generation with overall coherence.

Even more exciting, by leveraging both strategies, Video-Infinity reduces memory overhead from a quadratic to a linear scale. With the power of multiple device parallelism and sufficient VRAM, our system can generate videos of any, potentially even infinite length.

- As a results, our method significantly extends the maximum length of videos that can be generated and accelerates the speed of long video generation. Specifficly, on an 8 × Nvidia 6000 Ada (48G) setup, our method manages to generate videos up to 2300 frames in just 5 minutes. Our contributions are summarized as follows: (1) We are the first to address long video generation using distributed parallel computation, enhancing scalability and reducing generation times. (2) We introduce two interconnected mechanisms: Clip parallelism, which optimizes context information sharing across GPUs, and Dual-scope attention, which adjusts temporal self-attention to ensure video coherence across devices. (3) Our experiments show that, compared to the existing ultra-long text-to-video method Streaming T2V [6], our approach can be up to 100 times.

### 2 Related works

##### 2.1 Diffusion models

Diffusion models have gained significant attention in recent years due to their impressive ability to generate high-quality media. Originally introduced for image synthesis, models like Denoising Diffusion Probabilistic Models (DDPM) [8] and Latent Diffusion Models (LDM) [20] have demonstrated state-of-the-art performance in image generation. These models progressively denoise a Gaussian

noise distribution by learning a sequence of reverse transformations. Beyond images [8, 20], diffusion models have also shown promise in audio [12, 28, 14] and 3D generation [15, 18]. Adaptations of diffusion models for video generation incorporate temporal modules to capture the sequential nature of video frames. For instance, Video Diffusion Models (VDM) [9] and Flexible Diffusion Model (FDM) [5] effectively extend diffusion frameworks to video data, overcoming challenges like temporal consistency and quality degradation. More recent models such as AnimateDiff [4], ModelScope [26], and VideoCrafter [1, 2] can now produce video clips with better dynamics and improved visual quality.

##### 2.2 Techniques for long video generation

Streaming T2V [6] introduces a method that relies on a conditional attention module to ensure smooth transitions between video segments and a scene-preserving mechanism for content consistency. However, this method requires training and is not end-to-end, posing limitations on its practicality. FreeNoise [19] utilizes rescheduled noise sequences and window-based temporal attention to improve video continuity. Despite these innovations, the rescheduled noise contributes to limited dynamics in the generated videos, and the overlapping attention windows introduce additional computational overhead. NUWA-XL [29] from the NUWA series employs a “coarse-to-fine” autoregressive approach, using a global diffusion model to generate keyframes and local models to fill the intermediate frames. Although promising, NUWA-XL has been trained only within a narrow domain and has not yet made its models and code available, limiting both its evaluation and reproducibility. Gen-L-Video [25] adapts short video diffusion models to handle long videos conditioned on multiple texts without requiring additional training. This approach cleverly uses latent overlaps to extend video length, which is a common strategy among recent methodologies. SEINE [3] leverages a random-mask diffusion method to automate the generation of transition videos between scenes, guided by textual descriptions. Like other models, SEINE employs an autoregressive approach and requires image conditioning to facilitate the generation process.

##### 2.3 Distributed diffusion

Recently, to reduce the latency of each denoising step in diffusion models, various distributed parallel methods have been applied to image diffusion models. ParaDiGMS [23] utilizes step-based parallelism, where each denoising step is executed on a different GPU device in parallel. However, this approach tends to waste much computation. Another method, DistriFusion [13], employs a technique of dividing images into patches, allowing different patches to be denoised on separate GPUs. This approach ensures synchronization among patches and achieves minimal computational waste. However, it is designed specifically for image diffusion and requires significant communication overhead and specialized hardware support to achieve low latency.

### 3 Preliminaries

##### Diffusion Models in Video Generation

The process of generating videos using diffusion models involves progressively denoising the latent representation, denoted as xt, where t ranges from 0 to T. The initial noisy video latent is represented by a random noise tensor xT. With each denoising step, xt is updated to a clearer latent xt−1. This iterative process continues until xT is denoised to x0, which is then fed into a decoder to generate the final video. The key aspect of updating xt to xt−1 is computing the noisy prediction ϵt, given by:

ϵt = Eθ(xt), (1) where Eθ represents the diffusion model.

The diffusion model Eθ can be implemented using various architectures, such as U-Net [21, 9, 5, 4, 1] or DiT [17, 10, 30]. These diffusion models are generally composed of several similar layers. More

specifically, the initial random noise tensor is written as xT ∈ RF×H×W×C, where F represents the number of frames, H and W denote the height and width of each frame, respectively, and C is the number of channels.

′×W′×C′, where F remains constant across layers. The dimensions H′, W′, and C′ can vary due to the down-sampling and up-sampling operations of the U-Net architecture.

The latent tensor v in each layer generally maintains a consistent shape, v ∈ RF×H

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

device(i-1) device(i) device(i+1)

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Spatial

Temporal

device(0) device(1) device(2)

send() receive()

all_gather()

Communication

(a) (b)

Figure 2: (a) Pipeline of Video-Infinity: The latent tensor is split into clips and distributed to different devices. The diffusion model predicts noise in parallel with communication, and the noises are concatenated to produce the final output. (b) Illustration of Clip parallelism: In each layer of the video diffusion module, spatial modules operate independently, whereas temporal modules synchronize context elements cipre, cipost, and ciglobal. Peer-to-peer and collaborative communications are employed.

These layers in the diffusion model Eθ are usually composed of two main types of modules: spatial and temporal. The spatial modules receive slices of the latent v shaped v ∈ RH

′×W′×C′ (a single frame), representing tokens for each video frame in the latent space. They independently process spatial features within each frame. The temporal modules receive elongated strips of the latent tensor v shaped v ∈ RF×C

′

, representing tokens containing temporal information across frames at specific spatial locations. They capture temporal dependencies between frames at each location.

### 4 Distributed Long Video Generation

- At the core of our pipeline, Video-Infinity segments the video latent into chunks, which are then distributed across multiple devices. An overview of our method is shown in Figure 3, where we divide the video latent along the temporal dimension. Such partitioning allows for parallel denoising on different devices, each handling non-overlapping frames. To facilitate this, we propose Clip parallelism, detailed in in Section 4.1, a mechanism that efficiently synchronizes temporal information across devices. Additionally, we incorporate Dual-scope attention in Section 4.2, which modulates temporal attention to ensure training-free long video coherence.

Formally, Video-Infinity splits the noisy latent xT ∈ RF×H×W×C into N sub-latent clips xiT ∈ RF

clip×H×W×C, where i ∈ [1,N], Fclip = F/N represents the number of frames in each clip, and N represents the total number of clips. This structured segmentation facilitates an even load distribution across N devices. Additionally, the spatial modules of video diffusion models operate independently across frames, which eliminates the need for inter-device communication and maintains consistency in the outputs across different devices.

##### 4.1 Clip parallelism for video diffusion

To ensure coherence among clips distributed on different devices, we propose Clip parallelism, shown in Figure 3. It parallelizes the temporal layers for video diffusion models and enables efficient inter-device communication.

Parallelized temporal modules. In the standard diffusion model, a temporal module aggregates features across frames, which could be simplified as

vout = temporal(vin), (2) where vin ∈ RF×∗×C

′

is the input feature of this temporal layer.

However, Video-Infinity distributes input feature tensors vin across multiple devices, dividing them into several clips vini ∈ RF

clip×∗×C′, each placed on device(i). To facilitate distributed inference without modifying the original structure of the temporal modules, we redefine the temporal operation. This modified operation now considers not only the current clip, but also adjacent clips and global semantics. Conceptually, the parallelized temporal modules are defined as follows:

vouti = temporalParallel vini ,ci , (3) ci = cipre,cipost,cglobal (4)

where ci stands for the temporal information that enriches each device’s computation by incorporating inter-device context. Each ci includes temporal information from the preceding device(i-1) via cipre, and from the succeeding device(i+1) via cipost. Furthermore, cglobal is a selective aggregate of inputs from all devices, optimizing global information coherence and reducing overhead.

The output for each device, vouti , reflects localized computations augmented by these contextual inputs. The complete output of the layer, vout, is obtained by concatenating the outputs from all devices:

vout = Concat vouti |i ∈ [1,N] (5)

This concatenation provides a holistic view of the processed features, maintaining temporal coherence across the distributed system. Further details on how these temporal modules integrate context will be discussed in Section 4.2.

Three-round context communication. Redefining the temporal modules necessitates efficient communication of the context components ci = cipre,cipost,cglobal . This is achieved through a three-stage synchronization process, where each stage addresses a specific part of the context, as illustrated in Figure 3.

device(0)

device(1) device(2) device(3) device(4)

- T1

Synchronized Unsynchronized

- T2
- T3

| | |
|---|---|
| | |

| | |
|---|---|
| | |

In the first stage, T1, each device(i) broadcasts its global context ciglobal with all other devices through an all_gather() operation. This operation ensures that every device receives the global context, maintaining global consistency across the entire video.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

End

Share & All gather

The subsequent stages, T2 and T3 focus on exchanging neighboring contexts. Due to connection limits3, we employ an interleaved strategy. In T2, odd-numbered nodes send their cipre+1 to their subsequent device(i+1), and even-numbered nodes send their cipost−1 to device(i-1). In T3, this pattern reverses—odd-numbered devices receive context from their predecessors, and even-numbered devices from their successors. This approach prevents bottlenecks, optimizes channel usage, and minimizes deadlock risks.

Figure 3: Tree different stages in the communication process of Clip parallelism

Finally, all nodes complete context synchronization, ensuring that each device has the full context needed to perform its computations. More details can be found in the pseudocode in Appendix A.2.

##### 4.2 Putting each module in parallel

Building upon the Clip parallelism, this section details that how information is synchronized in each temporal module. A key technique here is the Dual-scope attention, which facilitates training-free long video generation and reduces the communication cost.

There are typically three temporal modules in video diffusion models: attention module [24] Attention(), convolution module [16] Conv(), and group normalization module [27] GroupNorm(). We have tailored these modules to integrate into Clip parallelism, enabling distributed processing across multiple devices for efficient and coherent video content synchronization.

DualScope attention. Applying attention in parallel inference incurs new challenges. The original attention module require simultaneous access to all input tokens [22]. To adopt it under Clip

3Only one device can communicate with another at a time.

parallelism, it necessitates aggregating tokens across devices, resulting in tremendous communication costs. Additionally, those attention trained on shorter video clips often degrade in quality when applied to longer sequences.

To address these issues, we introduce the DualScope attention module. It revises the computation of K-V pairs to incorporate both local and global contexts into the attention. For each query token from frame a, its corresponding keys and values are computed from tokens in the frame set Aa = Na ∪ G:

- • Local Context (Na). This includes the |Na| neighboring frames of a, from which the keys and values are derived to capture the local context. This local setup is typically achieved through window attention, focusing on the nearby frames to enhance the temporal coherence.
- • Global Context (G). In contrast, the global context consists of frames uniformly sampled from videos across all devices. This context provides keys and values from a broader range, giving the model access to long-range information

In practice, the keys K and values V are constructed by concatenating the tokens from both contexts K = Concat(Klocal,Kglobal) and V = Concat(Vlocal,Vglobal), where Klocal and Qlocal is derived from Na and Kglobal and Qglobal from G. We find that this modified key-value computation can be easily incorporated into existing temporal attention without additional training, enhancing the coherence of long videos.

In the implementation of Clip parallelism, above reformulated attention largely reduce the communication overhead. Comparing to gathering all tokens of length F, we only synchronize constant

a|

number of tokens. Specifically, we set |cipre| = |cipost| = |N

2 and |cglobal| = |G|, with both |Na| and |G| configured to 16. This reduces data synchronization demands while still capturing essential local and global information.

Convolution module. The temporal convolution module Conv() applies convolution along the temporal dimension to its input vini ∈ RF

clip×C′. In Clip parallelism, the context ci of the Conv() includes cipre and cipost. They are padded to the original sequences. Specifically, cipre consists of the last n frames of vini−1, and cipost consists of the first n frames of vini+1, where n is the receptive field size of the convolution.

Group normalization. In video diffusion model, group normalization is applied to the input tensor vini ∈ RF

clip×H×W×C′ to maintain consistent feature scaling across different frames. In Clip parallelism, each device first computes the group mean µi of its respective video clip. These means are aggregated to compute the global mean µ¯ =

N i=1 µi

N , where N is the number of devices. Subsequently, using µ¯, each device computes its standard deviation σ¯i, which is shared to calculate the global standard deviation σ¯. The global mean µ¯ and global standard deviation σ¯, serving as cglobal, are used for normalization 4.

### 5 Experiments

- 5.1 Setups Base model. In the experiments, the text to video model VideoCrafter2 [2] (320 x 512) is selected

- as the base model of our method. VideoCrafter2, which was trained on 16-frame videos, excels
- at generating video clips that are both consistent and of high quality. It is also the highest-scoring open-source video generation model under the VBench [11] evaluation, achieving the top total score.

Metrics evaluation. VBench [11] is utilized as a comprehensive video evaluation tool, featuring a broad array of metrics across various video dimensions. For each method, videos are generated using the prompts provided by VBench for evaluation. The metrics measured encompass all the indicators under the Video Quality category in VBench, including subject consistency, background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality and imaging quality. Given that VBench’s evaluation is typically performed on video clips of 16 frames, we

4Note that simply averaging the individual standard deviations σi does not yield the true global standard deviation σ¯.

have modified the evaluation method for videos longer than 16 frames: we randomly sample five 16-frame clips from each video to evaluate separately, and then calculate the average score of these assessments.

Baslines. Our approach is benchmarked against several other methods:

- • FreeNoise [19]: We chose FreeNoise as a baseline because it is also a training-free method that can base the VideoCrafter2 [2] model, which also serves as our base model, to generate long videos. It employs a rescheduling technique for the initialization noise and incorporates Window-based Attention Fusion to generate longer videos.
- • Streaming T2V [6]: To assess our method’s effectiveness in generating longer videos, StreamingT2V was chosen as our baseline. Streaming T2V involves training a new model that uses an auto-regressive approach to produce long-form videos. Like our approach, it also has the capability to generate videos exceeding 1000 frames.

OpenSora V1.1 [10], a video diffusion model based on DiT [17], supports up to 120 frames, can generate videos at various resolutions, and has been specifically trained on longer video sequences to enhance its extended video generation capabilities.

Dual-scope attentionsetting. In the implementation of the Dual-scope attention, the number of neighboring frames Ni is set to 16, with 8 frames coming from the preceding clip and 8 frames from the subsequent clip. The number of global frames, G, is set to 16. To balance consistency and dynamics during the denoising process, the weights of frames in G and Ni are dynamically adjusted. Specifically, the weight of G increases by 10 for timesteps t greater than 800, whereas the weight of Ni increases by 10 for timesteps t less than or equal to 800.

Implementation details. By default, all parameters of the diffusion are kept consistent with the original inference settings of VideoCrafter2 [2], with the number of denoising steps set to 30. Our experiments are conducted on 8 × Nvidia 6000 Ada (with 48G memory) . To implement the temporal module in Clip parallelism, we utilized the torch.distributed tool package, employing Nvidia’s NCCL as the backend to facilitate efficient inter-GPU communication. Additionally, all fps conditions are set to 24, and the resolution is set to 512 × 320. Note that the resolution for Streaming T2V cannot be modified; thus, videos are generated at its default resolution (256 × 256 for preview videos and 720 × 720 for final videos).

##### 5.2 Main results

##### Capacity and efficiency.

We evaluated the capabilities of our method on an 8 × Nvidia 6000 Ada (48G) setup. Our approach successfully generated videos of 2300 frames at a resolution of 512 × 320, equivalent to a duration of 95 seconds at 24 frames per second. Remarkably, the entire computation process took approximately 5 minutes (312s), benefiting from efficient communication and the leveraging of multi-GPU parallel processing.

Method Max Time Coast (second)

Frames 128 frames 1024 frames

Ours 2300 21 131 ST2V (preview) - 277 2,196 ST2V (final cut) - 1730 13,726 FreeNoise [19] 128 201 × Open-Sora v1.1 [10] 120 234 ×

Table 1: Comparison of maximum frames and generation times for different methods.

- Table 1 presents the capacities for long video generation of various methods, all measured under the same device specifications. To ensure comparability, we standardized the resolution of the videos generated by all methods to 512x320. For StreamingT2V, we provide two sets of data: one for generating preview videos at 256x256 resolution, and another for final videos produced at a resolution of 720x720. The results demonstrate that our method is the most capable within the end-to-end category, generating the longest videos of up to 2300 frames — 8.2 times more than OpenSora V1.1. Additionally, our method consistently produces the final videos in the shortest time, both for short videos of 128 frames and long videos of 1024 frames. Notably, in the generation of 1024-frame videos, our method is over 100 times faster than StreamingT2V, the only baseline method capable of producing videos of this length. Even when compared to the speed of generating smaller, lower-resolution preview videos by StreamingT2V, our method is 16 times faster.

@Frame 0 @Frame 24 @Frame 48

@Frame 72

@Frame 96

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

S.T2VOursFreeNoiseOursFreeNoiseS.T2V

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

###### O.S.V1.1O.S.V1.1

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

A beagle wearing diving goggles swimming in the ocean while the camera is moving, coral reefs in the background.

[Figure 85]

[Figure 86]

[Figure 87]

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

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

A musician wearing sunglasses playing a guitar on a busy street corner with people passing by.

Figure 4: Comparison of frame images from sample videos generated by different methods.

Video quality. We compared the videos generated by our method with those produced by FreeNoise [19] and StreamingT2V [6] for long video generation. Figure 4 visualizes some frames from videos generated by different methods using the same prompt. Additionally, Table 2 displays the quality of the videos produced by these methods, evaluated across various metrics in VBench [11].

- Figure 4 shows that while the StreamingT2V [6] method generates long videos with sufficient dynamism, they lack consistency between the beginning and end. Conversely, videos generated by FreeNoise [19] maintain consistency in object placement throughout but exhibit minimal variation in visuals. For example, as shown in Figure 4, the video of the person playing the guitar maintains a single pose with only minimal movement. Similarly, the dog on the left remains intently focused on the camera, with no changes in the position of its ears, nose, or body. OpenSora V1.1 [10] failed to generate the first video and the second video’s background was not smooth. In contrast, our method not only ensures better consistency but also features more significant motion in the generated videos.

Method Number of Subject Background Temporal Motion Dynamic Aesthetic Imaging Overall

Frames consistency consistency flickering smoothness degree quality quality Score

V.C.2 16 96.85% 98.22% 98.41% 97.73% 42.50% 63.13% 67.22% 80.58% FreeNoise 64 94.16% 96.63% 98.37% 97.04% 44.44% 60.53% 67.44% 79.80% OpenSora v1.1 64 86.18% 95.83% 98.47% 97.27% 73.61% 51.69% 50.61% 79.09% Ours 64 97.77% 93.90% 97.77% 96.84% 81.94% 59.38% 67.90% 85.07% ST2V 192 75.02% 87.93% 95.96% 94.71% 80.56% 48.08% 57.85% 77.16% Ours 192 91.32% 92.93% 97.40% 95.55% 77.78% 55.49% 66.93% 82.49%

- Table 2: Evaluation metrics: Comparison of performance metrics for various video generation methods as benchmarked by VBench. Bold values represent the best performance within each group.

|[Figure 105]<br><br>23|[Figure 106]<br><br>24|
|---|---|
|original| |

|[Figure 107]<br><br>23|[Figure 108]<br><br>24|
|---|---|
|w.o. ResNet sync| |

|[Figure 109]<br><br>23|[Figure 110]<br><br>24|
|---|---|
|w.o. Attn sync| |

A robot assembling parts in a high-tech futuristic factory.

|[Figure 111]<br><br>14|
|---|

|[Figure 112]<br><br>16|
|---|

|[Figure 113]<br><br>20|
|---|

|[Figure 114]<br><br>24|
|---|

|[Figure 115]<br><br>28|
|---|

|[Figure 116]<br><br>32|
|---|

w.o.localw.o.globaloriginal

|[Figure 117]<br><br>12|
|---|

|[Figure 118]<br><br>16|
|---|

|[Figure 119]<br><br>20|
|---|

|[Figure 120]<br><br>24|
|---|

|[Figure 121]<br><br>28|
|---|

|[Figure 122]<br><br>32|
|---|

|[Figure 123]<br><br>18|
|---|

|[Figure 124]<br><br>20|
|---|

|[Figure 125]<br><br>22<br><br>|
|---|

|[Figure 126]<br><br>24|
|---|

|[Figure 127]<br><br>26|
|---|

|[Figure 128]<br><br>28|
|---|

A beagle wearing diving goggles swimming in the ocean while the camera is moving, coral reefs in the background.

on device 1 on device 2

- Figure 5: Visualization of ablation studies on temporal module communication and context effects in video generation. Top panel: Ablation of communication between the ResLayer module and the Attention module, showcasing two adjacent frames from the video sequence generated on different GPUs. Bottom panel: Effects of ablating different contexts within the Attention module, displaying frames from videos generated post-ablation.

- Table 2 reveals that our method, when compared to our base model VideoCrafter 2 [2], experiences a slight decrease in most metrics except for the metric of dynamic. In the generation of 64-frame videos, the performance of our method shows mixed results compared to other methods, with both advantages and disadvantages noted. However, our average metric scores are higher than those of both FreeNoise and OpenSora V1.1. In the generation of longer 192-frame videos, our method outperforms StreamingT2V, the only other method capable of producing videos of this length, across the majority of evaluated metrics.

##### 5.3 Ablation

As mentioned in Section 4.2, three types of temporal modules (Conv(), GroupNorm(), and Attention()) are adapted to synchronize context in Clip parallelism. To demonstrate the effectiveness of context synchronization by these modules, we conducted ablation experiments and visualized in Figure 5 the impact of removing certain parts of the context synchronization on the quality of the generated videos. We performed ablation on the communication between the temporal Attention() module and the temporal ResNet() module in the video diffusion model, where the ResNet module includes temporal Conv() and temporal GroupNorm() as submodules. Subsequently, we conducted ablations on the global context cglobaland the local context cipre,cipostwithin the Attention() module.

Removing local context. From the top panel of Figure 5, it can be observed that the absence of synchronized information from the ResNet() leads to discrepancies in detail between the last frame on device(1) (frame 23) and the first frame on device(2) (frame 24), which are highlighted in red. These discrepancies, such as differences in the color of the clothes of the person behind the robot and the shape of the parts in the robot’s hands on the table, do not appear in the original inference. When context of the Attention() module is absent, frame 23 and frame 24 become markedly different images, illustrating a significant discontinuity between video segments generated on adjacent devices. These observations suggest that synchronization in both ResNet() and Attention() modules is crucial for preserving visual coherence and continuity across video frames generated on different devices.

Removing global context. The bottom panel of Figure 5 demonstrates that when synchronization of the global context is absent, content consistency within the video is difficult to maintain. For example, in frames 12 and 16, the horizon remains high, but in frames beyond 20, there is a noticeable rise in the horizon. Furthermore, when the local context synchronization is removed, although the content across different device clips remains consistent, the lack of shared context in the transition areas leads to anomalies. For instance, the content of snow in frame 22 abruptly transitions to a dog, highlighted in red. These examples highlight the importance of global and local context synchronization for video generation.

### 6 Conclusion

We presented Video-Infinity, a distributed inference pipeline that leverages multiple GPUs for longform video generation. We present two mechanisms, Clip parallelism and Dual-scope attention, to addressed key challenges associated with distributed video generation. Clip parallelism reduces communication overhead by optimizing the exchange of context information, while Dual-scope attention modified self-attention to ensure coherence across devices. Together, these innovations enable the rapid generation of videos up to 2,300 frames long, vastly improving generation speeds compared to existing methods. This approach not only extends the practical utility of diffusion models for video production but also sets a new benchmark for efficiency in long-form video generation.

### 7 Limitation

To fully harness the potential of our method, it relies on the availability of multiple GPUs. Additionally, our approach does not effectively handle video generation involving scene transitions.

### References

- [1] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023.
- [2] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024.
- [3] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023.
- [4] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [5] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. Advances in Neural Information Processing Systems, 35:27953–27965, 2022.
- [6] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773, 2024.

- [7] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [8] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [9] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [10] hpcaitech. Open-sora. https://github.com/hpcaitech/Open-Sora, 2024.
- [11] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023.
- [12] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.
- [13] Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Ming-Yu Liu, Kai Li, and Song Han. Distrifusion: Distributed parallel inference for high-resolution diffusion models. arXiv preprint arXiv:2402.19481, 2024.
- [14] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audioldm: Text-to-audio generation with latent diffusion models. arXiv preprint arXiv:2301.12503, 2023.
- [15] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2837–2845, 2021.
- [16] Keiron O’shea and Ryan Nash. An introduction to convolutional neural networks. arXiv preprint arXiv:1511.08458, 2015.
- [17] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [18] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [19] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.
- [20] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [21] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.
- [22] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. arXiv preprint arXiv:1803.02155, 2018.
- [23] Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, and Nima Anari. Parallel sampling of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [24] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [25] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023.
- [26] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [27] Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pages 3–19, 2018.

- [28] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. Diffsound: Discrete diffusion model for text-to-sound generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.
- [29] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023.
- [30] Zhang Zhaoyang, Yuan Ziyang, Ju Xuan, Gao Yiming, Wang Xintao, Yuan Chun, and Shan Ying. Mira: A mini-step towards sora-like long video generation. https://github.com/mira-space/Mira, 2024.
- [31] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024.

### A Appendix

- A.1 Communication overhead

Table 3 demonstrates the additional time overhead caused by communication between different temporal modules. The experiments were conducted on multiple Nvidia A5000 GPUs, with two settings: a dual-GPU configuration and an eight-GPU configuration.

Sync Inference Time (s)

2×GPU 8 × GPU Plain 145.4 149.5

+ Conv() 152.9 (5.1% ↑) 157.1 (5.1% ↑) + GroupNorm() 158.3 (8.9% ↑) 160.1 (7.1% ↑)

+ Attention() 170.7 (17.4% ↑) 180.2 (20.5% ↑) Full Sync 182.3 (25.3% ↑) 192.3 (28.6% ↑)

Table 3: Effect of Synchronization on Inference Time

- A.2 Communication Algorithm

Algorithm 1 Distributed Temporal Module Communication Require: i (the ID of the device), vini (the input latent segment) Ensure: Seamless and efficient distribution of frames for video processing.

- 1: Prepare the global context ciglobal using vini
- 2: dist.all_gather(ciglobal)
- 3: if i mod 2 == 1 then
- 4: cipre = dist.recv(i+1)
- 5: Prepare the local context for device(i+1) using vini
- 6: dist.send(cipost+1 )
- 7: cipost = dist.recv(i-1)
- 8: Prepare the local context for device(i-1) using vini
- 9: dist.send(cipre−1)
- 10: else
- 11: cipost = dist.recv(i-1)
- 12: Prepare the local context for device(i-1) using vini
- 13: dist.send(cipre−1)
- 14: cipre = dist.recv(i+1)
- 15: Prepare the local context for device(i+1) using vini
- 16: dist.send(cipost+1 )
- 17: end if

#### B Gallery More videos are available in the supplementary materials and at the following link. Link

