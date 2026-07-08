# arXiv:2408.08189v4[cs.CV]15Aug2025

## FancyVideo: Towards Dynamic and Consistent Video Generation via Cross-frame Textual Guidance

Jiasong Feng1,2,∗ , Ao Ma1,3,∗,† , Jing Wang1,4,∗ , Ke Cao1,5,∗ and Zhanjie Zhang1,6,‡ 1 360 AI Research 2 Beijing University of Technology

- 3 Wuhan University
- 4 Sun Yat-sen University

5 University of Science and Technology of China 6 Zhejiang University {maaoaoma, zhangzhanj}@126.com Abstract

##### 1 Introduction

With the advancement of the diffusion model, the text-toimage (T2I) generative models [Blattmann et al., 2023b; Ho et al., 2022; Luo et al., 2023; Ma et al., 2024; Liu et al., 2025; Cao et al., 2025; Ling et al., 2025; Bi et al., 2024; Ma et al., 2025] can produce high-resolution and photo-realistic images by complex text prompts, resulting in various applications. Currently, many studies [Wang et al., 2024; Guo et al., 2023a] explore the text-to-video (T2V) generative model due to the great success of T2I models. However, building a powerful T2V model remains challenging as it requires maintaining temporal consistency while generating coherent motions simultaneously. Moreover, due to limited memory, most diffusion-based T2V models [Wang et al., 2024; Guo et al., 2023a; Zhang et al., 2024a; Guo et al., 2023b; Chen et al., 2023; Menapace et al., 2024] can only produce fewer than 16 frames of video per sampling without extra assistance (i.e., super-resolution).

Synthesizing motion-rich and temporally consistent videos remains a challenge in artificial intelligence, especially when dealing with extended durations. Existing text-to-video (T2V) models commonly employ spatial cross-attention for text control, equivalently guiding different frame generations without frame-specific textual guidance. Thus, the model’s capacity to comprehend the temporal logic conveyed in prompts and generate videos with coherent motion is restricted. To tackle this limitation, we introduce FancyVideo, an innovative video generator that improves the existing text-control mechanism with the well-designed Cross-frame Textual Guidance Module (CTGM). Specifically, CTGM incorporates the Temporal Information Injector (TII) and Temporal Affinity Refiner (TAR) at the beginning and end of crossattention, respectively, to achieve frame-specific textual guidance. Firstly, TII injects frame-specific information from latent features into text conditions, thereby obtaining cross-frame textual conditions. Then, TAR refines the correlation matrix between cross-frame textual conditions and latent features along the time dimension. Extensive experiments comprising both quantitative and qualitative evaluations demonstrate the effectiveness of FancyVideo. Our approach achieves state-ofthe-art T2V generation results on the EvalCrafter benchmark and facilitates the synthesis of dynamic and consistent videos. Note that the T2V process of FancyVideo essentially involves a text-to-image step followed by T+I2V. This means it also supports the generation of videos from user images, i.e., the image-to-video (I2V) task. A significant number of experiments have shown that its performance is also outstanding.

The existing T2V models [Zhang et al., 2024a; Guo et al., 2023b; Chen et al., 2023; Menapace et al., 2024; Bi et al., 2025; Wang et al., 2025a; Shao et al., 2025] typically employ spatial cross-attention between text conditions and latent features for achieving text control generation. However, as shown in Fig. 2(I), this manner shares the same text condition across different frames, thus lacking the specific textual guidance tailored to each frame. Consequently, these T2V models struggle to comprehend the temporal logic of text prompts and produce videos with coherent motion. Taking AnimateDiff [Guo et al., 2023b] as an example, in Fig. 1, we exhibit its generated video and visualize the [verb]-focused region (which is closely associated with the video motion) based on the attention map from the cross-attention module. Ideally, these regions should transition smoothly over time and align with the semantics of motion instructions. However, as observed in the upper right of the figure, the [verb]-focused region remains nearly identical across different frames due to the consistent textual guidance between frames. Meanwhile, the video exhibits poor motion in the upper left of the figure.

Furthermore, we perform a similar visual analysis for the longer video (e.g., 64 frames) generation and find that this problem is more prominent, as illustrated in the lower part

∗ Equal contribution. † Project leader. ‡ Corresponding author.

Prompt1: “Impressionist style, a yellow rubber duck floating on the wave on the sunset.”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Slight change based on timeline and verb semantics.

AnimateDiffAnimateDiffFancyVideoFancyVideo

t=0 t=5 t=10 t=15

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Noticeable change based on timeline and verb semantics.

t=0 t=5 t=10 t=15

Prompt2: “Slow motion steam rises from a hot cup of coffee.”

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

t=0 t=7 t=14 t=21 t=28

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

t=35 t=42 t=49 t=56 t=63

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

t=0 t=7 t=14 t=21 t=28

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

t=35 t=42 t=49 t=56 t=63

Synthesized videos The attention maps of [verb]

Figure 1: The generated videos and the attention maps of [verb] belong to FancyVideo and AnimateDiff. We present the 16-frame video (top) and longer 64-frame video (bottom). Due to the inadequate time-specific textual guidance in the AnimateDiff, the [verb] focused region remains almost constant, resulting in a lack of motion in the video. In contrast, FancyVideo effectively alleviates this issue through crossframe textual guidance. The [verb] focused region changes based on the timeline and semantics, thereby generating motion-rich videos.

of Fig. 1. Therefore, we believe this approach hampers the advancement of video dynamics and consistency and is suboptimal for video generation tasks based on text prompts.

To this end, we present a novel T2V model named FancyVideo, capable of comprehending complex spatialtemporal relationships within text prompts. By employing a cross-frame textual guidance strategy, FancyVideo can generate more dynamic and plausible videos in a sampling process. Specifically, to boost the model’s capacity for understanding spatial-temporal information in text prompts, we optimize the spatial cross-attention through the proposed Cross-frame Textual Guidance Module (CTGM), comprising a Temporal Information Injector (TII) and Temporal Affinity Refiner (TAR). As illustrated in Fig. 2(II), TII injects temporal information from latent features into text conditions, building cross-frame textual conditions. Then, TAR refines the affinity between frame-specific text embedding and video along time dimension, adjusting the temporal logic of textual guidance. Through the cooperative interaction between TII and TAR, FancyVideo fully captures the motion logic embedded within

images and text. Consequently, its motion token-focused area shifts logically with frames, as illustrated in the lower right part of Fig. 1. This characteristic enables FancyVideo to produce dynamic videos, as displayed in the lower left part of the figure. Experiments demonstrate that FancyVideo successfully generates dynamic and consistent videos, achieving the SOTA results on the EvalCrafter [Liu et al., 2023] benchmark and the competitive performance on UCF-101 [Soomro et al., 2012] and MSR-VTT [Xu et al., 2016]. Additionally, FancyVideo supports generating videos from user-input images, i.e., the image-to-video task. We have also conducted extensive experiments to demonstrate the superiority of our method.

Contributions. 1) We introduce FancyVideo, the pioneering endeavor as far as our knowledge extends, delving into cross-frame textual guidance for the T2V task. This approach offers a fresh perspective to enhance current text-control methodologies. 2) We propose the Cross-frame Textual Guidance Module (CTGM), which constructs cross-frame textual conditions and subsequently guides the modeling of latent

###### Ⅱ. CTGM

× Matrixmultiplication

Temporal Affinity Refiner

|TAR|
|---|

Temporal Information Injector

|TII|
|---|

|TAR| | |
|---|---|---|
| | | |

###### ×

Ⅰ. Spatial Cross-Attention

V

T

× K

Q

×

Spatial Latent

Text Condition

V

Feature

T

× K

TII

Q

Spatial Latent Feature

Spatial Latent Feature

Text Condition

Text Condition

Figure 2: The structure of spatial cross-attention and CTGM.

features with robust temporal plausibility. It can effectively enhance the motion and consistency of video. 3) We demonstrate that incorporating cross-frame textual guidance represents an effective approach for achieving high-quality video generation. Our experiments showcase that this approach attains state-of-the-art results on both quantitative and qualitative evaluations.

##### 2 Related Work

Text to Video Generation. Generative models like GANs [Wang et al., 2020; Munoz et al., 2021; Gur et al., 2020], auto-regressive models [Wang et al., 2019; Yan et al., 2021], and implicit neural representations [De Luigi et al., 2023] have been explored for video generation. Recently, diffusion models [Rombach et al., 2022; Zhang et al., 2024b; Zhang et al., 2024c] have advanced text-to-image quality. Stable Diffusion [Rombach et al., 2022] uses a VAE [Kingma and Welling, 2013] latent space to reduce cost [Jiang et al., 2023]. T2V models [Wu et al., 2023a] add temporal layers to T2I models but often lack frame-to-frame consistency. We propose cross-frame textual guidance to improve temporal coherence.

Image-conditioned Video Generation. To bridge the gap between text and video, recent work leverages images for clearer video generation. SVD [Blattmann et al., 2023a] treats images as noisy latent inputs, while MoonShot [Zhang et al., 2024a] improves semantic consistency using a CLIP encoder. Though effective, these I2V methods rely on input images. Hierarchical approaches [Zeng et al., 2023; Chen et al., 2023] use images as keyframes to extend video length with fewer constraints. These methods, though I2Vcapable, are essentially T2V. FancyVideo adopts a hierarchical design with cross-frame textual guidance, enabling more frames per iteration and faster inference.

##### 3 Method

###### 3.1 Preliminaries

Latent Diffusion Models. LDMs [Sohl-Dickstein et al., 2015; Ho et al., 2020; Zhang et al., 2025; He et al., 2025; Lu et al., 2025] enhance efficiency by running diffusion in the VAE-compressed latent space [Kingma and Welling, 2013]

instead of pixel space. The forward process adds Gaussian noise (ϵ ∼ N(0,I)) to the latent code z, yielding:

zt = √α¯tz + √1 − α¯tϵ, (1)

where α¯t denotes a noise scheduler with timestep t. For the inverse process, it trains a denoising model (fθ) with the objective:

Ez∼p(z),ϵ∼N(0,I),t ∥y − fθ(zt,c,t)∥2 , (2)

where c represents the condition and target y can be noise ϵ, denoising input z or v-prediction (v = √α¯tϵ −

√1 − α¯tz) in [Salimans and Ho, 2022]. In this paper, we adopt the vprediction as the supervision.

Zero terminal-SNR Noise Schedule. Previous studies proposed zero terminal SNR [Lin et al., 2024] to handle the signal-to-noise ratio (SNR) difference between the testing and training phase, which hinders the generation quality. At training, due to the residual signal left by the noise scheduler, the SNR is still not zero at the terminal timestep T. However, the sampler lacks realistic data when sampling from random gaussian noise during the test, resulting in a zero SNR. This train-test discrepancy is unreasonable and an obstacle to generating high-quality videos. Therefore, following the [Lin et al., 2024; Girdhar et al., 2023], we scale up the noise schedule and set α¯T = 0 to fix this problem.

###### 3.2 Model Architecture

Fig. 3 illustrates the overall architecture of FancyVideo. The model is structured as a pseudo-3D UNet, which integrates frozen spatial blocks, sourced from a text-to-image model, along with Cross-frame Textual Guidance Modules (CTGM) and temporal attention blocks. The model takes three features as input: noisy latent Zn ∈ Rf×h×w×c, where h and w indicate the height and width of the latent, f signifies the number of frames, and c denotes the channels of the latent; mask indicator M ∈ Rf×h×w×1, with elements set to 1 for the first frame and 0 for all other frames; image indicator I ∈ Rf×h×w×c, with initial image as the first frame and 0 for all other frames. The denoising input Z is formed by concatenating Zn, M and I along the channel dimension, represented as Z = [Zn;M;I] ∈ Rf×h×w×(2c+1). Within each spatial block, we first incorporate prior knowledge of the motion score as embeddings. In each subsequent cross-attention layer, CTGM is employed to capture the intricate dynamics described in the text prompts. Afterward, we apply temporal attention blocks to enhance the temporal relationships across various patches.

###### Motion Embedding

To achieve more controllable video generation in terms of motion amplitude, we introduce motion score information calculated by the RAFT [Teed and Deng, 2020] alongside the timestep information. Specifically, we calculate a motion score for the training samples in the dataset within a range of 0.1 to 10. The score are then encoded into motion features through a motion embedding layer. By controlling the motion score, we can generate videos with stronger motion. However, simply adjusting the score may lead to unrealistic motion. We use CTGM to prevent these issues.

###### Ⅰ. Overall Architecture

###### Ⅱ. Spatial Block

Cross-frame Textual Guidance Module

Temporal Information Injector

“Teddy bear surfer rides the

| | | | |
|---|---|---|---|

Text-to-Image Model

𝒯

Text embedding

[Figure 57]

|[Figure 58]|
|---|

wave in tropics.”

(CTGM)

CLIP

Temporal Affinity Refiner

ResNet Block

SelfAttention

Cross Attention

𝒯

Noisy latent

time + motion embedding

[Figure 59]

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

[Figure 60]

0 0

0 0

0 0

0 0

[Figure 61]

0 0

0 0

0 0

0 0

- 0 1 1

- 0

- 0

- 0

0 0

0 0

0

0

0 0

0 0

0 0

- 0 0 … 0 0

- 1

- 1

- 1

- 1

1 1

1 1

1 1

1

1

1 1

- 1 1 … 1 1

[Figure 62]

[Figure 63]

0 0

0 0

0 0

0 0

0 0 … 0 0

0 0 … 0 0

Ⅲ. Cross-frame Textual Guidance Module

0 0 … 0 0

0 0 … 0 0

0 0

0 0

0 0

0 0

0 0 … 0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

0 0

Temporal Information Injector

Q

Cross-attention

𝒵𝑛 ℳ ℐ

𝒯𝑟𝑒𝑝

Q

| | | | |
|---|---|---|---|

𝑛

K

Temporary Attention

| | | | | |
|---|---|---|---|---|
| | | | | |

K V

V

𝒵

ℎ × 𝑤

𝒜𝑟𝑒𝑓

Q K V

𝒜

Temporary Attention

Linear

×

𝒵𝑡

Linear

Temporal Affinity Refiner

𝒯𝑧

×

Linear

Text Embedding Noisy latent Attention map

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

…

𝒵𝑟𝑒𝑓

× Multiplication

Motion Module

Spatial Block CTGM Motion Module

Rearrange

- Figure 3: The overall architecture of our method. FancyVideo is a T+I2V model that concatenates noise latent, mask indicator, and image indicator as input. We insert our Cross-frame Textual Guidance Module (CTGM) into each spatial block. CTGM consists of three components: Temporal Information Injector, Temporal Affinity Refiner, and Temporal Feature Booster (see supplementary materials). These components are inserted at the beginning, middle, and end of cross-attention, respectively.

###### Cross-frame Textual Guidance Module

CTGM advances the existing text control method through two sub-modules: Temporal Information Injector (TII) and Temporal Affinity Refiner (TAR) as depicted in Fig. 3(III). Before engaging in cross-attention, TII initially extracts temporal latent feature Zt and then incorporates temporal information into text embedding Trep based on Zt, obtaining cross-frame textual condition Tz. Subsequently, TAR refines the affinity between Zt and Tz along the time axis, enhancing the temporal coherence of textual guidance. The computation process of the CTGM can be formalized as:

Zt,Tz = TII(Z,Trep), (3) Zref = Softmax(

TAR(WqZt,WkTz) √dk

Wv(Tz), (4)

where Wq, Wk, and Wv represent the linear layers for query, key, and value in original cross-attention, respectively. The

hyper-parameter dk is acquired from the query dimensions. TII(·,·) and TAR(·) denotes the functions of TII and TAR. In

the end, we get refined noisy latent feature Zref. A detailed description of these three modules is provided as follows.

Temporal Information Injector. In previous work [Guo et al., 2023b; Girdhar et al., 2023], the text embedding Trep is repeated equally f times, resulting in Trep ∈ Rf×n×c, n denoting the length of the embedding vector. We inject temporal information into the embedding before performing

spatial cross-attention, thereby enabling distinct focal points on the text within different frames. In Temporal Information Injector (TII), we initially reshape the noisy latent Z from Rf×h×w×c to R(hw)×f×c and apply temporal self-attention to acquire Zt. Then, we conduct spatial cross-attention, using the repeated text embedding Trep as queries and the noisy latent Zt ∈ Rf×(hw)×c as both keys and values, resulting in the text embedding Tz with frame-specific temporal information. The formalization of the TII module can be expressed as follows:

Zt,Tz = TII(Z,Trep)

(5)

= SelfAttnt(Z), CrossAttns(SelfAttnt(Z),Trep)

where SelfAttnt denotes temporal self-attention and CrossAttns denotes spatial cross-attention. Through TII, we obtain the noisy latent Zt with temporal information and the latent-aligned text embedding Tz.

Temporal Affinity Refiner. To dynamically allocate attention to text embedding across different frames, we design the Temporal Affinity Refiner (TAR) to refine the attention map of spatial cross-attention. In spatial cross-attention, the noisy latent serves as the query, while the text embedding serves as both the key and value. The attention map A ∈ Rf×(hw)×n, compute as A = (WqZt)(WkTz)T/√dk, reflects the affinity between the text and patches. Then, TAR applies temporal

self-attention to the attention map A ∈ R(hw)×f×n, obtaining the refined attention map Aref, which can be represented as:

Aref = TAR(A) = SelfAttnt(A) (6)

With the TAR, Aref establishes a more logical temporal connection in the affinity matrix. It can perform more dynamic action while ensuring no additional video distortion occurs. Finally, the cross-attention process is completed with the refined attention map as Zref = Softmax(Aref)(WvTz).

##### 4 Experiments

In the quantitative experiments, FancyVideo utilizes the T2I base model to generate images as the first frame. In the qualitative experiments, for aesthetic purposes and to remove watermarks, an external model is used to generate a beautiful first frame.

###### 4.1 Qualitative Evaluation

We choose AnimateDiff [Guo et al., 2023b], DynamiCrafter [Xing et al., 2023], and two commercialized products, Pika [PikaLabs, 2024] and Gen2 [Runway, 2024], for a composite qualitative analysis. It is worth noting that in the quantitative experiments, the first frame of FancyVideo is generated by SDXL to achieve a more aesthetically pleasing result and to minimize the appearance of watermark (although subsequent frames may still exhibit it).

As shown in Fig. 4, our approach exhibits superior performance, outperforming previous methods regarding temporal consistency and motion richness. In contrast, AnimateDiff, DynamiCrafter, and Gen2 generate videos with less motion. Pika struggles to produce object-consistent and high-quality video frames. Remarkably, our method can accurately understand the motion instructions in the text prompt (e.g., ”A teddy bear walking ... beautiful sunset.” and ”A teddy bear running ... City.” case).

###### 4.2 Quantitative Evaluation

For a comprehensive comparison with the SOTA methods, we adopt three popular benchmarks (e.g., EvalCrafter [Liu et al., 2023], UCF-101 [Soomro et al., 2012], and MSRVTT [Xu et al., 2016] ) and human evaluation to evaluate the quality of video generation. Among them, EvalCrafter is a relatively comprehensive benchmark for video generation currently. UCF-101 and MSR-VTT are benchmarks commonly used in previous methods [Girdhar et al., 2023; Zhang et al., 2023]. Meanwhile, human evaluation can compensate for the inaccuracies in existing text-conditioned video generation evaluation systems.

EvalCrafter Benchmark. EvalCrafter [Liu et al., 2023] quantitatively evaluates the quality of text-to-video generation from four aspects (including Video Quality, Text-video Alignment, Motion Quality, and Temporal Consistency). Each dimension contains multiple subcategories of indicators shown in the Table. 1. As discussed in community [Liu and Cun, 2024], the authors acknowledge that the original manner of calculating the comprehensive metric was inappropriate. For a more intuitive comparison, we introduce a com-

prehensive metric for every aspect by considering each subindicators numerical scale and positive-negative attributes.

In detail, we compare the performance of the previous video generation SOTA methods (e.g., Pika [PikaLabs, 2024], Gen2 [Runway, 2024], Show-1 [Zhang et al., 2023], Lumiere [Bar-Tal et al., 2024], DynamiCrafter [Xing et al., 2023], and AnimateDiff [Guo et al., 2023b]) and exhibit in Table. 1. Our method demonstrates outstanding performance beyond existing methods at the Video Quality and Text-video Alignment aspect. Although Show-1 has the best Motion Quality (81.56), its Video Quality is poor (only 85.08). That indicates that it cannot generate high-quality videos with reasonable motion. However, our method has the second highest Motion Quality (72.99) and the best Video Quality (177.72), achieving the trade-off between quality and motion. The above results indicate the superiority of FancyVideo and its ability to generate temporal-consistent and motion-accurate video.

UCF-101 & MSR-VTT. Following the prior work [Zhang et al., 2023], we evaluate the zero-shot generation performance on UCF-101 [Soomro et al., 2012] and MSR-VTT [Xu et al., 2016] as shown in Table. 2. We use Fr´echet Video Distance (FVD) [Unterthiner et al., 2019], Inception Score (IS) [Wu et al., 2021], Fr´echet Inception Distance (FID) [Heusel et al., 2017], and CLIP similarity (CLIPSIM) as evaluation metrics and compared some current SOTA methods. FancyVideo achieves competitive results, particularly excelling in IS and CLIPSIM with scores of 43.66 and 0.3076, respectively. Besides, previous studies [Ho et al., 2022; Girdhar et al., 2023; Wu et al., 2023b] have pointed out that these metrics do not accurately reflect human perception and are affected by the gap between the distribution of training and test data and the image’s low-level detail.

Human Evaluation. Inspired by EvalCrafter [Liu et al.,

- 2023], we introduce a multi-candidate ranking protocol with four aspects: video quality, text-video alignment, motion quality, and temporal consistency. In this protocol, participants rank the results of multiple candidate models for each aspect. Each candidate model receives a score based on its ranking. For instance, if there are N candidate models ranked by video quality, the first model gets N −1 points, the second gets N − 2 points, and so on, with the last model receiving 0 points. Adhering to this protocol, we selected 108 samples from the EvalCrafter validation set and gathered judgments from 100 individuals. As depicted in Fig. 5, our method significantly outperforms text-to-video conversion methods, including AnimateDiff [Guo et al., 2023b], Pika [PikaLabs,
- 2024], and Gen2 [Runway, 2024], across all four aspects. FancyVideo demonstrates exceptional motion quality while preserving superior text-video consistency. Additionally, we conducted a similar comparison of four image-to-video methods, including DynamiCrafter [Xing et al., 2023], Pika, and Gen2, as shown in Fig. 5.

###### 4.3 Ablation Studies

In this section, we conduct extensive experiments and exhibit detailed visual comparisons on the EvalCrafter benchmark [Liu et al., 2023] to thoroughly explore the effect of critical designs in CTGM. The ablation study includes two

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

DynamiCrafterDynamiCrafterAnimateDiffAnimateDiffGen2Gen2OursOursPikaPika

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

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

“A dog swimming.”

“A teddy bear running in New York City.”

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

“A teddy bear walking down the street during a beautiful sunset.”

“Waves crashing against a lone lighthouse, ominous lighting.”

- Figure 4: Qualititive analysis. We compare the video generation results from AnimateDiff [Guo et al., 2023b], DynamiCrafter [Xing et al., 2023], Pika [PikaLabs, 2024], Gen-2 [Runway, 2024], and our FancyVideo.

|Dimensions Metrics<br><br>|Pika Gen2 Show-1 Lumiere DynamiCrafter AnimateDiff<br><br>|FancyVideo|
|---|---|---|
|VQAA(↑) Video VQAT(↑)<br><br>Quality IS(↑) Comprehensive(↑)|59.09 59.44 23.19 40.06 74.56 65.94 64.96 76.51 44.24 32.93 59.48 52.02 14.81 14.53 17.65 17.64 18.37 16.54 138.86 150.48 85.08 90.63 152.41 134.50<br><br>|85.78 74.56 17.38 177.72<br><br>|
|CLIP-Score(↑) BLIP-BLEU(↑) SD-Score(↑)<br><br>Text-Video Detection-Score(↑) Alignment Color-Score(↑)<br><br>Count-Score(↑) OCR Score(↓) Celebrity ID Score(↓) Comprehensive(↑)<br><br>|20.46 20.53 20.66 20.36 20.80 19.70<br>21.14 22.24 23.24 22.54 20.93 20.67 68.57 68.58 68.42 67.93 67.87 66.13 58.99 64.05 58.63 50.01 64.04 51.19 34.35 37.56 48.55 38.72 45.65 42.39 51.46 53.31 44.31 44.18 53.53 22.40 84.31 75.00 58.97 71.32 60.29 45.21 45.31 41.25 37.93 44.56 26.35 42.26<br><br><br>325.35 350.02 366.91 327.86 386.18 335.01<br><br>|20.85 21.33 68.14 66.66 51.09 59.19 64.85 25.76 396.65<br><br>|
|Action Score(↑) Motion Motion AC-Score(→) Quality Flow-Score(→)<br><br>Comprehensive(↑)<br><br>|71.81 62.53 81.56 72.12 72.22 61.94 44 44 50 42 46 32 0.50 0.70 2.07 6.99 0.96 2.403<br><br>71.81 62.53 81.56 72.12 72.22 61.94<br><br>|72.99 52 1.7413 72.99|
|CLIP-Temp(↑) Temporal Warping Error(↓) Consistency Face Consistency(↑) Comprehensive(↑)|99.97 99.94 99.77 99.74 99.75 99.85 0.0006 0.0008 0.0067 0.0162 0.0054 0.0177 99.62 99.06 99.32 98.94 99.34 99.63 199.59 199.00 199.09 198.68 199.09 199.48<br><br>|99.84 0.0051 99.31 199.15|

- Table 1: Quantitative evaluation on the EvalCrafter. The best and second performing metrics are highlighted in bold and underline. Comprehensive denotes the composite metrics for these dimensions.

Method Data

UCF-101 MSR-VTT FVD(↓) IS(↑) FID(↓) FVD(↓) CLIPSIM (↑) Emu Video 34M 606.20 42.70 - - -

AnimateDiff 10M 584.85 37.01 61.24 628.57 0.2881

DynamiCrafter 10M 404.50 41.97 32.35 219.31 0.2659 Show-1 10M 394.46 35.42 - 538.00 0.3072 Lumiere 10M 332.49 37.54 - 550.00 0.2939

FancyVideo 10M 412.64 43.66 47.01 333.52 0.3076

- Table 2: Quantitative evaluation on the UCF-101 [Soomro et al., 2012] and MSR-VTT [Xu et al., 2016] .The best and second performing metrics are highlighted in bold and underline respectively.

|TAR TII<br><br>|Video Text-Video Motion Temporal Quality Alignment Quality Consistency|
|---|---|
|✓<br><br>✓ ✓ ✓|163.15 361.92 66.99 198.83<br><br>172.44 379.40 71.24 199.08<br><br>173.82 380.24 71.84 199.04<br><br><br>177.72 396.65 72.99 199.15|

Table 3: Ablation studies on TAR and TII, where TFB is used by default in ablation experiments.

key modules (TII and TAR), each enhancing video quality As shown in Table 3, TAR significantly improves both metrics, highlighting the importance of temporal attention refinement. Adding TII further enhances performance by refining latent features and enabling frame-level text control.

Text-to-Video AnimateDiff Pika Gen-2 Ours

300

250

##### 5 Conclusion

200

In this work, we present a novel video-generation method named FancyVideo, which optimizes common text control mechanisms (e.g., spatial cross-attention) from the cross-frame textual guidance. It improves cross-attention with a well-designed Cross-frame Textual Guidance Module (CTGM), implementing the temporal-specific textual condition guidance for video generation. A comprehensive qualitative and quantitative analysis shows it can produce more dynamic and consistent videos. This characteristic becomes more noticeable as the number of frames increases. Our method achieves state-of-the-art results on the EvalCrafter benchmark and human evaluations.

150

100

50

Video Quality

Text-video Alignment

Motion Quality

Temporal Consistency

Image-to-Video DynamiCrafter Pika Gen-2 Ours

300

250

200

150

100

50

Video Quality

Text-video Alignment

Motion Quality

Temporal Consistency

- Figure 5: Human Evaluation Comparison. FancyVideo stands out significantly compared to other text-to-video and image-to-video generators in terms of Motion Quality and Temporal Consistency.

### Supplementary Material for FancyVideo: Towards Dynamic and Consistent Video Generation via Cross-frame Textual Guidance

##### 1 Appendix Section

Overview. In this supplemental material, we provide the following items:

- • (Sec. 1) Temporal Feature Booster.
- • (Sec. 2) Experimental setup.
- • (Sec. 3) More details on evaluation metrics employed in UCF101, MSR-VTT, human evaluation and the EvalCrafter [Liu et al., 2023].
- • (Sec. 4) More applications about Personalized Video Generation, high-resolution and multi-scale Video Generation, Video Predication and Video Backtracking, among others.
- • (Sec. 5) More results on video generation, including videos with different frame rates.
- • (Sec. 6) Prompts set for human evaluation.

- 1.1 Temporal Feature Booster To further boost the temporal consistency of the feature,

we process the Zref through the Temporal Feature Booster (TFB). This allows us to establish closer temporal connections. Specifically, TFB includes a simple yet effective temporal self-attention layer to refine the noisy latent feature along the time dimension, represented as:

Z′ref = TFB(Zref) = SelfAttnt(Zref) + Zref (7) The ablation study of the Temporal Feature Booster (TFB)

is presented in Table 4.

|TFB<br><br>|Video Text-Video Motion Temporal Quality Alignment Quality Consistency|
|---|---|
|✓<br><br>|175.28 391.21 72.44 199.05 177.72 396.65 72.99 199.15|

Table 4: Ablation studies on the TFB, where both TII and TAR are used by default in experimental configurations.

- 1.2 Experimental Setup

Datasets. We utilize WebVid-10M [Bain et al., 2021] as the training data. The WebVid-10M dataset contains 10.7 million video-caption pairs, with most videos having a resolution of 336 × 596. Since every clip in the WebVid-10M has a watermark, the generated video inevitably appears watermarked.

Implementation details. FancyVideo is trained on the WebVid-10M dataset. The video clips are initially sampled with a stride of 4, followed by resizing and center-cropping to a resolution of 256 × 256. We utilize Stable-Diffusion v1.5 [Rombach et al., 2022] as the text-to-image (T2I) base model and train exclusively with the temporal attention block and

our CTGM block. For the 16-frame training, we use a batch size of 512 and train for 12,000 iterations. For training with 32 frames, 48 frames, and 64 frames, we use batch sizes of 256, 256, and 128, respectively. The training process comprises 24,000 iterations for 32 frames, 48,000 iterations for 48 frames, and 96,000 iterations for 64 frames on 64 A100 GPUs with 80G memory. At inference, the sampling strategy for video generation is DDIM [Song et al., 2020] with 50 steps. Also, we utilize the classifier-free guidance [Ho and Salimans, 2022] with a 7.5 guidance scale. Similar to AnimateDiff [Guo et al., 2023b], we swap the base model with Realistic-Vision v5.1. The evaluations on EvalCrafter [Liu et al., 2023] and all the qualitative experiments are conducted at a resolution of 512. Regarding the evaluations on UCF101 [Soomro et al., 2012] and MSR-VTT [Xu et al., 2016], following [Zhang et al., 2023], we conduct assessments on videos generated at a resolution of 256.

###### 1.3 More Details on Evaluation Metrics

###### Details of evaluation on UCF101

To calculate the Fr´echet Video Distance (FVD) [Unterthiner et al., 2019] and Inception Score(IS) [Blattmann et al., 2023b; Ren et al., 2024; Saito et al., 2020], we produce 2048 videos based on the class distribution of the UCF101 dataset. These videos are generated at a resolution of 256 pixels. Subsequently, we extract I3D embeddings from our videos. Next, we compute the FVD score by comparing the I3D embeddings of our videos with those of the UCF101 videos. For computing the Inception Score (IS), the same set of generated videos was utilized to extract C3D embeddings.

###### Details of evaluation on MSR-VTT

The MSR-VTT dataset is an open-domain video retrieval and captioning dataset with 10,000 videos, each having 20 captions. The standard splits include 6,513 training videos, 497 validation videos, and 2,990 test videos. For our experiments, we use the official test split and randomly select a text prompt for each video during evaluation. Using TorchMetrics, we compute our CLIPSIM [Wu et al., 2021] metrics with the CLIP-VIT-B/32 model. We calculate the CLIP similarity for all frames in the generated videos and report the averaged results.

###### Details of Human Evaluation

We filter out 108 prompts from the 700 prompts in EvalCrafter for human evaluation. This subset of prompts contains more detailed descriptions. The prompt-108 we utilized is presented in 1.6.

###### About EvalCrafter

As mentioned in 4.2, there are unreasonable aspects in how EvalCrafter calculates Video Quality, Text-Video Alignment, Motion Quality, and Temporal Consistency. Therefore, we

###### 1.6 A list of prompts for human evaluation

propose our calculation scheme. Specifically, we remove some neutral sub-metrics and those with significant differences in scale, and based on whether each sub-metric is positive or negative, we obtain a comprehensive score. For Video Quality, it can be represented as follows:

- • goldfish in glass
- • A peaceful cow grazing in a green field under the clear blue sky
- • A fluffy grey and white cat is lazily stretched out on a sunny window sill, enjoying a nap after a long day of lounging.
- • a horse
- • Two elephants are playing on the beach and enjoying a delicious beef stroganoff meal.
- • A slithering snake moves through the lush green grass
- • A cute and chubby giant panda is enjoying a bamboo meal in a lush forest. The panda is relaxed and content as it eats, and occasionally stops to scratch its ear with its paw.
- • Pikachu snowboarding
- • a dog wearing vr goggles on a boat
- • In an African savanna, a majestic lion is prancing behind a small timid rabbit. The rabbit tried to run away, but the lion catches up easily.
- • A photo of a Corgi dog riding a bike in Times Square. It is wearing sunglasses and a beach hat.
- • In the lush forest, a tiger is wandering around with a vigilant gaze while the birds chirp and monkeys play.
- • A family of four fluffy, blue penguins waddled along the icy shore.
- • Two white swans gracefully swam in the serene lake
- • A bear rummages through a dumpster, searching for food scraps.
- • light wind, feathers moving, she moves her gaze, 4k
- • fashion portrait shoot of a girl in colorful glasses, a breeze moves her hair
- • Two birds flew around a person, in the style of Sci-Fi
- • flying superman, hand moves forward
- • Batman turns his head from right to left
- • Iron Man is walking towards the camera in the rain at night, with a lot of fog behind him. Science fiction movie, close-up
- • Bruce Lee shout like a lion ,wild fighter
- • A woman is walking her dog on the beach at sunset.
- • Valkyrie riding flying horses through the clouds
- • A surfer paddles out into the ocean, scanning the waves for the perfect ride.
- • A man cruises through the city on a motorcycle, feeling the adrenaline rush
- • A musician strums his guitar, serenading the moonlit night
- • Leaves falling in autumn forest
- • Thunderstorm at night

V ideo Quality = V QAA + V QAT + IS (8) As for Text-Video Alignment, it can be represented as:

Text V ideo Alignment = CLIP Score + SD Score

+ BLIP BLEU

+ Count Score + Color Score

+ Detection Score

+ (100 − OCR Score)

+ (100 − Celebrity ID Score)

As for Motion Quality, we do not consider neutral metrics when calculating:

###### Motion Quality = Action Score (9)

As for Temporal Consistency, we neglect the warping error, which has a scale that differs significantly from other metrics:

Temporal Consistency = CLIP Temp+Face Consistency

(10) Temporal Consistency = CLIP Temp

+ Face Consistency

###### 1.4 More Applications

Given the flexibility of our method in swapping T2I base models, we conduct experiments similar to prior work [Guo et al., 2023b], with different base models. Fig. 7 displays the results of our method using models downloaded from the Civitai[Civitai, 2024] community.

Due to the specificity of our input, we investigate the potential of our model to perform additional functionalities by modifying the mask indicator and image indicator. This encompasses tasks such as Video Prediction and Video Extending, as depicted in Fig. 8 and Fig. 9.

Due to the flexibility of our framework, we can easily combine our method with text-to-image super-resolution modules [Cheng et al., 2024; Xu et al., 2025; Wang et al., 2025b; Wang et al., 2022]. We conduct experiments to achieve multiscale and high-resolution video generations, as depicted in Fig. 11.

###### 1.5 More Generation Results

We further showcase additional results of the FancyVideo generation, including videos with 16 frames, 32 frames, 48 frames, and 64 frames. As shown in Fig. 12, our method effectively maintains consistency while also addressing motion dynamics.

Temporal Consistency

###### Video Quality

###### Text-Video Alignment

###### Motion Quality

200

500

200

500

400

500

80

500

450

450

450

450

EvalCrafterComprehensive(↑)

EvalCrafterComprehensive(↑)

EvalCrafterComprehensive(↑)

EvalCrafterComprehensive(↑)

180

70

198

400

400

360

400

400

160

HumanEvaluate

HumanEvaluate

HumanEvaluate

HumanEvaluate

350

350

350

350

60

196

300

300

320

300

300

140

250

250

50

250

250

120

200

280

200

200

194

200

40

150

150

150

150

100

100

240

100

100

192

100

30

80

50

50

50

50

60

0

200

0

20

0

190

0

16 32 48 64

16 32 48 64

16 32 48 64

16 32 48 64

Frame Number

Frame Number

Frame Number

Frame Number

AnimateDiff HE Baseline HE Ours HE AnimateDiff EC Baseline EC Ours EC

- Figure 6: Evaluation results of varying-frames video generation. EC denotes the comprehensive metric on EvalCrafter benchmark and HE indicates Human Evaluation.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

RealCartoon3DToonyou

“A dog wearing a Superhero outfit with red cape flying

“A bear wearing sunglasses and hosting a talk show.”

through the sky.”

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

“A woman is walking her dog on the beach at sunset.” “A fashion girl wears a glasses.”

Figure 7: Qualitative experiments. We demonstrate the ability to generate personalized videos by using various T2I-based models.

- • A snow avalanche crashed down a mountain peak, causing destruction and mayhem
- • A thick fog covers a street, making it nearly impossible to see. Cars headlights pierce through the mist as they slowly make their way down the road.
- • The flowing water sparkled under the golden sunrise in a peaceful mountain river.
- • A warm golden sunset on the beach, with waves gently lapping the shore.
- • Mount Fuji
- • A beautiful leather handbag caught my eye in the store window. It had a classic shape and was a rich cognac color. The material was soft and supple. The golden text label on the front read ’Michael Kors’.
- • balloons flying in the air
- • a motorcycle race through the city streets at night
- • A silver metal train with blue and red stripes, speeding through a mountainous landscape.
- • hot ramen
- • Juicy and sweet mangoes lying in a woven basket
- • A delicious hamburger with juicy beef patty, crispy lettuce and melted cheese.

- • In Marvel movie style, supercute siamese cat as sushi chef
- • With the style of Egyptian tomp hieroglyphics, A colossal gorilla in a force field armor defends a space colony.
- • a moose with the style of Hokusai
- • a cartoon pig playing his guitar, Andrew Warhol style
- • A cat watching the starry night by Vincent Van Gogh, Highly Detailed, 2K with the style of emoji
- • impressionist style, a yellow rubber duck floating on the wave on the sunset
- • A Egyptian tomp hieroglyphics painting ofA regal lion, decked out in a jeweled crown, surveys his kingdom.
- • Macro len style, A tiny mouse in a dainty dress holds a parasol to shield from the sun.
- • A young woman with blonde hair, blue eyes, and a prominent nose stands at a bus stop in a red coat, checking her phone. in the style of Anime, anime style
- • pikachu jedi, film realistic, red sword in renaissance style style
- • abstract cubism style, Freckles dot the girl’s cheeks as she grins playfully

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

“A photo of a Corgi dog riding a bike in Times Square. It is wearing sunglasses and a beach hat.”

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

|[Figure 191]|
|---|

“Valkyrie riding flying horses through the clouds.”

- Figure 8: Qualitative experiments. Under the FancyVideo framework, we train the Video Interpolation model. The red border indicates the first four frames of the original video, and we inserted three frames between every two original frames.

- • Howard Hodgkin style, A couple walks hand in hand along a beach, watching the sunset as they talk about their future together.
- • A horse sitting on an astronaut’s shoulders. in Andrew Warhol style
- • With the style of Howard Hodgkin, a woman with sunglasses and red hair
- • The old man the boat. in watercolor style
- • One morning I chased an elephant in my pajamas, Disney movie style
- • A rainbow arched across the sky, adding a burst of color to the green meadow. in Egyptian tomp hieroglyphics style
- • In Roy Lichtenstein style, In the video, a serene waterfall cascades down a rocky terrain. The water flows gently, creating a peaceful ambiance.
- • The night is dark and quiet. Only the dim light of streetlamps illuminates the deserted street. The camera slowly pans across the empty road. with the style of da Vinci
- • New York Skyline with ’Hello World’ written with fireworks on the sky. in anime style
- • A car on the left of a bus., oil painting style
- • traditional Chinese painting style, a pickup truck at the beach at sunrise
- • a sword, Disney movie style
- • a statue with the style of van gogh
- • orange and white cat., slow motion
- • slow motion, A brown bird and a blue bear.
- • Two elephants are playing on the beach and enjoying a delicious beef stroganoff meal., camera rotate anticlockwise
- • camera pan from right to left, A trio of powerful grizzly bears fishes for salmon in the rushing waters of Alaska
- • a Triceratops charging down a hill, camera pan from left to right
- • hand-held camera, A real life photography of super mario, 8k Ultra HD.
- • drove viewpoint, a man wearing sunglasses and business suit

- • a girl with long curly blonde hair and sunglasses, camera pan from left to right
- • close-up shot, high detailed, a girl with long curly blonde hair and sunglasses
- • an old man with a long grey beard and green eyes, camera rotate anticlockwise
- • camera pan from left to right, a smiling man
- • drove viewpoint, fireworks above the Parthenon
- • zoom in, A serene river flowing gently under a rustic wooden bridge.
- • large motion, a flag with a dinosaur on it
- • still camera, an F1 race car
- • hand-held camera, Three-quarters front view of a blue 1977 Corvette coming around a curve in a mountain road and looking over a green valley on a cloudy day.
- • drove viewpoint, wine bottles
- • a paranoid android freaking out and jumping into the air because it is surrounded by colorful Easter eggs, camera rotate anticlockwise
- • A traveler explores a scenic trail on the back of a sturdy mule, taking in the breathtaking views of the mountains.
- • A farmer drives a tractor through a vast field, tending to the crops with care and expertise.
- • A violinist moves her bow in a large motion sweep, creating a beautiful melody.
- • A swimmer dives into the water with a large motion splash, beginning a race.
- • A drummer hits the cymbals with a large motion crash, punctuating the music.
- • Slow motion raindrops fall gently from the sky, creating ripples in a puddle.
- • Slow motion leaves fall from a tree, swirling through the air.
- • Slow motion lightning illuminates the dark sky, followed by the rumble of thunder.
- • Slow motion bubbles rise to the surface of a glass of champagne.
- • Slow motion smoke curls up from a burning candle.

Forward_extending_1 Forward_extending_2

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

“Iron Man is walking towards the camera in the rain at night, with a lot of fog behind him.”

Forward_extending_1 Forward_extending_2

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

“A cute happy Corgi playing in park, sunset, 4k.”

Backward_extending_2 Backward_extending_1

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

“A car moving slowly on an empty street, rainy evening, van Gogh painting.”

Backward_extending_2 Backward_extending_1

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

“A dog swimming in the river.”

- Figure 9: Qualitative experiments. Under the FancyVideo framework, we train the Video Extending models, which includes extending videos forward and extending videos backward. In the forward expansion model, the input consists of 4 frames in red border, and the output is the subsequent 4 frames. With two iterations, we extend a 4-frame video to 12 frames. In the backward expansion model, the input consists of 4 frames, and the output is the subsequent 4 frames. With two iterations, we extend a 4-frame video to 12 frames. Similarly, in the backward expansion model, the input consists of 4 frames in red border, and the output is the preceding 4 frames. We also perform two iterations.

- • Slow motion confetti falls from the sky, celebrating a victory.
- • Slow motion steam rises from a hot cup of coffee.
- • Slow motion birds soar through the sky, their wings outstretched.
- • Three dogs playfully chase each other around a park.
- • Three horses gallop across a wide open field, tails and manes flying in the wind.
- • A blue boat sailing on the water with a red flag.
- • A person riding a green motorbike with an orange helmet.
- • A green cow grazing in a field with a yellow sun.
- • A yellow cat sleeping on a green bench.
- • Brad Pitt smirks charmingly, his blue eyes sparkling with mischief.
- • Angelina Jolie’s full lips curve into a smile, her gaze intense and captivating.
- • Tom Cruise’s intense stare conveys determination, his jaw set firmly.
- • Leonardo DiCaprio’s eyes glimmer with passion, his handsome face displaying intensity.

- • Robert Downey Jr.’s smug grin conveys his character’s confidence, his eyes full of wit.
- • Johnny Depp’s face shows playfulness, his eyes twinkling with mischief.

##### References

[Bain et al., 2021] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738, 2021.

[Bar-Tal et al., 2024] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024.

- [Bi et al., 2024] Xiuli Bi, Haowei Liu, Weisheng Li, Bo Liu, and Bin Xiao. Using my artistic style? you must obtain my authorization. In European Conference on Computer Vision, pages 305–321. Springer, 2024.
- [Bi et al., 2025] Xiuli Bi, Jian Lu, Bo Liu, Xiaodong Cun, Yong Zhang, Weisheng Li, and Bin Xiao. Customttt: Motion and appearance customized video generation via test-

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

“A man wearing sunglasses and business suit.”

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

“A colossal gorilla in a force field armor defends a space colony.”

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

“The old man the boat. in watercolor style.”

Figure 10: Qualitative analysis of our approach to long video generation (64 frames). More results are shown in 1.5.

time training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1871–1879, 2025.

- [Blattmann et al., 2023a] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [Blattmann et al., 2023b] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563– 22575, 2023.

[Cao et al., 2025] Ke Cao, Jing Wang, Ao Ma, Jiasong Feng, Zhanjie Zhang, Xuanhua He, Shanyuan Liu, Bo Cheng, Dawei Leng, Yuhui Yin, et al. Relactrl: Relevance-guided efficient control for diffusion transformers. arXiv preprint arXiv:2502.14377, 2025.

[Chen et al., 2023] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023.

[Cheng et al., 2024] Jiaxiang Cheng, Pan Xie, Xin Xia, Jiashi Li, Jie Wu, Yuxi Ren, Huixia Li, Xuefeng Xiao, Min Zheng, and Lean Fu. Resadapter: Domain consistent resolution adapter for diffusion models. arXiv preprint arXiv:2403.02084, 2024.

[Civitai, 2024] Civitai. Civitai community discord server. https://civitai.com/, 2024.

[De Luigi et al., 2023] Luca De Luigi, Adriano Cardace, Riccardo Spezialetti, Pierluigi Zama Ramirez, Samuele Salti, and Luigi Di Stefano. Deep learning on implicit neural representations of shapes. arXiv preprint arXiv:2302.05438, 2023.

[Girdhar et al., 2023] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.

- [Guo et al., 2023a] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Chongyang Ma, Weiming Hu, Zhengjun Zha, Haibin Huang, Pengfei Wan, et al. I2vadapter: A general image-to-video adapter for video diffusion models. arXiv preprint arXiv:2312.16693, 2023.
- [Guo et al., 2023b] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

[Gur et al., 2020] Shir Gur, Sagie Benaim, and Lior Wolf. Hierarchical patch vae-gan: Generating diverse videos from a single sample. Advances in Neural Information Processing Systems, 33:16761–16772, 2020.

[He et al., 2025] Runze He, Bo Cheng, Yuhang Ma, Qingxiang Jia, Shanyuan Liu, Ao Ma, Xiaoyu Wu, Liebucha Wu, Dawei Leng, and Yuhui Yin. Plangen: Towards unified layout planning and image generation in auto-regressive vision language models. arXiv preprint arXiv:2503.10127, 2025.

[Heusel et al., 2017] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[Ho and Salimans, 2022] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[Ho et al., 2020] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

#### ×1024768

“With the style of Howard Hodgkin, a woman with sunglasses and red hair.”

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

“The old man the boat. in watercolor style.”

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

#### ××102410241024768

“A dog swimming in the river.”

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

“A cute and chubby giant panda is enjoying a bamboo meal in a lush forest.”

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

“A yellow rubber duck floating on the wave on the sunset.”

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

“A panda standing on a surfboard in the ocean in sunset.”

Figure 11: Qualitative experiments. By switching different base models, we demonstrate experimental results at multiple pixel scales.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

64frames16frames32frames48frames

“A polar bear is playing bass guitar in snow.” “A confused grizzly bear in calculus class.”

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

“A golden retriever has a picnic on a beautiful tropical beach at sunset.”

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

“A fat rabbit wearing a purple robe walking through a fantasy landscape.”

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

“A warm golden sunset on the beach.”

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

“A man cruises through the city on a motorcycle.”

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

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

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

“Goldfish in glass.”

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

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

[Figure 381]

“Two white swans gracefully swam in the serene lake.”

Figure 12: Qualitative experiments. We demonstrate the experimental results of FancyVideo with 16, 32, 48, and 64 frames.

[Ho et al., 2022] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

[Jiang et al., 2023] Zeyinzi Jiang, Chaojie Mao, Ziyuan Huang, Ao Ma, Yiliang Lv, Yujun Shen, Deli Zhao, and Jingren Zhou. Res-tuning: A flexible and efficient tuning paradigm via unbinding tuner from backbone. Advances in Neural Information Processing Systems, 36:42689–42716, 2023.

[Kingma and Welling, 2013] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[Lin et al., 2024] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5404–5411, 2024.

[Ling et al., 2025] Run Ling, Wenji Wang, Yuting Liu, Guibing Guo, Linying Jiang, and Xingwei Wang. Ragar: Retrieval augment personalized image generation guided by recommendation. arXiv preprint arXiv:2505.01657, 2025.

[Liu and Cun, 2024] Yaofang Liu and Xiaodong Cun. evalcrafter github project. https://github.com/evalcrafter/ evalcrafter, 2024.

[Liu et al., 2023] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023.

[Liu et al., 2025] Shanyuan Liu, Bo Cheng, Yuhang Ma, Liebucha Wu, Ao Ma, Xiaoyu Wu, Dawei Leng, and Yuhui Yin. Bridge diffusion model: Bridge chinese text-to-image diffusion model with english communities. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 5541–5549, 2025.

[Lu et al., 2025] Shuo Lu, Yanyin Chen, Wei Feng, Jiahao Fan, Fengheng Li, Zheng Zhang, Jingjing Lv, Junjie Shen, Ching Law, and Jian Liang. Uni-layout: Integrating human feedback in unified layout generation and evaluation. arXiv preprint arXiv:2508.02374, 2025.

[Luo et al., 2023] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jinren Zhou, and Tieniu Tan. Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320, 2023.

- [Ma et al., 2024] Yuhang Ma, Shanyuan Liu, Ao Ma, Xiaoyu Wu, Dawei Leng, and Yuhui Yin. Hico: Hierarchical controllable diffusion model for layout-to-image generation. Advances in Neural Information Processing Systems, 37:128886–128910, 2024.
- [Ma et al., 2025] Yuhang Ma, Bo Cheng, Shanyuan Liu, Ao Ma, Xiaoyu Wu, Liebucha Wu, Dawei Leng, and Yuhui

Yin. Nami: Efficient image generation via progressive rectified flow transformers. arXiv preprint arXiv:2503.09242, 2025.

[Menapace et al., 2024] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. arXiv preprint arXiv:2402.14797, 2024.

[Munoz et al., 2021] Andres Munoz, Mohammadreza Zolfaghari, Max Argus, and Thomas Brox. Temporal shift gan for large scale video generation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3179–3188, 2021.

[PikaLabs, 2024] PikaLabs. Pika lab discord server. https: //www.pika.art/, 2024.

[Ren et al., 2024] Weiming Ren, Harry Yang, Ge Zhang, Cong Wei, Xinrun Du, Stephen Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-tovideo generation. arXiv preprint arXiv:2402.04324, 2024.

[Rombach et al., 2022] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[Runway, 2024] Runway. Gen2 discord server. https:// research.runwayml.com/gen2/, 2024.

[Saito et al., 2020] Masaki Saito, Shunta Saito, Masanori Koyama, and Sosuke Kobayashi. Train sparsely, generate densely: Memory-efficient unsupervised training of highresolution temporal gan. International Journal of Computer Vision, 128(10):2586–2606, 2020.

[Salimans and Ho, 2022] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

[Shao et al., 2025] Yihua Shao, Haojin He, Sijie Li, Siyu Chen, Xinwei Long, Fanhu Zeng, Yuxuan Fan, Muyang Zhang, Ziyang Yan, Ao Ma, et al. Eventvad: Trainingfree event-aware video anomaly detection. arXiv preprint arXiv:2504.13092, 2025.

[Sohl-Dickstein et al., 2015] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.

[Song et al., 2020] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[Soomro et al., 2012] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. A dataset of 101 human action classes from videos in the wild. Center for Research in Computer Vision, 2(11):1–7, 2012.

[Teed and Deng, 2020] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In

Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020.

[Unterthiner et al., 2019] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019.

- [Wang et al., 2019] Tsun-Hsuan Wang, Yen-Chi Cheng, Chieh Hubert Lin, Hwann-Tzong Chen, and Min Sun. Point-to-point video generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10491–10500, 2019.
- [Wang et al., 2020] Yaohui Wang, Piotr Bilinski, Francois Bremond, and Antitza Dantcheva. Imaginator: Conditional spatio-temporal gan for video generation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1160–1169, 2020.

[Wang et al., 2022] Yun Wang, Longguang Wang, Hanyun Wang, and Yulan Guo. Spnet: Learning stereo matching with slanted plane aggregation. IEEE Robotics and Automation Letters, 7(3):6258–6265, 2022.

- [Wang et al., 2024] Jing Wang, Ao Ma, Jiasong Feng, Dawei Leng, Yuhui Yin, and Xiaodan Liang. Qihoo-t2x: An efficiency-focused diffusion transformer via proxy tokens for text-to-any-task. arXiv e-prints, pages arXiv–2409, 2024.
- [Wang et al., 2025a] Jing Wang, Ao Ma, Ke Cao, Jun Zheng, Zhanjie Zhang, Jiasong Feng, Shanyuan Liu, Yuhang Ma, Bo Cheng, Dawei Leng, et al. Wisa: World simulator assistant for physics-aware text-to-video generation. arXiv preprint arXiv:2503.08153, 2025.

[Wang et al., 2025b] Yun Wang, Longguang Wang, Chenghao Zhang, Yongjian Zhang, Zhanjie Zhang, Ao Ma, Chenyou Fan, Tin Lun Lam, and Junjie Hu. Learning robust stereo matching in the wild with selective mixture-ofexperts. arXiv preprint arXiv:2507.04631, 2025.

[Wu et al., 2021] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.

- [Wu et al., 2023a] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.
- [Wu et al., 2023b] Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, and Ziwei Liu. Freeinit: Bridging initialization gap in video diffusion models. arXiv preprint arXiv:2312.07537, 2023.

[Xing et al., 2023] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023.

[Xu et al., 2016] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016.

[Xu et al., 2025] Yexing Xu, Longguang Wang, Minglin Chen, Sheng Ao, Li Li, and Yulan Guo. Dropoutgs: Dropping out gaussians for better sparse-view rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 701–710, 2025.

[Yan et al., 2021] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.

[Zeng et al., 2023] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: High-dynamic video generation. arXiv preprint arXiv:2311.10982, 2023.

- [Zhang et al., 2023] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023.
- [Zhang et al., 2024a] David Junhao Zhang, Dongxu Li, Hung Le, Mike Zheng Shou, Caiming Xiong, and Doyen Sahoo. Moonshot: Towards controllable video generation and editing with multimodal conditions. arXiv preprint arXiv:2401.01827, 2024.

- [Zhang et al., 2024b] Zhanjie Zhang, Quanwei Zhang, Huaizhong Lin, Wei Xing, Juncheng Mo, Shuaicheng Huang, Jinheng Xie, Guangyuan Li, Junsheng Luan, Lei Zhao, et al. Towards highly realistic artistic style transfer via stable diffusion with step-aware and layer-aware prompt. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 7814–7822, 2024.
- [Zhang et al., 2024c] Zhanjie Zhang, Quanwei Zhang, Wei Xing, Guangyuan Li, Lei Zhao, Jiakai Sun, Zehua Lan, Junsheng Luan, Yiling Huang, and Huaizhong Lin. Artbank: Artistic style transfer with pre-trained diffusion model and implicit style prompt bank. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 7396–7404, 2024.

- [Zhang et al., 2025] Zhanjie Zhang, Ao Ma, Ke Cao, Jing Wang, Shanyuan Liu, Yuhang Ma, Bo Cheng, Dawei Leng, and Yuhui Yin. U-stydit: Ultra-high quality artistic style transfer using diffusion transformers. arXiv preprint arXiv:2503.08157, 2025.

