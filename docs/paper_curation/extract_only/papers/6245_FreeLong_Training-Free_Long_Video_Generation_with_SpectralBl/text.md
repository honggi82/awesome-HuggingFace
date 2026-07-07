# arXiv:2407.19918v1[cs.CV]29Jul2024

## FreeLong: Training-Free Long Video Generation with SpectralBlend Temporal Attention

Yu Lu1, Yuanzhi Liang1, Linchao Zhu2 and Yi Yang2 ReLER Lab 1University of Technology Sydney 2Zhejiang University aniki.yulu@gmail.com,liangyzh18@outlook.com zhulinchao7@gmail.com,yangyics@zju.edu.cn

### Abstract

Video diffusion models have made substantial progress in various video generation applications. However, training models for long video generation tasks require significant computational and data resources, posing a challenge to developing long video diffusion models. This paper investigates a straightforward and training-free approach to extend an existing short video diffusion model (e.g., pre-trained on 16-frame videos) for consistent long video generation (e.g., 128 frames). Our preliminary observation has found that directly applying the short video diffusion model to generate long videos can lead to severe video quality degradation. Further investigation reveals that this degradation is primarily due to the distortion of high-frequency components in long videos, characterized by a decrease in spatial high-frequency components and an increase in temporal high-frequency components. Motivated by this, we propose a novel solution named FreeLong to balance the frequency distribution of long video features during the denoising process. FreeLong blends the low-frequency components of global video features, which encapsulate the entire video sequence, with the high-frequency components of local video features that focus on shorter subsequences of frames. This approach maintains global consistency while incorporating diverse and high-quality spatiotemporal details from local videos, enhancing both the consistency and fidelity of long video generation. We evaluated FreeLong on multiple base video diffusion models and observed significant improvements. Additionally, our method supports coherent multi-prompt generation, ensuring both visual coherence and seamless transitions between scenes. Project page: https://yulu.net.cn/freelong

### 1 Introduction

Video diffusion models [1, 2, 3, 4, 5, 6, 7] trained on vast video-text datasets [8, 9] have demonstrated impressive capabilities in generating high-quality videos. Inspired by Sora [10], multiple studies [11, 12, 13] have concentrated on training these models to create longer videos using extensive, long video-text datasets [14, 15, 16, 17, 18]. However, these methods demand significant computational resources and data annotations.

A more practical approach involves adapting pre-trained short video models to generate consistent longer video sequences without retraining. Recent research [19, 20] has explored sliding window temporal attention to ensure smooth transitions between video clips in the generation of long videos. Nonetheless, these techniques often struggle to maintain global temporal consistency across extended sequences and require multiple passes of temporal attention.

Preprint. Under review.

A tiger walks in the forest, photorealistic, 4k, high definition a polar bear playing drum kit in NYC Times Square, 4k

1 7 16 7 16

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

1

LaVie 16 frames

LaVie 16 frames

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

1 64 128

1

64 128

LaVie 128 frames

LaVie 128 frames

- 1 64 128 64 128

1

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

FreeLong 128 frames

FreeLong 128 frames

A girl walked along a forest path, surrounded by tall trees A Corgi walking in the park at sunrise, oil painting style

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

1 7 16 7 16

1

VideoCrafter 16 frames

VideoCrafter 16 frames

1 64 128 64 128

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

1

VideoCrafter 128 frames

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

64 128

[Figure 35]

[Figure 36]

1 64 128

1

FreeLong 128 frames

FreeLong 128 frames

- Figure 1: Results of Short and Long Videos. The first row of each case shows 16-frame videos generated using short video diffusion models (LaVie [1] and VideoCrafter2 [2]). Directly extending these models to longer videos, like those with 128 frames, preserves temporal consistency but lacks fine spatial-temporal details. In contrast, our proposed FreeLong adapts short video diffusion models to create consistent long videos with high fidelity.

In this study, we propose a simple, training-free method to adapt existing short video diffusion models (e.g., pretrained on 16 frames) for generating consistent long videos (e.g., 128 frames). Initially, we examine the direct application of short video diffusion models for long video generation. As depicted in Figure 1, straightforwardly using a 16-frame video diffusion model to produce 128-frame sequences yields globally consistent yet low-quality results.

To delve further into these issues, we conducted a frequency analysis of the generated long videos. As shown in Figure 2 (a), the low-frequency components remain stable as the video length increases, while the high-frequency components exhibit a noticeable decline, leading to a drop in video quality. The findings indicate that although the overall temporal structure is preserved, fine-grained details suffer notably in longer sequences. Specifically, there is a decrease in high-frequency spatial components (Figure 2 (b)) and an increase in high-frequency temporal components (Figure 2 (c)). This high-frequency distortion poses a challenge in maintaining high fidelity over extended sequences. As illustrated in the middle row of each case in Figure 1, intricate textures like forest paths or sunrises become blurred and less defined, while temporal flickering and sudden changes disrupt the video’s narrative flow.

To tackle these challenges, we introduce FreeLong, a novel framework that employs SpectralBlend Temporal Attention (SpectralBlend-TA) to balance the frequency distribution of long video features in the denoising process. SpectralBlend-TA integrates global and local features via two parallel streams, enhancing the fidelity and consistency of long video generation. The global stream deals with the entire video sequence, capturing extensive dependencies and themes for narrative continuity. Meanwhile, the local stream focuses on shorter frame subsequences to retain fine details and smooth transitions, preserving high-frequency spatial and temporal information. SpectralBlend-TA combines global and local video features in the frequency domain, improving both consistency and fidelity by blending low-frequency global components with high-frequency local components. Our method is entirely training-free and allows for the easy integration of FreeLong into existing video

diffusion models by adjusting the original temporal attention of video diffusion models. Comparative experiments demonstrate significant improvements in temporal consistency and video fidelity when applying our method to generate long video sequences.

Our contributions can be summarized as follows: 1) We conduct a frequency analysis on the direct application of short video models for longer video generation and identify high-frequency distortions in the longer videos. 2) We devise a SpectralBlend Temporal Attention mechanism to merge the consistent low-frequency components of global videos with the high-fidelity highfrequency components of local videos. 3) Our training-free approach, FreeLong, outperforms existing state-of-the-art models in both temporal consistency and video fidelity.

### 2 Related Work

Text-to-Video Diffusion Models: Text-to-video (T2V) generation has progressed significantly from early variational autoencoders [21, 22] and GANs [23] to advanced diffusion-based techniques [3, 4, 24, 25, 26], marking a major leap in synthesis methods. Modern video diffusion models build on pre-trained image-to-text diffusion models [27, 28, 29], incorporating temporal transformers in the diffusion UNet to capture temporal relationships. These models achieve impressive video generation results through post-training on video-text data [14, 9, 8, 1], enhancing coherence and fidelity. However, due to computational constraints and limited dataset availability, current video diffusion models are typically trained on fixed-length short videos (e.g., 16 frames), limiting their ability to produce longer videos. In this paper, we propose extending these short video diffusion models to generate long and consistent videos without requiring any additional training videos.

Long-video Generation: Generating long videos is challenging due to temporal complexity, resource constraints, and the need for content consistency. Recent advancements focus on improving temporal coherence and visual quality using GAN-based [30, 31] and diffusion-based techniques [32, 33, 34, 35, 36]. For instance, Nuwa-XL [36] employs a parallel diffusion process, while StreamingT2V [11] uses an autoregressive approach with a short-long memory block to improve the consistency of long video sequences. Despite their effectiveness, these methods require substantial computational resources and large-scale datasets. Recent research has explored training-free adaptations using short video diffusion models for long video generation. Gen-L-Video [37] extends videos by merging overlapping sub-segments with a sliding-window method during denoising. FreeNoise [19] employs sliding-window temporal attention and a noise initialization strategy to maintain temporal consistency. However, these approaches focus on smooth transitions between video clips and fail to capture global consistency across long video sequences. This paper proposes FreeLong, a novel approach that blends global and local video features during the denoising process to enhance both global temporal consistency and visual quality in long video generation.

### 3 Observation and Analysis

When attempting to adapt short video diffusion models to generate long videos, a straightforward approach is to input a longer noise sequence into the short video models. The temporal transformer layers in the video diffusion model are not constrained by input length, making this method seemingly viable. However, our empirical study reveals significant challenges, as demonstrated in Figure 1. Generated long videos often exhibit fewer detailed textures, such as blurred forests in the background, and more irregular variations, like abrupt changes in motion. We attribute these issues to two main factors: the limitations of the temporal attention mechanism and the distortion of high-frequency components.

Attention Mechanism Analysis: The temporal attention mechanism in video diffusion models is pre-trained on fixed-length videos, which complicates its ability to generate longer videos. As shown in Figure 3 , increasing video length hinders the temporal attention’s ability to accurately capture frame-to-frame relationships. For 16-frame videos, the attention maps show a diagonal pattern, indicating high correlations with adjacent frames that preserve spatial-temporal details and motion patterns. In contrast, for 128-frame videos, the less structured attention maps suggest difficulty in focusing on relevant information across distant frames, leading to missed subtle motion patterns and over-smoothed or blurred generations.

[Figure 37]

[Figure 38]

[Figure 39]

(a) Space-Time SNR (b) Space SNR (c) Time SNR

- Figure 2: Ratio of short video SNR on high/low frequency to different long videos. Our findings reveal that: (a) When direct extend short video diffusion model to generate long videos, the SNR of high-frequency components in the space-time frequency domain degrades significantly as video length increases. (b) In the spatial frequency domain, the SNR of high-frequency components decreases even more substantially, resulting in the over-smoothing of each frame. (c) Conversely, in the temporal frequency domain, the SNR of high-frequency components increases significantly, introducing temporal flickering.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

16 Frame 64 Frame 128 Frame 16 Frame 64 Frame 128 Frame

LaVie VideoCrafter

[Figure 45]

- Figure 3: Temporal Attention Visualization. We visualize the temporal attention by average across all layers and time steps from LaVie [1] and VideoCrafter [2]. The attention maps for 16-frame videos exhibit a diagonal-like pattern, indicating a high correlation with adjacent frames, which helps preserve high-frequency details and motion patterns when generating new frames. In contrast, attention maps for longer videos are less structured, such as 128 frames, making the model struggle to identify and attend to the relevant information across distant frames. This lack of structure in the attention maps results in the distortion of high-frequency components of long videos, which results in the degradation of fine spatial-temporal details.

Frequency Analysis: To better understand the generation process of long videos, we analyzed the frequency components in videos of varying lengths using the Signal-to-Noise Ratio (SNR) as a metric. Ideally, short video diffusion models generate 16-frame videos with high quality, and robust longer videos derived from such models should exhibit consistent SNR values across all frequency components. However, Figure 2 reveals significant differences in the SNR of high/low frequency components1 between generated short and long videos. The SNR of low-frequency components remains relatively consistent for long videos (1.0 for 16 frames to 0.93 for 128 frames), suggesting that the model maintains overall structure and low-frequency details in extended sequences. However, the SNR of high-frequency components drops significantly for longer videos (1.0 for 16 frames to 0.73 for 128 frames), indicating a loss of fine details and increased distortion, leading to suboptimal visual fidelity.

Further investigation into the spatial and temporal frequency domains revealed two key findings: (1) In the spatial domain, the high-frequency components of long videos degrade significantly (0.68 for 128 frames), causing substantial degradation of spatial details in each frame and resulting in blurred

1We split the frequency components into high-frequency (ϕ ∼ (0.25π − 1.00π)) and low-frequency (ϕ ∼ (0.00π − 0.25π)) and compared the SNR of each component in long videos to the corresponding SNR in 16-frame videos.

Iterative Denoising

Attention Blocks

|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]|
|---|

Space attn Cross attn

× 𝑇

Decoder

SpetralBlend-TA

𝑍

𝑍

𝑍

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

3D FFT

[Figure 58]

[Figure 59]

=

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

*

[Figure 66]

High Pass 3D IFFT

Local-TA

| |
|---|

[Figure 67]

[Figure 68]

𝑍 ℱ

𝑍

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

𝑍′

[Figure 76]

[Figure 77]

Global-TA

[Figure 78]

3D FFT

[Figure 79]

Low Pass

Local-global ℱ Attention Decoupling

𝑍

𝑍

Spectral Blending

- Figure 4: Overview of FreeLong. FreeLong facilitates consistent and high-fidelity video generation using SpectralBlend Temporal Attention (SpectralBlend-TA). SpectralBlend-TA effectively blends low-frequency global video features with high-frequency local video features through a two-step process: local-global attention decoupling and spectral blending. Local video features are obtained by masking temporal attention to concentrate on fixed-length adjacent frames, while global temporal attention encompasses all frames. During spectral blending, 3D FFT projects features into the frequency domain, where high-frequency local components and low-frequency global components are merged. The resulting blended feature, transformed back to the time domain via IFFT, is then utilized in the subsequent block for refined video generation.

frames. (2) In the temporal domain, the high-frequency components increase with video length (1.5 for 128-frame videos), resulting in temporal flickering and incoherent video outputs.

### 4 FreeLong: Training-free Long Video Generation

Motivated by the above analysis, we propose FreeLong, a method designed to generate high-fidelity and consistent long videos using the inherent power of the diffusion model. As illustrated in Figure 4, our FreeLong uses a diffusion UNet from pre-trained short video diffusion models and introduces a SpectralBlend Temporal Attention (SpectralBlend-TA) to facilitate long video generation. The SpectralBlend-TA consists of two steps: local-global attention decoupling and spectral blending.

#### Local-global Attention Decoupling:

The temporal attention in short video models is optimized to model short frame sequences accurately, maintaining high-fidelity visual information. Conversely, the long-range temporal attention from short video models tends to maintain overall layout and and object consistency. Given these properties, we first decouple the local and global attention. The local attention matrix can be obtained as:

 

⊤ √ j

Softmax QiK

d if |i − j| ≤ α 0 otherwise,

(1)

Alocal(i,j) =



where Q and K are the query and key matrices derived from the input video feature Zin. The local attention Alocal leads to each frame i only attending to frames within a window of 2α frames. Given the local attention matrix Alocal, the local video features Zlocal can be obtained by: Zlocal = AlocalV , where V is the value matrix derived from the input video feature Zin. By restricting the temporal attention to adjacent local frames, we preserve the capabilities of short video models, thereby retaining high-fidelity visual details in local video features.

We then define the global attention matrix where each frame attends to all other frames. The global attention matrix can be computed as follows:

QiKj⊤ √

, (2)

Aglobal(i,j) = Softmax

d

Given the global attention matrix Aglobal, the global video features Zglobal can be obtained by: Zglobal = AglobalV . The global video features process the entire video sequence, ensuring narrative continuity and coherence, while capturing long-range dependencies and overarching themes.

Spectral Blending: After obtaining the global and local video features, a frequency filter is used to blend the low-frequency components of the global video latent Zglobal with the high-frequency components of the local video latent Zlocal, resulting in a new video latent Z′. This fused latent retains the global coherence and structure provided by Zglobal, while benefiting from the enhanced high-frequency details introduced by Zlocal. The process is described by:

FzLglobal = FFT3D(Zglobal) ⊙ P, (3) FzHlocal = FFT3D(Zlocal) ⊙ (1 − P), (4) Z′ = IFFT3D(FzLglobal + FzHlocal) (5)

where FFT3D is the Fast Fourier Transformation operated on both spatial and temporal dimensions, IFFT3D is the Inverse Fast Fourier Transformation that maps back the blended representation Z′ from the frequency domain, and P ∈ R4×N×h×w is the spatial-temporal Low Pass Filter (LPF), which is a tensor of the same shape as the latent. The final fused video feature Z′ serves as the input to our subsequent video generation module.

The rationale behind using low-frequency components from the global video features and highfrequency components from the local video features stems from our analysis. The global features provide a stable, coherent structure, preserving the overall layout and object consistency throughout the video. This is crucial for maintaining temporal consistency in long videos. On the other hand, local features retain high-fidelity details, which are essential for capturing fine textures and intricate motion patterns that tend to degrade in long sequences. By blending these components in the frequency domain, we harness the strengths of both global consistency and local detail preservation, addressing the issues of blurred frames and temporal flickering observed in our analysis.

Recent studies [38, 39] indicate that latent diffusion models [27] generate varying levels of visual content at different stages of the denoising process: scene layout and object shapes in the early steps, and fine details in the later steps. We propose fusing global and local video features in the early τ steps of the denoising process and using local video features in the remaining steps. This fusion ensures that the overall layout and object appearance of the generated long video follow the global features, thereby maintaining temporal consistency in the generated videos.

### 5 Experiments

#### 5.1 Implementation Details

Baseline Models: To evaluate the effectiveness and generalization of our proposed method, we apply FreeLong on two publicly available diffusion-based text-to-video models, LaVie [1] and VideoCrafter [2]. These models are trained on short videos with fixed length (i.e., 16 frames), we extend them to produce long videos (i.e., 128 frames [40]). We set α = 8 for the local attention setting and set τ to 25. During inference, the parameters of the frequency filter for each model are kept the same for a fair comparison. Specifically, we use a Gaussian Low Pass Filter (GLPF) PG with a normalized spatiotemporal stop frequency of D0 = 0.25. Multi-prompt videos are generated with random noise, and FreeNoise [19] is used for single-prompt long video generation.

Test Prompts: We chose 200 prompts from VBench [41] to validate the effectiveness of our method. Evaluation Metrics: For text-to-video generation, we employed several metrics from VBench [41] to evaluate two aspects: video consistency and video fidelity. For video consistency measurement, we use two metrics: 1). Subject consistency, computed by the DINO [42] feature similarity across frames to assess whether object appearance remains consistent throughout the whole video. 2). Background consistency, calculated by CLIP [43] feature similarity across frames. For video fidelity measurement, we use three metrics: 1). Motion smoothness, which utilizes the motion priors in the video frame interpolation model AMT [44] to evaluate the smoothness of generated motions. 2). Temporal flickering, which takes static frames and computes the mean absolute difference across frames. 3). Imaging quality, calculated using the MUSIQ [45] image quality predictor trained on the SPAQ [46] dataset.

Table 1: Quantitative Comparison. “Direct sampling” and “Sliding window” indicate directly sampling 128 frames and applying temporal sliding windows based on short video generation models, respectively. Compared to these methods, our FreeLong achieves consistent long video generation with high fidelity.

Methods Sub (↑) Back (↑) Motion (↑) Flicker (↑) Imaging (↑) Inference Time (↓)

Direct sampling 88.95 93.23 92.77 91.44 64.76 1.8s Sliding window 85.80 92.83 95.79 94.00 66.57 2.6s FreeNoise [19] 92.30 95.87 96.32 94.94 67.14 2.6s

###### Ours 95.16 96.80 96.85 96.04 67.55 2.2s

|[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>Sliding-Window<br><br>Prompt: A white yacht traveling on a river and passing under the bridge. Prompt: A woman sitting by the indoor fire pit.<br><br>FreeNoise<br><br>FreeLong<br><br>Direct<br><br>Sliding-Window<br><br>FreeNoise<br><br>FreeLong<br><br>Direct<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]|
|---|

Results from LaVie

|[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>Prompt: Truck carrying the tree logs. Prompt: A man eating a donut.<br><br>Sliding-Window<br><br>FreeNoise<br><br>FreeLong<br><br>Direct<br><br>Sliding-Window<br><br>FreeNoise<br><br>FreeLong<br><br>Direct<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]|
|---|

Results from VideoCrafter

- Figure 5: Qualitative Comparison. Results from LaVie [1] and VideoCrafter [2] are presented. Direct videos exhibit consistent frames, but they appear over-smoothed. FreeNoise and the slidingwindow approach struggle to capture global consistency effectively. Our FreeLong method achieves consistent long video generation while maintaining high fidelity, preserving crucial details and textures across the entire sequence.

#### 5.2 Quantitative Comparison

We compare our FreeLong method with other training-free approaches for long video generation using diffusion models. Our comparison includes three methods: (1) Direct sampling. It directly samples 128 frames from the short video models. (2) Sliding window. It adopts temporal sliding windows [20] to process a fixed number of frames at a time. (3) FreeNoise [19]. FreeNoise introduces repeat input noise to maintain temporal coherence across long sequences.

Table 1 presents the quantitative results. Direct generation of long videos suffers from high-frequency distortion, leading to significant quality degradation. This method results in low fidelity scores, including imaging quality, temporal flickering, and motion smoothness. The sliding-window method and FreeNoise show improved video quality thanks to the fixed effective temporal attention window

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Global

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Local

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Global+Local

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Low Frequency of Global

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

High Frequency of Local

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

FreeLong

- Figure 6: Ablation Study. Global features and low-frequency components of global features ensure consistency but degrade fidelity. Local features and high-frequency local features maintain spatial-temporal details but lack temporal consistency. Directly adding global and local features degrades fidelity. Our method achieves both high fidelity and temporal consistency.

but still face challenges in maintaining consistency across long videos. Our FreeLong method achieves the highest scores across all metrics, producing consistent long videos with high fidelity. Moreover, we also examine the inference time of these methods on the NVIDIA A100. As delineated in Table 1, our approach achieves a faster speed compared to preceding methods by employing single-pass temporal attentions.

#### 5.3 Qualitative Comparison

The synthesis results of each method are shown in Figure 5. In the first row, directly sampling 128 frames through a model trained on 16 frames will bring poor quality results due to the high-frequency distortion. For example, the yacht (left) and the girl (right) have blurred and the background is not clear. As shown in second row in Figure 5, using temporal sliding windows helps generate more vivid videos, but this approach ignores long-range visual consistency, causing the subject and background to appear significantly different across frames. FreeNoise attempts to promote global consistency by repeating and shuffling the initial noise for each frame; however, it fails to maintain long-range visual consistency and suffers from content mutations. In contrast, our method, FreeLong, explicitly enforces global constraints during the denoising process, achieving temporal consistency while preserving high fidelity across frames. Results shown in Figure 5 demonstrate that FreeLong successfully renders temporal consistent longer videos, outperforming all other methods.

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

- Prompt1: Ironman running on the road, 4K, high resolution
- Prompt2: Ironman standing on the road, 4K, high resolution
- Prompt3: Ironman flying on the sky, 4K, high resolution

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

- Prompt1: A musician with a guitar performs at the edge of a mountain at morning.;
- Prompt2: The musician performs in a busy urban park, a crowd gathers, enchanted by the music.;
- Prompt3: A serene lakeside at sunset where the musician plays alone, reflecting.;
- Prompt4: Night falls, and the musician joins a lively street festival, lights and music filling the air.;

- Prompt4: The journey ends on a quiet balcony overlooking the city at night, the musician performs towards the city.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

- Prompt1: The morning sun rises, illuminating a solitary lighthouse on a rocky shore.
- Prompt2: Clouds gather, and rain starts to fall, the lighthouse stand firm under storm with heavy rain and thunder.
- Prompt3: The storm clears, the lighthouse stand under rainbow.
- Prompt4: Night falls, and the lighthouse raise light through the darkness.

- Figure 7: Results of Multi-Prompt Video Generation. Our method ensures coherent visual continuity and motion consistency across different video segments.

#### 5.4 Ablation Studies

To validate the effectiveness of each module in our FreeLong method—global video feature, local video feature, and our combined approach—we present the generated results by ablating each component.

As shown in the top part of Figure 6, videos generated solely from global video features maintain consistent content but suffer from severe fidelity degradation. Conversely, videos generated using only local video features preserve fidelity due to the fixed effective temporal attention window but fail to maintain temporal consistency, as evidenced by the changing color of the cow. Simply combining global and local video features results in fidelity degradation because the high-frequency components of the global video features degrade significantly.

In the bottom part of Figure 6, we show the videos generated by combining the low-frequency components from global video features with the high-frequency components from local video features. Our approach effectively combines the consistency of global videos with the high fidelity of local videos, achieving both high fidelity and temporal consistency.

#### 5.5 Multi-Prompt Video Generation

Our method can be seamlessly extended to multi-prompt video generation by providing different prompts for each video segment. As illustrated in Figure 7, our approach ensures coherent visual continuity and motion consistency. For instance, Ironman is shown running on the road, then standing, and finally flying into the sky, all within a consistent scene and with smooth action transitions. In the second row, we demonstrate a more complex prompt sequence describing weather and scene transitions. Our method effectively models the transition from “sunrise" to “storm with heavy rain and thunder" to the final “rainbow," maintaining consistency and capturing the fine-grained details of each prompt transition.

### 6 Conclusion

In this paper, we introduced FreeLong, a training-free method to adapt short video diffusion models for long video generation. Our research reveals that directly generating long videos from short video diffusion models results in poor quality, primarily due to high-frequency distortion. To resolve this issue, we employ the SpectralBlend Temporal Attention (SpectralBlend-TA) mechanism, which blends low-frequency global features with high-frequency local features to enhance consistency and fidelity in long videos. Our experiments demonstrate that FreeLong significantly outperforms existing models, achieving superior temporal consistency and video fidelity. Our experiments show that FreeLong significantly outperforms existing models, achieving better temporal consistency and video fidelity. FreeLong also supports coherent multi-prompt generation, offering a practical solution for high-quality long video creation without extensive retraining.

### References

- [1] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [2] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024.
- [3] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [4] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [5] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [6] Yu Lu, Linchao Zhu, Hehe Fan, and Yi Yang. Flowzero: Zero-shot text-to-video synthesis with llm-driven dynamic scene syntax. arXiv preprint arXiv:2311.15813, 2023.
- [7] Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang. Eva: Zero-shot accurate attributes and multi-object video editing. arXiv preprint arXiv:2403.16111, 2024.
- [8] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In The Twelfth International Conference on Learning Representations, 2023.
- [9] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.
- [10] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. Accessed: 2024-05-09.
- [11] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773, 2024.
- [12] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.
- [13] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Jingmin Chen, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, et al. Videotetris: Towards compositional text-to-video generation. arXiv preprint arXiv:2406.04277, 2024.

- [14] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. arXiv preprint arXiv:2402.19479, 2024.
- [15] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023.
- [16] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. arXiv preprint arXiv:2401.09414, 2024.
- [17] Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. Videodrafter: Content-consistent multiscene video generation with llm. arXiv preprint arXiv:2401.01256, 2024.
- [18] Yu Lu, Feiyue Ni, Haofan Wang, Xiaofeng Guo, Linchao Zhu, Zongxin Yang, Ruihua Song, Lele Cheng, and Yi Yang. Show me a video: A large-scale narrated video dataset for coherent story illustration. IEEE Transactions on Multimedia, 2023.
- [19] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.
- [20] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023.
- [21] Yitong Li, Martin Renqiang Min, Dinghan Shen, David E. Carlson, and Lawrence Carin. Video generation from text. CoRR, abs/1710.00421, 2017.
- [22] Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David E. Carlson, and Jianfeng Gao. Storygan: A sequential conditional GAN for story visualization. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 6329–6338. Computer Vision Foundation / IEEE, 2019.
- [23] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. CoRR, abs/1406.2661, 2014.
- [24] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. CoRR, abs/2211.11018, 2022.
- [25] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
- [26] Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, and Ziwei Liu. Freeinit: Bridging initialization gap in video diffusion models. arXiv preprint arXiv:2312.07537, 2023.
- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [28] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023.
- [29] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [30] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3626–3636, 2022.
- [31] Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei Efros, and Tero Karras. Generating long videos of dynamic scenes. Advances in Neural Information Processing Systems, 35:31769–31781, 2022.

- [32] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. Advances in Neural Information Processing Systems, 35:27953–27965, 2022.
- [33] Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. Advances in neural information processing systems, 35:23371–23385, 2022.
- [34] Hung-Yu Tseng, Qinbo Li, Changil Kim, Suhib Alsisan, Jia-Bin Huang, and Johannes Kopf. Consistent view synthesis with pose-guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16773–16783, 2023.
- [35] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [36] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023.
- [37] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023.
- [38] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [39] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22560– 22570, 2023.
- [40] Chengxuan Li, Di Huang, Zeyu Lu, Yang Xiao, Qingqi Pei, and Lei Bai. A survey on long video generation: Challenges, methods, and prospects. arXiv preprint arXiv:2403.16407, 2024.
- [41] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023.
- [42] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [44] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9801–9810, 2023.
- [45] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021.
- [46] Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, and Zhou Wang. Perceptual quality assessment of smartphone photography. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3677–3686, 2020.

