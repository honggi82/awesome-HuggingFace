## FASTERCACHE: TRAINING-FREE VIDEO DIFFUSION MODEL ACCELERATION WITH HIGH QUALITY

#### Zhengyao Lv1∗ Chenyang Si2‡ Junhao Song3 Zhenyu Yang3 Yu Qiao3 Ziwei Liu2† Kwan-Yee K. Wong1† 1The University of Hong Kong 2S-Lab, Nanyang Technological University 3Shanghai Artificial Intelligence Laboratory Code: https://github.com/Vchitect/FasterCache

ABSTRACT

# arXiv:2410.19355v2[cs.CV]12Mar2025

In this paper, we present FasterCache, a novel training-free strategy designed to accelerate the inference of video diffusion models with high-quality generation. By analyzing existing cache-based methods, we observe that directly reusing adjacent-step features degrades video quality due to the loss of subtle variations. We further perform a pioneering investigation of the acceleration potential of classifier-free guidance (CFG) and reveal significant redundancy between conditional and unconditional features within the same timestep. Capitalizing on these observations, we introduce FasterCache to substantially accelerate diffusion-based video generation. Our key contributions include a dynamic feature reuse strategy that preserves both feature distinction and temporal continuity, and CFG-Cache which optimizes the reuse of conditional and unconditional outputs to further enhance inference speed without compromising video quality. We empirically evaluate FasterCache on recent video diffusion models. Experimental results show that FasterCache can significantly accelerate video generation (e.g., 1.67× speedup on Vchitect-2.0) while keeping video quality comparable to the baseline, and consistently outperform existing methods in both inference speed and video quality.

|Original|Δ - DiT|PAB|Ours|
|---|---|---|---|

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Open-SoraOpen-Sora-PlanLatte

Lat: 192.07s Lat: 168.69s, LPIPS: 0.332 Lat: 156.73s, LPIPS: 0.152 Lat: 118.44s, LPIPS: 0.094

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|Lat: 103.76s|Lat: 86.88s, LPIPS: 0.156|Lat: 78.72s, LPIPS: 0.307|Lat: 61.68s, LPIPS: 0.062|
|---|---|---|---|

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

|Lat: 29.22s|Lat: 23.80s, LPIPS: 0.246|Lat: 22.84s, LPIPS: 0.390|Lat: 18.98s, LPIPS: 0.082|
|---|---|---|---|

(Lat denotes latency, measured on a single A100 GPU. Video synthesis configuration: 192 frames at 480P for Open-Sora, 65 frames at 512×512 for Open-Sora-Plan, and 16 frames at 512× 512 for Latte.)

Figure 1: Comparison of visual quality and inference speed with competing methods.

1 INTRODUCTION

Diffusion transformers (DiT) (Peebles & Xie, 2023) have achieved notable success in image (Chen et al., 2023; 2024b; Esser et al., 2024) and video generation (Ma et al., 2024a; Zheng et al., 2024; PKU-Yuan Lab and Tuzhan AI etc., 2024), attracting significant attention for their potential. Although iterative denoising, classifier-free guidance (CFG) (Ho & Salimans, 2022), and transformer

† Corresponding authors. ‡ Project leader. ∗The work was done during an internship at Shanghai AI Lab.

Iteration denoising Iteration denoising

Iteration denoising

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

𝒙𝒕 𝟐 𝒙𝒕 𝟏

𝒙𝒕 𝒙𝒕 𝟏

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Feature reuse

Feature reuse

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Attention Attention

Attention Attention

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

FFN FFN

FFN FFN

Figure 2: Vanilla cache-based acceleration method typically reuses features cached from previous timesteps directly for the current timestep. attention mechanisms have significantly improved the generative capabilities of diffusion models, they also lead to substantial computational costs and increased memory requirements for inference, especially for video generation which typically takes 2-5 minutes to synthesize a 6-second 480P video, limiting their practical use. This calls for the development of new techniques that require less computational cost for diffusion models (Salimans & Ho, 2022; Ma et al., 2024b; Chen et al., 2024c; Zhao et al., 2024c).

Among the recently proposed solutions, cache-based acceleration has emerged as one of the most widely adopted approaches. This approach speeds up the sampling process by reusing intermediate features across timesteps, thereby reducing redundant computations and significantly improving computational efficiency. Besides, it requires no additional training costs for inference acceleration and offers straightforward generalization to other video diffusion models. Examples include cachebased methods for U-Net based diffusion models (Ma et al., 2024b; Li et al., 2023b), residual caching in ∆-DiT (Chen et al., 2024c) for transformer based diffusion models, and hierarchical attention caching of PAB (Zhao et al., 2024c) for video generation. Despite their proven effectiveness, there exist two critical concerns: 1) Whether directly reusing intermediate features aligns with the iterative denoising mechanism, considering the inherent feature variations between timesteps. 2) Current cache-based methods focus primarily on the attention features within the transformer networks, with limited exploration of accelerating different parts of the pipeline. In this work, we aim to address these two concerns.

To thoroughly investigate the acceleration potential of DiT inference for video generation, we delve into the feature reuse process of existing cache-based methods. As shown in Fig. 2, these methods typically assume a high degree of feature similarity between adjacent timesteps in the iterative denoising process, and achieve accelerated inference by sharing features across consecutive timesteps. However, our investigation reveals that while features in the same attention module (e.g., spatial attention) appear to be nearly identical between adjacent timesteps, there exist some subtle yet discernible differences. As a result, a naive feature caching and reuse strategy often leads to degradation of details in generated videos, as shown in Fig. 3 (a).

Following this analysis, we further extend the scope of our investigation to explore potential redundancy within the classifier-free guidance (CFG). As shown in Fig. 3 (b), compared to internal network modules (e.g., spatial attention and temporal attention), CFG almost doubles the inference time due to the additional computation required for unconditional outputs. Our experiments reveal a notable difference from our earlier conclusion regarding attention modules. In CFG, the conditional and unconditional outputs at the same timestep exhibit a very high degree of similarity, suggesting significant information redundancy. In contrast, the similarity of unconditional features between adjacent timesteps is relatively weak. We further discover that the differences between the conditional and unconditional outputs are predominantly concentrated in low- to mid-frequency features during the mid-sampling phase, shifting to highfrequency features in the late-sampling phase, with these differences evolving gradually.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Original Cache-based

- (a)
- (b)

###### conditional unconditional

###### FFN TA SA CA Other

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

2%

12%

[Figure 42]

34%

unconditional 50%

conditional 50%

23%

[Figure 43]

29%

Figure 3: (a) Vanilla cache-based methods typically lead to detail loss. (b) Time overhead proportions of different components in video models.

Based on the above insights, we propose a novel strategy, termed FasterCache, to accelerate the inference of video diffusion models while ensuring high-quality generation and remaining trainingfree. Specifically, we first introduce a dynamic feature reuse strategy for attention modules which dynamically adjusts the reused features across different timesteps, ensuring both distinction and

continuity of features between adjacent timesteps are maintained. This strategy preserves the subtle variations essential for the iterative denoising process while ensuring temporal consistency, resulting in accelerated inference with minimal loss of details in the generated videos. Furthermore, we introduce CFG-Cache, an innovative technique that stores the residuals between conditional and unconditional outputs, dynamically enhancing their high-frequency and low-frequency components before reuse. This significantly accelerates inference while preserving details in generated videos.

We evaluate our FasterCache on various video diffusion models, including Open-Sora 1.2 (Zheng et al., 2024), Open-Sora-Plan (PKU-Yuan Lab and Tuzhan AI etc., 2024), Latte (Ma et al., 2024a), CogVideoX (Yang et al., 2024), and Vchitect-2.0 (Fan et al., 2025). As shown in Fig 1, experimental results demonstrate that FasterCache can significantly accelerate inference while preserving high-quality video generation across all tested models. Specifically, on Vchitect-2.0, FasterCache achieves 1.67× speedup, with performance comparable to the baseline (VBench: baseline 80.80% → FasterCache 80.84%). Furthermore, our method outperforms existing approaches in both inference speed and video generation quality, highlighting its effectiveness and efficiency in real-world applications.

Overall, the contributions of this work are as follows:

- • We analyze the feature reuse process in cache-based methods and discover that while adjacent-step features in attention modules appear to be similar, their subtle differences can degrade output quality if ignored.
- • We conduct a pioneering investigation of CFG’s potential for acceleration, finding high redundancy within the same timestep but weaker similarity across adjacent timesteps, revealing new acceleration opportunities.
- • We propose FasterCache, a training-free strategy that dynamically adjusts feature reuse, preserving both feature distinction and continuity. It also introduces CFG-Cache to accelerate inference while preserving details in generated videos.
- • We empirically evaluate our approach on various video diffusion models, demonstrating significant improvement in inference speed while maintaining high video quality.

- 2 METHODOLOGY

- 2.1 PRELIMINARY

Diffusion model is a generative model consisting of a forward process and a reverse process. Specifically, its forward diffusion process progressively adds noise to the data x0 ∼ pdata(x0), eventually destroying the signal. This can be formulated as:

##### q(xt|x0) = N(xt;√αtx0,√1 − αtI), (1)

where {αt}Tt=1 controls the noise schedules and T represents the total number of diffusion timesteps. The reverse process is typically parameterized as a UNet or transformer architecture ϵθ which is trained to predict the noise with the following loss function:

##### LDM = Ex,ϵ∼N(0,1),t[||ϵ − ϵθ(xt,t)||22]. (2)

A clean signal x0 can be recovered through iterative inference steps which predict xt−1 from xt using ϵθ. This can formulated as:

p(xt−1|xt) = N(xt−1;µθ(xt,t),Σθ(xt,t)), (3) where µθ and Σθ are the mean and variance parameterized with learnable θ.

Video diffusion models recently employ diffusion transformers as the backbone for noise prediction. This work explores video synthesis acceleration based on Open-Sora 1.2 (Zheng et al., 2024). This model is composed of 56 stacked transformer layers, with alternating spatial and temporal layers. Each layer contains not only a spatial or temporal attention module but also a cross-attention and a feed-forward network. Latte (Ma et al., 2024a) and Open-Sora-Plan (PKU-Yuan Lab and Tuzhan AI etc., 2024) also adopt a similar architecture as their noise prediction networks.

Classifier-Free Guidance (CFG) has proven to be a powerful technique for enhancing the quality of synthesized images/videos in diffusion models. During the sampling process, CFG computes two

outputs, namely ϵθ(xt,c) for the conditional input c and ϵθ(zt,∅) for the unconditional input ∅ (often an empty or negative prompt). The final output is given by:

##### ϵ˜θ(xt,c) = (1 + g)ϵθ(xt,c) − gϵθ(zt,∅), (4)

where g is the guidance scale. As shown in Fig. 3 (b), while CFG significantly enhances visual quality, it also increases computational cost and inference latency due to the additional computation required for unconditional outputs.

- 2.2 RETHINKING ATTENTION FEATURE REUSE

Attention feature reuse has become a primary focus for cache-based acceleration methods in video generation (e.g., pyramid attention reuse of PAB). In video diffusion models, features of attention modules (e.g., spatial attention and temporal attention) exhibit a high similarity between adjacent timesteps, as illustrated in Fig. 4. Hence, existing methods completely bypass the attention computations in subsequent timesteps by reusing the cached attention features, thereby significantly reducing computational costs. To gain a better understanding of the implications of attention feature reuse in video generation, we first visualize the videos generated with the same random seed and observe that existing feature reuse methods result in a noticeable loss of details in the output. For example, as illustrated in Fig. 5, compared to the original video generated without feature reuse, the video generated with vanilla feature reuse exhibits a smoother sky, with a lack of visible stars, indicating a noticeable degradation in fine details.

[Figure 44]

Figure 4: Comparison of the mean squared error (MSE) of attention features between the current and previous diffusion steps. Smaller values indicate higher similarity.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| |
|---|

[Figure 49]

| |
|---|

[Figure 50]

| |
|---|

[Figure 51]

| |
|---|

Original Vanilla Feature Reuse Step21 Step22 - Step21 Step22 Step23 - Step22 Step23

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

- Figure 5: Visual quality degradation caused by Vanilla Feature Reuse (left) and feature differences between adjacent timesteps (right).

To investigate the underlying causes of this phenomenon, we subsequently visualize the attention features between adjacent timesteps and analyze their differences. The results indicate that while the attention features between adjacent timesteps are highly similar, there exist noticeable differences between them. These subtle variations between timesteps are essential for preserving fine details in video generation. Therefore, directly reusing features without accounting for these differences leads to the loss of important visual information, resulting in smoother and less detailed outputs. This highlights the need for a more refined approach to feature reuse, i.e., one that can retain computational efficiency while preserving key inter-step variations.

- 2.3 FEATURE REDUNDANCY IN CFG

Following the observation of feature redundancy in attention modules across adjacent timesteps, we further extend our investigation into other critical components of the diffusion models. Through this broader analysis of the entire denoising process, we find that classifier-free guidance (CFG) significantly increases inference time, as it requires the computation of both conditional and unconditional outputs at every timestep. While CFG has been widely adopted for enhancing visual quality, there is little exploration to reduce its computational burden, leaving this aspect largely uncharted.

To explore potential redundancy within CFG, we first conduct a quantitative analysis of the similarity between conditional and unconditional outputs at the same timestep as well as across adjacent timesteps based on mean squared error (MSE). As shown in Fig. 6 (a), the results reveal that, in

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Original Reuse unconditional outputs directly

(a) Feature MSE Curve between CFG Outputs (b) Direct reuse of unconditional outputs degrades visual quality

- Figure 6: (a) The MSE between conditional and unconditional outputs at the same timestep as well

- as across adjacent timesteps. (b) Directly reusing unconditional outputs from previous timesteps will lead to a significantly degraded visual quality. the mid to later stages of sampling, the similarity between conditional and unconditional outputs
- at the same timestep is remarkably high, significantly surpassing that of adjacent steps. Hence, as illustrated in Fig. 6 (b), directly reusing unconditional outputs from adjacent timesteps, as suggested in existing methods, leads to significant error accumulation, resulting in a decline in video quality. These results indicate substantial redundancy in the CFG process and highlight the necessity for a new strategy to accelerate CFG without compromising the quality of the generated outputs.

[Figure 64]

|[Figure 65]|
|---|
|[Figure 66]|

Original

[Figure 67]

|[Figure 68]|
|---|
|[Figure 69]|

Reuse Conditional Output

(a) Visualization results of different methods to avoid CFG redundancy (b) MSE of different frequency features bias

[Figure 70]

|[Figure 71]|
|---|
|[Figure 72]|

CFG-Cache

[Figure 73]

CacheStartsCFG

- Figure 7: (a) Simply reusing the conditional output from the same time step results in the poor generation of intricate details. (b) Trend curves of high and low-frequency biases between conditional and unconditional outputs change as sampling progresses.

- 2.4 FASTERCACHE FOR VIDEO DIFFUSION MODEL

Capitalizing on the above discoveries, we introduce an innovative approach, FasterCache, which accelerates inference for video diffusion models while preserving high-quality generation. This is accomplished through a Dynamic Feature Reuse Strategy that maintains feature distinction and temporal continuity. Furthermore, we introduce CFG-Cache to optimize the reuse of conditional and unconditional outputs, further enhancing inference speed without compromising visual quality.

Dynamic Feature Reuse Strategy As discussed in Section 2.2, vanilla attention feature reuse strategy neglects the feature differences between adjacent timesteps which leads to visual quality degradation. Hence, instead of directly reusing previously cached features at the current timestep, we propose a Dynamic Feature Reuse Strategy that can more effectively capture and preserve critical details in the generated videos. Specifically, for the attention modules in diffusion models, we compute the attention module outputs at every alternate timestep. For example, we calculate the attention outputs for each layer at t + 2 and t timesteps, denoted as Ft+2 and Ft, and store them in the feature cache as Fcachet+2 and Fcachet . To dynamically adjust feature reuse, we compute the difference between the adjacent cached features. This serves as a bias for approximating the feature variation trend and enables the reused features to more accurately capture the evolving details across timesteps. For the intermediate t − 1 timestep, its features can be computed as:

##### Ft−1 = Fcachet + (Fcachet − Fcachet+2 ) ∗ w(t), (5)

where w(t) is a weighting function that modulates the contribution of the feature difference to account for variation between adjacent timesteps, ensuring both efficiency and the preservation of fine details in the generated videos. In our experiments, w(t) gradually increases as the sampling process progresses, allowing the model to place greater emphasis on the feature differences at later stages of generation. Further discussions on the design of feature bias term and the selection of w(t) in Eq. (5) can be found in Appendix A.3.1. Consequently, our approach significantly accelerates inference while preserving the visual quality of the synthesized videos.

𝑡 step 𝑡 − 1 step 𝑡 − 𝑛 step

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

𝑥 𝑥 𝑥

𝑥

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

…

###### DiT DiT DiT DiT

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

𝜖 𝑥 ,𝑡,𝑐 𝜖 𝑥 ,𝑡,∅ 𝜖 𝑥 ,𝑡 − 1,𝑐 𝜖 𝑥 ,𝑡 − n,𝑐

𝜖̂ 𝑥 ,𝑡 − 1,∅ 𝜖̂ 𝑥 ,𝑡 − n,∅

[Figure 102]

[Figure 103]

[Figure 104]

Δ ,Δ

- Figure 8: Overview of the CFG-Cache. CFG-Cache accelerates the computation of the unconditional output (in the dashed orange box) by caching the high- and low-frequency biases between the conditional and unconditional outputs, and dynamically enhancing them during reuse.

CFG-Cache As analyzed in Section 2.3, the conditional and unconditional outputs at the same timestep exhibit high similarity in CFG, indicating significant information redundancy. A naive approach to take advantage of this would be to directly reuse the conditional features for the corresponding unconditional outputs at the same timestep. However, this often leads to a noticeable degradation in detail generation. As illustrated in Fig 7 (a), this approach results in poor generation of intricate details, such as the texture of the spacesuit which shows a lack of details and clarity. Since both the conditional and unconditional outputs in CFG represent predicted noise, and drawing inspiration from the Dynamic Feature Reuse Strategy and FreeU (Si et al., 2024), we analyze the differences between these two outputs in the frequency domain. In Fig 7 (b), we observe that, from the activation of CFG-Cache until the end of the sampling, the difference between the conditional and unconditional outputs gradually shifts from being dominated by low-frequency components to being dominated by high-frequency components. This indicates that the effects of CFG in the sampling process is primarily to influence perceptual features like layout and shape during the early and mid-stages, while contributing to detail synthesis in the later stages. A similar phenomenon can also be observed in Hsiao et al. (2024). This observation suggests that despite their overall similarity, key differences in frequency components must be addressed to avoid the degradation of fine details. More discussion and visualization can be found in Appendix A.3.2.

Building on this discovery, we propose CFG-Cache, a novel approach designed to account for both high- and low-frequency biases, coupled with a timestep-adaptive enhancement technique. Specifically, as shown in Fig. 8, at timestep t, a full inference is performed to obtain both the conditional output ϵθ(xt,t,c) and the unconditional output ϵθ(xt,t,∅). We then separately calculate the biases for the high-frequency (∆HF) and low-frequency (∆LF) components between these two outputs:

∆LF = FFT (ϵθ(xt,t,∅))low − FFT (ϵθ(xt,t,c))low, (6) ∆HF = FFT (ϵθ(xt,t,∅))high − FFT (ϵθ(xt,t,c))high. (7)

These biases ensure that both high- and low-frequency differences are accurately captured and compensated during the reuse process. In the subsequent n timesteps (from t − 1 to t − n), we infer only the outputs of the conditional branches and compute the unconditional outputs using the cached ∆HF and ∆LF as follows:

ϵˆθ(xt−i,t − i,∅) = IFFT (Flow,Fhigh), (8) Flow = ∆LF ∗ w1 + FFT (ϵθ(xt−i,t − i,c))low, (9)

Fhigh = ∆HF ∗ w2 + FFT (ϵθ(xt−i,t − i,c))high. (10)

Here, w1 and w2 are adaptively adjusted based on the sampling timestep t, with greater emphasis on different frequency components at distinct sampling phases. The weighting scheme is defined as:

w1 = 1 + α1 · I(t > t0),w2 = 1 + α2 · I(t <= t0), (11)

where α1 and α2 are hyperparameter weights, t0 is the manually set switching timestep, and I(·) is the indicator function. This formulation ensures that mid-low frequencies are prioritized in the mid-sampling phase, while high-frequency components receive more attention in the later phase.

### 3 EXPERIMENTS

- 3.1 EXPERIMENTAL SETTINGS Base models and compared methods To demonstrate the effectiveness of our method, we apply our acceleration technique to different video synthesis diffusion models, including the Open-Sora

1.2 (Zheng et al., 2024), Open-Sora-Plan (PKU-Yuan Lab and Tuzhan AI etc., 2024), Latte (Ma

- et al., 2024a), CogVideoX (Yang et al., 2024), and Vchitect-2.0 (Fan et al., 2025). We compare our base models with recent efficient video synthesis techniques, including PAB (Zhao et al., 2024c) and ∆-DiT (Chen et al., 2024c), to highlight the benefits of our approach. Notably, ∆-DiT was originally designed as an acceleration method for image synthesis. Here we have adapted it for video synthesis to facilitate comparison. Please refer to the Appendix for more details of the base models and compared methods.

Evaluation metrics and datasets To assess the performance of video synthesis acceleration methods, we focus primarily on two aspects, namely inference efficiency and visual quality. To evaluate inference efficiency, we employ Multiply-Accumulate Operations (MACs) and inference latency as metrics. We utilize VBench (Huang et al., 2024), LPIPS (Zhang et al., 2018), PSNR, and SSIM for visual quality evaluation. VBench is a comprehensive benchmark suit for video generative models. It is well-aligned with human perceptions and capable of providing valuable insights from multiple perspectives. LPIPS, PSNR, and SSIM measure the similarity between videos generated by the accelerated sampling method and those from the original model. PSNR quantifies pixel-level fidelity between outputs, LPIPS measures perceptual consistency, and SSIM assesses structural similarity. In general, higher similarity scores indicate better fidelity and visual quality.

Implementation details All experiments conduct full attention inference for spatial and temporal attention modules every 2 timesteps to facilitate dynamic feature reuse. The weight w(t) increases linearly from 0 to 1 starting from the beginning of dynamic feature reuse until the end of sampling. For CFG output reuse, full inference is conducted every 5 timesteps, starting from 1/3 of the total sampling steps (e.g., for Open-Sora 1.2, which has 30 total sampling steps, this begins at step 10). The hyperparameters α1 and α2 are set to a default value of 0.2, which performs well for most models. For more details on the selection of hyperparameters, please refer to Appendix A.5. All experiments are carried out on NVIDIA A100 80GB GPUs using PyTorch, with FlashAttention (Dao

- et al., 2022) enabled by default. Table 1: Comparison of efficiency and visual quality on a single GPU.

|Method|Efficiency MACs (P) Speedup Latency (s)| | |Visual Quality VBench LPIPS SSIM PSNR| | | |
|---|---|---|---|---|---|---|---|
| |↓<br><br>|↑|↓<br><br>|↑|↓<br><br>|↑|↑|

Open-Sora 1.2 (192 frames, 480P)

|Open-Sora 1.2 (T = 30)<br><br>∆-DiT (Nc = 14,N = 2) ∆-DiT (Nc = 28,N = 2) PAB<br><br>|6.30 5.51 4.72 5.33<br><br>|1× 1.14× 1.34× 1.23×|192.07 168.69 143.14 156.73<br><br>|78.79%<br><br>77.43%<br><br>76.60%<br><br>78.15%<br><br><br>|0.2834 0.3321 0.1041<br><br>|0.7403 0.7092 0.8821|17.77 16.24 26.43<br><br>|
|---|---|---|---|---|---|---|---|
|Ours|4.13<br><br>|1.62×<br><br>|118.44|78.46%<br><br>|0.0835|0.8932<br><br>|27.03|

Open-Sora-Plan (65 frames, 512×512)

|Open-Sora-Plan (T = 150)<br><br>∆-DiT (Nc = 14,N = 3) ∆-DiT (Nc = 28,N = 3) PAB|10.30 8.60 6.90 7.39<br><br>|1× 1.19× 1.46× 1.32×<br><br>|103.76 86.88 70.99 78.72<br><br>|80.16% 78.12% 77.71% 80.06%|0.4515 0.4819 0.2423<br><br>|0.4813 0.4467 0.7126<br><br>|16.08 15.42 20.29|
|---|---|---|---|---|---|---|---|
|Ours<br><br>|5.51<br><br>|1.68×|61.68|80.19%<br><br>|0.1348|0.8138<br><br>|23.72|

Latte (16 frames, 512×512)

|Latte (T = 50)<br><br>∆-DiT (Nc = 14,N = 2) ∆-DiT (Nc = 28,N = 2) PAB<br><br>|3.05 2.67 2.29 2.24<br><br>|1× 1.23× 1.43× 1.28×<br><br>|29.22 23.80 20.38 22.84|77.05% 76.27% 76.01% 76.70%<br><br>|0.1731 0.2245 0.2904<br><br>|0.8107 0.7620 0.7083|22.69 21.00 18.98<br><br>|
|---|---|---|---|---|---|---|---|
|Ours|1.97<br><br>|1.54×|18.98|76.89%<br><br>|0.0817|0.8948<br><br>|28.21|

CogVideoX (48 frames, 480P)

|CogVideoX (T = 50)<br><br>∆-DiT (Nc = 4,N = 2) ∆-DiT (Nc = 8,N = 2) ∆-DiT (Nc = 12,N = 2)<br><br>PAB<br><br>|6.03 5.62 5.23 4.82 4.45<br><br>|1× 1.08× 1.15× 1.26× 1.35×|78.48 72.72 68.19 62.50 57.98<br><br>|80.18% 79.61% 79.31% 79.09% 79.76%<br><br>|0.3319 0.3822 0.4053 0.0860<br><br>|0.6612 0.6277 0.6126 0.8978|17.93 16.69 16.15 28.04<br><br>|
|---|---|---|---|---|---|---|---|
|Ours|3.71<br><br>|1.62×|48.44|79.83%<br><br>|0.0766<br><br>|0.9066|28.93|

Vchitect-2.0 (40 frames, 480P)

|Vchitect-2.0 (T = 100) ∆-DiT (Nc = 6,N = 3) ∆-DiT (Nc = 12,N = 3) PAB<br><br>|14.57 13.00 11.79 12.20|1× 1.11× 1.24× 1.26×<br><br>|260.32 233.59 209.78 206.23<br><br>|80.80% 79.98% 79.50% 79.56%<br><br>|0.4153 0.4534 0.0489|0.5837 0.5519 0.8806<br><br>|14.26 13.68 27.38|
|---|---|---|---|---|---|---|---|
|Ours<br><br>|8.67<br><br>|1.67×<br><br>|156.13|80.84%<br><br>|0.0282|0.9224|31.45|

- 3.2 MAIN RESULTS

Quantitative comparison Table 1 presents a quantitative comparison of our method with ∆-DiT and PAB in terms of efficiency and visual quality. We synthesize videos with prompts provided by VBench and use the synthesized videos to compute the VBench metrics as well as calculate LPIPS, SSIM, and PSNR with videos sampled by the original model. The results demonstrate that our method achieves stable acceleration efficiency and superior visual quality across different base models, sampling schedulers, video resolutions, and lengths.

[Figure 105]

[Figure 106]

[Figure 107]

Original

|Δ-DiT| |
|---|---|
|PAB| |
| | |
|Ours| |

OursPABΔ-DiT

Open-Sora Open-Sora-Plan Latte

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 9: Visual quality comparison of different methods. Differences are highlighted in red boxes. Visual quality comparison Fig. 9 compares the videos generated by our method against those by the original model, PAB, and ∆-DiT. The results demonstrate that our method can effectively preserve the original quality and fine details. More visual results can be found in the Appendix.

- 3.3 ABLATION STUDY To comprehensively assess the effectiveness and efficiency of our method, we perform extensive ablation studies based on Open-Sora, synthesizing videos of 48 frames at 480P.

Efficiency Table 2 compares the efficiency of the original Open-Sora and its variants with different acceleration components. There are two key observations. (1) The Dynamic Feature Reuse Strategy and CFG-Cache independently contribute to significant reductions in inference costs. When combined, they further minimize inference overhead. (2) Compared to Vanilla Feature Reuse, the proposed Dynamic Feature Reuse strategy has a negligible impact on efficiency.

Table 2: Impact on inference efficiency.

VanillaFR DynamicVariantsFR CFG-Cache MACs (P) Latency (s) ∆ (s) 1.54 41.28 -

✓ 1.33 33.25 -8.03 ✓ 1.33 33.50 -7.78 ✓ 1.16 31.32 -9.96

✓ ✓ 1.01 26.12 -15.16

(Vanilla FR denotes Vanilla Feature Reuse, and ∆ represents the reduction in latency compared to the original model.)

Visual quality Table 3 compares the visual quality of the original Open-Sora with its variants implementing different acceleration components. Note that vanilla feature reuse leads to a performance drop in VBench and LPIPS. The introduction of the dynamic feature reuse strategy mitigates the loss of information and thereby improves the performance of these metrics (e.g., VBench: 78.34% → 78.69%). Fig. 10 (a) provides a visual comparison of the results. It can be observed that vanilla feature reuse shows reduced details (e.g., the moon and snowflakes), whereas dynamic feature reuse strategy can significantly alleviate this problem. The Feature MSE curves show that adding the bias term can lower the MSE between intermediate features from the original and accelerated sampling process, aligning with the visual results.

Table 4: Scaling to multiple GPUs with DSP.

Table 3: Impact on visual quality.

Method 1× A100 2× A100 4× A100 8× A100

|Variants|VBench LPIPS PSNR SSIM<br><br>|
|---|---|
|Original Open-Sora|78.99% - - -<br><br>|
|Full (w/ Vanilla FR) Full (w/ Dynamic FR)<br><br>|78.34% 0.0657 28.20 0.8785 78.69% 0.0590 28.41 0.8938<br><br>|
|CFG-Cache w/o Enhancement<br><br>Enhance LF only Enhance HF only Full (w/ full CFG-Cache)<br><br>|78.42% 0.0709 27.97 0.8727 78.58% 0.0617 28.29 0.8894 78.49% 0.0686 28.08 0.8834 78.69% 0.0590 28.41 0.8938<br><br>|

Open-Sora ( 192 frames, 480P)

|Open-Sora<br><br>|192.07 (1×)<br><br>|72.82 (2.64×)<br><br>|39.09 (4.92×)<br><br>|21.62(8.89×)|
|---|---|---|---|---|
|PAB<br><br>|156.73 (1.23×)<br><br>|58.11(3.31×)|30.91 (6.21×)|17.21 (11.16×)|
|Ours|118.44 (1.62×)|42.18(4.55×)|22.55 (8.52×)|12.57 (15.28×)|

Open-Sora-Plan(221 frames, 512×512)

|Open-Sora-Plan|316.71 (1×)<br><br>|169.21 (1.87×)<br><br>|89.10 (3.55×)|49.13(6.44×)|
|---|---|---|---|---|
|PAB<br><br>|243.33 (1.30×)<br><br>|127.30 (2.49×)|71.17 (4.45×)|37.13(8.53×)|
|Ours|187.91 (1.69×)|104.37 (3.03×)|57.70 (5.49×)|31.82(9.95×)|

(FR denotes Feature Reuse.)

###### Referring to Table 3, it can be seen that introducing CFG-Cache without enhancement reduces the visual quality. On the other hand, CFG-Cache with dynamic enhancement of either the low- or high-

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

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

w/o Feature Reuse

Original CFG-Cache w/o Enhancement

Feature MSE Feature MSE

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

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

Vanilla Feature Reuse Dynamic Feature Reuse

Enhance LF only Enhance HF only Full CFG-Cache

- (a) Impact of Dynamic FR on Feature MSE and visual results (b) Impact of CFG-Cache on Feature MSE and visual results Figure 10: Comparison of Feature MSE curves and visual results from the ablation study.

frequency bias helps to improve the visual quality, and their combined effect achieves the best visual quality. Fig. 10 (b) shows that enhancing low-frequency bias improves the fidelity of low-frequency components (e.g., clouds, tornado outlines) while enhancing high-frequency bias enriches highfrequency details (e.g., lightning). The Feature MSE curve of CFG-Cache without enhancement aligns with the reduced visual quality. Dynamic enhancement helps to mitigate error accumulation, leading to higher visual fidelity.

- 3.4 SCALABILITY AND GENERALIZATION

Scaling to multiple GPUs To evaluate the sampling efficiency of our method on multiple GPUs, we adopt the approach used in PAB and integrate Dynamic Sequence Parallelism (DSP) (Zhao et al., 2024b) to distribute the workload across GPUs. Table 4 illustrates that, as the number of GPUs increases, our method consistently enhances inference speed across different base models, surpassing the performance of the compared methods.

Performance at different resolutions and lengths To evaluate the effectiveness of our method in accelerating sampling for videos of varying sizes, we conduct tests across different video lengths and resolutions and report the results in Fig. 11. Our method maintains stable acceleration performance when faced with increasing resolutions and frame counts in videos, demonstrating its potential to accelerate sampling longer and higher-resolution videos in line with practical demands.

|41.26<br><br>24.32<br><br>10.45<br><br>6.45<br><br>4.14<br><br>0<br><br>5<br><br>10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>35<br><br>40<br><br>45 48frames, 480P<br><br>Original 1 GPU 2 GPU 4 GPU 8 GPU<br><br>1.70x<br><br>3.95x<br><br>6.40x<br><br>9.97x|
|---|
| |
|132.56<br><br>76.71<br><br>30.39<br><br>15.89<br><br>8.62<br><br>0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>120<br><br>140 240frames, 360P<br><br>Original 1 GPU 2 GPU 4 GPU 8 GPU<br><br>1.73x<br><br>4.36x<br><br>8.34x<br><br>15.39x|

|192.07<br><br>118.44<br><br>42.18<br><br>22.55<br><br>12.57<br><br>0<br><br>50<br><br>100<br><br>150<br><br>200<br><br>250 192frames, 480P<br><br>Original 1 GPU 2 GPU 4 GPU 8 GPU<br><br>1.62x<br><br>4.55x<br><br>8.52x<br><br>15.28x|
|---|
| |
|178.88<br><br>101.06<br><br>43.83<br><br>26.17<br><br>17.34<br><br>0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>120<br><br>140<br><br>160<br><br>180<br><br>200 32frames, 1080P<br><br>Original 1 GPU 2 GPU 4 GPU 8 GPU<br><br>1.77x<br><br>4.08x<br><br>6.83x<br><br>10.24x|

100 96frames, 480P

86.11

90

80

1.70x

1.73x

1.62x

70

3.95x

4.25x

7.97x

6.40x

60

12.74x

9.97x

4.55x

49.89

15.28x

8.52x

50

40

30

20.24

20

10.81

6.76

10

0

Original 1 GPU 2 GPU 4 GPU 8 GPU

###### 120 48frames, 720P

103.78

1.73x

100

1.77x

1.74x

4.36x

80

4.08x

4.32x

8.34x

6.83x

15.39x

7.40x

59.55

10.24x

13.37x

60

40

24.02

14.01

20

7.76

0

Original 1 GPU 2 GPU 4 GPU 8 GPU

Figure 11: Acceleration efficiency of our method at different video resolutions and lengths. I2V and image synthesis performance We integrate our acceleration method to the state-of-theart image-to-video model DynamiCrafter (Xing et al., 2023) and image synthesis model PixArtsigma (Chen et al., 2024a). As shown in Fig. 12, our method significantly accelerates sampling while maintaining visual fidelity, demonstrating its potential for extension to various base models.

### 4 RELATED WORK

- 4.1 DIFFUSION MODELS FOR VIDEO SYNTHESIS Diffusion models have demonstrated potential in high-quality image synthesis (Ho et al., 2020; Rombach et al., 2022; Chen et al., 2023; 2024b), attracting significant attention. Subsequent works have adapted these models for video synthesis to generate high-fidelity videos (Ho et al., 2022). Motivated by advancements in image synthesis, early studies typically employed the diffusion UNet architecture (Blattmann et al., 2023; Wang et al., 2023; Zhang et al., 2023; Wu et al., 2023; Zhang et al., 2024c). As the scalability of diffusion transformer (Peebles & Xie, 2023) was validated in image synthesis, an increasing number of works have adopted the diffusion transformer as the noise

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

OriginalOurs

Prompt

OriginalOriginalOursOurs

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

|[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>75.23<br><br>49.56 52.7438.81 0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>DynamiCrafter pixArt-sigma<br><br>[Figure 133]<br><br>Original Ours<br><br>[Figure 134]<br><br>1.52 x<br><br>(b) Results of Image Synthesis Based on PixArt-Sigma<br><br>1.36 x|
|---|

( ma

[Figure 135]

Prompt

[Figure 136]

[Figure 137]

(a) Results of Image-to-Video Sampling Acceleration Based on DynamiCrafter

(c) Comparison of Inference Time

- Figure 12: Visual results and inference time of our method on I2V and image synthesis models.

estimation network (Ma et al., 2024a; Zheng et al., 2024; PKU-Yuan Lab and Tuzhan AI etc., 2024; Yang et al., 2024).

- 4.2 EFFICIENCY IMPROVEMENTS IN DIFFUSION MODELS Despite the impressive performance of diffusion models in image and video synthesis, their substantial inference cost limits their practicality. Prior research on improving the efficiency of diffusion models has primarily focused on two perspectives, namely reducing the number of sampling steps and lowering the inference cost per sampling step. Regarding the reduction of sampling steps, most approaches achieve high-quality samples with fewer steps by employing efficient SDE or ODE solvers (Song et al., 2020; Lu et al., 2022a;b). Other methods reduce sampling steps by progressively distilling the model (Salimans & Ho, 2022; Meng et al., 2023; Sauer et al., 2023; Lin & Yang, 2024; Li et al., 2024b) or employing consistency models (Luo et al., 2023; Song et al., 2023). More works have focused on reducing the inference cost per timestep. Some approaches improve network efficiency through pruning (Zhang et al., 2024a) or quantization (Shang et al., 2023; So et al., 2024a; He et al., 2024; Li et al., 2024a; Sui et al., 2024; Zhao et al., 2024a), while others obtain more lightweight network architectures through search techniques (Li et al., 2023a; Yang et al., 2023). However, these methods often require additional computational resources for fine-tuning or optimization. Some training-free approaches (Bolya & Hoffman, 2023; Wang et al., 2024) focus on the input tokens, accelerating the sampling process by reducing the number of tokens to be processed by eliminating token redundancy in image synthesis. Other methods reuse intermediate features between adjacent sampling timesteps, avoiding redundant computations (Wimbauer et al., 2024; So

- et al., 2024b). TGATE (Zhang et al., 2024b) accelerates image generation by caching and reusing attention outputs at scheduled timesteps. DeepCache (Ma et al., 2024b) and Faster Diffusion (Li

- et al., 2023b) employ a feature caching mechanism to indirectly alter the UNet diffusion for acceleration. ∆-DiT (Chen et al., 2024c) adapts this mechanism to the diffusion transformer architecture by caching the residuals between attention layers. PAB (Zhao et al., 2024c) caches and broadcasts intermediate features at different timestep intervals based on the characteristics of varying attention blocks. Although these methods have achieved some improvements in diffusion efficiency, the efficiency enhancements for diffusion transformers in video synthesis remain insufficient.

### 5 CONCLUSION AND DISCUSSION

In this work, we present FasterCache, a training-free strategy that significantly accelerates video synthesis inference while preserving high-quality generation. Through analysis of existing cachebased methods, we find that directly reusing adjacent-step features in attention modules can degrade video quality. Additionally, we investigate the acceleration potential of CFG, identifying redundancy between conditional and unconditional features at the same timestep. Leveraging these insights, FasterCache integrates a dynamic feature reuse strategy that maintains feature distinction and temporal continuity, and CFG-Cache which optimizes the reuse of conditional and unconditional outputs to further boost speed without sacrificing detail quality. Extensive experiments demonstrate its strong performance in both efficiency and synthesis quality across diverse video models, sampling schedules, video lengths and resolutions, highlighting its potential for real-world applications.

Limitation Despite the effectiveness shown by our method, certain limitations remain. When the synthesis quality of the model is suboptimal, our acceleration method is unlikely to yield satisfactory results either. We believe that advancements in base video models will mitigate this issue. Additionally, in complex scenes with substantial video motion, our method may occasionally produce degraded results. At present, this can be remedied through manual adjustments of hyperparameters. In the future, we plan to investigate strategies for adaptive caching to further enhance performance.

- 6 ACKNOWLEDGEMENTS

This study is supported by the National Key R&D Program of China No.2022ZD0160102, and by the video generation project (Intern-Vchitect) of Shanghai Artificial Intelligence Laboratory. This study is also supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

REFERENCES

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4599–4603, 2023.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation, 2024a.

Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-δ: Fast and controllable image generation with latent consistency models, 2024b.

Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. δ-dit: A training-free acceleration method tailored for diffusion transformers. arXiv preprint arXiv:2406.01125, 2024c.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Weichen Fan, Chenyang Si, Junhao Song, Zhenyu Yang, Yinan He, Long Zhuo, Ziqi Huang, Ziyue Dong, Jingwen He, Dongwei Pan, et al. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025.

Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate posttraining quantization for diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633– 8646, 2022.

Yi-Ting Hsiao, Siavash Khodadadeh, Kevin Duarte, Wei-An Lin, Hui Qu, Mingi Kwon, and Ratheesh Kalarot. Plug-and-play diffusion distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13743–13752, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Lijiang Li, Huixia Li, Xiawu Zheng, Jie Wu, Xuefeng Xiao, Rui Wang, Min Zheng, Xin Pan, Fei Chao, and Rongrong Ji. Autodiffusion: Training-free optimization of time steps and architectures for automated diffusion model acceleration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7105–7114, 2023a.

Senmao Li, Taihang Hu, Fahad Shahbaz Khan, Linxuan Li, Shiqi Yang, Yaxing Wang, Ming-Ming Cheng, and Jian Yang. Faster diffusion: Rethinking the role of unet encoder in diffusion models. arXiv preprint arXiv:2312.09608, 2023b.

Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, and Baochang Zhang. Q-dm: An efficient low-bit quantized diffusion model. Advances in Neural Information Processing Systems, 36, 2024a.

Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024b.

Shanchuan Lin and Xiao Yang. Animatediff-lightning: Cross-model diffusion distillation. arXiv preprint arXiv:2403.12706, 2024.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022a.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022b.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024a.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15762–15772, 2024b.

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14297–14306, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, April 2024. URL https://doi.org/10. 5281/zenodo.10948109.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.

Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1972–1981, 2023.

Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In CVPR, 2024.

Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, and Eunhyeok Park. Temporal dynamic quantization for diffusion models. Advances in Neural Information Processing Systems, 36, 2024a.

Junhyuk So, Jungwon Lee, and Eunhyeok Park. Frdiff : Feature reuse for universal training-free acceleration of diffusion models, 2024b. URL https://arxiv.org/abs/2312.03517.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

Yang Sui, Yanyu Li, Anil Kag, Yerlan Idelbayev, Junli Cao, Ju Hu, Dhritiman Sagar, Bo Yuan, Sergey Tulyakov, and Jian Ren. Bitsfusion: 1.99 bits weight quantization of diffusion model. arXiv preprint arXiv:2406.04333, 2024.

Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024. Hongjie Wang, Difan Liu, Yan Kang, Yijun Li, Zhe Lin, Niraj K Jha, and Yuchen Liu. Attention-

driven training-free efficiency enhancement of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16080–16089, 2024.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.

Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6211–6220, 2024.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7623–7633, 2023.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023.

Shuai Yang, Yukang Chen, Luozhou Wang, Shu Liu, and Yingcong Chen. Denoising diffusion step-aware models. arXiv preprint arXiv:2310.03337, 2023.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Dingkun Zhang, Sijia Li, Chen Chen, Qingsong Xie, and Haonan Lu. Laptop-diff: Layer pruning and normalized distillation for compressing diffusion models. arXiv preprint arXiv:2404.11098, 2024a.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Wentian Zhang, Haozhe Liu, Jinheng Xie, Francesco Faccio, Mike Zheng Shou, and J¨urgen Schmidhuber. Cross-attention makes inference cumbersome in text-to-image diffusion models. arXiv preprint arXiv:2404.02747, 2024b.

Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023.

Yabo Zhang, Yuxiang Wei, Xianhui Lin, Zheng Hui, Peiran Ren, Xuansong Xie, Xiangyang Ji, and Wangmeng Zuo. Videoelevator: Elevating video generation quality with versatile text-to-image diffusion models. arXiv preprint arXiv:2403.05438, 2024c.

Tianchen Zhao, Tongcheng Fang, Enshu Liu, Wan Rui, Widyadewi Soedarmadji, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, Huazhong Yang, et al. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. arXiv preprint arXiv:2406.02540, 2024a.

Xuanlei Zhao, Shenggan Cheng, Zangwei Zheng, Zheming Yang, Ziming Liu, and Yang You. Dsp: Dynamic sequence parallelism for multi-dimensional transformers. arXiv preprint arXiv:2403.10266, 2024b.

Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024c.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024. URL https://github.com/hpcaitech/Open-Sora.

A APPENDIX

- A.1 FURTHER DETAILS OF BASE MODELS In this work, we applied our FasterCache to various video synthesis models, including Open-Sora 1.2 (Zheng et al., 2024), Open-Sora-Plan (PKU-Yuan Lab and Tuzhan AI etc., 2024), Latte (Ma et al., 2024a), CogVideoX (Yang et al., 2024), and Vchitect 2.0 (Fan et al., 2025). Open-Sora 1.2 integrates 2D-VAE and 3D-VAE to enhance video compression and employs ST-DiT blocks for the diffusion process. Open-Sora-Plan adopts CausalVideoVAE to compress visual representations better and 3D full attention architecture to capture joint spatial and temporal features. Latte extracts spatio-temporal tokens from input videos and then adopts a series of transformer blocks to model video distribution in the latent space. CogVideoX employs a 3D VAE to compress videos along spatial and temporal dimensions and an expert transformer with the expert adaptive LayerNorm to facilitate the fusion between the two modalities.
- A.2 FURTHER DETAILS OF COMPARED METHODS PAB (Zhao et al., 2024c) employs a pyramid-style broadcasting mechanism to propagate attention outputs across subsequent steps. It optimizes efficiency by applying distinct broadcast strategies to each attention layer based on their respective variances. Additionally, the method introduces broadcast sequence parallelism to enhance the efficiency of distributed inference. This paper follows the default parameter configuration of PAB.

∆-DiT (Chen et al., 2024c) accelerates inference by caching feature offsets instead of the full feature maps while preventing input information loss. It caches the residuals of the blocks in the latter part of DiT for approximation during early-stage sampling and caches the residuals of the blocks in the earlier part during later-stage sampling. In ∆-DiT, the parameters that need to be configured are the residual cache interval N, the number of cached blocks Nc, and the timestep boundary b for determining the position of the cached blocks. Since the source code of ∆-DiT is not publicly available, we implemented its method based on the paper for accelerating video synthesis. Following the guidelines in ∆-DiT, we experimented with different configurations of Nc and N to balance visual quality and inference speed, allowing for a fair evaluation of the method.

- A.3 MORE DISCUSSION

- A.3.1 MORE DISCUSSION ON DYNAMIC FEATURE REUSE

Effectiveness of Dynamic Feature Reuse Assume that the output features of a particular layer in the diffusion model are a function of the timestep t, denoted as F(t). The motivation behind Vanilla Feature Reuse lies in the observation that features at adjacent timesteps are highly similar. Vanilla Feature Reuse avoids the computation at the current timestep by directly reusing the features from the previous timestep, i.e. F(t) = F(t + ∆t). Although F(t) and F(t + ∆t) are very close with a minimal error E = F(t) − F(t + ∆t), the difference is not zero. To estimate this error, we assume that F(t) is a smooth and differentiable function with respect to t, allowing us to perform a Taylor expansion, yielding:

dF(t) dt

F(t + ∆t) = F(t) +

∆t +

d2F(t) dt2

∆t2 2

+ O(∆t3), (12)

dF(t) dt

F(t + 3∆t) = F(t) + 3

∆t + 3

d2F(t) dt2

∆t2 2

+ O(∆t3). (13)

By subtracting these expansions, we derive:

dF(t) dt

F(t + ∆t) − F(t + 3∆t) = (

∆t) × (−2) + O(∆t2), (14)

Based on the statistics of approximately 200 video samples, we plotted the magnitudes of the firstorder and second-order terms of F(t). When ∆t (e.g., ∆t = 1) is sufficiently small, the norm of second-order term is smaller than that of the first-order term, as shown in Fig. 13 (c). Furthermore, we tested three different estimations for F(t), denoted as Fˆ(t): (a) Fˆ(t) = F(t + 1), (b) Fˆ(t) =

2F(t) 2dt2 . Subsequently, we calculated the L1

F(t + 1) − dFdt(t), and (c) Fˆ(t) = F(t + 1) − dFdt(t) − d

distance between each Fˆ(t) and F(t). As shown in the Fig. 13 (d), incorporating the second-order term yields only a marginal reduction in the L1 distance compared to the first-order term. Therefore, the second-order terms contribute only marginally to the improvement in visual quality (VBench: 78.77% → 78.80%). However, the computation of second-order terms incurs significant costs in memory and latency. Considering both simplicity and efficiency, we use only the first-order term for error estimation in Dynamic Feature Reuse. Based on these analyses and statistical results, we define the error term as:

dF(t) dt

∆t = (F(t + ∆t) − F(t + 3∆t)) ∗ w. (15)

E = F(t) − F(t + ∆t) ≈ −

The scale factor w is introduced to scale the bias term to approximate the error E. In Eq. (5), E = Ft−1 − Fcachet ≈ (Fcachet − Fcachet+2 ) ∗ w(t). By introducing this feature bias term, the information loss could be reduced, thereby improving the quality of the synthesis videos while maintaining computational efficiency.

Design choices for w(t) in Dynamic Feature Reuse As shown in Fig. 13 (a), we tried different design choices for Dynamic Feature Reuse (DFR) and found that the linear increasing strategy is a simple and effective manner for dynamically capturing missing features. Different design choices for DFR: (1) Constant weights w(t). A constant weight of w(t) = 0.5 is applied to the feature biases at each accelerated timesteps. (2) Learnable weights w(t). We introduced a set of learnable parameters w(t), which are optimized by minimizing the MSE loss between the features output by DFR during accelerated sampling and those generated in the original unaccelerated sampling process, resulting in the learned w(t). (3) Linearly increasing w(t) (Our DFR). Starting from the application of DFR to the end of sampling proces, the weight function w(t), used for weighting feature biases, linearly increases from 0 to 1.

The trend of the optimized w(t) is shown in Fig 13 (a), the result indicates that w(t) obtained through optimization gradually increases as sampling progresses. This trend is primarily attributed to the increasing stability of feature biases in Eq. 5 with respect to the sampling timesteps and the growing reliance on bias features for synthesizing high-quality details in the later stages of sampling. The performance of different strategies is shown in Table 5. All results incorporating feature biases outperform those without them. The linearly increasing w(t) achieves comparable performance to optimized learnable w(t), both outperforming constant w(t). Given the simplicity of linear interpolation, we ultimately adopt linearly interpolated w(t) to weight the feature biases.

Table 5: Performance of different Dynamic FR strategies.

|Variants|LPIPS PSNR SSIM<br><br>|
|---|---|
|Vanilla FR Dynamic FR (Constant w(t) = 0.5) Dynamic FR(Learned w(t))|0.0657 28.20 0.8785 0.0615 28.33 0.8889 0.0596 28.45 0.8941<br><br>|
|Dynamic FR(Linear w(t))|0.0590 28.41 0.8938<br><br>|

Comparison between Dynamic FR and Vanilla FR Fig. 13 (b) presents the generated results of Vanilla Feature Reuse (FR) and Dynamic FR and the differences between the features produced by Vanilla FR and Dynamic FR compared to the original features. It is evident that, due to the introduction of feature biases, the feature differences between Dynamic FR and the original features are less significant. In contrast, the features produced by the model accelerated with Vanilla FR exhibit detail loss compared to the original features, leading to noticeable detail degradation in the synthesized images (as highlighted by the red box).

- A.3.2 FURTHER DISCUSSION ON CFG-CACHE

Effectiveness of CFG-Cache The reliability of CFG-Cache stems from three key factors: (a) After the early stage tearly, the similarity between conditional output cond(t) and unconditional output uncond(t) at the same timestep t:

##### uncond(t) = cond(t) + ∆,when t >= tearly. (16)

- (b) The predictability of biases between conditional and unconditional output from previous timesteps, expressed as:

##### ∆ = uncond(t + ∆t) − cond(t + ∆t) = uncond(t) − cond(t) + ϵ. (17)

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

| |
|---|

| |
|---|

FeatureOutput

[Figure 142]

[Figure 143]

[Figure 144]

| |
|---|

| |
|---|

Original Feature | VFR – Original | | DFR – Original |

(b) Difference between VFR/DFR features and original features, and corresponding outputs.

|(a) Different setting of 𝑤(𝑡)|
|---|

[Figure 145]

[Figure 146]

- (c) The norm of first-order and second-order term (d) L1 distance of different approximation.

- Figure 13: Design choices for Dynamic Feature Reuse and comparison between Dynamic Feature Reuse (DFR) and Vanilla Feature Ruse (VFR).

In practice, we find that when ∆t is sufficiently small, the ϵ can be considered negligible. Then:

uncond(t) ≈ cond(t) + (uncond(t + ∆t) − cond(t + ∆t)) (18)

(c) The dynamic variations of the frequency-domain distribution of feature biases, as illustrated in Fig. 7(b) and Fig.14.

Visualization of CFG biases From the onset of CFG-Cache to the end of sampling, the differences between the conditional and unconditional output features progressively shift from being dominated by low-frequency features to high-frequency features. As shown in Fig 14, this observation aligns with the feature visualization analysis: during the early and middle sampling stages, CFG primarily guides the model to synthesize perceptual features such as reasonable shapes and layouts, which are often represented in the low-frequency feature domain. In contrast, during the later stages of sampling, CFG contributes primarily to the synthesis of high-quality details, typically governed by high-frequency features. This insight motivates us to assign higher weights to features of different frequencies at different stages, allowing to gain more emphasis, thereby preserving the visual quality.

t=T t=0

Cond-

Uncond

|𝑥ො0|
|---|

𝑥ො0

[Figure 147]

[Figure 148]

- Figure 14: The variation in differences between the conditional and unconditional outputs during the sampling process.

###### A.3.3 FASTERCACHE UNDER DIFFERENT CFG SCALES AND NEGATIVE PROMPTS

We compared two different negative prompt settings on Open-Sora: (1) default empty negative prompt and (2) non-empty negative prompt:

“worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy”

We calculated the LPIPS, SSIM, and PSNR between the videos generated by FasterCache and those generated by the original model. As shown in Fig. 15 (a) and (b), the experimental results show that FasterCache performs similarly under both prompt settings. This is consistent with our expectations, as CFG-Cache caches the biases between the conditional and unconditional outputs, which are not significantly affected by changes in the negative prompt setting.

We also experimented with different CFG guidance scales g on Open-Sora. As shown in Fig 15 (b) and (c), regardless of increasing or decreasing the scale, while the adjustment affects the original Open-Sora results, FasterCache consistently maintains a high level of alignment with the original results, particularly in preserving details. Therefore, FasterCache is not affected by changes in the CFG guidance scale and maintains high-quality acceleration.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

FasterCacheOriginalFasterCacheOriginal

[Figure 153]

(a) Non-empty, g=7 (b) FasterCache performance across different CFG scales under empty and non-empty negative prompts.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

|g=3|g=5|g=7|g=9|g=11|
|---|---|---|---|---|

(c) Visual results of FasterCache across different CFG scales. (Empty negative prompt)

Figure 15: The performance of FasterCache under different CFG scales with empty and non-empty negative prompt settings.

[Figure 164]

[Figure 165]

[Figure 166]

Figure 16: Different Settings of α in CFG-Cache.

###### A.4 ADDITIONAL QUALITATIVE EXPERIMENTS More visual results on Text-to-Video models The additional visual comparison results for OpenSora 1.2 (Zheng et al., 2024), Open-Sora-Plan (PKU-Yuan Lab and Tuzhan AI etc., 2024), and Latte (Ma et al., 2024a) are presented in Fig. 17, Fig. 18, and Fig. 19, while further comparisons for CogVideoX-2B (Yang et al., 2024) and Vchitect-2.0 (Fan et al., 2025) are shown in Fig. 20. Our method demonstrates reliable fidelity across various models and styles or content in video synthesis, while simultaneously achieving acceleration.

Additionally, Fig. 21 demonstrates the visual performance of FasterCache on state-of-the-art models CogVideoX-5B and Mochi-10B (Team, 2024). FasterCache achieves an acceleration of 1.63 times (206s → 126s) on CogVideoX-5B and 1.74 times (320s → 184s) on Mochi-10B. As model scale increases, FasterCache consistently accelerates the sampling process while maintaining fidelity in synthesized videos. We also observe that as the generative capability of the base model improves, FasterCache becomes more robust in synthesizing videos with complex scenes or rapid motion. For instance, in Fig. 21, the 1st example shows subtle details of small groups of fish, the 3rd example highlights intricate finger details and complex non-rigid motions, and the 4th and 5th examples exhibit rapid and large-scale movements. These results demonstrate the broad potential of FasterCache in practical applications.

More visual results on Image-to-Video models We conducted image-to-video sampling acceleration experiments based on DynamiCrafter (Xing et al., 2023), achieving a 1.52× speedup on a single GPU. Additional visual results are provided in Fig. 22. Our method demonstrates good fidelity in the acceleration of image-to-video models, indicating broad potential for practical applications.

- A.5 ADDITIONAL QUANTITATIVE EXPERIMENTS

- A.5.1 USER PREFERENCE STUDY

To assess the effectiveness of our FasterCache, we additionally conduct a human evaluation. We randomly selected 30 videos for each model. Each rater receives a text prompt and two generated videos from different sampling acceleration methods (in random order). They are then asked to select the video with better visual quality. Five raters evaluate each sample, and the voting results are summarized in Table 6. As one can see, compared to other acceleration methods, the raters strongly prefer the videos generated by our method.

- Table 6: User preference study. The numbers represent the percentage of raters who favor the videos synthesized by our method.

|Method comparison<br><br>|Open-Sora 1.2 Open-Sora-Plan Latte|
|---|---|
|Ours vs. ∆-DiT<br><br>|80.67% 78.00% 77.33%|
|Ours vs. PAB<br><br>|69.33% 72.67% 74.00%|

A.5.2 HYPERPARAMETER SELECTION

- Table 7: Different Dynamic FR caching intervals.

|Interval|LPIPS PSNR SSIM<br><br>|
|---|---|
|2<br><br>3<br><br>4<br><br>5<br><br><br>|0.0590 28.41 0.8938 0.0698 27.95 0.8853 0.0751 27.61 0.8823 0.0897 27.39 0.8712|

Table 8: Different CFG-Cache caching intervals.

|Interval|LPIPS PSNR SSIM<br><br>|
|---|---|
|1 3 5 7 9|0.0496 28.88 0.8964 0.0537 28.56 0.8947 0.0590 28.41 0.8938 0.0724 27.68 0.8818 0.0104 27.44 0.8706<br><br>|

Caching timestep interval of Dynamic Feature Reuse We experimented with different caching timestep intervals for Dynamic Feature Reuse. According to Table 7, it can be observed that as the caching timestep interval increases, the fidelity of the synthesized results gradually decreases. In practice, the caching timestep interval for Dynamic Feature Reuse can be adjusted as needed.

Caching timestep interval of CFG-Cache We experimented with different CFG-Cache intervals and found that when the interval exceeds 5 timesteps, there is a significant decline in fidelity, as shown in Table 8. Therefore, to balance fidelity and efficiency, we chose a CFG-Cache caching interval of 5. This means that after CFG-Cache is initiated, the model performs full inference for both the conditional and unconditional branches every 5 timesteps and caches the features.

The configuration of α in CFG-Cache. In CFG-Cache, we experimented with different configurations of α, where α1 is used to enhance low-frequency biases and α2 is used to enhance high-frequency biases. Through these experiments shown in Fig. 16, we found that α1 = 0.2 and α2 = 0.2 works effectively.

An astronaut flying in space, tilt down a bird building a nest from twigs and leaves

[Figure 167]

[Figure 168]

Δ-DiTΔ-DiTΔ-DiTΔ-DiTOursOursOursOursOriginalOriginalOriginalOriginalPABPABPABPAB

a cat and a dog a cup and a couch

[Figure 169]

[Figure 170]

A tranquil tableau of the Parthenon stands an umbrella and a handbag

[Figure 171]

[Figure 172]

Ashtray full of butts on table, smoke flowing on black background Fireworks

[Figure 173]

[Figure 174]

Figure 17: More visual results on Open-Sora (480P 192 frames). Zoom in for details.

a bear catching a salmon in its powerful jaws a cake A car moving slowly on an empty street, rainy evening

[Figure 175]

[Figure 176]

[Figure 177]

Δ-DiTΔ-DiTΔ-DiTOriginalOriginalOriginalOursOursOursPABPABPAB

A corgi is playing drum kit

A panda cooking in the kitchen a zebra

[Figure 178]

[Figure 179]

[Figure 180]

A tranquil tableau of in the quaint village square, a traditional wrought-iron streetlamp featured delicate filigree patterns and amber-hued glass panels

A robot DJ is playing the turntable, in heavy raining futuristic tokyo rooftop cyberpunk night, sci-fi, fantasy

A jellyfish floating through the ocean, with bioluminescent tentacles

[Figure 181]

[Figure 182]

[Figure 183]

Figure 18: More visual results on Open-Sora-Plan (512×512 65 frames). Zoom in for details.

a cat on the right of a dog, front view a shark is swimming in the ocean, pixel art a bird soaring gracefully in the sky

[Figure 184]

[Figure 185]

[Figure 186]

Δ-DiTOriginalOursPAB

a hot dog on the bottom of a pizza, front view

A jellyfish floating through the ocean, with bioluminescent tentacles

A jellyfish floating through the ocean, with bioluminescent tentacles

[Figure 187]

[Figure 188]

[Figure 189]

Δ-DiTOriginalOursPAB

A panda drinking coffee in a cafe in Paris, A super robot protecting city racking focus

A fat rabbit wearing a purple robe walking through a fantasy landscape

[Figure 190]

[Figure 191]

[Figure 192]

Δ-DiTOriginalOursPAB

Figure 19: More visual results on Latte (512×512 16 frames). Zoom in for details.

a giraffe on the right of a bird, front view Fireworks Turtle swimming in ocean

[Figure 193]

[Figure 194]

[Figure 195]

Δ-DiTΔ-DiTΔ-DiTΔ-DiTOursOursOursOursΔ-DiTOriginalOriginalOriginalOriginalOursOriginal

Vchitect-2.0CogVideoX

A beautiful coastal beach in spring, waves lapping on sand, featuring a steady and smooth perspective

A squirrel eating a burger a bear sniffing the air for scents of food

[Figure 196]

[Figure 197]

[Figure 198]

A person is arranging flowers a bird flying over a snowy forest a cake

[Figure 199]

[Figure 200]

[Figure 201]

A space shuttle launching into orbit, with flames and smoke billowing out from the engines

a bus accelerating to gain speed a cow

[Figure 202]

[Figure 203]

[Figure 204]

A teddy bear washing the dishes a white bird Hyper-realistic spaceship landing on Mars

[Figure 205]

[Figure 206]

[Figure 207]

Turtle swimming in ocean

scissors and a teddy bear fountain

[Figure 208]

[Figure 209]

[Figure 210]

Δ-DiTOursOriginal

Figure 20: More visual results on CogVideoX-2B (480P 48 frames) & Vchitect-2.0 (480P 40frames).

CogVideoX-5B 48frames 480P

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

OriginalFasterCache

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

OriginalFasterCacheFasterCacheFasterCacheFasterCacheOriginalOriginalOriginal

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Mochi-10B 163frames 480P

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

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

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Figure 21: More visual results on CogVideoX-5B and Mochi-10B. Zoom in for details.

[Figure 261]

OriginalOriginalOriginalOriginalOriginalOursOursOursOursOurs

prompt

[Figure 262]

[Figure 263]

prompt

[Figure 264]

[Figure 265]

prompt

[Figure 266]

[Figure 267]

prompt

[Figure 268]

[Figure 269]

prompt

[Figure 270]

Figure 22: More visual results on DynamiCrafter (1024×576 16frames). Zoom in for details.

