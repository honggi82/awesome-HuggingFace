## Long-Context Autoregressive Video Modeling with Next-Frame Prediction

Yuchao Gu, Weijia Mao, Mike Zheng Shou

### arXiv:2503.19325v3[cs.CV]18May2025

Abstract—Long-context video modeling is essential for enabling generative models to function as world simulators, as they must maintain temporal coherence over extended time spans. However, most existing models are trained on short clips, limiting their ability to capture long-range dependencies, even with test-time extrapolation. While training directly on long videos is a natural solution, the rapid growth of vision tokens makes it computationally prohibitive. To support exploring efficient long-context video modeling, we first establish a strong autoregressive baseline called Frame AutoRegressive (FAR). FAR models temporal dependencies between continuous frames, converges faster than video diffusion transformers, and outperforms token-level autoregressive models. Based on this baseline, we observe context redundancy in video autoregression. Nearby frames are critical for maintaining temporal consistency, whereas distant frames primarily serve as context memory. To eliminate this redundancy, we propose the long short-term context modeling using asymmetric patchify kernels, which apply large kernels to distant frames to reduce redundant tokens, and standard kernels to local frames to preserve fine-grained detail. This significantly reduces the training cost of long videos. Our method achieves state-of-the-art results on both short and long video generation, providing an effective baseline for long-context autoregressive video modeling. The code is released at https://github.com/showlab/FAR.

Index Terms—Video Generation, Autoregressive Video Modeling, Diffusion Model.

✦

1 INTRODUCTION

# L

Ong-context video modeling is essential for advancing video generative models toward real-world simula-

tion [1]. However, current state-of-the-art video generative models (e.g., Wan [2], Cosmos [3]) fall short in this aspect. These models are typically trained on short video clips; for example, Wan and Cosmos are trained on approximately 5second video segments. These methods effectively capture short-term temporal consistency, such as object or human motion. However, they fail to maintain long-term consistency, like memorizing the observed environments, which requires learning from long video observations.

Learning from long videos presents inherent challenges, primarily due to the prohibitive computational cost associated with processing the large number of vision tokens. As a result, many previous efforts have focused on test-time long video generation, employing training-free approaches to produce extended video sequences [4], [5]. However, while these methods can generate visually plausible long videos, they fail to effectively leverage long-range context. To truly capture long-range dependencies, it is essential to train or fine-tune models directly on long videos. However, this remains computationally expensive. Concurrent efforts, such as direct long-context tuning [6], face high computational costs, while test-time training [7] typically requires specialized architectural designs and incurs additional inference overhead. Therefore, an efficient framework for long-video training is urgently needed to enable effective long-context video modeling.

- • Corresponding author: Mike Zheng Shou (E-mail: mikeshou@nus.edu.sg)
- • Yuchao Gu, Weijia Mao and Mike Zheng Shou are with the Showlab, National University of Singapore.

[Figure 1]

Fig. 1: Evaluation on Long Video Prediction. FAR better exploits long video contexts and achieves accurate prediction.

To investigate this problem, we first establish a baseline for video autoregressive modeling, namely the Frame AutoRegressive (FAR) model. Unlike token-based autoregressive models (Token-AR), FAR operates in a continuous latent space, capturing causal dependencies between frames while still allowing full attention modeling within each frame. FAR is trained using a frame-wise flow matching objective with autoregressive context. As a hybrid ARDiffusion model, FAR also encounters a gap between the observed context during training and inference. This is a common issue in similar models (e.g., [8], [9], [10]). To address this problem, we propose training FAR with stochastic clean context, which enables the model to leverage clean context signals during training and thus reduces bias

at inference time. We demonstrate that FAR trained with stochastic clean context achieves better performance than video diffusion transformers and Token-AR, establishing itself as a strong autoregressive video generation baseline.

Building on FAR, we observe context redundancy in video autoregressive modeling. Specifically, the current frame relies more heavily on nearby frames to capture local motion consistency, while distant frames primarily function as context memory. To exploit this property, we introduce long short-term context modeling with asymmetric patchify kernels. In this approach, we apply a large patchify kernel to distant context frames in order to compress redundant tokens, while using the standard patchify kernel on nearby context frames to preserve fine-grained temporal consistency. This method significantly reduces training costs on long videos and leads to notable improvements in FAR’s long-context modeling capability, as demonstrated in actionconditioned long video prediction.

Our contributions are summarized as follows:

- 1) We introduce FAR, an strong autoregressive video generation baseline, combined with stochastic clean context to bridge the training-inference gap in observed context.
- 2) Building on FAR, we observe context redundancy in video autoregressive modeling and propose long short-term context modeling with asymmetric patchify kernels to substantially reduce long-video training costs.
- 3) FAR achieves state-of-the-art performance in both short- and long-video modeling.

TABLE 1: Model Variants of FAR. We follow the model size configurations of DiT [39] and SiT [40].

Models #Layers Hidden Size MLP #Heads Params

FAR-B 12 768 3072 12 130M FAR-M 12 1024 4096 16 230M FAR-L 24 1024 4096 16 457M FAR-XL 28 1152 4608 18 674M

FAR-B-Long 12 768 3072 12 158M FAR-M-Long 12 1024 4096 16 280M

show that FAR can learn causal dependencies from continuous frames and achieve better performance than Token AR in both short- and long-video modeling.

Hybrid AR-Diffusion Models. To leverage the strengths of both continuous latent spaces and autoregressive modeling, recent studies [10], [22], [23] have explored hybrid ARDiffusion models. These models typically employ a diffusion objective for image-level modeling with autoregressive contexts. Hybrid AR-Diffusion models are widely applicable to both visual [8]–[10] and language generation [24], [25]. Recent research has also applied it in frame-level autoregressive modeling [8], [9] for video generation. However, they suffer from a training-inference discrepancy in the observed context. Some studies [26], [27] have attempted to mitigate this issue by maintaining a clean copy of the noised sequence during training, but this approach doubles the training cost. Among these methods, FAR efficiently addresses the training-inference gap through the proposed stochastic clean context, demonstrating its superior performance in long-context video modeling.

- 2 RELATED WORK

- 2.1 Video Generation

Video Diffusion Models. Recent advances in video generation have led to the scaling of video diffusion transformers [1], [11], [12] for text-to-video generation, resulting in superior visual quality. Pretrained text-to-video models are subsequently fine-tuned to incorporate images as conditions for image-to-video generation [11], [13], [14]. The trained image-to-video models can be utilized for autoregressive long-video generation using a sliding window [4], [15], but their ability to leverage visual context is limited by the sliding window’s size. In this work, we show that FAR achieves better convergence than video diffusion transformers for short-video generation while naturally supporting variablelength visual context.

Token Autoregressive Models. Video generation based on token autoregressive models (i.e., Token AR) aims to follow the successful paradigm of large language models. These models typically quantize continuous frames into discrete tokens [16], [17] and learn the causal dependencies between tokens using language models [18], [19]. While they achieve plausible performance, their generation quality remains inferior to that of video diffusion transformers due to information loss from vector quantization. Additionally, unidirectional visual token modeling may be suboptimal [20]. Subsequent studies have explored continuous tokens [21] without vector quantization but have not demonstrated their effectiveness in video generation. In this work, we

##### 2.2 Long-Context Language Modeling

Long-context language modeling typically follows two main approaches: test-time extrapolation and direct fine-tuning on long sequences. In the test-time extrapolation setting, many studies explore extrapolatable positional embeddings [28]–[30]. While these methods enable inference on longer sequences, they often underperform compared to models trained directly on long-text corpora. To reduce the computational cost, recent work [31], [32] has proposed efficient long-sequence fine-tuning strategies for large language models. However, training on long video sequences poses greater challenges, as vision tokens grow much faster than language tokens with increasing context length. To address this, we propose a long short-term context modeling approach using asymmetric patchify kernels, which effectively reduce context redundancy during long-video training.

##### 2.3 Long-Context Video Modeling

Recent advancements in video generation models have enabled their use as interactive world simulators [33]–[35], which require the ability to exploit long-range context and memorize the observed environment. However, existing video diffusion transformers lack effective mechanism to utilize long-range context. Although early work [36]–[38] has explored long-video prediction, it has been limited in visual quality and long-range consistency. In this work, we introduce FAR, a efficient framework for long-context autoregressive video modeling.

[Figure 2]

- Fig. 2: Illustration of FAR’s Training and Inference Pipeline. In short-video training, a portion of frames is randomly replaced with clean context frames, marked with a unique timestep embedding (e.g., -1) beyond the flow-matching scheduler. In long-video training, we adopt long short-term context modeling. A long-term context window with aggressive patchification is adopted to reduce redundant vision tokens, while a short-term context window is used to model finegrained temporal consistency.

[Figure 3]

- Fig. 3: Visualization of Attention Mask. FAR enables full attention within a frame while maintaining causality at the frame level. In long-context training, we adopt aggressive patchification for long-term context frames to reduce tokens.

[Figure 4]

Fig. 4: Effect of Stochastic Clean Context. This technique eliminate training-inference gap in observed context.

- 3 PRELIMINARY

##### 3.1 Flow Matching

Flow Matching [41]–[43] is a simple alternative objective for training diffusion models. Rather than modeling the reverse process with stochastic differential equations, Flow Matching learns a continuous vector field that deterministically conntect two distribution.

Specifically, given a data sample x0 ∼ pdata(x) and a noise sample x1 ∼ N(0,I), we construct a continuous trajectory connecting them via linear interpolation:

x(t) = (1 − t)x0 + tx1, t ∈ [0,1]. (1) This formulation implies a constant velocity:

dx(t) dt

= v∗ = x1 − x0. (2)

To enable the model to learn the optimal transport between the data and noise distributions, we introduce a learnable time-dependent velocity field vθ(x,t). During training, a

random time t ∼ U(0,1) is sampled, and the model is optimized by minimizing the following objective:

L(θ) = Ex0,x1,t ∥vθ(x(t),t) − v∗∥2 . (3)

3.2 Autoregressive Models

Autoregressive models are a class of probabilistic models where each element in a sequence is conditioned on its preceding elements, denote as context. Formally, given a sequence of tokens (x1,x2,...,xn), an autoregressive model assumes that each token xi is generated based on its previous tokens (x1,x2,...,xi−1). The generative process can be expressed as a factorization of the joint probability:

p(x1,x2,...,xn) =

n

p(xi|x1,x2,...,xi−1). (4)

i=1

By modeling each token conditioned on its preceding tokens, autoregressive models naturally capture the sequential dependencies inherent in data.

[Figure 5]

- Fig. 5: Comparison of FAR and video diffusion transformer. FAR achieves better convergence than video diffusion transformer in unconditional video generation on UCF101.

#### 4 FAR

We first present the FAR framework in Sec. 4.1, followed by training challenges and solutions in Sec. 4.2. Sec. 4.3 analyzes the design for long-context video modeling, and Sec. 4.4 introduces the KV cache for faster inference.

##### 4.1 Framework Overview

Architecture. As shown in Fig. 2 (a), FAR is built upon the diffusion transformer [39], [40]. We adopt the model configuration of DiT [39] and Latte [44], as listed in Table. 1. The key architectural difference between FAR and video diffusion transformers (e.g., Latte [44]) lies in the attention mechanism. As shown in Fig. 3(a), for each frame, we apply causal attention at the frame level while maintaining full attention within each frame. We adopt this causal spatiotemporal attention for all layers, instead of the interleaved spatial and temporal attention used in Latte. In FAR, image generation and image-conditioned video generation are jointly learned thanks to the causal mask, whereas video diffusion transformer [44] requires additional image-video co-training.

Basic Training Pipeline. The training pipeline of FAR is illustrated in Fig. 2 (a). Given a video sequence X, we first employ a pretrained VAE to compress it into the latent space Z ∈ RT×H×W, where T, H, and W denote the number of frames, height, and width of the latent features, respectively. Note that although we primarily adopt an image VAE in this work, FAR can also be trained with a video VAE since our autoregressive unit is the latent frame. Following diffusion forcing [8], we independently sample a timestep for each frame. We then interpolate between the clean latent and the sampled noise using Eq. Eq. 1 and apply the framewise flow matching objective in Eq. Eq. 3 for learning. The key difference between FAR and image flow matching lies in that we adopt causal spatiotemporal attention, allowing each frame to access previous context frames during denoising.

##### 4.2 Short-Video Modeling

Training-Inference Gap in Observed Context. As a hybrid AR-diffusion model, FAR also encounters a traininginference gap in the observed context. As illustrated in

TABLE 2: Comparison of Routes for Long-Context Video Modeling. In test-time extrapolation, FAR is trained on short videos and evaluated on long videos using different extrapolation methods.

Method SSIM↑ PSNR↑ LPIPS↓ FVD↓ Test-Time Extrapolation

Sliding Window 0.365 12.3 0.415 161 Naive RoPE Ext. 0.372 12.2 0.397 396 RIFLEx [5] 0.372 12.2 0.398 391

Long-Video Training

FAR-B-Long 0.576 19.3 0.153 34

Fig. 2(a), each clean latent is fused with sampled noise for the flow matching objective, as defined in Eq. Eq. 1. Consequently, later frames can only access the noised version of previous frames during training. However, during inference, this leads to a distribution shift when clean context frames is used.

As shown in the example in Fig. 4(a), the traininginference gap in the observed context leads to a distribution shift when inferring with a clean context. Although adding mild noise to the context during inference can help mitigate this effect, it still causes low-level flickering, degrading the quality of the generated video. Recent works [26], [27] attempt to address this issue by maintaining a clean copy of the noised sequence during training. However, this approach doubles the training costs.

Our Solution: Stochastic Clean Context. To bridge the gap in observed context, we introduce stochastic clean context for training FAR. As illustrated in Fig. 2(a), we randomly replace a portion of the noised frames with their corresponding clean context and assign them a unique timestep embedding (e.g., -1) beyond the flow-matching timestep scheduler. These clean context frames are excluded from loss computation and are implicitly learned through later frames that use them as context. During inference, this unique timestep embedding guides the model to use clean context effectively. Training FAR with stochastic clean context does not add extra computation and does not conflict with different timestep sampling strategies during training (e.g., logit-normal sampling [45]). It effectively resolves the training-inference discrepancy, as exemplified in Fig. 4(b).

FAR vs. Video Diffusion Transformer. FAR and video diffusion transformer differ only in their training schemes. FAR is trained with independent noise and causal attention, while the video diffusion transformer is trained with uniform noise and full attention. This raises an interesting question: Can FAR surpass video diffusion transformers? To explore this, we convert FAR to video diffusion transformer as a baseline, denoted as Video DiT. We align the training settings to compare the two paradigms. As shown in Fig. 5, FAR achieves better convergence than the Video DiT, demonstrating its potential to become a strong baseline for autoregressive video modeling.

##### 4.3 Long-Context Video Modeling

Test-Time Extrapolation vs. Long-Video Training. Two promising approaches to achieving long-context modeling

[Figure 6]

- Fig. 6: Relation between Token Context Length and Vision Context Length. With the proposed long short-term context modeling, the token context length scales more slowly with increasing vision context length compared to uniform context modeling. When training on long videos, the reduced number of tokens leads to significantly lower training costs and memory usage.

[Figure 7]

- Fig. 7: KV Cache for Short-Video Modeling in FAR. We additionally add a caching step to encode current decoded frame into the KV cache for autoregressive generation.

To eliminate context redundancy, we propose a long shortterm context modeling with asymmetric patchify kernels. As shown in Fig. 2(b), we maintain a high-resolution shortterm context window to capture fine-grained temporal consistency, and a low-resolution long-term context window, where we apply a large patchify kernel to compress context tokens. During training, given that the data has a maximum sequence length of m frames, we fix the short-term context window to n frames and randomly sample the long-term context frames from the range [0,m − n]. The attention mask with long short-term context modeling is shown in Fig. 3(b), where the long-term context uses fewer tokens. As demonstrated in Fig. 6(a), this strategy ensures that increasing the vision context length maintains a manageable token context length. With long short-term context modeling, we can reduce the cost and memory usage of the long-video training significantly, as shown in Fig. 6(b, c). To prevent interference between long-term and short-term contexts, we adopt separate projection layers for each context, inspired by MM-DiT [45]. This approach results in a slightly larger number of parameters, referred to as FAR-Long in Table. 1.

in language modeling are test-time extrapolation [29], [30], [46] and long-sequence fine-tuning [31], [32]. In video modeling, most efforts [4], [5], [15] have focused on test-time extrapolation to generate long videos. However, we question whether test-time extrapolation can effectively solve long-context video modeling. To investigate this, we train FAR on action-conditioned short-video prediction and extend it to long-video prediction using various extrapolation techniques. As summarized in Table. 2, test-time extrapolation results in significantly lower quality than the baseline sliding window approach, leading to poor predictions. Therefore, direct training on long videos may be necessary to achieve effective long-context video modeling.

##### 4.4 Inference-Time KV Cache

KV Cache for Short-Video Modeling. Due to the autoregressive nature of FAR, we can leverage KV-Cache to accelerate inference. As illustrated in Fig. 7, for each frame, we first use the flow-matching schedule to decode it into the clean latent frame. We then introduce an additional caching step to encode the clean latent frame into the KV cache. As discussed in Sec. 4.2, we use timestep t = −1 to denote the clean context frame in the caching step. These KV caches are subsequently used for autoregressive decoding of future frames.

Explosive Token Growth in Long Video Training. Visual data contains redundancy, causing vision tokens to expand much faster than language tokens as context increases. For example, a video sequence of 128 frames requires more than 8K tokens, as illustrated in Fig. 6(a). Consequently, training on long videos becomes computationally prohibitive, as shown in Fig. 6(b, c).

Multi-Level KV Cache for Long-Video Modeling. In longcontext video modeling, we employ long short-term context to reduce redundant visual tokens. To accommodate this, we introduce a multi-level KV cache. As illustrated in Fig. 8, the frames in long-term context window is encoded into L2 cache (4 tokens per frame), while the frames in short-term context window is encoded into L1 cache. When decoding current frame but exceed the short-term context window, the earliest frame in the short-term context window is moved to the long-term context window and encode it into the L2

Our Solution: Long Short-Term Context Modeling. To address explosive token growth in long-video training, we leverage the concept of context redundancy. Specifically, in video autoregression, the current frame depends more heavily on nearby frames to capture local motion consistency, whereas earlier frames primarily serve as context memory.

[Figure 8]

- Fig. 8: Multi-Level KV Cache for Long-Context Video Modeling in FAR. When a frame leaves the short-term context window, we encode it to the L2 cache and re-encode the L1 cache in the window. We then use those encoded KV cache for decoding the current frame. Note that we divide the process into three steps for better illustration, though it can be merged into a single forward pass in implementation.

- TABLE 3: Quantitative Comparison of Conditional and Unconditional Video Generation on UCF-101. We follow the evaluation setup of Latte [44]. † denotes FVD reported on 10,000 videos.

Double Cond. Gen Uncond. Gen

Methods Type Params

Train Cost FVD2048 ↓ FVD2048 ↓

Resolution-128×128 MAGVITv2-MLM [16] Non-AR 307 M ✗ 58† MAGVITv2-AR [16] Token-AR 840 M ✗ 109† TATS [48] Token-AR 331 M ✗ 332 420 FAR-L (Ours) Frame-AR 457 M ✗ 99 (57†) 280

Resolution-256×256 LVDM [49] Video-DiT 437 M ✗ - 372 Latte [44] Video-DiT 674 M ✗ - 478 CogVideo [19] Token-AR 9.4 B ✗ 626 OmniTokenizer [50] Token-AR 650 M ✗ 191 ACDIT [26] Frame-AR 677 M ✓ 111 MAGI [27] Frame-AR 850 M ✓ - 421 FAR-L (Ours) Frame-AR 457 M ✗ 113 303 FAR-XL (Ours) Frame-AR 674 M ✗ 108 279

cache. Since this modifies the cache state, we subsequently re-encode the L1 cache of the frames in the short-term context window. The encoded cache is then used to decode the current frame. Note that in practice, these three steps can be merged into a single forward pass for efficiency.

Main Results. From the results listed in Table. 3, we achieve state-of-the-art performance in both unconditional and conditional video generation. Specifically, Latte [44] is based on video diffusion transformer, while OmniTokenizer [50] is based on Token AR. Our method significantly outperforms both. Furthermore, compared to recent frameautoregressive models [26], [27], which require twice the training cost, FAR achieves superior performance without any additional training cost.

- 5.2.2 Short-Video Prediction

Dataset and Evaluation Settings. We evaluate FAR on the UCF-101 [58] and BAIR [60] datasets, following the evaluation settings in MCVD [51] and ExtDM [52]. We randomly sample 256 videos based on provided context frames, each with 100 different trajectories, and select the best trajectory to compute pixel-wise metrics. For FVD, we report the average over all trajectories.

- Main Results. We summarize the results in Table. 4. Unlike previous works such as MCVD [51] and ExtDM [52], which introduce complex multi-scale fusion strategies and optical flow, FAR achieves superior results on both datasets without requiring additional design.

5.2.3 Long-Video Prediction

Dataset and Evaluation Settings. We benchmark longcontext video modeling results on action-conditioned video prediction using the Minecraft and DMLab datasets [36]. The Minecraft dataset contains approximately 200K videos, while the DMLab dataset contains about 40K videos. Each video consists of 300 frames with action annotations. We follow the evaluation setup in TECO [36], which uses 144 observed context frames to predict 156 future frames and compute pixel metrics. Additionally, we compute FVD on 264 generated frames based on 36 context frames.

- Main Results. We summarize the results in Table. 5. The previous work, TECO [36], adopts aggressive downscaling for all frames to reduce tokens for temporal modeling, creating a trade-off between training efficiency and prediction accuracy. In contrast, FAR employs long short-term context modeling, effectively achieving the lowest prediction error (i.e., LPIPS) without prohibitive computation cost.

- 5 EXPERIMENT

- 5.1 Implementation Details

We follow the DiT’s structure [39] to implement FAR. To compress video latents, we train a series of image DCAE [47] on the corresponding dataset, resulting in 64 tokens per frame. All models are trained from scratch without image pretraining. We provide training hyperparameters and evaluation setting in the Table. 8.

- 5.2 Quantitative Comparison 5.2.1 Video Generation Dataset and Evaluation Settings. We benchmark both unconditional and conditional video generation on the UCF101 dataset [58], which consists of approximately 13,000 videos. Following Latte [44], we use the entire dataset for training. For evaluation, we randomly sample 2,048 videos to compute FVD [59] against the ground-truth videos. For conditional video generation, we set the guidance scale to 2.0 during inference.

- TABLE 4: Quantitative Comparison on Short Video Prediction. We follow the evaluation setup of MCVD [51] and ExtDM [52], where c denotes the number of context frames and p denotes the number of predicted frames.

Methods Params c = 4, p = 12 SSIM↑ PSNR↑ LPIPS↓ FVD↓

RaMViD [53] 235 M 0.639 21.37 0.090 396.7 LFDM [54] 108 M 0.627 20.92 0.098 698.2 MCVD-cp [51] 565 M 0.658 21.82 0.088 468.1 ExtDM-K2 [52] 119 M 0.754 23.89 0.056 394.1

FAR-B (Ours) 130 M 0.818 25.64 0.037 194.1

(a) Evaluation on UCF-101 (64×64)

Methods Params c = 2, p = 14 c = 2, p = 28 SSIM↑ PSNR↑ LPIPS↓ FVD↓ SSIM↑ PSNR↑ LPIPS↓ FVD↓

RaMViD [53] 235 M 0.758 17.55 0.085 166.5 0.691 16.51 0.109 238.7 LFDM [54] 108 M 0.770 17.45 0.084 167.6 0.730 16.68 0.106 276.8 VIDM [55] 194 M 0.763 16.97 0.080 131.7 0.728 16.20 0.096 194.6 MCVD-cp [51] 565 M 0.838 19.10 0.075 87.8 0.797 17.70 0.078 119.0 ExtDM-K4 [52] 121 M 0.845 20.04 0.053 81.6 0.814 18.74 0.069 102.8

FAR-B (Ours) 130 M 0.849 20.87 0.038 99.3 0.819 19.40 0.049 144.3

(b) Evaluation on BAIR (64×64)

- TABLE 5: Quantitative Comparison on Long-Context Video Prediction. We follow the evaluation setup of TECO [36], where c denotes the number of context frames and p denotes the number of predicted frames.

Methods Params c = 144, p = 156 c = 36, p = 264

Methods Params c = 144, p = 156 c = 36, p = 264 SSIM↑ PSNR↑ LPIPS↓ FVD↓

SSIM↑ PSNR↑ LPIPS↓ FVD↓ FitVid [56] 165 M 0.356 12.0 0.491 176 CW-VAE [57] 111 M 0.372 12.6 0.465 125 Perceiver AR [37] 30 M 0.304 11.2 0.487 96 Latent FDM [38] 31 M 0.588 17.8 0.222 181 TECO [36] 169 M 0.703 21.9 0.157 48 FAR-B-Long (Ours) 150 M 0.687 22.3 0.104 64

FitVid [56] 176 M 0.343 13.0 0.519 956 CW-VAE [57] 140 M 0.338 13.4 0.441 397 Perceiver AR [37] 166 M 0.323 13.2 0.441 76 Latent FDM [38] 33 M 0.349 13.4 0.429 167 TECO [36] 274 M 0.381 15.4 0.340 116

FAR-M-Long (Ours) 280 M 0.448 16.9 0.251 39

(b) Evaluation on Minecraft (128×128)

(a) Evaluation on DMLab (64×64)

##### 5.3 Qualitative Comparison

We present a qualitative comparison of long-video prediction in Fig. 9. Compared to previous methods, FAR effectively utilizes the observed context and generates predictions that most closely resemble the ground truth, demonstrating its ability to leverage long-range context.

##### 5.4 Ablation Study

How to decide patchify kernel for remote context? The key to selecting an appropriate patchify kernel lies in whether the compressed latent representation can reliably retain information from past observations. Since patchifying is a spatial-to-channel transformation, we hypothesize that it should satisfy the condition c × c × d ≤ D, where c is the patchify kernel size, d is the latent dimension, and D is the model’s channel dimension. For example, in our experiment, the latent dimension is 32 and the model dimension is 768. Using a patchify kernel of size 4 × 4, we get 4 × 4 × 32 = 512 < 768, which suggests that nearly all input information can be preserved during patchification. As demonstrated in Table. 7, the [4,4] patchify kernel significantly reduces training cost, enabling feasible training on long videos, without sacrificing prediction accuracy compared to a larger [8,8] kernel.

How to decide the local context length? We gradually increase the local context length during training and observe that the performance converges at certain short-term context length (i.e., 8 frames in Fig. 10). Further increasing the local context length significantly raises the training cost without improving performance. This experiment verifies the presence of context redundancy in video autoregressive modeling. Therefore, we select the saturate point as the optimal short-term context length.

Effect of Stochastic Clean Context. We have visualized the effectiveness of stochastic clean context in Fig. 4. Based on the quantitative evaluation of video prediction in Table. 6, FAR with stochastic clean context achieves significantly improved performance.

- TABLE 6: Ablation Study of Stochastic Clean Context (SCC) on UCF-101. Stochastic clean context mitigates the training-inference discrepancy in observed context, leading to improved performance.

Methods c = 1, p = 15 SSIM↑ PSNR↑ LPIPS↓ FVD↓

w/o. SCC 0.540 16.42 0.211 399 w/. SCC 0.596 18.46 0.187 347

- TABLE 7: Ablation Study on the Patchify Kernel of Distant Context. Larger patchify kernels significantly reduce training cost.

Patchify Kernel

Training Memory

SSIM↑ PSNR↑ LPIPS↓ FVD↓

- [1, 1] - - - - OOM

- [2, 2] 0.570 19.1 0.156 38 38.9 G [4, 4] 0.576 19.3 0.153 34 15.3 G [8, 8] 0.558 18.6 0.171 33 0.9 G

Effect of the KV Cache on Inference Speedup. As shown in Fig. 11, the baseline FAR model, which samples without using long short-term context or KV cache requires approximately 1341 seconds to generate a 256-frame video. When the KV cache is introduced, the inference time is significantly reduced to 171 seconds. Finally, by incorporating both long short-term context and the corresponding multi-level KV cache, the sampling time further decreases to approximately 104 seconds. These results demonstrate that KV caching, especially when combined with long shortterm context modeling, significantly improves sampling efficiency for long video generation.

#### 6 CONCLUSION

In this paper, we systematically investigate long-context video modeling using the proposed Frame Autoregressive

[Figure 9]

- Fig. 9: Qualitative Comparison of Long-Context Video Prediction on DMLab. FAR fully utilizes the long-range context (144 frames), resulting in more consistent prediction (156 frames) compared to previous methods.

[Figure 10]

- Fig. 10: Ablation Study of the Short-Term Context Window Size. Performance saturates as the window size increases.

[Figure 11]

Fig. 11: Ablation Study of the KV Cache. FAR-Long with proposed multi-level KV cache achieves the best speedup on long videos.

Model (FAR). First, We show that direct test-time extrapolation is insufficient for effective long-context video modeling, highlighting the necessity of efficient long-video training. We then identify context redundancy as a key bottleneck in video autoregression. To address this, we propose a long short-term context modeling strategy with asymmetric patchify kernels, a simple yet effective method to eliminate redundant context and significantly reduce the training cost for long videos. Extensive experiments validate the effectiveness of FAR in handling long-context video generation and highlight a promising direction for the evolution of next-generation video generative models—shifting the focus from short-term temporal consistency to long-term world modeling.

Limitations. The primary limitation lies in the lack of

scaled-up experiments. Although FAR demonstrates great potential, we still lack large-scale training on text-to-video generation datasets. Additionally, restricted by the available datasets, we only experiment with FAR on up to 300 frames (about 20 seconds), not fully investigating its ability on minute-level videos.

Future Work. One future direction is to scale up FAR and benchmark it against video diffusion transformers on largescale text-to-video generation tasks. Additionally, we plan to simulate a longer video dataset (on the minute level) to better evaluate the model’s long-context capabilities. Finally, it would be interesting to explore whether FAR’s longcontext modeling can enable video-level in-context learning.

#### REFERENCES

- [1] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh, “Video generation models as world simulators,” 2024. [Online]. Available: https://openai. com/research/video-generation-models-as-world-simulators 1, 2
- [2] A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025. 1
- [3] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding et al., “Cosmos world foundation model platform for physical ai,” arXiv preprint arXiv:2501.03575, 2025. 1
- [4] Y. Lu, Y. Liang, L. Zhu, and Y. Yang, “Freelong: Training-free long video generation with spectralblend temporal attention,” arXiv preprint arXiv:2407.19918, 2024. 1, 2, 5
- [5] M. Zhao, G. He, Y. Chen, H. Zhu, C. Li, and J. Zhu, “Riflex: A free lunch for length extrapolation in video diffusion transformers,” arXiv preprint arXiv:2502.15894, 2025. 1, 4, 5
- [6] Y. Guo, C. Yang, Z. Yang, Z. Ma, Z. Lin, Z. Yang, D. Lin, and L. Jiang, “Long context tuning for video generation,” arXiv preprint arXiv:2503.10589, 2025. 1
- [7] K. Dalal, D. Koceja, G. Hussein, J. Xu, Y. Zhao, Y. Song, S. Han, K. C. Cheung, J. Kautz, C. Guestrin et al., “One-minute video generation with test-time training,” arXiv preprint arXiv:2504.05298,

2025. 1

- [8] B. Chen, D. Mart´ı Mons´o, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann, “Diffusion forcing: Next-token prediction meets full-sequence diffusion,” Advances in Neural Information Processing Systems, vol. 37, pp. 24081–24125, 2025. 1, 2, 4
- [9] Y. Jin, Z. Sun, N. Li, K. Xu, H. Jiang, N. Zhuang, Q. Huang, Y. Song, Y. Mu, and Z. Lin, “Pyramidal flow matching for efficient video generative modeling,” arXiv preprint arXiv:2410.05954, 2024. 1, 2
- [10] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou, “Show-o: One single transformer to unify multimodal understanding and generation,” arXiv preprint arXiv:2408.12528, 2024. 1, 2
- [11] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024. 2
- [12] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang et al., “Hunyuanvideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024. 2
- [13] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai, “Animatediff: Animate your personalized textto-image diffusion models without specific tuning,” arXiv preprint arXiv:2307.04725, 2023. 2
- [14] J. Xing, M. Xia, Y. Zhang, H. Chen, W. Yu, H. Liu, G. Liu, X. Wang, Y. Shan, and T.-T. Wong, “Dynamicrafter: Animating open-domain images with video diffusion priors,” in European Conference on Computer Vision. Springer, 2024, pp. 399–417. 2
- [15] F.-Y. Wang, W. Chen, G. Song, H.-J. Ye, Y. Liu, and H. Li, “Genl-video: Multi-text to long video generation via temporal codenoising,” arXiv preprint arXiv:2305.18264, 2023. 2, 5
- [16] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, V. Birodkar, A. Gupta, X. Gu et al., “Language model beats diffusion–tokenizer is key to visual generation,” arXiv preprint arXiv:2310.05737, 2023. 2, 6
- [17] Y. Gu, X. Wang, Y. Ge, Y. Shan, and M. Z. Shou, “Rethinking the objectives of vector-quantized tokenizers for image synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7631–7640. 2
- [18] D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, G. Schindler, R. Hornung, V. Birodkar, J. Yan, M.-C. Chiu et al., “Videopoet: A large language model for zero-shot video generation,” arXiv preprint arXiv:2312.14125, 2023. 2
- [19] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang, “Cogvideo: Large-scale pretraining for text-to-video generation via transformers,” arXiv preprint arXiv:2205.15868, 2022. 2, 6
- [20] L. Fan, T. Li, S. Qin, Y. Li, C. Sun, M. Rubinstein, D. Sun, K. He, and Y. Tian, “Fluid: Scaling autoregressive text-to-image generative models with continuous tokens,” arXiv preprint arXiv:2410.13863,

2024. 2

- [21] T. Li, Y. Tian, H. Li, M. Deng, and K. He, “Autoregressive image generation without vector quantization,” Advances in Neural Information Processing Systems, vol. 37, pp. 56424–56445, 2025. 2
- [22] C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy, “Transfusion: Predict the next token and diffuse images with one multi-modal model,” arXiv preprint arXiv:2408.11039, 2024. 2
- [23] Y. Ma, X. Liu, X. Chen, W. Liu, C. Wu, Z. Wu, Z. Pan, Z. Xie, H. Zhang, L. Zhao et al., “Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation,” arXiv preprint arXiv:2411.07975, 2024. 2
- [24] L. Barrault, P.-A. Duquenne, M. Elbayad, A. Kozhevnikov, B. Alastruey, P. Andrews, M. Coria, G. Couairon, M. R. Costa-juss`a, D. Dale et al., “Large concept models: Language modeling in a sentence representation space,” arXiv e-prints, pp. arXiv–2412,

2024. 2

- [25] T. Wu, Z. Fan, X. Liu, H.-T. Zheng, Y. Gong, J. Jiao, J. Li, J. Guo, N. Duan, W. Chen et al., “Ar-diffusion: Auto-regressive diffusion model for text generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 39957–39974, 2023. 2
- [26] J. Hu, S. Hu, Y. Song, Y. Huang, M. Wang, H. Zhou, Z. Liu, W.-Y. Ma, and M. Sun, “Acdit: Interpolating autoregressive conditional modeling and diffusion transformer,” arXiv preprint arXiv:2412.07720, 2024. 2, 4, 6
- [27] D. Zhou, Q. Sun, Y. Peng, K. Yan, R. Dong, D. Wang, Z. Ge, N. Duan, X. Zhang, L. M. Ni et al., “Taming teacher forcing for masked autoregressive video generation,” arXiv preprint arXiv:2501.12389, 2025. 2, 4, 6
- [28] O. Press, N. A. Smith, and M. Lewis, “Train short, test long: Attention with linear biases enables input length extrapolation,” arXiv preprint arXiv:2108.12409, 2021. 2
- [29] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, “Yarn: Efficient context window extension of large language models,” arXiv preprint arXiv:2309.00071, 2023. 2, 5
- [30] bloc97, “NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any finetuning and minimal perplexity degradation.” 2023. [Online]. Available: https://www.reddit.com/r/LocalLLaMA/comments/ 14lz7j5/ntkaware scaled rope allows llama models to have/ 2, 5

- [31] S. Chen, S. Wong, L. Chen, and Y. Tian, “Extending context window of large language models via positional interpolation,” arXiv preprint arXiv:2306.15595, 2023. 2, 5
- [32] Y. Chen, S. Qian, H. Tang, X. Lai, Z. Liu, S. Han, and J. Jia, “Longlora: Efficient fine-tuning of long-context large language models,” arXiv preprint arXiv:2309.12307, 2023. 2, 5
- [33] D. Valevski, Y. Leviathan, M. Arar, and S. Fruchter, “Diffusion models are real-time game engines,” arXiv preprint arXiv:2408.14837, 2024. 2
- [34] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps et al., “Genie: Generative interactive environments,” in Forty-first International Conference on Machine Learning, 2024. 2
- [35] J. Parker-Holder, P. Ball, J. Bruce, V. Dasagi, K. Holsheimer, C. Kaplanis, A. Moufarek, G. Scully, J. Shar, J. Shi, S. Spencer, J. Yung, M. Dennis, S. Kenjeyev, S. Long, V. Mnih, H. Chan, M. Gazeau, B. Li, F. Pardo, L. Wang, L. Zhang, F. Besse, T. Harley, A. Mitenkova, J. Wang, J. Clune, D. Hassabis, R. Hadsell, A. Bolton, S. Singh, and T. Rockt¨aschel, “Genie 2: A large-scale foundation world model,” 2024. [Online]. Available: https://deepmind.google/discover/blog/ genie-2-a-large-scale-foundation-world-model/ 2
- [36] W. Yan, D. Hafner, S. James, and P. Abbeel, “Temporally consistent transformers for video generation,” in International Conference on Machine Learning. PMLR, 2023, pp. 39062–39098. 2, 6, 7, 11
- [37] C. Hawthorne, A. Jaegle, C. Cangea, S. Borgeaud, C. Nash, M. Malinowski, S. Dieleman, O. Vinyals, M. Botvinick, I. Simon et al., “General-purpose, long-context autoregressive modeling with perceiver ar,” in International Conference on Machine Learning. PMLR, 2022, pp. 8535–8558. 2, 7
- [38] W. Harvey, S. Naderiparizi, V. Masrani, C. Weilbach, and F. Wood, “Flexible diffusion modeling of long videos,” Advances in Neural Information Processing Systems, vol. 35, pp. 27953–27965, 2022. 2, 7
- [39] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205. 2, 4, 6

- [40] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. VandenEijnden, and S. Xie, “Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers,” in European Conference on Computer Vision. Springer, 2024, pp. 23–40. 2, 4
- [41] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” arXiv preprint

- arXiv:2209.03003, 2022. 3

[42] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint

- arXiv:2210.02747, 2022. 3

- [43] M. S. Albergo and E. Vanden-Eijnden, “Building normalizing flows with stochastic interpolants,” arXiv preprint arXiv:2209.15571, 2022. 3
- [44] X. Ma, Y. Wang, G. Jia, X. Chen, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao, “Latte: Latent diffusion transformer for video generation,” arXiv preprint arXiv:2401.03048, 2024. 4, 6, 11
- [45] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller,¨ H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first international conference on machine learning, 2024. 4, 5
- [46] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “Roformer: Enhanced transformer with rotary position embedding,” Neurocomputing, vol. 568, p. 127063, 2024. 5
- [47] J. Chen, H. Cai, J. Chen, E. Xie, S. Yang, H. Tang, M. Li, Y. Lu, and S. Han, “Deep compression autoencoder for efficient highresolution diffusion models,” arXiv preprint arXiv:2410.10733, 2024. 6, 11
- [48] S. Ge, T. Hayes, H. Yang, X. Yin, G. Pang, D. Jacobs, J.-B. Huang, and D. Parikh, “Long video generation with time-agnostic vqgan and time-sensitive transformer,” in European Conference on Computer Vision. Springer, 2022, pp. 102–118. 6
- [49] Y. He, T. Yang, Y. Zhang, Y. Shan, and Q. Chen, “Latent video diffusion models for high-fidelity long video generation,” arXiv preprint arXiv:2211.13221, 2022. 6
- [50] J. Wang, Y. Jiang, Z. Yuan, B. Peng, Z. Wu, and Y.-G. Jiang, “Omnitokenizer: A joint image-video tokenizer for visual generation,” arXiv preprint arXiv:2406.09399, 2024. 6
- [51] V. Voleti, A. Jolicoeur-Martineau, and C. Pal, “Mcvd-masked conditional video diffusion for prediction, generation, and interpolation,” Advances in neural information processing systems, vol. 35, pp. 23371–23385, 2022. 6, 7
- [52] Z. Zhang, J. Hu, W. Cheng, D. Paudel, and J. Yang, “Extdm: Distribution extrapolation diffusion model for video prediction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19310–19320. 6, 7, 11
- [53] T. H¨oppe, A. Mehrjou, S. Bauer, D. Nielsen, and A. Dittadi, “Diffusion models for video prediction and infilling,” arXiv preprint arXiv:2206.07696, 2022. 7
- [54] H. Ni, C. Shi, K. Li, S. X. Huang, and M. R. Min, “Conditional image-to-video generation with latent flow diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18444–18455. 7
- [55] K. Mei and V. Patel, “Vidm: Video implicit diffusion models,” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 8, 2023, pp. 9117–9125. 7
- [56] M. Babaeizadeh, M. T. Saffar, S. Nair, S. Levine, C. Finn, and D. Erhan, “Fitvid: Overfitting in pixel-level video prediction,” arXiv preprint arXiv:2106.13195, 2021. 7
- [57] V. Saxena, J. Ba, and D. Hafner, “Clockwork variational autoencoders,” Advances in Neural Information Processing Systems, vol. 34, pp. 29246–29257, 2021. 7
- [58] K. Soomro, A. R. Zamir, and M. Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” arXiv preprint arXiv:1212.0402, 2012. 6
- [59] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Fvd: A new metric for video generation,” 2019. 6
- [60] F. Ebert, C. Finn, A. X. Lee, and S. Levine, “Self-supervised visual planning with temporal skip connections.” CoRL, vol. 12, no. 16, p. 23, 2017. 6

7 APPENDIX

- 7.1 Experimental Settings

As shown in Table. 8, we list the detailed training and evaluation configurations of FAR. For the ablation study in this paper, we halve the training iterations while keeping other settings the same.

- 7.2 Qualitative Comparison

We provide additional visualization of long-video prediction results on DMLab and Minecraft in Fig. 12 and Fig. 13. From the results, FAR better exploits the provided context and provides more consistent results in later predictions compared to previous works.

TABLE 8: Experimental Configurations of FAR. We follow the evaluation settings from Latte [44], MCVD [52], and TECO [36].

Short-Video Generation Short-Video Prediction Long-Video Prediction Cond. UCF-101 Uncond. UCF-101 BAIR UCF-101 Minecraft DMLab

Hyperparameters

Dataset Configuration

Resolution 256/128 256/128 64 64 128 64 Total Training Samples 13,320 13,320 43,264 9,624 194,051 39,375

###### Training Configuration

Training Cost (H100 Days) 12.7 12.7 2.6 3.6 18.2 17.5 Batch Size 32 32 32 32 32 32 Latent Size 8×8 (DC-AE [47]) 8×8 (DC-AE [47]) 8×8 (DC-AE [47]) 8×8 (DC-AE [47]) 8×8 (DC-AE [47]) 8×8 (DC-AE [47]) Training Sequence Length 16 16 32 16 300 300 LR 1 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 LR Schedule constant constant constant constant constant constant Warmup Steps - - - - 10K 10K Total Training Steps 400K 400K 200K 200K 1M 1M Stochastic Clean Context 0.1 0.1 0.1 0.1 0.1 0.1 Short-Term Context Window 16 16 32 16 16 16 Patchify Kernel for Distant Context - - - - [4, 4] [4, 4]

###### Evaluation Configuration

Samples 4×2048 4×2048 100×256 100×256 4×256 4×256 Guidance Scale 2.0 - - - 1.5 1.5 Reference Work Latte [44] Latte [44] MCVD [52] MCVD [52] TECO [36] TECO [36]

[Figure 12]

- Fig. 12: Qualitative Comparison of Long-Context Video Prediction on DMLab. FAR fully utilizes the long-range context (144 frames), resulting in more consistent prediction (156 frames) compared to previous methods.

[Figure 13]

###### Fig. 13: Qualitative Comparison of Long-Context Video Prediction on Minecraft. FAR fully utilizes the long-range context (144 frames), resulting in more consistent prediction (156 frames) compared to previous methods.

