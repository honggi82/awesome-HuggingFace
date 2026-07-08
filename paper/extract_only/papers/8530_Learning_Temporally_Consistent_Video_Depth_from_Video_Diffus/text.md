arXiv:2406.01493v4[cs.CV]7Jun2025

# Learning Temporally Consistent Video Depth from Video Diffusion Priors

Jiahao Shao1* Yuanbo Yang1* Hongyu Zhou1 Youmin Zhang2,6 Yujun Shen4 Vitor Guizilini5 Yue Wang3 Matteo Poggi2 Yiyi Liao1†

1Zhejiang University 2University of Bologna 3University of Southern California 4Ant Group 5Toyota Research Institute 6Rock Universe AI

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

Input Video Marigold

Ours

- Figure 1. We present ChronoDepth, a novel method derived from video diffusion model. It can estimate video depth from arbitrarylength open-world videos while exhibiting high spatial accuracy and state-of-the-art temporal consistency.

## Abstract

This work addresses the challenge of streamed video depth estimation, which expects not only per-frame accuracy but, more importantly, cross-frame consistency. We argue that sharing contextual information between frames or clips is pivotal in fostering temporal consistency. Therefore, we reformulate depth prediction into a conditional generation problem to provide contextual information within a clip and across clips. Specifically, we propose a consistent contextaware training and inference strategy for arbitrarily long videos to provide cross-clip context. We sample independent noise levels for each frame within a clip during training while using a sliding window strategy and initializing overlapping frames with previously predicted frames without adding noise. Moreover, we design an effective training strategy to provide context within a clip. Extensive experimental results validate our design choices and demonstrate the superiority of our approach, dubbed ChronoDepth. Project page: xdimlab.github.io/ChronoDepth.

## 1. Introduction

Monocular image depth estimation methods have achieved significant advancements in recent years. Despite notable

∗ denotes equal contribution. † corresponding author.

progresses in spatial precision [19, 22, 27, 40, 55, 56, 74, 75, 79, 82], these single-image depth estimation methods are built based on an i.i.d. assumption between frames. As there is no contextual information before or after each frame, this approach is inherently prone to flickering and temporal inconsistency. In the meantime, many applications in AR/VR, robotics, and video editing require an estimation of consistent depth over time, which calls for video depth estimation tools that exhibit both spatial accuracy and temporal consistency.

For arbitrarily long videos, given the purpose of autoregressive inference and the limitation of computing resources, it is necessary to split it into multiple smaller video clips to process them in a streamed manner. Existing methods [69, 73] leverage the attention mechanism in a generalizable manner to aggregate contextual information for each video clip. However, such approaches cannot fully encode contextual information between clips. Therefore, flickering issues still persist across clips. Another line of methods leverages test-time training (TTT) paradigms [43, 48, 64, 70, 85] to alleviate this issue, wherein they fine-tune a single-image depth model on the testing videos and inject temporal context through test-time optimization. However, these approaches heavily hinge on precise camera poses and require a long optimization time.

Recent advancements in video generative models [8, 9, 15, 31] have demonstrated significant growth, achieving

[Figure 13]

- Figure 2. Illustration of different inference strategies for infinitely long videos. (a) Naive sliding window inference; (b) Inference with replacement trick; (c) Our proposed consistent context-aware inference. F, W denotes number of clip and overlapping frames respectively.

high spatial quality and temporal consistency. We argue video diffusion models can be repurposed for video depth estimation in the wake of diffusion-based single-image depth estimatiors [22, 40], yielding feed-forward estimation without requiring camera poses across frames. This repurposing process, however, is not merely a straightforward extension of monocular depth prediction. It requires careful consideration of the contextual information between clips to ensure temporal consistency across a long video.

Therefore, we propose a novel video depth estimator, named ChronoDepth, to leverage a video diffusion model to provide context cues within and between clips, resulting in much higher temporal consistency compared to the use of single-image diffusion models – see Fig. 1, as highlighted by the smoother depth depicted by extracting y-t slices from the results by our model against a state-ofthe-art single-image model [40]. Specifically, to effectively model context within a single clip with good generalization, we finetune video diffusion models, drawing inspiration from recent efforts that leverage image generative model priors for single-frame tasks [17, 18, 22, 27, 37, 40, 59, 74, 87]. In terms of cross-clip context, we systematically evaluate different strategies, and propose a consistent contextaware inference as shown in Fig. 2. At inference time, processing video through a sliding window without carrying context across frames (Fig. 2 (a)) does not ensure temporal consistency. A common method to add context between clips is the “replacement trick” in video diffusion models [31, 88], used in our concurrent work DepthCrafter [33]. This approach involves initializing overlapping frames by adding noise to previously predicted depth frames (Fig. 2 (b)), but this leads to inconsistent contextual information due to noise variations [31]. To address this, our proposed strategy uses previously predicted depth frames for context without adding noise, ensuring consistency across frames (Fig. 2 (c)). This is enabled by applying different noise levels to each frame during training, allowing the model to denoise across varying noise levels.

Furthermore, we conduct thorough comparisons between various protocols to train ChronoDepth, empirically finding an effective training strategy fully exploring image and video depth datasets, identifying a sequential training of spatial and temporal layers as the most effective one, where the spatial layers are first trained and kept

frozen during the training of the temporal layers.

Quantitative and qualitative results on open-world video benchmarks confirm that ChronoDepth achieves stateof-the-art temporal consistency on video depth estimation, while maintaining spatial accuracy comparable with stateof-the-art single-image depth estimation methods.

## 2. Related Work

Discriminative Monocular Depth Estimation. These approaches are trained end-to-end to regress depth, according to two alternative categories: metric versus relative depth estimation. Early attempts focused on the former category [20], yet were limited to single scenarios – i.e., training specialized models for driving environments [25] or indoor scenes [61]. Among these, some frameworks used ordinal regression [21], local planar guidance [44], adaptive bins [6] or fully-connected CRFs [81]. In pursuit of achieving generalizability across different environments [55], the community recently shifted towards training models estimating relative depth, through the use of affine-invariant loss functions [55, 56] to be robust against different scale and shift factors across diverse datasets. On this track, the following efforts focused on recovering the real 3D shape of scenes [79], exploiting surface normals as supervision [78], improving high-frequency details [49, 82] or exploiting procedural data generation [19]. To combine the best of the two worlds, new attempts to learn generalizable, metric depth estimation have emerged lately [7, 45], either explicitly handling camera parameters through canonical transformations [32, 80] or as direct prompts to the model [29, 53].

Generative Monocular Depth Estimation. Eventually, some methodologies have embraced the use of pre-trained generative models for depth estimation. Some exploited Low-Rank Adaptation (LoRA) [17] or started from selfsupervised pre-training [59], while others re-purposed Latent Diffusion Models (LDM) by fine-tuning the pre-trained UNet [40] to denoise depth maps. Further advances of this latter strategy jointly tackled depth estimation and surface normal prediction [22], or exploited Flow Matching [27] for higher efficiency. Despite the strong priors learned by diffusion models allow for capturing intricate details in complex environments, these frameworks expose scarce temporal consistency and flickering artifacts over frame sequences.

#### Video Depth Estimation. In addition to spatial ac-

curacy, temporal consistency is paramount when predicting depth from videos. This task has been mostly tackled through discriminative methods: some approaches estimate the poses of any frame in the video and use them to build cost volumes [28, 47, 71] or running test-time training [43, 48, 85], with both heavily depending on the accuracy of the poses and the latter lacking generalizability as it overfits a single video. Others deploy recurrent networks [65, 83], while most recent works exploit attention mechanisms [13, 69], yet with sub-optimal results compared to state-of-the-art single-image depth predictors. NVDS [70] introduces a stabilization network so as to conduct temporal refinement to single-frame results from an off-the-shelf depth predictor. Concurrently with us, DepthCrafter [33] proposes a generative framework for video depth estimation, using latent interpolation based on the replacement trick to enforce temporal consistency over long clips, leading to inconsistency due to noise variation.

Diffusion Models. Image Diffusion Models (IDMs) by Sohl-Dickstein et al. [62] conquered the main stage for image generation tasks [16, 42] at the expense of GANs. Further developments aimed at improving both generation conditioning [1, 51] and computational efficiency [58], with Latent Diffusion Models (LDMs) [58] notably emerging as a solution for the latter. Among conditioning techniques, the use of cross-attention [2, 10, 23, 30, 39, 41, 51, 52, 54] and the encoding of segmentation masks into tokens [3, 23] stand as the most popular, with additional schemes being proposed to enable the generation of visual data conditioned by diverse factors such as text, images, semantic maps, sketches, and other representations [4, 5, 34, 50, 67, 84]. Lately, DMs have been extended for video generation [8, 14, 31, 36, 72, 86], focusing on obtaining consistent content over time thanks to the integration of temporal layers, incorporating self-attention mechanisms and convolutions between frames within conventional image generation frameworks. One line of work [15, 24, 31] explores infinite video generation via adapting both training and sampling paradigms based on the original diffusion model. Diffusion Forcing [15] is mostly similar with ChronoDepth, yet with notable differences as it focuses on generation task while ChronoDepth focuses on video depth estimation, a deterministic task. Furthermore, Diffusion Forcing is instantiated with causal architectures and implemented with a Recurrent Neural Network (RNN), where they maintain hidden states to capture the information of past tokens. However, ChronoDepth is implemented with an attention network without additional hidden states, directly fusing the information of past tokens with current noisy input.

## 3. Method

In this section, we introduce ChronoDepth, a consistent video depth estimator derived from a video foundation

model, specifically from Stable Video Diffusion (SVD) [8]. Given an arbitrary-length video, our goal is to generate spatial-accurate and temporal-consistent video depth. We first introduce the straightforward way of repurposing SVD for video depth estimation in Sec. 3.1 and discuss its issues in artitray-length video inference. Next, we elaborate on our consistent context-aware strategy in Sec. 3.2 and our training protocols in Sec. 3.3.

### 3.1. Naive Diffusion Formulation

In order to align with the video foundation model, SVD [8], a straightforward way is to follow [40] and reformulate monocular video depth estimation as a continuous-time denoising diffusion [38, 63] generation task conditioned on the RGB video. The diffusion model consists of a stochastic forward pass to inject one noise level Gaussian noise into the input sequence and a reverse process to remove noise with a learnable denoiser Dθ. Following SVD, our diffusion model is defined in a latent space of lower spatial dimension for better computation efficiency, where a variational autoencoder (VAE) consists of an encoder E and a decoder D is used to compress the original signals. In order to repurpose the VAE of SVD, which accepts a 3-channel (RGB) input, we replicate the depth map into three channels to mimic an RGB image. Then, during inference, we decode and calculate the average of these three channels, which serves as our predicted depth map following [40].

Training. During training, one F-frame RGB video clip x ∈ RF×W×H×3 and the corresponding depth d ∈ RF×W×H are first encoded into the latent space with the

VAE encoder: z(x) = E(x),z(0d) = E(d). For each training step, we sample one noise level σt for the whole clip , where log σt ∼ N(Pmean,Pstd2 ) [38] with Pmean = 0.7 and Pstd = 1.6. Then we add noise with this noise level to the depth latent z(0d) to obtain the noisy depth latent z(td) as

z(td) = z(0d) + σt2ϵ, ϵ ∼ N(0,I). (1) In the reverse process, diffusion model denoises z(td) towards predicted clean zˆ(0d) with a learnable denoiser Dθ as

zˆ(0d) = Dθ(z(td);σt,z(x)), (2) which is trained via denoising score matching (DSM) [66]

L = Ez(d),z(x),σt λ(σt)∥zˆ(0d) − z(0d)∥22 , (3)

with weighting function λ(σ) = (1 + σ2)σ−2. In this work we follow EDM [38] and parametrize the denoiser Dθ as

Dθ(z(td);σt,z(x)) = cskip(σt)z(td)+ cout(σt)Fθ(cin(σt)z(td);cnoise(σt),z(x)),

(4)

where Fθ is a UNet to be trained in our case, and cskip, cout, cin, and cnoise are preconditioning functions. The condition,

[Figure 14]

F frames

RGB Video Conditioning

Sequential Spatial-Temporal Fine-Tune

* : concat

[Figure 15]

[Figure 16]

: frozen

[Figure 17]

[Figure 18]

: trainable

Spatial

Temporal

[Figure 19]

F frames

Video Depth Estimator Training

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

add noise

*

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- Figure 3. Our adapted training pipeline. We add an RGB video conditioning branch to a pre-trained video diffusion model (SVD) and fine-tune it via DSM for depth estimation, by sampling different noise levels for each frame. Our training involves two stages: 1) we train the spatial layers with single-frame depths; 2) we freeze spatial layers and train the temporal layers on clips of random lengths.

RGB video latent z(x), in this formulation is introduced via concatenation with depth video latent on feature dimension. Single-Clip Inference. At testing time, single-clip depth

latent zˆ(0d) is restored from a randomly-sampled Gaussian noise zˆ(Td) conditioning on the RGB clip by iteratively applying the denoising process with trained denoiser Dθ

zˆT(d) ∼ N(0,σT2 I) (5) z˜(0d) = Dθ(z(td);σt,z(x)) (6)

zˆ(td) − z˜(0d) σt

zˆ(t−d)1 =

(σt−1 − σt) + zˆ(td), 0 < t ≤ T (7)

where σ0,...,σT are sampled from a fixed variance schedule of a denoising process with T steps. Video depth dˆ could

be obtained with the VAE decoder dˆ = D(zˆ(0d)).

Arbitrary-Length Video Inference. Based on the purpose of autoregressive inference and the limitation of computing resources, it is necessary to split a long video into smaller video clips and process them in streamed manner. A solution could consists of using a naive sliding window strategy: the given video sequence is split into multiple independent video clips with overlap W. The model makes independent inferences for each video clip: the whole clip is initialized via Eq. (5) and updated via Eq. (7) ( see Algorithm 1). The

depth latent is sampled from pθ(zˆ(t−dW1 :F)|zˆt(d),x), therefore without any contextual information exchange between clips, causing flickering artifacts.

Inspired by recent advancements in video generative models [31], a straightforward method to share contextual information across clips involves conditioning via replacement trick – i.e., employing a sliding window strategy and initializing overlapping frames by adding noise to previ-

ously predicted depth frames. Let zˆ0(d0:W) denote the latent code of W previously predicted depth frames. The first W frames are obtained via forward diffuse as below

zˆ(td0:W) = zˆ(0d0:W) + σtϵ, ϵ ∼ N(0,I), 0 < t ≤ T, (8)

and the last F −W frames are initialized via Eq. (5) and updated via Eq. (7), where zˆ(td) = z ˆ(td0:W),zˆ(tdW:F) (see Algorithm 2). In this way, the content is injected to some extent. However, Ho et al. [31] highlights that this conditioning approach lacks mathematical rigor and results in inconsistent contextual information during diffusion sampling at each step (see Appendix B). The depth latent currently is sampling from pθ(zˆ(t−dW1 :F)|zˆ(td0:W),zˆ(tdW:F),x). Due to the noise added to zˆ(td0:W), the conditioning varies during diffusion sampling at each step causing discrepancies in the generated sequence. Consequently, when this trick is applied to video depth tasks, it exacerbates inconsistency across clips, manifesting both on geometry and scale.

### 3.2. Consistent Context-Aware Strategy

The issue with the replacement trick is that, at each sampling step, the conditioning changes due to the varying noise added each time, leading to fluctuating guidance. Ideally, the previous inference results should be provided without noise to guide the prediction of remaining frames. Based on this intuition, we propose a novel consistent context-aware strategy, inspired by [15]. During training, instead of sampling a single noise level for the entire video clip, we independently sample distinct noise levels for each individual frame within the clip. During inference, we initialize overlapping frames with previously predicted depth and adjust the conditioning noise levels to each frame.

Training. We independently sample distinct noise levels for each individual frame within the clip as σt , where σt = [σ1,σ2,...,σF], log σi ∼ N(Pmean,Pstd2 ) for i = 1,2,...,F. Then we add these F noise levels to the depth latent z(0d) to obtain the noisy depth latent z(td) as

z(td) = z(0d) + σ2tϵ, ϵ ∼ N(0,I). (9) In the reverse process, we correspondingly adapt the previ-

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

* : concat : frozen

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Standard Noise

* * Denoise

[Figure 44]

- Figure 4. Inference pipeline of our consistent context-aware inference strategy. Given an infinitely long video, we segment it into several F-frame clips with overlap W and use a sliding window strategy for inference. Besides, we initialize overlapping W frames with previously predicted depth frames zˆ(0d0:W) to support consistent context information during each sampling step. ous Eqs. (2) to (4) as follows

critical role in generative-based video depth estimation. Our overall inference strategy is depicted in Fig. 4.

zˆ0(d) = Dθ(zt(d);σt,z(x)), (10) L = Ez(d),z(x),σt λ(σt)∥zˆ0(d) − z(0d)∥22 , (11)

### 3.3. Multi-Stage Training Protocol

In contrast to image-generative models, video-generative models incorporate both spatial and temporal dimensions. One central question thereby becomes how to effectively fine-tune these pre-trained video generative models to achieve a satisfying level of spatial accuracy and temporal coherence in geometry. In response to this, we have conducted comprehensive analyses to ascertain the best practices for taming the initial foundational video model into a consistent depth estimator. Note that the VAE for compressing the original pixel space remains frozen.

Dθ(z(td);σt,z(x)) = cskip(σt)z(td)+ cout(σt)Fθ(cin(σt)z(td);cnoise(σt),z(x)).

(12)

Fig. 3 contains an overview of our adapted training pipeline. Arbitrary-length video inference. There is no previously predicted depth available for the first clip, though it is available for the subsequent clips. Therefore we perform inference of the first clip using the standard inference strategy as Eqs. (5) to (7). For subsequent clips, we initialize overlapping W frames with previously predicted depth frames

Using Single-Frame Datasets. The availability of singleframe depth datasets is generally higher than the one of video depth datasets. These latter usually count more frames compared to the former, yet they feature a reduced number of scenes and thus lack diversity. Accordingly, we argue that jointly using single-frame and multi-frame depth datasets can play a significant role in achieving good spatial and temporal accuracy. Therefore, we also make use of the single-frame datasets throughout the full training process, in contrast to SVD [8] that uses single-view image datasets [60] only during the first pre-training stage.

zˆ(0d0:W) without adding any noise, and the last F − W frames are initialized via Eq. (5). Note that the noise con-

dition of the denoiser Dθ is changed correspondingly as σt = [σϵ,σϵ,...,σϵ

] and σϵ indicates a very

,σt,σt,...,σt F−W

W

small noise level (see Algorithm 3 for pseudocode). Training with our adapted pipeline has empowered our model to effectively denoise different frames within one clip under different noise levels. In this way, the depth latent is sam-

pling from pθ(zˆt(−dW1 :F)|zˆ(0d0:W),zˆ(tdW:F),x). This method ensures that the contextual information between clips remains consistent across any sampling step. Moreover, the rationale behind conditioning previously predicted depth frames with a small noise level rather than a clean noise level is that such depth frames are not ground-truth, so they cannot be fully trusted. This small noise level accounts for the inherent uncertainty from the previous inference and mitigates long-term compounding errors, thereby enhancing the robustness of our model. We provide experimental results of this small noise level in Appendix D. This idea is also explored in a concurrent work, Diffusion Forcing [15], focusing on generative modeling, whereas we uncover its

Randomly Sampled Clip Length. As introduced before, splitting the video into fixed-length clips could be a natural choice at inference time to maintain consistent resource requirements. However, a fixed-length clip could involve slower or faster motion, making it harder for the model to generalize. Accordingly, we argue that sampling clips of random length at training time can act as a form of data augmentation, making the model more robust to such different behaviors. Both fixed and random length choices are naturally supported by design, thanks to the spatial layers interpreting the video as a batch of independent images and temporal layers applying both convolutions and temporal

Algorithm 1 Naive Sliding Window

Algorithm 2 Replacement trick

Algorithm 3 Consistent Context-Aware

- 1: Input: Model θ, rgb latent z(x), previously predicted depth latents zˆ(0d0:W )
- 2: Initialize zˆ(TdW:F ) ∼ N(0, σT2 I).
- 3: for t = T, . . . , 1 do
- 4: zˆ(td0:F ) = z ˆ(0d0:W ), zˆ(tdW:F )
- 5: σt = [σϵ, σϵ, ..., σϵ W

, σt, σt, ..., σt

F−W

]

- 6: z˜(0d0:F ) = Dθ(zˆt(d0:F ); σt, z(x))
- 7: zˆ(t−dW1 :F ) = zˆ

(dW :F ) t −z˜(0dW:F )

σt (σt−1 − σt) + zˆ(tdW:F )

- 8: end for
- 9: Return zˆ0(dW:F ).

- 1: Input: Model θ, rgb latent z(x), previously predicted depth latents zˆ(0d0:W )
- 2: Initialize zˆ(TdW:F ) ∼ N(0, σT2 I).
- 3: for t = T, . . . , 1 do
- 4: zˆt(d0:W ) =ForwardDiffuse(zˆ(0d0:W ), σt)
- 5: zˆt(d0:F ) = z ˆ(td0:W ), zˆ(tdW:F )
- 6: z˜0(d0:F ) = Dθ(zˆt(d0:F ); σt, z(x))
- 7: zˆ(t−dW1 :F ) = zˆ

(dW :F ) t −z˜(0dW:F )

σt (σt−1− σt) + zˆ(tdW:F )

- 8: end for
- 9: Return zˆ(0dW:F ).

- 1: Input: Model θ, rgb latent z(x)
- 2: Initialize zˆ(Td0:F) ∼ N(0, σT2 I).
- 3: for t = T, . . . , 1 do
- 4: z˜(0d0:F) = Dθ(zˆt(d0:F); σt, z(x))
- 5: zˆt(−dW1 :F) = zˆt(dW:F )−z˜0(dW:F )

σt (σt−1 − σt) + zˆt(dW:F)

- 6: end for
- 7: Return zˆ(0dW:F).

attention along the time axis, thus allowing us to randomly sample clips of length F ∈ [1,Fmax] during training.

Sequential Spatial-Temporal Fine-Tuning. The original SVD fine-tunes the full UNet Dθ on video datasets, meaning that both the spatial layers and the temporal layers are trained jointly. However, we argue splitting the fine-tuning of spatial and temporal components into distinct phases could favor achieving the highest temporal consistency that we pursue when dealing with videos. Accordingly, we investigate an alternative training protocol – sequential spatial-temporal training. Specifically, we first train the spatial layers using single-frame supervision. After convergence, we keep the spatial layers frozen and finetune the temporal layers using clips of randomly sampled length as supervision. We will demonstrate how this sequential protocol favors temporal consistency while maintaining spatial accuracy intact.

## 4. Experimental Results 4.1. Training Protocol

Implementation Details. We implement ChronoDepth using diffusers by fine-tuning SVD, specifically the imageto-video variant, following the strategy discussed in Sec. 3.3. We disable the cross-attention conditioning of the original SVD, we use the standard EDM noise schedule and the network preconditioning [38] and set image and video resolution to 576 × 768. We first pre-train our model with single-frame samples as input for 20k steps, using batch size 8; then, we fine-tune it with F-frame video clips for 18k steps using batch size 1, with F randomly sampled in [1,Fmax], setting Fmax = 5. For the consistent context-aware inference strategy, we set the length of each clip T = 10 and the number of overlapping frames between two adjacent clips W = 5. The entire training takes about 1.5 days on a cluster of 8 Nvidia Tesla A100-80GB GPUs. We use the Adam optimizer with a learning rate of 3e-5.

Training Datasets. We utilize four synthetic datasets, in-

cluding single-frame and multi-frame datasets. For the former category, we use Hypersim [57], a photorealistic synthetic dataset with 461 indoor scenes. We use the official split with around 54K samples from 365 scenes for training. We filter out incomplete samples, yielding 39K samples. The latter category involves multiple datasets: Tartanair [68] is a dataset collected in simulation environments for robot navigation tasks including 18 scenes. We use all the 738 sequences for training. Virtual KITTI 2 [12] is a synthetic urban dataset providing 5 scenes with various weather or modified camera configurations. We use 4 scenes with 80 sequences for training. MVS-Synth [35] is a synthetic urban dataset captured in the video game GTA. We use all 120 sequences for training. In total, our training set counts 39K single-frame samples and 938 video sequences.

### 4.2. Evaluation Protocol

Datasets. We quantitatively evaluate our model on three video datasets, ranging from synthetic to realistic, indoor to outdoor, and static to dynamic, thereby assessing the generalization ability of our model across diverse openworld scenarios: KITTI-360 [46] is a driving dataset collected in urban scenes, from which we select 8 video sequences with 200 frames each. We project LiDAR point clouds into image space to obtain sparse ground-truth depth. ScanNet++ [77] is an indoor dataset featuring depth maps captured by a laser scanner. In our evaluation, we use the nvs sem val set, which comprises 50 video sequences across diverse scenes from which we selected the first 90 frames for each. Sintel [11] is a synthetic dataset with precise depth labels. It comprises 23 sequences, each approximately 50 frames in length, within the training set.

Metrics. Following the protocol for affine-invariant depth evaluation [55], we initially align the estimated depth with ground-truth depth using least squares fitting. Differently from single-image depth evaluation, for which a different pair of scale and shift factors is fitted for each frame, a global pair of scale and shift values is estimated over the

KITTI-360 ScanNet++ Sintel Avg. Rank ↓

Method

AbsRel↓ δ1↑ MFC↓ AbsRel↓ δ1↑ MFC↓ AbsRel↓ δ1↑ MFC↓ MFC∗ ↓ AbsRel δ1 MFC

Marigold [40] 0.213 0.665 0.776 0.192 0.699 0.109 0.573 0.529 1.112 0.932 4.33 3.33 4.00 Marigold (SVD) 0.247 0.608 0.694 0.197 0.686 0.112 0.539 0.510 1.005 0.796 5.33 5.33 3.67 DepthAnything [75] 0.215 0.635 0.952 0.170 0.712 0.103 0.329 0.565 1.399 1.038 1.00 3.00 5.00 DepthAnything V2 [76] 0.207 0.656 0.807 0.170 0.713 0.103 0.387 0.554 1.504 1.125 1.67 2.67 5.00

NVDS [70] 0.379 0.384 1.276 0.239 0.565 0.136 0.442 0.465 1.220 0.924 6.00 7.00 6.00 DepthCrafter [33] 0.293 0.462 0.655 0.199 0.642 0.094 0.374 0.566 1.270 0.889 4.67 4.33 3.00 ChronoDepth (Ours) 0.215 0.654 0.407 0.176 0.726 0.092 0.493 0.555 0.728 0.516 4.00 2.33 1.00

Table 1. Quantitative Comparison on zero-shot depth benchmarks. Top: single-image depth estimators. Bottom: video depth estimators. MFC∗ denotes the use of ground-truth optical flow to account for moving objects when computing MFC.

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Marigold DepthAnything DepthAnything V2

Input Video

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

NVDS DepthCrafter Ours

Ground Truth

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Marigold DepthAnything DepthAnything V2

Input Video

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

NVDS DepthCrafter Ours

Ground Truth

- Figure 5. Qualitative comparison for video depth estimation. For enhanced visualization of temporal quality, we present the y-t slice of each result within red boxes, achieved by slicing the depth values along the time axis at the designated red line positions.

entire video following [26, 85] to reflect scale consistency. Then, we consider two widely recognized metrics [55] for evaluation, Absolute Mean Relative Error (AbsRel) and δ1 accuracy with a specified threshold value of 1.25. To measure temporal consistency, we introduce multi-frame consistency (MFC) by warping the prediction from one frame to its adjacent frame, and evaluating the discrepancy between the two depth map predictions. On datasets featuring several moving objects such as Sintel, we use groundtruth optical flow to compute MFC. Note that both groundtruth flow and camera poses are used only to compute MFC. Please refer to Appendix A for details of the MFC metric.

### 4.3. Comparison with State-of-the-Art

Baselines. We compare ChronoDepth with five existing baselines, covering both single-image and video depth estimation. For single-image depth estimation, we select both generative and discriminative state-of-the-art methods – respectively Marigold [40] versus DepthAnything [75] and DepthAnything V2 [76]. Additionally, we employ an improved Marigold model derived by repurposing SVD in place of Stable Diffusion and deploy it for single-image depth estimation – Marigold (SVD). For video depth estimation, we select a discriminative video depth estimator, NVDS [70], and DepthCrafter [33], a video depth estimator

derived from SVD [8] concurrently with ChronoDepth. We employ the default paper settings for any method, except for DepthCrafter. The primary reason for this deviation is that DepthCrafter conditions the prediction process over all frames in a video sequence, thus implying that all frames are known in advance – which is not available when processing videos in a streamed manner through autoregressive prediction. Accordingly, we set the overlap and length of the clip to 5 and 10 to align it with ChronoDepth.

Quantitative Analysis. In Table 1, we compare our model with both existing single-image (top) and video (bottom) depth estimators. We highlight the best , second-best , and third-best scores achieved on any metrics, and report the average ranking achieved across the three datasets on the rightmost columns. Starting with KITTI-360 and ScanNet++, on the one hand, ChronoDepth retains spatial accuracy comparable to state-of-the-art single-frame depth estimators such as Depth Anything and Depth Anything V2, despite these latter being trained on about 500× the number of images we used to train ChronoDepth; on the other hand, ChronoDepth achieves state-of-the-art temporal consistency – see the MFC metric, with a relative improvement of 98% on KITTI-360. Compared to other video methods such as NVDS and the concurrent DepthCrafter, ChronoDepth outperforms them both in terms of spatial

Training Inference KITTI-360 ScanNet++ Image RandomClip S-T FT Naive Replacement Ours AbsRel↓ δ1↑ MFC↓ AbsRel↓ δ1↑ MFC↓

- (A) ✓ ✓ ✓ 0.252 0.575 0.555 0.201 0.676 0.097

- (B) ✓ ✓ ✓ 0.236 0.609 0.470 0.205 0.669 0.104

- (C) ✓ ✓ ✓ 0.229 0.625 0.482 0.224 0.631 0.107

- (D) ✓ ✓ ✓ ✓ 0.233 0.614 0.505 0.194 0.692 0.098

- (E) ✓ ✓ ✓ ✓ 0.231 0.618 0.479 0.194 0.693 0.097

- (F) ✓ ✓ ✓ ✓ 0.215 0.654 0.407 0.176 0.726 0.092 Table 2. Quantitative ablation studies We measure the impact of different pre-training protocols and inference strategies.

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

| |
|---|

|[Figure 86]|
|---|

Input Video (a) (b) (c)

- Figure 6. Qualitative ablation results. (a) Naive sliding window inference strategy; (b) Inference with replacement trick; (c) Our consistent context-aware inference.

accuracy and temporal consistency. On Sintel, despite the lower performance in terms of AbsRel, ChronoDepth excels in δ1 and maintains superior temporal consistency compared to the other baselines, confirming to be better suited for applications like AR/VR where temporal consistency plays a predominant role over spatial accuracy.

Qualitative Analysis. Fig. 5 shows a qualitative comparison between the baselines and ChronoDepth on video depth estimation. For easing the perception of temporal consistency, we present the y-t slice of each result within red boxes, extracted by slicing the depth values along the time axis at the designated red line positions following [85]. ChronoDepth generates smooth depth over time while preserving fine-grained details. In contrast, single-frame depth estimators display high-frequency bands in their y-t slice, indicative of flickering artifacts in the estimated depth sequences. Additionally, our two video depth competitors also exhibit similar artifacts at a per-clip level.

### 4.4. Ablation Studies

In Table 2, we run ablation studies on the KITTI-360 and ScanNet++ datasets, analyzing the following factors:

Single-Frame Training Datasets. We investigate the impact of single-frame data used during the training phase. Our experiments highlight that merely using multi-frame data only (A) yields sub-optimal results over combining both multi-frame and single-frame datasets (D), with this latter improveing spatial accuracy significantly.

Random Clip Length Sampling. Next, we ablate the effectiveness of random video clip length sampling during training. Removing this sampling (B) leads to performance degradation. This indicates that sampling clips of random length acts as an effective form of data augmentation, mitigating the risk of model overfitting.

Sequential Spatial-Temporal Fine-Tune. We evaluate the effect of sequential spatial-temporal training protocol by jointly training the full network (C). The sequential training strategy always leads to better temporal consistency, and better spatial accuracy on ScanNet++, which means disentangling spatial and temporal layers could be the better way to tame video foundation into a depth estimator. Due to the minimal view change and extensive depth range on KITTI360, both AbsRel and δ1 exhibit limited sensitivity.

Inference Strategy. We compare the results obtained by naive sliding window inference (D), inference with replacement trick (E) and our consistent context-aware inference (F). Compared to naive sliding window inference, inference with replacement trick barely improves the temporal consistency, whereas our approach leads to better results in terms of both spatial accuracy and temporal consistency. We also visualize the y-t slice for these three inference strategies in Fig. 6. Both naive sliding window inference and inference with replacement trick exhibit high-frequency bands at a per-clip level, while our consistent context-aware inference strategy ensures temporal consistency by providing consistent contextual information over the clip, significantly reducing flickering artifacts between windows.

## 5. Conclusion

This paper introduced ChronoDepth, a video depth estimator that prioritizes temporal consistency by leveraging video generation priors. Our exploration of various training and inference strategies has led us to identify the most effective approach to tame SVD into a consistent depth predictor, resulting in superior performance. Specifically, ChronoDepth outperforms existing methods in terms of temporal consistency, surpassing both image and video depth estimation techniques, while maintaining comparable spatial accuracy. We contend our empirical insights into harnessing video generation models for depth estimation lay the groundwork for future investigations in this domain.

Acknowledgements: This work is supported by NSFC under grant U21B2004, 62202418, and 62441223. This work was supported by Ant Group Research Fund. Yiyi Liao is with the Zhejiang Provincial Key Laboratory of Information Processing, Communication and Networking (IPCAN).

## References

- [1] Alembics. Disco diffusion. https://github.com/alembics/ disco-diffusion, 2022. 3
- [2] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 3
- [3] Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, and Xi Yin. Spatext: Spatio-textual representation for controllable image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18370–18380, 2023. 3
- [4] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. arXiv preprint arXiv:2302.08113, 2023. 3
- [5] Dina Bashkirova, Jos´e Lezama, Kihyuk Sohn, Kate Saenko, and Irfan Essa. Masksketch: Unpaired structure-guided masked image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1879–1889, 2023. 3
- [6] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4009–4018, 2021. 2
- [7] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023. 2
- [8] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv.org, 2311.15127, 2023. 1, 3, 5, 7
- [9] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 1
- [10] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 3
- [11] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy,

October 7-13, 2012, Proceedings, Part VI 12, pages 611–

625. Springer, 2012. 6

- [12] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 6
- [13] Yuanzhouhan Cao, Yidong Li, Haokui Zhang, Chao Ren, and Yifan Liu. Learning structure affinity for video depth estimation. In Proceedings of the 29th ACM International Conference on Multimedia, pages 190–198, 2021. 3
- [14] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In ICCV, 2023. 3
- [15] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. arXiv preprint arXiv:2407.01392, 2024. 1, 3, 4, 5
- [16] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 3
- [17] Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, and Anand Bhattad. Generative models: What do they know? do they know things? let’s find out! arXiv.org, 2023. 2
- [18] Yiqun Duan, Xianda Guo, and Zheng Zhu. Diffusiondepth: Diffusion denoising approach for monocular depth estimation. arXiv preprint arXiv:2303.05021, 2023. 2
- [19] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multitask mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10786–10796, 2021. 1, 2
- [20] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems, 27, 2014. 2
- [21] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep Ordinal Regression Network for Monocular Depth Estimation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2018. 2

- [22] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. arXiv.org, 2024. 1, 2
- [23] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. In European Conference on Computer Vision, pages 89–106. Springer, 2022. 3
- [24] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398, 2024. 3
- [25] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? The KITTI vision benchmark suite. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2012. 2

- [26] Cl´ement Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3828–3838,

2019. 7

- [27] Ming Gui, Johannes S. Fischer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. Depthfm: Fast monocular depth estimation with flow matching. arXiv.org, 2403.13788, 2024. 1, 2
- [28] Vitor Guizilini, Rares, Ambrus,, Dian Chen, Sergey Zakharov, and Adrien Gaidon. Multi-frame self-supervised depth with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 160– 170, 2022. 3
- [29] Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rares Ambrus, and Adrien Gaidon. Towards zero-shot scale-aware monocular depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2
- [30] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [31] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 1, 2, 3, 4
- [32] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. arXiv preprint arXiv:2404.15506, 2024. 2
- [33] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. arXiv preprint arXiv:2409.02095, 2024. 2, 3, 7
- [34] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 3
- [35] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multiview stereopsis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2821–2830,

2018. 6

- [36] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via maskeddiffusion. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [37] Yuanfeng Ji, Zhe Chen, Enze Xie, Lanqing Hong, Xihui Liu, Zhaoqiang Liu, Tong Lu, Zhenguo Li, and Ping Luo. Ddp: Diffusion model for dense visual prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21741–21752, 2023. 2
- [38] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative

- models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 3, 6
- [39] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 3
- [40] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 3, 7
- [41] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2426– 2435, 2022. 3
- [42] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021. 3
- [43] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1611–1621, 2021. 1, 3
- [44] Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326, 2019. 2
- [45] Zhenyu Li, Shariq Farooq Bhat, and Peter Wonka. Patchfusion: An end-to-end tile-based framework for highresolution monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10016–10025, 2024. 2
- [46] Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3292–3310, 2022. 6
- [47] Xiaoxiao Long, Lingjie Liu, Wei Li, Christian Theobalt, and Wenping Wang. Multi-view depth estimation using epipolar spatio-temporal networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8258–8267, 2021. 3
- [48] Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. Consistent video depth estimation. ACM Transactions on Graphics (ToG), 39(4):71–1, 2020. 1, 3
- [49] S Mahdi H Miangoleh, Sebastian Dille, Long Mai, Sylvain Paris, and Yagiz Aksoy. Boosting monocular depth estimation models to high-resolution via content-adaptive multiresolution merging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9685–9694, 2021. 2
- [50] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 3
- [51] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and

- Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 3
- [52] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 3
- [53] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [54] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 3

- [55] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 1, 2, 6, 7
- [56] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021. 1, 2
- [57] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021. 6
- [58] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [59] Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, and David J. Fleet. The surprising effectiveness of diffusion models for optical flow and monocular depth estimation. In Advances in Neural Information Processing Systems (NIPS), 2023. 2
- [60] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 5
- [61] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGB-D images. In Proc. of the European Conf. on Computer Vision (ECCV), 2012. 2
- [62] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 3
- [63] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based

- generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [64] Zachary Teed and Jia Deng. Deepv2d: Video to depth with differentiable structure from motion. arXiv preprint arXiv:1812.04605, 2018. 1
- [65] Patil Vaishakh, Wouter Van Gansbeke, Dengxin Dai, and Luc Van Gool. DonˆaC™t forget the past: Recurrent depth estimation from monocular video. IEEE Robotics and Automation Letters, 5(4):6813–6820, 2020. 3

- [66] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661– 1674, 2011. 3
- [67] Andrey Voynov, Kfir Aberman, and Daniel Cohen-Or. Sketch-guided text-to-image diffusion models. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 3
- [68] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 6
- [69] Yiran Wang, Zhiyu Pan, Xingyi Li, Zhiguo Cao, Ke Xian, and Jianming Zhang. Less is more: Consistent video depth estimation with masked frames modeling. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6347–6358, 2022. 1, 3
- [70] Yiran Wang, Min Shi, Jiaqi Li, Zihao Huang, Zhiguo Cao, Jianming Zhang, Ke Xian, and Guosheng Lin. Neural video depth stabilizer. arXiv.org, 2023. 1, 3, 7
- [71] Jamie Watson, Oisin Mac Aodha, Victor Prisacariu, Gabriel Brostow, and Michael Firman. The temporal opportunist: Self-supervised multi-frame monocular depth. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1164–1174, 2021. 3
- [72] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 3
- [73] Ke Xian, Juewen Peng, Zhiguo Cao, Jianming Zhang, and Guosheng Lin. Vita: Video transformer adaptor for robust video depth estimation. IEEE Transactions on Multimedia, 26:3302–3316, 2024. 1
- [74] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. Diffusion models trained with large data are transferable visual models. arXiv preprint arXiv:2403.06090, 2024. 1, 2
- [75] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 7
- [76] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024. 7, 2

- [77] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the International Conference on Computer Vision (ICCV), 2023. 6
- [78] Wei Yin, Yifan Liu, and Chunhua Shen. Virtual normal: Enforcing geometric constraints for accurate and robust depth prediction. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. 2
- [79] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 204–213, 2021. 1, 2
- [80] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9043–9053, 2023. 2
- [81] Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, and Ping Tan. Newcrfs: Neural window fully-connected crfs for monocular depth estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,

2022. 2

- [82] Chi Zhang, Wei Yin, Billzb Wang, Gang Yu, Bin Fu, and Chunhua Shen. Hierarchical normalization for robust monocular depth estimation. Advances in Neural Information Processing Systems, 35:14128–14139, 2022. 1, 2
- [83] Haokui Zhang, Chunhua Shen, Ying Li, Yuanzhouhan Cao, Yu Liu, and Youliang Yan. Exploiting temporal consistency for real-time video depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1725–1734, 2019. 3
- [84] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 3
- [85] Zhoutong Zhang, Forrester Cole, Richard Tucker, William T Freeman, and Tali Dekel. Consistent depth of moving objects in video. ACM Transactions on Graphics (TOG), 40(4):1– 12, 2021. 1, 3, 7, 8
- [86] Zhongwei Zhang, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Ting Yao, Yang Cao, and Tao Mei. Trip: Temporal residual learning with image noise prior for image-to-video diffusion models. In CVPR, 2024. 3
- [87] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5729–5739, 2023. 2
- [88] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. https://github.com/hpcaitech/Open-Sora, 2024. 2

# Learning Temporally Consistent Video Depth from Video Diffusion Priors Supplementary Material

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.64

0.50

0.63

0.26

0.62

0.48

###### AbsRel

0.25

###### MFC

0.61

###### δ1

0.46

0.60

0.24

0.59

0.44

0.23

0.58

0.42

0.57

0.22

-10 -8 -6 -4 -2 0

-10 -8 -6 -4 -2 0

-10 -8 -6 -4 -2 0

σ

σ

σ

Figure S1. Ablation Study. We report accuracy and consistency metrics of our method on KITTI-360 with different σϵ.

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Ground Truth NVDS DepthCrafter Ours

Figure S2. More qualitative comparison for video depth estimation. Results on ScanNet++ dataset.

## A. Details on Evaluation Protocol

Datasets. The sequences on KITTI-360 we chose follows the rules below. Given the absence of ground-truth poses for the initial frames of each sequence, we extracted frames 300-500, ultimately utilizing 200 frames for evaluation.

Metrics. To measure temporal consistency, we introduce multi-frame consistency (MFC): given two depth maps Dm,Dn ∈ RW×H at frame m, n of the video sequence, we unproject Dm into a point cloud; using the ground-truth world-to-camera poses Pm,Pn ∈ R3×4 for frames m,n’ camera, we transform the point cloud from frame m’ camera space to frame n’ camera space, and project it onto frame n’s image plane to yield Dm→n. We measure temporal consistency as the average L1 distance between Dm→n and Dn. We mask out invalid pixels in both frames. In practice, we calculate multi-frame consistency on adjacent frames.

## B. Detailed Proof on Mathematical Rigor and Fluctuating Guidance

To ensure enough context information, we aim to sample depth latents zˆ(d

0:W), which is pθ(zˆ(d

W:F) conditioned on zˆ(d

0:W)). For the replacement trick, the sampling of zˆ(d

W:F)|zˆ(d

W:F) follows standard unconditional sampling from pθ(zˆ(t−d0:1F)|zˆ(td0:F)), where zˆ(td0:F) = z ˆ(td0:W),zˆ(tdW:F) . Crucially, samples zˆ(td0:W) are replaced at each step by exact forward process samples q(zˆ(td0:W)|zˆ(d

0:W)). This causes to update zˆt(−dW1 :F) using zˆ(t−dW1,θ:F)(zˆ(tdW:F)) ≈ Eq[zˆ(t−dW1 :F)|zˆ(tdW:F),zˆ(td0:W)], while what is needed instead is Eq[zˆ(t−dW1 :F)|zˆ(tdW:F),zˆ(d

0:W)] = Eq[zˆ(t−dW1 :F)|zˆ(td0:F),zˆ(td0:W)] + (σt2/αt)∇zˆ(dW:F)

0:W)|zˆ(td0:W)). The missing second term introduces dynamic guidance variations across sampling steps. As for our context-aware strategy, we

log q(zˆ(d

t−1

Marigold DAv2 NVDS DC Ours Inference Speed (s) 5.64 0.80 1.05 1.30 0.49

Compute (GB) 5.67 23.7 20.5 8.04 6.6

# of Parameters (B) 1.29 0.33 0.35 2.25 2.25 Training data

# of frames 74K 62.6M 1.4M - 39K # of scenes - - 14.2K 203K 938

Table S1. Speed and compute comparison.

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Ground Truth NVDS DepthCrafter Ours

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Ground Truth NVDS DepthCrafter Ours

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

|[Figure 146]|
|---|

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Ground Truth NVDS DepthCrafter Ours

Figure S3. M qualitative comparison for depth estimation. Results KITTI-360 datasets.

|[Figure 151]<br><br>More<br><br>W:F )<br><br>:F )<br><br>W :F|
|---|

|[Figure 152]<br><br>video|
|---|

|[Figure 153]<br><br>on<br><br>spatial<br><br>σϵ =|
|---|

|[Figure 154]<br><br>ϵ also Con-|
|---|

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

can do conditional sampling from pθ(zˆt(−d0:1F)|zˆ(td0:F)), with zˆ(td0:F) = z ˆ(d

results in degraded sp and temporal performance due to compounded errors. Conversely, an overly large σ o leads to diminished spatial and temporal performance. sequently, we opt for σ −4.

0:W),zˆt(d without forward process q(·|·). As a result, zˆt(−dW1 is updated in the direction provided by E[zˆt(−dW1 :F)|zˆ(td ),zˆ(d

0:W)].

Input Video Marigold DepthAnything DepthAnything V2

## E. More Qualitative Results

|[Figure 159]<br><br>],|
|---|

|[Figure 160]<br><br>40] the<br><br>to|
|---|

|[Figure 161]|
|---|

|[Figure 162]<br><br>by the|
|---|

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

## C. Speed and Compute Comparison

We provide more qualitative comparisons from KITTI-360, ScanNet++ and Bonn datasets in Figs. S2 to S4. First, we highlight the remarkable spatial accuracy achieved b our method, being comparable to or even better than th one by state-of-the-art models. Furthermore, we can notice how the y-t slice by most methods shows high-frequency artifacts, whereas ours is consistently smoother, confirming the superior temporal consistency we achieve.

Tab. S1 shows runtime, compute and model parameters. ChronoDepth is significantly faster than Marigold [4 and DepthCrafter (DC) [33 and requires a fraction of t memory used by DepthAnything v2(DAv2) [76], thanks our more lightweight UNet architecture compared with the baselines.

Ground Truth NVDS DepthCrafter Ours

|[Figure 167]|
|---|

|[Figure 168]<br><br>σϵ|
|---|

|[Figure 169]<br><br>to|
|---|

|[Figure 170]<br><br>(Scan-|
|---|

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

## D. Additional Ablation

## F. Limitation

We investigate the significance of the small noise level in the context of overlapping frames within arbitrarily long videos. As illustrated in Fig. S1, an excessively small σϵ

Our method is robust rapid ego-camera motion ( net++) and long video (KITTI-360). However, we observe

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

2

NVDS DepthCrafter Ours

Ground Truth

||
|---|

||
|---|

||
|---|

||
|---|

Input Video Marigold DepthAnything DepthAnything V2

||
|---|

||
|---|

||
|---|

||
|---|

Ground Truth NVDS DepthCrafter Ours

||
|---|

||
|---|

||
|---|

||
|---|

Input Video Marigold DepthAnything DepthAnything V2

||
|---|

||
|---|

||
|---|

||
|---|

Ground Truth NVDS DepthCrafter Ours

|[Figure 215]|
|---|

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

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Ground Truth NVDS DepthCrafter Ours

|[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Ground Truth NVDS DepthCrafter Ours

|[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Input Video Marigold DepthAnything DepthAnything V2

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

NVDS DepthCrafter Ours

Ground Truth

Figure S4. More qualitative comparison for video depth estimation. Results on ScanNet++ and Bonn datasets.

a slight degradation in AbsRel when handling scenes with abundant dynamic objects (Sintel). We attribute this to the limited moving objects in the training data, which can be extended in the future.

