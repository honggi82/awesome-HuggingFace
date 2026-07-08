arXiv:2506.02497v1[cs.CV]3Jun2025

[Figure 1]

LumosFlow: Motion-Guided Long Video Generation

Jiahao Chen1,2, Hangjie Yuan2,3, Yichen Qian†2,3, Jingyun Liang2,3, Jiazheng Xing4, Pengwei Liu4, Weihua Chen2,3, Fan Wang2, Bing Su†1

1Gaoling School of Artificial Intelligence, Renmin University of China 2DAMO Academy, Alibaba Group 3Hupan Lab 4Zhejiang University

†Corresponding authors

Long video generation has gained increasing attention due to its widespread applications in fields such as entertainment and simulation. Despite advances, synthesizing temporally coherent and visually compelling long sequences remains a formidable challenge. Conventional approaches often synthesize long videos by sequentially generating and concatenating short clips, or generating key frames and then interpolate the intermediate frames in a hierarchical manner. However, both of them still remain significant challenges, leading to issues such as temporal repetition or unnatural transitions. In this paper, we revisit the hierarchical long video generation pipeline and introduce LumosFlow, a framework introduce motion guidance explicitly. Specifically, we first employ the Large Motion Text-to-Video Diffusion Model (LMTV-DM) to generate key frames with larger motion intervals, thereby ensuring content diversity in the generated long videos. Given the complexity of interpolating contextual transitions between key frames, we further decompose the intermediate frame interpolation into motion generation and post-hoc refinement. For each pair of key frames, the Latent Optical Flow Diffusion Model (LOF-DM) synthesizes complex and large-motion optical flows, while MotionControlNet subsequently refines the warped results to enhance quality and guide intermediate frame generation. Compared with traditional video frame interpolation, we achieve 15× interpolation, ensuring reasonable and continuous motion between adjacent frames. Experiments show that our method can generate long videos with consistent motion and appearance. Code and models will be made publicly available upon acceptance. Our project page: https://jiahaochen1.github.io/LumosFlow/

Date: June 11, 2025

- 1 Introduction

Video diffusion models (Ho et al., 2022b; Singer et al., 2022b; Ho et al., 2022a; Yuan et al., 2024; Wang et al., 2023b; Liang et al., 2025; Kong et al., 2024; Singer et al., 2022a; Chefer et al., 2025) have demonstrated impressive capabilities in generating short clip videos (14 frames (Blattmann et al., 2023) or 49 frames (Yang et al., 2024)). However, in most practical scenarios, there is a need for the generation of longer videos, which often consist of hundreds or even thousands of frames. The ability to generate long videos is crucial for a variety of applications, including film production, virtual reality, and video-based simulations. Current methods (Chen et al., 2023; Xing et al., 2023), however, struggle to adapt to long video generation due to the challenges of maintaining temporal coherence, global consistency, and efficient computational performance over extended sequences. As a result, there remains a significant gap between generating short clips and producing high-quality long videos.

As shown in Fig. 1, the strategies for generating long videos can be primarily categorized into two kinds of approaches: the first involves generating short video clips sequentially and then splicing them together (Lu et al., 2025; Qiu et al., 2023), while the second adopts a hierarchical pipeline that first generates key frames and subsequently interpolates the intermediate frames between these key frames to construct a continuous long video (Ge et al., 2022; Harvey et al., 2022; Xie et al., 2024). However, both strategies have inherent challenges. As shown in Fig. 2, long videos generated clip by clip may suffer from inconsistencies and lack coherence when concatenated, resulting in noticeable artifacts or temporal repetition. While the hierarchical generation method can mitigate temporal repetition by adjusting the generation of key frames, the generation of intermediate frames remains a significant challenge, leading to issues such as unnatural transitions or missed

Generate clip by clip

Time

frames in clips

Generate by key frames and intermediate interpolation

key frames

intermediate frames

Time

Generate by key frames and motion-guided interpolation

motion frames

guide

Time

Figure 1 The comparison of different long video generation pipelines. LumosFlow generates long videos through the process of generating key frames and performing motion-guided intermediate frame interpolation.

### motion fluidity.

In this paper, we revisit the hierarchical generation pipeline and highlight that motion guidance is critical for the intermediate frame interpolation. To verify this point, we explore two methods for generating intermediate frames: (1) we adapt the current Image-to-Video diffusion model (Yang

Frame 1 Frame 64 Frame 128

Frame 32

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

FreeLong

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

- et al., 2024) for intermediate frame interpolation, referred to as Motion-Free; and (2) we integrate motion information into the existing Image-to-Video diffusion model to facilitate frame interpolation, referred to as Motion-Guidance. As illustrated in Fig. 3, the frames generated without motion restrictions exhibit unnatural transitions; in contrast, those generated with motion guidance demonstrate a more realistic and fluid movement between the various frames.1

FreeNoise

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

LumosFlow

Figure 2 Generated long videos with the prompt “a man in a blue plaid shirt and a white cowboy hat is seen drinking whiskey from a glass..." by FreeLong, FreeNoise, and LumosFlow. We randomly select some frames for comparison.

Based on the preivous findings, we propose LumosFlow to generate long videos in a motion guidance manner. Firstly, we propose our Large Motion Text-to-Video Diffusion Model (LMTV-DM) to produce key frames that exhibit significant inter-frame motion in a single pass. After deriving key frames, the generation of intermediate frames between each pair of them can be decomposed into motion generation and post-hoc refinement. Leveraging the powerful generative capabilities of latent diffusion models, we introduce the Optical Flow Variational AutoEncoder (OF-VAE) and the Latent Optical Flow Diffusion Model (LOF-DM). OF-VAE compresses optical flows into a compact latent space, while LOF-DM generates optical flows in a generative manner. Compared with other flow estimation methods, our method is key frame-aware and uses their semantic information as a guide. The generated optical flow is more in line with natural laws and more suitable for higher-rate interpolation tasks. For post-hoc refinement, we propose MotionControlNet, which incorporates

1More experiments to verify the importance of motion guidance are in the Sec. 5.3

### wrapped frames for enhanced results. By capitalizing on the strengths of diffusion models and motion guidance, LumosFlow can generate high-quality intermediate frames.

Our contribution can be summarized as follows: (1) We identify the importance of motion guidance in achieving realistic and fluid transitions in intermediate frame interpolation. Building on these findings, we propose LumosFlow, which decomposes the generation pipeline into key frame generation and intermediate frame interpolation. During the interpolation process, we explicitly integrate motion information to enhance the results. (2) LumosFlow comprises three diffusion models: LMTVDM, LOF-DM, and MotionControlNet. In the generation process, LMTV-DM produces key frames with significant intervals, while LOF-DM and MotionControlNet collaborate to create realistic intermediate frames, effectively injecting motion (optical flow) into the generation. (3) LumosFlow achieves the generation of 273 frames, producing 18 key frames and interpolating 16 frames between each pair of key frames. We obtain promising results in both long video generation and video frame interpolation. Additionally, we perform frame interpolation at a significantly higher rate (15×) compared to traditional methods, which typically achieve rates of less than 8×.

First frame Intermediate frames Last frame

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Motion-Free

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Motion-Guidance

Figure 3 Generated intermediate frames based on the first and last frames. Since the absence of motion, the frames produced using the Motion-Free method exhibits unnatural movements from the subjects. In contrast, the result generated with the Motion-Guidance is significantly more realistic.

- 2 Related Work

Long video generation. Long video generation focuses on producing videos with a significantly higher number of frames (e.g., 256, 512, or 1024 frames). Among various methods, two primary strategies have emerged: autoregressive modeling (Qiu et al., 2023; Wang et al., 2023a; Lu et al., 2025) and hierarchical generation (Yin et al., 2023; Brooks et al., 2022; Ge et al., 2022; Harvey et al., 2022; Xie et al., 2024). Gen-L-Video (Wang et al., 2023a) enables multi-text conditioned long video generation and editing by extending short video diffusion models without additional training, ensuring content consistency across diverse semantic segments. FreeNoise (Qiu et al., 2023) enhances long video generation with multi-text conditioning by rescheduling noise initialization for long-range temporal consistency and introducing a motion injection method, achieving superior results. NUWA-XL (Yin et al., 2023) introduces a Diffusion over Diffusion architecture for extremely long video generation, employing a coarse-to-fine, parallel generation strategy that reduces the traininginference gap and significantly accelerates inference. Differently, LumosFlow introduces motion as guidance based on hierarchical generation, making the generation of intermediate frames more controllable than previous methods.

Video frame interpolation. Video frame interpolation (VFI) involves generating intermediate frames between existing ones to achieve smoother motion or higher frame rates. Among different strategies, flow-based methods (Liu et al., 2017; Bao et al., 2019; Huang et al., 2022; Liu et al., 2024; Lew et al., 2024) are drawing wide attention since they have better temporal consistency. RIFE (Huang et al., 2022) uses a lightweight neural network to predict intermediate optical flows directly, enabling fast and accurate interpolation. VFIMamba (Zhang et al., 2024) utilizes Selective State Space Models (S6) and proposes a novel video frame interpolation method. Recently, diffusion models (Ho et al., 2020) have demonstrated exceptional capabilities in generative tasks, prompting researchers to extend their application to VFI. LDMVFI (Danier et al., 2024) first applys latent diffusion models to VFI, incorporating a vector-quantized autoencoding model to enhance diffusion performance. VIDIM (Jain et al., 2024) introduces a generative video interpolation approach that produces high-fidelity short videos by utilizing cascaded diffusion models for low-to-high resolution generation.

Achieving great success, these methods are weak in estimating complex and large non-linear motions between two frames. Benefiting from LOF-DM, LumosFlow can generate more realistic motion and provides more precise guidance during interpolation tasks.

## 3 Preliminary on Diffusion model

Diffusion models (Ho et al., 2020) are a class of probabilistic generative models that aim to model the data distribution p(x) through a latent variable process. They are expressed as pθ(x0) = pθ(x0:T)dx1:T, where x0 represents the data, and x1,··· ,xT are progressively noisier latent variables generated by adding noise in a forward process. The parameter θ denotes the learnable model parameters. Formally, the forward process, also known as the diffusion process, is a fixed Markov chain of length T defined as:

√αtxt−1,(1 − αt)I), (1)

q(xt | xt−1) = N(xt |

where xt = √αtxt−1 + √1 − αtϵ, ϵ ∼ N(0,I), and αt ∈ (0,1) is a variance schedule that governs the amount of noise added at each step t.

The reverse process, which is parameterized by the model, aims to denoise the noisy latent variables xt back to the original data x0. It is defined as another Markov chain:

pθ(xt−1 | xt) = N(xt−1 | µθ(xt,t),Σθ(xt,t)), (2)

where µθ(xt,t) and Σθ(xt,t) are learnable functions that approximate the true posterior mean and variance. The overall joint distribution of the reverse process is:

pθ(x0:T) = pθ(+xT)

T

pθ(xt−1 | xt). (3)

t=1

During training, the optimization target is: L = Ex

0,ϵ∼N(0,I),t∥ϵθ(xt,t) − ϵ∥2. (4)

During inference, the reverse process begins by sampling xT ∼ N(0,I) and iteratively applying the learned denoising steps to generate x0. This iterative denoising process enables high-quality sample generation.

## 4 LumosFlow

In this section, we present LumosFlow, a novel method for long video generation. Our method is divided into three stages: key frame generation, optical flow generation, and post-hoc refinement. In Sec. 4.1, we describe the Large Motion Text-to-Video Diffusion Model designed to generate key frames. From Sec. 4.2 to Sec.4.4, we detail the design of components in intermediate frame interpolation. Specifically, in Sec. 4.2, we introduce the Optical Flow Variational AutoEncoder (OF-VAE), which efficiently compresses optical flow into a latent space. In Sec. 4.3, we describe the design of the diffusion model, which generates the optical flows. Finally, in Sec. 4.4, we introduce the proposed MotionControlNet for refining the warped frames to produce the final interpolated results.

- 4.1 Large Motion Text-to-Video Diffusion Model

Previous text-to-video diffusion models are capable of generating continuous videos with numerous frames; however, they lack the ability to produce key frames with larger intervals. These key frames are essential,

- as they significantly influence both the motion range and the overall scene of the video. Therefore, it is particularly important to generate key frames that are not only distinct but also consistent in subject matter. Unfortunately, prior hierarchical generation strategies do not explicitly account for these factors, which limits their effectiveness in capturing coherent narratives and maintaining visual continuity. To enhance the ability to generate key frames with substantial motion, we establish an additional training set consisting of videos

Step 1: Key frame generation

[Figure 24]

ZT Z

[Figure 25]

| |0|
|---|---|
| | |

[Figure 26]

A panda, dressed in a small……

[Figure 27]

Encoder (Text)

Video Decoder

Key frames

LMTV-DM

Step 2: Motion-guided intermediate frame generation

[Figure 28]

[Figure 29]

CLIP Encoder

[Figure 30]

Key frames

Generated optical flow

ZT Z0

[Figure 31]

[Figure 32]

OP-VAE Encoder

OP-VAE Decoder

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Linear flow initilization

LOF-DM

W

[Figure 39]

Backward warpping

ZT

ZT

Z0

[Figure 40]

| | |
|---|---|
| | |

[Figure 41]

[Figure 42]

Video Encoder

Video Decoder

[Figure 43]

Motion ControlNet Image-to-Video Foundation model

Intermediate frames

- Figure 4 The overall framework of LumosFlow includes key frame generation and intermediate frame generation. The intermediate frame generation comprises two components: motion generation (highlighted in yellow) and post-hoc refinement (highlighted in orange).

- at a lower frame rate, which results in videos with larger intervals. By fine-tuning the current text-to-video generation model, we can sample a video (key frames) that exhibits higher motion in accordance with the provided prompt. Formally, it has:

v ∼ pθ(v | P), (5) where v and P are the generated video (a set of key frames) and the given text prompt.

- 4.2 Optical Flow VAE

Given two images I1 and IK, it is difficult to synthesize intermediate frames {Iˆk | k = 2,··· ,K − 1} at time stamp k directly. Previous methods like RIFE (Huang et al., 2022) decouple intermediate frame generation into motion (optical flow) estimation and appearance refinement. Therefore, we first investigate the generation of optical flow. Formally, we denote Fk→1 and Fk→K as the optical flow from I1 to Ik and IK to Ik, respectively, and Fˆ:→1 = {Fˆt→1}Kt=2−1, Fˆ:→K = {Fˆt→K}Kt=2−1.

Optical flows and videos share the same dimensionality, the direct generation of optical flow in the pixel space still incurs substantial computational costs. Considering the information conveyed by optical flow is less rich than that of RGB data, allowing it to support greater compression ratios, we propose an Optical Flow Variational AutoEncoder (OF-VAE) that compresses optical flow within the latent space, rather than directly generating it at the pixel level.

Formally, given a video v ∈ RK×C×H×W, we extract the first (I1) and last frames (IK) as the key frames and compute the optical flow for all intermediate frames, denoted as F:→1 ∈ R(K−2)×2×H×C and F:→K. Since F:→1 and F:→K share common motion information, we concatenate them along the channel dimension and

use an encoder E map them into the latent space z = E([F:→1,F:→K]), where z ∈ Rk×c×h×w, and a decoder D reconstruct optical flows, giving [Fˆ:→1,Fˆ:→K] = D(z). Similar to Stable Diffusion (Rombach et al., 2022), the encoder downsamples the optical flow by a factor f = H/h = W/w, g = K/k. Considering the sparsity of optical flow, we set f = 32 and g = 4. Compared with the previous image VAE (Podell et al., 2023), OF-VAE has a higher compression ratio. Finally, the optimization target is shown as follows, where KLreg denotes the Kullback-Leibler Divergence (Kullback and Leibler, 1951) regularization term.

LOF-VAE = ∥F:→1 − Fˆ:→1∥1 + ∥F:→K − Fˆ:→K∥1 + KLreg. (6)

- 4.3 Latent Optical Flow Diffusion Model

With our trained OF-VAE, consisting of E and D, we efficiently compress optical flow into a low-dimensional latent space. In this section, we provide a brief overview of the Latent Optical Flow Diffusion Model (LOFDM), which generates optical flows within the latent space. We also discuss the design of the conditioning mechanism and outline the fundamental approach for utilizing these conditions.

Basic architecture. As shown in Fig. 4, we visualize the inference phase of LOF-DM. The backbone ϵθ is parametered by θ and realized by a DiT model (Peebles and Xie, 2023). Overall, we extract the semantic information of I1 and IK via the CLIP (Radford et al., 2021) and calculate the linear flow between existing

- frames as a prior to help the model learning. Notably, this linear flow can be directly computed from the first frame and the last frame, thereby providing a coarse estimation of the ground-truth optical flow. During the inference phase, the sampling target is:

z ∼ pθ(z | I1,IK), (7) where z denote the sampled optical flow in the latent space, and t is uniformly sampled from {1,··· ,T}

Existing frames guidance. Interpreting the semantics between the existing frames is important for optical flow generation. In detail, we utilize CLIP to extract semantic information from the given image to achieve image-to-video generation, we apply the same way to extract the useful information. Considering the fact that our task requires more fine-grained information, we adapt the ViT (Dosovitskiy et al., 2020) architecture used in CLIP by replacing the global [CLS] token with features derived from individual patch tokens. This approach enables us to capture richer and more detailed representations, providing a better foundation for tasks that rely on fine-grained information. Similar to CogVideoX (Yang et al., 2024), we concatenate the embeddings of both CLIP features and the optical flow in the latent space at the input stage to better align visual and semantic information.

Linear optical flow initialization. Directly generating optical flow from semantic information can be quite challenging, since the complexities and nuances of motion often require a level of detail that pure semantic representations may not capture effectively. Therefore, we consider introducing linear optical flow as a prior to assist this process. While linear optical flow may not be entirely precise, it provides useful information that can help guide the learning model, allowing it to better approximate the underlying motion dynamics and improve overall performance. Formally, we calculate linear flow as follows:

FˆkL→1 = kFK→1, FˆkL→K = (1 − k)F1→K, (8) where k = {2,··· ,K − 1}. These estimated flows are encoded into the latent space using the OF-VAE, expressed as zl = E([FˆkL→1,FˆkL→K]). During both the training and inference phases, zl is concatenated along the channel dimension to support optical flow prediction. Although real-world motions are inherently more complex, we find that initializing with linear motion improves the denoising learning process and accelerates training convergence.

- 4.4 Post-Hoc Refinement

After estimating the optical flow Fˆ:→1 and Fˆ:→K using the OF-VAE and LOF-DM given I1 and IK, we can reconstruct the intermediate frames Iˆk as follows:

Iˆk = P(W(I1,Fˆk→1),W(IK,Fˆk→K)), (9)

where P and W denotes the reconstruction method and backward warping, respectively. Previous methods like RIFE use convolutional neural networks to refine the warped results, limiting their ability to generate diverse and detailed content. Instead, we propose our MotionControlNet, utilizing the strong video prior learned in current Image-to-Video diffusion models to refine the intermediate frames, leading to better realistic generation. Formally, the process is represented as follows:

v ∼ pϕ(V | I1,IK,P,W(I1,Fˆ:→1),W(IK,Fˆ:→K)), (10)

where P and v denote the given text and the final generation frames, respectively. In our experiments, we observe that setting an appropriate prompt can significantly enhance the model’s generative capabilities, leading to improved results.

MotionControlNet. Inspired by ControlNet (Zhang et al., 2023), which can inject different conditions to existing image diffusion models, we propose the MotionControlNet to generated videos with motion guidance. Formally, we use the CogVideoX-5b-I2V (Yang et al., 2024) as the basic model and introduce additional trainable zero convolution layers. The complete MotionControlNet then computes:

(I1,IK,P,W(I1,Fˆ:→1),W(IK,Fˆ:→K)). (11) where Fϕ1

y = Fϕ1

(I1,IK,P) + Zϕ2

and y denote the Image-to-Video foundation model parameterized by ϕ1, MotionControlNet parameterized by ϕ2, and output feature, respectively. We inject the motion information via the backward warping on key frames, allowing for precise alignment of the generated frames with the motion dynamics of the video. The backward warping process effectively transfers spatial and temporal information from the key frames to the intermediate frames, ensuring smooth transitions and realistic motion generation. By incorporating motion information into the diffusion model, our generated results demonstrate enhanced motion and contextual consistency in comparison to models that generate frames solely using the pair of key frames without integrating motion data. This strategic injection of motion information facilitates a more coherent frame generation process, ultimately leading to superior visual fidelity and continuity in the generated sequences.

, Zϕ2

## 5 Experiment

- 5.1 Experimental Setup

For key frame generation, we randomly select 600k text-video pairs based on aesthetic scores and the degree of motion throughout the videos from our self-collected data. During the fine-tuning phase, we sample uniformly at intervals of 16 frames across the entire video, resulting in the formation of video clips comprising a total of 18 frames. In the intermediate frame generation phase, the OF-VAE and LOF-DM are trained using our self-collected dataset with 50M samples, and ground-truth optical flows are estimated through the RAFT (Teed and Deng, 2020). For post-hoc refinement, we further filter an additional 500k high-quality samples based on resolution and aesthetic scores from our self-collected data.

Our LumosFlow supports two long video generation resolutions, producing videos consisting of 273 frames, which include 18 key frames and 16 intermediated frames between each pair of key frames. One pipeline is optimized for lower resolution at 256×256 pixels, while the other is designed for higher resolution at 480×640 pixels.

- 5.2 Results on Long Video Generation

Quantitative comparison We compare LumosFlow with other long video generation methods FreeLong (Lu

- et al., 2025), FreeNoise (Qiu et al., 2023), and Video-Infinity (Tan et al., 2024) and report FVD, FID, SSIM, Subject Consistency (S-C), Motion Smoothness (M-S), Temporal Flickering (T-F), and Dynamic Degree (D-D). For a fair comparison, we construct a small test set containing 100 high-quality text-video pairs and apply these methods to generate long videos corresponding to these texts. As shown in Tab. 1, LumosFlow demonstrates outstanding performance in FVD and FID, suggesting that it effectively generates high-quality and diverse video content. Moreover, LumosFlow achieves the best performance in Motion Smoothness and

Frame 1 Frame 26 Frame 243 Frame 272

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

LumosFlow

Frame 1 Frame 112 Frame 128

Frame 16

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

FreeLong

Frame 1 Frame 112 Frame 128

Frame 16

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

FreeNoise

Frame 1 Frame 32 Frame 224 Frame 256

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Video-Infinity

#### Figure 5 The generated long videos via LumosFlow, FreeLong, FreeNoise, Video-Infinity by given the prompt “A man with curly hair, dressed in a black shirt and wearing a white virtual reality headset......".

Dynamic Degree, indicating that our method not only ensures smooth and natural motion transitions but also captures a high level of diversity in the generated video sequences. This highlights LumosFlow’s ability to produce videos with both realistic motion and a broad range of dynamic variations, making it ideal for applications that require both quality and diversity, such as animation, game development, and video synthesis. Although FreeLong and FreeNoise achieve excellent performance in Subject Consistency, both methods show subpar performance in Dynamic Degree, which confirms that these methods suffer from temporal repetition to some extent. Additional generated videos for each method can be found in the supplementary materials.

Human study For human study, we collect 14 videos and consider Text Alignment (T-A), Frame Consistency (F-C), Dynamic Degree (D-D), and Video Quality (V-Q) metrics, ranging from 1 (very low) to 5 (very large). In addition, all users are required to choose the best video generated by different methods, referring to Preference Rate (P-R). As shown in Tab. 2, LumosFlow achieves the best performance among all the metrics. Specifically, LumosFlow achieves the highest score of Dynamic Degree and Video Quality at the same time, indicating the generated video has large frame-to-frame differences, but the overall quality is high.

Frame 1 Frame 4 Frame 7 Frame 8 Frame 9 Frame 10 Frame 13 Frame 17

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

LumosFlow

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

RIFE

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

LDMVFI

#### Figure 6 The generated intermediate frames referring to the Frame 1 and Frame 17. We randomly select some frames for visualization.

- Table 1 Experimental results on different evaluation metrics for long video generation. FVD ↓ FID ↓ SSIM ↑ S-S ↑ M-S ↑ T-F ↑ D-D↑

FreeLong (Lu et al., 2025) 1829.30 482.47 0.2597 0.9843 0.9752 0.9565 0.2175 FreeNoise (Qiu et al., 2023) 2175.56 481.69 0.2167 0.9778 0.9722 0.9672 0.3142 Video-Infinity (Tan et al., 2024) 1788.95 483.33 0.3058 0.9342 0.9566 0.9542 0.5470

LumosFlow 912.83 479.56 0.2713 0.9541 0.9898 0.9762 0.5700

- Table 2 Human study of Long video generation. T-A F-C D-D V-Q P-R

FreeLong 2.18 2.73 1.92 2.07 0.85% FreeNoise 2.37 2.03 2.52 1.83 0.00% Video-Infinity 2.44 2.11 2.46 1.88 0.85%

LumosFlow 4.14 3.69 3.44 3.91 98.2%

Quantitative results on OF-VAE. For OF-VAE, we achieve 32× compression in the spatial dimension and 4× compression in the temporal dimension. As shown in Tab. 3, we report the End-Point Error (EPE) between the reconstructed optical flow and the truth flow, based on a small self-collected validation set. For comparison, we also present the EPE between the linear flow and the truth flow. Despite the high compression ratios, the optical flow reconstructed by OF-VAE is more accurate than that obtained through direct use of linear optical flow. Experiments in Sec. 5.3 have shown that our OF-VAE can reconstruct sufficiently accurate optical flow for the motion generation.

Visualization of generated frames. As shown in Fig. 5, we generate long videos based on a specific prompt using LumosFlow, FreeLong, FreeNoise, and Video-Infinity. Compared to the other methods, LumosFlow demonstrates significant movement between the various frames, indicating dynamic activity throughout the sequence. Despite this motion, the core elements of the video are consistently maintained, ensuring visual coherence. In contrast, the videos generated by the other methods exhibit temporal repetition, existing minimal motion across different frames. More generated results are in the Appendix.

we visualize the generated intermediate frames by inputting Frame 1 and Frame 17. LumosFlow adeptly captures complex and nonlinear motion. For instance, the movement of the woman’s hand over her legs exhibits a considerable range of motion, which our method models with notable accuracy. For comparative analysis, we also visualize the interpolation results generated by RIFE and the diffusion-based method LDMVFI. Both of these methods struggle with frame interpolation in scenarios involving significant motion, resulting in apparent inconsistencies in the generated frames. For example, the woman’s arm and face are distorted in the frames generated via LDMVFI. These results further indicate that current intermediate frame generation methods are unable to effectively handle interpolation in the presence of significant motion. More generated results are in the Appendix.

Moreover, we also visualize the generated key frames in Fig. 7. We find that there is a large motion between different frames, which shows that our LMTV-DM can better generate videos with lower FPS.

- 5.3 Results on Video Frame Interpolation

Not only is it applicable to long video generation, our proposed method is also applicable to the traditional VFI task, i.e., generating intermediate frames via the given first and last frames. Formally, we re-train the LOF-DM and the MotionControlNet to adapt to the 7 frames interpolation based on 256 × 256 resolution, which is the most common setting for the VFI task (Jain et al., 2024). In addition, to verify the importance of our motion guidance, we train an additional generation model, generating intermediate frames by simply giving first and last frames, denoted as Motion-free. As shown in Tab. 4, compared to existing methods, LumosFlow shows competitive performance across various metrics, particularly in terms of PSNR and LPIPS on the Davis-7 and UCF101-7 datasets (Jain et al., 2024). The FVD metrics further reinforce the superiority of LumosFlow in generating high-quality intermediate frames. In addition, the performance of the Motion-free is notably inferior, highlighting the critical role of motion guidance in video interpolation.

- Table 3 Reconstruction results of optical flow are presented, with EPE (First) and EPE (Last) indicating the Error of the optical flow computed based on the first frame and the last frame, respectively.

EPE (First) ↓ EPE (Last) ↓

Linear flow (256 × 256) 1.670 1.504 OF-VAE (256 × 256) 0.520 0.592

Linear flow (480 × 640) 2.982 2.808 OF-VAE (480 × 640) 0.744 0.839

Key frame 1 Key frame 3 Key frame 5 Key frame 7 Key frame 9

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

#### Figure 7 Generated key frames via LMTV-DM. We observe that different key frames can represent a significant range of motion.

Quantitative results on generated optical flow.

First frame Intermediate frame Last frame

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

To assess the quality of the generated optical flow, we filtered 100 samples from the DAVIS-7 dataset that exhibited significant flow strength. We measured the discrepancy between the generated flow based on the provided first and last frames and the ground-truth flow. For comparison, we also estimated the flow of intermediate frames using RIFE and linear mapping techniques. As shown in Tab. 5, LOF-DM generates optical flow with greater accuracy compared to methods that rely solely on linear flow or those estimated by RIFE. Furthermore, we observed that the initialization with linear flow is critical for LOF-DM, facilitating the training process. When this initialization is omitted, LOF-DM struggles to predict the correct optical flow.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

𝐹 → 

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

𝐹 → 

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

𝐹→ 

In addition, we randomly select a video and extract the first and last frames as a pair of key frames to generate the possible motion between them, as shown in Fig. 8,. For the given sample, it can be clearly seen that LOFDM models the movement of the arm. Moreover, the direction of the arm’s movement has changed, indicating the difference between our generated optical flow and linear optical flow.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

𝐹→ 

Figure 8 The optical flow generated from first and last frames. Fˆ→1 and Fˆ→9 denote the generated optical flow from the first frame and last frame, respectively. F→1 and F→9 denote the OF-VAE reconstructed optical flow from the first frame and last frame.

- Table 4 Quantitative results on the Davis-7 and UCF101-7 datasets. Motion-F and Motion-G denote Motion-Free and Motion-Guidance, respectively

Davis-7 UCF101-7

PSNR↑ LPIPS↓ FVD↓ PSNR↑ LPIPS↓ FVD↓ AMT (Li et al., 2023) 21.09 0.254 234.5 26.06 0.1442 344.5 RIFE (Huang et al., 2022) 20.48 0.258 240.04 25.73 0.1359 323.8 FILM (Reda et al., 2022) 20.71 0.2707 214.8 25.90 0.1373 328.2 LDMVFI (Danier et al., 2024) 19.98 0.2764 245.02 25.57 0.1356 316.3 VIDIM (Jain et al., 2024) 19.62 0.2578 199.32 24.07 0.1495 278

- Motion-F 19.41 0.1949 162.19 23.41 0.1307 301.61

- Motion-G 19.72 0.1872 157.94 24.52 0.1176 236.54

- Table 5 The EPE calculated between the generated flow and the ground truth flow. Avg (First) and Avg (Last) represent the EPE of the optical flow computed based on the first frame and the last frame, respectively. We also extract the middle frame and calculate the EPE separately, denoted as Mid (First) and Mid (Last). In addition, † represents LOF-DM without linear flow initialization.

Avg (First) Avg (Last) Mid (First) Mid(Last)

Linear flow 1.342 1.297 1.477 1.493 RIFE (Huang et al., 2022) - - 1.387 1.422

LOF-DM† 10.171 8.728 9.993 10.665 LOF-DM 1.306 1.243 1.362 1.375

- 6 Conclusion

In this paper, we revisit the long video generation and propose LumosFlow that effectively decouples the process into key frame generation and video frame interpolation. By leveraging the LMTV-DM, we generate key frames that encapsulate significant motion intervals, promoting content diversity and enhancing the overall narrative flow of the videos. To tackle the complexities involved in interpolating contextual transitions, we further refine the intermediate frame generation by integrating motion generation with post-hoc refinement. The LOF-DM facilitates the synthesis of complex optical flows between key frames, while MotionControlNet enhances the quality of the interpolated frames, ensuring continuity and coherence in motion. In the future, we plan to expand LumosFlow to enable the generation of longer videos, such as those consisting of 1,000-2,000 frames.

References

Wenbo Bao, Wei-Sheng Lai, Chao Ma, Xiaoyun Zhang, Zhiyong Gao, and Ming-Hsuan Yang. Depth-aware video frame interpolation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3703–3712, 2019.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei Efros, and Tero Karras. Generating long videos of dynamic scenes. Advances in Neural Information Processing Systems, 35:31769–31781, 2022.

Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023.

Duolikun Danier, Fan Zhang, and David Bull. Ldmvfi: Video frame interpolation with latent diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 1472–1480, 2024.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and time-sensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022.

William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. Advances in Neural Information Processing Systems, 35:27953–27965, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022a.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022b.

Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In European Conference on Computer Vision, pages 624–642. Springer, 2022.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Siddhant Jain, Daniel Watson, Eric Tabellion, Ben Poole, Janne Kontkanen, et al. Video interpolation with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7341–7351, 2024.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics, 22

(1):79–86, 1951. Jaihyun Lew, Jooyoung Choi, Chaehun Shin, Dahuin Jung, and Sungroh Yoon. Disentangled motion modeling for video frame interpolation. arXiv preprint arXiv:2406.17256, 2024.

Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9801–9810, 2023.

Jingyun Liang, Yuchen Fan, Kai Zhang, Radu Timofte, Luc Van Gool, and Rakesh Ranjan. Movideo: Motion-aware video generation with diffusion model. In European Conference on Computer Vision, pages 56–74. Springer, 2025.

Chunxu Liu, Guozhen Zhang, Rui Zhao, and Limin Wang. Sparse global matching for video frame interpolation with large motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19125–19134, 2024.

Ziwei Liu, Raymond A Yeh, Xiaoou Tang, Yiming Liu, and Aseem Agarwala. Video frame synthesis using deep voxel flow. In Proceedings of the IEEE international conference on computer vision, pages 4463–4471, 2017.

Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37:131434–131455, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless. Film: Frame interpolation for large motion. In European Conference on Computer Vision, pages 250–266. Springer, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

- 2022a.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

- 2022b.

Zhenxiong Tan, Xingyi Yang, Songhua Liu, , and Xinchao Wang. Video-infinity: Distributed long video generation. arXiv preprint arXiv:2406.16260, 2024.

Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020.

Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023a.

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023b.

Zhifei Xie, Daniel Tang, Dingwei Tan, Jacques Klein, Tegawend F Bissyand, and Saad Ezzini. Dreamfactory: Pioneering multi-scene long video generation with a multi-agent framework. arXiv preprint arXiv:2408.11788, 2024.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. 2023.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023.

Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6463–6474, 2024.

Guozhen Zhang, Chunxu Liu, Yutao Cui, Xiaotong Zhao, Kai Ma, and Limin Wang. Vfimamba: Video frame interpolation with state space models. arXiv preprint arXiv:2407.02315, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

# Appendix

Overview

This appendix presents comprehensive experimental details, evaluation details, and more visualization results. The content is organized into five main sections:

- • Sec. A presents the training and inference details of our three diffusion models, LMTV-DM, LOF-DM, and MotionControlNet.
- • Sec. B presents the evaluation details of quantitative results and human study.
- • Sec. C visualize more examples of generated intermediate frames via LumosFlow.
- • Sec. D visualize more examples of generated optical flows via LumosFlow.
- • Sec. E visualize more examples of generated long videos via LumosFlow.

- A Training and Inference Details

LMTV-DM LMTV-DM is fine-tuned based on the CogVideoX-5b Text-to-Video diffusion model. The training set, consisting of 600,000 samples, is filtered according to aesthetic scores and motion degrees, which are estimated using one-align and optical flow, respectively. During the fine-tuning phase, we randomly sample

- frames at fixed intervals and apply the CogVideoX video Variational Autoencoder (VAE) to encode these frames. In contrast to the original CogVideoX, which encodes four frames simultaneously, we encode the selected frames in a frame-by-frame manner, as larger intervals result in lower motion consistency. The overall encoding pipeline is depicted in Fig. 9.

Fixed interval

Frames

Frames (fixed interval)

CogVideoX VAE Encoder

Latent

- Figure 9 The overall encoding pipeline of LMTV-DM. Given a set of frames, the selected frames (indicated in dark green) are sampled at fixed intervals. Subsequently, we employ the CogVideoX VAE to encode these frames in a frame-by-frame manner, resulting in the corresponding latent representations (shown in blue), which are then utilized in the conducted diffusion pipeline.

LOF-DM The LOF-DM model is developed based on the DiT architecture, with hyperparameters detailed in Tab. 6. During the training phase, we employ a regularization strategy where we randomly drop semantic features and linear optical flow with a probability of 10%. This technique aims to enhance the model’s robustness by preventing overfitting. For the training process, we set the learning rate to 5 × 10−5 and utilize a batch size of 128, ensuring efficient model convergence. During the inference phase, we implement classifier-free guidance on the semantic features to improve the quality of the generated outputs.

Table 6 Hyper-parameters of LOF-DM.

LOF-DM Number of layers 30

Attention heads 32

Hidden size 1920 Position encoding sinusoidal Time Embedding Size 256

Training Precision Float32

MotionControlNet The MotionControlNet is developed based on the CogVideoX-5b Image-to-Video diffusion model. In the training process, we configure the learning rate to 1 × 10−4 and utilize a batch size of 8. Additionally, we randomly apply a negative prompt with a probability of 10%. This strategy is implemented to enhance the robustness of the model and improve its overall performance in video generation tasks.

- B Evaluation Details

For a fair comparison, we use the commonly used VBench Huang et al. (2024) to evaluate the quality of generated long videos as well as the commonly used metrics FVD, FID, and SSIM. VBench is designed specifically for benchmarking the performance of video generation models, providing a comprehensive suite of evaluation tools that facilitate the analysis of both qualitative and quantitative aspects of generated videos. In actual practice, we follow the instructions provided by VBench to divide the long videos into distinct sets of frames and evaluate each set separately.

For the human evaluation, we randomly select 14 prompts and generate long videos using various methods. We then invite 8 users to assess these videos based on four key metrics: Text Alignment, Frame Consistency, Dynamic Degree, and Video Quality. Text Alignment evaluates how accurately the videos correspond to the prompts, while Frame Consistency measures the stability of visual elements across frames. Dynamic Degree analyzes the level of motion captured in the videos, and Video Quality assesses overall visual appeal, including clarity and color fidelity. This comprehensive evaluation approach allows us to better understand the strengths and weaknesses of the generated videos, facilitating the improvement of our video generation methods.

- C More Generated Intermediate Frames We present more generated intermediate frames of LumosFlow in Fig. 10.
- D More Generated Optical Flows

We present more generated intermediate optical flows of LumosFlow in Fig. 11. Given the first and last frames, a rational movement involves the man’s arm moving from the top left to the bottom right. The generated optical flow effectively captures and models this movement, demonstrating improved coherence and realism in the depiction of motion.

- E More Generated Long Videos

We present more generated long videos of LumosFlow in Fig. 12. The generated long videos exhibit a high degree of smoothness and feature complex movement dynamics.

Frame 1 Frame 4 Frame 7 Frame 8 Frame 9 Frame 10 Frame 13 Frame 17

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

A golden retriever playing in the snow…...

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

A panda playing on a swing set……

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

A car is moving on the snow……

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

A dog is chasing on a sofa……

#### Figure 10 Generated intermediate frames via LumosFlow.

Frame 1 Frame 17

Intermediate frames

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

𝐹 → 

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

𝐹 →  

#### Figure 11 Generated optical flows and intermediate frames via the given first (Frame 1) and last frames (Frame 17). The first and last frames are generated by the LMTV-DM and the optical flows (Fˆ→1 and Fˆ→17) are generated by the LOF-DM. The intermediate frames are generated by the MotionControlNet.

Frame 1 Frame 18

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Frame 261 Frame 272

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

A dog is chasing on a sofa……

Frame 1 Frame 18

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Frame 261 Frame 272

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

A man with curly hair, dressed in a black shirt and wearing a white virtual reality headset……

Frame 1 Frame 18

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Frame 261 Frame 272

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

A woman with long, wavy hair tied back, wearing a sheer, light blue top with thin straps, ……

Frame 1 Frame 18

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Frame 261 Frame 272

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

A couple dances intimately on a rooftop, captured from an eye-level camera angle…….

Frame 1 Frame 18

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Frame 261 Frame 272

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

A woman in a brown jacket and black helmet sits on a brown horse in an indoor equestrian arena, …….

#### Figure 12 Generated long videos via LumosFlow

