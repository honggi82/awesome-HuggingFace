# arXiv:2408.13413v1[cs.CV]24Aug2024

## TVG: A Training-free Transition Video Generation Method with Diffusion Models

### Rui Zhang1,2,3* Yaosen Chen1,4†, Yuegen Liu1, Wei Wang1, Xuming Wen1, Hongxia Wang2,3‡

1Sobey Media Intelligence Laboratory, Chengdu, China 2School of Cyber Science and Engineering, Sichuan University, Chengdu, China 3Key Laboratory of Data Protection and Intelligent Management (Sichuan University), Ministry of Education, China 4University of Electronic Science and Technology of China, Chengdu, Sichuan, 611731 China

##### Abstract

Transition videos play a crucial role in media production, enhancing the flow and coherence of visual narratives. Traditional methods like morphing often lack artistic appeal and require specialized skills, limiting their effectiveness. Recent advances in diffusion model-based video generation offer new possibilities for creating transitions but face challenges such as poor inter-frame relationship modeling and abrupt content changes. We propose a novel training-free Transition Video Generation (TVG) approach using videolevel diffusion models that addresses these limitations without additional training. Our method leverages Gaussian Process Regression (GPR) to model latent representations, ensuring smooth and dynamic transitions between frames. Additionally, we introduce interpolation-based conditional controls and a Frequency-aware Bidirectional Fusion (FBiF) architecture to enhance temporal control and transition reliability. Evaluations of benchmark datasets and custom image pairs demonstrate the effectiveness of our approach in generating high-quality smooth transition videos. The code are provided in https://sobeymil.github.io/tvg.com.

### Introduction

Transition videos have been fundamental in media production, traditionally using techniques such as morphing (Wolberg 1998) to seamlessly connect segments, thus improving flow, visual appeal, and narrative coherence. With video emerging as the dominant medium on platforms such as TikTok and Kuaishou, the demand for innovative content formats such as vlogs and short films has surged. This trend underscores the critical need for effective and engaging transitions. However, traditional transition techniques often lack artistic appeal, fail to fully engage viewers, and require specialized production skills, posing challenges for creators.

Recent advancements in diffusion model-based image (Song, Meng, and Ermon 2021; Rombach et al. 2022) and video generation (Liu et al. 2024) have introduced novel methods for creating transition videos by generating intermediate frames between segments. Although these ap-

*This work was done during an internship at Sobey Media Intelligence Laboratory

†Co-corresponding authors ‡Co-corresponding authors

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- (a）

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

- (b）

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

- (c）

Figure 1: Failed results from some commercial products: (a) LUMA AI, (b) Jimeng AI, (c) Kling AI.

proaches offer more creative and accessible transition generation, they face limitations, especially when there is a significant disparity between frames. Existing techniques, including both open-source models (Zhang et al. 2024a; Chen et al. 2024; Xing et al. 2023) and commercial products, often produce suboptimal transitions, leading to abrupt changes, simple fades, or even the absence of transitions, as shown in Figure 1. Additionally, current video generation models are typically limited to producing short clips, making effective transition generation crucial for maintaining continuity in longer videos (Yuan et al. 2024). The inability to generate consistent, high-quality transitions could significantly hinder the broader adoption of video generation technologies.

Several factors contribute to these challenges. First, some models (Zhang et al. 2024a; Chen et al. 2024) extend and optimize image-level diffusion models, which often lack robust inter-frame relationship modeling, particularly when the primary subjects in the frames differ significantly. Although these models generally avoid abrupt transitions, they often fail to produce dynamic results, leading to outputs with simple fading effects or disruptive artifacts. Second, while video-level diffusion models (Xing et al. 2023) attempt to establish inter-frame relationships through spatio-temporal attention, these can be unstable due to issues like conditional image leakage (Zhao et al. 2024), cross-attention leakage (Yang et al. 2024), and improper activation of attention mechanisms, resulting in abrupt transitions or anomalous content.

To address these challenges, we propose a novel, trainingfree Transition Video Generation (TVG) approach based

on video-level diffusion models. Unlike methods like SEINE (Chen et al. 2024), which require full retraining, or DiffMorpher (Zhang et al. 2024a), which relies on LoRA-based (Hu et al. 2022) fine-tuning, our approach generates video transitions without any additional training. While existing image-to-video pre-trained models demonstrate high-quality generative capabilities, they often lack effective inter-frame relationship control for transitions. Our approach avoids redundant retraining, reducing computational costs and enabling broader applicability to future, potentially more advanced video generation algorithms.

We implemented several optimizations based on DynamiCrafter (Xing et al. 2023). Building on prior work (Zhang et al. 2024a; Jang et al. 2024), we introduce a novel conditional control approach using interpolation techniques. Specifically, we blend images within the conditional latent space, enhancing motion dynamics. Additionally, we utilize spherical linear interpolation (SLERP) for prompt embeddings, enabling precise control and fine-grained adjustments of each frame’s content, addressing the lack of dynamic variation caused by conditional image leakage. We propose that the content between transition frames should adhere to a smooth stochastic process to balance dynamic motion with content continuity. To achieve this, we employ Gaussian Process Regression (GPR) to model the latent representations of the initial and final frames. The intermediate content is then mapped into this smooth feature space, improving transition quality by preventing abrupt changes while maintaining dynamic continuity. Furthermore, we observed that existing image-to-video models can produce vastly different transition videos depending on the content of the initial and final frames, even when simply swapping the two. To address this variability and enhance the reliability of transition generation, we introduced a Frequence-aware Bidirectional Fusion (FBiF) architecture. We evaluated our algorithm on existing benchmarks, including MorphBench (Zhang et al. 2024a) and TC-Bench-I2V (Feng et al. 2024), as well as on custom image pairs. The experimental results demonstrate the effectiveness of our proposed modules. Our contributions are summarized as follows:

- 1. We propose a training-free transition video generation method based on Gaussian Process Regression (GPR) with video-level diffusion models.
- 2. We introduce interpolation-based conditional controls to enhance temporal control and ensure smooth video transitions.
- 3. We develop a Frequence-aware Bidirectional Fusion (FBiF) architecture to further improve the reliability of video transition generation.

### Related work

Training-free Video Generation. ControlVideo (Zhang et al. 2023), adapted from ControlNet, improves video generation by introducing cross-frame interaction, interleavedframe smoothing, and hierarchical sampling. MotionClone (Ling et al. 2024) employs temporal attention in video inversion for motion representation and cloning, while VideoElevator (Zhang et al. 2024b) decomposes sampling

into temporal motion refinement and spatial quality enhancement. MVOC (Wang et al. 2024) propose a multiple video object composition method based on diffusion models with DDIM inversion features injection. These methods enhance video generation and editing capabilities by building on existing models, reducing the need for extensive retraining.

Transition Video Generation. Traditional video transitions, such as fade, dissolve, wipe, or cut, are essential for seamless scene changes in video editing. Morphing (Wolberg 1998) offers smooth transitions by finding pixel-level similarities and estimating offsets. Generative models (Van Den Oord, Vinyals et al. 2017) have advanced this by capturing semantic similarities through latent code interpolation, with applications in style transfer (Chen et al. 2018) and object transfiguration (Sauer, Schwarz, and Geiger 2022). SEINE (Chen et al. 2024) introduced a random-mask video diffusion model for text-guided transitions between different scenes, while DiffMorpher (Zhang et al. 2024a) uses LoRA parameter interpolation with image diffusion models for smooth semantic transitions. Despite these advances, achieving natural video transitions remains challenging.

### Preliminary

#### Diffusion Models

Diffusion models (Sohl-Dickstein et al. 2015; Ho, Jain, and Abbeel 2020; Rombach et al. 2022) are a class of probabilistic generative models which learns a data distribution by denoising noisy. The forward process of diffuses the data samples with a Markov process contains T timesteps: The forward process is gradually adding noise to the data sample x0 to xt with a Markov process contains T timesteps:

q(xt|xt−1) = N xt; 1 − βtxt−1,βtI), (1a) q(xt|x0) = N xt;√αtx0,(1 − αt)I , (1b)

where βt is a predefined variance schedule, αt = ts=1 αs, is derived from the variance noise schedule, and αt = 1−βt. The reverse process begins from the noise and transitions to-

wards the original data q(x0). The reverse denoising process obtains less noisy data xt−1 from the noisy input xt at each timestep:

1 − αt √1 − α¯t

1 √αt

(xt,t)),σt2)

(xt −

(xt−1|xt) = N xt;

ϵθ

pθ

l

l

(2) where ϵθ

(xt,t) is the denoising function to estimate the noise added at each step and θl is the learnable network parameters.

l

#### Diffusion Models for Transition Video Generation

Given two input images, x0 and xS, the goal of transition video generation is to produce a video sequence Xt = {xi}Si=1, starting with x0 and ending with xS. The intermediate frames, x1 to xS−1, should exhibit smooth transitions that blend the characteristics of both input images, forming a coherent sequence. This process can be formalized as a conditional distribution pθ

(Xt | x0,xS), modeled by a conditional diffusion process.

l

In video generation, Latent Diffusion Models (LDMs) (Rombach et al. 2022; Zhou et al. 2022; An et al. 2023) are commonly used to reduce computational complexity by fitting the conditional distribution within the latent space, where denoising occurs. The initial and final frames, x0 and xS, anchor the sequence, with intermediate frames initialized by adding Gaussian noise. These frames are encoded into a latent representation zt ∈ RS×C×H×W, where S, C, H, and W represent the number of frames, channels, height, and width, respectively. A backward denoising process, such as DDIM (Song, Meng, and Ermon 2021), is applied to obtain the clean latent representation z0, which is then decoded into the video sequence Xt. The reverse process typically incorporates additional denoising conditions to enhance the quality of the generated video, such as text prompt latents, ctxt, and the latent of the initial frame, cimg.

Our base model, DynamiCrafter (Xing et al. 2023), further leverages FPS as a denoising condition. It uses CLIP (Radford et al. 2021) to encode text prompts and the initial frame, while employing a frame-wise Variational Auto-Encoder (VAE) architecture for the video encoder and decoder. The denoising process is carried out using a U-Net architecture.

#### Gaussian Process

A Gaussian process (GP) (Williams and Rasmussen 2006) is a collection of random variables, any finite number of which have a joint Gaussian distribution. Consider a data set with inputs and targets consisting of N data-points as {xi,yi}Ni=1, where the inputs are denoted by X ={x1,...,xN} , and targets by Y ={y1,...,yN}. The goal of GP is to learn an unknown function f that maps elements from input space to a target space, which can provide predictive distributions on test points X∗ ={x∗1,...,x∗M}. Assuming the GP prior on f(x) with some additive Gaussian white noise with variance σ2:

yi = f(xi) + ϵi,ϵi ∼ N(0,σ2) (3)

A Gaussian process regression (GPR) formulates the functional form of f by drawing random variables from a multi-variate normal distribution given by [f(x1),f(x2),...,f(xn)] ∼ N(µ,KX,X), with the mean of µ and the covariance matrix of KX,X. µi = µ(xi), where µ(·) is a mean function of the GP; (KX,X)ij = k(xi,xj), where k(·) is a kernel function of the GP. The conditional distribution at any unseen points X∗ is given by:

f∗|X∗,X,Y ∼ N(µf∗,Σf∗) (4) where

µf∗ = µX∗ + KX∗,X[KX,X + σ2I]−1Y (5) Σf∗ = KX∗,X∗ − KX∗,X[KX,X + σ2I]−1KX,X∗

(6)

### Method

Our framework is based on DynamiCrafter (Xing et al. 2023), a state-of-the-art image-to-video generative model.

However, DynamiCrafter faces challenges in video transition tasks, particularly when there are significant differences between the initial and final frames, often resulting in transitions that overly depend on the starting image and appear abrupt. To address these issues, we propose a training-free TVG method, as shown in Figure 2.

We optimized the pre-trained model during inference in three key areas. First, we refined both conditional images and prompts to enhance the controllability of the video generation process and reduce conditional image leakage. Second, we integrated Gaussian Process Regression (GPR) into the latent space to enforce temporal consistency between frames, improving inter-frame coherence and preventing abrupt transitions. Lastly, we introduced a Frequenceaware Bidirectional Fusion (FBiF) structure, which combines bidirectional generation with frequency-domain feature fusion. To ensure the final output’s visual quality, we apply close-distance frame selection at the final stage.

#### Interpolation-based Conditional Controls

In our experiments, we found that DynamiCrafter often produced abrupt transitions when using input images with significantly different content, a result of conditional image leakage. To mitigate this, we applied linear interpolation to blend the two input frames x0 and xS:

xb = β · x0 + (1 − β) · xS. (7) Setting β to 0.5 balances information from both frames, creating a blended image xb that integrates details from both inputs. Other values would favor one frame over the other, which is not suitable for our video transition task. This image is then processed by CLIP to extract feature embeddings, serving as the condition cimg. This approach leverages the pre-trained model without requiring additional training. In contrast, separate extraction and fusion of features using CLIP can create a semantic gap, complicating their integration in diffusion models and often leading to errors. Addressing this gap usually requires retraining a feature projection module, as done in StoryDiffusion (Zhou et al. 2024), but our method avoids this by fully utilizing the pre-trained model.

Traditional methods typically employ a single prompt for the entire video, limiting frame-wise control. To improve this, inspired by (Zhang et al. 2024a; Jang et al. 2024), we propose Temporal-Level Prompts Control. We begin by obtaining prompts t0 and tS for the initial and final frames, then use CLIP to extract their semantic features, Ftxt0 and FtxtS. These features are interpolated using spherical linear interpolation (SLERP).

sin((1 − α)θ) sinθ · Ftxt0 +

sin(αθ)

sinθ · FtxtS, (8) where α ∈ [0,1] is the interpolation parameter, and θ = cos−1 F

Ftxts =

0 txt·FtxtS

∥Ftxt0∥∥FtxtS∥ is the angle between the two vectors. We sample S weights uniformly between 0.9 and 0.1, corresponding to the temporal progression from the initial to the final frame, creating S interpolated semantic features. These

Forward Video

Output Video

[Figure 22]

Temporal-Level Prompts Control

[Figure 23]

Text Prompt A A vase sea of clouds

CLIP text encoder

Text Prompt B

CLIP text encoder Input A Input B

a winding river through patchwork fields

[Figure 24]

[Figure 25]

Denoising

[Figure 26]

[Figure 27]

[Figure 28]

VAE encoder

VAE decoder

[Figure 29]

SLERP Blending Image

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Query Transformer Close-distance Frame Selection

CLIP image encoder

Frequence-aware Bidirectional Fusion

[Figure 34]

[Figure 35]

······

Gaussian Space

[Figure 36]

Input A Input B

[Figure 37]

[Figure 38]

[Figure 39]

VAE encoder VAE decoder

[Figure 40]

[Figure 41]

Denoising

- Text Prompt A
- Text Prompt B

CLIP text encoder

a winding river through patchwork fields

[Figure 42]

CLIP text encoder

A vase sea of clouds

[Figure 43]

Temporal-Level Prompts Control

Reverse Video

Figure 2: Illustration of Our Proposed Training-free TVG Method.

features form ctxt ∈ RS×L×D, ensuring smooth transitions and minimizing semantic gaps, allowing for seamless integration into the diffusion model and reducing the likelihood of abrupt content changes in the video.

a video can be modeled as a smooth stochastic process. The objective is to derive a distribution function for this process, with the first frame as the observation input and the last frame as the prediction target.

In the latent space, the features of the first and last frames are extracted from the input images, while intermediate frames are generated by denoising Gaussian noise. This allows us to construct a stochastic process distribution function based on the initial and final frames, predicting intermediate frames to ensure coherent transitions that accurately reflect the relationship between the start and end frames.

#### Latent Space Gaussian Process Regression

The denoising U-Net employs temporal-level and spatiallevel Transformer modules, integrating self-attention and cross-attention to capture spatial relationships within frames and temporal relationships between frames. However, the spatial modules overlook inter-frame dependencies, while the temporal modules impose only limited inter-frame constraints through attention mechanisms. This results in poor controllability, leading to error propagation across frames and degrading the quality of transition videos.

To implement this, we leverage GPR to model the distribution function, utilizing its capability to flexibly capture the underlying data distribution and smoothly interpolate between frames while maintaining consistency with the input frames. Specifically, the first frame of z is defined as X ∈ RN×P, where X consists of N data points serving as input to GPR, as outlined in the Preliminary section. Similarly, the last frame of z is defined as Y ∈ RN×P, representing the target with N data points. We employ the Radial Basis Function as the kernel k(·) to formulate the distribution function f∗. The remaining frames of z are used as the input X∗ to the function, as shown in Equation 4. The mean µf∗ obtained from the function, as shown in Equation 5, is then used to refine the corresponding frame features.

To address these issues, we incorporate GPR into the latent space of the U-Net within video generation models. This integration with attention modules explicitly constrains inter-frame relationships, enhancing the overall video quality.

Given a temporal-level latent vector z ∈ RS×N×P, where

- N represents the vector size and P denotes the number of channels, we compute the attention mechanism with GPR for both temporal and spatial attention as follows:

z′ = Attention(z) + γ · z + (1 − γ) · GPR(z), (9)

This approach effectively maps features into a Gaussian space, aligning inter-frame relationships with a stationary random distribution, thereby enhancing both correlation and smoothness between frames. However, due to inherent

where γ ∈ [0,1] serves as a balancing factor, controlling the trade-off between different feature contributions. We hypothesize that the transition content between two frames in

differences between the original model’s feature distribution and the Gaussian distribution, a complete substitution of the original features would necessitate further training to achieve optimal performance. To reduce computational costs and avoid additional training, we integrate these features as a penalty term within the original attention mechanism, rather than fully replacing the original features. Despite this conservative approach, it significantly enhances inter-frame information exchange and smooths transitions between frames, while maintaining compatibility with the feature distribution of the pre-trained model.

to 0.1, controls the contributions of the low-frequency components. The high-frequency components are weighted by a fixed factor λfreq. To ensure consistent video generation, we propose a Close-distance Frame Selection mechanism. Starting with the first frame of the forward video as a reference, we reverse the sequence of the reversed video. For each subsequent frame, we compare the perceptual loss (Zhang et al. 2018) between the next frames of both videos relative to the previous frame. The frame with the smallest distance is selected as the next frame, and this process is repeated to generate the final video.

### Experiments

#### Frequence-aware Bidirectional Fusion

#### Experimental Setup

In practice, we observed significant quality differences in transition videos generated from two images, x0 and xS, depending on whether the sequence was processed in original or reverse order. As shown in Figure 3, this issue is linked to conditional image leakage, where the transition video tends to resemble the initial image more closely. To address the shortcomings in the latter part of the video, particularly the deviations from the final frame and the lack of smoothness, we propose a Frequency-aware Bidirectional Fusion (FBiF) approach to improve overall generation quality.

We implemented our Transition Video Generation approach using the pre-trained DynamiCrafter model (Xing et al. 2023). Most hyperparameters were kept at default settings, except for the FPS, which was set to 16 to enhance visual dynamics. The hyperparameter γ was set to 0.9, λfreq was 0.1. We used a DDIM (Song, Meng, and Ermon 2021) schedule with 10 steps during denoising. All experiments ran on a single NVIDIA A800 GPU.

For quantitative evaluation, we tested the MorphBench (Zhang et al. 2024a) and TC-Bench-I2V (Feng et al. 2024) datasets, generating 16 frames per video following established protocols (Chen et al. 2024; Zhang et al. 2024a; Feng et al. 2024). The video resolution was set to 512 × 512 for MorphBench and 320 × 512 for TC-Bench-I2V. For qualitative evaluation, we collected 10 image pairs from the Internet for comparison with other models and commercial products.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

#### Qualitative results

Figure 3: Samples of forward and reverse videos generated by DynamiCrafter, showing selected frames at 0, 5, 10, and 15. The top row shows the forward video, and the bottom row shows the reverse video.

To demonstrate the effectiveness of our algorithm, we have presented some of the results generated by our model in Figure 4. It can be seen that for different scenarios, our model exhibits strong video-transition capabilities, with the video maintaining dynamic elements while the content gradually transitions. Additionally, we compared our algorithm with several commercial products as shown, some of which were even unable to produce transition videos as shown in Figure 1. For more visualization results, we recommend readers refer to the supplementary materials.

As illustrated in Figure 2, we generate a forward transition video, then repeat the process by reversing the order of the input images and prompts fed into the video generation model. In the latent space, we apply GPR to map features into a unified Gaussian space, aligning the feature distributions of both the forward and reversed videos. Let zg = GPR(z), where zg ∈ RS×N×P; the reversed features are similarly defined as zgrev ∈ RS×N×P. Inspired by (Pan, Cai, and Zhuang 2022), we consider the averagepooled (avgpool) features to represent low-frequency components, while the max-pooled (maxpool) features capture high-frequency components. Since our goal is to emphasize the primary concepts in the video content rather than merely enhancing details, we employ a frequency-aware fusion method, combining the two feature sets along the temporal sequence as follows:

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Figure 4: Visualization of Our Generated Samples.

zfuseds = λs · avgpool(zgs) + (1 − λs) · avgpool(reverse(zgrev)s)

+ λfreq · maxpool(zgs) + λfreq · maxpool(reverse(zgrev)s).

#### Quantitative Results

(10)

where zfuseds represents the fused feature at frame s. The weighting factor λs, which varies across frames from 0.9

Evaluation on TC-Bench-I2V: We evaluated our model on the TC-Bench-I2V dataset using the Transition Comple-

tion Ratio (TCR, ↑) and TC-Score (↑) (Feng et al. 2024). These metrics utilize GPT-4 to provide assessments closely aligned with human perception. TCR measures the percentage of videos that successfully match the provided prompts, while TC-Score offers a comprehensive evaluation considering factors such as transition completion, object consistency, and handling of additional objects, reflecting user experiences more realistically.

and the first frame of another, enhancing continuity and fluidity. Although dynamic videos, which are favored for their rich visual appeal (Roggeveen et al. 2015; Steuer et al. 1995; Coyle and Thorson 2001), often introduce greater frame variation that can reduce quantitative performance, the enhanced user experience remains paramount.

Figure 6 illustrates content generated from a pair of images in the MorphBench dataset, using identical prompts across different methods. Our method and DiffMorpher employ separate prompts for each image, while SEINE and DynamiCrafter merge the descriptions into a single prompt, transitioning from input A to input B without introducing additional content. Figure 7 presents the perceptual loss (Zhang et al. 2018) between consecutive frames and final PPL, starting with input A at frame index 0 and ending with input B at frame index 17, covering both input images and generated frames. DiffMorpher achieves the lowest PPL, indicating superior quantitative performance. This is primarily because only a small portion of the generated frames retain perceptual similarity to input A, with the rest transitioning gradually toward input B in a low-dynamic manner, resulting in final frames that closely resemble the target. Although perceptual loss is relatively high during the initial dynamic phase, it remains low thereafter, enhancing DiffMorpher’s quantitative metrics. However, this approach results in a near-static user experience, making it difficult to effectively engage the audience.

- Table 1 presents a comparison of our algorithm with

three state-of-the-art methods across three video transition scenarios: Attribute Transition (Attribute), involving changes in color, shape, material, and texture; Object Relation Change (Object), focusing on object interactions like passing or striking; and Background Shifts (Background), such as transitions from day to night.

Our results demonstrate consistent superior performance across most metrics and scenarios. Notably, our method ranks second in TCR for the background scenario, likely due to the lower dynamic demands, which favor DiffMorpher, a frame-by-frame generation-based approach. However, in dynamic scenarios like Object Relation Change, our algorithm significantly outperforms others, aligning more closely with user experience expectations. Figure 5 illustrates comparative examples of generated results, with samples displayed every two frames.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

DynamiCrafter

In contrast, other methods, including ours, prioritize transforming image characteristics while maintaining high dynamic variability, leading to fluctuating PPL values throughout the sequence and impacting the final quantitative results. While videos with high dynamic variability often appeal more to users, SEINE and DynamiCrafter demonstrate more abrupt transitions. SEINE fully transitions to input B midway through the sequence with minimal change afterward, while DynamiCrafter adopts the style of input B only in the final quarter, lacking the smoothness needed for a seamless transition. In contrast, our method excels at gradually blending specific regions into the style of input B while preserving natural motion, achieving both high dynamic performance and smooth, visually coherent transitions throughout the entire sequence, underscoring the superiority of our approach.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

DiffMorpher

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

SEINE

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Ours

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

DynamiCrafter

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

DiffMorpher

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

SEINE

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Ours

- Figure 5: Comparison of Generated Sequences from TCBench Across Various Models.

Evaluation on MorphBench: Following previous work (Zhang et al. 2024a), we evaluated the fidelity and smoothness of video content using traditional metrics, specifically Frechet Inception Distance (FID, ↓) (Heusel et al. 2017) and Perceptual Path Length (PPL, ↓) (Karras et al. 2020). The results on the MorphBench dataset, summarized in Table 2, show that while our approach does not outperform all existing methods, it remains competitive with current diffusion-based models (Song, Meng, and Ermon 2021; Wang and Golland 2023; Zhang et al. 2024a; Chen et al. 2024; Xing et al. 2023). Notably, even traditional techniques like Warp&Blend surpass various generative algorithms in certain metrics, highlighting that superior quantitative scores do not necessarily translate to better visual effects in video transitions.

Another factor contributing to the significant perceptual loss between frames in highly dynamic content is the limitation of existing open-source video generation models, which struggle to produce longer video sequences. Typically, these models generate around 16 frames, and content beyond this length often becomes unstable. As our method is trainingfree, it currently cannot be extended to generate videos with more frames. Ensuring a high dynamic range under this frame limitation inevitably impacts the quality and smoothness of the video. However, this impact is acceptable under current circumstances. If more advanced open-source models for generating longer videos become available, our method should achieve further performance improvements.

Human Evaluation: To assess the real-world user experience of our algorithm, we conducted a Human Evaluation study using 22 curated videos: 9 from MorphBench,

Our primary objective is to develop an algorithm that seamlessly transitions between the final frame of one video

Table 1: Quantitative evaluation on TC-Bench-I2V, We report TCR and TC-Score to assess the transition videos.

Attribute Object Background Overall

Methods

TCR↑ TC-Score ↑ TCR ↑ TC-Score ↑ TCR ↑ TC-Score ↑ TCR ↑ TC-Score ↑ DynamiCrafter (2023) 16.55 0.745 13.91 0.707 25.56 0.795 16.89 0.738

SEINE (2024) 17.86 0.720 10.48 0.654 7.96 0.742 13.57 0.698 DiffMorpher (2024a) 41.82 0.844 19.57 0.765 50.00 0.819 34.45 0.810 Ours 41.82 0.877 30.44 0.822 38.89 0.864 36.98 0.853

Input A Input B

[Figure 140]

[Figure 141]

Generated Frames

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

DynamiCrafter

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

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

DiffMorpher

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

SEINE

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Ours

0 Frame Index 17

- Figure 6: Visualization of Wave Sequence from the MorphBench dataset compared with other methods. Our method achieves gradual changes while preserving motion.

Table 2: Quantitative evaluation on MorphBench, We report FID and PPL to assess the fidelity and smoothness of the transition videos, respectively. Bold indicates the best result, underline indicates the second best result.

Method

Metamorphosis Animation Overall FID ↓ PPL ↓ FID ↓ PPL ↓ FID ↓ PPL ↓

Warp&Blend (1998) 79.63 15.97 56.86 9.58 67.57 14.27

DGP (2022) 150.29 29.65 194.65 27.50 138.20 29.08 StyicGAN-XL (2022) 122.42 41.94 133.73 33.43 112.63 39.67

DDIM (2021) 95.44 27.38 174.31 18.70 101.68 25.38

DiffInterp (2023) 169.07 108.51 148.95 96.12 146.66 105.23 DynamiCrafter (2023) 87.32 42.09 43.31 11.16 69.13 33.84

DiffMorpher (2024a) 70.49 18.19 43.15 5.14 54.69 21.10 SEINE (2024) 82.03 47.72 48.25 16.26 67.60 39.33 Ours 86.92 35.18 42.99 12.46 64.05 29.08

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17

Frame Index

0

20

40

60

80

100

120

PerceptualLoss

DiffMorpher PPL: 27.0748

DynamiCrafter PPL: 60.9417

SEINE PPL: 58.3082

Ours PPL: 54.7672

- Figure 7: Perceptual Losses between Generated Frames of Wave Sequence in MorphBench

9 from TC-Bench, and 4 from our own dataset. 15 volunteers participated, yielding a total of 990 votes. Participants compared pairs of videos and voted on which appeared more natural and smooth across three scenarios: Ours vs. DiffMorpher, Ours vs. SEINE, and Ours vs. DynamiCrafter. The results, summarized in Table 3, indicate a strong preference for our algorithm, highlighting its effectiveness in generating smoother, more natural transitions.

Table 3: Human Preference on transition video quality.

Metrics Ours >DiffMorpher Ours >SEINE Ours >DynamiCrafter Perference 71.82% 75.45% 75.45%

### Ablation Study

To validate the effectiveness of our proposed module, we conducted ablation studies on the MorphBench dataset. The results, presented in Table 4 and Figure 8, highlight the impact of each component on overall performance. Specifically, Method A represents the baseline model, DynamiCrafter; Method B incorporates interpolation-based conditional controls; Method C integrates GPR; and Method D corresponds to our final model. As shown in Table 4, our approach outperforms others across key quantitative metrics. Although these metrics may not always directly correlate with final visual quality, they strongly suggest that our module significantly enhances the baseline model’s performance, especially in a training-free setting.

The qualitative results in Figure 8 further illustrate the effectiveness of our approach. In some videos, the baseline model fails to generate appropriate transitions, leading to abrupt changes or blurred content. Incorporating

interpolation-based conditional controls improves transition content, though some issues with blurred frames remain. The integration of GPR substantially improves visual quality, although it occasionally introduces artifacts. Finally, with the addition of the FBiF module, most videos exhibit smooth and visually coherent transitions, effectively addressing these challenges.

Chen, X.; Wang, Y.; Zhang, L.; Zhuang, S.; Ma, X.; Yu, J.; Wang, Y.; Lin, D.; Qiao, Y.; and Liu, Z. 2024. SEINE: Shortto-Long Video Diffusion Model for Generative Transition and Prediction. In ICLR. OpenReview.net.

Chen, X.; Xu, C.; Yang, X.; Song, L.; and Tao, D. 2018. Gated-gan: Adversarial gated networks for multi-collection style transfer. IEEE Trans. Image Process., 28(2): 546–560. Coyle, J. R.; and Thorson, E. 2001. The effects of progressive levels of interactivity and vividness in web marketing sites. Journal of advertising, 30(3): 65–77.

Table 4: Ablation studies of our method.

Metamorphosis Animation Overall FID ↓ PPL ↓ FID ↓ PPL ↓ FID ↓ PPL ↓

Method

Feng, W.; Li, J.; Saxon, M.; Fu, T.; Chen, W.; and Wang, W. Y. 2024. TC-Bench: Benchmarking Temporal Compositionality in Text-to-Video and Image-to-Video Generation. arXiv preprint arXiv:2406.08656.

- A 87.32 42.09 43.31 11.16 69.13 33.84
- B 89.50 35.68 35.84 11.27 70.30 29.17
- C 90.51 36.25 42.05 12.45 72.13 29.90
- D 86.92 35.18 42.99 12.46 64.05 29.08

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In NeurIPS, 6626–6637.

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

- A
- B
- C
- D

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Diffusion Probabilistic Models. arXiv preprint arxiv:2006.11239.

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

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

Jang, Y. K.; Huynh, D.; Shah, A.; Chen, W.-K.; and Lim, S.N. 2024. Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval. arXiv preprint arXiv:2405.00571.

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

- A
- B
- C
- D

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Karras, T.; Laine, S.; Aittala, M.; Hellsten, J.; Lehtinen, J.; and Aila, T. 2020. Analyzing and Improving the Image Quality of StyleGAN. In CVPR, 8107–8116.

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Kuaishou. 2024. KLing AI. Accessed: 2024-08-10, Retrieved from https://kling.kuaishou.com.

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Ling, P.; Bu, J.; Zhang, P.; Dong, X.; Zang, Y.; Wu, T.; Chen, H.; Wang, J.; and Jin, Y. 2024. MotionClone: Training-Free Motion Cloning for Controllable Video Generation. arXiv preprint arXiv:2406.05338.

Figure 8: Visualization of Ablation Study.

### Conclusion

Liu, Y.; Zhang, K.; Li, Y.; Yan, Z.; Gao, C.; Chen, R.; Yuan, Z.; Huang, Y.; Sun, H.; Gao, J.; et al. 2024. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177.

We introduced a novel training-free transition video generation approach that combines interpolation-based conditional controls, GPR, and FBiF. Our method consistently outperforms existing models in both quantitative metrics and visual quality, especially in dynamic scenarios. Ablation studies confirm that each component enhances the overall performance, leading to smoother and more coherent transitions. This approach shows significant potential for improving video generation without additional training, particularly in high-dynamic transitions. In the future, we plan to explore the use of entire video segments, rather than just two frames, to achieve even more seamless and effective transitions.

LUMAAI. 2024. LUMA AI. Accessed: 2024-08-10, Retrieved from https://lumalabs.ai/dream-machine.

Pan, X.; Zhan, X.; Dai, B.; Lin, D.; Loy, C. C.; and Luo, P. 2022. Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation. IEEE Trans. Pattern Anal. Mach. Intell., 44(11): 7474–7489.

Pan, Z.; Cai, J.; and Zhuang, B. 2022. Fast vision transformers with hilo attention. NeurIPS, 35: 14541–14554.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In ICML.

### References

An, J.; Zhang, S.; Yang, H.; Gupta, S.; Huang, J.-B.; Luo, J.; and Yin, X. 2023. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477.

Roggeveen, A. L.; Grewal, D.; Townsend, C.; and Krishnan, R. 2015. The impact of dynamic presentation format on consumer preferences for hedonic products and services. Journal of Marketing, 79(6): 34–49.

ByteDance. 2024. Jimeng AI. Accessed: 2024-08-10, Retrieved from https://jimeng.jianying.com.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In CVPR, 10684–10695.

Sauer, A.; Schwarz, K.; and Geiger, A. 2022. StyleGANXL: Scaling StyleGAN to Large Diverse Datasets. In SIGGRAPH, 49:1–49:10. ACM.

Sohl-Dickstein, J.; Weiss, E.; Maheswaranathan, N.; and Ganguli, S. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2256–2265. PMLR.

Song, J.; Meng, C.; and Ermon, S. 2021. Denoising Diffusion Implicit Models. In ICLR. OpenReview.net.

Steuer, J.; Biocca, F.; Levy, M. R.; et al. 1995. Defining virtual reality: Dimensions determining telepresence. Communication in the age of virtual reality, 33(37-39): 1.

Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. NeurIPS, 30.

Wang, C. J.; and Golland, P. 2023. Interpolating between Images with Diffusion Models. arXiv preprint arXiv:2307.12560.

- Wang, W.; Chen, Y.; Liu, Y.; Yuan, Q.; Yang, S.; and Zhang, Y. 2024. MVOC: a training-free multiple video object composition method with diffusion models. arXiv preprint arXiv:2406.15829. Williams, C. K.; and Rasmussen, C. E. 2006. Gaussian processes for machine learning, volume 2. MIT press Cambridge, MA. Wolberg, G. 1998. Image morphing: a survey. Vis. Comput., 14(8/9): 360–372. Xing, J.; Xia, M.; Zhang, Y.; Chen, H.; Yu, W.; Liu, H.;
- Wang, X.; Wong, T.-T.; and Shan, Y. 2023. DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors. arXiv preprint arXiv:2310.12190. Yang, F.; Yang, S.; Butt, M. A.; van de Weijer, J.; et al. 2024. Dynamic prompt learning: Addressing cross-attention leakage for text-based image editing. NeurIPS, 36. Yuan, Z.; Chen, R.; Li, Z.; Jia, H.; He, L.; Wang, C.; and Sun, L. 2024. Mora: Enabling generalist video generation via a multi-agent framework. arXiv preprint arXiv:2403.13248. Zhang, K.; Zhou, Y.; Xu, X.; Dai, B.; and Pan, X. 2024a. DiffMorpher: Unleashing the Capability of Diffusion Models for Image Morphing. In CVPR, 7912–7921. Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang,

- O. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR, 586–595. Zhang, Y.; Wei, Y.; Jiang, D.; Zhang, X.; Zuo, W.; and Tian, Q. 2023. Controlvideo: Training-free controllable text-tovideo generation. arXiv preprint arXiv:2305.13077. Zhang, Y.; Wei, Y.; Lin, X.; Hui, Z.; Ren, P.; Xie, X.; Ji, X.; and Zuo, W. 2024b. Videoelevator: Elevating video generation quality with versatile text-to-image diffusion models. arXiv preprint arXiv:2403.05438. Zhao, M.; Zhu, H.; Xiang, C.; Zheng, K.; Li, C.; and Zhu, J. 2024. Identifying and Solving Conditional Image Leakage in Image-to-Video Diffusion Model. arXiv preprint arXiv:2406.15735.

Zhou, D.; Wang, W.; Yan, H.; Lv, W.; Zhu, Y.; and Feng, J. 2022. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018.

Zhou, Y.; Zhou, D.; Cheng, M.-M.; Feng, J.; and Hou, Q. 2024. StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation. arXiv preprint arXiv:2405.01434.

### Appendix A: Implementation Details

In all of our experiments, we utilized the publicly available DynamiCrafter (Xing et al. 2023) (interpolation variation@320×512 resolution). The experiments were conducted without any model training, leveraging a PyTorchbased implementation on an Ubuntu 22.04 system. Video generation of 16 frames at a resolution of 320×512 using an A800 GPU required approximately 55 seconds, with a memory usage of about 15GB, demonstrating the system’s relatively low resource consumption.

We employed the MorphBench (Zhang et al. 2024a) and TC-Bench-I2V (Feng et al. 2024) datasets datasets for our evaluations, which . Due to the high cost associated with the TC-Bench-I2V dataset, which necessitates invoking the GPT-4 API, our evaluation was limited to comparative experiments, and no ablation studies were performed. However, the prompts used were sourced from the official dataset, ensuring fairness in the comparisons.

For the MorphBench dataset, we conducted a comprehensive evaluation, including ablation studies. The prompts were primarily simple descriptions of image content, with detailed prompts provided in the supplementary material. Due to the interdependencies between our modules, the FBiF module was not ablated individually, as the absence of GPR for unified mapping of the feature space would result in feature summation that could severely disrupt the generative model, rendering video content generation infeasible.

### Appendix B: Limitations

Our model is a training-free approach, which allows us to enhance the transition video generation performance of pretrained models while also being subject to the inherent limitations of the base model’s generation capabilities. When employing a higher number of DDIM steps, such as 50 steps, existing image-to-video generation models are prone to a phenomenon known as conditional image leakage (Zhao et al. 2024). This issue leads to a reduction in the dynamic content of the generated videos as the number of steps increases. Consequently, our method may occasionally struggle to produce a proper transition content under these conditions. However, our success rate still exceeds that of the original model.

When using fewer steps, such as 10 steps, our method consistently outperforms the original model in terms of transition effects, as demonstrated in our experiments. Although there is a slight reduction in per-frame quality compared to the 50-step scenario, this difference does not significantly affect the user experience. As shown in Figure 9, a comparison between some frames generated with 10 steps and those

with 50 steps reveals that this quality difference is negligible in terms of perceptual impact on the user.

### Appendix C: User Study

Considering that quantitative metrics are insufficient to evaluate the appeal of transitions to users, which is more critical in real-world scenarios, we conducted a survey involving 15 volunteers. The survey included 22 randomly selected videos from our dataset, evaluated through three rounds of pairwise comparisons between our method and SEINE (Chen et al. 2024), DiffMorpher (Zhang et al. 2024a), and DynamiCrafter (Xing et al. 2023). Volunteers assessed the quality of the transition videos, with examples of the questionnaire content shown in Figure 10. The results, illustrating the preference for our method in each sample, are shown in Figure 11, Figure 12and Figure 13,.

The results of the user study, combined with the findings in the main text, demonstrate that our method outperforms the other three algorithms in most cases. Among the compared methods, DiffMorpher exhibited slightly higher quality relative to SEINE and DynamiCrafter, aligning with the TC-Bench experimental results discussed earlier in the paper. We also provide examples from a subset of user evaluations, including cases with high user preference, cases with moderate disagreement (i.e., preference scores between 0.4 and 0.6), and cases with relatively lower quality samples, as illustrated in Figures 14, 15, and 16. It is evident that our approach maintains effective video transitions regardless of user preference, demonstrating consistent dynamics. In contrast, other methods exhibit unstable transition effects, which are noticeably unacceptable in certain scenarios.

### Appendix D: Qualitative results

To further demonstrate the effectiveness of our algorithm, we compared it with several state-of-the-art commercial products, including LUMA AI (LUMAAI 2024), KLing AI (Kuaishou 2024), and Jimeng AI (ByteDance 2024). Due to the varying video generation lengths of these commercial models, direct quantitative comparisons are challenging and may not be fair. Therefore, we conducted a visual comparison analysis instead. Figure 17, Figure 18,Figure 19 and Figure 20 present the results of our method compared with these commercial solutions. It is worth noting that LUMA AI and KLing AI typically generate longer videos with fixed settings, which cannot be modified. Since existing open-source models struggle to generate long sequences, we selected 16frame segments from the commercial products that involve transition content for comparison with our method. For more comprehensive insights, the actual videos can be found in the supplementary multimedia materials. A comparative analysis of these figures reveals that our method consistently produces videos with smooth transitions, whereas the commercial products occasionally fail under varying data conditions. However, commercial products tend to generate richer details, likely because of their underlying robust models. We believe that integrating our approach into these commercial models could potentially enhance their transition generation success rate, resulting in even more seamless and refined

video transitions.

Following the quantitative analysis of the TC-BenchI2V and MorphBench datasets presented in the main text, we now provide additional visualization results generated by our model on these datasets. Figure 21 and Figure 22 presents some of these generated examples. Some videos can be viewed in the supplementary multimedia materials. Due to file size limitations for attachments, only a subset of the samples has been included.

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

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

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

DynamiCrafter 10 Steps

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

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

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

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

DynamiCrafter 50 Steps

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

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

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

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

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

Ours 10 Steps

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

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

[Figure 483]

[Figure 484]

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

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

Ours 50 Steps

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

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

[Figure 519]

[Figure 520]

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

[Figure 531]

[Figure 532]

[Figure 533]

Figure 9: Visualization of Different Steps.

[Figure 534]

###### Figure 10: Example of the Questionnaire Content

###### Proportion of Users Preferring Our over DiffMorpher

1.0

0.8

ProportionofVotesforOur

0.6

0.4

0.2

0.0

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21

Sample Index

###### Figure 11: Ours vs. DiffMorpher

###### Proportion of Users Preferring Our over DynamiCrafter

1.0

0.8

ProportionofVotesforOur

0.6

0.4

0.2

0.0

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21

Sample Index

###### Figure 12: Ours vs. DynamiCrafter

###### Proportion of Users Preferring Our over SEINE

1.0

0.8

ProportionofVotesforOur

0.6

0.4

0.2

0.0

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21

Sample Index

Figure 13: Ours vs. SEINE

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

Sample 4

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

Sample 8

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

Sample 9

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

Sample 14

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

Sample 15

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

- Figure 14: Comparison samples from our user study. The top row of each group shows results from our algorithm, while the bottom row shows results from DiffMorpher.

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

Sample 13

Sample 2

Sample 21

Sample 17

Sample 6

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

- Figure 15: Comparison samples from our user study. The top row of each group shows results from our algorithm, while the bottom row shows results from DynamiCrafter.

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

Sample 0

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

Sample 3

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

Sample 12

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

Sample 17

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

Sample 19

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

- Figure 16: Comparison samples from our user study. The top row of each group shows results from our algorithm, while the bottom row shows results from SEINE.
- Figure 17: Generated Samples from Our Method

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

###### Figure 18: Generated Samples from LUMA AI

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

[Figure 1400]

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1432]

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

###### Figure 19: Generated Samples from KLing AI

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

[Figure 1551]

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1593]

[Figure 1594]

[Figure 1595]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

[Figure 1602]

[Figure 1603]

[Figure 1604]

[Figure 1605]

[Figure 1606]

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

[Figure 1611]

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

[Figure 1618]

[Figure 1619]

[Figure 1620]

[Figure 1621]

[Figure 1622]

[Figure 1623]

[Figure 1624]

[Figure 1625]

[Figure 1626]

[Figure 1627]

[Figure 1628]

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

[Figure 1633]

[Figure 1634]

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

[Figure 1643]

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

[Figure 1650]

[Figure 1651]

[Figure 1652]

[Figure 1653]

- Figure 20: Generated Samples from Jimeng AI
- Figure 21: Our Generated Samples from MorphBench Dataset

[Figure 1654]

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

[Figure 1661]

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

[Figure 1668]

[Figure 1669]

[Figure 1670]

[Figure 1671]

[Figure 1672]

[Figure 1673]

[Figure 1674]

[Figure 1675]

[Figure 1676]

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

[Figure 1691]

[Figure 1692]

[Figure 1693]

[Figure 1694]

[Figure 1695]

[Figure 1696]

[Figure 1697]

[Figure 1698]

[Figure 1699]

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

[Figure 1705]

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

[Figure 1714]

[Figure 1715]

[Figure 1716]

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1733]

[Figure 1734]

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

[Figure 1741]

[Figure 1742]

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

[Figure 1752]

[Figure 1753]

[Figure 1754]

[Figure 1755]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

[Figure 1771]

[Figure 1772]

[Figure 1773]

[Figure 1774]

[Figure 1775]

[Figure 1776]

[Figure 1777]

[Figure 1778]

[Figure 1779]

[Figure 1780]

[Figure 1781]

[Figure 1782]

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

[Figure 1791]

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

[Figure 1815]

[Figure 1816]

[Figure 1817]

[Figure 1818]

[Figure 1819]

[Figure 1820]

[Figure 1821]

[Figure 1822]

[Figure 1823]

[Figure 1824]

[Figure 1825]

[Figure 1826]

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

[Figure 1834]

[Figure 1835]

[Figure 1836]

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

[Figure 1846]

[Figure 1847]

[Figure 1848]

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

[Figure 1862]

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

[Figure 1871]

[Figure 1872]

[Figure 1873]

[Figure 1874]

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

[Figure 1887]

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1894]

[Figure 1895]

[Figure 1896]

[Figure 1897]

[Figure 1898]

[Figure 1899]

[Figure 1900]

[Figure 1901]

[Figure 1902]

[Figure 1903]

[Figure 1904]

[Figure 1905]

[Figure 1906]

[Figure 1907]

[Figure 1908]

[Figure 1909]

[Figure 1910]

[Figure 1911]

###### Figure 22: Our Generated Samples from TC-Bench-I2V Dataset

