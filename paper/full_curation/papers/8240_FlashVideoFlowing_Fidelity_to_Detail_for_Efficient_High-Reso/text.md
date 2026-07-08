# arXiv:2502.05179v4[cs.CV]30Jan2026

[Figure 1]

## Flowing Fidelity to Detail for Efficient High-Resolution Video Generation

Shilong Zhang1∗ Wenbo Li2∗ Shoufa Chen1 Chongjian GE1 Peize Sun1 Yifu Zhang3 Yi Jiang3† Zehuan Yuan3 Bingyue Peng3 Ping Luo1

1The University of Hong Kong 2The Chinese University of Hong Kong 3ByteDance Code & Model: https://github.com/FoundationVision/FlashVideo

### Abstract

DiT models have achieved great success in text-to-video generation, leveraging their scalability in model capacity and data scale. High content and motion fidelity aligned with text prompts, however, often require large model parameters and a substantial number of function evaluations (NFEs). Realistic and visually appealing details are typically reflected in high-resolution outputs, further amplifying computational demands-especially for single-stage DiT models. To address these challenges, we propose a novel two-stage framework, FlashVideo, which strategically allocates model capacity and NFEs across stages to balance generation fidelity and quality. In the first stage, prompt fidelity is prioritized through a low-resolution generation process utilizing large parameters and sufficient NFEs to enhance computational efficiency. The second stage achieves a nearly straight ODE trajectory between low and high resolutions via flow matching, effectively generating fine details and fixing artifacts with minimal NFEs. To ensure a seamless connection between the two independently trained stages during inference, we carefully design degradation strategies during the second-stage training. Quantitative and visual results demonstrate that FlashVideo achieves state-of-the-art high-resolution video generation with superior computational efficiency. Additionally, the two-stage design enables users to preview the initial output and accordingly adjust the prompt before committing to full-resolution generation, thereby significantly reducing computational costs and wait times as well as enhancing commercial viability.

### 1 Introduction

In recent years, text-to-video (T2V) generation has achieved remarkable progress, driven by advances in diffusion probabilistic modeling [Sohl-Dickstein et al. 2015; Ho et al. 2020; Liu et al. 2022; Lipman et al. 2022], cutting-edge architectures [Ronneberger et al. 2015; Peebles & Xie 2022], and the integration of extensive model parameters and large-scale datasets [He et al. 2022; Hong et al. 2022; Chen et al. 2023, 2024; Kondratyuk et al. 2024; Zheng et al. 2024b; Yang et al. 2024b; OpenAI 2024]. Among these, DiT-based models [Peebles & Xie 2022] stand out for their excellent scalability in accommodating larger model capacities and datasets.

In video DiTs, the key operator is the 3D full attention mechanism across time (T), height (H), and width (W), which effectively models visual relations in scenarios with large object motions and 3D consistency. The computational complexity scales as O(T2H2W2 · C · N), where C represents the feature dimension (linked to model size) and N is the number of denoising steps (function evaluation). State-of-the-art methods [team @ Meta 2024; Kong et al. 2024; Yang et al. 2024b] typically require

∗:Equal Contribution, †: project leader

[Figure 2]

- Figure 1: Comparison between FlashVideo and other text-to-video generation paradigms. (a) Single Stage DiT suffers from an explosive increase in computation cost when generating at large resolutions, rising from 30s to 2150s (circle in (d)) when increasing the resolution from 270p to 1080p. (b) Though the vanilla cascade can reduce the model size in the high resolution, its second stage still samples from Gaussian noise and only uses the first-stage results as a condition. This approach cannot effectively reduce the number of function evaluations at high resolution and still costs 571.5s (square in (d)) to generate a 1080p video. (c) In contrast, FlashVideo not only decreases the model size in the second stage but also starts sampling from the first-stage results, requiring only 4 function evaluations at high resolution while integrating a wealth of visually pleasant details, which can generate 1080P video with only 102.3s (triangle in (d)). Details on obtaining these statistics are provided in our Supplementary Materials.

large model capacities (e.g., 12 billion parameters), high-resolution modeling (e.g., 1080p), and up to 50 denoising steps, for high-quality outputs.

These requirements arise from the need to tackle key challenges in video generation, particularly ensuring high prompt fidelity and visual quality. First, achieving fidelity in both content and motion demands the model to encode extensive world knowledge. Research has shown significant improvements when increasing model parameters (C) from 2 billion to 12 billion [Yang et al. 2024b; Kong et al. 2024]. Additionally, an adequate number of denoising steps (N) [team @ Meta 2024; Kong et al. 2024; Yang et al. 2024b] is essential for generating high-quality videos. While some efforts to reduce the number of steps have shown promising progress [Ding et al. 2024], they are limited to lower resolutions and simpler motions. Moreover, visual quality has been proven to be tightly tied to resolution in text-to-image generation (H × W) [Blattmann et al. 2023b; Chen et al. 2025; Ren et al. 2024], and for T2V tasks, the integrity of motion (T) must also be maintained. However, the combination of these challenges—large parameters, sufficient denoising steps, and high resolution—significantly increases the computational cost. For instance, a 5-billion-parameter model takes 2150s to generate 1080p videos, up from just 30s at the 270p resolution (Figure 1 (d)).

To overcome these challenges, we introduce FlashVideo, a two-stage framework designed to separately optimize prompt fidelity and visual quality, as illustrated in Figure 1 (c). In the first stage, we focus on generating video content and motion that closely aligns with the user prompt. By operating at a lower resolution (e.g., 270p), even though we utilize a large model with 5 billion parameters with 50 evaluation steps, the model still remains efficient, requiring only 30 seconds function evaluation times (as shown in Figure 1 (d)). And as demonstrated in our experiments (Section. 4.4), this approach preserves semantic fidelity and motion smoothness. In the second stage, we enhance the generated video at 1080p, focusing on fine-grained detail enhancement while minimizing computational overhead. This is achieved using a lighter 2-billion-parameter model and an efficient flow-matching process

ii

with fewer evaluation steps. The two-stage framework effectively balances computational efficiency with high-quality results.

While previous two-stage frameworks [Zhou et al. 2024; Wang et al. 2023b; He et al. 2024] treat the first-stage low-resolution output as a condition and begin the second stage from Gaussian noise (Figure 1 (c)), this design requires 30–50 evaluation steps and still incurs significant computational cost (e.g., 571 seconds for 1080p generation). In contrast, FlashVideo uses flow matching to directly traverse ODE trajectories from first stage low-quality video to the final high-quality videos, eliminating the need to start from Gaussian noise. The flow matching target also tries to constrain the ODE trajectories to be straight. This design efficiently reduces the number of function evaluations to just 4 steps. As a result, FlashVideo reduces the function evaluation time for 1080p videos to just 102s, nearly 1/20 of the time required by a single-stage model (Figure 1 (a)), and 5 times faster than vanilla cascade frameworks (Figure 1 (b)).

Despite this speedup, crafting an appropriate degradation simulation remains essential: naive pixel-space operations (e.g., resizing and blurring [Chan et al. 2022; Yang et al. 2024a]) fail to reproduce the fine-scale artifacts introduced by model synthesis. To this end, we inject noise in the latent representation to eliminate all small structures in real videos, forcing their faithful reconstruction in the second stage. Moreover, we demonstrate that—within 3D full-attention architectures, which are critical for spatio-temporal coherence—the degradation intensity must be carefully tuned to prevent the model from “cheating” by merely copying details across frames, a pitfall neglected in prior work [Pernias et al. 2023; Yu et al. 2024; Wang et al. 2023b; He et al. 2024].

In summary, our contributions are:

- • We propose FlashVideo, a method that decouples video generation into two objectives: prompt fidelity and visual quality. By tailoring model sizes, resolutions, and optimization strategies in two stages, our approach achieves superior effectiveness and efficiency compared to existing methods.
- • Innovatively, we construct nearly straight ODE trajectories starting from low-quality videos to high-quality videos through flow matching, which enables ample detail to be integrated into the video within only 4 function evaluations.
- • Our method achieves top-tier performance on VBench-Long (83.29 score) while achieving impressive function evaluation time. The two-stage design allows users to preview initial output before full-resolution generation, curtailing computational costs and wait times.

### 2 Related Work

Video generation models. Recent advancements in text-to-video (T2V) generation have been remarkable [Yan et al. 2021; Hong et al. 2022; Kondratyuk et al. 2024; Ho et al. 2022b; Blattmann

- et al. 2023b,a; OpenAI 2024; Team 2024b; Bao et al. 2024; lumalabs.ai 2024; team @ Meta 2024; Jin et al. 2024]. Key breakthroughs have been driven by the introduction of video diffusion and flow-matching algorithms [Sohl-Dickstein et al. 2015; Ho et al. 2020; Liu et al. 2022; Lipman et al. 2022], alongside scaled text-video datasets and DiT parameters [Peebles & Xie 2023]. Despite impressive generation quality, a major challenge remains the high computational cost, particularly for generating high-resolution videos.

Cascade diffusion models. Numerous attempts have been made to explore cascade architectures in the text-to-image and text-to-video domains [Saharia et al. 2022; Gu et al. 2023; Ho et al. 2022a; Pernias et al. 2023; Zhou et al. 2024; Yu et al. 2024; Wang et al. 2023b; He et al. 2024]. Researchers are motivated by the challenge that generating high-resolution images/videos in a single stage is both difficult and resource-intensive. In a cascade design, generation starts with a low-resolution sample, followed by an upsampling model to enhance visual appeal at higher resolutions. However, most methods perform the second-stage upsampling from pure noise, conditioning it on the low-resolution input, which requires a large number of function evaluations. While [Zheng et al. 2024a; Teng et al. 2023; Zhang et al. 2023b; Xing et al. 2024] have attempted to start from the first-stage distribution, their theories and implementations are complex, resulting in a high number of inference steps. Moreover, [Fischer et al. 2023] proposes a pure super-resolution method for T2I using flow matching, but the limited generative priors in the second-stage model hinder substantial visual improvements. In this paper, we adhere to the principle of retaining only the most effective designs, developing

iii

[Figure 3]

- Figure 2: The overall pipeline of FlashVideo. FlashVideo adopts a cascade paradigm comprised of a 5-billion-parameter DiT at the low resolution (i.e., Stage I) and a 2-billion-parameter DiT at a higher resolution (i.e., Stage II). The 3D RoPE is employed at both stages to model the global and relative spatiotemporal distances efficiently. We construct training data pairs for Stage I by randomly sampling Gaussian noise and low-resolution video latent. For Stage II, we apply both pixel and latent degradation to high-quality videos to obtain low-quality latent values. These are then paired with high-quality latents to serve as training data. During inference, we retain a sufficient NFE = 50 at a low resolution of 270p for Stage I. The generated videos retains high fidelity and seamless motion, albeit with detail loss. These videos are then upscaled to a higher resolution of 1080p and processed by latent degradation. With only 4 steps, our Stage II regenerates accurate structures and rich high-frequency details.

FlashVideo, an efficient yet simple two-stage framework that achieves high-quality, high-resolution video generation with excellent computational efficiency.

Diffusion speeding up. The generation process in diffusion models can be viewed as solving ordinary differential equations. To reduce the number of function evaluations, researchers have developed advanced samplers [Song et al. 2020; Lu et al. 2022; Zhang & Chen 2022]. Additionally, techniques for distilling pre-trained diffusion models into fewer steps have shown success [Salimans & Ho 2022; Meng et al. 2023; Yin et al. 2024; Nguyen & Tran 2024; Berthelot et al. 2023]. Adversarial training has also been employed to create few-step generators [Xu et al. 2024; Sauer et al. 2025; Lin et al. 2024b]. Recently, rectified flow [Liu et al. 2022] with straight ODE trajectories has been introduced, further refined by subsequent works [Liu et al. 2023; Yan et al. 2024], to enable faster sampling in T2I. However, few attempts have been made in the T2V field, where the added time dimension complicates the trajectories and increases computational demands. While some efforts to reduce the number of steps in T2V have shown promise [Ding et al. 2024], they remain limited to low resolutions and simple motion. In this work, we propose an efficient flow matching pipeline that enables high-resolution video generation. Notably, the acceleration techniques discussed above are compatible with our framework, allowing for further speed improvements in both stages.

- 3 Method

#### 3.1 Overview

In the FlashVideo framework, video pixels x ∈ RH×W×T are first compressed into latent features f ∈ Qh×w×t using a 3D causal VAE [Yang et al. 2024b], where h = H/8, w = W/8, and t = (T − 1)/4 + 1. The model is designed to generate 6-second videos (with 8 frames per second, so T = 49) at 1080p resolution. As shown in Figure 2, we then employ a two-stage, low-to-highresolution generation pipeline, where each stage is optimized with tailored model sizes and training strategies to ensure computational efficiency. The following subsections provide a detailed description of each stage.

iv

#### 3.2 Low-Resolution Stage I

In the first stage, the goal is to generate videos with well-aligned content and motion corresponding to the input prompt. To achieve this, we initialize with a large-capacity model, CogVideoX-5B [Yang et al. 2024b], which contains 5 billion parameters. For improved computational efficiency, we perform parameter-efficient fine-tuning (PEFT) to adapt the model to a lower resolution of 270p. We find that adjusting the target resolution of the MMDiT architecture [Esser et al. 2024] is straightforward, which is achieved by applying LoRA [Hu et al. 2021] with rank 128 to all attention [Vaswani 2017], FFN, and adaptive layer normalization [Perez et al. 2018] layers. Compared to full-parameter tuning, PEFT demonstrates greater robustness, especially when fine-tuned with a small batch size of 32. In contrast, full-parameter tuning with such a small batch size significantly degrades generation quality. All other configuration settings, including the denoising scheduler and prediction target, are kept consistent with CogVideoX-5B.

#### 3.3 High-Resolution Stage II

Model architecture. For fine-grained detail enhancement, we employ another model that adheres to the block design specified in CogvideoX-2B [Yang et al. 2024b]. But, we replace the original position frequency embedding with 3D RoPE [Su et al. 2024], as it offers better scalability for higher resolutions during inference (see Figure 8). Unlike the approach in [He et al. 2024], which uses spatial-temporal decomposition and time-slicing attention, we find that utilizing full 3D attention is crucial for maintaining consistency of enhanced visual details in videos with significant motion and scale variance, as shown in Figure 7 and discussed in Section 4.5. As illustrated in Figure 2, the language embedding from the first stage is directly utilized in this stage.

Low-cost resolution transport. Applying the conventional diffusion process at the high-resolution stage—starting from Gaussian noise and conditioned on low-resolution video—demands substantial computational resources. To improve efficiency while maintaining high-quality detail generation, we adopt flow matching [Liu et al. 2022; Lipman et al. 2022] to map the low-resolution latent representation, ZLR, to the high-resolution latent representation, ZHR. Intermediate points are computed through linear interpolation between ZLR and ZHR, as outlined in Algorithm 1. This approach eliminates redundant sampling steps at the initialization phase and avoids reliance on additional control parameters, such as those proposed in [Zhang et al. 2023a; Yu et al. 2024; He et al. 2024]. Furthermore, the t-independent target ZHR − ZLR results in straighter ODE trajectories, enabling few-step generation. During training, ZLR is simulated, as discussed later. In the testing phase, noise-augmented videos generated in the first stage serve as the starting point, and a commonly used Euler solver with S = 4 steps, as outlined in Algorithm 2, is employed. Other higher-order solvers can also be used for practical applications.

Algorithm 1: Training Stage

Algorithm 2: Inference Stage

- 1 Inputs: The video sample XLR generated during the first stage, model Fθ with parameters θ, VAE encoder E and VAE decoder D, step number S
- 2 Procedure:
- 3 ZLR = DEGlatent(E(XLR)))
- 4 ∆t = 1/S
- 5 Z = ZLQ
- 6 t = 0
- 7 for step in [0, 1, · · · , S − 1] do

- 8 ∆z = Fθ (Z, t) ∗ ∆t
- 9 Z = Z + ∆z
- 10 t = t + ∆t
- 11 ZHR = Z
- 12 XHR = D(ZHR)
- 13 Return: High quality video XHR

- 1 Input: High quality video dataset DHR, model Fθ with parameters θ, VAE encoder E
- 2 Procedure:
- 3 Repeat
- 4 XHR ∼ DHR
- 5 ZHR = E(XHR)
- 6 ZLR = DEGlatent(E(DEGpixel(XHR)))
- 7 Target = ZHR − ZLR
- 8 t ∼ Uniform([0, 1])
- 9 Zt = (1 − t) · ZLR + t · ZHR
- 10 Take gradient descent step on
- 11 ∇θ ∥Target − Fθ (Zt, t)∥2
- 12 Until Converged
- 13 Return: Model Fθ

Low quality video simulation. To train the second-stage model, we establish paired low-resolution and high-resolution latent representations, ZLR and ZHR. Starting from a high-quality video XHR, we apply a sequence of blur and resize operations with randomized strengths in the pixel space (details provided in the Supplementary Materials), yielding the low-resolution video. This process, denoted as DEGpixel, is outlined in Algorithm 1. Training on this simulated data enables the model to enhance images with high-frequency details, improving overall clarity, as demonstrated in Figure 3.

v

[Figure 4]

[Figure 5]

[Figure 6]

| | | | |
|---|---|---|---|
|[Figure 7]| | |[Figure 8]|

|[Figure 9]|[Figure 10]|
|---|---|

|[Figure 11]|[Figure 12]|
|---|---|

DegLatent& DegPixel

DegPixel

Input

- Figure 3: Visual showcase of DEGpixel and DEGlatent impact on quality enhancement. From left to right, the first is the input, generated by the first-stage model. The term DEGpixel stands for the improved result yielded from the model trained only with pixel-space degradation, which adds

high-frequency details to the input. Further, DEGpixel & DEGlatent refers to the enhanced result with model trained under both types of degradation, which further improves small structures, such as generating branches for small trees. The improvement is significantly apparent when compared to pixel degradation only.

However, simulating low-resolution data solely through DEGpixel retains strong fidelity between low- and high-resolution videos, which limits the model’s ability to regenerate accurate structures for small objects at high resolutions—especially when artifacts are present in the first-stage output. This limitation often manifests when there are poor structural representations for small objects, such as blurry tree branches in Figure 3 or distorted eye features in Figure 5 (e). To address this issue, we introduce latent degradation, DEGlatent, which perturbs the latent representation with Gaussian noise. This approach allows the model to diverge from the input and generate more reasonable structures for small objects. As shown in Figure 3, compared to DEGpixel, the combination of DEGlatent enables the model to produce sharper and more detailed tree branches and tiny background objects, significantly enhancing visual quality.

The overall simulation process during training can be described as follows: First, pixel-space degradation is applied to the high-quality video, yielding a degraded version. This is then encoded into the latent space, represented as:

Z = E (DEGpixel (XHR)) . (1) Next, the latent representation is blended with Gaussian noise n ∼ N(0,1) to simulate low-quality latents, defined as:

ZLR = DEGlatent(Z) = αstep · Z + βstep · n, where αstep2 + βstep2 = 1. (2) The parameter step determines the strength of noise augmentation. To ensure the model can perceive the noise strength in the latent space, we introduce a noise strength embedding, which is added to the time embedding. At the inference stage, only DEGlatent is applied to the first-stage output. However, setting this degradation strength requires careful design in our full attention framework, as discussed in the subsequent paragraph.

Latent degradation strength search. Although the Stage II full-attention architecture enhances spatio-temporal coherence (see Figure 7), it may exhibit a “cheating” behavior: under weak latent degradation (e.g., noise steps < 400), the model simply copies details across frames, leaving artifacts uncorrected. Conversely, overly aggressive degradation (e.g., noise steps > 800) overwhelms the generator, degrading overall fidelity. Hence, precise tuning of the latent degradation magnitude (DEGlatent) is essential to erase undesirable artifacts while retaining structures produced in Stage I, which is largely unexplored. Traditional super-resolution approaches [Chan et al. 2022; Yang

- et al. 2024a] do not target model-introduced artifacts (Figure 6), while recent diffusion-based video enhancement techniques without global attention [Wang et al. 2023b; Zhou et al. 2024; He et al. 2024] avoid this cheating at the expense of temporal consistency (Figure 7). To identify an optimal

DEGlatent range, we begin training with a broad noise-step interval (600–900), then refine it to 650–750 based on empirical performance (Table 10).

vi

Coarse-to-fine training. Training directly on high resolution requires substantial computational costs. The use of 3D RoPE [Su et al. 2024; Yang et al. 2024b], a relative spatiotemporal encoding, offers good resolution scalability for our model (Section 5.2). As a result, we first conduct large-scale pre-training on low-resolution images and videos (540×960) before extending to the target resolution of 1080p (1080 × 1920). Observing obvious performance fluctuations in the later stages, we further fine-tune the model with a small set of high-quality samples aligned to human preferences. This low-cost additional fine-tuning stage greatly improves the model’s performance.

### 4 Experiments

#### 4.1 Data Collection

We construct a high-quality dataset by first collecting a large corpus of 1080p videos, followed by aesthetic and motion-based filtering, resulting in 2 million high-quality samples. Motion filtering is performed using RAFT [Teed & Deng 2020] to compute the average optical flow, discarding clips with low motion scores (< 1.1). To ensure the second-stage model learns diverse texture details, we further collect 1.5 million high-quality images at a resolution of 2048 × 2048. All videos and images are annotated with detailed captions generated by an internal captioning model. For human preference alignment, we manually curate a subset of 50,000 videos exhibiting high aesthetic quality, rich textures, and significant motion diversity.

#### 4.2 Training Setup

For training the first-stage model, we use only video data, which are resized to the 270p resolution. The model is trained for 50,000 iterations with a batch size of 32 and a base learning rate of 4×10−5. We employ the AdamW optimizer with β1 = 0.9, β2 = 0.95, a weight decay of 1 × 10−4, and gradient clipping set to 0.1.

The second-stage model, which includes both pre-training and human preference alignment, is trained with a batch size of 64, while other hyperparameters remain consistent with those used in the first stage. The pre-training is structured into three phases: (1) training for 25,000 iterations on 540 × 960 image patches cropped from 2048 × 2048 high-resolution images, (2) 30,000 iterations on a mixed dataset of 540 × 960 image patches and videos at a 1:2 ratio, and (3) training on full-resolution 1080 × 1920 videos for 5000 iterations. Finally, we perform (4) fine-tuning on the human preference alignment dataset for 700 iterations. For latent degradation, we initially apply noise within the step range of 600–900 for phases (1), (2), and the first 1000 iterations of (3). Based on the findings in Table 10, we then narrow the noise range to 650–750 for the remaining training in (3) and (4).

#### 4.3 Qualitative Results

In this section, we present visualizations of the two-stage video generation results based on various user prompts. The first-stage output prioritizes high fidelity in both content and motion, while the second stage further refines details and mitigates generation artifacts, thereby enhancing overall visual quality.

Two-stage generation results. As shown in Figure 4, the first-stage outputs (top rows) exhibit strong prompt fidelity with smooth motion. The key visual elements specified in the prompt, highlighted in bold, are accurately generated. However, artifacts and insufficient texture details, marked by the red bounding box, may still be present. In contrast, the second-stage outputs (bottom rows) significantly improve visual quality by refining small objects with plausible structures and enhancing texture richness. Notable improvements include the refined depiction of human faces (a, d), the detailed rendering of animal fur (b, c), the intricate structures of plants (a, b), and the enhanced fabric textures (d), as highlighted in the green bounding box of the second row. Moreover, despite substantial motion, high-frequency details remain temporally consistent, owing to the full attention mechanism integrated into the second stage. More uncompressed cases can be found on our project page.

Artifact correction and detail enhancement in Stage II. To further demonstrate the effectiveness of the second-stage refinement, we provide additional examples of key frames in Figure 5. Compared to the first-stage outputs (marked in red), the second-stage results (marked in green) exhibit significant

vii

|[Figure 13]|
|---|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

|[Figure 19]|
|---|

|[Figure 20]|
|---|

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

|[Figure 26]|
|---|

- (a) beautiful young woman with long, wavy brown hair, she is smiling to the camera ,

wearing a red hat sits elegantly on a wooden bench, the red hat has rich fabric texture wearing black pleated skirt and yellow sweater…

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

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

- (b) A sunny day, a pure, white cat moves through a verdant garden teeming with stately trees

and vibrant flowers. The radiance of the sun's rays dapples through the broad leaves of the trees, casting a patchwork quilt of shadows onto the ….

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

- (c) In the vast expanse of the savannah, a majestic giraffe with richly patterned

spots and gentle eyes extends its incredibly long neck gracefully toward the lush, tender leaves at the crown of a tree…

- (d) A tiger wearing a red bow on the head and black sunglasses keeps nodding, and a

|[Figure 55]|
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

|[Figure 61]|
|---|

|[Figure 62]|
|---|

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

|[Figure 68]|
|---|

smiling old man with white beard, red knitted scarf, and knitted Santa hat. Fireplace with burning flames in the background, rich brick texture…

- Figure 4: Generated videos of FlashVideo. The results in the top and bottom rows are from Stage I and Stage II, respectively. Stage I generates videos with natural motion and high prompt fidelity, as evident from the visual elements (bold in prompts). However, they lack detailed structures for small objects and high-frequency textures (see the red box). In Stage II, details are significantly enriched (see the green box), while content remains highly consistent with the original. Visualization results are compressed. More uncompressed cases can be found on our project page.

improvements by suppressing artifacts and enriching fine details. These enhancements are evident in the more coherent depiction of oil painting-style sunflowers in (a), the refined rendering of wrinkles and hair in (b), the improved texture structures of animals and plants in (c) and (d), and the correction of facial and object artifacts in (e).

#### 4.4 Quantitative Results

We first evaluate our model on the VBench-Long [Huang et al. 2024] benchmark utilizing its long prompt. Subsequently, we assess the visual quality improvements achieved in Stage II by employing several widely used non-reference image and video quality assessment metrics.

VBench-Long benchmark. We follow the standard evaluation protocol of VBench-Long, generating five videos per prompt. Noting that VBench metrics tend to favor higher frame rates, we apply a real-time video frame interpolation method [Huang et al. 2022] to upscale the frame rate from 8 fps to 24 fps. This interpolation incurs negligible post-processing time (within 4 seconds), ensuring

viii

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

[Figure 74]

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

- (a)

- (b)

- (c)

- (d)

- (e)

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

[Figure 81]

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

[Figure 89]

|[Figure 90]|
|---|

|[Figure 91]|
|---|

[Figure 92]

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

[Figure 99]

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

[Figure 108]

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

- Figure 5: Quality improvements in Stage II . We mark regions with artifacts and lacking detail in the first-stage videos using red boxes, while improvements from the second stage are highlighted in green. Zoom in for a better view. Our Stage II significantly elevates visual quality across diverse content—enhancing oil painting–style sunflowers in (a), refining wrinkles and hair in (b), enriching texture structures of animals and plants in (c) and (d), and mitigating facial and object artifacts in (e).

fair comparisons with high-frame-rate methods. A more detailed discussion on VBench’s frame rate preference is provided in the Supplementary Materials.

As shown in Table 1, both our 8fps and 24fps models achieve high semantic scores exceeding 81. However, relying solely on the first-stage model results in aesthetic and imaging quality scores below top-tier methods, with 60.74 and 61.87 for 270p. After applying the second stage, both quality scores improve significantly, reaching state-of-the-art levels of approximately 62.55 and 66.96, respectively, as reported in Table 1. These results validate our approach of initially reducing the resolution in Stage I to ensure high prompt fidelity at a lower computational cost, followed by quality enhancement in Stage II. On the other hand, our entire functional evaluation only takes about 2 minutes, significantly outperforming other methods in terms of efficiency. For example, a concurrent work, Hunyuan Video [Kong et al. 2024], which achieves a total score of 83.24 using a larger 13B single-stage model, requires 1742 seconds for function evaluation to generate 720p (720 × 1280) results. In contrast, our method not only demonstrates superior efficiency but also generates outputs at higher resolution. Furthermore, users can obtain preliminary previews in just 30 seconds for 270p, allowing them to decide whether to proceed with the second stage or refine the input prompt. This flexibility significantly enhances the user experience.

Frame and video quality assessment. As shown in Table 2, we present a comprehensive comparison of visual quality between the two stages with all VBench-Long prompts. We utilize widely recognized image quality assessment metrics, including MUSIQ (↑) [Ke et al. 2021], MANIQA

ix

background consistency

temporal flickering motion smoothness

subject consistency

overall consistency

spatial relationship

appearance style temporal style

dynamic degree aesthetic quality imaging quality

multiple objects human action

Total Score Quality Score Semantic Score

object class

scene

color

Method

HunyuanVideo 83.24 85.09 75.82 97.37 97.76 99.44 98.99 70.83 60.36 67.56 86.10 68.55 94.40 91.60 68.68 53.88 19.80 23.89 26.44 Vchitect(VEnhancer) 82.24 83.54 77.06 96.83 96.66 98.57 98.98 63.89 60.41 65.35 86.61 68.84 97.20 87.04 57.55 56.57 23.73 25.01 27.57 CogVideoX-1.5 82.17 82.78 79.76 96.87 97.35 98.88 98.31 50.93 62.79 65.02 87.47 69.65 97.20 87.55 80.25 52.91 24.89 25.19 27.30 CogVideoX-5B 81.61 82.75 77.04 96.23 96.52 98.66 96.92 70.97 61.98 62.90 85.23 62.11 99.40 82.81 66.35 53.20 24.91 25.38 27.59 CogVideoX-2B 80.91 82.18 75.83 96.78 96.63 98.89 99.02 59.86 60.82 61.68 83.37 62.63 98.00 79.41 69.90 51.14 24.80 24.36 26.66 Mochi-1 80.13 82.64 70.08 96.99 97.28 99.40 99.02 61.85 56.94 60.64 86.51 50.47 94.60 79.73 69.24 36.99 20.33 23.65 25.15 LTX-Video 80.00 82.30 70.79 96.56 97.20 99.34 98.96 54.35 59.81 60.28 83.45 45.43 92.80 81.45 65.43 51.07 21.47 22.62 25.19 OpenSora-1.2 79.76 81.35 73.39 96.75 97.61 99.53 98.50 42.39 56.85 63.34 82.22 51.83 91.20 90.08 68.56 42.44 23.95 24.54 26.85 OpenSoraPlan-V1.1 78.00 80.91 66.38 95.73 96.73 99.03 98.28 47.72 56.85 62.28 76.30 40.35 86.80 89.19 53.11 27.17 22.90 23.87 26.52

FlashVideo8fps 82.80 82.99 82.03 96.91 96.77 98.56 96.84 63.47 62.55 66.96 90.02 81.47 99.00 85.71 83.20 55.34 24.64 25.23 27.65 FlashVideo24fps 83.29 83.72 81.60 97.14 97.07 98.57 98.83 59.86 62.41 66.12 88.45 80.27 99.00 84.14 82.27 56.71 24.60 25.23 27.60

- Table 1: Comparison with state-of-the-art open-source models on VBench-Long benchmark [Huang et al. 2024]. This includes the recent HunyuanVideo [Kong et al. 2024], Vchitect-2.0 incorporated with VEnhancer [He et al. 2024], varying versions of CogVideoX [Yang et al. 2024b], Mochi-1 [Team 2024a], LTX-Video [HaCohen et al. 2024], OpenSora [Zheng et al. 2024b] and OpenSoraPlan [Lin et al. 2024a]. FlashVideo employs a cascade paradigm to deliver top-tier semantic fidelity and quality.

Frame Quality Video Quality

#NFE / Time MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Technical(↑) Aesthetic(↑)

- Stage I (270p) 50 / 30.1s 24.54 0.226 0.334 11.77 7.280 96.15
- Stage II (1080p) 4 / 72.2s 53.46 0.302 0.436 5.380 11.68 97.87

- Table 2: Comparison of frame quality and video quality between two stages with Vbench-Long prompts.The best results are emphasized in bold.

Frame Quality Video Quality Attributes

#NFE / Time MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Tech(↑) Aesth(↑)

First Stage as Preview

Artifacts Correction

Long-Range Detail Consistency

Avoids Detail Flickering

RealbasicVSR 1 / 71.5s 54.26 0.272 0.418 5.281 10.71 99.42 ✓ × NA ✓ Upscale-A-Video 30 / 376.6s 23.67 0.201 0.285 12.02 7.690 97.61 ✓ × NA × VEnhancer 30 / 549.2s 51.69 0.280 0.385 5.330 11.63 98.39 × ✓ × ✓ FlashVideo (Ours) 4 / 72.2s 58.69 0.296 0.439 4.501 11.86 98.92 ✓ ✓ ✓ ✓

- Table 3: Comparison with video enhancement methods. Best results are bold, second-best underlined. We highlight key design attributes for Stage II; only our tailored method meets all criteria. Visual examples are available on the project page. “NA” indicates limited new details, preventing evaluation of long-range consistency.

(↑) Yang et al. [2022], CLIPIQA (↑) [Wang et al. 2023a], and NIQE (↓) [Mittal et al. 2012], along with the video metric DOVER [Wu et al. 2023], to assess the perception of distortions (Technical ↑) and content preference and recommendation (Aesthetic ↑). It is evident that all metrics show significant improvements following the application of Stage II. We argue that increasing the resolution in the second stage (Section 5.2), ultimately producing higher outputs (e.g., 2K), would further enhance visual quality, and this will be explored in future work.

#### 4.5 Comparison with Video Enhancement Methods

To comprehensively evaluate the effectiveness of our tailored Stage II, we compare it against several state-of-the-art video enhancement methods, including VEnhancer [He et al. 2024], Upscale-aVideo [Zhou et al. 2024], and RealBasicVSR [Chan et al. 2022]. Our evaluation comprises both quantitative and qualitative analyses based on the first-stage outputs. Specifically, we construct a curated test set of 100 text prompts with detailed descriptions and generate the corresponding low-resolution 6-second 49-frame videos using Stage I, incorporating diverse visual elements such as characters, animals, fabrics, and landscapes. We refer to this test set as Texture100. The following ablation study is also conducted on this test set.

x

[Figure 115]

- Figure 6: Visual comparison with various video enhancement methods. We present our results alongside enhanced versions, derived from the first-stage outputs, of four video enhancement methods.

As shown in Table 3, we summarize the key attributes for designing Stage II: maintaining high fidelity for a reliable preview, correcting artifacts from Stage I, and ensuring detail consistency across both adjacent frames (avoid flickering) and long-range frames. Notably, only FlashVideosatisfies all these requirements. The frame and video quality metrics are also reported in Table 3, where FlashVideo consistently surpasses competing methods by a substantial margin while maintaining superior efficiency. Notably, although the GAN-based RealBasicVSR achieves competitive scores on some metrics, its outputs frequently exhibit excessive smoothing, indicating a misalignment between these metrics and human perceptual preferences. Consequently, we recommend interpreting quantitative evaluations as supplementary references while prioritizing qualitative assessments. On the other hand, the diffusion-based VEnhancer demonstrates stronger generative capabilities. However, its outputs often undergo significant deviations from the input, contradicting our core design principle of enhancing visual quality while preserving fidelity. Furthermore, VEnhancer employs separate spatial-temporal modules and time slicing instead of 3D full attention, leading to reduced content consistency across extended video sequences—an issue we will explore in subsequent discussions. Additionally, its high NFE results in increased computational overhead, making high-resolution generation time-intensive. In contrast, our model achieves nearly a sevenfold speedup over VEnhancer while producing sharper high-frequency details, as evidenced in Table 3.

- Figure 6 (a) illustrates a case where the woman’s face contains noticeable artifacts, and the background appears blurry. Our method effectively reconstructs intricate facial details while enriching the background with high-frequency textures, maintaining both structural integrity and fidelity. In comparison, although VEnhancer yields a relatively clear face, it also significantly alters the background, losing fidelity entirely. Essential visual elements like “standing water” on the ground and the overall dim tones are completely lost. This result is contrary to our intent of using the first-stage results for preview. Other methods, such as Upscale-a-Video and RealBasicVSR, fail to correct facial artifacts and instead generate excessively smoothed patterns, further reducing realism. A similar trend is observed in Figure 6 (b), where our approach delivers richer textures—such as distinct individual hairs on the cat’s body—while preserving consistency with the original input. As discussed earlier, the full attention mechanism in our model plays a crucial role in maintaining content consistency, outperforming VEnhancer in this regard. Figure 7 presents a sequence of three frames featuring substantial motion, where the camera transitions from a distant to a close-up view, leading to significant scale variations in the subject’s appearance. While both FlashVideo and VEnhancer exhibit clear improvements over the initial input, VEnhancer struggles to preserve facial identity across the key frames and introduces inconsistencies in fine details such as jacket textures and background elements. In contrast, our method effectively mitigates these issues, ensuring stable and coherent visual quality throughout the sequence.

xi

[Figure 116]

- Figure 7: Comparison of long-range detail consistency in large-motion videos. We select a first-stage generated video with significant motion and sample three key frames. The girl in this video undergoes substantial scale variation from distant to close-up views. VEhancer He et al. [2024], with spatial-temporal module and time slicing, fails to preserve identity and detail consistency. In contrast, FlashVideo leverages 3D full attention to maintain consistent facial identity and texture details.

Frame Quality Video Quality Sematics

MUSIQ(↑) CLIPIQA(↑) Technical(↑) Aesthetic(↑) Object Class(↑) Overall Consistency(↑)

Full Fine-Tuning 20.53 0.273 8.531 97.64 85.6 26.1 LoRA 23.93 0.286 8.569 97.87 90.3 27.9

Table 4: Comparison of LoRA and full parameter fine-tuning in Stage I. Best results are in bold.

### 5 Ablation

In this section, we conduct a series of ablation studies to evaluate the key designs of our approach. First, we examine the advantage of LoRA fine-tuning compared to full fine-tuning for adapting Stage I to a new resolution. We then assess the effectiveness of RoPE in Stage II. Next, we detail the low-quality video simulation strategy employed for training the Stage II model. Additionally, we explore the importance of aligning the model’s output with human preferences. Finally, we analyze the influence of various inference hyperparameters on the final performance.

#### 5.1 LoRA v.s. Full Parameter Fine-Tuning in Stage I

In the setup with a batch size of 32, we compare LoRA fine-tuning with full parameter fine-tuning for training the first-stage model at 270p resolution over the same number of iterations. The frame and video quality are evaluated on Texture100, and the semantics-related scores are assessed on VBench-Long, as shown in Table 4. In this configuration, full parameter fine-tuning tends to produce more artifacts, resulting in a degradation of both visual quality and semantic fidelity. In contrast, LoRA fine-tuning preserves the generative capabilities of the original model while efficiently adapting it to a lower resolution. Based on efficiency and performance, we opt for the LoRA strategy.

#### 5.2 Position Embedding in Stage II

To achieve high training efficiency, we first train the Stage II model at low resolution and then apply fine-tuning at higher resolutions, as detailed in Sec.4.2. Additionally, we aim for our model to generate high-quality videos at resolutions that exceed those used during training. To enable effective resolution generalization, we explore the use of representative position embeddings. Specifically, we compare the default absolute position embeddings [Vaswani 2017] from the 2-billion DiT model [Yang

- et al. 2024b] with the rotary position embedding (RoPE) [Su et al. 2024], and find that RoPE offers superior performance in such a video enhancement task.

We train the model using both position embeddings at a 540 × 960 resolution and test it across three settings: 540 × 960, 1080 × 1920, and 1440 × 2560. For the larger resolutions, we employ position embedding extrapolation. As shown in Figure 8, while both position embeddings yield satisfactory

xii

|[Figure 117]|
|---|

| |
|---|

| |
|---|

[Figure 118]

[Figure 119]

(a) Sinusoidal Extrapolation

|[Figure 120]|
|---|

| |
|---|

| |
|---|

[Figure 121]

[Figure 122]

(b) RoPE Extrapolation

- Figure 8: Results of resolution extrapolation using absolute sinusoidal and RoPE position embeddings. Both settings perform well at the training resolution. However, while RoPE preserves detail enhancement at higher resolutions, absolute position embedding introduces noticeable artifacts beyond the training range.

Frame Quality Video Quality

#NFE / Time MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Technical(↑) Aesthetic(↑)

FlashVideo-1080p 4 / 72.2s 58.69 0.296 0.439 4.501 11.86 98.92 FlashVideo-2K 4 / 209.8s 62.40 0.354 0.497 4.463 12.25 99.20

- Table 5: Inference resolution scaling results of FlashVideo with RoPE. Best results are in bold.

Degradation Frame Quality Video Quality

DEGpixel DEGlatent MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Technical(↑) Aesthetic(↑)

23.61 0.200 0.286 12.02 6.43 97.32 ✓ 49.12 0.253 0.364 4.95 7.12 99.02 ✓ ✓ 55.45 0.273 0.409 4.69 9.09 98.96

- Table 6: Comparison of frame quality and video quality when applying different degradations. Best results are in bold.

results at the training resolution, RoPE consistently enhances details when inferring at larger scales. In contrast, absolute position embeddings exhibit clear artifacts beyond the trainining resolution. Based on these findings, we incorporate RoPE for training the second-stage model.

After training the model with RoPE at the 1080p (1080 × 1920) resolution, we further extend the inference resolution to 2K (1440 × 2560) using RoPE-based extrapolation. As shown in Table 5, our model demonstrates improved visual quality at 2K resolution, as observed from the visual comparisons. However, the inference time increases significantly, from 74.4 seconds to 209.8 seconds. We hypothesize that larger resolutions better stimulate the detail-generation capabilities of our model, aligning with the inference scaling law [Snell et al. 2024] observed in large language models.

#### 5.3 Low-Quality Video Simulation in Stage II

As discussed in Section 3.3, we visually demonstrate (see Figure 3) the significance of incorporating latent and pixel degradation for simulating low-quality videos during the training of Stage II. In this section, we provide a more detailed quantitative evaluation. For computational efficiency, we conduct the experiment using 5-frame 1080p video inputs. We train two models for 10,000 iterations: one with only pixel degradation applied, and the other with both pixel and latent degradation. As shown in Table 6, the baseline represents the results from Stage I. When the Stage II model is applied with pixel degradation (DEGpixel), the first-stage output is significantly improved, with high-frequency textures being added and overall visual quality boosted. Furthermore, incorporating latent degradation

xiii

###### Frame Quality Video Quality

MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Technical(↑) Aesthetic(↑)

Before 55.61 0.278 0.427 4.667 11.76 98.90 After 58.69 0.296 0.439 4.501 11.86 98.92

- Table 7: Performance comparison of FlashVideo before and after human preference alignment. Best results are in bold.

Frame Quality Video Quality

NFE MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Tech(↑) Aesth(↑)

- 1 48.60 0.253 0.307 5.148 8.643 98.03
- 2 55.10 0.287 0.390 4.730 10.57 98.38
- 3 57.59 0.290 0.418 4.543 11.39 98.62

- 4 58.69 0.296 0.439 4.501 11.86 98.92

- 5 59.24 0.299 0.441 4.492 12.15 99.05

- 6 59.17 0.295 0.440 4.521 12.48 99.05

- 7 59.48 0.298 0.445 4.578 12.20 99.01
- 8 59.64 0.298 0.451 4.554 12.05 99.16

Table 8: Results of FlashVideo under different numbers of function evaluations (NFEs). The recommended range is highlighted in gray.

Frame Quality Video Quality

CFG MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Tech(↑) Aesth(↑)

1 45.01 0.253 0.359 5.395 10.75 98.98 4 50.92 0.278 0.397 5.102 11.71 99.16 7 54.26 0.287 0.418 4.905 11.97 99.10 10 57.37 0.298 0.441 4.692 12.15 99.12 13 58.69 0.296 0.439 4.501 11.86 98.92 16 58.42 0.285 0.416 4.353 11.54 98.57 19 57.66 0.277 0.397 4.143 11.32 97.84 22 57.48 0.270 0.379 3.982 10.94 97.76

Table 9: Results of FlashVideo under different classifier-free guidance (CFG) scales. The recommended range is highlighted in gray.

Frame Quality Video Quality

Training Noise Step Inf Noise MUSIQ(↑) MANIQA(↑) CLIPIQA(↑) NIQE(↓) Tech(↑) Aesth(↑)

600 53.62 0.269 0.403 4.911 11.85 99.03 650 53.98 0.269 0.399 4.832 11.77 99.06 700 53.82 0.274 0.399 4.763 11.93 99.02 750 54.06 0.279 0.400 4.785 11.96 98.92 800 53.50 0.276 0.403 4.663 11.72 98.91 850 51.39 0.279 0.391 4.787 11.26 98.72

600-900

650 58.49 0.294 0.431 4.583 11.96 98.84 675 58.69 0.296 0.439 4.501 11.86 98.92 700 57.80 0.290 0.418 4.531 12.01 98.78 725 57.97 0.295 0.426 4.462 11.98 98.83 750 57.62 0.294 0.422 4.437 12.10 98.72

650-750

Table 10: Results of FlashVideo under different latent degradation strengths. During initial training, a noise step range of 600–900 is applied, with model performance evaluated across different steps. The range of 650–750 consistently yields satisfactory results (see upper half of Table). This refined range is then adopted for subsequent training, with final performance presented in the lower half of Table.

(DEGlatent) leads to even further enhancement, producing clearer and more realistic structures for small objects and background details.

#### 5.4 Human Preference Alignment in Stage II

In our experiments, training at 1080p resolution reveals instability, characterized by performance fluctuations across different checkpoints (every 500 iterations). We attribute this inconsistency to the varying quality of the training samples. To address this issue, we manually curate a highquality dataset of 50,000 samples, specifically selected based on strong human preference. Our model undergoes a quick fine-tuning process on this refined dataset to stabilize training and improve performance, and then is evaluated on the Texture100 benchmark, as presented in Table 7. Despite the relatively small size of the selected dataset, we observe substantial improvements in both aesthetic quality and the richness of fine details. These results highlight the effectiveness of incorporating human preference into the fine-tuning process.

During the testing phase, users can flexibly adjust several hyperparameters—namely the number of function evaluations (NFEs), classifier-free guidance (CFG), and latent degradation strength (noise strength)—to suit their specific needs. We provide a detailed analysis of how these hyperparameters affect performance in Figure 9, with corresponding quality scores reported in Tables 8, 9, and 10. Unless otherwise specified, the default values for these hyperparameters are set to NFE=4, CFG=13, and NOISE=675.

xiv

[Figure 123]

Figure 9: Results of stage II under different inference hyper-parameters.

#### 5.5 Number of Function Evaluations

As depicted in Figure 9 (a), the processed video exhibits slight haziness and blurriness when NFE=1. Increasing the NFE improves visual quality, with more defined facial details, e.g., teeth and hair, and sharper textures on elements such as leaves and sweaters observed at NFE=4. Beyond NFE=4, increasing the value further (i.e., to NFE=5 or higher) does not result in significant visual enhancement in most cases. The qualitative results on some metrics reported in Table 8 confirm this trend, aligning with the visual observations. We recommend users to adjust the NFE to between 4 and 6 during actual use.

#### 5.6 Classifier-free Guidance

The impact of the CFG scale is illustrated in Figure 9 (b). At CFG=1, the result remains blurry, with insufficient details. As the CFG value increases, the video content becomes clearer and more defined, with finer details such as earrings becoming more distinctly visible. Specifically, CFG values between 10 and 13 yield satisfactory results, striking a balance between sharpness and details. However, further increasing CFG beyond 13 results in excessive sharpness, leading to unnaturally textured visuals. As shown in Table 9, both image and video quality scores improve as CFG increases from 1 to 13, but several metric scores degrade when CFG exceeds 13.

#### 5.7 Latent Degradation Strength

We find that latent degradations below 500 noise steps lead the model to “cheat” and fail to correct first-stage artifacts. To identify an effective strength, we conduct a noise-step search over the 600–900 interval (upper half of Table 10) and observe that the 650–750 window consistently delivers the best artifact removal and visual content preservation. Accordingly, we adopt the 650–750 noise-step range for our final training regime (results shown in the lower half of Table 10). The qualitative visualizations further support this analysis. As shown in Figure 9 (c), at lower degradation levels, the enhanced video retains higher fidelity to the original input. This preservation of fidelity, while beneficial for maintaining overall content integrity, can impede the repair of artifacts and restrict the generation of finer details, such as those seen in fingers, guitar strings, and surface textures. On the other hand, increasing the noise strength promotes the generation of additional visual details. Yet, if the noise is excessive, it can distort structures or introduce blurriness, due to the inherent limitations of Stage II’s generative capacity.

xv

### 6 Discussion and Limitation

#### 6.1 Discussion

In this section, we share some insights from our exploration to help readers gain a clearer understanding of the design principles and positioning of our work, as well as to provide guidance for potential future improvements.

Principles of adjusting latent degradation strength. Selecting an appropriate latent degradation strength is crucial for training the Stage II model. Achieving the balance between minimizing artifacts and preserving the integrity of the original content is key. We recommend adjusting the latent degradation strength based on the Signal-to-Noise Ratio (SNR), meaning that the noise step should be increased when either the resolution or the number of video frames increases. Notably, the number of frames has a greater impact than resolution, as visual content across multiple frames exhibits stronger correlations that are harder to disrupt. For example, in preliminary experiments with 17 video frames, we find that artifacts in the input could be corrected with a noise step of 500, which is significantly lower than the optimal noise range of 650 to 750 observed when the frame count is increased to 49.

Fidelity vs. visual quality improvement. A delicate balance exists between maintaining fidelity and enhancing visual quality. Unlike real-world video enhancement, where input videos purely lack high-frequency details, the first-stage generated video often contains subtle structural flaws or artifacts that require refinement. Traditional super-resolution methods, which focus on maintaining high fidelity, are unable to address these issues effectively. Conversely, regenerating new content by treating the first-stage output as a rough guide also falls short, as it conflicts with our design philosophy. We view the first-stage output as a low-cost preview, and it must align closely with the final result. To achieve this balance, we carefully adjust the strength of both strategies, ensuring that visual quality is enhanced without compromising the integrity of the original content.

Can Stage II be a general video enhancement model? It is noteworthy that the current training setup is specifically tailored for 1080p and is not suitable as a general enhancement method for videos with varying resolutions or frame counts. However, we believe that with further refinement, such as incorporating additional input information regarding resolution and frame number, the model could be adapted to handle a wider range of scenarios. We aim to explore this direction in future work.

Challenges with increased video length. Video enhancement is more challenging than singleimage processing, as it requires ensuring the consistency of newly added details across the entire video sequence. This task calls for a model that not only improves visual quality but also manages the intricate visual relationships and motion across frames. In Stage II, we address these challenges by employing 3D full attention and adjusting the degradation strength. However, as the video length increases, the computational demand of 3D full attention escalates quadratically. Moreover, if the degradation strength is not carefully adjusted, the model may resort to recovering details by directly referencing multiple frames, which can compromise its generative capacity during inference.

Sparse attention in Stage II. We visualize the attention maps in Stage II and observe significant sparsity, particularly in space compared to time. We attribute this phenomenon to the moderate motion intensity in the current first-stage output. To reduce the computational cost of Stage II, we apply FlexAttention [PyTorch 2024] to implement window-based spatial-temporal attention with H = 11,W = 11,T = 7. As a result, the method performs well with significantly improved efficiency when the first-stage output contains low motion. However, we observe inconsistencies and blurred patterns in the regenerated visual details when motion is large. We propose that dynamically adjusting the window size based on motion intensity could be a promising solution in future work.

Resolutions of two stages. Given sufficient computational resources, higher resolutions in both stages could be pursued. Our choice of 270p for the first stage is driven by its ability to produce preliminary results in only 30 seconds, allowing users to quickly assess whether further computation in Stage II is necessary. This provides a clear advantage over contemporary methods.

xvi

#### 6.2 Limitation

Time-Consuming VAE decoding for high-resolution videos. Due to GPU memory constraints, decoding 1080p videos requires spatial and temporal slicing, a process that is time-consuming. Engineering advances in parallel processing and more efficient VAE architectures are essential for enabling faster generation of high-resolution videos.

Long Prompt for inference. The text descriptions adopted during training are typically long and highly detailed. This may increase complexity when users provide prompts in inference. Future research could employ joint training with short prompts or engage language models designed for prompt rewriting [Ji et al. 2024]. This advancement can significantly enhance the user experience.

Challenges with fast motion. Due to constraints in data quantity, quality, and diversity, Stage II may fail when processing videos with extreme and fast motion. Potential solutions include incorporating more training data with large motion and scaling up the model capacity.

### 7 Conclusions

We introduce FlashVideo, a novel two-stage framework that separately optimizes prompt fidelity and visual quality. This decoupling allows for strategic allocation of both model capacity and the number of function evaluations (NFEs) across two resolutions, greatly enhancing computational efficiency. In the first stage, FlashVideo prioritizes fidelity at a low resolution, utilizing large parameters and sufficient NFEs. The second stage performs flow matching between low and high resolutions, efficiently generating fine details with fewer NFEs. Extensive experiments and ablation studies demonstrate the effectiveness of our approach. Moreover, FlashVideo delivers preliminary results at a very low cost, enabling users to decide whether to proceed to the enhancement stage. This decision-making capability can significantly reduce costs for both users and service providers, offering substantial commercial value.

xvii

### References

Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.

David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023a.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22563–22575, 2023b.

Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In IEEE Conference on Computer Vision and Pattern Recognition, 2022.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pp. 74–91. Springer, 2025.

Zihan Ding, Chi Jin, Difan Liu, Haitian Zheng, Krishna Kumar Singh, Qiang Zhang, Yan Kang, Zhe Lin, and Yuchen Liu. Dollar: Few-step video generation via distillation and latent reward optimization. arXiv preprint arXiv:2412.15689, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Johannes S. Fischer, Ming Gui, Pingchuan Ma, Nick Stracke, Stefan A. Baumann, Vincent Tao Hu, and Björn Ommer. Boosting latent diffusion with flow matching, 2023.

Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Joshua M Susskind, and Navdeep Jaitly. Matryoshka diffusion models. In The Twelfth International Conference on Learning Representations, 2023.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667, 2024.

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022a.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022b.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

xviii

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In Proceedings of the European Conference on Computer Vision (ECCV), 2022.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Yatai Ji, Jiacheng Zhang, Jie Wu, Shilong Zhang, Shoufa Chen, Chongjian GE, Peize Sun, Weifeng Chen, Wenqi Shao, Xuefeng Xiao, et al. Prompt-a-video: Prompt your video diffusion model via preference-aligned llm. arXiv preprint arXiv:2412.15156, 2024.

Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5148–5157, 2021.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Joshua V. Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A Ross, Bryan Seybold, and Lu Jiang. VideoPoet: A large language model for zero-shot video generation. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 25105–25124. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/kondratyuk24a.html.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024a.

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024b.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.

lumalabs.ai. dream-machine, 2024. URL https://lumalabs.ai/dream-machine. Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans.

On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14297–14306, 2023.

Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.

xix

Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7807–7816, 2024.

OpenAI. Sora. 2024. URL https://openai.com/index/sora/. William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748,

2022. William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Pablo Pernias, Dominic Rampas, Mats L Richter, Christopher J Pal, and Marc Aubreville. Würstchen: An

efficient architecture for large-scale text-to-image diffusion models. arXiv preprint arXiv:2306.00637, 2023. Team PyTorch. flexattention. March 2024. URL https://pytorch.org/blog/flexattention/. Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu.

Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234–241. Springer, 2015.

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence, 45

(4):4713–4726, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= TIdIXIpzhoI.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103. Springer, 2025.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced

transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024a. Kuaishou AI Team. Kling. 2024b. URL https://kling.kuaishou.com/en. The Movie Gen team @ Meta. Movie gen: A cast of media foundation models, 2024. URL https://ai.meta.

##### com/research/movie-gen/.

Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pp. 402–419. Springer, 2020.

Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023.

xx

A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images.

In AAAI, 2023a.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023b.

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou Hou, Annan Wang, Wenxiu Sun Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In International Conference on Computer Vision (ICCV), 2023.

Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7827–7839, 2024.

Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8196–8206, 2024.

Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. 2024. URL http://arxiv.org/abs/2405.07510.

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers, 2021.

Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1191–1200, 2022.

Xi Yang, Chenhang He, Jianqi Ma, and Lei Zhang. Motion-guided latent diffusion for temporally consistent real-world video super-resolution. 2024a.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024b.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024.

Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 25669–25680, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023a.

Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.

Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023b.

Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogview3: Finer and faster text-to-image generation via relay diffusion. arXiv preprint arXiv:2403.05121, 2024a.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024b. URL https://github.com/hpcaitech/Open-Sora.

Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-A-Video: Temporalconsistent diffusion model for real-world video super-resolution. In CVPR, 2024.

xxi

