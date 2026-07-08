# arXiv:2603.13089v1[cs.CV]13Mar2026

### V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration

###### Shenghe Zheng1⋆, Junpeng Jiang2⋆, and Wenbo Li3†

1 The Hong Kong University of Science and Technology

- 2 Harbin Institute of Technology, Shenzhen
- 3 The Chinese University of Hong Kong

shenghez.zheng@gmail.com, jjunpeng1122@outlook.com, fenglinglwb@gmail.com

- (a) Traditional Image Restoration

- (b) All-in-One Image Restoration

- (c) Ours

Video Generation for Progressive Haze Image Restoration

Drift Correction

[Figure 1]

[Figure 2]

Model for Rain

[Figure 3]

[Figure 4]

[Figure 5]

Model for Haze

[Figure 6]

Hard to Generalize

Rain

[Figure 7]

[Figure 8]

[Figure 9]

Unified Model

[Figure 10]

[Figure 11]

[Figure 12]

Huge Training Data

Low-Light

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Video Generation Model Few-Shot & Generalizable

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Low-Light + Blur

[Figure 21]

Data Efficiency of Our Method

+0.5 dB with 1000x less data

Our Method with Only 1K Multi-task Training Data

- 19
- 20
- 21
- 22
- 23

Snow (Unseen Task)

HigherisBetter

[Figure 22]

PSNR

Data Scaling Law Discoverend for Image Restoration

Lower is Better

Frame 1 Frame 2 Frame 3 …… Frame n

Corrected Frame

0.0E+0 2.0E+5 4.0E+5 6.0E+5 8.0E+5 1.0E+6

Training Data Scale

Fig. 1: Left: Image restoration is formulated as progressive video generation with frame drift correction. Right: Leveraging video generative priors leads to stronger generalization under limited data compared to current image restoration method [25].

Abstract. Large-scale video generative models are trained on vast and diverse visual data, enabling them to internalize rich structural, semantic, and dynamic priors of the visual world. While these models have demonstrated impressive generative capability, their potential as generalpurpose visual learners remains largely untapped. In this work, we introduce V-Bridge, a framework that bridges this latent capacity to versatile few-shot image restoration tasks. We reinterpret image restoration not as a static regression problem, but as a progressive generative process, and leverage video models to simulate the gradual refinement from

⋆ Equal Contribution. † Corresponding Author. Open-source project: V-Bridge.

degraded inputs to high-fidelity outputs. Surprisingly, with only 1,000 multi-task training samples (less than 2% of existing restoration methods), pretrained video models can be induced to perform competitive image restoration, achieving multiple tasks with a single model, rivaling specialized architectures designed explicitly for this purpose. Our findings reveal that video generative models implicitly learn powerful and transferable restoration priors that can be activated with only extremely limited data, challenging the traditional boundary between generative modeling and low-level vision, and opening a new design paradigm for foundation models in visual tasks.

Keywords: Video Generation · Image Restoration · Prior Transfer

#### 1 Introduction

Large-scale video generative models have recently emerged as powerful visual models [14,39,42]. Trained on massive and diverse video corpora, they internalize not only appearance statistics but also structural regularities, object dynamics, lighting variations, and long-range spatio-temporal coherence. Although primarily optimized for video synthesis, the scale, diversity, and structural richness of their training data imply that these models encode far more general visual priors. Such priors extend beyond generation and suggest substantial untapped potential for a wide range of visual understanding and reconstruction tasks.

Among these visual problems, image restoration [3] remains largely confined to task-specific modeling. From denoising to deblurring, dominant approaches rely on carefully engineered architectures trained with substantial supervision for each degradation type [20, 27, 41]. Despite their efficacy, these paradigms remain decoupled from the rapid advances in generative modeling that have redefined high-level vision. Consequently, they necessitate massive supervision, even exceeding a million samples [25], to learn restoration from scratch for each degradation. This data-intensive approach underutilizes the rich, transferable priors already embedded within large-scale generative models.

In this work, we reimagine the conventional methodology by recasting image restoration as a video generation process, simulating progressive restoration dynamics instead of performing static, one-step regression. Specifically, the degraded image is treated as the initial state, while the high-fidelity reconstruction serves as the terminal point along a quality-refinement trajectory. This formulation allows extensive video generation priors to be seamlessly and efficiently integrated into image restoration tasks, potentially alleviating the massive data requirements typical of traditional paradigms.

Driven by this perspective, we introduce V-Bridge (Fig. 2), a framework that harnesses video generative priors for versatile few-shot image restoration, requiring less than 2% of the training data typical of contemporary methods. V-Bridge models image restoration as a step-wise quality evolution toward highfidelity outputs. To bridge the resolution gap between moderate-resolution video pretraining and high-resolution restoration, we propose a coarse-to-fine training

###### Data Construction

Main Model Training Correction Model Training

Optimizing for Visual Details

[Figure 23]

[Figure 24]

[Figure 25]

DataConstruction

Progressive Training Data

[Figure 26]

[Figure 27]

[Figure 28]

andTraining

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Training Data For Stage 1

Training Data For Stage 2 Training Data For Stage 3

Last Frame of Main Model

Corrupted Image

Image Under Restoration

Correction Model

High-Quality Image

Fixed Prompt for All Tasks

[Figure 33]

High-Quality Image

Video Generation Model

Phase 1 Video Generation

Phase 2 Drift Correction

Video Generation Model

Fixed Prompt

Correction Model

[Figure 34]

Inference

Result

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Corrupted

Last-Frame Correction

Image Progressive Image Restoration

- Fig. 2: Overview of the proposed pipeline. The upper part shows data construction and training, where paired low- and high-quality images are used to build pseudo-temporal sequences for progressive restoration learning. A progressive resolution training strategy is adopted to improve fine-grained detail modeling, and an auxiliary generative model is trained for final-frame correction. The lower part shows inference, where the model generates a restoration trajectory and uses the refined last frame as the final output.

curriculum that progressively optimizes the model across increasing scales. This strategy allows the model to first establish global structural coherence before refining high-frequency details, thereby ensuring computational efficiency. Furthermore, we incorporate a drift correction module with minimal overhead to enhance fine-grained texture and color fidelity. Extensive experiments demonstrate that V-Bridge transforms a single video generation model into a versatile restoration expert with only 1,000 multi-task training samples. Our approach achieves a 1.6dB gain over baselines trained on 15× to 1,000× more data. Remarkably, as shown in Fig. 1, V-Bridge generalizes effectively to unseen tasks, showcasing superior out-of-distribution adaptability. Our work opens a new avenue for leveraging rich video priors in low-level vision, paving the way for more general and versatile unified visual models.

Our contributions are threefold:

- – New Restoration Paradigm: We pioneer the use of video generative models as universal priors, demonstrating that their inherent representations serve as a powerful, transferable foundation across diverse low-level tasks.
- – The V-Bridge Framework: We propose V-Bridge, a framework designed for data-efficient image restoration via progressive generative refinement. We introduce a coarse-to-fine training curriculum together with a lightweight drift correction mechanism, enabling sophisticated quality enhancement with minimal task-specific supervision.

- – Empirical Validation: Extensive evaluations reveal that V-Bridge achieves state-of-the-art results with extreme data sparsity (using only 1K samples). Our findings validate the extraordinary out-of-distribution adaptability of video priors and point toward a unified future for low-level modeling.

#### 2 Related Work

##### 2.1 Video Generation

With the success of diffusion models, an increasing number of studies have extended them to video generation. Early approaches typically adopted UNetbased architectures with 2D VAE [4,7,19]. However, these designs struggled to achieve substantial performance breakthroughs in terms of scalability and temporal coherence. Inspired by the strong scalability demonstrated by Sora [5], the field has shifted toward large-scale video generation models built upon 3D VAE and Diffusion Transformers (DiT) [35]. State-of-the-art systems now include a series of open-source models such as OpenSora [47], HunyuanVideo [22], and Wan [39], as well as commercial models including Kling [23], Seedance [14], and Veo [15]. Notably, recent advances from Veo and Seedance demonstrate remarkable capability in producing temporally consistent and semantically realistic video content. These developments suggest that video generative models are evolving beyond content synthesis, showing strong potential as general visual foundation models for unified representation learning across diverse vision tasks.

##### 2.2 All-in-One Image Restoration

Traditional image restoration methods are typically designed for a single predefined degradation type, such as motion blur, rain streaks, or noise, requiring separate models for different corruption scenarios [13,17,21,28]. While effective within narrow settings, these task-specific designs limit scalability and practical deployment. All-in-one image restoration aims to handle diverse degradations within a single unified model. Early efforts focused on degradation-aware representation learning, where AirNet [24] utilizes contrastive learning and methods like PromptIR [36] and ProRes [33] introduce lightweight, learnable visual prompt modules to dynamically adapt the network to specific input conditions. To further refine this process, Perceive-IR [45] leverages multi-level quality-driven prompt for fine-grained quality control across various degradation types and severity levels. Diffusion-based frameworks such as DiffUIR [46] and DiffBIR [29] have been introduced to leverage generative priors for higher perceptual quality. Advanced models like AutoDIR [20] and InstructIR [11] further involve the continuous guidance of image restoration via human language instructions However, these approaches still typically require large-scale training and do not fully exploit large-scale visual priors. In contrast, our work explores restoration from the perspective of leveraging pretrained video generative priors as a general visual prior, enabling progressive quality refinement without designing separate restoration models for each degradation type.

##### 2.3 Chain-of-Frames Reasoning

The rapid maturation of video generation models has catalyzed a paradigm shift from simple motion synthesis to complex visual inference, a phenomenon encapsulated by the Chain-of-Frames (CoF) reasoning [42]. To systematically quantify this emergent intelligence, a diverse array of empirical studies and specialized benchmarks has been established, scrutinizing model performance across dimensions such as spatial relationships, logical reasoning, action planning, and physical dynamics [12,18,26,30,44]. Parallel to these evaluative efforts, recent advancements have focused on augmenting the CoF reasoning capabilities of video models through supervised fine-tuning (SFT) on curated video sequences [43] and test-time prompt optimization [8]. More recently, the versatility of CoF reasoning has been extended to text-to-image synthesis, yielding promising results by treating image generation as the final-state of a reasoning chain [37]. However, while existing literature predominantly focuses on high-level semantic and logical orchestration, the potential of CoF-driven temporal priors to address low-level vision tasks remains a conspicuously unmapped frontier. This leaves a significant research gap: the question of whether the structured visual thinking inherent in CoF can be harnessed to resolve granular pixel-level challenges or restore fine-grained structural integrity remains entirely unexplored.

#### 3 Methodology

##### 3.1 Overview

In this section, we present V-Bridge, a progressive restoration framework that repurposes pretrained video generative priors for image-to-image translation. Unlike vanilla one-step regression, we reformulate restoration as a temporally evolving trajectory that iteratively refines image quality. By harnessing the inherent spatio-temporal consistency and generative priors of video generation models, V-Bridge achieves remarkable performance with only 0.1% to 2% of the task-specific training data required by current methods [20,25].

The methodology is organized as follows: Sec. 3.2 details the construction of pseudo-temporal sequences from paired low-quality (LQ) and high-quality (HQ) images. Sec. 3.3 presents a Progressive Curriculum Training strategy that transitions from structural recovery to fine-grained synthesis. Finally, Sec. 3.4 describes a Drift Correction mechanism designed to bridge the resolution gap between video generative priors and high-resolution restoration. Through this dynamic formulation, we transform static restoration into a learnable flow problem, fully unlocking the few-shot potential of video foundation models.

##### 3.2 Pseudo-Temporal Data Construction

To translate static restoration into a dynamic generation task, we we lift each low-quality–high-quality (LQ-HQ) pair (ILQ,IHQ) into a pseudo-temporal sequence with explicit quality progression.

Given ILQ as the anchor (initial frame) and IHQ as the target (terminal frame), we construct a sequence {It}Tt=0 of length T + 1 such that:

I0 = ILQ, IT = IHQ. (1)

For intermediate frames t ∈ {1,...,T −1}, we define a continuous transition path in pixel space via linear interpolation:

t T

. (2)

It = (1 − αt)ILQ + αtIHQ, αt =

This formulation encapsulates a monotonic quality evolution, providing temporally consistent supervision that guides the model to learn a stable low-to-high quality trajectory. By converting static pairs into a pseudo-video stream, we provide the video model with temporally consistent supervision, enabling it to learn the entire restoration trajectory rather than a singular mapping.

##### 3.3 Progressive Curriculum Training

In this section, we introduce a multi-stage training strategy for efficient and effective restoration. The overall optimization objective remains identical across all stages and follows a supervised fine-tuning paradigm, where the model learns to regress the constructed progressive data sequences. The stage-wise design does not modify the objective itself, but progressively improves the model’s ability to synthesize fine details and enhance fidelity.

Overall Training Objective. Formally, let {It}Tt=0 denote the pseudo-temporal sequence and θ the model parameters. Given the conditional input I0 and the time index t, the model predicts ˆIt = fθ(I0,t). The training objective is:

0,It) ℓ fθ(I0,t),It , (3)

L(θ) = E(I

where ℓ(·,·) denotes a reconstruction loss. This formulation encourages the video model to mimic the image restoration process by progressively approximating the intermediate states, thereby fully unleashing its potential for restoration tasks while learning the complete low-to-high quality trajectory.

Curriculum Training. A key gap between image restoration and video generation lies in training resolution: existing video generative models are rarely trained on high-resolution data (e.g., 4K), which is crucial for fine-detail restoration. However, directly training at such resolutions is computationally expensive and may reduce learning efficiency due to the pre-training–fine-tuning data gap.

To overcome this, we construct a progressive resolution curriculum {rt}Tt=1, which controls the difficulty of restoration learning by modulating spatial fidelity across training stages. Let {vi}Ni=1 denote the training corpus, where each video vi = {fi,1,...,fi,T

i} is a spatio-temporal sample from the video distribution. Instead of training on the original high-resolution data, we apply stage-dependent downsampling. At stage t, video samples are re-encoded using a resolution-aware degradation operator as following:

vi(t) = DownUp(vi,rt), rt ∈ {r1,r2,...,rT}, r1 < r2 < ··· < rT. (4)

In simple terms, the video resolution is gradually increased during training. This enables the model to first capture global restoration at low resolution and then progressively enhance fine-grained detail generation as resolution grows.

From a generative modeling perspective, this progressive learning strategy discretizes a continuous restoration probability flow by learning conditional transition kernels across quality levels. By gradually increasing resolution complexity, the model captures hierarchical semantics and high-frequency perceptual statistics in a coarse-to-fine manner, improving few-shot generalization while preserving perceptual fidelity and temporal consistency.

##### 3.4 Drift Correction

The proposed curriculum training reduces cost and optimization difficulty but cannot fully bridge the large gap between pretraining resolution and the finegrained detail generation capability required in our task. Most video generation models are pretrained at moderate resolutions (e.g., 720p), whereas practical restoration tasks often require recovering ultra high-resolution content (e.g., 4K). This discrepancy limits the model’s ability to faithfully recover high-quality fine details, often leading to struggles in reconstructing high-frequency structures.

We interpret this limitation as an implicit distribution drift induced by the resolution-constrained generative prior. Let xˆ denote the final restored frame predicted by the base video generation model, and let xHR denote the corresponding ground-truth high-resolution image. Due to the pretrained resolution bias, xˆ exhibits a systematic drift from the target high-fidelity manifold, and can be viewed as a sample drawn from a lower-fidelity distribution:

xˆ ∼ pLRθ (x), xHR ∼ pHR(x), (5)

where pLRθ represents the distribution distorted by low-resolution pretraining.

To mitigate this drift, we introduce an additional drift correction model that explicitly learns a short corrective trajectory from xˆ toward xHR. The training samples are constructed as short pseudo-temporal sequences interpolating between the drifted base model output and the ground-truth high-quality image, enabling a smooth transition from resolution-limited restoration to full-fidelity reconstruction. We view this transformation as a form of degradation type tailored to the unique characteristics of video generative models.

The drift correction model is trained to parameterize this conditional generative transition, effectively modeling a mapping:

gϕ : pLRθ (x) → pHR(x), (6)

such that the final output approximates xHR while preserving structural consistency with xˆ. By restricting the trajectory to only a few intermediate frames, the correction process remains computationally efficient, yet substantially eliminates the resolution-induced bias and enhances perceptual quality.

###### Corrupted Image

LQ Patch AutoDIR FoundIR-G FoundIR Ours GT Patch

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Blur

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

Haze

- Fig. 3: Visualization results on a subset of the FoundIR test set. FoundIR-G is the generalist model of FoundIR. GT denotes ground truth. Bounding boxes with different colors indicate zoomed regions for detailed comparison. Compared with other methods, our approach achieves higher visual fidelity and stronger structural consistency, while showing superior robustness across diverse degradation patterns.
- 4 Experiments

##### 4.1 Experimental Settings

Training Details. Our training data are sampled from FoundIR [25] and RealCE [32]. Progressive training sequences are constructed as described in Sec. 3.2 to facilitate gradual restoration learning. Unless otherwise specified, we randomly select 50 samples per task category from each dataset (FoundIR and RealCE) for training, and also use 50 samples per category for the drift correction stage. Both models adopt Wan2.2-TI2V-5B [39] as the backbone network. More implementation and training details are provided in Appendix A.1.

Test Details. We evaluate our method on the FoundIR [25] test split, covering diverse degradations including blur, noise, JPEG compression, haze, rain, raindrop, low-light conditions, and mixed degradations. In addition, we report results on several external benchmark datasets [1,2,6,9,16,40] to further validate crossdataset generalization and real-world applicability. We further evaluate out-ofdistribution (OOD) robustness in Sec. 4.4 to examine model stability under unseen and more severe degradation scenarios. These evaluations are designed to comprehensively verify both restoration quality and generalization capability. More implementation and evaluation details are provided in Appendix A.2.

Evaluation Metrics. We evaluate restoration quality using PSNR and SSIM, two widely adopted metrics in image restoration. PSNR quantifies pixel-wise reconstruction fidelity, while SSIM measures structural consistency between the restored image and the ground truth. Higher values for both metrics indicate better performance. Details are provided in the Appendix A.2.

- Table 1: Quantitative comparison (PSNRSSIM , both higher is better) on the FoundIR test set. FoundIR-G denotes the generalist model of FoundIR and DC denotes Drift Correction. Underlining and boldface denote the second-best and best methods, respectively.

RealESRGAN

Ours w/o DC

TransWeather

Ours Data Scale 15K 16K 19K 77K 30K 53K 40K 15K 23K 1M 1K 1K Blur

DGUNet

PromptIR DiffUIR DA-CLIP X-Restormer InstructIR AutoDIR FoundIR-G

25.20 21.86 21.34 21.91 25.31 20.92 21.88 20.15 20.31 24.34 21.02 24.92 0.7868 0.7334 0.7103 0.7339 0.7979 0.6954 0.7325 0.6801 0.6946 0.7856 0.6862 0.7809 Noise

34.46 36.69 30.12 36.57 34.48 29.45 35.86 38.58 36.84 38.61 32.61 32.87 0.9585 0.9494 0.8945 0.9478 0.9622 0.9027 0.9412 0.9628 0.9221 0.9662 0.9397 0.9498 JPEG

27.62 29.80 23.52 29.63 30.09 25.77 28.58 32.99 32.77 34.03 28.21 26.59 0.9108 0.8906 0.7296 0.8842 0.9390 0.7816 0.8646 0.9378 0.9280 0.9379 0.8487 0.8269 Haze

22.07 18.58 17.72 15.36 19.97 15.91 16.47 16.85 15.23 16.65 21.25 21.70 0.8380 0.6762 0.5635 0.4982 0.8193 0.5399 0.6257 0.7555 0.6264 0.7814 0.6918 0.6711 Rain

28.95 25.47 23.49 27.59 30.17 23.04 25.70 30.18 25.69 33.09 26.40 25.42 0.9226 0.8245 0.6886 0.8411 0.9350 0.6849 0.8076 0.8997 0.7711 0.9387 0.7957 0.7836 Raindrop

28.94 24.83 22.94 25.83 26.98 20.86 26.11 21.05 20.82 28.52 25.77 24.99 0.9115 0.7924 0.6814 0.8327 0.8908 0.6490 0.8368 0.6828 0.6687 0.9110 0.7698 0.7380 Lowlight

19.26 16.16 14.95 16.33 14.02 17.34 16.02 20.04 21.90 12.35 19.18 26.94 0.8709 0.6494 0.6295 0.6353 0.7101 0.7412 0.6404 0.8542 0.8288 0.7185 0.8278 0.8944 B+N

23.48 22.90 22.19 22.93 24.44 22.15 22.90 21.70 21.90 22.53 23.80 27.31 0.7728 0.7633 0.7363 0.7587 0.8039 0.7293 0.7544 0.7185 0.7293 0.7654 0.7703 0.8471 B+J

20.41 22.86 22.14 22.90 21.36 21.36 22.85 21.39 22.03 28.33 24.05 25.33 0.6562 0.7358 0.7105 0.7360 0.6915 0.6940 0.7321 0.6814 0.7088 0.8491 0.7571 0.8020 N+J

29.71 35.28 25.59 35.08 31.96 26.03 33.83 39.90 37.55 39.21 33.33 31.63 0.9566 0.9564 0.8377 0.9524 0.9782 0.8602 0.9405 0.9770 0.9616 0.9745 0.9422 0.9323 R+H

20.40 18.79 18.43 16.61 20.24 13.97 15.17 13.49 14.90 15.42 18.70 18.69 0.7153 0.4981 0.4245 0.4985 0.7268 0.3563 0.4354 0.5535 0.4213 0.6766 0.5574 0.5222

21.79 13.20 15.99 15.36 19.31 12.62 14.75 13.52 14.50 17.05 19.74 20.79 0.8084 0.4182 0.4285 0.4982 0.7745 0.3220 0.4416 0.4983 0.4456 0.7556 0.5983 0.6209 L+R

L+H

32.16 30.66 28.29 31.03 32.35 21.58 30.93 29.87 27.15 32.77 32.90 31.27 0.9116 0.9097 0.8663 0.9039 0.9255 0.7246 0.9026 0.8866 0.8576 0.9281 0.9126 0.9064 L+B

21.95 11.54 21.35 20.29 18.97 16.17 19.38 17.43 19.14 18.07 20.37 23.78

0.7254 0.4922 0.6764 0.6723 0.6939 0.6241 0.6605 0.6676 0.6570 0.7551 0.6543 0.7378 L+N

17.49 12.56 11.56 10.83 13.88 15.70 9.75 16.37 17.49 12.77 17.36 24.60 0.7358 0.5349 0.5181 0.4499 0.6262 0.6365 0.3866 0.4625 0.6900 0.7454 0.6842 0.7765 L+J

22.89 12.17 19.16 21.52 19.98 15.86 18.66 18.06 16.14 17.66 22.72 22.89

0.8704 0.5433 0.6528 0.7671 0.8706 0.6048 0.7205 0.7787 0.6375 0.8489 0.7635 0.7400 L+B+N

21.43 10.60 20.83 22.63 18.89 15.30 22.00 12.78 18.91 17.63 20.41 24.20 0.6612 0.4608 0.6607 0.6924 0.6705 0.6202 0.6841 0.5392 0.6499 0.6889 0.6362 0.7491 L+B+J

19.49 9.57 19.64 12.40 18.64 12.46 13.63 17.39 16.91 18.64 21.16 25.71 0.6582 0.4897 0.6918 0.5597 0.6751 0.5774 0.5738 0.7092 0.6761 0.7899 0.7514 0.8094 L+N+J

26.11 9.64 21.41 22.71 20.66 15.34 16.40 19.13 18.78 20.09 26.09 27.64

0.9496 0.5456 0.8068 0.8616 0.9338 0.7068 0.7810 0.9105 0.8122 0.9162 0.9115 0.9161 B+N+J

21.40 22.43 21.64 22.47 23.72 21.19 22.41 22.42 22.37 26.72 24.69 25.46 0.7197 0.6945 0.6577 0.6942 0.7861 0.6589 0.6894 0.6980 0.6876 0.7310 0.7659 0.7787 Average

24.26 20.27 21.11 22.49 23.27 19.15 21.66 22.18 22.07 23.57 23.68 25.18 0.8173 0.6779 0.6782 0.7209 0.8105 0.6554 0.7075 0.7426 0.7187 0.8199 0.7490 0.7729

##### 4.2 Comparative Experiment

We first evaluate our method on the FoundIR test set, as shown in Tab. 1, comparing with restoration methods including Real-ESRGAN [41], DGUNet [34], TransWeather [38], PromptIR [36], DiffUIR [46], DA-CLIP [31], X-Restormer [10], InstructIR [11], AutoDIR [20], FoundIR [25]. Since we develop an all-in-one image restoration model, we compare with FoundIR-Generalist (FoundIR-G), its all-in-one variant. Trained with only 1K samples from the FoundIR training set, which accounts for merely 0.1% to 7% of the data used by existing approaches, our method matches or surpasses prior methods on the FoundIR test set. Notably, compared with FoundIR models trained on 1M samples, our method even outperforms the all-in-one variant (FoundIR-G) of FoundIR trained using 1M samples on several metrics. The qualitative results

###### Table 2: Quantitative comparisons (PSNRSSIM ) on the public benchmarks. Underlining and boldface indicate the second-best and best methods, respectively.

Methods PromptIR TransWeather DA-CLIP DiffUIR AutoDIR InstructIR X-Restormer AgenticIR FoundIR-G Ours Data Scale ↓ 77K 19K 53K 30K 23K 80K 40K 15K 1M 1K

PSNR ↑ 9.57 10.51 10.94 9.59 12.33 11.03 9.57 10.11 9.29 11.97 SSIM ↑ 0.4333 0.4523 0.459 0.4326 0.4862 0.4649 0.4295 0.3884 0.4307 0.4897

Dense-Haze

PSNR ↑ 11.76 12.3 18.51 11.27 22.52 20.03 11.56 12.82 10.61 17.87 SSIM ↑ 0.6170 0.6557 0.8120 0.5975 0.8572 0.7356 0.6113 0.6649 0.5775 0.7999

UHD-LL

PSNR ↑ 11.38 11.58 12.35 11.39 12.71 12.24 11.36 12.2 11.43 12.97 SSIM ↑ 0.4343 0.4110 0.4662 0.422 0.4774 0.4984 0.4131 0.4495 0.4491 0.4241

NH-Haze

PSNR ↑ 15.16 14.85 15.38 15.2 15.41 13.75 15.16 14.26 15.11 15.56 SSIM ↑ 0.6605 0.5381 0.6035 0.6389 0.5834 0.3240 0.6397 0.5300 0.6411 0.5152

UAV-Rain1k

PSNR ↑ 10.43 12.88 14.78 11.58 11.67 10.92 10.33 16.29 11.57 27.70 SSIM ↑ 0.4485 0.4774 0.5398 0.4827 0.4697 0.3972 0.4376 0.5681 0.5220 0.8747

HQ-NightRain

in Fig. 3 further confirm its superiority. These results demonstrate remarkable data efficiency and clearly validate the effectiveness of introducing pretrained video generative priors, which substantially enhance restoration capability under severely constrained training data.

More importantly, the superiority of our method extends beyond in domain evaluation. Results on out of distribution benchmarks including Dense-Haze [1], UHD-LL [40], NH-Haze [2], UAV-Rain1K [6], and HQ-NightRain [9], as shown in Tab. 2, demonstrate strong cross dataset generalization, where our method consistently achieves clear performance gains over competing approaches. This further confirms that video generative models implicitly capture robust and transferable visual priors, which can be effectively adapted to restoration scenarios.

We also evaluate the effectiveness of our correction module. As reported in Tab. 1, PSNR increases by 1.4 dB and SSIM improves by 0.024. Visual comparisons in Fig. 5(a) also demonstrate enhanced perceptual fidelity. We attribute these gains to the resolution bias of video generative models, where models are typically trained on moderate resolution data such as 720p [39], which limits the ability to recover high frequency textures. By incorporating a dedicated correction model for detail enhancement, our method decouples structural restoration from high frequency recovery, leading to improved reconstruction quality.

We also show qualitative comparisons of the video generative model before and after fine tuning in Fig.4. The off the shelf model struggles to understand the restoration task, which demonstrates the necessity of our few shot training strategy. Our approach effectively unlocks the task specific potential of pretrained generative priors, enabling high fidelity restoration with strong data efficiency.

P S N R S S I M

- 1 0

- 1 5
- 2 0

- 2 5

- 0 .5
- 0 .6
- 0 .7
- 0 .8

###### P e r f o r m a n c e G a in s f r o m F e w - s h o t T r a in in g

W a n 2 . 2 - T I 2 V - 5 B F e w - s h o t T r a in in g

These results provide strong evidence that video generative models can serve as general visual foundation models for low-

Fig. 4: Performance improvement on the test dataset of FoundIR brought by few-shot training.

###### Corrupted Image

LQ Patch Ours w/o DC Ours GT Patch

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Rain + Haze

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

| | |
|---|---|
| | |
| | |
| | |
| | |

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Lowlight + Blur

(a) (b)

- Fig. 5: (a) Comparison on the FoundIR test set with and without the refine model. GT denotes ground truth. Correction improves visual quality and enhances fine details. (b) Ablation study results. Top: effect of different training frame numbers for image restoration. Bottom: performance improves as training data scale increases.

- Table 3: Ablation study about the number of frames (PSNRSSIM , both higher is better).

Isolated Degradation Coupled Degradation

Methods

Average

Blur Noise JPEG Haze Rain Raindrop Lowlight B+N B+J N+J R+H L+H L+R L+B L+N L+J L+B+N L+B+J L+N+J B+N+J 21.25 31.20 26.49 17.20 25.01 25.29 19.55 25.11 24.50 27.07 14.39 18.15 30.76 19.92 15.22 21.17 20.33 21.11 22.80 24.73 22.16 0.6852 0.9439 0.8258 0.6119 0.7697 0.7524 0.8477 0.8012 0.7551 0.8668 0.4491 0.5811 0.8933 0.6515 0.6256 0.7214 0.6510 0.7385 0.8971 0.7525 0.7242 9 frames

- 5 frames

- 21.42 33.60 27.05 20.32 25.54 25.62 20.62 23.34 23.09 26.39 18.18 19.70 32.64 21.24 16.93 22.72 23.12 22.37 25.52 21.54 23.35 0.7023 0.9404 0.8294 0.6833 0.7615 0.7516 0.8405 0.7550 0.7181 0.8541 0.5330 0.5923 0.9069 0.6564 0.6879 0.7594 0.7115 0.7466 0.8916 0.6770 0.7385 17 frames

- 22.05 33.86 26.32 21.32 24.69 24.51 20.08 24.41 24.26 28.36 18.45 19.23 32.20 19.78 16.53 21.62 20.36 23.03 24.29 22.93 23.28 0.7100 0.9353 0.8166 0.6849 0.7439 0.7359 0.8455 0.7790 0.7732 0.8964 0.5338 0.5767 0.8976 0.6302 0.6880 0.7387 0.6337 0.7851 0.8977 0.7111 0.7370 33 frames

- 20.65 34.04 26.12 20.60 24.86 24.52 19.99 23.14 26.27 29.51 18.15 19.18 32.30 18.98 16.86 21.31 19.66 20.81 24.11 25.59 23.04 0.6808 0.9295 0.8006 0.6665 0.7473 0.7228 0.8434 0.7483 0.7937 0.9025 0.5204 0.5640 0.9034 0.6378 0.6896 0.7298 0.6248 0.7571 0.8939 0.7640 0.7294 61 frames

- 21.62 33.60 25.87 20.65 26.25 24.90 19.33 24.49 24.12 31.00 19.29 19.42 31.77 20.25 12.43 19.70 20.01 21.63 23.16 23.04 23.07 0.6889 0.9360 0.7971 0.6457 0.7842 0.7253 0.8024 0.7807 0.7489 0.9164 0.5300 0.5721 0.8996 0.6091 0.5589 0.7064 0.6181 0.7326 0.8800 0.7213 0.7199

level vision tasks. By formulating restoration as a progressive generative refinement process, our approach demonstrates that high-quality restoration capability can be efficiently activated under few-shot supervision. This suggests a scalable paradigm toward universal visual prior learning, enabling effective knowledge transfer across diverse degradation types and low-level vision tasks in practice.

##### 4.3 Ablation Study

Frame Count. We analyze the impact of frame numbers in the video generation process for image restoration. As shown in Tab.3 and Fig.5(b), we evaluate frame numbers ranging from 5 to 61 on the FoundIR test set. Interestingly, performance does not monotonically improve with more frames. Instead, the 9-frame setting consistently yields better results than 33 and 61 frames.

This observation suggests that the generative prior learned by the video model primarily emphasizes global structural coherence rather than fine grained temporal modeling. In this view, image restoration can be formulated as a conditional generative process guided by strong semantic and geometric priors, where missing content is inferred through high level structural reasoning. Therefore,

###### Table 4: Ablation study on Progressive Curriculum Training (PSNRSSIM ).

Isolated Degradation Coupled Degradation

Resolution

Average Blur Noise JPEG Haze Rain Raindrop Lowlight B+N B+J N+J R+H L+H L+R L+B L+N L+J L+B+N L+B+J L+N+J B+N+J

- 21.25 31.20 26.49 17.20 25.01 25.29 19.55 25.11 24.50 27.07 14.39 18.15 30.76 19.92 15.22 21.17 20.33 21.11 22.80 24.73 22.16 0.6852 0.9439 0.8258 0.6119 0.7697 0.7524 0.8477 0.8012 0.7551 0.8668 0.4491 0.5811 0.8933 0.6515 0.6256 0.7214 0.6510 0.7385 0.8971 0.7525 0.7242 720

- 22.33 33.64 27.97 20.64 25.76 25.63 19.34 22.24 23.21 28.48 19.79 19.57 32.06 19.18 17.36 22.63 22.10 21.95 24.83 21.45 23.35 0.7253 0.9440 0.8464 0.6854 0.7792 0.7623 0.8363 0.7275 0.7131 0.8869 0.5594 0.5948 0.9010 0.6411 0.7036 0.7631 0.7030 0.7368 0.9055 0.6823 0.7442 960

512

21.55 34.40 28.10 20.55 26.37 25.47 19.45 23.42 25.87 33.60 18.73 18.01 32.18 18.81 17.18 22.25 19.50 20.48 25.82 22.85 23.41 0.7089 0.9454 0.8477 0.6867 0.7970 0.7700 0.8294 0.7657 0.7937 0.9440 0.5570 0.5808 0.9112 0.6404 0.6782 0.7555 0.6312 0.7402 0.9097 0.7167 0.7474 512+720

21.70 30.71 28.18 21.08 25.76 25.93 19.89 24.29 25.55 28.99 18.61 19.13 32.71 20.83 17.62 22.65 22.45 23.86 25.59 23.11 23.59 0.6999 0.9394 0.8469 0.6842 0.7693 0.7640 0.8256 0.7718 0.7900 0.8876 0.5377 0.5862 0.9103 0.6438 0.6840 0.7687 0.6852 0.7832 0.9101 0.7117 0.7451 720+512

21.38 32.32 26.87 19.43 25.36 25.40 19.95 24.15 25.78 26.40 16.01 18.74 31.96 20.03 17.23 22.03 20.26 22.16 23.85 24.77 22.90 0.6951 0.9416 0.8341 0.6612 0.7683 0.7596 0.8394 0.7730 0.7790 0.8572 0.4952 0.5742 0.9024 0.6430 0.6915 0.7512 0.6198 0.7637 0.9011 0.7541 0.7355 512+720+960

21.02 32.61 28.21 21.25 26.40 25.77 19.18 23.80 24.05 33.33 18.70 19.74 32.90 20.37 17.36 22.72 20.41 21.16 26.09 24.69 23.68 0.6862 0.9397 0.8487 0.6918 0.7957 0.7698 0.8278 0.7703 0.7571 0.9422 0.5574 0.5983 0.9126 0.6543 0.6842 0.7635 0.6362 0.7514 0.9115 0.7659 0.7490

increasing the number of frames does not necessarily provide additional informative signals, as neighboring frames tend to become semantically redundant under smooth restoration trajectories in the latent space. Consequently, a moderate number of frames is sufficient for approximating the restoration process, while excessive temporal sampling may introduce redundant constraints and increase computational burden without significant performance gains. This further indicates that restoration performance is mainly determined by learned spatial semantic priors rather than dense temporal sampling.

Progressive Curriculum Training. We study the effectiveness of the proposed progressive curriculum training strategy, as shown in Tab. 4, where we compare fixed resolution training with different progressive schedules. Here, numbers denote the training resolution (e.g., 512 indicates resizing the input image such that the shorter side is aligned to 512 while preserving the aspect ratio, followed by 512×512 random cropping). All models are trained for 300 epochs in total, with epochs evenly distributed across stages for multi-stage training.

The results show that performance consistently improves with increasing resolution, since higher resolutions provide richer high frequency structural and textural cues for learning more discriminative restoration priors. Moreover, progressive resolution training outperforms single resolution training (e.g., 512+720 and 512+720+960), suggesting that a coarse to fine optimization paradigm better matches the hierarchical nature of image restoration. In contrast, decreasing resolution schedules lead to performance degradation (e.g. 720+512), as the model tends to prioritize global appearance consistency while sacrificing fine detail modeling. Overall, the proposed strategy consistently enhances both training stability and restoration quality by better aligning optimization with the intrinsic coarse to fine restoration process.

##### 4.4 Discussions

Few-shot Capability. We discuss the effectiveness of leveraging video generation priors for few shot image restoration. As shown in Tab. 1, our method achieves competitive or even superior performance while using significantly less training data. Specifically, our model is trained using only 0.1% – 7% of the data required by conventional methods, demonstrating that large scale video generative pretraining captures strong generic structural and motion aware priors that

###### Table 5: Ablation study about the scale of training data (PSNRSSIM , both higher is better).

Isolated Degradation Coupled Degradation

Data Scale

Average Blur Noise JPEG Haze Rain Raindrop Lowlight B+N B+J N+J R+H L+H L+R L+B L+N L+J L+B+N L+B+J L+N+J B+N+J

22.14 33.90 27.01 19.64 25.96 24.34 20.21 22.49 22.03 30.06 18.64 18.52 30.37 21.39 16.85 19.41 21.85 20.37 23.91 21.85 22.94 0.7213 0.9463 0.8340 0.6568 0.7964 0.7540 0.8548 0.7380 0.6789 0.9156 0.5297 0.5822 0.8931 0.6811 0.6414 0.7064 0.7109 0.6582 0.9116 0.6954 0.7360

0.2K

- 1K

22.33 33.64 27.97 20.64 25.76 25.63 19.34 22.24 23.21 28.48 19.79 19.57 32.06 19.18 17.36 22.63 22.10 21.95 24.83 21.45 23.35 0.7253 0.9440 0.8464 0.6854 0.7792 0.7623 0.8363 0.7275 0.7131 0.8869 0.5594 0.5948 0.9010 0.6411 0.7036 0.7631 0.7030 0.7368 0.9055 0.6823 0.7442

- 2K

- 21.92 31.98 27.50 19.64 26.73 25.74 20.63 25.79 25.18 28.86 18.37 19.81 32.25 21.65 16.08 22.26 22.08 21.01 26.28 23.79 23.54 0.7082 0.9432 0.8361 0.6501 0.8003 0.7611 0.8630 0.8199 0.7718 0.8930 0.5273 0.6120 0.9081 0.6822 0.6486 0.7367 0.6946 0.7447 0.9135 0.7279 0.7469 8K

- 22.98 33.41 27.57 21.18 26.54 25.11 21.49 25.10 26.02 30.64 19.97 20.00 31.67 21.76 16.28 21.14 22.86 21.86 28.27 23.91 24.14 0.7284 0.9454 0.8375 0.6704 0.7982 0.7611 0.8628 0.7999 0.7827 0.9088 0.5562 0.6200 0.9063 0.7153 0.6458 0.7214 0.7354 0.7505 0.9137 0.6786 0.7549

###### Table 6: Quantitative comparisons (PSNRSSIM ) on the unseen task for our method about desnowing. Underlining and boldface indicate the second-best and best methods, respectively. FoundIR-G denotes the generalist version of FoundIR.

Methods PromptIR TransWeather DA-CLIP DiffUIR X-Restormer AgenticIR FoundIR-G Ours Data Scale ↓ 77K 19K 53K 30K 40K 15K 1M 1K

PSNR ↑ 21.54 20.94 21.59 21.68 21.57 20.35 21.57 20.88 SSIM ↑ 0.7767 0.7579 0.7731 0.7795 0.7813 0.7304 0.7794 0.7105

WeatherBench

can be transferred to restoration tasks. From a learning perspective, our approach focuses on activating and adapting pretrained generative knowledge with limited task specific image restoration supervision, reducing data dependency while preserving model expressiveness.

As shown in Tab.5 and Fig.5, the performance of our method consistently improves as the amount of training data increases. For a fair comparison, all models are trained for 300 epochs at a fixed 720 resolution. Notably, using only 200 training samples, our method already achieves performance comparable to existing full-data baselines, highlighting strong data efficiency and generalization capability. This suggests that the pretrained video generative prior acts as a powerful regularization mechanism, enabling robust restoration under limited supervision while allowing further performance gains as more data becomes available. However, it is worth noting that expanding the training set does not consistently improve performance across all degradation types, suggesting a potential trade-off or balance during model learning.

Overall, these results further validate the strong potential of exploiting video generative priors for few-shot image restoration, suggesting that large-scale generative pretraining can serve as a general knowledge foundation for data-efficient visual learning. By transferring high-level semantic, structural, and motionaware priors from video, generative models provide a promising paradigm for building scalable and data-efficient visual intelligence that generalizes beyond restoration tasks to broader visual understanding and synthesis problems.

Generalization Capability. As shown in Tab. 2, although trained on only a 1K subset of FoundIR training data, our model generalizes well to multiple external benchmarks and achieves performance comparable to top-tier methods, demonstrating strong data efficiency and the feasibility of competitive restoration without large scale task specific supervision.

Furthermore, Fig. 6 and Tab. 6 present qualitative and quantitative results on snow removal [16], an unseen task for our model. Although trained with only lim-

LQ FoundIR-G Ours GT LQ FoundIR-G Ours GT

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

- Fig. 6: Visualization results on an OOD snow removal scenario. FoundIR-G denotes the Generalist model of FoundIR. GT denotes ground truth. Both methods are evaluated under OOD conditions. Although our model is trained with only 1K samples, it shows significantly better generalization than FoundIR-G trained with 1M images.

ited image restoration data and without additional training on new degradation types, the model generalizes effectively to unseen corruptions and successfully removes snow artifacts while producing visually coherent and competitive results. This demonstrates that the model can learn transferable restoration priors and directly transfer knowledge learned during pretraining to new tasks, such as applying its understanding of snow patterns to desnowing tasks. This capability reflects the strong inductive power of video generation models, which can implicitly capture rich semantic and structural knowledge during pretraining. It is particularly important for few-shot image restoration, where minimal training data is sufficient to enable universal degradation modeling. Overall, these results highlight the strong generalization potential of generative foundation models, showing that minimal supervision can unlock broad cross-domain adaptability and pave the way toward more general visual intelligence.

#### 5 Conclusion

In this paper, we introduced V-Bridge, a framework that unlocks the restoration capability of pretrained video generative models with few-shot training. By reformulating restoration as a progressive generative refinement process and exploiting chain-like frame reasoning, we show that video foundation models can serve as powerful and transferable visual priors. Remarkably, our approach activates strong restoration capability using only a very small number of task-specific samples, further demonstrating the broad potential of video generative models beyond video synthesis. We also propose efficient progressive resolution training and lightweight refinement strategies to achieve high-fidelity restoration under limited supervision. This work provides new evidence that video generative models can serve as general visual foundation models and opens new directions for unified generative modeling in vision tasks.

#### References

- 1. Ancuti, C.O., Ancuti, C., Sbert, M., Timofte, R.: Dense-haze: A benchmark for image dehazing with dense-haze and haze-free images. In: 2019 IEEE international conference on image processing (ICIP). pp. 1014–1018. IEEE (2019)
- 2. Ancuti, C.O., Ancuti, C., Timofte, R.: NH-HAZE: an image dehazing benchmark with non-homogeneous hazy and haze-free images. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. IEEE CVPR 2020 (2020)
- 3. Banham, M.R., Katsaggelos, A.K.: Digital image restoration. IEEE signal processing magazine 14(2), 24–41 (2002)
- 4. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023)

- 5. Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators (2024), https://openai.com/research/videogeneration-models-as-world-simulators
- 6. Chang, W., Chen, H., He, X., Chen, X., Shen, L.: Uav-rain1k: A benchmark for raindrop removal from uav aerial imagery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15–22 (2024)
- 7. Chen, H., Xia, M., He, Y., Zhang, Y., Cun, X., Yang, S., Xing, J., Liu, Y., Chen, Q., Wang, X., et al.: Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023)
- 8. Chen, H.H., Lan, D., Shu, W.J., Liu, Q., Wang, Z., Chen, S., Cheng, W., Chen, K., Zhang, H., Zhang, Z., et al.: Tivibench: Benchmarking think-in-video reasoning for video generative models. arXiv preprint arXiv:2511.13704 (2025)
- 9. Chen, X., Pan, J., Dong, J., Tang, J.: Towards unified deep image deraining: A survey and a new benchmark. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)
- 10. Chen, X., Li, Z., Pu, Y., Liu, Y., Zhou, J., Qiao, Y., Dong, C.: A comparative study of image restoration networks for general backbone network design. In: European Conference on Computer Vision. pp. 74–91. Springer (2024)
- 11. Conde, M.V., Geigle, G., Timofte, R.: Instructir: High-quality image restoration following human instructions. In: Proceedings of the European Conference on Computer Vision (ECCV) (2024)
- 12. Deng, H.: Video models start to solve chess, maze, sudoku, mental rotation, and raven’matrices. arXiv preprint arXiv:2512.05969 (2025)
- 13. Fang, Y., Zhang, H., Wong, H.S., Zeng, T.: A robust non-blind deblurring method using deep denoiser prior. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 735–744 (2022)
- 14. Gao, Y., Guo, H., Hoang, T., Huang, W., Jiang, L., Kong, F., Li, H., Li, J., Li, L., Li, X., et al.: Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113 (2025)
- 15. Google DeepMind: Veo (2025), https://deepmind.google/models/veo
- 16. Guan, Q., Yang, Q., Chen, X., Song, T., Jin, G., Jin, J.: Weatherbench: A realworld benchmark dataset for all-in-one adverse weather image restoration. In: Proceedings of the 33rd ACM international conference on multimedia. pp. 12607–12613

(2025)

- 17. Guo, C.L., Yan, Q., Anwar, S., Cong, R., Ren, W., Li, C.: Image dehazing transformer with transmission-aware 3d position embedding. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5812–5820

(2022)

- 18. Guo, Z., Chen, X., Zhang, R., An, R., Qi, Y., Jiang, D., Li, X., Zhang, M., Li, H., Heng, P.A.: Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802 (2025)
- 19. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022)
- 20. Jiang, Y., Zhang, Z., Xue, T., Gu, J.: Autodir: Automatic all-in-one image restoration with latent diffusion. In: European Conference on Computer Vision. pp. 340–

359. Springer (2024)

- 21. Jin, X., Han, L.H., Li, Z., Guo, C.L., Chai, Z., Li, C.: Dnf: Decouple and feedback network for seeing in the dark. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18135–18144 (2023)
- 22. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)
- 23. Kuaishou: Kling video model. https://kling.kuaishou.com/en (2024)
- 24. Li, B., Liu, X., Hu, P., Wu, Z., Lv, J., Peng, X.: All-In-One Image Restoration for Unknown Corruption. In: IEEE Conference on Computer Vision and Pattern Recognition. New Orleans, LA (Jun 2022)
- 25. Li, H., Chen, X., Dong, J., Tang, J., Pan, J.: Foundir: Unleashing million-scale training data to advance foundation models for image restoration. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 12626–12636

(2025)

- 26. Li, Y., Gu, Y., Min, Y., Liu, Z., Du, Y., Zhou, K., Yang, M., Zhao, W.X., Qiu, M.: Viper: Process-aware evaluation for generative video reasoning. arXiv preprint arXiv:2512.24952 (2025)
- 27. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1833–1844 (2021)
- 28. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1833–1844 (2021)
- 29. Lin, X., He, J., Chen, Z., Lyu, Z., Dai, B., Yu, F., Qiao, Y., Ouyang, W., Dong, C.: Diffbir: Toward blind image restoration with generative diffusion prior. In: European conference on computer vision. pp. 430–448. Springer (2024)
- 30. Luo, Y., Zhao, X., Lin, B., Zhu, L., Tang, L., Liu, Y., Chen, Y.C., Qian, S., Wang, X., You, Y.: V-reasonbench: Toward unified reasoning benchmark suite for video generation models. arXiv preprint arXiv:2511.16668 (2025)
- 31. Luo, Z., Gustafsson, F.K., Zhao, Z., Sjölund, J., Schön, T.B.: Controlling visionlanguage models for multi-task image restoration. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/forum? id=t3vnnLeajU
- 32. Ma, J., Liang, Z., Xiang, W., Yang, X., Zhang, L.: A benchmark for chinese-english scene text image super-resolution. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 19452–19461 (2023)

- 33. Ma, J., Cheng, T., Wang, G., Zhang, Q., Wang, X., Zhang, L.: Prores: Exploring degradation-aware visual prompt for universal image restoration. arXiv preprint arXiv:2306.13653 (2023)
- 34. Mou, C., Wang, Q., Zhang, J.: Deep generalized unfolding networks for image restoration. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 17399–17410 (2022)
- 35. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)
- 36. Potlapalli, V., Zamir, S.W., Khan, S.H., Shahbaz Khan, F.: Promptir: Prompting for all-in-one image restoration. Advances in neural information processing systems 36, 71275–71293 (2023)
- 37. Tong, C., Chang, M., Zhang, S., Wang, Y., Liang, C., Zhao, Z., An, R., Zeng, B., Shi, Y., Dai, Y., et al.: Cof-t2i: Video models as pure visual reasoners for text-toimage generation. arXiv preprint arXiv:2601.10061 (2026)
- 38. Valanarasu, J.M.J., Yasarla, R., Patel, V.M.: Transweather: Transformer-based restoration of images degraded by adverse weather conditions. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2353– 2363 (2022)
- 39. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)
- 40. Wang, T., Zhang, K., Shen, T., Luo, W., Stenger, B., Lu, T.: Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 2654– 2662 (2023)
- 41. Wang, X., Xie, L., Dong, C., Shan, Y.: Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1905–1914 (2021)
- 42. Wiedemer, T., Li, Y., Vicol, P., Gu, S.S., Matarese, N., Swersky, K., Kim, B., Jaini, P., Geirhos, R.: Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328 (2025)
- 43. Wu, J., Huang, T., He, C., Long, M.: Miniveo3-reasoner: Thinking with videos from open-source priors. https://github.com/thuml/MiniVeo3-Reasoner (2025)
- 44. Yang, C., Wan, H., Peng, Y., Cheng, X., Yu, Z., Zhang, J., Yu, J., Yu, X., Zheng, X., Zhou, D., et al.: Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks. arXiv preprint arXiv:2511.15065

(2025)

- 45. Zhang, X., Ma, J., Wang, G., Zhang, Q., Zhang, H., Zhang, L.: Perceive-ir: Learning to perceive degradation better for all-in-one image restoration. IEEE Transactions on Image Processing (2025)
- 46. Zheng, D., Wu, X.M., Yang, S., Zhang, J., Hu, J.F., Zheng, W.S.: Selective hourglass mapping for universal image restoration based on diffusion model. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 25445–25455 (2024)
- 47. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404 (2024)

## Appendix for V-Bridge

#### A Details

##### A.1 Training Details

Data. For the training data, we construct high-quality (HQ) and low-quality (LQ) image pairs using samples from FoundIR and RealCE. Specifically, FoundIR contains several common degradation categories, including blur, lowlight, JPEG compression, haze, rain, as well as combinations of multiple degradations. To maintain a balanced distribution, we perform uniform sampling across these degradation categories. In contrast, RealCE mainly consists of text degradation data in both Chinese and English, and its samples are treated as a single category during sampling. Unless otherwise specified in the main paper, the default training set size used in our experiments is 1K. The detailed data construction procedure can be found in Sec. 3.2.

Table 7: Training Configurations for V-Bridge. Parameter Value

BF16 True Learning Rate 2e-5 LR Scheduler Type constant_with_warmup LR Warmup Steps 100 Adam Weight Decay 3e-2 Adam Epsilon 1e-10 Maximum Grad Norm 0.05 GPUs Per Node 8 Number of Nodes 1 Seed 42

Details. During training, the hyperparameters used in our experiments are summarized in Tab. 7. As described in Sec. 3.3, we adopt a three-stage training strategy. The primary difference across stages lies in the input video resolution, while all other training hyper-parameters remain consistent throughout the entire training process. Specifically, the three training stages use resolutions of 512, 720, and 960, respectively. The meaning of these resolutions is defined as follows. Taking 512 as an example, for each frame in the original video, we first resize the frame while preserving its aspect ratio such that the shorter side is scaled to 512. Then, a 512×512 patch is randomly cropped from the resized frame for training. The resolutions used in the other stages follow the same procedure. The total training schedule consists of 300 epochs, which are evenly divided across the three stages.

##### A.2 Test Details

Benchmarks. Next, we provide a detailed description of each benchmark used in our experiments, including the specific degradation types involved and their corresponding detailed information.

- – FoundIR [25]. This test set includes common degradation types such as blur, low-light, JPEG compression, haze, rain, as well as combinations of multiple degradations. It contains a total of 1,500 samples, with the majority of the data at resolutions of 1080p or higher.
- – Dense-Haze [1]. Dense-Haze comprises 33 pairs of real-world hazy and corresponding haze-free images, capturing a variety of outdoor scenes. The hazy conditions are densely distributed and visually uniform, created using professional haze machines to introduce authentic atmospheric haze.
- – UHD-LL [40]. The first real UHD low-light image enhancement dataset, featuring 4K image pairs captured under diverse conditions with different levels of darkness and noise. The dataset provides a test set consisting of 115 image pairs for evaluation purposes.
- – UAV-Rain1K [6]. This dataset is specifically designed for studying rain removal in aerial imagery captured by unmanned aerial vehicles (UAVs). It provides a comprehensive collection of images that reflect various rainy conditions and environmental scenarios. The test set consists of 220 paired images, each with an average resolution of around 1500 × 1000 pixels, offering sufficient detail for evaluating the performance of deraining algorithms.
- – HQ-NightRain [9]. The test portion of the dataset consists of 300 image pairs and is categorized based on the type of nighttime rain degradation into three distinct subsets: rain streaks (RS), raindrops (RD), and mixed rain (SD), which combines both streaks and drops.
- – WeatherBench [16]. This dataset contains images with snow, haze, and rain degradations. In our evaluation, we only utilize the snow subset, which provides 200 pairs of corresponding low-quality and high-quality images. It is worth noting that snow removal is an unseen task for our model, as this type of degradation is not included in the tasks encountered during training.

Inference. During inference, a unified prompt is employed across all tasks, with the full prompt details provided in Box.A.1 and A.2. Since generating highresolution videos (e.g., 4K) directly with video generation models incurs substantial VAE decoding time, we adopt a practical approach for efficiency: frames with resolutions exceeding 2K are first resized to approximately 2K while maintaining their original aspect ratio for inference. The outputs are then rescaled back to the original resolution prior to metric computation. During generation, we use UniPC, adapted for flow matching, as the sampling scheduler with 50 sampling steps. The classifier-free guidance scale is set to 5.0, and the timestep shift is also set to 5.0.

##### Box A.1: Prompt for Image Restoration

A restoration-focused video strictly based on the input image. The camera is completely static with no movement, no zoom, and no rotation. The original composition, objects, layout, and perspective are preserved exactly. Focus on visual restoration and enhancement: remove noise, reduce blur, eliminate rain artifacts, remove compression artifacts, and improve clarity, sharpness, and fine details while maintaining natural textures, accurate colors, and balanced lighting. Only extremely subtle and natural temporal consistency is allowed. The video should appear stable, clean, and realistic, as if the input image has been gently restored over time.

##### Box A.2: Negative Prompt for Image Restoration

camera movement, panning, tilting, zooming, rotation, scene change, object movement, new objects, object deformation, style change, artistic style, illustration, painting, cartoon, over-saturated colors, overexposure, underexposure, motion blur, jitter, flickering, shaking, low quality, worst quality, noise, blur, rain, fog, compression artifacts, jpeg artifacts, aliasing, text, subtitles, watermark, logo, distorted anatomy, extra limbs, duplicated objects, exaggerated motion, creative animation

Metrics. In this work, we adopt PSNR and SSIM as evaluation metrics, where higher values indicate better performance. Following the computation methodology in FoundIR , these metrics are defined as:

##### 1. PSNR (Peak Signal-to-Noise Ratio):

H

W

1 HW

MSE =

i=1

j=1

PSNR = 10 · log10

I(i,j) − Iˆ(i,j) 2, (7)

L2 MSE

(8)

where I and Iˆ denote the ground-truth and reconstructed images, H and W are the height and width of the images, and L is the maximum pixel value (e.g., 255 for 8-bit images).

##### 2. SSIM (Structural Similarity Index):

(2µIµIˆ + C1)(2σIIˆ + C2) (µ2I + µ2Iˆ + C1)(σI2 + σI2ˆ + C2)

SSIM(I,Iˆ) =

(9)

where µI,µIˆ and σI2,σI2ˆ are the means and variances of I and Iˆ, σIIˆ is their covariance, and C1,C2 are stabilizing constants.

Both metrics are computed per image and averaged over the dataset to obtain the final evaluation scores.

#### B Additional Experimental Results

- B.1 Visualization Results We present additional visualization results in Fig.7, Fig.8, and Fig.9.
- B.2 Failure Cases

We present several failure cases of our method as shown in Fig.10, along with the corresponding visualization results of FoundIR-G on the same data.

#### C Limitations and Future Work

This work primarily explores the feasibility of transferring the prior knowledge of video generation models to low-level vision tasks, and demonstrates the effectiveness and potential of such a transfer. However, the current approach faces two main limitations: task generalization and efficiency. In future work, we aim to extend the transfer of video generation priors to a broader range of tasks, further validating the potential of video generation models as foundation models for vision. Additionally, since the present study focuses solely on demonstrating potential, we did not incorporate any acceleration strategies to avoid confounding factors in the analysis. Moving forward, we plan to introduce acceleration techniques to make the approach practically usable in real-world scenarios.

###### Corrupted Image LQ Patch FoundIR-G Ours GT Patch

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Haze

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Haze

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Low-Light

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Low-Light

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Rain

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Low-Light

###### Corrupted Image LQ Patch FoundIR-G Ours GT Patch

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Low-Light

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Low-Light

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Low-Light

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Low-Light

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Rain

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Rain

###### Corrupted Image LQ Patch FoundIR-G Ours GT Patch

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Rain

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

Rain

Snow Corrupted Image FoundIR-G Ours GT Patch

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

###### Corrupted Image LQ Patch FoundIR-G Ours GT Patch

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Haze

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Rain

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

Low-Light

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Rain

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Rain

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

Haze

###### Fig. 10: Failure cases.

