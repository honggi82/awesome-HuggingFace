# arXiv:2601.21716v1[cs.CV]29Jan2026

[Figure 1]

## DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning

### Mingshuang Luo123,∗, Shuang Liang1,∗, Zhengkun Rong1,∗, Yuxuan Luo1,†, Tianshu Hu1,§, Ruibing Hou2,§, Hong Chang23, Yong Li4, Yuan Zhang1, Mingyuan Gao1

1ByteDance Intelligent Creation 2Key Lab of Intell. Info. Process., ICT, CAS 3University of Chinese Academy of Sciences 4Southeast University

∗Equal Contribution, †Project Lead, §Corresponding Authors

### Abstract

Character image animation aims to synthesize high-fidelity videos by transferring motion from a driving sequence to a static reference image. Despite recent advancements, existing methods suffer from two fundamental challenges: (1) suboptimal motion injection strategies that lead to a trade-off between identity preservation and motion consistency, manifesting as a "see-saw", and (2) an over-reliance on explicit pose priors (e.g., skeletons), which inadequately capture intricate dynamics and hinder generalization to arbitrary, non-humanoid characters. To address these challenges, we present DreamActor-M2, a universal animation framework that reimagines motion conditioning as an in-context learning problem. Our approach follows a two-stage paradigm. First, we bridge the input modality gap by fusing reference appearance and motion cues into a unified latent space, enabling the model to jointly reason about spatial identity and temporal dynamics by leveraging the generative prior of foundational models. Second, we introduce a self-bootstrapped data synthesis pipeline that curates pseudo cross-identity training pairs, facilitating a seamless transition from pose-dependent control to direct, end-to-end RGB-driven animation. This strategy significantly enhances generalization across diverse characters and motion scenarios. To facilitate comprehensive evaluation, we further introduce AW Bench, a versatile benchmark encompassing a wide spectrum of characters types and motion scenarios. Extensive experiments demonstrate that DreamActor-M2 achieves state-of-the-art performance, delivering superior visual fidelity and robust cross-domain generalization.

Date: January 30, 2026 Correspondence: tianshu.hu@bytedance.com, houruibing@ict.ac.cn Project Page: https://grisoon.github.io/DreamActor-M2/

### 1 Introduction

Character image animation [5, 7, 13, 19, 30, 37, 38, 43, 52] aims to synthesize high-fidelity videos by transferring motion from a driving sequence to a static reference image, a task with vast potential in digital entertainment. While recent foundational video diffusion models [3, 14, 33, 34, 42, 48] provide powerful generative priors,

[Figure 2]

Figure 1 The proposed DreamActor-M2 exhibits strong generalization capability, producing diverse animations while preserving consistent character appearance. The arrow −→ denotes the transfer of character motion from the driving video to the character depicted in the reference image.

effectively adapting them for animation without compromising their intrinsic generative capabilities remains a challenge. Moreover, concurrently achieving robust identity preservation,accurate motion fidelity, and strong cross-domain generalization still pose a major challenge for existing frameworks.

These limitations primarily stem from two fundamental drawbacks. First, prevailing motion injection strategies exhibit an inherent trade-off between identity preservation and motion consistency. Specifically, pose-aligned channel-wise injection methods [19, 27, 43, 52] frequently suffer from "shape leakage", whereby the structural priors embedded in the driving signal distort the reference identity. Conversely, cross-attention based methods [9, 37, 38] often overly compress motion representations, resulting in the loss of fine-grained temporal dynamics and degraded motion coherence. This "see-saw" effect prevents existing frameworks from simultaneously achieving high-fidelity animation and stable identity preservation. Second, the heavy reliance on explicit pose priors (e.g., skeletons) imposes a representation bottleneck that restricts model flexibility and generalization [7, 30, 46, 53]. Such pose estimators [29, 47] are inherently error-prone in complex human-centric dynamics and, more critically, are intrinsically incapable of generalizing to arbitrary non-humanoid characters, including cartoons and animals. Although implicit motion representations have been explored to mitigate this dependency [35, 50], these methods either remain tethered to pose-derived supervision during training or necessitate costly per-video fine-tuning to capture motion-specific details. Consequently, such constraints

substantially impede their scalability and applicability in diverse real-world scenarios.

To address these challenges, we propose DreamActor-M2, a universal character animation framework that reframes motion conditioning as an in-context learning problem. Departing from traditional approaches that rely on complex motion injection modules, our method adopts a simple yet effective design: motion control signals are spatiotemporally concatenated with the reference image to construct a unified input representation. Through this design, the pre-trained video backbone can naturally interpret motion cues as visual context, thereby effectively bridging the modality gap between appearance and motion. This formulation preserves the original model architecture while fully exploiting the inherent generative priors of foundation video models, enabling high-fidelity animation without compromising their intrinsic capabilities.

Our framework evolves through a strategic two-stage paradigm. In the first stage, we develop Pose-based DreamActor-M2, which leverages augmented 2D skeletons as an initial form of motion context. To alleviate the semantic insufficiency of pose-based conditioning, we incorporate a target-oriented motion-semantic guidance module powered by Multimodal Large Language Models (MLLMs). This module aligns visual cues with fine-grained semantic descriptions, enabling the model to simultaneously preserve identity fidelity and motion consistency. In the second stage, we advance to End-to-End DreamActor-M2, with the goal of removing the reliance on external pose estimators. To this end, we propose a self-bootstrapped data synthesis pipeline that exploits the pose-based variant to curate a high-quality pseudo-paired dataset. Training on these synthesized cross-identity pairs allows the end-to-end model to derive motion directly from raw RGB sequences, without any explicit pose supervision. This progressive transition not only circumvents the inherent limitations of pose estimation but also substantially extrapolates the model’s generalization capability to arbitrary characters and complex motion scenarios.

To facilitate a rigorous evaluation, we introduce AW Bench, a comprehensive benchmark encompassing a wide spectrum of character categories and motion types. Extensive experimental results demonstrate that DreamActor-M2 achieves state-of-the-art performance in term of visual fidelity and cross-domain generalization.

Our key contributions are summarized as follows:

- • We propose a spatiotemporal in-context motion conditioning strategy that bridges modality gaps and effectively balances identity preservation with motion consistency for character image animation.
- • We propose a self-bootstrapped synthesis-and-training pipeline that enables a seamless transition to end-to-end, pose-free animation, substantially improving generalization across diverse character domains.
- • We construct a comprehensive benchmark encompassing diverse motion patterns and character categories, offering the community a more challenging and rigorous evaluation platform for character image animation.

### 2 Related Work

Latent Video Diffusion Models. Diffusion-based generative models [3, 14, 25, 33, 42, 48, 51] have recently achieved remarkable success in video generation. Notably, Wan2.1 [42] and Seedance 1.0 [14] have emerged as high-performance video foundation models, supporting both text-to-video and image-to-video generation. Since character image animation inherently aligns with the image-to-video setting, we adopt Seedance 1.0 as the pre-trained backbone for our DreamActor-M2 framework.

Pose Guidance in Character Image Animation. Pose-guided approaches have emerged as the dominant paradigm for character image animation. A line of works [6, 13, 17, 19, 40, 43, 45, 52, 53] inject 2D skeleton or SMPL signals as motion conditions via channel-wise concatenation or additive fusion. While ensuring spatial alignment and motion consistency, training on same-identity data often induces identity leakage, where identity-specific appearance cues become entangled with motion features, substantially degrading cross-identity generalization. To address this issue, alternative methods [9] adopt cross-attention injection mechanisms. In these approaches, an auxiliary pose encoder compresses motion signals into latent representations, which are then injected into the generation backbone through cross-attention. This strategy often decouples pose from identity by employing skeleton scaling [39] or SMPL normalization. Additionally, [24] concatenates conditioning and target frames along the temporal dimension, leveraging the backbone’s temporal modeling

[Figure 3]

Figure 2 The schematic overview of proposed DreamActor-M2.

capacity for capturing global motion patterns. However, such designs inevitably distort fine-grained motion semantics, ultimately compromising motion fidelity.

Another line of research [44, 50] avoids explicit motion representations altogether, instead directly exploiting raw RGB frames for motion guidance. DreamVideo [44] adopts a "one-model-per-identity" paradigm, which severely limits generalization and necessitates retraining for each novel character. FlexiAct [50] requires a dedicated frequency-aware embedding for each driving video, introducing substantial computational overhead. X-Unimotion [35] leverages implicit motion representations during generation, yet still relies on 2D pose signals as supervision during training, and thus remains constrained by inherent limitations of pose-based methods.

In-Context Learning. Despite its success in large language models (LLMs) [4, 10, 18, 32], vision-language models (VLMs) [1, 11, 31], and image generation [20, 21, 26, 28, 36], in-context learning (ICL) remains relatively underexplored in video generation. TIC-FT [24] temporally concatenates condition and target frames to boost fidelity and training/generation efficiency, it does not address the unique challenges of character animation. A concurrent work, SCAIL [46],leverages full 3D pose sequences as contextual inputs to achieve studio-grade animation quality. However, its performance is highly contingent on accurate 3D motion representations, which limits its applicability in practical settings where such structural signals are noisy, incomplete, or unavailable. In contrast, our end-to-end DreamActor-M2 eliminates explicit pose dependencies by directly interpreting raw motion signals as contextual inputs. This design significantly enhances generalization across diverse character types and unconstrained motion patterns.

### 3 Approach

This section presents DreamActor-M2, a universal character animation framework conditioned on a reference image and driving signals (e.g., pose sequences or video clips). The methodology is organized as follows: Sec. 3.1 briefly reviews the Diffusion Transformer (DiT) backbone that serves as the foundation of our approach. Sec. 3.2 introduces our core in-context motion injection strategy. Building upon this mechanism, Sec. 3.3 elaborates on the Pose-based DreamActor-M2 framework. Finally, Sec. 3.4 details the evolution toward an End-to-End DreamActor-M2 pipeline that eliminates the reliance on explicit pose priors. A schematic overview of the proposed framework is provided in Fig. 2.

##### 3.1 Preliminary

Latent Diffusion Model. Our framework is built on the Latent Diffusion Model (LDM), where a Variational Autoencoder (VAE) encodes input images I into latent representations z = ξ(I). Gaussian noise ϵ is progressively injected into latents zt at different timesteps, with the optimization objective:

t,c,ϵ,t ∥ϵ − ϵθ (zt,c,t)∥22 (1)

L = Ez

where ϵθ denotes the denoising network and c is the conditional input. At inference, noise latents are iteratively denoised and reconstructed into images via VAE decoder. We adopt Seedance 1.0 [14] as our backbone, which employs the MMDiT architecture [12] to support multi-modal and multi-task video generation.

##### 3.2 Motion Injection via In-Context Learning

Existing motion injection methods suffer from an inherent trade-off between identity preservation and motion consistency. To contextualize our approach, we first review representative alternatives before introducing our spatiotemporal in-context injection strategy.

- Alternative 1: Pose Alignment Injection. Several works [19, 43] encode 2D pose signals into latent representations and inject them into noise latents via channel-wise concatenation or additive fusion. Such designs enforce motion consistency through strict spatial alignment, however, pose encoding inevitably carry pose shape and structural cues, leading to dentity leakage and noticeable identity distortion during animation.
- Alternative 2: Cross-Attention Injection. Other methods [9] compress motion signals into latents via an auxiliary pose encoder and inject them into the generation backbone via cross-attention. By decoupling motion control from explicit spatial alignment, these methods partially alleviate identity leakage. Nevertheless, the required latent compression often discards fine-grained motion details, frequently resulting in temporally or anatomically unnatural animations.
- Alternative 3: Temporal-Level In-Context Injection. As explored in [24], conditioning and target frames can be concatenated along the temporal dimension, allowing the backbone’s temporal modeling capacity to learn global motion patterns effectively. However, the absence of frame-wise spatial correspondence tends to degrade fine-grained motion details and leads to suboptimal reconstruction quality.

Our Method: Spatiotemporal In-Context Injection. Inspired by in-context learning (ICL) in LLMs [4, 18] and VLMs [1, 11, 31], which achieves seamless task adaptation via direct input integration, we propose a spatiotemporal ICL strategy to resolve the aforementioned trade-off. To model the video’s spatiotemporal structure, we create a unified sequence by: (1) spatially concatenating the reference image and the first motion frame as a hybrid anchor, (2) aligning the subsequent motion frames with a reference-sized blank mask, and (3) temporally stacking all frames. This approach avoids lossy compression, bridges modality gaps, and unleashes the pre-trained model’s potential for superior identity and motion fidelity.

In-Context Operation. Our objective is to generate a video where the subject identity of Iref ∈ RH×W×3 follows driving motion signals D ∈ RT×H×W×3. We construct a composite input sequence C ∈ RT×H×2W×3 as follows:

Iref ⊕ D[t], t = 0, 0 ⊕ D[t], t > 0.

(2)

C[t] =

where ⊕ denotes spatial concatenation along the width axis, and 0 is a zero image matching Iref dimensions. To guide the model, we construct a motion mask Mm (all elements set to 1) for motion region highlighting, and a reference mask Mr (1 for the first frame, 0 otherwise) to distinguish the identity source. Their spatial concatenation yields M = Mr ⊕ Mm. The composite video C is projected into the latent sequence Z via a

- 3D VAE. Finally, Z, noise latent Znoise, and mask M are channel-concatenated to serve as the comprehensive input for the diffusion transformer.

##### 3.3 Pose-Based DreamActor-M2

Pose-Based DreamActor-M2 utilizes 2D pose skeletons as motion signals under a self-supervised training paradigm. Given a video V, we extract a pose sequence P as the driving signal and designate the first frame Iref = V [0] as the reference image. The model is then trained to reconstruct the original video V.

Pose Augmentation. 2D skeletons inherently encode "body shape" cues (e.g., limb length and body proportion), which can induce identity leakage and degrade cross-identity generalization. To mitigate this issue while preserving the underlying motion dynamics, we apply two complementary pose augmentation strategies: (1) Random Bone Length Scaling: Skeleton bones are grouped into anatomical segments and subjected to random scaling. This operation perturbs limb proportions while maintaining joint connectivity and temporal motion patterns, thereby decoupling structural priors from motion dynamics. (2) Bounding-Box-Based

Normalization: Joint coordinates are normalized with respect to the bounding box enclosing all joints in a clip. This normalization removes absolute spatial dependencies and produces a scale-invariant pose representation.

Text Guidance. While pose augmentation alleviates identity leakage, it may inadvertently weaken fine-grained motion semantics. For example, perturbations can obscure subtle pose configurations such as clasped hands in a “prayer” gesture. To compensate for this loss, we introduce a target-oriented text guidance mechanism that explicitly integrates motion semantics with appearance descriptions. Specifically, a multimodal large language model (MLLM) is first employed to parse the driving video V into motion semantics Tm (e.g., "a person is waving both hands") and to analyze the reference image Iref to obtain appearance semantics Ta (e.g., "a gray bird with colorful feathers"). An LLM subsequently fuses these descriptions into a target-oriented prompt Tfusion (e.g., "a gray bird with colorful feathers, is waving its wings"). This fused prompt serves as a high-level semantic prior that complements low-level pose signals, improving motion controllability and enhancing the expressiveness of the synthesized animation.

LoRA Fine-tuning. The pre-trained diffusion backbone inherently encodes strong generative priors for structural coherence and temporal consistency. By treating in-context images and textural descriptions as native input modalities, our framework enables seamless integration without architectural modifications, thereby preserving the model’s intrinsic generative capacity. For efficient adaptation, we adopt lightweight LoRA fine-tuning, freezing the backbone parameters and inserting LoRA modules exclusively into feed-forward layers. Notably, the text branch is excluded from adaptation to maintain robust semantic alignment. This design achieves plug-and-play customization with minimal computational overhead.

##### 3.4 End-to-End Training Paradigm

Pose-based DreamActor-M2 relies on explicit pose estimation, which constrains its applicability in complex or non-human animation scenarios. To overcome this, we introduce an end-to-end variant capable of processing raw RGB frames as motion signals, denoted as End-to-End DreamActor-M2. The training procedure involves two stages: (i) data synthesis and quality filtering; (ii) model optimization.

Data Synthesis and Quality Filtering. A key obstacle in end-to-end training lies in the absence of large-scale paired data that simultaneously exhibit motion consistency and cross-identity diversity. To overcome this, we propose a self-bootstrapped data synthesis pipeline that leverages the pre-trained Pose-based DreamActor-M2 to generate high-fidelity pseudo-paired supervision. Formally, given a driving video Vsrc, we extract its pose sequence Psrc as motion signals. Combined with a reference image Io, Psrc is fed to the pose-based DreamActor-M2 Mpose to synthesize a new video Vo:

Vo = Mpose (Psrc,Io). (3)

The synthesized video Vo preserves the motion dynamics of Vsrc while adopting a novel subject identity, thereby forming a pseudo-pair sample (Vsrc,Vo).

To ensure the reliability of the generated supervision, we employ a dual-stage quality filtering strategy. Specifically, we first perform automatic scoring using Video-Bench [16], followed by manual verification focusing on identity fidelity and motion coherence. Only high-quality, semantically consistent pairs are retained for subsequent end-to-end training.

Model Optimization. For end-to-end training, we treat Vo as driving video and Iref = Vsrc [0] as reference image, yields a dataset:

D = {(Vo,Iref,Vsrc)}. (4)

Each triplet supervises the model to reconstruct Vsrc from (Vo,Iref), thereby learning to transfer motion patterns directly from raw RGB sequences. To facilitate stable and efficient optimization, we warm-start with pre-trained Pose-based DreamActor-M2 model. This initialization accelerates convergence and allows the model to inherit robust motion priors learned from explicit pose supervision, substantially improving training efficiency.

By learning motion transfer directly from raw RGB inputs, this paradigm bypasses the need for intermediate pose representations. Together, the Pose-based and End-to-End variants form a unified and versatile framework for character animation. To the best of our knowledge, this work presents the first fully end-to-end solution in this domain, making a significant step toward scalable and practical character animation systems.

#### 4 AW Bench

The core objective of DreamActor-M2 is to enable universal character image animation: it takes a driving videos of any subject (which can be human or non-human) as input, and generates a animated video based on a reference image of any subject (also human or non-human). However, existing evaluation datasets or their setups fail to meet the requirements for evaluating our framework. For instance, some datasets [23, 49] are only applicable to human animation tasks and cannot support broader subject scopes. While other datasets [37, 38] have expanded the subjects of reference images from humans to anthropomorphic characters, their coverage remains insufficient and lacks universality. In practical scenarios, our goal is to use driving videos containing one or more arbitrary subject types (e.g., humans, animals, cartoon characters, etc.) to animate reference images that feature one or more arbitrary subjects (e.g., humans, animals, cartoon characters, etc.).

To comprehensively evaluate the efficacy and generalizability of our DreamActor-M2, we propose the "Animate in the Wild" Benchmark (AW Bench), which encompasses a wide range of motion types and reference identities. The benchmark consists of 100 driving videos and 200 reference images, where the driving corpus covers human as well as non-human motion categories. Human motions are sampled across different body regions (face, upper body, full body), age groups (child, young adult, elderly), and activity categories (e.g., dancing, daily activities), and include both camera-tracked and static-camera sequences. Non-human motions include videos of animals (such as cats, chickens, parrots, monkeys, and orangutans) and animated characters (such as Tom the cat, Jerry the mouse, groundhogs, and cartoon aliens). The reference image corpus includes subjects as rich and diverse in types as those in the driving video corpus. We also explore multi-subject driving scenarios that have not been investigated in existing works, including many-to-many and one-to-many driving. Consequently, our AW Bench further includes multi-subject motion videos and multi-subject reference images. Finally, after data collecting and filtering, AW Bench contains 100 driving videos and 200 reference characters. Visual examples of the driving video corpus and reference image corpus are shown in Fig. 3.

### 5 Experiments

##### 5.1 Implementations

Implementation Details. To train the pose-based DreamActor-M2, we employ two skeleton augmentation strategies: bone length scaling (U(0.8,1.2) applied to 30% of samples) and scale normalization. The model is trained on 100,000 web-collected human videos, with clips randomly sampled between 49 and 121 frames. To ensure a seamless pose transition from the reference image to the driving sequence, we mask the driving

[Figure 4]

Figure 3 Visual examples of driving video corpus and reference image corpus.

signals during the initial one-second segment of each training clip and prepend one second of zero frames (devoid of motion signals) at inference. Gemini 2.5 [8] serves as our (M)LLM for its superior multi-modal reasoning.

For end-to-end DreamActor-M2, we first synthesize large-scale data with diverse reference characters (humans,cartoons,animals) via the pre-trained pose-based model, followed by a two-stage filtering protocol. Specifically, an automated stage via Video-Bench [16] filters for videos with average scores above 4.5, followed by manual verification to ensure rigorous motion consistency (between driving and generated videos) and identity preservation (between reference image and generated characters). This pipeline yields 60,000 high-quality video triplets for reliable end-to-end training.

In our experiments, we train all stages for 50,000 training steps, with a batch size of 2. The LoRA rank is set to 256. We optimize via AdamW with a learning rate of 5 × 10−5 and weight decay of 0.01.

Evaluation Metrics. Most evaluation metrics, such as FID-FVD [2], FVD [41], and CD-FVD [15], rely on comparisons with ground-truth videos, which are unavailable in cross-identity animation scenarios. As a result, these metrics fail to accurately reflect model performance. Moreover, prior studies have revealed that these metrics are often inconsistent with human judgment [22]. To address this, we adopt VideoBench’s human-aligned automatic protocol [16], focusing on four key perceptual dimensions: Imaging Quality, Motion Smoothness, Temporal Consistency, and Appearance Consistency. All dimensions (automatic/human evaluation) use a 1–5 scale (1=very poor, 2=poor, 3=moderate, 4=good, 5=excellent), enabling comprehensive, reliable evaluation in real-world scenarios.

##### 5.2 Quantitative Comparison

Automatic evaluations and Human evaluations. We evaluate the model’s performance with AW Bench to evaluate performance in more diverse scenarios. This test set comprises 60 human-to-human and 40 human-tocartoon animation pairs. We compare our method with several recent state-of-the-art visual approaches. Due to the inherent lack of ground-truth in cross-identity tasks, we adopt Video-Bench [16] for automatic evaluation, focusing on the four key metrics mentioned above. Furthermore, we conduct a user study involving 12 participants, who evaluated 100 randomly selected samples per method on a 5-point scale. As summarized in Tab. 1, our DreamActor-M2 variants outperform all competitors across all automatic metrics, achieving significant

Table 1 Quantitative comparisons with automatic evaluations and human evaluations on AW Bench.

Automatic Evaluations (Video-Bench) Human Evaluations

Method

Imaging Quality ↑

Motion Smoothness ↑

Temporal Consistency ↑

Appearance Consistency ↑

Imaging Quality ↑

Motion Consistency ↑

Appearance Consistency ↑

Animate-X++ [38] 3.45 3.42 4.15 3.21 3.18 ± 0.23 2.95 ± 0.29 2.86 ± 0.34 MTVCrafter [9] 3.71 3.81 4.02 3.53 3.35 ± 0.26 3.26 ± 0.28 3.07 ± 0.36 DreamActor-M1 [30] 4.17 3.92 4.21 4.06 3.96 ± 0.21 3.72 ± 0.26 3.54 ± 0.31 Wan2.2-Animate [7] 4.05 4.06 4.17 3.92 3.91 ± 0.20 3.83 ± 0.25 3.51 ± 0.30

Ours (Pose-based DreamActor-M2) 4.68 4.53 4.61 4.28 4.23 ± 0.19 4.18 ± 0.24 4.12 ± 0.28 Ours (End-to-End DreamActor-M2) 4.72 4.56 4.69 4.35 4.27 ± 0.18 4.24 ± 0.23 4.20 ± 0.29

[Figure 5]

Figure 4 Qualitative comparisons between our method and state-of-the-art approaches on AW Bench.

###### improvements in every evaluation dimension. This superiority is further mirrored in the human evaluation,

where our methods surpass existing baselines by a substantial margin. The consistency between the automated metrics and human evaluation further underscores the superior generation quality and robustness of our DreamActor-M2 framework.

[Figure 6]

[Figure 7]

vs. DreamActor-M1

0.66

0.25

0.09

vs. Wan2.2-Animate

0.62

0.27

0.11

Good Same Bad

vs. Kling-O1

0.5

0.44

0.06

vs. Kling 2.6

0.32

0.46

0.22

GSB comparison with other products.

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1

We conduct a GSB subjective evaluation to compare DreamActor-M2 with mainstream platform-level products based on a same testing dataset. The results in

Figure 5 GSB comparison of DreamActor-M2 and other products.

- Fig. 5 show DreamActor-M2 matches the overall performance of Kling 2.6 with a +9.66% GSB lead, and outperforms other products by a significant margin: +43.66% over Kling-O1, +51.43% over Wan2.2-Animate, and +57.04% over the prior DreamActor-M1. These results fully validate the improvements of DreamActor-M2, which delivers competitive and leading subjective performance against industry platform-level products.

[Figure 8]

Figure 6 Qualitative visualization for ablation study.

##### 5.3 Qualitative Results

Comparison with State-of-the-arts. Fig. 4 presents a qualitative comparison across diverse animation scenarios, ranging from standard intra-domain driving to challenging cross-domain transfer and multi-person scenarios. In the first row, DreamActor-M2 demonstrates plausible visual fidelity, maintaining stringent identity preservation and motion alignment where baselines exhibit blurring. The second row highlights its remarkable body shape preservation and faithful motion alignment with driving inputs, outperforming others in capturing fine-grained motion details. In the third row, the accurate generation of the "heart gesture" validates the model’s superior grasp of motion semantics. Furthermore, the fourth row underscores the robustness of our method in one-to-many driving scenarios, a demanding task where competitors typically suffer from severe visual artifacts or structural collapse. Overall, these results demonstrate that DreamActor-M2 excels in handling heterogeneous animation tasks with exceptional identity preservation and motion consistency.

Visualization for generalization. To qualitatively evaluate DreamActor-M2’s generalization, we conduct experiments across four key scenarios as shown in Fig. 1. (1) Shot types: The model demonstrates exceptional cross-morphology mapping. Notably, in Half2Full tasks, despite the absence of lower-body driving signals, DreamActor-M2 leverages the pre-trained backbone’s generative priors to synthesize plausible lower-body motions while maintaining precise upper-body synchronization. (2) Reference characters: Moving beyond human-centric constraints, our model supports diverse subjects. Fig. 1 showcases high-fidelity animation for animals (rabbit), objects (juice bottle), and cartoon characters (Detective Conan, Pikachu). (3) Driving

characters: Our end-to-end framework generalizes to non-human motion sources. Fig. 1 demonstrates its capacity to process diverse driving signals (e.g., Animal2Animal and Cartoon2Cartoon), enabling transfers that transcend traditional human motion priors. (4) Multi-person scenarios: DreamActor-M2 excels in both One2Multi and Multi2Multi tasks. It effectively synchronizes motion across multiple distinct characters or maps complex multi-person dynamics to new groups without structural collapse. More qualitative results are provided in Fig. 7, Fig. 8, Fig. 9 and Fig. 10.

##### 5.4 Ablation Study

We conduct ablation studies to evaluate each core component of DreamActorM2. First, we compare our spatiotemporal injection approach against the temporal injection method (Temp-IC). As shown in Tab. 2, DreamActorM2 yields better overall generation quality. Moreover, as illustrated in

Human Evaluations

Method

Imaging Quality ↑

Motion

Appearance Consistency ↑

Consistency ↑

Temp-IC 4.12 3.98 4.06 w/o-PoseAug 4.15 3.80 3.92 w/o-TOTG 4.21 3.85 4.08

- Fig. 6 (a), it better preserves intricate structural details—such as hand gestures—underscoring its advantages in spatial fidelity. Next, we ablate pose augmentation (w/o-PoseAug) to evaluate its necessity. As shown in Tab. 2 and Fig. 6(b), pose augmentation consistently improves generation quality and is crucial for maintaining the reference subject’s original body shape. Finally, comparing DreamActor-M2 with a variant lacking target-oriented text guidance (w/o-TOTG) validates the impact of LLM-driven refinement. As demonstrated in Tab. 2 and Fig. 6 (c), injecting targetoriented text information via the LLM yields superior results, enabling DreamActor-M2 to better reconstruct semantically specific motions and preserve character identity. Fig. 6 (d) shows our end-to-end model outperforms the pose-based counterpart in challenging 2D keypoint detection scenarios, including direction ambiguity and hand overlapping.

Ours (Pose-based) 4.23 4.18 4.12

Table 2 Ablation study on proposed DreamActor-M2 framework.

### 6 Conclusion

We introduce DreamActor-M2, a universal framework for character animation. At its core is a spatiotemporal in-context learning strategy that integrates motion and reference signals into a unified representation. This design not only harnesses the pre-trained backbone’s generative priors, but further facilitates an evolution toward direct end-to-end motion transfer from raw videos, thereby eliminating the need for explicit pose estimation. This versatility enables application to increasingly diverse and challenging scenarios. Extensive experiments, including benchmarks on our newly curated AW Bench, demonstrate that DreamActor-M2 establishes a robust and unified paradigm with exceptional fidelity and generalization.

### 7 Limitations and Future Works

While DreamActor-M2 demonstrates robust performance across various scenarios, it occasionally struggles with complex interactions, such as two characters rotating around each other. This is primarily attributed to the scarcity of training data featuring motion trajectory crossing. In the future, we plan to curate more diverse datasets with intricate multi-person interactions to further extend the model’s applicability.

### 8 Ethics considerations

In our data and experiments, a number of human-related images and videos are involved. Meanwhile, our framework is capable of implementing human image animation. Human image animation has possible social risks, like being misused to make fake videos. The proposed technology could be used to create fake videos of people, but existing detection tools can spot these fakes. To reduce these risks, clear ethical rules and responsible usage guidelines are necessary. We will strictly restrict access to our core models and codes to

###### prevent misuse. Images and videos are all from publicly available sources. If there are any concerns, please contact us and we will delete it in time.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Driving

[Figure 16]

Video Reference

Image

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

[Figure 27]

[Figure 28]

[Figure 29]

Driving

[Figure 30]

Video Reference

Image

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

###### Figure 7 Qualitative visualization for various shot types.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Driving Video

[Figure 45]

Reference Image

[Figure 46]

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

[Figure 57]

[Figure 58]

[Figure 59]

Driving Video

[Figure 60]

Reference Image

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

###### Figure 8 Qualitative visualization for reference character types.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Driving Video

[Figure 75]

Reference Image

[Figure 76]

[Figure 77]

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

Driving Video

[Figure 90]

Reference Image

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

###### Figure 9 Qualitative visualization for various driving character types.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Driving Video

[Figure 105]

Reference Image

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

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

Driving

[Figure 120]

Video Reference

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Image

###### Figure 10 Qualitative visualization for multi-person settings.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

- [2] Yogesh Balaji, Martin Renqiang Min, Bing Bai, Rama Chellappa, and Hans Peter Graf. Conditional gan with discriminative filter generation for text-to-video synthesis. volume 1, page 2. IJCAI, 2019.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [5] Di Chang, Yichun Shi, Quankai Gao, Hongyi Xu, Jessica Fu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. In Forty-first International Conference on Machine Learning, 2023.

- [6] Di Chang, Hongyi Xu, You Xie, Yipeng Gao, Zhengfei Kuang, Shengqu Cai, Chenxu Zhang, Guoxian Song, Chao Wang, Yichun Shi, et al. X-dyna: Expressive dynamic human image animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5499–5509, 2025.

- [7] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, et al. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025.

- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

- [9] Yanbo Ding, Xirui Hu, Zhizhi Guo, Chi Zhang, and Yali Wang. Mtvcrafter: 4d motion tokenization for open-world human image animation. arXiv preprint arXiv:2505.10238, 2025.

- [10] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.

- [11] Sivan Doveh, Shaked Perek, M Jehanzeb Mirza, Wei Lin, Amit Alfassy, Assaf Arbelle, Shimon Ullman, and Leonid Karlinsky. Towards multimodal in-context learning for vision and language models. In European Conference on Computer Vision, pages 250–267. Springer, 2024.

- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [13] Qijun Gan, Yi Ren, Chen Zhang, Zhenhui Ye, Pan Xie, Xiang Yin, Zehuan Yuan, Bingyue Peng, and Jianke Zhu. Humandit: Pose-guided diffusion transformer for long-form human motion video generation. arXiv preprint arXiv:2502.04847, 2025.

- [14] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [15] Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, Jun-Yan Zhu, and Jia-Bin Huang. On the content bias in fréchet video distance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7277–7288, 2024.

- [16] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Yufan Deng, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, et al. Video-bench: Human-aligned video generation benchmark. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18858–18868, 2025.

- [17] Jingxuan He, Busheng Su, and Finn Wong. Posegen: In-context lora finetuning for pose-controllable long human video generation. arXiv preprint arXiv:2508.05091, 2025.

- [18] Clyde Highmore. In-context learning in large language models: A comprehensive survey. 2024.
- [19] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117, 2023.

- [20] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Huanzhang Dou, Yupeng Shi, Yutong Feng, Chen Liang, Yu Liu, and Jingren Zhou. Group diffusion transformers are unsupervised multitask learners. arXiv preprint arXiv:2410.23775, 2024.

- [21] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024.

- [22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [23] Yasamin Jafarian and Hyun Soo Park. Learning high fidelity depths of dressed humans by watching social media dance videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12753–12762, 2021.

- [24] Kinam Kim, Junha Hyung, and Jaegul Choo. Temporal in-context fine-tuning for versatile control of video diffusion models. arXiv preprint arXiv:2506.00996, 2025.

- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [26] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- [27] Hongxiang Li, Yaowei Li, Yuhang Yang, Junjie Cao, Zhihong Zhu, Xuxin Cheng, and Long Chen. Dispose: Disentangling pose guidance for controllable human image animation. In The Thirteenth International Conference on Learning Representations, 2025.

- [28] Zhong-Yu Li, Ruoyi Du, Juncheng Yan, Le Zhuo, Zhen Li, Peng Gao, Zhanyu Ma, and Ming-Ming Cheng. Visualcloze: A universal image generation framework via visual in-context learning. arXiv preprint arXiv:2504.07960, 2025.

- [29] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, October 2015.

- [30] Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, Tianshu Hu, and Yongming Zhu. Dreamactor-m1: Holistic, expressive and robust human image animation with hybrid guidance. arXiv preprint arXiv:2504.01724, 2025.

- [31] Matteo Nulli, Anesa Ibrahimi, Avik Pal, Hoshe Lee, and Ivona Najdenkoska. In-context learning improves compositional understanding of vision-language models. arXiv preprint arXiv:2407.15487, 2024.

- [32] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633, 2021.

- [33] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025.

- [34] Team Seedance. Seedance 1.5 pro: A native audio-visual joint generation foundation model, 2025. URL https://arxiv.org/abs/2512.13507.
- [35] Guoxian Song, Hongyi Xu, Xiaochen Zhao, You Xie, Tianpei Gu, Zenan Li, Chenxu Zhang, and Linjie Luo. X-unimotion: Animating human images with expressive, unified and identity-agnostic motion latents. arXiv preprint arXiv:2508.09383, 2025.

- [36] Zeyi Sun, Ziyang Chu, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. X-prompt: Towards universal in-context image generation in auto-regressive vision language foundation models. arXiv preprint arXiv:2412.01824, 2024.

- [37] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image animation with enhanced motion representation. arXiv preprint arXiv:2410.10306, 2024.

- [38] Shuai Tan, Biao Gong, Zhuoxin Liu, Yan Wang, Xi Chen, Yifan Feng, and Hengshuang Zhao. Animate-x++: Universal character image animation with dynamic backgrounds. arXiv preprint arXiv:2508.09454, 2025.

- [39] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image animation with enhanced motion representation. ICLR 2025, 2025.

- [40] Shuyuan Tu, Qi Dai, Zihao Zhang, Sicheng Xie, Zhi-Qi Cheng, Chong Luo, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. Motionfollower: Editing video motion via score-guided diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12822–12831, 2025.

- [41] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

- [42] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [43] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. Unianimate-dit: Human image animation with large-scale video diffusion transformer. arXiv preprint arXiv:2504.11289, 2025.

- [44] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024.

- [45] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024.

- [46] Wenhao Yan, Sheng Ye, Zhuoyi Yang, Jiayan Teng, ZhenHui Dong, Kairui Wen, Xiaotao Gu, Yong-Jin Liu, and Jie Tang. Scail: Towards studio-grade character animation via in-context learning of 3d-consistent pose representations. arXiv preprint arXiv:2512.05905, 2025.

- [47] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4210–4220, 2023.

- [48] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [49] Polina Zablotskaia, Aliaksandr Siarohin, Bo Zhao, and Leonid Sigal. Dwnet: Dense warp-based network for pose-guided human video generation. arXiv preprint arXiv:1910.09139, 2019.

- [50] Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, and Yansong Tang. Flexiact: Towards flexible action control in heterogeneous scenarios. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025.

- [51] Yifu Zhang, Hao Yang, Yuqi Zhang, Yifei Hu, Fengda Zhu, Chuang Lin, Xiaofeng Mei, Yi Jiang, Zehuan Yuan, and Bingyue Peng. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025.

- [52] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680, 2024.

###### [53] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision (ECCV), 2024.

