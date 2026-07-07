# arXiv:2512.18181v3[cs.CV]7May2026

## MACE-Dance: Motion-Appearance Cascaded Experts for Music-Driven Dance Video Generation

KAIXING YANG∗, Renmin University of China, China JIASHU ZHU†, AMAP, Alibaba Group, China XULONG TANG, Malou Tech Inc, USA ZIQIAO PENG, Renmin University of China, China XIANGYUE ZHANG, Wuhan University, China PUWEI WANG‡, Renmin University of China, China JIAHONG WU‡, AMAP, Alibaba Group, China XIANGXIANG CHU, AMAP, Alibaba Group, China HONGYAN LIU‡, Tsinghua University, China JUN HE, Renmin University of China, China

showcase the power and beauty of human movement [Butterworth* 2004; Tseng et al. 2023]. In the era of the internet, dance videos have become highly prominent on platforms such as YouTube and TikTok. In parallel, rapid advances [Chen et al. 2025a,e; Lei et al. 2025; Yang et al. 2025a; Zhuo et al. 2023] in AI-generated content (AIGC) has created the technical preconditions for automating dance video creation, making it a timely and impactful research direction. Nevertheless, this task faces two key challenges: (1) generating dance motions that are kinematically plausible while artistically expressive; and (2) achieving high-fidelity visual appearance with strong spatiotemporal consistency.

With the rise of online dance-video platforms and rapid advances in AIGC, music-driven dance generation task has emerged as a compelling research direction. Despite substantial progress in related domains such as musicdriven 3D dance generation, pose-driven image animation, and audio-driven talking-head synthesis, these approaches are not readily transferable to this task due to fundamental mismatches in generation targets and constraints. Moreover, research on music-driven dance video generation remains limited and fails to capture the inherently 3D nature of dance, resulting in compromised motion quality and visual appearance. Accordingly, we present MACE-Dance, a music-driven dance video generation framework with cascaded Mixture-of-Experts (MoE). The Motion Expert performs music-to-3D motion enforcing kinematic plausibility and artistic expressiveness, while the Appearance Expert carries out motion-and-reference conditioned video synthesis, preserving visual identity with spatiotemporal coherence. Specifically, the Motion Expert adopts Diffusion Model with BiMamba-Transformer hybrid architecture and Guidance-Free Training (GFT) strategy, achieving state-of-the-art (SOTA) performance in 3D dance generation task; the Appearance Expert adopts a decoupled Kinematic–Aesthetic fine-tuning strategy, achieving state-of-the-art (SOTA) performance in the pose-driven image animation task. To better benchmark this task, we curate a large-scale dataset, and design a motion–appearance evaluation protocol. Based on them, MACE-Dance also achieves the state-of-the-art (SOTA) performance. Code is available at https://github.com/AMAP-ML/MACE-Dance.

Recent progress in dance generation has focused primarily on 3D dance [Li et al. 2024b, 2023; Tseng et al. 2023], with numerous strong methods emerging across model families-autoregressive [Siyao et al. 2022; Yang et al. 2025a,b], GAN-based [Huang and Liu 2021; Sun et al. 2019; Yang et al. 2024b], and diffusion-based [Li et al. 2024c,b; Tseng et al. 2023]. Although 2D dance videos can be rendered from 3D motion, such renderings typically lack realistic human–scene interactions and detailed appearance cues, resulting in visually suboptimal outputs [Yang et al. 2024d]. In contrast, human-centric image animation leverages a reference image along with various driving signals to generate videos. Traditionally, pose-driven image animation has achieved notable advances [Cheng et al. 2025; Hu 2024; Tan et al. 2024]. However, its utility for dance video generation is limited, as pose design—widely regarded as the most challenging and time-consuming step—still remains manual [Butterworth* 2004]. Similarly, audio-driven talking head generation has also achieved significant breakthroughs [Peng et al. 2025b, 2024, 2025c]. However, its direct transfer to dance video generation remains challenging, as it primarily focuses on relatively simple upper-body gesture rather than the complex full-body motion required in dance [Peng et al. 2023]. Research on music-driven dance video generation remains limited [Chen et al. 2025d; Tang et al. 2025; Wang et al. 2025b] and fails to capture the inherently 3D nature of dance, resulting in compromised motion quality and visual appearance.

CCS Concepts: • Applied computing → Arts and humanities; • Computing methodologies → Computer vision; • Human-centered computing;

1 Introduction

Dance is a vital part of human culture. Moving to the beat and melody, dancers both convey emotion and narrative intent and

∗Work done during an internship at AMap, Alibaba Group. †Project leader. ‡Corresponding authors.

Authors’ Contact Information: Kaixing Yang, yangkaixing@ruc.edu.cn, Renmin University of China, Beijing, China; Jiashu Zhu, zhujiashu.zjs@alibaba-inc.com, AMAP, Alibaba Group, Beijing, China; Xulong Tang, xulong.tang@maloutech.com, Malou Tech Inc, Richardson, Texas, USA; Ziqiao Peng, pengziqiao@ruc.edu.cn, Renmin University of China, Beijing, China; Xiangyue Zhang, xiangyuezhang@whu.edu.cn, Wuhan University, Wuhan, China; Puwei Wang, wangpuwei@ruc.edu.cn, Renmin University of China, China; Jiahong Wu, hongxi.wjh@alibaba-inc.com, AMAP, Alibaba Group, Beijing, China; Xiangxiang Chu, cxxgtxy@gmail.com, AMAP, Alibaba Group, Beijing, China; Hongyan Liu, liuhy@sem.tsinghua.edu.cn, Tsinghua University, Beijing, China; Jun He, hejun@ruc.edu.cn, Renmin University of China, Beijing, China.

Accordingly, we present MACE-Dance, a music-driven dance videogenerationframeworkwithcascaded mixture-of-experts (MoE), as shown in Fig. 1. The Motion Expert performs music-to-3D motion enforcing kinematic plausibility and artistic expressiveness, while

[Figure 1]

The performance showcases a graceful and flowing Eastern folk dance

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

This is an oldschool and groovy hip-hop piece .

[Figure 8]

MACE-Dance

[Figure 9]

[Figure 10]

Sharp hits and electric rhythm define this powerful popping showcase.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Appearance Expert

Motion Expert 3D Motion

This dance is a stylish and energetic K-pop performance.

[Figure 16]

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

(a) Overview (b) Cases

- Fig. 1. Leveraging the synergistic collaboration among the cascaded experts, MACE-Dance can generate diverse dance videos that not only exhibit kinematically plausible and artistically expressive motion, but also maintain spatiotemporal coherent appearance.

the Appearance Expert carries out motion-and-reference conditioned video synthesis, preserving visual identity with spatiotemporal coherence. Notably, MACE-Dance adopts 3D SMPL [Loper et al. 2023] parameters rather than 2D keypoints as the intermediate representation, as 3D provides view-invariant and physically consistent supervision, while 2D projections introduce irreversible information loss and viewpoint ambiguity. (1) Motion Expert. Motion Expert adopts Diffusion Model with BiMamba-Transfomer hybrid architecture. The bidirectional Mamba [Gu and Dao 2023] captures intra-modal local dependencies in music or dance, while the Transformer [Vaswani 2017] models cross-modal global context. Owing to this architecture, the Motion Expert generates entire sequence in non-autoregressive manner during inference, not only improving generation efficiency, but also avoiding the exposure bias problem in autoregressive [Yang et al. 2025b] and inpainting-based [Tseng

- et al. 2023] methods. To enhance generation stability and accelerate inference, we employ guidance-free training (GFT [Chen et al. 2025b]) instead of conventional classifier-free guidance (CFG [Ho and Salimans 2022]), enhancing the physical plausibility and artistic expressiveness for the generated dance. (2) Appearance Expert. Wan-Animate [Cheng et al. 2025] has recently garnered substantial attention in both industry and academia. However, directly applying it to dance video generation yields limited effectiveness, as dance videos exhibit significantly more complex patterns than general videos. Thus, the Appearance Expert adopts a decoupled Kinematic–Aesthetic two-stage fine-tuning strategy to achieve highfidelity appearance synthesis. In Kinematic stage, it fine-tunes the Body Adapter to strengthen kinematic conditioning and motion adherence. In Aesthetic stage, it attaches a LoRA [Hu et al. 2022] branch to each DiT block and fine-tunes for aesthetic refinement, enhancing texture fidelity and stylistic consistency.

To better benchmark music-driven dance video generation task, we curate a large-scale dataset and design a motion–appearance evaluation protocol. Firstly, we curate a large-scale dance video dataset, named MA-Data, comprising 70k clips of 5–10 seconds each (totaling 116 hours), spanning over 20 dance genres. The dataset consists of two complementary sources: (1) 3D-rendered data (motion-centric): Derived from FineDance [Li et al. 2023]—the largest 3D dance dataset recorded by professional dancers—we render front-view videos and extract random 5–10 s segments via a sliding window, yielding 20k clips (28 h). This subset emphasizes professional dance motion rather than visual appearance. (2) In-the-wild internet data (appearancecentric): Collected from high-engagement videos on platforms such as TikTok and YouTube, using the same sliding-window strategy to obtain 50k 5-10 s clips (88 h). This subset emphasizes visual appearance, while motions are relatively unprofessional. Secondly, we introduce a motion–appearance evaluation protocol. For the motion dimension, we assess the fidelity, diversity, and synchronization [Li et al. 2021, 2023] from Human-Kinematics perspective based on the 2D keypoints extracted by ViTPose [Xu et al. 2022]. For the appearance dimension, we adopt VBench [Huang et al. 2024]—a widely used benchmark in video generation—and select a set of dance-specific metrics.

In conclusion, our contributions are as follows: (1) To better benchmark the music-driven dance video generation task, we curate a large-scaledatasetnamedMA-Data,alongwithamotion–appearance evaluation protocol. (2) Based on them, we introduce MACE-Dance, a music-driven dance video generation framework with cascaded experts, achieving SOTA performance. (3) The Motion Expert adopts Diffusion Model with BiMamba-Transformer hybrid architecture and Guidance-Free Training strategy, achieving SOTA performance on the FineDance dataset in music-driven 3D dance generation task. (4) Appearance Expert adopts a decoupled Kinematic-Aesthetic

fine-tuning strategy, achieving SOTA performance on the MA-Data dataset in the pose-driven image animation task.

2 Related Work

- 2.1 Music-Driven 3D Dance Generation

Music and dance are deeply intertwined, and recent progress in music-to-dancegenerationhaslargely centered on 3D motion. Broadly, existing methods fall into three families: GAN-based, autoregressive, and diffusion-based models. 1) GAN-based models. Generators synthesize motion from music while discriminators provide adversarial feedback. Examples include CoheDancers [Yang et al. 2024b] and DeepDance [Sun et al. 2019]. 2) Autoregressive models. These methods typically adopt a two-stage pipeline: curating choreographic units by VQ-VAE [van den Oord et al. 2017] or FSQ [Mentzer et al. 2023], followed by autoregressive modeling of music-conditioned distributions over these units [Li et al. 2024a; Yang et al. 2024a, 2026]. Works such as Bailando [Siyao et al. 2022], Bailando++ [Siyao et al. 2023], and MEGADance [Yang et al. 2025b] fall into this paradigm. 3) Diffusion-based models. These methods corrupt motion with noise and train denoising networks to iteratively recover sequences conditioned on music [Yang et al. 2025c], enabling diverse and temporally coherent dances. Representative works include EDGE [Tseng et al. 2023], FineNet [Li et al. 2023], Lodge [Li et al. 2024b], Lodge++ [Li et al. 2024c], and GCDance [Liu et al. 2025]. Despite substantial progress, 3D dance generation only focuses on motion generation and underemphasizes visual appearance—an essential aspect of dance as an art form. Although 2D dance videos can be rendered from 3D motion, the outputs typically lack realistic human–scene interactions and high-fidelity human textures.

- 2.2 Human-Centric Image Animation

In contrast, human-centric image animation leverages a reference image along with various driving signals to generate videos that exhibit high-quality visual appearance, making it a promising direction for dance video generation. Firstly, pose-driven image animation utilizes 2D keypoints to generate motion videos, achieving notable advances [Cheng et al. 2025; Hu 2024; Tan et al. 2024], including Animate-X [Tan et al. 2024], Animate Anyone [Hu 2024] and Wan-Animate [Cheng et al. 2025]. However, its utility for dance video generation is limited, as pose design—widely regarded as the most challenging and time-consuming step— still remains manual [Butterworth* 2004]. Secondly, speech-driven image animation employs audio features to generate talking head videos, also achieving significant breakthroughs [Peng et al. 2025b, 2024, 2025c], such as SyncTalk [Peng et al. 2024], OmniSync [Peng et al. 2025c] and Hallo2 [Cui et al. 2024]. However, its direct transfer to dance video generation remains challenging, as these methods primarily focus on relatively simple upper-body gestures rather than the complex full-body motion required in dance [Peng et al. 2023; Zhang et al. 2025b,c,d]. Finally, research on music-driven dance video generation remains limited. DabFusion [Wang et al. 2025b] introduces an endto-end Diffusion-based method, but the generated videos exhibit blurry foreground subjects and backgrounds, thereby degrading visual fidelity. X-Dancer [Chen et al. 2025d], STG-Mamba [Tang et al.

2025] and ChoreoMuse [Wang et al. 2025a] predict 2D keypoints from music and then drives image animation with these keypoints. However, they remain limited in handling limb occlusions and complex full-body locomotion in dance videos. In conclusion, existing works for dance video generation still fails to capture the inherently 3D nature of dance, resulting in compromised motion quality and visual appearance. Thus, we propose MACE-Dance, a cascaded expert framework that synergistically integrates motion and appearance generation, producing kinematically plausible and artistically expressive motion while maintaining spatiotemporally coherent visual appearance.

3 Methodology 3.1 Overview

Given a music 𝑀 ∈ 𝑅𝑇×𝐶𝑚 and reference image 𝐼 ∈ 𝑅𝐻×𝑊 ×3, our objective is to synthesize the corresponding dance videos 𝐷 ∈ 𝑅𝑇×𝐻×𝑊 ×3 with high-quality visual appearance and human motion. Overall, MACE-Dance is with cascaded mixture-of-experts (MoE), as shown in Fig. 2. The Motion Expert (ME) transfers music sequence 𝑀 into 3D motion sequence 𝑋 ∈ 𝑅𝑇×𝐶𝑥 , enforcing kinematic plausibility and artistic expressiveness. The Appearance Expert (AE) utilizes the above 3D motion sequence 𝑋 and reference image 𝐼 to drive video synthesis, preserving visual identity with spatiotemporal coherence. This task decoupling significantly reduces the complexity of learning a direct music-to-video mapping by isolating motion semantics from visual appearance. Moreover, the explicit 3D motion representation suppresses spurious cross-modal correlations and provides an interpretable intermediate interface for robust and controllable video synthesis.

Unlike prior works [Chen et al. 2025d; Tang et al. 2025] that adopt

- 2D keypoints as the intermediate representation, we instead use
- 3D motion as the bridge between the two experts for three reasons.

(1) Richer spatial fidelity. 3D motion preserves full-body geometric structure, including global translation and orientation, which is essential for dance phrases with large-amplitude locomotion and complex spatial choreography, whereas 2D projections inevitably discard depth and global movement information. (2) Cleaner supervision. 3D representation disentangles pose from camera viewpoint and subject-specific appearance, providing a more stable and generalizable signal for learning the music-to-motion correspondence, while 2D keypoints are entangled with perspective and body proportions. (3) Better robustness. 3D motion is inherently more robust to self-occlusion and viewpoint variation, whereas 2D poses often suffer from missing joints, depth ambiguity, and inconsistent observations. Additionally, we adopt SMPL [Loper et al. 2023] as the representation of the 3D motion sequence 𝑋 for two reasons. (1) Prior focus on body motion. Most existing 3D dance generation methods primarily model body-level motion rather than detailed hand articulation. In our setting, body-level motion alone is sufficient to produce strong visual results, as also evidenced by our demo videos. (2) Extensibility. Our framework can be readily extended to richer motion representations, such as SMPL-X, when suitable data become available.

[Figure 27]

[Figure 28]

[Figure 29]

###### : Trainable : Frozen

[Figure 30]

[Figure 31]

~ : Emb + :Add

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

BiMamba

DiTBlockLoRA

DiTBlockLoRA

Body Adapter

DiTBlockLoRA

Librosa

[Figure 36]

[Figure 37]

[Figure 38]

~

+

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

~

[Figure 44]

[Figure 45]

[Figure 46]

Projector

Fusion

[Figure 47]

[Figure 48]

[Figure 49]

Transformer

[Figure 50]

[Figure 51]

BiMamba

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

FiLM

FiLM

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Noise

(a) Motion Expert (b) Appearance Expert

- Fig. 2. Overview of MACE-Dance.Leveraging the cascaded Mixture-of-Experts (MoE) design, the Motion Expert generates kinematically plausible and artistically expressive 3D motion 𝑋 conditioned on the music 𝑀, and the Appearance Expert then animates the reference image 𝐼 with the 3D motion 𝑋, yielding the dance video 𝐷 that exhibits spatiotemporally coherent appearance. Thanks to the Guidance-Free Training (GFT) strategy, 𝛽 ∈ [0, 1] can serve as a controllable knob that governs the diversity of the generated motion.
- 3.2 Motion Expert

where ∅ denotes the unconditional setting, and sg represents the stop-gradient operation. 𝛽 serves as a temperature parameter that is also provided to the model 𝜃 as an additional conditioning input. During training, 𝛽 and 𝑡 are sampled randomly from𝑈 (0, 1) and the integer set {0, 1, . . .,𝑇}, respectively. Moreover, we further apply the reconstruction loss, 3D joint loss, velocity loss, foot contact loss, to enhance physical plausibility and aesthetic expressiveness:

- 3.2.1 Generative Strategy.

Diffusion. DDPM [Ho et al. 2020] defines diffusion as a Markov

noising process with latents {𝑧𝑡}𝑇𝑡=0 that follow a forward noising process 𝑞(𝑧𝑡|𝑥), where 𝑥 ∼ 𝑝(𝑥) is drawn from the 3D dance data distribution. The forward noising process is defined as:

Lrec = E ∥𝑥𝛽 − 𝑥∥22 , Ljoint = E ∥𝐹𝐾(𝑥𝛽) − 𝐹𝐾(𝑥)∥22 ,

√𝛼¯𝑡𝑥, (1 − 𝛼¯𝑡)𝐼), (1)

𝑞(𝑧𝑡|𝑥) ∼ N(

(4)

where 𝛼¯𝑡 ∈ (0, 1) are constants which follow a monotonically decreasing schedule such that when 𝛼¯𝑡 approaches 0. Timestep𝑇 are commonly set to 1000, and 𝑧𝑇 ∼ N(0,𝐼). With paired music conditioning 𝑐, we can reverse the forward diffusion process by learning to estimate 𝑥ˆ𝜃 (𝑧𝑡,𝑡,𝑐) ≈ 𝑥 with model parameters 𝜃 for all 𝑡. We can optimize 𝜃 by the naive reconstruction loss in Diffusion Model [Ho et al. 2020]:

Lvel = E ∥𝐹𝐾(𝑥𝛽)′ − 𝐹𝐾(𝑥)′∥22 Lfoot = E ∥𝐹𝐾(𝑥𝛽)′ · bˆ∥22 ,

where 𝐹𝐾(·) denotes the forward kinematic function that converts joint angles into joint positions, and bˆ is the model’s own prediction of the binary foot contact label’s portion of the pose. Our overall training loss L is the weighted sum of the above losses, where the weights 𝜆 were chosen to balance the magnitudes of the losses:

LDM = E ∥𝑥ˆ𝜃 (𝑧𝑡,𝑡,𝑐) − 𝑥∥22 . (2)

L = 𝜆recLrec + 𝜆jointLjoint + 𝜆velLvel + 𝜆footLfoot. (5)

Guidance-Free Training. Conventional classifier-free guidance (CFG [Ho and Salimans 2022]) modifies the sampling distribution only at inference time by combining conditional and unconditional predictions, which can introduce distribution mismatch and insufficient optimization toward the guided target distribution. In contrast, Guidance-Free Training (GFT [Chen et al. 2025b]) retains the same maximum-likelihood training objective as CFG but adopts a different parameterization that enables a single model to implicitly represent temperature-controlled sampling behavior during training, thereby mitigating distribution mismatch and yielding more stable and consistent high-fidelity generation. Accordingly, we establish 𝑥𝛽 as the new optimization target for our model 𝜃:

Inference. At each of the denoising timesteps 𝑡, Motion Expert predicts the denoised sample and noises it back to timestep 𝑡 − 1: 𝑧ˆ𝑡−1 ∼ 𝑞(𝑥ˆ𝜃 (𝑧𝑡,𝑡,𝑐, 𝛽),𝑡 − 1), terminating when it reaches 𝑡 = 0. We utilize Denoising Diffusion Implicit Models (DDIM [Song et al. 2021]) to accelerate the sampling procedure. Values of 𝛽 near 0 favor high fidelity, while values near 1 favor high diversity. Thus, 𝛽 can also be regarded as a control signal, and we set its value to 0.75. Notably, GFT inherently achieves theoretically double the generation efficiency compared to conventional CFG, as it only requires a single conditional computation per step, eliminating the need for simultaneous conditional and unconditional predictions.

𝑥𝛽 = 𝛽𝑥ˆ𝜃 (𝑧𝑡,𝑡,𝑐, 𝛽) + (1 − 𝛽)sg[𝑥ˆ𝜃 (𝑧𝑡,𝑡, ∅, 1)] (3)

3.2.2 Model Architecture.

Overview. Motion Expert adopts a BiMamba–Transformer hybrid backbone, thereby enabling the generation of temporally coherent and musically aligned dance motions. BiMamba captures intra-modal local dependencies in music or dance, while the Transformer models cross-modal global context. As shown in Fig. 2, the architecture details are as follows: Firstly, our model conditions the generator on the Librosa [McFee et al. 2015]-extracted music features from 𝑀 as [Li et al. 2021], which are then processed by an 𝐿𝑚-layer BiMamba to capture intra-modal temporal dynamics. Secondly, the diffusion time step 𝑡 and temperature parameter 𝛽 are encoded as sinusoidal embeddings and fused by element-wise addition to yield a 𝑡-𝛽 embedding used throughout the generator. Third, the dance generator consists of 𝐿𝑑 stacked blocks. In each block: (1) the current state 𝑧𝑡 is first passed through a BiMamba to model intra-modal local dependencies; (2) FiLM [Perez et al. 2018] is applied to modulate the features with the fused 𝑡-𝛽 embedding; (3) a Transformer performs cross-modal attention over the music encoding to integrate global musical context, and subsequently passes the result through a feed-forward network; and (4) a second FiLM [Perez et al. 2018] further reinforces the 𝑡-𝛽 conditioning. Fi-

nally, the generator outputs the 3D motion sequence 𝑥ˆ𝜃 (𝑧𝑡,𝑡,𝑐, 𝛽) (i.e. 𝑋 in Sec. 3.1 Overview), represented as SMPL [Loper et al. 2023] parameters. Owing to this architecture, the Motion Expert generates the entire sequence in a non-autoregressive manner during inference, not only improving generation efficiency but also avoiding the exposure-bias problem in autoregressive [Yang et al. 2025b] and inpainting-based [Tseng et al. 2023] methods.

Intra-Modal Local-Dependency. While the Transformer excels at temporal modeling, it is inherently position-invariant and captures sequence order only through positional encodings [Vaswani 2017], which limits its deep understanding of local dependencies. In contrast, music-to-dance generation demands strong local continuity between movements. Owing to its inherent sequential inductive bias, Mamba [Gu and Dao 2023] has demonstrated strong performance in modeling fine-grained local dependencies [Fu et al. 2024; Xu et al. 2024a]. Moreover, Bidirectional Mamba processes inputs in both forward and backward directions, enabling wider representations and deeper understanding of music and dance. Specifically, the Selective State Space Model (Mamba) integrates a selection mechanism and a scan module (S6) [Gu and Dao 2023] to dynamically emphasize salient input segments for efficient sequence modeling. Unlike traditional SSMs with time-invariant parameters, Mamba generates input-dependent 𝐴¯𝑡,𝐵¯𝑡,𝐶𝑡 through fully connected layers, enhancing generalization. For each time step 𝑡, the input 𝑥𝑡, hidden state ℎ𝑡, and output 𝑦𝑡 evolve as:

ℎ𝑡 = 𝐴¯𝑡ℎ𝑡−1 + 𝐵¯𝑡𝑥𝑡, 𝑦𝑡 = 𝐶𝑡ℎ𝑡, (6)

where 𝐴¯𝑡,𝐵¯𝑡,𝐶𝑡 are dynamically updated, and the state transitions become:

𝐴¯ = exp(Δ𝐴), 𝐵¯ = (Δ𝐴)−1(exp(Δ𝐴) − 𝐼) · Δ𝐵, (7)

where Δ is the discretization step size, 𝐴 is the continuous-time state transition matrix, 𝐵 is the input projection matrix, and𝐶 is the output projection matrix.

Cross-Modal Global-Context. While BiMamba [Gu and Dao 2023] excels at capturing local dependencies, it is less effective at modeling cross-modal global interactions. Thus, we employ a Transformer [Vaswani 2017] module after BiMamba in each denoising block, which is crucial for aligning the overall dance structure with long-term musical phrasing. This block consists of a crossattention layer followed by a feed-forward network (FFN). In the cross-attention layer, motion features serve as queries, while music features provide keys and values:

𝑄𝑑 · 𝐾𝑚𝑇

𝑉𝑚. (8)

Attention = softmax

##### √

𝐶

In this way, the two components play complementary roles: BiMamba stabilizes short-range intra-modal dynamics, while the Transformer injects cross-modal global musical context to align the generated motion with the overall rhythm and phrase structure.

3.3 Appearance Expert

Wan-Animate [Cheng et al. 2025] has recently garnered substantial attention in both industry and academia. However, it is designed for general-purpose motion synthesis; direct transfer to dance video generation is suboptimal due to the domain gap and the richer spatiotemporal complexity of dance—namely intricate whole-body coordination and dynamic camera choreography. Accordingly, the Appearance Expert adopts a decoupled Kinematic–Aesthetic finetuning strategy to achieve high-fidelity appearance synthesis for dance videos. Specifically, the Kinematic Stage fine-tunes only the Body Adapter while freezing the remaining components, whereas the Aesthetic Stage fine-tunes only the LoRA parameters while keeping the rest of the network fixed.

Model Architecture. As illustrated in Fig. 2, the architecture of our Appearance Expert is built upon the Wan-Animate [Cheng et al. 2025], which takes a reference image 𝐼 for appearance and a 3D motion sequence 𝑋 for motion guidance. The motion sequence 𝑋 is first projected to 2D keypoints, which are then encoded by a Body Adapter to yield motion features. These features are subsequently fused with the latent extracted from the reference image 𝐼. The resulting latent is processed by a backbone of stacked DiT blocks, where lightweight LoRA adapters are integrated into each block. The facial processing pipeline remains identical to that of Wan-Animate and is therefore omitted for clarity.

Projector. We introduce a 3D-to-2D Motion Projector to convert the SMPL sequence generated by the Motion Expert into the 2D pose format required by Wan-Animate. For each frame, we first transform the SMPL parameters into a 3D mesh and render it with pyrender under a fixed frontal-view camera, then apply ViTPose [Xu et al. 2022] to extract the corresponding 2D keypoints. In this way, the projector preserves the benefits of 3D motion modeling while enabling seamless integration with the downstream Appearance Expert.

Kinematic Stage. In dance, body pose is paramount. The original Wan-Animate prioritizes facial cues, allocating a dedicated cross-attention branch to the face while fusing body signals only via additive injection. We therefore strengthen kinematic conditioning by fine-tuning the Body Adapter in the Kinematic Stage to reweight

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

|This music is an elegant and rhythmic Eastern folk piece.|
|---|

|Driven by funk grooves, this music bursts with power.|
|---|

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

Hallo2

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

EDGE

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

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Lodge

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

MEGA

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

WANS2V Echo mimicV3

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

Ours

Fig. 3. Qualitative comparison with SOTAs across reference image domains (real-person vs. anime-character) and music genres (Eastern Folk vs. Popping) in the music-driven dance video generation task.

and calibrate body features across scales, thereby enforcing motion adherence without altering the backbone. We intentionally do not introduce an additional body cross-attention branch because (i) it disturbs the pretrained inductive bias and can compete with the facial cross-attention, causing feature entanglement, (ii) it adds substantial memory/latency overhead and training instability on long, fast dance sequences.

Aesthetic Stage. To refine visual quality without disturbing motion control, we freeze the kinematic pathways and attach lightweight LoRA adapters to the attention (query/key/value/output) and feed-forward projections in each DiT block of Wan-Animate. These rank-r adapters enable parameter-efficient specialization toward dance-specific aesthetics—sharpening textures (skin, hair, fabric), stabilizing clothing and accessories, and handling rich camera choreography (pans, zooms, handheld motion)—while preserving pretrained content priors. Specifically, LoRA is an effective technique for adapting large pre-trained models to down-streaming tasks with few training-able parameters. To achieve this goal, LoRA introduces a low-rank decomposition-based method to the model’s weight matrix, enabling efficient adaptation to new tasks while maintaining the model’s original capabilities. Given the weight matrix𝑊0 ∈ R𝑚×𝑛 of the original pre-trained model, LoRA [Hu et al. 2022] uses two low-rank matrices 𝐴 ∈ R𝑚×𝑟 (𝑟 ≪ 𝑚) and 𝐵 ∈ R𝑟×𝑛 (𝑟 ≪ 𝑛) to shift the trained distribution according to the new data training. Thanks to the low-rank matrix in 𝐴 and 𝐵, LoRA updates the model more efficiently than the full rank matrix and shows comparable results with full training. Formally, the new weight matrix

𝑊 can be represented as: 𝑊 =𝑊0 + Δ𝑊 =𝑊0 + 𝐴𝐵. (9)

4 Experiment 4.1 Dataset

To support music-driven dance video generation task, we curate a large-scale dance video dataset, named MA-Data. It comprises 70k clips of 5–10 seconds each (totaling 116 hours), and spans over 20 distinct dance genres, such as Jazz, Latin, Eastern Folk. MA-Data consists of two complementary sources. (1) 3D-rendered data (motion-centric). This subset is derived from FineDance [Li et al. 2023], the largest 3D dance dataset recorded by professional dancers, and emphasizes professional dance motion rather than visual appearance. Specifically, we first retarget the motion sequence to a character model, then render front-view videos from the 3D character, and extract random 5–10 s segments with a sliding-window strategy for data augmentation, yielding 20k clips ( 28 hours). (2) In-the-wild internet data (appearance-centric). This subset is curated from high-engagement creators on platforms such as TikTok and YouTube, emphasizing visual appearance; motions typically prioritize entertainment value over technical rigor. As raw crawls include many samples misaligned with our task, we apply a multistage cleaning pipeline: (i) perform shot boundary detection with TransNet V2 [Soucek and Lokoc 2024], segment accordingly, and discard segments shorter than 5 s; (ii) remove near-static clips using an optical-flow magnitude threshold; (iii) enforce a single-performer

Table 1. Quantitative comparison with SOTAs on the MA-Data dataset in Music-Driven Dance Video Generation task.

Appearance Motion IQ↑ AQ↑ SC↑ BC↑ MS↑ TF↑ FID𝑘↓ FID𝑔↓ DIV𝑘↑ DIV𝑔↑ BAS↑

Ground Truth 67.12 53.51 91.86 92.97 98.20 96.88 – – 9.24 5.31 0.526 Hallo2 [Cui et al. 2024] [ICLR’25] 62.64 50.79 92.48 93.84 98.30 96.56 16.55 1.29 8.11 5.47 0.505 WAN-S2V [Gao et al. 2025] [Arxiv’25] 64.10 50.20 92.30 93.40 98.20 96.70 18.90 1.45 7.60 5.44 0.485 Echomimic-V3 [Meng et al. 2025] [AAAI’26] 63.20 49.00 91.90 93.10 98.05 96.40 19.60 1.32 7.20 4.60 0.460 EDGE [Li et al. 2023] [CVPR’23] 63.05 49.70 91.79 93.30 98.64 97.10 21.77 1.39 9.08 5.74 0.498 Lodge [Li et al. 2024b] [CVPR’24] 63.69 49.22 91.67 92.98 98.46 97.05 18.73 1.49 8.87 5.71 0.474 MEGA [Yang et al. 2025b] [NeurIPS’25] 66.14 49.89 92.95 94.13 97.45 96.32 18.98 1.65 8.78 5.59 0.513

MACE-Dance 65.35 51.79 93.97 94.57 98.46 97.10 16.46 0.28 9.74 6.34 0.523

Table 2. Quantitative comparison with SOTAs on the FineDance dataset in Music-Driven 3D Dance Generation task.

FID𝑘↓ FID𝑔↓ FSR↓ DIV𝑘↑ DIV𝑔↑ BAS↑ FPS↑

|Ground Truth FACT [Li et al. 2021] [ICCV’21] MNET [Kim et al. 2022] [CVPR’22] Bailando [Siyao et al. 2022] [CVPR’22] EDGE [Tseng et al. 2023] [CVPR’23] Lodge [Li et al. 2024b] [CVPR’24] MEGA [Yang et al. 2025b] [NeurIPS’25]<br><br>|– – 0.216 9.94 7.54 0.201 – 113.38 97.05 0.284 3.36 6.37 0.183 29 104.71 90.31 0.394 3.12 6.14 0.186 26<br><br>82.81 28.17 0.188 7.74 6.25 0.202 188 94.34 50.38 0.200 8.13 6.45 0.212 119 50.00 35.52 0.028 5.67 4.96 0.226 224 50.00 13.02 0.243 6.23 6.27 0.226 238<br><br>|
|---|---|
|GFT → CFG BiMamba → Mamba BiMamba → Transformer Motion Expert (Full)|25.54 36.31 0.295 11.57 7.03 0.223 475<br><br>65.10 51.74 0.35 9.59 7.66 0.224 1044<br><br>104.93 114.42 0.12 7.90 4.22 0.266 683 17.83 25.09 0.210 10.30 8.09 0.229 770<br><br>|

constraint via ViTPose [Xu et al. 2022] by discarding clips that contain multiple people or exhibit little to no human motion; and (iv) split long videos into 5–10 s clips with a sliding window and random offsets. The final set comprises 50k clips ( 88 hours). Finally, we collect an additional 200 5-second clips to construct the test set, with high engagement on TikTok and across multiple dance genres.

- 4.2 Evaluation

The keychallengesofmusic-drivendance video generation are [Yang et al. 2024d; Zhang et al. 2025a]: (1) generating dance motions that are kinematically plausible while artistically expressive; and (2) achieving high-fidelity visual appearance with strong spatiotemporal consistency. Inspired by this, we introduce a motion–appearance evaluation protocol. (1) Motion dimension. We extract 2D keypoint sequences using ViTPose [Xu et al. 2022] from the dance videos and evaluate from a Human-Kinematics perspective. To evaluate the fidelity and diversity, we report FID and DIV across two feature spaces [Li et al. 2021; Siyao et al. 2022]: (1) kinetic (k), capturing motion dynamics, and (2) geometric (g), encoding spatial joint relations. To measure music–motion synchronization, we utilize the Beat Alignment Score (BAS) [Li et al. 2021, 2024b]. (2) Appearance dimension. Inspired by [Chen et al. 2025c; Li et al. 2025; Ling et al. 2025], we adopt VBench [Huang et al. 2024]—a widely used benchmark in video generation—and select a set of dance-specific metrics. Our evaluation includes imaging quality (IQ), aesthetic quality (AQ), subject consistency (SC), background consistency (BC), motion smoothness (MS), temporal flickering (TF).

Table 3. Quantitative comparison with SOTAs on the MA-Data dataset in Pose-Driven Image Animation task.

FVD↓ SSIM↑ LPIPS↓ PSNR↑

|Animate-Anyone [Hu 2024] [CVPR’24] Magic-Animate [Xu et al. 2024b] [CVPR’24] Wan-Animate [Cheng et al. 2025] [Arxiv’25]<br><br>|515.26 0.648 0.091 19.65 1032.06 0.311 0.207 14.00<br><br>332.82 0.707 0.078 21.11<br><br>|
|---|---|
|w/o. Kinematic stage w/o. Aesthetic stage Appearance Expert<br><br>|328.91 0.596 0.107 18.69 445.93 0.563 0.121 17.89 274.94 0.739 0.066 22.40|

4.3 Comparison

- 4.3.1 Music-Driven Dance Video Generation. As there is currently no open-source implementation for Music-Driven Dance Video Generation, we compare MACE-Dance against two baseline families on the MA-Data dataset: (1) 3D dance generation methods pipelined with Wan-Animate, including EDGE [Tseng et al. 2023], Lodge [Li et al. 2024b], and MEGA [Yang et al. 2025b]; (2) General human-motion video generation methods. We perform inference using the pretrained EchoMimic-V3 [Meng et al. 2025] and WANS2V [Gao et al. 2025] models, which supports general human motion generation. In addition, we adapt the Hallo2 [Cui et al. 2024] by replacing facial masks with full-body masks, and then fine-tune it on the MA-Data dataset. As shown in Tab. 1, our proposed MACEDance demonstrates state-of-the-art (SOTA) performance in both appearance and motion quality. Specifically, for the motion aspect, MACE-Dance achieves the best results across all metrics (𝐹𝐼𝐷𝑘 = 16.46, 𝐹𝐼𝐷𝑔 = 0.28,𝐷𝐼𝑉𝑘 = 9.74,𝐷𝐼𝑉𝑔 = 6.34,𝐵𝐴𝑆 = 0.523); for the appearance aspect, it also attains best performance on most metrics, with scores of 𝐼𝑄 = 65.35,𝐴𝑄 = 51.79,𝑆𝐶 = 93.97,𝐵𝐶 = 94.57,𝑀𝑆 = 98.46, and 𝑇𝐹 = 97.10. By effectively decoupling the task into 3D Dance Generation and Pose-Driven Image Animation, and leveraging the strong performance of the Motion Expert and Appearance Expert on their respective sub-tasks, MACE-Dance delivers exceptional generation quality. Note: User Study can be found in the supplementary material Sec. 2.
- 4.3.2 Music-Driven 3D Dance Generation. Music-Driven 3D Dance Generation is a canonical task and determines the motion quality of MACE-Dance. We compare the Motion Expert against FACT [Li et al. 2021], MNET [Kim et al. 2022], Bailando [Siyao et al. 2022],

Uyghur

Dunhuang

Dai

K-Pop

Popping

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

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

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Fig. 4. MACE-Dance generates high-quality dance videos across diverse dance genres.

EDGE [Tseng et al. 2023], Lodge [Li et al. 2024b], and MEGA [Yang et al. 2025b] on the FineDance dataset with metrics following [Li et al. 2024b]. As shown in Tab. 2, the Motion Expert attains overall state-of-the-art (SOTA) performance. Specifically, it achieves the best 𝐹𝐼𝐷𝑘 = 17.83 and a competitive 𝐹𝐼𝐷𝑔 = 25.09, indicating high fidelity; the best 𝐷𝐼𝑉𝑘 = 10.30 and 𝐷𝐼𝑉𝑔 = 8.09, indicating strong diversity; a competitive FSR, supporting physical plausibility; the best 𝐵𝐴𝑆 = 0.229, demonstrating superior audio-motion synchronization; and a substantially higher 𝐹𝑃𝑆 = 770, evidencing excellent generation efficiency. These advances primarily stem from: (1) adopting a Diffusion Model with a BiMamba-Transformer hybrid architecture, enabling high-quality long motion sequences in a non-autoregressive manner; and (2) a Guidance-Free Training (GFT) strategy that improves generation quality without requiring dual-pass inference (conditioned + unconditioned). Note: Qualitative Comparison can be found in the supplementary material Sec. 3.2.

- 4.3.3 Pose-Driven Image Animation. Pose-Driven Image Animation is likewise a canonical task and governs the appearance quality of MACE-Dance. We compare the Appearance Expert against Animate-Anyone [Hu 2024], Magic-Animate [Xu et al. 2024b], and Wan-Animate [Cheng et al. 2025] on the MA-Data dataset with metrics following [Cheng et al. 2025]. The Appearance Expert achieves state-of-the-art (SOTA) performance on all metrics (Tab. 3, 𝐹𝑉𝐷 = 274.94, 𝑆𝑆𝐼𝑀 = 0.739, 𝐿𝑃𝐼𝑃𝑆 = 0.066, 𝑃𝑆𝑁𝑅 = 22.40). Its strong video synthesis quality primarily benefits from Wan-Animate (Baseline)’s powerful cross-modal understanding and the KinematicAesthetic decoupled fine-tuning strategy. Note: Qualitative Comparison can be found in the supplementary material Sec. 3.3.

- 4.4 Qualitative Analysis

- 4.4.1 Effect Comparison. We also present a qualitative comparison against other methods across reference-image domains (real

person vs. anime character) and music genres (elegant and rhythmically rich Eastern Folk vs. powerful and funk-inspired Popping) for the music-driven dance video generation task. As shown in Fig. 3, Hallo2 exhibits significant blurring in human details and noticeable artifacts; EDGE often shows abrupt motion discontinuities; Lodge frequently produces abnormal movements that violate physical plausibility; MEGA, WAN-S2V, and Echomimic-V3 often produce overly simple and repetitive motions, limiting expressiveness. In contrast, videos generated by MACE-Dance not only present kinematically plausible and artistically expressive human motion, but also maintain spatiotemporally coherent visual appearance.

- 4.4.2 Cross-Genre Generation. Moreover, MACE-Dance generalizes effectively across dance genres as shown in Fig. 4, producing distinct genre-specific motion signatures, including (1) Uyghur dance exhibits light, continuous upper-body rotations with expressive arm trajectories; (2) Dunhuang motion features stable lower-body stances and elegant, circular arm patterns; (3) Dai style emphasizes soft, flowing wrist and elbow movements; (4) K-Pop example demonstrates crisp transitions, symmetrical poses, and rhythm-driven gestures; (5) Popping is characterized by sharp isolations and staccato movements, reflecting its percussive movement vocabulary.
- 4.4.3 Long-Sequence Generation. Additionally, a complete music track typically lasts 30 seconds to 5 minutes, making long-sequence generation crucial for practical dance video synthesis [Feng et al. 2025]. Tomitigate motiondriftorvisual degradation in long-sequence generation, MACE-Dance incorporates dedicated designs in both stages: (1) a BiMamba–Transformer hybrid in the Motion Expert for drift-free long motion synthesis, and (2) pose-driven relay rendering with identity anchoring in the Appearance Expert. As shown in Fig. 14, MACE-Dance produces coherent long-sequence dance video.

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Transformer

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Mamba

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

BiMamba

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

Fig. 5. Ablation for the model architecture of Motion Expert.

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

t = 0 s t = 30 s

GT Pose Wan-Animate w.o. Aesthetic Stage w.o. Kinematic Stage Appearance Expert

Fig. 6. MACE-Dance produces coherent long-sequence dance videos.

Fig. 7. Ablation for the Appearance Expert.

- 4.5 Ablation Study

- 4.5.1 Motion Expert. Given that the BiMamba–Transformer hybrid architecture and the Guidance-Free Training (GFT) strategy are central to the Motion Expert’s performance, we ablate them individually to assess their effects, as shown in Tab. 2 and Fig. 5. (1) Model Architecture. Replacing BiMamba with Mamba removes bidirectional context, weakening temporal understanding. Quantitatively, although generation efficiency improves, all dance-quality metrics degrade, indicating this is not a worthwhile trade-off. Qualitatively, the generated dances tend to resort to simple, common movements, diminishing the model’s artistic expressiveness. Replacing BiMamba with Transformer deprives the model of its ability to generate dance in a non-autoregressive manner, owing to selfattention’s scale-dependent positional extrapolation. Quantitatively, most metrics drop to unacceptable levels. Qualitatively, the model collapses to in-place side-to-side jitter, i.e., a poor local optimum. This also explains why BAS and FSR increase instead—these gains come at the cost of severely compromised dance quality. (2) Generative Strategy. Replacing GFT with naive classifier-free guidance (CFG) leads to a modest decline across most metrics. Notably, our generation efficiency improves by approximately 1.62×, because inference requires only a single conditional setting. Note: The effect of 𝛽 in GFT can be found in the supplementary material Sec. 4.2.

- 4.5.2 Appearance Expert. Since the two-stage fine-tuning strategy is central to our Appearance Expert, we evaluate the contribution of each stage via ablation, as summarized in Tab. 3 and Fig. 7. (1) Kinematic Stage. We fine-tune the Body Adapter while freezing all other components. Removing this stage leads to a modest decline across all metrics quantitatively and noticeable kinematic errors and motion blur qualitatively, indicating its effectiveness for ensuring human kinematic plausibility in video. (2) Aesthetic Stage. We fine-tune LoRA parameters in each DiT block. Omitting this stage

causes a substantial degradation across all metrics quantitatively and obvious ghosting artifact qualitatively, underscoring its critical role to preserve video aesthetic. Finally, Appearance Expert also demonstrates the superior performance over the Wan-Animate (Baseline), which validates the overall effectiveness of the proposed Kinematic-Aesthetic fine-tuning strategy.

- 4.5.3 Motion Representation (2D vs. 3D). Most pose-driven image animation methods rely on 2D keypoints, which naturally motivates using 2D poses as the intermediate motion representation. In contrast, MACE-Dance adopts 3D motion as its intermediate representation. To validate this design, we compare 2D and 3D representations at both the motion-generation level and the final video-generation level. Specifically, for the motion level, we train the same Motion Expert on FineDance with either 2D or 3D motion targets. For the video level, we further render final videos on MA-Data using either 2D pose sequences directly or 3D motion projected to 2D via our projector. As shown in Tab. 4, the 3D-based representation consistently outperforms the 2D-based one across both settings. On FineDance, 3D achieves better fidelity, diversity, and synchronization, indicating that it provides more stable and physically consistent supervision for music-to-motion learning. On MA-Data, 3D further yields clearly better subject consistency, visual fidelity, and beat alignment in the final rendered videos, showing that the advantages of 3D are preserved after the downstream animation stage. These results demonstrate that 3D motion serves as a more reliable intermediate interface than 2D pose for both controllable dance generation and high-quality video synthesis.
- 4.5.4 Role of Each Expert. To further analyze the role of each expert in MACE-Dance, we conduct an additional cross-composition study on MA-Data by replacing one expert at a time with its corresponding baseline counterpart. Specifically, w/o.ME denotes the variant that

Table 4. Comparison of 2D and 3D motion representations.

Kinematic Appearance FID𝑘↓ FID𝑔↓ DIV𝑘↑ DIV𝑔↑ BAS↑ AQ↑ SC↑ FID↓ BAS↑

- 2D 22.8 8.6 6.12 5.24 0.527 51.86 91.84 23.73 0.496

- 3D 19.5 4.1 8.87 5.92 0.543 51.79 93.97 16.46 0.523

Table 5. Cross-composition analysis of the two experts on MA-Data.

Method AQ↑ SC↑ FID↓ BAS↑ w/o.ME 50.21 92.10 20.84 0.499 w/o.AE 50.36 91.42 17.92 0.519 Ours 51.79 93.97 16.46 0.523

Table 6. Comparison with general-purpose video foundation models.

Method AQ↑ SC↑ FID↓ BAS↑ CogVideoX1.5-5B 50.38 89.92 22.47 0.477 WAN2.2-5B 53.22 90.77 17.53 0.452 Ours 51.79 93.97 16.46 0.523

[Figure 261]

Fig. 8. Comparison with general-purpose video foundation models.

uses the baseline Motion Expert (EDGE [Tseng et al. 2023]) together with our Appearance Expert, while w/o.AE denotes the variant that uses our Motion Expert together with the baseline Appearance Expert(WAN-Animate [Cheng et al. 2025]). As shown in Tab. 5, the full MACE-Dance consistently achieves the best performance across all evaluated metrics, confirming that both experts contribute positively to the final music-driven dance video generation quality. More specifically, replacing our Appearance Expert with the baseline model leads to clear degradation in appearancerelated metrics such as AQ and SC, while replacing our Motion Expert results in a more noticeable drop in the motion-related metric BAS. These observations suggest that the two experts play complementary roles: the Motion Expert mainly strengthens musicmotion alignment and body dynamics, whereas the Appearance Expert further improves visual quality, temporal coherence, and identity consistency in the rendered videos.

- 4.6 Comparison with Video Foundation Models

We further compare MACE-Dance with general-purpose video foundation models, including CogVideoX1.5-5B [Yang et al. 2024c] and WAN2.2-5B [Wan et al. 2025], to examine how a structured motionto-appearance pipeline performs against recent large-scale video generation models. Although these models demonstrate strong generation ability in broad video domains, they are not specifically designed for music-driven dance video generation, where accurate modeling of beat, rhythm, and body-motion coherence is particularly important. As shown in Table 6, MACE-Dance achieves the best overall performance on SC, FID, and BAS, indicating stronger music-motion alignment, better visual quality, and more consistent identity preservation. Although WAN2.2-5B attains a slightly higher AQ score, it underperforms our method on the other three metrics. Qualitatively, CogVideoX1.5-5B tends to produce weaker and slower dance motions with noticeable blur, while WAN2.2-5B generates larger motion amplitudes but often suffers from temporal identity inconsistency, as shown in Fig. 8. Overall, these results support the effectiveness of explicitly decomposing music-driven dance video generation into a Motion Expert and an Appearance Expert.

5 Conclusion

In conclusion, we present MACE-Dance, a music-driven dance videogenerationframeworkwithcascaded Mixture-of-Experts (MoE). The Motion Expert enforces kinematic plausibility and artistic expressiveness, while the Appearance Expert preserves visual identity with spatiotemporal coherence. Specifically, the Motion Expert adopts Diffusion Model with a BiMamba–Transformer hybrid backbone and Guidance-Free Training strategy, while the Appearance Expert adopts a decoupled Kinematic–Aesthetic fine-tuning strategy. To better benchmark this task, we curate a large-scale dataset, and design a motion–appearance evaluation protocol. Extensive experiments demonstrate the superiority of MACE-Dance and of its Motion and Appearance Experts. For future work, we plan to extend MACE-Dance with textual descriptions to enable more interactive and flexible dance generation, and improve system-level efficiency to support low-latency authoring and real-time user feedback.

Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grants 62436010, 72572090, 62572474 and 62172421, and in part by Tsinghua University School of Economics and Management Research Grant.

References

Jo Butterworth*. 2004. Teaching choreography in higher education: A process continuum model. Research in dance education 5, 1 (2004), 45–67.

Chubin Chen, Sujie Hu, Jiashu Zhu, Meiqi Wu, Jintao Chen, Yanxun Li, Nisha Huang, Chengyu Fang, Jiahong Wu, Xiangxiang Chu, et al. 2025a. Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning. arXiv preprint arXiv:2512.24146 (2025).

Chubin Chen, Jiashu Zhu, Xiaokun Feng, Nisha Huang, Meiqi Wu, Fangyuan Mao, Jiahong Wu, Xiangxiang Chu, and Xiu Li. 2025e. S2-Guidance: Stochastic Self Guidance for Training-Free Enhancement of Diffusion Models. arXiv preprint arXiv:2508.12880 (2025).

Huayu Chen, Kai Jiang, Kaiwen Zheng, Jianfei Chen, Hang Su, and Jun Zhu. 2025b. Visual generation without guidance. arXiv preprint arXiv:2501.15420 (2025).

Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. 2025c. Finger: Content aware fine-grained evaluation with reasoning for ai-generated videos. In Proceedings of the 33rd ACM International Conference on Multimedia. 3517–3526.

Zeyuan Chen, Hongyi Xu, Guoxian Song, You Xie, Chenxu Zhang, Xin Chen, Chao Wang, Di Chang, and Linjie Luo. 2025d. X-dancer: Expressive music to human dance video generation. arXiv preprint arXiv:2502.17414 (2025).

Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, et al. 2025. Wan-Animate: Unified Character Animation and Replacement with Holistic Replication. arXiv preprint arXiv:2509.14055 (2025).

Jiahao Cui, Hui Li, Yao Yao, Hao Zhu, Hanlin Shang, Kaihui Cheng, Hang Zhou, Siyu Zhu, and Jingdong Wang. 2024. Hallo2: Long-duration and high-resolution audiodriven portrait image animation. arXiv preprint arXiv:2410.07718 (2024).

Xiaokun Feng, Haiming Yu, Meiqi Wu, Shiyu Hu, Jintao Chen, Chen Zhu, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. 2025. Narrlv: Towards a comprehensive narrative-centric evaluation for long video generation models. arXiv e-prints (2025), arXiv–2507.

Chencan Fu, Yabiao Wang, Jiangning Zhang, Zhengkai Jiang, Xiaofeng Mao, Jiafu Wu, Weijian Cao, Chengjie Wang, Yanhao Ge, and Yong Liu. 2024. MambaGesture: Enhancing Co-Speech Gesture Generation with Mamba and Disentangled Multi-Modality Fusion. In Proceedings of the 32nd ACM International Conference on Multimedia. 10794–10803.

Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, et al. 2025. Wan-s2v: Audio-driven cinematic video generation. arXiv preprint arXiv:2508.18621 (2025).

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023). Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851. Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. ICLR 1, 2 (2022), 3.

Li Hu. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8153–8163.

Yin-Fu Huang and Wei-De Liu. 2021. Choreography cGAN: generating dances with music beats using conditional generative adversarial networks. Neural Computing and Applications 33, 16 (2021), 9817–9833.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21807–21818. Jinwoo Kim, Heeseok Oh, Seongjean Kim, Hoseok Tong, and Sanghoon Lee. 2022. A brand new dance partner: Music-conditioned pluralistic dancing controlled by multiple dance genres. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3490–3500.

Dorothée Legrand and Susanne Ravn. 2009. Perceiving subjectivity in bodily movement: The case of dancers. Phenomenology and the Cognitive Sciences 8 (2009), 389–408.

Jiachen Lei, Keli Liu, Julius Berner, Haiming Yu, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. 2025. There is no vae: End-to-end pixel-space generative modeling via self-supervised pre-training. arXiv preprint arXiv:2510.12586 (2025).

Huaqiu Li, Yong Wang, Tongwen Huang, Hailang Huang, Haoqian Wang, and Xiangxiang Chu. 2025. Ld-rps: Zero-shot unified image restoration via latent diffusion recurrent posterior sampling. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13684–13694.

Ronghui Li, Yuqin Dai, Yachao Zhang, Jun Li, Jian Yang, Jie Guo, and Xiu Li. 2024a. Exploring Multi-Modal Control in Music-Driven Dance Generation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 8281–8285.

Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. 2021. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13401–13412.

Ronghui Li, Hongwen Zhang, Yachao Zhang, Yuxiang Zhang, Youliang Zhang, Jie Guo, Yan Zhang, Xiu Li, and Yebin Liu. 2024c. Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns. arXiv preprint arXiv:2410.20389 (2024).

Ronghui Li, YuXiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li. 2024b. Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1524–1534.

Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. 2023. Finedance: A fine-grained choreography dataset for 3d full body dance generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10234–10243.

Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. 2025. Vmbench: A benchmark for perception-aligned video motion generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13087–13098.

Xinran Liu, Xu Dong, Diptesh Kanojia, Wenwu Wang, and Zhenhua Feng. 2025. GCDance: Genre-Controlled 3D Full Body Dance Generation Driven By Music. arXiv preprint arXiv:2502.18309 (2025).

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. 2023. SMPL: A skinned multi-person linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 851–866.

Brian McFee, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. 2015. librosa: Audio and music signal analysis in python.. In SciPy. 18–24.

Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, and Chenguang Ma.

2025. Echomimicv3: 1.3 b parameters are all you need for unified multi-modal and multi-task human animation. arXiv preprint arXiv:2507.03905 (2025).

Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. 2023. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505 (2023).

Ziqiao Peng, Yi Chen, Yifeng Ma, Guozhen Zhang, Zhiyao Sun, Zixiang Zhou, Youliang Zhang, Zhengguang Zhou, Zhaoxin Fan, Hongyan Liu, et al. 2025a. ActAvatar: Temporally-Aware Precise Action Control for Talking Avatars. arXiv preprint arXiv:2512.19546 (2025).

Ziqiao Peng, Wentao Hu, Junyuan Ma, Xiangyu Zhu, Xiaomei Zhang, Hao Zhao, Hui Tian, Jun He, Hongyan Liu, and Zhaoxin Fan. 2025b. SyncTalk++: High-Fidelity and Efficient Synchronized Talking Heads Synthesis Using Gaussian Splatting. arXiv preprint arXiv:2506.14742 (2025).

Ziqiao Peng, Wentao Hu, Yue Shi, Xiangyu Zhu, Xiaomei Zhang, Hao Zhao, Jun He, Hongyan Liu, and Zhaoxin Fan. 2024. Synctalk: The devil is in the synchronization for talking head synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 666–676.

Ziqiao Peng, Jiwen Liu, Haoxian Zhang, Xiaoqiang Liu, Songlin Tang, Pengfei Wan, Di Zhang, Hongyan Liu, and Jun He. 2025c. Omnisync: Towards universal lip synchronization via diffusion transformers. arXiv preprint arXiv:2505.21448 (2025).

Ziqiao Peng, Haoyu Wu, Zhenbo Song, Hao Xu, Xiangyu Zhu, Jun He, Hongyan Liu, and Zhaoxin Fan. 2023. Emotalk: Speech-driven emotional disentanglement for 3d face animation. In Proceedings of the IEEE/CVF international conference on computer vision. 20687–20697.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville.

2018. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, Vol. 32.

Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. 2022. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11050–11059.

Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. 2023. Bailando++: 3d dance gpt with choreographic memory. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2021. Denoising Diffusion Implicit Models. In International Conference on Learning Representations (ICLR).

Tomás Soucek and Jakub Lokoc. 2024. Transnet v2: An effective deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia. 11218–11221.

Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. 2019. Deep high-resolution representation learning for human pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5693–5703.

Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. 2024. Animate-x: Universal character image animation with enhanced motion representation. arXiv preprint arXiv:2410.10306 (2024).

Hao Tang, Ling Shao, Zhenyu Zhang, Luc Van Gool, and Nicu Sebe. 2025. SpatialTemporal Graph Mamba for Music-Guided Dance Video Synthesis. IEEE transactions on pattern analysis and machine intelligence (2025).

Jonathan Tseng, Rodrigo Castellon, and KarenLiu. 2023. Edge:Editable dancegeneration from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 448–458.

Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 2017. Neural Discrete Representation Learning. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 30. https://arxiv.org/abs/1711.00937

A Vaswani. 2017. Attention is all you need. Advances in Neural Information Processing Systems (2017).

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025).

Xuanchen Wang, Heng Wang, and Weidong Cai. 2025a. Choreomuse: Robust music-todance video generation with style transfer and beat-adherent motion. In Proceedings of the 33rd ACM International Conference on Multimedia. 7912–7921.

Xuanchen Wang, Heng Wang, Dongnan Liu, and Weidong Cai. 2025b. Dance any beat: Blending beats with visuals in dance video generation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 5136–5146.

Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. 2022. Vitpose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems 35 (2022), 38571–38584.

Zunnan Xu, Yukang Lin, Haonan Han, Sicheng Yang, Ronghui Li, Yachao Zhang, and Xiu Li. 2024a. Mambatalk: Efficient holistic gesture synthesis with selective state space

models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2024b. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1481–1490.

Kaixing Yang, Xulong Tang, Ran Diao, Hongyan Liu, Jun He, and Zhaoxin Fan. 2024a. CoDancers: Music-Driven Coherent Group Dance Generation with Choreographic Unit. In Proceedings of the 2024 International Conference on Multimedia Retrieval. 675–683.

Kaixing Yang, Xulong Tang, Yuxuan Hu, Jiahao Yang, Hongyan Liu, Qinnan Zhang, Jun He, and Zhaoxin Fan. 2025a. MatchDance: Collaborative Mamba-Transformer Architecture Matching for High-Quality 3D Dance Synthesis. arXiv preprint arXiv:2505.14222 (2025).

Kaixing Yang, Xulong Tang, Ziqiao Peng, Yuxuan Hu, Jun He, and Hongyan Liu. 2025b. Megadance: Mixture-of-experts architecture for genre-aware 3d dance generation. arXiv preprint arXiv:2505.17543 (2025).

Kaixing Yang, Xulong Tang, Ziqiao Peng, Xiangyue Zhang, Puwei Wang, Jun He, and Hongyan Liu. 2025c. FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation. arXiv preprint arXiv:2511.21029 (2025).

Kaixing Yang, Xulong Tang, Haoyu Wu, Qinliang Xue, Biao Qin, Hongyan Liu, and Zhaoxin Fan. 2024b. CoheDancers: Enhancing Interactive Group Dance Generation through Music-Driven Coherence Decomposition. arXiv preprint arXiv:2412.19123 (2024).

Kaixing Yang, Xukun Zhou, Xulong Tang, Ran Diao, Hongyan Liu, Jun He, and Zhaoxin Fan. 2024d. BeatDance: A Beat-Based Model-Agnostic Contrastive Learning Framework for Music-Dance Retrieval. In Proceedings of the 2024 International Conference

on Multimedia Retrieval. 11–19.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al.2024c. Cogvideox: Textto-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024).

Ziyue Yang, Kaixing Yang, and Xulong Tang. 2026. TokenDance: Token-to-Token Musicto-Dance Generation with Bidirectional Mamba. arXiv preprint arXiv:2603.27314

(2026).

Xiangyue Zhang, Yifan Jia, Jiaxu Zhang, Yijie Yang, and Zhigang Tu. 2025a. Robust 2D skeleton action recognition via decoupling and distilling 3D latent features. IEEE Transactions on Circuits and Systems for Video Technology (2025).

Xiangyue Zhang, Jianfang Li, Jianqiang Ren, and Jiaxu Zhang. 2025b. Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints. arXiv preprint arXiv:2511.10076 (2025).

Xiangyue Zhang, Jianfang Li, Jiaxu Zhang, Ziqiang Dang, Jianqiang Ren, Liefeng Bo, and Zhigang Tu. 2025c. Semtalk: Holistic co-speech motion generation with framelevel semantic emphasis. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13761–13771.

Xiangyue Zhang, Jianfang Li, Jiaxu Zhang, Jianqiang Ren, Liefeng Bo, and Zhigang Tu. 2025d. Echomask: Speech-queried attention-based mask modeling for holistic cospeech motion generation. In Proceedings of the 33rd ACM International Conference on Multimedia. 10827–10836.

Le Zhuo, Zhaokai Wang, Baisen Wang, Yue Liao, Chenxi Bao, Stanley Peng, Songhao Han, Aixi Zhang, Fei Fang, and Si Liu. 2023. Video background music generation: Dataset, method and evaluation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15637–15647.

###### Dance Creativity (DC)

###### Dance Quality (DQ)

###### Dance Sync (DS)

80

80

80

70

70

70

65.1%

65.1%

60.9%

60

60

60

Preference(%)

Preference(%)

Preference(%)

50

50

50

40

40

40

30

30

30

20

20

20

16.3%

15.0%

14.0%

13.0%

9.8%

9.6%

8.8%

8.6%

10

10

10

6.7%

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

2.5%

2.3%

2.3%

| | |
|---|---|
| | |

0

0

0

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Hallo2 WAN-SV Lodge MEGA Ours

Hallo2 WAN-SV Lodge MEGA Ours

Hallo2 WAN-SV Lodge MEGA Ours

###### Identity Consistency (IC)

###### Perceptual Quality (PQ)

Temporal Consistency (TC)

80

80

80

70

70

70

60.9%

60

60

60

56.2%

50.0%

Preference(%)

Preference(%)

Preference(%)

50

50

50

40

40

40

30

30

30

21.2%

20

20

20

17.4%

16.7%

15.4%

10.6%

10.3%

9.6%

9.0%

8.4%

- 10

10

10

6.2%

4.3%

| | |
|---|---|
| | |

3.8%

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0

0

0

Hallo2 WAN-SV Lodge MEGA Ours

Hallo2 WAN-SV Lodge MEGA Ours

Hallo2 WAN-SV Lodge MEGA Ours

Fig. 9. User study results comparing our method with four baselines. The bar charts display the percentage of user preferences across six dimensions: Dance Synchronization (DS), Dance Quality (DQ), Dance Creativity (DC), Perceptual Quality (PQ), Temporal Consistency (TC), and Identity Consistency (IC). Our method (Ours) consistently achieves the highest preference rates across all motion and appearance metrics.

6 Implementation Details

MACE-Dance is a music-driven dance video generation framework with cascaded Mixture-of-Experts (MoE), decoupling this task into music-to-3D motion generation task (Motion Expert) and posedriven image animation task (Appearance Expert). Additionally, due to the specific data requirements of each expert, the Motion Expert is trained exclusively on the 3D-rendered, motion-centric subset of the data, while the Appearance Expert is trained on the entire MA-Data dataset. We will introduce them in turn.

- 6.1 Motion Expert

For Motion Expert, we adopt the Diffusion Model with BiMambaTransformer hybrid architecture and Guidance-Free Training (GFT) strategy on the FineDance datasets. For training setup, we adopt the Adam optimizer with a learning rate 4 × 10−4, weight decay 0.02. The model is trained for 4000 epochs with a batch size of 128 using the Accelerate library for distributed training on 8 NVIDIA H20 Tensor Core GPUs. We train on sequences of 240 frames (8s) and perform inference on sequences of 1024 frames (34.13s). EMA (decay 0.9999) is applied to stabilize training, and checkpoints are periodically saved for evaluation (50 epochs). We combine multiple objectives: reconstruction loss (𝜆𝑟𝑒𝑐=0.636), 3D joint position loss (𝜆𝑗𝑜𝑖𝑛𝑡=0.636), velocity loss (𝜆𝑣𝑒𝑙=2.964) and foot contact loss (𝜆𝑓 𝑜𝑜𝑡=10.942). For model architecture, The conditional processing part contains 2 layers of BiMamba with Genre-Gate, and the vector generation part includes 8 layers of BiMamba-Transformer-based block. Each Mamba unit sets state 16, convolutional kernel size 4, and expansion factor 2, and latent dimension 512; each Transformer block utilizes 4 attention heads, a feed-forward network dimension of 1024, a dropout rate of 0.1, and the GELU activation function. We set temperature parameter 𝛽 in GFT 0.75 during inference.

- 6.2 Appearance Expert

For Appearance Expert, we adopt the Kinematic-Aesthetic decoupled fine-tuning strategy on the MA-Data Dataset. In the Kinematic

Stage, we exclusively fine-tune the Body Adapter to strengthen kinematic conditioning while freezing the entire DiT backbone and VAE. Training is conducted on NVIDIA H20 Tensor Core GPUs. We employ the Adam optimizer with a learning rate of 1×10−5 and a batch size of 128. This stage is trained for 50k iterations using the standard simple diffusion noise prediction loss to ensure strict motion adherence without altering the pre-trained generative prior. In the Aesthetic Stage, we freeze the kinematic pathways and fine-tune the extended LoRA branches to capture dance-specific visual patterns. We insert low-rank adapters with rank 𝑟 = 32 into the query, key, value, and output projections of the attention modules, as well as the feed-forward networks (FFN) within each DiT block. This stage is optimized using Adam with a learning rate of 5 × 10−5 for 50k iterations, minimizing the reconstruction loss to refine texture fidelity and spatiotemporal coherence. The training is distributed across 128 NVIDIA H20 GPUs.

7 User Study 7.1 Experimental Setting

User feedback is essential for evaluating generated dance movements in the music-to-dance generation task, due to the inherent subjectivity of dance [Legrand and Ravn 2009]. Following [Yang et al. 2025b], we select 30 real-world music segments, each lasting 8 seconds, and generated dance sequences using the models described in main paper Sec. 4.3.1. These sequences are evaluated through a double-blind questionnaire completed by 40 participants with dance backgrounds, including undergraduate and graduate students. Participants are compensated at a rate exceeding the local average hourly wage.

Different from scoring individual videos in isolation, we adopted a preference-based ranking mechanism to capture subtle differences between methods. For each query, participants were presented with generated videos from five distinct methods (our proposed method and four baselines) displayed side-by-side in randomized order. For every test case, participants were asked to select all videos that performed best according to the specific criteria, allowing for multiple selections in cases where several methods exhibited equally superior performance.

The evaluation was conducted across six dimensions, categorized into two key challenges of music-driven dance video generation:

- (1) Human Motion, which focuses on kinematic plausibility and artistic expressiveness. This includes:

- • Dance Synchronization (DS): Alignment with rhythm and style.
- • Dance Quality (DQ): Physical plausibility and aesthetic expressiveness.
- • Dance Creativity (DC): Originality and diversity of the movements.

- (2) Visual Appearance, which focuses on high-fidelity rendering and spatiotemporal consistency. This includes:

- • Perceptual Quality (PQ): Naturalness and overall aesthetic quality.
- • Temporal Consistency (TC): Smoothness and consistency of the subject and background over time.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

EDGE

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Lodge

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

MEGA

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

### Ours

Fig. 10. Motion Expert in MACE-Dance can generate high-quality 3D Motion with artistic expressiveness and physical plausibility.

• Identity Consistency (IC): Maintenance of the subject’s

identity relative to the reference image.

- 7.2 Result Analysis

The quantitative results of the user study are summarized in Fig. 9. Our method demonstrates a dominant preference rate across all six evaluated dimensions, significantly outperforming the four baseline methods.

- (1) Motion Performance. In terms of human motion genera-

tion, our method achieves the highest user preference. Specifically, for Dance Synchronization (DS) and Dance Quality (DQ), our method received over 60% of the user votes. This indicates that our approach not only aligns dance beats more precisely with the music rhythm but also generates kinematically more plausible and aesthetically pleasing movements compared to competitors. Notably, in Dance Creativity (DC), our method also leads by a substantial margin, suggesting that our model avoids repetitive patterns and produces more diverse choreographic sequences.

- (2) Appearance Performance. Regarding visual quality, users

overwhelmingly preferred our results. For Perceptual Quality (PQ) and Identity Consistency (IC), our method secured the vast majority of preferences, validating the effectiveness of our generation pipeline in preserving fine-grained details and subject identity. Furthermore, the high preference rate in Temporal Consistency (TC) demonstrates our model’s superior ability to maintain stability across frames, effectively mitigating flickering and temporal artifacts that are common in baseline methods.

Overall, the user study results align with the qualitative visualizations, confirming that our method sets a new state-of-the-art standard in both motion expressiveness and visual fidelity.

7.3 Evaluation Analysis

To examine whether the proposed motion–appearance evaluation protocol aligns with human perception, we compare the quantitative results of Tab. 1 in main paper with the User Study outcomes presented in Fig. 9. Across all six human-rated dimensions—Dance Creativity (DC), Dance Quality (DQ), Dance Sync (DS), Identity Consistency (IC), Perceptual Quality (PQ), and Temporal Consistency (TC)—MACE-Dance is overwhelmingly preferred by participants, with preference ratios ranging from 50% to 65.1%. These human judgments exhibit strong correspondence with our quantitative metrics.

- (1) Motion Side. Methods ranked highest by participants on DQ and DS are exactly those achieving superior FID𝑘, FID𝑔, DIV𝑘, DIV𝑔, and BAS scores. In particular, the substantial improvement of MACE-Dance in BAS (0.523), FID𝑔 (0.28), and DIV𝑘 (9.74) is mirrored by its leading human preference in DQ (65.1%) and DS (65.1%). This demonstrates that our motion metrics faithfully capture the perceptual qualities that users associate with expressive, synchronized, and natural dance movement.
- (2) Appearance Side. The VBench-derived metrics (IQ, AQ, SC, BC, MS, TF) show clear alignment with user ratings in PQ and IC. For example, MACE-Dance achieves the highest scores in SC (93.97), BC (94.57), and TF (97.10), which directly correspond to its large margins in Identity Consistency (50.0%) and Temporal Consistency (56.2%) in the user study. Similarly, its strong IQ and AQ scores coincide with the highest PQ rating (60.9%) among all compared methods.

Taken together, the strong consistency between quantitative metrics and human preference validates the effectiveness of our motion–appearance evaluation protocol. This confirms that the proposed metrics not only provide reliable automatic assessment but also closely reflect human perceptual judgments, making them a meaningful and principled framework for evaluating music-driven dance video generation.

MagicAnimate

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

AnimateAnyone

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

WANAnimate

#### Ours

Pose

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

Fig. 11. The Appearance Expert in MACE-Dance drives image-based dancing with spatiotemporally coherent appearance.

- 8 Qualitative Analysis

- 8.1 Music-Driven Dance Video Generation

We further provide additional qualitative comparisons across diverse reference-image domains and music genres to complement the evaluations in the main paper Sec. 4.3.1. As illustrated in Fig. 12, Hallo2 produces blurred facial regions and introduces substantial background artifacts; EDGE frequently suffers from abrupt motion discontinuities that degrade temporal smoothness; Lodge often yields physically implausible body configurations and irregular motion patterns; and WAN-S2V and Echomimic-V3 tends to generate overly simplified and repetitive motion sequences that lack expressive variety. In contrast, our method (MACE-Dance) generates videos with kinematically plausible and artistically expressive movements while

preserving spatiotemporally coherent appearance across frames. These results further validate the superior qualitative performance of MACE-Dance across a wide range of reference-image inputs and musical styles.

8.2 Music-Driven 3D Dance Generation

We also conduct qualitative analyses for the music-driven 3D dance generation task. As shown in Fig. 10, the observations are consistent with those reported in Sec 3.1 of the Appendix. Specifically, EDGE exhibits abrupt motion discontinuities that compromise temporal smoothness; Lodge often produces physically implausible body configurations and irregular motion patterns; and MEGA tends to generate overly simplified and repetitive motion sequences with limited expressive diversity. In contrast, our Motion Expert synthesizes 3D

Hallo2

EDGE

Lodge

WANS2V

Echo mimicV3

Ours

Hallo2

EDGE

Lodge

WANS2V

Echo mimicV3

Ours

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

(a) Case 1

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

(b) Case 2 Fig. 12. More cases of qualitative comparison with SOTAs in music-driven dance video generation task.

[Figure 459]

where 𝑁 is the number of frames and 𝐷 is the motion dimensionality. Given a partial motion constraint 𝑥known together with a binary mask 𝑚 ∈ {0, 1}𝑁×𝐷, where 𝑚𝑖𝑗 = 1 indicates that the corresponding element is fixed, we perform masked denoising at each reverse step by replacing the constrained region with the forward-diffused version of the known motion at the same noise level:

[Figure 460]

𝑧˜𝑡−1 = 𝑚 ⊙ 𝑞(𝑥known,𝑡 − 1) + (1 −𝑚) ⊙ 𝑧ˆ𝑡−1, (10)

where 𝑧ˆ𝑡−1 is the current reverse sample predicted by the model, 𝑞(𝑥known,𝑡 − 1) denotes the forward diffusion of the known motion to timestep 𝑡 − 1, and ⊙ is element-wise multiplication. In this way, the constrained region remains faithful to the user-provided motion signal, while the unconstrained region is generated by the diffusion prior to ensure temporal smoothness, physical plausibility, and musical coherence. Importantly, this editing mechanism is fully compatible with our DDIM-based inference and requires no additional training.

[Figure 461]

As illustrated in Fig. 13, this formulation naturally supports three practical editing modes through different mask designs. First, temporal inpainting preserves motion at the beginning and/or end of a sequence and synthesizes the missing middle part, enabling motion in-betweening and smooth transition generation. Second, joint-wise inpainting fixes selected body parts while allowing the model to infer the remaining joints, such as preserving upperbody motion while completing the lower-body dance, or vice versa. Third, trajectory-guided inpainting constrains sparse trajectoryrelated channels such as root translation or turning direction, and lets the model generate the full-body pose sequence that follows the prescribed path. These examples show that MACE-Dance is not limited to one-shot motion generation, but can also function as a controllable motion editing tool for choreography and animation workflows.

- Fig. 13. Visualization for motion editing. From top to bottom, the first row shows temporal-level motion editing (yellow indicates the given motion sequence, and green indicates the completed part); the second row shows joint-level motion editing (the upper body indicates the given motion sequence, and the red lower body indicates the completed part); the third row shows trajectory-level completion (given the blue trajectory, the motion sequence is completed accordingly).

motion that is both kinematically plausible and artistically expressive, demonstrating stable dynamics and rich stylistic detail. These results further validate the superiority of the proposed Motion Expert in modeling high-quality, music-driven 3D dance motion.

Another important advantage of the proposed Motion Expert is that its output is explicit 3D motion, which can be directly transferred to standard character rigs through conventional motion retargeting pipelines, as also shown in Fig. 13. This substantially broadens the applicability of MACE-Dance beyond music-driven video synthesis. In addition to being rendered by our Appearance Expert, the generated dance motion can be reused as a structured motion asset for CG animation, VR avatars, interactive character control, and other human-computer interaction scenarios that require editable and transferable body motion. More broadly, because the output remains in a structured 3D form, the framework is also potentially extensible to embodied platforms such as humanoid agents or dancing robots after appropriate skeleton mapping and control-level adaptation. In this sense, the Motion Expert is not only a component for improving video generation quality, but also a general-purpose music-to-motion generator with strong downstream utility in animation, XR, and embodied AI applications.

8.3 Pose-Driven Image Animation

We further conduct qualitative comparisons against Magic-Animate, Animate-Anyone, and Wan-Animate on the MA-Data test set. As shown in Fig. 11, existing methods exhibit several limitations when dealing with dance-specific motion patterns. Magic-Animate and Animate-Anyone often produce noticeable spatial distortions and temporal flickering in fast or large-amplitude motions, leading to unstable body shapes and inconsistent textures across frames. Wan-Animate, while stronger in preserving subject identity, still struggles with motion adherence—particularly in rapid limb movements—resulting in lagging body parts and partial pose mismatch. These qualitative observations highlight the advantage of our twostage specialization and demonstrate that the proposedAppearance Expert effectively adapts general-purpose image animation models to the unique demands of pose-driven dance video synthesis.

- 9 Motion Editing

Beyond unconditional music-driven dance synthesis, the Motion Expert in MACE-Dance also supports motion editing at inference time through a masked denoising strategy, similar to diffusion-based inpainting. Since the model operates on structured 3D motion sequences rather than pixels, it can preserve user-specified motion constraints while plausibly completing the remaining unknown regions. Formally, let a motion sequence be denoted as 𝑥 ∈ R𝑁×𝐷,

10 Further Discussion about MACE-Dance 10.1 Temperature Parameter 𝛽 in GFT

As mentioned in main paper Sec. 3.2.1, we adopt Guidance-Free Training (GFT [Chen et al. 2025b]). GFT reformulates conditional training to directly learn a 𝛽-indexed sampling model via linear

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

t = 0 s t = 30 s

- Fig. 14. MACE-Dance produces long-sequence dance videos with artistic expressiveness and physical plausibility.

Table 7. Effect of the 𝛽 in Guidance-Free Training (GFT).

𝛽 FID𝑘↓ FID𝑔↓ FSR↓ DIV𝑘↑ DIV𝑔↑ BAS↑ GT – – 0.216 9.94 7.54 0.201 1.00 29.35 31.91 0.270 13.29 9.68 0.220 0.75 17.83 25.09 0.210 10.30 8.09 0.229 0.50 15.11 24.15 0.210 8.64 6.79 0.233 0.00 NaN NaN NaN NaN NaN NaN

interpolation with the unconditional output. This allows a single model to capture an entire family of diversity-fidelity trade-offs robustly, eliminating the need for post-hoc guidance. 𝛽 serves as a temperature parameter that is also provided to the model 𝜃 as an additional conditioning input. During inference, values of 𝛽 near 0 favor high fidelity, while values near 1 favor high diversity. Thus, 𝛽 can also be regarded as a control signal, and we set its value to 0.75. To empirically validate the effect of the parameter 𝛽 and justify our choice, we conducted an ablation study as presented in Tab. 7. The results confirm the expected trade-off between diversity and fidelity. Specifically, 𝛽 = 1.00 yields the highest diversity scores (𝐷𝐼𝑉𝑘 = 13.29,𝐷𝐼𝑉𝑔 = 9.68) but suffers from the poorest fidelity. Conversely,

- 𝛽 = 0.50 achieves the best fidelity (𝐹𝐼𝐷𝑘 = 15.11, 𝐹𝐼𝐷𝑔 = 24.15) at the expense of diversity, which drops below the ground truth. A value of 𝛽 = 0.00 leads to numerical instability, confirming it is unsuitable for inference. We ultimately select 𝛽 = 0.75 as it offers the most compelling balance. It dramatically improves fidelity over
- 𝛽 = 1.00 (e.g., 𝐹𝐼𝐷𝑘 drops from 29.35 to 17.83) while retaining strong diversity (𝐷𝐼𝑉𝑘 = 10.30,𝐷𝐼𝑉𝑔 = 8.09) that surpasses both the highfidelity setting (𝛽 = 0.50) and the ground truth. This makes it the optimal choice for producing results that are both high-quality and varied.

10.2 Task Decoupling Analysis

MACE-Dance is a music-driven dance video generation framework with a cascaded Mixture-of-Experts (MoE) architecture, which decouples the task into music-to-3D motion generation (Motion Expert) and pose-driven image animation (Appearance Expert). This design is motivated by the principles of reducing learning complexity and improving data utilization, as detailed below:

(1) Complexity reduction via task factorization. By separating the original cross-modal mapping from music directly to pixels into two more constrained subproblems, each expert can focus on a well-defined objective. The Motion Expert specializes in modeling the temporal relationship between music and human kinematics, without interference from visual factors such as texture or lighting.

Conversely, the Appearance Expert addresses a conditional image synthesis task given explicit pose inputs, without requiring an understanding of musical semantics. This specialization enables each expert to learn a more robust and domain-appropriate representation.

- (2) Suppression of spurious cross-modal correlations. Endto-end models are prone to learning incidental correlations between musical features and visual artifacts present in the training data (e.g., background or clothing cues). Introducing an explicit 3D motion representation acts as a structured information bottleneck, compelling the model to focus on the intrinsic relationship between music and movement while filtering out irrelevant visual factors. We empirically observe this phenomenon when adapting several representative end-to-end human motion generation models, including Hallo2, EchoMimic-V3, and WAN-S2V. Despite architectural modifications or fine-tuning, these models exhibit clear spurious correlations. This limitation is reflected in the consistent performance gap between these baselines and our method, as reported in Tab. 1, Fig. 3, Fig. 8 of the main paper, and Fig. 4 in the appendix.
- (3) Interpretability and explicit control through structured representations. The intermediate 3D motion representation provides a transparent and editable interface that can be inspected, modified, or replaced prior to final rendering. Such interpretability and controllability are fundamentally unavailable in monolithic endto-end models. Overall, the cascaded MoE design facilitates model specialization, improves data efficiency, and enables user-level control, leading to more robust and reliable dance video generation.

10.3 Long-Sequence Generation

In the domain of dance video generation, long-sequence generation is not merely an enhancement but a fundamental requirement for practical applications. Its importance is multifold: first, a complete dance performance is an expressive narrative with an emotional arc, intrinsically tied to the full duration of a musical piece (typically 30s-4min). Short clips fail to capture the choreographic structure, narrative progression, and full artistic integrity. Second, to achieve precise music synchronization, the model must process motion sequences matching the entire length of the musical score, ensuring long-term alignment of movements with the beat, melody, and mood. However, prevailing methods in general human video generation are often constrained by the limited temporal window of their underlying base models [Peng et al. 2025b, 2024, 2025c] (e.g., under 5 seconds). Naively extending these models to long-sequence tasks inevitably confronts the critical challenge of error accumulation. This error manifests as motion drift, identity degradation, and temporal incoherence. To overcome this core problem, our framework employs a synergistic two-stage strategy, achieving high-quality long-sequence dance video generation, as shown in Fig. 14:

(1) Motion Expert with length extrapolation capability. The Motion Expert employs a BiMamba–Transformer hybrid architecture that combines global structural modeling with local temporal continuity. Transformer blocks capture global choreographic structure and long-range dependencies via self-attention, while BiMamba layers model local motion dynamics with linear complexity. Although trained on short motion clips (e.g., 8 seconds), the model can generate sequences of arbitrary length at inference time. This

is enabled by the state-space recurrence of Mamba, which serves as a temporal memory that continuously propagates local dynamics beyond the training horizon, while the Transformer provides high-level structural guidance within its receptive field.

(2) Pose-anchored relay generation in the Appearance Expert. Given the coherent long motion sequence, the Appearance Expert renders the final video using a pose-driven image animation paradigm rather than generic video prediction. Each generation chunk is constrained by three complementary anchors: (i) the globally consistent 2D pose sequence from the Motion Expert, which provides an absolute geometric reference; (ii) the last frame of the previous chunk, ensuring appearance continuity (e.g., lighting and clothing); and (iii) a constant reference image, enforcing identity consistency. Together, these constraints effectively prevent error accumulation and maintain long-term visual coherence, in contrast to unconstrained autoregressive video generation.

- 11 Ethical Considerations

Although MACE-Dance is designed for music-driven dance video generation in creative and entertainment contexts, it may also introduce ethical risks. In particular, as with other human video generation systems, the model could be misused to synthesize realistic videos of individuals without their consent, potentially enabling misleading or deceptive media. This concern is especially relevant because the Appearance Expert preserves identity-related cues from a reference image while generating temporally coherent videos.

Moreover, the training data may exhibit biases in dance style, body shape, clothing, scene composition, and cultural representation, which can lead to uneven generation quality across different subjects or styles. Accordingly, the outputs of the model should not be interpreted as neutral or universally representative.

We stress that MACE-Dance is intended for research on controllable dance video synthesis, not for identity manipulation or harmful content creation. Any practical deployment should respect consent, portrait rights, and copyright constraints, and future releases should consider safeguards such as usage restrictions, provenance disclosure, or watermarking mechanisms.

- 12 Limitations and Future Work 12.1 Customized Dance Generation.

Although our framework MACE-Dance achieves strong performance in music-driven dance video generation. Music serves as

a fixed-form carrier and cannot fully capture diverse user intentions. To address the limitations, we envision extending control modalities to incorporate free-form textual descriptions. Text offers the lowest-cost input modality while allowing users to express choreographic requirements in a more flexible and semantically rich manner, thereby facilitating personalized and expressive dance generation. Specifically, text provides a rich, hierarchical control mechanism, enabling users to articulate dance from multiple levels of abstraction. It can define high-level artistic concepts like mood and style (e.g., ’an energetic hip-hop dance’), while also specifying low-level kinematic details such as a sequence of actions or the movement of a particular limb (e.g., ’spin and then raise both arms’). This direction not only enhances user interactivity and creativity but also unlocks new opportunities for content-driven applications in human–computer interaction. While recent studies have explored text-controlled human video generation [Peng et al. 2025a], current approaches are hindered by the limited scale of available dance video and the difficulty in acquiring textual descriptions that not only align with natural user expression patterns but also precisely reflect the essential characteristics of dance movements. Thus, leveraging text as a control modality is a pivotal next step, promising to unlock truly personalized and creative dance generation.

12.2 Dance Generation with Efficiency.

Real-time interaction represents a critical and compelling direction for dance video generation. While our Motion Expert has achieved state-of-the-art (SOTA) generation efficiency in the 3D motion synthesis stage, a significant performance bottleneck remains in the Appearance Expert. Specifically, although our fine-tuned Appearance Expert, based on the 14B-parameter Wand-Animate model, delivers SOTA quality in pose-driven image animation, its substantial computational demands preclude its use in real-time applications. To bridge this efficiency gap, several promising research avenues can be explored. These include knowledge distillation, where a compact student model is trained to mimic the large teacher model; model compression techniques like quantization and pruning; and, more fundamentally, designing a novel, lightweight Appearance Expert architecture optimized for speed. Ultimately, achieving a harmonious balance between generation quality and computational efficiency is the key to unlocking the full potential of interactive dance video synthesis.

