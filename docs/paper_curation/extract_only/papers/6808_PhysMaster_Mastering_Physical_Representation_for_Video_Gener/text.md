# arXiv:2510.13809v1[cs.CV]15Oct2025

PHYSMASTER: MASTERING PHYSICAL REPRESENTATION FOR VIDEO GENERATION VIA REINFORCEMENT LEARNING

∗

†

Sihui Ji1

, Xi Chen1, Xin Tao2, Pengfei Wan2, Hengshuang Zhao1

1The University of Hong Kong 2Kling Team, Kuaishou Technology

### https://sihuiji.github.io/PhysMaster-Page/

ABSTRACT

Video generation models nowadays are capable of generating visually realistic videos, but often fail to adhere to physical laws, limiting their ability to generate physically plausible videos and serve as “world models”. To address this issue, we propose PhysMaster, which captures physical knowledge as a representation for guiding video generation models to enhance their physics-awareness. Specifically, PhysMaster is based on the image-to-video task where the model is expected to predict physically plausible dynamics from the input image. Since the input image provides physical priors like positions, materials, and interactions of objects in the scenario, we devise PhysEncoder to encode such physical representation as an extra condition. Nevertheless, there is no well-established definition for a physical representation. Thus, we could not use an off-the-shelf model to extract the physical condition, or set a straightforward supervision to train PhysEncoder. To solve this challenge, we adopt a top-down optimization strategy, where the PhysEncoder is optimized based on the physical plausibility of the final generated videos using reinforcement learning (RL). Through this top-down optimization, the PhysEncoder can effectively capture implicit physical cues from the starting image and inject them into the video generation process. Experiment results prove that, our method significantly enhances the model’s physics-awareness as a plugin, and demonstrate strong performance on both specialized proxy tasks and general open-world scenarios.

1 INTRODUCTION

Video generation models (Brooks et al., 2024; Kuaishou, 2024; Yang et al., 2024c) have developed rapidly nowadays, achieving significant performances in generating visually appealing videos (RunwayML, 2024; Team, 2024; Kong et al., 2024). However, they primarily act as sophisticated pixel predictors based on case-specific imitation, and often face challenges in adherence to physical laws (Kang et al., 2024; Liu et al., 2025a; Meng et al., 2025). This limits their ability to generate physically plausible videos and further comprehend physical principles to serve as “world models”. To evolve these models from content creators to world simulators, we aim to incorporate physical knowledge into the video generation process to enhance their physical realism.

We summarize the specific challenges of physics-aware video generation into the following two points. First, the commonly used Mean Squared Error (MSE) loss for data-driven finetuning focuses on appearance fitting rather than comprehension of physical knowledge, and it is non-trivial to directly supervise the physical performance of pretrained models beyond merely appearance. Second, generative models struggle to extract appropriate physical knowledge from a textual instruction or an input image and translate it into physical guidance for generation, which demands logical reasoning from descriptions or images to physical knowledge, and to visual phenomena. Progress for enhancing physics-awareness of video generation has been made and the solutions can be broadly categorized into two types based on the usage of simulation. Simulation-based approaches (Lv et al., 2024; Liu et al., 2024b) attempt to apply physics-based simulation results to guide video generation, but they are often constrained in the range of simulable physical processes and modalities, lacking

∗ Work done during an internship at Kling Team, Kuaishou Technology. † Corresponding author.

the potential to generalize to diverse phenomena. Simulation-free methods (Xue et al., 2024; Furuta et al., 2024) rely on post-training on physics-rich data or employ reinforcement learning for aligning to human preference. The former highly depends on fitting similar training samples, and the latter utilizes either expensive human annotation suffering from rater variability, or scalable but inaccurate AI evaluators. In summary, existing works find it hard to truly abstract and understand physics of the world (Lin et al., 2025; Motamed et al., 2025), hindering generalization to diverse physics.

Facing the aforementioned challenges, we propose to learn a physical representation as the bridge between the necessary physical knowledge and generated videos to guide generative models towards physics awareness. Specifically, we focus on image-to-video (I2V) generation where an initial frame and textual description are given, the model is supposed to predict physically plausible dynamics from input scenes. The input image offers visual cues like object configurations, relative positions, and potential interactions that largely dictate the subsequent physical evolution of video, making it a reliable source of physical priors. Thus firstly, we devise a physical encoder, PhysEncoder, to extract implicit physical representation from the input image as an extra input condition to guide the generation process for enhancing physics-awareness of the model.

However, how to learn an effective physical representation for video generation remains an open question. Without a well-established definition for physical representation, we can not conduct straightforward supervision of PhysEncoder for training. Thus we propose a top-down optimization strategy to optimize its ability for guiding physically plausible video generation by reinforcement learning with human feedback (RLHF) framework, which has proven its effectiveness in finetuning of both large language models (LLMs) (Yuan et al., 2023; Xu et al., 2024; Yuan et al., 2023) and generative models (Lee et al., 2023; Prabhudesai et al., 2024; Fan et al., 2023). The physical plausibility of generated videos guided by PhysEncoder acts as feedback for optimizing PhysEncoder to extract effective physical representations. Specifically, we train PhysEncoder on human preference data via Direct Preference Optimization (DPO) (Rafailov et al., 2023) in a three-stage training pipeline. We first conduct supervised fine-tuning (SFT) of both base model and PhysEncoder, then we adopt a two-stage DPO with pairwise supervision based on the physical plausibility of the final generated videos to enhance PhysEncoder’s capacity to capture physical representations and model’s physical performance, with trainable module separately set as LoRA (Hu et al., 2021) of the DiT model and PhysEncoder. With SFT providing the model with initial ability to predict physically plausible videos under the guidance of simultaneously finetuned PhysEncoder, the subsequent DPO processes further enable PhysEncoder to effectively capture implicit physical cues from the starting image and inject them into video generation, thus helping improve physical understanding of the model.

Last but not least, we condition the video generation model on physical representation in a plug-in manner, enhancing physics-awareness by injecting physical knowledge into it. Such a paradigm enables the model to learn general physical properties, rather than overfitting to specific phenomena or being constrained to particular motion modalities as in previous works, thus allowing it to generalize to diverse scenarios. We demonstrate the effectiveness of PhysEncoder starting from a specialized proxy task of “free-fall”, and then generalize to general open-world physical scenarios governed by a wide range of physical laws. PhysEncoder proves its capablity by guiding the generation model towards enhanced physical performance, and such generalization implies that PhysMaster, our representation learning paradigm facilitates the physical comprehension in a broad scope.

In summary, PhysMaster provides a more generalizable solution for video generation models to capture physical knowledge across diverse physical phenomena, showing its advantage in acting as a foundational solution for physics-aware video generation and potential to energize more fancy applications (Agarwal et al., 2025; Yang et al., 2024b; 2023).

- 2 RELATED WORKS

Physics-aware video generation. While recent video generation models achieve impressive visual effects (Brooks et al., 2024; Kong et al., 2024), they still struggle with adherence to real-world physical laws (Lin et al., 2025; Kang et al., 2024). Physics-aware video generation approaches can be broadly categorized based on the application of explicit physical simulation. Simulation-based methods (Lv et al., 2024; Xie et al., 2025; Montanaro et al., 2024; Zhang et al., 2024b) guide generation with simulation results. PhysGen (Liu et al., 2024b) utilizes rigid-body dynamics simulated with physical parameters inferred by large foundation models. PhysMotion (Tan et al., 2024) relies

[Figure 1]

Text emb Physical emb Image emb Video emb

Generatedvideo 3D VAE Decoder

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

## Transformer Blocks

[Figure 6]

❄

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Text Encoder PhysEncoder🔥

3D VAE Encoder

[Figure 11]

[Figure 12]

❄ ❄

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

”A bag is dropping.“

[Figure 19]

Input image Input image

Ground truth

Preference Data Construction Gradient backpropagation

- Figure 1: Overall architecture of PhysMaster. Given an input image, PhysEncoder encodes its physical feature and concatenates with visual features, then the DiT model predicts subsequent frames conditioned on physical, visual, and text embeddings. We optimize PhysEncoder ’s physical representation via feedback from generated video pairs of the model by maximizing reward derived from “positive” and “negative” video outputs in a DPO paradigm.

on MPM-based simulation to generate coarse videos which are refined by a video diffusion model. As for simulation-free approaches, they either fine-tune on large-scale video datasets to implicitly internalize physical priors (Wang et al., 2025; Zhang et al., 2025), or use reinforcement learning with feedback from human annotators or vision-language models (Xue et al., 2024; Furuta et al., 2024). PhyT2V (Xue et al., 2024) uses MLLMs to refine prompts iteratively through multiple rounds of generation and reasoning. WISA (Wang et al., 2025) incorporates structured physical information into the generative model and uses Mixture-of-Experts for different physics categories. However, those methods are restricted to fixed physical categories or exhibit limited physical comprehension. Our PhysMaster incorporates physical knowledge into video generation process via physical representation to enhance general physics-awareness.

RLHF for video generation. Inspired by the success of RLHF in LLMs (Ouyang et al., 2022; Jaech et al., 2024), researchers have explored applying this paradigm to video generation (Zhang et al., 2024a; Qian et al., 2025). VideoDPO (Liu et al., 2024a) pioneers the adaptation of DPO (Rafailov et al., 2023) to video diffusion models by considering both visual quality and semantic alignment for data pair construction. VideoAlign (Liu et al., 2025b) introduces a multi-dimensional video reward model and DPO for flow-based video generation model based on it. PISA (Li et al., 2025) investigates specifically for video generation of object free-fall, improving physical accuracy through reward modeling based on depth and optical flow. Unlike the aforementioned methods, we optimize a physical encoder rather than the whole video generation model by leveraging generative feedback from the model’s outputs. This paradigm mitigates overfitting to specific physical processes and promotes the encoder’s generalizability for learning universal physical knowledge through RLHF.

- 3 METHOD

Based on I2V setting, PhysMaster extracts physical representation from the input image and optimizes both the generation model and PhysEncoder in a three-stage training pipeline. It seeks direct supervision from groundtruth via SFT and pairwise supervision from generated videos via DPO, and is implemented on the specialized proxy task and general open-world scenarios. We will separately detail physical representation (Sec 3.1), task formulation (Sec 3.2) and training scheme (Sec 3.3).

- 3.1 PHYSICAL REPRESENTATION

PhysMaster is implemented upon a transformer-based diffusion model (DiT) (Peebles & Xie, 2023), which employs 3D Variational Autoencoder (VAE) (Kingma, 2013) to transform videos and initial frame to latent space, and T5 encoder ET5 (Raffel et al., 2020) for text embeddings ctext.

We propose to learn a physical representation from input image as extra guidance for the I2V model to inject physical information, since the input image contains not only explicit physical states, such as object material and spatial distribution, but also implicit physical laws, like the gravitational field.

It is worth expecting that the learned physical representation can be used as a generalizable guidance of both physical properties and dynamics for physics-aware video generation. Following the structure of Depth Anything (Yang et al., 2024a), we build PhysEncoder with a DINOv2 (Oquab et al., 2023) encoder and a physical head. The former adopts pretrained weights from Yang et al. (2024a) for initialization and takes the role of semantic perception, while the latter adapts the extracted highlevel semantic features into an appropriate dimension to be injected into the DiT model. Taking the first frame as image input, PhysEncoder encodes it into physical embeddings cphys, which are then fed into DiT model after concatenated with image embeddings cimage. For SFT, the flow-based DiT model with weights θ directly parameterizes the vθ(zt,t,ctext,cimage,cphys) to regress velocity (z1 − z0) with the Flow Matching objective (Lipman et al., 2022):

#### 0,ϵ||vθ(zt,t,ctext,cimage,cphys) − (z1 − z0)||22. (1)

LLCM = Et,z

- 3.2 TASK FORMULATION

Our work aims to provide a scalable and generalizable methodology for learning physics from targeted data, so for demonstrating the effectiveness of our PhysMaster, we start by defining a proxy task under specialized physical principles and construct domain-specific data for preliminary validation; then we verify its generalizability across a broader range of physical laws and various tasks.

Proxy task. For preliminary verification, “free-fall” (involving the complete physical process of objects dropping from mid-air and colliding with other objects on a surface), a specialized yet expressive scenario is chosen as the proxy task for the following characteristics. First, “free-fall” embodies clear and fundamental physical principles (e.g., energy and momentum conservation) shared across diverse physical scenarios, making it a suitable representative for further generalization. Second, such physical scenario involves a wide range of object-level physical properties, such as density, elasticity, and hardness, allowing proof of generalizability of learned representations across different physical attributes. Third, this task can be easily simulated for scalable generation of synthetic data and allows for straightforward evaluation by comparing generated videos against ground-truths. The reason is that by assuming the falling object starts from rest and is only influenced by gravity, the trajectories of objects become fully deterministic given the initial frame, which also enables automatic construction of preference video pairs for DPO by similarity with ground-truths.

General open-world scenarios. We further substantiate generalization capabilities of PhysMaster across diverse physical processes. Following WISA (Wang et al., 2025), we include large-scale scenarios broadly covering common physical phenomena observed in real world for PhysEncoder to acquire a far more comprehensive and generalizable understanding of physical laws and thus effectively enhances the physics awareness of the video generation model. Different from the proxy task implementation, we modify the text prompts provided to the generation model by adding domainspecific prefixes (e.g., “Optic, A ray of light ...”, “Thermodynamic, A glass of water ...”). This conditions the model on the type of involved physics laws and guides it to associate visual phenomena with underlying physics extracted from PhysEncoder. For preference assignment, we rely on human annotators to provide pairwise labels for DPO data construction and evaluation.

- 3.3 TRAINING SCHEME

We propose a three-stage training pipeline for PhysMaster to enable physical representation learning of PhysEncoder by leveraging the generative feedback from I2V model. The core idea is formulating DPO for PhysEncoder with the reward signal from generated videos of pretrained DiT model, thus help physical knowledge learning without explicit modeling.

- Stage I: SFT for DiT and PhysEncoder. First, we condition the I2V base model on physical representation from PhysEncoder by SFT, thus it is possible for us to optimize PhysEncoder with the performance of model as feedback in following stages. Since PhysEncoder’s training starts from the frozen DINOv2 with pretrained weights from Depth Anything (Yang et al., 2024a) and trainable physical head with randomly initialized weights, this stage can be viewed as adapting Depth Anything for physical condition injection. As in Figure 1, by concatenating physical embeddings extracted by PhysEncoder with visual embeddings encoded by VAE, we inject physical representation as extra condition to the model. SFT following Eq 1 equips the model with the initial ability to predict subsequent frames from the input image, guided by simultaneously finetuned PhysEncoder.

- Stage II: DPO for DiT. Second, we expect to adapt the output of the pretrained model to a more physically plausible distribution, paving the way for the PhysEncoder to learn from generated videos with higher physical accuracy. Then in Stage II, we apply LoRA (Hu et al., 2021) to finetune the DiT model on preference dataset with DPO, during which the model learns to generate positive samples with higher probability and negative samples with lower probability. Regarding I2V setting, each sample in our preference dataset includes a prompt p, an image i, a human-chosen video xw and a human-rejected video xl. The goal of DPO is to learn a conditional distribution πθ(x | p,i) that maximizes the reward rϕ(x,p,i) while staying close to a reference model πref:

max

πθ

Ep∼D

p,i∼Di,x∼πθ(x|p,i) [rϕ(x,p,i)] − β DKL [πθ(x | p,i)∥πref(x | p,i)] (2)

where β controls the regularization term (KL-divergence) from πref. For our flow-based DiT model, Flow-DPO objective (Liu et al., 2025b) LFD(θ) is then given by:

−E log σ −

β 2 ∥vw − vθ(xwt , t)∥2 − ∥vw − vref(xwt , t)∥2 − ∥vl − vθ(xlt, t)∥2 − ∥vl − vref(xlt, t)∥2 ,

(3)

where the conditioning prompt p and image i are omitted for simplicity, vθ denotes the predicted velocity field, vw, vl are the target velocity of “preferred” and “less preferred” data.

The pretrained DiT model from Stage I is regarded as the reference model and is used to construct the preference data pairs. Specifically, we generate two groups of videos with the pretrained model using the same prompt p and initial frame i but different seeds. By establishing clear distinctions between positive samples xw and negative samples xl, the model learns to generate physically plausible videos. As a result, we further enhance the model’s physics awareness.

- Stage III: DPO for PhysEncoder. We leverage generative feedback from the pretrained DiT model to optimize PhysEncoder’s physical representation via DPO paradigm. As illustrated in Figure 1, our framework consists of two parts: PhysEncoder to be optimized and the pretrained DiT model providing generative feedback. With physical head of PhysEncoder the only trainable module, Stage III shares the same training objective Eq 3 with Stage II, differing solely in the learnable parameters. LFD(θ) leads PhysEncoder to learn a physical representation that guides the predicted velocity field vθ closer to the target velocity vw of the “preferred” data. In this manner, by directing the DiT model to generate more accurate physical dynamics, the PhysEncoder’s original representation is gradually optimized with more physical knowledge through model feedback.

- 4 EXPERIMENTS

To evaluate the effectiveness of PhysMaster for physical representation learning and demonstrate its potential to enhance physical performance of the DiT model, comprehensive experiments are conducted on both the proxy task and wide-ranging scenarios.

- 4.1 IMPLENTATION DETAILS

Training configuration. The training of both PhysEncoder and the DiT model is conducted on 8 NVIDIA-A800 GPUs in all three stages, with 20 hours for SFT, 15 hours for DPO on LoRA and 8 hours for DPO on PhysEncoder. The training process employs the Adam optimizer (Kingma, 2014), and we utilize 50 DDIM steps (Song et al., 2020) and set the CFG scale to 7.5 during inference.

Dataset construction. For the proxy task, we follow PISA (Li et al., 2025) to use Kubric (Greff et al., 2022) to create synthetic datasets of “free-fall”. The object assets are sourced from the Google Scanned Objects (GSO) dataset (Downs et al., 2022). For generalizability demonstration, we utilize WISA-80K (Wang et al., 2025) encompassing 17 types of real-world physical events across three major branches of physics (Dynamics, Thermodynamics, and Optics).

Evaluation protocols. PisaBench (Li et al., 2025) is introduced to evaluate our model’s performance on the proxy task. We use SAM 2 (Ravi et al., 2025) for segmentation of object masks and

- Table 1: Ablation study for models from different training stages and strategies on proxy task, evaluated on the test set split into “seen” and “unseen”. vθ is DiT model, Ep is PhysEncoder.

Seen Unseen Average L2 (↓) CD (↓) IoU (↑) L2 (↓) CD (↓) IoU (↑) L2 (↓) CD (↓) IoU (↑)

Training Stages

Base 0.1066 0.323 0.119 0.1065 0.339 0.111 0.1066 0.331 0.115 SFT for vθ 0.0532 0.134 0.137 0.0512 0.133 0.135 0.0522 0.134 0.136 SFT for vθ & Ep (Stage I) 0.0568 0.141 0.137 0.0498 0.128 0.134 0.0533 0.134 0.135 SFT for vθ + DPO for vθ 0.0560 0.143 0.144 0.0446 0.115 0.128 0.0503 0.129 0.136 SFT for vθ & Ep + DPO for vθ (Stage II) 0.0520 0.125 0.133 0.0454 0.116 0.136 0.0487 0.120 0.134 SFT for vθ & Ep + DPO for Ep 0.0559 0.140 0.138 0.0503 0.129 0.134 0.0531 0.134 0.136 SFT for vθ & Ep + DPO for vθ & Ep 0.0501 0.124 0.141 0.0458 0.121 0.140 0.0480 0.123 0.141 SFT for vθ & Ep + DPO for vθ + DPO for Ep (Stage III) 0.0489 0.120 0.141 0.0450 0.115 0.145 0.0470 0.118 0.143 SFT for vθ & Ep + DPO for vθ + DPO for vθ & Ep 0.0477 0.121 0.144 0.0452 0.115 0.141 0.0466 0.118 0.142

|[Figure 20]|
|---|

A plastic bottle falls.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

|[Figure 30]|
|---|

A plastic box is dropping.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Ours PISA PhysGen

- Figure 2: Qualitative comparison with video generation models specialized for rigid-body motion proves the advantage of our model in shape consistency and trajectory accuracy on “free-fall”.

compute the following metrics between corresponding masks of generated and ground truth videos for evaluation: L2 distance between the centroids of the masked regions, chamfer distance (CD) and Intersection over Union (IoU) of the mask regions. We utilize VIDEOPHY (Bansal et al., 2024) for evaluating physics awareness of video generation in general open-world scenarios. We test on 344 carefully crafted prompts from it, which reflect a wide array of physical principles, and report the physical commonsense (PC) and semantic adherence (SA) scores.

- 4.2 EVALUATION ON PROXY TASK

To validate that our training pipeline can effectively improve the physical performance of base model on the proxy task, we compare the physical accuracy of our model on ”free-fall” motion with existing works and ablate different training techniques of PhysEncoder.

Comparison. We compare our model with PhysGen (Liu et al., 2024b) and PISA (Li et al., 2025) on the real-world subset from PisaBench (Li et al., 2025) which is unseen to any model during training for robust evaluation. We apply more rigorous metrics of similarity over all objects in the scene by comparing against the ground truth. Table 2 shows that our model outperforms both baselines. PhysGen struggles with accurately modeling spatial relationships between objects and surfaces like ground or tables, thus often leads to physically implausible object interactions. For PISA, its best variant with depth-based reward optimizes for trajectory accuracy (comparable L2/CD) at the cost of shape consistency (lower IoU). In contrast, ours excels in IoU while maintaining competitive trajectory accuracy, achieving the best overall performance, which is also proved in Figure 2.

Ablation study. We report the qualitative results from different training stages and pipelines on the synthetic subset of PisaBench in Table 1, where block 1 denotes the I2V base model; block 2 - 4 refer to our model and its variants in Stage I - III of training pipeline. “Seen” corresponds to a split of videos with objects and backgrounds seen during training, and “Unseen” with novel objects and backgrounds. 1) Ablation for different training stages. row 3, 5, and 8 indicate that our SFT endows the model with preliminary ability to predict objects’ motion of “free-fall”, the subsequent DPO for DiT model further steers the generated videos’ distribution towards physically plausible paths, and the optimization of PhysEncoder in the last stage improves its capability in guiding model towards higher level of physics-awareness. The qualitative results in Figure 3 consistently prove our pipeline’s efficacy. 2) Ablation for PhysEncoder. The comparative pipeline in row 2 and

- 4 is not equipped with PhysEncoder, thus SFT and DPO are both implemented on the DiT model. Although the performance after SFT on both DiT and PhysEncoder (row 3) is even worse than SFT on DiT alone (row 2), showing that single SFT cannot help PhysEncoder learn appropriate physical

|[Figure 40]|
|---|

|[Figure 41]|
|---|

###### A brown bottle falls. A black bag is dropping.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Base

Base

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

SFT for 𝒗𝜽 & 𝑬𝒑

SFT for 𝒗𝜽 & 𝑬𝒑

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

SFT for 𝒗𝜽 & 𝑬𝒑+DPO for 𝒗𝜽

SFT for 𝒗𝜽 & 𝑬𝒑+DPO for 𝒗𝜽

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

SFT for 𝒗𝜽 & 𝑬𝒑+DPO for 𝒗𝜽 +DPO for 𝑬𝒑

SFT for 𝒗𝜽 & 𝑬𝒑+DPO for 𝒗𝜽 +DPO for 𝑬𝒑

- Figure 3: Qualitative ablation for models in each training stage on proxy task. The model exhibits a preliminary capability for predicting object motion trends after SFT. Two-stage DPO further improves model performance in preserving objects’ rigidity and complying with physical laws (e.g., gravitational acceleration and collision). vθ is DiT model and Ep is PhysEncoder.

- Table 2: Quantitative comparison with video generation models specialized for rigid-body motion verifies superiority of our model on proxy task.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Input 𝑬𝒑 in Stage I 𝑬𝒑 in Stage III Input 𝑬𝒑 in Stage I 𝑬𝒑 in Stage III

Figure 4: Visualization of the first three PCA components of physical representation. Ep in Stage III reveals similarities in objects under the same external forces (red: on the ground; green: in the air) over Ep in Stage I.

Methods L2 (↓) CD (↓) IoU (↑)

PhysGen 0.0433 0.0967 0.418 PISA 0.0294 0.0570 0.433 Ours 0.0299 0.0567 0.468

representations for guiding the DiT model towards physics-awareness, DPO unlocks PhysEncoder ’s potential to extract physical information and guide the model to generate videos with better physical performance (row 4, 5,8). 3) Ablation for DPO strategies. All strategies of DPO succeed in further improving physical accuracy on average than previous stages. Only optimizing PhysEncoder (row 6) encounters difficulties in performance improvements. The model itself has not been aligned to adherence to physics before providing feedback to PhysEncoder, preventing DPO from functioning effectively. Although Stage II of our training pipeline underperforms joint DPO of DiT and PhysEncoder (row 7), our Stage III surpasses all other methods. Joint optimization of DiT and PhysEncoder in Stage III (row 9) achieves comparable overall performances with our Stage III but performs worse on ”unseen” split, probably because this variant with trainable DiT is more likely to overfit training data, harming the model’s generalizability to novel scenarios.

PCA analysis. We also visualize the principal component analysis (PCA) on the physical features from PhysEncoder in Stage I and Stage III in Figure 4. In our Stage III physical feature maps, similarities are shown clearly for objects under the same external forces, (green for objects in mid-air and only affected by gravity, red for objects on the ground subjected to the support force); differences are also shown more obviously between materials (e.g., deformable object in white box has clearly distinct colors), which proves two aspects of physical understanding of our PhysEncoder.

A glass strikes the ground and shatters, scattering sharp fragments in all directions. A slice of lemon drops into water, splashing droplets upward.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

CogVideoX-5B

CogVideoX-5B

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

WISA (Wan2.1-T2V-14B)

WISA (Wan2.1-T2V-14B)

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Wan2.1-T2V-1.3B

Wan2.1-T2V-1.3B

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Ours

Ours

Multiple rigid balls roll along a metal track, clashing into each other as they move. A plastic bottle slides down a chute, driven forward by the force of rushing water.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

CogVideoX-5B

CogVideoX-5B

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

WISA (Wan2.1-T2V-14B)

WISA (Wan2.1-T2V-14B)

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Wan2.1-T2V-1.3B

Wan2.1-T2V-1.3B

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Ours

Ours

Rigid-body movements Fluid motion

- Figure 5: Qualitative comparison with existing T2V models on general open-world scenarios including objects of various materials and in different environments, validates the generalizability of our method.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

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

[Figure 142]

[Figure 143]

[Figure 144]

A close-up of a clear glass bottle is being topped off with a stream of water.

|[Figure 145]|
|---|

A hand pours beer into a tall glass filled with ice and a slice of lime on the rim.

|[Figure 146]|
|---|

|[Figure 147]|
|---|

Stage I

Stage III

Stage I

Stage I

Stage I

A man shaves his beard with a razor, pushing away the foam from his face. A towel on the table is used to wipe away dust from the surface.

Stage III

Stage III

Stage III

- Figure 6: Qualitative ablation for models in different stages on general open-world scenarios. DPO following Stage I improves the physical coherence of model in Stage III (e.g., fluid mechanics and gravitation).

- 4.3 GENERALIZATION ON GENERAL OPEN-WORLD SCENARIOS

PhysEncoder demonstrates its physics-awareness for enhancing model’s physical realism on the proxy task, suggesting its potential to generalize to broader open-world scenarios. We apply our

- Table 3: Quantitative comparison with existing video generation models on general open-world scenarios. Our model shows superior performance in both physics-awareness and efficiency. Methods Inference Time (s) SA (↑) PC (↑)

HunyuanVideo 1080 0.46 0.28 Wan2.1-T2V-1.3B 180 0.49 0.24 CogvideoX-5B 210 0.60 0.33 Cosmos 600 0.57 0.18 PhyT2V 1800 0.61 0.37 WISA 220 0.67 0.38 Our base model 23 0.59 0.29 Our final model 26 0.67 0.40

Table 4: Ablation study for PhysEncoder on general open-world scenarios. vθ is DiT model, Ep is PhysEncoder. The results validate that DPO enables Ep to acquire a comprehensive understanding of real-world physics and thus effectively enhances the physics awareness of vθ.

Methods SA (↑) PC (↑) Base 0.59 0.29 SFT for vθ 0.63 0.33 SFT for vθ + DPO for vθ 0.64 0.35 SFT for vθ & Ep (Stage I) 0.61 0.33 SFT for vθ & Ep + DPO for vθ + DPO for Ep (Stage III) 0.67 0.40

training pipeline on a large-scale dataset (Wang et al., 2025) broadly covering common physical phenomena observed in real world to substantiate the generalizability of our method.

Comparison. We compare with two types of video generation models, general models including HunyuanVideo (Kong et al., 2024), CogVideoX-5B (Yang et al., 2024c), Cosmos-Diffusion7B (Agarwal et al., 2025), Wan2.1-T2V-1.3B and specialized physics-focused models represented by PhyT2V (Xue et al., 2024) and WISA (Wang et al., 2025). Table 3 shows that, although our base model is surpassed by CogvideoX-5B, the base model of WISA, our final model in Stage III achieves state-of-the-art performance on both SA and PC metrics, demonstrating that our proposed method enhances the realism of generated videos physically and semantically. Our model also has a significant advantage in efficiency. It is approximately 70x faster than PhyT2V—an iterative method requiring feedback from VLM, and 8x faster than WISA. Our model generates a 5-second video in just 26 seconds on a single A800 GPU, establishing it as a highly practical solution without sacrificing physical or semantic adherence. Figure 5 includes qualitative comparison with existing T2V models, demonstrating our superior ability in challenging cases of both rigid-body and fluid motion.

Table 5: User study for models from different stages validates the effect of our training pipeline in two selected physical scenarios. Our final model shows superior ability of PhysEncoder in Stage III in enhancing the model’s physics-awareness over the base model and Stage I.

Ablation study. We conduct ablation analysis to verify the effectiveness of our core component and strategy of training in Table 4. 1) Effectiveness of PhysEncoder: Compared to our base model (row 1), our final model (row 5) improves SA and PC scores by 0.08 and 0.11. The comparative pipeline is not equipped with PhysEncoder, with SFT (row 2) and the following DPO (row 3) both implemented on the DiT model only. Such a pipeline without PhysEncoder improves SA and PC scores by 0.05 and 0.06, proving the advantage of our proposed PhysEncoder in successfully extracting crucial physical knowledge from the training data and using it to guide the generator toward greater physical realism, which is unattainable by simply applying SFT or DPO to the DiT model alone. 2) Effectiveness of DPO: Simply applying SFT to PhysEncoder (row 2 vs. row 4) does not yield an immediate benefit, suggesting that SFT alone is insufficient for PhysEncoder to learn a useful guiding physical representation. However, DPO unlocks the potential of PhysEncoder, allowing it to effectively translate its learned physical representations into improved generation quality in both physical commonsense and semantic adherence (row 4 vs. row 5). Additionally, Figure 6 visualizes the videos generated by our model in Stage I and III, further validating the effectiveness of DPO. 3) Effectiveness of whole training pipeline: Table 5 includes human preference rates among models from different training stages in two types of real-world scenarios, showing that annotators prefer videos from Stage III model than Stage I model or I2V base model in adherence to physical laws.

Methods Rigid-body movement Fluid motion

Base 7.8 12.2 Stage I 25.3 16.7 Stage III 66.9 71.1

- 5 CONCLUSIONS

We propose PhysMaster, which learns physical representation from input image for guiding I2V model to generate physically plausible videos. We optimize physical encoder based on generative feedback from a pretrained video generation model via DPO, which proves to enhance the model’s physical accuracy and demonstrate generalizability across various physical processes by injecting

physical knowledge into generation, proving its potential to act as a generic and plug-in solution for physics-aware video generation and broader applications.

Limitations. We rely on human annotators to construct preference datasets for DPO in real-world scenarios, which is costly and time-consuming. Existing AI evaluators, however, have flawed physics knowledge and inherit biases, limiting the scalability of reinforcement learning. Fortunately, our DPO training paradigm is effective even with a small amount of human-labeled data (500 in our experiment), mitigating this limitation.

REFERENCES

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv:2501.03575, 2025. 2, 9

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024. 6

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. OpenAI Blog, 2024. URL https://openai.com/ research/video-generation-models-as-world-simulators. 1, 2

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In ICRA, 2022. 5

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. In NeurIPS, 2023. 2

Hiroki Furuta, Heiga Zen, Dale Schuurmans, Aleksandra Faust, Yutaka Matsuo, Percy Liang, and Sherry Yang. Improving dynamic object interactions in text-to-video generation with ai feedback. arXiv:2412.02617, 2024. 2, 3

Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In CVPR, 2022. 5

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv:2106.09685,

2021. 2, 5

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv:2412.16720, 2024. 3

Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv:2411.02385,

2024. 1, 2 Diederik P Kingma. Auto-encoding variational bayes. arXiv:1312.6114, 2013. 3 Diederik P Kingma. Adam: A method for stochastic optimization. arXiv:1412.6980, 2014. 5 Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li,

Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv:2412.03603, 2024. 1, 2, 9

Kuaishou. Kling video model. https://kling.kuaishou.com/en, 2024. 1

Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv:2302.12192, 2023. 2

Chenyu Li, Oscar Michel, Xichen Pan, Sainan Liu, Mike Roberts, and Saining Xie. Pisa experiments: Exploring physics post-training for video diffusion models by watching stuff drop. arXiv:2503.09595, 2025. 3, 5, 6

Minghui Lin, Xiang Wang, Yishan Wang, Shu Wang, Fengqi Dai, Pengxiang Ding, Cunxiang Wang, Zhengrong Zuo, Nong Sang, Siteng Huang, et al. Exploring the evolution of physics cognition in video generation: A survey. arXiv:2503.21765, 2025. 2

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv:2210.02747, 2022. 4

Daochang Liu, Junyu Zhang, Anh-Dung Dinh, Eunbyung Park, Shichao Zhang, Ajmal Mian, Mubarak Shah, and Chang Xu. Generative physical ai in vision: A survey. arXiv:2501.10928, 2025a. 1

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv:2501.13918, 2025b. 3, 5

Runtao Liu, Haoyu Wu, Zheng Ziqiang, Chen Wei, Yingqing He, Renjie Pi, and Qifeng Chen. Videodpo: Omni-preference alignment for video diffusion generation. arXiv:2412.14167, 2024a. 3

Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physicsgrounded image-to-video generation. In ECCV, 2024b. 1, 2, 6

Jiaxi Lv, Yi Huang, Mingfu Yan, Jiancheng Huang, Jianzhuang Liu, Yifan Liu, Yafei Wen, Xiaoxin Chen, and Shifeng Chen. Gpt4motion: Scripting physical motions in text-to-video generation via blender-oriented gpt planning. In CVPR, 2024. 1, 2

Siwei Meng, Yawei Luo, and Ping Liu. Grounding creativity in physics: A brief survey of physical priors in aigc. arXiv:2502.07007, 2025. 1

Antonio Montanaro, Luca Savant Aira, Emanuele Aiello, Diego Valsesia, and Enrico Magli. Motioncraft: Physics-based zero-shot video generation. Advances in Neural Information Processing Systems, 37:123155–123181, 2024. 2

Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models learn physical principles from watching videos? arXiv:2501.09038, 2025. 2

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv:2304.07193, 2023. 4

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In NeurIPS, 2022. 3

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3 Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak.

Video diffusion alignment via reward gradients. arXiv:2407.08737, 2024. 2

Wenxu Qian, Chaoyue Wang, Hou Peng, Zhiyu Tan, Hao Li, and Anxiang Zeng. Rdpo: Real data preference optimization for physics consistency video generation. arXiv preprint arXiv:2506.18655, 2025. 3

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023. 2, 3

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 3

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In ICLR, 2025. 5

RunwayML. Gen-3 alpha. https://runwayml.com/research/ introducing-gen-3-alpha, 2024. 1

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020. 5

Xiyang Tan, Ying Jiang, Xuan Li, Zeshun Zong, Tianyi Xie, Yin Yang, and Chenfanfu Jiang. Phys-

motion: Physics-grounded dynamics from a single image. arXiv:2411.17189, 2024. 2 Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024. 1 Jing Wang, Ao Ma, Ke Cao, Jun Zheng, Zhanjie Zhang, Jiasong Feng, Shanyuan Liu, Yuhang Ma,

Bo Cheng, Dawei Leng, et al. Wisa: World simulator assistant for physics-aware text-to-video generation. arXiv:2503.08153, 2025. 3, 4, 5, 9

Tianyi Xie, Yiwei Zhao, Ying Jiang, and Chenfanfu Jiang. Physanimator: Physics-guided generative cartoon animation. arXiv:2501.16550, 2025. 2

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. arXiv:2401.08417, 2024. 2

Qiyao Xue, Xiangyu Yin, Boyuan Yang, and Wei Gao. Phyt2v: Llm-guided iterative self-refinement for physics-grounded text-to-video generation. arXiv:2412.00596, 2024. 2, 3, 9

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In NeurIPS, 2024a. 4

Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv:2310.06114, 2023. 2

Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv:2402.17139, 2024b. 2

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv:2408.06072, 2024c. 1, 9

Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback. In NeurIPS, 2023. 2

Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng Xiao, Weilin Huang, and Kai Han. Onlinevpo: Align video diffusion model with online video-centric preference optimization. arXiv:2412.15159, 2024a. 3

Ke Zhang, Cihan Xiao, Yiqun Mei, Jiacong Xu, and Vishal M Patel. Think before you diffuse: Llms-guided physics-aware video generation. arXiv preprint arXiv:2505.21653, 2025. 3

Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In European Conference on Computer Vision, pp. 388–406. Springer, 2024b. 2

