## Ctrl-Crash: Controllable Diﬀusion for Realistic Car Crashes

Anthony Gosselina,b,∗, Ge Ya Luoa,c, Luis Laraa, Florian Golemoa, Derek Nowrouzezahraia,d,e, Liam Paulla,c,e, Alexia Jolicoeur-Martineauf, Christopher Pala,b,e

aMila, Montreal, Quebec, Canada bPolytechnique Montréal, Montreal, Quebec, Canada cUniversité de Montréal, Montreal, Quebec, Canada dMcGill University, Montreal, Quebec, Canada eCIFAR AI Chair, Toronto, Canada fSamsung SAIL Montréal, Montreal, Quebec, Canada

# arXiv:2506.00227v2[cs.CV]13Dec2025

#### Abstract

Video diﬀusion techniques have advanced signiﬁcantly in recent years; however, they struggle to generate realistic imagery of car crashes due to the scarcity of accident events in most driving datasets. Improving traﬃc safety requires realistic and controllable accident simulations. To tackle the problem, we propose Ctrl-Crash, a controllable car crash video generation model that conditions on signals such as bounding boxes, crash types, and an initial image frame. Our approach enables counterfactual scenario generation where minor variations in input can lead to dramatically diﬀerent crash outcomes. To support ﬁne-grained control at inference time, we leverage classiﬁer-free guidance with independently tunable scales for each conditioning signal. Ctrl-Crash achieves stateof-the-art performance across quantitative video quality metrics (e.g., FVD and JEDi) and qualitative measurements based on a human-evaluation of physical realism and video quality compared to prior diﬀusion-based methods.

Project page: https://anthonygosselin.github.io/Ctrl-Crash-ProjectPage/

Keywords: Video diﬀusion, Controllable generation, Autonomous Driving, Rare Event Simulation, Counterfactual Reasoning

#### 1. Introduction

ethical complexity of crash events [3]. Moreover, most generative approaches focus on normal driving behavior, avoiding the complexity and unpredictability inherent in crash dynamics, where agent interactions, rare motion patterns, and semantic context all matter deeply [4, 5].

Autonomous vehicle (AV) systems must be rigorously tested in a wide range of driving situationsincluding rare and dangerous edge cases such as collisions to ensure safe deployment. Much of the current progress in perception, planning, and control for AVs has been driven by large-scale datasets collected in harmless, non-crash scenarios. However, realistic video data of car crashes remains extremely scarce, making it diﬃcult to simulate, anticipate, or learn eﬀectively from these critical events [1].

To address this gap, we introduce Ctrl-Crash, a controllable video diﬀusion framework for generating realistic crash videos from a single initial frame. Our method operates with inputs and outputs in pixel space, as opposed to using computer graphics primitives and explicit models of physics. Our approach can generate video conditioned on an initial image frame, spatial control signals consisting of bounding box sequences of cars and pedestrians, and semantic intent signals encoded as discrete crash types, enabling the generation of diverse crash outcomes. Through these conditioning signals, we can direct the narrative of a crash, simulate plausible sequences of interactions, and explore counterfactual variants of a given scene, answering the following types of questions with high quality generated video: How might the scene evolve diﬀerently under a diﬀerent agent trajectory or crash type?

Prior work has largely approached the challenge of crash scenario modeling in two main ways.

On one hand, physics-based rendering approaches use game engines or physics simulators to model accident dynamics, but often fall short on visual realism, require expensive rendering pipelines, and demand costly human eﬀort for environment and asset creation [2]. On the other hand, data-driven methods, such as generative models, rely on real-world footage, which is difﬁcult to acquire in suﬃcient volume due to the infrequency and

Ctrl-Crash builds on latent diﬀusion models [6] and classiﬁer-free guidance [7], and we extend the latter to allow independently tunable guidance strengths for each control modality, making our system highly ﬂexible at inference time. Our two-stage training procedure ﬁrst ﬁnetunes a pretrained Stable Video Diﬀusion (SVD) [8] model on in-the-wild egoview accident videos, then trains a ControlNet [9] adapter to

∗Corresponding author

Email addresses: anthony.gosselin@mila.quebec (Anthony Gosselin), xugeya@mila.quebec (Ge Ya Luo), luis.lara@mila.quebec (Luis Lara), golemofl@mila.quebec (Florian Golemo), derek@mila.quebec (Derek Nowrouzezahrai), paulll@iro.umontreal.ca (Liam Paull), alexia.jolicoeur-martineau@mail.mcgill.ca (Alexia Jolicoeur-Martineau), christopher.pal@mila.quebec (Christopher Pal)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Ground-Truth

💥Crash type: car vs car

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Generated

💥Crash type: self vs car

Input

🖼Initial frame

3 bbox frames

Frame 0 Frame 5 Frame 10 Frame 20

- Figure 1: Counterfactual Crash Generation: this diagram demonstrates the ability of our model to generate counterfactual crashes while beginning from the identical initial frame. Top: a ground truth accident between two vehicles other than the ego-vehicle, where the red car hits the rear of the blue car and spins into the lane in front of the ego-vehicle. Bottom: the model generates an alternative accident involving the ego-vehicle. In this alternative future the red car avoids the blue car but turns into the path of the ego-vehicle leading to the crash.

handle conditioning in order to direct the video generation. By leveraging this data-driven framework, our model can generate controllable crash videos that are visually realistic, semantically diverse, and behaviorally plausible. We see this work as a step toward not only improving the diversity and coverage of safety-critical AV testing, but also enabling counterfactual safety reasoning: the ability to simulate alternate outcomes from identical initial conditions, and better understand the causality of crashes.

#### Our contributions:

- 1. We introduce Ctrl-Crash, a fully data-driven generative framework for realistic and controllable car crash video generation. Our method obtains state-of-the-art performance compared to prior diﬀusion-based car accident video generation methods in quantitative (e.g., FVD) and qualitative analysis (human-evaluation of physical realism and video quality).
- 2. Our approach can generate plausible and diverse crash outcomes from the same initial frame and initial bounding boxes, enabling counterfactual video simulation for safety-critical reasoning.
- 3. We develop a data-processing pipeline to ﬁlter driving videos and extract bounding box trajectories of road users. Using it, we release a curated extension of the MM-AU dataset [10] (with bounding box annotations), as well as held-out test sets from Car Crash Dataset RUSSIA [11] (with bounding box annotations) and BDD100k [12] (with existing boxes), along with tools to support research in crash simulation and controllable video generation. Code and dataset extensions are all made open-source.

#### 2. Related Work

Video Diﬀusion Models. Diﬀusion models [13] have emerged as a powerful paradigm for generative modeling, particularly in the domain of image synthesis. They operate by learning to reverse a gradual noising process applied to data, generating high-ﬁdelity samples through iterative denoising. Recent advances have extended these techniques to the video domain, where temporal consistency and spatial coherence are critical [14, 8].

Given the noise-level t, and task-speciﬁc conditions c, the diﬀusion loss is the following:

[∥ϵ − ϵθ(xt,t,c))∥22

], (1)

L = Ex

0,t,c,ϵ∼N(0,I)

where xt = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,I), x0 is a training sample video, α¯t at t ∈ [1,T] controls the diﬀusion schedule, and ϵθ is a function approximator intended to predict the noise ϵ from the noise-corrupted video xt.

Latent video diﬀusion models (LVDMs) [15, 8] address the computational challenges of high-resolution video generation by operating in a compressed latent space. This enables the generation of long, high-quality video sequences from compact representations. Speciﬁcally, Stable Video Diﬀusion (SVD) [8], an LVDM variant, leverages a UNet-based denoiser trained on video latents conditioned on an initial frame, making it suitable for tasks like image-to-video generation and temporal extension.

Our work builds on this foundation by ﬁne-tuning a pretrained SVD model on a large curated car crash dataset and giving it additional controllability mechanisms to make it wellsuited for generating complex driving scenes and crash dynamics.

Controllable Generative Models. Recent progress in generative modeling has emphasized not only ﬁdelity, but also controllability the ability to guide outputs through structured input signals. In image generation, this includes control via text prompts, sketches, bounding boxes, semantic maps, and more. In diﬀusion models, classiﬁer-free guidance (CFG) [7] and ControlNet [9] have been introduced as eﬀective methods to allow adaptable conditioning while preserving high-quality generation.

CFG is a widely-used technique in diﬀusion models to improve conditional generation by combining conditional and unconditional predictions, scaled to control how strongly the model follows the conditioning input. It involves jointly training the diﬀusion model for conditional and unconditional denoising by randomly setting the conditioning to a null value c = ∅ during training. During inference, the denoising prediction is interpolated between a conditional noise estimate ϵθ(xt,c) and an unconditional noise estimate ϵθ(xt,∅), scaled by a guidance factor γ to obtain the modiﬁed score estimate:

ϵˆθ(xt,c) = ϵθ(xt,∅) + γ · (ϵθ(xt,c) − ϵθ(xt,∅)). (2)

Vision models, such as InstructPix2Pix [16], use textual or mixed-mode conditioning with CFG to manipulate generation intent and content mid-sampling. ControlNet [9] introduced an eﬀective approach for injecting spatial control into diﬀusion models by adding a parallel, trainable network that processes conditioning signals and modulates the main denoising backbone. It works by branching oﬀ from intermediate layers in the main U-Net and processing a control input in parallel. Its outputs are then added to the original U-Net features before the denoising step, eﬀectively steering the generation without retraining the base model.

Our method advances prior research by combining semantic control (crash type) and spatial control (bounding box) through CFG and via the ControlNet adapter. This integration enables both precise descriptive control and generative reasoning about critical driving outcomes.

Car Crash Simulation and AV Safety. Video diﬀusion models oﬀer a compelling solution for car crash simulation, as they can simulate both the visual realism and behavioral dynamics of complex driving scenes with very little user eﬀort. Recent car video generation models [17, 5, 18] focus on structured driving video generation using control signals like bounding boxes or text prompts. These models generate temporally consistent driving scenes and support the synthesis of structured traﬃc interactions. However, they are typically limited to noncrash scenarios or coarse control.

Physics-based driving simulators [2] are useful tools for evaluating autonomous vehicle safety [19, 20, 21]. Due to the scarcity and ethical challenges of real-world crash data, there is growing interest in generating synthetic safety-critical scenarios [21, 22]. These simulations provide control and physical realism but often lack visual authenticity and broad applicability. In contrast, our work focuses on rare, adversarial cases that challenge perception and planning systems in real-world scenarios. To address the data challenge, we propose a method for processing car crash videos.

While most prior work focus on general driving scenes, notable examples of prior work tackling realistic car crash generation using real videos include:

DrivingGen [23], which generates crash videos from textual accident reports, AVD2 [24] and OAVD [10] which also generate videos of car crashes, but their main focus is car crash video description rather than generation. These methods are designed to stress-test AV policies and enhance safety coverage beyond what real-world datasets oﬀer.

Our work builds on this vision by proposing a controllable video diﬀusion model that can generate crash outcomes directly from initial conditions and desired outcomes. Unlike physics simulators, Ctrl-Crash supports semantic control (e.g., crash type) and spatial trajectory speciﬁcation (e.g., bounding boxes), enabling targeted stress testing of vision-based AV stacks. Moreover, our method supports counterfactual safety reasoning, allowing one to explore how small changes in agent behavior or intent could lead to dramatically diﬀerent outcomesa critical capability for understanding near-miss events and decision boundary failures. Importantly, our method has much higher visual and motion ﬁdelity than prior works.

#### 3. Our Method: Ctrl-Crash

In this section, we present Ctrl-Crash, our controllable video diﬀusion framework for generating crash scenarios from a single image. We describe the overall architecture 3.1, data processing pipeline 3.2, conditioning mechanisms 3.3, training strategy 3.4, and our extension of classiﬁer-free guidance for ﬁne-grained control 3.5.

3.1. Overview

We propose Ctrl-Crash (Figure 2), a controllable video diffusion framework designed to generate realistic car crash scenarios from a single initial frame, guided by both spatial and semantic control signals. Ctrl-Crash builds on Ctrl-V [5], a framework for generating videos from rendered bounding box trajectories, by extending its capabilities to crash-speciﬁc scenarios, oﬀering richer control and greater ﬂexibility. Speciﬁcally, we incorporate a new semantic control signal representing crash type and introduce a reﬁned training procedure to handle partial and noisy conditioning.

Our method follows a two-stage training pipeline. In the ﬁrst stage, we ﬁne-tune the Stable Video Diﬀusion (SVD) model on the MM-AU [10] crash video dataset to improve its ability to synthesize dynamic and physically plausible driving and accident scenes. In the second stage, we train a ControlNet module to inject conditioning information in two forms: (1) bounding box sequences representing road user motion across time, and (2) discrete crash type labels encoding high-level semantic intent. To ensure generalization to incomplete or noisy control, we introduce a curriculum-based random masking strategy that progressively masks out parts of the control inputs during training. Masked bounding box frames are replaced by a learnable embedding that preserves scene plausibility. We further extend classiﬁer-free guidance to support independent scaling of each control modality, enabling nuanced and ﬂexible control at inference. Unconditional noise predictions are obtained from the pretrained base model for improved generation diversity and stability.

Ctrl-Crash supports three task settings, each enabled by varying the available control signals: (1) Crash Reconstruction: Given an initial image, full bounding box sequence, and a crash type, the model reconstructs a consistent video combining the visual context of the initial frame with agent motion derived from the bounding boxes. (2) Crash Prediction: Given the initial frame and either none or a few initial bounding box frames (e.g., 09), the model predicts the future motion of agents in a way that aligns with the target crash type. (3) Crash Counterfactuals: Extending the prediction task, this mode varies the crash type signal while keeping other inputs ﬁxed, enabling the generation of multiple plausible outcomes for the same scene, supporting counterfactual safety reasoning (Figures 1 and 3). We use the term “counterfactual” here in an informal sense, not to imply a formal causal model, but rather to describe alternate plausible outcomes from the same initial conditions. For example, Ctrl-Crash can be applied to non-crash datasets such as BDD100K, allowing the generation of realistic crash outcomes

[Figure 9]

- Figure 2: Ctrl-Crash architecture: Ctrl-Crash treats Bounding Boxes (BBs) as images. Both BBs and images frames are put through a VAE image encoder. The crash type and BB embeddings are fed to ControlNet. The images embedding after adding noise (xt), the noise level (t), and the ControlNet intermediate outputs (c) are fed to the Stable Video Diﬀusion (SVD) model to obtain the predicted noise ϵθ(xt,t,c). CLIP embeddings [25] are computed by passing the ﬁrst image for each video through a pretrained CLIP encoder. These CLIP embeddings are then given to the ControlNet and SVD models. The diﬀusion process is solved over multiple steps using classiﬁer-free guidance in the latent space, and then decoded back to images using the VAE image decoder. See Section 3 for details.

from otherwise uneventful driving scenes (see Appendix A.2 for examples).

- 3.2. Data Preparation

Processing and preparing crash data is an essential element of our approach, enabling the construction of control signals from a diverse set of car crashes captured by dashboard cameras and sourced from online videos.

Video Processing. For training, we use the MM-AU dataset, a large-scale collection of dashcam crash videos collected from online sources. To ensure high quality we curate this dataset by applying a series of ﬁltering steps. This includes automated detection and ﬁltering of low resolution videos, shot change detection with PySceneDetect, and manual exclusion of scenes involving visible humans getting struck. We manually exclude scenes where visible humans are hit to avoid exposing the model to violent content and to prevent it from learning to depict human injury, reducing the risk of harmful or inappropriate generations post-deployment. Additionally, we standardize clips to 25-frame segments at 6 fps and 512 × 320 resolution. After the ﬁltering steps, we retain approximately 7,500 videos from the original 11,727 videos. We split these videos into a training set and a held-out test set by randomly sampling by accident type categories with a 90/10 ratio. For more details on the precise processing steps and the dataset, please refer to Appendix C and Appendix H.

Bounding Box Extraction Pipeline. To obtain reliable bounding box annotations for all road users, we design a hybrid pipeline that combines detection and segmentation models. For detection, we use YOLOv8 [26] for frame-by-frame object detection. YOLO provides class-speciﬁc bounding boxes with high conﬁdence and speed. For tracking, we use SAM2 [27] to produce instance-level masks and reliable tracking particularly when objects get occluded or deformed, common in crash videos. This combined approach yields temporally aligned

bounding boxes across all video frames. Importantly, it supports agents that enter or exit the scene dynamicallycritical for realistic dynamic driving scenarios. A full breakdown of the bounding-box annotation pipeline is provided in Appendix D. We collect bounding-box annotations for all road users for the MM-AU dataset and the RUSSIA Car Crash Dataset [11]. These annotations along with the processed video frames will be made publicly available on our project page.

3.3. Conditioning Signals

Ctrl-Crash generates videos from three complementary conditioning signals that guide both the visual realism and semantic plausibility of the crash scenario (as illustrated in Figure 2):

Initial Frame (Scene Context): A single starting image grounds the generation process, providing the model with the appearance, layout, and environment of the driving scene before the crash occurs. It is encoded using a pretrained visual encoder (VAE) and fed into the main diﬀusion model as context.

Bounding-Box Trajectories (Spatial Control): The motion of road users is guided by sequences of 2D bounding boxes, which encode object positions, dimensions, unique track ID (via ﬁll color), and classes (via border color) over time. These control frames are encoded using the same pretrained VAE as the initial image and processed by a ControlNet and injected into the denoising process to allow the model to follow speciﬁc trajectories and react to interactions in a physically plausible way. See Appendix E for a visualization.

Crash Type Label (Semantic Control): Crash types are represented by a discrete class label from ﬁve categories: (0) none, (1) ego-only, (2) ego/vehicle, (3) vehicle-only, and (4) vehicle/vehicle. These indicate which agents are involved in the crash, with, for example, vehicle/vehicle describing a crash between two non-ego agents (ego being the vehicle from which we observe the scene from). The crash type index is embedded and projected into the cross-attention layers of the ControlNet

encoder, allowing the model to generate outcomes consistent with high-level semantic intent.

- 3.4. Training Pipeline Our training strategy is divided into two phases. In the ﬁrst

stage, we ﬁne-tune the SVD model on crash-heavy video data from our preprocessed version of the MM-AU dataset (see Section 3.2). The setup is an image-to-video task with an MSE loss in latent space (Equation 1). This improves the models ability to generate plausible driving videos, including crashes, though without explicit controllability. In the second stage, we freeze the ﬁne-tuned base model and train a ControlNet adapter module with the same objective to introduce conditional controllability. This stage allows us to direct the generation using spatial and semantic control signalsnamely, bounding box frames and crash type identiﬁers. Additional training and implementation details are provided in Appendix G for reproducibility.

Conditioning Masking. To improve robustness and support ﬂexible guidance at inference, we apply randomized masking of control signals during training. For bounding boxes, we use a temporal dropout strategy: a random timestep k is sampled, and all subsequent frames are replaced with a learnable null embedding to avoid misinterpreting missing input as the absence of agents. A curriculum schedule gradually increases the masking ratio to encourage generalization to partial control. For semantic inputs (initial image and crash type), we apply joint masking with a ﬁxed probability schedule: with 10% chance only the image is masked, 10% only the crash type, and 10% bothencouraging balanced reliance across modalities and enabling eﬀective classiﬁer-free guidance.

- 3.5. Multi-Condition Classiﬁer-Free Guidance Ctrl-Crash supports three conditioning modalities: initial im-

age cI, bounding box frames cB, and crash type cT. To enable independent control over each during inference, we extend classiﬁer-free guidance (CFG) following [28, 16], and denote the ﬁnal noise estimate under multi-condition CFG as ϵˆθ,ϕ:

ϵˆθ,ϕ(xt,cI,cB,cT) = ϵϕ(xt,cI,∅,∅)

+ γB [ϵθ(xt,cI,cB,∅) − ϵϕ(xt,cI,∅,∅)]

(3)

+ γT [ϵθ(xt,cI,cB,cT) − ϵθ(xt,cI,cB,∅)]

Unlike standard CFG (Equation 2), which uses the same model for both conditional and unconditional predictions, we follow [29] and decouple these roles: ϵϕ (the oﬀ-the-shelf SVD model) predicts the unconditional term, while ϵθ (the ﬁne-tuned Ctrl-Crash model) handles conditional terms. This avoids degraded unconditional priors and improves conditional ﬁdelity. The coeﬃcients γB and γT modulate the strength of bounding box and crash type conditionings, respectively, enabling ﬂexible and interpretable control at inference.

#### 4. Experiments

- 4.1. Quantitative Evaluation We evaluate the generation quality of Ctrl-Crash across two

core settings: (1) general crash video generation in Table 1;

Table 1: Comparison of accident video generation quality across diﬀusionbased methods. We report FVD and JEDi scores (↓ lower is better). Scores marked with * are taken directly from the original papers and may not be strictly comparable due to diﬀerences in evaluation setup. The "Conditions" column describes inputs used as control signals. "img": initial image frame, "BB": bounding box frames, "text": textual description, "action": discrete crash category value. See Figure 4 for visuals.

#### Method Conditions FVD↓ JEDi↓

OAVD img + BB + text 5238* DrivingGen img + BB + text 978.0* -

SVD base img 1420 3.628 AVD2 img + text 1321 2.029 Ctrl-V img + BB 517.1 0.2910 Ctrl-Crash img + BB + action 449.5 0.1219

and (2) varying the number of bounding box guidance frames provided in the prediction task to assess the impact of partial trajectory information in Table 2. All generated videos are 25 frames long at a resolution of 512 × 320, and metrics are computed over 200 samples unless otherwise speciﬁed.

[Figure 10]

Figure 3: Counterfactual Crash Generation: this diagram demonstrates the ability of our model to generate counterfactual crashes (Middle: no crash, Bottom: ego/car crash) while beginning from the initial frame and 3 boundingboxes frames of the real video (Top: the real car crash).

We evaluate generation quality using several video and frame-level metrics. Fréchet Video Distance (FVD) [30] measures the similarity between the distributions of generated and real videos by analyzing video embeddings in I3D space under the assumption of Gaussian distributions; lower FVD values indicate a closer resemblance to real data. JEDi [31] is a novel metric designed as an alternative to FVD by addressing its limitations through relaxing the assumption of Gaussiandistributed video feature embeddings, modifying the embedding space, and enabling convergence with a smaller sample size. In addition to these distributional-based video metrics, we report LPIPS (Learned Perceptual Image Patch Similarity) [32], SSIM (Structural Similarity Index), and PSNR (Peak Signal-toNoise Ratio) in Table 2, which evaluate frame-level ﬁdelity and perceptual closeness relative to ground truth, averaged over the video frames. These metrics provide complementary insights into how well the generated videos preserve appearance, struc-

ture, and temporal dynamics.

As shown in Table 1, Ctrl-Crash achieves the best results across both FVD and JEDi among compared methods, indicating strong alignment with real crash dynamics and superior video quality. For fair comparison, we compute the ground truth (GT) distribution for FVD and JEDi from a set of 500 randomly sampled MM-AU [10] validation videos, to accommodate for methods without GT alignment (e.g., AVD2) (see Appendix F in supplementary material for more details). The SVD Base model, which serves as the foundation for Ctrl-Crash, performs poorly, as it was not trained for driving or crash-related content. Ctrl-V, while similar in architecture, lacks crashspeciﬁc training data and semantic control, leading to notable quality degradation near the crash event. AVD2 performs moderately well but exhibits inconsistent visual quality and weaker temporal coherence compared to Ctrl-Crash, as conﬁrmed by FVD and JEDi. These results highlight the importance of both targeted training and structured control for crash simulation.

Table 2: Eﬀect of bounding box conditioning on crash video prediction quality. We evaluate Ctrl-Crash on the Crash Prediction task by varying the number of initial bounding box frames provided as input (out of 25 total frames).

#### #BBs FVD↓ JEDi↓ LPIPS↓ SSIM↑ PSNR↑

0 422.1 0.3155 0.3856 0.5188 16.57 3 375.7 0.2949 0.3594 0.5434 17.27 9 353.3 0.2160 0.3392 0.5614 17.83

#### 25 323.9 0.1219 0.3113 0.5836 18.33

We also study the impact of varying the number of bounding box frames used as conditioning for Ctrl-Crash, in Table 2. Speciﬁcally, we compare using partial bounding-box frame conditioning from zero frames to a few bounding box frames (Crash Prediction task) all the way to a fully deﬁned motion sequence by providing all 25 frames (Crash Reconstruction task). As shown in Table 2, generation quality improves consistently with the number of provided bounding box frames. This trend is visible across both distributional metrics (FVD and JEDi) and frame-level scores (LPIPS, SSIM, PSNR), and supports the hypothesis that denser spatial constraints lead to easier prediction tasks and more stable outputs. The results validate that Ctrl-Crash gracefully interpolates between unconditional prediction and fully supervised reconstruction.

Across all benchmarks, Ctrl-Crash delivers signiﬁcant improvements over prior diﬀusion-based crash generation models. It handles both unconstrained and highly conditioned inputs, demonstrating its utility for generating diverse crash outcomes and precise reconstructions. In the next section, we complement these quantitative results with a user study and qualitative visualizations.

Crash?

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Ctrl-CrashDrivingGenCtrl-VAVD2

✅

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

✅

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

❌

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

✅

- Figure 4: Qualitative comparison of AVD2, DrivingGen, Ctrl-V, and CtrlCrash. Ctrl-Crash achieves superior visual ﬁdelity and realistic crash dynamics, outperforming baselines that suﬀer from low quality, inconsistency, or lack of plausible crash events.

consistently struggle to depict physically plausible accidents. In our experiments, Cosmos and Sora failed to generate convincing crashes across several prompt variations, while Veo3, which we tested with a single sample, showed limited realism despite momentarily resembling a crash (see Appendix A.2 for exact prompts). These examples, shown in Figure 5, suggest that current general-purpose models lack the exposure, the ﬁne-grained control, and physical grounding required for realistic accident simulation.

Veo3SoraCosmos

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- Figure 5: State-of-the-art text-to-video diﬀusion models struggle to generate a realistic car accident. Top: Nvidia Cosmos (7B Text2World), Middle: OpenAI Sora, Bottom: DeepMind Veo3.

Additionally, we conducted a brief user study to evaluate visual quality and physical realism of the generated videos based on human preference. Results show clear and signiﬁcant preference for Ctrl-Crash among similar methods. Results and details are presented in Appendix I.

- 4.2. Qualitative Evaluation

We qualitatively tested leading video diﬀusion modelsOpenAI’s Sora [33], Nvidia’s Cosmos [18], and DeepMind’s Veo3 [34]for their ability to generate car crash scenarios. While these models produce high-resolution, cinematic visuals, they

#### 5. Conclusion

In this work, we introduced Ctrl-Crash, a controllable video diﬀusion framework that generates realistic car crash scenarios from a single frame, which achieves state-of-the-art performance among diﬀusion-based methods designed for crashes,

and enables counterfactual reasoning by varying spatial and semantic control inputs. To support training and evaluation, we also developed a processing pipeline for extracting bounding boxes from crash videos and released curated, annotated versions of MM-AU, RussiaCrash, and BDD100k to facilitate future research in crash simulation and generative modeling.

Despite its strong performance, our approach has several limitations, motivating promising directions for future work. Counterfactual outcomes can be diﬃcult to generate when the initial scene strongly suggests a diﬀerent crash outcome than the one being conditioned on. The model also relies heavily on 2D bounding boxes, which makes it sensitive to tracking errorsparticularly in fully conditioned reconstruction. Moreover, 2D boxes often lack information about agent orientation and rotation, limiting realism for behaviors such as spinouts or rollovers. Future extensions may explore 3D bounding boxes or richer trajectory representations to overcome this. Another limitation is the use of a ﬁxed set of ﬁve discrete crash type labels to deﬁne semantic intent. While simple and eﬀective, this formulation constrains expressiveness. Incorporating natural language as a conditioning signal could unlock more nuanced control and allow the model to interpret detailed scene descriptions or intentions that go beyond predeﬁned categories and bounding-box information. We envision Ctrl-Crash as a foundation for building more general, interpretable, and controllable generative models for safety-critical autonomous driving research.

Acknowledgments: We thank Samsung, the IVADO and the Canada First Research Excellence Fund (CFREF) / Apogée Funds, the Canada CIFAR AI Chairs Program, and the NSERC Discovery Grants program for ﬁnancial support. We also thank Mila - the Quebec AI Institute for compute resources.

#### References

- [1] T. Wang, S. Kim, W. Ji, E. Xie, C. Ge, J. Chen, Z. Li, P. Luo, Deepaccident: A motion and accident prediction benchmark for v2x autonomous driving (2023), arXiv:2304.01168.
- [2] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, V. Koltun, CARLA: An open urban driving simulator, in: Proceedings of the 1st Annual Conference on Robot Learning, 2017, pp. 1–16.
- [3] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, O. Beijbom, nuscenes: A multimodal dataset for autonomous driving, CoRR abs/1903.11027 (2019), arXiv:1903.11027.
- [4] W. Wu, X. Guo, W. Tang, T. Huang, C. Wang, D. Chen, C. Ding, Drivescape: Towards high-resolution controllable multi-view driving video generation (2024), arXiv:2409.05463.
- [5] G. Y. Luo, Z. Luo, A. Gosselin, A. Jolicoeur-Martineau, C. Pal, Ctrl-v: Higher ﬁdelity autonomous vehicle video generation with bounding-box controlled object motion, Transactions on Machine Learning Research

(2025).

- [6] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, Highresolution image synthesis with latent diﬀusion models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695.
- [7] J. Ho, T. Salimans, Classiﬁer-free diﬀusion guidance (2022), arXiv: 2207.12598.
- [8] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, V. Jampani, R. Rombach, Stable video diﬀusion: Scaling latent video diﬀusion models to large datasets (2023), arXiv:2311.15127.
- [9] L. Zhang, A. Rao, M. Agrawala, Adding conditional control to text-toimage diﬀusion models (2023), arXiv:2302.05543.

- [10] J. Fang, L.-l. Li, J. Zhou, J. Xiao, H. Yu, C. Lv, J. Xue, T.-S. Chua, Abductive ego-view accident video understanding for safe driving perception, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22030–22040.
- [11] Sivoha, Car crash dataset russia 2022–2023, Kaggle Dataset, accessed: 2025-05-15 (2023).
- [12] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, T. Darrell, Bdd100k: A diverse driving dataset for heterogeneous multitask learning (2020), arXiv:1805.04687.
- [13] J. Ho, A. Jain, P. Abbeel, Denoising diﬀusion probabilistic models

(2020), arXiv:2006.11239.

- [14] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, D. J. Fleet, Video diﬀusion models, in: S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, A. Oh (Eds.), Advances in Neural Information Processing Systems, Vol. 35, Curran Associates, Inc., 2022, pp. 8633–8646.
- [15] Y. He, T. Yang, Y. Zhang, Y. Shan, Q. Chen, Latent video diﬀusion models for high-ﬁdelity long video generation (2023), arXiv:2211.13221.
- [16] P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros, Image-to-image translation with conditional adversarial networks (2018), arXiv:1611.07004.
- [17] G. Zhao, X. Wang, Z. Zhu, X. Chen, G. Huang, X. Bao, X. Wang, Drivedreamer-2: Llm-enhanced world models for diverse driving video generation, in: Proceedings of the AAAI Conference on Artiﬁcial Intelligence, Vol. 39, 2025, pp. 10412–10420.
- [18] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al., Cosmos world foundation model platform for physical ai, arXiv preprint arXiv:2501.03575 (2025).
- [19] L. Rowe, R. Girgis, A. Gosselin, B. Carrez, F. Golemo, F. Heide, L. Paull, C. Pal, Ctrl-sim: Reactive and controllable driving agents with oﬄine reinforcement learning, 2025, arXiv:2403.19918.
- [20] S. Dupuy, A. Egges, V. Legendre, P. Nugues, Generating a 3d simulation of a car accident from a written description in natural language: The carsim system, arXiv preprint cs/0105023 (2001).
- [21] T. Wang, S. Kim, J. Wenxuan, E. Xie, C. Ge, J. Chen, Z. Li, P. Luo, Deepaccident: A motion and accident prediction benchmark for v2x autonomous driving, in: Proceedings of the AAAI Conference on Artiﬁcial Intelligence, Vol. 38, 2024, pp. 5599–5606.
- [22] Z. Huang, X. Gao, G. Zheng, L. Wen, X. Yang, X. Sun, Safety-critical traﬃc simulation with adversarial transfer of driving intentions, arXiv preprint arXiv:2503.05180 (2025).
- [23] Z. Guo, Y. Zhou, C. Gou, Drivinggen: Eﬃcient safety-critical driving video generation with latent diﬀusion models, in: 2024 IEEE International Conference on Multimedia and Expo (ICME), 2024, pp. 1–6, doi:10.1109/ICME57554.2024.10688119.
- [24] C. Li, K. Zhou, T. Liu, Y. Wang, M. Zhuang, H.-a. Gao, B. Jin, H. Zhao, Avd2: Accident video diﬀusion for accident video description, arXiv preprint arXiv:2502.14801 (2025).
- [25] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision

(2021), arXiv:2103.00020.

- [26] G. Jocher, J. Qiu, A. Chaurasia, Ultralytics YOLO (Jan. 2023).
- [27] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, C. Feichtenhofer, Sam 2: Segment anything in images and videos (2024), arXiv:2408.00714.
- [28] N. Liu, S. Li, Y. Du, A. Torralba, J. B. Tenenbaum, Compositional visual generation with composable diﬀusion models (2023), arXiv:2206. 01714.
- [29] P. Phunyaphibarn, P. Y. Lee, J. Kim, M. Sung, Unconditional priors matter! improving conditional generation of ﬁne-tuned diﬀusion models

(2025), arXiv:2503.20240.

- [30] T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, S. Gelly, FVD: A new metric for video generation (2019).
- [31] G. Y. Luo, G. M. Favero, Z. Luo, A. Jolicoeur-Martineau, C. Pal, Beyond FVD: An enhanced evaluation metrics for video generation distribution quality, in: The Thirteenth International Conference on Learning Representations, 2025.
- [32] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, The unreasonable eﬀectiveness of deep features as a perceptual metric (2018), arXiv:1801.03924.
- [33] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo,

- OpenAI, Video Generation Models as World Simulators (2024), https://openai.com/index/ video-generation-models-as-world-simulators/. Technical report.
- [34] DeepMind, Veo 3: Google’s Text-toVideo Model with Native Audio

(2025), https://deepmind.google/models/veo/. Released May 2025; model card published May 23, 2025.

- [35] C. A. Castellano, PySceneDetect: Video Scene Detection in Python (2021), Version 0.6, https://github.com/Breakthrough/ PySceneDetect. Accessed: 2025-07-29.
- [36] J. Fang, D. Yan, J. Qiao, J. Xue, H. Yu, DADA: Driver Attention Prediction in Driving Accident Scenarios (2023), arXiv:1912.12148.
- [37] M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance (1937), Journal of the american statistical association, Vol. 32, No. 200, pp. 675–701. Taylor & Francis.
- [38] P. B. Nemenyi, Distribution-free multiple comparisons. (1963), Princeton University.

#### Appendix A. Additional Results

- Appendix A.1. Additional Quantitative Results: Ablations

We present in Table A.3 additional results to complement results given in Table 1 of the main paper. In addition to FVD [30] and JEDi [31] values, we present LPIPS [32], SSIM, and PSNR for Ctrl-Crash and its ablations. Ctrl-V is similar to Ctrl-Crash but has not been trained on crash data and was not designed with dynamic crash video generation in mind. SVD base is the base model Ctrl-Crash is derived from through ﬁnetuning. For CtrlCrash and Ctrl-V the full sequence of bounding boxes is given, so we are expecting a reconstruction similar to the ground-truth crash. Across all metrics, we see that Ctrl-Crash performs the best at reconstructing plausible crashes.

Table A.3: Generated accident video quality compared to baseline ablations methods

Method Conditions FVD↓ JEDi↓ LPIPS↓ SSIM↑ PSNR↑

SVD base img 1420 3.628 0.5800 0.3074 11.74 Ctrl-V img + bbox 517.1 0.2910 0.3670 0.5372 16.81 Ctrl-Crash (Ours) img + bbox + action 449.5 0.1219 0.3113 0.5836 18.33

Table A.4: Eﬀect of crash type conditioning on crash video generation quality. We evaluate Ctrl-Crash on the Crash Counterfactuals task by varying the crash type conditioning and nothing else. For each case, 200 videos were generated using the same initial image and three bounding box frames as initial context, but with diﬀerent desired crash types.

Crash Type FVD↓ JEDi↓ GT crash type 375.7 0.2949

- 0 - no crash 400.9 0.4514
- 1 - ego-only 379.5 0.3091
- 2 - ego/vehicle 372.9 0.3001
- 3 - vehicle-only 398.1 0.3856
- 4 - vehicle/vehicle 383.4 0.3416

In Table A.4, we evaluate Ctrl-Crash on the Crash Counterfactuals task by varying only the crash type while keeping the initial image and partial bounding box sequence ﬁxed. The model achieves its best performance on ego/vehicle crashes (type 2), likely due to their higher frequency in the training data, while vehicle-only crashes (type 3) are harder to generate plausibly, reﬂecting their visual complexity. Interestingly, ego-only crashes (type 1), though underrepresented in the dataset, also perform wellpossibly because they involve simpler dynamics without visible impact. Overall, counterfactual generations remain close in quality to the ground-truth-guided baseline, suggesting that Ctrl-Crash can produce realistic alternate outcomes, though performance declines when strong visual cues in the initial frame conﬂict with the intended crash type. Class frequency in the training set are provided in Appendix Appendix H.

- Appendix A.2. Additional Qualitative Results

SOTA models. State-of-the-art diﬀusion-based video models, such as OpenAI’s Sora [33], Nvidia’s Cosmos [18], and DeepMind’s Veo3 [34] fail to generate plausible crash scenarios. We tested these three models with several samples and though they tend to generate high deﬁnition and smooth video, they consistently fail to generate physically plausible crashes (if any crash at all), we show some examples in Figure A.6.

In the top row of the ﬁgure we show a sample result from Nvidia Cosmos-Predict1-7B-Text2World that was instructed with the following text prompt: "On a highway two cars collide at very fast speeds head-on". This produces a highly implausible scene, where the car on the left suddenly starts to levitate from its rear-end and a cloud of smoke resembling an explosion appears, followed by disjointed fragments of torn metal emerging from the ground that transforms in a dark vehicle rushing towards the camera.

The Nvidia Cosmos model suite has been speciﬁcally trained for physics-aware generation, but still fail to generate a car crash. Alternatively, we show a sample from OpenAI’s Sora video model in Figure A.6 in the middle row. The Sora video model was given the prompt: "At an intersection two cars collide with each other at full speed resulting in a crash". The resulting video shows a car that spins erratically, changes shape and direction, and produces visible artifactsculminating in a pile of twisted metal and glass.

Finally, we show an example using DeepMind’s Veo3 in the bottom row of Figure A.6. We used the following prompt: "At an intersection two cars collide with each other at full speed resulting in a crash". Perhaps the closest to depicting a car crash when looking at individual frames, but when watching the video as a whole it becomes

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

### Veo3CosmosSora

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

- Figure A.6: State-of-the-art text-to-video diﬀusion models struggle to generate a realistic car accident (as shown in main paper). Top: Nvidia Cosmos (7B Text2World), Middle: OpenAI Sora, Bottom: DeepMind Veo3.

clear that the events are completely unrealistic. In the generated video, we see two still vehicles very close to each other that start accelerating, then we see debris and smoke as the cars slowly move towards each other, then the front of the cars phase through each other and reveal completely undamaged cars, and ﬁnally the red car drives away from the scene.

In contrast, through our approach of sourcing crash data, preprocessing and annotating sequences with bounding boxes, crash type conditioning, and our stochastic conditioning approach, Ctrl-Crash can much more reliably generate physically plausible crash imagery and collision dynamics.

Generating Crashes from Non-Crash Data. To demonstrate the practical utility of Ctrl-Crash for safety-critical dataset augmentation, we apply our model to generate crash scenarios seeded from the BDD100k dataset, an established real-world driving dataset that contains no actual crashes. By conditioning on initial frames and coarse agent trajectories from BDD100k, Ctrl-Crash is able to hallucinate plausible accident outcomes, transforming otherwise uneventful driving clips into diverse crash scenarios. This capability is especially valuable for enriching datasets used in autonomous driving research, where real crash data is scarce, sensitive, or ethically challenging to collect. Synthesizing rare and hazardous events from benign scenes enables safer training and evaluation of perception and planning systems without requiring exposure to real-world danger. In Figure A.7, we show some examples of car crash videos generated by Ctrl-Crash from non-crash scenes in the BDD100k dataset.

Car Crash Video Diﬀusion Models Comparison. Other recent work [23, 24] focus on the task of car crash video generation, but lack the visual quality for convincing car crashes. As mentioned in the main paper (and presented in Appendix Appendix I), we conducted a user survey that shows that Ctrl-Crash signiﬁcantly outperforms these existing methods for visual ﬁdelity and physical plausibility of generated car crashes. We show additional qualitative comparisons in Figure A.8 and Figure A.9, along with many samples generated from Ctrl-Crash in Figure A.10, and Figure A.11.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

💥

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

💥

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

💥

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

💥

- Figure A.7: Crash scenarios generated by Ctrl-Crash from non-crash scenes in the BDD100k dataset. Each row shows 5 frames from a generated 25-frame clip. Despite originating from benign driving scenes, the model produces visually coherent and diverse crash outcomes across examples.

Crash?

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

|❌<br><br>✅<br><br>✅<br><br>❌<br><br>❌<br><br>✅|
|---|

SVDAVD2DrivingGenCosmosCtrl-VCtrl-Crash(ours)

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

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

- Figure A.8: Qualitative comparison of "rear-end crashes" between diﬀerent methods. For each method, we show 5 frames from the video along with either a green check mark if there appears to have a crash in the video otherwise a red ’X’. From top to bottom: SVD (stable-videodiﬀusion-img2vid) [8] prompted with the initial frame from a rear-end crash video, we see some normal driving but very inconsistent lighting and color shades with visible distorsions. AVD2 [24], we see what appears to be a rear-end crash with a very distorted leading vehicle and background. DrivingGen [23], we see a rear-end crash with a leading vehicle that changes appearance every frame. Overall the video is very choppy with little temporal consistency. Cosmos (Cosmos-Predict1-7B-Video2World) [18], prompted with text suggesting a rear-end crash and 9 initial images where a car is rapidly approaching a truck, the predicted frames show the car unrealistically shrinking as it approaches the truck without any signs of a collision. Ctrl-V [5], prompted with a sequence of bounding-boxes suggesting a rear-end crash with a leading car, we see the leading car keep its distance and not crash occurs. Ctrl-Crash (ours): prompted with the same bounding box sequence as Ctrl-V and with the discrete crash type "ego/vehicle crash", we see a physically plausible rear-end collision with the ego vehicle visibly shaking from the impact.

Crash?

|❌<br><br>✅<br><br>✅<br><br>❌<br><br>❌<br><br>✅|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

SVDAVD2DrivingGenCtrl-Crash(ours)Ctrl-VCosmos

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

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

- Figure A.9: Qualitative comparison of "t-bone crashes" between diﬀerent methods. For each method, we show 6 frames from the video along with either a green check mark if there appears to have a crash in the video otherwise a red ’X’. From top to bottom: SVD (stable-videodiﬀusion-img2vid) [8] prompted with the initial frame from a t-bone car crash video, we see blurry vehicle a distorted motion blur as it drives in front of the ego vehicle without any collision. AVD2 [24], we can make out what seems to be a t-bone crash with a heavily distorted black car. There are many artifacts and temporal inconsistencies which makes the sequence of events hard to follow. DrivingGen [23], a gray sedan drive in front of the ego vehicle progressively getting closer until it seems to collide with it. Motion is jerky and uneven between timesteps and the appearance of the gray car shapes almost every frame. Cosmos (Cosmos-Predict1-5B-Video2World) [18], prompted with creating a t-bone crash and 9 initial frames showing a car turn in front of the ego car, we see the leading car start to distort as the ego approaches it and then it shrivels and shrinks until it nearly disappears. Ctrl-V [5], prompted with a sequence of bounding-boxes suggesting a t-bone crash with a car incoming from the left, we see a car drive in from the left and then just passed the ego car without any collision. Ctrl-Crash (ours): prompted with the same bounding box sequence as Ctrl-V and with the discrete crash type "ego/vehicle crash", we see a physically plausible t-bone collision with the a black sedan incoming from the left.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

💥

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

💥

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

💥

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

💥

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

💥

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

💥

- Figure A.10: Ctrl-Crash qualitative results conditioned on an initial 9 bounding box frames (i.e., the ﬁrst two frames of these sequences were conditioned on bounding box frames, but not the others).

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

💥 💥

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

💥

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

💥

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

💥

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

💥

Figure A.11: Ctrl-Crash qualitative results conditioned on 25 (all) bounding box frames.

#### Appendix B. Low Resolution Filtering Heuristic

Algorithm 1 Estimate Upscaling Factor from Image

- 1: procedure EstimateUpsizingFactor(image_path)
- 2: Load image in grayscale: I ← cv2.imread(image_path, GRAYSCALE)
- 3: Compute 2D FFT and shift: F ← np.fft.fftshift(np.fft.fft2(I))
- 4: Compute magnitude spectrum: M ← ||F||
- 5: Get image size: (h,w) ← shape(I)
- 6: Center: (cx,cy) ← (w//2,h//2)
- 7: Deﬁne low-frequency radius: r ← min(cx,cy)//4
- 8: Create circular mask L of radius r centered at (cx,cy)
- 9: Compute high-frequency mask: H ← 1 − L
- 10: Compute high-frequency energy: Ehigh ← ∑

M · H

- 11: Normalize energy: e ← Ehigh/(h · w)
- 12: Compute upscaling factor: f ← 1/(1 + e)
- 13: return f
- 14: end procedure

A key challenge in curating video datasets from online sources is ensuring that the selected samples reﬂect genuine high-resolution content rather than upscaled of heavily compressed footage (via resizing). This is particularly relevant for dashcam videos, where visual clarity and high-frequency details are critical for learning accurate dynamics and semantics. As part of the video preprocessing pipeline (see Section Appendix C), we introduce a frequency-based ﬁltering heuristic to estimate the degree of upscaling in video frames.

The heuristic is based on the observation that upscaled or low-quality images tend to exhibit diminished highfrequency content due to interpolation artifacts and smoothing. By analyzing the energy distribution in the frequency domain, we deﬁne an upsizing factor that acts as a proxy for the likelihood of artiﬁcial upscaling. The method proceeds as follows:

- 1. Image Loading. Each video frame is converted to grayscale to simplify analysis and focus on structural image content rather than color channels.
- 2. Fourier Transform. A 2D Fast Fourier Transform (FFT) is applied to the grayscale image to obtain its frequency representation. The spectrum is shifted so that low-frequency components are centered.
- 3. Magnitude Spectrum. The magnitude of the frequency components is computed by taking the absolute value of the FFT coeﬃcients. This magnitude spectrum represents the intensity of diﬀerent spatial frequencies in the image.
- 4. High-Frequency Energy Calculation. A circular mask is applied to exclude the low-frequency region in the center of the spectrum. The sum of magnitudes outside this central region deﬁnes the high-frequency energy.
- 5. Normalization. The high-frequency energy is normalized by the total number of pixels to produce a scaleinvariant energy score.
- 6. Upsizing Factor Estimation. The ﬁnal upsizing factor U is computed as: U = 1+1E, where E is the normalized high-frequency energy. This formulation ensures that frames with low high-frequency energyindicative of potential upscalingyield lower scores.

- 7. Interpretation. The resulting upsizing factor provides a heuristic score: lower values indicate a higher likelihood of artiﬁcial enlargement or blur, while higher values suggest more genuine, detailed high-resolution frames.

This frequency-domain metric serves as a lightweight yet eﬀective automated quality check, helping to ﬁlter out training samples that lack meaningful visual detail. The corresponding implementation is summarized formally in Algorithm 1.

#### Appendix C. Video Processing

Video Processing. To train our model, we use the MM-AU dataset [10], a large-scale collection of in-the-wild dashcam crash videos sourced from public platforms. However, due to the variable quality and content of these videos, we curate the dataset using a multi-stage ﬁltering and preprocessing pipeline. The process can be summarized as follows:

- 1. Filtering out low-quality, compressed, or blocky videos based on the low-resolution heuristics (described in Appendix Appendix B).
- 2. Removing clips with shot changes or unnatural scene cuts using scene detection.
- 3. Normalizing the format: frame rate, resolution, clip length.
- 4. Excluding scenes involving visible humans (e.g., pedestrians, cyclists, motorbikes).

In the ﬁrst step, we apply frequency-domain heuristics to identify and remove videos that suﬀer from excessive compression artifacts or poor resolution. These issues can hinder the models ability to learn coherent motion and object dynamics. We use a Fast Fourier Transform (FFT)-based method to detect videos with large blocky regions or low-frequency dominancesignals of strong compression or blur. This approach, described in Appendix Appendix B, helps prioritize clips with high visual clarity and well-deﬁned agent motion.

Next, we use PySceneDetect [35] to identify and discard videos containing abrupt scene transitions or camera disruptionssuch as sudden viewpoint changes or dashcams that fall mid-recording. Removing such discontinuities ensures more stable temporal consistency within each training clip.

We then normalize all video clips to a consistent resolution of 512 × 320 (width Œ height) and sample them at 6 frames per second. This not only aligns with the spatial and temporal input requirements of our model but also helps crop out unwanted overlays such as watermarks that often appear near video borders. We segment each video into ﬁxed-length clips of 25 frames (approximately 4 seconds), discarding any segments shorter than this threshold.

Finally, for ethical and safety considerations, we exclude all videos depicting visible humans involved in crashes. This includes scenes involving pedestrians, cyclists, or motorcyclists. Our intention is to avoid training the model to depict harmful or distressing scenarios involving vulnerable road users.

In addition to the automated ﬁltering steps, we also perform manual review of many video samples using internal tooling (made available on the project GitHub). This step helps identify edge cases and poor-quality examples that may have bypassed the automated pipeline, ensuring a cleaner and more appropriate training set.

#### Appendix D. YOLO-SAM: Hybrid Bounding Box Annotation Pipeline

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Frame 13 Frame 14 Frame 15 Frame 16

YOLO SAM2

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Frame 23 Frame 24 Frame 25 Frame 26

SAM2

- Figure D.12: Example of bounding box tracking for a vehicle veering oﬀ the road. SAM2 segmentations for the two vehicles are shown in purple and orange. YOLOv8 detects the purple car only starting at frame 16 (light pink box), while SAM2 successfully infers its presence across many more frames by propagating the segmentation bidirectionally in time. This provides more complete and temporally consistent annotations, especially in cases where objects become partially occluded or deformed.

We develop a hybrid annotation pipeline, YOLO-SAM, to generate high-quality bounding boxes for visible road users in crash and driving videos. This combines the speed and class-awareness of YOLOv8 [26] with the instance-level segmentation and temporal propagation capabilities of SAM2 [27]. The hybrid design addresses detection failures due to occlusion, deformation, or sudden motioncommon in crash scenariosand supports dynamic entry/exit of agents.

YOLOv8 is run frame-by-frame to detect standard object classes (e.g., cars, trucks, buses) and assign initial bounding boxes and track IDs. While fast and accurate under normal conditions, YOLO has several shortcomings:

- 1. Duplicate Detections: YOLO can assign multiple detections to the same object over time. We reject predictions if their IoU with previous detections exceeds 0.8.
- 2. Tracking Loss: YOLO may lose an object (e.g., under occlusion) and redetect it with a new track ID. These failures are corrected using SAM2.

To address YOLOs limitations, we use SAM2 to reﬁne and extend YOLO detections:

- 1. Shape Correction: SAM2 provides temporally consistent masks, reducing YOLOs tendency to distort box sizes and shapes.
- 2. Redetection Veriﬁcation: When YOLO redetects a lost object with a new ID, we compare its spatial overlap with SAM2s mask to either restore the original ID or assign a new one.
- 3. Track ID Switches: If YOLO mistakenly assigns an existing ID to a new object, SAM2’s predictions are used to detect the mismatch and correct the ID.
- 4. Early-frame Recovery: SAM2 supports bidirectional propagation, allowing recovery of early frames missed by YOLO. However, to avoid hallucinated presence, we only accept early-frame completions up to a few frames before YOLOs ﬁrst detection, and only when conﬁdence is high.

Figure D.12 shows an example where a vehicle exits the road and YOLO detects it only mid-sequence, while SAM2 successfully tracks it throughout by propagating the mask temporally.

For each YOLO-detected object, SAM2 is prompted to generate per-frame instance masks. These masks are converted to tight bounding boxes, and tracking continuity is enforced by ID consistency checks based on IoU overlap. The result is a per-frame annotation of bounding boxes with consistent track IDs and object classes.

Compared to YOLO or SAM2 alone, the hybrid approach provides signiﬁcantly more complete and robust annotations, crucial for training controllable video generation models like Ctrl-Crash. The full implementation is available in our project repository for reproducibility.

#### Appendix E. Bounding Box Conditioning

[Figure 231]

[Figure 232]

- Figure E.13: Left: example frame from a driving video. Right: associated bounding box frame conditioning generated from our pipeline. Road users are represented as 2D bounding boxes with unique ﬁll colors representing their track ID and speciﬁc border colors representing their class.

To provide spatial guidance to the generative model, we convert the bounding box trajectories of road users into RGB control frames that serve as conditioning input to the ControlNet. An example of such a control frame, alongside its corresponding real image, is shown in Figure E.13. Each control frame encodes the complete set of bounding boxes for a single timestep using a color-coded rasterization scheme that encodes both object identity and semantic class.

Track ID Encoding (Fill Color). Each bounding box is ﬁlled with a unique RGB color derived from its objects persistent track ID. The RGB values are deterministically generated via a hashing function to ensure consistency across frames. RGB values vary within [50,255] for all three color channels. This allows the model to temporally link the same agent across timesteps and learn coherent motion patterns. The use of color ﬁlls avoids the need for explicit ID embeddings and leverages the spatial structure of the image.

Class Encoding (Border Color). To distinguish between diﬀerent types of road users (e.g., cars, trucks, buses, pedestrians, cyclists), we draw a thin border around each bounding box in a class-speciﬁc color. These colors are chosen from a ﬁxed palette (see Table E.5), and the mapping between semantic classes and RGB border values is consistent across all training data . This helps the model diﬀerentiate object behaviors by class, which is particularly useful in crash prediction (e.g., trucks tend to behave diﬀerently from bicycles).

Table E.5: Class encoding color scheme for bounding box border color

###### Class Border Color RGB values

person Blue (0, 0, 255) car Red (255, 0, 0) truck Orange (247, 162, 44) bus Yellow (250, 255, 2) train Green (0, 255, 0) motorcycle Purple (204, 153, 155) bicycle Pink (255, 209, 22)

The ﬁnal control frame is an RGB image of the same resolution as the input video (e.g., 512 × 320) and can represent any number of agents per frame. If not depth information is available, overlapping boxes are drawn in arbitrary order, with later boxes overwriting earlier ones.

These control frames are encoded using the same VAE encoder used for the initial frame. The resulting latent tensor is passed through a ControlNet branch and injected into the U-Net backbone of the diﬀusion model at selected layers during training and inference. This spatial representation allows the model to attend to object motion in a dense and learnable way without requiring symbolic or token-level processing.

This conditioning mechanism supports variable numbers of agents and enables ﬁne-grained motion control over multiple timesteps. It also integrates seamlessly into the denoising process of the diﬀusion model, allowing consistent agent-level motion to be expressed throughout the generated video sequence.

#### Appendix F. FVD and JEDi Metrics Computation

We evaluate the quality of generated videos using both distributional video-level metrics and frame-level ﬁdelity metrics. Fréchet Video Distance (FVD) [30] measures the distance between the distributions of generated and real videos (lower is better). This is done by embedding the two distributions in the feature space of a pretrained Inﬂated 3D ConvNet (I3D) and computing the Fréchet distance under the assumption of a multivariate Gaussian. JEDi [31] addresses limitations of FVD by relaxing the Gaussian assumption, employing features from a Joint Embedding Predictive Architecture (JEPA), and using Maximum Mean Discrepancy (MMD) with a polynomial kernel to improve both temporal sensitivity and sample eﬃciency. Together, these metrics provide complementary views of video realism at the distributional level.

To evaluate distributional similarity between generated and real videos, we compute Fréchet Video Distance (FVD) and JEDi under two complementary evaluation protocols, which diﬀer based on how the ground-truth distribution is constructed:

- 1. Condition-Aligned Evaluation: For models such as Ctrl-Crash, Ctrl-V, and SVD Base, we have access to the ground-truth data used to seed generationnamely, the initial frame, bounding box trajectories, and even crash type category. In this setting, we sample 200 videos from the MM-AU validation set, and generate corresponding videos using their conditioning information. We do not directly compare generated videos to their exact ground-truth counterpart; instead, we treat these 200 ground-truth videos as a reference distribution that is semantically aligned with the generated one (e.g., similar vehicle types, scene layouts, and crash categories). FVD and JEDi are then computed between these two condition-aligned distributions. This setup ensures that the real and generated samples share similar structural priors, making the distributional comparison more meaningful. This method is used for FVD and JEDi results reported in Table 2 and Table 3 of the main paper.
- 2. Random GT Evaluation: Some baselines, such as AVD2, only provide generated videos without access to the original conditioning inputs, preventing us from constructing a semantically aligned ground-truth distribution. To ensure fair comparison in such cases, we instead sample 500 videos randomly from the MM-AU validation set to represent the ground-truth distribution. We then compare this to 200 generated samples from each model (including AVD2, Ctrl-Crash, Ctrl-V, and SVD Base). While this evaluation provides less precise alignment between real and generated content, it allows for inclusion of baselines where seed information is unavailable. This method is used for FVD and JEDi results reported in Table 1 of the main paper.

By using both evaluation protocols, we balance semantic alignment (when possible) with broad comparability, enabling fair and informative assessments across diﬀerent classes of models.

In addition, we report frame-level metrics that directly assess perceptual quality relative to ground truth frames. These include LPIPS (Learned Perceptual Image Patch Similarity) [32], SSIM (Structural Similarity Index), and PSNR (Peak Signal-to-Noise Ratio). These metrics quantify visual similarity, structural consistency, and reconstruction ﬁdelity, respectively, oﬀering ﬁner-grained insight into how well the generated frames preserve appearance and local coherence. To extend these frame-level metrics to video, we average the scores for all the frames of a given video.

#### Appendix G. Training and Implementation Details

Model Architecture. Ctrl-Crash builds on the Stable Video Diﬀusion (SVD) framework as the base image-to-video generation model. In the ﬁrst stage, we ﬁne-tune the SVD model on curated crash and driving clips from the MMAU dataset. In the second stage, we freeze the base model and train a ControlNet module to integrate spatial (bounding box control frames) and semantic (crash type) inputs via additional encoder and cross-attention layers. All control signals are processed using the same pretrained VAE encoder used by SVD. The number of parameters for each model (and each sub-module) is given in Table G.6.

Training Setup. We use the AdamW optimizer with a constant learning rate of 4 × 10−5 and batch size of 1. Extensive hyperparameter tuning was intentionally avoided, as our focus is on validating the methods eﬀectiveness rather than maximizing performance through optimization. The ﬁrst-stage ﬁne-tuning of the base SVD model is performed for 101k steps, using an MSE loss in latent space. The second-stage ControlNet training is run for 31k steps, with conditioning dropout applied as described in Section 3.4 (main paper). We use mixed precision during training by setting weights and inputs to fp16 for non-trainable (frozen) parts of the model (i.e., VAE encoder, VAE decoder, CLIP encoder) and keep the trainable parts at fp32 to reduce memory usage. Training is performed on 4 NVIDIA 80GB A100 GPUs over approximately 2 weeks for both stages combined. All models are implemented in PyTorch using the Hugging Face accelerate library as a base.

Inference and Guidance Parameters. During inference, we apply multi-condition classiﬁer-free guidance with tunable guidance scales. Unless otherwise speciﬁed, we use γB ∈ [1,3] ⊂ R for bounding box control and γT ∈ [6,12] ⊂ R for crash type conditioning. These values represent the ranges of values within the guidance scales will increase linearly throughout the denoising process (e.g., guidance scale starting at 1 at ﬁrst denoising step and ﬁnishes at 3 at the last denoising step). The base model ϵϕ used for unconditional noise prediction is the original pretrained SVD base checkpoint prior to any ﬁne-tuning. We sample videos with 25 frames at a resolution of 512 × 320, using DDIM sampling with 30 denoising steps.

Dataset Splits. We curate a training set of 7,500 clips from MM-AU after ﬁltering. A held-out validation set of 900 clips is used for evaluation. All reported metrics in the main paper (Tables 1, 2) are computed on generated samples from the held-out validation set.

#### Submodule Status (Stage 1) Status (Stage 2) Number of Parameters

VAE-Encoder Frozen Frozen 34,163,592 VAE-Decoder Frozen Frozen 63,579,183 CLIP-Image Encoder Frozen Frozen 632,076,800 UNet Trainable Frozen 1,524,623,082 ControlNet N/A Trainable 681,221,585

Total 2,935,664,242 ≈ 3B

Table G.6: Number of parameters by submodule. Refer to architecture diagram in Figure 2 of the main paper for more information on the submodules. Stage 1 and Stage 2 refer to the two stages of training for our method.

#### Appendix H. Dataset

Dataset annotations will be made publicly available and it will be possible to obtain them by following instructions from our open-source code base. We include a subset of the annotations in the supplementary ZIP ﬁle for reference, as the full set would exceed the size limit for submission.

All annotated video samples in our dataset (MM-AU extension, RussiaCrash test set, etc.) are stored in JSON format, where each annotation ﬁle corresponds to a single video. Below, we describe the structure and semantics of the annotation schema.

Annotation Structure. Each annotation consists of three main ﬁelds:

- • video_source: The ﬁlename of the source video (e.g., "7_00951.mp4").
- • metadata: High-level information about the annotated scenario, including:

- – ego_involved (bool): Indicates whether the ego vehicle (assumed to be the camera holder) is involved in the accident.
- – accident_type (int): A categorical index representing the accident type as deﬁned by DADA2000 dataset. See Figure H.14 and Table H.7 for more information).

- • data: A list of per-frame annotations. Each frame entry includes:

- – image_source: The ﬁlename of the corresponding frame image (e.g., "7_00951_0000.jpg").
- – labels: A list of annotated objects (bounding boxes) in the frame. Each object contains:

- * track_id (int): A persistent ID assigned to each object across frames.
- * name (str): The object class as a string (e.g., "car", "person", "truck").
- * class (int): The numerical class index used internally (e.g., 0 = person, 1 = car).
- * box (list[ﬂoat]): The bounding box coordinates, normalized to the range [0, 1], in the format [x_min, y_min, x_max, y_max].

All bounding boxes are temporally linked using consistent track_id values. Object classes follow the taxonomy deﬁned by the YOLOv8 model used in the annotation pipeline.

The MM-AU dataset [10] labels each video with an accident type as deﬁned by the DADA2000 dataset [36]. These accident types are presented in Figure H.14. In our work, we reduce these numerous crash types to 5 types as deﬁned in Section 3.3 and repeated here: Crash types are represented by a discrete class label from ﬁve categories: (0) none, (1) ego-only, (2) ego/vehicle, (3) vehicle-only, and (4) vehicle/vehicle. These indicate which agents are involved in the crash, with, for example, vehicle/vehicle describing a crash between two non-ego agents. Table H.7 shows the association from the DADA2000 crash types to the Ctrl-Crash crash types that were used in this work.

Table H.7: Ctrl-Crash to DADA2000 crash type association

#### Crash Type DADA2000 Crash Types # of Training Samples

- 0 - no crash N/A 1745
- 1 - ego-only 13, 14, 15, 16, 17, 18, 61, 62 267
- 2 - ego/vehicle 1-12 3182
- 3 - vehicle-only 19-37, 39, 41, 42, 44 577
- 4 - vehicle/vehicle 38, 40, 43, 45-51 2168

For training and evaluation, we split all videos into 25-frame clips. The MM-AU dataset gives exact frame labeling to indicate when the accident occurs and when abnormal driving starts and ends before and after the accident. For each video we sample a 25-frame video containing the accident frame and label it according to the accident type. Then we sample, if possible, a 25-frame video containing only "normal" driving (i.e., with no overlap with abnormal or accident labeled frames) and label these clips with a crash type "no crash". Ultimately this yields 6,927 clips containing a crash and 1,964 clips without a crash (see Table H.7 for number of clips for each class during training).

[Figure 233]

###### Figure H.14: Original accident type deﬁnition used by the MM-AU dataset [10] as deﬁned by the DADA2000 dataset [36]

#### Appendix I. User Survey

[Figure 234]

[Figure 235]

(a) Visual preference comparison in generated videos. (b) Realism preference comparison in physical appearance.

- Figure I.15: User study results comparing generated videos from AVD2, DrivingGen and Ctrl-Crash with 40 participants across 5 crash types: Participants exhibit a strong preference for Ctrl-Crash generated videos, citing superior visual quality and physical realism.

We conducted a brief user study with n = 21 participants who were asked to rank k = 3 videos (from Ctrl-Crash, AVD2, and DrivingGen) across 5 diﬀerent crash scenarios each. The participants consist of students from our lab who are not associated with the project in any way. No further demographic data was collected. The users were asked to rank the 3 videos in each of the 5 situations by (a) physical plausibility and (b) visual ﬁdelity from best to worst. The users were required to choose a best/medium/worst video in each question and visual/physical category. We used the non-parametric Friedman test [37] to determine with p ≤ 0.01 that there is a method that is consistently ranked higher than the others (Ctrl-Crash). We further used the Nemenyi post-hoc analysis [38] to ﬁnd that with p ≤ 0.01, our method consistently outperforms both AVD2 and DrivingGen in both physical realism and visual ﬁdelity (see Figure I.15).

The participants were not given any information about the study ahead of time other than that it revolves around state-of-the-art video generation. The survey was carried out through Google Forms. At the start of the survey, they received the following instructions:

Content warning: ai-generated mild car crashes with no humans depicted. No blood/injury, just car-on-car action.

Please help us get some human feedback on a new video generation method. We’re asking you to rank 5 sets of videos. Should take less than 5 min, and you’ll see why AI won’t take over anything anytime soon.

Instructions: For each of these 5 questions, we will show you 3 short video clips that have ALL been AI-generated. We will ask you:

- • Does each video depict a crash?
- • Rank the 3 videos by highest physical accuracy/plausibility
- • Rank the 3 videos by highest visual ﬁdelity (aka are they nice to look at)

Important: Please rank the videos relative to one another, i.e. "best" means best of the 3.

Each of the 5 accident types (with 3 videos each, labeled A,B,C, randomly shuﬄed from AVD2, DriveGen, ours) were presented like in Figure I.16. The 5 accident types we selected were: head-on collision, t-bone, rear-ending, dangerous overtaking, and loosing control of the vehivle and going oﬀ the road. The ranking was implemented as shown in Figure I.17 by forced-choice.

25

[Figure 236]

- Figure I.16: User Survey Screenshot 1, showing 2 (out of 3) samples from question 1. What is shown as static image here was a GIF in the original Google Form.

[Figure 237]

Figure I.17: User Survey Screenshot 2, showing the evaluation questions for each batch of 3 videos. Users were forced to rank all videos and to only use each rank once (i.e., it is not possible to submit the form when more than one video in each batch has the same rank).

