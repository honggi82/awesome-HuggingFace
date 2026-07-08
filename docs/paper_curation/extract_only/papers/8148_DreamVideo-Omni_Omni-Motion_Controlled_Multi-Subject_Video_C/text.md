# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

Yujie Wei∗ , Xinyu Liu∗ , Shiwei Zhang† , Hangjie Yuan , Jinbo Xing , Zhekai Chen , Xiang Wang , Haonan Qiu , Rui Zhao , Yutong Feng , Ruihang Chu , Yingya Zhang, Yike Guo , Fellow, IEEE,

Xihui Liu , Member, IEEE and Hongming Shan , Senior Member, IEEE

## arXiv:2603.12257v1[cs.CV]12Mar2026

Abstract—While large-scale diffusion models have revolutionized video synthesis, achieving precise control over both multi-subject identity and multi-granularity motion remains a significant challenge. Recent attempts to bridge this gap often suffer from limited motion granularity, control ambiguity, and identity degradation, leading to suboptimal performance on identity preservation and motion control. In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm. In the first stage, we integrate comprehensive control signals for joint training, encompassing subject appearances, global motion, local dynamics, and camera movements. To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance. Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances. In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pretrained video diffusion backbone. This provides motion-aware identity rewards in the latent space, prioritizing identity preservation aligned with human preferences. Supported by our curated large-scale dataset and the comprehensive DreamOmni Bench for multi-subject and omni-motion control evaluation, DreamVideoOmni demonstrates superior performance in generating highquality videos with precise controllability. Our project webpage: https://dreamvideo-omni.github.io.

Index Terms—Video Diffusion Models, Video Customization, Motion Control, Reinforcement Learning

I. INTRODUCTION

The landscape of video generation has been revolutionized by the advent of diffusion models [1], [2], [3], [4], [5], [6], [7], [8], [9]. While these foundation models demonstrate impressive capability in synthesizing high-fidelity videos from

∗Equal Contribution †Project Leader Corresponding Author Email: Yujie Wei yjwei22@m.fudan.edu.cn and Xinyu Liu

xliugd@connect.ust.hk

Yujie Wei and Hongming Shan are with Fudan University. Xinyu Liu and Yike Guo are with The Hong Kong University of Science and Technology. Shiwei Zhang, Jinbo Xing, Xiang Wang, Yutong Feng, Ruihang Chu, Yingya Zhang are with Tongyi Lab, Alibaba Group. Hangjie Yuan is with Zhejiang University. Zhekai Chen and Xihui Liu are with MMLab, The University of Hong Kong. Haonan Qiu is with Nanyang Technological University. Rui Zhao is with Show Lab, National University of Singapore.

textual descriptions, the demand for real-world applications necessitates more granular control. Specifically, users often require the generation of videos that simultaneously preserve the identities of multiple subjects while adhering to comprehensive motion control, including global object motion, local limb motion, and camera movements.

Despite the remarkable advances in controllable video generation, achieving this dual objective—robust multi-subject ID preservation and precise, multi-granularity motion controlremains an open challenge. Existing approaches typically diverge into two independent research directions. On one hand, subject-driven methods [10], [11], [12] utilize adapters or tuning-free mechanisms to inject appearance, yet they often lack the capacity for precise spatial control, resulting in videos where subjects drift uncontrollably or remain static. On the other hand, motion-controlled methods [13], [14], [15], [16] excel at guiding movement via bounding boxes or trajectories but fail to achieve omni-motion control and customize userspecified subjects, limiting their practical applicability.

Recent works attempt to integrate these capabilities within a unified framework, aiming to synthesize videos that faithfully preserve subject identities while adhering to specified motion patterns [17], [18], [19], [20]. However, these methods often yield suboptimal performance due to the intrinsic tradeoff between subject preservation and motion control. This limitation manifests primarily in three aspects: 1) Limited Motion Control Granularity. Most existing methods rely on a single type of motion signal, such as bounding boxes [17], depth maps [18], [19], or sparse trajectories [20], to guide generation. This restricted conditioning fails to support the simultaneous control of global object placement, fine-grained local dynamics, and camera movement, thereby limiting the flexibility and realism of the generated videos. 2) Ambiguity in Motion Control. Current approaches typically inject all conditioning signals indiscriminately without explicit binding mechanisms [18], [19]. In multi-subject scenarios, this leads to severe ambiguity, as the model struggles to discern which motion pattern corresponds to which specific reference subject. This confusion is further exacerbated when integrating multigranular motion controls. 3) Identity Degradation. Compared to independent subject customization, introducing motion control often compromises identity fidelity. This stems from the divergent nature of the objectives: identity preservation en-

|[Figure 1]|[Figure 2]|
|---|---|

A man engaging in a pleasant conversation with another man. They are smiling and occasionally gesturing with their hands, … Their mouths move naturally, vividly simulating a real-life exchange. The entire scene is illuminated by meticulously crafted cinematic lighting, creating a warm and intimate atmosphere. The soft light gently falls upon both of them …

###### Multi-Subject Customization

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Half body view of a foreign white man and woman taking selfie outdoors. Male wearing a light shirt, a black hat, holding a camera, female wearing a brown top, carrying a bag, two smiling, the background is a cityscape, including bridges and buildings, the lens moves, and the video is shot in a head-up lens.

###### Motion Control + Camera Movement

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

|[Figure 13]|[Figure 14]| |
|---|---|---|

A little boy playing soccer on a green grass field. The boy is wearing a yellow sports uniform and is intently chasing after a soccer ball. …

Subject Customization + Omni-Motion Control

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 20]

|[Figure 21]<br><br>[Figure 22]| |[Figure 23]| |
|---|---|---|---|

A … little girl interacting with a puppy in the park. The little girl, wearing an orange jacket and black pants, is sitting on the grass … The puppy … stands up and

Multi-Subject Customization + Omni-Motion Control

stretches out its front paws, waiting for the food …

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- Fig. 1. Zero-shot multi-subject customization and omni-motion control achieved by DreamVideo-Omni. Our method enables seamless multi-subject customization, precise motion and camera control, and simultaneous single/multi-subject customization with omni-motion control.

courages pixel-level consistency with a static reference image, whereas motion control necessitates dynamic pixel variation and temporal evolution to render movement. Standard diffusion reconstruction losses are insufficient to reconcile this conflict, leading to degradation of fine-grained identity details, particularly when synthesizing large-amplitude motions.

should prioritize perceptual alignment with human experience rather than relying solely on low-level reconstruction metrics.

Based on these insights, we propose DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization and omni-motion control through a progressive two-stage training paradigm illustrated in Fig. 1. In the first stage, referred to as omni-motion and identity supervised finetuning, we integrate comprehensive control signals, formulated as structured triplets of ⟨Reference Subject, Global Box, Local Trajectory⟩, into a single DiT architecture. To effectively process these heterogeneous inputs, we design a condition-aware 3D Rotary Positional Embedding (RoPE) that assigns distinct spatiotemporal indices to diverse conditions, facilitating faster convergence and enhanced training stability. To guarantee precise global motion control, we employ a hierarchical motion injection strategy, infusing bounding box conditions into each transformer block to reinforce spatial guidance. Furthermore,

To address these issues, we posit that it is pivotal to simultaneously enhance motion controllability and identity preservation. First, motion control signals must be explicitly bound to their corresponding reference subjects to resolve ambiguity, thereby facilitating precise, multi-granular control. Second, to further reinforce identity fidelity, the learning objective should be aligned with human preferences. We recognize that subject customization is inherently subjective and distinct from rigid pixel-wise correspondence, as a subject’s visual appearance naturally varies with viewpoints and poses, yet their identity remains consistent. Consequently, the optimization process

to resolve control ambiguity in multi-subject scenarios, we introduce learnable group and role embeddings that distinguish distinct control units and specify signal modalities, explicitly anchoring motion signals to their corresponding identities. Collectively, these architectural designs establish a foundation for integrating subject customization with controllable motion generation within a unified framework.

In the second stage, referred to as latent identity reward feedback learning, we move beyond standard diffusion losses and employ a reward feedback strategy within the latent space to mitigate identity degradation during dynamic motion generation. Specifically, we train a specialized Latent Identity Reward Model (LIRM) to provide rewards. Departing from previous methods that rely on static image encoders (e.g., CLIP [21] or DINO [22]) as reward models, which overlook temporal dynamics, our LIRM is constructed upon a pretrained Video Diffusion Model (VDM), yielding two advantages: 1) Motion-Aware Identity Assessment: By leveraging the VDM’s inherent spatiotemporal priors, the reward model evaluates video-level identity consistency that integrates motion dynamics, penalizing static “copy-paste” artifacts while encouraging robust identity preservation under large motion.

- 2) Computationally Efficient Training: Computing rewards in the latent space bypasses expensive VAE decoding. This enables direct gradient backpropagation from reward models to the video generation model, fully leveraging the potential of reward feedback learning.

Remarkably, this progressive two-stage training paradigm enables the seamless composition of multiple tasks while naturally facilitating the emergence of novel generative abilities. Despite being built upon a text-to-video base model, our framework spontaneously unlocks zero-shot image-to-video generation and first-frame-conditioned trajectory control.

Finally, to support the training of our unified framework, we curate a large-scale dataset for multi-subject customization and omni-motion control, which comprises 2M video clips, enriched with multi-subject reference images, bounding boxes, and trajectory conditions. Beyond training resources, we also address the critical lack of comprehensive evaluation protocols. Existing benchmarks either isolate customization from controllable generation [11], [10] or focus exclusively on simple point trajectories [16]. To bridge this gap, we construct the DreamOmni Bench, a holistic evaluation suite composed of 1,027 high-quality, real-world videos. This benchmark explicitly categorizes single- and multi-subject scenarios and is equipped with dense annotations, enabling the first unified evaluation of identity preservation and complex motion controllability in zero-shot settings.

In summary, our contributions are five-fold: 1) We present DreamVideo-Omni, the first unified framework that harmoniously integrates multi-subject customization with omnimotion control within a single DiT architecture. 2) We propose specialized architectural components to ensure precise controllability. Specifically, we introduce group and role embeddings to resolve multi-subject ambiguity, condition-aware 3D RoPE to coordinate heterogeneous inputs for stable training, and hierarchical motion injection to enhance global motion control.

- 3) We design a latent identity reward feedback learning

paradigm. We train a VDM-based latent identity reward model to evaluate motion-aware identity preservation, effectively mitigating identity degradation under large motion. 4) We establish DreamOmni Bench, a new benchmark consisting of over 1K curated videos with comprehensive annotations, designed to simultaneously quantify multi-subject consistency and motion control precision. We also design a comprehensive data processing pipeline for multi-subject customization and omni-motion control tasks. 5) Extensive experimental results demonstrate our framework’s superiority over state-of-the-art methods in both identity preservation and motion control.

II. RELATED WORK

Customized video generation. Customized image generation has garnered growing attention [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34]. Recently, many works explore customized video generation using a few subject or facial images [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], while several works study the challenging multi-subject video customization task [47], [48], [49], [50]. For example, ConsisID [45] leverages frequency decomposition to decouple facial contours and details in video DiT for consistent identity across frames. For multi-subject customization, VideoMage [51] and Video Alchemist [49] extend single-subject methods to open-set personalization, improving multi-subject identity consistency without test-time tuning. However, these methods focus on independent subject customization and often struggle with “copy-paste” artifacts and limited controllability, restricting their applicability in real-world scenarios. Considering that spatial content and temporal dynamics are two essential components of videos, DreamVideo [11] customizes both subject and motion by training two adapters and combining them at inference, while MotionBooth [52] fine-tunes a model to learn subjects and edits attention maps to control motion during inference. More recently, Tora2 [20] integrates trajectory control into subject customization by introducing a decoupled personalization extractor and a gated self-attention mechanism. However, these methods rely predominantly on standard diffusion losses and suffer from the trade-off between motion control and identity preservation, often resulting in identity degradation under large-amplitude motions. In contrast, DreamVideo-Omni effectively resolves this dilemma by aligning identity preservation with human preferences. By introducing a latent identity reward feedback learning paradigm and training a specialized latent reward model, our approach ensures harmonious multisubject customization with omni-motion control.

Motion control in video generation. Recent advancements in video generation focus on enhancing motion dynamics using extra control signals [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65]. Many motion customization methods learn motions from reference videos [66], [67], [68], [69], [70], but require complicated fine-tuning during inference. To alleviate this, some training-free approaches [71], [72], [73], [74], [75] employ attention manipulation or guidance to achieve zero-shot control. However, these methods often sacrifice motion precision and temporal consistency in

complex scenarios. In contrast, several works use trajectories or coordinates as additional conditions to train a motion control module [76], [13], [14], [77], [16], [78], [79], [15]. For example, Motion Prompting [57] conditions generation on spatio-temporal trajectories for camera control and motion transfer, with prompt expansion for complex inputs. MagicMotion [80] uses object masks and bounding boxes to control motion, trained on a pretrained image-to-video diffusion model. Wan-Move [16] uses dense point trajectories projected into latent space to propagate features, enabling motion control based on the first frame in image-to-video models. Nonetheless, they are incapable of delivering comprehensive fine-grained motion control, such as the simultaneous control of global motion, local dynamics, and camera movements. Moreover, they fail to incorporate user-specified subject appearances, which restricts their real-world applicability. In contrast, our DreamVideoOmni unifies user-specified subject customization and multigranular motion control into a single framework, incorporating a binding mechanism to explicitly resolve motion ambiguity and ensure precise control.

Identity-based reinforcement learning. To ensure identity consistency in customized video generation, recent research has increasingly integrated reinforcement learning into optimization frameworks [81], [82], [83], [84]. For instance, MagicID [81] employs DPO [85] to enhance text-to-video identity stability, though it still necessitates costly per-identity LoRA adaptation and test-time fine-tuning. More recently, IdentityGRPO [83] leverages human preference-driven GRPO [86] to preserve stable facial features during complex interactions by constructing multi-character reward models. IPRO [82] adopts the Reward Feedback Learning (ReFL) [87] paradigm. It backpropagates gradients from similarity-based rewards directly into the diffusion model, bypassing explicit reward model training or identity-specific tuning. However, both IdentityGRPO and IPRO require decoding latents into pixel space for reward calculation, which incurs heavy GPU overhead and restricts feedback to the final denoising steps, resulting in limited performance improvements. In contrast, we conduct identitydriven reinforcement learning directly within the latent space, significantly reducing computational overhead. While the very recent general video generation method PRFL [88] also introduces latent-space reward modeling to mitigate computational costs, it primarily focuses on optimizing general motion quality, lacking the capacity to distinguish and preserve intricate subject identities. Unlike PRFL, we process the generated video and the reference image in parallel under varying noise levels, explicitly leveraging the clean reference latents as queries to attend to the noisy video latents for computing identity rewards. This design substantially enhances identity customization capabilities by incorporating identity information across the full range of timesteps, thereby fully unleashing the potential of ReFL in customization tasks.

III. OUR MEHTOD: DREAMVIDEO-OMNI

Given a set of reference images and motion signals (i.e., bounding boxes and trajectories), our DreamVideo-Omni employs a unified video diffusion transformer to jointly condition

on multi-subject appearances, global and local object motions, and camera movements, enabling flexible compositional video generation without test-time fine-tuning. The overall pipeline is shown in Fig. 2, which follows a progressive two-stage training paradigm. In the first omni-motion and identity supervised fine-tuning stage, we train an all-in-one video DiT to integrate heterogeneous control signals via hierarchical injection, condition-aware 3D RoPE, and group/role embeddings, effectively resolving multi-subject ambiguity and enabling fine-grained motion control. In the second latent identity reinforcement learning stage, we employ latent identity reward feedback learning, where a specialized latent identity reward model provides direct identity supervision on intermediate noisy latents, efficiently bypassing the computationally expensive VAE decoding. In the following sections, we detail this two-stage paradigm in Secs. III-A and III-B. Subsequently, we describe the dataset construction in Sec. III-C and introduce our newly constructed DreamOmni Bench in Sec. III-D.

A. Omni-Motion and Identity Supervised Fine-Tuning

This subsection will dive into the key designs of DreamVideo-Omni, including model architecture, task design, and details of the designed components.

1) Model Architecture and Task Design: As illustrated in Fig. 2, we instantiate an all-in-one framework by adapting a pre-trained text-to-video DiT [8]. Our model is jointly trained on a comprehensive set of tasks, including single- and multisubject customization, global and local object motion control, and camera movement control. To enable precise and flexible composition of these tasks, we carefully craft four compact, interaction-friendly conditioning signals.

- 1) Subject appearance. We use one reference image per subject and segment it to obtain the image with a blank background, preserving distinct identity features while reducing background interference.
- 2) Global object motion. We employ scene-anchored bounding boxes to indicate global object motion. These boxes serve as an intuitive proxy for scene-level spatial attributes, effectively capturing object position, scale, aspect ratio, and relative depth. In practice, users can simply specify start and end boxes with optional intermediate boxes to achieve flexible global motion control.
- 3) Local object motion. We represent local object motion using sparse point-wise trajectories. Compared to global motion, local object motion targets finer-grained, complex in-place dynamics (e.g., limb raises, head turns) that enrich per-object motion details. This point-based control flexibly captures complex non-rigid deformations.
- 4) Camera movement. We also use point-wise trajectories to achieve camera movement control. Prior works typically rely on explicit 3D camera parameters and auxiliary datasets for training [13], [19]. While effective, these approaches increase training cost and hinder usability. In contrast, we observe that camera movement can be effectively induced by applying pointwise trajectories to background pixels. This allows us to unify camera and local motion control under the same

Stage 1: Omni-Motion and Identity Supervised Fine-Tuning

[Figure 29]

| | |[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]|
|---|---|---|

[Figure 33]

| |[Figure 34]|[Figure 35]<br><br>[Figure 36]|
|---|---|---|

[Figure 37]

[Figure 38]

[Figure 39]

A heartwarming scene of two girls interac ng with a golden retriever in a parking lot. …

[Figure 40]

Prompt Video

Bounding Boxes Subject Reference Images Trajectories VAE Encoder VAE Encoder VAE Encoder

Embedder

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

T5 Encoder

padding

+ + + + +

Add Noise

|Object 🔥<br><br>[Figure 45]| |
|---|---|
|Group 1🔥<br><br>[Figure 46]| |
| | |

|Object 🔥<br><br>[Figure 47]| |
|---|---|
|Group 2🔥<br><br>[Figure 48]| |
| | |

|Object 🔥<br><br>[Figure 49]| |
|---|---|
|Group 3🔥<br><br>[Figure 50]| |
| | |

|Group 1 2 3🔥<br><br>[Figure 51]|
|---|

|Control 🔥<br><br>[Figure 52]|
|---|

|Control 🔥<br><br>[Figure 53]| |
|---|---|
|Group 1 2 3🔥<br><br>[Figure 54]| |
| | |

ZeroConv Layers 🔥

+

[Figure 55]

| | | |
|---|---|---|
|3 4| | |

Input Tokens

Pos Embed

1 2 1 2 3 4

[Figure 56]

### DreamVideo-Omni 🔥

[Figure 57]

Position Embedding

1

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Reweighted Diffusion Loss

Pos Embed for Ref Img Pos Embed for Pad Token

#### Stage 2: Latent Identity Reinforcement Learning

[Figure 64]

###### Latent ID Reward Model

[Figure 65]

[Figure 66]

Add Noise

[Figure 67]

[Figure 68]

[Figure 69]

🔥

[Figure 70]

[Figure 71]

🔥 🔥

𝒛 ,𝒕

Reward Model Training

Video Diffusion Model

Cross Attention

Head Reward

VAE Encoder

Posi ve/Nega ve Video

Spatiotemporal Features

𝒓𝒕

|[Figure 72]|
|---|

𝓛𝐋𝐈𝐑𝐌

𝒛

Query

Reference Image

Preference Labels

ID Feature

Prompt & Timestep

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

🔥

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

DreamVideoOmni

Latent ID Reward Model

###### Reward

…

𝓛𝐋𝐈𝐑𝐞𝐅𝐋

𝒓𝒕

𝒛𝑻 𝒛𝒕

𝒛𝒕

Faster training & lower memory

###### Reward Feedback Learning

[Figure 86]

|[Figure 87]|
|---|

###### Reference Image VAE

Reward Model

Decoder

- Fig. 2. Overview of DreamVideo-Omni. In Stage 1, the framework introduces an all-in-one video DiT that incorporates reference images, bboxes, and trajectories for multi-subject customization and omni-motion control. Stage 2 further enhances identity fidelity via the proposed latent identity reward feedback learning mechanism, which utilizes a latent identity reward model to directly evaluate intermediate latents, completely bypassing the expensive VAE decoder for faster training.

trajectory conditioning mechanism, reducing training overhead while improving interactivity.

sequences into the DiT, we first filter out clips exhibiting abrupt object fluctuations by detecting adjacent-frame bounding box IoU to ensure training stability. The valid sequences are then rendered as RGB videos on a white background, where each object is assigned a unique random color and pixelwise averaging is applied in overlapping regions to resolve ambiguity. These rendered videos are projected into the latent space via a 3D VAE. To enable efficient and effective control signal injection, we implement a hierarchical motion injection strategy: bounding box latents are added to both the noisy input latents and the output of each DiT block via learnable,

- 2) Conditioning Signal Injection: Beyond the task formu-

lation, the effective injection of conditioning signals is pivotal for achieving precise control.

- 1) Subject appearance. To enable robust multi-subject customization while mitigating “copy-paste” artifacts, reference images are extracted from frames temporally disjoint from the current training clip and undergo extensive data augmentation, detailed in Sec. III-A3. These processed images are then encoded via a 3D VAE, and their latent representations are concatenated to form the conditioning input for the video DiT.
- 2) Global object motion. Before inputting bounding box

layer-specific zero-convolutions, formulated as:

###### h0 = zt + Zin(zbox), hl+1 = Blockl(hl) + Zl(zbox), (1)

where zt and zbox are the input noisy video latents and bounding box latents, respectively. Zin and Zl denote the zeroconvolutions at the input stage and the l-th DiT block. hl is the input hidden state to the l-th block. This dense injection mechanism enhances the precision of global motion control without increasing the token sequence length.

- 3) Local object motion and camera movement. Since we employ point trajectories to control both local object dynamics and camera movements, the sampled points must comprehensively cover both foreground subjects and background regions. While uniform sampling is an intuitive solution, it often struggles with the fine-grained motion control due to insufficient sampling density on object boundaries. To address this, we devise a hybrid sampling strategy that stochastically alternates between two modes: (i) random grid sampling, which ensures broad coverage of whole scene dynamics (background and objects); and (ii) object-aware sampling, which samples strictly within foreground masks to focus on intricate local dynamics. To improve robustness on trajectory densities, a subset of trajectories is randomly dropped during training. Following Motion Prompting [57], we construct trajectory tokens by generating unique sinusoidal positional encodings and scattering them into blank feature maps according to their discretized spatiotemporal coordinates. These tokens are subsequently concatenated with the noisy video and reference image latents to condition the DiT training.

3) Specialized Architectural Components : Conditionaware 3D RoPE. To process heterogeneous inputs, including video latents, multi-subject reference images, and motion control signals, within a unified DiT architecture, we concatenate all latent tokens along the temporal dimension and design a condition-aware 3D Rotary Positional Embedding (RoPE). While our condition-aware 3D RoPE maintains standard indexing for spatial dimensions to preserve geometric structure, it employs a specialized temporal indexing strategy to distinguish input types:

- (i) Video frame tokens: We assign sequential temporal

indices t ∈ [0,T − 1] to these tokens to ensure temporal consistency. Note that bounding box latents are element-wise added to these frames, thereby naturally inheriting the same positional embeddings.

- (ii) Reference image tokens: We assign a shared, distinct

time index tref to all valid reference image tokens. This design explicitly decouples reference subjects from the video tokens, instructing the model to treat these tokens as static visual conditions rather than sequential frames.

- (iii) Padding tokens: To handle varying numbers of refer-

ence subjects across videos, we pad the reference image tokens to a fixed capacity Nmax within the batch. These padded tokens are assigned a distinct “invalid” time index tpad, allowing the model to identify and ignore these non-informative tokens.

- (iv) Trajectory tokens: To provide precise pixel-level motion

control, trajectory tokens inherit the same temporal indices t ∈ [0,T − 1] as the video frame tokens, ensuring strict spatiotemporal alignment with the corresponding video.

By integrating these specific indices into the 3D RoPE, we enable the unified DiT architecture to effectively process heterogeneous inputs, including both reference images and diverse motion control signals.

Group and role embeddings. To mitigate control ambiguity in multi-subject generation and distinguish the functional roles of heterogeneous inputs, we introduce two types of learnable embeddings: group embeddings and role embeddings. First, we formulate the fundamental control unit as a triplet of ⟨Reference Subject,Global Box,Local Trajectory⟩, and assign a unique group embedding to each unit. Given that a reference image captures a single subject, its corresponding group embedding is added to all latent tokens of the image. In contrast, for the bounding box and trajectory latents, this same group embedding is injected exclusively into the spatial regions and track points corresponding to the subject. This explicit binding mechanism ensures that each subject is correctly associated with its corresponding bounding box and trajectories, effectively preventing control confusion. Second, we introduce role embeddings, comprising object and control embeddings, to differentiate input signals’ functionalities. Specifically, an object embedding is added to all reference image tokens to designate them as visual appearance assets, whereas a control embedding is applied to all bounding box and trajectory tokens to mark them as motion control guidance. This functional distinction enables the model to effectively process heterogeneous conditions.

Data augmentation for subject customization. To mitigate “copy-paste” artifacts caused by directly training on reference subjects, we apply a robust augmentation pipeline to the reference images. Specifically, we stochastically employ geometric transformations (e.g., flipping, rotation, affine shearing, and cropping) and visual degradations (e.g., blur, color jitter) to prevent overfitting. These perturbations effectively enhance the robustness of identity preservation.

Training loss. Following DreamVideo-2 [17], we use a reweighted diffusion loss that differentiates the contributions of regions inside and outside the bounding boxes. Specifically, we amplify the contributions within bounding boxes to enhance subject learning while preserving the original diffusion loss for regions outside these boxes. The training loss of stage 1 is defined as:

###### Lsft = Ez,ϵ,C,t (1 + λ1M) · ϵ − ϵθ(zt,C,t) 22 , (2)

where C = {ctxt,zref,zbox,ztraj} is the comprehensive conditioning set. ctxt is the text prompt, zref is the latent codes of reference images Iref, and ztraj is the constructed trajectory feature map. M denotes the binary bounding box masks (1 for foreground, 0 otherwise), and λ1 > 0 is the balancing factor. ϵ ∼ N(0,I) and ϵθ denotes the denoising DiT network.

B. Latent Identity Reinforcement Learning

While the omni-motion and identity SFT stage establishes unified controllability, relying solely on low-level reconstruction losses is insufficient for preserving fine-grained appearance details. To further enhance identity fidelity by aligning with human preferences, we introduce the latent identity

reinforcement learning stage, which trains a latent identity reward model for reward feedback learning, as shown in Fig. 2.

- 1) Latent Identity Reward Model: To provide fine-grained,

identity-consistent feedback during reinforcement learning, we introduce the Latent Identity Reward Model (LIRM). Unlike conventional reward models (e.g., CLIP or VLMs) that evaluate videos in RGB space, our LIRM operates directly in latent space, mitigating computational overhead and facilitating the subsequent reward feedback learning.

Architecture. Fig. 2 shows that LIRM comprises three key modules: a video diffusion model (VDM) backbone, an identity cross-attention layer, and a reward prediction head. Specifically, given a video pair V ∈ {Vpos,Vneg} and a reference image Iref, we first project them into a shared latent space via a 3D VAE encoder, yielding latent representations zV and zref. We perturb zV with gaussian noise at timestep t to yield zV,t, while maintaining zref in its clean state. We then leverage the pretrained VDM backbone Φ to extract spatiotemporal features fV = Φ(zV,t,t,ctxt) from the noisy video and identity features fref = Φ(zref,t0,ctxt) from the reference image. Subsequently, the identity features serve as the query Q in a cross-attention layer to attend to the video’s spatiotemporal features acting as key K and value V, measuring the alignment between the subject’s identity and the generated video content:

hattn = Attention(Q,K,V) = Softmax

##### QK⊤

√

d

V, (3)

where Q = frefWQ, and K,V = fV WK,fV WV. Finally, a residual connection fuses the aligned features with the query, and the resulting representation is passed through a lightweight MLP head H to predict the scalar reward rt:

rt = H(hattn + Q). (4)

Unlike prior works [83], [82] that rely on static image-based encoders as reward models, our LIRM leverages the inherent spatiotemporal priors of the VDM. This facilitates motionaware identity assessment by evaluating identity consistency integrated with motion dynamics, penalizing “copy-paste” artifacts while ensuring robust preservation under large motion. Latent identity preference optimization. We curate a highquality preference dataset DLIRM = {(V,Iref,ctxt,y)i}Ni=1 from our in-house data, comprising ∼27,500 training videos and 500 testing videos. Each sample consists of video win-lose pairs coupled with corresponding single- or multi-subject reference images. Each video is assigned a human-annotated label y ∈ {0,1} to indicate whether the video V aligns with the identity defined by Iref. Leveraging this dataset, we optimize the LIRM parameters via a binary cross-entropy loss:

y log σ(rt) + (1 − y)log(1 − σ(rt)) , (5)

###### LLIRM = −ED

LIRM

where σ(·) is the sigmoid activation. To mitigate computational overhead, we utilize the first eight blocks of the VDM as the backbone, following [88]. During training, the VDM backbone, identity cross-attention layer, and reward prediction head are jointly updated.

2) Latent Identity Reward Feedback Learning: Benefiting from the architectural efficiency of our LIRM, we perform Reward Feedback Learning (ReFL) within the latent space to further enhance identity preservation by aligning with human preferences. Standard ReFL faces severe computational bottlenecks in video generation, as it necessitates expensive VAE decoding for pixel-level evaluation. Furthermore, it typically restricts reward feedback to the final denoised result, thereby neglecting structural information established in the early diffusion stages. In contrast, by bypassing the VAE decoder, our Latent Identity Reward Feedback Learning (LIReFL) significantly mitigates memory overhead. This design enables direct gradient backpropagation to the video generator and dense reward feedback at arbitrary diffusion timesteps, thereby fully leveraging the potential of ReFL.

Specifically, we initialize the latents from Gaussian noise and sample a target intermediate timestep tm ∼ U(0,T − 1). We first perform standard gradient-free denoising from step T down to tm+1 to conserve memory. Upon reaching step tm+1, we execute a single gradient-enabled denoising step to derive the predicted latent zt

, which is formulated as: zt

m

,tm+1,ctxt,zref), (6)

= µθ(zt

m

m+1

where µθ denotes the single-step solver function (e.g., UniPC [89] step) parameterized by the video generator ϵθ. The resulting zt

is immediately evaluated by the frozen LIRM to predict the identity reward rt

m

= LIRM(zt

,tm,ctxt,zref). The reinforcement loss is then formulated to maximize this expected identity fidelity:

m

m

]. (7)

LLIReFL = −Et

m,ctxt,zref[rt

m

To prevent “reward hacking”, where the model overoptimizes for identity scores at the expense of visual quality or diversity, we incorporate the supervised SFT objective from the first stage as a regularizer. The final training objective is a weighted combination:

L = Lsft + λ2LLIReFL, (8)

where λ2 controls the strength of the reward feedback. This balanced strategy ensures the model aligns with human identity preferences while preserving precise motion control and generative diversity established during the SFT stage.

C. Dataset Construction Pipeline

To facilitate the SFT stage of DreamVideo-Omni, which demands precise alignment across subject identity, global and local motion, and camera movement, we construct a largescale, densely annotated, high-quality video dataset. Fig. 3 illustrates our automated pipeline with four sequential stages:

- 1) Motion-based filtering. Robust motion control learning necessitates training samples with significant temporal dynamics. We estimate dense optical flow using RAFT [90] and compute the average motion magnitude across frames. Videos with small motion magnitude are discarded, ensuring the dataset focuses on meaningful motion patterns.
- 2) Subject discovery and captioning. To identify moving subjects and generate dense descriptions, we first utilize

TABLE I COMPARISON WITH DATASETS FOR VIDEO CUSTOMIZATION AND CONTROLLABLE GENERATION. OUR DATASET UNIQUELY SUPPORTS MULTI-SUBJECT CUSTOMIZATION WITH COMPREHENSIVE MOTION CONTROL ANNOTATIONS.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Motion Amplitude

RAFT

Videos

Optical Flow

Filtered Videos

- (1) Filter Videos by Motion Amplitude
- (2) Tag Subjects and Get Caption

["girl", "dog", "car"]

["girl", "dog", "car", "tree"]

Is Moving Subjects?

RAM++

Qwen3 Max

No. of Videos

Reference Images

MultiSubject

All-Frame Mask

All-Frame Box

All-Frame Trajectory

[Figure 95]

Subjects

[Figure 96]

[Figure 97]

A heartwarming scene of two girls interacting with a golden retriever in a parking lot. …The girls are dressed casually; one wears …, and the other … . The parking lot is surrounded by cars … in the background.

WebVid-10M [97] ∼10M ✗ ✗ ✗ ✗ ✗ UCF-101 [98] 13,320 ✗ ✗ ✗ ✗ ✗ DAVIS [99] 50 ✗ ✗ ✓ ✓ ✗ GOT-10k [100] 9,695 ✗ ✗ ✗ ✓ ✗ VideoBooth [10] 48,724 ✓ ✗ ✗ ✗ ✗ DreamVideo-2 [17] 230,160 ✓ ✗ ✓ ✓ ✗ Video Alchemist [49] ∼37.8M ✓ ✓ ✗ ✗ ✗ Phantom [12] ∼1M ✓ ✓ ✗ ✗ ✗ Wan-Move [16] ∼1.98M ✗ ✗ ✗ ✗ ✓

Qwen3-VL

Caption

Videos

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Bounding Boxes

Grounding DINO

["girl", "dog", "car"]

Subject Tags

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

###### SubjectFiltering

###### Our Dataset ∼2.12M ✓ ✓ ✓ ✓ ✓

Subject Masks

SAM2

###### GT Video

[Figure 113]

Bounding Boxes

[Figure 114]

[Figure 115]

Camera Trajectories

| | |
|---|---|
| | |

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

CoTracker3

[Figure 125]

[Figure 126]

[Figure 127]

Object Trajectories

Videos

[Figure 128]

###### Comprehensive Annotations

(3) Extract Subject Bounding Boxes, Masks, and Trajectories

Detailed Caption

|[Figure 129]|
|---|
|[Figure 130]|
|[Figure 131]|

|[Figure 132]<br><br>[Figure 133]|
|---|
|[Figure 134]|
|[Figure 135]<br><br>[Figure 136]|

[Figure 137]

Three yoga practitioners performing yoga in an indoor yoga studio. They are dressed in casual sportswear, executing a series of yoga poses on blue and pink yoga mats. The video is filmed from a fixed perspective, capturing their fluid transitions from one pose to another. The background features a minimalist interior with wooden floors and white walls.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Random Frame (Outside Clip) Videos

[Figure 142]

[Figure 143]

Reference Images

x

[Figure 144]

Data Augment ation

Bounding Boxes Trajectories Subject Masks

[Figure 145]

Subject Masks

|[Figure 146]|
|---|

(4) Extract Subject Reference Images

- Fig. 3. Pipeline of dataset construction.

Reference Images

|[Figure 147]|
|---|
|[Figure 148]|

|[Figure 149]|
|---|
|[Figure 150]|

[Figure 151]

RAM++ [91] to extract semantic tags from the video. These tags are subsequently refined by Qwen3 Max [92] to retain only significant moving subjects. Finally, we employ Qwen3VL [93] to generate detailed captions for each video.

[Figure 152]

- 3) Spatiotemporal annotation extraction. This stage extracts core structural conditions, including global bounding boxes, subject masks, and motion trajectories. Guided by the filtered tags, we first employ Grounding DINO [94] to detect subject bounding boxes, which serve as inputs for SAM

- 2 [95] to yield precise binary segmentation masks of subjects. Then, we utilize CoTracker3 [96] for dense point tracking and classify the resulting trajectories based on the subject masks: points falling within subject regions are labeled as object trajectories, while those in the background are designated as camera trajectories.

- 4) Reference image construction. To facilitate zero-shot customization and mitigate trivial copy-paste solutions, we sample reference images from frames temporally disjoint from the training clip. The subjects are then isolated via their corresponding segmentation masks and applied to extensive data augmentation to yield final reference images.

Fig. 4. Visualization of a test sample from DreamOmni Bench. Our benchmark supports fine-grained evaluation through comprehensive annotations, including multiple reference images for each subject, detailed captions, and precise spatial-temporal ground truths such as bounding boxes, motion trajectories, and subject masks.

lishes a solid foundation for both video customization and motion control tasks.

D. DreamOmni Bench

Existing benchmarks typically isolate video customization from controllable generation, rendering them inadequate for evaluating the holistic capabilities of DreamVideo-Omni. On the one hand, current personalization benchmarks [11], [17], [10] are predominantly confined to single-subject scenarios, lacking the capacity to assess multi-subject consistency and quantify motion controllability. On the other hand, recent motion control benchmarks, such as Wan-Move [16], focus exclusively on point trajectory precision. These protocols

Table I shows that distinct from previous video customization and controllable generation datasets, our dataset uniquely supports multi-subject customization with comprehensive motion annotations, including segmentation masks, bounding boxes, and trajectories. This richly annotated corpus estab-

are neither comprehensive nor capable of measuring identity preservation. Consequently, there is a critical absence of a unified benchmark that simultaneously evaluates multi-subject customization and comprehensive motion control, including bounding boxes and dense point trajectories.

To bridge this gap, we construct the DreamOmni Bench, composed of high-quality real-world videos sourced independently of our training dataset to ensure zero-shot evaluation. We perform manual filtering to retain high-resolution videos that exhibit meaningful subject motion and camera movement, explicitly excluding static videos and frames with text overlays or watermarks. After filtering, we leverage the automated pipeline detailed in Sec. III-C to generate dense captions and comprehensive annotations (including subject masks, bounding boxes, and trajectories) for each video. The resulting benchmark comprises a total of 1,027 videos, explicitly categorized into 436 single-subject and 591 multisubject samples, featuring diverse categories spanning humans, general objects, animals, and faces.

Our DreamOmni Bench facilitates a unified evaluation of both identity preservation (covering both generic objects and human faces) and motion control precision (measuring bounding box and trajectory accuracy). Specific quantitative metrics are detailed in Sec. IV-A, while visual samples from the benchmark are provided in Fig. 4.

IV. EXPERIMENT A. Experimental Setup

Implementation details. We utilize Wan2.1-1.3B T2V as our foundational model. Across all training stages, we employ the AdamW optimizer [101] to process video clips at a resolution of 480 × 832 with 49 frames. 1) In the first omni-motion and identity SFT stage, the model is fine-tuned for 40,000 iterations on 64 NVIDIA A100 GPUs with total batch size 64, using a learning rate of 5 × 10−5 and weight decay of 1×10−3. To enhance robustness, we randomly drop bounding box and trajectory conditions with a probability of p = 0.5, and apply data augmentation to reference images with the same probability (p = 0.5). The reweighted diffusion loss weight λ1 is set to 2. 2) The second latent identity reinforcement learning stage comprises two sub-steps: Latent Identity Reward Model (LIRM) training and Latent Identity Reward Feedback Learning (LIReFL). Both are conducted on 16 A100 GPUs with a batch size of 16 and a weight decay of 1×10−2. For LIRM training, we initialize the backbone using the first 8 layers of Wan2.1-1.3B and train for ∼4,000 steps. We use differential learning rates: 1×10−5 for the prediction head and attention layer, and 1×10−6 for the VDM backbone. During training, we freeze the text and patch embedding layers of the pretrained VDM. For LIReFL, we fine-tune the DiT from the SFT stage for 3,400 steps while keeping the reward model frozen. We incorporate an SFT loss as a regularizer with a weight of λ2 = 0.1, and set the learning rate to 5 × 10−6. The condition dropping and reference augmentation strategies follow the same protocol as the SFT stage. During inference, we employ the UniPC scheduler [89] with 50 steps and a classifier-free guidance scale [102] of 5.0.

Baselines. Due to the absence of open-source methods capable of simultaneously supporting multi-subject customization and comprehensive motion control, we benchmark DreamVideo-Omni against prior methods from three distinct categories on both the DreamOmni Bench and MSRVTTPersonalization Bench [49]. On the DreamOmni Bench, we compare with: (1) DreamVideo-2 [17], representing singlesubject customization with motion control; (2) VACE [103] and Phantom [12], focusing on single and multi-subject customization; and (3) Wan-Move [16], specializing in trajectory control. Additionally, on the MSRVTT-Personalization Bench, we compare with the recent state-of-the-art methods Video Alchemist [49] and Tora2 [20]. Since their codes are not publicly available, we cite the quantitative results directly from their original papers.

Evaluation metrics. We quantitatively evaluate our method using 6 metrics across three comprehensive dimensions: 1) Overall Consistency. To assess the semantic alignment between generated videos and text prompts, we employ the CLIP-Text similarity (CLIP-T) using the CLIP ViT-B/32 [21]. 2) Subject and Face Fidelity. Whole-image similarity metrics are inadequate for multi-subject customization, as background noise and other subjects interfere with accurate identity extraction. To address this, we adopt region-based metrics to evaluate both subject and face identity preservation, specifically Region CLIP-Image similarity (R-CLIP), Region DINOImage similarity (R-DINO), and Face Similarity (Face-S). For R-CLIP and R-DINO, we utilize GroundingDINO [94] to detect and crop subject regions in generated frames based on their textual tags. We then compute the cosine similarity between the cropped regions and reference images using CLIP ViT-B/32 and DINO-ViT-S/16 [22], respectively. For FaceS, we employ the InsightFace library with ArcFace [104] for identity verification. To handle multi-person scenarios, we detect all faces in the generated frames and extract their embeddings; subsequently, we compute the cosine similarity between each detected face and the reference face, matching the generated face with the highest similarity to the ground truth for evaluation. 3) Motion Control Precision. We employ Mean Intersection over Union (mIoU) and End Point Error (EPE) to measure the accuracy of spatial layout and trajectory control. For mIoU, we detect subjects in the generated videos using GroundingDINO and calculate the overlap between the detected bounding boxes and the ground-truth control boxes. For fine-grained trajectory evaluation (EPE), we initialize query points using the ground-truth coordinates from the first frame. These points are tracked in the generated video using CoTracker3 [96], and EPE is computed as the average Euclidean distance between tracked and ground-truth trajectories.

B. Main Results

Subject customization with omni-motion control. To evaluate the capability of our framework in joint subject customization and motion control, we benchmark DreamVideoOmni against DreamVideo-2 [17], a representative baseline in this domain. It is worth noting that while recent works like Tora2 [20] explore this task, their code remains closed-source,

|[Figure 153]|[Figure 154]| |
|---|---|---|

|[Figure 155]|[Figure 156]| |
|---|---|---|

A cute little dog happily running on a sunlit grassy field. Its ears stand up and its tongue out, looking very excited and lively. …

A person wearing a black top manipulating a transparent crystal ball with both hands by the seaside.

...

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

DreamVideo-2

###### DreamVideo-2

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Ours Ours

- Fig. 5. Qualitative comparison of joint subject customization and motion control. Previous methods struggle to balance identity preservation with accurate motion control. In contrast, our method delivers high-fidelity subject customization that strictly follows complex spatial trajectories.

TABLE II QUANTITATIVE COMPARISON ON DREAMOMNI BENCH.

###### Method R-CLIP ↑ R-DINO ↑ Face-S ↑ mIoU ↑ EPE ↓ CLIP-T ↑

DreamVideo-2 [17] 0.731 0.429 0.157 0.212 24.05 0.297 DreamVideo-Omni (Ours) 0.739 0.499 0.301 0.558 9.31 0.308

TABLE III QUANTITATIVE COMPARISON ON MSRVTT-PERSONALIZATION BENCH. WE FOLLOW THE EXPERIMENTAL SETTINGS AND EVALUATION PROTOCOLS OF TORA2 [20] AND VIDEO ALCHEMIST [49], REPORTING RESULTS FROM THEIR ORIGINAL PAPERS.

Subject Mode Face Mode CLIP-T ↑ R-DINO ↑ EPE ↓ CLIP-T ↑ Face-S ↑ EPE ↓

Method

Tora + Flux.1 [79] 0.254 0.587 19.72 0.265 0.363 17.41 Video Alchemist [49] 0.268 0.626 - 0.272 0.411 Tora2 [20] 0.273 0.615 17.43 0.274 0.419 13.52 DreamVideo-Omni (Ours) 0.273 0.628 11.21 0.273 0.417 8.50

precluding direct comparison. Furthermore, DreamVideo-2 is inherently limited to single-subject customization with coarse bounding box control. In contrast, our DreamVideo-Omni supports a more versatile setting, enabling multi-subject customization with omni-motion control (combining boxes and trajectories). For a fair comparison, we restrict our evaluation to single-subject scenarios compatible with the baselines.

from identity degradation under large motion. Conversely, for complex trajectories shown on the right, DreamVideo-2 often exhibits trajectory drift or static motion, failing to achieve precise motion control, while our method precisely aligns with the bounding boxes and trajectories. These results reveal that DreamVideo-2 struggles to balance identity preservation and motion control. In contrast, our framework effectively resolves this conflict, simultaneously achieving precise motion control and high-fidelity subject customization.

Tables II and III present the quantitative results on DreamOmni Bench and MSRVTT-Personalization Bench [49], respectively. As observed, DreamVideo-Omni significantly outperforms DreamVideo-2 across all metrics on DreamOmni Bench, demonstrating our superior subject customization and precise motion control capabilities. Results on MSRVTTPersonalization further validate our robustness: in Subject Mode, we achieve the highest R-DINO and best EPE scores. For Face Mode, while our Face-S score is comparable to Tora2 due to the limited video quality and resolution of the MSRVTT dataset, we still achieve significantly better EPE. These consistent improvements across diverse benchmarks underscore the generalization of DreamVideo-Omni in delivering high-fidelity subject customization and precise motion control.

Subject customization. We evaluate the pure subject customization capability of DreamVideo-Omni on the DreamOmni Bench, comparing it against state-of-the-art methods including VACE [103] and Phantom [12] across both single-subject and multi-subject scenarios.

Table IV shows that our method achieves state-of-theart performance on single-subject customization. Specifically, DreamVideo-Omni yields the highest R-DINO and R-CLIP scores, indicating superior identity preservation. In the more challenging multi-subject setting, our method consistently surpasses baselines in R-DINO, Face-S, and CLIP-T scores. This demonstrates that our design effectively prevents identity mixing and leakage while achieving superior text alignment.

Fig. 5 demonstrates the robustness of DreamVideo-Omni across diverse scenarios. Fig. 5 (left) shows that while our method maintains high identity fidelity, DreamVideo-2 suffers

Fig. 6 (left) illustrates the results in a single-subject scenario where the subject reveals her face from behind a leaf.

[Figure 169]

11

[Figure 170]

A man and a woman interacting in a modern kitchen. The woman, …, is focused on chopping vegetables, while the man, …, is busy around the kitchen. … The kitchen is …, featuring white cabinets and decorative tiles on the walls. …

| |
|---|

| |[Figure 171]<br><br>[Figure 172]| |
|---|---|---|

A woman in front of a yellow background, wearing a white turtleneck sweater and a pink rabbit ears headband. Her expression changes from surprise to confusion, and finally back to surprise. Throughout the video, the camera angle remains unchanged, focusing on the woman's face …

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

VACE

VACE

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

Phantom

Phantom

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Ours

Ours

- Fig. 6. Qualitative comparison of subject customization. DreamVideo-Omni generates videos with accurate subject appearance and enhanced motion dynamics, aligning with provided prompts.

TABLE IV QUANTITATIVE COMPARISON OF SUBJECT CUSTOMIZATION ON DREAMOMNI BENCH. WE REPORT RESULTS SEPARATELY FOR SINGLE-SUBJECT AND MULTI-SUBJECT SCENARIOS. ALL MODELS HAVE 1.3B PARAMETERS.

TABLE V QUANTITATIVE COMPARISON OF MOTION CONTROL CAPABILITY ON DREAMOMNI BENCH. WE REPORT RESULTS SEPARATELY FOR SINGLE-SUBJECT AND MULTI-SUBJECT SCENARIOS.

Method mIoU ↑ EPE ↓ CLIP-T ↑ Single-Subject Mode

Method R-CLIP ↑ R-DINO ↑ Face-S ↑ CLIP-T ↑ Single-Subject Mode

0 7 31

Tora [79] (T2V, 1.1B) 0.163 31.74 0.307 Wan-Move [16] (I2V, 14B) 0.507 14.43 0.305 DreamVideo-Omni (T2V, 1.3B) 0.558 9.31 0.308

VACE [103] 0.732 0.480 0.174 0.293 Phantom [12] 0.738 0.485 0.299 0.296 DreamVideo-Omni 0.739 0.499 0.301 0.308

###### Multi-Subject Mode

###### Multi-Subject Mode

Tora [79] (T2V, 1.1B) 0.162 32.84 0.306 Wan-Move [16] (I2V, 14B) 0.541 9.02 0.303 DreamVideo-Omni (T2V, 1.3B) 0.570 6.08 0.306

VACE [103] 0.719 0.497 0.275 0.293 Phantom [12] 0.722 0.517 0.305 0.293 DreamVideo-Omni 0.720 0.524 0.329 0.306

a 1.3B parameter model, DreamVideo-Omni surpasses the 14B parameter Wan-Move across all metrics in both settings, demonstrating the significant parameter efficiency and superior controllability of our method.

VACE exhibits suboptimal facial preservation and unnatural motion, while Phantom generates unexpected multiple faces. In contrast, our method generates a natural transition with the face faithfully preserved. In the multi-subject scenario involving a complex interaction between a man and a woman, both baselines fail to preserve identity fidelity, exhibiting discrepancies in skin tone, hairstyle, facial details, and clothing color compared to the reference. Conversely, DreamVideoOmni successfully disentangles the two identities, rendering distinct, correct appearances with coherent interactions. This validates the effectiveness of our approach in maintaining high-fidelity identity, even in challenging scenarios.

Fig. 7 further highlights these differences. Tora struggles with trajectory adherence, failing to maintain effective control in multi-subject settings. While Wan-Move generates highquality visual content, it tends to deviate from complex trajectories, as evidenced by the inaccurate path of the soccer ball and the loose alignment between the puppy and the girl’s hand movements. In contrast, DreamVideo-Omni precisely follows complex motion trajectories. In the single-subject case, the man’s interaction with the soccer ball strictly adheres to the intricate looping trajectory. In the multi-subject scenario, our model accurately coordinates the distinct movements of the girl and the puppy, verifying our robust control capabilities.

Motion control. To validate the effectiveness of our motion control capabilities, we compare DreamVideo-Omni against state-of-the-art models, including the Tora [79] (T2V, 1.1B) and the large-scale Wan-Move [16] (I2V, 14B).

User study. To further evaluate the perceptual quality of generated videos, we conduct a comprehensive user study focusing on three distinct capabilities: joint subject customization with motion control, pure subject customization, and pure motion control. We invite 18 evaluators to rate 270 groups of videos generated by different methods. Each evaluation group

Table V shows that DreamVideo-Omni consistently outperforms baselines across both single-subject and multi-subject scenarios. Compared to Tora, our method achieves a substantial improvement in motion precision, with mIoU increasing by 0.395 and EPE reducing by nearly 70%. Notably, despite being

A man wearing a white T-shirt interacting with a soccer ball in front of an orange background. At the beginning of the video, the man has his eyes closed, seemingly in deep thought or feeling. Then, he opens his eyes, smiling, and starts to play with the soccer ball. He gently tosses the ball up and catches it, and …

A heartwarming scene of a little girl interacting with a puppy in the park. The little girl, wearing an orange jacket and black pants, is sitting on the grass, holding some food in her hands. The puppy looks very excited; it stands up and stretches out its front paws, seemingly waiting for the food from the little girl. …

|[Figure 192]<br><br>Tora|[Figure 193]|[Figure 194]|
|---|---|---|

|[Figure 195]<br><br>Tora|[Figure 196]|[Figure 197]|
|---|---|---|

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Wan-Move

Wan-Move

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Ours

Ours

- Fig. 7. Qualitative comparison of motion control. Our DreamVideo-Omni achieves precise motion trajectory control.

TABLE VI HUMAN EVALUATION. WE REPORT THE PERCENTAGE OF USER VOTES FOR EACH METHOD ACROSS DIFFERENT SETTINGS.

Setting Method

Subject Fidelity

Motion Consistency

Text Alignment

Overall Quality

Joint ID & Motion

DreamVideo-2 [17] 22.4% 18.3% 21.5% 10.8% Ours 77.6% 81.7% 78.5% 89.2%

Pure Subject Customization

VACE [103] 16.3% - 15.6% 19.5% Phantom [12] 19.5% - 16.8% 20.2% Ours 64.2% - 67.6% 60.3%

Pure Motion Control

Tora [79] - 9.5% 16.5% 13.4% Wan-Move [16] - 20.2% 20.4% 26.4% Ours - 70.3% 63.1% 60.2%

consists of reference subject images, a target textual prompt, corresponding motion conditions (i.e., bounding boxes or trajectories), and the videos generated by competing methods. Evaluators are asked to select the best video based on four criteria: Subject Fidelity, Motion Consistency, Text Alignment, and Overall Quality. Table VI shows that our method achieves the highest user preference across diverse settings.

C. Emergent Capabilities

Benefiting from the robust multi-subject customization and omni-motion control capabilities of DreamVideo-Omni, our multi-task training paradigm facilitates the emergence of novel generative abilities: Image-to-Video (I2V) generation and first-frame-conditioned trajectory control, despite our base DiT being a text-to-video (T2V) model, as illustrated in

- Fig. 8. In fact, I2V generation can be considered as a specialized form of customization, where the entire first frame serves as the comprehensive reference identity. Furthermore, our versatile omni-motion mechanism seamlessly extends to this setting, enabling precise spatial trajectory guidance directly conditioned on the provided initial frame. These emergent properties demonstrate the strong generalization and unified control capacity of our framework without requiring taskspecific fine-tuning.

TABLE VII QUANTITATIVE ABLATION STUDIES ON EACH COMPONENT IN DREAMVIDEO-OMNI. WE REPORT RESULTS SEPARATELY FOR SINGLE-SUBJECT AND MULTI-SUBJECT SCENARIOS TO ANALYZE THE IMPACT OF EACH MODULE.

15 22 48

8 21 40

Method R-CLIP↑ R-DINO↑ Face-S↑ mIoU↑ EPE↓ CLIP-T↑ Single-Subject Mode

w/o Cond-Aware 3D RoPE 0.625 0.139 0.039 0.274 30.22 0.216 w/o Group & Role Emb. 0.738 0.486 0.254 0.524 26.24 0.309 w/o Hierarchical BBox Injection 0.733 0.508 0.257 0.400 31.84 0.307 Ours (Stage1) 0.733 0.483 0.251 0.556 10.53 0.306

w/o LIReFL (Stage2 SFT only) 0.735 0.487 0.266 0.561 10.01 0.307 Ours (Full) 0.739 0.499 0.301 0.558 9.31 0.308

###### Multi-Subject Mode

w/o Cond-Aware 3D RoPE 0.647 0.157 0.047 0.278 20.71 0.224 w/o Group & Role Emb. 0.708 0.503 0.289 0.459 20.69 0.308 w/o Hierarchical BBox Injection 0.714 0.510 0.269 0.289 25.56 0.305 Ours (Stage1) 0.713 0.506 0.287 0.532 6.80 0.305

w/o LIReFL (Stage2 SFT only) 0.715 0.512 0.316 0.556 6.29 0.306 Ours (Full) 0.720 0.524 0.329 0.570 6.08 0.306

D. Ablation Studies

Effects of each component in DreamVideo-Omni. To investigate the contribution of individual components, we conduct an ablation study under both single-subject and multisubject settings on the DreamOmni Bench.

Table VII reveals the critical role of each component. (i) Removing the condition-aware 3D RoPE leads to a catastrophic performance drop across all metrics in both scenarios, confirming its fundamental role in handling multi-condition heterogeneous inputs within the unified DiT framework. (ii) Ablating the group and role embeddings degrades motion control precision, resulting in inferior mIoU and EPE metrics, particularly in multi-subject modes. (iii) Omitting hierarchical BBox injection, where bounding box latents are added solely to the input noisy latents, leads to a significant collapse in motion performance, with mIoU dropping to 0.289 in multi-subject mode. This underscores that simple input-level fusion is insufficient and hierarchical injection is critical for effective motion control. (iv) Regarding the training paradigm,

[Figure 210]

|[Figure 211]|[Figure 212]|
|---|---|

[Figure 213]

Close-up of two people exchanging flowers in front of a white background. They were wearing white clothes, the person on the left had dark skin and the person on the right had yellow skin. They held the red flowers in their hands and then exchanged the flowers in their hands.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

|[Figure 219]|[Figure 220]|
|---|---|

Panoramic view of a cartoon character climbing the stairs. In the flat shot, he was wearing a white shirt and green pants, a pair of red shoes and a brown backpack. His hair is brown and he wears glasses. The background is a simple white wall with a green door frame on it.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

- Fig. 8. Emergent generative capabilities of DreamVideo-Omni. Despite being built on a text-to-video (T2V) base model, our framework naturally enables zero-shot Image-to-Video (I2V) generation and first-frame-conditioned trajectory control without task-specific fine-tuning.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Ours

w/o Cond-Aware 3D RoPE

w/o Group & Role Embed

w/o Hierarchical BBox Injection

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

w/o LIReFL

|[Figure 234]|[Figure 235]|
|---|---|
| | |

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

A group of divers exploring a coral reef in clear seawater. Wearing professional diving gear, the diver move slowly over the reef, observing the marine life around them. Fish swim among the corals, adding lively movement to the scene. The video is shot from a ﬁxed angle, allowing viewers to clearly see the interac on between the divers and the ﬁsh in their natural environment.

- Fig. 9. Ablation study of each component in DreamVideo-Omni.

TABLE VIII ABLATION STUDIES ON THE LATENT IDENTITY REWARD MODEL. THE COLUMNS DENOTE FIVE INTERVALS OF NORMALIZED DIFFUSION TIMESTEPS t ∈ [0, 1]. THE FIRST ROW REPRESENTS OUR DEFAULT SETTING (BCE LOSS, REF. IMAGE AS Q, FROZEN TEXT & PATCH EMBED.), WHILE SUBSEQUENT ROWS ILLUSTRATE THE IMPACT OF ALTERING SPECIFIC COMPONENTS.

Method [0, 0.2] (0.2, 0.4] (0.4, 0.6] (0.6, 0.8] (0.8, 1.0] Avg Ours (Default) 0.702 0.722 0.709 0.724 0.743 0.720 Optimization Objective

w/ BT Loss [105] 0.491 0.657 0.681 0.706 0.743 0.656 Image Injection Strategy

w/ Ref. Image as KV 0.451 0.555 0.415 0.445 0.408 0.455 Parameter Tuning Scope

Tuning text & patch embed. 0.680 0.718 0.709 0.716 0.752 0.715

TABLE IX ABLATION STUDY ON THE RANGE OF TIMESTEP tm IN LIREFL.

Range of tm R-CLIP↑ R-DINO↑ Face-S↑ mIoU↑ EPE↓ CLIP-T↑ Single-Subject Mode

Last 3 timesteps 0.737 0.494 0.293 0.543 9.98 0.307 All timesteps (Ours) 0.739 0.499 0.301 0.558 9.31 0.308

Multi-Subject Mode

Last 3 timesteps 0.717 0.518 0.324 0.573 6.30 0.307 All timesteps (Ours) 0.720 0.524 0.329 0.570 6.08 0.306

The absence of group and role embeddings leads to control ambiguity, where the model struggles to disentangle multiple subjects or bind specific motions to the correct identities. (iii) Removing hierarchical BBox injection causes subjects to fail in adhering to bounding boxes or trajectories, demonstrating that hierarchical motion injection is essential for achieving effective motion control. (iv) Compared to the standard SFT (w/o LIReFL), the integration of LIReFL effectively refines identity details, delivering the harmonious balance between subject fidelity and motion control.

although Stage 1 establishes a solid baseline, subsequent finetuning via standard SFT (w/o LIReFL) offers limited gains. In contrast, our full model equipped with LIReFL achieves the best performance across most metrics, particularly in multisubject scenarios, effectively boosting subject customization while maintaining precise motion control.

Fig. 9 further corroborates these findings. (i) Without condition-aware 3D RoPE, the generation suffers from training collapse, resulting in severe artifacts or meaningless noise. (ii)

TABLE X ABLATION STUDY ON LOSS WEIGHT λ2 OF LIREFL.

λ2 R-CLIP↑ R-DINO↑ Face-S↑ mIoU↑ EPE↓ CLIP-T↑ Single-Subject Mode

0.01 0.737 0.505 0.279 0.560 9.85 0.307 0.10 (Ours) 0.739 0.499 0.301 0.558 9.31 0.308 0.25 0.735 0.492 0.272 0.555 9.65 0.307

- 0.50 0.718 0.482 0.223 0.541 9.75 0.306
- 1.00 0.674 0.350 0.120 0.350 25.00 0.280 Multi-Subject Mode

0.01 0.718 0.518 0.322 0.557 6.70 0.306 0.10 (Ours) 0.720 0.524 0.329 0.570 6.08 0.306 0.25 0.714 0.515 0.331 0.538 5.95 0.306

- 0.50 0.710 0.504 0.295 0.530 6.73 0.305

- 1.00 0.692 0.380 0.287 0.380 15.06 0.280

Design choices for latent identity reward model. We investigate the impact of different design choices on the pairwise classification accuracy of the latent identity reward model in Table VIII. The accuracy is measured on the test set of winlose pairs, where a prediction is correct if the reward score of the winner video is higher than that of the loser. (i) Regarding the optimization objective, our default Binary Cross-Entropy (BCE) loss yields superior performance (Avg 0.720) compared to the Bradley-Terry (BT) model [105] (Avg 0.656). Notably, the BT loss exhibits significant instability at early timesteps (t ∈ [0,0.2]). (ii) For the image injection strategy, employing the reference image as the Query is critical. Treating it as Key/Value (KV) results in a catastrophic accuracy drop (to 0.455), indicating that the reference image must actively attend to the noisy latents to effectively discern identity features. (iii) Regarding the parameter tuning scope, freezing the text and patch embeddings outperforms fine-tuning them, suggesting that preserving pre-trained priors prevents overfitting and is sufficient for effective reward modeling.

Effect of the range of timestep tm in LIReFL. Table IX investigates the influence of the range of timestep tm on model performance. We compare sparse feedback on the last 3 steps against dense feedback across all timesteps. In singlesubject mode, this full-range strategy yields comprehensive improvements across all metrics. In multi-subject scenarios, it further enhances identity metrics while maintaining precise motion control. These findings indicate that providing reward feedback at arbitrary timesteps is essential to fully leverage the potential of LIReFL, enhancing identity fidelity throughout the generation process.

Effect of loss weight λ2 in LIReFL. Table X illustrates the impact of varying the feedback learning strength λ2. We observe that values within the range of [0.01,0.25] yield robust overall performance. Notably, identity fidelity metrics, such as R-DINO and Face-S, consistently surpass those of the SFT-only model (see Table VII), indicating that identity reward feedback effectively enhances customization and refines appearance details. However, performance begins to decline when λ2 increases to 0.50. Furthermore, at λ2 = 1.00, the model suffers from reward hacking, where it finds shortcuts to maximize the reward at the expense of visual realism and mo-

tion coherence. This suggests that excessively strong feedback disrupts the generative process, a phenomenon also observed in previous reinforcement learning studies [106], [88]. Balancing robust identity preservation with motion control, we adopt λ2 = 0.10 as the optimal configuration for our final model.

V. CONCLUSION

In this work, we present DreamVideo-Omni, a unified framework that achieves harmonious multi-subject customization with omni-motion control, encompassing global and local object motion as well as camera movement. By introducing a progressive two-stage training paradigm, we effectively resolve the conflict between identity preservation and complex motion control. Specifically, to ensure robust and precise controllability, we incorporate a condition-aware 3D RoPE to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance. To further eliminate control ambiguity in multi-subject scenarios, we propose group and role embeddings to explicitly bind motion signals to their corresponding identities. Furthermore, we design a latent identity reward feedback learning paradigm, leveraging a VDM-based latent identity reward model to prioritize motionaware identity preservation aligned with human preferences. To advance the field, we design a comprehensive automated data construction pipeline and establish the DreamOmni Bench, a holistic benchmark for evaluating multi-subject and omni-motion control. Extensive experiments demonstrate that DreamVideo-Omni significantly outperforms state-of-the-art methods in generating high-quality, customizable videos with precise and flexible motion control.

REFERENCES

- [1] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” arXiv preprint arXiv:2204.03458, 2022.
- [2] Y. He, T. Yang, Y. Zhang, Y. Shan, and Q. Chen, “Latent video diffusion models for high-fidelity long video generation,” arXiv preprint arXiv:2211.13221, 2022.
- [3] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang, “Modelscope text-to-video technical report,” arXiv preprint arXiv:2308.06571, 2023.
- [4] H. Chen, M. Xia, Y. He, Y. Zhang, X. Cun, S. Yang, J. Xing, Y. Liu, Q. Chen, X. Wang et al., “Videocrafter1: Open diffusion models for high-quality video generation,” arXiv preprint arXiv:2310.19512, 2023.
- [5] Y. Wang, X. Chen, X. Ma, S. Zhou, Z. Huang, Y. Wang, C. Yang, Y. He, J. Yu, P. Yang et al., “Lavie: High-quality video generation with cascaded latent diffusion models,” arXiv preprint arXiv:2309.15103, 2023.
- [6] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang, “Cogvideo: Largescale pretraining for text-to-video generation via transformers,” arXiv preprint arXiv:2205.15868, 2022.
- [7] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-tovideo diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.
- [8] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.
- [9] Y. Wei, S. Zhang, H. Yuan, Y. Han, Z. Chen, J. Wang, D. Zou, X. Liu, Y. Zhang, Y. Liu et al., “Routing matters in moe: Scaling diffusion transformers with explicit routing guidance,” arXiv preprint arXiv:2510.24711, 2025.
- [10] Y. Jiang, T. Wu, S. Yang, C. Si, D. Lin, Y. Qiao, C. C. Loy, and Z. Liu, “Videobooth: Diffusion-based video generation with image prompts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6689–6700.

- [11] Y. Wei, S. Zhang, Z. Qing, H. Yuan, Z. Liu, Y. Liu, Y. Zhang, J. Zhou, and H. Shan, “Dreamvideo: Composing your dream videos with customized subject and motion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6537–6549.
- [12] L. Liu, T. Ma, B. Li, Z. Chen, J. Liu, G. Li, S. Zhou, Q. He, and X. Wu, “Phantom: Subject-consistent video generation via cross-modal alignment,” arXiv preprint arXiv:2502.11079, 2025.
- [13] Z. Wang, Z. Yuan, X. Wang, Y. Li, T. Chen, M. Xia, P. Luo, and Y. Shan, “Motionctrl: A unified and flexible motion controller for video generation,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
- [14] J. Wang, Y. Zhang, J. Zou, Y. Zeng, G. Wei, L. Yuan, and H. Li, “Boximator: Generating rich and controllable motions for video synthesis,” arXiv preprint arXiv:2402.01566, 2024.
- [15] J. Xing, L. Mai, C. Ham, J. Huang, A. Mahapatra, C.-W. Fu, T.T. Wong, and F. Liu, “Motioncanvas: Cinematic shot design with controllable image-to-video generation,” in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2025, pp. 1–11.
- [16] R. Chu, Y. He, Z. Chen, S. Zhang, X. Xu, B. Xia, D. Wang, H. Yi, X. Liu, H. Zhao et al., “Wan-move: Motion-controllable video generation via latent trajectory guidance,” arXiv preprint arXiv:2512.08765, 2025.
- [17] Y. Wei, S. Zhang, H. Yuan, X. Wang, H. Qiu, R. Zhao, Y. Feng, F. Liu, Z. Huang, J. Ye et al., “Dreamvideo-2: Zero-shot subjectdriven video customization with precise motion control,” arXiv preprint arXiv:2410.13830, 2024.
- [18] X. Ju, W. Ye, Q. Liu, Q. Wang, X. Wang, P. Wan, D. Zhang, K. Gai, and Q. Xu, “Fulldit: Multi-task video generative foundation model with full attention,” arXiv preprint arXiv:2503.19907, 2025.
- [19] Y. Cai, H. Zhang, X. Chen, J. Xing, Y. Hu, Y. Zhou, K. Zhang, Z. Zhang, S. Y. Kim, T. Wang et al., “Omnivcus: Feedforward subjectdriven video customization with multimodal control conditions,” arXiv preprint arXiv:2506.23361, 2025.
- [20] Z. Zhang, J. Liao, X. Meng, L. Qin, and W. Wang, “Tora2: Motion and appearance customized diffusion transformer for multi-entity video generation,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 9434–9443.
- [21] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning, 2021, pp. 8748–8763.
- [22] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 9650–9660.
- [23] H. Chen, Y. Zhang, S. Wu, X. Wang, X. Duan, Y. Zhou, and W. Zhu, “Disenbooth: Identity-preserving disentangled tuning for subject-driven text-to-image generation,” arXiv preprint arXiv:2305.03374, 2023.
- [24] Y. Wei, Y. Zhang, Z. Ji, J. Bai, L. Zhang, and W. Zuo, “Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15943–15953.
- [25] J. Shi, W. Xiong, Z. Lin, and H. J. Jung, “Instantbooth: Personalized text-to-image generation without test-time finetuning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8543–8552.
- [26] N. Ruiz, Y. Li, V. Jampani, W. Wei, T. Hou, Y. Pritch, N. Wadhwa, M. Rubinstein, and K. Aberman, “Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6527–6536.
- [27] M. Hua, J. Liu, F. Ding, W. Liu, J. Wu, and Q. He, “Dreamtuner: Single image is enough for subject-driven generation,” arXiv preprint arXiv:2312.13691, 2023.
- [28] Y. Han, J. Zhu, K. He, X. Chen, Y. Ge, W. Li, X. Li, J. Zhang, C. Wang, and Y. Liu, “Face adapter for pre-trained diffusion models with finegrained id and attribute control,” arXiv preprint arXiv:2405.12970, 2024.
- [29] Y. Gu, X. Wang, J. Z. Wu, Y. Shi, Y. Chen, Z. Fan, W. Xiao, R. Zhao, S. Chang, W. Wu et al., “Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [30] G. Xiao, T. Yin, W. T. Freeman, F. Durand, and S. Han, “Fastcomposer: Tuning-free multi-subject image generation with localized attention,” arXiv preprint arXiv:2305.10431, 2023.

- [31] N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J.-Y. Zhu, “Multiconcept customization of text-to-image diffusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1931–1941.
- [32] Y. Zhou, D. Zhou, M.-M. Cheng, J. Feng, and Q. Hou, “Storydiffusion: Consistent self-attention for long-range image and video generation,” arXiv preprint arXiv:2405.01434, 2024.
- [33] Z. Li, M. Cao, X. Wang, Z. Qi, M.-M. Cheng, and Y. Shan, “Photomaker: Customizing realistic human photos via stacked id embedding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8640–8650.
- [34] M. Huang, Z. Mao, M. Liu, Q. He, and Y. Zhang, “Realcustom: narrowing real text word for real-time open-domain text-to-image customization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7476–7485.
- [35] E. Molad, E. Horwitz, D. Valevski, A. R. Acha, Y. Matias, Y. Pritch, Y. Leviathan, and Y. Hoshen, “Dreamix: Video diffusion models are general video editors,” arXiv preprint arXiv:2302.01329, 2023.
- [36] H. Chefer, S. Zada, R. Paiss, A. Ephrat, O. Tov, M. Rubinstein, L. Wolf, T. Dekel, T. Michaeli, and I. Mosseri, “Still-moving: Customized video generation without customized video data,” arXiv preprint arXiv:2407.08674, 2024.
- [37] Z. Ma, D. Zhou, C.-H. Yeh, X.-S. Wang, X. Li, H. Yang, Z. Dong, K. Keutzer, and J. Feng, “Magic-me: Identity-specific video customized diffusion,” arXiv preprint arXiv:2402.09368, 2024.
- [38] X. He, Q. Liu, S. Qian, X. Wang, T. Hu, K. Cao, K. Yan, M. Zhou, and J. Zhang, “Id-animator: Zero-shot identity-preserving human video generation,” arXiv preprint arXiv:2404.15275, 2024.
- [39] Y. Zhou, R. Zhang, J. Gu, N. Zhao, J. Shi, and T. Sun, “Sugar: Subjectdriven video customization in a zero-shot manner,” arXiv preprint arXiv:2412.10533, 2024.
- [40] T. Wu, Y. Zhang, X. Cun, Z. Qi, J. Pu, H. Dou, G. Zheng, Y. Shan, and X. Li, “Videomaker: Zero-shot customized video generation with the inherent force of video diffusion models,” arXiv preprint arXiv:2412.19645, 2024.
- [41] Y. Huang, W. Xiong, H. Zhang, C. Chen, J. Liu, M. Yan, and S. Chen, “Dive: Taming dino for subject-driven video editing,” arXiv preprint arXiv:2412.03347, 2024.
- [42] D. She, M. Liu, J. Pang, J. Wang, Z. Yang, W. He, G. Zhang, Y. Wang, Q. Huang, H. Tang et al., “Customvideox: 3d reference attention driven dynamic adaptation for zero-shot customized video diffusion transformers,” arXiv preprint arXiv:2502.06527, 2025.
- [43] T. Wu, Y. Zhang, X. Wang, X. Zhou, G. Zheng, Z. Qi, Y. Shan, and X. Li, “Customcrafter: Customized video generation with preserving motion and concept composition abilities,” arXiv preprint arXiv:2408.13239, 2024.
- [44] Y. Zhang, Q. Wang, F. Jiang, Y. Fan, M. Xu, and Y. Qi, “Fantasyid: Face knowledge enhanced id-preserving video generation,” arXiv preprint arXiv:2502.13995, 2025.
- [45] S. Yuan, J. Huang, X. He, Y. Ge, Y. Shi, L. Chen, J. Luo, and L. Yuan, “Identity-preserving text-to-video generation by frequency decomposition,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 12978–12988.
- [46] Y. Wei, S. Zhang, H. Yuan, B. Gong, L. Tang, X. Wang, H. Qiu, H. Li, S. Tan, Y. Zhang et al., “Dreamrelation: Relation-centric video customization,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 12381–12393.
- [47] Z. Wang, A. Li, E. Xie, L. Zhu, Y. Guo, Q. Dou, and Z. Li, “Customvideo: Customizing text-to-video generation with multiple subjects,” arXiv preprint arXiv:2401.09962, 2024.
- [48] H. Chen, X. Wang, Y. Zhang, Y. Zhou, Z. Zhang, S. Tang, and W. Zhu, “Disenstudio: Customized multi-subject text-to-video generation with disentangled spatial control,” arXiv preprint arXiv:2405.12796, 2024.
- [49] T.-S. Chen, A. Siarohin, W. Menapace, Y. Fang, K. S. Lee, I. Skorokhodov, K. Aberman, J.-Y. Zhu, M.-H. Yang, and S. Tulyakov, “Multi-subject open-set personalization in video generation,” arXiv preprint arXiv:2501.06187, 2025.
- [50] Y. Huang, Z. Yuan, Q. Liu, Q. Wang, X. Wang, R. Zhang, P. Wan, D. Zhang, and K. Gai, “Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning,” arXiv preprint arXiv:2501.04698, 2025.
- [51] C.-P. Huang, Y.-S. Wu, H.-K. Chung, K.-P. Chang, F.-E. Yang, and Y.C. F. Wang, “Videomage: Multi-subject and motion customization of text-to-video diffusion models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 17603–17612.

- [52] J. Wu, X. Li, Y. Zeng, J. Zhang, Q. Zhou, Y. Li, Y. Tong, and K. Chen, “Motionbooth: Motion-aware customized text-to-video generation,” arXiv preprint arXiv:2406.17758, 2024.
- [53] X. Shi, Z. Huang, F.-Y. Wang, W. Bian, D. Li, Y. Zhang, M. Zhang, K. C. Cheung, S. See, H. Qin et al., “Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
- [54] W. Wu, Z. Li, Y. Gu, R. Zhao, Y. He, D. J. Zhang, M. Z. Shou, Y. Li, T. Gao, and D. Zhang, “Draganything: Motion control for anything using entity representation,” in European Conference on Computer Vision. Springer, 2025, pp. 331–348.
- [55] S. Bahmani, X. Liu, W. Yifan, I. Skorokhodov, V. Rong, Z. Liu, X. Liu, J. J. Park, S. Tulyakov, G. Wetzstein et al., “Tc4d: Trajectoryconditioned text-to-4d generation,” in European Conference on Computer Vision. Springer, 2025, pp. 53–72.
- [56] K. Namekata, S. Bahmani, Z. Wu, Y. Kant, I. Gilitschenski, and D. B. Lindell, “Sg-i2v: Self-guided trajectory control in image-to-video generation,” arXiv preprint arXiv:2411.04989, 2024.
- [57] D. Geng, C. Herrmann, J. Hur, F. Cole, S. Zhang, T. Pfaff, T. LopezGuevara, Y. Aytar, M. Rubinstein, C. Sun et al., “Motion prompting: Controlling video generation with motion trajectories,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1–12.
- [58] G. Zheng, T. Li, R. Jiang, Y. Lu, T. Wu, and X. Li, “Cami2v: Camera-controlled image-to-video diffusion model,” arXiv preprint arXiv:2410.15957, 2024.
- [59] T. Hu, J. Zhang, R. Yi, Y. Wang, J. Weng, H. Huang, Y. Wang, and L. Ma, “Comd: Training-free video motion transfer with cameraobject motion disentanglement,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 3459–3468.
- [60] H. Yesiltepe, T. H. S. Meral, C. Dunlop, and P. Yanardag, “Motionshop: Zero-shot motion transfer in video diffusion models with mixture of score guidance,” arXiv preprint arXiv:2412.05355, 2024.
- [61] M. Niu, X. Cun, X. Wang, Y. Zhang, Y. Shan, and Y. Zheng, “Mofa-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model,” in European Conference on Computer Vision. Springer, 2025, pp. 111–128.
- [62] H. Jeong, J. Chang, G. Y. Park, and J. C. Ye, “Dreammotion: Spacetime self-similar score distillation for zero-shot video editing,” in European Conference on Computer Vision. Springer, 2024, pp. 358– 376.
- [63] X. Li, X. Jia, Q. Wang, H. Diao, M. Ge, P. Li, Y. He, and H. Lu, “Motrans: Customized motion transfer with text-driven video diffusion models,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 3421–3430.
- [64] W. Feng, T. Qi, J. Liu, M. Sun, P. Tu, T. Ma, F. Dai, S. Zhao, S. Zhou, and Q. He, “I2vcontrol: Disentangled and unified video motion synthesis control,” arXiv preprint arXiv:2411.17765, 2024.
- [65] G. Y. Park, H. Jeong, S. W. Lee, and J. C. Ye, “Spectral motion alignment for video motion transfer using diffusion models,” arXiv preprint arXiv:2403.15249, 2024.
- [66] H. Jeong, G. Y. Park, and J. C. Ye, “Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9212–9221.
- [67] Y. Ren, Y. Zhou, J. Yang, J. Shi, D. Liu, F. Liu, M. Kwon, and A. Shrivastava, “Customize-a-video: One-shot motion customization of text-to-video diffusion models,” arXiv preprint arXiv:2402.14780, 2024.
- [68] D. Yatim, R. Fridman, O. Bar-Tal, Y. Kasten, and T. Dekel, “Spacetime diffusion features for zero-shot text-driven motion transfer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8466–8476.
- [69] L. Wang, G. Shen, Y. Liang, X. Tao, P. Wan, D. Zhang, Y. Li, and Y. Chen, “Motion inversion for video customization,” arXiv preprint arXiv:2403.20193, 2024.
- [70] R. Wu, L. Chen, T. Yang, C. Guo, C. Li, and X. Zhang, “Lamp: Learn a motion pattern for few-shot-based video generation,” arXiv preprint arXiv:2310.10769, 2023.
- [71] Y. Jain, A. Nasery, V. Vineet, and H. Behl, “Peekaboo: Interactive video generation via masked-diffusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8079–8088.
- [72] S. Yang, L. Hou, H. Huang, C. Ma, P. Wan, D. Zhang, X. Chen, and J. Liao, “Direct-a-video: Customized video generation with userdirected camera movement and object motion,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–12.

- [73] W.-D. K. Ma, J. P. Lewis, and W. B. Kleijn, “Trailblazer: Trajectory control for diffusion-based video generation,” arXiv preprint arXiv:2401.00896, 2023.
- [74] C. Chen, J. Shu, L. Chen, G. He, C. Wang, and Y. Li, “Motion-zero: Zero-shot moving object control framework for diffusion-based video generation,” arXiv preprint arXiv:2401.10150, 2024.
- [75] Y. Huang, Y. Chen, L. Ding, X. Zhang, W. Dai, J. Zou, H. Xiong, and Q. Tian, “Im-zero: Instance-level motion controllable video generation in a zero-shot manner,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 7265–7275.
- [76] S. Yin, C. Wu, J. Liang, J. Shi, H. Li, G. Ming, and N. Duan, “Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory,” arXiv preprint arXiv:2308.08089, 2023.
- [77] Y. Li, X. Wang, Z. Zhang, Z. Wang, Z. Yuan, L. Xie, Y. Zou, and Y. Shan, “Image conductor: Precision control for interactive video synthesis,” arXiv preprint arXiv:2406.15339, 2024.
- [78] Z. Zhang, F. Long, Z. Qiu, Y. Pan, W. Liu, T. Yao, and T. Mei, “Motionpro: A precise motion controller for image-to-video generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 27957–27967.
- [79] Z. Zhang, J. Liao, M. Li, Z. Dai, B. Qiu, S. Zhu, L. Qin, and W. Wang, “Tora: Trajectory-oriented diffusion transformer for video generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 2063–2073.
- [80] Q. Li, Z. Xing, R. Wang, H. Zhang, Q. Dai, and Z. Wu, “Magicmotion: Controllable video generation with dense-to-sparse trajectory guidance,” arXiv preprint arXiv:2503.16421, 2025.
- [81] H. Li, L. Jiang, X. Xiao, T. Wang, H. Yi, B. Wu, and D. Cai, “Magicid: Hybrid preference optimization for id-consistent and dynamicpreserved video customization,” arXiv preprint arXiv:2503.12689, 2025.
- [82] L. Shen, W. Jiang, Y. Zhu, J. Li, T. Ge, Z. Cao, and B. Zheng, “Identitypreserving image-to-video generation via reward-guided optimization,” arXiv preprint arXiv:2510.14255, 2025.
- [83] X. Meng, Z. Zhang, Z. Zhang, J. Liao, L. Qin, and W. Wang, “Identitygrpo: Optimizing multi-human identity-preserving video generation via reinforcement learning,” arXiv preprint arXiv:2510.14256, 2025.
- [84] B. Zhu, Y. Jiang, B. Xu, S. Yang, M. Yin, Y. Wu, H. Sun, and Z. Wu, “Aligning anime video generation with human feedback,” arXiv preprint arXiv:2504.10044, 2025.
- [85] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53728–53741, 2023.
- [86] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.
- [87] J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong, “Imagereward: Learning and evaluating human preferences for textto-image generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 15903–15935, 2023.
- [88] X. Mi, W. Yu, J. Lian, S. Jie, R. Zhong, Z. Liu, G. Zhang, Z. Zhou, Z. Xu, Y. Zhou et al., “Video generation models are good latent reward models,” arXiv preprint arXiv:2511.21541, 2025.
- [89] W. Zhao, L. Bai, Y. Rao, J. Zhou, and J. Lu, “Unipc: A unified predictor-corrector framework for fast sampling of diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 49842–49869, 2023.
- [90] Z. Teed and J. Deng, “Raft: Recurrent all-pairs field transforms for optical flow,” in European conference on computer vision. Springer, 2020, pp. 402–419.
- [91] Y. Zhang, X. Huang, J. Ma, Z. Li, Z. Luo, Y. Xie, Y. Qin, T. Luo, Y. Li, S. Liu et al., “Recognize anything: A strong image tagging model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1724–1732.
- [92] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.
- [93] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng,

- H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu, “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.
- [94] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.
- [95] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. R¨adle, C. Rolland, L. Gustafson et al., “Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024.
- [96] N. Karaev, Y. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht, “Cotracker3: Simpler and better point tracking by pseudolabelling real videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 6013–6022.
- [97] M. Bain, A. Nagrani, G. Varol, and A. Zisserman, “Frozen in time: A joint video and image encoder for end-to-end retrieval,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 1728–1738.
- [98] K. Soomro, A. R. Zamir, and M. Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” arXiv preprint arXiv:1212.0402, 2012.
- [99] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbel´aez, A. Sorkine-Hornung, and L. Van Gool, “The 2017 davis challenge on video object segmentation,” arXiv preprint arXiv:1704.00675, 2017.
- [100] L. Huang, X. Zhao, and K. Huang, “Got-10k: A large high-diversity benchmark for generic object tracking in the wild,” IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 5, pp. 1562– 1577, 2019.
- [101] I. Loshchilov, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.
- [102] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.
- [103] Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu, “Vace: Allin-one video creation and editing,” arXiv preprint arXiv:2503.07598, 2025.
- [104] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4690–4699.
- [105] R. A. Bradley and M. E. Terry, “Rank analysis of incomplete block designs: I. the method of paired comparisons,” Biometrika, vol. 39, no. 3/4, pp. 324–345, 1952.
- [106] J. Skalse, N. Howe, D. Krasheninnikov, and D. Krueger, “Defining and characterizing reward gaming,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 9460–9471.

