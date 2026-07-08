FairyGen: Storied Cartoon Video from a Single Child-Drawn Character

JIAYI ZHENG and XIAODONG CUN∗, GVC Lab, Great Bay University, China

[Figure 1]

[Figure 2]

[Figure 3]

- Shot #1: A purple robot dances joyfully in the spaceship corridor, ready to begin his journey.
- Shot #2: He steps out, leaving the spaceship.
- Shot #3: He slowly descends in a soft beam of light emiting from the spaceship.

- Shot #4: He begins exploring the wonderland.
- Shot #5: He looks around beneath giant, mushroom-shaped trees.
- Shot #6: Finding no life, he leaves, unaware two cat-like creatures quietly emerge.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

arXiv:2506.21272v2[cs.GR]27Jun2025

Shot # 1

Shot # 2

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Shot # 3

Shot # 4

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Shot # 5 Shot # 6

Fig. 1. We present FairyGen, a visual story generation framework to generate multi-shot cartoon videos from a single child-drawn character with consistent style and motion between the foreground and the background. Project page: https://jayleejia.github.io/FairyGen/.

We propose FairyGen, an automatic system for generating story-driven cartoon videos from a single child’s drawing, while faithfully preserving its unique artistic style. Unlike previous storytelling methods that primarily focus on character consistency and basic motion, FairyGen explicitly disentangles character modeling from stylized background generation and incorporates cinematic shot design to support expressive and coherent storytelling. Given a single character sketch, we first employ a Multimodal Large Language Model (MLLM) to generate a structured storyboard with shot-level descriptions that specify environment settings, character actions, and camera perspectives. To ensure visual consistency, we introduce a style propagation adapter that captures the character’s visual style and applies it to the background, faithfully retaining the character’s full visual identity while synthesizing style-consistent scenes. A shot design module further enhances visual diversity and cinematic quality through frame cropping and multiview synthesis based on the storyboard. To animate the story, we reconstruct a 3D proxy of the character to derive physically plausible motion sequences, which are then used to fine-tune an MMDiT-based image-to-video diffusion model. We further propose a two-stage motion customization adapter: the first stage learns appearance features from temporally unordered frames, disentangling identity from motion; the second stage models temporal dynamics using a timestep-shift strategy with frozen identity weights. Once trained, FairyGen directly renders diverse and coherent video scenes aligned with the storyboard. Extensive experiments demonstrate that our system produces animations that are stylistically faithful, narratively structured, and rich in smooth, natural motion, highlighting its potential for personalized and engaging story animation.

1 INTRODUCTION

Children often express vivid imagination through abstract, stylized drawings featuring simple cartoon characters and expressive visual elements. Despite their lack of photorealistic detail, these illustrations convey unique artistic styles and emotional intent. Transforming such drawings into coherent animated stories bridges youthful creativity with expressive storytelling, offering promising applications in education, digital art therapy, personalized content creation, and interactive entertainment. Story visualization has become a major research area in computer graphics, with recent advances in generative video model continue to expand its horizons. Prior works such as StoryGAN [Li et al. 2019], AR-LDM [Pan et al. 2024a] and Make-A-Story [Rahman et al. 2023] improve visual fidelity and semantic coherence, but are limited by style diversity and training data constraints. More recent LLM-driven pipelines like TaleCrafter [Gong et al. 2023], DreamStory [He et al. 2024], and AnimateA-Story [He et al. 2023] introduce modular task decomposition for better controllability but often suffer from character inconsistency, fragmented narratives, and weak motion quality. Diffusion-based video models such as MEVG [Oh et al. 2024], MovieDreamer [Zhao et al. 2024], and Vlogger [Zhuang et al. 2024] improve temporal consistency and narrative flow, yet still struggle with cross-shot character consistency, artistic style preservation, and complex motion synthesis—primarily due to reliance on real-world data priors. These limitations become more pronounced when dealing with

∗Corresponding Author

abstract hand-drawn characters, especially in single-example scenarios where the input style diverges significantly from training data. To address this, we consider cartoon story generation with a decoupled design that explicitly separates the foreground character from background synthesis. Using 3D reconstruction that adheres to physical constraints, we preserve the character identity while enabling plausible motion generation. In coordination, background generation is treated as a style adaptation process that propagates the character’s visual style to scene elements [Brooks and Efros 2022; Kulal et al. 2023; Pan et al. 2024b]. This decoupled design further extends to video generation, where first learns the motion priors from 3D-derived sequences, and then animate the full scene using a large pre-trained video diffusion model. This pipeline naturally preserves character consistency, supports complex motion, and enables cinematic storytelling. Building on these insights, we present FairyGen, a novel framework for generating animated story videos from a single hand-drawn character. FairyGen produces expressive motion, stylistically aligned backgrounds, and cinematic compositions without requiring additional training data. Specifically, given a hand-drawn character, we first generate a structured storyboard that describes actions, settings, and camera compositions using an MLLM. To ensure visual consistency, we propose a style propagation adapter that preserves the character’s full visual identity while learning its stylistic traits, which are then propagated to the background using a pre-trained inpainting diffusion model. Next, to animate the story, we reconstruct a 3D proxy of the character and derive physically plausible motion sequences, which are then used to finetune an MMDiT-based image-to-video diffusion model [Wan et al. 2025]. For robust motion synthesis, we introduce a two-stage motion customization adapter: the first stage learns spatial features from temporally shuffled frames to remove temporal bias, and the second stage captures dynamics via a novel timestep-shift strategy with identity weights frozen, ensuring smooth and coherent animation. Extensive experiments demonstrate that FairyGen effectively generates personalized animated stories that are stylistically consistent, narratively coherent, and rich in natural motion. In summary, our main contributions are as follows:

- • We present a novel story video generation framework that synthesizes stylistically consistent, narratively coherent, and temporally smooth animations from a single child-drawn character image.
- • We propose a novel style propagation adapter that learns fromcharacterillustrationsandgenerates background scenes in a compatible style while preserving character-specific visual and semantic features.
- • We demonstrate that shifting diffusion timesteps in the image-to-videogenerationsignificantly enhances the model’s ability to learn natural, fluid motions.

2 RELATED WORK

Story Generation. Generating the visual story from the text description contains many challenging problems, including the consistency of the overall style of the video, the consistency of the subject, etc. Early works directly train models based on a well-prepared small dataset utilizing a GAN-based framework [Li 2022; Li et al. 2019]

or transformer [Chen et al. 2022; Maharana et al. 2022] to directly generate the story video. Recently, image and video diffusion models [Ho et al. 2020; Rombach et al. 2022a] and the Large Language Models [DeepSeek 2025; OpenAI 2022; Qwen 2025] bring the universal generative priors to this problem for planning, generation, and animation. SEED-Story [Yang et al. 2024a] directly trains the Large Language Model for the consistency and long story visualization. As for diffusion-based methods, TaleCrafter [Gong et al. 2023] proposes a multi-stage framework to generate the visual story from the customized text-to-image model with a traditional camera movement, which is then extended by AutoStory [Wang et al. 2024] with more conditions. StoryDiffusion [Zhou et al. 2024b] proposes a customized multi-frame attention layer for consistent identity generation, similar to [Liu et al. 2024c]. StoryAgent [Hu et al. 2024] involves the multiple stages of the LLM as the agent for storytelling, etc. However, all these methods [Su et al. 2023; Tao et al. 2024; Zhang et al. 2025b; Zhu and Tang 2025] focus mainly on a custom identity pipeline for characters, then the video can be generated via the pre-trained video diffusion model. We argue that this kind of pipeline is still hard to customize the child-draw images with a similar corresponding style, and is difficult to generate a suitable motion utilizing the generation prior directly. Different, to avoid the complex movement of the foreground character, we involve a 3D proxy to solve the complex motion and the character consistency.

Customized Generation. In the era of the large generative model, fine-tuning the text-to-image/video models [Chen et al. 2023, 2024; Kong et al. 2024; Wan et al. 2025; Yang et al. 2024b] for customizing usage is natural and practical. Our method also shares a similar inspiration with customization methods since we need to generate the customized background style and motions. For subject customization, early works [Gal et al. 2022; Hu et al. 2021; Ruiz

- et al. 2023] focus on single subject customization via additional parameter-efficient training, as well as multiple subjects [Kumari et al. 2023; Yuan et al. 2023], and also for video [Jiang et al. 2024]. As for both subject and motion customization. DreamVideo [Wei
- et al. 2024], CustomTTT [Bi et al. 2024], MotionDirector [Zhao et al.
- 2025] trains different LoRAs [Hu et al. 2021] for appearance and motion customization. DynamicConcept [Abdal et al. 2025] trains a two-stage LoRAs [Hu et al. 2021] for customized motion generation. Stylization customization shares a similar idea behind the subject customization. For example, B-LoRA [Frenkel et al. 2025] splits the subject and style via layer-specific learning. StyleDrop [Sohn et al. 2023] increases the stylization effects via iterative training. However, it is still unclear how to utilize these methods in our child-draw background generation with a given foreground character and in the image-to-video framework.

3 PRELIMINARIRES

Latent Diffusion Model. Most of the current generation models are based on the latent diffusion model [Rombach et al. 2022b]. Take the image diffusion model as an example, it contains a pre-trained VAE to encode/decode the image to the latent space. Then, the diffusion model aims to train a denosing network to remove the added singlestep noise via the simple MSE loss. Formally, given the image 𝐼 and its corresponding latent 𝑧 = E(𝐼), we first add the 𝑡 step noise 𝜖 to

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|Sec. 4.2 Propagated Stylization<br><br>[Figure 28]| |
|---|---|
| | |

|[Figure 29]<br><br>Sec. 4.3 Motion Sequence Gen via 3D Proxy|
|---|

[Figure 30]

[Figure 31]

Tell me a story about this character?

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Stylized image

| | |
|---|---|
|Storyboard Planning<br><br>Actions Planning<br><br>[Figure 36]<br><br>[Figure 37]<br><br>Sec.4.1 Story Planning| |

Train Data

Children Draw

|[Figure 38]|[Figure 39]|
|---|---|
|[Figure 40]|[Figure 41]|

|Sec. 4.4 Propagated Animation (I2V)<br><br>[Figure 42]<br><br>[Figure 43]| |
|---|---|
| | |

|[Figure 44]<br><br>Sec. 4.1 Multi-Shot Planning|
|---|

|M-LLM<br><br>[Figure 45]|
|---|

[Figure 46]

Fig. 2. The pipeline of the whole FairyGen.

the latent 𝑧 for 𝑧𝑡 where we train a denoising network to remove the added noise via:

then used as trainning data to finetune an MMDiT-based image-tovideo diffusion model network, where we utilize a two-stage LoRA training scheme to disentangle spatial identity and temporal motion features (Sec. 4.4). Overall, this pipeline preserves character identity, boosts motion fidelity, and heightens narrative tension. We discuss each part in detail in the following section.

L = ||𝜖 − 𝜖𝜃 (𝑧𝑡,𝑐,𝑡)||2, (1)

where 𝑐 is the condition signal, which is often the text features from the pre-trained text encoder [Radford et al. 2021; Raffel et al. 2020]. After training, the image can be generated from noise via the multi-step sampling process and VAE decoding.

LoRA for Customization.LoRA [Hu et al.2021] is first proposed for efficiently training the language model. Given the weight𝑊 ∈ R𝑚×𝑛 of any pre-trained model, LoRA only updates the parameters of the two low-rank matrices 𝐴 ∈ R𝑚×𝑙 and 𝐵 ∈ R𝑙×𝑛, where 𝑙 < 𝑚 and 𝑙 < 𝑛, to efficiently adapt the trained knowledge to the learned domain, which can be defined as: 𝑦 = 𝑊𝑥 + 𝐴𝐵𝑥, where 𝑥 and 𝑦 is the input vector and new weighted vector, respectively. In the diffusion model, LoRA is a common customization technique to tune the weight of 𝜖𝜃 using customized datasets.

4.1 Story and Shot Planning from a Single Character

To enable narrative-driven video synthesis from a single character draft, we introduce a story and shot planning module that decomposes the narrative into cinematic shot descriptions. Unlike prior work that focuses on low-level frame interpolation, our approach defines a clear storyboard to guide motion synthesis, shot composition, and narrative pacing. By grounding animation in a storyboard, we ensure coherent spatial composition and temporal progression aligned with storytelling conventions.

As illustrated in Fig. 3, our system begins with storyboard planning, which organizes the narrative through a hierarchical structure: a global narrative overview and a detailed shot-level storyboard. The global narrative outlines the character’s appearance, background context, and a high-level abstraction of the main event. Furthermore, the shot-level storyboard details the background, character action and camera configurations (e.g., shot type, perspective, focal region) for each shot. To operationalize the storyboard’s components, we introduce two modules: action planning and multi-shot planning. In action planning stage, action-related keywords are extracted using an LLM and then used to retrieve matching motions from a 3D animation platform. These motions are then adapted to the input character through rigging and retargeting. In the multi-shot planning stage, shot type and focal region descriptions guide the generation of bounding boxes, via an LLM, to crop synthesized backgrounds. Meanwhile, for multiview consistency, various character perspectives are rendered using a 3D proxy (see Sec. 4.3), and viewconditioned synthesis is applied to ensure coherent background generation (see Sec. 4.2).

4 METHOD

Given a single child hand-drawn character image 𝐼 with a blank background, our goal is to generate a fully stylized, long cartoon video 𝑉 that unfolds as a continuous story composed of multiple distinct shots. The generated video should faithfully maintain character consistency while enabling complex motion, coherent scenes, and cinematic storytelling. As illustrated in Fig. 2, we propose a multi-stage pipeline to achieve this goal. First, we employ an MLLM [OpenAI 2022] to infer a structured storyboard from the given character draft, forming the basis for temporal and spatial shot planning (Sec. 4.1). Next, conditioned on the input stylized image, we synthesize style-consistent backgrounds using a customized style propagation module, which propagates the character’s aesthetic traits to the surrounding environment, introducing rhythmic pacing and diverse spatiotemporal cues crucial for cinematic storytelling (Sec. 4.2). Then, we reconstruct a 3D proxy from the 2D character image, and derive physically plausible motion sequences by rigging and retargeting (Sec. 4.3). These motion sequences are

[Figure 47]

Tell me a Storyboard Planning story of the character?

|[Figure 48]|
|---|

[Overall Narrative Setting]

## MLLM

A old man in sunglasses and a spotted outfit is preparing for a baseball swing by the seaside, he is interrupted by a sudden surprise…

Sec 4.2

#### Multi-shot Planning

##### [Structured Storyboard]-shot1

[Figure 49]

|[Figure 50]| |
|---|---|
| | |

[Figure 51]

[Figure 52]

<Background Setting> A bright, seaside setting with the ocean and beach visible in the distance. <Character Action> The man stands still, intensely focused behind his sunglasses as he prepares for a baseball swing. <Shot Composition> close-up shot, frontal view centered on the character’s face, slow zoom to emphasize his focused gaze

[Figure 53]

LLM

[Figure 54]

crop multi-view

[Figure 55]

|[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]|
|---|

### Action Planning

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

<Character Action> LLM Search Retarget

Fig. 3. The pipeline of the storyboard generation. We first plan the whole story using the M-LLM and build a storyboard containing the scenes, events, character action, background, and camera shots. Then, we crop the stylized image using different camera shot and generate final shot images.

adapter is applied only to background tokens during inference to maintain targeted stylization. In detail, our propagated adapter is implemented as a low-rank adapter [Hu et al. 2021; Liu et al. 2024b]. Specifically, we find that DoRA [Liu et al. 2024b] demonstrates better performance in capturing stylistic detail.

[Figure 70]

Formally, let 𝑥 donate the full image tokens, and 𝑚 be the binary foreground mask. The propagated adapter 𝑃𝐴(·) updates the model as follows during training:

𝑦 = 𝑊𝑥 + 𝑃𝐴(𝑥 ·𝑚), (2)

where𝑊 is the original weights of SDXL, and 𝑦 is the customized feature output.

Fig. 4. Style Consistent Scene Generation.

During inference, we select the background region for image stylization, which can be formulated as:

4.2 Style-Consistent Scene Generation from Character

A character image with no background is insufficient for expressive storytelling. To visually support the narrative, we aim to generate scenes that are both contextually aligned with the storyline and stylistically consistent with the hand-drawn aesthetic of the foreground character. This consistency is particularly crucial in cartoon story videos, where visual uniformity enhances continuity, immersion, and emotional resonance. A key challenge is ensuring that the generated background faithfully reflects the artistic style, such as brushstroke texture, color palette, and line density, of the original hand-drawn character. Unlike traditional stylization approaches that transfer style from a reference background, our method propagates style from the character to the background, requiring the background to inherit the foreground’s visual attributes.

𝑦 = 𝑊𝑥 + 𝑃𝐴(𝑥 · (1 −𝑚)). (3)

In summary, our method learns the character’s visual style during training and selectively propagates it to the background during inference. This simple yet effective strategy ensures that the generated scenes remain stylistically unified with the character, supporting coherent and visually immersive animation.

4.3 Character Video Sequence Generation via 3D Proxy

Previous work learns to generate id-consistent video using customized image generation methods [Ruiz et al. 2023], and then generates video using image-to-video diffusion models for storytelling [Gong et al. 2023]. However, restricted by the generative prior of the image-to-video diffusion model, generating id-consistent and coherent motion is inherently challenging, since the foreground human motion is very complex.

To achieve the goal, we start from a pre-trained text-to-image diffusion model, i.e., SDXL [Podell et al. 2023], and adapt it using a propagation-based customization strategy. As shown in Fig. 4, in training, we customize only the foreground tokens to learn the artistic style, and for inference, we utilize the BrushNet [Ju et al. 2024] adapter based on SDXLto inpaint the background using the learned style, which is propagated through the adapter. Notably, the

Differently, we draw inspiration from traditional computer graphics pipelines, which enable fine-grained control over character motion through intermediate 3D representations. Specifically, following DrawingSpinUp [Zhou et al. 2024a], we adopt a 3D proxy-based

motion modeling approach that reconstructs the underlying 3D geometry of the character from a single 2D sketch. This proxy enables us to apply skeleton-based rigging and motion retargeting techniques, allowing for the transfer of complex motion sequences onto the character while preserving structural fidelity and visual consistency. By incorporating this explicit motion structure, we provide a robust foundation for generating animation semantically meaningful and visually coherent animation sequences. More details can be found in the original paper.

Stage2: Motion learning

Stage1: ID learning

|[Figure 71]|
|---|

|[Figure 72]|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

UnorderedFrames

Motion_B

|[Figure 76]|
|---|

|[Figure 77]|
|---|

OrderedFrames

ID_A

ID_A

[Figure 78]

[Figure 79]

ID_B

ID_B

|[Figure 80]|
|---|

|[Figure 81]|
|---|

[Figure 82]

[Figure 83]

Video DiT

Video DiT

…

…

|[Figure 84]|
|---|

|[Figure 85]|
|---|

[Figure 86]

Inference

[Figure 87]

Motion_B

ID_A

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

ID_B

- 4.4 Shots Animation via Motion Customization

[Figure 93]

Video DiT

To generate animatable video shots from background-composited frames, we leverage an image-to-video diffusion model. However, existing image-to-video diffusion models [Blattmann et al. 2023; Kong et al. 2024; Wan et al. 2025] struggle to generate complex motion for stylized or anthropomorphic characters, resulting in identity inconsistency and temporal flickering. Moreover, the ControlNet [Zhang et al. 2023]-like video control models (e.g., pose-guided and depthguided) exhibit limited generalization to non-human characters and often produce unnatural or scene-disconnected motion due to overly rigid constraints. Differently, we leverage character motion sequences derived from 3D reconstruction as train data to finetune the video diffusion model. Our main challenge is achieving shotlevel animation, where only specific body parts (e.g., head or legs) are animated. Direct training under such partial-motion often fails to maintain appearance consistency across frames. Additionally, existing video diffusion models require extensive training iterations to learn complex motion patterns, even for a single sequence. To address these challenges, we adopt a two-stage training strategy inspired by DynamicConcept [Abdal et al. 2025], , explicitly disentangling spatial appearance learning from temporal motion learning, as illustrated in Fig. 5. In the first stage, the model is trained on temporally shuffled frames to learn identity features without temporal correlations. In the second stage, the identity adapter is frozen, and a separate motion adapter is introduced. The model is then trained on temporally ordered frames using a novel timestep-shift strategy to effectively capture temporal dynamics. Instead of replicating motions exactly, our stategy learns to generate motion sequences that adapt to diverse backgrounds and narative contexts. During inference, the learned motion is composited with background-composited scenes to produce coherent, stylized animations. Let𝑊 donate the base weights of the video diffusion model, and let 𝐴id,𝐵id be the low-rank identity adapter matrices i.e., LoRA [Hu et al. 2021]. The identity-adapted features are computed as:

+ Prompt The robot jumps off the roof, through some trees and finally lands on the road.

Fig. 5. Two-stage motion train stratage. We first use unorded frames to learn character spatial features without temporal bias. Then, with the identity LoRA frozen, motion residuals are learned from sequential video frames.

Dropout is also applied to 𝐵motion to stabilize training and prevent overfitting, ensuring that 𝐴id remains a stable shared basis across both training stages.

While this two-stage training effectively separates motion from appearance, we further improve motion modeling through a novel timestep-shift sampling strategy, which we identify as critical for capturing realistic, coherent character dynamics in image-to-video motion customization. Standard diffusion training samples timesteps 𝑡 ∈ {1, . . .,𝑇} uniformly, emphasizing clean and noisy frames equally. However, we hypothesize that biasing training toward noisier timesteps—later in the diffusion process—forces the model to rely on global structure rather than low-level pixel cues. We implement this bias using Gaussian sampling followed by a sigmoid transformation:

1 1 + 𝑒−𝑧

, 𝑧 ∼ N(𝜇,𝜎2) (6)

𝑡 = 𝜎(𝑧) =

where 𝜇 controls the sampling bias and 𝜎 the variance. By setting 𝜇 closer to 𝑇, we construct a late-bias sampling distribution that increases the probability of sampling high-noise training steps. This late-biased sampling encourages the model to learn robust motion representations under challenging conditions. Empirically, we observe that this strategy produces smoother and more temporally consistent motion trajectories, especially for long sequences with complex character interactions.

5 EXPERIMENT 5.1 Implementation Details

𝑦 = 𝑊𝑥 + 𝐴id𝐵id𝑥, (4)

For the experiment and comparison with other methods, we utilize the AnimatedDrawings Dataset [Smith et al. 2023] as our childdrawn characters. We generate 24 images of different styles and 12 videos with different motions for style and motion comparison, respectively. All the experiments are based on one NVIDIA L20 GPU, it takes 120 minutes to learn the style and 180 minutes for motion customization. For stylization, instead of relying on artificial identifiers (e.g., “a [v] style”) as in DreamBooth, we find that using descriptive language prompts (e.g., “a childlike whimsical style”)

where 𝑥 is the input feature. To prevent the model from inadvertently learning temporal patterns during identity training, we apply dropout to 𝐵id: 𝐵id = 𝐵id ⊙ 𝑀𝑝, where 𝑀𝑝 is a binary mask with dropout probability𝑝. In the second stage, we fix the identity adapter and introduce a motion-specific adapter 𝐵motion, applied to sequential frames. Motion is modeled as a residual deformation on top of the identity representation:

𝑦 = 𝑊𝑥 + 𝐴id𝐵id𝑥 + 𝐴id𝐵motion𝑥. (5)

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

|[Figure 99]|
|---|

Sequence

Depth-Guided Motion

(Pose-Guided) Wan2.1-Fun

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Animate-X

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

Ours

Fig. 6. Compare with Motion Customization. We compare the proposed motion customization method with the depth-guided image-to-video method using Wan2.1 [Wan et al. 2025], pose-guided image-to-video character animation method, i.e., Animate-X [Tan et al. 2024], the proposed method shows very similar results to the original motion sequence with this complex motion.

yields better alignment with fine-grained stylistic attributes such as texture, brushstroke direction, and line quality. To automate this, we employ GPT-4 [OpenAI 2022] to generate textual descriptions of the reference image’s style, which are then used as part of the training prompt similar to StyleDrop [Sohn et al. 2023]. After training, we can use text to generate the background style and motion.

Numerical Comparsion User Study Methods Style Text Style Visual Align Align Quality Quality

B-LoRA 0.5060 0.2829 0.0267 0.3429 Instant Style 0.5468 0.2368 0.3403 0.0517 DreamBooth 0.6371 0.2819 0.0965 0.2803 Ours 0.6580 0.2702 0.5365 0.3251

For style evaluation, we calculate the CLIP [Radford et al. 2021] distance of the generated image and the source image as a style alignment score, where we also calculate the CLIP distance of the generated image and the corresponding CLIP text features. As for motion evaluation, we choose two metrics from the VBench [Huang et al. 2024; Liu et al. 2024a], including the motion smoothness and the subject consistency.

Table 1. Style Comparsion with other methods.

[Figure 118]

[Figure 119]

[Figure 120]

InstantStyleDreamBoothB-LoRAOurs

- 5.2 Comparison with Other Methods

Since there are no previous baselines for this task, we first compare our method with multi-event video generation from text description, i.e., MEVG [Oh et al. 2024] and Vlogger [Zhuang et al. 2024], as well as the state-of-the-art few-shot subject-driven video generation method, i.e., DreamVideo [Wei et al. 2024]. As shown in Fig. 10 and Fig. 11, the proposed method shows better results in maintaining character appearance consistency, motion smoothness, and style preservation. In contrast, multi-event generation models struggle to produce coherent narratives and consistent visual styles, while subject-driven methods fail to robustly preserve identity and motion, and often produce unrealistic or inconsistent backgrounds. By leveraging a 3D proxy as motion prior and a customized style propagation adapter, our method achieves more coherent, stylistically faithful, and temporally smooth video generation results.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Reference Style

[Figure 128]

[Figure 129]

[Figure 130]

Fig. 7. Compare with Stylization Methods. We compare our method with different stylization methods on stylization customization.

Numerical Comparsion User Study Methods Motion Subject Motion Visual Smooth. Consist. Realness Quality

[Figure 131]

|[Figure 132]|
|---|

Ablation: two stage lora

| |
|---|

[Figure 133]

Animate-X 0.974 0.908 0.106 0.023 Wan2.1-Fun 0.977 0.842 0.114 0.106 Ours 0.987 0.955 0.780 0.871

Reference Frame Reference ID

[Figure 134]

[Figure 135]

Table 2. Motion Comparsion. with other methods.

Ablation: style propagation

|[Figure 136]|
|---|

[Figure 137]

[Figure 138]

[Figure 139]

LoRA Directly

[Figure 140]

|[Figure 141]|
|---|

[Figure 142]

|[Figure 143]|
|---|

[Figure 144]

[Figure 145]

[Figure 146]

w/ Two Stage Motion Adapter

Fig. 9. Ablation on two-stage Motion Adapter. We ablated the twostage adapters in our proposed motion customization in image-to-video generation. Here, the first stage of training improves the identity similarity.

[Figure 147]

[Figure 148]

[Figure 149]

24 stylized image sets, each set contains 4 different methods, and needs to be evaluated from two aspects. As for motion, we utilize 12 video sets using 3 different methods, and the users need to be evaluated from two aspects. Finally, we obtain 3360 opinions. As shown in tables, the users consistently prefer our method in terms of style alignment, motion realism, and visual coherence. As shown in Tab. 1, our method achieves the highest score in style similarity, surpassing B-LoRA, InstantStyle, and DreamBooth. While our visual impression score (0.3251) is slightly lower than B-LoRA (0.3429), we attribute this to B-LoRA’s photorealistic output, which may be perceived as more visually appealing than child-style cartoon imagery. For video results (Table 2), our method demonstrates clear advantages over existing approaches, highlighting its ability to produce temporally consistent, stylistically faithful animations.

[Figure 150]

[Figure 151]

[Figure 152]

LoRA DoRA PA (Ours)

Fig. 8. Ablation Study on Style Customization. Compared with the baseline LoRA [Hu et al. 2021] and DoRA [Liu et al. 2024b], the proposed method can successfully propagate the foreground style to the background with different prompts. Best viewed with zoom in.

5.3 Ablation Studies

- 5.3.1 Propagation Style Adapter. We first evaluate the effectiveness of the proposed style propagation adapter. As shown in Fig. 8, given the image of the foreground character, the DoRA customization method shows a better stylization propagation result than the original LoRA method. Besides, the proposed propagation adapter can successfully learn the style better with more detailed streak preservation on the background.
- 5.3.2 Motion Adapter. Our proposed Motion Adapter utilizes a twostage customization strategy for motion customization. Here, we give the effectiveness of each stage. As shown in Fig. 9, directly training the LoRA on the image-to-video diffusion model causes unnatural identity generation, while the two-stage motion adapter achieves better performance.
- 5.3.3 Time-Shift for Motion Learning. One of our key techniques is to utilize the timestep shift in the step sampling to obtain better

In addition to visual comparison, we also perform detailed quantitative evaluations. For style comparison, we assess both the style alignment and the relevance to the accompanying text description. As shown in Tab 1, our method outperforms previous approaches in both subjective and objective stylization metrics. As for the quality of motion, we compare against two video character animation methods: the pose-guided method Animate-X, which uses human motion videos as reference for accurate keypoint detection, and a depth-guided method that utilizes depth sequences extracted from 3D-reconstructed character motion sequences. We compare the motion quality with the previous baselines in Tab. 2, where the proposed method shows significantly better results.

We further conduct subject experiments on both stylized images and generated videos to evaluate the effectiveness of the proposed method. As shown in Tab. 2 and Tab. 1, we invite 24 users to evaluate

Story Prompts: [r]_character_[j]_motion stands on the edge of a tall city rooftop, ready to jump down to the street below. [r]_character first extends its arms, then jumps upward. As [r]_character drops, the building walls and city skyline move past in the background. [r]_character lands freely on the pavement, with its arms touching the ground, as dust and

|[Figure 153]|
|---|

[r]_Character debris scatter slightly upon landing.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

OursVloggerMEVG

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

- Fig. 10. Comparsion on Multi-Event Video Generation. Our method splits the foreground and the background modelling, which is easy for longer and multi-event video generation. Here, we use the same story prompt to generate the video, where the proposed method shows much consistent results as well as the text description.

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

DreamVideotraindataOurs

id_motion_baseline

- Fig. 11. Comparison on the appearance and motion customization method. We compare our method with state-of-the-art appearance and motion customization method, i.e., DreamVideo [Wei et al. 2024], the proposed method shows significantly better results considering the stylization, motions, and overall quality.

motion. We show the results of the different sampling steps in Fig. 12. As shown in the figure, directly using uniform sampling might not learn a similar motion to the original sample, while 𝜇 = 6 provides better results than uniform sampling and other hyperparameters.

5.4 Limitation and Future Work

We only show the results of the single character. However, it is easy for our method to be extended into multiple subjects with multiple 3D proxies. The foreground character (or animal) may not always be correctly reconstructed by the 3D proxy, we believe the

Ablation: timestep shift FairyGen: Storied Cartoon Video from a Single Child-Drawn Character • 9

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

=-6=6=0µµTargetuniformµ

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

- Fig. 12. Ablation Study on timestep shift. The proposed timestep shift strategy in the motion customization can learn to represent the motion better.

more advanced rigging method [Zhang et al. 2025a] will help us to generate the motion of the foreground better. The generative prior of the video diffusion model may not always accurately generate a stable and animable background. As shown in Fig. 13, the proposed method generates a still background image. We will try different image-to-video diffusion models [Kong et al. 2024] and include more camera motion [Bai et al. 2025] to improve the motion realism.

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Frame 1 Frame 8 Frame 14

Frame 16 Frame 21 Frame 28

- Fig. 13. Limitation. Due to the uncontrollable generative prior of the video diffusion model, the proposed method might only generate a still background with the animated foreground motion (e.g., running).

large language model (M-LLM), followed by a style propagation adapter for generating stylized backgrounds consistent with the narrative context. To customize motion, we introduce motion-aware adapters and a timestep-sampling strategy for flexible control over character dynamics. Compared with several baselines, our method achieves high-quality results in stylized background generation and motion customization, demonstrating superior adaptability and visual coherence.

REFERENCES

Rameen Abdal, Or Patashnik, Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, Sergey Tulyakov, Daniel Cohen-Or, and Kfir Aberman. 2025. Dynamic Concepts Personalization from Single Videos. arXiv:2502.14844 [cs.GR]

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. 2025. ReCamMaster: CameraControlled Generative Rendering from A Single Video. arXiv:2503.11647 [cs.CV] https://arxiv.org/abs/2503.11647

Xiuli Bi, Jian Lu, Bo Liu, Xiaodong Cun, Yong Zhang, WeiSheng Li, and Bin Xiao. 2024. CustomTTT: Motion and Appearance Customized Video Generation via Test-Time Training. arXiv preprint arXiv:2412.15646 (2024).

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Tim Brooks and Alexei A Efros. 2022. Hallucinating pose-compatible scenes. In European Conference on Computer Vision. Springer, 510–528.

Hong Chen, Rujun Han, Te-Lin Wu, Hideki Nakayama, and Nanyun Peng. 2022. Character-centric story visualization via visual planning and token alignment. arXiv preprint arXiv:2210.08465 (2022).

6 CONCLUSION

We present FairyGen, a novel framework for story visualization that decomposes the narrative into foreground character motion and environmental dynamics, enabling layered modeling with unified style and movement. We first plan the storyline using a multi-modal

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. 2023. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023).

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. 2024. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7310–7320.

DeepSeek. 2025. DeepSeek-R1: A Reasoning Model. https://deepseek.com/. Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. 2025. Implicit style-

content separation using b-lora. In European Conference on Computer Vision. Springer, 181–198.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Yingqing He, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, and Yujiu Yang. 2023. TaleCrafter: Interactive Story Visualization with Multiple Characters. arXiv:2305.18247 [cs.CV]

Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. 2024. Dreamstory: Open-domain story visualization by llm-guided multi-subject consistent diffusion. arXiv preprint arXiv:2407.12899 (2024).

Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. 2023. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940 (2023).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic

models. Advances in neural information processing systems 33 (2020), 6840–6851. Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

Panwen Hu, Jin Jiang, Jianqi Chen, Mingfei Han, Shengcai Liao, Xiaojun Chang, and Xiaodan Liang. 2024. StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration. arXiv preprint arXiv:2411.04925 (2024).

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21807–21818. Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. 2024. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6689–6700.

Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. 2024. BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion. arXiv:2403.06976 [cs.CV]

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A Efros, and Krishna Kumar Singh. 2023. Putting people in their place: Affordanceaware human insertion into scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17089–17099.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu.

2023. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1931–1941.

Bowen Li. 2022. Word-level fine-grained story visualization. In European conference on computer vision. Springer, 347–362.

Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. 2019. Storygan: A sequential conditional gan for story visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 6329–6338.

Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. 2024c. Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6190–6200.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024b. Dora: Weight-decomposed lowrank adaptation. In Forty-first International Conference on Machine Learning.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. 2024a. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22139–22149.

Adyasha Maharana, Darryl Hannan, and Mohit Bansal. 2022. Storydall-e: Adapting pretrained text-to-image transformers for story continuation. In European conference on computer vision. Springer, 70–87.

Gyeongrok Oh, Jaehwan Jeong, Sieun Kim, Wonmin Byeon, Jinkyu Kim, Sungwoong Kim, and Sangpil Kim. 2024. Mevg: Multi-event video generation with text-to-video models. In European Conference on Computer Vision. Springer, 401–418.

OpenAI. 2022. ChatGPT. https://openai.com/blog/chatgpt.

Boxiao Pan, Zhan Xu, Chun-Hao Huang, Krishna Kumar Singh, Yang Zhou, Leonidas J Guibas, and Jimei Yang. 2024b. Actanywhere: Subject-aware video background generation. Advances in Neural Information Processing Systems 37 (2024), 29754– 29776.

Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. 2024a. Synthesizing coherent story with auto-regressive latent diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2920–2930.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023).

Qwen. 2025. Qwen Model. https://help.aliyun.com/product/170553.html. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.

Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. 2023. Make-a-story: Visual memory conditioned consistent story generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2493–2502.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022a. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022b. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22500–22510.

Harrison Jesse Smith, Qingyuan Zheng, Yifei Li, Somya Jain, and Jessica K. Hodgins. 2023. A Method for Animating Children’s Drawings of the Human Figure. ACM Trans. Graph. 42, 3, Article 32 (jun 2023), 15 pages. https://doi.org/10.1145/3592788

Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. 2023. Styledrop: Text-toimage generation in any style. arXiv preprint arXiv:2306.00983 (2023).

Sitong Su, Litao Guo, Lianli Gao, Heng Tao Shen, and Jingkuan Song. 2023. Make-astoryboard: A general framework for storyboard with disentangled and merged control. arXiv preprint arXiv:2312.07549 (2023).

Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobin Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. 2024. Animate-X: Universal Character Image Animation with Enhanced Motion Representation. arXiv preprint arXiv:2410.10306 (2024).

Ming Tao, Bing-Kun Bao, Hao Tang, Yaowei Wang, and Changsheng Xu. 2024. Storyimager: A unified and efficient framework for coherent story visualization and completion. In European Conference on Computer Vision. Springer, 479–495.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314 (2025).

Wen Wang, Canyu Zhao, Hao Chen, Zhekai Chen, Kecheng Zheng, and Chunhua Shen.

2024. AutoStory: Generating Diverse Storytelling Images with Minimal Human Efforts. International Journal of Computer Vision (2024), 1–22.

Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. 2024. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6537–6549.

Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. 2024a. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683 (2024).

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024b. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv preprint arXiv:2408.06072 (2024).

Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. 2023. Inserting Anybody in Diffusion Models via Celeb

Basis. arXiv preprint arXiv:2306.00926 (2023).

Jinlu Zhang, Jiji Tang, Rongsheng Zhang, Tangjie Lv, and Xiaoshuai Sun. 2025b. Storyweaver: A unified world model for knowledge-enhanced story character customization. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 9951–9959.

Jia-Peng Zhang, Cheng-Feng Pu, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. 2025a. One Model to Rig Them All: Diverse Skeleton Rigging with UniRig. arXiv preprint arXiv:2504.12451 (2025).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 3836–3847.

Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. 2024. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655 (2024).

FairyGen: Storied Cartoon Video from a Single Child-Drawn Character • 11

Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. 2025. Motiondirector: Motion customization of text-to-video diffusion models. In European Conference on Computer Vision. Springer, 273–290.

Jie Zhou, Chufeng Xiao, Miu-Ling Lam, and Hongbo Fu. 2024a. DrawingSpinUp: 3D Animation from Single Character Drawings. SIGGRAPH Asia 2024 Conference Papers

(2024). https://api.semanticscholar.org/CorpusID:272654016

Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. 2024b. StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation. NeurIPS 2024 (2024).

Zhongyang Zhu and Jie Tang. 2025. Cogcartoon: towards practical story visualization. International Journal of Computer Vision 133, 4 (2025), 1808–1833.

Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. 2024. Vlogger: Make your dream a vlog. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8806–8817.

