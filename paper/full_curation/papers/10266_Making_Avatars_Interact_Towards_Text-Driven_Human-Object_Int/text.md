## Making Avatars Interact: Towards Text-Driven Human-Object Interaction for Controllable Talking Avatars

Youliang Zhang1,2 † Zhengguang Zhou2 † Zhentao Yu2 Ziyao Huang 2

Teng Hu2 Sen Liang2 Guozhen Zhang2 Ziqiao Peng2 Shunkai Li2 Yi Chen2 Zixiang Zhou2 Yuan Zhou2 § Qinglin Lu2 ‡ Xiu Li1‡

1Tsinghua University. 2Tencent HY.

# arXiv:2602.01538v1[cs.CV]2Feb2026

Raise both hand to touch the eyeglasses, then take off the eyeglasses to chest. Bend down and place the bag in your hand on the ground.

First make a fist, then make a peace sign.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Ref

Ref

Ref

Human Action

Lower the apple to your chest and display it.

[Figure 10]

[Figure 11]

[Figure 12]

Ref

Pick up the apple on the table.

Pick up the cup and drink water. Raise the basketball over head.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Ref

Ref

Ref

Speak Words: Do you want to take a bite?

Human Talk

Grounded Human-Object Interaction

Figure 1. Input text, audio, and reference image, our method can generate human videos that can talk, act, and interact with objects. Grounded means that all action and object interaction occurs within the environment provided by the reference image.

#### Abstract

action motions. Additionally, an Audio-Interaction Aware Generation Module (AIM) is proposed to synthesize vivid talking avatars performing object interactions. With a specially designed motion-to-video aligner, PIM and AIM share a similar network structure and enable parallel cogeneration of motions and plausible videos, effectively mitigating the control-quality dilemma. Finally, we establish a benchmark, GroundedInter, for evaluating GHOI video generation. Extensive experiments and comparisons demonstrate the effectiveness of our method in generating grounded human-object interactions for talking avatars. Project page: https://interactavatar.github.io/.

Generating talking avatars is a fundamental task in video generation. Although existing methods can generate full-body talking avatars with simple human motion, extending this task to grounded human-object interaction (GHOI) remains an open challenge, requiring the avatar to perform text-aligned interactions with surrounding objects. This challenge stems from the need for environmental perception and the control-quality dilemma in GHOI generation. To address this, we propose a novel dualstream framework, InteractAvatar, which decouples perception and planning from video synthesis for grounded human-object interaction. Leveraging detection to enhance environmental perception, we introduce a Perception and Interaction Module (PIM) to generate text-aligned inter-

#### 1. Introduction

Human-centric video generation has achieved remarkable progress, with significant strides made in tasks like photorealistic talking head synthesis, focusing primarily on accurate lip-synchronization and facial expression genera-

† Equal Contribution ‡ Corresponding author § Project Leader

* Work done during an internship at Tencent HY.

tion [3, 22, 27, 30, 35]. However, to construct vivid and practical digital humans, we must move beyond facial modeling and endow them with the ability to perform complex actions and interact meaningfully with their surroundings.

We introduce Grounded Human-Object Interaction for talking avatars, aiming to enhance current digital humans with environmental perception and text-driven human-object interaction generation ability. Different from the current talking avatar and HOI generation task, GHOI features: 1) Environment-aware, perceives its environment and performs reasonable actions within it. 2) Consistent with the initial frame, rather than generating a new scene. 3) Text-following, do not demand explicit pose or object trajectories for control, and 4) Free of additional object conditions, operating objects in the scene based on text.

Despite recent advancements, existing paradigms still fall short of enabling such nuanced, scene-aware interaction due to fundamental architectural limitations: (1) Audiodriven methods, through several pioneering works on simple full-body animation [3, 7, 19, 31], typically learn a direct mapping from acoustic features to pixel space, lacking explicit modeling of objects and environments, thus making complex human-object interactions difficult to control; (2) Pose-driven approaches [5, 26, 28, 33] offer explicit control but offload the planning burden to the user. These methods require pre-defined skeletal sequences as input, which are not only difficult and expensive to obtain but also often misaligned with the specific context of a reference image; (3) Subject-consistent methods [2, 13, 20] excel in preserving subject identity and achieving coherent integration, but lack mechanisms for grounded interaction. They are designed to synthesize new videos from text rather than to carry out contextual commands in an interactive environment. Therefore, enabling a speaking avatar to perform stable, text-driven object interactions remains a significant and unresolved challenge. The core difficulties are twofold.

(1) Scene-Action Grounding: Instead of merely generating a video featuring human-object interaction, GHOI demands the interaction to occur within a specific environment involving designated objects. This requires the model to interpret the spatial layout and semantic content, and to ground textual commands within this visual context. (2) Control-Quality Dilemma: Owing to the complexity of GHOI, existing methods often face a trade-off between controllability and visual quality. They either generate highfidelity videos that struggle to follow instructions or achieve reasonable responses at the expense of video fidelity.

To address these challenges, we propose InteractAvatar, a novel dual-stream Diffusion Transformer (DiT) framework designed to generate grounded human-object interactions in talking avatars. This framework explicitly decouples perception and planning from video synthesis. For clarity, we define motion as the rendered skeletal poses and ob-

ject boxes. We introduce a Perception and Interaction Module (PIM) to tackle the grounding challenge, which generates text-aligned motion sequences based on the perception of the reference image. The PIM is jointly trained on detection and motion generation tasks, ensuring the production of scene-aware and text-aligned motions. Conditioned on both audio and the motion latent features, an Audio-Interactionaware Generation Module (AIM) is proposed to synthesize the final video. PIM and AIM share a similar network architecture and achieve the parallel co-generation of motions and videos. With a specially designed motion-to-video (M2V) aligner, our symmetrical yet decoupled framework effectively alleviates the control-quality dilemma in complex GHOI video generation, achieving vivid and plausible grounded human-object interactions for talking avatars.

To rigorously evaluate our approach, particularly its ability to model grounded human-object interactions, we constructed the GroundedInter benchmark. GroundedInter comprises 600 distinct test cases, each containing a reference image, a structured textual description of the interaction, and the corresponding speech audio. Extensive experiments conducted on this benchmark demonstrate that our proposed method significantly outperforms existing approaches in generating plausible, controllable, and highquality grounded human-object interaction videos.

Our main contributions are summarized as follows:

- • We propose InteractAvatar, a novel dual-stream DiT framework that explicitly decouples perception planning from video synthesis, achieving text-driven grounded human-object interaction generation for talking avatars.
- • We introduce a Perception and Interaction Module (PIM), trained on both detection and motion generation tasks, to generate scene-aware and text-aligned motions. In addition, we propose an Audio-Interaction Aware Generation Module (AIM) with an M2V aligner to produce videos with controllability and high visual quality.
- • Beyond the core GHOI generation, our framework provides unified and flexible multimodal control, accepting any combination of text, audio, and motion as input.
- • We establish GroundedInter, a benchmark for finegrained evaluation of GHOI. Experiments demonstrate that our method significantly outperforms existing approaches in producing controllable high-quality talking avatars with grounded human-object interactions.

#### 2. Related Work

##### 2.1. Audio-driven human animation.

Given audio and human reference images as input, audiodriven human animation aims to generate videos with wellaligned lip movements. Initial research predominantly focused on head-and-shoulder animations, SadTalker [41] employing 3D motion coefficients to improve realism,

Hallo [35] using a hierarchical diffusion approach for lipsync accuracy, EchoMimic [4] combining audio with facial landmarks to enhance stability, and Loopy [15] leveraging temporal modules to generate natural motion without manual templates. Subsequently, research has progressively extended this capability toward half-body and full-body digital human synthesis with body action modeling. Hallo3 [7] first applied a pre-trained DiT model for portrait animation, Huanyuanvideo-Avatar [3] achieves dynamic, emotion-controllable, and multi-character dialogue video generation. FantasyTalking [31] proposes a duallevel optimization for global motion and lip synchronization, and OmniHuman-1 [19] scales up training data with hybrid motion conditions to generate more realistic fullbody videos. However, these methods lack the capacity to generate more complex grounded human-object interactions. Our work extends audio-driven animation to incorporate human-object interaction, thereby enhancing the realism and practical applicability of the generated avatars.

##### 2.2. Human-Object Interaction Video Generation

Recently, Human-Object Interaction (HOI) video generation has garnered increasing attention. Some works focus on editing existing footage. MIMO [21] and AnimateAnyone2 [12] substitute humans within dynamic scenes, whereas HOI-Swap [37] and ReHoLD [9] replace handheld objects. However, a primary limitation of these methods is their inability to animate a static reference image from scratch. Another works seek fine-grained control via complex conditional inputs. AnchorCrafter [36] utilizes pose and depth maps for interaction-aware human motion, ManiVideo [23] employs precise 3D models of both the human and object for accurate control, and HunyuanVideoHOMA [14] explores weak HOI conditions with a novel appearance injection method. Nevertheless, acquiring and aligning these complex conditions with an arbitrary ref image is often impractical, hindering their use in digital human applications. In contrast, our method overcomes these challenges by generating realistic human-object interaction videos conditioned solely on text, thereby eliminating the need for source videos or other complex control signals.

Recently, subject-consistent video generation also exhibited the capacity to synthesize HOI video. Phantom [20] leverages the inherent consistency of DiT-based models by concatenating reference image latents with video latents. Concurrent works such as InterActHuman [34] and HunyuanCustom [13] integrate subject-consistent video generation with audio-driven animation, while HuMo [2] proposes a progressive multimodal training paradigm to enable flexible control. However, a critical distinction between GHOI and subject-consistent video generation is that these methods do not preserve the world defined by the environment of the ref image. Instead, they typically generate an entirely

new scene based on text and subject appearance, rather than situating the interaction within the provided environment.

#### 3. Method

As shown in Fig. 2, we introduce a dual-stream Diffusion Transformer (DiT) for generating text-controlled grounded human-object interactions. Our architecture decomposes this complex task into two sub-problems, perceptual planning and video rendering, handled by the Perception and Interaction Module (PIM) and the Audio-Interaction aware Generation Module (AIM), respectively. We define motion as the rendered human skeletal poses and object boxes. A key aspect of our design is the parallel co-generation of structural motion sequences and corresponding video frames. The PIM acts as the planning brain, focusing on high-level structure by generating scene-aware and textaligned motions. The AIM serves as the rendering engine, synthesizing high-fidelity videos with precise audio lipsync. Rather than a sequential pipeline, AIM is informed throughout the generation process via the M2V aligner, which employs a layer-wise residual injection mechanism to ensure that the generated video consistently adheres to the evolving structural motion from PIM in lockstep.

Both modules are trained under a unified Flow Matching paradigm. A pre-trained Variational Autoencoder (VAE) [29] encodes all visual inputs (videos, motions, ref images) into latent space, while a T5 [25] encoder embeds the textual commands. The model uθ is optimized to predict a vector field vt with the following objective function:

0,c ∥vt(zt) − uθ(zt,t,c)∥22 , (1)

LFM = Et,z

where z0 is the clean latent from the VAE, zt is its noised version at timestep t, vt(zt) is the ground-truth vector field, and c represents all conditioning information.

##### 3.1. Perception and Interaction Module

The PIM is responsible for parsing the environmental context from a reference image Iref and generating a spatiotemporally reasonable interaction motion with the guidance of the text prompt T. We define the interaction motion m as a composite visual representation, comprising a sequence of human skeletal keypoints, P = {pt}Nt=0−1, and object bounding box trajectories, O = {ot}Nt=0−1, where N is the total number of frames. This motion serves as an intermediate representation that is injected at the feature level into the video generation module AIM, thereby enabling disentangled control over the human-object interaction dynamics.

Interaction Motion Generation. To precisely control the generation process, we explicitly partition the motion generation into two distinct task types: pure action generation, involving only the human pose sequence P, and HOI generation, including both the pose sequence P and object

Audio-Interaction Aware Generation Module(AIM) Face Mask Overall Training Strategy

Feature Extraction

| |
|---|

| |
|---|

𝑀

### ...

AudioInjection ...

PIM Pre-train AIM Pre-train

Detection

𝑓

Audio Injection

[Figure 19]

- Stage 1
- Stage 2

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

...

...

Ref Image

Ref Video Motion

Motion

𝑧

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

VideoHead

VideoBlock

VideoBlock

VideoBlock

VideoBlock

...

...

... JointFine-Tuning

[Figure 31]

[Figure 32]

...

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

... ...

𝐼 𝑓

VAE

###### PIM Pre-train Strategy

Flow Matching Loss

| | | |M2V Aligner| |
|---|---|---|---|---|
| |M2V Aligner| | | |

𝐼 𝑧 𝑚 𝑚

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

...

[Figure 44]

- Stage 1

- Stage 2

𝑚 𝑚

[Figure 45]

MotionHead

MotionBlock

MotionBlock

MotionBlock

MotionBlock

[Figure 46]

[Figure 47]

[Figure 48]

### ...

Audio Text Wav2Vec

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

...

[Figure 53]

[Figure 54]

[Figure 55]

T5

𝐼

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

... ...

𝑓

𝑓 𝑓

Perception and Interaction Module(PIM)

Detection Generation Continuation

Figure 2. Overview of our dual-stream InteractAvatar framework and multimodal conditioned training strategy.

trajectories O. We inject the task-specific information by concatenating a task embedding vector ftask corresponding to the task type ctask ∈ {ACTION,HOI} with the text prompt embedding ftext. This creates a unified conditioning vector for the cross-attention layers of the model:

is masked out from the input. The model is tasked with generating the entire sequence, including the first frame mref, conditioned solely on Iref and T. To further intensify the focus on scene perception, we probabilistically set the target video length to 1. This reduces the generation task to a detection-like objective, the model performs a single-frame generation where the ground truth is the motion mref.

###### ftext = Concat(ftext,ftask). (2)

This detection-like task is also optimized using the same flow matching objective, with the loss computation simply confined to the first frame. This unified framework seamlessly trains the PIM to leverage a single set of network parameters for two seemingly disparate yet fundamentally related tasks: parsing scene structure and generating temporal motions from perception. This strategy obviates the need for heterogeneous loss functions and their associated hyperparameters, which may introduce training instabilities.

The reference image Iref is temporally prepended to the first motion frame mref, the model takes mref as first frame to generates overall motion sequence {mt}Nt=1−1 conditioned on Iref. To ensure the reference image provides effective global guidance without conflicting with the positional encodings of the motion sequence, we employ a custom mapping for its Rotary Position Embeddings (RoPE). For a given patch zI from the reference image with an original 3D position index (l,i,j), its position is remapped as:

##### 3.2. Audio-Interaction Aware Video Generation

RoPEz

(l,i,j) = RoPE(−1,i + w,j + h). (3)

I

Audio Injection. Precise lip synchronization with the input audio is paramount for generating realistic avatars. We employ a pre-trained Wav2Vec model [1] as our audio feature extractor. To capture the co-articulation and temporal dynamics of speech, rather than relying on isolated phonemes, we extract features from a contextual window around each timestep. For the i-th video frame, its corresponding audio feature faudio,i is computed by encoding and aggregation:

This strategy positions the reference image as a special environment frame at a virtual timestep of -1, allowing it to condition the entire sequence generation without disrupting the temporal integrity of the video frames themselves. Given that motions encode structure rather than fine-grained texture, we therefore constrain the shorter side of the motions to 256 pixels, preserving essential structural information while significantly reducing computational overhead.

faudio,i = Aggregate Wav2Vec(a[c

i−w:ci+w]) , (4)

Environment Perception Training. Naive conditioning on the reference image is insufficient for the model to understand the complex scene layouts and object geometries. To address this, we develop an environment perception training curriculum that encourages the model to engage more deeply with the reference image. Our training curriculum dynamically alternates between two task modes: (1) Conditional Continuation. In this mode, the model receives the reference image Iref, the first frame motion mref, and the text prompt T as conditions. Its objective is to generate the subsequent motion sequence {mt}Nt=1−1. (2) Perception as Generation. In this mode, the first frame motion mref

where a is the raw audio, ci is the center sample aligned with frame i, w denotes the half-width of the contextual window, and Aggregate(·) is a feature aggregation operation, such as concatenation, for temporal video audio align.

To achieve frame-level audio-visual alignment, the extracted audio feature sequence {faudio,i} is injected into the AIM via cross-attention. To accelerate convergence and explicitly guide this conditioning, especially in early training, we introduce a spatial face mask, Mface. This mask spatially weights the output of the audio injection module, concen-

trating its influence on the facial region for stable training:

h′v,l = hv,l + CrossAttention(hv,l,faudio) ⊙ Mface, (5)

where hv,l represents the visual features at layer l of the AIM and ⊙ denotes element-wise multiplication. This strategy effectively channels the impact of the audio signal to the target facial area, leading to a more stable learning process.

Motion to Video Aligner. Leveraging the isomorphic architecture of the PIM and AIM, we introduce the M2V aligner with a layer-wise residual injection mechanism for motion control. Specifically, we inject the feature residual, the difference between the output of each DiT block in the PIM, into the corresponding layer of the AIM. For given layer l > 0 in the PIM, its residual ∆hm,l is computed as:

∆hm,l = houtm,l − houtm,l−1. (6) For the base layer (l = 0), we use houtm,0 directly.

To reconcile the discrepancy of the low-resolution motion and the high-resolution video, we first upsample the residual feature ∆hm,l to match the spatial dimensions of video features hv,l using simple bilinear interpolation. This is followed by a projection through a zero-initialized linear layer. The final output of an AIM block is computed as:

houtv,l = Blockv(hinv,l + Linearzero(Interp(∆hm,l))). (7) This residual injection strategy significantly reduces

loss fluctuations during training. Furthermore, the zeroinitialization of the linear layer ensures stability in the early stages of training by allowing the model to gradually learn the guidance of the motion. This effectively prevents visual artifacts, such as the ghosting of the motion’s skeletal lines, from leaking into the final generated video.

Unified Training for Generative and Driven Control. A truly versatile digital human system should not only generate motions from textual prompts but also faithfully follow explicit driving signals, such as skeletal sequences, interaction motions, or object trajectories. To achieve this, we co-train the model on two complementary tasks: Joint Generation and External Driving. For External Driving, the model is provided with a clean driving signal, such

- as a skeleton sequence, which is fed directly to the PIM. Here, the PIM actually acts as a feature encoder, which is elegantly achieved by setting the diffusion timestep t to

0. Driven signals are also extracted as layer-wise feature residuals {∆hm,l}, which are then injected into the AIM to guide the generation process. To balance these dual capabilities, we employ a 4:1 training data ratio between the Joint Generation and External Driving tasks. This co-training strategy endows our model to function both as a creative, text-guided motion video generator and as a precise, controllable animator, all within a single, unified framework.

##### 3.3. Multimodal Conditioned Training Strategy

We designed a three-stage training curriculum to progressively integrate multimodal conditions, ensuring stable learning and preventing conflicts between modalities.

PIM Pre-training for Scene-Aware Planning. The initial stage is dedicated to independently training the PIM. The goal is to establish its core ability to parse static scenes from a reference image and generate coherent motion plans based on text. The training curriculum blends multiple tasks: (1) a static detection task to build scene perception, (2) a continuation task, motion continuation from a given first motion mref, and (3) a crucial hybrid generation task that requires generating the motion sequence from only the image Iref and text. We found that this mixed-task approach, which forces a synergy between perception and generation, is superior to the pure generation pipeline. The training strategy of those tasks is shown in Fig. 2.

AIM Pre-training for Audio-Visual Synchronization. The second stage imbues AIM with the ability for audiodriven lip synchronization while preserving its underlying image-to-video (I2V) quality. Crucially, we found that the training order is paramount. The model should be trained with audio before being exposed to interaction motion. We attribute this to the distinct nature of condition signals: audio is a soft, localized, and heterogeneous modality injected through cross-attention, whereas motion serves as a hard, global, and homogeneous structural constraint introduced via residual injection. Introducing the dominant structural motion condition only after the weaker local audio conditioning has stabilized effectively prevents the audio signal from being suppressed during training optimization.

End-to-End Joint Fine-tuning. In the final stage, the pre-trained PIM and AIM are jointly fine-tuned, where the PIM’s outputs are injected into the AIM as layer-wise residuals. The training strategy adopts a probabilistic mix of tasks and conditions to unify the model’s capability to accommodate any combination of text, audio, and motion inputs. Specifically, 30% of the samples are conditioned on audio to preserve lip sync ability, 15% on ground-truth motions for driven ability, and 60% on joint video-motion generation. Retaining a considerable portion of pure imageto-video samples serves as a critical regularizer, improving identity preservation, video dynamics, and generalization.

#### 4. Experimental Results and Analysis 4.1. Experimental Details

Both our PIM and AIM are initialized from the wan2.25B [29]. The PIM is trained for 30,000 steps on 3-10 second clips of skeleton sequences and object trajectories (short side 256px), while the AIM is trained for 5,000 steps on 3-6 second video clips (short side 704px). In the initial pretraining phases for PIM and AIM, the learning rate is fixed

- at 1e-5. This is followed by a joint training phase of 4,000 steps where the learning rate is lowered to 2e-6.

##### 4.2. Benchmark Construction

To facilitate a rigorous evaluation for human object interaction generation, we constructed a benchmark with images generated by jimeng4.0, named GroundInter. GroundInter benchmark includes 400 images depicting actual or potential human-object interactions with 1-3 objects, with 100 different common objects. Each image is richly annotated with object types, 1-3 corresponding action descriptions (detailing interaction type, objects, and temporal segments), and matching dialogue scripts synthesized into audio using CosyVoice [8], resulting in 600 test cases. Additionally, we provide ground-truth object segmentation masks [17], detection results [40], and DW-Pose keypoints [38]. We introduce a suite of metrics focused on interaction quality: (1) VLM-QA, we design 30 questions structured around three categories (object, human, interaction), and the VLM [32] assigns a score of 1 or 0 to each based on the provided reference image and video. (2) Semantic Consistency (CLIPre), the CLIP-score [24] between the original text prompt and a VLM-generated caption of the generated video; (3) Hand Quality (HQ), assessed by the product of hand dynamics score and hand Laplacian sharpness score [18]; (4) Object Quality (OQ), assessed by the product of object dynamics score [42] and object DINO consistency [40]; and (5) Pixel-Level Interaction (PI), which verifies contact between DINO object boxes and DW-Pose keypoints. Our evaluation is further complemented by standard metrics. Ref Consistency with DINO for human appearance (DINOsubject) and ref image appearance (DINOref). Sync confidence (Syncconf) for audio-visual consistency [6], CLIP-B-T (CLIPtext) for text-video alignment, temporal consistency (Temp-C) and dynamic degree (DD) from VBench [39]. Please refer to the supplementary materials for more benchmark details.

##### 4.3. Comparison with the State-of-the-Art

As there are currently no methods specifically designed for the task of GHOI, we compare our method with state-ofthe-art (SOTA) approaches from several adjacent domains. For audio-driven methods, we select Hunyuan-Avatar [3], Wan-S2V [11], OmniAvatar [10], and Fantasy-Talking [31] for comparison. For subject-consistent generation, we compare against HuMo [2]. For pose-driven animation, we benchmark against UniAnimate-DiT [33]. VACE [16] is also utilized for comparison. As shown in Tab. 1, the audiodriven methods demonstrate an advantage in ref consistency metrics but perform poorly on human-object interaction, particularly on Hand Quality (HQ) and Object Quality (OQ) scores. This is attributable to their primary focus on facial dynamics, which only allows for the generation

of simple, low-dynamic actions and largely avoids interactive behaviors, also confirmed by their low dynamic degree (DD) scores. While this low dynamism facilitates the preservation of background and subject consistency, the absence of meaningful action is clearly reflected in the poor interaction scores. In contrast, our TA2V variant achieves a substantial lead in interaction metrics, representing an improvement of approximately 180% in HQ, 111% in OQ compared to the strong baseline Wan-S2V. Crucially, this is achieved while maintaining comparable reference preservation in terms of DINOsubject and DINOref. On lip synchronization performance, our method even outperforms all audio-driven method in the HOI scenario. For pose-driven methods, UniAnimate-DiT takes motion sequences generated by our PIM as input. Even with this high-quality guidance, our TAM2V variant outperforms it across all metrics, achieving a particularly notable margin on the OQ score. This discrepancy arises because pose-driven methods do not account for object morphology, leading to interaction misalignments and unnatural object deformations. Their inherent limitations also include the difficulty of obtaining and aligning pose sequences with the ref image, as well as the limited speaking ability. Regarding the subject-consistent method, HuMo generates an entirely new video with subjects borrowed from the ref image, rather than continuing from the original scene, resulting in lower scores on reference consistency. Furthermore, our method significantly surpasses HuMo in terms of interaction quality. This is because interactions generated by subject-consistent models tend to be stereotypical and fail to produce smooth transitions from a non-interactive to an interactive state. Notably, all our variants share a single model with different inference modes, highlighting its versatility and practical utility.

##### 4.4. Qualitative Evaluation

In Fig. 3, we present a qualitative comparison between our method and SOTA approaches from related domains for the Grounded Human-Object Interaction task. VACE exhibits responsiveness to text but suffers from severe artifacts during interaction, failing to preserve object integrity or accurately execute the interaction instructions. Moreover, it lacks the crucial audio-driven capability. Wan-S2V successfully preserves the interaction environment specified by the reference image. However, its dynamism is largely confined to the audio-driven facial region, limiting its ability to respond to broader interaction instructions. HuMo excels at generating fine-grained interaction details and facial dynamics. However, it fails to maintain the original interaction environment, instead using the subject from the reference image as an appearance template to synthesize an entirely new video, which is unsuitable for the GHOI task. In contrast, our method accurately follows textual instructions, enabling the human to interact stably and plausibly with ob-

Table 1. Quantitative comparison results on GroundedInter. With different input signals, our single model can infer with various modes. TA2V means audio-driven, TAM2V means audio and motion-driven, and T2MV represents the use of self-generated motion.

Human-Object Interaction Ref Consistency Text-Video Align Audio and Video VLM-QA ↑ HQ ↑ OQ ↑ DD ↑ PI ↑ DINOsubject ↑ DINOref ↑ CLIPtext ↑ CLIPre ↑ Temp-C ↑ Syncconf ↑

Task Method

Wan-S2V [11] 24.65 0.336 0.063 0.095 0.619 0.670 0.870 0.281 0.885 0.955 5.43 Hunyuan-Avatar [3] 26.88 0.745 0.112 0.321 0.666 0.661 0.841 0.285 0.873 0.934 5.98 Fantasy-Talking [31] 25.35 0.416 0.098 0.127 0.593 0.673 0.855 0.276 0.889 0.941 5.79

Audio-Driven

OminiAvatar [10] 26.76 0.732 0.118 0.305 0.732 0.639 0.847 0.282 0.885 0.940 5.95

Ours (TA2V) 27.32 0.931 0.133 0.765 0.803 0.671 0.857 0.286 0.899 0.945 6.04 Pose-Driven

UniAnimate-DiT [33] 24.65 0.862 0.082 0.441 0.669 0.621 0.782 0.277 0.875 0.938 – Ours (TAM2V) 28.47 0.957 0.141 0.773 0.832 0.648 0.842 0.287 0.902 0.940 5.73

VACE [16] 26.74 0.908 0.118 0.719 0.705 0.629 0.817 0.285 0.893 0.940 –

HuMo [2] 24.12 0.910 0.101 0.380 0.491 0.566 0.726 0.281 0.880 0.935 5.15 Ours (T2MV) 29.05 0.975 0.150 0.792 0.852 0.637 0.835 0.289 0.904 0.932 – Ours (TA2MV) 29.07 0.973 0.147 0.783 0.850 0.641 0.839 0.290 0.904 0.942 5.92

Text-Driven

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Text Prompt

Text Prompt

Text Prompt

Text Prompt

Ref Image + Prompt

Raise the camera to your eyes and take a picture

Pick up the apple on the table and show it to me

Take the knife off the apple and put it on table

Grabbed the tea cup off the table with one hand

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

UniAnimate-DiT

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

VACE

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

Wan-S2V

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

HuMo

[Figure 112]

[Figure 113]

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

OmniAvatar

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

HY-Avatar

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

FantasyTalking

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

Ours

Figure 3. Qualitative comparison with SOTA methods. Refer to Supp. Mat. for results of Hunyuan-Avatar and Fantasy-Talking.

Table 2. Effect of Environmental Perception Training.

jects within the reference image environment. This is facilitated by the explicit decoupling of perception and planning from video synthesis. The PIM plans plausible scene-aware interaction motions, while the AIM is responsible for synthesizing the photorealistic details of the interactions.

Rope Det. Det.+Cont. Last VLM-QA ↑ CLIPre ↑ HQ ↑ OQ ↑ PI ↑ 26.36 0.895 0.826 0.118 0.711

✓ 26.71 0.897 0.818 0.115 0.710 ✓ 27.15 0.902 0.845 0.123 0.730 ✓ ✓ 27.87 0.902 0.917 0.132 0.751 ✓ ✓ ✓ 28.89 0.904 0.925 0.144 0.780

##### 4.5. Ablation Studies

of our environmental perception training process, with the results summarized in Tab. 2. By default, the ref image is prepended to the video sequence. Last indicates that the

Effect of Environmental Perception Training. We conduct an ablation study to evaluate different configurations

###### Table 3. Ablation of PIM (top) and the Multimodal Training Strategy (bottom).

Method VLM-QA ↑ HQ ↑ OQ ↑ DD ↑ PI ↑ DINOsubject ↑ DINOref ↑ CLIPtext ↑ CLIPre ↑ Temp-C ↑ Syncconf ↑

w/o PIM 26.19 0.711 0.104 0.587 0.685 0.638 0.840 0.285 0.892 0.936 5.41 Cascade 28.19 0.876 0.137 0.730 0.746 0.642 0.841 0.289 0.901 0.940 5.27

Last-layer 28.31 0.880 0.141 0.752 0.769 0.643 0.849 0.289 0.901 0.938 5.32

Addition 28.46 0.903 0.134 0.751 0.777 0.630 0.846 0.286 0.887 0.935 5.14 Ours 28.89 0.925 0.144 0.761 0.780 0.646 0.846 0.290 0.904 0.942 5.43 AM 28.23 0.912 0.133 0.745 0.770 0.639 0.838 0.289 0.900 0.933 4.23

M → AM 28.84 0.920 0.141 0.764 0.778 0.636 0.837 0.290 0.903 0.934 3.98 A → AM 28.02 0.906 0.134 0.738 0.755 0.640 0.842 0.287 0.897 0.940 5.49

AI → IAM (Ours) 28.89 0.925 0.144 0.761 0.780 0.646 0.846 0.290 0.904 0.942 5.43

Ref Image Raise both hands to hold the ball and prepare to shoot Ref Image

Pick up the phone and check at it in front of you

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Cascade (Pose-Driven)

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

w/o PIM (TIA2V)

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Ours

Figure 4. Visualization of the PIM effect.

Table 4. Motion Representation Ablation.

ref image is appended after the last frame. RoPE specifies whether the modified Rotary Positional Embeddings (RoPE) are applied to the ref image. Det. denotes the use of detection data during PIM pre-training, which involves generating a single motion frame based on the ref image. Det.+Cont. represents generating a full motion sequence without ref motion. Our findings indicate that prepending the ref image yields superior performance compared to appending it, due to the structural similarity between the reference image and motion frames. However, this advantage is contingent upon modifying RoPE to mitigate the excessive influence of the ref image on the generated frames. Furthermore, we observe that detection data is crucial for establishing environmental perception abilities, while Det.+Cont. data serves as a vital bridge between perception and generation. Both types of data benefit the training of PIM.

Method VLM-QA ↑ HQ ↑ OQ ↑ DD ↑ PI ↑ DINOsub ↑ Syncconf ↑ 2D (x,y) Rep. 26.07 0.705 0.094 0.698 0.655 0.623 4.72

2D (x,y) Cascade 25.58 0.884 0.121 0.659 0.672 0.625 5.04 Ours 28.89 0.925 0.144 0.761 0.780 0.646 5.43

arately trained (frozen) PIM generates a motion sequence to drive the AIM. The cascaded approach creates an information bottleneck. Its reliance on explicit motion conditions is too restrictive to capture object shape changes, which in turn leads to artifacts and reduces the plausibility of the generated hands. In contrast, joint training allows for a richer flow of information. Within the joint training paradigm, injecting only the last-layer feature from PIM into all AIM block layers mirrors the performance of the cascaded model, suggesting that globally applied motion is less effective than adaptive, layer-wise guidance. Our layerwise residual injection mechanism, augmented with a zeroinitialized linear layer, also outperforms simple layer-wise addition, confirming the advantages of our design.

Effect of PIM. The homologous architecture of our PIM and AIM establishes intrinsic alignment in their feature representations, which we leverage for effective information fusion. As detailed in Tab. 3 (top), we conduct an ablation study on the injection of interaction motion features. w/o PIM refers to training a single AIM as a text-image-audiodriven video generator, which exhibits relatively poor text alignment and human-object interaction quality. This results in the absence of environmental perception capabilities and exacerbates the control-quality dilemma in GHOI. With PIM, we observe that a unified joint generation significantly outperforms a cascaded pipeline, in which a sep-

As shown in Fig. 4, we visualize the effects of our PIM. With frozen PIM, the cascaded variant can be regarded as a pose-driven model, while w/o PIM corresponds to a TIA2V model. The pose-driven variant can generate textaligned interaction videos but is prone to abnormal object deformations. The TIA2V variant often struggles with semantic comprehension, resulting in inaccurate execution of requested interactions, whereas our method successfully achieves reasonable and text-aligned interaction videos.

###### Motion Representation. In Tab. 4 (top), we compare

###### Table 5. Real Scene Comparision.

Method VLM-QA ↑ HQ ↑ OQ ↑ PI ↑ DINOsub ↑ CLIPtext ↑ Syncconf ↑

Wan-S2V 24.97 0.851 0.115 0.740 0.646 0.285 5.36 HY-Avatar 25.29 0.817 0.108 0.724 0.644 0.285 5.45

HuMo 26.23 0.840 0.116 0.781 0.635 0.288 5.33 Ours 28.49 0.910 0.135 0.794 0.665 0.289 5.51

our RGB motion representation with a simpler 2d coordinate representation. The coordinate motion generation model lacks priors and detection ability (generating motion requires an initial box and joint location) , and generalizes poorly. Its huge modality gap with video hinders alignment (most driven methods use RGB), while cascaded “generate (x,y) then render RGB for driven” markedly underperforms joint generation. RGB motion leverages video generation priors, improving text-to-motion accuracy and generalization without requiring large-scale datasets. Shared RGB space simplifies video-motion alignment and reduces training complexity, yielding better results for GHOI generation.

Real Scene Evaluation. For better SOTA comparison, we collect 50 cases from real scenes to build an additional test set. Quantitative experimental results on Tab. 5 demonstrate that our method performs well in real scenes.

Multimodal Training Strategy. This section investigates the optimal strategy for sequencing multimodal conditions during AIM training. As detailed in Tab. 3 (bottom), A → AM denotes first training on audio, followed by joint training on audio and motion. AM represents training both modalities from the beginning, without any AIM audio pre-training. The A → AM strategy achieves substantially better performance on audio metrics than either joint training from the beginning (AM) or pre-training on the motion modality first (M → AM). This outcome stems from the asymmetric nature of the conditioning signals. Consistent with the conclusions of Omnihuman [19], audio is a comparatively weaker signal than interaction motion and requires a dedicated pre-training stage for effective learning; otherwise, its influence is overshadowed by the dominant motion signal. Additionally, we find that enriching the audio-driven pre-training stage with Text-Image-to-Video data (without paired audio) yields notable benefits. This ensures that the model retains general motion dynamic priors, providing a stronger foundation for learning complex interaction motion during the final joint training phase.

User Study. We conduct a user study with 20 participants in 50 generation samples (each method). Given two videos generated from different methods, participants are asked to choose the better one. Results in Fig. 5.

|interaction<br><br>text alignm video quality<br><br>id preservation audio sync<br><br>|quality<br><br>ent 52%<br><br>54%<br><br>61%<br><br>87%<br><br>95%|
|---|---|

|interaction<br><br>text alignm video quality<br><br>id preservatio audio sync<br><br>|quality ent<br><br>n 66% 51%<br><br>63%<br><br>85%<br><br>92%|
|---|---|

Ours VS HY-Avatar

Ours VS Wan-S2V

|interaction q<br><br>text alignm video quality<br><br>id preserva audio sync|uality ent<br><br>tion 81%<br><br>72%<br><br>58%<br><br>81%<br><br>84%|
|---|---|

|interaction<br><br>text align video quali<br><br>id preservatio audio sync|quality<br><br>ment ty n 65%<br><br>66%<br><br>75%<br><br>93%<br><br>95%|
|---|---|

Ours VS HuMo

Ours VS Fantasy

Figure 5. User Study Results.

#### 5. Conclusion

This paper introduces a dual-stream talking avatar video generation framework that enables the generation of grounded human-object interactions. Our method decouples perception and planning from video synthesis, enabling the generation of plausible, controllable, and high-quality GHOI videos. The PIM is carefully designed to perceive the context of ref image and plan text-aligned interaction motion, while AIM generates vivid talking and interacting avatars with an M2V aligner. A benchmark is proposed for evaluating GHOI generation. Comprehensive experiments showcase our model’s performance and highlight each module’s contributions and impacts. Our work provides valuable insights for future research in this field. The main limitation of our work is that it can only handle a single-person scene and is unable to generate multi-person involved HOI.

#### References

- [1] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020. 4
- [2] Liyang Chen, Tianxiang Ma, Jiawei Liu, Bingchuan Li, Zhuowei Chen, Lijie Liu, Xu He, Gen Li, Qian He, and Zhiyong Wu. Humo: Human-centric video generation via collaborative multi-modal conditioning, 2025. 2, 3, 6, 7
- [3] Yi Chen, Sen Liang, Zixiang Zhou, Ziyao Huang, Yifeng Ma, Junshu Tang, Qin Lin, Yuan Zhou, and Qinglin Lu. Hunyuanvideo-avatar: High-fidelity audio-driven human animation for multiple characters. arXiv preprint arXiv:2505.20156, 2025. 2, 3, 6, 7
- [4] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, and Chenguang Ma. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2403–2410, 2025. 3
- [5] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, et al. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025. 2
- [6] J. S. Chung and A. Zisserman. Out of time: automated lip sync in the wild. In Workshop on Multi-view Lip-reading, ACCV, 2016. 6
- [7] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21086–21095, 2025. 2, 3
- [8] Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117, 2024. 6

- [9] Yingying Fan, Quanwei Yang, Kaisiyuan Wang, Hang Zhou, Yingying Li, Haocheng Feng, Errui Ding, Yu Wu, and Jingdong Wang. Re-hold: Video hand object interaction reenactment via adaptive layout-instructed diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17550–17560, 2025. 3
- [10] Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, and Steven Hoi. Omniavatar: Efficient audio-driven avatar video generation with adaptive body animation. arXiv preprint arXiv:2506.18866, 2025. 6, 7
- [11] Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, et al. Wan-s2v: Audio-driven cinematic video generation. arXiv preprint arXiv:2508.18621, 2025. 6, 7
- [12] Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, and Liefeng Bo. Animate anyone 2: High-fidelity character image animation with environment affordance. arXiv preprint arXiv:2502.06145, 2025. 3
- [13] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation, 2025. 2, 3
- [14] Ziyao Huang, Zixiang Zhou, Juan Cao, Yifeng Ma, Yi Chen, Zejing Rao, Zhiyong Xu, Hongmei Wang, Qin Lin, Yuan Zhou, et al. Hunyuanvideo-homa: Generic human-object interaction in multimodal driven human animation. arXiv preprint arXiv:2506.08797, 2025. 3
- [15] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. arXiv preprint arXiv:2409.02634, 2024. 3
- [16] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 6, 7
- [17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 6
- [18] Gaojie Lin, Jianwen Jiang, Chao Liang, Tianyun Zhong, Jiaqi Yang, and Yanbo Zheng. Cyberhost: Taming audiodriven avatar diffusion model with region codebook attention. arXiv preprint arXiv:2409.01876, 2024. 6
- [19] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025. 2, 3, 9
- [20] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via crossmodal alignment. arXiv preprint arXiv:2502.11079, 2025. 2, 3
- [21] Yifang Men, Yuan Yao, Miaomiao Cui, and Liefeng Bo. Mimo: Controllable character video synthesis with spatial decomposed modeling. In Computer Vision and Pattern Recognition (CVPR), 2025 IEEE Conference on, 2025. 3

- [22] Rang Meng, Xingyu Zhang, Yuming Li, and Chenguang Ma. Echomimicv2: Towards striking, simplified, and semibody human animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5489–5498,

2025. 2

- [23] Youxin Pang, Ruizhi Shao, Jiajun Zhang, Hanzhang Tu, Yun Liu, Boyao Zhou, Hongwen Zhang, and Yebin Liu. Manivideo: Generating hand-object manipulation video with dexterous and generalizable grasping. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12209–12219, 2025. 3
- [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 6
- [25] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. 3
- [26] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image animation with enhanced motion representation. arXiv preprint arXiv:2410.10306, 2024. 2
- [27] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244–260. Springer, 2024. 2
- [28] Shuyuan Tu, Zhen Xing, Xintong Han, Zhi-Qi Cheng, Qi Dai, Chong Luo, and Zuxuan Wu. Stableanimator: Highquality identity-preserving human image animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21096–21106, 2025. 2
- [29] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3, 5
- [30] Cong Wang, Kuan Tian, Jun Zhang, Yonghang Guan, Feng Luo, Fei Shen, Zhiwei Jiang, Qing Gu, Xiao Han, and Wei Yang. V-express: Conditional dropout for progressive training of portrait video generation. arXiv preprint arXiv:2406.02511, 2024. 2

- [31] Mengchao Wang, Qiang Wang, Fan Jiang, Yaqi Fan, Yunpeng Zhang, Yonggang Qi, Kun Zhao, and Mu Xu. Fantasytalking: Realistic talking portrait generation via coherent motion synthesis. arXiv preprint arXiv:2504.04842, 2025. 2, 3, 6, 7
- [32] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 6
- [33] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. Unianimate-dit: Human image animation with large-scale video diffusion transformer. arXiv preprint arXiv:2504.11289, 2025. 2, 6, 7
- [34] Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, and Dahua Lin. Interacthuman: Multi-concept human animation with layoutaligned audio conditions. arXiv preprint arXiv:2506.09984,

2025. 3

- [35] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 2, 3
- [36] Ziyi Xu, Ziyao Huang, Juan Cao, Yong Zhang, Xiaodong Cun, Qing Shuai, Yuchen Wang, Linchao Bao, Jintao Li, and Fan Tang. Anchorcrafter: Animate cyberanchors saling your products via human-object interacting video generation. arXiv preprint arXiv:2411.17383, 2024. 3
- [37] Zihui Sherry Xue, Romy Luo, Changan Chen, and Kristen Grauman. Hoi-swap: Swapping objects in videos with handobject interaction awareness. Advances in Neural Information Processing Systems, 37:77132–77164, 2024. 3
- [38] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4210–4220, 2023. 6
- [39] Fan Zhang, Shulin Tian, Ziqi Huang, Yu Qiao, and Ziwei Liu. Evaluation agent: Efficient and promptable evaluation framework for visual generative models. arXiv preprint arXiv:2412.09645, 2024. 6
- [40] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection, 2022. 6
- [41] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8652–8661, 2023. 2
- [42] Youliang Zhang, Zhaoyang Li, Duomin Wang, Jiahe Zhang, Deyu Zhou, Zixin Yin, Xili Dai, Gang Yu, and Xiu Li. Speakervid-5m: A large-scale high-quality dataset for audiovisual dyadic interactive human generation. arXiv preprint arXiv:2507.09862, 2025. 6

