[Figure 1]

## Any2Caption : Interpreting Any Condition to Caption for Controllable Video Generation

Shengqiong Wu1,2* Weicai Ye1, Jiahao Wang1 Quande Liu1 Xintao Wang1 Pengfei Wan1 Di Zhang1 Kun Gai1 Shuicheng Yan2 Hao Fei2, Tat-Seng Chua2 1Kuaishou Technology 2National University of Singapore

https://sqwu.top/Any2Cap/

# arXiv:2503.24379v2[cs.CV]12Dec2025

Disentangled !

|[Figure 2]|
|---|

[Figure 3]

|A stylish woman walksthrough a bustling …|
|---|

[Figure 4]

[Figure 5]

[Figure 6]

Stage-1: Condition Interpretation

Stage-2: Video Generation

Camera Pose

Sketch

Text

(Text + Depth + Camera) → Video

|[Figure 7]|
|---|

[Figure 8]

[Figure 9]

Off-the-shelf SoTA Video Generators

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Structured Dense Captions of Any Cond

(Updated )

[Figure 14]

(Frozen )

First Frame Multiple Identities

(Text + First Frame ) → Video

Dense caption

[Figure 15]

[Figure 16]

[Figure 17]

A scene of a woman in the kitchen …

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

CTRL-Adapter

[Figure 22]

Main object caption

VideoComposer

Depth Sequence

（Text + Multiple Identities ) → Video

A woman with gold hair is wearing blue …

### Any2Caption

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

ControlVideo

Background caption

[Figure 29]

[Figure 30]

[Figure 31]

There are a painting hanging up on the …

CameraCtrl

（Text + Human Pose ) → Video

Human Pose Sequence

Camera caption

[Figure 32]

[Figure 33]

[Figure 34]

ConceptMaster

[Figure 35]

[Figure 36]

[Figure 37]

The fixed camera is shooting the upper …

Style caption

MotionCtrl

Segmentation Normal Bae Style

（Text + Key Frames ) → Video

Realistic with warm, bright light …

HunYuan

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Action caption

###### …

A woman is standing behinda desk …

CogVideoX

…

（Text + Style ) → Video

Key Video Frames

Any/Diverse Input Conditions Generation Output Videos

Interpreting Condition → Caption

Figure 1. We propose Any2Caption, an efficient and versatile framework for interpreting diverse conditions to structured captions, which then can be fed into any video generator to generate highly controllable videos.

#### Abstract

with better guidance. We also introduce Any2CapIns, a large-scale dataset with 337K instances and 407K conditions for any-condition-to-caption instruction tuning. Comprehensive evaluations demonstrate significant improvements of our system in controllability and video quality across various aspects of existing video generation models.

To address the bottleneck of accurate user intent interpretation within the current video generation community, we present Any2Caption, a novel framework for controllable video generation under any condition. The key idea is to decouple various condition interpretation steps from the video synthesis step. By leveraging modern multimodal large language models (MLLMs), Any2Caption interprets diverse inputs—text, images, videos, and specialized cues such as region, motion, and camera poses—into dense, structured captions that offer backbone video generators

#### 1. Introduction

Video serves as a fundamental medium for capturing realworld dynamics, making diverse and controllable video generation a key capability for modern artificial intelligence (AI) systems. Recently, video generation has gained signif-

∗Work done during internship at Kuaishou Technology.

Corresponding Author.

icant attention, driven by advancements in Diffusion Transformers (DiT) [2, 29, 44, 52, 76], which have demonstrated the ability to generate realistic, long-duration videos from text prompts. These advancements have even led to industrial applications, such as filmmaking. However, we observe that a major bottleneck in the current video generation community lies in accurately interpreting user intention, so as to produce high-quality, controllable videos.

In text-to-video (T2V) generation, studies [24, 30, 67] have suggested that detailed prompts, specifying objects, actions, attributes, poses, camera movements, and style, significantly enhance both controllability and video quality. Thus, a series of works have explored video recaption techniques (e.g., ShareGPT4Video [10], MiraData [30], and InstanceCap [15]) to build dense structured captions for optimizing generative models. While dense captions are used during training, in real-world inference scenarios, users most likely provide concise or straightforward input prompts [15]. Such a gap inevitably weakens instruction following and leads to suboptimal generation due to an incomplete understanding of user intent. To combat this, there are two possible solutions, manual prompt refinement or automatic prompt enrichment [15, 67] using large language models (LLMs). Yet, these approaches either require substantial human effort or risk introducing noise from incorrect prompt interpretations. As a result, this limitation in precisely interpreting user intent hinders the adoption of controllable video generation for demanding applications such as anime creation and filmmaking.

In addition, to achieve more fine-grained controllable video generation, one effective strategy is to provide additional visual conditions besides text prompts—such as reference images [17, 62], identity [22, 46, 69], style [42, 68], human pose [33, 45], or camera [21, 75]—or even combinations of multiple conditions together [41, 58, 74]. This multimodal conditioning approach aligns well with real-world scenarios, as users quite prefer interactive ways to articulate their creative intent. Several studies have examined video generation under various conditions, such as VideoComposer [58], Ctrl-Adapter [41], and ControlVideo [74]. Unfortunately, these methods tend to rely on the internal encoders of Diffusion/DiT to parse rich heterogeneous input conditions with intricate requirements (e.g., multiple object IDs, and complex camera movements). Before generation, the model must accurately interpret the semantics of varied visual conditions in tandem with textual prompts. Yet even state-of-the-art (SoTA) DiT backbones have limited capacity for reasoning across different input modalities, resulting in suboptimal generation quality.

This work is dedicated to addressing these bottlenecks of any-conditioned video generation. Our core idea is to decouple the first job of interpreting various conditions from the second job of video generation, motivated by

two important observations:

- a) SoTA video generation models (e.g., DiT) already excel at producing high-quality videos when presented with sufficiently rich text captions;
- b) Current MLLMs have demonstrated robust visionlanguage comprehension.

Based on these, we propose Any2Caption, an MLLMbased universal condition interpreter designed not only to handle text, image, and video inputs but also equipped with specialized modules for motion and camera pose inputs. As illustrated in Fig. 1, Any2Caption takes as inputs any/diverse condition (or combination), and produces a densely structured caption, which is then passed on to any backbone video generators for controllable, high-quality video production. As Any2Caption disentangles the role of complex interpretation of multimodal inputs from the backbone generator, it advances in seamlessly integrating into a wide range of well-trained video generators without the extra cost of fine-tuning.

To facilitate the any-to-caption instruction tuning for Any2Caption, we construct Any2CapIns, a large-scale dataset that converts a concise user prompt and diverse non-text conditions into detailed, structured captions. Concretely, the dataset encompasses four main categories of conditions: depth maps, multiple identities, human poses, and camera poses. Through extensive manual labeling combined with automated annotation by GPT-4V [1], followed by rigorous human verification, we curate a total of 337K high-quality instances, with 407K condition annotations, with the short prompts and structured captions averaging 55 and 231 words, respectively. In addition, we devise a comprehensive evaluation strategy to thoroughly measure the model’s capacity for interpreting user intent under these various conditions.

Experimentally, we first validate Any2Caption on our Any2CapIns, where results demonstrate that it achieves an impressive captioning quality that can faithfully reflect the original input conditions. We then experiment with integrating Any2Caption with multiple SoTA video generators, finding that (a) the long-form semantically rich prompts produced by Any2Caption are pivotal for generating high-quality videos under arbitrary conditions, and (b) Any2Caption consistently enhances performance across different backbone models, yielding noticeably improved outputs. Furthermore, Any2Caption shows a pronounced advantage when handling multiple combined conditions, effectively interpreting and synthesizing intricate user constraints into captions that align closely with user expectations. Our contributions are threefold:

• We for the first time pioneer a novel any-condition-tocaption paradigm of video generation, which bridges the gap between user-provided multimodal conditions and structured video generation instructions, leading to

Step-2: Structured Video Caption Generation Step-3: User-centric Short Prompt Generation

- Step-1: Data Collection

Target Videos Structured Caption Short Prompt

[Figure 43]

[Figure 44]

Constraints

[Figure 45]

[Figure 46]

Tools AnythingDepth

[Figure 47]

[Figure 48]

- 1. Dense caption: The video begins in a cozy...
- 2. Main object caption: The main objects are a ...
- 3. Background caption: The background consists ....
- 4. Camera caption: The camera first pans to the …
- 5. Style caption: Clean, bright, and inviting ...
- 6. Action caption: A woman is standing besides...

[Figure 49]

Video

[Figure 50]

[Figure 51]

Structured Caption

Source

[Figure 52]

Conditions

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

A stylish woman in a dark jacket walks through a

ManualChecker

ManualChecker

SAM

[Figure 58]

| |
|---|

[Figure 59]

[Figure 60]

StructuredVideo CaptionSystem

[Figure 61]

GPT-4V

337K

[Figure 62]

Any2CapIns

bustling urban

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Target Video

[Figure 70]

[Figure 71]

[Figure 72]

area. Capture the tall buildings and busy street behind her.

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

Target Video

[Figure 84]

[Figure 85]

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

Target Video

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- Figure 2. The pipeline for constructing the Any2CapIns dataset involves three key steps: 1) data collection, 2) structured video caption generation, and 3) user-centric short prompt generation.

highly-controllable video generation.

- • We propose Any2Caption to effectively integrate and comprehend diverse multimodal conditions, producing semantically enriched, long-form, structured captions, which consistently improve both condition flexibility and video quality. Any2Caption can also be widely integrated as a plug-in module for any existing video generation framework.
- • We introduce Any2CapIns, a large-scale, highquality benchmark dataset for the any-condition-tocaption task, and also establish a suite of evaluation metrics to rigorously assess the quality and fidelity of condition-based caption generation.

#### 2. Related Work

Controllable video generation [11, 16, 21, 53] has long been a central topic in AI. Recent advanced DiT methods, such as OpenAI’s Sora [3] and HunyuanVideo [34], yield photorealistic videos over extended durations. Early work focused on text-controlled video generation [24, 52], the prevalent approach. Yet, text prompts alone may insufficiently capture user intent, spurring exploration of additional inputs including static images [17, 62], sketches [58, 74], human poses [33, 45, 77], camera views [21, 75], and even extra videos [14, 32, 70]. Thus, unifying these diverse conditions into an “any-condition” framework is highly valuable.

Recent works such as VideoComposer [58], CtrlAdapter [41], and ControlVideo [74] have explored anycondition video generation. However, they face challenges in controlling multiple modalities due to the limited interpretability of text encoders in Diffusion or DiT. Motivated by existing MLLMs’ multimodal reasoning [39, 43, 56], we propose leveraging an MLLM to consolidate all possible conditions into structured dense captions for better controllable generation. SoTA DiT models already exhibit the capacity to interpret dense textual descriptions as long as the input captions are sufficiently detailed in depicting both the scene and the intended generation goals. Thus, our MLLMbased encoder alleviates the comprehension bottleneck, enabling higher-quality video generation. To our knowledge, this is the first attempt in the field of any-condition video generation. Moreover, as the captioning stage is decoupled

from backbone DiT, Any2Caption can integrate with existing video generation solutions without additional retraining.

Our approach also relates to video recaptioning, as our system produces dense captions from given conditions. In text-to-video settings, prior work [15, 27, 47, 67] shows that recaptioning yields detailed annotations that improve DiT training. For instance, ShareGPT4Video [10] uses GPT-4V [1] to reinterpret video content, while MiraData [30] and InstanceCap [15] focus on structured and instance-consistent recaptioning. Unlike these methods, we avoid retraining powerful DiT models with dense captions by training an MLLM as an any-condition encoder on pairs of short, dense captions that are easier to obtain. Moreover, recaptioning entire videos can introduce noise or hallucinations that undermine DiT training, whereas our framework sidesteps this risk. Finally, while previous studies rely on dense-captiontrained DiT models, the real-world user concise prompts might create a mismatch that degrades generation quality.

#### 3. Any2CapIns Dataset Construction

While relevant studies recaption target videos for dense captions for enhanced T2V generation [10, 15, 30], these datasets exhibit two key limitations: (1) the absence of non-text conditions, and (2) short prompts that do not account for interactions among non-text conditions, potentially leading to discrepancies in real-world applications. To address these limitations, we introduce a new dataset, Any2CapIns, specifically designed to incorporate diverse multimodal conditions for generating structured video captions. The dataset is constructed through a three-step process (cf. Fig. 2), including data collection, structured caption construction, and user-centric short prompt generation.

Step-1: Data Collection. We begin by systematically categorizing conditions into four primary types: 1) Spatialwise conditions, which focus on the structural and spatial properties of the video, e.g., depth maps, sketches, and video frames. 2) Action-wise conditions, which emphasize motion and human dynamics in the target video, e.g., human pose, motion. 3) Composition-wise conditions, which focus on scene composition, particularly in terms of ob-

Short Caption Structured Caption

NumberofVideos

80k

60k

40k

20k

0k

0 26 34 42 51 59 67 76 84 92

170 186 202 218 235 251 267 284 300 316

Length

- Figure 3. Distribution of the short/structured caption length (in words) in Any2CapIns.

ject interactions and multiple identities in the target video. 4) Camera-wise conditions, which control video generation from a cinematographic perspective, e.g., camera angles, movement trajectories. Since it is infeasible to encompass all possible conditions in dataset collection, we curate representative datasets under each category, specifically including depth maps, human pose, multiple identities, and camera motion. During the data collection process, we leverage SoTA tools to construct conditions. For instance, Depth Anything [65] is used to generate depth maps, DWPose [66] provides human pose annotations, and SAM2 [50] is utilized for segmentation construction. In total, we collect 337K video instances and 407K conditions, with detailed statistics of the dataset presented in Tab. 1.

- Step-2: Structured Video Caption Generation. The granularity of a caption, specifically the detailed selection of elements they encompass, plays a critical role in guiding the model to produce videos that closely align with the users’ intentions while preserving coherence and realism. Drawing inspiration from [30], we inherit its structured caption format consisting of (1) Dense caption, (2) Main object caption, (3) Background caption, (4) Camera caption, and (5) Style caption. Furthermore, the action descriptions of the subjects significantly influence the motion smoothness of the videos [57], we explicitly incorporate the (6) Action caption of subjects to form the final structured caption. Technically, following [57], we independently generate each caption component and subsequently integrate them to construct the final structured caption.
- Step-3: User-centric Short Prompt Generation. In this step, we construct short prompts from a user-centric perspective, considering how users naturally express their intentions. Firstly, our analysis highlights three key characteristics of user-generated prompts: 1) Conciseness and Simplicity, where users favor brief and straightforward wording; 2) Condition-Dependent Omission, whereby users often omit explicit descriptions of certain attributes (e.g., camera movement) when such conditions are already specified; and 3) Implicit instruction of Target Video: where users convey their intent indirectly (e.g., specifying multiple identities without explicitly detailing their interactions).

Category #Inst. #Condition #Avg. Len. #Total Len.

Depth 182,945 182,945 9.87s 501.44h Human Pose 44,644 44,644 8.38s 108.22h Multi-Identities 68,255 138,089 13.01s 246.69h Camera Movement 41,112 41,112 6.89s 78.86h

Table 1. Statistics of the collected dataset across four types of conditions. #Inst. means the number of instances, and #Condition denotes the number of unique conditions. #Avg. / #Total Len. indicate the average and total video durations, respectively.

Guided by these observations, we employ GPT-4V to infer potential user prompts under condition-specific constraints. Given a structured video caption, target video, and associated conditions, we apply tailored constraints to preserve condition invariance when relevant. We also explicitly control the length of the generated prompts to ensure conciseness. Finally, we conduct manual verification and filtering to further refine the dataset. Fig. 3 presents the length distribution of the resulting short and structured prompts.

#### 4. Any2Caption Model

In this section, we introduce Any2Caption, a novel MLLM explicitly designed to comprehensively model and interpret arbitrary multimodal conditions for controllable video caption generation, as illustrated in Fig. 4(a). Formally, the user provides a short text prompt T along with non-text conditions C = [c1,··· ,cn], where the non-text conditions can be either a single condition (n = 1) or a combination of multiple conditions. The objective of this task is to generate a detailed and structured caption that serves as a control signal for video generation.

Architecture. Similar to existing MLLMs [39, 43, 63], Any2Caption incorporates an image encoder FI, a video encoder FV, a motion encoder FM and a camera encoder FC} to process non-text conditions. These encoders are then integrated into an LLM backbone FLLM (i.e., Qwen2LLM) to facilitate structured video captioning. Specifically, we leverage a ViT-based visual encoder from Qwen2VL as FI and FV for the unified modeling of images and videos, achieving effective interpretation of input conditions represented in image or video formats, such as depth maps and multiple identities. To enable human pose understanding, we represent the extracted human pose trajectories as H={(xkn,ynk)|k = 1,··· ,K,n = 1,··· ,N}, where N denotes the number of video frames and K is the number of keypoints. These trajectories are then visualized within video frames to enable further processing by the motion encoder, which shares the same architectural structure and initialization as the vision encoder. For camera motion understanding, inspired by [21], we introduce a camera encoder that processes a pl¨ucker embedding sequence P ∈ RN×6×H×W, where H, W are the height and width of the video. This embedding accurately captures camera pose information, enabling pre-

[Figure 106]

VL weights, we fine-tune the model on the Any2CapIns dataset for multimodal condition interpretation. However, direct fine-tuning leads to catastrophic forgetting due to the fixed output structure. To address this, we propose a progressive mixed training strategy. Specifically, the model is first trained on a single condition to establish a strong condition-specific understanding. As new conditions are introduced, we gradually incorporate vision-language instructions such as LLaVA-instruction [43] and Alpaca-52K [54]. This stepwise strategy ensures robust multimodal condition interpretation while preventing knowledge degradation.

Structured Caption

[Figure 107]

Qwen2-LLM

[Figure 108]

[Figure 109]

Image Encoder

Camera Encoder

Text Encoder

Motion Encoder

Video Encoder

CTRL-Adapter

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

VideoCompose r

r

[Figure 117]

ControlVideo

- (a) Overall architecture of Any2Caption
- (b) Progressive Mixed Training

CameraCtrl

Conditions Short Caption

Training Ratio

[Figure 118]

[Figure 119]

Multi-IDs

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 124]

ConceptMaster

#### 5. Evaluation Suite

[Figure 125]

Joint Training dropout

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Human Pose

In this section, we introduce the evaluation suite for comprehensively assessing the capability of Any2Caption in interpreting user intent and generating structured captions.

[Figure 131]

[Figure 132]

[Figure 133]

MotionCtrl

[Figure 134]

0.4

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Camera

0.6

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Joint Training dropout

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Lexical Matching Score. To assess caption generation quality from a lexical standpoint, we employ standard evaluation metrics commonly used in image/video captioning tasks, including BLEU [48], ROUGE [40], and METEOR [4]. We also introduce a Structural Integrity score to verify whether the generated captions adhere to the required six-component format, thereby ensuring completeness.

[Figure 154]

[Figure 155]

0.8

Depth

[Figure 156]

[Figure 157]

[Figure 158]

Any2CapIns

Additional Vision/Text Ins.

- Figure 4. Architecture illustration of Any2Caption (a), where Qwen2-LLM serves as the backbone and is paired with text, image, video, motion, and camera encoders to produce structured captions. After alignment learning, we perform a progressive mixed training strategy (b), where additional vision/text instruction datasets are progressively added for joint training, and meanwhile, for input short caption, we adopt a random-dropout mechanism at the sentence level to enhance robustness.

Semantic Matching Score. To evaluate the semantic alignment of generated captions, we employ BERTSCORE [71], which computes similarity by summing the cosine similarities between token embeddings, effectively capturing both lexical and compositional meaning preservation. Additionally, we utilize CLIP Score [23] to assess the semantic consistency between the input visual condition and the generated videos.

cise modeling of camera trajectories. Finally, in line with Qwen2-VL, we employ special tokens to distinguish nontext conditions from texts. Besides the existing tokens, we introduce <|motion start|>, <|motion end|>, <|camera start|>, <|camera end|>, to demarcate the start and end of human and camera pose features.

Intent Reasoning Score. Evaluating structured captions typically focuses on their linguistic quality, yet it rarely assesses whether the model truly understands user intent and accurately reflects it across aspects such as style, emotion, and cinematic language. To bridge this gap and draw inspiration from [8], we introduce the Intent Reasoning Score (IRSCORE), a novel quantitative metric that leverages LLMs to assess whether generated captions accurately capture user intentions. IRSCORE identifies user-specified control factors, then decomposes the generated caption into QA pairs aligned with these factors. The framework has four steps: (1) User Intention Extraction: Analyze provided conditions to categorize user intent into six aspects: subject, background, movement, camera, interaction, and style. (2) Ground-Truth QA Pair Construction: Formulate aspect-specific QA pairs with defined requirements (e.g., for subject-related attributes, emphasize object count, appearance). (3) Predicted Answer Generation: Prompt GPT-4V to parse the predicted caption and generate answers based solely on it. (4) Answer Evaluation: Following [36], GPT-4V outputs two scores (correctness and quality) for

Training Recipes. To accurately interpret user generation intent under arbitrary conditions and yield structured target video captions, large-scale pretraining and instruction tuning are required. To this end, we adopt a two-stage training procedure: Stage-I: Alignment Learning. In this stage, as image and video encoders have been well-trained in Qwen2-VL, we only focus on aligning human pose features from the motion encoder and camera movement features with the word embeddings of the LLM backbone. To achieve this, we freeze the LLM and vision encoder while keeping the motion encoder trainable and optimizing it on a human pose description task. Similarly, for camera movement alignment, we unfreeze the camera encoder and train it on a camera movement description task, ensuring that camera-related conditions are embedded into the model’s latent space. This alignment phase establishes a strong foundation for effective representation learning for these conditions. Stage II: Condition-Interpreting Learning Building upon the aligned encoders and pretrained Qwen2-

Lexical Matching Semantic Matching Intent Reasoning

Category Structural Integrity

B-2 R-L METER BERTSCORE Accuracy Quality

Entire Structured Caption 91.25 54.99 48.63 52.47 91.95 68.15 3.43 Dense Caption - 44.24 42.89 49.51 92.42 78.47 3.47 Main Object Caption - 38.54 47.46 52.48 92.02 56.28 2.74 Background Caption - 44.65 46.73 48.87 92.90 69.37 2.69 Camera Caption - 60.21 96.10 94.32 99.31 66.31 3.75 Style Caption - 41.71 47.70 55.9 93.48 63.75 3.05 Action Caption - 31.91 39.83 45.25 91.44 57.98 2.13

- Table 2. Quantitative results of structured caption generation quality under four aspects: structural Integrity, lexical matching, semantic matching, and intent reasoning. We demonstrate the overall caption generation capability and the individual component generation performance within the structure. “B-2” and “R-L” denotes BLEU-2 and ROUGE-L, respectively.

Caption Enrich

Text Video Generation

CLIP-T↑ Smoothness↑ Aesthetic↑ Integrity↑

Short Cap. 18.31 93.46 5.32 55.39 Short Cap.

19.19 93.41 5.41 54.91 w/ Condition Cap.

Structured Cap. 19.87 94.38 5.46 57.47

- Table 3. Quantitative results comparing short caption, short caption combined with condition caption, and structured caption for multi-identity video generation.

Training Strategy

Caption Vieo Generation

B-2↑ Accuracy↑ Smoothness↑ Dynamics↑ Aesthetic↑

Any2Caption 47.69 67.35 94.60 17.67 5.53 w/o Two-Stage 33.70 51.79 93.31 16.36 5.50 w/o Dropout 49.24 69.51 94.16 14.54 5.51

- Table 4. Ablation study on training strategy. “w/o stage” means alignment learning is not applied during training, while “w/o Dropout” denotes that short captions are not randomly dropped.

ties, camera, and compositional conditions) to evaluate the model’s performance. Additionally, we assess the model on publicly available benchmarks (e.g., [41, 68, 78]). For further details, please refer to the Appendix §F.

Implementation Details. We leverage Qwen2VL-7B [56] as the backbone of our model, which supports both image and video understanding. The human pose in the input conditions is encoded and processed in the video format. The camera encoder adopts the vision encoder architecture with the following settings: the input channel of 96, the patch size of 16, the depth of 8, and 8 attention heads. During training, to simulate the brevity and randomness of user inputs, we randomly drop sentences from the short caption with a dropout rate of 0.6; a similar dropout strategy is applied to non-textual conditions. We conducted the training on 8×80G GPUs. For further details on the training parameters for each stage, please refer to the Appendix §F.

each answer, which are then averaged across all QA pairs. More details are in Appendix §E.

##### 6.2. Experimental Results and Analyses

In this section, we present the experimental results and provide in-depth analyses to reveal how the system advances. Following this, we try to ground the answers for the following six key research questions.

Video Generation Quality Score. The primary objective of generating structured captions is to enhance the video generation quality and controllability. Therefore, we adopt a series of metrics to assess the quality of videos generated based on structured captions. Following [26, 30], we evaluate video generation quality across four key dimensions: motion smoothness, dynamic degree, aesthetic quality, and image integrity. To further verify adherence to specific non-text conditions, we use specialized metrics: RotErr, TransErr, and CamMC [21] for camera motion accuracy; Mean Absolute Error (MAE) for depth consistency [18]; DINO-I [51], CLIP-I [51] Score to evaluate identity preservation under multiple identities, and Pose Accuracy (Pose Acc.) [45] to access the alignment in the generated videos.

RQ-1: How well is the structured caption generation quality? We first evaluate whether our proposed model could accurately interpret user intent and generate highquality structured captions. From a caption-generation perspective, we compare the predicted captions with goldstandard captions across various metrics (refer to Tab. 2). We observe that our model successfully produces the desired structured content, achieving 91.25% in structural integrity. Moreover, it effectively captures the key elements of the gold captions, attaining a ROUGE-L score of 48.63 and a BERTSCORE of 91.95. Notably, the model demonstrates the strongest performance in interpreting camera-related details compared to other aspects. Finally, regarding user intent analysis, we found that the model reliably incorporated user preferences into its structured outputs.

#### 6. Experiments

##### 6.1. Setups

Dataset. We manually construct 200 test cases for each type of condition (i.e., depth, human pose, multiple identi-

|Diffusion Models|Diffusion Models|
|---|---|

Example-1: A man gestures while the woman listens. They sit in a sunny park. The camera captures close-up shots of their heads and shoulders.

Example-3: A young man carrying a messenger bag runs down a narrow, cobblestone street filled with sandbags and crates, suggesting a wartime.

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

IDs

ID CogVideoX-2B

Structured Cap. Video Short Cap. + ID Structured Cap. Video

Short Cap. + IDs

HunyuanVideo

Any2Caption

Any2Caption

Example-4: A woman walks in a minimalist, modern room. She is holding two mugs and looks slightly displeased. The room has natural light.

Example-2: A serene winter backyard with snow-covered ground and bare trees, revealing a blue shed with a white garage door and a doghouse

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

Depth IDs Depth

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Camera

Short Cap. + Camera + Depth Structured Cap. Video Short Cap. + IDs + Depth Structured Cap. Video

HunyuanVideo

Any2Caption

CogVideoX-2B

Any2Caption

- Figure 5. Illustrations of generated videos where only the structured captions yielded by Any2Caption are fed into the CogVideoX-2B (Left), and HunyuanVideo (Right). We can observe that some key features of the input identity images, such as the background and main object, can be accurately visualized in the generated videos.

Text Camera Identities Depth Human Pose Overall Quality

Model

CLIP-T↑ RotErr↓ TransErr↓ CamMC↓ DINO-I↑ CLIP-I↑ MAE↓ Pose Acc.↑ Smoothness↑ Dynamic↑ Aesthetic↑ Integrity↑

[Figure 190]

###### • Camera to Video

[Figure 191]

[Figure 192]

MotionCtrl [60] 19.67 1.54 4.49 4.80 - - - - 96.13 9.75 5.40 73.69

[Figure 193]

- + Structured Cap. 20.16 1.45 4.37 4.78 - - - - 96.16 11.43 5.71 74.63

CameraCtrl [21] 18.89 1.37 3.51 4.78 - - - - 94.11 12.59 4.26 71.84

- + Structured Cap. 21.70 0.94 2.97 4.37 - - - - 95.16 13.72 4.66 72.47

• Depth to Video

Ctrl-Adapter [41] 20.37 - - - - - 25.63 - 94.53 20.73 4.63 46.98

- + Structured Cap. 23.30 - - - - - 21.87 - 95.54 15.14 5.31 54.20

ControlVideo [74] 22.17 - - - - - 30.11 - 92.88 5.94 5.29 63.85

- + Structured Cap. 24.18 - - - - - 23.92 - 94.47 18.27 9.77 66.28

###### • Identities to Video

ConceptMaster [25] 16.04 - - - 36.37 65.31 - - 94.71 8.18 5.21 43.68 + Structured Cap. 17.15 - - - 39.42 66.74 - - 95.05 10.14 5.68 49.73

###### • Human Pose to Video

FollowYourPose [45] 21.11 - - - - - - 30.47 91.71 14.29 4.95 58.84 + Structured Cap. 21.39 - - - - - - 31.59 92.87 16.47 5.88 56.30

- Table 5. Performance comparison on four types of conditions (e.g., camera, depth, identities, and human pose) between directly using short captions and integrating structured captions under various video quality evaluation metrics. Better results are marked in bold.

To further showcase the model’s capacity to understand and leverage input conditions, we directly feed the structured captions—derived from our model’s interpretation—into downstream text-to-video generation systems (e.g., CogvideoX-2B [67] and Hunyuan [34]), as illustrated in Fig. 5. Even without explicit visual conditions (e.g., identities, depth, or camera movement), the resulting videos clearly align with the input prompts, such as the hat’s color and style, or the woman’s clothing texture in Example 1, indicating that our structured captions successfully capture intricate visual details. In particular, the model is able to accurately grasp dense conditions, such as depth sequences or compositional requirements in Example 4, ultimately enabling controllable video generation. While certain finegrained elements may not be exhaustively described in text, resulting in occasional discrepancies with the actual visual content, the overall controllability remains robust.

- RQ-2: Is the video generation quality enhanced with structured caption? Here, we investigate whether integrating structured captions consistently improves controllable video generation in multiple methods. We explore the impact of adding camera, depth, identities, and human pose conditions to various controllable video generation methods. As shown in Tab. 5, all tested models exhibit consistent gains in overall video quality, such as smoothness and frame fidelity, after incorporating structured captions, without requiring any changes to the model architectures or additional training. Moreover, these models show enhanced adherence to the specified conditions, suggesting that our generated captions precisely capture user requirements and lead to more accurate, visually coherent video outputs. More examples can be found in the Appendix §G.2.
- RQ-3: Is the structured caption necessary? We examine whether a structured caption design is essential. We

Text Camera Identities Depth Overall Quality

Compositional Condition

CLIP-T↑ RotErr↓ TransErr↓ CamMC↓ DINO-I↑ CLIP-I↑ MAE↓ Smoothness↑ Dynamic↑ Aesthetic↑ Integrity↑ Camera+Identities 14.81 1.37 4.04 4.24 25.63 64.14 - 94.43 28.87 4.99 59.81

+ Structured Cap. 19.03 1.30 4.36 4.03 26.75 68.45 - 94.38 34.99 5.25 63.02 Camera+Depth 20.80 1.57 3.88 4.77 - - 32.15 95.36 30.12 4.82 63.90 + Structured Cap. 21.19 1.49 4.41 4.84 - - 25.37 95.40 30.10 4.96 65.05 Depth+Identities 20.01 - - - 35.24 57.82 23.00 93.15 32.21 4.96 61.21 + Structured Cap. 20.76 - - - 36.25 63.48 24.78 92.50 36.43 5.18 60.81 Camera+Identities+Depth 18.49 2.05 7.74 8.47 35.86 64.25 18.37 92.02 30.09 3.91 60.62 + Structured Cap. 19.52 1.57 7.74 8.20 38.74 64.37 17.41 93.03 32.81 4.99 61.22

Table 6. Quantitative comparison of structured captions when handling compositional conditions. Better results are marked in bold.

Short Cap. Structured Cap.

CLIP-T

CLIP-T

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Image Quality

Image Quality

Motion Smoothness

Motion Smoothness

Aesthetic Quality

Aesthetic Quality

(a) Segmentation

(b) Style

CLIP-T

CLIP-T

| | |
|---|---|
|Motion| |

| |Image|
|---|---|
| | |

e Quality

Image Quality

Motion

M Smoothness

Smoothness

Aesthetic Quality

Aesthetic Quality

(c) Masked Image (d) Sketch

- Figure 6. Quantitative results on unseen conditions (i.e., segmentation [58], style [68], masked image [58], and sketch [58]) when using short and structured captions, respectively.

compare our structured caption approach with a simpler method, where we first caption the input condition (e.g., multiple identity images) and then concatenate that caption with the original short prompt, as shown in Tab. 3. Our results indicate that merely appending the condition’s caption to the short prompt can reduce video smoothness and image quality. One likely reason is that the identity images may contain extraneous details beyond the target subject, potentially conflicting with the original prompt and causing inconsistencies. Consequently, controllability in the final output is compromised. In contrast, our structured caption method accurately identifies the target subject and augments the prompt with relevant information, yielding more controllable video generation (cf. §G.2).

RQ-4: How effective is the training strategy? Next, we investigate the contribution of the training mechanism, and the results are shown in Tab. 4. During training, we employ a two-stage training approach, consisting of alignment learning followed by instruction-tuning. When alignment learning is omitted, and the model proceeds directly

to instruction tuning, both captioning and video generation performance degrade significantly. A possible explanation is that bypassing alignment learning disrupts the encoder’s adaptation process, which has been aligned to the LLM backbone, leading to suboptimal results in subsequent stages. Additionally, we compare the performance of the model without the dropout mechanism. Although removing dropout yields a marked improvement in captioning quality, the benefit to video generation is marginal. This suggests that without dropout, the model may rely on shortcuts from the input captions rather than fully understanding the underlying intent, thereby increasing the risk of overfitting.

- RQ-5: How well does the model perform on compositional conditions? We examine the impact of structured captions under compositional conditions. As shown in Tab. 6, we compare the combined camera, identities, and depth on our customized model and observe that structured captions consistently enhance its performance. Moreover, from the Example 2 and 4 in Fig. 5, our model demonstrates a thorough understanding of the interactions among various conditions, for instance, capturing a woman’s hair color and the position of a mug, accurately guiding the production of videos that align with the specified requirements. This finding further highlights that our approach can automatically equip existing T2V models with the ability to handle compositional conditions without requiring additional training.
- RQ-6: How well is the generalization capability of Any2Caption? Finally, we investigate the model’s generalization ability by evaluating its performance on “unseen” conditions, including style, segmentation, sketch, and masked images. As demonstrated in Fig. 6, the structured captions generated by our model consistently enhance existing T2V frameworks, offering benefits such as increased motion smoothness, aesthetic quality, and more accurate generation control. We attribute these advantages to two primary factors: the strong reasoning capabilities of our MLLM backbone and our training strategy, i.e., progressive mixed training, which leverages existing vision and text instructions for fine-tuning while mitigating knowledge forgetting, thereby ensuring robust generalization.

#### 7. Conclusion

In this work, we focus on addressing the challenge of more accurately interpreting user generation intention from any condition for controllable video generation. We introduce Any2Caption, a framework that decouples multimodal condition interpretation from video synthesis. Built based on an MLLM, Any2Caption converts diverse inputs into dense captions that drive high-quality video generation. We further present Any2CapIns, a large-scale dataset for effective instruction tuning. Experiments show that our method improves controllability and video quality across various backbones.

#### References

- [1] Gpt-4v(ision) system card, 2023. 2, 3, 14
- [2] Kuaishou. kling, 2024. 2
- [3] Sora, 2024. 3, 13
- [4] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In ACL, pages 65–72, 2005. 5
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023. 13
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, pages 22563–22575, 2023. 13
- [7] Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei A. Efros, and Tero Karras. Generating long videos of dynamic scenes. In NeurIPS, 2022. 13
- [8] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jeng-Neng Hwang, Saining Xie, and Christopher D. Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. CoRR, abs/2410.03051, 2024. 5
- [9] Ling-Hao Chen, Shunlin Lu, Ailing Zeng, Hao Zhang, Benyou Wang, Ruimao Zhang, and Lei Zhang. Motionllm: Understanding human behaviors from human motions and videos. CoRR, abs/2405.20340, 2024. 14
- [10] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Lin Bin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. In NeurIPS, 2024. 2, 3, 14
- [11] Yiran Chen, Anyi Rao, Xuekun Jiang, Shishi Xiao, Ruiqing Ma, Zeyu Wang, Hui Xiong, and Bo Dai. Cinepregen: Camera controllable video previsualization via engine-powered diffusion. CoRR, abs/2408.17424, 2024. 3, 13
- [12] Mengyu Chu, You Xie, Jonas Mayer, Laura Leal-Taix´e, and Nils Thuerey. Learning temporal coherence via self-

- supervision for gan-based video generation. ACM Trans. Graph., 39(4):75, 2020. 13
- [13] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen2-audio technical report. CoRR, abs/2407.10759, 2024. 14
- [14] Yufan Deng, Ruida Wang, Yuhao Zhang, Yu-Wing Tai, and Chi-Keung Tang. Dragvideo: Interactive drag-style video editing. In ECCV, pages 183–199, 2024. 3, 13
- [15] Tiehan Fan, Kepan Nan, Rui Xie, Penghao Zhou, Zhenheng Yang, Chaoyou Fu, Xiang Li, Jian Yang, and Ying Tai. Instancecap: Improving text-to-video generation via instanceaware structured caption. CoRR, abs/2412.09283, 2024. 2, 3, 13, 14
- [16] Haopeng Fang, Di Qiu, Binjie Mao, Pengfei Yan, and He Tang. Motioncharacter: Identity-preserving and motion controllable human video generation. CoRR, abs/2411.18281,

2024. 3, 13

- [17] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, Haibin Huang, and Chongyang Ma. I2vadapter: A general image-to-video adapter for diffusion models. In ACM SIGGRAPH, page 112, 2024. 2, 3, 13
- [18] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In ECCV, pages 330–348,

2024. 6

- [19] Shir Gur, Sagie Benaim, and Lior Wolf. Hierarchical patch VAE-GAN: generating diverse videos from a single sample. In NeurIPS, 2020. 13
- [20] Niv Haim, Ben Feinstein, Niv Granot, Assaf Shocher, Shai Bagon, Tali Dekel, and Michal Irani. Diverse generation from a single video made possible. In ECCV, pages 491– 509, 2022. 13
- [21] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. CoRR, abs/2404.02101, 2024. 2, 3, 4, 6, 7, 13
- [22] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, Man Zhou, and Jie Zhang. Idanimator: Zero-shot identity-preserving human video generation. CoRR, abs/2404.15275, 2024. 2
- [23] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In EMNLP, pages 7514–7528,

2021. 5

- [24] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2023. 2, 3, 13
- [25] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. CoRR, abs/2501.04698, 2025. 7
- [26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin

- Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024. 6
- [27] Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and Gedas Bertasius. Video recap: Recursive captioning of hour-long videos. In CVPR, pages 18198–18208, 2024. 3, 13
- [28] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. In NeurIPS, 2023. 14
- [29] Junpeng Jiang, Gangyi Hong, Lijun Zhou, Enhui Ma, Hengtong Hu, Xia Zhou, Jie Xiang, Fan Liu, Kaicheng Yu, Haiyang Sun, Kun Zhan, Peng Jia, and Miao Zhang. Dive: Dit-based video generation with enhanced control. CoRR, abs/2409.01595, 2024. 2
- [30] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. In NeurIPS, 2024. 2, 3, 4, 6, 14
- [31] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Multi-task video generative foundation model with full attention. arXiv preprint arXiv:2503.19907, 2025. 18
- [32] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M. Rehg, and Pinar Yanardag. RAVE: randomized noise shuffling for fast and consistent video editing with diffusion models. In CVPR, pages 6507–6516, 2024. 3, 13
- [33] Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman. Dreampose: Fashion image-to-video synthesis via stable diffusion. In ICCV, pages 22623–22633, 2023. 2, 3, 13
- [34] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Daquan Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models. CoRR, abs/2412.03603,

2024. 3, 7, 13

- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICM, pages 19730–19742, 2023. 14
- [36] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. CoRR, abs/2305.06355,

2023. 5

- [37] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In ECCV, pages 323–340, 2024. 14
- [38] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya

- Jia. Mini-gemini: Mining the potential of multi-modality vision language models. CoRR, abs/2403.18814, 2024. 14
- [39] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, pages 5971–5984, 2024. 3, 4, 13, 14
- [40] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, 2004. 5
- [41] Han Lin, Jaemin Cho, Abhay Zala, and Mohit Bansal. Ctrl-adapter: An efficient and versatile framework for adapting diverse controls to any diffusion model. CoRR, abs/2404.09967, 2024. 2, 3, 6, 7, 13
- [42] Gongye Liu, Menghan Xia, Yong Zhang, Haoxin Chen, Jinbo Xing, Yibo Wang, Xintao Wang, Ying Shan, and Yujiu Yang. Stylecrafter: Taming artistic video diffusion with reference-augmented adapter learning. ACM Trans. Graph., 43(6):251:1–251:10, 2024. 2
- [43] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26286–26296, 2024. 3, 4, 5, 13, 14
- [44] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. CoRR, abs/2401.03048, 2024. 2
- [45] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In AAAI, pages 4117–4125, 2024. 2, 3, 6, 7, 13
- [46] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic-me: Identity-specific video customized diffusion. CoRR, abs/2402.09368, 2024. 2
- [47] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. CoRR, abs/2407.02371, 2024. 3, 13
- [48] Kishore Papineni, Salim Roukos, Todd Ward, and Wei jing Zhu. Bleu: a method for automatic evaluation of machine translation. pages 311–318, 2002. 5
- [49] Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, and Nong Sang. Hierarchical spatio-temporal decoupling for text-to- video generation. In CVPR, pages 6635–6645, 2024. 13
- [50] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chlo´e Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross B. Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. CoRR, abs/2408.00714, 2024. 4
- [51] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023. 6
- [52] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual,

- Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2023. 2, 3, 13
- [53] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. CoRR, abs/2411.04928, 2024. 3, 13
- [54] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 5
- [55] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, Xichen Pan, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024. 14
- [56] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. CoRR,

- abs/2409.12191, 2024. 3, 6, 13

[57] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, Fei Yang, Pengfei Wan, and Di Zhang. Koala36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. CoRR,

- abs/2410.08260, 2024. 4

- [58] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. In NeurIPS, 2023. 2, 3, 8, 13
- [59] Yaohui Wang, Piotr Bilinski, Franc¸ois Br´emond, and Antitza Dantcheva. Imaginator: Conditional spatio-temporal GAN for video generation. In WACV, pages 1149–1158, 2020. 13
- [60] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, page 114, 2024. 7
- [61] Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Zhe Chen, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Ping Luo, Yu Qiao, and Jifeng Dai. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. In NeurIPS,

2024. 14

- [62] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, pages 7589–7599, 2023. 2, 3, 13
- [63] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal LLM. In ICML. OpenReview.net, 2024. 4, 14
- [64] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng

- Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. CoRR, abs/2407.10671, 2024. 14
- [65] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything V2. In NeurIPS, 2024. 4
- [66] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In ICCV, pages 4212–4222, 2023. 4
- [67] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer, 2024. 2, 3, 7, 13
- [68] Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, and Wenhan Luo. Stylemaster: Stylize your video with artistic generation and translation. CoRR, abs/2412.07744, 2024. 2, 6, 8
- [69] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. CoRR, abs/2411.17440, 2024. 2
- [70] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. CoRR, abs/2311.04145,

2023. 3, 13

- [71] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with BERT. In ICLR, 2020. 5
- [72] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, and Shuicheng Yan. Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding. In NeurIPS, 2024. 14
- [73] Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, and Kai Chen. PIA: your personalized image animator via plug-and-play modules in text-to-image models. In CVPR, pages 7747–7756, 2024. 13
- [74] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. CoRR, abs/2305.17098, 2023. 2, 3, 7, 13
- [75] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. Cami2v: Camera-controlled image-to-video diffusion model. CoRR, abs/2410.15957, 2024. 2, 3, 13
- [76] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang

- You. Open-sora: Democratizing efficient video production for all. CoRR, abs/2412.20404, 2024. 2
- [77] Yong Zhong, Min Zhao, Zebin You, Xiaofeng Yu, Changwang Zhang, and Chongxuan Li. Posecrafter: One-shot personalized video synthesis following flexible pose control. In ECCV, pages 243–260, 2024. 3, 13
- [78] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Trans. Graph., 37(4): 65, 2018. 6

#### Appendix Overview

The appendix presents more details and additional results not included in the main paper due to page limitations. The list of items included is:

- • Limitation in Section §A.
- • Ethic Statement results in Section §B.
- • Extended Related Work in Section §C.
- • Extended Dataset Construction Details in Section §D.
- • More Statistics Information of IRSCORE in Section §E.
- • Detailed Setups in Section §F.
- • Extended Experiment Results and Analyses in Section §G.

#### A. Limitation

Despite the advancement of our proposed framework, several limitations may remain:

Firstly, the diversity of annotated data is constrained by the capabilities of the current annotation tools, which may limit the variety of generated content. Moreover, the scarcity of real-world data introduces potential domain gaps, reducing the model’s generalizability in practical scenarios.

Secondly, due to inherent model limitations, hallucinations may occur, resulting in inaccurate structured captions and consequently degrading the quality of generated videos.

- A possible direction to mitigate this issue is to develop an end-to-end approach that jointly interprets complex conditions and handles video generation.

Lastly, the additional condition-understanding modules inevitably increase inference time. However, our empirical results suggest that the performance gains from these modules are substantial, and future work may explore more efficient architectures or optimization techniques to balance speed and accuracy.

- B. Ethic Statement

This work relies on publicly available datasets and manually constructed datasets, ensuring that all data collection and usage adhere to established privacy standards. We recognize that automatic annotation processes may introduce biases, and we have taken measures to evaluate and mitigate these biases. Nonetheless, we remain committed to ongoing improvements in this area.

By enhancing video generation capabilities, Any2Caption could inadvertently facilitate negative societal impacts, such as the production of deepfakes and misinformation, breaches of privacy, or the creation of harmful content. We, therefore, emphasize the importance of strict ethical guidelines, robust privacy safeguards, and careful dataset curation to minimize these risks and promote responsible research practices.

#### C. Extended Related Work

##### C.1. Text-to-Video Generation

The development of video generation models has progressed from early GAN- and VAE-based approaches [7, 12, 19, 20, 59] to the increasingly popular diffusionbased methods [5, 6, 49, 73]. Among these, diffusionin-transformer (DiT) architectures, such as OpenAI’s Sora [3] and HunyuanVideo [34], have demonstrated remarkable performance, producing photorealistic videos over extended durations. Controllable video generation [11, 16, 21, 53] has become an essential aspect of this field. Initially, research efforts centered predominantly on text-to-video generation [24, 52], which remains the most common approach. However, relying solely on text prompts can be insufficient for accurately capturing user intent, prompting exploration into other conditioning inputs such as static images [17, 62], user sketches [58, 74], human poses [33, 45, 77], camera perspectives [21, 75], and even additional videos [14, 32, 70]. Given this diversity of potential conditions, unifying them into a single “any-condition” video generation framework is highly valuable.

##### C.2. Controllable Video Generation

Recent methods like VideoComposer [58], Ctrl-Adapter [41], and ControlVideo [74] have investigated anycondition video generation. Nevertheless, they still struggle with comprehensive controllability due to the complexity of multiple modalities and the limited capacity of standard diffusion or DiT encoders to interpret them. Inspired by the strong multimodal reasoning capabilities of modern MLLMs [39, 43, 56], we propose leveraging an MLLM to consolidate all possible conditions into a unified reasoning process, producing structured dense captions as inputs to a backbone Video DiT. SoTA DiT models already exhibit the capacity to interpret dense textual descriptions, as long as the input captions are sufficiently detailed in depicting both the scene and the intended generation goals. Building on this, our MLLM-based condition encoder directly addresses the comprehension bottleneck, theoretically enabling higher-quality video generation. To our knowledge, this work is the first to develop an MLLM specifically tailored for any-condition video generation. Because the caption-generation mechanism is decoupled from DiT, our proposed Any2Caption can be integrated into existing DiT-based methods without additional retraining.

##### C.3. Video Captioning

Our approach is closely related to video recaptioning research, as our MLLM must produce dense captions based on the given conditions. In text-to-video settings, prior work [15, 27, 47, 67] has demonstrated the benefits of recaptioning videos to obtain more detailed textual annota-

tions, thereby improving the training of longer and higherquality video generation via DiT. ShareGPT4Video [10], for example, employs GPT-4V [1] to reinterpret video content and produce richer captions. MiraData [30] introduces structured dense recaptioning, while InstanceCap [15] focuses on instance-consistent dense recaptioning. Although we also pursue structured dense captions to enhance generation quality, our method diverges fundamentally from these previous approaches. First, because DiT models are already sufficiently powerful, we directly adopt an off-theshelf Video DiT without incurring the substantial cost of training it with dense captions. Instead, we train an MLLM as an any-condition encoder at a comparatively lower cost; in the text-to-video scenario, for instance, we only need to train on pairs of short and dense captions, which are far easier and more abundant to obtain. Second, prior methods that recapturing the entire video risk introducing noise or even hallucinated content due to the current limitations of MLLMs in video understanding, potentially undermining DiT training quality, whereas our framework avoids this issue. Most importantly, while these approaches may rely on dense-caption-trained DiT models, real-world inference often involves very concise user prompts, creating a mismatch that can diminish final generation quality.

##### C.4. Multimodal Large Language Models

Recent advances in Large Language Models (LLMs) [64] have catalyzed a surge of interest in extending their capabilities to multimodal domains [37–39, 61, 72]. A number of works integrate a vision encoder (e.g., CLIP [43], DINOv2[55], OpenCLIP[38]) with an LLM, often through a lightweight “connector” module (e.g., MLP [43], Qformer [35]), enabling the model to process image and video inputs with minimal additional training data and parameters [39, 43, 63]. These approaches have demonstrated promising performance on tasks such as image captioning, visual question answering, and video understanding. Beyond purely visual data, some researchers have investigated broader modalities, such as 3D motion [9, 28] or audio [13, 63], thereby expanding the application range of multimodal LLMs. Despite these advances, most existing MLLMs focus on a limited set of visual modalities and do not readily accommodate more specialized inputs like human pose or camera motion. This gap restricts their ability to handle more diverse and complex conditions in the controllable video generation field. In contrast, our work targets a broader spectrum of modalities, aiming to build a single model capable of interpreting and unifying image, video, human pose, and camera conditions. Specifically, we augment an existing MLLM with dedicated encoders for motion and camera features, thereby equipping the model to process arbitrary multimodal conditions and facilitate controllable video generation.

#### D. Extended Dataset Details

Visualization of Short and Structured Caption. Here, we visualize the short and structured caption in Fig. 7, 8. Notably, the structured caption captures the video content in greater detail and across multiple aspects. In contrast, the short caption deliberately omits any information already supplied by a non-text condition—for example, camera movement is excluded in Fig. 7’s short caption but included in Fig. 8 because it is not specified by the provided multiple-identity images. Moreover, we visualize the word distribution of the structured captions in Any2CapIns in Fig. 9.

Prompt Visualization for Short Caption Construction. In Tab. 7 and 8, we show the system prompts used by GPT4V to generate short captions. The prompt explicitly instructs GPT-4V to consider the given conditions comprehensively and produce short prompts that focus on information not covered by the non-textual inputs. For instance, when multiple identities are specified, the short prompt should avoid repeating their appearance attributes and instead highlight interactions among the identities. Conversely, when depth is the input condition, the short prompt should include more detailed appearance-related information.

#### E. More Statistics Information of IRSCORE

We generate a total of 15,378 question-answer (QA) pairs, averaging 19.2 pairs per structured caption. Fig. 10 presents the distribution of constructed questions across different aspects in the structured caption, and Tab. 9 shows representative QA pairs for each aspect. Notably, questions under the main object category emphasize fine-grained details such as clothing color or hairstyles, while action questions focus on object interactions and movements. This level of specificity allows us to rigorously assess whether the generated captions are both complete and precise.

#### F. Detailed Setups

##### F.1. Detailed Testing Dataset

Here, we present the statistics of the test in Tab. 10, which covers four types of single conditions (e.g., Depth, Camera, Identities, and Human pose), and four types of compositional conditions(e.g., Camera+Identities, Camera+Depth, Identities+Depth and Camera+Identities+Depth). Each category contains 200 instances.

##### F.2. Detailed Training Procedure

We employ a two-stage training process to enhance the alignment and interpretability of multimodal conditions in Any2Caption.

Diffusion Models Diffusion Models

Short Caption + Camera Video

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

 Short Caption: A cozy and well-lit home. Start by showing a dining table with chairs and a chandelier, then capture the living room with a sofa and fireplace. Move towards the large windows to reveal the deck outside and the grassy area. Emphasize the warm and inviting atmosphere.

######  Structured Caption:

- 1. Dense caption: An elegantly composed interior scene unfolds, where the dining and living areas are interwoven through warm tones and architectural cohesion. The narrative begins with a glimpse of a chandelier-lit dining table, then flows organically toward a plush seating arrangement and a glowing fireplace. A slow and smooth camera shift unveils tall glass windows that eventually open up to a serene deck, gracefully bridging interior comfort with exterior peace.
- 2. Main object caption: Key interior elements include a traditional dining ensemble centered under a sparkling chandelier, and a living room anchored by a soft sofa and an old-fashioned fireplace—framed by panoramic glass windows that serve as portals to nature.
- 3. Background caption: The setting is enriched by a timeless ambiance: gentle upholstery, a vintage-style wall clock, and natural daylight pouring through large windows. Outside, a sunlit wooden patio bordered by greenery extends the space beyond the glass.
- 4. Camera caption: Captured with a steady cam at eye-level. Begins with a lateral sweep from the dining corner before proceeding with a slow dolly movement toward the windowed wall, culminating in a reveal of the outdoor patio.
- 5. Style caption: Cozy and classic, with a warm color palette, soft lighting gradients, and a slightly rustic design aesthetic. The stylistic cohesion evokes a calm and welcoming atmosphere.
- 6. Action caption: Executing a deliberate leftward pan, the camera surveys the interior before advancing forward, transitioning the viewer’s perspective from indoor warmth to the outdoor environment.

Diffusion Models Diffusion Models

Figure 7. Illustrations of constructed short and structured captions under the camera-to-video generation.

Short Caption + Multi-IDs Video

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

 Short Caption: A person stands in a vast field under a stormy sky. They raise their hands to their face, then above their head, transforming their headpiece into a horn-like structure. The camera moves backward, capturing the gloomy atmosphere and their confident stance.

######  Structured Caption:

- 1. Dense caption: Amid a thunderous, cloud-laden sky, a lone figure clad in futuristic, obsidian-colored armor stands motionless at the center of an expansive, empty field. The moment intensifies as the individual lifts their arms toward their face in a calculated motion. Gradually, their helmet morphs into an imposing, horned structure. As this metamorphosis completes, the figure returns to a commanding stance, now equipped with two elegant, blade-like weapons—one held firmly in each hand.
- 2. Main object caption: The armored character—fully suited in sleek, dark tech-gear—occupies the center of the frame. Their headgear shifts shape, evolving into horn-like extensions as they raise their hands in a slow, deliberate movement.
- 3. Background caption: Sweeping grasslands stretch beneath a foreboding sky, with rolling hills far in the distance. The atmosphere is heavy with tension, and the setting remains still, emphasizing the transformation's magnitude and isolating the character in a vast, ominous world.
- 4. Camera caption: Camera begins with a mid-range, centered composition. As the sequence unfolds, it subtly dollies backward to reveal more of the character’s form while maintaining a consistent eye-level perspective.
- 5. Style caption: Dramatic and futuristic, with a moody color palette dominated by greys and blacks. The visual tone draws from dystopian cinema, focusing on solitude, metamorphosis, and subtle power.
- 6. Action caption: A solitary warrior lifts her arms toward her helmet, triggering its transformation. Once reformed into a horned shape, she lowers her arms, retrieves dual blades, and assumes a poised, forward-facing stance.

Figure 8. Illustrations of constructed short and structured captions under the multiIDs-to-video generation.

[Figure 206]

Diffusion Models Diffusion Models

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

(1) Dense Caption (2) Main Object Caption (3) Background Caption

(4) Action Caption (5) Style Caption (6) Camera Caption

Figure 9. Word cloud of different structured captions in Any2CapIns dataset, showing the diversity.

Multi-IDs Here is the scenario: We have an MLLM model that supports a text image-conditioned intrinsic-video-caption generation task. The system

input consists of:

- 1. A reference image composed of 2-3 horizontally stitched images (provided by the user), each stitched image containing one or several

target objects for reference); and

- 2. A concise textual prompt (referred to as text B, the user’s instruction).

The model’s output is a detailed descriptive caption (**text A**) that thoroughly describes the video corresponding to the user’s input prompt (**text B**) in great detail. Your task is to perform a reverse engineering. Based on the given reference image (the target objects) and the detailed target video caption (text A), you need to generate a **reasonable and concise user prompt (text B)** through your understanding, analysis, and imagination. You must adhere to the following rules:

- 1. Text A is a dense caption of a video, including all the key objects, their attributes, relationships, background, camera movements, style,

and more. Carefully analyze this caption for all relevant details.

- 2. Analyze the provided reference images in detail to identify the differences or missing details compared to the target video description.

These may include environment details, the interaction between objects, the progression of actions, camera movements, style, or any elements not covered by the reference image. Based on these analyses, generate the user’s instructions.

- 3. The user’s prompt must include the following aspects: first, an overall description of where the target objects are and what they are doing,

along with the temporal progression of their actions. Then, it should describe the background, style, and camera movements.

- 4. If the target video introduces new objects not present in the reference images, the user’s prompt should describe the attributes of the new

target objects and their interactions with the other target objects.

- 5. If the video’s style differs from the reference, briefly describe the style in a few words.
- 6. When the background needs to be described, include details about people, settings, and styles present in the background.
- 7. Avoid repeating information that can be inferred from the reference images, and eliminate redundant descriptions in the user prompt.
- 8. The user prompt (text B) must be written in simple wording, maintaining a concise style with short sentences.
- 9. The user’s instructions should vary in expression; For example, prompts do not always need to start with the main subject. They can begin

with environmental details, camera movements, or other contextual aspects. Here are three examples representing the desired pattern:

==================================================================================================== [In-context Examples]

==================================================================================================== [Input]

Table 7. Demonstration of the prompt used for GPT-4V to generate the short prompt when the input condition is the multi-IDs.

Stage-1: Alignment learning. This stage focuses on aligning features extracted by the camera encoder with the LLM feature space. To achieve this, we first extract camera movement descriptions (e.g., fixed, backward, pan to the right) from the camera captions in Any2CapIns to construct a camera movement description dataset. We then introduce two specialized tokens, <|camera start|> and <|camera end|>, at the beginning and end of the camera feature embeddings. During training, only the camera encoder is optimized, while all other parameters in Any2Caption remain frozen. Similarly, for motion align-

ment, we construct a motion description dataset by extracting action descriptions (e.g., walking, dancing, holding) from the action captions in Any2CapIns. We then freeze all model parameters except those in the motion encoder to ensure the LLM effectively understands motion-related conditions.

Stage-2: Condition-Interpreting Learning. After alignment learning, we initialize Any2Caption with the pretrained Qwen2-VL, motion encoder, and camera encoder

Depth Here is the scenario: We have an MLLM model that supports a text & image-conditioned intrinsic-video-caption generation task. The system

input consists of:

- 1. A reference image composed of 3-5 horizontally stitched depth maps in temporal sequence (provided by the user, each map containing

depth information for reference); and

- 2. A concise textual prompt (referred to as text B, the user’s instruction).

The model’s output is a detailed descriptive caption (text A) that thoroughly describes the video corresponding to the user’s input prompt (text B) in great detail. Now, I need you to perform a reverse engineering task. Based on the given reference image (the depths) and the detailed target video caption (text A), you must generate a reasonable and concise user prompt (text B) through your understanding, analysis, and imagination. To ensure accurate and effective outputs, follow these rules strictly:

- 1. Text A is a dense caption of a video, including all the key objects, their attributes, relationships, background, camera movements, style,

and more. Carefully analyze this caption to extract the necessary details.

- 2. Since the depth information already provides the necessary geometric outlines and layout details. Do not repeat this information in the

user prompt. Instead, focus on the aspects not covered by the depth maps.

- 3. The user’s instruction should highlight details not included in the depth map, such as environmental details, the appearance of the

subjects, interactions between subjects, the progression of actions, relationships between the subjects and the environment, camera movements, and overall style.

- 4. For dense depth maps (more than 5 maps), assume the maps provide the camera movements and actions between objects, focusing on

describing the appearance of the subjects and environment, the atmosphere, and subtle interactions between subjects and their environment.

- 5. For sparse depth maps (5 maps or fewer), assume the maps only provide scene outlines. Emphasize details about the subjects’ appearance,

environment, interactions between subjects, relationships between subjects and the environment, and camera movements.

- 6. The user prompt (text B) must be written in simple wording, maintaining a concise style with short sentences, with a total word count not

exceeding 100.

- 7. Your output should be a continuous series of sentences, not a list or bullet points.
- 8. The user’s instructions should vary in expression; they don’t always need to begin with a description of the main subject. They could also

start with environmental details or camera movements. Here are three examples representing the desired pattern:

==================================================================================================== [In-context Examples]

==================================================================================================== [Input]

Table 8. Demonstration of the prompt used for GPT-4V to generate the short prompt when the input condition is the depth.

Style Caption: 15.79%

Main Object Caption: 36.84%

Camera Caption:

10.53%

Action Caption:

19.3%

Background Caption: 17.54%

Figure 10. QA pairs proportion in structured captions.

weights. We then employ a progressive mixed training strategy, updating only the lm head while keeping the multimodal encoders frozen. The training follows a sequential order based on condition complexity: identities ⇒ human pose ⇒ camera ⇒ depth. Correspondingly, the integration ratio of additional vision/text instruction datasets is progressively increased, set at 0.0, 0.4, 0.6, and 0.8, ensuring a balanced learning process between condition-specific specialization and generalization.

##### F.3. Detailed Implementations

In Tab. 11, we list the detailed hyperparameter settings in two stages. All the training is conducted on 8×80G GPUs.

#### G. Extended Experiment Results and Analyses

##### G.1. The Capability for Understanding Complex Instruction.

We further examine Any2Caption’s ability to handle complex user instructions, particularly regarding whether it accurately captures the user’s intended generation targets. From Fig. 11, we observe that the model focuses precisely on the user-specified main objects, such as a “woman warrior” or a background “filled with chaos and destruction”—when producing structured captions. In contrast, a short caption combined with condition captions often includes extraneous objects or background details present in the identity images, which distract from the user’s intended targets in the final video generation.

Additionally, we assess the model’s performance on instructions containing implicit objects or actions, as shown in Fig. 12 and 13. In these examples, the model correctly interprets phrases like “the most right person” as “a young Black woman with long, curly brown hair, wearing a black and white outfit” and similarly associates implicitly specified objects with the provided conditions, generating structured captions that align with the user’s goals.

Lastly, Fig. 16 compares videos generated using different captions. The results indicate that structured captions significantly improve both the smoothness of motion and the overall consistency of the generated videos.

Aspect QA Pairs Main Object What is the young woman adjusting as she walks down the corridor? Her wide-brimmed hat.

What color is the young woman’s T-shirt? Light blue. How does the young woman feel as she walks down the corridor? Happy and carefree. What is the young woman wearing? Light blue t-shirt with pink lettering, blue jeans, and a wide-brimmed hat. What is the young woman’s hair length? Long. What is the position of the young woman in the frame? In the center of the frame. What is the main object in the video? A large shark. What is the color of the underwater scene? Blue. What are the two scientists wearing? White lab coats and gloves. What is the first scientist using? A microscope.

Background Where is the young woman walking? Down a corridor. What time of day does the scene appear to be set? Daytime. What can be seen in the background of the corridor? Beige walls and large windows. What is the weather like in the video? Clear. Where is the shark located? On the ocean floor. What surrounds the shark in the video? Smaller fish. Where is the laboratory setting? In a brightly lit environment with shelves filled with bottles. What detail does the background highlight? The scientific setting with static emphasis.

Camera How does the camera follow the young woman? Moving backward What is the camera’s height relative to the person? Roughly the same height as the person. What shot type does the camera maintain? Medium close-up shot of the upper body. How does the camera position itself to capture the subject? At a higher angle, shooting downward. How does the camera capture the environment? From a medium distance. How is the camera positioned? At approximately the same eye level as the subjects, maintaining a close-up shot. How does the camera move in the video? It pans to the right.

Style What is the style of the video? Casual and candid. What kind of design does the corridor have? Modern and clean design. What style does the video portray? Naturalistic style with clear, vivid visuals. What does the video style emphasize? Clinical, high-tech, and scientific precision. What is the color theme of the lighting? Bright and cool. What kind of atmosphere does the laboratory have? Professional and scientific.

Action What does the young woman do with both hands occasionally? Adjusts her hat. What is the young woman doing as she moves? Walking forward with her hands on her hat. What is the main action of the shark in the video? Lying motionless. What is the movement of the fish like? Calm and occasionally darting. What is the movement of the first scientist at the beginning? Examines a microscope. What task is the second scientist engaged in? Handling a pipette and a beaker filled with green liquid. How does the second scientist transfer the liquid? Carefully using a pipette into the beaker. Are there any noticeable movements in the background? Occasional small particles floating.

Table 9. Demonstration of generated question-answer pairs utilized in IRSCORE calculation.

Short Cap. #Structured Cap. #Avg. Len. #Avg. Len.

Type #Inst.#Condi.

Identities 200 350 65.28 284.97 Camera 200 200 50.25 208.01 Depth 200 200 54.22 225.09 Human Pose 200 200 58.38 259.03 Camera+Identities 200 622 53.41 209.17 Camera+Depth 200 400 51.43 208.81 Identities+Depth 200 555 53.14 286.83 Camera+Identities+Depth 200 756 58.35 289.21

Table 10. Statistics of the constructed test datasets. #Inst. denotes the number of instances, and #Condi. indicates the number of unique conditions. Short Cap. #Avg. Len represents the average caption length of short captions, and Structured Cap. #Avg. Len. represents the average caption length of structured captions.

camera-controlled video generation results, and Fig. 15 illustrates depth-controlled outcomes. We observe that structured captions improve image quality and motion smoothness by providing richer scene details.

For multi-identity-controlled video generation, as depicted in Fig. 16, 17, and 18, structured captions lead to more expressive and realistic representations of the referenced identities, with more accurate colors and smoother actions.

##### G.2. More Video Visualization

Here, we compare the results of different customized models [31] after integrating structured captions. Fig. 14 shows

Stage-1: Alignment Learning Stage-2: Condition-Interpreting Learning

Configuration

Camera Motion Identities Human pose Camera Depth

Optimizer AdamW AdamW AdamW AdamW AdamW AdamW Precision bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 Learning Rate 5e5 5e5 5e5 5e5 5e5 1e5 Weight Decay 0.01 0.01 0.01 0.01 0.01 0.01 Joint Train Ratio 0.0 0.0 0.0 0.4 0.6 0.8 Betas (0.9, 0.99) (0.9, 0.99) (0.9, 0.99) (0.9, 0.99) (0.9, 0.99) Dropout Possibility 0.0 0.0 0.4 0.6 0.6 0.6 Dropout (Short Cap.) 0.0 0.0 0.6 0.6 0.6 0.6 Batch Size Per GPU 4 4 4 4 4 4

Depth LLaVA-150K Alpaca-50K

Camera Movement Description Dataset (Manual)

Action Description Dataset (Manual)

Human Pose LLaVA-150K

Camera LLaVA-150K

Training Data

MultiIDs

Table 11. Training recipes for Any2Caption.

Diffusion Models Diffusion Models

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

###### Input Multiple Identities Gold Video

 Input Short Caption: A woman warrior in armor wields a shield and sword, fighting a soldier in a chaotic, battle-torn room.

######  Predicted Structured Caption:

- 1. Dense caption: Set in a war-torn interior littered with fractured machinery and scattered debris, a fierce confrontation unfolds between a battle-hardened woman and an armored soldier. Standing amid the wreckage, she brandishes a polished sword and a finely adorned shield, moving with the precision of a trained combatant. The surrounding space, filled with remnants of violence, amplifies the urgency and focus of the encounter. Despite the destruction, both fighters hold their ground, locked in a tense standoff.
- 2. Main object caption: A female warrior clad in segmented armor combining dark leather and polished metal stands prominently. She wields a sword in her right hand and a shield in her left—its ornate design featuring an embossed emblem at its center. Her posture is alert and deliberate, facing off against a cape-wearing soldier lurking just behind her in full tactical gear.
- 3. Background caption: The room is a battlefield—walls partially collapsed, wires dangling, shattered glass on the floor. Military-grade vehicles and fragmented gear lie overturned, and thick shadows stretch across the space, lit only by scattered sparks and faint ambient lighting from damaged fixtures.
- 4. Camera caption: The scene is captured from a static low-angle position, slightly tilted upward to magnify the warrior's authority. Framing is tight enough to highlight detail in the characters’ armor but wide enough to expose the chaotic environment surrounding them.
- 5. Style caption: Gritty and immersive, the visual aesthetic leans toward post-apocalyptic realism, employing low-key lighting and subdued tones to emphasize both the destruction and the emotional weight of the scene.
- 6. Action caption: With sharp, calculated motions, the woman shifts between offense and defense—raising her shield to deflect and stepping forward to strike with her blade. Her movements convey strength, control, and combat experience.

 Short Caption w/ Condition Caption: A woman warrior in armor wields a shield and sword, fighting a soldier in a chaotic, battle-torn room.

- ID Image 1: A behind-the-scenes moment from a superhero film set, featuring Wonder Woman, Batman, and Superman in costume, with a film crew and green screen in the background.
- ID Image 2: A round shield with intricate designs and a central starburst pattern.

Introduce noise information

- Figure 11. Illustrations of input short caption, predicted structured caption, and short caption combined with condition captions. The condition captions may introduce noisy information due to the focus distraction of the background of input identities, for example, the Batman, Superman, film crew.

Diffusion Models Diffusion Models

[Figure 217]

[Figure 218]

 Input Short Caption: The most right person in the picture 1 is dancing on the desk in picture 2.

Implicit objects for generation

Input Multiple Identities

######  Predicted Structured Caption:

- 1. Dense caption: Inside a sunlit modern kitchen, a young woman confidently dances atop a wooden dining table, her movements vibrant and self-assured. Clad in a monochrome outfit that blends fashion with flair, she glides, twirls, and gestures to an internal rhythm. Light pours in from generous windows, casting soft highlights on the table’s surface—where everyday elements like fruit and a meal serve as subtle reminders of domestic life. The warm tones of the wood and the greenery visible outside enhance the feeling of life, spontaneity, and celebration.
- 2. Main object caption: A confident Black woman with voluminous, curly brown hair performs expressive dance movements on a dining table. She wears a black-and-white color-block jacket, slim black pants, and ankle boots. Every motion of her body radiates joy, from the sway of her arms to the rhythm in her steps, all anchored at the center of the scene.
- 3. Background captionThe room is bright and contemporary—featuring polished wooden cabinets, clean surfaces, and a dining table set casually with fruit and a dish. Beyond the expansive windows lies a view of lush foliage, infusing the interior with calm, natural energy. Nothing in the setting feels staged—it’s a real kitchen full of warmth and life.
- 4. Camera caption: The scene is observed from a slightly elevated camera angle, keeping the subject fully in frame while subtly emphasizing the table’s height and the dancer’s grounded energy.
- 5. Style caption: Effortless and candid. The style emphasizes natural light, domestic textures, and a relaxed color palette, evoking authenticity and comfort with a hint of artistic spontaneity.
- 6. Action caption: She steps, spins, and gestures with increasing enthusiasm, transforming the tabletop into her stage. Her expressive movements seem improvised yet intentional, turning an everyday moment into a spontaneous performance.

- Figure 12. Illustrations of predicted structured captions based on the input multiple identities and the short instruction that expresses the implicit objects for the videoDiffusiongeneration.Models Diffusion Models

Input Multiple Identities Input Human Pose Input Short Caption: The right person in picture 1 is following the human pose movement in the video while the other person is watching beside them.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Implicit objects and action for generation

 Predicted Structured Caption:

- 1. Dense caption: In a sun-drenched home interior, a young woman executes a series of delicate and measured dance-like motions. With arms raised and gestures flowing seamlessly from one to the next, her performance is composed and deliberate, harmonizing with the serene domestic surroundings. Her movement unfolds within a modest space, where light filters softly through sheer curtains, accentuating the gentle rhythm of the moment.
- 2. Main object caption: The focus is a light-skinned woman in her early twenties, wearing a pastel blue shirt and fitted dark jeans. Her long brown hair sways gently as she smiles and moves with rhythmic grace. Just off to the side, a man dressed in a black coat stands quietly, watching with a neutral expression and relaxed posture, offering a quiet counterpoint to her energy.
- 3. Background caption: Set in a cozy, well-maintained living room with soft white walls and a large window draped in translucent curtains. Minimal yet thoughtful decor—like a clean-lined sofa, a floor plant, and a small lamp-lit side table—evoke warmth and simplicity. Bright daylight streams in, creating a welcoming and peaceful domestic space.
- 4. Camera caption: Stationary low-angle setup with the lens tilted slightly upward, capturing the dancer’s full body in frame. The perspective subtly magnifies the performer’s presence and motion while anchoring her firmly in the surrounding domestic setting.
- 5. Style caption: Simple and unfiltered, with a naturalistic tone. The visual style mimics an everyday candid recording, emphasizing authenticity over polish, with even lighting and muted colors contributing to the understated charm.
- 6. Action caption: The woman moves with gentle enthusiasm—stretching, pivoting, and sweeping her arms in smooth arcs. Her expressive dance contrasts with the stillness of the man nearby, who watches calmly, contributing to the quiet intimacy of the scene.

- Figure 13. Illustrations of predicted structured captions based on the input multiple identities and the short instruction that expresses the implicit objects and the action for the target video generation.

Diffusion Models Diffusion Models Ø Short Caption: A serene video of a large house with a red roof and a spacious porch, surrounded by lush greenery. A peaceful countryside setting with vibrant colors and a tranquil atmosphere.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Short Caption + Camera → Video

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

- Example - 1
- Example - 2

Ø Short Caption: The scene is bathed in bright sunlight, emphasizing the warm and inviting atmosphere. A modern house with large windows

and a balcony is showcased. Potted plants accent the architectural details. Distant mountains frame the view. Lush greenery surrounds the scene. The sky is a clear blue, dotted with scattered clouds. A dreamy lens flare effect adds to the serene quality. The overall ambiance is tranquil and picturesque.

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Structured Caption + Camera → Video

Short Caption + Camera → Video

- Example - 3

Structured Caption + Camera → Video

Ø Short Caption: A well-lit dining and living room with elegant and classic decor. The dining table is surrounded by chairs and has a chandelier above it. There's a wooden cabinet against the wall. The background features a hallway with a staircase and another dining area visible. The decor includes wooden furniture and framed pictures on the walls.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Short Caption + Camera → Video

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Structured Caption + Camera → Video

- Figure 14. Illustrations of predicted structured captions based on the input multiple identities and the short instruction that expresses the implicit objects and the action for the target video generation.

Diffusion Models Diffusion Models Ø Short Caption: A dark, static background enhances the brightly colored, rotating spiral of small blocks. The camera remains fixed,

capturing the mesmerizing effect of the colors shifting subtly. The dynamic movement creates a hypnotic, abstract visual style.

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

- Example-4
- Example-5

Short Caption + Depth → Video

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Structured Caption + Depth → Video

Ø Short Caption: The park is sunny and lush with green trees and a small pond. A young couple in their late twenties embraces and shares a kiss. The woman, in a white sleeveless wedding dress and holding a bouquet, playfully touches the man's face. He is dressed in a black suit. The scene is romantic and intimate, with soft, natural lighting. The camera pans gently, capturing their affectionate interaction and the serene environment.

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Short Caption + Depth → Video

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

Structured Caption + Depth → Video

- Figure 15. Illustrations of predicted structured captions based on the input multiple identities and the short instruction that expresses the implicit objects and the action for the target video generation.

Diffusion Models Diffusion Models

Ø Short Caption: A woman in a military uniform talks on the phone while holding a document, standing beside a man in a blue uniform against a wall. The setting is formal and professional, suggesting an official procedure in an institutional environment with light-colored walls and framed documents. The camera captures their upper bodies, moving backward and tilting upward, transitioning from a close-up to a medium close-up shot.

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Short Caption + Identities → Video

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Example - 6

Short Caption w/ Condition Caption + Identities → Video

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Structured Caption + Identities → Video

Ø Short Caption: A young woman in a traditional colorful outfit rides a galloping black horse through a lush green landscape. The camera follows her movements, capturing the dynamic and vibrant scene, with her hair flowing in the wind. The background is blurred to emphasize the speed and joy of the rider. The overall feel is natural and bright.

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Short Caption + Identities → Video

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Example -7

Short Caption w/ Condition Caption + Identities → Video

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Structured Caption + Identities → Video

- Figure 16. Illustrations of predicted structured captions based on the input multiple identities and the short instruction that expresses the implicit objects and the action for the target video generation.

Diffusion Models Diffusion Models

Ø Short Caption: A martial arts dojo scene where an instructor in black demonstrates techniques, throwing a student in white to the ground. Students sit in a circle on the green mat floor, observing attentively. In the background, banners and signs indicate martial arts training, with a seated audience and standing spectators. The camera moves from a medium to a long shot, capturing the full scene with respect and focus.

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Short Caption + Identities → Video

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Short Caption w/ Condition Caption + Identities → Video Example - 8

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Structured Caption + Identities → Video

Ø Short Caption: A young girl wearing a school uniform and a young man in casual clothes are walking side by side along a dimly lit concrete wall at night. The girl walks on the left while the boy rides a bicycle on the right. The background is urban and gritty, with warm, moody lighting. The camera follows them closely, capturing a medium close-up shot of their upper bodies from different angles as they move. The scene has a nostalgic and contemplative atmosphere.

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Short Caption + Identities → Video

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

Short Caption w/ Condition Caption + Identities → Video

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

###### Example - 9

Structured Caption + Identities → Video

- Figure 17. Illustrations of predicted structured captions based on the input multiple identities and the short instruction that expresses the implicit objects and the action for the target video generation.

Diffusion Models Diffusion Models

Ø Short Caption: A woman wearing fashionable clothes stands in the room, smiling and showing the goods in her hand. Then the camera zooms in and focuses on the details of the goods in the person's hand.

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

Short Caption + Identities → Video

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Short Caption w/ Condition Caption + Identities → Video

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

###### Example - 10

Structured Caption + Identities → Video

Ø Short Caption: Two cartoon characters are smiling at the camera together.

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

Short Caption + Identities → Video

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Short Caption w/ Condition Caption + Identities → Video

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

###### Example - 11

Structured Caption + Identities → Video

- Figure 18. Illustrations of predicted structured captions based on the input multiple identities and the short instruction that expresses the implicit objects and the action for the target video generation.

