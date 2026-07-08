# arXiv:2605.16079v1[cs.CV]15May2026

## VideoSeeker: Incentivizing Instance-level Video Understanding via Native Agentic Tool Invocation

Yiming Zhao1,2∗, Yu Zeng1,2∗, Wenxuan Huang2,3∗, Zhen Fang1,2∗, Qing Miao4, Qisheng Su1, Jiawei Zhao2, Jiayin Cai2, Lin Chen1, Zehui Chen1

Yukun Qi1,Yao Hu2,Xiaolong Jiang2, Feng Zhao1† 1 University of Science and Technology of China 2 Xiaohongshu Inc. 3 East China Normal University 4 Xi’an Jiaotong University Project Page: https://gaotiexinqu.github.io/VideoSeeker/

### Abstract

Large Vision-Language Models (LVLMs) have shown significant progress in video understanding, yet they face substantial challenges in tasks requiring precise spatiotemporal localization at the instance level. Existing methods primarily rely on text prompts for human-model interaction, but these prompts struggle to provide precise spatial and temporal references, resulting in poor user experience. Furthermore, current approaches typically decouple visual perception from language reasoning, centering reasoning around language rather than visual content, which limits the model’s ability to proactively perceive fine-grained visual evidence. To address these challenges, we propose VideoSeeker, a novel paradigm for instance-level video understanding through visual prompts. VideoSeeker seamlessly integrates agentic reasoning with instance-level video understanding tasks, enabling the model to proactively perceive and retrieve relevant video segments on demand. We construct a four-stage fully automated data synthesis pipeline to efficiently generate large-scale, high-quality instance-level video data. We internalize tool-calling and proactive perception capabilities into the model via cold-start supervision and RL training, building a powerful video understanding model. Experiments demonstrate that our model achieves an average improvement of +13.7% over baselines on instance-level video understanding tasks, surpassing powerful closed-source models such as GPT-4o and Gemini-2.5-Pro, while also showing effective transferability on general video understanding benchmarks. The relevant datasets and code will be released publicly.

### 1 Introduction

Large Vision Language Models (LVLMs) have achieved significant progress in recent years, demonstrating exceptional capabilities across diverse tasks including image captioning [46, 9, 40, 7], visual question answering [4, 1, 45, 41, 5], video understanding [50, 12, 23, 15, 36, 25], and complex multimodal reasoning [29, 3]. By deeply integrating visual and textual modalities, these models have developed strong multimodal perception and reasoning capabilities. Recently, methods [11, 34, 35] have successfully introduced reinforcement learning (RL) into video question answering and temporal localization. By leveraging environmental reward signals to guide models in exploring superior reasoning strategies, these approaches have achieved remarkable performance improvements in video understanding tasks, further expanding the temporal reasoning capabilities of LVLMs.

∗ Equal Contribution † Corresponding Author

Preprint.

(A) Instance-level video understanding task via visual prompts

(B) Our VideoSeeker Paradigm

Highly Automated Data Pipeline

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|
|---|
<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>|[Figure 36]|
|---|
|
|---|

Fully Automated Instance-Level VQA Synthesis

Multi-turn TrajectoryGeneration

[Figure 37]

[Figure 38]

[Figure 39]

What is the person outlined doing?

VideoSeeker

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

The person outlined in yellow is wearing a long black gown with a simple yet elegant design. She…

VQA

😊

[Figure 44]

{

[Figure 45]

Tool executed successfully. Here is the processed result:

"name": "view_visual_prompt", "arguments": { "frame_path": "frame_000391.png"

[Figure 46]

[Figure 47]

[Figure 48]

What is the seventh person from the left in the first row on stage doing at the beginning of the video?

[Figure 49]

[Figure 50]

View Target Instance

} }

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

The seventh person from the left in the first row on stage is wearing a black tuxedo, paired with…

🤨

[Figure 55]

{

Tool executed successfully. Here is the processed result:

"name": ”crop_video", "arguments": {

[Figure 56]

Multi-round InferenceTrajectory

”video_path": ”nobel

[Figure 57]

Fine-grained Localization

[Figure 58]

.mp4”, “start_time”: “510”, end_time: “740”

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

…

} }

Python

Localization Reasoning Final Answer

Query

Grounding Reasoning

Python

- Figure 1: Overview of VideoSeeker. (A): Instance-level video understanding tasks require models to accurately locate and reason about specific instances in videos guided by visual prompts, given a video, a visual prompt frame, and a query. Compared to text-only prompts that require lengthy referential descriptions, visual prompts provide a more intuitive interaction method. (B): Pipeline overview. We design a four-stage pipeline to construct instance-level video data, followed by a two-stage training strategy to integrate multimodal instance-level video understanding capabilities.

However, existing methods still suffer from two key limitations. (1) Most current approaches decouple visual perception from language reasoning, centering reasoning on language rather than visual evidence [11, 34, 35]. This weakens visual reasoning and often causes hallucinations in long-video scenarios [43]. Moreover, the widely used single-pass uniform sampling strategy is a passive perception mechanism that cannot adaptively capture key visual evidence, frequently missing fine-grained details critical for reasoning [12]. As a result, such methods struggle with precise localization tasks, e.g., identifying when a person appears for the second time. (2) Existing methods and benchmarks mainly focus on holistic video understanding [12, 38], emphasizing global semantics and coarse-grained events while lacking fine-grained spatio-temporal localization and reasoning for specific instances [37]. In addition, current approaches rely solely on text queries (Figure 1. A), which cannot provide precise spatial-temporal references [50]. This makes evaluating LVLMs in complex multi-object scenarios difficult and forces users to describe targets with lengthy referential language, reducing interaction efficiency and user experience.

To address these issues, we propose VideoSeeker, a novel paradigm for instance-level video understanding based on visual prompts (Figure 1. B). Unlike text-based prompts that rely on language descriptions, visual prompts enable users to directly annotate target regions on video frames, achieving more precise spatial and temporal references. As illustrated in Figure 2, we construct a four-stage fully automated visual prompt video question answering data synthesis pipeline to obtain high-quality data. Subsequently, through a two-stage strategy of SFT for cold-start combined with Agentic RL, we guide the model to explore the policy space with high information gain, ultimately integrating multi-round agentic reasoning paradigms and instance-level video understanding tasks into the baseline model. In the data pipeline, we first employ a lightweight language model for low-cost text pre-screening, then leverage powerful video understanding models to perform target uniqueness verification ensuring question solvability. Additionally, we integrate SAM3 [2] to achieve pixel-level instance segmentation, ultimately rendering diverse visual prompt types and generating instance-level video QA data ready for training. Extensive experiments demonstrate that our proposed VideoSeeker significantly outperforms all open-source baselines on the instance-level video understanding benchmark V2P-Bench, with our 8B model achieving an average improvement of +13.7% over baseline, surpassing powerful closed-source models such as GPT-4o and Gemini-2.5-Pro, while also exhibiting effective transferability to general video understanding scenarios.

###### AutomatedDataCuration Pipeline：Visual-Prompt-Dependent VideoQADataset

###### (b)Video-level Verification

###### (a)Low-cost Text Filtering

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Camera

[Figure 74]

Gemini-3.1-Pro

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Environment

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Arbitrary Video QA Datasets

Uniqueness Assessment

Semantic Tag Generation

Temporal Localization

Entity Removal

[Figure 86]

Retained Removed

###### (c)Pixel-level Mask Generation (d) RenderingandRewriting

[Figure 87]

|[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]|
|---|

[Figure 124]

[Figure 125]

[Figure 126]

| |
|---|

[Figure 127]

[Figure 128]

Input: (Semantic Tag + Temporal Window)

…in which the man in blue top…?

Oringinal Video

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Visual Prompt QA

[Figure 133]

[Figure 134]

[Figure 135]

…in which the target within the box…?

[Figure 136]

[Figure 137]

Visual Prompt Frame

1 fps Sampling

Final VisualPrompt Dataset

Diverse Visual Prompt Types

Entity Obfuscation

Accurate Pixel-level Mask Sequences

Video Segmentation (SAM3)

- Figure 2: Our Data Pipeline. (1) Low-cost Text Filtering rapidly filters pure text QA pairs; (2) Video-level Verification verifies target uniqueness and generates semantic tags; (3) Pixel-level Mask Generation produces pixel-wise masks via SAM3; (4) Visual Prompt Rendering renders diverse visual prompt types and rewrites QA to depend on them. In a nutshell, our contributions are as follows:

- • We propose VideoSeeker, an agentic instance-level video understanding paradigm. By organically integrating agentic reasoning, VideoSeeker breaks through the limitations of text queries and achieves more precise references.
- • We construct a four-stage instance-level video question answering data synthesis pipeline and efficiently generates large-scale, high-quality instance-level video data, providing an effective solution to the scarcity of relevant training data.
- • Extensive experiments demonstrate that VideoSeeker significantly outperforms all opensource and proprietary baselines on instance-level video understanding tasks, while also exhibiting effective transferability to general video understanding scenarios.

### 2 Related Works

Reinforcement Learning for Vision Language Models. Inspired by the success of large reasoning models such as OpenAI o1 [18] and DeepSeek-R1 [13], recent studies extend GRPO-style RL [26] from text-only reasoning to multimodal domains [24]. In vision, methods enhance reasoning for image QA [16, 22, 10], grounding [21, 27]. For example, Perception-R1 [44] leverages object matching and IoU as reward signals to improve grounding, and DeepEyes [52] shows how RL can encourage models to invoke visual tools, thereby expanding perceptual abilities. Video-centric approaches further tackle temporal reasoning tasks such as video QA [11, 34] and temporal grounding [37, 20], with VideoR1 [11], VideoChat-R1 [20] and VideoRFT [34] being representative works. Additionally, VisionR1 [16] and R1-OneVision [42] construct multimodal CoT datasets by converting visual information into textual representations to support stronger reasoning. Despite these advances, most methods still rely on text-based CoT reasoning [11, 20, 6], which remains largely language-centric [43], limiting visual reasoning and increasing hallucinations in long-video scenarios. This motivates us to explore how to enable more effective video reasoning through visual tool augmentation.

Tool-Augmented Agentic Vision Language Models. Recent advances in LVLMs show that equipping models with external tools can enhance capabilities beyond pure text understanding and generation [33, 52]. In the image domain, methods [52, 33, 30, 32, 14] enable MLLMs to “think with images” by integrating visual tools for image reasoning, while VILA-SR [39] reinforces spatial

reasoning with interwoven visual drawing. In the video domain, LongVT [43] proposes iMCoTT that enables MLLMs to perform native temporal retrieval and reasoning by dynamically selecting and re-inspecting relevant video segments, without an auxiliary retriever. VITAL [47] constructs a visual toolbox that allows models to densely sample new video frames on demand during reasoning, enabling precise long video reasoning. Additionally, Ego-R1 [31] explores chain-of-tool-thought reasoning in first-person videos, and PyVision [49] proposes dynamic tool calling. However, our method differs from prior works such as LongVT [43] and VITAL [47] in the following key aspects: (1) VideoSeeker targets instance-level video understanding tasks, focusing on precise localization and tracking of specific target instances within videos; whereas LongVT and VITAL primarily emphasize holistic semantic modeling. (2) VideoSeeker employs visual prompts (e.g., bounding boxes, points, and masks) as queries, enabling direct specification of target instances with more precise spatial and temporal references; whereas prior works rely entirely on pure text queries, requiring extensive referential language to describe targets. (3) We design a four-stage fully automated data pipeline that efficiently generates large-scale, high-quality instance-level video data, and propose a two-stage training paradigm to internalize native tool-calling capabilities into the base model, enabling native instance-level video understanding.

### 3 Method

#### 3.1 Task Formulation And Environmental Interaction

Task Formulation. Given a query Q, a visual prompt frame Fvp and a video V of arbitrary length, the goal of instance-level video understanding is to accurately answer the query Q with respect to the

specific instance indicated by Fvp, and output a grounded answer A. Unlike general video question answering where the answer is independent of a particular object, instance-level video understanding requires the model to (1) precisely associate the visual prompt with the corresponding target instance in V and (2) reason about the temporal dynamics of that specific instance across V to produce the final answer A.

Environmental Interaction. The policy model πθ interacts with the video environment through multiturn active perception control, rather than passively encoding all context in a single pass. Specifically, the model is equipped with a perception tool set T = {view_visual_prompt, crop_video}: the former continuously provides visual prompt frames Fvp, maintaining a cognitive anchor of the target instance appearance throughout reasoning; the latter endows the model with fine-grained local observation capability, enabling active filtering of keyframes and removal of redundant information when processing long videos with complex visual prompts. The two tools are formally defined as:

Ivp = view_visual_prompt Pvp , Pvp ∈ RH×W×3, (1) Vcrop = crop_video Pv,τs,τe , τs,τe ∈ R+, τs < τe, (2)

where Pvp denotes the visual prompt frame path and Ivp represents the decoded image; Pv denotes the video path, and τs,τe denote the start and end timestamps, respectively, yielding the cropped temporal segment Vcrop. In each round t (where t = 0,1,2,...,Tmax), the model samples a response Rt ∼ πθ(· | M) from the current message context M, which may contain ⟨tool_call⟩ blocks, ⟨answer⟩ blocks, or both. When the model decides to invoke a perception tool, the tool is executed and its result is appended to M for the next round; when an answer block appears, the ExtractAnswer function is called to extract answer A, and the interaction terminates. This iterative cognitive cycle of “active perception → local zoom → evidence-based reasoning” parallels the human cognitive strategy of “global browsing to local close-reading” when confronting complex visual scenes, thereby circumventing the context loss and evidence obscuration inherent in single-pass compression paradigms.

To better illustrate the overall procedure, the entire rollout process is presented in Algorithm 1.

#### 3.2 Data Construction

Preliminary Data Curation. To construct large-scale high-quality visual prompt video QA data, we propose a fully automated four-stage pipeline that transforms arbitrary video QA datasets into visual-prompt-dependent QA data without any manual annotation.

Dfinal = G4 ◦ G3 ◦ G2 ◦ G1(Draw), (3)

Algorithm 1 Multi-turn Interactive Inference Process of VLM with Environment

Require: Query Q, Visual Prompt Frame Fvp, Video V, Tool Set T = {view_visual_prompt, crop_video},

Policy Model πθ, Maximum Tool Rounds Tmax.

Ensure: Final Answer A, Interaction Trajectory Y, Tool Call History H.

- 1: Initialization: Y ← ∅, t ← 0, H ← ∅.
- 2: Encode V into visual frame sequence: Vframes ← EncodeVideoFrames(V).
- 3: Compose user message M ← Vframes + {Q, ToolPrompt(T , Fvp)}.
- 4: while t ≤ Tmax do
- 5: Sample model response: Rt ∼ πθ(· | M).
- 6: Append Rt to trajectory: Y ← Y + Rt.
- 7: if <tool_call></tool_call> detected in Rt then
- 8: Parse {(funck, argsk)} from Rt, append to H.
- 9: Execute tools and append results to M: M ← M + ExecuteTools({(funck, argsk)}).
- 10: end if
- 11: if <answer></answer> detected in Rt then
- 12: Extract answer A ← ExtractAnswer(Rt).
- 13: return (A, Y, H).
- 14: end if
- 15: t ← t + 1.
- 16: end while
- 17: return (NULL, Y, H).

where G1 to G4 correspond to Filtering, Verification, Mask Generation, and Rendering, respectively.

- (1) Low-cost Text Filtering. Since video tokens are computationally expensive, processing all data with video understanding leads to significant resource waste. We employ GPT-4o [17] to rapidly filter pure text QA pairs, eliminating samples unsuitable for visual prompting and preserving only QA pairs targeting concrete visual entities for the next stage:

Ffilter : D  → {0,1}, Dfilter = {d ∈ Draw | Ffilter(d) = 1}, (4) where D denotes the dataset space and d = (v,q,a) ∈ D contains video v, question q, and answer a.

- (2) Video-level Verification. For pre-filtered samples, we further verify whether the target is uniquely identifiable in the video. We use Gemini-3.1-Pro [8] to jointly process videos and original QA pairs through a five-step reasoning pipeline: target extraction with uniqueness judgment, generation of a unique semantic tag for SAM3 segmentation, temporal window localization, and QA rewriting with a unified < vp > placeholder:

Rrewrite : V × QA  → QAvp, QAvp = Rrewrite(V,QA;ϕ), (5)

where ϕ denotes the internal five-step reasoning process comprising target extraction with uniqueness judgment, semantic tag generation for SAM3, temporal window localization, and < vp > substitution.

- (3) Pixel-level Mask Generation. Semantic tags alone are insufficient for pixel-level visual prompt rendering. We adopt SAM3 [2] to conduct text-driven video diffusion segmentation based on semantic tags, sampling at one frame per second to generate precise pixel-level masks:

Mτ = SAM3(V,τ;ω), ∀τ ∈ T, T = {⌊t⌋ | t ∈ [0,T)}, (6) where ω denotes the semantic tag condition and T denotes the total video duration in seconds.

- (4) Visual Prompt Rendering. To enhance data diversity and establish alignment between visual prompt symbols and natural language descriptions, we uniformly sample eight visual prompt types and render them on video frames. We then invoke a language model to replace the < vp > placeholder with natural language descriptions corresponding to the visual prompt types, producing visual prompt QA data ready for training:

QArendered = LLM QAvp,VP , (7) where VP denotes the sampled visual prompt type. The unified < vp > facilitates community extensions by enabling seamless substitution across different visual prompt types without modifying downstream model interfaces.

SFT and RL Data Curation. Due to the limited capability of the base VLM, which exhibits poor instruction-following and high tool-calling error rates, we adopt a reject sampling strategy to generate

high-quality multi-turn tool-calling trajectories. Specifically, we use data from the Preliminary Data Curation stage as input, and leverage Qwen3-VL-235B-A22B-Thinking to interact with the video environment using predefined tools. Subsequently, a rule-based discriminator filters out trajectories where the model responds correctly, ultimately yielding 34.2k high-quality samples for SFT stage. During the RL training phase, we further filter the SFT data based on the pass-k metric, resulting in 4.1k samples for GRPO training.

#### 3.3 Training Strategy

Supervised Fine-Tuning. We first conduct SFT to equip the model with foundational behaviors required for multimodal tool-calling VLMs, thereby ensuring effective interaction with the environment. Following the procedure described in Section 3.2, we collect 34.2k high-quality trajectories for training. The model is trained by minimizing the standard autoregressive cross-entropy loss. The objective of SFT is to guide the model toward learning multi-turn, multi-scale active perception patterns in video environments, integrating visual evidence during reasoning, endowing the policy model with basic capabilities for interacting with the video environment, and establishing a foundation for agentic reinforcement learning.

Agentic Reinforcement Learning. In this stage, we treat the model as an agent capable of autonomously using tools, which actively decides whether to view the visual prompt, how to crop segments, and how to integrate retrieved evidence into the reasoning process. We employ GRPO to achieve this objective. The policy model is optimized by maximizing the following objective:

Ex,{y

i}Gi=1

G

1 G

1 t I(yi,t)

min ri,t, clipped(ri,t) A ˆi,t − β DKL(πθ∥πref), (8)

i=1

t:I(yi,t)=1

where ri,t = πθ(yi,t|x,yi,<t)/πold(yi,t|x,yi,<t) and clipped(r) = clip(r,1 − ϵ,1 + ϵ). The rollout module samples a group of trajectories {y1,y2,...,yG} from the old policy πold for each input question x through interaction with the external environment V. The advantage term Aˆi,t is computed based on the relative rewards of outputs within each group. Additionally, we introduce a three-component reward modeling approach that jointly optimizes sampled trajectories across three dimensions: answer accuracy, format compliance, and generation efficiency. This design enhances final answer correctness, promotes more effective tool usage during inference, and produces more reliable and well-reasoned trajectories.

- 1. Answer Accuracy. For the k-th rollout, let aˆ(k) and a⋆ denote the extracted answer and the ground truth, respectively. We adopt Qwen3-VL-235B-A22B-Instruct [1] as a judge to assess their semantic consistency and output a score in {1,0.5,0} (fully correct, partially correct, or incorrect). The accuracy reward is defined as:

Racc(k) = JudgeLLM a ˆ(k), a⋆ ∈ {1, 0.5, 0}. (9)

- 2. Format Compliance. Let y(k) denote the complete textual output of the k-th rollout and S be the predefined output schema. This reward encourages the model to consistently produce well-structured outputs with properly organized tool invocations and final answers, enabling reliable downstream parsing and verification. The format reward is computed as:

Rformat(k) = ⊮ y(k) matches S . (10)

- 3. Parsimony Reward. We introduce a parsimony reward to encourage the model to accomplish tasks with fewer tool-calling rounds while maintaining answer correctness. Specifically, let N(k) denote the total number of perception tool invocations triggered in the k-th rollout. The parsimony reward is computed as:

Rpar(k) = max{0, 1 − λ · N(k)}, (11) where λ controls the strength of the parsimony penalty. This design implicitly incentivizes the model to only invoke tools when additional evidence is needed, thereby achieving a balance between effective reasoning and resource efficiency.

Table 1: Evaluation Results on V2P-Bench across Dimensions. The "Agent" column indicates whether native tool calling is enabled (✓) or disabled (✗) in the prompt. The best results are bold and the second-best are underlined.

Model Agent OA HA OD FM CR PU CI FT RT AS SR GC Avg. Proprietary VLMs

GPT-4o ✗ 76.6 68.9 41.3 60.8 67.0 73.3 67.6 68.1 70.5 50.0 54.0 48.4 65.4 Gemini-2.5-Pro ✗ 84.0 72.4 68.2 71.8 75.0 73.3 22.6 66.7 72.7 47.4 67.5 63.6 69.8

###### Open-source VLMs

LLaVA-OV-7B ✗ 57.1 52.1 28.3 47.1 63.8 59.1 41.0 42.1 35.6 63.2 62.8 43.2 52.8 LLaVA-OV-72B ✗ 65.5 59.9 34.7 47.0 63.8 43.2 38.5 50.0 41.1 66.3 66.9 45.9 56.7 InternVL3-8B ✗ 73.9 69.1 39.1 60.8 58.1 65.9 41.0 52.6 41.1 61.1 69.7 64.9 61.7 LLaVA-Video-7B ✗ 60.5 58.1 37.0 49.0 62.9 54.5 41.0 52.6 48.9 57.9 56.6 40.5 54.8 LLaVA-Video-72B ✗ 62.2 60.8 30.4 54.9 61.0 54.5 43.6 47.4 42.2 70.5 71.0 59.5 58.6

###### Ours

Qwen3-VL-4B ✗ 73.7 67.1 41.2 58.8 51.7 52.9 51.2 42.1 42.2 63.2 71.9 48.6 59.2 Qwen3-VL-4B ✓ 73.0 68.6 26.7 52.9 26.9 62.5 41.0 50.0 44.5 67.6 64.8 40.5 57.6 VideoSeeker-4B ✓ 85.7 79.4 60.3 70.6 55.6 59.1 61.4 55.6 56.0 76.5 69.2 66.7 70.6 ∆ - +12.0 +12.3 +19.1 +11.8 +3.9 +6.2 +10.2 +13.5 +13.8 +13.3 -2.7 +18.1 +11.4

Qwen3-VL-8B ✗ 81.8 68.1 41.3 67.8 45.4 58.3 58.3 47.4 38.5 56.2 70.5 51.3 60.8 Qwen3-VL-8B ✓ 71.1 71.2 39.1 58.8 48.3 41.2 64.3 42.1 51.7 57.9 68.4 59.4 59.9 VideoSeeker-8B ✓ 94.3 79.1 53.7 76.5 56.5 78.6 66.8 62.3 55.0 78.8 78.8 70.3 74.5 ∆ - +12.5 +11.0 +12.4 +8.7 +11.1 +20.3 +8.5 +14.9 +16.5 +22.6 +8.3 +19.0 +13.7

- 4. Integrated Reward Function. The final reward function is a weighted combination of the three components described above, with weights used to balance the contributions of each component:

R(k) = α · Racc(k) + β · Rformat(k) + γ · Rpar(k) . (12)

where α+β+γ = 1. By integrating these three components into the reward function, our VideoSeeker provides a comprehensive and fine-grained evaluation mechanism, guiding the model to better align with real-world application requirements when optimizing its reasoning capabilities.

### 4 Experiments

#### 4.1 Implementation Details.

Training and Evaluation Setup. In the SFT and RL stages, we leverage 34.2k trajectories and a curated dataset of 4.1k samples collected in Section 3.2. All experiments are built upon Qwen3-VL-4B and Qwen3-VL-8B as base models. We evaluate VideoSeeker against a comprehensive suite of baselines, including open-source models like VideoR1 [11], VideoRFT [34], VideoThinker [35] and proprietary models like GPT-4o [17], Gemini-2.5Pro [8]. Evaluations are conducted on four video understanding benchmarks: V2P-Bench [50], a dedicated instance-level video understanding evaluation framework, and three general video understanding benchmarks: Video-MME [12], LongVideoBench [38], and LongVT [43]. We deploy models based on vLLM [19] with native tool-calling mechanisms compatible with the OpenAI SDK, enabling multi-round tool-augmented reasoning. Specifically, we equip models with multiple visual tools, including frame sampling for temporal localization and object detection for spatial grounding, allowing models to dynamically invoke tools based on query complexity. For all evaluations, we set the temperature to 0 to ensure reproducibility of the results.

Table 2: Evaluation Results on General Benchmarks. The bests are bold and the second-best are underlined.

Model Agent Video-MME LongVideoBench LongVT Avg. Proprietary VLMs

GPT-4o ✗ 77.2 81.3 17.4 58.6 Gemini-2.5 Pro ✗ 84.8 - - -

###### Open-source VLMs

Video-R1-7B ✗ 61.0 - 27.9 VideoRFT-7B ✗ 60.9 - 26.5 Video-Thinker-7B ✗ 61.0 - 10.4 LongVT-7B ✗ 66.1 - 31.0 -

###### Ours

Qwen3-VL-4B ✗ 65.3 62.6 38.5 55.5 Qwen3-VL-4B ✓ 61.5 50.4 36.9 49.6 VideoSeeker-4B ✓ 66.1 64.2 45.7 58.7 ∆ - +0.8 +1.6 +7.2 +3.2

Qwen3-VL-8B ✗ 67.4 64.6 39.4 57.1 Qwen3-VL-8B ✓ 58.3 42.9 11.7 37.6 VideoSeeker-8B ✓ 68.1 66.5 46.5 60.4 ∆ - +0.7 +1.9 +7.1 +3.3

Training Infrastructure. We conduct SFT on LLaMA-Factory [51] and RL training on verl [28], both employing full-parameter fine-tuning. All experiments are performed on 8 NVIDIA H800 GPUs. More detailed training hyperparameters are provided in Appendix C.

#### 4.2 Main Results

As illustrated in Table 1, our VideoSeeker series achieves the best performance among open-source models and is competitive with powerful closed-source models. Specifically, VideoSeeker-4B improves over the baseline Qwen3-VL-4B by +11.4% on average, with particularly notable gains in HA, OD, and AS; scaling up to VideoSeeker-8B further improves over Qwen3-VL-8B by +13.7% on average, showing clear advantages across most fine-grained dimensions while surpassing Gemini2.5-Pro and GPT-4o. As shown in Table 2, although our training data exclusively comes from instance-level video understanding tasks, VideoSeeker demonstrates generalization ability on general video understanding benchmarks, achieving an average improvement of +3.2% and +3.3% over three tasks. This indicates that our proposed tool-calling paradigm for instance-level video understanding can effectively transfer to broader general video understanding scenarios.

#### 4.3 Ablation Studies Table 3: Tools Ablation.

Tools Ablation. As shown in Table 3, we decomposed the tool set to analyze the contribution of each tool. The consistent performance improvements brought by the gradual introduction of the tool set clearly validate the effectiveness of our methodological paradigm. Notably, the combination of the two tools yields synergistic gains that exceed their individual contributions, indicating that the two tools form a complementary relationship in information acquisition.

VP. Crop. Avg. Qwen3-VL-8B (Baseline) 60.8

✓ 69.4 ✓ 63.7 ✓ ✓ 74.5

Data Ablation. We construct several subsets by progressively increasing the sampling ratio from the full training corpus to investigate the impact of SFT data scale on model performance. As shown in Figure 3, performance improves with increasing data volume, and the gains gradually diminish as the data scale expands further. This observation reveals a prominent diminishing marginal returns pattern in performance improvement, where the model approaches saturation beyond a certain data scale. These findings provide insights for balancing dataset scale and model performance.

Impact of Data Scale

70

68

###### Accuracy(%)

66

64

62

60

0.0 10.0 20.0 30.0 34.2

SFT Data Scale (k)

Increasing SFT data improves model performance

Figure 3: Effect of Data Scale.

Reward Ablation. As shown in Table 4, our reward system provides a stable training signal, and we systematically analyze the contribution of each reward signal during RL training. The format reward substantially outperforms the accuracy-only baseline, while the efficiency reward encourages more concise tool usage. Notably, the combined three-reward scheme surpasses the sum of individual contributions, revealing complementary effects across reward dimensions that jointly enhance effective reasoning.

#### Table 4: Reward Ablation.

###### Reward Type

Acc. Racc Rformat Reff

✓ 65.4 ✓ ✓ 73.1 ✓ ✓ 68.7 ✓ ✓ ✓ 74.5

#### Table 5: Stage Ablation.

Stage Ablation. As shown in Table 5, we systematically ablate the contributions of the SFT and RL training stages to model performance. Experimental results demonstrate that high-quality SFT data endows the model with robust reasoning patterns, yielding a substantial performance boost (+9.6%). In the zero-shot RL setting, single-turn RL leads to marginal improvement (+1.8%). In contrast, agentic RL paradigm achieves +5.1% improvement, which is more ef-

SFT RL (Single Turn) RL (Agentic) Acc. Qwen3-VL-8B (Baseline) 60.8 ✓ 70.4 ✓ 62.6 ✓ 65.9 ✓ ✓ 74.5

fective (+3.3%) than single-turn RL. This validates the agentic paradigm as a critical enabler for effective RL training on instance-level video understanding tasks. The cascaded two-stage training paradigm leverages synergistic gains from both strategies, achieving optimal performance (74.5%) and thereby establishing the optimal training pipeline in our framework.

#### 4.4 Analysis

Generalization to General Video Understanding Tasks. Despite being trained exclusively on instance-level video understanding tasks, VideoSeeker demonstrates strong cross-task generalization on general video benchmarks (Table 2), achieving +3.2% and +3.3% improvements in average. This reveals that core capabilities learned from instance-level tasks, such as long-range visual reasoning and multi-turn reasoning, transfer compositionally to broader video understanding scenarios. These findings highlight the value of instance-level video data in instilling generalizable priors, enabling cross-task improvements without additional general data.

The heterogeneous distillation paradox: stronger teachers may produce weaker students. We experiment with two teacher models: Qwen3-VL235B-A22B-Thinking and Gemini-3.1-Pro, achieving 78.4% and 83.8% accuracy on the rejectionsampled dataset respectively. After SFT training Qwen3-VL-8B, the resulting student models achieve 70.4% and 64.7% on V2P-Bench, with relative performance degradation of 8.0% and 19.1%. As illustrated in Figure 4, this reveals a counter-intuitive finding: The raw capability of a teacher model does not proportionally transfer to distillation performance. In homogeneous distillation, teachers and students share similar patterns, enabling efficient

[Figure 138]

Stronger teachers may produce weaker students when architectures diverge !

Figure 4: Distillation Paradox.

knowledge transfer; in heterogeneous distillation, pattern divergence is significant, causing stronger teachers’ knowledge to be less effectively absorbed and leading to greater performance degradation.

Table 6: Reward Hacking.

RL training suffers from reward hacking on multiple-choice data. As shown in Table 6, RL training on multiple-choice (MC) data leads to a significant performance drop to 43.8%, as models exploit random guessing rather than learning robust video understanding. In contrast, open-ended (OE) training achieves 74.5%, demonstrating that OE with LLM judges provides a more robust strategy.

MC OE Avg. VideoSeeker-8B (SFT) 70.4

✓ 43.8 ✓ 74.5

Time Efficiency. We uniformly evaluate inference costs under the Agent mode. As illustrated in Figure 5, VideoSeeker substantially reduces inference costs in both generation and tool-calling phases. Baseline models frequently exhibit frequent tool invocations accompanied by verbose chain-of-thought trajectories, resulting in prohibitively high computational overhead. In contrast, VideoSeeker converges to the correct answer with fewer total action steps through streamlined tool-calling strategies and more compact reasoning chains.

[Figure 139]

VL-8B Qwen3

VL-4B Video

Seeker-4B

Seeker-8B Qwen3

Video

Figure 5: Inference Latency.

Case Study. The case study in Appendix F demonstrates how VideoSeeker successfully invokes tools to examine instance targets, clips specific segments for precise localization, and ultimately completes the task. This agentic interaction paradigm enables the model to handle instance-level video understanding with high precision, avoiding the localization errors inherent in traditional methods that rely on vague textual descriptions.

### 5 Conclusion

In this work, we propose VideoSeeker, an agentic paradigm that enables LVLMs to perform instancelevel video understanding through native tool invocation. By integrating agentic reasoning with instance-level video understanding tasks, VideoSeeker empowers models to proactively perceive and retrieve relevant video segments on demand, achieving more precise spatial and temporal references than traditional text-only approaches. We construct a four-stage fully automated data synthesis pipeline to generate large-scale instance-level video data, and develop a two-stage training strategy to internalize tool-calling capabilities into LVLMs. Experiments on V2P-Bench demonstrate that VideoSeeker-8B achieves an average improvement of +13.7%, surpassing GPT-4o and Gemini-2.5Pro, while also exhibiting effective transferability to broader video understanding scenarios.

### References

- [1] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [2] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.
- [3] A. Chen, A. Li, B. Gong, B. Jiang, B. Fei, B. Yang, B. Shan, C. Yu, C. Wang, C. Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.
- [4] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024.
- [5] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.
- [6] Y. Chen, W. Huang, B. Shi, Q. Hu, H. Ye, L. Zhu, Z. Liu, P. Molchanov, J. Kautz, X. Qi, et al. Scaling rl to long videos. arXiv preprint arXiv:2507.07966, 2025.
- [7] C. Clark, J. Zhang, Z. Ma, J. S. Park, M. Salehi, R. Tripathi, S. Lee, Z. Ren, C. D. Kim, Y. Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. arXiv preprint arXiv:2601.10611, 2026.
- [8] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [9] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 91–104, 2025.
- [10] Y. Deng, H. Bansal, F. Yin, N. Peng, W. Wang, and K.-W. Chang. Openvlthinker: Complex vision-language reasoning via iterative sft-rl cycles. arXiv preprint arXiv:2503.17352, 2025.
- [11] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, J. Wu, X. Zhang, B. Wang, and X. Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [12] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118, 2025.
- [13] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [14] J. Hong, C. Zhao, C. Zhu, W. Lu, G. Xu, and X. Yu. Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.
- [15] W. Hong, X. Gu, Z. Pan, Z. Yang, Y. Wang, Y. Wang, Y. Yue, Y. Wang, Y. Wang, Y. Wang, et al. Glm5v-turbo: Toward a native foundation model for multimodal agents. arXiv preprint arXiv:2604.26752, 2026.
- [16] W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F. Zhao, Z. Xu, X. Tang, Y. Hu, and S. Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [17] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [18] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [19] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

- [20] X. Li, Z. Yan, D. Meng, L. Dong, X. Zeng, Y. He, Y. Wang, Y. Qiao, Y. Wang, and L. Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.
- [21] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang. Visual-rft: Visual reinforcement fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2034–2044, 2025.
- [22] F. Meng, L. Du, Z. Liu, Z. Zhou, Q. Lu, D. Fu, T. Han, B. Shi, W. Wang, J. He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint

- arXiv:2503.07365, 2025.

[23] Y. Qi, Y. Zhao, Y. Zeng, X. Bao, W. Huang, L. Chen, Z. Chen, J. Zhao, Z. Qi, and F. Zhao. Vcrbench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint

- arXiv:2504.07956, 2025.

- [24] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [25] S. Ren, L. Yao, S. Li, X. Sun, and L. Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024.
- [26] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [27] H. Shen, P. Liu, J. Li, C. Fang, Y. Ma, J. Liao, Q. Shen, Z. Zhang, K. Zhao, Q. Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.
- [28] G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [29] K. Team, T. Bai, Y. Bai, Y. Bao, S. Cai, Y. Cao, Y. Charles, H. Che, C. Chen, G. Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.
- [30] O. Team. Thinking with images. https://openai.com/index/thinking-with-images/, 2025.
- [31] S. Tian, R. Wang, H. Guo, P. Wu, Y. Dong, X. Wang, J. Yang, H. Zhang, H. Zhu, and Z. Liu. Ego-r1: Chain-of-tool-thought for ultra-long egocentric video reasoning. arXiv preprint arXiv:2506.13654, 2025.
- [32] C. Wang, K. Feng, D. Chen, Z. Wang, Z. Li, S. Gao, M. Meng, X. Zhou, M. Zhang, Y. Shang, et al. Adatooler-v: Adaptive tool-use for images and videos. arXiv preprint arXiv:2512.16918, 2025.
- [33] H. Wang, A. Su, W. Ren, F. Lin, and W. Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025.
- [34] Q. Wang, Y. Yu, Y. Yuan, R. Mao, and T. Zhou. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025.
- [35] S. Wang, J. Jin, X. Wang, L. Song, R. Fu, H. Wang, Z. Ge, Y. Lu, and X. Cheng. Video-thinker: Sparking" thinking with videos" via reinforcement learning. arXiv preprint arXiv:2510.23473, 2025.
- [36] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [37] Y. Wang, Z. Wang, B. Xu, Y. Du, K. Lin, Z. Xiao, Z. Yue, J. Ju, L. Zhang, D. Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025.
- [38] H. Wu, D. Li, B. Chen, and J. Li. Longvideobench: A benchmark for long-context interleaved videolanguage understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024.
- [39] J. Wu, J. Guan, K. Feng, Q. Liu, S. Wu, L. Wang, W. Wu, and T. Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965, 2025.

- [40] L. Xing, X. Dong, Y. Zang, Y. Cao, J. Liang, Q. Huang, J. Wang, F. Wu, and D. Lin. Caprl: Stimulating dense image caption capabilities via reinforcement learning. arXiv preprint arXiv:2509.22647, 2025.
- [41] G. Xu, P. Jin, Z. Wu, H. Li, Y. Song, L. Sun, and L. Yuan. Llava-cot: Let vision language models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2087–2098, 2025.
- [42] Y. Yang, X. He, H. Pan, X. Jiang, Y. Deng, X. Yang, H. Lu, D. Yin, F. Rao, M. Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2376–2385, 2025.
- [43] Z. Yang, S. Wang, K. Zhang, K. Wu, S. Leng, Y. Zhang, B. Li, C. Qin, S. Lu, X. Li, et al. Longvt: Incentivizing" thinking with long videos" via native tool calling. arXiv preprint arXiv:2511.20785, 2025.
- [44] E. Yu, K. Lin, L. Zhao, J. Yin, Y. Wei, Y. Peng, H. Wei, J. Sun, C. Han, Z. Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025.
- [45] Y. Zeng, W. Huang, S. Huang, X. Bao, Y. Qi, Y. Zhao, Q. Wang, L. Chen, Z. Chen, H. Chen, et al. Agentic jigsaw interaction learning for enhancing visual perception and reasoning in vision-language models. arXiv preprint arXiv:2510.01304, 2025.
- [46] Y. Zeng, Y. Qi, Y. Zhao, X. Bao, L. Chen, Z. Chen, S. Huang, J. Zhao, and F. Zhao. Enhancing large visionlanguage models with ultra-detailed image caption generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26703–26729, 2025.
- [47] H. Zhang, X. Gu, J. Li, C. Ma, S. Bai, C. Zhang, B. Zhang, Z. Zhou, D. He, and Y. Tang. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416, 2025.
- [48] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.
- [49] S. Zhao, H. Zhang, S. Lin, M. Li, Q. Wu, K. Zhang, and C. Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025.
- [50] Y. Zhao, Y. Zeng, Y. Qi, Y. Liu, X. Bao, L. Chen, Z. Chen, Q. Miao, C. Liu, J. Zhao, et al. V2p-bench: Evaluating video-language understanding with visual prompts for better human-model interaction. arXiv preprint arXiv:2503.17736, 2025.
- [51] Y. Zheng, R. Zhang, J. Zhang, Y. Ye, and Z. Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pages 400–410, 2024.
- [52] Z. Zheng, M. Yang, J. Hong, C. Zhao, G. Xu, L. Yang, C. Shen, and X. Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

### Appendix Overview

- • Section A: Dataset Details.
- • Section B: Benchmark Information.
- • Section C: Hyperparameters.
- • Section D: Limitations and Social Impacts.
- • Section E: Training Curves.
- • Section F: Case Study.
- • Section G: Prompts.

### A Dataset Details Table 7: Video source distribution.

Data Source. Our data construction pipeline uses the original videos and QA data from LLaVA-Video-178K [48] as source material, comprising 178k videos and approximately 1.3 million instruction samples. This dataset integrates 10 mainstream video sources, covering domains such as activity recording, cooking, film, first-person perspective, and more. Multi-dimensional filtering rules are applied to select unedited raw videos with rich temporal variations, ensuring narrative completeness. Figure 7 illustrates the source distribution of the video data.

Source Count Ratio YouTube 5,405 72.1% Charades 826 11.0% ActivityNet 526 7.0% YouCook2 454 6.1% NextQA 258 3.4% Ego4D 28 0.4%

Data Construction Pipeline. As described in Section 3.2, we propose a fully automated four-stage pipeline, starting from 147,245 raw video QA samples from LLaVA-Video-178K [48] and transforming them into visual-prompt-dependent QA data: (1) Text Filtering, using GPT-4o to quickly pre-filter text QA and remove samples unsuitable for visual prompts (e.g., camera/cinematography questions, scene/background descriptions, overall activity summaries, counting questions, abstract/non-visual questions, ambient lighting/color queries), retaining 44.5%; (2) Video Verification, using Gemini-3.1Pro to perform five-step reasoning jointly with the video, excluding multi-target ambiguous samples, retaining 32.9%; (3) SAM3 Segmentation, generating pixel-level masks at 1 fps based on semantic labels, retaining 27.9%; (4) Visual Prompt Rendering, uniformly sampling 8 visual prompt types (rectangle, mask contour, ellipse, triangle, scribble, point, arrow, set-of-mark) to render on video frames and rewrite QA, retaining 27.8%. Table 8 presents detailed stage and statistics information. The final dataset covers 8 visual prompt types, providing diverse spatial and geometric variations for model training.

Table 8: Data pipeline retention statistics.

Stage Output Relative Retention Description

- Step 0: Original Dataset 147,245 100.0% LLaVA-Video-178K original data
- Step 1: Text Filtering 65,594 44.5% GPT-4o filters text QA
- Step 2: Video Verification 48,457 32.9% Gemini-3.1-Pro serves as a verifier
- Step 3: SAM3 Segmentation 41,083 27.9% 1 FPS pixel-level mask generation
- Step 4: Visual Prompt Rendering 40,929 27.8% Rendering and QA rewriting

### B Benchmark Information

We evaluate on four video understanding benchmarks, including one instance-level video understanding benchmark and three general video understanding benchmarks. This section introduces each benchmark. During evaluation, we uniformly segment videos into 256 frames on average.

- • V2P-Bench [50] is a benchmark for evaluating LVLMs on visual-prompt-driven instancelevel video understanding. Unlike text-only approaches, it introduces visual prompting to require precise spatial-temporal reasoning. It contains 980 videos with 1,172 QA pairs,

- covering three core tasks across twelve evaluation dimensions, assessing instance-level video understanding.
- • Video-MME [12] is a video understanding benchmark for multimodal LLMs, evaluating capabilities in long-video and complex-reasoning scenarios. It contains approximately 900 manually curated videos spanning multiple domains, with 2.7K multiple-choice QA pairs. All data undergo rigorous human annotation. The dataset supports video, subtitles, and audio inputs. In our evaluation, we exclusively use video modality.
- • LongVideoBench [38] is a large-scale benchmark for long-context video-language understanding, evaluating multimodal models on videos up to one hour. It contains 3,763 diverse web videos covering movies, daily life, knowledge, and news, with 6,678 human-annotated multiple-choice questions. Video durations range from 8 seconds to 60 minutes. Its core innovation is the "Referring Reasoning" paradigm, embedding referring queries to locate relevant segments and requiring both precise retrieval and coherent contextual reasoning.
- • LongVT [43] is a benchmark for long-video open-domain question answering, containing 244 long videos and 1,280 QA pairs verified through manual review. The average video duration is approximately 1,688 seconds, with most videos (71.84%) in the 15-30 minute range and 28.16% exceeding 30 minutes. Its core design features a "needle-in-a-haystack" setting where supporting evidence exists only in narrow time windows, effectively evaluating models’ abilities to locate and reason about fine-grained information within long videos.

### C Hyperparameters

We detail the hyperparameters used in our training in Table 9 and Table 10. During agentic RL training, we set α = 0.8, β = 0.15, γ = 0.05, with a sampling frame rate of 1, a maximum number of frames set to 256, and a maximum single-frame resolution set to 112896. During both SFT and RL training, LLaMA-Factory and verl automatically inject timestamps for videos, while during inference, we manually add corresponding timestamps to each frame.

Table 9: Key hyperparameters for SFT.

Name Value Finetuning type Full

Freeze vision tower True Freeze multi-modal projector False Freeze language model False

Cutoff len 16384 image/video max pixels 112896 Video FPS 1.0

Video max length 256 Batch size per device 1

Gradient accumulation steps 2 Learning rate 1.0e-5

LR scheduler type cosine Warmup ratio 0.1

Epochs 1.0

Table 10: Key hyperparameters for RL.

Name Value

Algorithm GRPO Max tool rounds 5

Agent loop Tool agent

Rollout num 8 Train batch size 32 Mini batch size 8

Micro batch size per GPU 1

Learning rate 1.0e-6 KL loss coefficient 0.001 Entropy coefficient 0.0 Max prompt length 16384

Max response length 4096 Total epochs 1 GPU memory utilization 0.8

### D Limitations and Social Impacts

While VideoSeeker demonstrates excellent performance on visual-prompt-driven video understanding tasks, it still has some limitations: First, our data construction pipeline relies on LLaVA-Video [48] as the source, which means the generated data may inherit the domain bias and imbalance issues present in that dataset. On the positive side, VideoSeeker has the potential to enhance accessibility of video content, helping visually impaired users understand video content through intuitive visual prompts. However, similar to other vision models, the outputs may reflect biases in the training data, and we recommend thorough evaluation before applying it to critical scenarios.

- E Training Curves See Figure 6.

[Figure 140]

[Figure 141]

Figure 6: RL training curves.

- F Case Study See Figure 7 and Figure 8.
- G Prompts In this section, we illustrate all the prompts used in our paper.

#### G.1 Text Filtering Prompt

This prompt performs rapid pre-screening of QA samples to remove questions unsuitable for visual prompting (e.g., camera movements, scene backgrounds, counting). See Figure 9.

#### G.2 Video Verification Prompt

This prompt guides five-step reasoning with the video: target extraction, uniqueness judgment, temporal localization, QA rewriting, and visual prompt type recommendation. See Figure 10.

#### G.3 Rendering and Rewriting Prompt

This prompt replaces target descriptions with generic visual prompt references, ensuring questions cannot be answered without visual prompting. See Figure 11.

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

Question: What did the child in the detection frame do before jumping on the adult?

[Figure 180]

- (A) Pick up things from the ground.
- (B) Sitting on the ground.
- (C) Took off the clothes. … Correct Answer: (A)

<think> The user is asking about the specific action performed by the child highlighted in the detection frame before they jumped on the adult. First, I need to examine the visual prompt frame to identify the target subject. In the provided image, a child wearing a white shirt and brown bib is seen standing on the stairs, while an adult is seated on the sofa. To accurately determine the behavior preceding the "jumping" action, I will call the visual prompt tool to confirm the child's identity and location, then proceed to analyze the temporal sequence of the video. </think>

[Figure 181]

[Figure 182]

<tool_call> {"name": "view_visual_prompt", "arguments": {"frame_path": "NEXTVideo/4151403382_35.jpg"}} </tool_call>

<think> By observing the visual prompt, I have locked onto the target child in the white shirt. Now, I need to analyze the dynamic process leading up to the jump. I will crop the video segment covering the child’s movement from the stairs to the sofa to capture the exact behavioral transition. Watching the video stream reveals that after standing on the stairs, the child performs a distinct bending motion to retrieve several objects—identified as books—from the floor. The child then holds these items while walking toward the sofa, immediately before the final action of leaping onto the adult. </think>

[Figure 183]

[Figure 184]

<tool_call> {"name": "crop_video", "arguments": {"video_path": "NExTVideo/4151403382.mp4", "start_time": 30, "end_time": 45}} </tool_call>

<think> Based on the comprehensive video analysis, the target child follows a clear behavioral sequence: initially standing on the stairs, then bending down to retrieve books from the ground, walking toward the sofa with the items, and finally jumping on the adult. Therefore, the task performed immediately before the jump was picking up things from the ground. Options B (sitting), C (taking off clothes), and D (putting on clothes) are inconsistent with the observed actions. </think> <answer>A</answer>

[Figure 185]

- Figure 7: Case Study 1. The model invokes tools to proactively perceive instances and retrieve video segments, enabling instance-level video understanding tasks.

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

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Question: What color is the top of the woman pointed by the red arrow?

[Figure 224]

(A) Green. (B) Red. (C) Blue. … Correct Answer: (A)

<think> The question requires identifying the color of a specific clothing item worn by a person marked with a red arrow. To provide an accurate answer, I first need to examine the visual prompt frame to locate the red arrow and identify the target individual among the subjects in the scene. By invoking the visual prompt tool, I can clearly observe the spatial layout and the specific person being pointed out, which is essential for determining the correct visual attribute. </think>

[Figure 225]

<tool_call> {"name": "view_visual_prompt", "arguments": {"frame_path": "MLVU/en_tv_24/en_tv_24.mp4_20250105_132503.472.jpg"}} </tool_call>

[Figure 226]

<think> After examining the visual prompt, the red arrow is clearly pointing to a woman standing within the scene. Focusing on her attire as indicated, her top exhibits a distinct green hue. The lighting in the scene is sufficient to confirm that the color is vibrant and consistent with the "Green" category. Comparing this observation with the available options, it is evident that the attribute matches the first choice. </think> <answer>A</answer>

[Figure 227]

- Figure 8: Case Study 2. The question only requires visual cue information, so the model adaptively invokes only the visual cue tool, avoiding unnecessary tool calls.

Text Filtering Prompt

### System Prompt:

###### Task Definition:

You are a data filter for visual prompting QA construction. Given ONLY the text of a question-answer pair (no video), quickly determine whether the question is potentially suitable for a visual-prompted QA task — i.e., whether it targets a specific, concrete visual entity (person / object / animal) that could in principle be highlighted by a bounding box, arrow, or contour on a video frame.

Reject (is_valid: false) — clear disqualifiers Reject if the question clearly falls into ANY of the following types:

- • Camera / cinematography: asks about camera movement, angle, zoom, transition, or how the video is shot. e.g., “How does the camera move?”, “What is the initial camera view?”
- • Scene / background / setting: asks about the overall scene, background, environment, setting, or atmosphere — not a specific foreground entity. e.g., “What can be seen in the background?”, “What is the primary setting of the video?”

- • Overall / generic activity: asks what the video is mainly about, without targeting a specific entity. e.g., “What is the main activity in the video?”, “What is the main focus?”
- • Counting: asks how many of something, rather than targeting one specific entity. e.g., “How many shelves are visible?”, “How many children are there?”
- • Abstract / non-visual: asks about abstract concepts, emotions, reasons without referencing a concrete visual entity; or targets non-visual properties (sound, smell, text labels, brand names on packaging). e.g., “What indicates the cooking process?”, “Which brand is on the jar?”
- • Lighting / color of environment: asks about ambient lighting changes or the color of background elements (walls, countertops) rather than a specific object.

Accept (is_valid: true): Accept if the question appears to target a specific, concrete visual entity:

- • A person described by clothing, role, or action: “the person in the gray hoodie”, “the worker with the helmet”
- • A specific named tool or object: “the wooden spoon”, “the blue measuring cup”, “the needle”
- • A specific animal: “the dog”, “the kitten”, “the calf”
- • A question asking WHERE / WHAT COLOR / WHAT ACTION / WHY about a clearly identifiable entity

When in doubt → accept (is_valid: true). The downstream video analysis performs stricter uniqueness checking. Output ONLY valid JSON, no extra text. Output Format:

Accept: {"is_valid": true} Reject: {"is_valid": false, "reason": "one short sentence"}

### User Prompt:

Judge whether the following QA pair targets a specific visual entity suitable for visual prompting (text-only, no video). Here is the question: {question} Here is the options: {options} Here is the answer: {answer}

Output JSON only.

Figure 9: Prompt of Text Filtering.

Video Verification Prompt

### System Prompt:

You are a video understanding expert specializing in visual prompting data construction.

Your task: Given a video and an existing question-answer pair about the video, analyze whether the question targets a specific, uniquely identifiable object/person in the video, and if so, produce structured metadata for constructing visual-prompted QA data.

###### Definitions:

- - Target: The primary object or person that the question is asking about.
- - Uniqueness: The target must be visually distinguishable from all other entities in the video. If multiple similar entities exist (e.g., “a climber” when there are several climbers), the target lacks uniqueness unless the description includes differentiating attributes (e.g., “the climber in the red jacket”).
- - Visual Prompt: A visual annotation (e.g., rectangle, contour, arrow, etc.) overlaid on a video frame to unambiguously indicate the target, replacing textual descriptions. Workflow:

- Step 1: Target Extraction & Uniqueness Judgment Identify the target entity from the question. Determine:

- • Is the question about a specific visual entity (person/object/animal)? Questions about abstract concepts, events, scenes, or the whole video are NOT target-specific.
- • Is this target uniquely identifiable in the video? Watch the video and verify that no other entity could be confused with the target.

Output is_valid: false if either condition fails, with a brief reason. If ambiguity exists, prefer rejecting (is_valid=false) over guessing. Examples:

- - “What colour shirt is the girl playing the violin wearing?” → valid if exactly one such girl is identifiable.
- - A tag like “climber scaling vertical rock face” → invalid if multiple climbers appear without disambiguation.

- Step 2: Generate Target Tag Produce a concise English tag (3-10 words) that:

- • Uniquely identifies the target among all entities in the video
- • Describes stable visual attributes (clothing, color, size, position, species, etc.)
- • Is suitable as a text prompt for a detection/segmentation model (e.g., SAM3)

- Step 3: Temporal Localization Identify the primary continuous time window where the target is visible and relevant to the question:

- • If the target appears in multiple segments, select the single most relevant segment
- • Timestamps in seconds, rounded to 1 decimal place

- Step 4: QA Rewriting Rewrite the question by replacing the target’s textual description with a visual prompt <vp>:

- • Original: “What does the man playing the drums do with his feet?”
- • Rewritten: “What does <vp> do with his feet?”

For answer options: only rewrite options that explicitly mention the target; leave others unchanged.

- Step 5: Visual Prompt Type Recommendation Select the most appropriate visual prompt type for this target from:

- • rectangle: Bounding box. Best for well-bounded, non-overlapping targets.
- • mask_contour: Outline of segmentation mask. Best for irregularly shaped or partially occluded targets.
- • ellipse: Elliptical highlight. Best for faces or compact targets.
- • triangle: Triangular highlight around or pointing to the target.
- • scribble: Freehand scribble over the target region.
- • point: Single dot on target. Best for very small targets.
- • arrow: Arrow pointing to target. Best for small or point-like targets.
- • set_of_mark: Numeric label on target (e.g., “the person marked with 1”). Best when multiple targets need simultaneous disambiguation.

Choose based on target size, shape, and scene complexity. Output Format: Output ONLY valid JSON, no extra text. Here is a success sample: {

"is_valid": true, "reason": null, "sample_id": "<copy from input if provided, else null>",

"target_description": "the original textual description of the target", "tag": "concise unique English tag for detection/segmentation", "timestamp": {

"start": 2.5, "end": 18.3

}, "rewritten_question": "question with target replaced by <vp>", "rewritten_options": ["A. option1", "B. option2", "..."], "rewritten_answer": "the correct answer (unchanged)", "visual_prompt_type": "rectangle"

}

When is_valid is false, only output: {

"is_valid": false, "reason": "explanation of why this QA is not suitable"

}

### User Prompt:

Here is the video: {video} Here is the question: {question} Here is the options: {options} Here is the answer: {answer} Analyze this QA pair following your workflow. Determine if the question targets a uniquely identifiable entity in the video, and if so, produce the complete structured output.

Figure 10: Prompt of Video Verification.

Rendering and Rewrite Prompt

### System Prompt:

You are an expert dataset writer for visual prompted video QA. Your job is to rewrite QA text so that answering requires the visual prompt on the frame, not the target’s original textual attributes.

###### Context:

- - Upstream data already contains: question, options
- - The placeholder <vp> marks where the target reference should be replaced by a visual prompt description. Core objective:
- - Replace <vp> with a natural visual prompt phrase.
- - Keep semantic meaning unchanged. Hard constraints:
- - Do NOT change factual intent of the question.
- - Do NOT alter option semantics, order, labels, or correctness.
- - If an option does not contain <vp>, keep it exactly the same (except minimal grammar smoothing if absolutely necessary).
- - Preserve language style and fluency.
- - Never reveal concrete target identity in rewritten text.

Visual prompt phrase policy: Use generic, prompt-driven references:

- • rectangle → “the target in the highlighted box”
- • mask_contour → “the target outlined by the contour”

- • ellipse → “the target inside the ellipse”
- • triangle → “the target indicated by the triangle marker”
- • scribble → “the target under the scribble mark”
- • point → “the target marked by the point”
- • arrow → “the target indicated by the arrow”
- • set_of_mark → “the target marked with number {n}” (if number unknown, use “the numbered target”)

###### Options handling:

- - Rewrite rewritten_answer only when it contains <vp>.
- - Otherwise keep it unchanged. Output format:

{

"question_refined": [question], }

### User Prompt:

Please rewrite this sample for visual prompt dependency. Here is the visual prompt type: {visual_prompt_type} Here is the question: {question} Here is the options: {options} Please rewrite the question and options according to the system instructions.

Figure 11: Prompt of Rendering and Rewriting.

### NeurIPS Paper Checklist

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes] Justification: The abstract and introduction clearly state our contributions. Guidelines:

- • The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

##### 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss limitations in Appendix D. Guidelines:

- • The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate “Limitations” section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A] Justification: This paper does not present theoretical results or formal proofs

Guidelines:

- • The answer [N/A] means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes] Justification: We fully disclose all key information required to reproduce the experimental results in the paper. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] Justification: The relevant datasets, code and models will be released publicly upon publication. Guidelines:

- • The answer [N/A] means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes] Justification: We provide complete details on all training and testing configurations, with detailed hyperparameters reported in Appendix C. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [N/A] Justification: All evaluations use a consistent sampling temperature of 0. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).

- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- • If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes] Justification: We report the computational resources in Section 4. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: The research conforms to the NeurIPS Code of Ethics. Guidelines:

- • The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes] Justification: We discuss potential positive and negative societal impacts in Section D. Guidelines:

- • The answer [N/A] means that there is no societal impact of the work performed.
- • If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.
- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [Yes] Justification: VideoSeeker is an academic research project trained on publicly available datasets. Guidelines:

- • The answer [N/A] means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: We explicitly cite and comply with the licenses and usage terms of all datasets and assets used. Guidelines:

- • The answer [N/A] means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes] Justification: All new assets are introduced and documented in the paper. Guidelines:

- • The answer [N/A] means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A] Justification: VideoSeeker does not involve crowdsourcing nor research with human subjects. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A] Justification: VideoSeeker does not involve research with human subjects. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes] Justification: We only use LLMs for writing assistance, editing, and formatting, consistent with the NeurIPS policy on LLM usage. Guidelines:

- • The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.

