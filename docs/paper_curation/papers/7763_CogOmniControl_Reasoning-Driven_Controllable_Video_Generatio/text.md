# arXiv:2605.19995v1[cs.CV]19May2026

COGOMNICONTROL: REASONING-DRIVEN CONTROLLABLE VIDEO GENERATION VIA CREATIVE INTENT COGNITION

Hongji Yang1,*, Songlian Li2,*, Yucheng Zhou1, Xiaotong Zhao2 Alan Zhao2, Chengzhong Xu1, Jianbing Shen1,

1SKL-IOTSC, CIS, University of Macau 2Online-Video BU, Tencent *Equal contribution. Corresponding author.

ABSTRACT

Recent diffusion models achieve strong photorealism and fluency in video generation, yet remain fragile under abstract, sparse or complex conditions, leading to poor performance in professional production workflows such as storyboard sketches and clay render conditions. Existing video generation models, either inject conditions through adapters or couple a generic vision-language model (VLM) within a diffusion backbone, leaving a capability gap and failing to produce the videos that align with the user’s creative intent. We present CogOmniControl, a reasoning-driven framework that factorizes controllable video generation into creative intent cognition and generation. Specifically, we train a specialized CogVLM using authentic anime production data. Compared to generic VLMs, it generates more professional and clear outputs, accurately cognizing user creative intent from sparse and abstract conditions and tuning these cues into dense reasoning output. Besides, CogOmniDiT unifies the controls from various conditions through in-context generation and is aligned to the CogVLM reasoning outputs via reinforcement learning. Furthermore, leveraging CogVLM’s robust capability in guiding video generation, we release its potential in planning specific evaluators and enable a Best-of-N selection for the generated videos. This integration transforms the entire framework into a closed-loop “harness-like” architecture. We further introduce CogReasonBench and CogControlBench, built from professional workflows data that carry genuine creative intent rather than simulated ones. Experiments on two benchmarks show that CogOmniControl surpassed the existing open-source models. The project website: https://um-lab.github.io/CogOmniControl/.

1 INTRODUCTION

Recent advances in diffusion-based video generative models (Hong et al., 2022; Yang et al., 2024; HaCohen et al., 2024; Wan et al., 2025) have pushed text-to-video generation to a level of photorealism and motion fluency. Current research (Jiang et al., 2025; Pan et al., 2026) is moving toward omni-level controllable generation, pursuing a single system to support multimodal inputs, professional intent conditions and abstract constraints. Inspired by the powerful multimodal understanding capabilities of VLMs, these frameworks (Tan et al., 2025b; Yang et al., 2026; Pan et al., 2026) attempt to employ VLMs to identify and correlate different condition inputs and then cognize the creative intents to infer coherent control signals. However, video generation still faces the key challenges: ① Cognitive Gap: When confronted with complex or even conflicting multimodal control signals in professional workflows, current VLMs struggle to fully comprehend the underlying creative intent. Consequently, they fail to formulate reasonable generation plans grounded in domain-specific creative knowledge. ② Alignment Gap: It remains an open question whether the outputs of VLMs under abstract conditions are properly aligned with the generated videos. Besides, the adoption of reasoning output from generic VLM also brings additional noise (Yang et al., 2026; Chen et al., 2026) for the generation. As shown in Fig. 1, it remains challenging for controllable video generation models to understand abstract conditions, infer creative intent, and then generate correct video outputs.

###### Adapter-based video generation (e.g., VACE)

##### Condition Inputs

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

T5 Text Encoder

Ref Image

DiT

ControlNet / Adapter

[Figure 6]

Bad quality Wrong following Identity drift Ghosting Intent missing

General Reasoner driven Video Generation (e.g., VINO, OmniWeaving)

[Figure 7]

What is the connection between them?

Text Prompt

[Figure 8]

[Figure 9]

[Figure 10]

A woman in a flowing blue dress walks through a forest, holding her dress with both hands. She turns her head to look around, appearing concerned. She continues walking, then turns her body to face the camera, looking down and clasping her hands together…

Generic Thinking VLM

DiT

Intent misunderstanding Wrong visual effects Fail to follow more abstract control

(e.g., Qwen2.5-VL-Thinking)

Control Video (storyboard)

Ours: CogOmniControl

|[Figure 11]|
|---|

|[Figure 12]|
|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

CogVLM

<think>

###### CogOmniDiT

Frame 1 Frame 2

- 1. Visual Consistency
- 2. Action smoothness
- 3. Visual special effects
- 4. Static-to-dynamic
- 5. Creative intent

|[Figure 18]|
|---|

|[Figure 19]|
|---|

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Multimodal inputs

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Cond. Understand Intent Cognition

[Figure 30]

[Figure 31]

[Figure 32]

Identity kept Smooth motion Correct creative intent High quality

Frame 3 Frame 4

[Figure 33]

[Figure 34]

|[Figure 35]|
|---|

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

##### Real-World Professional Workflows Assets Director Creating… Production

Creative Intent

- Figure 1: The motivation of our CogOmniControl. The adapter-based methods and video generation models with generic VLM fail to generate the final video from the given condition.

To bridge this gap between the abstract condition and creative intent, we present CogOmniControl, which includes the CogVLM to cognize the creative intent and CogOmniDiT to transform the intent into video output. To enable VLMs to understand abstract conditions and creative intent for more efficient reasoning, we employ a combination of Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT). This process transforms a generic VLM into a specialized CogVLM, equipped with deeper controllable video generation knowledge to more effectively drive video generation models. By incorporating high-level features from CogVLM and conditional inputs, CogOmniDiT achieves more robust controllable generation with abstract and sparse conditions. Unlike previous approaches that simulated user intent from existing videos, our dataset was collected from real-world professional workflows, including the storyboard, clay rending video, and their corresponding video, which represent genuine creative intent from initial sketches to final production. Drawing on LLM harness engineering (Gao et al., 2024; Lee et al., 2026; Lin et al., 2026), CogVLM goes beyond specifying the generation for the DiT, it can also identify the required evaluators derived from its reasoning through the conditions. This enables the model to pick suitable evaluators for optional Best-of-N selection, establishing a fully integrated closed-loop pipeline in video generation. We also define a suite of tools as evaluators, including both VLMs and specialized pre-trained models in the framework. To further evaluate the understanding of abstract conditions and the quality of video generation from both VLM and video generation models, we introduce two benchmarks, CogReasonBench and CogControlBench, to validate our proposed method. Experimental results on two benchmarks demonstrate that our model outperforms existing open-source models. Our contribution can be summarized as follows:

- • We present CogOmniControl, a reasoning-driven framework for controllable video generation. By leveraging professional reasoning to bridge the gap between pixel-level priors and high-level intent, our framework ensures structural integrity and creative intent alignment, particularly in sparse and abstract controllable generation scenarios.
- • We propose CogVLM and CogOmniDiT. CogVLM understands abstract and sparse conditions, infers the creative intent, and translates multimodal cues into dense logical outputs. CogOmniDiT integrates diverse control signals with the high-level semantic features from CogVLM, faithfully synthesizing videos aligned with the inferred intent.
- • We further extend CogOmniControl into a closed-loop Reasoning-Generation-Verification system through an evaluator harness emitted by CogVLM. In a single forward pass, CogVLM produces a solution as well as the evaluator, which scores the candidates in Best-of-N selection.

- • To evaluate the conditions understanding and abstract reasoning of VLM and instruction following of controllable video generation, we construct two new benchmarks, CogReasonBench and CogControlBench, for CogVLM and CogOmniControl, respectively. These benchmarks are collected from human-drawn storyboards or clay render videos during real-world professional animation productions.

- 2 RELATED WORK

Video Generation. With the rapid development of image (Rombach et al., 2022; Podell et al.,

- 2023; Peebles & Xie, 2023; Labs, 2024) and video (Hong et al., 2022; Yang et al., 2024; HaCohen et al., 2024; Kong et al., 2024; Wan et al., 2025) generative models, diffusion models have been proven to produce high-fidelity visual content and are widely applied in diverse domains, including artistic creation, animation production, visual special effects and game development (Brooks et al.,
- 2024; Midjourney, 2026). To faithfully realize specific creative intentions, conditional guidance has evolved from abstract natural language to diverse explicit constraints for precise control. Early breakthroughs introduce additional adapter (Zhang et al., 2023; Ye et al., 2023; Li et al., 2025b; Yang et al., 2025a; Guo et al., 2023; Zhao et al., 2023; Jiang et al., 2025; Guo et al., 2024; Lin et al.,

- 2024; Liu et al., 2025a) to support condition injection without compromising the original generative quality. However, these adapter-based paradigms often exhibit limited flexibility in handling diverse conditions, particularly in those that are non-pixel-aligned or serve merely as visual references. To achieve omni-level, OmniGen (Xiao et al., 2025) and OmniGen2 (Wu et al., 2025a) integrated autoregressive transformers with diffusion to realize a unified generation. OmniControl (Tan et al., 2025a) and UNO (Wu et al., 2025b) introduced in-content visual generation.

The omni-level generation has also been extended into the video domain, the emergence of proprietary models, such as Seedance2.0 (Seedance et al., 2026), Kling-O1 (Team et al., 2025), Sora2 (OpenAI, 2025), Vidu (AI, 2026), Veo3 (Google, 2025), has established a transformative vision for omni-level video generation. However, current open-source models still fail to realize robust unified video generation. VACE (Jiang et al., 2025), UniVideo (Wei et al., 2025), and VINO (Chen et al., 2026) attempted to achieve omni-level generation by integrating various basic tasks, they often lack a deep understanding across diverse conditions. In contrast, OmniWeaving (Pan et al., 2026) successfully incorporated the abstract reasoning of VLM into the video diffusion model to execute complex multimodal compositional tasks. However, the reasoning processes of its LLM components have not yet undergone professional evaluation or systematic benchmarking on creative intentions, leaving the model without sufficient guidance when tackling more challenging tasks.

Reinforcement Learning for Visual Generation. Inspired by the success of LLM fine-tuning using RL from human feedback, RL for visual generation is gaining momentum. For example, DDPO (Black et al., 2024), Diffusion-DPO (Wallace et al., 2024) and DPOK (Fan et al., 2023) introduced Direct Preference Optimization (Rafailov et al., 2023) into T2I Diffusion to align with human preference. Motivated by DeepSeep-R1 (Guo et al., 2025) using GRPO (Shao et al., 2024) to provide more dense rewards through computing relative rewards in a sample group, Flow-GRPO (Liu et al., 2025b) and DanceGRPO (Xue et al., 2025) extended this paradigm into flow-matching models (Liu et al., 2022) by transforming the deterministic ODE formulation into a stochastic SDE, thereby enabling effective online exploration and policy alignment. Beyond this, several GRPO-based studies (Wang et al.,

- 2025; Li et al., 2025a; He et al., 2025b; Yang et al., 2025b) have focused on refining reward design to enhance performance in visual generation.

- 3 METHOD

- 3.1 COGOMNICONTROL FRAMEWORK

In this section, we present the overall framework of CogOmniControl, a robust pipeline that accommodates diverse types of control conditions (e.g., pose, depth, lineart, storyboard sketch, clay render) to facilitate high-quality controllable video generation. As illustrated in Fig 2, the proposed method consists of two key modules, CogVLM for reasoning and CogOmniDiT for generation.

###### Inference Pipeline

[Figure 44]

[Figure 45]

[Figure 46]

The control video begins with a wide shot of distant mountains and morning sunlight, followed by a rapid cut to a close-up of the character’s face, showcasing sharp, focused eyes during flight. The reference image defines the protagonist‘s specific image and the video’s style… The protagonist flies fluently through the air, with clothes fluttering naturally and lighting dynamically changing in response to the distant sunrise…\n [Tools] [ID Consistency Evaluator] [Reference Style Consistency Evaluator] [Physical Dynamics Evaluator] [Motion Smoothness Evaluator]

VLM feature Text Prompt

Ref Image Control Video

Noise

[Figure 47]

Best-of-N

“

[Figure 48]

[Figure 49]

1 2

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

A man in the sky. He has black hair and fair skin, dressed in white and grey robes adorned with gold patterns, ….

[Figure 58]

3 4

[Figure 59]

[Figure 60]

CoT feature

Visual feature

[Figure 61]

”

Evaluators

VAE Encoder

VAE Encoder

[Figure 62]

Connector

umT5

Generated Video

</think> token token token … <eos>

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

|2<br><br>3 4<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]|
|---|

token

1

Noisy Latent 𝒁𝒕

Control Latent 𝒁𝒄𝒕𝒓𝒍

T5 Embeddings

VLM Embeddings

Ref Latent 𝒁𝒓𝒆𝒇

CogVLM

…

Concat

|2<br><br>3 4<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]|
|---|

1

[Figure 83]

“

[Figure 84]

[Figure 85]

CogOmniDiT

A man in the sky. He has black hair

###### Input sequence = [ ]

|𝒁𝒕 ∥ 𝒁𝒓𝒆𝒇 ∥ 𝒁𝒄𝒕𝒓𝒍 ∥ VLM embeds|
|---|

”

Control Video Ref Image Text Prompt Instruction

CogVLM Training CogOmniDiT Training

[Figure 86]

[Figure 87]

Stage-1: Supervised Fine-tuning Stage-2: Reinforcement Fine-tuning Stage-1: Supervised Fine-tuning Stage-2: Reinforcement Fine-tuning

###### Inputs

[Figure 88]

[Figure 89]

[Figure 90]

###### Update

Holistic Reward Acc. Reward

[Figure 91]

###### CogOmniDiT (LoRA🔥)

[Figure 92]

[Figure 93]

Reasoning Outputs & Evaluators

Control Video

Binary Question Yes or No

MM Intent, Physics, Integrity, Motion

Input Sequence

1 2

[Figure 94]

[Figure 95]

𝑅 𝑅 … 𝑅

Samples

3

4

[Figure 96]

[Figure 97]

[Figure 98]

Compute Rewarding Advantage

###### CogVLM (LoRA🔥)

[Figure 99]

CogOmniDiT (LoRA🔥)

|Ref Image & Prompt|
|---|

[Figure 100]

Video#1 Video#2 Video#3 Video#4

###### …

###### …

𝐴 𝐴 𝐴 𝑦 𝑦 𝑦

Reward Function

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Vid token img token txt token

Predict gradient

Condition Following Reward

Video Quality Reward

Update Policy

Sampling

###### Target Video

Vision Encoder Tokenizer

[Figure 107]

1 2

|3 4<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]|
|---|

Predicted CogVLM (LoRA🔥) velocity

Rewards: [0.43, 0.96, 0.37, 0.62]

[Figure 112]

[Figure 113]

“

[Figure 114]

A man in the sky. He has black hair

[Figure 115]

”

Policy Optimization

Control Video Ref Image Text Prompt

- Figure 2: The overall framework of the proposed CogOmniControl. During inference, CogVLM outputs reasoning results based on the given conditions, along with optional evaluator tools. Subsequently, the features from the last layer of CogVLM are concatenated with other latents as the inputs of CogOmniDiT to generate the final result. This process can be repeated multiple times, employing Best-of-N filtering using the evaluator selected by CogVLM. The bottom-left and bottom-right sections illustrate the training processes for CogVLM and CogOmniDiT, respectively.

The input condition set C we define is formulated as a multimodal tuple comprising Control Video Vctrl, Reference Image Iref and Textual Description Tdesc, which can be formatted as:

C = {Vctrl,Iref,Tdesc}, (1)

Control video provides temporal and spatial cues (e.g., trajectories and layouts), the reference image offers visual appearance or spatial references, and the textual description provides global semantic guidance for the entire generation process.

The core idea of CogOmniControl is to integrate the reasoning of VLM into the controllable generation model. We formalize the generation process as a conditional mapping F: V ← {Vctrl,Iref,Tdesc}. Then the whole generation process of CogOmniControl can be formatted as:

, (2)

## P(V | C) = P(V | R,C)

## ·P(R | Vctrl,Iref,Tdesc)

Generation

Reasoning

- 3.2 COGVLM: COGNIZING CREATIVE INTENT FROM MULTIMODAL CONDITIONS

Given a variety of conditions, we observe that they play distinct roles during the creative process. For example, some conditions (i.e., reference images) provide visual information, pose and depth conditions impose strict spatial layouts, and some conditions (i.e., storyboard) may carry additional creative intent. However, previous controllable video generation models often treat input conditions as direct pixel-level constraints and fail to align with the creative intent, particularly when conditions exhibit significant conflicts or semantic discrepancies. Besides, video generative models primarily lack a deep understanding of the diverse input conditions and the underlying correlations between

- them, making it difficult to coordinate the final generation.

Therefore, we propose CogVLM to perform visual reasoning on how to generate the final video that aligns with the creative intent from different conditions. CogVLM plays the role of the professional director, which ingests multi-modal drafts to formulate explicit production schemes. Specifically, we

prompt the VLM to interpret the given conditions and then identify the corresponding cross-modal entities. By reasoning through conflicting constraints and extrapolating implicit details. For example, given ‘raining’ in the text and ‘standing water’ in the reference image, the VLM can infer emergent visual features like ‘rippling effects on the water’s surface’ and then generate a dense response.

Training. To empower CogVLM with professional-grade insight, we employ a two-stage training strategy, SFT and RFT. For RFT, we design a Holistic Reward and Fact Verification Reward based on LLM-as-a-Judge (Chen et al., 2024) to optimize the fine-tuned model.

The holistic reward function Rholistic is to assess the qualitative alignment of the reasoning output R with respect to the input conditions C:

Rholistic =

wk · VLMk(R,C), (3)

k∈K

where K = {intent,phys,info,dyn} represents the four critical dimensions: Creative Intent, Physical Plausibility, Information Integrity, and Motion description. The function VLMk(·) denotes the normalized score assigned by the judge model specifically for dimension k, weighted by wk.

To ensure the reasoning is grounded in factual accuracy and avoid hallucinations, we implement the Accuracy Reward function Racc. For each condition set C, the teacher model is asked to return N binary questions {q1,q2,...qN}. Then, the judge model verifies whether the reasoning output R satisfies these atomic facts qi:

N

1 N

VLM(R,qi), (4) This reward mechanism transforms subjective narrative evaluation into a verifiable accuracy metric.

Racc =

i=1

- 3.3 COGOMNIDIT: UNIFIED VIDEO DIFFUSION TRANSFORMER

To enable different condition inputs, we present CogOmniDiT, where heterogeneous conditions and noisy latents are processed within a unified sequence. Leveraging the powerful in-context learning (Zhou et al., 2024; 2026) of the transformer backbone, the noisy latent and various conditions can model themselves and others within the self-attention. This ensures the conditions are effectively injected into the latent, facilitating precise controllable video generation.

Input Sequence = Concat(Zt,Zref,Zctrl,EmbVLM), (5)

where Zt, Zref, and Zctrl denote the noisy latent, ref image latent and control video latent. The EmbVLM is the VLM embedding after the connector.

While the preceding stages establish a strong foundation for controllable generation, the complex nature of “reasoning-driven” control often leads to a creative intention gap, where the CogOmniDiT may struggle to faithfully translate reasoning output into pixel-level dynamics. To bridge this, we perform RFT for CogOmniDiT, specifically designed to enforce rigorous adherence to both pixel-level conditions and high-level reasoning results.

Rvisual =

m∈M

wm · VLMm(V,R,C), (6)

where M = {condition following,video quality} represents the two critical dimensions: condition following and video quality. The RFT is performed on lower resolution and inference in high-resolution due to the scaling capability of video diffusion transformer (Ping et al., 2025; 2026).

- 3.4 CLOSED-LOOP VERIFICATION WITH EVALUATOR HARNESS

Conventional best-of-N selection for heterogeneous video generation relies on a fixed set of evaluators applied uniformly across all samples. In practice, however, each controllable generation carries a distinct intent, and different types of conditions contribute unequally to the final outcome. For example, identity consistency is irrelevant for generations that do not involve any character or identity. As a result, effective test-time scaling calls for an evaluator set that is adaptively selected per input rather than fixed in advance. Since CogVLM has been trained to understand conditions and infer

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Clay render

[Figure 120]

Storyboard-to-video

[Figure 121]

Control Video

[Figure 122]

Control Video

[Figure 123]

|2<br><br>4<br><br>[Figure 124]|
|---|

|2<br><br>4<br><br>[Figure 125]|
|---|

|2<br><br>4<br><br>[Figure 126]|
|---|

|2<br><br>4<br><br>[Figure 127]|
|---|

|2<br><br>4<br><br>[Figure 128]|
|---|

Prompt Ctrl Video Ref Image Trg Video

Semantic Alignment

Target Video

[Figure 129]

Target Video

[Figure 130]

[Figure 131]

[Figure 132]

Gemini3.1-Pro

|2<br><br>4<br><br>[Figure 133]|
|---|

|2<br><br>4<br><br>[Figure 134]|
|---|

|2<br><br>4<br><br>[Figure 135]|
|---|

|2<br><br>4<br><br>[Figure 136]|
|---|

|2<br><br>4<br><br>[Figure 137]|
|---|

Annotation

[Figure 138]

Reasoning & Video Solution

[Figure 139]

Normal Condition

[Figure 140]

Target Video Ref Image Trg Video Ctrl Video

[Figure 141]

Subject-based

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

|2<br><br>4<br><br>[Figure 147]|
|---|

Depth Extractor

1

Keyframe Extraction

Nano Banana

[Figure 148]

[Figure 149]

3

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Pose Extractor

[Figure 156]

[Figure 157]

Subject Grounding

Qwen Edit

Human Verification & Filtering

|2<br><br>4<br><br>[Figure 158]|
|---|

[Figure 159]

1

[Figure 160]

[Figure 161]

[Figure 162]

Lineart Extractor

3

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

- ① Professional workflows data (e.g., Storyboard | Clay render )
- ② General data (e.g., Subject | Pose | Depth )

CogControlBench CogReasonBench

For video generation For VLM reasoning

- Figure 3: The construction pipeline of CogControlBench and CogReasonBench. We include the professional workflows data and general video generation data in the training set and benchmark.

Table 1: The Comparison of CogControlBench with the existing video benchmark.

Benchmark #Size Resolution

Input Modality Data Attributes

VLM-as-a-Judge

Image Multi-Image Video Condition Source Creative Intent

VBench (Huang et al., 2024) 946 480P - VBench++ (Huang et al., 2025) 2384 480P - OpenVE-Bench (He et al., 2025a) 431 720P - TGVE+ (Singer et al., 2024) 1417 480P Synthetic Simulated OpenS2V-Eval (Yuan et al., 2025) 180 720P Synthetic Simulated VACE-Bench (Jiang et al., 2025) 480 480P Synthetic Simulated IntelligentVBench (Pan et al., 2026) 1030 480P Synthetic Simulated

CogControlBench 200 720P Native Raw

how to generate the intended video, it inherently possesses the knowledge to identify appropriate evaluators for the video. Formally, let F denote a fixed video generation model, and we make CogVLM output reasoning R and harness H in a single forward pass:

(R,H) ∼ πCogV LM(·|C), (7)

where πCogV LM denotes the CogVLM. Then, we execute a rollout {V1,V2,...,Vn} = F(R,C), the objective of the harness is to find the output that maximizes the expected final video:

V∗ = arg max

Vi∈{V1,V2,...,Vn}

S(Vi;H). (8)

where the S(·|) denotes the score function based of the defined H. The specific evaluators are adaptively assigned by CogVLM from the [tools] library as it reasons through the generation conditions. Please refer to the Appendix for details of these designed tools.

- 4 BENCHMARK

To further demonstrate the capabilities of CogOmniControl, we curated a new video reasoning and generation benchmark consisting of the storyboard/clay render video and the final videos collected from in-house professional anime production pipelines. This type of data reflects the inherent gap between the abstract condition provided by the user and the raw creative intent in professional production. Additionally, to showcase generalizability of CogOmniControl in controllable video generation tasks, we incorporated a variety of general controllable generation data, including samples from community1 and VACE-Bench (Jiang et al., 2025). Leveraging these data, we built the CogReasonBench to measure the VLM’s ability to cognize creative intent and reasoning, and the CogControlBench to measure the quality and condition following of controllable video generation of the model under the abstract and sparse conditions.

As shown in Fig. 3, for professional workflow data in CogControlbench, we perform manual semantic alignment and annotation to ensure that the control clip and the final clip share the same semantics. For general data, we incorporate reference-to-video by extracting subjects from key frames and

1https://createwithclint.com

Table 2: The results of CogVLM on CogReasonBench.

Models MM Intent Physics Integrity Motion Avg

Qwen3-VL-8B-Instruct 2.480 4.045 3.905 4.420 3.712 Qwen3-VL-8B-Thinking 2.670 3.824 3.829 4.727 3.752 CogVLM (SFT) 3.725 4.445 4.266 4.955 4.343 CogVLM (RFT) 3.985 4.449 4.599 4.959 4.473

- then editing them using Nano-Banana or Qwen-Image-Edit. We also apply condition extractors
- to extract conditions frame-by-frame to make the dataset support general controllable generation. Tab. 3 shows the comparison of CogControlBench with other video generation benchmarks. To align with the high-quality standards of anime production while optimizing for validation efficiency, we curated a set of 200 high-resolution representative samples. This scale is aligned with established high-resolution benchmarks. For CogReasonBench in VLM, we prompt Gemini3.1-Pro (Google) to reason across the input conditions and the target video to formulate the generative solution. To ensure the correctness of the chain of thought and the solution, the whole process is under human verification and filtering.

- 5 EXPERIMENT

- 5.1 EXPERIMENT SETUP

Experiments are conducted on the Qwen3-VL-8B-Thinking (Bai et al., 2025) as the base VLM and Wan2.2-T2V-14B (Wan et al., 2025) as the base DiT with 32 NVIDIA H20 96GB GPUs. For SFT in CogVLM, we employ LoRA (Hu et al., 2022) training with a rank of 16 and an alpha of 64, respectively. The SFT is performed for 3 epochs with a learning rate of 1e-5. For RFT in CogVLM, we train our model with an initial learning rate of 1e-6 for 500 steps. For SFT in CogOmniDiT, we implement a three-stage training strategy using LoRA with a rank of 256. In stage-1, we train only LoRA for in-context generation, and training of the stage-2 introduces freeze CogVLM and a trainable connector. Finally, we perform joint training of the LoRA and connector. For more details, please refer to the Appendix.

- 5.2 METRICS

To comprehensively evaluate the performance of CogOmniControl in controllable video generation, we utilize numeric metrics based on VBench (Huang et al., 2024) and a VLM-as-a-Judge (Zheng et al.,

- 2023) paradigm, employing Gemini 3.1-Pro (Google) as the authoritative evaluator. Our evaluation focuses on two dimensions:

Condition Following. The core of our evaluation lies in whether CogOmniControl faithfully adheres to the creative intent implied by the condition set {Vctrl,Iref,Tdesc}. Unlike traditional methods that treat conditions as isolated constraints, we assess the model’s ability to interpret these multimodal signals as a holistic objective. For this task, the evaluation of multimodal intent alignment is based on the following considerations: whether the model effectively resolves conflicts between disparate conditions, whether it integrates conditions accurately when significant discrepancies exist, and whether it can infer plausible physical properties or dynamic effects based on the association among the conditions. Besides, the evaluation also includes the preservation of the visual information from Iref and the instruction following from Tdesc in this task.

Visual Quality. Visual quality evaluates the aesthetic quality, imaging quality, temporal flickering, motion smoothness and dynamic degree of the generated video inspired by VBench (Huang et al.,

- 2024). Besides, this type of evaluation also provides dimensions on identity consistency and dynamic plausibility.

- 5.3 RESULTS

Quantitative Results. The results of CogVLM via SFT and RFT are shown in Tab 2, generic VLMs (e.g., Qwen3-VL-8B-Instruct and Thinking) fail to cognize creative intent from multimodal inputs, they also underperformed compared to the CogVLM in terms of information integrity and motion description. Furthermore, they struggle to generate accurate descriptions for inferred physical effects.

Table 3: The comparison on CogControlBench. AQ=Aesthetic Quality, IQ=Image Quality, T F=Temporal Flickering, MS=Motion Smoothness, DD=Dynamic Degree, MI=Multimodal Intent, AF=Appearance Follow, SF=Style Follow, CF=Content Follow, DF=Dynamic Follow, MN=Motion Naturalness, IC=Identity Consistency, DP=Dynamic Plausibility.

Speciesist Metrics VLM-as-a-Judge Metrics VLM-as-a-Judge Metrics

Models

Avg

AQ IQ T F MS DD MI AF SF CF DF AQ IQ MN IC DP

Proprietary Models

Kling-3O (Team et al., 2025) 0.571 0.644 0.979 0.987 0.511 3.510 4.205 4.267 2.679 3.526 3.936 3.453 2.465 3.140 3.203 0.704 Seedance2.0 (Seedance et al., 2026) 0.589 0.653 0.980 0.989 0.517 4.110 4.252 4.348 4.412 3.054 4.050 3.731 2.731 3.469 3.494 0.750

Open-Source Models

VACE-Wan2.1 (Jiang et al., 2025) 0.549 0.636 0.975 0.986 0.528 3.421 3.361 3.712 3.886 2.614 3.777 3.680 2.757 3.592 3.330 0.665 VACE-LTX (Jiang et al., 2025) 0.496 0.617 0.980 0.989 0.345 2.807 2.051 1.849 3.377 2.412 2.797 2.588 1.887 2.492 2.299 0.556 VINO (Chen et al., 2026) 0.570 0.581 0.980 0.989 0.280 3.324 3.853 4.020 4.116 2.327 3.855 3.626 2.710 3.341 3.344 0.686 OmniWeaving (Pan et al., 2026) 0.512 0.549 0.976 0.982 0.396 2.630 2.119 2.550 3.963 2.574 3.257 2.941 2.408 3.033 3.000 0.607

CogOmniControl 0.594 0.602 0.978 0.990 0.528 3.588 3.762 4.207 4.239 2.681 3.910 3.594 2.855 3.615 3.596 0.727 CogOmniControl (BoN) 0.594 0.635 0.980 0.990 0.513 3.795 3.905 4.176 4.325 2.714 4.017 3.594 2.769 3.594 3.552 0.733 CogOmniControl (Harness BoN) 0.596 0.637 0.980 0.990 0.531 3.904 3.949 4.217 4.330 2.853 4.028 3.617 2.858 3.644 3.602 0.742

Table 4: The ablation studies of CogOmniControl on CogControlBench.

VLM-as-a-Judge Metrics VLM-as-a-Judge Metrics MI AF SF CF DF AQ IQ MN IC DP

Models

Qwen3-VL-8B-Thinking + CogOmniDiT(SFT) 3.142 3.541 3.916 4.208 2.467 3.822 3.562 2.708 3.616 3.473 CogVLM(SFT) + CogOmniDiT(SFT) 3.397 3.726 4.096 4.298 2.631 3.929 3.541 2.765 3.597 3.638 CogVLM(RFT) + CogOmniDiT(SFT) 3.586 3.761 4.207 4.156 2.773 3.920 3.594 2.778 3.590 3.518 CogVLM(RFT) + CogOmniDiT(RFT) 3.588 3.762 4.207 4.239 2.681 3.910 3.594 2.855 3.615 3.596

Tab. 3 reports the results on CogControlBench. CogOmniControl achieves the highest average score (0.727) among all open-source competitors, surpassing the VINO (0.686) and VACE-Wan2.1 (0.665), while narrowing the gap to the strongest proprietary system Seedance2.0 (0.750) in this task. Better performance improvements can be observed when employing Best-of-N sampling. The N is set to 4. We demonstrate the results using both the full set of evaluators (0.733) and the specific evaluators (0.742) suggested by CogVLM during the inference process. The approach of selecting evaluators adaptively based on the input for Best-of-N yields better performance. This indicates that CogVLM can effectively serve as a harness for the entire framework.

Qualitative Results. The visual results are present in Fig. 4 and Fig. 5, the adapter-based methods (e.g., VACE-LTX and VACE) tend to align with control videos at the pixel-level, resulting in significant artifacts. Besides, these methods cause semantic misalignment, since clay render video consist of sparse and abstract control. In general Reference-to-video task, CogOmniControl remains strong in performance. The generated videos from VACE lack quality and reference following, while VINO produces virtually static outputs that lack meaningful temporal dynamics.

- 5.4 ABLATION STUDIES

The ablation studies primarily focus on how SFT and RFT of the CogVLM and CogOmniDiT impact the final generation quality. It is evident that the CogVLM after SFT, the model’s ability to capture multimodal intent improves significantly. The performance metrics rose from 3.142 to 3.588, compared to the vanilla Qwen3-VL-8B-Thinking. This demonstrates that the VLM specific for the generation model is necessary for controllable generation due to its understanding of various conditions and reasoning on creative intents.

- 6 CONCLUSION

In this work, we presented CogOmniControl, a reasoning-driven framework that bridges the longstanding gap between abstract conditions and faithful controllable video generation. Departing from prior paradigms that either rely on adapter-based condition injection or insert a generic VLM reasoner with a diffusion transformer, CogOmniControl explicitly factorizes generation into cognition and generation. On the cognition side, CogVLM is trained to act as a director that transcribes minimalist multimodal cues into dense, logically grounded production. On the generation side, CogOmniDiT unifies the pixel-level condition and semantic VLM features within a sequence and is further aligned to the reasoning output through reinforcement fine-tuning. Beyond the role of guidance, CogVLM can also provide harness engineering in the whole framework. By selecting the appropriate evaluators adaptively for best-of-N selection, CogOmniControl achieves further improvements in performance. In addition, we curated two benchmarks for CogVLM and CogOmniDiT, respectively.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Control Video

[Figure 172]

Highly cinematic realistic image, depicting a busy East Asian city street in the rain, with a group of pedestrians with varying expressions distributed in the center and foreground. In the foreground is a woman in a blue translucent hooded raincoat, her expression focused. In the middle, a man in a long, fully transparent raincoat is holding up a smartphone to take a picture, and next to him, another man in a purple jacket is also using his phone to record. Among the crowd are an efficient-looking woman in a black leather jacket, a man in a yellow raincoat, and a unique elderly person leaning forward. The background is a typical Chinese street scene, hanging with red and blue neon signs for "Noodle Shop" and "Supermarket", with slightly aged architectural styles. The ground is paved with gray stone bricks, wet and coldly shimmering due to the rain, clearly reflecting the neon lights and the figures of the pedestrians. The sky is drizzling, and the dense rain streaks are clearly visible under the lights. The overall tone is cold, full of story and a suppressed urban atmosphere, with extremely detailed image quality, realistic light and shadow representation, presenting a visual effect similar to movie stills. The camera pans slowly from right to left.

Ref Image & prompt

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

VACE-LTX

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

VACE

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

VINO

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

OmniWeaving

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

CogOmniControl

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Control Video

[Figure 203]

In a whimsical, animated subway car designed to resemble an underwater scene, a young man with short, light brown hair and piercing blue eyes sits on a pink bench. He wears a cozy blue hoodie and dark blue jeans, with black boots completing his casual yet stylish outfit. The subway car is adorned with fish and seaweed decorations, creating a serene aquatic ambiance. Through the large windows, a cityscape with tall buildings and greenery can be seen, contrasting beautifully with the underwater theme inside. As the subway moves, the man turns his head to the left, observing the scenery outside the window, before turning back to face forward. Various fish and seaweed continue to swim gracefully within the car, adding to the enchanting underwater setting. The camera remains steady throughout, capturing the tranquil moment as the man enjoys the unique blend of urban and aquatic environments.

Ref Image & prompt

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

VACE-LTX

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

VACE

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

VINO

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

OmniWeaving

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

CogOmniControl

- Figure 4: The comparison of CogOmniControl with other video generation models in clay render, which is a common intermediate draft stage in animation production. Zoom in for more details.

|[Figure 229]<br><br>1|
|---|

|[Figure 230]<br><br>2|
|---|

|[Figure 231]<br><br>3|
|---|

Control Video

|[Figure 232]<br><br>6|
|---|

|[Figure 233]<br><br>5|
|---|

|[Figure 234]<br><br>4|
|---|

High-quality dark fantasy game cinematic in a top-down isometric perspective featuring a group of burly, muscular Orc or Barbarian warriors with bronze skin and primitive armor trekking along a rugged wilderness path; one warrior pushes a wooden wheelbarrow filled with hay while the terrain of yellow soil and moss gives way to mist-shrouded hills, captured with a slow camera pan to the right.

[Figure 235]

Ref Image & prompt

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

VACE-LTX

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

VACE

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

VINO

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

OmniWeaving

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

CogOmniControl

[Figure 266]

A fluffy cat is walking down a cobblestone street, heading towards the camera. The cat's fur is soft and well-groomed, with long whiskers and bright yellow eyes. It appears to be in mid-stride, with one paw slightly raised. The background features traditional buildings with hanging signs and lanterns, suggesting an Asian setting. There are people in the distance, but they are blurred, drawing attention to the cat. The camera is positioned at ground level, capturing the cat from the front and emphasizing its movement. The lighting is natural and warm, creating a cozy and inviting atmosphere. The scene is in real-life.

Ref Image & prompt

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

VACE-LTX

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

VACE

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

VINO

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

OmniWeaving

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

CogOmniControl

### Figure 5: The comparison of CogOmniControl with other video generation models. Zoom in for more details.

Extensive experiments show that CogOmniControl consistently outperforms state-of-the-art opensource controllable video generators and narrows the gap with strong proprietary systems.

REFERENCES

Vidu AI. Vidu. https://www.vidu.com/, 2026. Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao

Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=YCWjhGrJFD.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yinuo Liu, Yaochen Wang, Huichi Zhou, Qihui Zhang, Pan Zhou, Yao Wan, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. arXiv preprint arXiv:2402.04788, 2024.

Junyi Chen, Tong He, Zhoujie Fu, Pengfei Wan, Kun Gai, and Weicai Ye. Vino: A unified visual generator with interleaved omnimodal context. arXiv preprint arXiv:2601.02358, 2026.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Neural Information Processing Systems, 36:79858– 79885, 2023.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

Google. Gemini-3. URL https://gemini.google.com. Accessed: 15 Dec. 2025. Google. Gemini ai video generator powered by veo 3.1. https://gemini.google/

overview/video-generation/, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pp. 330–348. Springer, 2024.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025a.

Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang, Dacheng Yin, Fengyun Rao, and Bo Zhang. Tempflow-grpo: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025b.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. International Conference on Learning Representations, 1(2):3, 2022.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17191–17202, 2025.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn.

Meta-harness: End-to-end optimization of model harnesses. arXiv preprint arXiv:2603.28052, 2026.

Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025a.

Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet ++: Improving conditional controls with efficient consistency feedback. In European Conference on Computer Vision, pp. 129–147. Springer, 2025b.

Han Lin, Jaemin Cho, Abhay Zala, and Mohit Bansal. Ctrl-adapter: An efficient and versatile framework for adapting diverse controls to any diffusion model. arXiv preprint arXiv:2404.09967, 2024.

Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Xuanjing Huang, Hang Yan, Zhenhua Han, and Tao Gui. Agentic harness engineering: Observability-driven automatic evolution of coding-agent harnesses. arXiv preprint arXiv:2604.25850, 2026.

Feng-Lin Liu, Hongbo Fu, Xintao Wang, Weicai Ye, Pengfei Wan, Di Zhang, and Lin Gao. Sketchvideo: Sketch-based video generation and editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23379–23390, 2025a.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and

transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. Midjourney. Midjourney. https://www.midjourney.com, 2026.

OpenAI. Sora2.0. https://openai.com/index/sora-2/, 2025. Kaihang Pan, Qi Tian, Jianwei Zhang, Weijie Kong, Jiangfeng Xiong, Yanxin Long, Shixue Zhang,

Haiyi Qiu, Tan Wang, Zheqi Lv, et al. Omniweaving: Towards unified video generation with free-form composition and reasoning. arXiv preprint arXiv:2603.24458, 2026.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In International Conference on Computer Vision, pp. 4195–4205, 2023.

Bowen Ping, Chengyou Jia, Minnan Luo, Changliang Xia, Xin Shen, Zhuohang Dang, and Hangwei Qian. Paco-rl: Advancing reinforcement learning for consistent image generation with pairwise reward modeling. arXiv preprint arXiv:2512.04784, 2025.

Bowen Ping, Chengyou Jia, Minnan Luo, Hangwei Qian, and Ivor Tsang. Flow-factory: A unified framework for reinforcement learning in flow-matching models. arXiv preprint arXiv:2602.12529,

2026. URL https://arxiv.org/abs/2602.12529.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Neural Information Processing Systems, 36:53728–53741, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video editing via factorized diffusion distillation. In European Conference on Computer Vision, pp. 450–466. Springer, 2024.

Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14940–14950, 2025a.

Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025b.

Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-omni technical report. arXiv preprint arXiv:2512.16776, 2025.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You

Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen

- Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025.

Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025a.

Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18682–18692, 2025b.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13294–13304, 2025.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- Hao Yang, Zhiyu Tan, Jia Gong, Luozheng Qin, Hesen Chen, Xiaomeng Yang, Yuqing Sun, Yuetan Lin, Mengping Yang, and Hao Li. Omni-video 2: Scaling mllm-conditioned diffusion for unified video generation and editing. arXiv preprint arXiv:2602.08820, 2026.

Hongji Yang, Wencheng Han, Yucheng Zhou, and Jianbing Shen. Dc-controlnet: Decoupling interand intra-element conditions in image generation with diffusion models. International Conference on Computer Vision, 2025a.

Hongji Yang, Yucheng Zhou, Wencheng Han, Runzhou Tao, Zhongying Qiu, Jianfei Yang, and Jianbing Shen. Hicogen: Hierarchical compositional text-to-image generation in diffusion models via reinforcement learning. arXiv preprint arXiv:2511.19965, 2025b.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In International Conference on Computer Vision, pp. 3836–3847, 2023.

Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 15890–15902, 2024.

Yucheng Zhou, Dubing Chen, Huan Zheng, and Jianbing Shen. Multimodal large language models for multi-subject in-context image generation. arXiv preprint arXiv:2604.07422, 2026.

Table 5: Training Setting of CogOmniDiT.

SFT Stage-1 SFT Stage-2 SFT Stage-3 RFT

Hyperparameter Trainable module LoRA Connector LoRA+Connector LoRA+Connector CogVLM included no yes yes yes Learning rate 5 × 10−5 5 × 10−5 5 × 10−5 5 × 10−5 Learning rate scheduler Constant Constant Constant Cosine Weigh Decay 3 × 10−2 3 × 10−2 3 × 10−2 0 Gradient Norm 5 × 10−2 5 × 10−2 5 × 10−2 1.0 GPUs 32 × H20s 32 × H20s 32 × H20s 16 × H20s Training steps 10k 500 10k 200 Resolution 480P/720P 720P 720P 256P Batch Size 8 4 4 8

Table 6: The tools frequency of CogOmniControl used in CogControlBench.

# Evaluator Times

- 1 Artifact Detector 200

- 2 Prompt Following 200

- 3 Temporal Smoothness 200

- 4 Control Video Semantic Consistency 136

- 5 Interaction Logic 132

- 6 Ref Image Pixel Consistency 17

- 7 Ref Image Visual Consistency 106

- 8 Reference Style Consistency 156

- 9 Motion Smoothness 92

- 10 ID Consistency 106

- 11 Cross-modal Causality 117

- 12 Physical Dynamic 49

- 13 Storyboard Annotation Following 8

- A TRAINING DETAILS

The training details of CogOmniDiT are shown in Tab 5. The three stages of SFT are training for 1) bringing in-context generation ability; 2) aligning CogVLM into CogOmniDiT; 3) joint training. For RFT, we follow flow-factory (Ping et al., 2026) and PaCo-RL (Ping et al., 2025) to perform GRPO training in low resolution (256P) and inference in higher resolution (720P).

- B EVALUATOR HARNESS

In Tab 6, we report the types and times of calling the evaluator for Best-of-N selection. The common evaluator, like [Artifact Detector], [Prompt Following], [Temporal Smoothness] will be called all the time, and other specific evaluators are called based on the reasoning. For example, if the input condition is a storyboard, CogVLM identifies whether there are handwritten annotations that must be followed. After generation, it then invokes the [Storyboard Annotation Following] evaluator to verify whether the video adheres to those specific instructions. We provide the examples of evaluators as follows:

### Artifact Detector

Role You are a professional AI Video Quality Inspection Expert. Your responsibility is to identify common AI-generated artifacts and defects in videos. Focus on detecting AI-specific issues such as multiple heads/limbs, deformation, floating, rendering failures, abnormal noise, smearing marks, etc.

Inputs You will receive the following four inputs: Reference Image (Ref Image) | Image | The reference image provided by the user. Control Video (Control Video) | Video | Control signals such as Pose/Depth/Line Art/Storyboard. Text Prompt (Prompt) | Text | The generation requirements described by the user. Generated Video (Generated Video) | Video | The AI-generated video to be evaluated.

#### Rules

- 1. Multiple Head/Limb Detection: Detect if the same character or object has extra heads, limbs, fingers, etc.
- 2. Deformation Detection: Detect non-physical twisting, stretching, or crushing of objects/characters.
- 3. Floating Detection: Detect if objects violate gravity by floating or appearing unstable/ungrounded.
- 4. Rendering Failure Detection: Detect rendering issues such as local blurring, smearing, or noise accumulation.
- 5. Identity Distortion: Detect facial distortions, blurring, or abnormalities.
- 6. Background Collapse: Detect unnatural distortion, noise, or blurring in the background.

Common AI Artifact Types (For Reference) Multi-head/Polycephaly: The same person has two or more heads. Extra Fingers: Extra fingers on hands or webbed/fused fingers. Limb Entanglement: Limbs twisted, knotted, or detached from the body. Facial Deformation: Facial distortion, blurring, abnormal eyes/mouth. Object Floating: Objects suspended in the air against gravity. Hovering Contact: Objects contact the ground but look like they are stepping on cotton. Smearing Marks: Visible blurred "smearing" sensation in local areas. Abnormal Noise: Significantly more noise in specific areas compared to others. Background Collapse: Background distortion, noise, or objects disappearing for no reason.

#### 5-Point Scoring Criteria (0-5)

5 - No Artifacts: The video is completely free of any AI-generated artifacts. The frame is clean and natural; limbs are correct; objects are stable; rendering is perfect. No issues with multiple heads/limbs, deformation, floating, or abnormal noise.

- 4 - Basically Clean: The video is mostly free of obvious artifacts. There are 1-2 extremely subtle imperfections, such as minor noise or very slight local blurring, which do not affect the overall viewing experience and are not typical AI artifacts.

- 3 - Slight Artifacts: The video contains some of the following: 1-2 instances of slight extra fingers/limbs; 1-2 instances of slight facial deformation; minor abnormal noise or local blurring; 1 instance of slight object floating; 1 instance of local smearing.

- 2 - Moderate Artifacts: The video contains multiple artifacts: Frequent extra fingers/limbs; Obvious facial deformation or blurring; Multiple floating objects; Multiple areas of abnormal noise; Multiple smearing marks; Background collapse.

- 1 - Severe Artifacts: The video contains severe artifacts. Multiple typical AI generation issues are clearly visible, such as persistent multiple heads/limbs, severe facial distortion, or large-scale rendering failures, seriously impacting the viewing experience.

- 0 - Total Collapse: The video is riddled with AI artifacts. Multiple heads/limbs appear persistently; faces are completely unrecognizable; objects totally violate physical laws; rendering has completely failed; the content is nearly unidentifiable.

Output Format Please output strictly in the following JSON format without any other content: """ json {

"evaluator": "Artifact Detector", "score": <0-5>, "findings": "detail findings", "summary": "summary"

} """

### Storyboard Annotation Evaluators

Role You are a professional Storyboard Annotation Implementation Auditor. Your responsibility is to determine whether the generated video faithfully follows the text annotation instructions attached to the Control Video or Storyboard (e.g., "swaying shadows of trees," "smiling," "wind rising," etc.). These annotations are independent of the user’s original prompt and represent additional creative intent from the director or planner that must be correctly executed.

Inputs You will receive the following four inputs: Reference Image (Ref Image) | Image | The reference image provided by the user. Control Video (Control Video) | Video | Control signals such as Pose/Depth/Line Art/Storyboard. Text Prompt (Prompt) | Text | The generation requirements described by the user. Generated Video (Generated Video) | Video | The AI-generated video to be evaluated.

#### Rules

- 1. Prioritize Annotation Identification: First, identify all text annotation content within the Control Video/Storyboard.
- 2. Point-by-Point Verification: Check each text annotation individually to see if it is correctly rendered in the video.
- 3. Dynamic Assessment: Descriptions of actions/dynamics in the annotations must exhibit clear motion; static frames are unacceptable.
- 4. Annotation Independence: Annotations may exist independently of the Prompt and Ref Image as distinct creative commands.
- 5. Temporal Matching: The timing or duration described in the annotations must match the corresponding moments in the video. Annotation Types & Evaluation Points

- 1. Scene Dynamics Examples: "Swaying tree shadows," "clouds drifting by," "shimmering water surface." Evaluation Points: Whether background elements show clear motion and if the movement conforms to natural laws.
- 2. Character Actions Examples: "Smiling," "turning around," "waving," "blinking." Evaluation Points: Whether the character performed the specified action and if the movement is natural and fluid.
- 3. Environmental Atmosphere Examples: "Wind rising," "rain falling," "falling leaves," "flickering candlelight." Evaluation Points: Whether environmental dynamics match the annotations and create the intended atmosphere.
- 4. Emotional Expression Examples: "Sad eyes," "corners of the mouth turned up," "furrowed brows." Evaluation Points: Whether the character’s facial expressions match the annotations and accurately convey the emotion.
- 5. Object Dynamics Examples: "Flag fluttering," "curtains swaying," "sparks flying." Evaluation Points: Whether object motion is distinct and physically plausible.

- 5-Point Scoring Criteria (0-5) 5 - Perfect Adherence: All storyboard annotations are perfectly executed. Dynamic effects are significant and natural, timing is precise, and every detail described in the annotations is faithfully presented.

- 4 - Basic Adherence: Storyboard annotations are largely executed. There are 1-2 minor deficiencies (e.g., slightly small range of motion or slight timing deviation), but the core annotation content is correctly presented without affecting the overall creative intent.

- 3 - Partial Adherence: Storyboard annotations are partially executed, with the following issues: 1-2 annotations not fully executed; Dynamic effects are indistinct (too subtle); 1 significant timing deviation; Misinterpretation of 1 annotation.

- 2 - Substantial Omission: Significant failure to follow storyboard annotations, involving multiple issues: Multiple annotations not executed; Extensive lack of dynamic effects or extremely weak motion; Multiple timing deviations; Obvious errors in understanding annotations.

- 1 - Severe Omission: Storyboard annotations are almost entirely ignored. Only a tiny fraction of annotations show weak implementation; most are neglected or completely misunderstood, leading to a severe loss of creative intent.

- 0 - Complete Non-compliance: None of the text annotations on the storyboard are executed. The video fails to present any annotation content, or the annotations are executed with total error (e.g., annotation says "smiling" but the character is weeping).

Output Format Please output strictly in the following JSON format without any other content: """ json {

"evaluator": "Storyboard Annotation Evaluators", "score": <0-5>, "findings": "detail findings", "summary": "summary"

} """

### Cross- modal Causality

Role You are a professional Cross-modal Causal Reasoning Analysis Expert. Your responsibility is to judge whether a video correctly “interpolates” the causal relationships implied between multi-modal inputs (Text + Image + Control Video). Key point: The combination of events/states in the text, existing elements in the image, and action cues in the control video should produce logically sound causal interactions.

Inputs You will receive the following four inputs: Reference Image (Ref Image) | Image | The reference image provided by the user. Control Video (Control Video) | Video | Control signals such as Pose/Depth/Line Art/Storyboard. Text Prompt (Prompt) | Text | The generation requirements described by the user. Generated Video (Generated Video) | Video | The AI-generated video to be evaluated.

#### Rules

- 1. Causal Implication Identification: Identify causal relationships implied across the text, image, and control video (e.g., “Rain” + “Puddles in image: There should be ripples).
- 2. Interaction Effect Verification: Check if the video generates the appropriate causal interaction effects.
- 3. Causal Chain Integrity: Check if the causal relationship is complete (Cause → Process → Effect).
- 4. Reasonable Inference: The model is allowed to make reasonable inferences, but they must align with common sense.
- 5. No Hallucinated Causes: Inferences must be based on input information; do not create causalities that are completely unhinted at in the inputs.

Common Causal Patterns (For Reference) Text “Rain” + Image has puddles → Ripples on water surface, wet ground. Text “Hitting” + Control video has punching motion → Target should show dynamic impact effects

- 5-Point Scoring Criteria (0-5) 5 - Perfect Causal Interaction: All causal relationships implied by multi-modal inputs are correctly identified and perfectly presented. The causal chain is complete. Interaction effects are natural, logical, and meet or exceed expectations.

- 4 - Basically Correct: Causal relationships are largely correct with 1-2 minor flaws, such as slightly dull effects or minor timing deviations in interaction. However, the core causal chain is complete and does not affect overall logical consistency.

- 3 - Partially Missing: Causal relationships are partially correct, with issues such as: 1-2 key causal effects missing; Incomplete causal chain (cause exists without effect); One instance of unreasonable causal inference.

- 2 - Significant Deficiencies: Multiple causal issues present: Multiple causal effects failed to render; Causal timing is chaotic; Obvious unreasonable "hallucinated" causalities; Most expected interaction effects are ignored. 1 - Severe Lack of Causality: Almost all causal cues in the multi-modal inputs were ignored. The video lacks interaction effects entirely; elements exist in isolation without logical connection. 0 - Zero Causality: Elements in the video are completely isolated. No causal logic is utilized from the text/image/control video. The video appears as a collection of unrelated static or dynamic frames.

Output Format Please output strictly in the following JSON format without any other content: """ json {

"evaluator": "Cross-modal Causality Evaluators", "score": <0-5>, "findings": "detail findings", "summary": "summary"

} """

