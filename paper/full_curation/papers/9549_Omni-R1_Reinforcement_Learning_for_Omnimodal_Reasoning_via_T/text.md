arXiv:2505.20256v1[cs.CV]26May2025

# Omni-R1: Reinforcement Learning for Omnimodal Reasoning via Two-System Collaboration

Hao Zhong∗, Muzhi Zhu∗, Zongze Du∗, Zheng Huang, Canyu Zhao, Mingyu Liu, Wen Wang, Hao Chen, Chunhua Shen†

Zhejiang University, China

### Abstract

Long-horizon video–audio reasoning and fine-grained pixel understanding impose conflicting requirements on omnimodal models: dense temporal coverage demands many low-resolution frames, whereas precise grounding calls for highresolution inputs. We tackle this trade-off with a two-system architecture: a Global Reasoning System selects informative keyframes and rewrites the task at low spatial cost, while a Detail Understanding System performs pixel-level grounding on the selected high-resolution snippets. Because “optimal” keyframe selection and reformulation are ambiguous and hard to supervise, we formulate them as a reinforcement-learning (RL) problem and present Omni-R1—an end-to-end RL framework built on Group Relative Policy Optimization. Omni-R1 trains the Global Reasoning System through hierarchical rewards obtained via online collaboration with the Detail Understanding System, requiring only one epoch of RL on small task splits. Experiments on two challenging benchmarks, Referring Audio-Visual Segmentation (RefAVS) and Reasoning Video Object Segmentation (REVOS), show that Omni-R1 not only surpasses strong supervised baselines but also outperforms specialized state-of-the-art models, while substantially improving out-of-domain generalization and mitigating multimodal hallucination.

Our results demonstrate the first successful application of RL to large-scale omnimodal reasoning and highlight a scalable path toward universally foundation models. Our code is released at: https://github.com/aim-uofa/Omni-R1.

### 1 Introduction

Enabling models to simultaneously perceive, understand, and reason over omnimodal inputs—such as text, video, and audio—in complex real-world scenarios remains a longstanding goal in artificial intelligence [1, 2, 3, 4]. Recent advances in omnimodal pretraining and instruction fine-tuning have led to the emergence of omnimodal models, bringing us closer to this objective [5, 6, 7]. Despite recent progress, current omnimodal models exhibit notable limitations in two key areas: (1) longhorizon reasoning over complex temporal sequences in video and audio, and (2) fine-grained spatial understanding at the pixel level. A fundamental challenge underlying these two problems is the inherent trade-off between temporal coverage and spatial resolution . Long-horizon reasoning [8, 9] over video and audio typically requires high frame rates to capture global temporal context, which significantly increases memory and computational overhead—often forcing models to operate on low-resolution frames. Conversely, fine-grained pixel understanding [10, 11] demands high-resolution inputs to preserve visual details, which in turn limits the number of frames that can be processed. This

∗Equal contribution. †C. Shen is the corresponding author.

Preprint. Under review.

[Figure 1]

#### Results

#### Inferring

[Figure 2]

[Figure 3]

###### Detail Understanding System

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Omni There are four.

[Figure 10]

How many knives can the robot see in the right when he can hear the water?

User

Omni-R1

[Figure 11]

###### Global Reasoning System

[Figure 12]

<Think>I need to find out the key frames when he doing the dishes …</Think>

[Figure 13]

[Figure 14]

<Answer>{“start_time": “01:50",

"end_time": “01:51”, "description": “robot doing the dishes”}</Answer>

high resolution key frames

Omni-R1

Training

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

###### Detail Understanding System GRPO Rollouts

<Think>The instruction asks to identify the brownish-yellow cat. The video shows a brownish-yellow cat and another cat on the right. Since the task is to segment the brownish-yellow cats, the relevant duration should be when both brownishyellow cats are visible, even though they appear at different times with one being on the right side of the plate at some point. </Think>

Which cow stepped back with the movement of its companion and then turned around in place?

[Figure 22]

<think> The instruction asks to identify which cow stepped back with the movement of its companion and then turned around in place. The key action here is the stepping back and turning around. The video shows two cows, and one of them steps back and then turns around. The other cow remains stationary during this action. </think>

Directly identify the brownish-yellow cat in this video.

<Answer>{“start_time":"09:00","end_t ime":“09:00”,"description":"brownishyellow cat”}</Answer>

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Omni Data

Key Frames Selection Omni-R1

Omni-R1 prediction

Detail Understanding System Reference Model

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

GRPO Rewards Feedback

Input Frames Output Grounding

ℛ𝒦 ℛ𝒜 ℛ𝒢

ℛ𝒦 ℛ𝒜 ℛ𝒢

Sa2VA prediction

| | | |
|---|---|---|
| | | |

Omni-

- Figure 1: Overview of the proposed Omni-R1 system for collaborative video understanding. Left: Performance comparison across multiple benchmarks shows Omni-R1 significantly outperforms existing omni-modal and video-reinforced MLLMs on both segmentation-centric and reasoningcentric tasks. Top-right: Omni-R1 employs a two-stage collaborative framework, integrating a detail understanding system (for precise visual QA) and a global reasoning system (for temporal grounding and high-resolution key frame identification). Bottom: A qualitative example highlights Omni-R1’s precise spatial-temporal segmentation and reasoning in identifying object-centric actions, outperforming prior expert models (e.g., Sa2VA) in complex scenarios.

Grounding …

Temporal Consistency

###### Frame-Instruction Alignment

Key Frame Quality

[Figure 35]

[Figure 36]

Match！

[Figure 37]

sam2

Temporal Segmentation

[Figure 38]

| | | |
|---|---|---|
| | | |

[Figure 39]

Whole Cat Only Hand

ℛ𝒦 ℛ𝒢

ℛ𝒜 OriginalInstruction GroundTruth

trade-off creates a tension between global context modeling and local detail preservation, making it difficult for existing models to excel at both simultaneously.

Wrong

Consistent Inconsistent

A natural way to address this trade-off is to decompose the problem into two stages. Accordingly, we frame our solution as a two-system architecture:

Local Reward Global Reward

- • System 1 (Global Reasoning System) performs coarse-grained, global reasoning over long video sequences at low spatial resolution—acting as a fast, context-aware selector that identifies critical temporal segments.
- • System 2 (Detail Understanding System), in contrast, conducts detailed, high-resolution analysis over a small number of keyframes, focusing on precise grounding and fine-grained understanding.

To illustrate how these systems interact, consider a task where the goal is to segment the last person to disappear (or make a sound) in a scene. System1 first processes the full video (with audio) sequence

to determine, through low-resolution multimodal abstraction, which person is the last to leave visually or to emit sound. It then selects a few key segments where this individual appears or speaks. Since System 2 operates only on short segments with high-resolution input and lacks access to long-range temporal or auditory context, System 1 needs to reformulate the original reference task—initially requiring long-horizon multimodal reasoning—into a simpler, localized problem. This reformulated task focuses on attributes, identity cues, and object permanence within the selected key segments, making it solvable using only fine-grained visual information. System 2 then takes these key segments and performs fine-grained visual grounding directly on the high-resolution input, bypassing the need for global reasoning. This two-system design enables scalable and efficient multimodal reasoning by eliminating the need to process entire videos at high resolution, and effectively addresses the dual challenge of long-horizon reasoning and fine-grained visual understanding.

It is worth noting that current multimodal models already perform well as Detail Understanding Systems in tasks such as visual grounding [11, 12], OCR [2, 13, 14] and fine-grained image understanding [15, 16] on high-resolution inputs. Given this progress, the bottleneck in our two-system framework lies primarily in the capabilities of System 1. In this work, we therefore focus on improving Global Reasoning System, particularly its ability to select informative keyframes and reformulate the task. However, defining what constitutes an “optimal” keyframe selection or task reformulation is inherently ambiguous and task-dependent, making it impractical to rely on manually curated SFT data. To address this, we propose Omni-R1, an end-to-end reinforcement learning framework tailored for omnimodal reasoning. Built upon the Group Relative Policy Optimization (GRPO) [17, 18] algorithm, our method simulates online collaboration between System 1 and System 2, applying policy gradient updates guided by a hierarchical reward framework to progressively train System 1 to select keyframes and reformulate tasks in long-horizon, omnimodal settings.

From a reinforcement learning (RL) perspective, although it has proven effective in enhancing reasoning within large language models [17, 18, 19], RL remains underexplored in omnimodal settings. One major challenge lies in the lack of effective multimodal reasoning data [20], along with uncertainty about whether language-based RL techniques can generalize across modalities. While Omni-R1 bridges this gap by reformulating long-horizon multimodal understanding as a collaborative process between two systems. In our design, the Global Reasoning System functions as an RL agent that selects keyframes and reformulates tasks for the Detail Understanding System to complete. Such an approach provides a scalable path toward improving temporal reasoning and summarization in omnimodal models, while also opening new opportunities for applying RL beyond purely linguistic tasks.

To validate the effectiveness of Omni-R1, we benchmark it on two especially demanding tasks, namely Referring Audio-Visual Segmentation (RefAVS [21]) and Reasoning Video Object Segmentation (REVOS [22]), both of which require temporal reasoning over video(audio) streams and fine-grained pixel understanding. Training Omni-R1 for just one epoch on the small datasets already lifts performance well beyond our baseline model and even surpasses the strongest, highly specialized state-of-the-art models on each benchmark. Even more striking, reinforcement learning improves out-of-domain generalization, whereas conventional supervised fine-tuning often weakens it. OmniR1 achieves higher scores in both pure video-understanding and omnimodal understanding settings, outperforming recent RL methods tailored specifically to video-reasoning tasks. Finally, we conduct a comprehensive suite of diagnostic studies—including ablations over key architectural and training choices and an analysis of RL’s impact on multimodal hallucination—which together highlight the versatility and reliability of our approach. We hope that Omni-R1 offers a new direction for applying reinforcement learning to future all-modality foundation models.

Our primary contributions are summarized as follows:

- • We present a scalable Global Reasoning, and Detail Understanding two-system architecture that separates long-horizon video–audio reasoning from fine-grained pixel-level grounding, effectively resolving the temporal–spatial trade-off that constrains existing omnimodal models.
- • We introduce an end-to-end reinforcement-learning framework Omni-R1, built on Group Relative Policy Optimization that trains System 1—via hierarchical rewards and simulated collaboration with System 2—to select keyframes and reformulate tasks in long-horizon omnimodal settings.

- • With one epoch RL training, Omni-R1 surpasses strong supervised baselines and specialized SOTA methods on RefAVS and REVOS, while markedly improving out-of-domain generalization including video understanding and omnimodal understanding.

### 2 Related Work

- 2.1 Omni-modal Large Models

The advent of Large Language Models (LLMs) has revolutionized artificial intelligence, showcasing unprecedented capabilities in understanding, generating, and reasoning with textual data [23, 24, 25, 26]. Building upon this foundation, Multimodal Large Language Models (MLLMs) have emerged, integrating multiple data modalities—such as vision, language, and audio—to achieve a more holistic understanding of complex tasks [11, 27, 28, 29, 30].

To differentiate from vision-language models (VLMs), multimodal large language models (MLLMs) incorporating the audio modality, such as Qwen2.5-Omni [31], are termed omni-modal models, abbreviated as omni. MiniCPM-o 2.6 [2] extends its vision-language foundation [2] with audio processing capabilities, allowing it to operate across more modalities. Baichuan-Omni-1.5 [1], trained and inferred in a fully end-to-end manner, surpasses GPT-4o-mini on the full-modality leaderboard OmniBench [32]. The recent development of omni-modal models further extends this integration, encompassing visual, linguistic, and auditory modalities to approach a comprehensive multimodal understanding [7].

- 2.2 MLLM with RL

Despite the remarkable progress enabled by supervised learning and instruction tuning, key challenges persist in aligning MLLMs with human preferences, mitigating harmful outputs, and enhancing their performance on complex reasoning tasks. Reinforcement Learning (RL), particularly Reinforcement Learning from Human Feedback (RLHF) [19], has proven effective in addressing these issues within unimodal LLMs, contributing to the success of models like ChatGPT [23]. A notable advancement in this domain is the introduction of DeepSeek-R1 [17], which employs Group Relative Policy Optimization (GRPO) to enhance reasoning capabilities. GRPO innovatively replaces traditional critic models with a group-based reward normalization approach, reducing computational costs while maintaining performance [18]. This technique has demonstrated that pure RL can effectively develop strong reasoning abilities without reliance on supervised data.

While reinforcement learning techniques have been widely explored in LLMs, their application to MLLMs is still at an early stage. Most recent efforts [33, 34, 35, 36] have primarily focused on vision and language modalities, with little attention paid to more comprehensive multimodal integration. Notably, a concurrent work, R1-Omni [37], is the first to include audio in addition to vision and language; however, its focus is limited to a single motion recognition task.

In contrast, our work targets more general long-horizon understanding tasks and conducts a more comprehensive and systematic investigation of omni-modal reinforcement learning. Building on recent advances, we propose Omni-R1, an omni-modal framework that unifies vision, language, and audio processing under an end-to-end RL optimization pipeline. Our two-system design, which separates temporal reasoning from spatial perception, enables enhanced long-horizon understanding and fine-grained attention, allowing Omni-R1 to better address complex multimodal tasks requiring both structured perception and dynamic decision-making.

- 3 Omni-R1

- 3.1 Task and System Formulation

We consider a long-horizon multimodal understanding task, where the model receives a video sequence V = {v1,v2,...,vT} and a synchronized audio stream A = {a1,a2,...,aT}, along with an instruction or query q. The goal is to produce a task-specific output y (e.g., a localized segment, a textual response, or a grounding prediction) that reflects both global temporal reasoning and finegrained visual understanding. To better facilitate global temporal reasoning, we transform the raw instruction q into a high-level instruction qglobal = T(q) via a template-based rewriting function T.

Training

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

###### Global Reasoning System GRPO Rollouts

[Figure 44]

<Think>The instruction asks to identify the brownish-yellow cat. The video shows a brownish-yellow cat and another cat on the right. Since the task is to segment the brownish-yellow cats, the relevant duration should be when both brownishyellow cats are visible, even though they appear at different times with one being on the right side of the plate at some point. </Think>

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Directly identify the brownish-yellow cat in this video.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

<Answer>{“start_time":"09:00","end_t ime":“09:00”,"description":"brownishyellow cat”}</Answer>

[Figure 59]

[Figure 60]

Omni Data

Key Frames Selection Omni-R1

[Figure 61]

[Figure 62]

Detail Understanding System & Reference Model

[Figure 63]

GRPO Rewards Feedback

Input Frames Output Grounding

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

[Figure 75]

[Figure 76]

ℛ𝒦𝒦 ℛ𝒜𝒜 ℛ𝒢𝒢

ℛ𝒦𝒦 ℛ𝒜𝒜 ℛ𝒢𝒢

| | | |
|---|---|---|
| | | |

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Omni-R1

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

| |
|---|

Grounding …

[Figure 89]

[Figure 90]

Temporal Consistency

Frame-Instruction Alignment

Key Frame Quality

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Match！

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

SAM2

[Figure 104]

Whole Cat

[Figure 105]

[Figure 106]

[Figure 107]

| | | |
|---|---|---|
| | | |

Temporal Segmentation

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

| |
|---|

[Figure 112]

Wrong

[Figure 113]

[Figure 114]

ℛ𝒦𝒦 ℛ𝒢𝒢

Consistent Inconsistent

## ℛ𝒜𝒜

Original Instruction

Ground Truth

[Figure 115]

Only Hand

Local Reward Global Reward

[Figure 116]

- Figure 2: Exclusively trained as System 1 on video segmentation tasks in an End-to-End RL pipeline, Omni-R1 improved general understanding capabilities.

- Stage 1: Global Reasoning System. We reduce the spatial resolution of the video as commonly adopted to obtain a low-resolution stream V˜ = {v˜1,...,v˜T} suitable for efficient global processing. Given (V˜,A,qglobal), System 1 produces a set of K selected segments(frames) S = {s1,s2,...,sK}

and a corresponding set of local queries {qlocal(i) }Ki=1 , intended to simplify the reasoning objective for System 2:

S,{qlocal(i) }Ki=1 = π(S1)(V˜,A,qglobal)

- Stage 2: Detail Understanding System. System 2 then receives the high-resolution frames

K} corresponding to the segments selected by System 1. Given (VS,{qlocal(i) }Ki=1), it performs fine-grained multimodal reasoning and produces the final output:

VS = {vs

,...,vs

1

y = π(S2)(VS,{qlocal(i) }Ki=1)

For tasks such as RefAVS and RVOS, one possible instantiation of System 2 is as a combination of a per-frame grounding model Fgrounding and a frozen video segmentation model Fseg (e.g., SAM2 [38]). Given the selected high-resolution frames VS = {vs

K} and the corresponding local instructions {qlocal(i) }Ki=1, the grounding model is applied independently to each pair (vs

,...,vs

1

,qlocal(i) ) to predict a set of bounding boxes Bs

i

= {b(1i),...,b(Ni)

}, where each b(ji) ∈ R4 denotes a box in

i

i

(x1,y1,x2,y2) format. These predicted boxes are then passed to the segmentation model to produce pixel-level instance masks and propagate them temporally across the entire video:

Mˆ = Fseg(V,VS,{Bs

i}Ki=1)

where the final output Mˆ = {mˆ 1,...,mˆ T} is a sequence of temporally aligned masks. The corresponding ground-truth mask sequence is denoted as M∗ = {m∗1,...,m∗T}, where each m∗t is the binary segmentation mask for frame vt. For more details on the segmentation model Fseg, please refer to Section B in Appendix.

##### 3.2 End-to-End Reinforcement Learning via GRPO

We now turn our focus to optimizing System 1 π(S1), with the goal of improving both the selection of key segments S and the formulation of task-specific local instructions {qlocal(i) }, in order to better support System 2 in performing fine-grained understanding. However, S, {qlocal(i) }, and the System 2 (i.e., π(S2)) are strongly coupled, making it difficult to directly define what constitutes an optimal pair (S,{qlocal(i) }) for downstream performance. As a result, constructing high-quality supervised fine-tuning (SFT) data for π(S1) is infeasible.

Instead, we propose to optimize π(S1) via reinforcement learning by designing a reward function R(S,{qlocal(i) },π(S2)) that evaluates the utility of System 1’s outputs in enabling System 2 to succeed. Under this framework, π(S1) is trained to explore and generate candidate outputs, and receives feedback from the environment through this reward. Specifically, we adopt a GRPO-based policy optimization scheme. At each iteration, we sample N responses from the current policy π(S1) and compute the corresponding rewards rn using the reward function R(·). We then normalize the rewards to estimate the advantage of each sample:

rn − mean({r1,...,rN}) std({r1,...,rN})

(1)

An =

Based on the computed advantages {An}, we perform PPO-style policy gradient updates to improve π(S1).

##### 3.3 Hierarchical Reward Design for System 1

Designing an effective reward function that accurately reflects the quality of the action pair (S,{qlocal(i) }) and provides a meaningful training signal for System 1 (π(S1)) is critical to the success of our framework. In this section, we describe our hierarchical reward formulation tailored for the Referring Video Object Segmentation (RVOS) task, which aims to guide System 1 to progressively learn to select informative keyframes and generate useful local instructions.

Due to the strong coupling among S, {qlocal(i) }, and System 2 (π(S2)), relying solely on the final task objective (e.g., segmentation mIoU) as the reward leads to unstable and inefficient training. This is

because such reward signals are sparse, non-decomposable, and difficult to attribute back to specific decisions made by π(S1). To address this, we propose a set of hierarchical reward functions, organized from weakly coupled to strongly coupled, and from local to global. These rewards are designed to incrementally shape the learning of System 1, starting from simpler supervision signals and gradually incorporating a more task-specific structure. We define three types of rewards:

Key Frame Quality Reward (RK): This reward evaluates the quality of the selected keyframes S, independently of the instructions or the performance of subsequent segmentation. It provides early learning signals to encourage the selection of visually salient or semantically diverse frames. We define the Key Frame Quality Reward as a weighted combination of three factors:

###### RK = λ1Rdiversity + λ2Rnum + λ3Rsaliency

The first term, Temporal Diversity Reward Rdiversity, encourages selected frames to spread over the video timeline, rather than being clustered within a short segment. This promotes broader temporal coverage and helps the model focus on long-range dynamics.

The second term, Frame Count Regularization Rnum, regularizes the number of selected frames K to stay near a predefined target K0, penalizes selections that include either too few or too many frames.

The third term, Object-Centric Saliency Reward Rsaliency, rewards keyframes that contain a large visible portion of the target object. This is based on the hypothesis that selecting such frames provides stronger visual anchors, which can facilitate more accurate and stable object tracking and segmentation throughout the video. It is calculated as the normalized average GT mask area:

1 K

Rsaliency =

K

###### area(m∗s

) maxt area(m∗ t)

i

i=1

Together, these components guide System 1 to select keyframes that are temporally diverse, reasonably sparse, and visually informative. Formal definitions of the reward are provided in Appendix Section B.

Frame-Instruction Alignment Reward (RA) measures how well each local instruction qlocal(i) aligns with its corresponding keyframe vs

. This reward evaluates whether the instruction provides sufficient grounding cues to locate the correct object in the frame. As it operates independently per frameinstruction pair, it does not depend on the segmentation model Fseg, and thus ignores temporal consistency. Concretely, given a frame vs

i

and its corresponding instruction qlocal(i) , we apply the grounding model Fgrounding to predict a set of bounding boxes Bs

i

###### = {b1(i),...,b(Ni)

}. We compare these predictions against the ground-truth target boxes B∗

i

i

si defined for that frame. Since a single instruction may refer to multiple target objects, both Bs

si can contain multiple instances.The reward is computed as the negative Hungarian matching loss commonly used in object detection [39]:

and B∗

i

K

1 K

RA =

i=1

1 − LHungarian(Bs

i

###### ,B∗s

) (2)

i

This loss is minimized when the predicted boxes exactly match the ground-truth targets.

Global Temporal Consistency Reward (RG) is the most strongly coupled and task-specific reward in our framework, directly reflecting the final objective of long-term video object segmentation. Unlike

previous rewards, which evaluate the selected keyframes or instructions in isolation, RG jointly considers how the selected keyframes S and local instructions {qlocal(i) } influence the performance of System 2 throughout the video. This reward is designed to capture both the spatial accuracy and the temporal consistency of the predicted instance masks. In particular, it encourages System 1 to select frames that are critical for robust tracking—such as those appearing after significant object deformations, occlusions, or disappearances—so that the segmentation model (e.g., SAM2) can re-anchor to the target effectively. Formally, given a candidate keyframe set S and corresponding instructions {qlocal(i) }, we feed them into System 2 (π(S2)) to obtain a full sequence of predicted masks Mˆ = {mˆ 1,...,mˆ T}.

The reward is computed as the average frame-wise Intersection over Union (IoU) with the ground-truth masks M∗ = {m∗1,...,m∗T}:

T

1 T

IoU( ˆmt,m∗t) (3)

RG =

t=1

Finally, we combine the above three components to form the overall reward used for training System

1. The total reward is a weighted sum of the three terms:

R = αKRK + αARA + αGRG (4) where αK,A,R are the weighting coefficients that control the importance of each reward component.

- Table 1: Performance comparison across models grouped by Seen and Unseen sets in RefAVSBench [21]. Some metrics curated from [21]. J&F represents the average of (J) score and (F) score. † indicates the results are tested on the masks predicted by SAM2 according to model’s grounding output.

Seen Unseen

Model

J&F J F J&F J F

AVSBench [43] + text 37.2 23.2 51.1 43.5 32.4 54.7 AVSegFormer [44] + text 40.2 33.5 47.0 43.1 36.1 50.1 GAVS [45] + text 39.4 28.9 49.8 39.8 29.8 49.7 ReferFormer [46] + audio 40.7 31.3 50.1 39.6 30.4 48.8 R2VOS [47] + audio 33.0 25.0 41.0 38.9 27.9 49.8 EEMC [21] 42.8 34.2 51.3 57.2 49.5 64.8 Qwen2.5-Omni-7B† 31.6 27.7 35.5 62.3 59.0 65.7

Qwen2.5-Omni-7B†(SFT) 39.1 35.4 42.8 66.2 63.1 69.3 Omni-R1-7B† 47.2 43.0 51.4 74.2 71.3 77.0 ∆ +16.4 +15.3 +9.4 +8.0 +8.2 +7.7

### 4 Experiments

##### 4.1 Experiments Setting

- System 1 and System 2 We adopt Qwen2.5-Omni-7B [7] as our base model, which serves as

- System 1 responsible for high-level reasoning. To construct a lightweight and stable System 2 during training, we use a frozen copy of the same pretrained Qwen2.5-Omni model, which also functions as a reference policy model for guiding optimization. For evaluation, unless otherwise stated, Omni-R1 is serving as both System 1 and System 2 for resource efficiency. However, due to the modular design and decoupled functionality of the two systems, System 2 can be flexibly replaced with a stronger perception module in a zero-shot manner.

Training Paradigm We train System 1 on 1,600 samples randomly selected from the RefAVS [21] dataset and 2,600 videos from the ReVOS [22] and MeViS [40] datasets for 1 epoch. To further enhance the model’s fine-grained understanding capabilities as system 2, we additionally train the model on 2,000 images from refCOCOg [41] for one epoch in the style of SegZero [42]. Unless otherwise specified, all experiments are conducted using a policy KL divergence hyperparameter of β = 0.04, a group size of 8, and an initial learning rate of 1 × 10−6 under the AdamW optimizer with a weight decay of 0.01. We adopt sam2-hiera-large as our SAM2 [38] version throughout the experiments.

##### 4.2 Referring Video Segmentation

Referring Audio-Visual Segmentation Ref-AVS [21] is specifically designed for audio-visual segmentation tasks, offering a diverse and well-annotated collection of samples that require integrated reasoning across both modalities. The dataset comprises 2,908 audio-equipped video clips in the training set, covering 5,366 annotated objects across 39 semantic categories.

We evaluated the performance of our collaborative system on Ref-AVSBench [21] with other Referring AVS methods. Omni-R1 outperforms previous SOTA EMMC [21] by +4.6% on J&F in seen set and +17.0% on unseen set.

Reasoning Video Object Segmentation ReVOS [22] is a VOS dataset that emphasizes reasoning about temporal behaviors through implicit object descriptions, comprising 35,074 pairs of instructionmask sequences derived from 1,042 diverse videos. In contrast to traditional referring video segmentation datasets, ReVOS includes text instructions that necessitate a sophisticated understanding of both video content and general world knowledge.

For our evaluation, we exclusively employed Sa2VA as System 2 to investigate the full reasoning capabilities of Omni-R1 as System 1.

- Table 2: Reasoning Video Object Segmentation performance comparison across different methods, the metric is J&F score(%). ‡ means the results are evaluated where Omni-R1-7B serves as System 1 and Sa2VA as System 2(1B and 4B).

Model

ReVOS Referring Reasoning Single Multi Overall

LISA-13B [10] - - - - 41.6 TrackGPT-13B [48] - - - - 45.0 VISA-13B [22] - - - - 50.9 Sa2VA-8B [49] - - - - 57.6 Sa2VA-26B [49] - - - - 58.4

Qwen2.5-Omni-7B† 46.3 26.9 38.6 37.4 36.6

- Omni-R1-7B† 53.2 41.9 48.3 46.5 47.6
- Omni-R1-8B‡ 61.6 50.7 56.6 47.3 56.2 Omni-R1-11B‡ 64.1 53.7 59.2 51.0 58.9

Our System 1 exhibits strong performance on video object segmentation tasks under both basic and reasoning-intensive conditions. When deployed as both systems (†), Omni-R1-7B significantly outperforms the base model on ReVOS, achieving a +11.0% improvement over Qwen2.5-Omni-7B. This result underscores its enhanced temporal reasoning and fine-grained recaption capabilities.

Furthermore, the collaborative system (‡) Omni-R1-11B achieves a score of 58.9% on ReVOS, surpassing much larger segmentation-specialized models such as Sa2VA-26B [49]. Notably, it achieves the best performance across all categories, including the reasoning subset in ReVOS (53.7%), underscoring the effectiveness of our disentangled system architecture and reinforcement learning-based training paradigm.

- 4.3 General Omni-Modal Understanding

In this section, we focus on the impressive progress of Omni-R1 on multi-modal tasks, in comparison to its base model Qwen2.5-Omni-7B and other leading multi-modal models.

Omni-R1 shows stable improvements over its base model Qwen2.5-Omni. Omni-R1 achieves an average improvement of +2.0%, +2.7% and +3.7% over baseline on OmniBench [32], VideoMME [53]

- Table 3: Performance comparison across models on general understanding QA benchmarks Omnibench, VideoMME, and MVBench.

Omnibench VideoMME MVBench Avg General Short General

Method

Vision-Language Models

Qwen2.5-VL-7B(CoT) - 56.1 71.3 57.4 LLaVA-OneVision-7B [50] - 58.2 - 56.7 Kangeroo-8B [51] - 56.0 - 61.1 VideoChat-R1 [36] - - 72.2 67.9 Video-R1 [35] - 59.3 - 63.9 Sa2VA-26B [49] - 52.6 - -

Omni-Modal Language Models

VITA-1.5-7B [52] 33.4 57.3 - 55.5 MiniCPM-o 2.6-7B [2] 40.5 63.4 - 58.6 Baichuan-Omni-1.5-7B [1] 42.9 60.1 - 63.7 Qwen2.5-Omni-7B 47.3 58.3 69.8 66.1 Omni-R1-7B 49.3 60.7 73.0 70.3 ∆ +2.0 +2.4 +3.2 +4.2

and MVBench [54] respectively, surpassing all other open-source omni-models. Specifically, in video understanding tasks, Omni-R1 gains more progress on short videos (less than 2 min) than long videos. This could be attributed to our VOS training videos, where almost all videos are less than 2 minutes, with MeViS even being less than 30 seconds.

System 1’s Strength in General Understanding Tasks Omni-R1 demonstrates significant improvements, achieving outstanding general performance on the omni-modal benchmark OmniBench, where it outperforms all other 7B models in the open-source space. With a score of 72.5 in the short subset of VideoMME, Omni-R1 surpasses VideoChat-R1 [36], which was exclusively fine-tuned for Video QA tasks through RL. Additionally, Omni-R1 achieves the highest score on MVBench, outperforming all other omni-modal models by a large margin.

The strong performance of Omni-R1 across both in-domain and general tasks showcases the effectiveness of our reinforcement learning approach. Leveraging System 1’s multi-modal reasoning, the model excels in task-specific scenarios and generalizes effectively to unseen tasks, demonstrating its robustness and adaptability in real-world environments.

### 5 Conclusion

We present Omni-R1, a novel reinforcement learning framework that addresses a key limitation in omnimodal models: the trade-off between long-horizon temporal reasoning and fine-grained spatial understanding. By decoupling these objectives into a two-system architecture comprising a Global Reasoning System and a Detail Understanding System Omni-R1 enables scalable and efficient processing of complex video–audio–text inputs.

Through task reformulation and keyframe selection trained via Group Relative Policy Optimization, our approach significantly improves performance on challenging benchmarks like RefAVS and ReVOS, while also enhancing out-of-domain generalization. Our diagnostic studies further confirm the robustness and versatility of the framework. We hope that this work opens new avenues for integrating reinforcement learning into next-generation omnimodal foundation models.

### A Appendix Overview

This appendix provides additional details on the experimental setup, model architecture along with training pipeline, and supplementary results that support the findings presented in the main paper.

- • Implementation Details: this section details additional aspects of our method: our twosystem architecture (including user instructions for its MLLM components and prompts for the downstream SAM2 model), our reward design, and the differences between training and inference procedures.
- • Ablation Studies: this section provides ablation studies of System 1 on the reward component and dataset selection.
- • Visualization Results: this section provides more visualization results, including examples in comparison with other methods and failure case analysis.
- • More Analysis: this section provides analysis on the hallucination issue and the influence of resolution on general video understanding tasks.
- • Limitations and Future Work: this section discusses the limitations of our method and potential future work.

### B Implementation Details

User Instructions on Two Systems. To enable MLLMs to perform keyframe selection and referred object captioning, we designed the prompt as shown in the figure 3. We formulate keyframes as time duration segments and assign spatial description text to each duration. Additionally, we observed that during training, the model could be influenced by the timestamp patterns seen in the prompt examples. Therefore to increase the diversity of keyframe distributions during training and prevent the model from overfitting to specific timestamps, we incorporated randomized timestamps into the prompt, encouraging the model to focus on learning keyframe selection and caption rather than simply copying training timestamps. For AVS tasks, we designed a similar prompt (see Figure 4) to guide the model in analyzing the audio content and identifying the corresponding visual grounding description. The prompt emphasizes the need to avoid temporal expressions and instead focus on visual cues.

The intermediate results are then interpreted as frames and paired descriptions before being fed into

- System 2. The prompt for System 2 follows the official grounding prompt of Qwen2.5-VL, where the output is a list of bounding boxes and their corresponding labels in JSON format.

Prompt Design for SAM2 as Downstream Segmenter. Once the keyframe grounding results are obtained from System 2, we assign a unique identifier to each detection result using a tuple format: (roll_out_idx, frame_idx, pred_obj_idx, bbox). This ensures that any detection within a single GRPO group can be uniquely referenced.

Since all detections within a group share the same input context, we optimize inference efficiency by processing all detection results in a single forward pass. Specifically, we feed the entire video segment into SAM2, and for each detection tuple, we assign a unique object ID. These object IDs are used as conditioning inputs to SAM2 to obtain their respective segmentation masks.

Specifically, we maintain a mapping dictionary between detection tuples and assigned object IDs P : tuple → obj_id, which enables us to reverse-map SAM2’s outputs {Mˆ obj_id|obj_id ∈ P(tuple)} back to the original detection structure Mˆ tuple. The segmentation results are then matched to the corresponding grounded predictions and used for reward evaluation.

Reward Design of RK To enable the model to learn from diverse keyframe selections, we design an evaluation reward function, RK, which assesses both the diversity and quality of the chosen keyframes. This function is formulated as:

RK = λ1Rdiversity + λ2Rnum + λ3Rsaliency

where the last component, Rsaliency, has been clarified in the main paper. Here, we detail the first two reward components.

Firstly, to discourage the model from selecting temporally adjacent keyframes, which can lead to redundant System 2 inference, we introduce a distribution reward function, Rdiversity. This component evaluates the distributional diversity of keyframes by calculating the temporal intervals between them. Specifically, all keyframes are sorted in chronological order. We then compute the temporal interval ti+1 − ti between each pair of consecutive keyframes. The final Rdiversity value is subsequently determined based on the collection of all such inter-frame temporal intervals.

The diversity reward Rdiversity(S) can be defined as the sum of an overlap punishment term and a distribution reward term:

Rdiversity(S) = overlap_punish · |I| + dist_reward · |D| where:

- • S is the set of selected items. Let Ssorted = (s1,s2,...,sM) be the sequence of M = |S| items sorted according to the relevant criteria (e.g., timestamps).
- • I is the set of indices i for which an "overlap" condition is met between si and si+1. Specifically, assuming idx(sj) gives an identifier for item sj:

I = {i ∈ {1,...,M − 1} | idx(si) = idx(si+1)}

|I| is the number of such identified overlaps (e.g., pairs of consecutive items with identical identifiers).

- • D is the set of indices of items in Ssorted that are not considered the start of an overlap as defined by I:

D = {j ∈ {1,...,M} | j ∈/ I} Therefore, |D| = M − |I|.

- • overlap_punish is the coefficient for the punishment. For this term to act as a punishment, overlap_punish should typically be a negative value (e.g., −0.2), or if it’s a positive value, it should be subtracted from the reward.
- • dist_reward is the coefficient for the reward given to items not initiating an overlap.

The formula Rdiversity(S) can also be written as:

Rdiversity(S) = (overlap_punish − dist_reward) · |I| + dist_reward · M Reward Design of RA The specific formulation of RA is as follows:

K

1 K

RA =

i=1

###### ,B∗s

1 − LHungarian(Bs

###### )

i

i

where Bs

si represents the corresponding set of ground truth bounding boxes. The function LHungarian refers to the Hungarian matching loss [39, 55], and K is the total number of selected keyframes.

denotes the set of predicted bounding boxes at the si-th frame, and B∗

i

The Hungarian matching loss LHungarian is computed based on the Intersection-over-Union (IoU) between predicted and ground truth bounding boxes. Specifically, a cost matrix M is first constructed using the IoU values between each pair of predicted and ground truth boxes. Then, the Hungarian algorithm is applied to the negative matrix −M to obtain the optimal one-to-one matching that minimizes the total negative cost, which corresponds to maximizing the overall IoU-based matching accuracy.

Reward Design of RG For RG, we adopt a simple aggregated IoU as the reward function. Specifically, for each detected object, we accumulate the predicted segmentation masks across all objects to

construct a per-frame mask set Mˆ t. Then, we compute the Intersection-over-Union (IoU) between the predicted masks and the corresponding ground truth masks M∗

t on each frame. The final reward is obtained by averaging the IoU values across all frames.

1 T

RG =

T

IoU( ˆmt,m∗t)

t=1

- • Given a [frames] seconds video and a reference instruction: [ref_prompt] that may involve temporal behavior, identify the exact object(s) [ref_prompt] in the video that matches the description.
- • Select about 4 most relevant moments that contain the referred object(s) with the best view.
- • Then, simplify the identified object into a short and clear visual grounding description that can be used for single-image reference at each moment.
- • Avoid temporal phrases and comparison phrases like “walking”, “moving”, “approaching”, “bigger” or “smaller”, but instead describe visible visual cues like clothing, pose, position, or grouping.
- • Try to select moments that are temporally well-distributed across the video, rather than clustered in the same part of the timeline. Avoid selecting multiple timestamps that are adjacent or overlapping; instead, prefer clearly distinct moments that each offer unique visual information. It is better to choose the most relevant and highly representative moments spanning the entire video, rather than picking all from the beginning.
- • Explain your reasoning in <think></think> and output the final result in <answer></answer>. Your final answer should be a JSON object in the following format:

<think> your analysis about the video and reference instruction </think> <answer> { "start_time": "00:[start]", "end_time": "00:[end]", "description": "direct description of referred object(s) at this moment" } </answer>

Figure 3: Keyframe selection and recaptioning prompt for System 1.

Training and Inference Strategy For video clips, we first feed them into System 1 at a relatively low resolution of a per-frame pixel 128×28×28, which allows us to process longer video segments during training and inference. Then System 2 predicts detection results at a higher resolution of 900×28×28. For VOS tasks, we adopt a random uniform sampling strategy during training, selecting between 8 and 24 frames per video to enhance temporal diversity and robustness. All SAM2-based segmentation and reward evaluations are then applied to these resampled clips at their original input resolution. For RefAVS tasks, we observed severe cross-modal hallucination issues during preliminary experiments, particularly when reasoning jointly over full-length audio and multi-frame video inputs. To mitigate this, we introduce a simplified variant, RefAID (Referring Audio-Image Detection), which reduces the AVS problem to object detection using only the first video frame and the corresponding full audio query. In this setting, no SAM2 segmentation is used; training is driven solely by detection-based rewards.

During inference, we resample a fixed maximum of 24 frames per video for VOS tasks. Unlike training, segmentation and evaluation are conducted over the full video sequence using SAM2 to align with standard benchmark protocols. For RefAVS, we adopt the same resampling and evaluation procedure as in VOS, ensuring consistency across task settings.

### C Ablation Studies

We conduct an ablation study to investigate the effect of progressively designed reward components RK (keyframe coverage), RA (alignment via Hungarian matching), and RG (global grounding IoU) on the overall performance. To ensure a fair comparison, all models are trained for one epoch on the ReVOS and MeVIS datasets and are constrained to select exactly four keyframes unless otherwise noted.

- Table 4 reports the results across four evaluation subsets of ReVOS: referring, reasoning, singleobject, and multi-object. We observe that the combination of RK + RG achieves the highest overall score (39.9%), outperforming the full combination RK + RA + RG (38.4%). This suggests that the inclusion of RA may introduce instability rather than improvement.

- • Given a [audio_duration] audio and a reference instruction: [ref_prompt], which involves temporal and audio-related behavior, first analyze the objects in the image that are producing sound, including both human voices and instrument sounds.
- • Based on the audio content, identify the exact object [ref_prompt] in the image that matches the audio.
- • Then, simplify the identified object into a short and clear visual grounding description that can be unambiguously recognized in a single image without relying on audio.
- • Avoid using temporal expressions such as “playing” or “moving”; instead, describe visible visual cues such as clothing, pose, position, or grouping.
- • Explain your reasoning in <think></think> and output the final result in <answer></answer>.

Figure 4: Audio analyzing and recaptioning prompt for System 1.

- Table 4: Ablation study on the reward function RK, RA and RG for System 1. The first model is trained with additional 2,000 samples from grounding dataset refcocog.

ReVOS Referring Reasoning Single Multi Overall

Method

Omni-R1 + refcocog 52.5 36.9 45.0 46.6 44.7 RK+RA+RG 44.2 32.5 38.2 41.9 38.4 RK+RG 45.5 34.2 39.6 42.6 39.9 RK+RA 43.1 29.5 36.8 37.5 36.3 RK 44.1 26.2 36.6 34.1 35.2

We hypothesize that this is due to the nature of RA, which relies on Hungarian matching over temporal sequences. Given that the segmentation model SAM2 already incorporates strong temporal priors, the additional alignment-based reward may not effectively capture useful gradients and could introduce variance from imperfect IoU estimation. In contrast, RG, which aggregates IoUs across frames, directly reinforces temporal consistency and spatial correctness, leading to more stable learning dynamics.

Interestingly, RK alone provides a surprisingly strong baseline (35.2%), demonstrating that ensuring keyframe coverage is already beneficial. However, only when combined with RG do we observe consistent improvements across all task types, including reasoning (34.2%) and multi-object scenarios (42.6%).

Lastly, we include a model additionally trained with 2,000 samples from the RefCOCOg grounding dataset [42] as a system 2 enhanced model. This model achieves 44.7% overall, demonstrating the potential of our design in VOS tasks lies in the grounding capabilities of the model.

These findings validate the effectiveness of RG as a grounding-aware reward and highlight the limitations of alignment-based matching in the presence of strong perceptual priors.

### D Visualization Results

Mask Quality Since our method utilizes the SAM2 model for segmentation, without fine-tuning mask decoder, the final mask output is more stable than those methods that rely on additional training on segmentation mask decoder. As can be seen in Figure 5a, in this simple example, our method is able to segment the target object with a mask consistent with the ground truth, while Sa2VA predicts the right target but generates a mask with holes and noise.

Temporal Reasoning Our System 1 leverages temporal reasoning to improve segmentation accuracy. As can be seen in Figure 5b, in this example, one has to watch the whole video and analyze the video context to make a correct prediction about the next bottle to be picked up. Our method is able to select the bottle that is about to be picked up, while Sa2VA fails to do so and segments

- Table 5: Video Object Segmentation performance on MeVIS across different methods, the metric is J&F score(%). ‡ means the results are evaluated where Omni-R1-7B serves as System 1 and Sa2VA-1B as System 2

MeVIS val_u

Model

Sa2VA-1B [49] 53.4 Sa2VA-4B [49] 55.4

Qwen2.5-Omni-7B† 33.6 Omni-R1-8B‡ 55.4

the bottle that is already picked up. A similar case is shown in Figure 6a, where one has to leverage world knowledge to understanding the target object and our method selects the right object while Sa2VA fails to do so. Both cases show that our method is able to leverage temporal reasoning to improve the segmentation accuracy.

Detail Understanding Our System 1 leverages detail reasoning to improve segmentation accuracy. Figure 5c shows a scenario where detail reasoning is needed to figure which wineglass is likely will be finished first. Our System 1 is already able to select the right wineglass but still makes a loose description for System 2 to analyze detail information, while Sa2VA fails to understand the instruction and segments all the wineglasses. This shows that our System 1 is able to delay detail reasoning for System 2 for detail understanding to improve the segmentation accuracy.

### E More Analysis

Hallucination Analysis During the training of our RefAVS task, we identified a significant hallucination problem, which we attribute to the complexity of multi-modal video and audio inputs. To systematically evaluate this issue, we conducted targeted assessments on audio-related hallucinations using the JUDGE subset of AVHBench [56], the first comprehensive benchmark designed to evaluate the perception and comprehension abilities of audio-visual large language models (LLMs).

As shown in Table 6, our base model (Qwen2.5-Omni-7B) achieves an accuracy of 58.5% on the JUDGE subset. Training on 1600 AVS samples leads to a modest improvement (60.8%), which is further enhanced to 61.5% by applying the GRPO KL loss with a reduced coefficient (β = 0.001). Notably, increasing the AVS training samples to 10400 does not yield better results, suggesting potential overfitting or task imbalance.

On the other hand, training with VOS tasks alone significantly boosts accuracy to 66.0%, and the best performance (71.9%) is obtained by jointly training on both AVS and VOS tasks. This represents a substantial improvement of 13.4% over the base model, demonstrating that multi-task training not only enhances audio-visual grounding but also mitigates hallucination issues more effectively.

These results confirm the effectiveness of leveraging task diversity and balanced reward optimization in improving the robustness of multimodal reasoning.

Video Resolution Influence on General Video Understanding Tasks To evaluate the influence of input resolution and prompting strategy on general video understanding, we compare model performance across different configurations on the VideoMME and MVBench benchmarks, as summarized in Table 7. All models are evaluated under two resolution settings: the default resolution of 128×28×28 and a higher resolution of 256×28×28 (denoted with *), with and without the proposed thinking prompting strategy.

We observe that increasing the input resolution consistently leads to performance gains across all models. For instance, Qwen2.5-Omni improves from 66.1% to 67.0% on MVBench when evaluated at higher resolution. Similarly, our Omni-R1-AVS model benefits from the resolution increase, achieving a performance gain from 68.3% to 68.7%. These improvements suggest that higher spatial resolution enhances the model’s ability to capture fine-grained visual details, particularly beneficial for multi-object reasoning and scene comprehension.

- Table 6: Performance on AVHBench (JUDGE subset, total 5302 samples). In the table, AVS tasks are trained on RefAVS dataset and VOS tasks are trained on ReVOS and MeViS datasets. The default GRPO KL loss weight β = 0.04.

Method

AVHBench JUDGE Correct Answers Accuracy

Base Model 3100 58.5% AVS 1600 samples 3222 60.8% AVS 1600 samples with β = 0.001 3261 61.5% AVS 10400 samples 3120 58.9% VOS 3500 66.0% AVS and VOS 3811 71.9%

- Table 7: Performance comparison across different resolutions and the use of a thinking prompt on VideoMME and MVBench. Resolutions are set to either 128×28×28 (default) or 256×28×28 (high). The thinking prompt provides an additional reasoning cue. The reported metric is the average of J and F scores (%).

VideoMME MVBench General Short Avg

Model Resolution Thinking

Qwen2.5-Omni 128×28×28 No 58.3 69.8 66.1 Qwen2.5-Omni 256×28×28 No 58.7 69.9 67.0 Qwen2.5-Omni 128×28×28 Yes 59.3 70.1 68.1 Qwen2.5-Omni 256×28×28 Yes 59.8 70.9 68.3

Omni-R1-AVS 128×28×28 No 59.0 71.9 68.3 Omni-R1-AVS 256×28×28 No 59.4 71.9 68.7 Omni-R1-AVS 128×28×28 Yes 59.9 72.1 69.4 Omni-R1-AVS 256×28×28 Yes 60.0 72.1 69.5

Omni-R1-VOS 128×28×28 No 59.7 72.3 68.9 Omni-R1-VOS 256×28×28 No 59.6 72.5 68.9 Omni-R1-VOS 128×28×28 Yes 59.8 72.5 69.8 Omni-R1-VOS 256×28×28 Yes 60.1 72.8 69.9

Omni-R1-VOS-AVS 128×28×28 No 60.1 72.5 69.1 Omni-R1-VOS-AVS 128×28×28 Yes 60.7 73.0 70.3

In addition to resolution, the thinking prompt designed to guide the model toward structured multi-step reasoning further boosts performance across all tested models. Omni-R1-AVS with thinking achieves 69.4% on MVBench, outperforming its baseline by 1.1%. The combination of both higher resolution and thinking yields the best results overall, with Omni-R1-VOS-AVS + thinking* reaching 60.7% on VideoMME (general) and 70.3% on MVBench. This indicates that resolution and prompting act as complementary strategies: resolution improves visual precision, while prompting enhances reasoning capability.

However, the performance gain obtained by increasing video resolution is marginal, suggesting that in general understanding benchmarks, resolution plays a limited role, and temporal understanding is more critical than fine-grained spatial details. This observation is consistent with our findings in the main paper, where the dual-system design significantly enhances the model’s temporal reasoning capabilities and yields notable improvements on reasoning-intensive VOS tasks.

### F Limitations and Future Work

Limitations Although our dual-system design significantly enhances the temporal reasoning capability of System 1, the complete functional decoupling between System 1 and System 2 introduces certain limitations. In particular, System 2 lacks temporal context, which may affect consistency in temporally coherent tasks. This consideration partially motivates our selection of VOS as a

primary training task: while VOS emphasizes temporal consistency, it also provides dense per-frame annotations that allow us to design stable training strategies to mitigate the context gap—such as frame-wise Hungarian matching loss and aggregated mask-based rewards. During inference, the missing temporal cues in System 2 are partially recovered through SAM2’s mask-based processing.

However, when extending to tasks requiring finer temporal sensitivity such as detecting and describing localized anomalous behaviors within a specific time span, our current architecture faces new challenges. While System 1 can still progressively narrow down relevant temporal segments, the need for dynamic temporal granularity exposes the limitations of a fully decoupled, one-way reasoning architecture.

Discussion on Future Work Our coarse-to-fine reasoning pathway aligns closely with human cognitive intuition, yet we recognize the crucial importance of enabling a bidirectional flow of information between global and local levels. Our current design can be viewed as a context-constrained tree search structure, where only the root node (System 1) has access to full contextual information, while downstream nodes (System 2) operate solely on partial, local input. This diverges from typical hierarchical systems, which often allow child nodes to access aggregated information from their parent nodes.

While the VOS task structure inherently compensates for this limitation by providing dense temporal supervision, to further enhance the flexibility of our two-system framework, future work should explore more interactive architectures that facilitate explicit information exchange between System 1 and System 2. Supporting backtracking within the reasoning tree would empower System 1 to refine its global reasoning using local insights from System 2. Conversely, it would enable

- System 2 to operate with broader contextual awareness provided by System 1. We firmly believe that transforming this pipeline into a bidirectional, cooperative reasoning structure holds immense potential for advancing multi-modal temporal understanding abilities on more flexible and complex tasks.

|[Figure 117]<br><br>[Figure 118]|
|---|

|[Figure 119]<br><br>[Figure 120]|
|---|

|[Figure 121]<br><br>[Figure 122]|
|---|

|[Figure 123]<br><br>[Figure 124]|
|---|

|[Figure 125]<br><br>[Figure 126]|
|---|

|[Figure 127]<br><br>[Figure 128]|
|---|

Which is the white object(s) made of cellulose that is/are held in people's hands?

[Figure 129]

[Figure 130]

<think> The instruction asks to segment the white object(s) made of cellulose that is/are held in people's hands. The video shows a person folding a white paper on a wooden table. The white object in question is the paper. The person’s hands are visible as they manipulate the paper. The action of folding the paper is the key behavior here. The most relevant duration for this action is from 0.5 to 2.0 seconds, as it captures the initial folding of the paper.</think> "answer": {"start_time": "00:00.00", "end_time": "00:00.50","description": "white paper being folded"}

[Figure 131]

|[Figure 132]<br><br>[Figure 133]|
|---|

|[Figure 134]<br><br>[Figure 135]|
|---|

|[Figure 136]<br><br>[Figure 137]|
|---|

|[Figure 138]<br><br>[Figure 139]|
|---|

|[Figure 140]<br><br>[Figure 141]|
|---|

|[Figure 142]<br><br>[Figure 143]|
|---|

Omni-R1 prediction

|[Figure 144]<br><br>[Figure 145]|
|---|

|[Figure 146]<br><br>[Figure 147]|
|---|

|[Figure 148]<br><br>[Figure 149]|
|---|

|[Figure 150]<br><br>[Figure 151]|
|---|

|[Figure 152]<br><br>[Figure 153]|
|---|

|[Figure 154]<br><br>[Figure 155]|
|---|

Sa2VA prediction

- (a) In this simple example, both our method and Sa2VA are able to select the right target object, but our method is able to segment the target object with a mask consistent with the ground truth, while Sa2VA generates a mask with holes and noise.

[Figure 156]

[Figure 157]

Which bottle(s) of beverage will be picked up soon?

[Figure 158]

<think>The instruction asks to identify which bottle(s) of beverage will be picked up soon. The video shows a group of men playing instruments and drinking beer. The beer bottles are placed on a small table in front of them. The instruction suggests that one of the bottles will be picked up soon. Based on the video, the bottle on the right side of the table is the one that will be picked up soon.</think> "answer": {"start_time": "00:01.50", "end_time": "00:02.00", "description": "the bottle on the right side of the table”}

Sa2VA prediction

Omni-R1 prediction

|[Figure 159]<br><br>[Figure 160]|
|---|

|[Figure 161]<br><br>[Figure 162]|
|---|

|[Figure 163]<br><br>[Figure 164]|
|---|

|[Figure 165]<br><br>[Figure 166]|
|---|

|[Figure 167]<br><br>[Figure 168]|
|---|

|[Figure 169]<br><br>[Figure 170]|
|---|

|[Figure 171]<br><br>[Figure 172]|
|---|

|[Figure 173]<br><br>[Figure 174]|
|---|

|[Figure 175]<br><br>[Figure 176]|
|---|

|[Figure 177]<br><br>[Figure 178]|
|---|

|[Figure 179]<br><br>[Figure 180]|
|---|

|[Figure 181]<br><br>[Figure 182]|
|---|

|[Figure 183]<br><br>[Figure 184]|
|---|

|[Figure 185]<br><br>[Figure 186]|
|---|

|[Figure 187]<br><br>[Figure 188]|
|---|

|[Figure 189]<br><br>[Figure 190]|
|---|

|[Figure 191]<br><br>[Figure 192]|
|---|

|[Figure 193]<br><br>[Figure 194]|
|---|

- (b) The target object can only be predicted after one watches the whole video and makes a prediction according to the video context.

[Figure 195]

[Figure 196]

Please find the wineglass in which the wine may be finished first?

[Figure 197]

<think>The instruction asks to segment the wineglass in which the wine may be finished first. The video shows two men sitting at a bar, each with a wineglass and a bottle of wine. The man on the left is gesturing with his hand, while the man

on the right is holding his wineglass. The wine in the glass of the man on the left appears to be lower than the wine in the glass of the man on the right. Therefore, the wine in the glass of the man on the left is likely to be finished first.</think> "answer": {"start_time": "00:00.00", "end_time": "00:00.50", "description": "the wineglass in which the wine may be finished first”}

Sa2VA prediction

Omni-R1 prediction

|[Figure 198]<br><br>[Figure 199]|
|---|

|[Figure 200]<br><br>[Figure 201]|
|---|

|[Figure 202]<br><br>[Figure 203]|
|---|

|[Figure 204]<br><br>[Figure 205]|
|---|

|[Figure 206]<br><br>[Figure 207]|
|---|

|[Figure 208]<br><br>[Figure 209]|
|---|

|[Figure 210]<br><br>[Figure 211]|
|---|

|[Figure 212]<br><br>[Figure 213]|
|---|

|[Figure 214]<br><br>[Figure 215]|
|---|

|[Figure 216]<br><br>[Figure 217]|
|---|

|[Figure 218]<br><br>[Figure 219]|
|---|

|[Figure 220]<br><br>[Figure 221]|
|---|

|[Figure 222]<br><br>[Figure 223]|
|---|

|[Figure 224]<br><br>[Figure 225]|
|---|

|[Figure 226]<br><br>[Figure 227]|
|---|

|[Figure 228]<br><br>[Figure 229]|
|---|

|[Figure 230]<br><br>[Figure 231]|
|---|

|[Figure 232]<br><br>[Figure 233]|
|---|

- (c) Though our method is able to select the right wineglass at System 1, it still lets System 2 analyze the detailed information. Figure 5: Visualization results of our method on three representative VOS cases. Each subfigure illustrates a different reasoning pattern.

|[Figure 234]<br><br>[Figure 235]|
|---|

|[Figure 236]<br><br>[Figure 237]|
|---|

|[Figure 238]<br><br>[Figure 239]|
|---|

|[Figure 240]<br><br>[Figure 241]|
|---|

|[Figure 242]<br><br>[Figure 243]|
|---|

|[Figure 244]<br><br>[Figure 245]|
|---|

Please segment the object(s) that serve as shock absorbers and provide protection?

[Figure 246]

[Figure 247]

<think>The instruction asks to identify the object(s) that serve as shock absorbers and provide protection. In the video, the object that serves as a shock absorber and provides protection is the sand pit where the athlete lands after the long jump. The sand pit is used to cushion the landing and prevent injury. The relevant duration is from 0.5 to 1.0 seconds, as this is when the athlete lands in the sand pit.</think>

[Figure 248]

"answer":{"start_time": "00:00.50", "end_time": "00:01.00", "description": "sand pit” }

|[Figure 249]<br><br>[Figure 250]|
|---|

|[Figure 251]<br><br>[Figure 252]|
|---|

|[Figure 253]<br><br>[Figure 254]|
|---|

|[Figure 255]<br><br>[Figure 256]|
|---|

|[Figure 257]<br><br>[Figure 258]|
|---|

|[Figure 259]<br><br>[Figure 260]|
|---|

Omni-R1 prediction

|[Figure 261]<br><br>[Figure 262]|
|---|

|[Figure 263]<br><br>[Figure 264]|
|---|

|[Figure 265]<br><br>[Figure 266]|
|---|

|[Figure 267]<br><br>[Figure 268]|
|---|

|[Figure 269]<br><br>[Figure 270]|
|---|

|[Figure 271]<br><br>[Figure 272]|
|---|

Sa2VA prediction

- (a) The case shows the video understanding abilities of our method. Our method analyses the function of the sand pit and is able to segment, while Sa2VA fails to do so.

[Figure 273]

[Figure 274]

Please segment the object(s) that making a sound after the violin.

[Figure 275]

<think>In the image, there are three musicians: one playing the violin, one playing the cello, and one playing the guitar. The violin is the first instrument to be played, and the cello is the second. The question asks for the object that makes a sound after the violin, which is the cello. The cello is positioned to the left of the violinist and is clearly visible in the image..</think> <answer>The cello</answer>

Omni-R1 prediction

|[Figure 276]<br><br>[Figure 277]|
|---|

|[Figure 278]<br><br>[Figure 279]|
|---|

|[Figure 280]<br><br>[Figure 281]|
|---|

|[Figure 282]<br><br>[Figure 283]|
|---|

|[Figure 284]<br><br>[Figure 285]|
|---|

|[Figure 286]<br><br>[Figure 287]|
|---|

|[Figure 288]<br><br>[Figure 289]|
|---|

|[Figure 290]<br><br>[Figure 291]|
|---|

|[Figure 292]<br><br>[Figure 293]|
|---|

|[Figure 294]<br><br>[Figure 295]|
|---|

[Figure 296]

- (b) The case shows the video understanding abilities of our method. Our method analyses the function of the sand pit and is able to segment, while Sa2VA fails to do so.

[Figure 297]

[Figure 298]

Please segment the object making the fastest rhythm.

[Figure 299]

<think>In the image, there are two main objects producing sound: a violinist and a pianist. The violinist is playing the violin, and the pianist is playing the piano. The violinist is likely to be making the fastest rhythm due to the nature of the violin’s sound, which can be very quick and precise. The pianist's rhythm is more controlled and less likely to be the fastest. The violinist is positioned in the foreground, and the pianist is in the background, both on a stage. The violinist is wearing a long, sparkly dress, and the pianist is wearing a black suit. The violinist is holding the violin and bow, and the pianist is at the piano. The audio is likely to be focused on the violinist's performance. </think> <answer>The violin</answer>

Omni-R1 prediction

|[Figure 300]<br><br>[Figure 301]|
|---|

|[Figure 302]<br><br>[Figure 303]|
|---|

|[Figure 304]<br><br>[Figure 305]|
|---|

|[Figure 306]<br><br>[Figure 307]|
|---|

|[Figure 308]<br><br>[Figure 309]|
|---|

|[Figure 310]<br><br>[Figure 311]|
|---|

|[Figure 312]<br><br>[Figure 313]|
|---|

|[Figure 314]<br><br>[Figure 315]|
|---|

|[Figure 316]<br><br>[Figure 317]|
|---|

|[Figure 318]<br><br>[Figure 319]|
|---|

[Figure 320]

- (c) The case shows the video understanding abilities of our method. Our method analyses the function of the sand pit and is able to segment, while Sa2VA fails to do so.

Figure 6: More visualization results of our method on representative VOS and AVS cases.

### References

- [1] Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv preprint arXiv:2501.15368, 2025. 1, 4, 9
- [2] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 1, 3, 4, 9
- [3] Chiori Hori, Takaaki Hori, Teng-Yok Lee, Ziming Zhang, Bret Harsham, John R Hershey, Tim K Marks, and Kazuhiko Sumi. Attention-based multimodal fusion for video description. In Proceedings of the IEEE international conference on computer vision, pages 4193–4202, 2017. 1
- [4] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16375–16387, 2022. 1
- [5] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1
- [6] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [7] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215,

2025. 1, 4, 8

- [8] Heqing Zou, Tianze Luo, Guiyang Xie, Fengmao Lv, Guangcong Wang, Junyang Chen, Zhuochen Wang, Hansheng Zhang, Huaijian Zhang, et al. From seconds to hours: Reviewing multimodal large language models on comprehensive long video understanding. arXiv preprint arXiv:2409.18938, 2024. 1
- [9] Yang Shi, Jiaheng Liu, Yushuo Guan, Zhenhua Wu, Yuanxing Zhang, Zihao Wang, Weihong Lin, Jingyun Hua, Zekun Wang, Xinlong Chen, et al. Mavors: Multi-granularity video representation for multimodal large language model. arXiv preprint arXiv:2504.10068, 2025. 1
- [10] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024. 1, 9
- [11] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923,

2025. 1, 3, 4

- [12] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3
- [13] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 3
- [14] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473, 2024. 3
- [15] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024. 3

- [16] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024. 3
- [17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3, 4
- [18] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3, 4
- [19] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. 3, 4
- [20] Zehan Wang, Ziang Zhang, Hang Zhang, Luping Liu, Rongjie Huang, Xize Cheng, Hengshuang Zhao, and Zhou Zhao. Omnibind: Large-scale omni multimodal representation via binding spaces. arXiv preprint arXiv:2407.11895, 2024. 3
- [21] Yaoting Wang, Peiwen Sun, Dongzhan Zhou, Guangyao Li, Honggang Zhang, and Di Hu. Refavs: Refer and segment objects in audio-visual scenes. In European Conference on Computer Vision, pages 196–213. Springer, 2024. 3, 8
- [22] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. Visa: Reasoning video object segmentation via large language models. arXiv preprint arXiv:2407.11325, 2024. 3, 8, 9
- [23] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 4
- [24] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 4
- [25] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 4
- [26] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 4
- [27] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271,

2024. 4

- [28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 4
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 4
- [30] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024. 4
- [31] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215,

2025. 4

- [32] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Jian Yang, Siwei Wu, Xingwei Qu, Jinjie Shi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Zhaoxiang Zhang, Zachary Liu, Emmanouil Benetos, Wenhao Huang, and Chenghua Lin. Omnibench: Towards the future of universal omni-language models, 2024. 4, 9
- [33] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785,

2025. 4

- [34] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 4
- [35] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025. 4, 9
- [36] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025. 4, 9, 10
- [37] Jiaxing Zhao, Xihan Wei, and Liefeng Bo. R1-omni: Explainable omni-multimodal emotion recognition with reinforcement learning, 2025. 4
- [38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 5, 8
- [39] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 7, 12
- [40] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. MeViS: A large-scale benchmark for video segmentation with motion expressions. In ICCV, 2023. 8
- [41] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. 2016. 8
- [42] Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Segzero: Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025. 8, 14
- [43] Jinxing Zhou, Jianyuan Wang, Jiayi Zhang, Weixuan Sun, Jing Zhang, Stan Birchfield, Dan Guo, Lingpeng Kong, Meng Wang, and Yiran Zhong. Avsbench: A pixel-level audio- visual segmentation benchmark. 8
- [44] Shengyi Gao, Zhe Chen, Guo Chen, Wenhai Wang, and Tong Lu. Avsegformer: Audio-visual segmentation with transformer. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 12155–12163, 2024. 8
- [45] Yaoting Wang, Weisong Liu, Guangyao Li, Jian Ding, Di Hu, and Xi Li. Prompting segmentation with sound is generalizable audio-visual source localizer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 5669–5677, 2024. 8
- [46] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4974–4984, 2022. 8
- [47] Xiang Li, Jinglu Wang, Xiaohao Xu, Xiao Li, Bhiksha Raj, and Yan Lu. Robust referring video object segmentation with cyclic structural consensus. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22236–22245, 2023. 8
- [48] Nicholas Stroh. Trackgpt–a generative pre-trained transformer for cross-domain entity trajectory forecasting. arXiv preprint arXiv:2402.00066, 2024. 9

- [49] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001, 2025. 9, 15
- [50] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 9
- [51] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024. 9
- [52] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Yangze Li, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025. 9
- [53] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075,

2024. 9

- [54] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 10
- [55] David F. Crouse. On implementing 2d rectangular assignment algorithms. IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679–1696, 2016. 12
- [56] Kim Sung-Bin, Oh Hyun-Bin, JungMok Lee, Arda Senocak, Joon Son Chung, and Tae-Hyun Oh. Avhbench: A cross-modal hallucination benchmark for audio-visual large language models. arXiv preprint arXiv:2410.18325, 2024. 15

