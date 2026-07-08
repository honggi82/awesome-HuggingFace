# arXiv:2511.13026v3[cs.CV]14May2026

## REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding

Jiaze Li1‡*, Hao Yin1‡, Wenhui Tan3‡, Jingyang Chen1‡, Boshen Xu3 Yuxun Qu1, Yijing Chen3, Jianzhong Ju1†, Zhenbo Luo1, Jian Luan1 1MiLM Plus, Xiaomi Inc. 3Renmin University of China

#### Abstract

Self-reflection mechanisms that rely on purely text-based rethinking processes perform well in most multimodal tasks. However, when directly applied to long-form video understanding scenarios, they exhibit clear limitations. The fundamental reasons for this lie in two points: (1) longform video understanding involves richer and more dynamic visual input, meaning rethinking only the text information is insufficient and necessitates a further rethinking process specifically targeting visual information; (2) purely text-based reflection mechanisms lack cross-modal interaction capabilities, preventing them from fully integrating visual information during reflection. Motivated by these insights, we propose REVISOR (REflective VIsual Segment Oriented Reasoning), a novel framework for tool-augmented multimodal reflection. REVISOR enables MLLMs to collaboratively construct introspective reflection processes across textual and visual modalities, significantly enhancing their reasoning capability for long-form video understanding. To ensure that REVISOR can learn to accurately review video segments highly relevant to the question during reinforcement learning, we designed the Dual Attribution Decoupled Reward (DADR) mechanism. Integrated into the GRPO training strategy, this mechanism enforces causal alignment between the model’s reasoning and the selected video evidence. Notably, the REVISOR framework significantly enhances long-form video understanding capability of MLLMs without requiring supplementary supervised fine-tuning or external models, achieving impressive results on four benchmarks including VideoMME, LongVideoBench, MLVU, and LVBench.

#### 1. Introduction

Multimodal reasoning is fundamental to many real-world applications, such as interpreting scientific figures, per-

*Project leader. ‡ These authors contributed equally. †Corresponding author.

forming geometric reasoning, and solving complex vision–language understanding tasks. However, when reinforcement learning (RL)-based reasoning strategies [12] are transferred from text-only models to multimodal settings [30, 36, 55], they often fail to show clear advantages over “fast-thinking” models and may even underperform in certain cases. This phenomenon primarily arises because current multimodal large language models (MLLMs) typically generate outputs under a token-level Markov assumption [26, 58], relying on local contextual dependencies. Such locality often leads to repetitive or incorrect reasoning steps [59]. Recent studies suggest that incorporating a selfreflection mechanism [16, 28] can mitigate these issues. By explicitly guiding the model to review, evaluate, and revise its reasoning trajectory, self-reflection helps prune invalid or erroneous reasoning paths, improve logical consistency, and promote deeper multimodal understanding[10, 41].

Despite significant progress in reflection mechanisms, most studies rely on text-only reconsideration processes. Such methods perform well on general multimodal tasks. However, when directly applied to long-form video understanding, their limitations become evident. As demonstrated in our experiments in Sec. 2.1, the representative text-based reflection approach, VL-Rethinker [42], leads to degraded performance in long-form video scenarios. We attribute this phenomenon to two main factors:

- • Unlike image understanding, long-form video reasoning involves richer and more dynamic visual inputs. Purely text-based reflection is insufficient to correct reasoning errors without explicitly reconsidering visual information.
- • Text-only reflection lack cross-modal interaction capabilities, preventing the model from integrating visual cues during reflection. This limitation restricts the reasoning improvement potential of MLLMs in video understanding tasks.

Motivated by the insights above, we propose REVI-

video segment for review: 220-260 seconds

full video: 297 seconds

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

###### User Question: Who is caught in the air by Superman? (A) The Flash. (B) Batman. (C) Wonder Woman. (D) Aquaman.

preliminary reasoning trace textual reflective reasoning

<think> In the fast pass through the fight sequence, I notice a red, lightning-like blur tumbling across the frame before a quick stabilization mid-air. That visual signature is commonly associated with the Flash’s motion. Interpreting the brief stabilization as Superman’s mid-air intervention, I infer that Superman momentarily catches the speedster to stop the uncontrolled trajectory. Given the red streaks and the abrupt ‘grab-and-steady’ beat, the most plausible candidate on a quick read is the Flash rather than any caped hero. </think> <answer> A </answer>

<think> On a closer pass with the 210 second frame, I see Superman elevating a caped, armored silhouette. The cowl isn’t cleanly visible due to angle and motion blur, and the rigid shoulder line plus bracer-like highlights suggest a warrior profile rather than a speedster. Since the Flash doesn’t wear a cape, my earlier read seems unlikely. Within the Trinity, a caped, armored figure in a vertical lift could be Wonder Woman after a leap or push, intercepted mid-air by Superman. Anchoring on the visible armor cues and the airborne posture, I revise toward Wonder Woman being the one Superman catches in the air. </think> <answer> C </answer>

Multimodal Reflective Reasoning 220-260 seconds

###### REVISOR: REflective VIsual Segment Oriented Reasoning

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

<think> Oh, I see. The question is asking about the specific moment in the video when Superman catches someone in the air. Around 220 seconds, I can observe Superman engaging directly with Batman — there’s clear physical confrontation suggesting that the catching moment might occur shortly after. As the scene progresses toward 260 seconds, the movement becomes more intense, indicating a possible transition to the airborne sequence. I'll use the temporal_zoom_in_tool to focus on the segment around 220 seconds to 260 seconds, so I can precisely locate the instant when Superman catches someone in the air. </think> <time_interval> [220.0, 260.0] </time_interval>

“<think> Let's analyze the video segment to pinpoint the exact moment when Superman catches someone in the air. Around 223 seconds, Superman begins to physically overpower Batman, and by the 228-second frame, Superman is visibly lifting Batman off the ground with one hand, holding him in mid-air. This visual clearly aligns with the description of Superman ‘catching someone in the air.’ To confirm this interpretation and capture the transition from grounded combat to aerial suspension, I'll use the temporal_zoom_in_tool to focus on the segment around 220 seconds to 260 seconds. This narrowed interval allows for precise observation of the moment Superman lifts Batman, which directly supports the answer choice. </think> <answer> B </answer>

full reasoning process

- Figure 1. Operational workflow of the proposed REVISOR framework, contrasting it with traditional reflection mechanisms. The top panel illustrates a typical traditional approach, often employing a text-based re-evaluation mechanism. In contrast, the bottom panel details the REVISOR framework. This process involves two distinct stages: (1) Initial Inference, which generates a preliminary reasoning trace and identifies critical regions for detailed analysis; and (2) Reflective Reasoning, which integrates this initial trace with newly sampled, fine-grained visual evidence to yield a refined and robust final prediction.

SOR (REflective VIsual Segment Oriented Reasoning), a novel two-stage reasoning framework designed to enhance video understanding. REVISOR leverages a toolaugmented multimodal reflection mechanism that enables MLLMs to collaboratively construct introspective reflection processes across textual and visual modalities, thereby enhancing reasoning capability. As illustrated in Fig. 1, REVISOR comprises an MLLM that collaborates with a visual toolbox. In Stage 1, the MLLM performs an initial reasoning step, identifies video segments requiring further examination, and invokes the visual toolbox to resample key frames from these segments as supplementary inputs. In Stage 2, the MLLM integrates the initial reasoning trace with the reviewed visual frames to iteratively refine its reasoning, ultimately generating a more accurate response.

Relying solely on verification-based reinforcement from the final answer during GRPO [35] training constrains REVISOR’s capacity to identify question-relevant review segments in its reflection stage. To overcome this limitation, we propose a Dual Attribution Decoupled Reward (DADR) mechanism in Sec. 3.2, which supplements the final-answer verification reward with a Causal Segment Sufficiency Reward (CSSR). The CSSR enforces causal alignment between the model’s reasoning and the selected video evidence, rewarding correctness only when the answer is de-

rived exclusively from those segments. This mechanism explicitly encourages the model to focus on informative visual cues while implicitly discouraging reliance on irrelevant or spurious content, thereby enhancing REVISOR’s multimodal reasoning consistency and overall robustness.

The REVISOR framework significantly enhances the video understanding capabilities of MLLMs without requiring supplementary supervised fine-tuning (SFT) or external models. Across the VideoMME [9], LongVideoBench [48], MLVU [62], and LVBench [43] benchmarks, REVISOR consistently improves the base model’s average accuracy by about 2%. Extended analysis in Sec. 5 validates two key drivers of REVISOR’s success: (1) for long-form video understanding, revisiting and refining visual information is more critical than rethinking the textual reasoning process; and (2) DADR mechanism ensures the precise recall and utilization of salient visual cues.

In summary, the contributions of this paper are threefold:

- • We diagnose the root cause for the poor performance of conventional text-based self-reflection mechanisms when applied to long-form video understanding.
- • We propose REVISOR, a two-stage reasoning framework that transforms the conventional text-based self-reflection mechanism into a tool-augmented multimodal one, significantly enhancing the model’s reasoning capability and

MathVista

Qwen2.5-VL-7B

74.9

VL-Rethinker-7B

68.2

MathVerse

EMMA

54.2

29.7

46.3

21.5

32.3

25.1

56.7

54.3

41.7

36.9

MathVision

MMMU

MMMU-Pro

[Figure 15]

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Textual reflection mechanisms excel in image understanding

Textual reflection mechanisms have struggled with long-form video understanding

Qwen2.5-VL-7B Qwen2.5-VL-7B + Ground-Truth Visual Reflection

0

CG-Bench

80

10

ReXTime

70

20

NExT-GQA

60

30

50 40

Revisiting critical video segments in the reflection stage can solve the problem

- Figure 2. Motivation for proposing a multimodal reflection mechanism. Left: Text-only reflection mechanisms, such as VL-Rethinker, achieve significant performance improvements in image understanding tasks. Middle: However, applying the same text-based reflection strategy to long-form video understanding leads to performance degradation. Right: Incorporating a revisit of key video segments during the reflection stage effectively improves performance on video understanding tasks.

accuracy in long-form video understanding tasks.

• We integrate the DADR mechanism into the GRPO method, enabling MLLMs to learn how to review the correct video content during self-reflection within the RL training process, thereby maximizing REVISOR’s performance boost on long-form video comprehension.

#### 2. Motivation

Sec. 2.1 shows that text-only reflection mechanisms are inadequate for long-form video understanding. The underlying reason is that, unlike static images, long-form videos present far richer and more dynamic visual information, making purely textual re-evaluation insufficient for correcting reasoning errors. In Sec. 2.2, we initially demonstrated the importance of incorporating a visual re-evaluation process during the reflection stage, with preliminary experiments showing that this significantly enhances the accuracy of MLLMs in long-form video understanding tasks.

##### 2.1. Dilemma of Text-Based Reflection

Self-reflection mechanisms, a pivotal strategy for enhancing model performance, were initially applied to LLMs to bolster their complex reasoning capabilities. This approach has since demonstrated promising outcomes when extended to multimodal contexts. As illustrated in the left part of Fig. 2, the representative VL-Rethinker method demonstrates substantial performance improvements across various image understanding benchmarks. However, despite their operation within multimodal frameworks, the reflection processes of such mechanisms remain purely text-based. They do not incorporate re-evaluation of visual inputs, thereby classifying them as text-driven reflection mechanisms.

As shown in the middle part of Fig. 2, applying these text-based reflection mechanisms to long-form video un-

derstanding tasks paradoxically leads to a decline in model performance. To eliminate the possibility that this degradation stems from VL-Rethinker’s lack of video training, we developed a comparable model equipped with a purely textbased reflection mechanism and trained it on video data. Details are provided in Appendix Sec. 9. Our experiments revealed that this model also failed to improve performance in long-form video understanding tasks.

We attribute this phenomenon to two primary factors: (1) Compared to static images, long-form videos encompass significantly richer and more dynamic visual content. Thus, re-evaluating only text-based representations is inadequate for correcting reasoning errors, necessitating a visual reevaluation process. (2) Purely text-based reflection mechanisms inherently lack cross-modal interaction capabilities, preventing full integration of visual information during reflection and thereby limiting improvements in MLLMs for long-form video understanding.

##### 2.2. Necessity of Integrating Visual Rethinking

To substantiate the claim advanced in Sec. 2.1, specifically that re-evaluating visual information can improve MLLMs’ accuracy in long-form video comprehension, we conducted a preliminary validation experiment. In datasets such as NExT-GQA [50], ReXTime [4], and CG-Bench [3], the critical video segments necessary for answering each question, representing only a small fraction of the total footage, are annotated. Using these datasets, we first allowed MLLMs to perform initial reasoning based on the raw video frames and questions. Subsequently, we provided the models with the annotated key segments and instructed them to re-evaluate the problem, integrating their previous reasoning with the newly supplied visual cues to generate the final answers.

As illustrated in the right part of Fig. 2, introducing a vi-

sual rethinking process during the reflection phase yields an average accuracy gain of approximately 7.3% on the NExTGQA, ReXTime, and CG-Bench datasets. In contrast, reflection based solely on textual information results in negligible improvement. These results substantiate two key insights: (1) In the domain of video understanding, visual reflection plays a far more crucial role than textual reflection; and (2) when MLLMs effectively revisit key visual cues during this rethinking stage, their accuracy in long-form video comprehension tasks can be substantially enhanced.

#### 3. Methodology

In this section, we present REVISOR (REflective VIsual Segment Oriented Reasoning), a novel two-stage reasoning framework designed to enhance long-form video understanding. REVISOR leverages a tool-augmented multimodal reflection mechanism that enables MLLMs to collaboratively construct reflection processes across textual and visual modalities, enhancing their reasoning capability.

##### 3.1. Tool-Augmented Multimodal Reflection

The core concept of REVISOR involves a two-stage reasoning framework: (1) an initial inference stage that generates a preliminary reasoning trace and identifies key moments for further examination, and (2) a reflective reasoning stage that integrates the initial analysis with newly sampled, finegrained visual evidence to yield a refined final answer. An overview of this information flow is illustrated in Fig. 1.

Stage 1: Initial Inference and Visual Review Proposal. The process begins with a video V and a user-posed question Q. To mitigate the computational cost and contextlength limitations of the MLLM, we first perform a sparse, uniform sampling of frames from the video, producing an initial frame set Finit = f1,f2,...,fN, where N denotes the number of sampled frames.

The MLLM, denoted as M, takes as input a question Q and the initial set of frames Finit. The model is prompted to engage in a chain-of-thought reasoning process to generate a preliminary reasoning trace. Importantly, beyond producing the reasoning content, the model is instructed to identify and output a temporal segment S that it considers most relevant or uncertain with respect to its conclusion. This output is represented as a structured tuple containing the initial reasoning trace Tinit and the proposed review segment S. We formalize this stage as:

(Tinit,S) = Minfer(Q,Finit), (1)

where Minfer denotes the MLLM operating in its initial inference mode. The segment S = [tstart,tend] specifies the start and end timestamps of the video interval that warrants closer examination. The model’s ability to propose S emerges naturally from its reasoning process Tinit, through

which it articulates which portions of the video were most pivotal or ambiguous.

Visual Toolbox Call: Retrieve Review Segment Frames. Upon receiving the proposed segment S from the MLLM, the REVISOR framework engages the Visual Toolbox, denoted as T . The role of T is to perform a targeted, highdensity re-sampling of frames from the original video V , restricted to the temporal window specified by S.

For a segment S = [tstart,tend], the toolbox generates a denser frame sequence, formalized as:

###### Freview = SampleDense(V,[tstart,tend]), (2)

where SampleDense(·) denotes a sampling function that extracts frames at a higher temporal resolution (e.g., increased FPS) compared to the initial sparse sampling. This toolassisted process offloads the computational and procedural burden of locating fine-grained visual details, enabling the MLLM to conduct a focused examination of critical moments without processing the entire video at full resolution.

Stage 2: Reflective Reasoning and Answer Refinement. The final stage centers on reflection and self-correction. The MLLM is re-invoked, now operating within a richer and more structured contextual frame. The inputs to this reflective stage are: (1) the original question Q, (2) the initial reasoning trace Tinit, and (3) the newly acquired, densely sampled visual evidence Freview.

By providing the model with access to its own initial reasoning Tinit, the REVISOR framework enables a form of in-context reflection, prompting the model to reassess its preliminary conclusions in light of the enhanced visual data from Freview. This process allows the MLLM to validate its earlier hypotheses, resolve ambiguities identified in Stage 1, or correct prior misinterpretations. The generation of the refined reasoning trace Trefine and final answer Afinal is formalized as:

(Trefine,Afinal) = Mreflect(Q,Tinit,Freview), (3)

where Mreflect denotes the MLLM operating in its reflection mode. This reflective mechanism mirrors human expert analysis, where an initial overview is followed by a focused examination of critical evidence before forming a conclusion. By structuring the MLLM’s reasoning in this iterative manner, REVISOR enhances both the accuracy and reliability of responses in complex video understanding tasks.

##### 3.2. Dual Attribution Decoupled Reward

In standard practice, reward signals based on the verification of the final answer are integrated with reinforcement learning to train the aforementioned tool-augmented MLLM in an end-to-end manner. Specifically, the GRPO algorithm is commonly employed to maximize the expected

Preliminary Reasoning Phase

Multi-modal Reflection Phase

preliminary sparsely sampling video frames reasoning trace

final answer

final answer verification reward

proposed review segment

reflective reasoning

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Video-LLM

Video-LLM

rollout

in-context reflection

user question

policy optimization

policy optimization

standard reasoning

attributed answer

causal segment sufficiency reward

dense frame resampling

Visual Toolbox

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Video-LLM

standard generation

Dual Attribution Decoupled Reward Mechanism

user question

Ensuring the Proposed Segment is Sufficient to Derive the Answer

- Figure 3. Overview of the Dual-Attribution Decoupled Reward Mechanism (DADR). Final Answer Verification Reward (top) is derived from verifying the correctness of the model’s synthesized final answer, directly targeting the accuracy objective of the reflective stage. Conversely, Causal Segment Sufficiency Reward (bottom) is granted upon verifying an attribution answer derived exclusively from reviewed video segments, thereby guiding the model to identify and utilize segments highly pertinent to the user query.

total reward, thereby stabilizing the training process and encouraging exploration within the complex video–temporal action space. The policy πθ parameterized by θ generates a reasoning trajectory τ = (Tinit,S,Trefine,Afinal) based on the input (Q,V ). The overall objective is to maximize the expected cumulative reward J(θ):

θ|(Q,V )[R(τ)]. (4)

J(θ) = E(Q,V )∼D,τ∼π

However, as shown in Tab. 3, training the REVISOR framework using the GRPO algorithm with final-answer verification as the sole reward function leads to decreased performance on long-form video understanding tasks compared to the base model. This limitation primarily arises from the complex nature of the REVISOR reasoning process, which consists of three components: the initial reasoning content, the video review segmentation interval, and the reflective reasoning phase. Consequently, even when the model correctly outputs the video review segments, it receives insufficient positive feedback; conversely, incorrect segment predictions fail to incur adequate penalties. This imbalance prevents the model from effectively learning to identify key video segments during reinforcement learning (see Sec. 5.2 for supporting evidence).

To address this issue, we decouple the reward for video review segment localization from the general reward signal and propose a Dual-Attribution Decoupled Reward Mechanism (DADR), as shown in Fig. 3. This mechanism integrates the reward for the final output with an additional causal reward, ensuring that the proposed segment S is sufficient for deriving the correct conclusion. The total reward

R(τ) is defined as a weighted sum of two distinct components: the Final Answer Verification Reward (Rfinal) and the Causal Segment Sufficiency Reward (Rcausal):

R(τ) = λ1Rfinal + λ2Rcausal, (5)

where λ1 and λ2 are hyperparameters controlling the balance of the rewards. The term Rfinal is a standard taskspecific reward that validates the correctness of the final refined answer Afinal against the ground truth answer A∗.

Causal Segment Sufficiency Reward. The Causal Segment Sufficiency Reward (CSSR) is designed to enforce the quality of the proposed review segment S (generated in Stage 1). The CSSR provides a positive signal only if the model is capable of deriving the correct answer A∗ solely based on the question Q and the densely sampled visual evidence Freview extracted from S. This implicitly encourages the MLLM to select segments that are truly causal and sufficient for the task, preventing the selection of irrelevant or overly long segments.

To formalize this, we define an attribution-based answer Aˆ using the same MLLM M, but conditioned solely on the original question Q and the fine-grained visual evidence Freview. This step serves as a crucial sufficiency test for the proposed segment S. Let Msuff denote this sufficiency prediction mode:

Aˆ = Msuff(Q,Freview). (6)

The Causal Segment Sufficiency Reward is then defined based on the correctness of this attribution-based prediction:

Rcausal = I(Aˆ = A∗). (7)

- Table 1. Evaluation results of the four long-form video understanding benchmarks. ∗ indicates models trained with the text-based reflection mechanism using the same dataset as ours. ⋆ represents our reproduction. ↑ highlights the superior performance of the REVISOR framework relative to the base model Qwen2.5-VL-7B.

VideoMME Model Model Size Video Tokens

LongVideoBench MLVU LVBench Gemini-1.5-Pro[39] - - 67.4 75 64 - -

Long Overall

GPT4o[14] - - 65.3 71.9 66.7 64.6 ShareGPT4Video[5] 8B - 37.9 43.6 39.7 46.4 -

Video-LLaVA[22] 7B - 38.1 40.4 39.1 47.3 LongVA[60] 7B 224K 47.6 54.3 - 56.3 LongVU[37] 7B 8K 59.5 60.6 - 65.4 -

Vamba[34] 10B - - 57.8 55.9 65.9 42.1 LLaVA-OneVision[17] 7B 6K - 46.7 56.4 64.7 -

Hour-LLaVA[23] 7B - 55 63.6 60.4 - 45.6 VideoChat-Flash[19] 7B 8K 55.4 65.3 - 74.7 48.2

NVILA[25] 8B 8K 54.8 64.2 70.1 57.7 Open-o3-Video[31] 7B 2K 54.9 63.6 - - Video-MTR[51] 7B 4K 51.0 59.0 - 48.4 -

Video-R1[8] 7B 8K - 61.4 - - LongVILA-R1[7] 7B - 55.2 65.1 58 - -

VL-Rethinker[42] 7B 8K 51.9 62.1 56.4 63.2 37.2 Qwen2.5-VL∗[2] 7B 8K 53.2 63.4 57.4 69 39.2 Qwen2.5-VL⋆[2] 7B 8K 53.4 64.3 56.5 67.3 40.2

Ours 7B 8K 56.2 ↑2.8 65.7 ↑1.4 57.5 ↑1.0 69.8 ↑2.5 42 ↑1.8

Incorporating Rcausal into the GRPO optimization loop guides the REVISOR framework toward policies πθ that not only yield accurate final answers but also robustly identify the critical temporal evidence required for those answers in the initial inference stage.

#### 4. Experiments

We first detail our experimental setup in Sec. 8.1. The performance of the REVISOR framework for long-form video understanding tasks is then presented in Sec. 4.2. Subsequently, Sec. 4.3 conducts an ablation study to evaluate the contribution of REVISOR’s various components. Please refer to Appendix Sec. 10 for the Case Study of REVISOR.

##### 4.1. Experimental Setup

Training Details. We adopt Qwen2.5-VL-7B [2] as our base model. The training is conducted using the verl framework [38], which we further extend to support the training of REVISOR. Our training process consists of a singlestage reinforcement learning phase following DAPO. [58].

Dataset Construction. We use a total of 25K training samples collected from STAR [20], PerceptionTest [32], NExT-QA [49], CLEVRER [57], LLaVA-Video-178K [61], TimeRFT [45], CG-Bench [3], and ReXTime [4]. Detailed information regarding the dataset selection procedure is available in Appendix Sec. 8.2.

Hyperparameter Selection. In our experiments, we set λ1 = 0.6 and λ2 = 0.3. We adopt AdamW [27] as the optimizer with a learning rate of 1 × 10−6 and a batch size of 32. The number of rollouts is set to 8. During both training and evaluation, the total number of video tokens in the input is limited to a maximum of 8192. For more details, please refer to Appendix Sec. 8.1

##### 4.2. Main Results

Long-Form Video Understanding Task. We evaluate our method on four widely used long-form video benchmarks: VideoMME [9], LongVideoBench [48], MLVU [62], and LVBench [43]. As shown in Tab. 1, with an input of 8K video tokens, REVISOR achieves 65.7% on VideoMME, outperforming the base model Qwen2.5-VL-7B by 1.4%. Notably, REVISOR outperforms the base model by 2.8% on the long subset of VideoMME and by 2.5% on MLVU, which contains videos up to 120 minutes. This indicates that as video duration increases, accurately reviewing the relevant video content becomes increasingly crucial.

Compared with the latest text-only reasoning approach Video-R1 [8], REVISOR achieves a 4.3% improvement on VideoMME, demonstrating that the integration of the DADR mechanism provides a clear advantage over pure text-based reasoning. Furthermore, compared with textbased reflection methods such as VL-Rethinker and our own

video-data-trained, text-only reflection method, REVISOR shows gains of 3.6% and 2.3%, respectively, highlighting the necessity of Visual Rethinking within REVISOR.

- Table 2. Evaluation results for the temporal video grounding task, including the Charades-STA and NExT-GQA benchmarks. Bold fonts highlight the best performance.

Charades-STA NExT-GQA Model

R@0.7 mIoU R@0.7 mIoU Qwen2.5-VL-7B[2] 15.5 36.9 7.5 20.9

VTimeLLM[13] 14.7 34.6 9.7 24.4 iMOVE[18] 26.1 47.3 - TimeChat[33] 13.4 - 6.2 20.6 VideoChat-TPO[53] 18.4 38.2 8.2 27.7

TVG-R1[6] 23.9 46.7 10 29.3 Ours 31.8 51.4 11.9 33.2

Temporal Video Grounding Task. As shown in Tab. 2, REVISOR achieves 51.4% mIoU on Charades-STA [11], surpassing the prior SFT-based SOTA method iMOVE [18] by 4.1% mIoU. Moreover, it outperforms the RL-based temporal grounding model TVG-R1 [6] by 4.7% mIoU on Charades-STA and 3.9% mIoU on NExT-GQA [50], showing its superior capability in temporal video grounding.

- Table 3. Ablation study of the Dual Attribution Decoupled Reward mechanism. V-MME, LongVB, LV, and NExT-G represent VideoMME, LongVideoBench, LVBench, and NExT-GQA, respectively. Bold fonts highlight the best performance. The row marked in gray represents our base model.

λ1 λ2 V-MME LongVB LV MLVU NExT-G - - 64.3 56.5 40.2 67.3 20.87

0.3 0.6 64.0 56.0 41.1 68.7 33.9 0.6 0.0 62.2 54.0 40.8 68.3 32.1 0.6 0.3 65.7 57.5 42.0 69.8 33.2

- 4.3. Ablation Study

Dual Attribution Decoupled Reward. We perform an indepth investigation of the DADR mechanism. As shown in Tab. 3, setting λ2 to 0 (i.e., using only the Final Answer Verification Reward) results in a drop of REVISOR’s Video-MME score from 65.7% to 62.2%, which is below the 64.3% achieved by the base model Qwen2.5-VL-7B. This result suggests that, without the CSSR component, the model fails to learn how to locate the correct review segment S from sparse reward signals. Please refer to Appendix Sec. 8.3 for more ablation study results.

When the value of λ2 exceeds that of λ1, the model’s temporal video grounding ability improves, but its longform video understanding performance declines (e.g.,

MLVU drops from 69.8% to 68.7%). This occurs because the model becomes overly focused on locating the correct review segment S, while neglecting how to utilize S to derive the correct answer. Therefore, we set the value of λ1 to be greater than that of λ2 to encourage the model not only to accurately identify the correct review segment S but also to leverage it to enhance its reasoning capability.

Table 4. Ablation study of the training data composition. SVQ, TRF, RXT, and CGB represent short video QA, TimeRFT, ReXTime, and CG-Bench, respectively. V-MME, LVB, LV, and NExTG represent VideoMME, LongVideoBench, LVBench, and NExTGQA, respectively. Bold fonts highlight the best performance. The row marked in gray represents our base model.

SVQ TRF RXT CGB V-MME LVB LV MLVU NExT-G - - - - 64.3 56.5 40.2 67.3 20.9

✓ 62.6 52.7 38.6 66.7 25.0 ✓ ✓ 63.8 56.8 40.3 67.6 22.6 ✓ ✓ ✓ 64.0 57.1 40.5 68.4 29.5 ✓ ✓ ✓ ✓ 65.7 57.5 42.0 69.8 33.2

Data Composition. As shown in Tab. 4, REVISOR’s longform video understanding capability improves with the inclusion of more diverse datasets. Specifically, its performance on MLVU increases from 66.7% to 69.8%. Notably, even when trained exclusively on short video QA datasets, REVISOR’s mIoU on NExT-GQA rises from 20.9% to 25%. This result demonstrates that, when combined with the DADR mechanism, our framework can effectively translate its understanding into the accurate localization of the correct review segment S.

#### 5. Extended Analysis

This section details comprehensive analytical experiments validating two key factors crucial for the REVISOR framework’s success: (1) For long-form video understanding tasks, the re-examination of visual information proves significantly more critical than that of textual inference processes. (2) The DADR mechanism effectively enhances the MLLM’s precision in retrieving critical visual information.

##### 5.1. Importance of Visual Rethinking

Sec. 2 posited and preliminarily demonstrated that the limitations of purely text-based reflection in long-form video understanding arise from the inherently richer and more dynamic nature of long-form video inputs compared to static images. Reflection based solely on textual information is insufficient to correct reasoning errors, necessitating the integration of a dedicated visual reflection process. In this section, we provide further indirect evidence supporting this claim through an analysis of REVISOR’s output.

Left part of Fig. 4 depicts the evolution of textual reasoning length and video review length during REVISOR’s reflection stage throughout training. As training progresses, the length of textual reasoning steadily decreases. This trend clearly indicates that, through interaction with the environment during reinforcement learning, the model gradually learns that textual reflection plays a relatively minor role in long-form video understanding tasks. In contrast, the length of the reviewed video segments first increases and then decreases. We attribute this to the model initially expanding its search range for critical video segments to ensure the inclusion of highly relevant content, and later learning to discard redundant portions. This refinement enables the model to precisely locate the minimal video segments sufficient for answering the given question.

VideoMME(short)

###### Unrestricted Thinking Forced Thinking

73.1

71.7

VideoMME(long)

LVBench

51

49.9

39.2

35.7

62.5

66.2

60.3

63.5

54.9

52.6

VideoMME (all)

MLVU

LongVideoBench

Figure 4. The superior efficacy of visual reflection over textual reflection in long-form video understanding. The left panel demonstrates that the length of the generated textual reflection consistently decreases throughout training. The right panel further indicates that forcing the model to perform longer textual reflection actually leads to a degradation in model performance.

Building on these findings, we conducted further comparative experiments on the length of the text generated during the reflection process. By encouraging the model to engage in deeper deliberation through the system prompt, we made the model output more extensive textual reasoning during the reflection stage. However, as shown in the right part of Fig. 4, this intervention paradoxically led to a decline in the model’s performance on the video understanding task. This result further substantiates that pure textual reflection is not critical for video understanding tasks; instead, generating more ineffectual reasoning content can have a negative impact on performance.

##### 5.2. Accuracy of Visual Information Retrieval

In Sec. 3.2, we propose a DADR mechanism to enable more accurate revisiting of key video segments during the reflection phase. To assess its efficacy, we conduct targeted experiments on the video segments REVISOR processes within the visual reflection stage.

We aim to examine the effect of incorporating the DADR mechanism during training on the accuracy of retrospective video segment extraction. To this end, we conducted an experiment in which models trained with and without the

DADR mechanism were required to extract key video segments necessary for answering specific questions. These extracted segments were subsequently provided to the original Qwen2.5-VL-7B model, which generated responses based solely on the given segments. As the model architecture and parameters remained identical, the correctness of the responses depended entirely on the relevance of the retrieved video segments to the corresponding questions. Thus, by comparing the answer accuracy of Qwen2.5-VL7B under both conditions, we can indirectly assess the precision of the video segment extraction process.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Figure 5. Comparative accuracy of key moment review across different methods. Methods based on the REVISOR framework and its variants are highlighted in blue, while different Temporal Grounding baselines are represented in green.

As shown in Fig. 5, when the DADR mechanism is applied during training, the video segments revisited by the model during the reflection phase are generally more accurate. This clearly demonstrates that the DADR mechanism enables MLLMs to learn to recall key video segments during the reinforcement learning process. Furthermore, we conducted a comparative analysis between the DADR mechanism and the four temporal grounding methods TimeR1 [45], TVG-R1 [6], Open-o3-Video [31], and VideoChatR1.5 [52], and found that DADR still achieves the highest accuracy in recalling relevant video segments.

#### 6. Conclusion

Text-based reflection fails to capture the complexity of long-form video reasoning, which depends critically on rich and dynamic visual information. We propose REVISOR, a tool-augmented multimodal reflection framework that revisits essential visual content to correct errors arising in the initial reasoning process. To train REVISOR to precisely localize the video segments most relevant to each query, we incorporate a Dual Attribution Decoupled Reward (DADR) mechanism into GRPO. Extensive experiments across four long-form video understanding benchmarks show that REVISOR consistently delivers significant performance gains. For related work, please refer to Appendix Sec. 7.

#### References

- [1] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457, 2025. 1
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6, 7, 2
- [3] Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, Jilan Xu, Yali Wang, Tong Lu, and Limin Wang. Cgbench: Clue-grounded question answering benchmark for long video understanding. arXiv preprint arXiv:2412.12075,

2024. 3, 6

- [4] Jr-Jen Chen, Yu-Chien Liao, Hsi-Che Lin, Yu-Chu Yu, YenChun Chen, and Frank Wang. Rextime: A benchmark suite for reasoning-across-time in videos. Advances in Neural Information Processing Systems, 37:28662–28673, 2024. 3, 6
- [5] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472–19495, 2024. 6
- [6] Ruizhe Chen, Tianze Luo, Zhiting Fan, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, and Zhang Huaijian. Datasets and recipes for video temporal grounding via reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 983–992, 2025. 7, 8, 2
- [7] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. arXiv preprint arXiv:2507.07966, 2025. 6
- [8] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 6

- [9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 2, 6
- [10] Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025. 1
- [11] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 7

- [12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1
- [13] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14271–14280, 2024. 7, 2
- [14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 6
- [15] Pu Jian, Junhong Wu, Wei Sun, Chen Wang, Shuo Ren, and Jiajun Zhang. Look again, think slowly: Enhancing visual reflection in vision-language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9262–9281, 2025. 1
- [16] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024. 1
- [17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6
- [18] Jiaze Li, Yaya Shi, Zongyang Ma, Haoran Xu, Huihui Xiao, Ruiwen Kang, Fan Yang, Tingting Gao, Di Zhang, et al. imove: Instance-motion-aware video understanding. In Findings of the Association for Computational Linguistics: ACL 2025, pages 23959–23975, 2025. 7, 2
- [19] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. Videochatflash: Hierarchical compression for long-context video modeling, 2025. 6
- [20] Yansheng Li, Linlin Wang, Tingzhu Wang, Xue Yang, Junwei Luo, Qi Wang, Youming Deng, Wenbin Wang, Xian Sun, Haifeng Li, et al. Star: A first-ever dataset and a large-scale benchmark for scene graph generation in large-size satellite imagery. IEEE Trans. Pattern Anal. Mach. Intell, 47(3): 1832–1849, 2025. 6
- [21] Hao Liang, Jiapeng Li, Tianyi Bai, Xijie Huang, Linzhuang Sun, Zhengren Wang, Conghui He, Bin Cui, Chong Chen, and Wentao Zhang. Keyvideollm: Towards large-scale video keyframe selection, 2024. 1
- [22] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5971–5984, 2024. 6
- [23] Jingyang Lin, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu, Yusheng Su, Xiaodong Yu, Hao Chen, Jiebo Luo, Zicheng Liu, et al. Unleashing hour-scale video training for long video-language understanding. arXiv preprint arXiv:2506.05332, 2025. 6

- [24] Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, and Mike Zheng Shou. Videomind: A chain-of-lora agent for long video reasoning, 2025. 1
- [25] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4122–4134, 2025. 6
- [26] Jieyi Long. Large language model guided tree-of-thought. arXiv preprint arXiv:2305.08291, 2023. 1
- [27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [28] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023. 1
- [29] Yuanbin Man, Ying Huang, Chengming Zhang, Bingzhe Li, Wei Niu, and Miao Yin. Adacm2: On understanding extremely long-term video with adaptive cross-modality memory reduction, 2025. 1
- [30] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 1
- [31] Jiahao Meng, Xiangtai Li, Haochen Wang, Yue Tan, Tao Zhang, Lingdong Kong, Yunhai Tong, Anran Wang, Zhiyang Teng, Yujing Wang, et al. Open-o3 video: Grounded video reasoning with explicit spatio-temporal evidence. arXiv preprint arXiv:2510.20579, 2025. 6, 8
- [32] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023. 6
- [33] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 7, 2
- [34] Weiming Ren, Wentao Ma, Huan Yang, Cong Wei, Ge Zhang, and Wenhu Chen. Vamba: Understanding hourlong videos with hybrid mamba-transformers. arXiv preprint arXiv:2503.11579, 2025. 6
- [35] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2
- [36] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 1

- [37] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 6, 1
- [38] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025. 6
- [39] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 6
- [40] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, Xinchi Chen, Jun Zhao, Xuanjing Huang, and Xipeng Qiu. Thinking with video: Video generation as a promising multimodal reasoning paradigm,

2025. 1

- [41] Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, et al. Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning. arXiv preprint arXiv:2506.01713, 2025. 1
- [42] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 1, 6
- [43] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22958–22967, 2025. 2, 6
- [44] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena YeungLevy. Videoagent: Long-form video understanding with large language model as agent, 2024. 1
- [45] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025. 6, 8
- [46] Zikang Wang, Boyu Chen, Zhengrong Yue, Yi Wang, Yu Qiao, Limin Wang, and Yali Wang. Videochat-a1: Thinking with long videos by chain-of-shot reasoning, 2025. 1
- [47] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos, 2025. 1
- [48] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 2, 6
- [49] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining

- temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 6
- [50] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13204– 13214, 2024. 3, 7
- [51] Yuan Xie, Tianshui Chen, Zheng Ge, and Lionel Ni. Videomtr: Reinforced multi-turn reasoning for long video understanding. arXiv preprint arXiv:2508.20478, 2025. 6
- [52] Ziang Yan, Xinhao Li, Yinan He, Zhengrong Yue, Xiangyu Zeng, Yali Wang, Yu Qiao, Limin Wang, and Yi Wang. Videochat-r1. 5: Visual test-time scaling to reinforce multimodal reasoning by iterative perception. arXiv preprint arXiv:2509.21100, 2025. 8
- [53] Ziang Yan, Zhilin Li, Yinan He, Chenting Wang, Kunchang Li, Xinhao Li, Xiangyu Zeng, Zilei Wang, Yali Wang, Yu Qiao, et al. Task preference optimization: Improving multimodal large language models with vision task alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29880–29892, 2025. 7, 2
- [54] Shuo Yang, Yuwei Niu, Yuyang Liu, Yang Ye, Bin Lin, and Li Yuan. Look-back: Implicit visual re-focusing in mllm reasoning. arXiv preprint arXiv:2507.03019, 2025. 1
- [55] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 1
- [56] Linli Yao, Haoning Wu, Kun Ouyang, Yuanxing Zhang, Caiming Xiong, Bei Chen, Xu Sun, and Junnan Li. Generative frame sampler for long video understanding, 2025. 1
- [57] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019. 6
- [58] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 1, 6
- [59] Jinghan Zhang, Xiting Wang, Fengran Mo, Yeyang Zhou, Wanfu Gao, and Kunpeng Liu. Entropy-based exploration conduction for multi-step reasoning. arXiv preprint arXiv:2503.15848, 2025. 1
- [60] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 6
- [61] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 6
- [62] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping

Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13691– 13701, 2025. 2, 6

## REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding

### Supplementary Material

#### 7. Related Work

Long-Form Video Understanding. In long-form video understanding, the visual inputs are far more complex than those in image-based tasks. Consequently, identifying and distilling only the question-relevant information from large amounts of redundant visual content becomes crucial for improving model performance. Existing approaches can be grouped into three categories: external-model augmentation, agentic methods, and model-internalized selection.

External-model augmentation techniques [21, 29, 37], rely on vision–text similarity models like CLIP for keyframe selection. While efficient, these methods perform static, one-shot selection, limiting their adaptability to complex or evolving queries. Agentic approaches [24, 46] address this limitation through iterative frameworks that dynamically refine frame selection. For example, VideoAgent [44] employs an LLM as a central agent in a state–action–observation loop, whereas VideoTree [47] adopts a hierarchical tree structure to perform coarse-to-fine search, directing computation toward the most relevant segments. More recent model-internalized selection methods [40] embed frame selection directly into the model’s reasoning process. GenS [56] trains a lightweight generative sampler to identify query-relevant frames end to end, though the selection and QA stages remain decoupled.

Our proposed REVISOR framework, after reinforcement learning training, can autonomously identify and explore visual content that requires additional careful review during the inference stage. Unlike previous methods that are limited to statically searching for important video segments based solely on the question itself, our approach fully integrates with the reasoning capabilities of MLLMs. This allows it to engage in deeper deliberation and, through comprehensive interaction with its ongoing reasoning, more precisely pinpoint critical visual information, thereby significantly enhancing the model’s reasoning ability.

Self-Reflection Mechanism. Incorporating self-reflection into inference has been shown to improve model performance [12]. Self-Refine [28] implements an iterative feedback–revision loop without requiring additional training or external supervision. SCoRe [16] adopts multi-round reinforcement learning to cultivate reflective reasoning skills, yielding substantial gains. GEPA [1] further introduces a Pareto-based reflective prompt-evolution framework that outperforms reinforcement learning in both sample efficiency and overall performance.

Extending self-reflection to multimodal settings likewise yields consistent improvements [15, 54]. VL-Rethinker [42] incorporates a self-verification–and-correction stage via reinforcement learning, substantially enhancing multimodal reasoning. SRPO [41] further strengthens selfreflection and self-correction by curating reflection-oriented supervised data and introducing a reflection-aware reward under group relative policy optimization, significantly boosting complex multimodal reasoning.

However, existing multimodal self-reflection mechanisms still depend on text-based re-evaluation processes. They lack explicit reflective operations over visual information and therefore remain fundamentally text-centric. The REVISOR framework proposed in this paper addresses this limitation by extending text-centric reflection into a truly multimodal reflective process.

921

1167

2500

2088

2500

1224

Training Data (25K)

1270

14561

1510

20856

2291

Grouded Video QA (2K)

CG-Bench (1167) ReXTime (921)

Short Video QA (20K)

LLaVA-Video-178K (14561) CLEVRER (1224) NExT-QA (1510) PerceptionTest (1270) STAR (2291)

Temporal Grounding (2.5K)

TimeRFT (2500)

Figure 6. REVISOR framework training dataset composition. The training dataset for the REVISOR framework consists of three tasks: Short Video QA, Temporal Grounding, and Grounded Video QA, totaling 25K training samples. The parenthetical value for each dataset denotes its specific sample contribution.

#### 8. More Experimental Details on REVISOR

In Sec. 8.1, we present the detailed experimental setup for experiments involving the REVISOR framework. Sec. 8.2 outlines the composition of the training data for the REVISOR framework. Sec. 8.3 then presents supplementary experimental results, including comprehensive results on the Temporal Video Grounding task and complete ablation studies on the reward scaling factors λ1 and λ2.

##### 8.1. Detailed Experimental Setup

Our experiments utilized Qwen2.5-VL-7B as the base model, which comprises a visual encoder, a merger projec-

Table 5. Comprehensive evaluation of REVISOR on the temporal video grounding task. Bold text indicates the best performance.

Charades-STA NExT-GQA Model R@0.3 R@0.5 R@0.7 mIoU R@0.3 R@0.5 R@0.7 mIoU

Qwen2.5-VL-7B[2] 57.1 33.6 15.5 36.9 31.6 18.1 7.5 20.9 VTimeLLM[13] 55.3 34.3 14.7 34.6 37.9 20.2 9.7 24.4 iMOVE[18] 71.7 51.3 26.1 47.3 - - - TimeChat[33] 51.5 32.2 13.4 - 34.1 17.9 6.2 20.6 VideoChat-TPO[53] 58.3 40.2 18.4 38.2 41.2 23.4 8.2 27.7 TVG-R1[6] 70.8 50.5 23.9 46.7 41.7 20.8 10.0 29.3 Ours 76.5 57.3 31.8 51.4 47.6 25.5 11.9 33.2

Table 6. Complete ablation study of the Dual Attribution Decoupled Reward mechanism. Bold fonts highlight the best performance. The row marked in gray represents our base model.

###### λ1(Rfinal) λ2(Rcausal) VideoMME LongVideoBench LVBench MLVU NExT-GQA

- - 64.3 56.5 40.2 67.3 20.87

0.3 0.6 64.0 56.0 41.1 68.7 33.9 0.6 0.0 62.2 54.0 40.8 68.3 32.1

0.45 0.45 64.3 57.3 41.2 68.5 33.7

- 0.5 0.4 64.9 57.1 41.8 69.0 33.4

- 0.6 0.3 65.7 57.5 42.0 69.8 33.2 0.8 0.1 64.6 57.6 40.3 68.4 31.6

tor, and a large language model. REVISOR was trained using a single stage of reinforcement learning, eliminating the need for an additional supervised fine-tuning phase. We extended the verl framework to support REVISOR’s training. Following the approach in DAPO, we removed the KL regularization term from GRPO.

Key hyperparameters were set as follows: λ1 = 0.6, λ2 = 0.3, a learning rate of 1 × 10−6, a batch size of 32, and 8 rollouts. During both training and evaluation, input video tokens were limited to a maximum of 8192, sampled at 1 FPS. For the Review Segment, the sampling rate was increased to 2 FPS, while still adhering to a maximum of 8192 video tokens. Absolute timestamps were displayed in the lower-left corner of each image frame. The entire training phase consisted of 792 optimization steps.

##### 8.2. Detailed Information of Training Data

As detailed in Fig. 6, our training corpus spans three tasks central to video understanding: Short Video QA, Temporal Grounding, and Grounded Video QA. For Short Video QA, we aggregate data from LLaVA-Video-178K, CLEVRER, NExT-QA, PerceptionTest, and STAR, all of which provide short video clips (≤ a few minutes). From these datasets, we sample a balanced set of 20K video–question–answer triplets. The Temporal Grounding task is sourced directly from TimeRFT without further filtering. For Grounded Video QA, we incorporate CG-Bench and ReXTime, from

which we randomly select 2K examples.

##### 8.3. Supplementary Experimental Results

Temporal Video Grounding. As summarized in Tab. 5, we conduct an extensive assessment of REVISOR’s temporal grounding performance across both the Charades-STA and NExT-GQA benchmarks. On Charades-STA, REVISOR attains an R@0.3 accuracy of 76.5%, surpassing the prior state-of-the-art method iMOVE by a margin of 4.8%. On NExT-GQA, REVISOR reaches an R@0.5 accuracy of 25.5%, outperforming TVG-R1, designed explicitly with reinforcement learning to enhance temporal localization, by 4.3%. Together, these results highlight the robustness and effectiveness of REVISOR, equipped with DADR, in achieving precise temporal video localization.

Ablation Study of DADR Mechanism. We conducted an in-depth analysis of the Dual Attribution Decoupled Reward (DADR) mechanism. As shown in Tab. 6, when λ2 = 0, that is, when only the Final Answer Verification Reward is applied, the performance of REVISOR on Video-MME decreases substantially from 65.7% to 62.2%, even falling below the base model. This demonstrates that, without the Causal Segment Sufficiency Reward (CSSR), the model struggles to reliably identify the correct review segment S due to the sparsity of the reward signal.

When λ2 exceeds λ1, the model’s temporal grounding improves; however, its long-video reasoning capability de-

video segment for review: 100-170 seconds

full video: 297 seconds

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

User Question: What expression did the school of fish in the cage show when they saw the cartoon snake? (A) Surprised (B) Happy (C) Sad (D) Angry.

video segment for review: 100-170 seconds

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

preliminary reasoning trace multimodal reflective reasoning

<think> Oh, I see. The question is asking about the expression of the school of fish in the cage when they see the cartoon snake. Let me focus on the segment where the snake appears and the fish's reaction. The snake seems to be the main character that the fish are reacting to. I'll zoom in on the part where the snake is introduced and the fish's expressions change. </think> <time_interval> [100.0, 170.0] </time_interval>

<rethink> Let me analyze the video segment to determine the expression of the school of fish in the cage when they see the cartoon snake. The video shows the snake emerging and the fish reacting. The fish appear to be surprised and scared, as indicated by their wide eyes and open mouths. This aligns with option A, Surprised. The other options, Happy, Sad, and Angry, do not match the fish's reactions in the video. </rethink> <answer> A </answer>

- Figure 7. Successful example of the REVISOR framework: achieving more accurate detail capture. By reviewing key video segments, the REVISOR framework accurately identified that the fish in the cage exhibited surprise when they saw the cartoon snake. Without visual reflection, the model mistakenly believed the fish’s emotion was anger.

grades. For example, MLVU performance declines from 69.8% to 68.7%, suggesting that the model overemphasizes locating the review segment S while underutilizing it for answer derivation. Consequently, we set λ1 > λ2 to encourage accurate localization of S while still promoting strong reasoning based on it.

Nonetheless, if λ1 is too large and λ2 too small, the model’s ability to locate S deteriorates, harming long-form video understanding. Conversely, when λ1 is only slightly greater than λ2, REVISOR identifies S effectively but fails to fully leverage it for reasoning, resulting in strong temporal grounding but weaker long-video comprehension. Empirically, we find that λ1 = 0.6 and λ2 = 0.3 provide an effective balance, enabling REVISOR to both accurately localize S and utilize it to enhance reasoning performance.

#### 9. Training Qwen2.5-VL-7B with Textual Reflection Mechanism on Video Data

To ensure a fair comparison, we train a Qwen2.5-VL-7B model equipped with a text-based self-reflection mechanism using the datasets listed in Fig. 6. Specifically, unlike REVISOR, the text-reflection model generates only textual output during the reflection phase. Apart from this distinction, all other experimental settings remain identical to those of REVISOR. The learning rate is set to 1 × 10−6, the total batch size is 32, and the number of rollouts is set to 8. In addition, the total number of input video tokens is capped at 8192. The prompt template is shown in Fig. 11.

#### 10. Case Study of REVISOR Framework

Improved Video Reasoning Capability. The REVISOR framework can significantly enhance the long-form video

understanding capabilities of MLLMs. Fig. 7, Fig. 8, and Fig. 9 respectively demonstrate these improvements from three perspectives: more precise detail capture, more comprehensive scene understanding, and more accurate object counting.

Visual Reflection is Even More Important. In this paper, we repeatedly emphasize that, for long-video understanding tasks, visual reflection is more important than textual reflection. In Sec. 5.1, we quantitatively validate this conclusion by monitoring changes in the length of textual reflections during training. Here, we further illustrate this point with a concrete example. As shown in Fig. 10, during training, the textual reflection generated by the REVISOR framework for a given question becomes increasingly concise, while its retrieval of key video segments becomes increasingly accurate. Ultimately, the model precisely identifies the critical 25-second segment within a 300-second video, confirming that respiratory cells could potentially be used in treatment for cardiovascular disease. This phenomenon is not an isolated incident; similar observations were made across virtually all samples.

#### 11. Prompt Templates of REVISOR

The complete prompt template used during the training of the REVISOR framework consists of three primary components: the system prompt, the initial reasoning stage, and the reflective reasoning stage. Fig. 12 illustrates the templates for both the initial reasoning and reflective reasoning stages, while Fig. 13 presents the system prompt template. A key distinction from the plain-text reflection mechanism is that, during the initial reasoning stage, the REVISOR framework identifies the critical video segments that require further examination.

video segment for review: 119-174 seconds

full video: 259 seconds

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

User Question: Why does the man in black clothes stand on the edge of the cliff?

(A) He is meditating or contemplating on the edge of the cliff (B) To avoid being hit by a truck (C) He is admiring the view from the cliff (D) He is taking photos on the edge of the cliff

video segment for review: 119-174 seconds

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

preliminary reasoning trace multimodal reflective reasoning

<think> Oh, I see. The question is asking about a specific scene where a man in black clothes stands on the edge of a cliff. The video seems to show a sequence of events involving a truck, a motorcycle, and a dramatic cliff scene. The man in black clothes appears to be in a tense situation, possibly trying to avoid being hit by the truck. The other options don't seem to fit the context of the video. Let's zoom in on the segment where the man in black is on the cliff to understand his actions better. </think> <time_interval> [119.2, 174] </time_interval>

<rethink> Let's analyze the video segment to determine why the man in black clothes stands on the edge of the cliff. The video shows a sequence of events involving a truck, a motorcycle, and a dramatic cliff scene. The man in black clothes appears to be in a tense situation, possibly trying to avoid being hit by the truck. The other options don't seem to fit the context of the video. The man in black clothes is likely trying to avoid the truck, which is why he is standing on the edge of the cliff. Therefore, the correct answer is B. </rethink> <answer> B </answer>

- Figure 8. Successful example of the REVISOR framework: achieving more accurate scene understanding. By reviewing key video segments, the REVISOR framework accurately determined that the man was standing on the cliff edge to avoid an oncoming truck. Without this visual reflection, the model erroneously concluded that he was merely taking photos on the cliff.

<think> Oh, I see. The question is about identifying the number of individuals in a team dressed in yellow. Let's analyze the video to find the segment where this team is performing. The video shows various teams performing acrobatic routines, and we need to focus on the team dressed in yellow. The segment from 48.1 to 59.2 seconds seems to feature the team in yellow performing a routine. I'll use the temporal_zoom_in_tool to zoom in on this interval to count the number of individuals in the team. </think> <time_interval> [48.1, 59.2] </time_interval>

User Question: How many individuals are in the team, with each person dressed in yellow? (A) 4 (B) 3 (C) 5 (D) 6

full video: 91 seconds

video segment for review: 48-59 seconds

<rethink> Let's analyze the video segment from 48.1 to 59.2 seconds to determine the number of individuals in the team dressed in yellow. In this segment, we can see a group of performers in yellow outfits executing a routine. By closely examining the frames, it appears that there are 4 individuals in the team. Therefore, the answer is (A). </rethink> <answer> A </answer>

video segment for review: 48-59 seconds

[Figure 62]

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

[Figure 75]

[Figure 76]

[Figure 77]

preliminary reasoning trace multimodal reflective reasoning

- Figure 9. Successful example of the REVISOR framework: achieving more accurate object counting. By reviewing key video segments, the REVISOR framework accurately identified four team members wearing yellow. Without this visual reflection, the model erroneously counted only three.

Segment:151-175 seconds Segment:181-205 seconds Segment:238-260 seconds

full video: 326 seconds

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

User Question: For which diseases will respiratory cells potentially be used in treatment? (A) Respiratory diseases (B) Cancer treatment (C) Cardiovascular disease (D) Alzheimer's disease.

current training steps: 204

total training steps: 792

video segment for review: 151-175 seconds

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

preliminary reasoning trace multimodal reflective reasoning

<think> In order to answer the question accurately, I will need to find the video segment where respiratory cells are discussed in relation to various diseases. The video introduces the terms "RESPIROCYTE" and "BLOOD CELL," which are vital for transporting oxygen and carbon dioxide in the body. There seems to be some mention of various diseases that could potentially be treated using respiratory cells. A specific focus is placed on "CARDIOVASCULAR DISEASE," and while this is an important clue, the video also makes a passing reference to conditions like "ALZHEIMER'S DISEASE" and even cancer. These references need to be explored further to determine if respiratory cells have any role in these conditions. It’s also worth noting that there is mention of other medical terms, but I will focus on the sections specifically related to respiratory cells and their functions in disease treatment. </think> <time_interval> [150.5, 175.0] </time_interval>

<rethink> After analyzing the segment, I can confirm that the video provides some key insights into the potential therapeutic use of respiratory cells, particularly in relation to their roles in transporting gases such as oxygen and carbon dioxide. However, the segment also includes mentions of various diseases, which could make it difficult to pinpoint a single condition. While "CARDIOVASCULAR DISEASE" is indeed discussed, the video also brings up "CANCER TREATMENT" and even touches on "ALZHEIMER'S DISEASE" in a less direct way. These references add complexity, as they may imply that respiratory cells could have broader applications in medicine. Although the information about cardiovascular disease is fairly clear, the mention of cancer and Alzheimer's raises some questions about the scope of the treatment possibilities for respiratory cells. In conclusion, while cardiovascular disease is the most likely option, the broader context of the video leads me to consider that respiratory cells could have a potential role in other diseases as well. </rethink> <answer> B </answer>

current training steps:442

total training steps: 792

video segment for review: 181-205 seconds

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

preliminary reasoning trace multimodal reflective reasoning

<think> To accurately answer the question, I need to pinpoint the video segment where the discussion of respiratory cells as potential treatments occurs. The video provides information on "RESPIROCYTE" and "BLOOD CELL," focusing on their involvement in transporting oxygen and carbon dioxide. Although the discussion seems to touch on a variety of diseases, the term "CARDIOVASCULAR DISEASE" is mentioned in connection with respiratory cells. This implies that respiratory cells could potentially be used in treatments related to cardiovascular conditions. The video offers hints on how respiratory cells function, and this segment seems to have information relevant to treatment applications. I will analyze the segment more closely to extract this critical information. </think> <time_interval> [180.5, 205.0] </time_interval>

<rethink> Upon reviewing the video segment, it becomes clear that respiratory cells are connected to the treatment of various diseases, particularly focusing on their role in transporting gases such as oxygen and carbon dioxide. The term "RESPIROCYTE" is directly mentioned, suggesting their involvement in medical treatments. The video also brings up "CARDIOVASCULAR DISEASE," which indicates the possible use of respiratory cells in treatment for this condition. However, the discussion also briefly mentions "ALZHEIMER'S DISEASE," creating some ambiguity. While cardiovascular disease is clearly suggested, the Alzheimer's mention could confuse the interpretation. Given the context of the segment, it seems the primary disease under consideration here is cardiovascular disease, although other conditions are hinted at in passing. </rethink> <answer> C </answer>

current training steps:761

total training steps: 792

video segment for review: 238-260 seconds

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

preliminary reasoning trace multimodal reflective reasoning

<think>To answer the question, I need to identify the segment in the video where respiratory cells are discussed in relation to treatment. The video mentions "RESPIROCYTE" and "BLOOD CELL" and discusses their role in oxygen and carbon dioxide transport. It also briefly touches on "CARDIOVASCULAR DISEASE," suggesting a potential use of respiratory cells in treating this condition. Let's zoom in on the segment around the discussion of respiratory cells and their function in the blood to determine the relevant information. </think> <time_interval> [238.9, 260.2] </time_interval>

<rethink> Let's analyze the video segment focusing on the discussion of respiratory cells and their potential use in treating diseases. The video mentions "RESPIROCYTE" and "BLOOD CELL," highlighting their role in oxygen and carbon dioxide transport. It also explicitly states "CARDIOVASCULAR DISEASE," indicating that respiratory cells might be used in treating this condition. Therefore, the correct answer is related to cardiovascular disease. The segment from 238.9 to 260.2 seconds is crucial for understanding this information. </rethink> <answer> C </answer>

- Figure 10. As training progressed, REVISOR’s outputted reflective text paragraphs became increasingly concise, and the video segments it referenced grew more precise. Ultimately, it accurately concluded that respiratory cells could potentially be used in the treatment of cardiovascular disease.

###### The Complete Prompt Template for Plain-Text Reflection Mechanism System Prompt: You are a helpful assistant.

- Stage 1: Preliminary Reasoning Phase User Instruction: <User Question>. Please think carefully and then provide your answer. Output Format: Format strictly as: <think> Your reasoning steps </think> <answer> Your answer </answer>
- Stage 2: Reflective Reasoning Phase User Instruction: Please rethink the reasoning process above and provide your final answer. Output Format: Format strictly as: <rethink> Your rethinking reasoning steps </rethink> <answer> Your final answer </answer>

- Figure 11. The complete prompt template used during the training of Qwen2.5-VL-7B equipped with a pure text-based reflection mechanism, covering both the initial reasoning and reflective reasoning stages.

###### Reasoning Prompt Template for REVISOR Framework Training

- Stage 1: Preliminary Reasoning Phase User Instruction: <User Question>. Please think first, and then use the temporal zoom in tool to find the video segment that can answer the user’s question. Output Format: Format strictly as: <think> Your reasoning steps </think> <time interval>[start time, end time] </time interval>

- Stage 2: Reflective Reasoning Phase User Instruction: Please refer to the Visual Review segment above, think carefully, and provide your final answer. Output Format: Format strictly as: <rethink> Your reasoning steps </rethink> <answer> Your final answer </answer>

Figure 12. Reasoning prompt template for REVISOR framework training

System Prompt Template for REVISOR Framework Training You are a helpful assistant. # Tools You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags: <tools>

- 1 {

- 2 "type": "function",

- 3 "function": {

- 4 "name": "temporal_zoom_in_tool",

- 5 "description": "Identify the precise time segment in the video that contains enough information to answer the question.",

- 6 "parameters": {

- 7 "properties": {

- 8 "interval": {

- 9 "type": "str",

- 10 "description": "The time range to zoom in on, formatted as ’start_time to end_time’. Timestamps are in seconds."

- 11 }

- 12 },

- 13 "required": ["interval"]

- 14 }

- 15 }

- 16 } </tools>

###### # Explanation of ”temporal zoom-in”

When you call temporal zoom in tool with <time interval> [start time, end time] </time interval>, the tool returns a new sequence of denser video frames sampled from the specified time range ([start time, end time]) in the original video. This provides higher temporal precision, helping you answer the user’s question more accurately.

# How to call a tool Return the interval directly in XML tags: <time interval> [12.3, 28.7] </time interval>

Figure 13. System prompt template for training the REVISOR framework.

