# arXiv:2510.20470v2[cs.CV]20Nov2025

[Figure 1]

## Conan: Progressive Learning to Reason Like a Detective over Multi-Scale Visual Evidence

Kun Ouyang1, Yuanxin Liu1, Linli Yao1, Yishuo Cai1, Hao Zhou2, Jie Zhou2, Fandong Meng2, Xu Sun1* 1State Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University 2WeChat AI, Tencent Inc., China

kunouyang10@gmail.com, xusun@pku.edu.cn

[Figure 2]

- A. The Black Sabbath figure sits down...
- B. The Black Sabbath figure holds a weapon...
- C. The Black Sabbath figure holds a small glass while facing the costumed figure.
- D. The Black Sabbath figure is tying the green cape...
- E. The Black Sabbath figure kneels down...

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

What distinguishes the interaction between the figure in the Black Sabbath shirt and the elaborately costumed figure?

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Frame identification

[Figure 20]

Distractor frames Context frames Evidence frames

[Figure 21]

Evidence Reasoning

The question asks for the distinctive interaction between the figure wearing a Black Sabbath T-shirt and the elaborately costumed figure (green cape, gold detail)...00:08:10: The Black T-shirt figure stands facing the costumed figure and is holding a small glass in the right hand. 00:08:45: ...Motivation for extra frames...Retrieve the clip segment 00:08:15–00:08:55 to verify and solidify the observed interaction.

[Figure 22]

[Figure 23]

Random Frames Sampling

Confident Question Answering

Action Decision 00:08:15–00:08:55

[Figure 24]

[Figure 25]

###### Action Candidates

Specific Frames Retrieval

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Frame identification

[Figure 36]

Context frames Evidence frames Context frames

[Figure 37]

[Figure 38]

Action Decision

[Figure 39]

[Figure 40]

Evidence Reasoning

...The only action that persists and is explicitly shown is the figure holding a small glass while facing the elaborately costumed figure. Answer the question with C.

C

Qwen2.5-VL-7B-Instruct GPT-4o Video-R1 Video-MTR

Conan

54.0

44.6

54

50

45

85

55

75

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

51.0

81.0

72.8

42.0

[Figure 45]

[Figure 46]

52.3

43

70.3

[Figure 47]

[Figure 48]

52

80

[Figure 49]

[Figure 50]

44.0

53

76.7

[Figure 51]

[Figure 52]

45

41

70

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

42.7

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

50

75

[Figure 61]

[Figure 62]

39

51

48.0 48.1

36.5 35.7

69.5 69.7

49.8

40

37

65

[Figure 63]

[Figure 64]

48

70

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

61.8 63.1

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

66.4

46.5

36.3 36.5

[Figure 75]

[Figure 76]

48.4

[Figure 77]

[Figure 78]

35

49

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

48.2

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

46

47.2

[Figure 91]

[Figure 92]

65

35

33

60

57.3

[Figure 93]

[Figure 94]

47

31

[Figure 95]

[Figure 96]

44

60

30.1

28.5

30

29

55

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

45

42

55

27

25

25

50

40

43

50

MMR-V

Video-Holmes

VRBench

VCRBench

Human-P&C

LongVideoReason

Figure 1. Top: The reasoning process of our Conan on an example question. Bottom: Performance comparison across six multi-step reasoning benchmarks.

### Abstract

Video reasoning, which requires multi-step deduction across frames, remains a major challenge for multimodal large language models (MLLMs). While reinforcement learning (RL)-based methods

*Corresponding Author

enhance reasoning capabilities, they often rely on text-only chains that yield ungrounded or hallucinated conclusions. Conversely, frame-retrieval approaches introduce visual grounding, yet still struggle with inaccurate evidence localization. To address these limitations, we present Conan, a framework for evidence-grounded multi-step video rea-

soning. Conan identifies context and evidence frames, reasons over cross-frame clues, and adaptively decides when to conclude or explore further. To achieve this, we 1) construct Conan-91K, a large-scale dataset of automatically generated reasoning traces that include frame identification, evidence reasoning, and action decision, and 2) design a multi-stage progressive cold-start strategy combined with an Identification-Reasoning-Action (AIR) RLVR training framework to progressively incentivize multi-step visual reasoning. Extensive experiments on six multi-step reasoning benchmarks demonstrate that Conan surpasses the baseline Qwen2.5-VL-7B-Instruct by an average of over 10% in accuracy, achieving state-of-the-art performance. Furthermore, Conan generalizes effectively to long video understanding tasks, validating its strong scalability and robustness.

[Figure 101]

Model: huggingface.co/RUBBISHLIKE/Conan-7B

Dataset: huggingface.co/RUBBISHLIKE/Conan-91k

Code: github.com/OuyangKun10/Conan

### 1. Introduction

There is only one truth!

– Edogawa Conan

Frontier multimodal large language models (MLLMs) [2, 12, 24, 28] have demonstrated remarkable progress on standard video understanding tasks such as question answering [1], temporal grounding [9], and captioning [25]. However, video reasoning [6, 20] remains a substantial challenge. Unlike conventional tasks, video reasoning [3, 6, 22, 32, 36] demands active visual information accumulation across temporal spans and multi-step logical inference to reach well-grounded conclusions.

Inspired by the success of reinforcement learning with verifiable rewards [10] (RLVR) in incentivizing reasoning ability of LLMs, recent works [7, 15, 21] have begun extending this paradigm to video reasoning, achieving promising gains. Nevertheless, these approaches primarily rely on puretext reasoning without explicit grounding in visual

evidence, often leading to superficial or hallucinated reasoning chains that fail to reflect the actual video content. To integrate visual evidence into the reasoning process, concurrent works [11, 29, 34] introduce frame retrieval mechanism to enable video chain-of-thought (Video-CoT) reasoning, boosting performance of long video understanding [27, 35]. However, these approaches usually suffer from inaccurate or implicit evidence localization, yielding unreliable reasoning paths. Additionally, some methods [11, 34] partially rely on benchmark-specific training data (e.g. VideoHolmes [6] and LongVideoReason [3]), making it difficult to disentangle solid reasoning improvements from in-domain overfitting.

Motivated by these limitations, we aim to equip MLLMs with multi-step, evidence-grounded video reasoning skills, analogous to how Conan acts as

- a detective (Figure 1): Specifically, our framework identifies relevant frames at multiple scales (context and evidence frames), reasons over crossframe clues to form coherent chains of deduction, and decides whether to draw the final conclusion or continue exploring the video. Achieving this goal raises two core challenges: 1) How to automatically construct a high-quality evidence-based reasoning dataset that explicitly captures evidence localization, multi-step deductive reasoning, and confident action decision. 2) How to design training curriculum for effective acquisition of visual reasoning across multi-scale evidence.

To tackle the first challenge, we introduce Conan-91k, a large-scale dataset for Conan-style evidence reasoning. Built upon the key-frame identification dataset GenS-Video-150K [31], we develop an automated pipeline to generate interleaved video-text reasoning traces using the advanced LLM Kimi K2 [23], as illustrated in Figure 2. Each reasoning trace contains three key components: a) Frame Identification distinguishes among evidence, context, and distractor frames.

- b) Evidence Reasoning conducts textual reasoning over the question and accumulated visual clues.
- c) Action Decision decides whether to retrieve additional frames or reach the final conclusion. Furthermore, we propose an evidence difficulty-aware

###### 2. Evidence Reasoning

1. Frame Identification

###### a) Reasoning Trace Construction

<think>...

Kimi K2

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

QA pair

Identified frame types

Dense frame descriptions

Interleaved timestamps

Raw video

Sampled frames

[Figure 113]

No evidence Random Frames Sampling

Partial evidence Specific Frames Retrieval

[Figure 114]

[Figure 115]

Sufficient evidence Confident Question Answering

3. Action Decision

Loop End

b) Data Example

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

...

Round 3 Frame Identification: Evidence frames

Round 1 Frame Identification: Distractor frames

Evidence Reasoning: ...A is the only description matching the visible external change: the people and another plane appear in plain sight through the cockpit glass during the pilot’s tablet-based attention window.Answer: A. Action Decision: Confident Question Answering

Evidence Reasoning: ...What must be seen: The question asks for what outside-the-cockpit visual element we first notice shifting focus precisely while the pilot is interacting with the tablet... Action Decision: Random Frames Sampling

Multimodal Alignment Reasoning Stage

Vision-Centric Reasoning Stage

Textual Reasoning Stage

[Figure 128]

- c) Multi-stage Progressive Cold-start
- d) Joint IdentificationReasoning-Action RLVR

Conan-CoT-60k

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Reference Model

- r1
- r2
- r3

- y1
- y2
- y3

GRPO Optimization

... rG

... yG

Rewards Computation

Conan-RLVR-31k

Figure 2. a) Reasoning Trace Construction. b) Data Example. c) Multi-stage Progressive Cold-start, including textual, multimodal alignment, and vision-centric reasoning stages. d) The Joint Identification-Reasoning-Action RLVR.

sampling strategy to facilitate progressive training from simple to complex reasoning, producing 60k samples (Conan-CoT-60k) for SFT and 31k samples (Conan-RLVR-31k) for RL.

To address the second challenge, we introduce a two-phase training curriculum: a) A multi-stage progressive cold-start strategy incrementally activates the model’s multi-step reasoning using Conan-CoT-60k, starting from textual reasoning, advancing to multimodal alignment, and culminating in vision-centric deduction. b) On this foundation, the joint Identification-ReasoningAction (AIR) RLVR framework further refines the model’s capacity to perform Conan-style reasoning on Conan-RLVR-31k by jointly optimizing frame identification, evidence reasoning, and action decision. Together, these components empower Conan to “seek, deduce, and act” across visual clues, achieving reliable multi-step reasoning.

Extensive experiments on six challenging multi-step reasoning benchmarks (e.g. MMRV [36], Video-Holmes [6], VRBench [32], VCRBench [22], Human-P&C [14], and LongVideoReason [3]) demonstrate that Conan consis-

tently surpasses the base model Qwen2.5-VL-7BInstruct [2], with an average accuracy improvement exceeding 10%. Moreover, Conan achieves promising enhancement on long video understanding tasks (e.g. LongVideoBench [27], MLVU [35], LVBench [26], and Video-MME [8]), validating strong scalability and robustness.

In summary, our contributions are threefold: ♠ We introduce Conan-91k, the first large-scale dataset for multi-scale evidence reasoning, including Conan-CoT-60k and Conan-RLVR-31k. ♥ We propose a multi-stage progressive cold-start strategy and a joint AIR RLVR framework to foster gradual acquisition of evidence-based multi-step reasoning skills. ♣ We conduct extensive experiments, demonstrating that Conan achieves substantial improvements in both multi-step reasoning and long video understanding.

### 2. Dataset Construction

#### 2.1. Data Collection & Processing

We collect source data from the GenS-Video150K [31] dataset, which provides dense frame de-

scriptions, multi-choice and free-form QA pairs, as well as frame-level relevance scores. Leveraging these relevance scores, we categorize video frames into three types: 1) Evidence frames, which are directly relevant to answering the question; 2) Context frames, which offer auxiliary hints that may support the reasoning process; and 3) Distractor frames, which bear no relation to the question. This multi-scale frame categorization establishes the foundation for subsequent stepwise reasoning trace construction.

#### 2.2. Reasoning Trace Construction

Starting from the processed data, we apply an automatic pipeline to construct Conan-style video-text interleaved reasoning traces with the assistance of a strong LLM, Kimi K2 [23], as illustrated in Figure 2. The core loop proceeds as follows:

- • We first sample 16 frames uniformly from the raw video and retain each frame’s type label (evidence, context, or distractor).
- • If all frames are distractors, we select the action Random Frames Sampling, which randomly samples 8 new frames and continue the loop.
- • If some sampled frames are evidence or context but the evidence proportion does not exceed a dynamic answering threshold at which K2 can reach the answer based on the acquired clues, we select the action Specific Frames Retrieval, which uniformly retrieves 8 frames within a single clip or multiple clips that contain evidence/context, and continue the loop.
- • If the proportion of evidence frames exceeds the threshold, we terminate the loop by selecting Confident Question Answering as the final action.

For every loop iteration, we prompt K2 with the chosen action, the frame types, the QA pair, dense frame descriptions, and timestamps to generate a coherent textual reasoning trace that a) analyzes the QA and sampled frames, and b) justifies the chosen action. More details are in Appendix B.

#### 2.3. Evidence Difficulty-Aware Sampling

To facilitate a progressive training curriculum from simple to complex reasoning cases, we propose an evidence difficulty-aware sampling strategy. In

particular, we define an Evidence Difficulty Index (EDI) to quantify the reasoning complexity based on the proportion and temporal dispersion of evidence frames. Let the evidence ratio be P = m/N, where m and N denote the numbers of evidence and total frames, respectively. The temporal variance of evidence Var = m1 mi=1 (xi − x¯)2, where xi represents the temporal position of i-th evidence frame, and x¯ is the mean position of all evidence frames. The overall difficulty is defined as EDI = (1 − P)Var, where higher EDI value indicates sparser and more temporally dispersed evidence, reflecting greater difficulty.

Based on the EDI distribution and reasoning round, samples are stratified and adaptively allocated across training phases. During the SFT stage, we focus on lower-EDI samples to emphasize foundational reasoning skills, selecting 60k instances with up to three reasoning rounds: 25k oneround, 25k two-round, and 10k three-round samples that align with the overall round distribution. In contrast, the RLVR stage employs higher-EDI samples without constraining reasoning rounds, selecting 31k instances spanning diverse reasoning rounds. This difficulty-aware sampling scheme establishes a principled curriculum that transitions smoothly from low-difficulty grounding in SFT to high-difficulty multi-hop reasoning in RLVR, facilitating the gradual and robust acquisition of evidence-based reasoning capabilities. The resulting datasets, Conan-CoT-60k and Conan-RLVR31k, together form the Conan-91k dataset. Dataset statistics and additional construction details are provided in Appendix B.

### 3. Two-Phase Training Curriculum

#### 3.1. Multi-Stage Progressive Cold-Start

To progressively activate the multi-step reasoning abilities, we conduct a multi-stage progressive cold-start on Conan-CoT-60k, guided by a stagewise incremental sampling that gradually expands data diversity and reasoning difficulty: 1) Textual Reasoning Stage. In the initial stage, the model is trained on 10k relatively low-EDI samples from one-round subset, where frames are rep-

resented by dense textual descriptions and timestamps. This stage focuses on temporal and causal reasoning across ordered frame descriptions, establishing a structured reasoning foundation for subsequent multimodal learning. 2) Multimodal Alignment Reasoning Stage. Compared with the first stage, we utilize 25k one-round samples (including 15k new ones) and a new set of 10k relatively low-EDI samples from two-round subset, inserting timestamps and textual descriptions before visual frames. This incremental expansion ensures that the model continues learning on partially new data, preventing overfitting to previously seen questions while promoting stable adaptation from textual to multimodal reasoning. The tworound samples allow the model to collect more evidence for reasoning. 3) Vision-Centric Reasoning Stage. In the final stage, the model is trained on the complete Conan-CoT-60k, including additional 15k two-round and 10k three-round samples. This stage compels the model to execute deep multi-step reasoning directly over visual frames with interleaved timestamps, thereby enhancing perceptual grounding and fostering vision-centric reasoning.

#### 3.2. AIR RLVR

Building upon the model Conan-SFT obtained from the previous cold-start process, we further refine its multi-step reasoning capabilities via the Identification-Reasoning-Action (AIR) RLVR framework. Given that the model has already learned to produce reasoning traces consisting of: 1) frame identification, 2) evidence reasoning, and 3) action decision, AIR RLVR aims to optimize the exploration of effective reasoning trajectories through a set of carefully designed reward functions. We first introduce one format reward and two outcome rewards to ensure both structural consistency and answer accuracy.

Format Reward. To enforce structural consistency in model outputs y, we define a format reward Rfmt that verifies whether specific tags are correctly applied. And the model is restricted to performing only one action (e.g. random frames sampling, specific frames retrieval, confident question answering) at a time. The format reward Rfmt is

defined as:

Rfmt(y) =

0.5, if y matches format, 0, otherwise.

(1)

Multi-choice Reward. For multi-choice QA, the outcome reward Rmc is determined by exact match between the predicted answer y and ground truth yˆ:

Rmc(y,yˆ) =

1, if y = y,ˆ 0, otherwise.

(2)

Free-form Reward. For free-form QA, the outcome reward Rfree is computed as the average of ROUGE-1, ROUGE-2, and ROUGE-L scores [17] between the predicted answer y and the ground truth yˆ:

1 3

R1(y,yˆ)+R2(y,yˆ)+RL(y,yˆ)

Rfree(y,yˆ) =

(3)

To evaluate the quality of the multi-scale frames identification, and frames retrieval actions, we design an identification reward Ride and a retrieval reward Rret, respectively: 1) The identification reward Ride measures the average accuracy of identified evidence/context frames across reasoning rounds. 2) The retrieval reward Rret evaluates the quality of retrieved frames by computing the average ratio of evidence/context frames among all retrieved frames. And the final joint identificationretrieval-outcome reward RJ is formulated as:

Rfmt + Ro + Ride + Rret, if Ro > 0, Rfmt + Ro, otherwise,

RJ =

(4) where Ro is the outcome reward and o ∈ {mc,free}. This reward shaping encourages the model to generate structurally valid, evidencegrounded, and accurate reasoning traces while improving retrieval efficiency. Finally, we prompt the model generate a group of responses {y1,y2,··· ,yG}, where G is the number of generated responses, and we adopt the GRPO [10] algorithm for reinforcement optimization to stabilize training and refine the reasoning policy.

###### #Params MMR-V Video-Holmes VRBench VCRBench∗ Human-P&C LVR Avg. VR Avg. LU

Closed-source Models GPT-4o [12] - 44.0 42.0 76.7 54.0 48.4 63.1 54.7 -

- Gemini 1.5 Pro - - 41.2 - - 52.6 69.3 - -

- Gemini 2.0 Flash - 42.6 30.6 - - 56.1 - - -

Open-source Models LLaVA-OneVision-7B [13] 7B 6.5 - - 30.7 48.5 - - InternVL3-8B [4] 8B 35.5 32.8 75.8 45.7 51.0 68.0 51.5 52.5 Kimi-VL-A3B-Instruct [24] 3B/16B 32.4 32.4 60.5 34.3 42.0 64.6 44.4 Qwen2.5-VL-72B-Instruct [2] 72B 39.1 40.2 72.7 50.8 55.7 72.3 55.1 53.4 Qwen2.5-VL-7B-Instruct [2] 7B 30.1 28.5 66.4 46.5 48.2 61.8 46.9 48.0 Text CoT Models

Video-R1 [7] 7B 36.3 36.5 69.5 48.0 49.8 70.3 51.7 53.8 VideoChat-R1 [15] 7B 36.1 33.0 61.5 48.2 51.8 67.9 49.8 52.4 Video CoT Models

Video-MTR [29] Concurrent work 7B 36.5 35.7 69.7 48.1 47.2 57.3 49.1 53.5 Rewatch-R1† [34] Concurrent work 7B 45.3 37.8 79.1 49.8 51.6 70.5 55.7 50.5

Conan SFT 7B 35.4 34.9 64.4 43.3 50.4 66.0 49.1 49.3 Conan 7B 42.7 (↑ 12.6 ) 44.6 (↑ 16.1 ) 81.0 (↑ 14.6 ) 51.0 (↑ 4.5 ) 52.3 (↑ 4.1 ) 72.8 (↑ 11.0 ) 57.4 (↑ 10.5 ) 54.9 (↑ 6.9 )

Table 1. Main results of Qwen2.5-VL-7B-Instruct , Conan , and other baselines. Avg. VR and Avg. LU denote the average results on six video reasoning benchmarks and four long video understanding benchmarks, respectively. † indicates partial training on the training set of LongVideoReason (LVR) [3], and ∗ marks the multiple-choice subset.

### 4. Experiment

#### 4.1. Evaluation Setups

Implementation Details 1) Training. We adopt Qwen-2.5-VL-7B-Instruct [2] as the base model. During the cold start, the model is trained for up to one epoch per stage with a global batch size of 32 and a learning rate of 1e-5. The trained model is then used for the AIR RLVR phase, also trained for one epoch under the batch size of 24 and the learning rate of 1e-6. The maximum completion length is set to 4,000 tokens, with a generation temperature of 1.0 and a generation number G of 8. Each input video contains 16 initial frames, and the model can retrieve up to 8 additional frames per reasoning step. 2) Evaluation. The generation temperature is fixed at 1.0, and each sample is evaluated three times. The maximum number of new tokens is set to 4,000 when reasoning traces are included and 128 for direct answering. The reasoning process is limited to three rounds. Videos are standardized to 16 frames for multi-step reasoning and 32 frames for long video understanding during evaluation, with a resolution of 448 × 28 × 28.

Benchmarks & Baselines. We conduct comprehensive evaluations on six multi-step reasoning benchmarks(i.e., MMR-V [36], Video-Holmes [6], VRBench [32], VCRBench [22] (multi-choice subset), Human-P&C [14], and LongVideoRea-

son [3]) and four long video understanding benchmarks (i.e., LongVideoBench [27], MLVU [35], LVBench [26], and Video-MME [8]), comparing Conan with a range of closed-source and opensource MLLMs. Descriptions of benchmarks and baselines are provided in Appendix C.1.

#### 4.2. Main Results

The main evaluation results are shown in Table 1, while the full results of long video understanding are in Table 3 of Appendix. The key observations are summarized as follows.

###### No evidence retriever

Qwen2.5-VL7B-Instruct Video-MTR Conan

Rewatch-R1

- 65
- 66
- 67
- 68
- 69
- 70
- 71
- 72
- 73

- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35

39

###### 34.0

###### 37.8

###### 71.7

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

37

[Figure 139]

[Figure 140]

32.5

35.7

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

34.1

35

30.7

33.5

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

33

[Figure 149]

[Figure 150]

68.0 68.3

28.5 29.1

30.1

31

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

67.2

29

[Figure 161]

[Figure 162]

66.4

[Figure 163]

[Figure 164]

27

25

Video-Holmes

###### VRBench

MMR-V

- 60
- 61
- 62
- 63
- 64
- 65
- 66
- 67
- 68

- 44
- 45
- 46
- 47
- 48
- 49
- 50

- 45
- 46
- 47
- 48
- 49
- 50
- 51
- 52

###### 66.7

###### 49.0

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

###### 49.7

48.2

65.1

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

47.6

[Figure 173]

[Figure 174]

48.9 48.6

[Figure 175]

[Figure 176]

47.1

[Figure 177]

[Figure 178]

48.4

48.2

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

62.7 63.5

46.5

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

61.8

[Figure 193]

[Figure 194]

LongVideoReason

VCRBench

Human-P&C

Figure 3. Performance comparison across different evidence retrievers, with Qwen2.5-VL-7B-Instruct as the question-answering model.

Overall Analysis. Across video reasoning bench-

MMR-V Video-Holmes VRBench VCRBench⋆ Human-P&C LVR Overall Dataset design

w/o multi-scale evidence 39.7 39.1 75.4 46.9 51.3 70.2 53.8 w/o difficulty sampling 40.0 40.8 78.9 48.4 50.7 72.3 55.2

Progressive cold-start

w/o textual reasoning 42.2 43.9 81.4 49.8 51.7 73.1 57.0 w/o multimodal alignment reasoning 43.5 44.2 80.4 48.2 49.3 72.5 56.4 w/o vision-centric reasoning 39.0 36.5 75.2 50.2 48.7 68.5 53.0 w/o cold-start 39.1 36.8 73.3 46.3 47.9 62.3 51.0

AIR RLVR

w/o identification reward 39.9 40.0 74.8 46.1 50.1 71.7 53.8 w/o retrieval reward 38.2 42.4 75.7 47.8 50.5 69.2 54.0 w/ text CoT 41.1 41.0 76.8 48.8 52.8 70.5 55.2

Conan 42.7 44.6 81.0 51.0 52.3 72.8 57.4

Table 2. Ablation results of Conan, where the best results are in boldface.

marks, Conan substantially surpasses its base model Qwen2.5-VL-7B-Instruct with an average accuracy gain of over 10%. Remarkably, Conan also outperforms the advanced GPT-4o on most benchmarks (e.g., Video-Holmes, VRBench, Human-P&C, and LongVideoReason), underscoring its superior capabilities of multi-step, evidencegrounded reasoning. And the two text-CoT models (e.g., Video-R1 and VideoChat-R1) perform notably worse than Conan, demonstrating the advantages of visual evidence identification and retrieval. Moreover, Conan consistently outperforms videoCoT approaches (e.g., Video-MTR and RewatchR1), showcasing more precise evidence localization, and more effective frame retrieval of Conan. Notably, Conan also generalizes well to long video understanding tasks, with an average improvement of 6.9%, proving the robustness and scalability of our framework.

Conan as an Evidence Retriever. To further assess Conan’s evidence localization capability, we compare different models used as evidence retriever, each selecting up to 16 key frames per video for Qwen2.5-VL-7B-Instruct to perform question answering. As shown in Figure 3, Conan yields the highest performance, highlighting its superior ability to identify informative visual evidence. Additional results under varying frame settings and base models are provided in Appendix C.2. And training dynamics can be found in Appendix C.3.

#### 4.3. Ablation Study

To investigate the contribution of each component in our framework, we conduct a comprehensive ablation study with multiple Conan variants.

For the dataset design, we introduce two variants: 1) w/o multi-scale evidence, which merges the context type into the distractor category in multiscale frame categorization; and 2) w/o difficulty sampling, which replaces the evidence difficultyaware sampling with random sampling.

For the multi-stage progressive cold-start strategy, we develop four variants: 1) w/o textual reasoning, which excludes the textual reasoning stage; 2) w/o multimodal alignment reasoning, which skips the multimodal alignment reasoning stage; and 3) w/o vision-centric reasoning, which removes the vision-centric reasoning stage. 4) w/o cold-start, which bypasses the three cold-start stages and directly employs AIR RLVR to train the model on the Conan-RLVR-31k dataset.

For the AIR RLVR framework, we design three variants: 1) w/o identification reward, which discards the identification reward; 2) w/o retrieval reward, which removes the retrieval reward; and 3) w/ text CoT, which enforces single-round text-CoT paradigm performing pure textual reasoning and answering without additional retrieval in training and evaluation procedures.

The ablation results in Table 2 reveal several key findings: a) Dataset design. w/o multi-scale evi-

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

The video shows characters in the Minecraft game choosing to wear different colored shirts, what effect might this have in the game?

8 frames Irrelevant frames

[Figure 212]

Frame Retrieval

[Figure 213]

Video CoT Reasoning

- A. Different colored shirts may unlock hidden crafting recipes or resource nodes.
- B. Different colored shirts may trigger different game events or interactions with other characters.
- C. Different colored shirts may change the time of day or weather conditions in the game.
- D. Different colored shirts may alter the player's movement speed and jumping ability.

[Figure 214]

[Figure 215]

Video-MTR

###### A

Text CoT ReasoningAfter reviewing: Let methethinkoptionsaboutcarefully,this forthea moment.most plausibleThe questionexplanationis askingis thataboutdifferentthe effectscoloredof charactersshirts maywearingunlock hiddendifferentcraftingcoloredrecipesshirtsorinresourcethe Minecraftnodesgame....

[Figure 216]

Inacurate perception

Video-R1

Answer: A

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Round 1

Frame Identification: Distractor frames

[Figure 233]

Evidence Reasoning: ...We need to identify how wearing differently colored shirts affects gameplay, specifically in the context of the Minecraft mod “Friendycraft.” The key is to find evidence that the shirt color triggers any in-game effect...Since the frames do not contain the required evidence, we must randomly retrieve more frames to verify the shirt-color-triggered events. Action Decision: Random Frames Sampling

Evidence Reasoning: In the previous round we lacked direct evidence of shirt-color effects...B: The only plausible effect is that different shirt colors trigger different game events or interactions...Since the evidence is not sufficient, we must retrieve the surrounding clips...verify the shirt-color-triggered events. Action Decision: Specific Frames Retrieval 00:15:50-00:16:05, 00:31:30-00:31:45...

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Frame Identification: Context frames Distractor frames

- Round 2 Evidence Reasoning: ...The frames describe various game events and interactions (flag capture, team lists, etc.) that are directly tied to the player’s shirt color...Conclusion: The evidence consistently shows that different colored shirts trigger different game events and interactions with other characters. Therefore, the correct answer is B.

- Round 3 Action Decision: Confident Question Answering B

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Context frames Frame Identification: Evidence frames

Figure 4. A qualitative example from VRBench, including the QA pair and the reasoning traces of Video-R1 (Text CoT), Video-MTR (Video CoT), and Conan.

dence underperforms Conan, confirming the benefits of multi-scale frame identification in providing richer contextual cues. Moreover, Conan outperforms w/o difficulty sampling, which validates that evidence difficulty-aware sampling effectively guides the model to progressively acquire multi-step reasoning abilities. b) Progressive coldstart. Conan consistently surpasses w/o textual reasoning, w/o multimodal alignment reasoning, w/o vision-centric reasoning, and w/o cold start, which demonstrates that the multi-stage progressive cold-start is crucial for gradually activating the model’s multi-hop reasoning capabilities. c) AIR RLVR. Conan outperforms w/o identification reward and w/o retrieval reward, proving their effectiveness in refining evidence localization accuracy and frame retrieval efficiency, respectively. And w/ text CoT underperforms Conan, which validates the superiority of the Conan-style reasoning paradigm.

#### 4.4. Qualitative Evaluation

Figure 4 presents a qualitative comparison on VRBench [32] among Video-R1, Video-MTR, and Conan. Video-R1 conducts pure textual reasoning

without visual grounding, resulting in a hallucinated answer driven by linguistic priors. VideoMTR incorporates frame retrieval but fails to localize relevant evidence, leading to weak alignment between retrieved frames and the question. In contrast, Conan performs multi-step reasoning with accurate evidence grounding and efficient frame retrieval. In Round 1, it recognizes the absence of causal cues and performs random frames sampling to expand the search space. In Round 2, guided by contextual signals, it retrieves frames around key timestamps of player interactions. In Round 3, Conan identifies visual evidence of color-triggered game events (e.g., team activities, flag captures) and integrates these clues to infer the correct option. This comparison highlights Conan’s superior ability to localize, reason over, and act upon multiscale visual evidence. More qualitative cases and error analyses are in Appendix C.4.

### 5. Conclusion and Future work

In this work, we present Conan, a novel framework that equips MLLMs with Conan-like visual reasoning through frame identification, evidence

reasoning, and action decision. Employing the Conan-91k dataset, constructed via multi-scale frame categorization, reasoning trace construction, and evidence difficulty-aware sampling, we devise a multi-stage progressive cold-start strategy alongside a joint Identification-Reasoning-Action (AIR) RLVR framework to progressively cultivate robust multi-step reasoning abilities. Extensive experiments across six multi-step reasoning and four long video understanding benchmarks demonstrate that Conan consistently outperforms the base model Qwen2.5-VL-7B-Instruct, achieving state-of-theart performance over both text-CoT and video-CoT models. In future work, we plan to extend Conan toward chain-of-frame reasoning, enabling dynamic frame generation during reasoning to provide visual evidence beyond the video frames for tackling more complex video reasoning tasks.

### References

- [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433,

2015. 2, 1

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3, 6, 1, 4, 5
- [3] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu, Hongxu Yin, Yao Lu, and Song Han. Scaling rl to long videos. In Advances in Neural Information Processing Systems, 2025. 2, 3, 6
- [4] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 6, 3, 4, 5
- [5] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for

- generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 3
- [6] Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. Video-holmes: Can mllm think like holmes for complex video reasoning? arXiv preprint arXiv:2505.21374, 2025. 2, 3, 6, 1
- [7] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 2, 6, 1, 3, 4

- [8] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 3, 6, 4
- [9] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 2, 1
- [10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 5, 1
- [11] Zefeng He, Xiaoye Qu, Yafu Li, Siyuan Huang, Daizong Liu, and Yu Cheng. Framethinker: Learning to think with long videos via multi-turn frame spotlighting. arXiv preprint arXiv:2509.24304,

2025. 2, 1, 3, 4

- [12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2, 6, 1, 3
- [13] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6, 3
- [14] Keliang Li, Hongze Shen, Hao Shi, Ruibing Hou, Hong Chang, Jie Huang, Chenghao Jia, Wen Wang, Yiling Wu, Dongmei Jiang, Shiguang Shan, and Xilin Chen. Humanpcr: Probing mllm capabili-

- ties in diverse human-centric scenes. arXiv preprint arXiv:2508.13692, 2025. 3, 6
- [15] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement finetuning. arXiv preprint arXiv:2504.06958, 2025. 2, 6, 1, 3, 4
- [16] Yunxin Li, Xinyu Chen, Baotian Hu, Longyue Wang, Haoyuan Shi, and Min Zhang. Videovista: A versatile benchmark for video understanding and reasoning. arXiv preprint arXiv:2406.11303, 2024. 1
- [17] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 5
- [18] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982,

2025. 3

- [19] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics, pages 8731–8772,

2024. 1

- [20] Yuanxin Liu, Kun Ouyang, Haoning Wu, Yi Liu, Lin Sui, Xinhao Li, Yan Zhong, Y Charles, Xinyu Zhou, and Xu Sun. Videoreasonbench: Can mllms perform vision-centric complex video reasoning? arXiv preprint arXiv:2505.23359, 2025. 2, 1
- [21] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025. 2, 1
- [22] Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, and Feng Zhao. Vcrbench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint arXiv:2504.07956, 2025. 2, 3, 6, 1
- [23] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. 2, 4
- [24] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl

technical report. arXiv preprint arXiv:2504.07491,

2025. 2, 6, 1, 3

- [25] Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, and Kate Saenko. Translating videos to natural language using deep recurrent neural networks. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1494–1504, 2015. 2, 1
- [26] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024. 3, 6, 1, 4
- [27] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for longcontext interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 2, 3, 6, 1, 4
- [28] LLM-Core-Team Xiaomi. Mimo-vl technical report. arXiv preprint arXiv:2506.03569, 2025. 2, 1
- [29] Yuan Xie, Tianshui Chen, Zheng Ge, and Lionel Ni. Video-mtr: Reinforced multi-turn reasoning for long video understanding. arXiv preprint arXiv:2508.20478, 2025. 2, 6, 1, 3, 4
- [30] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115,

2024. 3, 4

- [31] Linli Yao, Haoning Wu, Kun Ouyang, Yuanxing Zhang, Caiming Xiong, Bei Chen, Xu Sun, and Junnan Li. Generative frame sampler for long video understanding. In Findings of the Association for Computational Linguistics, pages 17900– 17917, 2025. 2, 3, 1
- [32] Jiashuo Yu, Yue Wu, Meng Chu, Zhifei Ren, Zizheng Huang, Pei Chu, Ruijie Zhang, Yinan He, Qirui Li, Songze Li, et al. Vrbench: A benchmark for multi-step reasoning in long narrative videos. arXiv preprint arXiv:2506.10857, 2025. 2, 3, 6, 8, 1
- [33] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 3

- [34] Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, and Bo Zheng. Rewatch-r1: Boosting complex video reasoning in large vision-language models through agentic data synthesis. arXiv preprint arXiv:2509.23652, 2025. 2, 6, 3, 4
- [35] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13691– 13701, 2025. 2, 3, 6, 1, 4
- [36] Kejian Zhu, Zhuoran Jin, Hongbang Yuan, Jiachun Li, Shangqing Tu, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. Mmr-v: What’s left unsaid? a benchmark for multimodal deep reasoning in videos. arXiv preprint arXiv:2506.04141, 2025. 2, 3, 6, 1

[Figure 250]

## Conan: Progressive Learning to Reason Like a Detective over

## Multi-Scale Visual Evidence Supplementary Material

### A. Related Work

#### A.1. Video Reasoning Tasks

Recent advances in multimodal large language models (MLLMs) such as Qwen2.5-VL [2], KimiVL [24], MiMo-VL [28], and GPT-4o [12], have substantially improved video understanding, including captioning [25], question answering [1], and temporal grounding [9]. However, these capabilities mainly reflect perceptual understanding [19], whereas video reasoning [16], which demands multi-hop deduction and causal inference across frames, remains insufficiently explored and evaluated. To address this gap, several benchmarks have been introduced to assess the reasoning capabilities of MLLMs, such as VideoHolmes [6], VideoReasonBench [20], MMRV [36], VRBench [32], and VCRBench [22]. Unlike conventional video understanding tasks focused on recognizing visual content, these benchmarks require models to actively locate, connect, and reason over multiple relevant clues, thereby demanding a deeper comprehension of temporal dependencies and causal structures in dynamic visual narratives.

#### A.2. Video Reasoning Models

Inspired by the reasoning advancements of the DeepSeek-R1 [10], several studies [7, 15] adopt reinforcement learning with verifiable rewards [10] (RLVR) to promote video reasoning in MLLMs. While these approaches [7, 15, 21] encourage stepby-step reasoning, most are limited to text-only chain of thought, lacking explicit grounding in visual evidence, which often leads to unverified or hallucinated reasoning. To bridge this gap, concurrent works like Video-MTR [29] and FrameThinker [11] incorporate frame retrieval actions

into the reasoning process, enabling dynamic evidence gathering and long-form understanding. Despite great improvements in long video understanding [26, 27, 35], these methods partially depend on benchmark-specific training sets and still lack a reliable evidence identification mechanism, rendering the retrieval actions less reliable. Motivated by this, we aim to develop a framework named Conan that incentivizes deductive-like reasoning abilities in MLLMs, combining precise evidence identification, logical multi-step reasoning, and confident action decision towards robust video reasoning.

### B. Dataset Construction

Multi-scale Frame Categorization. Frames are categorized into three types (i.e. evidence, context, and distractor), based on their frame-level scores in GenS-Video-150K [31]. Specifically, frames with scores below 3 are labeled as distractor, those equal to 3 as context, and those above 3 as evidence.

Reasoning Trace Construction. To establish the dynamic answering threshold, we adopt a doubleverification mechanism. In particular, the accumulated frame descriptions and QA pair are first provided to K2 to assess whether the current evidence suffices to derive the correct answer. If so, K2 is prompted again without the answer to verify whether it can independently arrive at the correct option using the accumulated evidence. In addition, the prompt templates used to guide Kimi K2 in generating reasoning traces are illustrated in Figure 5.

Dataset Statistics. The Conan-91k dataset comprises 91,350 samples, including 60,000 for Conan-CoT-60k and 31,350 for Conan-RLVR31k. The duration distribution of each subset is shown in Figure 6, and the distribution of QA types is presented in Figure 7, respectively.

You are an assistant tasked with generating a step-by-step, concise reasoning process. You should act as a multimodal large language model to think based on the provided input and make the final action.

Input:

- - Question: {question}
- - Frame(s): {caption}
- - Frame type(s): {type} Guidelines for reasoning:

- 1. Begin by analyzing the question, clarifying what kind of evidence is required.
- 2. Analyze the relevant frames (evidence/contextual frames) with high scores that help answer the question, referencing the corresponding score if necessary.
- 3. Compare the available evidence across frames, giving a summary.
- 4. Explain why to retrieve the additional clip(s) or frame(s).
- 5. Conclude with a structured justification that motivates the action: {action}. The action should be rephrased concisely and fluently while maintaining all timestamps of retrived clip(s) at the end of the reasoning process. The reasoning process is:

- (a) Prompt template for free-form question.

You are an assistant tasked with generating a step-by-step, concise reasoning process. You should act as a multimodal large language model to think based on the provided input and make the final action.

Input:

- - Question: {question}
- - Option: {option}
- - Frame(s): {caption}
- - Frame type(s): {type} Guidelines for reasoning:

- 1. Begin by analyzing the question and option, clarifying what kind of evidence is required.
- 2. Analyze the relevant frames (evidence/contextual frames) with high scores that help answer the question, referencing the corresponding score if necessary.
- 3. Compare the available evidence across frames, giving a summary.
- 4. Explain why to retrieve the additional clip(s) or frame(s).
- 5. Conclude with a structured justification that motivates the action: {action}. The action should be rephrased concisely and fluently while maintaining all timestamps of retrived clip(s) at the end of the reasoning process. The reasoning process is:

- (b) Prompt template for multi-choice question.

Figure 5. Prompt templates for reasoning trace construction.

9079

Multi-choice

Free-form

7686

[Figure 251]

[Figure 252]

5369

4507

[Figure 253]

[Figure 254]

ConanCoT-60k

29,175

16,500 Conan- 14,850 RLVR-31k

30,825

2128

1721

[Figure 255]

[Figure 256]

92 5-10 min 10-15 min 15-20 min >20 min

76 5-10 min 10-15 min 15-20 min >20 min

[Figure 257]

[Figure 258]

###### (a) Conan-CoT-60k (b) Conan-RLVR-31k

Figure 6. Duration distribution of (a) Conan-CoT-60k and (b) Conan-RLVR-31k.

Figure 7. QA types distribution of Conan-CoT-60k and Conan-RLVR-31k.

### C. Experiment

#### C.1. Benchmarks & Baselines

Benchmark Descriptions. 1) The six challenging video multi-step reasoning benchmarks.

- • MMR-V [36] is a benchmark for multimodal deep reasoning in videos, containing 317 videos and 1,257 QA tasks designed to assess complex visual reasoning.
- • Video-Holmes [6] evaluates high-level video reasoning across 1,837 questions derived from 270 manually annotated suspense short films, covering seven reasoning task types.
- • VRBench [32] is a long-form video reasoning benchmark with 960 curated narrative videos spanning eight languages and seven categories, featuring 8,243 human-labeled multi-step QA pairs targeting temporal and causal reasoning.
- • VCRBench [22] assesses video chain-of-thought reasoning over 859 videos with 1,034 highquality QA pairs. We adopt its multiple-choice subset for evaluation.
- • Human-P&C [14] includes over 6,000 humanverified multiple-choice questions evaluating nine reasoning dimensions, covering both perceptual recognition and higher-level visual comprehension integrating commonsense or domain knowledge.
- • LongVideoReason [3] consists of 1,000 long videos designed to comprehensively evaluate reasoning along four dimensions: temporal, goal-oriented, spatial, and narrative understanding.

The four long-video understanding benchmarks.

- • LongVideoBench [27] evaluates long-context multimodal video understanding with 6,678 multiple-choice questions derived from videos up to one hour long, covering diverse real-world scenarios. We use its validation set and remove video subtitles for fair evaluation.
- • MLVU [35] is a multi-task benchmark containing 3,102 questions across nine categories, specifically designed for long-video comprehension. We evaluate the multiple-choice subset from the dev split (2,593 samples).
- • LVBench [26] assesses models’ ability to under-

stand and extract information from long videos of up to two hours, comprising 1,549 QA pairs in total.

• Video-MME [8] is a comprehensive benchmark for general video understanding, including 900 videos and 2,700 high-quality multiple-choice questions across diverse scenarios. Subtitles are excluded during evaluation for consistency.

##### Baseline Descriptions.

- • GPT-4o [12] is a state-of-the-art MLLM developed by OpenAI, demonstrating strong performance across diverse vision-language tasks.
- • Gemini 1.5 Pro, Gemini 2.0 Flash are advanced models from Google’s Gemini family, achieving leading performance on video understanding. Particularly, Gemini 2.0 Flash exhibits improved complex reasoning capability.
- • LLaVA-OneVision-7B [13] integrates the Qwen2 [30] language backbone with the SigLIP [33] vision encoder, achieving strong open-source performance in fine-grained visual understanding.
- • Kimi-VL-A3B-Instruct [24] is an efficient Mixture-of-Experts (MoE) MLLM built upon the Moonlight [18] LLM and the high-resolution MoonViT [24] vision encoder.
- • InternVL3-8B [4] employs the InternViT300M-448px-V2 5 [5] vision encoder and Qwen2.5-7B [30] backbone, delivering competitive open-source performance.

- • Video-R1 [7] enhances Qwen2.5-VL-7BInstruct with CoT SFT and T-GRPO-based RLVR to improve video reasoning.
- • VideoChat-R1 [15] applies RLVR to Qwen2.5VL-7B-Instruct, achieving strong spatiotemporal reasoning performance.
- • Video-MTR [29] introduces multi-turn reasoning and a gated bi-level reward mechanism to enhance long-video understanding of Qwen2.5VL-7B-Instruct.
- • Rewatch-R1 [34] augments RLVR with an Observation & Reasoning reward to improve multistep visual reasoning based on Qwen2.5-VL-7BInstruct.
- • FrameThinker [11] interleaves textual reasoning with visual frames, enabling multimodal

chain-of-thought reasoning for long video understanding built upon Qwen2.5-VL-7B-Instruct. We do not include this model in the main results table, as it is not open-sourced prior to our submission.

- • Qwen2.5-VL-3B-Instruct, Qwen2.5-VL7B-Instruct, Qwen2.5-VL-72B-Instruct [2] are part of the Qwen2.5-VL series, which combine the Qwen2.5 [30] language model with a redesigned Vision Transformer (ViT) architecture for enhanced visual grounding and understanding.

#### C.2. More comparison results & analyses

LongVideoBench MLVU LVBench Video-MME Qwen2.5-VL-7B-Instruct [2] 48.9 52.8 34.4 55.8 Video-R1 [7] 55.6 62.5 38.3 58.6 VideoChat-R1 [15] 54.3 60.5 38.0 56.9 Rewatch-R1 [34] 50.5 55.2 37.2 58.9 Video-MTR [29] 56.4 59.7 38.6 59.3 FrameThinker [11] 52.9 59.1 36.6 Conan 56.6 (↑ 7.7 ) 63.4 (↑ 10.6 ) 39.2 (↑ 4.8 ) 60.5 (↑ 4.7 )

- Table 3. Evaluation results on long video understanding. The results of FrameThinker are taken from its original paper, which does not evaluate the entire Video-MME benchmark.

Long Video Understanding. Beyond multi-step reasoning, Conan exhibits strong generalization to long video understanding tasks. As presented in Table 3, Conan consistently outperforms Qwen2.5VL-7B-Instruct across LongVideoBench [27], MLVU [35], LVBench [26], and Video-MME [8], achieving state-of-the-art performance compared with both text-CoT and video-CoT models. These results indicate that the high-quality, multi-scale evidence reasoning data and progressive training curriculum not only enhance multi-step reasoning but also effectively boost long video understanding. Frame Efficiency Analysis. We compare the efficiency of Conan with Qwen2.5-VL-7B-Instruct under varying numbers of initial input frames. As shown in Figure 8, Conan with only 8 initial input frames outperforms the base model that uses 64 frames, demonstrating its superior efficiency and accuracy in visual reasoning. However, we observe a slight performance decline on the HumanP&C benchmark as the number of input frames in-

###### Conan Qwen2.5-VL-7B-Instruct

50

50

88

45

45

78

40

40

35

35

68

30

30

25

25

58

8 16 32 64

8 16 32 64

8 16 32 64

(a) MMR-V

(b) Video-Holmes

(c) VRBench

55

78

55

53

73

50

51

68

49

45

63

47

40

58

45

8 16 32 64

8 16 32 64

8 16 32 64

(d) VCRBench

(e) LongVideoReason

(f) Human P&C

Figure 8. Performance variations of Conan and Qwen2.5-VL-7B-Instruct across different input frame numbers.

creases. This is reasonable, as introducing more initial frames also introduces additional distractors that are irrelevant to answering the question, thereby increasing the difficulty of frame identification.

Generalizability across model sizes and architectures. To evaluate the generalizability of our dataset and training framework, we apply them to different base models: Qwen2.5-VL-3B-Instruct and InternVL3-8B. As shown in Table 4, both Conan-3B and Conan-Intern-8B achieve consistent performance improvements over their respective baselines, demonstrating the strong extensibility and scalability of the proposed Conan-91k dataset and two-phase training framework across varying model sizes and architectures. Notably, although InternVL3-8B demonstrates stronger baseline performance than Qwen2.5-VL-7B-Instruct, ConanIntern-8B underperforms Conan-7B in multi-step reasoning. This discrepancy likely arises because InternVL3-8B employs a perception-oriented visual encoder and a relatively shallow cross-modal fusion module, which constrain its ability to integrate multi-step temporal cues and causal dependencies during reasoning [4]. In contrast, Qwen2.5-VL-7B-Instruct inherits reasoning priors and instruction-following capabilities from its text-based foundation model, thereby facilitating more effective compositional and multi-hop reasoning [2].

###### #Params MMR-V Video-Holmes VRBench VCRBench∗ Human-P&C LVR Overall

Qwen2.5-VL-3B-Instruct [2] 3B 29.0 28.8 59.7 38.0 41.7 57.6 42.5 Conan-3B 3B 33.1 (↑ 4.1 ) 31.9 (↑ 3.1 ) 68.0 (↑ 8.3 ) 42.7 (↑ 4.7 ) 44.3 (↑ 2.6 ) 61.3 (↑ 3.7 ) 46.9 (↑ 4.4 )

InternVL3-8B [4] 8B 35.5 32.8 75.8 45.7 51.0 68.0 51.5 Conan-Intern-8B 8B 43.3 (↑ 7.8 ) 41.8 (↑ 9.0 ) 81.7 (↑ 5.9 ) 48.4 (↑ 2.7 ) 50.2 (↓ 0.8 ) 70.8 (↑ 2.8 ) 56.0 (↑ 4.5 )

- Table 4. Performance comparison on multi-step reasoning of Conan variants and their corresponding base models. Overall represents the average results across six multi-step reasoning benchmarks. ∗ indicates the multiple-choice subset.

(a) Stepwise Identification–RetrievalOutcome Reward (Free-Form)

(b) Stepwise Identification–RetrievalOutcome Reward (Multi-Choice)

(c) Inference Round(s)

[Figure 259]

0.30

0.8

[Figure 260]

[Figure 261]

StageⅠ StageⅡ StageⅠ StageⅡ StageⅠ StageⅡ

0.28

2.5

0.7

0.26

0.24

0.5

2.0

0.22

0.5

0.20

1.5

0.4

0.18

0.16

0.3

1.0

500 1k 1.5k 2.0k 2.5k 3.0k 500 1k 1.5k 2.0k 2.5k 3.0k 500 1k 1.5k 2.0k 2.5k 3.0k

Figure 9. Training dynamics in the AIR RLVR process of Conan.

#### C.3. Training Dynamics of AIR RLVR

reward accuracy, indicating that Conan has internalized a compact and efficient multi-step reasoning strategy, retrieving evidence only when necessary, much like a detective who strategically gathers key clues rather than exhaustively examining all information.

To gain a deeper understanding of how the model’s behavior evolves during end-to-end reinforcement learning, we perform a fine-grained analysis of its training dynamics. As shown in Figure 9, the training process can be divided into two stages:

- Stage I: Accuracy-Oriented Evidence Exploration. Building upon the solid foundation established through multi-stage progressive cold-start, the model initially enters a phase characterized by frequent yet progressively more accurate frame retrieval. During this stage, it actively queries additional clips to maximize the joint identification–retrieval–outcome reward across tasks. The steadily rising reward curve indicates that the model adopts an exploratory strategy: Frequently engaging with the visual context to accumulate sufficient evidence for accurate reasoning. This stage marks a transitional period in which the model learns the value of comprehensive evidence collection before optimizing retrieval efficiency.
- Stage II: Efficient Evidence Retrieval. As training progresses, the model transitions to a more refined and selective retrieval policy. It significantly reduces retrieval frequency while maintaining high

#### C.4. More qualitative analyses

As shown in Figure 10, in case (a), while Qwen2.5VL-7B-Instruct incorrectly selects option “E” by relying on superficial textual associations, Conan correctly identifies option “D” through structured, evidence-grounded reasoning. During the Frame Identification stage, Conan identifies context and evidence frames that emphasize the contrast in lighting and tone between the man and the girl. It then performs explicit evidence reasoning, linking the recurring visual pattern, bright, warm lighting accompanying the girl’s appearance, to the symbolic themes of joy and hope she brings to the man’s life. Finally, Conan synthesizes this reasoning into a confident and accurate conclusion. In case (b), despite Rewatch-R1 producing a plausible reasoning process, it fails to anchor its analysis in explicit visual evidence and ultimately predicts the

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

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

[Figure 278]

Question: The film uses completely different lighting and tones when depicting the man and the girl together. What does this symbolize?

(A) It symbolizes the inherent conflict between light and darkness. (B) It illustrates the environmental impact on human emotions. (C) It shows the contrast between urban and rural lifestyles. (D) The daughter brings joy and hope into the man's life. (E) It symbolizes the contrasting emotional perspectives between the man and the girl. (F) It depicts the balance between dreams and reality. (G) It represents the historical differences within the community. (H) It highlights societal roles during different eras. (I) It represents the diversity within the neighborhood. (J) It is symbolic of generational wisdom versus youthful ignorance.

Qwen2.5-VL-7B-Insturct: E

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

Round 1

Distractor frames Evidence frames Evidence frames Distractor frames

Frame Identification: Context frames

[Figure 295]

Evidence Reasoning:1. Question focus We need to identify what the stark difference in lighting and tone between the man and the girl’s scenes symbolizes...The man is outside with the girl, both happy; the lighting is bright and cheerful.3. Evidence summary The lighting and tone shift whenever the man and the girl are together. When they are apart, the man is often shown in a dim, somber setting (chair, wheelchair, sad expressions). When they are together, the scenes are consistently bright, warm, and hopeful. This pattern is repeated multiple times, emphasizing the positive impact the girl has on the man’s mood.4. Option evaluation A – No direct conflict between light and darkness is highlighted. B – No environmental impact is discussed. C – No urban vs. rural contrast is mentioned. D – The girl’s presence consistently brings joy and hope, aligning perfectly with the observed lighting changes. E – No contrasting emotional perspectives are explicitly stated. F – No balance between dreams and reality is emphasized. G – No historical or generational differences are highlighted. H – No societal roles during different eras are addressed. I – No diversity within the neighborhood is discussed. J – No generational wisdom vs. youthful ignorance is mentioned.5. Conclusion The consistent visual cue of brighter, warmer lighting whenever the man and the girl are together symbolizes the girl bringing joy and hope into the man’s life. Therefore, the correct answer is D. Action Decision: Confident Question Answering D

###### (a) A qualitative example from MMR-V.

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

Question: The video's creators aimed to connect one of the depicted animals to a prehistoric creature through contextual clues and visual storytelling. Which animal\'s presentation most strongly implies this intentional link to an extinct species?

A. Jaguar resting in tropical grass (0:00:00-0:00:10) B. Crocodile labeled "TITANOBOLA" (0:00:10-0:00:20) C. Large coiled snake fossil in a museum (0:00:40-0:00:50) D. Proboscis monkey in Bornean rainforest (0:04:00-0:04:10)

Rewatch-R1: Let me consider each option carefully. The Jaguar, (A), shows an animal still alive today. The Proboscis monkey (D) is also a living primate. For an explicit connection to a prehistoric creature, neither of these fits the bill perfectly. Now, let's look at the crocodile. It's explicitly labeled \"TITANOBOLA\", which immediately brings to mind the prehistoric reptile Titanoboa, one of the largest. However, a label alone doesn't directly relate to its current physical state or morphology. Finally, we have the large, coiled snake fossil (C). While the jaguar and monkeys are present, I'm drawn to this fossil because it’s a direct prehistoric representation. Unlike the modern animals, it's an inanimate object, clearly showing characteristics of an ancient species that would be classified with dinosaurs during the Mesozoic era. Its presence strongly implies an evolutionary lineage connecting primitive terrestrial reptiles or early tetrapods to modern snakes, even if dinosaurs weren't depicted in a specific way. Therefore, between the jaguar, monkey, crocodile, and the fossilised snake, the fossil clearly exemplifies the creators' intent to show an explicit connection to a prehistoric creature through its very nature as a 'fossil' and a display meant to represent extinction. Answer: B

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

Round 1

Evidence frames Evidence frames Context frames

Frame Identification: Context frames

[Figure 329]

Evidence Reasoning:1. Clarify the question: We need the animal whose presentation explicitly links it to a prehistoric creature, using contextual clues and visual storytelling. 2. Examine high-scoring frames that contain animals - 00:00:48– “large coiled snake fossil … in a museum setting.” 00:02:25 – “a close-up of a caterpillar on a leaf.” 00:03:13– “a large cockroach on a hand.” - 00:04:02 – “proboscis monkeys in a forest.” - 00:04:50– “a green and black poison dart frog.” - 00:05:39– “ants on a tree trunk.”

- 00:06:27– “a spider on a leaf.” - 00:08:04– “an owl in flight.” - 00:08:52– “a crocodile with the text ‘TITANOBOLA’.” - 00:09:41 – “a white axolotl.”3. Compare the evidence - 00:00:48 shows snake fossils in museums, explicitly labeled as “fossil” and “prehistoric.” - 00:08:52 shows a crocodile labeled “TITANOBOLA,” but the label is modern and the context is a zoo, not a museum. - Other frames either depict living animals or do not emphasize a prehistoric link.4. Reasoning The snake fossils are the only animals whose presentation directly connects them to prehistoric creatures via museum displays and explicit labeling. The crocodile’s label is modern and the setting is a zoo, so it does not fulfill the “prehistoric link” requirement as strongly.5. Conclusion The animal whose presentation most strongly implies a connection to an extinct species is the large coiled snake fossil. Answer: C Action Decision: Confident Question Answering C

(b) A qualitative example from LongVideoReason.

Figure 10. Two qualitative examples from MMR-V and LongVideoReason are presented, illustrating the reasoning traces and final answers of Rewatch-R1 and Conan for comparison.

wrong answer due to misalignment between reasoning (answer C) and action (answer B). In contrast, Conan accurately identifies the correct option through structured reasoning grounded in evidence frames. In the Frame Identification stage, Conan localizes key frames featuring the fossil exhibit, then performs evidence reasoning to associate the fossil’s visual features and its museum context, ex-

plicitly symbolizing extinction, with the concept of a prehistoric creature. Conan then integrates this reasoning into a coherent and confident response, showcasing robust visual grounding and strong reasoning–action alignment. Both of the two cases highlight Conan’s ability to perform compositional reasoning over visual cues, aligning symbolic interpretation with concrete visual evidence rather than

relying on spurious correlations.

Error Analyses. Figure 11 presents two representative error cases from Video-Holmes and VCRBench. In error case (a), Conan accurately identifies and aligns the key evidence frames showing the masked man’s repeated appearance on the phone screen. However, it incorrectly infers that the masked man is the boyfriend in disguise (Option B) rather than recognizing the correct explanation of supernatural infiltration (Option E). This misjudgment arises from a hallucinated causal link, interpreting visual recurrence as intentional disguise, while neglecting contextual cues, such as the absence of any indication of the boyfriend’s presence. The case reveals Conan’s limitation in distinguishing plausible narrative causality from repetitive visual patterns. In error case (b), Conan interprets the monkey’s seemingly gentle behavior toward the pigeon as an intention to befriend it (Option B). While the model accurately focuses on consistent proximity and interaction frames, it overlooks subtle affective cues, such as the monkey’s frustrated expressions and its act of using a scarecrow to drive the pigeon away, which point to the correct answer (Option C). This error underscores Conan’s limitation in capturing fine-grained emotional and social dynamics, leading to an over-reliance on surfacelevel behavioral patterns.

Collectively, these cases reveal Conan’s ongoing challenges in integrating implicit visual evidence and affective reasoning beyond superficial visual and temporal correlations.

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

Question: What is the root cause of the phone displaying a masked man when the girl took a photo for the second time?

- A. Mobile phone camera failure.
- B. Boyfriend disguised as a masked man.
- C. Photo post-processing.
- D. Light refraction illusion.
- E. The masked man has infiltrated the room.
- F. The girl had hallucinations.

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

Distractor frames Evidence frames Round 1 Evidence frames Distractor frames

Frame Identification: Context frames

[Figure 363]

Evidence Reasoning: Step-by-step reasoning 1. Question focus We need the *root cause* of the masked man appearing on the phone screen the second time the girl takes a photo.2. Key evidence from high-score frames - 00:00:09 – The first masked-man image appears on the phone. - 00:00:18 – The girl is now holding the phone and the same masked-man image is still visible. - 00:01:05 – The girl is again holding the phone; the masked man is still on the screen. - 00:01:24 – The girl is now taking a second photo, and the masked man is still visible on the phone. - 00:01:33 – The girl is still holding the phone; the masked man is still on the screen. - 00:02:20 – The girl is looking at the phone; the masked man is still visible.3. Comparison & summary Across every high-score frame, the masked man’s image is consistently shown on the phone screen while the girl is actively taking photos. No other explanations (camera failure, post-processing, light refraction, hallucinations, or infiltration) are supported by the visual evidence. 4. Option evaluation A: Camera failure – no technical issues are mentioned. B: Boyfriend disguised as a masked man – the repeated appearance of the same masked figure suggests a deliberate act rather than a one-time prank. C: Post-processing – no editing tools or filters are visible. D: Light refraction – no such effect is observed. E: Infiltration – no intruder is seen. F: Hallucinations – the masked man is repeatedly visible in the same image, not just once.5. Conclusion The only consistent and supported explanation is that the boyfriend is disguised as the masked man and is deliberately appearing in the photos.Answer: B Action Decision: Confident Question Answering B

###### (a) Error case from Video-Holmes.

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Question: What is the monkey's attitude towards the pigeon?

- A. Catching pigeons
- B. Make friends with pigeons
- C. Want to drive away pigeons
- D. Hope to take care of the injured pigeon.

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Round 1 Frame Identification: Distractor frames Evidence frames

Context frames

[Figure 397]

Evidence Reasoning: Step-by-step reasoning 1. Question focus We need to know the monkey’s attitude toward the pigeon(s). The key is to identify any overt actions or expressions that show the monkey’s feelings.2. High-score frames that contain both the monkey and the pigeon - 00:00:00 – 00:00:02: The monkey is simply sitting on the ground; no interaction yet. - 00:00:05 – 00:00:06: The monkey is now actively talking to the pigeon while it is on the ground. - 00:00:08 – 00:00:10: The monkey is still talking to the pigeon while it is on the ground. - 00:00:12 – 00:00:15: The monkey is now standing next to the pigeon, which is perched on a box, and continues to talk to it.3. Evidence summary Across these frames, the monkey’s consistent behavior is to approach the pigeon, sit or stand beside it, and speak to it. There is no attempt to catch, drive away, or care for the injured pigeon; the monkey’s sole focus is communication.4. Option comparison A. Catching pigeons – no attempt to grab or harm the bird. B. Make friends with pigeons – the monkey’s repeated interaction and conversation suggest friendship. C. Want to drive away pigeons – no aggressive gestures or movements. D. Hope to take care of the injured pigeon – no physical care or concern shown.5. Conclusion The monkey’s attitude is best described as wanting to make friends with the pigeon. Answer: B Action Decision: Confident Question Answering B

(b) Error case from VCRBench.

Figure 11. Two error cases of Conan from Video-Holmes and VCRBench.

