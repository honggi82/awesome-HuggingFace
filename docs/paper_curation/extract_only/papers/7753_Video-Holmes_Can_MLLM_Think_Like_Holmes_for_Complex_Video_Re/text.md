arXiv:2505.21374v1[cs.CV]27May2025

# Video-Holmes: Can MLLM Think like Holmes for Complex Video Reasoning?

### Junhao Cheng1,2, Yuying Ge1, , Teng Wang1, , Yixiao Ge1, Jing Liao 2, Ying Shan1 1ARC Lab, Tencent PCG 2City University of Hong Kong https://video-holmes.github.io/Page.github.io/

00:05-00:30

Q: What caused the man's death?

I think that A, B, and D are supported by evidence. The man's

- A. Chronic disease
- B. Overwork for a long time
- C. The file was poisoned by the boss
- D. Side effects of superpowers
- E. Sudden illness
- F. Violent injury

nosebleed might be due to a

The Man receive lots of documents. This indicate that he may be overworked.

chronic disease, which could lead to death. The man's boss giving him a heavy workload might result in death from overwork. The other

- 00:45-01:19

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

The man sneeze to finish work until a nosebleed,

suggested something abnormal with his body.

- 01:25-01:57

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

The man sneezed, then the camera showed an old man and a skeleton, suggesting all three are the same person experiencing time jumps.

- 02:10-02:40

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

options lack clear evidence to

The man sneezed, and the file was completed. He may have a time-jump-like superpower.

[Figure 19]

[Figure 20]

[Figure 21]

support them.

However, A and B overlook the key event highlighted in the video: sneezing may trigger time jumps.

I first need to analyze abnormal events related to the man and then link their relationships.

Due to the man's frequent use of

this ability, it could have negative effects, causing a time jump to much later (when the man is old and dies), leading to the tragedy.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

In conclusion, I think the most

likely reason is D: Side effects of superpowers.

[Figure 29]

[Figure 30]

Active Seeking Chain-of-clues Final Answer

- Figure 1: An example of Video-Holmes. Models are required to actively locate and connect multiple relevant visual clues scattered across different video segments to render the final answer.

## Abstract

Recent advances in Chain-of-Thought (CoT) reasoning and reinforcement learning (RL) post-training have been reported to enhance video reasoning capabilities of multimodal large language models (MLLMs). This progress naturally raises a question: can these models perform complex video reasoning in a manner comparable to human experts? However, existing video benchmarks primarily evaluate visual perception and grounding abilities, with questions that can be answered based on explicit prompts or isolated visual cues (e.g., “What is the woman wearing?”). Such benchmarks do not fully capture the intricacies of realworld reasoning, where humans must actively search for, integrate, and analyze multiple clues before reaching a conclusion. Empirical results show that even models with advanced thinking abilities achieve only marginal gains (e.g., from 68.3% to 69.4%) on these benchmarks, which raises doubts about the extent to which these tasks require genuine reasoning. To address this issue, we present Video-Holmes, a benchmark inspired by the reasoning process of Sherlock Holmes, designed to evaluate the complex video reasoning capabilities of MLLMs. VideoHolmes consists of 1,837 questions derived from 270 manually annotated suspense short films, which spans seven carefully designed tasks. Each task is constructed by first identifying key events and causal relationships within films, and then designing questions that require models to actively locate and connect multiple relevant visual

Preprint. Under review.

clues scattered across different video segments. We conduct a detailed analysis of model reasoning processes, examining the factors that lead to both correct and incorrect answers. Our comprehensive evaluation of state-of-the-art MLLMs reveals that, while these models generally excel at visual perception, they encounter substantial difficulties with integrating information and often miss critical clues. For example, the best-performing model, Gemini-2.5-Pro, achieves an accuracy of only 45%, with most models scoring below 40%. We aim that Video-Holmes can serve as a “Holmes-test” for multimodal reasoning, motivating models to reason more like humans and emphasizing the ongoing challenges in this field. The benchmark is released in https://github.com/TencentARC/Video-Holmes.

## 1 Introduction

The development of CoT reasoning [1] and RL post-training strategies [2] have contributed to significant improvements in the reasoning abilities of LLMs [3, 4, 5]. By generating human-like reasoning steps, these models have shown strong performance in addressing complex reasoning tasks. Furthermore, these advancements have been successfully adapted to MLLMs for video understanding and reasoning [6, 7, 8, 9]. This progress naturally raises a question: can these models perform complex video reasoning in a manner comparable to human experts?

However, existing evaluation benchmarks for video reasoning [10, 11, 12, 13, 14, 15, 16, 17] are limited by their predominant focus on assessing the visual perception and grounding capabilities of models, where questions that can answered based on explicit prompts or isolated visual cues (e.g., “What is the woman wearing?”). Such benchmarks do not fully capture the intricacies of real-world reasoning, where humans must actively search for, integrate, and analyze multiple clues before reaching a conclusion. Empirical results show that even models with advanced thinking abilities [9] achieve only marginal gains (e.g., from 68.3% to 69.4%) on these benchmarks [18], which raises doubts about the extent to which these tasks require genuine reasoning.

To address this issue, we present Video-Holmes, a benchmark inspired by the reasoning process of Sherlock Holmes. It is designed to assess the complex video reasoning abilities of MLLMs and exam the factors contributing their correct and incorrect answers. As demonstrated in Table 1, Video-Holmes differs from existing benchmarks in several key aspects: (1) We utilize suspense short films as the video sources with detailed manual annotations. These videos are characterized by rich elements of suspense, reasoning, and supernatural themes, making them particularly challenging for models to comprehend. (2) The questions in Video-Holmes require models to actively locate and connect multiple relevant visual clues scattered across different video segments to infer the final answer. As illustrated in Figure 1, models first need to identify the abnormal scene involving the man and then progressively integrate the extracted visual clues to deduce the cause of the man’s death, just like the reasoning process of Sherlock Holmes. (3) We provide detailed analysis of models’ reasoning processes, examining the factors that lead to both correct and incorrect answers.

Our comprehensive evaluation of state-of-the-art (SOTA) MLLMs reveals that, while these models generally excel at visual perception, they encounter substantial difficulties with integrating information and often miss critical clues. For example, the best-performing model, Gemini-2.5-Pro, achieves an accuracy of only 45%, with most models scoring below 40%.

We make the following contributions in this work:

- • We present Video-Holmes, a benchmark for complex video reasoning. Video-Holmes comprises 270 manually annotated suspense short films, along with 1,837 challenging questions, which require models to actively locate and link multiple relevant visual clues, offering the research community a high-quality and challenging video reasoning benchmark.
- • We conduct extensive experiments on Video-Holmes to evaluate existing SOTA MLLMs. We analyze the reasoning processes of these models and observe that while they perform well in visual perception, they face significant challenges in integrating clues and frequently overlook critical clues. These observations provide valuable insights for future research.

- Table 1: Comparison between Video-Holmes and existing video reasoning benchmarks across several key aspects: the video source domain (Domain), annotation methodology (Anno.), the number of reasoning QA pairs (RQA Pairs), necessity for models to actively seek out clues (Active Seeking), necessity for models to link multiple clues (Chain-of-Clues), whether provide reasoning process analysis (RPA), and whether provide audio information (Aud.).

Benchmarks Domain Anno. RQA Pairs Active Seeking Chain-of-Clues RPA Aud.

VSI-Bench [10] Indoor scenes A&M 5,130 ✗ ✗ ✗ ✗ MVBench [15] Open A 4,000 ✗ ✗ ✗ ✗ TempCompass [16] Open A&M 7,540 ✗ ✗ ✗ ✗ Video-MMMU [14] Academic M 900 ✗ ✗ ✗ ✓ MMVU [12] Science M 3,000 ✗ ✗ ✗ ✓ Video-MME [18] Open M 1,944 ✗ ✗ ✗ ✓ VCR-Bench [13] Open A&M 1,034 ✗ ✗ ✓ ✗

Video-Holmes (Ours) Suspense short films A&M 1,837 ✓ ✓ ✓ ✓

## 2 Related Works

MLLMs for Video Understanding and Reasoning. The advancements in MLLMs for image understanding and reasoning [19, 20, 21, 22] have enabled their extension to video-based tasks. Methods such as VideoChat [23], Video-ChatGPT [24], CogVLM2Video [25], InternVL [26, 27], LLaVAVideo [28], and Qwen-VL [29, 30] treat videos as sequences of images, allowing large language models (LLMs) to perform video understanding and reasoning. CoT reasoning and reinforcement post-training strategies have been shown to enhance the reasoning capabilities of LLMs [4, 3], and recent work extends these techniques to MLLMs for multimodal reasoning tasks involving images [31, 32, 33, 34, 35, 36] or videos [7, 6, 37, 8, 38]. For instance, Video-R1 [6] and VideoChatR1 [7] employ supervised fine-tuning (SFT) followed by reinforcement learning post-training using Group Relative Policy Optimization (GRPO) [2] based on Qwen-VL-2.5 [39]. These methods achieve higher performance compared to Vanilla Qwen-VL-2.5 across diverse reasoning tasks.

Video Reasoning Benchmarks. Early video understanding benchmarks primarily assess model capabilities within specific scenarios. For instance, MSRVTT-QA [40], ActivityNet-QA [41], and NExT-QA [42] focus on fundamental tasks such as action recognition and video question answering. Recently, benchmarks like MMBench [43], TempCompass [16], and MVBench [15] evaluate reasoning over short video clips, while LongVideoBench [44] and Video-MME [14] extend evaluations to longer video sequences. However, these tasks are generally straightforward and do not require complex reasoning. With the success of chain-of-thought (CoT) reasoning, there is increasing interest in advancing video reasoning in more challenging scenarios. Benchmarks such as MMVU [12] and VideoMMMU [11] evaluate reasoning in academic and scientific domains, while VSI-Bench [10] focuses on indoor environments. The recent VCR-Bench [13] introduces a benchmark specifically designed to assess CoT reasoning in video tasks. Despite these developments, such benchmarks primarily evaluate visual perception and grounding abilities, with questions that can be answered based on explicit prompts or isolated visual cues and do not fully capture the intricacies of real-world reasoning. In contrast, Video-Holmes require models to actively locate and connect multiple relevant visual clues, engaging them in a more complex and demanding reasoning scenario.

## 3 Video-Holmes

As shown in Figure 2, the construction of Video-Holmes involves three steps: video collection and annotation, task definition, and question-answer-explanation generation.

Video Collection and Annotation. Suspense short films serve as an ideal source for evaluating the complex video reasoning capabilities of MLLMs, as they are characterized by compact narratives enriched with hints, plot twists, and supernatural elements. We utilize the keyword “suspense short films” to search videos from YouTube with durations between 1 and 5 minutes. We incorporate nine subkeywords1 in this process to ensure diversity. From the initial pool of over 2,500 videos with audio information retrieved through our search, we manually curated a subset of 270 high-quality,

1Details are provided in Appendix C.

|[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]<br><br>[Figure 34]|
|---|

|[Figure 35]<br><br>[Figure 36]|
|---|

|[Figure 37]<br><br>[Figure 38]|
|---|

|[Figure 39]<br><br>[Figure 40]|
|---|

|[Figure 41]<br><br>[Figure 42]|
|---|

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Reasoning

Temporal Causal Inference

Segmented Plot Descriptions

Type

Let's break down the situation. The man in the suit is seen touching his pocket and then frowning. This action suggests a sense of urgency or concern. First, let's consider the options: A: Check if the wallet is lost. - This is a plausible option because touching the pocket

[00:00-00:15]: The man in the black hat is staking out the house, preparing to break in and commit a robbery……

Vision Omission Error

[Question]: Why the man in suit turn back

halfway? [Options]:

Reason

The model failed to correctly associate the scene where the man in the suit touches his pocket with the subsequent theft of the phone by the thief during the inference process. The analysis clearly indicates that the

- A. Check if the wallet is lost
- B. Forgot to bring phone
- C. Forgot to bring wallet …… [Explanation]: The action of the man patting his pocket is directly linked to the shot of the phone being stolen inside the room, and the forgotten phone triggers a return action.

Key Character Relationships

[Combination]: Man in the black hat-Man in a suit [Relation]: Burglar and homeowner [Reason]: The man in the black waited for the man in the suit to drive away before entering his house to steal items.

could indicate a search for something, and the

frown might suggest disappointment if he finds nothing. B: Realize that you forgot to bring your mobile phone. - This is less likely because

Reasoning Shots

there's no indication of a mobile phone in the scene, and the focus is on the pocket…... Given

camera switch implies the man in the

[Time]: 01:43 [Clues]: The man in the suit pats his pocket, the next shot shows the thief taking the phone from the table. [Conclusion]: The man forgot to bring his phone.

Social Reasoning Physical Anomaly Reasoning Multimodal Hint Reasoning Intention & Motive Chaining Timeline Analysis Core Theme Inference

suit realizes his phone is missing, but the model mistakenly attributes the action to the loss of a wallet, overlooking the crucial information of the phone being stolen in the video,

this, the most likely intention is that the man is

checking if his wallet is lost. The frown suggests disappointment, which aligns with the idea that he found nothing in his pocket. Therefore, the most reasonable answer is A.

Supernatural Elements

None

which is considered a vision omission

Core Theme

Answer

error.

The cycle of karma.

A

Human Annotation Question/Answer/Explanation Generation Model Evaluation Reason Process Analysis

- Figure 2: Construction and evaluation pipeline of Video-Holmes. We select 270 high-quality suspense short films for human annotation. Next, we design 7 challenging tasks and employ DeepSeek to generate questions. Finally, we evaluate SOTA MLLMs and use DeepSeek to analyze their responses.

reasoning-reach short films with a rigorous annotation process. Each film is annotated following a structured template that considers the following aspects:

- • Segmented Plot Descriptions: Annotators are asked to divide the video into segments based on the progression of the storyline, and provide detailed descriptions for each segment.
- • Key Character Relationships: Annotators are asked to present the relationships between key characters in the video, along with evidence that supports the identification.
- • Reasoning Shots: Annotators are asked to identify reasoning shots in the video, providing timestamps, visual clues, and the inferred conclusions associated with these shots.
- • Supernatural Elements: Annotators are asked to specify any supernatural elements present in the videos and the implications they introduce, whether positive or negative.
- • Core Theme: Annotators are asked to summarize the core themes of the videos.

These diverse short films with intricate reasoning chains, along with high-quality and well-formatted manual annotations ensures the reliability and quality of Video-Holmes.

Task Definition. To comprehensively evaluate the differences in MLLMs’ capabilities for complex video reasoning from multiple perspectives, we define seven distinct reasoning tasks for VideoHolmes. As illustrated in Figure 3, different from existing benchmarks that primarily designed around clue-given questions, Video-Holmes focus on tasks that require models to actively locate and connect multiple relevant visual clues scattered across different video segments:

- • Social Reasoning (SR): Inferring social relationships between characters. This includes identifying identity associations across time (e.g., the same man in youth and old age).
- • Physical Anomaly Reasoning (PAR): Identifying scenes in the video that deviate from real-world norms and reasoning about their underlying rules or implicit meanings.
- • Multimodal Hint Reasoning (MHR): Decoding cues or fact from multimodal hints, such as semantic implications of camera movements or gradual changes in object positions.
- • Intention & Motive Chaining (IMC): Observing characters’ actions or environmental cues to disentangle surface behaviors from underlying behavioral intentions.
- • Temporal Causal Inference (TCI): Inferring causal mechanisms between events across time and space using cinematic language and multimodal clues.
- • Timeline Analysis (TA): Integrating and reconstructing the narrative storyline of the film.
- • Core Theme Inference (CTI): Extracting the core theme or deeper meaning of the video by analyzing its plot, dialogues, and symbolic elements.

Question-Answer Generation. We utilize DeepSeek-R1 [3] with advanced reasoning capabilities to automatically generate questions based on formatted manual annotations and predefined question

Answer with Given Clues (Existing Benchmarks)

Temporal Reasoning

Spatial Reasoning

Object Grounding

Object Counting

Q: What will the woman do next at 00:16?

Q: What is the spatial relation between the toaster and the kettle?

Q: Where is the red mup at 02:32?

Q: How many bowls is on the table?

- A. open the curtains
- B. get off work
- C. take a nap

- D. play games
- E. eat food
- F. do cleaning

- A. on the Left
- B. above
- C. on the right

- D. below
- E. in front of
- F. behind

- A. [14,0,323,52]
- B. [14,0,100,52]
- C. [14,0,32,235]

- D. [14,0,100,12]
- E. [72,0,143,32]
- F. [32,5,323,52]

- A. 0
- B. 1
- C. 2

- D. 3
- E. 4
- F. 5

|[Figure 55]<br><br>[Figure 56]|
|---|

|[Figure 57]<br><br>[Figure 58]|
|---|

|[Figure 59]<br><br>[Figure 60]|
|---|

|[Figure 61]<br><br>[Figure 62]|
|---|

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

00:16

01:59

02:32

00:23

Actively Seek Clues (Video-Holmes)

[Figure 71]

Multimodal Hint Reasoning (MHR)

[Figure 72]

[Figure 73]

Physical Anomaly Reasoning (PAR)

Social Reasoning (SR)

Q: What is the relationship between the man in the white suit and the man in the red shirt?

Q: What are the rules of supernatural in the video?

Q: What do the apple and pencil symbolize in the video?

- A. Sneezing triggers time jumps
- B. Holding cups can make time pass faster
- C. The man complete work very fast
- D. The man can perform consciousness invasion
- E. Sneezing triggers time reversal
- F. There are no supernatural phenomena

- A. Superpowers are limited to filling out forms
- B. The man’s fruit allergy leads to tragedy
- C. The man’s talent for drawing
- D. Superpowers impact the objects themselves
- E. Hint at the nature of the man’s job
- F. The man is about to reach the top of his life

- A. colleague
- B. superior and subordinate
- C. brother

- D. competitor
- E. customer
- F. stranger

VisualCluesVisualCluesVisualClues

|[Figure 74]<br><br>[Figure 75]|
|---|

|[Figure 76]<br><br>[Figure 77]|
|---|

|[Figure 78]<br><br>[Figure 79]|
|---|

|[Figure 80]<br><br>[Figure 81]|
|---|

|[Figure 82]<br><br>[Figure 83]|
|---|

|[Figure 84]<br><br>[Figure 85]|
|---|

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

00:10 00:16 00:43 01:03 02:32 02:59

[Figure 94]

[Figure 95]

Intention & Motive Chaining (IMC) Temporal Causal Inference (TCI)

Q: What was the intention of the man in white looking at the leader's office door at 01:36?

Q: What was the direct reason why the man in white was finally trapped in the safe?

- A. Prepare to report on work
- B. Angry because of overtime work
- C. Plan to steal property

- D. Check if the door is locked for safe
- E. Recall past experiences
- F. Worry about the leader finding out

- A. Superpowers suddenly fail
- B. The leader locked the door
- C.The safe closes automatically

- D. Black hole paper falls to the ground
- E. Exhausted and unable to escape
- F. Black holes devour space and expand

|[Figure 96]<br><br>[Figure 97]|
|---|

|[Figure 98]<br><br>[Figure 99]|
|---|

|[Figure 100]<br><br>[Figure 101]|
|---|

|[Figure 102]<br><br>[Figure 103]|
|---|

|[Figure 104]<br><br>[Figure 105]|
|---|

|[Figure 106]<br><br>[Figure 107]|
|---|

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

00:20 00:42 01:36 01:13 01:47 02:07

Timeline Analysis (TA) Core Theme Inference (CTI)

[Figure 116]

[Figure 117]

Q: The core theme of the video is closest to which of the following?

Q: What is the correct sequence of the following events?

(1) The man in black discovers the basement (2) The man in the suit turns back

- A. Human nature is inherently evil
- B. Justice may be delayed but will prevail
- C. The helplessness of low-level criminals
- D. Continuation of childhood trauma
- E. The cycle of karma
- F. Malice between strangers

(3) The man in black is attacked by a girl (4) The man in the suit forgets his phone

(5) The man in black steals a phone (6) A car accident occurs

- A. 4-5-1-3-2-6
- B. 5-4-3-1-2-6
- C. 4-5-2-1-3-6

- D. 5-1-3-4-2-6
- E. 2-4-5-1-3-6
- F. 1-2-3-4-5-6

|[Figure 118]<br><br>[Figure 119]|
|---|

|[Figure 120]<br><br>[Figure 121]|
|---|

|[Figure 122]<br><br>[Figure 123]|
|---|

|[Figure 124]<br><br>[Figure 125]|
|---|

|[Figure 126]<br><br>[Figure 127]|
|---|

|[Figure 128]<br><br>[Figure 129]|
|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

00:10 00:37 01:04 01:11 01:53 02:02

- Figure 3: Comparison of question types between Video-Holmes and existing benchmarks. Existing benchmarks primarily involve clue-given questions, where models depend on explicitly provided clues to derive answers. In contrast, Video-Holmes adopts an active seeking paradigm, requiring models to actively locate and connect multiple relevant visual clues scattered across different video segments. (Key frames are marked with black boxes and magnified.)

types. Each question is generated by strictly adhering to the provided information, with manual sampling inspection to ensure quality and relevance. Additionally, the model is required to provide correct answer explanations for each question, which are used to compare and analyze the model’s reasoning process. Please refer to Appendix B for details.

After data verification and annotation, we have ultimately constructed a dataset comprising 270 videos and 1,837 question-answer pairs. The key statistics of Video-Holmes are presented in Appendix C.

- 4 Experiments

### 4.1 Setup

Evaluation Models. We conduct an evaluation of several mainstream MLLMs, including the opensource models: InternVL2.5 (8B) [45], InternVL3 (8B) [27], Qwen2.5-VL (7B, 32B) [39], and Qwen2.5-Omni (7B) [30]. Additionally, we assess open-source models that incorporate RL post-

- Table 2: Results of various models on Video-Holmes, where SR stands for Social Reasoning; IMC stands for Intention & Motive Chaining; TCI stands for Temporal Causal Inference; TA Timeline Analysis; MHR stands for Multimodal Hint Reasoning; PAR stands for Physical Anomaly Reasoning; CTI stands for Core Theme Inference. Blue represents the vanilla model, while Green represents its corresponding thinking version with RL post-training.

Model Frames SR IMC TCI TA MHR PAR CTI Overall Open Source Models

InternVL2.5-8B 32 28.0 32.2 21.5 07.7 25.7 23.8 22.6 23.8 InternVL3-8B 32 29.5 40.7 37.9 35.1 24.6 38.9 24.1 32.3 Qwen2.5-Omni-7B 32 27.1 19.9 13.9 7.5 14.8 14.9 13.7 16.4 Qwen2.5-VL-32B 32 43.2 44.2 31.5 51.0 36.4 31.4 32.2 38.4 Qwen2.5-VL-7B 32 38.4 34.8 17.6 30.0 27.1 18.6 25.2 27.8 SEED-Bench-R1 32 42.8 35.1 25.6 40.5 29.2 29.9 32.6 33.5 VideoChat-R1 32 42.1 38.8 24.5 39.5 29.5 27.8 29.3 33.0 Video-R1 32 48.6 41.7 28.9 34.5 31.0 33.5 35.9 36.5

Closed Source Models

GPT-4o 32 50.0 49.6 38.8 30.0 44.0 39.2 37.0 42.0 OpenAI o4-mini 32 36.3 31.2 20.5 34.0 30.1 30.9 27.4 29.9 Claud 3.5 Sonnet 20 48.6 43.5 30.8 41.0 39.8 36.6 33.7 39.3 Claud 3.7 Sonnet 20 45.9 48.2 33.7 39.5 40.7 39.7 38.1 41.0

- Gemini-1.5-Pro - 52.1 48.2 34.4 26.0 39.2 46.4 38.9 41.2
- Gemini-2.5-Pro - 46.6 49.3 46.9 53.0 40.1 44.3 37.4 45.0 Gemini-2.0-Flash - 41.8 33.7 23.1 20.5 30.1 26.8 33.7 30.6 Gemini-2.0-Flash-Thinking - 43.4 46.9 43.1 51.0 37.9 43.6 39.3 43.1

Table 3: Thinking model performances on Video-Holmes and other benchmarks.

Model MMVU Video-MME Video-MMMU MVBench Video-Holmes Gemini-2.0-Flash 65.2 68.3 54.3 55.3 30.6 Gemini-2.0-Flash-Thinking 64.5 (-0.7) 69.4 (+0.9) 56.4 (+1.9) 57.4 (+2.1) 43.1 (+12.5)

training based on Qwen2.5-VL (7B): SEED-Bench-R1 [8], Video-R1 [6], and VideoChat-R1 [7]. We also include several advanced closed-source models in our evaluation: Gemini-2.0-Flash [46], Gemini-2.0-Flash-Thinking [9], Gemini-1.5-Pro [47], Gemini-2.5-Pro [48], GPT-4o [49], OpenAI o4-mini [50], Claud 3.5 Sonnet [51], and Claud 3.7 Sonnet [51].

Implementation Details. For models with native video input support, such as Qwen-VL and Gemini, videos were processed directly without additional pre-processing. For models lacking native video input capabilities (e.g., GPT-4o), frames were uniformly extracted from the video along with corresponding timestamp annotations, and multi-image input was utilized for evaluation. To ensure a fair comparison, all models were deployed following their official guidelines and using the officially released checkpoints. During inference, models were required to first generate a reasoning process before providing the final answer. Specifically, the models were instructed to produce a step-by-step solution to the given question. For further details regarding model implementation and evaluation prompts, please refer to Appendix A and B.

### 4.2 Main Results

Table 2 presents the performance of each models on Video-Holmes benchmark. Most models achieve an accuracy below 40%, with the best-performing model, Gemini-2.5-Pro, reaching an overall accuracy of 45%. The widely-used open-source models, Qwen2.5-VL (7B) achieves an overall accuracy of 27.8%, far worse than its performance on other video reasoning benchmarks. This performance gap suggests that the Video-Holmes benchmark introduces unique challenges that are particularly demanding for current MLLMs in video reasoning tasks.

Models trained with thinking strategies exhibit notable improvements over their vanilla version. For instance, Gemini-2.0-Flash-Thinking demonstrates a 12.5% performance gain compared to Gemini-2.0-Flash. This observation indicates that the Video-Holmes benchmark imposes substantial

Table 4: Reasoning process analysis results. Where VPE represents visual perception error, VOE represents visual omission error, RE represents reasoning error, TRAW represents think right answer wrong, TWAR represents think wrong answer right, and TRAR represents think right answer right.

Answer Wrong Answer Right

Model

VPE VOE RE TRAW TRAR TWAR Count Ratio Count Ratio Count Ratio Count Ratio Count Ratio Count Ratio

Open Source Models InternVL2.5-8B 66 0.06 331 0.28 740 0.63 32 0.03 332 0.79 89 0.21 InternVL3-8B 32 0.03 248 0.22 810 0.73 22 0.02 419 0.76 135 0.24 Qwen2.5-Omni-7B 26 0.02 398 0.36 643 0.59 29 0.03 236 0.82 53 0.18 Qwen2.5-VL-32B 44 0.04 430 0.39 610 0.56 6 0.01 363 0.78 103 0.22 Qwen2.5-VL-7B 15 0.02 209 0.27 544 0.70 5 0.01 457 0.87 67 0.13 SEED-Bench-R1 32 0.03 325 0.30 688 0.63 54 0.05 543 0.80 133 0.20 VideoChat-R1 33 0.03 306 0.26 844 0.71 12 0.01 494 0.85 88 0.15 Video-R1 46 0.04 415 0.36 681 0.59 15 0.01 504 0.83 101 0.17

Closed Source Models GPT-4o 17 0.02 250 0.24 745 0.73 13 0.01 720 0.94 43 0.06 o4-mini 36 0.03 354 0.29 840 0.68 11 0.01 513 0.94 34 0.06 Claud 3.5 Sonnet 18 0.02 318 0.31 695 0.67 11 0.01 687 0.92 59 0.08 Claud 3.7 Sonnet 9 0.01 470 0.44 584 0.54 10 0.01 612 0.86 103 0.14

- Gemini-1.5-Pro 75 0.07 250 0.24 692 0.67 18 0.02 637 0.85 114 0.15

- Gemini-2.5-Pro 31 0.03 210 0.22 718 0.75 4 0.01 725 0.88 98 0.12 Gemini-2.0-Flash 33 0.03 242 0.20 893 0.76 13 0.01 542 0.89 65 0.11 Gemini-2.0-Flash-Thinking 31 0.03 210 0.22 718 0.75 4 0.01 774 0.94 49 0.06

reasoning challenges and effectively distinguishes models’ reasoning abilities. In contrast, other benchmarks do not reflect this pattern, as illustrated in Table 3.

Model performance across seven reasoning tasks in Video-Holmes remains relatively even, with most models achieving accuracy below 40% for each task. This highlights that each task in Video-Holmes poses substantial challenges, requiring advanced reasoning capabilities from existing methods.

### 4.3 Analytical Study

Reasoning Process Analysis. We analyze the factors contributing to the model’s answers by comparing its reasoning process with human-annotated descriptions and answer explanations. Specifically, we categorize the main causes of incorrect answers into the following four types:

- • Visual Perception Error (VPE): The model extracts incorrect visual information for analysis, leading to an incorrect answer.
- • Visual Omission Error (VOE): The model omits critical visual information (i.e., key objects or events), resulting in an incorrect answer.
- • Reasoning Error (RE): The model makes errors during the reasoning process, such as misinterpreting or incorrectly associating multiple visual clues.
- • Think Right Answer Wrong (TRAW): The model’s reasoning is largely aligned with the ground-truth explanation, but it selects an incorrect option when providing the final answer.

For correctly answered questions, we define the following two categories:

- • Think Wrong Answer Right (TWAR): The model’s reasoning process deviates significantly from the ground-truth explanation, yet it arrives at the correct answer.
- • Think Right Answer Right (TRAR): The model’s reasoning process is largely aligned with the ground-truth explanation and produces answers consistent with its reasoning.

We provide the human annotations, questions, the model’s reasoning outputs (if validated), and the type definitions as inputs to DeepSeek-R1 [3], prompting2 it to perform the analysis.

The results in Table 4 and Figure 4 show that both open-source and closed-source models generally demonstrate the ability to accurately extract visual information and provide answers consistent with their reasoning processes. Approximately 35% of errors are attributed to the omission of critical visual

2Detailed in Appendix B

00:05-00:30

00:45-01:04

01:25-01:57

02:10-02:40

Q: What is the core theme of this video? (CTI)

|[Figure 138]<br><br>[Figure 139]|
|---|

|[Figure 140]<br><br>[Figure 141]|
|---|

|[Figure 142]<br><br>[Figure 143]|
|---|

|[Figure 144]<br><br>[Figure 145]|
|---|

- A. Crime will eventually be punished
- B. The great power of maternal love
- C. The crisis of trust between strangers
- D. Morality overcome greed
- E. Social impact of surveillance equipment
- F. The harmfulness of impulsive behavior

The man returns to the store, where the elderly woman is in the middle of a robbery. She exchanges the stolen goods with him for her car keys and the man returns the items to

The man driving the car, followed by a shot

The man enters the

The man encounters an elderly woman. He casually takes her car

store, looks around,

revealing a baby inside.

and touches the gun in his pocket, suggesting he intends to commit a robbery.

Based on previous clues, the baby likely belongs to the elderly woman. Then the man decides to turn back.

keys, suggesting he

decides to steal her car.

the shop owner.

[Figure 146]

[Figure 147]

VPE

Model Response Thinking process analysis

VideoChat-R1

The video depicts a series of events that revolve around a man who appears to be in a state of distress or urgency. He is seen in a convenience store, driving a car, and interacting with a gas station attendant. The man's actions suggest a sense of urgency and possibly a need to leave quickly. The presence of a tattooed man and a woman in the video adds to the complexity of the situation, but the overall tone of the video seems to be one of tension and potential conflict. The core theme of the film, therefore, seems to be the crisis of trust between strangers, as the man's actions and the presence of the tattooed man and the woman suggest a potential conflict or a situation where trust is being tested. So the answer is B

[Figure 148]

[Figure 149]

The process of model thinking was disrupted by incorrectly extracted visual information "interact with gas station attendant" (which was not

[Figure 150]

part of the scenario), leading to a mistaken judgment that the theme was a crisis of trust.

[Figure 151]

[Figure 152]

Qwen-VL-2.5-7B

###### VOE

Model Response Thinking process analysis

The video depicts a series of events that revolve around a man who appears to be in a state of distress or urgency. He is seen in a convenience store, driving a car, and interacting with another person in a car. The man's actions, such as driving quickly and the presence of another person in the car, imply a sense of urgency and potential danger. The options provided suggest different themes, but the most fitting one based on the visual cues and the actions depicted is that crime will eventually be punished. This is because the video seems to be setting up a scenario where the man's actions could lead to consequences, which is a typical narrative structure in crime films. So the answer is A.

[Figure 153]

[Figure 154]

The model failed to extract the key visual information of “the baby in the back seat” and the moral awakening it triggered in the man in black, as well as the old woman’s act of exchanging stolen goods for the baby. These elements are central to the theme of “morality overcoming greed,” but the model’s reasoning did not mention them, leading to an incorrect answer.

[Figure 155]

[Figure 156]

[Figure 157]

###### RE

Model Response Thinking process analysis

Video-R1

The video begins with a man in a convenience store, highlighting a setting for transactions and interactions among strangers. His actions suggest a conflict or misunderstanding, testing trust between strangers. The scene shifts to a car driving through a forest. The presence of a baby and a possible chase scene underscores themes of pursuit and confrontation, indicating that the film's core revolves around interactions and conflicts between strangers, especially where trust is fragile. Now, let's consider the options: B: The crisis of trust between strangers. - This aligns well with the setting and the actions of the characters. D: Morality and kinship overcome greed. - There is no clear indication of kinship or morality overcoming greed in the video. After analyzing the video and the options, the most fitting theme is B: the crisis of trust between strangers.

[Figure 158]

[Figure 159]

The model correctly identified some visual elements but failed to establish the logical connection between the baby and moral awakening, or the elderly woman's failure and family affection restraining greed. It completely overlooked the key plot point (the baby prompting the man in black to return the stolen items) and mistakenly concluded the theme as a crisis of trust.

[Figure 160]

[Figure 161]

[Figure 162]

###### TWAR

Model Response Thinking process analysis

Intern-VL3-8B

Let's figure this out step by step. The video shows a man with his kid in a shopping bag, which seems to be a sign of kidnapping. The store owner calls the police and follows the man. It feels like a story of a thief taking someone else's's kid, leading to the police trying to catch them. Now, looking at the options: A. Crime punished - There’s mention od crime, but it's not really the heart of the story. How the situation is resolved is more important… F. Morality/kinship over greed - This could fit, as the person stealing is risking being caught for potential gain, and the mom and police are trying to restore what's right, which involves moral values about family. So, putting it all together, it seems the core of the story revolves around the tension and conflict arising from the theft, followed by the actions taken by the mom to rescue her kid. This points towards themes of morality. Therefore, my answer is D: Morality and kinship overcome greed.

[Figure 163]

[Figure 164]

The model exhibits significant errors in comprehension during reasoning, such as incorrectly inferring a "child kidnapping" and "mother rescuing child" scenario. While the model reasonably associates the presence of the infant with familial bonds, its core reasoning path deviates substantially from the actual storyline. Nonetheless, the model's final conclusion aligns with the correct answer.

[Figure 165]

Figure 4: Example of model reasoning processes on Video-Holmes. VideoChat-R1 misinterprets visual information, incorrectly perceiving a tattooed man, Qwen2.5-VL overlooks critical visual clues (the baby), Video-R1 fails to establish logical connections between the visual clues, and Intern-VL38B guesses the right option with a wrong reasoning process.

information, while a larger proportion (around 60%) stems from challenges in logical comprehension of multiple visual clues (Reasoning Errors).

For correctly answered questions, the proportion of responses based on valid reasoning (TRAR) exceeds 80% across most models. This highlights the difficulty of the Video-Holmes benchmark, where models struggle to infer correct answers through inconsistent reasoning.

Number of Input Frames. We analyze the performance of several models using different numbers of input frames. The results in Table 5 (a) indicate that increasing the number of input frames generally improves model performance, but does not lead to substantial gains. This observation indicates that in most cases, the visual information provided is sufficient, and the key challenge lies in the model’s ability to integrate and interpret visual clues effectively.

Table 5: Analyze experiment results on Video-Holmes. HA stands for human annotation; FLC stands for frame-level caption; VLC stands for video-level caption.

(a) Number of input frames

Model Frames Acc

Qwen2.5-VL-7B 64 30.2 (+2.4) Qwen2.5-VL-7B 80 33.0 (+5.2) Video-R1 64 37.4 (+1.2) Video-R1 80 38.5 (+2.0)

GPT-4o 40 43.3 (+1.3) GPT-4o 50 44.6 (+2.6)

(b) Audio input

Model Audio SR Overall

Qwen2.5-Omni-7B ✗ 27.1 16.4 Qwen2.5-Omni-7B ✓ 38.4 24.4

Gemini-2.5-Pro ✗ 46.6 45.0 Gemini-2.5-Pro ✓ 54.8 51.3 Gemini-1.5-Pro ✗ 52.1 41.2 Gemini-1.5-Pro ✓ 59.6 45.7

(c) Reasoning or not

Model Frames Reasoning Acc Qwen2.5-VL-7B 32 ✓ 27.8 Qwen2.5-VL-7B 32 ✗ 29.4 Video-R1 32 ✓ 36.5 Video-R1 32 ✗ 28.2 Gemini-2.0-Flash - ✓ 30.6 Gemini-2.0-Flash - ✗ 28.5

(d) Text-only input

Model Input Acc DeepSeek-R1 FLC 31.2 DeepSeek-R1 VLC 64.6 DeepSeek-R1 HA 92.0 OpenAI o3 FLC 25.4 OpenAI o3 VLC 61.3 OpenAI o3 HA 89.7

Audio Input. We evaluate the performance of several models with audio input as an additional modality. Table 5 (b) demonstrates that integrating audio input enhances model performance, especially in social reasoning tasks where conversational cues offer essential insights into interpersonal dynamics. These results underscore the importance of audio information in multimodal reasoning.

Reasoning or Not. We conduct experiments where models directly generate answers without using CoT prompts. The results in Table 5 (c) demonstrate that for stronger closed-source models, CoT prompting leads to higher accuracy compared to directly answering. In contrast, weaker open-source models exhibit the opposite trend. This suggests that the effectiveness of reasoning is contingent on the model’s overall capability—only models with sufficiently strong reasoning abilities can fully benefit from CoT prompts. Conversely, weaker models may amplify errors during CoT reasoning.

Text-only Input. We conduct experiments using three types of text-only inputs for advanced reasoning models [3, 5]: (1) human-annotated movie plots and key clues (excluding reasoning conclusions), (2) frame-level captions generated by Qwen2.5-VL (sampled at one frame per second), and (3) video-level captions generated by Gemini-2.5-pro. Table 5 (d) shows that models achieve around 90% accuracy with human annotations, while performance drops significantly with frame-level and video-level captions. This can be attributed to human annotations, which provide logical connections between critical visual clues in the video. In contrast, video-level and frame-level captions may overlook key visual details or offer incorrect logical interpretations, leading to reasoning errors. These findings underscore the challenge posed by Video-Holmes in requiring models to locate and and capture logical relationships within multiple visual clues.

## 5 Conclusion and Discussion

In this work, we propose Video-Holmes, a benchmark designed to evaluate the complex video reasoning capabilities of MLLMs. Video-Holmes consists of 1,837 questions derived from 270 manually annotated suspense short films, which spans seven carefully designed tasks that require models to actively locate and connect multiple relevant visual clues scattered across different video segments. We conduct a detailed analysis of model reasoning processes, examining the factors that lead to both correct and incorrect answers. Our comprehensive evaluation of state-of-the-art MLLMs reveals that, while these models generally excel at visual perception, they encounter substantial difficulties with integrating information and often miss critical clues. We aim that Video-Holmes can serve as a “Holmes-test” for multimodal reasoning, motivating models to reason more like humans and emphasizing the ongoing challenges in this field.

Appendix

- A Model Implementation Details

QwenVL: We utilize the official checkpoints for different QwenVL models: Qwen/Qwen2.5-VL-7B-Instruct for QwenVL-2.5-7B, Qwen/Qwen2.5-VL-7B-Instruct for QwenVL-2.5-7B, Qwen/Qwen2.5-VL-32B-Instruct for QwenVL-2.5-32B, Video-R1/Video-R1-7B for Video-R1, and OpenGVLab/VideoChat-R1-7B for VideoChatR1. The decoding configuration follows the settings provided in the official QwenVL-2.5 demo, with top-p set to 0.001 and temperature set to 0.01. During inference, we increase the frame resolution to 256 × 28 × 28 pixels to enhance visual fidelity.

InternVL: We utilize the official checkpoints for various InternVL models: OpenGVLab/InternVL3-8B for InternVL3-8B and OpenGVLab/InternVL2-5-8B for InternVL2.58B. The input image size is resized to 448 × 448 according to the official configuration.

API Models: We access the Gemini, GPT, and Claude model series via the official APIs provided by Google, OpenAI, and Anthropic. Specifically, for the GPT series, we use the official functions to retrieve image URLs and set the "detail" parameter to "low" as recommended.

- B Prompt Template

Question Generation Prompt

You are a professional logician and detective story enthusiast. I will provide you with the plot information of a short detective film. Your task is to create some reasoning multiple-choice questions for viewers who have watched this short detective film to test whether they truly understood it. Requirements: I will provide the following information:

- (1) Segmental plot description: chronological plot description
- (2) Key character relationships and cause positioning: the relationships between key characters in the film and the information that can be used to infer their relationships.
- (3) Reasoning shots: key hint shots in the plot, plot twists, and climaxes.
- (4) Supernatural elements: supernatural elements and rules appearing in the video.
- (5) Main idea of the video: the possible main idea the video wants to express.

You need to create at least one multiple-choice question from each of the following seven types of reasoning (according to the specific requirements):

- Type 1: Social Reasoning (SR) Task description: infer the implicit social relationship network between characters through clothing style matching, group interaction topology analysis, including identity association across time spans (such as the same character in youth and old age). Number of questions: Create one question for each combination based on the given key character relationships. If the [segmental plot description] provided includes the same character across time spans, create a question based on this. Example 1: What is the relationship between the man at the beginning of the video and the skeleton at the end? (Answer: A. The same person)
- Type 2: Intention and Motive Chaining (IMC) Task description: infer the underlying psychological state and predict non-explicit behavioral intentions by observing characters’ actions, expressions, or environmental clues. The question format must strictly follow the example: [Character] at [time] performed [very brief, non-key detail action description] with what intention/psychological state?

Question Generation Prompt (Continue)

Number of questions: 1-2 questions Example 1: What is the most likely psychological state of the man in black smoking at 00:03 in the video? (Answer: C. Anxiety about preparing for a crime)

- Type 3: Temporal Causal Inference (TCI) Task description: infer causal mechanisms between non-continuous temporal events through camera language and multimodal clues. Number of questions: 1-2 questions Example 1: Why did the man in the shirt die? (Answer: F. Overuse of superpowers)
- Type 4: Timeline Analysis (TA) Task description: reorganize video clips in chronological order. You need to provide 5 key events and ask to restore the timeline. Number of questions: 1 question Example 1: 1. The man in black wakes up 2. The man in black is attacked 3. The man in black goes out 4. The man in black is sleeping 5. The man in black meets the woman
- Type 5: Multimodal Hint Reasoning (MHR) Task description: analyze the visual and auditory hints set by the director: camera movement semantics, object position changes, sound and picture metaphors. Ask about specific camera transitions, the appearance, movement, or change of objects. Number of questions: Create one question for each given hint shot based on the given [reasoning conclusion]. Example 1: The correct interpretation of the dancing scene between the man and woman in the video is (Answer: A. The man’s own fantasy)
- Type 6: Physical Anomaly Reasoning (PAR) Task description: identify scenes in the video that do not conform to reality logic (such as magic, sci-fi elements). The first question asks about the rules of supernatural elements, and the second question asks about their implied meaning. Number of questions: (2n questions) Create questions based on the given supernatural elements. If none are provided, skip this type.
- Type 7: Core Theme Inference (CTI) Task description: infer the core theme or deep meaning through the plot, dialogue, or symbols in the video. (The given main idea of the video is a subjective conclusion and may not be entirely correct)

The questions you propose need to be understood and carefully reasoned to answer. Each question needs to provide 6 options, ensuring the correct option is objectively unique (distractor options cannot express a similar meaning to the correct option, they must be clearly distinguished), and the distractor options should overlap with the correct option as much as possible (e.g., A. Because the man forgot to bring his phone B. Because the man forgot to bring his tablet C. Because the man forgot to bring his wallet). You need to provide the question type, question, answer, and explanation. Key point: The question stem cannot reveal character psychology, factual information, hint clues, and hint conclusions, viewers must find them themselves. For example: Bad question: "What does the skeleton at the end of the video imply?" - Good question: "What does the skeleton at the end of the video imply?"; Bad question: "What is the psychological state of the man when he imagines the future in the lighting change scene?" - Good question: "What is the psychological state of the man in the lighting change scene?"

Answer format example: For each question, the 1st line: [Type]: CTI The 2nd line: [Question]: Your question; The 3rd-8th lines: A-F options; The 9th line: [Answer]: F; The 10th line: [Explanation]: Your explanation.

Model Evaluation Prompt (Thinking)

Based on the given video, reason and answer the single-choice question. Provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. The question is: question. The options are: options. Your answer:

Model Evaluation Prompt (Without Thinking)

Based on the given video, answer the single-choice question. Give your final answer choise between the <answer> and </answer> tags. The question is: question. The options are: options. Your answer:

Reasoning Process Analysis Prompt (Thinking Wrong)

You are an expert in logic. Your task is to analyze the cause of errors in a multimodal large model’s responses to video reasoning tasks. The requirements are as follows:

I will provide you with a plot description of a short reasoning film, a question, options, the correct answer, and an explanation of the correct answer.Then I will provide you with the [thought process] of a multimodal large model and its incorrect answer. Your task is to compare the multimodal large model’s thought process with the correct explanation and the film plot. Determine the most critical cause of the error from the following:

- (1) VPE (Visual Perception Error): The model extracted incorrect visual information for analysis, leading to the wrong answer. (For example, the video shows a man robbing with a knife, but the model perceives it as a man robbing with a gun.)
- (2) VOE (Visual Omission Error): The model failed to extract key visual information (such as key objects and events), leading to the wrong answer.
- (3) RE (Reasoning Error): The model made an error in the reasoning process, misinterpreting the implications of the visual information and incorrectly judging the relationships between multiple pieces of visual information.
- (4) TRAW (Think Right Answer Wrong): The model’s thought process is generally consistent with the answer explanation, but it chose the wrong option when answering. Your response format: Please provide the error abbreviation within <Type></Type>. Then give your brief judgment reason within <Reason></Reason>.

Reasoning Process Analysis Prompt (Thinking Right)

You are an expert in logic. Your task is to analyze the thought process of a multimodal large model in responding to video reasoning tasks. The requirements are as follows:

I will provide you with a [plot description] of a short reasoning film, a question, options, the correct answer, and an explanation of the correct answer. Then I will provide you with the [model’s thought process] and its correct answer. Your task is to compare the multimodal large model’s thought process with the correct explanation and the film plot. Determine which type the response belongs to:

- (1) TWAR (Think Wrong Answer Right): The model’s thought process shows a significant deviation from the answer explanation, using incorrect information to arrive at the correct conclusion.
- (2) TRAR (Think Right Answer Right): The model’s thought process is generally consistent with the answer explanation (allowing for minor deviations). Your response format: Please provide the response type abbreviation within <Type></Type>. Then give your brief judgment reason within <Reason></Reason>.

## C Key Statistics of Video-Holmes Table 6: Key Statistics of Video-Holmes.

The key statistics of Video-Holmes are shown in Table 6. To ensure diversity, we include nine subkeywords (Anim, Comic, Detective, Future, Horror, Social, Supernatural, Thriller) when searching for suspense short films. The distribution of the nine specifically designed tasks is relatively balanced, with a higher proportion of MHR tasks because a single video often contains more than one reasoning shot annotated by humans. PAR tasks are absent in videos without supernatural phenomena, as such questions are not applicable.

Statistic Number Total Suspense Short Films 270

- - Anime 23 (8.5%)
- - Comic 32 (11.9%)
- - Detective 31 (11.5%)
- - Future 7 (2.6%)
- - Horror 88 (32.6%)
- - Social 66 (24.4%)
- - Supernatural 36 (13.3%)
- - Thriller 23 (8.5%)

Total Questions 1,837

- - SR 292 (15.6%)
- - IMC 276 (15.0%)
- - TCI 273 (14.9%)
- - TA 200 (10.9%)
- - MHR 332 (18.1%)
- - PAR 194 (10.4%)
- - CTI 270 (14.7%)

## D Examples of Video-Holmes

Figures 5 to 12 illustrate an example of VideoHolmes. Specifically, Figure 5 presents the human annotation results, while Figures 6 to 12 display the questions and explanations generated by DeepSeek, along with the models’ answers and reasoning process analysis.

Average Question Word Count 17.4 Average Explanation Word Count 31.3

## E Broader Impact

The development and release of the Video-Holmes benchmark have the potential to impact the field of complex video reasoning by providing a rigorous and comprehensive evaluation benchmark. However, it is important to acknowledge the potential ethical considerations associated with the use of this benchmark. The video content used in Video-Holmes, derived from suspense short films, may contain elements of horror or thriller genres, which could be distressing or inappropriate for certain audiences. Researchers and developers utilizing this benchmark should be mindful of the nature of the content and ensure that it is used responsibly, with appropriate content warnings and considerations for the intended audience.

Suspense Short Films

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Human Annotation

{

"SegmentDescription": [ {

"TimeRange": "00:00-00:22", "Description": "A woman dressed as a clown performs in front of a crowd, pretending to hold a

pistol with her hand and breaking a set-up glass bottle in the air, earning everyone's applause." },

{

"TimeRange": "00:22-01:50", "Description": "The clown conjures up a green apple and chooses David to cooperate. He places

the green apple on top of David's head, covers his eyes, and pretends to shoot through the apple with a phantom gun. The man in green mocks David. Despite being very scared, the performance is successfully completed, and David is also very surprised."

}, {

"TimeRange": "01:50-03:55", "Description": "The onlookers, except for David and the man in green, tipped one by one. The

clown looked at the two men, signaling for a tip. The man in green pretended to give money but actually didn't put it in the bucket, and even mocked the clown's performance as terrible, causing the clown's facial expression to turn angry."

},

{

"TimeRange": "03:55-04:16", "Description": "After killing two people, the clown rummaged through the green man's pockets,

found the money and showed it to the green man's corpse with a laugh, then turned around and left." }

], "KeyRelationships": [

{

"Combination": "The clown and David", "Reason": "One is performing, and one is watching."

}, {

"Combination": "David and the man in green", "Reason": "The two people watched the performance together, talking and laughing, and the man

in green even said David's name."

}

], “Reasoning Shots": [

{

"Time": "00:28", "Clue": "The clown reaches out his palm to the camera."

}, {

"Time": "02:25", "Clue": "The clown swiped his face from top to bottom with both hands, and his expression

changed from a smile to anger."

} ],

"Supernatural Elements": {

"Phenomenon": "The clown can hold tools in the void and utilize their original functions", "Positive Impact": "Can perform acts like shooting bottles and apples", "Negative Impact": "Could become a tool for crime, such as using a void fishing rod to steal the Man

in Green's money or using a void gun to kill David and the Man in Green" }, "Core Theme": "As an audience, one should respect the performance of others, even if not paying,

avoid mocking. Performers should not forcefully demand money from the audience, and should not become angry or harm others if not paid." }

Figure 5: Example of human annotation.

"video ID": "ru0HUKY3nfM"

"Question ID": 1 "Question Type": "SR" "Question": "What is the relationship between the clown and David?" "Options": {

- "A": "siblings"
- "B": "colleague"
- "C": "stranger"
- "D": "enemy"
- "E": "couples"
- "F": "partners"

}, "Answer": "C" "Explanation": "The clown is a performer, David is an onlooker, and there are no other interactions between them to suggest a closer relationship, so they are strangers."

Model Answers and Reasoning Process Analysis

[Figure 173]

[Figure 174]

InternVL-3-8B

Model Response Thinking process analysis

###### TRAR

Based on the video content, the correct answer appears to be E: stranger. The clown is performing tricks for David and others at the station,

but there's no indication of any prior relationship between them. The interaction is casual and seems to be a form of entertainment or street

[Figure 175]

[Figure 176]

performance. There's no visible connection suggesting they are siblings, colleagues, enemies, couples, or partners.

[Figure 177]

The model's reasoning process is consistent with the core logic of the explanation, both indicating that the interaction between the clown and David is merely a temporary one between a performer and an observer, with no other relationship clues, demonstrating a correct reasoning path.

[Figure 178]

[Figure 179]

Model Response Thinking process analysis

Qwen-VL-2.5-7B

###### VPE

From the video, it appears that the clown and David are in a confrontational situation. The clown is holding a knife and pointing it at David, which suggests a hostile relationship. There is no indication that they are siblings, colleagues, couples, or partners. The clown's aggressive

[Figure 180]

behavior towards David implies that they are enemies.

[Figure 181]

The model mistakenly perceived the clown's action of 'pretending to shoot the apple with a hand gesture mimicking holding a gun' as 'pointing

[Figure 182]

a knife at David,' and based on this incorrect visual information, inferred a hostile relationship. In reality, there was no knife in the video; the key performance action was pretending to use a gun, thus it is a visual perception error.

[Figure 183]

[Figure 184]

###### RE

Model Response Thinking process analysis

Gemini-2.5-Pro

The user wants to know the relationship between the clown and David based on the video. I need to watch the video and identify David and the

clown, then analyze their interactions. They start as strangers, but by the end, the clown is clearly acting as an antagonist or enemy towards David (and his friend). 'Enemy' best describes the adversarial nature of their relationship by the end of the clip. 'Stranger' is only true for the beginning. The prompt asks for the relationship between them, and enemy implies a hostile relationship developed from the initial encounter as strangers. Therefore, the relationship that develops in the video is best described as antagonistic, making 'enemy' the most suitable choice among the options provided to characterize the *developed* relationship within the context of the scene.Based on the video. This establishes the mime as an adversary to David. While they started as strangers, the mime becomes David's enemy through hostile actions.

[Figure 185]

[Figure 186]

The model correctly identified the initial stranger relationship between the clown and David, but it mistakenly considered subsequent hostile as the basis for establishing an "enemy" relationship. Despite the conflict, there is no pre-existing connection between the two, and the question is

[Figure 187]

about the essential relationship rather than situational confrontation. Therefore, this constitutes a reasoning error (RE).

[Figure 188]

[Figure 189]

###### TWAR

Model Response Thinking process analysis

Video-R1

Let's break down the information provided in the video and analyze the relationship between the clown and David. 1. The clown is dressed in a costume that includes a striped shirt and a red bow tie, which is a common attire for clowns. This suggests that the clown is likely a performer or entertainer. 2. David is seen interacting with the clown, and there is a moment where the clown appears to be performing a magic trick or some sort of illusion. This interaction indicates that David is not a stranger to the clown, as he is willing to engage in a conversation and possibly

[Figure 190]

participate in the clown's performance. However, there is no indication that they are siblings, colleagues, or partners. Given these observations, the

most plausible relationship between the clown and David is that they are not strangers, but they are not necessarily close friends or family members. After careful consideration, the most reasonable answer is that the clown and David are strangers. Final answer: C: Stranger

[Figure 191]

During the model's thought process, it incorrectly assumed interactions such as "walking together" between the clown and based on this, inferred a certain level of familiarity between them. However, it still arrived at the correct answer, E. This represents a clear deviation between its reasoning logic and the analysis, constituting a case where the correct conclusion was reached using incorrect information.

[Figure 192]

Figure 6: Example of question, model answers and reasoning process analysis.

"Question Type": “IMC" "Question": "What is the clown's intention by sweeping his hands across his face at 02:25?" "Options": {

- "A": "Adjust makeup"
- "B": "Continue the performance process"
- "C": "Relieve fatigue",
- "D": "Decide to retaliate"
- "E": "attract the audience's attention"
- "F": "Express sadness"

} "Answer": "D" "Explanation": The action follows immediately after the man in green's mockery, with an expression turning to anger, suggesting that the clown decided to retaliate because he was provoked.

Model Answers and Reasoning Process Analysis

[Figure 193]

[Figure 194]

InternVL-3-8B

Model Response Thinking process analysis

###### RE

The act of sweeping his hands across his face implies a shift in look, as seen in professional make-up application where the makeup artist blends and smooths the products for a realistic effect. Additionally, the scene showing the man in green jacket pulling money out from his pocket might be a part of a performance where the clown is setting up a scenario, possibly for comedic or theatrical purposes, and changing in his expression aids in conveying different moods or reactions which are crucial to such performances. The act of sweeping across the face might also serve as a dramatic gesture to engage the audience or to enhance the narrative.

[Figure 195]

[Figure 196]

The model correctly identified the clown's face-sweeping gesture and the scene where the man in green takes out money, but it mistakenly interpreted the gesture as adjusting makeup. It overlooked the logical chain between the angry expression and subsequent retaliatory actions.

[Figure 197]

[Figure 198]

[Figure 199]

Qwen-VL-2.5-7B

Model Response Thinking process analysis

###### RE

At 02:25 in the video, the clown is seen sweeping his hands across his face. This action is typically associated with adjusting makeup, especially in a performance setting where a clown might need to touch up their appearance. The other options like continuing the performance process, relieving fatigue, deciding to retaliate, or attracting the audience's attention do not fit the context of the action shown. The clown's expression and the setting suggest that this is a performance, but the specific action of sweeping hands across the face is more indicative of adjusting makeup.

[Figure 200]

[Figure 201]

The model correctly identified the “face-sweeping” gesture but failed to accurately associate it with the key contextual event (the clown‘s expression turning angry after being mocked by the man in green). The analysis indicates that this gesture is a hint of the decision to retaliate, highlighting a misunderstanding in the reasoning phase regarding the causal relationship between visual information.

[Figure 202]

[Figure 203]

[Figure 204]

Model Response Thinking process analysis

###### TRAR

Gemini-2.5-Pro

The clown is performing on the street. At the beginning the clown breaks the glass and impresses the audience. Then clown does some trick with apple but the audience does not provide any tip. This annoys the clown, and he finally decides to take retaliatory action. At [02:25], the clown sweeps his hands across his face, likely signaling a shift from a lighthearted performance to a more aggressive one. The following actions are further confirmation of decision to retaliate. From the given options: Options A, B, C, E, and F appear inconsistent with the context.

[Figure 205]

[Figure 206]

The model correctly identified the connection between the clown's change in expression and the subsequent retaliatory actions, and it reasoned out the correct answer.

[Figure 207]

[Figure 208]

[Figure 209]

Model Response Thinking process analysis

###### VOE

Video-R1

Let's break down the context and the actions of the clown at 02:25 in the video. The clown is wearing a striped shirt and a red bow tie, and he is performing in a public space at night. The clown's actions, including sweeping his hands across his face, are likely part of a performance or a routine.\n\nFirstly, the clown's sweeping motion across his face is a common gesture in clown performances. It is often used to adjust makeup or to create a dramatic effect. This suggests that the clown might be adjusting his makeup to ensure it looks perfect for the …Given these

[Figure 210]

considerations, the most likely intention of the clown's action is to adjust his makeup and continue the performance process. The clown is

likely using the sweeping motion to ensure his makeup looks perfect and to maintain the flow of the performance.\n\nTherefore, the most appropriate answer is B: Continue the performance process.\Final answer: B

[Figure 211]

[Figure 212]

The model failed to correctly associate the causal relationship between the man in green's mockery and the clown's change in expression. The correct analysis indicates that the clown's expression turned angry after being provoked, leading to a decision to retaliate. However, the model mistakenly interpreted the face-sweeping gesture as part of a routine performance, overlooking the critical sequence and logical connection.

[Figure 213]

Figure 7: Example of question, model answers and reasoning process analysis.

"Question Type": "TCI" "Question": "What was the direct cause of death for the man in green?" "Options": {

- "A": "Steal the clown's money"
- "B": "be accidentally injured by David"
- "C": "Performance accident"
- "D": "Mock the clown"
- "E": "heart attack"
- "F": "The apple falls and hits the head."

}, "Answer": "D" "Explanation": "The man in green was killed by the clown using superpowers because he mocked the clown for not leaving a tip."

Model Answers and Reasoning Process Analysis

[Figure 214]

[Figure 215]

InternVL-3-8B

Model Response Thinking process analysis

###### RE

The video scene shows a clown performing an apple-balancing trick on two men. The man in green, upon getting mad, ends up being pushed and falls face-first onto the hard ground. This action leads to his unfortunate death. From the information provided, it can be inferred that the direct cause of death for the man in green was due to being accidentally injured by David during the clown's's performance. The other in green's seemingly losing his balance and hitting his head on the ground might be the immediate reason for his death. So, the most appropriate answer from the given options would be that it was an accidental injury caused by David during the clown's performance.

[Figure 216]

[Figure 217]

The model correctly identified the scene of the man in green's death (head hitting the ground) but incorrectly established the causal relationship. The narrative clearly shows that the clown, provoked by mockery, intentionally used superpowers to kill.

[Figure 218]

[Figure 219]

[Figure 220]

Qwen-VL-2.5-7B

Model Response Thinking process analysis

###### VOE

Let's analyze the sequence of events in the video to determine the direct cause of death for the man in green. The video shows a man in a striped shirt and a clown interacting with a man in a green jacket. However, there is no explicit information in the video about the cause of death. The options provided do not directly correspond to the events in the video, but we can infer that the man in the green jacket might have died due to a performance accident, as the clown is seen performing a trick with a green apple, which could have been the cause of the accident.\n\nFinal answer: C: Performance accident.

[Figure 221]

[Figure 222]

The model failed to correctly extract the crucial visual information that "the man in green mocked the clown and did not tip." The correct

[Figure 223]

analysis clearly indicates that the cause of death was the mockery, which angered the clown. However, the model focused on the green apple prop

used in the performance, neglecting the subsequent causal relationship where the clown, provoked by the mockery, intentionally killed.

[Figure 224]

[Figure 225]

Model Response Thinking process analysis

###### RE

Gemini-2.5-Pro

The video shows a mime putting on a performance for a group of people at a train station. The mime puts an apple on a man's head and pretends to shoot the apple off the man's head with a finger gun. The man in green offers the mime money but takes it back. This makes the mime angry, so the mime shoots the green jacket man. After being shot with an imaginary gun, the man in green falls to the ground and dies.\nThe option that seems most likely is to Steal the clown's money, the only thing that is true from this story, the man takes back the money and he is shot by the mime.

[Figure 226]

[Figure 227]

The model correctly identified the visual information that the man in green "pretended to give money but took it back," but incorrectly attributed this action to "stealing money" . In reality, according to the narrative, the direct cause of the clown's killing was the man in green "mocking the performance" The model erroneously established a causal relationship between "taking back money" and "stealing money".

[Figure 228]

[Figure 229]

[Figure 230]

Model Response Thinking process analysis

###### RE

Video-R1

Let's break down the sequence of events and the context provided in the video to determine the direct cause of death for the man in green.\n\n1. The man in green is seen interacting with the clown, who is dressed in a striped shirt and a red bow tie. The clown appears to be performing some sort of magic trick or actTherefore, the direct cause of death for the man in green is the apple falling and hitting his head.\n\nLet's verify the option Performance accident - This is a possibility, but it is not as specific as the apple hitting the head.\nD: heart attack - There is no indication of a heart

[Figure 231]

###### attack in the video.\nE: The apple falls and hits the head - This is the most specific and accurate cause of death.\nF: Mock the clown - This is

not related to the cause of death.\n\nThe most accurate answer is E.\n\nFinal answer: E\n\nLet's double-check the answer one last time to ensure it is correct.

[Figure 232]

[Figure 233]

The model incorrectly associated the successful piercing of the apple during the performance with the man in green's death, mistakenly believing that the apple's fall caused the death (performance accident). In reality, the man in green was killed by the clown using superpowers as a retaliatory act because he mocked the clown and did not tip. The death occurred as part of the clown's revenge after the performance.

[Figure 234]

Figure 8: Example of question, model answers and reasoning process analysis.

"video ID": "ru0HUKY3nfM" "Question ID": 4 "Question Type": "TA" "Question": "Which of the following events are arranged in the correct chronological order?\n① The clown shattered the glass bottle with a void gun.\n② The clown conjured a green apple and put it on David's head.\n

③ The man in green pretended to tip but did not put the money in the bucket.\n④ The clown's expression turned angry.\n⑤ The clown killed two people and took the man in green's money." "Options": {

- "A": "①→②→③→④→⑤"
- "B": "②→①→④→③→⑤"
- "C": "③→①→②→⑤→④"
- "D": "⑤→④→①→③→②"
- "E": "②→③→④→①→⑤"
- "F": "①→③→②→④→⑤"

}, "Answer": "A" "Explanation": "In the order of the plot: break the bottle first → turn into an apple → the man in green mocks

→ the clown is angry → kill someone to take money."

Model Answers and Reasoning Process Analysis

[Figure 235]

[Figure 236]

InternVL-3-8B

Model Response Thinking process analysis

###### RE

Based on the events described, the correct chronological order should start with the clown conjuring a green apple and putting it on David's head (②), followed by the man in green pretends to tip but did not put the money in the bucket (③), then the clown's expression turned angry (④), the clown shattered the glass bottle with a void gun (①), and finally, the clown killed two people and took the man in green money (⑤). So the correct answer is D: ②→③→④→①→⑤.

[Figure 237]

[Figure 238]

[Figure 239]

The model correctly identified the visual information corresponding to all events.However, it made an error in reasoning the sequence of events by placing event ① (breaking the bottle) after event ④ (the clown getting angry).

[Figure 240]

[Figure 241]

Qwen-VL-2.5-7B

Model Response Thinking process analysis

###### TRAR

1. Event ②:** After the bottle trick and applause, the clown mimes conjuring an apple (0:25-0:32), shows it (0:33), approaches the man with

glasses (David) (0:36-0:50), and places it on his head (0:51). This happens after event ①.\n * **Event ③:** After the apple-on-the-head trick

(which involves miming shooting the apple, 0:53-1:39), the clown asks for tips (1:55-2:02). The man in the green jacket (David's friend) takes out money, approaches the tip can, fakes putting it in, and pockets it, saying \"Psych\" (2:08-2:16). This happens after event ②.\n * **Event ④:** Immediately after the man in green pockets the money and says \"Learn better tricks\" (2:17-2:19), the camera focuses on the clown's face.

[Figure 242]

[Figure 243]

The thought process of the model provides a detailed analysis of the timestamps for each event in the video. Its reasoning sequence is completely

[Figure 244]

consistent with the correct order in the analysis (①→②→③→④→⑤). The logical chain is clear and without contradictions, representing a

situation where correct thinking leads to the correct answer.

[Figure 245]

[Figure 246]

Model Response Thinking process analysis

###### TRAR

Gemini-2.5-Pro

Okay, let's break down the order of events based on the video:\n\n1. **① The clown shattered the glass bottle with a void gun:** This happens early in the video as a performance to attract the crowd.\n2. **② The clown conjured a green apple and put it on David's head:** Immediately after shattering the bottle, the clown places a green apple on David's head and pretends to shoot it.\n3. **③ The man in green pretended to tip but did not put the money in the bucket:** After the apple trick, the man in green makes a show of tipping but then doesn't put the money in.\n4.

[Figure 247]

**④ The clown's expression turned angry:** After the man in green declines the tip, the clown appears offended.\n5. **⑤ The clown killed two people and took the man in green's money:** The clown then uses a void-gun to kill the man who declined to tip her and his friend for the money,

[Figure 248]

right after they mocked her.\n\nSo, the correct order is 1 -> 2 -> 3 -> 4 -> 5.

The thought process of the model provides a detailed analysis of the timestamps for each event in the video. Its reasoning sequence is completely consistent with the correct order in the analysis (①→②→③→④→⑤). The logical chain is clear and without contradictions, representing a situation where correct thinking leads to the correct answer.

[Figure 249]

[Figure 250]

[Figure 251]

Model Response Thinking process analysis

###### VOE

Video-R1

Let's break down the sequence of events in the video and match them to the given options. \n\n1. The clown shattered the glass bottle with a void gun. This event is clearly shown in the video, and it's a pivotal moment where the clown demonstrates his powers.\n2. The clown conjured a green apple and put it on David's head. This event follows the bottle-shattering scene, as the clown uses his powers to entertain the audience.\n3. The man in green pretended to tip but did not put the money in the bucket. This event happens after the clown's performance, as the man in green tries

[Figure 252]

to interact with the clown.\n4. The clown's expression turned angry. This event occurs after the man in green's attempt to tip, as the clown

becomes upset.\n5. The clown killed two people and took the man in green's money. This event is not shown in the video and is not part of the sequence of events.

[Figure 253]

[Figure 254]

During its thought process, the model mistakenly believed that event 5 was not shown in the video. However, according to the plot description, the actions of the clown killing two people and searching the pocket of the man in green to take money are clearly depicted between 03:55-04:16. The model failed to correctly extract this crucial visual information, resulting in the omission of event ⑤ in the sequence of events.

[Figure 255]

Figure 9: Example of question, model answers and reasoning process analysis.

"Question Type": "MHR" "Question": "What is the implication of the clown reaching out his palm to the camera at 00:28?" "Options": {

- "A": "Ask the audience for tips."
- "B": "Get ready to conjure up items."
- "C": "Expression of the end of the performance"
- "D": "Block the light of the lens"
- "E": "Imitate the action of shooting."
- "F": "Adjust the position of the gloves"

}, "Answer": "B" "Explanation": "After stretching out her palm, she conjured a green apple, implying that this action is the precursor to 'conjuring an item'."

Model Answers and Reasoning Process Analysis

[Figure 256]

[Figure 257]

InternVL-3-8B

Model Response Thinking process analysis

###### VOE

The frame where the clown reaches out his palm directly towards the camera at 00:28 is significant because it visually suggests that the clown is asking for something from the audience, likely tips or donations. This kind of gesture is commonly associated with street performers or entertainers who who perform to earn money from viewers. The extended palm, typically a gesture for receiving payment, implies that the clown has completed a performance and is now requesting compensation from the,the most accurate interpretation of the clown's gesture is that he is asking the audience for tips.

[Figure 258]

[Figure 259]

The model failed to correctly identify the causal relationship between the reaching gesture at 00:28 and the subsequent appearance of the green apple. However, the model mistakenly associated its attention with the later tipping scene at 03:55

[Figure 260]

[Figure 261]

[Figure 262]

Qwen-VL-2.5-7B

Model Response Thinking process analysis

###### TRAR

At 00:28 in the video, the clown reaches out his palm to the camera. This gesture is often associated with magic tricks where a magician might be about to perform a trick involving the appearance of an object. The clown's gesture is not a typical expression of the end of a performance, nor is it a request for tips or an attempt to block the light. It is also not a simple imitation of shooting or an adjustment of the gloves. The most plausible implication is that the clown is preparing to conjure up an item, as this is a common element in clown performances and magic shows.\n\nTherefore, the best answer is B: Get ready to conjure up items.

[Figure 263]

[Figure 264]

The thought process of the model is largely consistent with the answer analysis. Both indicate that the clown's reaching gesture is a preparatory action for the subsequent appearance of the green apple, which is a common precursor in magic performances for summoning objects. The model correctly eliminated other options (such as asking for a tip or ending the performance).

[Figure 265]

[Figure 266]

[Figure 267]

Model Response Thinking process analysis

###### TRAR

Gemini-2.5-Pro

The clown is doing a magic trick like action throughout the video. At 00:28, the clown's palm is empty. When he continues reaching out, the apple is magically conjured on his hand. Therefore, this action is part of the trick of conjuring something from nothing.\n\nOther options analysis.\n\nOption A: While the clown might be seeking tips eventually, that's not the immediate implication of the gesture at that specific moment.\nOption C: The performance is not over yet, as this conjuring trick leads to other actions later on, such as placing the apple on top of a guy's head and \"shooting\" the apple.\nOption D: No indication that the action has to do with lighting.\nOption E: Action of shooting happen later, not in that time stamp.\nOption F: There is no indication that the clown was trying to adjust gloves at the time frame, he was reaching out to the camera.

[Figure 268]

[Figure 269]

The thought process of the model is largely consistent with the answer analysis. Both indicate that the clown's reaching gesture is a preparatory action for the subsequent appearance of the green apple, which is a common precursor in magic performances for summoning objects. The model correctly eliminated other options (such as asking for a tip or ending the performance).

[Figure 270]

[Figure 271]

[Figure 272]

Model Response Thinking process analysis

###### TRAR

Video-R1

Let's break down the context and the implications of the clown reaching out his palm to the camera at 00:28. \n\nFirstly, the clown is dressed in a costume that includes a striped shirt and a red bow tie, which is typical attire for a clown. The presence of a clown in a nighttime setting with other people suggests that this could be part of a performance or a trick.\n\nNow, let's consider the options:\n\nA: Ask the audience for tips. - This seems unlikely because the clown is not directly addressing the audience in a way that suggests seeking advice or tips.\n\nB: Get ready to conjure

[Figure 273]

###### up items. - This is a plausible option. Clowns often use their hands to create the illusion of magic or conjuring. The palm gesture could be a

part of a trick where the clown is about to reveal or create something. After considering all the options, the most plausible implication of the clown reaching out his palm to the camera is that he is getting ready to conjure up items.\n\nTherefore, the best answer is B.

[Figure 274]

[Figure 275]

The model's thought process correctly identified that the clown's reaching gesture was a preparation for producing an object, aligning with the key point in the analysis that "the green apple appears after the reaching gesture." Although the model mentioned the subsequent tipping scene when excluding option A, the overall analytical logic is consistent with the correct answer analysis, with no significant deviation.

[Figure 276]

Figure 10: Example of question, model answers and reasoning process analysis.

"Question Type": "PAR" "Question": "What are the rules of the clown's superpowers?" "Options": {

- "A": "Can create life forms out of thin air."
- "B": "Void tools can exert practical functions."
- "C": "Manipulate other people's thoughts"
- "D": "Can teleport objects instantly"
- "E": "Predict the future",
- "F": "Can change the color of an object"

}, "Answer": "B" "Explanation": "He can shatter bottles with a gun she holds in the void, and use a fishing rod in the void to withdraw money, which conforms to the rule that 'tool functions take effect'."

Model Answers and Reasoning Process Analysis

[Figure 277]

[Figure 278]

InternVL-3-8B

Model Response Thinking process analysis

###### VOE

In the video, the clown uses his superpowers to make a green apple and money appear from nowhere and give them away. This demonstrates

that the clown can create objects (green apple and money) out of thin air. Looking at the provided answer options, option A states \"Can create life forms out of thin air,\" which is very similar to the scenario in the video where the clown creates a green apple. Options B, C, D, E, and F do not match the actions performed by the clown in the video. Therefore, the answer is A.

[Figure 279]

[Figure 280]

The model failed to correctly identify the essential function of the clown's "void tool," mistakenly interpreting as "creating objects out of thin air." Key visual information was missed: When the clown uses the void gun to shatter a bottle, there is no physical object in his hand.

[Figure 281]

[Figure 282]

[Figure 283]

Qwen-VL-2.5-7B

Model Response Thinking process analysis

###### RE

Based on the video, the clown is seen manipulating objects and people, which suggests that one of the rules of the clown's superpowers is to manipulate other people's thoughts. The clown is seen holding a green object and pointing at it, which could be interpreted as a manipulation of the object's properties. Therefore, the correct answer is C.

[Figure 284]

[Figure 285]

The model correctly perceived the key visual information of the clown using the void tool (such as the void gun), but in the reasoning process, it mistakenly interpreted the core mechanism of "void tool producing actual functionality" as "manipulating others' thoughts." The clown's performance leading to David's cooperation is a normal interaction, and the green-clad man's mockery is an autonomous behavior, neither of which demonstrates mind control.

[Figure 286]

[Figure 287]

[Figure 288]

Model Response Thinking process analysis

###### RE

Gemini-2.5-Pro

The video depicts a mime interacting with people at a train station. Initially, the mime cracks a bottle with finger guns. Then, she uses finger guns to shoot the apple from a man's head, while she also uses them to remove the money from another man's pocket. There isn't any evidence that the clown is creating anything from nothing, manipulating thoughts, or seeing the future. It seems that the clown can teleport objects and make finger guns into functional void tools, and those skills will help extract the money. Changing the object color isn't demonstrated by the clown. So options C and E look most reasonable.

[Figure 289]

[Figure 290]

The model correctly identified the core concept of the "void tool" (such as the finger gun), but mistakenly attributed the action of "taking money from the pocket" to "teleportation" rather than the "activation of the void tool's functionality." Its thought process confused the logical essence of two abilities: the correct analysis emphasizes that the void tool itself has practical functions (such as fishing rod retrieving items),

[Figure 291]

[Figure 292]

[Figure 293]

Model Response Thinking process analysis

###### RE

Video-R1

Let's break down the question and analyze the given options in the context of the video content.\n\nFirst, we need to identify the clown's superpowers. The clown in the video is seen performing various tricks and illusions, which suggests that the clown has some form of magical or supernatural abilities. The clown uses a green object to perform a trick, and the person in the green jacket seems to be the target of the clown's powers.\n\nNow, let's consider each option:\n\C: Manipulate other people's thoughts. - The clown does not seem to be manipulating thoughts directly. The person in the green jacket appears to be reacting to the clown's actions, but there is no clear evidence of thought manipulation.The clown does not teleport objects. Therefore, the correct answer is C.\n\nFinal answer: C"

[Figure 294]

[Figure 295]

[Figure 296]

The model mistakenly interpreted the actual function of the clown using void tools (such as pretending to use a handgun to smash objects or using a void fishing rod to take money) during the inference process as "manipulating others' thoughts" (Option B). However, the correct analysis clearly indicates that these actions fall under "void tools performing their actual function" (Option E).

[Figure 297]

Figure 11: Example of question, model answers and reasoning process analysis.

"Question Type": "CTI", "Question": "What is the core theme of the video?" "Options": {

- "A": "Superpowered individuals need to hide their identities."
- "B": "Friendship can overcome all difficulties."
- "C": "Audiences and performers should respect each other."
- "D": "Money is the root of all evil."
- "E": "Revenge is a means of justice."
- "F": "Performance requires professional training."

}, "Answer": "C" "Explanation": "The man in green was killed for disrespecting the performer, and the clown committed a crime for extorting tips, doubly criticizing the behaviors of disrespect and coercion."

Model Answers and Reasoning Process Analysis

[Figure 298]

[Figure 299]

InternVL-3-8B

Model Response Thinking process analysis

###### RE

The video revolves around a man-in-black costume, who representsingly known to deceive others., he uses his charm and tricks to lure people over the night at the train station., People are fooled, but finally know themselves, one even tries to retaliate but fails. So, the core theme could be that performing requires professional training. However, the idea that performance does require professional training only applies a little during the interactions between the man-in-black and his audience, and doesn't capture the overall message of the video, which is that audiences should respect performances, not try to interfere. Thus, the better choice would be that revenge is a means of justice.

[Figure 300]

[Figure 301]

Its reasoning process confused the causal relationship between the performance actions and the themes, and it did not correctly associate the key events (mockery and refusal to tip leading to conflict) with the core theme of "respect," resulting in a logical error in reasoning.

[Figure 302]

[Figure 303]

[Figure 304]

Qwen-VL-2.5-7B

Model Response Thinking process analysis

###### VOE

The clown seems to be performing some kind of magic trick or illusion, as he is seen holding a green object and pointing at the man. The man appears to be reacting to the clown's actions, possibly in a state of surprise or disbelief. The setting appears to be a nighttime outdoor environment, possibly a street or a park. The clown's costume and makeup suggest that he is a performer, and the man's reaction indicates that he is an audience member. The core theme of the video seems to be the interaction between a performer and an audience member, and the impact of the performer's actions on the audience's perception. The correct answer is not among the given choices.

[Figure 305]

[Figure 306]

In its reasoning process, the model failed to mention the crucial scenes where the man in green pretends to tip, mocks the clown's performance, and the clown becomes enraged and commits murder. These visual cues directly point to the core theme of "mutual respect between the audience and the performer."

[Figure 307]

[Figure 308]

[Figure 309]

Model Response Thinking process analysis

RE

Gemini-2.5-Pro

The core theme must relate to the act being performed and the interactions among people C: Audiences and performers should respect each other. The lack of respect on behalf of the viewer triggered a harmful revenge act by the mime. It thus reflects disrespect between a performer and a viewer. E: Revenge is a means of justice. The person who didn't tip the mime was \"punished\". The mime seeks revenge.\nF: Performance

[Figure 310]

requires professional training. The video does not focus on the performer's talent (or lack thereof). The quality of training is not mentioned.\nThe

most likely options are D and E. E is slightly more accurate since there isn't explicit respect.

[Figure 311]

The model correctly identified the elements of "disrespect" and "revenge," but it erroneously separated their causal relationship when determining the theme, the model extracted "revenge" as an isolated theme, overlooking the narrative logic of the video, which uses violent counteractions to emphasize the core conflict of "respect." This represents a misjudgment in the reasoning layer regarding the association of information.

[Figure 312]

[Figure 313]

[Figure 314]

Model Response Thinking process analysis

###### TRAR

Video-R1

The video depicts a clown-like character engaging with various individuals in a nighttime urban setting. The clown interacts with people in a playful and somewhat mischievous manner, suggesting a performance or act of some sort. The presence of money and the interaction with a man in a green jacket imply a transaction or exchange, possibly related to the performance. The clown's exaggerated gestures and the reactions of the people around him suggest that the core theme revolves around the nature of performance and the relationship between performers and their audience.\n\nLet's analyze the optionsC: Money is the root of all evil. - The video does not delve into the moral implications of money; it focuses on interaction.\n\nFinal answer: C

[Figure 315]

[Figure 316]

[Figure 317]

The model's reasoning process correctly identified the theme of respect in the interaction between the performer and the audience, aligning with the core viewpoint of "dual critique of disrespect and coercive behavior" .Although the model did not explicitly mention the criminal detail of "extorting a tip," it analyzed the essence of the interaction and eliminated incorrect options, leading its logic to align with the correct answer.

[Figure 318]

Figure 12: Example of question, model answers and reasoning process analysis.

## References

- [1] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 2
- [2] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 3
- [3] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 3, 4, 7, 9
- [4] OpenAI. Introducing openai o1. 2024. 2, 3
- [5] OpenAI. Openai o3. 2025. 2, 9
- [6] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025. 2, 3, 6
- [7] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025. 2, 3, 6
- [8] Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Lu Qiu, Ying Shan, and Xihui Liu. Exploring the effect of reinforcement learning on video understanding: Insights from seed-bench-r1. arXiv preprint arXiv:2503.24376, 2025. 2, 3, 6
- [9] Google. Gemini-2.0-flash-thinking, 2024. 2, 6
- [10] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024. 2, 3
- [11] Xuehai He, Weixi Feng, Kaizhi Zheng, Yujie Lu, Wanrong Zhu, Jiachen Li, Yue Fan, Jianfeng Wang, Linjie Li, Zhengyuan Yang, et al. Mmworld: Towards multi-discipline multi-faceted world model evaluation in videos. arXiv preprint arXiv:2406.08407, 2024. 2, 3
- [12] Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, et al. Mmvu: Measuring expert-level multi-discipline video understanding. arXiv preprint arXiv:2501.12380, 2025. 2, 3
- [13] Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, and Feng Zhao. Vcr-bench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint arXiv:2504.07956, 2025. 2, 3
- [14] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025. 2, 3
- [15] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 2, 3
- [16] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 2, 3
- [17] Zixu Cheng, Jian Hu, Ziquan Liu, Chenyang Si, Wei Li, and Shaogang Gong. V-star: Benchmarking video-llms on video spatio-temporal reasoning. arXiv preprint arXiv:2503.11495,

2025. 2

- [18] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075,

2024. 2, 3

- [19] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3
- [20] Kaizhi Zheng, Xuehai He, and Xin Eric Wang. Minigpt-5: Interleaved vision-and-language generation via generative vokens. arXiv preprint arXiv:2310.02239, 2023. 3
- [21] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218, 2023. 3
- [22] Qingxing Cao, Junhao Cheng, Xiaodan Liang, and Liang Lin. Visdiahalbench: A visual dialogue benchmark for diagnosing hallucination in large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12161–12176, 2024. 3
- [23] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355,

2023. 3

- [24] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 3
- [25] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024. 3
- [26] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 3
- [27] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 3, 5
- [28] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 3
- [29] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 3
- [30] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215,

2025. 3, 5

- [31] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025. 3
- [32] Zhenyi Liao, Qingsong Xie, Yanhao Zhang, Zijian Kong, Haonan Lu, Zhenyu Yang, and Zhijie Deng. Improved visual-spatial reasoning via r1-zero-like training. arXiv preprint arXiv:2504.00883, 2025. 3
- [33] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative selfimprovement. arXiv preprint arXiv:2503.17352, 2025. 3

- [34] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 3
- [35] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024. 3
- [36] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamav-o1: Rethinking step-by-step visual reasoning in llms. arXiv preprint arXiv:2501.06186, 2025. 3
- [37] Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang. Tinyllava-video-r1: Towards smaller lmms for video reasoning. arXiv preprint arXiv:2504.09641, 2025. 3
- [38] Ye Wang, Boshen Xu, Zihao Yue, Zihan Xiao, Ziheng Wang, Liang Zhang, Dingyi Yang, Wenxuan Wang, and Qin Jin. Timezero: Temporal video grounding with reasoning-guided lvlm. arXiv preprint arXiv:2503.13377, 2025. 3
- [39] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923,

2025. 3, 5

- [40] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In ACM Multimedia, 2017. 3
- [41] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 9127–9134,

2019. 3

- [42] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 3
- [43] Cheng Xu, Xiaofeng Hou, Jiacheng Liu, Chao Li, Tianhao Huang, Xiaozhi Zhu, Mo Niu, Lingyu Sun, Peng Tang, Tongqiao Xu, et al. Mmbench: Benchmarking end-to-end multi-modal dnns and understanding their hardware-software implications. In 2023 IEEE International Symposium on Workload Characterization (IISWC), pages 154–166. IEEE, 2023. 3
- [44] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for longcontext interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 3
- [45] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271,

2024. 5

- [46] Sundar Pichai, D Hassabis, and K Kavukcuoglu. Introducing gemini 2.0: our new ai model for the agentic era, 2024. 6
- [47] Google. Gemini-2.0-pro, 2025. 6
- [48] Google. Gemini-2.5-pro, 2025. 6
- [49] OpenAI. Hello gpt-4o, 2024. 6
- [50] OpenAI. o4-mini, 2025. 6
- [51] Anthropic. The claude 3 model family: Opus, sonnet, haiku, 2024. 6

