## Demo-ICL: In-Context Learning for Procedural Video Knowledge Acquisition

Yuhao Dong1* Shulin Tian1* Shuai Liu1 Shuangrui Ding2,3 Yuhang Zang2 Xiaoyi Dong2 Yuhang Cao2 Jiaqi Wang2 Ziwei Liu1

1 S-Lab, Nanyang Technological University 2 Shanghai AI Lab 3 CUHK-MMLab https://github.com/dongyh20/Demo-ICL

# arXiv:2602.08439v1[cs.CV]9Feb2026

Q: According to the video, till after heating the oil, what should I do now to cook the Mexican Rice?

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Demo-ICL

Search

Adapt

Demonstration Selection

Retrieve

Video-demo In-Context Learning Video Candidates

###### Text-demo In-Context Learning

Video Candidate Pool

Video Demonstration

###### Text Instructions

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

∫

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

- Step 1: Wash the rice; Heat the oil
- Step 2: Add the tomato purée
- Step 3: Add onion and garlic; fry for 2 minutes

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]|
|---|

…

[Figure 58]

[Figure 59]

...

Based on the video demonstration, which shows adding tomato purée during cooking, I can learn from it and tell that the answer to this question would be adding the tomato purée.

Based on the video candidates, I should select the last video to answer the given question. From the content of video, I can infer the final answer is adding the tomato purée.

Based on the text instructions of the process of making Mexican rice; I can see current video with the step of heating oil, and the next step could be adding the tomato purée.

Figure 1. Overview of the Demo-driven Video In-Context Learning Task with three distinct settings: (1) Text-demo in-context learning, where text instructions act as the demonstrations; (2) Video-demo in-context learning, where a video demonstration acts as the reference; and (3) Demonstration Selection, which requires identifying the most relevant video demonstrations among the video candidate pool and using them to guide video in-context learning.

### Abstract

video in-context learning capabilities. Demo-ICL-Bench is constructed from 1200 instructional YouTube videos with associated questions, from which two types of demonstrations are derived: (i) summarizing video subtitles for text demonstration; and (ii) corresponding instructional videos as video demonstrations. To effectively tackle this new challenge, we develop Demo-ICL, an MLLM with a two-stage training strategy: video-supervised fine-tuning and informationassisted direct preference optimization, jointly enhancing the model’s ability to learn from in-context examples. Extensive experiments with state-of-the-art MLLMs confirm the difficulty of Demo-ICL-Bench, demonstrate the effectiveness of Demo-ICL, and thereby unveil future research directions.

Despite the growing video understanding capabilities of recent Multimodal Large Language Models (MLLMs), existing video benchmarks primarily assess understanding based on models’ static, internal knowledge, rather than their ability to learn and adapt from dynamic, novel contexts from few examples. To bridge this gap, we present Demo-driven Video In-Context Learning, a novel task focused on learning from in-context demonstrations to answer questions about the target videos. Alongside this, we propose Demo-ICL-Bench, a challenging benchmark designed to evaluate demo-driven

*Authors contributed equally to this research. Corresponding authors.

### 1. Introduction

Video understanding remains challenging. Recent Multimodal Large Language Models (MLLMs) [2, 4, 12, 15, 29, 34, 51, 62, 77] show progress from short-clip recognition [17, 25, 44, 74] to long videos analysis [9] from daily-life settings [32, 44] to instructional videos [45, 58, 83]. However, existing video benchmarks typically pose questions that rely on either internal pre-trained knowledge (e.g., asking “what is a whisk?”) or by visible facts in the target video (e.g., “where is the whisk?”). This is fundamentally different from a more challenging scenario where a model must learn a new process or skill from demonstrations (e.g., a video tutorial that teaches the model how to cook Mexican Rice) and then apply that learned knowledge to answer questions based on a new, related target video sequence. This scenario reflects human learning and is crucial for downstream applications like robotics, where robots can learn from demonstrations to tackle new tasks. For example, in Fig. 1, after seeing only the first step of heating oil for Mexican Rice, the model is asked “what should you do next?” based on in-context text instructions or video demonstrations. This question requires knowing the specific sequence of steps for this particular version of Mexican Rice that the model is presumably meant to understand or follow, such as based on the in-context video demonstrations.

To better encourage models to learn new skills from context and adapt to novel tasks, we propose a challenging video understanding task called Demo-driven Video In-Context Learning (Demo-driven ICL). Our task embodies this by presenting target videos and questions alongside in-context text guidelines or video demonstrations. As shown in Fig. 1, Demo-driven ICL has three sub-settings: (1) text-demo incontext learning, (2) video-demo in-context learning, and (3) demonstration selection. These questions explicitly require models to use the knowledge provided within in-context examples, rather than just their static internal knowledge. A key difference between Demo-driven ICL and previous incontext learning paradigms lies in the data modality and interaction type: our in-context demonstrations can be videos, and the task involves choosing from a candidate pool to identify suitable in-context examples by the model itself. The Demo-driven ICL tasks mirror how one might learn a complex skill, such as cooking, by searching for related video demonstrations and watching them while also consulting supplementary visual or textual guides.

To evaluate the proposed Demo-driven ICL task, we present Demo-ICL-Bench, a benchmark that consists of text/video demonstrations, target videos, questions, and answers. We collected instructional YouTube videos from the HowTo100M dataset [45], ensuring subtitles and timestamps. Subsequently, we used an LLM to summarize these subtitles, generating text demonstrations to serve as in-context examples. Additionally, we employ video search ranking

methods and an LLM to identify and select videos similar to the target video to serve as in-context video demonstrations, and we also construct a video candidate pool for the model to select from and learn, in order to mimic real-world scenarios. Demo-ICL-Bench is complex and challenging: answering every question demands an accurate understanding of the demonstrations, resulting in frontier models like Gemini2.5-Pro achieving merely 46.6% and 32.0% accuracy when processing text and video demonstrations, respectively.

To address Demo-driven ICL, we present Demo-ICL with a two-stage training strategy: video supervised finetuning, and information-assisted Direct Preference Optimization (DPO) for demo-driven video ICL. We design an information-assisted DPO data generation pipeline to produce high-quality chosen responses by simplifying the task with additional contextual information. Demo-ICL outperforms existing MLLMs on the proposed Demo-driven ICL, VideoMMMU for video knowledge acquisition [28], and VideoMME [19] for general video understanding.

Our key contributions are: (i) New Challenging Tasks: We design three Demo-driven Video In-Context Learning (Demo-driven ICL) tasks, which enable models to answer questions by learning from text or video demonstrations, representing a significant step towards more human-like learning and decision-making processes in video understanding tasks. (ii) New Benchmark and Evaluation: We establish a new Demo-ICL-Bench that is specifically designed for evaluating demo-driven video in-context learning capabilities. Based on Demo-ICL-Bench, we conduct comprehensive evaluations of cutting-edge baselines, showcasing various challenges of our proposed task. (iii) New Demo-ICL Model: We present a new model, Demo-ICL, along with a customized two-stage training strategy that enhances a model’s ability to learn and adapt from in-context demonstrations. Compared with SOTA models, Demo-ICL shows competitive performance across existing benchmarks [19, 28, 66], demonstrating its superior video comprehension and in-context knowledge acquisition capabilities.

### 2. Related Work

Multimodal Video Understanding for Knowledge Acquisition. Multimodal video understanding is evolving from low-level perception toward knowledge acquisition - the ability to extract, structure, and apply information from complex instructional videos, where large-scale instructional datasets play a pivotal role in this shift. HowTo100M [45] introduced 1.2 million narrated videos with 136 million clip–caption pairs for procedure recognition and cross-task transfer. Other instruction-based datasets [58, 83] provide fine-grained task annotations or support weakly supervised step parsing across diverse procedures. More recently, benchmarks like VideoMMMU [28], Video-MMLU [56], and VideoMathQA [53] start to probe models’ capabilities to learn from educational

videos, shifting emphasis from perception to knowledge uptake and application. To further benchmark the video understanding task for knowledge acquisition that is closer to realworld settings, we introduce Demo-driven Video In-Context Learning and propose Demo-ICL-Bench, to systematically evaluate models’ capabilities of acquiring new concepts via given video demonstrations.

Multimodal In-Context Learning. In-context learning (ICL) enables models to perform new tasks by conditioning on a few examples at inference. Initially developed for large language models (LLMs), ICL has been extended to multimodal settings and shows consistent performance gains across language and image tasks [8, 42, 46, 80]. However, video-based ICL remains underexplored: current video MLLMs [38, 43, 76] mainly emphasize zero-shot performance through curated video instruction datasets for openended QA, captioning, and dialog capabilities. Emerging works add chain-of-thought methods for video understanding and reasoning tasks [3, 24, 26, 61, 65, 75] encourage stepwise evidence aggregation and explicit explanation, while some retrieval-based methods like VideoRAG [54, 60] establish a new paradigm of retrieving video moments and ground answers in cited segments. However, the aforementioned works merely use the context as a reference, rather than adapting to and learning from the provided context. In our work, we address this gap by introducing the Demo-driven ICL task on instructional video datasets, supported by an optimized training pipeline, to enhance the model’s capability to learn from the in-context video demonstrations.

### 3. Demo-ICL: Procedural Knowledge Learning from In-Context Demonstrations

In this section, we provide a detailed task formation and dataset construction of the proposed Demo-driven Video InContext Learning tasks. Sec. 3.1 presents task definitations, Sec. 3.2 outlines the dataset construction process. Finally, we demonstrate how we train our model to achieve demo-driven video knowledge acquisition in Sec. 3.3.

#### 3.1. Demo-driven Video In-Context Learning

Learning from demonstrations and imitating actions are crucial skills for humans when acquiring new abilities. Such capabilities enable individuals to rapidly master novel tasks from only a handful of examples, thereby supporting efficient adaptation and facilitating lifelong learning. In contrast, contemporary video models largely depend on supervised fine-tuning to acquire task-specific capabilities, neglecting the importance of learning from in-context examples and evaluating such capabilities for achieving human-like performance. Additionally, humans often learn new tasks incrementally. This alignment underscores the need for models that support procedural video knowledge acquisition, enabling the incremental internalization and generalization of

task procedures in a human-like manner.

To address such problems, we propose a new set of three tasks called Demo-driven Video In-Context Learning. These three tasks are designed to evaluate the model’s ability to learn from in-context demonstrations. Given an instructional video VD or text demonstration TD, the model must first interpret the example to understand how a task should be completed. It is then presented with a test video VTest, and its ability to transfer knowledge from the demonstration is assessed by predicting subsequent steps of an action A[t

1,t2] based on the demonstration and the available context VTest[0,t1]. Depending on the format and source of demonstrations, we define three distinct tasks:

- 1) Text-demo In-Context Learning: The model answers questions from an input video (e.g., “what should I do now to cook Mexican Rice?”) by retrieving information from the corresponding textual instructions (e.g., “Step 1: Wash the rice; Step 2: Add the tomato; Step 3: ...”), which serve as the demonstration. For example: “Based on the text instructions, the current video corresponds to Step 2. Therefore, the next step is Step 3: add the onion.”
- 2) Video-demo In-Context Learning: The model answers questions about a target video by conditioning on a provided video demonstration of a similar task, and using the demonstration as an in-context exemplar. For instance: “Given the video demonstration, the target clip aligns with Step 2; therefore, the next step is Step 3: add the onion.”
- 3) Demonstration Selection: The model is given the input video and a pool of video candidates (e.g., a pool containing “Mexican rice,” “fried rice,” and “pasta,”). The model must first select the most relevant demonstration (e.g., “Mexican rice”) from the pool and then use it to answer the question, simulating a scenario where a perfectly aligned demonstration is not provided.

Collectively, these three tasks constitute a systematic and comprehensive framework for demo-driven video in-context learning. They underscore distinct model capabilities, ranging from textual retrieval to demo-based knowledge extraction and adaptation, and correspond to successive stages of development, spanning from idealized oracle settings to practical real-world scenarios.

#### 3.2. Dataset Construction

In this section, we introduce a comprehensive data generation pipeline to support the proposed demo-driven video ICL task. The pipeline emphasizes four key qualities: ensuring that the video content is informative, the textual demonstration is precise, the video demonstrations are contextually relevant, and the generated questions remain answerable. We develop a structured step-by-step process (see Fig. 2 (i)) and detail each component below.

Video Collection and Annotation. We use video data from HowTo100M [45], a large-scale corpus of narrated YouTube

[Figure 60]

[Figure 61]

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

HowTo100M Tasks

ASR

###### xN

[Figure 75]

Video Annotations

: filter by rule : filter by LLMs

[Figure 76]

[Figure 77]

{metadata: rank, title}

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

###### TextGuidanceGeneration

- Step4: Add garlic t:…
- Step5: Fry for 2 mins t:…
- Step6: . . .

- Step1: Welcome t:25.14
- Step2: Heat the oil t:45.26
- Step3: Add onion t:74.87

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Top-2 Video Pairs

Raw Instructional Steps

Differentsamples

[Figure 91]

Text instruction 2

Text instruction 1

Step4: Add garlic t:… Step3: Fry 2 mins t:…

Step1: Welcome t:25.14

[Figure 92]

p < 𝜃

[Figure 93]

- Step1: Heat the oil t:45.26
- Step2: Add onion + garlic t:74.87

target demonstration

Summarized Instructional Steps

|[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]|
|---|

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Video Demonstrations

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

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

||[Figure 122]<br><br>[Figure 123]|[Figure 124]<br><br>[Figure 125]|[Figure 126]<br><br>[Figure 127]|
|---|---|---|
|
|---|

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

+

[Figure 134]

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

Text Demonstrations

Video Candidates Pool

Dataset Construction

Coarse-to-Fine Selection & Verification

[Figure 146]

Training Strategy

Reward Model

|question|
|---|

|0.8|
|---|

+

- answer 1
- answer 2

[Figure 147]

|Assistive Information<br><br>• video timestamps (text-demo)<br>• text guidance (video-demo)<br>|
|---|

|0.5|
|---|

+

Pretrained Model

[Figure 148]

|SFT|
|---|

|DPO| |
|---|---|
| | |

Demo-ICL SFT

rollout

Demo-ICL

...

...

[Figure 149]

|0.2|
|---|

answer n

(2) Stage 2: Information-assisted DPO

(1) Stage 1: Video Supervised Finetuning

Figure 2. Overview of Data Construction and Training Strategy. (i) illustrates our coarse-to-fine dataset collection pipeline (Sec. 3.2); (ii) presents the tailored two-stage training strategy for training the Demo-ICL model (Sec. 3.3).

instructional videos designed for complex tasks. This dataset is particularly suitable for demo-driven video in-context learning, as it allows models to acquire procedural knowledge from step-by-step demonstrations. With over 100 million clips covering 23,000 activities, HowTo100M provides diverse and extensive instructional material for building our benchmark. After selecting this source, we first filter videos based on video length, language, and title availability. To obtain high-quality annotations, we use ASR outputs as they offer more detailed descriptions of demonstrated activities compared with video captions. Specifically, we use annotations from HTM-AA [27], which employs WhisperX [6] to generate sentence and word-level timestamps.

and close alignment with the depicted actions. Through this multi-stage refinement, we obtain precise and reliable textual guidance that captures both the procedural structure of the task and its visual grounding.

Video Demonstration Selection. To enable video-based in-context learning, we construct pairs of videos that illustrate similar tasks. These paired demonstrations serve as explicit visual guidance, allowing the model to observe alternative executions and acquire procedural knowledge from reliable examples. Pair selection follows a coarse-to-fine process. We first leverage metadata from HowTo100M, which provides YouTube search rankings for specific tasks, and discard tasks with few relevant videos to ensure pairing quality. Next, we evaluate the titles of top-ranked videos for each task and select the two with the highest semantic similarity using Qwen2.5-72B. To further validate these candidates, we generate textual instructions for both videos using the previous pipeline, after which the LLM compares the instructions to confirm that the videos indeed demonstrate similar tasks that can be transferred. This process ensures both accuracy and reliability of the selected video guidance.

Text Demonstration Generation. We generate textual demonstrations for each video using a coarse-to-fine pipeline that produces step-by-step instructions. First, we use Qwen2.5-72B [5] to summarize the ASR transcripts into a sequence of clips, identifying the instructional steps needed to complete the task. Next, we filter out irrelevant steps, retaining valid steps while merging redundant ones into the nearest relevant segment to preserve continuity. This process yields a coherent, task-focused sequence of instructions. Finally, we incorporate Qwen2.5-VL-72B [5] to refine the demonstrations by jointly considering the step descriptions and corresponding video clips, ensuring contextual accuracy

Demo-driven Question Generation. With curated videos and corresponding instructions, we construct questions to evaluate demo-driven video in-context learning. For textbased in-context learning, we exclude tasks with fewer than

six steps to ensure sufficient complexity. From each valid sequence, one intermediate step (excluding the first and last) is randomly selected, and a question is generated from this step. The model is then required to predict the next action.

For video-based in-context learning, the goal is to test whether models can effectively leverage visual demonstrations. To this end, we employ an LLM to analyze the generated instructions for paired videos, determining whether they represent comparable tasks suitable for question generation. If validated, the LLM identifies the target step and its corresponding timestamp within the pair. Human annotators then assess the generated questions, focusing on whether the visual demonstrations provide meaningful contextual evidence for answering. To avoid trivial cases, we further filter out highly similar video pairs, ensuring the task meaningfully tests the model’s adaptability and generalization.

For the demonstration selection task, the validated video pairs are treated as ground truth and augmented with 3 carefully chosen irrelevant videos. To ensure diversity, for each video, we dynamically select the video candidates to form a unique video candidate pool for each question, rather than keeping the pool static. This construction requires models to distinguish informative demonstrations from distractors, an essential capability for real-world deployment.

Dataset Partition and Benchmark Statistics. Following established protocols, we first generate 5,000/2,000/1,000 questions for text-demo ICL/video-demo ICL and demonstration selection settings. To construct Demo-ICL-Bench, we then manually curate representative video demonstrations that highlight the role of demo-driven video in-context learning. Specifically, we sample 500 questions each for the text-demo and video-demo settings and 200 questions for the demonstration selection setting, resulting in a balanced benchmark of 1,200 questions in total.

#### 3.3. Learning from In-Context Demonstrations

We further train Demo-ICL models to validate the effectiveness of the proposed demo-driven video in-context learning task. The training pipeline is intentionally simple yet effective. As shown in Fig. 2 (ii), we adopt a two-stage strategy to progressively integrate demo-driven in-context learning. The model is first fine-tuned on a tailored dataset to enhance fine-grained video comprehension and general in-context reasoning. Then we employ a customized DPO framework to specifically strengthen the model’s capacity to learn from video demonstrations in context.

###### 3.3.1. Video Supervised Fine-tuning

In this stage, our goal is to equip Demo-ICL with finegrained video understanding and general in-context reasoning capabilities. To this end, we compile a large-scale dataset containing millions of samples drawn from diverse text–image pairs and video sources in open academic repositories. For image–text data, we rely on resources such as

LLaVA-OneVision [34], VisualWebInstruct [30], and other widely used collections. For video data, we incorporate material from open-source projects including LLaVA-Video [73], Oryx [40], and Ola [41]. To further enhance the model’s ability for instructional video understanding, we additionally incorporate datasets such as COIN [58] and Cross-Task [83]. We carefully exclude any videos that overlap with DemoICL-Bench to prevent data leakage and ensure fair evaluation. Finally, we perform subsampling on the generated dataset as described in Sec. 3.2 to explicitly introduce demo-driven video in-context learning signals during this stage. Together, these curated resources establish robust foundational capabilities and prepare the model for subsequent stages to enhance demo-driven video in-context learning.

###### 3.3.2. Information-Assisted Preference Optimization

Preference learning has become a critical component in the advancement of large language models, aiming to fine-tune outputs to better align with human preferences and improve real-world applicability. Traditional DPO algorithms generate multiple responses to the same query, after which a reward model ranks them and identifies preferred and rejected responses for training.

However, current models struggle with demo-driven video in-context learning, limiting their ability to generate high-quality responses. This limitation makes conventional DPO data construction pipelines less effective. To overcome these challenges, we propose an information-assisted DPO pipeline that integrates automatically generated assistive information, eliminating the need for manual annotation. For text-demo ICL tasks, we supply video timestamps to better align visual inputs with textual instructions, thereby improving accuracy. For video-demo ICL tasks, we pair video demonstrations with corresponding textual guidance to enhance response quality. For training, we define a preference dataset as P = {(x(i),Rc(i),Rr(i))}i=1,...,|P|, where each x(i) denotes the user request, and Rc(i) and Rr(i) represent the preferred and less preferred responses. We employ a reward model r∗(x,y) to approximate preferences, and a higher score denotes a stronger preference. Following the approach introduced by [52], we can model the human preference distribution p∗ using the Bradley-Terry (BT) model [7]:

exp(r∗((x,I),y1)) exp(r∗((x,I),y1)) + exp(r∗(x,y2))

p∗(y1 ≻ y2 | x) =

= σ(r∗((x,I),y1) − r∗(x,y2)),

(1) where I denotes the assistive information, and σ denotes the logistic function. To estimate the parameters of the reward model, we can formulate the problem as a binary classification task and minimize the negative log-likelihood:

LR(rϕ,P) = −E(x,R

c,Rr)∼P[log σ(rϕ(x,Rc)−rϕ(x,Rr))],

(2)

- Table 1. Evaluation results on Demo-ICL-Bench. The benchmark assesses models across three tasks. For Text-demo ICL and Video-demo ICL, we report two types of accuracy: Demo. Acc (with demos) and w/o Demo(without demos), and the improvement (∆ICL) attributed to demonstrations. DS refers to Demonstration Selection task. S.Acc refers to demonstration selection accuracy.

Text-demo ICL Video-demo ICL DS

Model Size Frame

Avg

Demo. Acc w/o Demo. ∆ICL Demo. Acc w/o Demo. ∆ICL S.Acc Acc Human - - 84.0 - - 80.4 - - 88.0 76.0 80.1 Proprietary MLLMs

Gemini-2.5-Pro [23] - - 54.4 - - 36.2 - - - 26.0 38.9 GPT-4o [49] - - 48.8 - - 31.4 - - - 24.5 34.9

Open-Source Video MLLMs

Qwen2-VL [62] 7B 32 29.0 21.8 +7.2 22.4 24.0 -1.6 38.0 14.5 22.0 Ola [41] 7B 32 32.2 23.0 +9.2 24.6 26.4 -1.8 43.0 16.0 24.3 LLaVA-Video [73] 7B 32 31.0 25.4 +5.6 30.2 29.0 +1.2 44.5 20.5 27.2 Qwen2.5-VL [5] 7B 32 32.8 26.0 +6.8 28.0 26.2 +1.8 46.0 18.0 26.3 InternVL-3 [81] 8B 32 31.4 26.6 +4.8 27.0 26.4 +0.6 44.0 16.5 25.0 Video-R1 [18] 7B 32 33.6 27.6 +6.0 27.4 26.6 +0.8 48.0 17.5 26.2 VideoChat-R1 [37] 7B 32 34.4 27.0 +7.4 28.2 26.8 +1.4 52.0 18.5 27.0 Qwen2.5-VL [5] 72B 32 45.0 24.2 +20.8 25.6 25.2 +0.4 54.0 18.0 29.5

Ola-Video [41] (Base) 7B 32 31.4 22.8 +8.6 25.0 26.0 -1.0 48.0 18.0 24.8 Demo-ICL (SFT) 7B 32 38.4 27.8 +10.6 29.4 26.2 +3.2 54.5 21.5 29.8 Demo-ICL 7B 32 43.4 29.4 +14.0 32.0 27.6 +4.4 58.0 24.0 33.1

where rϕ is the reward model. This approach enables effective alignment with human preferences by allowing the model to use additional information that can be generated automatically, thus producing high-quality responses in an effective and scalable way. Using these responses as preferred outputs and treating normal responses as rejected, we perform multiple training rounds to obtain a sequence of models M1,...,MT, where each model Mt+1 utilizes preference data Pt generated by the t-th model. Through the information-assisted DPO and iterative training strategy, we progressively endow Demo-ICL with strong demo-driven video in-context learning capabilities.

### 4. Experiments

We first perform detailed evaluation results and analyses on Demo-ICL-Bench in Sec. 4.1. We then compare our method against state-of-the-art Video MLLMs on widely used video benchmarks in Sec. 4.2. Finally, we present key ablation results in Sec. 4.3. These experimental results systematically validate the value of Demo-ICL-Bench and the advantages of our framework.

#### 4.1. Demo-ICL-Bench

Setup. We evaluate both representative proprietary MLLMs and state-of-the-art open-source video MLLMs, reporting performance across three tasks. In addition to standard evaluations, we design experiments on text-demo and video-demo in-context learning tasks, including settings without explicit guidance, in order to better characterize the current capabilities and limitations of MLLMs in demo-driven video in-

context learning. For proprietary models, we consider GPT4o and Gemini-2.5-Pro. For open-source video MLLMs, we benchmark a diverse set of representative models, including InternVL-3 [81], Qwen2-VL [62], Qwen2.5-VL [5], Ola [41], and LLaVA-Video [73]. To capture the role of specialized video reasoning, we further include Video-R1 [37] and VideoChat-R1 [18] as baselines for video reasoning models. Finally, to examine the effect of model capacity, we conduct experiments on both Qwen2.5-VL-7B and Qwen2.5VL-72B. We also incorporate the base model of Demo-ICL as a reference to more clearly demonstrate the effectiveness and advantages of our training strategy.

Text-demo In-context Learning. As shown in Tab. 1, models perform poorly without demonstrations, indicating that in-context learning is essential for task success. When text demonstrations are provided, all models improve, demonstrating their ability to integrate task-specific knowledge from in-context text demonstrations. The extent of this improvement, however, strongly depends on model size: small models typically gain less than 10 points, whereas Qwen2.5VL-72B improves by over 20 points, despite performing no better than smaller models without demonstrations. This highlights model scale as a critical factor for effective incontext learning. On Demo-ICL, the SFT model improves by over 10 points through targeted demonstration strategies, while the DPO model achieves state-of-the-art results among models of similar size. These results confirm that welldesigned data curation, combined with preference-based training, substantially enhances generalization and efficiency in video in-context learning.

- Table 2. General Video Understanding. Demo-ICL achieves superior performance on both general temporal understanding and knowledge acquisition tasks, highlighting the effectiveness of the proposed demo-driven video in-context learning framework. These results also demonstrate the robustness and generalizability of our training strategy across diverse evaluation settings.

General Temporal Understanding Knowledge VideoMME

Model Size

Long VideoBench

MLVU VideoMMMU Proprietary Models

MVBench

(wo / w sub)

GPT-4V [48] - 59.9/63.3 43.7 49.2 59.1 GPT-4o [50] - 71.9/77.2 - 66.7 66.7 61.2

- Gemini-1.5-Pro [59] - 73.2/79.8 - 64.0 - 70.4
- Gemini-2.5-Pro [14] - 84.3/86.9 - - - 83.6 Open-Sourced Video MLLMs

- VideoLLaMA2 [13] 7B 47.9 / 50.3 54.6 36.0 - LLaVA-OneVision [34] 7B 58.2 / 61.5 56.7 56.3 64.7 33.9
- VideoLLaMA3 [70] 7B 66.2 / 70.3 69.7 59.8 73.0 47.0 LLaVA-Video [76] 7B 63.3 / 69.7 58.6 58.2 70.8 Qwen2.5-VL [5] 7B 65.1 / 71.6 69.6 56.0 - 47.4 InternVL3.5 [63] 8B 66.0 / 68.6 72.1 62.1 70.2 -

VILA-1.5 [39] 40B 60.1 / 61.1 - - 56.7 34.0 VideoLLaMA2 [13] 72B 61.4 / 63.1 62.0 - - LLaVA-OneVision [34] 72B 66.2 / 69.5 59.4 61.3 66.4 48.3 LLaVA-Video [76] 72B 70.5 / 76.9 64.1 61.9 74.4 49.7 Qwen2.5-VL [5] 72B 73.3 / 79.1 70.4 60.7 74.6 60.2

Ola-Video (Base) 7B 63.0 / 68.9 66.9 60.7 69.1 46.2 Demo-ICL (SFT) 7B 65.6 / 70.2 69.4 61.6 70.4 48.8 Demo-ICL 7B 65.2 / 69.7 68.6 61.2 70.4 52.6

Video-demo In-context Learning. In the Video-demo ICL task, performance diverges from text-demo results: while some models extract information from video demonstrations, the gains are limited, and models such as InternVL-3, Qwen2-VL, and Ola even suffer degradation. This highlights the difficulty current MLLMs face in extracting and transferring temporal–visual cues for effective ICL. By contrast, Demo-ICL, equipped with demo-driven video ICL, consistently benefits from video demonstrations, though less pronounced compared to text-demo ICL tasks. Our findings indicate that dedicated strategies for video demonstrations are essential to narrow the gap between text and video guidance and to unlock further multimodal generalization.

Demonstration Selection. To approximate real-world scenarios where models must retrieve relevant demonstrations from large video pools, we evaluate them on the demonstration selection task. This task assesses the ability to identify the correct reference video and answer the corresponding questions. We report both video selection accuracy and final question accuracy conditioned on the selected video. Results show that current models often struggle to capture global semantic information, leading to failures in retrieving appropriate demonstrations and producing a substantial gap from human performance. Existing approaches lack not only effective mechanisms for knowledge extraction and transfer, but also robust search and selection capabilities essential for

demo-driven video in-context learning in real-world scenarios. Further analysis is provided in Sec. 4.3.

#### 4.2. General Video Understanding

Setup. To evaluate the generalization ability of our DemoICL model, we conduct experiments on several widely used video benchmarks. Our analysis focuses on two main directions. First, we assess video knowledge acquisition using benchmarks such as VideoMMMU [28], a representative dataset designed to test how models acquire knowledge from videos. In this setting, the model must watch an entire video and answer questions based on its content, thereby evaluating its ability to learn, retain, and apply information in new contexts. This directly highlights the effectiveness of demo-driven video in-context learning, as the model uses demonstrations to generalize beyond the training distribution. Second, we evaluate on general temporal understanding benchmarks, including VideoMME [19], MVBench [35], LongVideoBench [66], and MLVU [79], which target diverse tasks such as common video perception, action recognition, and long video understanding. Together, these benchmarks provide a comprehensive evaluation of Demo-ICL, covering both its demo-driven in-context learning generalization and its foundational video understanding capabilities.

Results. As shown in Tab. 2, Demo-ICL demonstrates competitive performance across all open-source MLLMs. On

the knowledge acquisition benchmark VideoMMMU, DemoICL performs on par with recently released models of comparable size and even surpasses several larger counterparts. These results underscore not only the strong capability of the model in visual reasoning, but also the effectiveness of the demo-driven video in-context learning paradigm for scalable knowledge acquisition. Using demonstrations within the input context, Demo-ICL extends its understanding beyond memorized content, reflecting a step towards more flexible and human-like video comprehension. Furthermore, for general video understanding benchmarks, Demo-ICL exhibits performance comparable to models of similar size, suggesting that the proposed demo-driven ICL mechanism can be seamlessly integrated without compromising general video understanding, while simultaneously enhancing knowledge acquisition. Collectively, these findings indicate that DemoICL offers a promising and scalable pathway for advancing video reasoning and general temporal understanding.

#### 4.3. Analysis Experiments

In this section, we present detailed analyses focusing on the challenges of video-demo in-context learning and strategies for training an effective demo-driven model. We highlight the limitations of current models and demonstrate the effectiveness of information-assisted DPO.

Why is the Video-demo ICL task challenging? We provide deeper insights into why the Video-demo ICL task poses significant challenges for current MLLMs, with results summarized in Tab. 3. First, we test Demo-ICL with more densely sampled frames, and the improvements demonstrate that fine-grained visual cues are critical for demo-driven video in-context learning.

Using 128 frames, we further conduct an experiment where the reference video is identical to the query video, thus providing the model with the full content as context. The performance gains in this setting suggest that direct grounding and perception are far easier than knowledge transfer through in-context demonstrations, as the model can process visuals effectively but struggles to adapt that knowledge to new scenarios. We further evaluate the use of reference clips as contextual demonstrations, where only the segments depicting the immediate next-step action are provided as in-context examples. This setting reveals a fundamental challenge for Video-demo ICL: models struggle to accurately align and match temporal evidence across demonstrations. Moreover, replacing clips with ASR transcripts and captions yields additional improvements, revealing that current MLLMs still lack robust fine-grained video compre-

Table 3. Ablation study on evaluation settings.

Settings Video-demo ICL

Base(32 frames) 29.4 128 frames 30.4 +Repeat Video 38.6 +Reference Clips 35.8 +ASR & Captions 45.4

Table 4. Training setting ablations. Instructional Video SFT, InfoAssisted DPO, and iterative DPO training strategies collectively enhance performance on the Demo-ICL-Bench benchmark.

Settings Text-ICL Video-ICL DS Avg w/o Instructional Videos 34.0 26.2 19.0 26.4 Demo-ICL (SFT) 38.4 29.4 21.5 29.8 Vanilla DPO 40.0 30.0 22.0 30.7 Demo-ICL (DPO) 1-round 41.8 30.8 22.5 31.7 Demo-ICL (DPO) 43.4 32.0 24.0 33.1

hension and often fail to abstract or summarize clips into precise knowledge for reasoning and further adaptation.

Taken together, these findings highlight why video-demo ICL is uniquely challenging: it requires not only perception but also temporal alignment, abstraction, and flexible knowledge transfer. This underscores the need for models that can truly leverage demonstrations as dynamic sources of contextual information, a critical capability for advancing video understanding and reasoning.

How to train a good demo-driven video in-context learning model? We perform ablation studies to assess the effectiveness of our training strategies, with the results summarized in Tab. 4. The findings indicate that incorporating instructional videos allows Demo-ICL to use in-context demonstrations and adapt to novel scenarios, yielding significant improvements on Demo-ICL-Bench. Our results emphasize the importance of high-quality instructional data in enabling models to generalize beyond basic perception and toward deeper contextual video understanding.

We further investigate the impact of training algorithms. When trained with vanilla DPO, the model struggles to produce high-quality responses, yielding noisy data pairs and only marginal improvements. In contrast, our informationassisted DPO method provides richer feedback signals, which significantly enhance response quality and overall performance. Through an iterative training strategy, DemoICL gradually learns from in-context demonstrations and ultimately reaches superior performance. These comparisons reveal that both the quality of demo-driven video data and the design of training strategies are essential for effective video in-context learning. Together, these results indicate that building a strong video in-context understanding model requires not only carefully structured demonstrations but also training paradigms that encourage the model to leverage contextual information.

### 5. Conclusion

In this paper, we introduce a novel task, Demo-driven Video In-Context Learning, which focuses on learning from incontext instructional demonstrations. To facilitate evaluation, we construct Demo-ICL-Bench, a benchmark consisting of 1,200 challenging questions designed to assess demo-driven video in-context learning capabilities. To effectively address this task, we further propose Demo-ICL, a video MLLM

equipped with enhanced in-context learning abilities. Extensive experiments reveal that existing MLLMs struggle with Demo-driven ICL, while Demo-ICL demonstrates improved performance, achieving superior video understanding and in-context knowledge acquisition capabilities, suggesting its potential to facilitate future developments in this area.

### Appendix

We provide supplementary documents to support our research. Implementation details are outlined in Section A. Additional visualization results are presented in Section B, followed by further experimental analysis in Section C. We also provide a more comprehensive discussion of related

- work in Section D. Finally, we discuss the limitations of our
- work in Section E.

### A. Implementation Details

#### A.1. Experiment Details

In this section, we detail the implementation of Demo-ICL. The Demo-ICL model is built upon Ola-Video, a highly pretrained multimodal understanding model that integrates OryxViT as its visual encoder to process native arbitraryresolution visual inputs, alongside Qwen2.5 as the language model. For the training process, we construct a customized dataset to establish foundational image and video understanding capabilities. For image data, resolutions range from 768 to 1536, while for video data, the number of frames is capped at 64, with frame resolutions varying between 288×288 pixels and 480×480 pixels. During training, the maximum token length is set to 16,384, and a learning rate of 1e-5 is used throughout both stages. In the DPO (Direct Preference Optimization) training phase, we curate 5,000 samples using the specified pipeline and apply a learning rate of 5e-7. A batch size of 256 is maintained across both fine-tuning stages and the DPO phase, with experiments conducted using 64 NVIDIA A100 80G GPUs.

#### A.2. Data Collection Details

In the data generation process, we utilize Qwen2.5-72B as our LLM and Qwen2.5-VL-72B as our MLLM within the pipeline. For generating text instructions, we first use Qwen2.5-72B to create summarized instruction steps. Then, when refining these steps with the MLLM, we forward each step along with 64 uniformly sampled frames from the corresponding video clips. For generating questions for videodemo ICL, we provide the text instructions of paired videos and ask the LLM to assess their reasonableness for question generation. Both the LLM and MLLM are deployed using four NVIDIA A800 GPUs.

#### A.3. Evaluation Details

For evaluation on the Demo-ICL-Bench, we design three experimental settings. For the Text-demo ICL task, we sample 32 frames from each video and provide textual demonstrations alongside the question during evaluation. For the Video-demo ICL task, we sample 32 frames for both the reference (demonstration) video and the target video. Finally, for the Demonstration Selection task, due to context-length limitations in some models, we uniformly sample 16 frames from each candidate demonstration video and 32 frames from the target video. In all three settings, video subtitles are not provided.

### B. Visualizations

We present visualization results to clarify the task design of Demo-ICL-Bench. These results are shown in Fig. 3 and Fig. 4.

### C. More Analysis Experiments C.1. General Video Understanding on Video-MME

We further evaluate the Demo-ICL model on general video understanding tasks of varying lengths and scenarios. Specifically, we employ the VideoMME benchmark to highlight its offline video comprehension capabilities, providing a broader assessment beyond domain-specific settings.

Setup. To further evaluate the generalization ability of Demo-ICL on diverse video understanding tasks, we adopt the Video-MME benchmark [19]. The dataset consists of 900 videos (254 hours) covering 6 visual domains and 30 subfields, with durations ranging from 11 seconds to 1 hour, categorized into Short, Medium, and Long. In addition to visual content, VideoMME provides audio and subtitles, enabling a multimodal and comprehensive evaluation of video MLLMs. Under this setting, the model is required to watch an entire video and then answer corresponding questions, which allows us to systematically assess robustness across varying durations, modalities, and domains, in comparison with both open-source and commercial MLLMs.

Results. Table 5 summarizes the overall performance of Demo-ICL across short, medium, and long video tracks. Demo-ICL achieves strong results on all three tracks, demonstrating robust capabilities across different temporal lengths. It surpasses open-source video MLLMs with similar parameter sizes (7B), achieves comparable results to larger models (34B), and competes closely with some commercial MLLMs. Notably, on long-duration videos, which pose greater challenges due to extended temporal dependencies, Demo-ICL demonstrates its long video understanding capabilities, maintaining consistent performance over time.

##### Setting 1: Text-demo In-Context Learning

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

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

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

Question: What to do after simmering chicken?

###### Question: What to do after pouring batter?

|TEXT INSTRUCTIONS:<br><br>1. Dry mix: Combine flour, sugar, salt, baking powder, and cloves.<br>2. Wet mix: Whisk together oil, egg, pumpkin puree, and grated ginger.<br>3. Combine: Pour wet mixture into dry ingredients, stir until just blended.<br>4. Heat griddle: Preheat and grease with butter or oil.<br>5. Portion: Pour batter and gently shape into rounds.<br>6. Cook first side: 2 minutes until golden, then flip.<br>7. Cook second side: 1–2 minutes until springy to the touch.<br>8. Serve: Top with butter and maple syrup or honey.<br>|
|---|
| |
|A. Cook pancake for about 2 minutes, check browning, flip.<br>B. Heat griddle, add butter or oil, pour batter, nudge into circle.<br>C. Prepare wet ingredients: milk, vinegar, oil, egg, pumpkin puree, grated ginger, mix well.<br>D. Serve with butter and maple syrup or honey.<br>|

|TEXT INSTRUCTIONS:<br><br>1. Parboil chicken: Boil then drain.<br>2. Simmer chicken: Transfer to a pot, cover, cook on low for 1 hour.<br>3. Soak mushrooms: Rehydrate dried mushrooms in water for 1 hour, then slice.<br>4. Blanch tomato: Boil until skin splits, cool.<br>5. Prep tomato: Peel, discard core, and dice.<br>6. Boil base: Combine chicken broth, mushrooms, and salt; bring to a boil.<br>7. Finish soup: Add tomato, season with salt and white pepper, then stir in beaten egg and chopped green onion.<br>|
|---|

|A. Put tomato in boiling water until skin cracks, then cool.<br>B. Put chicken in boiling water, then remove.<br>C. Cut and peel tomato, discard part, dice.<br>D. Soak dried mushrooms in water for 1 hour, then slice.<br>|
|---|

Based on the text demonstrations, after simmering the chicken, the next action should be soaking mushrooms. Current video shows procedure illustrated in step 3, hence, the answer to the question could be **D**.

Based on the text demonstrations, after pouring batter, the next action should be cook side by side. Current video shows procedure illustrated as step 6, the next step could be **A**.

(i) Example 1

(ii) Example 2

- Figure 3. Visualization of Text-demo In-Context Learning. This figure provides 2 examples to illustrate the text-demo in-context learning task, where the text instructions will be provided along with the target video as the inputs.

###### Table 5. Performance of Demo-ICL compared to previous MLLMs on Video-MME across short, medium, and long durations, under without “subtitles” and with “subtitles” settings.

Short (%) Medium (%) Long (%) Overall (%)

Models LLM Params

w/o subs w/ subs w/o subs w/ subs w/o subs w/ subs w/o subs w/ subs Proprietary MLLMs

GPT-4V [47] - 70.5 73.2 55.8 59.7 53.5 56.9 59.9 63.3 GPT-4o [49] - 80.0 82.8 70.3 76.6 65.3 72.1 71.9 77.2 Gemini 1.5 Flash [23] - 79.7 83.6 68.4 74.7 61.1 68.8 70.3 75.0 Gemini 1.5 Pro [23] - 81.7 84.5 74.3 81.0 67.4 77.4 75.0 81.3

Open-source Video MLLMs

LongVA [71] 7B 61.1 61.6 50.4 53.6 46.2 47.6 52.6 54.3 VITA 1.5 [21] 7B 67.0 69.9 54.2 55.7 47.1 50.4 56.1 58.7 mPLUG-Owl3 [69] 7B 70.0 72.8 57.7 66.9 50.1 64.5 59.3 68.1 TimeMarker [10] 8B 71.0 75.8 54.4 60.7 46.4 51.9 57.3 62.8 MiniCPM-V 2.6 [68] 8B 71.3 73.5 59.4 61.1 51.8 56.3 60.9 63.7

VILA-1.5 [39] 34B 68.1 68.9 58.1 57.4 50.8 52.0 59.0 59.4 Oryx-1.5 [40] 34B 77.3 80.6 65.3 74.3 59.3 69.9 67.3 74.9 Qwen2-VL [62] 72B 80.1 82.2 71.3 76.8 62.2 74.3 71.2 77.8 LLaVA-Video [72] 72B 81.4 82.8 68.9 75.6 61.5 72.5 70.6 76.9

Demo-ICL 7B 78.6 79.1 63.9 68.8 53.2 61.1 65.2 69.7

#### C.2. Results on Video-MMLU Benchmark

evaluate our approach on the Video-MMLU [56] benchmark, a video understanding dataset designed to assess lecture comprehension and video-based knowledge acquisition.

To further validate the effectiveness and generalization capability of the proposed demo-driven video in-context learning paradigm and the info-assisted DPO training strategy, we

###### Setting 2: Video-demo In-Context Learning

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

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Question: What is next step to design artificial lawn after laying out the artificial grass?

Question: What to do to create a storybook after creating shot list?

|VIDEO REFERENCE:<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>|[Figure 260]<br><br>[Figure 261]|
|---|
<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]|
|---|

|VIDEO REFERENCE:<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>|[Figure 290]<br><br>[Figure 291]|
|---|
<br><br>[Figure 292]<br><br>[Figure 293]|
|---|

|A. Outline the Plot<br>B. Storyboard Your Pages<br>C. Develop Your Characters & Setting<br>D. Decide on Style & Layout<br>|
|---|

|A. Add sharp sand, compact, and screed to level<br>B. Roll out Grono lawn, face pile towards home, trim edges<br>C. Cover area with MOT, compact to 75mm for drainage<br>D. Join grass pieces with tape and glue, add nails for security<br>|
|---|

The video demonstration shows that after creating the shot list, the person creates the drawing of shots. Based on the knowledge of creating story book I have acquired from the video demonstration the answer could be **C**.

After setting up the lawn, the person in video demo joined the grass pieces. Based on the knowledge learned from the video demo, the answer could be **C**.

(i) Example 1 (ii) Example 2

- Figure 4. Visualization of Video-demo In-Context Learning. This figure provides 2 examples to illustrate the video-demo in-context learning task, where a video demonstration will be provided together with the target video input.

Setup. The Video-MMLU benchmark consists of 1,065 videos, primarily sourced from lecture recordings. The associated questions are categorized into two types: caption and reasoning questions. Since the answers are open-ended, model performance is assessed using the LLM-as-a-Judge framework, with Qwen-2.5-72B serving as the judge model.

primarily focuses on the Text-demo ICL and Demonstration Selection tasks. For the Text-demo ICL task, we examine the accuracy of the textual demonstrations and assess whether the provided information is sufficient for models to correctly answer the corresponding questions. For the Demonstration Selection task, as the correct demonstrations have undergone the same verification process as the Video-demo ICL tasks, we mainly evaluate the overall quality of all candidate video demonstrations. In total, we sample 200 Text-demo ICL tasks and 100 Demonstration Selection tasks, which are manually reviewed by human annotators. The results show that 96% of the Text-demo ICL tasks pass the human quality check. For the Demonstration Selection tasks, all video samples meet the quality criteria, and in 88% of the cases, human annotators can confidently identify the correct demonstration video, as shown in 2 in our paper. We plan to refine and correct the remaining tasks that did not pass the human review to further enhance the overall dataset quality.

Results. As shown in Table 6, incorporating demo-driven video in-context learning significantly enhances the model’s ability to learn from lecture videos, where the videos act as demonstrations for subsequent reasoning questions. While the model maintains competitive performance on the caption task, its performance on the Quiz track improves by a substantial margin compared to previous models. It is worth noting that we employ Qwen2.5-7B as the backbone LLM, the same base model used in Qwen2.5-VL, yet our method achieves markedly better results on the Quiz track. This demonstrates the effectiveness of the proposed demo-driven video in-context learning and info-assisted DPO strategy in general video knowledge acquisition.

#### C.4. Why Demonstration Selection Difficult?

To better understand why the Demonstration Selection task remains challenging for current models, we conduct a detailed analysis of this track. First, we observe that demonstration selection itself poses significant difficulty. By calculating the demonstration accuracies of existing models, we

#### C.3. Quality of Demo-ICL-Bench

We conduct a user study to further evaluate the quality of our Demo-ICL-Bench. Since human annotators have already verified the quality of the Video-demo ICL tasks, the study

- Table 6. Video Knowledge Acquisition on Video-MMLU. DemoICL achieves superior performance on the Video-MMLU benchmark, showing substantial gains on the Quiz track, where the model must leverage knowledge from lecture videos to answer novel reasoning questions. These results demonstrate the effectiveness of our demo-driven video in-context learning paradigm and the strong generalization capability enabled by the info-assisted DPO training strategy.

Model Size Notebook Quiz Overall Proprietary MLLMs

Gemini-1.5-Flash - 39.5 47.8 43.6 GPT-4o - 53.9 44.9 49.4 Claude-3.5-sonnet - 67.4 71.2 69.3

Open-Source MLLMs

InstructBLIP-7B 7B 19.3 2.9 11.1 Cambrian-8B 8B 20.2 5.2 12.7 LLaVA-1.5 7B 22.3 9.1 15.7 Video-LlaVA-7B 7B 15.3 16.5 15.9 LLaVA-NeXT (Vicuna) 7B 18.1 24.9 21.5 Qwen-VL 7B 24.4 19.6 22.0 mPLUG-Owl3 7B 22.6 22.6 22.6 LLaVA-NeXT (LLaMA3) 8B 16.5 30.1 23.3 InternVL2-8B 7B 31.4 16.7 24.1 LLaVA-NeXT (Mistral) 7B 20.3 31.5 25.8 Qwen2-VL-7B 7B 34.2 23.4 28.8 LLaVA-NeXT-Video-7B 7B 35.8 27.4 31.6 LLaVA-OneVision-OV 7B 34.6 33.4 34.0 Qwen2.5-VL-7B 7B 42.0 32.9 37.5 InternVL2.5-8B 8B 34.5 44.5 39.5

Demo-ICL 7B 41.0 50.4 45.7

find that their poor performance indicates an inability to effectively utilize the correct in-context demonstrations when multiple candidates are provided. This suggests that current models exhibit fragile reasoning capabilities in complex, real-world scenarios, where single correct demonstrations can not be directly provided.

Second, based on the Qwen2.5-VL model, we compute the final accuracy for tasks where the model successfully selects the correct demonstrations and compare these results with those from the Video-demo ICL setting for the same questions. Interestingly, even when the correct demonstrations are identified, the model achieves only 22.2% final accuracy, compared to 25.4% in the Video-demo ICL setting. These findings reveal that, although current models can recognize the correct videos, they still struggle to focus on relevant information within the selected demonstrations and remain easily distracted by irrelevant context. This analysis also underscores the necessity of info-assisted DPO, which enables models to learn to emphasize the most informative and correct elements within in-context demonstrations, thereby improving overall task performance.

Table 7. More experiment results on Demo-ICL-Bench. The T. ICL and V. ICL refers to the original Text-demo ICL Demo. Acc and Video-demo ICL Demo. Acc, where DS refers to the original DS Acc.

Model Size T. ICL V. ICL DS Overall Proprietary Methods

Gemini-2.5-Pro - 54.4 36.2 26.0 38.9 GPT-4o - 48.8 31.4 24.5 34.9

Open-Source Methods

Qwen2-VL [62] 7B 29.0 22.4 14.5 22.0 Ola [41] 7B 32.2 24.6 16.0 24.3 LLaVA-Video [73] 7B 31.0 30.2 20.5 27.2 Qwen2.5-VL [5] 7B 32.8 28.0 18.0 26.3 InternVL-3 [81] 8B 31.4 27.0 16.5 25.0 Video-R1 [18] 7B 33.6 27.4 17.5 26.2 VideoChat-R1 [37] 7B 34.4 28.2 18.5 27.0

RAG-based Methods VideoRAG [54] - 52.6 22.8 18.0 31.3 Agent-based Methods

VideoAgent [64] - 36.2 23.4 17.5 25.7 Demo-ICL 7B 43.4 32.0 24.0 33.1

#### C.5. The Model Learns or Recalls?

To assess whether questions in Demo-ICL-Bench can be answered using only a model’s internal knowledge, we design a series of validation experiments. First, we evaluate Gemini-2.5-Pro on the original questions without providing any demonstrations or answer options, making the task open-ended. We use Qwen2.5-72B as the LLM judge model. Under this setting, Gemini-2.5-Pro successfully answers only 5% of the questions.

These results suggest that Demo-ICL-Bench cannot be solved solely through the internal knowledge of current models; they must instead rely on in-context demonstrations to achieve strong performance. Simply recalling prior knowledge is insufficient, as the videos contain detailed, task-specific action sequences on which the questions are based—information that the model cannot infer without actually observing the video content.

#### C.6. More Evaluation Results

To comprehensively evaluate existing video understanding methods on Demo-ICL-Bench, we further include additional categories of approaches, specifically RAG-based and agentbased methods. For each category, we select a representative model for evaluation: VideoRAG [54] for the RAGbased approach and VideoAgent [64] for the agent-based approach. As shown in Table 7, the RAG-based method VideoRAG performs well on the Text-demo ICL task, primarily because its video knowledge indexing and multimodal re-

Table 8. Related Work for Demo-ICL-Bench. Demo-ICL-Bench stands out due to its demo-driven video in-context learning settings, setting it apart from previous video benchmarks.

Benchmark Video Domain #Videos #QAs Video-ICL Annotation

ActivityNet-QA [17] Human Activities 800 8000 ✗ Manual How2QA [36] Instructional Videos 1166 2852 ✗ Manual KnowIT-VQA [22] TV Show 207 24k ✗ Manual NExT-QA [67] Web Videos (Causal/Temporal) 5.4k 52k ✗ Manual MVBench [35] Benchmark Videos 3641 4000 ✗ Auto VideoMME [20] YouTube Videos 900 2700 ✗ Manual VideoMathQA [53] Instructional Videos 420 420 ✗ Manual VideoMMMU [28] Lectures 300 900 ✗ Manual

Demo-ICL-Bench Instructional Videos 1200 1200 ✓ Mixed

trieval paradigm effectively aligns videos with the provided text demonstrations. However, in the video-demo ICL and demonstration selection tasks, where the model must transfer knowledge learned from demonstrations to solve problems in a target video, VideoRAG still underperforms. This gap underscores the importance of these two task designs and highlights that beyond simple video knowledge acquisition, current video MLLMs must also develop the ability to transfer learned knowledge to new tasks. For the agent-based method VideoAgent, we observe that incorporating agentic strategies enhances performance on the Text-demo ICL task, primarily because they strengthen the grounding capabilities of current video MLLMs and improve alignment between videos and textual demonstrations. However, the method still underperforms on the other tracks, suggesting that while agentic approaches can enhance perceptual accuracy, the ability to transfer learned or perceived knowledge to new tasks remains an open challenge, which is likely crucial for advancing video understanding and reasoning.

- D. More Discussion on Related Works In this section, we will include more details of related works.

Multimodal Video Understanding for Knowledge Acquisition. Recent research in video understanding has moved beyond low-level perception towards extracting structured knowledge from videos, like procedural steps, events, and concepts. Large-scale instructional datasets have been instrumental in this shift. For example, as mentioned in 2, a lot of instructional datasets [45, 58, 83] have driven the development of models that seek to learn high-level knowledge from video, rather than just recognize objects or actions. Moreover, VidSitu [55] addresses video situation recognition by densely annotating 10-second movie clips with semantic role labels, which provides a symbolic knowledge representation of the video. By learning to predict such structured representations, models can acquire a form of event knowledge from videos. Similarly, HT-Step [1] aligns the textual instructions

from wikiHow [31] with corresponding segments in instructional videos. It provides 116k temporal segment annotations in 20k how-to videos, each labeled with a step description from wikiHow, enabling models to learn to ground declarative knowledge in procedural video footage.

To better learn from such knowledge-intensive data, early multimodal learning approaches applied language-modeling techniques to video data. For example, VideoBERT [57] quantizes video frames into discrete “visual words” and then uses a BERT-like transformer to learn joint representations of sequences of visual tokens and narration text. Following models such as ActBERT [82] extended this masked language modeling paradigm to action recognition data, and ClipBERT [33] improved efficiency by sampling sparse key frames for end-to-end video-text pretraining. By learning from millions of narrated video clips, these models demonstrate an ability to embed procedural and commonsense knowledge implicitly in their representations. Zhou et al. [78] proposed the model Paprika used PKG-based pretrainng procedure to generate psuedo labels for instructional video to train. StepFormer [16] addresses the problem of discovering and localizing key procedure steps in instructional videos without human supervision. It uses video with subtitles (ASR) only, with a transformer decoder that attends to video frames via learnable queries to produce a sequence of key steps. Chen et al. [11] proposes a framework, MPTVA, that aligns video segments with procedure steps derived via LLM from narration text via long-term semantic similarity and short-term fine-grained similarity.

Multimodal In-Context Learning. Inspired by the textual CoT prompting, recent works curate multimodal datasets with human-written rationales to encourage step-by-step prompting. Video-CoT [65] provides video QA examples paired with detailed explanations, while Video-Espresso [26] scales this approach to large collections of reasoning exemplars. Beyond data-centric methods, Arnab et al. [3] propose Temporal Chain-of-Thought, an inference strategy

for long videos where the model iteratively selects relevant clips and reasons over them, enabling efficient multi-step reasoning over extended sequences. A complementary line of work extends retrieval-augmented generation (RAG) to video. VideoRAG [54] and related work [60] index long videos into databases of visual and textual descriptors. At query time, relevant segments and transcripts are retrieved and passed to the language model as context, grounding answers in explicit video evidence. This improves factual accuracy, transparency, and scalability, especially for long videos where direct end-to-end processing is infeasible.

### E. Limitations and Future Directions

In this section, we discuss the limitations of our work. The Demo-ICL model does not include a specialized architecture for demo-driven video in-context learning. Instead, we employ a customized training strategy to achieve this functionality. Our goal is to equip current MLLMs with demo-driven video in-context learning capability without requiring architectural modifications, thereby simplifying the integration of these new capabilities and the maintenance of previous multimodal understanding.

Additionally, we did not explore how models can effectively learn from diverse contexts, such as different modalities or resources. This ability is more similar to the natural human learning process, where individuals can draw on a wide range of resources, such as text instructions and instructional videos, to enhance understanding simultaneously. Combining various types of contextual information to improve in-context learning and ultimately enhance a model’s performance on new tasks remains a significant challenge.

### References

- [1] Triantafyllos Afouras, Effrosyni Mavroudi, Tushar Wang Huiyu Nagarajan, and Lorenzo Torresani. Ht-step: Aligning instructional articles with how-to videos. In Neural Information Processing Systems, 2023. 13
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 2

- [3] Anurag Arnab, Ahmet Iscen, Mathilde Caron, Alireza Fathi, and Cordelia Schmid. Temporal chain of thought: Long-video understanding by thinking in frames, 2025. 3, 13
- [4] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 2
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun

- Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4, 6, 7, 12
- [6] Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. Whisperx: Time-accurate speech transcription of long-form audio. arXiv preprint arXiv:2303.00747, 2023. 4
- [7] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 5
- [8] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. 3
- [9] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Crist´obal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Fei-Fei Li. HourVideo: 1-hour video-language understanding. Advances in Neural Information Processing Systems, 37:53168–53197, 2024. 2
- [10] Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211, 2024. 10
- [11] Yuxiao Chen, Kai Li, Wentao Bao, Deep Patel, Yu Kong, Martin Renqiang Min, and Dimitris N. Metaxas. Learning to localize actions in instructional videos with llm-based multipathway text-video alignment, 2024. 13
- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks, 2024. 2
- [13] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in videollms, 2024. 7
- [14] Gheorghe Comanici and et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. 7
- [15] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9062–9072, 2025. 2
- [16] Nikita Dvornik, Isma Hadji, Ran Zhang, Konstantinos G. Derpanis, Animesh Garg, Richard P. Wildes, and Allan D. Jepson. Stepformer: Self-supervised step discovery and localization in instructional videos, 2023. 13
- [17] Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 961–970, 2015. 2, 13

- [18] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 6, 12

- [19] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2024. 2, 7, 9
- [20] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 13
- [21] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, Long Ma, Xiawu Zheng, Rongrong Ji, Xing Sun, Caifeng Shan, and Ran He. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction, 2025. 10
- [22] Noa Garcia, Mayu Otani, Chenhui Chu, and Yuta Nakashima. Knowit vqa: Answering knowledge-based questions about videos. In Proceedings of the AAAI conference on artificial intelligence, pages 10826–10834, 2020. 13
- [23] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 6, 10
- [24] Sara Ghazanfari, Francesco Croce, Nicolas Flammarion, Prashanth Krishnamurthy, Farshad Khorrami, and Siddharth Garg. Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning. arXiv preprint arXiv:2506.00318, 2025. 3
- [25] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzy´nska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz MuellerFreitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The ”something something” video database for learning and evaluating visual common sense,

2017. 2

- [26] Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, and Si Liu. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26181–26191, 2025. 3, 13
- [27] Tengda Han, Weidi Xie, and Andrew Zisserman. Temporal alignment networks for long-term video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2906–2916, 2022. 4
- [28] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos, 2025. 2, 7, 13
- [29] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you

- need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36:72096–72109, 2023. 2
- [30] Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, and Wenhu Chen. Visualwebinstruct: Scaling up multimodal instruction data through web search. arXiv preprint arXiv:2503.10582, 2025. 5
- [31] Mahnaz Koupaee and William Yang Wang. Wikihow: A large scale text summarization dataset. arXiv preprint arXiv:1810.09305, 2018. 13
- [32] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L. Berg. Tvqa: Localized, compositional video question answering, 2019. 2
- [33] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling, 2021. 13
- [34] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer, 2024. 2, 5, 7
- [35] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 7, 13
- [36] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. Hero: Hierarchical encoder for video+ language omni-representation pre-training. arXiv preprint arXiv:2005.00200, 2020. 13
- [37] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958,

2025. 6, 12

- [38] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection, 2024. 3
- [39] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2024. 7, 10
- [40] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatialtemporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 5, 10
- [41] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment, 2025. 5, 6, 12
- [42] Man Luo, Xin Xu, Yue Liu, Panupong Pasupat, and Mehran Kazemi. In-context learning with retrieved demonstrations for language models: A survey. arXiv preprint arXiv:2401.11624,

2024. 3

- [43] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models, 2024. 3

- [44] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding, 2023. 2
- [45] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630– 2640, 2019. 2, 3, 13
- [46] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943, 2021. 3
- [47] OpenAI. Gpt-4 with vision. https://openai.com/ research/gpt-4v-system-card, 2023. Accessed: 2025-05-21. 10
- [48] OpenAI. Gpt-4 technical report, 2023. 7
- [49] OpenAI. Gpt-4o system card, 2024. 6, 10
- [50] OpenAI. Gpt-4o system card, 2024. 7
- [51] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 2
- [52] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 5
- [53] Hanoona Rasheed, Abdelrahman Shaker, Anqi Tang, Muhammad Maaz, Ming-Hsuan Yang, Salman Khan, and Fahad Shahbaz Khan. Videomathqa: Benchmarking mathematical reasoning via multimodal understanding in videos. arXiv preprint arXiv:2506.05349, 2025. 2, 13
- [54] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag: Retrieval-augmented generation with extreme long-context videos. arXiv preprint arXiv:2502.01549, 2025. 3, 12, 14
- [55] Arka Sadhu, Tanmay Gupta, Mark Yatskar, Ram Nevatia, and Aniruddha Kembhavi. Visual semantic role labeling for video understanding, 2021. 13
- [56] Enxin Song, Wenhao Chai, Weili Xu, Jianwen Xie, Yuxuan Liu, and Gaoang Wang. Video-mmlu: A massive multidiscipline lecture understanding benchmark. arXiv preprint arXiv:2504.14693, 2025. 2, 10
- [57] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning, 2019. 13
- [58] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis, 2019. 2, 5, 13
- [59] Gemini Team and et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 7
- [60] Yannis Tevissen, Khalil Guetari, and Fr´ed´eric Petitpont. Towards retrieval augmented generation over large video libraries. In 2024 16th International Conference on Human System Interaction (HSI), pages 1–4. IEEE, 2024. 3, 14

- [61] Shulin Tian, Ruiqi Wang, Hongming Guo, Penghao Wu, Yuhao Dong, Xiuying Wang, Jingkang Yang, Hao Zhang, Hongyuan Zhu, and Ziwei Liu. Ego-r1: Chain-of-toolthought for ultra-long egocentric video reasoning. arXiv preprint arXiv:2506.13654, 2025. 3
- [62] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024. 2, 6, 10, 12
- [63] Weiyun Wang and et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,

2025. 7

- [64] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena YeungLevy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76. Springer, 2024. 12
- [65] Yan Wang, Yawen Zeng, Jingsheng Zheng, Xiaofen Xing, Jin Xu, and Xiangmin Xu. Videocot: A video chain-ofthought dataset with active annotation tool. arXiv preprint arXiv:2407.05355, 2024. 3, 13
- [66] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 2, 7
- [67] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa:next phase of question-answering to explaining temporal actions, 2021. 13
- [68] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 10
- [69] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840,

- 2024. 10

[70] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. Videollama 3: Frontier multimodal foundation models for image and video understanding,

- 2025. 7

- [71] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision, 2024. 10
- [72] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 10
- [73] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 5, 6, 12
- [74] Yuanhan Zhang, Yunice Chew, Yuhao Dong, Aria Leo, Bo Hu, and Ziwei Liu. Towards video thinking test: A holistic

- benchmark for advanced video reasoning and understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20626–20636, 2025. 2
- [75] Yongheng Zhang, Xu Liu, Ruihan Tao, Qiguang Chen, Hao Fei, Wanxiang Che, and Libo Qin. Vitcot: Video-text interleaved chain-of-thought for boosting video understanding in large language models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 5267–5276,

2025. 3

- [76] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data, 2025. 3, 7
- [77] Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. Mmicl: Empowering vision-language model with multi-modal in-context learning. arXiv preprint arXiv:2309.07915, 2023. 2
- [78] Honglu Zhou, Roberto Mart´ın-Mart´ın, Mubbasir Kapadia, Silvio Savarese, and Juan Carlos Niebles. Procedure-aware pretraining for instructional video understanding, 2023. 13
- [79] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv e-prints, pages arXiv–2406, 2024. 7
- [80] Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. arXiv preprint arXiv:2402.11574, 2024. 3
- [81] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 6, 12
- [82] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations, 2020. 13
- [83] Dimitri Zhukov, Jean-Baptiste Alayrac, Ramazan Gokberk Cinbis, David Fouhey, Ivan Laptev, and Josef Sivic. Crosstask weakly supervised learning from instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3537–3545, 2019. 2, 5, 13

