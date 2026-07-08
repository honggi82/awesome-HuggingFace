## MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos

Xuehai He1 Weixi Feng∗2 Kaizhi Zheng∗1 Yujie Lu∗2 Wanrong Zhu∗2 Jiachen Li∗2 Yue Fan∗1 Jianfeng Wang3 Linjie Li3 Zhengyuan Yang3 Kevin Lin3 William Yang Wang2 Lijuan Wang3 Xin Eric Wang1 1UC Santa Cruz 2UC Santa Barbara 3Microsoft {xhe89,xwang366}@ucsc.edu https://mmworld-bench.github.io/

# arXiv:2406.08407v3[cs.CV]30Jul2024

[Figure 1]

e

F n a

Makeng

|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|
|---|---|---|---|

|[Figure 6]|
|---|

nc

|[Figure 7]|
|---|

Acc o u

nt

g

E-c o m

e

a

orkin

Tra din

nt

m

n

nc e

e

g Taxation

Aerobics

e

F

Supply Chain

m

ntin g

g

Grafiti

w

e

m

Instru

a

- u

a

n

a

- v

d

n

erc e

o

Baking

a

Stock Chart

o

M

W

Basketbal

Econo mics

Com

Footbal

Manage ment

Q

mercial Promotion

[Figure 8]

Painting

Wealth

Management

Moosic Drawing

Investment Physics

And

Voleybal

Dance

Chemistry Biology

Business

Business

Music

Sports & Arts

Sports & Arts

Weather Wildlife Restoration Botany

Gymnastic

Anatomy

Science

Public Health Clinical Medicine Diagnostic Basic Medical Science Assembly Ego-motion Single Object Manipulation Robotics Behavior

Science

Embodied Tasks

EmbodiedTasks

Health &

Engineering

Zoology

Engineering

MedicineHealth&

Medicine

Mycology Geology Nature Math

&

Tech

&

Tech

Game

Game

Geography Animal Behavior Materials Architecture Mechanical

To wer

Engineering

R e al-ti

Defense Ga me

Pharmacy

R ole

wer

m e

F P S

Po

Playin g

AdvenueGame

War Ga me

Fig

Strate gy

&

Ga me

R ac n g G a m e

g

nin

Energy

htin g G a m e

ulture

cs

otics

e

G a

ard

n

ro

G a m

m e

b

gric

G

o

ec

R

|Load|ing [MathJax]/extensions/MathMenu.js|
|---|---|
| |Figure 1:|

A

e

E

MMWorld covers seven broad disciplines and 69 subdisciplines, focusing on the evaluation of multi-faceted reasoning beyond perception (e.g., explanation, counterfactual thinking, future prediction, domain expertise). On the right is a video sample from the Health & Medicine discipline.

### Abstract

Multimodal Language Language Models (MLLMs) demonstrate the emerging abilities of "world models"—interpreting and reasoning about complex real-world dynamics. To assess these abilities, we posit videos are the ideal medium, as they encapsulate rich representations of real-world dynamics and causalities. To this end, we introduce MMWorld, a new benchmark for multi-discipline, multi-faceted multimodal video understanding. MMWorld distinguishes itself from previous video understanding benchmarks with two unique advantages: (1) multi-discipline, covering various disciplines that often require domain expertise for comprehensive understanding; (2) multi-faceted reasoning, including explanation, counterfactual thinking, future prediction, etc. MMWorld consists of a human-annotated dataset to evaluate MLLMs with questions about the whole videos and a synthetic dataset to analyze MLLMs within a single modality of perception. Together, MMWorld encompasses 1,910 videos across seven broad disciplines and 69 subdisciplines, complete with 6,627 question-answer pairs and associated captions. The evaluation includes 2 proprietary and 10 open-source MLLMs, which struggle on MMWorld (e.g., GPT-4V performs the best with only 52.3% accuracy), showing large room for improvement. Further ablation studies reveal other interesting findings such as models’ different skill sets from humans. We hope MMWorld can serve as an essential step towards world model evaluation in videos.

∗Equal Contribution

### 1 Introduction

Foundation models, such as Large Language Models (LLMs) [OpenAI, 2023c; Touvron et al., 2023a; Jiang et al., 2023; Anil et al., 2023] and Multimodal LLMs (MLLMs) [OpenAI, 2023b; Team et al., 2023; Lin et al., 2023a; Li et al., 2023c; Maaz et al., 2024; Chen et al., 2023], have demonstrated remarkable abilities in text and image domains, igniting debates about their potential pathways to Artificial General Intelligence (AGI). This raises a critical question: how well do these models understand the dynamics of the real world? Are they equipped with an inherent World Model [LeCun,

- 2022; Chen et al., 2024; Ha and Schmidhuber, 2018; Xiang et al., 2024] that can understand and reason about the underlying principles and causalities of the dynamic, multimodal world?

Videos, with their rich, dynamic portrayal of the real world, are ideally suited for evaluating the "world modeling" capabilities of MLLMs. Existing video understanding benchmarks [Li et al.,

- 2023d; Ning et al., 2023b; P˘atr˘aucean et al., 2023; Li et al., 2023d], however, fall short in two key perspectives for such evaluations. First, as LeCun et al. [LeCun, 2022] discussed, the world model should be able to (1) estimate missing information about the state of the world not provided by perception, and (2) predict plausible future states of the world. Evaluation of such capabilities requires multi-faceted reasoning beyond perception level, including explaining the video dynamics, counterfactual thinking of alternative consequences, and predicting future activities within videos. Moreover, the multi-discipline nature of the multimodal world necessitates a grasp of diverse fundamental principles—ranging from physics and chemistry to engineering and business. Hence, domain expertise across a variety of disciplines is imperative for a thorough evaluation of a model’s world understanding towards AGI [Morris et al., 2023; Yue et al., 2023].

Therefore, we introduce MMWorld, a multi-discipline multi-faceted multimodal video understanding benchmark to comprehensively evaluate MLLMs’ abilities in reasoning and interpreting real-world dynamics 2. MMWorld encompasses a wide range of disciplines and presents multi-faceted reasoning challenges that demand a combination of visual, auditory, and temporal understanding. It consists of 1,910 videos that span seven common disciplines, including Art & Sports, Business, Science, Health & Medicine, Embodied Tasks, Tech & Engineering, and Games, and 69 subdisciplines (see Figure 1) such as Robotics, Chemistry, Trading, and Agriculture, thereby fulfilling the objective of breadth in discipline coverage. The dataset includes a total of 1,559 question-answer pairs and video captions annotated and reviewed by humans. Meanwhile, for multi-faceted reasoning, MMWorld mainly contains seven kinds of questions focusing on explanation (explaining the phenomenon in videos), counterfactual thinking (answering what-if questions), future prediction (predicting future events), domain expertise (answering domain-specific inquiries), temporal understanding (reasoning about temporal information), and etc. A video example with these four questions from the Health & Medicine discipline is depicted in Figure 1. MMWorld comprises two datasets: a human-annotated dataset for evaluating MLLMs on the whole video and a synthetic dataset designed to analyze MLLMs’ perception within single visual or audio modalities. We evaluate 12 MLLMs that can handle videos or image sequences on MMWorld, including both open-source (e.g., Video-LLaVA-7B [Lin et al., 2023a]) and proprietary models (GPT-4V [OpenAI, 2023b] and Gemini [Team et al., 2023]).

We summarized the contributions and key findings as follows:

- • We introduce MMWorld, a new benchmark designed to rigorously evaluate the capabilities of Multimodal Large Language Models (MLLMs) in world modeling through the realm of video understanding. MMWorld spans a broad spectrum of disciplines, featuring a rich array of question types for multi-faceted reasoning.
- • In addition to the human-annotated dataset, we develop an automatic data collection pipeline, streamlining video content selection and question-answer generation, and construct a wellcontrolled synthetic dataset to analyze MLLMs within single visual or audio modalities.
- • We observe that existing MLLMs still face substantial challenges posed by MMWorld. Even the best performer, GPT-4o, can only achieve a 52.30% overall accuracy, and four MLLMs particularly trained on videos perform worse than random chance.
- • Although there is stll a clear gap between open-source and proprietary models, the best open-source model Video-LLaVA-7B outperforms GPT-4V and Gemini on Embodied Tasks

2Note that MMWorld is not a sufficient testbed for world model evaluation, but we believe overcoming the unique challenges presented in MMWorld is essential and necessary towards comprehensive world modeling.

- Table 1: Comparison between MMWorld and previous benchmarks for real-world video understanding on a variety of criteria. Multi-faced include Explanation (Explain.), Counterfactual Thinking (Counter.), Future Prediction (Future.) and Domain Expertise (Domain.) MMWorld is the first multi-discipline and multitask video understanding benchmark that covers wider reasoning questions, and also included first-party data annotations.

MultiDiscipline

MultiTask

Multi-Faceted Reasoning First-Party

Benchmarks

Annotation Explain. Counter. Future. Domain.

MovieQA [Tapaswi et al., 2016] ✓ ✓ TVQA [Lei et al., 2018] ✓ ✓ ActivityNet-QA [Yu et al., 2019b] ✓ MSVD-QA [Xu et al., 2017] [Xu et al., 2016] ✓ MSRVTT-QA [Xu et al., 2016] ✓ Sports-QA [Li et al., 2024] ✓ ✓ ✓ VaTeX [Wang et al., 2019] ✓ ✓ VALUE [Li et al., 2021] ✓

- Video-Bench [Ning et al., 2023a] ✓ ✓ ✓ MVBench [Li et al., 2023d] ✓ ✓ ✓ Perception Test [P˘atr˘aucean et al., 2023] ✓ ✓ ✓ ✓ MMWorld (Ours) ✓ ✓ ✓ ✓ ✓ ✓ ✓

by a large margin and performs similarly on Art & Sports, where spatiotemporal dynamics play a more crucial role in video understanding. This is further validated with its leading results on the Temporal Understanding question type.

• In our study comparing MLLMs with average humans (non-experts), we notice some correlation between question difficulties as perceived by humans and MLLMs. However, MLLMs present different skill sets than humans in that they can answer reasonable amount of difficult questions that humans completely fail but also struggle at easy questions that humans excel at. This indicates different perception, cognition, and reasoning abilities between MLLMs and humans.

### 2 Related Work

- 2.1 Multimodal Large Language Models (MLLMs) Emerging MLLMs With recent breakthroughs [OpenAI, 2023a; Google, 2023; Touvron et al.,

- 2023a; Chiang et al., 2023; Touvron et al., 2023b; Bai et al., 2023a] in Large Language Models (LLMs), several counterparts in the vision-and-language domain have been proposed [Dai et al., 2023; Liu et al., 2023b,a; Li et al., 2023a; Zhu et al., 2023; Zheng et al., 2023; Bai et al., 2023b], and recently released GPT-4V [OpenAI, 2023b], followed by Gemini Vision family [Team et al., 2023]. Many MLLMs have expanded their capabilities beyond handling only text and image inputs. VideoChat [Li et al., 2023c] leverages the QFormer [Li et al., 2023b] to map visual representations to LLM [Chiang et al., 2023], and performs a multi-stage training pipeline. Otter [Li et al., 2023a] proposes to conduct instruction finetuning based on Openflamingo [Awadalla et al., 2023]. PandaGPT [Su et al., 2023] employs the ImageBind [Han et al., 2023] as the backbone and finetunes it. mPLUGOwl [Ye et al., 2023] introduces an abstractor module to perform visual and language alignment. VideoLLaMA [Zhang et al., 2023a] introduces a frame embedding layer and also leverages ImageBind to inject temporal and audio information into the LLM backend. Chat-UniVi [Jin et al., 2023] uses clustering to do feature fusion. Observing their emerging abilities in multimodal video understanding, we propose MMWorld to evaluate these models’ skills in understanding the dynamics of the real world.

Benchmarking MLLMs To evaluate MLLMs, there is a flourishing of analysis [Liu et al., 2024a; Zhang et al., 2023b; Jiang et al., 2022; Lu et al., 2024; Fan et al., 2024; Cui et al., 2023; Guan et al.,

- 2024; Yu et al., 2023; Fu et al., 2023a] and the establishment of innovative benchmarks such as VisIB-Bench [Bitton et al., 2023] which evaluates models with real-world instruction-following ability given image inputs, MMMU [Yue et al., 2023] designed to access models on college-level imagequestion pairs that span among different disciplines, and VIM [Lu et al., 2023] which challenges the model’s visual instruction following capability. However, these recent analyses and benchmarks only cover the image input, which hinders the evaluation of MLLM’s performance as a world model. Recently, video benchmarks such as Perception Test [P˘atr˘aucean et al., 2023] is proposed to focus

on perception and skills like memory and abstraction. However, it uses scenarios with a few objects manipulated by a person, which limits the variety of contexts. MVBench [Li et al., 2023d] centers on temporal understanding, while MMWorld not only includes temporal reasoning but also evaluates other multi-faceted reasoning abilities.

#### 2.2 Video Understanding Benchmarks

Previous video benchmarks, as shown in Table 1, focus on video understanding tasks, including activity-focused on web videos [Yu et al., 2019a], description-based question answering [Zeng et al., 2017], video completion [Fu et al., 2023b], and video infilling [Himakunthala et al., 2023]. Recently, Video-Bench [Ning et al., 2023b] introduces a benchmark by collecting videos and annotations from multiple existing datasets. LWM [Liu et al., 2024b] collects a large video and language dataset from public books and video datasets and trains a world model that is capable of processing more than millions of tokens. However, modeling millions of tokens is extremely difficult due to high memory cost, computational complexity, and lack of suitable datasets. Mementos [Wang et al.,

- 2024a] builds a benchmark for MLLM reasoning for input image sequences. STAR [Wu et al., 2021] builds a benchmark for situated reasoning in real-world videos. CLEVER [Yi et al., 2020] builds a benchmark containing videos focusing on objects with simple visual appearance. Our contribution, in contrast, presents a new video understanding benchmark designed to evaluate models on several pivotal components crucial for a comprehensive world model. These components encompass interdisciplinary coverage, task diversity, and multifaceted reasoning capabilities—including future prediction, counterfactual thinking, and more—underpinned by original human annotations and integrated domain knowledge.

### 3 The MMWorld Benchmark

The MMWorld benchmark is built on three key design principles: multi-discipline coverage and multi-faceted reasoning. It spans various disciplines that require domain expertise and incorporates diverse reasoning skills such as explanation, counterfactual thinking, and future prediction. The benchmark consists of two parts: a human-annotated dataset and a synthetic dataset. The humanannotated dataset serves as the main test bed to evaluate MLLMs from multiple perspectives. The synthetic dataset contains two subsets, focusing on evaluating MLLMs’ perception behavior from both visual signals and audio inputs, respectively.

#### 3.1 Manual Data Collection

We collect videos from YouTube with the Creative Licence in seven disciplines: Art & Sports (18.5%), Business (12.0%), Science (20.4%), Health & Medicine (12.0%), Embodied Tasks (12.0%%), Tech & Engineering (12.9%), and Game (12.2%). For Art & Sports, 29 videos are collected from the SportsQA dataset [Li et al., 2024]. And for Embodied Tasks, 24 videos are sourced from IKEA Assembly [Ben-Shabat et al., 2021], RT-1 [Brohan et al., 2022], and Ego4D [Grauman et al., 2022] datasets to increase video diversity.

Our manual benchmark collection takes two stages. In the first stage, we conduct a detailed examination of each of the seven primary disciplines to identify a comprehensive range of subdisciplines for inclusion in our benchmark. Our selection of videos is driven by three key principles:

- • The first principle, multi-discipline coverage, emphasizes the requirement for domain knowledge—selecting videos that inherently demand an understanding of specialized content across various disciplines.
- • The second principle, multi-faceted annotation, involves collecting videos that enable the creation of question-answer pairs from multiple perspectives to evaluate world model properties comprehensively.
- • The third principle, temporal information, prioritizes the inclusion of videos that provide meaningful content over time, as understanding temporal information is crucial for grasping world dynamics. This allows models to engage in temporal reasoning. Therefore, answering questions in our dataset requires implicit temporal reasoning, e.g., the model needs to understand temporal information to explain “why does the robot need to do the step shown

[Figure 9]

[Figure 10]

|[Figure 11]|[Figure 12]|[Figure 13]|
|---|---|---|

|[Figure 14]|[Figure 15]|[Figure 16]|
|---|---|---|

Q: How many animals appear in the video? A: Two. There are a horse and a dog

[Figure 17]

[Figure 18]

Attribution Understanding

|[Figure 19]|[Figure 20]|[Figure 21]|
|---|---|---|

Q: What has been changed in the video? A: The bottom drawer has been closed.

Temporal Understanding

Domain Expertise

Procedure Understanding

Q: What is the reason that the lady decides to use the easy frost? A: Because it has no-fuss frosting.

|[Figure 22]|[Figure 23]|[Figure 24]<br><br>[Figure 25]|
|---|---|---|

[Figure 26]

|[Figure 27]|[Figure 28]|[Figure 29]|
|---|---|---|

Future Prediction

Multifaceted Reasoning

Q: What was first added into the milk? A: Cocoa powder.

Q: What will happen next as the price is below the blue and red lines? A: The price will go down.

[Figure 30]

|[Figure 31]<br><br>[Figure 32]| |[Figure 33]|
|---|---|---|

[Figure 34]

|[Figure 35]|[Figure 36]|[Figure 37]|
|---|---|---|

Counterfactual Explanation Thinking

Q: What would happen if the man skipped the step shown in the video? A: The desktop of the coffee table will be upside down, which will make it impossible to mount the legs.

Q: How do the pulleys move when the hands are off the pulley system? A: Two static and two moving upward.

- Figure 2: The questions in MMWorld primarily evaluate seven understanding and reasoning abilities of models. We give one example for each category.

in the video”. We also design a “temporal understanding” question type to explicitly test models’ ability to reason about temporal information (examples can be found in Section F in the Appendix).

During the second stage, our team embark on the task of question annotation. We craft questions that primarily test seven aspects of multimodal video understanding also from the perspective of multi-faceted reasoning: 1) Explanation: Questions ask the model to elucidate the underlying logic or purpose within the video; 2) Counterfactual Thinking: Tests the model’s ability to hypothesize and consider alternative outcomes; 3) Future Prediction: Aims to predict future events based on the current scenario, challenging the model’s foresight; 4) Domain Expertise: Evaluates the model’s depth of knowledge in specific fields, such as how to assemble a coffee table; 5) Temporal Understanding: Assesses the model’s capability to reason about temporal sequences and dynamics; 6) Attribution Understanding: These questions focus on identifying cause-and-effect relationships within the video, including tasks like counting; 7) Procedure Understanding: Tests the model’s ability to comprehend and explain procedural tasks shown in the video. The detailed distribution and examples are shown in Figure 2.

#### 3.2 Automated Data Collection

Understanding real-world dynamics requires models to process both audio and visual modalities. To evaluate MLLMs’ perception abilities in these modalities, we designed an automated data collection pipeline. This pipeline collects targeted videos and generates QA pairs based on either audio or visual information, ensuring the model’s capabilities are assessed independently for each modality. By using information from a single modality to generate QA pairs, our pipeline ensures that the synthetic data remains unbiased regarding input modality.

The synthetic data generation pipeline is illustrated in Figure 3. We employ a systematic approach to gather videos with Creative Commons licenses from YouTube and the extensive YouTube-8M dataset [Abu-El-Haija et al., 2016]. This method ensures a diverse and comprehensive collection of video data, which is important for the robust evaluation of multimodal video understanding models.

Video Collection and Processing We start with the video Query Generator. We start with the same seven disciplines as the manually collected dataset. For each discipline, a set of subdisciplines is defined to encapsulate a wide spectrum of topics, ensuring a diverse and comprehensive dataset. Once the queries are generated, the Video Mapping and Filtering step is initiated. We perform mapping of videos to YouTube-8M and online videos, constrained by a strict time limit of two minutes per query, keeping only the most pertinent videos that satisfy the predefined criteria. Simultaneously, the works in conjunction with the video transcripts to extract key terms and concepts. This iterative process refines the search parameters and enhances the semantic richness of the dataset by identifying and encoding the salient themes present in the videos. The Video Summarization module utilizes

[Figure 38]

- Figure 3: Schematic diagram of the synthetic data generation pipeline in MMWorld. It starts with generating subdiscipline-specific queries, followed by video retrieval from YouTube-8M [Abu-ElHaija et al., 2016] and YouTube. Keyframes are extracted for visual-based QA generation, and videos are transcribed using an ASR module for audio-based QA generation.

t

- Table 2: Key Statistics of the MMWorld Benchmark. The main subset is the human-annotated subset.

- Synthetic Subset I contains generated QA pairs focused exclusively on the audio content, while
- Synthetic Subset II contains QA pairs focused exclusively on the visual content of the video.

Statistics Main Subset Synthetic I Synthetic II #Discipline/#Subdiscipline 7/61 7/51 7/54 #<Video-QA> <417-1,559> <746-2,969> <747-2,099> Avg Video Lengths (s) 102.3 103.4 115.8 Avg #Questions per Video 4.05 3.98 2.81 Avg #Options 3.90 4.00 4.00 Avg Question Length 11.39 15.12 17.56 Avg Option Length 7.27 6.01 5.19 Avg Answer Length 6.42 6.71 5.67 Avg Caption Length 27.00 71.87 82.33

Query-focused video summarization techniques based on Katna3 and UniVTG [Lin et al., 2023b]. This module selects ten representative frames from each video, distilling the essence of the content while preserving the narrative context. This summarization facilitates efficient storage and quicker processing times, which are crucial for large-scale analysis.

QA Generation The final stage in our pipeline is the QA / Caption Generation module, where we leverage the capabilities of GPT-4V to generate accurate and contextually relevant questions and answers, as well as captions, based on the video frames and transcripts. This step not only provides rich annotations for each video but also equips the dataset with a multimodal dimension that supports various downstream tasks such as video QA, captioning, and more.

Quality of the Synthetic Dataset Human evaluators were engaged to ascertain the reasonableness of automatically generated questions and answers, ensuring that the synthetic dataset maintains a high standard of quality and relevance. The findings from this human evaluation phase are detailed in Section D of the Appendix, offering insights into the dataset’s efficacy and the realism of its constructed queries and responses.

Finally, the statistics of automated curated data, which is used for the ablation study, are shown in Table 2. The taxonomy of our dataset is shown in Figure 1. We note that only a portion of the subdisciplines are shown due to space concerns. Please refer to the Appendix for full information.

3https://github.com/keplerlab/katna

- Table 3: MLLM accuracy across diverse disciplines (averaging over three runs). GPT-4V and Gemini Pro lead at most disciplines and achieve the best overall accuracy. The best open-source model Video-LLaVA-7B outperforms them on Embodied Tasks and perform similarly on Art & Sports.

Model

Art&

Business Science

Health& Embodied Tech&

Game Average Sports Medicine Tasks Engineering

Random Choice 25.03 25.09 26.44 25.00 26.48 30.92 25.23 26.31 Proprietary MLLMs

GPT-4o [OpenAI, 2024] 47.87 ±1.47 91.14 ±0.87 73.78 ±2.88 83.33 ±1.47 62.94 ±3.47 75.53 ±2.61 80.32 ±2.05 62.54 ±0.79 Claude-3.5-Sonnet [Anthropic, 2024] 54.58 ±0.45 63.87 ±0.40 59.85 ±1.28 54.51 ±1.28 30.99 ±0.40 58.87 ±0.61 59.44 ±0.68 54.54 ±0.29 GPT-4V [OpenAI, 2023b] 36.17 ±0.58 81.59 ±1.74 66.52 ±1.86 73.61 ±0.49 55.48 ±2.70 61.35 ±1.00 73.49 ±1.97 52.30 ±0.49 Gemini Pro [Team et al., 2023] 37.12 ±2.68 76.69 ±2.16 62.81 ±1.83 76.74 ±1.30 43.59 ±0.33 69.86 ±2.01 66.27 ±2.60 51.02 ±1.35

Open-source MLLMs

Video-LLaVA-7B [Lin et al., 2023a] 35.91 ±0.96 51.28 ±0.87 56.30 ±0.76 32.64 ±0.49 63.17 ±1.44 58.16 ±1.00 49.00 ±3.16 44.60 ±0.58 Video-Chat-7B [Li et al., 2023c] 39.53 ±0.06 51.05 ±0.00 30.81 ±0.21 46.18 ±0.49 40.56 ±0.57 39.36 ±0.00 44.98 ±0.57 40.11 ±0.06 ChatUnivi-7B [Jin et al., 2023] 24.47 ±0.49 60.84 ±1.51 52.00 ±0.73 61.11 ±1.96 46.15 ±2.06 56.74 ±1.33 52.61 ±2.84 39.47 ±0.42 mPLUG-Owl-7B [Ye et al., 2023] 29.16 ±1.62 64.10 ±1.84 47.41 ±3.29 60.07 ±1.30 23.78 ±3.47 41.84 ±5.09 62.25 ±3.16 38.94 ±1.52 PandaGPT-7B [Su et al., 2023] 25.33 ±0.54 42.66 ±3.02 39.41 ±2.67 38.54 ±3.07 35.43 ±0.87 41.84 ±2.79 40.16 ±4.65 32.48 ±0.45 ImageBind-LLM-7B [Han et al., 2023] 24.82 ±0.16 42.66 ±0.99 32.15 ±1.11 30.21 ±1.47 46.85 ±1.14 41.49 ±1.50 41.37 ±0.57 31.75 ±0.14 X-Instruct-BLIP-7B [Panagopoulou et al., 2023] 21.08 ±0.27 15.85 ±0.87 22.52 ±1.11 28.47 ±0.49 18.41 ±1.44 22.34 ±0.87 26.10 ±0.57 21.36 ±0.18 LWM-1M-JAX [Liu et al., 2024b] 12.04 ±0.53 17.48 ±0.57 15.41 ±0.91 20.49 ±0.98 25.87 ±1.98 21.99 ±2.19 11.65 ±3.01 15.39 ±0.32 Otter-7B [Li et al., 2023a] 17.12 ±1.17 18.65 ±0.87 9.33 ±0.36 6.94 ±0.98 13.29 ±1.51 15.96 ±1.74 15.26 ±0.57 14.99 ±0.77 Video-LLaMA-2-13B [Zhang et al., 2023a] 6.15 ±0.44 21.21 ±0.66 22.22 ±1.45 31.25 ±1.70 15.38 ±1.14 19.15 ±1.74 24.90 ±5.93 14.03 ±0.29

- 4 Experiments

#### 4.1 Experimental Settings

In our study, we compare MLLM’s performance on the MMWorld benchmark, including GPT4V [OpenAI, 2023b], Gemini Pro [Team et al., 2023], Video-Chat [Li et al., 2023c], VideoLLaMA [Zhang et al., 2023a], ChatUnivi [Jin et al., 2023], mPLUG-Owl [Ye et al., 2023], Otter [Li et al., 2023a], ImageBind-LLM [Han et al., 2023], PandaGPT [Su et al., 2023], LWM [Liu et al., 2024b], and X-Instruct-BLIP [Panagopoulou et al., 2023]. For both Gemini Pro and GPT-4V, we adhere to the default settings provided by their official APIs. They both take ten image frames extracted from the video content as the input. The Gemini Pro is set to process visual input and configured with safety settings to filter a range of harmful content. The configuration thresholds are set to ‘BLOCK_NONE’. For PandaGPT, we set ‘top_p’ to 0.7 and ‘temperature’ to 0.5. For VideoChat, we set ‘max_frames’ to 100. For X-Instruct-BLIP, the model is implemented using four image frames. We use GPT-4-32K as the judge for judging whether the model answer is correct when it can not mapped to the option letter using the rule-based method. For others, we all use the default setting. All inferences are run on a NVIDIA A6000 workstation. The detailed implementation is given in the Appendix.

#### 4.2 Evaluation

Our dataset includes multiple-choice questions and captions corresponding to each video, enabling tasks such as video question answering and video captioning. We focus on video question answering by evaluating a model’s performance based on its accuracy in selecting the correct answer from the provided options. One challenge lies in reliably parsing the model’s response to map it to one of the predefined choices. To address this, we employ two mapping strategies. We employ two mapping strategies. The first method employs automated scripts to parse the models’ predictions and compare the parsed results with the ground truth, similar to the approach used in [Yue et al., 2023]. The second method involves models freely generating answers, which are then evaluated by GPT-4. Given the question, correct answer, and model’s prediction, GPT-4 returns a True or False judgment. This approach is based on recent works in model evaluation [Maaz et al., 2024; Hsu et al., 2023; Hackl

- et al., 2023; Liu et al., 2023c]. We validated this method with human evaluators, showing an error rate of 4.76% across 189 examples, confirming the effectiveness of GPT-4 as an evaluator. Detailed results for human evaluation and for these two different strategies are provided in Appendix B. In the main paper, all results are evaluated using the second approach.

[Figure 39]

[Figure 40]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 41]

Explanation

0.5

0.7

[Figure 42]

[Figure 43]

0.9

- Figure 4: Results of different MLLMs on multi-faceted reasoning. The detailed performance numbers can be found in the Appendix.

#### 4.3 Main Evaluation Results

We show in Table 3 the main evaluation results of different MLLMs. Among these, GPT-4V emerges as the top performer, closely followed by Gemini Pro. Video-LLaVA also demonstrates strong results, primarily due to the extensive training data which consists of 558K LAION-CCSBU image-text pairs and 702K video-text pairs from WebVid [Bain et al., 2021]. For instruction tuning, datasets were gathered from two sources: a 665K image-text instruction dataset from LLaVA v1.5 and a 100K video-text instruction dataset from Video-ChatGPT [Maaz et al., 2024]. This superior performance may also be attributed to Video-LLaVA’s adoption of CLIP ViT-L/14 trained in LanguageBind [Lin et al., 2023a] as its vision model and the inclusion of a large volume of image-video-text pairings within the training data. On the other hand, models like Otter and LWM perform poorly across most disciplines, possibly due to their weaker backbone and architecture used. Otter uses the LLaMA-7B language encoder and a CLIP ViT-L/14 vision encoder, both of which are frozen, with only the Perceiver resampler module fine-tuned, which may contribute to its lower performance. Additionally, some MLLMs perform even worse than random, highlighting the challenging nature of MMWorld.

#### 4.4 Study on Multi-faceted Reasoning on MMWorld

- Figure 4 illustrates the multi-faceted reasoning performance for each MLLM. GPT-4V emerges as the strongest model across Future Prediction, Domain Expertise, and Attribution Understanding. Closed-source models like GPT-4V and Gemini Pro perform similarly on counterfactual thinking and outperform all others. However, for temporal understanding, Video-LLaVA performs the best. This may be due to its extensive training on large amounts of video-language data, which enhances its spatio-temporal reasoning abilities. This can be also observed in its high scores on the Art & Sports and Embodied Tasks, which involve dense spatio-temporal information, as shown in Table 3. Video-LLaVA’s performance is comparable to GPT-4V and Gemini on explanation tasks, likely because of its two-stage training process and exposure to a large amount of instruction-tuning data in the second stage, which includes similar instructions.

4.5 Study on MLLM Performance at Different Difficulty Levels for Average Humans

- Figure 5a indicate some correlation between the difficulty levels as perceived by humans and the performance of MLLMs. MLLMs generally follow a trend where accuracy decreases as the difficulty level increases, which aligns with human performance patterns. However, the correlation is not perfect, suggesting that while models and humans share some common ground in understanding question difficulty, there are also notable differences in their capabilities. The data reveals that MLLMs exhibit different skill sets compared to humans. As highlighted in Figure 5b, models like GPT-4V can correctly answer expert-level questions that humans often get wrong, particularly in disciplines such as Business and Health & Medicine, where humans often struggle, yet they sometimes falter on easier questions, likely due to the lack of contextual understanding. Notably, discrepancies in disciplines like Art & Sports and Tech & Engineering highlight areas where MLLMs’ performance does not align with human results, suggesting different perception, cognition, and reasoning abilities in handling abstract concepts. These differences suggest that MLLMs can complement human capabilities, offering potential for enhanced task performance by combining the data-driven insights of models with human intuition and contextual knowledge.

0.6

| |
|---|

ChatUnivi Gemini Pro

| |
|---|

0.5

| | | |
|---|---|---|
| | | |

| |
|---|

GPT-4V

| |
|---|

ImageBind-LLM

0.4

LWM

mPLUG-Owl

Otter

0.3

PandaGPT

Video-Chat

0.2

Video-LLAMA

X-Instruct-BLIP

0.1

Easy Medium Hard Expert

(a) Accuracy of MLLMs at difficulty levels.

92 85 84 80 60 87 46 51 89 91 69 63 71 78 77 64 83 71 73 60 26 36 39 32 60 59 61 69

Business Embodied Tasks Game Health & Medicine Science

90

[Figure 44]

80

70

60

50

40

Sports & Arts

30

Tech & Engineering

Easy Medium Hard Expert

(b) GPT-4V results by disciplines at difficulty levels.

- Figure 5: Model performance at different difficulty levels for average humans. Average human difficulty levels are defined by 3 turkers’ performance per question: Easy (3/3 correct answers), medium (2/3 correct), hard (1/3 correct), and expert (0/3 correct).

- Table 4: Performance on Synthetic Subset I (Audio) and II (Visual). Synthetic Subset I contains QAs based solely on the audio content, while Synthetic Subset II focuses exclusively on the visual content of the video. We evaluated four MLLMs processing both audio and visual inputs along with Gemini Pro (for the audio setting, only providing the question).

Art&Sports Business Science Health&Medicine Embodied Tasks Tech&Engineering Game Average Audio Visual Audio Visual Audio Visual Audio Visual Audio Visual Audio Visual Audio Visual Audio Visual

Model

Random Choice 31.59 30.14 31.18 26.58 36.98 32.89 38.74 32.64 32.81 31.25 27.23 32.60 32.01 30.78 32.44 30.91 Video-Chat [Li et al., 2023c] 33.98 32.48 46.47 41.46 41.86 39.15 45.95 36.81 32.81 46.88 37.48 35.91 32.98 46.70 38.82 39.07 ChatUnivi [Jin et al., 2023] 30.03 43.22 30.19 52.85 38.75 54.59 34.76 50.69 20.14 40.63 24.17 46.41 29.98 45.44 31.82 48.44 Video-LLaMA [Zhang et al., 2023a] 30.15 30.23 36.18 33.17 31.33 31.34 30.90 32.78 33.13 30.05 31.18 30.55 20.49 27.20 29.08 30.47 Otter [Li et al., 2023a] 14.22 16.82 16.77 14.24 16.12 17.00 19.82 13.19 10.94 12.50 15.63 12.43 6.65 10.44 12.83 13.41 Gemini Pro [Team et al., 2023] 20.88 61.38 29.43 77.35 30.62 74.26 30.14 81.53 22.57 70.31 18.83 66.22 29.96 65.01 24.45 69.97

#### 4.6 Study on Modality of Perception

We conduct ablations to evaluate MLLMs ability to perceiving the world on the synthetic dataset of MMWorld. With our synthetic dataset, we considered scenarios where only one modality—either audio or visual—is available. Table 4 shows the results which evaluates the model’s ability to interpret spoken language, background noises, and other audio elements without the aid of visual context and the model’s perception ability to operate without any audio input. For the visual perception test, Gemini Pro performed the best, demonstrating its strong ability to process visual information. Interestingly, Video-Chat exhibited better audio perception than ChatUnivi, despite its poorer visual perception. This may be attributed to its use of the Whisper [Radford et al., 2022] speech recognition model. It also explains that in Table 3, Video-Chat outperforms ChatUnivi in the Art & Sports discipline, which requires a greater understanding of music, voice, and background audio. However, in other disciplines such as Science and Health & Medicine, Video-Chat’s performance is significantly poorer.

#### 4.7 Error Analysis

To gain deeper insights into the limitations of MLLMs, we prompted the models to explain the reasoning behind their choices, particularly when errors occurred. Through this analysis, we identified common error patterns and summarized them into seven distinct categories. We conducted a simple test where the same questions that triggered errors in GPT-4V were also posed to other MLLMs. The frequencies of each type of error are presented in Figure 6, as annotated by human evaluators. Detailed qualitative examples of these errors and further analysis are provided in the Appendix.

### 5 Conclusion

Our MMWorld Benchmark represents a significant step forward in the quest for advanced multi-modal language models capable of understanding complex video content. By presenting a diverse array of videos across seven disciplines, accompanied by questions that challenge models to demonstrate explanation, counterfactual thinking, future prediction, and domain expertise, we have created a rigorous testing ground for the next generation of AI. While using LLMs for data generation can introduce hallucination issues, these challenges are manageable and are commonly addressed [Wang

6

Frequency

VideoLLaVA VideoChat ImageBind-LLM PandaGPT ChatUnivi

VideoLLaMA X-Instruct-13B LWM Otter

| |
|---|

4

| |
|---|

| |
|---|

2

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0

QUE AUE VPE HE RE LDK RA Error Type

| |
|---|

- Figure 6: The frequency of different error types across various MLLMs. For each error type, 10 examples were evaluated. Error types are abbreviated as follows: QUE (Question Understanding Error), AUE (Audio Understanding Error), VPE (Visual Perception Error), HE (Hallucination Error), RE (Reasoning Error), LDK (Lack of Domain Knowledge), and RA (Reject to Answer).

- et al., 2024b; Shen et al., 2023]. Another potential risk is the misuse of MLLMs for surveillance or privacy invasion. The ability of models to understand video content and perform reasoning could be exploited to monitor individuals without their consent, leading to serious ethical and legal concerns regarding privacy.

### References

Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, and Sudheendra Vijayanarasimhan. Youtube-8m: A large-scale video classification benchmark. arXiv preprint arXiv:1609.08675, 2016.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023.

Anthropic. Introducing the next generation of Claude. https://www.anthropic.com/news/

claude-3-family, 2024. Accessed: 2024-07-29.

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023a.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023b.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.

Yizhak Ben-Shabat, Xin Yu, Fatemeh Saleh, Dylan Campbell, Cristian Rodriguez-Opazo, Hongdong Li, and Stephen Gould. The ikea asm dataset: Understanding people assembling furniture through actions, objects and pose. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 847–859, 2021.

Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, and Ludwig Schimdt. Visit-bench: A benchmark for vision-language instruction following inspired by real-world use. arXiv preprint arXiv:2308.06595, 2023.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning, 2023.

William Chen, Oier Mees, Aviral Kumar, and Sergey Levine. Vision-language models provide promptable representations for reinforcement learning. arXiv preprint arXiv:2402.02651, 2024.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/.

Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao. Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges. arXiv preprint arXiv:2311.03287, 2023.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.

Yue Fan, Jing Gu, Kaiwen Zhou, Qianqi Yan, Shan Jiang, Ching-Chen Kuo, Xinze Guan, and Xin Eric Wang. Muffin or chihuahua? challenging large vision-language models with multipanel vqa, 2024.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023a.

Tsu-Jui Fu, Licheng Yu, Ning Zhang, Cheng-Yang Fu, Jong-Chyi Su, William Yang Wang, and Sean Bell. Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023b.

Google. Bard - chat based ai tool from google, powered by palm 2. https://bard.google.com/?hl=en, 2023.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large visionlanguage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

Veronika Hackl, Alexandra Elena Müller, Michael Granitzer, and Maximilian Sailer. Is gpt-4 a reliable rater? evaluating consistency in gpt-4 text ratings. arXiv preprint arXiv:2308.02575, 2023.

Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, et al. Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905, 2023.

Vaishnavi Himakunthala, Andy Ouyang, Daniel Rose, Ryan He, Alex Mei, Yujie Lu, Chinmay Sonar, Michael Saxon, and William Yang Wang. Let’s think frame by frame with vip: A video infilling and prediction dataset for evaluating video chain-of-thought, 2023.

Ting-Yao Hsu, Chieh-Yang Huang, Ryan Rossi, Sungchul Kim, C Lee Giles, and Ting-Hao K Huang. Gpt-4 as an effective zero-shot evaluator for scientific figure captions. arXiv preprint arXiv:2310.15405, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023.

Kenan Jiang, Xuehai He, Ruize Xu, and Xin Eric Wang. Comclip: Training-free compositional image and text matching. arXiv preprint arXiv:2211.13854, 2022.

Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046, 2023.

Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1), 2022.

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering. arXiv preprint arXiv:1809.01696, 2018.

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023a.

Haopeng Li, Andong Deng, Qiuhong Ke, Jun Liu, Hossein Rahmani, Yulan Guo, Bernt Schiele, and Chen Chen. Sports-qa: A large-scale video question answering benchmark for complex and professional sports. arXiv preprint arXiv:2401.01505, 2024.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023b.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023c.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. arXiv preprint arXiv: 2311.17005, 2023d.

Linjie Li, Jie Lei, Zhe Gan, Licheng Yu, Yen-Chun Chen, Rohit Pillai, Yu Cheng, Luowei Zhou, Xin Eric Wang, William Yang Wang, et al. Value: A multi-task benchmark for video-and-language understanding evaluation. arXiv preprint arXiv:2106.04632, 2021.

Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023a.

Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified video-language temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2794–2804, 2023b.

Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning. In Proceedings of the International Conference on Learning Representations, 2024a.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024b.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023b.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. Gpteval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634, 2023c.

Yujie Lu, Xiujun Li, William Yang Wang, and Yejin Choi. Vim: Probing multimodal large language models for visual embedded instruction following, 2023.

Yujie Lu, Dongfu Jiang, Wenhu Chen, William Wang, Yejin Choi, and Yuchen Lin. Wildvision arena: Benchmarking multimodal llms in the wild, February 2024. URL https://huggingface.co/ spaces/WildVision/vision-arena/.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024.

Meredith Ringel Morris, Jascha Sohl-dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement Farabet, and Shane Legg. Levels of agi: Operationalizing progress on the path to agi. arXiv preprint arXiv:2311.02462, 2023.

Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language

- models. arXiv preprint arXiv:2311.16103, 2023a.

Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language

- models. arXiv preprint arXiv:2311.16103, 2023b.

OpenAI. Gpt-4: Technical report. arXiv preprint arXiv:2303.08774, 2023a. OpenAI. Gpt-4v(ision) system card. https://openai.com/research/gpt-4v-system-card, 2023b. OpenAI. Gpt-4 technical report, 2023c. OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024. Accessed: 2024-

07-29.

Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, and Juan Carlos Niebles. X-instructblip: A framework for aligning x-modal instruction-aware representations to llms and emergent cross-modal reasoning. arXiv preprint arXiv:2311.18799, 2023.

Viorica P˘atr˘aucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=HYEGXFnPoq.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, C. McLeavey, and I. Sutskever. Robust speech recognition via large-scale weak supervision. International Conference on Machine Learning, 2022. doi: 10.48550/arXiv.2212.04356.

Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. "do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models. arXiv preprint arXiv: 2308.03825, 2023.

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.

Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. Movieqa: Understanding stories in movies through question-answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4631–4640, 2016.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4581–4591, 2019.

Xiyao Wang, Yuhang Zhou, Xiaoyu Liu, Hongjin Lu, Yuancheng Xu, Feihong He, Jaehong Yoon, Taixi Lu, Gedas Bertasius, Mohit Bansal, et al. Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences. arXiv preprint arXiv:2401.10529,

- 2024a.

Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. Do-not-answer: Evaluating safeguards in LLMs. In Yvette Graham and Matthew Purver, editors, Findings of the Association for Computational Linguistics: EACL 2024, pages 896–911, St. Julian’s, Malta, March

- 2024b. Association for Computational Linguistics. URL https://aclanthology.org/2024. findings-eacl.61.

Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Jiannan Xiang, Guangyi Liu, Yi Gu, Qiyue Gao, Yuting Ning, Yuheng Zha, Zeyu Feng, Tianhua Tao, Shibo Hao, Yemin Shi, Zhengzhong Liu, Eric P. Xing, and Zhiting Hu. Pandora: Towards general world model with natural language actions and video states. 2024.

Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In IEEE International Conference on Computer Vision and Pattern Recognition (CVPR). IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), June 2016. URL https://www.microsoft.com/en-us/research/publication/ msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.

Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B. Tenenbaum. CLEVRER: collision events for video representation and reasoning. In ICLR, 2020.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynetqa: A dataset for understanding complex web videos via question answering. In AAAI, pages 9127–9134, 2019a.

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 9127–9134, 2019b.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.

Kuo-Hao Zeng, Tseng-Hung Chen, Ching-Yao Chuang, Yuan-Hong Liao, Juan Carlos Niebles, and Min Sun. Leveraging video descriptions to learn video question answering. Proceedings of the AAAI Conference on Artificial Intelligence, 31(1), Feb. 2017. doi: 10.1609/aaai.v31i1.11238. URL https://ojs.aaai.org/index.php/AAAI/article/view/11238.

Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023a.

Xinlu Zhang, Yujie Lu, Weizhi Wang, An Yan, Jun Yan, Lianke Qin, Heng Wang, Xifeng Yan, William Yang Wang, and Linda Ruth Petzold. Gpt-4v(ision) as a generalist evaluator for visionlanguage tasks, 2023b.

Kaizhi Zheng, Xuehai He, and Xin Eric Wang. Minigpt-5: Interleaved vision-and-language generation via generative vokens. arXiv preprint arXiv:2310.02239, 2023.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

### A Overview of the Appendix

We host the project website on https://mmworld-bench.github.io/. The benchmark and code implementations can be found at https://github.com/eric-ai-lab/MMWorld. The link to Croissant metadata record documenting the dataset/benchmark available for viewing and downloading is available at https://github.com/eric-ai-lab/MMWorld/blob/main/data/croissanta_ hf_data.json. This Appendix is organized as follows:

- • Section B contains additional experimental results;
- • Section C contains the implementation details;
- • Section D contains the settings and results from human evaluations;
- • Section E contains the error analysis;
- • Section F contains the data examples from MMWorld;
- • Section G contains additional data statistics of MMWorld;
- • Section H contains the datasheet of MMWorld;
- • Section I contains the author statement, licence, and maintenance plan.

### B Additional Results

- B.1 Results Across Different Seed for Each Model In Table 5, we show detailed results using three different seeds for each evaluated models.

- Table 5: Detailed results of model performance, measured as accuracy percentages across diverse disciplines for three runs. The random choice baseline involves shuffling candidate answers for each video question before consistently selecting answer ‘a’. GPT-4V and Gemini Pro utilize 10 image frames extracted from the video content.

Model

Art&

Business Science

Health& Embodied Tech&

Game Average Sports Medicine Tasks Engineering

- GPT-4o-seed 1 [OpenAI, 2024] 47.10 92.31 75.11 81.25 65.03 72.34 78.31 62.22
- GPT-4o-seed 2 [OpenAI, 2024] 46.58 90.91 69.78 84.38 65.73 75.53 83.13 61.77
- GPT-4o-seed 3 [OpenAI, 2024] 49.94 90.21 76.44 84.38 58.04 78.72 79.52 63.63

- Claude-3.5-seed 1 [Anthropic, 2024] 54.32 64.34 59.11 53.12 30.77 59.57 59.04 54.27
- Claude-3.5-seed 2 [Anthropic, 2024] 54.32 63.64 61.33 54.17 30.77 58.51 59.04 54.52
- Claude-3.5-seed 3 [Anthropic, 2024] 55.10 63.64 59.11 56.25 31.47 58.51 60.24 54.84

- GPT-4V-seed 1 [OpenAI, 2023b] 36.90 79.72 64.00 73.96 51.75 60.64 71.08 51.64
- GPT-4V-seed 2 [OpenAI, 2023b] 35.48 83.92 68.44 73.96 58.04 60.64 75.90 52.79
- GPT-4V-seed 3 [OpenAI, 2023b] 36.13 81.12 67.11 72.92 56.64 62.77 73.49 52.47

- Gemini Pro-seed 1 [Team et al., 2023] 40.90 79.72 60.44 78.12 43.36 71.28 65.06 52.92
- Gemini Pro-seed 2 [Team et al., 2023] 35.10 75.52 63.11 75.00 44.06 71.28 69.88 50.16
- Gemini Pro-seed 3 [Team et al., 2023] 35.35 74.83 64.89 77.08 43.36 67.02 63.86 49.97

- Video-LLaVA-seed 1 [Lin et al., 2023a] 34.58 51.05 57.33 32.29 61.54 57.45 50.60 43.94
- Video-LLaVA-seed 2 [Lin et al., 2023a] 36.77 52.45 56.00 32.29 65.03 57.45 51.81 45.35
- Video-LLaVA-seed 3 [Lin et al., 2023a] 36.39 50.35 55.56 33.33 62.94 59.57 44.58 44.52

- Video-Chat-seed 1 [Li et al., 2023c] 39.48 51.05 30.67 46.88 39.86 39.36 44.58 40.03
- Video-Chat-seed 2 [Li et al., 2023c] 39.48 51.05 30.67 45.83 41.26 39.36 45.78 40.15
- Video-Chat-seed 3 [Li et al., 2023c] 39.61 51.05 31.11 45.83 40.56 39.36 44.58 40.15

- mPLUG-Owl-seed 1 [Ye et al., 2023] 31.35 65.73 45.78 61.46 28.67 48.94 65.06 41.05
- mPLUG-Owl-seed 2 [Ye et al., 2023] 28.65 65.03 44.44 58.33 21.68 37.23 57.83 37.52
- mPLUG-Owl-seed 3 [Ye et al., 2023] 27.48 61.54 52.00 60.42 20.98 39.36 63.86 38.23

- ChatUnivi-seed 1 [Jin et al., 2023] 24.13 60.14 52.00 62.50 48.95 56.38 56.63 39.77
- ChatUnivi-seed 2 [Jin et al., 2023] 25.16 62.94 51.11 62.50 44.06 58.51 50.60 39.77
- ChatUnivi-seed 3 [Jin et al., 2023] 24.13 59.44 52.89 58.33 45.45 55.32 50.60 38.87

- PandaGPT-seed 1 [Su et al., 2023] 26.06 44.06 38.22 41.67 35.66 39.36 42.17 32.97
- PandaGPT-seed 2 [Su et al., 2023] 24.77 45.45 36.89 34.38 34.27 40.43 44.58 31.88
- PandaGPT-seed 3 [Su et al., 2023] 25.16 38.46 43.11 39.58 36.36 45.74 33.73 32.58

- ImageBind-LLM-seed 1 [Han et al., 2023] 24.77 41.96 30.67 31.25 46.85 43.62 40.96 31.62
- ImageBind-LLM-seed 2 [Han et al., 2023] 25.03 41.96 32.44 31.25 45.45 40.43 40.96 31.69
- ImageBind-LLM-seed 3 [Han et al., 2023] 24.65 44.06 33.33 28.12 48.25 40.43 42.17 31.94

- X-Instruct-BLIP-seed 1 [Panagopoulou et al., 2023] 21.42 14.69 22.22 29.17 16.78 21.28 26.51 21.23
- X-Instruct-BLIP-seed 2 [Panagopoulou et al., 2023] 20.77 16.78 24.00 28.12 20.28 22.34 25.30 21.62
- X-Instruct-BLIP-seed 3 [Panagopoulou et al., 2023] 21.03 16.08 21.33 28.12 18.18 23.40 26.51 21.23

- LWM-seed 1 [Liu et al., 2024b] 11.35 18.18 16.44 19.79 24.48 24.47 10.84 15.20
- LWM-seed 2 [Liu et al., 2024b] 12.13 17.48 15.56 19.79 24.48 22.34 8.43 15.14
- LWM-seed 3 [Liu et al., 2024b] 12.65 16.78 14.22 21.88 28.67 19.15 15.66 15.84

- Otter-seed 1 [Li et al., 2023a] 18.45 19.58 8.89 8.33 14.69 15.96 14.46 15.84
- Otter-seed 2 [Li et al., 2023a] 17.29 17.48 9.33 6.25 13.99 18.09 15.66 15.14
- Otter-seed 3 [Li et al., 2023a] 15.61 18.88 9.78 6.25 11.19 13.83 15.66 13.98

- Video-LLaMA-seed 1 [Zhang et al., 2023a] 5.55 21.68 24.00 29.17 15.38 21.28 18.07 13.66
- Video-LLaMA-seed 2 [Zhang et al., 2023a] 6.58 20.28 20.44 31.25 13.99 17.02 32.53 14.05
- Video-LLaMA-seed 3 [Zhang et al., 2023a] 6.32 21.68 22.22 33.33 16.78 19.15 24.10 14.37

Table 6: Performance of different set of turkers

Model

Art&

Business Science

Health& Embodied Tech&

Game& Average Sports Medicine Tasks Engineering

- Turker Set 1 25.224 39.860 32.444 40.625 51.049 50.000 40.964 33.227
- Turker Set 2 30.452 46.154 35.556 42.708 53.846 51.064 46.988 37.652
- Turker Set 3 26.710 41.958 36.889 46.875 53.147 42.553 38.554 34.830

- B.2 Results from Amazon Turkers

Table 6 presents the evaluation results from three sets of Amazon Turkers across various disciplines. The results indicate that there is slightly variability in performance across different human evaluators.

- B.3 Results for the Two Different Evaluation Strategies

- In Table 7, we give additional evaluation results for different MLLMs evaluated in this paper. For closed-source models, the evaluation pipeline is the one used in the main paper, which involves utilizing GPT-4V as a judger. The process consists of presenting GPT-4V with the question, a corresponding answer generated by the baseline model, and the set of possible options. GPT-4V then assesses whether the model-generated answer is accurate within the given context; Another is open-ended generation where we employ a two-step methodology. We first prompt each model to do

Table 7: Performance of different MLLMs across different disciplines.

Art&

Health& Embodied Tech&

Model

Business Science

Average

Sports Medicine Tasks Engineering Video-Chat (Open-ended) [Li et al., 2023c] 27.484 9.091 18.137 10.417 29.371 19.149 22.887 Video-Chat [Li et al., 2023c] 39.355 48.951 31.863 45.833 39.161 38.298 39.588 Video-LLaMA (Open-ended) [Zhang et al., 2023a] 5.419 27.972 24.020 31.250 11.816 15.957 16.096 Video-LLaMA [Zhang et al., 2023a] 27.355 31.469 31.373 48.958 16.084 28.723 28.729 ChatUnivi (Open-ended) [Jin et al., 2023] 21.161 61.538 42.157 61.458 30.070 37.234 32.646 ChatUnivi [Jin et al., 2023] 12.387 58.042 50.000 60.417 30.070 43.617 29.072 Otter (Open-ended) [Li et al., 2023a] 37.677 32.867 37.255 32.292 22.378 27.660 34.639 Otter [Li et al., 2023a] 17.677 16.783 12.255 5.208 17.483 15.957 15.876 ImageBind-LLM (Open-ended) [Han et al., 2023] 3.355 3.497 14.706 10.417 21.678 18.085 8.179 ImageBind-LLM [Han et al., 2023] 23.742 34.965 51.471 33.333 48.951 56.383 33.952 PandaGPT (Open-ended) [Su et al., 2023] 22.581 16.084 24.020 21.875 19.580 21.277 21.718 PandaGPT [Su et al., 2023] 27.613 44.056 39.706 25.000 40.559 21.277 31.615 LWM (Open-ended) [Liu et al., 2024b] 16.000 20.979 14.706 16.667 19.580 20.213 16.976 LWM [Liu et al., 2024b] 16.387 18.182 18.137 19.792 22.378 21.277 17.938 X-Instruct-BLIP (Open-ended) [Panagopoulou et al., 2023] 3.613 11.888 14.706 25.000 17.483 13.830 9.416 X-Instruct-BLIP [Panagopoulou et al., 2023] 19.355 13.287 22.549 29.167 18.881 14.894 19.519

Table 8: Detailed results of different MLLMs on multi-faceted reasoning.

Counterfactual Future Domain Attribution Temporal Thinking Prediction Expertise Understanding Understanding

Model Explanation

Proprietary Models

GPT-4V 44.90 64.90 78.59 61.07 59.61 27.17 Gemini Pro 48.58 65.49 65.45 53.87 43.92 24.65

Open-source Models

Video-LLaVA-7B 42.46 42.55 64.96 47.86 36.86 34.45 VideoChat-7B 41.66 43.73 45.74 40.95 30.59 25.77 ImageBind-LLM-7B 29.51 26.86 50.61 33.93 34.90 19.89 PandaGPT-7B 29.55 37.45 46.47 33.93 26.27 28.01 ChatUnivi-7B 33.91 48.82 61.80 45.95 33.33 22.97 VideoLLaMA-2-13B 10.55 23.92 25.30 16.31 8.63 6.16 X-Instruct-BLIP-7B 23.05 15.29 27.25 21.07 24.31 11.20 LWM-1M-JAX 11.62 18.82 30.66 17.98 21.57 7.00 Otter-7B 16.91 10.98 15.82 13.10 17.65 9.52 mPLUG-Owl-7B 35.20 49.61 55.47 47.74 24.71 20.17

open-ended generation. Subsequently, we prompt the model to align its generative response with one of the predefined options: ‘a’, ‘b’, ‘c’, or ‘d’.

#### B.4 Detailed Results on Multi-faceted Reasoning

- In Table 8, we give detailed performance numbers of different MLLMs on multi-faceted reasoning corresponding to Figure 4 in the main paper.

### C Implementation Details

We use the optimum number of video frames and report the performance in the main paper. The numbers of the sampled frames are 10 for GPT-4V/o and Gemini Pro, 8 for Video-LLaVA, 32 for ChatUniVi. For closed-source models, for both Gemini Pro and GPT-4V, we use the default settings provided by their official APIs. We use Katna 4 to extract key video frames as input to these two models. The Gemini Pro is set to process visual input and configured with safety settings to filter a range of harmful content. The configuration thresholds are set to ‘BLOCK_NONE’. For PandaGPT, we set ‘top_p’ to 0.7, and ‘temperature’ to 0.5. For VideoChat, we set ‘max_frames’ to 100. For LWM, we use the LWM-Chat-1M variant. For X-Instruct-BLIP, the model is implemented using four image frames. For Otter, we use the video variant. We use GPT-4-32K as the judge for judging whether the model answer is correct when it can not mapped to the option letter using the rule-based method. The prompt provided to GPT-4-32K is structured as follows: "I will present a response from a question-answering model alongside several answer options. Your task is to evaluate the response and determine which of the following

4https://github.com/keplerlab/katna

[Figure 45]

Figure 7: The interface of using Amazon Mechanical Turk to do human evaluation.

Table 9: Category-wise and overall error rates Category Incorrect/Total Error Rate (%) Sports & Arts 5/62 8.06 Health & Medicine 2/7 28.57 Science 1/52 1.92 Robotics 0/12 0.00 Business 0/10 0.00 Tech & Engineering 1/46 2.17 Overall 9/189 4.76

##### options it most closely aligns with, denoting the most similar option by its corresponding letter (a, b, c, or d).".

Query Generation in Synthetic Data Generation Pipeline For the discipline of Science, queries are generated for subdisciplines such as Geography, Chemistry, Wildlife Restoration, Mycology, Nature, Physics, Weather, Zoology, Math, Botany, Biology, and Geology. In the Tech & Engineering discipline, our queries span across Electronics, Animal Behavior, Mechanical Engineering, Energy & Power, Architecture, Agriculture, Nature, Physics, Robotics, Woodworking, and Gardening. The Sports & Arts discipline encompasses a broad range of cultural and physical activities, including Music, Drawing and Painting, Football, Volleyball, Aerobic Gymnastics, Basketball, Instrument, Baking, Dance, Woodworking, Graffiti, Anatomy, and additional Music-related topics. Embodied Tasks are represented through queries for Assembly, Ego-motion, and Single Object Manipulation, focusing on the interaction between agents and their physical environment. The Health & Medicine discipline is segmented into Pharmacy, Public Health, Clinical Medicine, and Basic Medical Science, reflecting the multifaceted nature of healthcare and medical studies. The Business discipline is stratified into fundamental areas such as accounting, finance, management, marketing, and economics, each representing key facets of the commercial and economic world. Lastly, the Game discipline consists of Role Playing Game, First Person Shooting game, Racing Game, Adventure Game, Real-Time Strategy Game, Tower Defense game, and Fighting Game.

Each generated query retrieves relevant video content, which is then filtered and processed to align with the specific needs of our research objectives. Videos that meet our criteria in terms of content, length, and quality are downloaded and incorporated into our dataset, forming the basis for subsequent analysis and model training.

[Figure 46]

Figure 8: Human evaluation interface for GPT judger.

### D Human Evaluation

#### D.1 Quality of Data

We hired Amazon Mechanical Turk to do human evaluation on the data with the results shown in Table 6. Workers were required to have completed more than 1000 Human Intelligence Tasks (HITs) and have an HIT approval rate greater than 95% to qualify for our tasks. We show in Figure 7 the human evaluation interface on the generated data. Each worker was compensated 0.20 for completing an assignment. This amount was determined based on the estimated time and effort required to complete each task. We set the number of unique workers per task to 3 to collect diverse perspectives while avoiding redundancy. Workers were given 1 hour to complete each assignment. This time frame was chosen to enable thoughtful responses from workers.

We also hired students from campus to do human evaluation on subset of the data. The results are shown in Table 10. The performance of the human evaluators did not surpass that of GPT-4V and Gemini-Pro. This outcome underscores the challenging nature of the dataset, which often necessitates specialized domain knowledge that our evaluators—primarily non-experts—found demanding. These

[Figure 47]

- Figure 9: Examples from MMWorld in the Embodied Tasks discipline.

[Figure 48]

[Figure 49]

|[Figure 50]<br><br>[Figure 51]|
|---|

[Figure 52]

[Figure 53]

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

- Figure 10: Examples from MMWorld in the Tech & Engineering discipline.

results highlight the complexity of the questions and the potential necessity for discipline-specific understanding to achieve high accuracy

#### D.2 Quality of Using GPT as the Judger

For a comprehensive assessment of GPT-4V’s accuracy when using it as the judger, we devised a human evaluation protocol also resort to Amazon Mechanical Turk, as visualized in Figure 8. The evaluators present a series of statements derived from the video, and GPT-4V is tasked with selecting the most accurate answer from a set of multiple-choice questions. Through this interface, human evaluators can efficiently gauge GPT-4V’s performance across different types of questions—when using it as the judger.

The results obtained from this human evaluation process are shown in Table 9, across 189 examples, there are only 9 incorrect ones with the error rate of 4.76%, validating the effectiveness of using

- GPT-4V as the judger.

### E Error Analysis

In this section, we delve into the analysis of errors from evaluated MLLMs. We summarized error types as follows:

[Figure 57]

[Figure 58]

[Figure 59]

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

[Figure 64]

- Figure 11: Examples from MMWorld in the Science discipline.

[Figure 65]

[Figure 66]

[Figure 67]

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

[Figure 72]

- Figure 12: Examples from MMWorld in the Business discipline.

Question Understanding Error (QUE): Models misinterpret the question’s intent, such as misunderstanding how a pendulum’s period would change if a condition in the scenario is altered.

Audio Understanding Error (AUE): Models fail to interpret audio cues correctly, shown by their failure to recognize blue and red lines on a stock chart.

Visual Perception Error (VPE): There is a misinterpretation of visual content, leading to incorrect assumptions about the visual data presented in the video.

Hallucinations (HE): Models generate content or details that are not present in the actual data, essentially ‘hallucinating’ information.

Reasoning Error (RE): Models demonstrate a lack of logical reasoning, leading to incorrect conclusions based on the given data.

Lack of Domain Knowledge (LDK): Models show an inability to answer questions that require specific domain expertise, indicating a gap in their knowledge.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

[Figure 81]

Figure 13: Examples from MMWorld in the Arts & Sports discipline.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

- Figure 14: Examples from MMWorld of explicit temporal understanding and implicit temporal understanding (e.g., in explanation).

Reject to Answer (RA): An example of this error was observed when the model was asked to select an answer regarding the outcome of an experiment involving liquid nitrogen. Instead of choosing an option, the model provided an unrelated response concerning a light bulb, indicating either a misunderstanding or a cautious approach due to the potential for the question to be interpreted as pertaining to a sensitive topic, which can trigger content filters focused on safety and compliance policies.

We show in Figure 15, 16, 17, 18 some error cases of Question Understanding Error, Audio Understanding Error, Visual Perception Error, Hallucinations, Reasoning Error, Lack of Domain Knowledge, and Reject to Answer respectively from MLLMs evaluated on MMWorld.

### F Data Examples

We show in Figure 9, 10, 11, 12, 13, 14 some additional examples from MMWorld.

Table 10: Comparison of Human Evaluation on subset of 75 videos.

Art&

Health& Embodied Tech&

Model

Average Sports Medicine Tasks Engineering

Business Science

Human Evaluation 31.183 59.782 42.103 48.858 56.429 50.134 43.758 GPT-4V [OpenAI, 2023b] 30.399 89.203 68.731 80.059 38.432 69.108 48.793 Gemini-Pro [Team et al., 2023] 28.745 80.909 69.425 80.023 50.987 80.479 48.083

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

|Q: How many eggs can a frog lay at once?<br><br>A: I'm sorry, but the images provided are of mushrooms in a forest setting, which do not provide any information about the egg-laying habits of frogs.|
|---|

[Figure 93]

Lack of Domain Knowledge

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

|Q: How do the pulleys move when the hands are off the pulley system?<br><br>A: When the hands are removed from the pulley system, as shown in the series of images, there is no force being applied to any of the pulleys to cause movement. Therefore, all four pulleys remain static.|
|---|

Reasoning Error

- Figure 15: Error pattern of lack of domain knowledge and reasoning error. In the first case, the model does not give the correct answer because of lacking the domain knowledge. In the second case, the model makes the wrong reasoning.

### G Additional Data Statistics

For human annotated dataset, the length of each video was capped at approximately two minutes. The statistical distribution of the disciplines within the dataset for this part is as follows:

- • Sports & Arts: The subset that consists of 77 videos, showcasing a vibrant collection that covers a wide range of topics from athletic endeavors to various forms of artistic expression.
- • Science: A subset of 75 videos, which delves into the empirical world of scientific inquiry, spanning a multitude of specializations from fundamental physics to advanced biological studies.
- • Tech & Engineering: Encompassing 54 videos, this segment captures the cutting-edge advancements and foundational concepts that drive innovation and infrastructure in the modern world.
- • Embodied Tasks: With 50 videos, the dataset provides a focused insight into the dynamic field of Embodied Tasks, highlighting the intersection of AI, mechanics, and automation.
- • Health & Medicine: This essential discipline is well-represented with 50 videos, offering perspectives on medical breakthroughs, healthcare practices, and life sciences.
- • Business: This discipline includes 50 videos, reflecting on the multifaceted nature of commerce, from economics to management sciences.
- • Game: This discipline includes 51 videos, reflecting various aspects of gaming.

Altogether, the MMWorld Benchmark’s diversity is visually encapsulated in Figure 19, which delineates the distribution of videos across 61 subdisciplines. The horizontal bar chart provides a

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

|Q: What would happen to the pendulum period if the professor sit up straight on the ball instead?<br><br>A: The pendulum period would not afford it.|
|---|

Question Understanding Error

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

|Q: What would happen if the man skipped the step shown in the video? A: If the man skipped the step of rolling the blue tube on the floor to uncover the red box, he would likely have encountered difficulty in accessing the contents of the box. This step is important because it helps the man to create a flat surface to work with the red box, ensuring that the box is visible and accessible.|
|---|

Visual Perception Error, Hallucinations

- Figure 16: Error pattern of question understanding, visual perceptron error, and hallucinations. In the first case, the model does not understand the question correctly where the model fails to accurately discern the query regarding the pendulum’s period. In the second scenario, the model erroneously identifies objects within the visual input, leading to the hallucination of non-existent elements, such as a red box.

quantified representation of the dataset’s range, reflecting the careful curation process that has gone into ensuring breadth across various knowledge areas.

The world we live in is rich with both audio and visual information, and effective world modeling requires an understanding of how these modalities interact and convey meaning. To achieve this, we annotated additional attributes such as "Requires Audio," "Requires Video," and "Question Only." These annotations help determine whether correctly answering a question necessitates audio information, visual cues from the video, or can be addressed based solely on the question itself. By doing so, we ensure that our benchmark tests the full spectrum of multimodal comprehension, reflecting the complex, sensory-rich environment in which real-world understanding takes place. The statistics of these annotations are shown in Figure 20.

### H Datasheets

- H.1 Motivation

For what purpose was the dataset created?

To introduce a multi-discipline multi-faceted multimodal video understanding benchmark to comprehensively evaluate MLLMs’ abilities in reasoning and interpreting real-world dynamics.

Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

The dataset is created by authors from UCSC, UCSB, and Microsoft. Who funded the creation of the dataset? UCSC, UCSB, and Microsoft Azure.

- H.2 Composition

What do the instances that comprise the dataset represent? (e.g., documents, photos, people, countries)

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

|Q: What if we replace the boiling water with iced water? The question will trigger the content filter and there will be no answer|
|---|

Reject to Answer

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

|Q: What does the blue and red lines in the stock chart represents?<br><br>A: The blue and red lines in the stock chart represent the support and resistance levels, respectively. The support level is where the stock price tends to stop falling, while the resistance level is where the stock price tends to stop rising. In the video, the blue line indicates a level of support, while the red line indicates a level of resistance.|
|---|

Audio Understanding Error

- Figure 17: Error pattern of reject to answer and audio understanding error. In the first case, the model will reject to answer because of the dangerous action showing in the video. In the second case, the model does not understand the audio correctly as the answer to the question is given in the audio of the video.

Videos along with captions and question/answer pairs. How many instances are there in total (of each type, if appropriate)?

6,627 instances. The data distribution over different types can be found in Figure 2 of the main paper. Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

Yes. Is there a label or target associated with each instance? Yes. Is any information missing from individual instances? No.

Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

N/A. Are there recommended data splits (e.g., training, development/validation, testing)? The MMWorld is used for evaluation purpose only. Are there any errors, sources of noise, or redundancies in the dataset? No.

Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

Yes. Does the dataset contain data that might be considered confidential? No.

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

|Q: How many capacitors are there in the final circuit? A: There are several capacitors in the final circuit. We can see at least 14 of them in the video.|
|---|

Visual Perception Error

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

|Q: What could be the three gases in order?<br><br>A: I'm unable to provide assistance as the request requires information to be extracted from images/videos which is beyond my current capabilities.|
|---|

Visual Perception Error, Lack of Domain Knowledge

- Figure 18: Error pattern due to visual perception inaccuracies and insufficient domain knowledge. The first case demonstrates a visual perception error where the model incorrectly identifies the number of capacitors present. The second case showcases a compound error where the model not only fails to discern the colors indicative of different gases but also lacks the domain knowledge necessary to infer their identity correctly.

Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No.

- H.3 Collection Process The data collection process is described in Section 3 of the main paper.
- H.4 Preprocessing/cleaning/labeling

Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values

We extract video frames from collected videos in automatically generated.

Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

Yes. The raw video urls are given. Is the software that was used to preprocess/clean/label the data available? Yes. The source code can be found in https://github.com/eric-ai-lab/MMWorld.

- H.5 Uses

#### Has the dataset been used for any tasks already? Yes. We have used the dataset to evaluate video question answering. Is there a repository that links to any or all papers or systems that use the dataset? Yes. The GitHub repository https://github.com/eric-ai-lab/MMWorld here.

Gardening

| | | | |
|---|---|---|---|
| | | | |

Basketball

Drawing And Painting

Single Object Manipulation

Electroincs

Physics

Architecture

Accounting

Instrument

Mechanical Engineering

Energy & Power

Botany

Geography

First Person Shooting Game

Wealth Management

Clinical Medicine

Mycology

Anatomy

Materials

Music

Taxation

Trading

Zoology

Football

Gymnastic

Role Playing Game

Quantitative Finance

Supply Chain Management

Chemistry

Basic Medical Science

Weather

Fighting Game

Robotics

Adventure Game

Wildlife Restoration

Commercial Promotion

Nature

Agriculture

Math

Animal Behavior

Graffiti

Marketing

War

Baking

Tower Defense Game

E-commerce

Robotics Behavior

Ego-motion

Stock Chart

Management

Volleyball

Finance

Investment

Economics

Moosic

Biology

Assembly

Pharmacy

Racing Game

Public Health

Diagnostic

Geology

Manual Collected Subset

Aerobic Gymnastics

- Synthetic Subset I
- Synthetic Subset II

Dance

Woodworking

Electronics

Real-time Strategy Game

0 10 20 30 40 50 60 70 Number Of Videos

- Figure 19: The number of videos per subdiscipline in MMWorld. Each horizontal bar indicates the quantity of videos corresponding to a subdiscipline, showcasing the dataset’s diversity and coverage across various domains of knowledge. Synthetic Subset I is collected with audio-only data and Synthetic Subset II is collected with visual-only data.

What (other) tasks could the dataset be used for? Video captioning and evaluating faithfulness of evaluation metrics.

Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

No. Are there tasks for which the dataset should not be used?

The videos in this dataset are from different sources and are unique. The dataset should not be used for tasks such as video editing.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Requiring Audio

Requiring Question Only

Attribute

Requiring Domain Knowledge

Requiring

Visual Information

0 200 400 600 800 1000 1200 Count

Figure 20: The distribution statistics of questions in the MMWorld benchmark by annotations.

- H.6 Distribution

Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?

Yes. The benchmark is publicly available. How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? We host it on the webpage, GitHub, and Huggingface. When will the dataset be distributed? It’s availale and open to the public now. Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? CC-By 4.0.

Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.

Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

- H.7 Maintenance

Who will be supporting/hosting/maintaining the dataset? The authors will be supporting/hosting/maintaining the dataset. How can the owner/curator/manager of the dataset be contacted (e.g., email address)? The email address is xhe89@ucsc.edu. Is there an erratum? No. We will make it if there is any erratum. Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? Yes. We will make announcements on GitHub if there is any update.

If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

N/A. Will older versions of the dataset continue to be supported/hosted/maintained? Yes. Old versions can still be accessed from Huggingface.

If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes. Contributors can post issues or submit pull requests on GitHub. We will review and verify contributions, and update the dataset if the contribution is useful.

### I Author Statement, Hosting, Licensing, and Maintenance Plan

Author Statement We bear all responsibility in case of violation of rights and confirmation of the data license.

Hosting MMWorld is hosted on https://mmworld-bench.github.io/. The dataset is provided in the JSON file format. The metadata can be found at https://huggingface.co/datasets/ Xuehai/MMWorld.

License MMWorld is licensed under the CC-BY 4.0 license. Maintenance Plan We will keep maintaining and updating the dataset and benchmark, including the leaderboard.

