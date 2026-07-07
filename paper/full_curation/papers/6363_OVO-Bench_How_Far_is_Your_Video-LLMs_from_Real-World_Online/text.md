# arXiv:2501.05510v2[cs.CV]27Mar2025

### OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?

Yifei Li∗1,2†, Junbo Niu∗1,3†, Ziyang Miao3, Chunjiang Ge2, Yuanhang Zhou2, Qihao He4, Xiaoyi Dong1,5 , Haodong Duan1, Shuangrui Ding1,5†, Rui Qian1,5†,

Pan Zhang1, Yuhang Zang1, Yuhang Cao1, Conghui He6, Jiaqi Wang1,7

1Shanghai Artificial Intelligence Laboratory, 2Tsinghua University, 3 Beihang University, 4 Communication University of China, 5 The Chinese University of Hong Kong, 6 SenseTime Group, 7 Shanghai Innovation Institute

[Figure 1]

Figure 1. A demonstrative comparison between offline and online video understanding [5]. Offline video understanding focuses on answering questions based on the entirety of a video. In contrast, online video understanding involves posing queries about the context of a video at intermediate points, demanding the ability to trace back past information, perceive ongoing events, and adapt to continuous input.

###### Abstract

Temporal Awareness—the ability to reason dynamically based on the timestamp when a question is raised—is the key distinction between offline and online video LLMs. Unlike offline models, which rely on complete videos for static, post hoc analysis, online models process video streams incrementally and dynamically adapt their responses based on the timestamp at which the question is posed. Despite its

* indicates equal contribution. † indicates interns at IXCLab, Shanghai AI Laboratory

significance, temporal awareness has not been adequately evaluated in existing benchmarks. To fill this gap, we present OVO-Bench (Online-VideO-Benchmark), a novel video benchmark that emphasizes the importance of timestamps for advanced online video understanding capability benchmarking. OVO-Bench evaluates the ability of video LLMs to reason and respond to events occurring at specific timestamps under three distinct scenarios: (1) Backward tracing: trace back to past events to answer the question. (2) Real-time understanding: understand and respond to events as they unfold at the current timestamp.

(3) Forward active responding: delay the response until sufficient future information becomes available to answer the question accurately. OVO-Bench comprises 12 tasks, featuring 644 unique videos and approximately humancurated 2,800 fine-grained meta-annotations with precise timestamps. We combine automated generation pipelines with human curation. With these high-quality samples, we further developed an evaluation pipeline to systematically query video LLMs along the video timeline. Evaluations of eleven Video-LLMs reveal that, despite advancements on traditional benchmarks, current models struggle with online video understanding, showing a significant gap compared to human agents. We hope OVO-Bench will drive progress in video LLMs and inspire future research in online video reasoning. Our benchmark and code can be accessed at https://github.com/JoeLeelyf/OVO-Bench.

###### 1. Introduction

Large Vision Language Models (LVLMs) [27, 34, 46, 59] and Video-LLMs [23, 33, 56] have shown remarkable progress, achieving impressive scores on existing benchmarks [11, 12, 24]. Recent works, such as VideoLLM-online [5] and Flash-VStream [57], have pioneered J.A.R.V.I.S1-like real-world video assistants by integrating pre-trained vision encoders [39] with LLMs [9, 45]. However, a critical question remains: How far are current state-of-the-art models from achieving human-level online video understanding?

Despite the existence of dozens of evaluation benchmarks in video understanding, there remains a significant domain gap between these evaluations and real-world video understanding tasks. Early evaluations [19, 52, 54] are largely based on video understanding and retrieval datasets [2, 53], assessing models through coarse-grained QA tasks, such as “Q: Who is dancing? A: Man”. These QAs predominantly focus on short videos with fixed question types and lack temporal indispensability [11]. Subsequent works [12, 24, 62] attempt to address these limitations by extending video temporal length and incorporating more diverse tasks and video sources. E.T.Bench [29] advances this further by exploring inherent temporal information in videos and evaluating fine-grained temporal event detection capabilities. However, all the aforementioned works are limited to offline settings, where models have access to all video frames when answering queries. While these models exhibit impressive performance on offline video understanding benchmarks, a substantial gap remains between their demonstrated capabilities and the requirements of a real-world assistant or autonomous agent.

- 1J.A.R.V.I.S. is a fictional AI assistant from Marvel’s Iron Man and

Avengers series.

A pioneering benchmark, VStream-QA [57], represents one of the earliest efforts to evaluate streaming understanding, leveraging video sources from Ego4d [15] and MovieNet [17]. Meanwhile, StreamingBench [26], a most recent work, expands the scope by evaluating Video-LLMs on a larger scale in streaming scenarios. However, three primary evaluation categories of StreamingBench primarily target the leverage of existing visual inputs to respond to incoming queries immediately, resulting in an incomplete portrayal of streaming perception.

In this work, we propose that effective online video understanding requires simultaneous capabilities to trace back past information, perceive the going-on, and forward active responding simultaneously. Given a query during a streaming video, a Video-LLM must determine whether to respond immediately using past and ongoing information or wait until sufficient evidence has been accumulated. We refer to this as the Video Chain-of-Time thinking process (Figure 3), inspired by the Chain-ofThought reasoning in LLMs [48].

We introduce OVO-Bench (Online-VideO-Benchmark) to evaluate Video-LLMs’ online video understanding capabilities. The benchmark comprises 644 videos from diverse sources, including curated datasets and web videos, spanning 7 major domains (Sports, Video Games, Ego Centric, etc.) with durations ranging from minutes to half an hour. Using a hybrid approach combining semiautomated MLLM generation and human curation, we created 2814 high-quality samples (Meta-Annotations) with precise event timestamps. These Meta-Annotations are organized into 12 tasks across three categories: Backward Tracing, Real-Time Visual Perception, and Forward Active Responding, reflecting the human video understanding process illustrated in Fig. 3. Notably, the proposed Forward Active Responding marks the first evaluation that requires models to continuously adapt their responses to ongoing visual input for online video understanding.

Building on the human-reviewed meta-annotations, we develop an evaluation pipeline that queries Video-LLMs densely along temporal axes to simulate continuous information processing. For Backward Tracing and RealTime Visual Perception, we adopt multiple-choice evaluation, converting videos into segments from start to query time to accommodate offline models. With this approach, we explore the potential of explicitly leveraging state-ofthe-art offline Video-LLMs for online video understanding. We evaluated eleven Video-LLMs, including proprietary models GPT-4o [34] and Gemini-1.5-Pro [44], alongside six recent open-source MLLMs like Qwen2-VL [46] and LLaVA-OneVision [22]. Despite their strong offline performance, these models struggle with online-style queries (e.g., What is happening now?), showing a significant gap from human performance. Further experiments on recent

streaming models, such as Flash-VStream [57], reveal an even wider performance gap compared to offline counterparts, highlighting a substantial research space for further exploration and improvement.

###### 2. Related Works

Video Large Language Models. Video Large Language Models (VLLMs) can process a video by treating it as a sequence of video frames. Projects like VideoChat [23], Video-LLaMA [56], and Video-ChatGPT [33] project the CLIP-ViT [40] embeddings of selected video frames through a Multi-Layer Perceptron (MLP) projector into the LLM embedding space, then concatenate these embeddings with text embeddings for enhanced video understanding. However, the context length of MLLMs limits their effectiveness in understanding long videos [23, 33], as longer videos require more frames and a longer context length. To address this limitation, two major approaches have been developed: compressing video features and selecting critical frames.

In the realm of feature compression, Chat-UniVi [21] merges similar visual tokens through clustering techniques. MovieChat [42] and MA-LLM [16] employ a memory bank to store a fixed number of video tokens by iteratively merging the most similar tokens. ST-LLM [28] and MovieChat [42] reduce video tokens to 32 using a pretrained Q-Former from BLIP2 [10]. LLaMA-VID [25] takes a more radical approach, compressing each frame into a content token and a context token.

On the other hand, frame selection methods aim to identify the most representative frames. VideoStreaming [37] utilizes a small LLM to select critical video clips, while FlashVstream [57] employs a clustering method to choose representative frames for high-resolution processing. LongVU [41] leverages question embeddings to select question-related frames, thereby enhancing video understanding.

Benchmarks for Video Understanding. Traditional video benchmarks, e.g., MSVD-QA [52], MSRVTT-QA [52], and ActivityNet-QA [54], predominantly consist of short videos, typically ranging from 1 to 2 minutes in duration. These datasets are meticulously annotated with corresponding questions and ground truth answers. GPT-4 [34] is employed to assess the accuracy of the answers by comparing

- them against the provided questions and ground truth responses. However, these benchmarks primarily focus on evaluating short, static video scenes. Hence, new benchmarks designed to test causal and temporal understanding, e.g., NExT-QA [51], TemporalBench [3], and AutoEvalVideo [7] are proposed.

To gauge the capabilities of models on long-duration videos, benchmarks like EgoSchema [32] covering over 5,000 egocentric videos with an average length of 3 min-

utes have been introduced. In contrast, Video-MME [12], LVBench [47], and LongVideoBench [50] feature videos spanning from 20 minutes to over an hour, evaluating a broad spectrum of video understanding capabilities. HourVideo [4] stands out with egocentric videos extending up to 2 hours, accompanied by more than 12,976 multiplechoice questions. Unlike these offline video benchmarks, our proposed OVO-Bench is designed to evaluate online, interactive video understanding.

Online Video Understanding. Traditional offline video understanding methodologies primarily focus on accessing entire video sequences to facilitate prediction tasks. Conversely, online video understanding demands models to process video streams sequentially, making decisions based on current and past information. This approach is particularly well-suited for scenarios where future data is unavailable, such as in embodied intelligence, autonomous driving, and augmented reality applications. Among online video understanding methods [38, 60], FlashVStream [57] employs a clustering method to select representative frames, enabling MLLMs for real-time interactions. LIVE [5] introduces a comprehensive framework for learning in video streams, which includes a training objective, data generation schema, and an inference pipeline for online video understanding.

###### 3. OVO-Bench

In this section, we present the construction process of our OVO-Bench. We start with a detailed introduction to the three different modes of online video understanding, followed by a comprehensive description of the data collection and annotation procedures. A statistical report of our proposed benchmark is displayed at the end of this section.

###### 3.1. Online Video Understanding Mode Taxonomy

Online video understanding aims to equip real-world, always-on agents with the ability to receive and process video inputs continuously, which closely mimics the human visual perception process. We categorize online video understanding into three distinct problem-solving modes: (1) Backward Tracing, (2) Real-Time Visual Perception, and (3) Forward Active Responding. Given a userprovided text query Qt

at the current time t0 and a streaming video input X(−∞,+∞), these modes are formally defined as follows:

0

###### 1. Backward Tracing:

Rt

= P(Qt

,X(−∞,−T])

0

0

###### 2. Real-Time Visual Perception:

Rt

= P(Qt

,X(−T,t

0])

0

0

###### 3. Forward Active Responding:

R(t

0,+∞) = P(Qt

,X(t

0,+∞))

0

00:00 00:28 06:30 08:00

###### Q: Where was the sieve before I picked it?

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### BackwardTracing

A) on the counter B) in the trash C) on the floor D) in the sink

EPM

EPisodic Memory

Clues Query

00:10 00:40 01:02

00:00

###### Q: Which object did the person eat before they threw the

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Action Sequence Identification

clothes? A) The sandwich B) The medicine C) Unable to answer D) The water

ASI

Clues Query

Reference

01:45

00:00 00:40 01:42

###### Q: Is the bedroom door open?

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

A) The bedroom door is open B) The bedroom door is closed C) Unable to answer

HLD

HaLlucination Detection

Clues

Query

00:00 00:21 00:59 01:39

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

STU SpaTial Understanding Q:A) OneHowB)manyTwo playersC) FourareD) Fivecompeting on the field now?

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

###### Misleading Misleading Query & Clue

00:00 01:00 04:00 05:00

Q: What is on the washing machine?

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Real-TimeVisualPerception

A) A bottle of fabric softener B) Two bottles of laundry detergent C) A box of dryer sheets D) A yellow bag

OJR ObJect Recognition

Query & Clue

00:00 12:05 28:15 30:00

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

ATR ATtribute Recognition Q:A) GreenWhat B)is thePurplecolorC)ofRedfireD)I’mOrangecollecting?

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

###### Misleading Query & Clue

00:00 01:00 02:30 05:00

Q: What am I doing with the motorcycle? A) Fastening a bolt B) Inspecting the brakes C) Adjusting the handlebars D) Checking the tire pressure

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

ACR ACtion Recognition

###### Misleading Query & Clue

00:00

01:05 02:30 05:00

Q: What is displayed on the house number of plate? A) 33-D B) 56-E C) 21-B D) 12-A

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Optical Character

OCR

Recognition

Query & Clue

00:00

00:25 00:27 05:00

Q: What is the man going to do? A) Taking a napkin on the table. B) Prepare to standup C) Take the hut on the table

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

FPD Future PreDiction

Query & Clue

00:00 00:06 00:19 00:41

Q: How many times did they long jump? Query Time: 00:00 Answer: 0(00:00-06) 1(00:06-19) 2(00:19-41)

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Repetition Event Count

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

REC

ForwardActive

Responding

Clues

Query Clues

00:00 00:16 00:24 01:07

Q: Illustrate me on the main steps of how to rewrap a battery.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Sequential Steps Recognition

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Query Time: 00:00 Answer: Remove the old wrapper (00:04-16) Wrap with the new wrapper (00:20-24)

SSR

Query Clues Clues

00:00 12:20 12:26

01:45

###### Q: The man turns his head and look at something, what is he looking?

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Clues Reveal Responding

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

CRR

Query Time: 12:20 Answer: A painting on the wall (12:26)

Query

Clues

- Figure 2. Examples of each task in OVO-Bench. The 14 tasks are categorized into three different kinds of perceiving modes in online video understanding: Backward Tracing, Real-Time Visual Perception, and Forward Active Responding.

in which T represents a threshold that defines the boundary for recent times, and R denotes the model’s response. The first two modes, Backward Tracing and Real-Time Visual Perception, involve collecting visual information from past and current timeframes respectively, and are expected to give immediate responses. In contrast, Forward Active Responding requires the model to withhold a response until sufficient future information becomes available to ensure a confident answer. Based on these distinctions, we have meticulously designed tasks tailored to each mode to effectively evaluate the performance of Video-LLMs across these diverse capabilities.

###### 3.1.1. Backward Tracing

Memory, particularly long-term memory, is a crucial aspect of human intelligence. In video understanding systems, this capability involves recalling and reasoning about past

events. We focus on the following three tasks to evaluate this capability:

- 1. [EPM] Episodic Memory: Backtrack and retrieve key moments from past video inputs.
- 2. [ASI] Action Sequence Identification: Identify the correct sequence of human actions in the video streams.
- 3. [HLD] Hallucination Detection: Ask questions irrelevant to existing video inputs.

###### 3.1.2. Real-Time Visual Perception

Accurate real-time perception of visual content is crucial, as actions undertaken in the present shape future outcomes. In various real-world scenarios, an immediate and precise understanding of ongoing visual inputs is essential. We propose six critical categories that constitute the foundational capabilities for effective real-time visual perception:

###### 1. [STU] Spatial Understanding. Reason over the spa-

tial relationships between objects occurring in nearby frames.

- 2. [OJR] Object Recognition. Recognize the objects appearing in the current frames.
- 3. [ATR] Attribute Recognition. Identify the characteristics or properties of objects, such as color, texture, and size that appear in nearby frames.
- 4. [ACR] Action Recognition. Recognize and interpret the actions being performed by individuals in the current frame.
- 5. [OCR] Optical Character Recognition. Recognize and interpret characters that appear within the frame.
- 6. [FPD] Future Prediction. Forecast the most probable subsequent phase of the current scene, including changes in object states, actions, and other dynamic elements.

###### 3.1.3. Forward Active Responding

Transitioning from passive reception to active perception is essential for advanced video understanding systems. Existing benchmarks primarily focus on the aforementioned two understanding modes, where Video-LLMs are required to respond immediately based on available information. In contrast, we introduce the Forward Active Responding mode, which allows the model to adjust its responses based on forthcoming visual inputs. We devise three task dimensions to evaluate the models’ active responding abilities:

- 1. [REC] Repetition Event Count. Respond when a repetitive event occurs again, including both highfrequency repetitive actions over short durations and semantically long-term repetitive occurrences of certain events.
- 2. [SSR] Sequential Steps Recognition. Respond when a certain procedure or sequence of actions has transitioned to another stage.
- 3. [CRR] Clues Reveal Responding. Delay responding until sufficient information or clues are provided.

###### 3.2. Benchmark Construction

Under the taxonomy guidelines above, we make our first step by collecting video data and annotations from existing datasets and crawling data from the web to increase diversity. As our proposed evaluation pipeline highly relies on the accurate timestamp annotations of the referred events in the constructed prompt, the scarcity of event-level timestamps in existing datasets [49][31][36] promotes the design of our highly efficient meta-data generation pipeline 3. Raw annotations with coarse timestamps are then refined by humans to ensure accuracy. Our final questions and options for evaluation are constructed using our rule-based pipeline based on these human-refined meta-annotations. All QA samples undergo manual inspection before being included in the final test set.

###### 3.2.1. Video and Annotation Collection

Video Source Selection. We follow existing benchmarks [29][24] by exploiting high-quality customized video datasets, and enrich our diversity by utilizing self-crawling videos from different domains. (1) Human-annotated Video Dataset. Our main consideration for utilizing organized datasets is to alleviate the labor-intensive source video collection process. Specifically, we include QA-Ego4D[1] and OpenEQA[30] for the [EPM] task, STAR[49], YouCook2[63], CrossTask[65], HiREST[55], and COIN[43] for the [ASI] task, Perception-Test[36] and Thumos[20][13] for the [REC] task, COIN[43] for the [SSR] task, MovieNet[17] for the [CRR] task, and Ego4D[15] for tasks under Real-Time Visual Perception. All samples are selected from val or test sets to avoid potential data leakage. (2) Web-crawling Videos. To further extend the diversity of our benchmark, we follow the existing practice [11][26] of crawling source videos from YouTube.

Meta-Annotations Collection. We employ three approaches to collect our meta-annotations, which contain event-level timestamps: (1) Existing Annotation Repurposing. For human-annotated datasets with accurate event-level timestamps [1][43][15], we explicitly take advantage of these labels and reconstruct them to our final prompt. (2) Semi-Automatic Generation. For datasets that provide video-level QA pairs without complete temporal localization, including [31][49][36][20][13], we prompt temporal-sensitive Video-LLMs like Gemini-1.5[44] to provide coarse-grain timestamps which fit the event referred in question and answer. For tasks under the Real-Time Visual Perception scenario, timestamps are given during our automatic QA construction process, which will be illustrated in 3.2.2. (3) Human-annotated. For the [SSR] and [CRR] tasks, questions, answers, and ground-truth timestamps are collected by our recruited volunteers. We then perform a meticulous inspection of all collected source videos and the corresponding meta-annotations to ensure precision.

###### 3.2.2. Prompt Generation

Question and Answer Generation. Besides carefully selecting QA pairs from existing datasets to fit into our proposed tasks, we also adopt a highly efficient automatic question and answer generation pipeline, particularly for the Real-Time Visual Perception scenario. We randomly sample short clips from original long-form videos and then leverage GPT-4o[18] to select potential candidates and construct questions and corresponding answers using human-refined prompts. Human-proposed questions are also adopted as a part of these tasks to alleviate possible LLM preferences. For the novel [CRR] task, even the strongest Video-

|Video Source Selection|
|---|

|Meta-Annotation Generation|
|---|

|MCQ Generation|
|---|

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Web-Crawling

Human annotated Datasets

[Figure 104]

[Figure 105]

[Figure 106]

Videos

VLLM Generation

VLLM Annotate

Human Annotate

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Human Refine

Raw Video Data

Videos with Fine-Grained Annotations Videos with MCQ

|Online Understanding Taxonomy|
|---|
|Where Can I Find Clues for the Query?<br><br>Can on-going visual content provide enough information?<br><br>No<br><br>Yes<br><br>Real-Time Visual Perception<br><br>Use memory information… Can past visual streams provide enough information?<br><br>I should wait for future input to provide a confident answer<br><br>Chain-of-Time Thinking Process<br><br>No<br><br>Yes<br><br>Backward Tracing<br><br>Yes<br><br>Forward Active Responding|

- Figure 3. Generation pipeline of OVO-Bench. Within public annotations, data is carefully filtered and relevant multiple-choice QAs are auto-generated. The effective system prompt and efficient answer prompt are employed to guide MLLMs toward precise outputs. The Video-LLMs we use to annotate videos are GPT-4o and Gemini-1.5 Pro.

[Figure 112]

[Figure 113]

[Figure 114]

- Figure 4. Left: Queries Temporal Distribution in OVO-Bench. Center: Linguistic Characteristics of Text Queries. Right: Video category distribution of OVO-Bench.

LLMs/MLLMs like Gemini-1.5-Pro struggle to construct desired problems. Volunteers are then recruited to provide QA pairs under our guidance.

Options Generation and Selection. We adopt multiplechoice questions as testing forms for Backward Tracing and Real-Time Visual Perception scenarios. However, as revealed in [6], the naively designed options of a multi-choice form query can cause information leakage about answers. We propose to generate options using a carefully designed rule-based and visually grounded transformation of correct answers, bringing misleading information from original videos to increase difficulty. Specifically, we prompt VideoLLMs with original QA pairs and corresponding video clips to generate visual-related options. A careful human review is then conducted to further ensure the options’ effectiveness. All options are shuffled after human review to avoid potential preference bias.

Prompting Offline-Models for Simulated Online Understanding. With the significant performance gap between main-streaming powerful offline Video-LLMs [44][35][46] and existing online models [5][57], one natural question is made: Is it effective to prompt offline models directly for online video understanding? For the Real-Time Visual Percep-

tion setting, we make human curation to the original question to include implies about the real-time query scenarios, for example, by using sentence patterns like What is/What am I or containing words like Now/Currently. We made another intuitive attempt to prompt offline models to solve tasks under our novel Forward Active Responding scenario, which asks for a continuous adapting capability. Specifically, we devise a multiple-triggering densely query and evaluation pipeline, allowing the model to decide whether existing information has provided enough clues to answer the user’s query.

###### 3.3. Datasets Statistics

OVO-Bench consists of 644 unique videos spanning 7 major domains, including Sports, Video Games, and Tutorial, among others. The video durations range from a few minutes to half an hour, with the average query timepoint being 428.89 seconds. Figure 4 Left illustrates the duration distribution of the queries within OVO-Bench. The benchmark includes 2,814 question-answer (QA) pairs, featuring a large number of multiple-choice questions and a smaller set of open-ended questions. The number of options for the multiple-choice questions varies between 2 and 5, rather

than being fixed at four. The distribution of video category is visualized in Figure 4 Right.

###### 4. Experiments

This section presents comprehensive experiments and indepth analyses of OVO-Bench.

###### 4.1. Models and Evaluation Strategies

We evaluate four existing types of models: (1) Offline Multimodal Models, including GPT-4o [35], Gemini-1.5Pro [44], Qwen2-VL [46], LLaVA-Video [61], LLaVAOneVision [27], InternVL-V2 [8] and LongVU [41], (2) Online Multimodal Models, including FlashVStream [57] , Videollm-Online[5] and Dispider[38], (3) Blind LLMs, including GPT-4-turbo [34]. (4) Human Agents. To ensure a fair comparison of model performance, we adhere to the principle of consistency by maintaining the same number of frames or frames per second (fps) across all models.

Considering the input video length limitations for offline Video-LLMs, we adopt specialized video input methods tailored to such models. Specifically, we segment the video into clips based on the timestamps of the questions.For instance, for a question Qi posed at timestamp ti, we extract the video clip Video[0 : ti] as the visual input. This approach simulates a streaming question-answering scenario in online video understanding.

We also conduct a runtimes study of five models, including QWen2-VL-7B [46], LLaVA-Video [61], LLaVAOneVision [22], InternVL-V2[8], and FlashVStream [57]. In this setting, we randomly select 100 samples from tasks in Backward Tracing and Real-Time Visual Perception and

- then plot the change of average inference delay on these videos with the number of sampled frames.

###### 4.2. Main Results

Table 1 reports the performance of eleven models under different settings on OVO-Bench, including the Real-Time Visual Perception, Backward Tracing, and Forward Active Responding. Our evaluation brings several important findings, as follows:

Offline Video-LLMs’ video understanding capabilities can be effectively transferred to real-time video understanding. The results demonstrate that offline VideoLLMs, despite being designed for offline processing, perform competitively in Real-Time Visual Perception tasks. This suggests that the advanced video comprehension abilities developed in offline settings are transferable and can enhance performance in certain online scenarios, thereby partially bridging the gap between offline and online video understanding.

Current Video-LLMs lack temporal prioritization when handling VQA tasks. Existing Video-LLMs do

not prioritize real-time temporal information when answering questions, leading to an inability to accurately locate the correct scene when multiple misleading scenes matching the question appear in the video stream, as shown in Fig2. Even the best current proprietary models achieve only 58.43% and 66.97% on [STU] and [ACR] tasks, respectively, which represents a significant gap compared to Human Agents.

Hallucinations are prevalent in Video-LLMs. The [HLD] in Table 1 measures hallucinations in VideoLLMs [58], indicating that hallucinations are a significant issue, particularly in open-source and online models. Proprietary models like Gemini 1.5 Pro perform better in managing hallucinations, yet there remains a notable gap compared to human performance(52.69% vs. 91.37%). This problem arises due to the models’ inability to fully comprehend complex visual and temporal contexts, leading to errors in interpretation and response. Addressing hallucinations is crucial for improving the reliability and accuracy of Video-LLMs in real-world applications.

Current Video-LLMs need more efficient inference frameworks to achieve real-time visual question answering. As shown in Fig6, the inference latencies of current Video-LLMs exhibit an exponential growth trend as frame numbers increase. Specifically, when using 64 frames as visual input, most efficient Video-LLMs, like QWen2VL7B[46] and FlashVStream [57], still need around 4 seconds on average to perform a response, making real-time video dialogue far from reach.

###### 4.3. Comparison between online Video-LLMs and offline Video-LLMs

Models like Gemini 1.5 Pro and Qwen2-VL-72B, representative of offline Video-LLMs, demonstrate strong performance across various tasks, as shown in Fig5. Specifically, Gemini 1.5 Pro achieves the highest average score among these models. This superior performance suggests that offline models, despite not being designed for online or realtime processing, can effectively comprehend and process complex visual information when provided with sufficient computational resources and pre-processing time. Their architectures typically allow for processing the entire video sequence holistically, leveraging global context and detailed temporal information, which enhances their temporal understanding and reasoning capabilities.

In contrast, Flash-VStream-7B, representing online Video-LLMs, shows comparatively lower performance in real-time perception tasks compared to offline models. This model is designed to process video in a streaming manner, handling inputs frame by frame with strict latency constraints to achieve real-time responsiveness. The performance gap highlights a potential trade-off between realtime processing capabilities and the depth of visual under-

|Model<br><br>|# Frames|Real-Time Visual Perception<br><br>| |Backward Tracing| |Forward Active Responding<br><br>| |Overall Avg.|
|---|---|---|---|---|---|---|---|---|
| | |OCR ACR ATR STU FPD OJR|Avg.<br><br>|EPM ASI HLD<br><br>|Avg.|REC SSR CRR<br><br>|Avg.|Overall Avg.|

###### Human

Human Agents - 93.96 92.57 94.83 92.70 91.09 94.02 93.20 92.59 93.02 91.37 92.33 95.48 89.67 93.56 92.90 92.81

###### Blind LLMs

GPT-4-turbo[34] - 28.86 24.77 25.67 33.76 27.72 26.63 27.90 42.76 48.65 70.05 53.82 - - 52.92 - -

###### Proprietary Multimodal Models-Offline

Gemini 1.5 Pro[44] 1fps 85.91 66.97 79.31 58.43 63.37 61.96 69.32 58.59 76.35 52.64 62.54 35.53 74.24 61.67 57.15 63.00 GPT-4o[35] 64 69.8 64.22 71.55 51.12 70.3 59.78 64.46 57.91 75.68 48.66 60.75 27.58 73.21 59.4 53.40 59.54

###### Open-source Multimodal Models-Offline

Qwen2-VL-72B[46] 64 65.77 60.55 69.83 51.69 69.31 54.35 61.92 52.53 60.81 57.53 56.95 38.83 64.07 45.00 49.30 56.27 LLaVA-Video-7B[61] 64 69.13 58.72 68.83 49.44 74.26 59.78 63.52 56.23 57.43 7.53 40.4 34.10 69.95 60.42 54.82 52.91 LLaVA-OneVision-7B[22] 64 66.44 57.80 73.28 53.37 71.29 61.96 64.02 54.21 55.41 21.51 43.71 25.64 67.09 58.75 50.50 52.74 Qwen2-VL-7B[46] 64 60.40 50.46 56.03 47.19 66.34 55.43 55.98 47.81 35.48 56.08 46.46 31.66 65.82 48.75 48.74 50.39 InternVL-V2-8B[8] 64 67.11 60.55 63.79 46.07 68.32 56.52 60.39 48.15 57.43 24.73 43.44 26.5 59.14 54.14 46.60 50.15 LongVU-7B[41] 1fps 53.69 53.21 62.93 47.75 68.32 59.78 57.61 40.74 59.46 4.84 35.01 12.18 69.48 60.83 47.50 46.71

Open-source Multimodal Models-Online Flash-VStream-7B[57] 1fps 24.16 29.36 28.45 33.71 25.74 28.80 28.37 39.06 37.16 5.91 27.38 8.02 67.25 60.00 45.09 33.61 VideoLLM-online-8B[5] 2fps 8.05 23.85 12.07 14.04 45.54 21.20 20.79 22.22 18.80 12.18 17.73 - - - - Dispider[38] 1fps 57.72 49.54 62.07 44.94 61.39 51.63 54.55 48.48 55.41 4.3 36.06 18.05 37.36 48.75 34.72 41.78

Table 1. Detailed evaluation results on OVO-Bench. To enhance the challenge of the questions by increasing the time interval between the question and the clues, the question time for [EPM] and [ASI] in the table is uniformly placed at the end of the video. For Forward Active Responding, accuracy-based evaluation metrics are utilized in this table.

[Figure 115]

standing.

###### 4.4. Forward Active Responding

We include our evaluation pipeline design for our proposed Forward Active Responding. While our high-quality human-annotated queries and clues lay an ideal testbed for future real-world online understanding models, existing naively designed online video models usually collapse in our evaluation process. We made our initial attempts to leverage our multiple-triggering query pipeline to prompt offline VideoLLMs to perform online video understanding thinking schema and further explore their potential in always-on visual perception.

- Figure 5. Performance comparison between online VideoLLMs and offline Video-LLMs. The figure illustrates the average scores of different models on the OVO-Bench in real-time visual perception tasks.

[Figure 116]

- Figure 6. Inference Latency (y-axis) v.s. Frames Number (xaxis). Latency test on an A100 GPU for FlashVStream and four A100 GPUs for other models.

Evaluation Pipeline and Metrics. As illustrated in Fig.7, We propose to query the Video-LLMs densely along the temporal axes, particularly around the interested events. Our main concerns are twofold: 1) Encourage models’ timely finding of the right clues, and 2) Avoid any possible hallucination before the right clue appears. For the [REC] task, larger counting numbers are awarded. Based on this, we proposed our designed scoring metrics for the three tasks in the Forward Active Responding.

Offline Models for Online Video Understanding. Despite their promising performance on the Backward-Tracing and Real-Time Visual Perception, in which the models are given full information for making confident responses, our preliminary results show that even state-of-the-art offline models like Gemini-1.5-Pro, fails to capture the linguistic information of ongoing querying, showing limited understanding of online video content.

###### 5. Conclusion and Future Work

In this work, we introduced OVO-Bench, a comprehensive benchmark designed to assess online video understanding capabilities of Video-LLMs across three critical modes:

|Multiple Triggering Evaluation|
|---|

|𝑀𝑁,𝑖|
|---|

|𝑀0,𝑖|
|---|

|𝑀1,𝑖|
|---|

|𝑀2,𝑖|
|---|

[REC]

…

|𝑀1,𝑖<br><br>| |𝑁1,𝑖|
|---|---|---|
| | | |

|𝑀𝑁,𝑖<br><br>| |𝑁𝑁,𝑖|
|---|---|---|
| | | |

[SSR] …

| |𝑀𝑁,𝑖|
|---|---|
|𝑁𝑁| |

|𝑁0| |𝑀0,𝑖|
|---|---|---|

|𝑁1| |𝑀1,𝑖|
|---|---|---|

[CRR] …

- Figure 7. Multiple triggering evaluation pipeline of prompt offline models for online video understanding. Offline VideoLLMs are densely queried along the temporal axes to make independent decisions of whether existing visual content provide enough clues for answering.

Backward Tracing, Real-Time Visual Perception, and Forward Active Responding.We anticipate that OVO-Bench will serve as a valuable resource for the research community, guiding the development of Video-LLMs toward practical, real-world applications. By highlighting current limitations and providing a platform for rigorous evaluation, we hope to inspire future research dedicated to advancing online video understanding and achieving human-level comprehension in artificial intelligence systems.

###### Acknowledgement

This project is funded in part by Shanghai Artificial lntelligence Laboratory, Shanghai Innovation Institute, the National Key R&D Program of China (2022ZD0160201). This project is also supported by the Shanghai Postdoctoral Excellence Program (No.2023023), China Postdoctoral Science Fund (No.2024M751559).

### OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?

#### Supplementary Material

###### 6. More Details of Evaluation

###### 6.1. Evaluation for Online Models on Forward Active Responding

As no existing online models can satisfy the demand imposed by our original designs, we choose not to cover this part in our main paper. We introduce an effective evaluation metric tailored for each task consisting of two different dimensions.

###### Guidance for evaluation metrics design.

- • Accuracy-Based. The model’s responses should, first of all, be correct without misleading information. We judge the effectiveness of the answer given by the model, and simply average all of them to give the accuracy.
- • Score-Based. Based on the accuracy-based evaluation, we encourage the response to be both accurate and timely and therefore devise a scoring metric.

###### Details of evaluation metrics. Given the user’s queries Qt

at time ti, the referred events Ej (such as a specific step of a tutorial procedure) with the time interval from tj to t′j, the appropriate response Am at time tm, the model’s responses Rm′ at time t′m, the evaluation function F(Rm′,Am), which directly compare the models’ responses against the right ones, the evaluation metrics of different models are formally given as follows.

i

- 1. [REC] In this task, the query is only made at a certain time before one complete repetition event happens. In our benchmark, the query is made at the start of the video, i.e. only Q0 is made.

- • Accuracy-Based.

Acc =

N i=1 F(Rm′,Am)

N

- • Score-Based.

Score =

N

i=1

ei·p

1

· F(Rm′,Am) · 2−(m

′−m)·p2

where F(Rm′,Am) = [Am == Rm′], which gives 1 if the model’s response is the same as the answer, and gives 0 otherwise. p1 and p2 are parameters to balance the weight. In our evaluation, they are set to 0.2 and 0.05 respectively.

- 2. [SSR] In this task, a query like Illustrate me on how to make a sandwich according to the video is made before the start of the procedure. Akin to [REC], the query is only made at the start of the video, i.e. only Q0 is made.

###### • Accuracy-Based.

N i=1 F(Rm′,Am)

Acc =

N

###### • Score-Based.

Score =

N

′−m)·p

F(Rm′,Am) · 2−(m

i=1

where we leverage GPT-4o to give F(Rm′,Am), measuring the effectiveness of Rm′ given the reference answer Am and relevant visual content. p is set to 0.5 to balance weight in our evaluation.

3. [CRR] In this setting, queries are made before every

Am, i.e. range(i) == range(m).

###### • Accuracy-Based.

N i=1 F(Rm′,Am)

Acc =

N

###### • Score-Based.

Score =

N

′−m)·p

F(Rm′,Am) · 2−(m

i=1

where we leverage GPT-4o to give F(Rm′,Am), measuring the effectiveness of Rm′ given the reference answer Am and relevant visual content. p is set to 0.5 to balance weight in our evaluation.

Prompt Design. To adapt to the online scenarios, we constructed streaming mode prompts with accurate timestamps and also deleted the complicated instructional statement compared to 6.2. Prompts and examples of models’ responses are shown in 8.

###### 6.2. Prompt Design for Offline Models on Forward Active Responding

The Forward Active Responding task is intrinsically inappropriate for offline models, as these models only support queries about existing video contents and can not receive additional visual frames after the query is made. However, considering the superiority of offline models against existing online models, we design a multiple-triggering evaluation pipeline and prompt offline models to decide whether the current time is appropriate for answering the user’s query. Formally, given the user’s query Qt

at t0, we leverage offline models to decide at ti,i ≥ 1;ti > t0 whether

0

|6.1. Online Models on Forward Active Responding<br><br>Prompt Used & Response Examples| | |
|---|---|---|
|[REC] Repetition Event Count<br><br>|[SSR] Sequential Steps Recognition|[CRR] Clues Reveal Responding|
|[Prompt]<br><br>In the video, the man/woman is [Action] repetitively. Remind me every time when he/she finishes one.<br><br>[Examples]<br><br>[Action] Showing something to the camera [Complete Query] In the video, the man/woman is showing something to the camera repetitively. Remind me every time when he/she finishes one.<br><br>[Query Time] 0:00/Start of the video [GT Times] (0:00-0.07) – (0:09-0:19) –<br><br>(0:21-0:25) – (0:27-0:34)<br><br>[Response]<br><br>[videollm-online] (0:00) You look around.|[Prompt]<br><br>Illustrate me on the steps of [Procedure] according to the video.<br><br>[Examples]<br><br>[Procedure] Make Sugar Coated Haws [Complete Query] In the video, the man/woman is showing something to the camera repetitively. Remind me every time when he/she finishes one. [Query Time] 0:00/Start of the video [Steps]<br><br>- String the fruit together<br>- Melt the sugar<br>- Soak sugar gourd in sugar [GT Times] (0:44-1.07) – (1:22-1:19) – (1:51-2:11)<br><br><br>[Response]<br><br>[videollm-online] (0:00) You hold a haw in your left hand<br><br>(0:05) You hold a haw in your left hand<br><br>… (0:49) You hold a haw in your left hand|[Prompt]<br><br>[Question]<br><br>[Examples]<br><br>[Question/Complete Query] The woman in the black coat walks towards the direction of the man in yellow, what action does she do with the man? [Clues Reveal Time] 5:17 [Answer] She walks past him. [Query Time] 5:10<br><br>[Response]<br><br>[videollm-online] (5:10) It seems like the woman in the black coat is walking towards the man in the yellow coat. She is likely to interact with him.|
|6.2. Offline Models on Forward Active Responding Prompt Used & Response Examples| | |
|[REC] Repetition Event Count|[SSR] Sequential Steps Recognition|[CRR] Clues Reveal Responding<br><br>|
|[Prompt]<br><br>You're a helpful assistant proficient in video question-answering.<br><br>You're watching a video in which people may perform a certain<br><br>type of action repetitively. The person performing are referred to as 'they' in the following statement. You're task is to count how many times did different people in the video perform this kind of action in total. Now, answer the following question: [Question] Your response type should be INT, for example, 0/1/2/3...<br><br>[Examples]<br><br>[Question] How many times did they showing something to the camera? [Ground Truth] 1 – 2 – 3 – 4<br><br>[GT Times] (0:00-0.07) – (0:09-0:19) –<br><br>(0:21-0:25) – (0:27-0:34)<br><br>[Query Times] (0:07) – (0:19) – (0:25) – (0:34)<br><br>[Response]<br><br>[Gemini-1.5 Pro] 1 – 1 – 1 - 6<br><br>[GPT-4o] 0 – 0 – 1 – 3<br><br>[Qwen-VL-72B] 1 – 4 – 1 – 4 [Qwen-VL-7B] 1 – 4 – 1 – 4 [LLaVA-NeXT-Video-7B] 2 – 2 – 2 – 2 [LLaVA-OneVision-7B] 2 – 0 – 2 – 1 [LongVU-7B] 3 – 3 – 3 – 3 [Flash-VStream-7B] 2 – 2 – 3 – 3|[Prompt]<br><br>You're a helpful assistant proficient in video question-answering. You're watching a tutorial video which contain a sequential of steps. The following is one step from the whole procedures: [Query Step] Your task is to decide: Is the man/woman in the video currently carrying out this step?. Return "Yes" if the man/woman in the video is currently performing<br><br>this step;<br><br>Return "No" if not<br><br>[Examples]<br><br>[Procedure] Make Sugar Coated Haws [Query Step] melt the sugar [Ground Truth] N – N – Y – Y – Y [Step Intervals] 1:22 – 1:46 [Query Times] (1:17) – (1:20) –<br><br>(1:24) – (1:27) – (1:46)<br><br>[Response]<br><br>[Gemini-1.5 Pro] N – N – N – N - Y<br><br>[GPT-4o] N – N – N – Y - Y<br><br>[Qwen-VL-72B] N – N – N – N - N [Qwen-VL-7B] N – N – N – N - Y [LLaVA-NeXT-Video-7B] N – N – N – Y – Y [LLaVA-OneVision-7B] N – N – N – N - Y [LongVU-7B] N – N – N – N – Y [Flash-Vstream-7B] Y – N – Y – Y –Y|[Prompt]<br><br>You're a helpful assistant proficient in video question-answering.<br><br>You're responsible of answering questions based on the video<br><br>content. The following question are relevant to the latest frames, i.e. the end of the video. [Question] Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question. Return "Yes" if existing visual content has provided enough information; Return "No" otherwise.<br><br>[Examples]<br><br>[Question] The woman in the black coat walks towards the direction of the man in yellow, what action does she do with the man? [Ground Truth] N – N – Y – Y – Y [Clues Reveal Time] 5:17 [Query Times] (5:10) – (5:13) –<br><br>(5:19) – (5:27) – (5:47)<br><br>[Response]<br><br>[Gemini-1.5 Pro] N – N – N – N - N<br><br>[GPT-4o] N – N – N – Y - N [Qwen-VL-72B] N – N – N – N - N [Qwen-VL-7B] N – N – N – N - N [LLaVA-NeXT-Video-7B] N – Y – Y – N – Y [LLaVA-OneVision-7B] Y – Y – Y – Y – Y [LongVU-7B] N – N – N – Y - N|

- Figure 8. Prompts used for Online(up) and Offline(down) Models on Forward Active Responding and Response Examples. Despite our vision for online models, existing online models, like videollm-online, are still far from satisfactory, showing limited adaptation ability, and would easily encounter collapse when processing complicated or out-of-training-domain video and queries. Offline models are inclined

- to perform random guessing when the queries contain words like ”is/currently/ongoing”.

video contents from t0 to ti offer sufficient clues. Specifically, for each of the tasks under the Forward Active Responding setting, instructional prompts and examples of models’ [5][57] responses are shown in Fig. 8.

###### 6.3. Prompt Design for Models on Backward Tracing and Real-Time Visual Perception

We use the clip from the beginning to the query time to query models. Prompts and examples of models’ responses are shown in Fig. 9.

###### 7. More Details of Benchmark Construction

###### 7.1. Human-annotated QA Generation

We leverage meticulous human labor for part of the QA generation.

Real-Time Visual Perception. For tasks, including [STU], [OJR], and [ATR], we invite volunteers to propose candidate questions in supplement to our Video-LLMsbased automatic generation pipeline. This procedure is designed to alleviate possible bias and increase diversity. Specifically, we provide our volunteers with the following guidelines:

• Watch the video and decide whether this candidate is appropriate for constructing questions that can be classified into the above three types.

|6.3-1 Online Models on Backward Tracing and Real-Time Visual Perception<br><br>Prompt Used & Response Examples| | |
|---|---|---|
|[ACR] Action Recognition<br><br>|[OCR] Optical Character Recognition|[ASI] Action Sequence Identification|
|[Prompt]<br><br>Question: [Question] Options: [Options] Answer with the option's letter from the given choices directly.<br><br>[Examples]<br><br>[Question] What is he doing? [Prompt] Question: What is he doing? Options:<br><br>A: He is wiping something with a rag.<br>B: He is hitting objects with a tool.<br>C: He is inspecting an object closely.<br>D: He is connecting the pipe to the interface. Answer with the option's letter from the given choices directly. [Query Time] 4:47 [GT] B(Time 3:21)->D(Time 1:43)->A(Time 4:47)<br><br><br>[Response]<br><br>[videollm-online] (4:47) You are inspecting the object closely.<br><br>.|[Prompt]<br><br>Question: [Question] Options: [Options] Answer with the option's letter from the given choices directly.<br><br>[Examples]<br><br>[Question] What name and number are visible on the back of this person's jacket? [Prompt] Question: What name and number are visible on the back of this person's jacket? Options:<br><br>A: LAFFONT, 00.<br>B: TOM, 21.<br>C: MOTEEA, 18.<br>D: GUS, 83. Answer with the option's letter from the given choices directly. [Query Time] 7:46 [GT] D(Time 1:43)->C(Time 5:09)->A(Time 7:46) [Response] [videollm-online] (7:46) According to the information provided, the name and number on the back of the person's jacket are "LAFFONT, 00".<br>|[Prompt]<br><br>Question: [Question] Options: [Options] Answer with the option's letter from the given choices directly.<br><br>[Examples]<br><br>[Question] Where did I put the shoe? [Prompt] Question: Where did I put the shoe? Options:<br><br>A: Under the table.<br>B: On the shelf.<br>C: Shoes organizer at the back of the door.<br>D: Unable to answer. Answer with the option's letter from the given choices directly. [Query Time] 7:00 [GT] D(Time 7:00) [Clues Time] 8:00<br><br><br>[Response]<br><br>[videollm-online] (7:00) You put the shoe on the shelf.<br><br>.|
|6.3-2 Offline Models on Backward Tracing and Real-Time Visual Perception Prompt Used & Response Examples<br><br>.| | |
|[ACR] Action Recognition<br><br>|[OCR] Optical Character Recognition|[ASI] Action Sequence Identification|
|[Response]<br><br>[Gemini-1.5 Pro] A [GPT-4o] A [Qwen-VL-72B] D [Qwen-VL-7B] B [LLaVA-NeXT-Video-7B] D [LLaVA-OneVision-7B] D [LongVU-7B] A [Flash-VStream-7B] D<br><br>[Prompt]<br><br>Question: [Question]<br><br>Options: [Options] Answer with the option's letter from the given choices directly.<br><br>[Examples]<br><br>[Question] What is he doing? [Prompt] Question: What is he doing? Options:<br><br>A: He is wiping something with a rag.<br>B: He is hitting objects with a tool.<br>C: He is inspecting an object closely.<br>D: He is connecting the pipe to the interface. Answer with the option's letter from the given choices directly. [Query Time] 4:47 [GT] B(Time 3:21)->D(Time 1:43)->A(Time 4:47)<br>|[Prompt]<br><br>Question: [Question] Options: [Options] Answer with the option's letter from the given choices directly.<br><br>[Examples]<br><br>[Question] What name and number are visible on the back of this person's jacket? [Prompt] Question: What name and number are visible on the back of this person's jacket?<br><br>Options:<br><br>A: LAFFONT, 00.<br>B: TOM, 21.<br>C: MOTEEA, 18.<br>D: GUS, 83. Answer with the option's letter from the given choices directly. [Query Time] 7:46 [GT] D(Time 1:43)->C(Time 5:09)->A(Time 7:46)<br><br><br>[Response]<br><br>[Gemini-1.5 Pro] A [GPT-4o] A [Qwen-VL-72B] A [Qwen-VL-7B] C [LLaVA-NeXT-Video-7B] A [LLaVA-OneVision-7B] C [LongVU-7B] A [Flash-VStream-7B] B|[Prompt]<br><br>Question: [Question]<br><br>Options: [Options] Answer with the option's letter from the given choices directly.<br><br>[Examples]<br><br>[Question] Where did I put the shoe? [Prompt] Question: Where did I put the shoe? Options:<br><br>A: Under the table.<br>B: On the shelf.<br>C: Shoes organizer at the back of the door.<br>D: Unable to answer. Answer with the option's letter from the given choices directly. [Query Time] 7:00 [GT] D(Time 7:00) [Clues Time] 8:00<br><br><br>[Response]<br><br>[Gemini-1.5 Pro] D [GPT-4o] D [Qwen-VL-72B] D [Qwen-VL-7B] D [LLaVA-NeXT-Video-7B] A [LLaVA-OneVision-7B] C [LongVU-7B] A [Flash-VStream-7B] A|

- Figure 9. Prompts used for Online(up) and Offline(down) Models on Real-Time Visual Perception and Response Examples. Three tasks including [ACR], [OCR], and [ASI] are included as demonstrations. Our benchmarks involve a large ratio of questions, whose answers shift over time, which means that models can hardly figure out the answer by randomly selecting frames from original videos.

- • Selected appropriate moments for problem construction. Consider whether the moment contains: 1. Obvious spatial relationships between several objects; 2. Interested objects, such as something that appears in the moment, and so on; 3. Objects with unusual attributes, e.g. green fire, smooth woods.
- • Construct options for the questions. Ensure that 1. Options should be relevant to the visual content; 2. Incorrect options should bring misleading information from the visual content; 3. Options should be as close in length as possible.

Clue Reveal Responding. For our novel [CRR] task, we find it difficult to construct satisfactory question proposals by straightly prompting Video-LLMs with original video content as reference or LLMs with the provided scripts and subtitles as reference. So we recruit volunteers to propose

queries and corresponding answers. Our guidelines for volunteers are as follows:

- • Find scenes with apparent discontinuity. For example,

character A performs a certain action at query time Qi. However, the action’s complete process or outcome is not immediately shown during query time.

- • Continue watching the video, find clues for your query, and annotate the clues revealing time as Ai.
- • Try to provide concise timestamps, let Ai be the time when enough visual information has just been revealed.

###### 8. Additional Dataset Analysis

###### 8.1. Task and Sample Distribution

Fig. 10 illustrates the distribution of questions and videos in OVO-Bench across the twelve tasks listed in Fig. 2.

[Figure 117]

- Figure 10. Distribution of questions and video in OVO-Bench.

[Figure 118]

- Figure 11. Distribution of averaged query timestamps and video duration (in seconds) in OVO-Bench. Specifically, the averaged video duration in CRR is 6,857 seconds.

###### 8.2. Query Timestamps and Video Duration

Fig. 11 illustrates the distribution of averaged query timestamps and video duration in OVO-Bench across the twelve tasks listed in Fig. 2.

###### 10. Licenses

The annotations of our OVO-Bench are provided to the community under CC BY-NC-SA 4.0 license. By downloading our dataset from our website or other sources, the user agrees to adhere to the terms of CC BY-NC-SA 4.0 and licenses of the source datasets. Download links are provided for our self-crawled YouTube videos. Licenses of the source datasets are listed in 2

Dataset License

QAEgo4D [1] N/A OpenEQA [30] MIT License STAR [49] Apache License 2.0 HiREST [55] MIT License YouCook2 [64] MIT License CrossTask [65] BSD 3-Clause License COIN [43] Research Purpose Only Ego4D [14] MIT License

- THUMOS’14 [20] Research Purpose Only

- THUMOS’15 [13] Research Purpose Only Perception Test [36] CC BY 4.0 MovieNet [17] N/A E.T.Bench [29] CC BY 4.0

Table 2. License of source datasets in OVO-Bench.

###### 11. Data Examples

We provide more examples extracted from our benchmark. We try to cover different video categories in every task to offer a holistic overview of OVO-Bench.

###### 9. Limitations

While we have tried hard to cover a wide range of reasonable video domains and QA generation methods, the scarcity of existing datasets with annotations that fit requirements, the unsatisfactory results of automatic QA generation, and the high human annotation cost, hinder diversity and can cause potential bias.

Offline Models for Online Video Understanding. As implied in our analysis 8, offline models usually perform random guesses in the forward active responding scenarios, making our evaluation unfair. For example, a model that always outputs ”Yes” can still achieve a score above zero in our evaluation. Moreover, the absence of online models with satisfactory performance, makes our benchmarks more suitable for future advancements. We hope our intensive work and intuitive ideas can guide the development of video understanding models toward real-world online video understanding.

## [EPM] Episodic Memory

Clue Time: 5:22 Options:

Clue Time: 5:22

Clue Time: 3:44 Question: What did I pick from the fridge? Options: A. bread; B. milk; C. vegetable; D. water

Question: Where was the kitchen paper towel before I picked it? Options: A. top of the kitchen sink; B. under the sink;

C. on the counter; D. on the stove

| | |
|---|---|
|[Figure 119]| |

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

[Figure 130]

[Figure 131]

[Figure 132]

Clue Time: 5:25 Question: How many rolls of paper towel did I cut? Options: A. two; B. one; C. three; D. four

Clue Time: 6:36 Question: Did I leave the drawer open? Options: A. yes; B. no

Query At the End

QA-Ego4D

Clue Time: 3:05 Question: What drawer did I pull?

Clue Time: 1:59 Question: Where was the paper towel before I picked it? Options: A. on the counter; B. on the cupboard;

Options: A. dish washer; B. microwave;

C. cupboard; D. fridge

C. on the table; D. in the dish washer

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]| |
|---|---|
|[Figure 136]<br><br>| |

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

|[Figure 141]|
|---|

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Clue Time: 3:13 Question: Where was the plate before I picked it? Options: A. on the counter; B. on the cupboard

Clue Time: 4:09 Question: What did I pour in the container? Options: A. flour; B. bread crumbs; C. salt; D. pepper

Query At the End

C. on the table; D. in the dish washer

Clue Time: 0:50 Question: Is this home on the first floor? Options: A. Yes, it’s on the first floor;

Clue Time: 0:43 Question: What is the gold object on the nightstand? Options: A. A painting; B. A mirror;

B. No, it’s on the second floor;

C. A nightlamp; D. A vase

|[Figure 147]<br><br>[Figure 148]|
|---|

|[Figure 149]|
|---|

[Figure 150]

[Figure 151]

[Figure 152]

##### Open-EQA

|[Figure 153]|
|---|

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Clue Time: 1:10

###### Clue Time: 1:10

Question: Where can I sit and eat if I don't want to use the dining table?

Question: What color is the smoke detector?

Query At

Options: A. Use the kitchen bar counter; B. Use the floor in the hallway; C. Use the bed in the bedroom; D. Use the couch in the living room

Options: A. White; B. Yellow or off-white; C. Black; D. Blue

the End

## [HLD] Hallucination Detection

Query Time: 6:11 Question: what did I put in the black dustbin? Options: A. empty water bottles; B. Unable to answer;

Query Time: 6:44 Question: Where did I put the vacuum cleaner head? Options: A. closet; B. Unable to answer;

C. old newspapers; D. food scraps

C. bathroom; D. kitchen

|[Figure 158]|
|---|

[Figure 159]

|[Figure 160]|
|---|

[Figure 161]

[Figure 162]

[Figure 163]

|[Figure 164]| |
|---|---|
| | |

[Figure 165]

|[Figure 166]| |
|---|---|
| | |

[Figure 167]

QA-Ego4D

Clue Time: 7:10 Clue Time: 7:30

Query Time: 6:48 Question: Where were game boards?

Options: A. Unable to answer; B. in the shelves;

C. in the fridge; D. in the bags

|[Figure 168]<br><br>| |
|---|---|
|[Figure 169]| |

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

|[Figure 176]| |
|---|---|
| | |

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

###### Clue Time: 7:55

Query Time: 0:35 Question: what color is the flower in the bottom floor? Options: A. Red; B. Unable to answer;

C. Blue; D. White

| | |
|---|---|
|[Figure 182]| |

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Open-EQA

|[Figure 187]|
|---|

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

###### Clue Time: 1:15

Clue Time: 0:02

Reference Time: 0:10

| | |
|---|---|
|[Figure 192]| |

|[Figure 193]|
|---|

[Figure 194]

[Figure 195]

[Figure 196]

##### STAR

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Query Time: 0:27/End of the video Question: What happened before the person took the pot? Options: A. close the refrigerator.; B. Unable to answer.;

C. Took the box.; D. Throw the broom.; E. Open the book.

Reference Time: 0:10 Clue Time: 0:02

|[Figure 202]<br><br>| |
|---|---|
|[Figure 203]| |

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

|[Figure 209]|
|---|

[Figure 210]

[Figure 211]

Query Time:11:46/End of the video Question: What does the person do after cover up and cook for 6 to 8 minutes? Options: A. chop up the tomatoes.; B. chop up oleander.;

C. add more garam masala powder.; D. add cloves to the pot

YouCook2

###### Clue Time: 0:02 Reference Time: 0:10

|[Figure 212]|
|---|

|[Figure 213]|
|---|

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Query Time:9:03/End of the video Question: What does the person do before roughly chop garlic and peppers and add them to the bowl? Options: A. pour some olive oil on the kabob.; B. pour olive oil on shrimp.;

C. cut up some onions and peppers into squares.; D. skewer the vegetables and shrimps

Clue Time: 6:40

Reference Time: 5:00

[Figure 222]

[Figure 223]

[Figure 224]

|[Figure 225]|
|---|

[Figure 226]

##### CrossTask

|[Figure 227]|
|---|

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Query Time: 9:34/End of the video Question: What does the person do after add spices? Options: A. pack cucumbers in jar.; B. add sugar.;

C. pour water.; D. add salt

###### Reference Time: 3:38 Clue Time: 5:00

| |[Figure 232]<br><br>|
|---|---|
|[Figure 233]| |

| |[Figure 234]<br><br>|
|---|---|
|[Figure 235]| |

[Figure 236]

[Figure 237]

[Figure 238]

##### HiREST

[Figure 239]

[Figure 240]

[Figure 241]

Query Time:5:54/End of the video Question: What does the person do after attach pebble piece? Options: A. attach background piece.; B. merge them. C. sew the edges.; D. cut the fabric

Clue Time: 3:00 Reference Time: 1:48

| |[Figure 242]<br><br>|
|---|---|
|[Figure 243]| |

[Figure 244]

[Figure 245]

|[Figure 246]|
|---|

[Figure 247]

##### COIN

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Query Time: 3:50/End of the video Question: What does the person do after load the wheel? Options: A. pump up the tire.; B. upload the wheel.;

C. load the tire.; D. load the inner tube

## [STU] Spatial Understanding

Query Time/Clue Time: 1:27

Question: What is the relative position of the person to the car ? Options: A. The person is standing at the front of the car.;

- B. The person is on the co-pilot side of the car.;
- C. The person is standing beside the driver's side of the car ;
- D. The person is behind the trunk of the car.

| | |
|---|---|
|[Figure 252]| |

Ego4D

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Query Time/Clue Time: 1:47 Question: Which container is located closer to the top left corner of the table? Options: A. The blue container;

- B. The white container;
- C. The red container;
- D. The green container.

| | |
|---|---|
|[Figure 264]| |

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

YouTubeYouTube-HumanAnnotated

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Query Time/Clue Time: 0:12 Question: Which road did I take? Options: A. The road on the right;

- B. The road on the left;
- C. Unable to answer.

| | |
|---|---|
|[Figure 274]| |

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

## [OJR] Object Recognition

Query Time/Clue Time: 2:08 Question: What object is being used to construct the paper aircraft wing? Options: A. A plastic frame.;

- B. A wooden stick..;
- C. A paper sheet;
- D. A metal rod.

Ego4D

|[Figure 284]|
|---|

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Query Time/Clue Time: 1:27 Question: What object is being attached to the back of it? Options: A. A bag.;

- B. A saddlebag.;
- C. A basket;
- D. A pannier.

| | |
|---|---|
|[Figure 294]| |

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

##### YouTubeYouTube-HumanAnnotated

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

|[Figure 312]| |
|---|---|
| | |

[Figure 313]

[Figure 314]

[Figure 315]

Query Time/Clue Time: 4:28 Question: What weapon do I have in my hand? Options: A. Bow; B. Sword;

C. Unable to answer. D. Axe. E. Spear

## [ATR] Attribute Recognition

Query Time/Clue Time: 1:33

Question: What is the material of the object being cleaned?

Options: A. Plastic; B. Metal; C. Wood ; D. Glass.

Ego4D

[Figure 316]

[Figure 317]

[Figure 318]

|[Figure 319]|
|---|

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Query Time/Clue Time: 1:47 Question: What is notable about the man's tie ? Options: A. The man's tie is yellow and textured.;

- B. The man's tie has a geometric pattern.;
- C. The man's tie is purple and striped;
- D. The man's tie features cartoon characters.

| | |
|---|---|
|[Figure 326]| |

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

YouTubeYouTube-HumanAnnotated

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Query Time/Clue Time: 9:20 Question: What is the color of the monsters' clothes? Options: A. Red; B. Blue;

C. Yellow. D. Green

| | |
|---|---|
|[Figure 336]| |

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

## [ACR] Action Recognition

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

##### Ego4D

|[Figure 352]| |
|---|---|
| | |

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

Query Time/Clue Time: 4:18 Question: What action is he performing with the blue checkered cloth? Options: A. He is wiping the tabletop with the cloth.;

B. He is folding the cloth.; C. He is covering a basket with the cloth; D. He is tying the cloth around his neck.

Query Time/Clue Time: 1:03 Question: What is he doing? Options: A. He is playing a musical instrument.; B. He is taking off his clothes.; C. He is taking a shower; D. He is answering the phone.

| | |
|---|---|
|[Figure 358]| |

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

YouTube

Query Time/Clue Time: 2:37 Question: What is she doing with the chicken in her hands? Options: A. She is placing the chicken into a pot of water.;

- B. She is putting the chicken down.;
- C. She is holding the chicken while preparing to cook it.;
- D. She is cutting the chicken into smaller pieces..

| | |
|---|---|
|[Figure 368]| |

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

## [OCR] Optical Character Recognition

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

|[Figure 383]| |
|---|---|
| | |

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

Query Time/Clue Time: 2:16 Question: What is the leading player's time at the 150m mark? Options: A. 1:16.96 B. 1:15.89;

C. 1:14.32. D. 1.19.01

Query Time/Clue Time: 1:35 Question: What is the text on the package? Options: A. ANDOUILLE.;

Query Time/Clue Time: 0:23 Question: What is the text on the package?

Options: A. ANDOUILLE.;

- B. THICK CUT HAM.;
- C. THICK CUT PORK;
- D. THICK CUT BACON.

- B. THICK CUT HAM.;
- C. THICK CUT PORK;
- D. THICK CUT BACON.

|[Figure 388]|
|---|

|[Figure 389]|
|---|

[Figure 390]

[Figure 391]

[Figure 392]

YouTube

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

|[Figure 403]|
|---|

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Query Time/Clue Time: 10:12 Question: What text is displayed now? Options: A. Milk Bucket.;

- B. Butter;
- C. Peanut Butter Cup.;
- D. Peppermint Swirl.

## [FPD] Future Prediction

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

|[Figure 414]|
|---|

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

Query Time/Clue Time: 1:45 Question: What is the person preparing to manipulate? Options: A. The person is about to handle or manipulate the wire;

- B. The person is about to handle or manipulate the plastic tubing;
- C. The person is about to handle or manipulate the circuit board;
- D. The person is about to handle or manipulate the metal sheet

Query Time/Clue Time: 0:59 Question: What action is this person preparing to take ? Options: A. The person is about to turn on the faucet;

B. The person is preparing to start the car engine; C. The person is about to open the refrigerator door; D. The person is reaching for the light switch.

| | |
|---|---|
|[Figure 420]| |

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

Ego4D

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

Query Time/Clue Time: 0:59 Question: What is this person about to do? Options: A. The person is about to press a button.;

- B. The person is about to grab a book;
- C. The person is about to open the drawer;
- D. The person is about to adjust the lamp.

| | |
|---|---|
|[Figure 432]| |

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

## [REC] Repetition Event Count

Query Time:0.26/1.53/End of the video Question: How many times did they clean and jerk? Answers: 1/2/2

##### PerceptionTestThumos14/15

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Clue Time: 0:01-0.26 Clue Time: 0:27-1.53

Clue Time: 0:00-0.06 Clue Time: 0:07-0.14

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

Clue Time: 0:17-0.24 Clue Time: 0:25-0.34

Query Time:0.06/0.14/0.24/0.34/End of the video Question: How many times did they showing something to the camera? Answers: 1/2/3/4/4

Clue Time: 0:11-0.13 Clue Time: 0:13-0.14

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

Clue Time: 0:14-0.15 Clue Time: 0:15-0.17

Query Time:0.06/0.13/0.14/0.15/0.17/0.27/End of the video Question: How many times did they take something out of something?

Answers: 0/1/2/3/4/4/4

## [SSR] Sequential Steps Recognition

Clue Time: 0:31-0:49 put on the hair extensions

Clue Time: 0:25-0:30 Pull up the hair to reserve place for the hair extensions

|[Figure 474]|
|---|

|[Figure 475]|
|---|

|[Figure 476]|
|---|

[Figure 477]

[Figure 478]

|[Figure 479]|
|---|

|[Figure 480]|
|---|

|[Figure 481]|
|---|

[Figure 482]

[Figure 483]

Clue Time: 1:47-1:57 Put down the hair and comb

###### Query Time: Start Of the Video

Question: Illustrate me on how to put on hair

extensions according to the video

Clue Time: 1:07-1:11 Pour the egg into the pot

Clue Time: 0:41-0:49 Pour the egg into the bowl

| | |
|---|---|
|[Figure 484]| |
| | |

[Figure 485]

|[Figure 486]|
|---|

[Figure 487]

[Figure 488]

COIN

| | |
|---|---|
|[Figure 489]| |
| | |

|[Figure 490]|
|---|

|[Figure 491]|
|---|

|[Figure 492]|
|---|

[Figure 493]

Clue Time: 1:42-1:52 Pour the egg onto the plate

###### Clue Time: 1:12-1:33 Fry eggs

Query Time: Start Of the Video Question: Illustrate me on how to cook

omelet according to the video

Clue Time: 1:41-1:46 Melt the sugar

Clue Time: 0:44-1:07 String the fruit together

|[Figure 494]|
|---|

|[Figure 495]|
|---|

|[Figure 496]|
|---|

[Figure 497]

[Figure 498]

|[Figure 499]|
|---|

|[Figure 500]|
|---|

|[Figure 501]|
|---|

[Figure 502]

[Figure 503]

Clue Time: 1:51-2:11 Soak sugar gourd in sugar

Query Time: Start Of the Video Question: Illustrate me on how to make sugar

coated haws according to the video

## [CRR] Clues Reveal Responding

Clue Time: 3:25 Answer: To listen to the older

Query Time: 2:25 Question: Women came out of doors of different colors and went into the center door. What is the purpose of doing so?

woman talking

[Figure 504]

[Figure 505]

|[Figure 506]|
|---|

|[Figure 507]|
|---|

[Figure 508]

|[Figure 509]|
|---|

|[Figure 510]|
|---|

[Figure 511]

[Figure 512]

[Figure 513]

- Query Time: 2:25 Question: The man picked up several books from the ground.

What does the man do with the books he picked up?

Clue Time: 3:25 Answer: He handed these

books to the woman

Query Time: 4:02 Question: A policeman is driving the car away,

what is his destination

Clue Time: 5:15

Answer: A residential house with a woman inside.

Query Time: 10:19 Question: The policeman is stepping into a wooden house,

what does the police see?

Clue Time: 10:26 Answer: A black man sitting on a bench.

|[Figure 514]| |
|---|---|
| |[Figure 515]<br><br>|

|[Figure 516]| |
|---|---|
| |[Figure 517]<br><br>|

- Query Time: 3:23 Question: The woman with the gray dress is stepping

[Figure 518]

|[Figure 519]|
|---|

|[Figure 520]|
|---|

[Figure 521]

[Figure 522]

MovieNet

|[Figure 523]|
|---|

[Figure 524]

[Figure 525]

|[Figure 526]|
|---|

[Figure 527]

###### Clue Time: 3:27

Answer: She talks to the man

into the room, what does she do in the room?

|[Figure 528]|
|---|

|[Figure 529]|
|---|

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

Clue Time: 8:28 Answer: A group

Query Time: 8:07 Question: The couple are sitting on their trunks,

of friends

who do they meet then

###### References

- [1] Leonard B¨armann and Alex Waibel. Where did i leave my keys?-episodic-memory-based question answering on egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1560– 1568, 2022. 5, 4
- [2] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 2
- [3] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models, 2024. 3
- [4] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Crist´obal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Li Fei-Fei. Hourvideo: 1-hour video-language understanding. arXiv preprint arXiv:2411.04998, 2024. 3
- [5] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In CVPR, 2024. 1, 2, 3, 6, 7, 8
- [6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330,

2024. 6

- [7] Xiuyuan Chen, Yuan Lin, Yuchen Zhang, and Weiran Huang. Autoeval-video: An automatic benchmark for assessing large vision language models in open-ended video question answering, 2024. 3
- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 7, 8
- [9] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023. 2
- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023. 3
- [11] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024. 2, 5

- [12] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 2, 3
- [13] Alex Gorban, Haroon Idrees, Yu-Gang Jiang, A Roshan Zamir, Ivan Laptev, Mubarak Shah, and Rahul Sukthankar. Thumos challenge: Action recognition with a large number of classes, 2015. 5, 4
- [14] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022. 4
- [15] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022. 2, 5
- [16] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding supplementary material. 3
- [17] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 709–727. Springer, 2020. 2, 5, 4
- [18] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5
- [19] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2758–2766, 2017. 2
- [20] Yu-Gang Jiang, Jingen Liu, A Roshan Zamir, George Toderici, Ivan Laptev, Mubarak Shah, and Rahul Sukthankar. Thumos challenge: Action recognition with a large number of classes, 2014. 5, 4
- [21] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046, 2023. 3
- [22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 7, 8
- [23] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2, 3
- [24] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al.

- Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 2, 5
- [25] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. 2024. 3
- [26] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024. 2, 5
- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 2, 7
- [28] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. St-llm: Large language models are effective temporal learners. In European Conference on Computer Vision, pages 1–18. Springer, 2025. 3
- [29] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Chang Wen Chen, and Ying Shan. E.t. bench: Towards open-ended event-level video-language understanding. In Neural Information Processing Systems (NeurIPS), 2024. 2, 5, 4
- [30] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16488– 16498, 2024. 5, 4
- [31] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16488– 16498, 2024. 5
- [32] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding, 2023. 3
- [33] Salman Khan Muhammad Maaz, Hanoona Rasheed and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. ArXiv 2306.05424, 2023. 2, 3
- [34] OpenAI. Gpt-4 technical report, 2023. Technical report. 2,

- 3, 7, 8

[35] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-

- 4o, 2024. 6, 7, 8

- [36] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36, 2024. 5, 4
- [37] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. Advances in Neural Information Processing Systems, 37:119336–119360,

2024. 3

- [38] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider:

- Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. arXiv preprint arXiv:2501.03218, 2025. 3, 7, 8
- [39] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 2
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 3
- [41] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 3, 7, 8
- [42] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, and Gaoang Wang. Moviechat: From dense token to sparse memory for long video understanding,

2023. 3

- [43] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207– 1216, 2019. 5, 4
- [44] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2, 5, 6, 7, 8
- [45] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [46] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 6, 7, 8
- [47] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024. 3
- [48] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. 2
- [49] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reason-

ing in real-world videos. arXiv preprint arXiv:2405.09711,

2024. 5, 4

- [50] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024. 3
- [51] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 3
- [52] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017. 2, 3
- [53] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 2
- [54] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9127–9134, 2019. 2, 3
- [55] Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal. Hierarchical video-moment retrieval and step-captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23056–23065, 2023. 5, 4
- [56] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 2, 3
- [57] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams, 2024. 2, 3, 6, 7, 8
- [58] Jiacheng Zhang, Yang Jiao, Shaoxiang Chen, Jingjing Chen, and Yu-Gang Jiang. Eventhallusion: Diagnosing event hallucinations in video llms. arXiv preprint arXiv:2409.16597,

2024. 7

- [59] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition, 2023. 2
- [60] Pan Zhang, Xiaoyi Dong, Yuhang Cao, Yuhang Zang, Rui Qian, Xilin Wei, Lin Chen, Yifei Li, Junbo Niu, Shuangrui Ding, et al. Internlm-xcomposer2. 5-omnilive: A comprehensive multimodal system for long-term streaming video and audio interactions. arXiv preprint arXiv:2412.09596,

2024. 3

- [61] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. 7, 8
- [62] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng

- Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 2
- [63] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. 5
- [64] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. 4
- [65] Dimitri Zhukov, Jean-Baptiste Alayrac, Ramazan Gokberk Cinbis, David Fouhey, Ivan Laptev, and Josef Sivic. Crosstask weakly supervised learning from instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3537–3545, 2019. 5, 4

