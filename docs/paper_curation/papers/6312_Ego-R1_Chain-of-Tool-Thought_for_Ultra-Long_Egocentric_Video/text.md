# Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning

##### Shulin Tian1,2∗ Ruiqi Wang1,3∗ Hongming Guo4 Penghao Wu1 Yuhao Dong1 Xiuying Wang1 Jingkang Yang1 Hao Zhang3 Hongyuan Zhu2 Ziwei Liu1 1S-Lab, Nanyang Technological University 2A*STAR, Singapore 3Simon Fraser University 4Shanghai AI Lab

arXiv:2506.13654v1[cs.CV]16Jun2025

[Figure 1]

Day 1 Day 2 Day 3 Day 4 Day 5 Day 6 Day 7

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

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Daily Log Daily Log Daily Log Daily Log

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

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

Tool #2: Video_LLM

Hour … Hour Hour … Hour Hour … Hour Hour … Hour Hour

Toolbox

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Tool #1: Hierarchical_RAG Tool #3: VLM

Q: Who was in front of me last time I stood on the escalator in the supermarket? <timestamp>DAY5_16062618</timestamp>

Day 1 Day 2 Day 3 Day 4

<think> I need to find who was in front of me when I was standing on the escalator in the supermarket last time. I’ll use RAG to retrieve week level info with keywords ["escalator", "supermarket"] </think>

[Figure 58]

[Figure 59]

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

[Figure 75]

[Figure 76]

[Figure 77]

###### <tool> Hierarchical_RAG("week", ["escalator", "supermarket"], ["DAY1_00000000", "DAY5_16062618"]) </tool>

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Day 2

OBSERVATION: "DAY3": … Supermarket Shopping and Logistics: The day included a coordinated shopping trip …

… … …

…

[Figure 85]

[Figure 86]

Chain-of-Tool-Thought

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Reasoning with

[Figure 93]

[Figure 94]

[Figure 95]

<think> Based on the observations, there are a group of people standing on the escalator. Now I will use VLM to check 'DAY3_17142510' and ask 'Who is standing in front of me on the escalator here?'. </think>

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

###### <tool> VLM ("Who is standing in front of me on the escalator here?", "DAY3_17142510"]) </tool>

OBSERVATION: Based on the image, the person in front of you is a girl wearing a green jacket and a blue cap.

[Figure 100]

<think> This frame shows a girl in green jacket is in front of me, while the RAG info shows during that period, I stepped onto the escalator with Tasha. I have gathered enough info to answer this question. </think>

<answer> Tasha. </answer>

- Figure 1: Overview of Ego-R1. In this figure, we demonstrate how the Ego-R1 Agent orchestrates specialized tools (e.g., Hierarchical_RAG, Video LLM, and VLM) to answer the question step-by-step, based on the observations and previous actions. The system effectively answers questions that require careful searching within ultra-long videos and precise analysis of frame details.

#### Abstract

We introduce Ego-R1, a novel framework for reasoning over ultra-long (i.e., in days and weeks) egocentric videos, which leverages a structured Chain-ofTool-Thought (CoTT) process, orchestrated by an Ego-R1 Agent trained via reinforcement learning (RL). Inspired by human problem-solving strategies, CoTT decomposes complex reasoning into modular steps, with the RL agent invoking specific tools, one per step, to iteratively and collaboratively answer sub-questions tackling such tasks as temporal retrieval and multi-modal understanding. We design a two-stage training paradigm involving supervised finetuning (SFT) of a pretrained language model using CoTT data and RL to enable our agent to dynamically propose step-by-step tools for long-range reasoning. To facilitate training, we construct a dataset called Ego-R1 Data, which consists of Ego-CoTT-25K for SFT and Ego-QA-4.4K for RL. Furthermore, our Ego-R1 agent is evaluated on a newly curated week-long video QA benchmark, Ego-R1 Bench, which contains human-verified QA pairs from hybrid sources. Extensive results demonstrate that the dynamic, tool-augmented chain-of-thought reasoning by our Ego-R1 Agent can effectively tackle the unique challenges of understanding ultra-long egocentric videos, significantly extending the time coverage from few hours to a week.

Preprint. Code and data are available at https://egolife-ai.github.io/Ego-R1/.

#### 1 Introduction

Egocentric videos, which capture human daily lives from a first-person perspective, are inherently long - often spanning hours to days or even weeks [75]. Understanding these videos is crucial for supporting practical tasks such as memory recall, multi-step activity tracking, and goal monitoring [5, 23, 40]. But the ensuing problem poses significant challenges due to the video length, multi-modality, and the need for long-horizon reasoning across diverse temporal contexts and dependencies.

Recent advances in multimodal long-context modeling have led to promising progress, extending video understanding capabilities from minutes to hours [7, 29, 81, 82]. However, these models still face significant computational challenges and scale poorly when applied to videos of extended durations, such as those spanning a day or longer. To this end, prior works have proposed token compression [25, 52–54, 68] or sampling-based strategies that reframe video understanding as a temporal retrieval task [48, 79]. Nevertheless, these approaches risk missing key events due to the lossy representations or incomplete temporal localization. Another line of works, commonly referred to as video agents, leverages external language models as high-level control and reasoning entities to call specialized vision modules/tools for video reasoning [63, 79, 84]. While allowing more flexible and more granular perception, these approaches still rely on predefined reasoning pipelines or fixed-order tool invocations, limiting the video lengths they can handle, i.e., up to hour-long.

To address these limitations, we propose Ego-R1, a novel framework that leverages fine-tuned large language models (LLMs) and reinforcement learning (RL) for dynamic tool-driven reasoning of ultra-long (i.e., in days and weeks) egocentric videos. The key distinction from prior video agents [63, 79, 84] designed for long-form video understanding is the dynamic tool calling of our Ego-R1 Agent, which iteratively processes both visual information and contexts to select and execute specialized perception tools on demand, based solely on previously observed content and thought to preceding sub-questions. We call such a video understanding paradigm Chain-of-Tool-Thought (CoTT) reasoning. Furthermore, unlike traditional methods that either feed the entire video to the model or select a subset of the frames, Ego-R1 utilizes a structured toolkit for perception which consist of three core modules designed specifically to facilitate efficient temporal retrieval and detailed visual comprehension. For retrieval, Hierarchical Retrieval-Augmented Generation (H-RAG) extracts timestamped, question-relevant information in the language space. For visual analysis, a specialized Video-LLM interprets localized visual contexts, while a general-purpose Vision-Language Model (VLM) extracts fine-grained visual details. Coordinated by an orchestrating LLM trained through RL, Ego-R1 enables scalable, step-by-step compositional reasoning over ultra-long videos. The modular design of our framework enables easy integration with a wide range of state-of-the-art visual understanding models, allowing the visual perception components, i.e., the Video-LLM and VLM, to seamlessly integrate into our framework.

To facilitate the training of Ego-R1, which consists of a supervised fine-tuning (SFT) stage and an RL stage, we construct Ego-R1 Data, a comprehensive hybrid-source dataset consists of 25K CoTT reasoning traces and 4.4K annotated question-answer (QA) instances to support SFT of a pretrained LLM and RL training of our Ego-R1 agent, respectively. Each task within the dataset requires reasoning over substantial temporal spans, with an average of 7.42 tool-calling steps per task. Additionally, we introduce Ego-R1 Bench, a carefully curated evaluation framework consisting of week-long egocentric videos that combine human-annotated and post-verified synthetic data, designed specifically to assess long-horizon reasoning capabilities in the egocentric setting.

Extensive experiments across diverse long-video benchmarks demonstrate that the dynamic, toolaugmented chain-of-thought reasoning by our Ego-R1 Agent can effectively tackle the unique challenges of understanding ultra-long egocentric videos, significantly extending the time coverage from few hours to a week. We also perform ablation studies to replace the visual modules in Ego-R1 to showcase that our framework is customized to integrate current MLLMs scope, validating our method’s robustness and generalization. At last, while we focus on egocentric long videos in this work, we show that our framework generalizes well in the exocentric setting as well.

#### 2 Related Work

Egocentric long video understanding. Existing large-scale egocentric datasets such as Ego4D [22], EgoExo4D [23], Epic-Kitchens [10], and HD-Epic [45] have established comprehensive bench-

- Table 1: Comparison between Ego-R1 and other frameworks. Ego-R1 develops an agentic tool-calling schema that enables interpretable reasoning over ultra-long videos while preserving critical temporal information.

Method Ultra Long Information Retention Interpretability Temporal Awareness Adaptability

LongVA [81] ✗ ✗ ✗ ✗ ✗ LLaVA-Video [82] ✗ ✗ ✗ ✗ ✗ T∗ [79] ✗ ✓ ✗ ✓ ✓ Video-RAG[37] ✗ ✓ ✗ ✓ ✓ VideoAgent [63] ✗ ✓ ✗ ✓ ✓ Video-R1 [16] ✗ ✗ ✓ ✗ ✗

Ego-R1 ✓ ✓ ✓ ✓ ✓

marks [6, 8, 40] focused on temporal understanding of daily activities, object interactions, and episodic memory tasks [12, 19, 31, 49, 55, 57]. While these benchmarks typically span only minutes, recent extensions have reached hours [4, 79] but multi-personal interactions and cross-day behavioral patterns remain unexplored. Recently, EgoLife [75] provides a week-long egocentric dataset; however, its question-answering tasks remain vanilla, lacking requirements for deep visual reasoning. Our benchmark addresses these limitations with more challenging tasks requiring sophisticated reasoning about visual details across diverse scenarios.

While egocentric datasets and benchmarks continue to expand in temporal scope, methods specifically designed for egocentric long video understanding remain absent. As shown in Table 1, existing approaches face critical limitations: proprietary models [1, 58] and some MLLMs [3, 28] usually process videos as unified inputs, which becomes prohibitively token-intensive for hour-long videos; general frame sampling approaches [34, 36, 64, 81, 82] cannot guarantee question-relevant frames selection; and sophisticated video agents [54, 63, 65, 66, 79] analyze frames in isolation, missing narrative structure and temporal dynamics. Though RAG shows promising direction for long video understanding [37, 72], existing approaches often lack contextual specificity for multi-day egocentric videos, where personal routines and social dynamics evolve over time. To address this challenge, our Ego-R1 implements multi-step reasoning upon a hierarchical RAG paradigm, enabling comprehensive understanding of evolving contexts beyond previous single thinking step approach Video-R1 [16]. A detailed qualitative results comparison is shown in Fig. 5.

Multimodal agentic tool-use. Agentic systems with Tool-Integrated Reasoning (TIR) effectively enhance LLMs’ complex problem-solving and reasoning capabilities [44, 78], particularly in mathematical domains [21, 61, 73, 85] through search engines [26, 83] and code interpreters [32, 74, 77]. For training paradigms in tool-integrated learning, RL has emerged as a promising approach offering more scalable and generalizable tool utilization strategies [15, 30, 46, 60], compared to traditional SFT [47, 50]. Recent research has extended tool-augmented foundation models to multimodal domains, exploring the integration of diverse tool-use for visual reasoning tasks [11, 27, 38, 39, 56, 84]. These initial efforts leverage specialized visual perception modules [14, 63], to enhance grounded and context-aware reasoning in complex visual environments [7, 33]. Coinciding with OpenAI’s o3 [43], Ego-R1 Agent employs dynamic tool-calling mechanisms, enabling multi-step reasoning and contextual tool selection, determining the appropriate tool for optimal problem-solving.

CoT reasoning. Chain-of-Thought (CoT) reasoning [67] has emerged as a fundamental mechanism to enhance the reasoning capabilities of both LLM and VLM [35, 59, 62, 70, 71]. RL-based reasoning approaches further require high-quality CoT samples to advance multimodal reasoning capabilities [13, 24, 76, 80]. However, existing datasets lack adequate, high-quality CoT annotations for long video understanding tasks. To fill this gap, we introduce Ego-CoTT-25K, featuring CoT reasoning with dynamic tool-calling capabilities.

#### 3 Egocentric Long Video Reasoning via Dynamic Tool-Calling

The egocentric long-video reasoning task represents a crucial frontier beyond understanding, as firstperson perspectives capture complex, temporally interdependent human behaviors over ultra-long durations. Actions that occur many hours or even days apart may be guided by consistent personal strategies and habits; thus, correctly answering a query often relies on recognizing enduring human traits and linking them to cues dispersed across the entire timeline. This requires the models to therefore maintain long-range temporal dependencies, identify subtle evidence in earlier segments, and reason about the actor’s underlying preferences to generate dynamic context-aware solutions.

Although recent MLLMs demonstrate promising performance in general video understanding, they still struggle in answering questions in truly long-context videos with extended temporal relationships. This underscores the importance of egocentric long-video reasoning as a fundamental challenge for multimodal systems. In this section, we introduce Ego-R1, a novel framework that unifies visual content comprehension and contextual reasoning by combining chain-of-thought prompting with dynamic tool calling. We provide a formal task definition in Section 3.1, followed by a comprehensive presentation of our specialized toolkit architecture designed for dynamic tool call in Section 3.2.

##### 3.1 Egocentric Long Video Reasoning Tasks

Compared to general exocentric videos, egocentric videos offer continuous, context-rich recordings from a first-person perspective, naturally documenting extensive temporal experiences including daily routines, social interactions, and object manipulations. This unique viewpoint requires sophisticated high-order inference to interpret actions, intentions, and contexts across substantial temporal spans, making it require reasoning models with strong temporal understanding and contextual integration capabilities. This necessitates a flexible reasoning framework that dynamically processes both visual information and contextual details through an intelligent tool-calling mechanism, determining which analytical approaches are most relevant for comprehending complex temporal narratives spanning multiple days of recorded experience.

In our task, we provide egocentric video spanning several days alongside questions posed at a specific query time. The system analyzes all preceding video content to generate accurate responses, simulating human temporal reasoning in real-life scenarios. This tool-based approach enables multimodal reasoning by leveraging contextual information across extended periods, requiring the system to choose optimal tools during the thinking process to effectively integrate perception, memory, and action when generating responses based solely on previously observed content.

##### 3.2 Dynamic Tool-Calling

Current MLLMs struggle with extended egocentric content due to limited context windows, inadequate temporal understanding, and insufficient structured reasoning capabilities, preventing effective analysis of long-duration egocentric videos containing sparse events that require multi-step, contextaware interpretation. To address the inherent difficulty posed by the overly long context of long-form egocentric video reasoning, we adopt a dynamic tool-calling framework that empowers the LLM rather than an MLLM to invoke specialized perception tools on demand. Our approach enables the LLM to actively decompose complex queries, selectively retrieve relevant segments, and iteratively perform stepwise reasoning grounded in video observations. This modular design overcomes the context-length bottleneck of MLLMs while enabling the fine-grained, multi-turn reasoning essential for practical egocentric video understanding. Our framework leverages three complementary tools one text-based and two visual-based - each addressing distinct temporal and perceptual dimensions of egocentric understanding. The text-based hierarchical RAG system handles longer temporal information retrieval, while the visual-based tools (Video-LLM and VLM) perform detailed visual analysis at different visual granularities.

h-rag : Our hierarchical system efficiently localizes relevant temporal information from the memory bank. Videos are first segmented into 30-second clips, each summarized via a video captioning model and temporally aligned with the ASR results as clip logs. These clip logs are hierarchically aggregated through a bottom-up generation process into multi-level granularity, creating comprehensive temporal summaries. The hierarchical structure facilitates effective top-down inference to locate and retrieve logs of relevant video segments, thus reducing computational load while preserving accuracy and temporal coherence across long egocentric videos spanning days. The system accepts specific search parameters, including temporal granularity, keywords, and time ranges for retrieval, returning the most relevant observations that match the query constraints.

video-llm : Our video-llm is a short-horizon visual-perception module that operates on local temporal windows ranging from a few seconds up to ten minutes. We sample each clip within the proposed time range at 1 FPS, keeping the input size compatible with modern multimodal language models and thus maintaining broad architectural flexibility. Given a question and its corresponding video segment, the tool correlates visual content with temporal context to produce

###### Raw QA Data Collection CoTT Generation

###### EgoLife Raw Videos (A1-A6 views)

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Question: Why is it so important to monitor the power banks and keep their lights on? <timestamp>DAY3_11175412</timestamp>

[Figure 108]

Video to Text

[Figure 109]

30s segments of A1 – A6’s log

… "view": "A1","date": "DAY4","time": "17240000-17243000", "text": "I saw Shure standing nearby, dressed in a blue hoodie, while Nicous was sitting on a chair playing the guitar, … Her expression was serious, as if she was facing a tricky problem."

<think> To answer this question, I need to… </think> <tool> Hierarchical_RAG("week", ["power banks"], ["DAY1_00000000", "DAY3_11175412"]) </tool> OBSERVATION: "DAY1": … Technical preparations like power bank distribution, device checks were addressed …

<think>

<tool>

...

Reasoning Chains

[Figure 110]

[Figure 111]

[Figure 112]

Raw MCQ Pairs

[Figure 113]

[Figure 114]

<think> Based on the observation, I need to… </think> <tool> Hierarchical_RAG("day", ["power banks"], ["DAY1_00000000", "DAY1_24000000"]) </tool> OBSERVATION: "DAY1_12100000": … The team found that the power bank has technical issues in charging …

<think>

###### Annotated MCQ Pairs

human veriﬁcatio n

[Figure 115]

<tool>

[Figure 116]

human annotated

AI generated

Optimized MCQ Pairs × 6

…

/

×𝒏

[Figure 117]

[Figure 118]

|"Query Time": ["DAY3", "11175412"] "Question": "Why is it so important to monitor the power banks and keep their lights on?”<br><br>ﬁnal MCQ pairs|
|---|

<think> Based on the above observations, the power bank was not plugged in correctly before, results in … Final Answer: B) Prevent data loss and re-recording. </think>

<think>

[Figure 119]

A) Avoid losing work time waiting ✅B) Prevent data loss and re-recording C) Maintain light proper functioning D) Save battery life for later use "Target Time": ["DAY1", "12193000 - 12200000"] "Reason": "...Shure said, 'That one isn't plugged in… 'Oh no, it's out of battery. ' He replied, 'It’s dead already. … ' You need to record it again.'"

re-generate

[Figure 120]

[Figure 121]

🔴

[Figure 122]

🔠

[Figure 123]

Veriﬁcation 🟢

<answer>

- Figure 2: Data generation pipeline of the Ego-R1 Data. We first obtained raw QA pairs from both AI-generated and human-annotated sources based on 6 raw videos collected from 6 participants and the corresponding log. The verified and processed Multiple Choice Questions (MCQs) serve as the foundation of the Ego-R1 Data (left). We take questions without answers for Chain-of-Tool-Thought (CoTT) generation, which involves creating reasoning chains that include explicit thinking steps and dynamic tool-calling sequences (right).

detailed observations that capture dynamic interactions and sequential events, and, when possible, directly answers the query for the specified time range.

vlm : This general-purpose vlm operates at the finest temporal granularity, analyzing individual frames to extract high-resolution details like text on packaging, object attributes or specific visual elements missed in broader video analysis. It augments the temporal reasoning of video-llm with precise visual evidence for comprehensive egocentric understanding.

#### 4 Ego-R1 Data: Chain-of-Tool-Thought (CoTT) for Video Reasoning

To unleash the reasoning capabilities of LLM under the CoT prompting paradigm and to enable dynamic tool selection conditioned on current observations and past actions, we introduce Ego-R1 Data, a dataset designed to enable agentic tool-use with Chain-of-Tool-Thought (CoTT) reasoning chains. Figure 2 illustrates the data generation pipeline of the Ego-R1 Data, including raw QA data collection and CoTT generation. In this section, we define the structure of CoTT in Section 4.1, and provide details of Ego-R1 Data generation in Section 4.2.

##### 4.1 Chain-of-Tool-Thought (CoTT)

Our goal is to generate synthetic CoTT data and use it to train multi-turn tool-use language models. We define a CoTT data C as a sequence of steps Si, where each step consists of a thought Tith, a tool Tito, and an observation oi. A CoTT trajectory is defined as follows:

C = (S0,S1,...,Sn), Si = Tith, Tito, oi (1)

where C is a sequence of n reasoning steps. At each step i, the agent will generate a thought Tith and a tool call Tito based on all the previous steps’ observations {o0,o1,...,oi−1} and the query q.

To formalize this reasoning process, we define two essential components that characterize how the agent operates: the action space, which specifies the available tools the agent can utilize, and the observation space, which captures the structured outputs returned from tool executions.

Action space. We define the action space A = Fj as a union of available tools to be used during reasoning. We use the three fundamental tools defined in Section 3.2: 1) h-rag for text-based long-range temporal retrieval, 2) video-llm for short-range video understanding, and 3) vlm for

framewise image understanding, plus an auxiliary terminate tool for data generation only. The h-rag tool retrieves relevant information from the current-view knowledge base by querying specified keywords within a target time window. By projecting long videos into a semantically and temporally structured language space, it rapidly pinpoints the approximate temporal interval of an event while summarizing sparse visual cues into a concise textual summary. The video-llm tool analyses short video segments specified by a query and an associated time window, providing detailed interpretations of local visual–temporal content. The vlm tool performs image-level analysis on a single frame selected by timestamp and query, providing precise, frame-specific visual details.

Observation space. At each reasoning step i, the agent receives an observation oi =

oragi , ovidi , ovlmi ∈ O, where each component oragi , ovidi , ovlmi represents the output of corresponding tool rag, video-llm, and vlm. The observation space O = {O0,O1,...,On} encompasses the collection of all tool outputs. Each tool call executes via the parsed arguments, producing observations that guide subsequent reasoning steps.

##### 4.2 Data Generation

We carefully curate Ego-R1 Data, comprising 4.4K annotated question-answer pairs sourced from over 500 hours of egocentric videos recorded across six distinct first-person perspectives. We select 2.9K high-quality questions for CoTT generation. For each selected QA pair, we construct a CoTT trace that decomposes the reasoning process into interpretable steps, yielding an average of 7.42 tool calls per task. In total, 25K CoTT traces are generated, and subsequently used during the SFT stage to train our multi-turn tool-use language model.

Ego-QA-4.4K. Long-form egocentric videos are hard to collect in nature. Following the dataset construction pipeline of EgoLifeQA [75], we collected 2.9K high-quality human-annotated data from 6 videos with distinct viewpoints. To expand the dataset scale, we employ proprietary models to analyze Automatic Speech Recognition (ASR) transcripts with video captioning outputs from the 30-second segments. These textual logs were combined and examined across various temporal granularities, spanning single or multiple days, to generate candidate questions with answers. Human annotators subsequently selected and cross-validated those QA pairs using Fleiss’ kappa [17], refining each query and its ground-truth answer according to a unified criteria of rationale coherence, importance, relevance, and difficulty level. In total, Ego-R1 Data comprises 4.4K question-answer pairs from both human-labeled and synthetic data sources.

Ego-CoTT-25K. We develop a systematic CoTT generation system to automatically generate CoTT data based on the selected question-answer pairs. By leveraging proprietary LLMs with longer context windows and stronger instruction-following capabilities, we enable the automatic generation of comprehensive reasoning chains that would otherwise be challenging to produce manually. In the CoTT generation system, each tool is exposed to the model as an executable function whose signature and semantics are implicitly embedded in the system. This design, paired with a textual system prompt (Table 5), prevents parsing errors during execution. The prompt also encodes the current viewpoint identity and enumerates the available tools. Given an input question q, the model iteratively generates reasoning steps Si = (Tith,Tito), where Tith denotes the thought and Tito denotes the corresponding tool call with fully specified arguments (e.g., time ranges, keywords, sub-questions). All the proposed arguments are validated by a pre-verification module to ensure syntactic correctness. Once a call is emitted, its name and arguments are extracted via special tokens and dispatched to an external server for execution. The returned observation is then fed back to the model, guiding the next step and enabling dynamic, multi-turn tool use for egocentric long-video reasoning.

#### 5 Ego-R1 Agent: Towards Tools Integrated Video Understanding Agent

Our goal is to train a language model capable of performing long-form video reasoning via a structured long-chain reasoning schema that automatically invokes multi-turn tool calls to collaboratively solve the problem. Inspired by the recent post-training techniques [9], we design our training framework with a two-stage strategy, with an illustration in Fig. 3.

##### 5.1 Stage 1: Supervised fine-tuning (SFT)

In the first stage, we perform SFT on a pretrained language model using the synthetic CoTT dataset. This "cold-start" initialization equips the model with the foundational ability to produce correctly

-

[Figure 124]

|External Tool-Calling<br><br>hierarchical_rag; video_llm; image_llm| |
|---|---|
| | |

Reward Model

| | |
|---|---|
|𝑆 𝑆<br><br>𝑆<br><br>...| |

CoTT Question

|𝑂 𝑂<br><br>𝑂<br><br>...|
|---|

; ;

|1|
|---|

- answer 1
- answer 2

|Supervised Finetuning| |
|---|---|
| | |

|1|
|---|

Pretrained Model

rollout

|GRPO| |
|---|---|
| | |

+ Ego-R1Agent

Ego-R1-SFT

...

;

|0|
|---|

answer n

xN

(1) Stage 1: SFT with CoTT

(2) Stage 2: GRPO for Ego-R1

- Figure 3: Overview of the two-stage training strategies in Ego-R1. Ego-R1 employs a two-stage training approach: Stage 1 utilizes supervised fine-tuning with CoTT data to establish structured tool-calling capabilities, while Stage 2 applies multi-turn reinforcement learning with rule-based rewards to optimize iterative reasoning and tool execution across diverse question types.

formatted tool calls as prescribed by the CoTT reasoning schema. The CoTT data, presented in a structured, multi-turn conversational format, simulates realistic stepwise tool interactions, explicitly combining natural language reasoning with structured tool invocation. Each step in the reasoning trajectory consists of a thought enclosed within the special token <think>...</think>, followed by either a proposed tool call, enclosed within <tool>...</tool>, or an answer, enclosed with in <answer>...</answer>. The tool call is automatically parsed and executed by an external environment, which then returns an observation. This observation is formatted and fed back into the model as part of the input for the next reasoning step. After fine-tuning, the resulting Ego-R1-SFT model reliably produces well-formed tool calls and coherent step-by-step reasoning, laying the groundwork for subsequent reinforcement learning stage.

##### 5.2 Stage 2: Reinforcement learning (RL)

To further improve the multi-turn tool-calling capabilities of our fine-tuned Ego-R1-SFT model, we adopt Gradient-Regularized Policy Optimization (GRPO) [51] to train the model. GRPO optimizes the model to maximize the expected final task reward while regularizing the variance of policy gradients across reasoning steps to encourage stable and coherent decision-making. Specifically, we define the GRPO objective as follows:

|Siy|

T

G

πθ(Si,t|q,Iy,Si,<t) πθ

1 |Siy|

1 G

Aˆyi,t,

JGRPO(θ) = E[q∼P(Q),{o

min

i}Gi=1∼πθold(O|q)]

(Si,t|q,Iy,Si,<t)

old

y=1

t=1

i=1

πθ(Si,t|q,Iy,Si,<t) πθ

,1 − ε,1 + ε A ˆyi,t − βDKL[πθ∥π0]

clip

(Si,t|q,Iy,Si,<t)

old

In this equation, πθ represents the policy model that generates reasoning tokens Siy sequentially at turn y, where i denotes the token position. The generation is conditioned on the preceding sequence

Sy ∗ < i, the observation Iy at turn y, and the question q. The final reward R ∗ final(C,q) evaluates the correctness of the answer at the end of the reasoning chain C. The reference policy π0 denotes the original model, and the KL divergence term KL(πθ|π0) regularizes the policy to prevent excessive

drift from the initial parameters. The advantage estimates Aˆyi,t are computed by standardizing rewards within each group G, subtracting the group mean and dividing by the group standard deviation.

During training, we generate rollout trajectories by sequentially executing tools based on the model’s reasoning outputs, providing realistic stepwise observations that inform subsequent reasoning steps. Each rollout terminates when either a valid final answer is produced or the maximum step limit N is reached. This training procedure enables the model to effectively generalize multi-turn tool usage, reflecting the iterative nature of egocentric long-video reasoning tasks. The resulting model after second-stage reinforcement learning training constitutes our final system, termed the Ego-R1 Agent.

- Table 2: Quantitative results on video question-answering benchmarks. The proposed Ego-R1 model demonstrates superior performance across multiple metrics. Bold indicates best performance, underscored values show second best. The results from the 72B version of the model or using less frames are marked in gray. As some of the QA pairs in EgoLifeQA were used for CoTT generation and training, we excluded these from evaluation and retained only a clean subset for fair testing.

Exocentric Egocentric

Method

Size Frames VideoMME (long) EgoSchema EgoLifeQA Ego-R1 Bench Average durations 41 min 3 min 44.3 h 44.3 h MLLMs

LongVA [81] 7B 64 45.0 44.1 33.0 23.0 LLaVA-Video [82] 7B 64 61.5 57.3 36.4 29.0 LLaVA-OneVision [28] 7B 1 FPS 60.0 60.1 30.8 31.6 InternVideo2.5 [64] 8B 512 53.4 63.9 33.0 34.0 Gemini-1.5-Pro [58] - - 67.4 72.2 36.9 38.3

RAG Methods

LLaVA-Video + Video-RAG [37] 7B 64 46.0 66.7 30.0 29.3 LongVA + Video-RAG [37] 7B 64 55.7 41.0 26.0 31.0

Reasoning Models Video-R1 [16] 7B 64 50.8 - 34.0 20.0

Video Agents

VideoAgent [63] - 8 50.8 54.1 29.2 32.6 LLaVA-OneVision + T∗ [79] 7B 8 46.3 66.6 35.4 35.6

Ours Ego-R1 3B - 64.9 68.2 36.0∗ 46.0

#### 6 Experiments

##### 6.1 Experiment Setup

To evaluate the effectiveness of the CoTT reasoning traces in answering the ultra-long video understanding question, we utilize Qwen-2.5-3B-Instruct as our base model. To mitigate the hallucination problem caused by the increasing CoTT length, we introduce an additional summary model with a longer context window length to help conclude the reasoning trace to answer the question.

Benchmarks. We evaluate the performance of Ego-R1 Agent on three existing long video understanding benchmarks covering both exocentric and egocentric views: Video-MME (long w/o subtitle) [18], EgoSchema [40], EgoLifeQA [75]. Among them, Vide-MME has a third-person view, and the rest have the first-person view. We follow the same paradigm as h-rag to generate the knowledge base for each video in these benchmarks. The hierarchy depth of each memory bank varies by datasets: only EgoLifeQA contains videos long enough to necessitate day-level summaries, while others extend to 10-minute-level or hour-level summaries at most. To further evaluate the capability of Ego-R1 Agent in handling multi-perspective and long temporal reasoning question answering tasks, we establish Ego-R1 Bench, a reasoning based benchmark for ultra-long egocentric video understanding. Distinct from Ego-R1 Data, Ego-R1 Bench comprises 300 QAs evenly distributed across six first-person perspectives. For each perspective, Ego-R1 Bench includes a balanced mixture of human-labeled and human verified QAs.

Comparison Methods. We benchmark Ego-R1 Agent against recent representative approaches, including MLLM-based video understanding methods [28, 58, 64, 81, 82], RAG-based method [37], reasoning model [16] and video agents [63, 79]. For each question, we restrict the input to video content occurring before the query timestamp, ensuring causal consistency in all comparisons. To ensure fair comparison across methods with different architectural constraints, we adopt an adaptive frame-sampling protocol: 1) Standard frame-based MLLMs [16, 81, 82] and LLaVA-OneVision [28] receive 64 uniformly sampled frames per query; 2) Video-RAG [37] uses its native setting of 64 frames; 3) Higher-capacity models such as InternVideo2.5 [64] and Gemini 1.5 Pro [58] are provided with 512 uniformly sampled frames; 4) Agent-based methods that rely on caption-guided key-frame selection [63, 79] are supplied with 1 024 uniformly sampled frames, recomposed into 1 FPS videos. This protocol equalizes input budgets while respecting each model’s architectural constraints.

##### 6.2 Results

Table 2 presents a quantitative comparison of Ego-R1 with state-of-the-art video understanding models on both exocentric and egocentric benchmarks. Ego-R1 achieves the best or second-best score on three of the four datasets, despite using far fewer parameters than most competitors.

Exocentric setting. On VideoMME (long), whose clips average 41 min, Ego-R1 achieves 64.9% accuracy, which is the highest score among open-weight models and second overall, falling behind only the proprietary Gemini-1.5-Pro (67.4%). It surpasses other public MLLMs, such as LLaVAVideo (61.5%) and InternVideo2.5 (53.4%), while using less than half their parameter count. These results indicate that, although Ego-R1 is trained in an egocentric regime, it generalizes effectively to exocentric settings.

Egocentric settings. Ego-R1 achieves the highest accuracy on the proposed egocentric long video reasoning benchmark - Ego-R1 Bench (with an average time of 44.3 h), achieves 46.0% accuracy. This result exceeds Gemini-1.5-Pro by 7.7% and surpasses the strongest open baseline, LLaVA-Video, by 17.0%, underscoring the benefit of hierarchical retrieval and multi-turn tool calling for reasoning tasks with sparsely distributed events. On EgoSchema (3 min clips), Ego-R1 records 68.2%, second only to Gemini (72.2%); on EgoLifeQA we obtain 36.0% after removing any training overlap, comparable with LLaVA-Video (36.4%) and approaching Gemini (36.9%).

Analysis. Both frame-based MLLMs and RAG variants exhibit marked performance drops on Ego-R1 Bench, and agent-based approaches remain in the 32 - 36% range, well below the 46% achieved by Ego-R1. These findings indicate that agent-based approaches provide a more effective solution for long-video reasoning tasks, while our CoTT style dynamic tool calling, enables even a compact 3B model to conduct reliable, long-horizon reasoning over hours-long egocentric video.

##### 6.3 Ablation Study

To better understand the contribution of different training components in Ego-R1, we conduct ablation studies using identical base models under varying training regimes. Specifically, we compare models trained with: (1) SFT only, (2) RL only, and (3) a combination of both. Quantitative results are reported in Table 3.

Table 3: Ablation study on different training regimes. We use Qwen-2.5-3B-Instruct as our base model to validate the effectiveness of the two training components.

Training Regimes

Format Acc.%

Base Model

Acc.%

SFT RL

– – 1.4 4.3 Qwen-2.5 – ✓ 0.0(↓1.4) 13.3(↑9.0) 3B-Instruct ✓ – 34.3(↑32.9) 100.0(↑95.7)

✓ ✓ 46.0(↑44.6) 100.0(↑95.7)

The zero-shot base model achieves only 1.4% task accuracy on Ego-R1 Bench and 4.3% format accuracy for intermediate tool calls. Interestingly, after applying vanilla RL training using GRPO without any intermediate CoTT supervision, the task accuracy drops to 0%, while tool-call format accuracy improves by 9%. This indicates that although the model can learn the structural format of tool calls during RL from their emergent capabilities, the absence of reasoning trace supervision leads to unstable or ungrounded predictions, ultimately harming task performance.

In contrast, applying SFT with CoTT data, even in limited epochs (e.g., 3 epochs), significantly improves both task and format accuracy. This highlights the importance of structured reasoning demonstrations during pretraining: they not only teach the model to produce correctly formatted tool calls, but also establish a foundation for multi-step reasoning in long-horizon tasks.

#### 7 Conclusion and Outlook

We introduce Ego-R1, a novel framework addresses challenges in long-horizon egocentric video reasoning through its Chain-of-Tool-Thought approach, which demonstrates that decomposing complex reasoning into modular, tool-grounded steps creates a more robust foundation for video understanding than traditional methods. This integration of structured reasoning with dynamic tool-calling not only enhances model interpretability but also reveals promising directions for future multimodal AI systems. The superior performance across temporal coverage and information

retention suggests that hybrid architectures combining symbolic and neural components may be essential for tackling open-world, ultra-long video understanding. Beyond immediate applications in egocentric video analysis, Ego-R1 points toward broader implications for human-AI collaborative systems where transparent reasoning processes are critical, boosting the potential for life-oriented assistants that accompany humans over very long timeframes, making them genuinely useful and seamlessly integrated into everyday life.

#### 8 Acknowledgment

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/ claude-3-5-sonnet/.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Cristóbal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Fei-Fei Li. Hourvideo: 1-hour videolanguage understanding. Advances in Neural Information Processing Systems, 37:53168–53197, 2024.
- [5] Qirui Chen, Shangzhe Di, and Weidi Xie. Grounded multi-hop videoqa in long-form egocentric videos, 2024.
- [6] Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu. Egoplan-bench: Benchmarking egocentric embodied planning with multimodal large language models. CoRR, 2023.
- [7] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024.
- [8] Sijie Cheng, Zhicheng Guo, Jingwen Wu, Kechen Fang, Peng Li, Huaping Liu, and Yang Liu. Egothink: Evaluating first-person perspective thinking capability of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14291–14302, 2024.
- [9] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training, 2025.
- [10] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European conference on computer vision (ECCV), pages 720–736, 2018.
- [11] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative selfimprovement. arXiv preprint arXiv:2503.17352, 2025.
- [12] Shangzhe Di and Weidi Xie. Grounded question-answering in long egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12934–12943, 2024.
- [13] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. arXiv preprint arXiv:2411.14432, 2024.
- [14] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision, pages 75–92. Springer, 2025.
- [15] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536, 2025.

- [16] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [17] Joseph L Fleiss and Jacob Cohen. The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability. Educational and psychological measurement, 33(3):613–619, 1973.
- [18] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [19] Gabriele Goletto, Tushar Nagarajan, Giuseppe Averta, and Dima Damen. Amego: Active memory from long egocentric videos. In European Conference on Computer Vision, pages 92–110. Springer, 2024.
- [20] Google DeepMind. Gemini 2.5: Our most intelligent ai model, 2025.
- [21] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.
- [22] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.
- [23] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400, 2024.
- [24] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [25] Jindong Jiang, Xiuyu Li, Zhijian Liu, Muyang Li, Guo Chen, Zhiqi Li, De-An Huang, Guilin Liu, Zhiding Yu, Kurt Keutzer, Sungjin Ahn, Jan Kautz, Hongxu Yin, Yao Lu, Song Han, and Wonmin Byeon. Token-efficient long video understanding for multimodal llms, 2025.
- [26] Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [27] Fucai Ke, Xingjian Leng, Zhixi Cai, Zaid Khan, Weiqing Wang, Pari Delir Haghighi, Hamid Rezatofighi, Manmohan Chandraker, et al. Dwim: Towards tool-aware visual reasoning via discrepancy-aware workflow generation & instruct-masking tuning. arXiv preprint arXiv:2503.19263, 2025.
- [28] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [29] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2024.
- [30] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl. arXiv preprint arXiv:2503.23383, 2025.
- [31] Yuxuan Li, Vijay Veerabadran, Michael L Iuzzolino, Brett D Roads, Asli Celikyilmaz, and Karl Ridgeway. Egotom: Benchmarking theory of mind reasoning from egocentric videos. arXiv preprint arXiv:2503.22152, 2025.

- [32] Minpeng Liao, Wei Luo, Chengxi Li, Jing Wu, and Kai Fan. Mario: Math reasoning with code interpreter output–a reproducible pipeline. arXiv preprint arXiv:2401.08190, 2024.
- [33] Benlin Liu, Yuhao Dong, Yiqin Wang, Yongming Rao, Yansong Tang, Wei-Chiu Ma, and Ranjay Krishna. Coarse correspondence elicit 3d spacetime understanding in multimodal language model. arXiv preprint arXiv:2408.00754, 2024.
- [34] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: Ondemand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024.
- [35] Zuyan Liu, Yuhao Dong, Yongming Rao, Jie Zhou, and Jiwen Lu. Chain-of-spot: Interactive reasoning improves large vision-language models. arXiv preprint arXiv:2403.12966, 2024.
- [36] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment. arXiv preprint arXiv:2502.04328, 2025.
- [37] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. Video-rag: Visually-aligned retrieval-augmented long video comprehension. arXiv preprint arXiv:2411.13093, 2024.
- [38] Zixian Ma, Jianguo Zhang, Zhiwei Liu, Jieyu Zhang, Juntao Tan, Manli Shu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Caiming Xiong, Ranjay Krishna, and Silvio Savarese. Taco: Learning multi-modal action models with synthetic chains-of-thought-and-action, 2024.
- [39] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models, 2024.
- [40] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023.
- [41] OpenAI. Hello gpt-4o, 2024.
- [42] OpenAI. Introducing gpt-4.1 in the api, 2025.
- [43] OpenAI. Openai o3 and o4-mini system card, 2025.
- [44] Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. arXiv preprint arXiv:2205.12255, 2022.
- [45] Toby Perrett, Ahmad Darkhalil, Saptarshi Sinha, Omar Emara, Sam Pollard, Kranti Parida, Kaiting Liu, Prajwal Gatti, Siddhant Bansal, Kevin Flanagan, et al. Hd-epic: A highly-detailed egocentric video dataset. arXiv preprint arXiv:2502.04144, 2025.
- [46] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. arXiv preprint arXiv:2504.13958, 2025.
- [47] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.
- [48] Tianyuan Qu, Longxiang Tang, Bohao Peng, Senqiao Yang, Bei Yu, and Jiaya Jia. Does your vision-language model get lost in the long video sampling dilemma?, 2025.
- [49] Ivan Rodin, Antonino Furnari, Kyle Min, Subarna Tripathi, and Giovanni Maria Farinella. Action scene graphs for long-form understanding of egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18622–18632, 2024.
- [50] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539– 68551, 2023.

- [51] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [52] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding, 2024.
- [53] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding, 2024.
- [54] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, Yan Lu, Jenq-Neng Hwang, and Gaoang Wang. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18221–18232, June 2024.
- [55] Yale Song, Eugene Byrne, Tushar Nagarajan, Huiyu Wang, Miguel Martin, and Lorenzo Torresani. Ego4d goal-step: Toward hierarchical understanding of procedural activities. Advances in Neural Information Processing Systems, 36:38863–38886, 2023.
- [56] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, and Yu Cheng. Openthinkimg: Learning to think with images via visual tool reinforcement learning, 2025.
- [57] Hao Tang, Kevin J Liang, Kristen Grauman, Matt Feiszli, and Weiyao Wang. Egotracks: A long-term egocentric visual object tracking dataset. Advances in Neural Information Processing Systems, 36:75716–75739, 2023.
- [58] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [59] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamav-o1: Rethinking step-by-step visual reasoning in llms. arXiv preprint arXiv:2501.06186, 2025.
- [60] Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, and Heng Ji. Otc: Optimal tool calls via reinforcement learning. arXiv preprint arXiv:2504.14870, 2025.
- [61] Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. arXiv preprint arXiv:2310.03731, 2023.
- [62] Ruiqi Wang and Hao Zhang. Resanything: Attribute prompting for arbitrary referring segmentation. arXiv preprint arXiv:2505.02867, 2025.
- [63] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76. Springer, 2024.
- [64] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025.
- [65] Ying Wang, Yanlai Yang, and Mengye Ren. Lifelongmemory: Leveraging llms for answering queries in long-form egocentric videos. arXiv preprint arXiv:2312.05269, 2023.

- [66] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. arXiv preprint arXiv:2405.19209, 2024.
- [67] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [68] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models, 2024.
- [69] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation, 2023.
- [70] Qiong Wu, Xiangcong Yang, Yiyi Zhou, Chenxin Fang, Baiyang Song, Xiaoshuai Sun, and Rongrong Ji. Grounded chain-of-thought for multimodal large language models. arXiv preprint arXiv:2503.12799, 2025.
- [71] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.
- [72] Jilan Xu, Yifei Huang, Junlin Hou, Guo Chen, Yuejie Zhang, Rui Feng, and Weidi Xie. Retrieval-augmented egocentric video captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13525–13536, 2024.
- [73] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.
- [74] Jingkang Yang, Yuhao Dong, Shuai Liu, Bo Li, Ziyue Wang, Haoran Tan, Chencheng Jiang, Jiamu Kang, Yuanhan Zhang, Kaiyang Zhou, et al. Octopus: Embodied vision-language programmer from environmental feedback. In European Conference on Computer Vision, pages 20–38. Springer, 2024.
- [75] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, Bei Ouyang, Zhengyu Lin, Marco Cominelli, Zhongang Cai, Yuanhan Zhang, Peiyuan Zhang, Fangzhou Hong, Joerg Widmer, Francesco Gringoli, Lei Yang, Bo Li, and Ziwei Liu. Egolife: Towards egocentric life assistant, 2025.
- [76] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.
- [77] Jiarui Yao, Ruida Wang, and Tong Zhang. Fans–formal answer selection for natural language math reasoning using lean4. arXiv preprint arXiv:2503.03238, 2025.
- [78] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.
- [79] Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, Jiajun Wu, and Manling Li. Re-thinking temporal search for long-form video understanding, 2025.
- [80] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025.
- [81] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.

- [82] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024.
- [83] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. arXiv preprint arXiv:2504.03160, 2025.
- [84] Zhuo Zhi, Qiangqiang Wu, Minghe shen, Wenbo Li, Yinchuan Li, Kun Shao, and Kaiwen Zhou. Videoagent2: Enhancing the llm-based agent system for long-form video understanding by uncertainty-aware cot, 2025.
- [85] Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921, 2023.

#### Appendix

The supplementary document provides (1) details of the hierarchical RAG during the dynamic toolcalling in Section A; (2) comprehensive implementation details including the prompts we used for data generation and training, experiment setup in Section B; (3) additional experiments and ablation studies in Section C; (4) future works D, respectively.

#### A Hierarchical RAG

As mentioned in the paper, during the dynamic tool-calling, our framework leverages three complementary tools - one text-based and two visual-based - each addressing distinct temporal and perceptual dimensions of egocentric understanding. In this section, we provide additional details.

To facilitate more efficient and accurate reasoning over extremely long videos, we construct a hierarchical RAG system, as shown in Figure 4. Specifically, for each video V , we first segment it into 30-second clips vi, while preserving natural recording boundaries. The memory bank of the RAG system is built upon these 30-second clips. For each clip vi, we employ a VLM (Gemini 1.5 Pro [58]) to generate comprehensive summaries Sclip,i that include both dense captions of visual content and transcripts of spoken dialogue. We then leverage an LLM (GPT 4 [1]) to progressively synthesize these fine-grained summaries into increasingly coarser temporal resolutions, while respecting natural recording boundaries at each level. Typically, the clip-level summaries are combined into 10-minute summaries S10min,m, which are further aggregated into hourly summaries summaries Shour,h, and finally sorted into day-level summaries Sday,d.

The hierarchical RAG system serves as a critical component within our CoTT framework for long video reasoning. During the reasoning process, when a <think> step determines that the RAG system is the optimal tool to retrieve the related information, it formulates a query q = (level, [keywords], starting_timestamp, query_timestamp). Here, level designates the temporal granularity, with ‘week’ targeting a specific day within the week, ‘day’ targeting a specific hour range within a day, and ‘hour’ targeting a specific 10-minute segment within an hour. The keywords specify the search terms, while timestamps are represented with precise DAY_X specification and HHMMSSss format. The subsequent <tool> step passes this query to the RAG system, initiating the hierarchical retrieval process. The retrieval follows a top-down approach, cascading from the specified entry level through the hierarchy and returning relevant summaries. Such hierarchical navigation aligns naturally with the temporal structure of extremely long egocentric videos, which inherently follow daily patterns of human activities. Keywords retrieval at each stage is conducted through LLM-based (GPT4) keyword matching on the textual summaries, with timeindexed metadata maintained throughout the hierarchy to enable fast localization. Importantly, the

<think> step determines both the initial level for the search and whether to continue to finer levels based on the summaries returned. When sufficient information is found at a coarser level (day or hour), the <think> step may choose not to proceed to finer granularities. After receiving summaries from the RAG system, the subsequent <think> step evaluates these results and determines whether to extract answers directly or employ additional tools for further analysis and evidence localization. This hierarchical approach significantly reduces computational overhead by avoiding exhaustive search across the entire video corpus and by terminating the search at the earliest level that yields sufficient information, while maintaining high keywords retrieval accuracy through the preservation of temporal relationships.

#### B Implementation Details

##### B.1 Environment Setup

We use Gemini 1.5 Pro [58] as the backbone of Video-LLM, and GPT-4o [41] as the backbone of VLM. For CoTT data generation and keywords retrieval in our hierarchical RAG system, we utilize GPT-4.1 [42] as the routing LLM. During raw QA data collection for Ego-R1 Data and Ego-R1 Bench, we use Gemini 1.5 Pro for video-to-text processing, Gemini 2.5 Pro [20] for raw MCQ pairs generation, and Claude3.5 Sonnet [2] for MCQ pairs post-processing.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

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

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

… … … … … …

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
|[Figure 153]<br><br>[Figure 154]<br><br>30s<br><br>|[Figure 155]<br><br>[Figure 156]<br><br>30s|…|[Figure 157]<br><br>[Figure 158]<br><br>30s|[Figure 159]<br><br>[Figure 160]<br><br>30s|…|[Figure 161]<br><br>[Figure 162]<br><br>30s|[Figure 163]<br><br>[Figure 164]<br><br>30s| |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
|[Figure 165]<br><br>[Figure 166]<br><br>30s|…|[Figure 167]<br><br>[Figure 168]<br><br>30s|[Figure 169]<br><br>[Figure 170]<br><br>30s|…|[Figure 171]<br><br>[Figure 172]<br><br>30s|[Figure 173]<br><br>[Figure 174]<br><br>30s|…|[Figure 175]<br><br>[Figure 176]<br><br>30s|…|

###### 44.3 hrs

[Figure 177]

[Figure 178]

[Figure 179]

…

30s clip 𝒗𝒊

30s

[Figure 180]

30 Seconds … 30 Seconds 30 Seconds … 30 Seconds 30 Seconds … 30 Seconds 30 Seconds … 30 Seconds 30 Seconds … 30 Seconds 30 Seconds … 30 Seconds

30 Seconds

𝑺 𝒄𝒍𝒊𝒑,𝒊

Memory BankGeneration Keywords Retrieval

### … 10Minutes … 10Minutes … 10Minutes … 10Minutes … 10Minutes 1Hour … 1Hour … 1Hour … 1Hour DAY1 DAY2 … DAY6 DAY7

[Figure 181]

10 Minutes

𝑺 𝟏𝟎𝒎𝒊𝒏,𝒎

[Figure 182]

𝑺 𝒉𝒐𝒖𝒓,𝒉

[Figure 183]

𝑺 𝒅𝒂𝒚,𝒅

- Figure 4: Overview of the Hierarchical RAG system. Based on the raw video and its 30-second clips, we generate the memory bank for each video from its 30-second-level summaries to day-level summaries. During the keywords retrieval, the system searches efficiently by starting with day-level summaries and drilling down to 10-minute segments as needed.

We conducted the training of SFT and RL on 4 NVIDIA 80GB A100 GPUs, and experiments for baseline comparison on 1 NVIDIA 80GB A100 GPU.

##### B.2 Data Generation

We leverage a proprietary model GPT-4.1, with the AutoGen framework [69] to systematically generate the CoTT data. The AutoGen framework supports tool-use without parse or execution failures by leveraging structured message passing and function calling via standardized protocols like OpenAI function-calling or JSON schema-based interfaces. Unlike prompt-only approaches that rely on natural language parsing, AutoGen agents interact with tools through well-defined wrapped functions, ensuring that function arguments are syntactically and semantically valid before execution. The framework includes built-in validation, exception handling, and modular components (e.g., user proxy, assistant agent, and tool agent) that collaboratively manage errors, retries, and fallbacks. This design reduces the likelihood of malformed calls and enhances robustness in executing complex multi-step tool-use workflows. We have attached the prompts used for data generation in Table 5

##### B.3 RAG Construction for Other Benchmarks

One key component of our framework is the personalized RAG system tailored to each egocentric perspective, which requires additional effort to build and maintain. To ensure fair comparisons across benchmarks, we adapt the RAG construction strategy to the temporal characteristics of each dataset. Since the average timespan per question varies across benchmarks, we adjust the indexing granularity accordingly that proportional to the average length of the benchmark.

For the EgoLife [75] benchmark, where the recording time span and task setting align with our framework with average time span for 44.3 hours, we adopt a hierarchical RAG structure with consistent temporal levels: week → day → hour → 10-minute. This allows for flexible, top-down retrieval over long egocentric sequences while preserving temporal precision. For other benchmarks with shorter video durations or coarser question alignment, we use simplified RAG levels to match the dataset’s temporal scope.

For the VideoMME [18] benchmark, we tested on the long video subset, which has an average time span of 41 minutes. We divide videos into 30-second segments, and use Gemini-1.5-Pro to summarize each segment by combining visual content and transcribed dialogue. We then use GPT-4.1 to hierarchically aggregate these segments into 10-minute summaries. The hierarchical RAG’s temporal structure would be: 10-minute → 30-second. This two-level summarization enables our RAG system to be adapted to a shorter task duration while maintaining hierarchy.

For EgoSchema [40] benchmark, we segment the videos into 30-second clips and summarize each using LLaVA-video-7b [82], operating on 1 FPS sampled frames. The summaries are written in the first person to reflect egocentric perspective and are used as memory entries in our RAG system.

- Table 4: Ablation studies on tool use. (Left) Stronger visual understanding (Gemini-1.5-Pro) improves performance on Ego-R1 Bench. (Right) Using only RAG degrades performance, confirming the need to combine retrieval and visual tools for long-horizon video reasoning.

Method Video_LLM Ego-R1 Bench Ego-R1

Method Tool-used Ego-R1 Bench Ego-R1

LLaVA-Video [82] 43.7 Gemini-1.5-Pro [58] 46.0

RAG only 39.7 Full 46.0

Given the limited timespan of questions in this dataset, no additional hierarchical summarization is applied beyond the clip level.

#### C Additional Experiments

- C.1 Ablation Studies

We have conducted extensive experiments to validate the usage of different tools in Ego-R1. Table 4 presents ablation studies analyzing the impact of different tool-use configurations in Ego-R1. In the left subtable, we compare the effect of using different video_llm modules. Replacing LLaVAVideo with Gemini-1.5-Pro leads to a fair improvement on Ego-R1 Bench (from 43.7% to 46.0%), demonstrating the benefit of stronger localized visual understanding generally improves the overall . It further validates that our framework is modular and adaptive to different MLLM backbones. In the right subtable, we examine the contribution of the full toolset. Using only the RAG module results in a performance drop to 39.7%, highlighting the importance of combining both retrieval and visual grounding tools for effective long-horizon reasoning. These results collectively validate the necessity of a modular, multi-tool architecture in achieving robust performance on egocentric long video tasks.

- C.2 Qualitative Results

In the attached demo video, we show two examples of our step-by-step reasoning with agentic tool-use. Additionally, we present the reasoning comparison with Video-R1 [16] in Figure 5. As illustrated in the visual demos, our method offers a more interpretable reasoning process through the explicit CoTT outputs. These structured reasoning chains reveal how collaborative tool use contributes to improved performance on complex video reasoning tasks. A common failure case is shown in the last example in Figure 5. Although the observation from step 1 indicates the agent has successfully located the time range that possibly has relevant information, in the following step, the agent does not explore further from the observed time range.

#### D Future Works

Despite the possible directions emerging in the insights mentioned above, there are other possible directions to be explored based on the current dataset: Social behavior tasks. This dataset include different view of videos, as well as including some collaboration tasks that achieved by group of people.

Social-behaviour analysis. Because the dataset contains synchronized recordings from multiple viewpoints, it can support tasks that model collaborative activities and social dynamics. For example, inferring group intentions, role allocations, or inter-person dependencies during joint tasks.

Personal habits tracker. A key feature of egocentric data is its tight linkage to a single, specific individual, whose routine actions reveal stable behavioral patterns. In other words, there are some special patterns for each person, for example, whether a subject brushes their teeth before or after breakfast could inform personalized reasoning models that use long-term behavioral priors to predict future actions and preferences, ultimately improving action inference accuracy.

###### Q: Which supermarket did we enter? <timestamp>DAY3_16391802</timestamp> A. Carrefour B. Hema Fresh C. Walmart D. Costco

- CASE 1
- CASE 2
- CASE 3
- CASE 4

…

<think> Let me think about this for a moment. The question is asking which … First, let’s break it down and consider each option. A. Carrefour – is a major supermarket chain. I’ve been to Carrefour many times, but I’m not sure if it was the one we went to last time. It’s a common choice, but I need to verify if it fits the context. B. Hema Fresh – is a Dutch supermarket chain. I’ve heard of it, but I’m not sure if we’ve visited one recently. It’s a bit unusual for me to go there, so I doubt it’s the one. C. Walmart – is another major supermarket chain. I’ve been to Walmart several times … It’s a familiar place, but I need to consider …. D. Costco – is a well-known warehouse club, but I don’t recall entering one recently… Hmm, let’s think about the context… The most likely option is Carrefour, as it’s a common choice. </think>

<think> To answer the question, I need to determine the most recent instance before this timestamp when we entered a supermarket, and identify which supermarket it was. . . First, I will search at the “day” level backward from DAY3_16391802, using the keyword “supermarket” to identify … </think>

<think> The content includes a related log that the group decided to visit a nearby Hema store … To confirm this and ensure accuracy, need to narrow the temporal search … I will use the rag tool at the ”hour” level, with keywords “Hema Fresh” and “supermarket” to confirm the details of the visit. </think>

<think> Based on the sequence of reasoning and the results from the previous actions, the day-level and hour-level rag searches both mentioned the group checking out at a nearby Hema store, discussing going there for barbecue ingredients. Therefore, I will output the answer as <B> </think>

<tool> Hierarchical_RAG(”day", ["supermarket"], ["DAY3_00000000", "DAY3_16391802"])</tool>

<tool> Hierarchical_RAG(”hour", ["supermarket”, “Hema Fresh”], ["DAY3_15400000", "DAY3_16391802"])</tool>

OBSERVATION: "DAY3_15400000": The group completed their supermarket purchases, with Jake … We excited … settling on visiting a

OBSERVATION: "DAY3_15400000": As the weather remained hot, we discussed walking to a Hema store nearby, with Katrina checking the distance and confirming the direction ...

[Figure 184]

<answer>B.HemaFresh</answer> ✅

❌ nearbyHemastoreforbarbecueingredients.

[Figure 185]

###### <answer> A. Carrefour </answer> Video-R1 Ego-R1

Q: What fruit are we planning to buy? <timestamp>DAY5_16104514</timestamp> A. Orange B. Strawberry C. Apple D. Blueberry

…

<think> Let me think about this for a moment. The question is asking which … First, let’s break it down and consider each option. A. Orange – a versatile fruit that can be eaten as a snack, used in juices. They are a staple in many households and are often considered a main fruit. B. Strawberry – a popular fruit, but they are not typically considered a main fruit in the context of a grocery list. They are often used as a toping or in desserts. C. Apple – another popular fruit that can be eaten as a snack … D. Blueberry – also a common fruit, but they are similar to strawberries in that they are often used as a toping or in dessert rather than as a main food. …. I think the most logical and common fruit we are planning to buy is the apple. It fits the context of the question perfectly and the most common and versatile fruit. </think>

<think> To answer the question about which fruit is being planned for purchase, I need to determine if there was any discussion or plan related to buying fruit around that time. Start by searching with RAG at day level with keywords like “fruit”, “buy”, “shopping” </think>

OBSERVATION: There’s a lot of conversation focused on the durian near the end of the escalator at about [00:00:17-00:00:20] and again in the produce section…

<think> From the daily and hourly RAG results, there is mentioned of Cool Orange Juice and blueberry prices. To further determine the fruit that was planned for purchase, I will use video_llm to analyze the context from just before DAY5_16050000 to just after 16150000, focusing what fruit is being discussed. </think>

<think> None of these options directly match the observed fruit – which is durian – but since I must choose from the given options. Based on the daily context of “comparing blueberry prices”, if forced to pick one that is most contextually plausible in this shopping segment, the closest match by <D> </think>

<tool> Hierarchical_RAG(”day", [”fruit”, “buy”, “shopping”], ["DAY5_00000000", "DAY5_16104514"])</tool>

OBSERVATION: "DAY5_15500000": During shopping, I proposed adding … settled on Cool Orange Juice… The group compare blueberry

<tool> video_llm("Which specific fruit is being planned for purchase or focus on by anyone in this shopping interval?"])</tool>

[Figure 186]

<answer>D.Blueberry </answer> ✅

❌ prices,finalizedpurchases…

[Figure 187]

<answer> C. Apple </answer>

Video-R1 Ego-R1

Q: Why does the narrator mention frequently needing to do laundry because of oil stains? <timestamp>DAY6_21045900</timestamp> A. Their clothes are prone to oil absorption B. They spilled oil while cooking. C. They got oil on their clothes when repairing the car D. They touched oil stains while clearing the pipe

…

<think> Let me think about this for a moment. The narrator mentions needing to do laundry frequently because of oil stains. Now, let’s break it down and consider each option carefully. A. … – This seems possible, but it’s not as specific as other options. B. … – This is less likely. While it’s possible to spill oil while cooking, it’s not as common or frequent as getting oil on clothes while repairing a car. C. … –This seems plausible. It’s common to get oil on clothes, and it would require frequent laundry to clean them. D. … – This is also less likely. It’s not common or frequent as getting oil on clothes while cleaning a pipe. The narrator likely got oil on their clothes while working on the car, which would require frequently laundry. Oh, I see. The most accurate and specific reason is indeed related to the car repair activity. </think>

<think> To answer the question, I need to determine the actual context around oil stains and their association with laundry. The question is about reasoning from events or practices. I will use the rag tool to search for keywords related to “oil stains”, “laundry”, and possibly events at day level. </think>

<think> The rag search returned only general information about kitchen logistics, appliance troubleshooting and group dynamics, but did not specify any incident involving oil stains… I will use video_llm tool to analyze the sequence and ask if there was a spill during cooking or another clear explanation. </think>

<think> Based on the sequence of reasoning and the results from the previous actions, the day level RAG and video llm show the cooking related conversation and kitchen logistics. Although there is no direct evidence showing that laundry is for oil spill during cooking, it is the most likely option: <B> </think>

<tool> Hierarchical_RAG(”day", [”oil stains”, “laundry”, “clothes”, “wash”], ["DAY6_00000000", "DAY6_21045900"])</tool>

<tool> video_llm(”What happened related to oil stains or laundry in the oments before and after this timestamp"])</tool>

OBSERVATION: "DAY6_14400000": I shifted focus to kitchen logistics, troubleshooting appliances and managing water dispenser

OBSERVATION: From [00:00:55 to 00:01:07] the narrator says, “I like to do laundry… I get oil on y clothes every day and do laundry every day… should be more carefully when cooking ..

[Figure 188]

<answer>B.Theyspilledoil</answer>✅

❌ issueswhileoverseenggroupdynamics...

[Figure 189]

<answer> C. They got oil … when repairing the car </answer>

Video-R1 Ego-R1

Q: Where were the purple flowers on the table before? <timestamp>DAY2_17505703</timestamp> A. Living room. B. Garden C. Dining room D. Kitchen

## …

<think> Let me think about this for a moment. Thie question is asking where … First, let’s break it down and consider each option. A. Living room – a common place to have flowers on a table, but it’s not the most likely place for flowers to be before being placed. B. Garden - a place where flowers are typically grown, and it’s common for people to bring flowers from the garden into the house. This seems like a plausible option. C. Dining room – a place for the where flowers are often placed on a table, but it’s not the most likely place for flowers to be before being placed on a table. D. Kitchen – it’s possible that the flowers were in the kitchen, but it’s not a common place to find flowers on a table in a kitchen. … Overall, the flowers were most likely in the garden before being placed on the table </think>

<think> From the day-level rag search, I found a relevant log about a flower arrangement task, … to refine the search for exact history of its location, …. I will use the rag tool at the hour level, searching for “purple flowers”, “table”, “move”. </think>

<think> To answer the question, I need to determine the prior location of the purple flowers before they appeared on the table observed at the given timestamp. … My first action will be to use the rag tool with the keywords “purple flowers”, “table”, “move” at the day level, and down to granular if ..</think>

<think> In this frame, the purple flowers are visible in a vase on the left side of the table. The flowers are placed in a vase near other objects like chargers and wires, which suggested they might have been there for a while and are part of the setup. Therefore, it seems more likely that the flowers has been there for some time rather than just beging brought in. <A> </think>

<tool> Hierarchical_RAG(”hour", [”purple flowers”, “table”, “move”], ["DAY2_17000000", "DAY2_17505703"])</tool>

<tool> Hierarchical_RAG(”day", [”purple flowers”, “table”, “move”], ["DAY2_00000000", "DAY2_17505703"])</tool>

OBSERVATION: "DAY2_17200000": I observed Alice as she arranged flowers and discussed the arrangement with her, comparing the vases … as Lucia. Prompted me to operate near the flowers …

OBSERVATION: "DAY2_15500000": … Katrina proposed flower arrangements,….I trimmed flowers, handed scissors to Alice, and finalized

[Figure 190]

❌

<answer> A. Living room </answer>

❌ ownershipdeclarationsdebatedplacement…

[Figure 191]

###### <answer> B. Garden </answer> Video-R1 Ego-R1

- Figure 5: Qualitative results comparison with Video-R1. Case 1-3 illustrate successful examples where Ego-R1 outperforms Video-R1 by producing more detailed, interpretable step-by-step reasoning chains through dynamic tool-calling. In contrast, Case 4 highlights a failure case from Ego-R1 Agent. Although the observation in Step 1 correctly identified relevant information near timestamp DAY2_15500000, the subsequent tool call failed to adjust the temporal range accordingly, resulting in an incorrect or suboptimal retrieval in the next step, leading to the final error answer.

- Table 5: System prompt for data generation. Tool-call functions have been designed inside the AutoGen framework so that the agent is aware of how to use them.

[BEGIN OF GOAL] You are an expert AI assistant specializing in analyzing human behavior and reasoning from egocentric video descriptions. You will be provided with a list of useful tools to help in reasoning the task, and your goal is to solve the user’s question. The user’s question is following the format: Question: <question> <timestamp> Options: <options>. You can either rely on your own capabilities or perform actions with external tools to help you. You should consider both the frequency and cost of each tool to make the best decision. [END OF GOAL]

[BEGIN OF FORMAT INSTRUCTIONS] When answering questions: 1. You will be provided with previous actions you have taken, based on these actions, think step-by-step about how to approach the problem. 2. Show your reasoning process clearly before providing your next action. 3. The video observation length is 10-min max. 4. For visual questions, use ‘video_llm‘ and ‘vlm‘ to explore the visual context. 5. For temporal questions, use ‘rag‘ to explore the context before and after the event. 6. Only use the ‘terminate‘ tool after you have thoroughly explored the question with multiple tools. [END OF FORMAT INSTRUCTIONS]

[BEGIN OF HINTS]

1. All tools provided are crucial to the solvement of the question. You MUST exploit the usage of all tools before answering the question. 2. You may want to use the same tool multiple times with different arguments to explore the problem from different angles, if needed. 3. Make a balance between the cost and the frequency of the tools. 4. Usually, solving a question requires over 5 10 steps of reasoning, and follows a hierarchical calling structure: rag => video_llm => vlm. 5. Do not use the terminate tool too early. Instead, try to explore the question with the available tools, and only use the terminate tool when you are confident enough or have considered all the options. [END OF HINTS]

Always structure your responses with your thought process first, followed by any tool calls. Think before you act. Think stepby-step about what information you need and which tool to use, then execute your plan exactly as reasoned without deviation. Output your thought process before using the tool, and you must strictly follow your thought process for the tool call. Currently, you are under the view of {identity}.

- Table 6: System prompt used during training. We explicitly define the function-calling syntax and tool usage format, guiding the model to generate structured reasoning steps and valid tool calls that align with CoTT.

INSTRUCTIONS Answer the given question. You must conduct reasoning inside <think> and </think> first every time before you get new information. After reasoning, if you find you lack some knowledge, you can call a tool from [rag, video_llm, vlm] by <tool> query </tool> and it will return the information between <information> and </information>. You can use tools as many times as your want. If you find no further external knowledge needed, you can provide the answer inside <answer> and </answer> after another thinking. The tools you can use are:

- 1 {

- 2 "name": "rag",

- 3 "description": "Use this tool to search for information in the RAG database.",

- 4 "arguments": {

- 5 "type": "object",

- 6 "properties": {

- 7 "level": {

- 8 "type": "str",

- 9 "description": "The granularity of the search, choose from week|day|hour"

- 10 },

- 11 "keywords": {

- 12 "type": "List[str]",

- 13 "description": "The keywords to search for in the RAG database."

- 14 },

- 15 "start_time": {

- 16 "type": "str",

- 17 "description": "The timestamp of the start time of the search. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))."

- 18 },

- 19 "query_time": {

- 20 "type": "str",

- 21 "description": "The timestamp of the query that was proposed by the user."

- 22 }

- 23 },

- 24 "required": ["level", "keywords", "start_time", "query_time"]

- 25 }

- 26 }

- 27 {

- 28 "name": "video_llm",

- 29 "description": "Use this tool to get the answer from the video language model.",

- 30 "arguments": {

- 31 "type": "object",

- 32 "properties": {

- 33 "question": {

- 34 "type": "str",

- 35 "description": "The question you want to use the video language model to answer."

- 36 },

- 37 "range": {

- 38 "type": "str",

- 39 "description": "The timestamp range of the video to answer the question. Use the format ’DAYX_HHMMSSFF -DAYX_HHMMSSFF ’. The ending timestamp should be strictly larger than the start timestamp. The length of the range should be smaller than 10 minutes, greater than 1 second."

- 40 }

- 41 },

- 42 "required": ["question", "range"]

- 43 }

- 44 }

- 45 {

- 46 "name": "vlm",

- 47 "description": "Use this tool to get the answer from the vision language model.",

- 48 "arguments": {

- 49 "type": "object",

- 50 "properties": {

- 51 "question": {

- 52 "type": "str",

- 53 "description": "The question you want to use the vision language model to answer."

- 54 },

- 55 "timestamp": {

- 56 "type": "str",

- 57 "description": "The timestamp of the video to answer the question."

- 58 }

- 59 },

- 60 "required": ["question", "timestamp"]

- 61 }

- 62 } For example, if you want to search for information in the RAG database, you can use the following tool:

- 1 <tool >

- 2 {

- 3 "name": "rag",

- 4 "arguments": {

- 5 "level": "day",

- 6 "keywords": ["screwdriver", "applause"],

- 7 "start_time": "DAY1_11210217",

- 8 "query_time": "DAY1_11220217"

- 9 }

- 10 }

- 11 </tool >

