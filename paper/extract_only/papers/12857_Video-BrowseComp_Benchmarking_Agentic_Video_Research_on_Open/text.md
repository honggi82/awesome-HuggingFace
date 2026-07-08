## Video-Browser: Towards Agentic Open-web Video Browsing

Zhengyang Liang1,♣, Yan Shu2,♣, Xiangrui Liu3, Minghao Qin3, Nicu Sebe2, Zheng Liu4, Lizi Liao1 1Singapore Management University 2University of Trento 3Beijing Academy of Artificial Intelligence 4Hong Kong Polytechnic University ♣Equal contribution (zyliang@smu.edu.sg; yan.shu@unitn.it; lzliao@smu.edu.sg)

# arXiv:2512.23044v2[cs.CV]16Jan2026

### Abstract

The evolution of autonomous agents is redefining information seeking, transitioning from passive retrieval to proactive, open-ended web research. However, a significant modality gap remains in processing the web’s most dynamic and information-dense modality: video. In this paper, we first formalize the task of Agentic Video Browsing and introduce VideoBrowseComp, a benchmark evaluating openended agentic browsing tasks that enforce a mandatory dependency on videos. We observe that current paradigms struggle to reconcile the scale of open-ended video exploration with the need for fine-grained visual verification. Direct visual inference (e.g., RAG) maximizes perception but incurs prohibitive context costs, while text-centric summarization optimizes efficiency but often misses critical visual details required for accurate grounding. To address this, we propose Video-Browser, a novel agent leveraging Pyramidal Perception, filtering with cheap metadata and zooming in with expensive visual perception only when necessary. Experiments demonstrate that our approach achieves a 37.5% relative improvement while reducing token consumption by 58.3% compared to Direct visual inference, establishing a foundation for verifiable open-web video research. We open-source all codes, benchmark at https://github.com/chrisx599/VideoBrowser.

### 1 Introduction

The rapid evolution of Large Language Models (LLMs) has catalyzed a paradigm shift from static question-answering to autonomous agents capable of actively navigating the web to solve complex problems (Nakano et al., 2021b; He et al., 2024). This transition towards agentic web browsing has become a dominant trend in AI research. Pioneering benchmarks such as GAIA (Mialon et al., 2023), BrowseComp (Wei et al., 2025), and MMBrowseComp (Li et al., 2025a) have established

standards for agents operating within textual and static multimodal environments. These works reflect an emerging consensus: real-world agents must proactively seek, retrieve, and reason over information rather than passively receiving it.

Despite these advancements, the current landscape of deep research agents faces a fundamental limitation: a modality blind spot. While the community has focused heavily on static web information like text and images (Jin et al., 2025; Geng et al., 2025), it has largely overlooked the most dynamic and information-dense modality on the web: video, which houses everything from product reviews that demonstrate dynamic usage to complex tutorial procedures. Current AI research, video benchmarks like (Fu et al., 2024; Zhou et al., 2025a; Li et al., 2024) and video agents like (Fan et al., 2024; Wang et al., 2024, 2025b), treat video primarily as a static file to be perceived rather than a dynamic source to be explored. They typically input a curated clip to a model and query its content without requiring external information retrieval. This setting fails to reflect the agentic nature of realworld research. Users do not simply ask video a question, they search for videos through problemdriven then summarize the answers.

We argue that the field should move towards Agentic Video Browsing. To address these problems, we present the following three contributions.

First, we formally define this task not as video QA, but as an iterative decision-making process. To operationalize this task, we introduce VideoBrowseComp, a challenging benchmark designed to evaluate open-ended agentic browsing tasks that enforce a mandatory dependency on videos. To systematically assess agentic limits, we stratify the benchmark into three difficulty levels (Level 1 to Level 3), scaling from explicit retrieval to complex, cross-source reasoning that demands aggregating fragmentary evidence across disparate videos.

Second, to overcome the limitations of current

paradigms, we propose Video-Browser, a novel agent leveraging Pyramidal Perception. Existing approaches face a dilemma: direct video inference (e.g, RAG) incurs prohibitive context costs, while text-centric summarization suffers from a modality gap that misses visual details. Our architecture resolves this by treating the web videos as a pyramid. Utilizing lightweight video metadatas to prune irrelevant search results and zooming in with high-fidelity perception only when necessary.

Third, comprehensive experiments validate the effectiveness of our approach. Results on VideoBrowseComp demonstrate that Video-Browser achieves a 37.5% relative improvement in accuracy while reducing token consumption by 58.3% compared to direct video inference baselines. This confirms that our method successfully balances the scalability required for open-web video research with precision needed for fine-grained verification.

### 2 Related Works

#### 2.1 Video Understanding

The field of video understanding has witnessed rapid advancements with the emergence of Multimodal Large Language Models (MLLMs) (Comanici et al., 2025; OpenAI et al., 2024; Liu et al., 2023; Zhu et al., 2025). Video LLMs (Shu et al., 2025; Qin et al., 2025; Li et al., 2025d; Liu et al., 2025; Chen et al., 2024; Shen et al., 2024) and agents (Wang et al., 2024; Fan et al., 2024; Wang et al., 2025b) have demonstrated impressive capabilities in processing long-context visual inputs. Concurrently, benchmarks (Fu et al., 2024; Zhou et al., 2025a; Li et al., 2024; Wang et al., 2025a) have been established to rigorously evaluate these models on traditional video QA tasks.

However, these existing works predominantly operate under a paradigm of passive perception. In typical evaluation settings, a model is fed a curated video clip and queried about its internal content in a closed-world manner. This approach remains at the perception level, falling short of researchoriented investigation. In real-world scenarios, answering complex queries often requires more than just “watching” a single video (Peng et al., 2025; He et al., 2025; Jang et al., 2025; Fu et al., 2025). It necessitates combining internal video evidence with external knowledge or actively searching across multiple videos to triangulate facts. Current benchmarks’ inability to measure external verification and cross-source reasoning underscores

the need for agents with active search capabilities.

#### 2.2 Web Browsing Agents

To address the limitations of passive models, the community has shifted focus towards autonomous browsing agents. This evolution began with textbased agents, where pioneers (Nakano et al., 2021a; Jin et al., 2025; Li et al., 2025c; Wu et al., 2025a; Li et al., 2025b) demonstrated that LLMs could effectively utilize search engine tools to solve complex text-based questions (Wei et al., 2025; Zhou et al., 2025b; Mialon et al., 2023; Phan et al., 2025; Chen et al., 2025). Recently, this paradigm has expanded into the multimodal domain. Models and benchmarks (Li et al., 2025a; Wu et al., 2025b; Jiang et al., 2024) have pushed the boundary by requiring agents to process static web information, including text and images, marking a significant step towards multimodal information seeking.

Despite these advancements, current agents face a dilemma in balancing context costs with perceptual granularity when confronting dynamic video content. First, regarding visual granularity, direct visual inference strategies (e.g., standard RAG) theoretically offer the highest recall by streaming raw video directly to the model. However, this approach incurs prohibitive context costs, causing a “context explosion” that renders it unscalable for open-ended video browsing tasks where agents must verify evidence across multiple long videos (Yuan et al., 2025; Zhang et al., 2025; Ren et al., 2025). Second, to mitigate this computational burden, existing frameworks typically resort to textcentric summarization or rely on sparse metadata (Jiang et al., 2024; Li et al., 2025e). While this significantly reduces the context load, it introduces a critical modality gap. By compressing dynamic video streams into static textual proxies, these methods lose the fine-grained visual details essential for deep research, leaving agents unable to verify complex visual queries that lack explicit textual descriptions.

3 Benchmark

#### 3.1 Task Definition

We formulate the video browsing task as an openended information seeking problem constrained by multimodal evidence verification. Given a natural language query Q, an agent is provided access to the open web W. The agent must autonomously plan a sequence of actions S = {a1,a2,...,at},

such as searching the web, retrieving specific video timestamps, or reading subtitles, to generate a final result.

#### 3.2 Principles and Scope

To ensure Video-BrowseComp serves as a rigorous benchmark for agentic video research, we adhere to three foundational design principles:

1) Mandatory Video Dependency. A core objective of this benchmark is to evaluate video perception and reasoning, not merely text-based web search. To prevent models from solving questions via textual shortcuts (e.g., retrieving a movie plot from Wikipedia without watching the video). 2) Hard-to-Find, Easy-to-Verify. The questions are intentionally complex, often requiring multi-step reasoning, cross-referencing, or long-context scanning. However, the output format is constrained to short, objective strings, such as specific entity names, counts, or colors, allowing for automated and unambiguous verification against the ground truth. 3) Answer Uniqueness. To guarantee evaluation reliability, we ensure that every question has a single answer. Questions are grounded in objective spatio-temporal attributes rather than subjective interpretations.

#### 3.3 Difficulty Stratification

To systematically assess the limits of agentic capabilities, we stratify Video-BrowseComp into three distinct difficulty levels based on the complexity of the search space and the depth of reasoning required. We provide detailed benchmark cases in Appendix A.1.

Level 1: Explicit Retrieval. Questions in this level provide explicit constraints regarding the search domain, such as specific publication dates, platform names, or title keywords. The challenge lies in instruction following to locate the target video and temporal grounding to extract visual details. Level 2: Implicit Retrieval. Removing explicit search keywords, the target video is referenced only through indirect descriptions. Agents must first bridge the gap between visual cues and textual search queries, and then filter through multiple candidate videos to find the one matching the event description. Level 3: Multi-Source Reasoning. Representing the highest tier of difficulty, these questions cannot be answered by viewing a single video. They require cross-source reasoning where information retrieved from one source serves as a prerequisite for locating the next.

#### 3.4 Dataset Construction

Annotation Pipeline. We adopt a reverse construction strategy that begins with video exploration and culminates in complex query formulation. First, each annotator is assigned 2 video categories and encouraged to explore relevant topics through web browsing. After identifying candidate videos, annotators watch the complete video and design an initial question-answer pair focused on a specific clip. These simple questions are then expanded into complex queries following two principles: Indirect & Compositional Queries: Questions are augmented with compositional conditions to ensure the target video cannot be easily retrieved through a simple search. Annotators use indirect descriptions to reference video content rather than explicit keywords. Multi-Source Reasoning: Level 1 and Level 2 samples can be derived from the above steps with single-video evidence. For Level 3 samples, annotators first identify a set of related videos covering the same topic or event, then design questions that require aggregating complementary information across these videos, such as comparing statistics from different matches, verifying conflicting claims across multiple sources, or tracing an event’s progression through several recordings. Finally, annotators document the complete evidence chain or authoritative source video URLs, ensuring full traceability of the ground-truth answers.

Quality Control. We implement a two-stage quality control protocol to ensure the quality of Video-BrowseComp. 1.Easy Question Filtering. Although annotators are required to design questions based on the visual content of videos, some samples may still be answerable through commonsense knowledge without watching the video. To identify and remove such cases, we use the Google text search api1 to return 5 relevant results, input it to GPT-5 to filter out questions that can be answered correctly. 2.Answer Uniqueness Validation. This step ensures that each question has only one correct and unambiguous answer. We employ a cross-validation strategy among the annotators: each annotator answers questions annotated by others following the evidence chain, and we compare their responses against the original ground-truth answers. If an alternative answer satisfies all task constraints but differs from the original, the sample is considered ambiguous and discarded.

1https://serper.dev

125

Documentary 5%

Film 29%

120

100

Music & Variety 5%

Sports 21%

80

Category

62

60

Education & Knowledge 12% Games & Esports 10%

Video & Shorts 9%

40

23

20

TV Series 9%

0

Level 1 Level 2 Level 3

- Figure 1: Statistical overview of our Benchmark, including the category (Left) and difficulty distribution (Right).

#### 3.5 Dataset Statistics

The final benchmark consists of 210 validated questions spanning 8 diverse genres. As illustrated in Figure 1, the distribution prioritizes complex visual narratives, with Film (29%) and Sports (21%) forming the core, ensuring agents are evaluated on dynamic events rather than just static metadata. The difficulty is stratified into a pyramidal structure: while Level 1 (125 samples) tests explicit retrieval, Level 2 (62) and Level 3 (23) specifically target advanced capabilities in implicit retrieval and cross-source reasoning. We provide some analysis of the benchmark in Appendix A.5.

4 Method

#### 4.1 Framework Formulation

We formalize Video-Browser as an agentic system S = ⟨P,W,A,M⟩, comprising three specialized modules: Planner P, Watcher W, and Analyst A, sharing a global memory M. Given a user query Q, the system operates iteratively over time steps t to navigate the open video corpus Vweb and generate an answer A. The overview of the framework is shown in Figure 2.

Memory M. Memory stores the interaction history, denoted as Ht = {Q,(q1,o1),...,(qt−1,ot−1)}, where qi represents search queries and oi represents the structured observations returned by the Watcher.

Planner P. Acting as the cognitive controller, the Planner orchestrates the process as follows: Upon receiving the query Q, the Planner first decomposes it into sequential sub-tasks:

Splan = {s1,s2,...,sm} = P(Q) (1)

Then at each time step t, the Planner focuses on the current sub-task st and generates specific search keywords qt tailored to this sub-task based on the

interaction history Ht:

qt = P(st,Ht) (2)

These search queries qt are then used to retrieve candidate videos Vret from the open video corpus. Once all sub-tasks in Splan are resolved or the maximum steps Tmax are reached, the Planner triggers the Analyst.

Watcher W. Watcher addresses the PerceptionContext Trade-off by functioning as a highefficiency filter and localizer. It takes the raw candidate videos Vret retrieved by the search engine and the current query context qt as input to produce a focused evidence set Et. It employs a Pyramidal Perception mechanism Ψ:

Et,ot = W(Vret,qt;Ψ) (3)

Here, Et = {(vi,[tstart,tend])} denotes the visual evidence set consisting of prioritized temporal windows. Simultaneously, the Watcher generates a structured observation ot (containing metadata and relevance summaries of Et) to provide feedback to the Planner, updating Ht+1 = Ht ∪{(qt+1,ot+1)}. Analyst A. Upon termination at step T, the Analyst performs fine-grained reasoning. It takes the accumulated evidence Etotal = Tt=1 Et to synthesize the final answer:

A = A(Q,Etotal) (4)

Iterative Reasoning Loop. The system operates through continuous feedback between modules. At each iteration t, the Planner evaluates the Watcher’s observations ot to assess evidence quality. If the collected evidence is insufficient or ambiguous for the current sub-task st, the Planner adaptively refines its search strategy—either generating new queries to explore alternative perspectives or adjusting keywords for improved precision. This reasoning loop continues until all sub-tasks in Splan are adequately resolved or the maximum iteration limit Tmax is reached, whereupon the Planner triggers the Analyst for final answer synthesis.

#### 4.2 Pyramidal Perception

Open-ended research requires retrieving and synthesizing information from multiple video sources. To address the inherent scale of open-ended video browsing, the Watcher employs Pyramidal Perception.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

- Figure 2: The overview of Video-Browser framework (Left), which consists of Planner, Watcher and Analyst. The Watcher employs Pyramidal Perception mechanism (Right), effectively mitigating the perception-context trade-off.

#### Stage I: Semantic Filter. At the base of the pyra-

mid, the agent confronts a noisy candidate set Vret retrieved from the open web. Since performing visual inference on this raw distribution is computationally prohibitive, the Watcher functions as a metadata-driven filter. By prompting an LLM to analyze solely textual metadata (e.g., titles, snippets), it assesses the relevance of each candidate to the sub-task at zero visual inference cost. This step aggressively prunes the search space before data loading occurs, ensuring that expensive perceptual resources are reserved strictly for promising candidates.

##### Stage II: Sparse Localization. For the retained videos, the challenge shifts from selection to localization, finding where the relevant information resides without incurring the cost of full decoding. The Watcher employs a sparse scanning strategy that balances context with efficiency. It ingests the full textual transcript to grasp the narrative structure, while simultaneously sampling a fixed set of sparse frames to capture visual states. An MLLM leverages this combined audio-visual context to identify temporal proposals, specific time

windows [tstart,tend] that potentially contain the visual evidence. This process compresses hourlong videos into focused “glimpses”, providing the Planner with sufficient feedback to make informed routing decisions.

##### Stage III: Zoom-in. At the tip of the pyramid, the system performs Targeted Visual Decoding on the prioritized temporal windows. This stage represents the transition from scanning to scrutinizing. The Analyst re-decodes the video strictly within the identified windows at high frequency to con-

struct a dense visual context. By applying a MLLM to this high-fidelity stream, the agent can resolve fine-grained visual details that were invisible during the sparse stage. This ensures that the most expensive reasoning power is allocated exclusively to the most information-dense moments, effectively resolving the perception-context trade-off.

### 5 Experiments

In this section, we conduct comprehensive experimental studies to evaluate the effectiveness of Video-Browser on the Video-BrowseComp benchmark. We structure our analysis around four core research questions: RQ1: (Performance) How does the Video-Browser compare to the SOTA models? RQ2: (Efficiency) Does the Pyramidal Perception architecture effectively minimize token consumption while maintaining good visual perception? RQ3: (Test-time scaling) Can VideoBrowser effectively leverage increased test-time compute to yield continuous performance gains? RQ4: (Ablation) What is the contribution of each architectural module to the overall system effectiveness? See each experiment setting in Appendix B.1.

#### 5.1 Metrics

Accuracy (OA). We employ Overall Accuracy as the primary metric. Following the established evaluation protocol of BrowseComp (Wei et al., 2025), we utilize LLM as a judge2 (Zheng et al., 2023) to assess the semantic equivalence between the model’s prediction and the ground truth, rather than relying on rigid string matching.

2Evaluation prompts are in Appendix A.4.

OA Difficulty CE Genre Accuracy (%) (%) L1 L2 L3 (%) Doc Edu Film Game Music Sport TV Vid

Model

Tool-Free Models

Qwen3-VL-8B-Think 7.14 12.00 0.00 0.00 52.49 10.00 0.00 13.11 0.00 20.00 0.00 5.26 16.67 Qwen3-VL-235B-Ins 13.33 22.40 0.00 0.00 77.64 0.00 12.00 19.67 13.64 0.00 2.22 21.05 27.78 GLM-4.6V 10.95 16.80 3.23 0.00 44.40 0.00 12.00 11.48 13.64 10.00 2.22 26.32 16.67 gpt-4o-2024 17.62 28.00 3.23 0.00 58.81 10.00 16.00 26.23 13.64 0.00 4.44 31.58 27.78

- gpt-4o-mini 9.52 16.00 0.00 0.00 63.55 0.00 12.00 18.03 9.09 0.00 0.00 15.79 5.56

- gpt-5-mini 15.71 26.40 0.00 0.00 37.47 0.00 28.00 22.95 13.64 10.00 0.00 21.05 22.22 gemini-2.5-flash 16.67 27.20 1.61 0.00 77.79 20.00 20.00 24.59 4.55 0.00 2.22 36.84 22.22 gemini-2.5-pro 19.52 31.20 3.23 0.00 79.18 10.00 28.00 29.51 13.64 0.00 2.22 36.84 22.22

Search Models

gemini-2.5-flash (S) 20.95 32.80 4.84 0.00 35.98 0.00 36.00 29.51 4.55 10.00 11.11 36.84 16.67 gemini-2.5-pro (S) 23.81 37.60 4.84 0.00 31.45 20.00 32.00 31.15 9.09 10.00 8.89 57.89 16.67 gpt-5.1 (S) 15.24 21.60 6.45 4.35 30.20 0.00 8.00 21.31 18.18 10.00 8.89 21.05 22.22 o4-mini-deep-research 22.86 30.40 12.90 8.70 42.55 10.00 28.00 29.51 27.27 10.00 17.78 21.05 16.67

Video Browser

VideoBrowser (Qwen3) 19.05 25.60 9.68 8.70 69.81 0.00 36.00 22.95 4.55 10.00 11.11 15.79 38.89 VideoBrowser (GPT-5.1) 26.19 37.60 11.29 4.35 44.60 10.00 40.00 31.15 4.55 20.00 20.00 26.32 44.44

- Table 1: Comprehensive results with a heatmap on Genre Accuracy. Bold indicates best performance. (S) means model has web search ability.

Calibration Error (CE). To quantify the reliability of the agent’s uncertainty estimation, we report the Calibration Error (CE). This metric measures the alignment between the model’s self-assigned confidence scores and its actual empirical accuracy, where a lower CE indicates a less overconfident and more trustworthy agent. Detailed formulations for CE are provided in Appendix A.3.

#### 5.2 Baselines

To systematically evaluate the agentic capabilities for video-based research, we categorize our baselines into two groups: Tool-Free Models and Search-Augmented Models.

Tool-Free Models. This category evaluates the capability of state-of-the-art Multimodal Large Language Models (MLLMs), including Qwen3-VL-8B-Thinking, Qwen3-VL-235BA22B-Instruct (Yang et al., 2025), GLM-4.6V (Team et al., 2025), gpt-4o-2024-11-20, gpt-4omini-2025-0807 (OpenAI et al., 2024), gemini-2.5flash-2025-06, gemini-2.5-pro-2025-06 (Comanici et al., 2025), to answer questions relying solely on their internal parametric knowledge and the provided context.

Search-Augmented Models. This category evaluates the official, search-augmented model services provided by major vendors, including gemini-2.5flash-2025-06 (w/ Search), gemini-2.5-pro-202506 (w/ Search), gpt-5.1-2025-11-13 (w/ Search) (OpenAI, 2025b), o4-mini-deep-research-2025-0626 (OpenAI, 2025a). These systems possess integrated web search capabilities that enable them to

query the internet for up-to-date information autonomously during inference.

#### 5.3 Performance

Accuracy. As shown in Table 1, Tool-Free models struggle significantly, plateauing at 19.52% (Gemini-2.5-pro) due to reliance on parametric memory. Integrating web search yields immediate gains in explicit retrieval tasks (Level 1), but performance collapses in Level 2 and Level 3 tasks, exposing that search agents can locate metadata but cannot verify visual details. Even the advanced o4-mini-res, is capped at 22.86% overall due to this visual blindness. Qualitative analysis of SOTA models is in Appendix A.6.

By equipping the agent with Pyramidal Perception, Video-Browser (GPT-5.1) achieves a stateof-the-art accuracy of 26.19%. This represents a decisive 71.8% relative improvement over the standard Search-Augmented GPT-5.1 (15.24%). Notably, our approach outperforms the Deep Research baseline (o4-mini) without requiring expensive reinforcement learning, demonstrating that complex planning cannot fully compensate for the lack of direct perception, and that visual grounding serves as a prerequisite for effective reasoning in video tasks.

Calibration Error. Tool-Free models exhibit severe overconfidence, with generally high Calibration Errors stemming from parametric hallucination. Search augmentation acts as a critical grounding mechanism, yielding a substantial reduction in CE across all baselines. Our Video-Browser main-

[Figure 7]

Figure 3: Test-time Scaling comparison including Search Breadth, Reasoning Depth and Perceptual Granularity.

tains a comparable calibration profile to searchaugmented models, effectively balancing confidence with capability even while attempting significantly more challenging visual queries.

#### 5.4 Efficiency

In Table 2, we report both Tokens (the cumulative cost of scanning, filtering, and reasoning) and Context (the final prompt size fed to the Analyst).

Model OA CE Tokens Context Direct Visual Inference

- w/ Qwen3 13.81 70.83 55,047 53,764 w/ GPT-5.1 19.05 50.80 78,229 76,955 Summarization w/ Qwen3 9.52 82.10 55,971 1,260 w/ GPT-5.1 18.57 51.34 75,728 919 Pyramidal Perception

- w/ Qwen3 14.29 65.74 26,525 7,678 w/ GPT-5.1 26.19 38.10 32,627 11,198

- Table 2: Comparison of Three Paradigms on Video BrowseComp. Bold denotes the best result. Direct Visual Inference. Input raw video frames directly to the model yields a baseline accuracy of 19.05% (GPT-5.1). However, this approach is prohibitively expensive, consuming 78,229 total tokens and filling the context window with 76,955 tokens. This confirms the context explosion bottleneck: the agent is forced to watch every frame, leaving little room for multi-step reasoning and limiting scalability. Summarization. The Summarization paradigm effectively compresses the video input to text evidence, reducing the context load to a negligible 919 tokens. However, this reveals two critical flaws: 1). High Pre-processing Cost: The total tokens remain high (75,728) because the model must still process the full video to generate the summary. 2). Lossy Compression: Accuracy drops to 18.57%, and Calibration Error spikes (51.34%). Validating

the modality gap: textual proxies fail to capture the fine-grained visual details.

Pyramidal Perception. Our Pyramidal Perception reduces the Total Tokens to 32,627, a 58.3% (GPT5.1) reduction compared to Direct Visual Inference. Crucially, unlike Summarization, this efficiency does not compromise perception. We achieves the highest accuracy of 26.19%. This demonstrates that our architecture successfully identifies where to spend compute, minimizing consumption while maximizing visual perception.

#### 5.5 Test-time Scaling

Figure 3 visualizes the performance of VideoBrowser under varying computational budgets. We isolate three critical scaling dimensions: search breadth, reasoning depth, and perceptual granularity.

Search Breadth (Video Number). As the number of processed candidate videos increases from 1 to 5, accuracy improves from 10.0% to 14.8%. This suggests that for open-ended queries, by aggregating evidence from a broader set of videos, the agent can mitigate single-source hallucinations and improves recall.

Reasoning Depth (Search Loops). Increasing the maximum search loops allows the agent to iteratively refine its queries and explore new paths. We observe a strong positive correlation, with accuracy peaking at 19.0% with 5 loops. This confirms that the iterative feedback loop is a driver of complex reasoning, rather than a redundant overhead.

Perceptual Granularity (Watcher Frames). Increasing the number of scanning frames (from 4 to 64) leads to a consistent performance gain (10.5% → 15.7%). This indicates that while the Watcher operates on sparse signals to save tokens, increasing its sampling density significantly enhances its ability to localize correct temporal windows.

Question: In a film starring and directed by Ben Stiller, the ordinary photo technician named Walter Mitty (Walter Mitty) has a pen that runs throughout the entire film in his shirt pocket; the pen's cap appearing hints that Walter's inner longing is for a passionate life. What color is this pen? Ground Truth: red

Direct Visual Inference: Across the frames from the analyzed clips, Walter Mitty’s shirt pocket and chest area are visible multiple times, but no pen or pen cap can be seen in any of them. The transcripts likewise never mention the pen or its color. Therefore, the videos provided do not contain the information needed to determine the color of Walter Mitty’s pen. Answer: The color of the pen cannot be determined from the provided videos.

Summarization: None of the provided video summaries mention or show Walter Mitty’s shirt pocket or the pen, nor do they discuss its color or symbolism. Therefore, the color of the pen cannot be determined from these sources. Answer: The color of the pen is not specified in the provided videos.

###### Pyramidal Perception:

In the early close-up frames of Walter Mitty, his shirt pocket area is visible and the only pen present shows a distinctly red cap protruding from the pocket. This recurring red-capped pen is used symbolically in the film to hint at his inner desire for a more passionate life.

###### Answer: Red

Figure 4: Qualitative comparison of three paradigms.

Setting OA CE Token Context Component

Modality. The w/ Transcript setting (only input the video transcript) yields the lowest accuracy of 7.14%. This 44% relative performance drop compared to the full model (12.86%) serves as definitive empirical evidence of the modality gap. Explicit visual perception is mandatory for highfidelity grounding.

- w/o stage 1 9.52 66.64 16,846 6,068
- w/o stage 2 12.38 69.50 44,961 42,401
- w/o stage 3 10.48 67.52 18,330 4,198

Modality

w/ Transcript 7.14 52.49 15,606 5,155 w/ Visual 12.38 37.31 8,353 3,901

#### 5.7 Qualitative comparison

To intuitively understand the advantages of our approach, we present a qualitative case study in Figure 4. More case studies are in the Appendix B.3.

Full 12.86 68.23 19,505 5,649

Table 3: Detailed analysis of Ablation and Modality.

#### 5.6 Ablation Study

#### 5.8 Reproducibility

- Table 3 dissects the contribution of each architectural component and modality. Component. Stage 1 serves as the foundation for relevance. Removing it yields the lowest performance among all component variants (9.52%), significantly below the full model (12.86%). This drop indicates that without effective initial filtering, the agent wastes its perceptual budget on irrelevant content. Removing Stage 2 causes a catastrophic spike in computational cost: total token usage more than doubles (19,505 → 44,961) and the final context load explodes by ∼7.5× (5,649 → 42,401). This confirms that Stage 2 is essential for preventing context explosion. Removing Stage 3 degrades accuracy to 10.48%, highlighting that sparse sampling alone lacks the spatial-temporal resolution to verify fine-grained details. In summary, the architecture follows a logical progression: Stage 1 ensures relevance, Stage 2 ensures efficiency, and Stage 3 ensures precision.

We provide comprehensive implementation details in Appendix B.1 & B.2. Including the prompts for each module, and configurations etc.

### 6 Conclusion

We introduced Video-BrowseComp, a benchmark enforcing mandatory video dependency to rigorously evaluate agentic browsing. To overcome the limitations of existing paradigms, specifically the modality gap in textual summarization and the prohibitive costs of direct retrieval. We proposed Video-Browser. Leveraging Pyramidal Perception, our agent hierarchically filters content to resolve the perception-context dilemma. Experiments demonstrate that Video-Browser achieves a 37.5% accuracy improvement while reducing token consumption by 58.3% compared to direct visual inference, establishing a solid foundation for verifiable and efficient video agents.

### Limitations

Benchmark Scale and Computational Accessibility. We acknowledge that Video-BrowseComp operates at a modest scale (210 samples) compared to traditional, single-turn QA datasets. This scale is a deliberate design choice driven by the trade-off between annotation rigor and computational accessibility. First, unlike scalable synthetic generation, Video-Browsecomp necessitates expensive expert validation to guarantee answer uniqueness in an open-ended search space. Second, and more critically, agentic video browsing is computationally intensive. Unlike standard VQA, a single query involves multi-step reasoning loops, web search retrieval, and high-frequency video decoding, often incurring substantial token costs. A massive-scale benchmark would impose a prohibitive computational barrier for the broader research community. By prioritizing sample difficulty and quality over magnitude, we position Video-BrowseComp as a high-fidelity “Golden Test Set” that enables rigorous yet affordable comparisons of expensive agentic pipelines.

### Ethical Considerations

This work studies agents that browse and reason over open-web videos. To respect copyright and platform policies, we do not redistribute or host video content; any benchmark release should provide only question–answer pairs and pointers (e.g., URLs/timestamps) needed to locate publicly available sources, and we will honor takedown requests and removal of problematic items when notified. Because videos may contain personal data (faces, names, incidental background information), we avoid collecting non-public sources and do not design tasks that require identifying private individuals; the benchmark is intended for factual, publicly verifiable information seeking rather than surveillance or profiling.

### References

Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. 2024. Longvila: Scaling long-context visual language models for long videos. Preprint, arXiv:2408.10188.

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama

Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. 2025. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. Preprint, arXiv:2508.06600.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, and etc. Ori Ram. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arXiv:2507.06261.

Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. 2024. Videoagent: A memory-augmented multimodal agent for video understanding. Preprint, arXiv:2403.11481.

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, and 1 others. 2024. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075.

Mingyang Fu, Yuyang Peng, Dongping Chen, Zetong Zhou, Benlin Liu, Yao Wan, Zhou Zhao, Philip S. Yu, and Ranjay Krishna. 2025. Seeking and updating with live visual knowledge. Preprint, arXiv:2504.05288.

Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. Webwatcher: Breaking new frontier of vision-language deep research agent. Preprint, arXiv:2508.05748.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. Webvoyager: Building an end-toend web agent with large multimodal models. arXiv preprint arXiv:2401.13919.

Zhihao He, Tianyao He, Yun Xu, Tieyuan Chen, Huabin Liu, Chaofan Gan, Zuxuan Wu, and Weiyao Lin. 2025. Enhancing video large language models with structured multi-video collaborative reasoning. Preprint, arXiv:2509.13161.

Lawrence Keunho Jang, Yinheng Li, Dan Zhao, Charles Ding, Justin Lin, Paul Pu Liang, Rogerio Bonatti, and Kazuhito Koishida. 2025. Videowebarena: Evaluating long context multimodal agents with video understanding web tasks. In The Thirteenth International Conference on Learning Representations.

Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Chaoyou Fu, Guanglu Song, Peng Gao, Yu Liu, Chunyuan Li, and Hongsheng Li. 2024. Mmsearch: Benchmarking the potential of large models as multimodal search engines. Preprint, arXiv:2409.12959.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. Preprint, arXiv:2503.09516.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. 2024. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22195–22206.

Shilong Li, Xingyuan Bu, Wenjie Wang, Jiaheng Liu, Jun Dong, Haoyang He, Hao Lu, Haozhe Zhang, Chenchen Jing, Zhen Li, Chuanhao Li, Jiayi Tian, Chenchen Zhang, Tianhao Peng, Yancheng He, Jihao Gu, Yuanxing Zhang, Jian Yang, Ge Zhang, and 5 others. 2025a. Mm-browsecomp: A comprehensive benchmark for multimodal browsing agents. Preprint, arXiv:2508.13186.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025b. Search-o1: Agentic search-enhanced large reasoning models.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and Zhicheng Dou. 2025c. Webthinker: Empowering large reasoning models with deep research capability. arXiv preprint arXiv:2504.21776.

Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. 2025d. Videochat-flash: Hierarchical compression for long-context video modeling. Preprint, arXiv:2501.00574.

Zijian Li, Xin Guan, Bo Zhang, Shen Huang, Houquan Zhou, Shaopeng Lai, Ming Yan, Yong Jiang, Pengjun Xie, Fei Huang, Jun Zhang, and Jingren Zhou. 2025e. Webweaver: Structuring web-scale evidence with dynamic outlines for open-ended deep research. Preprint, arXiv:2509.13312.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Preprint, arXiv:2304.08485.

Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, Xiuyu Li, Haotian Tang, Yunhao Fang, Yukang Chen, Cheng-Yu Hsieh, DeAn Huang, An-Chieh Cheng, Jinyi Hu, Sifei Liu, and 6 others. 2025. Nvila: Efficient frontier visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4122–4134.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia: a benchmark for general ai assistants. Preprint, arXiv:2311.12983.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. 2021a. Webgpt: Browserassisted question-answering with human feedback. CoRR, abs/2112.09332.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, and 1 others. 2021b. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.

OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Madry, Alex Baker-Whitcomb, and etc. Alex Beutel. 2024. Gpt-4o system card. Preprint, arXiv:2410.21276.

- OpenAI. 2025a. Deep Research System Card.
- OpenAI. 2025b. GPT-5.1 Instant and GPT-5.1 Thinking System Card Addendum.

Tianhao Peng, Haochen Wang, Yuanxing Zhang, Zekun Wang, Zili Wang, Gavin Chang, Jian Yang, Shihao Li, Yanghai Wang, Xintao Wang, Houyi Li, Wei Ji, Pengfei Wan, Steven Huang, Zhaoxiang Zhang, and Jiaheng Liu. 2025. Mvu-eval: Towards multivideo understanding evaluation for multimodal llms. Preprint, arXiv:2511.07250.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, and etc Richard Ren. 2025. Humanity’s last exam.

Minghao Qin, Xiangrui Liu, Zhengyang Liang, Yan Shu, Huaying Yuan, Juenjie Zhou, Shitao Xiao, Bo Zhao, and Zheng Liu. 2025. Video-xl-2: Towards very longvideo understanding through task-aware kv sparsification. Preprint, arXiv:2506.19225.

Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. 2025. Videorag: Retrieval-augmented generation with extreme longcontext videos. Preprint, arXiv:2502.01549.

Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, and 1 others. 2024. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434.

Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. 2025. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26160–26169.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, and 69 others. 2025. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. Preprint, arXiv:2507.01006.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. 2025a. Lvbench: An extreme long video understanding benchmark. Preprint, arXiv:2406.08035.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. 2024. Videoagent: Long-form video understanding with large language model as agent. Preprint, arXiv:2403.10517.

Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. 2025b. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. Preprint, arXiv:2405.19209.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. Browsecomp: A simple yet challenging benchmark for browsing agents. Preprint, arXiv:2504.12516.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, and 1 others. 2025a. Webdancer: Towards autonomous information seeking agency. arXiv preprint arXiv:2505.22648.

Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, and Ziwei Liu. 2025b. Mmsearchr1: Incentivizing lmms to search.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint,

- arXiv:2505.09388.

Huaying Yuan, Zheng Liu, Junjie Zhou, Hongjin Qian, Yan Shu, Nicu Sebe, Ji-Rong Wen, and Zhicheng Dou. 2025. Videoexplorer: Think with videos for agentic long-video understanding. Preprint,

- arXiv:2506.10821.

Xiaoyi Zhang, Zhaoyang Jia, Zongyu Guo, Jiahao Li, Bin Li, Houqiang Li, and Yan Lu. 2025. Deep video discovery: Agentic search with tool use for long-form video understanding. Preprint, arXiv:2505.18079.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,

Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. 2025a. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13691–13701.

Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, Yuxin Gu, Sixin Hong, Jing Ren, Jian Chen, Chao Liu, and Y. Hua. 2025b. Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, and etc. Jie Shao. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. Preprint, arXiv:2504.10479.

### Appendix

|Contents<br><br>• §A Video-Browsecomp (Benchmark Details)<br><br>◦ §A.1 Benchmark Samples<br>◦ §A.2 Annotators<br>◦ §A.3 Evaluation Metrics Details<br>◦ §A.4 Evaluation Prompts<br>◦ §A.5 Benchmark Analysis<br>◦ §A.6 Benchmark Case Studies<br><br><br>• §B Video-Browser (Agent Details)<br><br><br>◦ §B.1 Experiment Settings<br>◦ §B.2 Prompts<br>◦ §B.3 Case Studies<br>|
|---|

### A Video-Browsecomp

- A.1 Benchmark Samples

We provide some benchmark examples in Figure 5 for reference.

- A.2 Annotators

To curate Video-BrowseComp, we recruited 8 student volunteers (holding Master’s or PhD degrees) with extensive experience in LLM usage, web search, and video understanding. The construction process follows a rigorous annotation pipeline with two-stage quality control in Sec 3.4.

Informed consent was obtained from all participants prior to the study. They were informed that the collected data would be used for research purposes and released publicly.

- A.3 Evaluation Metrics Details

- A.3.1 LLM-based Accuracy Judge

Since the reference answers in Video-BrowseComp are designed to be short and verifiable (e.g., specific entities, colors, or counts), rigid string matching often fails due to minor formatting differences. We utilize gpt-5-mini-2025-08-07 as the automated judge. The judge is prompted to verify if the prediction conveys the same factual information as the ground truth. The specific prompt used for evaluation is shown in Figure 6.

#### A.3.2 Calibration Error (CE)

To calculate CE, we first prompt the model to provide a confidence score p ∈ [0,1] alongside its final answer. We partition the predictions into B = 5 equally spaced bins (i.e., [0,0.2),...,[0.8,1.0]). The CE is calculated as the weighted average of the absolute difference between the accuracy and the average confidence within each bin:

CE =

B

ni N |acc(i) − conf(i)| (5)

i=1

Where N is the total number of samples, ni is the number of samples in the i-th bin, acc(i) is the empirical accuracy of the samples in bin i, and conf(i) is the average predicted confidence of the samples in that bin.

#### A.4 Evaluation Prompts

We provide the prompts used in evaluation and judge process at Figure 6.

#### A.5 Benchmark Analysis A.5.1 Category-Wise Performance

[Figure 8]

Figure 8: Performance heatmap.

Figure 8 provides a granular breakdown of model accuracy across the eight video genres. A clear performance dichotomy is observable based on the availability of external textual metadata. In categories like TV Series and Education, where transcripts, wikis, and plot summaries are abundant on the open web, search-augmented models achieve their highest scores. For example, Gemini2.5-Pro (w/ Search) reaches 57.9% accuracy on TV Series. Conversely, performance collapses in dynamic, stochastic environments like Games and Sports. These categories require temporal grounding in specific visual moments (e.g., a specific foul or gameplay sequence) that are rarely indexed

by text search engines. Consequently, the same model’s accuracy drops to 9.1% in Games and 8.9% in Sports , highlighting the "Modality Gap" where agents fail to process visual information when text shortcuts are unavailable.

#### A.5.2 Cost-Efficiency Analysis (Token Usage)

[Figure 9]

Figure 9: Token usage.

Figure 9 visualizes the trade-off between computational cost (measured in average token usage per query) and performance (accuracy). As indicated by the directional arrow, the top-right quadrant represents the ideal balance of High Accuracy and Low Token Usage. This is the target frontier for future agentic development. The plot reveals that current high-performing agents like o4-mini-deepresearch achieve superior accuracy but at the expense of significantly higher token consumption, likely due to extensive iterative browsing and multistep reasoning. In contrast, lightweight models like gpt-4o-mini are token-efficient but lack the deep reasoning capabilities required for complex video tasks. The arrow suggests that the goal of future research is to push these agents "upwards" towards the ideal zone, optimizing the reasoning process to be both accurate and token-efficient.

#### A.6 Benchmark Case Studies

Our evaluation reveals a critical reliance on textual metadata. In dynamic scenarios like sports, where specific gameplay moments (e.g., a specific foul sequence or a buzzer-beater in overtime) are not indexed in search engine results, models fail to answer even when the visual evidence is clear. Consider the example of identifying two NBA teams based on a double-overtime sequence in Figure 10. The visual evidence explicitly shows the Houston Rockets and Oklahoma City Thunder. However, because this specific game narrative does not appear in textual game logs, both GPT-4o and Gemini-2.5-

Pro refuse to answer, stating that the information is impossible to determine.

Similarly, in Figure 11 involving technical fouls and baseline altercations , models like GPT-5.1 (w/ Search) default to Unknown. This demonstrates that current search-augmented models often function merely as text search, incapable of using temporal grounding to fill information gaps when the open web falls silent.

- In Figure 12, the agent is asked to read the temperature on a digital display inside a space module. The visual ground truth is 85. However, o4-minideep-search retrieves an external news report stating the oven’s design specification is 190◦C and confidently outputs that incorrect value. The model successfully retrieved relevant text but failed to verify it against the visual reality.
- In Figure 13, a complex query requiring the identification of a comic villain (Loki), a related film (Iron Man), and a specific visual detail (a food stain), models struggle to maintain coherence. While GPT-5.1 correctly identifies the “Cheeseburger” by linking the text clues to the correct movie scenes, Gemini-2.5-Pro fails the reasoning chain. It incorrectly identifies the stain as an “Acai popsicle” by hallucinating a scene from Avengers: Infinity War rather than the target scene from Iron Man . This highlights the difficulty agents face in maintaining context across disparate video sources.

### B Video-Browser

#### B.1 Experiment Settings

To ensure reproducibility, we detail the specific hyperparameter configurations used for the VideoBrowser agent across different experimental sections. The settings for the Planner (search loops), Watcher (sparse sampling), and Analyst (dense sampling) are adjusted to verify specific capabilities as follows:

Main Results (Table 1). For the comprehensive performance evaluation reported in Table 1, we configure the agent to maximize reasoning accuracy. Search Strategy: We set the maximum search loops to Tmax = 5. In each iteration, the top K = 3 most relevant videos are retrieved and processed. Perception: The Watcher employs a sparse sampling rate of 16 frames per video for ROI localization. For the Analyst’s Zoom-in stage, we employ a dense sampling rate of 1 FPS within the identified temporal windows, with a maximum cap of 32 frames per window to balance context limits.

Efficiency Analysis (Table 2). To strictly evaluate the token efficiency of Pyramidal Perception against baselines, we standardize the interaction rounds. Search Strategy: We fix the search process to a single loop (Tmax = 1) with K = 3 retrieved videos. Baselines: For Direct Visual Inference and Summarization baselines, we adopt a uniform sampling strategy of 128 frames per video. Video-Browser (Ours): We use 16 sparse frames for the Watcher. For the Analyst, we maintain the dense sampling setting of 1 FPS with a maximum window size of 32 frames.

Test-time Scaling Analysis (Figure 3). We investigate the scaling laws by varying one parameter while fixing others. The specific configurations for each scaling dimension are: Scaling Video Number: We fix the search loop to T = 1 and sparse sampling to 16 frames, while varying the number of retrieved videos. Scaling Search Loops: We fix the sparse sampling to 16 frames. To prevent context explosion during multi-step reasoning, we fix the number of retrieved videos per round to K = 2, while varying the max search loops. Scaling Watcher Frames: We fix the video number to K = 2 and search loop to T = 1. The dense sampling is kept at 1 FPS (max 32 frames), while we vary the sparse sampling frames in the Watcher.

Ablation Experiments (Table 3). Search Strategy: We fix the search process to a single loop (Tmax = 1) with K = 2 retrieved videos. We use 16 sparse frames for the Watcher. For the Analyst, we maintain the dense sampling setting of 1 FPS with a maximum window size of 32 frames.

- B.2 Prompts We provide all prompts used in the Video-Browser.

- B.2.1 Direct Visual Perception Prompts Check at Figure 15.

- B.2.2 Text-centric Summarization Prompts Check at Figure 16 and Figure 17.
- B.2.3 Pyramidal Perception Prompts Check at Figure 18, Figure 19, Figure 20 and Figure 21.

#### B.3 Case Studies

In this section, we present comprehensive case studies to provide a qualitative analysis of VideoBrowser. Specifically, we illustrate: (1) the end-toend workflow of the agent; (2) comparative analy-

ses against key baselines, including Direct Visual Inference and Text-Centric Summarization; and (3) a discussion of typical failure cases.

#### B.3.1 End-to-End Workflow Analysis

To demonstrate the autonomous decision-making capabilities of Video-Browser, we present a complete execution trace in Figure 7. The user query involves a complex anecdote about a legendary power forward and a No. 1 draft pick center from Asia, asking for a specific game score associated with a bet.

Strategic Planning and Self-Correction. Initially, the Planner attempts a broad, natural language query mixing all semantic elements (NBA bet kiss donkey butt...”), which yields zero results due to search engine limitations. Crucially, the agent does not hallucinate or give up. Instead, it performs a Gap Analysis, realizing the need for specific entity grounding. In the second step, it correctly identifies the entities as Charles Barkley” and “Yao Ming” and refines the query to be keyword-focused.

Pyramidal Perception in Action. Upon retrieving 10 candidate videos with the refined query, the Watcher employs the semantic sieve to select the top-3 most relevant videos. Through sparse scanning, the Planner confirms that the retrieved content contains the specific answer (Yao Ming went 9-for-9... scored 20 points”) and terminates the search loop early to save tokens. Finally, the Analyst zooms in on the relevant timestamps (e.g., 20.2s - 31.5s in Video 2) to synthesize the final answer: 20 points. This case exemplifies the agent’s ability to navigate from zero knowledge to precise verification through iterative feedback.

#### B.3.2 Paradigm Comparison

To validate the superiority of Pyramidal Perception over existing paradigms, we provide two distinctive comparison cases: another on fine-grained attribute verification (Figure 4) and one focusing on complex event reasoning (Figure 22).

Case 1: Fine-grained Visual Attributes. Figure 4 presents a needle-in-a-haystack query regarding the color of a pen cap in the film The Secret Life of Walter Mitty. Summarization fails explicitly due to the Modality Gap. Textual summaries and metadata focus on plot narratives and rarely index minute visual details like props or their colors. Direct Visual Inference also fails to determine the answer. Despite processing visual frames, the model struggles to attend to the small pen cap” amidst

the noise of the full video context, concluding that no pen or pen cap can be seen”. Video-Browser succeeds by leveraging its Zoom-in mechanism. It identifies the relevant close-up shots of the character’s pocket and performs targeted decoding, allowing it to clearly distinguish the distinctly red cap” that serves as a symbolic plot device.

Case 2: Complex Dynamic Reasoning. In Figure 22, the user query describes a highly specific, multistage basketball sequence from the 2024 NBA playoffs” (alley-oop, block, trip, slam floor). Direct Visual Inference fails because it retrieves irrelevant videos (e.g., Giannis’s 2021 block or Bronny James clips) and cannot semantically match the complex visual sequence to the query within the raw pixel stream. Summarization also fails, as the textual summaries describe unrelated games (e.g., Wizards vs. Pelicans” or 2016 NBA Finals”), exhibiting a severe hallucination issue where the model claims no relevant sequence exists. In contrast, VideoBrowser correctly filters the candidate videos and localizes the exact moment in the 2024 NBA Finals, Mavericks vs. Celtics.” By accurately interpreting the visual commentary and action (“Tatum step back, stopped, blocked”), it correctly identifies the player as Jayson Tatum.

#### B.3.3 Failure Case Analysis

Despite the significant improvements, VideoBrowser still exhibits limitations in specific complex scenarios. We analyze typical failure modes to provide insights for future improvements.

- 1. Semantic Distractors (Fig. 25). In open-ended browsing, multiple videos may share similar semantic attributes. In the Icelandic food case, the agent correctly identified the context (uncomfortable food in Iceland) but was misled by a distractor candidate, fermented shark (hákarl), which shares the attribute of being smelly and hard to eat. The agent failed to distinguish the specific target (“sheep’s butt”) from the semantically similar distractor, highlighting the need for stricter entity verification.
- 2. Fine-grained Visual Hallucination (Fig. 26. For extremely small objects, the model may hallucinate specific attributes. In the friendship test case, while the agent correctly located the scene and the object (a beverage can), it misidentified the brand as Dr Pepper instead of the ground truth “Coca-Cola.” This suggests that even with Zoom-in, current MLLMs still struggle with zero-shot OCR or logo recognition on low-resolution objects.

#### 3. Information Deficit & Entity Mismatch The

- case 23 (Kyle Lowry) demonstrates an Information Deficit: the agent successfully retrieved the correct highlight clip but failed to answer because the specific statistic (final score) was neither spoken in the transcript nor explicitly displayed in the visual summary. The agent honestly reported not stated, failing to infer or retrieve external box scores. The
- case 24 shows an Action-to-Entity Mismatch. The agent observed the correct basketball plays (block, steal, save) but misattributed them to LeBron James instead of the target player (who played for the Lakers and Thunder), leading to a factually correct answer for the wrong entity.

###### Level 1

In an NBA Christmas promo video from a certain year, a lanky small forward nailed the first three-pointer that rang with a bell, and a player wearing jersey number 6 later dunked. Who was the passer to the player wearing number 6, and which NBA team did he play for at the time?

Question

Los Angeles Lakers

Ground Truth

- 1: Confirm that this promotional video is the 2013 NBA Christmas promo.
- 2: Confirm that the player wearing jersey number 6 who dunked is LeBron James.
- 3: Confirm that the passer to James was Steve Nash, who was playing for the Los Angeles Lakers at the time.

Checklist

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

###### Level 2

The first Sequoia Chair Professor at a renowned comprehensive university taught a course on “Deep Learning for Computer Vision” at a world-top-ranked university in 2025. In one lecture devoted to “A core task in Computer Vision,” the professor introduced the concept of the semantic gap. When the slides asked, “What does the computer see?”, the answer given in the lecture was:

Question

An image is a tensor of integers

Ground Truth

- 1: Confirm the instructor is Fei-Fei Li
- 2: Confirm the institution is Stanford and the course is CS231N | Spring 2025
- 3: Identify the lecture as Lecture 2: Image Classification with Linear Classifiers
- 4: Locate the segment “What the computer sees”; the answer is “An image is a tensor of integers”

Checklist

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

###### Level 3

An actor starred in a film adapted from a famous novel, playing the lead who goes to war to earn a better future for the woman he loves, makes money overseas at all costs, returns to find she has already married, yet still tries every means to see her and rekindle their romance, and ultimately dies alone. In another film the same actor plays an African smuggler, befriends a Black man, escapes alive, kills the colonel, and—mortally wounded—hands the crucial item to his Black friend before dying. What is it?

Question

Diamond

Ground Truth

- 1: Identify the first film as the 2013 version of “The Great Gatsby”
- 2: Confirm the actor is Leonardo DiCaprio as Gatsby
- 3: Identify the second film as “Blood Diamond”
- 4: Locate the scene where, dying, he gives the priceless diamond to Solomon to improve his family’s life; the answer is the diamond

Checklist

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

[Figure 53]

[Figure 54]

Figure 5: Benchmark samples.

Evaluation Prompt Your response should be in the following JSON format: {

"Explanation": "your explanation for your final answer", "Answer": "your succinct, final answer", "Confidence": "your confidence score between 0% and 100% for your answer"

}

Judge Prompt Question: {question} Ground Truth: {ground_truth} Model Prediction: {prediction}

Evaluate if the Prediction matches the Ground Truth. Return JSON: {{"is_correct": true}} or {{"is_correct": false}}

Figure 6: Full prompts used for evaluation and judging.

Workflow of the Video-Browser [User Query]

A legendary power forward, after switching careers to become a commentator, once bet with his co-host on a popular American basketball analysis show that a No. 1 draft pick center from Asia could not score 19 points in a single game. Subsequently, the center proved himself in a game, forcing the commentator to fulfill the bet — kissing a donkey’s butt on a subsequent live broadcast. What was the center’s final score in that game?

[Planner] Thought: Decomposition:

- 1) Identify the legendary power forward turned commentator on a popular American basketball analysis show.
- 2) Identify the No. 1 draft pick center from Asia involved in the bet.
- 3) Find the specific game where he needed at least 19 points.
- 4) Determine his final point total in that game. Gap Analysis: We have no prior videos or leads, so everything is missing. Strategy: Start broadly but targeted to the well-known anecdote about a commentator betting that an Asian No. 1 pick center wouldn’t score 19 points and then having to kiss a donkey’s butt on air. The key is to surface the exact game and box score. One precise query that includes all distinctive elements (bet, donkey, kiss, Asian No. 1 pick center, points) should quickly reveal the names, game, and final scoring line. [Planner] Queries: [’NBA bet kiss donkey butt Asian No. 1 pick center 19 points game what was his final score’]

#### [Searcher]

executing 1 queries: [’NBA bet kiss donkey butt Asian No. 1 pick center 19 points game what was his final score’] -> YouTube found 0 results for ’NBA bet kiss donkey butt Asian No. 1 pick center 19 points game what was his final score’

[Searcher] Found 0 unique video candidates.

[Watcher] Starting sparse sampling and window identification... [Planner]

- Step 1/3 -> Knowledge Accumulation: 0 videos in store, 1 queries tried. -> Round complete. Continuing search loop (strict mode)...

[Planner] Thought: Decomposition:

- 1. Identify the personalities and show: legendary power forward turned commentator, co-host, popular American NBA analysis show, bet about an Asian No. 1 pick center (clearly Yao Ming) scoring at least 19 points, and the loser kissing a donkey’s butt on live TV.
- 2. Then find the specific game where Yao Ming ‘proved himself’ and exceeded 19 points, and extract his final point total in that game.

Figure 7: Workflow of the Video-Browser

Gap Analysis: - We have no video or textual confirmation yet; everything is missing. - The prior search was very general and mixed many keywords, which may dilute results. Strategy: - Directly target the well-known anecdote about Charles Barkley (legendary PF) and Yao Ming on TNT’s Inside the NBA, and then focus on the game box score from the night that resolved the bet. The key is to find a video or segment that explicitly states Yao Ming’s point total in that proving game. - Use a concise, entity-focused query with both names and mention of the bet and points.

[Planner] Queries: [’Charles Barkley bet Yao Ming 19 points game final score donkey Inside the NBA’]

[Searcher] executing 1 queries: [’Charles Barkley bet Yao Ming 19 points game final score donkey Inside the NBA’] -> YouTube found 10 results for ’Charles Barkley bet Yao Ming 19 points game final score donkey Inside the NBA’ [Searcher] Found 10 unique video candidates.

#### [Watcher]

Starting sparse sampling and window identification... -> Selecting top videos using LLM... -> LLM selected video indices: [0, 1, 3] -> Selected 3 videos for processing. -> Processing Video 1/3: Inside the NBA: If Yao Ming scores 19 points...

Cache hit for video file: data/cache/videos/1ARR0XqanM.mp4 -> Extracted 16 frames with timestamps. Cache hit for transcript with timestamps......

Cache hit for video file: data/cache/videos/SOK0mVRPfdk.mp4 -> Extracted 16 frames with timestamps. Cache hit for transcript with timestamps......

Cache hit for video file: data/cache/videos/I5BqF2Dn5bk.mp4 -> Extracted 16 frames with timestamps. Cache hit for transcript with timestamps......

#### [Planner]

- Step 2/3 -> Knowledge Accumulation: 3 videos in store, 2 queries tried. -> Round complete. Continuing search loop (strict mode)...

#### [Planner]

Thought: The user already has the key details confirmed: the show is Inside the NBA, the commentator is Charles Barkley, the center is Yao Ming, and multiple verified clips explicitly state that in the game triggering the bet payoff Yao Ming went 9-for-9 in L.A. and scored 20 points. This directly answers the user’s question about the final score in that game. No remaining sub-question is unsolved, and any further search would be redundant verification of the same specific stat line.

[Searcher] No queries provided. Skipping.

[Watcher] Starting sparse sampling and window identification...

— Step: watcher —

#### [Checker]

- Step 3/3 -> Knowledge Accumulation: 3 videos in store, 2 queries tried. -> Max loops reached. Proceeding to Analyst.

— Step: checker [Analyst] Extracting relevant clips and synthesizing final answer... -> Processing Video 1: Inside the NBA: If Yao Ming scores 19 points... (3 windows) Cache hit for video file: data/cache/videos/1ARR0XqanM.mp4 -> Extracting clip 7.0s - 22.0s (1 FPS)... -> Added [‘[0.0s 7.0s] Next Thursday at Yao Ming, the Rockets taking on Dallas and also Seattle and Portland.’, “[7.0s - 11.0s] Will this be the week to have to kiss Kenny‘s ass?”, ’[11.0s - 13.0s] Why would you do that, Chuck......[51.0s - 54.0s] You say he’s going to get 19 points in the game.", "[54.0s 56.0s] I’m telling you, he has some offensive skills.", ’[56.0s - 59.0s] Why would you go without eating until he gets 19 points?’...... ’[142.0s - 145.0s] Man, I hope he gets to 19 next week.’, ’[145.0s - 147.0s] Anyway, in our game.’] transcript lines. -> Processing Video 2: Charles Barkley kisses Kenny Smith’s Donkey (1 windows) Cache hit for video file: data/cache/videos/SOK0mVRPfdk.mp4 -> Extracting clip 20.2s - 31.5s (1 FPS)... -> Added [’[19.2s - 20.2s] And’, "[20.2s - 23.4s] So Sunday night y’all mean went nine for nine", ’[24.4s 26.4s] in LA’, ’[26.8s - 31.5s] 20 points and and everybody who heard those words was saying well’, "[31.5s - 36.0s] I guess that means Charles are gonna have to pay up. I wonder how that’s gonna happen"] transcript lines. -> Processing Video 3: Inside the NBA on TNT Throwback to when Charles Barkley kiss kenny ass if Yao Ming scores 19 points (1 windows) Cache hit for video file: data/cache/videos/I5BqF2Dn5bk.mp4 -> Extracting clip 172.0s - 188.0s (1 FPS)... -> Added ["[169.0s - 172.0s] I’m going to kiss his ass right here in front of everybody.", ’[172.0s - 179.0s] And so Sunday night, Yao Ming went 9 for 9 in L.A.’, "[179.0s - 180.0s] That’s embarrassing.", ’[180.0s - 184.0s] He scored 20 points, and everybody who heard those words was saying,’, ’[184.0s - 188.0s] well, I guess that means Charles is going to have to pay up.’, "[188.0s - 189.0s] I wonder how that’s going to happen."] transcript lines. -> Invoking Analyst LLM with video context...

FINAL ANSWER: "Explanation": "The videos show Charles Barkley on ’Inside the NBA’ betting that Yao Ming would not score 19 points in a game. In the recap clip following the game in Los Angeles, the host states that ’Sunday night Yao Ming went 9 for 9 in L.A. ... He scored 20 points.’ This confirms Yao Ming’s final point total in the game that triggered Barkley to fulfill the bet.", "Answer": "20 points", "Confidence": "100%"

[Figure 55]

###### Figure 10: Level 2 case, NBA teams.

[Figure 56]

###### Figure 11: Level 2 case, NBA player.

[Figure 57]

###### Figure 12: Level 2 case, space module.

[Figure 58]

###### Figure 13: Level 3 case, cross-film.

Direct Visual Inference Prompt

You are an intelligent video assistant. User Query: "user_query" You have analyzed len(target_videos) videos. Here is the information extracted from them: combined_context_text Instructions: - Use the provided transcripts and visual frames from ALL videos to answer the user’s query.

- - Synthesize information if the answer is spread across multiple videos.
- - If none of the videos contain the answer, state that clearly.
- - Be concise and direct. Your response should be in the following JSON format: {{

"Explanation": "your explanation for your final answer", "Answer": "your succinct, final answer", "Confidence": "your confidence score between 0% and 100% for your answer"

}}

Figure 15: Direct Visual Inference Prompt

Summarization Prompt

You are a video analyst. User Query: "user_query" Video Title: video.get(’title’, ’Unknown’) Transcript: transcript_text[:20000] Instructions: - Analyze the visual frames and transcript provided.

- - Provide a detailed summary of the video content that is RELEVANT to the User Query.
- - If the video contains specific details (numbers, names, events) related to the query, include them explicitly.
- - If the video is irrelevant, state that it contains no relevant information.

Figure 16: Summarization Prompt

Summarization Analyst Prompt

You are an intelligent assistant. User Query: "user_query" You have received detailed summaries from several videos analyzed by your vision team: context_text Instructions: - Synthesize the information from these summaries to answer the user’s query.

- - If the answer is found, provide it clearly and concisely.
- - If the videos do not contain the answer, state that.
- - Do not make up information not present in the summaries. Your response should be in the following JSON format:

{{

"Explanation": "your explanation for your final answer", "Answer": "your succinct, final answer", "Confidence": "your confidence score between 0% and 100% for your answer"

}}

Figure 17: Summarization Analyst Prompt

Planner Prompt

You are the Planner for a Video Agent. Your primary role is to analyze the User’s Goal and current investigation status, then generate the most effective search queries to find relevant video content. Current Task: Based on the information gathered so far, formulate search queries to find missing information or refine existing knowledge to fully address the User’s Goal. Context:

- - You are part of a research loop. Your output (‘search_queries‘) will be used by a ‘Searcher‘ tool.
- - You do NOT need to decide to "watch" or "finish"
- - the system flow handles that. Your ONLY job is to determine *what to search for next*.

#### Action Guidelines:

- 1. **Decompose the User’s Goal:** Break down the complex query into smaller, distinct subquestions or requirements.
- 2. **Analyze Knowledge Gaps (The "Gap Analysis"):**

- * For each sub-question, check the ‘Current Knowledge Status‘.
- * **Solved:** Do we already have verified evidence for this part? -> *Do not search for this again.*
- * **Partial/Unverified:** Do we have watched videos waiting for verification? -> *Wait, or search for corroborating details if unsure.*
- * **Missing:** Is this part completely missing? Or did previous attempts fail (Rejected videos)?

-> *This is your Focus.*

- 3. **Follow the Leads (Deep Dive / Hypothesis Verification):**

- * **CRITICAL:** If previous videos (even rejected ones) mentioned specific **candidates, entities, or clues** (e.g., "Video mentioned onions and garlic as staples"), **PIVOT** your strategy.
- * Stop searching for the general topic (e.g., "Ancient Egyptian food").
- * Instead, **verify these candidates** against the user’s remaining conditions (e.g., "Did onions originate in Central Asia?", "Garlic symbolism medieval Europe").

- 4. **Formulate Targeted Queries:**

- * Generate queries **ONLY for the Missing parts** or **Hypothesis Verification**.
- * If previous queries failed (see ‘Rejected Videos‘ reasons), change your strategy (e.g., use more specific technical terms, add date filters like "2024", or switch from "how to" to "review").
- * **LIMIT:** Generate a maximum of max_queries queries. Example of Desired Reasoning:
- *User Query:* "History of the Roman Empire and the engineering behind aqueducts."
- *Current Knowledge:*

- - [Verified] "Roman Empire Overview" (Covers the history timeline, emperors)
- - [Rejected] "Aqueduct Tour" (Rejected: "Visuals only, no engineering explanation")

- *Your Output:*

{

"thought": "1. Decomposition:\n - Q1: History? -> [SOLVED] 'Roman Empire Overview' covers timeline.\n - Q2: Engineering of Aqueducts? -> [MISSING] 'Aqueduct Tour' was rejected for lacking details.\n2. Strategy: Ignore history. Focus specifically on the structural engineering and physics of aqueducts.", "search_queries": ["roman aqueduct structural engineering physics", "how roman

aqueducts work construction details"] }

Figure 18: Planner Prompt

**Example of Hypothesis Verification (Deep Dive):** *User Query:* "A food originating in Central Asia, depicted in Egyptian murals, symbolizing victory in Europe."

- *Current Knowledge:*

- [Rejected] "Egyptian Diet" (Reason: "Mentions **Garlic** and **Onions** as staples, but doesn’t confirm Central Asia origin.")

- *Your Output:*

{

"thought": "1. Leads Found: Previous video identified 'Garlic' and 'Onions' as candidates for the Egyptian part.\n

- 2. Gaps: Need to verify if either of them matches the 'Central Asia origin' and 'Victory symbolism' conditions.\n
- 3. Strategy: Switch from general search to specific verification of these candidates.", "search_queries": ["did garlic originate in Central Asia", "onion origin

Central Asia", "garlic symbolism victory medieval Europe", "onion symbolism

victory Europe"] }

**Output Format:** Provide a JSON object with the following structure: {

"thought": "Step-by-step reasoning: Decomposition -> Leads/Hypothesis -> Gap Analysis -> Strategy.", "search_queries": ["query1", "query2", "query3"]

}

Select Prompt

User Query: "user_query" You are provided with a list of video search results. Select the top top_k most relevant videos that are likely to contain the answer to the User Query. Candidates: candidates_info Return the indices of the selected videos as a JSON list of integers. Example: [0, 2, 4]

Figure 19: Select Prompt

Localization Prompt You are a video investigator. User Query: "user_query" Video Title: video.get(’title’, ’Unknown’) I have sampled len(frames_data) frames from the video at specific timestamps: frame_descriptions Transcript (with timestamps): truncated_transcript Task: Identify all specific temporal windows (start_time to end_time) in the video that are relevant to answering the User Query. - Combine visual cues from the frames and semantic cues from the transcript. - If the answer is in the transcript, use its timestamps. - If the answer is visual, use the frame timestamps to estimate the window. - If the video is irrelevant, return an empty list or set relevant to false. - Provide a brief reasoning for each selected window. Output Format (JSON): {{

"relevant": true/false, "windows": [

{{

"start_time_seconds": <float>, "end_time_seconds": <float>, "reasoning": "..."

}}, ...

] }}

Figure 20: Localization Prompt

Analyst Prompt Based on the video clips and transcripts provided above, answer the User Query. Response Format: {{

"Explanation": "your explanation for your final answer", "Answer": "your succinct, final answer", "Confidence": "your confidence score between 0% and 100% for your answer"

}}

Figure 21: Analyst Prompt

- Question: In a 2024 NBA playoff game, after finishing an alley oop dunk, on the defensive end he recorded a big block on the opponent's three-point attempt. He was then tripped by the opponent, and after being tripped the player angrily slammed the floor. Who was the player whose shot was blocked? Answer: Jayson Tatum

Direct Visual Inference: The described play—an alley oop dunk followed by a chase-down block on a three-point attempt, a trip, and the player angrily slamming the floor—does not appear in any of the three provided videos or their frames. Video 1 shows Giannis Antetokounmpo’s famous 2021 NBA Finals alley-oop from Jrue Holiday vs. the Phoenix Suns, which includes a steal on Devin Booker and a foul by Chris Paul, but there is no blocked three-point attempt or trip afterward. Video 2 is a general compilation of blocked dunks from many seasons and teams, not a specific 2024 playoff sequence. Video 3 is a Bronny James alley-oop highlight with no such follow-up defensive play. Because the specific 2024 playoff moment the user describes is not contained in the supplied material, I cannot identify whose shot was blocked from these videos alone. Answer: The videos provided do not contain the 2024 playoff sequence described, so the player whose shot was blocked cannot be determined from this material.

Summarization: None of the provided video summaries describe a 2024 NBA playoff sequence where a player finishes an alley oop dunk, then blocks a three-point attempt on the defensive end, is tripped, and angrily slams the floor. Video 1 is a regular-season Wizards vs. Pelicans game with no three-point attempt or trip, Video 2 is from the 2023 Western Conference Finals and only shows an offensive alley oop, and Video 3 is from the 2016 NBA Finals with no relevant sequence. Therefore, the player whose shot was blocked cannot be identified from the given information. Answer: The videos do not provide enough information to identify the player whose three-point shot was blocked.

Pyramidal Perception: The described sequence—an alley oop dunk followed by a big block on a three-point attempt, then the defender being tripped and angrily slamming the floor—occurs in the 2024 NBA Finals, Mavericks vs. Celtics. In the clip labeled “Luka Doncic alley-oop pass to Gafford and Block by gafford🔥🏀Mavs Vs Celtics Game 4,” the commentator explicitly calls the play: “Tatum step back, stopped, blocked by Gafford on a three-point attempt, and then a loose ball foul on Tatum.” This confirms that the blocked three-pointer was taken by Jayson Tatum. Answer: Jayson Tatum.

[Figure 59]

[Figure 60]

Figure 22: Qualitative comparison of three paradigms.

- Question: In a 2025 NBA regular-season game, one player first recorded a jaw-dropping rim block on defense, then missed an open layup on offense, and later on a fast break was called for an offensive foul. What was this player's final score for that game? Answer: 6

ours: The relevant clip is “Unbelievable Basketball Move Kyle Lowrys Jaw Dropping Under the Rim Offensive Foul 1.” It describes, in one 2025 regular-season game, Kyle Lowry committing two Shaqtin’-type plays in the same game, including a bizarre under-the-rim offensive foul on a fast break. However, the provided transcript only captures the commentators reacting to the play itself; it never states Lowry’s stat line, box score, or final points total for that game. None of the other supplied clips or transcripts add any game-stat information about this sequence. Without a box score or additional context, his final scoring total for that game cannot be determined from the materials given. Answer: The player is Kyle Lowry, but his final point total for that game is not stated anywhere in the provided clips or transcripts, so it cannot be determined from the given information.

- Figure 23: Failure Case 1.

Question: In a 2025 NBA regular-season game, a player recorded a steal and a slam dunk in the third quarter, and later, on a ball-saving play in the fourth quarter, he passed the ball to the bench. Which two teams did this player win championships with? Answer: Los Angeles Lakers, Oklahoma City Thunder

ours: From the Lakers–Clippers full game highlight transcript (December 20, 2025), the specific play described in the question appears in the third quarter: “Here it is LeBron all the way… three point opportunity…” followed soon after by commentary noting “LeBron … the king lays it up and in,” and later a sequence where he gets a steal and goes “straight to the house,” i.e., a slam. In the fourth quarter, the other highlight reel of the same game (the national broadcast version) includes a notable hustle/ball-saving play where LeBron, while saving the ball from going out of bounds, ends up passing/throwing it toward the Lakers bench area. The player in question is LeBron James. LeBron has won NBA championships with two franchises: the Miami Heat and the Cleveland Cavaliers. Answer: Miami Heat and Cleveland Cavaliers

- Figure 24: Failure Case 2.

Question: There is a travel reality show starring an actress who appeared in 'Eternal Love' and an actor who appeared in 'Romance in the Alley', both participating. They visit a country whose capital is the closest to the Arctic Circle, experiencing local culture. During a festival focused on food, an actress who acted in 'Being a Hero' tries a food that makes her extremely uncomfortable, yet she still pretends it's tasty. What is this food? Answer: sheep's butt

ours: The show is filmed in Iceland, whose capital Reykjavík is the closest national capital to the Arctic Circle. In the Iceland food-exploration clips, one dish is highlighted as extremely smelly and hard to eat: “冰岛发酵的酸鲨⻥⾁”, described as one of the smelliest foods in the world. The taster reacts with obvious discomfort but forces himself to continue and describes it despite clearly not liking itmatching the description that the actress from 'Being a Hero' finds the food very uncomfortable yet pretends it's tasty. Therefore, the food in question is Icelandic fermented shark. Answer: Icelandic fermented shark (fermented sour shark meat, hákarl).

- Figure 25: Failure Case 3.

Question: In the same film, after the wealthy grandfather dies, the grandson must pass 12 tests; the second test is to form a true friend. What drink did this friend drink when presenting themselves to the lawyer for confirmation?

Answer: Coca-Cola ours: In the lawyer Hamilton’s office, Jason’s ‘true friend’ Emily sits at the conference table with a beverage. The video frame (around 489.8s) clearly shows a can of Dr Pepper with a straw beside her as she talks to Hamilton. No other drink is shown or mentioned for her in this confirmation scene. Answer: Dr Pepper (from a can, with a straw)

- Figure 26: Failure Case 4.

