## TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding

### ♠†Max Ku∗ ♡Thomas Chong∗ ♠Jonathan Leung ♠Krish Shah ♡Alvin Yu ♠†Wenhu Chen

m3ku@uwaterloo.ca, thomas.chong@votee.ai, wenhu.chen@uwaterloo.ca

♠University of Waterloo ♡Votee AI †Vector Institute

# arXiv:2502.19400v2[cs.AI]25May2025

https://tiger-ai-lab.github.io/TheoremExplainAgent/

Prompt: Show your understanding on Bubble Sort.

[Figure 1]

[Figure 2]

[Figure 3]

(Multimodal Elements) Bubble Sort is … Explainability

(Text Element) Bubble Sort is a straightforward sorting algorithm that works on the principle of repeatedly comparing and swapping adjacent elements until the list is sorted. The process involves iterating

[Figure 4]

Compare 1st and 2nd Compare 2nd and 3rd Compare 3rd and 4th Compare 4th and 5th Repeat until sorted

|4|2|9|1|5|
|---|---|---|---|---|

|2|4|9|1|5|
|---|---|---|---|---|

Interpretability

[Figure 5]

|2|4|1|9|5|
|---|---|---|---|---|

through the list multiple times …

|2|4|1|5|9|
|---|---|---|---|---|

[Figure 6]

[Figure 7]

Harder to Grasp at a Glance More Intuitive (Clearer with Visuals)

Figure 1: We do not have knowledge of a thing until we have grasped its cause (Aristotle, 1901). A strong reasoning model should not only generate correct conclusions but also communicate them effectively. Visualization enhances human intuition by making abstract concepts more concrete and revealing hidden relationships. Moreover, visual explanations expose reasoning errors more clearly than text, making it easier to diagnose model mistakes.

### Abstract

### 1 Introduction

Understanding domain-specific theorems often requires more than just text-based reasoning; effective communication through structured visual explanations is crucial for deeper comprehension. While large language models (LLMs) demonstrate strong performance in text-based theorem reasoning, their ability to generate coherent and pedagogically meaningful visual explanations remains an open challenge. In this work, we introduce TheoremExplainAgent, an agentic approach for generating long-form theorem explanation videos (over 5 minutes) using Manim animations. To systematically evaluate multimodal theorem explanations, we propose TheoremExplainBench, a benchmark covering 240 theorems across multiple STEM disciplines, along with 5 automated evaluation metrics. Our results reveal that agentic planning is essential for generating detailed longform videos, and the o3-mini agent achieves a success rate of 93.8% and an overall score of 0.77. However, our quantitative and qualitative studies show that most of the videos produced exhibit minor issues with visual element layout. Furthermore, multimodal explanations expose deeper reasoning flaws that text-based explanations fail to reveal, highlighting the importance of multimodal explanations.

A key objective of AI systems is to assist humans in solving complex problems, particularly in domain-specific challenges. To achieve this, AI must go beyond surface-level pattern matching to achieve deeper conceptual understanding to effectively address these problems. Recent research has proposed evaluating AI performance on theoremdriven datasets through multiple-choice question answering (Zhang et al., 2024) and open-ended short question answering (Chen et al., 2023b). However, these approaches primarily assess textual reasoning and may not fully capture an AI system’s ability to grasp theorem concepts at a deeper level. Studies have shown that AI models can be sensitive to superficial cues, such as the order of answer choices in multiple-choice questions (Pezeshkpour and Hruschka, 2023; Keluskar et al., 2024). This raises concerns about the robustness of such evaluations in truly measuring comprehension. Moreover, current theorem-focused datasets are predominantly text-based, overlooking how complex concepts are often best understood through structured visualizations.

Theorem reasoning is inherently multimodal, particularly in areas such as geometry, topology,

I understand this theorem! I will explain it with visuals.

|Theorems|
|---|
|Thevenin's theorem|
|Octet Rule|
|Run Length Encoding|
|Linear Search|
|…|

Accuracyand Depth = 1.0

[Figure 8]

- = 0.8
- = 1.0

Visual Relevance

[Figure 9]

Output

Prompting

= 1.0

Logical Flow

Element Layout

Model

Context

= 1.8

Visual Consistency

Evaluation (Automatic/Human Expert)

Multimodal Elements

Metrics

Figure 2: An overview of the multimodal theorem explanation framework.

and certain aspects of algebra, where visual representations and spatial reasoning play a crucial role in understanding structures and proving properties. Cognitive science research suggests that multimodal elements improve conceptual understanding, aiding in the comprehension of abstract ideas. Although some studies leverage multimodal input to improve AI reasoning (Zhang et al., 2023b), currently there is no standardized evaluation framework to evaluate AI’s ability to generate multimodal explanations for complex concepts, which would require models to express knowledge in an interpretable manner. This raises the question: Can AI systems effectively generate multimodal theorem explanations?

As video is a classic example of multimodal data, we explore the question by introducing TheoremExplainAgent, an agentic AI system designed to generate theorem explanations in the form of explanatory videos. TheoremExplainAgent demonstrates the capability to plan and generate long, coherent videos by mimicking human video production processes. In this system, a planner agent generates story plans and narrations, and a coding agent generates Python animation scripts using Manim (The Manim Community Developers, 2024) to create long and meaningful videos. Additionally, to systematically evaluate AI-generated explanations, we develop TheoremExplainBench, a benchmark suite comprising 240 theorems spanning four STEM disciplines. We assess AI-generated explanations based on 5 dimensions related to factual correctness and perceptual quality, using automatic or humanevaluation metrics. An overview of the framework is illustrated in Figure 2.

Our experiments with TheoremExplainAgent yielded both promising results and clear areas for improvement in AI-generated multimodal theorem explanations. On the positive side, a key achievement was the system’s ability to generate extended video explanations, reaching durations of up to

10 minutes. This represents a significant advancement over agentless approaches, which we found to be limited to approximately 20-second videos. Furthermore, TheoremExplainAgent demonstrated versatility across different STEM disciplines, successfully creating videos for Mathematics, Physics, Chemistry, and Computer Science. Importantly, we observed that video-based theorem explanations inherently expose deeper reasoning flaws in AI systems that text-based evaluations often miss. Unlike text-based multiple-choice questions, where models can exploit superficial cues, generating visualtheorem explanations necessitates that the AI explicitly encodes structural and procedural knowledge, thus making underlying errors more apparent. In particular, the o3-mini model exhibited robust performance at varying levels of theorem difficulty, indicating a capacity to handle both fundamental and complex concepts. However, despite these successes, limitations persist. While the system could generate textually accurate explanations, the visual quality and pedagogical structure of the videos require further refinement. Generated animations frequently exhibited minor visual layout inaccuracies, such as misaligned text elements, overlapping shapes, and inconsistent object sizes. These visual errors, though often subtle, became more pronounced and potentially distracting. This happens particularly in the medium and hard difficulty levels of our TheoremExplainBench.

Therefore, the major contributions of this work:

- (1) Task Definition. We introduce the novel prob-

lem of AI-generated multimodal theorem explanations and identify the key challenges associated.

- (2) TheoremExplainAgent. We develop an agen-

tic approach to generating explanatory videos, as a baseline to assess current AI capabilities.

- (3) TheoremExplainBench. We curate a diverse

benchmark dataset spanning 4 STEM disciplines and propose 5 automatic evaluation metrics, measuring progress toward solving this problem.

[Figure 10]

[Figure 11]

##### Planner Agent

Theorem

###### IEEE Conversion

Vision Storyboard Plan

Technical Implementation Plan

Animation & Narration Plan

The IEEE-754 standard describes ﬂoating-point formats, a way to represent real numbers in hardware.

Scene Outline

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Code Agent

##### Rendered Video

[Figure 17]

[Figure 18]

from manim import * class IEEEOverview(Scene): def construct(self): # Create arrow pointing to the implicit 1 arrow = Arrow(start=mantissa[0][0].get_edge(UP) + UP * 0.1,

| | |
|---|---|
| | |
| | |

Query Generator

end=mantissa[0][0].get_edge(DOWN), color=YELLOW, buff=0.1)

[Figure 19]

[Figure 20]

- Version 0

- Version 1

[Figure 21]

[Figure 22]

TypeError: Mobject.__getattr__.<locals>.getter() takes 1 positional argument but 2 were given

<THINKING> Error Type: TypeError. Root Cause: `get_edge()` requires only one argument which is the direction (LEFT, RIGHT, UP, DOWN etc.). Solution: Modify the arrow's start point to reference the entire MathTex object instead of trying to access individual characters by indexing. Use `get_left()` and `get_right()` to get the left and right edges of the "1." MathTex object correctly. <THINKING>

Core Documentation

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Code Fix

[Figure 27]

Plugin Documentation

from manim import * class IEEEOverview(Scene): def construct(self): # Create arrow pointing to the implicit 1 arrow = Arrow(start=mantissa[0][0].get_left() + UP * 0.1,

Agentic RAG (Router)

end=mantissa[0][0].get_right(), color=YELLOW, buff=0.1)

[Figure 28]

Figure 3: TheoremExplainAgent consists of two LLM agents. Taking a theorem as input, the planner agent create plans for execution. The coding agent then generates Python scripts to produce visuals and audio.

### 2 Related Works

#### 2.2 LLM in Theorems Understanding

#### 2.1 LLM and Agents

LLMs have demonstrated remarkable capabilities in solving complex mathematical problems, including formal theorem proving and symbolic reasoning. To evaluate these abilities, researchers have introduced multiple benchmark datasets, primarily consisting of multiple-choice and short-answer question answering (QA) tasks (Zhang et al., 2024; Amini et al., 2019; Hendrycks et al., 2021). Early studies centered on elementary to high schoollevel mathematics, leading to datasets such as Math23K (Zhou et al., 2023), GSM8K (Cobbe

The rapid advancements in large language models (LLMs) and large vision-language models (VLMs) have unlocked unprecedented capabilities in understanding multimodal content. Models such as GPT-4 (OpenAI, 2023), Gemini (Gemini-Team

- et al., 2024), Claude-3.5 Sonnet v1 (Anthropic, 2024), and DeepSeek (DeepSeek-AI et al., 2024) have demonstrated strong abilities in processing complex textual information and analyzing visual inputs within a unified framework (Zhang et al.,

- 2023b). These breakthroughs have enabled transformative applications across various domains, including visual content understanding (Hu et al., 2023; Ku et al., 2023), code generation (Nijkamp et al., 2023; Jimenez et al., 2024; Yang et al., 2024a), and reasoning over structured data. To tackle complex tasks, researchers have explored LLM agents: AI systems that leverage LLMs to autonomously reason, plan, and execute tasks by interacting with structured environments or external tools. These agents have been deployed in various goal-oriented applications, such as scientific discovery (Lu et al., 2024; Si et al., 2024; Schmidgall

et al., 2025), coding solutions (Abramovich et al.,

- 2024), multimodal visual generation (He et al., 2024a), and computer environment interaction (Xie et al., 2024). In this work, we extend the use of LLM agents into the domain of theorem explanation and visualization.

- et al., 2021), and GeoQA (Chen et al., 2022a). As LLM capabilities advanced, more domainspecific benchmarks emerged, extending evaluation to fields like science reasoning (ScienceQA) (Lu
- et al., 2022), financial reasoning (FinQA) (Chen et al., 2022b), and theorem comprehension (TheoremQA) (Chen et al., 2023b). These datasets collectively assess LLMs’ ability to solve mathematical and scientific problems up to the university level. However, existing benchmarks remain predominantly text-based, overlooking the role of visual intuition in mathematical reasoning. Many mathematical concepts are best understood through structured diagrams and dynamic representations, which current LLM evaluations fail to capture. To address this gap, we introduce an AI framework to generate theorem explanations in long-form videos, integrating symbolic derivations with structured visualizations to enhance comprehension.

##### Planner Agent

Theorem

###### IEEE Conversion

Vision Storyboard Plan

Technical Implementation Plan

Animation & Narration Plan

The IEEE-754 standard describes ﬂoating-point formats, a way to represent real numbers in hardware.

Scene Outline

Code Agent

Rendered Video

from manim import * class IEEEOverview(Scene): def construct(self): # Create arrow pointing to the implicit 1 arrow = Arrow(start=mantissa[0][0].get_edge(UP) + UP * 0.1,

Query Generator

end=mantissa[0][0].get_edge(DOWN), color=YELLOW, buff=0.1)

- Version 0

- Version 1

TypeError: Mobject.__getattr__.<locals>.getter() takes 1 positional argument but 2 were given <THINKING> Error Type: TypeError. Root Cause: `get_edge()` requires only one argument which is the direction (LEFT, RIGHT, UP, DOWN etc.). Solution: Modify the arrow's start point to reference the entire MathTex object instead of trying to access individual characters by indexing. Use `get_left()` and `get_right()` to get the left and right edges of the "1." MathTex object correctly. <THINKING>

Core Documentation

Error Analysis

from manim import * class IEEEOverview(Scene): def construct(self): # Create arrow pointing to the implicit 1 arrow = Arrow(start=mantissa[0][0].get_edge(UP) + UP * 0.1,

Plugin Documentation

end=mantissa[0][0].get_edge(DOWN), color=YELLOW, buff=0.1)

Agentic RAG (Router)

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

- Figure 4: Subfields of TheoremExplainBench under Computer Science, Chemistry, Mathematics, and Physics.

#### 2.3 LLM in Visualizations

Recent advancements in AI-driven visualization have enabled AI systems to generate structured visual content from textual descriptions (Li et al.,

- 2024). These models typically process textbased inputs and produce programmatic representations, which are then converted into visual outputs (Ritchie et al., 2023; Goswami et al., 2025). This approach has been applied across various domains, including scientific visualization (Yang et al., 2024b), data representation (Galimzyanov

- et al., 2024), and motion graphics (Zhang et al., 2023a). Efforts such as Drawing-Pandas (Galimzyanov et al., 2024) have introduced benchmarks for evaluating code-based plotting in Matplotlib and Seaborn. Follow-up works like MatPlotAgent(Yang et al., 2024b) demonstrated that agentic approaches outperform agentless methods in visualization generation, while PlotGen (Goswami
- et al., 2025) incorporated multimodal feedback for iterative refinement, further improving visualization quality. Our work is the first to explore AIdriven visualization for generating animated theorem explanations, seamlessly integrating step-bystep symbolic derivations with structured motion graphics, bridging the gap between mathematical reasoning and visual comprehension.

### 3 Method

#### 3.1 Task Definition

Model Input. The model receives a theorem along with a short description that provides context, which helps the model identify the theorem.

Model Output. The model is to output a video that combines animations, structured derivations, and voiceover narration to provide a multimodal and comprehensive explanation of the theorem. The video is expected to be longer than a minute, featuring long animations across different scenes, with narration guiding the viewer through step-by-step proofs and real-world applications.

#### 3.2 TheoremExplainAgent (TEA)

We develop TheoremExplainAgent (TEA), an agentic pipeline designed to automate the generation of videos using multiple specialized agents as shown in Figure 3. The process begins with the planner agent, which creates a high-level video plan according to the specified theorem. This plan consists of multiple scenes, each corresponding to a key segment of the resulting video. Once the initial plan is created, the planner agent refines the details of each scene, breaking them down into smaller components that define the specific visual elements, animations, and transitions needed. These detailed

Agent Easy Medium Hard Math Phys CS Chem Overall

GPT-4o 61.3% 57.5% 46.2% 61.7% 55.0% 58.3% 45.0% 55.0% GPT-4o + RAG 42.5% 57.5% 37.5% 70.0% 40.0% 41.7% 31.7% 45.8% Claude 3.5-Sonnet v1 2.5% 1.2% 2.5% 1.7% 1.7% 1.7% 3.3% 2.1% Claude 3.5-Sonnet v1 + RAG 18.8% 13.8% 11.2% 23.3% 10.0% 20.0% 5.0% 14.6% Gemini 2.0-Flash 20.0% 11.2% 12.5% 16.7% 8.3% 21.7% 11.7% 14.6% Gemini 2.0-Flash + RAG 23.8% 21.2% 16.2% 26.7% 15.0% 20.0% 20.0% 20.4%

- o3-mini (medium) 93.8% 91.2% 96.2% 95.0% 93.3% 93.3% 93.3% 93.8%

- o3-mini (medium) + RAG 83.8% 82.5% 80.0% 81.7% 90.0% 88.3% 68.3% 82.1%

- Table 1: Agent success rate in generating complete videos across different difficulty levels and subjects.

Agent Accuracy Visual Logical Element Visual Overall

and Depth Relevance Flow Layout Consistency Score

GPT-4o 0.79 0.79 0.89 0.59 0.87 0.78 GPT-4o + RAG 0.75 0.77 0.88 0.57 0.86 0.76 Claude 3.5-Sonnet v1 0.75 0.87 0.88 0.57 0.92 0.79 Claude 3.5-Sonnet v1 + RAG 0.67 0.79 0.69 0.65 0.87 0.71 Gemini 2.0 Flash 0.82 0.77 0.80 0.57 0.88 0.76 Gemini 2.0 Flash + RAG 0.79 0.75 0.84 0.58 0.87 0.76

- o3-mini (medium) 0.76 0.76 0.89 0.61 0.88 0.77

- o3-mini (medium) + RAG 0.75 0.75 0.88 0.61 0.88 0.76 Human-made Manim Videos 0.80 0.81 0.70 0.73 0.87 0.77

- Table 2: Performance of our proposed metrics on successfully generated long-form videos by the agents.

scene descriptions are then passed to the coding agent, which generates the corresponding Python code. The voiceover is also generated through a text-to-speech service. Finally, the Python scripts are executed to produce the final video, which reflects the narrative or instructional goals outlined in the video plan. If the generated Python code encounters an error, the coding agent will review the error and generate a revised version of the code. We set a maximum of N attempts where N = 5. If this limit is exceeded, we mark the generation as unsuccessful. We found that at N = 0, the success rate is extremely low while N = 5 achieves up to 90% success rate, as discussed in Table 3.

Coding Toolkit. We choose Manim (The Manim Community Developers, 2024) as the coding toolkit because it is a popular open-source Python library designed for creating mathematical animations and educational videos through code-driven visualizations. YouTube channels such as 3Blue1Brown (Sanderson, 2020) have demonstrated how Manim-made videos can convey complex mathematical concepts in an intuitive way. In our context, the coding agent translates each scene’s specifications into executable Manim scripts, which define objects such as text, shapes, graphs, or equations, along with their correspond-

ing animations, timings, and transitions.

Agentic Retrieval-Augmented Generation. To enhance code generation ability, we implemented a multifaceted retrieval-augmented generation (RAG) approach, leveraging the Manim documentation as the primary knowledge base. Unlike a single monolithic retrieval step, our agentic approach first classifies whether the theorems are suitable for using specific Manim plugins. Then it generates relevant queries at different stages of the video creation process: (1) during storyboard generation, to retrieve visual examples and related concepts; (2) during technical implementation, to fetch specific code snippets and usage patterns; and (3) during error correction, to diagnose issues and suggest solutions. These queries are cached to prevent redundant computations, and the agent dynamically selects the most relevant documents based on a relevance scoring threshold, ensuring efficient and precise retrieval.

#### 3.3 TheoremExplainBench (TEB)

We curate an evaluation dataset comprising 240 theorems from various disciplines, including Computer Science, Chemistry, Mathematics, and Physics. Each entry includes the theorem name and a contextual description, sourced from Open-

Category N = 0 N = 1 N = 2 N = 3 N = 4 N = 5

Difficulty: Easy 7% / 5% 51% / 33% 73% / 66% 86% / 71% 91% / 73% 93% / 73% Difficulty: Medium 0% / 5% 33% / 43% 75% / 66% 83% / 72% 88% / 76% 91% / 77% Difficulty: Hard 3% / 3% 46% / 35% 81% / 51% 90% / 68% 95% / 71% 96% / 73%

- Table 3: Combined cumulative theorem success rates (Baseline / RAG) for o3-mini (medium) with varying N attempts. The overall success rate significantly improves with more attempts, reaching its peak at N = 5.

Stax (Baraniuk, 2025) and LibreTexts (Larsen,

- 2025). To facilitate structured assessment, the theorems are categorized into three difficulty levels: Easy (high school level), Medium (undergraduate level), and Hard (graduate level), with 80 entries in each category. TheoremExplainBench (TEB) features 68 sub-fields that cover a wide range of domains as shown in Figure 4.

To fully define this novel problem, we propose a comprehensive evaluation metric applicable to both human-created and AI-generated explanatory videos, ensuring a standardized assessment across different content sources. Our metric evaluates videos across five key dimensions. The first three dimensions assess the factual correctness of explanations, while the last two dimensions evaluate the perceptual quality of the videos.

Accuracy and Depth. Evaluates whether the narration provides a precise and well-structured explanation of the theorem, offering both intuitive insights and rigorous justifications for why it holds.

Visual Relevance. Assesses whether the video frames effectively align with the theorem’s concepts and derivations, reinforcing the explanation through appropriate visual representations.

Logical Flow. Examines whether the video follows a clear and coherent structure, ensuring a logical progression that builds upon ideas effectively.

Element Layout. Evaluates whether visual elements are well-positioned and appropriately sized within the frame, avoiding unintended overlap and ensuring clarity in presentation.

Visual Consistency. Assesses whether the motions are smooth, and whether the visual style remains uniform across frames.

In our metric implementation, Accuracy & Depth and Logical Flow are assessed using textbased evaluation with GPT-4o (OpenAI, 2023). The text elements are extracted from video transcripts in SubRip (SRT) format. For Visual Relevance and Element Layout, we apply image processing techniques to identify key frames and use GPT-4o to assign scores for each dimension. To

evaluate motions in Visual Consistency, we utilize Gemini 2.0-Flash (DeepMind, 2025) to analyze chunked video segments. The overall score (ranging from 0 to 1) is then computed as the geometric mean of all dimensions. To ensure output stability, we employ greedy decoding (i.e., temperature = 0) in the LLM evaluations.

To validate the effectiveness of our evaluation metrics, we conducted a small-scale human study. We sampled 40 videos from our results, selecting 10 from each discipline in TheoremExplainBench. We then recruited 12 experienced STEM student annotators to participate in the study. The rating process followed the same five evaluation dimensions as our proposed metrics, with human raters selecting scores from [0, 0.5, 1]. To assess alignment between our metrics and human evaluations, we computed the Spearman correlation on the sampled subset. To ensure result reliability, we measured inter-rater agreement of 3 people using Krippendorff’s alpha (Krippendorff, 2011), which is more suitable than Fleiss’ Kappa (Fleiss and Cohen, 1973) due to the ordinal nature of the ratings. Additionally, to contextualize human performance, we sourced 10 human-made theorem explanation videos from YouTube for comparison.

### 4 Experimental Results

For the agent candidates in TheoremExplainAgent, we experimented with GPT-4o (OpenAI, 2023), Gemini 2.0 Flash (DeepMind, 2025), Claude 3.5 v1 (Anthropic, 2024), and o3-mini (OpenAI, 2025). Each candidate was used for both the planner agent and coding agent, ensuring consistency across configurations. We evaluated all agents across 240 theorems from TheoremExplainBench, comparing their performance under different setups. Our findings indicate that an agentless approach fails to generate videos longer than 20 seconds, whereas TheoremExplainAgent successfully produces videos of up to 10 minutes. Consequently, all experimental results presented in this paper are based on the agentic approach.

Spearman Krippendorff’s α

Accuracy and Depth 0.14 0.45 Visual Relevance 0.72 0.36 Logical Flow 0.16 0.56 Element Layout 0.42 0.31 Visual Consistency 0.17 0.36

- Table 4: Correlation on Metric-Human correlation (Spearman) and Inter-rater Agreement (Krippendorff’s alpha) for the five evaluation dimensions.

Table 1 reveals that the success rate in generating long-form theorem explanation videos varies significantly across difficulty levels and subjects. Overall, o3-mini consistently outperforms other models, maintaining high success rates across both easy and hard tasks, as well as across different STEM domains. In contrast, GPT-4o performs moderately well but show a declining success rate as complexity increases, suggesting difficulties in handling longer and more structured explanations. Gemini 2.0-Flash struggles the most, with notably lower success rates across all conditions. Across subjects, Mathematics tends to have the highest success rates, whereas Chemistry appear to be the most challenging domain. This observation may be attributed to the fact that complex objects in Chemistry, such as flask shapes and atoms, are more challenging to illustrate than simpler primitives in Mathematics, like triangles.

Given the successfully generated videos, we compiled Table 2 to present the metric results. Among the evaluated models, GPT-4o and o3-mini performed the best overall, both achieving strong scores across multiple dimensions. GPT-4o excelled in accuracy and depth, as well as logical flow, while o3-mini demonstrated the strongest performance in logical flow and a solid element layout. On the other hand, Gemini 2.0 Flash with RAG performed the weakest overall, struggling particularly with element layout and logical flow, indicating challenges in maintaining structured and visually coherent outputs. Human-made Manim videos, while scoring the similar overall among AI-generated results, achieved the highest visual relevance and element layout. This may be because AI-generated videos tend to exhibit minor issues like overlapping elements and misalignment, which can affect clarity and structure. Interestingly, human-made videos scored lower in logical flow. This may be due to the more natural and less structured narration in human explanations, which often

prioritize engagement over strict logical progression. In contrast, AI-generated videos tend to maintain a consistent logical structure, adhering closely to predefined formats. However, this rigidity may sometimes come at the cost of expressiveness and contextual adaptability, making human explanations feel more fluid and accessible despite their lower scores in formal evaluation metrics.

Our experiments with the RAG setup yielded mixed results, as shown in Table 1 and Table 2. While RAG was intended to enhance function understanding and streamline object construction, its effectiveness proved inconsistent. Although retrieval of documentation and code examples provided additional context, the results often misaligned with specific use cases. Many retrieved references were overly generic or lacked relevance, leading to incorrect function calls and suboptimal parameter choices. These findings are consistent with previous research highlighting the critical importance of retrieval quality. Poorly structured documentation and imprecise retrieval can significantly compromise the effectiveness of RAG-based approaches (Soman and Roychowdhury, 2024).

We also conducted an ablation study on the cumulative success rates of our o3-mini (medium) models by varying the retries value N ∈ {0, 1, 2, 3, 4, 5}. The “N-attempt success rate" is defined as the percentage of theorems for which all constituent scenes are successfully rendered. This metric evaluates the cumulative performance as we allow for more attempts to correct initial errors. Both the baseline and RAG-enhanced o3-mini models demonstrate a substantial increase in theorem success rates as N increases, indicating that multiple attempts effectively mitigate the impact of initial code generation failures. The detailed results are presented in Table 3.

#### 4.1 Correlation Study

From Table 4, we observe that our proposed metrics show strong alignment with human ratings in Visual Relevance and Element Layout, while demonstrating weaker correlations in Accuracy & Depth, Logical Flow, and Visual Consistency. This suggests that humans are particularly sensitive to visual aspects, such as spatial layouts, but may struggle with evaluating long-form text or audio-based content in detail. Visual Consistency appears to be more subjective, which may explain its relatively lower correlation with human ratings. Additionally, Accuracy & Depth and Logical Flow exhibits the

Theorem: 8-Connectivity Chain Code Example: Star shape

(Video Explanation) A chain code is a method ...

Human Expert Solution:

[Figure 49]

[Figure 50]

Assume each edge with only

[Figure 51]

one code, starting from top and go anti-clockwise edge by edge.

(Text Explanation) A chain code is a method used in image processing to represent the boundary of a shape in a compact and lossless manner. It works by tracing the contour of the shape and

[Figure 52]

3

[Figure 53]

5

- 4

7

- 5 1 7

4

encoding the direction of movement

1

[Figure 54]

[Figure 55]

between consecutive boundary pixels. The directions are typically represented using numbers… …

[Figure 56]

2

[Figure 57]

- 3
- 4
- 5 6

- 0
- 1

3

| | |
|---|---|
| | |

[Figure 58]

Chain Code of a star shape would be [0, 1, 2, 3, 4, 5, 6, 7, 0, 1].

7

[Figure 59]

[Figure 60]

[Figure 61]

Ans: [5,4,7,5,1,7,3,1,4,3]

[Figure 62]

[Figure 63]

[Figure 64]

Incorrect, but we don’t know why Model misunderstood the direction encodes

- Figure 5: Visualizations expose reasoning errors more clearly than text, making it easier to diagnose model mistakes.

weakest correlation with human judgments, likely due to differences in how LLM and humans assess coherence. Humans can tolerate informal flow, while LLMs may penalize it. On the other hand, human ratings across all dimensions show moderate inter-rater agreement, as indicated by Krippendorff’s alpha values. Notably, text-based dimensions achieve slightly higher agreement than visualbased ones, suggesting that textual evaluations are more consistently interpreted among raters.

#### 4.2 Interpretability Study

We found that visual explanations more effectively reveal reasoning errors than text, facilitating error diagnosis. From Figure 5, we observe that while the text-based explanation allows us to detect that the model’s answer is incorrect, it does not provide insight into why the mistake occurred. It seems the model understand the chain code theorem, but it applies it incorrectly. Such explanation is making it difficult to pinpoint the exact reasoning flaw. In contrast, the video-based explanation clearly exposes the model’s misunderstanding, as incorrect movement direction encodes and misplaced arrows reveal how the model misinterpreted the chain coding process. This demonstrates that visual explanations not only confirm incorrect reasoning but also uncover the underlying cause of errors, making them a more effective diagnostic tool for analyzing AI-generated outputs.

To quantitatively assess whether video explanations better highlight LLM reasoning problems, we designed a human study. 15 participants were first shown a textual explanation of a theorem containing a subtle reasoning flaw and asked to judge its correctness. They were then shown a video explanation of the same theorem, embodying the same reasoning flaw, and asked to re-evaluate. Partic-

ipants also rated the intuitiveness of both explanations on a scale of 1 to 5. Initially, all 15 participants judged the textual explanation as correct. After watching the video, 9 participants (60%) revised their judgment to "incorrect," successfully identifying the conceptual flaw that was only apparent through the visual narration. The average intuitiveness rating improved from 3.3 for the textual explanation to 3.9 for the video explanation. This suggests that multimodal explanations can be more effective in revealing and helping users understand reasoning errors.

#### 4.3 Error Analysis

We analyzed the error logs from unsuccessful runs in the TheoremExplainAgent video generation process and identified three primary failure categories. The most common issue was Manim code hallucinations, which accounted for the majority of failures. These errors involved nonexistent functions, modules, object properties, or image assets, as well as incorrect function signatures with invalid parameter types and numbers, reflecting a misunderstanding of the Manim API. The second major issue stemmed from LaTeX rendering errors, primarily due to syntax mistakes and improper handling of special characters in mathematical expressions. Lastly, general coding errors were observed, including missing imports, undefined variables, and computational mistakes in NumPy-based operations. These findings reveal key challenges across LLMs, underscoring the need for better code reliability and API understanding in AI-generated videos.

#### 4.4 Case Study

We included representative video outputs in Figure 7, demonstrating that TheoremExplainAgent is capable of generating high-quality exploratory

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

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- Figure 6: Video quality comparison by video frame between LTXVideo, Veo2 and TheoremExplainAgent, where t represents the proportion of the total video duration. t correspond to the time. The approximate number of T for LTXVideo, Veo2 and TheoremExplainAgent are 7 seconds, 8 seconds, and 5 minutes respectively.

videos. For example, in Mathematics, the model effectively visualizes concepts such as Riemann sums, using animated grids and function plots to illustrate integral approximations. In Chemistry, the system successfully explains the Octet Rule, leveraging atomic models to depict electron sharing and bonding interactions. In Physics, it generates electromagnetic wave simulations, showcasing wave propagation and spectral analysis. In Computer Science, it produces a clear demonstration of Run-Length Encoding, using side-by-side comparisons of raw and compressed data representations. Videos in Mathematics, Physics, and Computer Science typically show higher visual quality and coherence than those in Chemistry. One notable observation is that Chemistry-related visualizations often rely on simple geometric primitives to illustrate complex lab equipment and molecules, which can limit their clarity and effectiveness. Additionally, most of the generated videos exhibit minor element layout issues, such as overlapping texts, inconsistent sizes, or suboptimal object positioning, which slightly affects the overall presentation quality, as illustrated in Figure 8.

TheoremExplainBench. As illustrated in Figure 6, the resulting videos were frequently visually incoherent, often manifesting as random noise or lacking meaningful relation to the intended scientific content. These outputs also lacked the structured, pedagogical qualities necessary for effective theorem explanations. Moreover, these models do not include voiceover narration, an integral component of our multimodal output. This highlights that while text-to-video models can synthesize visual content from text, they lack the explicit reasoning capabilities required to generate domain-specific explanatory videos. In contrast, TheoremExplainAgent enables the creation of detailed, pedagogically sound videos that align with the theorems.

### 5 Conclusion

This paper introduces TheoremExplainAgent, a novel agentic approach for generating multimodal theorem explanations through structured video content. We demonstrates that integrating visual explanations significantly enhances the clarity and interpretability of theorem reasoning, surpassing text-based methods alone. We also present a benchmark spanning multiple disciplines with five automated evaluation metrics. Experiments reveal that agentic planning is crucial for producing long-form, coherent explanations, with o3-mini achieving the highest success rate and overall performance. However, challenges remain in visual element layout, emphasizing the need for improved spatial reasoning and refinement in AI-generated animations. Additionally, our findings underscore the importance of multimodal explanations in identifying reasoning flaws that text-based assessments often miss, reinforcing the role of structured visual communication in AI-driven theorem understanding.

#### 4.5 Text-to-Video Model Baselines

To access whether explicit reasoning capabilities provided by LLM-based agents are essential for generating coherent theorem explanations, we conducted a baseline comparison using non-LLMbased text-to-video models. Specifically, we examined the performance of the recent open-source model LTXVideo (HaCohen et al., 2024) and the closed-source model Veo2 (Veo-Team et al., 2024). For this comparison, we prompted each model with: “a Manim-style explanatory video explaining <theorem>” for 20 randomly selected theorems from

### 6 Limitations

While our approach demonstrates the potential of AI-generated multimodal theorem explanations, several limitations remain.

Visual Structuring Challenges. TheoremExplainAgent exhibits limitations in visual structuring, with issues such as misaligned text, overlapping shapes, and inconsistent sizes. These visual imperfections, though sometimes subtle, can distract from the intended educational value and hinder comprehension, particularly in complex topics like Chemistry. As mentioned in Section 4.4 and illustrated in Table 2, both human and automated evaluations identified shortcomings in element layout and visual consistency, highlighting the need for further refinement in visual design.

Inconsistent Retrieval-Augmented Generation. The reliance on retrieval-augmented generation (RAG) to support Manim code generation proved inconsistent in practice. As noted in Section 4.4 and shown in Table 5, retrieved code snippets were often irrelevant or overly generic, resulting in hallucinated or incorrect function calls and suboptimal parameter choices. Moreover, the use of RAG significantly increased token usage and inference time, raising concerns about scalability and cost efficiency.

Evaluation Metric Limitations. Our automated metrics for Accuracy & Depth (Spearman ρ = 0.14), Logical Flow (ρ = 0.16), and Visual Consistency (ρ = 0.17) show weak correlation with human ratings, while Visual Relevance (ρ = 0.72) and Element Layout (ρ = 0.42) achieve moderate to high agreement with good p-values (0.001 and 0.03 respectively) to show statistical significance. We acknowledge that our automated metrics have room for improvement, particularly for aspects like narrative coherence and the nuances of visual consistency. Current video understanding capabilities in Vision Language Models (VLMs) are still in early stages of development. Future work could explore fine-tuning specialized video understanding models (e.g., VideoScore (He et al., 2024b)) for assessing these types of tasks more accurately. We agree that visual consistency is part of the visual assessment and will refine the evaluation design in future iterations.

Code Generation Fragility. The system remains vulnerable to common coding errors, including La-

TeX rendering failures, incorrect function calls, and general Python errors such as undefined variables and missing imports. These issues were detailed in Section 4.3 and highlight the fragile nature of current code-generation approaches. Despite the implementation of a retry mechanism with up to five attempts, code reliability remain critical challenges.

### 7 Potential Risks

AI-generated explanations have the potential to mislead users if errors go undetected, leading to false confidence in incorrect reasoning. This poses a risk where unverified AI-generated content could propagate misconceptions or misinformation if widely disseminated without proper validation. Ensuring the accuracy and reliability of AI-generated explanations remains a critical challenge.

### 8 Artifacts

We experimented TheoremExplainAgent with GPT4o (OpenAI, 2023), Gemini 2.0 Flash (DeepMind, 2025), Claude 3.5 v1 (Anthropic, 2024), and o3mini (OpenAI, 2025). We are releasing the TheoremExplainBench on Huggingface dataset with MIT licence. It features 240 theorems across Computer Science, Physics, Chemistry and Math subjects.

### 9 Computational Experiments

All the experiments were conducted on a NVIDIA A100-SXM4-80GB GPU. Approximately 1500 US dollars were spent on API call for closed-source model experiments.

### 10 Acknowledgement

We express our gratitude to Votee AI for sponsoring API calls from closed-source models. We also thank Xueguang Ma, Dongfu Jiang, Zhi-Rui Tam, Chiu-Wai Yan, and Kelly Chiu for their insightful discussions.

### References

Talor Abramovich, Meet Udeshi, Minghao Shao, Kilian Lieret, Haoran Xi, Kimberly Milner, Sofija Jancheska, John Yang, Carlos E. Jimenez, Farshad Khorrami, Prashanth Krishnamurthy, Brendan DolanGavitt, Muhammad Shafique, Karthik Narasimhan, Ramesh Karri, and Ofir Press. 2024. Enigma: Enhanced interactive generative model agent for ctf challenges. Preprint, arXiv:2409.16165.

Aida Amini, Saadia Gabriel, Peter Lin, Rik KoncelKedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. Preprint, arXiv:1905.13319.

Anthropic. 2024. The claude 3 model family: Opus, sonnet, haiku. Online. Accessed: 2025-02-11.

Aristotle. 1901. Aristotle’s Posterior Analytics. B.H. Blackwell.

Richard Baraniuk. 2025. Openstax: Free textbooks online with no catch.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. 2022a. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. Preprint, arXiv:2105.14517.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023a. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Preprint, arXiv:2211.12588.

Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. 2023b. Theoremqa: A theorem-driven question answering dataset. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2022b. Finqa: A dataset of numerical reasoning over financial data. Preprint, arXiv:2109.00122.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

DeepMind. 2025. Gemini 2.0 flash. Online. Accessed: 2025-02-11.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang,

Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. 2024. Deepseek-v3 technical report. Preprint, arXiv:2412.19437.

Joseph L Fleiss and Jacob Cohen. 1973. The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability. Educational and psychological measurement, 33(3):613–619.

Timur Galimzyanov, Sergey Titov, Yaroslav Golubev, and Egor Bogomolov. 2024. Drawing pandas: A benchmark for llms in generating plotting code. Preprint, arXiv:2412.02764.

Gatekeep. 2024. Gatekeep ai: Start learning faster with personalized videos.

Gemini-Team, Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry, Lepikhin, Timothy Lillicrap, Jean baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Rohan Anil, Sebastian Borgeaud, Andrew Dai, Katie Millican, Ethan Dyer, Mia Glaese, Thibault Sottiaux, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, James Molloy, Jilin Chen, Michael Isard, Paul Barham, Tom Hennigan, Ross McIlroy, Melvin Johnson, Johan Schalkwyk, Eli Collins, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Clemens Meyer, Gregory Thornton, Zhen Yang, Henryk Michalewski, Zaheer Abbas, Nathan Schucher, Ankesh Anand, Richard

Ives, James Keeling, Karel Lenc, Salem Haykal, Siamak Shakeri, Pranav Shyam, Aakanksha Chowdhery, Roman Ring, Stephen Spencer, Eren Sezener, Luke Vilnis, Oscar Chang, Nobuyuki Morioka, George Tucker, Ce Zheng, Oliver Woodman, Nithya Attaluri, Tomas Kocisky, Evgenii Eltyshev, Xi Chen, Timothy Chung, Vittorio Selo, Siddhartha Brahma, Petko Georgiev, Ambrose Slone, Zhenkai Zhu, James Lottes, Siyuan Qiao, Ben Caine, Sebastian Riedel, Alex Tomala, Martin Chadwick, Juliette Love, Peter Choy, Sid Mittal, Neil Houlsby, Yunhao Tang, Matthew Lamm, Libin Bai, Qiao Zhang, Luheng He, Yong Cheng, Peter Humphreys, Yujia Li, Sergey Brin, Albin Cassirer, Yingjie Miao, Lukas Zilka, Taylor Tobin, Kelvin Xu, Lev Proleev, Daniel Sohn, Alberto Magni, Lisa Anne Hendricks, Isabel Gao, Santiago Ontanon, Oskar Bunyan, Nathan Byrd, Abhanshu Sharma, Biao Zhang, Mario Pinto, Rishika Sinha, Harsh Mehta, Dawei Jia, Sergi Caelles, Albert Webson, Alex Morris, Becca Roelofs, Yifan Ding, Robin Strudel, Xuehan Xiong, Marvin Ritter, Mostafa Dehghani, Rahma Chaabouni, Abhijit Karmarkar, Guangda Lai, Fabian Mentzer, Bibo Xu, YaGuang Li, Yujing Zhang, Tom Le Paine, Alex Goldin, Behnam Neyshabur, Kate Baumli, Anselm Levskaya, Michael Laskin, Wenhao Jia, Jack W. Rae, Kefan Xiao, Antoine He, Skye Giordano, Lakshman Yagati, Jean-Baptiste Lespiau, Paul Natsev, Sanjay Ganapathy, Fangyu Liu, Danilo Martins, Nanxin Chen, Yunhan Xu, Megan Barnes, Rhys May, Arpi Vezer, Junhyuk Oh, Ken Franko, Sophie Bridgers, Ruizhe Zhao, Boxi Wu, Basil Mustafa, Sean Sechrist, Emilio Parisotto, Thanumalayan Sankaranarayana Pillai, Chris Larkin, Chenjie Gu, Christina Sorokin, Maxim Krikun, Alexey Guseynov, Jessica Landon, Romina Datta, Alexander Pritzel, Phoebe Thacker, Fan Yang, Kevin Hui, Anja Hauth, Chih-Kuan Yeh, David Barker, Justin Mao-Jones, Sophia Austin, Hannah Sheahan, Parker Schuh, James Svensson, Rohan Jain, Vinay Ramasesh, Anton Briukhov, DaWoon Chung, Tamara von Glehn, Christina Butterfield, Priya Jhakra, Matthew Wiethoff, Justin Frye, Jordan Grimstad, Beer Changpinyo, Charline Le Lan, Anna Bortsova, Yonghui Wu, Paul Voigtlaender, Tara Sainath, Shane Gu, Charlotte Smith, Will Hawkins, Kris Cao, James Besley, Srivatsan Srinivasan, Mark Omernick, Colin Gaffney, Gabriela Surita, Ryan Burnell, Bogdan Damoc, Junwhan Ahn, Andrew Brock, Mantas Pajarskas, Anastasia Petrushkina, Seb Noury, Lorenzo Blanco, Kevin Swersky, Arun Ahuja, Thi Avrahami, Vedant Misra, Raoul de Liedekerke, Mariko Iinuma, Alex Polozov, Sarah York, George van den Driessche, Paul Michel, Justin Chiu, Rory Blevins, Zach Gleicher, Adrià Recasens, Alban Rrustemi, Elena Gribovskaya, Aurko Roy, Wiktor Gworek, Sébastien M. R. Arnold, Lisa Lee, James Lee-Thorp, Marcello Maggioni, Enrique Piqueras, Kartikeya Badola, Sharad Vikram, Lucas Gonzalez, Anirudh Baddepudi, Evan Senter, Jacob Devlin, James Qin, Michael Azzam, Maja Trebacz, Martin Polacek, Kashyap Krishnakumar, Shuo yiin Chang, Matthew Tung, Ivo Penchev, Rishabh Joshi, Kate Olszewska, Carrie Muir, Mateo Wirth, Ale Jakse Hartman, Josh Newlan, Sheleem Kashem,

Vijay Bolina, Elahe Dabir, Joost van Amersfoort, Zafarali Ahmed, James Cobon-Kerr, Aishwarya Kamath, Arnar Mar Hrafnkelsson, Le Hou, Ian Mackinnon, Alexandre Frechette, Eric Noland, Xiance Si, Emanuel Taropa, Dong Li, Phil Crone, Anmol Gulati, Sébastien Cevey, Jonas Adler, Ada Ma, David Silver, Simon Tokumine, Richard Powell, Stephan Lee, Kiran Vodrahalli, Samer Hassan, Diana Mincu, Antoine Yang, Nir Levine, Jenny Brennan, Mingqiu Wang, Sarah Hodkinson, Jeffrey Zhao, Josh Lipschultz, Aedan Pope, Michael B. Chang, Cheng Li, Laurent El Shafey, Michela Paganini, Sholto Douglas, Bernd Bohnet, Fabio Pardo, Seth Odoom, Mihaela Rosca, Cicero Nogueira dos Santos, Kedar Soparkar, Arthur Guez, Tom Hudson, Steven Hansen, Chulayuth Asawaroengchai, Ravi Addanki, Tianhe Yu, Wojciech Stokowiec, Mina Khan, Justin Gilmer, Jaehoon Lee, Carrie Grimes Bostock, Keran Rong, Jonathan Caton, Pedram Pejman, Filip Pavetic, Geoff Brown, Vivek Sharma, Mario Luˇci´c, Rajkumar Samuel, Josip Djolonga, Amol Mandhane, Lars Lowe Sjösund, Elena Buchatskaya, Elspeth White, Natalie Clay, Jiepu Jiang, Hyeontaek Lim, Ross Hemsley, Zeyncep Cankara, Jane Labanowski, Nicola De Cao, David Steiner, Sayed Hadi Hashemi, Jacob Austin, Anita Gergely, Tim Blyth, Joe Stanton, Kaushik Shivakumar, Aditya Siddhant, Anders Andreassen, Carlos Araya, Nikhil Sethi, Rakesh Shivanna, Steven Hand, Ankur Bapna, Ali Khodaei, Antoine Miech, Garrett Tanzer, Andy Swing, Shantanu Thakoor, Lora Aroyo, Zhufeng Pan, Zachary Nado, Jakub Sygnowski, Stephanie Winkler, Dian Yu, Mohammad Saleh, Loren Maggiore, Yamini Bansal, Xavier Garcia, Mehran Kazemi, Piyush Patil, Ishita Dasgupta, Iain Barr, Minh Giang, Thais Kagohara, Ivo Danihelka, Amit Marathe, Vladimir Feinberg, Mohamed Elhawaty, Nimesh Ghelani, Dan Horgan, Helen Miller, Lexi Walker, Richard Tanburn, Mukarram Tariq, Disha Shrivastava, Fei Xia, Qingze Wang, ChungCheng Chiu, Zoe Ashwood, Khuslen Baatarsukh, Sina Samangooei, Raphaël Lopez Kaufman, Fred Alcober, Axel Stjerngren, Paul Komarek, Katerina Tsihlas, Anudhyan Boral, Ramona Comanescu, Jeremy Chen, Ruibo Liu, Chris Welty, Dawn Bloxwich, Charlie Chen, Yanhua Sun, Fangxiaoyu Feng, Matthew Mauger, Xerxes Dotiwalla, Vincent Hellendoorn, Michael Sharman, Ivy Zheng, Krishna Haridasan, Gabe Barth-Maron, Craig Swanson, Dominika Rogozi´nska, Alek Andreev, Paul Kishan Rubenstein, Ruoxin Sang, Dan Hurt, Gamaleldin Elsayed, Renshen Wang, Dave Lacey, Anastasija Ili´c, Yao Zhao, Adam Iwanicki, Alejandro Lince, Alexander Chen, Christina Lyu, Carl Lebsack, Jordan Griffith, Meenu Gaba, Paramjit Sandhu, Phil Chen, Anna Koop, Ravi Rajwar, Soheil Hassas Yeganeh, Solomon Chang, Rui Zhu, Soroush Radpour, Elnaz Davoodi, Ving Ian Lei, Yang Xu, Daniel Toyama, Constant Segal, Martin Wicke, Hanzhao Lin, Anna Bulanova, Adrià Puigdomènech Badia, Nemanja Raki´cevi´c, Pablo Sprechmann, Angelos Filos, Shaobo Hou, Víctor Campos, Nora Kassner, Devendra Sachan, Meire Fortunato, Chimezie Iwuanyanwu, Vitaly Nikolaev, Balaji Lakshminarayanan, Sadegh Jazayeri, Mani Varadarajan, Chetan Tekur, Doug Fritz, Misha Khalman, David

Reitter, Kingshuk Dasgupta, Shourya Sarcar, Tina Ornduff, Javier Snaider, Fantine Huot, Johnson Jia, Rupert Kemp, Nejc Trdin, Anitha Vijayakumar, Lucy Kim, Christof Angermueller, Li Lao, Tianqi Liu, Haibin Zhang, David Engel, Somer Greene, Anaïs White, Jessica Austin, Lilly Taylor, Shereen Ashraf, Dangyi Liu, Maria Georgaki, Irene Cai, Yana Kulizhskaya, Sonam Goenka, Brennan Saeta, Ying Xu, Christian Frank, Dario de Cesare, Brona Robenek, Harry Richardson, Mahmoud Alnahlawi, Christopher Yew, Priya Ponnapalli, Marco Tagliasacchi, Alex Korchemniy, Yelin Kim, Dinghua Li, Bill Rosgen, Kyle Levin, Jeremy Wiesner, Praseem Banzal, Praveen Srinivasan, Hongkun Yu, Ça˘glar Ünlü, David Reid, Zora Tung, Daniel Finchelstein, Ravin Kumar, Andre Elisseeff, Jin Huang, Ming Zhang, Ricardo Aguilar, Mai Giménez, Jiawei Xia, Olivier Dousse, Willi Gierke, Damion Yates, Komal Jalan, Lu Li, Eri Latorre-Chimoto, Duc Dung Nguyen, Ken Durden, Praveen Kallakuri, Yaxin Liu, Matthew Johnson, Tomy Tsai, Alice Talbert, Jasmine Liu, Alexander Neitz, Chen Elkind, Marco Selvi, Mimi Jasarevic, Livio Baldini Soares, Albert Cui, Pidong Wang, Alek Wenjiao Wang, Xinyu Ye, Krystal Kallarackal, Lucia Loher, Hoi Lam, Josef Broder, Dan HoltmannRice, Nina Martin, Bramandia Ramadhana, Mrinal Shukla, Sujoy Basu, Abhi Mohan, Nick Fernando, Noah Fiedel, Kim Paterson, Hui Li, Ankush Garg, Jane Park, DongHyun Choi, Diane Wu, Sankalp Singh, Zhishuai Zhang, Amir Globerson, Lily Yu, John Carpenter, Félix de Chaumont Quitry, Carey Radebaugh, Chu-Cheng Lin, Alex Tudor, Prakash Shroff, Drew Garmon, Dayou Du, Neera Vats, Han Lu, Shariq Iqbal, Alex Yakubovich, Nilesh Tripuraneni, James Manyika, Haroon Qureshi, Nan Hua, Christel Ngani, Maria Abi Raad, Hannah Forbes, Jeff Stanway, Mukund Sundararajan, Victor Ungureanu, Colton Bishop, Yunjie Li, Balaji Venkatraman, Bo Li, Chloe Thornton, Salvatore Scellato, Nishesh Gupta, Yicheng Wang, Ian Tenney, Xihui Wu, Ashish Shenoy, Gabriel Carvajal, Diana Gage Wright, Ben Bariach, Zhuyun Xiao, Peter Hawkins, Sid Dalmia, Clement Farabet, Pedro Valenzuela, Quan Yuan, Ananth Agarwal, Mia Chen, Wooyeol Kim, Brice Hulse, Nandita Dukkipati, Adam Paszke, Andrew Bolt, Kiam Choo, Jennifer Beattie, Jennifer Prendki, Harsha Vashisht, Rebeca SantamariaFernandez, Luis C. Cobo, Jarek Wilkiewicz, David Madras, Ali Elqursh, Grant Uy, Kevin Ramirez, Matt Harvey, Tyler Liechty, Heiga Zen, Jeff Seibert, Clara Huiyi Hu, Andrey Khorlin, Maigo Le, Asaf Aharoni, Megan Li, Lily Wang, Sandeep Kumar, Norman Casagrande, Jay Hoover, Dalia El Badawy, David Soergel, Denis Vnukov, Matt Miecnikowski, Jiri Simsa, Praveen Kumar, Thibault Sellam, Daniel Vlasic, Samira Daruki, Nir Shabat, John Zhang, Guolong Su, Jiageng Zhang, Jeremiah Liu, Yi Sun, Evan Palmer, Alireza Ghaffarkhah, Xi Xiong, Victor Cotruta, Michael Fink, Lucas Dixon, Ashwin Sreevatsa, Adrian Goedeckemeyer, Alek Dimitriev, Mohsen Jafari, Remi Crocker, Nicholas FitzGerald, Aviral Kumar, Sanjay Ghemawat, Ivan Philips, Frederick Liu, Yannie Liang, Rachel Sterneck, Alena Repina, Marcus Wu, Laura Knight, Marin Georgiev,

Hyo Lee, Harry Askham, Abhishek Chakladar, Annie Louis, Carl Crous, Hardie Cate, Dessie Petrova, Michael Quinn, Denese Owusu-Afriyie, Achintya Singhal, Nan Wei, Solomon Kim, Damien Vincent, Milad Nasr, Christopher A. Choquette-Choo, Reiko Tojo, Shawn Lu, Diego de Las Casas, Yuchung Cheng, Tolga Bolukbasi, Katherine Lee, Saaber Fatehi, Rajagopal Ananthanarayanan, Miteyan Patel, Charbel Kaed, Jing Li, Shreyas Rammohan Belle, Zhe Chen, Jaclyn Konzelmann, Siim Põder, Roopal Garg, Vinod Koverkathu, Adam Brown, Chris Dyer, Rosanne Liu, Azade Nova, Jun Xu, Alanna Walton, Alicia Parrish, Mark Epstein, Sara McCarthy, Slav Petrov, Demis Hassabis, Koray Kavukcuoglu, Jeffrey Dean, and Oriol Vinyals. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Preprint, arXiv:2403.05530.

GenerativeManim. 2024. Generative manim: Ai-driven animations for mathematics and education.

Kanika Goswami, Puneet Mathur, Ryan Rossi, and Franck Dernoncourt. 2025. Plotgen: Multi-agent llm-based scientific data visualization via multimodal feedback. Preprint, arXiv:2502.00988.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. 2024. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103.

Liu He, Yizhi Song, Hejun Huang, Daniel Aliaga, and Xin Zhou. 2024a. Kubrick: Multimodal agent collaborations for synthetic video generation. Preprint, arXiv:2408.10453.

Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, Kai Wang, Quy Duc Do, Yuansheng Ni, Bohan Lyu, Yaswanth Narsupalli, Rongqi Fan, Zhiheng Lyu, Yuchen Lin, and Wenhu Chen. 2024b. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. Preprint, arXiv:2406.15252.

Alec Helbling, Duen Horng, and Chau. 2023. Manimml: Communicating machine learning architectures with animation. Preprint, arXiv:2306.17108.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. NeurIPS.

Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. 2023. Tifa: Accurate and interpretable text-toimage faithfulness evaluation with question answering. arXiv preprint arXiv:2303.11897.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R

Narasimhan. 2024. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations.

Aryan Keluskar, Amrita Bhattacharjee, and Huan Liu. 2024. Do llms understand ambiguity in text? a case study in open-world question answering. Preprint, arXiv:2411.12395.

Klaus Krippendorff. 2011. Computing krippendorff’s alpha-reliability.

Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. 2023. Viescore: Towards explainable metrics for conditional image synthesis evaluation. Preprint, arXiv:2312.14867.

Delmar Larsen. 2025. Libretexts: The future is open.

Guozheng Li, Xinyu Wang, Gerile Aodeng, Shunyuan Zheng, Yu Zhang, Chuangxin Ou, Song Wang, and Chi Harold Liu. 2024. Visualization generation with large language models: An evaluation. Preprint, arXiv:2401.11255.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. 2024. The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS).

Fabio Missagia. 2025. Manim DSA.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2023. Codegen: An open large language model for code with multi-turn program synthesis. Preprint, arXiv:2203.13474.

OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774.

OpenAI. 2025. Openai o3 mini. Online. Accessed: 2025-02-11.

Pouya Pezeshkpour and Estevam Hruschka. 2023. Large language models sensitivity to the order of options in multiple-choice questions. Preprint, arXiv:2308.11483.

Daniel Ritchie, Paul Guerrero, R. Kenny Jones, Niloy J. Mitra, Adriana Schulz, Karl D. D. Willis, and Jiajun Wu. 2023. Neurosymbolic models for computer graphics. Preprint, arXiv:2304.10320.

Stuart Russell and Peter Norvig. 2020. Artificial Intelligence: A Modern Approach (4th Edition). Pearson.

G. [3Blue1Brown] Sanderson. 2020. Group theory, abstraction, and the 196,883-dimensional monster. YouTube.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. 2025. Agent laboratory: Using llm agents as research assistants. Preprint, arXiv:2501.04227.

Krish Shah, Chris Abey, and Hargun Mujral. 2024. 3brown1blue: Ai-generated educational videos with manim.

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. 2024. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers. Preprint, arXiv:2409.04109.

Sumit Soman and Sujoy Roychowdhury. 2024. Observations on building rag systems for technical documents. Preprint, arXiv:2404.00657.

The Manim Community Developers. 2024. Manim – Mathematical Animation Framework.

Veo-Team, :, Agrim Gupta, Ali Razavi, Andeep Toor, Ankush Gupta, Dumitru Erhan, Eleni Shaw, Eric Lau, Frank Belletti, Gabe Barth-Maron, Gregory Shaw, Hakan Erdogan, Hakim Sidahmed, Henna Nandwani, Hernan Moraldo, Hyunjik Kim, Irina Blok, Jeff Donahue, José Lezama, Kory Mathewson, Kurtis David, Matthieu Kim Lorrain, Marc van Zee, Medhini Narasimhan, Miaosen Wang, Mohammad Babaeizadeh, Nelly Papalampidi, Nick Pezzotti, Nilpa Jha, Parker Barnes, Pieter-Jan Kindermans, Rachel Hornung, Ruben Villegas, Ryan Poplin, Salah Zaiem, Sander Dieleman, Sayna Ebrahimi, Scott Wisdom, Serena Zhang, Shlomi Fruchter, Signe Nørly, Weizhe Hua, Xinchen Yan, Yuqing Du, and Yutian Chen. 2024. Veo 2.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. Preprint, arXiv:2201.11903.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. 2024. Osworld: Benchmarking multimodal agents for openended tasks in real computer environments. Preprint, arXiv:2404.07972.

John Yang, Carlos E. Jimenez, Alex L. Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighoff, Gabriel Synnaeve, Karthik R. Narasimhan, Diyi Yang, Sida I. Wang, and Ofir Press. 2024a. Swe-bench multimodal: Do ai systems generalize to visual software domains? Preprint, arXiv:2410.03859.

Zhiyu Yang, Zihan Zhou, Shuo Wang, Xin Cong, Xu Han, Yukun Yan, Zhenghao Liu, Zhixing Tan, Pengyuan Liu, Dong Yu, Zhiyuan Liu, Xiaodong Shi, and Maosong Sun. 2024b. Matplotagent: Method and evaluation for llm-based agentic scientific data visualization. Preprint, arXiv:2402.11453.

Sharon Zhang, Jiaju Ma, Jiajun Wu, Daniel Ritchie, and Maneesh Agrawala. 2023a. Editing motion graphics video via motion vectorization and transformation. ACM Trans. Graph.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2023b. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923.

Ziyin Zhang, Zhaokun Jiang, Lizhen Xu, Hongkun Hao, and Rui Wang. 2024. Multiple-choice questions are efficient and robust llm evaluators. Preprint, arXiv:2405.11966.

Zihao Zhou, Maizhen Ning, Qiufeng Wang, Jie Yao, Wei Wang, Xiaowei Huang, and Kaizhu Huang. 2023. Learning by analogy: Diverse questions generation in math word problem. Preprint, arXiv:2306.09064.

Math: Riemann Sums

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Chemistry: Octet Rule

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Physics: Electromagnetic Waves

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Computer Science: Run-Length Encoding

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Figure 7: We show the high-quality videos generated by TheoremExplainAgent,across the four STEM domains.

Math: Integration By Substitution

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Computer Science: K Means Clustering

[Figure 126]

[Figure 127]

| |
|---|

[Figure 128]

[Figure 129]

| |
|---|

| |
|---|

| |
|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Figure 8: We show the poorly generated videos from TheoremExplainAgent, zooming in the artifacts.

[Figure 134]

Figure 9: Comparison on a scene of a high quality animation and a low quality animation.

[Figure 135]

[Figure 136]

Figure 10: The user interface of our annoatation website.

### A Gallery

In Figure 7 we present the high-quality videos generated by TheoremExplainAgent across four STEM domains. The images are extracted from different scenes in the videos, showing the consistency of the topic. In Figure 8 we present the poorly generated videos from TheoremExplainAgent and examine their artifacts. In Figure 9 we compare a high quality animation and a low quality animation, and how they were rated with our proposed metric.

### B Runtime Statistics

We report the runtime and cost statistics in Table 5, assuming 4 fixed codes and 7 scenes per video, we evaluate the cost, inference time, and latency of different language models, and find that the Claude 3.5-Sonnet v1 model has the longest inference time (2240-2380s), while the Gemini 2.0-Flash and GPT4o are the fastest (around 1120s).The RAG integration increases the number of input tokens significantly. RAG integration significantly increases the number of input tokens, with Claude 3.5-Sonnet v1 + RAG being the most used (1,050,000). Output tokens are less variable, with the o3-mini model generating the most tokens (154,000). The Gemini 2.0-Flash model is the most cost-effective ($0.10$0.16), while the Claude 3.5-Sonnet v1 + RAG is the most expensive ($4.67).

### C Preference Study

To complement our automated metrics, we conducted a human evaluation study. 18 annotators were asked to rank videos on 3 topics generated by three different models (o3-mini, Gemini 2.0 Flash, and GPT-4o) for the same set of theorems, without revealing the model names. Annotators evaluated the videos based on clarity and visual appeal. The results, summarized in Table 6, indicate that videos generated by Gemini 2.0 Flash were most frequently ranked highest for both clarity and visual appeal. o3-mini generated videos were preferred over those from GPT-4o. This human preference study provides complementary insights into the perceived quality of the generated videos.

### D Implementation Details D.1 Human Annotation Process

We recruited 12 student volunteers in our annotation process for the human metric. We explained to the annotators that their annotations were to be

used in our study only and would not be released publicly.

We show the user interface of our annotation website in Figure 10, including the instructions presented to our annotators. We supplement each of the dimensions with guiding questions to clarify what the annotators should score.

#### D.2 Technical Implementation

To aid reproducibility, we provide the following key implementation details. Manim Versions:

- • Manim Community Edition version 0.18.1,
- • ManimPango 0.6.0,
- • manim-physics 0.4.0,
- • manim-ml (Helbling et al., 2023) 0.0.24,
- • manim-chemistry 0.4.4,
- • manim-dsa (Missagia, 2025) 0.2.0,
- • manim-circuit 0.0.3

RAG System: We utilize ChromaDB as the vector store. Documentation (Markdown and Python files from Manim core and specified plugins) is chunked using Langchain’s RecursiveCharacterTextSplitter with default settings for the respective languages. The embedding model used is text-embedding-005 from Google Vertex AI.

RAG Retrieval Threshold: A relevance score threshold of 0.5 is used during similarity search in both core and plugin vector stores. We retrieve k=2 documents per query by default.

#### D.3 Prompt Templates

We adapt Chain-of-Thoughts (CoT) (Wei et al., 2023) and Program-of-Thoughts (PoT) (Chen et al., 2023a) when we design the prompt for TheoremExplainAgent. We present our prompts templates in the end of the Appendix.

#### D.4 Computational Resources and Costs

The primary LLM computations were performed using closed-source model APIs. The average number of tokens, cost, and inference time per video generation are detailed in Table 5. Local Manim rendering does not require a GPU for the types of animations generated.

### E Potentials for Future Research

Recent community efforts (Shah et al., 2024; Gatekeep, 2024; GenerativeManim, 2024) have explored AI-driven Manim-based video generation for educational purposes. However, no scientific studies have systematically evaluated the effectiveness and robustness of these approaches. Our work introduces a novel agentic framework for generating multimodal theorem explanations and demonstrates that AI-generated videos can achieve performance comparable to human-made content, although the robustness is still limited. Nevertheless, further research is needed to assess their impact on AI’s reasoning capabilities, visualization quality, and learning outcomes. Future directions include establishing benchmarks for AI-generated educational videos (within EdTech), integrating interactive elements to enhance engagement (within HCI/Visualization), and refining evaluation metrics to assess LLMs’ multimodal explanation abilities (within NLP).

### F User Feedback on Educational Use

We conducted a small-scale user study with 15 students and 8 teachers at universities to gather feedback on the educational clarity and engagement of AI-generated videos on easy topics. Participants rated videos on two topics (Bubble Sort and Hamming Distance) on a scale of 1 to 5 for clarity and engagement. As shown in Table 7, the mean scores for clarity and engagement were high. Over 90% of participants expressed interest in using such videos in a classroom setting, although the majority also believed there is room for content improvement.

### G Definition of Agentic

Our use of the term “agentic" follows the definition (Russell and Norvig, 2020): “An agent is anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators." In the context of TheoremExplainAgent, the language model acts as an agent that perceives inputs (such as the theorem description and feedback from code execution errors) and acts upon its environment by generating Manim code to create video scenes.

Agent Input Tokens Output Tokens Cost(USD) Time(s)

GPT-4o 350000 84000 1.71 1120 GPT-4o + RAG 840000 84000 2.94 1260 Claude 3.5-Sonnet v1 350000 91000 2.42 2240 Claude 3.5-Sonnet v1 + RAG 1050000 101500 4.67 2380 Gemini 2.0-Flash 595000 119000 0.1 1120 Gemini 2.0-Flash + RAG 1120000 119000 0.16 1260

- o3-mini (medium) 434000 154000 1.16 1680

- o3-mini (medium) + RAG 945000 154000 1.72 1820

- Table 5: Average output tokens, cost, and inference time for TheoremExplainAgent generating one full video.

Model Agent Clarity Top-Rank % Visual Top-Rank %

Gemini-2.0-Flash (Gemini-Team et al., 2024) 70.6% 61.8% o3-mini (OpenAI, 2025) 20.6% 23.5% GPT-4o (OpenAI, 2023) 8.8% 14.7%

- Table 6: Preference Study: Percentage of times each model was ranked highest by human annotators for video clarity and visual appeal.

Metric Topic 1: Bubble Sort Topic 2: Hamming Distance

Clarity (Mean ± SD) 4.13 ± 0.87 4.35 ± 0.83 Engagement (Mean ± SD) 3.57 ± 1.16 3.78 ± 1.09 Would recommend for classroom use 13% 35% Would recommend for classroom use if video content improved 87% 61% Would not recommend for classroom use 0% 4%

Table 7: User Feedback on Educational Usefulness.

Scene Plan Generation Prompt Template

You are an expert in video production, instructional design, and {topic}. Please design a highquality video to provide in-depth explanation on {topic}.

Video Overview: Topic: {topic} Description: {description}

Scene Breakdown: Plan individual scenes. For each scene please provide the following:

- • Scene Title: Short, descriptive title (2-5 words).
- • Scene Purpose: Objective of this scene. How does it connect to previous scenes?
- • Scene Description: Detailed description of scene content.
- • Scene Layout: Detailed description of the spatial layout concept. Consider safe area margins and minimum spacing between objects.

Please generate the scene plan for the video in the following format: ...

Code Generation Prompt Template

You are an expert Manim (Community Edition) developer. Generate executable Manim code implementing animations as specified, strictly adhering to the provided Manim documentation context, technical implementation plan, animation and narration plan, and all defined spatial constraints.

Think of reusable animation components for a clean, modular, and maintainable library, prioritizing code structure and best practices as demonstrated in the Manim documentation context. Throughout code generation, rigorously validate all spatial positioning and animations against the defined safe area margins and minimum spacing constraints. If any potential constraint violation is detected, generate a comment in the code highlighting the issue for manual review and correction.

Input Context:

... Code Generation Guidelines:

...

Code Fixing Prompt Template

You are an expert Manim developer specializing in debugging and error resolution. Based on the provided implementation plan and Manim code, analyze the error message to provide a comprehensive fix and explanation.

Implementation Plan: {implementation_plan} Manim Code: {manim_code} Error Message: {error_message} Requirements:

- 1. Provide complete error analysis with specific line numbers where possible.
- 2. Include exact instructions for every code change.
- 3. Explain why the error occurred in plain language.
- 4. ...

Evaluation Prompt Template

You are a specialist in evaluating theorem explanation videos, known for giving clear and objective feedback. You will be given the transcript of a video. Your task is to evaluate and score the content of the video in several dimensions.

Evaluation Criteria:

- 1. Accuracy and Depth

- • Does the narration explain the theorem accurately?
- • Does the video provide intuitive and/or rigorous explanations for why the theorem holds?

- 2. Logical Flow

- • Does the video follow a clear and logical structure?
- • Does the video present a coherent buildup of ideas?

Scoring Instructions: Conduct a comprehensive evaluation and score each dimension from 0 to 1: (Score Descriptions)

You are tasked with analyzing and scoring a frame taken from a theorem explanation video. Note that you may not have the context of the video, so the captured frame may be a frame where some motion of visual elements is taking place. Your job is to assign a score from 1 to 5 for each criterion. Please provide a brief justification for your scores.

Evaluation Criteria:

- 1. Visual Relevance

• Does the video frame align with the theorem’s concepts and derivations?

- 2. Element Layout

- • Are the visual elements well-placed and appropriately sized within the frame?
- • Are the visual elements free of unintentional overlap?
- • Is the visual information conveyed in the frame clear and easy to understand?

...

You are tasked with analyzing and scoring a chunk of a theorem explanation video. Note that you may not have the full context of the video. Your job is to assign a score from 1 to 5 for each criterion. Please provide a brief justification for your scores.

Evaluation Criteria:

1. Visual Consistency

- • Does the visual style remain consistent across frames?
- • Are the motions and transitions smooth?

...

