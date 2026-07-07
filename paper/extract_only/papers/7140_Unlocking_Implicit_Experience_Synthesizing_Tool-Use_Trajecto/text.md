# arXiv:2601.10355v1[cs.CL]15Jan2026

## Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text

Zhihao Xu12,∗ Rumei Li2∗, Jiahuan Li2, Rongxiang Weng2†, Jingang Wang2†, Xunliang Cai2, Xiting Wang1 1Renmin University of China, 2Meituan, China zhihaoxu@ruc.edu.cn

#### Abstract

Enabling Large Language Models (LLMs) to effectively utilize tools in multi-turn interactions is essential for building capable autonomous agents. However, acquiring diverse and realistic multi-turn tool-use data remains a significant challenge. In this work, we propose a novel text-based paradigm. We observe that textual corpora naturally contain rich, multi-step problem-solving experiences, which can serve as an untapped, scalable, and authentic data source for multi-turn tool-use tasks. Based on this insight, we introduce GEM, a data synthesis pipeline that enables the generation and extraction of multi-turn tool-use trajectories from text corpora through a four-stage process: relevance filtering, workflow & tool extraction, trajectory grounding, and complexity refinement. To reduce the computational cost, we further train a specialized Trajectory Synthesizer via supervised fine-tuning. This model distills the complex generation pipeline into an efficient, end-to-end trajectory generator. Experiments demonstrate that our GEM-32B achieve a 16.5% improvement on the BFCL V3 Multi-turn benchmark. Our models partially surpass the performance of models trained on τbench (Airline and Retail) in-domain data, highlighting the superior generalization capability derived from our text-based synthesis paradigm. Notably, our Trajectory Synthesizer matches the quality of the full pipeline while significantly reducing inference latency and costs.

#### 1 Introduction

The pursuit of Artificial General Intelligence (AGI) relies on the development of autonomous agents capable of perceiving, reasoning, and acting in complex real-world environments [Team et al., 2025a, Liu et al., 2025, Team et al., 2025b]. Such agents are required to dynamically utilize diverse tools to extend their capabilities and accomplish multi-step tasks. While large language models (LLMs) have demonstrated impressive tool-use capabilities, they still struggle in realistic multi-turn interactions, particularly when faced with ambiguous instructions, long-context dependencies, and unexpected errors [Patil et al.]. These limitations constrain their practical application in agent-based systems [Yao et al., 2024, Barres et al., 2025].

The primary bottleneck in training autonomous agents lies in the scarcity of high-quality, multi-turn tool-use trajectories, which are rarely found in real-world scenarios. To overcome this data scarcity, current research favors a tool-centered simulation paradigm (Fig. 1). These methods typically rely on predefined API sets to synthesize user tasks and simulate interactions [Qin et al., 2023, Liu et al., 2024a, Guo et al., 2024, Zeng et al., 2025, Prabhakar et al., 2025, Liu et al., 2024b, Yin et al., 2025]. However, gathering a sufficiently diverse and comprehensive tool set is inherently expensive and difficult. The resulting tool-use training data is often limited by the scope of the predefined APIs, while the ultimate goal of agentic training is the exposure to a sufficiently broad range of scenarios during training to enable agents to generalize effectively to unseen environments and scenarios [Fang et al., 2025]. This raises a critical question: Can we bypass the dependency on predefined tools and synthesize more diverse, high-quality trajectories directly from real-world?

In this paper, we answer this question by proposing a novel text-based extraction paradigm. We observe that the raw text corpora used for pre-training large language models inherently contain multi-turn tool-use patterns. Although such texts do not contain explicit agentic trajectories, they often document rich, real-world mutli-step problem-solving experiences, which can be extracted and transformed into multi-turn tool-use data

∗Equal Contributions. †Corresponding authors.

[Figure 1]

- Figure 1 | We propose the "Text to Trajectory" paradigm, instead of relying on predefined tools to generate tasks and trajectories.

(e.g., "the procedure for hospital reimbursement claims"). Our preliminary analysis of Ultra-fineweb [Wang et al., 2025] confirms that such texts contain actionable logical sequences spanning diverse domains, indicating that unstructured text serves as an untapped, scalable, and authentic source for synthesizing agentic training data.

To operationalize this paradigm, we introduce GEM, a data synthesis pipeline that enables generation and extraction of multi-turn tool-use trajectory from text corpora. The GEM pipeline proceeds in four stages: (1) Selection, identifying text segments rich in multi-step workflows; (2) Extraction, deriving structured workflows and tool definitions to map human-described procedures to agent actions; (3) Generation, employing a powerful model to convert text and abstract workflows into concrete user-agent interactions; and (4) Refinement, enhancing the complexity and diversity of the trajectories followed by rigorous verification.

Extensive experiments validate the effectiveness of our approach. When fine-tuned on data synthesized by our pipeline, the GEM-32B model achieves a 14.9% improvement on BFCL V3 Multi-Turn benchmark. Notably, We find that out-of-domain training data generated through our paradigm achieve performance on τ2-bench that is comparable to models trained on τ-bench in-domain data, thereby demonstrating the strong generalization capability of our approach. Building upon this high-quality data, we further develop a specialized Trajectory Synthesizer to internalize the "text-to-trajectory" mapping. This synthesizer provides a cost-effective, end-to-end solution for large-scale data generation, matching the quality of the multi-stage synthesis pipeline while significantly reducing costs.

In summary, our contributions are threefold:

- • We propose a new paradigm for agent trajectory synthesis that directly extracts multi-turn tool-use trajectories from text corpora. This paradigm unlocks an untapped, scalable source of authentic human problem-solving behaviors for training autonomous agents.
- • We develop GEM, a data synthesis pipeline that transforms raw text into multi-turn tool-use trajectories to prove the effectiveness of our proposed paradigm. We use GEM to synthesize a high-quality dataset that leads to significant performance gains: 13.8% on τ2-bench and 16.5% on BFCL V3 Mutli-Turn.
- • To enable cost-effective and scalable data generation, we develop a specialized Trajectory Synthesizer via supervised fine-tuning. We demonstrate that this model successfully distills the full GEM pipeline into an end-to-end generator, matching the original synthesis quality while significantly reducing costs.

#### 2 Problem Formulation

In this paper, we address the challenge of acquiring general multi-turn tool-use agent trajectories. We propose a novel paradigm that synthesizes such trajectories by extracting tool-use workflows directly from large-scale textual corpora. This approach naturally enjoys these advantages: the scale of real-world text ensures scalability, its inherent diversity covers multiple domains, and its foundation in grounded human problem-solving experiences yields high-quality, realistic agent data. Formally, we define the paradigm as:

- • Input: Let C denote a large-scale text corpus, where each element ci ∈ C is a raw, unstructured text segment (e.g., a document or narrative) that contains multi-step workflows.
- • Output: A complete list of tools P = {p1,...,pm}, where m denotes the number of available tools and each pi represents a tool defined in the standard OpenAI format. Additionally, a structured multi-turn tool-use

trajectory T = {s,(u1,a1,o1),...,(un,an,on)} is produced, where s denotes the system prompt and at each turn t, ut corresponds to a user’s query or request, at denotes the assistant’s natural language response or tool call, and ot is the resulting observation from the tool’s execution.

#### 3 Methodology

In this section, we introduce GEM, an agentic synthesis pipeline designed to automatically extract and generate multi-turn tool-use trajectories directly from large-scale text corpora. We first conduct a preliminary analysis of text corpora in Section 3.1. Subsequently, we detail our trajectory collection pipeline in Section 3.2.

##### 3.1 Preliminary Analysis

(10.3%)Others

Cust.Support (22.4%)

CloudPlatforms(1.8%)

Ent.BI(1.7%)

Communication(2.2%)

Security&Iam(2.2%)

SocialMedia(2.3%)

Multimedia(2.4%)

###### Total 100%

WebsiteControl(2.5%)

FileSystems(3.0%)

Research&Data(15.5%)

Databases(3.7%)

Ecomm.&Retail

Search(4.1%)

Developer Tools(7.6%)

(6.3%)

(12.1%)ElearningEdu.&

- Figure 2 | Distribution of the extracted task category from raw documents, showing the diversity of tasks covered in the text corpora.

###### Creating a Music Visualizer in Adobe After Effects

- 1. Prepare your audio ﬁle: Before starting with After Effects, make sure you have an MP3 or WAV ﬁle of the song that you want to create a visualizer for.

- 2. Create a new composition: Launch Adobe After Effects, then click 'Composition' > 'New Composition'. Set the width to 1920 pixels (or any other size depending on your project requirements) and set the height to 1080 pixels (for Full HD). Choose a frame rate - usually, 30fps works well but if you plan on exporting at higher quality like 60fps, choose this setting instead. Name your composition appropriately.

- 3. Import the Audio File: Click 'File' > 'Import', select your audio ﬁle and hit 'Open'. Now drag and drop the imported audio ﬁle into your timeline within the Project panel.

User Query Environmental Tools Multi-step Workﬂows

Figure 3 | Preliminary case study.

Setup To assess the feasibility of our proposed paradigm, we first perform a preliminary analysis on the characteristics of large-scale unstructured data, using the Ultra-fineweb corpus [Wang et al., 2025] as our main data source. We randomly sample approximately 250,000 raw text segments. Each segment is then processed through a sequential labeling pipeline: we first use a classifier to determine whether the text contains multi-step operational procedures, which helps us identify segments suitable for conversion into tool-use trajectories. Only those segments identified as containing procedural content are subsequently annotated with rich metadata, including platform, domain, and task category, via Qwen3-8B to evaluate the diversity of the corpus. We also conduct a preliminary case study. The prompt templates used for metadata annotation are provided in Appendix A.1.

Results First, by observing a large number of text cases, we find that unstructured textual data inherently contain the three core components essential for constructing agentic trajectories as shown in Figure 3: (1) User Queries, which emerge naturally as goals or problems stated in the text; (2) Environmental Tools, whose descriptions, APIs, or functionalities are often embedded within explanatory or instructional contexts; and (3) Multi-step Workflows, manifesting as step-by-step procedures or operational narratives.

We find that around 14% of the sampled segments contain explicit multi-step workflows, indicating a substantial reservoir of procedural knowledge. Given the massive scale of available text corpora, this represents a significant potential source for generating diverse agent trajectories. Furthermore, the identified procedural segments cover a wide spectrum of task categories and application scenarios—including education & E-learning, data analysis as illustrated in Figure 2. This diversity confirms that textual sources can provide the necessary variety in tasks, tools, and environments required for robust agent training. The detailed domain distribution is further reported in Appendix E.

These findings collectively demonstrate that generating multi-turn tool-use trajectories directly from unstructured text is not only feasible but also taps into a vast and largely underexplored repository of agentic trajectory.

###### Phase 1: Tools & Workﬂow Extraction

###### Phase 2: Trajectory Generation Final Validation

(Stage 1: Text Filtering & Stage 2: Tool/Workﬂow Extraction)

(Stage 3: Trajectory Generation & Stage 4: Trajectory Reﬁnement)

Valid Format & Toolcalls No Hallucination

[Figure 2]

[Figure 3]

Raw Text Corpora

[Figure 4]

[Figure 5]

[Figure 6]

###### Initial Trajectory

[Figure 7]

[Figure 8]

Extract abstract workﬂows from text

| | |
|---|---|
| | |

Expand Toolsets & Reﬁne System Prompt

Workﬂow Extraction

Rule-Based Check

LLM-Based Judge

[Figure 9]

System Prompt User Task

Retained Segments (Multi-step)

Reﬁned Trajectory

[Figure 10]

[Figure 11]

Text Filtering

[Figure 12]

High-Quality Agentic Data

Enhance User & Trajectory Complexity

Assistant Action Tool Response

Filter non-multi-step operations

Design functional Tools (Json, OpenAI schema)

Tool Deﬁnitions

Encourage Diverse Patterns (e.g., clariﬁcation, error recovery)

Figure 4 | Overall Pipeline of GEM. The first phase extracts multi-step workflows from filtered text corpora and defines corresponding tools. The second phase generates and refines trajectories, incorporating diverse behavioral patterns and rigorous validation to produce high-quality training data.

##### 3.2 GEM Synthesis Pipeline

We now detail the data synthesis pipeline of GEM to generate and extract multi-turn tool-use data. The overall pipeline is illustrated in Figure 4. All prompts used in this section are in Appendix A.

- Stage 1: Text Filtering To ensure the quality and realism of generated trajectories, the initial stage filters out raw text segments c that do not describe multi-step operations. The filtering procedure uses the same annotation prompt and model as described in Section 3.1.
- Stage 2: Workflow & Tool Extraction For each retained text segment c containing multi-step operations, we proceed to extract structured abstract workflows and synthesize corresponding functional tools directly based on the text segment description. First, the model is instructed to identify all workflows and enumerate the individual steps within each (e.g., search_items and then edit_item). Importantly, the model is encouraged to recognize workflow complexity, including sequential dependencies, conditional logic, and uniqueness constraints, thereby enhancing the richness and practical relevance of the output. Concurrently, the model designs a set of API tools P in accordance with OpenAI schema standards to support the subsequent trajectory generation. Each tool p is designed to perform a single, coherent function, with self-explanatory parameter names and well-specified data types. The final structured output comprises the abstract workflow descriptions alongside the complete tool definitions.
- Stage 3: Trajectory Generation Based on the text, abstract workflows and tools synthesized in the previous stage, we now proceed to generate concrete multi-turn tool-use trajectories. For each workflow and its corresponding set of tools, we employ a strong teacher model (GLM-4.6 in this work) to generate a preliminary complete trajectory T. Our approach synthesizes the full trajectory directly in a single pass instead of simulating turn-by-turn conversations via a multi-agent system to ensure efficiency. Each generated trajectory include the following core components:

- • System Prompt s: Clear domain-specific rules extracted from the source text, establishing guidelines the assistant must follow throughout the conversation.
- • User Task (u1,...un): A series of natural user requests that may be ambiguous or complex, reflecting real-world scenarios.
- • Assistant Responses (a1,...,an): Demonstrations of intelligent problem-solving, strict adherence to domain rules, and correct tool invocation with appropriate parameters.
- • Tool Response (o1,...on): Simulated tool outputs that are both complete and realistic, providing authentic feedback.

Here, n denotes the number of conversational turns for each role. To capture the diverse practical challenges inherent in real-world multi-turn interactions, we encourage the inclusion of various interaction patterns during

Multi-Turn Multi-Turn Multi-Turn Multi-Turn Multi-Turn Overall Acc Base Miss Func Miss Param Long Context

Model

Proprietary & Large Scale Models

GPT-4.1 38.88 47.50 32.50 32.50 43.00 DeepSeek-V3.2-Exp 37.38 41.50 39.50 33.50 35.00 Gemini-2.5-Flash 36.25 41.50 36.00 32.00 35.50

8B Models

Qwen3-8B 18.00 24.00 17.00 13.50 17.50 APIGEN-MT 21.00 25.50 19.00 25.00 14.50 TOUCAN 21.88 27.00 21.50 21.00 18.00 MUA 21.13 29.50 18.00 20.00 17.00 Qwen3-8B-GEM 30.25 40.00 30.00 28.00 23.00

32B Models

Qwen3-32B 28.35 34.00 24.50 25.50 29.50 APIGEN-MT 29.50 36.00 27.00 28.50 26.50 TOUCAN 35.00 41.00 37.50 26.00 35.50 MUA 26.25 30.00 25.00 23.00 27.00 Qwen3-32B-GEM 44.88 52.00 40.00 38.50 49.00

- Table 1 | Performance comparison on the BFCL V3 benchmark. The results are categorized by model scale. The best performance in each category is highlighted in bold. All metrics are reported as accuracy scores. The GEM-based models are highlighted in blue .

generation. These include but are not limited to: refusing requests that exceed the assistant’s capabilities, clarifying ambiguous user queries, and actively recovering from errors.

- Stage 4: Refinement We observe that although the initial multi-turn dialogue trajectories T are complete, they often lack sufficient complexity and tend to be relatively straightforward. To fully utilize and enhance the quality of these trajectories, we adopt a refinement strategy to further process them. Specifically, given T, we generate a refined trajectory T′ by expanding the variety of tools used, improving the realism of environmental responses, increasing the ambiguity and complexity of user requests, and ensuring the inclusion of non-trivial tool-call chains. We find this refinement essential for obtaining higher-quality agentic data, as demonstrated in Section 4.4.

Validation The trajectory filtering process integrates rule-based verification with LLM-based assessments to ensure high-quality outputs. Given a refined trajectory T′, we first apply a rule-based check to guarantee structural correctness. This involves verifying that all tools are correctly defined according to the OpenAI format and that each tool call corresponds to a valid function within the designated toolset, with argument names and types matching their definitions. We further validate the conversation format to confirm that tool responses meet all requirements and that all role tags are properly closed. Beyond structural correctness, we employ an LLM-based judge (Qwen3-32B in this paper) to detect and eliminate hallucinations. This step specifically examines all tool calls to ensure that every generated parameter value is explicitly grounded in the dialogue context rather than being fabricated. Only those trajectories that successfully pass both validation stages are retained as the final set Tfinal for use in supervised fine-tuning (SFT).

Data Synthesizer Generating such trajectories is costly and time-consuming. To address this, we propose training a data synthesizer via supervised fine-tuning (SFT), which learns a end-to-end mapping from text to multi-turn tool-use trajectories. This approach allows for the cost-effective synthetic training data. For each data instance, the synthesizer takes as input x an instruction ("Turn the following text into multi-turn tool-use trajectories") paired with a text segment. It then generates the corresponding output y, which includes both the necessary tool definitions and the resulting multi-turn tool-use trajectory.

#### 4 Experiments

##### 4.1 Experimental Setups

Benchmark. We evaluate our approach on two challenging benchmarks designed to assess multi-turn tool-use capabilities: (1) BFCL V3 [Patil et al.]: This benchmark involves agents interacting with a Python-based API environment. We focus specifically on the multi-turn scenarios, which are divided into four categories: Multi Turn Base, Miss Param, Miss Func, and Long Context with 200 tasks per category. (2) τ2-bench [Barres et al., 2025] (Airline and Retail): This benchmark evaluates user-agent interactions comprehensively within specialized real-world domains. We employ GPT-4.1 as the user simulator Following the original experimental setting and report performance using the Avg@4 and Pass@4 metrics.

LLMs and Datasets. We source the training data from Ultra-FineWeb [Wang et al., 2025] and employ GLM-4.6 to generate 10K synthetic trajectories. These trajectories are then used to fine-tune both Qwen3-8B and Qwen3-32B. Additionally, we leverage the same 10K samples to train a data synthesizer model based on Qwen3-8B.

Baselines. We compare our synthetic data against the following open-source datasets: (1) APIGEN-MT [Prabhakar et al., 2025], (2) Simia-Tau [Li et al., 2025]: We sample 50K multi-turn data from this dataset, (3) MUA [Zhao et al., 2025], and (4) TOUCAN [Xu et al., 2025]: We sample 50K multi-turn data from this dataset. Notably, APIGEN-MT and Simia are in-domain training data generated in the τ-bench environment (Airline and Retail).

Training Details. For supervised fine-tuning, we set the learning rate to 5 × 10−6 and train for two epochs. Fine-tuning is performed using LLaMA-Factory [Zheng et al., 2024] under a full-parameter setting. Further details are provided in Appendix B.

##### 4.2 Performance of GEM Synthesis Pipeline

BFCL V3 Results Table 1 presents performance comparisons on the BFCL V3 benchmark. The results of Simia are not included, as we observed its scores to be generally low, likely due to its reliance solely on in-domain τ-bench data. Our proposed GEM method demonstrates clear improvements over baseline models at both the 8B and 32B scales. In the 8B category, Qwen3-8B-GEM achieves an overall accuracy of 30.25%, significantly surpassing the base Qwen3-8B model and outperforming other open-source baselines such as APIGEN-MT and TOUCAN. The performance gain is even more pronounced in the 32B category, where Qwen3-32B-GEM attains an accuracy of 44.88%. This result not only exceeds open-source synthesized datasets like APIGEN-MT and MUA by a wide margin but also outperforms proprietary large-scale models, including GPT-4.1 (38.88%) and DeepSeek-V3.2-Exp (37.38%). These findings confirm that our data synthesis strategy effectively enhances function-calling capabilities across various multi-turn interaction categories.

τ2-Bench Results Table 5 presents the performance on the τ2bench benchmark, covering the Airline and Retail domains. We would like to emphasize that our models are trained on synthetically generated data that is strictly out-of-domain with respect to the τ2-bench test sets, while APIGEN-MT and SIMIA are synthesized datasets within the τ environment (Retail and Airline), which can be regarded as in-domain training data. Despite this, our approach still demonstrates comparable performance. At the 8B scale, Qwen3-8B-GEM remains highly competitive, achieving results comparable to models like SIMIA that are fine-tuned on in-domain synthetic data, and surpassing APIGEN-MT in the Retail domain with a Pass@4 score of 75.44% versus 69.30%. At the 32B scale, our model exhibits strong generalization, outperforming both SIMIA and MUA in the Retail domain with a Pass@4 of 86.84% and delivering competitive results in the Airline domain. This indicates that our text-based synthesis pipeline instills a fundamental understanding of tool-use reasoning that transfers effectively to unseen, real-world domains, matching or even exceeding the performance of models trained

Airline Retail Avg@4 Pass@4 Avg@4 Pass@4

Model

8B Models

Qwen3-8B 13.00 18.00 38.16 66.67 APIGEN-MT 23.50 42.00 42.54 69.30 Simia 35.50 52.00 43.20 70.18 TOUCAN 20.50 42.00 26.32 52.51 MUA 20.50 40.00 30.26 61.40 Qwen3-8B-GEM 22.00 40.00 44.52 75.44

32B Models

Qwen3-32B 21.00 40.00 43.20 70.18 APIGEN-MT 36.00 52.00 44.52 74.56 Simia 38.00 62.00 48.03 73.68 TOUCAN 30.00 52.00 43.86 72.81 MUA 33.00 54.00 49.56 80.70 Qwen3-32B-GEM 35.50 56.00 55.48 86.84

Figure 5 | Results on τ2-bench. We report avg@4 and pass@4 metrics for Airline and Retail domains.

BFCL V3 Multi-Turn Tau2 Overall Base Miss Miss Long Airline Retail Acc Acc Func Param Context Avg@4 Pass@4 Avg@4 Pass@4

Model

Qwen3-8B (Base) 18.00 24.00 17.00 13.50 17.50 13.00 18.00 38.16 66.67 + GEM-GLM (Ultrafineweb) 30.25 40.00 30.00 28.00 23.00 22.00 40.00 44.52 75.44 + GEM-Synthesizer (Ultrafineweb) 28.38 41.50 23.50 27.50 21.00 26.00 40.00 42.11 73.68 + GEM-Synthesizer (Wikihow) 28.50 43.50 22.00 27.00 21.50 25.00 42.00 39.67 68.42

- Table 2 | Effects of data synthesizer and text sources. We compare the impact of different trajectory generation models (GLM vs. Trained Synthesizer) and data sources (Ultrafineweb and Wikihow) on the Qwen3-8B base model. The best performance is highlighted in bold.

on more domain-aligned data distributions. Overall, the results underscore the effectiveness of our proposed paradigm.

##### 4.3 Performance of GEM Synthsizer

We use 10K trajectories generated by GEM synthesizer to fine-tune Qwen3-8B, as shown in Table 2. When using the same data source (Ultrafineweb), our GEM-Synthesizer achieves an overall accuracy of 28.38% on the BFCL dataset, which is close to the performance obtained by directly using synthesized data generated by GLM-4.6, and also yields the Pass@4 score (73.68%) in the τ2 Retail domain. We also use another textual source, Wikihow [Koupaee and Wang, 2018], to show the generalization of our synthesizer. The GEM-Synthesizer also achieves a high overall performance on both BFCL (28.50%) and on the τ2-bench Airline domain. These findings indicate that our synthesizer can generate high-quality tool-use trajectories in a low-cost, end-to-end way, thereby enabling stronger generalization in complex tool-use scenarios.

##### 4.4 Ablation Study

To validate the effectiveness of our synthetic data pipeline, we conduct an ablation study with BFCL V3 benchmark focusing on two key components: the refinement stage and the LLM-based check as shown in Table 6. We report the ablation results on τ2-bench in Appendix C.

Overall Multi Miss Miss Long Acc Base Func Param Cxt.

Model

Qwen3-8B (Base) 18.00 24.00 17.00 13.50 17.50 Qwen3-8B-GEM 30.25 40.00 30.00 28.00 23.00 (w/o Refine) 26.00 33.50 23.50 27.00 20.00 (w/o LLM-Based Check) 27.38 37.50 25.00 24.00 23.00

Effect of Refinement. The refinement stage leads to a substantial performance improvement. For example, it raises the overall accuracy of Qwen3-32B from 32.50% to 44.88%, a gain of over 12 percentage points. This process increases the complexity and quality of the synthetic trajectories, which in turn enables more effective learning of multi-turn tool use. We report more details in Appendix D to illustrate out refinement strategy significantly enhances the overall complexity of the synthesized data. Notably, even the original trajectories extracted directly from the original text (though relatively simpler) still provide valuable training signals and contribute to improved tool-calling capability. This indicates that more effectively leveraging information from the original text to synthesize high-quality tool-calling trajectories is a promising research direction.

Qwen3-32B (Base) 28.35 44.00 24.50 25.50 29.50 Qwen3-32B-GEM 44.88 52.00 40.00 38.50 49.00 (w/o Refine) 32.50 40.00 30.00 27.00 33.00 (w/o LLM-Based Check) 44.25 52.00 40.50 40.00 44.50

Figure 6 | Ablation study. We conduct ablation study with BFCL V3 benchmark.

Effect of LLM-Based Check. This stage consistently improves results by filtering out samples with hallucinations or inconsistencies. For the 8B model, it raises overall accuracy from 27.38% to 30.25%. This mechanism also contributes to stronger performance across both model scales.

###### Distribution of Number of Messages

###### Distribution of Number of Tools

###### Distribution of Number of Tool Calls

30

Mean: 46.1

Mean: 8.6

Mean: 16.3

15.0

Median: 41.0

Median: 8.0

Median: 13.0

6

25

12.5

Percentage

Percentage

Percentage

20

10.0

4

15

7.5

10

5.0

2

5

2.5

0.0

0

0

0 25 50 75 100 125 Number of Messages

2.5 5.0 7.5 10.0 12.5 15.0 Number of Tools

0 10 20 30 40 50 60 Number of Tool Calls

Figure 7 | Data analysis. We find the synthesized trajectories are overall complex. Left: distribution of number of tools. Mid: distribution of number of messages. Right: distribution of number of tool calls in each trajectory.

[Figure 13]

Pattern 1: Clariﬁcation

Pattern 3: Adherence to Rules

###### Original Text Segment

[Figure 14]

[Figure 15]

User: I would like to add some text to my vacation I write on my photos before printing photo. Can you help me write "Summer 2024" on it? them. First I copy the picture, then I click edit, then text and you can

User: Make it really big! I want font size 150 so it stands out.

Assistant: To do this, I 'll need a few more details: [clariﬁcation for image_path and location to add]

Assistant: The font size must between 8 and 96. [Reject requests that violates the set rules]

[Figure 16]

[Figure 17]

add/edit text and write on the photo. Can make changes, font size from 8 to 96, and in any color I want. Then have it printed.

Pattern 2: Correct Tool Call Sequence

###### Pattern 4: Error Recovery

[Figure 18]

User: The image is at /home/vacation.jpg. I want the text in the bottom right corner.

User: Please help me print these photos.

[Figure 19]

[Figure 20]

Tools & System Constraints

Tool Call: print_image() but failed

Assistant: I'll help you add text to your photo. [ copy_image()-> open_editor() -> add_text() ]

[Figure 21]

[Figure 22]

Assistant:An error occur. This could be due to ... [detailed analysis of error and provide alternatives]

Cope_Image(), ... You can not ...

Figure 8 | Case study of our generated multi-turn tool-use trajecory.

##### 4.5 Data Analysis

Figure 7 presents a statistical overview of our synthesized multi-turn tool-use trajectories, highlighting three key dimensions: the number of distinct tools employed, the number of messages per dialogue, and the total number of tool calls. On average, each trajectory involves 8.6 distinct tools, indicating that the synthesis process requires models to meaningfully select and combine multiple tools within a single task. Moreover, the trajectory contains an average of 46 turns. The considerable length of these dialogues ensures that tasks cannot be resolved simply, helping models learn to maintain context, track task state, and engage in multi-step planning over prolonged dialogues. In comparison, existing open-source datasets such as APIGEN-MT [Prabhakar et al., 2025] average around 18.5 turns, while TOUCAN [Xu et al., 2025] contains only about 6.24 turns. Finally, each trajectory averages 16.3 tool calls. while APIGEN-MT contains only an average of 4.3 tool calls. This high frequency underscores the multi-step, tool-driven nature of the synthesized tasks. Taken together, these statistics demonstrate that our synthesized dataset exhibits substantial diversity and interaction depth, making it well-suited for training models in multi-turn tool-use scenarios.

##### 4.6 Case Study

For the case study illustrated in Figure 8, this synthesized trajectory is based on a real-world photo-editing scenario. The synthesis process first extract diverse tools from the text description, and identifies key constraints from the user’s actions, such as the restriction that font size must be between 8 and 96 points. Building on these tools and rules, the dialogue encompasses a variety of realistic interaction patterns: proactively clarifying missing parameters (i.e., image path and text placement), invoking tools in the correct sequence, rejecting requests that violate constraints (i.e., exceeding the font size limit) while offering compliant alternatives, and recovering from errors by retrying with another available printer. It guides the model to learn how to validate inputs, adhere to system constraints, execute tasks step by step, and respond flexibly to errors within multi-turn interactions, thereby achieving reliable tool-use capabilities. We also show a complete synthesized trajectory in Appendeix F to help readers better understand our methodology.

#### 5 Related Work

Tool-Use Data Synthesis To equip LLM-based agents with tool-calling capabilities, prior works focus on synthesizing tool-use training data. ToolBench [Qin et al., 2023] builds a large-scale function-calling dataset over thousands of APIs. ToolACE [Liu et al., 2024b] designs an automatic agentic pipeline that iteratively expands an API pool and synthesizes verified function-calling traces with high complexity and diversity. For multi-turn interaction, APIGen-MT [Prabhakar et al., 2025] generates structured task blueprints with ground-truth action sequences and then simulates realistic human–agent dialogues grounded in executable APIs. MagNet [Yin et al., 2025] represents multi-tool workflows as graph-structured function paths and converts them into multi-turn conversations with executable calls. ToolACE-MT [Zeng et al., 2025] adopts a non-autoregressive framework that drafts entire dialogues in one shot and refines them via iterative editing and verification. TOUCAN [Xu et al., 2025] crawls MCP servers and synthesizing 1.5M of tool-use data. Unlike prior works that rely on pre-defined tools, our work introduces a novel paradigm that directly extracts multi-turn trajectories from text, thereby unlocking a authentic and scalable source of tool-use agentic data.

Tool-Use Capability Evaluation To systematically assess tool-use capabilities of LLM agents, a series of benchmarks have been proposed. ToolBench [Qin et al., 2023] focuses on evaluating whether models can translate natural-language instructions into correct API calls across thousands of real-world tools. The Berkeley Function Calling Leaderboard (BFCL) [Patil et al.] provides a large-scale, syntax- and semantics-aware evaluation of function calling across diverse domains and programming languages, using AST-based checks to measure the accuracy and compositionality (e.g., parallel calls) of model-generated function invocations. ACEBench [Chen et al., 2025] evaluates multi-turn tool-use capability from three perspectives: normal, special, and agent. τ-bench [Yao et al., 2024] emulates dynamic conversations between a simulated user and a toolaugmented agent in domain-specific scenarios, jointly evaluating task success, tool selection, and adherence to domain policies. τ2-bench [Barres et al., 2025] further extends this line by introducing dual-control environments where both the agent and the user can invoke tools, highlighting the challenges of coordinating tool use and guiding user actions in more realistic customer-service-style tasks. VitaBench [He et al., 2025] focuses on evaluating agent’s performance on life-serving simulation environment.

#### 6 Conclusion

This paper presents a novel paradigm and the agentic data synthesis pipeline, which directly synthesize multi-turn tool-use trajectories from text corpora, effectively bypassing the dependency on predefined tools. Models trained on our data achieve significant performance gains on benchmarks, demonstrating the potential of leveraging open-world textual knowledge as a scalable source for advancing autonomous agents.

#### References

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025a.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

Meituan LongCat Team, Bei Li, Bingye Lei, Bo Wang, Bolin Rong, Chao Wang, Chao Zhang, Chen Gao, Chen Zhang, Cheng Sun, et al. Longcat-flash technical report. arXiv preprint arXiv:2509.01322, 2025b.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ2-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.

Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh RN, et al. Apigen: Automated pipeline for generating verifiable and diverse function-calling datasets. Advances in Neural Information Processing Systems, 37:54463–54482, 2024a.

Zhicheng Guo, Sijie Cheng, Hao Wang, Shihao Liang, Yujia Qin, Peng Li, Zhiyuan Liu, Maosong Sun, and Yang Liu. Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models. arXiv preprint arXiv:2403.07714, 2024.

Xingshan Zeng, Weiwen Liu, Lingzhi Wang, Liangyou Li, Fei Mi, Yasheng Wang, Lifeng Shang, Xin Jiang, and Qun Liu. Toolace-mt: Non-autoregressive generation for agentic multi-turn interaction. arXiv preprint arXiv:2508.12685, 2025.

Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, et al. Apigen-mt: Agentic pipeline for multi-turn data generation via simulated agent-human interplay. arXiv preprint arXiv:2504.03601, 2025.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, et al. Toolace: Winning the points of llm function calling. arXiv preprint arXiv:2409.00920, 2024b.

Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long Le, Kai-Wei Chang, Chen-Yu Lee, et al. Magnet: Multi-turn tool-use data synthesis and distillation via graph translation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32600–32616, 2025.

Runnan Fang, Shihao Cai, Baixuan Li, Jialong Wu, Guangyu Li, Wenbiao Yin, Xinyu Wang, Xiaobin Wang, Liangcai Su, Zhen Zhang, et al. Towards general agentic intelligence via environment scaling. arXiv preprint arXiv:2509.13311, 2025.

Yudong Wang, Zixuan Fu, Jie Cai, Peijun Tang, Hongya Lyu, Yewei Fang, Zhi Zheng, Jie Zhou, Guoyang Zeng, Chaojun Xiao, et al. Ultra-fineweb: Efficient data filtering and verification for high-quality llm training data. arXiv preprint arXiv:2505.05427, 2025.

Yuetai Li, Huseyin A Inan, Xiang Yue, Wei-Ning Chen, Lukas Wutschitz, Janardhan Kulkarni, Radha Poovendran, Robert Sim, and Saravan Rajmohan. Simulating environments with reasoning models for agent training. arXiv preprint arXiv:2511.01824, 2025.

Weikang Zhao, Xili Wang, Chengdi Ma, Lingbin Kong, Zhaohua Yang, Mingxiang Tuo, Xiaowei Shi, Yitao Zhai, and Xunliang Cai. Mua-rl: Multi-turn user-interacting agent reinforcement learning for agentic tool use. arXiv preprint arXiv:2508.18669, 2025.

Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, and Rameswar Panda. Toucan: Synthesizing 1.5 m tool-agentic data from real-world mcp environments. arXiv preprint arXiv:2510.01179, 2025.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372, 2024.

Mahnaz Koupaee and William Yang Wang. Wikihow: A large scale text summarization dataset. arXiv preprint arXiv:1810.09305, 2018.

Chen Chen, Xinlong Hao, Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Yuefeng Huang, et al. Acebench: Who wins the match point in tool usage? arXiv preprint arXiv:2501.12851, 2025.

Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, et al. Vitabench: Benchmarking llm agents with versatile interactive tasks in real-world applications. arXiv preprint arXiv:2509.26490, 2025.

Appendix

- A Prompts

- A.1 Tag Annotation Domain & Category Annotation

Determine whether the following text contains multi-step operations involving the use of an APP, website, computer, or other machine (such as robot, elevator, etc), if contains, generate one sentence summary of the task and identify the platform, domain and task category of the multi-step task.

# Detaild Instruction

- 1. Platform category: operator, computer, phone, machine, other

- 2. Domain category: adult, arts_and_entertainment, autos_and_vehicles, beauty_and_fitness, books_and_literature, business_and_industrial, computers_and_electronics, finance, food_and_drink, games, health, hobbies_and_leisure, home_and_garden, internet_and_telecom, jobs_and_education, law_and_government, news, online_communities, people_and_society, pets_and_animals, real_estate, science, sensitive_subjects, shopping,

sports, travel_and_transportation

- 3. Task category: databases, multimedia_processing, cloud_platforms, calendar_management, cryptocurrency, location_services, communication, search, file_systems, web_scraping, ecommerce_and_retail, customer_data_platforms, developer_tools, virtualization, version_control, research_and_data, aigc, travel_and_transportation, note_taking, language_translation, rag_systems, security_and_iam, social_media, monitoring, weather_services, customer_support, blockchain, knowledge_and_memory, financial_trading,

marketing, enterprise_business_intelligence, transportation_logistics, iphone_android, smart_home, education_elearning, robot_control, website_control, gaming_entertainment

# Input [text]

# Output Format <multi_step>False</multi_step> Or: <multi_step>True</multi_step> <summary>xxx</summary> <domain>Shopping, Sports</domain> <platform>Operator</platform> <task>customer_support</task>

- A.2 Workflow & Tool Discovery Workflow & Tool Discovery

You are an program design expert. Given a workflow description in a scenario, your task is to design multiple functions to

translate the execution process of this workflow into program. # Instruction

- 1. Extract all intermediate steps in the workflow, if the text contains multiple workflows, output them in a list.

- 2. Convert **every** step to a function and represent them as an execution graph (i.e. ( login)->(search_query)->..)

- 3. Based on the execution graph, generate real API calls that populate the tools with reasonable parameters, simulating a use case of actual tool invocation steps.

- 4. Provide detailed API definitions used in the above process.

- 5. Follow the following steps to generate more complex workflows and tools:

- - Workflow Exploration: You need to explore multiple workflows or complex constraints that may exist in the document

- - These workflows represent the possible interaction patterns of a real user-agent.

- - Dependencies: "X must happen before Y".

- - Uniqueness/Limits: "Only one Admin allowed", "Name must be unique".

- - Conditionals: "If user is X, they cannot do Y".

- - Tool Design (Functional API Level) Design a set of JSON-schema tools based on the text.
- - The required parameters of a tool need to be carefully considered and designed, mirroring the logic of the real world. For example, viewing system data typically requires authorization authentication, and providing user ID, product ID, etc.

- - It mimics a database structure and provides read and write tools. For example, it provides tools for querying user information, along with corresponding tools for modifying user information.

- - Each tool’s name should be short and readable, semantically clear and general, reusable (e.g., "flight_search" rather than "flight_detailed_search_for_tom_2025")

- - Each tool should implement a single, coherent capability. It should not bundle multiple unrelated or multi-stage workflows into one tool. (e.g., create two tools " plan_trip" + "book_trip" rather than only one tool named "plan_and_book_trip")

- - Each tool’s parameters should be explicitly defined in the schema with clear types and meanings. Parameter names should be self-explanatory rather than cryptic (e.g., use "

check_in_date" with type "string" and a short description, rather than a vague parameter named "d1").

- The majority of tools describe functional data operations that either retrieve information from or modify the state of the environment (e.g., get_status, update_permissions)

# Workflow Description [text]

# Output Format <workflow> <description>short task description</description> <steps>Step1: ...\nStep2: ...</steps> <execution_graph>(api_name1)->(api_name2, api_name3)->..</execution_graph> <actions>[{"name":"api_name", "arguments": {"arg_name": "value", ...}}, ... (more API calls

as required)]</actions>

<tools>[{"name":"api_name1","description":"","inputSchema":{"type":"object","properties":{" arg_name1":{"type":"","description":""},"arg_name2":{"type":"","description":""}}," required":["arg_name2"]}},{"name":"api_name2","description":"","inputSchema":{"type":" object","properties":{"arg_name1":{"type":"","description":""},"arg_name2":{"type":""," description":""}},"required":["arg_name2"]}}]</tools>

</workflow> <workflow> (more workflows) </workflow>

##### A.3 Trajectory Generation Trajectory Generation

You are tasked with generating high-quality multi-turn dialogue trajectories based on a given text document. The trajectory should demonstrate an AI assistant helping users complete tasks while strictly following domain-specific rules and constraints.

You will be provided with:

- - A list of Available Tool Candidates;

- - A source text document that contains the description of the scenario and task steps;

## Completion Requirements

- 1. System Prompt: Extract and explicitly state ALL important domain-specific rules and constraints from the source text document.

Example: <system> You are a agent specilized in retail domian. Here are some basic rules to follow:

- - An order can only be cancelled if its status is ’pending’ ...

- - Modify action can only be called once, and will change the order status to ’pending (items modifed)’ ...

- - ... </system>

- 2. User Task: Create natural, progressive user requests that test the system’s rule enforcement and constraint handling. Here are some features:

- - Naturalness: Requests should reflect real-world use cases

- - Ambiguity: User requests are often incomplete, requiring the assistant to analyze or clarify them.

Example: <user> I want to cancel order #W2575533. </user> (the user do not provide the

specific reason, and the assistant should ask for clarification); <user> Recommend me a desktop. I often go out. </user> (the user do not explicitly state the attribute of the item, but the assistant should analyze and know it based on the stated preference)

- - Consistent: User’s intention, persona, and their behavior should be consistent across the dialogue.

- - Complex: The user’s request is challenging enough to test the assistant’s ability. Users can make requests that violate domain rules and are not allowed to alert the assistant ( e.g., do not ask the model to verify the order status first).

At least in one turn, the user’s request is very complex and require assistant to handle it carefully.

- Example 1: <user> I need to make several changes to my order #W2575533. Can I change the E-Reader to

a different size, swap the Garden Hose color, and also update my shipping address </user >

(Requires assistant to: check order status, verify each item can be modified, handle address change separately, remind about one-time modification limit)

- Example 2: <user> Check my tire pressures. If any of them are low, find me the nearest service

station and also check if I have enough fuel to get there (Requires: check tire pressure, evaluate condition, conditionally call find_nearest_shop, check fuel level, calculate if sufficient)

- Example 3: <user> I’m planning a three-day trip starting from Hangzhou, and I need help creating an

itinerary. One more thing about the second day - I’m trying to be smart about my budget. If I end up booking a luxury hotel that costs 800 CNY or more per night, then I need to be more careful with other expenses: my total spending on both restaurants (lunch and dinner) should stay under 350 CNY, both restaurants should be rated at least 4.0 stars, and the afternoon attraction ticket needs to be less than 120 CNY. </user>

(Requires: check multiple constraints)

- 3. Assistant: Generate Responses that Demonstrate Rule Enforcement, Clear Communication, and Intelligent Problem-Solving:

- - Reasoning and Adaptive Planning: The Assistant should reason through problem contexts and plan appropriate steps. Sometimes users may not be able to directly provide the parameters for tool calls, and the assistant needs to accurately consider whether the parameter values can be obtained through other information and tools.

- - Precondition Checks: Before executing tasks, the Assistant should validate any necessary preconditions (e.g., authenticating identity, verifying the status of an order)

- - Domain Rules and Constraints: The Assistant must follow domain-specific rules at all times. Ensure the assistant’ tool call and response genuinely addresses those requirements.

- - User-Centric Principle: The Assistant must accurately understand and satisfy all user needs and preferences without breaking domain rules. For example, if user states " prefers A, wants B, and tell me C", the assistant should satisfy all requirements.

- - No Hallucination:

- - Context Faithfulness: Maintain absolute fidelity to tool outputs. If data contradicts user expectations, explicitly report the discrepancy instead of distorting facts to force a match.

- - Tool call arguments: The Assistant must only use argument values that are explicitly provided or implied by the user. It must not fabricate IDs, names, or other parameters; if any required value is missing or unclear, the Assistant should ask the user to supply

it before calling the tool.

- - Consequence-aware: Before executing any write operation that modifies the environment, the Assistant must actively think and assess its impact. For changes that are irreversible, the assistant should obtain explicit user comformation before proceeding.

- - Limitations: If the current request is beyond the assistant’s ability, the Assistant must communicate this limitation clearly. For example: If the user’s needs are beyond domain rules, the Assistant must explain the limitation but should still respect the user’s needs. (e.g., "I cannot do A, but I can do B. Should I proceed with B?" -> Wait for conformation).

- - Correct Tool Calls:

- - Tool call format: ‘<func>{"name":"exact_tool_name", "arguments": {"arg": "value"}}</func >‘.

- - Exact Matching: The Assistant must ensure that tool calls exactly match the available candidate tools. Calling tools that do not exist is prohibited.

- - Parameter Validation: The arguments passed to the tools must exactly match the tool definitions, including the required parameters. The Assistant should avoid any missing parameters and validate that the values are accurate.

- 4. Tool Responses: Structured and Complete

- - Correctness of Tool Call: A success response should only be returned if the tool call is correct (including both the correct function name and parameters). If any part of the tool call is incorrect or incomplete, the tool should return a failure response, indicating what went wron

- - Success Response: The tool’s success response must return all relevant information in a well-structured format (e.g., JSON). This includes not just the requested data, but also

any other relevant details, such as order ID, status, product ID, item ID, etc., in case the user requires further context.

- - Error Response: When there is an error, the tool should return only the error message. No additional information should be provided that directly aids the Assistant in recovering

from the error.

- - Consistent Response Structure: The format and content of the tool’s responses should remain consistent throughout the conversation. This ensures clarity and reliability in the tool’s output, helping to maintain a smooth user experience.

- - Don’t confuse the order between turns:

- - a user message or tool result should be followed by an assistant message. (<user>...< user> / <tool>...</tool> -> <assistant> ... </assistant>)

- - If the assistant message includes a tool call, it can be followed by a tool message; (< assistant>...<func>...</func></assistant> -> <tool>...</tool>)

- - otherwise, it should be followed by a user message. (<assistant> ... </assistant> -> < user> ... </user>)

- - Tool result cannot followed by user message. (MUST NOT OUTPUT: <tool>...</tool> -> <user >...</user> (incorrect example))

- 5. Trajectory Pattern: The trajectory should exhibit multiple interaction patterns, with at least 3 distinct patterns appearing in total. Each individual pattern should be used at most 2 times within a single trajectory.

- [Pattern 1: Domain Rules & User Need conflicts]

- [Pattern 2: Error Recovery]

- [Pattern 3: Clarification and Disambiguation]

- [Pattern 4: Assistant’s Multi-hop Reasoning]

- [Pattern 5: Assistant’s Awareness of Domain Rules] ## Other Requirements

- - Always use English.

- - The whole trajectory should be reasonable, realistic (conform to real-world dialogue scenarios and interactions), and fit the context of multi-turn tool usage.

- - Don’t confuse the order between rounds: a user message or tool result should be followed by an assistant message. If the assistant message includes a tool call, it can be followed by a tool message; otherwise, it should be followed by a user message. Tool result cannot followed by user message.

- - Tool call format: <func> {{"name": "exact_tool_name_in_toolsets", "arguments": {{"arg": "value"}}}} </func>

## Given Inputs ### Available Tool Candidates {candidate_tools}

### A source text document consisting of Task Steps Description {current_task}

## Output Format You must STRICTLY follow the following output format. Ensure ALL tags are properly opened and closed. Conversations like "<tool> ... </assistant>",

"<assistant>...</user>" are wrong!!! <system>

... </system> <user>

... </user> <assistant>

... <func> {{"name": "...", "arguments": {{...}}}} </func> </assistant> <tool>

... </tool>

##### A.4 Trajectory Refinement Trajectory Refinement

You are rewriting a complex, realistic multi-turn tool-use trajectory for agenic training. Goal: The trajectory should be complex, natural, no-hallucination and show the wisdom of the

assistant. It must show the assistant’s ability for correct tool use, reasoning, context understanding, and communication skills with users. Please carefully consider what problems exist with the existing synthetic trajectories and

how to improve them to create high-quality trajectories. You can follow the following guidelines. # Guideline ### System Prompt Complexity You need to refine and upgrade the constraints of the system prompts to make them more

structured, systematic, and consistent with real-world logic, in order to fully test the agent’s ability to make correct tool calls in complex scenarios. You also need to define the database schema in the system prompts, and the data structure returned by the tool will be based on this.

### User Request Complextity & Naturalness

- - Natural: Natural requests may include colloquial language, implied context, vague references to prior steps, or real-world motivations (e.g., saving time/money, lose weight). Avoid overly formal or purely instructional language.

- - User diversity: Create a user profile and maintain the user’s personality and characteristics throughout the conversation history.

- - Requires Deep Analysis & Reasoning: The user’s request must necessitate careful analysis and multi-step reasoning to identify the correct tool(s) and determine appropriate parameter values.

- - Requires Analysis of Tool Dependencies & Outputs: The request should force the assistant to understand dependencies between tools and use outputs from previous steps to decide which tool to invoke next.

- - In at least 1 turn, the user’s request contains multiple constraints, including explicit constraints, implicit requirements that require the assistant to infer.

- - In at least 1 turn, The user asks a question that can only be answered by reasoning across outputs from multiple tool calls in the long context.

- - Challenging Pitfalls: **MUST INCLUDE AT LEAST 1-2 PITTFALLs** for one trajectory. This pattern sets traps to test the assistant’s ability to correctly make robust tool

calls based on rules, constraints or user preferences.

The trajectory must include request for this kind of challenging tool calls, and the assistant must explicitly analyze and identify these pitfalls to achieve robust tool calls.

### Assistant Intelligibility Demonstrate the following capabilities of the assistant:

- (1) Communication Skills

- - User Intent Understanding: Accurately interpret the underlying goals and context of the user’s request.

- - Confirmation & Clarification: Proactively confirm key details or ask the user to clarify ambiguous information when necessary.

- - Capability Limitation Awareness: Clearly communicate the boundaries and limitations of the assistant’s available capabilities.

- - Result Explanation & Summarization: Interpret, distill, and present tool outputs or complex information in a structured, understandable manner.

- - Proactive Assistance: Anticipate potential user needs based on context and offer helpful suggestions in advance.

- (2) Robust Tool-Calling Capability

- - Tool Selection: Choose the most appropriate tool from the available set based on task requirements.

- - Sequential Tool Usage: Plan and execute multi-tool workflows with correct dependencies and order.

- - Parameter Handling: Correctly construct and validate complex argument structures (e.g., nested objects, lists), respecting type, range, and format constraints.

- - Result Analysis: Parse and evaluate tool responses, extract relevant information, and determine validity for subsequent steps.

- - State Tracking: Maintain awareness of completed and pending steps in a multi-step task.

- - Constraint Analysis: Identify and adhere to real-world constraints and business rules (e.g

., date ranges, mutually exclusive fields, batch limits).

- - Error Handling: Gracefully manage tool failures, diagnose error causes, and adjust strategy or inform the user appropriately.

- - Context Management: Effectively retain and utilize conversation history to ensure coherence across complex interactions.

- (3) Reasoning & Execution Ability

- - Planning & Task Decomposition: Break down complex or vague user requests into clear, executable step-by-step plans.

- - Prerequisite Management: Recognize and acquire necessary information or conditions before executing tasks (e.g., querying environment, requesting user input).

- - Verification and Validation: Perform essential checks before critical operations, such as exploring available tools, confirming permissions, or validating inputs.

### Realistic & Complex Environment IMPORTANT NOTES:

- - The tool called in the trajectory MUST exist in candidate toolsets, with the parameter type and value correct.

- - If it’s necessary to add tools when constructing complex trajectories, you can add them in the final output tools block.

GOALS:

- - This increases the difficulty of choosing tools. A diverse range of tools must be included, specifically reading and writing tools.

- - Increase the difficulty of making totally correct tool calls (parameter type, value). For example, include structured inputs (list, dict, nested objects) and meaningful constraints.

- - The tools should conform to the database schema as much as possible (may defined in system prompt), and mainly include read-write tools. You can add & modify tools.

- - Unique: Tool call parameters should use unique database fields (such as user ID, product ID, etc.) as much as possible to mimic real logic.

- - Realistic: Always avoid using empty placeholders. For example, do not return something like "5+ more results ...", "path=/example".

- - Success Response: A success response should only be returned if the tool call is correct. The tool’s response must return a complete data structure in a well-structured format (e. g., JSON).

- - Error Response: You are allowed to simulate non-simple errors that might occur in the real world. The tool’s response must return a concise error message. Avoid directly telling

the assistant how to solve the problem. ### Trajectory Diversity

- - Reduce the frequency of repeatedly using certain tools to solve problems, avoiding the reduction of trajectory diversity, and retain only the most valuable trajectories for learning.

- - Include but not limited to the following pattern, and make the following pattern more difficult, diverse and natural.
- - [Pattern 1: Environment Complexity]

- [1.1: Error Recovery] In multi-turn function calling, models may encounter errors, such as invalid input or failed execution that require recovery. If you think of any

suitable, non-trivial, real-world scenario errors, please include this pattern.

- [1.2: Long Context] Introduce large volumes of extraneous data to test how well the model can extract crucial details from an overwhelming array of information.

- - [Pattern 2: Clarification] Tests the model’s ability to recognize when essential information is missing from the user request.

- - [2.1: Can be inferred from the system] The assistant actively explore ways to find

the essential information and complete the task

- - [2.2: Can not be inferred from the system] The assistant actively clarifies the

situation.

- - [Pattern 3: Identify Limitation] Requires the model to identify that no available function can fulfill the user request.

- - [Pattern 4: Assistant guides user operation] For example, if a user reports no internet access, the assistant uses tool calls to discover that the SIM card is not inserted, and then guides the user to insert it (this process cannot be performed by the

assistant alone due to real-world phycial constraints and requires active communication and guidance between the assistant and the user).

# Input Data Toolset: {tools} Trajectory: {our_traj}

# Output Requirement

- - Reduce redundancy: Reduce the frequency of repeatedly using certain tools to solve problems, avoiding the reduction of trajectory diversity, and retain only the most valuable trajectories for learning.

- - Preserve turn order.

- - The assistant can only call the tool once per round.

- - Each tool message must be followed by an assistant message.

- - If an assistant message contains a tool call, it must be followed by a tool message ( tool result).

- - Tool messages must not be followed directly by user messages.

- - If the original trajectory violates these rules or misuses tools, fix it in the rewritten version.

- - Output all candidate tools (the original tools + new tools if needed)

- - You must strictly follow the following output format.

# Output format <toolsets> All candidate tools in JSON, OPENAI format. [

{{

"name": "", "description": "", "inputSchema": {{

"type": "", "properties": {{ }}, "required": []

}}

}}, ...

] </toolsets>

<system> [role and domain rules here] </system> <user>

... </user> <assistant>

... <func> {{"name": "...", "arguments": {{...}}}} </func> </assistant> <tool> [concrete tool response in JSON format if tool calls are made] </tool> <assistant>

... </assistant>

... (more conversations here) """

##### A.5 Hallucination Detection Hallucination Detection Prompt

You are given a multi-turn tool-use trajectory. Please evaluate the trajectory according to

the following rubrics. Your job is to score the trajectory on the following binary rubrics. For EACH rubric, you must output 0 or 1 only, according to the criteria below. For each rubric, if there is no hallucination of the following content throughout the entire

trajectory, the corresponding rubric score is 1. If any single round does not meet (the condition), the corresponding rubric should be scored as 0. Be strict in your evaluation.

## Rubric

- R1: Tool-call hallucination

- Check whether any tool call uses argument values that are not provided or reasonably derivable from the dialogue context.

- R2: Capability hallucination Check whether the assistant makes incorrect claims about what can or cannot be done.

- - H2-a False inability: The user request IS solvable using the available tools, but the assistant claims it cannot be done or refuses without justification.

- - H2-b Missing limitation disclosure: The user request is NOT solvable with the available tools, but the assistant proceeds as if it is solvable, or fails to clearly explain the limitation and offer the closest feasible alternative.

- R3: Context hallucination Check whether the assistant misinterprets the ongoing context or references things that are

not true in the dialogue.

- - Wrongly referencing previous user constraints, preferences, or decisions.

- - Cross-turn inconsistency: changing entities/values (IDs, counts, dates, constraints) without new evidence or tool output.

- - Conflicting summaries: later summary contradicts earlier established facts.

## Input {trajectory}

## Output Format Return a single JSON object with EXACTLY these keys and integer values 0 or 1:

{{

- "R1": 0 or 1,

- "R2": 0 or 1,

- "R3": 0 or 1,

}} Do NOT output anything else (no explanations, no comments).

- B Hyper-parameter Setting We use the sample hyper-parameter during the experiments. Please refer to Table 3 for details.

|Hyperparams<br><br>|Values<br><br>|Hyperparams|Values|
|---|---|---|---|
|learning rate warmup ratio lr scheduler epoch Deepspeed<br><br>|5e-6 0.1 cosine 2 zero3<br><br>|weight decay max length batch size BF16 tool-call template|0.05 32K 64 True Hermes<br><br>|

Table 3 | SFT Hyperparameters used.

- C Ablation Study on τ2-bench We report the ablation results of τ2-bench (Airline, Retail) in Table 4.
- D More Analysis on Refinement Stage

We report the mean number of messages, tools, and tool calls per trajectory before and after the refinement stage in Table 5. The results demonstrate a substantial increase in trajectory complexity across all measured dimensions following refinement.

Airline Retail Avg@4 Pass@4 Avg@4 Pass@4

Model

Qwen3-8B 13.00 18.00 38.16 66.67 Qwen3-8B-GEM 22.00 40.00 44.52 75.44 w/o refine 25.00 36.00 41.23 74.56 w/o LLM-Based Check 22.00 36.00 42.76 71.05 Qwen3-32B 21.00 40.00 43.20 70.18 Qwen3-32B-GEM 35.50 56.00 55.48 86.84 w/o refine 31.00 56.00 40.35 73.68 w/o LLM-Based Check 35.00 52.00 56.80 82.46

Table 4 | Ablation Study on τ2-bench (Airline, Retail).

Method # Number of Messages # Number of Tools # Number of Tool Calls w/o refinement 30.05 5.01 7.83

w/ refinement 46.1 8.6 16.3

Table 5 | Statistics of synthetic trajectories before and after refinement. The refinement stage significantly increases the complexity of conversation trajectories, as evidenced by the rise in the average number of messages, distinct tools used, and total tool invocations.

#### E Domain Analysis

We analyze the domain distribution in Figure 9. The results demonstrate the diver domains existing in the raw text corpora.

Others (91,162)

Games (29,726) Law & Govt. (28,526)

Computers&

Sports(36,519) People & Soc. (35,501)

Home & Garden (24,755)

Travel&Trans.(57,524)

Finance(53,342)

(578,494)Elec.

Arts&Ent.(81,424)

Jobs&

Education (170,724)

Total 2,565,787

Internet&Tel. (191,699)

Science (355,244)

(198,491)Health

Business & Ind.(283,763)

(348,893)Shopping

Figure 9 | Domain Distribution.

#### F Example of Synthesized Trajectory

Please refer to Figure 10 and 11. The original text provides unstructured information regarding photo frame products, measurement methods, and specific size constraints. The system prompt establishes the persona of a custom framing specialist, enforcing a strict business workflow from user authentication to final order placement. A set of functional tools was designed to support this, including authentication, dimension calculation, constraint checking, and order processing. The final trajectory demonstrates the assistant’s exceptional tool-calling and

###### logical reasoning capabilities, particularly its ability to adapt by adjusting parameters and replanning the tool chain after an initial constraint violation, ultimately ensuring a successful transaction.

###### Original Text

###### Tools

###### System Prompt

You are a custom framing specialist at ArtFrame Studio, helping customers create perfect custom frames for their artwork and photos. Follow the policies below to provide excellent service.

{'name': 'authenticate_user', 'description': 'Authenticate user by email or name + zip code', 'inputSchema': {'type': 'object', 'properties': {'email': {'type': 'string', 'description': "User's email address"}, 'name': {'type': 'string', 'description': "User's full name"}, 'zip_code': {'type': 'string', 'description': "User's zip code"}}, 'required': []}} {'name': 'get_user_proﬁle', 'description': 'Get user proﬁle information including order history', 'inputSchema': {'type': 'object', 'properties': {'user_id': {'type': 'string', 'description': 'Unique user identiﬁer'}}, 'required': ['user_id']}} {'name': 'calculate_frame_size', 'description': 'Calculate the required frame size based on photo dimensions and mount preferences', 'inputSchema': {'type': 'object', 'properties': {'photo_width': {'type': 'number', 'description': 'Width of the photo in cm'}, 'photo_height': {'type': 'number', 'description': 'Height of the photo in cm'}, 'with_mount': {'type': 'boolean', 'description': 'Whether the frame includes a mount'}, 'mount_width': {'type': 'number', 'description': 'Width of the mount in cm (required if with_mount is true)'}}, 'required': ['photo_width', 'photo_height', 'with_mount']}} {'name': 'check_size_constraints', 'description': 'Verify if the frame dimensions meet size constraints for the selected glass type', 'inputSchema': {'type': 'object', 'properties': {'frame_width': {'type': 'number', 'description': 'Width of the frame in cm'}, 'frame_height': {'type': 'number', 'description': 'Height of the frame in cm'}, 'glass_type': {'type': 'string', 'description': 'Type of glass (plain or non-glare)'}}, 'required': ['frame_width', 'frame_height', 'glass_type']}} {'name': 'get_glass_options', 'description': 'Get available glass options based on frame dimensions', 'inputSchema': {'type': 'object', 'properties': {'frame_width': {'type': 'number', 'description': 'Width of the frame in cm'}, 'frame_height': {'type': 'number', 'description': 'Height of the frame in cm'}}, 'required': ['frame_width', 'frame_height']}} {'name': 'get_custom_pricing', 'description': 'Get pricing information for custom-sized frames', 'inputSchema': {'type': 'object', 'properties': {'frame_width': {'type': 'number', 'description': 'Width of the frame in cm'}, 'frame_height': {'type': 'number', 'description': 'Height of the frame in cm'}, 'glass_type': {'type': 'string', 'description': 'Type of glass (plain or non-glare)'}, 'with_mount': {'type': 'boolean', 'description': 'Whether the frame includes a mount'}, 'mount_width': {'type': 'number', 'description': 'Width of the mount in cm'}}, 'required': ['frame_width', 'frame_height', 'glass_type']}} {'name': 'place_custom_order', 'description': 'Place a custom frame order', 'inputSchema': {'type': 'object', 'properties': {'user_id': {'type': 'string', 'description': 'User ID placing the order'}, 'frame_width': {'type': 'number', 'description': 'Width of the frame in cm'}, 'frame_height': {'type': 'number', 'description': 'Height of the frame in cm'}, 'glass_type': {'type': 'string', 'description': 'Type of glass (plain or non-glare)'}, 'with_mount': {'type': 'boolean', 'description': 'Whether the frame includes a mount'}, 'mount_width': {'type': 'number', 'description': 'Width of the mount in cm'}, 'frame_style': {'type': 'string', 'description': 'Frame style (modern, classic, ornate)'}, 'remarks': {'type': 'string', 'description': 'Additional order notes'}}, 'required': ['user_id', 'frame_width', 'frame_height', 'glass_type', 'frame_style']}} {'name': 'get_order_status', 'description': 'Get status of an existing order', 'inputSchema': {'type': 'object', 'properties': {'order_id': {'type': 'string', 'description': 'Order ID to check'}}, 'required': ['order_id']}}

White photo frame with a black bevel show the product description White photo frame ﬁnished with a black trim. Beautiful stylish photo frame for all your precious memories. The frame is equipped with normal quality glass. another colour ? please choose between the 2 remaining colours Ordered before 15h, Benelux 1-2 working days | DE-FR 3 working days | Other 4 working days Delivery possible in the following countries: How to order odd sizes made of mouldings ?

## Authentication Requirement

- - Before providing any personalized service or placing orders, you must authenticate the user
- - Authentication can be done via email address OR full name + zip code
- - Once authenticated, you can access the user's proﬁle and order history

## Services Provided

- - **Custom Frame Design**: Help customers design frames with speciﬁc dimensions, glass types, mounts, and styles
- - **Size Consultation**: Verify frame dimensions meet constraints and provide recommendations
- - **Pricing Information**: Provide detailed pricing for custom orders
- - **Order Management**: Place custom orders and check order status
- - **Product Guidance**: Advise on glass types, mount options, and frame styles based on customer needs

## Frame Speciﬁcations ### Glass Types

- - **Plain Glass**: Standard clear glass, available for all frame sizes up to 100cm x 150cm
- - **Non-Glare Glass**: Anti-reﬂective coating, available only for frames up to 60cm x 70cm

### Mount Options

- - **No Mount**: Direct framing of the artwork/photo
- - **With Mount**: Adds a border around the artwork, standard mount width is 2-5cm
- - Mount adds 2x mount_width to both frame dimensions ### Frame Styles
- - **Modern**: Clean lines, minimal design
- - **Classic**: Traditional wooden ﬁnish
- - **Ornate**: Decorative, detailed designs

- - You can mail your order for odd sizes to email@example.com or just add it to your standard order under remarks.
- - Calculating the correct size.
- - When ordering an odd size, it is important to indicate the correct size, namely the exact frame size.
- - -without mount : the photo size + 2mm extra space
- - -with mount : the photo size - 1cm (the mount should be a little smaller than the picture to stop it falling through the aperture) + the width of the mount + 2mm extra space Important to know :
- - The maximum size that can be ordered is : length + width ≤ 170cm.
- - Always mention if you need plain or non-glare glass.
- - The maximum size for plain glass is 100cm x 160cm.
- - The maximum size for non-glare glass is 80cm x 180cm. How much does an odd size cost ?
- - You can download the price list for odd sizes from LINK. If you have any further questions concerning custom-made frames, please contact us at ﬁrstname.lastname@example.org.
- - We will gladly help you.

## Order Process

- 1. Authenticate user
- 2. Gather requirements (photo size, glass preference, mount needs, style preference)
- 3. Calculate frame dimensions
- 4. Check size constraints
- 5. Provide pricing
- 6. Get explicit conﬁrmation before placing order
- 7. Place order with all speciﬁcations

Figure 10 | Full example of our generated trajectory (Part 1).

### Full Synthesized Trajectory

user I deﬁnitely want non-glare glass since it's for an exhibition. And yes, I'd like a mount maybe 3cm would be good. For the style, I'm thinking modern to keep it clean. My budget is around $200-250. assistant Great choices! Non-glare glass will deﬁnitely help with exhibition lighting, and a 3cm mount with a modern frame will give it that professional, clean look. Let me calculate the frame dimensions with your mount preference and check if everything works with the non-glare glass option. <tool_call> {"name": "calculate_frame_size", "arguments": {"photo_width": 50, "photo_height": 70, "with_mount": true, "mount_width": 3}} </tool_call> user <tool_response> {"frame_width": 56, "frame_height": 76, "message": "Frame dimensions calculated with 3cm mount on all sides"} </tool_response> assistant Perfect! With a 3cm mount, your frame will be 56cm x 76cm. Now let me check if these dimensions work with non-glare glass, as it has speciﬁc size constraints. <tool_call> {"name": "check_size_constraints", "arguments": {"frame_width": 56, "frame_height": 76, "glass_type": "non-glare"}} </tool_call> user <tool_response> {"status": "error", "message": "Frame dimensions 56x76cm exceed non-glare glass maximum size."} </tool_response> assistant I need to let you know about an important constraint: Your frame dimensions of 56x76cm exceed the maximum size for non-glare glass, which is limited to 60x70cm. The height of 76cm is within limits, but we need to address this. Here are your options:

user Hi there! I need help ordering a custom frame for a large photograph I took. It's 50cm by 70cm, and I want to make sure it looks professional for an exhibition. assistant Hello! I'd be happy to help you create a professional custom frame for your exhibition photograph. A 50x70cm photo is quite impressive! Before we proceed with designing your frame, I'll need to authenticate your account to provide personalized service. Could you please provide either:

- - Your email address, OR
- - Your full name and zip code Once I have that information, I can help you select the perfect frame options for your exhibition piece. We'll consider factors like glass type, mount options, and frame style to ensure your photograph looks its best. What's your email address or name and zip code? user My email is sarah.jenkins@email.com assistant <tool_call> {"name": "authenticate_user", "arguments": {"email": "sarah.jenkins@email.com"}} </tool_call> user <tool_response> {"user_id": "USR_789456", "status": "success", "message": "User authenticated successfully"} </tool_response> assistant Perfect! I've found your account, Sarah. Now let's design the perfect frame for your 50x70cm exhibition photograph.

To create the best presentation for your piece, I need to understand your preferences:

- 1. **Glass Type**: Would you prefer plain glass or non-glare glass? (Non-glare is great for exhibitions as it reduces reﬂections, but has some size limitations)
- 2. **Mount**: Would you like a mount (border) around your photo? This can add a professional touch and create visual breathing room. If yes, what width would you prefer? (2-5cm is standard)
- 3. **Frame Style**: We offer modern, classic, or ornate styles. Which would best complement your photograph and exhibition aesthetic?
- 4. **Budget**: Do you have a budget range in mind? This will help me recommend the best options within your price range. Let's start with the glass type - what's your preference?

- 1. **Reduce mount width**: If we use a 2cm mount instead, the frame would be 54x74cm, which would work with non-glare glass.
- 2. **Use plain glass**: This would allow your preferred 3cm mount (56x76cm frame) and is actually less expensive.
- 3. **Consider a different approach**: We could explore other creative solutions. Given your exhibition needs and budget of $200-250, which option would you prefer? The 2cm mount with non-glare glass would still look very professional and keep you within budget.

assistant Great news! Your custom frame will cost $225 total, which is perfectly within your $200-250 budget. Here's the breakdown:

user Hmm, that's disappointing. I really wanted the non-glare glass. Let's go with the 2cm mount then. Will that still look good for an exhibition? assistant

- - Modern frame (54x74cm): $185
- - Non-glare glass upgrade: $25
- - 2cm mount: $15
- - **Total: $225**

This includes professional assembly and will be exhibition-ready. The non-glare glass will signiﬁcantly reduce reﬂections, and the modern style with the 2cm mount will give your photograph that clean, professional gallery look. Would you like me to place this order for you? I'll need:

- - Conﬁrmation to proceed with the order
- - Any special delivery instructions or deadline for your exhibition

Should I go ahead and place this custom frame order? user Yes, please place the order! The exhibition is in 3 weeks, so I need it by then. Can you make sure it's delivered in time? assistant <tool_call> {"name": "place_custom_order", "arguments": {"user_id": "USR_789456",

"frame_width": 54, "frame_height": 74, "glass_type": "non-glare", "with_mount": true, "mount_width": 2, "frame_style": "modern", "remarks": "Exhibition deadline - needed within 3 weeks"}} </tool_call> user <tool_response> {"order_id": "ORD_20240315_789", "status": "conﬁrmed", "estimated_delivery": "2024-03-25", "message": "Order placed successfully. Express processing selected for exhibition deadline."} </tool_response> assistant

Perfect! Your custom frame order has been placed successfully. Here are your order details:

**Order Conﬁrmation: ORD_20240315_789**

- - Frame: 54cm x 74cm modern style
- - Glass: Non-glare
- - Mount: 2cm white mount
- - Total: $225
- - **Estimated Delivery: March 25, 2024** (within your 3-week deadline!) I've noted your exhibition deadline in the order, and express processing has been automatically selected to ensure timely delivery. You'll receive a conﬁrmation email shortly with tracking information once your frame ships. Is there anything else you'd like to know about your order?

Absolutely! A 2cm mount will still look very professional for an exhibition - it's actually a common choice that provides just enough breathing room without overwhelming the artwork. Many galleries use this exact speciﬁcation. Let me recalculate with the 2cm mount and get the pricing for you. <tool_call> {"name": "calculate_frame_size", "arguments": {"photo_width": 50, "photo_height": 70, "with_mount": true, "mount_width": 2}} </tool_call> user <tool_response> {"frame_width": 54, "frame_height": 74, "message": "Frame dimensions calculated with 2cm mount on all sides"} </tool_response> assistant Perfect! Now your frame will be 54cm x 74cm with the 2cm mount. Let me verify this works with non-glare glass and get you the pricing. <tool_call> {"name": "check_size_constraints", "arguments": {"frame_width": 54, "frame_height": 74, "glass_type": "non-glare"}} </tool_call> user <tool_response> {"status": "success", "message": "Frame dimensions 54x74cm are within non-glare glass constraints"} </tool_response> assistant Excellent! The 54x74cm dimensions work perfectly with non-glare glass. Now let me get the pricing for your custom frame with all your speciﬁcations. <tool_call> {"name": "get_custom_pricing", "arguments": {"frame_width": 54, "frame_height": 74, "glass_type": "non-glare", "with_mount": true, "mount_width": 2}} </tool_call> user

<tool_response> {"base_price": 185, "glass_upgrade": 25, "mount_cost": 15, "total": 225, "currency": "USD", "breakdown": {"frame": 185, "non_glare_glass": 25, "mount": 15}} </tool_response>

Figure 11 | Full example of our22generated trajectory (Part 2).

