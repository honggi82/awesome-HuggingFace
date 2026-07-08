## arXiv:2501.10893v1[cs.LG]18Jan2025

[Figure 1]

# Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments

###### Hongjin Su1 2 *, Ruoxi Sun1, Jinsung Yoon1, Pengcheng Yin1, Tao Yu2 and Sercan Ö. Arık1

1Google, 2The University of Hong Kong

Autonomous agents powered by large language models (LLMs) have the potential to enhance human capabilities, assisting with digital tasks from sending emails to performing data analysis. The abilities of existing LLMs at such tasks are often hindered by the lack of high-quality agent data from the corresponding environments they interact with. We propose Learn-by-interact, a data-centric framework to adapt LLM agents to any given environments without human annotations. Learn-byinteract synthesizes trajectories of agent-environment interactions based on documentations, and constructs instructions by summarizing or abstracting the interaction histories, a process called backward construction. We assess the quality of our synthetic data by using them in both training-based scenarios and training-free in-context learning (ICL), where we craft innovative retrieval approaches optimized for agents. Extensive experiments on SWE-bench, WebArena, OSWorld and Spider2-V spanning across realistic coding, web, and desktop environments show the effectiveness of Learn-by-interact in various downstream agentic tasks — baseline results are improved by up to 12.2% for ICL with Claude3.5 and 19.5% for training with Codestral-22B. We further demonstrate the critical role of backward construction, which provides up to 14.0% improvement for training. Our ablation studies demonstrate the efficiency provided by our synthesized data in ICL and the superiority of our retrieval pipeline over alternative approaches like conventional retrieval-augmented generation (RAG). We expect that Learn-by-interact will serve as a foundation for agent data synthesis as LLMs are increasingly deployed at real-world environments.

### 1 Introduction

Pre-trained large language models (LLMs) offer great potential for assisting humans with various tasks in digital settings, such as editing images, performing data analysis, resolving software engineering issues, and navigating commercial platforms (Jimenez et al., 2023; Xie et al., 2023, 2024; Yao et al., 2022a). By streamlining these, LLM agents can greatly enhance human efficiency and productivity, allowing users to shift their focus toward higher-level, creative, and strategic endeavors. To explore this potential, many benchmarks (Cao et al., 2024; Jimenez et al., 2023; Koh et al., 2024; Xie et al., 2024; Zhou et al., 2023b) and agentic frameworks (Chen et al., 2024a; Gur et al., 2023; Yang et al., 2024, 2023; Zhan and Zhang, 2023) have been established based on realistic digital environments, spanning web applications, code development, desktop computing, etc. However, LLMs often fall short of expected performance in these tasks, consistently displaying a significant gap compared to human capabilities. As a result, they remain less practical and reliable for real-world applications.

Efficient adaptation to new environments can be a key part of the performance improvements. Prior works have explored various prompt-based approaches (Gur et al., 2023; Yang et al., 2024; Yao et al., 2022b; Zhan and Zhang, 2023), that are constrained by the capabilities of underlying foundation models. Other studies on training LLMs with human-labeled examples (Chen et al., 2023,

Corresponding author(s): hjsu@cs.hku.hk

* This work was done while Hongjin was a student researcher at Google Cloud AI Research.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

- Figure 1 | Overview of the data synthesis and adaptation processes. Given an environment and standard resources, we first leverage self-instruct to create a diverse set of instructions. LLMs are then employed to complete these tasks, resulting in long trajectories of agent-environment interactions. We construct task instructions using LLMs for each sub-trajectory, a process called backward construction. The synthesized data are then filtered and used for both training and in-context learning, where we design agentic retrieval to retrieve demonstration examples based on information at each step, using both model-based and observation-based approaches. See Appendix E for the complete data synthesis example and Algorithm 2 for more details on agentic retrieval.

2024b; Li et al., 2020) on the other hand, come with the fundamental limitation of high annotation costs when new environments are considered. In particular, annotating agentic data can be quite difficult and expensive due to long-trajectory interactions with environments and specific domain expertise required. Few works have explored fully-autonomous data construction pipelines towards self-adaptive agents that can efficiently learn new environments (Aksitov et al., 2023; Gulcehre et al., 2023).

In this paper, we introduce Learn-by-interact, a data-centric framework for LLMs to selfadapt to new environments, utilizing agent data synthesis via interactions. Intuitively, the effects of actions executed in environments (e.g., the next webpage after clicking a button) serve as informative demonstrations that help LLMs in future navigation. Inspired by this, we design Learn-byinteract that first uses self-instruct (Wang et al., 2022b) to develop a variety of task instructions, referring to standard resources such as documentations and tutorials for a given environment. This covers most important scenarios that human users are interested in and avoids intensive prompt engineering to control the distribution and diversity of the generated data. We then collect diverse trajectories from interactions between LLMs and environments, as illustrated in Fig. 1. However, given the low performance of LLMs in existing agentic benchmarks (Cao et al., 2024; Xie et al., 2024), it is likely that a large percentage of synthesized trajectories would not match with the instructions. To tackle this challenge, we construct new instructions by summarizing or abstracting each sub-trajectory, leveraging the strong summarization capabilities of LLMs (Liu et al., 2023; Pu et al., 2023). We call this process backward construction. After obtaining synthesized instruction-trajectory pairs and filtering low-quality ones, we apply it to both training and ICL, where we craft innovative retrieval pipelines optimized for agents. Specifically, the approach comprises two components: (1) a model-based approach where LLMs generate queries guided by instructions, interaction histories, and current observations, followed by retrieval models selecting demonstration examples from synthesized data; and (2) an observation-based approach that identifies examples in which the current observation appears in trajectories, signaling that the current state was encountered during the data synthesis

process. Our comprehensive evaluations across four challenging benchmarks: SWE-bench (Jimenez et al.,

- 2023), WebArena (Zhou et al., 2023b), OSWorld (Xie et al., 2024), and Spider2-V (Cao et al., 2024), highlight the efficacy of the data generated by Learn-by-interact. With ICL, both Gemini-1.5pro (Reid et al., 2024) and Claude-3.5-sonnet (Anthropic, 2024) show consistent and remarkable improvements – for OSWorld (Xie et al., 2024), our generated data nearly doubles Claude-3.5-sonnet’s baseline performance, increasing it from 12.4% to 22.5%. This enables Learn-by-interact to achieve the best-class performance in all of the four leaderboards. Furthermore, substantial improvements are observed by training models of varying sizes and architectures with our synthesized data. As an example, Codestral-22B’s (Team, 2024b) performance on WebArena significantly increases from 4.7% to 24.2% after training. These results underscore the high quality of the generated agentic data and the broad applicability across diverse environments.

Our extensive ablation studies reveal that backward construction not only increases the quantity of the synthesized data, but also improves its overall quality (§3.5). With data synthesized by Learnby-interact, we observe significant improvements in both performance and efficiency during LLM inference (§4.1). Our empirical results demonstrate the superiority of the agentic retrieval in ICL (§4.2). We anticipate that this research will spark innovative developments in enhancing agentic performance using LLMs and contribute to its wider-spread adoption in real-world application scenarios.

### 2 Learn-by-interact

We introduce the proposed Learn-by-interact framework to synthesize agent data in an autonomous way by leveraging interactions between LLMs and environments. We first formalize the canonical agentic tasks (§2.1), and introduce the detailed synthesis (§2.2) and filtering (§2.3) procedures. We then describe the application of the synthesized data in adapting LLMs in both training-free and training-based settings (§2.4).

#### 2.1 Task formulation

Given an environment 𝐸 and a task instruction 𝐼, the objective of an agent 𝐴 is to achieve the target 𝐺 through multi-step interactions with 𝐸. At each step 𝑖, 𝐴 predicts the next action 𝑎𝑖 based on the instruction 𝐼 and the previous history 𝐻 = (𝑜0, 𝑎1, 𝑜1, 𝑎2, ..., 𝑜𝑖−1), which is then executed in the environment 𝐸 to get a new observation 𝑜𝑖. The interactions terminated until 𝐴 predicts the action 𝑠𝑡𝑜𝑝 or the maximum number of steps 𝑚 is reached.

#### 2.2 Agentic data synthesis

The essential idea of Learn-by-interact is manifested in synthesizing environment-specific agent data with zero human effort. In Algorithm 1, we show the overall process with pseudo-code. Given an environment for a downstream application (e.g., Visual Studio Code), we first leverage commonly-accessible resources such as documentation to generate diverse task instructions using self-instruct (Wang et al., 2022b) (line 5). These resources are usually created by human experts to address common concerns and provide usage suggestions, e.g., how to navigate a website or operate a software. Intuitively, such references often cover representative use cases of an application. Therefore,

the task instructions generated conditioned on them could cover most popular scenarios in the domain and avoid potentially unreasonable cases that may be of less value.

For each generated task, LLMs then aim to solve it, which results in a long trajectory 𝑇 = (𝑜0, 𝑎1, 𝑜1, ..., 𝑎𝑛, 𝑜𝑛) (line 9-14 in Algo-

Algorithm 1 Agent data synthesis

- 1: Input: 𝐿𝐿𝑀: Large Language Model; 𝐸: environment; 𝐷𝑜𝑐: standard resources like documentation; 𝑁: the number of instructions to generate per document; 𝐹: data filter.
- 2: Initialization: 𝐷 = []: synthesized data.
- 3: for 𝑑 in 𝐷𝑜𝑐 do
- 4: // self-instruct to generate 𝑁 task instructions
- 5: 𝐼𝑛𝑠𝑡𝑟𝑢𝑐𝑡𝑖𝑜𝑛𝑠 = 𝐿𝐿𝑀(𝑑, 𝑁)
- 6: for 𝐼 in 𝐼𝑛𝑠𝑡𝑟𝑢𝑐𝑡𝑖𝑜𝑛𝑠 do
- 7: 𝐸.reset()
- 8: 𝑇 = [] // initialize interaction trajectory
- 9: while not 𝐸.finished() do
- 10: 𝑜 = 𝐸.get_observation()
- 11: 𝑎 = 𝐿𝐿𝑀(𝐼,𝑇, 𝑜)
- 12: 𝑇 += [𝑜, 𝑎]
- 13: end while
- 14: 𝑇.𝑎𝑝𝑝𝑒𝑛𝑑(𝐸.get_observation())
- 15: // backward construction
- 16: for 𝑖 in range(0, 𝑙𝑒𝑛(𝑇) − 1, 2) do
- 17: for 𝑗 in range(𝑖 + 2, 𝑙𝑒𝑛(𝑇), 2) do
- 18: 𝑇′ = 𝑇[𝑖 : 𝑗]
- 19: 𝐼′ = 𝐿𝐿𝑀(𝑇′)
- 20: 𝐷.𝑎𝑝𝑝𝑒𝑛𝑑([𝐼′,𝑇′])
- 21: end for
- 22: end for
- 23: end for
- 24: end for
- 25: 𝐷 = 𝐹(𝐷) // Filter low-quality data
- 26: Return: 𝐷

- rithm 1). To address the potential misalignment between the instruction 𝐼 and the generated trajectories 𝑇, we introduce a novel mechanism, backward construction, to construct instructions based on trajectories (lines 15-22 in Algorithm 1). Specifically, for each sub-trajectory

𝑇′ = (𝑜𝑖, 𝑎𝑖+1, 𝑜𝑖+1, ..., 𝑎𝑗, 𝑜𝑗), 0 ≤ 𝑖 < 𝑗 ≤ 𝑛, we obtain two types of new instructions: (1) summaries of trajectory steps; and (2) abstractions of the trajectory purpose. In Fig. 1, the subtrajectory (𝑂𝑏𝑠1, 𝐴𝑐𝑡2, 𝑂𝑏𝑠2) is summarized into a new task instruction that requires to replicate the 𝐴𝑐𝑡2. The abstraction of the full trajectory updates the original task objective, which is no longer aligned with the generated trajectory due to the wrong prediction in the action 3. Overall, the Learn-by-interact pipeline offers two notable advantages: (1). It corrects the potential misalignment between instructions and predicted trajectories by updating task objectives, which enhances the data quality as verified by the experimental results in §3.5. (2). It maximizes the utility of each generated trajectory by crafting new instructions for each sub-trajectory. This results in a quadratic increase in the number of synthesized examples with respect to the steps in the sequence per generated trajectory. For a given target dataset size, backward construction substantially decreases the necessary interactions, which is particularly valuable in scenarios where such interactions are challenging and costly to obtain such as Robotics (Keipour, 2022).

#### 2.3 Filtering

To further enhance the data quality, we design the following criteria to filter inferior synthesized data: (1). Remove duplicate states: We remove duplicate (𝑎𝑖, 𝑜𝑖) from 𝑇′ if (𝑎𝑖, 𝑜𝑖)=(𝑎𝑖−1, 𝑜𝑖−1), which is potentially introduced by the invalid action or the environment error (inactivity). (2). LLM committee check: We feed the generated instruction-trajectory pair (𝐼′,𝑇′) into a committee of LLMs, and only classify it of high-quality if all LLMs consider the trajectory coherent, natural, reasonable and aligned with the instruction. The listed criteria are all fully-autonomous and canonically-applicable for filtering data synthesized in general agent scenarios. Additionally, we employ iterative prompting to augment LLMs with high-quality examples to enhance their capabilities in data generation. See Table 31 for our prompts used in LLM committee check.

#### 2.4 Adaptation

After obtaining the synthesized data 𝐷, we apply it to both ICL and training. Given the unique characteristics of multi-round interactions with environments in agent settings, we design agentic retrieval (pseudo-code in Algo-

- rithm 2) to maximize the effectiveness of the synthesized data. Specifically, we propose two retrieval pipelines: observation-based (line 5-

14) and model-based retrieval (line 15-17). In observation-based retrieval, we compare the current observation 𝑜 to the trajectory of each example 𝑒 in the synthesized data, where 𝑒 = [𝐼′, [𝑜0, 𝑎1, 𝑜1, ..., 𝑎𝑛, 𝑜𝑛]]. If 𝑜 matches one of the observations in 𝑒, i.e., 𝑜 = 𝑜𝑖, then we consider 𝑒 as a helpful example to the current task. For the model-based retrieval, we leverage LLMs to first write queries based on the instruction, the interaction history and the current observation (line 16), and then employ retrieval models to retrieve non-duplicate examples (line 17). LLMs are then augmented with the retrieved examples to predict the next action (line 18). Refer to Table 32 to 35 for prompts to write queries and predict actions.

Apart from using the synthesized data as demonstration examples in ICL, we further utilize them to fine-tune models. For a given generated example, we convert it to the format of action prediction (Table 32), and prepare input-output pairs for supervised fine-tuning. More details on the experimental settings can be found in §3.3.

- 3 Experiments

Algorithm 2 ICL with agentic retrieval

- 1: Input: 𝐿𝐿𝑀: Large Language Model; 𝐸: environment; 𝐷: synthesized data; 𝐵𝑀25: BM25 retrieval model; 𝑅𝑀: dense retriever; 𝐼: task instruction; 𝑚1: maximum number of examples from observation-based retrieval; 𝑚2: maximum number of examples from model-based retrieval.
- 2: Initialization: 𝐻 = []: interaction history; 𝑅: retrieved examples.
- 3: while not 𝐸.finished() do
- 4: 𝑜 = 𝐸.get_observation()
- 5: // observation-based retrieval
- 6: 𝑅 = 𝐵𝑀25(𝑜, 𝐷, 𝑚1)
- 7: // model-based retrieval
- 8: 𝑞 = 𝐿𝐿𝑀(𝐼, 𝐻, 𝑜)
- 9: 𝑅 += 𝑅𝑀(𝑞, 𝐷, 𝑚2, 𝑅)
- 10: 𝑎 = 𝐿𝐿𝑀(𝐼, 𝐻, 𝑜, 𝑅)
- 11: 𝐻+ = [𝑜, 𝑎]
- 12: end while

#### 3.1 Baselines

We compare ICL with agentic retrieval to the following prompt-based approaches.

- • Baseline: The vanilla prediction pipeline in each benchmark that includes the task instruction, interaction history and the state observation in the prompt. See more implementation details in Appendix A.
- • RAG: The conventional RAG pipeline that first retrieves from the resources like documentation based on the instruction, and augments LLMs with the retrieved content.
- • Data distill: We follow the same pipeline to synthesize data in Algorithm 1 except backward construction (replace lines 15-22 with 𝐷.𝑎𝑝𝑝𝑒𝑛𝑑(𝐼,𝑇)), and follow Algorithm 2 during the evaluation.
- • Reflexion (Shinn et al., 2024): A general framework to reinforce language agents through linguistic feedback from both executors and LLMs.
- • Language Agent Tree Search (LATS) (Zhou et al., 2023a): It integrates the combinatorial tree search into expanding ReAct (Yao et al., 2022b) and combine agent online reasoning, acting and planning throughout the trajectory.

For the training-based evaluation, we primarily compare to the data distillation, which also constructs

SWE-bench WebArena OSWorld Spider2-V Documents 6,464 3,578 7,362 11,231

Raw trajectories 4,568 3,967 1,125 1,226

Examples 41,237 32,319 19,688 21,525 Filtered examples 10,232 10,456 11,782 10,169

- Table 1 | Statistics for the number of crawled documents, generated raw trajectories, examples (instruction-trajectory pairs) and examples after filtering.

data from scratch and requires no human effort to annotate seed or preference data. Additionally, we include the model performance before training as another baseline.

#### 3.2 Datasets

We consider the four agentic datasets that involve multi-round interactions with realistic environments. They span diverse domains of code, web, computer desktop and professional software. Appendix B illustrates details of each dataset with examples.

- • SWE-bench (Jimenez et al., 2023) is an evaluation benchmark on realistic software engineering problems from realistic Github issues. We use the verified version by default throughout the experiments.
- • Webarena (Zhou et al., 2023b) evaluates agent capabilities to perform tasks in the web environments such as e-commerce, social forum discussion, and beyond.
- • OSWorld (Xie et al., 2024) is an integrated environment for assessing open-ended computer tasks, which involve diverse applications like Terminal, Chrome, etc.
- • Spider2-V (Cao et al., 2024) is a multimodal agent benchmark focusing on professional data science and engineering workflows, which includes BigQuery, Airbyte and more.

#### 3.3 Settings

We synthesize one separate set of environment-specific data for each evaluated benchmark. Throughout the data synthesis process, we employ the Claude-3.5-sonnet (Anthropic, 2024) as the generator model and both Gemini-1.5-pro (Reid et al., 2024) and Claude-3.5-sonnet as the LLM committee for filtering low-quality data. For each document, we sample three task instructions from LLMs. The statistics for generated raw trajectories, examples before and after filtering are shown in Table 1. In Appendix D, we list document sources used for each benchmark. During ICL, we retrieve examples until the maximum length of LLMs and set an upper bound of 5 for both model-based and observationbased retrieval (𝑚1 = 5, 𝑚2 = 5 in Algorithm 2). We leverage Gemini-1.5-pro (Reid et al., 2024) and Claude-3.5-sonnet (Anthropic, 2024)1, Codegemma-7B (Team, 2024a) and Codestral-22B (Team,

- 2024b) in the ICL evaluation, and tune Codegemma-7B and Codestral-22B with LoRA (Hu et al.,

2021) to evaluate the data quality as training sources. By default, we do not include retrieval content in evaluating the trained model to avoid the confusion in understanding the effectiveness of our synthesized data in training. We include more detailed hyper-parameter settings (both existing approaches and Learn-by-interact) and machine information in Appendix C.

1In the subsequent descriptions, Gemini refers to Gemini-1.5-pro, and Claude refers to Claude-3.5-sonnet.

Benchmark → SWE Web OS Spider2-V SWE Web OS Spider2-V Approach ↓ Gemini-1.5-pro Claude-3.5-sonnet

Existing approaches

Baseline 13.3 17.9 4.9 8.3 51.2 35.8 12.4 8.4 RAG 13.7 19.5 5.1 9.1 51.8 36.9 12.8 9.2 Data distill 14.0 19.8 5.7 9.1 54.0 39.2 12.9 9.7 Reflexion 14.3 20.2 5.7 9.3 54.4 40.4 15.6 10.5 LATS 15.3 21.0 6.5 11.3 55.2 41.3 16.8 11.2

Ours

Learn-by-interact 18.7 25.6 10.3 16.4 60.0 48.0 22.5 16.6 Δ over baseline +5.4 +7.7 +5.4 +8.1 +8.8 +12.2 +10.1 +8.2

- Table 2 | Comparison of Learn-by-interact to other existing training-free approaches. SWE refers to SWE-bench, Web refers to WebArena and OS refers to OSWorld. The best results are highlighted in bold.

#### 3.4 Evaluation

We follow the default evaluation metrics designed by the original benchmarks. On SWE-bench (Jimenez et al., 2023), we apply the generated patch program to the repository codebase, and measure the agent performance by execution accuracy (pass@1). On WebArena (Zhou et al., 2023b), we employ both LLM-based fuzzy match and string match that checks keywords in predictions. Slightly different from the original work that uses gpt-4-0613 as the LLM judge, we use Claude-3.5-sonnet as a similar replacement. On OSWorld (Xie et al., 2024), we leverage the sample-specific evaluation scripts to assess the functional correctness of the task completion, which processes environment states and checks if agents finish the task as expected. On Spider2-V (Cao et al., 2024), we utilize file-based comparison, information-based validation, execution-based verification to determine whether a task is successfully completed. All performance numbers throughout the paper are shown in the percentage of resolved instances with % omitted for brevity.

#### 3.5 Results

##### 3.5.1 Training-free Evaluation

We first consider Learn-by-interact in the training-free setting, where the proposed methods can be applied to the commercial LLMs even with prediction-only API access.

Results on Table 2 show marginal improvement of RAG compared to the baseline, which suggests limited effectiveness by simply concatenating standard resources to LLM prompts. By retrieving examples from distilled data, we observe better performance compared to RAG, but still no more than 2% improvement over the baseline, which indicates that the distilled data tend to be noisy in the setting with multi-round agent-environment interactions. This highlights the critical role of backward construction, which corrects the misalignment between instructions and trajectories by curating new task objectives.

Both Reflexion and LATS consistently improve over the baseline across 4 benchmarks, which

Benchmark → Web OS Web OS Web OS Web OS Model → Codegemma-7B Codestral-22B Codegemma-7B Codestral-22B Approach ↓ Before tuning After tuning

Existing approaches

Baseline 3.3 0.0 4.7 2.2 - - - Data distill 4.2 0.0 5.8 2.7 6.2 1.4 10.2 5.4

Ours

Learn-by-interact 7.6 3.5 9.9 5.4 14.6 6.5 24.2 11.7 Δ over baseline +4.3 +3.5 +5.2 +3.2 +11.3 +6.5 +19.5 +9.5

- Table 3 | Downstream task performance of models trained from data generated by Learning-by-interact and data distillation. We include the models results before training, where the synthesized data is used as demonstration examples, and after training, where the synthesized data is used to train models.

demonstrate their general applicability to agent tasks. Using the data synthesized from the Learnby-interact, we can see a significant performance gain compared to all other frameworks in both Gemini and Claude. For example, on OSWorld, augmenting Claude with synthesized environmentspecific data almost doubles the result compared to the baseline. This signifies the high quality of the generated data and the effectiveness of the Learn-by-interact framework.

- 3.5.2 Training-based Evaluation

We consider the data synthesized by Learn-by-interact in the scenario of LLM tuning, which is applicable to the LLMs with access to weight updates.

The results presented in Table 3 reveal that Learn-by-interact substantially surpasses both the baseline and data distillation, suggesting its capacity to generate high-quality training data that enables language models to learn and adapt efficiently. We discover that utilizing our synthesized data for model training yields better results compared to using it as in-context learning (ICL) examples.

- A notable instance is in WebArena, where Codestral-22B’s performance jumps from 4.7% to 24.2% when trained on our synthesized data, while only showing a 5.5% improvement in the ICL scenario. Remarkably, the Codestral-22B model trained with our synthesized data even outperforms Gemini when the latter uses our data as demonstration examples.

- 4 Analysis

#### 4.1 Inference Efficiency

We compare the efficiency of different pipelines at inference. We analyze the trade-off between downstream task performance and the required computational costs. We focus on measuring the number of LLM calls and consumed tokens per example, which are averaged across four evaluated datasets (§3.2) using Claude-3.5-sonnet. As illustrated in Fig. 2, while Reflexion and LATS demonstrate

Performance

LLM calls

Consumed tokens

250k

38

40

35

36

200k

30

34

25

150k

32

20

30

15

100k

28

10

26

5

50k

Baseline RAG

Data distill Reflexion LATS Learn-by-interaction

| |
|---|

- Figure 2 | Evaluation performance, the number of LLM calls and consumed tokens (per example) of various training-free pipelines during inference, which are averaged across four benchmarks: SWE-bench, Webarena, OSWorld and Spider2-V.

Benchmark → SWE Web OS Spider2-V SWE Web OS Spider2-V Retrieval ↓ Gemini-1.5-pro Claude-3.5-sonnet

No retrieval 13.3 17.9 4.9 8.3 51.2 35.8 12.4 8.4 Instruction-based 14.7 21.6 7.0 10.2 52.4 36.6 15.0 9.6 Observation-based 16.3 23.5 8.7 14.6 53.6 42.5 17.2 10.5 Model-based 17.0 24.3 9.5 15.4 57.8 44.8 20.3 13.7 Ours 18.7 25.6 10.3 16.4 60.0 48.0 22.5 16.6

- Table 4 | Model performance based on different retrieval paradigms. Observation-based and Modelbased retrieval prove to be particularly effective in agent tasks, whose combination (ours) gives the best results.

enhanced performance, this comes at the cost of significantly increased computational resources during inference. Specifically, LATS achieves an average improvement of 2.5 %, albeit at the cost of requiring nearly four times more tokens per instance compared to the baseline. In contrast, Learn-by-interact exhibits superior performance while utilizing fewer LLM calls and slightly more tokens compared to the baseline. Thanks to the rich environment information stored in the examples of synthesized data, LLMs can potentially make better decisions and thus finish the task in fewer steps. This removes the performance-efficiency trade-off during inference at the cost of data synthesis in advance and suggests that Learn-by-interact is particularly well-suited for real-world deployment that demands both low latency and high performance.

#### 4.2 The Impact of Retrieval

As mentioned in §2.4, we employ both model-based and observation-based retrieval in our evaluation with ICL. We analyze their effectiveness by incorporating only one of them (skip lines 5-14 in Algorithm 2 for model-based retrieval only and skip lines 15-17 for observation-based retrieval only). In addition, we compare to two baselines: (1) no retrieval: LLMs predict each action in the zero-shot setting; and (2) instruction-based: only use instructions to retrieve synthesized data and apply the same demonstration examples in every action prediction throughout the trajectory.

The results presented in Table 4 illustrate how various retrieval methods impact LLMs when using the synthetic data as the retrieval source. Despite having access to the same example pool (except the baseline without using retrieval), there are notable differences in performance across different retrieval strategies, highlighting the crucial role of agentic retrieval in effectively utiliz-

Benchmark → SWE Web OS Spider2-V Web OS Granularity ↓ Claude-3.5-sonnet Codestral-22B

Baseline 51.2 35.8 12.4 8.4 4.6 2.2 Short 54.2 39.4 17.9 10.8 13.5 4.9 Medium 53.6 38.8 16.6 9.7 12.6 4.0 Long 52.2 37.6 15.2 9.2 10.6 3.4 Short+Medium 54.6 41.2 18.8 11.3 14.6 5.7 Short+Long 54.0 40.5 17.8 10.7 14.4 5.3 Medium+Long 53.8 38.6 17.2 10.4 13.2 4.5 Short+Medium+Long 55.0 42.0 19.8 12.3 15.4 6.3

- Table 5 | Effectiveness of synthetic data with various granularity. In general, short-trajectory data is more advantageous to both training and ICL, while mixing all of short, medium and long-trajectory data provides the best performance.

ing synthesized data. Conventional Retrieval-Augmented Generation (RAG) methods, which only employs instructions for retrieval, show the least improvement across four benchmarks and two LLMs. In contrast, the observation-based approach proves particularly effective for agent-based tasks, significantly outperforming the instruction-based retrieval, for instance, achieving a 4.4% absolute improvement on Spider-2V when using Gemini. By leveraging task instructions, interaction history and the current observation, model-based retrieval demonstrates even better results compared to using the observation-based version. Ultimately, the most impressive scores are achieved by combining both model-based and observation-based retrieval, which results in our agentic retrieval pipeline. These findings underscore the importance of carefully designing retrieval pipelines to maximize the potential of synthetic data and LLMs in agent scenarios.

#### 4.3 Data granularity

As mentioned in §2.2, we synthesize data by taking contiguous sub-trajectories from the full generation paths of LLMs, i.e. 𝑇′ = 𝑇[𝑖 : 𝑗], which results in trajectories of diverse lengths in the synthesized data. We divide the synthetic data into three groups: (1). trajectory steps < 5 (short); (2). 5 ≤ trajectory steps < 10 (medium); (3). trajectory steps ≥ 10 (long), and leverage each group and their combinations in both the training-free and the training-based process. To ensure a fair comparison, we constraint the data size in each group and combined group to 200M tokens2, utilizing Su et al. (2022) for sub-sampling. Table 5 presents the results. In both training-free and training-based evaluation, LLMs derive greater advantages from short-trajectory data, as demonstrated by its consistently superior performance compared to medium and long-trajectory data with Claude-3.5-sonnet and Codestral-22B. This can be attributed to the versatility of short-trajectory data, which usually serves as a sub-step or a partial workflow in downstream tasks. The combination of any two data groups proves more effective than relying on a single group, showcasing the complementary nature of diverse data sets. For instance, in Webarena with Codestral-22B, incorporating examples with both short and medium-length trajectories shows additional improvement over using either one exclusively. This underscores the value of considering the trajectory length as a unique dimension of agentic data synthesis.

2We use the number of tokens to measure the data size due to the fact that long-trajectory example may contain more information compared to the short version.

#### 4.4 Scaling Laws

We examine how the model performance improves as the synthetic data size scales up. Figure 3 presents two sets of results, with trainingfree (where Claude, Gemini, Codegemma and Codestral use retrieval augmentation without training) and with training-based (where finetuned Codegemma and Codestral models are evaluated without retrieval). All results are averaged across Webarena and OSworld due to computational resource constraints. The findings indicate that both learning paradigms benefit from larger data, suggesting the synthetic data is diverse and high-quality. In the training-free evaluation, more substantial improvements are observed for larger models (Claude and Gemini) compared to smaller ones (Codegemma and Codestral), possibly due to the their enhanced in-context learning abilities. Our analysis also reveals that for a given amount of synthetic data, fine-tuning smaller models is more effective than using the data as demonstration examples during evaluation.

Claude-3.5-sonnet

Gemini-1.5-pro

Codegemma-7B

Codestral-22B

Codegemma-7B-trained

Codestral-22B-trained

35

30

25

Performance

20

15

10

5

0

0 2k 4k 6k 8k 10k Synthesized data size

Figure 3 | Scaling laws for the synthesized data. Compared to in-context learning, tuning achieves more significant improvements as the data scales up. The performance is averaged across WebArena and OSWorld.

### 5 Related work

Various agents based on LLMs have been developed (Huang et al., 2022; Shinn et al., 2024; Wang et al., 2023a, 2024a, 2023b; Zhang et al., 2024). React (Yao et al., 2022b) proposes to synergize reasoning and acting in LLMs. By integrating Monte Carlo Tree Search (Coulom, 2006; Kocsis and Szepesvári, 2006), Zhou et al. (2023a) leverages LLMpowered value functions and self-reflection (Madaan et al., 2024) to encourage proficient exploration and decision-making. However, it comes with increased computational costs and relies on the premise that the environment allows for state reversals. In contrast, Learn-by-interact removes such assumptions and improves both agent efficiency and performance by synthesizing high-quality data in advance.

Another line of research to improve agent models relies on training on human-labeled examples (Chen et al., 2024b; Deng et al., 2024; Wang et al., 2022a; Yin et al., 2023; Zeng et al., 2023) or data distilled from LLMs like GPT-4 (Chen et al., 2023; Zhao et al., 2024). AgentGen (Hu et al., 2024) explores automatic synthesis of both environments and tasks and then leverages FastDownward3 to generate trajectory data. AgentTuning (Zeng et al., 2023) utilizes both existing datasets and selfinstruct (Wang et al., 2022b) to derive instructions and then samples trajectories from GPT-4 (Achiam et al., 2023). In contrast, Learn-by-interact focuses on realistic environments and generate tasks and trajectories using backward construction. Some other researchers are also exploring ways to use data more efficiently with reinforcement learning (Ball et al., 2023; Nachum et al., 2018; Schwarzer et al., 2020, 2021; Thomas and Brunskill, 2016). Gulcehre et al. (2023) suggests using

3https://www.fast-downward.org/

data created by an LLM’s policy can enhance itself via offline reinforcement learning algorithms. Aksitov et al. (2023) takes this further by combining with ReAct (Yao et al., 2022b) to train agent models iteratively on experience trajectories. These typically require a reward model as the scoring function or LLM/execution-generated feedback to enhance data quality. Our work, however, takes a different approach by employing the backward construction to improve the data quality by aligning instructions and trajectories.

### 6 Conclusion

We introduce Learn-by-interact, a data-centric framework to adapt LLM agents to any given environments without human annotations. Based on commonly-accessible resources like documentaion, LLMs propose downstream tasks and complete them with multi-round interactions with environments. We address the misalignment between instructions and trajectories by updating objectives with new instructions derived from trajectories. Additionally, we design innovative retrieval approaches that leverage agent instructions, interaction histories, and current observations to retrieve synthesized examples. Through extensive experiments, we demonstrate that the synthetic data from Learn-byinteract significantly enhances model performance with both ICL and training. Compared with other leading approaches in agent tasks, Learn-by-interact shows much better performance with lower latency and computational costs, which make it particularly suitable for large-scale deployment. Further analysis has also shown the superiority of Learn-by-interact over the classical RAG. In future work, we plan to explore multi-modal settings and train general agent models widely applicable in realistic environments. We anticipate that Learn-by-interact will inspire future research to push the state-of-the-art in this direction.

### 7 Limitations

Although Learn-by-interact effectively synthesizes high-quality agentic data with trajectories, it requires a lot of LLM calls in generation and filtering. We hope that future works will explore more efficient approaches to complete annotations without sacrificing quality. Additionally, Learn-byinteract leverages the environment-related resources to generate instructions. In some scenarios, however, these resources may be incomplete or not available.

### References

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,

- S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

R. Aksitov, S. Miryoosefi, Z. Li, D. Li, S. Babayan, K. Kopparapu, Z. Fisher, R. Guo, S. Prakash, P. Srinivasan, et al. Rest meets react: Self-improvement for multi-step reasoning llm agent. arXiv preprint arXiv:2312.10003, 2023.

Anthropic. Introducing claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/ claude-3-5-sonnet.

P. J. Ball, L. Smith, I. Kostrikov, and S. Levine. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, pages 1577–1594. PMLR, 2023.

R. Cao, F. Lei, H. Wu, J. Chen, Y. Fu, H. Gao, X. Xiong, H. Zhang, Y. Mao, W. Hu, et al. Spider2-v: How far are multimodal agents from automating data science and engineering workflows? arXiv preprint arXiv:2407.10956, 2024.

- B. Chen, C. Shu, E. Shareghi, N. Collier, K. Narasimhan, and S. Yao. Fireact: Toward language agent fine-tuning. arXiv preprint arXiv:2310.05915, 2023.

- D. Chen, S. Lin, M. Zeng, D. Zan, J.-G. Wang, A. Cheshkov, J. Sun, H. Yu, G. Dong, A. Aliev, et al. Coder: Issue resolving with multi-agent and task graphs. arXiv preprint arXiv:2406.01304, 2024a.

Z. Chen, K. Liu, Q. Wang, W. Zhang, J. Liu, D. Lin, K. Chen, and F. Zhao. Agent-flan: Designing data and methods of effective agent tuning for large language models. arXiv preprint arXiv:2403.12881, 2024b.

- R. Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In International conference on computers and games, pages 72–83. Springer, 2006.

- X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36, 2024.

A. Drouin, M. Gasse, M. Caccia, I. H. Laradji, M. Del Verme, T. Marty, D. Vazquez, N. Chapados, and A. Lacoste. WorkArena: How capable are web agents at solving common knowledge work tasks? In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 11642–11662. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/drouin24a.html.

C. Gulcehre, T. L. Paine, S. Srinivasan, K. Konyushkova, L. Weerts, A. Sharma, A. Siddhant, A. Ahern, M. Wang, C. Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

- I. Gur, H. Furuta, A. Huang, M. Safdari, Y. Matsuo, D. Eck, and A. Faust. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023.

- E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

M. Hu, P. Zhao, C. Xu, Q. Sun, J. Lou, Q. Lin, P. Luo, S. Rajmohan, and D. Zhang. Agentgen: Enhancing planning abilities for large language model based agent via environment and task generation. arXiv preprint arXiv:2408.00764, 2024.

- W. Huang, P. Abbeel, D. Pathak, and I. Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022.

C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

A. Keipour. Physical interaction and manipulation of the environment using aerial robots. arXiv preprint arXiv:2207.02856, 2022.

L. Kocsis and C. Szepesvári. Bandit based monte-carlo planning. In European conference on machine learning, pages 282–293. Springer, 2006.

- J. Y. Koh, R. Lo, L. Jang, V. Duvvur, M. C. Lim, P.-Y. Huang, G. Neubig, S. Zhou, R. Salakhutdinov, and D. Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv e-prints, pages arXiv–2401, 2024.

- Y. Li, J. He, X. Zhou, Y. Zhang, and J. Baldridge. Mapping natural language instructions to mobile ui action sequences. arXiv preprint arXiv:2005.03776, 2020.

- Y. Liu, K. Shi, K. S. He, L. Ye, A. R. Fabbri, P. Liu, D. Radev, and A. Cohan. On learning to summarize with large language models as references. arXiv preprint arXiv:2305.14239, 2023.

A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye,

- Y. Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36, 2024.

O. Nachum, S. S. Gu, H. Lee, and S. Levine. Data-efficient hierarchical reinforcement learning. Advances in neural information processing systems, 31, 2018.

- X. Pu, M. Gao, and X. Wan. Summarization is (almost) dead. arXiv preprint arXiv:2309.09558, 2023. M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou,

- O. Firat, J. Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

M. Schwarzer, A. Anand, R. Goel, R. D. Hjelm, A. Courville, and P. Bachman. Data-efficient reinforcement learning with self-predictive representations. arXiv preprint arXiv:2007.05929, 2020.

- M. Schwarzer, N. Rajkumar, M. Noukhovitch, A. Anand, L. Charlin, R. D. Hjelm, P. Bachman, and A. C. Courville. Pretraining representations for data-efficient reinforcement learning. Advances in Neural Information Processing Systems, 34:12686–12699, 2021.
- N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.

H. Su, J. Kasai, C. H. Wu, W. Shi, T. Wang, J. Xin, R. Zhang, M. Ostendorf, L. Zettlemoyer, N. A. Smith, et al. Selective annotation makes language models better few-shot learners. arXiv preprint arXiv:2209.01975, 2022.

C. Team. Codegemma: Open code models based on gemma. arXiv preprint arXiv:2406.11409, 2024a. T. M. A. Team. Codestral: Hello, world!, 2024b. URL https://mistral.ai/news/codestral/.

- P. Thomas and E. Brunskill. Data-efficient off-policy policy evaluation for reinforcement learning. In International Conference on Machine Learning, pages 2139–2148. PMLR, 2016.

G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.

- R. Wang, P. Jansen, M.-A. Côté, and P. Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? arXiv e-prints, pages arXiv–2203, 2022a.

X. Wang, Y. Chen, L. Yuan, Y. Zhang, Y. Li, H. Peng, and H. Ji. Executable code actions elicit better llm agents. arXiv preprint arXiv:2402.01030, 2024a.

- X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, H. H. Tran, F. Li,

- R. Ma, M. Zheng, B. Qian, Y. Shao, N. Muennighoff, Y. Zhang, B. Hui, J. Lin, R. Brennan, H. Peng, H. Ji, and G. Neubig. OpenHands: An Open Platform for AI Software Developers as Generalist Agents, 2024b. URL https://arxiv.org/abs/2407.16741.

- Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022b.
- Z. Wang, S. Cai, G. Chen, A. Liu, X. Ma, and Y. Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023b.

T. Xie, F. Zhou, Z. Cheng, P. Shi, L. Weng, Y. Liu, T. J. Hua, J. Zhao, Q. Liu, C. Liu, et al. Openagents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023.

T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024.

J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press. Swe-agent: Agentcomputer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793, 2024.

- Z. Yang, J. Liu, Y. Han, X. Chen, Z. Huang, B. Fu, and G. Yu. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771, 2023.

- S. Yao, H. Chen, J. Yang, and K. Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744– 20757, 2022a.

- S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022b.

- D. Yin, F. Brahman, A. Ravichander, K. Chandu, K.-W. Chang, Y. Choi, and B. Y. Lin. Lumos: Learning agents with unified data, modular design, and open-source llms. arXiv preprint arXiv:2311.05657, 2023.

A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y. Dong, and J. Tang. Agenttuning: Enabling generalized agent abilities for llms. arXiv preprint arXiv:2310.12823, 2023.

- Z. Zhan and A. Zhang. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436, 2023.

J. Zhang, Y. Yu, M. Liao, W. Li, J. Wu, and Z. Wei. Ui-hawk: Unleashing the screen stream understanding for gui agents. arXiv preprint, 2024.

- Z. Zhao, K. Ma, W. Chai, X. Wang, K. Chen, D. Guo, Y. Zhang, H. Wang, and G. Wang. Do we really need a complex agent system? distill embodied agent into a single model. arXiv preprint arXiv:2404.04619, 2024.

A. Zhou, K. Yan, M. Shlapentokh-Rothman, H. Wang, and Y.-X. Wang. Language agent tree search unifies reasoning acting and planning in language models. arXiv preprint arXiv:2310.04406, 2023a.

- S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023b.

### A Baseline implementations

We follow the existing frameworks to set up baselines in each benchmark. In SWE-bench (Jimenez et al., 2023), we follow CodeAct (Wang et al., 2024b), where LLMs interact with environments to solve problems. In WebArena (Zhou et al., 2023b), we follow the implementation in Drouin et al. (2024), which concatenates task objectives, action space descriptions, general instructions (e.g., output formats) and webpage observations in the prompt, and ask LMs to predict the next action. By default, we use the accessibility tree4 as the observation space. In OSWorld (Xie et al., 2024) and Spider2V (Cao et al., 2024), we follow the original prompt style designed by the benchmark, which also concatenates task objectives, action space descriptions, general instructions and computer observations in the prompt. By default, we use the accessibility tree as the observation space for OSWorld, and use the set-of-mark for Spider2-V due to the significant information loss of the accessibility tree in the original benchmark. See an example in Table 18 and 19 for more details.

### B Dataset examples

From Table 8 to 17, we provide one example for each dataset with full instructions, interaction history with the environment.

### C Experimental settings

We retrieve documents until the maximum length of LLMs for RAG and set an upper bound number of 50 documents, where the retrieved documents remain unchanged throughout agent interaction trajectory because only instructions are used as the query for retrieval. For Reflexion (Shinn et al., 2024), we use the maximum trials 3. In LATS (Zhou et al., 2023a), we use the number of generated action 5, depth limit 15, value function weight 0.8, following the original setting in paper with WebShop (Yao et al., 2022a), which is also an agent task based on website. By default, we use https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings as the dense retriever for model-based retrieval. We use the temperature 0 throughout the experiments to ensure better reproductivity of the experiments. During training, we the batch size 128, learning rate 0.00002, warmup ratio 0.03 and maximum length 8192, and tune the model for 3 epochs. All experiments are conducted in H100 machines with 80GB memeory.

### D Document sources

We use all the non-repeated python files in SWE-bench-Verified (Jimenez et al., 2023) as the document sources. Although we may not always find abundant documentations and tutorials for each environment, we believe that documentations in the same domain still have a good coverage of frequent operations. For example, one subset of WebArena (Zhou et al., 2023b) focuses on the navigation of the shopping website OneStopMarket, we use the Amazon documentation as a good replacement. Regardless of the shopping websites, the frequent tasks usually include order change, product search, delivery checking, etc. Therefore, we use other documentations in the same domain to sample task

4https://developer.mozilla.org/en-US/docs/Glossary/Accessibility_tree

instructions when the exact version for the target environment is not available. Concretely, we use the following sources for WebArena:

- • https://docs.gitlab.com/ee/tutorials/
- • https://support.google.com/maps
- • https://www.amazon.com/hz/contact-us/foresight/hubgateway
- • https://support.reddithelp.com/hc/en-us/articles

The following sources are used for OSWorld:

- • https://support.google.com/chrome/?hl=en
- • https://www.gimp.org/tutorials/
- • https://books.libreoffice.org/en/CG72/CG72.html
- • https://books.libreoffice.org/en/WG73/WG73.html
- • https://ubuntu.com/tutorials/command-line-for-beginners
- • https://support.mozilla.org/en-US/products/thunderbird
- • https://wiki.videolan.org/Documentation:Documentation
- • https://code.visualstudio.com/docs

, The following sources are used for Spider2-V:

- • https://docs.getdbt.com/
- • https://release-1-7-2.dagster.dagster-docs.io/
- • https://docs.astronomer.io/
- • https://docs.airbyte.com/
- • https://airbyte.com/tutorials/
- • https://airbyte-public-api-docs.s3.us-east-2.amazonaws.com/rapidoc-api-docs.html
- • https://superset.apache.org/docs/
- • https://www.metabase.com/docs/v0.49/
- • https://www.metabase.com/learn/
- • https://docs.snowflake.com/en/
- • https://cloud.google.com/bigquery/docs/
- • https://jupyterlab.readthedocs.io/en/4.1.x/

### E Synthesized data examples

From Table 20 to 26, we provide a complete example of data synthesis. To begin with, an LLM generates instructions based on standard resources like tutorials, documentations and FAQs: Upload CSV data in Google Drive to BigQuery. (See prompt in Table 29) It then attempts solve the task by predicting actions and collecting feedback from environments (interactions). This produces a long trajectory showing how LLMs try to achieve the goal.

However, it is not guaranteed that the trajectory successfully achieves the target. In our example, the LLM makes a wrong prediction in the action 4. It selects the table source Google Cloud Storage, while the correct action should select “Drive" to align with the instruction that reuiqres to upload CSV data in Google Drive. This results in wrong actions in the subsequent predictions, and the generated trajectory is not aligned with the initial instruction, which leads to noisy data in this case.

Number of synthesized examples 0 5k 10k Task instructions generated based on environments 35.8 37.3 39.6 Task instructions generated based on related resources 35.8 43.2 48.0

- Table 6 | The comparison of Learn-by-interact to the version without replying on existing resources.

Instead of using the original instruction-trajectory pairs for downstream training and in-context learning, we fix the mentioned misalignment by crafting new instructions for each sub-trajectory (backward construction). Concretely, we feed the generated trajectory into LLM prompts, and ask it to summarize the trajectory or propose a new task based on it. For example, the LLM updates the task objective to “Link CSV file in Google Cloud Storage to BigQuery" after observing the trajectory, which makes the task instrucion and the trajectory aligned. Additionally, we also generate new instructions for each sub-trajectory, which would increase the utility of a generated full trajectory. For instance, based on the sub-trajectory (observation 0, Action 1, observation 1), the LLM generates a new instruction: When is dataset “demo" created? In Table 27 and 28, we list more generated instructions based on sub-trajectories.

### F Case study on filtered examples

In Table 36-45, we demonstrate the representative synthesized examples that fail to meet our designed criteria. The example in Table 36-41 is filtered because the trajectory shows detour in accomplishing the goal, i.e. Action 1-6 are not necessary. The example in Table 42-45 is filtered because it goes back and forth in states, i.e. repeat the actions of clicking "My Orders" and clicking "View Order". We filter these low-quality examples to avoid their negative influences in the downstream applications.

### G Synthesized data from environments

We compare Learn-by-interact with the version without relying on existing resources in WebArena. Except for sampling task instructions from LLMs based on given environments, we follow the same procedures in Learn-by-interact to synthesize 10k examples. The results of in-context learning with Claude-3.5-sonnet are shown in Table 6

However, we note the following potential concerns regarding the version without replying on existing resources: the distribution and the diversity of the generated data are hard to control. Without conditioning on prior documents, one will need intensive prompt engineering to guide LLMs in generating diverse task instructions. On the other hand, the related resources are usually crafted by experts or written by real users, which cover most important scenarios that people who interact with the environment are interested in.

### H Cross-website generalization

To evaluate the generalization capabilities of Learn-by-interact, in WebArena, we consider the content management systems (CMS) as a held-out test set and leverage the synthetic data from the

Model Claude-3.5-sonnet Codestral-22B Baseline 26.0 3.3 Learn-by-interact with synthetic data that excludes CMS 28.3 12.6 Learn-by-interact with all WebArena data that contains CMS 29.2 17.6

- Table 7 | Results for cross-website generalization.

remaining websites as the training data. To ensure a fair comparison and avoid the influences of the discrepancies in the training set size, we downsample the original data that covers all the WebArena domains so that both set contains the same number of instances. Following the same evaluation pipelines in the paper, we assess the performance of in-context learning with Claude-3.5-sonnet and training with Codestral-22B.

From the results in Table 7, we observe that, even without specifically using the data sampled from the CMS environment, Learn-by-interact demonstrates significant improvements over the baseline in both training and in-context learning. This indicates that the proposed approach holds the potential for cross-website generalization and is likely to achieve better performance when utilizing data from more websites.

Instruction Tell me the total cost of my latest cancelled order? Observation 0

[Figure 7]

Action 1 click [4918] // click the button ‘My Account’

- Table 8 | Webarena example

##### Observation 1

[Figure 8]

Action 2 Stop: The total cost of the latest cancelled order is $365.42

##### Table 9 | Webarena example cont.

Instruction Could you assist me in adding a new layer and naming it ‘Square’?

- Observation 0 (Interface of the software GIMP)

[Figure 9]

Action 1

import pyautogui pyautogui.hotkey(‘shift’, ‘ctrl’, ‘n’) // shortcut to initialize a new layer.

- Observation 1 (Interface of the software GIMP)

[Figure 10]

- Table 10 | OSWorld example

##### Action 2

import pyautogui pyautogui.typewrite(‘Square’) // change the layer name to Square.

Observation 2 (Interface of the software GIMP)

[Figure 11]

##### Action 3

import pyautogui pyautogui.click(1096, 892) // click OK to save the change.

- Observation 3 (Interface of the software GIMP)

[Figure 12]

- Table 11 | OSWorld example cont.

##### Instruction

I have established a connection from Faker to local .csv file. Could you help me change the running schedule? I hope it can be replicated at 18:00 pm every day.

##### Observation 0 (Interface of the software Airbyte)

[Figure 13]

- Action 1

import pyautogui pyautogui.click(550,280) // click the connection row with the name “Sample Data (Faker) → Local CSV"

##### Observation 1 (Interface of the software Airbyte)

[Figure 14]

##### Action 2

import pyautogui pyautogui.click(500,300) // click the button Replication

Observation 2 (Interface of the software Airbyte)

[Figure 15]

##### Action 3

import pyautogui pyautogui.click(500,450) // click the panel Configuration

##### Observation 3 (Interface of the software Airbyte)

[Figure 16]

##### Action 4

import pyautogui pyautogui.click(1270,430) // reveal the dropdown menu of the schedule type

Observation 4 (Interface of the software Airbyte)

[Figure 17]

##### Action 5

import pyautogui pyautogui.click(1200,565) // select the schedule type Cron

##### Observation 5 (Interface of the software Airbyte)

[Figure 18]

##### Action 6

import pyautogui pyautogui.click(1020,480) // focus on the content of the Cron expression

Observation 6 (Interface of the software Airbyte)

[Figure 19]

##### Action 7

import pyautogui pyautogui.hotkey(“ctrl",“a") // select all content in the textbox

##### Observation 7 (Interface of the software Airbyte)

[Figure 20]

##### Action 8

import pyautogui pyautogui.press(“delete") // delete the default content

Observation 8 (Interface of the software Airbyte)

[Figure 21]

##### Action 9

import pyautogui pyautogui.write(“0 0 18 * * ?") // update the content to align 18:00 pm in the instruction

##### Observation 9 (Interface of the software Airbyte)

[Figure 22]

##### Action 10

import pyautogui pyautogui.click(1450,900) // click the button save changes

- Observation 10 (Interface of the software Airbyte)

[Figure 23]

- Table 17 | Spider2-V example, cont.

Screenshot

[Figure 24]

Set-of-mark

[Figure 25]

- Table 18 | Observation space of Spider2-V.

[208, 13] menu Chromium Web Browser “" [1463, 13] menu System “" [35, 65] push-button Chromium Web Browser “" [753, 81] label Please download waiting software updates. “" [135, 109] label Home [35, 133] push-button Terminal “" [35, 201] push-button Visual Studio Code “" [35, 269] push-button Files “" [35, 337] push-button Text Editor “" [953, 370] label Updated software is available for this computer. Do you want to install it now? [35, 405] push-button LibreOffice Calc “" [951, 463] table-cell Security updates [1191, 463] table-cell 638.8 MB [35, 473] push-button LibreOffice Writer “" [963, 486] table-cell LibreOffice [1191, 486] table-cell 23.4 MB [963, 509] table-cell LibreOffice Calc [1191, 509] table-cell 8.7 MB [923, 524] toggle-button Details of updates “" [963, 532] table-cell LibreOffice Draw [1191, 532] table-cell 3.0 MB [35, 541] push-button Document Viewer “" [963, 555] table-cell LibreOffice Impress [1191, 555] table-cell 1.3 MB [963, 578] table-cell LibreOffice Math [1191, 578] table-cell 673 kB [35, 612] push-button Software Updater “" [935, 660] label 1157.8 MB will be downloaded. [35, 680] push-button Trash “" [671, 702] push-button Settings... “" [1054, 702] push-button Cancel “" [1176, 702] push-button Install Now “" [35, 884] toggle-button Show Applications “"

- Table 19 | Observation space of Spider2-V. The accessibility tree suffers from significant information loss. Compared to the screenshot and set-of-mark shown in Table 18, the presented accessibility tree fails to retrieve webpage information, and only shows the details of the desktop icons in the left panel.

Instruction Upload CSV data in Google Drive to BigQuery.

- Observation 0 (Bigquery Interface)

[Figure 26]

Action 1

import pyautogui pyautogui.doubleClick(332,447) // double click the dataset demo.

- Observation 1 (Bigquery Interface)

[Figure 27]

- Table 20 | Example of data synthesis - Bigquery

##### Action 2

import pyautogui pyautogui.doubleClick(1002,321) // double click the button CREATE TABLE.

Observation 2 (Bigquery Interface)

[Figure 28]

##### Action 3

import pyautogui pyautogui.click(1458,279) // click the dropdown menu to select table source.

##### Observation 3 (Bigquery Interface)

[Figure 29]

##### Action 4

import pyautogui // wrong prediction: click the table source Google Cloud Storage. // The correct prediction should select Drive to align with the instruction. pyautogui.click(1302,331)

Observation 4 (Bigquery Interface)

[Figure 30]

##### Action 5

import pyautogui pyautogui.click(1389,340) // click BROWSE to find file.

##### Observation 5 (Bigquery Interface)

[Figure 31]

##### Action 6

import pyautogui pyautogui.click(1341,282) // click to find files under directory.

Observation 6 (Bigquery Interface)

[Figure 32]

##### Action 7

import pyautogui pyautogui.doubleClick(1203,284) // select the CSV file.

##### Observation 7 (Bigquery Interface)

[Figure 33]

##### Action 8

import pyautogui pyautogui.click(1000,915) // click to select file.

Observation 8 (Bigquery Interface)

[Figure 34]

##### Action 9

import pyautogui pyautogui.click(1000,668) // click to focus on the text box Table.

##### Observation 9 (Bigquery Interface)

[Figure 35]

##### Action 10

import pyautogui pyautogui.write("test") // name the file "test".

Observation 10 (Bigquery Interface)

[Figure 36]

##### Action 11

import pyautogui pyautogui.click(1425,726) // click to reveal dropdown menu of the table type.

##### Observation 11 (Bigquery Interface)

[Figure 37]

##### Action 12

import pyautogui pyautogui.click(1297,801) // select the table source external table.

Observation 12 (Bigquery Interface)

[Figure 38]

##### Action 13

import pyautogui pyautogui.click(303,914) // click CREATE TABLE.

##### Observation 13 (Bigquery Interface)

[Figure 39]

sub-trajectory type instruction

- Observation 0 ↓

Action 1 New task When is dataset “demo" created?

↓

- Observation 1)

- Observation 1 Replicate the following: We are currently at the Google Cloud ↓ Console interface, specifically focused on a BigQuery project.

Action 2 Replicate trajectory The browser window displays details of a dataset named "demo"

↓ within a BigQuery project. The interface provides information

- Observation 2 about the dataset, including its creation date, last modified time,

data location (US), and other properties like default table expiry

and rounding mode. On the left side of the screen, there’s a

navigation panel showing the Explorer view with the "demo"

dataset selected. The top of the screen shows the Google Cloud

header with project selection and search functionality.

The overall layout is characteristic of a cloud-based data

management platform, with options to create tables, share data,

and manage dataset properties.

After taking the action to click the CREATE TABLE button,

we go to the user interface for creating a table. The screen displays a form titled "Create table" with various fields and options. The source section allows selecting a table to create

from, while the destination section includes fields for project,

dataset, and table name. There’s also a schema section and

partition and cluster settings. The interface is part of the Google

Cloud Console, as evident from the sidebar on the left showing

different Cloud services and project navigation.

- Observation 4 ↓

Action 5 ↓

- Observation 5 ↓

Action 6 ↓

- Observation 6 New task Select test.csv in the bucket test-1616 in Google Cloud Storage ↓ as the table source.

Action 7 ↓

- Observation 7 ↓

Action 8 ↓

- Observation 8

###### Table 27 | Instructions generated from trajectory from Table 20 to 26

sub-trajectory type instruction

- Observation 8 Replicate the following: We are in the the interface for creating ↓ a table in Google Cloud’s BigQuery service. The page is divided

Action 9 into several sections. At the top, it indicates the user is creating ↓ a table from a Google Cloud Storage source, with a CSV file

- Observation 9 Replicate trajectory selected. The destination section shows the project ID and allows ↓ input for the dataset and table name. The destination table is

Action 10 empty. The table type is set to “Native table". At the bottom,

↓ there’s an option for schema detection, with buttons to create the

- Observation 10 table or cancel the operation. The left side of the screen displays a

navigation menu for the Google Cloud Console, including options like Explorer and various project-related items. The overall layout suggests this is part of a larger cloud data management and

analysis platform. After we click on the text box Table, we select

and focus on the text box. We then type “test" into the box, which

gives the table a name. Except the textbox we are working on,

the other parts of the webpage has not changed after clicking

and typing.

- Observation 0 ↓

Action 1 ↓

- Observation 1 New task Link CSV file in Google Cloud Storage to BigQuery ↓

Action 2 ↓ ...... ↓ Observation 13

- Table 28 | Instructions generated from trajectory from Table 20 to 26

{Documentation} Based on the tutorial, examplify 3 tasks that users frequently perform. User the following format to output:

... ...

- Table 29 | self-instruct prompts to propose instructions based on tutorials, documentations and FAQs.

##### Prompt 1

Below is a trajectory to complete a task. Observation: {Observation𝑖} Action:

- {Action𝑖+1} Observation:

{Observation𝑖+1} Action:

- {Action𝑖+2}

... Action: {Action𝑗−1} Observation: {Observation𝑗}

Please write a reasonable task instruction that is completed by the trajectory. Wrap the instruction with ```.

##### Prompt 2

Below is a trajectory to complete a task. Observation: {Observation𝑖} Action:

- {Action𝑖+1} Observation:

{Observation𝑖+1} Action:

- {Action𝑖+2}

... Action: {Action𝑗−1} Observation: {Observation𝑗}

Please summarize the trajectory about each observation and changes after each action. Wrap the summarization with ```.

- Table 30 | Prompts to summarize (sub-)trajectories or propose new tasks based on the (sub)trajectories.

Task instruction: {instruction} Below is the trajectory to complete the task. Observation: {Observation𝑖} Action: {Action𝑖+1} Observation: {Observation𝑖+1} Action: {Action𝑖+2}

... Action: {Action𝑗−1} Observation: {Observation𝑗}

Here are the criteria to indicate a good pair of the instruction and the trajectory:

- 1. The instruction and the trajectory are aligned, which means the trajectory successfully accomplishes the goal in the instruction.
- 2. The trajectory is coherent, indicating that each action is logical based on its previous observation and the actions do not contradict with each other based on the task instruction.
- 3. The trajectory is natural, meaning that the trajectory closely mimics realworld interactions and a human user would possibly perform it when engaging in the environment.
- 4. The trajectory is reasonable, indicating that the trajectory finishes the task instruction using a reasonable solution, e.g., not using an over-complicated method, not over-simply the problem, not going back and forth in states, etc.

Please answer yes if the task instruction and the trajectory satisfies all the criteria, otherwise, answer with no.

- Table 31 | LLM prompts to filter low-quality data

SYSTEM MESSAGE: {system message} OBJECTIVE: {task instruction} INTERACTION HISTORY: {interaction history} OBSERVATIONS: {observations}

Your REASONING and ACTION in the format: REASON: Your reason to choose a specific action. ACTION: Your action

- Table 32 | Model inference prompts without external knowledge

SYSTEM MESSAGE: {system message} ADDITIONAL INFORMATION FOR REFERENCE: {external knowledge} OBJECTIVE: {task instruction} INTERACTION HISTORY: {interaction history} OBSERVATIONS: {observations}

Your REASONING and ACTION in the format: REASON: Your reason to choose a specific action. ACTION: Your action

- Table 33 | Model inference prompts with external knowledge

REASON:

... ACTION:

...

- Table 34 | Expected model outputs

SYSTEM MESSAGE: {system message} Here is the final goal we want to achieve: {task instruction} To achieve the goal, we have done the following: {interaction history} Now, we have observed: {observations}

To better finish the task, write a query to ask for useful information, e.g., what kind of examples or interaction history will be helpful to predict the next action.

- Table 35 | Model prompts to write query for retrieval

Instruction Sum numbers in the first column.

- Observation 0 (Interface of the software LibreOffice Calc)

[Figure 40]

Action 1

import pyautogui pyautogui.click(543,126) // click Tools.

- Observation 1

[Figure 41]

- Table 36 | OSWorld example (filtered)

##### Action 2

import pyautogui pyautogui.click(543,580) // click Solver.

- Observation 2

[Figure 42]

Action 3

import pyautogui pyautogui.click(772,892) // click Close.

- Observation 3

[Figure 43]

- Table 37 | OSWorld example (filtered) cont.

##### Action 4

import pyautogui pyautogui.click(520,126) // click Data.

- Observation 4

[Figure 44]

Action 5

import pyautogui pyautogui.moveTo(520,562) // move to Calculate.

- Observation 5

[Figure 45]

- Table 38 | OSWorld example (filtered) cont.

##### Action 6

import pyautogui pyautogui.click(498,126) // click Sheet.

- Observation 6

[Figure 46]

Action 7

import pyautogui pyautogui.click(152,591) // click cell A4.

- Observation 7

[Figure 47]

- Table 39 | OSWorld example (filtered) cont.

##### Action 8

import pyautogui pyautogui.click(480,302) // click the formula box.

- Observation 8

[Figure 48]

Action 9

import pyautogui pyautogui.write("=SUM(A1:A3)") // type the formula.

- Observation 9

[Figure 49]

- Table 40 | OSWorld example (filtered) cont.

##### Action 10

import pyautogui pyautogui.press("enter")

- Observation 10

[Figure 50]

- Table 41 | OSWorld example (filtered) cont.

Instruction What are items ordered in the latest cancelled order?

- Observation 0

[Figure 51]

Action 1 click [4918] // click the button "My Account"

- Observation 1

[Figure 52]

- Table 42 | WebArena example (filtered)

- Action 2 click [4922] // click the button "My Orders".

Observation 2

[Figure 53]

- Action 3 click [6357] // click the button "View Order".

- Observation 3

[Figure 54]

- Table 43 | WebArena example (filtered) cont.

Action 4 click [4922] // click the button "My Orders".

- Observation 4

[Figure 55]

Action 5 click [6357] // click the button "View Order".

- Observation 5

[Figure 56]

- Table 44 | WebArena example (filtered) cont.

Action 6 click [4922] // click the button "My Orders".

- Observation 6

[Figure 57]

Action 7 click [6357] // click the button "View Order".

- Observation 7

[Figure 58]

- Table 45 | WebArena example (filtered) cont.

