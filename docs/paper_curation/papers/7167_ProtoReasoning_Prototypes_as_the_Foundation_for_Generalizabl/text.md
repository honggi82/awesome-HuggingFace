arXiv:2506.15211v1[cs.CL]18Jun2025

[Figure 1]

# ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs

## Feng He1,†, Zijun Chen1,2,∗, Xinnian Liang1, Tingting Ma1, Yunqi Qiu1, Shuangzhi Wu1,†, Junchi Yan2

1ByteDance Seed, 2Shanghai Jiao Tong University ∗Work done at ByteDance Seed, †Corresponding authors

## Abstract

Recent advances in Large Reasoning Models (LRMs) trained with Long Chain-of-Thought (Long CoT) reasoning have demonstrated remarkable cross-domain generalization capabilities. However, the underlying mechanisms supporting such transfer remain poorly understood. We hypothesize that cross-domain generalization arises from shared abstract reasoning prototypes — fundamental reasoning patterns that capture the essence of problems across domains. These prototypes minimize the nuances of the representation, revealing that seemingly diverse tasks are grounded in shared reasoning structures. Based on this hypothesis, we propose ProtoReasoning, a framework that enhances the reasoning ability of LLMs by leveraging scalable and verifiable prototypical representations (Prolog for logical reasoning, PDDL for planning). ProtoReasoning features: (1) an automated prototype construction pipeline that transforms problems into corresponding prototype representations; (2) a comprehensive verification system providing reliable feedback through Prolog/PDDL interpreters; (3) the scalability to synthesize problems arbitrarily within prototype space while ensuring correctness. Extensive experiments show that ProtoReasoning achieves 4.7% improvement over baseline models on logical reasoning (Enigmata-Eval), 6.3% improvement on planning tasks, 4.0% improvement on general reasoning (MMLU) and 1.0% on mathematics (AIME24). Significantly, our ablation studies confirm that learning in prototype space also demonstrates enhanced generalization to structurally similar problems compared to training solely on natural language representations, validating our hypothesis that reasoning prototypes serve as the foundation for generalizable reasoning in large language models.

Date: June 19, 2025 Correspondence: Feng He at hefeng.hhh@bytedance.com

## 1 Introduction

Large Reasoning Models (LRMs) trained through Reinforcement Learning with Verifiable Rewards (RLVR) have demonstrated remarkable capabilities in complex reasoning tasks [10, 20, 23]. An intriguing phenomenon observed in recent research is that models trained with Long Chain-of-Thought (Long CoT) reasoning [6] in one domain exhibit significant generalization abilities in other domains. Notably, DeepSeek-R1 [10] generalizes from mathematical and coding domains to STEM and creative writing, while Logic-RL [29] transfers logical puzzle-solving capabilities to mathematical reasoning.

Question

[Figure 2]

Transform

Prototype

LLM Train Generalize

|Domain|[Figure 3]<br><br>Logic Planning etc.<br><br>[Figure 4]<br><br>[Figure 5]|
|---|---|

[Figure 6]

[Figure 7]

Domain

[Figure 8]

Verify

Verification System

- Figure 1 ProtoReasoning Overview: Our ProtoReasoning framework adopts prototypes as core representations to enhance model reasoning capabilities. The verification system built upon prototype systems provides accurate supervisory signals, ultimately enabling effective reasoning generalization.

Despite these empirical successes, the underlying mechanisms enabling cross-domain generalization remain poorly understood. A key question is: what allows models trained on specific reasoning tasks to transfer their abilities to different types of problems? We hypothesize that this transfer capability stems from the existence of abstract reasoning prototypes — fundamental reasoning patterns and thinking structures that underlie diverse problem domains [31]. Specifically, when two problems share the same or similar reasoning structures in their underlying solution processes, regardless of their apparent differences in representation, strong generalizations will emerge between them.

To validate this hypothesis and leverage the isomorphic reasoning process to enhance reasoning capabilities, we introduce the ProtoReasoning framework, as illustrated in Figure 1. This framework leverages powerful prototype systems (abstract expressive representation together with a verification system) to formally represent reasoning patterns. Training models to solve problems encoded in abstract prototypes improves performance on a wide range of problems that share similar underlying reasoning structures. We validate the effectiveness of ProtoReasoning in two core domains: logical reasoning and planning, employing Prolog (Programming in Logic) [7] as the prototype representation for logical reasoning and PDDL (Planning Domain Definition Language) [3] for planning tasks. These languages serve as ideal prototypes due to three key characteristics: (1) Declarative nature — Both focus on problem specification instead of procedural implementation, maintaining the reasoning structure found in natural language, as shown in Figure 2; (2) Expressiveness — Prolog captures relational reasoning and constraint satisfaction through first-order predicate logic, while PDDL formalizes state transition systems for sequential planning; (3) Verifiability — both possess mature interpreters (SWI-Prolog [27] and VAL [14]) enabling rigorous verification of reasoning chains.

Building on these foundational representations, our methodology establishes a comprehensive pipeline that includes dataset construction, verification, and model training. For logical reasoning, we develop an automated workflow that converts natural language problems into Prolog representations with interpreter-verified solutions. For planning domains, we introduce three novel task formulations (Plan Generation, Plan Completion, and Plan Reordering) with specialized verification systems ensuring correctness.

The key advantage of our approach is leveraging the strengths of prototype systems, which incorporate comprehensive representation capabilities accompanied with built-in verification mechanisms. These properties enable us to automatically generate diverse problems within the prototype space while guaranteeing their correctness. Without the demand for problem-answer pairs, we only need executable prototype code representing the problem and can derive solutions via an interpreter. Thus, this approach eliminates annotation requirements, providing significant scaling potential.

We employ Supervised Fine-Tuning (SFT) on large language models to confirm the effectiveness of our methodology. Compared to baseline, ProtoReasoning achieves a 4.7% performance improvement on the Enigmata-Eval [5] logical reasoning benchmark, a 6.3% improvement on planning tasks, 4.0% improvement on MMLU [13] general reasoning benchmark and 1.0% on AIME24 [2] mathematical reasoning. Significantly, our ablation experiments confirm that training in reasoning prototypes achieves transfer performance comparable

|Natural Language Representation|
|---|
|Background:<br><br>1. There are three humans named Haruki, Touma, and Setsuna.<br>2. Touma likes Haruki.<br>3. Setsuna likes Haruki.<br>4. Setsuna likes Touma.<br><br><br>For two people to become friends, the following conditions must be met:<br><br>1. Both individuals belong to the set of humans.<br>2. One person likes the other.<br>3. The other person also likes that person.<br>|

|Prolog Representation|
|---|
|human(haruki). human(touma). human(setsuna).<br><br>likes(touma, haruki). likes(setsuna, haruki). likes(setsuna, touma).<br><br>friend(X, Y) :-<br><br>human(X),<br>human(Y),<br><br><br>likes(X, Y),<br>likes(Y, X).<br>|

- Figure 2 Prolog Example: Prolog representation divides logical problems into facts and rules, closely resembling natural language while preserving the logical structure of reasoning problems. More information about the features of Prolog and details of PDDL is available in Appendix C and Appendix D.

to training on structurally similar problems in natural language, demonstrating that reasoning prototypes form the foundation for cross-domain reasoning generalization.

Our contributions are threefold:

- 1. We introduce the concept of reasoning prototypes as a foundation for understanding cross-domain generalization phenomena in reasoning models.
- 2. We propose ProtoReasoning: an automated, scalable framework with integrated verification for enhancing reasoning and planning through systematic data construction.
- 3. We empirically demonstrate that training on abstract prototype representations significantly enhances models’ ability to transfer reasoning capabilities across structurally similar problems.

## 2 The ProtoReasoning Framework

### 2.1 Overview

This section introduces the ProtoReasoning framework, which utilizes representative prototype representations to enhance the general reasoning capabilities of LLMs.

The ProtoReasoning framework consists of two key modules:

- • Prototype Constructor: Responsible for transforming problems into corresponding prototype representations.
- • Verification System: Responsible for evaluating the correctness of model outputs in the prototype representation space.

In the following sections, we will detail Prolog and PDDL as typical prototype representations for logical reasoning and planning tasks.

### 2.2 Logic Prototype Representations

Prolog (Programming in Logic) [7], a declarative language founded on first-order predicate logic [8], employs unification and backtracking mechanisms to represent and solve logical problems. Its declarative nature enables the expression of problems as logical constraints rather than procedural algorithms, creating a natural correspondence with human reasoning patterns. By training LLMs to solve Prolog-formulated problems, we systematically enhance their fundamental logical reasoning capabilities across diverse problem domains.

Next, we introduce our Prolog-based Logic Prototype Constructor and Verification System.

- 2.2.1 Prolog-based Logic Prototype Constructor We designed a four-stage, model-driven pipeline to create diverse, verifiable Prolog reasoning problems:

- 1) Data Initialization. We collected comprehensive corpora of reasoning problems from web sources, encompassing both structured question-answer pairs and unstructured logical narratives.
- 2) Prototype Transformation. Through prompt engineering with LLMs, we converted natural language logic problems into formal Prolog representations, while standardizing interpreter outputs as structured JSON format.
- 3) Data Evolution. We leveraged prompt engineering to systematically enhance problem complexity in a controllable manner while preserving JSON output constraints.
- 4) Answer Derivation. We employed the SWI-Prolog interpreter [26] to derive ground-truth answers. This interpreter-based approach eliminates reliance on pre-existing question-answer pairs and allows us to confidently evolve problem difficulty without introducing answer inaccuracies.

The resulting dataset consists of formalized problem-answer pairs: DProlog = ⟨QProlog,A⟩i , (1)

Where each instance i contains a logical problem QProlog formulated in Prolog representation, paired with its corresponding verified answer A derived from the Prolog interpreter. We present the prompt templates used for prototype transformation and data evolution in the Appendix B.

- 2.2.2 Prolog-based Verification System

Building upon the prototype transformation described in the preceding section, we constrain all reference answers A to be structured JSON dictionaries. We then developed a training prompt template that instructs models to generate predictions Aˆ in the same JSON format. This standardization facilitates rigorous evaluation between the outputs of the Prolog interpreter and model predictions, ensuring both are represented in a consistent format.

Our training prompt template is shown below, where "[program]" and "[query]" are replaced with the corresponding values for each sample:

Prolog Execution Prompt Template for Training

## Prolog Execution Parser As a Prolog code execution expert, your task is to simulate the execution process of SWI-Prolog version 8.2.4 in a Linux x86_64 environment. ## Workflow You will receive a Prolog program and query, then return the execution results, simulating the following command line operation: swipl -q -f prolog_program.pl -t solve_json,halt. ## Technical Specifications Results must 100% accurately reflect the actual output of SWI-Prolog. ## Input [Program] [Query] ## Output Rules Provide only the execution results in JSON format, wrapped with JSON and markers.

- 2.3 Planning Prototype Representations

PDDL (Planning Domain Definition Language) [3] is the standard representation for automated planning problems, modeling state transition systems through three essential components: state representations, actions

with preconditions and effects, and state transitions. This representation naturally aligns with human planning cognition, particularly in reasoning about action requirements and consequences.

Next, we will explain how to construct prototype learning for planning capabilities based on PDDL.

#### 2.3.1 PDDL-based Planning Prototype Constructor

For planning tasks, we employed PDDL-Generator [21] and FastDownward[12] to create problems and derive optimal solutions across diverse domains. Based on International Planning Competition (IPC) [1] benchmarks, we generated varying complexity problems in classical domains like BlocksWorld and Logistics. For each domain, we constructed three distinct task types:

- 1) Plan Generation: Given a domain definition and problem description, the model must generate a complete sequence of actions from initial to goal state, testing end-to-end planning ability.
- 2) Plan Completion: Provided with a partial action sequence, the model must fill in missing steps, requiring an understanding of intermediate states and bidirectional reasoning.
- 3) Plan Reordering: Given unordered action steps, the model must determine a valid execution sequence, testing comprehension of action dependencies and precondition relationships.

The resulting dataset is structured as follows:

DPDDL = {⟨QPDDL,Pref⟩i}, (2)

where each instance i contains a planning problem formulated in PDDL representation QPDDL, paired with its corresponding task-specific reference Pref. These references vary according to task type, with differences explained in detail in Section 2.3.2.

#### 2.3.2 PDDL-based Verification System

Unlike Prolog verification which relies on exact JSON dictionary comparison, PDDL tasks require specialized validation as multiple valid plans can exist for the same problem. Leveraging VAL (PDDL Plan Validator), we implemented custom verification procedures for each task type:

- 1) Plan Generation: For standard generation tasks, we accept any plan that passes VAL verification as correct, with no reference values required. For optimization tasks where plan cost matters (minimum or maximum cost plans), we use FastDownward solver [12] to establish reference optimal values. In these cases, a generated plan is considered correct only if it both passes VAL verification and achieves the required optimality criteria.
- 2) Plan Completion: Here, Pref contains the given partial plan. The model’s output must pass VAL verification while incorporating all actions from the partial plan in their exact positions.
- 3) Plan Reordering: In this case, Pref contains unordered action steps. A correct solution must pass VAL verification and contain the same set of actions as specified in Pref — no missing actions and no additional ones.

Similar to our Prolog approach, we employed three training prompt templates to guide model generation in verifiable formats. Details are available in the Appendix A.

### 2.4 Model Training Recipe

In this section, we introduce our training approach that utilizes supervised fine-tuning to validate effectiveness. Our methodology employs a three-phase process:

- 1) Teacher Model Distillation: We employed Deepseek-R1 [10], a high-performance language model, to generate explicit reasoning chains for our initial datasets. This transformation enriched our data to include step-by-step reasoning paths, creating augmented datasets DPrologAug = ⟨QProlog,RCoT,A⟩i for

- Prolog tasks and DPDDLAug = {⟨QPDDL,RCoT,Pref⟩i} for PDDL tasks. We subsequently fine-tuned our base model using these datasets.
- 2) Difficulty Stratification: Using the model trained in the previous phase, we implement a rejection sampling methodology to classify instances by difficulty. We evaluate each problem 10 times and categorize it based on pass rate:

 



Challenging if 0 < Pass Rate ≤ 0.3 Intermediate if 0.4 ≤ Pass Rate ≤ 0.6 Elementary if 0.7 ≤ Pass Rate < 1

(3)

We exclude both perfectly solved instances (pass rate = 1.0) and completely failed ones (pass rate = 0), then train an enhanced model on this stratified dataset.

- 3) Quality Filtration: With our refined model from the second phase, we conduct a final round of rejection sampling to create our definitive training dataset.

## 3 Experiments

### 3.1 Experimental Setup Table 1 Hyperparameters for Model Training

Dataset. Our baseline dataset comprises 100K diverse samples from the Seed Project, covering various tasks including mathematical reasoning, code generation, and creative writing. Applying our methodology, we initially collected 43,021 raw Prolog examples and 34,123 raw PDDL instances. After processing these through the three-phase pipeline described in our Training Recipe, we refined our collection to 4,196 high-quality Prolog problems and 2,424 PDDL tasks for the final training dataset.

Parameter Value

Model Architecture

Total parameters 150B Activated parameters 15B

Training Configuration

Learning rate 2e-5 Batch size 6 Sequence packing Yes Optimizer AdamW Weight decay 0.1 Epoch 2 Warmup step rate 0.01 Learning rate schedule Cosine decay Maximum sequence length 32768

Evaluation. We evaluated logical reasoning capabilities using the Enigmata-Eval benchmark [5]. To focus specifically on logical reasoning improvements rather than instructionfollowing abilities, we excluded four datasets (Campsite, Car Painting, Star Battle, and Sum Skyscraper) where baseline models struggled primarily with following instructions rather than the reasoning tasks themselves. For planning assessment, we employed a dual approach: direct evaluation using an internal planning test set from the Seed Project, and indirect function calling evaluation through the Nexus-Hard benchmark (another internal evaluation set representing a more challenging subset of Nexus [24]). This complementary evaluation strategy leverages the cognitive overlap between planning and function calling — both requiring sequential action formulation and parameter management — to provide insights into the cross-domain transfer of planning abilities. To assess out-of-domain generalization, we evaluate ProtoReasoner on MMLU [13] (general reasoning) and AIME24 [2] (mathematical reasoning), extending beyond its native logic and planning domains. To ensure result stability, we evaluated each sample 3 times and used the average score as the final result, except AIME2024, where each sample was evaluated 10 times. All benchmarks are 0-shot evaluations.

Hyperparameters. We adopt a Mixture-of-Experts (MoE) architecture that activates 15B parameters from a total parameter count of 150B as our training model. The training employed a learning rate of 2e-5 and a batch size of 6 with sequence packing enabled. The complete experimental hyperparameters are presented in Table 1.

- Table 2 Performance of baseline and ProtoReasoning on different benchmarks. Results demonstrate that our ProtoReasoning framework improves both logical reasoning and planning ability, showing a generalization from prototype representation to natural language representation. Scores represent the average performance across all samples in each test set.

Method Enigmata-Eval Nexus-Hard Task Planning AIME2024 MMLU Baseline 37.3 53.1 46.7 72.0 82.7 ProtoReasoning 42.0↑4.7% 59.5↑6.4% 53.0↑6.3% 73.0↑1.0% 86.7↑4.0%

- Table 3 Performance comparison between the baseline method and ProtoReasoning across Enigmata-Eval categories. ProtoReasoning achieves improvements in all tasks, with the largest gain in Crypto (+11.0%) and the smallest in Sequential (+0.3%).

Method Arith. Crypto Graph Search Seq. Grid Logic Baseline 61.0 28.8 43.4 26.1 16.0 43.8 65.3 ProtoReasoning 64.7↑3.7% 39.8↑11.0% 52.4↑9.0% 29.3↑3.2% 16.3↑0.3% 48.2↑4.4% 68.6↑3.3%

### 3.2 Experimental Results

As demonstrated in Table 2, incorporating synthetic Prolog and PDDL data led to substantial improvements across all reasoning benchmarks. On the Enigmata-Eval benchmark, our approach increased logical reasoning performance from 37.3% to 42.0%, representing a 4.7% improvement. This significant gain suggests that training with logic prototypes enhances the model’s ability to recognize and apply fundamental reasoning patterns. Detailed performance breakdowns by category are available in Table 3.

Similarly, planning capabilities showed marked enhancement, with Nexus-Hard scores rising from 53.1% to 59.5% and dedicated planning task performance increasing from 46.7% to 53.0%. The consistent improvements across both direct planning tasks and function calling evaluations indicate that the skills acquired through prototype-based training are transferred effectively across different manifestations of planning problems.

Notably, ProtoReasoning demonstrates strong generalization beyond its core training domains: compared to standard LLM training, it elevates performance on the general knowledge benchmark MMLU [13] from 82.7% to 86.7% (+4.0%), and on the mathematical reasoning benchmark AIME2024 [2] from 72.0% to 73.0% (+1.0%). These gains confirm ProtoReasoning’s effectiveness not only in logical reasoning and planning domains but also in general knowledge reasoning and mathematical reasoning.

### 3.3 Ablation Study

This section presents a detailed analysis of our proposed reasoning prototypes, examining their effectiveness at a fundamental level.

Ablation Setup. For our ablation study, the Enigmata-Eval benchmark [5] was partitioned into distinct training and development sets. We processed the training subset using two different methods: (1) conversion into Prolog representations with interpreter-verified solutions, and (2) preparation of the original natural language problems utilizing multi-stage rejection sampling validated by Enigmata-Eval’s native verifier. This procedure generated a matched training corpus of 453 samples, representing the intersection of both methods where the problems are encoded in both Prolog and natural language formats. This controlled experimental design isolates the impact of the representational format while keeping the problem content invariant between experimental configurations.

We design the following three experimental configurations:

1. Baseline (Method 1): Model trained exclusively on our standard dataset (absent Enigmata-related problems).

- Table 4 Performance comparison between prototype and natural language representations on the Enigmata-Eval benchmark. The Prolog prototype achieves performance comparable to natural language, demonstrating its generalizable reasoning ability.

Method Trans. Set Dev. Set

Baseline 35.2 38.5

+ Prolog 54.2↑19.0% 44.1↑5.6% + NL 58.1↑22.9% 45.0↑6.5% + Prolog (without CoT) 41.9↑6.7% 39.6↑1.1%

- Table 5 Prolog prototype achieves performance comparable to natural language (NL) representation across most categories in Enigmata-Eval. However, it underperforms the baseline on the logic transfer subset, mainly attributed to limited samples. For those categories contain sufficient samples, the performance of Prolog prototype match and can even surpass natural language representation, demonstrating stronger generalization capability.

Dataset Baseline +NL +Prolog Size

Trans. Set 48.5 60.6↑12.1% 56.1↑7.6% 22

Arith.

Dev. Set 62.0 67.1↑5.1% 69.4↑7.4% 278 Crypto

Trans. Set 22.8 48.9↑26.1% 51.1↑28.3% 120

Dev. Set 32.8 54.4↑21.6% 44.4↑11.6% 180 Graph

Trans. Set 61.5 80.5↑19.0% 76.9↑15.4% 65

Dev. Set 39.9 44.9↑5.0% 45.8↑5.9% 635 Search

Trans. Set 41.1 68.2↑27.1% 58.1↑17.0% 43

Dev. Set 25.7 35.7↑10.0% 34.4↑8.7% 757 Seq.

Trans. Set 8.3 25.0↑16.7% 25.0↑16.7% 4

Dev. Set 17.9 18.3↑0.4% 18.4↑0.5% 804 Grid

Trans. Set 16.7 41.7↑25.0% 38.6↑21.9% 44

Dev. Set 48.8 55.6↑6.8% 54.9↑6.1% 756 Logic

Trans. Set 39.6 58.3↑18.7% 38.5↓1.1% 32

Dev. Set 67.3 75.7↑8.4% 72.2↑4.9% 418

- 2. Baseline w/ Prolog Representation (Method 2): Model trained on standard dataset augmented with formalized Prolog representations of Enigmata reasoning problems, enabling evaluation of reasoning prototype effectiveness in enhancing reasoning capabilities.
- 3. Baseline w/ Natural Language (Method 3): Model trained on the standard dataset augmented with natural language versions of the same reasoning problems used in the prototype approach, enabling direct comparison between prototypes and natural language representations.

Ablation Evaluation. We evaluated model performance using two complementary test sets to assess different aspects of reasoning transfer. Similar to previous experiment, we sample each problem three times and used the average score for reliable assessment:

- 1. Prototype Transfer Set: This comprises the original natural language versions of problems in EnigmataEval used to create our Prolog training examples. Performance in this set measures how effectively reasoning capabilities acquired through logic prototype training transfer to solving natural language problems with similar logical structures.
- 2. Development Set: This consists of the remaining portion of Enigmata-Eval that was not used in creating the Prototype Transfer Set. Performance in this set evaluates broader generalization capabilities, assessing whether models can apply learned reasoning patterns to problems outside of the training

distribution.

Results and Analysis. Table 4 demonstrates that Prolog representations yield significant and consistent performance gains across both transfer and development sets. The performance gain of the baseline w/ Prolog representation (Method 2) over the baseline (Method 1) validates that training in the prototype representation effectively generalizes to natural language problems. Moreover, the comparable performance between method 2 (w/ Prolog representation) and method 3 (w/ natural language) confirms that prototype training achieves effect similar to direct natural language training, consolidating the generalization capability of the prototype. Additionally, our Prolog training experiment without CoT reasoning showed dramatically reduced performance, confirming that effective generalization through prototypes requires explicit reasoning processes, as they benefit from homogenized reasoning paths.

Table 5 details category-wise performance for both transfer and development sets. Insufficient samples in the transfer set of logic category resulted in unstable evaluation, yielding performance worse than baseline. However, in most cases with sufficient samples, training in the Prolog prototype achieved comparable or even superior performance to training in natural language representation. Notably, our analysis uses sampleaveraged results rather than category-specific metrics due to our limited training dataset. Since our transfer set consists of the natural language versions of the same problems used in the Prolog training set, sample-averaged metrics are more reliable than category-specific analysis.

## 4 Related Work

Long CoT and Reasoning Model. Recent advances in LLM reasoning, such as OpenAI-o1 [16], DeepSeekR1 [10], Seed-Thinking-v1.5 [20] and Kimi-k1.5 [23], have changed the focus from Chain-of-Thought (CoT) [25] and supervised fine-tuning (SFT) to reinforcement learning (RL). Deepseek-R1 [10] leveraged mathematical problems and code executions to increase Long CoT reasoning capabilities. Seed-Thinking-v1.5 [20] employed various task collections that span mathematics and logic puzzles. Logic-RL [29] utilized Knights and Knaves puzzles [28] to demonstrate generalization to challenging mathematical benchmarks. In this stage, RL algorithms, e.g., PPO [19], GRPO [22], DAPO [34], are adopted to guide the LLM exploring reasoning paths and to stimulate the long CoT reasoning ability, relying on verifiable rewards, such as accuracy based on ground-truth answers. These approaches collectively demonstrate the effectiveness of reinforcement learning with verifiable reward (RLVR) [15] in developing sophisticated reasoning abilities, including strategies such as recognizing correcting mistakes, breaking down difficult steps and iterating on alternative approaches, thus showcasing the powerful generalization capacity of long Chain-of-Thought reasoning [6]. Compared to previous work, our work introduces the concept of reasoning prototypes, aimed at understanding the underlying generalization mechanisms emerging from long chain-of-thought, and provides a more fundamental framework for cross-domain reasoning transfer.

Symbolic Reasoning in LLMs. Large language models conduct reasoning not only in natural language space [25] but also in a neuro-symbolic manner [9, 17, 33]. The intermediate reasoning steps can manifest themselves as code [9], domain-specific languages [33, 36], or mixtures of different symbolic representations [11, 35]. For Prolog programming language, some previous work [4, 7, 32] leveraged it as an intermediate representation to improve the reasoning ability of LLMs. Rather than emphasizing the specific manifestation of the Chain-ofThought process, we investigate how prototypes can stimulate effective reasoning by exploiting the inherent thinking structures between tasks.

## 5 Conclusion and Future Work

This paper introduces ProtoReasoning, a framework that validates the hypothesis that abstract reasoning prototypes serve as the foundation for cross-domain generalization in large language models. By training on prototype representations (Prolog for logical reasoning, PDDL for planning), we achieve significant improvements on both logical reasoning and planning tasks, with ablation studies confirming effective transfer to structurally similar problems. Additionally, we believe this framework is generalizable for other LLM capabilities. However, our theoretical understanding remains insufficient — the precise definition of "reasoning prototypes" lacks formal rigor, and the underlying mechanisms driving cross-domain transfer require deeper

investigation. In the future, we should develop more rigorous mathematical frameworks to characterize these prototypes and provide stronger theoretical foundations for our empirical findings. We will also open-source the curated prototype datasets (Prolog and PDDL) used in this study to accelerate research progress in the community. Further, we will reproduce our result in open-sourced large language models [18, 30] to ensure a broader validation of our hypothesis.

## References

- [1] IPC. International planning competition. URL https://www.icaps-conference.org/competitions/.
- [2] American invitational mathematics exam (aime), 2024. URL https://artofproblemsolving.com/wiki/index. php/AIME_Problems_and_Solutions/.
- [3] Constructions Aeronautiques, Adele Howe, Craig Knoblock, ISI Drew McDermott, Ashwin Ram, Manuela Veloso, Daniel Weld, David Wilkins Sri, Anthony Barrett, Dave Christianson, et al. Pddl| the planning domain definition language. Technical Report, Tech. Rep., 1998.

- [4] Nasim Borazjanizadeh and Steven T Piantadosi. Reliable reasoning beyond natural language. arXiv preprint arXiv:2407.11373, 2024.

- [5] Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, Hao Zhou, and Mingxuan Wang. Enigmata: Scaling logical reasoning in large language models with synthetic verifiable puzzles, 2025. URL https://arxiv.org/abs/2505.19914.
- [6] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv preprint arXiv:2503.09567, 2025.

- [7] Michael A Covington, Barbara J Grosz, and Fernando CN Pereira. Natural language processing for Prolog programmers. Prentice hall Upper Saddle River, 1994.

- [8] Herbert B Enderton. A mathematical introduction to logic. Elsevier, 2001.

- [9] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.

- [10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [11] Simeng Han, Tianyu Liu, Chuhan Li, Xuyuan Xiong, and Arman Cohan. Hybridmind: Meta selection of natural language and symbolic language for enhanced llm reasoning, 2025. URL https://arxiv.org/abs/2409.19381.
- [12] M. Helmert. The fast downward planning system. Journal of Artificial Intelligence Research, 26:191–246, July

2006. ISSN 1076-9757. doi: 10.1613/jair.1705. URL http://dx.doi.org/10.1613/jair.1705.

- [13] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. URL https://arxiv.org/abs/2009.03300.
- [14] Richard Howey, Derek Long, and Maria Fox. Val: Automatic plan validation, continuous effects and mixed initiative planning using pddl. In 16th IEEE International Conference on Tools with Artificial Intelligence, pages 294–301. IEEE, 2004.

- [15] Komal Kumar, Tajamul Ashraf, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Phillip HS Torr, Fahad Shahbaz Khan, and Salman Khan. Llm post-training: A deep dive into reasoning large language models. arXiv preprint arXiv:2502.21321, 2025.

- [16] OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.
- [17] Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang. Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning. In The 2023 Conference on Empirical Methods in Natural Language Processing.

- [18] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, et al. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- [19] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- [20] ByteDance Seed, Yufeng Yuan, Yu Yue, Mingxuan Wang, Xiaochen Zuo, Jiaze Chen, Lin Yan, Wenyuan Xu, Chi Zhang, Xin Liu, et al. Seed-thinking-v1. 5: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.

- [21] Jendrik Seipp, Álvaro Torralba, and Jörg Hoffmann. PDDL generators. https://doi.org/10.5281/zenodo. 6382173, 2022.
- [22] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [23] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

- [24] Nexusflow.ai team. Nexusraven-v2: Surpassing gpt-4 for zero-shot function calling, 2023. URL https://nexusflow. ai/blogs/ravenv2.
- [25] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- [26] Jan Wielemaker, Zhisheng Huang, and Lourens Van Der Meij. Swi-prolog and the web. Theory and practice of logic programming, 8(3):363–392, 2008.

- [27] Jan Wielemaker, Tom Schrijvers, Markus Triska, and Torbjörn Lager. Swi-prolog. Theory and Practice of Logic Programming, 12(1-2):67–96, 2012.

- [28] Chulin Xie, Yangsibo Huang, Chiyuan Zhang, Da Yu, Xinyun Chen, Bill Yuchen Lin, Bo Li, Badih Ghazi, and Ravi Kumar. On memorization of large language models in logical reasoning. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24.

- [29] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

- [30] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [31] Ling Yang, Zhaochen Yu, Bin Cui, and Mengdi Wang. Reasonflux: Hierarchical llm reasoning via scaling thought templates. arXiv preprint arXiv:2502.06772, 2025.

- [32] Xiaocheng Yang, Bingsen Chen, and Yik-Cheung Tam. Arithmetic reasoning with llm: Prolog generation & permutation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 699–710, 2024.

- [33] Xi Ye, Qiaochu Chen, Isil Dillig, and Greg Durrett. Satlm: Satisfiability-aided language models using declarative prompting. Advances in Neural Information Processing Systems, 36:45548–45580, 2023.

- [34] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

- [35] Tong Zheng, Lichang Chen, Simeng Han, R Thomas McCoy, and Heng Huang. Learning to reason via mixture-ofthought for logical reasoning. arXiv preprint arXiv:2505.15817, 2025.

- [36] Jin Peng Zhou, Charles Staats, Wenda Li, Christian Szegedy, Kilian Q Weinberger, and Yuhuai Wu. Don’t trust: Verify–grounding llm quantitative reasoning with autoformalization. arXiv preprint arXiv:2403.18120, 2024.

# Appendix

## A Prompt Template for Training Planning Tasks

##### PDDL Plan Generation Prompt Template

Now, as an authoritative expert in the field of PDDL planning, you will face an important challenge. I will provide you with a PDDL domain file that defines the basic framework and available operations for problem-solving, and a problem description file that clarifies the initial state and the target state. Your core mission is to use your professional knowledge, analyze these two files, and design a PDDL planning solution.

###Input [PDDL Domain File]

“‘pddl {Specific content of the PDDL domain definition} “‘

[PDDL Problem File]

“‘pddl {Detailed description content of the PDDL problem} “‘

###Output

Please output the finally generated planning solution in the strict PDDL format and enclose it with the tags “‘pddl and “‘ for clearly and accurately presenting the complete plan. An example is as follows: “‘pddl {valid plan} “‘

##### PDDL Planning Completion Prompt Template

Please act as a PDDL planning expert. I will provide a Domain, a Problem, and a Partial Plan obtained by randomly deleting parts from a complete and valid Plan. Your task is to restore the complete Plan based on this information.

### Input [Domain]

“‘pddl {pddl_domain} “‘

[Problem]

“‘pddl {pddl_problem} “‘

[Partial Plan]

“‘pddl {partial_plan} “‘

### Output

Please present the plan in PDDL format and enclose it with the tags “‘pddl and “‘, for example: “‘pddl {valid plan} “‘

##### PDDL Plan Reordering Prompt Template

From now on, assume the role of a senior expert in PDDL planning. I will provide you with three key pieces of information: a PDDL domain description, a PDDL problem definition, and a valid PDDL plan with its execution steps out of order. Your core task is to, based on the rules related to the domain and the problem, sort out the disordered planning steps and rearrange them into a logically coherent sequence that meets the execution

requirements.

###Input [Domain Description]

“‘pddl {pddl_domain} “‘

[Problem Definition]

“‘pddl {pddl_problem} “‘

[Out-of-Order Plan]

“‘pddl {output_of_order_plan} “‘

###Output

Please output the complete plan with the adjusted sequence in the standard PDDL format. Be sure to enclose it with the tags “‘pddl and “‘. The example is presented as follows: “‘pddl {valid plan} “‘

## B Prompt Engineering Template for Prolog

To systematically convert natural language problems into Prolog representations, we designed a structured prompt template that guides the transformation process:

Prolog Transformation Prompt Template

## [Role] You are an exceptional logical reasoning expert, skilled at deconstructing complex logical problems described in natural language and transforming them into precise formal expressions. As a Prolog language specialist, you can elegantly convert various logical problems into executable Prolog code, while ensuring complete logical consistency between the formalized expression and the original natural language description. ## [Constraints]

- 1. Logic transformation must be precise and error-free, with transparent and complete process:

- * Strictly prohibit presetting any conclusions in the code; answers must be derived through Prolog execution
- * Programs avoid any logical jumps
- * Information requiring calculation in the original text must show the complete calculation process in the code

- 2. Standardized queries:

- * Use only solve_json. as the query entry point
- * Results must output key logical outcomes in a concise and clear JSON format
- * Must import and correctly use :- use_module(library(http/json)) module for JSON conversion
- * You should ensure that if there is no valid answer, the JSON output is {“result”: “No valid solution found”}.
- * The final JSON string should be exclusively output to the standard output (stdout).(Like “json_write(current_output, JsonTerm, [width(0)]).”)

- 3. Code quality requirements:

- * Ensure code can be correctly executed by SWI-Prolog
- * Clearly divide into Program and Query sections
- * Code logic must be complete and correct, with no execution errors

- 4. Output specifications:

- * Generated JSON must be compatible with Python3 json.loads parsing
- * Prohibit outputting Prolog internal variables or intermediate results

- * Appropriately convert data structures that are difficult to represent in JSON
- * You should ensure that if there is no valid answer, the JSON output is {“result”: “No valid solution found”}.

- 5. Increasing problem complexity:

- * For simple multiple-choice questions, true/false questions, or other easily guessable problems, increase their complexity
- * Ensure the result space is sufficiently large and not easily randomly guessed

- 6. Determinism:

- * Code execution results must be deterministic; random logic is prohibited ## [Input]

{prompt} ## [Output] Because Query can only be solve_json. , you only need to output Program:

Prolog Problem Generalization Prompt Template

## [Instructions] You will continue the conversation above and complete the following functionality (please note that your generated code still needs to meet the format requirements mentioned above):

- ## [I. Background]

- * You need to generalize a logical or algorithmic problem, creating a more complex and challenging variant, and implement it in Prolog.
- * Generalization means preserving the core idea of the original problem while making it more universal or complex by changing parameters, adding constraints, or modifying objectives.

- ## [II. Role]

- * You are a computer scientist specializing in logic programming, particularly skilled in the Prolog language and problem formalization.
- * Your task is to creatively generalize problems while ensuring correctness and executability of their implementation.

- ## [III. Generalization Requirements]

- * The generalization should preserve the essence of the original problem while increasing its complexity or universality.
- * The generalized problem should have a sufficiently large solution space, avoiding:
- * Multiple-choice, true/false, or other easily guessable formats.
- * Problems with extremely few solutions.
- * Problems with overly obvious solution patterns.
- * Generalization directions may include but are not limited to:
- * Increasing problem scale (e.g., from 8-queens to n-queens).
- * Modifying constraints (e.g., adding mandatory waypoints, resource limitations).
- * Changing optimization goals (e.g., from shortest path to specific-length path).
- * Adding extra dimensions (e.g., adding a time dimension to a 2D problem).

- ## [IV. Examples]

- * Original problem: Select numbers from a set of 10 to sum to 30.
- * Generalized: Select combinations of items from a given list to have a total value within a specific range while not exceeding a backpack weight capacity.
- * Original problem: 8-queens problem.
- * Generalized: N-queens problem with certain squares pre-forbidden.
- * Original problem: Shortest path between cities.
- * Generalized: Delivery route optimization problem with time window constraints.

- ## [V. Evaluation Criteria]

- * Creativity: Whether the generalization is creative and meaningful (50%).
- * Complexity: Whether the problem has sufficient complexity and challenge (50%). ## [Final Requirements]
- * Please ensure your generalization truly adds depth to the problem, not just changing surface parameters.
- * Please ensure that the Prolog code has solutions in the end (not trivial solutions).

###### Knowledge Base Query Natural Language Description

"We have a family with parents, children, and various family relationships. John and Mary have two children, Bob and Lisa. Bob has two children, Ann and James, while Lisa has one child, Carol. We want to represent these relationships and query various family connections, such as identifying grandparents, siblings, and other family relations."

% Facts about parent

% Who are Bob’s children? ?- parent(bob, Child). Child = ann ; Child = james.

relationships parent(john, bob). parent(john, lisa). parent(mary, bob). parent(mary, lisa). parent(bob, ann). parent(bob, james). parent(lisa, carol).

% Who are the grandparents of

Carol? ?- grandparent(GP, carol). GP = john ; GP = mary.

% Rules defining relationships

This classic family relationship problem is formalized through Prolog facts and rules, which translate natural language relationships into logical predicates and inference rules. The knowledge base contains both explicit facts (direct parent relationships, gender) and rules that define derived relationships (grandparent, sibling). Prolog’s inference engine then answers queries by applying logical resolution to find all possible solutions that satisfy the given logical conditions.

% Are Lisa and Bob siblings? ?- sibling(lisa, bob). true.

father(X, Y) :parent(X, Y), male(X).

% Find all siblings ?- sibling(X, Y). X = bob, Y = lisa ; X = lisa, Y = bob ; X = ann, Y = james ; X = james, Y = ann.

mother(X, Y) :parent(X, Y), female(X).

grandparent(X, Z) :-

- parent(X, Y),

- parent(Y, Z).

% Is Mary a grandmother? ?- mother(mary, P), parent(P,

sibling(X, Y) :-

_). P = bob ; P = lisa.

- parent(P, X),

- parent(P, Y), X \= Y.

% Facts about gender male(john). male(bob). male(james). female(mary). female(lisa). female(ann). female(carol).

Table 6 Prolog knowledge base, queries and their description

## C Prolog as Prototype for Logic

Prolog (Programming in Logic) [7] is a declarative programming language based on first-order predicate logic. As a logic prototype, Prolog provides an elegant framework that represents logical reasoning through a fact and rule-based system, enabling direct modeling of relational knowledge and inference processes.

Structure and Components: The fundamental structure of Prolog is built around a knowledge base composed of facts and rules, formally represented as a program P = F ∪ R. Here F = {f1,f2,...,fm} represents the set of facts, where each fact fi is a predicate consisting of a relation name and terms. Meanwhile, R = {r1,r2,...,rn} represents the set of rules, where each rule rj is formulated as H : −B1,B2,...,Bk, with H being the head (conclusion) and the conjunction of Bi forming the body (conditions).

Logic Representation: The core of Prolog’s reasoning power lies in its unification algorithm and resolution strategy. Predicates represent relations between objects, variables allow for expressing general patterns, and rules enable the definition of complex logical relationships and inferences. This representation allows Prolog to model logical entailment, recursive definitions, and complex queries with remarkable clarity. Prolog’s

evolution has incorporated extensions such as constraint logic programming, tabling, and probabilistic logic programming, enhancing its expressive capabilities for diverse reasoning tasks.

Cognitive Alignment: Prolog’s expression paradigm closely aligns with human cognitive processes in reasoning tasks, particularly in applying deductive inference, pattern matching, recursive thinking, and relational reasoning. This alignment makes Prolog an ideal prototype for capturing the structure of logical reasoning problems in a way that mirrors human thought processes. The declarative nature of Prolog allows for focusing on the logical relationships (knowledge representation) rather than procedural details, creating a representation that reflects the abstract reasoning patterns used in human logical thinking.

## D PDDL as Prototype for Planning

PDDL (Planning Domain Definition Language) [3] is a standardized formal language designed to represent automated planning problems. As a logic prototype, PDDL provides a rich, expressive framework that captures the essential elements of planning domains through a declarative specification approach.

Structure and Components: The fundamental structure of PDDL emphasizes a clear separation between domain knowledge and problem-specific details, represented formally as a tuple (D,P). Here D = (T,P,A) represents the domain definition, containing a type hierarchy T (organizing objects into categories), a set of predicates P (defining relations between objects), and a collection of action schemas A (specifying state transitions). Meanwhile, P = (O,I,G) represents the problem instance, containing a set of objects O (populating the domain), an initial state I (describing the starting configuration), and a goal specification G (defining desired conditions).

Action Representation: The core of PDDL’s expressive power lies in its action representation. Each action schema a ∈ A is defined as a tuple (params,pre,eff), where params are typed variables representing action parameters, pre is a logical formula specifying preconditions, and eff describes effects as additions and deletions to the state. This representation enables PDDL to model causality, constraints, and state evolution with high fidelity to human planning processes. PDDL’s evolution through various extensions (PDDL2.1, PDDL2.2, PDDL3.0, etc.) has progressively incorporated features such as numeric fluents, durative actions, preferences, and trajectory constraints, enhancing its representational capabilities.

Cognitive Alignment: PDDL’s expression paradigm closely aligns with human cognitive processes in planning tasks, particularly in analyzing preconditions before action execution, predicting changes resulting from actions, decomposing complex goals into achievable subgoals, and reasoning about action sequences and their consequences. This alignment makes PDDL an ideal prototype for capturing the logical structure of planning problems in a way that facilitates both computational implementation and human understanding. The declarative nature of PDDL allows for focusing on the "what" (problem definition) rather than the "how" (solution procedure), creating a representation that mirrors abstract reasoning patterns used in human planning.

###### Domain Problem Natural Language Description

"You have a 3×3 grid with eight numbered tiles and one empty space. Tiles can slide into the adjacent empty space. The tiles begin in a scrambled configuration, and you must rearrange them to have tiles 1-8 in numerical order."

(define (domain n-puzzle-

(define (problem n-puzzle-3) (:domain n-puzzle-typed) (:objects

typed) (:requirements :typing) (:types position tile) (:predicates

- p_1_1 p_1_2 p_1_3

- p_2_1 p_2_2 p_2_3

- p_3_1 p_3_2 p_3_3 position

(at ?tile - tile ?position

- position) (neighbor ?p1 - position ?

This classic puzzle game, commonly referred to as the "N-puzzle" in natural language, is formalized through PDDL domain and problem definitions, which essentially translate the natural language problem into the PDDL code space. The actual feasible planning solution is subsequently derived by a solver.

t_1 t_2 t_3 t_4 t_5 t_6

p2 - position) (empty ?position -

t_7 t_8 - tile) (:init

position)) (:action move

- (at t_2 p_1_1)

- (at t_3 p_1_2)

- (at t_7 p_1_3)

- (at t_5 p_2_1)

(at t_4 p_2_2) (empty p_2_3) (at t_8 p_3_1)

- (at t_6 p_3_2) (at t_1 p_3_3)

:parameters (?tile - tile ?from ?to -

position) :precondition

(and (neighbor ?from ?to ) (at ?tile ?from) (empty ?to))

- (neighbor p_1_1 p_1_2)

- (neighbor p_1_2 p_1_1) )

:effect

(and (at ?tile ?to) (empty ?from) (not (at ?tile ? from))

(:goal (and

- (at t_1 p_1_1)

- (at t_2 p_1_2)

- (at t_3 p_1_3)

- (at t_4 p_2_1)

- (at t_5 p_2_2)

- (at t_6 p_2_3)

- (at t_7 p_3_1)

- (at t_8 p_3_2))))

(not (empty ?to)))) )

Table 7 PDDL domain, problem and their description

