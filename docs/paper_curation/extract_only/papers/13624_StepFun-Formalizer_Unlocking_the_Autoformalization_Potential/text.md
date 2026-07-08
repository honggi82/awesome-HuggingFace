## arXiv:2508.04440v3[cs.CL]26Dec2025

### StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion

#### Yutong Wu1,2* , Di Huang1, Ruosi Wan4, Yue Peng4, Shijie Shang4 Chenrui Cao1,3, Lei Qi1,3, Rui Zhang1, Zidong Du1, Jie Yan4, Xing Hu1†

1SKL of Processors, Institute of Computing Technology, CAS 2University of Chinese Academy of Sciences 3University of Science and Technology of China 4StepFun Inc. Contact: wuyutong22s@ict.ac.cn

###### Abstract

Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expertdesigned templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.

Code — https://github.com/stepfun-ai/StepFun-Formalizer Models — https://huggingface.co/stepfun-ai/StepFun-

Formalizer-32B

Extended version — https://arxiv.org/abs/2508.04440

#### 1 Introduction

Autoformalization aims to translate natural-language mathematical statements into formally verifiable statements in formal languages such as Lean (Moura and Ullrich 2021), Coq (Barras et al. 1999), and Isabelle (Paulson 1994). With recent advances in automated theorem proving (Li et al. 2024) and formal verification (Beg, O’Donoghue, and Monahan 2025), it has garnered growing interest (Weng et al. 2025) for underpinning data synthesis for theorem provers (Xin et al. 2024a,b; Lin et al. 2025a; Zhang et al. 2025; Wang

*Work done during an internship at StepFun. †Corresponding author

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

et al. 2025; Ren et al. 2025; Ji et al. 2025), the validation of informal mathematical reasoning (Zhou et al. 2024), and the generation of verifiable code (Lin et al. 2024; Thakur et al. 2025; Miranda et al. 2025).

The current mainstream approaches for autoformalization involve employing an LLM to translate informal mathematical problems into their corresponding formal statements. These approaches can be categorized into two types: (1) Directly distilling or prompting general-purpose LLMs, such as FormalMATH (Yu et al. 2025b), which generates autoformalization training data using GPT-4, and FMC (Xie et al. 2025), which directly employs DeepSeek-R1 (DeepSeek-AI 2025) as the translation model. (2) Training a specialized model for the autoformalization task from scratch, starting with existing manually annotated informal-formal pairs and followed by supervision from human experts, such as Lean Workbook (Ying et al. 2025) and Kimina-Autoformalizer (Wang et al. 2025). However, these two types of methods suffer from low accuracy and require substantial human effort to check and revise the human-generated formal statements (Yu et al. 2025b; Zhang, Valentino, and Freitas 2025).

We identify two key abilities for effective autoformalization models: (1) Comprehensive mastery of domain knowledge in formal language. A model must know how to express every mathematical concept in the target formal language. For instance, without familiarity with the Lean 4 definition of Euler’s totient function, it cannot formalize related problems (see Appendix E.1 in the FormalMATH paper (Yu et al. 2025b)). Because formal data are scarce in general corpus (Xin et al. 2024a) and significant differences exist between versions of formal languages (e.g., Lean 3 and Lean 4 (Community 2023)), general-purpose LLMs often mislearn or overlook crucial language details, limiting their performance (e.g., Claude4-thinking in Figure 1). (2) Reasoning capability of natural language problem understanding and informal-formal alignment. Models tackling real-world problems must first grasp the intended meaning, which is something specialized models with annotated data in narrow scenarios struggle with (e.g., Kimina-Autoformalizer in Figure 1). Moreover, informal problem statements often omit details (e.g., type hints) that must be made explicit in formal language for rigorous verification. Overcoming these

Informal Problem: Victoria wants to order at least 550 donuts from Dunkin’ Donuts for the HMMT 2014 November contest. However,

. Assuming every twelve donuts cost $7.49, what is the minimum amount Victoria needs to pay, in dollars? Prove that the answer is: 344.54.

|donuts only come in multiples of twelve|
|---|

Claude4-thinking (advanced general-purpose models, lack of formal knowledge): <think> Let me think about this problem step by step. Victoria wants to order at least 550 donuts. Donuts come in multiples of 12... </think> theorem b :

StepFun-Formalizer (ours, formal knowledge + reasoning): <think> The problem involves math several concepts:

- 2. **Cost calculation**: Each

|pack of 12|
|---|

donuts costs $7.49. The total cost is calculated by multiplying the number of packs by the cost per pack... Based on the problem, we need to work with the following mathematical objects:

- 3. **Cost**: This is a real number (R), calculated as 7.49 × n. We need to find the minimum cost... Several potential issues might arise when formalizing this problem in Lean:

let target_donuts : N := 550 let donuts_per_dozen : N := 12 let min_dozens := Nat.ceil_div

. . . ✗ Mistakes: Function Nat.ceil div does not exist in Lean 4. Kimina-Autoformalizer (specialized models, lack of reasoning): theorem a : IsLeast {x | D n : N, x = n * 7.49 ^

target_donuts donuts_per_dozen

1. **Type coercion**: For instance, 7.49 is a real number, but we’re multiplying it by a natural number n. Lean will handle this through automatic coercion, but we should be aware of it.

... </think> theorem c : IsLeast {x : R | D n : N, x =

} 344.54 := by sorry✗

|12 | n ^ n ě 550|
|---|

Mistakes: (1) It misunderstands the meaning of dozen (

). (2) The total payment x is not explicitly declared as a real number, causing a type error with the value 7.49.

|Box|
|---|

ě } 344.54 := by sorry

n * 7.49 ^

|12 * n 550|
|---|

# ✓

Figure 1: A case study to demonstrate the impact of formal knowledge and informal-to-formal reasoning capability on autoformalization models. It shows that general-purpose models without formal knowledge make mistakes in code implementation, while specialized ones without reasoning capability struggle with problem understanding and informal-formal alignment. StepFun-Formalizer improves autoformalization performance by combining these two capabilities.

gaps requires strong informal-to-formal reasoning abilities, as demonstrated by StepFun-Formalizer (Figure 1) and earlier studies (Liu et al. 2025b; Xuejun et al. 2025; Xie et al. 2025). Quantitatively, we leveraged GPT-4o (OpenAI 2024a) to classify errors in roughly 10K generated outputs and had human experts annotate a 100-example subset from two autoformalization models (Figure 2, see Appendix B for more details). The results show that Kimina-Autoformalizer suffers a high rate of these two types of errors as an evidence of weak reasoning, whereas our model alleviates these problems.

To equip the model with these two abilities, we introduce ThinkingF, the first data synthesis and training pipeline for the autoformalization model with both domain knowledge and informal-to-formal reasoning capability. We implement this through two key components: (1) large-scale distillation and selection of data with formal knowledge from specialized models, and (2) informal-to-formal reasoning trajectories synthesis guided by an expert-designed autoformalization template.

Specifically, we first construct two datasets to respectively supplement domain knowledge and enhance the model’s reasoning ability. For domain knowledge, we collect informalformal data pairs by employing a specialized model (e.g., Kimina-Autoformalizer) to translate a large number of natural language mathematical problems in public datasets (e.g., NuminaMath-1.5) into formal statements, followed by majority voting and an LLM-based judge to ensure data quality. For reasoning ability, we propose a reasoning template that

|33.7% 42.9%<br><br>35.7%<br><br>50.0%<br><br>33.0% 26.0%<br><br>35.7%<br><br>26.8%<br><br>28.1% 27.0%<br><br>24.3% 16.1%<br><br>5.2% 4.1%<br><br>4.3% 7.1%<br><br>Kimina StepFun (Ours)<br><br>Kimina StepFun (Ours)<br><br>Correct Natural Language Misunderstanding<br><br>Informal-formal Misalignment Other Error<br><br>GPT-4o evaluation:<br><br>Human evaluation:|
|---|

Figure 2: Categorical analysis for errors in autoformalization model. “Kimina-Auto” refers to Kimina-Autoformalizer. It illustrates the proportion of two error types in autoformalization, both of which are mitigated in StepFun-Formalizer.

involves problem decomposition and mathematical object mapping. With this template, we leverage a strong instructionfollowing model (e.g., Claude 3.7 Sonnet) to synthesize the reasoning process for each informal-formal data pair in existing human-annotated datasets. Next, we use the two datasets to perform supervised fine-tuning on a general-purpose LLM with strong informal mathematical and coding capabilities (e.g. DeepSeek-R1-Distill-Qwen), thereby integrating domain knowledge and informal-to-formal reasoning ability into a unified model. Finally, we apply reinforcement learning to the fine-tuned model to further promote the fusion of the two capabilities, using the BEq equivalence verification (Liu et al. 2025b) as a verifiable reward (RLVR). The

overview of ThinkingF is shown in Figure 3.

Based on this pipeline, we develop two sizes of autoformalization LLMs, StepFun-Formalizer-7B and StepFunFormalizer-32B. We evaluate them using the BEq verification on mainstream benchmarks, including FormalMATH-Lite (Yu et al. 2025b), ProverBench (Ren et al. 2025) and CombiBench (Liu et al. 2025a). Specifically, StepFun-Formalizer32B achieves 40.5% on FormalMATH-Lite and 26.7% on ProverBench in BEq@1 score (Liu et al. 2025b), setting new SOTA results among both specialized and general-purpose models.

#### 2 Related Work

Large Language Models for Autoformalization. Autoformalization is the process of converting mathematical expressions from natural language into their formal language representations (Weng et al. 2025). Traditional rule-based autoformalization methods are complex to implement and difficult to generalize (Jiang, Li, and Jamnik 2023). Therefore, we mainly focus on LLM-based methods. Earlier works enhance the autoformalization capability of existing LLMs by in-context learning (Wu et al. 2022), data synthesis with back-translation (Jiang, Li, and Jamnik 2023; Azerbayev et al. 2023; Wu et al. 2025), retrieval-augmented generation (Liu et al. 2025b), natural language inference (Ying et al. 2025), and expert iteration with LLM judgers (Wang et al. 2025). Mathesis (Xuejun et al. 2025) is the first autoformalization model with reinforcement learning in its training process, but it does not perform informal-to-formal reasoning during translation, resulting in informal-formal misalignment. Moreover, since Mathesis’s model and evaluation methods are not publicly available, we are unable to make a performance comparison. For other public works, we compare with them in Section 5.

The main difference between our method and the aforementioned works is that we equipped the model with both domain knowledge of formal language and informal-to-formal reasoning capabilities, thereby significantly improving its autoformalization capability.

Reasoning-Enhanced Large Language Models. Inspired by the powerful and effective RL paradigm of general reasoning LLMs like OpenAI-o1 (OpenAI 2024b), DeepSeekR1 (DeepSeek-AI 2025), and Kimi-K1.5 (Kimi et al. 2025), some previous works attempt to integrate reasoning capabilities from general-purpose LLMs into domain-specific LLMs to enhance their performance in solving complex problems, such as Fin-R1 (Liu et al. 2025c), Table-R1 (Yang et al. 2025), CodeV-R1 (Zhu et al. 2025), and R1-Code-Interpreter (Chen et al. 2025).

Since prior works rely on abundant corpora and general LLMs with rich domain knowledge, they typically distil these models to obtain domain-specific reasoning data, but distillation often limits performance below teacher models. In contrast, we first enhance the general model with formal knowledge, then train for reasoning, achieving performance that matches or exceeds both specialized and general models.

#### 3 Problem Statement

The autoformalization task discussed in this paper can be formulated as follows:

- Definition 3.1 (LLM-based Autoformalization). Given an informal mathematical problem x as the input to an autoformalization model M, its output Mpxq is the corresponding formal statement y with an optional reasoning process. We say that y is a correct formalization of x if and only if:

- (1) y passes the syntax check of the formal language.
- (2) y is semantically equivalent to x.

Due to the difficulty of automating the strict judgment of semantic equivalence between a model-generated formal statement and an informal problem, we use human-annotated formal statements as the ground truth. We then assess correctness by checking if the model’s formalization is equivalent to the human-annotated ground truth using the proof assistant, as in previous work (Liu et al. 2025b):

- Definition 3.2 (Bidirectional Extended Definitional Equivalence, BEq). Two formal statements y1 and y2 are bidirectional extended definitional equivalence (denoted as y1 „ y2) if and only if there exists a formal proof that derives y2 from y1 using semantics-preserving tactics, and vice versa.

Weighing both accuracy and computational cost, we use the strictest BEq check, i.e., only calling the symbolic heuristic tactic exact? to automatically search for an equivalence proof between two statements. In our implementation, we use sorry as the proof of y1 to assume its correctness, and use exact? for y2 to search for a proof from y1. If the concatenation of the two proofs passes verification in the Lean 4 REPL and indeed uses y1 in the proof, then y2 is provable from y1. If y1 is also provable from y2, then y1 „ y2.

4 Method

The data synthesis and training pipeline of ThinkingF consists of 4 stages (Figure 3). Stages 1 „ 2 are used to construct two datasets that enhance the model’s formal knowledge and reasoning capabilities: 1 Knowledge Distillation With Selection. To obtain a formal knowledge dataset, we need a large number of high-quality informal-formal pairs. Therefore, we employ a specialized LLM to translate informal problems into formal ones, followed by selection to improve the data quality. 2 Informal-to-formal Reasoning Data Synthesis. Since there is no reasoning process in previous specialized models and distilling reasoning trajectories from general models yields poor results (see Section 5.4), we design a reasoning template for the autoformalization task. Using this template, we synthesize an autoformalization dataset containing informal-to-formal reasoning processes. Stages

- 3 „ 4 involve the training process of our model, which is used to endow the model with the two capabilities and further promote their fusion: 3 Two-Stage Supervised Fine-tuning. A general-purpose LLM undergoes supervised fine-tuning (SFT) with the two datasets to obtain a model with both capabilities. 4 Reinforcement Learning With Verifiable Reward. To further facilitate the fusion of knowledge and reasoning, we apply reinforcement learning (RL) to train the fine-tuned model, with BEq checking as a verifiable reward.

Knowledge Distillation With Selection

[Figure 1]

###### Two-Stage Supervised Fine-tuning

[Figure 2]

Informal-to-Formal Translation

- <Formal Statement 1> theorem thm...
- <Formal Statement 2>

...

- <Formal Statement 3>

Formal Knowledge

[Figure 3]

<NL Problem>

<NL Problem> Let $n$ be a integer. Prove that...

###### <NL Problem>

... <think> <Reasoning>

[Figure 4]

... <think>NULL</think> <Formal Statement>

... </think> <Formal Statement>

...

...

[Figure 5]

<Useless Statement> theorem thm : 2 > 1 := by... <Useful Statement> theorem thm (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 3 / 2 ≤ a / (b + c) + b / (c + a) + c / (a + b) := by...

[Figure 6]

...

Majority Voting

Knowledge Dataset

[Figure 7]

[Figure 8]

Reasoning Dataset

[Figure 9]

[Figure 10]

Validity Assessment

[Figure 11]

<Formal Statement> theorem thm...

StepFunFormalizer

Reinforcement Learning With Verifiable Reward

[Figure 12]

DeepSeek-R1Distill-Qwen

- <Formal Statement 1> theorem my_theorem...
- <Formal Statement 2>

...

- <Formal Statement 3>

<Formal Statement> theorem putnam_1962

[Figure 13]

Informal-to-formal Reasoning Synthesis

... (Human Annotation)

Template-Guided Reasoning Construction

[Figure 14]

###### <NL Problem>

Equivalence Verification

<Reasoning> Let me analyze the math concepts...

[Figure 15]

... <Formal Statement> theorem putnam_1962... (Human Annotation)

...

Ground Truth

Reasoning Capability

Model-Generated Formal Statements

- Figure 3: The illustration of ThinkingF method. Our method mainly consists of the construction process for knowledge and reasoning dataset (Section 4.1 and 4.2), and the model training process (Section 4.3 and 4.4).

- 4.1 Knowledge Distillation With Selection Informal Problem Preparation Our distillation pipeline

begins with the informal mathematical problems txiu from an open-source dataset, NuminaMath-1.5. Following KiminaAutoformalizer (Wang et al. 2025), we first filter the dataset by manual rules (see Appendix A for details), resulting in approximately 256K informal problems.

Formal Statement Generation and Selection We prompt Kimina-Autoformalizer to generate 16 candidate formal statements tyiju16j“1 for each informal problem xi. Next, we perform a three-tier quality selection on the generated formal statements:

- (1) Syntax Check. We use the Lean4 REPL to perform syntax checking on the formal statements tyiju16j“1, and retain the syntactic correct statements, denoted as tyij˚ um where mi is the number of syntactically correct statements..

- i
- j“1,

- (2) Majority Voting. It is observed that majority voting can significantly improve the performance of autoformalization models (Table 1), just as it does in coding (Li et al. 2022) and informal math (Wang et al. 2023) tasks.

Metrics FormalMATH-Lite ProverBench CombiBench

BEq@1 35.1 13.3 2.6 Maj@16 45.9 17.2 3.0

Table 1: Accuracy gains from majority voting for KiminaAutoformalizer-7B. “Maj@16”: generating 16 outputs per problem and selecting one via majority voting for evaluation. See Section 5 for benchmarks and metrics details.

Specifically, for the syntactically correct formal statements tyij˚ um

- i
- j“1, we use BEq verification to partition the formal

statements into multiple equivalence classes. Then, one formal statement is randomly selected from the largest group as the optimal formalization yi˚˚ corresponding to the informal

problem xi. Formally:

ÿmi

yi˚˚ “ arg max

1pyik˚ „ yij˚ q

yij˚

k“1

where 1p¨q is the indicator function, and „ denotes semantic equivalence. Informal problems and their optimal formalization are collected as a dataset tpxi,yi˚˚qu.

###### (3) Problem Validity Assessment. The selected dataset

tpxi,yi˚˚qu is finally evaluated using DeepSeek-V3 (since it is efficient and affordable), removing oversimplified formal

statements and those containing inherent contradictions in conditions. We keep approximately 183K informal-formal pairs as the final training data. The ablation study without LLM selection (in Appendix B) shows that this step can reduce the amount of training data required while maintaining comparable model performance.

##### 4.2 Informal-to-Formal Reasoning Data Synthesis

Template Design for Reasoning According to the previous error analysis of LLM-based autoformalization, we propose a template-guided reasoning construction framework (Figure 4), which incorporates the human understanding of the autoformalization process to assist the model in generating reasoning trajectories. The framework consists of two parts:

###### (1) Informal Problem Understanding. Before delving

into the details of formal languages, the model needs to deeply understand the natural language problem, including rephrasing the original question, analyzing its high-level logical structure with the decomposition of the mathematical concepts and corresponding objects involved.

###### (2) Informal-to-Formal Analysis. To bridge the misalign-

ment between natural and formal language, the model should first consider the tricky issues that may arise during formalization. Then, following a divide-and-conquer paradigm (Chen et al. 2024), the model maps the natural language mathematical objects to formal language.

Mathematical Concepts Analysis The problem involves several mathematical concepts:

You are given a informal math problem along with its formal statement in Lean 4. You need to provide a formalization process including the following parts:

- (a)

Reasoning Trajectory Synthesis

- (b)

- 1. Positive integers: Natural numbers greater than zero. In this problem, we're focusing on n > 2.
- 2. Relatively prime numbers (coprime): Two integers are relatively prime if their greatest common divisor is 1. We're considering numbers from 1 to n that are relatively prime to n...

- 1. Start with a restatement of the natural language math problem.
- 2. Outline the high-level logical structure of the mathematical problem, such as if-and-only-if.
- 3. Analyze the mathematical concepts involved in the problem.
- 4. Determine the mathematical objects to be defined based on the concepts.

- 5. Analyze the headers needed before formalizing the problem into Lean 4 (e,g,, imports).
- 6. Anticipate tricky Lean issues (e.g., implicit type coercions).
- 7.Map the math concepts, amd objects in the problem to Lean 4.
- 8. Finally, review the complete code, ensuring it is consistent with the given formal statement.

###### (c)

Mapping Math Concepts to Lean

Now, let's map each mathematical concept to its corresponding Lean representation:

1. Set of numbers from 1 to n: In Lean, it is represented using

the `Finset.Icc` notation: ```lean Finset.Icc 1 n ```

(d)

The closed interval [1, n] is a finite set, which is exactly the set {1, 2, ..., n} ....

(e)

- Figure 4: The prompts and examples in the templateguided reasoning construction framework. (a) The task description for autoformalization. (b) Understanding of natural language problems. (c) An example of concept analysis in problem understanding. (d) Analysis of converting informal math objects into formal language. (e) An example of mapping concepts to Lean in informal-to-formal analysis.

Synthesizing reasoning trajectories upon existing humanannotated data. To maximize the correctness of the model’s reasoning, we use informal problems with a human-annotated ground-truth formal statement (denoted as tpxˆi,yˆiqu) as seed data. Following previous work (Wang et al. 2025; Lin et al. 2025a), the human-annotated data mainly comes from the matched informal-formal statements in automated theorem-proving datasets (see Section 5.1 for details). With the annotated data pairs tpxˆi,yˆiqu, we prompt Claude

- 3.7 Sonnet (since it has strong instruction-following capa-

bilities) to generate a reasoning trajectory cˆi from xˆi to yˆi following our reasoning template. In total, we synthesize 5.8K instances of informal-to-formal reasoning data.

- 4.3 Two-Stage Supervised Fine-tuning

With the knowledge-distilled dataset tpxi,yiqu and the reasoning dataset tpxˆi,cˆi,yˆiqu, we conduct two-stage supervised fine-tuning (SFT) on DeepSeek-R1-Distill-Qwen, known for its strong reasoning performance in informal mathematics and coding. Specifically, in the first stage of supervised fine-tuning, xi is used as the input, and special tokens <think></think> are inserted before the corresponding output yi to ensure internal format consistency within the model (Qwen 2025) and maintain its reasoning capability. During the second-stage supervised finetuning, we follow the standard format for reasoning models (DeepSeek-AI 2025) by enclosing the reasoning tra-

jectory cˆi within <think> and </think> in the output, i.e., <think>cˆi</think>yˆi. After the two-stage supervised fine-tuning, we obtain a preliminary model, StepFunFormalizer-SFT, equipped with both formal domain knowledge and informal-to-formal reasoning capability.

##### 4.4 Reinforcement Learning With Verifiable Reward

To further enhance the model’s reasoning capability, we perform RL on StepFun-Formalizer-SFT. Due to the lack of highquality human-annotated data, we use the same set of 5.8K problems tpxˆi,yˆiqu from the second-stage SFT training for RL training. Despite being used in SFT, continuing RL training on these data still leads to performance improvements (see Section 5.5). The rewards are calculated by performing BEq equivalence verification („) between model-generated statements and ground-truth. More formally, the accuracy reward function is defined as follows:

Rpyi,yˆiq “ "

1, if yi „ yˆi 0, otherwise

Taking both training speed and performance into account, we choose Group Relative Policy Optimization (GRPO) algorithm (Shao et al. 2024) in reinforcement learning training, which eliminates the value function and estimates the advantage in a group-relative manner. We also incorporate several improvements from Dynamic Sampling Policy Optimization (DAPO) (Yu et al. 2025a), including dynamic sampling and token-level loss.

5 Experiments

In this section, we present the implementation details of ThinkingF (Sec. 5.1). We conduct a series of experiments (settings in Sec. 5.2) to show the performance comparison between StepFun-Formalizer and previous SOTA models (Sec. 5.3), alternative designs in our method (Sec. 5.4), the further analysis of RL training and real-world applications (Sec. 5.5).

##### 5.1 Implementation Details

Dataset Size

MiniF2F (Zheng, Han, and Polu 2022) 488 ProofNet (Azerbayev et al. 2023) 357 PutnamBench (Tsoukalas et al. 2024) 659 Compfiles (Renshaw 2024) 115 FormalMATH-Train (Yu et al. 2025b) 5135

Training

FormalMATH-Lite (Yu et al. 2025b) 425 ProverBench (Ren et al. 2025) 174 CombiBench (Liu et al. 2025a) 100

Evaluation

Table 2: Data partitions and sizes. We select a subset of the three most recent datasets for evaluation to ensure fairness, with the remaining datasets for training. Specifically, we use a subset of FormalMATH (FormalMATH-Lite) for evaluation, and the remaining (FormalMATH-Train) for training.

FormalMATH-Lite ProverBench CombiBench BEq@1 BEq@16 BEq@1 BEq@16 BEq@1 BEq@16

Model

(General-purpose Models) OpenAI o3-pro 22.6 35.5 24.7 36.2 9.0 16.0 Claude4-thinking 20.8 32.2 24.4 35.6 9.7 18.0 Gemini-2.5-thinking 17.8 31.3 20.1 36.8 8.9 18.0 DeepSeek-R1-671B 18.4 31.3 23.5 34.5 8.1 20.0 DeepSeek-R1-Distill-7B 5.2 14.6 5.4 18.4 0.4 2.0

(Specialized Models) LeanFormalizer-PPO-7B 18.7 24.0 12.4 18.4 0.1 1.0 LeanFormalizer-SFT-7B 18.9 23.3 18.4 26.4 4.8 8.0 LeanFormalizer-CoT-7B 13.5 29.9 8.5 28.2 2.1 9.0 Herald-Translator-7B 13.6 24.7 8.2 27.0 1.3 5.0 Goedel-Formalizer-SonnetAnnotated-32B 18.7 29.2 13.6 27.6 3.4 10.0 Goedel-Formalizer-LeanWorkbookAnnotated-32B 15.1 26.4 5.0 12.1 0.3 2.0 Kimina-Autoformalizer-7B 35.1 60.2 13.3 25.3 2.6 6.0

(Ours) StepFun-Formalizer-7B 38.3 61.2 25.1 37.9 5.2 11.0 StepFun-Formalizer-32B 40.5 59.3 26.7 38.5 6.9 14.0

Table 3: BEq@1 and BEq@16 (%) results of StepFun-Formalizer and baselines on three benchmarks.

Datasets. The datasets we use for reasoning synthesis, RL training, and evaluation are all collected from automated theorem-proving problem sets (Table 2), which contain informal math problems paired with human-annotated (or modelgenerated with manually reviewed) formal statements. For problems containing multiple subproblems or lemmas, we retain only the last one. To prevent data contamination, we perform 13-gram decontamination (Brown et al. 2020) and remove training-evaluation overlaps based on problem names.

Training. We start our training process with DeepSeek-R1Distill-Qwen-7B / 32B. In SFT, we train the models for 2 epochs with a learning rate of 2.0 ˆ 10´5 and a batch size of 128 in the first stage, and 8 in the second stage. In RL, we use a batch size of 128, a learning rate of 1.0ˆ10´6, and train 450 steps for the 7B model and 350 steps for the 32B model. The rollout temperature is 1.0. We use the Kimina Lean Server (Santos et al. 2025), equipped with 100 CPU cores and 400 GB of memory, under a 60-second time limit for equivalence verification. The SFT and RL stages are respectively executed on 8 and 32 A800-80G GPUs. The context lengths are 16384. The training of 7B and 32B takes 45.38 hours and 55.85 hours. After training, we obtain StepFun-Formalizer-7B / 32B.

##### 5.2 Experimental Settings

Benchmarks and Baselines To show the performance of StepFun-Formalizer, we evaluate it on both in-domain and out-of-distribution (OOD) benchmarks. For in-domain evaluation, we use FormalMATH-Lite (Yu et al. 2025b), which shares a similar distribution with our training data (Table 2). For OOD evaluation, we use ProverBench (Ren et al. 2025) and CombiBench (Liu et al. 2025a). Some formal statements in these two datasets include additional definitions, functions, and lemmas. We provide them as prompts and ask the model

to generate main theorems, to evaluate its generalizability. The benchmarks encompass various areas, including algebra, number theory, and calculus, with difficulty levels ranging from high school to undergraduate.

We compare StepFun-Formalizer with advanced generalpurpose LLMs, including o3-pro (OpenAI 2025), Claude4thinking (Anthropic 2025), Gemini-2.5-thinking (Google 2025), DeepSeek-R1 and DeepSeek-R1-Distill-Qwen-7B (DeepSeek-AI 2025). Besides, we also evaluate specialized models, including LeanFormalizer (SJTULean 2024), Herald Translator (Gao et al. 2025), Goedel-Formalizer (Lin et al. 2025a) and Kimina-Autoformalizer (Wang et al. 2025).

Metrics. We use the BEq@k (Liu et al. 2025b) metric to evaluate the models, which is the portion of samples where predicted statements are BEq to ground-truths at least once in k attempts:

ÿN

1 N

BEq@k “

1pyi,j „ yˆiq

max

jPt1,¨¨¨ ,ku

i“1

where N is the sample number; k is the attempt number; 1p¨q is the indicator function, yˆi is the ground-truth and yi,j is the j-th attempt for the i-th sample. We use BEq@1 to measure the accuracy of the model’s single-shot generation, and BEq@16 to evaluate the upper bound of the autoformalization capability. The temperature is set to 0.6.

##### 5.3 Main Results

The evaluation results are shown in Table 3.

Our StepFun-Formalizer model establishes new state-ofthe-art results on both FormalMATH-Lite (in-domain) and ProverBench (OOD), demonstrating the effectiveness and generalization of our data synthesis and training

pipeline. Specifically, even StepFun-Formalizer-7B, surpasses every competing model on both benchmarks, offering a computational efficiency advantage: on FormalMATHLite it exceeds the previous best, Kimina-Autoformalizer-7B, which serves as its formal knowledge source, and on ProverBench it outperforms large general-purpose models such as DeepSeek-R1-671B. In addition, we observe that the 7B model performs comparably to or slightly better than the 32B model, which may be due to the limited size of the data. The 32B model requires more data for further improvement.

Our model outperforms specialized models with the same parameter scale in CombiBench. Since modelling combinatorial problems involves complex real-world scenarios and long contexts, it remains very challenging even for advanced general reasoning models to achieve high formalization accuracy on CombiBench, let alone smaller models. After training, our model shows a substantial improvement over specialized models of the same scale, highlighting our model’s capability in real-world scenario understanding.

##### 5.4 Design Alternatives of Our Method

We explore design alternatives of our method, including the ablation study of the knowledge and reasoning datasets, another reasoning data collection method, and a different base model (see Appendix B). To highlight how each choice impacts generalizable performance, we evaluate them on OOD benchmarks.

Both the knowledge and reasoning datasets contribute to the improvement of autoformalization. We conduct ablation studies to investigate the individual contributions of these two parts (knowledge and reasoning) of our data: StepFun-Formalizer-7B is re-trained with the same approach except removing the first stage (knowledge) and the second stage (reasoning) of SFT (Section 4.3) separately. The results (Table 4) show that informal-to-formal reasoning is the main contributor to model performance, while formal knowledge serves a complementary role. Notably, removing reasoning data leads to a sharp drop in BEq@16, highlighting its importance in boosting the performance upper bound.

ProverBench CombiBench BEq@1 BEq@16 BEq@1 BEq@16

Datasets

ThinkingF (Ours) 25.1 37.9 5.2 11.0 w/o Knowledge 24.5 37.4 3.9 10.0 w/o Reasoning 21.8 25.3 5.3 6.0

Table 4: Comparison of the roles of the two training datasets of the SFT stage.

The designed reasoning template helps the model better perform informal-to-formal translation. To show the effectiveness of the reasoning template, we replace the reasoning data in SFT with reasoning trajectories directly distilled from a general-purpose LLM. Specifically, we prompt Claude4-thinking (Anthropic 2025), a reasoning model of the same series as Claude 3.7 Sonnet, which we used to

synthesize reasoning trajectories, to translate the problems in annotated datasets into formal statements, and use BEq to select the correct translations along with reasoning. The training results are shown in Table 5. It indicates that the reasoning process produced by Claude4-thinking causes a significant decline in the model’s performance. We observe that, during formalization, the general reasoning model devotes its efforts to solving the informal problem instead of formalizing it, and thereby constraining its overall learning capability (refer to Appendix B for examples).

ProverBench CombiBench BEq@1 BEq@16 BEq@1 BEq@16

Method

Template (Ours) 25.1 37.9 5.2 11.0 Direct Distillation 21.8 33.3 4.8 10.0

Table 5: Comparison between template-guided reasoning trajectory synthesis and direct distillation from a generalpurpose reasoning model (Claude4-thinking).

##### 5.5 Further Analysis

Reinforcement learning can consistently improve the model’s autoformalization capability. To show the performance improvement brought by reinforcement learning, we evaluate StepFun-Formalizer-7B every 50 training steps and record the average BEq@1 of all benchmarks. The relationship curve between BEq@1 and the training reward is shown in Appendix B. As training progresses, both the reward and downstream task performance improve in tandem. After 450 training steps, the reward increases from 0.232 to 0.347, and the average BEq@1 improves from 0.258 to 0.303, which underscores the effectiveness of reinforcement learning.

||50.9<br><br>33.6<br><br>53.9<br><br>20.3<br><br>29.5<br><br>60.1<br><br>45.5<br><br>55.6<br><br>36.1<br><br>58.4<br><br>19.4<br><br>35.7<br><br>64.3<br><br>49.4<br><br>|Kimina-Autoformalizer-7B StepFun-Formalizer -7B (Ours)<br><br>|
|---|
|
|---|
<br><br>10<br><br>20<br><br>30<br><br>40<br><br>50<br><br>60<br><br>70<br><br>Algebra Number Theory<br><br>Logic and Puzzles<br><br>Inequalities Calculus Others Total<br><br>Provableformalstatements(%)|
|---|

Provableformalstateme

Figure 5: The proportion (%) of provable formal statements among 10K problems from NuminaMath-1.5 in each domain.

Our model generates more verifiable formal statements in different domains. We conduct a simulation experiment to show the performance of StepFun-Formalizer in end-toend theorem proving from natural language (Xuejun et al. 2025). Specifically, we randomly select 10K problems in NuminaMath-1.5, translate them into formal statements using StepFun-Formalizer-7B and Kimina-Autoformalizer-7B, and then use Kimina-Prover-7B (Wang et al. 2025) to generate 16 proofs for each statement. The results show that

Kimina-Prover proves 4940 formal statements from StepFunFormalizer-7B and 4549 from Kimina-Autoformalizer-7B. Detailed statistics of provable statements in each domain are shown in Figure 5. Our model generates a higher proportion of provable formal statements across all domains except inequalities, highlighting its effectiveness in end-to-end theorem proving.

#### 6 Conclusion

In this paper, we propose ThinkingF, a data synthesis and training pipeline for LLM-based autoformalization. The pipeline mitigates the lack of formal knowledge in generalpurpose models via large-scale distillation and selection, and improves the model’s understanding of natural language problems through template-guided reasoning synthesis, which facilitates complex informal-to-formal translation. We train 7B and 32B models with this pipeline and evaluate them on three benchmarks. Experimental results demonstrate that StepFun-Formalizer outperforms both general-purpose and specialized models on FormalMATH-Lite and ProverBench. We also conduct additional analyses to investigate the roles of each component in the pipeline.

#### Acknowledgements

This work is partially supported by the Strategic Priority Research Program of the Chinese Academy of Sciences (Grants No.XDB0660300, XDB0660301, XDB0660302), Science and Technology Major Special Program of Jiangsu (Grant No. BG2024028), the NSF of China (Grants No. 62341411, 62222214, 6240073476), CAS Project for Young Scientists in Basic Research (YSBR-029) and Youth Innovation Promotion Association CAS.

#### References

Anthropic. 2025. Claude 4. Accessed: 2025-07-24.

Azerbayev, Z.; Piotrowski, B.; Schoelkopf, H.; Ayers, E. W.; Radev, D.; and Avigad, J. 2023. ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics. arXiv:2302.12433.

Barras, B.; Boutin, S.; Cornes, C.; Courant, J.; Coscoy, Y.; Delahaye, D.; de Rauglaudre, D.; Filliˆatre, J.-C.; Gim´enez, E.; Herbelin, H.; et al. 1999. The Coq proof assistant reference manual. INRIA, version, 6(11).

Beg, A.; O’Donoghue, D.; and Monahan, R. 2025. A Short Survey on Formalising Software Requirements using Large Language Models. arXiv:2506.11874.

Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,

- A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. arXiv:2005.14165.

Chen, J.; Tang, H.; Chu, Z.; Chen, Q.; Wang, Z.; Liu, M.; and Qin, B. 2024. Divide-and-Conquer Meets Consensus:

Unleashing the Power of Functions in Code Generation. arXiv:2405.20092.

Chen, Y.; Liu, Y.; Zhou, J.; Hao, Y.; Wang, J.; Zhang, Y.; and Fan, C. 2025. R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning. arXiv:2505.21668.

Community, L. 2023. Lean Documentation - lean3changes. https://lean-lang.org/lean4/doc/lean3changes.html. Accessed: 2025-07-23.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

Gao, G.; Wang, Y.; Jiang, J.; Gao, Q.; Qin, Z.; Xu, T.; and Dong, B. 2025. Herald: A Natural Language Annotated Lean 4 Dataset. arXiv:2410.10878.

Google. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv:2507.06261.

Ji, X.; Liu, Y.; Wang, Q.; Zhang, J.; Yue, Y.; Shi, R.; Sun, C.; Zhang, F.; Zhou, G.; and Gai, K. 2025. Leanabell-Prover-V2: Verifier-integrated Reasoning for Formal Theorem Proving via Reinforcement Learning. arXiv:2507.08649.

Jiang, A. Q.; Li, W.; and Jamnik, M. 2023. Multilingual Mathematical Autoformalization. arXiv:2311.03755.

Kimi; Du, A.; Gao, B.; Xing, B.; Jiang, C.; Chen, C.; Li, C.; Xiao, C.; Du, C.; Liao, C.; et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599.

LI, J.; Beeching, E.; Tunstall, L.; Lipkin, B.; Soletskyi, R.; Huang, S. C.; Rasul, K.; Yu, L.; Jiang, A.; Shen, Z.; Qin, Z.; Dong, B.; Zhou, L.; Fleureau, Y.; Lample, G.; and Polu, S. 2024. NuminaMath.

- Li, Y.; Choi, D.; Chung, J.; Kushman, N.; Schrittwieser, J.; Leblond, R.; Eccles, T.; Keeling, J.; Gimeno, F.; Dal Lago, A.; Hubert, T.; Choy, P.; de Masson d’Autume, C.; Babuschkin,

I.; Chen, X.; Huang, P.-S.; Welbl, J.; Gowal, S.; Cherepanov, A.; Molloy, J.; Mankowitz, D. J.; Sutherland Robson, E.; Kohli, P.; de Freitas, N.; Kavukcuoglu, K.; and Vinyals, O. 2022. Competition-level code generation with AlphaCode. Science, 378(6624): 1092–1097.

- Li, Z.; Sun, J.; Murphy, L.; Su, Q.; Li, Z.; Zhang, X.; Yang,

- K.; and Si, X. 2024. A Survey on Deep Learning for Theorem Proving. arXiv:2404.09939.

- Lin, X.; Cao, Q.; Huang, Y.; Wang, H.; Lu, J.; Liu, Z.; Song,

L.; and Liang, X. 2024. FVEL: Interactive Formal Verification Environment with Large Language Models via Theorem Proving. arXiv:2406.14408.

- Lin, Y.; Tang, S.; Lyu, B.; Wu, J.; Lin, H.; Yang, K.; Li, J.; Xia,

- M.; Chen, D.; Arora, S.; and Jin, C. 2025a. Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving. arXiv:2502.07640. Lin, Y.; Tang, S.; Lyu, B.; Yang, Z.; Chung, J.-H.; Zhao, H.; Jiang, L.; Geng, Y.; Ge, J.; Sun, J.; Wu, J.; Gesi, J.; Lu, X.; Acuna, D.; Yang, K.; Lin, H.; Choi, Y.; Chen, D.; Arora, S.; and Jin, C. 2025b. Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and SelfCorrection. arXiv:2508.03613.

Liu, J.; Lin, X.; Bayer, J.; Dillies, Y.; Jiang, W.; Liang, X.; Soletskyi, R.; Wang, H.; Xie, Y.; Xiong, B.; Yang, Z.; Zhang,

- J.; Zhi, L.; Li, J.; and Liu, Z. 2025a. CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics. arXiv:2505.03171.

Liu, J.; Xia, C. S.; Wang, Y.; and Zhang, L. 2023. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. arXiv:2305.01210.

Liu, Q.; Zheng, X.; Lu, X.; Cao, Q.; and Yan, J. 2025b. Rethinking and Improving Autoformalization: Towards a Faithful Metric and a Dependency Retrieval-based Approach. In The Thirteenth International Conference on Learning Representations.

Liu, Z.; Guo, X.; Lou, F.; Zeng, L.; Niu, J.; Wang, Z.; Xu,

- J.; Cai, W.; Yang, Z.; Zhao, X.; Li, C.; Xu, S.; Chen, D.; Chen, Y.; Bai, Z.; and Zhang, L. 2025c. Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning. arXiv:2503.16252.

Miranda, B.; Zhou, Z.; Nie, A.; Obbad, E.; Aniva, L.; Fronsdal, K.; Kirk, W.; Soylu, D.; Yu, A.; Li, Y.; and Koyejo, S. 2025. VeriBench: End-to-End Formal Verification Benchmark for AI Code Generation in Lean 4. In 2nd AI for Math Workshop @ ICML 2025.

Moura, L. d.; and Ullrich, S. 2021. The Lean 4 Theorem Prover and Programming Language. In Automated Deduction–CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings 28, 625–635. Springer.

- OpenAI. 2024a. GPT-4o System Card. arXiv:2410.21276.
- OpenAI. 2024b. OpenAI o1 System Card. arXiv:2412.16720.

OpenAI. 2025. Introducing O3 and O4 Mini. Accessed: 2025-07-24.

Paulson, L. C. 1994. Isabelle: A Generic Theorem Prover. Springer.

Qwen. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Ren, Z. Z.; Shao, Z.; Song, J.; Xin, H.; Wang, H.; Zhao, W.; Zhang, L.; Fu, Z.; Zhu, Q.; Yang, D.; Wu, Z. F.; Gou, Z.; Ma, S.; Tang, H.; Liu, Y.; Gao, W.; Guo, D.; and Ruan, C. 2025. DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition. arXiv:2504.21801.

Renshaw, D. 2024. Compfiles: Catalog Of Math Problems Formalized In Lean. https://github.com/dwrensha/compfiles. GitHub repository.

Santos, M. D.; Wang, H.; de Saxc´e, H.; Wang, R.; Baksys, M.; Unsal, M.; Liu, J.; Liu, Z.; and Li, J. 2025. Kimina Lean Server: Technical Report. arXiv:2504.21230.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

SJTULean. 2024. LeanFormalizer PPO. https://huggingface. co/SJTULean/LeanFormalizer PPO. Accessed: 2025-07-30.

Thakur, A.; Lee, J.; Tsoukalas, G.; Sistla, M.; Zhao, M.; Zetzsche, S.; Durrett, G.; Yue, Y.; and Chaudhuri, S. 2025. CLEVER: A Curated Benchmark for Formally Verified Code Generation. arXiv:2505.13938.

Tsoukalas, G.; Lee, J.; Jennings, J.; Xin, J.; Ding, M.; Jennings, M.; Thakur, A.; and Chaudhuri, S. 2024. PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition. arXiv:2407.11214.

Wang, H.; Unsal, M.; Lin, X.; Baksys, M.; Liu, J.; Santos, M. D.; Sung, F.; Vinyes, M.; Ying, Z.; Zhu, Z.; Lu, J.; de Saxc´e, H.; Bailey, B.; Song, C.; Xiao, C.; Zhang, D.; Zhang, E.; Pu, F.; Zhu, H.; Liu, J.; Bayer, J.; Michel, J.; Yu, L.; Dreyfus-Schmidt, L.; Tunstall, L.; Pagani, L.; Machado, M.; Bourigault, P.; Wang, R.; Polu, S.; Barroyer, T.; Li, W.D.; Niu, Y.; Fleureau, Y.; Hu, Y.; Yu, Z.; Wang, Z.; Yang, Z.; Liu, Z.; and Li, J. 2025. Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning. arXiv:2504.11354.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q.; Chi, E.; Narang,

- S.; Chowdhery, A.; and Zhou, D. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171. Weng, K.; Du, L.; Li, S.; Lu, W.; Sun, H.; Liu, H.; and Zhang,
- T. 2025. Autoformalization in the Era of Large Language Models: A Survey. arXiv:2505.23486.

Wu, Y.; Huang, D.; Shi, W.; Wang, W.; Pu, Y.; Gao, L.; Liu, S.; Nan, Z.; Yuan, K.; Zhang, R.; Zhang, X.; Du, Z.; Guo, Q.; Yin, D.; Hu, X.; and Chen, Y. 2025. InverseCoder: self-improving instruction-tuned code LLMs with inverseinstruct. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’25/IAAI’25/EAAI’25. AAAI Press. ISBN 978-1-57735-897-8.

Wu, Y.; Jiang, A. Q.; Li, W.; Rabe, M. N.; Staats, C.; Jamnik, M.; and Szegedy, C. 2022. Autoformalization with Large Language Models. arXiv:2205.12615.

Xie, J.; Liu, C.; Yuan, Y.; Li, S.; Xiao, Z.; and Zhang, M. 2025. FMC: Formalization of Natural Language Mathematical Competition Problems. arXiv:2507.11275.

Xin, H.; Guo, D.; Shao, Z.; Ren, Z.; Zhu, Q.; Liu, B.; Ruan, C.; Li, W.; and Liang, X. 2024a. DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data.

Xin, H.; Ren, Z. Z.; Song, J.; Shao, Z.; Zhao, W.; Wang, H.; Liu, B.; Zhang, L.; Lu, X.; Du, Q.; Gao, W.; Zhu, Q.; Yang, D.; Gou, Z.; Wu, Z. F.; Luo, F.; and Ruan, C. 2024b. DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search.

Xuejun, Y.; Zhong, J.; Feng, Z.; Zhai, P.; Yousefzadeh, R.; Ng, W. C.; Liu, H.; Shou, Z.; Xiong, J.; Zhou, Y.; Ong, C. B.; Sugiarto, A. J.; Zhang, Y.; Tai, W. M.; Cao, H.; Lu, D.; Sun, J.; Xu, Q.; Xin, S.; and Li, Z. 2025. Mathesis: Towards Formal Theorem Proving from Natural Languages. arXiv:2506.07047.

Yang, Z.; Chen, L.; Cohan, A.; and Zhao, Y. 2025. Table-R1: Inference-Time Scaling for Table Reasoning. arXiv:2505.23621.

Ying, H.; Wu, Z.; Geng, Y.; Yuan, Z.; Lin, D.; and Chen,

- K. 2025. Lean Workbook: A large-scale Lean problem set formalized from natural language math problems. arXiv:2406.03847.

Ying, H.; Zhang, S.; Li, L.; Zhou, Z.; Shao, Y.; Fei, Z.; Ma, Y.; Hong, J.; Liu, K.; Wang, Z.; Wang, Y.; Wu, Z.; Li, S.; Zhou, F.; Liu, H.; Zhang, S.; Zhang, W.; Yan, H.; Qiu, X.; Wang, J.; Chen, K.; and Lin, D. 2024. InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning.

- arXiv:2402.06332.

Yu, Q.; Zhang, Z.; Zhu, R.; Yuan, Y.; Zuo, X.; Yue, Y.; Dai, W.; Fan, T.; Liu, G.; Liu, L.; Liu, X.; Lin, H.; Lin, Z.; Ma, B.; Sheng, G.; Tong, Y.; Zhang, C.; Zhang, M.; Zhang, W.; Zhu, H.; Zhu, J.; Chen, J.; Chen, J.; Wang, C.; Yu, H.; Song, Y.; Wei, X.; Zhou, H.; Liu, J.; Ma, W.-Y.; Zhang, Y.-Q.; Yan, L.; Qiao, M.; Wu, Y.; and Wang, M. 2025a. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476.

Yu, Z.; Peng, R.; Ding, K.; Li, Y.; Peng, Z.; Liu, M.; Zhang, Y.; Yuan, Z.; Xin, H.; Huang, W.; Wen, Y.; Zhang, G.; and Liu, W. 2025b. FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models. arXiv:2505.02735.

Zhang, J.; Wang, Q.; Ji, X.; Liu, Y.; Yue, Y.; Zhang, F.; Zhang, D.; Zhou, G.; and Gai, K. 2025. Leanabell-Prover: Posttraining Scaling in Formal Reasoning. arXiv:2504.06122.

Zhang, L.; Valentino, M.; and Freitas, A. 2025. Formalizing Complex Mathematical Statements with LLMs: A Study on Mathematical Definitions. arXiv:2502.12065.

Zheng, K.; Han, J. M.; and Polu, S. 2022. MiniF2F: a crosssystem benchmark for formal Olympiad-level mathematics. arXiv:2109.00110.

Zhou, J. P.; Staats, C.; Li, W.; Szegedy, C.; Weinberger, K. Q.; and Wu, Y. 2024. Don’t Trust: Verify – Grounding LLM Quantitative Reasoning with Autoformalization.

- arXiv:2403.18120.

Zhu, Y.; Huang, D.; Lyu, H.; Zhang, X.; Li, C.; Shi, W.; Wu, Y.; Mu, J.; Wang, J.; Zhao, Y.; et al. 2025. CodeV-R1: Reasoning-Enhanced Verilog Generation. arXiv preprint arXiv:2505.24183.

#### A Implementation Details

##### A.1 Informal Problem Preparation

To obtain high-quality informal mathematical problems for knowledge distillation (Section 4.1), we use manual rules to filter the problems in NuminaMath-1.5 (LI et al. 2024). Specifically, we retain data samples that satisfy the following requirements:

- 1. The problem needs to be complete and should not contain multiple sub-questions, which can be determined by the problem is valid field.

- 2. The problem type field should not be Geometry or Combinatorics, since these two types of problems are still challenging to formalize for existing LLMs.

- 3. The problem is a proof problem, or is a calculation problem with an answer containing only numbers, commas, and parentheses. For the latter, we prepend “Show that it is” to the answer and append it to the problem, thereby transforming it into a proof problem.

After filtering, we collect 256K informal problems from the 896K data in NuminaMath-1.5 that are suitable for formal knowledge distillation. We present the distribution of problem types before and after filtering in Section B.3.

##### A.2 Dataset Construction

In the knowledge distillation stage (Section 4.1), we follow Kimina-Autoformalizer (Wang et al. 2025), using the same prompts (Figure 6(a)) and inference hyperparameters. In the reasoning trajectory synthesis stage (Section 4.2), we use the reasoning template (Figure 12, Page 15) with few-shot examples (Figure 13„15, Page 16„21) to prompt claude-3-7-sonnet-20250219 for reasoning synthesis. We set the temperature to 0.6, and the maximum context length is 16384.

Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem. {informal_problem}

- (a)

Autoformalization Prompt

- (b)
- (c)

Your code should start with ```Lean4 {header} ```

You should only output the theorem statement in Lean 4 format, ending with `sorry`. You should NOT output the proof.

Figure 6: The prompt for autoformalization. (a) The original prompt of Kimina-Autoformalizer. (b) The prompt used to indicate which libraries need to be imported, or which predefined functions or lemmas are available for use. (c) Additional requirements to prevent the model from generating a formal proof.

##### A.3 Training

Hyperparameters. For SFT, we perform full-parameter fine-tuning using DeepSpeed ZeRO-3 for parallel, AdamW as

the optimizer, and cosine decay as the learning rate scheduling strategy. We do not use warm-up or packing. The numerical precision of our model is BF16. For RL, we sample 32 data points at each step, and we generate 32 rollouts for each sample. The coefficient for the KL divergence is set to 1 ˆ 10´4. We use a linear scheduler for the learning rate. Other important hyperparameters can be found in Section 5.1.

Prompts. For knowledge datasets, we use the same prompt as Kimina-Autoformalizer (Figure 6(a)). For reasoning datasets and RL training, we add prompts with headers in training data (Figure 6(b)) to align with the evaluation of OOD tasks, which may provide additional definitions or lemmas for the model to apply.

##### A.4 Evaluation

Hyperparameters. We align the hyperparameters of the evaluated specialized models with those they released, if available. For the general-purpose models and StepFunFormalizer, we set the temperature to 0.6 and the maximum context length to 16384.

Prompts. For a specialized model, if its training prompt has been released, we use the released prompt for evaluation, such as Goedel-Formalizer (Lin et al. 2025a). Otherwise, we use the prompt in Figure 6(a). Following EvalPlus (Liu et al. 2023), we prepend the header in benchmarks as the response prefix for specialized models to maximize their performance. For general-purpose models, since some of them do not understand the autoformalization task well (e.g., DeepSeek-R1-Distill-Qwen-7B), we add additional instructions (Figure 6(c)) to prevent them from generating a formal proof. For StepFun-Formalizer, we keep the evaluation prompt consistent with RL training, i.e. Figure 6(a)(b).

##### A.5 BEq Verification

Using an additional theorem-proving model to check the equivalence of formal statements incurs significant computational overhead, while providing only marginal accuracy gains. We randomly sample 320 formal statements generated by the model and conduct equivalence checking with the ground truth using both Lean4’s built-in search tactic exact? and InternLM-Math-Plus-20B (Ying et al. 2024), as in the BEq paper (Liu et al. 2025b). The results show that the latter identified only 3 more correct formal statements than the former, while taking approximately 4 times longer time cost (3 min vs 12 min) and requiring additional GPU resources (CPU only vs 8 A100-80G GPUs). Considering the computational cost of RL training and large-scale evaluation, we choose to use only exact? for equivalence checking.

#### B Further Analysis

##### B.1 Categorical Analysis for Errors in Autoformalization Models

To identify the key factors affecting the performance of the autoformalization model (Section 1), we employ two human experts in Lean 4 to conduct error analysis on 100 modelgenerated formal statements. The most common error types

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(a) Kimina-Autoformalizer-7B

(b) StepFun-Formalizer-7B Figure 7: Categorical analysis for errors in two models.

identified by the experts in their analysis are detailed as follows:

- 1. Misunderstanding of the original meaning of the problem.
- 2. Missing or incorrect addition of mathematical concepts or objects involved in the problem (e.g., the constant was missing in indefinite integrals).
- 3. Missing or incorrect addition of conditions or conclusions in the problem (e.g., the verification of the equality condition was missing in inequality problems).
- 4. Mapping natural language mathematical objects to incorrect Lean 4 objects.
- 5. Logical structure errors (e.g., the informal problem implies a “if and only if” relationship, but the formal statement only includes one direction).
- 6. Type mismatch or undeclared variable types (e.g., the problem does not specify the type of a mathematical object, but this type must be explicitly stated in the formal language; otherwise, a syntax error will occur).

We further abstract the first three types of errors as Natural Language Misunderstanding, where the model misinterprets certain natural language concepts before converting them into formal language. The last three types of errors are summarized as Informal-Formal Misalignment, where the natural language problem is correctly understood, but mistakes in formal language occur during the translation process.

Then, we used GPT-4o to conduct error analysis on roughly 10K formal statements generated by Kimina-Autoformalizer (Wang et al. 2025) and StepFun-Formalizer across three benchmarks. Based on the analysis, we identified the distribution of two main errors. The detailed statistical results are shown in Figure 7. It demonstrates that these two types of errors can be mitigated by StepFun-Formalizer. The evaluation prompts are elaborated in Figure 8.

###### Categorical Analysis for Errors

You will receive a math problem consisting of its natural language statement and, in some cases, a natural language proof or solution, along with its formal statement in LEAN 4. There are some inconsistencies between the formal statement and the natural language math problem. Please summarize a few keywords to describe the categorize of inconsistencies, for example:

- - Missing or incorrect addition of conditions or conclusions in the problem
- - Misunderstanding of the original meaning of the problem
- - Logical structure errors, such as confusing if-and-only-if with implication
- - Missing or incorrect addition of mathematical concepts or objects involved in the problem
- - Type mismatch or undeclared variable types
- - Mapping natural language mathematical objects to incorrect Lean 4 objects

Your keywords should be concise, including but not limited to the examples mentioned above. If no inconsistencies are found, output None as the keyword.

[YOUR TASK HERE] ### Natural Language Statement {problem} ### Formal Statement in LEAN 4 ```lean4 {formal_statement} ```

Your answer should be in the following format: ### Keywords:

- - keywords1
- - keywords2
- - keywords3

... [Your keywords to describe the categories of inconsistencies if they exist, otherwise None]

Figure 8: The prompt for LLM-based error type analysis.

##### B.2 Knowledge Data Selection by LLM

LLM-based data selection in knowledge distillation can improve training efficiency without compromising performance. In Section 4.1, we use DeepSeek-V3 to conduct validity filtering on the distilled formal statements. Specifically, we remove invalid problems due to tautology, contradictions, irrelevance, or triviality. The prompt is shown in Figure 9. After LLM selection, the total data volume decreased from 253913 to 182548. The distribution of problem types before and after LLM selection is shown in Section B.3.

To verify the effectiveness of LLM selection, we conduct an additional ablation experiment to compare the training performance (Table 6), with and without LLM selection. The results show that using LLM selection can reduce the amount of training data with minimal impact on performance.

Distillation Data Selection Using LLM

You are an expert in mathematical logic and Lean 4. Given a Lean 4 math problem (in the form of a theorem, lemma, or example), analyze and judge the quality of the problem. First, provide a brief analysis of the problem, discussing whether:

-- The problem is a tautology (the conclusion follows trivially from the premises).

-- The premises are contradictory (impossible for all premises to be true at the same time).

-- The statement to prove is irrelevant to the premises (the conclusion does not logically depend on the premises).

-- The problem is otherwise trivial, ill-posed, or uninteresting.

-- Or, if the problem is well-posed, non-trivial, and interesting. After your analysis, output only a single digit on a new line:

- -- Output 0 if the problem quality is bad (any of the first four issues above).
- -- Output 1 if the problem quality is good (well-posed, non-trivial, and interesting). Format your response as follows: Analysis: <your analysis here>

<0 or 1>

Figure 9: The prompt for LLM-based data selection.

ProverBench CombiBench BEq@1 BEq@16 BEq@1 BEq@16

Method

ThinkingF (Ours) 25.1 37.9 5.2 11.0 w/o LLM selection 25.4 35.6 4.9 13.0

Table 6: Performance comparison of with and without LLM selection in knowledge data distillation.

##### B.3 Distribution of Problem Types

We report the proportions of problem types in datasets for (1) original 896K informal problems in NuminaMath-1.5, (2) 256K subset filtered by manual rules, and (3) 183K subset with formal statements further selected via LLM-based methods. The distribution of problem types is shown in Figure 10. It shows that the rule-based method tends to filter out more algebra and number theory problems, which are easier to formalize. After LLM-based selection, the proportion of logic and puzzle problems decreases, suggesting that some of the formalizations in this category are invalid.

70

| |47.5%<br><br>3.0%<br><br>4.7% 4.3%<br><br>10.3%<br><br>1.4%<br><br>8.3%<br><br>20.6%<br><br>60.7%<br><br>1.4%<br><br>5.7%<br><br>7.4%<br><br>23.0%<br><br>1.8%<br><br>59.7%<br><br>1.8%<br><br>7.4%<br><br>4.2%<br><br>26.3%<br><br>0.6%<br><br>NuminaMath-1.5<br><br>After Rule-Based Filtering<br><br>After LLM-Based Selection| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

60

ProblemCategoryPercentage(%)

50

40

30

20

10

0

Other Combinatorics Geometry

Algebra Calculus Inequalities Logic and Puzzles

Number Theory

Figure 10: Comparison of problem type distributions.

##### B.4 Alternative Base Models

Training from a general-purpose reasoning model helps improve the model’s formalization capability. We replace the base model from DeepSeek-R1-Distill-Qwen-7B to Kimina-Autoformalizer-7B in training, which possesses more comprehensive formal knowledge rather than general reasoning capabilities. The results (Table 7) show that a base model with strong math and coding reasoning capabilities also contributes to the autoformalization process.

ProverBench CombiBench BEq@1 BEq@16 BEq@1 BEq@16

Models

StepFun-Formalizer-R1D (Ours) 25.1 37.9 5.2 11.0 StepFun-Formalizer-KM 21.4 30.5 4.3 8.0

Table 7: The impact of different base models on the training performance of StepFun-Formalizer. “R1D”: DeepSeek-R1Distill-Qwen-7B; “KM”: Kimina-Autoformalizer-7B.

##### B.5 Training Curve in Reinforcement Learning

0.31

Weighted Avg BEq@1

0.400

Reward (Smoothed)

0.30

0.375

WeightedAvgBEq@1

0.350

Reward(Smoothed)

0.29

0.325

0.300

0.28

0.275

0.27

0.250

0.225

0.26

0 100 200 300 400 500 600

Step

Figure 11: The performance curves of reward and weighted average BEq@1 across all benchmarks.

RL training curves of the reward and average BEq@1 on three benchmarks are shown in Figure 11. We observe that both the reward and the performance of downstream

tasks increase as RL progresses, which demonstrates the effectiveness of RL training.

##### B.6 Off-Task Behaviours in General Reasoning LLMs During Formalization

We observe that general-purpose reasoning models like Claude4-thinking tend to exhibit off-task behaviour during the formalization reasoning process. Specifically, they devote much of their reasoning to solving the informal problem rather than formalizing it, yet still manage to produce the correct formal statement. Figure 16 (Page 22) shows an example. This phenomenon suggests that directly distilling general reasoning models to obtain autoformalization reasoning trajectories may harm the model’s performance (Section 5.4). That is why we use a manually designed reasoning template for reasoning data synthesis (Section 4.2).

##### B.7 Applications: Data synthesis for Training Theorem-proving Models

Our model can be used to synthesize more diverse verifiable formal statements, thereby facilitating the training of theorem-proving models. Inspired by Goedel-Formalizer (Lin et al. 2025a), we attempt to combine the formalization capabilities of StepFun-Formalizer and Kimina-Autoformalizer to provide more diverse training data for theorem-proving models.

We conduct an experiment to simulate the process of distilling the theorem-proving ability from a larger model into a smaller one. Specifically, we employ StepFun-Formalizer and Kimina-Autoformalizer to formalize the 256K problems from NuminaMath-1.5, and then employ Kimina-Prover-7B to generate the corresponding formal proofs. We merge and deduplicate the data from two models and verify the proofs by Lean 4 REPL. The verified proofs are used to fine-tune Qwen2.5-Math-1.5B-Instruct, which is the base model of Kimina-Prover-1.5B. We use FormalMATH-Lite (Yu et al. 2025b) and CombiBench (Liu et al. 2025a) to evaluate the theorem-proving performance of the models fine-tuned with

- 4 different kinds of synthetic datasets (Table 8):

FormalMATH-Lite CombiBench

Data Source Data Size

Pass@1 Pass@32 Pass@1 Pass@32

Kimina-Prover-1.5B Unknown 30.2 43.5 1.0 4.0 Kimina 89473 31.4 44.9 1.0 2.0 Kimina ˆ 2 112121 32.5 45.4 1.0 3.0 Kimina + Ours 120602 32.5 46.4 1.8 4.0

Table 8: Performance (%) comparison between theoremproving models trained on different distilled datasets.

- 1. Kimina-Prover-1.5B: The official model, distilled from Kimina-Prover-72B.
- 2. Kimina: Generating one formal statement using KiminaAutoformalizer.
- 3. Kimina ˆ 2: Generating two formal statement using Kimina-Autoformalizer with deduplication.

4. Kimina + Ours: One formal statement from KiminaAutoformalizer, and another formal statement from StepFun-Formalizer, followed by deduplication.

The results demonstrate that the combined dataset generated from StepFun-Formalizer and Kimina-Autoformalizer is the largest under the same computation cost, and the model trained on it achieves the best performance, even surpassing the official model, which implies that our model can generate more diverse verifiable formal statements, thereby promoting the development of theorem-proving models.

##### B.8 Distribution Differences Across Benchmarks

Since we use a subset of FormalMATH (Yu et al. 2025b) (FormalMATH-Lite) for evaluation and the remaining portion for training, we consider FormalMATH-Lite as an indomain benchmark. To highlight the differences between FormalMATH-Lite and ProverBench (Ren et al. 2025), we compute the n-gram perplexities within each dataset and between the two datasets:

Training Dataset Test Dataset Perplexity FormalMATH-Lite FormalMATH-Lite 12.31 FormalMATH-Lite ProverBench 436.20 ProverBench ProverBench 12.11 ProverBench FormalMATH-Lite 468.00

- Table 9: N-gram perplexities within and between datasets.

Specifically, we train an n-gram model on the dataset in the first column (Training Dataset) in Table 9 and compute the average perplexity on the dataset in the second column (Test Dataset). The results show that the average OOD ratio between FormalMATH-Lite and ProverBench is 37.04, indicating a large distributional difference.

B.9 Comparison with Goedel-Formalizer-V2

We compare StepFun-Formalizer with the concurrent work Goedel-Formalizer-V2 (2025-08-05) (Lin et al. 2025b), which distils 50K reasoning data from Claude Sonnet 4, at a cost substantially higher than that of our data synthesis method. The evaluation results are shown in Table 10:

Models

ProverBench CombiBench BEq@1 BEq@16 BEq@1 BEq@16

StepFun-Formalizer-7B (Ours) 25.1 37.9 5.2 11.0 Goedel-Formalizer-V2-8B 16.3 28.7 3.3 10.0 StepFun-Formalizer-32B (Ours) 26.7 38.5 6.9 14.0 Goedel-Formalizer-V2-32B 15.6 28.2 4.1 13.0

- Table 10: Performance (%) comparison between our model and Goedel-Formalizer-V2.

We note that Goedel-Formalizer-V2’s training prompt does not include headers required in BEq, which causes difficulties for our evaluation. Since it is a reasoning model, we cannot simply take headers as prefixes in its output, as is done for

Kimina-Autoformalizer. Although we attempt to add headers to its prompts, the model’s limited instruction-following ability prevents it from generating specified headers. As a result, its BEq scores may be underestimated. Some works note this issue and replace BEq with LLM-as-a-judge, but it may introduce additional bias due to the randomness of LLM’s outputs. We will refine our evaluation methodology in future work.

#### C Case Study

A detailed case study of StepFun-Formalizer ’s comprehensive reasoning trajectories is presented in Figure 17 (Page 23„24).

The Prompt for Informal-to-Formal Reasoning Synthesis

You are given a natural language math problem along with its corresponding formal theorem statement in Lean 4 code. You need to provide a step-by-step analysis and reasoning of the process that bridges the gap from the natural language formulation to its Lean 4 formalization, including but not limited to the following parts:

- 1. Start with a succinct restatement of the natural language math problem.
- 2. Outline the high-level logical structure of the mathematical problem, such as if-and-only-if, implication, universal quantification, or conjunction. Then, offer an in-depth analysis of the formalization process specific to this type of logical structure, carefully identifying any subtleties, potential pitfalls, or technical considerations that must be addressed in Lean 4.
- 3. Analyze the mathematical concepts involved in the problem using natural language.
- 4. Carefully analyze and then determine the mathematical objects to be defined and their corresponding types (e.g., natural numbers) based on the mathematical concepts.
- 5. Analyze the headers needed before formalizing the problem into a Lean 4 theorem, including importing libraries, opening namespaces, executing additional commands, or defining auxiliary functions, etc.
- 6. Anticipate tricky Lean issues (e.g., implicit type coercions, automatic variable declarations, operator overloading, etc.).
- 7. Map the mathematical concepts, objects and operations mentioned in the problem to their corresponding representations in Lean 4. Before mapping each natural language mathematical item, first analyze its correspondence with the relevant Lean 4 objects in as much detail as possible.
- 8. Finally, review and output the complete Lean 4 code, ensuring it is fully consistent with the given formal theorem statement. Depth requirement:

- 1. Each item should be as detailed as possible, and each subpoint should contain more than three sentences.
- 2. Your reasoning must focus entirely on **formalization**, that is, how to translate the natural language problem into Lean 4 code step by step, and should NOT include any analysis of how to prove the statement. Your response should be entirely in the tone of imitating the reasoning process, without exposing the task description, such as using terms like “provided code” or “given code” Avoid using such expressions. Here are some examples. Your reasoning should be more detailed and much longer than them („ 4000 words):

- [Example 1]

- {Few Shots 1}

[Example 2]

- {Few Shots 2}

[Example 3]

- {Few Shots 3}

[YOUR TASK HERE] ### Natural Language Problem {informal statement}

### Lean 4 Code ```Lean4 {formal statement} ```

Figure 12: The complete prompt for informal-to-formal reasoning trajectory synthesis.

- Few-Shot Example 1 in Reasoning Trajectory Synthesis.

### Natural Language Problem: Suppose that f is holomorphic in an open set Ω. Prove that if Repfq is constant, then f is constant. ### Lean 4 Code: ```Lean4

import Mathlib open Complex Function Metric open scoped Topology theorem my_favorite_theorem {f : C Ñ C} (Ω : Set C) (a b : Ω) (h : IsOpen Ω)

(hf : DifferentiableOn C f Ω) (hc : D (c : R), @ z P Ω, (f z).re = c) : f a = f b :=

sorry

``` [Response] We should formalize the following theorem: If a function f is holomorphic in an open set Ω and its real part Repfq is constant, then f must be a constant function.

The problem involves a conditional statement, which is a typical ”if-then” structure. More formally, it follows an implication of two assumption leading to a conclusion: Assumption:

- 1. Function f is holomorphic in an open set Ω.
- 2. The real part of f is constant. Conclusion: f is constant on Ω. In addition, ”Repfq is constant” implies that for any input, the real part of f is the same. This needs to be described using both the existential quantifier (D) and the universal quantifier (@).

There are some mathematical concepts in the problem:

- 1. holomorphic A holomorphic function is a complex-valued function of one or more complex variables that is complex differentiable in a neighbourhood of each point in a domain in complex coordinate space.
- 2. open set An open set in a metric space is a set that, with every point P in it, contains all points of the metric space that are sufficiently near to P.
- 3. Re It means taking the real part of a complex function.
- 4. constant The first “constant” refers to a fixed number, a value that remains unchanged. The second “constant” refers to a constant function, which is a type of function that always produces the same output, regardless of the input. By analyzing the variable names in the problem, we find that it involves the following mathematical entities:

- 1. f Since the problem mentions that f is “holomorphic”, it should be a mapping from complex numbers to complex numbers.
- 2. Ω A set of complex numbers. Now we can do some preparatory work for formalizing this problem in Lean 4: First, we need to import the mathematical library `Mathlib`. Since both f and Ω are defined on the complex domain, the `Complex`namespace is required. Additionally, since the problem involves functions and metric spaces, we also need the `Function`and `Metric`namespaces. Furthermore, since the open set is a concept in topology, we will need the `Topology`namespace as well. In summary, we need to execute

the following Lean 4 code before formalizing the problem: ```Lean4

import Mathlib open Complex Function Metric open scoped Topology

``` Next, There are several potential issues when formalizing this proof in Lean:

- 1. Implicit type coercion. Lean’s handling of complex numbers and real numbers can sometimes introduce implicit coercions. Specifically, f is a complex-valued function, and we need to ensure that Lean correctly interprets the real part of f as a real number.
- 2. Type class resolution. We must ensure that Lean can resolve the necessary type classes, such as differentiability and the real part of a complex function.
- 3. Correctness of quantification. We are quantifying over elements of the set Ω, so we must ensure that the universal quantifier is applied properly to all complex number in Ω. Then, let’s analyze the Lean 4 code corresponding to the specific concepts and objects in the problem:

- 1. An open set Ω on the complex domain. In Lean, a set in the complex domain is represented by `Ω : Set C`, which means Ω is a collection of complex numbers. We use `IsOpen`predicate to ensure the topological open property for Ω.

###### ```

Ω : Set C h : IsOpen Ω

###### ```

- 2. A holomorphic function f. We first establish that f is defined in the complex plane, i.e. `f : C Ñ C`. Next, we focus on the holomorphic property. In Lean, the `DifferentiableOn`predicate checks that f is complex differentiable at every point in the open set Ω, which ensures that f is holomorphic.

```

f : C Ñ C hf : DifferentiableOn C f Ω

```

- 3. Repfq is constant. It means that there exists (D) a real number c such that for any (@) z in Ω, the real part of f(z) is equal to c. We use this “existential-universal” logical structure to describe “the real part of f is constant.”

```

hc : D (c : R), @ z P Ω, (f z).re = c

```

- 4. f is a constant function. The final goal is to prove that the function f is constant on Ω, meaning that f takes the same value at any two points a and b within Ω. In other words, the function f satisfies the condition f(a) = f(b), which would imply that f is constant on Ω. We can formalize the goal as `@a b : Ω, f a “ f b`, but alternatively, we place `a b : Ω `in the conditions, as this better aligns with common expression practices.

###### ```

a b : Ω f a = f b

``` In conclusion, the complete Lean 4 code is as follows: ```Lean4

import Mathlib open Complex Function Metric open scoped Topology theorem my_favorite_theorem {f : C Ñ C} (Ω : Set C) (a b : Ω) (h : IsOpen Ω)

(hf : DifferentiableOn C f Ω) (hc : D (c : R), @ z P Ω, (f z).re = c) : f a = f b :=

sorry

```

Figure 13: The first few-shot example for informal-to-formal reasoning trajectory synthesis.

- Few-Shot Example 2 in Reasoning Trajectory Synthesis.

### Natural Language Problem: If Tα is a family of topologies on X, show that

Ş Tα is a topology on X.

### Lean 4 Code: ```Lean4

import Mathlib open Set TopologicalSpace open scoped Topology def is_topology (X : Type*) (T : Set (Set X)) :=

H P T ^ univ P T ^ (@ s t, s P T Ñ t P T Ñ s X t P T) ^ (@s, (@t P s, t P T) Ñ sUnion s P T)

theorem my_favorite_theorem (X I : Type*) (T : I Ñ Set (Set X)) (h : @ i, is_topology X (T i)) : is_topology X (

Ş

i : I, T i) := sorry

###### ```

[Response] Let’s restate the natural language math problem: The intersection of a family of topologies on X, i.e.

Ş Tα, is also a topology on X. The logical structure of the problem involves a universal quantification and an implication:

- 1. Universal quantification (@). For each index α, Tα is a topology on X.
- 2. Implication.

The problem follows an “if-then” structure, where the assumption is that Tα is a family of topologies, and the conclusion is that the intersection of these topologies is also a topology. We should formalize these two parts separately.

Ş

The problem involves two mathematical concepts, topology and

:

- 1. Topologies on sets. Let X be a set and let T be a family of subsets of X. Then T is called a topology on X if:

- (1) Both the empty set and X are elements of T.
- (2) Any intersection of finitely many elements of T is an element of T.
- (3) Any union of elements of T is an element of T.

- 2.

Ş Tα It means the intersection of all sets in T. We also identify these mathematical entities in the natural language problem:

- 1. Set X and its set family. We use `set X` to denote a set of elements of an arbitrary type, and `set (set X)` to denote the set of all its subsets, i.e., a family of sets.
- 2. A topology on X. According to the definition of a topology, its type is also `set (set X)`.
- 3. A family of topologies. It means a set of all topologies on X. We can use a mapping from an index type I to a family of sets to define a family of topologies.

Before considering the formalization of the statement, we should add some helper code. We should import `Mathlib` first. Since the problem is about the topologies on set, we can open namespaces `Set`, `TopologicalSpace` and `Topology`. When we are already in these namespaces, we can define an auxiliary function `is topology`to simplify the formalization of the problem. The input of this function is a family of sets `T`, and the output is whether it satisfies the three defining properties of a topology. Here are the helper code: ```Lean4

import Mathlib open Set TopologicalSpace open scoped Topology def is_topology (X : Type*) (T : Set (Set X)) :=

H P T ^ univ P T ^ (@ s t, s P T Ñ t P T Ñ s X t P T) ^ (@s, (@t P s, t P T) Ñ sUnion s P T)

``` We should pay attention to the following potential formalization issues in Lean 4:

- 1. Function application in dependent types. Since the family of topologies should be formalized as a function like `T : I Ñ Set (Set X)` in Lean, we must be careful with how we apply it and refer to elements of `T i` . Mistakenly writing or misplacing parentheses could lead to type mismatch errors. Lean is strict about such syntax.

- 2. Make full use of already defined functions. So far, we have defined the auxiliary function `is topology` to determine whether a family of sets forms a topology. We should invoke it in both the conditions and the conclusion of the formalization, rather than repeatedly write out the definition of a topology. Next, let’s examine the different parts of Lean 4 code that corresponds to the problem:

- 1. Declaration of Tα. We need a type `X` to represent the underlying set over which the topologies are defined. Additionally, we introduce a type `I` to serve as the indexing set for the family of topologies, which corresponds to the symbol α in the natural language formulation. Building on these type declarations, we can use the function `T : I Ñ Set (Set X)` to represent the family of topologies. ```

X I : Type* T : I Ñ Set (Set X)

```

- 2. Tα is a family of topologies on X. It is the premise of the proposition to be formalized. We consider over all indices `i` and apply `is topology` to ensure that each corresponding collection `T i` is a topology under type `X`.

```

h : @ i, is_topology X (T i)

```

- 3.

Ş Tα is a topology on X. It is the conclusion. We use `(

Ş

i : I, T i)` to represent the intersection of all the topologies, and similarly apply `is topology` to claim that

it is also a topology. ```

Ş

is_topology X (

i : I, T i)

``` Above all, the complete Lean 4 code is as follows: ```Lean4

import Mathlib open Set TopologicalSpace open scoped Topology def is_topology (X : Type*) (T : Set (Set X)) :=

H P T ^ univ P T ^ (@ s t, s P T Ñ t P T Ñ s X t P T) ^ (@s, (@t P s, t P T) Ñ sUnion s P T)

theorem my_favorite_theorem (X I : Type*) (T : I Ñ Set (Set X)) (h : @ i, is_topology X (T i)) : is_topology X (

Ş

i : I, T i) := sorry

```

Figure 14: The second few-shot example for informal-to-formal reasoning trajectory synthesis.

- Few-Shot Example 3 in Reasoning Trajectory Synthesis.

### Natural Language Problem: Prove that the Cauchy product of two absolutely convergent real series converges absolutely.

### Lean 4 Code: ```Lean4

import Mathlib open Topology Filter Real Finset open scoped BigOperators theorem my_favorite_theorem

(a b : N Ñ R)

- (ha : D y, (Tendsto (λ n => (Σ i in (range n), |a i|)) atTop (N y)))
- (hb : D y, (Tendsto (λ n => (Σ i in (range n), |b i|)) atTop (N y))) : D y, (Tendsto (λ n => (Σ i in (range n),

λ i => (Σ j in range (i + 1), |a j * b (i - j)|))) atTop (N y)) := sorry

###### ```

[Response] We are asked to formalize the following natural language statement to Lean 4: If two infinite series made from sequences of real numbers are absolutely convergent, then the series formed by multiplying them term by term in the Cauchy product sense also converges absolutely.

The logical structure of the problem is a classic implication involving existential quantification (D) and series convergence:

- 1. Premises: Two infinite series are absolutely convergent, meaning the limit of the sum of the absolute values of their terms exists.
- 2. Conclusion: The Cauchy product of the two series converges absolutely. In this problem, we encounter some mathematical concepts:

- 1. Cauchy product

Let

ř8

i“0 ai and

ř8

j“0 bj be two infinite series with real terms. The Cauchy product of these two infinite series is defined by a discrete convolution as follows: ˜ 8

ÿ

i“0

ai¸

¨

˜ 8

ÿ

j“0

bj¸

“

ÿ8

k“0

ck

where ck “ řk

l“0 albk´l.

- 2. Absolutely convergent series

ř8

n“0 |an| “ L for some real number L. Through analyzing the variables in the problem, we recognize that it involves these mathematical entities:

An infinite series of numbers an is said to converge absolutely (or to be absolutely convergent) if

- 1. Two real series We represent these two series using mappings from natural numbers to real numbers, and denote them as a and b.
- 2. The Cauchy product Directly compute the Cauchy product of a and b using discrete convolution. This is also a mapping, and for simplicity, we can represent it using a λ-expression.
- 3. The limit of the partial sums of a series. We need to characterize the absolute convergence of a series by the existence of the limit of its partial sums. This limit value must be explicitly defined using an existential quantifier.

Prior to formalizing the statement, we need to include some auxiliary code. With Mathlib imported, we characterize the convergence of a series by considering the limit as they tend to infinity. This can be done using `Tendsto` and `atTop` from the `Filter` module, so we need to open the `Filter` namespace. We also need `Topology` to use N to describe the neighborhood. Since the problem is considered in the real number domain, we need to open `Real` namespace. For convenience in expression, we also use `Finset` and `BigOperators` to represent summation over a specific range. The Lean 4 code is shown below: ```lean4

import Mathlib open Filter Topology Real Finset open scoped BigOperators

``` Before delving into the details of Lean 4, we need to be mindful of the following points:

- 1. Managing nested sums and indexing dependencies. The Cauchy product involves double summation where inner indices depend on outer ones. It’s crucial to ensure that index bounds are valid to prevent undefined expressions.
- 2. Formalizing convergence with filters. Lean uses `Tendsto` and the `atTop` filter to express convergence, which refers to the convergence of partial sums. Confusing this with convergence of the series itself can lead to incorrect formulations.

- 3. Balancing implicit inference and explicit types.

ř

Lean performs implicit type inference for notations like `

` and `| ¨ |` , but this can fail in complex expressions. Adding explicit type

annotations when needed helps prevent type-checking errors. Now, let’s break down the Lean 4 code associated with the specific concepts and objects in the problem:

- 1. Two real series. In Lean 4, we represent the two infinite series `a` and `b` as functions from the set of natural numbers `N` to the set of real numbers `R` .

```

a b : N Ñ R

```

- 2. The counterparts of the partial sums of the absolute series. For a fixed n, the expression `

ř

i in range n, | a i |` computes

řn´1

i“0 |ai|, which is how the partial sums are formalized. We formalize them in terms of λ-expression for the further formalization of absolute convergence. ```

- λ n => (Σ i in (range n), |a i|
- λ n => (Σ i in (range n), |b i| ```

- 3. Absolutely convergent series.

We should formalize that

ř8

i“0 |ai| converges. In Lean, this is expressed by saying that the sequence of partial sums tends to some real limit `y` as `n` tends to infinity, using the `Tendsto` predicate, the `atTop` filter, and the `N y` notation: In Lean, `Tendsto` is a predicate used to formalize the notion of convergence of functions with respect to filters. Specifically, `Tendsto f l1 l2` means that the function `f` tends to the filter `l2‘ as its input varies along the filter `l1`. The `atTop` filter on natural numbers captures the idea of “tending towards infinity”. In other words, `atTop` describes the behavior of a sequence as its index grows without bound. The notation `N y` denotes the neighborhood filter of the point `y` in a topological space (here, the real numbers R). ```

- ha : D y, (Tendsto (λ n => (Σ i in (range n), |a i|)) atTop (N y))
- hb : D y, (Tendsto (λ n => (Σ i in (range n), |b i|)) atTop (N y)) ```

- 4. Their Cauchy product.

For each n, the outer sum computes

řn´1

i“0 ci, where each ci is itself a sum of products

ři

j“0 |ajbi´j|; that inner sum is exactly the j-indexed expression inside. Be mindful of indexing: since `Finset.range k`represents the half-open interval [0, k), if we want to include k in the range, we must use `Finset.range (k + 1)`instead. ```

λ n => (Σ i in (range n), λ i => (Σ j in range (i + 1), |a j * b (i - j)|))

```

- 5. The absolute convergence of the Cauchy product. Similar to the earlier formalization of absolute convergence.

###### ```

D y, (Tendsto (λ n => (Σ i in (range n),

λ i => (Σ j in range (i + 1), |a j * b (i - j)|))) atTop (N y))

``` In summary, the complete Lean 4 code is outlined as follows: ```Lean4

import Mathlib open Topology Filter Real Finset open scoped BigOperators theorem my_favorite_theorem

(a b : N Ñ R)

- (ha : D y, (Tendsto (λ n => (Σ i in (range n), |a i|)) atTop (N y)))
- (hb : D y, (Tendsto (λ n => (Σ i in (range n), |b i|)) atTop (N y))) : D y, (Tendsto (λ n => (Σ i in (range n),

λ i => (Σ j in range (i + 1), |a j * b (i - j)|))) atTop (N y)) := sorry

```

Figure 15: The third few-shot example for informal-to-formal reasoning trajectory synthesis.

Off-Task Example of Claude4-thinking Prompt: Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my favorite theorem. Let a and b be positive integers and satisfy the relation

###### p11111 ` aqp11111 ´ bq “ 123456789.

Prove that a ´ b is a multiple of 4. Your code should start with: ```Lean4 import Mathlib ``` You should only output the theorem statement in Lean 4 format, ending with `sorry `. You should NOT output the proof. Response (Claude4-thinking): <think> Let me think about this problem. We have:

- - a and b are positive integers
- - p11111 ` aqp11111 ´ bq “ 123456789 We need to prove that a ´ b is a multiple of 4. Let me first expand the left side: p11111 ` aqp11111 ´ bq “ 111112 ` a ¨ 11111 ´ b ¨ 11111 ´ ab “ 111112 ` 11111pa ´ bq ´ ab So we have: 111112 ` 11111pa ´ bq ´ ab “ 123456789 Let me calculate 111112: 111112 “ 123454321 So: 123454321 ` 11111pa ´ bq ´ ab “ 123456789 11111pa ´ bq ´ ab “ 123456789 ´ 123454321 11111pa ´ bq ´ ab “ 2468 So we have: 11111pa ´ bq ´ ab “ 2468 Now let me think about this modulo 4. First, what is 11111 pmod 4q? 11111 “ 4 ¨ 2777 ` 3, so 11111 ” 3 pmod 4q. What is 2468 pmod 4q? 2468 “ 4 ¨ 617, so 2468 ” 0 pmod 4q. So we have: 11111pa ´ bq ´ ab ” 0 pmod 4q 3pa ´ bq ´ ab ” 0 pmod 4q Now let me think about this differently. Let’s consider a and b modulo 4.

... </think> Looking at this problem, I need to formalize the statement that for positive integers a and b satisfying p11111`aqp11111´bq “ 123456789, we have that a ´ b is divisible by 4. ```Lean4 import Mathlib theorem my_favorite_theorem (a b : Z)

(ha : 0 < a) (hb : 0 < b) (h : (11111 + a) * (11111 - b) = 123456789) : 4 | (a - b) := by sorry

```

- Figure 16: An example of off-task during formalization (Claude4-thinking). It spends an extremely long reasoning process on solving the natural language problem itself rather than translating it into formal language. Strangely, it still outputs the correct formal statement in the end.

Case Study: abstract algebra p15 (ProverBench) Informal Problem: Let Z3 be the finite field of integers modulo 3, and let c P Z3. The quotient ring Z3rxs{px3 ` x2 ` cq is a field if and only if c “ 2.

Headers: ```Lean4

import Mathlib open Polynomial Ideal

variable {R : Type*} [CommRing R]

###### ``` Kimina-Autoformalizer-7B: ```Lean4

import Mathlib open Polynomial Ideal

variable {R : Type*} [CommRing R] def ZMod3 : Type* := {x : R | x ˆ 3 - 1 = 0} def ZMod3_fintype : Fintype ZMod3 := {x : R | x ˆ 3 - 1 = 0} def ZMod3_quotient (c : ZMod3) : Type* := {x : ZMod3 | x ˆ 3 + x ˆ 2 + c = 0} def ZMod3_quotient_fintype (c : ZMod3) : Fintype (ZMod3_quotient c) := {x : ZMod3 | x ˆ 3 + x ˆ 2 + c =

0} theorem my_favorite_theorem (c : ZMod3) : IsField (ZMod3_quotient c) Ø c = 2 := by sorry

``` (Syntax Error)

StepFun-Formalizer-7B (Ours): <think> # Formalizing the Field Property of Quotient Rings in Lean 4 Let me restate the natural language problem succinctly: The quotient ring Z3rxs{px3 ` x2 ` cq is a field if and only if c “ 2 (where c P Z3).

## Logical Structure Analysis This problem follows an if-and-only-if (biconditional) logical structure, which is represented by the “Ø” symbol in Lean. The biconditional statement connects two propositions:

- 1. Left side: The quotient ring Z3rxs{px3 ` x2 ` cq is a field.
- 2. Right side: c “ 2. Formalizing a biconditional statement in Lean 4 requires proving both directions of the implication:

- - Forward direction: If the quotient ring is a field, then c “ 2.
- - Backward direction: If c “ 2, then the quotient ring is a field. This logical structure is particularly important in algebraic statements where we need to establish equivalence between algebraic properties and specific parameter values. The biconditional ensures that we capture exactly when the algebraic structure has the desired property, without including unnecessary cases or excluding valid ones.

## Mathematical Concepts Analysis The problem involves several mathematical concepts:

- 1. Finite field Z3: This is the field of integers modulo 3, consisting of the elements 0, 1, 2 with arithmetic operations performed modulo 3.
- 2. Polynomial ring Z3rxs: This is the ring of polynomials with coefficients from Z3. Elements have the form anxn ` an´1xn´1 ` . . . ` a1x ` a0 where each ai P Z3.
- 3. Ideal generated by a polynomial: The ideal px3 ` x2 ` cq consists of all polynomials that are multiples of x3 ` x2 ` c.
- 4. Quotient ring: The quotient ring Z3rxs{px3 `x2 `cq consists of equivalence classes of polynomials where two polynomials are equivalent if their difference is in the ideal px3 ` x2 ` cq.
- 5. Field: A field is a ring where every non-zero element has a multiplicative inverse. For a quotient ring Z3rxs{pppxqq to be a field, the polynomial ppxq must be irreducible over Z3.

## Mathematical Objects and Types Based on the problem, we need to define or work with the following mathematical objects:

- 1. The ring Z3: In Lean, this is represented by the type `ZMod 3`, which is the ring of integers modulo 3.
- 2. Polynomials over Z3: These are represented by the type `Polynomial (ZMod 3)`in Lean.
- 3. The specific polynomial x3 ` x2 ` c: This is constructed using polynomial operations in Lean.
- 4. The ideal generated by this polynomial: This is represented using Lean’s ideal notation.
- 5. The quotient ring: This is represented using Lean’s quotient ring construction.

## Required Headers and Imports To formalize this problem in Lean 4, we need to import appropriate libraries and open relevant namespaces:

- 1. Mathlib: This is the standard mathematical library for Lean, which provides definitions and theorems about rings, fields, polynomials, etc.
- 2. Polynomial namespace: This provides operations and properties of polynomials.

- 3. Ideal namespace: This provides operations and properties of ideals in rings.
- 4. CommRing namespace: This provides properties of commutative rings, which are relevant since we’re working with fields and quotient rings. The import statement and namespace openings would look like:

###### ```

import Mathlib open Polynomial Ideal

###### ```

## Potential Lean Issues When formalizing this problem in Lean 4, we need to be aware of several potential issues:

- 1. Type coercions: Lean might need to coerce between different types, such as between elements of `ZMod 3`and polynomials over `ZMod

3`.

- 2. Variable declarations: We need to properly declare the variable `c` and ensure it’s interpreted as an element of `ZMod 3`.
- 3. Notation for polynomials: Lean has specific syntax for constructing polynomials, and we need to ensure we’re using it correctly.
- 4. Quotient ring construction: Lean has specific syntax for constructing quotient rings, and we need to ensure we’re using it correctly.
- 5. Field property: We need to use the appropriate Lean predicate to express that a ring is a field.

## Mapping Mathematical Concepts to Lean 4 Now, let’s map the mathematical concepts from the problem to their Lean 4 representations:

- 1. The ring Z3: In Lean, the ring of integers modulo 3 is represented by the type `ZMod 3`. This is a finite field with elements {0, 1, 2} and arithmetic operations performed modulo 3.
- 2. The polynomial x3 ` x2 ` c: In Lean, polynomials are constructed using the `X`variable for the indeterminate and standard arithmetic operations. The polynomial x3 ` x2 ` c would be represented as `Xˆ3 + Xˆ2 + C c`, where `C c` is the constant term lifted from the coefficient ring.
- 3. The ideal px3 ` x2 ` cq: In Lean, ideals are constructed using the ideal notation. The ideal generated by the polynomial x3 ` x2 ` c would be represented as `span ({Xˆ3 + Xˆ2 + C c}: Set (Polynomial (ZMod 3)))`, where `span`is the function that constructs the ideal generated by a set of elements.
- 4. The quotient ring Z3rxs{px3 ` x2 ` cq: In Lean, quotient rings are constructed using the quotient notation. The quotient ring Z3rxs{px3 ` x2 ` cq would be represented as `(Polynomial (ZMod 3)) / (span ({Xˆ3 + Xˆ2 + C c} : Set (Polynomial (ZMod 3))))`.
- 5. The field property: In Lean, the property of being a field is represented by the predicate `IsField`. So, the statement that the quotient ring is a field would be represented as `IsField ((Polynomial (ZMod 3)) / (span ({Xˆ3 + Xˆ2 + C c} : Set (Polynomial (ZMod 3))))`.
- 6. The condition c “ 2: In Lean, this is simply represented as `c = 2`, where `c`is a variable of type `ZMod 3`.

## Complete Lean 4 Code Based on the above analysis, the complete Lean 4 code for formalizing the problem would be: ```lean

import Mathlib open Polynomial Ideal

variable {R : Type*} [CommRing R] theorem my_favorite_theorem {c : ZMod 3} :

IsField ((Polynomial (ZMod 3)) { (span ({Xˆ3 + Xˆ2 + C c} : Set (Polynomial (ZMod 3))))) Ø c = 2 := by

sorry

###### ```

...

</think> ```Lean4

import Mathlib open Polynomial Ideal

variable {R : Type*} [CommRing R] theorem my_favorite_theorem {c : ZMod 3} :

IsField ((Polynomial (ZMod 3)) { (span ({Xˆ3 + Xˆ2 + C c} : Set (Polynomial (ZMod 3))))) Ø c = 2 := by

sorry

```

- Figure 17: A comprehensive case study in the OOD benchmark ProverBench. The output of Kimina-Autoformalizer indicates that it could not understand the concept of a quotient ring, while StepFun-Formalizer correctly interprets the problem by analyzing the natural langauge definitions of polynomial rings and ideals, and successfully maps it to the corresponding Lean 4 constructs.

