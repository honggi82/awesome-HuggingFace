# arXiv:2406.19227v1[cs.CL]27Jun2024

## Aligning Teacher with Student Preferences for Tailored Training Data Generation

Yantao Liu*†, Zhao Zhang*†, Zijun Yao*, Shulin Cao, Lei Hou, Juanzi Li Department of Computer Science and Technology, BNRist; KIRC, Institute for Artificial Intelligence, Tsinghua University, Beijing, 100084, China yaozj20@mails.tsinghua.edu.cn, {houlei,lijuanzi}@tsinghua.edu.cn

### Abstract

Large Language Models (LLMs) have shown significant promise as copilots in various tasks. Local deployment of LLMs on edge devices is necessary when handling privacy-sensitive data or latency-sensitive tasks. The computational constraints of such devices make direct deployment of powerful large-scale LLMs impractical, necessitating the Knowledge Distillation from large-scale models to lightweight models. Lots of work has been done to elicit diversity and quality training examples from LLMs, but little attention has been paid to aligning teacher instructional content based on student preferences, akin to “responsive teaching” in pedagogy. In response, we propose ARTE, dubbed Aligning TeacheR with StudenT PreferencEs, a framework that aligns the teacher model with student preferences to generate tailored training examples for Knowledge Distillation. Specifically, we first elicit draft questions and rationales from the teacher model, then collect student preferences on these questions and rationales using students’ performance with in-context learning as a proxy, and finally align the teacher model with student preferences with Direct Preference Optimization. In the end, we repeat the first step with the aligned teacher model to elicit tailored training examples for the student model on the target task. Through extensive experiments on academic benchmarks, we demonstrate the superiority of our method over existing instructiontuning datasets distilled from powerful LLMs. Moreover, we thoroughly investigate the generalization of ARTE, including the generalization of fine-tuned student models in reasoning ability and the generalization of aligned teacher models to generate tailored training data across tasks and students. In summary, our contributions lie in proposing a novel framework for tailored training example generation, demonstrating its efficacy in extensive experiments, and investigating the generalization of both fine-tuned student and aligned teacher models in ARTE. The code is released at https://github.com/THU-KEG/ARTE.

### 1 Introduction

Recently, Large Language Models (LLMs) have demonstrated significant potential as copilots for various tasks [1, 56]. Some applications involve sensitive data, such as medical diagnostics or legal advice [56, 39], or require low latency, such as robot control or autonomous driving [7, 14]. This necessitates the deployment of LLMs on computationally constrained edge devices, like smartphones or laptops. However, most existing successes [53, 39, 37] depend on powerful, large-scale LLMs [1, 9, 48], which are computationally prohibitive for edge deployment. This scenario highlights the need

*Equal contribution. †Work was done during their internship at Tsinghua University.

Preprint. Under review.

for Knowledge Distillation [18], a method to distill specific capabilities such as reasoning, planning, and decision-making from large-scale LLMs to lightweight models suitable for edge deployment.

There are two steps in Knowledge Distillation in LLMs: 1) Knowledge Elicitation, where the training data is curated from the teacher model , and 2) Supervised Fine-Tuning, where the student model learns from the generated examples. Since the research paradigm in the era of LLMs has shifted from model-centric to data-centric [60], the key challenge lies in step 1 curation of training data. Hsieh et al. [19], Mukherjee et al. [35] proposed to guide the teacher language model to generate a reasoning process with chain-of-thought [54, 59], which provides more information for the student model to imitate. Wang et al. [51] utilize the ROUGE-L similarity [27] to filter out the redundant examples to ensure the diversity. Inspired by the high quality of the textbook, Li et al. [26] proposed to synthesize “textbook-like” text from the teacher model. Mitra et al. [34] contends that different tasks should be combined with different reasoning strategies when generating rationales of examples. While these works have advanced in curating training examples, they overlook aligning the teacher model with responses from the student model to refine the training data, referred to as “responsive teaching” in pedagogy [16], which effectively lowers the learning barriers for students [17].

To this end, we propose ARTE, dubbed Aligning TeacheR with StudenT PreferencEs, a novel framework that aligns the teacher language model with the student language model’s preferences to generate tailored training examples for Knowledge Distillation. Our framework is illustrated in Figure 1. There are three main steps in our framework: 1) Knowledge Elicitation (Section 3.1): we prompt the teacher model with seed questions to generate a set of draft examples consisting of question-rational pairs. 2) Preference Collection (Section 3.2): we collect the preference of each draft example through one-shot in-context learning on the preference set which contains the top-k most discriminative questions from the validation set of the target task. 3) Preference Alignment (Section

- 3.3): we align the teacher model with the student model’s preferences through Direct Preference Optimization (DPO) [40] to improve instructional content. Finally, we repeat the first step with the aligned teacher model to curate tailored training examples for the student model and use them to Supervised Fine-Tune the student model.

Extensive experiments in academic reasoning benchmarks show that our method outperforms existing instruction-tuning datasets [38, 20, 35, 57] by a large margin. Specifically, compared to the state-ofthe-art instruction-tuning dataset, our method achieves improvements of 9.6%, 1.0%, 0.8% and 8.5% on the logic reasoning, commonsense reasoning, math reasoning, and knowledge reasoning tasks respectively in Big-Bench-Hard [44] benchmark. We also testify the generalization of ARTE in two aspects: 1) We investigate generalization in reasoning of fine-tuned student models with in-context learning in out-of-domain reasoning benchmarks, including PIQA [4], CommonsenseQA [45], ARCEasy [11], ARC-Challenge [11], and GSM8K [12]. Results show that ARTE still outperforms baselines, testifying the effectiveness of ARTE in enhancing the reasoning ability of the student model. 2) We further investigate the generalization of the aligned teacher model across tasks and students. The result provides two insights: First, the teacher model aligned in BBH can generate tailored training examples on unseen reasoning benchmarks, such as ARC or GSM8K. Second, the teacher model aligned with Gemma-2B can generate tailored training examples for different student models from different families or different domains as long as they share a similar parameter capacity, such as Qwen1.5-1.8B or CodeGemma-2B.

To summarize, our contributions are three-fold: 1) Inspired by the responsive teaching in pedagogy, we propose ARTE, a novel framework that aligns the teacher language model with the student language model’s preferences to generate tailored training examples for Knowledge Distillation. 2) Extensive experiments in in-domain and out-of-domain reasoning benchmarks show the student model fine-tuned with tailored training examples generated by ARTE outperforms existing instructiontuning datasets by a large margin. 3) We also investigate the generalization of the aligned teacher model across tasks and students. The results show that the aligned teacher model can curate tailored training examples across different reasoning tasks and different student models with similar parameter capacities.

### 2 Preliminaries

Knowledge Distillation Knowledge distillation is a widely used technique to transfer knowledge from a large teacher model to a smaller student model [18]. In this study, we focus on scenarios where

[Figure 1]

Question Generation

[Figure 2]

Expand more questions with brainstorm based on the seed questions

[Figure 3]

- Question 1: (True and False) or True is
- Question 2: xxxx

###### Rationale Generation

- Question 1: ( True and not False ) or True is

- Rationale 1 (Recall then Calc)

The rule of boolean is bracket not and or respectively. Then we first calcu ..

- Rationale 2 (Step By Step Calu)

Based on the rule of boolean, step 1: not false equals to true …

- Rationale 3 (Answer with trick)

Note that there exists ”or true” in the end, thus the answer would be True.

… …

- Question 2: True or False or True is

Rationale 1 (Recall then Calc)

The rule of boolean is bracket not and or respectively. Then we first calcu ..

Collecting the Preference Score Using One-Shot ICL Accuracy as a proxy on Pref Set

Q1: ( True and not False ) or True is Score: 0.67

- Rationale 1 The rule of boolean is … Score: 0.51
- Rationale 2 Based on the rule of … Score: 0.43
- Rationale 3 Note that there exists .. Score: 0.37 … … Question 2: True or False or True is Score: 0.52 Rationale 1 The rule of boolean is .. Score: 0.27

Question Level Alignment

Rationale Level Alignment

[Figure 4]

Expand more questions with brainstorm based on the seed questions

R2 R3

>

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

>

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Solve this question about Boolean step by step: ( True and not False ) or True is

Preference Set Construction

[Figure 19]

[Figure 20]

[Figure 21]

Find Discriminative Question with Item Response Thoery

To Form Pref Set

PREF

[Figure 22]

[Figure 23]

[Figure 24]

VAL

|Teacher LM|
|---|

|Prompt|
|---|

[Figure 25]

|Student LM|
|---|

|Uesr|
|---|

|Prompt|
|---|

|Prompt|
|---|

|Val Set|
|---|

|Pref Set|
|---|

|High-Score Ques.|
|---|

|Low-Score Ques.|
|---|

|High-Score Rati.|
|---|

|Low-Score Rati.|
|---|

[Figure 26]

Preference Alignment

[Figure 27]

[Figure 28]

Estimate Preference Score

Figure 1: The overall framework of ARTE.

both teacher and student models are pre-trained autoregressive language models. Given a teacher language model LMt, a student language model LMs, and a target task T, the goal of knowledge distillation is to improve the performance of the student model LMs on task T by fine-tuning it with the training examples generated by the teacher model LMt. There are only a few labeled examples D served as validation set available (e.g., 100 examples) for the target task T, each example consists of a question and an answer pair (q,a).

Alignment of L to align the language model with human or AI preferences. In t where the teacher model LMt is aligned with the student model L tailored training examples for Knowledge Distillation. Direct Preference selected to align the teacher model with the student model’s preferences simplicity compared to other methods like PPO [42]. It can be formulated as a simple loss function that can be optimized directly on a dataset of preference pairs (x,yw,yl), where x is an input prompt and yw and yl are the preferred and dispreferred responses, respectively. DPO trains the teacher model to align with the student model’s preferences by minimizing the loss function.

- 3 ARTE

Knowledge Elicitation Preference Collection

| | |
|---|---|
| |ARTE: AlignTeacheRwithStudenTPreferencEs|

[Figure 29]

|Question 1 Rationale 1|0.7|
|---|---|
|Rationale 2<br><br>LLMs(StepByStepCalu)Alig|0.8nment of LLMs aims|
|Rationale 3 (Answer with trick)<br><br>this study, we|[Figure 30]<br><br>Notice that there exists ”or true” in the end, thus the answer would be True<br><br>focus on the scenario|
|Question1 Score<br><br>LMs’s prefer Optimiza|…<br><br>ences to generate tion (DPO) [40] is|
|Rationale 1<br><br>due to i|xxxx<br><br>ts stability and|

We introduce ARTE, dubbed Aligning TeacheR with StudenT PreferencEs, a novel framework that aligns the teacher LMt with student LMs’s preferences to generate tailored training examples in the target task T. Our overall framework is illustrated in Figure 1. There are three main steps in our framework: 1) Knowledge Elicitation: we prompt the teacher model with seed questions to generate a draft dataset consisting of draft questions and corresponding draft rationales. 2) Preference Collection: We collect the preference scores from the student model for both draft questions and rationales using the one-shot in-context learning performance of the student model on the preference set as a proxy.

- 3) Preference Alignment: Based on the preference score of questions and rationales, we align the teacher model with the student model’s preferences through Direct Preference Optimization (DPO). Finally, we repeat the first step with the aligned teacher model to curate tailored training examples for the student model and use them to Supervised Fine-Tune the student model. In the following, we will elaborate on each step in detail.

##### 3.1 Knowledge Elicitation

In this step, we construct a set of draft dataset Ddraft = {(q,Rq)} for the target task T, where q is the draft question and Rq = {rq} is the set of rationales associated with the question qi. For question generation, we prompt the teacher model LMt with seed questions to expand them into m draft questions. Specifically, we use three seed questions to construct one question generation prompt which guides the teacher model to brainstorm multiple questions. We decode the question generation prompt at a temperature 1.0 multiple times until we get m draft questions.

For rationale generation, inspired by the observation that one question may have multiple different reasoning processes leading to the same answer and the optimal reasoning process may vary for different questions and different models [34]. Then, for each draft question q, LMt is guided to generate n rationales rq to form a rationale set Rq using different reasoning techniques, such as Explain Like I’m Five, Step-by-Step, or Math Symbols. Finally, we combine each draft question q with its corresponding rationales Rq to form a draft sample (q,Rq). Since each draft question q would be expanded into n rationales, there are m × n draft examples Edraft = {(q,rq)} in the draft dataset Ddraft.

##### 3.2 Preference Collection

In this step, our target is to collect the preference of the student model in both question and rationale levels. Namely, we aim to determine which question or rationale is more likely to be accepted by the student model during the fine-tuning process. Dai et al. [13] points out that language models secretly perform gradient descent w.r.t the in-context examples during in-context learning like supervised fine-tuning. Inspired by this, we contend that the student model’s preference for one draft example (q,r) in supervised fine-tuning can be reflected by the performance of the student model with (q,r) as the example in one-shot in-context learning.

Therefore, we use the in-context performance of the student model with draft examples on the validation set Dval as the preference score. Specifically, for rationale r whose corresponding question is q, the preference score sr is defined as follows:

1 |Dval|

I(LMs(q,r;q′) = a′) (1)

sr =

(q′,a′)∈Dval

where LMs(q,r;q′) denotes the student model’s answer to the question q′ when q and r are used as the example in one-shot in-context learning, and I(·) is the indicator function. Then the preference

score sq of question q is defined as the average of the preference scores of its associated rationales:

1 |Rq| r∈R

sr (2)

sq =

q

where Rq is the set of rationales associated with the question q.

However, the computation cost would be high if we evaluate the student model on the entire validation set for all m × n draft examples. Existing alignment methods such as DPO [40] and PPO [42] only require understanding the partial order of outputs relative to a single input to learn preferences. Inspired by this, we aim to sample a small, discriminative subset of questions from the validation set to estimate their preference scores effectively. Specifically, Item Response Theory (IRT) [32, 15] is adopted to measure the discriminability of each question in the validation set. First, we collect a scoring matrix S ∈ Ra×b from a different LLMs on b samples from the validation set. Then, we use the two-parameter logistic model [46] to estimate the discrimination parameter α and the difficulty parameter β for each question based on the score matrix S through bayesian inference [36, 25, 41]. Finally, we select the top-k most discriminative questions as the preference set Dpref. Thus the estimated preference score sˆr and sˆq can be formulated as:

1 |Dpref|

I(LMs(q,r;q′) = a′) (3)

sˆr =

(q′,a′)∈Dpref

and

1 |Rq| r∈R

sˆr (4) Without special note, we refer to sˆr and sˆq as reference scores in the following sections.

sˆq =

q

##### 3.3 Preference Alignment

After collecting the preference score of questions and rationales, we aim to align the teacher model with the student model’s preferences to generate tailored examples for the target task T. There are two main aspects to align, Question Generation and Rationale Generation, towards generating not only tailored rationales but also tailored questions. Direct Preference Optimization (DPO) [40] is selected to align the teacher model with the student model’s preferences due to its stability and simplicity compared to other methods like PPO [42]. For question level alignment, for the input question generation prompt, we randomly sample out k draft questions with the top-25% and bottom-25% preference scores as the chosen and rejected responses respectively. For rationale level alignment, for each question q, we select the rationale from Rq with the highest preference score as the chosen response and the rationale with the lowest preference score as the rejected response. Note that the input prompt in rationale level alignment is just a naive step-by-step prompt as shown in Table 10. Finally, we mix the question level and rationale level DPO datasets to perform DPO to align the teacher model with the student model’s preferences.

After the alignment, we repeat the first step to generate tailored examples for the target task T and use them to Supervised Fine-Tune the student model. Note that when generating tailored questions and rationales, we use the same input prompt from the alignment step and decode at temperature τ = 0 for deterministic decoding. More details can be found in Appendix F.

### 4 Experiments

In this section, we conduct experiments to evaluate the effectiveness of our method to transfer the reasoning capabilities from the teacher model to the student model to target tasks Big-BenchHard [44].

##### 4.1 Experimental Setup

Datasets We conduct experiments on the Big-Bench-Hard [44], a suite of 23 challenging tasks from the Big-Bench benchmark [43]. The tasks are designed to test the generalization ability of models on a wide range of tasks, including logical reasoning, commonsense reasoning, world knowledge, math ability, etc. For better presentation, we divide the tasks into four categories based on the capabilities required by the tasks: (1) logical reasoning, (2) commonsense reasoning, (3) math reasoning and (4) world knowledge, denoted as BBHLogic, BBH-CS, BBH-Math and BBH-Knowl respectively. Details of the task categories are shown in Appendix D. Table 1 shows the statistical information of the tasks in Big-Bench-Hard.

As mentioned in Section 3.2, we collect the top-40 most discriminative questions in each task from the validation set of the Big-BenchHard to form the preference set to reduce the computation cost. Specifically, first, sampling 100 questions for each task from original BigBench dataset to form the validation set.* Sec-

Category #Tasks #Pref #Test BBH-Logic 10 400 3,146 BBH-CS 7 280 1,615 BBH-Math 3 120 750 BBH-Knowl 3 120 750 Total 23 920 6,261 Table 1: Statistics of the Big-Bench-Hard dataset.

- ond, we collect the detailed results of Gemma (2B,7B), CodeGemma-2B, Qwen1.5-1.8B, and MiniCPM-2B on the validation set under 0,1,2,3shot settings. Finally, we use the 2P-IRT model to find out the top-40 most discriminative questions to form the preference set for each task and testify the consistency of preference scores collected from the preference set and the original validation set. The details are in Appendix E.

Baselines We compare our method with the following instruction-tuning datasets as academic baselines: 1) GPT-4-LLM [38] is an instruction-tuning dataset distilled from GPT-4 with Self-Instruct [51]. 2) Tulu-v2 [20] is a mix of multiple existing high-quality instruction-tuning datasets, including FLAN [31], OpenAssistant [24] etc. 3) OpenOrca is a reproduction of Orca [35], which augments

*For Causal Judgment, Penguins in a Table, and Snarks, there is no extra samples apart from the test set in BBH, thus we use GPT-4 to generate the validation set and manually filter out the questions with low quality.

Models BBH-Logic BBH-CS BBH-Math BBH-Knowl BBH zero-shot result Vanilla Gemma-2B 0.80 0.00 6.00 0.00 1.09

+ GPT-4-LLM 1.35 0.00 1.60 0.93 1.00 + Tulu-v2 6.15 5.83 16.00 0.67 6.33 + WizardLM 6.24 2.14 12.27 15.33 6.88 + OpenOrca 5.17 4.68 20.13 8.53 7.42

+ Orignal Teacher 35.57 40.61 33.87 40.27 37.55

- + Rationale Only 36.07 37.54 37.87 40.53 37.94

+ Question Only 40.29 43.45 40.80 40.67 41.73 + ARTE 39.07 44.44 42.00 42.53 41.96 three-shot result Vanilla Gemma-2B 28.16 40.92 42.13 44.40 35.57

+ GPT-4-LLM 29.58 40.79 36.27 49.20 36.11 + Tulu-v2 30.28 39.74 38.67 50.00 36.35 + WizardLM 29.62 43.44 45.33 45.33 37.29 + OpenOrca 28.91 41.91 43.47 49.87 36.92

+ Orignal Teacher 34.32 43.06 44.13 52.13 40.04

- + Rationale Only 37.43 42.17 47.47 49.73 41.30

+ Question Only 40.39 43.85 41.73 51.87 42.78 + ARTE 39.86 44.45 46.13 53.87 43.44

- Table 2: Accuracy (%) of the student model Gemma-2B fine-tuned with different instruct-tuning datasets on Big-Bench-Hard under zero-shot and three-shot settings. Cells are colored blue if the method improves over Vanilla Gemma-2B, and orange if it declines.

FLAN data with additional GPT-4 or GPT-3.5-Turbo generated explanations. 4) WizardLM-EvolInstruct [57] is instruction-fine-tuning dataset which is distilled from the GPT-4 through Evol-Instruct.

Apart from the above academic baselines, we also compare our method with ablation baselines to testify the effectiveness of the two different alignment aspects in our method: 1) Orignal Teacher: the dataset constructed by the draft questions and its corresponding rationale with the highest preference score in the preference set. Namely, in this baseline, both the questions and rationales are generated by the original teacher model. 2) Rationale Only: In this setting, only the rationales are generated by the aligned teacher model, while the questions are generated by the original teacher model. 3) Question Only: In this setting, only the questions are generated by the aligned teacher model, while the rationales are generated by the original teacher model. To ensure a fair comparison, we sample out 6,750 examples from each baseline dataset to conduct the experiment. Supervised Fine-Tune is used to train the student model on these datasets with 1 epoch using the Adam optimizer [23] and a learning rate of 2e − 5. We use 3% of the total training steps for cosine learning rate warm-up and adopt a linear decay learning rate schedule.

Implementation In our experiments, we adopt Llama-3-70B-Instruction [33] as the teacher model and Gemma-2B [2] as the student model.

In the Knowledge Elicitation step, 1) Question Generation: We decode the question generation prompt in Table 11 with the aligned teacher at a temperature of 1.0 until we achieve a total of 250 draft questions per task. 2) Rationale Generation: For each draft question, the teacher model is prompted to generate n = 4 rationales using the prompt in Table 12 with diverse reasoning techniques. Each draft question with its corresponding rationales forms a draft sample (q,Rq).

In the Preference Collection step, we collect the preference scores of questions and rationales through in-context learning on the preference set. Specifically, we use the prompt template in Table 13 and decoding at a temperature τ = 0 to ensure the model’s answer is deterministic.

In the Preference Alignment step, we utilize preference scores to create the DPO dataset and align the teacher model with the student model’s preferences. For question generation, we select 50

Models PIQA CSQA ARC-E ARC-C GSM8K Average Vanilla Gemma-2B 61.6 38.2 57.2 42.8 20.0 43.96

+ GPT-4-LLM 60.2 42.8 55.8 46.2 18.6 44.72 + Tulu-v2 65.2 42.8 62.2 49.6 18.8 47.72 + WizardLM 63.6 41.4 56.4 49.2 26.2 47.36 + OpenOrca 64.6 43.6 61.0 48.4 23.6 48.24

#### + ARTE 63.4 42.9 63.6 57.2 32.0 51.82

- Table 3: Accuracy (%) of the student model Gemma-2B fine-tuned with ARTE and academic baselines on out-of-domain reasoning benchmarks. Cells are colored blue if the method improves over Vanilla Gemma-2B, and orange if it declines.

draft questions with the highest and lowest preference scores per task as preferred and dispreferred respectively, creating 50 DPO training examples at the question level. For rationale generation, the highest and lowest score rationales per question are selected similarly, resulting in 250 DPO training examples at the rationale level per task. These datasets are combined to form the final DPO dataset. We employ Direct Preference Optimization (DPO) to align the models at a learning rate of 1e − 7, batch size of 16, for 1 epoch, incorporating a linear warm-up over 10% of training steps followed by a cosine decay learning rate schedule.

Finally, we repeat the first step to generate tailored examples for BBH tasks and use them to Supervised Fine-Tune the student model with the same hyperparameters as baselines. All the experiments are conducted on 8 * NVIDIA A100 80G GPUs, which will cost about 1 day in total.

##### 4.2 Results

- Table 2 shows the results of our method and baselines on the Big-Bench-Hard dataset under the zero-shot and three-shot in-context learning settings. 1) Compared to academic baselines: we can see that our method outperforms all the baselines including the baselines that simply imitate the reasoning process of the most powerful LLM (GPT-4-LLM), or mix multiple existing high-quality instruction-tuning datasets (Tulu-v2), and other two baselines that distill the teacher model with handcrafted curations (OpenOrca and WizardLM). 2) Compared to the ablation baselines: our method outperforms both question-only and rationale-only baselines by a large margin, indicating that both the questions and rationales generated by the aligned teacher model are important for the student model to achieve better performance. Besides, notice that the performance of the question-only baseline achieves better performance than the rationale-only baseline. This interesting phenomenon indicates that in Knowledge Distillation for the language model, the quality of questions plays a more important role than the quality of rationales.

- 5 Analysis on Generalization

Although ARTE outperforms the baselines on the Big-Bench-Hard dataset, it is still unclear the generalization performance of ARTE. Thus in this section, we present a thorough examination of its generalizability in the following two aspects: 1) Generalization of the fine-tuned student model. 2) Generalization of the aligned teacher model.

##### 5.1 Generalization of the Fine-tuned Student Model

To investigate the generalization ability of the fine-tuned student model Gemma-2B, we conduct experiments on the following academic reasoning benchmarks: 1) PIQA [4] is a physical commonsense reasoning dataset that is designed to test the model’s ability to build, craft, or manipulate objects using everyday physical knowledge. 2) CSQA [45] is a question-answering benchmark targeting commonsense reasoning. 3) ARC-Easy and 4) ARC-Challenge [11] are multiple-choice questionanswering benchmarks designed to test the model’s ability to reason about scientific knowledge. 5) GSM8K [12] is a mathematical benchmark that is designed to test language models’ ability in math and logic reasoning.

- Table 3 shows the results of our method and baselines on the five benchmarks. On average, our method outperforms the best baseline by 1.5% in accuracy. The results show that our method helps the student model Gemma-2B to achieve better generalization ability on reasoning tasks. In some tasks like PIQA and CSQA, the performance of our method is not the best but still competitive. While in more challenging tasks like ARC-Challenge and GSM8K, which are both sampled from real-world grade school exams, our method outperforms all the baselines. This phenomenon reveals for more challenging tasks, more carefully curated examples are needed in knowledge distillation. Compared to the heuristic baselines, the tailored examples generated by the aligned teacher model are more effective in these tasks as they can better capture the student model’s preferences.

##### 5.2 Generalization of the Aligned Teacher Model

Due to the high computational cost of preference collection and alignment, it is desirable that the aligned teacher model can generate tailored examples for unseen tasks and unseen student models.

Generalization Across Tasks To investigate whether our aligned teacher model is capable of generating tailored examples for student models on unseen tasks, we conduct experiments on the PIQA, ARC-EASY, ARC-Challenge, and GSM8K benchmarks.

We repeat Step 1 Knowledge Elicitation to generate training examples on the four benchmarks. Specifically, we guide both original and aligned teacher models to generate 2500 training examples for each unseen benchmark. We train the student model Gemma-2B with the generated examples on the four unseen benchmarks separately with the same hyperparameters mentioned in Section 4.

Models PIQA ARC-E ARC-C GSM8K Average zero-shot results

Original 62.4 62.4 51.0 22.2 49.5

- Aligned 65.6 59.4 51.2 25.6 50.45 Delta ↑ 3.2 ↓ 3.0 ↑ 0.2 ↑ 3.4 ↑ 0.95

three-shot results Original 65.0 64.2 51.8 23.4 51.1

- Aligned 66.6 65.2 52.6 24.8 52.3 Delta ↑ 1.6 ↑ 1.0 ↑ 0.8 ↑ 1.4 ↑ 1.2

Table 4: Accuracy (%) of the student model Gemma2B fine-tuned with training examples generated by the original teacher model and the aligned teacher model on out-of-domain reasoning benchmarks.

Results in Table 4 show that the student model fine-tuned with training examples generated by the aligned teacher model outperforms the student model fine-tuned with training examples generated by the original teacher model. This finding suggests that through preference alignment, the aligned teacher model gains a deeper understanding of the student model’s preferences, which helps with unseen tasks. This enhanced understanding of the teacher model makes it easier to distill the specific abilities of the aligned teacher model to the student model compared to the original teacher model.

Generalization Across Student Models To investigate the generalization ability of the aligned teacher model across unseen student models, we conduct experiments on the Big-Bench-Hard dataset. Specifically, we adopt Gemma-7B as the student model from the same family with a different capacity, Qwen1.5-1.8B and CodeGemma2B as the student models with the same capacity but from different families or different domains. We train the student models with the training examples generated by the original teacher model and the aligned teacher model under the same hyperparameters mentioned in Section 4.

Models Gemma-7B Qwen1.5-1.8B CodeGemma-2B zero-shot results

Original 49.0 36.3 38.7 Aligned 49.0 39.4 41.2 Delta 0.0 ↑ 3.1 ↑ 2.5

three-shot results

Original 51.2 38.2 39.8 Aligned 51.0 40.3 42.7 Delta ↓ 0.2 ↑ 2.1 ↑ 2.9

Table 5: Accuracy (%) of the different student models fine-tuned with training examples generated by the original teacher model and the aligned teacher model on Big-Bench-Hard.

Table 5 shows the results of the student models trained with the original teacher model and the aligned teacher model, denoted as Original and Aligned respectively. The results show that in Qwen1.5-1.8B and CodeGemma-2B, the models from different families or different domains share similar parameter capacity with the Gemma-2B, and

achieve better performance with the tailored examples generated by the aligned teacher model than the original teacher model. In Gemma-7B, the model from the same family with a larger parameter capacity does not present a promising improvement. This indicates that the language model with a similar parameter capacity shares similar preferences in training examples. This finding suggests that the aligned teacher model can be applied to different student models as long as they share similar parameter capacities.

### 6 Related Work

Knowledge Distillation for LLMs Knowledge distillation is a widely used technique to transfer knowledge from a large teacher model to a smaller student model [18]. Recently, the most powerful LLMs are computationally expensive highlighting the need for knowledge distillation to transfer advanced capabilities from large LLMs to lightweight models [57, 34, 47]. There are two main steps in knowledge distillation [58]: (1) Knowledge Elicitation, which generates training examples for the student model, and (2) Distillation Algorithms, which centers on injecting the elicited knowledge, namely training examples, into the student model. Although there are some other distillation algorithms [49], Supervised Fine-Tuning [8] has been the de facto standard for knowledge distillation due to its simplicity and effectiveness. Most work focus on knowledge elicitation, specifically on how to curate training data from LLMs [58] Hsieh et al. [19], Mukherjee et al. [35] proposed to guide the teacher language model to generate not only the final answer but also the reasoning process with chain-of-thought [54, 59], which provides more information for the student model to imitate. Since it is easy to synthesize large-scale text from the teacher model, the key challenge lies in the quality and diversity control of the generated examples. Wang et al. [51] utilize the ROUGE-L similarity [27] to filter out the redundant examples to ensure the diversity. Inspired by the high quality of the textbook, Li et al. [26] proposed to synthesize “textbook-like” text from the teacher model. Mitra et al. [34] contends that different tasks should be combined with different reasoning strategies when generating rationales of examples. All these works have advanced in curating training examples, but they overlook aligning the teacher model with responses from the student model to refine the training data, referred to as “responsive teaching” in pedagogy [16], which effectively lowers the learning barriers for students [17].

Alignment of LLMs Large language models (LLMs) trained on large-scale corpora have shown remarkable performance on various natural language processing tasks [1, 48, 9, 21]. Despite the success, these models may not be aligned with the needs of human users generating text with biases, factual inaccuracies, or inappropriate content [52]. Many works have been proposed to align LLMs with human preferences. Supervised Fine-Tuning [8] is the basic approach to align LLMs with human preferences, which aligns the model by minimizing the cross-entropy loss between the model’s prediction and the human-labeled data. Reinforcement learning from human preference (RLHF) [10] prosed to optimize for maximum reward operates by engaging with a reward model trained using the Bradley-Terry model [6, 5] through reinforcement algorithm Proximal Policy Optimization (PPO) [42]. While RLHF exhibits remarkable performance compared to supervised fine-tuning, it faces challenges such as reward hacking [29]. Direct Preference Optimization (DPO) [40] has emerged as a promising alternative to RLHF, which does not require a reward model as a proxy. It recasts the alignment formulation as a simple loss function that can be optimized directly on a dataset of preference pairs (x,yw,yl), where x is prompt and yw and yl are the preferred and dispreferred responses, respectively. Since DPO is RL-free and does not require a reward model, its training process is more stable than RLHF. [52] In this work, we leverage the DPO to align the teacher model with the student model.

### 7 Conclusion

In this study, we propose ARTE, a novel framework in Knowledge Distillation that aligns the teacher language model with the student language model’s preferences to generate tailored training examples for Knowledge Distillation, which is inspired by responsive teaching in pedagogy, which effectively lowers the learning barriers for students. Our framework consists of three main steps: Knowledge Elicitation, Preference Collection, and Preference Alignment. Extensive experiments on various academic benchmarks show that our method outperforms existing instruction-tuning datasets distilled from powerful LLMs by a large margin. Moreover, we investigate the generalization of the

aligned teacher model, showing that the aligned teacher model can be generalized to other reasoning benchmarks and different student models.

### References

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] T. W. J. Banks and T. Warkentin. Gemma: Introducing new state-of-the-art open models, 2024.
- [3] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 610–623, 2021.
- [4] Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.
- [5] H. Bong and A. Rinaldo. Generalized results for the existence and consistency of the mle in the bradley-terry-luce model, 2022.
- [6] R. A. Bradley and M. E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.
- [7] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.
- [8] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [9] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.
- [10] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.
- [11] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.
- [12] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [13] D. Dai, Y. Sun, L. Dong, Y. Hao, S. Ma, Z. Sui, and F. Wei. Why can GPT learn in-context? language models secretly perform gradient descent as meta-optimizers. In A. Rogers, J. BoydGraber, and N. Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 4005–4019, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.247. URL https://aclanthology.org/ 2023.findings-acl.247.
- [14] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.
- [15] S. E. Embretson and S. P. Reise. Item response theory. Psychology Press, 2013.
- [16] G. Gay. Culturally responsive teaching: Theory, research, and practice. multicultural education series. 2000.

- [17] J. Hattie and H. Timperley. The power of feedback. Review of educational research, 77(1): 81–112, 2007.
- [18] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
- [19] C.-Y. Hsieh, C.-L. Li, C.-k. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 8003–8017, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.507. URL https://aclanthology.org/2023.findings-acl.507.
- [20] H. Ivison, Y. Wang, V. Pyatkin, N. Lambert, M. Peters, P. Dasigi, J. Jang, D. Wadden, N. A. Smith, I. Beltagy, et al. Camels in a changing climate: Enhancing lm adaptation with tulu 2. arXiv preprint arXiv:2311.10702, 2023.
- [21] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b, 2023.
- [22] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. HatfieldDodds, N. DasSarma, E. Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.
- [23] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [24] A. Köpf, Y. Kilcher, D. von Rütte, S. Anagnostidis, Z. R. Tam, K. Stevens, A. Barhoum, D. Nguyen, O. Stanley, R. Nagyfi, et al. Openassistant conversations-democratizing large language model alignment. Advances in Neural Information Processing Systems, 36, 2024.
- [25] J. P. Lalor, H. Wu, and H. Yu. Learning latent parameters without human response patterns: Item response theory with artificial crowds. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.
- [26] Y. Li, S. Bubeck, R. Eldan, A. Del Giorno, S. Gunasekar, and Y. T. Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.
- [27] C.-Y. Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013.
- [28] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024.
- [29] T. Liu, Y. Zhao, R. Joshi, M. Khalman, M. Saleh, P. J. Liu, and J. Liu. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023.
- [30] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
- [31] S. Longpre, L. Hou, T. Vu, A. Webson, H. W. Chung, Y. Tay, D. Zhou, Q. V. Le, B. Zoph, J. Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.
- [32] F. Maia Polo, L. Weber, L. Choshen, Y. Sun, G. Xu, and M. Yurochkin. tinybenchmarks: evaluating llms with fewer examples. arXiv preprint arXiv:2402.14992, 2024.
- [33] Meta. The official meta llama 3 github site. https://github.com/meta-llama/llama3, 2024.

- [34] A. Mitra, L. Del Corro, S. Mahajan, A. Codas, C. Simoes, S. Agarwal, X. Chen, A. Razdaibiedina, E. Jones, K. Aggarwal, et al. Orca 2: Teaching small language models how to reason. arXiv preprint arXiv:2311.11045, 2023.
- [35] S. Mukherjee, A. Mitra, G. Jawahar, S. Agarwal, H. Palangi, and A. Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707, 2023.
- [36] P. Natesan, R. Nandakumar, T. Minka, and J. D. Rubright. Bayesian prior choice in irt estimation using mcmc and variational bayes. Frontiers in psychology, 7:1422, 2016.
- [37] J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22, 2023.
- [38] B. Peng, C. Li, P. He, M. Galley, and J. Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.
- [39] C. Qian, X. Cong, C. Yang, W. Chen, Y. Su, J. Xu, Z. Liu, and M. Sun. Communicative agents for software development. arXiv preprint arXiv:2307.07924, 2023.
- [40] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model, 2023.
- [41] P. Rodriguez, J. Barrow, A. M. Hoyle, J. P. Lalor, R. Jia, and J. Boyd-Graber. Evaluation examples are not equally informative: How should that change nlp leaderboards? In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4486–4503, 2021.
- [42] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017.
- [43] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.
- [44] M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. Le, E. Chi, D. Zhou, and J. Wei. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.824. URL https://aclanthology.org/2023.findings-acl.824.
- [45] A. Talmor, J. Herzig, N. Lourie, and J. Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. URL https://aclanthology.org/N19-1421.
- [46] D. Thissen and M. Orlando. Item response theory for items scored in two categories. In Test scoring, pages 85–152. Routledge, 2001.
- [47] I. Timiryasov and J.-L. Tastet. Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. In A. Warstadt, A. Mueller, L. Choshen, E. Wilcox, C. Zhuang, J. Ciro, R. Mosquera, B. Paranjabe, A. Williams, T. Linzen, and R. Cotterell, editors, Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, pages 279–289, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.conll-babylm.24. URL https:// aclanthology.org/2023.conll-babylm.24.

- [48] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [49] F. Wan, X. Huang, D. Cai, X. Quan, W. Bi, and S. Shi. Knowledge fusion of large language models. arXiv preprint arXiv:2401.10491, 2024.
- [50] X. Wang and D. Zhou. Chain-of-thought reasoning without prompting. arXiv preprint arXiv:2402.10200, 2024.
- [51] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022.
- [52] Y. Wang, W. Zhong, L. Li, F. Mi, X. Zeng, W. Huang, L. Shang, X. Jiang, and Q. Liu. Aligning large language models with human: A survey. arXiv preprint arXiv:2307.12966, 2023.
- [53] Z. Wang, S. Cai, A. Liu, X. Ma, and Y. Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023.
- [54] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.
- [55] S. Welleck, I. Kulikov, S. Roller, E. Dinan, K. Cho, and J. Weston. Neural text generation with unlikelihood training. In International Conference on Learning Representations, 2019.
- [56] T. Xie, F. Zhou, Z. Cheng, P. Shi, L. Weng, Y. Liu, T. J. Hua, J. Zhao, Q. Liu, C. Liu, et al. Openagents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023.
- [57] C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, and D. Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023.
- [58] X. Xu, M. Li, C. Tao, T. Shen, R. Cheng, J. Li, C. Xu, D. Tao, and T. Zhou. A survey on knowledge distillation of large language models. arXiv preprint arXiv:2402.13116, 2024.
- [59] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.
- [60] D. Zha, Z. P. Bhat, K.-H. Lai, F. Yang, Z. Jiang, S. Zhong, and X. Hu. Data-centric artificial intelligence: A survey. arXiv preprint arXiv:2303.10158, 2023.
- [61] Y. Zhang, Y. Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao, Y. Zhang, Y. Chen, et al. Siren’s song in the ai ocean: a survey on hallucination in large language models. arXiv preprint arXiv:2309.01219, 2023.

Appendix

- A Limitations

Although most of the data used in ARTE are automatically generated by the teacher model, there still needs some manual efforts to construct the prompts and collect the preference scores. Specifically, there are two main limitations in the current implementation of ARTE: First, in draft rationale generation, to elicit diverse and high-quality rationales, a set of carefully designed prompts is required. In this work, we hand-crafted the system prompts for each task with different reasoning techniques by the paper authors themselves. Totally, we constructed 4 prompts for each task as presented in Table 12. Recently, Wang and Zhou [50] proposed to Chain-of-Thought Decoding (CoT-Decoding) to uncover reasoning processes of questions from language models without prompts. In the future, we will explore the possibility of using CoT-Decoding to generate a reasoning process automatically. Second, in preference collection, a set of labeled examples consisting of questions and answers is required to act as the validation set and preference set. Preference scores are collected on these labeled question-answer pairs to measure the preference of the student model towards the draft questions and rationales. In this work, we simply reuse the data from the original Big-Bench dataset as the validation set. In the future, we will explore the possibility of directly measuring the preference through the internal states of the student model [22].

- B Broader Impacts

The possible broader impacts of this work lie in the hallucination of language models. Language models have been shown to generate biased and harmful content [3, 61]. In this work, we focus on the enhancement of the student model’s specific capability, such as reasoning, which is not directly relevant to the generation of harmful content. It is worth noting that the user should be cautious when using the student model or the teacher model to generate text, especially when the generated text is used in critical applications such as medical diagnosis or legal advice.

- C Insights for Generating Rationale for Small Language Models

In Step 2 Preference Collection, we collect the preference of small language models, such as Gemma2B, towards the draft examples through one-shot in-context learning on the preference set. Through careful analysis of the collected preference scores, we conclude with two insights for generating a tailored example for language models with limited capacity:

- Insight 1: The more detailed the rationale does not necessarily mean the better the performance of the small language model. In previous works [19, 35], the teacher model is guided to generate a detailed reasoning process with chain-of-thought [54, 59] to provide more information for the small language model to imitate. However, after taking a closer look at the preference scores, we find that there is no significant linear correlation between the length of the rationale and the preference score, namely the accuracy of the small language model within one-shot in-context learning. Figure 2 shows the relationship between the word number of the rationale and the one-shot in-context learning accuracy of the small language model on boolean expressions and sports understanding tasks.

As shown in Figure 2, the rationale with most words does not necessarily mean the best performance of the small language model, while the rationale with moderate words (75-100 words in Boolean Expressions and 50-75 words in Sports Understanding) achieves better results. This indicates that a rationale with complete and concise reasoning steps is more beneficial for the small language model to learn, compared to a long meticulous rationale.

The reasons behind this are twofold: First, the Lost-in-the-Middle Phenomenon [28] of language models. Language models may lose information when the input context is too long. For limitedcapacity language models, when example rationales are too long, the small language model may get lost in the narrative and forget the question to solve. Second, the Repetition Problem [55] of language models. In long meticulous rationales, the teacher model may repeat the same step multiple times, for example acting as a stack machine when solving a math problem. For limited-capacity language

boolean_expressions

sports_understanding

0.8

0.8

0.7

0.7

Accuracy

Accuracy

0.6

0.6

0.5

0.5

0.4

50 75 100 125 150 175 200 225

60 80 100 120 140 160 180

Word Number in Rationale

Word Number in Rationale

Figure 2: The relationship between the word number of the rationale and the one-shot in-context learning accuracy of the Gemma-2B on boolean expressions and sports understanding tasks.

models, this repeated step may cause the small language model to get stuck in a loop and repeat the same step infinitely.

- Insight 2: Although different questions in the same task prefer different reasoning strategies, the small language model prefers consistent reasoning strategies for

Method BBH Vanilla Gemma-2B 35.57

- one task in Supervised Fine-tuning. The diversity of tests in the training corpus is crucial for the pretraining stage of language models [30]. To investigate the impact of the diversity of reasoning strategies in the training dataset, we fine-tuned the Gemma-2B with four different training datasets on Big-Bench-Hard. 1) Randomly Selected: the rationale of each question is randomly selected from different reasoning strategies. 2) Most Preferred: the rationale
- of each question is selected based on the highest preference score. 3) Task Consistent: all rationales are selected from within the same reasoning strategy for one task. 4) Aligned Teacher: the rationales are generated by the aligned teacher model with the small language model’s preference. All the questions in these four training datasets are generated by the original teacher model to ensure fairness. Table 6 presents the performance of the small language model fine-tuned with four different training datasets on Big-Bench-Hard. The results present an interesting phenomenon: the performance of Gemma-2B fine-tuned with the Most Preferred dataset is similar to the Randomly Selected dataset, while Task Consistent and Aligned Teacher datasets both outperform the other two datasets by a decent margin. This indicates that in the fine-tuning stage, especially when we try to enhance one or two specific capabilities of the small language model, for example, reasoning, the consistent reasoning strategies are more beneficial. The reason behind this is that the small language model may get confused when the reasoning strategies are too diverse for one task, while the consistent reasoning strategies provide clear guidance for the small language model to imitate for specific capability enhancement.

+ Randomly Selected 39.94 + Most Preferred 40.04 + Task Consistent 42.12 + Aligned Teacher 42.78

Table 6: Accuracy (%) of the Gemma2B fine-tuned with different training examples on Big-Bench-Hard.

### D Big-Bench-Hard Category

We split the Big-Bench-Hard tasks into four categories based on the capabilities required by the tasks: (1) logical reasoning, (2) commonsense reasoning, (3) math reasoning and (4) world knowledge, denoted as BBH-Logical, BBH-Commonsense, BBH-Math and BBH-Knowledge respectively. Table 7 shows the detailed information of the tasks in Big-Bench-Hard.

##### Category Task Quantity

dyck_languages, formal_fallacies_syllogisms_negation, logical_deduction, navigate, object_counting, penguins_in_a_table, temporal_sequences, tracking_shuffled_objects, web_of_lies, word_sorting

BBH-Logic

10

causal_judgment, date_understanding, disambiguation_qa, hyperbaton, reasoning_about_colored_objects, ruin_names, snarks

BBH-CS

7 BBH-Math boolean_expressions, geometric_shapes, multistep_arithmetic 3 BBH-Knowl

movie_recommendation, salient_translation_error_detection, sports_understanding

3

Table 7: Big-Bench-Hard task categories.

### E 2P-IRT Model and its Consistency

To find out the top 40 most discriminative questions to form the preference set, we first use the 2P-IRT model to estimate parameters. The 2P-IRT model involves two main parameters: difficulty parameter and discrimination parameter. The difficulty parameter measures the difficulty level of a question, representing the probability of an LLM reaching or exceeding that level in order to answer the question correctly. The discrimination parameter measures the question’s ability to differentiate between different levels of ability, higher discrimination parameters indicate greater item discrimination. Then we use the MLE method to iteratively optimize the parameter estimates. The likelihood function is maximized by adjusting the parameter values step by step. Finally, we choose the top 40 questions with the highest discrimination parameters.

To evaluate the consistency between the preference score collected from the preference set and the original validation set, we conducted an experiment where we selected one task from each category and generated 10 draft samples (q,Rq) for each task. For each draft sample, we computed the preference scores of each rationale in Rq against both sets. The top-rated rationale aligned in 93.3% of cases across both sets, and the lowest-rated had a consistency of 86.7%. These results affirm the reliability of using the preference set to collect scores.

### F Repeating Knowledge Elicitation to Generate Training Examples

After the Preference Alignment step, we need to repeat the Knowledge Elicitation step to generate training examples for the student model with the aligned teacher model. Specifically, first, we prompt the aligned teacher model with question generation prompts as shown in Table 11 to generate draft examples. We will decode the question generation prompts with decoding temperature 1.0 multiple times until we get enough draft questions. Then, for each question, we prompt the aligned teacher model with naive Step-by-Step prompts as shown in Table 8 to generate rationales with decoding temperature 0.0.

Task Description: {Task Description} Question: {Question} Answer: Let’s think step by step. {to be completed by the teacher model}

Table 8: Naive Step-by-Step prompt for the teacher model.

### G Direct Preference Optimization Details

In the Preference Alignment step, we align the teacher model with the student model’s preferences through Direct Preference Optimization (DPO) [40]. Here we provide the details of constructing the DPO dataset

DPO dataset for Question Generation For each task, we construct the DPO dataset for question generation using the following template, The question generation prompt is constructed based on

Input Prompt x: { The Question Generation Prompt } Preferred Response yw: {Preferred Question} Dispreferred Response yl: {Dispreferred Question}

Table 9: DPO dataset template for question generation.

the task description and seed questions, detailed in Table 11. The preferred response is randomly selected from the draft questions with top-25% preference scores, while the dispreferred response is randomly selected from the draft questions with bottom-25% preference scores.

DPO dataset for Rationale Generation For each draft question, we construct the DPO dataset for rationale generation using the following template, The naive Step-by-Step prompt is constructed

Input Prompt x: { The Naive Step-by-Step Prompt } Preferred Response yw: {Preferred Rationale} Dispreferred Response yl: {Dispreferred Rationale}

Table 10: DPO dataset template for rationale generation.

based on the task description and the draft question, detailed in Table 8. The idea of using naive Step-by-Step prompts is insipred by Prompt Erasure [34], which lets the teacher model learn to select the most suitable reasoning strategy to generate tailored training examples for the student model. The preferred response is the draft rationale to the draft question with the highest preference scores, while the dispreferred response is the draft rationale with the lowest preference scores.

Finally, we mix the DPO dataset for question generation and rationale generation together to train the teacher model with DPO.

### H Question Generation Prompt

We provide some prompts for guiding the teacher model to generate questions in the Knowledge Elicitation step.

Task Prompt

boolean_expressions boolean expressions is a task whose description is as follows: Evaluate the result of a random Boolean expression. I will provide you 3 example questions about this task in the following. Based on the following example, I want you to generate 5 more questions about boolean expressions with brainstorming. You may not stick to context, but the question should be related to boolean expressions. The output should be like this: <Question_x><input>...</input><target>...</target></Question_x> x can be any number, but it should be unique.

- <Question_1><input>not ( ( not not True ) ) is</input> <tar-

- get>False</target></Question_1>

<Question_2><input>True and False and not True and True is</input> <tar-

- get>False</target></Question_2>

- <Question_3><input>not not ( not ( False ) ) is</input> <target>True</target></Question_3>

causal_judgment causal judgment is a task whose description is as follows: Answer questions about causal attribution. I will provide you 3 example questions about this task in the following. Based on the following example, I want you to generate 5 more questions about causal judgment with brainstorming. You may not stick to context, but the question should be related to causal judgment. The output should be like this: <Question_x><input>...</input><target>...</target></Question_x> x can be any number, but it should be unique.

- <Question_1><input>How would a typical person answer each of the following questions about causation? Frank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor’s body, causing significant injury. Did Frank T. intentionally shoot his neighbor in the body? Options:

- - Yes
- - No</input><target>No</target></Question_1>

- <Question_2><input>How would a typical person answer each of the following questions about causation? Suzy and Billy are working on a project that is very important for our nation’s security. The boss tells them both: "Be sure that you are here at exactly 9 am. It is absolutely essential that you arrive at that time." Both Billy and Suzy arrive at 9 am. As it happens, there was a motion detector installed in the room where they arrived. The motion detector was set up to be triggered if at least one person appeared in the room at the same time. So the motion detector went off. Did Billy cause the motion detector to go off? Options:

- - Yes
- - No</input><target>Yes</target></Question_2>

- <Question_3><input>How would a typical person answer each of the following questions about causation? George and his sister Lena reunite at their parent’s house for Thanksgiving. Whereas George just got into medical school, Lena is unhappy in her marriage and recently lost her job. Over the course of the day, George and Lena get into a number of heated arguments. Later in the afternoon, they play a game of darts. They split the first two games, and the third game was close until the end. Who will win comes down to George’s last shot. If he hits a high point region, he wins; if he hits a low point region, Lena wins. George thinks of the difficult time Lena is having, and he really wants to let her win. He aims the dart at the low point region. He sets up his shot and the dart lands in the low point region. After his shot, Lena wins the game and is very happy. Did George hit the low point region intentionally? Options:

- - Yes
- - No</input><target>Yes</target></Question_3>

date_understanding date understanding is a task whose description is as follows: Infer the date from context. I will provide you 3 example questions about this task in the following. Based on the following example, I want you to generate 5 more questions about date understanding with brainstorming. You may not stick to context, but the question should be related to date understanding.

The output should be like this: <Question_x><input>...</input><target>...</target></Question_x> x can be any number, but it should be unique.

- <Question_1><input>Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY? Options:

- (A) 12/14/2026
- (B) 12/14/1950
- (C) 12/14/2007
- (D) 12/14/1937
- (E) 07/14/1938
- (F) 12/14/1988</input><target>(D)</target></Question_1>

- <Question_2><input>Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY? Options:

- (A) 09/04/2018
- (B) 11/11/2018
- (C) 08/25/2018
- (D) 11/02/2018
- (E) 11/04/2018</input><target>(B)</target></Question_2>

- <Question_3><input>Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY? Options:

- (A) 01/11/1961
- (B) 01/03/1963
- (C) 01/18/1961
- (D) 10/14/1960
- (E) 01/03/1982
- (F) 12/03/1960</input><target>(B)</target></Question_3>

movie_recommendation movie recommendation is a task whose description is as follows: Recommend movies similar to the given list of movies. I will provide you 3 example questions about this task in the following. Based on the following example, I want you to generate 5 more questions about movie recommendations by brainstorming. You may not stick to context, but the question should be related to movie recommendations. The output should be like this: <Question_x><input>...</input><target>...</target></Question_x> x can be any number, but it should be unique.

- <Question_1><input>Find a movie similar to Star Wars Episode IV - A New Hope, Indiana Jones and the Last Crusade, Star Wars Episode V - The Empire Strikes Back, The Big Lebowski: Options:

- (A) Tetsuo
- (B) the Ironman
- (C) The Princess Bride
- (D) The Barkley Marathons The Race That Eats Its Young
- (E) Bug</input><target>(C)</target></Question_1>

- <Question_2><input>Find a movie similar to Twister, The Silence of the Lambs, Independence Day, Braveheart: Options:

- (A) They Shoot Horses
- (B) Don’t They
- (C) Forrest Gump
- (D) The Salton Sea
- (E) Extreme Days</input><target>(C)</target></Question_2>

- <Question_3><input>Find a movie similar to Minority Report, Total Recall, Inside Out, Forrest Gump: Options:

- (A) Phenomena
- (B) Lilting
- (C) Catwoman

- (D) Edge of Tomorrow</input><target>(D)</target></Question_3>

Table 11: Prompts for question generation.

### I Rationale Generation Prompt for Different Reasoning Techniques

We provide some prompts for guiding the teacher model to generate rationale with different reasoning techniques in the Knowledge Elicitation step.

Task Prompt

Image you are an expert in Boolean expression evaluation. Now you will be given a random Boolean expression, you should first evaluate the expressions inside brackets, then follow the order of operations from highest priority to lowest priority namely "not", "and", "or", respectively, and finally evaluate the result of the random Boolean expression. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

boolean_expressions

You are an expert in Math. Given a random Boolean expression, you should first recall the rules of Boolean algebra and then evaluate the expression step by step. Finally, you should provide the result of the expression. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

Evaluate the result of a random Boolean expression. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

Given you are a binary classification question, solve the question step by step as follows: 1. Read the question and options 2. Find the best option among the remaining ones. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

I want you to act as a judge in a causal judgment process. There are 3 key points you should consider when assessing whether an action was intentional or not. 1. Intentionality and Outcome: When evaluating an action, consider the actor’s intentions and the sequence of events leading to the outcome. If the actor clearly intended the outcome and acted to bring it about, the action is intentional. 2. Accidents and Unintended Consequences: If the outcome resulted from an accident or slip, and not from a deliberate act aimed at causing that specific outcome, it is considered unintentional. Unexpected or uncontrollable factors can influence the final result, diverging from the actor’s original intent. 3. Contribution to Outcome: In cases where multiple factors contribute to an outcome, analyze each factor’s role in producing the final effect. If an agent’s actions directly contribute to triggering an event, they can be considered a cause, even if other factors are also involved. Carefully assess the situation and the actor’s mindset to make a fair and accurate judgment. Based the three key points, given one scenario, you should first analyze the situation and the actor’s mindset. Then, decide which key point is the most relevant to the scenario. Finally, you should conclude by answering the question of whether an action was intentional or not. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

causal_judgment

You are given a scenario and asked to figure out whether the person in the scenario intentionally caused the outcome. Use your common sense and the information given in the scenario to answer the question step by step. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

Answer questions about causal attribution. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

Given you are a binary classification question, solve the question step by step as follows: 1. Read the question and options 2. Find the best option among the remaining ones. Remember you should output your final answer in the end like <ans>True</ans> or <ans>False</ans>

You are given a known date and asked to calculate a date based on a specific time interval or event. To solve these types of questions, follow these steps: 1. Identify the known date provided in the question. 2. Understand the time interval or event mentioned (e.g., "10 days ago", "one year ago", "5-year anniversary"). 3. Calculate the required date by counting backward or forwards from the known date according to the interval or event. 4. Compare the calculated date with the options given to find the correct answer. Finally, select the option that matches the calculated date. Remember you should output your final answer in the end like <ans>(A)</ans> or <ans>(B)</ans>, etc.

date_understanding

Use your knowledge of the calendar and common sense. Carefully read the question and the options. And then, handle the math calculation step by step to find the answer. Finally, select the correct answer.Remember you should output your final answer in the end like <ans>(A)</ans> or <ans>(B)</ans>, etc.

Infer the date from context. Remember you should output your final answer in the end like <ans>(A)</ans> or <ans>(B)</ans>, etc.

Given a multi-choice question, your task is to solve the question step by step as follows: 1. Read the question and options. 2. Eliminate the options that are clearly wrong. 3. Find the best option among the remaining ones. Remember you should output your final answer in the end like <ans>(A)</ans> or <ans>(B)</ans>, etc.

Image you are a movie buff and you are asked to recommend movies similar to a given list of movies. Follow the steps below to find the answer. (1) Identify the common genres and themes of the movies in the list. (2) Consider the time period of production, focusing on classics or movies from a specific era. (3) Compare the options provided, looking for films that match the identified genres, themes, and production period. (4) Select the option that best aligns with the criteria established from the given list of movies. Remember you should output your final answer in the end like <ans>(A)</ans> or <ans>(B)</ans>, etc.

movie_recommendation

Image you are a movie buff and you are asked to recommend movies similar to a given list of movies. You need to think about the genre, the year of production, the actors, and the overall style of the movies. Think step by step and eliminate the options that are not similar to the given list of movies. Choose the option that is most similar to the given list of movies. Remember you should output your final answer in the end like

- <ans>(A)</ans> or <ans>(B)</ans>, etc.

Recommend movies similar to the given list of movies. Remember you should output your final answer in the end like <ans>(A)</ans> or

- <ans>(B)</ans>, etc.

Given a multi-choice question, your task is to solve the question step by step as follows: 1. Read the question and options. 2. Eliminate the options that are clearly wrong. 3. Find the best option among the remaining ones. Remember you should output your final answer in the end like <ans>(A)</ans> or <ans>(B)</ans>, etc.

Table 12: Task prompts for different reasoning techniques.

- J One-shot In-context Learning Template for Preference Collection We provide the in-context learning prompt for the student model in the Preference Collection step.

Task Description: {Task Description} Remember you should include your final answer with the tag <ans> and </ans>. Question: {Example Question} Answer: Let’s think step by step. {Example Rationale} Question: {Test Question} Answer: Let’s think step by step. {to be completed by the student model}

Table 13: One-shot in-context learning prompt for preference collection.

- K Licenses for existing assets The names of the licenses for each asset used in this paper are detailed below.

Asset License

BBH MIT License GPT-4-LLM Apache License Version 2.0 Tulu-v2 ODC-BY WizardLM Apache License Version 2.0 OpenOrca MIT License PIQA Academic Free License v3.0 CSQA MIT License ARC-E CC-BY 4.0 ARC-C CC-BY 4.0 GSM8K MIT License

Gemma-2B Gemma Terms of Use Gemma-7B Gemma Terms of Use Qwen1.5-1.8B Tongyi Qianwen RESEARCH LICENSE CodeGemma-2B Gemma Terms of Use Llama-3-70B-Instruction META LLAMA 3 COMMUNITY LICENSE MiniCPM-2B Apache License Version 2.0

Table 14: Licenses for each asset in the paper.

