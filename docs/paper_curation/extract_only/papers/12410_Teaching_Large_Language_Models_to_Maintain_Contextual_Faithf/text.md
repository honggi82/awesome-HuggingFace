## Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning

### Shuzheng Si*♠♢, Haozhe Zhao*♣, Cheng Gao*♠, Yuzhuo Bai♠, Zhitong Wang♠ Bofei Gao♡, Kangyang Luo♠, Wenhao Li♠, Yufei Huang♠, Gang Chen♢ Fanchao Qi†♠♢, Minjia Zhang♣, Baobao Chang♡, Maosong Sun†♠

♠ Tsinghua University ♡ Peking University ♢ DeepLang AI

♣ University of Illinois Urbana-Champaign

###### Abstract

# arXiv:2505.16483v3[cs.CL]12Nov2025

Optimal Performance/Params Ratio

Teaching large language models (LLMs) to be faithful in the provided context is crucial for building reliable informationseeking systems. Therefore, we propose a systematic framework, CANOE, to reduce faithfulness hallucinations of LLMs across different downstream tasks without human annotations. Specifically, we first synthesize short-form question-answering (QA) data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. Also, we propose Dual-GRPO, a rule-based reinforcement learning method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while simultaneously optimizing both short-form and long-form response generation. Notably, Dual-GRPO eliminates the need to manually label preference data to train reward models and avoids over-optimizing short-form generation when relying only on the synthesized short-form QA data. Experimental results show that CANOE greatly improves the faithfulness of LLMs across 11 different tasks, even outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1. 1

Ours-LLaMA3 -8B-Instruct

75

Ours-Qwen2.5 -14B-Instruct

Avg.Scoreon11DownstreamTasks(%)

[Figure 1]

70

Ours-Qwen2.5 -7B-Instruct

[Figure 2]

DeepSeek-R1

65

DeepSeek-V3

o1

60

SOTA LLMs

Our CANOE

GPT-4o

55

Vanilla LLMs

Qwen2.5-72B -Instruct

[Figure 3]

Claude 3.7 Sonnet

Qwen2.5-7B -Instruct

[Figure 4]

Qwen2.5-14B -Instruct

50

LLaMA3-70B -Instruct

[Figure 5]

LLaMA3-8B -Instruct

Qwen2.5-32B -Instruct

45

[Figure 6]

7B 14B 32B 70B 671B

N/A

Model Size (# Parameters)

Figure 1: Average score on 11 downstream tasks vs model size. With only 7B parameters, CANOE already exceeds stateof-the-art LLMs like GPT-4o and OpenAI o1.

### 1 Introduction

Recent progress in large language models (LLMs) has revolutionized text generation with their remarkable capabilities (OpenAI 2023). In practice, LLMs are widely used to generate fluent and coherent text responses based on the provided contextual information, e.g., document question answering (QA) (Wang et al. 2024) and text summarization (Zhang, Yu, and Zhang 2024). However, LLMs often generate responses that are not faithful or grounded in the input context, i.e., faithfulness hallucinations (Huang et al. 2024; Si et al. 2025a), which can undermine their trustworthiness. Maintaining faithfulness to the context is especially important in fields where accurate information transfer is essential (Duong et al. 2025). For instance, in legal summarization (Dong et al. 2025), the text output must reflect the content of legal documents without introducing any distortions.

works (Xie et al. 2024; Li et al. 2025) find that LLMs may overly rely on internal knowledge learned from extensive pretraining data while disregarding provided contexts, i.e., the knowledge conflicts (Xu et al. 2024). When the model parameters increase and internal knowledge grows, this may lead to greater knowledge conflicts and further lower the faithfulness of LLMs (Ming et al. 2025). Thus, it is necessary to explore a tailored post-training method to improve the faithfulness instead of simply scaling the model parameters. (2) Faithfulness is challenging to consistently boost across different downstream tasks: Recently, several methods (Li, Li, and Zhang 2024; Duong et al. 2025) have been proposed to improve the faithfulness of LLMs for different tasks. For example, Bi et al. (2024) aligns LLMs through DPO (Rafailov et al. 2023) with constructed faithful and unfaithful shortform completions, improving the performance of LLMs on short-form QA tasks. However, these recent methods are designed for specific tasks, so they fail to consistently improve the faithfulness of LLMs across various tasks, like text summarization and multiple-choice questions, because these tasks can vary greatly. (3) Data used to enhance faithfulness is hard to scale: This issue is especially problematic with

However, improving the faithfulness of LLMs faces three key challenges. Specifically, (1) Faithfulness is difficult to improve by simply scaling model parameters: Previous

* Equal Contribution. † Corresponding Authors. 1 The data, code, and models will be available at https://github.

com/S1s-Z/CANOE. Email: ssz24@mails.tsinghua.edu.cn.

###### The Framework of CANOE

###### Overall Pipeline

Short-form QA Data Synthesis Dual-GRPO Training Process

Collect Triples from Wikidata

[Figure 7]

[Figure 8]

[Figure 9]

h r t

[Figure 10]

[Figure 11]

[Figure 12]

Policy Model

[Figure 13]

[Figure 14]

Collected Triples

10,000 Synthesized Training Samples

Wikidata

Generate a Set of Responses

[Figure 15]

Training Data Synthesis

[Figure 16]

[Figure 17]

Context Synthesis: Considering given facts, generate a brief description of head entity.

[Figure 18]

Reasoning Process Long-form Response Short-form Response

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

KL Divergence Constraint

Rule-based RL Training

Question Synthesis: Considering given facts, generate a question about the triple.

Data Synthesis

[Figure 23]

Optimize Policy via Dual-GRPO

Calculate Rewads

[Figure 24]

[Figure 25]

###### Dual-GRPO Trainer

Reference Model

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

|[Figure 31]<br><br>×1|
|---|
|[Figure 32]<br><br>×2|
|[Figure 33]<br><br>×3|
|[Figure 34]<br><br>×4|

Straightforward Context Reasoning-required Context Inconsistent Context Counterfactual Context

ComplexityLevels

[Figure 35]

[Figure 36]

Format Reward Proxy Reward Accuracy Reward

[Figure 37]

[Figure 38]

1

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Reward Function for Generated Responses

CANOE Training

Final Reward

Synthesized QA Data with Four Diverse Tasks

- Figure 2: An overview of CANOE framework. CANOE first synthesizes easily verifiable short-form QA data and then proposes the Dual-GRPO with designed rule-based rewards to improve the faithfulness of LLMs.

data used to improve the faithfulness in long-form generation tasks. Unlike tasks with clear answers, e.g., short-form factseeking QA tasks (Wei et al. 2024), there is no standard way to ensure data quality in long-form generation tasks (Duong et al. 2025). Thus, data is typically annotated by humans (Zhu, Qi, and Lau 2023), which is costly and not scalable.

then feed it to the LLMs to evaluate whether a long-form answer can drive the LLMs toward the correct short-form answer. If the generated long-form response enables LLMs to generate the correct final answer, this indicates that it remains context-faithful and contains easy-to-understand sentences that answer the question correctly. We also introduce format rewards to ensure more structured outputs and contribute to more stable training. To obtain the data used for training without human annotation, we collect head-relation-tail triples from the knowledge base, apply the advanced GPT-4o (OpenAI 2023) to synthesize the question and contextual information, and use the tail entity from the triple as the answer to ensure the correctness. Moreover, we introduce four diverse QA tasks to ensure the complexity and diversity of the training data. Combined with the rule-based Dual-GRPO and data synthesis, CANOE can teach LLMs to remain contextfaithful in both short-form and long-form generation tasks without relying on human annotations.

To tackle these challenges, we propose a systematic posttraining method called CANOE. The main idea behind CANOE is to synthesize easily verifiable short-form QA data and then leverage reinforcement learning (RL) with tailored rulebased rewards to improve the faithfulness of LLMs in both short-form and long-form generation tasks. CANOE firstly introduces Dual-GRPO, a variant of GRPO (Shao et al. 2024) that includes three carefully tailored rule-based RL rewards derived from synthesized short-form QA data, while optimizing both short-form and long-form response generation. For the provided contextual information and question, DualGRPO first prompts LLMs to produce a reasoning process, followed by a long-form answer composed of detailed and complete sentences, and finally a concise short-form and easily verifiable answer in just a few words. For example, given the context, if the question is “What is the country of origin of Super Mario?”, the long answer could be “Super Mario originated from Japan.”, while the short answer could simply be “Japan”. In this way, we can assign different rewards to long-form and short-form responses, optimizing both simultaneously. Note that we assign accuracy rewards on generated short-form responses since the short-form QA task enables reliable rule-based verification of faithfulness. To overcome the problem of the faithfulness of the generated long-form responses being difficult to evaluate via rule-based verification (OpenAI 2025), we propose proxy rewards to evaluate it implicitly. Specifically, we construct the new input by replacing the given context with the generated long-form answer,

We evaluate the effectiveness of CANOE across 11 different downstream tasks, covering short-form and long-form generation tasks. Results show that CANOE significantly reduces faithfulness hallucinations. Specifically, CANOE significantly improves the overall score, e.g., 22.6% for Llama3-Instruct8B. Meanwhile, CANOE surpasses the most advanced LLMs (e.g., GPT-4o) in the overall score. As shown in Figure 1, these results are unprecedented for open-source models that do not rely on additional human annotations.

### 2 Methodology

In this section, we will detail our proposed framework CANOE, which aims to teach LLMs to remain faithful across different tasks without human annotation. Specifically, we first synthesize easily verifiable short-form QA data and then

propose the Dual-GRPO with designed rule-based rewards to improve the faithfulness of LLMs in both short-form and long-form response generation. We start with the introduction of the short-form data synthesis process, then a brief overview of RL protocol, and the tailored rule-based rewards used in the proposed Dual-GRPO training. An overview of the CANOE framework is presented in Figure 2.

#### 2.1 Training Data Construction

Constructing high-quality and easily verifiable data is crucial for rule-based RL training (Shao et al. 2024). Inspired by knowledge base question generation (Cui et al. 2019; Guo et al. 2024), we attempt to collect triples from the knowledge base and use the advanced LLMs to synthesize the context and question. Concretely, we first collect about 30,000 headrelation-tail triples from Wikidata (Vrandeˇci´c and Kr¨otzsch 2014). Each collected triple (h,r,t) includes a head entity h, a tail entity t, and the relation r between two entities. Then we craft prompt templates and query the most advanced GPT4o to synthesize the contextual information c and question q based on the triple (h,r,t). We directly use the tail entity t as the final answer a to ensure the correctness and easy validation of the synthesized data. Each synthetic short-form QA sample (c,q,a) consists of a contextual passage c, a question q, and a ground truth answer a. In this way, we can obtain short-form QA data that can be easily verified; thus, we can utilize a rule-based RL method to optimize our LLMs to be more faithful. Meanwhile, to ensure the complexity and diversity of training data, we design four diverse QA tasks, including straightforward context, reasoning-required context, inconsistent context, and counterfactual context. The model is expected to answer the question by leveraging the information in the provided context.

Straightforward Context. A straightforward context means that the context clearly contains statements of the final answer. It requires models to accurately locate and utilize information from the context in order to answer questions. Specifically, we keep the original collected triple as input to query GPT-4o to synthesize the data (c,q,a).

Reasoning-required Context. This type of context contains multiple related entities and relations, and requires models to correctly answer multi-hop reasoning questions. Firstly, we construct a subgraph based on the sampled triples and extract 2, 3, 4-hop paths [(h1,r1,t1),...,(hn,rn,tn)]n≤4. Then, we use the n-th tail entity tn as the ground truth answer and employ the constructed paths to query GPT-4o to obtain the multi-hop context and question.

Inconsistent Context. This involves multiple randomly ordered contexts generated from different triples. This simulates noisy and inconsistent scenarios, where models need to detect inconsistencies and focus on useful and relevant contexts to answer the questions. We construct such a sample by combining the contexts from up to three QA samples.

Counterfactual Context. A counterfactual context contains statements that contradict common sense within the collected triples. Firstly, we replace the tail entity t of the original collected triple with a similar but counterfactual entity tcf. Then, we query GPT-4o to generate questions and counterfactual contexts to construct counterfactual samples. Unlike the

aforementioned tasks, this task further highlights the importance of faithfulness for LLMs to answer the questions correctly, as it prevents models from depending on their learned factual knowledge to find the right answers.

By introducing four different tasks, we construct 10,000 QA pairs used for training without human annotation. These short-form QA data can be easily verified and include tasks varying in complexity, which can make rule-based RL training more efficient in improving the faithfulness. More details are shown in the Appendix A, e.g., used prompts, data mixing recipes, and data statistics.

#### 2.2 Reinforcement Learning Protocol

For RL training of LLMs, methods based on policy optimization, such as PPO (Schulman et al. 2017) and GRPO (Shao et al. 2024), have been explored. Given the effectiveness of GRPO in training models and its advantages over PPO, e.g., eliminating the need for human-annotated preference data to train a reward model, we utilize GRPO to optimize and improve the faithfulness of the policy model πθ.

For each input, consisting of provided contextual information c, a natural language question q, the model generates a group of G candidate answers, {o1,o2,...,oG}. Each candidate is evaluated using a designed composite rule-based reward function to capture the end goal of faithfulness. GRPO leverages the relative performance of candidates within the group to compute an advantage Ai for each output, guiding policy updates according to the following objective:

JGRPO(θ) = Ec,q,{o

i}∼πθold

G

1 G

Li − βDKL(πθ||πref) , (1)

i=1

Li = min(wiAi,clip(wi,1 − ϵ,1 + ϵ)Ai), (2) where wi = π

θ(oi|q)

πθold(oi|q), πθ

is the policy before the update,

old

πref is the reference policy (i.e., the initial model), ϵ and β are hyperparameters controlling the update step and divergence

regularization and Ai is computed using the normalized reward within the group. We use our synthesized QA data as training data, which is easily verifiable, so that we can apply GRPO and train LLMs using rule-based rewards. Also, employing the rule-based GRPO removes the need for humans to label preference data for training the reward model.

#### 2.3 Reward Design

Having a well-designed reward is key to the effectiveness of RL training (Du et al. 2025). To use easily verifiable shortform QA data to improve the faithfulness of LLMs, the most intuitive reward would be the accuracy reward, which can check if the generated responses match the ground truth answers. However, in our early experiments, we found that relying solely on short-form QA data and accuracy rewards fails to enhance the faithfulness of long-form response generation, as the models may over-optimize short-form generation and learn a false pattern. The tuned models tend to copy text spans from the context as answers and lose their ability to generate long-form responses. Meanwhile, directly evaluating the faithfulness of long, free-form responses via the rule-based verification continues to pose an unresolved challenge.

Therefore, we propose Dual-GRPO, which includes a set of well-designed rewards that provide more harmonized guidance for optimizing LLMs to generate faithful responses. Unlike the original GRPO that over-optimizes short-form generation, we first prompt LLMs to generate both long-form and short-form responses, then assign different rewards to the two generated responses to improve the faithfulness.

System Prompt and Rollouts. For the provided context and question, we introduce a system prompt that requires LLMs to produce a reasoning process, then a long-form answer composed of detailed and complete sentences, and finally a concise short-form answer in just a few words. For example, given the context, if the question is “What is the country of origin of Super Mario?”, the long answer could be “Super Mario originated from Japan.”, while the short answer could simply be “Japan”. In this way, we can assign different reward scores to long-form and short-form answers while optimizing them both at once. This system prompt also triggers zero-shot chain-of-thought reasoning in the policy model, which progressively improves as training advances to optimize for the reward. The system prompt used for Dual-GRPO rollouts is shown in the Appendix B.

Accuracy Reward for Short-form Response Generation. This reward directly assesses whether the generated shortform responses match the ground truth answers. We use the exact matching (EM) to measure accuracy, giving a score of 1 for a match and 0 for a mismatch. Thus, we can ensure that the generated short-form response correctly answers the question based on the context, making LLMs more faithful in short-form response generation.

Proxy Reward for Long-form Response Generation. Evaluating the faithfulness of long-form responses via the rulebased verification remains challenging. This is because these long-form answers are often free-form, making rule-based verification ineffective (OpenAI 2025). Therefore, instead of directly evaluating the faithfulness of the long-form response, we propose a proxy reward to evaluate it implicitly, as the faithfulness of a long-form answer can be measured by its ability to drive the LLMs toward a correct short-form answer. Specifically, for each generated long-form answer ylf, we replace the given context c with it as new input and feed it to the LLM to check whether the LLM can produce the correct short-form answer based on ylf. If the generated long-form response can enable the LLM to generate the correct answer, it indicates that the long-form response stays faithful to the context, contains complete and easy-to-understand sentences, and correctly addresses the question. Thus, we assign a reward score of 1 for the positive long-form response that helps the LLM to produce the correct final answer, and a reward score of 0 for those that lead to an incorrect answer.

Format Reward. We also include a format reward that encourages adherence to a predefined output structure (e.g., using <think>, <long answer>, and <short answer>tags). Outputs that conform to this pattern receive a reward boost, thereby enhancing consistency. We use string matching to evaluate whether the generated responses adhere to the format, giving a score of 1 for a match and 0 for a mismatch.

Finally, we use the sum of these three rewards as the final composite reward. It enhances the efficacy of the RL training

framework, guiding the model toward generating more faithful responses in both short-form and long-form tasks. More details are shown in the Appendix B.

### 3 Experiments

In this section, we conduct experiments and provide analyses to justify the effectiveness of CANOE.

#### 3.1 Tasks and Datasets

To evaluate our method CANOE comprehensively, we select a range of downstream datasets, including short-form and long-form generation tasks.

Short-form Generation Tasks. For short-form generation tasks, we use two counterfactual QA datasets, including ConFiQA (Bi et al. 2024) and CNQ (Longpre et al. 2021), a multiple-choice questions dataset FaithEval (Ming et al. 2025), and a factual QA dataset FiQA (Bi et al. 2024) that is the factual version of ConFiQA. These datasets ensure the answers appear in the contexts to evaluate the faithfulness. We also evaluate our method on four open-domain QA datasets within the FollowRAG benchmark (Dong et al. 2024) to evaluate the abilities of LLMs in real-world retrievalaugmented generation (RAG) scenarios, including NaturalQA (Kwiatkowski et al. 2019b), TriviaQA (Joshi et al. 2017), HotpotQA (Yang et al. 2018), and WebQSP (Yih et al. 2016). In real-world RAG scenarios, the answer may not appear in the retrieved passages, and these passages tend to be noisy. We evaluate models based on whether gold answers are included in the generated responses (i.e., Acc) following Asai et al. (2024) and exact matching (EM) for QA tasks. For multiple-choice questions, we follow Ming et al. (2025) and use keyword matching to verify the accuracy.

Long-form Generation Tasks. We include a text summarization task XSum (Narayan, Cohen, and Lapata 2018), a text simplification task WikiLarge (Zhang and Lapata 2017), and a long-form QA task CLAPNQ (Rosenthal et al. 2025). To evaluate the faithfulness of generated long-form answers, called FaithScore (FS), we use MiniCheck (Tang, Laban, and Durrett 2024) to check whether the model response is grounded in the provided context. MiniCheck is a state-ofthe-art method to recognize if LLM output can be grounded in given contexts. If the model response contains at least one statement that cannot be inferred from the context, we consider it as a negative response; otherwise, it is a positive response. We also query GPT-4o to evaluate the quality of generated responses, namely QualityScore.

More details can be found in the Appendix C.

#### 3.2 Baselines and Implementation Details

Baselines. We compare several baselines, including (1) Vanilla LLMs: including LLaMA-3-Instruct (Grattafiori et al. 2024) and Qwen-2.5-Instruct (Yang et al. 2024) of different sizes. We also conduct supervised fine-tuning on synthesized 10,000 short-form data as SFT baselines. (2) SOTA LLMs: We further evaluate the most advanced LLMs, including GPT-4o, GPT-4o-mini, OpenAI o1 (Jaech et al. 2024), Claude 3.7 Sonnet (Anthropic 2025), Claude 3.7 SonnetThinking, DeepSeek R1, and DeepSeek V3 (DeepSeek-AI

Short-form Generation Tasks Long-form Generation Tasks

Avg. Score ConFiQA FiQA CNQ FaithEval FollowRAG XSum WikiLarge CLAPNQ

Model

EM Acc EM Acc EM Acc Acc EM Acc FS FS FS Avg EM Avg Acc The state-of-the-art LLMs

GPT-4o 31.5 42.7 66.8 79.6 43.4 55.9 47.5 42.2 57.8 80.7 88.1 70.3 58.8 65.3 GPT-4o mini 49.5 63.7 67.1 78.8 47.8 54.3 50.9 38.5 51.3 75.4 91.0 66.0 60.8 66.4 DeepSeek V3 49.5 58.6 67.0 76.5 54.6 67.3 51.0 37.7 55.2 82.8 85.6 71.0 62.4 68.5 Claude 3.7 Sonnet 26.0 36.0 56.4 72.2 41.4 65.0 45.6 36.3 53.7 78.3 81.7 68.3 54.3 62.6 OpenAI o1 49.0 57.9 78.0 89.7 29.5 39.1 52.0 40.5 57.0 81.0 88.1 68.0 60.8 66.6 DeepSeek R1 68.4 74.3 68.4 80.7 60.3 70.2 60.1 42.9 56.6 80.3 83.0 73.5 67.1 72.3 Claude 3.7 Sonnet-Thinking 27.1 38.7 59.5 76.7 42.1 67.0 57.0 38.8 55.3 79.0 81.4 72.2 57.1 65.9

###### LLaMA-3-Instruct Series

LLaMA-3-Instruct-8B 49.2 58.2 11.4 59.3 37.8 45.2 52.0 31.1 44.8 64.2 77.1 58.5 47.7 57.4 LLaMA-3-Instruct-70B 38.1 54.5 9.1 66.8 54.2 65.0 50.9 38.7 45.7 72.0 77.4 47.2 48.5 59.9 SFT-8B 65.1 70.3 35.9 59.9 52.6 65.7 43.0 19.2 21.0 62.2 74.2 55.3 50.9 56.4 Context-DPO-8B 66.3 72.9 40.9 59.5 54.6 62.3 37.5 29.9 43.8 65.2 78.2 59.1 54.0 59.8 SCOPEsum-8B 35.7 64.6 7.1 68.7 33.8 60.6 55.7 30.1 46.2 70.3 80.3 59.8 46.6 63.3 CANOE-LLaMA-8B 73.5 80.9 82.7 84.9 66.7 73.4 74.6 40.9 51.7 74.4 84.4 64.9 70.3 73.6 ∆ Compared to Vanilla. +24.3 +22.6 +71.3 +25.6 +28.9 +28.2 +22.6 +9.8 +6.9 +10.2 +7.3 +6.4 +22.6 +16.2

###### Qwen-2.5-Instruct Series

Qwen-2.5-Instruct-7B 52.5 61.0 13.2 68.4 55.3 68.2 56.1 32.6 45.3 63.4 57.8 61.2 49.0 60.2 Qwen-2.5-Instruct-14B 34.1 47.3 0.8 61.4 43.1 64.3 51.6 34.8 51.2 68.2 82.3 63.4 47.3 61.2 Qwen-2.5-Instruct-32B 44.5 66.4 39.2 81.1 37.7 66.4 47.0 33.9 53.1 20.2 57.7 31.7 39.0 52.9 Qwen-2.5-Instruct-72B 43.7 52.3 4.8 67.3 51.8 62.2 45.2 38.5 55.7 71.2 90.4 64.8 51.3 63.6 SFT-7B 62.8 69.8 48.8 76.6 60.1 65.3 50.3 29.0 41.7 55.2 51.3 57.2 51.8 58.4 Context-DPO-7B 64.5 70.6 57.1 78.2 62.3 70.1 45.7 31.0 43.7 60.2 53.4 62.8 54.6 60.6 SCOPEsum-7B 39.3 47.9 12.9 60.9 50.2 55.3 52.3 30.6 46.0 68.3 72.0 63.2 48.6 58.2 CANOE-Qwen-7B 67.6 75.2 78.1 83.5 67.2 76.4 70.5 37.0 50.2 72.4 86.1 65.2 68.0 72.4 ∆ Compared to Vanilla. +15.1 +14.2 +64.9 +15.0 +11.9 +8.2 +14.4 +4.4 +4.9 +9.0 +28.3 +4.0 +19.0 +12.3

CANOE-Qwen-14B 85.7 87.4 87.8 88.5 81.8 84.2 67.4 46.1 54.6 75.7 91.1 68.4 75.5 77.2 ∆ Compared to Vanilla. +51.6 +40.1 +87.0 +27.1 +38.7 +19.9 +15.8 +11.3 +3.4 +7.5 +8.8 +5.0 +28.2 +16.0

Table 1: Experimental results (%) on eleven datasets. The FollowRAG results represent the results averaged over these four open-domain QA datasets, as shown in the Appendix C, including NaturalQA, TriviaQA, HotpotQA, and WebQSP. Bold numbers indicate the best performance of models with the same model size. Avg EM/Acc represents the average score between short-form task metrics (EM/Acc) and long-form task metric FaithScore (FS).

et al. 2025a,b). (3) The Methods to Improve Faithfulness: Context-DPO (Bi et al. 2024) aligns LLMs through DPO with constructed faithful and unfaithful short-form answers, thus improving the faithfulness in short-form generation. SCOPE (Duong et al. 2025) proposes a pipeline to generate selfsupervised task-specific data and applies preference training to enhance the faithfulness in a special task. We train it on the sampled training set of the summarization task XSum as SCOPEsum, regarding it as the method designed to improve the faithfulness of long-form response generation.

Implementation Details. Our experiments are conducted on LLaMA-3-Instruct and Qwen-2.5-Instruct. Details are shown in Appendix D, e.g., hyperparameters.

#### 3.3 Main Results

CANOE Improves the Faithfulness of LLMs in Both Shortform and Long-form Response Generation. As shown in Table 1, CANOE shows significant improvements on 11 faithfulness-related benchmarks. CANOE achieves substantial improvements in the overall score compared to original LLMs, e.g., 22.6% for Llama3-8B and 19.0% for Qwen2.57B in Avg EM score. CANOE also surpasses the most advanced LLMs (e.g., GPT-4o) in the overall score (both Avg EM and Avg Acc scores). This shows that CANOE effectively aligns LLMs to be context-faithful. Also, for real-world RAG scenarios, our proposed CANOE can also improve the performance even though the answer may not appear in the retrieved passages, and these passages are often noisy.

CANOE Maintains the Factuality of LLMs. We further

Model XSum WikiLarge CLAPNQ Avg GPT-4o 98.5 97.5 81.2 92.4 LLaMA-3-Instruct-8B 70.9 82.9 39.2 64.3 LLaMA-3- Instruct-70B 86.2 83.0 30.1 66.4 CANOE-LLaMA-8B 85.8 87.8 65.5 79.7 Qwen-2.5-Instruct-7B 79.4 79.0 64.6 74.3 Qwen-2.5-Instruct-14B 90.5 83.1 63.6 79.1 Qwen-2.5-Instruct-32B 90.3 83.9 58.6 77.6 Qwen-2.5-Instruct-72B 95.7 94.1 75.4 88.4 CANOE-Qwen-7B 91.5 87.3 68.2 82.3 CANOE-Qwen-14B 91.9 89.7 73.5 85.0

Table 2: QualityScore (%) on long-form generation tasks.

evaluate whether CANOE will reduce the factuality of LLMs. Following Ming et al. (2025), we modify the original FaithEval and make it a closed-book QA setting, where no context is provided and LLMs need to give factual answers. In this case, the models rely entirely on their parametric knowledge of common facts, and we find that our proposed CANOE maintains the factuality compared to the untuned LLM as shown in Figure 3. However, when a new context with counterfactual evidence that contradicts the model’s parametric knowledge is introduced, performance declines sharply. For example, GPT-4o achieves 96.3% accuracy on factual closed-book QA task but only 47.5% on counterfactual QA task that evaluates the faithfulness of LLMs. This highlights that, unlike factuality, the faithfulness of LLMs is difficult to improve by

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

- Figure 3: Model performance comparison on FaithEval in a closed-book QA setting and counterfactual context setting. Our models are colored in orange. We report the results from the chat version of LLaMA-3 and Qwen-2.5.

Acc EM QA MR MC QA MR MC

Model

GPT-4o 52.2 45.6 30.3 43.3 32.4 18.7 LLaMA-3-Instruct-8B 69.7 55.9 49.1 60.0 47.9 39.6 CANOE-LLaMA-8B 82.7 80.1 79.8 76.4 73.5 70.5 Qwen-2.5-Instruct-7B 72.8 59.1 51.1 64.9 50.2 42.5 Qwen-2.5-Instruct-14B 62.4 44.9 34.7 44.7 34.3 23.3 Qwen-2.5-Instruct-32B 74.1 65.9 59.3 55.9 42.8 34.8 Qwen-2.5-Instruct-72B 63.3 50.3 43.3 54.3 42.2 34.7 CANOE-Qwen-7B 79.5 76.1 70.1 73.3 67.9 61.7 CANOE-Qwen-14B 91.8 86.4 84.1 89.7 85.2 82.1

Table 3: Results (%) on three tasks in ConFiQA.

simply scaling model parameters, which further indicates the necessity of a post-training method to improve faithfulness. CANOE Improves the Quality of Long-form Response Generation. As shown in Table 2, we can find that our proposed CANOE consistently improves the generation quality in the three long-form tasks. This is because the proxy reward implicitly requires LLMs to generate easy-to-understand responses, which further optimizes the response quality.

CANOE Enhances LLMs’ Reasoning in Short-form Response Generation. ConFiQA consists of three different tasks: question answering (QA), multi-hop reasoning (MR), and multi-conflicts reasoning (MC). QA focuses on the singlehop task with context containing one corresponding answer, while MR and MC involve multi-hop reasoning tasks with context containing one and multiple related counterfactual contexts, respectively. As shown in Table 3, CANOE not only improves the faithfulness in the single-hop QA task but also enhances the reasoning ability in reasoning tasks.

CANOE Mitigates Overconfidence Bias. For each model, we select a total of 110 unfaithful samples with the highest perplexity from the 11 datasets, 10 samples per dataset. Then we report the average perplexity score on these negative samples shown in Figure 4. We can find that CANOE produces the high perplexity scores, indicating low confidence scores, for these bad cases. This shows that CANOE mitigates overconfidence in these false statements.

#### 3.4 Analysis

Ablation Study. We conduct an ablation study in Table 4. The result reveals that our proposed CANOE (including Dual-

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Figure 4: The average perplexity score of 110 negative samples for each model from eleven datasets.

Short-form Tasks Long-form Tasks EM Acc FaithScore QualityScore

Model

CANOE-LLaMA-8B 67.7 73.1 74.6 79.7

- -w/o. Dual-GRPO & Data Synthesis 36.3 51.9 66.6 64.3

- -w/o. Dual-GRPO (i.e., original GRPO) 60.5 66.6 N/A 23.5

- -w/o. Reasoning-required Context. 63.7 69.4 71.7 75.3
- -w/o. Inconsistent Context. 64.4 70.2 70.2 72.5
- -w/o. Counterfactual Context. 62.6 67.8 69.7 73.7

Table 4: Results (%) of ablation study. EM/Acc in shortform tasks represents the average score between QA metrics (EM/Acc) and the accuracy of FaithEval. N/A means a false generation pattern hacks this metric.

GRPO and the designed short-form data synthesis) significantly improves the faithfulness of LLMs in both short-form and long-form generation tasks. For Dual-GRPO, we observe that directly applying GRPO to synthesized short-form data leads to over-optimizing short-form generation and a false response generation pattern. We find that tuned models tend to directly copy text spans from the given context as the final answer instead of following instructions in long-form generation tasks (we show the case study in Table 5 to visually show this phenomenon). Thus, the generated responses do not contain syntactically and semantically complete sentences for long-form generation tasks, which leads to low QualityScore performance and also invalidates the metric used for evaluating faithfulness. For the designed QA tasks used to ensure the complexity and diversity of training data, we replace the designed Reasoning-required Context and Inconsistent Context samples with the same number of samples that contain

Text Summarization Task Inputs: {Documents} Instruction: Refer to the passage above and provide a summary as the response. GRPO-LLaMA-8B: Comedy clubs. (✗)

CANOE-LLaMA-8B: Kareem Badr and his friends were able to turn around the struggling comedy club, the Hideout, by applying their skills in problem-solving and efficiency. They broke down the business into small chunks and figured out how to make each aspect better and more efficient. This approach helped them to improve the club’s performance and eventually expand it. The passage also mentions that the growing popularity of stand-up comedy is fueled in part by younger audiences, who view humorous TV hosts as their primary source for news. Many comedy clubs now require talent, and there is a career and a living to be made by good comedians who are not superstars. The clubs also make money from selling drinks and dinners, as well as from cover charges and talent fees. In addition, digital media has given live comedy a boost, with one terrific joke being able to get a million hits online. (✓)

- Table 5: Case study between the models trained via GRPO and our model for the summarization task.

Question: What do Greeks say when they break plates?

Qwen-2.5-Instruct-7B: Based on the information provided in the passages, there is no specific mention of what exactly Greeks say when they break plates during the plate-breaking ceremony. (✗)

CANOE-Qwen-7B: According to the information given, in the Three’s Company episode “Opening Night,” when Jack breaks the plates at the end of the celebration, he exclaims “Opa!”.(✓) This suggests that “Opa!” might be a phrase commonly used in Greek culture during such festivities. (✓) “Opa!” is often used to express joy, excitement, or approval, which aligns with the celebratory nature of plate smashing in Greek traditions. (✓)

- Table 6: Case study from long-form QA task CLAPNQ. For different useful statements, we use different colors.

Straightforward Context. We find that involving these more challenging instances can improve the effectiveness of RL training. We also replace the data points that contain Counterfactual Context with the same number of factual samples. The designed Counterfactual Context improves the final performance as it prevents models from depending on their learned factual knowledge to find the right answers.

Case Study. We find that directly applying GRPO instead of our Dual-GRPO to synthesized short-form data leads to over-optimizing short-form generation and a false response generation pattern. As shown in Table 5, we find that the tuned model GRPO-LLaMA-8B tends to copy text spans from the given context as the final answer instead of following instructions in the summarization task. We also observe the same phenomenon in the tasks of text simplification and long-form QA; more details can be found in the Appendix F. However, when we apply Dual-GPRO, we find that trained models can generate fluent and complete sentences. Thus, Dual-GRPO not only improves the faithfulness of LLMs in two types of response generation but also ensures the utility. We also conduct a case study in Table 6 to show the advantages of CANOE. CANOE ensures the statements are faithful and comprehensive, and the text flows naturally.

Human Evaluation. Evaluating long-form generation tasks remains challenging. Thus, we conduct human evaluation in the Appendix E to show the effectiveness. We find that our method reduces faithfulness hallucinations and also ensures

the response quality for three long-form generation tasks.

Discussion. We discuss possible concerns in the Appendix F, e.g., the effect of the amount of synthesized data, multilingual transfer ability, and context length generalization of CANOE.

### 4 Related Work

Recently, the demand for utilizing LLMs to generate coherent text responses based on the provided contexts has continued to grow, particularly in text summarization and RAG scenarios. However, LLMs are often criticized for generating outputs that deviate from the provided contents, namely faithfulness hallucination (Li et al. 2022; Ji et al. 2023; Huang et al. 2024). Many approaches have been proposed to improve the faithfulness of language models (Zhu et al. 2021; Jones et al. 2024; Hu et al. 2024; Li and Ng 2025; Yang et al. 2025; Zhang et al. 2025; Wang et al. 2025). The first line of work focuses on the inference stage, such as designing prompts to encourage context integration (Zhou et al. 2023), improving context quality via explicit denoising (Xu, Shi, and Choi 2024), and context-aware decoding to amplify contexts (Shi et al. 2024). Although effective, these approaches primarily serve as a compensatory way rather than enabling the model to inherently learn to prevent generating unfaithful responses. Thus, many studies attempt to apply post-training methods to improve the faithfulness. Bi et al. (2024) utilizes constructed faithful and unfaithful short-form completions and applies DPO to align LLMs to be context-faithful in short-form QA tasks. Huang et al. (2025) trains LLMs to discriminate between faithful and unfaithful responses in long-form QA tasks by unfaithful response synthesis and contrastive tuning. Li et al. (2025) introduces a self-evolving framework to maintain the faithfulness of LLMs in long-form QA tasks through iterative self-improvement. Duong et al. (2025) proposes a pipeline to generate a self-supervised task-specific dataset and applies preference training to enhance the faithfulness for a special task. However, these methods struggle to consistently improve the faithfulness of LLMs across various tasks, as these methods are designed for specific tasks. Thus, how to consistently improve the faithfulness of LLMs on different downstream tasks, including short-form and long-form generation tasks, still remains under-explored.

### 5 Conclusion

In this paper, we propose CANOE, a systematic post-training method for teaching LLMs to remain faithful in both shortform and long-form generation tasks without human annotations. By synthesizing diverse short-form QA data and introducing Dual-GRPO, a tailored RL method with three welldesigned rule-based rewards, CANOE effectively improves the faithfulness of LLMs. We first synthesize short-form QA data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. We then propose Dual-GRPO, a rule-based RL method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while optimizing both short-form and long-form response generation simultaneously. Experimental results show that CANOE consistently improves the faithfulness of LLMs across diverse downstream tasks.

### References

Anthropic. 2025. Claude 3.7 Sonnet System Card.

Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations.

Bai, Y.; Lv, X.; Zhang, J.; Lyu, H.; Tang, J.; Huang, Z.; Du, Z.; Liu, X.; Zeng, A.; Hou, L.; Dong, Y.; Tang, J.; and Li, J. 2023. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. arXiv preprint arXiv:2308.14508.

Bi, B.; Huang, S.; Wang, Y.; Yang, T.; Zhang, Z.; Huang, H.; Mei, L.; Fang, J.; Li, Z.; Wei, F.; Deng, W.; Sun, F.; Zhang,

- Q.; and Liu, S. 2024. Context-DPO: Aligning Language Models for Context-Faithfulness. arXiv:2412.15280.

Clark, P.; Cowhey, I.; Etzioni, O.; Khot, T.; Sabharwal, A.; Schoenick, C.; and Tafjord, O. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457.

Cui, W.; Zhou, M.; Zhao, R.; and Norouzi, N. 2019. KBNLG: From Knowledge Base to Natural Language Generation. In Axelrod, A.; Yang, D.; Cunha, R.; Shaikh, S.; and Waseem, Z., eds., Proceedings of the 2019 Workshop on Widening NLP, 80–82. Florence, Italy: Association for Computational Linguistics.

DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang,

- R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; Zhang, X.; Yu,

- X.; Wu, Y.; Wu, Z. F.; Gou, Z.; Shao, Z.; Li, Z.; Gao, Z.; Liu, A.; Xue, B.; Wang, B.; Wu, B.; Feng, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Chen, D.; Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li, G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Ding, H.; Xin, H.; Gao, H.; Qu, H.; Li, H.; Guo, J.; Li, J.; Wang, J.; Chen, J.; Yuan, J.; Qiu, J.; Li, J.; Cai, J. L.; Ni, J.; Liang, J.; Chen, J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.; Zhang, L.; Zhao, L.; Wang, L.; Zhang, L.; Xu, L.; Xia, L.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Wang, M.; Li, M.; Tian, N.; Huang, P.; Zhang, P.; Wang, Q.; Chen, Q.; Du, Q.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Chen, R. J.; Jin, R. L.; Chen, R.; Lu, S.; Zhou, S.; Chen, S.; Ye, S.; Wang, S.; Yu,

S.; Zhou, S.; Pan, S.; Li, S. S.; Zhou, S.; Wu, S.; Ye, S.; Yun, T.; Pei, T.; Sun, T.; Wang, T.; Zeng, W.; Zhao, W.; Liu, W.; Liang, W.; Gao, W.; Yu, W.; Zhang, W.; Xiao, W. L.; An, W.; Liu, X.; Wang, X.; Chen, X.; Nie, X.; Cheng, X.; Liu, X.; Xie, X.; Liu, X.; Yang, X.; Li, X.; Su, X.; Lin, X.; Li, X. Q.; Jin, X.; Shen, X.; Chen, X.; Sun, X.; Wang, X.; Song, X.; Zhou, X.; Wang, X.; Shan, X.; Li, Y. K.; Wang, Y. Q.; Wei,

- Y. X.; Zhang, Y.; Xu, Y.; Li, Y.; Zhao, Y.; Sun, Y.; Wang, Y.; Yu, Y.; Zhang, Y.; Shi, Y.; Xiong, Y.; He, Y.; Piao, Y.; Wang,

DeepSeek-AI; Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Guo, D.; Yang, D.; Chen, D.; Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li, G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Zhang, H.; Ding, H.; Xin, H.; Gao, H.; Li, H.; Qu, H.; Cai, J. L.; Liang, J.; Guo, J.; Ni, J.; Li, J.; Wang, J.; Chen, J.; Chen, J.; Yuan, J.; Qiu, J.; Li, J.; Song, J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.; Zhang, L.; Xu, L.; Xia, L.; Zhao, L.; Wang, L.; Zhang, L.; Li, M.; Wang, M.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Tian, N.; Huang, P.; Wang, P.; Zhang, P.; Wang, Q.; Zhu, Q.; Chen, Q.; Du, Q.; Chen, R. J.; Jin, R. L.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Xu, R.; Zhang, R.; Chen, R.; Li, S. S.; Lu, S.; Zhou, S.; Chen, S.; Wu, S.; Ye, S.; Ye, S.; Ma, S.; Wang,

- S.; Zhou, S.; Yu, S.; Zhou, S.; Pan, S.; Wang, T.; Yun, T.; Pei,
- T.; Sun, T.; Xiao, W. L.; Zeng, W.; Zhao, W.; An, W.; Liu,

- W.; Liang, W.; Gao, W.; Yu, W.; Zhang, W.; Li, X. Q.; Jin,
- X.; Wang, X.; Bi, X.; Liu, X.; Wang, X.; Shen, X.; Chen, X.; Zhang, X.; Chen, X.; Nie, X.; Sun, X.; Wang, X.; Cheng, X.; Liu, X.; Xie, X.; Liu, X.; Yu, X.; Song, X.; Shan, X.; Zhou,

- X.; Yang, X.; Li, X.; Su, X.; Lin, X.; Li, Y. K.; Wang, Y. Q.; Wei, Y. X.; Zhu, Y. X.; Zhang, Y.; Xu, Y.; Xu, Y.; Huang,
- Y.; Li, Y.; Zhao, Y.; Sun, Y.; Li, Y.; Wang, Y.; Yu, Y.; Zheng,

- Y.; Zhang, Y.; Shi, Y.; Xiong, Y.; He, Y.; Tang, Y.; Piao, Y.; Wang, Y.; Tan, Y.; Ma, Y.; Liu, Y.; Guo, Y.; Wu, Y.; Ou, Y.; Zhu, Y.; Wang, Y.; Gong, Y.; Zou, Y.; He, Y.; Zha, Y.; Xiong,

- Y.; Ma, Y.; Yan, Y.; Luo, Y.; You, Y.; Liu, Y.; Zhou, Y.; Wu,
- Z. F.; Ren, Z. Z.; Ren, Z.; Sha, Z.; Fu, Z.; Xu, Z.; Huang, Z.; Zhang, Z.; Xie, Z.; Zhang, Z.; Hao, Z.; Gou, Z.; Ma, Z.; Yan,

- Z.; Shao, Z.; Xu, Z.; Wu, Z.; Zhang, Z.; Li, Z.; Gu, Z.; Zhu,

- Z.; Liu, Z.; Li, Z.; Xie, Z.; Song, Z.; Gao, Z.; and Pan, Z. 2025b. DeepSeek-V3 Technical Report. arXiv:2412.19437.

- Y.; Tan, Y.; Ma, Y.; Liu, Y.; Guo, Y.; Ou, Y.; Wang, Y.; Gong,

- Y.; Zou, Y.; He, Y.; Xiong, Y.; Luo, Y.; You, Y.; Liu, Y.; Zhou,

- Y.; Zhu, Y. X.; Xu, Y.; Huang, Y.; Li, Y.; Zheng, Y.; Zhu, Y.; Ma, Y.; Tang, Y.; Zha, Y.; Yan, Y.; Ren, Z. Z.; Ren, Z.; Sha,
- Z.; Fu, Z.; Xu, Z.; Xie, Z.; Zhang, Z.; Hao, Z.; Ma, Z.; Yan,

- Z.; Wu, Z.; Gu, Z.; Zhu, Z.; Liu, Z.; Li, Z.; Xie, Z.; Song, Z.; Pan, Z.; Huang, Z.; Xu, Z.; Zhang, Z.; and Zhang, Z. 2025a. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

Dong, G.; Song, X.; Zhu, Y.; Qiao, R.; Dou, Z.; and Wen, J.-R. 2024. Toward General Instruction-Following Alignment for Retrieval-Augmented Generation. arXiv:2410.09584.

Dong, X.; Li, W.; Le, Y.; Jiang, Z.; Zhong, J.; and Wang, Z. 2025. TermDiffuSum: A Term-guided Diffusion Model for Extractive Summarization of Legal Documents. In Rambow, O.; Wanner, L.; Apidianaki, M.; Al-Khalifa, H.; Eugenio, B. D.; and Schockaert, S., eds., Proceedings of the 31st International Conference on Computational Linguistics, 3222–3235. Abu Dhabi, UAE: Association for Computational Linguistics.

Du, A.; Gao, B.; Xing, B.; Jiang, C.; Chen, C.; Li, C.; Xiao, C.; Du, C.; Liao, C.; Tang, C.; Wang, C.; Zhang, D.; Yuan, E.; Lu, E.; Tang, F.; Sung, F.; Wei, G.; Lai, G.; Guo, H.; Zhu, H.; Ding, H.; Hu, H.; Yang, H.; Zhang, H.; Yao, H.; Zhao, H.; Lu, H.; Li, H.; Yu, H.; Gao, H.; Zheng, H.; Yuan, H.; Chen, J.; Guo, J.; Su, J.; Wang, J.; Zhao, J.; Zhang, J.; Liu, J.; Yan, J.; Wu, J.; Shi, L.; Ye, L.; Yu, L.; Dong, M.; Zhang, N.; Ma, N.; Pan, Q.; Gong, Q.; Liu, S.; Ma, S.; Wei, S.; Cao, S.; Huang, S.; Jiang, T.; Gao, W.; Xiong, W.; He, W.; Huang, W.; Wu, W.; He, W.; Wei, X.; Jia, X.; Wu, X.; Xu, X.; Zu, X.; Zhou, X.; Pan, X.; Charles, Y.; Li, Y.; Hu, Y.; Liu, Y.; Chen, Y.; Wang, Y.; Liu, Y.; Qin, Y.; Liu, Y.; Yang, Y.; Bao, Y.; Du, Y.; Wu, Y.; Wang, Y.; Zhou, Z.; Wang, Z.; Li, Z.; Zhu, Z.; Zhang, Z.; Wang, Z.; Yang, Z.; Huang, Z.; Huang, Z.; Xu,

- Z.; and Yang, Z. 2025. Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv:2501.12599.

Duong, S.; Bronnec, F. L.; Allauzen, A.; Guigue, V.; Lumbreras, A.; Soulier, L.; and Gallinari, P. 2025. SCOPE: A Self-supervised Framework for Improving Faithfulness in Conditional Text Generation. In The Thirteenth International Conference on Learning Representations.

Face, H. 2025. Open R1: A fully open reproduction of DeepSeek-R1.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; Yang, A.; Fan, A.; Goyal, A.; Hartshorn, A.; Yang, A.; Mitra, A.; Sravankumar, A.; Korenev, A.; Hinsvark, A.; Rao, A.; Zhang, A.; Rodriguez, A.; Gregerson, A.; Spataru, A.; Roziere, B.; Biron, B.; Tang, B.; Chern, B.; Caucheteux, C.; Nayak, C.; Bi, C.; Marra, C.; McConnell, C.; Keller, C.; Touret, C.; Wu, C.; Wong, C.; Ferrer, C. C.; Nikolaidis, C.; Allonsius, D.; Song, D.; Pintz, D.; Livshits, D.; Wyatt, D.; Esiobu, D.; Choudhary, D.; Mahajan, D.; GarciaOlano, D.; Perino, D.; Hupkes, D.; Lakomkin, E.; AlBadawy, E.; Lobanova, E.; Dinan, E.; Smith, E. M.; Radenovic, F.; Guzm´an, F.; Zhang, F.; Synnaeve, G.; Lee, G.; Anderson, G. L.; Thattai, G.; Nail, G.; Mialon, G.; Pang, G.; Cucurell,

- G.; Nguyen, H.; Korevaar, H.; Xu, H.; Touvron, H.; Zarov, I.; Ibarra, I. A.; Kloumann, I.; Misra, I.; Evtimov, I.; Zhang, J.; Copet, J.; Lee, J.; Geffert, J.; Vranes, J.; Park, J.; Mahadeokar, J.; Shah, J.; van der Linde, J.; Billock, J.; Hong, J.; Lee, J.; Fu, J.; Chi, J.; Huang, J.; Liu, J.; Wang, J.; Yu, J.; Bitton,

- J.; Spisak, J.; Park, J.; Rocca, J.; Johnstun, J.; Saxe, J.; Jia,

- J.; Alwala, K. V.; Prasad, K.; Upasani, K.; Plawiak, K.; Li,
- K.; Heafield, K.; Stone, K.; El-Arini, K.; Iyer, K.; Malik,

- K.; Chiu, K.; Bhalla, K.; Lakhotia, K.; Rantala-Yeary, L.; van der Maaten, L.; Chen, L.; Tan, L.; Jenkins, L.; Martin, L.; Madaan, L.; Malo, L.; Blecher, L.; Landzaat, L.; de Oliveira,
- L.; Muzzi, M.; Pasupuleti, M.; Singh, M.; Paluri, M.; Kardas, M.; Tsimpoukelli, M.; Oldham, M.; Rita, M.; Pavlova,
- M.; Kambadur, M.; Lewis, M.; Si, M.; Singh, M. K.; Hassan, M.; Goyal, N.; Torabi, N.; Bashlykov, N.; Bogoychev,
- N.; Chatterji, N.; Zhang, N.; Duchenne, O.; C¸elebi, O.; Alrassy, P.; Zhang, P.; Li, P.; Vasic, P.; Weng, P.; Bhargava, P.; Dubal, P.; Krishnan, P.; Koura, P. S.; Xu, P.; He, Q.; Dong, Q.; Srinivasan, R.; Ganapathy, R.; Calderer, R.; Cabral, R. S.; Stojnic, R.; Raileanu, R.; Maheswari, R.; Girdhar, R.; Patel, R.; Sauvestre, R.; Polidoro, R.; Sumbaly, R.; Taylor, R.; Silva, R.; Hou, R.; Wang, R.; Hosseini, S.; Chennabasappa, S.; Singh, S.; Bell, S.; Kim, S. S.; Edunov, S.; Nie, S.; Narang,

- S.; Raparthy, S.; Shen, S.; Wan, S.; Bhosale, S.; Zhang, S.; Vandenhende, S.; Batra, S.; Whitman, S.; Sootla, S.; Collot, S.; Gururangan, S.; Borodinsky, S.; Herman, T.; Fowler,
- T.; Sheasha, T.; Georgiou, T.; Scialom, T.; Speckbacher, T.; Mihaylov, T.; Xiao, T.; Karn, U.; Goswami, V.; Gupta, V.; Ramanathan, V.; Kerkez, V.; Gonguet, V.; Do, V.; Vogeti, V.; Albiero, V.; Petrovic, V.; Chu, W.; Xiong, W.; Fu, W.; Meers, W.; Martinet, X.; Wang, X.; Wang, X.; Tan, X. E.; Xia, X.; Xie, X.; Jia, X.; Wang, X.; Goldschlag, Y.; Gaur, Y.; Babaei,

- Y.; Wen, Y.; Song, Y.; Zhang, Y.; Li, Y.; Mao, Y.; Coudert,
- Z. D.; Yan, Z.; Chen, Z.; Papakipos, Z.; Singh, A.; Srivastava,

- A.; Jain, A.; Kelsey, A.; Shajnfeld, A.; Gangidi, A.; Victo-

ria, A.; Goldstand, A.; Menon, A.; Sharma, A.; Boesenberg, A.; Baevski, A.; Feinstein, A.; Kallet, A.; Sangani, A.; Teo,

- A.; Yunus, A.; Lupu, A.; Alvarado, A.; Caples, A.; Gu, A.; Ho, A.; Poulton, A.; Ryan, A.; Ramchandani, A.; Dong, A.; Franco, A.; Goyal, A.; Saraf, A.; Chowdhury, A.; Gabriel, A.; Bharambe, A.; Eisenman, A.; Yazdan, A.; James, B.; Maurer,
- B.; Leonhardi, B.; Huang, B.; Loyd, B.; Paola, B. D.; Paranjape, B.; Liu, B.; Wu, B.; Ni, B.; Hancock, B.; Wasti, B.; Spence, B.; Stojkovic, B.; Gamido, B.; Montalvo, B.; Parker,
- C.; Burton, C.; Mejia, C.; Liu, C.; Wang, C.; Kim, C.; Zhou, C.; Hu, C.; Chu, C.-H.; Cai, C.; Tindal, C.; Feichtenhofer, C.; Gao, C.; Civin, D.; Beaty, D.; Kreymer, D.; Li, D.; Adkins,
- D.; Xu, D.; Testuggine, D.; David, D.; Parikh, D.; Liskovich,

- D.; Foss, D.; Wang, D.; Le, D.; Holland, D.; Dowling, E.; Jamil, E.; Montgomery, E.; Presani, E.; Hahn, E.; Wood, E.; Le, E.-T.; Brinkman, E.; Arcaute, E.; Dunbar, E.; Smothers,
- E.; Sun, F.; Kreuk, F.; Tian, F.; Kokkinos, F.; Ozgenel, F.; Caggioni, F.; Kanayet, F.; Seide, F.; Florez, G. M.; Schwarz,

- G.; Badeer, G.; Swee, G.; Halpern, G.; Herman, G.; Sizov, G.; Guangyi; Zhang; Lakshminarayanan, G.; Inan, H.; Shojanazeri, H.; Zou, H.; Wang, H.; Zha, H.; Habeeb, H.; Rudolph,
- H.; Suk, H.; Aspegren, H.; Goldman, H.; Zhan, H.; Damlaj,
- I.; Molybog, I.; Tufanov, I.; Leontiadis, I.; Veliche, I.-E.; Gat,

- I.; Weissman, J.; Geboski, J.; Kohli, J.; Lam, J.; Asher, J.; Gaya, J.-B.; Marcus, J.; Tang, J.; Chan, J.; Zhen, J.; Reizenstein, J.; Teboul, J.; Zhong, J.; Jin, J.; Yang, J.; Cummings,
- J.; Carvill, J.; Shepard, J.; McPhie, J.; Torres, J.; Ginsburg,

J.; Wang, J.; Wu, K.; U, K. H.; Saxena, K.; Khandelwal, K.; Zand, K.; Matosich, K.; Veeraraghavan, K.; Michelena,

- K.; Li, K.; Jagadeesh, K.; Huang, K.; Chawla, K.; Huang, K.; Chen, L.; Garg, L.; A, L.; Silva, L.; Bell, L.; Zhang, L.; Guo, L.; Yu, L.; Moshkovich, L.; Wehrstedt, L.; Khabsa, M.; Avalani, M.; Bhatt, M.; Mankus, M.; Hasson, M.; Lennie, M.; Reso, M.; Groshev, M.; Naumov, M.; Lathi, M.; Keneally, M.; Liu, M.; Seltzer, M. L.; Valko, M.; Restrepo, M.; Patel, M.; Vyatskov, M.; Samvelyan, M.; Clark, M.; Macey, M.; Wang, M.; Hermoso, M. J.; Metanat, M.; Rastegari, M.; Bansal, M.; Santhanam, N.; Parks, N.; White, N.; Bawa, N.; Singhal, N.; Egebo, N.; Usunier, N.; Mehta, N.; Laptev, N. P.; Dong, N.; Cheng, N.; Chernoguz, O.; Hart, O.; Salpekar, O.; Kalinli, O.; Kent, P.; Parekh, P.; Saab, P.; Balaji, P.; Rittner, P.; Bontrager, P.; Roux, P.; Dollar, P.; Zvyagina, P.; Ratanchandani, P.; Yuvraj, P.; Liang, Q.; Alao, R.; Rodriguez, R.; Ayub,

- R.; Murthy, R.; Nayani, R.; Mitra, R.; Parthasarathy, R.; Li,

- R.; Hogan, R.; Battey, R.; Wang, R.; Howes, R.; Rinott, R.; Mehta, S.; Siby, S.; Bondu, S. J.; Datta, S.; Chugh, S.; Hunt,
- S.; Dhillon, S.; Sidorov, S.; Pan, S.; Mahajan, S.; Verma,

- S.; Yamamoto, S.; Ramaswamy, S.; Lindsay, S.; Lindsay, S.; Feng, S.; Lin, S.; Zha, S. C.; Patil, S.; Shankar, S.; Zhang, S.; Zhang, S.; Wang, S.; Agarwal, S.; Sajuyigbe, S.; Chintala, S.; Max, S.; Chen, S.; Kehoe, S.; Satterfield, S.; Govindaprasad, S.; Gupta, S.; Deng, S.; Cho, S.; Virk, S.; Subramanian, S.; Choudhury, S.; Goldman, S.; Remez, T.; Glaser, T.; Best, T.; Koehler, T.; Robinson, T.; Li, T.; Zhang, T.; Matthews, T.; Chou, T.; Shaked, T.; Vontimitta, V.; Ajayi, V.; Montanez, V.; Mohan, V.; Kumar, V. S.; Mangla, V.; Ionescu, V.; Poenaru,

- V.; Mihailescu, V. T.; Ivanov, V.; Li, W.; Wang, W.; Jiang,
- W.; Bouaziz, W.; Constable, W.; Tang, X.; Wu, X.; Wang, X.; Wu, X.; Gao, X.; Kleinman, Y.; Chen, Y.; Hu, Y.; Jia, Y.; Qi,

- Y.; Li, Y.; Zhang, Y.; Zhang, Y.; Adi, Y.; Nam, Y.; Yu; Wang; Zhao, Y.; Hao, Y.; Qian, Y.; Li, Y.; He, Y.; Rait, Z.; DeVito,
- Z.; Rosnbrick, Z.; Wen, Z.; Yang, Z.; Zhao, Z.; and Ma, Z.

2024. The Llama 3 Herd of Models. arXiv:2407.21783.

Guo, S.; Zhang, J.; Ke, X.; Li, C.; and Chen, H. 2024. Diversifying Question Generation over Knowledge Base via External Natural Questions. In Calzolari, N.; Kan, M.-Y.; Hoste, V.; Lenci, A.; Sakti, S.; and Xue, N., eds., Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 5096–5108. Torino, Italia: ELRA and ICCL.

He, W.; Liu, K.; Liu, J.; Lyu, Y.; Zhao, S.; Xiao, X.; Liu, Y.; Wang, Y.; Wu, H.; She, Q.; Liu, X.; Wu, T.; and Wang, H. 2018. DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications. In Choi, E.; Seo, M.; Chen, D.; Jia, R.; and Berant, J., eds., Proceedings of the Workshop on Machine Reading for Question Answering, 37–46. Melbourne, Australia: Association for Computational Linguistics.

Hu, M.; He, B.; Wang, Y.; Li, L.; Ma, C.; and King, I. 2024. Mitigating Large Language Model Hallucination with Faithful Finetuning. arXiv:2406.11267.

Huang, L.; Feng, X.; Ma, W.; Fan, Y.; Feng, X.; Ye, Y.; Zhong, W.; Gu, Y.; Wang, B.; Wu, D.; Hu, G.; and Qin,

- B. 2025. Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization. arXiv:2501.13573.

Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.; Chen, Q.; Peng, W.; Feng, X.; Qin, B.; and Liu, T. 2024. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Trans. Inf. Syst. Just Accepted.

Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky, A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney, A.; et al. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y.; Ishii,

- E.; Bang, Y. J.; Madotto, A.; and Fung, P. 2023. Survey of Hallucination in Natural Language Generation. ACM Comput. Surv., 55(12).

Jones, E.; Palangi, H.; Ribeiro, C. S.; Chandrasekaran, V.; Mukherjee, S.; Mitra, A.; Awadallah, A. H.; and Kamar, E. 2024. Teaching Language Models to Hallucinate Less with Synthetic Tasks. In The Twelfth International Conference on Learning Representations.

Joshi, M.; Choi, E.; Weld, D.; and Zettlemoyer, L. 2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In Barzilay, R.; and Kan, M.-Y., eds., Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1601–1611. Vancouver, Canada: Association for Computational Linguistics.

Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,

- J.; Lee, K.; Toutanova, K.; Jones, L.; Kelcey, M.; Chang, M.-

- W.; Dai, A. M.; Uszkoreit, J.; Le, Q.; and Petrov, S. 2019a.

Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics, 7: 452–466.

Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Kelcey, M.; Devlin, J.; Lee, K.; Toutanova, K. N.; Jones, L.; Chang, M.-W.; Dai, A.; Uszkoreit, J.; Le, Q.; and Petrov, S. 2019b. Natural Questions: a Benchmark for Question Answering Research. Transactions of the Association of Computational Linguistics.

- Li, J.; and Ng, H. T. 2025. The Hallucination Dilemma: Factuality-Aware Reinforcement Learning for Large Reasoning Models. arXiv:2505.24630.
- Li, K.; Zhang, T.; Li, Y.; Luo, H.; Moustafa, A.; Wu, X.; Glass, J.; and Meng, H. 2025. Generate, Discriminate, Evolve: Enhancing Context Faithfulness via Fine-Grained Sentence-Level Self-Evolution. arXiv:2503.01695. Li, T.; Li, Z.; and Zhang, Y. 2024. Improving Faithfulness of Large Language Models in Summarization via Sliding Generation and Self-Consistency. In Calzolari, N.; Kan, M.-Y.; Hoste, V.; Lenci, A.; Sakti, S.; and Xue, N., eds., Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 8804–8817. Torino, Italia: ELRA and ICCL. Li, W.; Wu, W.; Chen, M.; Liu, J.; Xiao, X.; and Wu, H. 2022. Faithfulness in Natural Language Generation: A Systematic Survey of Analysis, Evaluation and Optimization Methods. arXiv:2203.05227. Longpre, S.; Perisetla, K.; Chen, A.; Ramesh, N.; DuBois, C.; and Singh, S. 2021. Entity-Based Knowledge Conflicts in Question Answering. In Moens, M.-F.; Huang, X.; Specia, L.; and Yih, S. W.-t., eds., Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 7052–7063. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics. Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. arXiv:1711.05101. Ming, Y.; Purushwalkam, S.; Pandit, S.; Ke, Z.; Nguyen, X.-P.; Xiong, C.; and Joty, S. 2025. FaithEval: Can Your Language Model Stay Faithful to Context, Even If ”The Moon is Made of Marshmallows”. In The Thirteenth International Conference on Learning Representations. Narayan, S.; Cohen, S. B.; and Lapata, M. 2018. Don‘t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization. In Riloff, E.; Chiang, D.; Hockenmaier, J.; and Tsujii, J., eds., Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 1797–1807. Brussels, Belgium: Association for Computational Linguistics. OpenAI. 2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774. OpenAI. 2025. Deep Research System Card. Technical report, OpenAI. Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Ermon, S.; and Finn, C. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In

Thirty-seventh Conference on Neural Information Processing Systems.

Ravichander, A.; Ghela, S.; Wadden, D.; and Choi, Y. 2025. HALoGEN: Fantastic LLM Hallucinations and Where to Find Them. arXiv:2501.08292.

Rosenthal, S.; Sil, A.; Florian, R.; and Roukos, S. 2025. CLAPnq: Cohesive Long-form Answers from Passages in Natural Questions for RAG systems. Transactions of the Association for Computational Linguistics, 13: 53–72.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Shi, W.; Han, X.; Lewis, M.; Tsvetkov, Y.; Zettlemoyer, L.; and Yih, W.-t. 2024. Trusting Your Evidence: Hallucinate Less with Context-aware Decoding. In Duh, K.; Gomez, H.; and Bethard, S., eds., Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), 783–791. Mexico City, Mexico: Association for Computational Linguistics.

Si, S.; Zhao, H.; Chen, G.; Gao, C.; Bai, Y.; Wang, Z.; An,

- K.; Luo, K.; Qian, C.; Qi, F.; Chang, B.; and Sun, M. 2025a. Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering. In Che, W.; Nabende, J.; Shutova, E.; and Pilehvar, M. T., eds., Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 16469–

16488. Vienna, Austria: Association for Computational Linguistics. ISBN 979-8-89176-251-0.

Si, S.; Zhao, H.; Chen, G.; Li, Y.; Luo, K.; Lv, C.; An, K.; Qi, F.; Chang, B.; and Sun, M. 2025b. GATEAU: Selecting Influential Samples for Long Context Alignment. In Christodoulopoulos, C.; Chakraborty, T.; Rose, C.; and Peng, V., eds., Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 7391–7422. Suzhou, China: Association for Computational Linguistics. ISBN 979-8-89176-332-6.

Tang, L.; Laban, P.; and Durrett, G. 2024. MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

Vrandeˇci´c, D.; and Kr¨otzsch, M. 2014. Wikidata: a free collaborative knowledgebase. Commun. ACM, 57(10): 78–85.

Wang, H.; Prasad, A.; Stengel-Eskin, E.; and Bansal, M. 2025. Retrieval-Augmented Generation with Conflicting Evidence. arXiv:2504.13079.

Wang, M.; Chen, L.; Cheng, F.; Liao, S.; Zhang, X.; Wu, B.; Yu, H.; Xu, N.; Zhang, L.; Luo, R.; Li, Y.; Yang, M.; Huang,

- F.; and Li, Y. 2024. Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA. In Al-Onaizan, Y.; Bansal, M.; and Chen, Y.-N., eds., Proceedings of the 2024 Conference on Empirical Methods in

Natural Language Processing, 5627–5646. Miami, Florida, USA: Association for Computational Linguistics.

Wei, J.; Karina, N.; Chung, H. W.; Jiao, Y. J.; Papay, S.; Glaese, A.; Schulman, J.; and Fedus, W. 2024. Measuring short-form factuality in large language models. arXiv:2411.04368.

Wu, H.; Zhan, M.; Tan, H.; Hou, Z.; Liang, D.; and Song, L. 2023. VCSUM: A Versatile Chinese Meeting Summarization Dataset. In Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds., Findings of the Association for Computational Linguistics: ACL 2023, 6065–6079. Toronto, Canada: Association for Computational Linguistics.

Xie, J.; Zhang, K.; Chen, J.; Lou, R.; and Su, Y. 2024. Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts. In The Twelfth International Conference on Learning Representations.

Xu, F.; Shi, W.; and Choi, E. 2024. RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation. In The Twelfth International Conference on Learning Representations.

Xu, R.; Qi, Z.; Guo, Z.; Wang, C.; Wang, H.; Zhang, Y.; and Xu, W. 2024. Knowledge Conflicts for LLMs: A Survey. In Al-Onaizan, Y.; Bansal, M.; and Chen, Y.-N., eds., Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 8541–8565. Miami, Florida, USA: Association for Computational Linguistics.

Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu, K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.; Zhu, Q.; Men, R.; Lin, R.; Li, T.; Xia, T.; Ren, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Cui, Z.; Zhang, Z.; and Qiu, Z. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.

Yang, C.; Lin, X.; Xu, C.; Jiang, X.; Ma, S.; Liu, A.; Xiong, H.; and Guo, J. 2025. LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data. In Che, W.; Nabende, J.; Shutova, E.; and Pilehvar, M. T., eds., Findings of the Association for Computational Linguistics: ACL 2025, 3236–3256. Vienna, Austria: Association for Computational Linguistics. ISBN 979-8-89176-256-5.

Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W.; Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Riloff, E.; Chiang, D.; Hockenmaier, J.; and Tsujii, J., eds., Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2369–2380. Brussels, Belgium: Association for Computational Linguistics.

Yih, W.; Richardson, M.; Meek, C.; Chang, M.-W.; and Suh, J. 2016. The Value of Semantic Parse Labeling for Knowledge Base Question Answering. In Annual Meeting of the Association for Computational Linguistics.

Zhang, H.; Yu, P. S.; and Zhang, J. 2024. A Systematic Survey of Text Summarization: From Statistical Methods to Large Language Models. arXiv:2406.11289.

Zhang, J.; Hou, Z.; Lv, X.; Cao, S.; Hou, Z.; Niu, Y.; Hou, L.; Dong, Y.; Feng, L.; and Li, J. 2025. LongReward: Improving Long-context Large Language Models with AI Feedback. In Che, W.; Nabende, J.; Shutova, E.; and Pilehvar, M. T., eds., Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 3718–3739. Vienna, Austria: Association for Computational Linguistics. ISBN 979-8-89176-251-0.

Zhang, X.; and Lapata, M. 2017. Sentence Simplification with Deep Reinforcement Learning. In Palmer, M.; Hwa, R.; and Riedel, S., eds., Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 584– 594. Copenhagen, Denmark: Association for Computational Linguistics.

Zhou, W.; Zhang, S.; Poon, H.; and Chen, M. 2023. Contextfaithful Prompting for Large Language Models. In Bouamor,

- H.; Pino, J.; and Bali, K., eds., Findings of the Association for Computational Linguistics: EMNLP 2023, 14544–14556. Singapore: Association for Computational Linguistics. Zhu, C.; Hinthorn, W.; Xu, R.; Zeng, Q.; Zeng, M.; Huang,

- X.; and Jiang, M. 2021. Enhancing Factual Consistency of Abstractive Summarization. In Toutanova, K.; Rumshisky, A.; Zettlemoyer, L.; Hakkani-Tur, D.; Beltagy, I.; Bethard, S.; Cotterell, R.; Chakraborty, T.; and Zhou, Y., eds., Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 718–733. Online: Association for Computational Linguistics.

Zhu, R.; Qi, J.; and Lau, J. H. 2023. Annotating and Detecting Fine-grained Factual Errors for Dialogue Summarization. In Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds., Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 6825–6845. Toronto, Canada: Association for Computational Linguistics.

### Appendix

This appendix is organized as follows.

- • In Section A Training Data Details, we report the details of constructing training data, e.g., the used triples and introduction of four designed tasks.
- • In Section B Dual-GRPO Details, we go into detail about the proposed Dual-GRPO, including the system prompt and formal expressions of three well-designed rewards.
- • In Section C Evaluation Details, we show the details of evaluations, e.g., the introduction of the used benchmarks and evaluation prompts.
- • In Section D Implementations Details, we show the details of our implementation and training, e.g., hyperparameters.
- • In Section E Human Evaluation, we show the implementation details of human evaluation.
- • In Section F Discussion, we discuss some possible questions about the proposed CANOE. For example, we discuss the effect of the amount of synthesized data for training.

### A Training Data Details

#### A.1 Triples from Wikidata

To ensure the usability of the synthetic data and collected triples, we follow Bi et al. (2024) to collect entities corresponding to the top 1,000 most-visited Wikipedia pages from 2016 to 2023 and 41 relations selected by Bi et al. (2024) shown in Table 13. The most-visited Wikipedia pages are based on monthly page views and retain the most popular entities using criteria such as the number of hyperlinks. We finally collected 6,316 entities and 30,762 triples. We randomly select these triples to synthesize our training data, and finally construct 10,000 samples as the final training data.

#### A.2 Construction of Four Different Tasks

We design four different tasks to enhance the complexity and diversity of our training data. Meanwhile, we select GPT-4o2024-08-06 to construct the contexts and questions.

Straightforward Context. As shown in Sec. 2.1, we keep the original collected factual triple as input to query GPT-4o to synthesize the data (c,q,a). The prompts for querying GPT-4o to obtain the generated questions and contexts can be found in Figure 7 and Figure 8. We keep 2,000 such samples in the synthesized 10,000 training data, i.e., 20% of the data. Reasoning-required Context. We construct paths [(h1,r1,t1),...,(hn,rn,tn)]n≤4 from a sub-graph; more details can be found in Sec. 2.1. Then, we use the n-th tail entity tn as the ground truth answer and use the constructed paths to query GPT-4o to obtain the multi-hop context and question. The prompts for querying GPT-4o to obtain the generated questions and contexts can be found in Figure 9 and Figure 10. We finally keep 2,000 such samples in the synthesized 10,000 training data, i.e., 20% of the data.

Inconsistent Context. This involves multiple randomly ordered contexts generated from different triples. This simulates noisy and inconsistent scenarios, where models need to detect inconsistencies and focus on useful and relevant contexts to answer the questions. We construct such a sample

Type Num Avg Len

Straightforward Context. 2,000 186.3 Reasoning-required Context. 2,000 262.2 Inconsistent Context. 1,000 421.2 Counterfactual Context. 5,000 260.8

Table 7: Statistics of the training data. Num shows the number of samples. Avg Len shows the average length of the samples.

by combining the contexts from up to three QA samples with reasoning-required context and use the original tn as the answer. In this way, we can obtain more complex samples than ones with the reasoning-required context. To avoid duplicating the 2,000 samples with the reasoning-required context collected above, we reconstruct the new samples with the reasoning-required context used to obtain the samples with the inconsistent context. We keep 1,000 such samples in the synthesized 10,000 training data, i.e., 10% of the data.

Counterfactual Context. A counterfactual context includes statements that go against common sense found in the collected triples. Specifically, we construct samples with counterfactual contexts below by modifying previously collected triples (of three types, including straightforward context, reasoning-required context, and inconsistent context). We replace the tail entity t of the original collected triple with a similar but counterfactual entity tcf, which is obtained by query GPT-4o using prompt “Generate me a noun for an entity that is similar to the {t} but different, and require the entity to exist in the real-world, please tell me the answer directly:”. Then, we query GPT-4o to generate questions and counterfactual contexts to construct counterfactual samples, using the counterfactual triples. The prompts used in constructing samples with counterfactual contexts are the same as the prompts used in constructing the three different tasks above. The reason we construct samples with counterfactual context in this way is that this prevents the model from learning the appropriate factual knowledge to answer the question correctly, rather than correctly exploiting the given contextual information. Therefore, we construct the same number of samples as the summed number of the three types above (including straightforward context, reasoning-required context, and inconsistent context), i.e., 5000 samples (50% of the data). This task stresses the importance of keeping answers faithful in contexts, as it stops them from relying solely on the learned knowledge of LLMs to provide correct answers.

#### A.3 Statistics

We show the statistics of the training data in Table 7. Even though the length of the data we synthesize is short, we find that our model can be generalized with consistently state-ofthe-art results on a wide range of tasks with different input lengths by utilizing our proposed Dual-GRPO, e.g., longform QA and RAG generation with long texts as inputs.

### B Dual-GRPO Details

In this section, we give a more detailed introduction to our proposed Dual-GRPO, including the designed system prompt

and formal expressions of three different rewards.

System Prompt. For the provided contextual information and question, Dual-GRPO employs the designed system prompt that requires LLMs to produce a reasoning process, then a long-form answer that consists of detailed and complete sentences, and finally a concise short-form answer in just a few words. In this way, we can assign different reward scores to long-form answers and short-form answers while optimizing them both at once. Meanwhile, this system prompt also triggers zero-shot chain-of-thought reasoning in the policy model, which progressively improves as training advances to optimize for the reward. We use the same system prompt to train both LLaMA and Qwen models. We show our used system prompt in Figure 11.

Accuracy Reward. For short-form generation, we directly assign the accuracy reward. Specifically, for the generated short-form response ysf based on the given context c and question q, which is extracted from the whole generated response ywhole via string matching, and the ground truth answer ygt from the synthesized training data, the accuracy reward Racc for the LLM θ can be calculated as:

Racc =

1 if ysf(c,q|θ) = ygt, 0 otherwise.

We use the exact matching (EM) to measure accuracy, giving a score of 1 for a match and 0 for a mismatch. Thus, we can ensure that the generated short-form response correctly answers the question based on the given context, making LLMs more faithful in short-form response generation.

Proxy Reward. Instead of directly evaluating the faithfulness of the generated long-form response, we propose a proxy reward to evaluate it implicitly. For each generated long-form answer ylf, we replace the given context c with it as new input and infer the LLM θ to determine whether the LLM can produce the correct short-form answer ysf based on ylf for the question q. Proxy reward Rproxy can be calculated as:

Rproxy =

1 if ysf(ylf,q|θ) = ygt, 0 otherwise.

If the generated long-form response can help LLMs generate the correct answer, it indicates that the long-form response is faithful to the context, contains meaningful sentences, and correctly addresses the question. Thus, we assign a reward score of 1 for the positive long-form response that helps the LLM to produce the correct answer, and a reward score of 0 for those that lead to incorrect answers.

Format Reward. To enforce the desired output format, we assign a reward on the whole generated response ywhole to evaluate whether it contains the proper XML tags. We use three types of tags as shown in our system prompt, as shown in Figure 11, including <think>, <long answer>, and <short answer>tags. Formally,

Rformat =

1 if correct formatting is present, 0 if incorrect formatting.

We use the string matching method to evaluate whether the responses adhere to the format.

Final Reward. Finally, we use the sum of these three rewards as the final composite reward Rfinal. This well-designed reward Rfinal of Dual-GRPO enhances the efficacy of the rulebased RL training framework to guide the model toward generating more faithful responses in both short-form and long-form tasks. Formally,

Rfinal = Racc + Rproxy + Rformat.

Finally, we use this reward Rfinal to compute an advantage Ai for each output, guiding policy updates according to the rule-based GRPO objective.

Potential Reward Hacking Concerns. In the early experiments, we have also tried adding the length reward for longform responses (i.e., the content between <long answer>and </long answer>tags) to avoid the potential reward hacking, e.g., avoiding the policy model directly copying the given context as the long-form response, but found that the task performance does not have a significant difference.

### C Evaluation Details

#### C.1 Datasets

ConFiQA (Counterfactual QA). This is a dataset that incorporates knowledge conflicts through counterfactual passages to evaluate the faithfulness of LLMs on short-form generation. ConFiQA consists of three tasks: QA (Question Answering), MR (Multi-hop Reasoning), and MC (MultiConflicts). QA features single-hop question-answering tasks with context containing one corresponding counterfactual, while MR and MC involve multi-hop reasoning tasks with context containing one and multiple related counterfactual contexts, respectively. ConFiQA contains 1,500 data points used for testing (500/500/500 from QA/MC/MR).

CNQ (Counterfactual QA). CNQ is constructed based on Natural Questions (Kwiatkowski et al. 2019a). In CNQ, the context is modified to support counterfactual answers following (Longpre et al. 2021). It contains 2,773 samples that incorporate counterfactual passages to evaluate the faithfulness of LLMs on short-form generation.

FaithEval (Counterfactual Multiple-choice QA). FaithEval is a novel and comprehensive benchmark tailored to evaluate the faithfulness of LLMs in contextual scenarios across three diverse tasks: unanswerable, inconsistent, and counterfactual contexts. We select the counterfactual task to evaluate the faithfulness, which contains 1,000 multiple-choice QA samples curated based on ARC-Challenge (Clark et al. 2018). FiQA (Factual QA). FiQA is a factual version of ConFiQA, which shares the same questions as ConFiQA but contains the factual contexts and answers. The contexts and answers are provided by Bi et al. (2024), thus we can evaluate the faithfulness of LLMs in factual short-form response generation. It contains 1,500 samples for evaluation.

FollowRAG (RAG Scenarios for short-form QA). FollowRAG aims to assess the model’s ability to follow user instructions in complex multi-document contexts. It consists of four well-known open-domain QA datasets for RAG scenarios, including NaturalQA, TriviaQA, HotpotQA, and WebQSP. We utilize the provided passages in FollowRAG as context and original query (instead of the version with added

instruction constraints proposed by Dong et al. (2024)) as questions. We also use the original answers to report the results. FollowRAG contains 2,800 samples used for testing (700/700/700/700 from NaturalQA/TriviaQA/HotpotQA/WebQSP). Different from short-form generation tasks that the contexts always contain answers, in real-world RAG scenarios, the answer may not appear in the retrieved passages, and these passages tend to be noisy.

XSum (Summarization). Text summarization is a contentgrounded task where a model is provided a piece of text and tasked with synthesizing the most salient information within that text. XSum is a widely used dataset for text summarization, which consists of about 220,000 BBC articles as input documents. To facilitate our evaluation, we use the first 1,000 data points from the test set to evaluate our method.

WikiLarge (Simplification). Simplification is a task where a model is provided a piece of text and is tasked with paraphrasing it to make the text easier to read and understand. We use 1k instances sampled from the WikiLarge dataset as a test set, following Ravichander et al. (2025).

CLAPNQ (Long-form QA). CLAPNQ is a grounded longform QA benchmark dataset for Retrieval Augmented Generation of LLMs. The answers are typically long, 2-3 sentences grounded on a single gold passage, in contrast to datasets based on machine reading comprehension, such as short-form Natural Questions, which are just a few words. CLAPNQ includes long answers with grounded gold passages from Natural Questions. We utilize the provided passages and questions from the dev set to evaluate the faithfulness of LLMs in long-form response generation for open-domain questions, which contains 600 data points.

#### C.2 Metrics and LLM-as-a-Judge

Metrics for Short-form Generation Tasks. We evaluate performance based on whether gold answers are included in the generated responses (i.e., Acc) following Asai et al. (2024) and exact matching (EM) for QA tasks. For multiplechoice questions in FaithEval, we use keyword matching to verify the accuracy, i.e., Acc.

Metrics for Long-form Generation Tasks. To evaluate the faithfulness in long-form generation tasks, we use MiniCheck to check whether the model response is grounded in the provided context. MiniCheck is the SOTA method to recognize if LLM output can be grounded in given contexts. We select the MiniCheck-FT52 because it is the best fact-checking model, outperforming GPT-4o in evaluating the faithfulness. If the model response contains at least one statement that cannot be inferred from the context, we consider it as a negative response; otherwise, it is a positive response. To evaluate the quality of the generated long-form responses for three different tasks (QualityScore), including summarization, simplification, and long-form QA, we design different prompts to query GPT-4o-2024-11-20 as a judge to get the quality scores. We report the average results of the quality score results by querying GPT-4o twice. The prompts for three tasks can be found in Figure 12, Figure 13, and Figure 14.

2https://huggingface.co/lytang/MiniCheck-Flan-T5-Large

#### C.3 Baselines

For SOTA LLMs, we select the following versions of these models to report the results. Specifically, we use GPT-4o2024-08-06 for GPT-4o, GPT-4o-mini-2024-07-18 for GPT4o-mini, Claude 3.7 Sonnet-2025-02-19 for Claude 3.7 Sonnet and Claude 3.7 Sonnet-thinking, Deepseek R1 2025-01-20 for Deepseek R1, Deepseek V3 2024-12-26 for Deepseek V3, and o1-2024-12-17 for OpenAI o1. To get stable experimental results, we query these models twice and report the average results on each task. For the methods that are designed for improving the faithfulness, we reproduce their released code based on LLaMA-3-Instruct and Qwen-2.5-Instruct. For SCOPE, we train it on the 10,000 sampled training set of the summarization task XSum as SCOPEsum, which keeps the same number of data we used for training CANOE and provides a fair comparison.

#### C.4 More Detailed Experimental Results

FollowRAG contains four different QA datasets in RAG scenarios. We report the average results in Table 1. We show the more detailed results of FollowRAG in Table 8.

##### C.5 Test-time Prompts For baselines, the prompts for different tasks can be found in

- Figure 15, Figure 16, Figure 17, Figure 18, and Figure 19. It is worth noting that some test-time prompt templates differ from those in the original paper, which may lead to some variation in the reported results. For example, in the original paper of Context-DPO (Bi et al. 2024), the authors used a very simple prompt template, i.e., {Passages} Q: {} A:, to report performance results of popular LLMs on the ConFiQA dataset. Such a simple prompt template will lead to a very low EM score when the grounded-truth answer is just an entity, which may lead to unfair comparison and useless evaluation. Therefore, we redesign the corresponding prompt templates by adding length constraints as shown in Figure 15 to ensure a fair comparison. Meanwhile, since the authors of FaithEval (Ming et al. 2025) do not provide official evaluation scripts and a specific task prompt for FaithEval-Counterfactual, we write our evaluation scripts and use the prompt shown in
- Figure 16 in our experiments. We find that the state-of-theart LLMs, e.g., GPT-4o, show stable performance consistent with the original papers. At the same time, we observe that slightly smaller models, such as LLaMA-3-Instruct-8B, show decreased performance on our designed prompt templates. To evaluate the factuality of LLMs, we modify the original FaithEval and make it a closed-book QA setting, and use the prompts shown in Figure 20.

During the evaluation for CANOE, we apply the same system prompt during the Dual-GRPO training, and extract the content between <short answer>and </short answer>tags as the final answers for short-form generation tasks. Also, for long-form generation tasks, we extract the content between <long answer>and </long answer>tags as the final answers. We also find that the long-form responses generated by CANOE can provide correct answers in short-form generation tasks in the Appendix F. Thus, for real-world applications, we recommend using the generated long-form

HotpotQA NaturalQA TriviaQA WebQSP EM Acc EM Acc EM Acc EM Acc The state-of-the-art LLMs GPT-4o 24.7 32.0 37.0 55.0 62.3 72.3 44.9 71.7 GPT-4o mini 18.0 26.2 35.0 48.2 59.5 65.5 41.4 65.3 DeepSeek V3 18.7 27.7 34.9 54.3 60.0 70.0 37.1 68.9 Claude 3.7 Sonnet 15.3 24.1 33.6 53.9 62.5 72.5 33.7 64.3 OpenAI o1 27.0 34.0 37.0 50.0 63.0 76.0 35.0 68.0 DeepSeek R1 26.0 29.3 38.7 52.9 68.0 73.0 38.9 71.3 Claude 3.7 Sonnet-Thinking 20.1 30.2 35.6 53.0 63.4 72.0 36.0 66.0 LLaMA-3-Instruct Series LLaMA-3-Instruct-8B 13.0 18.2 31.0 40.3 45.5 60.2 35.0 60.4 LLaMA-3-Instruct-70B 24.1 28.7 36.5 45.3 63.0 66.6 31.3 42.1 SFT-8B 3.7 5.4 15.9 18.7 26.6 26.3 30.4 33.6 Context-DPO-8B 10.1 16.7 23.4 37.8 53.3 62.3 32.8 58.3 SCOPEsum-8B 12.0 20.5 25.7 42.5 46.4 58.6 36.1 63.2 CANOE-LLaMA-8B 21.4 23.3 37.4 46.9 60.0 67.3 44.9 69.3 Qwen-2.5-Instruct Series Qwen-2.5-Instruct-7B 14.0 17.6 32.2 42.3 50.3 62.3 33.9 58.8 Qwen-2.5-Instruct-14B 17.5 21.7 29.3 48.0 55.6 69.3 36.9 65.7 Qwen-2.5-Instruct-32B 16.5 24.6 26.3 50.2 50.0 70.7 42.7 66.7 Qwen-2.5-Instruct-72B 21.8 28.0 34.5 51.0 61.8 73.0 35.7 70.6 SFT-7B 16.2 18.3 26.5 30.2 43.2 58.2 30.2 60.2 Context-DPO-7B 13.0 17.2 25.2 40.2 50.1 63.2 35.7 54.3 SCOPEsum-7B 12.5 19.5 27.2 43.5 48.4 60.1 34.2 60.7 CANOE-Qwen-7B 18.0 22.6 35.7 47.4 57.4 65.7 36.9 65.0 CANOE-Qwen-14B 19.9 25.7 41.9 51.6 63.3 71.7 59.4 69.3

Model

- Table 8: Experimental results (%) on FollowRAG. Bold numbers indicate the best performance of models with the same size.

Ours Wins Tie Initial Wins

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- Figure 5: Human evaluation across four key dimensions.

responses as the system responses for the user’s instructions, because these long-form responses can not only faithfully complete long-form generation tasks, but also provide correct answers in short-form generation tasks.

### D Implementation Details

We implement our method based on the RL framework openr1 (Face 2025). We use AdamW optimizer (Loshchilov and Hutter 2019), with a 1 × 10−6 learning rate, a batch size of

14 for 7B/8B models, and a batch size of 7 for the 14B model, steering the training across two epochs. We set the maximum input length for the models to 1,024 and the maximum generation length to 1,024. The number of generations G during the RL training is set to 7, which is used in Eq. (1). We set 0.04 for β used in Eq. (1). We set 0.2 for ϵ used for the clip shown in Eq. (2). We set 0.9 for temperature in RL training to generate responses. We conduct our experiments on NVIDIA A800-80G GPUs with DeepSpeed+ZeRO2 for 7B/8B models, DeepSpeed+ZeRO2+Offloading for the 14B model, and BF16. During the inference, we set 0.7 for temperature for the evaluation of our models and baselines. For each task, we infer twice and report the average scores as final results.

### E Human Evaluation

Evaluating long-form generation tasks remains challenging. Thus, we conduct a human evaluation on the 90 samples from long-form generation tasks, including 30/30/30 for summarization/simplification/long-form QA tasks. We evaluate these samples across four key dimensions: readability, faithfulness, helpfulness, and naturalness. For each comparison, three options are given (Ours Wins, Tie, and Initial Model Wins), and the majority voting determines the final result. The participants follow the principles in Figure 21 to make the decision. We invite three Ph.D. students to compare the responses generated by the models. Before participants begin to make judgments, we describe the principles of our

ConFiQA FiQA CNQ FaithEval HotpotQA NaturalQA TriviaQA WebQSP

Model

Avg Acc Acc Acc Acc Acc Acc Acc Acc

The state-of-the-art LLMs GPT-4o 42.7 79.6 55.9 47.5 32.0 55.0 72.3 71.7 57.1 GPT-4o mini 63.7 78.8 54.3 50.9 26.2 48.2 65.5 65.3 56.6 DeepSeek V3 58.6 76.5 67.3 51.0 27.7 54.3 70.0 68.9 59.3 Claude 3.7 Sonnet 36.0 72.2 65.0 45.6 24.1 53.9 72.5 64.3 54.2 OpenAI o1 57.9 89.7 39.1 52.0 34.0 50.0 76.0 68.0 58.3 DeepSeek R1 74.3 80.7 70.2 60.1 29.3 52.9 73.0 71.3 64.0 Claude 3.7 Sonnet-Thinking 38.7 76.7 67.0 57.0 30.2 53.0 72.0 66.0 57.6

LLaMA-3-Instruct Series LLaMA-3-Instruct-8B 58.2 59.3 45.2 52.0 18.2 40.3 60.2 60.4 49.2 LLaMA-3-Instruct-70B 54.5 66.8 65.0 50.9 28.7 45.3 66.6 42.1 52.5 SFT-8B 70.3 59.9 65.7 43.0 5.4 18.7 26.3 33.6 40.4 Context-DPO-8B 72.9 59.5 62.3 37.5 16.7 37.8 62.3 58.3 50.9 SCOPEsum-8B 64.6 68.7 60.6 55.7 20.5 42.5 58.6 63.2 54.3 CANOE-LLaMA-8B 80.9 84.9 73.4 74.6 23.3 46.9 67.3 69.3 65.1

- Using Generated Long-form Responses. 92.3 95.5 81.6 78.2 32.7 59.3 74.1 79.1 74.1 ∆ Compared to Using Generated Short-from Response. +11.4 +10.6 +8.2 +3.6 +9.4 +12.4 +6.8 +9.8 +9.0

Qwen-2.5-Instruct Series Qwen-2.5-Instruct-7B 61.0 68.4 68.2 56.1 17.6 42.3 62.3 58.8 54.3 Qwen-2.5-Instruct-14B 47.3 61.4 64.3 51.6 21.7 48.0 69.3 65.7 53.7 Qwen-2.5-Instruct-32B 66.4 81.1 66.4 47.0 24.6 50.2 70.7 66.7 59.1 Qwen-2.5-Instruct-72B 52.3 67.3 62.2 45.2 28.0 51.0 73.0 70.6 56.2 SFT-7B 69.8 76.6 65.3 50.3 18.3 30.2 58.2 60.2 53.6 Context-DPO-7B 70.6 78.2 70.1 45.7 17.2 40.2 63.2 54.3 54.9 SCOPEsum-7B 47.9 60.9 55.3 52.3 19.5 43.5 60.1 60.7 50.0 CANOE-Qwen-7B 75.2 83.5 76.4 70.5 22.6 47.4 65.7 65.0 63.3

- Using Generated Long-form Responses. 82.9 92.3 83.2 73.2 29.8 56.9 70.6 72.7 70.2 ∆ Compared to Using Generated Short-from Response. +7.7 +8.8 +6.8 +2.7 +7.2 +9.5 +4.9 +7.7 +6.9

CANOE-Qwen-14B 87.4 88.5 84.2 67.4 25.7 51.6 71.7 69.3 68.2 - Using Generated Long-form Responses. 89.8 94.4 87.1 70.6 30.0 58.0 73.1 76.6 72.5 ∆ Compared to Using Generated Short-from Response. +2.4 +5.9 +2.9 +3.2 +4.3 +6.4 +1.4 +7.3 +4.2

- Table 9: Experimental accuracy score results (%) on short-form generation tasks. Bold numbers indicate the best performance among all the models.

design in detail and ensure that each participant correctly understands the principles. If the final result can not be determined by majority voting, we will hold a discussion among the participants and vote on the result again. We compare two models, including CANOE-LLaMA-8B as our method and LLaMA-3-Instruct-8B as the initial model. As shown in Figure 5, we can find that CANOE reduces faithfulness hallucinations and also ensures the response quality for three long-form generation tasks.

### F Discussion

Can Long-form Responses Generated by CANOE Provide Correct Answers in Short-form Generation Tasks? This exploration is important because, in real-world applications, it is difficult to pre-determine whether to use generated short-form responses (i.e., the context between <short answer>and </short answer>tags) or long-form responses (i.e., the context between <long answer>and </long answer>tags) as answers to respond to user instructions. This contrasts with the evaluation of LLMs on different datasets, as described in the test-time strategies outlined in C.5. Thus, we first explore whether the long-form responses generated by CANOE (i.e., the context between <long answer>and </long answer>tags) can provide cor-

Model Accuracy Proxy Format

CANOE-LLaMA-8B 70.3 66.1 99.4 CANOE-Qwen-7B 64.1 63.4 99.9 CANOE-Qwen-14B 83.5 76.5 100.0

Table 10: Final rewards (%) in the RL training stage.

rect answers in short-form generation tasks. As shown in Table 9, when evaluating the generated long-form responses that contain the free-form answers, the accuracy scores consistently increase in all the short-form generation tasks compared to using the generated short-form responses. It indicates that the generated short-form responses maintain conciseness, which is important for measuring the EM score, but can slightly reduce the accuracy score. Therefore, in real-world applications, we can directly use the generated long-form responses as the system responses for the user’s instructions, because these long-form responses can not only efficiently and faithfully complete long-form generation tasks, but also provide correct answers in short-form generation tasks.

Final Rewards in the RL Training Stage. We show the final rewards in Table 10. We can find that models can easily

###### Model MultiFieldQA-zh DuReader VCSUM

LLaMA-3-Instruct-8B 80.1 65.2 42.2 CANOE-LLaMA-8B 88.2 75.3 65.2 Qwen-2.5-Instruct-7B 82.3 70.3 45.5 Qwen-2.5-Instruct-14B 83.5 72.2 47.8 Qwen-2.5-Instruct-32B 85.1 77.2 52.7 Qwen-2.5-Instruct-72B 88.9 80.1 57.1 CANOE-Qwen-7B 90.1 78.3 66.5 CANOE-Qwen-14B 93.2 84.3 70.4

Table 11: Results (%) on three Chinese datasets. Bold numbers indicate the best performance of models with the same model size.

[Figure 46]

- Figure 6: The Avg EM results (%) on 11 datasets with different numbers of synthesized training data. We conduct the experiments based on LLaMA-3-Instruct-8B models.

learn the designed format, while accuracy and proxy rewards still remain challenging. Meanwhile, in the early stages of RL training, the format reward increases quickly and converges rapidly, and as training proceeds, the accuracy reward and the proxy reward gradually increase and eventually converge. This indicates that our well-designed training data construction strategy is effective and ensures the complexity and diversity, avoiding overfitting and reward hacking.

Multilingual Transfer Ability and Context Length Generalization of CANOE. To further explore the multilingual transfer ability of CANOE, we further evaluate our model on the Chinese dataset. Specifically, we use the single-document QA dataset MultiFieldQA-zh (Bai et al. 2023), the multidocument QA dataset DuReader (He et al. 2018), and the summarization dataset VCSUM (Wu et al. 2023) within LongBench (Bai et al. 2023). Following Si et al. (2025b) that utilizes the GPT-4 to evaluate the correctness of QA tasks and the faithfulness of the summarization task, we use the advanced LLM to evaluate these datasets. We use the same prompts for three tasks as Si et al. (2025b) to query GPT-4o. The test-time prompts for these tasks can be found in Figure 23, Figure 24, and Figure 25. As shown in Table 11, we can find that our proposed CANOE also improves the faithfulness in Chinese datasets, indicating that our proposed method has

Long-form QA Inputs:{} Question: What is the type and definition of this dance? GRPO-LLaMA-8B: Ballet. (✗)

CANOE-LLaMA-8B: Based on the passage, ballet is a type of dance that developed from lavish court spectacles in Italy and France. It is a combination of music, drama, poetry, song, costumes, and dance, and involves a combination of different elements, including music, costumes, and performance. The passage also mentions that members of the court nobility took part as performers, and that professional dancers began to take the place of court amateurs. This suggests that ballet is a type of dance that is characterized by its combination of different art forms and its performance by trained dancers. (✓)

###### Simplification

Inputs: The Pok´emon series began with the release of Pok´emon Red and Green for the Game Boy in Japan. When these games proved extremely popular, an enhanced blue version was released sometime after, and the blue version was reprogrammed as Pok´emon Red and Blue for international release.

GRPO-LLaMA-8B: Pocket Monsters Red and Green. (✗)

CANOE-LLaMA-8B: The Pok´emon series started with the release of Pok´emon Red and Green in Japan, which were later enhanced and reprogrammed for international release. (✓)

Summarization Inputs: {} GRPO-LLaMA-8B: Comedy clubs. (✗)

CANOE-LLaMA-8B: Kareem Badr and his friends were able to turn around the struggling comedy club, the Hideout, by applying their skills in problem-solving and efficiency. They broke down the business into small chunks and figured out how to make each aspect better and more efficient. This approach helped them to improve the club’s performance and eventually expand it. The passage also mentions that the growing popularity of stand-up comedy is fueled in part by younger audiences, who view humorous TV hosts as their primary source for news. Many comedy clubs now require talent, and there is a career and a living to be made by good comedians who are not superstars. The clubs also make money from selling drinks and dinners, as well as from cover charges and talent fees. In addition, digital media has given live comedy a boost, with one terrific joke being able to get a million hits online. (✓)

Table 12: Case study between the models trained via GRPO and our model for long-form generation tasks. Models are implemented on LLaMA-3-Instruct-8B.

a strong multilingual transfer ability. Meanwhile, these results also indicate that our method achieves better faithfulness even when our training data is short, i.e., CANOE guarantees a consistently strong performance in long-context scenarios. Effect of the Amount of the Synthesized Short-form Data. To further explore the effect of the amount of the synthesized short-form data, we conduct the corresponding experiments as shown in Figure 6. We can find that increasing the number of training data improves the performance of our method, but when the number of data is greater than 10,000, the performance will be stable. For the models with different sizes, the optimal amount of training data may be different, e.g., the larger models may need more training data to achieve optimal performance. Fortunately, due to our training data construction strategy, we can simply scale and synthesize training data without human annotation.

Case Study between GRPO and the proposed DualGRPO. We find that directly applying GRPO instead of our proposed Dual-GRPO to synthesized short-form data leads to over-optimizing short-form generation and a false response

generation pattern. The used system prompt for applying GRPO can be found in Figure 22. Shown in Table 12, we can find that the tuned model GRPO-LLaMA-8B tends to directly copy text spans from the given context as the final answer instead of following instructions in long-form generation tasks. However, when we apply Dual-GPRO to our synthesized data, we find that trained models can generate fluent and complete sentences. Thus, Dual-GRPO not only improves the faithfulness of LLMs in two types of response generation but also ensures the utility of models.

Relation Description P6 head of government P17 country

- P26 spouse
- P27 country of citizenship P30 continent

- P35 head of state
- P36 capital
- P37 official language
- P38 currency
- P39 position held P50 author P54 member of sports team P57 director P86 composer P101 field of work P103 native language P108 employer P112 founder P127 owned by P136 genre P1376 capital of P140 religion P155 follows P159 headquarters location P166 award received P170 creator P172 ethnic group P175 performer P178 developer P264 record label P276 location P286 head coach P407 language of work or name P413 position played P463 member of P488 chairperson P495 country of origin P641 sport P800 notable work P937 work location P169 chief executive officer

Table 13: Manually selected relations that are used to construct training data. We utilize the same relations as Bi et al. (2024).

Prompt for question generation for the samples with straightforward context.

[Instructions] You are a sophisticated question generator. Given a triple {(h,r,t)} collected from Wikidata, generate a question that asks about the final tail entity {t} using the head entity {h} and the relation {r}.

Directly give me the generated question:

Figure 7: Prompt for question generation for the samples with straightforward context.

Prompt for context generation for the samples with straightforward context.

[Instructions] You are a sophisticated context generator. Given a triple {(h,r,t)} collected from Wikidata, generate a brief description of the head entity {h}, approximately 150 words long. Ensure the tail entity {t} and relation {r} are accurately mentioned in the generated description.

Directly give me the generated context:

Figure 8: Prompt for context generation for the samples with straightforward context.

Prompt for question generation for the samples with reasoning-required context.

[Instructions] You are a sophisticated question generator. Given a chain of triples {[...]} collected from Wikidata, generate a question that asks about the final tail entity {t} using the head entity {h} and the relation {r}. Do not include any bridge entities in the question; instead, phrase the question as if directly asking about the relationship from the head entity to the tail entity

Directly give me the generated question:

Figure 9: Prompt for question generation for the samples with reasoning-required context.

Prompt for context generation for the samples with reasoning-required context.

[Instructions] You are a sophisticated context generator. Given a chain of triples {[...]} collected from Wikidata, generate a brief description of the head entity {h}, approximately {150*n} words long. Ensure the tail entity {t} and relation {r} are accurately mentioned in the generated description.

Directly give me the generated context:

Figure 10: Prompt for context generation for the samples with reasoning-required context.

System prompt for Dual-GRPO.

A conversation between User and Assistant. The user gives an instruction that consists of two parts: a passage and the actual instruction, separated by two newline characters.

The passage is provided within <context>and </context>tags. The Assistant needs to refer to the given passage and complete the instruction.

The Assistant solves the question by first thinking about the reasoning process internally, according to the given passage, and then providing the response.

The response must be structured and include the following three sections, clearly marked by the respective tags:

- - Reasoning Process: Explain your thought process or logical steps to derive the answer. Enclose this within <think>and </think>tags.
- - Long Answer: Provide a long response that consists of syntactically and semantically complete sentences to answer the question. Enclose this within <long answer>and </long answer>tags.

- - Short Answer: Present a concise response that directly answers the question. Enclose this within <short answer>and </short answer>tags.

Format your response exactly as follows: <think>reasoning process here. </think><long answer>detailed answer here. </long answer><short answer>the concise answer here. </short answer>.

Figure 11: System prompt for Dual-GRPO.

Prompt used to calculate quality score for text summarization.

You are asked to evaluate the quality of the AI assistant’s generated summary as an impartial judge, and your evaluation should take into account factors including readability (whether the summary is clear and easy to understand) and coherence (whether the assistant’s summary is logical and orderly).

Read the AI assistant’s summary and input passages, and give an overall integer rating in on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the evaluation criteria, strictly in the following format:“[[rating]]”, e.g. “[[5]]”.

Input Passages: {} Assistant’s summary:{} Rating:

Figure 12: Prompt used to calculate quality score for text summarization.

Prompt used to calculate quality score for text simplification.

You are asked to evaluate the quality of the AI assistant’s generated text simplification as an impartial judge, and your evaluation should take into account factors including readability (whether the simplification is clear and easy to understand) and coherence (whether the assistant’s simplification is logical and orderly).

Read the AI assistant’s simplified version and the original text, and give an overall integer rating on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the evaluation criteria, strictly in the following format: “[[rating]]”, e.g. “[[5]]”.

Original text: {} AI assistant’s simplification: {} Rating:

Figure 13: Prompt used to calculate quality score for text simplification.

Prompt used to calculate quality score for long-form QA.

You are asked to evaluate the quality of the AI assistant’s generated long-form answer as an impartial judge, and your evaluation should take into account factors including readability (whether the answer is clear and easy to understand) and coherence (whether the answer is logical and well-organized).

Read the AI assistant’s long-form answer and the original question, and give an overall integer rating on a scale of 1 to 5, where 1 is the lowest and 5 is the highest, based on the evaluation criteria, strictly in the following format: “[[rating]]”, e.g., “[[5]]”.

Question: {} Assistant’s long-form answer: {} Rating:

Figure 14: Prompt used to calculate quality score for long-form QA.

Test-time prompt used for short-form QA tasks. Passages: {} Refer to the passages above and answer the following question with just a few words. Question: {} Answer:

Figure 15: Test-time prompt used for short-form QA tasks.

Test-time prompt used for multiple-choice QA task. Passages: {} Refer to the passages above and answer the following question with just a few words. Question: {} Please select the correct option according to the question, and output the option letter (e.g. A/B/C/D): Options: {} Answer:

Figure 16: Test-time prompt used for multiple-choice QA task.

Test-time prompt used for text summarization. Passage: {} Refer to the passage above and provide a summary as the response. Summary:

Figure 17: Test-time prompt used for text summarization.

Test-time prompt used for text simplification. Passage: {} Refer to the passage above and simplify it to improve its readability, ensuring its core meaning remains intact. Please provide only the simplified text as the response. Simplified text:

Figure 18: Test-time prompt used for text simplification.

Test-time prompt used for long-form QA task. Passage: {} Refer to the passages above and answer the following question. Question: { }

Figure 19: Test-time prompt used for long-form QA task.

Test-time prompt used for FaithEval in closed-book QA settings. Question: {} Please select the correct option according to the question, and output the option letter (e.g. A/B/C/D): Options: {} Answer:

Figure 20: Test-time prompt used for FaithEval in closed-book QA settings.

The principles of human evaluation for long-form responses generation.

You are asked to evaluate the responses generated by different models. You should choose the preferred responses according to the following perspectives independently:

- 1. Readability: Whether the response is clear and easy to understand?
- 2. Faithfulness: Whether the response is faithful to the context and the information can be grounded in the provided context.
- 3. Helpfulness: Whether the response provides useful information and follows the instructions from users?
- 4. Naturalness: Whether the response sounds natural and fluent? Finally, please make a decision among the 3 opinions, including Win, Tie, and Loss.

Figure 21: The principles of human evaluation for long-form responses generation.

System prompt for GRPO in the ablation study.

A conversation between User and Assistant. The user gives an instruction that consists of two parts: a passage and the actual instruction, separated by two newline characters.

The passage is provided within <context>and </context>tags. The Assistant needs to refer to the given passage and complete the instruction.

The Assistant solves the question by first thinking about the reasoning process internally, according to the given passage, and then providing the response.

The response must be structured and include the following two sections, clearly marked by the respective tags:

- - Reasoning Process: Explain your thought process or logical steps to derive the answer. Enclose this within <think>and </think>tags.
- - Answer: Present a concise response that directly answers the question. Enclose this within <answer>and </answer>tags.

Format your response exactly as follows: <think>reasoning process here. </think><answer>answer here. </answer>.

Figure 22: System prompt for GRPO in the ablation study.

Test-time prompt used for MultiField-zh. 阅读以下文字并用中文简短回答：{} 现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。 问题：{} 回答：

Figure 23: Test-time prompt used for MultiField-zh.

Test-time prompt used for DuReader. 请基于给定的文章回答下述问题。

文章：{} 问题：{} 回答：

Figure 24: Test-time prompt used for DuReader.

Test-time prompt used for VCSUM. 下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。 会议记录：{} 会议总结：

Figure 25: Test-time prompt used for VCSUM.

