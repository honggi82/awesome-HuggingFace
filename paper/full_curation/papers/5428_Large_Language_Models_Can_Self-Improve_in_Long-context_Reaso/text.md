# arXiv:2411.08147v1[cs.CL]12Nov2024

## Large Language Models Can Self-Improve in Long-context Reasoning

### Siheng Li♡ Cheng Yang♡ Zesen Cheng♠ Lemao Liu♢ Mo Yu♢ Yujiu Yang♣ Wai Lam♡

♡The Chinese University of Hong Kong

♠Peking University ♣Tsinghua University ♢Tencent

sihengli24@gmail.com Correspondence: moyumyu@tencent.com wlam@se.cuhk.edu.hk

### Abstract

Large language models (LLMs) have achieved substantial progress in processing long contexts but still struggle with long-context reasoning. Existing approaches typically involve fine-tuning LLMs with synthetic data, which depends on annotations from human experts or advanced models like GPT-4, thus restricting further advancements. To address this issue, we investigate the potential for LLMs to selfimprove in long-context reasoning and propose SEALONG, an approach specifically designed for this purpose. This approach is straightforward: we sample multiple outputs for each question, score them with Minimum Bayes Risk, and then apply supervised fine-tuning or preference optimization based on these outputs. Extensive experiments on several leading LLMs demonstrate the effectiveness of SEALONG, with an absolute improvement of 4.2 points for Llama-3.1-8B-Instruct. Furthermore, SEALONG achieves superior performance compared to prior approaches that depend on data produced by human experts or advanced models. We anticipate that this work will open new avenues for self-improvement techniques in long-context scenarios, which are essential for the continual advancement of LLMs.1

### 1 Introduction

Large language models (LLMs) with long-context processing capabilities have spurred a range of novel applications, such as repository-level coding assistance (Jimenez et al., 2024), multi-document analysis (Wang et al., 2024b) and autonomous agents (Ma et al., 2024). Delivering high-quality service in these domains requires LLMs to reason effectively over long contexts, necessitating the retrieval of essential details and integration of dispersed information throughout the reasoning process. While recent advancements have enabled

1The repository can be accessed at https://github.com/ SihengLi99/SEALONG.

LLMs to attain near-perfect accuracy on the needlein-a-haystack (NIAH; Kamradt (2023); Li et al. (2024b)) task (Hsieh et al., 2024; Yen et al., 2024b; Dubey et al., 2024), which involves locating evidence within vast amounts of irrelevant text, substantial performance declines persist on tasks that require reasoning over long contexts (Levy et al., 2024; Hsieh et al., 2024; Vodrahalli et al., 2024; Yen et al., 2024b; Li et al., 2024a), limiting their applicability in real-word scenarios.

To address this limitation, recent studies have investigated fine-tuning LLMs to improve longcontext reasoning, with effective data synthesis as a primary challenge. Two main approaches have emerged: one relies on human annotations (Chen et al., 2024b; Li et al., 2024c,a), which are expensive and difficult to scale; the other leverages expert models, such as GPT-4o (Hurst et al., 2024), for data synthesis. For example, Bai et al. (2024); Zhang et al. (2024c,b) apply self-instruct techniques (Wang et al., 2023b) to create longcontext instruction-following data. Despite substantial progress, the dependence on pre-existing expert models limits the potential for achieving more advanced capabilities.

This work investigates whether LLMs can selfimprove in long-context reasoning. Drawing on evidence of LLMs’ near-perfect long-context retrieval and strong reasoning abilities in general domains (Bubeck et al., 2023; Zhong et al., 2024), we hypothesize that LLMs have untapped potential in long-context reasoning. Our preliminary studies show that refined prompting strategies achieve notable improvements over both default prompting methods and direct answer requests. Furthermore, scaling the number of sampled outputs reveals a marked performance gap between the optimal outputs and those derived via greedy search. These results suggest that LLMs hold substantial potential to advance in long-context reasoning.

Inspired by these observations, we propose a

Llama-3.1-8B-Instruct Llama-3.1-70B-Instruct HotpotQA MuSiQue 2WikiMQA HotpotQA MuSiQue 2WikiMQA

Prompt

Default 55.5 33.0 66.0 60.0 54.0 77.0 Direct answer 49.0 28.5 55.0 61.5 51.5 74.0 Think step-by-step (Kojima et al., 2022) 62.5 50.5 77.5 75.5 62.5 85.0 Fact-and-reflection (Zhao et al., 2024b) 67.0 49.0 76.5 78.0 62.0 84.0 Plan-and-solve (Wang et al., 2023a) 64.0 49.5 82.0 74.0 68.5 85.5

Table 1: Comparison of various prompting methods. The best result is highlighted in bold.

Self-improving method for rEAsoning over LONGcontexts (SEALONG). This involves first sampling multiple reasoning trajectories from the LLM, then scoring each based on Minimum Bayes Risk (MBR) (Bickel and Doksum, 1977), which prioritizes outputs that are more consistent with others. This idea is intuitive, as reasoning trajectories that deviate from the majority are more likely to be hallucinations (Manakul et al., 2023; Farquhar et al., 2024). Following this, we can either conduct supervised fine-tuning using high-scoring outputs or apply preference optimization by utilizing both high-scoring and low-scoring outputs.

We apply SEALONG to several leading LLMs and conduct evaluations on multiple long-context reasoning tasks (Bai et al., 2023; Yang et al., 2018; Trivedi et al., 2022; Ho et al., 2020; Dasigi

- et al., 2021). The results reveal that LLMs can self-improve in long-context reasoning. Specifically, SEALONG raises the score of Llama-3.1-8BInstruct (Dubey et al., 2024) from 50.8 to 55.0. Additionally, SEALONG enables Qwen-2.5-14BInstruct (Yang et al., 2024a) to outperform its 32B variant (54.7 vs. 53.1). In comparison to previous synthetic data, SEALONG demonstrate notable improvement without requiring human or expert model annotation. We hope that SEALONG can pave the way for self-improving approaches in long-context scenarios, supporting the continual advancement of LLM capabilities.

- 2 Understanding the Potential of LLMs in Long-context Reasoning

We explore the potential of LLMs in long-context reasoning through experiments on three reasoningintensive tasks from LongBench (Bai et al., 2023): HotpotQA (Yang et al., 2018), MuSiQue (Trivedi

- et al., 2022) and 2WikiMQA (Ho et al., 2020). These tasks involve handling multiple documents within the context and addressing multi-hop questions that span several paragraphs. Following pre-

vious work (Mallen et al., 2023; Asai et al., 2024; Yen et al., 2024b), we use substring exact match (SubEM) for evaluation, assessing whether the golden answer is included in the output.

#### 2.1 Prompting Strategies Matter

Numerous long-context evaluation benchmarks assess LLMs by simply asking them to respond to a query based on a long context (Bai et al., 2023; An et al., 2024a; Zhang et al., 2024d; Wang et al., 2024b; Yen et al., 2024b). We suggest that this approach may underestimate LLMs’ potential in long-context scenarios, particularly for questions requiring complex, multi-step reasoning to arrive at an answer. To further investigate this, we examine various prompting strategies for long-context reasoning, including:

- • Default: Prompting the LLM with the long context and a question.
- • Direct Answer: Asking the LLM to directly answer the question based on the long context.
- • Think Step-by-step: Providing the LLM with the context, question, and an instruction to think step-by-step (Kojima et al., 2022).
- • Fact-and-reflection: Providing the LLM with the long context, question, and an instruction to first identify the relevant information from the long context, and then carry out step-bystep reasoning and provide the answer (Zhao et al., 2024b; Li et al., 2024a).
- • Plan-and-solve: Providing the LLM with the long context, question, and an instruction to first devise a plan and then follow it to solve the problem step-by-step (Wang et al., 2023a).

The detailed prompts for these strategies are presented in Tab. 9 (Appx. B). As shown in Tab. 1, prompting strategies play a crucial role in longcontext reasoning. A notable performance gap

[Figure 1]

- Figure 1: Scaling up the number of sampled outputs improves the performance of both the oracle sample and MBR decoding (§3.1). The results are based on Llama-3.1-8B-Instruct.

exists between default prompting and reasoningtargeted prompting strategies, aligning with observations in short-context tasks (Wei et al., 2022; Zhou et al., 2023). Manual inspection reveals that with an appropriate prompting strategy, the LLM breaks down multi-hop questions into simpler parts, addresses each part using the long context, and ultimately arrives at an answer.

- 2.2 The Potential of LLMs for Correct Long-context Reasoning

We further investigate the potential of LLMs for long-context reasoning by expanding the generation space. Specifically, we use temperature sampling to produce multiple outputs per question, evaluate each with SubEM, and designate the highestscoring output as the oracle sample. As shown in Fig. 1, there is a notable gap between oracle performance and that of greedy search, even with just 8 outputs. Scaling up to 128 samples achieves over 90% correct answers. These results underscore the potential of LLMs for long-context reasoning and motivate the development of methods that enable LLMs to self-improve in this area.

3 SEALONG

Motivated by the potential of LLMs in long-context reasoning (§2), we propose SEALONG, a selfimproving method for reasoning over long contexts. This approach consists of two stages: creating self-supervision and fine-tuning the model. An overview of SEALONG is provided in Fig. 2.

- 3.1 Self-supervision

We begin by leveraging plan-and-solve prompting (Wang et al., 2023a) to sample multiple reasoning

trajectories for each question and its corresponding long context. The primary challenge lies in evaluating these outputs. The fundamental idea behind SEALONG is that correct reasoning trajectories typically exhibit higher semantic consistency. For example, they tend to follow similar planning steps and reference the same information within the long context. This observation aligns with hallucination detection methods (Manakul et al., 2023; Farquhar et al., 2024), where less consistent outputs are more likely to indicate hallucinations, representing incorrect reasoning in our scenario.

We formalize this idea using Minimum Bayes Risk (MBR) (Bickel and Doksum, 1977; Bertsch et al., 2023; Wu et al., 2024), which prioritizes outputs that exhibit higher consistency with others. In the MBR literature, the quality of an output is assessed by its expected utility under the model distribution (Bickel and Doksum, 1977; Kumar and Byrne, 2004; Tromble et al., 2008):

s(y) = Ey∗∼πθ(y|x) [u(y,y∗)]

Here, s(y) is the score assigned to output y, where x denotes the input, including the long context, question and instruction. The term πθ(y | x) represents the policy distribution of the LLM and the utility metric u(y,y∗) assesses y based on y∗. We approximate this expectation using the Monte Carlo method with N sampled outputs:

1 N

s(y) ≈

N

[u(y,y∗)]

i=1

The utility metric measures the consistency between two outputs. We use sentence embedding similarity as this metric, as it effectively captures

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

“Let the Heartaches Begin” is a song performed

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

|[Figure 12]<br><br>! Chosen<br><br>[Figure 13]|
|---|

…... (30k-word long document) What is the country of citizenship of Just a Little Heartache's performer?

[Figure 14]

[Figure 15]

!

[Figure 16]

[Figure 17]

!

|[Figure 18]<br><br>" Rejected<br><br>[Figure 19]|
|---|

Self-supervision Fine-tuning

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Ⅰ: To solve the problem, we need to … Step 1: Identify the performer of the song. According to Passage 3 … Step 2: Determine Maria Arredondo’s citizenship …

Ⅰ Ⅱ Ⅲ

Score

Self-supervision

|1.0|0.8|0.6|Minimum<br><br>|
|---|---|---|---|
|0.8|1.0|0.4| |
| | | |Bayes Risk|
|0.6|0.4|1.0| |

|0.80<br><br>[Figure 24]<br><br>!|
|---|
|0.73|
|0.67<br><br>[Figure 25]<br><br>"|

- Ⅰ
- Ⅱ
- Ⅲ

- Ⅰ
- Ⅱ
- Ⅲ

[Figure 26]

[Figure 27]

[Figure 28]

- Ⅱ: We can break down this … First, let’s recognize the performer. Based on Passage 3 … Then, we need to find out the residence of Maria Arredondo …

[Figure 29]

[Figure 30]

- Ⅲ : To find out the citizenship of ... Step 1: Confirm the performer. As stated in Passage 9 … Step 2: Determine the citizenship of Silje Nergaard …

Sampling Outputs

- Figure 2: SEALONG consists of two stages: self-supervision creation and fine-tuning. Given a long context and a corresponding query, multiple outputs are sampled, each assigned a score based on Minimum Bayes Risk. Fine-tuning is then conducted using either the highest-scoring output for supervised fine-tuning or both high-scoring and low-scoring outputs for preference optimization.

the semantic alignment between the two reasoning trajectories. Formally:

likelihood of the output as follows:

1 |y|

log πθ(y | x)

LSFT = −

##### u(y,y∗) = Sim(Emb(y),Emb(y∗))

|y|

1 |y|

We employ a lightweight RoBERTa-based model (Liu, 2019) to embed outputs and measure similarity with the inner product. This approach allows us to assign each output y a score s(y), and selecting the output with the highest score is referred to as MBR decoding (Bickel and Doksum, 1977; Bertsch et al., 2023; Wu et al., 2024). As demonstrated in Fig. 1, MBR decoding substantially surpasses greedy search: with absolute improvements of 11.5% on MuSiQue (Trivedi et al., 2022), 5.0% on HotpotQA (Yang et al., 2018), and 5.0% on 2WikiMultihopQA (Ho et al., 2020) when N = 128. These results highlight the potential for LLMs to self-improve by leveraging multiple samples and an effective evaluation metric based on output consensus, eliminating the need for human experts or advanced models. Furthermore, this evaluation approach produces preference pairs by contrasting high-scoring and low-scoring outputs, allowing straightforward preference optimization (Ouyang et al., 2022; Rafailov et al., 2024).

= −

log πθ(yi | x,y<i)

i=1

Here, y denotes the MBR decoding output.

Preference Optimization. Alternatively, we can conduct preference optimization to reinforce the tendency toward high-scoring outputs and reduce the likelihood of low-scoring outputs. Among the various preference optimization methods, we adopt the monolithic odds ratio preference optimization (ORPO) algorithm (Hong et al., 2024) due to its strong empirical performance. ORPO introduces an odds ratio loss to minimize the negative log odds ratio between a preferred output yw and a less-preferred output yl:

oddsθ(yw|x) oddsθ(yl|x)

LOR = −log σ log

Here, σ represents the sigmoid function, and oddsθ(y|x) measures how much more likely y is to be generated than not:

#### 3.2 Fine-tuning

πθ(y|x) 1 − πθ(y|x)

oddsθ(y|x) =

Leveraging self-provided supervision, we can either perform supervised fine-tuning on the highestscoring outputs or apply preference optimization using preference pairs.

The final objective in ORPO combines SFT and OR losses, with a hyperparameter β controlling their relative importance:

Supervised Fine-tuning. For supervised finetuning (SFT), we minimize the negative log-

##### LORPO = LSFT + β · LOR

Model Qasper MultiFieldQA-En HotpotQA MuSiQue 2WikiMQA Avg. Qwen-2.5-7B-Instruct (Yang et al., 2024a) 21.0 28.0 70.5 48.0 77.5 49.0

+ SEALONG 26.0 29.3 72.5 51.5 79.5 51.8 Qwen-2.5-14B-Instruct (Yang et al., 2024a) 21.0 32.0 73.0 52.0 83.0 52.2

+ SEALONG 24.0 30.0 75.0 57.0 87.5 54.7 Llama-3.1-8B-Instruct (Dubey et al., 2024) 29.0 29.3 64.0 49.5 82.0 50.8

###### + SEALONG 32.5 31.3 68.0 58.5 84.5 55.0

Qwen-2.5-32B-Instruct (Yang et al., 2024a) 24.5 26.0 72.0 55.0 88.0 53.1 Qwen-2.5-72B-Instruct (Yang et al., 2024a) 27.0 28.7 74.5 58.5 89.0 55.5 Llama-3.1-70B-Instruct (Dubey et al., 2024) 30.0 33.3 74.0 68.5 85.5 58.3 GPT-4o (Hurst et al., 2024) 21.5 28.0 74.5 64.0 84.0 54.4

- Table 2: Main evaluation results. Substring exact match (SubEM) serves as the evaluation metric, with the topperforming results emphasized in bold. SEALONG utilizes the training set of MuSiQue with self-supervision (§3.1), and its performance on other tasks demonstrates the generalization ability of SEALONG.

Task # Example Max Tokens Avg. Tokens Qasper 200 21,110 4,921 MultiFieldQA-en 150 14,947 6,888 HotpotQA 200 16,322 12,779 MuSiQue 200 16,335 15,542 2WikiMultihopQA 200 16,319 7,096

- Table 3: Statistics of evaluation tasks, with token counts calculated using the tokenizer of Llama-3.1-8B-Instruct.

Model Avg. Long-context Avg. Output Tokens Qwen-2.5-Instruct 7B 49.0 375

+ SEALONG 51.8 371 Llama-3.1-Instruct 8B 50.8 289 + SEALONG 55.0 295

- Table 4: Average performance on long-context tasks (Tab. 2) and average token count in model predictions for these tasks, measured with the model’s tokenizer.

temperature of 0.7. By default, we synthesize 2048 examples for fine-tuning, with context lengths randomly specified between 4K and 31K tokens. We conduct experiments using the Llama-3.1 models (Dubey et al., 2024) and Qwen-2.5 models (Yang et al., 2024a), with jina-embeddings-v3 serving as the sentence embedding model (Sturua et al., 2024). ORPO (Hong et al., 2024) is employed as the default fine-tuning method. More training details can be found in Appx. A.

#### 4.2 Evaluation Setup

We conduct evaluations in long-context scenarios across a wide range of tasks. For single-document QA, we include Qasper (Dasigi et al., 2021) and MultiFieldQA-En (Bai et al., 2023) from the LongBench benchmark (Bai et al., 2023). For multidocument QA, we use HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2022) and 2WikiMultihopQA (Ho et al., 2020), also from LongBench. Task statistics are presented in Tab. 3. We adopt plan-and-solve prompting for evaluation due to its strong performance (Tab. 1). Following previous research (Mallen et al., 2023; Asai et al., 2024; Yen et al., 2024b), we use substring exact match (SubEM) as the evaluation metric, measuring whether the output contains the golden answer.

In our implementation, we use the MBR decoding output as yw and randomly select a low-scoring output to serve as yl.

### 4 Experiments

#### 4.1 Implementation

SEALONG requires query and long-context pairs to synthesize training data. Specifically, we leverage the training dataset of MuSiQue (Trivedi et al.,

- 2022), where each question is related to several Wikipedia documents. To achieve a specified number of tokens in the context, we randomly sample some unrelated documents, shuffle them with the related ones and concatenate them into a single context. We use the original questions in MuSiQue without the annotated answer, relying on the LLM to produce self-supervision (§3.1). For each question, we sample N = 32 outputs with a sampling

#### 4.3 Main Results

SEALONG Improves Various Models. We implement SEALONG on the leading open-source LLMs, including Qwen-2.5 models (Yang et al., 2024a) and Llama-3.1 models (Dubey et al., 2024). As illustrated in Tab. 2, SEALONG brings notable improvements: when implemented on Qwen-2.57B-Instruct, it closes the performance gap with

Model Qasper MultiFieldQA-En HotpotQA MuSiQue 2WikiMQA Avg. Llama-3.1-8B-Instruct 29.0 29.3 64.0 49.5 82.0 50.8 Supervised Fine-tuning

+ TULU-V2-mix 26.5 27.3 49.5 27.5 54.0 37.0 + WildChat 20.5 29.3 46.5 28.0 58.0 36.5 + LongAlpaca 22.5 31.3 48.0 31.0 45.0 35.6 + LongAlign 25.0 36.7 58.5 47.5 76.0 48.7 + LongMIT 20.0 30.0 56.0 36.0 66.5 41.7 + LongReward-SFT 22.0 28.7 58.0 52.0 76.5 47.4 + GPT-4o-MuSiQue 21.5 31.3 64.0 54.0 83.5 50.9 + SEALONG-SFT 28.5 30.7 68.5 50.5 84.0 52.4

Preference Optimization

+ UltraFeedback 26.0 27.3 47.5 28.5 46.0 35.1 + LongReward-Preference 26.5 32.0 63.5 52.0 80.5 50.9 + SEALONG 32.5 31.3 68.0 58.5 84.5 55.0

- Table 5: A comparison between SEALONG and previous datasets. The results are based on Llama-3.1-8B-Instruct finetuned on the corresponding dataset. To ensure fairness, 2K examples are randomly sampled from each dataset, with the exception of TULU-V2-mix, WildChat, and UltraFeedback, where the longest 2K examples are selected. The preference optimization strategy is ORPO (Hong et al., 2024).

Dataset Supervision Avg. Tokens TULU-V2-mix (2023) [1], [2], [3] 3,788 WildChat (2024a) [2], [3] 32,230 LongAlpaca (2024b) [1], [4] 9,160 LongAlign (2024) [4] 16,881 LongMIT (2024c) [5] 78,412 LongReward-SFT (2024b) [6] 22,206 LongReward-Preference (2024b) [6] 22,689 UltraFeedback (2023) [3] 1,356 GPT-4o-MuSiQue [7] 18,476 SEALONG [8] 18,532

- Table 6: Dataset statistics, including supervision source and average token count, measured with the Llama3.18B-Instruct tokenizer. Sources: [1] Human, [2] GPT-

answer. To explore this, we examine output token counts. As shown in Tab. 4, SEALONG has minimal effect on the number of output tokens.

SEALONG Competes with Previous Datasets. We compare SEALONG with several previous datasets, including short-context datasets such as TULU-V2-mix (Ivison et al., 2023), WildChat (Zhao et al., 2024a), UltraFeedback (Cui et al., 2023), as well as long-context datasets including LongAlpaca (Chen et al., 2024b), LongAlign (Bai et al., 2024), LongMIT (Chen et al., 2024c), and LongReward (Zhang et al., 2024b). Additionally, we utilize GPT-4o to synthesize data using the same question and long-context as SEALONG, creating a dataset we term GPT-4o-MuSiQue. Dataset statistics are presented in Tab. 6. To ensure fairness, 2K examples are randomly sampled from each dataset, with the exception of TULU-V2-mix, WildChat, and UltraFeedback, where the longest 2K examples are selected. As demonstrated in Tab. 5, most previous datasets negatively affect the performance of Llama-3.1-8B-Instruct, consistent with the observation of Gao et al. (2024). We hypothesize that this is because Llama-3.1-8B-Instruct already has strong long-context processing capabilities, and additional training on low-quality synthetic data could diminish its performance. However, we observe a performance improvement with SEALONG (50.8 to 55.0), indicating that self-improvement holds promise, which is particularlly promising as

- 3.5-Turbo (OpenAI, 2022), [3] GPT-4 (Achiam et al.,

- 2023), [4] Claude (Anthropic, 2023), [5] Qwen2-72BInstruct (Yang et al., 2024a), [6] GLM-4 (GLM et al.,
- 2024), [7] GPT-4o (Hurst et al., 2024), and [8] Self.

Qwen-2.5-14B-Instruct (51.8 vs. 52.2); when applied to Qwen-2.5-14B-Instruct, it even exceeds the performance of Qwen-2.5-32B-Instruct (54.7

- vs. 53.1). Additionally, SEALONG yields an absolute improvement of 4.2 on Llama-3.1-8B-Instruct, outperforming GPT-4o (Hurst et al., 2024) (55.0
- vs. 54.4). Although SEALONG utilizes MuSiQue for data synthesis, it achieves strong performance across other tasks as well, highlighting its generalization potential. One possible shortcut of SEALONG is producing more tokens, as the evaluation metric, SubEM, might favor outputs with more tokens, which are more likely to contain the golden

current LLMs advance rapidly.

#### 4.4 Analysis

Method HotpotQA MuSiQue 2WikiMQA Greedy Search 64.0 49.5 82.0 Random 61.0 50.5 79.5 Reference-free Self-evaluation 64.0 51.5 83.0

Minimum Bayes Risk

ROUGE 66.5 53.5 85.0 BERTScore 67.5 50.0 86.5 Reference-based Self-evaluation 63.5 51.5 84.5 Sentence Embedding 67.5 56.0 88.0

- Table 7: Comparison of various scoring methods and greedy search. Each scoring method evaluates 16 outputs sampled from Llama-3.1-8B-Instruct. The results indicate the performance of the highest-scoring output for each method.

Scoring Methods. Effective scoring methods are critical for creating self-supervision signals (§3.1). We explore several approaches, including random selection, and reference-free self-evaluation, which prompts the LLM to assess its prediction in a separate turn based on the question and context. Additionally, we investigate various strategies for the utility metric u(y,y∗) within Minimum Bayes Risk (§3.1), such as ROUGE (Lin, 2004), BERTScore (Zhang et al., 2019) and referencebased self-evaluation, which prompts the LLM to assess y using y∗ as the reference. The detailed prompts for reference-free and reference-based selfevaluation are presented in Tab. 10 (Appx. B). For each question, we sample N = 16 outputs using a temperature of 0.7. Subsequently, we evaluate the highest-scoring output across different scoring methods and further compare these results to the performance of greedy search as a reference. As shown in Tab. 7, MBR-based methods outperform reference-free self-evaluation, even with simple N-gram-based ROUGE. We attribute this to the limited self-evaluation capabilities of current LLMs (Huang et al., 2024; Jiang et al., 2024), which might be more challenging in long-context scenarios. Incorporating more semantic information through sentence embeddings further improves MBR-based methods.

Number of Synthetic Examples. We analyze the impact of the number of training examples synthesized by SEALONG on long-context tasks. As shown in Fig. 3, SEALONG demonstrates strong data efficiency, achieving competitive performance with only 1K examples, after which additional ex-

[Figure 31]

- Figure 3: Long-context performance of SEALONG with varying numbers of synthetic training examples, evaluated based on Llama-3.1-8B-Instruct fine-tuned on the corresponding dataset.

amples provide limited benefit. This suggests that SEALONG is unlocking the inherent potential of LLMs for long-context reasoning rather than introducing a new skill that would require more data.

[Figure 32]

- Figure 4: Long-context performance of SEALONG with varying numbers of samples per example during data synthesis, evaluated based on Llama-3.1-8B-Instruct fine-tuned on the corresponding dataset.

Number of Samples per Example. We continue to explore the effect of the number of samples, N, per example. As illustrated in Fig. 4, increasing N from 8 to 32 consistently improves performance, likely due to more accurate MBR estimation (§3.1). Beyond 32, except for MuSiQue, the performance improvement diminishes. This may indicate a fundamental limitation of our scoring method, which appears to struggle with selecting higher-quality outputs from larger output sets when N > 32 (see also in Tab. 1). We believe the scoring method is pivotal to self-improvement and will investigate this aspect further in future work.

Short-context Performance. Improving longcontext reasoning should not compromise shortcontext performance. To investigate this further, we evaluate SEALONG on the Open LLM Leaderboard (Beeching et al., 2023), covering 6 tasks that represent diverse capabilities: MMLU (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021), ARCchallenge (Clark et al., 2018), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021) and TruthfulQA (Lin et al., 2022). As shown in Tab.

Long-Context Short-Context

Model

Avg. MMLU GSM8K ARC-Challenge HellaSwag Winogrande TruthfulQA Avg. Qwen-2.5-7B-Instruct 49.0 74.2 82.4 67.1 81.5 74.7 64.7 74.1

+ SEALONG 51.8 74.1 83.2 66.5 81.3 74.4 64.8 74.1 Llama-3.1-8B-Instruct 50.8 68.3 77.7 60.2 80.1 77.4 54.1 69.6

+ SEALONG 55.0 68.4 77.8 60.3 79.9 77.3 53.8 69.6

- Table 8: Evaluation results on short-context tasks from the Open LLM Leaderboard (Beeching et al., 2023), with the long-context average performance referenced from Tab.2. SEALONG demonstrates a marked improvement in long-context performance, with minimal impact on short-context performance.

8, while SEALONG achieves substantial improvements in long-context performance, it has minimal impact on short-context performance.

### 5 Related Work

Long-context Language Modeling. Numerous studies explore methods to extend the long-context processing abilities of LLMs. One line of research approaches addresses this challenge from a modelcentered perspective, with some studies focusing on minimal modifications to existing LLMs, such as adjustments to position embeddings (Chen et al.,

- 2023; Peng et al., 2024; Ding et al., 2024; Zhu et al., 2024; Xiong et al., 2024) and refinements to the attention mechanism (Ding et al., 2023; Jin et al., 2024; An et al., 2024b,c). Additionally, some works propose novel architectures for efficient longcontext processing (Wu et al., 2022; Bertsch et al.,
- 2024; Wang et al., 2024d; Yen et al., 2024a; Lieber et al., 2024; Ye et al., 2024; Sun et al., 2024). Another line of research adopts a data-centric perspective, focusing on data engineering strategies. For example, Dubey et al. (2024); Lieber et al. (2024); Fu et al. (2024); Gao et al. (2024) continue pretraining models on long sequences, while An et al. (2024d); Bai et al. (2024); Zhang et al. (2024b); Chen et al. (2024c,b) leverage expert models or human annotations to create long-context data for fine-tuning. In contrast to these approaches, this work aims to facilitate the self-improvement of LLMs in long-context reasoning.

Self-improving. The self-improvement of LLMs has become a vital area of research as these models advance toward human intelligence. Research in this area follows two main approaches. The first approach investigates the self-reflection capabilities of LLMs, where models are prompted to assess and refine their own outputs (Ganguli et al., 2023; Madaan et al., 2024; Shinn et al., 2024; Xie et al., 2024; Gou et al., 2024; Chen et al., 2024a; Pan

et al., 2024). However, the reliability of these selfrefinement has been questioned in recent studies (Huang et al., 2024; Jiang et al., 2024). The second approach involves generating synthetic training data through the models themselves. This process typically involves generating multiple outputs for a given input, filtering out inaccurate results based on ground-truths, and using the remaining correct responses for model fine-tuning Zelikman et al. (2022); Hosseini et al. (2024); Pang et al. (2024); Wang et al. (2024c); Gulcehre et al. (2023); Zhang et al. (2024a). Additionally, Yuan et al. (2024) finetune LLMs to assign rewards to their own outputs using human preference data and facilitate continual improvement in instruction following. To reduce reliance on human annotations, some studies adopts consensus-based supervision, designating the output with the higher consensus across multiple outputs as better, with applications in areas such as arithmetic and logical reasoning (Huang et al., 2023; Prasad et al., 2024), machine translation (Finkelstein and Freitag, 2024; Wang et al., 2024a; Yang et al., 2024b), and instruction following (Wu et al., 2024). SEALONG first reveals the underestimated potential of LLMs in long-context reasoning and then leverages a consensus-based supervision strategy to enable LLMs to self-improve in long-context reasoning.

### 6 Conclusion

In this study, we investigate the potential of LLMs to self-improve in long-context reasoning and propose SEALONG for this purpose. This method achieves substantial improvements across multiple long-context reasoning tasks. We hope this research will open new avenues for self-improvement in long-context reasoning, which is vital for the sustained progress of LLMs, particularly as they advance toward surpassing human intelligence.

### Limitations

We recognize that this work has several limitations that warrant further investigation.

Scoring Method. To establish self-supervision (§3.1), we score each output according to Minimum Bayesian Risk (MBR), which reflects consensus across multiple sampled outputs. However, a substantial performance gap remains between the highest MBR-scored output and the oracle sample (see Tab. 1 for details). Future research should explore more effective approaches for self-evaluation of outputs. One possible direction could involve examining the critic capabilities of LLMs in longcontext scenarios (Lan et al., 2024b; Lin et al., 2024; Lan et al., 2024a).

Synthetic Data. Another limitation of this work is its reliance on MuSiQue (Trivedi et al., 2022) for synthetic data, which consists of multi-hop questions spanning multiple paragraphs. While this approach has enabled some progress, MuSiQue dose not cover all challenging question types, such as those requiring full-context reasoning, which remains a key limitation of current long-context LLMs (Karpinska et al., 2024; Wang et al., 2024b; Vodrahalli et al., 2024; Yen et al., 2024b). We advocate for future work to prioritize the creation of high-quality prompt sets, which are essential for the development of long-context LLMs.

Experimental Setup. Due to the computational limitations, we restrict the implementation of SEALONG to LLMs with up to 14B parameters, though its effectiveness at larger scales warrants further investigation. Likewise, the maximum sequence length is set to 32K tokens, whereas current leading LLMs support context lengths of up to 128K tokens or more. We leave the exploration of longer context lengths for future work.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024a. L-eval: Instituting standardized evaluation for long context language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1:

Long Papers), Bangkok, Thailand. Association for Computational Linguistics.

Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. 2024b. Training-free long-context scaling of large language models. In Forty-first International Conference on Machine Learning.

Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu, and Lingpeng Kong. 2024c. Why does the effective context length of llms fall short? arXiv preprint arXiv:2410.18745.

Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, and Jian-Guang Lou. 2024d. Make your llm fully utilize the context. arXiv preprint arXiv:2404.16811.

Anthropic. 2023. Anthropic: Introducing claude 2.1.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations.

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. 2024. Longalign: A recipe for long context alignment of large language models. arXiv preprint arXiv:2401.18058.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. 2023. Open llm leaderboard.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. 2024. Unlimiformer: Long-range transformers with unlimited length input. Advances in Neural Information Processing Systems, 36.

Amanda Bertsch, Alex Xie, Graham Neubig, and Matthew R Gormley. 2023. It’s mbr all the way down: Modern generation techniques through the lens of minimum bayes risk. In Proceedings of the Big Picture Workshop, pages 108–122.

P.J. Bickel and K.A. Doksum. 1977. Mathematical Statistics: Basic Ideas and Selected Topics. Prentice Hall.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. 2024a. Teaching large language models to self-debug. In The Twelfth International Conference on Learning Representations.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2024b. LongloRA: Efficient fine-tuning of long-context large language models. In The Twelfth International Conference on Learning Representations.

Zhi Chen, Qiguang Chen, Libo Qin, Qipeng Guo, Haijun Lv, Yicheng Zou, Wanxiang Che, Hang Yan, Kai Chen, and Dahua Lin. 2024c. What are the essential factors in crafting effective long context multi-hop instruction datasets? insights and best practices. arXiv preprint arXiv:2409.01893.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2024. Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36.

Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. 2023. Longnet: Scaling transformers to 1,000,000,000 tokens. arXiv preprint arXiv:2307.02486.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. 2024. LongroPE: Extending LLM context window beyond 2 million tokens. In Fortyfirst International Conference on Machine Learning.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630.

Mara Finkelstein and Markus Freitag. 2024. MBR and QE finetuning: Training-time distillation of the best and most expensive decoding methods. In The Twelfth International Conference on Learning Representations.

Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. 2024. Data engineering for scaling language models to 128k context. arXiv preprint arXiv:2402.10171.

Deep Ganguli, Amanda Askell, Nicholas Schiefer, Thomas I Liao, Kamil˙e Lukoši¯ut˙e, Anna Chen, Anna Goldie, Azalia Mirhoseini, Catherine Olsson, Danny Hernandez, et al. 2023. The capacity for moral selfcorrection in large language models. arXiv preprint arXiv:2302.07459.

Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. 2024. How to train long-context language models (effectively). arXiv preprint arXiv:2410.02660.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. CRITIC: Large language models can self-correct with tool-interactive critiquing. In The Twelfth International Conference on Learning Representations.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. 2023. Reinforced selftraining (rest) for language modeling. arXiv preprint arXiv:2308.08998.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609– 6625.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. Orpo: Monolithic preference optimization without reference model. arXiv preprint arXiv:2403.07691, 2(4):5.

Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. 2024. V-STar: Training verifiers for self-taught reasoners. In First Conference on Language Modeling.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654.

Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2023. Large language models can self-improve. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1051–1068.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A Smith, Iz Beltagy, et al. 2023. Camels in a changing climate: Enhancing lm adaptation with tulu 2. arXiv preprint arXiv:2311.10702.

Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. 2023. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509.

Dongwei Jiang, Jingyu Zhang, Orion Weller, Nathaniel Weir, Benjamin Van Durme, and Daniel Khashabi. 2024. Self-[in] correct: Llms struggle with refining self-generated responses. arXiv preprint arXiv:2404.04298.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations.

Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, and Xia Hu. 2024. LLM maybe longLM: Selfextend LLM context window without tuning. In Forty-first International Conference on Machine Learning.

Greg Kamradt. 2023. Needle in a haystack - pressure testing llms.

Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, and Mohit Iyyer. 2024. One thousand and one pairs: A" novel" challenge for long-context language models. arXiv preprint arXiv:2406.16264.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Shankar Kumar and Bill Byrne. 2004. Minimum bayesrisk decoding for statistical machine translation. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLTNAACL 2004, pages 169–176.

Tian Lan, Wenwei Zhang, Chengqi Lyu, Shuaibin Li, Chen Xu, Heyan Huang, Dahua Lin, Xian-Ling Mao, and Kai Chen. 2024a. Training language models to critique with multi-agent feedback. arXiv preprint arXiv:2410.15287.

Tian Lan, Wenwei Zhang, Chen Xu, Heyan Huang, Dahua Lin, Kai Chen, and Xian-ling Mao. 2024b. Criticbench: Evaluating large language models as critic. arXiv preprint arXiv:2402.13764.

Mosh Levy, Alon Jacoby, and Yoav Goldberg. 2024. Same task, more tokens: the impact of input length on the reasoning performance of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Bangkok, Thailand. Association for Computational Linguistics.

Huayang Li, Pat Verga, Priyanka Sen, Bowen Yang, Vijay Viswanathan, Patrick Lewis, Taro Watanabe, and Yixuan Su. 2024a. A retrieve-then-reason framework for long-context question answering. arXiv preprint arXiv:2410.03227.

Mo Li, Songyang Zhang, Yunxin Liu, and Kai Chen. 2024b. Needlebench: Can llms do retrieval and reasoning in 1 million context window? arXiv preprint arXiv:2407.11963.

Yanyang Li, Shuo Liang, Michael Lyu, and Liwei Wang. 2024c. Making long-context language models better multi-hop reasoners. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2462– 2475.

Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. 2024. Jamba: A hybrid transformer-mamba language model. arXiv preprint arXiv:2403.19887.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252.

Zicheng Lin, Zhibin Gou, Tian Liang, Ruilin Luo, Haowei Liu, and Yujiu Yang. 2024. CriticBench: Benchmarking LLMs for critique-correct reasoning. In Findings of the Association for Computational Linguistics: ACL 2024, Bangkok, Thailand. Association for Computational Linguistics.

Yinhan Liu. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 364.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. Agentboard: An analytical evaluation board of multi-turn llm agents. arXiv preprint arXiv:2401.13178.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2024. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822.

Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9004–9017.

OpenAI. 2022. Chatgpt blog post. https://openai. com/blog/chatgpt.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. 2024. Automatically correcting large language models: Surveying the landscape of diverse automated correction strategies. Transactions of the Association for Computational Linguistics, 12:484–506.

Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason E Weston. 2024. Iterative reasoning preference optimization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. YaRN: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations.

Archiki Prasad, Weizhe Yuan, Richard Yuanzhe Pang, Jing Xu, Maryam Fazel-Zarandi, Mohit Bansal, Sainbayar Sukhbaatar, Jason Weston, and Jane Yu. 2024. Self-consistency preference optimization. Preprint, arXiv:2411.04109.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2024. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36.

Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Nan Wang, et al. 2024. jina-embeddings-v3: Multilingual embeddings with task lora. arXiv preprint arXiv:2409.10173.

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. 2024. You only cache once: Decoderdecoder architectures for language models. arXiv preprint arXiv:2405.05254.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554.

Roy Tromble, Shankar Kumar, Franz Josef Och, and Wolfgang Macherey. 2008. Lattice minimum bayesrisk decoding for statistical machine translation. In Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing, pages 620– 629.

Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, Jeffrey Hui, Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, et al. 2024. Michelangelo: Long context evaluations beyond haystacks via latent structure queries. arXiv preprint arXiv:2409.12640.

Jun Wang, Eleftheria Briakou, Hamid Dadkhahi, Rishabh Agarwal, Colin Cherry, and Trevor Cohn. 2024a. Don’t throw away data: Better sequence knowledge distillation. arXiv preprint arXiv:2407.10456.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023a. Plan-and-solve prompting: Improving zeroshot chain-of-thought reasoning by large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2609–2634.

Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al. 2024b. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa. arXiv preprint arXiv:2406.17419.

Tianduo Wang, Shichen Li, and Wei Lu. 2024c. Selftraining with direct preference optimization improves chain-of-thought reasoning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11917–11928.

Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. 2024d. Augmenting language models with long-term memory. Advances in Neural Information Processing Systems, 36.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023b. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Ian Wu, Patrick Fernandes, Amanda Bertsch, Seungone Kim, Sina Pakazad, and Graham Neubig. 2024. Better instruction-following through minimum bayes risk. arXiv preprint arXiv:2410.02902.

Yuhuai Wu, Markus Norman Rabe, DeLesley Hutchins, and Christian Szegedy. 2022. Memorizing transformers. In International Conference on Learning Representations.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. 2024. Self-evaluation guided beam search for reasoning. Advances in Neural Information Processing Systems, 36.

Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, Madian Khabsa, Han Fang, Yashar Mehdad, Sharan Narang, Kshitiz Malik, Angela Fan, Shruti Bhosale, Sergey Edunov, Mike Lewis, Sinong Wang, and Hao Ma. 2024. Effective long-context scaling of foundation models. In Proceedings of the 2024 Conference

of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Mexico City, Mexico. Association for Computational Linguistics.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. 2024a. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Guangyu Yang, Jinghong Chen, Weizhe Lin, and Bill Byrne. 2024b. Direct preference optimization for neural machine translation with minimum bayes risk decoding. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 391– 398.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380.

Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, and Furu Wei. 2024. Differential transformer. arXiv preprint arXiv:2410.05258.

Howard Yen, Tianyu Gao, and Danqi Chen. 2024a. Long-context language modeling with parallel context encoding. arXiv preprint arXiv:2402.16617.

Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izasak, Moshe Wasserblat, and Danqi Chen. 2024b. Helmet: How to evaluate long-context language models effectively and thoroughly. arXiv preprint arXiv:2410.02694.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason E Weston. 2024. Self-rewarding language models. In Forty-first International Conference on Machine Learning.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. STar: Bootstrapping reasoning with reasoning. In Advances in Neural Information Processing Systems.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a

machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800.

Dan Zhang, Sining Zhoubian, Yisong Yue, Yuxiao Dong, and Jie Tang. 2024a. Rest-mcts*: Llm selftraining via process reward guided tree search. arXiv preprint arXiv:2406.03816.

Jiajie Zhang, Zhongni Hou, Xin Lv, Shulin Cao, Zhenyu Hou, Yilin Niu, Lei Hou, Yuxiao Dong, Ling Feng, and Juanzi Li. 2024b. Longreward: Improving long-context large language models with ai feedback. arXiv preprint arXiv:2410.21252.

Peitian Zhang, Ninglu Shao, Zheng Liu, Shitao Xiao, Hongjin Qian, Qiwei Ye, and Zhicheng Dou. 2024c. Extending llama-3’s context ten-fold overnight. arXiv preprint arXiv:2404.19553.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024d. ∞Bench: Extending long context evaluation beyond 100K tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15262– 15277, Bangkok, Thailand. Association for Computational Linguistics.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. 2024a. Wildchat: 1m chatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations.

Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, Tongshuang Wu, and Jianshu Chen. 2024b. Fact-and-reflection (FaR) improves confidence calibration of large language models. In Findings of the Association for Computational Linguistics ACL 2024, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Tianyang Zhong, Zhengliang Liu, Yi Pan, Yutong Zhang, Yifan Zhou, Shizhe Liang, Zihao Wu, Yanjun Lyu, Peng Shu, Xiaowei Yu, et al. 2024. Evaluation of openai o1: Opportunities and challenges of agi. arXiv preprint arXiv:2409.18486.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations.

Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. 2024. PoSE: Efficient context window extension of LLMs via positional skip-wise training. In The Twelfth International Conference on Learning Representations.

### A Training Details

To support efficient fine-tuning for long-context scenarios, we implement sequence parallelization (Jacobs et al., 2023) with a parallel size of 8. Additionally, we utilize QLoRA (Dettmers et al., 2024) to reduce memory consumption during fine-tuning. The LoRA rank, alpha, and dropout are set to 128, 128, and 0.05, respectively, with all attention and feedforward linear layers designated as target modules. All models are fine-tuned for one epoch. The batch size, learning rate, and maximum sequence length are set to 8, 5e − 5, and 32K, respectively. The β for ORPO is configured to 0.1. All experiments are conducted on a computing setup with 8 × H100 GPUs.

### B Prompts

We provide the prompts for various prompting strategies (§2.1) in Tab. 9, and the prompts for the reference-free and reference-based self-evaluation strategies (§4.4) in Tab. 10.

#### Strategy Prompt

{context}

Default

{input}

{context}

Direct Answer

{input}

Let’s answer the question directly.

{context}

Think step-by-step (Kojima et al., 2022)

{input}

Let’s think step by step.

{context}

Fact-and-reflection (Zhao et al., 2024b)

{input} Let’s first identify the relevant information from the long context and list it. Then, carry out step-by-step reasoning based on that information, and finally, provide the answer.

{context}

Plan-and-solve (Wang et al., 2023a)

{input}

Let’s first understand the problem and devise a plan to solve it. Then, let’s carry out the plan and solve the problem step-by-step.

- Table 9: The prompts for various prompting strategies (§2.1), where {context} and {input} serve as placeholders for the long context and input query, respectively.

#### Strategy Prompt

[Context] {context}

[Question] {question}

Reference-free Self-Evaluation

[Predicted Response]

{prediction}

Please evaluate the correctness of the predicted response based on the context and the question. Begin your evaluation by providing a brief explanation. Be as objective as possible. After giving your explanation, you must rate the response on a scale from 1 to 5, following this format exactly: “[[rating]]”. For example, “Rating: [[3]]”.

Here is a question along with two responses: one is the reference response, and the other is the predicted response. Please determine whether the two responses provide the same answer to the question. Respond with “True” or “False” directly.

Reference-based Self-Evaluation

[Question] {question}

[Reference Response]

{reference}

[Predicted Response]

{prediction}

- Table 10: The prompts for the reference-free and reference-based self-evaluation strategies (§4.4), where {question}, {reference}, {prediction}, and {context} serve as placeholders for their respective elements.

