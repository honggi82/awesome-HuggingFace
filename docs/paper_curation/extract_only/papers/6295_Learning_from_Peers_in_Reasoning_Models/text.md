# arXiv:2505.07787v1[cs.CL]12May2025

## Learning from Peers in Reasoning Models

#### Tongxu Luo1∗ Wenyu Du2∗ Jiaxi Bi3 Stephen Chung2 Zhengyang Tang1

Hao Yang4 Min Zhang4 Benyou Wang1† 1The Chinese University of Hong Kong, Shenzhen 2DualityRL 3USTB 4Huawei tongxuluo@gmail.com wenyu.du@dualityrl.com wangbenyou@cuhk.edu.cn

### Abstract

Large Reasoning Models (LRMs) have the ability to self-correct even when they make mistakes in their reasoning paths. However, our study reveals that when the reasoning process starts with a short but poor beginning, it becomes difficult for the model to recover. We refer to this phenomenon as the “Prefix Dominance Trap”. Inspired by psychological findings that peer interaction can promote selfcorrection without negatively impacting already accurate individuals, we propose Learning from Peers (LeaP) to address this phenomenon. Specifically, every T tokens, each reasoning path summarizes its intermediate reasoning and shares it with others through a routing mechanism, enabling paths to incorporate peer insights during inference. However, we observe that smaller models sometimes fail to follow summarization and reflection instructions effectively. To address this, we fine-tune them into our LeaP-T model series. Experiments on AIME 2024, AIME 2025, AIMO 2025, and GPQA Diamond show that LeaP provides substantial improvements. For instance, QwQ-32B with LeaP achieves nearly 5 absolute points higher than the baseline on average, and surpasses DeepSeek-R1-671B on three math benchmarks with an average gain of 3.3 points. Notably, our fine-tuned LeaPT-7B matches the performance of DeepSeek-R1-Distill-Qwen-14B on AIME 2024. In-depth analysis reveals LeaP’s robust error correction by timely peer insights, showing strong error tolerance and handling varied task difficulty. LeaP marks a milestone by enabling LRMs to collaborate during reasoning. Our code, datasets, and models are available at https://learning-from-peers.github.io/.

### 1 Introduction

Large reasoning models (LRMs) [1–5] demonstrate strong performance on complex reasoning tasks. This success is largely attributed to their self-correction capability in test-time scaling [6–9], which consists of two emergent features: during generation, LRMs evaluate their current reasoning trajectories (self-verification) and may generate alternative ones (self-refinement) — a behavior often referred to as the “aha moment” [4]. However, recent research reveals notable limitations in this self-correction mechanism. For example, researchers [8] observe that LRMs frequently become stuck in incorrect reasoning paths that are rarely corrected. This raises an important question: Is the self-correction ability of LRMs truly as strong as we think?

To answer this question, we design a task where LRMs are required to solve problems starting from fixed beginnings, extracted from responses that lead to either correct or incorrect answers. If an LRM possesses strong self-correction ability, it should be able to recover from a low-quality beginning and still arrive at the correct answer. Surprisingly, many LRMs, including QwQ-32B [5], exhibit a

∗ Equal Contributions. † Corresponding Author.

Preprint. Under review.

performance drop of nearly 20% when starting from a poor beginning that accounts for only 15% of the average response length. This striking result suggests that the self-correction ability of LRMs is significantly weaker than previously thought. A short and flawed prelude can lead to a substantial degradation in final performance. We refer to this phenomenon as the “Prefix Dominance Trap”.

Recent findings [10–12] in psychology indicate that peer-based instruction effectively supports students in correcting misconceptions and improving performance with minimal negative impact. Inspired by this, we hypothesize that enabling LRMs to engage in peer learning may extend their selfverification capabilities beyond their own reasoning to include the reasoning of others. This broadens the search space for refinement, allowing LRMs to focus more on verifying existing reasoning paths rather than generating new ones from scratch. Since verification is generally less complex than generation, this shift reduces the cognitive burden and facilitates more effective reasoning refinement. Building on this insight, we propose Learning from Peers (LeaP) to improve reasoning in large models: instead of relying solely on individual reasoning paths, we enable LRMs to conduct cross-path interaction during parallel inference. Concretely, during generation, after every T tokens, each reasoning path summarizes its intermediate reasoning into a concise message, which is then shared with other paths through a heuristic routing mechanism. Simultaneously, each path receives summaries from its peers.

We first validate our hypothesis by placing LeaP under the “Prefix Dominance Trap” setting. Experimental results show that models with LeaP reduce the performance gap by nearly 10% compared to those without LeaP. This finding suggests that LeaP encourages each reasoning path to verify not only its own trajectory but also those of its peers, thereby decreasing the risk of overlooking correct solutions. To further assess the effectiveness of LeaP, we conduct comprehensive evaluations on AIME 2024 [13], AIME 2025 [14], the reference set of AIMO 2025 [15], and GPQA Diamond [16]. All reasoning models show significant improvement when using LeaP compared to those without LeaP, under comparable inference token budgets.

However, during these experiments, we observe that smaller models without further training, such

- as DeepSeek-R1-Distill-Qwen-7B [4], sometimes struggle to follow summarization and reflection instructions effectively (Case studies in Appendix D.2). To address this, we propose LeaP-T model series, including 1.5B, 7B, and 14B, which empowers LeaP with further training adaptation. Our experiments show significant improvements. For example, the LeaP-T-7B model achieves a Pass@1 score of 64.38 on AIME 2024, which is comparable to DeepSeek-R1-Distill-Qwen-14B [4]
- at 64.47. To summarize, our contributions are as follows:

- • Quantifying the Prefix Dominance Trap: We quantitatively validate the existence of the Prefix Dominance Trap in Large Reasoning Models (LRMs), showing that even short low-quality prefixes can lead to significant performance degradation.
- • A parallel inference method with cross-path interaction: We propose LeaP, which enables reasoning paths to communicate during inference. With LeaP, QwQ-32B gains nearly 5 points on average over the baseline and even surpasses DeepSeek-R1-671B [4] on various benchmarks. We also conduct a comprehensive analysis of LeaP from multiple perspectives, including sensitivity to communication and robustness under varying conditions.
- • A series of trained models: We train a series of models that are adapted to the LeaP framework and release them as open-source resources to facilitate future research.

### 2 Enabling Cross-path Interaction in Parallel Inference: LeaP

#### 2.1 Motivation: Prefix Dominance Trap

To assess the self-correction of LRMs, we introduce a task where LRMs are required to solve problems starting from fixed beginnings. If an LRM possesses strong correction abilities, it should more easily recover from reasoning paths that may lead to incorrect answers, thereby generating correct responses and resulting in a narrower accuracy gap with no-beginning.

Specifically, we utilize DeepSeek-R1-Distill-Qwen series [4] and QwQ-32B [5] to generate 32 responses per question in AIME 2024 [13]. For each model, we randomly select 10 incorrect responses from 10 distinct questions containing both correct and incorrect outputs. We retain the first

- 15% of tokens from each selected response as the prefix. We then generate 16 continuations for each prefix using the original LRMs with temperature τ = 1 to encourage diversity. To assess LRM self-

correction, we compute the average accuracy gap PG between these constrained and unconstrained generations. As shown in Figure 1, we report Pass@1 performance on AIME 2024 subsets (different for each model). Surprisingly, all selected LRMs exhibit a substantial performance drop of nearly 20% when the first 15% tokens are fixed from incorrect responses. This result highlights that LRM self-correction ability is limited. We term this phenomenon the “Prefix Dominance Trap”.

#### 2.2 Methodology: Cross-path Interaction in Parallel Inference

To ameliorate “Prefix Dominance Trap”, we introduce Learning from Peers (LeaP), a novel inference-time strategy that draws inspiration from collaborative learning paradigms observed in psychology [10–12]. Our central hypothesis is that enabling LRMs to engage in structured peer interaction can extend their self-verification capabilities. Instead of solely relying on individual trial-and-error, paths can leverage the reasoning processes of others. This paradigm broadens the search space for refinement, crucially shifting the LRM’s focus from the demanding task of path generation towards the often less complex task of verifying and refining existing diverse paths. This shift aims to reduce the cognitive burden and enhance the overall effectiveness of reasoning refinement.

With beginnings No beginnings

|18.95<br><br>19.88 18.75<br><br>20<br><br>30<br><br>40<br><br>50<br><br>60<br><br>70<br><br>Distill-7B Distill-14B QwQ-32B<br><br>Pass@1(%)<br><br>|
|---|

Pass@1(%)

Figure 1: The results of starting with bad beginnings.

[Figure 1]

- Figure 2: The illustration of (a) Independent Reasoning and (b) the proposed method Learning from Peers (LeaP). In independent reasoning, multiple paths are generated independently in parallel. In contrast, LeaP inserts a LeaP block into reasoning path, encouraging the model to learn from peers.

Overview. As depicted in Figure 2b, LeaP distinguishes itself from traditional independent reasoning (Figure 2a) by strategically inserting LeaP blocks into the parallel inference process. These blocks serve as dedicated modules for cross-path communication and collaborative refinement. Each LeaP block orchestrates a two-stage process: (1) Summarization, where each reasoning path concisely articulates its current state, and (2) Routing, where these distilled insights are efficiently disseminated to peer paths. An example of how peer insights can trigger self-correction is provided in Figure 3.

[Figure 2]

- Figure 3: An example of how LeaP enables communication between path i and j. Text in red indicates the current path is incorrect. Text in green shows a correct summary received from a peer.

- 1. Summarization stage: generating shareable insights. Effective peer learning requires efficient insight sharing. The first stage in a LeaP block is Summarization. Each LRM path condenses its current approach, key insights, and intermediate results into a concise summary. This summary is strictly limited to 256 tokens to maintain token efficiency. A prompt, comprising a dynamically selected summary trigger and a summary template, directs this summarization. These elements are randomly chosen from predefined lists (see Appendix A), promoting variability in summary expression while ensuring core information capture. Once these concise summaries are generated by each path, the next step is to distribute them among peers. This is handled by the Routing stage.
- 2. Routing stage: disseminating peer insights. Although summaries provide condensed information, exposing each path to all peer summaries is overwhelming and token-inefficient, especially with numerous parallel paths. The Routing stage mitigates this by selecting which peer summaries a given path receives. For N reasoning paths and their summaries {s1,s2,...,sN}, a routing function R selects for each path i a subset of peer summaries Ci ⊂ {sj | j ̸= i}. We explore three routing mechanisms, each employing a different heuristic to foster effective collaboration:

- • Dispersed routing: Grounded in the intuition that diverse insights are crucial for breaking out of erroneous reasoning patterns and discovering novel solutions, Dispersed Routing prioritizes summaries that are least similar to the receiving path’s own summary:

Ci = Bottom-k ((similarity(si,sj) | j ̸= i)) (1)

- • Clustered routing: Conversely, the assumption here is that paths with similar reasoning are likely converging on a viable solution trajectory. Clustered Routing selects the top-k most similar summaries, facilitating collaboration among paths that are already aligned:

Ci = Top-k ({similarity(si,sj) | j ̸= i}) (2)

- • Hybrid routing To achieve a synthesis of the above strategies, Hybrid Routing seeks to balance the benefits of collaborative reinforcement with the injection of diverse perspec-

tives. It selects k2 most similar summaries (exploitation) and k2 most dissimilar summaries (exploration):

k 2

Ci = Cisim ∪ Cidis, where |Cisim| = |Cidis| =

(3)

To quantify the similarity between summaries si and sj for these routing mechanisms, we employ the normalized Levenshtein similarity [17]. The similarity score similarity(si,sj) is defined as:

Dlev(si,sj) max(|si|,|sj|)

(4)

similarity(si,sj) = 1 −

where Dlev(si,sj) is the Levenshtein distance [17] between si and sj, and |si| denotes the length of the string si. This score, ranging from [0,1], provides a normalized measure where higher values indicate greater textual similarity.

#### 2.3 Can LeaP help in “Prefix Dominance Trap”?

With beginnings No beginnings

|10.16<br><br>7.81<br><br>11.80<br><br>20<br><br>30<br><br>40<br><br>50<br><br>60<br><br>70<br><br>Distill-7B Distill-14B QwQ-32B<br><br>Pass@1(%)<br><br>|
|---|

To verify if LeaP can help overcome this “Prefix Dominance Trap”–the phenomenon where short, flawed beginnings significantly hinder recovery–we assess reasoning models with LeaP under the setting in Section 2.1. As shown in Figure 4, we present Pass@1 results of models with LeaP on the AIME 2024 subset from Section 2.1. LeaP consistently reduces the performance gap across all three model sizes. For example, on DeepSeek-DistillQwen-14B, LeaP narrows the gap from 19.88 to 7.81. Appendix D.1 provides case studies comparing the baseline with LeaP. We also conduct experiments starting from good beginnings, where truncated initial tokens likely lead to correct answers. As presented in Appendix B, when the beginning is good, our method also performs better than standard independent reasoning. These results on both bad and good beginnings demonstrate that LRMs leverage strong self-verification to assess expanded reasoning paths from peers, improving LRM self-correction.

Pass@1(%)

Figure 4: The results of starting with bad beginnings for models with LeaP.

### 3 Comprehensive Evaluation of LeaP

To assess LeaP comprehensively, we evaluate the LeaP on four reasoning benchmarks. Section 3.1 describes the evaluation setup and implementation details. In Section 3.2, we report results showing the effectiveness of LeaP across four benchmarks.

#### 3.1 Evaluation Setup

Benchmarks and metrics. We evaluate our method on challenging benchmarks. Specifically, we use competition-level mathematics datasets: AIME 2024 [13], AIME 2025 [14], and the AIMO 2025 [15] reference set. AIME 2024 [13] and AIME 2025 [14] each contain 30 problems; AIMO 2025 [15] contains 10 problems. Additionally, we test on the GPQA [16] Diamond subset (198 questions), the most difficult GPQA [16] portion, requiring PhD-level domain-specific knowledge in physics, chemistry, and biology. For evaluation, we adopt Pass@1 and Cons@N [4]. We generate N responses

per question using a non-zero temperature. Pass@1 is calculated as: Pass@1 = N1 Ni=1 Ii, where Ii denotes i-th response correctness. We also report consensus results by voting from N responses, termed Cons@N.

Models and baselines. We select LRMs with various sizes from 7B to 32B, including the 7B and 14B versions of DeepSeek-R1-Distill-Qwen [4] and QwQ-32B [5]. We run these models using the vllm framework [18]. We use independent reasoning as the baseline for Pass@1, and adopt Majority Voting [19] and Shortest Majority Voting [8] as baseline methods for Cons@N.

Hyperparameters. Following the settings reported in QwQ-32B [5], we use a sampling temperature of 0.6, a Top-p value of 0.95, and a Top-k value of 40 to generate N responses. Typically, N is set to 8 for GPQA [16] tasks and 32 for math tasks [13–15], depending on the size of the benchmark. For the maximum tokens, we use 16,384 for the 7B and 14B models, and 32,768 for the 32B model.

#### 3.2 Results

The results are presented in Table 1. Overall, LRMs with LeaP significantly outperform the baselines. For example, in terms of average performance across four benchmarks, our method using DeepSeekR1-Distill-Qwen-7B with top-4 Dispersed routing exceeds the baseline by 6.49. Similarly, on DeepSeek-R1-Distill-Qwen-14B, it surpasses the baseline by 6.08. Particularly, QwQ-32B with top-4 Dispersed routing even beats R1-671B in all three math datasets.

We also observe performance improvements when transferring from Top-2 to Top-4 peer summaries. This trend suggests that as the number of top-k peer summaries increases, the LRM benefits thinking paths from a greater number of peers. Among three routing strategies, Dispersed, and Hybrid clearly outperform the Clustered approach. This is expected, as receiving similar summaries from peers limits the overall diversity of trajectories for self-verification. In contrast, the diverse or complementary

perspectives provided by Dispersed and Hybrid routing introduce a broader range of reasoning paths. To provide a comprehensive evaluation, we also report the Cons@N results in Appendix C.2

- Table 1: We evaluate the baseline and LeaP on AIME 2024 [13], AIME 2025 [14], AIMO 2025 [15], and GPQA Diamond [16]. Our method significantly outperforms the baseline. We also report DeepSeek-R1-671B results to demonstrate our method’s advantages. Since DeepSeek-R1-671B lacks an official AIMO 2025 score, we evaluate it on the official website, repeating inference four times. R1-7B and R1-14B represent the 7B and 14B versions of DeepSeek-R1-Distill-Qwen, respectively.

Clustered Hybrid Dispersed

Benchmarks Models Baseline

R1-671B Top-2 Top-4 Top-2 Top-4 Top-2 Top-4

- AIME 2024

R1-7B 51.35 56.15 59.27 59.17 61.67 60.31 60.52

R1-14B 64.47 72.08 71.04 71.77 74.48 71.15 77.29 79.8 QwQ-32B 79.69 78.96 81.56 81.88 81.67 81.56 85.83

AIMO 2025

R1-7B 37.50 37.81 39.06 41.88 40.31 39.06 45.00

R1-14B 46.87 51.50 55.31 51.88 52.19 49.38 51.25 65.0 QwQ-32B 63.75 63.75 65.63 64.06 64.69 67.19 67.19

- AIME 2025

R1-7B 37.81 36.98 37.50 37.19 39.27 40.93 38.44

R1-14B 48.64 50.42 51.88 54.38 54.38 50.31 54.17 70.0 QwQ-32B 68.13 68.85 71.35 71.04 72.50 70.83 71.67

R1-7B 46.91 52.97 52.65 51.83 53.47 53.28 55.56

GPQA

71.5 Diamond

R1-14B 53.47 54.80 58.33 54.42 57.89 55.68 55.05 QwQ-32B 58.00 65.03 65.28 61.87 65.21 66.16 63.32

R1-7B 43.39 45.98 47.12 47.52 48.68 48.40 49.88

Avg.

R1-14B 53.36 57.20 59.14 58.11 59.74 56.63 59.44 71.58 QwQ-32B 67.39 69.15 70.96 69.71 71.02 71.44 72.00

Efficiency on generation token. We analyze LeaP’s inference efficiency based on the total number of tokens. Results for QwQ-32B [5] are presented in Figure 5a; Appendix C.3 contains additional results. LeaP does not generate significantly more tokens than the baseline; in fact, its average response length is sometimes shorter. We further analyze “aha” moments using keywords from previous studies [20–22] in Figure 5b. LeaP exhibits 16.4% fewer “aha” moments than the baseline across three math benchmarks. The number of “aha” moments for Top-4 settings is consistently lower than for Top-2 settings. These findings indicate that receiving peer opinions and results reduces the model’s need for reflection.

|0<br><br>3000<br><br>6000<br><br>9000<br><br>12000<br><br>15000<br><br>18000<br><br>MATH Tasks GPQA Diamond<br><br>Tokens<br><br>Baseline Dispersed-Top2 Dispersed-Top4<br><br>Hybrid-Top2 Hybrid-Top4<br><br>|
|---|

|0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>120<br><br>140<br><br>MATH Tasks GPQA Diamond<br><br>”Aha”moments<br><br>Baseline Dispersed-Top2 Dispersed-Top4<br><br>Hybrid-Top2 Hybrid-Top4<br><br>|
|---|

###### ”Aha”moments

(a) Number of Tokens

(b) Number of “Aha” moments

Figure 5: We illustrate the average number of tokens and “Aha” moments on QwQ-32B. Our method produces a comparable number of tokens to the baseline, while yielding fewer “Aha” moments.

### 4 Empowering LeaP with Further Training Adaptation: LeaP-T

By analyzing the responses in Section 3, we observe that reasoning models with small size, such as DeepSeek-R1-Distill-Qwen-7B, sometimes fail to summarize and reflect on peers’ summaries effectively. The case studies can refer to Appendix D.2. To this end, we introduce LeaP-T model series, where we attempt to alleviate this problem through supervised fine-tuning.

#### 4.1 Experimental Setup

We use approximately 1,000 AIME problems from 1984 to 2023 [13, 14] as source data. We synthesize responses by applying LeaP to DeepSeek-R1-Distill-Qwen-32B [4] and filter suitable responses using a rule-based selection mechanism. We use supervised finetuning to train our LeaP-T models, starting from the 1.5B, 7B, and 14B versions of DeepSeek-R1-Distill-Qwen [4]. Specifically, a response is selected if its final answer is correct and the length of all summaries is less than 256 tokens. We also generate SFT data without LeaP to train baseline models. Training hyperparameters are in Appendix E. We use the hyperparameters and metrics setting from Section 3.1. For LeaP, we use the Top-4 Dispersed routing setting.

##### 4.2 Results Table 2: Evaluation of our LeaP-T from 1.5B to 14B on three math benchmarks.

AIME 2024 AIME 2025 AIMO 2025 Avg.

Models

Pass@1 Cons@32 Pass@1 Cons@32 Pass@1 Cons@32 Pass@1 Cons@32

R1-1.5B 32.00 50.00 24.69 30.00 14.00 30.00 23.56 36.67 + SFT 31.04 56.67 23.23 36.67 15.31 30.00 23.19 41.11 + LeaP 34.90 46.67 26.46 30.00 15.63 30.00 25.66 35.56 LeaP-T-1.5B 37.08 56.67 26.67 40.00 20.31 40.00 28.02 45.56

R1-7B 51.35 73.33 37.81 50.00 37.50 50.00 42.22 57.78 + SFT 51.56 80.00 35.73 53.33 33.75 40.00 40.35 57.78 + LeaP 60.52 76.67 38.44 53.33 45.00 50.00 47.99 60.00 LeaP-T-7B 64.38 80.00 41.25 56.67 44.06 60.00 49.90 65.56

R1-14B 64.47 80.00 48.64 60.00 46.87 60.00 53.33 66.67 + SFT 65.63 83.33 46.88 63.33 45.63 60.00 52.71 68.89 + LeaP 77.29 83.33 54.17 60.00 51.25 60.00 60.90 67.78 LeaP-T-14B 76.46 83.33 54.27 70.00 52.50 60.00 61.08 71.11

- Table 2 presents Pass@1 and Cons@32 results from DeepSeek-R1-Distill-Qwen-1.5B to DeepSeek-R1-DistillQwen-14B on three math benchmarks. The results show consistent improvement across various model sizes. For example, LeaP-T-1.5B outperforms DeepSeek-R1-DistillQwen-1.5B by 4.45 in average Pass@1 and 8.89 in average Cons@32. The 7B and 14B models exhibit a similar trend. Additionally, comparing DeepSeek-R1-Distill-Qwen models with LeaP to LeaP-T models, we observe that LeaP-T models demonstrate stronger performance. To exclude the impact of distilling with a 32B model, we conduct distillation using SFT without LeaP. Surprisingly, models distilled by DeepSeek-R1-Distill-Qwen-32B do not show improvements, and in some cases, show a marginal decrease in performance. This is reasonable as the DeepSeekR1-Distill-Qwen 1.5B, 7B, and 14B models are already well-trained by distilling from R1-671B. Thus, distilling them with DeepSeek-R1-Distill-Qwen-32B does not yield additional benefits. In other words, our training incorporates the Learning from Peers paradigm, rather than simply transferring knowledge through distillation.

N=6

R1-7B

| |
|---|

N=5

72.5

LeaP-T-7B

| |
|---|

N=4

N=7

70.0

| |
|---|

N=6

N=3

N=8

67.5

###### Accuracy(%)

65.0

N=5

N=2

62.5

N=4

| |
|---|

N=3

60.0

57.5

N=2

55.0

16k 24k 32K 40K 48K 56k 64k 72K 80K

Avg Total Compute (Thousands of Tokens)

Figure 6: Accuracy vs. Total tokens on AIME 2024. LeaP-T-7B achieves higher accuracy compared to independent parallel reasoning under similar budgets.

We also assess the efficient test-time scaling ability of LeaP-T by comparing it with the baseline under the 7B size on AIME 2024 [13]. As shown in Figure 6, we increase inference tokens by parallel generating multiple responses. For each point in Figure 6, we report average results from four repeated runs. It is clear that our LeaP-T-7B consistently outperforms DeepSeek-R1-Distill-Qwen-7B. This result demonstrates that our method scales more efficiently during test-time. We also compare our LeaP-T with MoA [23]; details can be found in Appendix F.

### 5 In-depth Analysis of LeaP

To gain a deeper understanding of how and why LeaP works, we conduct a comprehensive analysis from three aspects: communication sensitivity (Section 5.1), robustness (Section 5.2), and human

verification (Section 5.3). This analysis not only guides the practical deployment of LeaP, but also sheds light on the inner workings of learning from peers.

#### 5.1 Sensitivity Analysis of Communication

To better deploy LeaP in practice, we investigate the sensitivity of communication on several factors, including granularity T (Section 5.1.1), traffic (Section 5.1.2), evolution tendency of types (Section 5.1.3), and position (Section 5.1.4).

#### 5.1.1 On Communication Granularity T

|Tokens baseline Tokens Pass@1 baseline Pass@1| |Tokens baseline Tokens Pass@1 baseline Pass@1|
|---|---|---|
|70% 12000 64.48%<br><br>10000 60.31% 65% 9045 8000 55.94% 60%| |80% 12000 73.96% 73.13%<br><br>71.15% 69.38% 75%<br><br>10000<br><br>9869<br><br>70% 8000|
|53.44% 6000<br><br>55% 4000 51.35%| |65% 6000 64.47% 4000 60%|
|50% 2000| |2000 55%|
|0 45%<br><br>2048 3072 4096 6144| |0 50%<br><br>2048 3072 4096 6144|
|Interval Tokens| |Interval Tokens|

Pass@1

Pass@1

Tokens

Tokens

(a) 7B

(b) 14B

Figure 7: Pass@1 and total tokens on AIME 2024 for 7B and 14B models with LeaP, evaluated across interval tokens of LeaP from 2048 to 6144.

To study the impact of communication granularity, we fix the top-2 Dispersed routing strategy and vary the interval T between two consecutive LeaP blocks on the 7B and 14B versions of DeepSeekR1-Distill-Qwen. As shown in Figure 7, increasing the granularity of communication between reasoning paths slightly decreases performance on AIME 2024. Specifically, the Pass@1 score drops from 64.48 to 53.44 for the 7B model, and from 73.96 to 69.38 for the 14B model. Meanwhile, fewer tokens are consumed as communication becomes less sparse. This is expected, since each LeaP block involves summarization and message routing, which costs additional tokens. Overall, frequent communication (i.e., smaller T) slightly improves performance at the expense of more token usage, highlighting a trade-off between accuracy and efficiency.

#### 5.1.2 On the Communication Traffic (Top-k)

For communication traffic, a natural assumption is that increasing it—i.e., routing more summaries from other reasoning paths—leads to better performance, as the model gains access to more diverse information. However, our experiments reveal this intuition does not always hold. To investigate this, we vary the number of routed summaries k from 1 to 16 under the Dispersed routing strategy, while fixing communication granularity T to 4K tokens. As shown in Figure 8, we evaluate DeepSeek-R1-DistillQwen-14B on AIME 2024 and observe a clear trend: the Pass@1 score rises sharply from Top-1 to Top-4, reaching its peak at k = 4. Surprisingly, performance then declines and fluctuates as k increases beyond 4. This finding suggests that larger communication traffic is not always better. When k is too small, the model lacks sufficient information, leading to limited perspectives. On the other hand, a large k introduces excessive summaries that may overwhelm the reasoning process, especially under constrained token budgets. The resulting noise can impair coherence and hinder solution quality. Overall, a trade-off exists between communication richness and cognitive overload, with Top-4 striking the most effective balance.

77.3%

With LeaP

Without LeaP

76%

74.8%75.0% 74.0%

74.4%74.2%

73.9%

74%

72.9%

73.1%

72.1% 72.2%

72%

Pass@1

71.2%

70%

68%

67.4%

66%

64.5%

64%

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

Top-k

Figure 8: Performance of the 14B model with LeaP on AIME 2024, evaluated across the top-k from 1 to 16.

#### 5.1.3 On Evolution Tendency of Communication Types

Having established the optimal communication traffic (Top-k) as discussed previously, we further investigated the evolution of communication types during reasoning to more comprehensively understand the dynamic characteristics of peer learning. For this analysis, we fixed communication granularity at T = 4K tokens and used the top-4 Dispersed routing strategy. Communication outcomes were categorized as: Consistent (paths align with peers before and after communication); Unaffected (paths differ from peers and remain unchanged); or Influenced (paths initially differ but adjust after receiving peer summaries). GPT-4o [24] annotated these types, with human validation on 10% of samples (approximately 120) confirming over 95% model-human agreement. Figure 9 shows this type distribution for QwQ-32B on AIME 2024 (from 4K to 24K tokens). Unaffected cases increase significantly later, indicating reduced communication effectiveness as reasoning progresses. Conversely, the Influenced ratio, after rising from 0.13 (4K) to 0.18 (8K), steadily declines to 0.06 (24K). This suggests communication is most impactful in early to mid-stage reasoning, with limited influence later.

100

| |12.9%<br><br>30.3%<br><br>56.8%<br><br>17.9%<br><br>32.7%<br><br>49.3%<br><br>15.1%<br><br>33.5%<br><br>51.4%<br><br>11.1%<br><br>27.4%<br><br>61.5%<br><br>12.7%<br><br>40.2%<br><br>47.1%<br><br>5.9%<br><br>67.7%<br><br>26.5%<br><br>Influenced<br><br>Unaffected Consistent<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80

Percentage(%)

60

40

20

0

4K 8K 12K 16K 20K 24K

Position

Figure 9: Communication types distribution of QwQ-32B on AIME 2024 at different positions, categorized into Consistent, Unaffected, and Influenced.

#### 5.1.4 On the Communication Position (When only Communicating Once)

Setting on a new LeaP variant in single communication. To further analyze sensitivity to communicating at different positions, we introduce a simplified variant where communication occurs only once during the reasoning process. This setting allows us to investigate the impact of communication position: whether it is more beneficial to communicate early, in the middle, or late. In Figure 10, we report results of DeepSeek-R1-Distill-Qwen-14B [4] on

|| |
|---|
| |
| |
|69.48%|
|68.23%|
| |
|66.25% 66.25%<br><br>65.94%|
|65.33%|
| |
| |
<br><br>71.15%<br><br>64.79%<br><br>64%<br>65%<br>66%<br>67%<br>68%<br>69%<br>70%<br>71%<br>72%<br><br><br>2K 4K 6K 8K 10K 12K<br><br>Pass@1<br><br>Position<br><br>Partial Full No<br><br>|
|---|

- AIME 2024 [13] under this single communication setting. We observe that performance improves when communication occurs early in the reasoning process, increasing from 66.25 at 2K tokens to a peak of 69.48 at 4K tokens. However, beyond this point, performance declines, dropping to 65.33 by the end. These results suggest that early-stage communication is more effective than latestage interaction. These findings are consistent with our previous conclusions from Section 5.1.3. In particular, enabling a single LeaP block early in reasoning already yields substantial improvements over the baseline. We refer to this variant as LeaP in Single Communication.

Table 3: The Pass@1 results of independent reasoning and LeaP in singler-generation on AIME 2024,

- AIME 2025, AIMO 2025 and GPQA Diamond for 14B model. Benchmarks R1-14B T = 2K T = 4K

Figure 10: Performance of partial LeaP on the 14B model in inference, evaluated across the positions of LeaP block ranging from 2K to 12K tokens

- AIME 2024 64.47 66.25 68.13 AIMO 2025 46.87 49.38 50.94

- AIME 2025 48.64 45.83 47.29 GPQA Diamond 53.47 55.11 57.42 Avg. 53.86 54.14 55.95

Results. We evaluate a simplified LeaP variant that retains only the first LeaP block at T tokens and continues generation along a single path. This mirrors how humans often work: gathering ideas early, then independently verifying them. As shown in Table 3, this single communication variant outperforms DeepSeek-R1-Distill-Qwen-14B. For example, on AIME 2024, it improves Pass@1 from 64.47 to 68.13; on GPQA Diamond, from 53.47 to 57.42. Even with just one block, it achieves a higher average Pass@1 (55.95 vs. 53.86), showing that early peer exposure effectively guides reasoning. Gains are most notable in math and multi-hop tasks, where early external signals

reduce error accumulation. This variant also balances performance and efficiency, making it suitable for single-path scenarios. We further explore different hyperparameters on AIME 2024 (Table 4). Increasing the interval T from 2K to 4K improves accuracy (e.g., 67.92 to 69.48 at k = 8), and using more peers (e.g., k = 8 vs. k = 2 at T = 4K) also yields better results.

- Table 4: The Pass@1 results for varying k values and position of LeaP block on AIME 2024 with the 14B model in single communication LeaP.

k = 2 k = 4 k = 6 k = 8 k = 16 (Top-2) k = 16 (Top-4)

T = 2k 65.21 66.25 65.73 67.92 66.46 66.86 T = 4k 65.73 68.13 68.85 69.48 67.92 69.27

#### 5.2 Post-hoc Robustness Analysis of LeaP

To better understand the reliability of LeaP in practical scenarios, we conduct a post-hoc robustness analysis. Specifically, we analyze whether the benefits of LeaP persist under two settings: (1) Vary the error path in peers (Section 5.2.1), and (2) Vary the difficulty levels (Section 5.2.2).

#### 5.2.1 Robustness on Error Tolerance

A common concern for LeaP is that low-quality paths may mislead others, especially when the majority of them are incorrect. Intuitively, one might expect that effective collaboration in LeaP requires a high proportion of correct reasoning paths; otherwise, noise from incorrect paths could dominate interaction and degrade performance.

84.4%

LeaP with mixed beginnings Baseline with bad beginning Baseline with good beginning

82.5%

80%

74.4%

72.5%

70.0%

68.8% 68.8%

70%

65.6%

Pass@1

60%

49.4%

50%

To test this hypothesis and assess LeaP’s error tolerance, we design an experiment by varying the proportion of good beginnings. Specifically, we select 10 questions with good and bad beginnings using the first 30% tokens of responses for DeepSeek-R1-Distill-Qwen-14B by the method in Section 2.1. For each question, we construct 16 mixed initial responses, varying the percentage of good beginnings from 0As shown in Figure 11, the results reveal a surprising trend. Across all configurations, the model with LeaP consistently outperforms the baseline, even when none of the beginnings are correct. For instance, at 0% good beginnings, LeaP achieves a Pass@1 score of 41.88, far exceeding the baseline score of 28.75. Even more remarkably, at just 43% good beginnings, LeaP surpasses the baseline’s performance when all beginnings are good. These results challenge the assumption that effective collaboration in LeaP requires mostly correct paths. Instead, they demonstrate that LeaP is highly robust: peers can distill useful signals from noisy summaries. This can be attributed to the strong self-verification ability of LRMs.

41.9%

40%

30%

28.7%

0% 6% 12% 18% 25% 31% 37% 43% 50%

Ratio of good beginning

Figure 11: The Pass@1 under various ratios of good beginnings, where the length of the beginnings is 30% of the average response length.

#### 5.2.2 Robustness at Various Difficulty Levels

Another concern is whether LeaP maintains robustness across various difficulty levels. It is possible that LeaP performs well primarily because the benchmark contains a certain proportion of simple questions. But can LeaP still provide benefits when problems become significantly more challenging?

To answer this question, we conduct a detailed analysis across different difficulty levels. Specifically, for each model, we first divide questions in AIME 2024 into five difficulty levels by counting correct responses in the baseline (independent 32 parallel reasoning without peers). The difficulty levels are as follows: “Very Easy” corresponds to 32 correct responses, “Easy” includes 25 to 31 correct, “Medium” consists of 9 to 24 correct responses, “Hard” includes 1 to 8 correct, and “Very Hard” refers to 0 correct responses. Then, we compute accuracy and token usage under each level. To better analyze tokens used solely for reasoning, we exclude summaries in the LeaP blocks collected from other paths. As shown in Figure 12, we report results for DeepSeek-Distill-Qwen-7B and DeepSeekDistill-Qwen-14B. We find a consistent trend: LeaP improves accuracy across all difficulty levels, including those where the baseline model completely fails. This result is surprising and encouraging. Even for “Very Hard” questions—where the baseline fails to solve a single instance—LeaP achieves

meaningful gains. This suggests that LeaP not only amplifies partial correctness but can also help recover from complete failure, enabling models to solve problems that were previously unreachable. In addition, after removing tokens attributed to peer summaries, LeaP often consumes fewer tokens than the baseline. Combined with our earlier finding in Section 3.2 that LeaP reduces the number of “Aha” moments, this supports the view that LeaP helps the reasoning model reach consensus earlier, thereby reducing overthinking. We include additional results for the LeaP-T series in Appendix G.2, which further confirm the robustness of LeaP across varying levels of difficulty.

| |Baseline Tokens<br><br>Ours Tokens<br><br>Baseline Accuracy<br><br>Ours Accuracy| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

100

14000

12000

80

10000

Accuracy(%)

60

###### Tokens

8000

6000

40

4000

20

2000

0

0

Very Easy Easy Medium Hard Very Hard

| |Baseline Tokens<br><br>Ours Tokens<br><br>Baseline Accuracy<br><br>Ours Accuracy| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

100

17500

15000

80

12500

Accuracy(%)

60

###### Tokens

10000

7500

40

5000

20

2500

0

0

Very Easy Easy Medium Hard Very Hard

(a) DeepSeek-R1-Distill-Qwen-7B

(b) DeepSeek-R1-Distill-Qwen-14B

Figure 12: The Pass@1 and average token distribution across different difficulty levels, from Very Easy to Very Hard, for the 7B and 14B versions of DeepSeek-R1-Distill-Qwen.

#### 5.3 Human Verification on LeaP

While results across various benchmarks provide consistent evidence that LeaP significantly improves reasoning ability, they do not always reveal underlying behavioral changes induced by LeaP. To gain deeper insight, we perform a detailed human evaluation by comparing QwQ-32B outputs with and without LeaP on a specific example—the 11th problem from AIME 2024. As shown in Figure 13, the baseline model (without LeaP) produces correct answers in only 25.00% of cases (8 out of 32). In contrast, when using LeaP, the proportion of correct responses rises significantly to 62.50% (20 out of 32), confirming the substantial benefit of learning from peers. To better understand the source of this improvement, we categorize responses into four types: Correct → Correct, Incorrect → Correct, Incorrect → Incorrect, and Correct → Incorrect. Remark-

| |
|---|

| |
|---|

| |
|---|

Correct

Incorrect

Correct->Correct

| |
|---|

| |
|---|

| |
|---|

Incorrect->Correct

Incorrect->Incorrect

Correct->Incorrect

|| | | | |37.50%| |
|---|---|---|---|---|---|
| |75.00%| | | | |
| | | | | | |
| | | | |40.62%| |
| | | | | | |
| | | | | | |
| | | | | | |
| |25.00%| | |21.88%| |
| | | | | | |
<br><br>0<br><br>0.2<br><br>0.4<br><br>0.6<br><br>0.8<br>1<br><br><br>Baseline Learning from peers<br><br>Percentage|
|---|

Figure 13: Case study comparing QwQ32B responses with and without LeaP on the 11th problem of AIME 2024.

ably, we observe that 40.62% of responses fall into the Incorrect → Correct category. This indicates that many reasoning paths are initially flawed but are successfully repaired after communicating with other peers. Even more telling, no responses fall into the Correct → Incorrect case, suggesting that peer communication rarely disrupts already correct reasoning. Instead, it primarily serves as a corrective mechanism that identifies and fixes errors. Case studies are provided in Appendix D.3

### 6 Related Work

#### 6.1 Self-Correction Bottleneck

Recent studies show that while current Large Reasoning Models can generate lengthy reasoning chains, their performance does not consistently improve with increasing chain length [8]. In fact, longer reasoning chains often result in lower accuracy, which contradicts the assumption that extended reasoning improves problem-solving capabilities. Recent study [25] further investigates the reasoning behavior of DeepSeek-R1, highlighting its tendency to dwell on previously explored problem formulations. This behavior hinders further exploration and leads to suboptimal reasoning paths. In this paper, we reaffirm their findings and present an effective method to address this issue.

#### 6.2 (Interactive) Parallel Inference in LLMs

Parallel inference [26, 6, 7] enables LLMs to generate multiple reasoning paths simultaneously and aggregate them into a final answer. Self-Consistency (Majority Voting) [8, 7, 19, 27–30] selects the final answer by voting over candidate responses and choosing the one with the highest number of votes. Best-of-N [31–36] improves response quality by generating multiple candidates and selecting the one with the highest reward. Another important direction involves interaction among multiple LLMs within collaborative frameworks during parallel inference. Multi-agent debate [37] establishes a symmetric discussion mechanism among agents. ReConcile [38] and Corex [39] treat collaboration as multi-round discussions or deliberations, using consensus mechanisms and role specialization (e.g., proposer, reviewer) to improve answer reliability. Building on these ideas, methods such as CoMM [40] and MALT [41] introduce explicit agent roles and diverse reasoning paths, enabling joint training for complex tasks. MoA [23, 42] further proposes architectural hierarchies and networkbased communication patterns to enhance collective reasoning. The key difference between our approach and methods like MoA lies in the interaction mechanism between reasoning paths. In MoA, each round only accesses the output from the previous round, without reference to earlier context. In contrast, our method enables direct collaboration across multiple reasoning paths, maintaining a finer-grained and more complete history of the reasoning process.

### 7 Conclusion and Future Work

In this work, we identify a phenomenon we term the “Prefix Dominance Trap”, which reveals that the self-correction ability in large reasoning models (LRMs) is still limited. To address this issue, we introduce a novel approach to enhance their performance through Learning from Peers (LeaP). Intuitively, rather than a closed-book exam, this method facilitates note-passing among reasoning path peers, allowing them to borrow insights from one another. We further develop a series of trained models to empower LeaP, referred to as LeaP-T. Our experiments show that LeaP significantly improves models’ ability to recover from initial errors, often outperforming baselines and even larger models in certain cases. Our in-depth analysis further uncovers the mechanisms behind these improvements, offering valuable insights into overcoming the self-correction limitations in LRMs. Overall, our findings highlight the potential of leveraging peer collaboration to improve the reasoning abilities of large models.

We also propose two promising directions for future work:

- • Learning from peers in Reinforcement Learning: Extending peer learning to reinforcement learning (RL) is an exciting direction. By incorporating LeaP into RL, models could learn to collaborate more effectively in solving complex problems, potentially unlocking the full capabilities of base models.
- • Learning from peers with different expertise: Another avenue is to leverage peers with specialized expertise, using different prompts and tools. For example, one group of peers could use web searches while another employs Python for problem-solving. This could enhance reasoning quality, particularly in multi-faceted tasks.

### References

- [1] OpenAI. Learning to reason with llms, 2024. Accessed: 2025-05-03.
- [2] OpenAI. Openai o1 system card, 2024. Accessed: 2025-05-03.
- [3] OpenAI. Openai o3 mini, 2025. Accessed: 2025-05-03.
- [4] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [5] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
- [6] Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Zhihan Guo, Yufei Wang, Irwin King, Xue Liu, and Chen Ma. What, how, where, and how well? a survey on test-time scaling in large language models. arXiv preprint arXiv:2503.24235, 2025.

- [7] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.
- [8] Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Yunhua Zhou, and Xipeng Qiu. Revisiting the test-time scaling of o1-like models: Do they truly possess test-time scaling capabilities? arXiv preprint arXiv:2502.12215, 2025.
- [9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [10] Mauricio J Giuliodori, Heidi L Lujan, and Stephen E DiCarlo. Peer instruction enhanced student performance on qualitative problem-solving questions. Advances in physiology education, 30(4):168–173, 2006.
- [11] Julia J Snyder, B Elijah Carter, and Jason R Wiles. Implementation of the peer-led team-learning instructional model as a stopgap measure improves student achievement for students opting out of laboratory. CBE—Life Sciences Education, 14(1):ar2, 2015.
- [12] Armin Falk and Andrea Ichino. Clean evidence on peer effects. Journal of labor economics, 24(1):39–57, 2006.
- [13] MAA. American Invitational Mathematics Examination - AIME 2024, February 2024. https://maa.org/math-competitions/ american-invitational-mathematics-examination-aime.
- [14] MAA. American Invitational Mathematics Examination - AIME 2025, February 2025. https://maa.org/math-competitions/ american-invitational-mathematics-examination-aime.
- [15] Simon Frieder, Sam Bealing, Arsenii Nikolaiev, Geoff C. Smith, Kevin Buzzard, Timothy Gowers, Peter J. Liu, Po-Shen Loh, Lester Mackey, Leonardo de Moura, Dan Roberts, D. Sculley, Terence Tao, David Balduzzi, Simon Coyle, Alex Gerko, Ryan Holbrook, Addison Howard, and XTX Markets. Ai mathematical olympiad - progress prize 2. https://kaggle.com/ competitions/ai-mathematical-olympiad-progress-prize-2, 2024. Kaggle.
- [16] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.
- [17] Vladimir I Levenshtein et al. Binary codes capable of correcting deletions, insertions, and reversals. In Soviet physics doklady, volume 10, pages 707–710. Soviet Union, 1966.
- [18] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [19] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [20] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.
- [21] Chengpeng Li, Mingfeng Xue, Zhenru Zhang, Jiaxi Yang, Beichen Zhang, Xiang Wang, Bowen Yu, Binyuan Hui, Junyang Lin, and Dayiheng Liu. Start: Self-taught reasoner with tools. arXiv preprint arXiv:2503.04625, 2025.

- [22] Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, Lei Fang, Zhongyuan Wang, and Ji-Rong Wen. An empirical study on eliciting and improving r1-like reasoning models. arXiv preprint arXiv:2503.04548, 2025.
- [23] Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents enhances large language model capabilities. arXiv preprint arXiv:2406.04692, 2024.
- [24] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024. Accessed: 2025-05-03.
- [25] Sara Vera Marjanovi´c, Arkil Patel, Vaibhav Adlakha, Milad Aghajohari, Parishad BehnamGhader, Mehar Bhatia, Aditi Khandelwal, Austin Kraft, Benno Krojer, Xing Han Lù, et al. Deepseek-r1 thoughtology: Let’s< think> about llm reasoning. arXiv preprint arXiv:2504.07128, 2025.
- [26] Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Bo Wang, Shimin Li, Yunhua Zhou, Qipeng Guo, Xuanjing Huang, and Xipeng Qiu. Scaling of search and learning: A roadmap to reproduce o1 from reinforcement learning perspective. arXiv preprint arXiv:2412.14135, 2024.
- [27] Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5315–5333, 2023.
- [28] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.
- [29] Yifan Song, Guoyin Wang, Sujian Li, and Bill Yuchen Lin. The good, the bad, and the greedy: Evaluation of llms should not ignore non-determinism. arXiv preprint arXiv:2407.10457, 2024.
- [30] Alex Nguyen, Dheeraj Mekala, Chengyu Dong, and Jingbo Shang. When is the consistent prediction likely to be a correct prediction? arXiv preprint arXiv:2407.05778, 2024.
- [31] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.
- [32] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [33] Hanshi Sun, Momin Haider, Ruiqi Zhang, Huitao Yang, Jiahao Qiu, Ming Yin, Mengdi Wang, Peter Bartlett, and Andrea Zanette. Fast best-of-n decoding via speculative rejection. arXiv preprint arXiv:2410.20290, 2024.
- [34] Lin Gui, Cristina Gârbacea, and Victor Veitch. Bonbon alignment for large language models and the sweetness of best-of-n sampling. arXiv preprint arXiv:2406.00832, 2024.
- [35] Afra Amini, Tim Vieira, Elliott Ash, and Ryan Cotterell. Variational best-of-n alignment. arXiv preprint arXiv:2407.06057, 2024.
- [36] Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Ramé, Bobak Shariari, Sarah Perrin, Abe Friesen, Geoffrey Cideron, et al. Bond: Aligning llms with best-of-n distillation. arXiv preprint arXiv:2407.14622, 2024.
- [37] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, 2023.
- [38] Justin Chih-Yao Chen, Swarnadeep Saha, and Mohit Bansal. Reconcile: Round-table conference improves reasoning via consensus among diverse llms. arXiv preprint arXiv:2309.13007, 2023.

- [39] Qiushi Sun, Zhangyue Yin, Xiang Li, Zhiyong Wu, Xipeng Qiu, and Lingpeng Kong. Corex: Pushing the boundaries of complex reasoning through multi-model collaboration. arXiv preprint arXiv:2310.00280, 2023.
- [40] Pei Chen, Boran Han, and Shuai Zhang. Comm: Collaborative multi-agent, multi-reasoning-path prompting for complex problem solving. arXiv preprint arXiv:2404.17729, 2024.
- [41] Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, Rafael Rafailov, Ivan Laptev, Philip HS Torr, Fabio Pizzati, Ronald Clark, and Christian Schroeder de Witt. Malt: Improving reasoning with multi-agent llm training. arXiv preprint arXiv:2412.01928, 2024.
- [42] Dawei Li, Zhen Tan, Peijia Qian, Yifan Li, Kumar Satvik Chaudhary, Lijie Hu, and Jiayi Shen. Smoa: Improving multi-agent large language models with sparse mixture-of-agents. arXiv preprint arXiv:2411.03284, 2024.
- [43] Ke Ji, Jiahao Xu, Tian Liang, Qiuzhi Liu, Zhiwei He, Xingyu Chen, Xiaoyuan Liu, Zhijie Wang, Junying Chen, Benyou Wang, et al. The first few tokens are all you need: An efficient and effective unsupervised prefix fine-tuning method for reasoning models. arXiv preprint arXiv:2503.02875, 2025.

### A Prompts

Summarization Prompt Structure

Summary Trigger: (randomly select one)

- • Alright, let’s take a step back and summarize what we’ve figured out so far.
- • Wait, let me quickly recap what I’ve concluded so far.
- • Alright, let me shortly review the conclusions I’ve drawn so I can move forward more efficiently.
- • Hmm, a quick summary of what I’ve figured out might help streamline the next part of my reasoning.
- • Hold on, I should summarize the key points briefly to ensure I’m on the right track.
- • Okay, before continuing, let me put together a brief summary of the insights I’ve gathered so far.
- • Okay, time to consolidate everything I’ve found into a concise summary.

Summary Template: (randomly select one)

- • In short, my current conclusions are that ...
- • To summarize, based on my previous reasoning, I have currently found that ...
- • In conclusion, the current key takeaways and results are ...
- • In short, I’ve currently concluded that ...
- • To summarize, my recent findings are ...
- • In conclusion, the current insights and results I’ve gathered are ...

Figure 14: The structure of the summarization prompt used during LeaP.

Prompt for Different Tasks

GPQA: Please show your choice in the answer field with only the choice letter, e.g.,“ANSWER”: “C”. Math Tasks: Please reason step by step, and put your final answer within \boxed.

Figure 15: Prompts for different tasks.

Prompt for Mixture-of-Agents

Problem: {problem} You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Responses from models:

Figure 16: Prompts for mixture of agents.

### B Prefix Dominance Trap for Starting with Good Beginnings

While the “Prefix Dominance Trap” highlights how poor beginnings can severely constrain reasoning, a natural follow-up question is whether good beginnings are sufficient to ensure correct final answers. To this end, we conduct a follow-up experiment similar to those in Section 2.1 under the good

beginnings setting, where the model starts reasoning with a beginning of 15% average response length from correct reasoning paths.

As shown in Figure 17a, models initialized with good beginnings indeed perform better than random or bad initializations. This suggests that early correct cues provide useful guidance. However, these beginnings do not fully eliminate reasoning errors—many responses still deviate in later steps and arrive at incorrect conclusions. This observation reinforces that good beginnings can reduce—but not entirely prevent—reasoning failures.

We then examine how LeaP performs in the same setting. As illustrated in Figure 17b, our method consistently surpasses independent reasoning, even when starting from already high-quality beginnings. For instance, DeepSeek-Distill-Qwen-14B improves from 74.38 to 87.50 in Pass@1. This significant gain indicates that LeaP not only repairs faulty reasoning from poor prefixes but also mitigates subtle errors that emerge even when the reasoning starts correctly.

With beginnings No beginnings

With beginnings No beginnings

90

90

80

80

23.44

70

70

Pass@1(%)

Pass@1(%)

11.25

10.32

8.75

60

60

17.58

14.84

50

50

40

40

30

30

Distill-7B Distill-14B QwQ-32B

Distill-7B Distill-14B QwQ-32B

(a) Independent Reasoning

(b) LeaP

Figure 17: The results of starting with a good beginning of 15% average response length.

### C Additional Evaluation Results

#### C.1 Random Routing Results

Table 5: The results of Pass@1 for four routing mechanisms on four benchmarks.

Random Clustered Hybrid Dispersed Top-2 Top-4 Top-2 Top-4 Top-2 Top-4 Top-2 Top-4

Benchmarks Models

- AIME 2024

R1-7B 58.65 60.83 56.15 59.27 59.17 61.67 60.31 60.52 R1-14B 72.60 75.83 72.08 71.04 71.77 74.48 71.15 77.29

QwQ-32B 80.21 81.67 78.96 81.56 81.88 81.67 81.56 85.83

AIMO 2025

R1-7B 40.62 42.81 37.81 39.06 41.88 40.31 39.06 45.00 R1-14B 52.50 53.13 51.50 55.31 51.88 52.19 49.38 51.25

QwQ-32B 63.75 68.44 63.75 65.63 64.06 64.69 67.19 67.19

- AIME 2025

R1-7B 38.02 39.17 36.98 37.50 37.19 39.27 40.93 38.44 R1-14B 48.96 52.29 50.42 51.88 54.38 54.38 50.31 54.17

QwQ-32B 71.35 71.98 68.85 71.35 71.04 72.50 70.83 71.67 GPQA

R1-7B 52.96 52.65 52.97 52.65 51.83 53.47 53.28 55.56 Diamond

R1-14B 54.99 57.26 54.80 58.33 54.42 57.89 55.68 55.05 QwQ-32B 64.65 64.89 65.03 65.28 61.87 65.21 66.16 63.32

R1-7B 47.56 48.87 45.98 47.12 47.52 48.68 48.40 49.88 R1-14B 57.26 59.63 57.20 59.14 58.11 59.74 56.63 59.44

Avg.

QwQ-32B 69.99 71.75 69.15 70.96 69.71 71.02 71.44 72.00

To better evaluate the effectiveness of the routing mechanisms introduced in Section 2.2, we use Random routing as an extra baseline. In Table 5, we present the Pass@1 results across four routers on four benchmarks. It is evident that Top-4 outperforms Top-2 under the Random setting, as discussed in Section 3. To our surprise, when comparing the results across different Routers, we find that Random even outperforms Hybrid. For example, the average Pass@1 of QwQ-32B reaches 71.75 under the Top-4 setting, which surpasses Hybrid by 0.73. This is because in the Random, the proportion of dissimilar summaries coming from different peers in the model’s sampling paths is higher than that of similar summaries. In contrast, similar and dissimilar summaries each account for half in Hybrid. As a result,the Random can introduce more diverse or complementary perspectives, providing the model with more useful information. Similarly, under the same setting of Top-4, the average Pass@1 of QwQ-32B for Dispersed reaches 72, which surpasses the Random by 0.25, as the Dispersed can introduce more diverse or complementary perspectives than Random.

#### C.2 Cons@N Results

To provide a comprehensive evaluation, we also report the Cons@N results in Table 6. We combine the AIME 2024 [13], AIME 2025 [14], and AIMO 2025 [15] benchmarks into a single category, denoted as Math Comb., due to the small size of the individual math benchmarks. In general, the results under the selected two routing strategies consistently surpass the two baselines by a large margin. Specifically, on GPQA [16], the Top-4 Hybrid setting achieves a 10.10 improvement over the Majority Voting [19] baseline using DeepSeek-R1-Distill-Qwen-7B [4]. On the average of the three math benchmarks, the Top-2 Hybrid setting outperforms the baseline by 4.29 on QwQ-32B [5]. Moreover, we observe that the improvement on GPQA [16] is more pronounced than on math tasks. Since GPQA [16] requires a combination of factual knowledge and reasoning, the cues learned from peer reasoning paths lead to even greater performance gains.

Table 6: The Cons@N results of Math benchamrks and GPQA Diamond. Benchmarks Models

Majority Shortest Dispersed Hybrid Vote Vote Top-2 Top-4 Top-2 Top-4

R1-7B 60.00 60.00 62.86 61.43 61.43 62.86 R1-14B 68.57 70.00 68.57 70.00 70.00 71.43

Math Comb.

QwQ-32B 78.57 80.00 80.00 80.00 82.86 81.43

R1-7B 50.00 51.52 59.60 58.59 56.57 60.10 R1-14B 55.05 55.05 59.60 59.60 60.61 61.11

GPQA Diamond

QwQ-32B 60.61 60.10 67.68 63.64 63.13 68.18

In order to provide a more detailed analysis of the math tasks, we analyze the AIME 2024 [13], AIME 2025 [14], and AIMO 2025 [15] benchmarks individually. In Table 7, we present the Cons@N results across three math benchmarks. We observe a significant improvement in both AIME 2024 and AIME 2025 with the two selected routing strategies. For example, QwQ-32B reaches 86.67 under the Top-4 setting using either the Dispersed or Hybrid routing strategy on AIME 2024, surpassing the baseline by 3.34. However, these two routing strategies result in almost no improvement on AIMO 2025. This is because AIMO has relatively few problems, and the difficulty gap between them is large, making significant improvements hard to achieve.

Table 7: The Cons@32 results on AIME 2024, AIME 2025 and AIMO 2025. Benchmarks Models

Majority Shortest Dispersed Hybrid Vote Vote Top-2 Top-4 Top-2 Top-4

- AIME 2024

R1-7B 73.33 73.33 76.67 73.33 80.00 80.00 R1-14B 80.00 80.00 80.00 83.33 80.00 80.00 QwQ-32B 83.33 83.33 86.67 86.67 90.00 86.67

AIMO 2025

R1-7B 50.00 50.00 50.00 50.00 50.00 50.00 R1-14B 60.00 60.00 60.00 60.00 60.00 60.00 QwQ-32B 70.00 80.00 70.00 70.00 70.00 70.00

- AIME 2025

R1-7B 50.00 50.00 53.33 53.33 46.67 50.00 R1-14B 60.00 70.00 60.00 60.00 63.33 66.67 QwQ-32B 76.67 70.00 76.67 76.67 80.00 80.00

#### C.3 Results of Number of Tokens and “Aha” moments

We provide more details for the results of LeaP on the DeepSeek-R1-Distill-Qwen-7B and DeepSeekR1-Distill-Qwen-14B models compared to the baseline in Figures 18 and 19. We observe that LeaP does not generate significantly more tokens than the baseline for math-related problems. On the contrary, in the GPQA Diamond, the average length of generated responses is shorter than that of the baseline. Furthermore, the number of “aha” moments for LeaP is generally fewer than that of the baseline in math-related problems and in the GPQA diamond. This trend is consistent with our analysis on QwQ-32B in Section 3.2.

|0<br><br>2000<br><br>4000<br><br>6000<br><br>8000<br><br>10000<br><br>12000<br><br>Math Tasks GPQA Diamond<br><br>Tokens<br><br>Baseline Dispersed-Top2 Dispersed-Top4<br><br>Hybrid-Top2 Hybrid-Top4<br><br>|
|---|

|0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>Math Tasks GPQA Diamond<br><br>”Aha”moments<br><br>Baseline Dispersed-Top2 Dispersed-Top4<br><br>Hybrid-Top2 Hybrid-Top4<br><br>|
|---|

###### Tokens

(a) The number of tokens

(b) The number of “Aha” moments

- Figure 18: We illustrate the average number of tokens and “Aha” moments on DeepSeek-R1-DistillQwen-7B. Our method produces a comparable number of tokens to the baseline, while yielding fewer “Aha” moments.

|0<br><br>3000<br><br>6000<br><br>9000<br><br>12000<br><br>15000<br><br>MATH Tasks GPQA Diamond<br><br>Tokens<br><br>Baseline Dispersed-Top2 Dispersed-Top4<br><br>Hybrid-Top2 Hybrid-Top4<br><br>|
|---|

(a) The number of tokens

|0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>MATH Tasks GPQA Diamond<br><br>”Aha”moments<br><br>Baseline Dispersed-Top2 Dispersed-Top4<br><br>Hybrid-Top2 Hybrid-Top4<br><br>|
|---|

”Aha”moments

(b) The number of “Aha” moments

- Figure 19: We illustrate the average number of tokens and “Aha” moments on DeepSeek-R1-DistillQwen-14B. Our method produces a comparable number of tokens to the baseline, while yielding fewer “Aha” moments.

### D Case Study

#### D.1 Case Study on Comparing LeaP and Baseline with Bad Beginning

Case Study on LeaP with Bad Beginning

Problem:

Let A, B, C, and D be point on the hyperbola x202 − y242 = 1 such that ABCD is a rhombus whose diagonals intersect at the origin. Find the greatest real number that is less than BD2 for all such rhombi.

Model Response: Okay, so I have this problem where there are four points A, B, C,

- D on the hyperbola given by the equation x202 − y242 = 1. ... So, we need to find the supremum (the least upper bound) of BD2 = 4(r2 + s2), such that the points satisfy the hyperbola

equation and the condition pr + qs = 0. So, let me formalize this. Given: 1. p202 − q242 = 1. 2. r2 20 − s242 = 1. 3. pr + qs = 0. We need to find the supremum of 4(r2 + s2). (Bad Beginning)

...

<summarize> In conclusion, the current key takeaways and results are: ... Using hyperbola parametrization and vector properties, I derived an expression for BD2 in terms of a parameter

b, which is tanh u. The expression for BD2 is 80+961−b2b2 with the constraint |b| ∈ (65, 1). By analyzing the derivative of this function, I found that it’s increasing in the interval, implying

the minimum BD2 occurs as b approaches 65. Therefore, the minimal BD2 is achieved in the limit as b approaches 56. </summarize> (Incorrect Summary) <peer_summaries>

- Peer 1: "To summarize, based on my previous reasoning, I have currently found that BD2 can be expressed as 80+(22v)/3, where v = s2. The variable v is constrained by the equation

u = [600(v − 24)]/(11v − 600), where u = q2. The variable v must be either less than 24 or greater than approximately 54.545...

- Peer 2: "In short, I’ve currently concluded that the expression for BD2 in terms of u = p2 is 480 × 1111uu−−120720. I also noticed that as u approaches 72011 from above, BD2 tends to infinity, which seems contradictory because points on a hyperbola shouldn’t allow for infinitely long diagonals...

- Peer 3: "To summarize, my recent findings are: I attempted to express p and s in terms of q and r using the condition pr + qs = 0, leading to p = kq and s = −kr...
- Peer 4: "In conclusion, the current insights and results I’ve gathered are: Expressed BD2 in terms of t (where t = q2), transformed the equation into a function f(t) = 80 + [4400(t + 24)] / (11t - 600), and simplified it to f(t) = 480 + (345600/11)/(t - 600/11). The domain of t is t > 600/11. So, as t approaches infinity, f(t) approaches 480, and as t approaches 600/11 from the right, f(t) approaches infinity. Therefore, BD2 can be made arbitrarily large, but the problem asks for the greatest real number less than BD2 for all such rhombi. So, that number is 480." (Correct Summary from Peers) </peer_summaries>

... From the comments, it appears that some peers have approached the problem differently and arrived at different expressions for BD2.Notably, Peer 4 suggests that BD2 approaches 480 as t approaches infinity, and that BD2 can be made arbitrarily large, but the greatest number less than BD2 for all rhombi is 480. (Self-Verification) Wait, that seems conflicting with my earlier conclusion where BD2 approaches infinity as b approaches 1. So, perhaps I made a mistake in my parametrization. Wait, let me check Peer 4’s approach. Peer 4 says: Expressed BD2 in terms of t (where t = q2)

... Therefore, the infimum of BD2 is 480, so the greatest real number less than BD2 for all rhombi is 480. But in my own reasoning, I thought BD2 can be made as large as desired, which would suggest that the infimum is 480, and the supremum is infinity. So, the greatest real number less than BD2 for all rhombi is 480... Therefore, the answer is 480. But wait, in my earlier parametrization, I had BD2 tending to infinity as b approaches 1, but here, it’s showing that BD2 can be made as small as 480 and as large as infinity. Therefore, the greatest real number less than BD2 for all rhombi is 480.

So, the answer is 480. **Final Answer**

|480|
|---|

Figure 20: Case Study on LeaP with bad beginning.

Case Study on Baseline with Bad Beginning

Problem:

Let A, B, C, and D be point on the hyperbola x202 − y242 = 1 such that ABCD is a rhombus whose diagonals intersect at the origin. Find the greatest real number that is less than BD2 for all such rhombi.

Model Response:

Okay, so I have this problem where there are four points A, B, C, D on the hyperbola given by the equation x202 − y242 = 1. ... So, we need to find the supremum (the least upper bound) of BD2 = 4(r2 + s2), such that the points satisfy the hyperbola equation and the condition pr +qs = 0. So, let me formalize this. Given: 1. p202 − q242 = 1. 2. r202 − s242 = 1. 3. pr +qs = 0. We need to find the supremum of 4(r2 + s2). (Bad Beginning)

...

Therefore, f(t) decreases from +infinity to 80 as t increases from 5/6 to infinity. Therefore, the infimum of BD2 is 80, but it’s never actually reached; it’s a limit as t approaches infinity. Therefore, BD2 can be made arbitrarily close to 80, but never less than 80. Therefore, the

greatest real number less than BD2 for all such rhombi is 80. Hence, the answer is

.

|80|
|---|

**Final Answer**

|80|
|---|

Figure 21: Case Study on baseline with bad beginning.

#### D.2 Case Study on Failure Examples

We provide some case studies to better illustrate our approach. In Figure 22, we present a failure case when applying LeaP to DeepSeek-R1-Distill-Qwen-7B. In this example, the model fails to clearly summarize its own reasoning, which results in the loss of partial reasoning information during summarization. This issue primarily relates to model size. Specifically, when the model is relatively small, incomplete summarization tends to occur more frequently.

In the second case, we demonstrate a situation where the model refuses to engage in communication. As shown in Figure 23, the model assumes it is alone and therefore ignores the perspectives of other peers, continuing its own reasoning trajectory. This case is observed in QwQ-32B but not in the distilled model from R1. We hypothesize that this difference may be related to the training method: QwQ-32B is trained via reinforcement learning (RL), which may lead the model to exhibit higher confidence.

Case Study on Summarization Failure

Problem:

Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s + 2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop.

Suppose Aya walks at s+ 21 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop.

Model Response:

... Wait, wait, from x253 − 34x + 33 = 0, x = k8, so x = 1 is a root, so we can factor it as (x − 1)(something) = 0. But since x = 1 is a root, and we’re interested in the other root, perhaps it’s a very small x, but I’m not sure. Wait, perhaps I can write the equation as x253 = 34x − 33. Hold on, I should summarize the key points briefly to ensure I’m on the right track. <summarize> To summarize, my recent findings are: The problem involves two circles of radius 34 and 2024 arranged in a triangle ABC, with each circle tangent to two sides

and the previous circle. The inradius R of ABC can be expressed as R = 34· 11−−kk8 for the 34radius circles, and R = 1−1k−2024k for the 1-radius circles. Setting these equal gives the equation 34·(1−k8) = 1−k2024, which simplifies to k2024−34k8+33 = 0. Letting x = k8, this becomes x253 − 34x + 33 = 0, which factors as (x − 1)(x252 + x251 + · · · + x + 1) − 34x + 33 = 0. Since x = 1 is a root, but we need another real root where x < 1. Approximating, x ≈ 0.970588, leading to k ≈ 0.9963. Then, R ≈ ... (truncated at 256 tokens) </summarize>

...

Figure 22: The case study about the reasoning model fail to summarize concisely.

Case Study on Reflection Failure

Problem:

Torus T is the surface produced by revolving a circle with radius 3 around an axis in the plane of the circle that is a distance 6 from the center of the circle (so like a donut). Let S be a sphere with a radius 11. When T rests on the inside of S, it is internally tangent to S along a circle with radius ri, and when T rests on the outside of S, it is externally tangent to S along a circle with radius ro. The difference ri − ro can be written as mn , where m and n are relatively prime positive integers. Find m + n. [asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); draw((-1.2,0.1)..(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11)..(0.8,-0.03)..(1.2,0.1)); draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12)..(1,0.04)); draw((0,2.4)–(0,-0.15)); draw((0,0.15)–(0,-1.75), dashed); draw((0,-1.75)–(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); draw((2.03,0.02)–(2.9,-0.4)); [/asy]

Model Response:

... <peer_summaries>

... Peer 4: "In conclusion, the current insights and results I’ve gathered are: By setting up the equations for the sphere and the torus, and analyzing the gradients for tangency, I derived that at points of tangency, the z-component gives k = 2 (assuming z ̸= 0). This led to an equation relating the radial distance ρ = sqrt(x2 + y2) to the major radius R = 6. Solving this gives a specific ρ value, which when substituted back into the equations, yields a valid solution for z. This allows computing the radii ri and ro for internal and external tangency, leading to the desired difference. However, I must ensure that the configuration (internal vs. external) corresponds correctly to the direction of the normals." </peer_summaries> Hmm, it seems that my peers have given me some comments, so let me check if anyone’s conclusions are different from mine before I continue my own reasoning. But since I’m alone, I’ll proceed. Let me try to proceed with the standard equation of the torus. Let me define the torus such that its axis is the x − axis. The generating circle has a major radius R = 6, so its center is at (6, 0, 0), and minor radius r = 3. The equation of the torus is: (sqrt(y2 +z2))2 +(x−6)2 = 32 ? Wait, no, perhaps I need to think differently. Let me recall that the standard equation for a torus rotated around the x − axis with major radius R and minor radius r is: (sqrt((sqrt(y2 + z2) − R)2 + x2))? Wait, perhaps I should use a different approach.

...

Figure 23: The case study that the QwQ-32B reject to understand the summaries from peers.

#### D.3 Case Study on Success Examples

In the third case, we present a scenario where the model ultimately reaches the correct answer through interaction with its peers. As shown in Figure 24, the model initially fails to fully summarize its own reasoning process, leading to the loss of partial reasoning information during summarization. However, during the subsequent interaction with peers, the model receives a clearer and more explicit reasoning path provided by them. By verifying the summaries generated by other peers, the model gradually identifies the correct solution to the problem.

In the fourth case, we present a scenario where the model initially generates an incorrect answer during the reasoning process. Meanwhile, other peers also produce flawed summaries, though some provide correct ones. Through interaction and mutual verification, the model eventually identifies the correct reasoning path. As shown in Figure 25, the model first summarizes its own reasoning and arrives at the incorrect answer of 389. Peer 1 and Peer 2 provide incorrect summaries, while only Peer 3 offers a reasonable one. By verifying the summaries from all peers, the model determines that the summaries from Peer 1 and Peer 2 are inconsistent. In contrast, the summary from Peer 3 is considered reliable. Based on this verification process, the model gradually uncovers the correct solution to the problem.

<peer_summaries>

... <summarize> In conclusion, the current insights and results I’ve gathered are as follows: **Good Subsets**: Subsets with 0 to 3 elements are all good, totaling 1 + 8 + 28 + 56 = 93 subsets. Subsets of size 4 may include some bad subsets, while subsets of size 5–8 are mostly bad, though not all. - **Bad Subsets of Size 4**: Examples like 0,1,3,4 cannot be rotated to be disjoint, indicating that some size-4 subsets are bad. Calculating their exact count requires identifying patterns or symmetries that make these subsets resistant to rotation. - **Symmetry and Complementarity**: The problem’s symmetry suggests that the number of bad subsets of size k equals those of size 8 − k. However, since subsets of size 3 (complement of 5) are all good, subsets of size 5 must also be good, contradicting earlier assumptions.This implies a need to reevaluate. - **Reevaluation**: Perhaps subsets of size 5 are actually good, and the real bad subsets are only those of size 7, 8, and some specific size 4 subsets. This would mean the total bad subsets are 1 (... </summarize> (truncated at 256 tokens) <peer_summaries> Peer 1: "..."Peer 2:"..."Peer 3: "..."Peer 4:"To summarize, my recent findings are as follows: - **Subsets of Size 4**: - **Rotation by 1 or 7**: Only two valid subsets (checkerboard patterns 0,2,4,6 and 1,3,5,7). - **Rotation by 2 or 6**: Requires selecting opposite pairs in each of the two 4-cycles formed by the rotation. This yields 2×2 = 4 subsets. - **Rotation by 4**: Selecting one vertex from each of the four diametrical pairs, resulting in 24 = 16 subsets. - **Overlap Considerations**: No overlaps between these categories, as the subsets from different rotations are structurally distinct. **Total Valid Size 4 Subsets**: 2 +4 +16 =22. - **Total Favorable Subsets**: - Size 0:1 Size 1:8 - Size 2:28 - Size 3:56 - Size 4:22 - **Total**:1+8+28+56+22 =115 - **Probability Calculation**: - Total subsets:..."</peer_summaries> Hmm, it seems that my peers have given me some comments, so let me check if anyone’s conclusions are different from mine before I continue my own reasoning. Peer 1 is talking about independent sets on a cycle with distance constraints. Peer 2 suggests that size4 subsets can be counted by analyzing different rotation types, leading to 22 valid subsets. Peer 3 mentions inclusion-exclusion over rotation classes. Peer 4 calculates total favorable subsets as 115, leading to probability 115/256. Wait, Peer4’s calculation: Size0:1 Size1:8 Size2:28 Size3:56 Size4:22 (only those that are good) Total: 1+8+28+56+22 =115 Thus probability 115/256, which simplifies to 115 and 256 are coprime? 256 is 28. 115 is 5 × 23, so yes. Thus m+n=115+256=371. But I need to verify if this is correct. According to Peer4’s reasoning: For size4 subsets, the valid (good) subsets are 22. The bad subsets of size4 are 70-22=48. Therefore, total bad subsets are: Size5:56 Size6:28 Size7:8 Size8:1 Size4 bad:48 Total bad:56+28+8+1+48=141 Thus total good subsets:256-141=115 Therefore, probability 115/256 Thus the answer is 115+256=371. But I need to verify if Peer4’s count of size4 subsets is correct. Peer4 says

...

Yes. Thus, these subsets are good. The checkerboard patterns are also good. Therefore, the count of22 valid subsets of size4 is correct. Thus, the total good subsets are115, leading to

the probability of115/256. Thus, the final answer is115 +256= 371.( **Final Answer** )

|371|
|---|

- Figure 24: The case study that the model learn from peers successfully, which lead to a correct final answer.

<summarize> In conclusion, the current insights and results I’ve gathered are as follows: The problem involves two configurations of circles arranged in triangle ABC: eight circles of radius 34 and 2024 circles of radius 1,

...

- After extensive exploration and considering peer insights, the most consistent approach involves setting up equations based on the path lengths required by both configurations and

solving for the inradius using proportional reasoning, leading to an inradius of (3845 ),( resulting in (m + n = 384 + 5 = 389... )</summarize> <peer_summaries>Peer 1: "To summarize, based on my previous reasoning, I have currently found

...

Thus, GCD is 1, making 682007,800 already in simplest terms. Adding numerator and denominator gives 68, 800 + 2007 = 70, 807. However, this contrad..." Peer 2: "In short, my current conclusions are that the problem involves arranging circles in geometric progression along an angle bisector of triangle ABC’s vertex B.

...

The relationship 2024 = 8 × 253 hints that k2016 = 34 simplifies to (k8)252 = 34, implying k8 = 252√34. This still leaves the inradius..." Peer 3: "In short, I’ve currently concluded that by modeling the problem using a geometric progression of radii along the angle bisector at vertex B, and setting up equations for both configurations (8 circles of radius 34 and 2024 circles of radius 1), I derived that the sine of half the angle θ/2 is 119011 . Substituting back into the inradius formula yields R = 1925 , which simplifies to 1925 . Since 192 and 5 are coprime, the final answer is m + n = 197. This result aligns with the calculations when assuming the incenter is positioned such that the distance from vertex B to the incenter along the angle bisector equals the distance to the last circle plus its radius." Peer 4: "In conclusion, the current key takeaways and results are as follows: By modeling the problem using geometric progressions for the radii of the circles arranged along the angle at vertex B,

... the expression for R simplifies to R = 234×134/2016253/+1252 . Recognizing that 34253/252 = 341+1/252, this becomes R =..." </peer_summaries>

...

Peer 3 suggests an inradius of 192/5, so 192+5=197. Peer 1 and 2 are getting large fractions. Peer 4’s answer is unclear. Given that in the Asymptote example, the inradius was about 0.45 for small circles, and the problem has 2024 circles of radius 1,

...

which is not a fraction. Alternatively, the answer is 34/253 +1/2024? No. Alternatively, the problem is a trick question where the inradius is simply the sum of the radii multiplied by the number of circles divided by some number. Alternatively, the inradius is the solution to the equation: 8*(2*34) =2024*(2*1) Which is 544 =4048 → impossible. Thus, the only plausible answer is Peer3’s 192/5, which is 38.4.

...

Given that 384/5 is 76.8, and the path length for first case is 544, which is 7*68 + 68=544, and 544/7=77.714. Close to 76.8. Perhaps Peer1’s calculation is correct. Alternatively, the inradius is 544/7=77.714= 544/7=77.714285..., which is 544 divided by7 is 77.714. 544 divided by7:7×77=539, so 544/7=77+5/7=77.714285... Thus, as a fraction, 544/7, which reduces to 544/7 (since 544 and7 are coprime). Thus, m+n=544+7=551.

. . .

then r=34 +476 sinθ To also satisfy the second configuration: r=1 +4046 sinθ Thus, 34+476 sinθ=1+4046 sinθ 33= 3570 sinθ sinθ=33/3570=11/1190=0.0092 Thus, sinθ=0.0092 Then, r=1 +4046*(11/1190)=1 + (4046×11)/1190 Calculate numerator:4046×11=44506 Divide by1190: 44506/1190 =37.4 Thus, r =1+37.4=38.4=384/10=192/5=38.4 Ah, so this gives r=192/5=38.4 Thus, m=192, n=5, so m+n=197 This matches Peer3’s answer. Therefore, the inradius is 192/5. The steps: Assuming the progression ratio k=1 (even though impossible), then the inradius is derived as above, leading to sinθ=11/1190, and substituting back gives r=192/5. Thus, the answer is 192+5=197. **Final Answer**

) ...

|197|
|---|

- Figure 25: The case study that the model learn from peers successfully, which lead to a correct final answer.

### E Training Hyperparameters of LeaP-T

Table 8: Hyperparameters for LeaP-T training. Hyperparameter 1.5B Model 7B Model 14B Model

Batch size 16 16 16 Learning rate 1 × 10−5 1 × 10−5 1 × 10−5 Learning rate scheduler Cosine decay Cosine decay Cosine decay Warmup ratio 0.05 0.05 0.05 Optimizer AdamW AdamW AdamW Weight decay 1 × 10−4 1 × 10−4 1 × 10−4 Max sequence length 16K 16K 16K Training epochs 8.0 8.0 5.0 Precision bfloat16 bfloat16 bfloat16

### F Comparison with MoA

N=6

R1-7B

| |
|---|

N=5

72.5

LeaP-T-7B

| |
|---|

R1-7B + MoA

N=4

N=7

70.0

| |
|---|

N=6

N=3

N=8

67.5

Accuracy(%)

65.0

N=5

N=2

62.5

N=4

| |
|---|

N=3

60.0

57.5

N=2

55.0

52.5

16k 24k 32K 40K 48K 56k 64k 72K 80K

Avg Total Compute (Thousands of Tokens)

Figure 26: Accuracy vs. Total tokens on AIME 2024.

We also test MoA [23] as a baseline under the setting of 4 Layers and 3 Agents, with the prompts available in Appendix A. As shown in Figure 26, MoA costs approximately 80K tokens per problem, but the results show that it does not transfer well to LRMs. The reason is that LRMs cannot follow the user’s instructions effectively. We find that even when the correct answer is achieved in the intermediate layers, the model still reaches incorrect conclusions in the final layer.

### G Other In-depth Analysis

- G.1 Sensitivity Analysis

- G.1.1 On Temperature

| |LeaP with different temperatures| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| |77.3%| | | | | | | | | | |
| | | | | | | | | | | | |
| |71.9%<br><br>72.9%<br><br>73.2%| | | | | | | | | | |
| |70.4%| | | | | | | | | | |
| | | | | | | | | | | | |

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Temperature

70%

72%

74%

76%

78%

Pass@1

Figure 27: Pass@1 results of various temperatures on AIME 2024.

We conduct an in-depth analysis of the temperature parameter τ by varying it from 0.1 to 1.0, while keeping all other settings fixed. We use Top-4 Dispersed routing and set the communication interval to T = 4k tokens. As shown in Figure 27, when the temperature is low (τ ≤ 0.3), the model achieves an average accuracy of around 71, similar to the Clustered router baseline. This is expected, as a low temperature reduces output diversity, thereby limiting the benefits of peer communication. On the other hand, when the temperature is too high (e.g., τ ≥ 0.9), performance drops noticeably. This is likely due to excessive randomness, which may lead the model away from coherent reasoning or instruction following. Overall, a moderate temperature appears to strike a good balance between diversity and stability.

- G.1.2 On Communication Granularity T for LeaP-T

To further study the effect of communication granularity, we analyze LeaP-T under different token intervals T. As shown in Figure 28a, for the 7B version, the Pass@1 score on AIME 2024 decreases from 66.35 to 61.25 as T increases, while the number of generated tokens drops accordingly. This trend is consistent with the analysis in Section 5.1.1: more frequent communication (i.e., smaller T) improves performance slightly, but increases token consumption due to more frequent summarization and message exchange. We observe a similar trend for the 14B version in Figure 28b.

|Tokens baseline Tokens Pass@1 baseline Pass@1| |Tokens baseline Tokens Pass@1 baseline Pass@1|
|---|---|---|
|14000 70% 66.35%<br><br>12000 65.52% 64.38%<br><br>10845 65% 10000 61.25%<br><br>8000 60%<br><br>Pass@1<br><br>Pass@1<br><br>Tokens<br><br>Tokens<br><br>6000 55% 4000 51.56%<br><br>50%<br><br>2000| |14000 80% 12000<br><br>|76.46%| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | |75.31%<br><br>74.38%<br><br>73.23%| | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
<br><br>75%<br><br>10000<br><br>9629 8000<br><br>70% 6000<br><br>65.83% 4000<br><br>65%<br><br>2000|
|0 45%<br><br>2048 3072 4096 6144| |0 60%<br><br>2048 3072 4096 6144|
|Interval Tokens| |Interval Tokens|

(a) 7B (b) 14B

- Figure 28: Pass@1 and total tokens on AIME 2024 for LeaP-T-7B and LeaP-T-14B models, evaluated across interval tokens of LeaP from 2048 to 6144.

#### G.1.3 On Communication Position for LeaP-T

Furthermore, Figure 29 analyzes when communication is most effective. The results indicate that performance peaks when the LeaP block is inserted at 4K tokens. Specifically, Pass@1 increases from 68.85 at 2K to 71.77, then declines to 69.69 at later positions. Although a slight recovery to 71.25 is observed, the overall pattern suggests that earlier communication tends to yield better results. These results are consistent with our analysis in Section 5.1.4, and highlight the importance of timely information exchange in improving performance.

|| |
|---|
| |
| |
|71.77%|
|70.52%<br><br>70.83%<br><br>69.69%<br><br>71.25%|
|68.85%|
| |
| |
| |
<br><br>76.46%<br><br>64.48%<br><br>64%<br><br>66%<br><br>68%<br><br>70%<br><br>72%<br><br>74%<br><br>76%<br><br>2K 4K 6K 8K 10K 12K<br><br>Pass@1<br><br>Position<br><br>Partial Full No<br><br>|
|---|

###### Pass@1

- Figure 29: Performance of partial LeaP on the 14B model with LeaP-T, evaluated across the positions of LeaP block ranging from 2K to 12K tokens

#### G.2 Robustness at Various Difficulty Levels for LeaP-T

We also report the performance of LeaP-T across the five difficulty levels of AIME 2024. These difficulty levels follow the same categorization introduced in Section 5.2.2, based on model ( without peer reasoning) accuracy. To exclude the impact of distilling with a 32B model, we conduct distillation using SFT without LeaP, which serves as our baseline. This setup allows for a direct comparison with our LeaP-T approach. Across all five difficulty levels, LeaP-T generally outperforms the baseline. For instance, as shown in Figure 32, LeaP-T achieves higher accuracy while consuming fewer tokens. These findings are consistent with the analysis presented in Section 5.2.2. This is because LeaP facilitates earlier consensus during the reasoning process, thereby reducing unnecessary computational overhead caused by overthinking. Notably, the improvement of LeaP-T over the baseline suggests that its performance gain is not solely attributed to knowledge transfer through distillation. Instead, it underscores the effectiveness of a training paradigm centered on learning from peers.

18000

| |Baseline Tokens<br><br>Ours Tokens<br><br>Baseline Accuracy<br><br>Ours Accuracy| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

100

16000

14000

80

12000

Accuracy(%)

60

###### Tokens

10000

8000

40

6000

4000

20

2000

0

0

Very Easy Easy Medium Hard Very Hard

- Figure 30: The Pass@1 and average token distribution across different difficulty levels, from Very Easy to Very Hard, for the LeaP-T-1.5B.

| |Baseline Tokens<br><br>Ours Tokens<br><br>Baseline Accuracy<br><br>Ours Accuracy| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Very Easy Easy Medium Hard Very Hard

0

2500

5000

7500

10000

12500

15000

17500

Tokens

0

20

40

60

80

100

Accuracy(%)

- Figure 31: The Pass@1 and average token distribution across different difficulty levels, from Very Easy to Very Hard, for the LeaP-T-7B.

| |Baseline Tokens<br><br>Ours Tokens<br><br>Baseline Accuracy<br><br>Ours Accuracy| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Very Easy Easy Medium Hard Very Hard

0

2000

4000

6000

8000

10000

12000

14000

16000

Tokens

0

20

40

60

80

100

Accuracy(%)

- Figure 32: The Pass@1 and average token distribution across different difficulty levels, from Very Easy to Very Hard, for the LeaP-T-14B.

#### G.3 Behavior Difference of LeaP using RL-ed and Non-RL models

We observe that reinforcement learning (RL) models, such as QwQ, display higher confidence in their reasoning. This is particularly evident in the increasing proportion of Unaffected cases at later stages of reasoning. The reinforcement learning process tends to encourage more consistent behavior, with certain tokens’ sampling probabilities increasing, guiding the model back to familiar reasoning paths. This aligns with findings from a previous study [43], which shows that RL-trained models tend to prefer their learned reasoning strategies, exhibiting greater self-assurance in their conclusions. This confidence may explain the model’s resistance to altering its reasoning path when provided with peer insights, especially in the later stages of reasoning.

