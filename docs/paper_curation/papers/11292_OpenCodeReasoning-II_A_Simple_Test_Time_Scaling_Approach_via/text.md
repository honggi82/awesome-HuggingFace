# arXiv:2507.09075v1[cs.CL]11Jul2025

## OPENCODEREASONING-II: A Simple Test Time Scaling Approach via Self-Critique

Wasi Uddin Ahmad, Somshubra Majumdar, Aleksander Ficek, Sean Narenthiran, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Vahid Noroozi, Boris Ginsburg NVIDIA Santa Clara, CA 95051, USA {wasiuddina, smajumdar, aficek, snarenthiran}@nvidia.com https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2

#### Abstract

Recent advancements in reasoning-based Large Language Models (LLMs), particularly their potential through test-time scaling, have created significant opportunities for distillation in code generation and critique. However, progress in both areas fundamentally depends on large-scale, high-quality datasets. In this work, we introduce OPENCODEREASONING-II, a dataset consists of 2.5M questionsolution-critique triples (≈ 35K unique programming questions), making it nearly twice the size of the previous largest publicly available code reasoning dataset. In this work, we employ a two-stage supervised fine-tuning strategy. The first stage focuses on fine-tuning for code generation, while the second stage involves the joint training of models for both code generation and critique. Our resulting finetuned Qwen2.5-Instruct models achieve performance in code generation that either exceeds or equals the best prior open-weight distilled models. Notably, the integration of our code generation and critique models leads to significant improvements in competitive coding performance. Furthermore, we present an extension of the LiveCodeBench benchmark to specifically support the C++ programming language, thereby facilitating more comprehensive LLM evaluation using this benchmark.

#### 1 Introduction

Large language models (LLMs) have undergone rapid advancements in recent years, from chain-ofthought (CoT) prompting (Wei et al., 2022), followed by System-2 reasoning that employs test-time compute scaling approaches (Li et al., 2025b; Zhang et al., 2025b). Test-time scaling has allowed LLMs to dedicate more computational resources during inference to perform logical reasoning. The release of models like DeepSeek-R1 (DeepSeek-AI et al., 2025), which demonstrated impressive reasoning capabilities, has spurred increased interest in distilling these test-time compute capabilities into smaller fine-tuned models. This pursuit is driven by the observation that providing LLMs with more computational power during inference directly translates to tangible improvements in their outputs, particularly on complex tasks like competitive coding.

As models have improved on reasoning tasks by leveraging test-time compute, a natural research question emerges: how can test-time compute be effectively scaled? Inference time scaling has been found to emerge from applying reinforcement learning on domains with verifiable outcomes such as with math and coding (Setlur et al., 2025; Qu et al., 2025; Yu et al., 2025a). Other works have found benefits by scaling inference compute by using repeated sampling in parallel (Wang et al., 2023a; Brown et al., 2024; Wu et al., 2025). After sampling multiples times the best solution can be selected using a variety of methods such as majority voting, reward models or LLM-as-a-judge (Chen

Preprint.

Pass@1

Pass@1 with Self-Critique

Pass@10

| |
|---|

| |
|---|

80

10.0

8.2

11.9

11.8

9.3

11.4

70

+6.1

+3.8

+4.0

+2.1

+4.3

+4.3

60

50

OlympicCoder-32B OpenThinker2-32B QwQ-32B OCR-2-32B DeepSeek-R1 Qwen3-32B

Figure 1: Demonstrating performance gains on LiveCodeBench, achieved through test-time scaling by generating 10 solutions per problem and employing self-critique for selecting the final output. Self-critique led to the greatest performance boost in our finetuned model, OCR-2-32B.

- et al., 2024; Liu et al., 2025a; Zeng et al., 2025a; Moshkov et al., 2025). Recent works have found that scaling test-time compute also results in better critique models in the form of reasoning-based judges or Generative Reward Models (GenRM’s) (Mahan et al., 2024; Zhang et al., 2025a; Liu et al., 2025b). Additionally, Wang et al. (2025b) introduces Critique Fine-Tuning (CFT), a method that uses model-based critiques to more effectively distill reasoning capabilities than standard SupervisedFinetuning (SFT) for math problems. Our work unifies Critique Fine-Tuning (CFT) with reasoning data distillation to effectively scale test-time compute and enhance coding capabilities.

Reasoning-based data distillation has proven to be a powerful technique for enhancing coding performance, often without requiring reinforcement learning (BespokeLabs, 2025; Penedo et al., 2025b; OpenThoughts, 2025; Li et al., 2025a). Follow-up works have found continual gains from supervised fine-tuning on increasingly larger CoT reasoning datasets (Xu et al., 2025; Ahmad et al., 2025b). Despite the recognized benefits of critique fine-tuning and applying test-time compute to generative reward models, a significant gap persists: there are no publicly available coding datasets that specifically include reasoning-based CoT critiques. Existing datasets with critique data feature non-reasoning critiques, execution test pass-rates, or focus on different domains (Ahmad et al., 2025a,b; Zeng et al., 2025a; Zhang et al., 2025a; Wang et al., 2024d).

To bridge this gap and further leverage test-time scaling, we present OPENCODEREASONING-II, the largest publicly accessible code reasoning dataset created to date. This dataset comprises 2.5 million question-solution-critique triples originating from roughly 35,000 distinct programming problems. Both solutions and critiques are structured as reasoning CoTs, offering detailed justifications for solution generation and validation. Using this dataset, we trained models with 7B, 14B, and 32B parameters using a two-stage finetuning approach. We conduct evaluation under parallel test-time scaling and demonstrated significant improvements compared to open-weight models, as illustrated in Figure 1. Furthermore, we contribute an extension to LiveCodeBench, with a specific focus on supporting LLM evaluation using C++. The contributions of this work can be summarized as follows:

- 1. We introduce OPENCODEREASONING-II, a large-scale dataset containing 1.4 million Python and 1.1 million C++ solutions, along with their corresponding critique labels, detailed reasoning traces, and execution pass rates, all derived from 35,000 unique programming questions.
- 2. With the aim of facilitating more comprehensive LLM evaluation in the C++ programming language, we have extended LiveCodeBench to incorporate C++ support. This enhanced benchmark is now publicly available to promote further research and development.
- 3. We show the effectiveness of OPENCODEREASONING-II by fine-tuning Qwen2.5-Instruct models in a two-stage process. This fine-tuning enabled our models to achieve code generation performance that surpasses or matches the leading prior open-weight distilled models. Furthermore, integrating code generation with critique through a simple test-time scaling strategy resulted in substantial gains on the LiveCodeBench benchmark, in both Python and C++.
- 4. We perform an in-depth analysis to provide insights on the opportunities in self-critique methods under test-time scaling, impact of data scaling, transfer between Python and C++ languages.

OpenCodeReasoning-II Question, Thinking trace, Solution, Critique thinking trace, Judgement (binary), Execution feedback (pass/fail)

###### Programming Questions, Unit tests

[Figure 1]

[Figure 2]

[Figure 3]

<think> … </think> …

<think> … </think> …

```python # solution code ``` assert … assert … assert … assert … assert …

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Code Generation

Critique Generation

Code Execution

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

```python # solution code ```

<judgement> right/wrong </judgement>

Figure 2: Overview of the OPENCODEREASONING-II development stages.

#### 2 Development of OPENCODEREASONING-II and LiveCodeBench-C++

##### 2.1 Construction of OPENCODEREASONING-II

The construction of the OPENCODEREASONING-II dataset involved a four-stage approach which is demonstrated in Figure 2. The initial stage consisted of compiling a diverse collection of competitive coding problems with unit tests from various origins. Following this, a large language model (LLM) equipped with reasoning capabilities was leveraged to produce corresponding solutions. Next, a reasoning-capable LLM was employed to generate critique for these solutions. In the final stage, we obtained execution results for a portion of the generated solutions. During stage-II and stage-III, the generated solutions and their corresponding critiques underwent a post-processing stage aimed at guaranteeing their structural consistency.

##### 2.1.1 Programming Questions Collection

To create our dataset, we drew problems from the TACO corpus (Li et al., 2023), the APPS benchmark (Hendrycks et al., 2021), the CodeContests collection (Li et al., 2022), and CodeForces problems through the OpenR1 initiative (Penedo et al., 2025a). Due to the frequent overlap found in public datasets, we applied a fuzzy matching-based de-duplication method, leading to a final set of 34,799 unique questions of diverse difficulty. The distribution of these questions is detailed in Table 1.

Contamination Assessment To ensure the integrity of OPENCODEREASONING-II, we rigorously investigated potential data leakage between our collected programming questions and major code generation evaluation suites (Jain et al., 2025; Li et al., 2022; Chen et al., 2021; Austin et al., 2021). Our methodology mirrored the protocol outlined by Yang et al. (2023), which involved computing the cosine similarity (with a cutoff of 0.7) to identify the closest counterpart within the benchmark datasets for every distinct question in OPENCODEREASONING-II. We used Llama-3.3-70B-Instruct (Grattafiori et al., 2024) as a judge to assess semantic similarity, and it identified 674 questions that potentially overlap with evaluation benchmarks. Following this decontamination, we proceeded to generate solutions for the remaining 34,125 programming questions.

##### 2.1.2 Solution Generation using DeepSeek-R1

In this stage, we generated multiple solutions for each question leveraging the DeepSeek-R1 model (DeepSeek-AI et al., 2025). These solutions were generated in Python and C++ and sampled using Nucleus Sampling (Holtzman et al., 2020) with temperature 0.6 and top-p 0.95. We utilized SGLang (Zheng et al., 2024) for this generation process, allowing for a maximum output sequence length of 32k tokens. The prompt used to generate solutions is provided in Figure 6.

Post-processing and Filtering We post-processed and filtered the generated responses to ensure the required information were present in the output. Initially, we checked if each responses contained reasoning traces enclosed by the <think> and </think> tags. Next, we extracted the solution segments, separating the reasoning traces from the rest of the response. We then confirmed the presence of code blocks delimited by ```python ...```or ```cpp ...```. Finally, we used Tree Sitter (TreeSitter,

Python C++

Source

# Question # Sample # Question # Sample

AIZU 2151 71,681 2067 35,471 AtCoder 2080 64,468 1988 62,493 CodeChef 3869 120,040 3830 171,882 CodeForces 15641 834,523 11887 355,180 Codewars 2506 79,771 2492 155,162 GeeksForGeeks 2670 58,154 2668 167,610 HackerEarth 2285 73,559 2273 82,765 HackerRank 912 26,106 903 43,867 Kattis 1235 39,938 1209 49,699 LeetCode 777 29,926 775 50,346

Total 34,125 1,398,166 30,092 1,174,475

- Table 1: Number of questions and corresponding samples in OPENCODEREASONING-II, spanning across ten programming platforms.

2013) to validate the syntactic correctness of these code blocks. Notably, these filtering procedures led to the removal of a very few responses.

##### 2.1.3 Critique Generation using QwQ-32B

In this stage, we prompted QwQ-32B (Team, 2025b) to to generate critiques for the programming questions and their corresponding code solutions. The specific prompt used to generate critique is detailed in Figure 7. We employed the same generation settings as for solution generation: temperature-based nucleus sampling via SGLang with a maximum output length of 24k tokens. Following a similar post-processing and filtering approach to solution generation, we verified that each critique contained reasoning traces (within <think> and </think> tags) and a final judgment (within <judgment> and </judgment> tags). We retain responses only if their final judgment is binary: either right or wrong. Otherwise, we discard them. The rationale behind this choice stems from our preliminary experiments. When evaluating reasoning-enabled LLMs (R1 and QwQ-32B) with binary, categorical (correct, partially correct, incorrect), and numeric (1-5) judgment options, we observed a significant tendency for the models to favor binary responses (such as 1 or 5, or correct/incorrect). Consequently, we adopted binary judgment generation for our critique responses.

##### 2.1.4 Verifying Solutions with Unit Tests

In the final stage of OPENCODEREASONING-II construction, we executed the generated code solutions against their corresponding unit tests, which were collected alongside the questions from public benchmarks. To ensure meaningful and manageable execution outputs, we selected a subsample of OPENCODEREASONING-II where each question had at least 5 unit tests, with a maximum of 50 randomly selected if more were available. This subsample comprised 60% of OPENCODEREASONINGII. Following execution, we calculated the pass rate, which is included in the public release of OPENCODEREASONING-II. We expect these execution outputs to facilitate future research, including the application of offline reinforcement learning for LLM improvement.

##### 2.2 Extending LiveCodeBench for C++

While LiveCodeBench (Jain et al., 2025) aims to provide a contamination-free evaluation of LLMs for code, its limitation to Python hindered our ability to assess LLMs on C++, a widely used language in competitive coding. To address this, we extended LiveCodeBench to include C++. We selected problems from release_v5 within the date range of 2408 to 2502, resulting in 279 problems (175 from AtCoder and 104 from LeetCode). Notably, AtCoder problems utilize standard input/output for testing, whereas LeetCode problems provide starter code, requiring function invocation for evaluation. We collected the C++ starter code for LeetCode problems and adapted their test cases to enable evaluation in our extended benchmark. The dataset is publicly available at https://huggingface.co/ datasets/nvidia/LiveCodeBench-CPP.

#### 3 A Simple Test-time Scaling Approach via Self-Critique

To showcase the potential of OPENCODEREASONING-II, we establish a straightforward self-critiquebased test-time scaling approach as a simple baseline for future research. This section outlines our fine-tuning methodology and the subsequent inference setup for test-time scaling.

Finetuning Setup We fine-tuned the Qwen2.5-Instruct models in two stages. Stage I involved fine-tuning for code generation, followed by Stage II where we jointly fine-tuned for both code generation and self-critique. We used the same prompts for fine-tuning as those employed for data generation (illustrated in Figure 6 and Figure 7). The models underwent three epochs of fine-tuning in Stage I and one epoch in Stage II. While this work presents our initial approach, we plan to investigate more advanced fine-tuning techniques in the future.

Inference Setup for Test-Time Scaling At inference time, we prompt the fine-tuned models to first produce a solution to a programming question and then to critique their own output (self-critique). This process facilitates parallel scaling (Zeng et al., 2025b), where multiple solutions are generated concurrently, and the best is chosen as the final result. The success of parallel scaling hinges on two factors: (1) coverage, the probability of generating at least one correct solution, and (2) the selection method’s accuracy in identifying a correct solution if one or more solutions are labeled as correct. In this study, we evaluate coverage using pass@k and selection efficacy using pass@1|select@k (which we term as critique@k). A limitation of our work is the binary nature of the self-critique judgments, which necessitates a strategy for selecting the best solution among multiple (right) generations. Motivated by the findings of Wang et al. (2025a), which found that the longer reasoning traces often correlate with incorrect final solutions, we utilize a simple heuristic to choose the right solution from a pool of "right"-labeled candidates: select the solution with the shortest critique reasoning trace. A detailed comparison against a randomized selection approach is provided in Appendix A. Recognizing the naivety of this method, we leave the investigation of more sophisticated selection techniques for future work.

#### 4 Main Evaluation

Training and Inference Hyper-Parameters By leveraging OPENCODEREASONING-II, we gauged the efficacy of supervised fine-tuning (SFT) through the adaptation of Qwen2.5-Instruct models, spanning parameter counts of 7B, 14B, and 32B. The model training uses AdamW optimizer (Kingma and Ba, 2015) with a learning rate 5e − 5, a batch size of 256, and a maximum context length of 32,768 tokens. We used a CosineAnnealing learning rate schedule with a 5% warmup, and the final checkpoint was used for evaluation. To accelerate training, we utilized sequence packing (Shen et al., 2024), tensor and context parallelism, and BF16 precision. For generating outputs during inference, we employed temperature-based nucleus sampling (Holtzman et al., 2020) via vLLM (Kwon et al., 2023), setting a maximum output length of 30,720 tokens.

Baselines The following open-weight models were chosen as baselines in our evaluation: DeepSeekR1 and R1-Distill-Qwen models (DeepSeek-AI et al., 2025), QwQ-32B (Team, 2025b), Qwen332B (Team, 2025a), OlympicCoder (Penedo et al., 2025a), OpenThinker2 (OpenThoughts, 2025), DeepCoder-14B-Preview (Luo et al., 2025), and OCR-Qwen (Ahmad et al., 2025b) models.

Benchmarks and Metrics For our evaluation, we used the same LiveCodeBench (Jain et al., 2025) split that we utilized for our C++ expansion. This benchmark contains 67 easy, 89 medium, and 279 hard coding questions. We report pass@1 for code generation, pass@1|select@k under test-time scaling setup. Furthermore, we report the accuracy of the self-critique capability. An LLM’s prediction is considered correct if it accurately judges the correctness of all generated solutions (k unless otherwise mentioend) for a given input question. We also evaluate critique accuracy using CodeContests benchmark (Li et al., 2022) and provide details in Appendix B.

##### 4.1 Main Results

Tables 2 and 3 summarize the performance of our distilled models against various competing baselines. We consistently observed the following three trends.

|Model<br><br>|LiveCodeBench-Python Easy Medium Hard All|LiveCodeBench-C++ Easy Medium Hard All|
|---|---|---|
|DeepSeek-R1 QwQ-32B Qwen3-32B|98.5 79.8 37.4 65.6<br><br>97.0 79.8 28.5 61.3<br>98.0 79.9 34.6 64.3<br>|95.5 75.3 29.3 59.9 94.0 68.5 26.0 55.9<br>96.3 73.0 35.0 61.9<br>|

##### Distilled 7B+ Models

R1-Distill-Qwen-7B 86.6 43.8 7.0 38.0 26.9 5.6 1.6 9.0 OpenThinker2-7B 80.6 16.9 1.6 25.5 43.3 2.3 0 11.1 OlympicCoder-7B 82.1 49.4 12.2 40.9 85.7 46.7 10.2 40.0 OCR-Qwen-7B-Instruct 95.4 64.0 18.0 51.3 13.4 2.3 0.8 4.3 OCR-2-7B 97.0 71.1 20.9 55.2 91.4 64.5 21.2 51.9

##### Distilled 14B+ Models

R1-Distill-Qwen-14B 98.5 62.9 17.1 51.3 68.7 39.3 6.5 31.9 DeepCoder-14B-Preview 97.0 65.2 19.5 52.7 61.2 39.3 8.9 39.3 OCR-Qwen-14B-Instruct 97.6 74.4 27.6 59.4 47.8 16.9 0.8 17.2 OCR-2-14B 97.9 75.4 26.8 59.4 91.8 68.6 25.7 55.3

##### Distilled 32B+ Models

R1-Distill-Qwen-32B 98.5 68.5 28.5 58.1 80.6 39.3 11.4 36.9 OpenThinker2-32B 97.0 65.2 22.8 54.1 97.0 60.7 25.2 53.8 OlympicCoder-32B 98.5 71.9 24.4 57.4 91.0 62.8 21.3 51.4 OCR-Qwen-32B-Instruct 98.4 77.2 30.4 61.7 65.7 33.7 4.1 28.3 OCR-2-32B 97.9 77.1 31.8 62.1 94.7 72.2 28.0 58.1

- Table 2: Performance comparison of reasoning models on LiveCodeBench. Highlighted rows show our finetuned models’ performances. Bold indicates the highest performance. Python results are averaged across 64 runs, and C++ results across 16 runs.

|Model<br><br>|Pass@1 Python C++<br><br>|Pass@10 Python C++<br><br>|Pass@1|Select@10 Python C++|
|---|---|---|---|
|DeepSeek-R1 QwQ-32B Qwen3-32B|61.3 60.1 60.2 54.1 63.6 61.9<br><br>|75.3 72.0 73.5 70.3 77.4 75.3|63.4 (+2.1) 62.7 (+2.6)<br>64.2 (+4.0) 56.3 (+2.2) 67.4 (+3.8) 63.8 (+1.9)<br>|
|DeepCoder-14B-Preview OpenThinker2-32B OlympicCoder-32B OCR-2-7B OCR-2-14B OCR-2-32B<br><br>|53.0 29.4 58.1 51.6 55.6 52.3 55.2 51.8 58.6 56.4 61.3 59.8<br><br>|65.9 54.5<br><br>74.2 68.8<br><br>71.3 68.5 67.7 69.9<br><br>72.0 69.5<br><br><br>75.6 73.1<br><br><br>|57.7 (+4.4) 35.1 (+5.7) 62.4 (+4.3) 53.4 (+1.8)<br><br>59.9 (+4.3) 55.6 (+3.3)<br><br>60.2 (+5.0) 54.1 (+2.3) 60.6 (+2.0) 58.4 (+2.0) 67.4 (+6.1) 60.6 (+0.8)<br><br><br>|

- Table 3: Performance comparison of reasoning models under test-time scaling setup. Highlighted rows show our finetuned models’ performances. The pass@1 scores are averaged over 10 runs. The performance gains with self-critique are highlighted in blue and bold values indicate the largest gains.

Scaling Yields Large Performance Boosts for Smaller Models A comparison between our OCR-2 models and the previous OCR-Qwen models in Table 2 demonstrates that scaling the quantity of synthetic solutions particularly benefits smaller models. Increasing the fine-tuning samples from 737K to 2.5M yields a more substantial relative performance improvement for smaller models (e.g., 7B parameters) compared to their larger counterparts. This trend is also evident in C++ performance, especially when compared to models like OlympicCoder, which were trained on C++ data. Although the 32B parameter model also improves, the gains suggest it might be approaching a performance ceiling achievable through mere scaling of synthetic data quantity for existing problem types.

LLMs Show Similar Capabilities in Python and C++ Notably, OPENCODEREASONING-II includes approximately 1.4M Python samples and 1.17M C++ samples as seen in Table 1. Consequently,

|Model|LiveCodeBench-Python Easy Medium Hard All|LiveCodeBench-C++ Easy Medium Hard All<br><br>|
|---|---|---|
|DeepSeek-R1 QwQ-32B Qwen3-32B|91.0 47.2 8.9 40.9 67.2 39.3 9.8 33.0 77.6 41.6 9.8 36.2<br><br>|82.1 40.4 13.0 38.4 74.6 25.8 9.8 30.5 74.6 30.3 9.8 31.9|
|DeepCoder-14B-Preview OpenThinker2-32B OlympicCoder-32B OCR-2-7B OCR-2-14B OCR-2-32B<br><br>|10.4 1.1 0 2.9<br>11.9 2.2 0.8 3.9 56.7 39.3 4.9 28.3 80.6 47.2 14.6 40.9 88.1 43.8 16.3 42.3 92.5 53.9 17.1 47.0<br><br><br>|4.5 1.1 0 1.4 35.8 1.1 0.8 9.3 59.7 27.0 0.8 23.3 70.1 24.7 8.9 28.7 82.1 36.0 8.9 35.1 80.6 44.9 15.5 40.5<br><br>|

- Table 4: Accuracy comparison of reasoning models on self-critique. Highlighted rows show our finetuned models’ performances. Bold indicates the best performance.

our models maintain or improve their Python capabilities while demonstrating dramatically enhanced C++ solution quality. Despite being trained on both languages, our models exhibit significantly superior performance on LiveCodeBench-C++ compared to other models of similar size. In particular, the 32B parameter model we train exceeds QwQ-32B and other open-weight reasoning models in both Python and C++. Overall, joint training on substantial, comparably-sized Python and C++ solution sets results in strong performance in both languages, often outperforming models trained on single-language data. We anticipate this positive transfer learning behavior could extend to other programming languages, a direction we leave for future work.

Test-time Scaling with Self-Critique Yields Significant Improvements Table 3 highlights the benefits of applying self-critique at test-time, a capability developed by fine-tuning models on data that includes self-critique labels. For instance, when considering our flagship OCR-2-32B model, using its self-critique ability to select the best solution advances the Pass@1 score by approximately 6 percentage points. This enhancement significantly narrows the performance gap between Pass@1 and Pass@10 for all our model sizes, in both Python and C++, and results in our models outperforming other similarly sized competitors. Therefore, training with self-critique data not only improves baseline Pass@1 scores but also demonstrates that self-critique is an effective test-time strategy for selecting a higher accuracy solution from multiple parallel generations.

- 5 Ablation and Analyses

##### 5.1 Test-time Scaling: Opportunities for Enhanced Critique

Although self-critique based selection under parallel test-time scaling has proven effective in boosting LLMs’ performance with a limited number of samples, an important question left to address is, what is the gap between a model’s pass@1, pass@1|select@k, and pass@k as k grows? In this ablation, we aim to answer this question using a far larger value of k (up to 100) and show the gap that exists between them. It can be seen in Figure 3 that OCR-2-32B quickly attains a high pass@k score. For comparison, we plot the pass@1 score for each individual sample, computed independently from the rest of the samples, as well as the pass@1|select@k score, as k increases to include all past solutions.

Two important observations can be drawn from this figure. First, while pass@1|select@k is consistently higher than pass@1, this increase does not go beyond a certain limit. This observation is consistent with the fact that out of the 279 problems in LiveCodeBench split we evaluate on, nearly 212 problems fall under the Medium/Hard category where self-critique accuracy is insufficient to correctly determine the correctness of the all generated samples, as can be seen in Table 4. On evaluating the accuracy of the critique labels using various models on a single solution for each problem, we find that the accuracy of these models for medium difficulty problems tends to be 47% and for hard problems is less than 14% at best when critiquing 10 solutions. This inaccurate final determination of correctness label limits the scope of improvement on the overall benchmark, as the vast majority (75.9%) of samples in LiveCodeBench are inaccurately critiqued. Secondly, the simple heuristic described in section 3, which is to select the solution with shortest critique reasoning trace, may be ineffective in incorporating information from additional solutions and prevents substantial

82

78

74

PassRate(%)

70

pass@1 pass@1|select@k pass@k

66

62

58

54

0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100

Number of Samples

Figure 3: Performance gap between pass@1, pass@1|select@k, and pass@k under test-time scaling - large number of samples drawn from OCR-2-32B.

improvements. We leave the exploration of more sophisticated heuristics and methods to enhance the accuracy of self-critique-based selection for future research.

##### 5.2 Impact of Temperature on Self Critique

We tested how the critic LLMs responded to different decoding temperatures by re-evaluating them at temperature [0.2,0.4,0.6,0.7]. The pass@1|select@k and the accuracy of self-critique showed minimal variation (less than 3%) across these temperatures, and t-tests indicated no statistically significant differences. These findings suggest that within the range below 0.7, the decoding temperature does not substantially affect the critic’s performance. This implies that the model’s judgments are mainly based on its internal knowledge rather than the randomness of the sampling process. Therefore, we used a temperature of 0.6 throughout this work.

##### 5.3 Transfer Learning: Python ↔ C++

OPENCODEREASONING-II features solutions in both Python and C++, allowing study to investigate how well models perform when trained on a single language and tested on the other. The results of these experiments are detailed in Table 5. First, our findings suggest that crosslanguage transfer does occur, and combining Python and C++ data during training enhances overall performance on both languages. Second, we observe an asymmetry: models trained solely on C++ achieve noticeable scores on Python, whereas models trained only on Python, while performing well on Python, experience a significant drop in accuracy on C++. This difference isn’t simply due to dataset size, and further research is needed to understand the underlying causes of this performance degradation.

Dataset Size LiveCodeBench

Model

Python C++ Python C++

1.4M 0 54.9 1.7

OCR-2-7B

0 1.1M 36.0 45.3 1.4M 1.1M 55.2 51.9

1.4M 0 56.3 9.8

OCR-2-14B

0 1.1M 54.0 53.6 1.4M 1.1M 59.4 55.3

1.4M 0 60.5 16.0 0 1.1M 56.6 56.4 1.4M 1.1M 62.1 59.4

OCR-2-32B

Table 5: Pass@1 scores of OCR-2 models trained individually on Python and C++ vs. jointly using OPENCODEREASONING-II.

##### 5.4 Impact of Scaling Up Data on Code Generation

We analyze the data scaling study of OpenCodeReasoning (Ahmad et al., 2025b), and substantially increase the dataset size in order to determine whether data scaling shows limits for a given model size. As such, we redo the scaling study from 25K samples all the way to 1.4M samples in Python using OPENCODEREASONING-II and plot the trajectory of scores on LiveCodeBench in Figure 4. While the 7B model shows substantial improvements in score with data scaling, such improvement is not observed in the 14B and 32B models. It remains to be seen if access to more novel and complex

65

| |
|---|

| |
|---|

60

| |
|---|

| |
|---|

55

LiveCodeBench

50

Pass@1in

45

40

35

30

OCR-2-7B

OCR-2-14B

OCR-2-32B

25

R1-Distill-Qwen-7B

R1-Distill-Qwen-14B

R1-Distill-Qwen-32B

25,000 50,000 100,000 200,000 400,000 700,000 1,400,000

Number of samples from OpenCodeReasoning-II

- Figure 4: Impact of scaling up data from 25k to 1.4M samples in OPENCODEREASONING-II.

instructions may further improve scores, or if the number of solutions to the existing problems must be scaled by orders of magnitude to improve scores further measurably.

#### 6 Related Works

Our work builds upon a foundation of research in synthetic data generation for code, the growing reasoning capabilities of LLMs, and model-based critique, particularly within the coding context. The demonstrated effectiveness of LLMs in coding has spurred the creation of numerous impactful synthetic datasets for instruction tuning (Wang et al., 2023b; Wei et al., 2024b; Xu et al., 2024; Wu et al., 2024; Luo et al., 2024; Wei et al., 2024a; Majumdar et al., 2024; Ahmad et al., 2025a). Inspired by the positive impact of extended inference-time computation on code quality, synthetic data generation has also been adapted for training LLMs with reasoning abilities for code. These efforts have shown substantial gains from fine-tuning on as few as 17k reasoning-style CoT examples (Li et al., 2025a) and scaling up to 447k (BespokeLabs, 2025; Penedo et al., 2025b; OpenThoughts, 2025; Li et al., 2025a; Xu et al., 2025). Recent work by Ahmad et al. (2025b) further scaled this data distillation strategy to 737k samples, achieving superior SFT performance in code and reasoning foundation models (Bercovich et al., 2025). In this paper, we aim to push the boundaries of synthetic data scaling for coding and, crucially, explore critique fine-tuning for reasoning data distillation.

Training reward models to critique solutions is a well-explored research area across various domains. Generalist reward models, trained on preference pairs, have shown strong performance in alignment and reasoning (Nvidia et al., 2024; Liu et al., 2024; Wang et al., 2024c,b). Specialized reward models have also been developed for tasks like math or coding (Wang et al., 2024a; Zeng et al., 2025a; Liu et al., 2025a; Zhang et al., 2025c; Yang et al., 2024). Notably, some of these works have released their training datasets, providing valuable resources for the research community.

LLM-based solution critiques, when combined with reasoning, significantly enhance model capabilities in both mathematical and coding tasks. Increased test-time computation consistently improves verifiers like reward models and test case generators (Mahan et al., 2024; Ficek et al., 2025; Liu et al., 2025b; Chen et al., 2025; Moshkov et al., 2025), as demonstrated by Zhang et al. (2025a)’s improved math reward model and released CoT data. Moreover, critique fine-tuning positively impacts question-answering (Sun et al., 2024; Yu et al., 2025b). This synergy, further explored in recent work combining LLM critiques with reasoning-driven test-time scaling, offers a promising route to advance coding abilities (Wang et al., 2025b; Zhou et al., 2025).

#### 7 Conclusion

This research addresses the critical need for high-quality large-scale data to propel advancements in reasoning-based LLMs for test-time scaling. We introduced OPENCODEREASONING-II, a significantly larger and richer dataset of question-solution-critique triples that has enabled us to train powerful distilled models. Our two-stage fine-tuning approach yielded Qwen2.5-Instruct models that demonstrate state-of-the-art or comparable code generation capabilities among open-weight distilled models. More importantly, the synergistic combination of our code generation and critique models led to tangible improvements in competitive coding benchmarks. Additionally, our C++ extension of LiveCodeBench broadens the evaluation landscape for LLMs in code.

#### References

Ahmad, W.U., Ficek, A., Samadi, M., Huang, J., Noroozi, V., Majumdar, S., Ginsburg, B., 2025a. Opencodeinstruct: A large-scale instruction tuning dataset for code llms. URL: https://arxiv.org/ abs/2504.04030, arXiv:2504.04030.

Ahmad, W.U., Narenthiran, S., Majumdar, S., Ficek, A., Jain, S., Huang, J., Noroozi, V., Ginsburg, B., 2025b. Opencodereasoning: Advancing data distillation for competitive coding. URL: https://arxiv.org/abs/2504.01943, arXiv:2504.01943.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., Sutton, C., 2021. Program synthesis with large language models. URL: https: //arxiv.org/abs/2108.07732, arXiv:2108.07732.

Bercovich, A., Levy, I., Golan, I., Dabbah, M., El-Yaniv, R., Puny, O., Galil, I., Moshe, Z., Ronen, T., Nabwani, N., Shahaf, I., Tropp, O., Karpas, E., Zilberstein, R., Zeng, J., Singhal, S., Bukharin, A., Zhang, Y., Konuk, T., Shen, G., Mahabaleshwarkar, A.S., Kartal, B., Suhara, Y., Delalleau, O., Chen, Z., Wang, Z., Mosallanezhad, D., Renduchintala, A., Qian, H., Rekesh, D., Jia, F., Majumdar, S., Noroozi, V., Ahmad, W.U., Narenthiran, S., Ficek, A., Samadi, M., Huang, J., Jain,

- S., Gitman, I., Moshkov, I., Du, W., Toshniwal, S., Armstrong, G., Kisacanin, B., Novikov, M., Gitman, D., Bakhturina, E., Scowcroft, J.P., Kamalu, J., Su, D., Kong, K., Kliegl, M., Karimi, R., Lin, Y., Satheesh, S., Parmar, J., Gundecha, P., Norick, B., Jennings, J., Prabhumoye, S., Akter, S.N., Patwary, M., Khattar, A., Narayanan, D., Waleffe, R., Zhang, J., Su, B.Y., Huang, G., Kong, T., Chadha, P., Jain, S., Harvey, C., Segal, E., Huang, J., Kashirsky, S., McQueen, R., Putterman, I., Lam, G., Venkatesan, A., Wu, S., Nguyen, V., Kilaru, M., Wang, A., Warno, A., Somasamudramath, A., Bhaskar, S., Dong, M., Assaf, N., Mor, S., Argov, O.U., Junkin, S., Romanenko, O., Larroy, P., Katariya, M., Rovinelli, M., Balas, V., Edelman, N., Bhiwandiwalla, A., Subramaniam, M., Ithape, S., Ramamoorthy, K., Wu, Y., Velury, S.V., Almog, O., Daw, J., Fridman, D., Galinkin, E., Evans, M., Ghosh, S., Luna, K., Derczynski, L., Pope, N., Long, E., Schneider, S., Siman, G., Grzegorzek, T., Ribalta, P., Katariya, M., Alexiuk, C., Conway, J., Saar,
- T., Guan, A., Pawelec, K., Prayaga, S., Kuchaiev, O., Ginsburg, B., Olabiyi, O., Briski, K., Cohen, J., Catanzaro, B., Alben, J., Geifman, Y., Chung, E., 2025. Llama-nemotron: Efficient reasoning models. URL: https://arxiv.org/abs/2505.00949, arXiv:2505.00949.

BespokeLabs, 2025. Bespoke-stratos: The unreasonable effectiveness of reasoning distillation. www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoningdistillation. Accessed: 2025-01-22.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q.V., Ré, C., Mirhoseini, A., 2024. Large language monkeys: Scaling inference compute with repeated sampling. URL: https://arxiv.org/abs/2407. 21787, arXiv:2407.21787.

- Chen, L., Davis, J.Q., Hanin, B., Bailis, P., Stoica, I., Zaharia, M., Zou, J., 2024. Are more llm calls all you need? towards scaling laws of compound inference systems. URL: https: //arxiv.org/abs/2403.02419, arXiv:2403.02419.
- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H.P.D.O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al., 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 .

Chen, X., Li, G., Wang, Z., Jin, B., Qian, C., Wang, Y., Wang, H., Zhang, Y., Zhang, D., Zhang, T., Tong, H., Ji, H., 2025. Rm-r1: Reward modeling as reasoning. URL: https://arxiv.org/abs/2505. 02387, arXiv:2505.02387.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z.F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J.L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q.,

Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R.J., Jin, R.L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S.S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W.L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X.Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X.,

- Wang, X., Shan, X., Li, Y.K., Wang, Y.Q., Wei, Y.X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y.,
- Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y.X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren,
- Z.Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., Zhang, Z.,

2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. URL: https://arxiv.org/abs/2501.12948, arXiv:2501.12948.

Ficek, A., Majumdar, S., Noroozi, V., Ginsburg, B., 2025. Scoring verifiers: Evaluating synthetic verification in code and reasoning. URL: https://arxiv.org/abs/2502.13820, arXiv:2502.13820.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al., 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 .

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., Steinhardt, J., 2021. Measuring coding challenge competence with apps. NeurIPS .

Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y., 2020. The curious case of neural text degeneration, in: International Conference on Learning Representations. URL: https://openreview. net/forum?id=rygGQyrFvH.

Jain, N., Han, K., Gu, A., Li, W.D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., Stoica, I., 2025. Livecodebench: Holistic and contamination free evaluation of large language models for code, in: The Thirteenth International Conference on Learning Representations. URL: https://openreview.net/forum?id=chfJJYC3iL.

Kingma, D.P., Ba, J., 2015. Adam: A method for stochastic optimization, in: Bengio, Y., LeCun, Y. (Eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings. URL: http://arxiv.org/abs/1412.6980.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., Stoica, I.,

2023. Efficient memory management for large language model serving with pagedattention, in: Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Li, D., Cao, S., Griggs, T., Liu, S., Mo, X., Tang, E., Hegde, S., Hakhamaneshi, K., Patil, S.G., Zaharia, M., Gonzalez, J.E., Stoica, I., 2025a. Llms can easily learn to reason from demonstrations structure, not content, is what matters! URL: https://arxiv.org/abs/2502.07374, arXiv:2502.07374.

Li, R., Fu, J., Zhang, B.W., Huang, T., Sun, Z., Lyu, C., Liu, G., Jin, Z., Li, G., 2023. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852 .

- Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., Hubert, T., Choy, P., de Masson d’Autume, C., Babuschkin, I., Chen, X., Huang, P.S., Welbl, J., Gowal, S., Cherepanov, A., Molloy, J., Mankowitz, D., Sutherland Robson, E., Kohli, P., de Freitas, N., Kavukcuoglu, K., Vinyals, O., 2022. Competition-level code generation with alphacode. arXiv preprint arXiv:2203.07814 .
- Li, Z.Z., Zhang, D., Zhang, M.L., Zhang, J., Liu, Z., Yao, Y., Xu, H., Zheng, J., Wang, P.J., Chen, X., Zhang, Y., Yin, F., Dong, J., Li, Z., Bi, B.L., Mei, L.R., Fang, J., Guo, Z., Song, L., Liu, C.L., 2025b. From system 1 to system 2: A survey of reasoning large language models. URL: https://arxiv.org/abs/2502.17419, arXiv:2502.17419.

Liu, C.Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C., Yan, S., Liu, Y., Zhou, Y., 2024. Skyworkreward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451 .

Liu, Z., Chen, Y., Shoeybi, M., Catanzaro, B., Ping, W., 2025a. Acemath: Advancing frontier math reasoning with post-training and reward modeling. URL: https://arxiv.org/abs/2412.15084, arXiv:2412.15084.

Liu, Z., Wang, P., Xu, R., Ma, S., Ruan, C., Li, P., Liu, Y., Wu, Y., 2025b. Inference-time scaling for generalist reward modeling. URL: https://arxiv.org/abs/2504.02495, arXiv:2504.02495.

Luo, M., Tan, S., Huang, R., Patel, A., Ariyak, A., Wu, Q., Shi, X., Xin, R., Cai, C., Weber, M., Zhang, C., Li, L.E., Popa, R.A., Stoica, I., 2025. Deepcoder: A fully open-source 14b coder at o3-mini level. https://pretty-radio-b75.notion.site/ DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51.

Notion Blog.

Luo, Z., Xu, C., Zhao, P., Sun, Q., Geng, X., Hu, W., Tao, C., Ma, J., Lin, Q., Jiang, D., 2024. Wizardcoder: Empowering code large language models with evol-instruct, in: The Twelfth International Conference on Learning Representations. URL: https://openreview.net/forum?id=UnUwSIgK5W.

Mahan, D., Phung, D.V., Rafailov, R., Blagden, C., Lile, N., Castricato, L., Fränken, J.P., Finn, C., Albalak, A., 2024. Generative reward models. URL: https://arxiv.org/abs/2410.12832, arXiv:2410.12832.

Majumdar, S., Noroozi, V., Narenthiran, S., Ficek, A., Balam, J., Ginsburg, B., 2024. Genetic instruct: Scaling up synthetic generation of coding instructions for large language models. arXiv preprint arXiv:2407.21077 .

Moshkov, I., Hanley, D., Sorokin, I., Toshniwal, S., Henkel, C., Schifferer, B., Du, W., Gitman, I.,

2025. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. URL: https://arxiv.org/abs/2504.16891, arXiv:2504.16891.

Nvidia, :, Adler, B., Agarwal, N., Aithal, A., Anh, D.H., Bhattacharya, P., Brundyn, A., Casper, J., Catanzaro, B., Clay, S., Cohen, J., Das, S., Dattagupta, A., Delalleau, O., Derczynski, L., Dong, Y., Egert, D., Evans, E., Ficek, A., Fridman, D., Ghosh, S., Ginsburg, B., Gitman, I., Grzegorzek, T., Hero, R., Huang, J., Jawa, V., Jennings, J., Jhunjhunwala, A., Kamalu, J., Khan, S., Kuchaiev, O., LeGresley, P., Li, H., Liu, J., Liu, Z., Long, E., Mahabaleshwarkar, A.S., Majumdar, S., Maki, J., Martinez, M., de Melo, M.R., Moshkov, I., Narayanan, D., Narenthiran, S., Navarro, J., Nguyen, P., Nitski, O., Noroozi, V., Nutheti, G., Parisien, C., Parmar, J., Patwary, M., Pawelec, K., Ping, W., Prabhumoye, S., Roy, R., Saar, T., Sabavat, V.R.N., Satheesh, S., Scowcroft, J.P., Sewall, J., Shamis, P., Shen, G., Shoeybi, M., Sizer, D., Smelyanskiy, M., Soares, F., Sreedhar, M.N., Su, D., Subramanian, S., Sun, S., Toshniwal, S., Wang, H., Wang, Z., You, J., Zeng, J., Zhang, J., Zhang, J., Zhang, V., Zhang, Y., Zhu, C., 2024. Nemotron-4 340b technical report. URL: https://arxiv.org/abs/2406.11704, arXiv:2406.11704.

OpenThoughts, 2025. Open Thoughts. https://open-thoughts.ai. Penedo, G., Lozhkov, A., Kydlíˇcek, H., Allal, L.B., Beeching, E., Lajarín, A.P., Gallouédec, Q.,

Habib, N., Tunstall, L., von Werra, L., 2025a. Codeforces. https://huggingface.co/datasets/open-r1/ codeforces.

Penedo, G., Lozhkov, A., Kydlíˇcek, H., Allal, L.B., Beeching, E., Lajarín, A.P., Gallouédec, Q., Habib, N., Tunstall, L., von Werra, L., 2025b. Codeforces cots. https://huggingface.co/datasets/ open-r1/codeforces-cots.

Qu, Y., Yang, M.Y.R., Setlur, A., Tunstall, L., Beeching, E.E., Salakhutdinov, R., Kumar, A., 2025. Optimizing test-time compute via meta reinforcement fine-tuning. URL: https://arxiv.org/abs/2503. 07572, arXiv:2503.07572.

Setlur, A., Rajaraman, N., Levine, S., Kumar, A., 2025. Scaling test-time compute without verification or rl is suboptimal. URL: https://arxiv.org/abs/2502.12118, arXiv:2502.12118.

Shen, G., Wang, Z., Delalleau, O., Zeng, J., Dong, Y., Egert, D., Sun, S., Zhang, J.J., Jain, S., Taghibakhshi, A., Ausin, M.S., Aithal, A., Kuchaiev, O., 2024. Nemo-aligner: Scalable toolkit for efficient model alignment, in: First Conference on Language Modeling. URL: https://openreview. net/forum?id=yK2eGE8QVW.

Sun, S., Li, J., Yuan, W., Yuan, R., Li, W., Liu, P., 2024. The critique of critique, in: Ku, L.W., Martins, A., Srikumar, V. (Eds.), Findings of the Association for Computational Linguistics: ACL 2024, Association for Computational Linguistics, Bangkok, Thailand. pp. 9077–9096. URL: https://aclanthology.org/2024.findings-acl.538/, doi:10.18653/v1/2024.findings-acl.538.

- Team, Q., 2025a. Qwen3. URL: https://qwenlm.github.io/blog/qwen3/.
- Team, Q., 2025b. Qwq-32b: Embracing the power of reinforcement learning. URL: https://qwenlm. github.io/blog/qwq-32b/.

TreeSitter, 2013. Tree sitter. https://github.com/tree-sitter/tree-sitter. Wang, P., Li, L., Shao, Z., Xu, R.X., Dai, D., Li, Y., Chen, D., Wu, Y., Sui, Z., 2024a. Math-shepherd:

Verify and reinforce llms step-by-step without human annotations. URL: https://arxiv.org/abs/2312. 08935, arXiv:2312.08935.

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D., 2023a. Self-consistency improves chain of thought reasoning in language models. URL: https://arxiv.org/ abs/2203.11171, arXiv:2203.11171.
- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N.A., Khashabi, D., Hajishirzi, H., 2023b. Selfinstruct: Aligning language models with self-generated instructions, in: Rogers, A., BoydGraber, J., Okazaki, N. (Eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Association for Computational Linguistics, Toronto, Canada. pp. 13484–13508. URL: https://aclanthology.org/2023.acl-long.754/, doi:10.18653/v1/2023.acl-long.754.

- Wang, Y., Liu, Q., Xu, J., Liang, T., Chen, X., He, Z., Song, L., Yu, D., Li, J., Zhang, Z., Wang, R., Tu, Z., Mi, H., Yu, D., 2025a. Thoughts are all over the place: On the underthinking of o1-like llms. URL: https://arxiv.org/abs/2501.18585, arXiv:2501.18585.

- Wang, Y., Yue, X., Chen, W., 2025b. Critique fine-tuning: Learning to critique is more effective than learning to imitate. URL: https://arxiv.org/abs/2501.17703, arXiv:2501.17703.
- Wang, Z., Bukharin, A., Delalleau, O., Egert, D., Shen, G., Zeng, J., Kuchaiev, O., Dong, Y., 2024b. Helpsteer2-preference: Complementing ratings with preferences. URL: https://arxiv.org/abs/2410. 01257, arXiv:2410.01257.

- Wang, Z., Dong, Y., Delalleau, O., Zeng, J., Shen, G., Egert, D., Zhang, J.J., Sreedhar, M.N., Kuchaiev, O., 2024c. Helpsteer2: Open-source dataset for training top-performing reward models. arXiv:2406.08673.

- Wang, Z., Dong, Y., Zeng, J., Adams, V., Sreedhar, M.N., Egert, D., Delalleau, O., Scowcroft, J., Kant, N., Swope, A., Kuchaiev, O., 2024d. HelpSteer: Multi-attribute helpfulness dataset for SteerLM, in: Duh, K., Gomez, H., Bethard, S. (Eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Association for Computational Linguistics, Mexico City, Mexico. pp. 3371–3384. URL: https://aclanthology.org/2024.naacl-long.185/, doi:10.18653/v1/ 2024.naacl-long.185.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., brian ichter, Xia, F., Chi, E.H., Le, Q.V., Zhou, D., 2022. Chain of thought prompting elicits reasoning in large language models, in: Oh, A.H., Agarwal, A., Belgrave, D., Cho, K. (Eds.), Advances in Neural Information Processing Systems. URL: https://openreview.net/forum?id=_VjQlMeSB_J.

Wei, Y., Cassano, F., Liu, J., Ding, Y., Jain, N., Mueller, Z., de Vries, H., Werra, L.V., Guha, A., ZHANG, L., 2024a. Selfcodealign: Self-alignment for code generation, in: The Thirty-eighth Annual Conference on Neural Information Processing Systems. URL: https://openreview.net/ forum?id=xXRnUU7xTL.

Wei, Y., Wang, Z., Liu, J., Ding, Y., Zhang, L., 2024b. Magicoder: empowering code generation with oss-instruct, in: Proceedings of the 41st International Conference on Machine Learning, JMLR.org.

Wu, Y., Huang, D., Shi, W., Wang, W., Gao, L., Liu, S., Nan, Z., Yuan, K., Zhang, R., Zhang, X., et al.,

2024. Inversecoder: Unleashing the power of instruction-tuned code llms with inverse-instruct. arXiv preprint arXiv:2407.05700 .

Wu, Y., Sun, Z., Li, S., Welleck, S., Yang, Y., 2025. Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models. URL: https://arxiv.org/abs/ 2408.00724, arXiv:2408.00724.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., Jiang, D., 2024. WizardLM: Empowering large pre-trained language models to follow complex instructions, in: The Twelfth International Conference on Learning Representations. URL: https://openreview.net/forum?id= CfXh93NDgH.

Xu, Z., Liu, Y., Yin, Y., Zhou, M., Poovendran, R., 2025. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding. URL: https://arxiv.org/abs/2503.02951, arXiv:2503.02951.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., Lu, K., Xue, M., Lin, R., Liu, T., Ren, X., Zhang, Z., 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122 .

Yang, S., Chiang, W.L., Zheng, L., Gonzalez, J.E., Stoica, I., 2023. Rethinking benchmark and contamination for language models with rephrased samples. arXiv:2311.04850.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Fan, T., Liu, G., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M., Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang, C., Yu, H., Dai, W., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W.Y., Zhang, Y.Q., Yan, L., Qiao, M., Wu, Y., Wang, M., 2025a. Dapo: An open-source llm reinforcement learning system at scale. URL: https://arxiv.org/abs/2503.14476, arXiv:2503.14476.

Yu, Y., Chen, Z., Zhang, A., Tan, L., Zhu, C., Pang, R.Y., Qian, Y., Wang, X., Gururangan, S., Zhang, C., Kambadur, M., Mahajan, D., Hou, R., 2025b. Self-generated critiques boost reward modeling for language models, in: Chiruzzo, L., Ritter, A., Wang, L. (Eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Association for Computational Linguistics, Albuquerque, New Mexico. pp. 11499–11514. URL: https://aclanthology.org/2025.naacl-long.573/.

Zeng, H., Jiang, D., Wang, H., Nie, P., Chen, X., Chen, W., 2025a. Acecoder: Acing coder rl via automated test-case synthesis. URL: https://arxiv.org/abs/2502.01718, arXiv:2502.01718.

Zeng, Z., Cheng, Q., Yin, Z., Zhou, Y., Qiu, X., 2025b. Revisiting the test-time scaling of o1-like models: Do they truly possess test-time scaling capabilities? URL: https://arxiv.org/abs/2502. 12215, arXiv:2502.12215.

Zhang, L., Hosseini, A., Bansal, H., Kazemi, M., Kumar, A., Agarwal, R., 2025a. Generative verifiers: Reward modeling as next-token prediction. URL: https://arxiv.org/abs/2408.15240, arXiv:2408.15240.

Zhang, Q., Lyu, F., Sun, Z., Wang, L., Zhang, W., Hua, W., Wu, H., Guo, Z., Wang, Y., Muennighoff, N., King, I., Liu, X., Ma, C., 2025b. A survey on test-time scaling in large language models: What, how, where, and how well? URL: https://arxiv.org/abs/2503.24235, arXiv:2503.24235.

Zhang, Z., Zheng, C., Wu, Y., Zhang, B., Lin, R., Yu, B., Liu, D., Zhou, J., Lin, J., 2025c. The lessons of developing process reward models in mathematical reasoning. URL: https://arxiv.org/abs/2501. 07301, arXiv:2501.07301.

Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C.H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J.E., Barrett, C., Sheng, Y., 2024. SGLang: Efficient execution of structured language model programs, in: The Thirty-eighth Annual Conference on Neural Information Processing Systems. URL: https://openreview.net/forum?id=VqkAKQibpq.

Zhou, C., Zhang, X., Song, D., Chen, X., Gu, W., Ma, H., Tian, Y., Zhang, M., Hu, L., 2025. Refinecoder: Iterative improving of large language models via adaptive critique refinement for code generation. URL: https://arxiv.org/abs/2502.09183, arXiv:2502.09183.

### Technical Appendices and Supplementary Material

#### A Final Output Selection using Critique

To calculate pass@1, we select the shortest reasoning trace of the critique. A straightforward baseline for final solution selection is uniform random selection. Figure 5 contrasts these two selection methods, revealing that our heuristic for choosing the shortest trace yields considerably better scores than randomly picking from the positive critique samples. We hypothesize that the critique model’s comparatively lower accuracy in identifying correct solutions for medium and hard problems explains the substantially weaker performance of the random selection baseline.

pass@1 pass@1 with random critique

pass@1 with shortest critique pass@10

| |
|---|

| |
|---|

| |
|---|

80

77.4

75.6

75.3

75

73.5

72.0

70

67.7 67.4

67.4

64.9

64.2

65

63.6

63.4

62.7

62.0

61.3

61.3

60.2 60.6

60.2

59.9

60

58.6

58.1

55.2

54.8

55

50

OCR-2-7B OCR-2-14B OCR-2-32B QwQ-32B R1 Qwen3-32B

Figure 5: Differences in pass@1 scores between randomly selecting the final output vs. choosing right solution with shortest critique thinking, using OCR-2-32B on LiveCodeBench-Python.

#### B Evaluating Critique LLM Accuracy in Judging Code Solutions

This study initially assessed LLMs’ self-critique ability by having them evaluate their own generated solutions to select the best output in a parallel scaling setup. Recognizing that this might not reflect their true critique capabilities due to varying generation accuracy, we further evaluated them as external critics. To do this, we used the CodeContests benchmark (Li et al., 2022), randomly selecting one correct and one incorrect human-written solution for each of its 165 test questions. We limited code solutions to a maximum of 4096 tokens (based on the Qwen2.5 tokenizer), resulting in 238 Python and 329 C++ samples. The critique performance results when calculated on single positivenegative sample per problem are shown in Table 6. Notably, while OCR-2-32B excelled in critiquing Python code, OCR-2-14B surprisingly achieved the best performance for C++.

Model # Language # Accuracy # Language # Accuracy

DeepCoder-14B-Preview Python 13.4 C++ 28.6 OpenThinker2-32B Python 43.7 C++ 45.9 OlympicCoder-32B Python 46.6 C++ 49.5 QwQ-32B Python 60.1 C++ 56.8 Qwen3-32B Python 63.9 C++ 65.0

OCR-2-7B Python 60.1 C++ 54.7 OCR-2-14B Python 63.9 C++ 65.3 OCR-2-32B Python 66.0 C++ 65.0

Table 6: Critique accuracy of reasoning-enabled LLMs on human-written solutions provided in the test split of Code-Contests benchmark (Li et al., 2022).

Prompt for solution generation in Python

system: "" user: |-

You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.

Please use python programming language only.

You must use ```python for just the final solution code block with the following format: ```python # Your code here ``` {input}

Prompt for solution generation in C++

system: "" user: |-

You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.

Please use c++ programming language only.

You must use ```cpp for just the final solution code block with the following format: ```cpp # Your code here ``` {input}

Figure 6: Prompt template used for solution generation using R1 for OPENCODEREASONING-II.

Prompt for critique generation using QwQ-32B

system: "" user: |-

You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.

You have solved a programming problem. Now, you will critique your solution and conclude with <judgment>right/wrong</judgment>.

## Question {question}

## Solution {solution}

Figure 7: Prompt template used for critique data generation for OPENCODEREASONING-II.

