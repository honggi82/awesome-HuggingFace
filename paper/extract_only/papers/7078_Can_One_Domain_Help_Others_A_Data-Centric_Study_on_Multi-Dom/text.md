# arXiv:2507.17512v1[cs.AI]23Jul2025

[Figure 1]

### Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning

Yu Li1,2†, Zhuoshi Pan1,2†, Honglin Lin1,2†, Mengyuan Sun1,2, Conghui He1,2, Lijun Wu1,2∗ 1OpenDataLab, 2Shanghai Artificial Intelligence Laboratory

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs). Existing research has predominantly concentrated on isolated reasoning domains—such as mathematical problemsolving, coding tasks, or logical reasoning. However, real-world reasoning scenarios inherently demand an integrated application of multiple cognitive skills. Despite this, the interplay among these reasoning skills under reinforcement learning (RL) remains poorly understood. To bridge this gap, we present a systematic investigation of multi-domain reasoning within the RLVR framework, explicitly focusing on three primary domains: mathematical reasoning, code generation, and logical puzzle solving. We conduct a comprehensive study comprising four key components: (1) Leveraging the GRPO algorithm and the Qwen-2.5-7B model family, our study thoroughly evaluates the models’ in-domain improvements and cross-domain generalization capabilities when trained on single-domain datasets. (2) Additionally, we examine the intricate interactions—including mutual enhancements and conflicts—that emerge during combined cross-domain training. (3) To further understand the influence of supervised fine-tuning (SFT) on RL, we also analyze and compare performance differences between base and instruct models under identical RL configurations. (4) Furthermore, we delve into critical RL training details, systematically exploring the impacts of curriculum learning strategies, variations in reward design, and language-specific factors (e.g., Chinese vs. English datasets). Through extensive experiments, our results offer significant insights into the dynamics governing domain interactions, revealing key factors influencing both specialized and generalizable reasoning performance. These findings provide valuable guidance for optimizing RL methodologies to foster comprehensive, multi-domain reasoning capabilities in LLMs.

Date: July 24, 2025 Correspondence: Lijun Wu, wulijun@pjlab.org.cn Code: https://github.com/Leey21/A-Data-Centric-Study Equal contribution: Yu Li, Zhuoshi Pan, Honglin Lin

### 1 Introduction

Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) [42, 19, 35, 20, 44], exemplified by DeepSeek-R1-Zero [8], have demonstrated that Reinforcement Learning (RL) can substantially enhance the reasoning capabilities of Large Language Models (LLMs) even without relying on supervised fine-tuning (SFT) [46, 6, 3]. This approach has revealed emergent reasoning capacities, notably through length-dependent performance improvements. Later, multiple studies building upon this framework have validated the effectiveness of RLVR across specialized reasoning domains. For instance, Logic-RL [38] has significantly advanced deductive reasoning, while OpenReasoner-Zero [11] has set new performance benchmarks in mathematical reasoning tasks via RLdriven methods. These successes highlight the broad versatility and effectiveness of RLVR as a post-training framework to enhance reasoning skills across diverse domains.

Despite these breakthroughs, existing research has largely concentrated on reasoning tasks within isolated domains, such as mathematical problem-solving [45], code generation [18], or logical reasoning [22] tasks individually. In practice, however, comprehensive reasoning [16, 24] often demands the seamless integration of multiple cognitive skills. Crucially, the interactions among these reasoning skills under RLVR—particularly regarding how domain-specific training influences cross-domain generalization, training dynamics, reward structures, curriculum strategies, and training languages—have remained underexplored. A comprehensive, systematic investigation of these multi-domain interactions is thus essential to understand and optimize RLVR for holistic reasoning applications.

In this paper, we conduct a systematic study of multi-domain reasoning under the RLVR paradigm, explicitly focusing on three critical reasoning domains: Math, Code, and Puzzle. Leveraging the Group Relative Policy Optimization (GRPO) [30] algorithm and the Qwen-2.5 [27] model family, (1) we first examine the impacts of single-domain training on in-domain performance and cross-domain generalization. (2) Then, we identify complex interactions, including mutual enhancements and conflicts, that emerge when integrating multiple domains during training. (3) To further elucidate the role of supervised fine-tuning in enhancing RL effectiveness, we systematically analyze performance differences between base and instruct models. (4) Moreover, our analysis explores critical training strategies, such as curriculum learning, variations in reward design, and language-specific effects (e.g., Chinese versus English training datasets).

Through rigorous experimental evaluation, we uncover nuanced insights into domain interactions, revealing fundamental mechanisms that influence both domain-specific expertise and generalized reasoning capabilities. Our contributions provide valuable guidelines for future research aiming to refine RL methodologies, ultimately fostering more robust and integrated multi-domain reasoning in LLMs. The primary findings of this study are summarized as follows:

#### Overall Takeaways

- • Puzzle and math data provide mutual support. Logical reasoning and mathematical capabilities complement each other and enhance overall model performance.
- • Code reasoning has mixed cross-domain effects. It strengthens reasoning transfer for the instruct model but may constrain the base model’s reasoning capacity.
- • Cross-domain data leads to more robust performance. Combining diverse data often results in stronger or more balanced model capabilities, but requires more sophisticated design to address conflicts that may arise between different domains.
- • SFT boosts the effectiveness of RL. Incorporating an SFT stage before RL leads to substantial improvements in model performance.
- • Template consistency is critical. Misalignment between training and evaluation templates can significantly degrade performance, which also indicates that the robustness of RLVR’s generalization ability is challenged when trained on specific domains.
- • Policy refresh Benefits. Periodic updates to the reference model and optimizer state in curriculum learning can somewhat improve model stability and performance.
- • Reward design should adapt to difficulty. Tailoring reward settings to how the model performs on the training data can improve learning efficiency.
- • RLVR is language-sensitive. Models trained in Chinese underperforms that trained in English with a consistent performance gap.

Besides the above overall takeaways, more detailed and throughful observations are illustrated in specific sections in the following studies.

### 2 Experimental Configuration

This study aims to explore the model’s fine-grained reasoning capabilities from a data-centric perspective, through various training approaches including single-domain data training, cross-domain data combination, curriculum learning, different reward settings, and training languages.

#### 2.1 Multi-Domain Training Setup

We categorize our reasoning domains into Math, Code, and Puzzle. To support multi-domain training, we curate domain-specific datasets for these areas, as detailed in Table 1. (1) For the Math domain, we select the popular DeepScaleR (DSR) [21] and CountDown (CD) [22]. (2) In the Code domain, our experimental data consists of CodeR1-12k [18], which includes 2K reliable LeetCode [36] data and 10K verified data filtered from 26K TACO [15] data. (3) For the Puzzle domain, we focus on two main categories: Knights-and-Knaves (KK) [37] and Logic Puzzle Baron (LPB) [2]. Since the LPB dataset lacks ground-truth answers, we utilize DeepSeek-R1 [8] to annotate 2.4K easy-level puzzles, treating these annotations as pseudo ground truth answer for our RL training. For consistency across domains, we randomly sample larger datasets like DSR (40.3k) and CD (490k) to 10k samples, equalizing the data scale for subsequent training.

Table 1: Datasets for multi-domain training.

Domain Training Dataset Data Size Reward Scheme Math

DeepScaleR (DSR) [21] 10k Binary 0-1 CountDown (CD) [22] 10k Binary 0-1

Code CodeR1-12k [18] 12k Binary 0-1 Puzzle

Knights-and-Knaves (KK) [37] 5.4k Binary 0-1 Logic Puzzle Baron (LPB) [2] 2.4k Proportional 0-1

For the reward, we design task-specific schemes based on careful analysis of each dataset and the model’s initial performance. The LPB dataset stands out for its higher difficulty: models often fail to produce correct answers in a single attempt at the start of training. To address this, LPB uses a proportional 0–1 reward based on the fraction of correctly predicted cells, while all other datasets adopt a simpler binary 0–1 reward based solely on final answer correctness, without additional format checks. More details on reward design are provided in Section 7.

For model selection, we adopt the Qwen2.5-7B-Base and Qwen2.5-7B-Instruct models as starting points for training. Notably, as emphasized in [20], training templates play a crucial role in both the training and testing phases. Our experiments demonstrate that template consistency is crucial; hence, we standardize the use of the R1-template (Table 2) [8] during training. In testing, we also adopt the R1-template to ensure consistency between training and testing phases. We observe that mismatched templates severely degrade model performance, with a detailed analysis provided in Section 5.

Table 2: Template for DeepSeek-R1-Zero. prompt will be replaced with the specific reasoning question.

R1-template: A conversation between the User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process internally and then provides the User with the answer. The reasoning process and the answer are enclosed within <think> </think> and <answer>

</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: prompt. Assistant:

For the optimization algorithm, we adopt Group Relative Policy Optimization (GRPO) [30] as the core RL algorithm. Compared with Proximal Policy Optimization (PPO) [29], GRPO dispenses with

the traditional value model. Instead, it evaluates the advantage of different responses by assessing the quality differences among answers within a rollout group. GRPO formally maximizes the following objective:

LGRPO(θ) = Eτ∼πθ min rθ(τ)A(τ), clip(rθ(τ),1 − ϵ,1 + ϵ)A(τ) , (1)

where τ denotes the response sampled from the current policy πθ, and rθ(τ) = ππθ(τ)

old(τ) represents the probability ratio between the current policy and the previous policy before each actor update. Unlike PPO, the advantage in GRPO does not depend on a critic model. Instead, it estimates the advantage by calculating a baseline directly from the rollout group’s scores {Ri}i∈G(τ):

Rτ − mean({Ri}i∈G(τ)) std({Ri}i∈G(τ))

. (2)

A(τ) =

For Experimental Framework, all experiments are conducted using the veRL framework [31]. Both training and testing are conducted on a cluster equipped with 8 × A100 GPUs.

#### 2.2 Evaluation Settings

To ensure a comprehensive evaluation of model performance, we employ representative benchmarks across three key domains: mathematical reasoning, code generation, and logical problem-solving.

Math Domain: We evaluate in-domain mathematical reasoning using MATH500 [10], AIME241, and CountDown [22]. Notably, our implementation of MATH500 adopts a strict 0-shot evaluation, without providing any prior examples—this contrasts with many existing works where the number of shots is often unspecified. For CountDown, we follow the dataset split defined in TinyZero [22] and augment it with 24-game dataset (1.36k)2, which we also evaluate under a 0-shot setting.

Code Domain: We utilize HumanEval [4] and MBPP [1] to evaluate code generation proficiency. For MBPP, we employ 3-shot prompting, while HumanEval is evaluated in a 0-shot setting.

Puzzle Domain: We assess performance using test sets derived from KK (the dataset’s own test set) and ZebraLogicBench (Zebra) [17], which provide diverse scenarios for evaluating logical reasoning abilities. These benchmarks are evaluated in a 0-shot configuration.

Evaluation Details: All evaluations are performed using OpenCompass [7] toolkit, conducted with consistent hyperparameters: temperature = 0.7, top-p = 0.95, and a maximum output length of 8,192 tokens. We emphasize that many existing studies report inconsistent results when reproducing baseline models or conducting new evaluations, often due to misaligned templates or unspecified few-shot configurations. To promote reproducibility, we provide complete details of our prompt templates and few-shot examples in Appendix C, and encourage future work to maintain similar transparency in benchmark reporting.

### 3 Performance with Single-Domain Data

In this section, we evaluate the performance of models trained via RL using single-domain data. Our goal is to investigate the impact of single-domain training on both in-domain and out-of-domain (OOD) benchmark performance, providing insights into the models’ generalization capabilities. Additionally,

- 1https://huggingface.co/datasets/AI-MO/aimo-validation-aime
- 2https://huggingface.co/datasets/nlile/24-game

findings from single-domain training will guide the design of subsequent experiments involving combined-domain data. In all experimental results, Blue denotes positive improvements, while Orange indicates a decline relative to the baseline model.

For clarity in the subsequent analysis, we adopt the notation Base-DSR and Instruct-DSR to represent models trained on the DSR dataset using the Base model and the Instruct model, respectively. The same naming convention applies to other datasets.

#### 3.1 Math Domain

Next, we focus on the math domain, conducting all experiments under identical settings to ensure a fair comparison. Key training hyperparameters are detailed in Table 3. Given that mathematical tasks often require longer Chains of Thought (CoT) for reasoning [39, 34, 26, 23], we set a larger max token for training. For brevity, Batch Size is abbreviated as BS.

Table 3: Key hyperparameters for math domain.

Max Token Rollout BS Mini BS LR Rollout Times Epochs 8,192 256 128 1 × 10−6 8 12

As shown in Table 4, our results highlight two key findings:

- (1) RLVR enhances in-domain performance. Across all math domain experiments, RLVR consistently improves average model performance. For instance, the Base-DSR model increases MATH500 accuracy by 19.60 over the base model’s 56.40, while the Base-CD model boosts CountDown accuracy by 75.56, far surpassing the base model’s 1.05. Similar improvements are observed with instruct models. However, both the base and Base-DSR models perform poorly on the CountDown dataset, achieving only 1.05 and 0.04, respectively. Analysis of model outputs reveals that the base model struggles to meet the task requirement of using all numbers exactly once, highlighting its limited instruction-following capabilities without specialized training.
- (2) Math training improves puzzle performance but impairs coding skills. Math training improves puzzle-solving performance, with Base-DSR and Base-CD models increasing puzzle averages to 24.08 and 21.13, respectively, from 9.07 (base). However, coding performance declines significantly; for instance, the Base-CD model’s code performance drops to 29.59 from 67.46. This suggests that while math training enhances puzzle-solving capabilities, it may hinder coding skills due to differing reasoning requirements.

Table 4: Model performance (%) after training in the math domain.

Math Code Puzzle

Data

MATH500 CountDown AIME24 Avg. HumanEval MBPP Avg. KK Zebra Avg. Base 56.40 1.05 10.00 22.48 70.12 64.80 67.46 17.86 0.27 9.07

DSR 76.00 0.04 13.33 29.79 82.00 39.20 60.60 26.71 21.46 24.08 CD 67.20 76.61 13.33 52.38 23.78 35.40 29.59 21.40 21.11 21.13 DSR&CD 72.00 53.77 16.67 47.48 66.46 62.00 64.23 26.43 18.42 22.42

Instruct 69.00 24.35 13.33 35.56 82.93 62.80 72.87 10.14 31.50 20.82 DSR 72.60 35.33 10.00 39.31 78.05 53.60 65.82 25.00 29.66 27.33 CD 72.40 66.89 20.00 53.10 79.88 61.20 70.54 24.29 29.36 26.82 DSR&CD 74.60 64.79 13.33 50.91 80.49 54.40 67.44 26.86 24.64 25.75

#### Takeaway for Math RL

- • Math training boosts mathematical performance. Math-focused training significantly enhances model performance on mathematical tasks.
- • Math skills aid puzzles but hinder coding. Math training improves puzzle-solving abilities through shared logical reasoning but often reduces coding performance.

#### 3.2 Code Domain

In the code domain, RL typically focuses on generating executable code from user-provided instructions and verifying correctness using predefined test cases. This verification requires a secure sandbox environment to safely run generated code and enforce strict execution time limits to prevent timeouts. The key hyperparameters and sandbox configurations used in our experiments are summarized in Table 5. We highlight several key observations below:

Table 5: Training hyperparameters for code domain.

Max Token Rollout BS Mini BS LR Rollout Times Epochs Sandbox Timeout(s)

4,096 128 64 1 × 10−6 5 15 FireJail 30

- (1) Improvement in in-domain performance. The in-domain performance after RL training on code data is presented in Figures 1 and 2. Both the base and instruct models exhibit substantial improvements on Humaneval and MBPP, demonstrating the efficacy of code data for RL training. The base model, in particular, reveals significant untapped potential, with its Humaneval score surging from 70.12 to 80.49 (+10.37). On MBPP, the base model improves from 64.80 to 67.40. Despite these gains, the instruct model, enhanced by SFT, consistently achieves the highest performance. On Humaneval, the instruct model reaches a superior score of 84.15, improving by 1.83. On MBPP, it overcomes an initial deficit (62.80 vs. 64.80) to attain 68.40, surpassing the base model’s 67.40. This consistent outperformance across both benchmarks underscores the critical importance of SFT in unlocking the full potential of RL training, enabling the instruct model to outperform the base model despite the latter’s remarkable progress.

Initial 50 100 150 200 240

Training Steps

70

75

80

85

Accuracy(%)

Base-CodeR1

Instruct-CodeR1

Figure 1: Performance on HumanEval.

Initial 50 100 150 200 240

Training Steps

62

64

66

68

70

Accuracy(%)

Base-CodeR1

Instruct-CodeR1

Figure 2: Performance on MBPP.

- (2) Distinct cross-domain effects of code reasoning. To examine how enhanced code reasoning influences performance in other domains, we report OOD results in Table 6, using checkpoints after

240 training steps. The results reveal contrasting cross-domain effects between the base and instruct models. For the instruct model, improved code reasoning generally brings gains across most OOD benchmarks (except for CountDown). In contrast, the base model shows performance drops on most OOD tasks, except for Zebra. Further analysis of Base-CodeR1 outputs suggests that the rigid structure of code training data can constrain the base model’s output flexibility, leading to format inconsistencies that hinder correct answer extraction in non-code tasks.

Table 6: Model performance (%) after training in the code domain.

Math Puzzle

Data

MATH500 CountDown AIME24 KK Zebra Base 56.40 1.05 10.00 17.86 0.27 CodeR1 50.80 0.04 6.67 13.85 31.24 Instruct 69.00 24.35 13.33 10.14 31.50 CodeR1 72.00 22.59 16.67 17.57 32.14

#### Takeaway for Code RL

- • Coding ability enhancement. Code RL effectively improves the model’s ability to handle coding tasks.
- • Code reasoning has mixed cross-domain effects. It strengthens reasoning transfer for the instruct model but may constrain the base model’s reasoning capacity.

#### 3.3 Puzzle Domain

Logic reasoning puzzles require complex logical deduction and multi-step reasoning, posing unique challenges for RL models due to their need for sequential decision-making and pattern recognition. We evaluate RL performance in the puzzle domain using base and instruct models under identical training conditions for fair comparison. Key training hyperparameters are summarized in Table 7.

Table 7: Key hyperparameters for puzzle domain.

Max Token Rollout BS Mini BS LR Rollout Times Epochs 4,096 128 64 1 × 10−6 5 25

The results in Table 8 reveal the following key findings:

- (1) Puzzle task performance is substantially enhanced. Training on puzzle-specific datasets—namely KK and LPB—significantly boosts performance within the puzzle domain. Exclusive KK training achieves outstanding KK accuracy (94.29 for the base model, 99.14 for the instruct model), while LPB training notably raises Zebra scores (34.60 for the base model, 36.20 for the instruct model). Combining KK and LPB produces more balanced but slightly lower peak performance, with average puzzle accuracies of 61.98 (base) and 59.96 (instruct), indicating that mixed-dataset training offers limited gains beyond single-source specialization.
- (2) Cross-Domain Generalization from Puzzle Reasoning to Math Tasks. Training on puzzle datasets often enhances the mathematical reasoning ability of models, demonstrating effective transfer of logical skills across domains. For example, training on the KK dataset boosts the base model’s scores to 68.40 on MATH500 and 20.00 on AIME24, approaching the instruct model’s original scores of 69.00 and 13.33, respectively, indicating strong cross-domain generalization. In contrast, the Instruct-LPB

- model exhibits a sharp performance drop on Countdown, falling from 24.35 to 2.47. This decline may stem from LPB’s relatively fixed problem format, which imposes significant constraints on the problem-solving process for Countdown. Tracking this drop, as shown in Table 21, reveals a trend of initial improvement followed by a decline, suggesting that fixed data formats during training can lead to overfitting, thereby limiting the model’s out-of-domain performance.
- (3) Limited code domain impact. Puzzle training has an inconsistent effect on coding performance. Training on individual datasets often leads to reduced coding scores, but combining KK&LPB helps mitigate this decline, yielding Code averages of 71.35 (base) and 71.25 (instruct). This is likely due to the mismatch between the fixed format of puzzle data and the requirements of coding tasks. However, the increased data diversity from combining both datasets helps reduce the performance drop seen with single datasets.

Table 8: Model performance (%) after training in the puzzle domain.

Data

Math Code Puzzle

MATH500 CountDown AIME24 Avg. HumanEval MBPP Avg. KK Zebra Avg. Base 56.40 1.05 10.00 22.48 70.12 64.80 67.46 17.86 0.27 9.07

KK 68.40 19.36 20.00 35.92 60.37 51.80 56.09 94.29 30.69 62.49 LPB 69.00 7.40 10.00 28.80 74.40 61.60 68.00 16.60 34.60 25.60 KK&LPB 67.60 10.81 10.00 29.47 78.70 64.00 71.35 89.29 34.66 61.98

Instruct 69.00 24.35 13.33 35.56 82.93 62.80 72.87 10.14 31.50 20.82 KK 73.20 33.95 23.33 43.49 74.39 62.80 62.60 99.14 17.91 58.53 LPB 69.40 2.47 13.33 28.41 70.70 63.80 67.25 14.00 36.20 25.10 KK&LPB 72.40 30.30 23.33 42.01 80.49 62.00 71.25 83.29 36.62 59.96

Takeaway for Puzzle RL

• Puzzle tasks enhance logical reasoning for math tasks. Puzzle tasks improve logical reasoning, leading to better performance on mathematical tasks. However, this effect does not extend to coding tasks.

- 4 Performance with Combined-Domain Data

In this section, we reorganize the experimental results of cross-domain RL and provide a systematic analysis of performance across different domain combinations and their interaction patterns. To facilitate a clear examination of how various combinations influence the model’s generalization capability and domain-specific task performance, we divide the analysis into two subsections: dualdomain combinations and triple-domain combinations.

Given that many prior studies [11, 41, 45, 22, 38] primarily focus on base models, we employ the Qwen2.5-7B Base as the foundation for continued training. For experimental hyperparameters, whenever mathematical data is included, we adopt the same configurations as those used for math tasks (Table 3), in order to accommodate the higher token requirements inherent to math-specific training.

#### 4.1 Combinations of Dual Domains

We first examine the model’s performance under pairwise domain combinations, specifically Math + Puzzle, Puzzle + Code, and Math + Code. The results for these configurations are directly compared against their respective single-domain baselines. From the domain perspective, the outcomes are summarized in Table 9.

Table 9: Performance (%) of the RL model with ∆ compared to Base.

Math Avg Code Avg Puzzle Avg All Avg

Training Data

Value ∆ Value ∆ Value ∆ Value ∆ Base 22.48 – 67.46 – 9.07 – 31.50 –

Math 47.48 +25.00 64.23 -3.23 22.42 +13.35 45.11 +13.61 Puzzle 29.47 +6.99 71.35 +3.89 61.98 +52.91 50.72 +19.22 Code 19.17 -3.31 73.95 +6.49 22.55 +13.48 35.78 +4.28

Math + Puzzle 49.72 +27.24 44.90 -22.56 49.78 +40.71 48.36 +16.86 Puzzle + Code 32.06 +9.58 74.88 +7.42 55.15 +46.08 50.89 +19.39 Math + Code 47.22 +24.74 75.06 +7.60 25.34 +16.27 48.92 +17.42

Key observations are summarized as follows:

- (1) Joint training with specific domain pairs can lead to clear synergistic benefits. For example, when training with the combination of Math + Puzzle, the model’s performance on Math improves to 49.72, surpassing the Math-only performance of 47.48. Similarly, for Code tasks, both additional Puzzle and Math data lead to improvements in code-related tasks when compared to Code-only training. These findings indicate that joint training can facilitate beneficial transfer of knowledge across domains in certain settings.
- (2) Adding an extra domain does not always lead to better performance and may introduce new generalization challenges. For the Puzzle task, all configurations involving additional domains perform worse than the Puzzle-only setting, suggesting that increased data diversity can hinder the model’s ability to specialize in solving puzzles. This also reflects the high degree of specialization required by the Puzzle task. Notably, in the Math + Puzzle configuration, the model’s performance on Code tasks drops significantly, falling below both the Math-only and Puzzle-only baselines. This may be due to the unique characteristics of the Code task, which differs structurally and linguistically from Math and Puzzle, making generalization more difficult when training data is dominated by other domains. Among all combinations, only Puzzle + Code achieves overall strong performance, with an overall improvement of 19.39. These results highlight that incorporating more domains does not guarantee universal improvements and may sometimes impede the model’s ability to adapt to tasks with distinct forms or representations.

In summary, these results show that dual-domain training can yield non-trivial benefits and better task balance in specific settings, but its effectiveness depends on how domains are combined and how training data is allocated. Careful design choices are necessary to leverage synergy and mitigate potential negative interactions.

#### 4.2 Combinations of Triple Domains

Furthermore, we investigate the impact of training with data from all domains (Math + Code + Puzzle), comparing this setting to the previous optimal configuration (Puzzle + Code) in Figure 3. Additionally, Figure 4 illustrates the trends in overall performance as domain combinations are progressively expanded. The following key takeaways emerge:

- (1) Combining data from all three domains further enhances overall performance, though negative transfer occurs on specific tasks. As shown in Figure 3, joint training on data from the Math, Code, and Puzzle domains yields an overall average performance of 56.57, surpassing the previous best configuration (Puzzle + Code, 50.89). While overall accuracy improves and Math tasks reach their highest performance (49.75), the performance on Puzzle tasks drops to 49.73, notably lower than the Puzzle + Code setting (55.15). This supports our earlier observation that the Puzzle domain demands a

56.57

Overall

50.89

49.73

Puzzle

55.15

73.63

Code

74.88

49.75

Math

Puzzle + Code Math + Code + Puzzle

32.06

| |
|---|

10 20 30 40 50 60 70 80 Accuracy

Figure 3: Performance comparison of tripledomain and optimal dual-domain data.

60

| |56.57| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| |50.72<br><br>48.92<br><br>50.89| | | | | | | | |
| |45.11<br><br>48.36| | | | | | | | |
| | | | | | | | | | |
| |35.78| | | | | | | | |
| |31.50<br><br>Base Single Domain<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Dual Domain Triple Domain<br><br>| |
|---|
<br><br>Trend Line| | | | | | | | |
| | | | | | | | | | |

55

OverallAccuracy

50

45

40

35

30

Base Code Math Puzzle M+P M+C P+C M+P+C Domain Combination

Figure 4: Overall model performance under different data combinations.

high degree of specialization, and that incorporating out-of-domain data can negatively impact its results. These results indicate that while broader domain coverage improves overall performance and generalization, the inclusion of Math data fails to yield positive effects on Puzzle tasks and instead leads to performance degradation. This outcome likely stems from the increased complexity introduced by the additional domain, which adversely impacts specific tasks.

- (2) Enhanced data diversity contributes to further model performance improvements. As shown in Figure 4, excluding the outlier (50.72) observed with the Puzzle-only data, the model’s overall performance exhibits a positive trend as domain combinations increase. However, the unusually high value in the Puzzle task is primarily due to the model’s exceptional performance on the KK task (94.29), which disproportionately elevates the overall score. This outlier does not necessarily reflect balanced performance improvements across all tasks, but rather is driven by a particularly strong result in one specific sub-task.
- (3) The triple-domain combination improves performance balance across tasks compared to certain dual-domain combinations. Unlike some dual-domain configurations (e.g., Math + Puzzle), which experience significant performance degradation on Code tasks (–22.56), the triple-domain approach maintains more balanced performance across all tasks. While some task-specific specialization may be slightly reduced, the inclusion of Code data ensures that performance on the Code task remains strong and consistent. These results indicate that expanding domain coverage can mitigate the performance collapse on specific tasks, achieving a more stable and generalized performance across tasks.

In summary, the triple-domain combination exemplifies the balanced advantages of multi-skill training. Although incorporating additional domains may lead to a modest reduction in peak performance for individual tasks, this approach achieves the highest overall performance while maintaining competitive results across all subtasks. These findings underscore the substantial potential of carefully designed multi-domain training strategies in building versatile models with broad adaptability.

#### Takeaway for Dual-Domain and Triple-Domain Combinations

- • Multi-domain training improves overall performance. Combining multiple domains generally leads to better overall performance, with the triple-domain combination showing moderate gains.
- • Multi-domain training improves task balance and overall stability. By providing broader coverage, multi-domain setups help maintain consistent performance across tasks, preventing extreme drops in any single area and promoting more robust, generalized models.

### 5 Evaluating Template Variations in Reinforcement Learning

A commonly overlooked issue in RL is the mismatch between templates used for training and those applied during testing [9, 32, 13]. Such discrepancies can significantly degrade model performance during evaluation. For instance, training data may utilize an R1-style template, while testing might employ a Qwen template (the default for many Qwen-series models [40]) or no template at all, as some evaluation tools, such as OpenCompass [7], default to a blank template for base model testing (see Table 10). In this section, we train base and instruct models on the KK dataset using the R1-style template and evaluate their performance with various templates. The results, presented in Table 11, assess the impact of mismatched training and testing templates, highlighting the models’ sensitivity to template alignment.

Qwen Template: <|im start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im end|>\n<|im start |>user\n{question}<|im end|>\n<|im start|> assistant\n Base Template: {question}

Table 10: Template for qwen and base model. question will be replaced with the specific reasoning question during training.

Table 11: Model performance under different templates.

Math Code Puzzle

Model Template

MATH500 CountDown AIME24 HumanEval MBPP KK Zebra

Base Template 72.80 0.00 6.67 60.98 3.00 31.29 16.18 Qwen Template 69.20 20.79 13.33 63.78 64.40 94.00 0.56 R1 Template 68.40 19.36 10.00 60.37 51.80 94.29 30.69

Base

Base Template 3.20 0.63 3.33 51.22 1.60 41.57 21.22 Qwen Template 1.80 0.29 0.00 32.93 42.60 62.71 9.84 R1 Template 73.20 33.95 23.33 74.39 62.80 99.14 17.91

Instruct

As shown in Table 11, a mismatch in templates significantly impacts the model’s performance. We can break this down into the following points:

60

54.56

47.84

50

46.58

No Template Qwen Template R1 Template

- (1) Mismatched Templates Significantly Impact Model Performance. Mismatched templates substantially reduce the performance of base and instruct models across diverse tasks. For the base model, a mismatched base template lowers scores to 0, 3.00, and 31.29 on Countdown, MBPP, and KK, respectively, while a mismatched instruct template decreases the Zebra score to 0.56. Similarly, the instruct model experiences performance drops to 1.80 on MATH500 and 0.29 on Countdown with a mismatched template like Qwen, among other tasks. These declines highlight the models’ sensitivity to template mismatches.
- (2) Matched Templates Typically Achieve Optimal Model Performance. Matched templates consistently enhance the average performance of base and instruct models across benchmarks. As shown in Figure 5, the R1 template produces scores of 47.84 and 54.56 for the base and instruct models, respectively, surpassing mismatched conditions. Although no single template excels in every task, the superior performance of matched templates underscores their critical role in ensuring stable and effective outcomes, especially for intricate datasets like KK and Zebra.

| |
|---|

| |
|---|

40

AvgPerformance

27.27

30

21.45

17.54

20

10

0

Base Instruct

Figure 5: The average test performance of base and instruct models on different templates.

#### Takeaway for Template

• Template consistency is critical. Mismatched templates degrade model performance on

certain tasks, highlighting the current lack of robustness in RLVR.

### 6 The Role of Curriculum Learning in Reinforcement Learning

While curriculum learning is well-established in SFT [5, 14, 12], its application in RLVR remains insufficiently explored [25, 33]. To address this gap, we systematically investigate curriculum learning strategies in the Puzzle domain, leveraging the KK dataset. This dataset features clearly defined difficulty variations based on the number of sub-questions per problem, enabling us to effectively categorize data by difficulty levels. This facilitates a focused evaluation of the generalizability of our approach within a specific cognitive challenge.

Difficulty stratification and curriculum design: Effective curriculum learning hinges on accurately quantifying task difficulty. For the puzzle task, difficulty is defined by the number of sub-questions per problem (PPL), ranging from 3PPL to 8PPL across six levels. Training progresses sequentially from easier to harder tasks, enabling the model to build proficiency in simpler reasoning before tackling more complex problems.

We propose a novel policy refresh strategy, where after each training stage (175 steps), corresponding to a specific difficulty level, the reference model is updated by replacing it with the latest actor model. Additionally, the optimizer state is reset to prevent overfitting to prior difficulty levels. This strategy provides the model with a fresh starting point at each new difficulty level, facilitating stable learning and better adaptation to progressively challenging tasks. The corresponding experimental results are presented in Figure 6. The analysis is as follows:

Standard Curriculum Learning

Policy Refresh Curriculum Learning

3PPL4PPL5PPL6PPL7PPL8PPL

3PPL4PPL5PPL6PPL7PPL8PPL

- 96 90 93 77 66 49 20 70.14
- 97 95 98 82 70 55 45 77.43
- 98 98 96 86 79 65 62 83.43

100 100 99 98 93 92 75 93.86

- 99 99 100 97 95 91 86 95.29
- 100 100 99 99 96 94 93 97.29

- 98 96 90 70 60 51 31 70.86
- 99 100 98 89 86 73 55 85.71
- 100 100 100 100 96 94 85 96.43

- 100 100 100 99 100 93 90 97.43
- 100 100 100 100 100 99 99 99.71

100 100 100 100 100 99 99 99.71

2PPL 3PPL 4PPL 5PPL 6PPL 7PPL 8PPL Average

2PPL 3PPL 4PPL 5PPL 6PPL 7PPL 8PPL Average

|[Figure 2]| | | | | |
|---|---|---|---|---|---|
| | | | | | |

40 50 60 70 80 90 100

Figure 6: Model performance on the KK dataset with different curriculum settings. The x-axis represents the KK difficulty levels, and the y-axis shows the training data sequence from 3PPL to 8PPL.

- (1) Curriculum learning improves the upper bound of model performance. Figure 6 demonstrates that curriculum learning under both settings effectively enhances the model’s upper performance bound, achieving accuracies of 97.29 and 99.71, respectively, which significantly surpass the 94.29 accuracy obtained under mixed training. This improvement highlights the key advantages of curriculum learning, including more structured and progressive learning patterns that enable the model to better capture complex task dependencies and enhance generalization capabilities.

- (2) Policy refresh further improves the performance and convergence rate of curriculum learning. The right panel of Figure 6 reveals that policy refresh accelerates convergence while delivering superior final performance. Beginning from the second stage, models incorporating policy refresh consistently outperform standard curriculum learning, achieving 97.43 accuracy at 6PPL—already exceeding the latter’s final score of 97.29. Remarkably, the policy refresh model nearly reaches perfect accuracy, even surpassing the final result of the instruct model under mixed training (99.14).

#### Takeaway for Curriculum Learning

• Curriculum learning demonstrates effectiveness and achieves additional improvements through policy refresh. Staged training raises the model’s performance upper bound, while periodic reference model updates further accelerate convergence and enhance final results.

### 7 Impact of Reward Styles on Model Performance

In this section, we investigate how different reward styles affect model performance using the KK and LPB datasets, selected for their complex problem structures involving multiple interdependent entities. Unlike typical Math and Code datasets, which often feature problems with a single correct answer and rely predominantly on binary reward schemes, the KK and LPB present unique challenges. Each problem requires filling multiple blanks, resembling a cloze task, enabling evaluation of diverse reward strategies. We compare two primary schemes: a binary reward (R1), granting credit only for fully correct responses, and a partial reward (R2), based on the fraction of correctly filled blanks. Additionally, we explore a format reward (R3) using the <think> tag to promote intermediate reasoning, and a rescaled reward (R4) that extends the reward range to [−1,1] to penalize incorrect responses.

- (1) Formally, for the KK dataset, we define the reward function R(response) as follows:

R(response) =

 



⌊Nc/N⌋ (Binary Reward, R1) Nc/N (Partial Reward, R2) ⌊Nc/N⌋ + format reward (Format Reward, R3)

1 if Nc = N, −1 otherwise (Rescaled Reward, R4),

where Nc denotes the number of blanks correctly completed by the model and N represents the total number of blanks in the puzzle.

For the LPB dataset, the reward functions differ slightly from those defined for the KK puzzle dataset. Specifically, the forms of the binary reward (R1) and the partial reward (R2) remain identical. However, two modifications are introduced: (1) the format reward (R3) is modified from ⌊Nc/N⌋ + format reward to Nc/N + format reward, and the rescaled reward (R4) is defined by linearly scaling the partial reward (R2) to a continuous range from −1 to 1, leading to 2 × (Nc/N − 0.5).

- (2) Finally, the reward function for the LPB dataset is defined as follows:

 

⌊Nc/N⌋ (Binary Reward, R1) Nc/N (Partial Reward, R2) Nc/N + format reward (Format Reward, R3)

R(response) =



2 × (Nc/N − 0.5) (Rescaled Reward, R4),

The rationale behind these adjustments is supported by empirical results on the LPB dataset, where the partial reward (R2)—which measures the proportion of correctly filled blanks—yields the best

performance, whereas the binary reward (R1) results in the poorest performance. Consequently, we convert the first component of the format reward (R3) from binary to partial, and adjust the rescaled reward (R4) into a continuous form to avoid using discrete rewards. To systematically evaluate the impact of different reward styles, we conducted a comprehensive analysis across both in-domain and out-of-domain settings.

#### 7.1 Impact of Reward Models on In-Domain Performance

To investigate the impact of different reward styles on model training, we conduct experiments where only the reward varies, training for a total of 800 steps (or fewer if collapse occurs). We compare performance across the KK and LPB, with results shown in Figures 7 and 8. These comparisons reveal that reward efficacy is highly dataset-dependent, as summarized in the following key findings:

- (1) Binary reward excels on KK but fails on LPB due to sparsity differences. On KK, the simplest R1 achieves the best final performance, outperforming more nuanced alternatives by providing clear, direct signals. This aligns with the “proportional reward trap” phenomenon [28] in programming tasks, where R2 introduce noise. In contrast, R1 leads to consistent training collapse on LPB, where extreme reward sparsity arises because the model rarely predicts all puzzle cells correctly, yielding few positive signals. We limit R1 training on LPB to 200 steps to avoid unnecessary computation. This observation underscores that binary rewards are well-suited for relatively easier tasks, like KK, where the base model can achieve complete success sometimes, but are untenable for harder ones like LPB.
- (2) Partial reward underperforms on KK but offers a viable baseline on LPB, though with limitations. On KK, R2 shows no advantage over R1 and ultimately degrades performance by injecting noisy learning signals. Conversely, on LPB, R2 emerges as a feasible alternative to R1’s collapse, delivering initial promise with a peak accuracy of 38.63 at 200 steps. However, its gains are not sustained, as it fails to accurately penalize the specific erroneous cells in the response, leading to a slight decline. This highlights R2’s utility in sparse settings but exposes its inadequacy compared to more sophisticated reward mechanism.

100 200 300 400 500 600 700 800

Training Steps

40

50

60

70

80

90

100

KKAveragePerformance(%)

- Reward 1

- Reward 2

- Reward 3

- Reward 4

Figure 7: Performance on KK.

100 200 300 400 500 570

Training Steps

10

15

20

25

30

35

40

ZebraLogicBenchPerformance(%)

- Reward 1

- Reward 2

- Reward 3

- Reward 4

Figure 8: Performance on LPB.

- (3) Format reward and rescaled reward excel on LPB, while falling short on KK. On KK, despite early gains from format correction or error suppression, both R3 and R4 yield inferior final performance to R1, indicating that their added complexity does not pay off in domains favoring binary signals. In contrast, on LPB, R3 and R4 initially trail R2 but eventually surpass it, benefiting from more informative signals: R3 stabilizes training via well-formed outputs, and R4 amplifies behavioral differences for better optimization.

Overall, these contrasts demonstrate that optimal reward design is not universal but critically tied to dataset characteristics like sparsity and task complexity. These factors must be carefully considered when designing RLVR training.

#### 7.2 Impact of Reward Models on OOD Generalization

Figures 9 and 10 illustrate the effects of various reward schemes on OOD tasks, highlighting how reward efficacy varies across datasets. Our key observations reveal some insightful findings:

- (1) In mathematical reasoning tasks, reward schemes yield different outcomes depending on the training data. For KK, all rewards produce similar performance, with the R3 offering no clear advantage over the base model. This contradicts prior claims [38] about the benefits of structured reasoning incentives. In contrast, for LPB, the choice of reward significantly impacts performance: R2 achieves the highest accuracies (e.g., outperforming others on both AIME24 and MATH500 benchmarks), while R1 leads to substantial declines (e.g., MATH500 accuracy dropping from 56.4 to 41.8).

Reward1 Reward2 Reward3 Reward4

51.59

57.64

63.69

69.73

75.78

81.83

Accuracy(%)

HumanEval

65.2

61.6

76.8

70.1 70.7

Reward1 Reward2 Reward3 Reward4

26.40

35.16

43.92

52.68

61.44

70.20

Accuracy(%)

MBPP

59.8

36.4

65.2

47.6

64.8

Reward1 Reward2 Reward3 Reward4

46.40

52.32

58.24

64.16

70.08

76.00

Accuracy(%)

MATH500

70.6 70.2

71.0

69.6

56.4

Reward1 Reward2 Reward3 Reward4

0.00

4.40

8.81

13.21

17.62

22.02

Accuracy(%)

CountDown

17.0

13.5 13.9

12.0

1.1

Reward1 Reward2 Reward3 Reward4

0

5

10

15

20

25

Accuracy(%)

AIME24

20.0

6.7 6.7

13.3

10.0

Figure 9: KK-impact of reward configurations (base model shown with dashed lines).

- (2) In code generation tasks, different datasets exhibit distinct reward sensitivities. For example, training on KK is relatively sensitive to reward design, with the performance of different rewards varying considerably. Most rewards generally degrade performance compared to the base model, though R3 effectively mitigates this negative impact and provides modest improvements. In contrast, training on LPB is less sensitive to reward design: most rewards (excluding R1) perform similarly, yielding gains on the HumanEval benchmark but experiencing drops on the MBPP benchmark. R1, however, suffers from significantly worse performance on both HumanEval and MBPP, which further aligns with the observed in-domain training collapse on LPB.
- (3) A significant limitation lies in current reward mechanisms. These comparisons expose a critical limitation in current reward mechanisms: they operate at the response level rather than the cell level. This means they fail to accurately penalize the erroneous predicted cells but treat all cells equally within a response. This issue is particularly evident in KK, where even R2 led to poor outcomes, aligning with challenges noted in [28]. Overall, the absence of a universal reward strategy emphasizes the task-dependent nature of reward design and the essential role of data diversity in RL training. Optimal rewards should be tailored to specific dataset characteristics—e.g., finer-grained, cell-level schemes for datasets like KK—to overcome the limitations of current response-level rewards.

###### HumanEval

###### MBPP

83.1

69.80

78.1 77.4

74.4

64.8

73.5

64.44

70.1

61.6 62.0 61.8

Accuracy(%)

Accuracy(%)

63.9

59.08

54.3

53.72

53.0

45.1

44.7

48.36

35.1

43.00

Reward1 Reward2 Reward3 Reward4

Reward1 Reward2 Reward3 Reward4

###### MATH500

###### CountDown

###### AIME24

74.00

14.40

15

69.0

65.56

11.52

12

10.0

10.0

9.4

Accuracy(%)

Accuracy(%)

Accuracy(%)

56.4

57.12

8.64

9

7.4

7.3

48.4

48.68

5.76

6

47.2

5.0

41.8

3.3

3.3

40.24

2.88

3

1.1

0.0

31.80

0.00

0

Reward1 Reward2 Reward3 Reward4

Reward1 Reward2 Reward3 Reward4

Reward1 Reward2 Reward3 Reward4

Figure 10: LPB-impact of reward configurations (base model shown with dashed lines).

#### Takeaway for Reward

- • Reward design should match task complexity. Binary rewards work well for simpler tasks, while partial rewards are more suitable for complex tasks.
- • Current partial rewards lack precision. Designing more fine-grained partial reward signals is a promising direction for further improvement.

### 8 Influence of Training Language

To investigate the impact of training data language [43], we translate the DeepScaleR dataset into Chinese using GPT-4.1-nano3 and performed RL training with the identical hyperparameters to those used for the English version.

To ensure the model uses Chinese for reasoning during training, we employ the “langid”4 package to detect the language of each rollout trajectory. A reward of 1 is given only when the language is Chinese and the final answer is correct. If the language is not Chinese, even if the final answer is correct, the reward is 0. This strict reward function is necessary, as we observe that without it, the model would default to reasoning in English even when the questions are translated into Chinese. Figure 11 illustrates the impact of different training languages. It is clear that models trained to reason in Chinese consistently perform worse than their English counterparts. This inefficiency in Chinese language reasoning highlights the need for more advanced post-training

80

71.6

70

58.4

60

Accuracy(%)

50

40

34.2

30

22.3

20.0

20

10.9

10

6.7

1.8

0

En Zh En Zh En Zh En Zh

Figure 11: The effect of different training language.

- 3https://openai.com/index/gpt-4-1/
- 4https://pypi.org/project/langid/

algorithms capable of improving cross-lingual generalization for complex reasoning tasks.

#### Takeaway for Training Language

• RLVR is language-sensitive. Models trained in Chinese underperforms that trained in

English with a consistent performance gap.

### 9 Conclusion

In this work, we focus on the reasoning capabilities of large language models, with a data-centric approach. We classify reasoning data into three main domains: Math, Code, and Puzzle. A detailed discussion is provided on the effects of domain-specific data in reinforcement learning and the generalization to out-of-domain data. Through cross-domain data combinations, we reveal the potential interactions between different model capabilities, including both assisting and conflicting phenomena. Additionally, we thoroughly discuss the impact of various factors on model performance in Reinforcement Learning, including model template, curriculum learning, reward styles, and training languages, offering some insights from multiple perspectives.

For future work, we believe it would be beneficial to further refine the categorization of reasoning capabilities. For example, introducing data from the science and general reasoning domains would allow for more detailed discussions on data combinations. Furthermore, due to hardware limitations, this paper primarily focuses on the base and instruct models of Qwen 2.5. Future research could expand on the impact of different datasets on models like Llama and DeepSeek. We also believe that RLVR will become a significant milestone in the development of large language models, with data remaining the cornerstone of any training process. We hope that future work will delve deeper into exploring the impact of data on RLVR.

### References

- [1] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021.
- [2] Oleg Bask. The logic puzzle baron dataset. https://huggingface.co/datasets/olegbask/LogicPuzzleBaron, 2024.
- [3] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025.
- [4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.
- [5] Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Pich´e, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025.
- [6] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.
- [7] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.
- [8] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [9] Jia He, Mukund Rungta, David Koleczek, Arshdeep Sekhon, Franklin X Wang, and Sadid Hasan. Does prompt formatting have any impact on llm performance?, 2024. URL https://arxiv.org/abs/2411.10541.
- [10] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.
- [11] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Openreasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.
- [12] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5901–5910, 2020.
- [13] Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. Chatbug: A common vulnerability of aligned llms induced by chat templates, 2025. URL https://arxiv.org/abs/2406.12935.
- [14] Yajing Kong, Liu Liu, Jun Wang, and Dacheng Tao. Adaptive curriculum learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5067–5076, 2021.
- [15] Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

- [16] Yu Li, Qizhi Pei, Mengyuan Sun, Honglin Lin, Chenlin Ming, Xin Gao, Jiang Wu, Conghui He, and Lijun Wu. Cipherbank: Exploring the boundary of llm reasoning capabilities through cryptography challenges. arXiv preprint arXiv:2504.19093, 2025.
- [17] Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. Zebralogic: On the scaling limits of llms for logical reasoning, 2025. URL https://arxiv.org/ abs/2502.01100.
- [18] Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards. 2025.
- [19] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.
- [20] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [21] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/ DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2,

2025. Notion Blog.

- [22] Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero. https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.
- [23] Zhuoshi Pan, Yu Li, Honglin Lin, Qizhi Pei, Zinan Tang, Wei Wu, Chenlin Ming, H Vicky Zhao, Conghui He, and Lijun Wu. Lemma: Learning from errors for mathematical advancement in llms. arXiv preprint arXiv:2503.17439, 2025.
- [24] Zhuoshi Pan, Qizhi Pei, Yu Li, Qiyao Sun, Zinan Tang, H. Vicky Zhao, Conghui He, and Lijun Wu. Rest: Stress testing large reasoning models by asking multiple problems at once, 2025. URL https://arxiv.org/ abs/2507.10541.
- [25] Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, et al. Curriculum reinforcement learning from easy to hard tasks improves llm reasoning. arXiv preprint arXiv:2506.06632, 2025.
- [26] Qizhi Pei, Lijun Wu, Zhuoshi Pan, Yu Li, Honglin Lin, Chenlin Ming, Xin Gao, Conghui He, and Rui Yan. Mathfusion: Enhancing mathematical problem-solving of llm through instruction fusion. arXiv preprint arXiv:2503.16212, 2025.
- [27] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- [28] Abhinav Rastogi, Albert Q Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, et al. Magistral. arXiv preprint arXiv:2506.10910, 2025.
- [29] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.
- [30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [31] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

- [32] Shijian Wang, Linxin Song, Jieyu Zhang, Ryotaro Shimizu, Ao Luo, Li Yao, Cunjian Chen, Julian McAuley, and Hanqian Wu. Template matters: Understanding the role of instruction templates in multimodal language model evaluation and training. In ICLR 2025 Workshop on Navigating and Addressing Data Problems for Foundation Models, 2025. URL https://openreview.net/forum?id=aDAaoRhYW4.
- [33] Zhenting Wang, Guofeng Cui, Yu-Jhe Li, Kun Wan, and Wentian Zhao. Dump: Automated distribution-level curriculum learning for rl-based llm post-training. arXiv preprint arXiv:2504.09710, 2025.
- [34] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460, 2025.
- [35] Jialong Wu, Shaofeng Yin, Ningya Feng, and Mingsheng Long. Rlvr-world: Training world models with reinforcement learning. arXiv preprint arXiv:2505.13934, 2025.
- [36] Yunhui Xia, Wei Shen, Yan Wang, Jason Klein Liu, Huifeng Sun, Siyue Wu, Jian Hu, and Xiaolong Xu. Leetcodedataset: A temporal dataset for robust evaluation and efficient training of code llms, 2025. URL https://arxiv.org/abs/2504.14655.
- [37] Chulin Xie, Yangsibo Huang, Chiyuan Zhang, Da Yu, Xinyun Chen, Bill Yuchen Lin, Bo Li, Badih Ghazi, and Ravi Kumar. On memorization of large language models in logical reasoning. 2024. URL https: //arxiv.org/abs/2410.23123.
- [38] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.
- [39] Haotian Xu, Xing Wu, Weinong Wang, Zhongzhi Li, Da Zheng, Boyuan Chen, Yi Hu, Shijia Kang, Jiaming Ji, Yingying Zhang, et al. Redstar: Does scaling long-cot data unlock better slow-reasoning systems? arXiv preprint arXiv:2501.11284, 2025.
- [40] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [41] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.
- [42] Tianyu Yu, Bo Ji, Shouli Wang, Shu Yao, Zefan Wang, Ganqu Cui, Lifan Yuan, Ning Ding, Yuan Yao, Zhiyuan Liu, et al. Rlpr: Extrapolating rlvr to general domains without verifiers. arXiv preprint arXiv:2506.18254, 2025.
- [43] Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, and Chao Zhang. Large language model as attributed training data generator: A tale of diversity and bias, 2023. URL https://arxiv.org/abs/2306.15895.
- [44] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.
- [45] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.
- [46] Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s” aha moment” in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.

## Appendix

### A Step-Level Performance Results for Each Training Dataset

In the following figures, we present detailed performance results for the base and instruct models across different training datasets. The results show performance at each evaluation checkpoint, illustrating the models’ progression and how different training data influence overall performance.

• DeepscaleR: Figures 12 and 13 illustrate the training dynamics on the DeepscaleR dataset. The results show that the model’s mathematical reasoning ability improves consistently and partially generalizes to logical reasoning tasks, while its code reasoning capability drops significantly.

###### Math

80

70

60

Accuracy(%)

50

Math500

40

CountDown

AIME24

30

20

10

0

0 100 200 300 400

###### Code

70

HumanEval

MBPP

60

50

40

30

0 100 200 300 400

###### Puzzle

25

20

15

10

5

KK

ZebraLogicBench

0

0 100 200 300 400

Figure 12: Base model’s detailed performance on DeepscaleR.

###### Math

70

60

Accuracy(%)

50

Math500

CountDown

40

AIME24

30

20

10

0 100 200 300 400

###### Code

80

75

70

65

60

HumanEval

MBPP

55

0 100 200 300 400

###### Puzzle

30

25

20

15

KK

ZebraLogicBench

10

0 100 200 300 400

Figure 13: Instruct model’s detailed performance on DeepscaleR.

• CountDown: Figures 14 and 15 present the training dynamics on the CountDown dataset. Although the model initially underperforms on CountDown, domain-specific training significantly improves its results. Meanwhile, CountDown training degrades the base model’s coding ability, whereas the instruct model’s coding performance remains relatively stable, indirectly demonstrating the positive impact of SFT on the robustness of RL training.

###### Math

80

70

60

Accuracy(%)

50

MATH500

40

CountDown

AIME24

30

20

10

0

0 25 50 75 100 125 150 175 200

###### Code

HumanEval

70

MBPP

60

50

40

30

0 25 50 75 100 125 150 175 200

###### Puzzle

25

20

15

10

5

KK

ZebraLogicBench

0

0 25 50 75 100 125 150 175 200

Figure 14: Base model’s detailed performance on CountDown.

70

30

80

60

Accuracy(%)

25

50

75

MATH500

HumanEval

CountDown

40

MBPP

20

AIME24

70

30

15

20

65

KK

ZebraLogicBench

10

10

60

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Figure 15: Instruct model’s detailed performance on CountDown.

• CodeR1: Figures 16 and 17 show the training dynamics on the CodeR1 dataset. While code data substantially improves the models’ coding reasoning ability, it also introduces negative effects on mathematical reasoning, with the impact being more pronounced for the base model.

###### Math

60

50

Accuracy(%)

40

MATH500

CountDown

30

AIME24

20

10

0

0 50 100 150 200 250

###### Code

80

78

76

74

HumanEval

MBPP

72

70

68

66

64

0 50 100 150 200 250

###### Puzzle

30

25

20

15

10

KK

5

ZebraLogicBench

0

0 50 100 150 200 250

Figure 16: Base model’s detailed performance on CodeR1.

###### Math

###### Code

###### Puzzle

85

70

30

80

60

Accuracy(%)

25

50

MATH500

75

KK

CountDown

40

ZebraLogicBench

20

AIME24

70

30

15

20

HumanEval

65

MBPP

10

10

0 50 100 150 200 250

0 50 100 150 200 250

0 50 100 150 200 250

Figure 17: Instruct model’s detailed performance on CodeR1.

• Knights-and-Knaves: Figures 18 and 19 show the effects of training on the KK dataset. Targeted training significantly boosts KK performance, with this improvement in logical reasoning generalizing well to mathematical tasks but negatively impacting coding ability.

###### Math

70

60

Accuracy(%)

50

MATH500

40

CountDown

AIME24

30

20

10

0

0 100 200 300 400 500 600 700 800

###### Code

70.0

HumanEval

MBPP

67.5

65.0

62.5

60.0

57.5

55.0

52.5

0 100 200 300 400 500 600 700 800

###### Puzzle

80

60

40

20

KK

ZebraLogicBench

0

0 100 200 300 400 500 600 700 800

Figure 18: Base model’s detailed performance on KK.

100

70

80

80

60

Accuracy(%)

75

50

MATH500

HumanEval

KK

60

AIME24

MBPP

ZebraLogicBench

40

70

CountDown

40

30

65

20

20

10

60

0 100 200 300 400 500 600 700 800

0 100 200 300 400 500 600 700 800

0 100 200 300 400 500 600 700 800

Figure 19: Instruct model’s detailed performance on KK.

• Logic Puzzle Baron: Figures 20 and 21 show the effects of training on the LPB dataset. Similar to the KK setting, the model’s performance on the ZebraLogicBench improves rapidly with targeted training.

###### Math

70

60

Accuracy(%)

50

MATH500

40

CountDown

30

AIME24

20

10

0

0 100 200 300 400 500

###### Code

75

70

65

60

55

HumanEval

50

MBPP

0 100 200 300 400 500

###### Puzzle

40

35

30

25

20

15

10

KK

5

ZebraLogicBench

0

0 100 200 300 400 500

Figure 20: Base model’s detailed performance on LPB.

###### Math

70

60

Accuracy(%)

50

MATH500

40

CountDown

AIME24

30

20

10

0

0 100 200 300 400 500

###### Code

80

75

HumanEval

MBPP

70

65

60

0 100 200 300 400 500

###### Puzzle

40

35

30

KK

25

ZebraLogicBench

20

15

10

0 100 200 300 400 500

Figure 21: Instruct model’s detailed performance on LPB.

### B Detailed Performance Results for Cross-Domain Composition

- • Puzzle + Math: As shown in Figure 22, under this setting, the model’s performance on math and puzzle tasks shows a steady improvement. However, for HumanEval, there is a slight improvement followed by a significant decline, which also demonstrates the catastrophic forgetting phenomenon in RLVR.
- • Math + Code: As shown in Figure 23, the model exhibits stable improvements across all aspects in this setting, both in-domain for math and code tasks. Additionally, the extra math data further enhances performance on code tasks. The model also generalizes to unseen puzzle-solving capabilities.
- • Puzzle + Code: As shown in Figure 24, although code performance shows gradual improvement, the in-domain puzzle performance does not exhibit significant improvement compared to other settings, indicating that code data has an additional negative impact on puzzle performance.

70

KK

70

70

60

ZebraLogicBench

60

60

50

Accuracy(%)

50

50

Math500

40

40

CountDown

40

30

AIME24

30

30

20

20

HumanEval

10

20

10

MBPP

0

0

10

0 20 40 60 80 100

0 20 40 60 80 100

0 25 50 75 100 125 150 175 200

Figure 22: Base model’s detailed performance on Puzzle + Math domain data.

###### Math

70

60

Accuracy(%)

50

MATH500

40

AIME24

CountDown

30

20

10

0

0 200 400 600 800 1000

###### Code

82.5

80.0

77.5

75.0

HumanEval

MBPP

72.5

70.0

67.5

65.0

0 200 400 600 800 1000

###### Puzzle

25

20

15

10

5

KK

ZebraLogicBench

0

0 200 400 600 800 1000

Figure 23: Base model’s detailed performance on Math + Code domain data.

###### Math

60

50

Accuracy(%)

40

MATH500

AIME24

30

CountDown

20

10

0

0 200 400 600 800 1000

###### Code

85.0

82.5

80.0

77.5

HumanEval

75.0

MBPP

72.5

70.0

67.5

65.0

0 200 400 600 800 1000

###### Puzzle

80

70

60

50

40

30

20

KK

10

ZebraLogicBench

0

0 200 400 600 800 1000

Figure 24: Base model’s detailed performance on Puzzle + Code domain data.

In addition, we have summarized the specific performance of all cross-domain combinations in Table 12:

Table 12: Model performance during cross-domain data composition.

Math Code Puzzle ALL

Training Data

MATH500 CountDown AIME24 Avg. HumanEval MBPP Avg. KK Zebra Avg. Avg. Base 56.40 1.05 10.00 22.48 70.12 64.80 67.46 17.86 0.27 9.07 31.50 Math 72.00 53.77 16.67 47.48 66.46 62.00 64.23 26.42 18.42 22.42 45.11 Puzzle 67.60 10.81 10.00 29.47 78.70 64.00 71.35 89.29 34.66 61.98 50.72 Code 50.80 0.04 6.67 19.17 80.49 67.40 73.95 13.85 31.24 22.55 35.78 Math + Puzzle 72.80 66.35 10.00 49.72 25.00 64.80 44.90 68.57 30.99 49.78 48.36 Puzzle + Code 60.60 22.25 13.33 32.06 84.15 65.60 74.88 75.00 35.30 55.15 50.89 Math + Code 69.60 62.07 10.00 47.22 82.32 67.80 75.06 24.00 26.67 25.34 48.92 Math + Puzzle + Code 73.60 52.33 23.33 49.75 78.66 68.60 73.63 68.14 31.31 49.73 56.57

### C Prompt Formats for Each Benchmark Evaluation

- Example C.1: MATH500 Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> (\boxed{}) </answer>. {problem}\n

- Example C.2: AIME24 Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> (\boxed{}) </answer>. {question}\n

- Example C.3: CountDown Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> </answer>. Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.

- Example C.4: HumanEval Now the user asks you to solve a code problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> </answer>. Complete the following python code:\n{prompt}\n

- Example C.5: Knights-and-Knaves (KK) Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight (2) Oliver is a knight (3)... </answer>. New question: {prompt}

- Example C.6: MBPP You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests: assert similar elements((3, 4, 5, 6),(5, 7, 4, 10))==(4, 5) assert similar elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) assert similar elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)

[BEGIN] ’def similar elements(test tup1, test tup2):

res = tuple(set(test tup1) & set(test tup2)) return (res)’

[DONE]

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests: assert is not prime(2) == False assert is not prime(10) == True assert is not prime(35) == True

[BEGIN] ’import math def is not prime(n):

result = False for i in range{ 2,int(math.sqrt(n)) + 1 }:

if n % i == 0:

result = True return result ’

[DONE]

You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests: assert heap queue largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3} == (85, 75, 65) assert heap queue largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2) == (85, 75) assert heap queue largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5) == (85, 75, 65, 58, 35)

[BEGIN] ’ import heapq as hq def heap queue largest(nums,n):

largest nums = hq.nlargest(n, nums) return largest nums ’

[DONE]

26

You are an expert Python programmer, and here is your task: {text} Your code should pass these tests: {test list}

- Example C.7: ZebraLogicBench (Zebra) # Example Puzzle There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:

- - Each person has a unique name: ‘Peter’, ‘Eric’, ‘Arnold’.
- - Each person has a unique favorite drink: ‘tea’, ‘water’, ‘milk’ ## Clues for the Example Puzzle

- 1. Peter is in the second house.
- 2. Arnold is directly left of the one who only drinks water.
- 3. The one who only drinks water is directly left of the person who likes milk. ## Answer to the Example Puzzle {

”reasoning”: ”Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.”, ”solution”: {

- ”House 1”: { ”Name”: ”Arnold”, ”Drink”: ”tea”

},

- ”House 2”: { ”Name”: ”Peter”, ”Drink”: ”water”

},

- ”House 3”: { ”Name”: ”Eric”, ”Drink”: ”milk”

} }

} # Puzzle to Solve {puzzle} # Instruction Now please solve the above puzzle. Present your reasoning and solution in the following json format: {json template}

