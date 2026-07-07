# arXiv:2511.06307v1[cs.LG]9Nov2025

[Figure 1]

2025-11-11

## DRIVE: Data Curation Best Practices for Reinforcement Learning wIth VErifiable Reward in Competitive Code Generation

#### Speed Zhu, Jianwei Cai, Guang Chen, Lulu Wu, Saiyong Yang, Wiggin Zhou∗†

Hunyuan Team, Tencent

### Abstract

Recent reasoning-first models (e.g., OpenAI o1, DeepSeek R1) have spurred a resurgence of interest in RLVR. Nevertheless, advances are dominated by mathematics (e.g., AIME), with competitive-programming code generation underexplored and data curation receiving less attention than RL algorithm design. We investigate how to construct RLVR datasets (i.e., RL prompts) and present practical training techniques that yield strong performance on competitiveprogramming code generation. Our pipeline begins with supervised fine-tuning (SFT) distilled from strong open-source models, augmented with general-purpose and reasoning-intensive data. RL then follows a two-stage process with executable, testcase-driven rewards: first, training on a large, uniformly distributed set of competitive-programming problems using Group Relative Policy Optimization (GRPO) with 8 rollouts per prompt and a relatively short response-generation window (e.g., 32k during SFT and 24k in this stage) to expand entropy and mitigate repetition and truncation; second, we perform Pre-GRPO: updating on a small, high-quality set of challenging problems with a large rollout budget (64 rollouts per prompt) under a hard-focus curriculum that continuously retains the most difficult instances throughout training. We implement our method on Qwen2.5-32B and evaluate on LeetCode and Codeforces weekly contests to avoid data leakage. The resulting model achieves state-of-the-art performance among models of similar scale and is comparable to leading systems such as DeepSeek v3.1 and Doubao-1.5-Thinking. We also examine scaling trends and observe strong RL scaling on an internal large-scale MoE model. Our study distills concise best practices for data curation, entropy expansion, and curriculum design in RLVR for competitive-programming code generation.

[Figure 2]

Figure 1: Performance of our models on various benchmark

### 1 Introduction

Recent advances in reasoning-first language models, exemplified by OpenAI o1 (Jaech et al., 2024) and DeepSeek R1 (Guo et al., 2025), have demonstrated remarkable capabilities in complex problem-solving tasks through reinforcement learning with verifiable rewards (RLVR). Although a large body of research centers on math benchmarks such as AIME (Yu et al., 2025; Li et al., 2025b; Lou et al., 2025; Yue et al., 2025; Liu et al., 2025), applying these models to competitive programming—a domain demanding both deep algorithmic insight and precise implementation—remains relatively underexplored (Zhao et al., 2024; Zhang et al., 2024). This gap is particularly notable given that competitive programming presents unique challenges: solutions must not only be logically correct but also executable, efficient, and capable of handling edge cases within strict computational constraints.

∗Corresponding author: wigginzhou@tencent.com †Emails: speedzhu@tencent.com; cmathxcai@tencent.com; maxluxchen@tencent.com; lukywu@tencent.com;

stevesyang@tencent.com

Additionally, prior work on RLVR has predominantly focused on developing novel algorithms and training techniques (Shao et al., 2024; Hu et al., 2025; Zheng et al., 2025; Li et al., 2025c), with considerably less attention paid to the critical aspects of data curation and curriculum design (Shen et al., 2025). Most existing approaches either apply uniform training strategies across all problem difficulties or rely on simplistic difficulty categorization (Seed et al., 2025; Yu et al., 2025; Li et al., 2025b; 2023), potentially limiting the model’s ability to tackle the most challenging problems that define competitive programming excellence. Furthermore, the computational demands of training large-scale models often make extensive experimentation prohibitive, necessitating more efficient and targeted training strategies.

In this work, we present a comprehensive study on applying RLVR to competitive programming code generation, with particular emphasis on practical data curation techniques and curriculum design. Our two-stage RL framework directly addresses limitations in standard SFT and RLVR, increasing exploration entropy, reducing repetitive generation, and improving performance on challenging problems. In stage one, we expand exploration entropy with a large, uniformly distributed pool of competitive-programming problems and moderate rollout budgets. In stage two, Pre-GRPO, we adopt a hard-focus curriculum and a high rollout budget to master the most challenging cases.

We implement our method on Qwen2.5-32B (Hui et al., 2024) and demonstrate its effectiveness through extensive evaluation on recent LeetCode and Codeforces weekly contests, benchmarks carefully selected to avoid data contamination. Our model achieves state-of-the-art performance among similarly sized models and remains competitive against much larger model such as DeepSeek V3.1 (Ta¸syürek et al., 2025), with particularly strong gains on challenging problems (up to 58% relative improvement over similarly sized models on Codeforces). Our comprehensive ablation studies confirm that entropy expansion delivers robust generalization and the hard-focus curriculum extends the model’s problem-solving frontier. In addition, we observe strong RL scaling on an internal large-scale MoE model.

Our key contributions are:

- • A two-stage RL framework that systematically addresses the limitations of normal RLVR methods through entropy expansion followed by hard-focus curriculum learning.
- • Empirical evidence demonstrating that large rollout budgets are crucial for learning challenging problems, while moderate budgets suffice for entropy expansion.
- • Comprehensive analysis of training dynamics revealing that standard RL struggles with difficult cases, motivating our curriculum-based approach.
- • State-of-the-art results on competitive programming benchmarks with a 32B parameter model, achieving performance comparable to models with 5–10× more parameters.
- • Practical insights for RLVR implementation, including data curation strategies, difficulty-aware curriculum design, and scaling trends validated on an internal large-scale MoE model

Our findings suggest that careful curriculum design and strategic use of computational resources can yield substantial improvements in competitive programming capabilities, providing a roadmap for future work in this challenging domain.

### 2 Related Work

RLVR Algorithms. Since DeepSeek R1 (Guo et al., 2025) demonstrated successful RLVR implementation on both Qwen-32B and DeepSeekV3, numerous studies have investigated performance improvements, particularly on the AIME benchmark. DAPO (Yu et al., 2025) was the first open-source method to enhance RLVR performance on AIME using the GRPO algorithm with Qwen2.5-32B. Subsequently, VAPO (Yue et al., 2025) proposed an enhanced PPO algorithm for further gains, while ProRL (Liu et al., 2025) explored GRPO optimization tricks to boost benchmark performance. Other works have focused on stabilizing RLVR training, including expert replay strategies for MoE models (Zheng et al., 2025) and truncated importance sampling to address mismatches between inference and training engines (Yao et al., 2025). However, despite these algorithmic advances, remarkably little attention has been paid to the critical aspect of constructing suitable RL prompts to enhance RLVR performance—a gap our work addresses.

RLVR Data Construction. Few studies examine how to construct RL training prompts to enhance RLVR or RLHF. Gao et al. (2025) propose a principled data-selection method for DPO, showing that overly difficult examples hinder alignment and should be filtered during training. Li et al. (2025a) introduce a strategic selection procedure that identifies key prompts from a full set, achieving comparable RLHF performance with only a subset of the data. Shen et al. (2025) propose Pre-PPO, a data-selection algorithm that enables RL scaling in RLHF by selecting suitable examples from large datasets. However, RL data construction for RLVR, especially for competitive-programming code generation, remains largely unexplored. To our knowledge, this is the first study to systematically investigate this problem in RLVR, and we analyze both performance and scaling trends on larger models.

RLVR performance scaling analysis. Although recent works—e.g., DeepSeek R1 (Guo et al., 2025), SEED-1.5Thinking (Seed et al., 2025), and Qwen3 (Yang et al., 2025)—present RLVR strategies across multiple benchmarks, few report performance on both small and large models; notable exceptions include DeepSeek R1 and Qwen3. In the RLHF domain, Shen et al. (2025) provide a comprehensive analysis of scaling trends. However, most RLVR evaluations focus on dense 7–32B models, leaving open questions about scaling to much larger models. In this paper, we conduct a comprehensive performance and strategy ablation on Qwen2.5-32B and then directly apply the resulting recipe to on an internal large-scale MoE model, achieving strong results. This demonstrates that our strategy is effective and suitable for scaling up.

### 3 Method

[Figure 3]

Figure 2: The training pipeline of our models.

Our training pipeline is illustrated in Figure 2. Our approach consists of two main phases: supervised fine-tuning followed by a two-stage reinforcement learning process with verifiable rewards.

#### 3.1 Supervised Fine-Tuning

We begin with supervised fine-tuning (SFT) on Qwen2.5-32B, distilling knowledge from strong open-source models. The training data is augmented with general-purpose and reasoning-intensive datasets to enhance the model’s problem-solving capabilities.

Specifically, we first collect competitive programming prompts from open-source datasets. We then train a small model to classify problem difficulty and assign difficulty labels to each problem, categorizing them into three types: easy, medium, and hard. We duplicate hard problems twice in the training data, following a similar approach to Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving (Tong et al., 2024). However, we do not directly adopt their RFT method, as it requires extensive model sampling which is computationally expensive for our use case. Our simpler duplication strategy achieves similar benefits of emphasizing difficult problems while maintaining computational efficiency. Furthermore, we collect additional prompts from other code generation tasks beyond competitive programming. We find that including both general-purpose coding data and reasoning-intensive problems significantly improves performance on competitive programming tasks, likely due to enhanced reasoning and code comprehension abilities.

[Figure 4]

Figure 3: The entropy comparison of 24k-style training and 32k-style training Table 1: Results on LeetCode Weekly OJ (32 problems) and Codeforces OJ (33 problems).

LeetCode Weekly OJ (32) Codeforces OJ (33) Pass@10 Avg@1 Pass@10 Avg@1

Method

SFT Model 96.88% 57.81% 24.24% 11.52% DeepseekV3.1 96.88% 68.75% 33.33% 16.06% Seed1.6-0715 96.88% 74.38% 39.39% 18.79%

#### 3.2 Two-Stage Reinforcement Learning

Analysis of the SFT Model for the RL Phase. Following extensive distilled SFT, we identify three critical issues that motivate our two-stage RL design. First, as shown in Figure 3, the model exhibits low entropy and limited exploration, repeatedly converging to similar solution modes. Second, it shows repetitive generation patterns, often producing redundant code structures or falling into loops that lead to truncated outputs (see Sec. B for details). Third, harder problems tend to require longer responses, which are harder to learn and generalize during SFT due to long-sequence dependency challenges. Empirically (Table 1), the SFT model’s pass@10 on the LeetCode Weekly OJ is comparable to DeepSeek V3.1 and Seed1.6-0715, but on the (harder) Codeforces Weekly OJ its pass@10 lags behind both. These observations directly inform our RL strategy: Stage 1 targets entropy expansion and pattern diversification, while Stage 2 strengthens performance on challenging problems.

- Stage 1: Entropy Expansion. In the first stage, we perform training on a mixed dataset of approximately 9k competitive programming problems. Each prompt is sampled with 8 rollouts with a total sequence length of 24k tokens (including both prompt and response) to increase output entropy and mitigate mode collapse. This stage serves several critical purposes: (1) Entropy Enhancement - training on uniformly distributed, medium-difficulty problems expands the model’s output diversity; (2) Pattern Reduction - the increased entropy reduces repetitive generation patterns, significantly decreasing truncated errors; (3) Overall Performance - this training stage provides substantial improvements to the model’s general competitive programming capabilities.
- Stage 2: Hard-Focus Curriculum. The second stage focuses on high-quality, challenging problems using LiveCode V6 data filtered by Pre-GRPO. Pre-GRPO carries forward a subset of low–pass-rate cases from each phase to the next. Specifically, we train for 32k steps under a three-phase progressive curriculum: Phase 1 uses the 72 hardest cases with a 64-step budget, Phase 2 the 50 hardest with a 32-step budget, and Phase 3 the 25 hardest with a 32-step budget. We sample 80 rollouts per prompt in this stage, as large rollout sampling is crucial for stable performance gains. This curriculum continuously retains the most difficult instances, pushing the model to master increasingly challenging problems while maintaining performance on easier cases.

Table 2: Pass@1 performance comparison across different evaluation benchmarks.

Model LiveCode LiveCode LiveCode LeetCode Codeforces 08-11 V5 V6 Weekly (32) OJ (33)

DeepseekV3.1 (64k) 0.692 0.713 0.693 0.688 0.161 Seed1.6-0715 (64k) 0.803 0.824 0.770 0.743 0.188 Qwen3-235B-2507 (64k) 0.681 0.713 0.646 0.688 0.200 QwQ-32B (32k) 0.578 0.569 0.537 0.472 0.124 OpenReasoning-Nemotron-32B (32k) 0.623 0.618 0.600 0.609 0.132

SFT model (32k) 0.602 0.594 0.549 0.578 0.115 RL Stage 1 model (24k) 0.625 0.627 0.634 0.603 0.112 RL model (32k) 0.699 0.697 0.703 0.653 0.182

Relative Improvement (RL vs SFT) +16.1% +17.3% +28.1% +13.0% +58.3%

### 4 Experiments

- 4.1 Experimental Setup We describe our experimental setup as follows:

- • Models: We conducted experiments with the open-source Qwen2.5-32B-Instruct and trained our SFT model on top of this foundation. We also examined scaling trends on an internal large-scale MoE model.
- • Data: For SFT data, we initially collected 1.27M open-source prompts and generated corresponding responses through distillation from the open-source model DeepSeekR1-0528. We then refined this dataset to 470K highquality prompts using a 5-round arena learning method. For the first stage of RL training, we utilized 9K prompts from open-source repositories. In the second stage of RL, we employed the LiveCode V6 dataset, which comprises 175 high-quality examples with comprehensive test cases.
- • Experimental Details of SFT: We trained the model on the selected 470K SFT data for 3 epochs using the Qwen2.5-32B-Instruct as our base model. The training was conducted with a learning rate of 1 × 10−5, utilizing 256 GPUs with a global batch size of 512.
- • Experimental Details of RL: In the Entropy Expansion RL stage, we trained the model on 9K prompts for 32 steps by GRPO algorithm. Each prompt is sampled with 8 rollouts with a total sequence length of 24k tokens. In the Hard-Focus Curriculum RL stage, we continued training on progressively harder problems as specified in Section 3, where each prompt is sampled with 64 rollouts with a total sequence length of 32k tokens.
- • Evaluations: We construct a comprehensive evaluation set covering competitive code generation tasks. Specifically, we adopt LiveCode08-11 (166 problems), LiveCodeV5 (167 problems), and LiveCodeV6 (175 problems) benchmarks as our validation sets. While LiveCodeV6 was exposed during the second RL stage, both LiveCode0811 and LiveCodeV5 remain uncontaminated throughout the SFT and RL stages. Furthermore, to ensure the integrity of our experimental results and avoid data leakage, we evaluate our models on recent LeetCode and Codeforces weekly contests that were released after our training data collection cutoff.

- 4.2 Experimental Results The experimental results presented in Table 1 demonstrate the following key findings:

- • Competitive Performance with Smaller Scale: Despite having only 32B parameters and 32k context length, our RL model achieves comparable or superior performance to mainstream large-scale models like DeepSeek-V3.1 (64k), with particularly strong results on LiveCode 08-11 and V5 benchmarks (0.699-0.697 vs 0.692-0.713), demonstrating remarkable efficiency in the parameter-performance trade-off.
- • Significant Advancement Over 32B SOTA: Our RL model substantially outperforms the 32B state-of-the-art OpenReasoning-Nemotron-32B across all benchmarks, with improvements ranging from +0.076 (08-11) to

+0.079 (V6) on LiveCode benchmarks and a notable +0.056 improvement on the challenging Codeforces OJ (0.182 vs 0.132), representing a 37.8% relative improvement on competitive programming tasks.

- • Substantial Improvement Over SFT Baseline: The RL training delivers consistent and significant improvements across all evaluation benchmarks, with absolute gains ranging from +0.077 (08-11) to +0.103 (V6), representing relative improvements of 16.1% (08-11) to 17.3% (V6) on LiveCode benchmarks.
- • Robust Generalization Across Diverse Benchmarks: Our model demonstrates strong generalization capabilities, achieving substantial improvements on both easier benchmarks (LiveCode series) and challenging competitive programming tasks (Codeforces), indicating effective learning of fundamental coding principles.

Table 3: Ablation study on different SFT training strategies.

LiveCode 08-11

LiveCode V5

LiveCode V6

LeetCode Weekly (32)

Codeforces OJ (33)

Method

Basic SFT Strategy 0.582 0.603 0.545 0.558 0.112 Arena Learning Strategy 0.600 0.598 0.542 0.553 0.111 Twice Hard Learning Strategy 0.602 0.594 0.549 0.578 0.115

Table 4: Ablation study on different RL training strategies.

LiveCode 08-11

LiveCode V5

LiveCode V6

LeetCode Weekly (32)

Codeforces OJ (33)

Method

SFT Model 0.602 0.594 0.549 0.578 0.115 RL with all LiveCodeV6 dataset 0.506 0.512 0.522 0.296 0.105 RL with all 9k data 0.676 0.688 0.675 0.592 0.102 RL without First Stage Only Second Stage

0.636 0.626 0.691 0.550 0.142

Our Method 0.699 0.697 0.703 0.653 0.182 Our Method with More Harder Samples in Stage 2

0.712 0.707 0.743 0.678 0.188

- • Effective RL Training Strategy: The stepwise improvement from SFT to RL Stage 1 to the final RL model validates our approach. The RL model delivers the largest gains on complex reasoning tasks, with 13.0% relative improvement on LiveCode weekly OJ and 58.3% on Codeforces weekly OJ.

#### 4.3 SFT Experiments

Ablation Study. To assess how different SFT strategies affect model generalization, we conduct an ablation comparing three methods. Results are shown in Table 3.

- • Basic SFT strategy: The baseline model is trained on the full dataset of 1.27M prompts.
- • Arena Learning strategy (Luo et al., 2024): An iterative curriculum-learning approach. The dataset is split into five folds. We train an initial model on the first fold, use it to make predictions on the next fold, and retain only the samples the model fails on (“hard samples”) for the subsequent training stage. Repeating this process yields a condensed dataset of 470K hard samples.
- • Twice Hard Learning strategy: Building on Arena Learning, this method first identifies the hard samples and then doubles their exposure during training, keeping the overall training token budget comparable to the Basic SFT strategy while focusing computation on more challenging data.

As shown in the Table 3, the Twice Hard Learning Strategy almostly outperforms the other methods across all evaluated benchmarks, achieving the best results.

Key Findings. Allocating a larger training token budget to hard samples proves effective. We observe that the Arena Learning Strategy, despite reducing the training dataset from 1.27M to 470K prompts (a reduction of over 60%), maintains performance comparable to the Basic SFT Strategy. This indicates that focusing on hard samples is an efficient approach. However, its slight underperformance can be attributed to an insufficient training token budget, which leads to incomplete learning during the SFT stage. To validate this hypothesis, the Twice Hard Learning Strategy was designed to restore the token budget by specifically oversampling the identified hard examples. The superior performance of this strategy, as shown in Table 3, confirms our hypothesis. This demonstrates that for optimal performance, it is crucial not only to identify challenging data but also to allocate sufficient computational resources for the model to learn from them effectively.

#### 4.4 The RL Experiments

Ablation Study. To validate the effectiveness of our two-stage RL approach, we conduct ablation experiments by removing key components from our method. Table 4 and Figure 4 present the results of three ablation variants compared to our full method, showing performance on both LiveCodeV6 (training set) and LiveCode08-11 (validation set) during the training process. Specifically, we examine: (1) RL with all LiveCode V6 dataset, which applies 24k-style RLVR using all 175 LiveCodebench V6 problems directly on the SFT model; (2) RLVR with all 9k data, which uses the complete 9k training dataset from the first stage to perform 32k-style RLVR on the SFT model;

[Figure 5]

[Figure 6]

(a) The performance on LiveCodeV6 during training (b) The performance on LiveCode08-11 during training

Figure 4: The performance of different RL training strategies on LiveCodeV6 and LiveCode08-11 during training

and (3) RL without First Stage (Second Stage Only), which directly applies the Stage 2 RLVR strategy (training only on the harder samples) to the SFT model, bypassing the first stage entirely.

Key Findings: Difficulty-aware training is crucial. Difficulty-aware training is crucial. RL works best when driven by challenging samples: focusing Stage 2 RLVR on the hard subset yields stable improvements. In contrast, mixing easy problems into RL (training on the full 9k or the entire LiveCodeV6 without hardness filtering) fails to obtain best performance and can even cause training/performance to collapse, with the largest drop on LeetCode Weekly (-48.8%). This indicates that hard samples provide a useful reward signal, whereas easy samples dilute the signal and even destabilize optimization. (2) Entropy expansion stage matters. Removing the first stage and only applying hard-focus curriculum RL (row 3) yields mixed results—while it improves on contaminated LiveCodeV6 (+25.9%), it actually hurts performance on out-of-distribution benchmarks like LeetCode weekly OJ (-4.8%). This suggests that the entropy expansion stage helps the model develop robust problem-solving capabilities that generalize beyond the training distribution. (3) Both stages synergize effectively. Our complete two-stage method (highlighted in green) achieves the best performance across all benchmarks, with improvements ranging from 13.0% on LeetCode weekly OJ to 58.3% on Codeforces weekly OJ over the SFT baseline, confirming that the combination of entropy expansion followed by hard-focus curriculum learning is essential for optimal performance.

#### 4.5 Further Analysis

Analysis of training dynamics during standard RL training on 175 LeetCodeV6 cases. We investigate the training dynamics by clustering cases based on their initial rollout accuracy (using rollout = 8 as the evaluation metric) and monitoring the evolution of training accuracy across different difficulty clusters. As illustrated in Figure 5, cases with medium initial accuracy demonstrate the most rapid improvement, substantially outperforming both low- and high-accuracy clusters. While the modest gains in high-accuracy clusters are expected—these problems are already largely solved—the stagnant progress in low-accuracy clusters raises concerns about the model’s ability to master challenging problems. This pattern indicates that standard RL training struggles with difficult cases, potentially creating a capability ceiling that limits the model’s performance on complex problemsolving tasks. These findings motivate our two-stage approach, which specifically addresses this limitation through targeted curriculum learning.

Analysis of few-case and single-case training. To obtain a finer-grained view of how the model learns individual cases, we trained on drastically reduced datasets—four cases and, in the extreme, a single case. The results mirror those from large-scale training: certain “hard” cases are exceptionally difficult to master. Even when the model was trained exclusively on a single hard case for 60 steps, it struggled to reach satisfactory performance. This motivates our second-stage strategy, which combines large rollouts with targeted focus on hard cases. The goal is to compel the model to learn these challenging examples, thereby pushing its capabilities and extending its generalization frontier. More details are provided in Appendix A.

Ablation of the large-rollout setting in single-case training. Due to computational constraints, we conduct this ablation in the single-case training setting. Consistent with the analysis above, some hard cases remain difficult to learn even in this setting. Intuitively, increasing the rollout count should improve fitting efficiency. As shown in Figure 6, higher rollout counts accelerate learning. Furthermore, we observe: 1) Training on a single case has negligible impact on the performance of other cases, including easy ones. 2) Learning a hard case can improve generalization to other hard cases. More details are provided in Appendix A.

More challenge training samples in the stage2. Although harder samples in LiveCodeBench V6 already boost

[Figure 7]

- Figure 5: The Accuracy Trends by First Apperance Accuracy Clusters

[Figure 8]

- Figure 6: Effect of Rollout Number on Single-Case Training Performance

performance, we further investigated whether increasing the proportion of difficult training examples would continue to help. We curated 109 hard cases from our internal competitive-programming dataset (out of 633 candidates) and incorporated them into Stage 2. As shown in Table 4, training with a larger share of challenging samples consistently improves results across all benchmarks. Overall, this supports the conclusion that targeted inclusion of hard cases is an effective and scalable lever for improving model performance and generalization under limited compute.

Table 5: Scaling trends for RL training strategies. Metrics are decimals (3 d.p.); relative improvements are percentages.

LeetCode Weekly (32) avg@1

Codeforces (33) avg@1 SFT model (64k) 0.681 0.692 0.656 0.627 0.155

LiveCode 08–11

LiveCode V5

LiveCode V6

Method

- RL Stage 1 (32k train, 64k test) 0.690 0.740 0.665 0.611 0.123

- RL Stage 2 (64k, train step=50) 0.708 0.744 0.737 0.722 0.194 Relative Improvement

- (Stage 1 vs SFT) +1.32% +6.94% +1.37% -2.55% -20.65%

Relative Improvement

- (Stage 2 vs SFT) +3.96% +7.51% +12.35% +15.17% +25.16%

- 4.6 Scaling Trends

To evaluate the scalability and effectiveness of our training methodology on a large-scale model, we applied it to competitive programming tasks for both SFT and RLVR. The experiments were conducted using a large-scale internal Mixture-of-Experts (MoE) model.

Our two-stage training process was configured as follows:

- • Stage 1 (Entropy Expansion): The model was trained for 30 steps on a dataset of 9k samples. During this stage, we used a batch size of 512 with a rollout number of 16.
- • Stage 2 (Curriculum-based RL): This stage consisted of a three-phase curriculum totaling 50 steps: 20 steps for Phase 1, 20 for Phase 2, and 10 for Phase 3. For this RL phase, the training configuration was a batch size of 128 with a larger rollout number of 64.

The performance results of this process are detailed in Table 5. An analysis of the table reveals several key insights:

- • Baseline Performance: The initial SFT model serves as a strong baseline, establishing the model’s capabilities before RL fine-tuning.
- • Intermediate Stage Effect: After the entropy-expansion first stage, the model shows improved performance on some internal benchmarks (e.g., LiveCodeBench-V5) but a slight regression on external, more challenging ones like LeetCode and Codeforces weakly OJ. This suggests that while exploration broadens the model’s policy, it requires the targeted learning of Stage 2 to become effective.
- • Final Stage Efficacy: Upon completing the second stage, the model demonstrates substantial and consistent performance gains across all benchmarks. As highlighted by the Relative Improvement metrics, our final model achieves gains of up to +15.17% on the LeetCode Weekly OJ and +7.51% on LiveCodeBench-V5 compared to the SFT baseline.

These results validate the effectiveness of our two-stage strategy and demonstrate positive scaling trends when applying RL with a carefully designed curriculum. It is important to note that these experiments were constrained by computational resources, and training was not continued to full convergence. For future work, we plan to integrate a greater number of challenging problems into the second stage curriculum. We anticipate this will lead to further performance gains, which will be presented in a subsequent formal report.

### 5 Conclusion

In this work, we presented a two-stage reinforcement learning framework for competitive programming that addresses key limitations of supervised fine-tuning through entropy expansion and hard-focus curriculum learning. Our approach achieves state-of-the-art performance among 32B parameter models, with improvements ranging from 13% to 58% across various benchmarks, demonstrating that careful data curation and curriculum design are as crucial as algorithmic innovations for RLVR success.

Our analysis reveals three critical insights: (1) standard RL training struggles with hard problems, creating a capability ceiling that limits model performance; (2) large rollout budgets (64+ samples) are essential for mastering challenging cases; and (3) progressive curriculum learning that continuously retains the hardest problems outperforms uniform difficulty distribution. Ablation studies confirm both stages are necessary—entropy expansion enables robust generalization while hard-focus curriculum pushes the problem-solving frontier.

The scaling experiments on an internal large-scale MoE model validate that our principles transfer to larger scales. While our approach requires substantial computational resources for large rollout sampling, it provides a practical roadmap for training models capable of tackling complex algorithmic challenges. Future work could explore adaptive curriculum strategies and more efficient sampling techniques to further improve the cost-effectiveness of RLVR training for competitive programming.

### References

Chengqian Gao, Haonan Li, Liu Liu, Zeke Xie, Peilin Zhao, and Zhiqiang Xu. Principled data selection for alignment: The hidden risks of difficult examples. arXiv preprint arXiv:2502.09650, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models. arXiv preprint arXiv:2501.03262, 2025.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling. arXiv preprint arXiv:2502.11886, 2025a.

Yizhi Li, Qingshui Gu, Zhoufutu Wen, Ziniu Li, Tianshun Xing, Shuyue Guo, Tianyu Zheng, Xin Zhou, Xingwei Qu, Wangchunshu Zhou, et al. Treepo: Bridging the gap of policy optimization and efficacy and inference efficiency with heuristic tree-based modeling. arXiv preprint arXiv:2508.17445, 2025b.

Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023.

Ziniu Li, Congliang Chen, Tianyun Yang, Tian Ding, Ruoyu Sun, Ge Zhang, Wenhao Huang, and Zhi-Quan Luo. Knapsack rl: Unlocking exploration of llms via optimizing budget allocation. arXiv preprint arXiv:2509.25849, 2025c.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.

Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, and Shuangzhi Wu. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning. arXiv preprint arXiv:2505.11896, 2025.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Qingwei Lin, Jianguang Lou, Shifeng Chen, Yansong Tang, and Weizhu Chen. Arena learning: Build data flywheel for llms post-training via simulated chatbot arena. arXiv preprint arXiv:2407.10627, 2024.

ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Wei Shen, Guanlin Liu, Zheng Wu, Ruofei Zhu, Qingping Yang, Chao Xin, Yu Yue, and Lin Yan. Exploring data scaling trends and effects in reinforcement learning from human feedback. arXiv preprint arXiv:2503.22230, 2025.

Makbule Ta¸syürek, Özkan Adıgüzel, Mustafa Gündo˘gar, Myroslav Goncharuk-Khomyn, and Hatice Ortaç. Comparative evaluation of the responses from chatgpt-5, gemini 2.5 flash, and deepseek-v3. 1 chatbots to patient inquiries about endodontic treatment in terms of accuracy, understandability, and readability. International Dental Research, 15(Advanced Online), 2025.

Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He Dart-math. Difficulty-aware rejection tuning for mathematical problem-solving. 2024. URL https://arxiv.org/abs/2407.13690, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, August 2025. URL https://fengyao.notion. site/off-policy-rl.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Chuheng Zhang, Wei Shen, Li Zhao, Xuyun Zhang, Xiaolong Xu, Wanchun Dou, and Jiang Bian. Policy filtration for rlhf to mitigate noise in reward models. arXiv preprint arXiv:2409.06957, 2024.

Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. Marco-o1: Towards open reasoning models for open-ended solutions. arXiv preprint arXiv:2411.14405, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

### A Few Cases Training

[Figure 9]

[Figure 10]

(a) Case 1 Accuracy Evolution During Few-Shot Training (b) Case 2 Accuracy Evolution During Few-Shot Training

[Figure 11]

[Figure 12]

(c) Case 3 Accuracy Evolution During Few-Shot Training (d) Case 4 Accuracy Evolution During Few-Shot Training

- Figure 7: Case-wise Accuracy Trajectories Under Few-Shot Learning

[Figure 13]

- Figure 8: The Accuracy Trends by First Apperance Accuracy Clusters

To obtain a finer-grained view of how the model learns individual cases, we trained on drastically reduced datasets—four cases and, in the extreme, a single case. Specifically, as shown in Figure 7, we randomly selected 4 cases from LiveCodeBench V6’s 175 problems and trained them using the GRPO algorithm with rollout num = 8. The learning curves reveal striking disparities in training dynamics across different problems. Cases 2 and

- 3 demonstrate rapid convergence, with accuracy surging from 15.6% and 22% to approximately 80% within 60 steps, suggesting these problems align well with the model’s inductive biases. In stark contrast, Case 1 exhibits only modest improvement from 12% to 36.7%, while Case 4 remains virtually frozen at 12.5% throughout training, underscoring that certain "hard" cases pose exceptional learning challenges.

Further investigation through single-case training, as shown in Figure 8, reveals an intriguing phenomenon: when Case 1 is trained in isolation, its learning trajectory becomes noticeably slower compared to the four-case setting. This observation provides evidence for positive transfer effects—the simultaneous training on Cases 2, 3, and

- 4 appears to facilitate Case 1’s learning, suggesting that even diverse competitive programming problems share latent structural patterns that enable cross-problem generalization. However, the persistent difficulty with Case 1 (plateauing at merely 36.7% after 60 steps) and the complete stagnation of Case 4 expose a fundamental training imbalance inherent in standard RL approaches, where the optimization process naturally gravitates toward easily

learnable patterns while failing to break through on challenging problems. This phenomenon directly motivates our second-stage strategy employing large rollout budgets, specifically designed to compel the model to overcome these learning barriers on hard cases.

### B Case Study for Repetition Patterns

When training models directly on algorithmic problem-solving tasks without extensive 24k-style training data (as demonstrated in the following case study), we observe significant repetition patterns that indicate inefficient reasoning strategies. This case study examines a graph isomorphism optimization problem where the model demonstrates various forms of computational redundancy and cyclical reasoning behaviors.

- • Computational Template Repetition: The core permutation validation algorithm (inverse mapping construction, edge comparison, cost calculation) is applied systematically across dozens of different permutations, each following identical logical steps with different parameters.
- • Granular Analysis Loops: Each permutation analysis involves repetitive edge-by-edge evaluation following the same pattern: determine desired state → check current state → calculate cost difference → accumulate total, creating highly formulaic micro-computations.
- • Hypothesis-Testing Cycles: The reasoning exhibits recurring cycles of "try permutation X → calculate cost

→ compare to expected → express confusion → try next permutation," suggesting systematic but potentially inefficient exploration strategies.

- • Input Processing Redundancy: Nearly identical parsing and data structure construction logic appears multiple times across different sample inputs, indicating missed abstraction opportunities in edge normalization and set construction.
- • Debugging Iteration Patterns: Systematic testing of permutation variations (identity, single swaps, compound swaps) using identical verification procedures creates computational redundancy despite thorough coverage.
- • Meta-Cognitive Loops: Higher-level reasoning patterns repeat the cycle of acknowledging complexity → attempting enumeration → encountering discrepancies → expressing uncertainty → restarting with variations, suggesting cognitive inefficiencies when facing algorithmic uncertainty.

These patterns suggest that without proper training on diverse reasoning examples within 24k context, models tend to fall into repetitive computational habits that, while methodical, lack the efficiency and abstraction capabilities demonstrated by more extensively trained systems.

Given the extensive length of the RL model’s generated response in this case, we present only the analysis conducted by GPT-5, which identified numerous repetition patterns throughout the reasoning process.

- 1 Looking at the content in `<think></think>`, I found extensive repetitive permutation calculation patterns. Here are the main repetitive patterns:

- 2

- 3 ## Repetition Pattern 1: Basic Structure of Permutation Verification

- 4

- 5 ```python

- 6 # This pattern repeats dozens of times in <think>

- 7 def verify_permutation_cost(p, n, G_set, H_set, cost):

- 8 """Verify the cost calculation for permutation p"""

- 9 q = [0] * n

- 10 for i in range(n):

- 11 q[p[i]] = i

- 12

- 13 total_cost = 0

- 14 for i in range(n):

- 15 for j in range(i+1, n):

- 16 u = min(q[i], q[j])

- 17 v = max(q[i], q[j])

- 18 desired = 1 if (u, v) in G_set else 0

- 19 current = 1 if (i, j) in H_set else 0

- 20 if desired != current:

- 21 total_cost += cost[i][j]

- 22

- 23 return total_cost

- 24 ```

- 25

- 26 ## Repetition Pattern 2: Edge State Checking Logic

- 27

- 28 ```python

- 29 # Edge state checking that appears repeatedly in <think>

- 30 def check_edge_state(i, j, q, G_set, H_set):

- 31 """Check the desired and current state of edge (i,j)"""

- 32 u = min(q[i], q[j])

- 33 v = max(q[i], q[j])

- 34 desired = 1 if (u, v) in G_set else 0

- 35 current = 1 if (i, j) in H_set else 0

- 36 return desired, current

- 37 ```

- 38

- 39 ## Repetition Pattern 3: Detailed Permutation Calculations

- 40

- 41 There are many such repetitive calculations in `<think>`:

- 42

- 43 ```python

- 44 # Example extraction of repetitive patterns

- 45 def detailed_permutation_analysis():

- 46 """This detailed analysis repeats many times in think"""

- 47 examples = [

- 48 # p = [0, 1, 2, 3, 4] (identity)

- 49 {

- 50 "permutation": [0, 1, 2, 3, 4],

- 51 "analysis": [

- 52 "(0,1): desired=1, current=1 -> 0",

- 53 "(0,2): desired=0, current=1 -> flip: cost[0][2]=1",

- 54 "(0,3): desired=0, current=1 -> flip: cost[0][3]=4",

- 55 "(0,4): desired=0, current=1 -> flip: cost[0][4]=1",

- 56 "(1,2): desired=1, current=0 -> flip: cost[1][2]=5",

- 57 # ... more edge analyses

- 58 ],

- 59 "total": "1+4+1+5+6+3 = 20"

- 60 },

- 61

- 62 # p = [0, 1, 2, 4, 3] (swap 3 and 4)

- 63 {

- 64 "permutation": [0, 1, 2, 4, 3],

- 65 "analysis": [

- 66 "(0,1): 0",

- 67 "(0,2): desired=0, current=1 -> flip: cost[0][2]=1",

- 68 "(0,3): desired=0, current=1 -> flip: cost[0][3]=4",

- 69 # ... similar repetitive analysis

- 70 ],

- 71 "total": "some calculation result"

- 72 }

- 73 # There are dozens of such examples in think

- 74 ]

- 75 return examples

- 76 ```

- 77

- 78 ## Repetition Pattern 4: Sample Verification Structure

- 79

- 80 ```python

- 81 # Repetitive pattern for sample input processing

- 82 def process_sample_input(sample_name, n, G_edges, H_edges, cost_matrix, expected_output):

- 83 """Common pattern for processing sample inputs"""

- 84 print(f"Sample {sample_name}:")

- 85 print(f"N={n}")

- 86 print(f"G edges: {G_edges}")

- 87 print(f"H edges: {H_edges}")

- 88 print(f"Cost matrix: {cost_matrix}")

- 89 print(f"Expected: {expected_output}")

- 90

- 91 # Convert to sets

- 92 G_set = build_edge_set(G_edges)

- 93 H_set = build_edge_set(H_edges)

- 94

- 95 # Try various permutations

- 96 min_cost = float('inf')

- 97 for perm in all_permutations(n):

- 98 cost = verify_permutation_cost(perm, n, G_set, H_set, cost_matrix)

- 99 min_cost = min(min_cost, cost)

- 100

- 101 print(f"Computed: {min_cost}")

- 102 return min_cost == expected_output

- 103 ```

- 104

- 105 ## Repetition Pattern 5: Confusion Expression Statements

- 106

- 107 ```python

- 108 # Confusion and retry statements that appear repeatedly in think

- 109 confusion_patterns = [

- 110 "I am not finding 9.",

- 111 "But the sample output is 9.",

- 112 "How do we get 9?",

- 113 "This is not matching the sample operations.",

- 114 "Given the complexity...",

- 115 "How about we try the permutation: p = [...]",

- 116 "total = ... -> Not X.",

- 117 "But wait, ...",

- 118 "I see the mistake: ...",

- 119 "Therefore, ..."

- 120 ]

- 121 ```

- 122

- 123 These repetitive patterns indicate extensive trial-and-error processes and redundant calculations in the `<think>` section, which could be simplified by extracting common functions and establishing systematic verification workflows.

