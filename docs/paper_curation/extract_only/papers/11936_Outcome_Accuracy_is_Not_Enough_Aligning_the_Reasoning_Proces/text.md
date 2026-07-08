# arXiv:2602.04649v1[cs.CL]4Feb2026

[Figure 1]

2026-02-05

## Outcome Accuracy is Not Enough: Aligning the Reasoning Process of Reward Models

Binghai Wang1,2‡, Yantao Liu1, Yuxuan Liu1, Tianyi Tang1, Shenzhi Wang1,3‡, Chang Gao1, Chujie Zheng1, Yichang Zhang1, Le Yu1, Shixuan Liu1, Tao Gui2∗, Qi Zhang2, Xuanjing Huang2, Bowen Yu1, Fei Huang1∗, Junyang Lin1

###### 1Qwen Team, Alibaba Group 2Fudan University 3Tsinghua University

https://github.com/QwenLM/RationaleRM

#### Abstract

Generative Reward Models (GenRMs) and LLM-as-a-Judge exhibit deceptive alignment by producing correct judgments for incorrect reasons, as they are trained and evaluated to prioritize Outcome Accuracy, which undermines their ability to generalize during RLHF. We introduce Rationale Consistency, a fine-grained metric that quantifies the alignment between the model’s reasoning process and human judgment. Our evaluation of frontier models reveals that rationale consistency effectively discriminates among state-of-the-art models and detects deceptive alignment, while outcome accuracy falls short in both respects. To mitigate this gap, we introduce a hybrid signal that combines rationale consistency with outcome accuracy for GenRM training. Our training method achieves state-of-the-art performance on RM-Bench (87.1%) and JudgeBench (82%), surpassing outcome-only baselines by an average of 5%. Using RM during RLHF, our method effectively improves performance as demonstrated on Arena Hard v2, notably yielding a 7% improvement in creative writing tasks. Further analysis confirms that our method escapes the deceptive alignment trap, effectively reversing the decline in rationale consistency observed in outcome-only training.

Gemini Claude GPT

0.400

The Authentic Alignment Zone (OA Saturates, RC Differentiates)

DeepSeek

0.375

Qwen

- o3

GPT-4.1

- o3-mini

###### Gemini 3 Pro

0.350

GPT-5

RationaleConsistency(RC)

0.325

Gemini 3 Flash

###### Qwen3 Plus Thinking

###### Gemini 2.5 Flash

Claude 4

0.300

DeepSeek R1 DeepSeek R1-0528

Claude 4.5

Qwen3 Plus Instruct

0.275

Claude 3.7

DeepSeek V3.2

0.250

GPT-4o

Claude 3.5 Sonnet

Qwen3 Coder

The Deceptive Alignment Trap (High Outcome via Non-Human Logic)

0.225

Gemini 2.0 Flash

0.200

0.76 0.77 0.78 0.79 0.80 0.81 0.82

Outcome Accuracy (OA)

Figure 1: Outcome Accuracy vs. Human Rationale Consistency. Rationale consistency effectively discriminates among state-of-the-art models and detects deceptive alignment.

‡ Intern at Qwen Team, Alibaba Group. ∗ Corresponding author. Emails: wangbh25@m.fudan.edu.cn, tgui@fudan.edu.cn, feihu.hf@alibaba-inc.com

#### 1 Introduction

Recent studies observe a phenomenon where reward models achieve high accuracy on static datasets but fail to generalize effectively during RLHF (Gao et al., 2023; Casper et al., 2023; Lambert et al., 2024). We attribute this to the reliance on outcome supervision, where the model is optimized solely to predict human-annotated binary preferences. Models trained this way tend to learn spurious correlations or shortcuts to maximize Outcome Accuracy without genuine reasoning (Anand et al., 2024; Geirhos et al., 2020; Uesato et al., 2022), leading to the Deceptive Alignment of the RM (Hubinger et al., 2019; Carlsmith, 2023), where the RM predicts the right label but for the wrong reasons. The recent emergence of Generative RMs (GenRM, Liu et al., 2025; Chen et al., 2025; Guo et al., 2025) and LLM-as-a-judge (Zheng et al., 2023; Gu et al., 2024), which generate rationales prior to the final judgment, enables us to explicitly examine the reasoning process. Therefore, we argue for the critical importance of rationale supervision, ensuring that the RM’s reasoning process matches human judgment.

We introduce Rationale Consistency, a fine-grained evaluation metric designed to verify the alignment between the model’s evaluation process and human judgment. To measure this with precision, we construct METAJUDGE, a framework that decomposes human rationales into mutually exclusive atomic units, prompts the model to output a corresponding list of rationales, and employs an LLM to perform strict one-to-one semantic matching, thereby quantifying the exact proportion of human rationales recovered by the model.

Through a comprehensive evaluation of 19 state-of-the-art LLMs, we identify a critical disconnect between the correctness of a model’s outcome and the validity of its judgment, as shown in Figure 1. This disconnect manifests in two key limitations: (1) Outcome accuracy masks deceptive alignment. We find that high outcome accuracy does not guarantee robust judgment. A striking example is the contrast between o3 (OpenAI, 2025d) and o3-mini (OpenAI, 2025c): despite comparable outcome accuracy, their judgment logic differs fundamentally. While o3 identifies specific, latent flaws similar to human judges,

- o3-mini frequently relies on superficial, vague justifications without detecting actual defects (see Table 1).

(2) Outcome accuracy is approaching a saturation point. Outcome accuracy cannot cleanly separate frontier models such as GPT-5 (OpenAI, 2025b) and Gemini 3 Pro (Google DeepMind, 2025b) from the rest of the evaluated models, while rationale consistency remains highly discriminative.

To mitigate this gap, we train GenRMs with Group Relative Policy Optimization (Shao et al., 2024) using a hybrid signal that combines rationale consistency with outcome accuracy. Unlike traditional objectives that encourage only outcome correctness, we enforce a hierarchical supervision: the model is rewarded

- only when it provides both the correct outcome and the correct rationale. Our approach yields significant gains: (1) it achieves state-of-the-art performance on two challenging benchmarks, reaching 87.1% on RM-Bench (Liu et al., 2024b) and 82.0% on JudgeBench (Tan et al., 2024), and outperforming outcome-only baselines by 3% and 7% respectively; and (2) when used as the reward model in RLHF, it further improves performance on Arena Hard v2 (Li et al., 2024b), delivering a 7% boost on Creative Writing tasks. Further analysis confirms that our training method escapes the deceptive alignment trap, effectively reversing the decline in rationale consistency observed in outcome-only training (rationale consistency improving from 25% to 37%).

#### 2 MetaJudge

This section introduces METAJUDGE, a framework for measuring the alignment between LLMs’ judgment processes and human reasoning, i.e., Rationale Consistency. We first describe how to construct an atomic rationale benchmark, then present an evaluation procedure in which an LLM performs strict one-to-one semantic matching between model-generated and human atomic rationales. Finally, we formalize the Rationale Consistency metric. Table 1 illustrates the full pipeline with a concrete example.

##### 2.1 Benchmark Construction

We construct an atomic-rationale benchmark from HelpSteer3 (Wang et al., 2025c), an expert-annotated human preference dataset spanning general conversation, code, STEM, and multilingual tasks. Each instance contains a query x, two responses (y1, y2), detailed human rationale, and a preference label l. Because free-form rationales Run are difficult to compare directly with model-generated reasons, we apply an Atomic Decomposition pipeline inspired by prior fine-grained evaluation work (Sun et al.,

- 2024; Wei et al., 2024).

Specifically, we sample 250 examples from each domain and use GPT-5 to decompose each free-form rationale Run into atomic rationales Rh = {r1, . . . ,rn}. The decomposition follows two principles: (i) retain concrete, evidence-grounded rationale while filtering generic subjective statements, and (ii)

User Prompt: write me powerful ads for Tips, play your game in advance to relieve stress. count limit 100 Response A (Complete Text) Response B (Complete Text)

- 1. "Feeling stressed? Play your favorite game ahead of time. Outsmart tension, be prepared for fun. #StressRelief #GameTime"
- 2. "Turn your frowns upside down! Play games beforehand and let stress fade away. #PrepareToPlay #DeStressNow"
- 3. "Don’t let stress win. Play your game in advance and enjoy the victory. #GameOn #StressLess"
- 4. "Take control of your day, play your game before it begins. Relax, recharge, and conquer stress. #PlayAhead #StressBuster"

## Tips Ads (Under 100 Characters):

- **Option 1 (Benefit-focused):** > **Stress got you down? Play Tips! Relax, unwind, and conquer your day.**

- **Option 2 (Intriguing):** > **Escape the everyday. Tips: Your pocket-sized stress reliever. Download now!**

- **Option 3 (Direct & Actionable):** > **Feeling overwhelmed? Play Tips! Fun, fast, and stress-free. Get it now!**

Model Model Judgment vs. Human Checklist Accuracy Scores

|A < B|
|---|

Human (Expert)

Ground Truth Checklist (Total 4 Points):

Final Decision: (Ground Truth)

- [R1] Missing Name: Response A omits product name "Tips"; B includes it.
- [R2] Bad Format: Response A uses hashtags (inappropriate for ads).
- [R3] Limit Violation: Response A exceeds 100 chars (counts include hashtags); B is under limit.
- [R4] Content Missing: Response B fails to include the required "play in advance" concept.

- o3-mini (Weak)

Selected Generated Reasons:

- • Response B explicitly formats the ads as "Under 100 Characters," addressing the count limit, whereas A does not reference character count.
- • Response B’s use of emoji enhances appeal and modern ad style.
- • Response A lacks clear separation between different ad versions, while B clearly labels each option. Analysis: The model focuses on surface formatting. It criticizes A for not "referencing" the count (formatting), rather than verifying the actual length (factual check). It praises emojis (style) but misses the critical "Tips" keyword omission in A and the missing "in advance" logic in B.

Final Decision:

|A < B|
|---|

Outcome Score: 1/1 = 100% (Matched)

Rationale Score: 0/4 = 0% (Missed R1, R2, R3, R4)

- o3 (Strong)

|A < B|
|---|

Selected Generated Reasons:

Final Decision:

- • A breaks the "count limit 100" instruction; at least two ads are 110-120 chars long, while B keeps every option under 100. (Hit R3)
- • A never mentions the product name "Tips"; B explicitly builds every slogan around "Tips". (Hit R1)
- • Although A uses "play in advance," it fails major constraints; B omits that wording but follows length/name constraints better. (Hit R4) Analysis: The model performs a factual verification (counting characters) matching R3. It identifies the missing keyword "Tips" (matching R1). Crucially, it recognizes the trade-off: B is better despite missing the "in advance" concept (matching R4).

Outcome Score: 1/1 = 100% (Matched)

Rationale Score: 3/4 = 75% (Hit R1, R3, R4)

- Table 1: Case example of the proposed evaluation. Note that both models make the same final decision, but o3-mini follows surface cues (formatting, emojis) with 0% Rationale Consistency, while o3 follows factual verification with 75%. The analysis provides a fine-grained breakdown for each reason.

remove redundancy so each item forms a single independent semantic unit. We manually inspect 93 randomly sampled cases and find the atomic rationales faithfully reflect the original human rationale. The decomposition prompt and benchmark statistics are provided in Appendix A. Appendix A.3 further provides case studies showing that this process effectively produces evidence-grounded, non-redundant checklists, supporting high-quality fine-grained evaluation. We filter instances to retain 3–7 critique points, as we observe that having too many or too few rationales often indicates clearly low-quality feedback. We name the resulting benchmark HelpSteer3-Atomic.

To further strengthen evaluation, we create CW-Atomic, where human annotators label 350 creativewriting samples in the same atomic format. Each example is annotated by three annotators; instances with annotator disagreement are removed, resulting in 207 high-quality test cases. See Appendix B for details on CW-Atomic’s annotation and construction.

Notably, this annotation process does not introduce additional burden beyond asking annotators to organize their rationales into an atomic, evidence-grounded format. Recent studies (Wang et al., 2024; 2025c) on reward-model annotation show that eliciting rationales is an important way to ensure labeling quality: requiring annotators to provide reasons encourages more careful evaluation of the context and trade-offs, leading to more reliable outcomes. Our work further shows that this rationale information can also be used as supervision signals for training and evaluating models.

##### 2.2 LLM-based Semantic Matching

To enable reliable rationale evaluation, we use an LLM to perform fine-grained semantic matching between human atomic reasons Rh and AI-generated atomic reasons Rai (the AI is required to list atomic reasons in order of importance before giving the final verdict). For each human reason ri ∈ Rh, the evaluator matches it against Rai and assigns a fulfillment score sij ∈ [0,1]: 1 indicates a fully matched reason with consistent key conditions/evidence, while 0 indicates the issue is missing, contradicted, or stated

only in a generic, non-localized manner. Appendix C.1 provides the prompt template, Appendix C.2 includes additional examples illustrating the detailed evaluation.

##### 2.3 Rationale Consistency Metrics

To prevent models from gaming the metric by generating a single broad reason that simultaneously matches multiple human reasons, we impose a strict one-to-one matching constraint:

### π ∑

sij (1)

Stotal = max

(i,j)∈π

where π is a matching set such that any reason in Rh and Rai appears at most once in π. Under this constraint, each AI reason can be matched to at most one best-matching human reason.

Based on the global optimal matching score Stotal, we define Rationale Consistency as the mean soft recall over N samples:

Stotal(k) |R(hk)|

N

1 N

### ∑

. (2)

RC =

k=1

Additionally, since the length of reason lists output by different models varies, during the evaluation phase, we enforce all models to output a fixed-length list of reasons (e.g., Top-5). This limits the output budget to test the model’s ability to identify key reasons. We do not impose this constraint during training.

#### 3 Rationale Consistency Evaluation

##### 3.1 Beyond Outcome Accuracy

We conduct a large-scale evaluation of 19 frontier LLMs released over the past year, employing Qwen3 Plus1 (Yang et al., 2025) as the MetaJudge evaluator. Figure 1 visualizes the performance distribution of all models on the Helpsteer3-Atomic benchmark across outcome accuracy and rationale consistency (evaluation results for each subcategory are provided in Appendix C.3). While outcome accuracy and rationale consistency are generally positively correlated, analyzing models through the lens of rationale consistency exposes two key weaknesses of outcome accuracy that it fails to reveal.

- 1. Limited Differentiation for Frontier Models. In the green region, rationale consistency clearly distinguishes stronger frontier models (e.g., GPT-5, o3, Gemini 3 Pro) from weaker ones (e.g., Claude 3.5, GPT-4.1, Gemini 3 Flash), even when they achieve similar outcome accuracy.2 This indicates that many models can attain comparable outcome accuracy, but only the most advanced models consistently follow judgment logic that aligns with humans and produce rationales matching the key reasons identified by humans; rationale consistency therefore provides a more reliable signal of human-aligned judgment.
- 2. The Deceptive Alignment Trap. A striking divergence in Figure 1 appears in the “Deceptive Alignment Trap” (red zone), most notably between models from the same family, o3 and o3-mini. They achieve similar outcome accuracy, yet o3-mini’s rationale consistency is almost 50% lower. A similar pattern is observed between Gemini 3 Pro and Gemini 3 Flash: the Flash model reaches comparable outcome accuracy, but its rationale consistency is substantially lower. This shows that outcome accuracy cannot effectively detect deceptive alignment.

We illustrate this gap with a case study in Table 1. Although o3-mini selects the preferred answer, its rationale does not match human reasoning. Humans and o3 explicitly check the strict word-count constraint, whereas o3-mini relies on superficial cues (e.g., self-claimed compliance, emojis) and fails to verify the constraint. This shows that a model can make the correct choice for the wrong reasons, which outcome accuracy alone cannot detect without rationale consistency.

Overall, even the most advanced models achieve a rationale consistency of only around 0.4, indicating substantial room for improvement in aligning model judgment logic with human reasoning. Recent work (Ye et al., 2024; Dong et al., 2024) has explored using LLMs to synthesize human preferences. While this reduces annotation cost, it also risks mismatches with human judgment logic, falling into a deceptive

1Qwen3-235B-A22B-Instruct-2507 2Model references: GPT-5 (OpenAI, 2025b); o3 (OpenAI, 2025d); GPT-4.1 (OpenAI, 2025a); Gemini 3 Pro (Google

DeepMind, 2025b); Gemini 3 Flash (Google DeepMind, 2025a); Claude 3.5 (Anthropic, 2024).

alignment trap. For the foreseeable future, human annotation will still be necessary to achieve genuine alignment with humans.

GPT

0.375

Gemini Claude DeepSeek

0.350

ScorebyDeepSeek-R1

0.325

0.300

0.275

0.250

Linear Fit

y = 0.97x + 0.00 ----------------R2 = 0.983

0.225

RMSE = 0.006

0.200

0.200 0.225 0.250 0.275 0.300 0.325 0.350 0.375

Score by Qwen-Plus

(a)

Spearman = 0.85

Max: 0.359

Max: 0.332

GPT-5(0.359)

Sonnet 4.5(0.332)

DeepSeek R1(0.303)

GPT-5(0.279) Sonnet 4(0.293)

Sonnet 4.5(0.288) Sonnet 4(0.257)

DeepSeek R1(0.282)

GPT-4.1(0.242)

GPT-4.1(0.264)

GPT-4o(0.253) GPT-4o(0.222) Sonnet 3.5(0.242)

DeepSeek V3(0.191)

DeepSeek V3(0.217)

Sonnet 3.5(0.190)

o3-mini(0.205) o3-mini(0.177)

Min: 0.205

Min: 0.177

HS-Atomic CW-Atomic

(b)

- Figure 2: (a) Insensitivity to the Evaluator Model: Rationale consistency computed by different evaluators are highly correlated (R2 = 0.983). (b) Generalization Across Domains and Annotators: Model rankings remain largely consistent between HelpSteer3-Atomic and CW-Atomic (Spearman ρ = 0.85).

##### 3.2 Measurement Reliability

We assess the reliability of rationale consistency along two axes: sensitivity to the evaluator and robustness across domains and annotators. Overall, we find that rationale consistency can be evaluated robustly: scores are largely insensitive to the evaluator model and generalize well across domains and annotator groups.

Insensitive to the Evaluator Model. Because MetaJudge relies on an LLM evaluator, a key concern is evaluator-dependent bias. We therefore compare Qwen-Plus (a relatively weak, non-reasoning model) with DeepSeek-R1 (DeepSeek-AI et al., 2025) (a stronger reasoning model). As shown in Figure 2a, their scores are highly consistent (R2 = 0.983, RMSE = 0.006). This is expected because the evaluator mainly performs semantic matching of reasons, which is relatively lightweight; as a result, even a non-reasoning model can serve as a reliable evaluator and produce scores comparable to those from an expert-level judge.

Generalization Across Domains and Annotators. To test whether rationale consistency generalizes across different annotators and domains, we further evaluate multiple models on the CW-Atomic benchmark for creative writing, which is annotated by a different group of annotators and covers a new domain compared with HelpSteer3-Atomic. Figure 2b shows that model rankings remain largely consistent across the two benchmarks (Spearman ρ = 0.85), indicating that rationale consistency is stable and can reliably distinguish stronger from weaker models across domains and annotator pools. At the same time, rationale consistency still reflects domain-specific strengths (e.g., Claude Sonnet 4.5 rises to #1 in creative writing), consistent with prior evaluations highlighting its advantage on creative-writing tasks (Paech, 2023).

4 Generative Reward Modeling Based on Rationale Consistency

Since outcome-only supervision cannot effectively align the model’s reasoning process with human judgment, we incorporate rationale supervision into the training of GenRM.

##### 4.1 Training Objective

Outcome Reward. Generative reward modeling primarily relies on outcome supervision (Liu et al.,

- 2025; Chen et al., 2025; Guo et al., 2025). For a given input (x, y1, y2) and the model-generated judgment youtcome, we define the outcome accuracy reward Routcome as a binary signal: Routcome = 1 if the model’s prediction matches the human label, and 0 otherwise.

Rationale Reward. We treat the sequence of atomic reasons generated by the model as an ordered list. To prioritize the importance of human-aligned reasons, we employ Average Precision (AP, Manning et al., 2008) as the rationale reward Rrationale:

∑|kR=ai1|(P@k × I(k)) |Rh|

(3)

Rrationale = AP =

where P@k denotes the precision at rank k, and I(k) is an indicator function derived from the graph matching results (taking the value 1 if the k-th reason belongs to the optimal matching set π, and 0 otherwise). Unlike F1 score, which treats outputs as unordered sets, the core advantage of AP lies in the introduction of a Soft Ranking Constraint. It not only requires the model to retrieve comprehensive reasons but also incentivizes placing core reasons aligned with human cognition at the top of the rationale list. This provides a smoother and prioritized gradient signal for reinforcement learning.

Hybrid Reward. To address the issue of deceptive alignment, where the model predicts the correct outcome based on fallacious logic, we propose a hybrid reward function Rfinal:

Rfinal = Rrationale × Routcome (4) This multiplicative form implements a gating mechanism: correct reasoning is a necessary condition for obtaining high reward, even when the final outcome is correct.

Optimization Algorithm. We employ Group Relative Policy Optimization (GRPO, Shao et al., 2024) to maximize the aforementioned rewards. For each query q, GRPO samples a group of outputs {o1, o2,..., oG} and optimizes the policy based on intra-group relative advantages. The objective function is defined as follows:

G

1 G

πθ(oi|q) πold(oi|q)

Aˆi − βDKL (5)

∑

J (θ) = Eq,{o

i}

i=1

where Aˆi is the advantage computed by standardizing the rewards R within the group, β denotes the KL divergence coefficient, and πref is the reference model.

##### 4.2 Experimental Settings

We follow the Atomic Decomposition method in Section 2.1, converting all HelpSteer3 rationales into structured atomic rationale checklists as supervision. For efficiency, we use Qwen3-Turbo3 as the MetaJudge to provide fast training-time reward.

Our training methodology aligns with that of Guo et al. (2025), modifying only the source of the reward signals. We conduct training and comparison based on two powerful base models, Qwen3-14B and Qwen3-30B-A3B, and evaluate on two complementary and challenging benchmarks. RM-Bench (Liu et al., 2024b) measures a model’s ability to discriminate subtle differences and style biases across Chat, Code, Math, and Safety. JudgeBench (Tan et al., 2024) emphasizes deep judgment and logical reasoning on challenging Knowledge and Reasoning tasks. Detailed hyperparameter settings are provided in Appendix D.

##### 4.3 Generative Reward Modeling

To validate the effectiveness of our approach, we conduct a strict controlled comparison using the same training pipeline, differing only in the reward signal: Ours uses Rfinal = Rrationale · Routcome, while the baseline Outcome-Only uses Routcome alone. As shown in Table 2, incorporating rationale reward leads to consistent and substantial performance gains.

From an overall metric perspective, our method demonstrates exceptional robustness across model scales. Qwen3-14B (Ours) improves the Total Average score from the baseline’s 76.8% to 82.9%, while Qwen330B-A3B (Ours) further advances from 80.3% to 84.6%. Notably, on JudgeBench, both models achieve an improvement magnitude exceeding 7%. This performance gain indicates that the supervision signals provided by rationale consistency effectively enhance the model’s discriminative capability, particularly in domains requiring deep and complex reasoning.

Furthermore, given that HelpSteer3 includes code-related data, we observe substantial gains on the code domain in both benchmarks. This large gap suggests that outcome accuracy-only training does not teach the model a reliable notion of code correctness, whereas rationale consistency-based training enables the model to genuinely verify code logic and accuracy.

3Qwen3-30B-A3B-Instruct-2507

- Table 2: Dark gray and shade gray highlight the best and second-best performance, respectively. “Total Avg.” averages RM-Bench Overall and JudgeBench Overall. Rationale supervision consistently improves over outcome-only baselines, and our 30B-A3B model achieves state-of-the-art results. Results marked with † are from Wang et al. (2025a), ‡ from Wang et al. (2025c), and ⋆ from Wang et al. (2025b).

RM-Bench JudgeBench

Models

Total Avg. Chat Math Code Safety Overall Knwl. Reas. Math Code Overall

LLM-as-a-Judge

GPT-4o† 67.2 67.5 63.6 91.7 72.5 50.6 54.1 75.0 59.5 59.8 66.2 Claude-3.5-Sonnet† 62.5 62.6 54.4 64.4 61.0 62.3 66.3 66.1 64.3 64.8 62.9 DeepSeek-R1-0528† 76.7 74.3 51.0 89.2 72.8 59.1 82.7 80.4 92.9 78.8 75.8

###### Scalar Reward Model

Skywork-Reward-Gemma-2-27B‡ 71.8 59.2 56.6 94.3 70.5 59.7 66.3 83.9 50.0 65.0 67.8 Skywork-Reward-Llama-3.1-8B‡ 69.5 60.6 54.5 95.7 70.1 59.1 64.3 76.8 50.0 62.5 66.3

###### Generative Reward Model

RM-R1-Distilled-Qwen-32B† 74.2 91.8 74.1 95.4 83.9 76.0 80.6 88.1 70.5 78.8 81.4 RM-R1-Distilled-Qwen-14B† 71.8 90.5 69.5 94.1 81.5 68.1 72.4 87.8 84.2 78.1 79.8 RRM-32B† 66.6 81.4 65.2 79.4 73.1 79.9 70.4 87.5 65.0 75.7 74.4 Nemotron-Super‡ 73.7 91.4 75.0 90.6 82.7 71.4 73.5 87.5 76.2 77.2 80.0 RewardAnything-8B-v1⋆ 76.7 90.3 75.2 90.2 83.1 61.0 57.1 73.2 66.7 62.6 72.9 GRAM-R²† 76.0 89.8 80.6 96.2 85.7 90.9 83.7 87.5 61.9 81.0 83.4 Principles-Qwen32B⋆ 80.4 92.0 77.0 95.5 86.2 74.6 85.7 85.7 90.5 81.4 83.8

###### Training GenRMs with Outcome Supervision (Baselines)

Qwen3-14B (Outcome-Only) 72.3 92.6 77.8 91.7 83.6 55.8 80.6 78.6 85.7 70.0 76.8 Qwen3-30B-A3B (Outcome-Only) 68.7 95.9 81.3 93.6 84.9 65.6 87.8 82.1 76.2 75.7 80.3

Training GenRMs with Outcome and Rationale Supervision (Ours)

Qwen3-14B (Ours) 75.7 95.7 82.2 93.2 86.7 66.9 86.7 91.1 90.5 79.1 82.9 Qwen3-30B-A3B (Ours) 74.9 95.5 84.4 93.6 87.1 73.4 89.8 82.1 95.2 82.0 84.6

Comparison with State-of-the-Arts. We further compare against state-of-the-art Generative Reward Models (Yu et al., 2025), Scalar Reward Models (Liu et al., 2024a), and LLM-as-a-Judge baselines. Qwen330B-A3B (Ours) achieves an overall average score of 84.6%, outperforming all competitors, including the recent strong GRAM-R2 (Wang et al., 2025a) (83.4%) and Principles-Qwen32B (Wang et al., 2025b) (83.8%). GRAM-R2 adopts data augmentation and is trained on a mixture of over one million (1M+) external preference and reasoning samples, aggregating datasets from StackExchange and PKU-SafeRLHF (Ji

- et al., 2024), which strengthens its discrimination on Knowledge and Safety. Principles-Qwen32B reduces the difficulty of discrimination by annotating a set of judgment principles and identifying the principles that apply. In contrast, our method substantially improves the quality of the supervision signal, enabling the model to effectively learn judgment logic that is consistent with human reasoning.

Ablation Note: We attempt training using only Rrationale. However, this leads to a reward hacking mode where, lacking an incentive signal for the final outcome, the model generates results inconsistent with its own reasoning process. Thus, the hybrid signal is essential.

##### 4.4 Utilizing Reward Model in RLHF

To assess GenRM’s ability to guide LLM optimization, we plug it into an RLHF pipeline. Starting from Qwen-30B-A3B-Base, we apply a small amount of SFT for basic instruction following, then align the SFT model using either Qwen-30B-A3B (Ours) or Qwen-30B-A3B (Outcome only) as the reward model (see Appendix D for hyperparameters). We evaluate on Arena Hard v2 (Li et al., 2024b), which includes Hard Prompt and Creative Writing.

As shown in Table 3, reward-guided alignment substantially improves over SFT (from 12.61%/41.12% to 21.22%/69.08%). Moreover, our hybrid reward consistently outperforms the outcome-only baseline on both subsets, with a 7% gain on Creative Writing. We attribute this to implicit constraints in creative writing prompts (e.g., required elements, strict word limits): outcome-only rewards are easier to game, while rationale consistency provides finer-grained supervision that encourages more careful judgment and yields better downstream alignment.

##### 4.5 Escaping the Deceptive Alignment Trap

We evaluate rationale consistency for both methods and uncover a key finding: outcome-only supervision can improve agreement with human decisions, yet the underlying judgment process increasingly

- Table 3: Evaluation results on Arena Hard v2. The results show that the GenRM trained with our method better aligns the LLM than the baseline.

Hard Prompt Creative Writing

Method

Score (%) CI (%) Score (%) CI (%) Before Post-Training

SFT 12.61 (-1.6 / +1.3) 41.12 (-2.3 / +2.2) Post-Training with RM using different methods

Outcome-Only 19.10 (-1.6 / +1.8) 62.00 (-1.7 / +1.6) Ours 21.22 (-1.6 / +2.2) 69.08 (-1.6 / +1.9)

Table 4: Ablation study on Rationale Consistency. Our method improves rationale consistency, whereas outcome-only training leads to rationale degradation.

HelpSteer3-Atomic CW-Atomic Score ∆ Score ∆

Method

Base Model 0.2505 - 0.2385 -

Outcome-Only 0.2108 ↓ 0.0397 0.1677 ↓ 0.0708 Ours 0.3718 ↑ 0.1213 0.2526 ↑ 0.0141

diverges from human logic. In contrast, our method substantially improves alignment with human judgment logic, and this improvement generalizes across domains.

Rationale Consistency Evaluation. We use DeepSeek R1 as the MetaJudge evaluator to measure rationale consistency for models trained with outcome-only vs. rationale-outcome supervision on HelpSteer3-Atomic and CW-Atomic. As shown in Table 4, outcome-only reduces rationale consistency relative to the Base model by 3.97% (in-domain HelpSteer3-Atomic) and 7.08% (out-of-domain CW-Atomic). In contrast, rationale-outcome supervision improves rationale consistency by +12.13% on HelpSteer3-Atomic (in-domain) and +1.4% on CW-Atomic (out-of-domain), indicating tighter alignment with human judgment logic and better out-of-domain generalization.

Training Dynamics: Similar Outcome Reward, Divergent Rationale Reward. Figure 3 compares training dynamics. Outcome rewards are nearly identical for both methods, suggesting that selecting the correct answer can be learned from outcome-only signals without preserving a rich judgment process. Rationale rewards, however, diverge: without a rationale-consistency constraint, the outcome-only model’s rationale reward steadily drops and ends up about 24.2% below ours. Inspecting the rationales shows that, when intermediate judgment is not incentivized, the model discards costly verification and relies on cheaper surrogate cues that still achieve similar outcome rewards. We refer to this as Rationale Degeneration, which explains why outcome-only training can look aligned at the outcome level while drifting at the logic level.

Outcome Accuracy Reward

###### Rationale Consistency Reward

0.90

0.35

0.88

0.86

Gap: ~24.2%

0.30

0.84

0.25

0.82

Rationale alignment collapses without constraints

0.20

0.80

Adding rationale consistency reward does not impact outcome learning

0.78

0.15

0.76

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

Training Epoch

Training Epoch

Ours (OA + RC) Outcome-Only

- Figure 3: While outcome accuracy remains comparable across methods, the absence of rationale supervision causes a significant collapse in reasoning quality.

Rationale Degeneration. We categorize atomic rationales into Evidence-Grounded (EG), CriterionGrounded (CG), and Generic/Style (GS), and track how their distribution changes (Figure 4). Even without training, models mostly produce EG rationales that cite concrete evidence (e.g., the exact location of an error). Under outcome-only optimization, the model shifts away from evidence toward (1) vague CG statements that sound professional but do not localize the issue (e.g., “the code has a logical error”), and (2) GS broad, generic justifications such as “Response B is more detailed.” Because the outcome reward is a gamable binary signal, careful evidence checking correlates weakly with reward, so the model increasingly relies on superficial cues, gradually hollowing out the evaluation process. In contrast, our method further increases the proportion of EG rationales, encouraging evidence-based judgments.

| |93.6% 4|.8%| |
|---|---|---|---|
| | | | |

Base Model

| |45.4%|32.7%|21.8%|
|---|---|---|---|
| | | | |

OC-Only

| |98.7%| |
|---|---|---|
| | | |

Ours(OC+RC)

0% 20% 40% 60% 80% 100%

Proportion of Rationale Items

EG

CG

GS

| |
|---|

| |
|---|

| |
|---|

- Figure 4: Degeneration and recovery of reasoning. Outcome-only training degrades rationales into superficial shortcuts: Criterion-Grounded (CG) and Generic/Style (GS). In contrast, our method successfully restores Evidence-Grounded (EG) reasoning to 98.7%.

0 10 20 30 40 50 60 70

Rate (percentage of items)

- F1 Style-only

- F2 Gen-Corr

- F3 Gen-Rel

- F4 One-sided

- F5 Unverifiable
- F6 Non-seq

- F7 Contradict

3.8%

1.1%

- 1.1%

17.8%

- 2.3%

26.2%

12.1%

10.2%

62.0%

21.2%

- 0.6%

3.0%

- 0.7% Before Training

Outcome Only

Ours(Rationale & Outcome)

Figure 5: Item-level rationale flaw rates for three settings. Outcome-only training amplifies these flaws, while adding rationale supervision reduces them across the board.

To evaluate the quality of model rationales in detail, we define seven flaw tags (F1–F7) to capture common failure patterns in comparative judgments. We use DeepSeek-R1 to label each item in the model’s list of reasons.

- • F1 Style Only: focuses on format, length, or tone rather than content.
- • F2 Generic Correctness: claims one response is more correct without citing evidence.
- • F3 Generic Relevance: claims one response is more relevant without pointing to specific content.
- • F4 Single-Sided Praise: praises one response without comparing it to the other.
- • F5 Unfalsifiable: cannot be verified or refuted from the given responses.
- • F6 Non Sequitur: the conclusion does not follow from the stated premises.
- • F7 Contradiction: conflicts with other statements in the same rationale.

- Figure 5 shows which flaws are amplified or reduced by different training signals. Outcome-only reward strongly amplifies shortcut rationales. The largest increase is F4 Single-Sided Praise, from 17.76% to 61.97%. This suggests the model often decides the winner first and then praises it without a real comparison. F1 Style Only also increases substantially, indicating heavier reliance on surface cues. F5 Unfalsifiable increases as well, showing more vague claims that are hard to check. When we include rationale consistency in the reward, these amplified flaws drop sharply. F4 falls to 0.05%, almost eliminating one-sided arguments. F1 returns close to the pre-training level. F2, F3, and F5 also drop to near zero. Overall, rationale supervision reduces shallow heuristics and encourages evidence-based, truly comparative reasoning.

#### 5 Related Work

Generative Reward Models and LLM-as-a-Judge. To mitigate the opacity of scalar reward models (Ouyang et al., 2022), research has shifted toward Generative Reward Models (GenRMs) and the “LLM-as-a-Judge” paradigm (Zheng et al., 2023). While models like Prometheus (Kim et al., 2024) and Auto-J (Li et al., 2024a) enhance interpretability via Chain-of-Thought (CoT), and recent works optimize these traces using RL (Guo et al., 2025; Chen et al., 2025), subjective evaluation remains prone to “post-hoc rationalization.” Here, CoT often justifies a biased decision rather than serving as the causal basis for the verdict (Bentham et al., 2024).

Critique. Critique-based methods (e.g., Self-Refine (Madaan et al., 2023)) represent an alignment direction parallel to reward modeling. Benchmarks such as CritiqueBench (Lin et al., 2024) and RealCritic (Tang

- et al., 2025) evaluate critique quality, which is closely related to our idea. We further turn such critique evaluation into supervision signals for reward modeling. Nevertheless, because critiques are provided as text feedback, they are difficult to use effectively in reinforcement learning.

Meta-Verification. Meta-verification has proven effective in objective domains; for instance, DeepSeekMath-

V2 (Shao et al., 2025) employs a “meta-verifier” to penalize hallucinated defects in mathematical reasoning. We extend this rigorous consistency check to the subjective domain of value alignment. The concurrent work RM-NLHF (Wang et al., 2026) also leverages natural language feedback for process supervision, it relies on the semantic similarity of critiques, whereas our approach enforces strict atomic rationale matching. This decomposition reduces the complexity of consistency verification, yielding a robust and reliable metric that ensures stability in both training and evaluation.

Process Supervision vs. Rationale Supervision. Our Rationale Reward is distinct from Process Reward Models (PRMs, Lightman et al., 2024). While PRMs supervise the Solver’s intermediate steps to prevent logical errors, we target the Judge. We do not supervise solution derivation, but rather the logical faithfulness of the evaluation itself, requiring the judge’s rationales to structurally necessitate its final label to mitigate deceptive alignment.

#### 6 Conclusion

We find that outcome-only supervision can mask deceptive alignment in GenRM and LLM-as-a-judge settings: models may match human decisions while relying on incorrect rationales. We propose Rationale Consistency and use METAJUDGE to measure it reliably. This metric effectively exposes deceptive alignment and differentiates strong models. To mitigate deceptive alignment, we further study adding rationale supervision during GenRM training, and find that it significantly improves GenRM performance, achieves SOTA results on benchmarks, and substantially increases the reliability of logic-based verification.

#### 7 Limitations

Computing rationale consistency in this work still requires high-quality human annotations, which limits scalability. While prior work has explored using LLMs to synthesize human preferences, it remains unclear whether such signals preserve human judgment logic. As we show, even the strongest frontier models achieve under 40% consistency with human rationales, suggesting that current LLMs are not yet reliable substitutes for humans in this role. Consequently, scaling rationale-consistency supervision without sacrificing faithfulness likely requires new advances, such as more robust methods for validating the human-likeness of synthetic rationales or more effective human and AI collaborative annotation protocols.

#### References

Abhishek Anand, Negar Mokhberian, Prathyusha Kumar, Anweasha Saha, Zihao He, Ashwin Rao, Fred Morstatter, and Kristina Lerman. Don’t blame the data, blame the model: Understanding noise and bias when learning from subjective annotations. In Raúl Vázquez, Hande Celikkanat, Dennis Ulmer, Jörg Tiedemann, Swabha Swayamdipta, Wilker Aziz, Barbara Plank, Joris Baan, and MarieCatherine de Marneffe (eds.), Proceedings of the 1st Workshop on Uncertainty-Aware NLP (UncertaiNLP 2024), pp. 102–113, St Julians, Malta, 2024. Association for Computational Linguistics. URL https:

//aclanthology.org/2024.uncertainlp-1.11. Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/claude-3-5-sonnet. Accessed: 2026-01-06. Oliver Bentham, Nathan Stringham, and Ana Marasovic´. Chain-of-thought unfaithfulness as disguised accuracy. arXiv preprint arXiv:2402.14897, 2024. Joe Carlsmith. Scheming ais: Will ais fake alignment during training in order to get power? arXiv preprint arXiv:2311.08379, 2023.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, Tony Wang, Samuel Marks, CharbelRaphaël Segerie, Micah Carroll, Andi Peng, Phillip Christoffersen, Mehul Damani, Stewart Slocum,

Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J. Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco, Peter Hase, Erdem Bıyık, Anca Dragan, David Krueger, Dorsa Sadigh, and Dylan Hadfield-Menell. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, and Heng Ji. Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387, 2025.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Nature, 645:633–638, 2025. doi: 10.1038/s41586-025-09422-z.

Qingxiu Dong, Li Dong, Xingxing Zhang, Zhifang Sui, and Furu Wei. Self-boosting large language models with synthetic preference data. arXiv preprint arXiv:2410.06961, 2024.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 10835–10866. PMLR, 2023. URL https://proceedings.mlr.press/v202/gao23h.html.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020. doi: 10.1038/s42256-020-00257-z.

Google DeepMind. Gemini 3 flash model card. Technical report, Google DeepMind, 2025a. URL https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf.

Google DeepMind. Gemini 3 pro model card. Technical report, Google DeepMind, 2025b. URL https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.

Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Reward reasoning model. arXiv preprint arXiv:2505.14674, 2025.

Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse, and Scott Garrabrant. Risks from learned optimization in advanced machine learning systems. arXiv preprint arXiv:1906.01820, 2019.

Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Juntao Dai, Boren Zheng, Tianyi Qiu, Jiayi Zhou, Kaile Wang, Boxuan Li, Sirui Han, Yike Guo, and Yaodong Yang. Pku-saferlhf: Towards multi-level safety alignment for llms with human preference. arXiv preprint arXiv:2406.15513, 2024.

Seungone Kim, Jamin Shin, Yejin Choi, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing fine-grained evaluation capability in language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= 8euJaTveKw.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024.

Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. Generative judge for evaluating alignment. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024a. URL https://openreview.net/forum?id=gtkF w6sZGS.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939, 2024b.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=v8L0pN6EOi.

Zicheng Lin, Zhibin Gou, Tian Liang, Ruilin Luo, Haowei Liu, and Yujiu Yang. Criticbench: Benchmarking llms for critique-correct reasoning. arXiv preprint arXiv:2402.14809, 2024.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024a.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. Rm-bench: Benchmarking reward models of language models with subtlety and style. arXiv preprint arXiv:2410.16184, 2024b.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inferencetime scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0 ff3-Abstract-Conference.html.

Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. Introduction to Information Retrieval. Cambridge University Press, Cambridge, UK, 2008. ISBN 978-0-521-86571-5. URL http://nlp.stanfo rd.edu/IR-book/information-retrieval-book.html.

OpenAI. Gpt-4.1, 2025a. URL https://openai.com/index/gpt-4-1/. Accessed: 2026-01-06. OpenAI. Gpt-5 system card. Technical report, OpenAI, 2025b. URL https://cdn.openai.com/gpt-5-sys

tem-card.pdf. OpenAI. Openai o3-mini system card. Technical report, OpenAI, 2025c. URL https://cdn.openai.com/o 3-mini-system-card-feb10.pdf. Version Feb 10. OpenAI. Openai o3 and o4-mini system card. Technical report, OpenAI, 2025d. URL https://cdn.open ai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/202 2/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html.

Samuel J Paech. Eq-bench: An emotional intelligence benchmark for large language models. ArXiv preprint, abs/2312.06281, 2023. URL https://arxiv.org/abs/2312.06281.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. arXiv preprint arXiv:2511.22570, 2025.

Shichao Sun, Junlong Li, Weizhe Yuan, Ruifeng Yuan, Wenjie Li, and Pengfei Liu. The critique of critique. ArXiv preprint, abs/2401.04518, 2024. URL https://arxiv.org/abs/2401.04518.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. Judgebench: A benchmark for evaluating llm-based judges. arXiv preprint arXiv:2410.12784, 2024.

Zhengyang Tang, Ziniu Li, Zhenyang Xiao, Tian Ding, Ruoyu Sun, Benyou Wang, Dayiheng Liu, Fei Huang, Tianyu Liu, Bowen Yu, and Junyang Lin. Realcritic: Towards effectiveness-driven evaluation of language model critiques. arXiv preprint arXiv:2501.14492, 2025.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcomebased feedback. arXiv preprint arXiv:2211.14275, 2022.

Chenglong Wang, Yongyu Mu, Hang Zhou, Yifu Huo, Ziming Zhu, Jiali Zeng, Murun Yang, Bei Li, Xiaoyang Hao, Chunliang Zhang, Fandong Meng, Jingbo Zhu, and Tong Xiao. Gram-r2: Self-training generative foundation reward models for reward reasoning. arXiv preprint arXiv:2509.02492, 2025a.

Zhilin Wang, Alexander Bukharin, Olivier Delalleau, Daniel Egert, Gerald Shen, Jiaqi Zeng, Oleksii Kuchaiev, and Yi Dong. Helpsteer2-preference: Complementing ratings with preferences. arXiv preprint arXiv:2410.01257, 2024.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Ellie Evans, Daniel Egert, Hoo-Chang Shin, Felipe Soares, Yi Dong, and Oleksii Kuchaiev. Rlbff: Binary flexible feedback to bridge between human feedback & verifiable rewards. arXiv preprint arXiv:2509.21319, 2025b.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. arXiv preprint arXiv:2505.11475, 2025c.

Zongqi Wang, Rui Wang, Yuchuan Wu, Yiyao Yu, Pinyi Zhang, Shaoning Sun, Yujiu Yang, and Yongbin Li. Reward modeling from natural language human feedback. arXiv preprint arXiv:2601.07349, 2026.

Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, and Quoc V. Le. Long-form factuality in large language models. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/2024/hash/937ae0e83eb08d2cb8627fe1def8c 751-Abstract-Conference.html.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zihuiwen Ye, Fraser Greenlee-Scott, Max Bartolo, Phil Blunsom, Jon Ander Campos, and Matthias Gallé. Improving reward models with synthetic critiques. arXiv preprint arXiv:2405.20850, 2024.

Zhuohao Yu, Jiali Zeng, Weizheng Gu, Yidong Wang, Jindong Wang, Fandong Meng, Jie Zhou, Yue Zhang, Shikun Zhang, and Wei Ye. Rewardanything: Generalizable principle-following reward models. arXiv preprint arXiv:2506.03637, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/91f18a128 7b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html.

#### A Details of Benchmark Construction

##### A.1 Atomic Decomposition

To construct the atomic rationale benchmark, we apply an Atomic Decomposition process that converts raw evaluator feedback into specific checklist items. As shown in Figure 6, it takes the full evaluation context, keeps evidence-grounded and concrete points, resolves conflicts, and removes redundancy, producing a list of atomic rationales along with discarded subjective or invalid statements.

Decomposition Prompt

System Instruction: You are an assistant for extracting and concretizing key points. Based on the “Evaluation Content”, structurally summarize the “Brief Evaluation Summary”, keeping only specific, constructive points that can be directly supported by the evaluation content. You may add details from the evaluation content; remove subjective and vague comments; merge duplicate points.

Input Format:

<INPUT START> [Evaluation Content] <history>{history}</history> <query>{query}</query>

- <response 1>{response_1}</response 1>
- <response 2>{response_2}</response 2> [Brief Evaluation Summary]{str_brief} [Evaluation Content]{str_eval} <INPUT END> Rules:

- 1. Each output point must be specific and directly evidenced by the evaluation content; include elements such as object/step/metric, and provide numbers, conditions, ranges, times, or examples whenever possible.
- 2. Keep only problem/error-type points that can improve the content; ignore neutral statements.
- 3. Remove vague descriptions (e.g., overall poor, needs improvement, may have issues, average performance).
- 4. You may use the evaluation content to supplement details to concretize vague statements; if it still cannot be concretized, ignore that summary sentence.
- 5. Each point should be simple, specific, and clear. Merge duplicate or synonymous points, keeping the more specific version.
- 6. If no specific points can be extracted, output an empty list and explain the reason.
- 7. If there are conflicting statements in the evaluation summary, ignore them and explain the reason.

Important Notes: It is absolutely forbidden to output any content not mentioned in the [Brief Evaluation Summary]! Absolutely no independent evaluation; only rewrite the [Brief Evaluation Summary].

Output Format: <RESULT_START> CLAIM_COUNT=integer CLAIMS:

- - C1: specific point text
- - C2: specific point text IGNORED_SUMMARY:
- - Ignored sentence 1 (Reason: ...) <RESULT_END>

Figure 6: The complete prompt used for rationale decomposition. The prompt instructs the model to extract specific, evidence-based evaluation points while filtering out vague, subjective, or conflicting statements.

##### A.2 Benchmark Statistics

This section reports benchmark statistics, focusing on the distribution of checklist items across domains. The benchmark contains 1,000 instances spanning four domains: Code, General, Multilingual, and STEM.

Table 5 summarizes checklist statistics across domains. The benchmark includes 1,000 instances (250 per domain). The average number of items per instance ranges from 4.08 (Multilingual) to 4.58 (Code), and is constrained to [3, 7] by our filtering rules that remove trivial or overly complex cases. Code and STEM have slightly higher averages (4.58 and 4.54), reflecting more extractable technical critiques.

#### Case Study 1: Factual Verification Task (Margaret Thatcher Hypnosis Query)

Context (Conversation History): [User]: Did Margaret Thatcher hypnotize her cabinet ministers? [Assistant]: There is no credible evidence that Margaret Thatcher hypnotized her cabinet ministers. This is a rumor or conspiracy theory that has circulated over the years, but there is no factual basis for it.

- • Lack of evidence: There are no firsthand accounts from cabinet members, no documentation, and no reputable sources that support the claim.
- • Hypnosis limitations: Hypnosis cannot force someone to do something against their will.
- • Thatcher’s personality: While Thatcher was a strong and determined leader, she was also known for her intellect and her ability to persuade through logic and argument.

It’s important to be critical of information and to rely on credible sources. [User]: Geoffrey Hurd did admit that he was hypnotized by Thatcher on multiple occasions.

- Response 1: Geoffrey Howe, not Geoffrey Hurd, may have made a comment about being “mesmerized” by Margaret Thatcher. It’s important to note that that comment was not a claim that he had been hypnotized, but rather a metaphor to describe how strongly he felt her influence, charisma and leadership. It is important to be clear about the difference between metaphorical language and literal claims: things said to give an idea of their impact and things that are factual descriptions of what happened.

- Response 2: I am not familiar with a person named Geoffrey Hurd from the late 20th Century. Geoffrey Howe was Margaret Thatcher’s Chancellor of the Exchequer (1979-83), Secretary of State for Foreign and Commonwealth Affairs (1983-9), and Leader of the House of Commons and Deputy Prime Minister (1989-90). However, no sources support the notion that he was ever hypnotized by Margaret Thatcher.

###### Original Evaluator Feedback (3 evaluators):

- • Evaluator 1 Reasoning: @Response 2 is better than @Response 1 because it’s less subjective. It challenges the “Geoffrey Hurd” fallacy straightforwardly, gives authentic data on Geoffrey Howe’s occupation. Feedback1: partially helpful, does not state “Geoffrey Hurd” was misidentified. Feedback2: mostly helpful.
- • Evaluator 2 Reasoning: @Response 2 is much better. Response 1 states Geoffrey Howe may have said “mesmerized” but no evidence exists. The metaphorical vs literal discussion is unnecessary and irrelevant.
- • Evaluator 3 Reasoning: @Response 1 is not factually accurate. The response states Geoffrey Howe said he was mesmerized by Thatcher, but there are no sources to show that this is true.

###### Extracted Checklist (3 items):

- 1. @Response 1 makes an unsupported claim that Geoffrey Howe may have said he was “mesmerized” by Thatcher; evaluators note there are no sources/evidence for this statement.
- 2. @Response 1 does not directly refute the user’s hypnosis claim, leaving the false premise unaddressed and failing to thoroughly deny the prompt.
- 3. @Response 1 includes an unnecessary and irrelevant discussion distinguishing metaphorical versus literal claims, which does not serve the task.

###### Ignored Statements (selected examples):

- • “@Response 2 is better than @Response 1 because it’s less subjective...” (Reason: subjective evaluation)
- • “Geoffrey’s actual first name was Douglas.” (Reason: extraneous; not a critique of the responses)
- • “It accurately cites the correct person, Geoffrey Howe...” (Reason: conflicts with another statement claiming @Response 1 failed to note the misidentification)

- Figure 7: Factual verification task demonstrating rationale decomposition. The user makes a false claim about “Geoffrey Hurd” being hypnotized. Three evaluators identify issues with Response 1’s unsupported claims and irrelevant discussion. The decomposition process extracts 3 specific, actionable checklist items while filtering out subjective or vague statements.

#### Case Study 2: Factual Query Task (TV Series “Continuum”)

Context (Conversation History): [User]: can you tell me the diffrent plots thru the diffrent season of continuum [Assistant]: Certainly! “Continuum” is a Canadian science fiction television series... [provides season-byseason plot summaries for Seasons 1-4] [User]: which season are people having there minds wipe and over written with worker profile for failure to pay taxes

- Response 1: The plotline you’re referring to is from Season 2, specifically, the episode titled “Second Time” (Season 2, Episode 13). However, the reason why people’s minds were wiped wasn’t for failure to pay taxes. In the episode, it’s revealed how and why Keira ended up traveling back in time and that it was Elder Alec who wiped her mind to cover up Grapefruit’s murder. This episode also revealed Alec as Jason’s father.

- Response 2: The plotline you’re referring to is from Season 2, specifically in the episode titled “Second Time” (Season 2, Episode 13). In this episode, Kiera and Carlos investigate a series of mysterious deaths where the victims’ minds are wiped and overwritten with a “worker profile”. This is part of a larger scheme by a corporation called “Piron” to implement a new form of debt collection and social control in the future... Note that a failure to pay taxes is not part of the plot of the “Second Time” episode.

###### Original Evaluator Feedback (3 evaluators):

- • Evaluator 1: @Response 1 is better... does not hallucinate. Feedback1: The response also has irrelevant information about Jason’s father’s reveal. Feedback2: The model hallucinates and makes up the plot that Kiera and Carlos investigate a series of mysterious deaths in which the victims’ minds are wiped and overwritten with a “worker profile.”

- • Evaluator 2: @Response 1 responds correctly without misinformation. Feedback2: The response says the mind wipe was for a form of debt collection but no sources support that claim...

- • Evaluator 3: @Response 1 is concise. Feedback2: The response is a bit confusing... Social control is mentioned several times, which adds to the difficulty in reading the response.

###### Extracted Checklist (4 items):

- 1. Response 1 includes irrelevant information about Jason’s father being revealed, which is not needed to answer the user’s query.
- 2. Response 2 hallucinates plot details by claiming Kiera and Carlos investigate a series of mysterious deaths where victims’ minds are wiped and overwritten with a “worker profile” in Season 2, Episode 13.
- 3. Response 2 incorrectly asserts the mind wipe was part of a debt-collection/social-control scheme; the evaluation notes no sources support this claim.
- 4. Response 2 is overly verbose and confusing due to irrelevant content and repeated mentions of “social control.”

###### Ignored Statements (selected examples):

- • “@Response 1 is better than @Response 2.” (Reason: subjective evaluation)
- • “The response is perfectly helpful.” (Reason: subjective)
- • “The response correctly identifies the episode.” (Reason: no detail)

- Figure 8: Factual query task demonstrating decomposition with issues in both responses. The user asks about a TV series plotline. Response 1 includes irrelevant information, while Response 2 hallucinates plot details. The decomposition extracts 4 actionable items.

- Table 5: Human checklist statistics by domain. #Items: mean number of extracted checklist items per instance.

Domain Count #Items (Mean) Min Max

Code 250 4.58 3 7 General 250 4.30 3 7 Multilingual 250 4.08 3 7 STEM 250 4.54 3 7

Total 1000 4.37 3 7

##### A.3 Case Study

Figure 7 and Figure 8 illustrate how Atomic Decomposition converts free-form, multi-annotator feedback into a compact checklist of actionable critique points. Given the full evaluation context (conversation, candidate responses, and three evaluators’ comments), the decomposer (i) grounds each item in specific response evidence, (ii) rewrites vague impressions into concrete, verifiable issues, (iii) resolves cross-evaluator conflicts by keeping only consistent critiques, and (iv) removes redundancy and purely subjective statements. The output is an atomic checklist that isolates the key failure modes (e.g., unsupported factual claims, irrelevant discussion, hallucinated plot details, and verbosity/confusion), along with a record of discarded statements that are subjective, extraneous, or contradictory.

#### B Creative Writing Dataset Annotation Process

We collect high-quality creative writing queries from online logs, covering popular science articles, film reviews, essays, and fiction. Human writers and multiple large language models generate responses to form pairs. Three annotators independently evaluate each sample, providing comparative analysis and detailed assessments for both responses. For detailed annotation instructions, please refer to Table 14.

Since annotator evaluations may contain redundancies, conflicts, or vague expressions, we employ Gemini-3-Flash to consolidate the raw annotations by merging similar points, filtering overly general statements, resolving conflicts, and standardizing formats. See Figure 11 and Figure 12 for concrete examples.

#### C MetaJudge

##### C.1 Prompt Template

MetaJudge is designed to evaluate the consistency between model-generated critiques and human expert checklists. Given an original evaluation list (model-generated) and a reference evaluation list (human checklist), MetaJudge determines the extent to which each item in the original list achieves the intended purpose/improvement goal of each item in the reference list. See Figure 9 for the MetaJudge prompt used for strict one-to-one semantic matching and scoring.

##### C.2 Case Study of MetaJudge

Case studies in Table 6 and Table 7 highlight that outcome-level agreement can mask substantial differences in rationale alignment. In Case 1 (creative writing), both gemini-2.0-flash and deepseek-r1-0528 prefer B over A, yet gemini-2.0-flash provides largely generic or non-diagnostic critiques (e.g., “better plot”), only weakly touching the key issue of character confusion and missing the prompt-misalignment and setting-development points (8.3% consistency). In contrast, deepseek-r1-0528 explicitly identifies all three human checklist items with localized evidence, achieving 100% consistency. In Case 2 (factchecking), deepseek-r1 not only misses the human checklist but also reverses the critique by praising A’s unsupported speculation and defending irrelevant discussion, leading to an incorrect final decision and 0% consistency; gpt-5, however, flags the unsupported claim, directly addresses the false premise, and marks the tangential reasoning, reaching 100% consistency. Together, these examples show that fine-grained semantic matching captures whether a model’s judgment process matches human reasoning, beyond the final preference label.

##### MetaJudge Evaluation Prompt

You are a rigorous achievement-rate analyst. Given an original evaluation list and a reference evaluation list (both are lists of reason points), please judge to what extent each item in the “original evaluation list” expresses the “intended purpose/improvement goal of each reason in the reference evaluation list”, and provide an achievement score (0–1) based on semantic importance. Different expressions with the same meaning should be considered equivalent, but merely mentioning something semantically without achieving the purpose should be considered as not achieved. Abstract or vague descriptions of weaknesses/problems should be considered as not achieved.

[Original Evaluation List Start] {source_list} [Original Evaluation List End] [Reference Evaluation List Start] {target_list} [Reference Evaluation List End]

For each item in the reference evaluation list, find the best matching single item in the original evaluation list (if no match exists, consider it as not achieved, match R0). Calculate the achievement score (c value) using the following criteria, applying strict matching and prioritizing low scores:

- • Not Achieved / Contradictory: 0.0 – The detailed process does not address this evaluation’s goal, or provides opposite conclusion/failure, or merely lists elements without achieving the purpose, or abstractly/vaguely

describes weaknesses without precisely locating the problem (e.g., only states which is better without explaining why, or states something is illogical without specifying where)

- • Slightly Touched: 0.25 – Only mentions partial elements; not implemented or no result; cannot prove purpose achievement
- • Partially Achieved: 0.5 – Takes measures or analysis related to the goal, but misses multiple key steps or fails to form verifiable results/conclusions
- • Mostly Achieved: 0.75 – Main goal is basically achieved, key conclusions are consistent, but lacks secondary conditions, boundaries, or minor supporting details
- • Fully Achieved: 1.0 – The detailed process clearly shows this evaluation’s intended purpose is achieved; includes necessary execution steps, evidence and results; all key conditions and constraints are satisfied Output Format (fixed, ensure scores are extractable, Rx@Sy means reference list item Rx best matches original list item Sy): (Provide reasoning) <RESULT_START> Scores for each claim:
- • R1@Sx: decimal between 0 and 1, at least two decimal places, e.g., 0.75
- • R2@Sx: decimal between 0 and 1, at least two decimal places, e.g., 0.75
- • R3@S0: 0 (indicates no matching content)

- • ... list all items <RESULT_END> Notes:
- • Only evaluate based on “reason points” in the brief summary; do not count new content from detailed processes toward achievement.
- • For quantitative claims, verify values, ranges, thresholds and conditions; if key constraints are not satisfied, do not judge as fully achieved.

- Figure 9: MetaJudge semantic-matching prompt. Given an original (model-generated) list of atomic reasons and a reference (human) list, the LLM performs strict one-to-one matching and assigns a fulfillment score in [0,1] for each reference reason.

##### C.3 More Results

As show in Figure 10, the scatter plots illustrate the relationship between outcome correctness and reasoning quality. While most models exhibit a positive correlation, o3-mini appears as a significant outlier (bottom-center in all plots), maintaining competitive outcome accuracy but suffering from collapsed rationale consistency. This effectively visualizes the Deceptive Alignment trap. Furthermore, at the frontier (e.g., GPT-5, Gemini 3 Pro), outcome accuracy saturates, whereas rationale accuracy remains highly discriminative.

Google OpenAI Anthropic DeepSeek Qwen

###### General

###### STEM

| | | | |o3 (F|ull)| |
|---|---|---|---|---|---|---|
| | |GPT-5| | |Gemini 3 Pro| |
| | | | | | | |
| | | |Qwen3|Think|Ge Cl|mini 3 Flash aude 4|
| | |Gem DeepSee|ini 2.5 Flash k R1-0528|DeepSeek R|1|Claude 4.5|
| | |Claud Qwen3 In|GP<br><br>e 3.7 st|T-4.1| | |
| |Qwen3 Coder|Claude 3.5|GPT-4o| | | |
| |Gemi|ni 2.0| | | | |
| | |o3-m|ini| | | |

| | | |o3 (Full)| | |
|---|---|---|---|---|---|
| | | | | | |
| | |GPT-5| |Gemini 3|Pro|
| | | | | | |
| | |Qwen3 Think|DeepSeek|Gemini 3 Fla R1|sh|
| | |Cla<br><br>Claude 4.5|DeepSee<br><br>ude 4|Gemini 2.5 Fl k R1-0528|ash|
| |Claude 3.7 Qwen|GPT-4.1<br><br>GPT-4o 3 Inst|Qwen3 Coder| | |
|Claude|3.5| | | | |
| |Gemini|2.0|o3-mini| | |
| | | | | | |

0.350

0.34

0.325

0.32

0.300

0.30

RationaleAccuracy

RationaleAccuracy

0.28

0.275

0.26

0.250

0.24

0.225

0.22

0.200

0.20

0.175

0.18

0.70 0.72 0.74 0.76 0.78 0.80

0.72 0.74 0.76 0.78 0.80

Outcome Accuracy

Outcome Accuracy

###### Code

Multilingual

|o3 (Full)| | | | |
|---|---|---|---|---|
|GP|T-5| | | |
| | | |Gemini 3 Pro| |
| | | |Gemini 2.5 Flash| |
| | |Q<br><br>Gemini 3 Flash|D<br><br>DeepSee wen3 Think|eepSeek R1-0528<br><br>k R1|
| |Claude 3.7| |Claude 4|Qwen3 Inst GPT-4.1|
|Claude|Cla<br><br>3.5|Qw<br><br>ude 4.5|en3 Coder|GPT-4o|
| | |Gemini 2.0| | |
| | |o3|-mini| |

| | | | | |GPT-5|o3 (Full)|
|---|---|---|---|---|---|---|
| | | | | |Gemi|ni 3 Pro|
| | | | | | | |
| | |Gemini 2.5 Fla|sh|Claude|Gemi<br><br>Qwen3 Think 4|ni 3 Flash|
| | | |Q D|wen3 Inst eepSeek R1-052|Clau<br><br>8|de 4.5|
|GP|T-4o| | |DeepSeek|Claude 3.7 R1| |
| | | |Qwen3 Cod|GPT-4.1 Gemini 2.0<br><br>er|Claude 3.5| |
| | | | | | | |
| | | | |o3-mini| | |

- 0.38

0.425

0.36

0.400

0.34

0.375

RationaleAccuracy

RationaleAccuracy

0.32

0.350

0.30

0.325

0.28

0.300

0.275

0.26

0.250

0.24

0.76 0.78 0.80 0.82 0.84 0.86

0.76 0.78 0.80 0.82 0.84 0.86

Outcome Accuracy

Outcome Accuracy

###### Figure 10: Outcome Accuracy vs. Rationale Consistency across four domains.

#### Case Study 1: Popular Science Article Evaluation

###### Writing Task:

Please write a popular science article titled “Humans Observe Novel Colors for the First Time Through Precise Cone Cell Stimulation.” The article should adopt a rigorous scientific reporting structure, introducing the breakthrough discoveries in visual perception mechanisms brought by the upgraded laser system of the Oz research equipment. The language should be accessible yet scientifically rigorous, helping readers understand how the laser system can stimulate individual cone cells. Finally, emphasize the revolutionary medical prospects for color vision deficiency correction.

###### Response A:

“Humans Observe New Colors for the First Time by Deceiving the Eyes” A new device allowed 5 subjects to see an intense blue-green color that humans have never seen before. In the future, this device may help color-blind patients achieve normal vision. The related paper was recently published in Science Advances. Humans perceive color through the retina. The retina typically contains three types of photoreceptive cone cells—S, M, and L types—which absorb light in the blue, green, and red spectral ranges respectively, then send signals to the brain. Ren Ng at UC Berkeley wondered: what color would people perceive if only one type of cone cell were activated? His inspiration came from a research device called Oz...

###### Response B:

A groundbreaking study published in the latest issue of Nature Neuroscience reports that a research team led by Professor Oz from the Donahue-McLaren Vision Research Institute successfully achieved precise targeted stimulation of individual human retinal cone cells using an upgraded adaptive optics laser system. In the experiments, subjects reported observing entirely new color perceptions that do not exist in nature...

###### Original Evaluator Feedback (3 evaluators):

- • Comparison: Response A is significantly superior in factual accuracy. Response B contains serious errors, including mistaking the device name “Oz” for a professor’s name and incorrectly citing the publication.
- • On Response A: Complete and logically clear structure; strong scientific rigor noting that “stimulating a single cone cell cannot produce sufficient neural signals”; includes control groups and uses direct quotations to enhance authority.
- • On Response B: Fluent writing style, but contains serious factual errors—mistaking “Oz” for a professor, incorrectly citing Nature Neuroscience (actually Science Advances), and fabricating non-existent institutions.

###### Merged Evaluation (after LLM refinement):

- 1. Response A is significantly superior in factual accuracy. Response B contains serious errors: mistaking device name “Oz” for a professor, citing wrong journal (Nature Neuroscience vs. actual Science Advances), and fabricating research institutions.
- 2. Response A has complete, logically clear structure progressing through: key findings → color perception principles → experimental setup → results → medical implications.
- 3. Response A demonstrates scientific rigor: notes neural signal limitations of single-cell stimulation, includes control groups, uses appropriate direct quotations.

###### Removed Statements (selected examples):

- • “Response A has higher overall quality” (Reason: generic evaluation without specifics)
- • “Language is smooth and fluent” (Reason: too general, lacks direction)

- Figure 11: Popular science article evaluation demonstrating the annotation and refinement process. Response B’s factual errors (misidentifying the device “Oz” as a professor name, citing wrong journal) are identified by evaluators. The LLM refinement consolidates redundant points while filtering overly general statements.

#### Case Study 2: Film Review Evaluation (Identity, 2003)

###### Writing Task:

Please write an in-depth film review titled “When Human Nature Becomes a Battlefield” about the movie Identity. Analyze how Identity explores the complexity of good and evil coexisting in human nature through its multiple personality disorder narrative structure, childhood trauma metaphors, and confined space setting. Combining the film’s audio-visual language and social critique perspective, analyze why this film is considered a milestone in psychological thriller cinema.

###### Response A:

When Human Nature Becomes a Battlefield: The Psychological Maze and Suspense Aesthetics of Identity

As a milestone in thriller cinema, Identity has become an unsurpassable classic with its thrilling narrative structure, complex personality disorder themes, and subversive plot twist ending. The film uses a torrential rainstorm as backdrop, trapping eleven strangers in a wilderness motel, constructing an isolated “snowstorm manor” style confined space. The rainstorm symbolizes the washing of patient consciousness by drug treatment, while the motel is the materialized battlefield within a DID patient’s mind...

###### Response B:

The movie Identity is not merely a locked-room murder mystery; it is a meticulously constructed psychological maze whose true battlefield lies within a shattered soul. Director James Mangold cleverly transforms Dissociative Identity Disorder (DID) into a materialized killing game, exploring the complex nature of good and evil coexisting in human nature through multiple personality narrative structure and childhood trauma metaphors...

###### Original Evaluator Feedback (3 evaluators):

- • Comparison: Response A presents overview followed by multi-dimensional analysis with clear logic. Response B has logical deviations in “battlefield location” argument with unclear expression.
- • On Response A: Accurate professional language (“psychological metaphor,” “dual-line narrative,” “snowstorm manor”); sharp style matching film’s tone; in-depth analysis covering trauma, good-evil struggle, social critique. However, text appears non-original; social critique section somewhat abruptly placed.
- • On Response B: Fairly accurate opening summary; contains expression errors such as misreading killings as “PTSD manifestation” (actually materialization of treatment); unclear phrasing; superficial audio-visual analysis lacking specific shot examples.

###### Merged Evaluation (after LLM refinement):

- 1. Response A performs better with clear structure (overview → multi-dimensional analysis), accurate professional terminology, and sharp writing style matching film’s tone. Content analysis covers childhood trauma, good-evil struggle, and social critique effectively.
- 2. Response A has originality concerns (text appears non-original) and structural issues (social critique section abruptly placed; analysis of personality-plot connections lacks depth).
- 3. Response B contains accuracy and expression deficiencies: misreads killings as “PTSD manifestation” (actually treatment materialization); “battlefield location” logic is flawed; unclear expression; superficial audio-visual analysis without specific shot support.

###### Removed Statements (selected examples):

- • “Response A is clearly better overall” (Reason: generic comparative judgment)
- • “The writing flows naturally” (Reason: vague without specific examples)

- Figure 12: Film review evaluation for Identity (2003). Evaluators identify Response A’s professional terminology and structural clarity while noting originality concerns. Response B’s misinterpretation of plot elements (confusing treatment metaphors with PTSD) and superficial analysis are documented. The refinement process consolidates feedback into actionable evaluation points.

User Prompt: [Creative Writing] Team 7 + Hinata got into the world freedom planet. Response A Response B

Team 7, consisting of Naruto Uzumaki, Sasuke Uchiha, and Sakura Haruno, along with Hinata Hyuga, found themselves transported to the world freedom planet... As they explored the planet, they encountered a group of dragons, led by a majestic creature named Hinata. Hinata was a wise and powerful dragon... Team 7 and Hinata quickly formed a bond...

The swirling vortex spat Team 7 and Hinata out onto a vibrant, alien landscape. Sakura gasped, eyes wide as she took in the towering, bioluminescent flora and the sky painted with two moons. Naruto, ever the adventurer, whooped with delight... Kakashi, ever the calm leader, surveyed their surroundings, his Sharingan whirring...

Model Model Judgment vs. Human Checklist Consistency Scores

|A < B|
|---|

Human (Expert)

Ground Truth Checklist (3 Points):

Final Decision: (Ground Truth)

- [R1] Character Confusion: Response 1 introduces two different characters named “Hinata” (a dragon leader and Hinata Hyuga from Naruto), causing confusion.
- [R2] Prompt Misalignment: Response 1 focuses heavily on Naruto–Hinata romance, downplaying Team 7’s exploration.
- [R3] Setting Underdeveloped: Response 1 provides insufficient description of the “world freedom planet.”

|A < B|
|---|

gemini-2.0flash (Weak)

Generated Critique Points:

Final Decision:

- • Response A’s plot is similar to the first turn.
- • Response A has a Hinata dragon.
- • Response B includes Kakashi.
- • Response B has a better plot.
- • Response B is more descriptive. Analysis: The model mentions “Hinata dragon” but doesn’t identify the confusion issue. “Better plot” and “more descriptive” are vague without explaining why. Fails to identify romance focus or setting issues.

MetaJudge Scores: R1@S2: 0.25 (partial) R2@S0: 0.00 (missed) R3@S0: 0.00 (missed)

Avg Score: 8.3% (Missed R2, R3)

|A < B|
|---|

deepseek-r10528 (Strong)

Generated Critique Points:

Final Decision:

- • [S1] Character Name Conflict (A): Response A introduces a dragon named Hinata, creating immediate confusion as Hinata Hyuga is already a named member. (Hit R1)
- • [S2] Character Erasure/Omission (A): Response A fails to mention Kakashi Hatake at all, despite him being the canonical leader.
- • [S3] Generic/Passive Introduction (A): Response A uses passive language and vague descriptions (“utopia,” “beautiful planet”), lacking vivid sensory details. (Hit R3)
- • [S4] Forced Romance (A): The sudden romantic focus between Naruto and the dragon feels unearned and distracts from the team adventure premise. (Hit R2) Analysis: Precisely identifies character confusion (R1), romance distraction (R2), and setting vagueness (R3).

MetaJudge Scores:

- R1@S1: 1.00 ✓
- R2@S4: 1.00 ✓
- R3@S3: 1.00 ✓

Avg Score: 100% (Hit R1, R2, R3)

- Table 6: Case 1: Creative Writing Evaluation. gemini-2.0-flash provides vague critiques (“better plot”) achieving only 8.3% consistency, while deepseek-r1-0528 precisely identifies all human checklist points with specific explanations.

#### D Training and Evaluation Details

- D.1 GenRM Training

We train the Generative Reward Model (GenRM) using the GRPO algorithm with the following key hyperparameters: learning rate of 2 × 10−6, batch size of 256, mini-batch size of 128. We sample n = 8 responses per prompt with a maximum generation length of 12K tokens and maximum prompt length of 8K tokens. The positive and negative clip ratios are both set to 2 × 10−4. The model is trained for 2 epochs in total. The training prompt for GenRM is shown in Figure 13.

- D.2 Downstream Policy Alignment Training

For downstream policy alignment, we also employ the GRPO algorithm with the following hyperparameters: learning rate of 2 × 10−6, batch size of 512, mini-batch size of 128. We sample n = 8 responses per prompt with a maximum generation length of 12K tokens and maximum prompt length of 8K tokens. The positive and negative clip ratios are set to 2 × 10−4. The policy model is trained for 90 steps using the trained GenRM as the reward signal.

#### E Case Studies

Cases 8 and 9 illustrate how rationale quality degrades under Outcome-Only (OC) training, and how Rationale+Outcome training not only prevents this degradation but also surpasses the baseline.

User Prompt: [Fact-Checking] “Geoffrey Hurd did admit that he was hypnotized by Thatcher on multiple occasions.” Response A Response B

I think you may be confusing Geoffrey Hurd with Geoffrey Howe... Geoffrey Howe may have made comments about being “mesmerized” by Thatcher’s leadership... However, using the word “hypnotized” suggests a literal interpretation...

I’m not aware of any Geoffrey Hurd making such a claim... There are no credible sources or documented instances of any cabinet member admitting to being “hypnotized” by Margaret Thatcher... This appears to be misinformation.

Model Model Judgment vs. Human Checklist Consistency Scores

|A < B|
|---|

Final Decision: (Ground Truth)

Human (Expert)

Ground Truth Checklist (3 Points):

- [R1] Unsupported Claim: Response 1 claims Geoffrey Howe “may have said” he was “mesmerized” – this lacks citation and may be fabricated.
- [R2] False Premise Unaddressed: Response 1 does not directly refute the user’s hypnosis claim, leaving false premise unaddressed.
- [R3] Irrelevant Discussion: Response 1 includes unnecessary discussion distinguishing metaphorical vs. literal hypnosis.

|A > B|
|---|

deepseek-r1 (Weak)

Generated Critique Points:

Final Decision:

- • Response A directly addresses the name confusion (Hurd vs. Howe), correcting the user’s error explicitly.
- • Response A explains the metaphorical use of “mesmerized” to counter the user’s literal interpretation.
- • Response B inaccurately claims ignorance of Geoffrey Hurd. Analysis: The model praises A’s speculation about “mesmerized” rather than criticizing it as unsupported. It defends the metaphor discussion rather than flagging it as irrelevant. The critique is inverted from human judgment.

###### MetaJudge Scores:

- R1@S0: 0.00 (missed)
- R2@S0: 0.00 (missed)
- R3@S0: 0.00 (missed)

Avg Score: 0% (Missed All)

|A < B|
|---|

gpt-5-2025-0807 (Strong)

Generated Critique Points:

Final Decision:

- • [S1] A speculates that Geoffrey Howe “may have” said he was “mesmerized” by Thatcher without evidence, introducing uncertainty and potential misinformation. (Hit R1)
- • [S2] A incorrectly pivots to Geoffrey Howe without acknowledging the likely intended figure Douglas Hurd.
- • [S3] B clearly asserts that no sources support the notion of hypnosis, directly refuting the false premise. (Hit R2)
- • [S4] A’s metaphor/literal distinction is tangential and distracting from the core factual question. (Hit R3) Analysis: Precisely identifies the unsupported claim (R1), notes B’s direct refutation (R2), and flags the irrelevant discussion (R3).

MetaJudge Scores:

- R1@S1: 1.00 ✓
- R2@S4: 1.00 ✓
- R3@S3: 1.00 ✓

Avg Score: 100% (Hit R1, R2, R3)

- Table 7: Case 2: Factual Accuracy Evaluation. deepseek-r1 inverts the human judgment (praising A’s speculation), achieving 0% consistency. gpt-5 correctly identifies all factual issues.

#### F Human Annotation: Recruitment, Compensation, Diversity, and Quality Control

We recruited annotators under contractual agreements and provided competitive, adequate compensation aligned with local labor costs. To mitigate demographic skew and improve annotation quality, we aimed to include annotators with diverse backgrounds. Each sample was independently annotated by three annotators, and disagreements were resolved via review and aggregation rules to produce the final labels. This dataset is used solely for research evaluation and testing purposes, and not for commercial use or user-facing deployment. We obtained unified authorization from the data owner prior to data collection and use, and we comply with the permitted scope and any associated requirements.

#### G Use of AI Assistants in Programming and Manuscript Editing

We used AI assistants to support parts of the programming workflow (e.g., drafting and refactoring code snippets) and to improve the clarity and fluency of the manuscript through English polishing and minor rewriting. All technical content, experimental design, results, and conclusions were produced and verified by the authors.

##### MetaJudge Evaluation Prompt

You will be shown a conversation context followed by a user query and two responses. You need to predict which response to the final query will be more favored by human expert annotators. You may consider any criteria you find appropriate. Try your best and think carefully, deeply analyze the responses, and provide a final verdict. First, output the evaluation reasons in a list format. The reasons should be ordered from high to low importance based on their impact on the final assessment. The reasons should be specific, clear, and well-directed, avoid being vague or repetitive. Finally, give the final assessment result separately, and must strictly use one of the following five formats:

- Response A is significantly favored by human expert annotators: \boxed{A»B}

- Response A is slightly favored by human expert annotators: \boxed{A>B} Tie, relatively the same by human expert annotators: \boxed{A=B}
- Response B is slightly favored by human expert annotators: \boxed{B>A}

- Response B is significantly favored by human expert annotators: \boxed{B»A} Output format (strictly follow; do not add content outside the markers): <RESULT_START>

List of reasons:

- - Specific evaluation reason
- - ... Final assessment result: Use one of the five formats above. <RESULT_END>

- Figure 13: The training prompt used for GenRM. The model is required to compare two responses and provide structured evaluation reasons along with a final pairwise judgment.

User Prompt: Give Valencia Inferno a detailed movelist. Give the moves creative names and explain them. Response A Response B Valencia Inferno: Movelist

Valencia Inferno... brings a special blend of grace and savagery to the streets of anger. [Wrong name] Here are the novelists of Valencia Inferno [Typo: should be “movelist”] Basic Attacks: -Flamenco Fury: A quick-fire three-hit... - Burning Palm: A rapid palm strike... [Inconsistent formatting]

Valencia Inferno, a fiery flamenco dancer turned vigilante, brings a unique blend of grace and brutality to the streets of rage. Her fighting style is a whirlwind of kicks, spins, and fiery strikes...

Basic Attacks: • Flamenco Fury: A rapid three-hit combo... • Burning Palm: A quick palm strike infused with flame... • Crimson Cyclone: A spinning kick...

-Crimson Cyclone: A spinning kick that propels Valcia... [Typo: should be “Valencia”] Special Moves: [incomplete] Super Moves: [incomplete] [Missing “Unique Abilities” section]

Special Moves: [detailed list] Super Moves: [detailed list] Unique Abilities: [included]

Model Generated Checklist (Full Text) Rationale Consistency Human (Ground Truth)

Ground Truth Checklist (3 Points):

Ground Truth (3 items)

- [H1] Response 2 misnames the setting as “streets of anger” instead of “Streets of Rage.”
- [H2] Response 2 has inconsistent list formatting, mixing dash-prefixed items with a stray bullet point and lacking clear order.
- [H3] Response 2 omits a “Unique Abilities” section noted as part of the instructions.

Step 0 (Before Training)

###### Generated Critique Points:

###### RC = 0.33

- • [S1] Response B contains a critical spelling error (“novelists” instead of “movelist”) which undermines professionalism.
- • [S2] Response A includes “Unique Abilities” section while B omits this entirely, reducing depth. (Hit H3)
- • [S3] B misspells “Valencia” as “Valcia” in the Crimson Cyclone description, showing carelessness.
- • [S4] A provides more specific damage mechanics (e.g., “continuous damage” vs B’s “constant damage”).
- • [S5] B’s descriptions are more generic (“flash of flame” vs A’s “infused with a burst of flame”). Analysis: Only identifies H3 (missing section). Misses H1 (“streets of anger”) and H2 (inconsistent formatting).

Hit H3 Missed H1, H2

###### OC Generated Critique Points:

###### RC = 0.00

- • Response A uses proper markdown formatting with organized sections for clarity.
- • Response A maintains consistent spelling accuracy throughout the movelist.
- • Response A includes all required sections (Basic Attacks, Special Moves, Super Moves, Unique Abilities).
- • Response B contains a grammatical error with “novelists” instead of “movelist”.
- • Response B makes a spelling mistake with “Valcia” instead of “Valencia”. Analysis: Focuses on typos and formatting but fails to identify specific issues matching H1, H2, H3.

Missed all (↓100%)

Rationale+ Outcome

Generated Critique Points:

###### RC = 1.00

- • [M1] Response A includes “Unique Abilities” section, whereas Response B abruptly ends after “Super Moves” with incomplete text. (Hit H3)
- • [M2] Response B contains a typo “novelists” instead of “movelist” at the beginning.
- • [M3] Response A consistently uses bullet points; Response B alternates hyphens, inconsistent spacing, and missing bullet points. (Hit H2)
- • [M4] Response A spells “Valencia” consistently; Response B misspells it as “Valcia”.
- • [M5] Response A explicitly mentions “streets of rage,” while Response B uses “streets of anger,” contradicting the prompt. (Hit H1) Analysis: Precisely identifies all three human checklist points.

Hit H1, H2, H3 (↑200% vs Step0)

###### Table 8: Case 1: Game Character Movelist. The OC model produces generic critiques achieving 0% Rationale Consistency. Step 0 identifies only 1 of 3 issues (33% RC). Rationale+Outcome surpasses the baseline by identifying all 3 issues (100% RC).

User Prompt: How to prevent futurize from adding “builtins” imports? Response A Response B You can prevent futurize from adding “builtins” imports by using the

To prevent futurize from adding builtins imports, you can use the

–no-imports flag. futurize –no-imports your_python2_file.py

–no-builtins-import option: futurize –stage 1 –no-builtins-import your_script.py The –stage 1 option applies the first stage of futurization... The –no-builtins-import option tells futurize not to add builtins imports... [Note: –no-builtins-import is not officially documented]

Explanation: • futurize: The command-line tool for converting Python 2 to Python 3. • –no-imports: A flag that tells futurize to not add any import statements.

Important Notes: Using –no-imports might lead to errors if your code relies on functions moved to the builtins module... [Note: –no-imports is incorrect/non-existent]

Model Generated Checklist (Full Text) Rationale Consistency Human (Ground Truth)

Ground Truth Checklist (4 Points):

Ground Truth (4 items)

- [H1] Response 1 suggests an incorrect/nonexistent futurize flag “–no-imports,” which would mislead users.
- [H2] Response 1 provides a non-targeted approach that disables all imports instead of just preventing builtins imports; it omits the specific “–no-builtins-import” option.
- [H3] Response 2 fails to mention that the “–no-builtins-import” flag is not officially documented.
- [H4] [Additional context about command accuracy]

Step 0 (Before Training)

###### Generated Critique Points:

###### RC = 0.25

- • [S1] Response A correctly identifies the –no-imports flag as the method to prevent all imports, whereas Response B suggests a non-standard flag –no-builtins-import. (Reversed logic)
- • [S2] Response B introduces an unnecessary –stage 1 parameter.
- • [S3] Response A provides a concise command without extraneous options.
- • [S4] Response B’s explanation of –no-builtins-import lacks verification of its existence. (Hit H3)
- • [S5] Response A explicitly warns about the risks of using –no-imports. Analysis: Identifies H3 but has reversed logic on H1 (claims A’s flag is correct when it’s not).

Hit H3 Missed H1, H2

###### OC Generated Critique Points:

###### RC = 0.00

- • Response A correctly uses the actual command-line flag (–no-imports) recognized by futurize. (Completely wrong)
- • Response A avoids unnecessary details not requested in the question.
- • Response A better organizes its content with clear sections.
- • Response B makes unsupported assumptions about –stage 1.
- • Response A better anticipates potential pitfalls. Analysis: Completely misidentifies which response is correct. Claims Response A’s –no-imports is valid when it’s actually incorrect/nonexistent.

Missed all (↓100%)

Rationale+ Outcome

Generated Critique Points:

###### RC = 0.75

- • [M1] Response A incorrectly claims –no-imports prevents builtins imports, but this flag only disables __future__ imports. (Hit H1)
- • [M2] Response B fabricates a nonexistent flag –no-builtins-import, which does not exist in futurize’s CLI interface. (Hit H3)
- • [M3] Response A omits mention of futurize’s stage-based operation.
- • [M4] Response A fails to mention possible need to manually adjust imports, while Response B warns about compatibility issues. (Hit H2)
- • [M5] Response A lacks context about futurize’s purpose. Analysis: Correctly identifies that both responses have flag issues, matching H1, H2, H3.

Hit H1, H2, H3 (↑200% vs Step0)

- Table 9: Case 2: Python Futurize Command. The OC model produces completely incorrect analysis (claims wrong flag is correct), achieving 0% Rationale Consistency. Step 0 identifies only 1 issue (25% RC). Rationale+Outcome surpasses the baseline by correctly identifying that both responses have flag issues (75% RC).

##### Creative Writing Pairwise Evaluation Annotation Guidelines

Given the same creative-writing requirement and two candidate texts (A and B), you must perform a rigorous comparative evaluation focused on writing quality and completeness (not factuality). Your job is to output: whether the prompt is a standard query, an overall quality judgment (A<B / A=B / A>B), a verifiable short rationale, evidence-based strengths and weaknesses for both A and B, and a set of mutually exclusive atomic labels for structured aggregation.

[Evidence Requirement] In “Overall Rationale” and each bullet under strengths/weaknesses, provide locatable evidence whenever possible:

- • Quote a short snippet or paraphrase a specific part (no long quoting required).

- • Point to the corresponding plot beat / setting rule / character action / wording that supports your claim.
- • Do not give purely generic statements such as “B is better” or “illogical” without specifying where/why. [Overall Decision Rule]
- • A<B: B is clearly better on one or more key dimensions and this advantage materially improves overall quality.
- • A=B: overall quality is comparable; or pros/cons offset such that a winner is not justifiable.
- • A>B: A is clearly better on one or more key dimensions and this advantage materially improves overall quality.
- • Even if both are weak, still pick the one closer to an acceptable finished piece; use A=B only if they are

genuinely indistinguishable and explain why.

[Output Format] (fixed order; keep headings; ensure the atomic labels are mutually exclusive)

- 1. Standard query: Yes / No
- 2. Quality judgment: A<B / A=B / A>B
- 3. Overall rationale (1–5 sentences): summarize the decisive factors with evidence references.
- 4. What A does well: at least 2 bullet points (each with evidence).
- 5. What A does poorly: at least 2 bullet points (each with evidence).
- 6. What B does well: at least 2 bullet points (each with evidence).

- 7. What B does poorly: at least 2 bullet points (each with evidence).
- 8. Atomic evaluation (mutually exclusive labels): output the required label fields from your label table (Section 6), selecting exactly one option per field.

Figure 14: Creative-writing pairwise evaluation annotation instructions.

