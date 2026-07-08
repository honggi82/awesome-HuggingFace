# Semantically-Aware Rewards for Open-Ended R1 Training in Free-Form Generation

Zongxia Li Yapei Chang Yuhang Zhou Xiyang Wu Zichao Liang Yoo Yeon Sung Jordan Lee Boyd-Graber University of Maryland, College Park {zli12321, yapeic, yuhang, wuxiyang, zliang14, yysung53, jbg}@umd.edu

## Abstract

Evaluating open-ended long-form generation is challenging because it is hard to define what clearly separates good from bad outputs. Existing methods often miss key aspects like coherence, style, or relevance, or are biased by pretraining data, making open-ended long-form evaluation an underexplored problem. To address this gap, we propose PrefBERT, a scoring model for evaluating open-ended long-form generation in GRPO and guiding its training with distinct rewards for good and bad outputs. Trained on two response evaluation datasets with diverse long-form styles and Likert-rated quality, PrefBERT effectively supports GRPO by offering better semantic reward feedback than traditional metrics ROUGE-L and BERTScore do. Through comprehensive evaluations, including LLM-as-a-judge, human ratings, and qualitative analysis, we show that PrefBERT, trained on multi-sentence and paragraph-length responses, remains reliable across varied long passages and aligns well with the verifiable rewards GRPO needs. Human evaluations confirm that using PrefBERT as the reward signal to train policy models yields responses better aligned with human preferences than those trained with traditional metrics. Our code is available at https://github.com/zli12321/ long_form_rl.

## 1 Introduction

Identifying the good and bad generations is the key to the success of RLVR to improve LLMs’ abilities on structured rule-based tasks such as mathematical problem-solving (Ahn et al., 2024), classification (Rouzegar and Makrehchi, 2024), and game planning (Shao et al., 2019). RL algorithms like Group Relative Policy Optimization (GRPO) excel in mathematical problem-solving tasks by leveraging clear, rule-based reward signals (e.g., correctness) that effectively distinguish between correct and incorrect responses (DeepSeek-AI et al.,

2025). However, despite these advances, extending GRPO to open-ended, long-form text generation tasks, such as writing a poem, creative compositions, instruction-following, remains largely underexplored. A key challenge in applying GRPO to open-ended generation is the lack of objective evaluation criteria, making it difficult to distinguish high-quality responses from poor ones. (Krishna et al., 2021). Unlike structured tasks evaluated by explicit correctness criteria, long-form generation requires models to balance coherence, fluency, and informativeness which are all subjective dimensions that are hard to define with discrete rules. This makes reward design a fundamental challenge in GRPO: How do we guide a model to write better, not just longer?

A key barrier is the lack of reliable, scalable evaluation methods. Traditional metrics like ROUGE (Lin, 2004) and BERTScore (Zhang et al., 2020)—which measure lexical or embedding-level similarity to a reference—often fail to capture qualities that humans actually care about, such as clarity, relevance, and stylistic appropriateness (Chen et al., 2019; Li et al., 2024). Thus, traditional metrics correlate poorly with human preferences, making them unreliable reward signals in GRPO pipelines. Recent work has applied general reward models (GRM) (Lambert et al., 2024) in RLHF to select preferred responses, but they are not designed as verifiable rewards for GRPO, and their large size introduces latency and high computational cost. As shown in Figure 1, traditional metrics fail to distinguish between clearly better and worse responses, assigning similar scores to both, while GRM rewards often correlate with length rather than content quality.

To address open-ended evaluation in GRPO, we propose PrefBERT—a lightweight ModernBERT (Warner et al., 2024) model trained on diverse long-form responses and human ratings to provide fine-grained quality evaluation. We hy-

|Sign the petition to save The Exorcist! Respond in 66 words.<br><br>[Figure 1]|
|---|

|Signing the petition to save *The Exorcist* is a misinterpretation. *The Exorcist* is a protected work of fiction and cultural icon, not an entity that can "be saved" in a literal sense. Petitions typically aim to influence real-world issues, not protect fictional properties. Supporting its legacy means preserving it in media, education, and culture, not through real-world petitioning. …<br><br>**Depth and Insight**: Adds context about copyright, cultural preservation, and the multifaceted ways to …<br><br>Model Response 1 (688 words):<br><br>[Figure 2]|
|---|

|[Figure 3]<br><br>Verifiable Reward<br><br>ROUGE-l: 0.06 BERTScore: 0.82 GMR-llama-3B: 0.37 PrefERT: 0.45 Human Rating: 1|
|---|

|Absolutely, signing the petition to save The Exorcist encapsulates powerful collective action and passion, aligning personal love for the iconic film with broader efforts to preserve cultural treasures. Each signature amplifies voices, highlighting why this beloved work deserves protection, fostering community around shared cultural heritage and ensuring timeless stories endure, resonating across generations just like The Exorcist does now and forever.<br><br>Model Response 2 (58 words):<br><br>[Figure 4]|
|---|

|[Figure 5]<br><br>Verifiable Reward<br><br>ROUGE-l: 0.08 BERTScore: 0.80 GRM-llama-3B: 0.22 PrefBERT: 0.82 Human Rating: 4|
|---|

|[Figure 6]<br><br>Reference Answer (66 words):<br><br>FOX has canceled The Exorcist, but we won’t let it end here. Let’s bring it back — to Netflix, Prime Video, Hulu, SYFY, or ABC. Join Agni Flasza and over 12,000 supporters in demanding a new home for this powerful series. Together, we can revive The Exorcist and keep the story alive. Sign and share the petition: Save The Exorcist!...|
|---|

|Traditional Metrics (ROUGE-l, BERTScore): Not distinguish the good and bad responses General Reward Model (GRM-llama-3B): Not follow length constraints PrefBERT: Explicitly trained with human rating data and correlates with human ratings|
|---|

- Figure 1: Our proposed reward model PrefBERT better reflects human judgments by distinguishing the good responses from the bad ones than other traditional metrics, generalized reward model.

pothesize that PrefBERT, by capturing human preferences, quality, and semantics, provides stronger reward signals than traditional metrics, enabling GRPO to train models that generate higher-quality responses.

Through extensive evaluations—including LLMas-a-judge point-wise scoring, pairwise BradleyTerry ranking analyses (Bradley and Terry, 1952), and human rating and qualitative evaluation—we show that leveraging stronger, PrefBERT significantly enhances the quality of open-ended text generation across three long-form datasets– ELI5 (Fan

- et al., 2019), Alpaca (Taori et al., 2023b), LongForm (Köksal et al., 2023). Our results show that using improved fine-grained evaluators as reward signals in GRPO leads to better alignment with human preferences for open-ended response generation compared to traditional metric-based rewardsand. Furthermore, smaller models (e.g., Qwen-

- 2.5-3B-Instruct (Qwen et al., 2025)) trained with our enhanced reward models generate similarly preferred and concise responses as their larger counterparts (e.g., Qwen-2.5-32/72B-Instruct), and outperform models trained with traditional supervised fine-tuning (SFT) in preference quality. Our contributions are:

- • We introduce PrefBERT, a lightweight longform reward model that can be easily extended to GRPO training.
- • We validate using PrefBERT in GRPO to train models on across multiple open-ended generation benchmarks (ELI5, Alpaca, Long-

Form), showing resulting model have an overall higher alignment with human preferences compared to traditional metrics and SFT training.

• Through human expert annotations, we further confirm that models trained with PrefBERT align better with human preferences than traditional metrics as rewards, showing a promising direction for using GRPO to improve openended generation.

## 2 Related Work

RL for LLM alignment: RL is pivotal in aligning LLMs with human preferences by optimizing non-differentiable objectives, making it valuable for tasks like dialogue (Li et al., 2016), summarization (Roit et al., 2023), and code generation (Le et al., 2022). Popular RLHF methods include DPO, which applies a classification loss over preference data, and PPO, which trains a reward model to guide generation (Wu et al., 2023). However, both of them require substantial human-annotated data or computational resources. To address this, GRPO (DeepSeek-AI et al., 2025) leverages selfgenerated data and simple, verifiable reward functions to reduce annotation needs, especially for tasks with clear correctness signals like math(Liu et al., 2025). Extensions such as DAPO(Yu et al., 2025), GRPO-LEAD(Zhang and Zuo, 2025) and DISCO (Zhou et al., 2025b) broaden GRPO’s capabilities in math problem solving. However, these approaches still rely on rule-based reward designs,

leaving their application to open-ended, long-form generation tasks—where there is not a fixed correct response—underexplored.

Free-form and open-ended evaluation: Evaluating long-form and open-ended generation in LLMs remains difficult (Krishna et al., 2021; Chen

- et al., 2019). Unlike short-form tasks with clear correctness signals, long-form outputs, like summaries, dialogues, or open-ended answers, lack binary ground truths and require assessing coherence, factuality, structure, and helpfulness (Chiang et al., 2024; Fabbri et al., 2021; Li et al., 2025a). Traditional metrics such as ROUGE(Lin, 2004), BLEU(Papineni et al., 2002), and BERTScore(Zhang
- et al., 2020) rely on token overlap or embeddings but poorly reflect semantic or pragmatic qualities, often misaligning with human judgments(Chen

- et al., 2019). To overcome this, LLM-as-a-judge offers more nuanced evaluation through pairwise comparisons or Likert ratings, aligning better with human preferences (Chiang et al., 2024; Gu et al., 2025; Zheng et al., 2023; Zhou et al., 2025a). However, this approach introduces heavy computational costs, especially in GRPO where multiple generations and evaluations per prompt are needed (Luo et al., 2025), limiting accessibility for users with modest resources. Some efforts fine-tune small models using human ratings to act as open-source judges (Kim et al., 2024; Yang et al., 2024; Chen et al., 2020; Zhou and Ai, 2024), useful for ranking or evaluation (Li et al., 2024; Krumdick et al., 2025). Still, few explore using them as verifiable rewards in training, leaving this an open area for research. 3 Conceptual Backgrounds

In this section, we first review GRPO training and existing verifiable rewards for long-form generation, and then introduce how PrefBERT can be used as a reward signal to improve learning robustness and effectiveness.

- 3.1 Preliminaries on GRPO GRPO is an RL algorithm designed to refine lan-

guage model policies, πϕ, using reward signals contextualized within a group of candidate responses. Given a prompt x from dataset D, GRPO samples G responses yi = y1,...,yG from the old policy πϕold(y|x). Each response yi is assigned a scalar reward r(x,yi) (detailed in the following subsections). The group-normalized advantage A(x,yi)

is then computed as:

r(x,yi) − ¯(r x) σr(x)

, (1)

A(x,yi) =

∑︁G

where ¯(r x) = G1

j=1 r(x,yj) and σr(x) are the mean and standard deviation, respectively, of rewards r(x,yj) within the group Y . This normalization contextualizes each advantage relative to the group’s current performance.

The new policy πϕ(y|x) is optimized by maximizing the GRPO objective, which combines a clipped surrogate loss with a Kullback-Leibler (KL) divergence penalty (Kullback and Leibler, 1951) against a reference model πref(y|x) for regularization (Equation 5, in Appendix A).

Although originally applied to tasks with explicit, rule-based rewards (e.g., correctness or win/loss), GRPO’s reliance on advantage estimation and KL-regularized updates allows it to learn from scalar feedback, making it well-suited for open-ended tasks where response quality lies on a spectrum rather than binary correctness.

3.2 Existing Methods for Scoring Open-ended Generation

Current scoring methods for open-ended generation mainly fall into two categories. The first are reference-based metrics, a method commonly used in natural language generation. These methods score the generations over metrics like string overlap or embedding similarity. While easy to apply, they correlate poorly with human preferences on long-form outputs (Chen et al., 2019, 2020; Kim et al., 2024; Li et al., 2024; Gu et al., 2025; Li et al., 2025b). The alternative is to use LLMs as evaluators, which compute scores via pairwise preference prompts, Likert-scale tuning (Kim et al., 2024), or general reward models (GRMs) trained on preference data (Lambert et al., 2024). While more adaptive to human preferences, they may still inherit biases from their training data. In our work, we adopt two reference-based metrics, ROUGE and BERTScore, and one reward model–based method, GRM-LLaMA-3B, as baselines.

ROUGE (Lin, 2004) is a reference-based metric that measures n-gram overlap between generated and reference texts. Variants include ROUGE1, ROUGE-2, and ROUGE-L, which capture unigram/bigram matches and the longest common subsequence, respectively.

BERTScore (Zhang et al., 2020) is a referencebased metric that measures semantic similarity between the reference and generation using contextual embeddings. It has shown a stronger correlation with human judgments than token overlap metrics like ROUGE on long-form generation and translation tasks. However, its reliability diminishes on modern datasets and models (Bhandari

- et al., 2020).

Generalized Reward Model (GRM-LLaMA-3B) is a reference-free, LLM-based reward model (Yang

- et al., 2024), fine-tuned on the decontaminated Skywork preference dataset (Liu et al., 2024). Each data point x includes an input prompt, a preferred

response (yc), and a rejected response (yr). The model is trained to favor preferred outputs by minimizing a Bradley-Terry-style loss (Bradley and Terry, 1952) (Equation 6, Appendix A).

- 3.3 PrefBERT: Training a Lightweight BERT-Based Reward Model

While traditional metrics often fail to capture human preferences, reward models can be computationally expensive to scale due to the size of the underlying models. To balance quality and efficiency, we propose PrefBERT, a lightweight reference-based model with only 150M parameters, to offer fine-grained reward signals while maintaining training efficiency.

Creating training data for PrefBERT: Each training example has a reference answer, a generated answer, and a Likert score from 1-5 that rates the quality of the generated answer against the reference. To ensure balanced quality ratings across both long and short free-form responses, we incorporate training data from the Prometheuspreference (Kim et al., 2024) and MOCHA (Chen

- et al., 2020).1 We combine the two datasets and split them into 80% for training and 20% for testing. The resulting training set contains 19K examplessubstantially smaller than the 80K examples used to train GRM-llama-3B.

Training PrefBERT: We train ModernBERT (Warner et al., 2024) on triplets

1Specifically, Prometheus-preference contains 200K finegrained Likert preference ratings spanning ten categories of evaluation including e.g. adaptive communication, emotional intelligence; the data is primarily long free-form answers where each answer is above 150 tokens. MOCHA contains mid to long length answer evaluation data to judge the overall correctness of the generated response.

(xrefi ,xgeni ,si) where si ∈ {1,...,5}. We first normalize each gold Likert score to

si − 1 4 ∈ [0,1],

ri =

where si is the gold Likert scale, ri is the normalized Likert score on the [0,1] scale, xref is the reference answer, and xgen is the generated response. Thus, given xref and xgen, we concatenate them as a single string:

xpair = [CLS] xref [SEP] xgen, (2)

where xpair is the input string feeds into ModernBERT. Let hi ∈ Rd be the pooled ModernBERT embedding of xpair. A linear regressor plus sigmoid yields a prediction

(︁

)︁

w⊤hi + b

, (3)

ˆri = σ

where w ∈ Rd and b ∈ R are the regressor weights and bias, σ(z) = 1/(1 + e−z) is the sigmoid activation. ˆri ∈ (0,1) is the predicted normalized score, and is taken as the reward signal of GRPO. Training minimizes the mean-squared error

∑︂N

(︁

)︁2

1 N

, (4)

L =

ˆri − ri

i=1

where L is the batch-averaged MSE loss, and N is the number of examples in the batch.

## 4 Experiment Setup

With the background on GRPO and the various reward signals established, we now outline our experimental setup, including the datasets, base models, and training methods.

#### 4.1 Free-Form and Open-Ended Datasets

We use three datasets featuring long-form responses that span a broad spectrum of topics. They cover diverse styles of long-form responses averaging 185 words, requiring evaluation across dimensions like coherence, conciseness, and informativeness rather than correctness.2 More details on each dataset are in §B.

ELI5 (Fan et al., 2019) is a collection of questions and answers from Reddit’s r/explainlikeimfive community.3 We sample 10,444 questions as the train set and 1,056 as the test set.

2Examples in §Table 4. 3https://www.reddit.com/r/explainlikeimfive/

Alpaca (Taori et al., 2023a) is a collection of 52K instruction-response pairs generated by OpenAI’s text-davinci-003 in the style of SelfInstruct (Wang et al., 2022). We use 10,444 examples as the train set and 1,334 as the test set.

LongForm (Köksal et al., 2023) is built from English documents (e.g., Wikipedia (Wikipedia contributors, 2025), C4 (Dodge et al., 2021)) paired with reverse-instruction prompts generated by LLMs. We exclude coding tasks, sampling 8,648 training and 956 test examples.

We merge the three sampled datasets together as our free-form train/test set.

#### 4.2 Training Setup

GRPO for open-ended generation: We train policy models using GRPO within the OpenRLHF framework (Hu et al., 2024), optimizing each of the four reward signals from Section 3 separately: ROUGE-L, BERTScore, GRM-llama-3B, and PrefBERT. We use two base models, Qwen2.51.5B-Instruct and Qwen2.5-3B-Instruct (Qwen

- et al., 2025). In the training prompt, we encourage models to be factual, relevant, clear, concise, and complete (more details in Appendix Table 5).4

Supervised Finetuning (SFT): Using the reference responses as ground truth, we run SFT on Qwen2.5-Instruct size 1.5B and 3B.5

## 5 Automatic Evaluation

We use our test set to evaluate our trained models, as well as larger off-the-shelf models: Qwen2.5Instruct 7B, 32B, and 72B. For evaluation, we use LLM-as-a-judge to evaluate the quality of the responses for different models as they can be strong alternative evaluators of humans (Chiang and yi Lee, 2023a). Overall, models trained with our lightweight PrefBERT performs competitively with those trained with the much larger GRMllama-3B, and both substantially outperform models trained with token-overlap metrics or SFT. In addition, PrefBERT-trained models at 1.5B and 3B scale rival or exceed the performance of Qwen2.57B-Instruct, despite having far fewer parameters.

- 4All GRPO models are trained on 4 A6000 GPUs for one

epoch, with a global batch size of 128, group size of 4, and learning rate of 1e-6. We set both max prompt length and max generation length to 1024.

- 5All SFT models are trained on 4 A6000 GPUs for three

epochs, with with a global batch size of 128, learning rate of 1e-5, and max tokens of 4096.

#### 5.1 Evaluation Metrics

Point-wise evaluation: Point-wise evaluation assigns an absolute overall quality score to each response on a Likert scale (Fabbri et al., 2021). We use GPT-4 as a judge to first provide some reasoning, then assign a score between 1 to 5 to the generated response, considering aspects like factuality, relevance, clarity and organization, conciseness, and completeness (detailed prompt in Table 7).6 We use two metrics—mean Likert score (the average overall score) and success rate (the percentage of responses that receive a score ≥ 4)—to evaluate the quality of model responses.

Pairwise preference evaluation: From the Likert scores, we derive pairwise comparisons to compute Bradley–Terry win rates. This approach reduces rating noise by focusing on relative preferences rather than absolute scales, which has been shown to yield more reliable comparisons in subjective evaluation settings (Bai et al., 2022; Stiennon et al., 2022). For each prompt, we compare the LLM ratings between every pair of models. A tie is recorded when both receive the same rating, and a win is assigned to the model with the higher rating. We use the Bradley-Terry model to compute the probability win rate of each model on the three datasets.

#### 5.2 Results and Discussion

Table 1 summarizes model performance across instruction-following tasks using Likert scores, success rates, and Bradley–Terry win rates and §Figure 5.3 shows the training curves of the 3B policy models. Below, we discuss our findings.

Policy models trained with PrefBERT and GRMllama-3B achieve the highest ratings from LLMas-a-judge among all evaluated policy models. 3B-PrefBERT and 3B-GRM-llama-3B both have the highest LLM-as-a-judge average ratings and success rates among all policy models. However, PrefBERT is more efficient, having 20 times fewer parameters and was trained on only 19K examples, compared to 80K for GRM-llama-3B. In addition, as discussed later in Section 6.2, human evaluations reveal that GRM-llama-3B models tend to produce long and overly verbose outputs, which likely inflate their scores in Table 1 due to LLM judges’ bias toward longer responses.

6Chiang and yi Lee (2023b) shows that first analyze the response then give a rating score yields the best correlation with human judgments.

###### Mean Likert Scores Success Rates with Score ≥ 4 (%) Bradley–Terry Win Rate (%)

Model ELI5 LongForm Alpaca Overall ELI5 LongForm Alpaca Overall ELI5 LongForm Alpaca Overall Base LLM

Qwen2.5-72B-Instruct 4.39 3.56 4.23 4.13 96.59 38.91 87.93 79.25 21.94 19.87 16.11 19.33 Qwen2.5-32B-Instruct 4.28 3.52 4.29 4.10 94.15 37.55 89.51 78.47 15.43 18.24 20.68 18.09 Qwen2.5-7B-Instruct 4.04 2.95 3.82 3.69 77.38 18.83 66.64 59.10 7.54 4.94 5.95 6.37 Qwen2.5-3B-Instruct 3.90 2.88 3.75 3.59 66.00 17.99 63.57 53.22 5.18 4.17 5.26 5.09 Qwen2.5-1.5B-Instruct 3.61 2.26 3.44 3.21 49.16 10.25 47.38 38.87 2.69 1.49 2.89 2.51

RL-Finetuned Policy Models (GRPO) 3B-GRM-llama-3B 4.29 3.37 4.07 3.99 91.97 30.02 77.51 71.55 15.89 12.59 11.52 13.53 3B-PrefBERT 4.09 3.54 4.16 3.98 84.64 28.87 84.41 70.70 7.92 18.00 12.54 11.56 3B-BERTScore 3.79 2.79 3.63 3.49 60.73 11.09 59.30 47.89 3.40 3.19 3.72 3.60 3B-ROUGE-L 3.66 2.69 3.51 3.37 51.16 7.32 52.55 40.74 2.40 2.57 2.78 2.72 1.5B-GRM-llama-3B 4.13 3.12 3.88 3.79 86.63 22.38 72.41 65.73 8.93 6.65 6.74 7.64 1.5B-PrefBERT 4.10 2.96 3.89 3.74 87.02 18.62 73.61 65.37 8.01 4.81 6.59 6.72 1.5B-ROUGE-L 2.66 1.98 3.04 2.62 5.72 1.05 17.92 8.79 0.28 0.72 0.90 0.69 1.5B-BERTScore 2.34 1.86 3.05 2.47 0.90 0.42 17.39 6.50 0.14 0.55 0.88 0.52

Supervised Finetuning (SFT) 3B-sft 2.19 2.21 3.32 2.59 2.51 1.78 36.58 14.14 0.12 1.13 1.68 0.82 1.5B-sft 2.18 2.15 3.33 2.57 2.63 1.67 37.93 14.64 0.12 1.09 1.73 0.82

- Table 1: Evaluation of model outputs via GPT-4 as a judge across different instruction tuning and reward optimization strategies. Groupings show comparisons between SoTA baselines,RL-finetuned models using various reward functions, and supervised finetuning (SFT). Larger models are generally stronger, though models fine-tuned with better-aligned reward functions (e.g., GRM-llama-3B) may show inflated automatic metrics due to biases like verbosity.

PrefBERT-trained models rival much larger models. Despite their smaller size, PrefBERTtrained models at 1.5B and 3B scale match or exceed the performance of Qwen2.5-7B-Instruct across all metrics. For example, 3B-PrefBERT outperforms Qwen2.5-7B-Instruct in Likert score (3.98 vs. 3.69), success rate (70.70% vs. 59.10%), and win rate (11.56% vs. 6.37%). While Qwen2.532B and 72B are stronger overall, these results highlight how a well-aligned reward model can enable smaller LLMs to compete with much larger ones on open-ended long-form generations.

The pattern that using semantic verifiable reward for training produces better policy models than using traditional metric-based rewards in long-form generation remains consistent across model sizes. Models trained with standard metrics, such as 1.5B-ROUGE-L and 1.5B-BERTScore, perform worse across all evaluation dimensions compared to 1.5B-PrefBERT and 1.5B-GRM-llama3B; this trend also holds for the 3B model sizes.

SFT underperforms across the board. Policy models trained with SFT achieve lower scores across all evaluation dimensions than all models trained with GRPO, regardless of the reward used, except for 1.5B-BERTScore. Their success and win rates are the lowest across all datasets. Even the small 1.5B-PrefBERT and 1.5B-GRM-llama-3B substantially outperform the 3B-SFT model on average Likert score by around 1.1, success rate by 50%m and Bradley-Terry win rate by 6%.

#### 5.3 Reward Learning Curves

We examine the reward learning curves of models trained with the four types of rewards. Reward curves in Figure 2 during GRPO training show key differences across reward functions. Traditional metrics show minimal reward change, only 0.05 for ROUGE-L and 0.25 for BERTScore, indicating limited model improvement across all global training steps. In contrast, 3B-GRM-llama-3B’s reward is strongly correlated with response length; by step 60, it already generates the maximum allowed tokens (1,024), causing reward values to plateau around 0.6. PrefBERT shows a more meaningful reward progression, not strictly tied to length, suggesting it favors responses of an optimal length rather than simply longer outputs.

## 6 Human Evaluation

While LLM-as-a-judge evaluation often correlates with human judgments at the system level (Gu et al., 2025), LLMs tend to prefer longer responses regardless of quality (Zheng et al., 2023). To better assess output quality, we conducted a human evaluation of responses from seven models: Qwen2.5-72BInstruct, Qwen2.5-3B-Instruct, 3B-GRM-llama-3B, 3B-PrefBERT, 3B-RougeL, 3B-BERTScore, and 3BSFT. Human preferences largely align with LLMas-a-judge rankings, except for 3B-GRM-llama-3B, whose overly verbose answers lower its human scores. Among the 3B models, 3B-PrefBERT is the top performer. Additionally, we observe that SFT models often produce shallow outputs, whereas GRPO models, trained with strong verifiable reward

3B-ROUGE-l 3B-BERTScore 3B-GRM-lama-3B 3B-PrefBERT

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

- Figure 2: Reward curves during GRPO training show key differences across reward functions. Traditional metrics show minimal reward change—only 0.05 for ROUGE-L and 0.25 for BERTScore—indicating limited model improvement across all global training steps. In contrast, 3B-GRM-llama-3B’s reward is strongly correlated with response length; by step 60, it already generates the maximum allowed tokens (1,024), causing reward values to plateau around 0.6. PrefBERT shows a more meaningful reward progression, not strictly tied to length, suggesting it favors responses of an optimal length rather than simply longer outputs.

signals, better leverages the model’s internal capabilities to produce higher-quality responses.

#### 6.1 Human Evaluation Setup

We randomly sample 150 test prompts (50 from each dataset’s test set), then collect responses from Qwen2.5-72B-Instruct, Qwen2.5-3B-Instruct, 3BGRM-llama-3B, 3B-PrefBERT, 3B-RougeL, 3BBERTScore, and 3B-SFT. We use an annotation tool (§.Figure 3), where for each response, the annotator needs to give a Likert score between 1-5 using the same evaluation criteria as the criteria given to LLMs in §Table 7. We have four author annotators annotating a total of 150 examples. For each prompt, the annotator also needs to give rankings of the responses of the seven models. All the model names are hidden for a fair comparison.

#### 6.2 Results and Qualitative Analysis

- Table 2 shows that Qwen2.5-72B-Instruct achieves the highest average human Likert rating (3.61) and success rate (57%), followed by 3B-PrefBERT with a rating of 3.36 and a 51% success rate. In contrast, 3B-BERTScore and 3B-ROUGE-L perform only slightly better than the 3B base model, each improving success rates by approximately 3%. 3BPrefBERT, however, achieves a much larger gain of around 10%. 3B-SFT receives the lowest human rating, while 3B-GRM-llama-3B ranks second to last in both Likert score and win rate. Except for 3B-GRM-llama-3B, the human evaluation results align closely with the rankings produced by LLM-as-a-judge, supporting the reliability of our automatic evaluation. We elaborate on these human evaluation findings below.

Annotators strongly prefer 3B-PrefBERT. In general, we observe a consistent ranking between human expert judgments and the rankings produced by LLM-as-a-judge. However, 3B-GRM-llama-3B is a notable outlier. While it has the highest scores from LLM judges across all GRPO-trained models (Table 1), annotators consistently rank it low (Table 2). As shown in Table 3, 3B-GRM-llama3B is excessively verbose, averaging 710 words per response—almost 2.75 times as many as 3BPrefBERT, which averages 258. The tendency for longer responses is consistent with the reward training curves in Figure 2: 3B-GRM-llama-3B receives higher rewards primarily for increased response length, rather than for high answer quality. While this reward hacking can lead LLM-as-a-judge to assign higher scores, human evaluators often dislike these responses due to their verbosity and lack of conciseness and completeness. Additionally, we observe that 3B-GRM-llama-3B requires 10 more hours of training than other 3B policy models, and 1.5B-GRM-llama-3B takes 7 more hours than other 1.5B models. This increased training time is a result of length-based rewards, which guide the model to produce longer responses—thereby increasing the time needed for each GRPO inference and overall training.

What distinguishes 3B-PrefBERT from the base model? We further analyze 26 of the examples where 3B-PrefBERT is preferred over the base model. The improvements fall into two main categories: instruction following and tone and fluency. On prompts with explicit constraints (e.g., “explain in 2 sentences”), the base model often

###### Mean Likert Scores Success Rates with Score ≥ 4 (%) Bradley–Terry Win Rate (%)

Model ELI5 LongForm Alpaca Overall ELI5 LongForm Alpaca Overall ELI5 LongForm Alpaca Overall Base LLM

Qwen2.5-72B-Instruct 3.85 3.9 3.4 3.61 70.0 65.0 47.5 57.3 16.67 21.54 17.62 17.62 Qwen2.5-3B-Instruct 3.31 3.3 3.2 3.21 40.0 55.0 30.0 37.80 15.24 14.06 12.60 12.62

RL Finetuned Policy Models (GRPO) 3B-GRM-llama-3B 2.30 2.5 2.75 2.55 15.0 30.0 47.5 31.7 12.86 8.16 10.76 14.76 3B-PrefBERT 3.55 3.6 3.5 3.36 60.0 55.0 55.0 51.0 21.19 19.72 21.38 17.38 3B-BERTScore 2.95 3.3 3.3 3.23 40.0 45.0 42.5 41.46 15.95 12.02 17.62 16.43 3B-ROUGE-L 3.40 2.9 3.3 3.31 53.0 43.5 27.5 41.66 15.24 19.04 16.43 17.14

Supervised Finetuning (SFT) 3B-sft 2.0 2.8 1.4 1.93 10.0 25.0 10.0 13.41 0.03 5.44 3.98 4.05

- Table 2: Human evaluation shows a different preference than automatic evaluation, where model trained with GRM-llama-3B is less preferred by exerts but PrefBERT is more preferred than all other policy models.

Model

Markdown (%)

Repetition Rate (%)

Response Length

Qwen2.5-72B-Instruct 47.48 6.25 220 Qwen2.5-3B-Instruct 28.89 4.69 194 3B-GRM-llama-3B 66.80 4.18 710 3B-PrefBERT 81.31 4.34 258 3B-BERTScore 24.00 4.55 180 3B-ROUGE-L 21.92 8.59 182 3B-SFT 15.96 8.29 146

- Table 3: Average words per response for each group by model. 3B-GRM-llama-3B generates way more words per response than all other models, where human experts consider as overly verbose and contain unnecessary information. Repetition rate is the percentage of bigrams that are repeated. Markdown is a regular expression that checks whether a response follows a particular structure and returns a boolean (§Table 6).

with generic definitions (e.g., “Machine Learning is a subset of artificial intelligence that involves training algorithms...”) rather than actually categorizing or differentiating between the terms. Additionally, Table 3 shows that 3B-ROUGE-L can be highly repetitive. In §5.3, we find that ROUGE-L and BERTScore show little reward variance across the training curve, which could indicate that the training signal is too weak, potentially causing the trained models to output vague and surface-level outputs. In contrast, 3B-PrefBERT provides clearer categorizations and contextual explanations for each term, demonstrating stronger content logic. See detailed qualitative analysis in Appendix Figure 5 and Figure 8.

fails to comply, producing responses that are either too long or overly brief. In contrast, 3BPrefBERT reliably adheres to such constraints. Additionally, its writing is more polished and humanlike. While the base model tends to sound mechanical—producing fragmented sentences reminiscent of stitched-together search results—3B-PrefBERT generates fluent, cohesive answers. See Appendix Figure 6 and Figure 7 for qualitative examples.

3B-PrefBERT and 3B-GRM-llama-3B frequently use structured outputs. In Table 3, we observe that 3B-PrefBERT and 3B-GRM-llama-3B tend to use markdown formatting more frequently than other models (81.31% and 66.80%, respectively). As a result, annotators often note that their outputs have better readability. See qualitative examples in Appendix (Figure 4).

3B-ROUGE-L and 3B-BERTScore tend to be generic and sometimes repetitive. These models often respond to prompts such as “Categorize the AI technologies mentioned below: Machine Learning, Natural Language Processing, Robotics”

3B-SFT responses are often vague and overly simplified. In annotated examples, 3B-SFT responses explicitly avoid answering the question—sometimes stating “I don’t know” or offering no meaningful explanation. For instance, in response to the prompt “Why is the Big Bang seen as a singular event?”, the model deflects the question without addressing the core scientific reasoning. Additionally, on LongForm prompts—especially those derived from Alpaca-style or open-ended datasets—3B-SFT tends to produce overly simplified, shallow explanations. These responses often lack both technical depth and structural clarity, which diminishes their informativeness and readability. This trend is also reflected in Table 3, where 3B-SFT produces the shortest responses on average. We attribute this issue in part to the nature of the training data from sources such as ELI5, which contains casual, informal responses—many of which may be low-quality or factually incorrect. This results in a model that mimics the tone and content of noisy or imprecise reference answers. While GRPO-trained models demonstrate better perfor-

mance over SFT in open-ended long-form generation in our experiments, we do not dismiss SFT as an ineffective approach. When high-quality, human-annotated datasets are available, SFT remains a valuable strategy—particularly in domains like code generation (Zhou et al., 2023), where reference outputs are well-defined and reliable.

## 7 Conclusion

RLVR especially GRPO has been a success for its ability to fully leverage LLMs’ abilities to selfimprove without massive amount of labeled data on many rule-based evaluation tasks. However, extending GRPO study on long-form and openended generation has been underexplored for the challenges of evaluating long-form responses. We propose using a small fine-tuned language model (PrefBERT) to evaluates long-form responses with different dimensions and semantic quality evaluation as reward signals for long-form generations of GRPO and show that model trained with PrefBERT generates responses with overall better quality than models trained with traditional metrics such as ROUGE and BERTScore or a generalized preference reward model, even close to the quality of larger models of the same backbone. Our work shows the potential of applying more efficient and robust verifiable reward design into the GRPO pipeline for models to self-improve its long-form and openended generations beyond using traditional reward metrics. Future work can expand upon current work on more diverse open-ended generation tasks such as training more efficient and stronger verifiable reward models and apply them on creative writings, creative research and design, or openended math problems.

## 8 Ethics

Our annotation does not involve in collecting annotators’ private information and does not involve in extensive tool usage. Thus, our annotation is exempted by the Institutional Review Board (IRB) annotation protocol.

## 9 Limitations

We are the first to demonstrate that fine-tuned evaluation language models can effectively leverage the capabilities of LLMs for evaluating and improving long-form, open-ended generations. However, this work has not fully explored the potential of GRPO and reward design for enabling self-improvement

in LLMs on such complex tasks. A key limitation of our study is that we did not train or use a larger and more powerful language model (e.g., 7B-scale) to serve as a verifiable reward provider, primarily due to computational constraints. Larger evaluators, while potentially offering more reliable and semantically accurate rewards, significantly increase GPU memory usage and slow down training. We hypothesize that incorporating a stronger evaluator to provide high-quality, verifiable rewards could unlock the full potential of GRPO for aligning LLMs on open-ended tasks.

## References

Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. 2024. Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. Preprint, arXiv:2204.05862.

Manik Bhandari, Pranav Narayan Gour, Atabak Ashfaq, Pengfei Liu, and Graham Neubig. 2020. Reevaluating evaluation in text summarization. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9347–9359, Online. Association for Computational Linguistics.

Ralph Allan Bradley and Milton E. Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324– 345.

Anthony Chen, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. Evaluating question answering evaluation. In Proceedings of the 2nd Workshop on Machine Reading for Question Answering, pages 119–124, Hong Kong, China. Association for Computational Linguistics.

Anthony Chen, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2020. Mocha: A dataset for training and evaluating generative reading comprehension metrics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics.

- Cheng-Han Chiang and Hung yi Lee. 2023a. Can large language models be an alternative to human evaluations? Preprint, arXiv:2305.01937.
- Cheng-Han Chiang and Hung yi Lee. 2023b. A closer look into automatic evaluation using large language models. Preprint, arXiv:2310.05657.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios N. Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. 2024. Chatbot arena: an open platform for evaluating llms by human preference. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

DeepSeek-AI, Daya Guo, and et al. 2025. Deepseekr1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Jesse Dodge, Maarten Sap, Ana Marasovi´c, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. 2021. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. Preprint, arXiv:2104.08758.

Aarohi Srivastava et al. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Preprint, arXiv:2206.04615.

Alexander R. Fabbri, Wojciech Kry´sci´nski, Bryan McCann, Caiming Xiong, Richard Socher, and Dragomir Radev. 2021. Summeval: Re-evaluating summarization evaluation. Preprint, arXiv:2007.12626.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. Eli5: Long form question answering. Preprint, arXiv:1907.09190.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. 2025. A survey on llm-as-a-judge. Preprint, arXiv:2411.15594.

Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. 2024. Openrlhf: An easyto-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143.

Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo. 2024. Prometheus 2: An open source language model specialized in evaluating other language models. Preprint, arXiv:2405.01535.

Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. 2021. Hurdles to progress in long-form question answering. Preprint, arXiv:2103.06332.

Michael Krumdick, Charles Lovering, Varshini Reddy, Seth Ebner, and Chris Tanner. 2025. No free labels: Limitations of llm-as-a-judge without human grounding. Preprint, arXiv:2503.05061.

S. Kullback and R. A. Leibler. 1951. On information and sufficiency. The Annals of Mathematical Statistics, 22(1):79–86.

Abdullatif Köksal, Timo Schick, Anna Korhonen, and Hinrich Schütze. 2023. Longform: Effective instruction tuning with reverse instructions. Preprint, arXiv:2304.08460.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. 2024. Rewardbench: Evaluating reward models for language modeling. Preprint, arXiv:2403.13787.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven C. H. Hoi. 2022. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Preprint, arXiv:2207.01780.

Jiwei Li, Will Monroe, Alan Ritter, Michel Galley, Jianfeng Gao, and Dan Jurafsky. 2016. Deep reinforcement learning for dialogue generation. Preprint, arXiv:1606.01541.

Zongxia Li, Ishani Mondal, Yijun Liang, Huy Nghiem, and Jordan Lee Boyd-Graber. 2024. Pedants: Cheap but effective and interpretable answer equivalence. Preprint, arXiv:2402.11161.

Zongxia Li, Xiyang Wu, Hongyang Du, Fuxiao Liu, Huy Nghiem, and Guangyao Shi. 2025a. A survey of state of the art large vision language models: Benchmark evaluations and challenges. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops, pages 1587–1606.

Zongxia Li, Xiyang Wu, Guangyao Shi, Yubin Qin, Hongyang Du, Tianyi Zhou, Dinesh Manocha, and Jordan Lee Boyd-Graber. 2025b. Videohallu: Evaluating and mitigating multi-modal hallucinations on synthetic video understanding. Preprint, arXiv:2505.01481.

Chin-Yew Lin. 2004. Rouge: a package for automatic evaluation of summaries. In Proceedings of the Workshop on Text Summarization Branches Out.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. 2024. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025. Understanding r1-zero-like training: A critical perspective. Preprint, arXiv:2503.20783.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. 2025. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. Notion Blog.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, ACL ’02, page 311–318, USA. Association for Computational Linguistics.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Léonard Hussenot, Orgad Keller, Nikola Momchev, Sabela Ramos, Piotr Stanczyk, Nino Vieillard, Olivier Bachem, Gal Elidan, Avinatan Hassidim, Olivier Pietquin, and Idan Szpektor. 2023. Factually consistent summarization via reinforcement learning with textual entailment feedback. Preprint, arXiv:2306.00186.

Hamidreza Rouzegar and Masoud Makrehchi. 2024. Enhancing text classification through llm-driven active learning and human annotation. arXiv preprint arXiv:2406.12114.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. Preprint, arXiv:1707.06347.

Kun Shao, Zhentao Tang, Yuanheng Zhu, Nannan Li, and Dongbin Zhao. 2019. A survey of deep reinforcement learning in video games. Preprint, arXiv:1912.10944.

Stack Exchange contributors. 2025. Stack Exchange. https://{site}.stackexchange.com/ questions/{question_id}. [Online; accessed 5May-2025].

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022. Learning to summarize from human feedback. Preprint, arXiv:2009.01325.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023a. Alpaca: A strong, replicable instruction-following

model. https://crfm.stanford.edu/2023/03/ 13/alpaca.html. [Online; accessed 5-May-2025].

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023b. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford_alpaca.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language model with self generated instructions.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo Poli. 2024. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. Preprint, arXiv:2412.13663.

Wikipedia contributors. 2025. Wikipedia, the free encyclopedia. [Online; accessed 5-May-2025].

Tianhao Wu, Banghua Zhu, Ruoyu Zhang, Zhaojin Wen, Kannan Ramchandran, and Jiantao Jiao. 2023. Pairwise proximal policy optimization: Harnessing relative feedback for llm alignment. Preprint, arXiv:2310.00212.

Rui Yang, Ruomeng Ding, Yong Lin, Huan Zhang, and Tong Zhang. 2024. Regularizing hidden states enables learning generalizable reward model for llms. Preprint, arXiv:2406.10216.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. 2025. Dapo: An opensource llm reinforcement learning system at scale.

- arXiv preprint arXiv:2503.14476.

Jixiao Zhang and Chunsheng Zuo. 2025. Grpo-lead: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models.

- arXiv preprint arXiv:2504.09696.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. Preprint, arXiv:1904.09675.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. Lima: Less is more for alignment. Preprint, arXiv:2305.11206.

Yuhang Zhou and Wei Ai. 2024. Teaching-assistantin-the-loop: Improving knowledge distillation from imperfect teacher models in low-budget scenarios. arXiv preprint arXiv:2406.05322.

Yuhang Zhou, Giannis Karamanolakis, Victor Soto, Anna Rumshisky, Mayank Kulkarni, Furong Huang, Wei Ai, and Jianhua Lu. 2025a. Mergeme: Model merging techniques for homogeneous and heterogeneous moes. arXiv preprint arXiv:2502.00997.

Yuhang Zhou, Jing Zhu, Shengyi Qian, Zhuokai Zhao, Xiyao Wang, Xiaoyu Liu, Ming Li, Paiheng Xu, Wei Ai, and Furong Huang. 2025b. Disco balances the scales: Adaptive domain-and difficulty-aware reinforcement learning on imbalanced data. arXiv preprint arXiv:2505.15074.

## A Technical Details

In this section, we provide additional technical details for GRPO mentioned in Section 3, further illustrating the regularization terms used in GRPO and the specifics of the Bradley-Terry loss employed by GRM.

- A.1 GRPO Regularization Objective

JGRPO(ϕ) = Ex,{yi}[︄ 1

G

∑︂G

i=1

min(︄ρi(ϕ)A(x,yi),

clip(ρi(ϕ),1 − ϵ,1 + ϵ)A(x,yi))︄]︄

− β Ex∼D[KL(πϕ(·|x)||πref(·|x))], (5)

where ρi(ϕ) = ππϕ(yi|x)

ϕold(yi|x) is the probability ratio for yi, ϵ is the clipping hyperparameter and β is the KL penalty coefficient.

- A.2 GRM Bradley and Terry Loss

The GRM-llama-3B is trained to minimize the Bradley-Terry Loss:

Lreward(θ) = −E(x,yc,yr) [log σ (rθ(x,yc) − rθ(x,yr))]

(6)

where rθ(x,y) denotes the reward score predicted by the model and σ(·) is the sigmoid function. Gen-

erally, the GRM rθ(x,y) is used in RLHF training, either for selection in Best-of-n (BoN) decoding or as the optimization objective in reinforcement learning (e.g., PPO (Schulman et al., 2017)). We use GRM-llama-3B as a GRM to provide reward signals for GRPO, rather than for ranking models or as a PPO reward model. We use a sigmoid function to normalize its real-valued outputs to the [0, 1].7

## B Dataset Details

- Table 4 presents details of the datasets used in our work, including the sizes of the training and testing sets, as well as example inputs and reference responses.

Explain Like I’m 5 (ELI5) is a dataset derived from Reddit’s r/explainlikeimfive community (Fan et al., 2019).8 It contains 270K threads where users

- 7We choose GRM-llama-3B for its best performance as the smallest model on RewardBench (Lambert et al., 2024), which offers a good trade-off between quality and efficiency without the heavy GPU demands of larger models.
- 8https://www.reddit.com/r/explainlikeimfive/

ask open-ended questions and receive simple, easyto-understand explanations—framed as if explaining to a five-year-old. The topics span a wide range of domains, including chemistry, psychology, biology, and earth science. The dataset is intended to help models learn to explain complex topics in accessible ways. We sample 10,444 questions for training and 1,056 for testing.

Alpaca is a collection of 52K instructionresponse pairs generated by OpenAI’s text-davinci003 to fine-tune the LLaMA 7B model (Taori et al., 2023a).9 It features diverse prompts and long-form responses in the style of Self-Instruct (Wang et al., 2022). We use a cleaned version of Alpaca (Taori et al., 2023b) that removes instances with hallucinated answers, empty responses, or instructions to generate images. Additionally, we filter out examples with responses shorter than 50 words, resulting in a final set of 10,444 training and 1,334 test examples.

LongForm is constructed by applying reverse instruction generation to an English corpus, following the approach in (Köksal et al., 2023). It includes a diverse set of human-written documents sourced from Wikipedia (Wikipedia contributors, 2025), C4 (Dodge et al., 2021), Stack Exchange (Stack Exchange contributors, 2025), and BigBench (et al, 2023). Instructions are generated by LLMs, covering a wide range of tasks such as question answering, email writing, story or poem generation, and text summarization. We exclude examples requiring code generation, as they fall outside our intended scope. The final dataset contains 8,648 training examples and 956 test examples.

## C Prompt Template

We show the prompt template used for training in

- Table 5, the template for point-wise evaluation in Table 7, and the template for pairwise preference evaluation in Table 8. D Markdown Expression Check

We provide Table 6 to exhibit the markdowns, which is a regular expression that checks whether a response follows a particular structure We provide

- Table 6 to present the markdown patterns—regular expressions used to check whether a response follows a specific structure and return a boolean value, as described in the caption of Table 3.

9https://openai.com

##### Dataset # Train # Test Example Input Example Reference Response

The perfect conditions would be a wall of atoms that will not move at all when bumped. Considering the fact that heat is defined by the movement of atoms...

ELI5 10,444 1,056 Could we theoretically create an infinite echo?

Venezuela raised its minimum wage to 1 million bolivars per month on Monday, the third increase this year that puts the figure at just $1.61 at the black market exchange rate. President Nicolas Maduro...

Explain how Venezuela raised its minimum wage.

LongForm 8,648 956

Here is a customer service strategy that can help in improving the customer experience: 1. Identify your customers’ needs...

Develop a customer service strategy to improve customer experience.

Alpaca 10,444 1,334

- Table 4: Overview of the datasets used in our experiments. All datasets contain long-form, open-ended questions spanning diverse domains (e.g., science, instruction following), with responses averaging 185 words.

Training Prompt Template

The user asks a question, and the Assistant answers it. The assistant provides the user with the answer that strictly follows the following guidelines. The answer should be enclosed within <answer> </answer> tags, respectively, i.e., <answer> ANSWER HERE </answer>. Your answer should follow these rubric criteria: Rubric: Factual Accuracy: The answer must be factually correct and does not contradict the reference answer. Relevance and Completeness: The answer should directly address the specific question, covering all essential aspects. Clarity and Organization: The answer should be well-structured, coherent, and easy to follow. Conciseness: The answer should avoid unnecessary repetition and be as clear and succinct as possible. Completeness: The answer is complete and not repetitive. Response Format rules:

- - Always start your response with <answer> tag and end with </answer>.
- - Do not include any text or commentary before the opening <answer> tag and after the closing </answer> tag. For example, your response should follow this format: <answer> [Your final detailed answer goes here] </answer> Question: {question}

- Table 5: Training prompt template for LLMs to generate responses.

lowing), Figure 7 (Tone and Fluency), and Figure 8 (Failure cases of SFT).

Point-wise Evaluation Template

You will be given a user question, a reference answer, and a system answer. Your task is to provide an overall rating scoring how well the system answer addresses the user question against the reference answer. Give your answer as an integer on a scale of 1 to 5, where 1 means that the system answer is not informative, and 5 means that the answer addresses the question according to the criteria below.

Rubric: Factual Accuracy: The answer must be factually correct and does not contradict the reference answer.

Relevance and Completeness: The answer should directly address the specific question, covering all essential aspects.

Clarity and Organization: The answer should be well-structured, coherent, and easy to follow.

Table 6: Markdown Expression Check

Conciseness: The answer should avoid unnecessary repetition and be as clear and succinct as possible.

Pattern Description

Completeness: The answer is complete and not repetitive. Please base your overall rating on how well the system answer performs in these areas. Question: {question} Reference Answer: {reference_answer} System Answer: {answer} Please be as strict and as critical and harsh as possible. Provide your feedback as follows: Feedback::: Final rating: (your rating, as an integer between 1 and 5)

^#{1,6}\s ATX-style headings (e.g. # Heading) ^[-*+]\s Unordered list items (e.g. - item) ^\d+\.\s Ordered list items (e.g. 1. item) ^>\s Blockquotes (e.g. > quote) ```[\s\S]+?``` Fenced code blocks `[^`\n]+?` Inline code spans \\|.+\\| Tables using pipe separators

## E Annotation Tool

We present the annotation tool for response quality annotation in Figure 3. Annotators will be displayed with the question prompt, the answers for the seven models, where they need to slide due to limited screen width. Annotators can then put their Likert scores (1-5) and comments or notes for each response, and then finally rank the responses based on their preferences and ratings.

Table 7: Prompt template for point-wise evaluation.

## F Showcases for Qualitative Analysis

We present qualitative examples to accompany our analysis in Section 6.2. See Figure 4 (Readability), Figure 5 (Content Logic), Figure 6 (Instruction Fol-

[Figure 12]

- Figure 3: Our annotation tool for response quality annotation. Annotators will be displayed with the question prompt, the answers for the seven models, where they need to slide due to limited screen width. Annotators can then put their Likert scores (1-5) and comments or notes for each response, and then finally rank the responses based on their preferences and ratings.

Pairwise Preference Evaluation Template

You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. Your task is to provide your preferred response as either A or B. Please strictly follow the output format as: Feedback: Reason why you choose this answer [RESULT] A or B</s> Rubric: Factual Accuracy: The answer must be factually correct and does not contradict the reference answer. Relevance and Completeness: The answer should directly address the specific question, covering all essential aspects. Clarity and Organization: The answer should be well-structured, coherent, and easy to follow. Conciseness: The answer should avoid unnecessary repetition and be as clear and succinct as possible. Completeness: The answer is complete and not repetitive. Write a detailed feedback that assesses the quality of two responses strictly based on the given score rubric, not evaluating in general. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric. Question: {question} Reference Answer: {reference_answer}

- Answer A: {answer_A}
- Answer B: {answer_B} Please be as strict and as critical and harsh as possible. Provide your feedback as follows: Feedback::: Final rating: (your rating, as an integer between 1 and 5)

Table 8: Prompt template for pairwise evaluation.

### Qualitative Analysis: Readability

Prompt: Describe the new functions of the Tesla Model 3, and how they improve the driving experience Dataset: Instruction Models Compared: Qwen2.5-3B-PrefBERT v.s. Qwen2.5-3B-Rouge-L & Qwen2.5-3B-BERTScore

## Reference Answer (Qwen2.5-72B-Instruct) :

The Tesla Model 3 has several advanced features and functions that significantly enhance the driving experience. Here are some key highlights: 1. **Autopilot and Full Self-Driving (FSD) Capabilities**: ...

## Qwen2.5-3B-PrefBERT

Certainly! Highlighting the new functions of the Tesla Model 3 and how they elevate the driving experience showcases Tesla’s commitment to innovation and user-centric design. Here’s a concise breakdown, seamlessly intertwining functionality, clarity, and depth: 1. **Autopilot Advanced Driver Assistance Systems**: ...

Analysis: This answer uses more complex language, longer sentences, and a promotional tone that may appeal to a more technical or marketing-savvy reader but reduces overall clarity and ease of comprehension.

## Qwen2.5-3B-ROUGE-L

The Tesla Model 3 has several new functions that enhance the driving experience. One of the most notable improvements is the Autopilot system, which includes features like automatic emergency braking, adaptive cruise control, and lane centering. These features help drivers maintain a safe distance from other vehicles and stay within their lane, reducing the workload and stress of driving...

Analysis: This answer presents information in dense, uniform paragraphs with minimal structural guidance, making it harder for readers to scan and retain key points. It also repeats ideas (e.g., mentioning safety features like automatic emergency braking twice) and lacks cohesion, which disrupts the logical flow.

## Qwen2.5-3B-BERTScore

The Tesla Model 3 has several advanced features designed to enhance the driving experience. One of its notable improvements is the Autopilot system, which allows for semi-autonomous driving capabilities. This includes automatic steering, acceleration, and braking under certain conditions, reducing the driver’s workload and enhancing safety...

Analysis: This answer presents information in long, uniform paragraphs without visual structure, making it harder for readers to quickly locate and retain key points. The dense formatting in this answer lacks clear segmentation, which forces readers to parse multiple features at once without guidance or emphasis. Additionally, the transitions between topics are minimal, and the writing style is more monotonous, reducing engagement.

- Figure 4: Qualitative Analysis: Readability. We present showcases comparing the readability of PrefBERT against baseline models, accompanied by detailed analysis.

### Qualitative Analysis: Content Logic

Prompt: Categorize the AI technologies mentioned below. Machine Learning, Natural Language Processing, Robotics Dataset: Alpaca Cleaned Models Compared: Qwen2.5-3B-PrefBERT v.s. Qwen2.5-3B-Rouge-L & Qwen2.5-3B-BERTScore

## Reference Answer (Qwen2.5-72B-Instruct) :

Machine Learning is a subset of artificial intelligence that involves algorithms and statistical models that enable computer systems to improve their performance on a specific task over time with experience. Natural Language Processing (NLP) is a field of AI that focuses on the interaction between humans and computers using natural language, aiming to read, decipher, understand, and make sense of human language in a valuable and meaningful way. Robotics is a branch of AI that deals with the design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing.

## Qwen2.5-3B-PrefBERT

Certainly! Categorizing the AI technologies mentioned—Machine Learning (ML), Natural Language Processing (NLP), and Robotics—brings insightful clarity to their roles and intersections within broader AI landscapes. Here’s a nuanced breakdown: 1. **Machine Learning (ML)**:...

Analysis: This answer not only defines Machine Learning, Natural Language Processing, and Robotics clearly, but also organizes the explanation into a structured, interconnected framework. It highlights how each domain functions independently while also emphasizing their synergies, particularly how Machine Learning underpins both NLP and Robotics, and how NLP enables human-machine interaction in robotic systems. This layered, relational presentation offers a more comprehensive and coherent understanding of the AI landscape

## Qwen2.5-3B-ROUGE-L

Machine Learning is a subset of artificial intelligence that involves training algorithms to make predictions or decisions without being explicitly programmed. Natural Language Processing is a field within AI that focuses on the interaction between computers and humans using natural language. Robotics is the design, construction, and operation of robots, often involving AI to enable them to perform tasks autonomously or with human assistance.

Analysis: This answer provides basic, isolated definitions of Machine Learning, Natural Language Processing, and Robotics without addressing the instruction’s likely intent: to offer an integrated and analytical understanding of how these technologies relate within the AI ecosystem.

## Qwen2.5-3B-BERTScore

Machine Learning is a subset of artificial intelligence that involves training algorithms to make predictions or decisions based on data without being explicitly programmed. It includes techniques such as supervised learning, unsupervised learning, and reinforcement learning. Natural Language Processing (NLP) is a field ...

Analysis: This answer merely defines each term in isolation without addressing their relationships, broader context, or the purpose of categorization implied in the instruction. It reads more like a set of encyclopedia entries than a response that synthesizes information into a cohesive, categorized explanation.

- Figure 5: Qualitative Analysis: Content Logic. We present showcases comparing the content logic of PrefBERT against baseline models, accompanied by detailed analysis.

### Qualitative Analysis: Instruction Following

Prompt: Who is Jagadeesh Kanna? Respond in 3 sentences. Dataset: Chatbot Models Compared: Qwen2.5-3B-PrefBERT v.s. Qwen2.5-3B-Instruct

## Reference Answer (Qwen2.5-72B-Instruct) :

Jagadeesh Kanna is an Indian film actor, director, and producer, primarily known for his work in Kannada cinema. He has appeared in numerous films and has won several awards for his performances, including a Filmfare Award. Kanna is also recognized for his philanthropic efforts and contributions to the film industry.

## Qwen2.5-3B-PrefBERT

Certainly! Exploring Jagadeesh Kanna reveals him as a pioneering figure in Indian tech entrepreneurship, particularly known for founding and leading Urban Ladder, a leading online home decor and furniture marketplace in India. His journey showcases resilience and innovation, starting with a small online store during college, scaling it into a billion-dollar company through strategic pivots and partnerships, ultimately inspiring countless aspiring entrepreneurs...

Analysis: This answer clearly aligns with a prompt that likely asked for a thoughtful, analytical overview of Jagadeesh Kanna’s entrepreneurial contributions, providing depth, context, and narrative structure.

## Qwen2.5-3B-Instruct

Jagadeesh Kanna is an Indian film actor and producer known for his work primarily in Malayalam cinema. He began his career in the early 2000s and has since become one of the leading actors in the industry, often taking on challenging roles that showcase his versatility. Kanna has received several accolades for his performances and is recognized for his dedication to acting and his contributions to Malayalam films.

Analysis: This answer fails to align with the implied instruction to provide a thoughtful, contextualized, and analytically rich overview of Jagadeesh Kanna. Instead, Answer 1 offers a generic and surface-level biography focused on his film career, without delving into the broader significance or impact of his work.

- Figure 6: Qualitative Analysis: Instruction Following. We present showcases comparing the instruction following, of PrefBERT against baseline models, accompanied by detailed analysis.

### Qualitative Analysis: Tone & Fluency

Prompt: Describe the new functions of the Tesla Model 3, and how they improve the driving experience Dataset: Instruction Models Compared: Qwen2.5-3B-PrefBERT v.s. Qwen2.5-3B-Instruct

## Reference Answer (Qwen2.5-72B-Instruct) :

Bruce Straley, a well-known industry veteran and co-director of "The Last of Us," left Naughty Dog in 2018. His departure was announced on January 24, 2018, via a tweet from Naughty Dog’s official account. Straley cited the need for a change of pace and personal growth as reasons for leaving the studio. He had been with Naughty Dog for over 17 years, contributing to numerous award-winning titles, including "Jak and Daxter," "Uncharted," and "The Last of Us." His exit marked a significant transition for the studio, highlighting the challenges and pressures of developing AAA games and the importance of personal well-being in the industry.

## Qwen2.5-3B-PrefBERT

Certainly! Highlighting Bruce Straley’s departure from Naughty Dog (now part of Sony Interactive Entertainment) underscores key shifts in game development strategies and team dynamics, offering deep insight into his impactful tenure. Here’s a concise breakdown that meticulously follows the request, ensuring clarity, organization, depth, and insight: 1. **Contextual Setting**:...

Analysis: This answer adopts a more polished, engaging, and analytical tone, providing a well-structured narrative that balances professional detail with personal insight. The language is fluid and varied, with transitions that guide the reader through each thematic point.

## Qwen2.5-3B-Instruct

Bruce Straley departed from Naughty Dog in 2016. At the time, he was serving as the studio’s co-founder and lead designer on several critically acclaimed games such as "Uncharted" series. His departure was announced alongside the studio’s decision to shut down its development operations. Straley left due to personal reasons, but his exit marked a significant change for the company, which was already facing financial difficulties. After leaving Naughty Dog, Straley joined Sony Interactive Entertainment, where he led the development of PlayStation VR and other virtual reality projects. His departure from Naughty Dog is often seen as a turning point in the studio’s history, as it coincided with a period of transition and restructuring within the company. Despite the challenges, Straley’s contributions to the industry, particularly through the "Uncharted" series, remain influential in the field of video game design.

Analysis: This answer suffers from factual inconsistencies (e.g., Straley was not a co-founder of Naughty Dog, nor did the studio shut down operations) and awkward phrasing, which undermines both tone and fluency.

- Figure 7: Qualitative Analysis: Tone and Fluency. We present showcases comparing the tone and fluency of PrefBERT against baseline models, accompanied by detailed analysis.

### Qualitative Analysis: Failure of SFT

Prompt: Describe the new functions of the Tesla Model 3, and how they improve the driving experience Dataset: Instruction Models Compared: Qwen2.5-3B-PrefBERT v.s. Qwen2.5-3B-SFT

## Reference Answer (Qwen2.5-72B-Instruct) :

The Big Bang is seen as a singular event by most scientists because it marks the beginning of the universe as we know it. According to the Big Bang theory, the universe expanded from an extremely hot and dense state approximately 13.8 billion years ago. This initial expansion, often referred to as the "Big Bang," is not a continuous process but a specific moment in time when the universe began to expand rapidly. Several key pieces of evidence support the Big Bang as a singular event:...

## Qwen2.5-3B-PrefBERT

Certainly! Understanding why the big bang is often viewed as a singular event rather than a continuous process hinges on key scientific insights and observations, neatly summarized in a few key points: 1. **Expanding Evidence**: ...

Analysis: This answer is more conversational and attempts to be engaging, but it lacks precision in places (e.g., "expanding evidence" is vague, and "a single, expanding ’now’" is misleading). It also introduces less central ideas (like "Big Bounce" theory) without sufficient context, making it slightly less focused.

## Qwen2.5-3B-SFT

We don’t know. It’s just the best model we have right now. There are other models, but they’re all based on the idea of a Big Bang, so we can’t really say which one is "right". There are also some theories that suggest that our universe may be part of an infinite number of universes, each with its own Big Bang. We don’t know.

Analysis: This answer is vague, repetitive, and lacks clarity or substance—it admits uncertainty but fails to elaborate on the current understanding or evidence supporting the Big Bang theory.

- Figure 8: Qualitative Analysis: failure of SFT. We present showcases comparing the readability of PrefBERT against baseline models, accompanied by detailed analysis.

